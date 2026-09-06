//! Phase probe for uncached dense decompose-fold; inputs and caches are built outside timing.
//!
//! Run with: `dense_decompose_probe <fp32|fp64|fp128|fp64b16|fp128b16>
//! <num-vars> <threads> <live|cached|extract|scan> <samples>`.
//! The `cached` mode supports the i8 fp32/fp64 cases. Extraction reuses small
//! scratch rather than writing a persistent digit cache; its time is not an
//! additive decomposition of the full fold time. Each mode discards one warmup.
//! Compare separate baseline/candidate executables in interleaved order.
use akita_algebra::{ring::cyclotomic::decompose_centering_threshold, CyclotomicRing};
use akita_challenges::{SparseChallenge, SparseChallengeConfig};
use akita_prover::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, cached_digit_decompose_fold_partitioned,
    decompose_ring_interleaved, decompose_ring_interleaved_i16, DecomposeParams,
};
use akita_types::sis::compute_num_digits_field_width;
use jolt_field::{CanonicalEncoding, Field, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};
fn run<F: Field + CanonicalEncoding, const D: usize>(
    bits: u32,
    basis: u32,
    nv: usize,
    threads: usize,
    samples: usize,
    mode: &str,
) {
    let n = 1usize << nv;
    let rings: Vec<_> = (0..n / D)
        .map(|r| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|c| {
                let x = (r * D + c) as u128;
                F::from_u128_reduced(
                    x.wrapping_mul(0x9e3779b97f4a7c156a09e667f3bcc909)
                        .rotate_left(37)
                        ^ x.wrapping_mul(0xbf58476d1ce4e5b994d049bb133111eb),
                )
            }))
        })
        .collect();
    let k = compute_num_digits_field_width(bits, basis);
    let q = (-F::one()).to_u128_checked().unwrap() + 1;
    let threshold = decompose_centering_threshold(k, basis, q);
    let p = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << basis) - 1,
        half_b: 1i128 << (basis - 1),
        b_val: 1i128 << basis,
        log_basis: basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };
    let positions = 512;
    let config = SparseChallengeConfig::production_for_ring_dim(D).unwrap();
    let challenges: Vec<_> = (0..rings.len().div_ceil(positions))
        .map(|b| SparseChallenge {
            positions: (0..config.weight())
                .map(|t| ((t * 37 + b * 13) % D) as u32)
                .collect(),
            coeffs: (0..config.weight())
                .map(|t| {
                    let m = if t < config.count_pm1 { 1 } else { 2 };
                    if (t + b).is_multiple_of(2) {
                        m
                    } else {
                        -m
                    }
                })
                .collect(),
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let cached = if mode == "cached" {
        assert!(basis <= 8);
        let mut v = vec![[0i8; D]; rings.len() * k];
        pool.install(|| {
            v.par_chunks_mut(k)
                .zip(rings.par_iter())
                .for_each(|(dst, r)| decompose_ring_interleaved(r, dst, k, &p))
        });
        Some(v)
    } else {
        None
    };
    let mut times = Vec::new();
    for sample in 0..samples + 1 {
        let start = Instant::now();
        pool.install(|| match mode {
            "live" => {
                black_box(balanced_ring_decompose_fold_partitioned(
                    &rings,
                    &challenges,
                    positions,
                    k,
                    &p,
                ));
            }
            "cached" => {
                black_box(cached_digit_decompose_fold_partitioned::<F, D>(
                    cached.as_ref().unwrap(),
                    &challenges,
                    positions,
                    k,
                    basis,
                ));
            }
            "extract" => {
                rings.par_chunks(64).for_each(|rs| {
                    if basis <= 8 {
                        let mut dst = vec![[0i8; D]; k];
                        for r in rs {
                            decompose_ring_interleaved(r, &mut dst, k, &p);
                            black_box(&dst);
                        }
                    } else {
                        let mut dst = vec![[0i16; D]; k];
                        for r in rs {
                            decompose_ring_interleaved_i16(r, &mut dst, k, &p);
                            black_box(&dst);
                        }
                    }
                });
            }
            "scan" => {
                black_box(
                    rings
                        .par_iter()
                        .map(|r| {
                            r.coeffs
                                .iter()
                                .fold(0u128, |a, c| a.wrapping_add(c.to_u128_checked().unwrap()))
                        })
                        .reduce(|| 0, u128::wrapping_add),
                );
            }
            _ => panic!("unknown mode"),
        });
        if sample > 0 {
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    times.sort_by(f64::total_cmp);
    println!("bits={bits},d={D},basis={basis},digits={k},nv={nv},threads={threads},mode={mode},median_ms={:.3},min_ms={:.3},max_ms={:.3},weight={}",times[samples/2],times[0],times[samples-1],config.weight());
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert_eq!(
        a.len(),
        6,
        "expected field, num-vars, threads, mode, samples"
    );
    let nv: usize = a[2].parse().unwrap();
    let threads = a[3].parse().unwrap();
    let samples: usize = a[5].parse().unwrap();
    assert!((11..usize::BITS as usize).contains(&nv));
    assert!(threads > 0 && samples > 0);
    match a[1].as_str() {
        "fp32" => run::<Prime32Offset99, 2048>(32, 8, nv, threads, samples, &a[4]),
        "fp64" => run::<Prime64Offset59, 1024>(64, 6, nv, threads, samples, &a[4]),
        "fp128" => run::<Prime128OffsetA7F7, 512>(128, 11, nv, threads, samples, &a[4]),
        "fp64b16" => run::<Prime64Offset59, 2048>(64, 16, nv, threads, samples, &a[4]),
        "fp128b16" => run::<Prime128OffsetA7F7, 1024>(128, 16, nv, threads, samples, &a[4]),
        _ => panic!("unknown field"),
    }
}
