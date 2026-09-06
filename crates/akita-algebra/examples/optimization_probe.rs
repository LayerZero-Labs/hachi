//! Small wall-clock probe for equality factors and balanced decomposition.
//!
//! Inputs and output buffers are built outside timing. Run separate binaries
//! from the base and candidate objects in ABBA/BAAB order.

use akita_algebra::{CyclotomicRing, EqPolynomial};
use jolt_field::solinas::{Ext2, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use jolt_field::{CanonicalEncoding, Field};
use std::hint::black_box;
use std::time::Instant;

fn sample_values<F: Field + CanonicalEncoding, const D: usize>() -> [F; D] {
    let q = (-F::one())
        .to_u128_checked()
        .expect("probe fields fit in u128")
        + 1;
    let mut state = 0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03u128;
    std::array::from_fn(|index| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        F::from_u128_reduced(state.wrapping_add(index as u128) % q)
    })
}

fn stream_values<F: Field + CanonicalEncoding, const D: usize>(ring_index: usize) -> [F; D] {
    let q = (-F::one())
        .to_u128_checked()
        .expect("probe fields fit in u128")
        + 1;
    std::array::from_fn(|coefficient| {
        let index = ring_index.wrapping_mul(D).wrapping_add(coefficient) as u128;
        let value = index
            .wrapping_mul(0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03)
            .wrapping_add(0x94d0_49bb_1331_11ebu128);
        F::from_u128_reduced(value % q)
    })
}

fn challenge_values<F: Field + CanonicalEncoding>(len: usize, mut state: u128) -> Vec<F> {
    let q = (-F::one())
        .to_u128_checked()
        .expect("probe fields fit in u128")
        + 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            F::from_u128_reduced(state % q)
        })
        .collect()
}

fn report(mode: &str, samples: usize, mut run: impl FnMut()) {
    for _ in 0..2 {
        run();
    }
    for sample in 0..samples {
        let start = Instant::now();
        run();
        println!("{mode},{sample},{}", start.elapsed().as_nanos());
    }
}

fn decompose_i8<F: Field + CanonicalEncoding, const D: usize>(
    mode: &str,
    levels: usize,
    log_basis: u32,
    samples: usize,
    iterations: usize,
) {
    let ring = CyclotomicRing::<F, D>::from_coefficients(sample_values());
    let mut out = vec![[0i8; D]; levels];
    report(mode, samples, || {
        for _ in 0..iterations {
            ring.balanced_decompose_pow2_i8_into(black_box(&mut out), log_basis);
        }
        black_box(&out);
    });
}

fn decompose_i16<F: Field + CanonicalEncoding, const D: usize>(
    mode: &str,
    levels: usize,
    log_basis: u32,
    samples: usize,
    iterations: usize,
) {
    let ring = CyclotomicRing::<F, D>::from_coefficients(sample_values());
    let mut out = vec![[0i16; D]; levels];
    report(mode, samples, || {
        for _ in 0..iterations {
            ring.balanced_decompose_pow2_i16_into(black_box(&mut out), log_basis);
        }
        black_box(&out);
    });
}

fn stream_decompose_i8<F: Field + CanonicalEncoding, const D: usize>(
    mode: &str,
    levels: usize,
    log_basis: u32,
    samples: usize,
    iterations: usize,
    total_coefficients: usize,
) {
    assert_eq!(total_coefficients % D, 0);
    let rings: Vec<_> = (0..total_coefficients / D)
        .map(|index| CyclotomicRing::<F, D>::from_coefficients(stream_values(index)))
        .collect();
    let mut out = vec![[0i8; D]; rings.len() * levels];
    report(mode, samples, || {
        for _ in 0..iterations {
            for (ring, digits) in rings.iter().zip(out.chunks_exact_mut(levels)) {
                black_box(ring).balanced_decompose_pow2_i8_into(digits, log_basis);
            }
        }
        black_box(&out);
    });
}

fn stream_decompose_i16<F: Field + CanonicalEncoding, const D: usize>(
    mode: &str,
    levels: usize,
    log_basis: u32,
    samples: usize,
    iterations: usize,
    total_coefficients: usize,
) {
    assert_eq!(total_coefficients % D, 0);
    let rings: Vec<_> = (0..total_coefficients / D)
        .map(|index| CyclotomicRing::<F, D>::from_coefficients(stream_values(index)))
        .collect();
    let mut out = vec![[0i16; D]; rings.len() * levels];
    report(mode, samples, || {
        for _ in 0..iterations {
            for (ring, digits) in rings.iter().zip(out.chunks_exact_mut(levels)) {
                black_box(ring).balanced_decompose_pow2_i16_into(digits, log_basis);
            }
        }
        black_box(&out);
    });
}

fn eq_base(samples: usize, iterations: usize) {
    let x = challenge_values::<Prime128OffsetA7F7>(30, 0xd1b5_4a32_d192_ed03);
    let y = challenge_values::<Prime128OffsetA7F7>(30, 0x94d0_49bb_1331_11eb);
    report("eq-base-n30", samples, || {
        for _ in 0..iterations {
            black_box(EqPolynomial::mle(black_box(&x), black_box(&y)).unwrap());
        }
    });
}

fn eq_ext(samples: usize, iterations: usize) {
    type E = Ext2<Prime64Offset59>;
    let x0 = challenge_values::<Prime64Offset59>(30, 0xd1b5_4a32_d192_ed03);
    let x1 = challenge_values::<Prime64Offset59>(30, 0x94d0_49bb_1331_11eb);
    let y0 = challenge_values::<Prime64Offset59>(30, 0x8538_eb5b_d456_ea3d);
    let y1 = challenge_values::<Prime64Offset59>(30, 0xda94_2042_e4dd_58b5);
    let x: Vec<E> = x0
        .into_iter()
        .zip(x1)
        .map(|(c0, c1)| E::new(c0, c1))
        .collect();
    let y: Vec<E> = y0
        .into_iter()
        .zip(y1)
        .map(|(c0, c1)| E::new(c0, c1))
        .collect();
    report("eq-ext2-n30", samples, || {
        for _ in 0..iterations {
            black_box(EqPolynomial::mle(black_box(&x), black_box(&y)).unwrap());
        }
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("fp128-b16");
    let samples = args.get(2).map_or(10, |s| s.parse().expect("samples"));
    let iterations = args.get(3).map_or(128, |s| s.parse().expect("iterations"));
    let total_coefficients = args
        .get(4)
        .map_or(1 << 22, |s| s.parse().expect("total coefficients"));
    match mode {
        "fp32-b7" => decompose_i8::<Prime32Offset99, 2048>(mode, 5, 7, samples, iterations),
        "fp64-b6" => decompose_i8::<Prime64Offset59, 1024>(mode, 11, 6, samples, iterations),
        "fp64-b16" => decompose_i16::<Prime64Offset59, 2048>(mode, 4, 16, samples, iterations),
        "fp128-b11" => decompose_i16::<Prime128OffsetA7F7, 512>(mode, 12, 11, samples, iterations),
        "fp128-b16" => decompose_i16::<Prime128OffsetA7F7, 1024>(mode, 8, 16, samples, iterations),
        "stream-fp32-b7" => stream_decompose_i8::<Prime32Offset99, 2048>(
            mode,
            5,
            7,
            samples,
            iterations,
            total_coefficients,
        ),
        "stream-fp64-b6" => stream_decompose_i8::<Prime64Offset59, 1024>(
            mode,
            11,
            6,
            samples,
            iterations,
            total_coefficients,
        ),
        "stream-fp64-b16" => stream_decompose_i16::<Prime64Offset59, 2048>(
            mode,
            4,
            16,
            samples,
            iterations,
            total_coefficients,
        ),
        "stream-fp128-b11" => stream_decompose_i16::<Prime128OffsetA7F7, 512>(
            mode,
            12,
            11,
            samples,
            iterations,
            total_coefficients,
        ),
        "stream-fp128-b16" => stream_decompose_i16::<Prime128OffsetA7F7, 1024>(
            mode,
            8,
            16,
            samples,
            iterations,
            total_coefficients,
        ),
        "eq-base" => eq_base(samples, iterations),
        "eq-ext" => eq_ext(samples, iterations),
        _ => panic!("unknown mode {mode}"),
    }
}
