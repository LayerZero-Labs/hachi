use super::single_cyclic::mat_vec_mul_single_i8_cyclic_with_params;
use super::*;

type CompressionProducts<F, const D: usize> = Vec<Vec<CyclotomicRing<F, D>>>;

pub(crate) fn validate_compression_batch_shape<const D: usize>(
    digit_vectors: &[&[[i8; D]]],
) -> Result<usize, AkitaError> {
    let column_count = digit_vectors.first().map_or(0, |digits| digits.len());
    if digit_vectors.is_empty() || column_count == 0 {
        return Err(AkitaError::InvalidInput(
            "compression batch must contain nonempty digit vectors".to_string(),
        ));
    }
    if digit_vectors
        .iter()
        .any(|digits| digits.len() != column_count)
    {
        return Err(AkitaError::InvalidInput(
            "compression batch digit vectors must have one exact shape".to_string(),
        ));
    }
    Ok(column_count)
}

/// Compute paired negacyclic/cyclic compression products after caller-owned
/// batch-shape validation.
///
/// Compression uses negative-binary digits, so the balanced range is fixed at
/// log basis one. Validate that range once, dispatch the prepared profile once,
/// and use stack row views for both transforms.
pub(crate) fn mat_vec_mul_ntt_compression_i8<F, const D: usize>(
    slot: &PreparedNttCache<D>,
    input_width: usize,
    digit_vectors: &[&[[i8; D]]],
) -> Result<(CompressionProducts<F, D>, CompressionProducts<F, D>), AkitaError>
where
    F: Field + CanonicalEncoding,
{
    let actual_width = validate_compression_batch_shape(digit_vectors)?;
    if actual_width != input_width {
        return Err(AkitaError::InvalidInput(format!(
            "compression digit width {actual_width} does not match requested width {input_width}"
        )));
    }
    for digits in digit_vectors {
        validate_digit_rows_for_log_basis(
            digits,
            input_width,
            1,
            "for negative-binary compression",
        )?;
    }

    macro_rules! run {
        ($base:expr) => {{
            let base = $base;
            let negacyclic = base.negacyclic().ok_or_else(|| {
                AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
            })?;
            let cyclic = base
                .cyclic()
                .ok_or_else(|| AkitaError::InvalidSetup("cyclic NTT domain not prepared".into()))?;
            let negacyclic_row = negacyclic.get(..input_width).ok_or_else(|| {
                AkitaError::InvalidSetup("compression negacyclic NTT prefix is too short".into())
            })?;
            let cyclic_row = cyclic.get(..input_width).ok_or_else(|| {
                AkitaError::InvalidSetup("compression cyclic NTT prefix is too short".into())
            })?;
            let negacyclic_rows = [negacyclic_row];
            let cyclic_rows = [cyclic_row];
            let params = base.params();
            let negacyclic_products =
                mat_vec_mul_digits_i8_with_params(&negacyclic_rows, digit_vectors, 1, params);
            let cyclic_products = digit_vectors
                .iter()
                .map(|digits| {
                    mat_vec_mul_single_i8_cyclic_with_params(&cyclic_rows, digits, 1, params)
                })
                .collect();
            Ok((negacyclic_products, cyclic_products))
        }};
    }

    if let Some(base) = slot.q32_base() {
        run!(base)
    } else if let Some(base) = slot.q64_base() {
        run!(base)
    } else if let Some(base) = slot.q128_base() {
        run!(base)
    } else {
        Err(AkitaError::InvalidSetup(
            "compression requires a paired base-profile NTT cache".into(),
        ))
    }
}
