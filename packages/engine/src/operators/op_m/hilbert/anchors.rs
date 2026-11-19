//! Canonical 15-Rate `H1` Hilbert Anchor Table & Exact `u128` Ties-to-Even Profile Interpolation
//! (`ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).

/// Canonical 15-rate `H1` Hilbert anchor table (`rate_hz`, 6 `I`-branch + 6 `Q`-branch `f32` bit
/// patterns; `780 B` little-endian binary SHA-256 `bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`).
pub const HILBERT_H1_ANCHORS_15: [(u32, [u32; 12]); 15] = [
    (
        44_100,
        [
            0x3D8EC459, 0x3EE935E1, 0x3F47659D, 0x3F6CCEAA, 0x3F7A0188, 0x3F7E7297, 0x3E7B1AD6,
            0x3F243369, 0x3F5E97F2, 0x3F752D7D, 0x3F7CCA31, 0x3F7F8A32,
        ],
    ),
    (
        48_000,
        [
            0x3D91BEE6, 0x3EECB8E1, 0x3F48DE0D, 0x3F6D91C9, 0x3F7A5939, 0x3F7E8E9D, 0x3E80025F,
            0x3F25F6B0, 0x3F5FADA1, 0x3F75B332, 0x3F7CFFA1, 0x3F7F92D2,
        ],
    ),
    (
        64_000,
        [
            0x3DA2AF27, 0x3EFE4625, 0x3F4FBDA5, 0x3F70B815, 0x3F7B8289, 0x3F7EE2C7, 0x3E8C7CF3,
            0x3F2E9DD3, 0x3F647F75, 0x3F77A752, 0x3F7DA785, 0x3F7FAC3E,
        ],
    ),
    (
        80_000,
        [
            0x3DA88500, 0x3F0212AF, 0x3F523110, 0x3F72069B, 0x3F7C1032, 0x3F7F0F58, 0x3E90CC38,
            0x3F318514, 0x3F665AC6, 0x3F788657, 0x3F7DFC4C, 0x3F7FBA14,
        ],
    ),
    (
        96_000,
        [
            0x3DB073C5, 0x3F05A9C4, 0x3F54E508, 0x3F7343B2, 0x3F7C852B, 0x3F7F30C7, 0x3E963AF7,
            0x3F34F3ED, 0x3F683FD9, 0x3F794AFF, 0x3F7E3E99, 0x3F7FC438,
        ],
    ),
    (
        176_400,
        [
            0x3E0D917D, 0x3F216E28, 0x3F624034, 0x3F77D413, 0x3F7DE528, 0x3F7F8887, 0x3ECDF9DC,
            0x3F49A494, 0x3F703BD8, 0x3F7BD39E, 0x3F7EF730, 0x3F7FDDFB,
        ],
    ),
    (
        192_000,
        [
            0x3E1A79AD, 0x3F2691E2, 0x3F64674D, 0x3F787EBC, 0x3F7E12E1, 0x3F7F9298, 0x3ED99328,
            0x3F4D2EB7, 0x3F717613, 0x3F7C2D21, 0x3F7F0DA9, 0x3F7FE0D4,
        ],
    ),
    (
        208_000,
        [
            0x3E2646E1, 0x3F2A6E87, 0x3F65C7A3, 0x3F78E266, 0x3F7E2EFD, 0x3F7F99AA, 0x3EE33E4A,
            0x3F4F9D5A, 0x3F723264, 0x3F7C621C, 0x3F7F1C5F, 0x3F7FE2ED,
        ],
    ),
    (
        224_000,
        [
            0x3E34BA6C, 0x3F2F8A17, 0x3F67DE51, 0x3F798480, 0x3F7E584A, 0x3F7FA231, 0x3EEF3412,
            0x3F53121B, 0x3F736170, 0x3F7CB521, 0x3F7F3010, 0x3F7FE547,
        ],
    ),
    (
        240_000,
        [
            0x3E4151D9, 0x3F33437E, 0x3F693973, 0x3F79E8DF, 0x3F7E72F8, 0x3F7FA845, 0x3EF8BC96,
            0x3F556DD1, 0x3F741F05, 0x3F7CE93F, 0x3F7F3D5F, 0x3F7FE706,
        ],
    ),
    (
        256_000,
        [
            0x3E4CBD88, 0x3F363B81, 0x3F6A3C44, 0x3F7A3322, 0x3F7E877C, 0x3F7FAD32, 0x3F006617,
            0x3F573C54, 0x3F74AA8C, 0x3F7D1079, 0x3F7F47DE, 0x3F7FE876,
        ],
    ),
    (
        288_000,
        [
            0x3E64086E, 0x3F3BDB10, 0x3F6C20CB, 0x3F7ABA23, 0x3F7EABAF, 0x3F7FB5AD, 0x3F083609,
            0x3F5AA15E, 0x3F75ACD3, 0x3F7D5697, 0x3F7F5A2A, 0x3F7FEAEB,
        ],
    ),
    (
        320_000,
        [
            0x3E7D6662, 0x3F41DCD7, 0x3F6E3EDD, 0x3F7B5343, 0x3F7ED216, 0x3F7FBDBA, 0x3F1068CD,
            0x3F5E589B, 0x3F76D2C7, 0x3F7DA41A, 0x3F7F6C8B, 0x3F7FED2A,
        ],
    ),
    (
        352_000,
        [
            0x3E88E622, 0x3F45A7EB, 0x3F6F6903, 0x3F7BA4C3, 0x3F7EE847, 0x3F7FC310, 0x3F163553,
            0x3F6082FF, 0x3F776EE6, 0x3F7DCEB5, 0x3F7F77E5, 0x3F7FEEB9,
        ],
    ),
    (
        384_000,
        [
            0x3E942913, 0x3F49F3A3, 0x3F70D45F, 0x3F7C094D, 0x3F7F0173, 0x3F7FC869, 0x3F1C84D4,
            0x3F631031, 0x3F783122, 0x3F7E017B, 0x3F7F8401, 0x3F7FF039,
        ],
    ),
];

/// Encodes `HILBERT_H1_ANCHORS_15` into its canonical `780 B` little-endian binary table
/// (`15 × 13 × 4 B`, `ref[dsp.op-m.hilbert.h1-anchors]`).
pub fn encode_hilbert_h1_anchor_table_780b() -> [u8; 780] {
    let mut out = [0u8; 780];
    let mut pos = 0;
    for &(rate, ref coeffs) in &HILBERT_H1_ANCHORS_15 {
        out[pos..pos + 4].copy_from_slice(&rate.to_le_bytes());
        pos += 4;
        for &c in coeffs {
            out[pos..pos + 4].copy_from_slice(&c.to_le_bytes());
            pos += 4;
        }
    }
    out
}

fn f32_anchor_scaled_27(bits: u32) -> u128 {
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let sig = (1u128 << 23) | ((bits & 0x007f_ffff) as u128);
    let shift = 4 + exp;
    sig << (shift as u32)
}

fn round_positive_u128_ratio_to_f32_bits(numerator: u128, denominator: u128) -> Option<u32> {
    if numerator == 0 || numerator >= denominator {
        return None;
    }
    let num_bits = (128 - numerator.leading_zeros()) as i32;
    let den_bits = (128 - denominator.leading_zeros()) as i32;
    let mut exponent = num_bits - den_bits;
    if exponent >= 0 {
        if numerator < (denominator << (exponent as u32)) {
            exponent -= 1;
        }
    } else if (numerator << ((-exponent) as u32)) < denominator {
        exponent -= 1;
    }
    if exponent < -126 {
        return None;
    }
    let shifted_num = numerator << ((23 - exponent) as u32);
    let mut significand = shifted_num / denominator;
    let remainder = shifted_num % denominator;
    if 2 * remainder > denominator || (2 * remainder == denominator && (significand & 1) != 0) {
        significand += 1;
    }
    if significand == (1 << 24) {
        significand >>= 1;
        exponent += 1;
    }
    Some((((exponent + 127) as u32) << 23) | ((significand as u32) - (1 << 23)))
}

/// Interpolates the 12 `H1` Hilbert coefficients (`[alpha_I[0..6], alpha_Q[0..6]]` as `f32` bit
/// patterns) between `low_rate` and `high_rate` at `rate_hz` (`low_rate <= rate_hz <= high_rate`)
/// using exact `u128` ties-to-even arithmetic (`ref[dsp.op-m.hilbert.ties-to-even]`).
pub fn interpolate_hilbert_h1_between_anchors_bits(
    rate_hz: f64,
    low_rate: u32,
    high_rate: u32,
) -> Option<[u32; 12]> {
    if !rate_hz.is_finite()
        || low_rate >= high_rate
        || rate_hz < (low_rate as f64)
        || rate_hz > (high_rate as f64)
    {
        return None;
    }
    let low_coeffs = HILBERT_H1_ANCHORS_15
        .iter()
        .find(|&&(r, _)| r == low_rate)?
        .1;
    let high_coeffs = HILBERT_H1_ANCHORS_15
        .iter()
        .find(|&&(r, _)| r == high_rate)?
        .1;

    // Decompose binary64 rate_hz = R / 2^k (0 <= k <= 37)
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (1u128 << 52) | ((bits & ((1u64 << 52) - 1)) as u128);
    let (r_int, k_shift): (u128, u32) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 0)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, raw_k - tz)
    };
    let left_weight = ((high_rate as u128) << k_shift) - r_int;
    let right_weight = r_int - ((low_rate as u128) << k_shift);
    let denominator = ((high_rate - low_rate) as u128) << (k_shift + 27);

    let mut out = [0u32; 12];
    for i in 0..12 {
        let a_scaled = f32_anchor_scaled_27(low_coeffs[i]);
        let b_scaled = f32_anchor_scaled_27(high_coeffs[i]);
        let numerator = left_weight * a_scaled + right_weight * b_scaled;
        out[i] = round_positive_u128_ratio_to_f32_bits(numerator, denominator)?;
    }
    Some(out)
}

/// Interpolates the 12 `H1` Hilbert coefficients for any supported `rate_hz` in either the `1×`
/// live band (`44,100..=96,000 Hz`) or the `4×` bounce band (`176,400..=384,000 Hz`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).
pub fn interpolate_hilbert_h1_profile_bits(rate_hz: f64) -> Option<[u32; 12]> {
    if !rate_hz.is_finite() {
        return None;
    }
    const LIVE_BAND: [u32; 5] = [44_100, 48_000, 64_000, 80_000, 96_000];
    const BOUNCE_BAND: [u32; 10] = [
        176_400, 192_000, 208_000, 224_000, 240_000, 256_000, 288_000, 320_000, 352_000, 384_000,
    ];
    for band in [&LIVE_BAND[..], &BOUNCE_BAND[..]] {
        for pair in band.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            if rate_hz >= (lo as f64) && rate_hz <= (hi as f64) {
                return interpolate_hilbert_h1_between_anchors_bits(rate_hz, lo, hi);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::sha256_digest;
    use alloc::{format, string::String};
    use proptest::prelude::*;

    /// verify[dsp.op-m.hilbert.h1-anchors]
    /// verify[dsp.op-m.hilbert.ties-to-even]
    #[test]
    fn test_hilbert_h1_anchors_sha256_and_ties_to_even() {
        let table_780b = encode_hilbert_h1_anchor_table_780b();
        let sha_hex: String = sha256_digest(&table_780b)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            sha_hex,
            "bd236421110a57438536f20286ada3d2d16de8709f418f7322136d80c639c352"
        );

        for &(rate, ref expected_coeffs) in &HILBERT_H1_ANCHORS_15 {
            assert_eq!(
                interpolate_hilbert_h1_profile_bits(rate as f64).unwrap(),
                *expected_coeffs
            );
        }
        for &(rate, lo, hi, idx, exp_bits) in &[
            (49_200.0_f64, 48_000u32, 64_000u32, 8usize, 0x3F60_0A2Au32),
            (54_800.0, 48_000, 64_000, 8, 0x3F61_BA02),
            (48_400.0, 48_000, 64_000, 10, 0x3F7D_03D4),
            (80_960.0, 80_000, 96_000, 3, 0x3F72_19A2),
            (81_600.0, 80_000, 96_000, 4, 0x3F7C_1BE4),
            (181_080.0, 176_400, 192_000, 4, 0x3F7D_F2E0),
        ] {
            let coeffs = interpolate_hilbert_h1_between_anchors_bits(rate, lo, hi).unwrap();
            assert_eq!(coeffs[idx], exp_bits);
        }

        for bad_rate in [44_099.0, 100_000.0, 176_399.0, 384_001.0, f64::NAN] {
            assert!(interpolate_hilbert_h1_profile_bits(bad_rate).is_none());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_hilbert_h1_interpolation_bounds(rate_live in 44_100.0_f64..=96_000.0_f64) {
            let coeffs = interpolate_hilbert_h1_profile_bits(rate_live).unwrap();
            for &b in &coeffs {
                let c = f32::from_bits(b);
                prop_assert!(c > 0.0 && c < 1.0);
            }
        }
    }
}
