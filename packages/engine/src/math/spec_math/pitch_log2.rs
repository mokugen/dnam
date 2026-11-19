//! Pitch-ratio and time-warp polynomial profile (`ref[dsp.ctrl.math.pitch-profile]`)
//! and 129-entry monotone `f32` `log2` lookup table (`ref[dsp.ctrl.math.log2-lut]`).

use super::ieee_bits::{round_f64_to_f32_bits, widen_f32_bits};
use super::saturation::InvalidFloatInput;

/// Scale factor $\ln(2)/12$ in IEEE-754 `f64` (`0x3FAD9303FEA2F7EA`, `ref[dsp.ctrl.math.pitch-profile]`).
pub const PITCH_LN2_OVER_12_F64_BITS: u64 = 0x3FAD_9303_FEA2_F7EA;

/// Evaluates the degree-14 `f64` Horner pitch-ratio $R = 2^{p/12}$ and time-warp $\nu = 1 - R$
/// for a finite `f32` semitone shift $p \in [-24.0, +24.0]$ (`ref[dsp.ctrl.math.pitch-profile]`).
/// Returns `None` if `semitones` is non-finite or `|semitones| > 24.0`.
#[inline]
pub fn eval_pitch_profile_f32(semitones: f32) -> Option<(f32, f32)> {
    let p = widen_f32_bits(semitones.to_bits())?;
    if !p.is_finite() || p.abs() > 24.0 {
        return None;
    }
    if p == 0.0 {
        return Some((1.0, 0.0));
    }
    let scale = f64::from_bits(PITCH_LN2_OVER_12_F64_BITS);
    let x = p * scale;
    const INV_FACT: [f64; 14] = [
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5_040.0,
        1.0 / 40_320.0,
        1.0 / 362_880.0,
        1.0 / 3_628_800.0,
        1.0 / 39_916_800.0,
        1.0 / 479_001_600.0,
        1.0 / 6_227_020_800.0,
        1.0 / 87_178_291_200.0,
    ];
    let mut poly = INV_FACT[13];
    let mut idx = 13;
    while idx > 0 {
        idx -= 1;
        poly = INV_FACT[idx] + x * poly;
    }
    let delta = x * poly;
    let ratio_bits = round_f64_to_f32_bits(1.0 + delta)?;
    let warp_bits = round_f64_to_f32_bits(-delta)?;
    Some((f32::from_bits(ratio_bits), f32::from_bits(warp_bits)))
}

/// 129-entry read-only `f32` bit table $T_j = \mathrm{RNE}_{f32}(\log_2(1 + j/128))$ (`516 B`,
/// SHA-256 `4d0d7c0ec194dd9d761bfe2982ca4847001f333c58a3f12f423ed2053de3a5f1`, `ref[dsp.ctrl.math.log2-lut]`).
pub const LOG2_MANTISSA_LUT_129_BITS: [u32; 129] = [
    0x00000000, 0x3c37f286, 0x3cb73cb4, 0x3d08e68f, 0x3d35d69c, 0x3d626fd6, 0x3d8759c5, 0x3d9d517f,
    0x3db31fb8, 0x3dc8c50b, 0x3dde4212, 0x3df39761, 0x3e0462c4, 0x3e0ee68d, 0x3e19574f, 0x3e23b550,
    0x3e2e00d2, 0x3e383a17, 0x3e42615f, 0x3e4c76e8, 0x3e567af1, 0x3e606db6, 0x3e6a4f72, 0x3e74205f,
    0x3e7de0b6, 0x3e83c857, 0x3e88983f, 0x3e8d602e, 0x3e92203d, 0x3e96d888, 0x3e9b8926, 0x3ea03232,
    0x3ea4d3c2, 0x3ea96df0, 0x3eae00d2, 0x3eb28c7f, 0x3eb7110e, 0x3ebb8e96, 0x3ec0052b, 0x3ec474e4,
    0x3ec8ddd4, 0x3ecd4012, 0x3ed19bb0, 0x3ed5f0c4, 0x3eda3f60, 0x3ede8797, 0x3ee2c97d, 0x3ee70525,
    0x3eeb3a9f, 0x3eef69ff, 0x3ef39355, 0x3ef7b6b4, 0x3efbd42b, 0x3effebcd, 0x3f01fed4, 0x3f0404e8,
    0x3f060828, 0x3f08089e, 0x3f0a0650, 0x3f0c0146, 0x3f0df989, 0x3f0fef1f, 0x3f11e20f, 0x3f13d260,
    0x3f15c01a, 0x3f17ab44, 0x3f1993e3, 0x3f1b7a00, 0x3f1d5da0, 0x3f1f3eca, 0x3f211d84, 0x3f22f9d5,
    0x3f24d3c2, 0x3f26ab53, 0x3f28808c, 0x3f2a5374, 0x3f2c2411, 0x3f2df268, 0x3f2fbe80, 0x3f31885c,
    0x3f335004, 0x3f35157d, 0x3f36d8cb, 0x3f3899f5, 0x3f3a58ff, 0x3f3c15ee, 0x3f3dd0c8, 0x3f3f8991,
    0x3f41404f, 0x3f42f506, 0x3f44a7ba, 0x3f465872, 0x3f480731, 0x3f49b3fb, 0x3f4b5ed7, 0x3f4d07c7,
    0x3f4eaed0, 0x3f5053f7, 0x3f51f740, 0x3f5398af, 0x3f553848, 0x3f56d60f, 0x3f587209, 0x3f5a0c3a,
    0x3f5ba4a4, 0x3f5d3b4e, 0x3f5ed039, 0x3f60636a, 0x3f61f4e5, 0x3f6384ad, 0x3f6512c7, 0x3f669f35,
    0x3f6829fb, 0x3f69b31e, 0x3f6b3a9f, 0x3f6cc083, 0x3f6e44cd, 0x3f6fc781, 0x3f7148a1, 0x3f72c832,
    0x3f744636, 0x3f75c2b0, 0x3f773da4, 0x3f78b714, 0x3f7a2f04, 0x3f7ba578, 0x3f7d1a71, 0x3f7e8df2,
    0x3f800000,
];

/// Status of decomposing an `f32` bit pattern for log-envelope lookup (`ref[dsp.ctrl.math.log2-lut]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Log2EnvelopeDecomp {
    /// Positive finite envelope $x = 2^E(1 + m/2^{23})$ with $E \in [-149, 127]$ and $m \in [0, 2^{23})$.
    Positive { exponent: i32, fraction: u32 },
    /// Exact zero (`+0.0` or `-0.0`): silent status.
    ZeroSilent,
    /// Negative or non-finite (`NaN`/`Inf`): rejected.
    Invalid,
}

/// Decomposes an `f32` bit pattern into `(exponent, 23-bit normalized fraction)` using leading-zero
/// normalization on subnormals (`ref[dsp.ctrl.math.log2-lut]`).
#[inline]
pub const fn decompose_positive_f32_bits(bits: u32) -> Log2EnvelopeDecomp {
    let magnitude = bits & 0x7fff_ffff;
    if magnitude == 0 {
        return Log2EnvelopeDecomp::ZeroSilent;
    }
    let exponent_bits = (bits >> 23) & 0xff;
    if (bits >> 31) != 0 || exponent_bits == 0xff {
        return Log2EnvelopeDecomp::Invalid;
    }
    let fraction = bits & 0x007f_ffff;
    if exponent_bits != 0 {
        Log2EnvelopeDecomp::Positive {
            exponent: (exponent_bits as i32) - 127,
            fraction,
        }
    } else {
        let top_bit = 31 - fraction.leading_zeros();
        let norm_frac = (fraction << (23 - top_bit)) & 0x007f_ffff;
        Log2EnvelopeDecomp::Positive {
            exponent: (top_bit as i32) - 149,
            fraction: norm_frac,
        }
    }
}

/// Evaluates the monotone `f32` fractional mantissa interpolation $p \in [0, 1]$ for `fraction < 2^23`
/// (`ref[dsp.ctrl.math.log2-lut]`).
#[inline]
pub fn lookup_log2_mantissa_f32(fraction: u32) -> f32 {
    let index = ((fraction >> 16) & 0x7f) as usize;
    let alpha = ((fraction & 0xffff) as f32) * f32::from_bits(0x3780_0000);
    let low = f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[index]);
    let high = f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[index + 1]);
    let d = high - low;
    high.min(low + alpha * d)
}

/// Evaluates $\widehat{\log_2(x)}$ for a positive finite `f32` bit pattern `bits` (`ref[dsp.ctrl.math.log2-lut]`).
#[inline]
pub fn eval_monotone_log2_f32_bits(bits: u32) -> Result<Option<f32>, InvalidFloatInput> {
    match decompose_positive_f32_bits(bits) {
        Log2EnvelopeDecomp::ZeroSilent => Ok(None),
        Log2EnvelopeDecomp::Invalid => Err(InvalidFloatInput),
        Log2EnvelopeDecomp::Positive { exponent, fraction } => {
            let p = lookup_log2_mantissa_f32(fraction);
            Ok(Some((exponent as f32) + p))
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::FtzDazGuard;
    use crate::provenance::sha256_digest;

    /// verify[dsp.ctrl.math.pitch-profile]
    #[test]
    fn test_pitch_profile_accuracy_endpoints_and_monotonicity() {
        let _guard = FtzDazGuard::new();
        for &(st, exp_r, exp_w) in &[
            (-24.0_f32, 0.25_f32, 0.75_f32),
            (-12.0, 0.5, 0.5),
            (-0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (12.0, 2.0, -1.0),
            (24.0, 4.0, -3.0),
        ] {
            let (r, w) = eval_pitch_profile_f32(st).unwrap();
            assert_eq!((r, w), (exp_r, exp_w));
            if st == 0.0 {
                assert_eq!(w.to_bits(), 0);
            }
        }
        let (r_pos, w_pos) = eval_pitch_profile_f32(1.0e-8).unwrap();
        let (r_neg, w_neg) = eval_pitch_profile_f32(-1.0e-8).unwrap();
        assert_eq!((r_pos, r_neg), (1.0, 1.0));
        assert!(w_pos < 0.0 && w_neg > 0.0);

        for bad in [
            -24.0001_f32,
            24.0001,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert!(eval_pitch_profile_f32(bad).is_none());
        }

        let steps = 50_000usize;
        let (mut prev_r, mut prev_w) = (0.0_f32, f32::INFINITY);
        for i in 0..=steps {
            let p = -24.0_f32 + 48.0_f32 * (i as f32) / (steps as f32);
            let (r, w) = eval_pitch_profile_f32(p).unwrap();
            let ideal_r = ((p as f64) / 12.0).exp2();
            assert!(((r as f64) / ideal_r - 1.0).abs() < 6.2e-8);
            assert!(((w as f64) - (1.0 - ideal_r)).abs() < 1.2e-7);
            assert!(r >= prev_r && w <= prev_w);
            prev_r = r;
            prev_w = w;
        }
        let (_, w0) = eval_pitch_profile_f32(-f32::from_bits(0x4180_0010)).unwrap();
        let (_, w1) = eval_pitch_profile_f32(-f32::from_bits(0x4180_0011)).unwrap();
        assert!(w1 >= w0);
    }

    /// verify[dsp.ctrl.math.log2-lut]
    #[test]
    fn test_monotone_log2_lut_sha256_subnormals_and_accuracy() {
        let _guard = FtzDazGuard::new();
        let mut table_bytes = [0u8; 129 * 4];
        for (i, &bits) in LOG2_MANTISSA_LUT_129_BITS.iter().enumerate() {
            table_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bits.to_le_bytes());
            if i > 0 {
                assert!(f32::from_bits(bits) > f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[i - 1]));
            }
        }
        let expected_sha256: [u8; 32] = [
            0x4d, 0x0d, 0x7c, 0x0e, 0xc1, 0x94, 0xdd, 0x9d, 0x76, 0x1b, 0xfe, 0x29, 0x82, 0xca,
            0x48, 0x47, 0x00, 0x1f, 0x33, 0x3c, 0x58, 0xa3, 0xf1, 0x2f, 0x42, 0x3e, 0xd2, 0x05,
            0x3d, 0xe3, 0xa5, 0xf1,
        ];
        assert_eq!(sha256_digest(&table_bytes), expected_sha256);

        assert_eq!(eval_monotone_log2_f32_bits(0x0000_0000), Ok(None));
        assert_eq!(eval_monotone_log2_f32_bits(0x8000_0000), Ok(None));
        for bad in [0x8000_0001_u32, 0xbf80_0000, 0x7f80_0000, 0x7fc0_0000] {
            assert!(eval_monotone_log2_f32_bits(bad).is_err());
        }
        assert_eq!(
            decompose_positive_f32_bits(0x0000_0001),
            Log2EnvelopeDecomp::Positive {
                exponent: -149,
                fraction: 0
            }
        );
        assert_eq!(
            decompose_positive_f32_bits(0x0080_0000),
            Log2EnvelopeDecomp::Positive {
                exponent: -126,
                fraction: 0
            }
        );

        let mut prev = -f64::INFINITY;
        for step in 0..=(1u32 << 15) {
            let frac = (step << 8).min((1 << 23) - 1);
            let p = lookup_log2_mantissa_f32(frac) as f64;
            let exact = (1.0 + (frac as f64) * (-23.0_f64).exp2()).log2();
            assert!((p - exact).abs() < 1.9e-5 && p >= prev);
            prev = p;
        }
        let mut prev_sub = -f64::INFINITY;
        for top_bit in 0..23u32 {
            let low = 1u32 << top_bit;
            let high = (1u32 << (top_bit + 1)) - 1;
            for bits in [low, (low + high) / 2, high] {
                let val = eval_monotone_log2_f32_bits(bits).unwrap().unwrap() as f64;
                let exact = (bits as f64).log2() - 149.0;
                assert!((val - exact).abs() < 1.9e-5 && val >= prev_sub);
                prev_sub = val;
            }
        }
    }

    proptest! {
        #[test]
        fn prop_pitch_and_log2_accuracy(
            semitones in -24.0_f32..=24.0_f32,
            pos_x in 1.0e-30_f32..1.0e30_f32,
        ) {
            let _guard = FtzDazGuard::new();
            let (ratio, warp) = eval_pitch_profile_f32(semitones).unwrap();
            let exact_ratio = ((semitones as f64) / 12.0).exp2();
            prop_assert!(((ratio as f64 - exact_ratio) / exact_ratio).abs() < 6.2e-8);
            prop_assert!((warp as f64 - (1.0 - exact_ratio)).abs() < 1.2e-7);

            let approx_log2 = eval_monotone_log2_f32_bits(pos_x.to_bits()).unwrap().unwrap();
            let exact_log2 = (pos_x as f64).log2();
            prop_assert!((approx_log2 as f64 - exact_log2).abs() < 1.9e-5);
        }
    }
}
