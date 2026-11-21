//! Pitch-ratio and time-warp polynomial profile (`ref[dsp.ctrl.math.pitch-profile]`)
//! and 129-entry monotone `f32` `log2` lookup table (`ref[dsp.ctrl.math.log2-lut]`).

use super::ieee_bits::{round_f64_to_f32_bits, widen_f32_bits};
use super::saturation::InvalidFloatInput;
use crate::math::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK, IEEE754_F32_MIN_SUBNORMAL_EXP,
    IEEE754_F32_SIGN_MASK, const_abs_f64, const_f32_pow2, const_inv_factorial_f64,
};

/// Scale factor $\ln(2)/12$ in IEEE-754 `f64` (`0x3FAD9303FEA2F7EA`).
pub const PITCH_LN2_OVER_12_F64_BITS: u64 = 0x3FAD_9303_FEA2_F7EA;

/// Evaluates the degree-14 `f64` Horner pitch-ratio $R = 2^{p/12}$ and time-warp $\nu = 1 - R$
/// for a finite `f32` semitone shift $p \in [-24.0, +24.0]$.
#[inline]
pub fn eval_pitch_profile_f32(semitones: f32) -> Option<(f32, f32)> {
    let p = widen_f32_bits(semitones.to_bits())?;
    if !p.is_finite() || const_abs_f64(p) > 24.0 {
        return None;
    }
    if p == 0.0 {
        return Some((1.0, 0.0));
    }
    let scale = f64::from_bits(PITCH_LN2_OVER_12_F64_BITS);
    let x = p * scale;
    const INV_FACT: [f64; 14] = [
        const_inv_factorial_f64(1),
        const_inv_factorial_f64(2),
        const_inv_factorial_f64(3),
        const_inv_factorial_f64(4),
        const_inv_factorial_f64(5),
        const_inv_factorial_f64(6),
        const_inv_factorial_f64(7),
        const_inv_factorial_f64(8),
        const_inv_factorial_f64(9),
        const_inv_factorial_f64(10),
        const_inv_factorial_f64(11),
        const_inv_factorial_f64(12),
        const_inv_factorial_f64(13),
        const_inv_factorial_f64(14),
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

/// 129-entry read-only `f32` bit table $T_j = \mathrm{RNE}_{f32}(\log_2(1 + j/128))$.
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

/// Status of decomposing an `f32` bit pattern for log-envelope lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Log2EnvelopeDecomp {
    /// Positive finite envelope.
    Positive { exponent: i32, fraction: u32 },
    /// Exact zero (`+0.0` or `-0.0`).
    ZeroSilent,
    /// Negative or non-finite (`NaN`/`Inf`).
    Invalid,
}

/// Decomposes an `f32` bit pattern into `(exponent, 23-bit normalized fraction)`.
#[inline]
pub const fn decompose_positive_f32_bits(bits: u32) -> Log2EnvelopeDecomp {
    let magnitude = bits & IEEE754_F32_ABS_MASK;
    if magnitude == 0 {
        return Log2EnvelopeDecomp::ZeroSilent;
    }
    let exponent_bits = (bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES;
    if (bits & IEEE754_F32_SIGN_MASK) != 0 || exponent_bits == IEEE754_F32_EXP_ALL_ONES {
        return Log2EnvelopeDecomp::Invalid;
    }
    let fraction = bits & IEEE754_F32_MANTISSA_MASK;
    if exponent_bits != 0 {
        Log2EnvelopeDecomp::Positive {
            exponent: (exponent_bits as i32) - (IEEE754_F32_EXP_BIAS as i32),
            fraction,
        }
    } else {
        let top_bit = 31 - fraction.leading_zeros();
        let norm_frac =
            (fraction << (IEEE754_F32_MANTISSA_BITS - top_bit)) & IEEE754_F32_MANTISSA_MASK;
        Log2EnvelopeDecomp::Positive {
            exponent: (top_bit as i32) + IEEE754_F32_MIN_SUBNORMAL_EXP,
            fraction: norm_frac,
        }
    }
}

/// Evaluates the monotone `f32` fractional mantissa interpolation $p \in [0, 1]$.
#[inline]
pub fn lookup_log2_mantissa_f32(fraction: u32) -> f32 {
    let index = ((fraction >> 16) & 0x7f) as usize;
    let alpha = ((fraction & 0xffff) as f32) * const_f32_pow2(-16);
    let low = f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[index]);
    let high = f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[index + 1]);
    let d = high - low;
    high.min(low + alpha * d)
}

/// Evaluates $\widehat{\log_2(x)}$ for a positive finite `f32` bit pattern `bits`.
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

    #[test]
    fn test_pitch_profile_and_monotone_log2_lut() {
        let _guard = FtzDazGuard::new();
        for &(st, exp_r, exp_w) in &[
            (-24.0_f32, 0.25_f32, 0.75_f32),
            (-12.0, 0.5, 0.5),
            (-0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (12.0, 2.0, -1.0),
            (24.0, 4.0, -3.0),
        ] {
            assert_eq!(eval_pitch_profile_f32(st), Some((exp_r, exp_w)));
        }
        for bad in [-24.0001_f32, 24.0001, f32::INFINITY, f32::NAN] {
            assert!(eval_pitch_profile_f32(bad).is_none());
        }
        assert_eq!(eval_monotone_log2_f32_bits(0), Ok(None));
        assert_eq!(eval_monotone_log2_f32_bits(0x8000_0000), Ok(None));
        assert!(eval_monotone_log2_f32_bits(0xbf80_0000).is_err());
        for i in 1..LOG2_MANTISSA_LUT_129_BITS.len() {
            assert!(
                f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[i])
                    > f32::from_bits(LOG2_MANTISSA_LUT_129_BITS[i - 1])
            );
        }
        let sub_val = eval_monotone_log2_f32_bits(1).unwrap().unwrap();
        assert!((sub_val - (-149.0_f32)).abs() < 1e-4);
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
