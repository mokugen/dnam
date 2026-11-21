//! Bit-preserving `f32 <-> f64` conversions immune to active FTZ/DAZ (`ref[dsp.ctrl.op-n.trust-clip]`)
//! and 6-step division-free `f64` Newton reciprocal (`ref[dsp.op-n.polyadaa.rail-crossing]`).

use crate::math::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK, IEEE754_F32_MIN_NORMAL_EXP,
    IEEE754_F32_MIN_SUBNORMAL_EXP, IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET, IEEE754_F32_SIGN_MASK,
    IEEE754_F32_SIGNIFICAND_BITS, IEEE754_F64_EXP_ALL_ONES, IEEE754_F64_EXP_BIAS,
    IEEE754_F64_IMPLICIT_BIT, IEEE754_F64_MANTISSA_BITS, IEEE754_F64_MANTISSA_MASK,
    IEEE754_F64_MAX_NORMAL_BIASED_EXP, IEEE754_F64_SIGN_MASK,
};

const F32_TO_F64_EXP_BIAS_OFFSET: u32 = (IEEE754_F64_EXP_BIAS as u32) - IEEE754_F32_EXP_BIAS;
const F32_SUBNORMAL_F64_EXP_BASE: u32 =
    (IEEE754_F64_EXP_BIAS + IEEE754_F32_MIN_SUBNORMAL_EXP) as u32;
const F64_MINUS_F32_MANTISSA_BITS: u32 = IEEE754_F64_MANTISSA_BITS - IEEE754_F32_MANTISSA_BITS;
const F64_TO_F32_SUBNORMAL_SHIFT_OFFSET: i32 =
    (IEEE754_F64_MANTISSA_BITS as i32) + IEEE754_F32_MIN_SUBNORMAL_EXP;
const F64_SIGNIFICAND_BITS: u32 = IEEE754_F64_MANTISSA_BITS + 1;

const _: () = assert!(F32_TO_F64_EXP_BIAS_OFFSET == 896);
const _: () = assert!(F32_SUBNORMAL_F64_EXP_BASE == 874);
const _: () = assert!(F64_MINUS_F32_MANTISSA_BITS == 29);
const _: () = assert!(F64_TO_F32_SUBNORMAL_SHIFT_OFFSET == -97);
const _: () = assert!(F64_SIGNIFICAND_BITS == 53);

/// Widens an IEEE-754 `f32` bit pattern (including subnormals) directly to `f64`
/// without executing hardware `f32` instructions subject to FTZ/DAZ.
#[inline(always)]
pub const fn widen_f32_bits(bits: u32) -> Option<f64> {
    let exponent = (bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES;
    if exponent != 0 && exponent != IEEE754_F32_EXP_ALL_ONES {
        let sign = ((bits & IEEE754_F32_SIGN_MASK) as u64) << 32;
        let frac = (bits & IEEE754_F32_MANTISSA_MASK) as u64;
        return Some(f64::from_bits(
            sign | (((exponent + F32_TO_F64_EXP_BIAS_OFFSET) as u64) << IEEE754_F64_MANTISSA_BITS)
                | (frac << F64_MINUS_F32_MANTISSA_BITS),
        ));
    }
    let sign = ((bits & IEEE754_F32_SIGN_MASK) as u64) << 32;
    let fraction = bits & IEEE754_F32_MANTISSA_MASK;
    let wide_bits = match (exponent, fraction) {
        (e, _) if e == IEEE754_F32_EXP_ALL_ONES => return None,
        (0, 0) => sign,
        (_, frac) => {
            let leading = 31 - frac.leading_zeros();
            let exponent64 = (leading + F32_SUBNORMAL_F64_EXP_BASE) as u64;
            let fraction64 =
                ((frac - (1 << leading)) as u64) << (IEEE754_F64_MANTISSA_BITS - leading);
            sign | (exponent64 << IEEE754_F64_MANTISSA_BITS) | fraction64
        }
    };
    Some(f64::from_bits(wide_bits))
}

#[inline]
const fn round_right_even_u64(mantissa: u64, shift: u32) -> u64 {
    if shift > F64_SIGNIFICAND_BITS {
        return 0;
    }
    let truncated = mantissa >> shift;
    let remainder = mantissa & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    let bump = if remainder > halfway || (remainder == halfway && (truncated & 1) != 0) {
        1
    } else {
        0
    };
    truncated + bump
}

/// Rounds a finite `f64` value to an IEEE-754 `f32` bit pattern using integer significand shifts
/// (`RNE`) without hardware `f32` instructions.
#[inline]
pub const fn round_f64_to_f32_bits(value: f64) -> Option<u32> {
    let bits = value.to_bits();
    let sign = ((bits >> 32) as u32) & IEEE754_F32_SIGN_MASK;
    let exponent_bits =
        ((bits >> IEEE754_F64_MANTISSA_BITS) & (IEEE754_F64_EXP_ALL_ONES as u64)) as i32;
    let fraction = bits & IEEE754_F64_MANTISSA_MASK;
    if exponent_bits == 0 {
        return if fraction == 0 { Some(sign) } else { None };
    }
    if exponent_bits == (IEEE754_F64_EXP_ALL_ONES as i32) {
        return None;
    }
    let f32_exp_bias = IEEE754_F32_EXP_BIAS as i32;
    let mut exponent = exponent_bits - IEEE754_F64_EXP_BIAS;
    let mantissa = IEEE754_F64_IMPLICIT_BIT | fraction;
    if exponent < IEEE754_F32_MIN_NORMAL_EXP {
        let shift = (F64_TO_F32_SUBNORMAL_SHIFT_OFFSET - exponent) as u32;
        return Some(sign | (round_right_even_u64(mantissa, shift) as u32));
    }
    if exponent > f32_exp_bias {
        return None;
    }
    let mut rounded = round_right_even_u64(mantissa, F64_MINUS_F32_MANTISSA_BITS);
    if rounded == (1 << IEEE754_F32_SIGNIFICAND_BITS) {
        exponent += 1;
        rounded >>= 1;
    }
    if exponent > f32_exp_bias {
        return None;
    }
    Some(
        sign | (((exponent + f32_exp_bias) as u32) << IEEE754_F32_MANTISSA_BITS)
            | ((rounded as u32) & IEEE754_F32_MANTISSA_MASK),
    )
}

/// Returns the unbiased base-2 exponent of a non-zero finite `f32` bit pattern (`-149..=127`).
#[inline]
pub const fn f32_unbiased_exponent(bits: u32) -> Option<i32> {
    let magnitude = bits & IEEE754_F32_ABS_MASK;
    let exponent = magnitude >> IEEE754_F32_MANTISSA_BITS;
    if magnitude == 0 || exponent == IEEE754_F32_EXP_ALL_ONES {
        return None;
    }
    if exponent != 0 {
        Some((exponent as i32) - (IEEE754_F32_EXP_BIAS as i32))
    } else {
        let bit_len = 32 - (magnitude & IEEE754_F32_MANTISSA_MASK).leading_zeros();
        Some((bit_len as i32) - IEEE754_F32_NORMAL_MANTISSA_EXP_OFFSET)
    }
}

/// Evaluates the 6-step division-free Newton reciprocal $1 / \Delta$ for nonzero $\Delta$ in `f64`.
#[inline]
pub fn newton6_reciprocal_f64(distance: f64) -> f64 {
    let bits = distance.to_bits();
    let sign_bit = bits & IEEE754_F64_SIGN_MASK;
    let exponent = (((bits >> IEEE754_F64_MANTISSA_BITS) & (IEEE754_F64_EXP_ALL_ONES as u64))
        as i32)
        - IEEE754_F64_EXP_BIAS;
    let inv_exp =
        (IEEE754_F64_EXP_BIAS - exponent - 1).clamp(1, IEEE754_F64_MAX_NORMAL_BIASED_EXP) as u64;
    let mut inverse = f64::from_bits(sign_bit | (inv_exp << IEEE754_F64_MANTISSA_BITS));
    for _ in 0..6 {
        inverse *= 2.0 - distance * inverse;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::math::const_f64_pow2;

    fn next_splitmix(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64) * (-53.0_f64).exp2()
    }

    #[test]
    fn bitwise_widening_rounding_and_newton6_reciprocal() {
        let _guard = FtzDazGuard::new();
        let quantum = const_f64_pow2(-149);
        for fraction in (0..=0x007f_ffff_u32).step_by(255) {
            let expected = (f64::from(fraction) * quantum).to_bits();
            assert_eq!(widen_f32_bits(fraction).unwrap().to_bits(), expected);
            assert_eq!(
                widen_f32_bits(fraction | 0x8000_0000).unwrap().to_bits(),
                expected | (1_u64 << 63)
            );
        }
        assert!(widen_f32_bits(0x7f80_0000).is_none());
        assert!(widen_f32_bits(0x7fc0_0001).is_none());

        let half_quantum = const_f64_pow2(-150);
        for lower in (0..0x0080_0000_u32).step_by(255) {
            let midpoint = f64::from(2 * lower + 1) * half_quantum;
            let expected = lower + (lower & 1);
            assert_eq!(round_f64_to_f32_bits(midpoint), Some(expected));
            assert_eq!(
                round_f64_to_f32_bits(-midpoint),
                Some(expected | 0x8000_0000)
            );
        }
        assert!(round_f64_to_f32_bits(f64::NAN).is_none());
        assert!(round_f64_to_f32_bits(1.0e50).is_none());
        assert!(round_f64_to_f32_bits(f64::from_bits(1)).is_none());
        assert_eq!(round_f64_to_f32_bits(-0.0), Some(0x8000_0000));
        assert!(f32_unbiased_exponent(0).is_none());
        assert!(f32_unbiased_exponent(0x7f80_0000).is_none());
        assert_eq!(f32_unbiased_exponent(0x0000_0001), Some(-149));
        assert_eq!(f32_unbiased_exponent(1.0_f32.to_bits()), Some(0));

        let tol = const_f64_pow2(-44);
        let mut seed = 0x1234_5678_u64;
        for exp in -140..=128 {
            let base = (exp as f64).exp2();
            let d = base * (1.0 + next_splitmix(&mut seed));
            let inv = newton6_reciprocal_f64(d);
            assert!((d * inv - 1.0).abs() < tol);
        }
    }
}
