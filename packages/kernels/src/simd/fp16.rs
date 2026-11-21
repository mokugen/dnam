//! Pure-`u32` IEEE-754 `binary32` (`f32`) $\leftrightarrow$ `binary16` (`fp16`) software conversion
//! primitives (`ref[sdk.rust.shared-nostd-kernels]`).
//!
//! Uses strictly `u32`/`i32` bit arithmetic with round-to-nearest-even (`RNE`), zero division,
//! zero `u64`/`f64`, and finite clamping of `[65520.0, f32::MAX]` to `0x7BFF`/`0xFBFF`.

use super::const_eval::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS, IEEE754_F32_EXP_MASK,
    IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK,
    IEEE754_F32_TO_FP16_MANTISSA_SHIFT, IEEE754_FP16_ABS_MASK, IEEE754_FP16_EXP_ALL_ONES,
    IEEE754_FP16_EXP_BIAS, IEEE754_FP16_EXP_MASK, IEEE754_FP16_MANTISSA_BITS,
    IEEE754_FP16_MANTISSA_MASK, IEEE754_FP16_MAX_FINITE_BITS, IEEE754_FP16_SIGN_MASK,
};

const FP16_RNE_ROUND_BIT_SHIFT: u32 = IEEE754_F32_TO_FP16_MANTISSA_SHIFT - 1;
const FP16_RNE_STICKY_MASK: u32 = (1_u32 << FP16_RNE_ROUND_BIT_SHIFT) - 1;

const _: () = assert!(FP16_RNE_ROUND_BIT_SHIFT == 12);
const _: () = assert!(FP16_RNE_STICKY_MASK == 0x0FFF);

/// Converts an `f32` to IEEE-754 `fp16` packed in the lower 16 bits of a `u32` (zero upper bits).
///
/// Uses exact round-to-nearest-even across normal and subnormal `fp16` values, preserves `±0.0`,
/// `±Inf`, and `NaN`, and clamps finite values with magnitude `>= 65520.0` to `±65504.0`
/// (`0x7BFF` / `0xFBFF`).
#[inline(always)]
#[must_use]
pub const fn f32_to_fp16_u32(val: f32) -> u32 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & IEEE754_FP16_SIGN_MASK;
    let abs_bits = bits & IEEE754_F32_ABS_MASK;
    if abs_bits == 0 {
        return sign;
    }
    let exp = ((bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES) as i32;
    let mant = bits & IEEE754_F32_MANTISSA_MASK;

    if exp == (IEEE754_F32_EXP_ALL_ONES as i32) {
        if mant == 0 {
            return sign | IEEE754_FP16_EXP_MASK;
        }
        let payload = mant >> IEEE754_F32_TO_FP16_MANTISSA_SHIFT;
        let non_zero_payload = if payload != 0 { payload } else { 1 };
        return sign | IEEE754_FP16_EXP_MASK | non_zero_payload;
    }

    let new_exp = exp - (IEEE754_F32_EXP_BIAS as i32) + (IEEE754_FP16_EXP_BIAS as i32);
    if new_exp >= (IEEE754_FP16_EXP_ALL_ONES as i32) {
        return sign | IEEE754_FP16_MAX_FINITE_BITS;
    }
    if new_exp <= 0 {
        if new_exp < -(IEEE754_FP16_MANTISSA_BITS as i32) {
            return sign;
        }
        let m = mant | IEEE754_F32_IMPLICIT_BIT;
        let shift = (((IEEE754_FP16_EXP_BIAS as i32) - 1) - new_exp) as u32;
        let half_mant = m >> shift;
        let round_bit = (m >> (shift - 1)) & 1;
        let sticky = (m & ((1_u32 << (shift - 1)) - 1)) != 0;
        let rounded = half_mant + ((round_bit != 0 && (sticky || (half_mant & 1) != 0)) as u32);
        return sign | rounded;
    }

    let round_bit = (mant >> FP16_RNE_ROUND_BIT_SHIFT) & 1;
    let sticky = (mant & FP16_RNE_STICKY_MASK) != 0;
    let raw_mant = mant >> IEEE754_F32_TO_FP16_MANTISSA_SHIFT;
    let half_mant = raw_mant + ((round_bit != 0 && (sticky || (raw_mant & 1) != 0)) as u32);
    let combined = ((new_exp as u32) << IEEE754_FP16_MANTISSA_BITS) + half_mant;
    let clamped = if combined > IEEE754_FP16_MAX_FINITE_BITS {
        IEEE754_FP16_MAX_FINITE_BITS
    } else {
        combined
    };
    sign | clamped
}

/// Converts an IEEE-754 `fp16` bit pattern in the lower 16 bits of `bits` to `f32`.
#[inline(always)]
#[must_use]
pub const fn fp16_u32_to_f32(bits: u32) -> f32 {
    let b = bits & 0xFFFF;
    let sign = (b & IEEE754_FP16_SIGN_MASK) << 16;
    if (b & IEEE754_FP16_ABS_MASK) == 0 {
        return f32::from_bits(sign);
    }
    let exp = (b >> IEEE754_FP16_MANTISSA_BITS) & IEEE754_FP16_EXP_ALL_ONES;
    let mant = b & IEEE754_FP16_MANTISSA_MASK;

    if exp == 0 {
        // Subnormal fp16 (`mant in 1..=0x03FF`): normalize using CLZ.
        let shift = mant.leading_zeros() - (32 - 1 - IEEE754_FP16_MANTISSA_BITS);
        let m = (mant << shift) & IEEE754_FP16_MANTISSA_MASK;
        let e = IEEE754_F32_EXP_BIAS - (IEEE754_FP16_EXP_BIAS - 1) - shift;
        return f32::from_bits(
            sign | (e << IEEE754_F32_MANTISSA_BITS) | (m << IEEE754_F32_TO_FP16_MANTISSA_SHIFT),
        );
    }
    if exp == IEEE754_FP16_EXP_ALL_ONES {
        return f32::from_bits(
            sign | IEEE754_F32_EXP_MASK | (mant << IEEE754_F32_TO_FP16_MANTISSA_SHIFT),
        );
    }
    let f32_exp = (exp + IEEE754_F32_EXP_BIAS - IEEE754_FP16_EXP_BIAS) << IEEE754_F32_MANTISSA_BITS;
    f32::from_bits(sign | f32_exp | (mant << IEEE754_F32_TO_FP16_MANTISSA_SHIFT))
}

/// Packs two `f32` values into a `u32` as `[lo_fp16, hi_fp16]` (`lo` in bits `0..16`, `hi` in `16..32`).
#[inline(always)]
#[must_use]
pub const fn pack_fp16x2_u32(lo: f32, hi: f32) -> u32 {
    f32_to_fp16_u32(lo) | (f32_to_fp16_u32(hi) << 16)
}

/// Unpacks a `pack_fp16x2_u32` `u32` word into `[lo, hi]` `f32` values.
#[inline(always)]
#[must_use]
pub const fn unpack_fp16x2_u32(packed: u32) -> [f32; 2] {
    [
        fp16_u32_to_f32(packed & 0xFFFF),
        fp16_u32_to_f32(packed >> 16),
    ]
}

/// Converts an `f32` value to IEEE-754 `fp16` (`u16`).
#[inline(always)]
#[must_use]
pub const fn f32_to_fp16(val: f32) -> u16 {
    f32_to_fp16_u32(val) as u16
}

/// Converts an IEEE-754 `fp16` (`u16`) value to `f32`.
#[inline(always)]
#[must_use]
pub const fn fp16_to_f32(bits: u16) -> f32 {
    fp16_u32_to_f32(bits as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn fp16_to_f64_oracle(h: u16) -> f64 {
        let sign = if (h & 0x8000) != 0 { -1.0_f64 } else { 1.0_f64 };
        let exp = (h >> 10) & 0x1F;
        let mant = h & 0x03FF;
        if exp == 0 {
            sign * (f64::from(mant) * 2.0_f64.powi(-24))
        } else if exp == 31 {
            if mant == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        } else {
            sign * (1.0 + f64::from(mant) / 1024.0) * 2.0_f64.powi(i32::from(exp) - 15)
        }
    }

    #[test]
    fn test_fp16_exhaustive_roundtrip_and_edge_cases() {
        for raw in 0_u32..=0xFFFF {
            let h = raw as u16;
            let f = fp16_to_f32(h);
            let oracle = fp16_to_f64_oracle(h);
            if oracle.is_nan() {
                assert!(f.is_nan());
                assert!(fp16_to_f32(f32_to_fp16(f)).is_nan());
            } else if oracle.is_infinite() {
                assert!(f.is_infinite());
                assert_eq!(f.is_sign_negative(), oracle.is_sign_negative());
                assert_eq!(f32_to_fp16(f), h);
            } else {
                assert_eq!(f64::from(f), oracle);
                assert_eq!(f.to_bits() >> 31, (u32::from(h) >> 15) & 1);
                assert_eq!(f32_to_fp16(f), h);
            }
        }

        // Signed zeros and tiny underflow
        assert_eq!(f32_to_fp16_u32(0.0), 0x0000);
        assert_eq!(f32_to_fp16_u32(-0.0), 0x8000);
        assert_eq!(fp16_u32_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(f32_to_fp16(1.0e-12), 0x0000);
        assert_eq!(f32_to_fp16(-1.0e-12), 0x8000);

        // Subnormal RNE boundary checks: halfway 2^-25 ties to even (0), above rounds to 0x0001
        let half_min_sub = crate::simd::const_f32_pow2(-25);
        assert_eq!(f32_to_fp16(half_min_sub), 0x0000);
        assert_eq!(
            f32_to_fp16(f32::from_bits(half_min_sub.to_bits() + 1)),
            0x0001
        );
        // Subnormal-to-normal carry at 2^-14 - 2^-25 -> 0x0400
        let carry_boundary = crate::simd::const_f32_pow2(-14) - half_min_sub;
        assert_eq!(f32_to_fp16(carry_boundary), 0x0400);

        // Overflow clamping near max finite fp16 ([65520.0, f32::MAX])
        for &near_max in &[
            65504.0_f32,
            65519.0,
            65520.0,
            65535.0,
            65536.0,
            1.0e20,
            f32::MAX,
        ] {
            assert_eq!(f32_to_fp16(near_max), 0x7BFF);
            assert_eq!(f32_to_fp16(-near_max), 0xFBFF);
        }

        // Signaling NaN with only low mantissa bits set must remain NaN, never Inf
        let low_nan = f32::from_bits(0x7F80_0001);
        let low_nan_h = f32_to_fp16(low_nan);
        assert_eq!(low_nan_h, 0x7C01);
        assert!(fp16_to_f32(low_nan_h).is_nan());

        // Packed fp16x2 roundtrip
        let packed = pack_fp16x2_u32(-1.5, 65520.0);
        let unpacked = unpack_fp16x2_u32(packed);
        assert_eq!(unpacked, [-1.5, 65504.0]);
    }

    proptest! {
        #[test]
        fn prop_fp16_rne_tie_breaks_and_monotonicity(
            a in -65504.0_f32..=65504.0_f32,
            b in -65504.0_f32..=65504.0_f32,
        ) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let q_lo = fp16_to_f32(f32_to_fp16(lo));
            let q_hi = fp16_to_f32(f32_to_fp16(hi));
            prop_assert!(q_lo <= q_hi);
            prop_assert!(q_lo.is_finite() && q_hi.is_finite());

            let packed = pack_fp16x2_u32(a, b);
            prop_assert_eq!(unpack_fp16x2_u32(packed), [fp16_to_f32(f32_to_fp16(a)), fp16_to_f32(f32_to_fp16(b))]);
        }
    }
}
