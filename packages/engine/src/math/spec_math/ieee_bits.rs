//! Bit-preserving `f32 <-> f64` conversions immune to active FTZ/DAZ (`ref[dsp.ctrl.op-n.trust-clip]`)
//! and 6-step division-free `f64` Newton reciprocal (`ref[dsp.op-n.polyadaa.rail-crossing]`).

/// Widens an IEEE-754 `f32` bit pattern (including subnormals) directly to `f64`
/// without executing hardware `f32` instructions subject to FTZ/DAZ (`ref[dsp.ctrl.op-n.trust-clip]`).
/// Returns `None` if `bits` encodes `±Inf` or `NaN`.
#[inline(always)]
pub const fn widen_f32_bits(bits: u32) -> Option<f64> {
    let exponent = (bits >> 23) & 0xff;
    if exponent != 0 && exponent != 0xff {
        let sign = ((bits & 0x8000_0000) as u64) << 32;
        let frac = (bits & 0x007f_ffff) as u64;
        return Some(f64::from_bits(
            sign | (((exponent + 896) as u64) << 52) | (frac << 29),
        ));
    }
    let sign = ((bits & 0x8000_0000) as u64) << 32;
    let fraction = bits & 0x007f_ffff;
    let wide_bits = match (exponent, fraction) {
        (0xff, _) => return None,
        (0, 0) => sign,
        (_, frac) => {
            let leading = 31 - frac.leading_zeros();
            let exponent64 = (leading + 874) as u64;
            let fraction64 = ((frac - (1 << leading)) as u64) << (52 - leading);
            sign | (exponent64 << 52) | fraction64
        }
    };
    Some(f64::from_bits(wide_bits))
}

#[inline]
const fn round_right_even_u64(mantissa: u64, shift: u32) -> u64 {
    if shift > 53 {
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
/// (`M >> 29` for normals, `M >> (-E - 97)` for subnormals, ties-to-even) without hardware `f32`
/// instructions (`ref[dsp.ctrl.op-n.trust-clip]`).
/// Returns `None` on `f64` subnormal, non-finite, or `f32` overflow.
#[inline]
pub const fn round_f64_to_f32_bits(value: f64) -> Option<u32> {
    let bits = value.to_bits();
    let sign = ((bits >> 32) as u32) & 0x8000_0000;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent_bits == 0 {
        return if fraction == 0 { Some(sign) } else { None };
    }
    if exponent_bits == 0x7ff {
        return None;
    }
    let mut exponent = exponent_bits - 1023;
    let mantissa = (1_u64 << 52) | fraction;
    if exponent < -126 {
        let shift = (-exponent - 97) as u32;
        return Some(sign | (round_right_even_u64(mantissa, shift) as u32));
    }
    if exponent > 127 {
        return None;
    }
    let mut rounded = round_right_even_u64(mantissa, 29);
    if rounded == (1 << 24) {
        exponent += 1;
        rounded >>= 1;
    }
    if exponent > 127 {
        return None;
    }
    Some(sign | (((exponent + 127) as u32) << 23) | ((rounded as u32) & 0x007f_ffff))
}

/// Returns the unbiased base-2 exponent of a non-zero finite `f32` bit pattern (`-149..=127`).
#[inline]
pub const fn f32_unbiased_exponent(bits: u32) -> Option<i32> {
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if magnitude == 0 || exponent == 0xff {
        return None;
    }
    if exponent != 0 {
        Some((exponent as i32) - 127)
    } else {
        let bit_len = 32 - (magnitude & 0x007f_ffff).leading_zeros();
        Some((bit_len as i32) - 150)
    }
}

/// Evaluates the 6-step division-free Newton reciprocal $1 / \Delta$ for nonzero $\Delta$ in `f64`
/// (`< 2^-44` relative error, `ref[dsp.op-n.polyadaa.rail-crossing]`).
#[inline]
pub fn newton6_reciprocal_f64(distance: f64) -> f64 {
    let bits = distance.to_bits();
    let sign_bit = bits & (1_u64 << 63);
    let exponent = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let inv_exp = (1023 - exponent - 1).clamp(1, 2046) as u64;
    let mut inverse = f64::from_bits(sign_bit | (inv_exp << 52));
    for _ in 0..6 {
        inverse *= 2.0 - distance * inverse;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::test_support::{Rat, SplitMix64};

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn native_guard_flushes_actual_f32_arithmetic() {
        let _guard = FtzDazGuard::new();
        let subnormal = std::hint::black_box(f32::from_bits(1));
        let one = std::hint::black_box(1.0_f32);
        let normal = std::hint::black_box(f32::MIN_POSITIVE);
        let half = std::hint::black_box(0.5_f32);
        assert_eq!((subnormal * one).to_bits(), 0);
        assert_eq!((normal * half).to_bits(), 0);
    }

    #[test]
    fn bitwise_widening_and_rounding_cover_subnormals_under_ftz_daz() {
        let _guard = FtzDazGuard::new();
        let quantum = f64::from_bits(((1023 - 149) as u64) << 52);
        for fraction in (0..=0x007f_ffff_u32).step_by(127) {
            let expected = (f64::from(fraction) * quantum).to_bits();
            assert_eq!(widen_f32_bits(fraction).unwrap().to_bits(), expected);
            assert_eq!(
                widen_f32_bits(fraction | 0x8000_0000).unwrap().to_bits(),
                expected | (1_u64 << 63)
            );
        }
        assert!(widen_f32_bits(0x7f80_0000).is_none());
        assert!(widen_f32_bits(0x7fc0_0001).is_none());

        let half_quantum = f64::from_bits(((1023 - 150) as u64) << 52);
        for lower in (0..0x0080_0000_u32).step_by(127) {
            let midpoint = f64::from(2 * lower + 1) * half_quantum;
            let expected = lower + (lower & 1);
            assert_eq!(round_f64_to_f32_bits(midpoint), Some(expected));
            assert_eq!(
                round_f64_to_f32_bits(-midpoint),
                Some(expected | 0x8000_0000)
            );
            assert_eq!(
                round_f64_to_f32_bits(f64::from_bits(midpoint.to_bits() - 1)),
                Some(lower)
            );
            assert_eq!(
                round_f64_to_f32_bits(f64::from_bits(midpoint.to_bits() + 1)),
                Some(lower + 1)
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
    }

    /// verify[dsp.op-n.polyadaa.rail-crossing]
    #[test]
    fn test_newton6_reciprocal_f64_accuracy() {
        let _guard = FtzDazGuard::new();
        let tol = f64::from_bits((1023 - 44) << 52);
        let mut rng = SplitMix64::new(0x1234_5678);
        for exp in -140..=128 {
            let base = (exp as f64).exp2();
            let d = base * (1.0 + rng.next_unit_f64());
            let inv = newton6_reciprocal_f64(d);
            let rel_err = (d * inv - 1.0).abs();
            assert!(rel_err < tol, "rel_err={rel_err} at d={d}");
        }
        let r = Rat::new(3, 8);
        assert_eq!(r.round_ties_to_even(), 0);
    }
}
