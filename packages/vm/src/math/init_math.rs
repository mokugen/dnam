//! Pure `#![no_std]` IEEE-754 `!libm`, `0-FDIV`, `0-FSQRT` cold initialization & ZOH math
//! (`rcp_nr6_f64`, `sqrt_f64`, `ln_f64`, `exp_f64`, `powf_f64`, `cold_atan_pos_f64`,
//! `prewarp_tan_pi_f_over_fs_f32`, `rescale_tpt_svf_g_zoh`, `rescale_real_pole_zoh`,
//! `ref[dsp.zoh.op-m-poles]`).

use super::const_eval::{
    F32_INT_EXACT_THRESHOLD, F64_INT_EXACT_THRESHOLD, IEEE754_F64_EXP_ALL_ONES,
    IEEE754_F64_EXP_BIAS, IEEE754_F64_MANTISSA_BITS, IEEE754_F64_MANTISSA_MASK,
    IEEE754_F64_MAX_NORMAL_BIASED_EXP, IEEE754_F64_SIGN_MASK, clamp_finite_f32, const_abs_f32,
    const_abs_f64, const_f64_pow2,
};
use super::kernels::algebraic_unit_rotor_f64;

/// Pure `#![no_std]` floor for `f32`.
#[inline(always)]
pub fn floor_f32(x: f32) -> f32 {
    if x.is_nan() || const_abs_f32(x) >= F32_INT_EXACT_THRESHOLD {
        return x;
    }
    let i = x as i32;
    let f = i as f32;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` floor for `f64`.
#[inline(always)]
pub fn floor_f64(x: f64) -> f64 {
    if x.is_nan() || const_abs_f64(x) >= F64_INT_EXACT_THRESHOLD {
        return x;
    }
    let i = x as i64;
    let f = i as f64;
    if x < f { f - 1.0 } else { f }
}

/// Pure `#![no_std]` ceiling for `f32`.
#[inline(always)]
pub fn ceil_f32(x: f32) -> f32 {
    -floor_f32(-x)
}

/// Pure `#![no_std]` ceiling for `f64`.
#[inline(always)]
pub fn ceil_f64(x: f64) -> f64 {
    -floor_f64(-x)
}

/// Pure `#![no_std]` symmetric round-half-away-from-zero for `f32`.
#[inline(always)]
pub fn round_f32(x: f32) -> f32 {
    if x >= 0.0 {
        floor_f32(x + 0.5)
    } else {
        -floor_f32(-x + 0.5)
    }
}

/// Pure `#![no_std]` symmetric round-half-away-from-zero for `f64`.
#[inline(always)]
pub fn round_f64(x: f64) -> f64 {
    if x >= 0.0 {
        floor_f64(x + 0.5)
    } else {
        -floor_f64(-x + 0.5)
    }
}

/// Alias matching canonical spec identifier `round_half_away` (`ref[dsp.zoh.ctrl-stride]`).
#[inline(always)]
pub fn round_half_away_f64(x: f64) -> f64 {
    if x >= 0.0 {
        floor_f64(x + 0.5)
    } else {
        ceil_f64(x - 0.5)
    }
}

/// 6-step `f64` Newton reciprocal $1/d$ for positive finite $d$ (`0 FDIV`).
#[inline]
pub fn rcp_nr6_f64(d: f64) -> f64 {
    let bits = d.to_bits();
    let sign = bits & IEEE754_F64_SIGN_MASK;
    let exp = (((bits >> IEEE754_F64_MANTISSA_BITS) & (IEEE754_F64_EXP_ALL_ONES as u64)) as i32)
        - IEEE754_F64_EXP_BIAS;
    let inv_exp =
        (IEEE754_F64_EXP_BIAS - 1 - exp).clamp(1, IEEE754_F64_MAX_NORMAL_BIASED_EXP) as u64;
    let mut inv = f64::from_bits(sign | (inv_exp << IEEE754_F64_MANTISSA_BITS));
    for _ in 0..6 {
        inv *= 2.0 - d * inv;
    }
    inv
}

/// Signed 6-step `f64` Newton reciprocal clamped away from zero (`|x| >= 1e-15`, `0 FDIV`).
#[inline]
pub fn rcp_signed_f64(x: f64) -> f64 {
    if x >= 0.0 {
        rcp_nr6_f64(x.max(1.0e-15))
    } else {
        -rcp_nr6_f64((-x).max(1.0e-15))
    }
}

/// Pure `#![no_std]` division-free Newton–Goldschmidt `sqrt` in `f64` (`0 FDIV`, `0 FSQRT`).
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    let (scaled_x, scale_out) = if x < f64::MIN_POSITIVE {
        (x * const_f64_pow2(54), const_f64_pow2(-27))
    } else {
        (x, 1.0)
    };
    let bits = scaled_x.to_bits();
    let rsqrt_bits = crate::math::IEEE754_F64_RSQRT_SEED.wrapping_sub(bits >> 1);
    let mut z = f64::from_bits(rsqrt_bits);
    let half_x = 0.5 * scaled_x;
    for _ in 0..6 {
        z *= 1.5 - half_x * z * z;
    }
    let y = scaled_x * z;
    (y + 0.5 * (scaled_x - y * y) * z) * scale_out
}

/// Pure `#![no_std]` natural logarithm $\ln(x)$ in `f64` (`0 FDIV`).
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = ((bits >> IEEE754_F64_MANTISSA_BITS) & (IEEE754_F64_EXP_ALL_ONES as u64)) as i64
        - i64::from(IEEE754_F64_EXP_BIAS);
    let mut mant = f64::from_bits(
        (bits & IEEE754_F64_MANTISSA_MASK)
            | ((IEEE754_F64_EXP_BIAS as u64) << IEEE754_F64_MANTISSA_BITS),
    );
    let mut k = exp;
    if mant > core::f64::consts::SQRT_2 {
        mant *= 0.5;
        k += 1;
    }
    let u = (mant - 1.0) * rcp_nr6_f64(mant + 1.0);
    let u2 = u * u;
    const INV_ODD: [f64; 10] = [
        1.0 / 3.0,
        1.0 / 5.0,
        1.0 / 7.0,
        1.0 / 9.0,
        1.0 / 11.0,
        1.0 / 13.0,
        1.0 / 15.0,
        1.0 / 17.0,
        1.0 / 19.0,
        1.0 / 21.0,
    ];
    let mut term = u;
    let mut sum = u;
    for &inv_n in &INV_ODD {
        term *= u2;
        sum += term * inv_n;
    }
    2.0 * sum + (k as f64) * core::f64::consts::LN_2
}

/// Pure `#![no_std]` exponential $\exp(x)$ in `f64` (`0 FDIV`).
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    if x < -708.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    let k = round_half_away_f64(x * core::f64::consts::LOG2_E) as i64;
    let r = x - (k as f64) * core::f64::consts::LN_2;
    const INV_STEP: [f64; 14] = [
        1.0,
        1.0 / 2.0,
        1.0 / 3.0,
        1.0 / 4.0,
        1.0 / 5.0,
        1.0 / 6.0,
        1.0 / 7.0,
        1.0 / 8.0,
        1.0 / 9.0,
        1.0 / 10.0,
        1.0 / 11.0,
        1.0 / 12.0,
        1.0 / 13.0,
        1.0 / 14.0,
    ];
    let mut term = 1.0;
    let mut sum = 1.0;
    for &inv_i in &INV_STEP {
        term *= r * inv_i;
        sum += term;
    }
    sum * const_f64_pow2(k as i32)
}

/// Pure `#![no_std]` power function $b^e$ in `f64` (cold ZOH rescaling, `0 FDIV`).
#[inline]
pub fn powf_f64(base: f64, exp: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    if const_abs_f64(exp - 1.0) < 1.0e-15 {
        return base;
    }
    if const_abs_f64(exp) < 1.0e-15 {
        return 1.0;
    }
    exp_f64(exp * ln_f64(base))
}

/// Pure `#![no_std]` cosine $\cos(\pi w)$ for $w \in [0, 1]$ (`0 FDIV`).
#[inline]
pub fn cos_pi_unit_f64(w: f64) -> f64 {
    let theta = (w.clamp(0.0, 1.0)) * core::f64::consts::PI;
    algebraic_unit_rotor_f64(theta).0
}

#[inline]
fn quad_sin_cos_f64(x: f64) -> (f64, f64) {
    crate::math::const_eval::const_sin_cos_f64(x)
}

/// Cold division-free `atan(g)` for `g > 0` via 52-step bisection (`0 FDIV`,
/// `ref[dsp.zoh.op-m-poles]`).
#[inline]
pub fn cold_atan_pos_f64(g: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = core::f64::consts::FRAC_PI_2 - 1.0e-7;
    for _ in 0..52 {
        let mid = 0.5 * (lo + hi);
        let (s, c) = quad_sin_cos_f64(mid);
        if s < g * c {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluates `tan(pi * clamp(r, 1e-5, 0.495))` via octant-reduced Maclaurin series + 3-step `f32`
/// Newton reciprocal (`0 FDIV`, `ref[dsp.zoh.op-m-poles]`).
#[inline]
pub fn prewarp_tan_pi_f_over_fs_f32(r: f32) -> f32 {
    dnam_kernels::prewarp_tan_pi_f_over_fs_f32(r)
}

/// Rescales `TPT-SVF` reference `g_ref` via bilinear ZOH prewarping across `rho_sr = fs_train / fs_core`
/// (`0 FDIV`, `ref[dsp.zoh.op-m-poles]`).
#[inline]
pub fn rescale_tpt_svf_g_zoh(g_ref: f32, rho_sr: f64) -> f32 {
    let g_pos = clamp_finite_f32(g_ref, 1.0e-5, 64.0, 0.1);
    if const_abs_f64(rho_sr - 1.0) <= 1.0e-9 {
        return g_pos;
    }
    let r_eff = (cold_atan_pos_f64(f64::from(g_pos)) * core::f64::consts::FRAC_1_PI * rho_sr)
        .clamp(1.0e-5, 0.495) as f32;
    prewarp_tan_pi_f_over_fs_f32(r_eff)
}

/// Rescales a real pole `p` clamped to `[1e-4, 0.999995]` by exponent `rho` (`0 FDIV`,
/// `ref[dsp.zoh.core-paralru]`).
#[inline]
pub fn rescale_real_pole_zoh(p: f32, rho: f64) -> f32 {
    let p_clamped = clamp_finite_f32(p, 1.0e-4, 0.999_995, 0.999_995);
    if const_abs_f64(rho - 1.0) < 1.0e-12 {
        return p_clamped;
    }
    (powf_f64(f64::from(p_clamped), rho) as f32).clamp(1.0e-4, 0.999_995)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_std_f64_init_math_accuracy_and_zoh_helpers() {
        for &x in &[0.25f64, 0.5, 0.995, 1.0, 2.0, 10.0, 640.0] {
            let s = sqrt_f64(x);
            assert!((s * s - x).abs() < 1.0e-12);
            let l = ln_f64(x);
            let e = exp_f64(l);
            assert!((e - x).abs() < 1.0e-12);
            assert!((x * rcp_nr6_f64(x) - 1.0).abs() < 1.0e-12);
        }
        assert_eq!(rcp_signed_f64(2.0), rcp_nr6_f64(2.0));
        assert_eq!(rcp_signed_f64(-2.0), -rcp_nr6_f64(2.0));
        assert!((powf_f64(0.995, 48000.0 / 96000.0) - sqrt_f64(0.995)).abs() < 1.0e-14);
        assert_eq!(powf_f64(0.5, 0.0), 1.0);
        assert_eq!(powf_f64(0.5, 1.0), 0.5);
        assert_eq!(powf_f64(0.0, 2.0), 0.0);

        assert_eq!(floor_f32(2.7), 2.0);
        assert_eq!(floor_f32(-2.7), -3.0);
        assert_eq!(ceil_f32(2.2), 3.0);
        assert_eq!(round_f32(2.6), 3.0);
        assert_eq!(round_f32(-2.6), -3.0);
        assert_eq!(round_half_away_f64(2.5), 3.0);
        assert_eq!(round_half_away_f64(-2.5), -3.0);
        assert_eq!(sqrt_f64(0.0), 0.0);
        assert_eq!(sqrt_f64(-5.0), 0.0);
        assert!(sqrt_f64(f64::NAN).is_nan());
        assert_eq!(sqrt_f64(f64::INFINITY), f64::INFINITY);
        assert!((sqrt_f64(1.0e-312_f64) - 1.0e-156).abs() < 1.0e-168);
        assert_eq!(ln_f64(0.0), f64::NEG_INFINITY);
        assert_eq!(exp_f64(-800.0), 0.0);
        assert_eq!(exp_f64(800.0), f64::INFINITY);
        assert!((cos_pi_unit_f64(0.5)).abs() < 1e-5);

        let g_48 = 0.25_f32;
        assert_eq!(rescale_tpt_svf_g_zoh(g_48, 1.0), g_48);
        assert_eq!(rescale_tpt_svf_g_zoh(f32::NAN, 1.0), 0.1);
        let g_96 = rescale_tpt_svf_g_zoh(g_48, 0.5);
        let expected = (f64::from(g_48).atan() * 0.5).tan() as f32;
        assert!((g_96 - expected).abs() < 1.0e-5);
        assert_eq!(rescale_real_pole_zoh(0.995, 1.0), 0.995);
        assert_eq!(rescale_real_pole_zoh(f32::NAN, 1.0), 0.999_995);
        let p_96 = rescale_real_pole_zoh(0.995, 0.5);
        assert!((f64::from(p_96) - 0.995_f64.sqrt()).abs() < 1.0e-6);
    }
}
