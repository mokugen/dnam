#![forbid(unsafe_code)]

//! Pure IEEE-754 `f32` `ste_fp16` coordinate decode / binary16 round-trip / re-encode helpers and
//! `rsqrt`/`exp`/`ln` primitives for `inner_opt_step` (zero `libm`, zero `FDIV`, zero `FSQRT`).

use crate::abi::{
    COORD_TRANSFORM_ASINH_BOX_NONNEG, COORD_TRANSFORM_ASINH_BOX_SYM, COORD_TRANSFORM_BOX,
    COORD_TRANSFORM_COMPLEMENT_LOG, COORD_TRANSFORM_FREE, COORD_TRANSFORM_LOG_BOX,
    STE_FP16_META_BIT,
};
use crate::reduce::{clamp_f32_safe, flush_zero, is_finite_bits, rcp_pos_f32};
use dnam_kernels::{
    GpuVec4, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_BIAS, IEEE754_F32_MANTISSA_BITS,
    IEEE754_F32_ONE_BITS, KernelVec4, const_f32_pow2, f32_to_fp16_u32, fp16_u32_to_f32,
};
use spirv_std::glam::Vec4;

pub(super) const U_SATURATION_LIMIT_F32: f32 = 24.0;
const EXP2_C2: f32 = 0.240_221_09;
const EXP2_C3: f32 = 0.055_504_11;
const EXP2_C4: f32 = 0.009_675_892;
const EXP2_C5: f32 = 0.001_333_355_8;
const TWO_POW_64: f32 = const_f32_pow2(64);
const TWO_POW_32: f32 = const_f32_pow2(32);
const TWO_POW_MINUS_64: f32 = const_f32_pow2(-64);
const TWO_POW_MINUS_32: f32 = const_f32_pow2(-32);

/// Exact positive reciprocal square root `1 / sqrt(d)` down to `d >= 1e-24` (`0` `FDIV`, `0` `FSQRT`).
#[inline]
#[must_use]
pub(super) fn rsqrt_exact_pos_f32(d: f32) -> f32 {
    if !is_finite_bits(d) || d <= 0.0 {
        return 0.0;
    }
    if d < 1.0e-5 {
        let ds = if d > 1.0e-24 { d } else { 1.0e-24 };
        let r = GpuVec4::splat(ds * TWO_POW_64).rsqrt_pos_nr3().to_array()[0];
        flush_zero(r * TWO_POW_32)
    } else if d > 1.0e20 {
        let ds = if d < 1.0e32 { d } else { 1.0e32 };
        let r = GpuVec4::splat(ds * TWO_POW_MINUS_64)
            .rsqrt_pos_nr3()
            .to_array()[0];
        flush_zero(r * TWO_POW_MINUS_32)
    } else {
        flush_zero(GpuVec4::splat(d).rsqrt_pos_nr3().to_array()[0])
    }
}

#[inline]
#[must_use]
pub(super) fn sqrt_pos_f32(d: f32) -> f32 {
    if !is_finite_bits(d) || d <= 0.0 {
        0.0
    } else {
        flush_zero(d * rsqrt_exact_pos_f32(d))
    }
}

/// Pure IEEE-754 `exp(x)` supporting `x in [-80.0, 80.0]` (`< 1e-6` rel error, zero `libm`).
#[inline]
#[must_use]
pub(super) fn exp_f32(x: f32) -> f32 {
    if !is_finite_bits(x) {
        return 1.0;
    }
    let y = clamp_f32_safe(x * core::f32::consts::LOG2_E, -100.0, 100.0);
    let k_f = if y >= 0.0 {
        ((y + 0.5) as i32) as f32
    } else {
        -(((-y + 0.5) as i32) as f32)
    };
    let k = if y >= 0.0 {
        (y + 0.5) as i32
    } else {
        -(((-y + 0.5) as i32) as i32)
    };
    let f = y - k_f;
    let p = EXP2_C4 + EXP2_C5 * f;
    let p = EXP2_C3 + p * f;
    let p = EXP2_C2 + p * f;
    let p = core::f32::consts::LN_2 + p * f;
    let mant = 1.0 + p * f;
    let exp_bits =
        ((k + (IEEE754_F32_EXP_BIAS as i32)).clamp(1, 254) as u32) << IEEE754_F32_MANTISSA_BITS;
    flush_zero(f32::from_bits(exp_bits) * mant)
}

/// Pure IEEE-754 `ln(x)` for `x > 0` (`< 1e-6` abs error, zero `libm`, zero `FDIV`).
#[inline]
#[must_use]
pub(super) fn ln_f32(x: f32) -> f32 {
    if !is_finite_bits(x) || x <= 0.0 {
        return -80.0;
    }
    let bits = x.to_bits() & IEEE754_F32_ABS_MASK;
    let mut k = ((bits >> IEEE754_F32_MANTISSA_BITS) as i32) - (IEEE754_F32_EXP_BIAS as i32);
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | IEEE754_F32_ONE_BITS);
    if m >= core::f32::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    let u = (m - 1.0) * rcp_pos_f32(m + 1.0);
    let u2 = u * u;
    let poly = 1.0
        + u2 * (0.333_333_34
            + u2 * (0.2 + u2 * (0.142_857_15 + u2 * (0.111_111_11 + u2 * 0.090_909_09))));
    flush_zero((k as f32) * core::f32::consts::LN_2 + 2.0 * u * poly)
}

#[inline]
#[must_use]
fn sigmoid_f32(u: f32) -> f32 {
    if u >= 0.0 {
        rcp_pos_f32(1.0 + exp_f32(-u))
    } else {
        let e = exp_f32(u);
        e * rcp_pos_f32(1.0 + e)
    }
}

#[inline]
#[must_use]
fn sinh_f32(z: f32) -> f32 {
    0.5 * (exp_f32(z) - exp_f32(-z))
}

#[inline]
#[must_use]
fn asinh_f32(x: f32) -> f32 {
    let ax = x.abs();
    let val = if ax > 1.0e4 {
        ln_f32(2.0 * ax)
    } else if ax < 1.0e-4 {
        ax
    } else {
        ln_f32(ax + sqrt_pos_f32(ax * ax + 1.0))
    };
    if x < 0.0 { -val } else { val }
}

/// Applies forward `binary16` STE snap on `u_raw` whenever `packed_meta` bit 8 is enabled,
/// returning `(u_effective, v_q)` matching `coords::ParamBoundSpec` transforms `0..=5`.
#[inline]
pub(super) fn apply_ste_fp16_u(u_raw: f32, bound_meta: Vec4) -> f32 {
    let meta = bound_meta.w.to_bits();
    let tid = meta & 0xF;
    if (meta & STE_FP16_META_BIT) == 0 {
        return if tid == COORD_TRANSFORM_FREE {
            u_raw
        } else {
            clamp_f32_safe(u_raw, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
        };
    }
    let uc = if tid == COORD_TRANSFORM_FREE {
        clamp_f32_safe(u_raw, -1.0e6, 1.0e6)
    } else {
        clamp_f32_safe(u_raw, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
    };
    let lo = bound_meta.x;
    let hi = bound_meta.y;
    let scale = bound_meta.z.max(1.0e-12);

    let phys = match tid {
        COORD_TRANSFORM_FREE => uc,
        COORD_TRANSFORM_BOX => lo + (hi - lo) * sigmoid_f32(uc),
        COORD_TRANSFORM_LOG_BOX => {
            let la = ln_f32(lo.max(1.0e-15));
            let lb = ln_f32(hi.max(lo * 1.000_001));
            exp_f32(la + (lb - la) * sigmoid_f32(uc))
        }
        COORD_TRANSFORM_COMPLEMENT_LOG => {
            let lmin = ln_f32(lo.max(1.0e-15));
            let lmax = ln_f32(hi.max(lo * 1.000_001));
            1.0 - exp_f32(lmax + (lmin - lmax) * sigmoid_f32(uc))
        }
        COORD_TRANSFORM_ASINH_BOX_SYM => {
            let v_lim = hi.abs().max(lo.abs()).max(1.0e-12);
            let c = asinh_f32(v_lim * rcp_pos_f32(scale));
            let sig = sigmoid_f32(uc);
            scale * sinh_f32(c * (2.0 * sig - 1.0))
        }
        COORD_TRANSFORM_ASINH_BOX_NONNEG => {
            let v_lim = hi.max(1.0e-12);
            let c = asinh_f32(v_lim * rcp_pos_f32(scale));
            scale * sinh_f32(c * sigmoid_f32(uc))
        }
        _ => uc,
    };

    let phys_q = fp16_u32_to_f32(f32_to_fp16_u32(phys));
    let u_q = match tid {
        COORD_TRANSFORM_FREE => phys_q,
        COORD_TRANSFORM_BOX => {
            let span = (hi - lo).max(1.0e-12);
            let vc = clamp_f32_safe(phys_q, lo + 1.0e-6 * span, hi - 1.0e-6 * span);
            let p = clamp_f32_safe((vc - lo) * rcp_pos_f32(span), 1.0e-6, 1.0 - 1.0e-6);
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_LOG_BOX => {
            let la = ln_f32(lo.max(1.0e-15));
            let lb = ln_f32(hi.max(lo * 1.000_001));
            let vc = clamp_f32_safe(phys_q, lo * (1.0 + 1.0e-6), hi * (1.0 - 1.0e-6));
            let p = clamp_f32_safe(
                (ln_f32(vc) - la) * rcp_pos_f32((lb - la).max(1.0e-12)),
                1.0e-6,
                1.0 - 1.0e-6,
            );
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_COMPLEMENT_LOG => {
            let c_min = lo.max(1.0e-15);
            let c_max = hi.max(c_min * 1.000_001);
            let lmin = ln_f32(c_min);
            let lmax = ln_f32(c_max);
            let c_val =
                clamp_f32_safe(1.0 - phys_q, c_min * (1.0 + 1.0e-6), c_max * (1.0 - 1.0e-6));
            let p = clamp_f32_safe(
                (lmax - ln_f32(c_val)) * rcp_pos_f32((lmax - lmin).max(1.0e-12)),
                1.0e-6,
                1.0 - 1.0e-6,
            );
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_ASINH_BOX_SYM => {
            let v_lim = hi.abs().max(lo.abs()).max(1.0e-12);
            let inv_s = rcp_pos_f32(scale);
            let c = asinh_f32(v_lim * inv_s).max(1.0e-12);
            let vc = clamp_f32_safe(phys_q, -v_lim, v_lim);
            let t = clamp_f32_safe(
                asinh_f32(vc * inv_s) * rcp_pos_f32(c),
                -(1.0 - 2.0e-6),
                1.0 - 2.0e-6,
            );
            let p = clamp_f32_safe(0.5 * (t + 1.0), 1.0e-6, 1.0 - 1.0e-6);
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_ASINH_BOX_NONNEG => {
            let v_lim = hi.max(1.0e-12);
            let inv_s = rcp_pos_f32(scale);
            let c = asinh_f32(v_lim * inv_s).max(1.0e-12);
            let vc = clamp_f32_safe(phys_q, 0.0, v_lim);
            let p = clamp_f32_safe(asinh_f32(vc * inv_s) * rcp_pos_f32(c), 1.0e-6, 1.0 - 1.0e-6);
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        _ => uc,
    };
    if tid == COORD_TRANSFORM_FREE {
        if is_finite_bits(lo) && is_finite_bits(hi) && lo < hi {
            clamp_f32_safe(u_q, lo, hi)
        } else {
            clamp_f32_safe(u_q, -1.0e6, 1.0e6)
        }
    } else {
        clamp_f32_safe(u_q, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
    }
}

#[inline]
#[must_use]
pub(super) fn pow_f32_u32(base: f32, mut exp: u32) -> f32 {
    let mut result = 1.0_f32;
    let mut b = base;
    while exp > 0 {
        if (exp & 1) != 0 {
            result *= b;
        }
        b *= b;
        exp >>= 1;
    }
    flush_zero(result)
}
