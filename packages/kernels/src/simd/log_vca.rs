//! Constrained degree-4 minimax IEEE-754 `fast_log2`, `exp2_clamped5_f32`, and the
//! `SensoryQuad` v4 `Unified 0-FDIV Log-Knee VCA` (`[-30 dB, +24 dB]`,
//! `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.router.ballistic-envelope-bank]`).
//!
//! Strictly `#![no_std]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`, zero 64-bit runtime types,
//! and `32-bit` `spirv-unknown-naga-wgsl`-clean.

use super::const_eval::{
    IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS, IEEE754_F32_EXP_MASK,
    IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK, IEEE754_F32_ONE_BITS, const_abs_f32,
    const_f32_pow2,
};
use super::op_constants::eval_exp2_clamped_scalar_f32;
use super::{KernelVec4, StereoVec8};

/// Minimum clamping floor for `UnifiedLogKneeVcaParams::c_knee` (`10^-12`).
pub const LOG_VCA_MIN_C_KNEE: f32 = 1.0e-12;
/// Maximum clamping ceiling for `UnifiedLogKneeVcaParams::c_knee` (`10^6`).
pub const LOG_VCA_MAX_C_KNEE: f32 = 1.0e6;
/// Maximum clamping ceiling for envelope energy `e` and `theta_lin` (`10^9`, `REQ-7.2`).
pub const LOG_VCA_MAX_ENV: f32 = 1.0e9;
/// Minimum octave exponent for `UnifiedLogKneeVca` (`-5.0` -> `2^-5 = -30.10 dB`).
pub const LOG_VCA_MIN_OCT: f32 = -5.0;
/// Maximum octave exponent for `UnifiedLogKneeVca` (`+4.0` -> `2^+4 = +24.08 dB`).
pub const LOG_VCA_MAX_OCT: f32 = 4.0;
/// Finite floor returned by [`fast_log2_f32`] for non-positive or `NaN` inputs (`-126.0`).
pub const FAST_LOG2_FLOOR: f32 = -126.0;
/// Finite ceiling returned by [`fast_log2_f32`] for `+Inf` (`+128.0`).
pub const FAST_LOG2_CEIL: f32 = 128.0;

// Constrained degree-4 Chebyshev minimax coefficients for `log2(1 + f)` on `f in [0, 1)`
// factored as `f + f * (1 - f) * (a0 + f * (a1 + f * a2))` (`<= 3.094e-4` oct / `0.00186 dB`,
// exact `0.0` at powers of 2, `0-ULP` binade-continuous, `ref[dsp.op-n.polyadaa.midpoint]`).
const LOG2_MINIMAX_A0: f32 = 0.440_780_8;
const LOG2_MINIMAX_A1: f32 = -0.242_669_46;
const LOG2_MINIMAX_A2: f32 = 0.081_615_74;

const fn const_fast_log2_unit_frac(f: f32) -> f32 {
    let poly = LOG2_MINIMAX_A0 + f * (LOG2_MINIMAX_A1 + f * LOG2_MINIMAX_A2);
    f + f * (1.0 - f) * poly
}

const _: () = assert!(const_fast_log2_unit_frac(0.0) == 0.0);
const _: () = assert!(const_abs_f32(const_fast_log2_unit_frac(0.5) - 0.584_962_5) < 4.0e-4);
const _: () = assert!(const_fast_log2_unit_frac(1.0) == 1.0);
const _: () = assert!(LOG_VCA_MIN_OCT == -5.0 && LOG_VCA_MAX_OCT == 4.0);

#[inline(always)]
fn clamp_nan_to(x: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if x >= lo {
        if x <= hi { x } else { hi }
    } else if x < lo {
        lo
    } else {
        fallback
    }
}

/// Pure IEEE-754 `binary32` division-free base-2 logarithm (`<= 3.094e-4` oct / `0.00186 dB`
/// max error across all positive normal and subnormal floats, `0-ULP` binade-continuous, exact
/// `0.0` fractional part at every power of 2, `ref[dsp.op-n.polyadaa.midpoint]`).
///
/// Non-positive or `NaN` inputs return [`FAST_LOG2_FLOOR`] (`-126.0`); `+Inf` returns
/// [`FAST_LOG2_CEIL`] (`128.0`).
#[inline(always)]
#[must_use]
pub fn fast_log2_f32(x: f32) -> f32 {
    if x.is_nan() || x <= 0.0 {
        return FAST_LOG2_FLOOR;
    }
    if !x.is_finite() {
        return FAST_LOG2_CEIL;
    }
    let bits = x.to_bits();
    let (norm_bits, exp_adj) = if (bits & IEEE754_F32_EXP_MASK) == 0 {
        let scaled = x * const_f32_pow2(24);
        (scaled.to_bits(), -24_i32)
    } else {
        (bits, 0_i32)
    };
    let exp_unbiased = (((norm_bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES)
        as i32)
        - (IEEE754_F32_EXP_BIAS as i32)
        + exp_adj;
    let mant_bits = (norm_bits & IEEE754_F32_MANTISSA_MASK) | IEEE754_F32_ONE_BITS;
    let f = f32::from_bits(mant_bits) - 1.0;
    let poly = LOG2_MINIMAX_A0 + f * (LOG2_MINIMAX_A1 + f * LOG2_MINIMAX_A2);
    let frac_log2 = f + f * (1.0 - f) * poly;
    (exp_unbiased as f32) + frac_log2
}

/// Evaluates [`fast_log2_f32`] across all 4 lanes of a [`KernelVec4`] register with `0-ULP`
/// scalar parity (`ref[dsp.op-n.polyadaa.midpoint]`).
#[inline(always)]
#[must_use]
pub fn fast_log2_vec4<V: KernelVec4>(x: V) -> V {
    let lanes = x.to_array();
    V::from_array([
        fast_log2_f32(lanes[0]),
        fast_log2_f32(lanes[1]),
        fast_log2_f32(lanes[2]),
        fast_log2_f32(lanes[3]),
    ])
}

/// Evaluates `2^clamp(x, -5.0, 5.0)` in pure IEEE-754 `f32` (`< 2.5e-7`, `0-ULP` `1.0` at `0.0`,
/// `ref[dsp.op-n.polyadaa.midpoint]`).
#[inline(always)]
#[must_use]
pub fn exp2_clamped5_f32(x: f32) -> f32 {
    eval_exp2_clamped_scalar_f32(x, -5.0, 5.0)
}

/// Unified `0-FDIV` Log-Knee VCA parameter bundle spanning optical hyperbolic (`s=1, c_knee=1/k`),
/// power-law (`theta_lin=0, c_knee=1e-12`), and expander/gate (`s < 0`) curves across
/// `[-30.1 dB, +24.1 dB]` (`ref[dsp.router.ballistic-envelope-bank]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UnifiedLogKneeVcaParams {
    /// Linear threshold floor `theta_lin` in `[0.0, 1.0e9]`.
    pub theta_lin: f32,
    /// Log-knee smoothing offset `c_knee` in `[1.0e-12, 1.0e6]`.
    pub c_knee: f32,
    /// Log2 threshold anchor `l_scale` in `[-40.0, 40.0]`.
    pub l_scale: f32,
    /// Gain reduction/expansion slope `s` in `[-32.0, 32.0]` (`>0` compress, `<0` expand, `0` unity).
    pub slope: f32,
    /// Makeup gain in octaves (`b_makeup`), clamped jointly with result into `[-5.0, 4.0]`.
    pub b_makeup: f32,
}

const _: () = assert!(core::mem::size_of::<UnifiedLogKneeVcaParams>() == 20);
const _: () = assert!(core::mem::align_of::<UnifiedLogKneeVcaParams>() == 4);

impl Default for UnifiedLogKneeVcaParams {
    #[inline(always)]
    fn default() -> Self {
        Self::idle()
    }
}

impl UnifiedLogKneeVcaParams {
    /// Returns bit-exact `1.0` (`0x3F80_0000`) idle pass-through VCA parameters.
    #[inline(always)]
    #[must_use]
    pub const fn idle() -> Self {
        Self {
            theta_lin: 0.0,
            c_knee: 1.0,
            l_scale: 0.0,
            slope: 0.0,
            b_makeup: 0.0,
        }
    }

    /// Clamps all fields into finite safe operational ranges (`0 FDIV`).
    #[inline(always)]
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            theta_lin: clamp_nan_to(self.theta_lin, 0.0, LOG_VCA_MAX_ENV, 0.0),
            c_knee: clamp_nan_to(self.c_knee, LOG_VCA_MIN_C_KNEE, LOG_VCA_MAX_C_KNEE, 1.0),
            l_scale: clamp_nan_to(self.l_scale, -40.0, 40.0, 0.0),
            slope: clamp_nan_to(self.slope, -32.0, 32.0, 0.0),
            b_makeup: clamp_nan_to(self.b_makeup, LOG_VCA_MIN_OCT, LOG_VCA_MAX_OCT, 0.0),
        }
    }
}

/// Evaluates the scalar `Unified 0-FDIV Log-Knee VCA` gain
/// `exp2(clamp(b_makeup - s * max(fast_log2(max(e - theta_lin, 0) + c_knee) - l_scale, 0), -5.0, 4.0))`
/// (`ref[dsp.router.ballistic-envelope-bank]`).
#[inline(always)]
#[must_use]
pub fn eval_unified_log_knee_vca_f32(e: f32, params: &UnifiedLogKneeVcaParams) -> f32 {
    let p = params.clamped();
    let e_clean = clamp_nan_to(e, 0.0, LOG_VCA_MAX_ENV, 0.0);
    let excess_lin = e_clean - p.theta_lin;
    let e_excess = (if excess_lin > 0.0 { excess_lin } else { 0.0 }) + p.c_knee;
    let oct = fast_log2_f32(e_excess) - p.l_scale;
    let excess_oct = if oct > 0.0 { oct } else { 0.0 };
    let exp_arg = clamp_nan_to(
        p.b_makeup - p.slope * excess_oct,
        LOG_VCA_MIN_OCT,
        LOG_VCA_MAX_OCT,
        0.0,
    );
    exp2_clamped5_f32(exp_arg)
}

/// Evaluates the `4`-lane `KernelVec4` `Unified 0-FDIV Log-Knee VCA` gain
/// (`ref[dsp.router.ballistic-envelope-bank]`).
#[inline(always)]
#[must_use]
pub fn eval_unified_log_knee_vca_vec4<V: KernelVec4>(e: V, params: &UnifiedLogKneeVcaParams) -> V {
    let p = params.clamped();
    let zero = V::zero();
    let e_clean = e.max(zero).min(V::splat(LOG_VCA_MAX_ENV));
    let e_excess = e_clean
        .sub(V::splat(p.theta_lin))
        .max(zero)
        .add(V::splat(p.c_knee));
    let excess_oct = fast_log2_vec4(e_excess).sub(V::splat(p.l_scale)).max(zero);
    let exp_arg = V::splat(p.b_makeup)
        .fms(V::splat(p.slope), excess_oct)
        .clamp(V::splat(LOG_VCA_MIN_OCT), V::splat(LOG_VCA_MAX_OCT));
    exp_arg.exp2_from_clamped()
}

/// Evaluates a shared `0-ULP` stereo-linked `Unified Log-Knee VCA` gain on `(x_l, x_r)` using
/// `e_link = max(|x_l|, |x_r|)` (`ref[dsp.router.stereo-link-detector]`).
#[inline(always)]
#[must_use]
pub fn apply_stereo_log_knee_vca_stereo8<S: StereoVec8>(
    x: S,
    e_link: f32,
    params: &UnifiedLogKneeVcaParams,
) -> S {
    let g = eval_unified_log_knee_vca_f32(e_link, params);
    x.mul(S::splat(g))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    fn fast_log2_and_deriv_f64_oracle(x: f64) -> (f64, f64) {
        let e = x.log2().floor();
        let pow2_e = 2.0_f64.powf(e);
        let f = x / pow2_e - 1.0;
        let a0 = f64::from(LOG2_MINIMAX_A0);
        let a1 = f64::from(LOG2_MINIMAX_A1);
        let a2 = f64::from(LOG2_MINIMAX_A2);
        let p = a0 + f * (a1 + f * a2);
        let dp = a1 + 2.0 * a2 * f;
        let val = e + f + f * (1.0 - f) * p;
        let dval_df = 1.0 + (1.0 - 2.0 * f) * p + f * (1.0 - f) * dp;
        (val, dval_df / pow2_e)
    }

    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.router.ballistic-envelope-bank]
    /// verify[arch.fp.range-saturation]
    #[test]
    fn test_fast_log2_binade_continuity_subnormals_and_vca_regimes() {
        // Exact 0.0 fractional part at every power of 2 and 0-ULP binade continuity.
        for exp in -149_i32..=127_i32 {
            let p2 = 2.0_f64.powi(exp) as f32;
            assert!(
                (fast_log2_f32(p2) - (exp as f32)).abs() <= 1.0e-6,
                "exp={exp}"
            );
            if exp >= -125 {
                let prev = f32::from_bits(p2.to_bits() - 1);
                let l_prev = fast_log2_f32(prev);
                let l_curr = fast_log2_f32(p2);
                assert!((l_curr - l_prev).abs() < 1.0e-4, "binade jump at exp={exp}");
            }
        }
        assert_eq!(fast_log2_f32(1.0), 0.0);
        assert_eq!(fast_log2_f32(0.0), FAST_LOG2_FLOOR);
        assert_eq!(fast_log2_f32(-1.0), FAST_LOG2_FLOOR);
        assert_eq!(fast_log2_f32(f32::NAN), FAST_LOG2_FLOOR);
        assert_eq!(fast_log2_f32(f32::NEG_INFINITY), FAST_LOG2_FLOOR);
        assert_eq!(fast_log2_f32(f32::INFINITY), FAST_LOG2_CEIL);

        // exp2_clamped5_f32 exact 0-ULP 1.0 at 0.0 and bounds.
        assert_eq!(exp2_clamped5_f32(0.0).to_bits(), 0x3F80_0000);
        assert_eq!(exp2_clamped5_f32(-5.0), 0.03125);
        assert_eq!(exp2_clamped5_f32(4.0), 16.0);
        assert_eq!(exp2_clamped5_f32(f32::NAN), 0.03125);

        // Idle VCA returns bit-exact 1.0 across [-1e9, 1e9] and non-finite inputs.
        let idle = UnifiedLogKneeVcaParams::idle();
        for &e in &[
            0.0_f32,
            -0.0,
            -1.0,
            1.0e-20,
            1.0,
            1.0e9,
            f32::NAN,
            f32::INFINITY,
        ] {
            assert_eq!(
                eval_unified_log_knee_vca_f32(e, &idle).to_bits(),
                0x3F80_0000
            );
        }

        // Hyperbolic optical VCA parity: 1 / (1 + k * e) (<= -57.60 dB RMS rel err).
        let k_opt = 8.0_f32;
        let opt_params = UnifiedLogKneeVcaParams {
            theta_lin: 0.0,
            c_knee: 1.0 / k_opt,
            l_scale: (1.0 / k_opt).log2(),
            slope: 1.0,
            b_makeup: 0.0,
        };
        let mut opt_err2 = 0.0_f64;
        let mut opt_ref2 = 0.0_f64;
        for idx in 0..=200 {
            let e = (idx as f32) * 0.02;
            let got = f64::from(eval_unified_log_knee_vca_f32(e, &opt_params));
            let ref_g = 1.0 / (1.0 + f64::from(k_opt) * f64::from(e));
            opt_err2 += (got - ref_g).powi(2);
            opt_ref2 += ref_g.powi(2);
        }
        let opt_db = 10.0 * (opt_err2 / opt_ref2).log10();
        assert!(opt_db <= -57.60, "optical parity db={opt_db}");

        // Power-law 8:1 VCA parity: (th / e)^(7/8) for e >= th (<= -59.29 dB RMS rel err).
        let th = 0.125_f32;
        let vca_params = UnifiedLogKneeVcaParams {
            theta_lin: 0.0,
            c_knee: 1.0e-12,
            l_scale: th.log2(),
            slope: 0.875,
            b_makeup: 0.0,
        };
        let mut vca_err2 = 0.0_f64;
        let mut vca_ref2 = 0.0_f64;
        for idx in 0..=200 {
            let e = th * 2.0_f32.powf((idx as f32) * 0.02);
            let got = f64::from(eval_unified_log_knee_vca_f32(e, &vca_params));
            let ref_g = (f64::from(th) / f64::from(e)).powf(0.875);
            vca_err2 += (got - ref_g).powi(2);
            vca_ref2 += ref_g.powi(2);
        }
        let vca_db = 10.0 * (vca_err2 / vca_ref2).log10();
        assert!(vca_db <= -59.29, "8:1 power-law VCA parity db={vca_db}");

        // Expander / gate (s < 0): gain rises from -30 dB floor up toward 0 dB above threshold.
        let gate_params = UnifiedLogKneeVcaParams {
            theta_lin: 0.0,
            c_knee: 1.0e-6,
            l_scale: 0.05_f32.log2(),
            slope: -2.0,
            b_makeup: -5.0,
        };
        let g_quiet = eval_unified_log_knee_vca_f32(0.01, &gate_params);
        let g_loud = eval_unified_log_knee_vca_f32(0.4, &gate_params);
        assert!((g_quiet - 0.03125).abs() < 1.0e-5);
        assert!(g_loud > 0.5 && g_loud <= 16.0);

        // ±1e9 / NaN / ±Inf finite bounds and 0-ULP stereo parity.
        for &e in &[-1.0e9_f32, 0.0, 1.0e9, f32::NAN, f32::INFINITY] {
            let g = eval_unified_log_knee_vca_f32(e, &gate_params);
            assert!(g.is_finite() && (0.03125..=16.0).contains(&g));
        }
        let st_in = InterleavedStereo8::<GpuVec4>::splat_lr(0.75, 0.75);
        let st_out = apply_stereo_log_knee_vca_stereo8(st_in, 0.5, &opt_params);
        let (sl, sr) = st_out.into_lr();
        assert_eq!(
            sl.to_array().map(f32::to_bits),
            sr.to_array().map(f32::to_bits)
        );
    }

    proptest! {
        #[test]
        fn prop_fast_log2_accuracy_monotonicity_vec4_and_vca_fd(
            e_bits in 0x0000_0001_u32..0x7F7F_FFFF_u32,
            arr in prop::array::uniform4(1.0e-6_f32..1.0e6_f32),
            e_fd in 0.25_f32..4.0_f32,
        ) {
            let x = f32::from_bits(e_bits);
            let got = fast_log2_f32(x);
            let expected = f64::from(x).log2();
            let err_oct = (f64::from(got) - expected).abs();
            prop_assert!(err_oct <= 3.15e-4, "x={x}, err_oct={err_oct}");

            let v_log = fast_log2_vec4(GpuVec4::from_array(arr)).to_array();
            for lane in 0..4 {
                prop_assert_eq!(v_log[lane].to_bits(), fast_log2_f32(arr[lane]).to_bits());
            }

            let vca_p = UnifiedLogKneeVcaParams {
                theta_lin: 0.05,
                c_knee: 0.01,
                l_scale: (-3.0),
                slope: 0.75,
                b_makeup: 0.5,
            };
            let vca_v = eval_unified_log_knee_vca_vec4(GpuVec4::from_array(arr), &vca_p).to_array();
            for lane in 0..4 {
                prop_assert_eq!(
                    vca_v[lane].to_bits(),
                    eval_unified_log_knee_vca_f32(arr[lane], &vca_p).to_bits()
                );
            }

            // Central FD < 1e-6 vs exact f64 VCA derivative inside smooth active region.
            let e_excess = f64::from(e_fd) - f64::from(vca_p.theta_lin) + f64::from(vca_p.c_knee);
            let (log_val, dlog_de) = fast_log2_and_deriv_f64_oracle(e_excess);
            let exp_arg = f64::from(vca_p.b_makeup) - f64::from(vca_p.slope) * (log_val - f64::from(vca_p.l_scale));
            let g_f64 = 2.0_f64.powf(exp_arg);
            let dg_de = g_f64 * core::f64::consts::LN_2 * (-f64::from(vca_p.slope)) * dlog_de;
            let eps = 1.0e-6_f64;
            let g_hi = {
                let (lv, _) = fast_log2_and_deriv_f64_oracle(e_excess + eps);
                2.0_f64.powf(f64::from(vca_p.b_makeup) - f64::from(vca_p.slope) * (lv - f64::from(vca_p.l_scale)))
            };
            let g_lo = {
                let (lv, _) = fast_log2_and_deriv_f64_oracle(e_excess - eps);
                2.0_f64.powf(f64::from(vca_p.b_makeup) - f64::from(vca_p.slope) * (lv - f64::from(vca_p.l_scale)))
            };
            let fd = (g_hi - g_lo) / (2.0 * eps);
            prop_assert!((dg_de - fd).abs() < 1.0e-6, "dg_de={dg_de}, fd={fd}");
        }
    }
}
