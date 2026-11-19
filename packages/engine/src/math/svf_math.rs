//! Division-free real-time SVF prewarping, Newton–Raphson reciprocal/inverse-sqrt
//! scalars/vectors, algebraic softclipping, and octave exponent helpers
//! (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[repo.code.constant-eval-and-derivations]`).

use crate::math::{
    IEEE754_F32_RCP_SEED, MathBackend, ScalarVec4, SimdVec4, backend::rsqrt_nr2_scalar_f32,
};

/// Minimum normalized cutoff ratio `r = f_c / f_s` (`1.0e-5`, `ref[dsp.op-n.stage2.tone-basis]`).
const SVF_MIN_FREQ_RATIO: f32 = 1.0e-5;

/// Maximum normalized cutoff ratio `r = f_c / f_s` (`0.495`, `ref[dsp.op-n.stage2.tone-basis]`).
const SVF_MAX_FREQ_RATIO: f32 = 0.495;

/// Minimum `SONA v3.0` normalized half-angle `w_half = pi * f0 / fs` (`pi * 1e-5`,
/// `ref[dsp.op-n.stage2.tone-basis]`).
const SVF_W_MIN: f32 = core::f32::consts::PI * SVF_MIN_FREQ_RATIO;

/// Maximum `SONA v3.0` normalized half-angle `w_half = pi * f0 / fs` (`pi * 0.495`,
/// `ref[dsp.op-n.stage2.tone-basis]`).
pub const SVF_W_MAX: f32 = core::f32::consts::PI * SVF_MAX_FREQ_RATIO;

/// Minimum prewarped `TPT-SVF` integrator coefficient `g = tan(w_half)` (`1.0e-5`,
/// `ref[dsp.op-n.stage2.tone-basis]`).
const SVF_G_MIN: f32 = 1.0e-5;

/// Maximum prewarped `TPT-SVF` integrator coefficient `g = tan(w_half)` (`64.0`,
/// `ref[dsp.op-n.stage2.tone-basis]`).
pub const SVF_G_MAX: f32 = 64.0;

/// Minimum positive denominator floor (`1.0e-6`) for Newton–Raphson reciprocals
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
const RCP_MIN_DENOM_F32: f32 = 1.0e-6;

/// Minimum positive argument floor (`1.0e-12`) for Newton–Raphson inverse square roots
/// (`ref[dsp.op-n.polyadaa.midpoint]`).
const RSQRT_MIN_DENOM_F32: f32 = 1.0e-12;

/// Input magnitude ceiling (`±64.0`) preventing `z^2` overflow in [`algebraic_softclip_vec4`].
const SOFTCLIP_INPUT_CLAMP_F32: f32 = 64.0;

/// Compile-time Maclaurin series coefficient `(-1)^neg / n!` evaluated in `f64` and rounded to
/// `f32` for octant-reduced `sin`/`cos` prewarping (`ref[repo.code.constant-eval-and-derivations]`).
const fn alternating_inv_factorial_f32(n: u32, negative: bool) -> f32 {
    let mut fact = 1.0_f64;
    let mut k = 2_u32;
    while k <= n {
        fact *= k as f64;
        k += 1;
    }
    let inv = 1.0_f64 / fact;
    (if negative { -inv } else { inv }) as f32
}

const _: () = assert!(SVF_W_MIN > 0.0 && SVF_W_MIN < core::f32::consts::FRAC_PI_4);
const _: () =
    assert!(SVF_W_MAX > core::f32::consts::FRAC_PI_4 && SVF_W_MAX < core::f32::consts::FRAC_PI_2);
const _: () = assert!(SVF_G_MIN > 0.0 && SVF_G_MIN < SVF_G_MAX);
const _: () = assert!(SVF_G_MAX.to_bits() == 64.0_f32.to_bits());
const _: () = assert!(RCP_MIN_DENOM_F32 > 0.0 && RSQRT_MIN_DENOM_F32 > 0.0);

#[inline(always)]
const fn newton_rcp_step_f32(d_safe: f32, y: f32) -> f32 {
    y + y * (1.0 - d_safe * y)
}

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using IEEE-754 integer seed
/// `0x7EF3_11C3` followed by 2 Newton–Raphson iterations (`0 FDIV`, `< 6.6e-6` relative error).
#[inline(always)]
#[must_use]
pub fn recip_nr2_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(RCP_MIN_DENOM_F32);
    let y0 = f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(d_safe.to_bits()));
    let y1 = newton_rcp_step_f32(d_safe, y0);
    newton_rcp_step_f32(d_safe, y1)
}

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`) using IEEE-754 integer seed
/// [`IEEE754_F32_RCP_SEED`] (`0x7EF3_11C3`) followed by 3 Newton–Raphson iterations (`< 1 ULP`).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(RCP_MIN_DENOM_F32);
    newton_rcp_step_f32(d_safe, recip_nr2_pos_f32(d_safe))
}

/// 4-lane SIMD positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`, 2 NR steps, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_nr2_pos_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rcp_pos_nr2()
}

/// 4-lane SIMD positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`, 3 NR steps, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rcp_pos_nr3()
}

/// Fast inverse square root `1.0 / sqrt(d)` (`d` clamped to `>= 1.0e-12`) using Quake III /
/// IEEE-754 seed `0x5F37_59DF` followed by 2 Newton–Raphson iterations (`0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr2_f32(d: f32) -> f32 {
    rsqrt_nr2_scalar_f32(d.max(RSQRT_MIN_DENOM_F32))
}

/// Fast inverse square root `1.0 / sqrt(d)` (`d` clamped to `>= 1.0e-12`) using
/// [`IEEE754_F32_RSQRT_SEED`] (`0x5F37_59DF`) followed by 3 Newton–Raphson iterations (`0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_f32(d: f32) -> f32 {
    let d_safe = d.max(RSQRT_MIN_DENOM_F32);
    let est = rsqrt_nr2_scalar_f32(d_safe);
    est * (1.5 - (0.5 * d_safe * est) * est)
}

/// 4-lane SIMD inverse square root `1.0 / sqrt(d)` using in-register integer seed + 2 Newton steps.
#[inline(always)]
#[must_use]
pub fn rsqrt_nr2_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rsqrt_pos_nr2()
}

/// 4-lane SIMD inverse square root `1.0 / sqrt(d)` using in-register integer seed + 3 Newton steps.
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rsqrt_pos_nr3()
}

/// Evaluates $z / \sqrt{1 + z^2}$ clamped safely to `[-64, 64]` to avoid overflow on 4 SIMD lanes.
#[inline(always)]
#[must_use]
pub fn algebraic_softclip_vec4<B: MathBackend>(z: B::Vec4) -> B::Vec4 {
    let zc = z.clamp(
        B::Vec4::splat(-SOFTCLIP_INPUT_CLAMP_F32),
        B::Vec4::splat(SOFTCLIP_INPUT_CLAMP_F32),
    );
    let denom = B::Vec4::splat(1.0).fma(zc, zc);
    zc.mul(rsqrt_nr3_vec4::<B>(denom))
}

/// Zero-DC asymmetric saturator: $\phi(x + b) - \phi(b_0)$ where $\phi(z) = z / \sqrt{1 + z^2}$.
///
/// Precomputing `bias0_phi = algebraic_softclip_vec4::<B>(static_bias)` per backend guarantees bit-exact
/// `+0.0` at `(x = 0, b = static_bias)`.
#[inline(always)]
#[must_use]
pub fn zero_dc_asym_sat_vec4<B: MathBackend>(
    x: B::Vec4,
    bias: B::Vec4,
    bias0_phi: B::Vec4,
) -> B::Vec4 {
    algebraic_softclip_vec4::<B>(x.add(bias)).sub(bias0_phi)
}

/// Octave-warping curve $2^{\text{oct}}$ for `oct in [-5.0, 5.0]` delegated to
/// [`dnam_kernels::KernelVec4::exp2_clamped5`] (`< 2.5e-7` relative error, `0 FDIV`, `0 libm`).
#[inline(always)]
#[must_use]
pub fn exp2_signed_oct_f32(oct: f32) -> f32 {
    <ScalarVec4 as dnam_kernels::KernelVec4>::exp2_clamped5(ScalarVec4::splat(oct)).to_array()[0]
}

/// Octant-reduced pure-`f32` bilinear `tan(w_half)` frequency pre-warping across 4 lanes (`0 FDIV`, `0 libm`).
#[inline(always)]
#[must_use]
pub fn tan_prewarp_vec4<B: MathBackend>(w_half: B::Vec4) -> B::Vec4 {
    const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;
    const QUARTER_PI: f32 = core::f32::consts::FRAC_PI_4;
    // Degree-11/10 Maclaurin coefficients on `[0, pi/4]` (`ref[repo.code.constant-eval-and-derivations]`).
    const SIN_C1: f32 = alternating_inv_factorial_f32(3, true);
    const SIN_C2: f32 = alternating_inv_factorial_f32(5, false);
    const SIN_C3: f32 = alternating_inv_factorial_f32(7, true);
    const SIN_C4: f32 = alternating_inv_factorial_f32(9, false);
    const SIN_C5: f32 = alternating_inv_factorial_f32(11, true);
    const COS_C1: f32 = alternating_inv_factorial_f32(2, true);
    const COS_C2: f32 = alternating_inv_factorial_f32(4, false);
    const COS_C3: f32 = alternating_inv_factorial_f32(6, true);
    const COS_C4: f32 = alternating_inv_factorial_f32(8, false);
    const COS_C5: f32 = alternating_inv_factorial_f32(10, true);

    let x_clamped = w_half.clamp(B::Vec4::splat(SVF_W_MIN), B::Vec4::splat(SVF_W_MAX));
    let q_pi = B::Vec4::splat(QUARTER_PI);
    let h_pi = B::Vec4::splat(HALF_PI);
    let x = <B::Vec4 as dnam_kernels::KernelVec4>::select_le(
        x_clamped,
        q_pi,
        x_clamped,
        h_pi.sub(x_clamped),
    );
    let z = x.mul(x);

    let sin_poly = B::Vec4::splat(SIN_C4).fma(B::Vec4::splat(SIN_C5), z);
    let sin_poly = B::Vec4::splat(SIN_C3).fma(sin_poly, z);
    let sin_poly = B::Vec4::splat(SIN_C2).fma(sin_poly, z);
    let sin_poly = B::Vec4::splat(SIN_C1).fma(sin_poly, z);
    let sin_poly = B::Vec4::splat(1.0).fma(sin_poly, z);
    let s = x.mul(sin_poly);

    let cos_poly = B::Vec4::splat(COS_C4).fma(B::Vec4::splat(COS_C5), z);
    let cos_poly = B::Vec4::splat(COS_C3).fma(cos_poly, z);
    let cos_poly = B::Vec4::splat(COS_C2).fma(cos_poly, z);
    let cos_poly = B::Vec4::splat(COS_C1).fma(cos_poly, z);
    let c = B::Vec4::splat(1.0).fma(cos_poly, z);

    let num = <B::Vec4 as dnam_kernels::KernelVec4>::select_le(x_clamped, q_pi, s, c);
    let den = <B::Vec4 as dnam_kernels::KernelVec4>::select_le(x_clamped, q_pi, c, s);
    num.mul(den.rcp_pos_nr3())
        .clamp(B::Vec4::splat(SVF_G_MIN), B::Vec4::splat(SVF_G_MAX))
}

/// Octant-reduced pure-`f32` bilinear `tan(w_half)` frequency pre-warping (`0 FDIV`, `0 libm`),
/// delegated to [`dnam_kernels::prewarp_tan_pi_f_over_fs_f32`] across `r = f / fs in [1e-5, 0.495]`
/// (`ref[dsp.op-n.stage2.tone-basis]`).
#[inline(always)]
#[must_use]
pub fn tan_prewarp_f32(w_half: f32) -> f32 {
    let clamped = if w_half.is_finite() {
        w_half.clamp(SVF_W_MIN, SVF_W_MAX)
    } else if w_half.is_sign_positive() && w_half.is_infinite() {
        SVF_W_MAX
    } else {
        SVF_W_MIN
    };
    dnam_kernels::prewarp_tan_pi_f_over_fs_f32(clamped * core::f32::consts::FRAC_1_PI)
        .clamp(SVF_G_MIN, SVF_G_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;

    #[test]
    fn test_svf_math_accuracy_and_zero_dc_invariants() {
        for &d in &[1.0e-4_f32, 0.25, 1.0, 2.5, 17.0, 120.0] {
            let exact_rcp = 1.0_f64 / f64::from(d);
            assert!(((f64::from(recip_nr2_pos_f32(d)) - exact_rcp) / exact_rcp).abs() < 1.0e-5);
            assert!(((f64::from(recip_nr3_pos_f32(d)) - exact_rcp) / exact_rcp).abs() < 1.0e-6);
            let exact_rsq = 1.0_f64 / f64::from(d).sqrt();
            assert!(((f64::from(rsqrt_nr2_f32(d)) - exact_rsq) / exact_rsq).abs() < 1.0e-5);
            assert!(((f64::from(rsqrt_nr3_f32(d)) - exact_rsq) / exact_rsq).abs() < 1.0e-6);
        }

        assert_eq!(exp2_signed_oct_f32(f32::NAN), 0.03125);
        for &oct in &[-5.0_f32, -2.5, 0.0, 1.25, 5.0] {
            let exact = f64::from(oct).exp2();
            assert!(((f64::from(exp2_signed_oct_f32(oct)) - exact) / exact).abs() < 1.0e-6);
        }

        let w = [1.0e-4_f32, 0.1, 0.5, 1.1];
        let v_tan = tan_prewarp_vec4::<ScalarBackend>(ScalarVec4::from_array(w)).to_array();
        for (idx, &wi) in w.iter().enumerate() {
            assert!((tan_prewarp_f32(wi) - f64::from(wi).tan() as f32).abs() < 1.0e-5);
            assert!((v_tan[idx] - tan_prewarp_f32(wi)).abs() < 1.0e-6);
        }

        // Verify SONA v3.0 full prewarp domain `[SVF_W_MIN, SVF_W_MAX]` (`g <= 64.0`) and NaN/Inf.
        let w_bounds = [SVF_W_MIN, 1.0e-3, 1.45, SVF_W_MAX];
        let v_bounds =
            tan_prewarp_vec4::<ScalarBackend>(ScalarVec4::from_array(w_bounds)).to_array();
        for (idx, &wi) in w_bounds.iter().enumerate() {
            let exact = f64::from(wi).tan();
            let g_scalar = tan_prewarp_f32(wi);
            let g_vec = v_bounds[idx];
            assert!((SVF_G_MIN..=SVF_G_MAX).contains(&g_scalar));
            assert!((SVF_G_MIN..=SVF_G_MAX).contains(&g_vec));
            assert!(((f64::from(g_scalar) - exact) / exact).abs() < 2.0e-5);
            assert!(((f64::from(g_vec) - exact) / exact).abs() < 2.0e-5);
        }
        assert!(tan_prewarp_f32(f32::NAN).is_finite());
        assert!(tan_prewarp_f32(f32::INFINITY) <= SVF_G_MAX);
        assert!(tan_prewarp_f32(f32::NEG_INFINITY) >= SVF_G_MIN);

        let bias = ScalarVec4::splat(0.35);
        let bias0 = algebraic_softclip_vec4::<ScalarBackend>(bias);
        let out_zero = zero_dc_asym_sat_vec4::<ScalarBackend>(ScalarVec4::splat(0.0), bias, bias0);
        assert_eq!(out_zero.to_array(), [0.0; 4]);
    }
}
