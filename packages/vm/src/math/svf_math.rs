//! Division-free real-time SVF prewarping, Newton–Raphson reciprocal/inverse-sqrt
//! scalars/vectors, algebraic softclipping, and octave exponent helpers
//! (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[repo.code.constant-eval-and-derivations]`).

use crate::math::{
    IEEE754_F32_RCP_SEED, MathBackend, ScalarVec4, SimdVec4, backend::rsqrt_nr2_scalar_f32,
};

const SVF_MIN_FREQ_RATIO: f32 = 1.0e-5;
const SVF_MAX_FREQ_RATIO: f32 = 0.495;
const SVF_W_MIN: f32 = core::f32::consts::PI * SVF_MIN_FREQ_RATIO;
/// Maximum `SONA v3.0` normalized half-angle `w_half = pi * f0 / fs` (`pi * 0.495`).
pub const SVF_W_MAX: f32 = core::f32::consts::PI * SVF_MAX_FREQ_RATIO;
const SVF_G_MIN: f32 = 1.0e-5;
/// Maximum prewarped `TPT-SVF` integrator coefficient `g = tan(w_half)` (`64.0`).
pub const SVF_G_MAX: f32 = 64.0;
const RCP_MIN_DENOM_F32: f32 = 1.0e-6;
const RSQRT_MIN_DENOM_F32: f32 = 1.0e-12;
const SOFTCLIP_INPUT_CLAMP_F32: f32 = 64.0;

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

#[inline(always)]
const fn newton_rcp_step_f32(d_safe: f32, y: f32) -> f32 {
    y + y * (1.0 - d_safe * y)
}

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`, 2 NR steps, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_nr2_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(RCP_MIN_DENOM_F32);
    let y0 = f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(d_safe.to_bits()));
    let y1 = newton_rcp_step_f32(d_safe, y0);
    newton_rcp_step_f32(d_safe, y1)
}

/// Fast positive reciprocal `1.0 / d` (`d` clamped to `>= 1.0e-6`, 3 NR steps, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_f32(d: f32) -> f32 {
    let d_safe = d.max(RCP_MIN_DENOM_F32);
    newton_rcp_step_f32(d_safe, recip_nr2_pos_f32(d_safe))
}

/// 4-lane SIMD positive reciprocal `1.0 / d` (2 NR steps, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_nr2_pos_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rcp_pos_nr2()
}

/// 4-lane SIMD positive reciprocal `1.0 / d` (3 NR steps, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_nr3_pos_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rcp_pos_nr3()
}

/// Fast inverse square root `1.0 / sqrt(d)` (2 NR steps, `0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr2_f32(d: f32) -> f32 {
    rsqrt_nr2_scalar_f32(d.max(RSQRT_MIN_DENOM_F32))
}

/// Fast inverse square root `1.0 / sqrt(d)` (3 NR steps, `0 FSQRT`, `0 FDIV`).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_f32(d: f32) -> f32 {
    let d_safe = d.max(RSQRT_MIN_DENOM_F32);
    let est = rsqrt_nr2_scalar_f32(d_safe);
    est * (1.5 - (0.5 * d_safe * est) * est)
}

/// 4-lane SIMD inverse square root (2 NR steps).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr2_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rsqrt_pos_nr2()
}

/// 4-lane SIMD inverse square root (3 NR steps).
#[inline(always)]
#[must_use]
pub fn rsqrt_nr3_vec4<B: MathBackend>(d: B::Vec4) -> B::Vec4 {
    d.rsqrt_pos_nr3()
}

/// Evaluates $z / \sqrt{1 + z^2}$ clamped to `[-64, 64]` on 4 SIMD lanes.
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

/// Zero-DC asymmetric saturator: $\phi(x + b) - \phi(b_0)$.
#[inline(always)]
#[must_use]
pub fn zero_dc_asym_sat_vec4<B: MathBackend>(
    x: B::Vec4,
    bias: B::Vec4,
    bias0_phi: B::Vec4,
) -> B::Vec4 {
    algebraic_softclip_vec4::<B>(x.add(bias)).sub(bias0_phi)
}

/// Octave-warping curve $2^{\text{oct}}$ for `oct in [-5.0, 5.0]`.
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

/// Scalar bilinear `tan(w_half)` pre-warping (`0 FDIV`, `0 libm`).
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
            assert!(((f64::from(exp2_signed_oct_f32(oct)) - exact) / exact).abs() < 2.0e-6);
        }
        let w = ScalarVec4::from_array([0.05, 0.2, 0.9, 1.4]);
        let g = tan_prewarp_vec4::<ScalarBackend>(w).to_array();
        for (wi, gi) in w.to_array().into_iter().zip(g) {
            assert!(((gi - tan_prewarp_f32(wi)).abs()) < 1.0e-4);
        }
        assert_eq!(tan_prewarp_f32(f32::INFINITY), tan_prewarp_f32(SVF_W_MAX));
        assert!(
            tan_prewarp_f32(f32::INFINITY) <= SVF_G_MAX && tan_prewarp_f32(f32::INFINITY) > 63.0
        );
        assert_eq!(tan_prewarp_f32(f32::NAN), tan_prewarp_f32(SVF_W_MIN));
        assert!(tan_prewarp_f32(f32::NAN) >= SVF_G_MIN && tan_prewarp_f32(f32::NAN) < 4.0e-5);
        let z0 = ScalarVec4::splat(0.5);
        let b0 = algebraic_softclip_vec4::<ScalarBackend>(z0);
        assert_eq!(
            zero_dc_asym_sat_vec4::<ScalarBackend>(ScalarVec4::zero(), z0, b0).to_array(),
            [0.0; 4]
        );
        let dv = ScalarVec4::from_array([1.0, 2.0, 4.0, 8.0]);
        assert!((recip_nr2_pos_vec4::<ScalarBackend>(dv).to_array()[1] - 0.5).abs() < 1e-5);
        assert!((recip_nr3_pos_vec4::<ScalarBackend>(dv).to_array()[2] - 0.25).abs() < 1e-6);
        assert!((rsqrt_nr2_vec4::<ScalarBackend>(dv).to_array()[2] - 0.5).abs() < 1e-5);
        assert!((rsqrt_nr3_vec4::<ScalarBackend>(dv).to_array()[2] - 0.5).abs() < 1e-6);
    }
}
