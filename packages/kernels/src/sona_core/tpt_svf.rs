//! Unconditionally passive `2x2` ZDF `TPT-SVF` (`ref[dsp.op-m.recurrence.symplectic]`,
//! `ref[dsp.op-n.core.stereo-simd]`), division-free pure-`f32` bilinear prewarping,
//! and forward/adjoint `2x2` associative scan operators.

use crate::simd::{GpuVec4, KernelVec4, StereoVec8, const_taylor_inv_fact_f32};

pub mod phs;
pub mod scans;
pub mod tone_folded;

pub use phs::{
    MAX_SVF_K_NL, PhsPredCorrAdjointTerms, PhsPredCorrPassBCoeffs, PhsPredCorrStepContext,
    SVF_PHS_E_MAX, TptSvfPhsPredCorrState, eval_phs_pred_corr_pass_b_coeffs,
    eval_phs_pred_corr_step_adjoint, phs_pred_corr_pass_a_adjoint_elem,
    phs_pred_corr_pass_a_scan_elem, phs_pred_corr_pass_b_adjoint_elem,
    phs_pred_corr_pass_b_scan_elem, step_tpt_svf_phs_stereo8, step_tpt_svf_phs_vec4,
    step_tpt_svf_pred_corr_phs_folded_stereo8, step_tpt_svf_pred_corr_phs_stereo8,
    step_tpt_svf_pred_corr_phs_vec4,
};
pub use scans::{
    TptAdjointElem, TptScanElem, apply_tpt_adjoint, apply_tpt_init, compose_tpt_adjoint_scan,
    compose_tpt_scan,
};
pub use tone_folded::{
    SonaToneFolded3, SonaToneFoldedBlock0, step_tpt_svf_folded3_stereo8, step_tpt_svf_folded3_vec4,
};

/// Minimum normalized cutoff ratio `f_c / f_s` accepted by `prewarp_tan_pi_f_over_fs_f32`.
const MIN_PREWARP_RATIO: f32 = 1.0e-5;
/// Maximum normalized cutoff ratio `f_c / f_s` accepted by `prewarp_tan_pi_f_over_fs_f32`.
const MAX_PREWARP_RATIO: f32 = 0.495;
/// Octant boundary `f_c / f_s = 0.25` (`pi / 4`), splitting `sin / cos` and `cos / sin`.
const OCTANT_SPLIT_RATIO: f32 = 0.25;
/// Nyquist ratio `0.5` used for complementary co-function folding `0.5 - r`.
const NYQUIST_RATIO: f32 = 0.5;
/// Minimum positive tangent gain `g` for passive `TPT-SVF` coefficient derivation.
pub const MIN_TPT_G: f32 = 1.0e-5;

// Compile-time degree-11/10 Maclaurin coefficients for `sin(x)` and `cos(x)` on `[0, pi/4]`
// (`ref[repo.code.constant-eval-and-derivations]`; truncation error `< 2e-9` at `x = pi/4`).
const SIN_C1: f32 = const_taylor_inv_fact_f32(3, true);
const SIN_C2: f32 = const_taylor_inv_fact_f32(5, false);
const SIN_C3: f32 = const_taylor_inv_fact_f32(7, true);
const SIN_C4: f32 = const_taylor_inv_fact_f32(9, false);
const SIN_C5: f32 = const_taylor_inv_fact_f32(11, true);

const COS_C1: f32 = const_taylor_inv_fact_f32(2, true);
const COS_C2: f32 = const_taylor_inv_fact_f32(4, false);
const COS_C3: f32 = const_taylor_inv_fact_f32(6, true);
const COS_C4: f32 = const_taylor_inv_fact_f32(8, false);
const COS_C5: f32 = const_taylor_inv_fact_f32(10, true);

const _: () = assert!(MIN_PREWARP_RATIO > 0.0 && MIN_PREWARP_RATIO < OCTANT_SPLIT_RATIO);
const _: () = assert!(OCTANT_SPLIT_RATIO < MAX_PREWARP_RATIO && MAX_PREWARP_RATIO < NYQUIST_RATIO);
const _: () = assert!(MIN_TPT_G > 0.0);

/// Four-lane `TPT-SVF` precomputed coefficient bundle `(a1, a2, a3, k, 2k)`
/// (`ref[dsp.op-m.recurrence.symplectic]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TptSvfCoeffs4<V: KernelVec4> {
    pub a1: V,
    pub a2: V,
    pub a3: V,
    pub k: V,
    pub two_k: V,
}

impl<V: KernelVec4> TptSvfCoeffs4<V> {
    /// Derives `TPT-SVF` coefficients from prewarped tangent gain `g = tan(pi * f_c / f_s)`
    /// and damping `k = 1 / Q >= 0` using `rcp_ge1_nr2` with zero hardware `FDIV`.
    #[inline]
    #[must_use]
    pub fn from_gk(g: V, k: V) -> Self {
        let g = g.max(V::splat(MIN_TPT_G));
        let k = k.max(V::zero());
        let denom = V::splat(1.0).fma(g, g.add(k));
        let a1 = denom.rcp_ge1_nr2();
        let a2 = g.mul(a1);
        let a3 = g.mul(a2);
        let two_k = k.add(k);
        Self {
            a1,
            a2,
            a3,
            k,
            two_k,
        }
    }
}

/// Pure-`f32`, zero-`FDIV`, zero-`libm` bilinear prewarping approximation of `tan(pi * r)`
/// for normalized cutoff ratio `r = clamp(f_over_fs, 1e-5, 0.495)` (`< 5e-6` relative error).
///
/// Uses `tan(pi * r) = sin(pi * r) / cos(pi * r)` on `r <= 0.25` and complementary folding
/// `tan(pi * r) = cos(pi * (0.5 - r)) / sin(pi * (0.5 - r))` on `r > 0.25` via `rcp_pos_nr3`.
#[inline]
#[must_use]
pub fn prewarp_tan_pi_f_over_fs_f32(f_over_fs: f32) -> f32 {
    let r = if f_over_fs.is_nan() {
        MIN_PREWARP_RATIO
    } else {
        f_over_fs.clamp(MIN_PREWARP_RATIO, MAX_PREWARP_RATIO)
    };
    let lower_octant = r <= OCTANT_SPLIT_RATIO;
    let t = if lower_octant { r } else { NYQUIST_RATIO - r };
    let x = core::f32::consts::PI * t;
    let z = x * x;

    let sin_poly = 1.0 + z * (SIN_C1 + z * (SIN_C2 + z * (SIN_C3 + z * (SIN_C4 + z * SIN_C5))));
    let s = x * sin_poly;

    let c = 1.0 + z * (COS_C1 + z * (COS_C2 + z * (COS_C3 + z * (COS_C4 + z * COS_C5))));

    let (num, den) = if lower_octant { (s, c) } else { (c, s) };
    let inv_den = GpuVec4::splat(den).rcp_pos_nr3().to_array()[0];
    num * inv_den
}

/// Simultaneous four-mode filter taps emitted by a passive `2x2` ZDF `TPT-SVF` step.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TptSvfOutputs<T> {
    pub lp: T,
    pub bp: T,
    pub hp: T,
    pub ap: T,
}

/// Advances 4 independent `TPT-SVF` modes in a `KernelVec4` register (`ref[dsp.op-m.recurrence.symplectic]`).
#[inline(always)]
pub fn step_tpt_svf_vec4<V: KernelVec4>(
    u: V,
    s1: &mut V,
    s2: &mut V,
    coeffs: &TptSvfCoeffs4<V>,
) -> TptSvfOutputs<V> {
    let v3 = u.sub(*s2);
    let v1 = coeffs.a1.mul(*s1).fma(coeffs.a2, v3);
    let v2 = s2.fma(coeffs.a2, *s1).fma(coeffs.a3, v3);
    *s1 = v1.add(v1).sub(*s1).flush_recursive_state();
    *s2 = v2.add(v2).sub(*s2).flush_recursive_state();
    let hp = u.fms(coeffs.k, v1).sub(v2);
    let ap = u.fms(coeffs.two_k, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

/// Advances stereo `L+R` `4`-mode `TPT-SVF` states in `StereoVec8` (`ref[dsp.op-n.core.stereo-simd]`),
/// broadcasting shared filter coefficients across both halves for `0-ULP` stereo parity.
#[inline(always)]
pub fn step_tpt_svf_stereo8<S: StereoVec8>(
    u: S,
    s1: &mut S,
    s2: &mut S,
    coeffs: &TptSvfCoeffs4<S::Half>,
) -> TptSvfOutputs<S> {
    let a1 = S::broadcast_half(coeffs.a1);
    let a2 = S::broadcast_half(coeffs.a2);
    let a3 = S::broadcast_half(coeffs.a3);
    let k = S::broadcast_half(coeffs.k);
    let two_k = S::broadcast_half(coeffs.two_k);
    let v3 = u.sub(*s2);
    let v1 = a1.mul(*s1).fma(a2, v3);
    let v2 = s2.fma(a2, *s1).fma(a3, v3);
    *s1 = v1.add(v1).sub(*s1).flush_recursive_state();
    *s2 = v2.add(v2).sub(*s2).flush_recursive_state();
    let hp = u.fms(k, v1).sub(v2);
    let ap = u.fms(two_k, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::InterleavedStereo8;
    use proptest::prelude::*;

    #[test]
    fn test_prewarp_tan_anchors_and_clamp_boundaries() {
        let cases = [
            (-1.0_f32, 1.0e-5_f64),
            (0.0, 1.0e-5),
            (1.0e-5, 1.0e-5),
            (0.001, 0.001),
            (0.125, 0.125),
            (0.25, 0.25),
            (0.250_01, 0.250_01),
            (0.4, 0.4),
            (0.495, 0.495),
            (0.9, 0.495),
            (f32::NAN, 1.0e-5),
        ];
        for (raw, clamped_f64) in cases {
            let got = prewarp_tan_pi_f_over_fs_f32(raw) as f64;
            let expected = (core::f64::consts::PI * clamped_f64).tan();
            let rel_err = (got - expected).abs() / expected.abs();
            assert!(
                rel_err < 5.0e-6,
                "raw={raw}: got={got}, expected={expected}, rel_err={rel_err}"
            );
        }
    }

    /// verify[dsp.op-n.core.tpt-tone-basis]
    #[test]
    fn test_tpt_svf_zero_equilibrium_and_filter_identities() {
        let g = GpuVec4::from_array([0.05, 0.2, 1.0, 4.0]);
        let k = GpuVec4::from_array([0.0, 0.5, core::f32::consts::SQRT_2, 2.0]);
        let coeffs = TptSvfCoeffs4::from_gk(g, k);

        let mut s1 = GpuVec4::zero();
        let mut s2 = GpuVec4::zero();
        let out = step_tpt_svf_vec4(GpuVec4::zero(), &mut s1, &mut s2, &coeffs);
        for v in [s1, s2, out.lp, out.bp, out.hp, out.ap] {
            for lane in v.to_array() {
                assert_eq!(lane.to_bits(), 0);
            }
        }

        let id_coeffs = TptSvfCoeffs4 {
            a1: GpuVec4::splat(1.0),
            a2: GpuVec4::zero(),
            a3: GpuVec4::zero(),
            k: GpuVec4::zero(),
            two_k: GpuVec4::zero(),
        };
        let mut sub_s1 = GpuVec4::splat(f32::from_bits(1));
        let mut sub_s2 = GpuVec4::splat(f32::from_bits(1));
        let _ = step_tpt_svf_vec4(GpuVec4::zero(), &mut sub_s1, &mut sub_s2, &id_coeffs);
        assert_eq!(sub_s1.to_array(), [0.0; 4]);
        assert_eq!(sub_s2.to_array(), [0.0; 4]);

        let mut st_s1 = InterleavedStereo8::<GpuVec4>::zero();
        let mut st_s2 = InterleavedStereo8::<GpuVec4>::zero();
        let u_l = GpuVec4::from_array([0.5, -0.75, 1.25, -0.25]);
        let u_r = GpuVec4::from_array([-0.3, 0.8, -1.1, 0.4]);
        let mut v_s1_l = GpuVec4::zero();
        let mut v_s2_l = GpuVec4::zero();
        let mut v_s1_r = GpuVec4::zero();
        let mut v_s2_r = GpuVec4::zero();

        for n in 0..16 {
            let scale = if n < 4 { 1.0 } else { 0.0 };
            let ul = u_l.mul(GpuVec4::splat(scale));
            let ur = u_r.mul(GpuVec4::splat(scale));
            let u_st = InterleavedStereo8::from_lr(ul, ur);
            let out_st = step_tpt_svf_stereo8(u_st, &mut st_s1, &mut st_s2, &coeffs);
            let out_l = step_tpt_svf_vec4(ul, &mut v_s1_l, &mut v_s2_l, &coeffs);
            let out_r = step_tpt_svf_vec4(ur, &mut v_s1_r, &mut v_s2_r, &coeffs);

            let (st_lp_l, st_lp_r) = out_st.lp.into_lr();
            let (st_bp_l, st_bp_r) = out_st.bp.into_lr();
            let (st_hp_l, st_hp_r) = out_st.hp.into_lr();
            let (st_ap_l, st_ap_r) = out_st.ap.into_lr();
            assert_eq!(st_lp_l.to_array(), out_l.lp.to_array());
            assert_eq!(st_lp_r.to_array(), out_r.lp.to_array());
            assert_eq!(st_bp_l.to_array(), out_l.bp.to_array());
            assert_eq!(st_bp_r.to_array(), out_r.bp.to_array());
            assert_eq!(st_hp_l.to_array(), out_l.hp.to_array());
            assert_eq!(st_hp_r.to_array(), out_r.hp.to_array());
            assert_eq!(st_ap_l.to_array(), out_l.ap.to_array());
            assert_eq!(st_ap_r.to_array(), out_r.ap.to_array());

            for lane in 0..4 {
                let sum = out_l.lp.to_array()[lane]
                    + k.to_array()[lane] * out_l.bp.to_array()[lane]
                    + out_l.hp.to_array()[lane];
                assert!((sum - ul.to_array()[lane]).abs() < 1.0e-5);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_prewarp_tan_matches_f64(r in 1.0e-4_f32..0.49_f32) {
            let got = prewarp_tan_pi_f_over_fs_f32(r) as f64;
            let expected = (core::f64::consts::PI * (r as f64)).tan();
            let rel_err = (got - expected).abs() / expected.abs();
            prop_assert!(rel_err < 5.0e-6, "r={r}, rel_err={rel_err}");
        }
    }
}
