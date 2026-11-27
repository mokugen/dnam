//! Block 0 Port-Hamiltonian (`PHS`) diode-bandpass `2x2` ZDF `TPT-SVF` kernels
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[dsp.op-m.recurrence.symplectic]`).
//!
//! Preserves v3.0 frozen-state `step_tpt_svf_phs_*` (`k_eff[n] = k + k_nl * min(s1[n-1]^2, 64)`)
//! alongside v4.0 1-step Predictor-Corrector Passive `PHS` (`step_tpt_svf_pred_corr_phs_*`,
//! `||M_n||_2 <= 1`, `O(log2 N)` 2-pass GPU scan).

pub mod scans;

pub use scans::{
    PhsPredCorrAdjointTerms, PhsPredCorrPassBCoeffs, PhsPredCorrStepContext,
    eval_phs_pred_corr_pass_b_coeffs, eval_phs_pred_corr_step_adjoint,
    phs_pred_corr_pass_a_adjoint_elem, phs_pred_corr_pass_a_scan_elem,
    phs_pred_corr_pass_b_adjoint_elem, phs_pred_corr_pass_b_scan_elem,
};

use crate::simd::{KernelVec4, StereoVec8, const_f32_pow2};

use super::tone_folded::SonaToneFoldedBlock0;
use super::{MIN_TPT_G, TptSvfCoeffs4, TptSvfOutputs};

/// Compile-time capacitor/inductor energy saturation ceiling `E_max = 2^6 = 64.0` for Block 0 PHS.
pub const SVF_PHS_E_MAX: f32 = const_f32_pow2(6);
/// Maximum nonlinear diode-bandpass damping gain `k_nl = 2^4 = 16.0` on Block 0 `CoreQuad`s.
pub const MAX_SVF_K_NL: f32 = const_f32_pow2(4);
/// Maximum prewarped tangent gain `g = 2^6 = 64.0` in `TPT-SVF`.
const MAX_SVF_G: f32 = const_f32_pow2(6);
/// Maximum base linear damping `k = 2^3 = 8.0` in `TPT-SVF`.
const MAX_SVF_K: f32 = const_f32_pow2(3);
/// Worst-case denominator `1 + 64 * (64 + 8 + 16 * 64) = 70,145.0` within `rcp_ge1_nr2` domain.
const MAX_PHS_DENOM: f32 = 1.0 + MAX_SVF_G * (MAX_SVF_G + MAX_SVF_K + MAX_SVF_K_NL * SVF_PHS_E_MAX);

const _: () = assert!(SVF_PHS_E_MAX == 64.0);
const _: () = assert!(MAX_SVF_K_NL == 16.0);
const _: () = assert!(MAX_SVF_K + MAX_SVF_K_NL * SVF_PHS_E_MAX == 1032.0);
const _: () = assert!(MAX_PHS_DENOM == 70_145.0);

/// Evaluates Block 0 linearly-implicit PHS `TPT-SVF` across 4 `KernelVec4` lanes.
///
/// Matches [`super::step_tpt_svf_vec4`] at `0-ULP` when `k_nl == 0.0` (with non-negative finite `k`).
#[inline(always)]
pub fn step_tpt_svf_phs_vec4<V: KernelVec4>(
    u: V,
    s1: &mut V,
    s2: &mut V,
    g: V,
    k: V,
    k_nl: V,
) -> TptSvfOutputs<V> {
    let g_safe = g.max(V::splat(MIN_TPT_G));
    let k_safe = k.max(V::zero());
    let k_nl_safe = k_nl.max(V::zero());
    let s1_prev = *s1;
    let s2_prev = *s2;
    let e = s1_prev.mul(s1_prev).min(V::splat(SVF_PHS_E_MAX));
    let k_eff = k_safe.fma(k_nl_safe, e);
    let denom = V::splat(1.0).fma(g_safe, g_safe.add(k_eff));
    let a1 = denom.rcp_ge1_nr2();
    let a2 = g_safe.mul(a1);
    let a3 = g_safe.mul(a2);
    let two_k_eff = k_eff.add(k_eff);

    let v3 = u.sub(s2_prev);
    let v1 = a1.mul(s1_prev).fma(a2, v3);
    let v2 = s2_prev.fma(a2, s1_prev).fma(a3, v3);
    *s1 = v1.add(v1).sub(s1_prev).flush_recursive_state();
    *s2 = v2.add(v2).sub(s2_prev).flush_recursive_state();
    let hp = u.fms(k_eff, v1).sub(v2);
    let ap = u.fms(two_k_eff, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

/// Evaluates Block 0 linearly-implicit PHS `TPT-SVF` jointly across stereo `L+R` `StereoVec8` lanes,
/// adapting damping independently for Left and Right channels from each channel's own `s1[n-1]^2`
/// (`ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
pub fn step_tpt_svf_phs_stereo8<S: StereoVec8>(
    u: S,
    s1: &mut S,
    s2: &mut S,
    g: S::Half,
    k: S::Half,
    k_nl: S::Half,
) -> TptSvfOutputs<S> {
    let g_safe = S::broadcast_half(g.max(S::Half::splat(MIN_TPT_G)));
    let k_safe = S::broadcast_half(k.max(S::Half::zero()));
    let k_nl_safe = S::broadcast_half(k_nl.max(S::Half::zero()));
    let s1_prev = *s1;
    let s2_prev = *s2;
    let e = s1_prev.mul(s1_prev).min(S::splat(SVF_PHS_E_MAX));
    let k_eff = k_safe.fma(k_nl_safe, e);
    let denom = S::splat(1.0).fma(g_safe, g_safe.add(k_eff));
    let a1 = denom.rcp_ge1_nr2();
    let a2 = g_safe.mul(a1);
    let a3 = g_safe.mul(a2);
    let two_k_eff = k_eff.add(k_eff);

    let v3 = u.sub(s2_prev);
    let v1 = a1.mul(s1_prev).fma(a2, v3);
    let v2 = s2_prev.fma(a2, s1_prev).fma(a3, v3);
    *s1 = v1.add(v1).sub(s1_prev).flush_recursive_state();
    *s2 = v2.add(v2).sub(s2_prev).flush_recursive_state();
    let hp = u.fms(k_eff, v1).sub(v2);
    let ap = u.fms(two_k_eff, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

/// Dual-pass recursive state for Block `0` 1-step Predictor-Corrector Passive `PHS` `TPT-SVF`
/// (`Pass A` linear predictor `(s1_lin, s2_lin)` + `Pass B` passive LTV corrector `(s1, s2)`,
/// `ref[dsp.op-n.core.tpt-tone-basis]`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct TptSvfPhsPredCorrState<T> {
    pub s1_lin: T,
    pub s2_lin: T,
    pub s1: T,
    pub s2: T,
}

impl<V: KernelVec4> TptSvfPhsPredCorrState<V> {
    /// Initializes all predictor and corrector states to canonical `+0.0`.
    #[inline(always)]
    #[must_use]
    pub fn zero_vec4() -> Self {
        Self {
            s1_lin: V::zero(),
            s2_lin: V::zero(),
            s1: V::zero(),
            s2: V::zero(),
        }
    }
}

impl<S: StereoVec8> TptSvfPhsPredCorrState<S> {
    /// Initializes stereo predictor and corrector states to canonical `+0.0`.
    #[inline(always)]
    #[must_use]
    pub fn zero_stereo8() -> Self {
        Self {
            s1_lin: S::zero(),
            s2_lin: S::zero(),
            s1: S::zero(),
            s2: S::zero(),
        }
    }
}

/// Advances 4 `KernelVec4` modes via 1-step Predictor-Corrector Passive `PHS` `TPT-SVF`
/// (`k_eff = k + k_nl * min(v1_lin^2, 64.0)`, `0-ULP` parity vs linear when `k_nl == 0.0`,
/// `ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
pub fn step_tpt_svf_pred_corr_phs_vec4<V: KernelVec4>(
    u: V,
    state: &mut TptSvfPhsPredCorrState<V>,
    g: V,
    k: V,
    k_nl: V,
) -> TptSvfOutputs<V> {
    let u_safe = u.flush_recursive_state();
    let g_safe = g.max(V::splat(MIN_TPT_G));
    let k_safe = k.max(V::zero());
    let k_nl_safe = k_nl.max(V::zero());

    // Pass A: Linear predictor (v1_lin)
    let lin = TptSvfCoeffs4::from_gk(g_safe, k_safe);
    let s1_a = state.s1_lin;
    let s2_a = state.s2_lin;
    let v3_a = u_safe.sub(s2_a);
    let v1_lin = lin.a1.mul(s1_a).fma(lin.a2, v3_a);
    let v2_a = s2_a.fma(lin.a2, s1_a).fma(lin.a3, v3_a);
    state.s1_lin = v1_lin.add(v1_lin).sub(s1_a).flush_recursive_state();
    state.s2_lin = v2_a.add(v2_a).sub(s2_a).flush_recursive_state();

    // Pass B: Passive LTV corrector
    let e = v1_lin.mul(v1_lin).min(V::splat(SVF_PHS_E_MAX));
    let k_eff = k_safe.fma(k_nl_safe, e);
    let denom = V::splat(1.0).fma(g_safe, g_safe.add(k_eff));
    let a1 = denom.rcp_ge1_nr2();
    let a2 = g_safe.mul(a1);
    let a3 = g_safe.mul(a2);
    let two_k_eff = k_eff.add(k_eff);

    let s1_b = state.s1;
    let s2_b = state.s2;
    let v3_b = u_safe.sub(s2_b);
    let v1 = a1.mul(s1_b).fma(a2, v3_b);
    let v2 = s2_b.fma(a2, s1_b).fma(a3, v3_b);
    state.s1 = v1.add(v1).sub(s1_b).flush_recursive_state();
    state.s2 = v2.add(v2).sub(s2_b).flush_recursive_state();
    let hp = u_safe.fms(k_eff, v1).sub(v2);
    let ap = u_safe.fms(two_k_eff, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

/// Advances stereo `L+R` Block `0` 1-step Predictor-Corrector Passive `PHS` `TPT-SVF` across
/// `StereoVec8` (`0-ULP` stereo symmetry and `0-ULP` linear parity when `k_nl == 0.0`).
#[inline(always)]
pub fn step_tpt_svf_pred_corr_phs_stereo8<S: StereoVec8>(
    u: S,
    state: &mut TptSvfPhsPredCorrState<S>,
    g: S::Half,
    k: S::Half,
    k_nl: S::Half,
) -> TptSvfOutputs<S> {
    let u_safe = u.flush_recursive_state();
    let g_safe = S::broadcast_half(g.max(S::Half::splat(MIN_TPT_G)));
    let k_safe = S::broadcast_half(k.max(S::Half::zero()));
    let k_nl_safe = S::broadcast_half(k_nl.max(S::Half::zero()));

    let lin = TptSvfCoeffs4::from_gk(g.max(S::Half::splat(MIN_TPT_G)), k.max(S::Half::zero()));
    let a1_a = S::broadcast_half(lin.a1);
    let a2_a = S::broadcast_half(lin.a2);
    let a3_a = S::broadcast_half(lin.a3);
    let s1_a = state.s1_lin;
    let s2_a = state.s2_lin;
    let v3_a = u_safe.sub(s2_a);
    let v1_lin = a1_a.mul(s1_a).fma(a2_a, v3_a);
    let v2_a = s2_a.fma(a2_a, s1_a).fma(a3_a, v3_a);
    state.s1_lin = v1_lin.add(v1_lin).sub(s1_a).flush_recursive_state();
    state.s2_lin = v2_a.add(v2_a).sub(s2_a).flush_recursive_state();

    let e = v1_lin.mul(v1_lin).min(S::splat(SVF_PHS_E_MAX));
    let k_eff = k_safe.fma(k_nl_safe, e);
    let denom = S::splat(1.0).fma(g_safe, g_safe.add(k_eff));
    let a1 = denom.rcp_ge1_nr2();
    let a2 = g_safe.mul(a1);
    let a3 = g_safe.mul(a2);
    let two_k_eff = k_eff.add(k_eff);

    let s1_b = state.s1;
    let s2_b = state.s2;
    let v3_b = u_safe.sub(s2_b);
    let v1 = a1.mul(s1_b).fma(a2, v3_b);
    let v2 = s2_b.fma(a2, s1_b).fma(a3, v3_b);
    state.s1 = v1.add(v1).sub(s1_b).flush_recursive_state();
    state.s2 = v2.add(v2).sub(s2_b).flush_recursive_state();
    let hp = u_safe.fms(k_eff, v1).sub(v2);
    let ap = u_safe.fms(two_k_eff, v1);
    TptSvfOutputs {
        lp: v2,
        bp: v1,
        hp,
        ap,
    }
}

/// Advances Block `0` 1-step Predictor-Corrector Passive `PHS` `TPT-SVF` with folded
/// [`SonaToneFoldedBlock0`] readout (eliminating `HP`/`AP`), returning `(tone_out, bp)`.
#[inline(always)]
pub fn step_tpt_svf_pred_corr_phs_folded_stereo8<S: StereoVec8>(
    u: S,
    state: &mut TptSvfPhsPredCorrState<S>,
    g: S::Half,
    k: S::Half,
    k_nl: S::Half,
    tone: &SonaToneFoldedBlock0<S::Half>,
) -> (S, S) {
    let u_safe = u.flush_recursive_state();
    let g_h = g.max(S::Half::splat(MIN_TPT_G));
    let k_h = k.max(S::Half::zero());
    let g_safe = S::broadcast_half(g_h);
    let k_safe = S::broadcast_half(k_h);
    let k_nl_safe = S::broadcast_half(k_nl.max(S::Half::zero()));

    let lin = TptSvfCoeffs4::from_gk(g_h, k_h);
    let a1_a = S::broadcast_half(lin.a1);
    let a2_a = S::broadcast_half(lin.a2);
    let a3_a = S::broadcast_half(lin.a3);
    let s1_a = state.s1_lin;
    let s2_a = state.s2_lin;
    let v3_a = u_safe.sub(s2_a);
    let v1_lin = a1_a.mul(s1_a).fma(a2_a, v3_a);
    let v2_a = s2_a.fma(a2_a, s1_a).fma(a3_a, v3_a);
    state.s1_lin = v1_lin.add(v1_lin).sub(s1_a).flush_recursive_state();
    state.s2_lin = v2_a.add(v2_a).sub(s2_a).flush_recursive_state();

    let e = v1_lin.mul(v1_lin).min(S::splat(SVF_PHS_E_MAX));
    let k_eff = k_safe.fma(k_nl_safe, e);
    let denom = S::splat(1.0).fma(g_safe, g_safe.add(k_eff));
    let a1 = denom.rcp_ge1_nr2();
    let a2 = g_safe.mul(a1);
    let a3 = g_safe.mul(a2);

    let s1_b = state.s1;
    let s2_b = state.s2;
    let v3_b = u_safe.sub(s2_b);
    let v1 = a1.mul(s1_b).fma(a2, v3_b);
    let v2 = s2_b.fma(a2, s1_b).fma(a3, v3_b);
    state.s1 = v1.add(v1).sub(s1_b).flush_recursive_state();
    state.s2 = v2.add(v2).sub(s2_b).flush_recursive_state();
    (tone.eval_stereo8(v2, v1, k_eff, u_safe), v1)
}

#[cfg(test)]
mod tests {
    use super::super::{TptSvfCoeffs4, step_tpt_svf_stereo8, step_tpt_svf_vec4};
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

    /// verify[dsp.op-n.core.tpt-tone-basis]
    #[test]
    fn test_phs_tpt_svf_zero_dc_and_e_max_saturation() {
        let g = GpuVec4::from_array([0.05, 0.25, 1.5, 12.0]);
        let k = GpuVec4::from_array([0.0, 0.5, 2.0, 8.0]);
        let k_nl = GpuVec4::from_array([1.0, 4.0, 8.0, 16.0]);

        // Zero-DC equilibrium (+0.0) at resting zero state and input
        let mut s1 = InterleavedStereo8::<GpuVec4>::zero();
        let mut s2 = InterleavedStereo8::<GpuVec4>::zero();
        for _ in 0..32 {
            let out =
                step_tpt_svf_phs_stereo8(InterleavedStereo8::zero(), &mut s1, &mut s2, g, k, k_nl);
            for v in [s1, s2, out.lp, out.bp, out.hp, out.ap] {
                let (vl, vr) = v.into_lr();
                assert_eq!(vl.to_array().map(f32::to_bits), [0; 4]);
                assert_eq!(vr.to_array().map(f32::to_bits), [0; 4]);
            }
        }

        // Above |s1| >= 8.0 (E_max = 64.0), s1 = 100.0 matches s1 = 8.0 coefficients (k_eff reaches 1032.0)
        let mut s1_a = GpuVec4::splat(100.0);
        let mut s2_a = GpuVec4::splat(-3.0);
        let mut s1_b = GpuVec4::splat(8.0);
        let mut s2_b = GpuVec4::splat(-3.0);
        let out_a = step_tpt_svf_phs_vec4(GpuVec4::splat(2.0), &mut s1_a, &mut s2_a, g, k, k_nl);
        let out_b = step_tpt_svf_phs_vec4(GpuVec4::splat(2.0), &mut s1_b, &mut s2_b, g, k, k_nl);
        // Check saturated effective k_eff = 8 + 16*64 = 1032 on lane 3 against TptSvfCoeffs4::from_gk without clamping to 8.0!
        let k_sat_expected = GpuVec4::from_array([
            0.0 + 1.0 * 64.0,
            0.5 + 4.0 * 64.0,
            2.0 + 8.0 * 64.0,
            8.0 + 16.0 * 64.0,
        ]);
        let sat_coeffs = TptSvfCoeffs4::from_gk(g, k_sat_expected);
        let mut s1_lin = GpuVec4::splat(8.0);
        let mut s2_lin = GpuVec4::splat(-3.0);
        let out_lin = step_tpt_svf_vec4(GpuVec4::splat(2.0), &mut s1_lin, &mut s2_lin, &sat_coeffs);
        assert_eq!(
            out_b.bp.to_array().map(f32::to_bits),
            out_lin.bp.to_array().map(f32::to_bits)
        );
        assert_eq!(
            out_b.lp.to_array().map(f32::to_bits),
            out_lin.lp.to_array().map(f32::to_bits)
        );
        assert!(out_a.bp.to_array().iter().all(|v| v.is_finite()));
        assert!(out_a.hp.to_array().iter().all(|v| v.is_finite()));

        // Exact worst-case boundary (g = 64, k = 8, k_nl = 16, |s1| >= 8 => denom = MAX_PHS_DENOM = 70_145).
        let max_g = GpuVec4::splat(MAX_SVF_G);
        let max_k = GpuVec4::splat(MAX_SVF_K);
        let max_k_nl = GpuVec4::splat(MAX_SVF_K_NL);
        let rcp = GpuVec4::splat(MAX_PHS_DENOM).rcp_ge1_nr2().to_array()[0];
        let rcp_ref = 1.0_f64 / (MAX_PHS_DENOM as f64);
        assert!(
            ((f64::from(rcp) - rcp_ref) / rcp_ref).abs() < 1.0e-6,
            "rcp_ge1_nr2({MAX_PHS_DENOM})={rcp}"
        );

        let mut e_s1 = GpuVec4::splat(SVF_PHS_E_MAX.sqrt()); // 8.0
        let mut e_s2 = GpuVec4::splat(-4.0);
        let mut prev_energy = e_s1.0[0] * e_s1.0[0] + e_s2.0[0] * e_s2.0[0];
        for _ in 0..32 {
            let out_e = step_tpt_svf_phs_vec4(
                GpuVec4::zero(),
                &mut e_s1,
                &mut e_s2,
                max_g,
                max_k,
                max_k_nl,
            );
            assert!(out_e.lp.to_array().iter().all(|v| v.is_finite()));
            let cur_energy = e_s1.0[0] * e_s1.0[0] + e_s2.0[0] * e_s2.0[0];
            assert!(
                cur_energy <= prev_energy + 1.0e-6,
                "energy grew at MAX_PHS_DENOM: {cur_energy} > {prev_energy}"
            );
            prev_energy = cur_energy;
        }
    }

    proptest! {
        #[test]
        fn prop_phs_tpt_svf_parity_odd_symmetry_and_energy_dissipation(
            g_arr in prop::array::uniform4(1.0e-5_f32..32.0_f32),
            k_arr in prop::array::uniform4(0.0_f32..8.0_f32),
            k_nl_arr in prop::array::uniform4(0.0_f32..16.0_f32),
            s1_0 in prop::array::uniform4(-12.0_f32..12.0_f32),
            s2_0 in prop::array::uniform4(-12.0_f32..12.0_f32),
            u_l in prop::array::uniform4(-8.0_f32..8.0_f32),
            u_r in prop::array::uniform4(-8.0_f32..8.0_f32),
        ) {
            let g = GpuVec4::from_array(g_arr);
            let k = GpuVec4::from_array(k_arr);
            let k_nl = GpuVec4::from_array(k_nl_arr);

            // 1. k_nl == 0.0 matches passive step_tpt_svf_stereo8 at 0 ULP
            let coeffs0 = TptSvfCoeffs4::from_gk(g, k);
            let mut lin_s1 = InterleavedStereo8::from_lr(GpuVec4::from_array(s1_0), GpuVec4::from_array(s2_0));
            let mut lin_s2 = InterleavedStereo8::from_lr(GpuVec4::from_array(s2_0), GpuVec4::from_array(s1_0));
            let mut phs_s1 = lin_s1;
            let mut phs_s2 = lin_s2;
            let u8 = InterleavedStereo8::from_lr(GpuVec4::from_array(u_l), GpuVec4::from_array(u_r));
            let out_lin = step_tpt_svf_stereo8(u8, &mut lin_s1, &mut lin_s2, &coeffs0);
            let out_phs0 = step_tpt_svf_phs_stereo8(u8, &mut phs_s1, &mut phs_s2, g, k, GpuVec4::zero());
            prop_assert_eq!(out_phs0.lp.into_lr().0.to_array().map(f32::to_bits), out_lin.lp.into_lr().0.to_array().map(f32::to_bits));
            prop_assert_eq!(out_phs0.bp.into_lr().0.to_array().map(f32::to_bits), out_lin.bp.into_lr().0.to_array().map(f32::to_bits));
            prop_assert_eq!(out_phs0.hp.into_lr().1.to_array().map(f32::to_bits), out_lin.hp.into_lr().1.to_array().map(f32::to_bits));
            prop_assert_eq!(phs_s1.into_lr().1.to_array().map(f32::to_bits), lin_s1.into_lr().1.to_array().map(f32::to_bits));

            // 2. stereo8 == vec4 at 0 ULP and odd symmetry under (u, s) -> (-u, -s)
            let mut phs8_s1 = InterleavedStereo8::from_lr(GpuVec4::from_array(s1_0), GpuVec4::from_array(s2_0));
            let mut phs8_s2 = InterleavedStereo8::from_lr(GpuVec4::from_array(s2_0), GpuVec4::from_array(s1_0));
            let out8 = step_tpt_svf_phs_stereo8(u8, &mut phs8_s1, &mut phs8_s2, g, k, k_nl);
            let mut v_s1_l = GpuVec4::from_array(s1_0);
            let mut v_s2_l = GpuVec4::from_array(s2_0);
            let mut v_s1_r = GpuVec4::from_array(s2_0);
            let mut v_s2_r = GpuVec4::from_array(s1_0);
            let ov_l = step_tpt_svf_phs_vec4(GpuVec4::from_array(u_l), &mut v_s1_l, &mut v_s2_l, g, k, k_nl);
            let ov_r = step_tpt_svf_phs_vec4(GpuVec4::from_array(u_r), &mut v_s1_r, &mut v_s2_r, g, k, k_nl);
            prop_assert_eq!(out8.bp.into_lr().0.to_array().map(f32::to_bits), ov_l.bp.to_array().map(f32::to_bits));
            prop_assert_eq!(out8.ap.into_lr().1.to_array().map(f32::to_bits), ov_r.ap.to_array().map(f32::to_bits));

            let mut neg_s1 = GpuVec4::from_array(s1_0.map(|x| -x));
            let mut neg_s2 = GpuVec4::from_array(s2_0.map(|x| -x));
            let ov_neg = step_tpt_svf_phs_vec4(GpuVec4::from_array(u_l.map(|x| -x)), &mut neg_s1, &mut neg_s2, g, k, k_nl);
            for lane in 0..4 {
                prop_assert!((ov_neg.hp.to_array()[lane] + ov_l.hp.to_array()[lane]).abs() < 1.0e-4);
                prop_assert!((ov_neg.lp.to_array()[lane] + ov_l.lp.to_array()[lane]).abs() < 1.0e-4);
            }

            // 3. Exact zero-input discrete energy dissipation: s1'^2 + s2'^2 - (s1^2 + s2^2) == -4 * g * k_eff * v1^2 <= 0
            let mut z_s1 = GpuVec4::from_array(s1_0);
            let mut z_s2 = GpuVec4::from_array(s2_0);
            let oz = step_tpt_svf_phs_vec4(GpuVec4::zero(), &mut z_s1, &mut z_s2, g, k, k_nl);
            for lane in 0..4 {
                let s1_p = f64::from(s1_0[lane]);
                let s2_p = f64::from(s2_0[lane]);
                let s1_n = f64::from(z_s1.to_array()[lane]);
                let s2_n = f64::from(z_s2.to_array()[lane]);
                let e_pre = s1_p * s1_p + s2_p * s2_p;
                let e_post = s1_n * s1_n + s2_n * s2_n;
                let k_eff = f64::from(k_arr[lane]) + f64::from(k_nl_arr[lane]) * (s1_p * s1_p).min(64.0);
                let rhs = -4.0 * f64::from(g_arr[lane]) * k_eff * f64::from(oz.bp.to_array()[lane]).powi(2);
                prop_assert!(e_post <= e_pre + 1.0e-4 * e_pre.max(1.0));
                prop_assert!(((e_post - e_pre) - rhs).abs() <= 1.0e-3 * e_pre.max(1.0));
            }
        }
    }

    #[test]
    fn test_pred_corr_phs_zero_dc_knl0_parity_extremes_and_passivity() {
        let g = GpuVec4::from_array([0.05, 0.25, 1.5, MAX_SVF_G]);
        let k = GpuVec4::from_array([0.0, 0.5, 2.0, MAX_SVF_K]);
        let k_nl = GpuVec4::from_array([1.0, 4.0, 8.0, MAX_SVF_K_NL]);

        // Zero-DC +0.0 even under -0.0 / subnormal / NaN / Inf inputs
        let mut st_dc = TptSvfPhsPredCorrState::<InterleavedStereo8<GpuVec4>>::zero_stereo8();
        for &dirty in &[0.0_f32, -0.0, f32::from_bits(1), f32::NAN, f32::INFINITY] {
            let out = step_tpt_svf_pred_corr_phs_stereo8(
                InterleavedStereo8::splat(dirty),
                &mut st_dc,
                g,
                k,
                k_nl,
            );
            for v in [
                st_dc.s1,
                st_dc.s2,
                st_dc.s1_lin,
                st_dc.s2_lin,
                out.lp,
                out.bp,
            ] {
                let (vl, vr) = v.into_lr();
                assert_eq!(vl.to_array().map(f32::to_bits), [0; 4]);
                assert_eq!(vr.to_array().map(f32::to_bits), [0; 4]);
            }
        }

        // k_nl == 0.0 -> 0-ULP parity vs linear step_tpt_svf_stereo8 & folded3
        let coeffs = TptSvfCoeffs4::from_gk(g, k);
        let tone0 = SonaToneFoldedBlock0 {
            c_lp: GpuVec4::splat(0.4),
            c_bp: GpuVec4::splat(0.3),
            c_kbp: GpuVec4::splat(-0.2),
            c_u: GpuVec4::splat(0.15),
        };
        let tone3 = tone0.to_folded3(k);
        let mut st_pc = TptSvfPhsPredCorrState::<InterleavedStereo8<GpuVec4>>::zero_stereo8();
        let mut st_pc_f = TptSvfPhsPredCorrState::<InterleavedStereo8<GpuVec4>>::zero_stereo8();
        let mut lin_s1 = InterleavedStereo8::<GpuVec4>::zero();
        let mut lin_s2 = InterleavedStereo8::<GpuVec4>::zero();
        let mut fold_s1 = InterleavedStereo8::<GpuVec4>::zero();
        let mut fold_s2 = InterleavedStereo8::<GpuVec4>::zero();

        for n in 0..16 {
            let u = InterleavedStereo8::splat_lr(0.7 * (n as f32).sin(), -0.4 * (n as f32).cos());
            let o_pc = step_tpt_svf_pred_corr_phs_stereo8(u, &mut st_pc, g, k, GpuVec4::zero());
            let o_lin = step_tpt_svf_stereo8(u, &mut lin_s1, &mut lin_s2, &coeffs);
            assert_eq!(
                o_pc.lp.into_lr().0.to_array().map(f32::to_bits),
                o_lin.lp.into_lr().0.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_pc.bp.into_lr().1.to_array().map(f32::to_bits),
                o_lin.bp.into_lr().1.to_array().map(f32::to_bits)
            );

            let (tf_pc, bp_pc) = step_tpt_svf_pred_corr_phs_folded_stereo8(
                u,
                &mut st_pc_f,
                g,
                k,
                GpuVec4::zero(),
                &tone0,
            );
            let (tf_lin, bp_lin) = super::super::tone_folded::step_tpt_svf_folded3_stereo8(
                u,
                &mut fold_s1,
                &mut fold_s2,
                &coeffs,
                &tone3,
            );
            assert_eq!(
                tf_pc.into_lr().0.to_array().map(f32::to_bits),
                tf_lin.into_lr().0.to_array().map(f32::to_bits)
            );
            assert_eq!(
                bp_pc.into_lr().1.to_array().map(f32::to_bits),
                bp_lin.into_lr().1.to_array().map(f32::to_bits)
            );
        }

        // ||M_n||_2 <= 1 zero-input energy dissipation + +-1e9 finite safety
        let mut st_e = TptSvfPhsPredCorrState {
            s1_lin: GpuVec4::splat(8.0),
            s2_lin: GpuVec4::splat(-4.0),
            s1: GpuVec4::splat(8.0),
            s2: GpuVec4::splat(-4.0),
        };
        let mut prev_e = st_e.s1.0[0] * st_e.s1.0[0] + st_e.s2.0[0] * st_e.s2.0[0];
        for _ in 0..16 {
            let _ = step_tpt_svf_pred_corr_phs_vec4(GpuVec4::zero(), &mut st_e, g, k, k_nl);
            let e = st_e.s1.0[0] * st_e.s1.0[0] + st_e.s2.0[0] * st_e.s2.0[0];
            assert!(e <= prev_e + 1.0e-5);
            prev_e = e;
        }
        let ext = step_tpt_svf_pred_corr_phs_vec4(
            GpuVec4::from_array([1.0e9, -1.0e9, f32::NAN, f32::INFINITY]),
            &mut st_e,
            g,
            k,
            k_nl,
        );
        assert!(ext.lp.to_array().iter().all(|v| v.is_finite()));
        assert!(ext.bp.to_array().iter().all(|v| v.is_finite()));
    }
}
