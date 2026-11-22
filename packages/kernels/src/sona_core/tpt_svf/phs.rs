//! Block 0 frozen-state linearly-implicit Port-Hamiltonian (`PHS`) diode-bandpass `2x2` ZDF
//! `TPT-SVF` kernel (`ref[dsp.op-n.core.tpt-tone-basis]`).
//!
//! Uses `k_eff[n] = k + k_nl * min(s1[n-1]^2, E_max)` with `E_max = 64.0` and zero inner
//! Newton iterations, evaluating `a1 = rcp_ge1_nr2(1 + g * (g + k_eff))` with zero `FDIV`.

use crate::simd::{KernelVec4, StereoVec8, const_f32_pow2};

use super::{MIN_TPT_G, TptSvfOutputs};

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
}
