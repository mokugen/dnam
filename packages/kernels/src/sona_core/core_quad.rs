//! SONA v3.0 & v4.0 4-lane `CoreQuad` kernels (`TPT-SVF -> Selective-Duhem ParaLRU + Stereo Link ->
//! `g_pre` / Sag -> `AlgADAA1` / `PolyADAA1` + $C^2$ Wavefolder -> `SE-Bus` -> Bounded Cross-Lane `GLU`)
//! and M/S input coupling (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.op-n.core.interblock-tiles]`).

pub mod dynamics;
pub mod v4;

use crate::simd::{KernelVec4, StereoVec8};

pub use dynamics::{
    G_PRE_E_MAX, MAX_G_PRE_GAIN, SE_BUS_ABS_MAX, eval_bounded_glu_stereo8, eval_g_pre_stereo8,
    eval_g_pre_vec4, eval_se_bus_gain_stereo8, eval_se_bus_gain_vec4,
};
pub use v4::{
    CoreQuadV4Block0StateStereo8, CoreQuadV4OutputStereo8, CoreQuadV4ParamsStereo4, MAX_ADAA_DRIVE,
    MAX_E_TILDE_ABS, MAX_GLU_MIX, MAX_LRU_BIAS_DRIVE, MAX_OUT_GAIN, rect_sign_from_mask,
    step_core_quad_v4_phs_stereo8, step_core_quad_v4_stereo8,
};

use super::paralru::{
    SonaParaLruDuhemCoeffs4, apply_stereo_link_stereo8, step_paralru_duhem_stereo8,
};
use super::poly_adaa;
use super::tpt_svf::{TptSvfCoeffs4, step_tpt_svf_phs_stereo8, step_tpt_svf_stereo8};
use super::{SonaBlockOutputStereo8, SonaBlockStateStereo8, SonaToneWeights4, TptSvfOutputs};

/// Minimum positive power-supply sag gain floor protecting the `PolyADAA1` linear drive path.
pub const MIN_SAG_GAIN: f32 = 0.05;

const _: () = assert!(MIN_SAG_GAIN > 0.0 && MIN_SAG_GAIN < 1.0);

/// Complete parameter bundle for one SONA v3.0 4-lane stereo `CoreQuad`
/// (`TPT-SVF -> Selective-Duhem ParaLRU + Stereo Link -> PolyADAA1 + C^2 Wavefolder -> GLU`,
/// `ref[dsp.op-n.core.matryoshka-blocks]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoreQuadParamsStereo4<V: KernelVec4> {
    pub svf: TptSvfCoeffs4<V>,
    pub tone: SonaToneWeights4<V>,
    pub lru: SonaParaLruDuhemCoeffs4<V>,
    /// Precomputed `2 * svf_g` (`[2e-5, 128.0]`) scaling `TPT-SVF` bandpass into Selective-Duhem flux.
    pub two_svf_g: V,
    /// Nonlinear diode-bandpass PHS damping `[0.0, 16.0]` on Block 0 (`0.0` on Blocks 1..3).
    pub svf_k_nl: V,
    /// Per-lane sign polarity (`+1.0` or `-1.0`); `rect_in = rect_sign * tone_in`.
    pub rect_sign: V,
    /// Power-supply sag depth modulated by `v_eff`.
    pub sag_depth: V,
    /// Linear pre-activation scale `(2/3) * d` for the sagged `PolyADAA1` path.
    pub adaa_drive: V,
    /// Bias scale injecting `v_eff` into the `PolyADAA1` pre-activation.
    pub lru_bias_drive: V,
    /// Blend `[0, 1]` between `PolyADAA1` odd output and `C^2` wavefolder odd output.
    pub wavefolder_mix: V,
    /// Bounded cross-lane `GLU` mix (`[-4.0, 4.0]`) adding `glu_mix * (adaa_odd * rotl1(adaa_odd))`
    /// onto `adaa_even` after `wavefolder_mix`.
    pub glu_mix: V,
}

/// Couples voltage-normalized stereo Mid/Side input drive `(u_mid, u_side)` into an accumulated
/// inter-block `CoreQuad` input register (`ref[dsp.op-n.core.interblock-tiles]`).
///
/// Evaluates `u_acc_L + in_mid_weight * u_mid + in_side_weight * u_side` and
/// `u_acc_R + in_mid_weight * u_mid - in_side_weight * u_side` with `0-ULP` Mechanism A2
/// sign-flip conjugation invariance.
#[inline(always)]
#[must_use]
pub fn couple_ms_input_stereo8<S: StereoVec8>(
    u_acc: S,
    u_in_l: f32,
    u_in_r: f32,
    in_mid_weight: S::Half,
    in_side_weight: S::Half,
) -> S {
    let u_mid = S::Half::splat(0.5 * (u_in_l + u_in_r));
    let u_side = S::Half::splat(0.5 * (u_in_l - u_in_r));
    let (acc_l, acc_r) = u_acc.into_lr();
    let out_l = acc_l.fma(in_mid_weight, u_mid).fma(in_side_weight, u_side);
    let out_r = acc_r.fma(in_mid_weight, u_mid).fms(in_side_weight, u_side);
    S::from_lr(out_l, out_r)
}

#[inline(always)]
fn step_core_quad_post_svf_stereo8<S: StereoVec8>(
    u: S,
    svf_out: TptSvfOutputs<S>,
    state: &mut SonaBlockStateStereo8<S>,
    params: &CoreQuadParamsStereo4<S::Half>,
) -> SonaBlockOutputStereo8<S> {
    let tone_in = svf_out
        .lp
        .mul(S::broadcast_half(params.tone.lp))
        .fma(svf_out.bp, S::broadcast_half(params.tone.bp))
        .fma(svf_out.hp, S::broadcast_half(params.tone.hp))
        .fma(svf_out.ap, S::broadcast_half(params.tone.ap))
        .fma(u, S::broadcast_half(params.tone.dir));
    let flux_inc = S::broadcast_half(params.two_svf_g).mul(svf_out.bp);

    let rect_in = S::broadcast_half(params.rect_sign).mul(tone_in);
    let v_raw = step_paralru_duhem_stereo8(rect_in, flux_inc, &mut state.lru_v, &params.lru);
    let v_eff = apply_stereo_link_stereo8(v_raw, params.lru.stereo_link);

    let sag_gain = S::splat(1.0)
        .fms(S::broadcast_half(params.sag_depth), v_eff)
        .max(S::splat(MIN_SAG_GAIN));
    let r_curr = tone_in
        .mul(S::broadcast_half(params.adaa_drive))
        .mul(sag_gain)
        .fma(S::broadcast_half(params.lru_bias_drive), v_eff);

    let (adaa_odd, mut adaa_even, u_curr_safe) = poly_adaa::eval_poly_adaa1_with_wavefolder_stereo8(
        state.adaa_prev,
        r_curr,
        params.wavefolder_mix,
    );
    state.adaa_prev = u_curr_safe;

    let odd_rot = adaa_odd.rotate_left_1();
    let glu_term = adaa_odd.mul(odd_rot);
    adaa_even = adaa_even
        .fma(S::broadcast_half(params.glu_mix), glu_term)
        .add(S::zero());

    SonaBlockOutputStereo8 {
        lru_state: v_eff,
        adaa_odd,
        adaa_even,
    }
}

/// Executes one Blocks 1..3 passive linear `CoreQuad` step (`TPT-SVF -> tone -> rect_sign
/// Selective-Duhem ParaLRU -> stereo link -> power-supply sag -> bias-injected PolyADAA1 + optional
/// C^2 wavefolder -> bounded cross-lane GLU`) in `StereoVec8` registers (`0 FDIV`, `0 FSQRT`,
/// `0-ULP` Mechanism A2 sign-flip conjugation invariance).
#[inline(always)]
pub fn step_core_quad_stereo8<S: StereoVec8>(
    u: S,
    state: &mut SonaBlockStateStereo8<S>,
    params: &CoreQuadParamsStereo4<S::Half>,
) -> SonaBlockOutputStereo8<S> {
    let svf_out = step_tpt_svf_stereo8(u, &mut state.svf_s1, &mut state.svf_s2, &params.svf);
    step_core_quad_post_svf_stereo8(u, svf_out, state, params)
}

/// Executes one Block 0 linearly-implicit Port-Hamiltonian (`PHS`) `CoreQuad` step (`k_eff = k +
/// k_nl * min(s1^2, 64.0)`) in `StereoVec8` registers with `0-ULP` parity against
/// [`step_core_quad_stereo8`] when `params.svf_k_nl == 0.0` (`ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
pub fn step_core_quad_phs_stereo8<S: StereoVec8>(
    u: S,
    state: &mut SonaBlockStateStereo8<S>,
    params: &CoreQuadParamsStereo4<S::Half>,
) -> SonaBlockOutputStereo8<S> {
    let svf_g = params.two_svf_g.mul(S::Half::splat(0.5));
    let svf_out = step_tpt_svf_phs_stereo8(
        u,
        &mut state.svf_s1,
        &mut state.svf_s2,
        svf_g,
        params.svf.k,
        params.svf_k_nl,
    );
    step_core_quad_post_svf_stereo8(u, svf_out, state, params)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use crate::sona_core::paralru::{
        apply_stereo_link_vec4_pair, step_paralru_duhem_vec4, step_paralru_vec4,
    };
    use crate::sona_core::poly_adaa::{
        eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4,
    };
    use crate::sona_core::tpt_svf::{prewarp_tan_pi_f_over_fs_f32, step_tpt_svf_vec4};

    /// verify[dsp.router.mid-side-input]
    /// verify[dsp.router.stereo-link-detector]
    #[test]
    fn test_core_quad_zero_dc_ms_coupling_and_stereo_link_convergence() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.02),
            prewarp_tan_pi_f_over_fs_f32(0.08),
            prewarp_tan_pi_f_over_fs_f32(0.18),
            prewarp_tan_pi_f_over_fs_f32(0.32),
        ]);
        let k = GpuVec4::from_array([0.5, 0.707_106_77, 1.0, 1.4]);
        let params = CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::from_array([0.4, 0.2, 0.1, 0.0]),
                bp: GpuVec4::from_array([0.3, 0.4, 0.2, 0.5]),
                hp: GpuVec4::from_array([0.0, 0.2, 0.3, 0.2]),
                ap: GpuVec4::from_array([0.1, 0.0, 0.2, 0.1]),
                dir: GpuVec4::from_array([0.2, 0.2, 0.2, 0.2]),
            },
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::from_array([0.98, 0.94, 0.88, 0.75]),
                GpuVec4::from_array([0.25, 0.4, 0.65, 0.8]),
                GpuVec4::from_array([0.1, 0.3, 0.6, 0.9]),
                GpuVec4::from_array([0.5, 2.0, 8.0, 16.0]),
                GpuVec4::from_array([0.0, 0.3, 0.7, 1.0]),
                GpuVec4::splat(1.0),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::zero(),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: GpuVec4::from_array([0.0, 0.35, 0.65, 0.85]),
            adaa_drive: GpuVec4::from_array([0.8, 1.5, 2.5, 4.0]),
            lru_bias_drive: GpuVec4::from_array([0.0, 0.4, -0.5, 0.7]),
            wavefolder_mix: GpuVec4::from_array([0.0, 0.25, 0.75, 1.0]),
            glu_mix: GpuVec4::from_array([1.5, -2.0, 0.75, -1.25]),
        };

        // Zero-DC canonical +0.0 across all outputs and states even for -0.0 input and nonzero GLU.
        let mut st_dc = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for step in 0..64 {
            let z = if (step & 1) == 0 { 0.0 } else { -0.0 };
            let u = couple_ms_input_stereo8(
                InterleavedStereo8::splat(z),
                z,
                -z,
                GpuVec4::splat(0.7),
                GpuVec4::splat(-0.4),
            );
            let out = step_core_quad_stereo8(u, &mut st_dc, &params);
            let (ll, lr) = out.lru_state.into_lr();
            let (ol, or) = out.adaa_odd.into_lr();
            let (el, er) = out.adaa_even.into_lr();
            assert_eq!(ll.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(lr.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(ol.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(or.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(el.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(er.to_array().map(f32::to_bits), [0; 4]);
        }

        // lambda = 1.0 stereo-link image convergence: asymmetric L/R inputs converge v_eff_L ~= v_eff_R.
        let mut st_link = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for n in 0..32 {
            let u_l = ((n as f32) * 0.17).sin() * 2.0;
            let u_r = ((n as f32) * 0.29).cos() * -1.2;
            let u = couple_ms_input_stereo8(
                InterleavedStereo8::zero(),
                u_l,
                u_r,
                GpuVec4::splat(0.5),
                GpuVec4::splat(0.8),
            );
            let out = step_core_quad_stereo8(u, &mut st_link, &params);
            let (vl, vr) = out.lru_state.into_lr();
            for lane in 0..4 {
                assert!((vl.to_array()[lane] - vr.to_array()[lane]).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn test_core_quad_matches_vec4_zero_ulp_and_degenerates_cleanly() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.03),
            prewarp_tan_pi_f_over_fs_f32(0.09),
            prewarp_tan_pi_f_over_fs_f32(0.21),
            prewarp_tan_pi_f_over_fs_f32(0.36),
        ]);
        let k = GpuVec4::from_array([0.4, 0.7, 1.1, 1.6]);
        let svf = TptSvfCoeffs4::from_gk(g, k);
        let tone = SonaToneWeights4 {
            lp: GpuVec4::from_array([0.3, 0.2, 0.1, 0.0]),
            bp: GpuVec4::from_array([0.3, 0.4, 0.2, 0.5]),
            hp: GpuVec4::from_array([0.1, 0.2, 0.4, 0.2]),
            ap: GpuVec4::from_array([0.1, 0.0, 0.1, 0.1]),
            dir: GpuVec4::from_array([0.2, 0.2, 0.2, 0.2]),
        };

        let alpha_rel = GpuVec4::from_array([0.97, 0.92, 0.85, 0.72]);
        let alpha_atk = GpuVec4::from_array([0.2, 0.35, 0.6, 0.85]);
        let kappa = GpuVec4::from_array([0.1, 0.4, 0.7, 0.95]);
        let core_deg_p = CoreQuadParamsStereo4 {
            svf,
            tone,
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                alpha_rel,
                alpha_atk,
                kappa,
                GpuVec4::zero(),
                GpuVec4::zero(),
                GpuVec4::zero(),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::zero(),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: GpuVec4::from_array([0.0, 0.3, 0.6, 0.9]),
            adaa_drive: GpuVec4::from_array([0.7, 1.4, 2.2, 3.5]),
            lru_bias_drive: GpuVec4::from_array([0.1, -0.3, 0.5, -0.6]),
            wavefolder_mix: GpuVec4::from_array([0.0, 0.25, 0.75, 1.0]),
            glu_mix: GpuVec4::zero(),
        };

        // Zero Duhem, zero stereo link, zero GLU reduces to 1D ParaLRU + PolyADAA1 + C2 wavefolder at 0 ULP.
        let mut st_core_deg = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let (mut d_s1_l, mut d_s2_l, mut d_v_l, mut d_p_l) = (
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
        );
        let (mut d_s1_r, mut d_s2_r, mut d_v_r, mut d_p_r) = (
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
        );
        for n in 0..64 {
            let xl = ((n as f32) * 0.19).sin() * 2.5;
            let xr = ((n as f32) * 0.11).cos() * -1.8;
            let ul_v = GpuVec4::splat(xl);
            let ur_v = GpuVec4::splat(xr);
            let o_core = step_core_quad_stereo8(
                InterleavedStereo8::splat_lr(xl, xr),
                &mut st_core_deg,
                &core_deg_p,
            );

            let svf_l = step_tpt_svf_vec4(ul_v, &mut d_s1_l, &mut d_s2_l, &core_deg_p.svf);
            let svf_r = step_tpt_svf_vec4(ur_v, &mut d_s1_r, &mut d_s2_r, &core_deg_p.svf);
            let tone_l = svf_l
                .lp
                .mul(tone.lp)
                .fma(svf_l.bp, tone.bp)
                .fma(svf_l.hp, tone.hp)
                .fma(svf_l.ap, tone.ap)
                .fma(ul_v, tone.dir);
            let tone_r = svf_r
                .lp
                .mul(tone.lp)
                .fma(svf_r.bp, tone.bp)
                .fma(svf_r.hp, tone.hp)
                .fma(svf_r.ap, tone.ap)
                .fma(ur_v, tone.dir);
            let lru_l = step_paralru_vec4(
                core_deg_p.rect_sign.mul(tone_l),
                &mut d_v_l,
                alpha_rel,
                alpha_atk,
                kappa,
            );
            let lru_r = step_paralru_vec4(
                core_deg_p.rect_sign.mul(tone_r),
                &mut d_v_r,
                alpha_rel,
                alpha_atk,
                kappa,
            );
            let sag_l = GpuVec4::splat(1.0)
                .fms(core_deg_p.sag_depth, lru_l)
                .max(GpuVec4::splat(0.05));
            let sag_r = GpuVec4::splat(1.0)
                .fms(core_deg_p.sag_depth, lru_r)
                .max(GpuVec4::splat(0.05));
            let rl = tone_l
                .mul(core_deg_p.adaa_drive)
                .mul(sag_l)
                .fma(core_deg_p.lru_bias_drive, lru_l);
            let rr = tone_r
                .mul(core_deg_p.adaa_drive)
                .mul(sag_r)
                .fma(core_deg_p.lru_bias_drive, lru_r);
            let (mut odd_l, even_l, np_l) = eval_poly_adaa1_odd_even_vec4(d_p_l, rl);
            let (mut odd_r, even_r, np_r) = eval_poly_adaa1_odd_even_vec4(d_p_r, rr);
            let (wfo_l, _) = eval_c2_wavefolder_adaa_vec4(d_p_l, rl);
            let (wfo_r, _) = eval_c2_wavefolder_adaa_vec4(d_p_r, rr);
            let omw = GpuVec4::splat(1.0).sub(core_deg_p.wavefolder_mix);
            odd_l = odd_l
                .mul(omw)
                .fma(core_deg_p.wavefolder_mix, wfo_l)
                .add(GpuVec4::zero());
            odd_r = odd_r
                .mul(omw)
                .fma(core_deg_p.wavefolder_mix, wfo_r)
                .add(GpuVec4::zero());
            d_p_l = np_l;
            d_p_r = np_r;

            assert_eq!(
                o_core.lru_state.into_lr().0.to_array().map(f32::to_bits),
                lru_l.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_core.lru_state.into_lr().1.to_array().map(f32::to_bits),
                lru_r.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_core.adaa_odd.into_lr().0.to_array().map(f32::to_bits),
                odd_l.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_core.adaa_odd.into_lr().1.to_array().map(f32::to_bits),
                odd_r.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_core.adaa_even.into_lr().0.to_array().map(f32::to_bits),
                even_l.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_core.adaa_even.into_lr().1.to_array().map(f32::to_bits),
                even_r.to_array().map(f32::to_bits)
            );
        }

        // GpuVec4 vs InterleavedStereo8<GpuVec4> 0-ULP parity with active Duhem, stereo link, wavefolder, and GLU.
        let active_p = CoreQuadParamsStereo4 {
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                alpha_rel,
                alpha_atk,
                kappa,
                GpuVec4::from_array([0.8, 2.5, 6.0, 12.0]),
                GpuVec4::from_array([0.2, 0.5, 0.8, 1.0]),
                GpuVec4::from_array([0.1, 0.4, 0.75, 1.0]),
            ),
            glu_mix: GpuVec4::from_array([1.2, -0.9, 2.5, -1.7]),
            ..core_deg_p
        };
        let mut st8 = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let (mut s1_l, mut s2_l, mut v_l, mut p_l) = (
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
        );
        let (mut s1_r, mut s2_r, mut v_r, mut p_r) = (
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
        );
        for n in 0..64 {
            let ul = GpuVec4::splat_pair(
                ((n as f32) * 0.23).sin() * 2.1,
                ((n as f32) * 0.31).cos() * 1.4,
            );
            let ur = GpuVec4::splat_pair(
                ((n as f32) * 0.13).cos() * -1.7,
                ((n as f32) * 0.27).sin() * 1.9,
            );
            let out8 =
                step_core_quad_stereo8(InterleavedStereo8::from_lr(ul, ur), &mut st8, &active_p);

            let svf_l = step_tpt_svf_vec4(ul, &mut s1_l, &mut s2_l, &active_p.svf);
            let svf_r = step_tpt_svf_vec4(ur, &mut s1_r, &mut s2_r, &active_p.svf);
            let tone_l = svf_l
                .lp
                .mul(tone.lp)
                .fma(svf_l.bp, tone.bp)
                .fma(svf_l.hp, tone.hp)
                .fma(svf_l.ap, tone.ap)
                .fma(ul, tone.dir);
            let tone_r = svf_r
                .lp
                .mul(tone.lp)
                .fma(svf_r.bp, tone.bp)
                .fma(svf_r.hp, tone.hp)
                .fma(svf_r.ap, tone.ap)
                .fma(ur, tone.dir);
            let raw_l = step_paralru_duhem_vec4(
                active_p.rect_sign.mul(tone_l),
                active_p.two_svf_g.mul(svf_l.bp),
                &mut v_l,
                &active_p.lru,
            );
            let raw_r = step_paralru_duhem_vec4(
                active_p.rect_sign.mul(tone_r),
                active_p.two_svf_g.mul(svf_r.bp),
                &mut v_r,
                &active_p.lru,
            );
            let (eff_l, eff_r) =
                apply_stereo_link_vec4_pair(raw_l, raw_r, active_p.lru.stereo_link);
            let sag_l = GpuVec4::splat(1.0)
                .fms(active_p.sag_depth, eff_l)
                .max(GpuVec4::splat(0.05));
            let sag_r = GpuVec4::splat(1.0)
                .fms(active_p.sag_depth, eff_r)
                .max(GpuVec4::splat(0.05));
            let rl = tone_l
                .mul(active_p.adaa_drive)
                .mul(sag_l)
                .fma(active_p.lru_bias_drive, eff_l);
            let rr = tone_r
                .mul(active_p.adaa_drive)
                .mul(sag_r)
                .fma(active_p.lru_bias_drive, eff_r);
            let (mut odd_l, mut even_l, np_l) = eval_poly_adaa1_odd_even_vec4(p_l, rl);
            let (mut odd_r, mut even_r, np_r) = eval_poly_adaa1_odd_even_vec4(p_r, rr);
            let (wfo_l, _) = eval_c2_wavefolder_adaa_vec4(p_l, rl);
            let (wfo_r, _) = eval_c2_wavefolder_adaa_vec4(p_r, rr);
            let omw = GpuVec4::splat(1.0).sub(active_p.wavefolder_mix);
            odd_l = odd_l
                .mul(omw)
                .fma(active_p.wavefolder_mix, wfo_l)
                .add(GpuVec4::zero());
            odd_r = odd_r
                .mul(omw)
                .fma(active_p.wavefolder_mix, wfo_r)
                .add(GpuVec4::zero());
            p_l = np_l;
            p_r = np_r;
            let ol_a = odd_l.to_array();
            let or_a = odd_r.to_array();
            let rot_l = GpuVec4::from_array([ol_a[1], ol_a[2], ol_a[3], ol_a[0]]);
            let rot_r = GpuVec4::from_array([or_a[1], or_a[2], or_a[3], or_a[0]]);
            even_l = even_l
                .fma(active_p.glu_mix, odd_l.mul(rot_l))
                .add(GpuVec4::zero());
            even_r = even_r
                .fma(active_p.glu_mix, odd_r.mul(rot_r))
                .add(GpuVec4::zero());

            assert_eq!(
                out8.lru_state.into_lr().0.to_array().map(f32::to_bits),
                eff_l.to_array().map(f32::to_bits)
            );
            assert_eq!(
                out8.lru_state.into_lr().1.to_array().map(f32::to_bits),
                eff_r.to_array().map(f32::to_bits)
            );
            assert_eq!(
                out8.adaa_odd.into_lr().0.to_array().map(f32::to_bits),
                ol_a.map(f32::to_bits)
            );
            assert_eq!(
                out8.adaa_odd.into_lr().1.to_array().map(f32::to_bits),
                or_a.map(f32::to_bits)
            );
            assert_eq!(
                out8.adaa_even.into_lr().0.to_array().map(f32::to_bits),
                even_l.to_array().map(f32::to_bits)
            );
            assert_eq!(
                out8.adaa_even.into_lr().1.to_array().map(f32::to_bits),
                even_r.to_array().map(f32::to_bits)
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_core_quad_mechanism_a2_sign_flip_conjugation_zero_ulp(
            fc in 0.02_f32..0.35_f32,
            k_val in 0.35_f32..1.5_f32,
            alpha_rel in 0.5_f32..0.99_f32,
            alpha_atk in 0.05_f32..0.9_f32,
            kappa in 0.0_f32..1.0_f32,
            gamma_duhem in 0.0_f32..12.0_f32,
            signed_mix in 0.0_f32..1.0_f32,
            stereo_link in 0.0_f32..1.0_f32,
            sag_depth in 0.0_f32..0.9_f32,
            adaa_drive in 0.2_f32..4.0_f32,
            lru_bias_drive in -1.2_f32..1.2_f32,
            wf_mix in prop_oneof![Just(0.0_f32), 0.01_f32..1.0_f32],
            glu_vals in prop::array::uniform4(-3.5_f32..3.5_f32),
            rect_sign_bits in [any::<bool>(); 4],
            conj_sign_bits in [any::<bool>(); 4],
            stim_l in prop::collection::vec(prop_oneof![-2.5_f32..-1.0e-3_f32, 1.0e-3_f32..2.5_f32], 8),
            stim_r in prop::collection::vec(prop_oneof![-2.5_f32..-1.0e-3_f32, 1.0e-3_f32..2.5_f32], 8),
        ) {
            let g_s = prewarp_tan_pi_f_over_fs_f32(fc);
            let g = GpuVec4::splat(g_s);
            let k = GpuVec4::splat(k_val);
            let rect_arr = rect_sign_bits.map(|b| if b { 1.0_f32 } else { -1.0_f32 });
            let conj_arr = conj_sign_bits.map(|b| if b { 1.0_f32 } else { -1.0_f32 });
            let conj_v = GpuVec4::from_array(conj_arr);
            let conj_rot = GpuVec4::from_array([conj_arr[1], conj_arr[2], conj_arr[3], conj_arr[0]]);
            let mid_w = GpuVec4::splat(0.6);
            let side_w = GpuVec4::splat(-0.35);

            let params = CoreQuadParamsStereo4 {
                svf: TptSvfCoeffs4::from_gk(g, k),
                tone: SonaToneWeights4 {
                    lp: GpuVec4::splat(0.35),
                    bp: GpuVec4::splat(0.25),
                    hp: GpuVec4::splat(0.15),
                    ap: GpuVec4::splat(0.10),
                    dir: GpuVec4::splat(0.15),
                },
                lru: SonaParaLruDuhemCoeffs4::from_clamped(
                    GpuVec4::splat(alpha_rel),
                    GpuVec4::splat(alpha_atk),
                    GpuVec4::splat(kappa),
                    GpuVec4::splat(gamma_duhem),
                    GpuVec4::splat(signed_mix),
                    GpuVec4::splat(stereo_link),
                ),
                two_svf_g: GpuVec4::splat(2.0 * g_s),
                svf_k_nl: GpuVec4::zero(),
                rect_sign: GpuVec4::from_array(rect_arr),
                sag_depth: GpuVec4::splat(sag_depth),
                adaa_drive: GpuVec4::splat(adaa_drive),
                lru_bias_drive: GpuVec4::splat(lru_bias_drive),
                wavefolder_mix: GpuVec4::splat(wf_mix),
                glu_mix: GpuVec4::from_array(glu_vals),
            };
            let params_conj = CoreQuadParamsStereo4 {
                rect_sign: params.rect_sign.mul(conj_v),
                lru_bias_drive: params.lru_bias_drive.mul(conj_v),
                glu_mix: params.glu_mix.mul(conj_v).mul(conj_rot),
                ..params
            };

            let mut st_orig = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_conj = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            for (&ul, &ur) in stim_l.iter().zip(stim_r.iter()) {
                let u_orig = couple_ms_input_stereo8(InterleavedStereo8::zero(), ul, ur, mid_w, side_w);
                let u_conj = couple_ms_input_stereo8(
                    InterleavedStereo8::zero(),
                    ul,
                    ur,
                    mid_w.mul(conj_v),
                    side_w.mul(conj_v),
                );
                let out_orig = step_core_quad_stereo8(u_orig, &mut st_orig, &params);
                let out_conj = step_core_quad_stereo8(u_conj, &mut st_conj, &params_conj);

                let (ol_lru, or_lru) = out_orig.lru_state.into_lr();
                let (cl_lru, cr_lru) = out_conj.lru_state.into_lr();
                let (ol_odd, or_odd) = out_orig.adaa_odd.into_lr();
                let (cl_odd, cr_odd) = out_conj.adaa_odd.into_lr();
                let (ol_even, or_even) = out_orig.adaa_even.into_lr();
                let (cl_even, cr_even) = out_conj.adaa_even.into_lr();

                prop_assert_eq!(cl_lru.to_array().map(f32::to_bits), ol_lru.to_array().map(f32::to_bits));
                prop_assert_eq!(cr_lru.to_array().map(f32::to_bits), or_lru.to_array().map(f32::to_bits));
                prop_assert_eq!(cl_even.to_array().map(f32::to_bits), ol_even.to_array().map(f32::to_bits));
                prop_assert_eq!(cr_even.to_array().map(f32::to_bits), or_even.to_array().map(f32::to_bits));

                let exp_odd_l = ol_odd.mul(conj_v).to_array();
                let exp_odd_r = or_odd.mul(conj_v).to_array();
                let got_odd_l = cl_odd.to_array();
                let got_odd_r = cr_odd.to_array();
                for lane in 0..4 {
                    if exp_odd_l[lane] != 0.0 || got_odd_l[lane] != 0.0 {
                        prop_assert_eq!(got_odd_l[lane].to_bits(), exp_odd_l[lane].to_bits());
                    }
                    if exp_odd_r[lane] != 0.0 || got_odd_r[lane] != 0.0 {
                        prop_assert_eq!(got_odd_r[lane].to_bits(), exp_odd_r[lane].to_bits());
                    }
                }
            }

            // PHS Block 0: k_nl == 0.0 matches linear 0-ULP; k_nl > 0.0 preserves Mechanism A2 0-ULP conjugation!
            let params_phs = CoreQuadParamsStereo4 {
                svf_k_nl: GpuVec4::splat(6.0),
                ..params
            };
            let params_phs_conj = CoreQuadParamsStereo4 {
                svf_k_nl: GpuVec4::splat(6.0),
                ..params_conj
            };
            let mut st_p_orig = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_p_conj = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_p0 = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_lin0 = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            for (&ul, &ur) in stim_l.iter().zip(stim_r.iter()) {
                let u_o = couple_ms_input_stereo8(InterleavedStereo8::zero(), ul, ur, mid_w, side_w);
                let u_c = couple_ms_input_stereo8(
                    InterleavedStereo8::zero(),
                    ul,
                    ur,
                    mid_w.mul(conj_v),
                    side_w.mul(conj_v),
                );
                let o0_p = step_core_quad_phs_stereo8(u_o, &mut st_p0, &params);
                let o0_l = step_core_quad_stereo8(u_o, &mut st_lin0, &params);
                prop_assert_eq!(
                    o0_p.adaa_even.into_lr().0.to_array().map(f32::to_bits),
                    o0_l.adaa_even.into_lr().0.to_array().map(f32::to_bits)
                );

                let op_o = step_core_quad_phs_stereo8(u_o, &mut st_p_orig, &params_phs);
                let op_c = step_core_quad_phs_stereo8(u_c, &mut st_p_conj, &params_phs_conj);
                prop_assert_eq!(
                    op_c.adaa_even.into_lr().0.to_array().map(f32::to_bits),
                    op_o.adaa_even.into_lr().0.to_array().map(f32::to_bits)
                );
            }
        }
    }
}
