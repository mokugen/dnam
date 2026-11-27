//! SONA v4.0 4-lane `CoreQuad` kernel (`TPT-SVF` -> smooth Selective-Duhem `ParaLRU` with
//! $C^2$ quartic `soft_rect` -> stereo link -> $C^\infty$ `g_pre(v^2)` -> clamped `AlgADAA1(4096, NR3)`
//! with odd-only $C^2$ wavefolder -> `lin_mix` -> post-ADAA rank-1 `SE-Bus` -> bounded `GLU` ->
//! `out_gain`, `ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.op-n.core.interblock-tiles]`).

use crate::simd::{KernelVec4, StereoVec8};

use super::dynamics::{
    clamp_range_safe_vec4, eval_bounded_glu_stereo8, eval_g_pre_stereo8, eval_se_bus_gain_stereo8,
};
use crate::sona_core::SonaBlockStateStereo8;
use crate::sona_core::paralru::{
    SonaParaLruDuhemCoeffs4, apply_stereo_link_stereo8, step_paralru_duhem_v4_stereo8,
};
use crate::sona_core::poly_adaa::eval_alg_adaa1_with_wavefolder_stereo8;
use crate::sona_core::tpt_svf::{
    SonaToneFoldedBlock0, TptSvfCoeffs4, TptSvfPhsPredCorrState, step_tpt_svf_folded3_stereo8,
    step_tpt_svf_pred_corr_phs_folded_stereo8,
};

/// Maximum clamped `adaa_drive` magnitude (`[-32.0, 32.0]`).
pub const MAX_ADAA_DRIVE: f32 = 32.0;

/// Maximum clamped `lru_bias_drive` magnitude (`[-16.0, 16.0]`).
pub const MAX_LRU_BIAS_DRIVE: f32 = 16.0;

/// Maximum clamped cross-lane `glu_mix` magnitude (`[-4.0, 4.0]`).
pub const MAX_GLU_MIX: f32 = 4.0;

/// Maximum clamped output gain `out_gain` magnitude (`[-4.0, 4.0]`).
pub const MAX_OUT_GAIN: f32 = 4.0;

/// Maximum `|e_tilde|` magnitude (`|out_gain| * (1 + |glu_mix|) = 4 * (1 + 4) = 20.0`).
pub const MAX_E_TILDE_ABS: f32 = MAX_OUT_GAIN * (1.0 + MAX_GLU_MIX);

const _: () = assert!(MAX_ADAA_DRIVE == 32.0);
const _: () = assert!(MAX_LRU_BIAS_DRIVE == 16.0);
const _: () = assert!(MAX_GLU_MIX == 4.0 && MAX_OUT_GAIN == 4.0);
const _: () = assert!(MAX_E_TILDE_ABS == 20.0);

/// Expands an explicit `u16` `rect_sign_mask` (`bit i == 1 -> -1.0`, `0 -> +1.0`) into 4 `KernelVec4`
/// polarity lanes without `-0.0` (`ref[dsp.op-n.polyadaa.clamp]`).
#[inline(always)]
#[must_use]
pub fn rect_sign_from_mask<V: KernelVec4>(mask: u16) -> V {
    V::from_array([
        if (mask & 1) != 0 { -1.0 } else { 1.0 },
        if (mask & 2) != 0 { -1.0 } else { 1.0 },
        if (mask & 4) != 0 { -1.0 } else { 1.0 },
        if (mask & 8) != 0 { -1.0 } else { 1.0 },
    ])
}

/// Complete parameter bundle for one SONA v4.0 4-lane stereo `CoreQuad`
/// (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.op-n.core.interblock-tiles]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoreQuadV4ParamsStereo4<V: KernelVec4> {
    pub svf: TptSvfCoeffs4<V>,
    /// Folded `TPT-SVF` tone basis (`c_kbp == 0.0` on Blocks `1..3`, non-zero on Block `0` PHS).
    pub tone: SonaToneFoldedBlock0<V>,
    pub lru: SonaParaLruDuhemCoeffs4<V>,
    /// Precomputed `2 * svf_g` (`[2e-5, 128.0]`) scaling `TPT-SVF` bandpass into Selective-Duhem flux.
    pub two_svf_g: V,
    /// Nonlinear diode-bandpass PHS damping `[0.0, 16.0]` on Block 0 (`0.0` on Blocks 1..3).
    pub svf_k_nl: V,
    /// Per-lane sign polarity (`+1.0` or `-1.0`, unpacked from explicit `u16` `rect_sign_mask`).
    pub rect_sign: V,
    /// Resting silence pre-ADAA gain `g0 in [0.0, 2.5]`.
    pub g0: V,
    /// High-drive asymptotic pre-ADAA gain `g_inf in [0.0, 2.5]`.
    pub g_inf: V,
    /// Pre-ADAA energy modulation sensitivity `d_pre >= 0.0`.
    pub d_pre: V,
    /// Post-ADAA `SE-Bus` non-negative lane pooling weights `se_w_sq >= 0.0`.
    pub se_w_sq: V,
    /// Post-ADAA `SE-Bus` gain reduction depth `se_d_bus >= 0.0`.
    pub se_d_bus: V,
    /// Linear pre-activation scale `[-32.0, 32.0]` for clamped `AlgADAA1(4096, NR3)`.
    pub adaa_drive: V,
    /// Bias scale `[-16.0, 16.0]` injecting `v_eff` into the `AlgADAA1` pre-activation.
    pub lru_bias_drive: V,
    /// Bounded linear bypass mix `lin_mix in [0.0, 1.0]` blending `A_odd_wf` with `g_pre * u_tone`.
    pub lin_mix: V,
    /// Blend `[0.0, 1.0]` between `AlgADAA1` odd output and $C^2$ wavefolder odd output.
    pub wavefolder_mix: V,
    /// Bounded cross-lane `GLU` mix `[-4.0, 4.0]` evaluated strictly on `g_se * A_odd_wf`.
    pub glu_mix: V,
    /// Post-`SE-Bus` / `GLU` output gain `[-4.0, 4.0]`.
    pub out_gain: V,
}

/// Features emitted by [`step_core_quad_v4_stereo8`] and [`step_core_quad_v4_phs_stereo8`]
/// (`ref[dsp.op-n.core.interblock-tiles]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoreQuadV4OutputStereo8<S: StereoVec8> {
    pub lru_state: S,
    pub o_tilde: S,
    pub e_tilde: S,
}

/// Block `0` SONA v4.0 recursive stereo state combining [`SonaBlockStateStereo8`] with the Pass A
/// linear predictor `(svf_lin_s1, svf_lin_s2)` for 1-step Predictor-Corrector Passive `PHS`
/// (`ref[dsp.op-n.core.tpt-tone-basis]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoreQuadV4Block0StateStereo8<S: StereoVec8> {
    pub block: SonaBlockStateStereo8<S>,
    pub svf_lin_s1: S,
    pub svf_lin_s2: S,
}

impl<S: StereoVec8> CoreQuadV4Block0StateStereo8<S> {
    /// Initializes Block `0` state at analytical Zero-DC resting equilibrium (`+0.0`).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            block: SonaBlockStateStereo8::zero(),
            svf_lin_s1: S::zero(),
            svf_lin_s2: S::zero(),
        }
    }
}

#[inline(always)]
fn step_core_quad_v4_post_svf_stereo8<S: StereoVec8>(
    tone_in: S,
    bp: S,
    state: &mut SonaBlockStateStereo8<S>,
    params: &CoreQuadV4ParamsStereo4<S::Half>,
) -> CoreQuadV4OutputStereo8<S> {
    let flux_inc = S::broadcast_half(params.two_svf_g).mul(bp);

    let rect_in = S::broadcast_half(params.rect_sign).mul(tone_in);
    let v_raw = step_paralru_duhem_v4_stereo8(rect_in, flux_inc, &mut state.lru_v, &params.lru);
    let v_eff = apply_stereo_link_stereo8(v_raw, params.lru.stereo_link);

    let g_pre = eval_g_pre_stereo8(v_eff, params.g0, params.g_inf, params.d_pre);
    let adaa_drive_c =
        clamp_range_safe_vec4(params.adaa_drive, -MAX_ADAA_DRIVE, MAX_ADAA_DRIVE, 0.0);
    let lru_bias_c = clamp_range_safe_vec4(
        params.lru_bias_drive,
        -MAX_LRU_BIAS_DRIVE,
        MAX_LRU_BIAS_DRIVE,
        0.0,
    );
    let r_curr = g_pre
        .mul(S::broadcast_half(adaa_drive_c))
        .mul(tone_in)
        .fma(S::broadcast_half(lru_bias_c), v_eff);

    let (a_odd_wf, a_even, r_curr_safe) =
        eval_alg_adaa1_with_wavefolder_stereo8(state.adaa_prev, r_curr, params.wavefolder_mix);
    state.adaa_prev = r_curr_safe;

    let g_se = eval_se_bus_gain_stereo8(v_eff, params.se_w_sq, params.se_d_bus);
    let lin_c = clamp_range_safe_vec4(params.lin_mix, 0.0, 1.0, 0.0);
    let out_c = clamp_range_safe_vec4(params.out_gain, -MAX_OUT_GAIN, MAX_OUT_GAIN, 0.0);

    let one_minus_lin = S::splat(1.0).sub(S::broadcast_half(lin_c));
    let o_pre = one_minus_lin
        .mul(a_odd_wf)
        .fma(S::broadcast_half(lin_c), g_pre.mul(tone_in));
    let o_tilde = S::broadcast_half(out_c)
        .mul(g_se)
        .mul(o_pre)
        .flush_recursive_state();
    let e_tilde = eval_bounded_glu_stereo8(a_odd_wf, a_even, g_se, lin_c, params.glu_mix, out_c);

    CoreQuadV4OutputStereo8 {
        lru_state: v_eff,
        o_tilde,
        e_tilde,
    }
}

/// Executes one Blocks `1..3` SONA v4.0 passive linear `CoreQuad` step using `3`-tap folded
/// `TPT-SVF` (`0 FDIV`, `0 FSQRT`, `0-ULP` Mechanism A2 sign-flip conjugation invariance).
#[inline(always)]
pub fn step_core_quad_v4_stereo8<S: StereoVec8>(
    u: S,
    state: &mut SonaBlockStateStereo8<S>,
    params: &CoreQuadV4ParamsStereo4<S::Half>,
) -> CoreQuadV4OutputStereo8<S> {
    let u_safe = u.flush_recursive_state();
    let tone_folded = params.tone.to_folded3(params.svf.k);
    let (tone_in, bp) = step_tpt_svf_folded3_stereo8(
        u_safe,
        &mut state.svf_s1,
        &mut state.svf_s2,
        &params.svf,
        &tone_folded,
    );
    step_core_quad_v4_post_svf_stereo8(tone_in, bp, state, params)
}

/// Executes one Block `0` SONA v4.0 1-step Predictor-Corrector Passive Port-Hamiltonian (`PHS`)
/// `CoreQuad` step with `0-ULP` parity against [`step_core_quad_v4_stereo8`] when
/// `params.svf_k_nl == 0.0`.
#[inline(always)]
pub fn step_core_quad_v4_phs_stereo8<S: StereoVec8>(
    u: S,
    state: &mut CoreQuadV4Block0StateStereo8<S>,
    params: &CoreQuadV4ParamsStereo4<S::Half>,
) -> CoreQuadV4OutputStereo8<S> {
    let u_safe = u.flush_recursive_state();
    let svf_g = params.two_svf_g.mul(S::Half::splat(0.5));
    let mut phs_state = TptSvfPhsPredCorrState {
        s1_lin: state.svf_lin_s1,
        s2_lin: state.svf_lin_s2,
        s1: state.block.svf_s1,
        s2: state.block.svf_s2,
    };
    let (tone_in, bp) = step_tpt_svf_pred_corr_phs_folded_stereo8(
        u_safe,
        &mut phs_state,
        svf_g,
        params.svf.k,
        params.svf_k_nl,
        &params.tone,
    );
    state.svf_lin_s1 = phs_state.s1_lin;
    state.svf_lin_s2 = phs_state.s2_lin;
    state.block.svf_s1 = phs_state.s1;
    state.block.svf_s2 = phs_state.s2;
    step_core_quad_v4_post_svf_stereo8(tone_in, bp, &mut state.block, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8, MAX_ALG_R};
    use crate::sona_core::SonaToneWeights4;
    use crate::sona_core::core_quad::couple_ms_input_stereo8;
    use crate::sona_core::paralru::SonaParaLruDuhemCoeffs4;
    use crate::sona_core::tpt_svf::prewarp_tan_pi_f_over_fs_f32;
    use proptest::prelude::*;

    fn sample_v4_params(g_val: f32) -> CoreQuadV4ParamsStereo4<GpuVec4> {
        let g = GpuVec4::splat(g_val);
        let k = GpuVec4::from_array([0.5, 0.707_106_77, 1.0, 1.4]);
        let weights = SonaToneWeights4 {
            lp: GpuVec4::from_array([0.4, 0.2, 0.1, 0.0]),
            bp: GpuVec4::from_array([0.3, 0.4, 0.2, 0.5]),
            hp: GpuVec4::from_array([0.0, 0.2, 0.3, 0.2]),
            ap: GpuVec4::from_array([0.1, 0.0, 0.2, 0.1]),
            dir: GpuVec4::from_array([0.2, 0.2, 0.2, 0.2]),
        };
        CoreQuadV4ParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneFoldedBlock0::from_weights(&weights),
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::from_array([0.98, 0.94, 0.88, 0.75]),
                GpuVec4::from_array([0.25, 0.4, 0.65, 0.8]),
                GpuVec4::from_array([0.1, 0.3, 0.6, 0.9]),
                GpuVec4::from_array([0.5, 2.0, 8.0, 16.0]),
                GpuVec4::from_array([0.0, 0.25, 0.5, 1.0]),
                GpuVec4::from_array([0.0, 0.3, 0.6, 1.0]),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::zero(),
            rect_sign: rect_sign_from_mask(0b1010),
            g0: GpuVec4::from_array([1.2, 0.4, 2.0, 0.8]),
            g_inf: GpuVec4::from_array([0.6, 1.8, 0.5, 2.2]),
            d_pre: GpuVec4::from_array([1.5, 3.0, 4.0, 0.5]),
            se_w_sq: GpuVec4::from_array([1.0, 0.5, 0.25, 0.125]),
            se_d_bus: GpuVec4::from_array([0.2, 0.8, 1.5, 2.5]),
            adaa_drive: GpuVec4::from_array([0.8, 2.0, 6.0, 16.0]),
            lru_bias_drive: GpuVec4::from_array([0.0, 0.4, -0.5, 0.7]),
            lin_mix: GpuVec4::from_array([0.0, 0.15, 0.5, 1.0]),
            wavefolder_mix: GpuVec4::from_array([0.0, 0.25, 0.75, 1.0]),
            glu_mix: GpuVec4::from_array([1.5, -2.0, 0.75, -1.25]),
            out_gain: GpuVec4::from_array([1.0, -1.5, 2.0, -3.0]),
        }
    }

    /// verify[dsp.op-n.commuted.dc-equilibrium]
    /// verify[dsp.op-n.polyadaa.linear-residual]
    /// verify[dsp.op-n.core.interblock-tiles]
    /// verify[arch.fp.range-saturation]
    #[test]
    fn test_core_quad_v4_zero_dc_sag_polarity_lin_mix_and_extremes() {
        let p = sample_v4_params(prewarp_tan_pi_f_over_fs_f32(0.08));
        assert_eq!(
            rect_sign_from_mask::<GpuVec4>(0b1010).to_array(),
            [1.0, -1.0, 1.0, -1.0]
        );

        // 1. Zero-DC canonical +0.0 across all outputs and states even for -0.0 inputs.
        let mut st_dc = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for step in 0..64 {
            let z = if (step & 1) == 0 { 0.0 } else { -0.0 };
            let out = step_core_quad_v4_stereo8(InterleavedStereo8::splat(z), &mut st_dc, &p);
            assert_eq!(
                out.lru_state.into_lr().0.to_array().map(f32::to_bits),
                [0; 4]
            );
            assert_eq!(
                out.lru_state.into_lr().1.to_array().map(f32::to_bits),
                [0; 4]
            );
            assert_eq!(out.o_tilde.into_lr().0.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(out.o_tilde.into_lr().1.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(out.e_tilde.into_lr().0.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(out.e_tilde.into_lr().1.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(
                st_dc.adaa_prev.into_lr().0.to_array().map(f32::to_bits),
                [0; 4]
            );
        }

        // 2. lin_mix == 1.0 zeroes e_tilde identically while keeping o_tilde active.
        let p_lin1 = CoreQuadV4ParamsStereo4 {
            lin_mix: GpuVec4::splat(1.0),
            ..p
        };
        let mut st_lin = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let out_lin = step_core_quad_v4_stereo8(
            InterleavedStereo8::splat_lr(0.75, -0.5),
            &mut st_lin,
            &p_lin1,
        );
        assert_eq!(
            out_lin.e_tilde.into_lr().0.to_array().map(f32::to_bits),
            [0; 4]
        );
        assert_eq!(
            out_lin.e_tilde.into_lr().1.to_array().map(f32::to_bits),
            [0; 4]
        );
        assert!(out_lin.o_tilde.into_lr().0.to_array()[0].abs() > 0.01);

        // 3. B+ sag polarity vs pre-only gain (g0=2.5, g_inf=0.5, d_pre=8.0 reduces high-drive gain).
        let mut p_sag = p;
        p_sag.g0 = GpuVec4::splat(2.5);
        p_sag.g_inf = GpuVec4::splat(0.5);
        p_sag.d_pre = GpuVec4::splat(8.0);
        p_sag.lin_mix = GpuVec4::splat(1.0);
        p_sag.se_d_bus = GpuVec4::zero();
        p_sag.out_gain = GpuVec4::splat(1.0);
        p_sag.tone = SonaToneFoldedBlock0::direct_only();
        let p_nosag = CoreQuadV4ParamsStereo4 {
            d_pre: GpuVec4::zero(),
            ..p_sag
        };
        let mut st_sag = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut st_nosag = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for _ in 0..8 {
            let o_s =
                step_core_quad_v4_stereo8(InterleavedStereo8::splat(2.0), &mut st_sag, &p_sag);
            let o_n =
                step_core_quad_v4_stereo8(InterleavedStereo8::splat(2.0), &mut st_nosag, &p_nosag);
            assert!(o_s.o_tilde.into_lr().0.to_array()[0] < o_n.o_tilde.into_lr().0.to_array()[0]);
        }

        // 4. ±1e9 / NaN / ±Inf boundedness: adaa_prev <= MAX_ALG_R, |e_tilde| <= 20.0, all finite.
        let mut st_ext = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for n in 0..8 {
            let x = if (n & 1) == 0 { 1.0e9 } else { -1.0e9 };
            let out_ext =
                step_core_quad_v4_stereo8(InterleavedStereo8::splat_lr(x, -x), &mut st_ext, &p);
            let (ol, or) = out_ext.o_tilde.into_lr();
            let (el, er) = out_ext.e_tilde.into_lr();
            let (pl, pr) = st_ext.adaa_prev.into_lr();
            for lane in 0..4 {
                assert!(ol.to_array()[lane].is_finite() && or.to_array()[lane].is_finite());
                assert!(el.to_array()[lane].abs() <= MAX_E_TILDE_ABS);
                assert!(er.to_array()[lane].abs() <= MAX_E_TILDE_ABS);
                assert!(pl.to_array()[lane].abs() <= MAX_ALG_R);
                assert!(pr.to_array()[lane].abs() <= MAX_ALG_R);
            }
        }

        // 5. High-drive (3996.0, 4096.0) and (-3996.0, -4096.0) regression: |e_tilde| <= MAX_E_TILDE_ABS (20.0).
        let mut p_edge = p;
        p_edge.tone = SonaToneFoldedBlock0::direct_only();
        p_edge.g0 = GpuVec4::splat(1.0);
        p_edge.g_inf = GpuVec4::splat(1.0);
        p_edge.d_pre = GpuVec4::zero();
        p_edge.adaa_drive = GpuVec4::splat(MAX_ADAA_DRIVE);
        p_edge.lru_bias_drive = GpuVec4::zero();
        p_edge.lin_mix = GpuVec4::zero();
        p_edge.wavefolder_mix = GpuVec4::zero();
        p_edge.se_d_bus = GpuVec4::zero();
        p_edge.glu_mix = GpuVec4::splat(MAX_GLU_MIX);
        p_edge.out_gain = GpuVec4::splat(MAX_OUT_GAIN);

        for (prev_r, u_prev, u_curr, out_sign) in [
            (3996.0_f32, 124.875_f32, 128.0_f32, 1.0_f32),
            (-3996.0_f32, -124.875_f32, -128.0_f32, -1.0_f32),
        ] {
            p_edge.out_gain = GpuVec4::splat(out_sign * MAX_OUT_GAIN);
            let mut st_direct = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            st_direct.adaa_prev = InterleavedStereo8::splat(prev_r);
            let out_direct = step_core_quad_v4_stereo8(
                InterleavedStereo8::splat(u_curr),
                &mut st_direct,
                &p_edge,
            );
            let (dl, dr) = out_direct.e_tilde.into_lr();
            assert!(dl.to_array().iter().all(|v| v.abs() <= MAX_E_TILDE_ABS));
            assert!(dr.to_array().iter().all(|v| v.abs() <= MAX_E_TILDE_ABS));

            let mut st_seq = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let _ =
                step_core_quad_v4_stereo8(InterleavedStereo8::splat(u_prev), &mut st_seq, &p_edge);
            let out_seq =
                step_core_quad_v4_stereo8(InterleavedStereo8::splat(u_curr), &mut st_seq, &p_edge);
            let (sl, sr) = out_seq.e_tilde.into_lr();
            let (ol, or) = out_seq.o_tilde.into_lr();
            assert!(sl.to_array().iter().all(|v| v.abs() <= MAX_E_TILDE_ABS));
            assert!(sr.to_array().iter().all(|v| v.abs() <= MAX_E_TILDE_ABS));
            assert!(ol.to_array().iter().all(|v| v.abs() <= MAX_OUT_GAIN));
            assert!(or.to_array().iter().all(|v| v.abs() <= MAX_OUT_GAIN));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// verify[dsp.op-n.core.matryoshka-blocks]
        /// verify[dsp.op-n.core.stereo-simd]
        #[test]
        fn prop_core_quad_v4_mechanism_a2_0ulp_and_stereo_parity(
            fc in 0.02_f32..0.35_f32,
            k_val in 0.35_f32..1.5_f32,
            svf_k_nl in prop_oneof![Just(0.0_f32), 0.1_f32..8.0_f32],
            g0 in 0.0_f32..2.5_f32,
            g_inf in 0.0_f32..2.5_f32,
            d_pre in 0.0_f32..8.0_f32,
            lin_mix in prop_oneof![Just(0.0_f32), 0.05_f32..0.95_f32, Just(1.0_f32)],
            wf_mix in prop_oneof![Just(0.0_f32), 0.05_f32..1.0_f32],
            glu_vals in prop::array::uniform4(-4.0_f32..4.0_f32),
            out_vals in prop::array::uniform4(-4.0_f32..4.0_f32),
            rect_mask in 0_u16..16_u16,
            conj_bits in [any::<bool>(); 4],
            stim_l in prop::collection::vec(prop_oneof![-4.0_f32..-1.0e-3_f32, 1.0e-3_f32..4.0_f32], 6),
            stim_r in prop::collection::vec(prop_oneof![-4.0_f32..-1.0e-3_f32, 1.0e-3_f32..4.0_f32], 6),
        ) {
            let g_s = prewarp_tan_pi_f_over_fs_f32(fc);
            let mut params = sample_v4_params(g_s);
            params.svf = TptSvfCoeffs4::from_gk(GpuVec4::splat(g_s), GpuVec4::splat(k_val));
            params.svf_k_nl = GpuVec4::splat(svf_k_nl);
            params.rect_sign = rect_sign_from_mask(rect_mask);
            params.g0 = GpuVec4::splat(g0);
            params.g_inf = GpuVec4::splat(g_inf);
            params.d_pre = GpuVec4::splat(d_pre);
            params.lin_mix = GpuVec4::splat(lin_mix);
            params.wavefolder_mix = GpuVec4::splat(wf_mix);
            params.glu_mix = GpuVec4::from_array(glu_vals);
            params.out_gain = GpuVec4::from_array(out_vals);

            let conj_arr = conj_bits.map(|b| if b { 1.0_f32 } else { -1.0_f32 });
            let conj_v = GpuVec4::from_array(conj_arr);
            let conj_rot = GpuVec4::from_array([conj_arr[1], conj_arr[2], conj_arr[3], conj_arr[0]]);
            let params_conj = CoreQuadV4ParamsStereo4 {
                rect_sign: params.rect_sign.mul(conj_v),
                lru_bias_drive: params.lru_bias_drive.mul(conj_v),
                glu_mix: params.glu_mix.mul(conj_v).mul(conj_rot),
                ..params
            };

            let mid_w = GpuVec4::splat(0.6);
            let side_w = GpuVec4::splat(-0.35);
            let mut st_o = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_c = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_l = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_r = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();

            for (&ul, &ur) in stim_l.iter().zip(stim_r.iter()) {
                let u_o = couple_ms_input_stereo8(InterleavedStereo8::zero(), ul, ur, mid_w, side_w);
                let u_c = couple_ms_input_stereo8(
                    InterleavedStereo8::zero(),
                    ul,
                    ur,
                    mid_w.mul(conj_v),
                    side_w.mul(conj_v),
                );
                let oo = step_core_quad_v4_phs_stereo8(u_o, &mut st_o, &params);
                let oc = step_core_quad_v4_phs_stereo8(u_c, &mut st_c, &params_conj);

                prop_assert_eq!(
                    oc.lru_state.into_lr().0.to_array().map(f32::to_bits),
                    oo.lru_state.into_lr().0.to_array().map(f32::to_bits)
                );
                prop_assert_eq!(
                    oc.e_tilde.into_lr().0.to_array().map(f32::to_bits),
                    oo.e_tilde.into_lr().0.to_array().map(f32::to_bits)
                );
                prop_assert_eq!(
                    oc.e_tilde.into_lr().1.to_array().map(f32::to_bits),
                    oo.e_tilde.into_lr().1.to_array().map(f32::to_bits)
                );

                let exp_ol = oo.o_tilde.into_lr().0.mul(conj_v).to_array();
                let got_ol = oc.o_tilde.into_lr().0.to_array();
                for lane in 0..4 {
                    if exp_ol[lane] != 0.0 || got_ol[lane] != 0.0 {
                        prop_assert_eq!(got_ol[lane].to_bits(), exp_ol[lane].to_bits());
                    }
                    prop_assert!(oo.e_tilde.into_lr().0.to_array()[lane].abs() <= MAX_E_TILDE_ABS);
                }

                // Dual-mono stereo parity + PHS(k_nl == 0) 0-ULP parity with linear CoreQuad v4.
                let u_mono = InterleavedStereo8::splat_lr(ul, ul);
                let om_phs = step_core_quad_v4_phs_stereo8(
                    u_mono,
                    &mut st_l,
                    &CoreQuadV4ParamsStereo4 {
                        svf_k_nl: GpuVec4::zero(),
                        ..params
                    },
                );
                let om_lin = step_core_quad_v4_stereo8(
                    u_mono,
                    &mut st_r,
                    &CoreQuadV4ParamsStereo4 {
                        svf_k_nl: GpuVec4::zero(),
                        ..params
                    },
                );
                prop_assert_eq!(
                    om_phs.o_tilde.into_lr().0.to_array().map(f32::to_bits),
                    om_phs.o_tilde.into_lr().1.to_array().map(f32::to_bits)
                );
                prop_assert_eq!(
                    om_phs.o_tilde.into_lr().0.to_array().map(f32::to_bits),
                    om_lin.o_tilde.into_lr().0.to_array().map(f32::to_bits)
                );
                prop_assert_eq!(
                    om_phs.e_tilde.into_lr().0.to_array().map(f32::to_bits),
                    om_lin.e_tilde.into_lr().0.to_array().map(f32::to_bits)
                );
            }
        }
    }
}
