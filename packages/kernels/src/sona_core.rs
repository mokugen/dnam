//! Shared `#![no_std]` `SONA` state-space core kernels (`ref[dsp.op-n.core.matryoshka-blocks]`,
//! `ref[dsp.op-n.core.stereo-simd]`, `ref[dsp.op-n.commuted.dc-equilibrium]`,
//! `ref[sdk.rust.shared-nostd-kernels]`):
//! - [`tpt_svf`]: Unconditionally passive `2x2` ZDF `TPT-SVF` step, prewarping, and forward/adjoint scans.
//! - [`paralru`]: Contractive `1D` scalar `ParaLRU` step, zero-DC soft rectifier, exact forward tile, & `O(log2 N)` linear adjoint scan.
//! - [`poly_adaa`]: Pure-`f32` divided-difference `PolyADAA1` and $C^2$ wavefolder kernels.

pub mod core_quad;
pub mod op_n_pre;
pub mod os2x_hbfir;
pub mod paralru;
pub mod poly_adaa;
pub mod tpt_svf;

pub use core_quad::{
    CoreQuadParamsStereo4, CoreQuadV4Block0StateStereo8, CoreQuadV4OutputStereo8,
    CoreQuadV4ParamsStereo4, G_PRE_E_MAX, MAX_ADAA_DRIVE, MAX_E_TILDE_ABS, MAX_G_PRE_GAIN,
    MAX_GLU_MIX, MAX_LRU_BIAS_DRIVE, MAX_OUT_GAIN, MIN_SAG_GAIN, SE_BUS_ABS_MAX,
    couple_ms_input_stereo8, eval_bounded_glu_stereo8, eval_g_pre_stereo8, eval_g_pre_vec4,
    eval_se_bus_gain_stereo8, eval_se_bus_gain_vec4, rect_sign_from_mask,
    step_core_quad_phs_stereo8, step_core_quad_stereo8, step_core_quad_v4_phs_stereo8,
    step_core_quad_v4_stereo8,
};
pub use op_n_pre::{
    OP_N_PRE_BETA_BYPASS_EPS, OP_N_PRE_FULL_PASS_THRESH, OP_N_PRE_INV_TAU_F32, OP_N_PRE_MAX_BETA_Q,
    OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OP_N_PRE_MIN_DEADZONE_WIDTH, OP_N_PRE_MIN_RHO_EFF,
    OP_N_PRE_QUANT_CLAMP_F32, OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH, OpNPreHoisted, OpNPreHotState,
    OpNPreV4Hoisted, OpNPreV4HotState, compute_op_n_pre_inc_q32, compute_op_n_pre_v4_inc_q32,
    quantize_hermite_v4_f32, quantize_nearest_step_f32, quantize_smooth_fourier_f32,
};
pub use os2x_hbfir::{
    HB15_DELAY_SAMPLES_1X, HB15_HIST_LEN, HB15_ODD_DELAY_LEN, HbFir15Downsampler4Ch,
    HbFir15UpsamplerStereo, eval_hb15_downsample_4ch_vec4, eval_hb15_downsample_stereo,
    eval_hb15_upsample_stereo,
};

use crate::simd::{KernelVec4, MAX_PARALRU_R_F32, StereoVec8};

pub use paralru::{
    LinearScanElem, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_TWO_SVF_G_F32, ONE_THIRD_F32,
    ParaLruDuhemAdjointTrace, ParaLruDuhemForwardOut, ParaLruDuhemParams, ParaLruDuhemTileGrads,
    ParaLruParams, ParaLruTileGrads, SOFT_RECT_EPS, SOFT_RECT_INV_TWO_EPS, SOFT_RECT_TWO_EPS,
    SonaParaLruDuhemCoeffs4, apply_linear_scan, apply_stereo_link_stereo8,
    apply_stereo_link_vec4_pair, compose_linear_scan, compose_paralru_adjoint_scan,
    paralru_adjoint_step_elem, paralru_duhem_adjoint_step_elem, paralru_duhem_v4_adjoint_step_elem,
    run_paralru_adjoint_scan_partial_tile, run_paralru_adjoint_scan_tile,
    run_paralru_duhem_adjoint_scan_partial_tile, run_paralru_duhem_adjoint_scan_tile,
    run_paralru_duhem_forward_partial_tile, run_paralru_duhem_forward_tile,
    run_paralru_duhem_v4_adjoint_scan_partial_tile, run_paralru_duhem_v4_adjoint_scan_tile,
    run_paralru_duhem_v4_forward_partial_tile, run_paralru_duhem_v4_forward_tile,
    run_paralru_forward_partial_tile, run_paralru_forward_tile, scan_linear_prefix_tile,
    soft_rect_quartic_f32, soft_rect_quartic_stereo8, soft_rect_quartic_vec4,
    soft_rect_zero_dc_f32, soft_rect_zero_dc_stereo8, soft_rect_zero_dc_vec4,
    step_paralru_duhem_stereo8, step_paralru_duhem_v4_stereo8, step_paralru_duhem_v4_vec4,
    step_paralru_duhem_vec4, step_paralru_stereo8, step_paralru_v4_stereo8, step_paralru_v4_vec4,
    step_paralru_vec4,
};
pub use poly_adaa::{
    AlgAdaa1BackwardOut, MAX_ADAA_ENDPOINT, MAX_ALG_R, QuadratureHysteresis2D, TINY_CROSSING_EPS,
    eval_alg_adaa1_backward_f32, eval_alg_adaa1_f32, eval_alg_adaa1_stereo8, eval_alg_adaa1_vec4,
    eval_alg_adaa1_with_wavefolder_stereo8, eval_alg_adaa1_with_wavefolder_vec4,
    eval_c2_wavefolder_adaa_stereo8, eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4,
    eval_poly_adaa1_quad_batched, eval_poly_adaa1_stereo8, eval_poly_adaa1_with_wavefolder_stereo8,
    eval_poly_adaa1_with_wavefolder_vec4, scrub_clamped_alg_endpoint_f32,
    scrub_clamped_alg_endpoint_stereo8, scrub_clamped_alg_endpoint_vec4,
};
pub use tpt_svf::{
    MAX_SVF_K_NL, MIN_TPT_G, PhsPredCorrAdjointTerms, PhsPredCorrPassBCoeffs,
    PhsPredCorrStepContext, SVF_PHS_E_MAX, SonaToneFolded3, SonaToneFoldedBlock0, TptAdjointElem,
    TptScanElem, TptSvfCoeffs4, TptSvfOutputs, TptSvfPhsPredCorrState, apply_tpt_adjoint,
    apply_tpt_init, compose_tpt_adjoint_scan, compose_tpt_scan, eval_phs_pred_corr_pass_b_coeffs,
    eval_phs_pred_corr_step_adjoint, phs_pred_corr_pass_a_adjoint_elem,
    phs_pred_corr_pass_a_scan_elem, phs_pred_corr_pass_b_adjoint_elem,
    phs_pred_corr_pass_b_scan_elem, prewarp_tan_pi_f_over_fs_f32, step_tpt_svf_folded3_stereo8,
    step_tpt_svf_folded3_vec4, step_tpt_svf_phs_stereo8, step_tpt_svf_phs_vec4,
    step_tpt_svf_pred_corr_phs_folded_stereo8, step_tpt_svf_pred_corr_phs_stereo8,
    step_tpt_svf_pred_corr_phs_vec4, step_tpt_svf_stereo8, step_tpt_svf_vec4,
};

/// Simultaneous 4-mode `TPT-SVF` + direct-feed tone mixing weights (`ref[dsp.op-n.core.tpt-tone-basis]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaToneWeights4<V: KernelVec4> {
    pub lp: V,
    pub bp: V,
    pub hp: V,
    pub ap: V,
    pub dir: V,
}

impl<V: KernelVec4> SonaToneWeights4<V> {
    /// Selects pure low-pass output (`lp = 1.0`, remaining taps `0.0`).
    #[inline]
    #[must_use]
    pub fn lp_only() -> Self {
        Self {
            lp: V::splat(1.0),
            bp: V::zero(),
            hp: V::zero(),
            ap: V::zero(),
            dir: V::zero(),
        }
    }

    /// Selects unity direct dry bypass (`dir = 1.0`, SVF taps `0.0`).
    #[inline]
    #[must_use]
    pub fn direct_only() -> Self {
        Self {
            lp: V::zero(),
            bp: V::zero(),
            hp: V::zero(),
            ap: V::zero(),
            dir: V::splat(1.0),
        }
    }
}

/// Vectorized `1D` scalar `ParaLRU` ballistics for 4 SIMD lanes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaParaLruCoeffs4<V: KernelVec4> {
    pub alpha_rel: V,
    pub alpha_atk: V,
    pub kappa: V,
}

impl<V: KernelVec4> SonaParaLruCoeffs4<V> {
    /// Clamps 4-lane `ParaLRU` coefficients into the unconditionally contractive domain
    /// (`alpha_rel in [0, MAX_PARALRU_R_F32]`, `alpha_atk, kappa in [0, 1]`).
    #[inline]
    #[must_use]
    pub fn from_clamped(alpha_rel: V, alpha_atk: V, kappa: V) -> Self {
        let zero = V::zero();
        let one = V::splat(1.0);
        Self {
            alpha_rel: alpha_rel.clamp(zero, V::splat(MAX_PARALRU_R_F32)),
            alpha_atk: alpha_atk.clamp(zero, one),
            kappa: kappa.clamp(zero, one),
        }
    }
}

/// Complete parameter bundle for one 4-lane stereo `SONA` core group (`TPT-SVF -> ParaLRU -> PolyADAA1`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaBlockParamsStereo4<V: KernelVec4> {
    pub svf: TptSvfCoeffs4<V>,
    pub tone: SonaToneWeights4<V>,
    pub lru: SonaParaLruCoeffs4<V>,
    /// Pre-activation linear scale `(2/3) * d` folded at the `3 kHz` control boundary.
    pub adaa_drive: V,
}

/// Recursive stereo state for one 4-lane `SONA` core group (`S: StereoVec8`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaBlockStateStereo8<S: StereoVec8> {
    pub svf_s1: S,
    pub svf_s2: S,
    pub lru_v: S,
    pub adaa_prev: S,
}

impl<S: StereoVec8> SonaBlockStateStereo8<S> {
    /// Initializes the block state at analytical zero-DC resting equilibrium (`+0.0f32`).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            svf_s1: S::zero(),
            svf_s2: S::zero(),
            lru_v: S::zero(),
            adaa_prev: S::zero(),
        }
    }
}

/// Features emitted by [`step_sona_block_stereo8`] for direct-register `VarPro` readouts and
/// downstream Matryoshka blocks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SonaBlockOutputStereo8<S: StereoVec8> {
    pub lru_state: S,
    pub adaa_odd: S,
    pub adaa_even: S,
}

/// Executes one stereo `4`-lane `SONA` core block step (`TPT-SVF -> ParaLRU -> PolyADAA1`) in
/// `StereoVec8` registers (`0 FDIV`, `0 FSQRT`, `0-ULP` cross-channel symmetry).
#[inline(always)]
pub fn step_sona_block_stereo8<S: StereoVec8>(
    u: S,
    state: &mut SonaBlockStateStereo8<S>,
    params: &SonaBlockParamsStereo4<S::Half>,
) -> SonaBlockOutputStereo8<S> {
    let svf_out = step_tpt_svf_stereo8(u, &mut state.svf_s1, &mut state.svf_s2, &params.svf);
    let tone_in = svf_out
        .lp
        .mul(S::broadcast_half(params.tone.lp))
        .fma(svf_out.bp, S::broadcast_half(params.tone.bp))
        .fma(svf_out.hp, S::broadcast_half(params.tone.hp))
        .fma(svf_out.ap, S::broadcast_half(params.tone.ap))
        .fma(u, S::broadcast_half(params.tone.dir));

    let lru_state = step_paralru_stereo8(
        tone_in,
        &mut state.lru_v,
        params.lru.alpha_rel,
        params.lru.alpha_atk,
        params.lru.kappa,
    );

    let r_curr = lru_state.mul(S::broadcast_half(params.adaa_drive));
    let (adaa_odd, adaa_even) = eval_poly_adaa1_stereo8(state.adaa_prev, r_curr);
    state.adaa_prev = poly_adaa::scrub_endpoint_stereo8(r_curr);

    SonaBlockOutputStereo8 {
        lru_state,
        adaa_odd,
        adaa_even,
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};

    /// verify[dsp.op-n.core.tpt-tone-basis]
    /// verify[dsp.op-n.polyadaa.drive-bank]
    #[test]
    fn test_sona_block_zero_dc_equilibrium_and_dual_mono_parity() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.01),
            prewarp_tan_pi_f_over_fs_f32(0.05),
            prewarp_tan_pi_f_over_fs_f32(0.15),
            prewarp_tan_pi_f_over_fs_f32(0.35),
        ]);
        let k = GpuVec4::from_array([
            0.5,
            core::f32::consts::FRAC_1_SQRT_2,
            1.0,
            core::f32::consts::SQRT_2,
        ]);
        let params = SonaBlockParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4::lp_only(),
            lru: SonaParaLruCoeffs4::from_clamped(
                GpuVec4::from_array([0.99, 0.95, 0.9, 1.5]),
                GpuVec4::from_array([0.2, 0.5, 0.8, -0.1]),
                GpuVec4::from_array([0.0, 0.5, 1.0, 1.2]),
            ),
            adaa_drive: GpuVec4::from_array([0.5, 1.0, 2.5, 6.0]),
        };

        let mut state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for _ in 0..1_000 {
            let out = step_sona_block_stereo8(InterleavedStereo8::zero(), &mut state, &params);
            let (lru_l, lru_r) = out.lru_state.into_lr();
            let (odd_l, odd_r) = out.adaa_odd.into_lr();
            let (even_l, even_r) = out.adaa_even.into_lr();
            assert_eq!(lru_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(lru_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(odd_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(odd_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even_r.to_array().map(f32::to_bits), [0; 4]);
        }

        // Verify direct_only tone weights and dual-mono 0-ULP parity.
        let mut mono_params = params;
        mono_params.tone = SonaToneWeights4::direct_only();
        let mut st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for n in 0..64 {
            let sample = ((n as f32) * 0.23).sin() * 1.5;
            let u = InterleavedStereo8::splat_lr(sample, sample);
            let out = step_sona_block_stereo8(u, &mut st, &mono_params);
            let (ol, or) = out.adaa_odd.into_lr();
            let (el, er) = out.adaa_even.into_lr();
            assert_eq!(ol.to_array(), or.to_array());
            assert_eq!(el.to_array(), er.to_array());
        }
    }

    #[test]
    fn test_sona_block_stereo8_matches_vec4_and_bibo_recovery() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(80.0 / 48_000.0),
            prewarp_tan_pi_f_over_fs_f32(500.0 / 48_000.0),
            prewarp_tan_pi_f_over_fs_f32(2_400.0 / 48_000.0),
            prewarp_tan_pi_f_over_fs_f32(8_000.0 / 48_000.0),
        ]);
        let k = GpuVec4::from_array([1.0 / 12.0, 0.25, 0.707_106_77, 1.2]);
        let params = SonaBlockParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::splat(0.4),
                bp: GpuVec4::splat(0.3),
                hp: GpuVec4::splat(-0.2),
                ap: GpuVec4::splat(0.1),
                dir: GpuVec4::splat(0.25),
            },
            lru: SonaParaLruCoeffs4::from_clamped(
                GpuVec4::from_array([0.98, 0.96, 0.94, 0.9]),
                GpuVec4::from_array([0.1, 0.35, 0.65, 0.85]),
                GpuVec4::from_array([0.2, 0.5, 0.8, 1.0]),
            ),
            adaa_drive: GpuVec4::from_array([0.4, 1.0, 3.0, 8.0]),
        };

        let mut st_state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut s1_l = GpuVec4::zero();
        let mut s2_l = GpuVec4::zero();
        let mut v_l = GpuVec4::zero();
        let mut prev_l = GpuVec4::zero();
        let mut s1_r = GpuVec4::zero();
        let mut s2_r = GpuVec4::zero();
        let mut v_r = GpuVec4::zero();
        let mut prev_r = GpuVec4::zero();

        // Burst of +/-100.0 asymmetric stereo audio (256 frames) + 256 zero frames verifying 0-ULP parity.
        for n in 0..512 {
            let (xl, xr) = if n < 256 {
                let t = n as f32;
                (100.0 * (0.11 * t).sin(), -75.0 * (0.19 * t).cos())
            } else {
                (0.0, 0.0)
            };
            let ul = GpuVec4::splat(xl);
            let ur = GpuVec4::splat(xr);

            let out_st = step_sona_block_stereo8(
                InterleavedStereo8::from_lr(ul, ur),
                &mut st_state,
                &params,
            );

            let svf_l = step_tpt_svf_vec4(ul, &mut s1_l, &mut s2_l, &params.svf);
            let tone_l = svf_l
                .lp
                .mul(params.tone.lp)
                .fma(svf_l.bp, params.tone.bp)
                .fma(svf_l.hp, params.tone.hp)
                .fma(svf_l.ap, params.tone.ap)
                .fma(ul, params.tone.dir);
            let lru_l = step_paralru_vec4(
                tone_l,
                &mut v_l,
                params.lru.alpha_rel,
                params.lru.alpha_atk,
                params.lru.kappa,
            );
            let r_l = lru_l.mul(params.adaa_drive);
            let (odd_l, even_l, next_prev_l) = eval_poly_adaa1_odd_even_vec4(prev_l, r_l);
            prev_l = next_prev_l;

            let svf_r = step_tpt_svf_vec4(ur, &mut s1_r, &mut s2_r, &params.svf);
            let tone_r = svf_r
                .lp
                .mul(params.tone.lp)
                .fma(svf_r.bp, params.tone.bp)
                .fma(svf_r.hp, params.tone.hp)
                .fma(svf_r.ap, params.tone.ap)
                .fma(ur, params.tone.dir);
            let lru_r = step_paralru_vec4(
                tone_r,
                &mut v_r,
                params.lru.alpha_rel,
                params.lru.alpha_atk,
                params.lru.kappa,
            );
            let r_r = lru_r.mul(params.adaa_drive);
            let (odd_r, even_r, next_prev_r) = eval_poly_adaa1_odd_even_vec4(prev_r, r_r);
            prev_r = next_prev_r;

            let (st_lru_l, st_lru_r) = out_st.lru_state.into_lr();
            let (st_odd_l, st_odd_r) = out_st.adaa_odd.into_lr();
            let (st_even_l, st_even_r) = out_st.adaa_even.into_lr();
            assert_eq!(st_lru_l.to_array(), lru_l.to_array());
            assert_eq!(st_lru_r.to_array(), lru_r.to_array());
            assert_eq!(st_odd_l.to_array(), odd_l.to_array());
            assert_eq!(st_odd_r.to_array(), odd_r.to_array());
            assert_eq!(st_even_l.to_array(), even_l.to_array());
            assert_eq!(st_even_r.to_array(), even_r.to_array());

            for lane in 0..4 {
                assert!(odd_l.to_array()[lane].abs() <= 1.0 + 1.0e-5);
                assert!(odd_r.to_array()[lane].abs() <= 1.0 + 1.0e-5);
                assert!(even_l.to_array()[lane].abs() <= 0.3);
                assert!(even_r.to_array()[lane].abs() <= 0.3);
            }
        }

        // Step 32,768 additional zero frames (> 14 decay time constants of 80 Hz Q=12 at 48 kHz).
        for _ in 0..32_768 {
            let _ = step_sona_block_stereo8(InterleavedStereo8::zero(), &mut st_state, &params);
        }

        // All recursive states across all 4 lanes decay back below 1e-3.
        let (fin_svf1_l, fin_svf1_r) = st_state.svf_s1.into_lr();
        let (fin_lru_l, fin_lru_r) = st_state.lru_v.into_lr();
        for lane in 0..4 {
            assert!(fin_svf1_l.to_array()[lane].abs() < 1.0e-3);
            assert!(fin_svf1_r.to_array()[lane].abs() < 1.0e-3);
            assert!(fin_lru_l.to_array()[lane].abs() < 1.0e-3);
            assert!(fin_lru_r.to_array()[lane].abs() < 1.0e-3);
        }
    }

    #[test]
    fn test_core_quad_zero_dc_wavefolder_and_bibo() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.02),
            prewarp_tan_pi_f_over_fs_f32(0.08),
            prewarp_tan_pi_f_over_fs_f32(0.18),
            prewarp_tan_pi_f_over_fs_f32(0.32),
        ]);
        let k = GpuVec4::from_array([0.5, 0.707_106_77, 1.0, 1.4]);
        let base_params = CoreQuadParamsStereo4 {
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
                GpuVec4::from_array([0.5, 1.5, 4.0, 8.0]),
                GpuVec4::from_array([0.0, 0.25, 0.5, 1.0]),
                GpuVec4::from_array([0.0, 0.3, 0.6, 1.0]),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::zero(),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: GpuVec4::from_array([0.0, 0.35, 0.65, 0.85]),
            adaa_drive: GpuVec4::from_array([0.8, 1.5, 2.5, 4.0]),
            lru_bias_drive: GpuVec4::from_array([0.0, 0.4, -0.5, 0.7]),
            wavefolder_mix: GpuVec4::zero(),
            glu_mix: GpuVec4::zero(),
        };

        // Zero-DC resting equilibrium (`+0.0` bits across all channels and lanes, even for `-0.0` input).
        let mut zero_state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for step in 0..128 {
            let z_in = if (step & 1) == 0 { 0.0 } else { -0.0 };
            let out = step_core_quad_stereo8(
                InterleavedStereo8::splat(z_in),
                &mut zero_state,
                &base_params,
            );
            let (lru_l, lru_r) = out.lru_state.into_lr();
            let (odd_l, odd_r) = out.adaa_odd.into_lr();
            let (even_l, even_r) = out.adaa_even.into_lr();
            let (prev_l, prev_r) = zero_state.adaa_prev.into_lr();
            assert_eq!(lru_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(lru_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(odd_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(odd_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(even_r.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(prev_l.to_array().map(f32::to_bits), [0; 4]);
            assert_eq!(prev_r.to_array().map(f32::to_bits), [0; 4]);
        }

        // Verify wavefolder blend alters `adaa_odd` while leaving lane 0 (`mix == 0`) `0-ULP` identical.
        let mut wf_params = base_params;
        wf_params.wavefolder_mix = GpuVec4::from_array([0.0, 0.25, 0.75, 1.0]);
        let mut st_plain = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut st_wf = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut max_lane3_diff = 0.0_f32;
        for n in 0..64 {
            let ul = ((n as f32) * 0.21).sin() * 1.4;
            let ur = ((n as f32) * 0.13).cos() * 1.1;
            let u = InterleavedStereo8::splat_lr(ul, ur);
            let o_plain = step_core_quad_stereo8(u, &mut st_plain, &base_params);
            let o_wf = step_core_quad_stereo8(u, &mut st_wf, &wf_params);
            let (p_odd_l, p_odd_r) = o_plain.adaa_odd.into_lr();
            let (w_odd_l, w_odd_r) = o_wf.adaa_odd.into_lr();
            assert_eq!(
                o_plain.lru_state.into_lr().0.to_array().map(f32::to_bits),
                o_wf.lru_state.into_lr().0.to_array().map(f32::to_bits)
            );
            assert_eq!(
                o_plain.adaa_even.into_lr().0.to_array().map(f32::to_bits),
                o_wf.adaa_even.into_lr().0.to_array().map(f32::to_bits)
            );
            assert_eq!(
                p_odd_l.to_array()[0].to_bits(),
                w_odd_l.to_array()[0].to_bits()
            );
            assert_eq!(
                p_odd_r.to_array()[0].to_bits(),
                w_odd_r.to_array()[0].to_bits()
            );
            max_lane3_diff =
                max_lane3_diff.max((p_odd_l.to_array()[3] - w_odd_l.to_array()[3]).abs());
            for lane in 0..4 {
                assert!(w_odd_l.to_array()[lane].is_finite());
                assert!(w_odd_r.to_array()[lane].is_finite());
                assert!(w_odd_l.to_array()[lane].abs() <= 1.05);
                assert!(w_odd_r.to_array()[lane].abs() <= 1.05);
            }
        }
        assert!(max_lane3_diff > 1.0e-2);
    }

    proptest! {
        #[test]
        fn prop_core_quad_dual_mono_stereo_link_and_glu_parity_zero_ulp(
            fc in 0.02_f32..0.35_f32,
            k_val in 0.35_f32..1.5_f32,
            alpha_rel in 0.5_f32..0.99_f32,
            alpha_atk in 0.05_f32..0.9_f32,
            kappa in 0.0_f32..1.0_f32,
            gamma_duhem in 0.0_f32..8.0_f32,
            signed_mix in 0.0_f32..1.0_f32,
            sag_depth in 0.0_f32..0.9_f32,
            adaa_drive in 0.2_f32..4.0_f32,
            lru_bias_drive in -1.2_f32..1.2_f32,
            wf_mix in prop_oneof![Just(0.0_f32), 0.01_f32..1.0_f32],
            glu_vals in prop::array::uniform4(-3.5_f32..3.5_f32),
            rect_sign_bits in [any::<bool>(); 4],
            stim in prop::collection::vec(
                prop_oneof![-2.5_f32..-1.0e-3_f32, 1.0e-3_f32..2.5_f32],
                8
            ),
        ) {
            let g_s = prewarp_tan_pi_f_over_fs_f32(fc);
            let g = GpuVec4::splat(g_s);
            let k = GpuVec4::splat(k_val);
            let rect_sign_arr = rect_sign_bits.map(|b| if b { 1.0_f32 } else { -1.0_f32 });

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
                    GpuVec4::splat(1.0),
                ),
                two_svf_g: GpuVec4::splat(2.0 * g_s),
                svf_k_nl: GpuVec4::zero(),
                rect_sign: GpuVec4::from_array(rect_sign_arr),
                sag_depth: GpuVec4::splat(sag_depth),
                adaa_drive: GpuVec4::splat(adaa_drive),
                lru_bias_drive: GpuVec4::splat(lru_bias_drive),
                wavefolder_mix: GpuVec4::splat(wf_mix),
                glu_mix: GpuVec4::from_array(glu_vals),
            };
            let params_unlinked = CoreQuadParamsStereo4 {
                lru: SonaParaLruDuhemCoeffs4::from_clamped(
                    GpuVec4::splat(alpha_rel),
                    GpuVec4::splat(alpha_atk),
                    GpuVec4::splat(kappa),
                    GpuVec4::splat(gamma_duhem),
                    GpuVec4::splat(signed_mix),
                    GpuVec4::zero(),
                ),
                ..params
            };

            let mut st_linked = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut st_unlinked = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            for &x in &stim {
                let u = InterleavedStereo8::splat_lr(x, x);
                let out_l = step_core_quad_stereo8(u, &mut st_linked, &params);
                let out_u = step_core_quad_stereo8(u, &mut st_unlinked, &params_unlinked);

                let (ll, lr) = out_l.lru_state.into_lr();
                let (ol, or) = out_l.adaa_odd.into_lr();
                let (el, er) = out_l.adaa_even.into_lr();
                let (ul_lru, ur_lru) = out_u.lru_state.into_lr();

                // Dual-mono symmetry: L == R to 0 ULP across both linked and unlinked CoreQuad paths.
                prop_assert_eq!(ll.to_array().map(f32::to_bits), lr.to_array().map(f32::to_bits));
                prop_assert_eq!(ol.to_array().map(f32::to_bits), or.to_array().map(f32::to_bits));
                prop_assert_eq!(el.to_array().map(f32::to_bits), er.to_array().map(f32::to_bits));
                for lane in 0..4 {
                    prop_assert!((ll.to_array()[lane] - ul_lru.to_array()[lane]).abs() < 1.0e-5);
                    prop_assert!((lr.to_array()[lane] - ur_lru.to_array()[lane]).abs() < 1.0e-5);
                }
            }
        }
    }
}
