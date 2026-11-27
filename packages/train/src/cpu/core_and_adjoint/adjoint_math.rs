//! Scalar `SONA v4.0` lane forward recomputation and reverse-mode BPTT pullback matching
//! `tools/sona-shaders`'s `adjoint_core` (`common`, `se_and_adaa`, `pullback`, `pass2`, and
//! `core_math/adaa_deriv`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod adaa_derivs;
mod duhem;
mod phs_svf;
mod se_and_adaa;

use dnam_kernels::{
    CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, MAX_ADAA_DRIVE, MAX_ALG_R, MAX_G_PRE_GAIN,
    MAX_GLU_MIX, MAX_LRU_BIAS_DRIVE, MAX_OUT_GAIN, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32,
    MAX_SVF_K_NL, PhsPredCorrPassBCoeffs, SonaParaLruDuhemCoeffs4, SonaToneFolded3,
    SonaToneFoldedBlock0, TptSvfCoeffs4, apply_stereo_link_vec4_pair, apply_tpt_adjoint,
    canonicalize_zero_f32_bits, eval_phs_pred_corr_pass_b_coeffs, scrub_clamped_alg_endpoint_f32,
    step_paralru_duhem_v4_vec4, step_tpt_svf_folded3_vec4, step_tpt_svf_vec4,
};
use duhem::{eval_duhem_and_tone_pullback, eval_duhem_step_terms, eval_stereo_link_pullback};
use phs_svf::{
    build_tpt_adjoint_elem, eval_tpt_svf_lp_bp_scalar, phs_pass_a_lin_coeffs,
    step_phs_pred_corr_reverse,
};
use se_and_adaa::{GPreTerms, eval_g_pre_terms_f32, eval_se_and_adaa_frame_pullback};

pub(super) use se_and_adaa::{needs_b_staging, stage_b_all_for_tile};

use crate::gpu::abi::QUAD_PARAMS_VEC4_STRIDE;

const IEEE_F32_EXP_MASK: u32 = 0x7f80_0000;
const MAX_NONNEG_MOD_SCALE: f32 = 1.0e12;

#[inline(always)]
pub(super) const fn is_finite_f32(x: f32) -> bool {
    (x.to_bits() & IEEE_F32_EXP_MASK) != IEEE_F32_EXP_MASK
}

#[inline(always)]
pub(crate) const fn flush_zero(x: f32) -> f32 {
    canonicalize_zero_f32_bits(x)
}

#[inline(always)]
pub(super) fn clamp_range_safe(x: f32, lo: f32, hi: f32) -> f32 {
    if is_finite_f32(x) {
        if x <= lo {
            flush_zero(lo)
        } else if x >= hi {
            flush_zero(hi)
        } else {
            flush_zero(x)
        }
    } else {
        0.0
    }
}

#[inline(always)]
pub(super) fn clamp_nonneg_safe(x: f32, hi: f32) -> f32 {
    if is_finite_f32(x) {
        if x <= 0.0 {
            0.0
        } else if x >= hi {
            hi
        } else {
            canonicalize_zero_f32_bits(x)
        }
    } else if x.to_bits() == IEEE_F32_EXP_MASK {
        hi
    } else {
        0.0
    }
}

#[derive(Copy, Clone)]
pub(crate) struct QuadLaneParams {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k: f32,
    pub two_k: f32,
    pub c_lp: f32,
    pub c_bp: f32,
    pub c_kbp: f32,
    pub c_u: f32,
    pub alpha_rel: f32,
    pub alpha_atk: f32,
    pub kappa: f32,
    pub gamma_duhem: f32,
    pub signed_mix: f32,
    pub stereo_link: f32,
    pub alpha_rel_open: bool,
    pub alpha_atk_open: bool,
    pub kappa_open: bool,
    pub gamma_open: bool,
    pub signed_mix_open: bool,
    pub stereo_link_open: bool,
    pub svf_g: f32,
    pub two_svf_g: f32,
    pub svf_k_nl: f32,
    pub rect_sign: f32,
    pub g0: f32,
    pub g_inf: f32,
    pub d_pre: f32,
    pub g0_open: bool,
    pub g_inf_open: bool,
    pub d_pre_open: bool,
    pub adaa_drive: f32,
    pub lru_bias_drive: f32,
    pub adaa_drive_open: bool,
    pub lru_bias_drive_open: bool,
    pub lin_mix: f32,
    pub wf_mix: f32,
    pub glu_mix: f32,
    pub out_gain: f32,
    pub lin_open: bool,
    pub wf_open: bool,
    pub glu_open: bool,
    pub out_open: bool,
}

impl QuadLaneParams {
    pub(crate) fn extract(p: &CoreQuadV4ParamsStereo4<GpuVec4>, c: usize) -> Self {
        let k = p.svf.k.to_array()[c];
        let svf_g = p.two_svf_g.to_array()[c] * 0.5;
        let rel_raw = p.lru.alpha_rel.to_array()[c];
        let atk_raw = p.lru.alpha_atk.to_array()[c];
        let kap_raw = p.lru.kappa.to_array()[c];
        let gam_raw = p.lru.gamma_duhem.to_array()[c];
        let eta_raw = p.lru.signed_mix.to_array()[c];
        let link_raw = p.lru.stereo_link.to_array()[c];
        let g0_raw = p.g0.to_array()[c];
        let g_inf_raw = p.g_inf.to_array()[c];
        let d_pre_raw = p.d_pre.to_array()[c];
        let adaa_raw = p.adaa_drive.to_array()[c];
        let bias_raw = p.lru_bias_drive.to_array()[c];
        let lin_raw = p.lin_mix.to_array()[c];
        let wf_raw = p.wavefolder_mix.to_array()[c];
        let glu_raw = p.glu_mix.to_array()[c];
        let out_raw = p.out_gain.to_array()[c];

        Self {
            a1: p.svf.a1.to_array()[c],
            a2: p.svf.a2.to_array()[c],
            a3: p.svf.a3.to_array()[c],
            k,
            two_k: k + k,
            c_lp: p.tone.c_lp.to_array()[c],
            c_bp: p.tone.c_bp.to_array()[c],
            c_kbp: p.tone.c_kbp.to_array()[c],
            c_u: p.tone.c_u.to_array()[c],
            alpha_rel: clamp_nonneg_safe(rel_raw, MAX_PARALRU_R_F32),
            alpha_atk: clamp_nonneg_safe(atk_raw, 1.0),
            kappa: clamp_nonneg_safe(kap_raw, 1.0),
            gamma_duhem: clamp_nonneg_safe(gam_raw, MAX_PARALRU_DUHEM_GAMMA_F32),
            signed_mix: clamp_nonneg_safe(eta_raw, 1.0),
            stereo_link: clamp_nonneg_safe(link_raw, 1.0),
            alpha_rel_open: is_finite_f32(rel_raw) && (0.0..=MAX_PARALRU_R_F32).contains(&rel_raw),
            alpha_atk_open: is_finite_f32(atk_raw) && (0.0..=1.0).contains(&atk_raw),
            kappa_open: is_finite_f32(kap_raw) && (0.0..=1.0).contains(&kap_raw),
            gamma_open: is_finite_f32(gam_raw)
                && (0.0..=MAX_PARALRU_DUHEM_GAMMA_F32).contains(&gam_raw),
            signed_mix_open: is_finite_f32(eta_raw) && (0.0..=1.0).contains(&eta_raw),
            stereo_link_open: is_finite_f32(link_raw) && (0.0..=1.0).contains(&link_raw),
            svf_g,
            two_svf_g: svf_g + svf_g,
            svf_k_nl: p.svf_k_nl.to_array()[c],
            rect_sign: p.rect_sign.to_array()[c],
            g0: clamp_range_safe(g0_raw, 0.0, MAX_G_PRE_GAIN),
            g_inf: clamp_range_safe(g_inf_raw, 0.0, MAX_G_PRE_GAIN),
            d_pre: clamp_range_safe(d_pre_raw, 0.0, MAX_NONNEG_MOD_SCALE),
            g0_open: is_finite_f32(g0_raw) && (0.0..=MAX_G_PRE_GAIN).contains(&g0_raw),
            g_inf_open: is_finite_f32(g_inf_raw) && (0.0..=MAX_G_PRE_GAIN).contains(&g_inf_raw),
            d_pre_open: is_finite_f32(d_pre_raw)
                && (0.0..=MAX_NONNEG_MOD_SCALE).contains(&d_pre_raw),
            adaa_drive: clamp_range_safe(adaa_raw, -MAX_ADAA_DRIVE, MAX_ADAA_DRIVE),
            lru_bias_drive: clamp_range_safe(bias_raw, -MAX_LRU_BIAS_DRIVE, MAX_LRU_BIAS_DRIVE),
            adaa_drive_open: is_finite_f32(adaa_raw)
                && (-MAX_ADAA_DRIVE..=MAX_ADAA_DRIVE).contains(&adaa_raw),
            lru_bias_drive_open: is_finite_f32(bias_raw)
                && (-MAX_LRU_BIAS_DRIVE..=MAX_LRU_BIAS_DRIVE).contains(&bias_raw),
            lin_mix: clamp_range_safe(lin_raw, 0.0, 1.0),
            wf_mix: clamp_range_safe(wf_raw, 0.0, 1.0),
            glu_mix: clamp_range_safe(glu_raw, -MAX_GLU_MIX, MAX_GLU_MIX),
            out_gain: clamp_range_safe(out_raw, -MAX_OUT_GAIN, MAX_OUT_GAIN),
            lin_open: is_finite_f32(lin_raw) && (0.0..=1.0).contains(&lin_raw),
            wf_open: is_finite_f32(wf_raw) && (0.0..=1.0).contains(&wf_raw),
            glu_open: is_finite_f32(glu_raw) && (-MAX_GLU_MIX..=MAX_GLU_MIX).contains(&glu_raw),
            out_open: is_finite_f32(out_raw) && (-MAX_OUT_GAIN..=MAX_OUT_GAIN).contains(&out_raw),
        }
    }

    pub(crate) fn svf_coeffs(&self) -> TptSvfCoeffs4<GpuVec4> {
        TptSvfCoeffs4 {
            a1: GpuVec4::splat(self.a1),
            a2: GpuVec4::splat(self.a2),
            a3: GpuVec4::splat(self.a3),
            k: GpuVec4::splat(self.k),
            two_k: GpuVec4::splat(self.two_k),
        }
    }

    pub(crate) fn to_tone_block0(self) -> SonaToneFoldedBlock0<GpuVec4> {
        SonaToneFoldedBlock0 {
            c_lp: GpuVec4::splat(self.c_lp),
            c_bp: GpuVec4::splat(self.c_bp),
            c_kbp: GpuVec4::splat(self.c_kbp),
            c_u: GpuVec4::splat(self.c_u),
        }
    }

    pub(crate) fn to_tone_folded3(self) -> SonaToneFolded3<GpuVec4> {
        self.to_tone_block0().to_folded3(GpuVec4::splat(self.k))
    }

    pub(crate) fn lru_coeffs(&self) -> SonaParaLruDuhemCoeffs4<GpuVec4> {
        SonaParaLruDuhemCoeffs4 {
            alpha_rel: GpuVec4::splat(self.alpha_rel),
            alpha_atk: GpuVec4::splat(self.alpha_atk),
            kappa: GpuVec4::splat(self.kappa),
            gamma_duhem: GpuVec4::splat(self.gamma_duhem),
            signed_mix: GpuVec4::splat(self.signed_mix),
            stereo_link: GpuVec4::splat(self.stereo_link),
        }
    }
}

#[derive(Copy, Clone, Default)]
pub(crate) struct LaneForwardFrame {
    pub u_l: f32,
    pub u_r: f32,
    pub s1_prev_l: f32,
    pub s2_prev_l: f32,
    pub s1_prev_r: f32,
    pub s2_prev_r: f32,
    pub s_lin_prev_l: [f32; 2],
    pub s_lin_prev_r: [f32; 2],
    pub v1_lin_l: f32,
    pub v1_lin_r: f32,
    pub tone_l: f32,
    pub tone_r: f32,
    pub bp_l: f32,
    pub bp_r: f32,
    pub v_prev_l: f32,
    pub v_prev_r: f32,
    pub v_raw_l: f32,
    pub v_raw_r: f32,
    pub v_eff_l: f32,
    pub v_eff_r: f32,
    pub g_pre_l: GPreTerms,
    pub g_pre_r: GPreTerms,
    pub r_prev_l: f32,
    pub r_prev_r: f32,
    pub r_curr_l: f32,
    pub r_curr_r: f32,
    pub r_curr_open_l: bool,
    pub r_curr_open_r: bool,
}

/// Scalar lane state `[s1_l, s2_l, s1_r, s2_r, v_l, v_r, adaa_prev_l, adaa_prev_r,
/// s1_lin_l, s2_lin_l, s1_lin_r, s2_lin_r]` (`12` floats matching `TILE_CKPT_VEC4_STRIDE`).
pub(crate) fn step_lane_forward_v4(
    lp: &QuadLaneParams,
    svf_c: &TptSvfCoeffs4<GpuVec4>,
    lru_c: &SonaParaLruDuhemCoeffs4<GpuVec4>,
    u_l: f32,
    u_r: f32,
    state: &mut [f32; 12],
    is_phs: bool,
) -> LaneForwardFrame {
    let [
        s1_prev_l,
        s2_prev_l,
        s1_prev_r,
        s2_prev_r,
        v_prev_l,
        v_prev_r,
        r_prev_l,
        r_prev_r,
        s1_lin_prev_l,
        s2_lin_prev_l,
        s1_lin_prev_r,
        s2_lin_prev_r,
    ] = *state;

    let (
        s1_next_l,
        s2_next_l,
        s1_next_r,
        s2_next_r,
        s1_lin_next_l,
        s2_lin_next_l,
        s1_lin_next_r,
        s2_lin_next_r,
        v1_lin_l,
        v1_lin_r,
        tone_l,
        tone_r,
        bp_l,
        bp_r,
    ) = if is_phs {
        let (lin_a1, lin_a2, lin_a3) = phs_pass_a_lin_coeffs(lp);
        let lin_svf = TptSvfCoeffs4 {
            a1: GpuVec4::splat(lin_a1),
            a2: GpuVec4::splat(lin_a2),
            a3: GpuVec4::splat(lin_a3),
            k: GpuVec4::splat(lp.k.max(0.0)),
            two_k: GpuVec4::splat(2.0 * lp.k.max(0.0)),
        };
        let mut sl1 = GpuVec4::splat(s1_lin_prev_l);
        let mut sl2 = GpuVec4::splat(s2_lin_prev_l);
        let lin_out_l = step_tpt_svf_vec4(GpuVec4::splat(u_l), &mut sl1, &mut sl2, &lin_svf);
        let mut sr1 = GpuVec4::splat(s1_lin_prev_r);
        let mut sr2 = GpuVec4::splat(s2_lin_prev_r);
        let lin_out_r = step_tpt_svf_vec4(GpuVec4::splat(u_r), &mut sr1, &mut sr2, &lin_svf);
        let v1l = lin_out_l.bp.to_array()[0];
        let v1r = lin_out_r.bp.to_array()[0];

        let pb_l = eval_phs_pred_corr_pass_b_coeffs(v1l, lp.svf_g, lp.k, lp.svf_k_nl);
        let pb_r = eval_phs_pred_corr_pass_b_coeffs(v1r, lp.svf_g, lp.k, lp.svf_k_nl);
        let mut s1_l = GpuVec4::splat(s1_prev_l);
        let mut s2_l = GpuVec4::splat(s2_prev_l);
        let svf_bl = TptSvfCoeffs4 {
            a1: GpuVec4::splat(pb_l.a1),
            a2: GpuVec4::splat(pb_l.a2),
            a3: GpuVec4::splat(pb_l.a3),
            k: GpuVec4::splat(pb_l.k_eff),
            two_k: GpuVec4::splat(pb_l.k_eff + pb_l.k_eff),
        };
        let out_bl = step_tpt_svf_vec4(GpuVec4::splat(u_l), &mut s1_l, &mut s2_l, &svf_bl);
        let mut s1_r = GpuVec4::splat(s1_prev_r);
        let mut s2_r = GpuVec4::splat(s2_prev_r);
        let svf_br = TptSvfCoeffs4 {
            a1: GpuVec4::splat(pb_r.a1),
            a2: GpuVec4::splat(pb_r.a2),
            a3: GpuVec4::splat(pb_r.a3),
            k: GpuVec4::splat(pb_r.k_eff),
            two_k: GpuVec4::splat(pb_r.k_eff + pb_r.k_eff),
        };
        let out_br = step_tpt_svf_vec4(GpuVec4::splat(u_r), &mut s1_r, &mut s2_r, &svf_br);
        let tb0 = lp.to_tone_block0();
        let tl = tb0
            .eval_vec4(
                out_bl.lp,
                out_bl.bp,
                GpuVec4::splat(pb_l.k_eff),
                GpuVec4::splat(u_l),
            )
            .to_array()[0];
        let tr = tb0
            .eval_vec4(
                out_br.lp,
                out_br.bp,
                GpuVec4::splat(pb_r.k_eff),
                GpuVec4::splat(u_r),
            )
            .to_array()[0];
        (
            s1_l.to_array()[0],
            s2_l.to_array()[0],
            s1_r.to_array()[0],
            s2_r.to_array()[0],
            sl1.to_array()[0],
            sl2.to_array()[0],
            sr1.to_array()[0],
            sr2.to_array()[0],
            v1l,
            v1r,
            tl,
            tr,
            out_bl.bp.to_array()[0],
            out_br.bp.to_array()[0],
        )
    } else {
        let tone_f3 = lp.to_tone_folded3();
        let mut s1_l = GpuVec4::splat(s1_prev_l);
        let mut s2_l = GpuVec4::splat(s2_prev_l);
        let (tl_v, bpl_v) =
            step_tpt_svf_folded3_vec4(GpuVec4::splat(u_l), &mut s1_l, &mut s2_l, svf_c, &tone_f3);
        let mut s1_r = GpuVec4::splat(s1_prev_r);
        let mut s2_r = GpuVec4::splat(s2_prev_r);
        let (tr_v, bpr_v) =
            step_tpt_svf_folded3_vec4(GpuVec4::splat(u_r), &mut s1_r, &mut s2_r, svf_c, &tone_f3);
        (
            s1_l.to_array()[0],
            s2_l.to_array()[0],
            s1_r.to_array()[0],
            s2_r.to_array()[0],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            tl_v.to_array()[0],
            tr_v.to_array()[0],
            bpl_v.to_array()[0],
            bpr_v.to_array()[0],
        )
    };

    let mut vl = GpuVec4::splat(v_prev_l);
    let mut vr = GpuVec4::splat(v_prev_r);
    let raw_l = step_paralru_duhem_v4_vec4(
        GpuVec4::splat(lp.rect_sign * tone_l),
        GpuVec4::splat(lp.two_svf_g * bp_l),
        &mut vl,
        lru_c,
    );
    let raw_r = step_paralru_duhem_v4_vec4(
        GpuVec4::splat(lp.rect_sign * tone_r),
        GpuVec4::splat(lp.two_svf_g * bp_r),
        &mut vr,
        lru_c,
    );
    let (eff_l, eff_r) = apply_stereo_link_vec4_pair(raw_l, raw_r, lru_c.stereo_link);
    let (v_raw_l, v_raw_r) = (raw_l.to_array()[0], raw_r.to_array()[0]);
    let (v_eff_l, v_eff_r) = (eff_l.to_array()[0], eff_r.to_array()[0]);
    let g_pre_l = eval_g_pre_terms_f32(v_eff_l, lp);
    let g_pre_r = eval_g_pre_terms_f32(v_eff_r, lp);

    let r_unclamped_l = (g_pre_l.g_pre * lp.adaa_drive) * tone_l + lp.lru_bias_drive * v_eff_l;
    let r_unclamped_r = (g_pre_r.g_pre * lp.adaa_drive) * tone_r + lp.lru_bias_drive * v_eff_r;
    let r_curr_open_l = is_finite_f32(r_unclamped_l) && r_unclamped_l.abs() <= MAX_ALG_R;
    let r_curr_open_r = is_finite_f32(r_unclamped_r) && r_unclamped_r.abs() <= MAX_ALG_R;
    let r_curr_l = scrub_clamped_alg_endpoint_f32(r_unclamped_l);
    let r_curr_r = scrub_clamped_alg_endpoint_f32(r_unclamped_r);

    *state = [
        s1_next_l,
        s2_next_l,
        s1_next_r,
        s2_next_r,
        v_raw_l,
        v_raw_r,
        r_curr_l,
        r_curr_r,
        s1_lin_next_l,
        s2_lin_next_l,
        s1_lin_next_r,
        s2_lin_next_r,
    ];

    LaneForwardFrame {
        u_l,
        u_r,
        s1_prev_l,
        s2_prev_l,
        s1_prev_r,
        s2_prev_r,
        s_lin_prev_l: [s1_lin_prev_l, s2_lin_prev_l],
        s_lin_prev_r: [s1_lin_prev_r, s2_lin_prev_r],
        v1_lin_l,
        v1_lin_r,
        tone_l,
        tone_r,
        bp_l,
        bp_r,
        v_prev_l,
        v_prev_r,
        v_raw_l,
        v_raw_r,
        v_eff_l,
        v_eff_r,
        g_pre_l,
        g_pre_r,
        r_prev_l,
        r_prev_r,
        r_curr_l,
        r_curr_r,
        r_curr_open_l,
        r_curr_open_r,
    }
}

#[derive(Copy, Clone, Default)]
pub(crate) struct ReverseCarry {
    pub adj_r_next_l: f32,
    pub adj_r_next_r: f32,
    pub lam_after_l: f32,
    pub lam_after_r: f32,
    pub adj_s_out_l: [f32; 2],
    pub adj_s_out_r: [f32; 2],
    pub adj_s_lin_out_l: [f32; 2],
    pub adj_s_lin_out_r: [f32; 2],
}

#[derive(Copy, Clone, Default)]
pub(crate) struct TileGradAccum {
    pub slots: [f32; QUAD_PARAMS_VEC4_STRIDE],
    pub duhem_two_svf_g: f32,
    pub svf_k_tone: f32,
}

pub(crate) struct ReverseStepContext<'a> {
    pub d_loss_d_features: &'a [[f32; 4]],
    pub feature_outputs: &'a [[f32; 4]],
    pub params: &'a CoreQuadV4ParamsStereo4<GpuVec4>,
    pub lp: &'a QuadLaneParams,
    pub q_global: usize,
    pub num_frames: usize,
    pub c: usize,
    pub tiny_eps: f32,
    pub is_phs: bool,
}

pub(crate) fn step_lane_reverse_v4(
    ctx: &ReverseStepContext<'_>,
    fwd: &LaneForwardFrame,
    frame: usize,
    b_staged: Option<([f32; 4], [f32; 4])>,
    carry: &mut ReverseCarry,
    acc: &mut TileGradAccum,
) -> (f32, f32) {
    let params = ctx.params;
    let lp = ctx.lp;
    let is_phs = ctx.is_phs;
    let nonlin = eval_se_and_adaa_frame_pullback(
        ctx,
        params,
        lp,
        fwd,
        b_staged,
        frame,
        (carry.adj_r_next_l, carry.adj_r_next_r),
    );
    carry.adj_r_next_l = nonlin.adj_r_prev_l;
    carry.adj_r_next_r = nonlin.adj_r_prev_r;

    use crate::gpu::abi::quad_param_slots;
    acc.slots[quad_param_slots::G0] += nonlin.grad_g0;
    acc.slots[quad_param_slots::G_INF] += nonlin.grad_g_inf;
    acc.slots[quad_param_slots::D_PRE] += nonlin.grad_d_pre;
    acc.slots[quad_param_slots::SE_W_SQ] += nonlin.grad_se_w_sq;
    acc.slots[quad_param_slots::SE_D_BUS] += nonlin.grad_se_d_bus;
    acc.slots[quad_param_slots::ADAA_DRIVE] += nonlin.grad_adaa_drive;
    acc.slots[quad_param_slots::LRU_BIAS_DRIVE] += nonlin.grad_lru_bias_drive;
    acc.slots[quad_param_slots::LIN_MIX] += nonlin.grad_lin_mix;
    acc.slots[quad_param_slots::WAVEFOLDER_MIX] += nonlin.grad_wf_mix;
    acc.slots[quad_param_slots::GLU_MIX] += nonlin.grad_glu_mix;
    acc.slots[quad_param_slots::OUT_GAIN] += nonlin.grad_out_gain;

    let st = eval_stereo_link_pullback(fwd, &nonlin, lp);
    if lp.stereo_link_open {
        acc.slots[quad_param_slots::LRU_STEREO_LINK] += st.g_stereo_link;
    }

    let tl = eval_duhem_step_terms(
        lp.rect_sign * fwd.tone_l,
        fwd.bp_l,
        fwd.v_prev_l,
        lp.alpha_rel,
        lp.alpha_atk,
        lp.kappa,
        lp.gamma_duhem,
        lp.signed_mix,
        lp.two_svf_g,
    );
    let tr = eval_duhem_step_terms(
        lp.rect_sign * fwd.tone_r,
        fwd.bp_r,
        fwd.v_prev_r,
        lp.alpha_rel,
        lp.alpha_atk,
        lp.kappa,
        lp.gamma_duhem,
        lp.signed_mix,
        lp.two_svf_g,
    );
    let lam_l = st.adj_v_raw_l + carry.lam_after_l;
    let lam_r = st.adj_v_raw_r + carry.lam_after_r;
    carry.lam_after_l = tl.jac * lam_l;
    carry.lam_after_r = tr.jac * lam_r;

    let svf_c = lp.svf_coeffs();
    let (pb_l, pb_r, lp_l, bp_l, lp_r, bp_r, k_eff_l, k_eff_r) = if is_phs {
        let pbl = eval_phs_pred_corr_pass_b_coeffs(fwd.v1_lin_l, lp.svf_g, lp.k, lp.svf_k_nl);
        let pbr = eval_phs_pred_corr_pass_b_coeffs(fwd.v1_lin_r, lp.svf_g, lp.k, lp.svf_k_nl);
        let bpl = pbl.a1 * fwd.s1_prev_l + pbl.a2 * (fwd.u_l - fwd.s2_prev_l);
        let lpl = fwd.s2_prev_l + pbl.a2 * fwd.s1_prev_l + pbl.a3 * (fwd.u_l - fwd.s2_prev_l);
        let bpr = pbr.a1 * fwd.s1_prev_r + pbr.a2 * (fwd.u_r - fwd.s2_prev_r);
        let lpr = fwd.s2_prev_r + pbr.a2 * fwd.s1_prev_r + pbr.a3 * (fwd.u_r - fwd.s2_prev_r);
        (pbl, pbr, lpl, bpl, lpr, bpr, pbl.k_eff, pbr.k_eff)
    } else {
        let (lpl, bpl) = eval_tpt_svf_lp_bp_scalar(fwd.u_l, fwd.s1_prev_l, fwd.s2_prev_l, &svf_c);
        let (lpr, bpr) = eval_tpt_svf_lp_bp_scalar(fwd.u_r, fwd.s1_prev_r, fwd.s2_prev_r, &svf_c);
        let z_pb = PhsPredCorrPassBCoeffs {
            v1_lin: 0.0,
            e: 0.0,
            k_eff: lp.k,
            a1: lp.a1,
            a2: lp.a2,
            a3: lp.a3,
        };
        (z_pb, z_pb, lpl, bpl, lpr, bpr, lp.k, lp.k)
    };

    let dt = eval_duhem_and_tone_pullback(
        fwd, &st, &tl, &tr, lam_l, lam_r, lp_l, bp_l, lp_r, bp_r, k_eff_l, k_eff_r, lp,
    );
    acc.slots[quad_param_slots::LRU_ALPHA_REL] += dt.g_alpha_rel;
    acc.slots[quad_param_slots::LRU_ALPHA_ATK] += dt.g_alpha_atk;
    acc.slots[quad_param_slots::LRU_KAPPA] += dt.g_kappa;
    acc.slots[quad_param_slots::LRU_GAMMA_DUHEM] += dt.g_gamma;
    acc.slots[quad_param_slots::LRU_SIGNED_MIX] += dt.g_signed_mix;
    acc.duhem_two_svf_g += dt.g_two_svf_g;
    acc.slots[quad_param_slots::RECT_SIGN] += dt.g_rect_sign;
    acc.slots[quad_param_slots::TONE_C_LP] += dt.g_tone_c_lp;
    acc.slots[quad_param_slots::TONE_C_BP] += dt.g_tone_c_bp;
    acc.slots[quad_param_slots::TONE_C_KBP] += dt.g_tone_c_kbp;
    acc.slots[quad_param_slots::TONE_C_U] += dt.g_tone_c_u;

    if is_phs {
        return step_phs_pred_corr_reverse(fwd, lp, &pb_l, &pb_r, &dt, carry, acc);
    }
    acc.svf_k_tone += dt.g_svf_k_tone;

    let tot_v1_l = dt.adj_v1_l + 2.0 * carry.adj_s_out_l[0];
    let tot_v2_l = dt.adj_v2_l + 2.0 * carry.adj_s_out_l[1];
    let tot_v1_r = dt.adj_v1_r + 2.0 * carry.adj_s_out_r[0];
    let tot_v2_r = dt.adj_v2_r + 2.0 * carry.adj_s_out_r[1];
    let v3_l = fwd.u_l - fwd.s2_prev_l;
    let v3_r = fwd.u_r - fwd.s2_prev_r;
    acc.slots[quad_param_slots::SVF_A1] += tot_v1_l * fwd.s1_prev_l + tot_v1_r * fwd.s1_prev_r;
    acc.slots[quad_param_slots::SVF_A2] +=
        (tot_v1_l * v3_l + tot_v2_l * fwd.s1_prev_l) + (tot_v1_r * v3_r + tot_v2_r * fwd.s1_prev_r);
    acc.slots[quad_param_slots::SVF_A3] += tot_v2_l * v3_l + tot_v2_r * v3_r;

    let du_l = dt.adj_u_nonrec_l + lp.a2 * tot_v1_l + lp.a3 * tot_v2_l;
    let du_r = dt.adj_u_nonrec_r + lp.a2 * tot_v1_r + lp.a3 * tot_v2_r;
    carry.adj_s_out_l = apply_tpt_adjoint(
        build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_l, dt.adj_v2_l),
        carry.adj_s_out_l,
    );
    carry.adj_s_out_r = apply_tpt_adjoint(
        build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_r, dt.adj_v2_r),
        carry.adj_s_out_r,
    );
    let _ = MAX_SVF_K_NL;
    (du_l, du_r)
}
