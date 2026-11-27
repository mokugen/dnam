#![forbid(unsafe_code)]

//! Local step pullback helpers (`Stereo-Link -> Smooth Selective-Duhem ParaLRU -> 3-Tap Folded
//! Tone -> Pred-Corr PHS / TPT-SVF`) shared by `adjoint_core::pass0` and `adjoint_core::pass2`
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::LaneForwardFrame;
use super::se_and_adaa::SeAndNonlinPullback;
use crate::abi::ADJOINT_V_TRAJ2_VEC2S;
use crate::core_math::{DuhemStepTerms, QuadLaneParams, eval_duhem_step_terms};
use dnam_kernels::{MAX_DUHEM_FLUX_ABS, MIN_TPT_G, PhsPredCorrPassBCoeffs, TptAdjointElem};
use spirv_std::glam::Vec2;

const PHS_E_MAX: f32 = 64.0;

/// Outputs of `eval_stereo_link_pullback` feeding into `ParaLRU` reverse suffix scan.
#[derive(Copy, Clone)]
pub struct StereoPullback {
    pub adj_tone_from_stage3_l: f32,
    pub adj_tone_from_stage3_r: f32,
    pub adj_v_raw_l: f32,
    pub adj_v_raw_r: f32,
    pub g_stereo_link: f32,
}

impl StereoPullback {
    #[inline(always)]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            adj_tone_from_stage3_l: 0.0,
            adj_tone_from_stage3_r: 0.0,
            adj_v_raw_l: 0.0,
            adj_v_raw_r: 0.0,
            g_stereo_link: 0.0,
        }
    }
}

/// Pulls back `adj_veff_total` through `Stereo-Link` to `v_raw_{l,r}` and collects `g_stereo_link`.
#[inline]
#[must_use]
pub fn eval_stereo_link_pullback(
    fwd: &LaneForwardFrame,
    nl: &SeAndNonlinPullback,
    lp: &QuadLaneParams,
    is_active: bool,
) -> StereoPullback {
    if !is_active {
        return StereoPullback::zero();
    }
    let half_lam = 0.5 * lp.stereo_link;
    let one_minus_half_lam = 1.0 - half_lam;
    let adj_v_raw_l = one_minus_half_lam * nl.adj_veff_total_l + half_lam * nl.adj_veff_total_r;
    let adj_v_raw_r = one_minus_half_lam * nl.adj_veff_total_r + half_lam * nl.adj_veff_total_l;

    let v_mid = 0.5 * (fwd.v_raw_l + fwd.v_raw_r);
    let g_stereo_link = if lp.stereo_link_open {
        nl.adj_veff_total_l * (v_mid - fwd.v_raw_l) + nl.adj_veff_total_r * (v_mid - fwd.v_raw_r)
    } else {
        0.0
    };

    StereoPullback {
        adj_tone_from_stage3_l: nl.adj_tone_from_stage3_l,
        adj_tone_from_stage3_r: nl.adj_tone_from_stage3_r,
        adj_v_raw_l,
        adj_v_raw_r,
        g_stereo_link,
    }
}

/// Outputs of `eval_duhem_and_tone_pullback` feeding into `TPT-SVF` / `Pred-Corr PHS` reverse scans.
#[derive(Copy, Clone)]
pub struct DuhemTonePullback {
    pub adj_v1_l: f32,
    pub adj_v2_l: f32,
    pub adj_v1_r: f32,
    pub adj_v2_r: f32,
    pub adj_u_nonrec_l: f32,
    pub adj_u_nonrec_r: f32,
    pub adj_k_eff_tone_l: f32,
    pub adj_k_eff_tone_r: f32,
    pub g_alpha_rel: f32,
    pub g_alpha_atk: f32,
    pub g_kappa: f32,
    pub g_gamma: f32,
    pub g_signed_mix: f32,
    pub g_two_svf_g: f32,
    pub g_rect_sign: f32,
    pub g_tone_c_lp: f32,
    pub g_tone_c_bp: f32,
    pub g_tone_c_kbp: f32,
    pub g_tone_c_u: f32,
    pub g_svf_k_tone: f32,
}

/// Computes `DuhemStepTerms` for `(L, R)` at frame `lid` using `v_traj2` and `(bp_l, bp_r)`.
#[inline]
#[must_use]
pub fn eval_frame_duhem_terms_lr(
    v_traj2: &[Vec2; ADJOINT_V_TRAJ2_VEC2S],
    fwd: &LaneForwardFrame,
    bp_l: f32,
    bp_r: f32,
    lid: usize,
    lp: &QuadLaneParams,
) -> (DuhemStepTerms, DuhemStepTerms) {
    let i = lid & 255;
    let (v_prev_l, v_prev_r) = if i == 0 {
        (fwd.v_init_l, fwd.v_init_r)
    } else {
        let prev = v_traj2[(i - 1) & 255];
        (prev.x, prev.y)
    };
    let tl = eval_duhem_step_terms(
        lp.rect_sign * fwd.tone_l,
        bp_l,
        v_prev_l,
        lp.alpha_rel,
        lp.alpha_atk,
        lp.kappa,
        lp.gamma_duhem,
        lp.signed_mix,
        lp.two_svf_g,
    );
    let tr = eval_duhem_step_terms(
        lp.rect_sign * fwd.tone_r,
        bp_r,
        v_prev_r,
        lp.alpha_rel,
        lp.alpha_atk,
        lp.kappa,
        lp.gamma_duhem,
        lp.signed_mix,
        lp.two_svf_g,
    );
    (tl, tr)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn eval_channel_duhem_tone_grads(
    lam: f32,
    t: &DuhemStepTerms,
    adj_tone_from_stage3: f32,
    tone: f32,
    u: f32,
    f_lp: f32,
    f_bp: f32,
    k_eff: f32,
    lp: &QuadLaneParams,
) -> (f32, f32, f32, f32, [f32; 12]) {
    let adj_u_rect = lam * t.rho * t.phi_eta_prime;
    let adj_tone = adj_tone_from_stage3 + adj_u_rect * lp.rect_sign;
    let phi_eta = if lp.signed_mix <= 0.0 {
        t.phi_rect
    } else {
        t.one_minus_eta * t.phi_rect + lp.signed_mix * t.psi_1
    };
    let adj_rho = lam * (phi_eta - lp.signed_mix * t.d);
    let adj_mod = adj_rho * t.d_rho_d_mod;

    let g_atk = if lp.alpha_atk_open {
        -adj_rho * t.inv_denom
    } else {
        0.0
    };
    let g_gam = if lp.gamma_open {
        adj_mod * t.flux_abs
    } else {
        0.0
    };
    let mut g_two_g = 0.0_f32;
    let mut adj_bp_duhem = 0.0_f32;
    let flux_raw_abs = (lp.two_svf_g * t.bp_clean).abs();
    if lp.gamma_duhem > 0.0 && flux_raw_abs < MAX_DUHEM_FLUX_ABS {
        adj_bp_duhem = adj_mod * lp.two_svf_g * lp.gamma_duhem * t.sign_bp;
        g_two_g = adj_mod * lp.gamma_duhem * t.bp_clean.abs();
    }

    let g_rel = if lp.alpha_rel_open {
        lam * t.v_clean * t.jac_div_rel
    } else {
        0.0
    };
    let g_kap = if lp.kappa_open {
        -lam * t.rho * t.one_minus_eta * t.d * t.phi_eta_prime
    } else {
        0.0
    };
    let g_eta = if lp.signed_mix_open {
        lam * t.rho * ((t.psi_1 - t.phi_rect) + lp.kappa * t.d * t.phi_eta_prime - t.d)
    } else {
        0.0
    };
    let g_rect = adj_u_rect * tone;

    let c_bp_eff = lp.c_bp + lp.c_kbp * k_eff;
    let adj_lp = adj_tone * lp.c_lp;
    let adj_bp = adj_tone * c_bp_eff + adj_bp_duhem;
    let adj_u_nonrec = adj_tone * lp.c_u;
    let adj_k_eff_tone = adj_tone * lp.c_kbp * f_bp;

    let grads = [
        g_rel,
        g_atk,
        g_kap,
        g_gam,
        g_eta,
        g_two_g,
        g_rect,
        adj_tone * f_lp,
        adj_tone * f_bp,
        adj_tone * k_eff * f_bp,
        adj_tone * u,
        adj_k_eff_tone,
    ];
    (adj_bp, adj_lp, adj_u_nonrec, adj_k_eff_tone, grads)
}

/// Pulls back smooth `ParaLRU` `(lam_l, lam_r)` and `adj_tone_from_stage3` through 3-tap folded
/// tone into `(adj_v1, adj_v2)`, `adj_k_eff_tone`, and per-frame parameter gradients.
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
pub fn eval_duhem_and_tone_pullback(
    fwd: &LaneForwardFrame,
    st: &StereoPullback,
    tl: &DuhemStepTerms,
    tr: &DuhemStepTerms,
    lam_l: f32,
    lam_r: f32,
    lp_l: f32,
    bp_l: f32,
    lp_r: f32,
    bp_r: f32,
    k_eff_l: f32,
    k_eff_r: f32,
    lp: &QuadLaneParams,
    is_active: bool,
) -> DuhemTonePullback {
    if !is_active {
        return DuhemTonePullback {
            adj_v1_l: 0.0,
            adj_v2_l: 0.0,
            adj_v1_r: 0.0,
            adj_v2_r: 0.0,
            adj_u_nonrec_l: 0.0,
            adj_u_nonrec_r: 0.0,
            adj_k_eff_tone_l: 0.0,
            adj_k_eff_tone_r: 0.0,
            g_alpha_rel: 0.0,
            g_alpha_atk: 0.0,
            g_kappa: 0.0,
            g_gamma: 0.0,
            g_signed_mix: 0.0,
            g_two_svf_g: 0.0,
            g_rect_sign: 0.0,
            g_tone_c_lp: 0.0,
            g_tone_c_bp: 0.0,
            g_tone_c_kbp: 0.0,
            g_tone_c_u: 0.0,
            g_svf_k_tone: 0.0,
        };
    }

    let (adj_v1_l, adj_v2_l, adj_u_nonrec_l, adj_k_eff_tone_l, gl) = eval_channel_duhem_tone_grads(
        lam_l,
        tl,
        st.adj_tone_from_stage3_l,
        fwd.tone_l,
        fwd.u_l,
        lp_l,
        bp_l,
        k_eff_l,
        lp,
    );
    let (adj_v1_r, adj_v2_r, adj_u_nonrec_r, adj_k_eff_tone_r, gr) = eval_channel_duhem_tone_grads(
        lam_r,
        tr,
        st.adj_tone_from_stage3_r,
        fwd.tone_r,
        fwd.u_r,
        lp_r,
        bp_r,
        k_eff_r,
        lp,
    );

    DuhemTonePullback {
        adj_v1_l,
        adj_v2_l,
        adj_v1_r,
        adj_v2_r,
        adj_u_nonrec_l,
        adj_u_nonrec_r,
        adj_k_eff_tone_l,
        adj_k_eff_tone_r,
        g_alpha_rel: gl[0] + gr[0],
        g_alpha_atk: gl[1] + gr[1],
        g_kappa: gl[2] + gr[2],
        g_gamma: gl[3] + gr[3],
        g_signed_mix: gl[4] + gr[4],
        g_two_svf_g: gl[5] + gr[5],
        g_rect_sign: gl[6] + gr[6],
        g_tone_c_lp: gl[7] + gr[7],
        g_tone_c_bp: gl[8] + gr[8],
        g_tone_c_kbp: gl[9] + gr[9],
        g_tone_c_u: gl[10] + gr[10],
        g_svf_k_tone: gl[11] + gr[11],
    }
}

/// Builds the reverse `2x2` `TPT-SVF` adjoint scan element pulling back `(adj_v1, adj_v2)` onto
/// `s_{t-1}` (`ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
#[must_use]
pub fn build_tpt_adjoint_elem(
    a1: f32,
    a2: f32,
    a3: f32,
    adj_v1: f32,
    adj_v2: f32,
    is_active: bool,
) -> TptAdjointElem {
    if !is_active {
        return TptAdjointElem::identity();
    }
    let two_a2 = a2 + a2;
    let two_a3 = a3 + a3;
    TptAdjointElem {
        t00: (a1 + a1) - 1.0,
        t01: two_a2,
        t10: -two_a2,
        t11: 1.0 - two_a3,
        g0: a1 * adj_v1 + a2 * adj_v2,
        g1: -a2 * adj_v1 + (1.0 - a3) * adj_v2,
    }
}

/// Evaluates Pass B -> Pass A coupling scalar `adj_v1_lin` for Block `0` 1-step Predictor-Corrector
/// `PHS`, including both Pass B `k_eff` sensitivity and Block `0` `c_kbp * k_eff * BP` tone coupling.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[must_use]
pub fn eval_phs_adj_v1_lin(
    pb: &PhsPredCorrPassBCoeffs,
    s_b_prev: [f32; 2],
    u: f32,
    g: f32,
    k_nl: f32,
    adj_bp_out: f32,
    adj_lp_out: f32,
    adj_k_eff_tone: f32,
    lam_b_post: [f32; 2],
) -> f32 {
    let g_safe = if g > MIN_TPT_G { g } else { MIN_TPT_G };
    let k_nl_safe = if k_nl > 0.0 { k_nl } else { 0.0 };
    let adj_v1_b = adj_bp_out + 2.0 * lam_b_post[0];
    let adj_v2_b = adj_lp_out + 2.0 * lam_b_post[1];
    let v3_b = u - s_b_prev[1];
    let adj_a1_b = adj_v1_b * s_b_prev[0];
    let adj_a2_b = adj_v1_b * v3_b + adj_v2_b * s_b_prev[0];
    let adj_a3_b = adj_v2_b * v3_b;
    let dl_da1_b = adj_a1_b + g_safe * adj_a2_b + (g_safe * g_safe) * adj_a3_b;
    let a1_b_sq = pb.a1 * pb.a1;
    let adj_k_eff = -(g_safe * a1_b_sq) * dl_da1_b + adj_k_eff_tone;
    if pb.v1_lin * pb.v1_lin < PHS_E_MAX {
        2.0 * k_nl_safe * pb.v1_lin * adj_k_eff
    } else {
        0.0
    }
}

/// Evaluates the extra `(adj_u, adj_g, adj_k, adj_k_nl)` contributions from Block `0` folded tone
/// `c_kbp * k_eff * BP` (`adj_k_eff_tone`), complementing `dnam_kernels::eval_phs_pred_corr_step_adjoint`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
#[must_use]
pub fn eval_phs_k_eff_tone_extras(
    u: f32,
    s_lin_prev: [f32; 2],
    g: f32,
    k: f32,
    k_nl: f32,
    pass_a_a1: f32,
    pass_a_a2: f32,
    pb: &PhsPredCorrPassBCoeffs,
    adj_k_eff_tone: f32,
) -> (f32, f32, f32, f32) {
    let g_safe = if g > MIN_TPT_G { g } else { MIN_TPT_G };
    let k_safe = if k > 0.0 { k } else { 0.0 };
    let k_nl_safe = if k_nl > 0.0 { k_nl } else { 0.0 };
    let delta_v1_lin = if pb.v1_lin * pb.v1_lin < PHS_E_MAX {
        2.0 * k_nl_safe * pb.v1_lin * adj_k_eff_tone
    } else {
        0.0
    };
    let v3_a = u - s_lin_prev[1];
    let adj_a1_a = delta_v1_lin * s_lin_prev[0];
    let adj_a2_a = delta_v1_lin * v3_a;
    let extra_u = delta_v1_lin * pass_a_a2;
    let dl_da1_a = adj_a1_a + g_safe * adj_a2_a;
    let a1_a_sq = pass_a_a1 * pass_a_a1;
    let extra_g = pass_a_a1 * adj_a2_a - (2.0 * g_safe + k_safe) * a1_a_sq * dl_da1_a;
    let extra_k = adj_k_eff_tone - (g_safe * a1_a_sq) * dl_da1_a;
    let extra_k_nl = pb.e * adj_k_eff_tone;
    (extra_u, extra_g, extra_k, extra_k_nl)
}
