#![forbid(unsafe_code)]

//! Local step pullback helpers (`Sag -> Stereo-Link -> Selective-Duhem ParaLRU -> Tone -> TPT-SVF`)
//! shared by `adjoint_core::pass0` and `adjoint_core::pass2` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::{LaneForwardFrame, LocalNonlinPullback};
use crate::abi::ADJOINT_V_TRAJ2_VEC2S;
use crate::core_math::{DuhemStepTerms, MIN_SAG_GAIN, QuadLaneParams, eval_duhem_step_terms};
use dnam_kernels::TptAdjointElem;
use spirv_std::glam::Vec2;

/// Outputs of `eval_sag_and_stereo_pullback` feeding into `ParaLRU` reverse suffix scan.
#[derive(Copy, Clone)]
pub struct SagStereoPullback {
    pub adj_tone_from_r_l: f32,
    pub adj_tone_from_r_r: f32,
    pub adj_v_raw_l: f32,
    pub adj_v_raw_r: f32,
    pub g_sag_depth: f32,
    pub g_adaa_drive: f32,
    pub g_lru_bias_drive: f32,
    pub g_stereo_link: f32,
}

/// Pulls back `adj_r` and `adj_veff_feat` through `Sag -> Stereo-Link` to `tone` and `v_raw`.
#[inline]
#[must_use]
pub fn eval_sag_and_stereo_pullback(
    fwd: &LaneForwardFrame,
    nl: &LocalNonlinPullback,
    lp: &QuadLaneParams,
    is_active: bool,
) -> SagStereoPullback {
    if !is_active {
        return SagStereoPullback {
            adj_tone_from_r_l: 0.0,
            adj_tone_from_r_r: 0.0,
            adj_v_raw_l: 0.0,
            adj_v_raw_r: 0.0,
            g_sag_depth: 0.0,
            g_adaa_drive: 0.0,
            g_lru_bias_drive: 0.0,
            g_stereo_link: 0.0,
        };
    }
    let sag_open_l = fwd.sag_raw_l > MIN_SAG_GAIN;
    let sag_open_r = fwd.sag_raw_r > MIN_SAG_GAIN;
    let adj_tone_from_r_l = nl.adj_r_l * lp.adaa_drive * fwd.sag_l;
    let adj_tone_from_r_r = nl.adj_r_r * lp.adaa_drive * fwd.sag_r;

    let adj_veff_l = nl.adj_veff_feat_l + nl.adj_r_l * lp.lru_bias_drive
        - if sag_open_l {
            nl.adj_r_l * fwd.tone_l * lp.adaa_drive * lp.sag_depth
        } else {
            0.0
        };
    let adj_veff_r = nl.adj_veff_feat_r + nl.adj_r_r * lp.lru_bias_drive
        - if sag_open_r {
            nl.adj_r_r * fwd.tone_r * lp.adaa_drive * lp.sag_depth
        } else {
            0.0
        };

    let half_lam = 0.5 * lp.stereo_link;
    let one_minus_half_lam = 1.0 - half_lam;
    let adj_v_raw_l = one_minus_half_lam * adj_veff_l + half_lam * adj_veff_r;
    let adj_v_raw_r = one_minus_half_lam * adj_veff_r + half_lam * adj_veff_l;

    let g_sag_depth = (if sag_open_l {
        -nl.adj_r_l * fwd.tone_l * lp.adaa_drive * fwd.v_eff_l
    } else {
        0.0
    }) + (if sag_open_r {
        -nl.adj_r_r * fwd.tone_r * lp.adaa_drive * fwd.v_eff_r
    } else {
        0.0
    });
    let g_adaa_drive = nl.adj_r_l * fwd.tone_l * fwd.sag_l + nl.adj_r_r * fwd.tone_r * fwd.sag_r;
    let g_lru_bias_drive = nl.adj_r_l * fwd.v_eff_l + nl.adj_r_r * fwd.v_eff_r;
    let v_mid = 0.5 * (fwd.v_raw_l + fwd.v_raw_r);
    let g_stereo_link = if lp.stereo_link_open {
        adj_veff_l * (v_mid - fwd.v_raw_l) + adj_veff_r * (v_mid - fwd.v_raw_r)
    } else {
        0.0
    };

    SagStereoPullback {
        adj_tone_from_r_l,
        adj_tone_from_r_r,
        adj_v_raw_l,
        adj_v_raw_r,
        g_sag_depth,
        g_adaa_drive,
        g_lru_bias_drive,
        g_stereo_link,
    }
}

/// Outputs of `eval_duhem_and_tone_pullback` feeding into the `TPT-SVF` reverse suffix scan.
#[derive(Copy, Clone)]
pub struct DuhemTonePullback {
    pub adj_v1_l: f32,
    pub adj_v2_l: f32,
    pub adj_v1_r: f32,
    pub adj_v2_r: f32,
    pub adj_u_nonrec_l: f32,
    pub adj_u_nonrec_r: f32,
    pub g_alpha_rel: f32,
    pub g_alpha_atk: f32,
    pub g_kappa: f32,
    pub g_gamma: f32,
    pub g_signed_mix: f32,
    pub g_two_svf_g: f32,
    pub g_rect_sign: f32,
    pub g_tone_lp: f32,
    pub g_tone_bp: f32,
    pub g_tone_hp: f32,
    pub g_tone_ap: f32,
    pub g_tone_dir: f32,
    pub g_svf_k: f32,
    pub g_svf_two_k: f32,
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
    adj_tone_from_r: f32,
    tone: f32,
    u: f32,
    svf_taps: (f32, f32, f32, f32),
    lp: &QuadLaneParams,
) -> (f32, f32, f32, [f32; 14]) {
    let (f_lp, f_bp, f_hp, f_ap) = svf_taps;
    let adj_u_rect = lam * t.rho * t.phi_eta_prime;
    let adj_tone = adj_tone_from_r + adj_u_rect * lp.rect_sign;
    let phi_eta = if lp.signed_mix <= 0.0 {
        t.phi_rect
    } else {
        t.one_minus_eta * t.phi_rect + lp.signed_mix * t.psi_1
    };
    let adj_rho = lam * (phi_eta - lp.signed_mix * t.d);

    let mut g_atk = 0.0_f32;
    let mut g_gam = 0.0_f32;
    let mut g_two_g = 0.0_f32;
    let mut adj_bp_duhem = 0.0_f32;
    if t.rho_open {
        if lp.alpha_atk_open {
            g_atk = -adj_rho;
        }
        if lp.gamma_open {
            g_gam = adj_rho * t.flux_abs;
        }
        if lp.gamma_duhem > 0.0 {
            adj_bp_duhem = adj_rho * lp.two_svf_g * lp.gamma_duhem * t.sign_bp;
            g_two_g = adj_rho * lp.gamma_duhem * t.bp_clean.abs();
        }
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

    let adj_lp = adj_tone * lp.w_lp;
    let adj_bp = adj_tone * lp.w_bp + adj_bp_duhem;
    let adj_hp = adj_tone * lp.w_hp;
    let adj_ap = adj_tone * lp.w_ap;
    let adj_u_nonrec = adj_tone * lp.w_dir + adj_hp + adj_ap;

    let adj_v1 = adj_bp - lp.k * adj_hp - lp.two_k * adj_ap;
    let adj_v2 = adj_lp - adj_hp;

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
        adj_tone * f_hp,
        adj_tone * f_ap,
        adj_tone * u,
        -adj_hp * f_bp,
        -adj_ap * f_bp,
    ];
    (adj_v1, adj_v2, adj_u_nonrec, grads)
}

/// Pulls back `ParaLRU` `(lam_l, lam_r)` and `adj_tone_from_r` into `(adj_v1, adj_v2)` and
/// per-frame parameter sensitivities for `L` and `R`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
pub fn eval_duhem_and_tone_pullback(
    fwd: &LaneForwardFrame,
    sag: &SagStereoPullback,
    tl: &DuhemStepTerms,
    tr: &DuhemStepTerms,
    lam_l: f32,
    lam_r: f32,
    taps_l: (f32, f32, f32, f32),
    taps_r: (f32, f32, f32, f32),
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
            g_alpha_rel: 0.0,
            g_alpha_atk: 0.0,
            g_kappa: 0.0,
            g_gamma: 0.0,
            g_signed_mix: 0.0,
            g_two_svf_g: 0.0,
            g_rect_sign: 0.0,
            g_tone_lp: 0.0,
            g_tone_bp: 0.0,
            g_tone_hp: 0.0,
            g_tone_ap: 0.0,
            g_tone_dir: 0.0,
            g_svf_k: 0.0,
            g_svf_two_k: 0.0,
        };
    }

    let (adj_v1_l, adj_v2_l, adj_u_nonrec_l, gl) = eval_channel_duhem_tone_grads(
        lam_l,
        tl,
        sag.adj_tone_from_r_l,
        fwd.tone_l,
        fwd.u_l,
        taps_l,
        lp,
    );
    let (adj_v1_r, adj_v2_r, adj_u_nonrec_r, gr) = eval_channel_duhem_tone_grads(
        lam_r,
        tr,
        sag.adj_tone_from_r_r,
        fwd.tone_r,
        fwd.u_r,
        taps_r,
        lp,
    );

    DuhemTonePullback {
        adj_v1_l,
        adj_v2_l,
        adj_v1_r,
        adj_v2_r,
        adj_u_nonrec_l,
        adj_u_nonrec_r,
        g_alpha_rel: gl[0] + gr[0],
        g_alpha_atk: gl[1] + gr[1],
        g_kappa: gl[2] + gr[2],
        g_gamma: gl[3] + gr[3],
        g_signed_mix: gl[4] + gr[4],
        g_two_svf_g: gl[5] + gr[5],
        g_rect_sign: gl[6] + gr[6],
        g_tone_lp: gl[7] + gr[7],
        g_tone_bp: gl[8] + gr[8],
        g_tone_hp: gl[9] + gr[9],
        g_tone_ap: gl[10] + gr[10],
        g_tone_dir: gl[11] + gr[11],
        g_svf_k: gl[12] + gr[12],
        g_svf_two_k: gl[13] + gr[13],
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
