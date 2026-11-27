//! Smooth `SONA v4.0` Selective-Duhem `ParaLRU` (`soft_rect_quartic_f32`) forward terms and
//! Stereo-Link + Duhem + `3`-tap/`4`-tap folded tone pullback matching
//! `tools/sona-shaders`'s `core_math/adaa_deriv.rs` and `adjoint_core/pullback.rs`.

use dnam_kernels::{
    GpuVec4, KernelVec4, MAX_DUHEM_FLUX_ABS, ONE_THIRD_F32, canonicalize_zero_f32_bits,
    soft_rect_quartic_f32,
};

use super::se_and_adaa::SeAndNonlinPullback;
use super::{LaneForwardFrame, QuadLaneParams, clamp_nonneg_safe};

#[derive(Copy, Clone)]
pub(super) struct DuhemStepTerms {
    pub flux_abs: f32,
    pub inv_denom: f32,
    pub rho: f32,
    pub d_rho_d_mod: f32,
    pub d: f32,
    pub one_minus_eta: f32,
    pub phi_rect: f32,
    pub psi_1: f32,
    pub phi_eta_prime: f32,
    pub jac_div_rel: f32,
    pub jac: f32,
    pub sign_bp: f32,
    pub v_clean: f32,
    pub bp_clean: f32,
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn eval_duhem_step_terms(
    u_rect: f32,
    bp: f32,
    v_prev: f32,
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
    gamma_duhem: f32,
    signed_mix: f32,
    two_svf_g: f32,
) -> DuhemStepTerms {
    let u_clean = canonicalize_zero_f32_bits(u_rect);
    let bp_clean = canonicalize_zero_f32_bits(bp);
    let v_clean = canonicalize_zero_f32_bits(v_prev);
    let flux_abs = canonicalize_zero_f32_bits(two_svf_g * bp_clean)
        .abs()
        .min(MAX_DUHEM_FLUX_ABS);
    let duhem_mod = if gamma_duhem <= 0.0 {
        0.0
    } else {
        gamma_duhem * flux_abs
    };
    let inv_denom = if duhem_mod <= 0.0 {
        1.0
    } else {
        GpuVec4::splat(1.0 + duhem_mod).rcp_ge1_nr2().to_array()[0]
    };
    let rho = clamp_nonneg_safe(1.0 - alpha_atk * inv_denom, 1.0);
    let d_rho_d_mod = alpha_atk * inv_denom * inv_denom;

    let d = alpha_rel * v_clean;
    let one_minus_eta = 1.0 - signed_mix;
    let z = u_clean - kappa * one_minus_eta * d;
    let (phi_rect, phi_rect_prime) = soft_rect_quartic_f32(z);
    let z_bar = z.clamp(-1.0, 1.0);
    let psi_1 = z_bar * (1.0 - ONE_THIRD_F32 * (z_bar * z_bar));
    let psi_1_prime = if z.abs() < 1.0 { 1.0 - z * z } else { 0.0 };
    let phi_eta_prime = if signed_mix <= 0.0 {
        phi_rect_prime
    } else {
        one_minus_eta * phi_rect_prime + signed_mix * psi_1_prime
    };
    let jac_div_rel = (1.0 - signed_mix * rho) - rho * kappa * one_minus_eta * phi_eta_prime;
    let jac = alpha_rel * jac_div_rel;
    let sign_bp = if bp_clean > 0.0 {
        1.0
    } else if bp_clean < 0.0 {
        -1.0
    } else {
        0.0
    };

    DuhemStepTerms {
        flux_abs,
        inv_denom,
        rho,
        d_rho_d_mod,
        d,
        one_minus_eta,
        phi_rect,
        psi_1,
        phi_eta_prime,
        jac_div_rel,
        jac,
        sign_bp,
        v_clean,
        bp_clean,
    }
}

#[derive(Copy, Clone)]
pub(super) struct StereoPullback {
    pub adj_tone_from_stage3_l: f32,
    pub adj_tone_from_stage3_r: f32,
    pub adj_v_raw_l: f32,
    pub adj_v_raw_r: f32,
    pub g_stereo_link: f32,
}

#[inline]
pub(super) fn eval_stereo_link_pullback(
    fwd: &LaneForwardFrame,
    nl: &SeAndNonlinPullback,
    lp: &QuadLaneParams,
) -> StereoPullback {
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

#[derive(Copy, Clone)]
pub(super) struct DuhemTonePullback {
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

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn eval_duhem_and_tone_pullback(
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
) -> DuhemTonePullback {
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
