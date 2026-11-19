#![forbid(unsafe_code)]
//! Reverse-mode BPTT sweeps through the tier readouts, downsampler, SVF, GRU cascade, and upsampler.

use std::f64::consts::LN_2;

use dnam_engine::stages::stage1_svf::SVF_G_MAX;
use dnam_engine::stages::stage2_gru::{MAX_BLOCK_LANES, NUM_GRU_BLOCKS};

use super::super::gru_core::{AdjointWorkspace, backward_gru_block};
use super::super::model_f64::V3MatryoshkaOptF64;
use super::super::primitives::{
    ClampedCoreParams, InputFeatSubTape, SvfChanTape, step_tpt_rev, tan_prewarp_with_deriv,
    upsample_2x_rev,
};

struct SharedCoreRevCarry {
    r_u_lp: f64,
    r_env_fast: f64,
    r_env_slow: f64,
    r_gru_h: [[f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    r_ic1_1: f64,
    r_ic2_1: f64,
    r_ic1_2: f64,
    r_ic2_2: f64,
    r_env: f64,
}

impl Default for SharedCoreRevCarry {
    fn default() -> Self {
        Self {
            r_u_lp: 0.0,
            r_env_fast: 0.0,
            r_env_slow: 0.0,
            r_gru_h: [[0.0_f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
            r_ic1_1: 0.0,
            r_ic2_1: 0.0,
            r_ic1_2: 0.0,
            r_ic2_2: 0.0,
            r_env: 0.0,
        }
    }
}

fn backward_feature_bus_step(
    cp: &ClampedCoreParams,
    ft: &InputFeatSubTape,
    d_feats: &[f64; 5],
    carry: &mut SharedCoreRevCarry,
    grad: &mut V3MatryoshkaOptF64,
) -> (f64, f64, f64) {
    let mut du_s2_direct = d_feats[0] + d_feats[2] * (1.0 - ft.pinch_inv);
    let mut d_ef = d_feats[3] + carry.r_env_fast;
    if cp.pinch_k.0 > 0.0 {
        let dp = ft.feats[0] * ft.pinch_inv * ft.pinch_inv * d_feats[2];
        grad.feat_pinch_k += cp.pinch_k.1 * dp * ft.feats[3] * ft.feats[3];
        d_ef += 2.0 * cp.pinch_k.0 * ft.feats[3] * dp;
    }
    let d_es = d_feats[4] + carry.r_env_slow;
    let sb = if ft.slow_atk_taken {
        cp.slow_atk.0
    } else {
        cp.slow_rel.0
    };
    let d_sb = (ft.rect_slow - ft.env_slow_prev) * d_es;
    if ft.slow_atk_taken {
        grad.feat_slow_atk_beta += cp.slow_atk.1 * d_sb;
    } else {
        grad.feat_slow_rel_beta += cp.slow_rel.1 * d_sb;
    }
    carry.r_env_slow = (1.0 - sb) * d_es;
    let d_rect_slow = sb * d_es;
    grad.feat_slow_det_comp +=
        cp.slow_det_comp.1 * (-0.5 * ft.rect_slow * ft.rect_slow * ft.rect_slow) * d_rect_slow;
    let d_rect_from_slow = ft.slow_rsq3 * d_rect_slow;

    let fb = if ft.fast_atk_taken {
        cp.fast_atk.0
    } else {
        cp.fast_rel.0
    };
    let d_fb = (ft.rect - ft.env_fast_prev) * d_ef;
    if ft.fast_atk_taken {
        grad.feat_fast_atk_beta += cp.fast_atk.1 * d_fb;
    } else {
        grad.feat_fast_rel_beta += cp.fast_rel.1 * d_fb;
    }
    carry.r_env_fast = (1.0 - fb) * d_ef;
    let d_rect = fb * d_ef + d_rect_from_slow;

    du_s2_direct += d_feats[1];
    let d_eff_lp = -d_feats[1];
    let (d_u_lp_from_hp, d_lp2) = if cp.feat_lp2_mix.0 > 0.0 {
        let u_lp_new = ft.u_lp_prev + cp.hp_beta.0 * (ft.feats[0] - ft.u_lp_prev);
        grad.svf_feat_lp2_mix += cp.feat_lp2_mix.1 * (ft.lp2 - u_lp_new) * d_eff_lp;
        (
            (1.0 - cp.feat_lp2_mix.0) * d_eff_lp,
            cp.feat_lp2_mix.0 * d_eff_lp,
        )
    } else {
        (d_eff_lp, 0.0)
    };
    let d_u_lp = d_u_lp_from_hp + carry.r_u_lp;
    grad.feat_in_hp_beta += cp.hp_beta.1 * (ft.feats[0] - ft.u_lp_prev) * d_u_lp;
    du_s2_direct += cp.hp_beta.0 * d_u_lp;
    carry.r_u_lp = (1.0 - cp.hp_beta.0) * d_u_lp;
    (du_s2_direct, d_rect, d_lp2)
}

#[allow(clippy::too_many_arguments)]
fn backward_svf_and_env_step(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    ws: &mut AdjointWorkspace,
    svf_ch: usize,
    ch: usize,
    t: usize,
    sub: usize,
    sub_t: usize,
    du_s2: f64,
    half_link: f64,
    carry: &mut SharedCoreRevCarry,
    grad: &mut V3MatryoshkaOptF64,
) {
    let st: SvfChanTape = ws.svf_sub[ch][sub_t];
    let mut dt = ws.d_sub_taps[ch][sub_t];
    for i in 0..7 {
        grad.svf_stage2_feed_taps[i] += st.taps[i] * du_s2;
        dt[i] += m.svf_stage2_feed_taps[i] * du_s2;
    }

    let (du2, dg2, dk2) = step_tpt_rev(
        dt[4],
        dt[5],
        dt[6],
        st.g2,
        cp.dk2.0,
        st.s2_c1,
        st.s2_v1,
        st.s2_a1,
        st.s2_a2,
        st.s2_a3,
        st.s2_v3,
        &mut carry.r_ic1_2,
        &mut carry.r_ic2_2,
    );
    grad.svf_damp_k2 += cp.dk2.1 * dk2;
    grad.svf_sec2_bp_feed += cp.s2_bp.1 * st.taps[2] * du2;
    grad.svf_sec2_hp_feed += cp.s2_hp.1 * st.taps[3] * du2;
    let (dxs_s1, dg1_raw, dk1) = step_tpt_rev(
        dt[1] + du2,
        dt[2] + cp.s2_bp.0 * du2,
        dt[3] + cp.s2_hp.0 * du2,
        st.g1,
        cp.dk1.0,
        st.s1_c1,
        st.s1_v1,
        st.s1_a1,
        st.s1_a2,
        st.s1_a3,
        st.s1_v3,
        &mut carry.r_ic1_1,
        &mut carry.r_ic2_1,
    );
    grad.svf_damp_k1 += cp.dk1.1 * dk1;
    if ch == 0 {
        ws.d_xl_sub[t][sub] += dt[0] + dxs_s1;
    } else {
        ws.d_xr_sub[t][sub] += dt[0] + dxs_s1;
    }

    let g2_uncl = st.g1 * cp.sec2_r.0;
    let g_hi = f64::from(SVF_G_MAX);
    let dg2_eff = if g2_uncl > 1.0e-4 && g2_uncl < g_hi {
        dg2
    } else {
        0.0
    };
    grad.svf_sec2_g_ratio += cp.sec2_r.1 * st.g1 * dg2_eff;
    let dg1_tot = dg1_raw + cp.sec2_r.0 * dg2_eff;
    let w1_cl = st.w1_unclamped.clamp(cp.g_min.0, cp.g_max.0);
    let (_, dtan_dw) = tan_prewarp_with_deriv(w1_cl);
    let dw1_cl = if st.g1 > 1.0e-4 && st.g1 < g_hi {
        dg1_tot * dtan_dw
    } else {
        0.0
    };
    let dw1_uncl = if st.w1_unclamped <= cp.g_min.0 {
        grad.svf_g_min += cp.g_min.1 * dw1_cl;
        0.0
    } else if st.w1_unclamped >= cp.g_max.0 {
        grad.svf_g_max += cp.g_max.1 * dw1_cl;
        0.0
    } else {
        dw1_cl
    };

    let oct_arg = cp.env_oct.0 * st.e;
    let oct_p = st.oct_p;
    let doct_darg = if (-5.0..=5.0).contains(&oct_arg) {
        oct_p * LN_2
    } else {
        0.0
    };
    grad.svf_base_g += cp.base_g.1 * oct_p * dw1_uncl;
    let d_arg = doct_darg * cp.base_g.0 * dw1_uncl;
    grad.svf_env_oct_sens += cp.env_oct.1 * st.e * d_arg;
    let d_e = cp.env_oct.0 * d_arg;
    grad.svf_env_comp += cp.env_comp.1 * (-0.5 * st.e * st.e * st.e) * d_e;

    let d_env_tot = st.rsq3 * d_e + carry.r_env;
    let dr0 = ws.svf_sub[0][sub_t].v_det_hp.abs();
    let dr1 = if svf_ch < 2 {
        dr0
    } else {
        ws.svf_sub[1][sub_t].v_det_hp.abs()
    };
    let det_ch = if ch == 0 {
        dr0 + half_link * (dr1 - dr0)
    } else {
        dr1 - half_link * (dr1 - dr0)
    };
    let alpha = if st.atk_taken { cp.atk.0 } else { cp.rel.0 };
    let d_alpha = (st.env_prev - det_ch) * d_env_tot;
    if st.atk_taken {
        grad.svf_atk_alpha += cp.atk.1 * d_alpha;
    } else {
        grad.svf_rel_alpha += cp.rel.1 * d_alpha;
    }
    carry.r_env = alpha * d_env_tot;
    ws.d_det_sub[ch][sub_t] = (1.0 - alpha) * d_env_tot;
}

#[inline(always)]
fn backward_detector_chan_step(
    cp: &ClampedCoreParams,
    tp: &SvfChanTape,
    d_dr_ch: f64,
    r_det_lp_ch: &mut f64,
    grad: &mut V3MatryoshkaOptF64,
) -> f64 {
    let d_vh = tp.det_raw_sign * d_dr_ch;
    grad.svf_det_hp_mix += cp.det_hp_mix.1 * (tp.taps[0] - tp.det_lp) * d_vh;
    let d_dlp = -cp.det_hp_mix.0 * d_vh + *r_det_lp_ch;
    grad.svf_det_lp_pole += cp.det_lp.1 * (tp.det_lp_prev - tp.taps[0]) * d_dlp;
    *r_det_lp_ch = cp.det_lp.0 * d_dlp;
    (1.0 + cp.det_hp_mix.0) * d_vh + (1.0 - cp.det_lp.0) * d_dlp
}

fn backward_stereo_detector_and_upsampler(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    n: usize,
    svf_ch: usize,
    half_link: f64,
    ws: &mut AdjointWorkspace,
    grad: &mut V3MatryoshkaOptF64,
) {
    let sub_steps = if m.os2x { 2 } else { 1 };
    let is_mono = svf_ch < 2;
    let (mut r_raw_prev, mut r_det_lp, mut r_up1, mut r_up2) =
        ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut r_up_hb = [[0.0_f64; 4]; 2];
    for t in (0..n).rev() {
        for sub in (0..sub_steps).rev() {
            let sub_t = if m.os2x { t * 2 + sub } else { t };
            let s0 = ws.svf_sub[0][sub_t];
            if is_mono {
                ws.d_xl_sub[t][sub] += backward_detector_chan_step(
                    cp,
                    &s0,
                    ws.d_det_sub[0][sub_t],
                    &mut r_det_lp[0],
                    grad,
                );
            } else {
                let d_det0 = ws.d_det_sub[0][sub_t];
                let d_det1 = ws.d_det_sub[1][sub_t];
                let s1 = ws.svf_sub[1][sub_t];
                grad.svf_stereo_link +=
                    cp.link.1 * 0.5 * (s1.v_det_hp.abs() - s0.v_det_hp.abs()) * (d_det0 - d_det1);
                let d_dr0 = (1.0 - half_link) * d_det0 + half_link * d_det1;
                let d_dr1 = half_link * d_det0 + (1.0 - half_link) * d_det1;
                ws.d_xl_sub[t][sub] +=
                    backward_detector_chan_step(cp, &s0, d_dr0, &mut r_det_lp[0], grad);
                ws.d_xr_sub[t][sub] +=
                    backward_detector_chan_step(cp, &s1, d_dr1, &mut r_det_lp[1], grad);
            }
        }
        if m.os2x {
            let fl = &ws.feat_sub[0][t * 2];
            let (_, duq_l, duz_l, dpe_l, dua_l) = upsample_2x_rev(
                ws.d_xl_sub[t][0],
                ws.d_xl_sub[t][1],
                fl.up_d2,
                fl.up_zoh_diff,
                fl.up_dxp_diff,
                fl.up_hb_diff,
                cp.up_quad.0,
                cp.up_zoh_mix.0,
                cp.up_pre_emph.0,
                cp.up_aa_mix.0,
                &mut r_raw_prev[0],
                &mut r_up1[0],
                &mut r_up2[0],
                &mut r_up_hb[0],
            );
            grad.os_up_quad += cp.up_quad.1 * duq_l;
            grad.os_up_zoh_mix += cp.up_zoh_mix.1 * duz_l;
            grad.os_up_pre_emph += cp.up_pre_emph.1 * dpe_l;
            grad.os_up_aa_mix += cp.up_aa_mix.1 * dua_l;
            if !is_mono {
                let fr = &ws.feat_sub[1][t * 2];
                let (_, duq_r, duz_r, dpe_r, dua_r) = upsample_2x_rev(
                    ws.d_xr_sub[t][0],
                    ws.d_xr_sub[t][1],
                    fr.up_d2,
                    fr.up_zoh_diff,
                    fr.up_dxp_diff,
                    fr.up_hb_diff,
                    cp.up_quad.0,
                    cp.up_zoh_mix.0,
                    cp.up_pre_emph.0,
                    cp.up_aa_mix.0,
                    &mut r_raw_prev[1],
                    &mut r_up1[1],
                    &mut r_up2[1],
                    &mut r_up_hb[1],
                );
                grad.os_up_quad += cp.up_quad.1 * duq_r;
                grad.os_up_zoh_mix += cp.up_zoh_mix.1 * duz_r;
                grad.os_up_pre_emph += cp.up_pre_emph.1 * dpe_r;
                grad.os_up_aa_mix += cp.up_aa_mix.1 * dua_r;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn backward_shared_core(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    n: usize,
    c_blk: [usize; NUM_GRU_BLOCKS],
    active_ch: usize,
    ws: &mut AdjointWorkspace,
    grad: &mut V3MatryoshkaOptF64,
) {
    let sub_steps = if m.os2x { 2 } else { 1 };
    let total_sub = n * sub_steps;
    let svf_ch = if !m.mono_out || cp.key_link.0 > 0.0 {
        2
    } else {
        1
    };
    ws.d_xl_sub[..n].fill([0.0; 2]);
    if svf_ch >= 2 {
        ws.d_xr_sub[..n].fill([0.0; 2]);
        if active_ch < 2 {
            ws.d_sub_taps[1][..total_sub].fill([0.0; 7]);
        }
    }
    let half_link = 0.5 * cp.link.0;
    let mut carries = [SharedCoreRevCarry::default(), SharedCoreRevCarry::default()];

    for t in (0..n).rev() {
        for sub in (0..sub_steps).rev() {
            let sub_t = if m.os2x { t * 2 + sub } else { t };
            let mut du_s2_ch = [0.0_f64; 2];
            let mut d_rect_ch = [0.0_f64; 2];
            let mut d_lp2_ch = [0.0_f64; 2];
            for ch in 0..active_ch {
                let ft = ws.feat_sub[ch][sub_t];
                let mut d_feats = [ws.d_sub_u[ch][sub_t], ws.d_sub_hp[ch][sub_t], 0.0, 0.0, 0.0];
                let mut d_prev_h = [0.0_f64; MAX_BLOCK_LANES];
                for b in (0..NUM_GRU_BLOCKS).rev() {
                    let h_old_block = sub_t
                        .checked_sub(1)
                        .map(|p| &ws.sub_h[ch][p][b * MAX_BLOCK_LANES..(b + 1) * MAX_BLOCK_LANES]);
                    let prev_h = (b > 0).then(|| {
                        &ws.sub_h[ch][sub_t][(b - 1) * MAX_BLOCK_LANES..b * MAX_BLOCK_LANES]
                    });
                    backward_gru_block(
                        m,
                        cp,
                        b,
                        c_blk[b],
                        &ws.gru_sub[ch][sub_t],
                        h_old_block,
                        prev_h,
                        &ft.feats,
                        &ws.d_sub_h[ch][sub_t],
                        &mut carries[ch].r_gru_h[b],
                        &mut d_prev_h,
                        &mut d_feats,
                        grad,
                    );
                }

                let (du_direct, d_rect, d_lp2) =
                    backward_feature_bus_step(cp, &ft, &d_feats, &mut carries[ch], grad);
                du_s2_ch[ch] = du_direct;
                d_rect_ch[ch] = d_rect;
                d_lp2_ch[ch] = d_lp2;
            }

            let mut d_r_mix = [0.0_f64; 2];
            for ch in 0..active_ch {
                let dr = d_rect_ch[ch];
                if cp.det_thresh.0 > 0.0 {
                    if ws.feat_sub[ch][sub_t].r_mix_above_thresh {
                        grad.feat_det_thresh += cp.det_thresh.1 * (-dr);
                        d_r_mix[ch] = dr;
                    }
                } else {
                    d_r_mix[ch] = dr;
                }
            }
            let (d_al, d_ar) = if cp.key_link.0 > 0.0 {
                let kl = cp.key_link.0;
                let al = ws.feat_sub[0][sub_t].raw_abs_u;
                let ar = if svf_ch >= 2 {
                    ws.feat_sub[1][sub_t].raw_abs_u
                } else {
                    al
                };
                let mut d_kl = d_r_mix[0] * (ar - al);
                if active_ch >= 2 {
                    d_kl += d_r_mix[1] * (al - ar);
                }
                grad.feat_key_link += cp.key_link.1 * d_kl;
                (
                    (1.0 - kl) * d_r_mix[0] + kl * d_r_mix[1],
                    kl * d_r_mix[0] + (1.0 - kl) * d_r_mix[1],
                )
            } else {
                (d_r_mix[0], d_r_mix[1])
            };
            if svf_ch == 1 {
                du_s2_ch[0] += ws.feat_sub[0][sub_t].sign_u * (d_al + d_ar);
            } else {
                let d_abs_lr = [d_al, d_ar];
                for ch in 0..svf_ch {
                    du_s2_ch[ch] += ws.feat_sub[ch][sub_t].sign_u * d_abs_lr[ch];
                }
            }
            for ch in 0..svf_ch {
                if ch < active_ch {
                    ws.d_sub_taps[ch][sub_t][4] += d_lp2_ch[ch];
                }
                backward_svf_and_env_step(
                    m,
                    cp,
                    ws,
                    svf_ch,
                    ch,
                    t,
                    sub,
                    sub_t,
                    du_s2_ch[ch],
                    half_link,
                    &mut carries[ch],
                    grad,
                );
            }
        }
    }

    backward_stereo_detector_and_upsampler(m, cp, n, svf_ch, half_link, ws, grad);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::stages::V3MatryoshkaModel;

    #[test]
    fn test_backward_feature_bus_step_zero_adjoint_is_zero() {
        let cold = V3MatryoshkaModel::identity(false, false);
        let m = V3MatryoshkaOptF64::from_matryoshka_cold(&cold);
        let cp = ClampedCoreParams::from_model(&m);
        let ft = InputFeatSubTape::default();
        let d_feats = [0.0; 5];
        let mut carry = SharedCoreRevCarry::default();
        let mut grad = V3MatryoshkaOptF64::from_matryoshka_cold(&cold);
        let init_hp_beta = grad.feat_in_hp_beta;
        let (du_s2, d_rect, d_lp2) =
            backward_feature_bus_step(&cp, &ft, &d_feats, &mut carry, &mut grad);
        assert_eq!(du_s2, 0.0);
        assert_eq!(d_rect, 0.0);
        assert_eq!(d_lp2, 0.0);
        assert_eq!(grad.feat_in_hp_beta, init_hp_beta);
    }
}
