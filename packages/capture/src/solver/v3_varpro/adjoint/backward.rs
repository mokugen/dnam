#![forbid(unsafe_code)]
//! Reverse-mode BPTT sweeps through the tier readouts, downsampler, SVF, GRU cascade, and upsampler.

use dnam_engine::stages::v3_core::cascade::{MAX_BLOCK_LANES, MAX_KAUTZ_POLES, NUM_GRU_BLOCKS};

use crate::solver::backends::TrainerBackend;

use super::super::gru_core::{AdjointWorkspace, backward_gru_block};
use super::super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64};
use super::super::primitives::{
    ClampedCoreParams, InputFeatSubTape, SvfChanTape, downsample_2x_rev, exp2_oct_with_deriv,
    step_tpt_rev, tan_prewarp_with_deriv, upsample_2x_rev,
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

#[allow(clippy::too_many_arguments)]
fn backward_substep_readout(
    r: &V3TierReadoutF64,
    gr: &mut V3TierReadoutF64,
    ws: &mut AdjointWorkspace,
    ch: usize,
    sub_t: usize,
    ddir: f64,
    ds2_s: f64,
    c_t: usize,
) {
    let dir_t = if ch == 0 {
        &r.svf_direct_taps_l
    } else {
        &r.svf_direct_taps_r
    };
    let d_dir_t = if ch == 0 {
        &mut gr.svf_direct_taps_l
    } else {
        &mut gr.svf_direct_taps_r
    };
    let st_taps = &ws.sub_taps[ch][sub_t];
    let su = ws.sub_u[ch][sub_t];
    let shp = ws.sub_hp[ch][sub_t];
    let sh = &ws.sub_h[ch][sub_t];
    let dt = &mut ws.d_sub_taps[ch][sub_t];
    for i in 0..7 {
        d_dir_t[i] += st_taps[i] * ddir;
        dt[i] += dir_t[i] * ddir;
    }
    gr.direct_s2_u_gain[ch] += su * ddir;
    gr.direct_s2_hp_gain[ch] += shp * ddir;
    gr.pre_cab_u_gain[ch] += su * ds2_s;
    gr.pre_cab_hp_gain[ch] += shp * ds2_s;
    ws.d_sub_u[ch][sub_t] += r.direct_s2_u_gain[ch] * ddir + r.pre_cab_u_gain[ch] * ds2_s;
    ws.d_sub_hp[ch][sub_t] += r.direct_s2_hp_gain[ch] * ddir + r.pre_cab_hp_gain[ch] * ds2_s;
    let dh = &mut ws.d_sub_h[ch][sub_t];
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for idx in base..base + c_t {
            gr.direct_s2_w[ch][idx] += sh[idx] * ddir;
            gr.pre_cab_w[ch][idx] += sh[idx] * ds2_s;
            dh[idx] += r.direct_s2_w[ch][idx] * ddir + r.pre_cab_w[ch][idx] * ds2_s;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn backward_tier_readout<B: TrainerBackend>(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    tau: usize,
    n: usize,
    active_ch: usize,
    c_t: usize,
    f_t: usize,
    k_t: usize,
    ws: &mut AdjointWorkspace,
    g_mut: &mut V3MatryoshkaOptF64,
) {
    let r = &m.readouts[tau];
    let (mut d_c_prev1, mut d_c0, mut d_aa_mix) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (gr, gk) = (&mut g_mut.readouts[tau], &mut g_mut.kautz_poles);
    for ch in 0..active_ch {
        ws.ds2_tier[ch][..n].fill(0.0);
        let (mut ds_re, mut ds_im) = ([0.0_f64; MAX_KAUTZ_POLES], [0.0_f64; MAX_KAUTZ_POLES]);
        let (mut r_dec_s2, mut r_dec_dir) = (0.0_f64, 0.0_f64);
        let (mut r_dec_s2_aa, mut r_dec_dir_aa) = ([0.0_f64; 4], [0.0_f64; 4]);
        let fir = if ch == 0 { &r.fir_l } else { &r.fir_r };
        let kc = if ch == 0 { &r.kautz_c_l } else { &r.kautz_c_r };
        for t in (0..n).rev() {
            let dy = ws.dy_tier[ch][t];
            let d_fir = if ch == 0 {
                &mut gr.fir_l
            } else {
                &mut gr.fir_r
            };
            for k in 0..f_t.min(t + 1) {
                d_fir[k] += ws.s2_tier[ch][t - k] * dy;
                ws.ds2_tier[ch][t - k] += fir[k] * dy;
            }
            let x_del = if t >= 16 { ws.s2_tier[ch][t - 16] } else { 0.0 };
            let off = t * k_t;
            let (prev_re, prev_im) = if t > 0 {
                let po = (t - 1) * k_t;
                (
                    Some(&ws.kautz_re[tau][ch][po..po + k_t]),
                    Some(&ws.kautz_im[tau][ch][po..po + k_t]),
                )
            } else {
                (None, None)
            };
            let d_kc = if ch == 0 {
                &mut gr.kautz_c_l[..k_t]
            } else {
                &mut gr.kautz_c_r[..k_t]
            };
            let dx_del = B::step_kautz_backward_frame(
                dy,
                x_del,
                &cp.kautz_pre[..k_t],
                &kc[..k_t],
                &ws.kautz_re[tau][ch][off..off + k_t],
                &ws.kautz_im[tau][ch][off..off + k_t],
                prev_re,
                prev_im,
                &mut ds_re[..k_t],
                &mut ds_im[..k_t],
                d_kc,
                &mut gk[..k_t],
            );
            if t >= 16 {
                ws.ds2_tier[ch][t - 16] += dx_del;
            }
            let ds2 = ws.ds2_tier[ch][t];

            let (sub_pair, n_sub) = if !m.os2x {
                ([(t, dy, ds2), (0, 0.0, 0.0)], 1)
            } else {
                let (s0, s1) = (t * 2, t * 2 + 1);
                let sp_prev = t.checked_sub(1).map(|p| p * 2 + 1);
                let old_dir = sp_prev.map_or(0.0, |p| ws.dir_sub_tier[ch][p]);
                let old_s2 = sp_prev.map_or(0.0, |p| ws.s2_sub_tier[ch][p]);
                let (dd0, dd1, dcp_d, dc0_d, daa_d) = downsample_2x_rev(
                    dy,
                    old_dir,
                    ws.dir_sub_tier[ch][s0],
                    ws.dir_sub_tier[ch][s1],
                    ws.dir_aa_diff[ch][t],
                    cp.dec_c_prev1.0,
                    cp.dec_c0.0,
                    cp.dec_aa_mix.0,
                    &mut r_dec_dir,
                    &mut r_dec_dir_aa,
                );
                let (ds0, ds1, dcp_s, dc0_s, daa_s) = downsample_2x_rev(
                    ds2,
                    old_s2,
                    ws.s2_sub_tier[ch][s0],
                    ws.s2_sub_tier[ch][s1],
                    ws.s2_aa_diff[ch][t],
                    cp.dec_c_prev1.0,
                    cp.dec_c0.0,
                    cp.dec_aa_mix.0,
                    &mut r_dec_s2,
                    &mut r_dec_s2_aa,
                );
                d_c_prev1 += dcp_d + dcp_s;
                d_c0 += dc0_d + dc0_s;
                d_aa_mix += daa_d + daa_s;
                ([(s0, dd0, ds0), (s1, dd1, ds1)], 2)
            };

            for &(sub_t, ddir, ds2_s) in &sub_pair[..n_sub] {
                backward_substep_readout(r, gr, ws, ch, sub_t, ddir, ds2_s, c_t);
            }
        }
    }
    g_mut.os_dec_c_prev1 += cp.dec_c_prev1.1 * d_c_prev1;
    g_mut.os_dec_c0 += cp.dec_c0.1 * d_c0;
    g_mut.os_dec_aa_mix += cp.dec_aa_mix.1 * d_aa_mix;
}

fn backward_feature_bus_step(
    cp: &ClampedCoreParams,
    ft: &InputFeatSubTape,
    d_feats: &[f64; 5],
    carry: &mut SharedCoreRevCarry,
    grad: &mut V3MatryoshkaOptF64,
) -> f64 {
    let mut du_s2 = d_feats[0] + d_feats[2] * (1.0 - ft.pinch_inv);
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
    let d_fb = (ft.feats[0].abs() - ft.env_fast_prev) * d_ef;
    if ft.fast_atk_taken {
        grad.feat_fast_atk_beta += cp.fast_atk.1 * d_fb;
    } else {
        grad.feat_fast_rel_beta += cp.fast_rel.1 * d_fb;
    }
    carry.r_env_fast = (1.0 - fb) * d_ef;
    du_s2 += ft.sign_u * (fb * d_ef + d_rect_from_slow) + d_feats[1];
    let d_u_lp = -d_feats[1] + carry.r_u_lp;
    grad.feat_in_hp_beta += cp.hp_beta.1 * (ft.feats[0] - ft.u_lp_prev) * d_u_lp;
    du_s2 += cp.hp_beta.0 * d_u_lp;
    carry.r_u_lp = (1.0 - cp.hp_beta.0) * d_u_lp;
    du_s2
}

#[allow(clippy::too_many_arguments)]
fn backward_svf_and_env_step(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    ws: &mut AdjointWorkspace,
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
    let dg2_eff = if g2_uncl > 1.0e-4 && g2_uncl < 1.45 {
        dg2
    } else {
        0.0
    };
    grad.svf_sec2_g_ratio += cp.sec2_r.1 * st.g1 * dg2_eff;
    let dg1_tot = dg1_raw + cp.sec2_r.0 * dg2_eff;
    let (_, dtan_dw) = tan_prewarp_with_deriv(st.w1_clamped);
    let dw1_cl = if st.g1 > 1.0e-4 && st.g1 < 1.45 {
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

    let (oct_p, doct_darg) = exp2_oct_with_deriv(cp.env_oct.0 * st.e);
    grad.svf_base_g += cp.base_g.1 * oct_p * dw1_uncl;
    let d_arg = doct_darg * cp.base_g.0 * dw1_uncl;
    grad.svf_env_oct_sens += cp.env_oct.1 * st.e * d_arg;
    let d_e = cp.env_oct.0 * d_arg;
    grad.svf_env_comp += cp.env_comp.1 * (-0.5 * st.e * st.e * st.e) * d_e;

    let d_env_tot = st.rsq3 * d_e + carry.r_env;
    let dr0 = ws.svf_sub[0][sub_t].v_det_hp.abs();
    let dr1 = if m.mono_out {
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

fn backward_stereo_detector_and_upsampler(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    n: usize,
    active_ch: usize,
    half_link: f64,
    ws: &mut AdjointWorkspace,
    grad: &mut V3MatryoshkaOptF64,
) {
    let sub_steps = if m.os2x { 2 } else { 1 };
    let (mut r_raw_prev, mut r_det_lp, mut r_up1, mut r_up2) =
        ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut r_up_hb = [[0.0_f64; 4]; 2];
    for t in (0..n).rev() {
        for sub in (0..sub_steps).rev() {
            let sub_t = if m.os2x { t * 2 + sub } else { t };
            let (d_det0, d_det1) = (
                ws.d_det_sub[0][sub_t],
                if active_ch == 2 {
                    ws.d_det_sub[1][sub_t]
                } else {
                    0.0
                },
            );
            let s0 = ws.svf_sub[0][sub_t];
            let s1 = if m.mono_out { s0 } else { ws.svf_sub[1][sub_t] };
            grad.svf_stereo_link +=
                cp.link.1 * 0.5 * (s1.v_det_hp.abs() - s0.v_det_hp.abs()) * (d_det0 - d_det1);
            let d_dr = [
                (1.0 - half_link) * d_det0 + half_link * d_det1,
                half_link * d_det0 + (1.0 - half_link) * d_det1,
            ];
            for (d_ch, tp) in [s0, s1].iter().enumerate() {
                let d_vh = tp.det_raw_sign * d_dr[d_ch];
                grad.svf_det_hp_mix += cp.det_hp_mix.1 * (tp.taps[0] - tp.det_lp) * d_vh;
                let d_dlp = -cp.det_hp_mix.0 * d_vh + r_det_lp[d_ch];
                grad.svf_det_lp_pole += cp.det_lp.1 * (tp.det_lp_prev - tp.taps[0]) * d_dlp;
                r_det_lp[d_ch] = cp.det_lp.0 * d_dlp;
                let d_xs = (1.0 + cp.det_hp_mix.0) * d_vh + (1.0 - cp.det_lp.0) * d_dlp;
                if m.mono_out || d_ch == 0 {
                    ws.d_xl_sub[t][sub] += d_xs;
                } else {
                    ws.d_xr_sub[t][sub] += d_xs;
                }
            }
        }
        if m.os2x {
            let fl = &ws.feat_sub[0][t * 2];
            let (_, duq_l, dpe_l, dua_l) = upsample_2x_rev(
                ws.d_xl_sub[t][0],
                ws.d_xl_sub[t][1],
                fl.up_d2,
                fl.up_dxp_diff,
                fl.up_hb_diff,
                cp.up_quad.0,
                cp.up_pre_emph.0,
                cp.up_aa_mix.0,
                &mut r_raw_prev[0],
                &mut r_up1[0],
                &mut r_up2[0],
                &mut r_up_hb[0],
            );
            grad.os_up_quad += cp.up_quad.1 * duq_l;
            grad.os_up_pre_emph += cp.up_pre_emph.1 * dpe_l;
            grad.os_up_aa_mix += cp.up_aa_mix.1 * dua_l;
            if !m.mono_out {
                let fr = &ws.feat_sub[1][t * 2];
                let (_, duq_r, dpe_r, dua_r) = upsample_2x_rev(
                    ws.d_xr_sub[t][0],
                    ws.d_xr_sub[t][1],
                    fr.up_d2,
                    fr.up_dxp_diff,
                    fr.up_hb_diff,
                    cp.up_quad.0,
                    cp.up_pre_emph.0,
                    cp.up_aa_mix.0,
                    &mut r_raw_prev[1],
                    &mut r_up1[1],
                    &mut r_up2[1],
                    &mut r_up_hb[1],
                );
                grad.os_up_quad += cp.up_quad.1 * duq_r;
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
    c_blk: usize,
    active_ch: usize,
    ws: &mut AdjointWorkspace,
    grad: &mut V3MatryoshkaOptF64,
) {
    let sub_steps = if m.os2x { 2 } else { 1 };
    ws.d_xl_sub[..n].fill([0.0; 2]);
    ws.d_xr_sub[..n].fill([0.0; 2]);
    let half_link = 0.5 * cp.link.0;

    for ch in 0..active_ch {
        let mut carry = SharedCoreRevCarry::default();
        for t in (0..n).rev() {
            for sub in (0..sub_steps).rev() {
                let sub_t = if m.os2x { t * 2 + sub } else { t };
                let ft = ws.feat_sub[ch][sub_t];
                let mut d_feats = [ws.d_sub_u[ch][sub_t], ws.d_sub_hp[ch][sub_t], 0.0, 0.0, 0.0];
                let mut d_prev_h = [0.0_f64; MAX_BLOCK_LANES];
                for b in (0..NUM_GRU_BLOCKS).rev() {
                    let prev_h = (b > 0).then(|| &ws.gru_sub[ch][sub_t][b - 1].h_new);
                    backward_gru_block(
                        m,
                        cp,
                        b,
                        c_blk,
                        &ws.gru_sub[ch][sub_t][b],
                        prev_h,
                        &ft.feats,
                        &ws.d_sub_h[ch][sub_t],
                        &mut carry.r_gru_h[b],
                        &mut d_prev_h,
                        &mut d_feats,
                        grad,
                    );
                }

                let du_s2 = backward_feature_bus_step(cp, &ft, &d_feats, &mut carry, grad);
                backward_svf_and_env_step(
                    m, cp, ws, ch, t, sub, sub_t, du_s2, half_link, &mut carry, grad,
                );
            }
        }
    }

    backward_stereo_detector_and_upsampler(m, cp, n, active_ch, half_link, ws, grad);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::stages::v3_core::V3MatryoshkaModel;

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
        let du_s2 = backward_feature_bus_step(&cp, &ft, &d_feats, &mut carry, &mut grad);
        assert_eq!(du_s2, 0.0);
        assert_eq!(grad.feat_in_hp_beta, init_hp_beta);
    }
}
