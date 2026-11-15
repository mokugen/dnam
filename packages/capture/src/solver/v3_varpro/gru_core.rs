#![forbid(unsafe_code)]
//! Shared `Zero-Lag 2x/1x` + `Stage I Env-SVF` + `Stage II 4-Lane-Group Vectorized GRU` Core & Reverse Block.

use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
};

use super::gabor::GaborAdjointScratch;
use super::ltas_adj::Welch33Scratch;
use super::model_f64::V3MatryoshkaOptF64;
use super::primitives::{
    ClampedCoreParams, GruBlockSubTape, InputFeatSubTape, SvfChanTape, downsample_2x_fwd,
    exp2_oct_with_deriv, sigmoid_pade5_with_deriv, step_tpt_fwd, tan_prewarp_with_deriv,
    tanh_pade5_with_deriv, upsample_2x_fwd,
};

/// Reusable workspace avoiding heap reallocations across L-BFGS iterations and VarPro calls.
#[derive(Debug, Default)]
pub struct AdjointWorkspace {
    pub(crate) tap_dec: [Vec<[f64; 7]>; 2],
    pub(crate) u_dec: [Vec<f64>; 2],
    pub(crate) hp_dec: [Vec<f64>; 2],
    pub(crate) h_dec: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
    pub(crate) sub_taps: [Vec<[f64; 7]>; 2],
    pub(crate) sub_u: [Vec<f64>; 2],
    pub(crate) sub_hp: [Vec<f64>; 2],
    pub(crate) sub_h: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
    pub(crate) d_sub_taps: [Vec<[f64; 7]>; 2],
    pub(crate) d_sub_u: [Vec<f64>; 2],
    pub(crate) d_sub_hp: [Vec<f64>; 2],
    pub(crate) d_sub_h: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
    pub(crate) s2_sub_tier: [Vec<f64>; 2],
    pub(crate) dir_sub_tier: [Vec<f64>; 2],
    pub(crate) s2_aa_diff: [Vec<f64>; 2],
    pub(crate) dir_aa_diff: [Vec<f64>; 2],
    pub(crate) svf_sub: [Vec<SvfChanTape>; 2],
    pub(crate) feat_sub: [Vec<InputFeatSubTape>; 2],
    pub(crate) gru_sub: [Vec<[GruBlockSubTape; NUM_GRU_BLOCKS]>; 2],
    pub(crate) s2_tier: [Vec<f64>; 2],
    pub(crate) y_tier: [Vec<f64>; 2],
    pub(crate) dy_tier: [Vec<f64>; 2],
    pub(crate) ds2_tier: [Vec<f64>; 2],
    pub(crate) err_buf: Vec<f64>,
    pub(crate) bezout_scratch: [Vec<f64>; 3],
    pub(crate) kautz_re: [[Vec<f64>; 2]; 4],
    pub(crate) kautz_im: [[Vec<f64>; 2]; 4],
    pub(crate) d_det_sub: [Vec<f64>; 2],
    pub(crate) d_xl_sub: Vec<[f64; 2]>,
    pub(crate) d_xr_sub: Vec<[f64; 2]>,
    pub(crate) gabor_scratch: GaborAdjointScratch,
    pub(crate) welch_scratch: Welch33Scratch,
}

impl AdjointWorkspace {
    pub(crate) fn ensure_capacity(
        &mut self,
        n: usize,
        sub_steps: usize,
        tiers_needed: &[usize; 4],
        need_grad: bool,
        decimate_all_lanes: bool,
    ) {
        let total_sub = n * sub_steps;
        for ch in 0..2 {
            if decimate_all_lanes && self.tap_dec[ch].len() < n {
                self.tap_dec[ch].resize(n, [0.0; 7]);
                self.u_dec[ch].resize(n, 0.0);
                self.hp_dec[ch].resize(n, 0.0);
                self.h_dec[ch].resize(n, [0.0; MAX_TOTAL_GRU_LANES]);
            }
            if !decimate_all_lanes {
                if self.sub_taps[ch].len() < total_sub {
                    self.sub_taps[ch].resize(total_sub, [0.0; 7]);
                    self.sub_u[ch].resize(total_sub, 0.0);
                    self.sub_hp[ch].resize(total_sub, 0.0);
                    self.sub_h[ch].resize(total_sub, [0.0; MAX_TOTAL_GRU_LANES]);
                }
                if self.s2_tier[ch].len() < n {
                    self.s2_tier[ch].resize(n, 0.0);
                    self.y_tier[ch].resize(n, 0.0);
                }
            }
            if need_grad {
                if self.svf_sub[ch].len() < total_sub {
                    self.svf_sub[ch].resize(total_sub, SvfChanTape::default());
                    self.feat_sub[ch].resize(total_sub, InputFeatSubTape::default());
                    self.gru_sub[ch]
                        .resize(total_sub, [GruBlockSubTape::default(); NUM_GRU_BLOCKS]);
                    self.d_det_sub[ch].resize(total_sub, 0.0);
                    self.d_sub_taps[ch].resize(total_sub, [0.0; 7]);
                    self.d_sub_u[ch].resize(total_sub, 0.0);
                    self.d_sub_hp[ch].resize(total_sub, 0.0);
                    self.d_sub_h[ch].resize(total_sub, [0.0; MAX_TOTAL_GRU_LANES]);
                    self.s2_sub_tier[ch].resize(total_sub, 0.0);
                    self.dir_sub_tier[ch].resize(total_sub, 0.0);
                }
                if self.dy_tier[ch].len() < n {
                    self.dy_tier[ch].resize(n, 0.0);
                    self.ds2_tier[ch].resize(n, 0.0);
                }
                if sub_steps == 2 && self.s2_aa_diff[ch].len() < n {
                    self.s2_aa_diff[ch].resize(n, 0.0);
                    self.dir_aa_diff[ch].resize(n, 0.0);
                }
            }
        }
        if self.err_buf.len() < n {
            self.err_buf.resize(n, 0.0);
        }
        if need_grad {
            if self.d_xl_sub.len() < n {
                self.d_xl_sub.resize(n, [0.0; 2]);
                self.d_xr_sub.resize(n, [0.0; 2]);
            }
            for tau in 0..4 {
                let need = n * tiers_needed[tau];
                for ch in 0..2 {
                    if self.kautz_re[tau][ch].len() < need {
                        self.kautz_re[tau][ch].resize(need, 0.0);
                        self.kautz_im[tau][ch].resize(need, 0.0);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn dot4(w: &[f64; MAX_BLOCK_LANES], v: &[f64; MAX_BLOCK_LANES], s: usize) -> f64 {
    w[s] * v[s] + w[s + 1] * v[s + 1] + w[s + 2] * v[s + 2] + w[s + 3] * v[s + 3]
}

/// Evaluates the shared `Zero-Lag 2x/1x` -> `Stage I Env-SVF` -> `Stage II 3-Block GRU` core once per stem.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_shared_core(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    in_l: &[f32],
    in_r: &[f32],
    n: usize,
    c_blk: usize,
    active_ch: usize,
    record_tape: bool,
    decimate_all_lanes: bool,
    ws: &mut AdjointWorkspace,
) {
    let sub_steps = if m.os2x { 2 } else { 1 };
    let g_cnt = (c_blk >> 2).clamp(1, MAX_GROUPS);
    let (mut up_raw_prev, mut up_p1, mut up_p2) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut up_hb = [[0.0_f64; 4]; 2];
    let (mut det_lp, mut env) = ([0.0_f64; 2], [0.0_f64; 2]);
    let (mut ic1_1, mut ic2_1, mut ic1_2, mut ic2_2) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
    let (mut u_lp, mut env_fast, mut env_slow) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut h_state = [[[0.0_f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS]; 2];
    let (mut dec_tap_p, mut dec_u_p, mut dec_hp_p) =
        ([[0.0_f64; 7]; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let mut dec_h_p = [[0.0_f64; MAX_TOTAL_GRU_LANES]; 2];
    let (mut dec_tap_aa, mut dec_u_aa, mut dec_hp_aa) =
        ([[[0.0_f64; 4]; 7]; 2], [[0.0_f64; 4]; 2], [[0.0_f64; 4]; 2]);
    let mut dec_h_aa = [[[0.0_f64; 4]; MAX_TOTAL_GRU_LANES]; 2];
    let half_link = 0.5 * cp.link.0;

    for t in 0..n {
        let xl = f64::from(in_l[t]);
        let xr = if m.mono_out { xl } else { f64::from(in_r[t]) };
        let (xs_sub, d2_lr, dxp_diff_lr, up_hb_diff_lr) = if m.os2x {
            let (l_pair, d2l, dxpl, hbl) = upsample_2x_fwd(
                xl,
                cp.up_quad.0,
                cp.up_pre_emph.0,
                cp.up_aa_mix.0,
                &mut up_raw_prev[0],
                &mut up_p1[0],
                &mut up_p2[0],
                &mut up_hb[0],
                record_tape,
            );
            let (r_pair, d2r, dxpr, hbr) = if m.mono_out {
                (l_pair, d2l, dxpl, hbl)
            } else {
                upsample_2x_fwd(
                    xr,
                    cp.up_quad.0,
                    cp.up_pre_emph.0,
                    cp.up_aa_mix.0,
                    &mut up_raw_prev[1],
                    &mut up_p1[1],
                    &mut up_p2[1],
                    &mut up_hb[1],
                    record_tape,
                )
            };
            (
                [[l_pair[0], r_pair[0]], [l_pair[1], r_pair[1]]],
                [d2l, d2r],
                [dxpl, dxpr],
                [hbl, hbr],
            )
        } else {
            (
                [[xl, xr], [0.0, 0.0]],
                [0.0, 0.0],
                [0.0, 0.0],
                [[0.0; 2], [0.0; 2]],
            )
        };

        let mut tap_sub = [[[0.0_f64; 7]; 2]; 2];
        let (mut u_sub, mut hp_sub) = ([[0.0_f64; 2]; 2], [[0.0_f64; 2]; 2]);
        let mut h_sub = [[[0.0_f64; MAX_TOTAL_GRU_LANES]; 2]; 2];

        for sub in 0..sub_steps {
            let sub_t = if m.os2x { t * 2 + sub } else { t };
            let xs = xs_sub[sub];
            let (mut dr, mut det_lp_prev, mut v_det_hp, mut raw_sign) =
                ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
            for ch in 0..2 {
                det_lp_prev[ch] = det_lp[ch];
                det_lp[ch] = cp.det_lp.0 * det_lp[ch] + (1.0 - cp.det_lp.0) * xs[ch];
                let vh = xs[ch] + cp.det_hp_mix.0 * (xs[ch] - det_lp[ch]);
                v_det_hp[ch] = vh;
                dr[ch] = vh.abs();
                raw_sign[ch] = if vh > 0.0 {
                    1.0
                } else if vh < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
            let diff = dr[1] - dr[0];
            let det = [dr[0] + half_link * diff, dr[1] - half_link * diff];

            for ch in 0..active_ch {
                let env_prev = env[ch];
                let atk_taken = det[ch] > env_prev;
                let alpha = if atk_taken { cp.atk.0 } else { cp.rel.0 };
                env[ch] = alpha * env_prev + (1.0 - alpha) * det[ch];
                let rsq = 1.0 / (1.0 + cp.env_comp.0 * env[ch] * env[ch]).sqrt();
                let e = env[ch] * rsq;
                let (oct_p, _) = exp2_oct_with_deriv(cp.env_oct.0 * e);
                let w1_uncl = cp.base_g.0 * oct_p;
                let w1_cl = w1_uncl.clamp(cp.g_min.0, cp.g_max.0);
                let (g1_tan, _) = tan_prewarp_with_deriv(w1_cl);
                let g1 = g1_tan.clamp(1.0e-4, 1.45);
                let g2 = (g1 * cp.sec2_r.0).clamp(1.0e-4, 1.45);

                let (lp1, bp1, hp1, s1_c1, s1_a1, s1_a2, s1_a3, s1_v1, s1_v3) =
                    step_tpt_fwd(xs[ch], g1, cp.dk1.0, &mut ic1_1[ch], &mut ic2_1[ch]);
                let u2 = lp1 + cp.s2_bp.0 * bp1 + cp.s2_hp.0 * hp1;
                let (lp2, bp2, hp2, s2_c1, s2_a1, s2_a2, s2_a3, s2_v1, s2_v3) =
                    step_tpt_fwd(u2, g2, cp.dk2.0, &mut ic1_2[ch], &mut ic2_2[ch]);
                let taps = [xs[ch], lp1, bp1, hp1, lp2, bp2, hp2];
                if decimate_all_lanes {
                    tap_sub[sub][ch] = taps;
                } else {
                    ws.sub_taps[ch][sub_t] = taps;
                }
                if record_tape {
                    let tp = SvfChanTape {
                        taps,
                        det_lp_prev: det_lp_prev[ch],
                        det_lp: det_lp[ch],
                        v_det_hp: v_det_hp[ch],
                        det_raw_sign: raw_sign[ch],
                        env_prev,
                        atk_taken,
                        rsq3: rsq * rsq * rsq,
                        e,
                        w1_unclamped: w1_uncl,
                        w1_clamped: w1_cl,
                        g1,
                        g2,
                        s1_c1,
                        s1_a1,
                        s1_a2,
                        s1_a3,
                        s1_v1,
                        s1_v3,
                        s2_c1,
                        s2_a1,
                        s2_a2,
                        s2_a3,
                        s2_v1,
                        s2_v3,
                    };
                    ws.svf_sub[ch][sub_t] = tp;
                    if m.mono_out {
                        ws.svf_sub[1][sub_t] = tp;
                    }
                }

                let mut u_s2 = 0.0_f64;
                for i in 0..7 {
                    u_s2 += m.svf_stage2_feed_taps[i] * taps[i];
                }
                let u_lp_prev = u_lp[ch];
                u_lp[ch] = u_lp_prev + cp.hp_beta.0 * (u_s2 - u_lp_prev);
                let u_hp = u_s2 - u_lp[ch];
                let rect = u_s2.abs();
                let fast_atk_taken = rect > env_fast[ch];
                let fb = if fast_atk_taken {
                    cp.fast_atk.0
                } else {
                    cp.fast_rel.0
                };
                let env_fast_prev = env_fast[ch];
                env_fast[ch] = env_fast_prev + fb * (rect - env_fast_prev);
                let slow_rsq = 1.0 / (1.0 + cp.slow_det_comp.0 * rect * rect).sqrt();
                let slow_rsq3 = slow_rsq * slow_rsq * slow_rsq;
                let rect_slow = rect * slow_rsq;
                let slow_atk_taken = rect_slow > env_slow[ch];
                let sb = if slow_atk_taken {
                    cp.slow_atk.0
                } else {
                    cp.slow_rel.0
                };
                let env_slow_prev = env_slow[ch];
                env_slow[ch] = env_slow_prev + sb * (rect_slow - env_slow_prev);
                let (u_pinch, pinch_inv) = if cp.pinch_k.0 > 0.0 {
                    let pinv = 1.0 / (1.0 + cp.pinch_k.0 * env_fast[ch] * env_fast[ch]);
                    (u_s2 * (1.0 - pinv), pinv)
                } else {
                    (0.0, 1.0)
                };
                let feats = [u_s2, u_hp, u_pinch, env_fast[ch], env_slow[ch]];
                if decimate_all_lanes {
                    u_sub[sub][ch] = u_s2;
                    hp_sub[sub][ch] = u_hp;
                } else {
                    ws.sub_u[ch][sub_t] = u_s2;
                    ws.sub_hp[ch][sub_t] = u_hp;
                }
                if record_tape {
                    let sign_u = if u_s2 > 0.0 {
                        1.0
                    } else if u_s2 < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    ws.feat_sub[ch][sub_t] = InputFeatSubTape {
                        feats,
                        u_lp_prev,
                        env_fast_prev,
                        env_slow_prev,
                        sign_u,
                        fast_atk_taken,
                        slow_atk_taken,
                        pinch_inv,
                        rect_slow,
                        slow_rsq3,
                        up_d2: d2_lr[ch],
                        up_dxp_diff: dxp_diff_lr[ch],
                        up_hb_diff: up_hb_diff_lr[ch],
                    };
                }

                let mut prev_h = [0.0_f64; MAX_BLOCK_LANES];
                for b in 0..NUM_GRU_BLOCKS {
                    let blk = &m.blocks[b];
                    let h_old = h_state[ch][b];
                    let mut h_new = [0.0_f64; MAX_BLOCK_LANES];
                    if record_tape {
                        ws.gru_sub[ch][sub_t][b].h_old[..c_blk].copy_from_slice(&h_old[..c_blk]);
                    }
                    for dst_g in 0..g_cnt {
                        let d_base = dst_g << 2;
                        for dl in 0..4 {
                            let dst = d_base + dl;
                            let (mut az, mut ar, mut an_in) =
                                (cp.b_z[b][dst].0, cp.b_r[b][dst].0, 0.0_f64);
                            for f in 0..NUM_IN_FEATS_B0 {
                                let fv = feats[f];
                                az += blk.w_in_z[f][dst] * fv;
                                ar += blk.w_in_r[f][dst] * fv;
                                an_in += blk.w_in_n[f][dst] * fv;
                            }
                            if b > 0 {
                                for sg in 0..=dst_g {
                                    let s = sg << 2;
                                    az += dot4(&blk.w_prev_z[dst], &prev_h, s);
                                    ar += dot4(&blk.w_prev_r[dst], &prev_h, s);
                                    an_in += dot4(&blk.w_prev_n[dst], &prev_h, s);
                                }
                            }
                            let mut rec_n = 0.0_f64;
                            for sg in 0..=dst_g {
                                let s = sg << 2;
                                az += dot4(&blk.w_rec_z[dst], &h_old, s);
                                ar += dot4(&blk.w_rec_r[dst], &h_old, s);
                                rec_n += dot4(&blk.w_rec_n[dst], &h_old, s);
                            }
                            let (z, dz) = sigmoid_pade5_with_deriv(az);
                            let (r, dr) = sigmoid_pade5_with_deriv(ar);
                            let a_sig_raw = an_in + rec_n;
                            let a_sig = a_sig_raw.clamp(-12.0, 12.0);
                            let a_sig_mask = if a_sig_raw > -12.0 && a_sig_raw < 12.0 {
                                1.0
                            } else {
                                0.0
                            };
                            let a_eff = a_sig + cp.k_quad[b][dst].0 * a_sig * a_sig;
                            let g_sag = 1.0 + cp.k_sag[b][dst].0 * feats[4];
                            let h_sag = 1.0 / g_sag;
                            let an = g_sag * a_eff + cp.b_n[b][dst].0;
                            let (tan_val, dt_an) = tanh_pade5_with_deriv(an);
                            let delta_tanh = tan_val - cp.tanh_bn[b][dst];
                            let nv = h_sag * delta_tanh;
                            let h_raw = (1.0 - z) * h_old[dst] + r * nv;
                            let h_cl = h_raw.clamp(-2.5, 2.5);
                            h_new[dst] = h_cl;
                            if decimate_all_lanes {
                                h_sub[sub][ch][b * MAX_BLOCK_LANES + dst] = h_cl;
                            } else {
                                ws.sub_h[ch][sub_t][b * MAX_BLOCK_LANES + dst] = h_cl;
                            }
                            if record_tape {
                                let tp = &mut ws.gru_sub[ch][sub_t][b];
                                tp.z[dst] = z;
                                tp.dz_daz[dst] = dz;
                                tp.r[dst] = r;
                                tp.dr_dar[dst] = dr;
                                tp.an_in[dst] = an_in;
                                tp.rec_n[dst] = rec_n;
                                tp.a_sig[dst] = a_sig;
                                tp.a_sig_mask[dst] = a_sig_mask;
                                tp.a_eff[dst] = a_eff;
                                tp.h_sag[dst] = h_sag;
                                tp.delta_tanh[dst] = delta_tanh;
                                tp.dtanh_an[dst] = dt_an;
                                tp.dtanh_bn[dst] = cp.dtanh_bn[b][dst];
                                tp.n[dst] = nv;
                                tp.h_new[dst] = h_cl;
                                tp.clamp_mask[dst] = if h_raw > -2.5 && h_raw < 2.5 {
                                    1.0
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                    h_state[ch][b] = h_new;
                    prev_h = h_new;
                }
            }
        }

        if decimate_all_lanes {
            for ch in 0..active_ch {
                if !m.os2x {
                    ws.tap_dec[ch][t] = tap_sub[0][ch];
                    ws.u_dec[ch][t] = u_sub[0][ch];
                    ws.hp_dec[ch][t] = hp_sub[0][ch];
                    ws.h_dec[ch][t] = h_sub[0][ch];
                } else {
                    for i in 0..7 {
                        let (v, _, _) = downsample_2x_fwd(
                            tap_sub[0][ch][i],
                            tap_sub[1][ch][i],
                            &mut dec_tap_p[ch][i],
                            &mut dec_tap_aa[ch][i],
                            cp.dec_c_prev1.0,
                            cp.dec_c0.0,
                            cp.dec_aa_mix.0,
                            false,
                        );
                        ws.tap_dec[ch][t][i] = v;
                    }
                    let (uv, _, _) = downsample_2x_fwd(
                        u_sub[0][ch],
                        u_sub[1][ch],
                        &mut dec_u_p[ch],
                        &mut dec_u_aa[ch],
                        cp.dec_c_prev1.0,
                        cp.dec_c0.0,
                        cp.dec_aa_mix.0,
                        false,
                    );
                    ws.u_dec[ch][t] = uv;
                    let (hv, _, _) = downsample_2x_fwd(
                        hp_sub[0][ch],
                        hp_sub[1][ch],
                        &mut dec_hp_p[ch],
                        &mut dec_hp_aa[ch],
                        cp.dec_c_prev1.0,
                        cp.dec_c0.0,
                        cp.dec_aa_mix.0,
                        false,
                    );
                    ws.hp_dec[ch][t] = hv;
                    for b in 0..NUM_GRU_BLOCKS {
                        let base = b * MAX_BLOCK_LANES;
                        for l in 0..c_blk {
                            let idx = base + l;
                            let (val, _, _) = downsample_2x_fwd(
                                h_sub[0][ch][idx],
                                h_sub[1][ch][idx],
                                &mut dec_h_p[ch][idx],
                                &mut dec_h_aa[ch][idx],
                                cp.dec_c_prev1.0,
                                cp.dec_c0.0,
                                cp.dec_aa_mix.0,
                                false,
                            );
                            ws.h_dec[ch][t][idx] = val;
                        }
                    }
                }
            }
        }
    }
}

/// Reverse-mode adjoint for one lower-triangular 4-lane-grouped GRU block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_gru_block(
    m: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    b: usize,
    c_blk: usize,
    tp: &GruBlockSubTape,
    prev_h: Option<&[f64; MAX_BLOCK_LANES]>,
    feats: &[f64; NUM_IN_FEATS_B0],
    dh_sub: &[f64; MAX_TOTAL_GRU_LANES],
    r_gru_h: &mut [f64; MAX_BLOCK_LANES],
    d_prev_h: &mut [f64; MAX_BLOCK_LANES],
    d_feats: &mut [f64; NUM_IN_FEATS_B0],
    grad: &mut V3MatryoshkaOptF64,
) {
    let blk = &m.blocks[b];
    let gb = &mut grad.blocks[b];
    let base = b * MAX_BLOCK_LANES;
    let g_cnt = (c_blk >> 2).clamp(1, MAX_GROUPS);
    let mut next_r_h = [0.0_f64; MAX_BLOCK_LANES];
    let mut d_prev_out = [0.0_f64; MAX_BLOCK_LANES];

    for dst_g in 0..g_cnt {
        let d_base = dst_g << 2;
        for dl in 0..4 {
            let dst = d_base + dl;
            let d_h = (dh_sub[base + dst] + r_gru_h[dst] + d_prev_h[dst]) * tp.clamp_mask[dst];
            next_r_h[dst] += (1.0 - tp.z[dst]) * d_h;
            let dz = -tp.h_old[dst] * d_h;
            let dr_val = tp.n[dst] * d_h;
            let dn = tp.r[dst] * d_h;
            let daz = dz * tp.dz_daz[dst];
            let dar = dr_val * tp.dr_dar[dst];
            let d_h_sag = dn * tp.delta_tanh[dst];
            let d_delta = dn * tp.h_sag[dst];
            let dan = d_delta * tp.dtanh_an[dst];
            let g_sag = 1.0 + cp.k_sag[b][dst].0 * feats[4];
            let d_a_eff = g_sag * dan;
            let d_g_sag = tp.a_eff[dst] * dan - tp.h_sag[dst] * tp.h_sag[dst] * d_h_sag;
            gb.k_quad[dst] += cp.k_quad[b][dst].1 * tp.a_sig[dst] * tp.a_sig[dst] * d_a_eff;
            let d_a_sig =
                (1.0 + 2.0 * cp.k_quad[b][dst].0 * tp.a_sig[dst]) * d_a_eff * tp.a_sig_mask[dst];
            let drec_n = d_a_sig;

            gb.b_z[dst] += cp.b_z[b][dst].1 * daz;
            gb.b_r[dst] += cp.b_r[b][dst].1 * dar;
            gb.b_n[dst] += cp.b_n[b][dst].1 * (dan - tp.dtanh_bn[dst] * d_delta);
            gb.k_sag[dst] += cp.k_sag[b][dst].1 * feats[4] * d_g_sag;
            d_feats[4] += cp.k_sag[b][dst].0 * d_g_sag;

            for f in 0..NUM_IN_FEATS_B0 {
                let fv = feats[f];
                gb.w_in_z[f][dst] += fv * daz;
                gb.w_in_r[f][dst] += fv * dar;
                gb.w_in_n[f][dst] += fv * d_a_sig;
                d_feats[f] += blk.w_in_z[f][dst] * daz
                    + blk.w_in_r[f][dst] * dar
                    + blk.w_in_n[f][dst] * d_a_sig;
            }
            if let Some(pv_arr) = prev_h {
                for sg in 0..=dst_g {
                    let s = sg << 2;
                    for k in 0..4 {
                        let sk = s + k;
                        let pv = pv_arr[sk];
                        gb.w_prev_z[dst][sk] += pv * daz;
                        gb.w_prev_r[dst][sk] += pv * dar;
                        gb.w_prev_n[dst][sk] += pv * d_a_sig;
                        d_prev_out[sk] += blk.w_prev_z[dst][sk] * daz
                            + blk.w_prev_r[dst][sk] * dar
                            + blk.w_prev_n[dst][sk] * d_a_sig;
                    }
                }
            }
            for sg in 0..=dst_g {
                let s = sg << 2;
                for k in 0..4 {
                    let sk = s + k;
                    let hv = tp.h_old[sk];
                    gb.w_rec_z[dst][sk] += hv * daz;
                    gb.w_rec_r[dst][sk] += hv * dar;
                    gb.w_rec_n[dst][sk] += hv * drec_n;
                    next_r_h[sk] += blk.w_rec_z[dst][sk] * daz
                        + blk.w_rec_r[dst][sk] * dar
                        + blk.w_rec_n[dst][sk] * drec_n;
                }
            }
        }
    }
    *r_gru_h = next_r_h;
    *d_prev_h = d_prev_out;
}
