#![forbid(unsafe_code)]
//! Exact `f64` mirror and reverse-mode adjoint gradient evaluator for `V3OptModelF64`
//! minimizing the exact degree-7 Binomial/Bézout 3-band `bESR` objective.

use std::f64::consts::PI;

use dnam_engine::stages::v3_core::cascade::{
    MAX_CASCADE_LANES, MAX_KAUTZ_POLES, NUM_CASCADE_BLOCKS, tier_cascade_lanes, tier_fir_segments,
    tier_kautz_banks,
};

use super::gabor::GaborAdjointScratch;
use super::model_f64::{HB_A0, HB_A1, V3OptModelF64, V3TrainDataset};
use super::primitives::{
    BlockSubTape, SvfChanTape, ap2_fwd, ap2_rev, exp2_oct_with_deriv, step_tpt_fwd, step_tpt_rev,
};

/// Forward/backward workspace reused across L-BFGS iterations.
#[derive(Clone, Debug, Default)]
pub struct AdjointWorkspace {
    svf_tape: Vec<[SvfChanTape; 2]>,
    s2_sub_tapes: [Vec<[BlockSubTape; NUM_CASCADE_BLOCKS]>; 2],
    u_s2: [Vec<f64>; 2],
    s2_out: [Vec<f64>; 2],
    dir_s2: [Vec<f64>; 2],
    pub(crate) y_out: [Vec<f64>; 2],
    kautz_re: [Vec<f64>; 2],
    kautz_im: [Vec<f64>; 2],
    err_buf: Vec<f64>,
    err_bands: [Vec<f64>; 3],
    d_bands: [Vec<f64>; 3],
    adj_dy_scratch: [Vec<f64>; 2],
    dy_c: [Vec<f64>; 2],
    ds2_out: [Vec<f64>; 2],
    du_s2: [Vec<f64>; 2],
    gabor_scratch: GaborAdjointScratch,
}

impl AdjointWorkspace {
    fn ensure_len(&mut self, n: usize, k_cnt: usize, os2x: bool) {
        let sub_n = if os2x { n * 2 } else { n };
        self.svf_tape.resize(n, [SvfChanTape::default(); 2]);
        for ch in 0..2 {
            self.s2_sub_tapes[ch].resize(sub_n, [BlockSubTape::default(); NUM_CASCADE_BLOCKS]);
            self.u_s2[ch].resize(n, 0.0);
            self.s2_out[ch].resize(n, 0.0);
            self.dir_s2[ch].resize(n, 0.0);
            self.y_out[ch].resize(n, 0.0);
            self.kautz_re[ch].resize(n * k_cnt, 0.0);
            self.kautz_im[ch].resize(n * k_cnt, 0.0);
            self.dy_c[ch].resize(n, 0.0);
            self.ds2_out[ch].resize(n, 0.0);
            self.du_s2[ch].resize(n, 0.0);
        }
        self.err_buf.resize(n, 0.0);
    }
}

#[inline(always)]
fn cl_mask(v: f64, lo: f64, hi: f64) -> (f64, f64) {
    (v.clamp(lo, hi), if v > lo && v < hi { 1.0 } else { 0.0 })
}

/// Evaluates the exact Bézout 3-band `bESR` objective on `dataset` in `f64` (no gradient).
pub fn evaluate_loss_f64(
    model: &V3OptModelF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) -> f64 {
    let mut grad = V3OptModelF64::zeros(model.tier, model.os2x, model.mono_out);
    evaluate_internal(model, dataset, ws, false, &mut grad)
}

/// Evaluates the exact Bézout 3-band `bESR` objective AND exact reverse-mode gradient in `f64`.
pub fn evaluate_loss_and_grad_f64(
    model: &V3OptModelF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) -> (f64, V3OptModelF64) {
    let mut grad = V3OptModelF64::zeros(model.tier, model.os2x, model.mono_out);
    let loss = evaluate_internal(model, dataset, ws, true, &mut grad);
    (loss, grad)
}

#[allow(clippy::too_many_lines)]
fn evaluate_internal(
    model: &V3OptModelF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    compute_grad: bool,
    grad: &mut V3OptModelF64,
) -> f64 {
    let c_cnt = tier_cascade_lanes(model.tier);
    let f_cnt = tier_fir_segments(model.tier) * 16;
    let k_cnt = tier_kautz_banks(model.tier) * 4;
    let active_ch = if model.mono_out { 1 } else { 2 };

    let (base_g, base_g_m) = cl_mask(model.svf_base_g, 1.0e-4, 1.45);
    let (sec2_ratio, sec2_ratio_m) = cl_mask(model.svf_sec2_g_ratio, 0.25, 4.0);
    let (env_oct, env_oct_m) = cl_mask(model.svf_env_oct_sens, -16.0, 16.0);
    let (env_comp, env_comp_m) = cl_mask(model.svf_env_comp, 0.0, 16.0);
    let (atk, atk_m) = cl_mask(model.svf_atk_alpha, 0.001, 0.9999);
    let (rel, rel_m) = cl_mask(model.svf_rel_alpha, 0.001, 0.99995);
    let (det_lp, det_lp_m) = cl_mask(model.svf_det_lp_pole, 0.05, 0.995);
    let (det_hp_mix, det_hp_m) = cl_mask(model.svf_det_hp_mix, -1.0, 2.0);
    let (damp_k1, damp_k1_m) = cl_mask(model.svf_damp_k1, 0.05, 4.0);
    let (damp_k2, damp_k2_m) = cl_mask(model.svf_damp_k2, 0.05, 4.0);
    let (sec2_bp, sec2_bp_m) = cl_mask(model.svf_sec2_bp_feed, -2.0, 2.0);
    let (sec2_hp, sec2_hp_m) = cl_mask(model.svf_sec2_hp_feed, -2.0, 2.0);
    let (link, link_m) = cl_mask(model.svf_stereo_link, 0.0, 1.0);
    let (g_min, g_min_m) = cl_mask(model.svf_g_min, 1.0e-4, 1.45);
    let (g_max, g_max_m) = if model.svf_g_max < g_min {
        (g_min, 0.0)
    } else if model.svf_g_max > 1.45 {
        (1.45, 0.0)
    } else {
        (model.svf_g_max, 1.0)
    };
    let mut k_poles = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); MAX_KAUTZ_POLES];
    for m in 0..k_cnt {
        let (r, r_m) = cl_mask(model.kautz_poles[m].0, 0.50, 0.997);
        let (th, t_m) = cl_mask(model.kautz_poles[m].1, 1.0e-3, PI - 1.0e-3);
        let bg = (1.0 - r * r).max(1.0e-7).sqrt();
        let (sin_t, cos_t) = th.sin_cos();
        k_poles[m] = (r, bg, cos_t, sin_t, r_m, t_m);
    }

    let mut total_loss = 0.0_f64;

    for (stem_idx, stem) in dataset.stems.iter().enumerate() {
        let n = stem
            .in_l
            .len()
            .min(stem.in_r.len())
            .min(stem.tgt_l.len())
            .min(stem.tgt_r.len());
        if n == 0 {
            continue;
        }
        let warmup = stem.warmup.min(n);
        ws.ensure_len(n, k_cnt, model.os2x);

        // --- Stage I Forward ---
        let (mut det_lp_s, mut env_s) = ([0.0_f64; 2], [0.0_f64; 2]);
        let (mut ic1_s1, mut ic2_s1, mut ic1_s2, mut ic2_s2) =
            ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
        for t in 0..n {
            let xl = stem.in_l[t] as f64;
            let xr = if model.mono_out {
                xl
            } else {
                stem.in_r[t] as f64
            };
            let xs = [xl, xr];
            let mut det_raw = [0.0_f64; 2];
            for ch in 0..2 {
                let prev_lp = det_lp_s[ch];
                let lp = det_lp * prev_lp + (1.0 - det_lp) * xs[ch];
                det_lp_s[ch] = lp;
                let v_det = xs[ch] + det_hp_mix * (xs[ch] - lp);
                det_raw[ch] = v_det.abs();
                let tp = &mut ws.svf_tape[t][ch];
                tp.det_lp_prev = prev_lp;
                tp.det_lp = lp;
                tp.v_det_hp = v_det;
                tp.det_raw_sign = if v_det > 0.0 {
                    1.0
                } else if v_det < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
            let half_link = 0.5 * link;
            let diff = det_raw[1] - det_raw[0];
            let det = [det_raw[0] + half_link * diff, det_raw[1] - half_link * diff];
            for ch in 0..active_ch {
                let env_prev = env_s[ch];
                let atk_taken = det[ch] > env_prev;
                let alpha = if atk_taken { atk } else { rel };
                let env_now = alpha * env_prev + (1.0 - alpha) * det[ch];
                env_s[ch] = env_now;
                let rsq = 1.0 / (1.0 + env_comp * env_now * env_now).sqrt();
                let e = env_now * rsq;
                let (oct_pow, _) = exp2_oct_with_deriv(env_oct * e);
                let g1_unclamped = base_g * oct_pow;
                let g1 = g1_unclamped.clamp(g_min, g_max);
                let g2 = (g1 * sec2_ratio).clamp(1.0e-4, 1.45);
                let (lp1, bp1, hp1, c1_1, a1_1, a2_1, a3_1, v3_1) =
                    step_tpt_fwd(xs[ch], g1, damp_k1, &mut ic1_s1[ch], &mut ic2_s1[ch]);
                let u2 = lp1 + sec2_bp * bp1 + sec2_hp * hp1;
                let (lp2, bp2, hp2, c1_2, a1_2, a2_2, a3_2, v3_2) =
                    step_tpt_fwd(u2, g2, damp_k2, &mut ic1_s2[ch], &mut ic2_s2[ch]);
                let taps = [xs[ch], lp1, bp1, hp1, lp2, bp2, hp2];
                ws.u_s2[ch][t] = taps
                    .iter()
                    .zip(&model.svf_stage2_feed_taps)
                    .map(|(a, b)| a * b)
                    .sum();
                ws.svf_tape[t][ch] = SvfChanTape {
                    env_prev,
                    atk_taken,
                    e,
                    rsq3: rsq * rsq * rsq,
                    g1_unclamped,
                    g1,
                    g2,
                    s1_c1: c1_1,
                    s1_v1: bp1,
                    s1_a1: a1_1,
                    s1_a2: a2_1,
                    s1_a3: a3_1,
                    s1_v3: v3_1,
                    s2_c1: c1_2,
                    s2_v1: bp2,
                    s2_a1: a1_2,
                    s2_a2: a2_2,
                    s2_a3: a3_2,
                    s2_v3: v3_2,
                    taps,
                    ..ws.svf_tape[t][ch]
                };
            }
        }

        // --- Stage II + III Forward ---
        for ch in 0..active_ch {
            let mut lp_s = [[0.0_f64; MAX_CASCADE_LANES]; NUM_CASCADE_BLOCKS];
            let mut env_q = [0.0_f64; NUM_CASCADE_BLOCKS];
            let (mut up0, mut up1, mut dn_s0, mut dn_s1, mut dn_d0, mut dn_d1) =
                ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
            let sub_steps = if model.os2x { 2 } else { 1 };
            for t in 0..n {
                let u_in = ws.u_s2[ch][t];
                let us = if model.os2x {
                    [
                        ap2_fwd(u_in, HB_A0, &mut up0),
                        ap2_fwd(u_in, HB_A1, &mut up1),
                    ]
                } else {
                    [u_in, 0.0]
                };
                let (mut s2_p, mut d_p) = ([0.0_f64; 2], [0.0_f64; 2]);
                for sub in 0..sub_steps {
                    let sub_t = if model.os2x { t * 2 + sub } else { t };
                    let mut h_buf = [0.0_f64; MAX_CASCADE_LANES];
                    for b in 0..NUM_CASCADE_BLOCKS {
                        let blk = &model.blocks[b];
                        let (ba, br) = (
                            blk.atk_alpha.clamp(0.001, 0.9999),
                            blk.rel_alpha.clamp(0.001, 0.99995),
                        );
                        let (gm, gs) = (
                            blk.gate_mix.clamp(0.0, 1.0),
                            blk.gate_sharpness.clamp(0.0, 1.0e4),
                        );
                        let h_prev = h_buf;
                        let tp = &mut ws.s2_sub_tapes[ch][sub_t][b];
                        tp.u_s2 = us[sub];
                        let mut abs_sum = 0.0_f64;
                        for l in 0..c_cnt {
                            let mut v = blk.in_u_gain[l] * us[sub];
                            if b > 0 {
                                for s in 0..c_cnt {
                                    v += blk.inter_mat[l][s] * h_prev[s];
                                }
                            }
                            let a = blk.lp_pole[l].clamp(0.02, 0.9985);
                            let sp = lp_s[b][l];
                            let sn = a * sp + (1.0 - a) * v;
                            lp_s[b][l] = sn;
                            let up = blk.lp_mix[l] * sn + blk.hp_mix[l] * (v - sn);
                            tp.lp_prev[l] = sp;
                            tp.lp_next[l] = sn;
                            tp.v[l] = v;
                            tp.up[l] = up;
                            tp.up_sign[l] = if up > 0.0 {
                                1.0
                            } else if up < 0.0 {
                                -1.0
                            } else {
                                0.0
                            };
                            abs_sum += up.abs();
                        }
                        let det = 0.25 * abs_sum;
                        let q_prev = env_q[b];
                        let atk_taken = det > q_prev;
                        let q = if atk_taken {
                            ba * q_prev + (1.0 - ba) * det
                        } else {
                            br * q_prev + (1.0 - br) * det
                        };
                        env_q[b] = q;
                        let (gg, ginv) = if gm > 0.0 {
                            let inv = 1.0 / (1.0 + gs * q * q);
                            ((1.0 - gm) + gm * (1.0 - inv), inv)
                        } else {
                            (1.0, 1.0)
                        };
                        tp.env_q_prev = q_prev;
                        tp.atk_taken = atk_taken;
                        tp.q = q;
                        tp.g_gate = gg;
                        tp.gate_inv = ginv;
                        for l in 0..c_cnt {
                            let sb = blk.static_bias[l].clamp(-4.0, 4.0);
                            let es = blk.env_bias_sens[l].clamp(-4.0, 4.0);
                            let z = tp.up[l] * gg + sb + es * q;
                            let zc = z.clamp(-64.0, 64.0);
                            let rz = 1.0 / (1.0 + zc * zc).sqrt();
                            let r0 = 1.0 / (1.0 + sb * sb).sqrt();
                            tp.dphi_dzc[l] = if z > -64.0 && z < 64.0 {
                                rz * rz * rz
                            } else {
                                0.0
                            };
                            tp.dphi0_db0[l] = r0 * r0 * r0;
                            let mut hn = (zc * rz - sb * r0) + blk.skip_u_gain[l] * tp.up[l];
                            if b > 0 {
                                hn += blk.prev_skip_gain[l] * h_prev[l];
                            }
                            tp.h_next[l] = hn;
                            h_buf[l] = hn;
                        }
                    }
                    let cab_dot: f64 = (0..c_cnt).map(|l| model.pre_cab_w[ch][l] * h_buf[l]).sum();
                    let dir_dot: f64 = (0..c_cnt)
                        .map(|l| model.direct_s2_w[ch][l] * h_buf[l])
                        .sum();
                    s2_p[sub] = model.pre_cab_u_gain[ch] * us[sub] + cab_dot;
                    d_p[sub] = dir_dot;
                }
                if !model.os2x {
                    ws.s2_out[ch][t] = s2_p[0];
                    ws.dir_s2[ch][t] = d_p[0];
                } else {
                    ws.s2_out[ch][t] = 0.5
                        * (ap2_fwd(s2_p[1], HB_A0, &mut dn_s0)
                            + ap2_fwd(s2_p[0], HB_A1, &mut dn_s1));
                    ws.dir_s2[ch][t] = 0.5
                        * (ap2_fwd(d_p[1], HB_A0, &mut dn_d0) + ap2_fwd(d_p[0], HB_A1, &mut dn_d1));
                }
            }

            let fir = if ch == 0 { &model.fir_l } else { &model.fir_r };
            let kc = if ch == 0 {
                &model.kautz_c_l
            } else {
                &model.kautz_c_r
            };
            let dir_taps = if ch == 0 {
                &model.svf_direct_taps_l
            } else {
                &model.svf_direct_taps_r
            };
            let (mut s_re, mut s_im) = (vec![0.0_f64; k_cnt], vec![0.0_f64; k_cnt]);
            for t in 0..n {
                let fir_v: f64 = (0..f_cnt.min(t + 1))
                    .map(|k| fir[k] * ws.s2_out[ch][t - k])
                    .sum();
                let x_del = if t >= 16 { ws.s2_out[ch][t - 16] } else { 0.0 };
                let mut kv = 0.0_f64;
                for m in 0..k_cnt {
                    let (r, bg, ct, st, _, _) = k_poles[m];
                    let (lr, li, pr, pi) = (r * ct, r * st, s_re[m], s_im[m]);
                    let (nr, ni) = (lr * pr - li * pi + bg * x_del, li * pr + lr * pi);
                    s_re[m] = nr;
                    s_im[m] = ni;
                    ws.kautz_re[ch][t * k_cnt + m] = nr;
                    ws.kautz_im[ch][t * k_cnt + m] = ni;
                    kv += kc[m].0 * nr - kc[m].1 * ni;
                }
                let dir_svf: f64 = ws.svf_tape[t][ch]
                    .taps
                    .iter()
                    .zip(dir_taps)
                    .map(|(a, b)| a * b)
                    .sum();
                ws.y_out[ch][t] = dir_svf + ws.dir_s2[ch][t] + fir_v + kv;
            }

            // Unmasked error -> Degree-7 Bézout 3-band split across all 0..n -> accumulate warmup..n
            let tgt = if ch == 0 { &stem.tgt_l } else { &stem.tgt_r };
            let ch_scale = if model.mono_out { 1.0 } else { 0.5 };
            for t in 0..n {
                ws.err_buf[t] = ws.y_out[ch][t] - (tgt[t] as f64);
            }
            dataset
                .bank
                .split_3band(&ws.err_buf[..n], &mut ws.err_bands);
            let bw = dataset.stem_band_weights[stem_idx][ch];
            for b in 0..3 {
                let eb: f64 = ws.err_bands[b][warmup..n].iter().map(|&v| v * v).sum();
                total_loss += ch_scale * bw[b] * eb;
            }

            if compute_grad {
                for b in 0..3 {
                    ws.d_bands[b].resize(n, 0.0);
                    let w = 2.0 * ch_scale * bw[b];
                    for t in 0..warmup {
                        ws.d_bands[b][t] = 0.0;
                    }
                    for t in warmup..n {
                        ws.d_bands[b][t] = w * ws.err_bands[b][t];
                    }
                }
                ws.dy_c[ch].fill(0.0);
                dataset.bank.accumulate_adjoint(
                    &ws.d_bands,
                    &mut ws.dy_c[ch][..n],
                    &mut ws.adj_dy_scratch,
                );
                if dataset.gabor_env_weight > 0.0 || dataset.gabor_ltas_weight > 0.0 {
                    total_loss += dataset.eval_gabor_loss_and_adjoint(
                        stem_idx,
                        ch,
                        ch_scale,
                        &ws.y_out[ch][..n],
                        Some(&mut ws.dy_c[ch][..n]),
                        &mut ws.gabor_scratch,
                    );
                }
            } else if dataset.gabor_env_weight > 0.0 || dataset.gabor_ltas_weight > 0.0 {
                total_loss += dataset.eval_gabor_loss_and_adjoint(
                    stem_idx,
                    ch,
                    ch_scale,
                    &ws.y_out[ch][..n],
                    None,
                    &mut ws.gabor_scratch,
                );
            }
        }

        if !compute_grad {
            continue;
        }

        // --- Stage III Backward + Stage I Direct Taps ---
        for ch in 0..active_ch {
            ws.ds2_out[ch].fill(0.0);
            let fir = if ch == 0 { &model.fir_l } else { &model.fir_r };
            let kc = if ch == 0 {
                &model.kautz_c_l
            } else {
                &model.kautz_c_r
            };
            let (mut ds_re, mut ds_im) = (vec![0.0_f64; k_cnt], vec![0.0_f64; k_cnt]);
            for t in (0..n).rev() {
                let dy = ws.dy_c[ch][t];
                let d_dir_t = if ch == 0 {
                    &mut grad.svf_direct_taps_l
                } else {
                    &mut grad.svf_direct_taps_r
                };
                for i in 0..7 {
                    d_dir_t[i] += ws.svf_tape[t][ch].taps[i] * dy;
                }
                let d_fir = if ch == 0 {
                    &mut grad.fir_l
                } else {
                    &mut grad.fir_r
                };
                for k in 0..f_cnt.min(t + 1) {
                    d_fir[k] += ws.s2_out[ch][t - k] * dy;
                    ws.ds2_out[ch][t - k] += fir[k] * dy;
                }
                let x_del = if t >= 16 { ws.s2_out[ch][t - 16] } else { 0.0 };
                let mut dx_del = 0.0_f64;
                for m in 0..k_cnt {
                    let nr = ws.kautz_re[ch][t * k_cnt + m];
                    let ni = ws.kautz_im[ch][t * k_cnt + m];
                    let pr = if t > 0 {
                        ws.kautz_re[ch][(t - 1) * k_cnt + m]
                    } else {
                        0.0
                    };
                    let pi = if t > 0 {
                        ws.kautz_im[ch][(t - 1) * k_cnt + m]
                    } else {
                        0.0
                    };
                    let d_kc = if ch == 0 {
                        &mut grad.kautz_c_l[m]
                    } else {
                        &mut grad.kautz_c_r[m]
                    };
                    d_kc.0 += nr * dy;
                    d_kc.1 += -ni * dy;
                    let dnr = kc[m].0 * dy + ds_re[m];
                    let dni = -kc[m].1 * dy + ds_im[m];
                    let (r, bg, ct, st, r_m, t_m) = k_poles[m];
                    let (lr, li) = (r * ct, r * st);
                    dx_del += bg * dnr;
                    let (d_lr, d_li, d_bg) =
                        (pr * dnr + pi * dni, -pi * dnr + pr * dni, x_del * dnr);
                    grad.kautz_poles[m].0 += r_m * (ct * d_lr + st * d_li - (r / bg) * d_bg);
                    grad.kautz_poles[m].1 += t_m * r * (-st * d_lr + ct * d_li);
                    ds_re[m] = lr * dnr + li * dni;
                    ds_im[m] = -li * dnr + lr * dni;
                }
                if t >= 16 {
                    ws.ds2_out[ch][t - 16] += dx_del;
                }
            }
        }

        // --- Stage II Backward ---
        for ch in 0..active_ch {
            ws.du_s2[ch].fill(0.0);
            let mut rev_lp = [[0.0_f64; MAX_CASCADE_LANES]; NUM_CASCADE_BLOCKS];
            let mut rev_q = [0.0_f64; NUM_CASCADE_BLOCKS];
            let (mut dn_s0_r, mut dn_s1_r, mut dn_d0_r, mut dn_d1_r, mut up0_r, mut up1_r) =
                ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
            let sub_steps = if model.os2x { 2 } else { 1 };
            for t in (0..n).rev() {
                let (mut sub_s2, mut sub_dir) = ([0.0; 2], [0.0; 2]);
                if !model.os2x {
                    sub_s2[0] = ws.ds2_out[ch][t];
                    sub_dir[0] = ws.dy_c[ch][t];
                } else {
                    sub_s2[1] = ap2_rev(0.5 * ws.ds2_out[ch][t], HB_A0, &mut dn_s0_r);
                    sub_s2[0] = ap2_rev(0.5 * ws.ds2_out[ch][t], HB_A1, &mut dn_s1_r);
                    sub_dir[1] = ap2_rev(0.5 * ws.dy_c[ch][t], HB_A0, &mut dn_d0_r);
                    sub_dir[0] = ap2_rev(0.5 * ws.dy_c[ch][t], HB_A1, &mut dn_d1_r);
                }
                let mut u_sub_g = [0.0_f64; 2];
                for sub in (0..sub_steps).rev() {
                    let sub_t = if model.os2x { t * 2 + sub } else { t };
                    let (d_s2, d_dir) = (sub_s2[sub], sub_dir[sub]);
                    let tapes = &ws.s2_sub_tapes[ch][sub_t];
                    let last_h = &tapes[NUM_CASCADE_BLOCKS - 1].h_next;
                    let mut du_sub = model.pre_cab_u_gain[ch] * d_s2;
                    grad.pre_cab_u_gain[ch] += tapes[0].u_s2 * d_s2;
                    let mut d_h_curr = [0.0_f64; MAX_CASCADE_LANES];
                    for l in 0..c_cnt {
                        grad.pre_cab_w[ch][l] += last_h[l] * d_s2;
                        grad.direct_s2_w[ch][l] += last_h[l] * d_dir;
                        d_h_curr[l] =
                            model.pre_cab_w[ch][l] * d_s2 + model.direct_s2_w[ch][l] * d_dir;
                    }
                    for b in (0..NUM_CASCADE_BLOCKS).rev() {
                        let (blk, gblk, tp) = (&model.blocks[b], &mut grad.blocks[b], &tapes[b]);
                        let h_prev = if b > 0 {
                            &tapes[b - 1].h_next
                        } else {
                            &tapes[0].h_next
                        };
                        let (ba, atk_m) = cl_mask(blk.atk_alpha, 0.001, 0.9999);
                        let (br, rel_m) = cl_mask(blk.rel_alpha, 0.001, 0.99995);
                        let (gm, gm_m) = cl_mask(blk.gate_mix, 0.0, 1.0);
                        let (gs, gs_m) = cl_mask(blk.gate_sharpness, 0.0, 1.0e4);
                        let (mut d_h_prev, mut d_up, mut d_q, mut d_gg) = (
                            [0.0; MAX_CASCADE_LANES],
                            [0.0; MAX_CASCADE_LANES],
                            rev_q[b],
                            0.0,
                        );
                        for l in 0..c_cnt {
                            let dh = d_h_curr[l];
                            gblk.skip_u_gain[l] += tp.up[l] * dh;
                            d_up[l] += blk.skip_u_gain[l] * dh;
                            if b > 0 {
                                gblk.prev_skip_gain[l] += h_prev[l] * dh;
                                d_h_prev[l] += blk.prev_skip_gain[l] * dh;
                            }
                            let dz = dh * tp.dphi_dzc[l];
                            d_up[l] += tp.g_gate * dz;
                            d_gg += tp.up[l] * dz;
                            let (_, sb_m) = cl_mask(blk.static_bias[l], -4.0, 4.0);
                            let (es, es_m) = cl_mask(blk.env_bias_sens[l], -4.0, 4.0);
                            gblk.static_bias[l] += sb_m * (dz - dh * tp.dphi0_db0[l]);
                            gblk.env_bias_sens[l] += es_m * tp.q * dz;
                            d_q += es * dz;
                        }
                        if gm > 0.0 {
                            let inv2 = tp.gate_inv * tp.gate_inv;
                            gblk.gate_mix += gm_m * (-tp.gate_inv) * d_gg;
                            gblk.gate_sharpness += gs_m * gm * inv2 * tp.q * tp.q * d_gg;
                            d_q += gm * inv2 * 2.0 * gs * tp.q * d_gg;
                        }
                        let alpha = if tp.atk_taken { ba } else { br };
                        let det = 0.25 * tp.up[..c_cnt].iter().map(|&v| v.abs()).sum::<f64>();
                        let d_det = (1.0 - alpha) * d_q;
                        if tp.atk_taken {
                            gblk.atk_alpha += atk_m * (tp.env_q_prev - det) * d_q;
                        } else {
                            gblk.rel_alpha += rel_m * (tp.env_q_prev - det) * d_q;
                        }
                        rev_q[b] = alpha * d_q;
                        for l in 0..c_cnt {
                            let dup = d_up[l] + 0.25 * tp.up_sign[l] * d_det;
                            gblk.lp_mix[l] += tp.lp_next[l] * dup;
                            gblk.hp_mix[l] += (tp.v[l] - tp.lp_next[l]) * dup;
                            let ds_tot = (blk.lp_mix[l] - blk.hp_mix[l]) * dup + rev_lp[b][l];
                            let (a, lp_m) = cl_mask(blk.lp_pole[l], 0.02, 0.9985);
                            gblk.lp_pole[l] += lp_m * (tp.lp_prev[l] - tp.v[l]) * ds_tot;
                            rev_lp[b][l] = a * ds_tot;
                            let dv = blk.hp_mix[l] * dup + (1.0 - a) * ds_tot;
                            gblk.in_u_gain[l] += tp.u_s2 * dv;
                            du_sub += blk.in_u_gain[l] * dv;
                            if b > 0 {
                                for s in 0..c_cnt {
                                    gblk.inter_mat[l][s] += h_prev[s] * dv;
                                    d_h_prev[s] += blk.inter_mat[l][s] * dv;
                                }
                            }
                        }
                        d_h_curr = d_h_prev;
                    }
                    u_sub_g[sub] = du_sub;
                }
                ws.du_s2[ch][t] = if !model.os2x {
                    u_sub_g[0]
                } else {
                    ap2_rev(u_sub_g[0], HB_A0, &mut up0_r) + ap2_rev(u_sub_g[1], HB_A1, &mut up1_r)
                };
            }
        }

        // --- Stage I Backward ---
        let (mut r_dlp, mut r_env) = ([0.0_f64; 2], [0.0_f64; 2]);
        let (mut r_ic1_1, mut r_ic2_1, mut r_ic1_2, mut r_ic2_2) =
            ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
        for t in (0..n).rev() {
            let mut d_det = [0.0_f64; 2];
            let d_raw0 = ws.svf_tape[t][0].v_det_hp.abs();
            let d_raw1 = ws.svf_tape[t][1].v_det_hp.abs();
            let diff = d_raw1 - d_raw0;
            for ch in 0..active_ch {
                let tp = &ws.svf_tape[t][ch];
                let (d_dir, d_u) = (ws.dy_c[ch][t], ws.du_s2[ch][t]);
                let dir_t = if ch == 0 {
                    &model.svf_direct_taps_l
                } else {
                    &model.svf_direct_taps_r
                };
                let mut d_taps = [0.0_f64; 7];
                for i in 0..7 {
                    grad.svf_stage2_feed_taps[i] += tp.taps[i] * d_u;
                    d_taps[i] = model.svf_stage2_feed_taps[i] * d_u + dir_t[i] * d_dir;
                }
                let (d_u2, d_g2, d_k2) = step_tpt_rev(
                    d_taps[4],
                    d_taps[5],
                    d_taps[6],
                    tp.g2,
                    damp_k2,
                    tp.s2_c1,
                    tp.s2_v1,
                    tp.s2_a1,
                    tp.s2_a2,
                    tp.s2_a3,
                    tp.s2_v3,
                    &mut r_ic1_2[ch],
                    &mut r_ic2_2[ch],
                );
                grad.svf_damp_k2 += damp_k2_m * d_k2;
                grad.svf_sec2_bp_feed += sec2_bp_m * tp.taps[2] * d_u2;
                grad.svf_sec2_hp_feed += sec2_hp_m * tp.taps[3] * d_u2;
                let (_, d_g1_dir, d_k1) = step_tpt_rev(
                    d_taps[1] + d_u2,
                    d_taps[2] + sec2_bp * d_u2,
                    d_taps[3] + sec2_hp * d_u2,
                    tp.g1,
                    damp_k1,
                    tp.s1_c1,
                    tp.s1_v1,
                    tp.s1_a1,
                    tp.s1_a2,
                    tp.s1_a3,
                    tp.s1_v3,
                    &mut r_ic1_1[ch],
                    &mut r_ic2_1[ch],
                );
                grad.svf_damp_k1 += damp_k1_m * d_k1;
                let g2_uncl = tp.g1 * sec2_ratio;
                let d_g2_u = if g2_uncl > 1.0e-4 && g2_uncl < 1.45 {
                    d_g2
                } else {
                    0.0
                };
                grad.svf_sec2_g_ratio += sec2_ratio_m * tp.g1 * d_g2_u;
                let d_g1_t = d_g1_dir + sec2_ratio * d_g2_u;
                let d_g1_u = if tp.g1_unclamped <= g_min {
                    grad.svf_g_min += g_min_m * d_g1_t;
                    0.0
                } else if tp.g1_unclamped >= g_max {
                    grad.svf_g_max += g_max_m * d_g1_t;
                    0.0
                } else {
                    d_g1_t
                };
                let (oct_p, d_oct) = exp2_oct_with_deriv(env_oct * tp.e);
                grad.svf_base_g += base_g_m * oct_p * d_g1_u;
                let d_eo = base_g * d_oct * d_g1_u;
                grad.svf_env_oct_sens += env_oct_m * tp.e * d_eo;
                let d_e = env_oct * d_eo;
                grad.svf_env_comp += env_comp_m * (-0.5 * tp.e * tp.e * tp.e) * d_e;
                let d_env_now = tp.rsq3 * d_e + r_env[ch];
                let alpha = if tp.atk_taken { atk } else { rel };
                let det_ch = if ch == 0 {
                    d_raw0 + 0.5 * link * diff
                } else {
                    d_raw1 - 0.5 * link * diff
                };
                if tp.atk_taken {
                    grad.svf_atk_alpha += atk_m * (tp.env_prev - det_ch) * d_env_now;
                } else {
                    grad.svf_rel_alpha += rel_m * (tp.env_prev - det_ch) * d_env_now;
                }
                r_env[ch] = alpha * d_env_now;
                d_det[ch] = (1.0 - alpha) * d_env_now;
            }
            let d_diff = 0.5 * link * (d_det[0] - d_det[1]);
            grad.svf_stereo_link += link_m * 0.5 * diff * (d_det[0] - d_det[1]);
            let d_dr = [d_det[0] - d_diff, d_det[1] + d_diff];
            for ch in 0..2 {
                let tp = &ws.svf_tape[t][ch];
                let xl = stem.in_l[t] as f64;
                let x_ch = if ch == 0 || model.mono_out {
                    xl
                } else {
                    stem.in_r[t] as f64
                };
                let d_vd = tp.det_raw_sign * d_dr[ch];
                grad.svf_det_hp_mix += det_hp_m * (x_ch - tp.det_lp) * d_vd;
                let d_lp_tot = r_dlp[ch] - det_hp_mix * d_vd;
                grad.svf_det_lp_pole += det_lp_m * (tp.det_lp_prev - x_ch) * d_lp_tot;
                r_dlp[ch] = det_lp * d_lp_tot;
            }
        }
    }

    total_loss
}
