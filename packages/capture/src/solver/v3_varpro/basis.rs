#![forbid(unsafe_code)]
//! Linear basis extractors, Bézout-weighted `uplo='U'` `VarPro` Cholesky solver, and
//! 120-pole orthogonal matching pursuit (`OMP`) Kautz dictionary selector.

use std::f64::consts::PI;

use dnam_engine::Tier;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::stages::v3_core::cascade::{
    MAX_CASCADE_LANES, MAX_KAUTZ_POLES, NUM_CASCADE_BLOCKS, tier_cascade_lanes, tier_fir_segments,
    tier_kautz_banks,
};
use dnam_engine::stages::v3_core::{V3ColdTierModel, V3CoreInstance};

use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};

use super::model_f64::{HB_A0, HB_A1, V3OptModelF64, V3TrainDataset, V3TrainStem};
use super::primitives::{ap2_fwd, exp2_oct_with_deriv, step_tpt_fwd};

/// Linear readout basis dimension `7 + C + F + 2*K` for a given `Tier`.
#[must_use]
pub const fn linear_basis_dim(tier: Tier) -> usize {
    7 + tier_cascade_lanes(tier) + tier_fir_segments(tier) * 16 + tier_kautz_banks(tier) * 8
}

#[inline]
fn dot4(a: &[f64], b: &[f64]) -> f64 {
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in ac.iter().zip(bc) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in ar.iter().zip(br) {
        sum += x * y;
    }
    sum
}

struct StemTraj {
    s2_out: [Vec<f64>; 2],
    taps: [Vec<[f64; 7]>; 2],
    h_dn: [Vec<[f64; MAX_CASCADE_LANES]>; 2],
}

fn extract_traj_opt(
    model: &V3OptModelF64,
    stem: &V3TrainStem,
    s2_override: Option<&[Vec<f64>; 2]>,
) -> StemTraj {
    let n = stem
        .in_l
        .len()
        .min(stem.in_r.len())
        .min(stem.tgt_l.len())
        .min(stem.tgt_r.len());
    let c_cnt = tier_cascade_lanes(model.tier);
    let ch_cnt = if model.mono_out { 1 } else { 2 };

    let mut taps_out = [vec![[0.0_f64; 7]; n], vec![[0.0_f64; 7]; n]];
    let mut h_dn_out = [
        vec![[0.0_f64; MAX_CASCADE_LANES]; n],
        vec![[0.0_f64; MAX_CASCADE_LANES]; n],
    ];
    let mut s2_out = [vec![0.0_f64; n], vec![0.0_f64; n]];

    let base_g = model.svf_base_g.clamp(1.0e-4, 1.45);
    let sec2_r = model.svf_sec2_g_ratio.clamp(0.25, 4.0);
    let env_oct = model.svf_env_oct_sens.clamp(-16.0, 16.0);
    let env_comp = model.svf_env_comp.clamp(0.0, 16.0);
    let atk = model.svf_atk_alpha.clamp(0.001, 0.9999);
    let rel = model.svf_rel_alpha.clamp(0.001, 0.99995);
    let det_lp_a = model.svf_det_lp_pole.clamp(0.05, 0.995);
    let det_hp = model.svf_det_hp_mix.clamp(-1.0, 2.0);
    let dk1 = model.svf_damp_k1.clamp(0.05, 4.0);
    let dk2 = model.svf_damp_k2.clamp(0.05, 4.0);
    let s2_bp = model.svf_sec2_bp_feed.clamp(-2.0, 2.0);
    let s2_hp = model.svf_sec2_hp_feed.clamp(-2.0, 2.0);
    let hl = 0.5 * model.svf_stereo_link.clamp(0.0, 1.0);
    let g_min = model.svf_g_min.clamp(1.0e-4, 1.45);
    let g_max = model.svf_g_max.clamp(g_min, 1.45);

    let (mut det_lp, mut env) = ([0.0_f64; 2], [0.0_f64; 2]);
    let (mut ic1_1, mut ic2_1, mut ic1_2, mut ic2_2) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
    let mut u_s2 = [vec![0.0_f64; n], vec![0.0_f64; n]];
    for t in 0..n {
        let xl = stem.in_l[t] as f64;
        let xr = if model.mono_out {
            xl
        } else {
            stem.in_r[t] as f64
        };
        let xs = [xl, xr];
        let mut dr = [0.0_f64; 2];
        for ch in 0..2 {
            det_lp[ch] = det_lp_a * det_lp[ch] + (1.0 - det_lp_a) * xs[ch];
            dr[ch] = (xs[ch] + det_hp * (xs[ch] - det_lp[ch])).abs();
        }
        let diff = dr[1] - dr[0];
        let det = [dr[0] + hl * diff, dr[1] - hl * diff];
        for ch in 0..ch_cnt {
            env[ch] = if det[ch] > env[ch] {
                atk * env[ch] + (1.0 - atk) * det[ch]
            } else {
                rel * env[ch] + (1.0 - rel) * det[ch]
            };
            let e = env[ch] / (1.0 + env_comp * env[ch] * env[ch]).sqrt();
            let (pow2, _) = exp2_oct_with_deriv(env_oct * e);
            let g1 = (base_g * pow2).clamp(g_min, g_max);
            let g2 = (g1 * sec2_r).clamp(1.0e-4, 1.45);
            let (lp1, bp1, hp1, ..) = step_tpt_fwd(xs[ch], g1, dk1, &mut ic1_1[ch], &mut ic2_1[ch]);
            let u2 = lp1 + s2_bp * bp1 + s2_hp * hp1;
            let (lp2, bp2, hp2, ..) = step_tpt_fwd(u2, g2, dk2, &mut ic1_2[ch], &mut ic2_2[ch]);
            let taps = [xs[ch], lp1, bp1, hp1, lp2, bp2, hp2];
            taps_out[ch][t] = taps;
            u_s2[ch][t] = taps
                .iter()
                .zip(&model.svf_stage2_feed_taps)
                .map(|(a, b)| a * b)
                .sum();
        }
    }

    let sub_steps = if model.os2x { 2 } else { 1 };
    for ch in 0..ch_cnt {
        let mut lp_s = [[0.0_f64; MAX_CASCADE_LANES]; NUM_CASCADE_BLOCKS];
        let mut env_q = [0.0_f64; NUM_CASCADE_BLOCKS];
        let (mut up0, mut up1, mut dn_s0, mut dn_s1) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
        let (mut dn_h0, mut dn_h1) = (
            [[0.0_f64; 2]; MAX_CASCADE_LANES],
            [[0.0_f64; 2]; MAX_CASCADE_LANES],
        );
        let mut h_sub = [[0.0_f64; MAX_CASCADE_LANES]; 2];
        let mut s2_p = [0.0_f64; 2];
        for t in 0..n {
            let u_in = u_s2[ch][t];
            let us = if model.os2x {
                [
                    ap2_fwd(u_in, HB_A0, &mut up0),
                    ap2_fwd(u_in, HB_A1, &mut up1),
                ]
            } else {
                [u_in, 0.0]
            };
            for sub in 0..sub_steps {
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
                    let mut up_arr = [0.0_f64; MAX_CASCADE_LANES];
                    let mut abs_sum = 0.0_f64;
                    for l in 0..c_cnt {
                        let mut v = blk.in_u_gain[l] * us[sub];
                        if b > 0 {
                            for s in 0..c_cnt {
                                v += blk.inter_mat[l][s] * h_prev[s];
                            }
                        }
                        let a = blk.lp_pole[l].clamp(0.02, 0.9985);
                        lp_s[b][l] = a * lp_s[b][l] + (1.0 - a) * v;
                        let up = blk.lp_mix[l] * lp_s[b][l] + blk.hp_mix[l] * (v - lp_s[b][l]);
                        up_arr[l] = up;
                        abs_sum += up.abs();
                    }
                    let det = 0.25 * abs_sum;
                    env_q[b] = if det > env_q[b] {
                        ba * env_q[b] + (1.0 - ba) * det
                    } else {
                        br * env_q[b] + (1.0 - br) * det
                    };
                    let q = env_q[b];
                    let gg = if gm > 0.0 {
                        (1.0 - gm) + gm * (1.0 - 1.0 / (1.0 + gs * q * q))
                    } else {
                        1.0
                    };
                    for l in 0..c_cnt {
                        let sb = blk.static_bias[l].clamp(-4.0, 4.0);
                        let es = blk.env_bias_sens[l].clamp(-4.0, 4.0);
                        let zc = (up_arr[l] * gg + sb + es * q).clamp(-64.0, 64.0);
                        let nl = zc / (1.0 + zc * zc).sqrt() - sb / (1.0 + sb * sb).sqrt();
                        h_buf[l] = nl + blk.skip_u_gain[l] * up_arr[l];
                        if b > 0 {
                            h_buf[l] += blk.prev_skip_gain[l] * h_prev[l];
                        }
                    }
                }
                h_sub[sub] = h_buf;
                s2_p[sub] = model.pre_cab_u_gain[ch] * us[sub]
                    + (0..c_cnt)
                        .map(|l| model.pre_cab_w[ch][l] * h_buf[l])
                        .sum::<f64>();
            }
            if !model.os2x {
                h_dn_out[ch][t][..c_cnt].copy_from_slice(&h_sub[0][..c_cnt]);
                s2_out[ch][t] = s2_p[0];
            } else {
                for l in 0..c_cnt {
                    h_dn_out[ch][t][l] = 0.5
                        * (ap2_fwd(h_sub[1][l], HB_A0, &mut dn_h0[l])
                            + ap2_fwd(h_sub[0][l], HB_A1, &mut dn_h1[l]));
                }
                s2_out[ch][t] = 0.5
                    * (ap2_fwd(s2_p[1], HB_A0, &mut dn_s0) + ap2_fwd(s2_p[0], HB_A1, &mut dn_s1));
            }
        }
    }
    if let Some(ov) = s2_override {
        s2_out = [ov[0].clone(), ov[1].clone()];
    }
    if model.mono_out {
        taps_out[1] = taps_out[0].clone();
        h_dn_out[1] = h_dn_out[0].clone();
        s2_out[1] = s2_out[0].clone();
    }
    StemTraj {
        s2_out,
        taps: taps_out,
        h_dn: h_dn_out,
    }
}

fn build_channel_basis(
    tier: Tier,
    kautz_poles: &[(f64, f64)],
    traj: &StemTraj,
    ch: usize,
    k_override: Option<usize>,
) -> Vec<Vec<f64>> {
    let n = traj.s2_out[ch].len();
    let c_cnt = tier_cascade_lanes(tier);
    let f_cnt = tier_fir_segments(tier) * 16;
    let k_cnt = k_override.unwrap_or_else(|| tier_kautz_banks(tier) * 4);
    let p_dim = 7 + c_cnt + f_cnt + 2 * k_cnt;
    let mut cols = vec![vec![0.0_f64; n]; p_dim];
    for t in 0..n {
        for i in 0..7 {
            cols[i][t] = traj.taps[ch][t][i];
        }
        for l in 0..c_cnt {
            cols[7 + l][t] = traj.h_dn[ch][t][l];
        }
    }
    let s2 = &traj.s2_out[ch];
    for k in 0..f_cnt.min(n) {
        let col = &mut cols[7 + c_cnt + k];
        col[k..n].copy_from_slice(&s2[..n - k]);
    }
    let k_base = 7 + c_cnt + f_cnt;
    let (mut s_re, mut s_im) = (vec![0.0_f64; k_cnt], vec![0.0_f64; k_cnt]);
    let pre: Vec<(f64, f64, f64)> = (0..k_cnt)
        .map(|m| {
            let r = kautz_poles[m].0.clamp(0.50, 0.997);
            let th = kautz_poles[m].1.clamp(1.0e-3, PI - 1.0e-3);
            let bg = (1.0 - r * r).max(1.0e-7).sqrt();
            let (st, ct) = th.sin_cos();
            (r * ct, r * st, bg)
        })
        .collect();
    for t in 0..n {
        let x_del = if t >= 16 { s2[t - 16] } else { 0.0 };
        for m in 0..k_cnt {
            let (lr, li, bg) = pre[m];
            let nr = lr * s_re[m] - li * s_im[m] + bg * x_del;
            let ni = li * s_re[m] + lr * s_im[m];
            s_re[m] = nr;
            s_im[m] = ni;
            cols[k_base + 2 * m][t] = nr;
            cols[k_base + 2 * m + 1][t] = -ni;
        }
    }
    cols
}

fn solve_weighted_basis(
    model: &mut V3OptModelF64,
    dataset: &V3TrainDataset,
    trajs: &[StemTraj],
    k_override: Option<usize>,
) {
    let tier = model.tier;
    let c_cnt = tier_cascade_lanes(tier);
    let f_cnt = tier_fir_segments(tier) * 16;
    let k_cnt = k_override.unwrap_or_else(|| tier_kautz_banks(tier) * 4);
    let p_dim = 7 + c_cnt + f_cnt + 2 * k_cnt;
    let k_start = 7 + c_cnt + f_cnt;
    let active_ch = if model.mono_out { 1 } else { 2 };
    let mut bands = [Vec::new(), Vec::new(), Vec::new()];
    let mut col_re = Vec::<f64>::new();
    let mut col_im = Vec::<f64>::new();

    for ch in 0..active_ch {
        let mut phi_w = vec![Vec::<f64>::new(); p_dim];
        let mut y_w = Vec::<f64>::new();
        for (st_idx, stem) in dataset.stems.iter().enumerate() {
            let n = trajs[st_idx].s2_out[ch].len();
            let w = stem.warmup.min(n);
            if n <= w {
                continue;
            }
            let bw = dataset.stem_band_weights[st_idx][ch];
            let sq_w = [bw[0].sqrt(), bw[1].sqrt(), bw[2].sqrt()];
            let tgt_src: Vec<f64> = if ch == 0 {
                &stem.tgt_l[..n]
            } else {
                &stem.tgt_r[..n]
            }
            .iter()
            .map(|&v| v as f64)
            .collect();
            dataset.bank.split_3band(&tgt_src, &mut bands);
            for b in 0..3 {
                y_w.extend(bands[b][w..n].iter().map(|&v| v * sq_w[b]));
            }
            let cols =
                build_channel_basis(tier, &model.kautz_poles, &trajs[st_idx], ch, k_override);
            for (j, col) in cols.iter().enumerate() {
                dataset.bank.split_3band(col, &mut bands);
                for b in 0..3 {
                    phi_w[j].extend(bands[b][w..n].iter().map(|&v| v * sq_w[b]));
                }
            }
            let gab_sum = &dataset.gabor_targets[st_idx][ch];
            let stem_w = dataset.stem_weights[st_idx];
            for (b, spec) in dataset.gabor_bank.bands.iter().enumerate() {
                let m_cnt = gab_sum.n_frames[b];
                if m_cnt == 0 {
                    continue;
                }
                let sq_g = (stem_w * gab_sum.varpro_band_w[b]).sqrt();
                if sq_g <= 0.0 {
                    continue;
                }
                let off = gab_sum.band_offsets[b];
                for m in 0..m_cnt {
                    y_w.push(gab_sum.tgt_re[off + m] * sq_g);
                    y_w.push(gab_sum.tgt_im[off + m] * sq_g);
                }
                col_re.resize(m_cnt, 0.0);
                col_im.resize(m_cnt, 0.0);
                for (j, col) in cols.iter().enumerate() {
                    spec.project_frames(col, w, &mut col_re, &mut col_im);
                    for m in 0..m_cnt {
                        phi_w[j].push(col_re[m] * sq_g);
                        phi_w[j].push(col_im[m] * sq_g);
                    }
                }
            }
        }
        if y_w.is_empty() {
            continue;
        }
        let mut gram = vec![0.0_f64; uplo_u_packed_len(p_dim)];
        let mut rhs = vec![[0.0_f64; 2]; p_dim];
        let mut diag_sum = 0.0_f64;
        for j in 0..p_dim {
            let base = uplo_u_index(0, j);
            for i in 0..=j {
                gram[base + i] = dot4(&phi_w[i], &phi_w[j]);
            }
            diag_sum += gram[base + j];
            rhs[j][0] = dot4(&phi_w[j], &y_w);
        }
        let avg_diag = (diag_sum / (p_dim as f64).max(1.0)).max(1.0e-12);
        let base_ridge = (1.0e-10 * avg_diag).max(1.0e-12);
        let direct_ridge = 1.0e-5 * avg_diag;
        for j in 0..7.min(p_dim) {
            gram[uplo_u_index(j, j)] += direct_ridge;
        }
        let kautz_ridge = 1.0e-7 * avg_diag;
        for j in k_start..p_dim {
            gram[uplo_u_index(j, j)] += kautz_ridge;
        }
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(&gram, &rhs, base_ridge) {
            let dir_t = if ch == 0 {
                &mut model.svf_direct_taps_l
            } else {
                &mut model.svf_direct_taps_r
            };
            for i in 0..7 {
                dir_t[i] = sol[i][0];
            }
            for l in 0..c_cnt {
                model.direct_s2_w[ch][l] = sol[7 + l][0];
            }
            let fir = if ch == 0 {
                &mut model.fir_l
            } else {
                &mut model.fir_r
            };
            for k in 0..f_cnt {
                fir[k] = sol[7 + c_cnt + k][0];
            }
            let kc = if ch == 0 {
                &mut model.kautz_c_l
            } else {
                &mut model.kautz_c_r
            };
            for m in 0..k_cnt {
                kc[m] = (sol[k_start + 2 * m][0], sol[k_start + 2 * m + 1][0]);
            }
        }
    }
    if model.mono_out {
        model.svf_direct_taps_r = model.svf_direct_taps_l;
        model.direct_s2_w[1] = model.direct_s2_w[0];
        model.fir_r = model.fir_l;
        model.kautz_c_r = model.kautz_c_l;
    }
}

/// Solves closed-form Bézout-weighted `VarPro` Cholesky directly on [`V3OptModelF64`].
pub fn solve_bezout_varpro_cholesky(model: &mut V3OptModelF64, dataset: &V3TrainDataset) {
    let trajs: Vec<StemTraj> = dataset
        .stems
        .iter()
        .map(|st| extract_traj_opt(model, st, None))
        .collect();
    solve_weighted_basis(model, dataset, &trajs, None);
}

/// Solves Bézout-weighted `VarPro` Cholesky using the actual `f32` `V3HotCore` pre-cabinet trajectory.
pub fn solve_bezout_varpro_cholesky_f32(model: &mut V3ColdTierModel, dataset: &V3TrainDataset) {
    let mut probe = model.clone();
    probe.svf.direct_taps_l = [0.0; 8];
    probe.svf.direct_taps_r = [0.0; 8];
    probe.direct_s2_w = [[0.0; MAX_CASCADE_LANES]; 2];
    probe.fir_l = [0.0; 64];
    probe.fir_r = [0.0; 64];
    probe.fir_l[0] = 1.0;
    probe.fir_r[0] = 1.0;
    probe.kautz_c_l = [(0.0, 0.0); MAX_KAUTZ_POLES];
    probe.kautz_c_r = [(0.0, 0.0); MAX_KAUTZ_POLES];

    let mut opt = V3OptModelF64::from_cold_model(model);
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&probe);
    let trajs: Vec<StemTraj> = dataset
        .stems
        .iter()
        .map(|st| {
            let n = st
                .in_l
                .len()
                .min(st.in_r.len())
                .min(st.tgt_l.len())
                .min(st.tgt_r.len());
            let (mut s2_l, mut s2_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
            let r_in = if model.mono_out {
                &st.in_l[..n]
            } else {
                &st.in_r[..n]
            };
            inst.reset_to_dc_equilibrium();
            inst.process_block_stereo(&st.in_l[..n], r_in, &mut s2_l, &mut s2_r);
            let ov = [
                s2_l.iter().map(|&v| v as f64).collect(),
                s2_r.iter().map(|&v| v as f64).collect(),
            ];
            extract_traj_opt(&opt, st, Some(&ov))
        })
        .collect();
    solve_weighted_basis(&mut opt, dataset, &trajs, None);
    *model = opt.to_cold_model();
}

/// Selects complex conjugate Kautz poles `start_pole..end_pole` via 120-pole Orthogonal Matching Pursuit.
pub fn select_omp_kautz_poles(
    model: &mut V3OptModelF64,
    dataset: &V3TrainDataset,
    start_pole: usize,
    end_pole: usize,
) {
    let end = end_pole.min(MAX_KAUTZ_POLES);
    if start_pole >= end {
        return;
    }
    let trajs: Vec<StemTraj> = dataset
        .stems
        .iter()
        .map(|st| extract_traj_opt(model, st, None))
        .collect();
    for m in start_pole..MAX_KAUTZ_POLES {
        model.kautz_c_l[m] = (0.0, 0.0);
        model.kautz_c_r[m] = (0.0, 0.0);
    }
    let mut sub_opt = model.clone();
    let c_cnt = tier_cascade_lanes(model.tier);
    let f_cnt = tier_fir_segments(model.tier) * 16;
    solve_weighted_basis(&mut sub_opt, dataset, &trajs, Some(start_pole));

    let active_ch = if model.mono_out { 1 } else { 2 };
    let mut res_w = Vec::<f64>::new();
    let mut bands = [Vec::new(), Vec::new(), Vec::new()];
    let mut s2_delayed_streams: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    for (st_idx, stem) in dataset.stems.iter().enumerate() {
        let n = trajs[st_idx].s2_out[0].len();
        let w = stem.warmup.min(n);
        if n <= w {
            continue;
        }
        for ch in 0..active_ch {
            let bw = dataset.stem_band_weights[st_idx][ch];
            let sq_w = [bw[0].sqrt(), bw[1].sqrt(), bw[2].sqrt()];
            let mut y_cur = vec![0.0_f64; n];
            let cols = build_channel_basis(
                model.tier,
                &sub_opt.kautz_poles,
                &trajs[st_idx],
                ch,
                Some(start_pole),
            );
            let dir_t = if ch == 0 {
                &sub_opt.svf_direct_taps_l
            } else {
                &sub_opt.svf_direct_taps_r
            };
            let fir = if ch == 0 {
                &sub_opt.fir_l
            } else {
                &sub_opt.fir_r
            };
            let kc = if ch == 0 {
                &sub_opt.kautz_c_l
            } else {
                &sub_opt.kautz_c_r
            };
            for t in 0..n {
                let mut yv = 0.0;
                for i in 0..7 {
                    yv += dir_t[i] * cols[i][t];
                }
                for l in 0..c_cnt {
                    yv += sub_opt.direct_s2_w[ch][l] * cols[7 + l][t];
                }
                for k in 0..f_cnt {
                    yv += fir[k] * cols[7 + c_cnt + k][t];
                }
                let kb = 7 + c_cnt + f_cnt;
                for m in 0..start_pole {
                    yv += kc[m].0 * cols[kb + 2 * m][t] + kc[m].1 * cols[kb + 2 * m + 1][t];
                }
                let tgt = if ch == 0 {
                    stem.tgt_l[t]
                } else {
                    stem.tgt_r[t]
                } as f64;
                y_cur[t] = tgt - yv;
            }
            dataset.bank.split_3band(&y_cur, &mut bands);
            for b in 0..3 {
                res_w.extend(bands[b][w..n].iter().map(|&v| v * sq_w[b]));
            }
            s2_delayed_streams.push((st_idx, ch, trajs[st_idx].s2_out[ch].clone()));
        }
    }
    if res_w.is_empty() {
        return;
    }

    // Build 160-pole dictionary (32 log-spaced freqs 18 Hz..16 kHz x 5 damping radii)
    let radii = [0.75_f64, 0.87, 0.94, 0.982, 0.9955];
    let mut dict_poles = Vec::with_capacity(160);
    let mut dict_re = Vec::with_capacity(160);
    let mut dict_im = Vec::with_capacity(160);
    let log_lo = 18.0_f64.ln();
    let log_hi = 16_000.0_f64.ln();
    for fi in 0..32 {
        let freq = (log_lo + (log_hi - log_lo) * (fi as f64) / 31.0).exp();
        let th = (2.0 * PI * freq / dataset.sample_rate_hz).clamp(1.0e-3, PI - 1.0e-3);
        let (st, ct) = th.sin_cos();
        for &r in &radii {
            let bg = (1.0 - r * r).max(1.0e-7).sqrt();
            let (lr, li) = (r * ct, r * st);
            let (mut col_re, mut col_im) = (
                Vec::with_capacity(res_w.len()),
                Vec::with_capacity(res_w.len()),
            );
            for (st_idx, ch, s2) in &s2_delayed_streams {
                let n = s2.len();
                let w = dataset.stems[*st_idx].warmup.min(n);
                let bw = dataset.stem_band_weights[*st_idx][*ch];
                let sq_w = [bw[0].sqrt(), bw[1].sqrt(), bw[2].sqrt()];
                let (mut raw_re, mut raw_im) = (vec![0.0_f64; n], vec![0.0_f64; n]);
                let (mut sr, mut si) = (0.0_f64, 0.0_f64);
                for t in 0..n {
                    let xd = if t >= 16 { s2[t - 16] } else { 0.0 };
                    let nr = lr * sr - li * si + bg * xd;
                    let ni = li * sr + lr * si;
                    sr = nr;
                    si = ni;
                    raw_re[t] = nr;
                    raw_im[t] = -ni;
                }
                dataset.bank.split_3band(&raw_re, &mut bands);
                for b in 0..3 {
                    col_re.extend(bands[b][w..n].iter().map(|&v| v * sq_w[b]));
                }
                dataset.bank.split_3band(&raw_im, &mut bands);
                for b in 0..3 {
                    col_im.extend(bands[b][w..n].iter().map(|&v| v * sq_w[b]));
                }
            }
            let n1 = dot4(&col_re, &col_re).sqrt().max(1.0e-15);
            for v in &mut col_re {
                *v /= n1;
            }
            let proj = dot4(&col_re, &col_im);
            for (v2, &v1) in col_im.iter_mut().zip(&col_re) {
                *v2 -= proj * v1;
            }
            let n2 = dot4(&col_im, &col_im).sqrt().max(1.0e-15);
            for v in &mut col_im {
                *v /= n2;
            }
            dict_poles.push((r, th));
            dict_re.push(col_re);
            dict_im.push(col_im);
        }
    }

    let mut chosen: Vec<(f64, f64)> = model.kautz_poles[..start_pole].to_vec();
    let mut orth_basis: Vec<Vec<f64>> = Vec::new();
    for slot in start_pole..end {
        let mut best_idx = 0;
        let mut best_score = -1.0_f64;
        for (di, &(r, th)) in dict_poles.iter().enumerate() {
            if chosen
                .iter()
                .any(|&(cr, ct)| (cr - r).abs() < 0.01 && (ct - th).abs() < 0.025)
            {
                continue;
            }
            let c1 = dot4(&res_w, &dict_re[di]);
            let c2 = dot4(&res_w, &dict_im[di]);
            let sc = c1 * c1 + c2 * c2;
            if sc > best_score {
                best_score = sc;
                best_idx = di;
            }
        }
        model.kautz_poles[slot] = dict_poles[best_idx];
        chosen.push(dict_poles[best_idx]);
        for cand in [&dict_re[best_idx], &dict_im[best_idx]] {
            let mut u = cand.clone();
            for q in &orth_basis {
                let p = dot4(&u, q);
                for (ui, &qi) in u.iter_mut().zip(q) {
                    *ui -= p * qi;
                }
            }
            let un = dot4(&u, &u).sqrt();
            if un > 1.0e-10 {
                for ui in &mut u {
                    *ui /= un;
                }
                let rp = dot4(&res_w, &u);
                for (ri, &ui) in res_w.iter_mut().zip(&u) {
                    *ri -= rp * ui;
                }
                orth_basis.push(u);
            }
        }
    }
    solve_weighted_basis(model, dataset, &trajs, None);
}
