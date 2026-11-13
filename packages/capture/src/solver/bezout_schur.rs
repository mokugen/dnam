#![forbid(unsafe_code)]
//! Post-`LUFS`-Trim Multi-Tier Nested SPD Normal Equations & Selective `> 1.5 kHz` `MM-Cholesky`
//! Polish (`solver::bezout_schur`, `r[train.pipeline.stage-c-schur]`).

use dnam_engine::schema::DnamError;

use super::bezout_gram::{
    BezoutHarvestStats, eval_4tier_besr_quad_1ch, eval_4tier_post_trim_band_besr_lin,
    eval_4tier_post_trim_besr_lin,
};
use super::stage_b_c::{StageBHarvest, eval_all_4_prefix_residuals, solve_cholesky_robust};
use super::{
    factorize_uplo_u_cholesky, solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len,
};

#[inline]
fn quantize_effective_1ch(w_eff: &[f64], role_b_mult: &[[f64; 2]; 256], ch: usize) -> [f64; 256] {
    let mut out = [0.0_f64; 256];
    for idx in 0..256 {
        let m = role_b_mult[idx][ch];
        let inv_m = if m.abs() > 1.0e-6 { 1.0 / m } else { 1.0 };
        let stored = ((w_eff[idx] * inv_m) as f32).clamp(-16.0, 16.0);
        out[idx] = f64::from(stored) * m;
    }
    out
}

#[inline]
fn head_l1_norm(w_1ch: &[f64; 256]) -> f64 {
    let mut s = 0.0_f64;
    for &v in w_1ch {
        s += v.abs();
    }
    s
}

fn solve_1ch_uplo_u_robust(
    dim: usize,
    packed_gram: &[f64],
    rhs: &[f64],
    base_ridge: f64,
) -> Result<Vec<f64>, DnamError> {
    let mut rhs_2ch = vec![[0.0_f64; 2]; dim];
    for (dst, &r) in rhs_2ch.iter_mut().zip(rhs.iter().take(dim)) {
        dst[0] = r;
    }
    let mut ridge = base_ridge.max(1.0e-12);
    for _ in 0..6 {
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(packed_gram, &rhs_2ch, ridge) {
            return Ok(sol.into_iter().map(|v| v[0]).collect());
        }
        ridge *= 10.0;
    }
    let sol = solve_uplo_u_cholesky_2ch(packed_gram, &rhs_2ch, ridge.max(1.0e-3))
        .map_err(|e| DnamError::SchemaValidationError(format!("bESR Cholesky failed: {e}")))?;
    Ok(sol.into_iter().map(|v| v[0]).collect())
}

fn solve_factorized_1ch(dim: usize, r_up: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; dim];
    for j in 0..dim {
        let col = uplo_u_index(0, j);
        let mut s = rhs[j];
        for i in 0..j {
            s -= r_up[col + i] * y[i];
        }
        y[j] = s / r_up[col + j];
    }
    let mut x = vec![0.0_f64; dim];
    for i in (0..dim).rev() {
        let mut s = y[i];
        for j in (i + 1)..dim {
            s -= r_up[uplo_u_index(i, j)] * x[j];
        }
        x[i] = s / r_up[uplo_u_index(i, i)];
    }
    x
}

/// Result of the 3-band Bézout Multi-Tier Nested SPD Normal Equations (`r[train.pipeline.stage-c-schur]`).
#[derive(Debug, Clone)]
pub struct MultitierBesrSolution {
    pub effective_w0_256: Vec<[f64; 2]>,
    pub max_homotopy_eta: f64,
    pub prefix_besr_db: [f32; 4],
}

/// Diagonal regularization floor for out-of-class targets: damps unipolar, slow-envelope,
/// DC rectifier, unrequested bistable `T1SubHarmonicTracker` slots, and higher-tier slots that tend to
/// overfit the 250 ms coreset and diverge on holdout.
fn out_of_class_slot_diag_floor(d: usize, max_g_dd: f64, stats: &BezoutHarvestStats) -> f64 {
    let slot_frac = if matches!(d, 4..=11) {
        if stats.stage1_ext_fb_active {
            1.0e-5
        } else if stats.stage1_svf_chorus_fb_active {
            5.0e-4
        } else {
            8.0e-2
        }
    } else if matches!(
        d,
        47 | 51 | 100 | 108 | 146..=149 | 174 | 175 | 182 | 183 | 220..=223 | 237 | 239 | 242 | 245
    ) {
        2.5e-2
    } else if matches!(d, 45 | 46 | 49 | 50 | 157 | 165 | 228 | 234) {
        1.5e-3
    } else if stats.limiting_saturator_active
        && matches!(
            d,
            12..=19 | 24 | 32 | 39 | 43 | 44 | 48 | 58..=61 | 68..=83 | 136..=151 | 208..=219
        )
    {
        1.2e-2
    } else {
        0.0
    };
    let tier = d / 64;
    let tier_frac = if tier >= 1 {
        let tf = tier as f64;
        2.5e-5 * tf * tf
    } else {
        0.0
    };
    (slot_frac + tier_frac) * max_g_dd
}

/// Solves the 3-band Bézout Multi-Tier Nested SPD Normal Equations across `[Lite, Full, Pro, Ultra]`
/// (`r[train.pipeline.stage-c-schur]`).
pub fn solve_multitier_besr_4tier(
    stats: &BezoutHarvestStats,
    role_b_multipliers: &[[f64; 2]; 256],
    eval_stereo: bool,
) -> Result<MultitierBesrSolution, DnamError> {
    let n_pack = uplo_u_packed_len(256);
    let n_ch = if eval_stereo { 2 } else { 1 };
    let is_post_monotone =
        |b: &[f64; 4]| b[3] <= b[2] + 1.0e-12 && b[2] <= b[1] + 1.0e-12 && b[1] <= b[0] + 1.0e-12;
    let bands_monotone = |bm: &[[f64; 3]; 4]| -> bool {
        (0..3).all(|band| {
            bm[3][band] <= bm[2][band] + 1.0e-12
                && bm[2][band] <= bm[1][band] + 1.0e-12
                && bm[1][band] <= bm[0][band] + 1.0e-12
        })
    };

    let g_bb = &stats.tier_lufs_gram[3];
    let mut trace_bb = 0.0_f64;
    for d in 0..256 {
        trace_bb += g_bb[uplo_u_index(d, d)];
    }
    let mean_bb_diag = (trace_bb / 256.0).max(1.0e-9);
    let base_r = (1.0e-10 * mean_bb_diag).max(1.0e-13);

    // Scale regularization by the unweighted feature-bus Gram diagonal `mean_bb_diag` (not the
    // stopband-inflated trace of `tier_gram_lr`) so steep stopband weights never damp passbands!
    let mut reg_ridge = [1.0e-10_f64; 2];
    let mut cond_ridge = [1.0e-9_f64; 2];
    let mut y_tot = [1.0_f64; 2];
    let mut base_lin = [1.0_f64; 4];
    let mut opt_tier_lin = [0.0_f64; 4];
    let mut opt_w3_ch = [[0.0_f64; 256]; 2];

    for ch in 0..n_ch {
        let mut tr_w = 0.0_f64;
        for d in 0..256 {
            tr_w += stats.tier_gram_lr[3][ch][uplo_u_index(d, d)];
        }
        let mean_diag = (tr_w / 256.0).min(mean_bb_diag * 250.0).max(1.0e-12);
        reg_ridge[ch] = (2.0e-8 * mean_diag).max(1.0e-12);
        cond_ridge[ch] = (1.0e-7 * mean_diag).max(1.0e-11);
        y_tot[ch] = (stats.sampled_target_band_energy_lr[ch][0]
            + stats.sampled_target_band_energy_lr[ch][1]
            + stats.sampled_target_band_energy_lr[ch][2])
            .max(1.0e-15);
        for t in 0..4 {
            let b0 = (stats.tier_base_res_lr[t][ch] / y_tot[ch]).max(1.0);
            base_lin[t] = if ch == 0 { b0 } else { base_lin[t].max(b0) };
            let dim = (t + 1) * 64;
            let rhs_t: Vec<f64> = (0..dim).map(|d| stats.tier_cross_256[t][d][ch]).collect();
            let sol_t =
                solve_1ch_uplo_u_robust(dim, &stats.tier_gram_lr[t][ch], &rhs_t, reg_ridge[ch])?;
            let mut pad = [0.0_f64; 256];
            pad[..dim].copy_from_slice(&sol_t);
            let q_pad = quantize_effective_1ch(&pad, role_b_multipliers, ch);
            if t == 3 {
                opt_w3_ch[ch] = q_pad;
            }
            let ol = eval_4tier_besr_quad_1ch(stats, ch, &q_pad)[t] / y_tot[ch];
            opt_tier_lin[t] = if ch == 0 { ol } else { opt_tier_lin[t].max(ol) };
        }
    }

    let pack_lr = |ch_arr: &[[f64; 256]; 2]| -> [[f64; 2]; 256] {
        let mut out = [[0.0_f64; 2]; 256];
        for d in 0..256 {
            out[d][0] = ch_arr[0][d];
            out[d][1] = if eval_stereo {
                ch_arr[1][d]
            } else {
                ch_arr[0][d]
            };
        }
        out
    };
    let (opt_u_bands, _) =
        eval_4tier_post_trim_band_besr_lin(stats, &pack_lr(&opt_w3_ch), eval_stereo);
    let opt_ultra_bands_neg = opt_u_bands[3].iter().all(|&v| v < 0.998);

    // 0. Silent / muted target (`Y_tot <= 1e-12`): cancel backbone via sequential 64-D Schur
    // so all 4 tiers (`Lite..Ultra`) output `-70.000 LUFS` silence (`0.000 LU` spread).
    let raw_y_max = (0..n_ch)
        .map(|ch| stats.sampled_target_band_energy_lr[ch].iter().sum::<f64>())
        .fold(0.0_f64, f64::max);
    if raw_y_max <= 1.0e-12 {
        let silent_ridge = (1.0e-6 * mean_bb_diag).max(1.0e-11);
        let mut ch_ws = [[0.0_f64; 256]; 2];
        for ch in 0..n_ch {
            let mut ws = [0.0_f64; 256];
            for t in 0..4 {
                let start = t * 64;
                let g_t = &stats.tier_lufs_gram[t];
                let mut sub_g = vec![0.0_f64; uplo_u_packed_len(64)];
                for j in 0..64 {
                    let src_col = uplo_u_index(0, start + j) + start;
                    let dst_col = uplo_u_index(0, j);
                    sub_g[dst_col..dst_col + j + 1].copy_from_slice(&g_t[src_col..src_col + j + 1]);
                }
                if let Ok(r_up) = factorize_uplo_u_cholesky(64, &sub_g, silent_ridge)
                    .or_else(|_| factorize_uplo_u_cholesky(64, &sub_g, silent_ridge * 100.0))
                {
                    let mut sub_rhs = [0.0_f64; 64];
                    for i in 0..64 {
                        let row = start + i;
                        let col_base = uplo_u_index(0, row);
                        let mut c = stats.tier_bb_res_cross_256[t][row][ch];
                        for (prev, &g_ij) in g_t[col_base..col_base + start].iter().enumerate() {
                            c -= g_ij * ws[prev];
                        }
                        sub_rhs[i] = c;
                    }
                    let blk_w = solve_factorized_1ch(64, &r_up, &sub_rhs);
                    ws[start..start + 64].copy_from_slice(&blk_w);
                    let q_cur = quantize_effective_1ch(&ws, role_b_multipliers, ch);
                    ws[..start + 64].copy_from_slice(&q_cur[..start + 64]);
                }
            }
            ch_ws[ch] = ws;
        }
        return Ok(MultitierBesrSolution {
            effective_w0_256: pack_lr(&ch_ws).to_vec(),
            max_homotopy_eta: 0.0,
            prefix_besr_db: [-120.0; 4],
        });
    }
    let eval_bb_mse_ultra = |ch: usize, q: &[f64; 256]| -> f64 {
        let mut quad = 0.0_f64;
        let mut lin = 0.0_f64;
        for (j, &qj) in q.iter().enumerate() {
            lin += qj * stats.tier_bb_res_cross_256[3][j][ch];
            if qj == 0.0 {
                continue;
            }
            let col = uplo_u_index(0, j);
            let mut off = 0.0_f64;
            for i in 0..j {
                off += g_bb[col + i] * q[i];
            }
            quad += qj * (2.0 * off + g_bb[col + j] * qj);
        }
        (stats.tier_bb_res_energy_lr[3][ch] - 2.0 * lin + quad).max(0.0)
    };
    let mut q_in_ch = [[0.0_f64; 256]; 2];
    let mut inclass_ok = true;
    for ch in 0..n_ch {
        let rhs_u: [f64; 256] = std::array::from_fn(|d| stats.tier_bb_res_cross_256[3][d][ch]);
        let sol = solve_1ch_uplo_u_robust(256, g_bb, &rhs_u, base_r)?;
        q_in_ch[ch] = quantize_effective_1ch(&sol, role_b_multipliers, ch);
        let bb_u = eval_bb_mse_ultra(ch, &q_in_ch[ch]);
        if bb_u > 1.0e-5 * y_tot[ch] || head_l1_norm(&q_in_ch[ch]) > 128.0 {
            inclass_ok = false;
        }
    }
    let w_in_lr = pack_lr(&q_in_ch);
    let (b_in, _) = eval_4tier_post_trim_besr_lin(stats, &w_in_lr, eval_stereo);
    let (band_in, _) = eval_4tier_post_trim_band_besr_lin(stats, &w_in_lr, eval_stereo);
    if inclass_ok
        && opt_tier_lin[0] > 1.0e-3
        && is_post_monotone(&b_in)
        && bands_monotone(&band_in)
        && b_in[0] <= 0.75
        && b_in[0] <= opt_tier_lin[0] * 12.0 + 0.05
    {
        let db = b_in.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
        return Ok(MultitierBesrSolution {
            effective_w0_256: w_in_lr.to_vec(),
            max_homotopy_eta: 0.0,
            prefix_besr_db: db,
        });
    }

    let max_l1 = 128.0_f64;
    let no_prefix_blowup = |b: &[f64; 4], bands: &[[f64; 3]; 4]| {
        let tiers_ok = (0..4).all(|t| {
            let hard_neg = if opt_tier_lin[t] < 0.995 {
                0.995
            } else {
                base_lin[t] * 1.01 + 1.0e-12
            };
            let rel_cap = if t < 3 {
                opt_tier_lin[t] * 4.5 + 0.12
            } else {
                f64::INFINITY
            };
            b[t] <= (base_lin[t] * 1.01 + 1.0e-12).min(hard_neg).min(rel_cap)
        });
        let ultra_bands_ok = !opt_ultra_bands_neg || bands[3].iter().all(|&v| v < 0.998);
        tiers_ok && ultra_bands_ok
    };

    // 1b. Rank-deficient / band-limited in-class targets (e.g. 2-tone sinusoids with distinct
    // per-tier backbones): sequential 64-D Gauss-Markov broadband Schur cascade, gated on
    // monotone post-trim 3-band bESR, zero prefix blowup, and non-degraded Ultra bESR.
    if inclass_ok {
        let base_seq_bb_ridge = (1.0e-6 * mean_bb_diag).max(1.0e-11);
        for bb_mult in [1.0_f64, 16.0, 256.0, 4_096.0] {
            let mut ch_ws = [[0.0_f64; 256]; 2];
            let mut seq_bb_ok = true;
            for ch in 0..n_ch {
                let mut ws = [0.0_f64; 256];
                for t in 0..4 {
                    let start = t * 64;
                    let seq_bb_ridge = base_seq_bb_ridge * bb_mult * (1.0 + 2.0 * (t as f64));
                    let g_t = &stats.tier_lufs_gram[t];
                    let mut sub_g = vec![0.0_f64; uplo_u_packed_len(64)];
                    for j in 0..64 {
                        let row = start + j;
                        let src_col = uplo_u_index(0, row) + start;
                        let dst_col = uplo_u_index(0, j);
                        sub_g[dst_col..dst_col + j + 1]
                            .copy_from_slice(&g_t[src_col..src_col + j + 1]);
                        let is_slow_higher_tap = matches!(row, 116..=125 | 184..=189 | 246..=253)
                            || (opt_tier_lin[0] <= 1.0e-4
                                && matches!(row, 84..=115 | 152..=183 | 214..=245));
                        if is_slow_higher_tap {
                            sub_g[dst_col + j] += 1.0e5 * base_seq_bb_ridge;
                        }
                    }
                    let Ok(r_up) = factorize_uplo_u_cholesky(64, &sub_g, seq_bb_ridge)
                        .or_else(|_| factorize_uplo_u_cholesky(64, &sub_g, seq_bb_ridge * 100.0))
                    else {
                        seq_bb_ok = false;
                        break;
                    };
                    let mut sub_rhs = [0.0_f64; 64];
                    for i in 0..64 {
                        let row = start + i;
                        let col_base = uplo_u_index(0, row);
                        let mut c = stats.tier_bb_res_cross_256[t][row][ch];
                        for (prev, &g_ij) in g_t[col_base..col_base + start].iter().enumerate() {
                            c -= g_ij * ws[prev];
                        }
                        sub_rhs[i] = c;
                    }
                    let blk_w = solve_factorized_1ch(64, &r_up, &sub_rhs);
                    ws[start..start + 64].copy_from_slice(&blk_w);
                    let q_cur = quantize_effective_1ch(&ws, role_b_multipliers, ch);
                    ws[..start + 64].copy_from_slice(&q_cur[..start + 64]);
                }
                if !seq_bb_ok
                    || head_l1_norm(&ws) > max_l1
                    || eval_bb_mse_ultra(ch, &ws) > 1.0e-5 * y_tot[ch]
                {
                    seq_bb_ok = false;
                    break;
                }
                ch_ws[ch] = ws;
            }
            if seq_bb_ok {
                let w_seq_bb = pack_lr(&ch_ws);
                let (b_si, _) = eval_4tier_post_trim_besr_lin(stats, &w_seq_bb, eval_stereo);
                let (band_si, _) =
                    eval_4tier_post_trim_band_besr_lin(stats, &w_seq_bb, eval_stereo);
                if is_post_monotone(&b_si)
                    && bands_monotone(&band_si)
                    && no_prefix_blowup(&b_si, &band_si)
                    && b_si[3] <= opt_tier_lin[3] * 2.0 + 1.0e-5
                {
                    let db = b_si.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
                    return Ok(MultitierBesrSolution {
                        effective_w0_256: w_seq_bb.to_vec(),
                        max_homotopy_eta: 1.0,
                        prefix_besr_db: db,
                    });
                }
            }
        }
    }

    // 2. Out-of-class physical targets: exact per-tier sequential 64-D Schur cascade (eta = 1.0)

    let build_seq_lr = |ridge_scale: f64, g_est: [f64; 4]| -> Result<[[f64; 2]; 256], DnamError> {
        let mut ch_ws = [[0.0_f64; 256]; 2];
        for ch in 0..n_ch {
            let blk_ridge = cond_ridge[ch] * ridge_scale;
            let mut ws = [0.0_f64; 256];
            for t in 0..4 {
                let start = t * 64;
                let g_tier = &stats.tier_gram_lr[t][ch];
                let mut sub_g = vec![0.0_f64; uplo_u_packed_len(64)];
                for j in 0..64 {
                    let src_col = uplo_u_index(0, start + j) + start;
                    let dst_col = uplo_u_index(0, j);
                    sub_g[dst_col..dst_col + j + 1]
                        .copy_from_slice(&g_tier[src_col..src_col + j + 1]);
                }
                if !inclass_ok {
                    let max_g_dd = (0..((t + 1) * 64))
                        .map(|idx| g_tier[uplo_u_index(idx, idx)])
                        .fold(0.0_f64, f64::max)
                        .max(1.0e-12);
                    for j in 0..64 {
                        sub_g[uplo_u_index(j, j)] +=
                            out_of_class_slot_diag_floor(start + j, max_g_dd, stats);
                    }
                }
                let r_up = factorize_uplo_u_cholesky(64, &sub_g, blk_ridge)
                    .or_else(|_| factorize_uplo_u_cholesky(64, &sub_g, blk_ridge * 100.0))
                    .map_err(|e| DnamError::SchemaValidationError(format!("64-D bESR: {e}")))?;
                let inv_g_m1 = (1.0 / g_est[t].clamp(0.25, 4.0)) - 1.0;
                let mut sub_rhs = [0.0_f64; 64];
                for i in 0..64 {
                    let row = start + i;
                    let col_base = uplo_u_index(0, row);
                    let mut c = stats.tier_cross_256[t][row][ch]
                        + inv_g_m1 * stats.tier_tgt_fb_cross[t][row][ch];
                    for (prev, &g_ij) in g_tier[col_base..col_base + start].iter().enumerate() {
                        c -= g_ij * ws[prev];
                    }
                    sub_rhs[i] = c;
                }
                let blk_w = solve_factorized_1ch(64, &r_up, &sub_rhs);
                ws[start..start + 64].copy_from_slice(&blk_w);
                let q_cur = quantize_effective_1ch(&ws, role_b_multipliers, ch);
                ws[..start + 64].copy_from_slice(&q_cur[..start + 64]);
            }
            ch_ws[ch] = ws;
        }
        Ok(pack_lr(&ch_ws))
    };

    let feat_energy = (mean_bb_diag * (stats.post_warmup_frames.max(1) as f64)).max(1.0e-12);
    let gain_scale = (raw_y_max / feat_energy).sqrt().clamp(1.0, 2.5);
    let out_max_l1 = if inclass_ok {
        max_l1
    } else {
        (48.0 * gain_scale).min(112.0)
    };
    let l1_ok = |w_lr: &[[f64; 2]; 256]| -> bool {
        (0..n_ch).all(|ch| {
            let s: f64 = w_lr.iter().map(|v| v[ch].abs()).sum();
            s <= out_max_l1
        })
    };

    let all_monotone = |w_lr: &[[f64; 2]; 256]| -> ([f64; 4], [[f64; 3]; 4], [f64; 4], bool, bool) {
        let (comp, g_next) = eval_4tier_post_trim_besr_lin(stats, w_lr, eval_stereo);
        let (bands, _) = eval_4tier_post_trim_band_besr_lin(stats, w_lr, eval_stereo);
        let comp_ok = is_post_monotone(&comp);
        (
            comp,
            bands,
            g_next,
            comp_ok && (inclass_ok || bands_monotone(&bands)),
            comp_ok,
        )
    };

    let mut seq_scale = 1.0_f64;
    let mut w_seq = [[0.0_f64; 2]; 256];
    let mut seq_l1_ok = false;
    for s in [
        1.0_f64,
        16.0,
        256.0,
        4_096.0,
        65_536.0,
        1_048_576.0,
        16_777_216.0,
        268_435_456.0,
    ] {
        let cand = build_seq_lr(s, [1.0; 4])?;
        if l1_ok(&cand) || s == 268_435_456.0 {
            seq_scale = s;
            w_seq = cand;
            seq_l1_ok = true;
            break;
        }
    }
    let (mut b_seq, mut bands_seq, mut g_seq, mut seq_all_mono, mut seq_comp_mono) =
        all_monotone(&w_seq);
    if seq_l1_ok {
        for _ in 0..2 {
            let w_next = build_seq_lr(seq_scale, g_seq)?;
            if !l1_ok(&w_next) {
                break;
            }
            let (b_next, bands_next, g_next, next_all_mono, next_comp_mono) = all_monotone(&w_next);
            let improves = (next_all_mono && (!seq_all_mono || b_next[3] <= b_seq[3]))
                || (!seq_all_mono && next_comp_mono && (!seq_comp_mono || b_next[3] <= b_seq[3]))
                || (!seq_comp_mono && b_next[2] - b_next[1] < b_seq[2] - b_seq[1]);
            if improves {
                w_seq = w_next;
                b_seq = b_next;
                bands_seq = bands_next;
                seq_all_mono = next_all_mono;
                seq_comp_mono = next_comp_mono;
            }
            for t in 0..4 {
                g_seq[t] = (0.5 * g_seq[t] + 0.5 * g_next[t]).clamp(0.25, 4.0);
            }
        }
    }

    let fb_key = |b: &[f64; 4], bands: &[[f64; 3]; 4], all_m: bool| -> (usize, usize, bool, f64) {
        let pos_tiers = b.iter().filter(|&&v| v >= 0.995).count();
        let pos_ubands = bands[3].iter().filter(|&&v| v >= 0.998).count();
        let tie_err = if inclass_ok { b[3] } else { b[3] + 0.15 * b[0] };
        (pos_tiers, pos_ubands, !all_m, tie_err)
    };
    let mut best_w = w_seq;
    let mut best_eta = 1.0_f64;
    let mut best_b = b_seq;
    let mut has_valid_best = seq_all_mono && no_prefix_blowup(&b_seq, &bands_seq);
    type BezoutFallback = ([[f64; 2]; 256], f64, [f64; 4], [[f64; 3]; 4], bool);
    let mut best_fb: Option<BezoutFallback> =
        seq_comp_mono.then_some((w_seq, 1.0, b_seq, bands_seq, seq_all_mono));
    let mut g_weighted = vec![0.0_f64; n_pack];
    let mut rhs_weighted = [0.0_f64; 256];

    // Sweep exact Multi-Tier SPD Normal Equations summing exact per-tier Hessians H^(gamma, g) w = b^(gamma, g)
    for (eta, gammas) in [
        (0.0_f64, [0.0_f64, 0.0, 0.0, 1.0]),
        (0.08, [0.15, 0.30, 0.55, 1.0]),
        (0.16, [0.35, 0.55, 0.75, 1.0]),
        (0.28, [0.75, 0.85, 0.95, 1.0]),
        (0.45, [2.00, 1.50, 1.20, 1.0]),
        (0.68, [8.00, 3.50, 1.80, 1.0]),
        (0.88, [36.0, 10.0, 3.00, 1.0]),
        (0.97, [144.0, 28.0, 5.50, 1.0]),
        (0.99, [200.0, 80.0, 32.0, 1.0]),
    ] {
        for ridge_mult in [
            1.0_f64,
            8.0,
            64.0,
            512.0,
            4_096.0,
            32_768.0,
            262_144.0,
            2_097_152.0,
            16_777_216.0,
        ] {
            let mut g_kkt = [1.0_f64; 4];
            let mut ridge_found_valid = false;
            for kkt_pass in 0..3 {
                let g_sq = g_kkt.map(|g| g * g);
                let big_gamma = [
                    gammas[0] * g_sq[0]
                        + gammas[1] * g_sq[1]
                        + gammas[2] * g_sq[2]
                        + gammas[3] * g_sq[3],
                    gammas[1] * g_sq[1] + gammas[2] * g_sq[2] + gammas[3] * g_sq[3],
                    gammas[2] * g_sq[2] + gammas[3] * g_sq[3],
                    gammas[3] * g_sq[3],
                ];
                let mut cand_ch = [[0.0_f64; 256]; 2];
                let mut solve_ok = true;
                for ch in 0..n_ch {
                    let cur_ridge = reg_ridge[ch] * ridge_mult;
                    g_weighted.fill(0.0);
                    rhs_weighted.fill(0.0);
                    for t in 0..4 {
                        let w_t = gammas[t] * g_sq[t];
                        if w_t <= 0.0 {
                            continue;
                        }
                        let dim = (t + 1) * 64;
                        let p_len = uplo_u_packed_len(dim);
                        let g_tier = &stats.tier_gram_lr[t][ch];
                        for idx in 0..p_len {
                            g_weighted[idx] += w_t * g_tier[idx];
                        }
                        let gt = g_kkt[t];
                        let w_k = gammas[t] * gt * (1.0 - gt);
                        for d in 0..dim {
                            rhs_weighted[d] += w_t * stats.tier_cross_256[t][d][ch]
                                + w_k * stats.tier_tgt_fb_cross[t][d][ch];
                        }
                    }
                    let max_g_dd = if !inclass_ok {
                        (0..256)
                            .map(|d| g_weighted[uplo_u_index(d, d)])
                            .fold(0.0_f64, f64::max)
                            .max(1.0e-12)
                    } else {
                        0.0
                    };
                    for d in 0..256 {
                        let extra = if !inclass_ok {
                            out_of_class_slot_diag_floor(d, max_g_dd, stats)
                        } else {
                            0.0
                        };
                        g_weighted[uplo_u_index(d, d)] += big_gamma[d / 64] * cur_ridge + extra;
                    }
                    let Ok(cand_raw) = solve_1ch_uplo_u_robust(
                        256,
                        &g_weighted,
                        &rhs_weighted,
                        cur_ridge * 1.0e-3,
                    ) else {
                        solve_ok = false;
                        break;
                    };
                    cand_ch[ch] = quantize_effective_1ch(&cand_raw, role_b_multipliers, ch);
                }
                if !solve_ok {
                    break;
                }
                let cand_lr = pack_lr(&cand_ch);
                if !l1_ok(&cand_lr) {
                    break;
                }
                let (b_cand, bands_cand, g_next, all_mono, comp_mono) = all_monotone(&cand_lr);
                if comp_mono
                    && best_fb.as_ref().is_none_or(|(_, _, mb, mbands, mall)| {
                        fb_key(&b_cand, &bands_cand, all_mono) < fb_key(mb, mbands, *mall)
                    })
                {
                    best_fb = Some((cand_lr, eta, b_cand, bands_cand, all_mono));
                }
                if all_mono && no_prefix_blowup(&b_cand, &bands_cand) {
                    ridge_found_valid = true;
                    if !has_valid_best || b_cand[3] <= best_b[3] {
                        best_w = cand_lr;
                        best_eta = eta;
                        best_b = b_cand;
                        has_valid_best = true;
                    }
                }
                if kkt_pass == 0 && g_next.iter().all(|&g| (g - 1.0).abs() < 1.0e-3) {
                    break;
                }
                for t in 0..4 {
                    g_kkt[t] = (0.5 * g_kkt[t] + 0.5 * g_next[t]).clamp(0.25, 4.0);
                }
            }
            if ridge_found_valid || (inclass_ok && best_fb.is_some()) {
                break;
            }
        }
        if has_valid_best && best_eta == 0.0 && best_b[0] <= opt_tier_lin[0] * 1.35 + 1.0e-4 {
            break;
        }
    }

    if !has_valid_best && let Some((mw, meta, mb, _, _)) = best_fb {
        best_w = mw;
        best_eta = meta;
        best_b = mb;
    }

    let prefix_besr_db = best_b.map(|v| (10.0 * v.max(1.0e-16).log10()) as f32);
    Ok(MultitierBesrSolution {
        effective_w0_256: best_w.to_vec(),
        max_homotopy_eta: best_eta,
        prefix_besr_db,
    })
}

/// Applies Hann-windowed Theorem 5 Selective `1/12`-Octave (`> 1.5 kHz`, $|\Delta L_b| > 0.25\text{ dB}$)
/// Majorize-Minimize polish on `w_cand`, strictly gated on non-regressing post-trim 3-band `bESR` & 4-tier monotonicity.
pub(super) fn apply_selective_hf_mm_and_ltas_polish(
    harvest: &StageBHarvest,
    packed_gram_256: &[f64],
    base_ridge: f64,
    role_b_multipliers: &[[f64; 2]; 256],
    w_cand: &mut [[f64; 2]],
) -> usize {
    const WIN_N: usize = 256;
    const HF_BINS: [usize; 16] = [
        9, 11, 13, 16, 19, 23, 27, 32, 38, 45, 53, 63, 74, 84, 94, 105,
    ];
    let n_total = harvest.residual_lr.len();
    let use_polish_win = harvest.polish_win_features.len() >= WIN_N;
    if harvest.weighted_target_energy <= 1.0e-12
        || n_total < WIN_N
        || (!use_polish_win && harvest.features.len() < WIN_N)
    {
        return 0;
    }
    let e_init = eval_all_4_prefix_residuals(
        packed_gram_256,
        &harvest.cross_256,
        w_cand,
        harvest.weighted_residual_energy,
    )[3];
    if e_init <= 1.0e-5 * harvest.weighted_target_energy.max(1.0e-12) {
        return 0;
    }

    let win_start = (n_total / 3).min(n_total - WIN_N);
    let mut hann = [0.0_f64; WIN_N];
    let mut pred = [[0.0_f64; 2]; WIN_N];
    for t in 0..WIN_N {
        hann[t] = 0.5 * (1.0 - (std::f64::consts::TAU * (t as f64) / (WIN_N as f64)).cos());
        let fb = if use_polish_win {
            &harvest.polish_win_features[t]
        } else {
            &harvest.features[win_start + t]
        };
        let (mut sl, mut sr) = (0.0_f64, 0.0_f64);
        for d in 0..256 {
            let fd = f64::from(fb[d]);
            sl += fd * w_cand[d][0];
            sr += fd * w_cand[d][1];
        }
        pred[t] = [sl, sr];
    }

    let mut relaxed_count = 0usize;
    let mut delta_time = [[0.0_f64; 2]; WIN_N];
    for &k_bin in &HF_BINS {
        let (s_step, c_step) = (std::f64::consts::TAU * (k_bin as f64) / (WIN_N as f64)).sin_cos();
        let (mut tgt_re_l, mut tgt_im_l, mut prd_re_l, mut prd_im_l) = (0.0, 0.0, 0.0, 0.0);
        let (mut tgt_re_r, mut tgt_im_r, mut prd_re_r, mut prd_im_r) = (0.0, 0.0, 0.0, 0.0);
        let (mut c, mut s) = (1.0_f64, 0.0_f64);
        for t in 0..WIN_N {
            let w_h = hann[t];
            let ([tl, tr], [pl, pr]) = (harvest.residual_lr[win_start + t], pred[t]);
            tgt_re_l += (w_h * tl) * c;
            tgt_im_l -= (w_h * tl) * s;
            prd_re_l += (w_h * pl) * c;
            prd_im_l -= (w_h * pl) * s;
            tgt_re_r += (w_h * tr) * c;
            tgt_im_r -= (w_h * tr) * s;
            prd_re_r += (w_h * pr) * c;
            prd_im_r -= (w_h * pr) * s;
            (c, s) = (c * c_step - s * s_step, s * c_step + c * s_step);
        }
        let e_tgt =
            (tgt_re_l * tgt_re_l + tgt_im_l * tgt_im_l + tgt_re_r * tgt_re_r + tgt_im_r * tgt_im_r)
                .max(1.0e-15);
        let e_prd =
            (prd_re_l * prd_re_l + prd_im_l * prd_im_l + prd_re_r * prd_re_r + prd_im_r * prd_im_r)
                .max(1.0e-15);
        if e_tgt < 1.0e-6 || (10.0 * (e_prd / e_tgt).log10()).abs() <= 0.25 {
            continue;
        }
        relaxed_count += 1;
        let norm = ((e_tgt / e_prd).sqrt().clamp(0.5, 2.0) - 1.0) * (4.0 / (WIN_N as f64));
        let (dl_re, dl_im, dr_re, dr_im) = (
            norm * prd_re_l,
            norm * prd_im_l,
            norm * prd_re_r,
            norm * prd_im_r,
        );
        let (mut c, mut s) = (1.0_f64, 0.0_f64);
        for (t, dt_lr) in delta_time.iter_mut().enumerate() {
            let w_h = 0.12 * hann[t];
            dt_lr[0] += w_h * (dl_re * c - dl_im * s);
            dt_lr[1] += w_h * (dr_re * c - dr_im * s);
            (c, s) = (c * c_step - s * s_step, s * c_step + c * s_step);
        }
    }
    if relaxed_count == 0 {
        return 0;
    }

    let mut delta_c = vec![[0.0_f64; 2]; 256];
    let scale = (n_total as f64) / (WIN_N as f64);
    for (t, &[dl, dr]) in delta_time.iter().enumerate() {
        let fb = if use_polish_win {
            &harvest.polish_win_features[t]
        } else {
            &harvest.features[win_start + t]
        };
        let w = harvest.frame_weights[win_start + t] * scale;
        for d in 0..256 {
            let wfd = w * f64::from(fb[d]);
            delta_c[d][0] += wfd * dl;
            delta_c[d][1] += wfd * dr;
        }
    }

    let mut accepted = false;
    if let Ok(dw) = solve_cholesky_robust(packed_gram_256, &delta_c, base_ridge * 10.0) {
        if let Some(ref stats) = harvest.besr_stats {
            let n_ch = if harvest.eval_stereo { 2 } else { 1 };
            let mut cur_lr = [[0.0_f64; 2]; 256];
            cur_lr.copy_from_slice(&w_cand[..256]);
            let (b_before, _) = eval_4tier_post_trim_besr_lin(stats, &cur_lr, harvest.eval_stereo);
            let (band_before, _) =
                eval_4tier_post_trim_band_besr_lin(stats, &cur_lr, harvest.eval_stereo);
            let was_bands_mono = (0..3).all(|b| {
                band_before[3][b] <= band_before[2][b]
                    && band_before[2][b] <= band_before[1][b]
                    && band_before[1][b] <= band_before[0][b]
            });
            for step in [0.5_f64, 0.15, 0.04] {
                let mut raw_ch = [[0.0_f64; 256]; 2];
                let mut q_ch = [[0.0_f64; 256]; 2];
                for ch in 0..n_ch {
                    for d in 0..256 {
                        raw_ch[ch][d] = cur_lr[d][ch] + step * dw[d][ch];
                    }
                    q_ch[ch] = quantize_effective_1ch(&raw_ch[ch], role_b_multipliers, ch);
                }
                if !(0..n_ch).all(|ch| head_l1_norm(&q_ch[ch]) <= 128.0) {
                    continue;
                }
                let mut cand_lr = [[0.0_f64; 2]; 256];
                for d in 0..256 {
                    cand_lr[d][0] = q_ch[0][d];
                    cand_lr[d][1] = q_ch[usize::from(harvest.eval_stereo)][d];
                }
                let (ba, _) = eval_4tier_post_trim_besr_lin(stats, &cand_lr, harvest.eval_stereo);
                let (ba_band, _) =
                    eval_4tier_post_trim_band_besr_lin(stats, &cand_lr, harvest.eval_stereo);
                let now_bands_mono = (0..3).all(|b| {
                    ba_band[3][b] <= ba_band[2][b]
                        && ba_band[2][b] <= ba_band[1][b]
                        && ba_band[1][b] <= ba_band[0][b]
                });
                let lower_ok = (0..3).all(|t| {
                    ba[t] <= b_before[t] * 1.02 + 1.0e-5 && (b_before[t] >= 1.0 || ba[t] < 1.0)
                });
                if ba[3] <= b_before[3]
                    && ba[3] <= ba[2]
                    && ba[2] <= ba[1]
                    && ba[1] <= ba[0]
                    && (!was_bands_mono || now_bands_mono)
                    && lower_ok
                {
                    w_cand[..256].copy_from_slice(&cand_lr);
                    accepted = true;
                    break;
                }
            }
        } else {
            let eb = eval_all_4_prefix_residuals(
                packed_gram_256,
                &harvest.cross_256,
                w_cand,
                harvest.weighted_residual_energy,
            );
            for step in [0.5_f64, 0.15, 0.04] {
                let mut cand = w_cand.to_vec();
                for (w, d) in cand.iter_mut().zip(&dw) {
                    w[0] += step * d[0];
                    w[1] += step * d[1];
                }
                if !(0..2).all(|ch| cand.iter().map(|v| v[ch].abs()).sum::<f64>() <= 128.0) {
                    continue;
                }
                let ea = eval_all_4_prefix_residuals(
                    packed_gram_256,
                    &harvest.cross_256,
                    &cand,
                    harvest.weighted_residual_energy,
                );
                let mono = ea[3] <= ea[2] && ea[2] <= ea[1] && ea[1] <= ea[0];
                let lower_ok = (0..3).all(|t| ea[t] <= eb[t] * 1.02 + 1.0e-9);
                if ea[3] <= eb[3] && mono && lower_ok {
                    w_cand.copy_from_slice(&cand);
                    accepted = true;
                    break;
                }
            }
        }
    }
    if accepted { relaxed_count } else { 0 }
}
