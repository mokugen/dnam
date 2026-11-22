//! Bounded adaptive-ridge Cholesky solving, stopband-safe EMR `MM-Cholesky`, 33-band stereo
//! `LTAS` KKT refinement, and MSE-guarded Gram coordinate snapping for Pass-3 `VarPro`
//! (`ref[train.pipeline.nested-varpro-schur]`, `ref[qual.tiers.floors-not-stopping-criteria]`).

use crate::control::clamp::{MAX_GAIN_6DB_LIN, evaluate_readout_spectral_magnitudes};
use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_u_cholesky;
use crate::linalg::spectral_kkt::{
    EmrMmConfig, EmrSpectralProblem, LtasKktConfig, solve_emr_mm_cholesky, solve_ltas_newton_kkt,
};
use crate::losses::canonical_ltas_33_centers_hz;
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::gram_snap::{
    build_tier_fp16_col_bounds, snap_fp16_val, snap_weights_direct, snap_weights_gram_coordinate,
};

/// Canonical per-tier 33-band LTAS max tilt limits `[Lite, Full, Pro, Ultra]` in dB
/// (`ref[qual.tiers.floors-not-stopping-criteria]`).
const TIER_LTAS_TILT_BUDGET_DB: [f64; 4] = [0.75, 0.35, 0.15, 0.10];

/// Safe pre-snap magnitude ceiling leaving headroom inside the `[-8.0, +8.0]` readout wire range.
const MAX_READOUT_ABS_BEFORE_SNAP: f64 = 7.5;
const MAX_CANDIDATE_ABS_BEFORE_SNAP: f64 = 7.8;

pub(super) struct ChannelSpectralCtx<'a> {
    pub gram: &'a [f64],
    pub rhs: &'a [f64],
    pub feats_post: &'a [f32],
    pub tgts_post: &'a [f32],
    pub sr_hz: f64,
    pub ro_pole_48k: f64,
    pub active_fir_taps: usize,
    pub emit_prefix_overrides: bool,
    pub used_ridge: f64,
}

/// Evaluates mean-squared error across `tgts_post` for one channel weight vector of length `dim`.
pub(super) fn compute_channel_mse(
    weights: &[f64],
    feats_post: &[f32],
    tgts_post: &[f32],
    dim: usize,
) -> f64 {
    let n = tgts_post.len();
    let m = n.max(1);
    let mut sq = 0.0_f64;
    for (t, &tgt) in tgts_post.iter().enumerate() {
        let mut pred = 0.0_f64;
        for (c, &wc) in weights.iter().enumerate().take(dim) {
            pred += wc * f64::from(feats_post[c * n + t]);
        }
        let d = f64::from(tgt) - pred;
        sq += d * d;
    }
    sq / (m as f64)
}

/// Evaluates first-difference high-frequency error MSE `mean((e[t] - e[t-1])^2)` across
/// `tgts_post` for comparing `1x` Lite vs `2x` Full high-frequency fidelity.
pub(super) fn compute_channel_hf_mse(
    weights: &[f64],
    feats_post: &[f32],
    tgts_post: &[f32],
    dim: usize,
) -> f64 {
    let n = tgts_post.len();
    if n < 2 {
        return 0.0;
    }
    let mut prev_err = 0.0_f64;
    let mut sq = 0.0_f64;
    for (t, &tgt) in tgts_post.iter().enumerate() {
        let mut pred = 0.0_f64;
        for (c, &wc) in weights.iter().enumerate().take(dim) {
            pred += wc * f64::from(feats_post[c * n + t]);
        }
        let err = f64::from(tgt) - pred;
        if t > 0 {
            let de = err - prev_err;
            sq += de * de;
        }
        prev_err = err;
    }
    sq / ((n - 1) as f64)
}

/// Computes trace-scaled Cholesky regularization `(rel * mean_diag).max(1e-5)` over `dim` columns.
pub(super) fn trace_scaled_ridge(gram: &[f64], dim: usize, rel: f64) -> f64 {
    let mut trace = 0.0_f64;
    for j in 0..dim {
        trace += gram[uplo_u_index(j, j)].max(0.0);
    }
    let mean_diag = (trace / (dim.max(1) as f64)).max(1.0e-12);
    (rel * mean_diag).max(1.0e-5)
}

/// Checks wire amplitude & `+6 dB` readout spectral ceiling (`[MAX_GAIN_6DB_LIN]`).
#[inline]
pub(super) fn within_6db_and_bounds(
    weights: &[f64],
    ro_pole_48k: f64,
    sr_hz: f64,
    max_abs_bound: f64,
) -> bool {
    if weights.len() < 2
        || weights
            .iter()
            .any(|&v| !v.is_finite() || v.abs() > max_abs_bound)
    {
        return false;
    }
    let mags =
        evaluate_readout_spectral_magnitudes(&[weights[0], weights[1]], ro_pole_48k, sr_hz, 16);
    mags.iter().all(|&v| v.is_finite() && v <= MAX_GAIN_6DB_LIN)
}

/// Solves a prefix sub-system `(gram[..packed(dim)], rhs[..dim])` with escalating trace-scaled
/// ridge `[1e-5 .. 1.0] * mean_diag` until weights remain well-conditioned inside `[-7.5, +7.5]`
/// and respect the `+6 dB` head-gain limit.
pub(super) fn solve_bounded_uplo_u(
    gram: &[f64],
    rhs: &[f64],
    dim: usize,
    ro_pole_48k: f64,
    sr_hz: f64,
) -> Result<(Vec<f64>, f64), TrainError> {
    let packed = uplo_u_packed_len(dim);
    let g_sub = &gram[..packed];
    let r_sub = &rhs[..dim];
    let mut last_sol = vec![0.0_f64; dim];
    let mut last_ridge = trace_scaled_ridge(g_sub, dim, 1.0e-5);

    for &rel in &[1.0e-5_f64, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0] {
        let ridge = trace_scaled_ridge(g_sub, dim, rel);
        if let Ok(sol) = solve_uplo_u_cholesky(g_sub, r_sub, ridge) {
            last_ridge = ridge;
            last_sol = sol;
            if within_6db_and_bounds(&last_sol, ro_pole_48k, sr_hz, MAX_READOUT_ABS_BEFORE_SNAP) {
                return Ok((last_sol, ridge));
            }
        }
    }

    for v in &mut last_sol {
        *v = if v.is_finite() {
            v.clamp(-MAX_READOUT_ABS_BEFORE_SNAP, MAX_READOUT_ABS_BEFORE_SNAP)
        } else {
            0.0
        };
    }
    Ok((last_sol, last_ridge))
}

/// Applies stopband-safe EMR `MM-Cholesky`, active-band 33-band `LTAS` KKT refinement, and
/// MSE-guarded Gram coordinate `fp16`/`f32` snapping for one channel/tier.
pub(super) fn refine_channel_tier_emr_and_ltas_kkt(
    base_weights: &[f64],
    dim: usize,
    tier_idx: usize,
    ctx: &ChannelSpectralCtx<'_>,
) -> Vec<f64> {
    let m = ctx.tgts_post.len();
    let dft_len = m.min(2_048);
    let centers = canonical_ltas_33_centers_hz();
    let n_bands = centers.len();
    let packed_sub = uplo_u_packed_len(dim);
    let gram_sub = &ctx.gram[..packed_sub];
    let rhs_sub = &ctx.rhs[..dim];

    let mut basis_re_all = vec![0.0_f64; n_bands * dim];
    let mut basis_im_all = vec![0.0_f64; n_bands * dim];
    let mut tgt_re_all = vec![0.0_f64; n_bands];
    let mut tgt_im_all = vec![0.0_f64; n_bands];
    let mut band_rows_all = vec![0.0_f64; n_bands * dim];
    let mut band_amps_all = vec![0.0_f64; n_bands];

    for (b, &fc_hz) in centers.iter().enumerate() {
        let omega = core::f64::consts::TAU * fc_hz / ctx.sr_hz.max(8_000.0);
        let (mut yr, mut yi) = (0.0_f64, 0.0_f64);
        for t in 0..dft_len {
            let (s, c) = (omega * (t as f64)).sin_cos();
            let yt = f64::from(ctx.tgts_post[t]);
            yr += yt * c;
            yi -= yt * s;
            for col in 0..dim {
                let fv = f64::from(ctx.feats_post[col * m + t]);
                basis_re_all[b * dim + col] += fv * c;
                basis_im_all[b * dim + col] -= fv * s;
            }
        }
        let amp = (yr * yr + yi * yi).sqrt();
        tgt_re_all[b] = yr;
        tgt_im_all[b] = yi;
        band_amps_all[b] = amp;
        let (ur, ui) = if amp > 1.0e-9 {
            (yr / amp, yi / amp)
        } else {
            (1.0, 0.0)
        };
        for col in 0..dim {
            band_rows_all[b * dim + col] =
                ur * basis_re_all[b * dim + col] + ui * basis_im_all[b * dim + col];
        }
    }

    let max_band_amp = band_amps_all.iter().copied().fold(0.0_f64, f64::max);
    let passband_floor = (1.0e-2 * max_band_amp).max(1.0e-6);
    let active_indices: Vec<usize> = (0..n_bands)
        .filter(|&b| band_amps_all[b] >= passband_floor)
        .collect();
    let stop_indices: Vec<usize> = (0..n_bands)
        .filter(|&b| band_amps_all[b] < passband_floor)
        .collect();

    let n_act = active_indices.len();
    let mut act_freqs = Vec::with_capacity(n_act);
    let mut act_re = Vec::with_capacity(n_act * dim);
    let mut act_im = Vec::with_capacity(n_act * dim);
    let mut act_tgt_re = Vec::with_capacity(n_act);
    let mut act_tgt_im = Vec::with_capacity(n_act);
    let mut act_rows = Vec::with_capacity(n_act * dim);
    let mut act_amps = Vec::with_capacity(n_act);
    for &b in &active_indices {
        act_freqs.push(centers[b]);
        act_re.extend_from_slice(&basis_re_all[b * dim..(b + 1) * dim]);
        act_im.extend_from_slice(&basis_im_all[b * dim..(b + 1) * dim]);
        act_tgt_re.push(tgt_re_all[b]);
        act_tgt_im.push(tgt_im_all[b]);
        act_rows.extend_from_slice(&band_rows_all[b * dim..(b + 1) * dim]);
        act_amps.push(band_amps_all[b]);
    }

    let eval_max_tilt_db = |w: &[f64]| -> f64 {
        let mut mx = 0.0_f64;
        for b_loc in 0..n_act {
            let (mut pr, mut pi) = (0.0_f64, 0.0_f64);
            for (col, &wc) in w.iter().enumerate().take(dim) {
                pr += wc * act_re[b_loc * dim + col];
                pi += wc * act_im[b_loc * dim + col];
            }
            let p_amp = (pr * pr + pi * pi).sqrt().max(1.0e-12);
            let t_amp = act_amps[b_loc].max(1.0e-9);
            mx = mx.max((20.0 * (p_amp / t_amp).log10()).abs());
        }
        mx
    };

    let eval_stopband_power = |w: &[f64]| -> f64 {
        let mut pwr = 0.0_f64;
        for &b in &stop_indices {
            let (mut pr, mut pi) = (0.0_f64, 0.0_f64);
            for (col, &wc) in w.iter().enumerate().take(dim) {
                pr += wc * basis_re_all[b * dim + col];
                pi += wc * basis_im_all[b * dim + col];
            }
            pwr += pr * pr + pi * pi;
        }
        pwr
    };

    let within_6db = |w: &[f64]| -> bool {
        within_6db_and_bounds(w, ctx.ro_pole_48k, ctx.sr_hz, MAX_CANDIDATE_ABS_BEFORE_SNAP)
    };

    let base_mse = compute_channel_mse(base_weights, ctx.feats_post, ctx.tgts_post, dim);
    let mut best_w = base_weights[..dim].to_vec();
    let mut best_mse = base_mse;
    let mut best_tilt = eval_max_tilt_db(&best_w);
    let mut best_stop_pwr = eval_stopband_power(&best_w);

    if !stop_indices.is_empty() && dft_len >= 64 {
        let mut g_stop = gram_sub.to_vec();
        let stop_w = 0.5 * ((1 + 2 * tier_idx) as f64) / (dft_len as f64);
        for &b in &stop_indices {
            let bre = &basis_re_all[b * dim..(b + 1) * dim];
            let bim = &basis_im_all[b * dim..(b + 1) * dim];
            for j in 0..dim {
                let (rj, ij) = (bre[j], bim[j]);
                for i in 0..=j {
                    g_stop[uplo_u_index(i, j)] += stop_w * (bre[i] * rj + bim[i] * ij);
                }
            }
        }
        if let Ok((cand, _)) =
            solve_bounded_uplo_u(&g_stop, rhs_sub, dim, ctx.ro_pole_48k, ctx.sr_hz)
            && within_6db(&cand)
        {
            let cand_mse = compute_channel_mse(&cand, ctx.feats_post, ctx.tgts_post, dim);
            let cand_tilt = eval_max_tilt_db(&cand);
            let cand_stop_pwr = eval_stopband_power(&cand);
            let tilt_ok = active_indices.is_empty() || cand_tilt <= best_tilt + 0.5;
            if cand_stop_pwr < best_stop_pwr && cand_mse <= best_mse * 1.15 + 1.0e-12 && tilt_ok {
                best_w = cand;
                best_mse = cand_mse;
                best_tilt = cand_tilt;
                best_stop_pwr = cand_stop_pwr;
            }
        }
    }

    if n_act >= 2
        && let Ok(emr) = solve_emr_mm_cholesky(
            &EmrSpectralProblem {
                num_bands: n_act,
                feature_dim: dim,
                band_freqs_hz: &act_freqs,
                basis_re: &act_re,
                basis_im: &act_im,
                target_re: &act_tgt_re,
                target_im: &act_tgt_im,
            },
            EmrMmConfig {
                ridge: ctx.used_ridge.clamp(1.0e-8, 1.0e-2),
                ..EmrMmConfig::default()
            },
        )
        && within_6db(&emr.weights)
    {
        let emr_mse = compute_channel_mse(&emr.weights, ctx.feats_post, ctx.tgts_post, dim);
        let emr_tilt = eval_max_tilt_db(&emr.weights);
        let emr_stop_pwr = eval_stopband_power(&emr.weights);
        let stop_ok = stop_indices.is_empty() || emr_stop_pwr <= best_stop_pwr * 1.05 + 1.0e-12;
        if emr_mse < best_mse * 0.99 && emr_tilt <= best_tilt + 0.05 && stop_ok {
            best_w = emr.weights;
            best_mse = emr_mse;
            best_tilt = emr_tilt;
            best_stop_pwr = emr_stop_pwr;
        }
    }

    if n_act >= 2
        && best_tilt > TIER_LTAS_TILT_BUDGET_DB[tier_idx]
        && let Ok(kkt) = solve_ltas_newton_kkt(
            dim,
            n_act,
            gram_sub,
            rhs_sub,
            &act_rows,
            &act_amps,
            LtasKktConfig {
                ridge: ctx.used_ridge.max(1.0e-5),
                max_outer_iters: 8,
                max_tilt_db: TIER_LTAS_TILT_BUDGET_DB[tier_idx],
                penalty_rho: 2.0,
            },
        )
        && within_6db(&kkt.weights)
    {
        let kkt_mse = compute_channel_mse(&kkt.weights, ctx.feats_post, ctx.tgts_post, dim);
        let kkt_tilt = eval_max_tilt_db(&kkt.weights);
        let kkt_stop_pwr = eval_stopband_power(&kkt.weights);
        let stop_ok = stop_indices.is_empty() || kkt_stop_pwr <= best_stop_pwr * 1.05 + 1.0e-12;
        if kkt_mse <= best_mse * 1.5 && kkt_tilt < best_tilt && stop_ok {
            best_w = kkt.weights;
            best_mse = kkt_mse;
            best_tilt = kkt_tilt;
        }
    }

    let base_within_6db = within_6db(&best_w);
    let fp16_bounds =
        build_tier_fp16_col_bounds(tier_idx, ctx.active_fir_taps, ctx.emit_prefix_overrides);
    let mut fallback = snap_weights_direct(&best_w, &fp16_bounds);
    if base_within_6db && !within_6db(&fallback) {
        let (b0, b1) = (fallback[0], fallback[1]);
        for &scale_k in &[0.98_f64, 0.90, 0.75, 0.5, 0.0] {
            let mags =
                evaluate_readout_spectral_magnitudes(&[b0, b1], ctx.ro_pole_48k, ctx.sr_hz, 16);
            let peak = mags.iter().copied().fold(0.0_f64, f64::max);
            let adj = if peak > MAX_GAIN_6DB_LIN && peak.is_finite() {
                scale_k * (MAX_GAIN_6DB_LIN / peak)
            } else {
                scale_k
            };
            fallback[0] = snap_fp16_val(b0 * adj, -8.0, 8.0);
            fallback[1] = snap_fp16_val(b1 * adj, -8.0, 8.0);
            if within_6db(&fallback) {
                break;
            }
        }
    }

    let snapped =
        snap_weights_gram_coordinate(&best_w, gram_sub, ctx.used_ridge.max(1.0e-5), &fp16_bounds);
    let tilt_budget = best_tilt.max(TIER_LTAS_TILT_BUDGET_DB[tier_idx]) + 0.05;
    if snapped.iter().all(|v| v.is_finite()) && (!base_within_6db || within_6db(&snapped)) {
        let snap_mse = compute_channel_mse(&snapped, ctx.feats_post, ctx.tgts_post, dim);
        let fallback_mse = compute_channel_mse(&fallback, ctx.feats_post, ctx.tgts_post, dim);
        let snap_tilt = eval_max_tilt_db(&snapped);
        if snap_tilt <= tilt_budget
            && snap_mse <= best_mse * 1.15 + 1.0e-12
            && snap_mse <= fallback_mse * 1.05 + 1.0e-12
        {
            return snapped;
        }
    }

    fallback
}
