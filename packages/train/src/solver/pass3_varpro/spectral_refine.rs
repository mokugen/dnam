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
    ColWireSpec, build_tier_col_wire_specs, snap_fp16_val, snap_weights_direct,
    snap_weights_gram_coordinate,
};

/// Canonical per-tier 33-band LTAS max tilt limits `[Lite, Full, Pro, Ultra]` in dB
/// (`ref[qual.tiers.floors-not-stopping-criteria]`).
const TIER_LTAS_TILT_BUDGET_DB: [f64; 4] = [0.75, 0.35, 0.15, 0.10];

/// Safe pre-snap magnitude ceiling leaving headroom inside the `[-8.0, +8.0]` readout wire range.
const MAX_READOUT_ABS_BEFORE_SNAP: f64 = 7.5;
const MAX_CANDIDATE_ABS_BEFORE_SNAP: f64 = 7.8;
/// Maximum BIBO l1 gain norm across linear readout + FIR columns (`ref[train.audit.phase2.clamp-rail]`).
const MAX_READOUT_L1_NORM: f64 = 48.0;
const _: () = assert!(MAX_READOUT_L1_NORM > 0.0 && MAX_READOUT_L1_NORM < 100.0);

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
    let l1_norm: f64 = weights.iter().map(|v| v.abs()).sum();
    if !l1_norm.is_finite() || l1_norm > MAX_READOUT_L1_NORM {
        return false;
    }
    let mags =
        evaluate_readout_spectral_magnitudes(&[weights[0], weights[1]], ro_pole_48k, sr_hz, 16);
    mags.iter().all(|&v| v.is_finite() && v <= MAX_GAIN_6DB_LIN)
}

/// Solves a prefix sub-system `(gram[..packed(dim)], rhs[..dim])` with escalating trace-scaled
/// ridge `[1e-5 .. 1e4] * mean_diag` until weights remain well-conditioned inside `[-7.5, +7.5]`,
/// respect the BIBO `l1` norm ceiling, and respect the `+6 dB` head-gain limit.
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

    for &rel in &[
        1.0e-5_f64, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4,
    ] {
        let ridge = trace_scaled_ridge(g_sub, dim, rel);
        if let Ok(sol) = solve_uplo_u_cholesky(g_sub, r_sub, ridge) {
            last_ridge = ridge;
            last_sol = sol;
            if within_6db_and_bounds(&last_sol, ro_pole_48k, sr_hz, MAX_READOUT_ABS_BEFORE_SNAP) {
                return Ok((last_sol, ridge));
            }
        }
    }

    let mut max_abs = 0.0_f64;
    let mut l1_norm = 0.0_f64;
    for v in &mut last_sol {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            let av = v.abs();
            max_abs = max_abs.max(av);
            l1_norm += av;
        }
    }
    let scale = (MAX_READOUT_ABS_BEFORE_SNAP / max_abs.max(1.0e-12))
        .min(MAX_READOUT_L1_NORM / l1_norm.max(1.0e-12))
        .min(1.0);
    for v in &mut last_sol {
        *v *= scale;
    }
    Ok((last_sol, last_ridge))
}

struct KktResnapCtx<'a> {
    ctx: &'a ChannelSpectralCtx<'a>,
    tier_idx: usize,
    dim: usize,
    n_act: usize,
    gram_sub: &'a [f64],
    rhs_sub: &'a [f64],
    act_rows: &'a [f64],
    act_amps: &'a [f64],
    act_re: &'a [f64],
    act_im: &'a [f64],
    lambda_diag: &'a [f64],
    col_specs: &'a [ColWireSpec],
    best_mse: f64,
    tilt_budget: f64,
    base_within_6db: bool,
}

fn eval_act_max_tilt_db(
    w: &[f64],
    act_re: &[f64],
    act_im: &[f64],
    act_amps: &[f64],
    dim: usize,
) -> f64 {
    let mut mx = 0.0_f64;
    for (b_loc, &t_amp_raw) in act_amps.iter().enumerate() {
        let (mut pr, mut pi) = (0.0_f64, 0.0_f64);
        for (col, &wc) in w.iter().enumerate().take(dim) {
            pr += wc * act_re[b_loc * dim + col];
            pi += wc * act_im[b_loc * dim + col];
        }
        let p_amp = (pr * pr + pi * pi).sqrt().max(1.0e-12);
        let t_amp = t_amp_raw.max(1.0e-9);
        mx = mx.max((20.0 * (p_amp / t_amp).log10()).abs());
    }
    mx
}

fn try_damped_kkt_resnap(
    p: &KktResnapCtx<'_>,
    snap_tilt: f64,
    fallback_tilt: f64,
) -> Option<Vec<f64>> {
    let ridge = p.ctx.used_ridge.max(1.0e-5);
    let target_tilt = (TIER_LTAS_TILT_BUDGET_DB[p.tier_idx] * 0.85).max(0.05);
    let kkt2 = solve_ltas_newton_kkt(
        p.dim,
        p.n_act,
        p.gram_sub,
        p.rhs_sub,
        p.act_rows,
        p.act_amps,
        LtasKktConfig {
            ridge,
            max_outer_iters: 6,
            max_tilt_db: target_tilt,
            penalty_rho: 4.0,
        },
    )
    .ok()?;
    if !kkt2.weights.iter().all(|v| v.is_finite())
        || !within_6db_and_bounds(
            &kkt2.weights,
            p.ctx.ro_pole_48k,
            p.ctx.sr_hz,
            MAX_CANDIDATE_ABS_BEFORE_SNAP,
        )
    {
        return None;
    }
    let resnapped =
        snap_weights_gram_coordinate(&kkt2.weights, p.gram_sub, p.lambda_diag, p.col_specs);
    let within = !p.base_within_6db
        || within_6db_and_bounds(
            &resnapped,
            p.ctx.ro_pole_48k,
            p.ctx.sr_hz,
            MAX_CANDIDATE_ABS_BEFORE_SNAP,
        );
    if !resnapped.iter().all(|v| v.is_finite()) || !within {
        return None;
    }
    let resnap_tilt = eval_act_max_tilt_db(&resnapped, p.act_re, p.act_im, p.act_amps, p.dim);
    let resnap_mse = compute_channel_mse(&resnapped, p.ctx.feats_post, p.ctx.tgts_post, p.dim);
    if (resnap_tilt <= p.tilt_budget || resnap_tilt < snap_tilt.min(fallback_tilt))
        && resnap_mse <= p.best_mse * 1.5 + 1.0e-12
    {
        Some(resnapped)
    } else {
        None
    }
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
    let col_specs =
        build_tier_col_wire_specs(tier_idx, ctx.active_fir_taps, ctx.emit_prefix_overrides);
    let mut fallback = snap_weights_direct(&best_w, &col_specs);
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
        if !within_6db(&fallback) {
            for &scale_all in &[0.98_f64, 0.90, 0.75, 0.5, 0.25, 0.0] {
                let scaled: Vec<f64> = best_w.iter().map(|&v| v * scale_all).collect();
                fallback = snap_weights_direct(&scaled, &col_specs);
                if within_6db(&fallback) {
                    break;
                }
            }
        }
    }

    let lambda_diag = vec![ctx.used_ridge.max(1.0e-5); dim];
    let snapped = snap_weights_gram_coordinate(&best_w, gram_sub, &lambda_diag, &col_specs);
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
        if n_act >= 2 && snap_tilt > tilt_budget {
            let fallback_tilt = eval_max_tilt_db(&fallback);
            let resnap_ctx = KktResnapCtx {
                ctx,
                tier_idx,
                dim,
                n_act,
                gram_sub,
                rhs_sub,
                act_rows: &act_rows,
                act_amps: &act_amps,
                act_re: &act_re,
                act_im: &act_im,
                lambda_diag: &lambda_diag,
                col_specs: &col_specs,
                best_mse,
                tilt_budget,
                base_within_6db,
            };
            if let Some(resnapped) = try_damped_kkt_resnap(&resnap_ctx, snap_tilt, fallback_tilt) {
                return resnapped;
            }
        }
    }

    fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::varpro::assemble_ozaki_normal_equations;
    use crate::runner::TrainRunner;

    #[test]
    fn test_post_snap_damped_kkt_resnap_and_budget_guards() {
        // Construct a correlated multi-tone system where Ultra column-0/2 quantization can perturb
        // 33-band tilt above TIER_LTAS_TILT_BUDGET_DB[3] = 0.10 dB, exercising damped KKT resnap
        // and verifying the output stays finite, +6 dB safe, and tilt-controlled.
        let m = 512_usize;
        let dim = 250_usize; // 2 + 40*3 + 128 FIR at Ultra
        let sr = 48_000.0_f64;
        let runner = TrainRunner::cpu();
        let mut feats = vec![0.0_f32; dim * m];
        let mut tgts = vec![0.0_f32; m];
        for t in 0..m {
            let th = core::f64::consts::TAU * (t as f64) / sr;
            let s_low = (th * 125.0).sin() + 0.6 * (th * 500.0).cos();
            let s_hi = (th * 2_500.0).sin() + 0.5 * (th * 6_300.0).cos();
            let f0 = (0.5 * s_low + 0.05 * s_hi) as f32;
            let f1 = (0.04 * s_low + 0.45 * s_hi) as f32;
            feats[t] = f0;
            feats[m + t] = f1;
            for c in 2..dim {
                feats[c * m + t] = if t >= (c % 8) {
                    0.02 * feats[(c % 2) * m + (t - (c % 8))]
                } else {
                    0.0
                };
            }
            // Target slightly beyond +0.10 dB tilt if col 0 snaps away from exact continuous solve.
            tgts[t] = (1.800_7 * f64::from(f0) - 0.900_6 * f64::from(f1)) as f32;
        }
        let (gram, rhs_v) =
            assemble_ozaki_normal_equations(&runner, &feats, &tgts, m, dim, 1, false).unwrap();
        let rhs: Vec<f64> = rhs_v.into_iter().map(|v| v[0]).collect();

        // Feed intentionally slightly skewed base_weights whose snap creates tilt overshoot so the
        // refinement loop exercises EMR, LTAS KKT, and post-snap damped KKT re-projection cleanly.
        let mut skewed = vec![0.0_f64; dim];
        skewed[0] = 1.92;
        skewed[1] = -0.82;
        let ctx = ChannelSpectralCtx {
            gram: &gram,
            rhs: &rhs,
            feats_post: &feats,
            tgts_post: &tgts,
            sr_hz: sr,
            ro_pole_48k: 0.0,
            active_fir_taps: 128,
            emit_prefix_overrides: true,
            used_ridge: 1.0e-5,
        };
        let refined = refine_channel_tier_emr_and_ltas_kkt(&skewed, dim, 3, &ctx);
        assert_eq!(refined.len(), dim);
        assert!(refined.iter().all(|v| v.is_finite()));
        assert!(within_6db_and_bounds(&refined, 0.0, sr, 8.0));

        // Direct smoke test of `try_damped_kkt_resnap`: accepts finite +6 dB-safe tilt-improving
        // re-snap when within MSE budget, and rejects impossible MSE or non-finite inputs.
        let col_specs = build_tier_col_wire_specs(3, 128, true);
        let lambda_diag = vec![1.0e-5; dim];
        let act_rows = vec![1.0_f64; 2 * dim];
        let act_re = act_rows.clone();
        let act_im = vec![0.0_f64; 2 * dim];
        let act_amps = vec![1.0_f64; 2];
        let mut p_test = KktResnapCtx {
            ctx: &ctx,
            tier_idx: 3,
            dim,
            n_act: 2,
            gram_sub: &gram,
            rhs_sub: &rhs,
            act_rows: &act_rows,
            act_amps: &act_amps,
            act_re: &act_re,
            act_im: &act_im,
            lambda_diag: &lambda_diag,
            col_specs: &col_specs,
            best_mse: 1.0,
            tilt_budget: 0.15,
            base_within_6db: true,
        };
        let accepted = try_damped_kkt_resnap(&p_test, 5.0, 5.0)
            .expect("expected valid damped KKT resnap within MSE budget");
        assert!(within_6db_and_bounds(&accepted, 0.0, sr, 8.0));

        p_test.best_mse = 1.0e-18;
        assert!(try_damped_kkt_resnap(&p_test, 5.0, 5.0).is_none());

        let rhs_nan = vec![f64::NAN; dim];
        p_test.best_mse = 1.0;
        p_test.rhs_sub = &rhs_nan;
        assert!(try_damped_kkt_resnap(&p_test, 5.0, 5.0).is_none());
    }

    #[test]
    fn test_solve_bounded_uplo_u_enforces_l1_norm_and_bibo_uniform_scaling() {
        let dim = 32_usize;
        let sr = 48_000.0_f64;
        let mut gram = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut rhs = vec![0.0_f64; dim];
        // Collinear features whose unregularized inverse produces huge +/- coefficients.
        for t in 0..64 {
            let base = ((t as f64) * 0.13).sin();
            let row: Vec<f64> = (0..dim)
                .map(|c| base + 1.0e-3 * (((t * 7 + c * 3) % 11) as f64 - 5.0))
                .collect();
            let tgt = 0.4 * base + 2.0 * (((t % 2) as f64) - 0.5);
            for c in 0..dim {
                rhs[c] += row[c] * tgt;
                for r in 0..=c {
                    gram[uplo_u_index(r, c)] += row[r] * row[c];
                }
            }
        }
        let (sol, _) = solve_bounded_uplo_u(&gram, &rhs, dim, 0.0, sr).unwrap();
        let l1: f64 = sol.iter().map(|v| v.abs()).sum();
        assert!(
            within_6db_and_bounds(&sol, 0.0, sr, MAX_READOUT_ABS_BEFORE_SNAP),
            "collinear solve must satisfy BIBO bounds: l1={l1}"
        );
        assert!(l1 <= MAX_READOUT_L1_NORM + 1.0e-9);

        // Extreme RHS forcing uniform scaling fallback preserves finite l1 <= MAX_READOUT_L1_NORM.
        let tiny_gram: Vec<f64> = (0..dim)
            .flat_map(|c| (0..=c).map(move |r| if r == c { 1.0e-6 } else { 0.0 }))
            .collect();
        let huge_rhs: Vec<f64> = (0..dim)
            .map(|c| if c % 2 == 0 { 10.0 } else { -10.0 })
            .collect();
        let (fallback_sol, _) = solve_bounded_uplo_u(&tiny_gram, &huge_rhs, dim, 0.0, sr).unwrap();
        let fb_l1: f64 = fallback_sol.iter().map(|v| v.abs()).sum();
        assert!(
            fallback_sol
                .iter()
                .all(|v| v.is_finite() && v.abs() <= MAX_READOUT_ABS_BEFORE_SNAP + 1.0e-9)
        );
        assert!(fb_l1 <= MAX_READOUT_L1_NORM + 1.0e-9, "fb_l1={fb_l1}");
    }
}
