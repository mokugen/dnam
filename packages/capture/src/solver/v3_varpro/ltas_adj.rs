#![forbid(unsafe_code)]
//! Differentiable 33-Band Base-10 1/3-Octave Welch `LTAS` Loss & Exact Reverse-Mode FFT Adjoint.
//!
//! Matches [`crate::metrics::companion::accumulate_welch_33band_powers`] sample-for-sample on
//! `y[warmup..]` (`win_cap = 16_384 * rate_scale`, 50%-overlap + trailing Hann segments, exact
//! bin-to-interval mapping) with single-FFT forward caching and single-FFT backward adjoints.

use std::f64::consts::LN_10;
#[cfg(test)]
use std::f64::consts::PI;

mod fft;
mod nyquist;
pub use fft::{Welch33Scratch, Welch33TargetSummary, build_welch33_target_summary};
use fft::{WelchPairScratch, accum_paired_backward_fft, accum_paired_forward_fft};
use nyquist::eval_nyquist_stopband_hinge;

use super::primitives::par_map_indexed;

#[inline]
pub(super) fn c2_bounded_hinge(u: f64, scale_m: f64, width: f64) -> (f64, f64) {
    if u <= 0.0 {
        return (0.0, 0.0);
    }
    let c = 0.25 * width * width;
    let u2 = u * u;
    let d = u2 + c;
    let val = scale_m * u2 * u / d;
    let deriv = scale_m * u2 * (u2 + 3.0 * c) / (d * d);
    (val, deriv)
}

#[inline]
fn pseudo_huber(s: f64, delta: f64) -> (f64, f64) {
    let r = s / delta;
    let root = (1.0 + r * r).sqrt();
    (delta * delta * (root - 1.0), s / root)
}

#[derive(Clone, Copy)]
enum WelchLossMode {
    L2Huber,
    HingeMaxDb(f64),
    LogSumExp { beta: f64, goal_db: f64 },
}

fn accumulate_welch33_band_powers_loss(
    summary: &Welch33TargetSummary,
    band_p: &[f64; 33],
    d_band: &mut [f64; 33],
    mode: WelchLossMode,
    loss_scale: f64,
    need_grad: bool,
) -> (f64, f64) {
    let db_per_ln = 10.0 / LN_10;
    let inv_act = 1.0 / (summary.active_count.max(1) as f64);
    d_band.fill(0.0);

    if let WelchLossMode::LogSumExp {
        beta: beta_raw,
        goal_db,
    } = mode
    {
        let beta = beta_raw.max(1.0);
        let mut db_errs = [0.0_f64; 33];
        let mut max_abs_db = 0.0_f64;
        let mut mean_sq = 0.0_f64;
        for b in 0..33 {
            if !summary.active[b] {
                continue;
            }
            let pb = band_p[b] + summary.floor;
            let pt = summary.tgt_band_p[b] + summary.floor;
            let e = db_per_ln * (pb.ln() - pt.ln());
            db_errs[b] = e;
            max_abs_db = max_abs_db.max(e.abs());
            mean_sq += inv_act * e * e;
        }
        let z0 = (-beta * max_abs_db).exp();
        let mut excess = 0.0_f64;
        let mut w_pos = [0.0_f64; 33];
        let mut w_neg = [0.0_f64; 33];
        for b in 0..33 {
            if !summary.active[b] {
                continue;
            }
            w_pos[b] = (beta * (db_errs[b] - max_abs_db)).exp();
            w_neg[b] = (beta * (-db_errs[b] - max_abs_db)).exp();
            excess += (w_pos[b] - z0) + (w_neg[b] - z0);
        }
        let denom = (z0 + excess).max(1.0e-30);
        let f_lse = (max_abs_db + denom.ln() / beta).max(0.0);
        let (sp, dsp) = if goal_db > 0.0 {
            let u = f_lse - goal_db;
            let r = (u * u + 1.0e-6).sqrt();
            (0.5 * (u + r), 0.5 * (1.0 + u / r))
        } else {
            (f_lse, 1.0)
        };
        let loss = loss_scale * (sp * sp + 0.25 * f_lse * f_lse + 0.25 * mean_sq);
        if need_grad {
            let inv_denom = 1.0 / denom;
            let scale_lse = if f_lse > 0.0 {
                2.0 * sp * dsp + 0.50 * f_lse
            } else {
                0.0
            };
            for b in 0..33 {
                if !summary.active[b] {
                    continue;
                }
                let pb = band_p[b] + summary.floor;
                let df_de =
                    scale_lse * (w_pos[b] - w_neg[b]) * inv_denom + 0.50 * inv_act * db_errs[b];
                d_band[b] = loss_scale * df_de * db_per_ln / pb;
            }
        }
        return (loss, max_abs_db);
    }

    let mut loss = 0.0_f64;
    let mut max_abs_db = 0.0_f64;
    for b in 0..33 {
        if !summary.active[b] {
            continue;
        }
        let pb = band_p[b] + summary.floor;
        let pt = summary.tgt_band_p[b] + summary.floor;
        let db_err = db_per_ln * (pb.ln() - pt.ln());
        max_abs_db = max_abs_db.max(db_err.abs());
        if let WelchLossMode::HingeMaxDb(tmax) = mode {
            let tau = (0.72 * tmax).max(0.015);
            let tw = (0.22 * tmax).max(0.010);
            let eps = 0.003_f64;
            let root_e = (db_err * db_err + eps * eps).sqrt();
            let abs_smooth = root_e - eps;
            let da_de = db_err / root_e;
            let (hub_v, hub_d) = pseudo_huber(db_err, 1.5);
            let u = abs_smooth - tau;
            let (hg_v, hg_d) = c2_bounded_hinge(u, 1.5, tw);
            let (qv, qd) = if u > 0.0 {
                let c = 0.25 * tw * tw;
                let u2 = u * u;
                let d = u2 + c;
                (u2 * u2 / d, 2.0 * u2 * u * (u2 + 2.0 * c) / (d * d))
            } else {
                (0.0, 0.0)
            };
            loss += loss_scale * (0.10 * inv_act * hub_v + 2.0 * hg_v + 3.0 * qv);
            if need_grad {
                d_band[b] = loss_scale
                    * (0.10 * inv_act * hub_d + (2.0 * hg_d + 3.0 * qd) * da_de)
                    * db_per_ln
                    / pb;
            }
        } else {
            let (hub_v, hub_d) = pseudo_huber(db_err, 2.0);
            let s = 2.0 * loss_scale * inv_act;
            loss += s * hub_v;
            if need_grad {
                d_band[b] = s * hub_d * db_per_ln / pb;
            }
        }
    }
    (loss, max_abs_db)
}

fn accumulate_welch33_band_loss(
    summary: &Welch33TargetSummary,
    mode: WelchLossMode,
    loss_scale: f64,
    need_grad: bool,
    scratch: &mut Welch33Scratch,
) -> (f64, f64) {
    accumulate_welch33_band_powers_loss(
        summary,
        &scratch.band_p,
        &mut scratch.d_band,
        mode,
        loss_scale,
        need_grad,
    )
}

fn eval_welch33_core(
    summary: &Welch33TargetSummary,
    y: &[f64],
    grad_dy: Option<&mut [f64]>,
    loss_scale: f64,
    mode: WelchLossMode,
    include_nyq_hinge: bool,
    scratch: &mut Welch33Scratch,
) -> f64 {
    if (loss_scale <= 0.0 && !include_nyq_hinge)
        || (summary.active_count == 0 && !include_nyq_hinge)
        || summary.seg_starts.is_empty()
        || summary.seg_len == 0
        || y.len() != summary.len
        || grad_dy.as_deref().is_some_and(|dy| dy.len() != summary.len)
    {
        scratch.last_max_abs_db = 0.0;
        return 0.0;
    }

    let n_fft = summary.plan.n_fft;
    let half_p1 = n_fft / 2 + 1;
    let n_wins = summary.seg_starts.len();
    let need_grad = grad_dy.is_some();

    if scratch.fft_re.len() != n_fft {
        scratch.fft_re.resize(n_fft, 0.0);
        scratch.fft_im.resize(n_fft, 0.0);
    }
    if need_grad {
        let total = n_wins * half_p1;
        if scratch.win_re.len() < total {
            scratch.win_re.resize(total, 0.0);
            scratch.win_im.resize(total, 0.0);
        }
    }

    scratch.band_p.fill(0.0);
    let mut p_nyq = 0.0_f64;
    let power_scale = summary.inv_wins * summary.inv_win_norm;
    let n_pairs = n_wins.div_ceil(2);
    let use_par = n_pairs >= 16 && summary.seg_len >= 8_192;
    if use_par {
        while scratch.pair_pool.len() < n_pairs {
            scratch
                .pair_pool
                .push(std::sync::Mutex::new(WelchPairScratch::default()));
        }
        let pool_ref = &scratch.pair_pool[..n_pairs];
        let win_re_need = if need_grad { 2 * half_p1 } else { 0 };
        let pair_bands = par_map_indexed(n_pairs, |p_idx| {
            let w0 = 2 * p_idx;
            let w1 = w0 + 1;
            let st0 = summary.seg_starts[w0];
            let seg0 = &y[summary.warmup + st0..summary.warmup + st0 + summary.seg_len];
            let seg1 = (w1 < n_wins).then(|| {
                let st1 = summary.seg_starts[w1];
                &y[summary.warmup + st1..summary.warmup + st1 + summary.seg_len]
            });
            let mut guard = pool_ref[p_idx]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let pair = &mut *guard;
            if pair.fft_re.len() != n_fft {
                pair.fft_re.resize(n_fft, 0.0);
                pair.fft_im.resize(n_fft, 0.0);
            }
            let mut local_band = [0.0_f64; 33];
            let mut local_nyq = 0.0_f64;
            if need_grad {
                if pair.win_re.len() < win_re_need {
                    pair.win_re.resize(win_re_need, 0.0);
                    pair.win_im.resize(win_re_need, 0.0);
                }
                let (wr0, wr1) = pair.win_re.split_at_mut(half_p1);
                let (wi0, wi1) = pair.win_im.split_at_mut(half_p1);
                let out0 = Some((&mut wr0[..half_p1], &mut wi0[..half_p1]));
                let out1 = seg1.map(|_| (&mut wr1[..half_p1], &mut wi1[..half_p1]));
                accum_paired_forward_fft(
                    &summary.plan,
                    summary.seg_len,
                    &summary.win,
                    &summary.bin_band,
                    seg0,
                    seg1,
                    &mut pair.fft_re,
                    &mut pair.fft_im,
                    power_scale,
                    &mut local_band,
                    include_nyq_hinge.then_some(&mut local_nyq),
                    out0,
                    out1,
                );
            } else {
                accum_paired_forward_fft(
                    &summary.plan,
                    summary.seg_len,
                    &summary.win,
                    &summary.bin_band,
                    seg0,
                    seg1,
                    &mut pair.fft_re,
                    &mut pair.fft_im,
                    power_scale,
                    &mut local_band,
                    include_nyq_hinge.then_some(&mut local_nyq),
                    None,
                    None,
                );
            }
            (local_band, local_nyq)
        });
        for (pb, nyq) in pair_bands {
            for b in 0..33 {
                scratch.band_p[b] += pb[b];
            }
            p_nyq += nyq;
        }
    } else {
        let mut w_idx = 0usize;
        while w_idx < n_wins {
            let st0 = summary.seg_starts[w_idx];
            let seg0 = &y[summary.warmup + st0..summary.warmup + st0 + summary.seg_len];
            let seg1 = if w_idx + 1 < n_wins {
                let st1 = summary.seg_starts[w_idx + 1];
                Some(&y[summary.warmup + st1..summary.warmup + st1 + summary.seg_len])
            } else {
                None
            };
            if need_grad {
                let off0 = w_idx * half_p1;
                let (wr_head, wr_tail) = scratch.win_re.split_at_mut(off0 + half_p1);
                let (wi_head, wi_tail) = scratch.win_im.split_at_mut(off0 + half_p1);
                let out0 = Some((&mut wr_head[off0..], &mut wi_head[off0..]));
                let out1 = seg1.map(|_| (&mut wr_tail[..half_p1], &mut wi_tail[..half_p1]));
                accum_paired_forward_fft(
                    &summary.plan,
                    summary.seg_len,
                    &summary.win,
                    &summary.bin_band,
                    seg0,
                    seg1,
                    &mut scratch.fft_re,
                    &mut scratch.fft_im,
                    power_scale,
                    &mut scratch.band_p,
                    include_nyq_hinge.then_some(&mut p_nyq),
                    out0,
                    out1,
                );
            } else {
                accum_paired_forward_fft(
                    &summary.plan,
                    summary.seg_len,
                    &summary.win,
                    &summary.bin_band,
                    seg0,
                    seg1,
                    &mut scratch.fft_re,
                    &mut scratch.fft_im,
                    power_scale,
                    &mut scratch.band_p,
                    include_nyq_hinge.then_some(&mut p_nyq),
                    None,
                    None,
                );
            }
            w_idx += 2;
        }
    }

    let (loss, max_abs_db) = if loss_scale > 0.0 && summary.active_count > 0 {
        accumulate_welch33_band_loss(summary, mode, loss_scale, need_grad, scratch)
    } else {
        scratch.d_band.fill(0.0);
        (0.0, 0.0)
    };
    scratch.last_max_abs_db = max_abs_db;
    let (nyq_loss, nyq_d) = if include_nyq_hinge {
        eval_nyquist_stopband_hinge(summary, p_nyq)
    } else {
        (0.0, 0.0)
    };

    if let Some(dy) = grad_dy {
        let seg_len = summary.seg_len;
        if use_par {
            let d_band = &scratch.d_band;
            let pool_ref = &scratch.pair_pool[..n_pairs];
            par_map_indexed(n_pairs, |p_idx| {
                let w0 = 2 * p_idx;
                let has_seg1 = w0 + 1 < n_wins;
                let dy_need = if has_seg1 { 2 * seg_len } else { seg_len };
                let mut guard = pool_ref[p_idx]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let pair = &mut *guard;
                if pair.dy_pair.len() < dy_need {
                    pair.dy_pair.resize(dy_need, 0.0);
                }
                pair.dy_pair[..dy_need].fill(0.0);
                accum_paired_backward_fft(
                    &summary.plan,
                    seg_len,
                    &summary.win,
                    &summary.bin_band,
                    &summary.active,
                    d_band,
                    nyq_d,
                    power_scale,
                    &pair.win_re,
                    &pair.win_im,
                    0,
                    has_seg1.then_some(half_p1),
                    &mut pair.fft_re,
                    &mut pair.fft_im,
                    &mut pair.dy_pair,
                    0,
                    has_seg1.then_some(seg_len),
                );
            });
            for (p_idx, pair_mtx) in pool_ref.iter().enumerate() {
                let guard = pair_mtx
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let w0 = 2 * p_idx;
                let base0 = summary.warmup + summary.seg_starts[w0];
                for (dst, &v) in dy[base0..base0 + seg_len]
                    .iter_mut()
                    .zip(&guard.dy_pair[..seg_len])
                {
                    *dst += v;
                }
                if w0 + 1 < n_wins {
                    let base1 = summary.warmup + summary.seg_starts[w0 + 1];
                    for (dst, &v) in dy[base1..base1 + seg_len]
                        .iter_mut()
                        .zip(&guard.dy_pair[seg_len..2 * seg_len])
                    {
                        *dst += v;
                    }
                }
            }
        } else {
            let mut w_idx = 0usize;
            while w_idx < n_wins {
                let base0 = summary.warmup + summary.seg_starts[w_idx];
                let off0 = w_idx * half_p1;
                let (off1_opt, base1_opt) = if w_idx + 1 < n_wins {
                    (
                        Some((w_idx + 1) * half_p1),
                        Some(summary.warmup + summary.seg_starts[w_idx + 1]),
                    )
                } else {
                    (None, None)
                };
                accum_paired_backward_fft(
                    &summary.plan,
                    seg_len,
                    &summary.win,
                    &summary.bin_band,
                    &summary.active,
                    &scratch.d_band,
                    nyq_d,
                    power_scale,
                    &scratch.win_re,
                    &scratch.win_im,
                    off0,
                    off1_opt,
                    &mut scratch.fft_re,
                    &mut scratch.fft_im,
                    dy,
                    base0,
                    base1_opt,
                );
                w_idx += 2;
            }
        }
    }

    loss + nyq_loss
}

/// Evaluates the smooth $L_2 + 0.5 L_4$ log-dB Welch-33 LTAS loss and optionally accumulates its
/// exact reverse-mode 2-for-1 paired FFT adjoint into `grad_dy`.
#[allow(clippy::too_many_arguments)]
pub fn eval_welch33_ltas_loss_and_adjoint(
    summary: &Welch33TargetSummary,
    y: &[f64],
    grad_dy: Option<&mut [f64]>,
    stem_w: f64,
    ch_scale: f64,
    welch_ltas_weight: f64,
    scratch: &mut Welch33Scratch,
) -> f64 {
    eval_welch33_core(
        summary,
        y,
        grad_dy,
        stem_w * ch_scale * welch_ltas_weight,
        WelchLossMode::L2Huber,
        false,
        scratch,
    )
}

/// Evaluates the bounded-derivative $C^2$ Huber + hinge Welch-33 LTAS loss keyed to `target_max_db`.
pub fn eval_welch33_minimax_loss_and_adjoint(
    summary: &Welch33TargetSummary,
    y: &[f64],
    grad_dy: Option<&mut [f64]>,
    weight: f64,
    target_max_db: f64,
    scratch: &mut Welch33Scratch,
) -> f64 {
    eval_welch33_core(
        summary,
        y,
        grad_dy,
        weight,
        WelchLossMode::HingeMaxDb(target_max_db),
        false,
        scratch,
    )
}

/// Evaluates the smooth goal-thresholded `log-sum-exp` minimax Welch-33 loss
/// `weight * (softplus(lse - goal_db)^2 + 0.10 * lse^2 + 0.05 * mean_sq)`.
#[cfg(test)]
pub(super) fn eval_welch33_lse_minimax_loss_and_adjoint(
    summary: &Welch33TargetSummary,
    y: &[f64],
    grad_dy: Option<&mut [f64]>,
    weight: f64,
    beta: f64,
    goal_db: f64,
    scratch: &mut Welch33Scratch,
) -> f64 {
    eval_welch33_core(
        summary,
        y,
        grad_dy,
        weight,
        WelchLossMode::LogSumExp { beta, goal_db },
        false,
        scratch,
    )
}

/// Evaluates the fused `log-sum-exp` minimax Welch-33 loss and out-of-band `[20 kHz, Nyquist]`
/// stopband hinge in a single paired FFT forward/backward pass.
pub(in crate::solver) fn eval_welch33_lse_minimax_with_nyq_and_adjoint(
    summary: &Welch33TargetSummary,
    y: &[f64],
    grad_dy: Option<&mut [f64]>,
    weight: f64,
    beta: f64,
    goal_db: f64,
    scratch: &mut Welch33Scratch,
) -> f64 {
    eval_welch33_core(
        summary,
        y,
        grad_dy,
        weight,
        WelchLossMode::LogSumExp { beta, goal_db },
        true,
        scratch,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::companion::{canonical_ltas_33_centers_hz, evaluate_ltas_33band_report};

    #[test]
    fn test_v3_welch33_ltas_adjoint_matches_central_finite_diff() {
        let n = 4_096;
        let warmup = 256;
        let sr = 48_000.0;
        let centers = canonical_ltas_33_centers_hz();
        let build_tone = |gain: f64, ph: f64| -> Vec<f64> {
            let scale = gain / (centers.len() as f64).sqrt();
            (0..n)
                .map(|i| {
                    let t = (i as f64) / sr;
                    let mut acc = 0.0_f64;
                    for (k, &fc) in centers.iter().enumerate() {
                        acc += (2.0 * PI * fc * t + (k as f64) * 0.73 + ph).sin();
                    }
                    scale * acc
                })
                .collect()
        };
        let tgt = build_tone(0.45, 0.0);
        let y = build_tone(0.38, 0.25);

        let summary = build_welch33_target_summary(&tgt, warmup, sr);
        assert!(summary.active_count > 10);
        let rep_eq = evaluate_ltas_33band_report(&tgt[warmup..], &tgt[warmup..], None, sr, true);
        assert!(rep_eq.max_adj_deviation_db < 1.0e-12);

        let mut scratch = Welch33Scratch::default();
        let loss_eq =
            eval_welch33_ltas_loss_and_adjoint(&summary, &tgt, None, 1.0, 1.0, 0.08, &mut scratch);
        assert!(loss_eq.abs() < 1.0e-20);

        let mut dy = vec![0.0_f64; n];
        let loss = eval_welch33_ltas_loss_and_adjoint(
            &summary,
            &y,
            Some(&mut dy),
            0.85,
            0.5,
            0.08,
            &mut scratch,
        );
        assert!(loss > 0.0 && loss.is_finite());
        assert!(dy[..warmup].iter().all(|&v| v == 0.0));

        let dy_peak = dy.iter().map(|v| v.abs()).fold(1.0e-8, f64::max);
        let h = 1.0e-6_f64;
        let mut max_rel = 0.0_f64;
        for idx in (0..n).step_by(193) {
            let mut yp = y.clone();
            let mut ym = y.clone();
            yp[idx] += h;
            ym[idx] -= h;
            let lp = eval_welch33_ltas_loss_and_adjoint(
                &summary,
                &yp,
                None,
                0.85,
                0.5,
                0.08,
                &mut scratch,
            );
            let lm = eval_welch33_ltas_loss_and_adjoint(
                &summary,
                &ym,
                None,
                0.85,
                0.5,
                0.08,
                &mut scratch,
            );
            let g_fd = (lp - lm) / (2.0 * h);
            let denom = g_fd.abs().max(dy[idx].abs()).max(dy_peak * 1.0e-2);
            let rel = (dy[idx] - g_fd).abs() / denom;
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1.0e-5,
                "Welch33 sample {idx} ana={} fd={} rel={rel}",
                dy[idx],
                g_fd
            );
        }
        assert!(max_rel < 1.0e-5);

        let mut yp = y.clone();
        let mut ym = y;
        let mut d_ana = 0.0_f64;
        for i in warmup..n {
            let d = dy[i] / dy_peak + 0.05 * ((i as f64) * 0.37).sin();
            d_ana += dy[i] * d;
            yp[i] += h * d;
            ym[i] -= h * d;
        }
        let lp =
            eval_welch33_ltas_loss_and_adjoint(&summary, &yp, None, 0.85, 0.5, 0.08, &mut scratch);
        let lm =
            eval_welch33_ltas_loss_and_adjoint(&summary, &ym, None, 0.85, 0.5, 0.08, &mut scratch);
        let d_fd = (lp - lm) / (2.0 * h);
        let d_rel = (d_ana - d_fd).abs() / d_ana.abs().max(d_fd.abs()).max(1.0e-10);
        assert!(
            d_rel < 1.0e-7,
            "Welch33 directional ana={d_ana} fd={d_fd} rel={d_rel}"
        );

        // Paired + trailing-odd overlapping multi-window verification (3+ windows at n=40_960) & minimax loss.
        let n_multi = 40_960;
        let build_multi = |gain: f64, ph: f64| -> Vec<f64> {
            let scale = gain / (centers.len() as f64).sqrt();
            (0..n_multi)
                .map(|i| {
                    let t = (i as f64) / sr;
                    let mut acc = 0.0_f64;
                    for (k, &fc) in centers.iter().enumerate() {
                        acc += (2.0 * PI * fc * t + (k as f64) * 0.59 + ph).sin();
                    }
                    scale * acc
                })
                .collect()
        };
        let tgt_m = build_multi(0.45, 0.0);
        let y_m = build_multi(0.36, 0.31);
        let sum_m = build_welch33_target_summary(&tgt_m, warmup, sr);
        assert!(sum_m.seg_starts.len() >= 3);
        let mut dy_m = vec![0.0_f64; n_multi];
        let mut dy_mm = vec![0.0_f64; n_multi];
        let l_m = eval_welch33_ltas_loss_and_adjoint(
            &sum_m,
            &y_m,
            Some(&mut dy_m),
            0.75,
            0.5,
            0.10,
            &mut scratch,
        );
        let l_mm = eval_welch33_minimax_loss_and_adjoint(
            &sum_m,
            &y_m,
            Some(&mut dy_mm),
            0.12,
            0.15,
            &mut scratch,
        );
        assert!(l_m > 0.0 && l_mm > 0.0);
        for (tag, loss_fn, dy_vec) in [("multi", true, &dy_m), ("minimax", false, &dy_mm)] {
            let pk = dy_vec.iter().map(|v| v.abs()).fold(1.0e-8, f64::max);
            let mut yp_m = y_m.clone();
            let mut ym_m = y_m.clone();
            let mut da = 0.0_f64;
            for i in warmup..n_multi {
                let d = dy_vec[i] / pk + 0.04 * ((i as f64) * 0.23).cos();
                da += dy_vec[i] * d;
                yp_m[i] += h * d;
                ym_m[i] -= h * d;
            }
            let lp_m = if loss_fn {
                eval_welch33_ltas_loss_and_adjoint(
                    &sum_m,
                    &yp_m,
                    None,
                    0.75,
                    0.5,
                    0.10,
                    &mut scratch,
                )
            } else {
                eval_welch33_minimax_loss_and_adjoint(&sum_m, &yp_m, None, 0.12, 0.15, &mut scratch)
            };
            let lm_m = if loss_fn {
                eval_welch33_ltas_loss_and_adjoint(
                    &sum_m,
                    &ym_m,
                    None,
                    0.75,
                    0.5,
                    0.10,
                    &mut scratch,
                )
            } else {
                eval_welch33_minimax_loss_and_adjoint(&sum_m, &ym_m, None, 0.12, 0.15, &mut scratch)
            };
            let dfd = (lp_m - lm_m) / (2.0 * h);
            let rel = (da - dfd).abs() / da.abs().max(dfd.abs()).max(1.0e-10);
            assert!(
                rel < 1.0e-6,
                "{tag} paired Welch33 rel={rel} ana={da} fd={dfd}"
            );
        }
    }

    #[test]
    fn test_welch33_parallel_multiwindow_minimax_fd_and_bit_exact() {
        let n = 45_056;
        let warmup = 2_048;
        let sr = 48_000.0;
        let centers = canonical_ltas_33_centers_hz();
        let make_sig = |gain: f64, ph: f64| -> Vec<f64> {
            let scale = gain / (centers.len() as f64).sqrt();
            (0..n)
                .map(|i| {
                    let t = (i as f64) / sr;
                    let s: f64 = centers
                        .iter()
                        .enumerate()
                        .map(|(k, &fc)| (2.0 * PI * fc * t + (k as f64) * 0.43 + ph).sin())
                        .sum();
                    scale * s
                })
                .collect()
        };
        let tgt = make_sig(0.42, 0.0);
        let y = make_sig(0.35, 0.27);
        let sum = build_welch33_target_summary(&tgt, warmup, sr);
        assert_eq!(sum.seg_len, 16_384);
        assert_eq!(sum.seg_starts.len(), 5);

        let mut sc_a = Welch33Scratch::default();
        let mut sc_b = sc_a.clone();
        let (mut dy_a, mut dy_b) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let la =
            eval_welch33_minimax_loss_and_adjoint(&sum, &y, Some(&mut dy_a), 0.15, 0.14, &mut sc_a);
        let lb =
            eval_welch33_minimax_loss_and_adjoint(&sum, &y, Some(&mut dy_b), 0.15, 0.14, &mut sc_b);
        assert_eq!(la.to_bits(), lb.to_bits());
        assert_eq!(dy_a, dy_b);

        let h = 1.0e-6_f64;
        let pk = dy_a.iter().map(|v| v.abs()).fold(1.0e-8, f64::max);
        let (mut yp, mut ym, mut d_ana) = (y.clone(), y.clone(), 0.0_f64);
        for i in warmup..n {
            let d = dy_a[i] / pk + 0.03 * ((i as f64) * 0.19).sin();
            d_ana += dy_a[i] * d;
            yp[i] += h * d;
            ym[i] -= h * d;
        }
        let lp = eval_welch33_minimax_loss_and_adjoint(&sum, &yp, None, 0.15, 0.14, &mut sc_a);
        let lm = eval_welch33_minimax_loss_and_adjoint(&sum, &ym, None, 0.15, 0.14, &mut sc_a);
        let d_fd = (lp - lm) / (2.0 * h);
        let rel = (d_ana - d_fd).abs() / d_ana.abs().max(d_fd.abs()).max(1.0e-10);
        assert!(rel < 1.0e-6, "parallel Welch33 minimax rel={rel}");

        let mut dy_lse = vec![0.0_f64; n];
        let l_lse = eval_welch33_lse_minimax_loss_and_adjoint(
            &sum,
            &y,
            Some(&mut dy_lse),
            10.0,
            14.0,
            0.02,
            &mut sc_a,
        );
        assert!(l_lse > 0.0 && l_lse.is_finite());
        let pk_l = dy_lse.iter().map(|v| v.abs()).fold(1.0e-8, f64::max);
        let (mut yp_l, mut ym_l, mut da_l) = (y.clone(), y, 0.0_f64);
        for i in warmup..n {
            let d = dy_lse[i] / pk_l + 0.03 * ((i as f64) * 0.19).sin();
            da_l += dy_lse[i] * d;
            yp_l[i] += h * d;
            ym_l[i] -= h * d;
        }
        let lp_l = eval_welch33_lse_minimax_loss_and_adjoint(
            &sum, &yp_l, None, 10.0, 14.0, 0.02, &mut sc_a,
        );
        let lm_l = eval_welch33_lse_minimax_loss_and_adjoint(
            &sum, &ym_l, None, 10.0, 14.0, 0.02, &mut sc_a,
        );
        let dfd_l = (lp_l - lm_l) / (2.0 * h);
        let rel_l = (da_l - dfd_l).abs() / da_l.abs().max(dfd_l.abs()).max(1.0e-10);
        assert!(rel_l < 1.0e-6, "parallel Welch33 LSE minimax rel={rel_l}");
    }
}
