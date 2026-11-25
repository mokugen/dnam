//! Exact 5-scale (`256, 512, 1024, 2048, 4096`), 32-probe unity-peak-gain 2nd-order bandpass
//! `MR-IIR` loss & reverse-time IIR state adjoint (`ref[qual.companion.mr-iir]`).

use core::f64::consts::PI;

use super::certification::{DrySilentGapMask, POST_T60_GAP_AMP_RATIO};
use super::noise_floor::{smooth_excess_sq, snr_weight};

const SCALES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const NUM_PROBES: usize = 32;
const PROBE_GROUPS: usize = NUM_PROBES / 4;
const RMS_SMOOTH_EPS: f64 = 1.0e-30;
const LOG_SMOOTH_EPS: f64 = 1.0e-6;
const _: () = assert!(
    SCALES[0] == 256
        && SCALES[4] == 4096
        && NUM_PROBES == 32
        && PROBE_GROUPS * 4 == NUM_PROBES
        && RMS_SMOOTH_EPS > 0.0
        && LOG_SMOOTH_EPS > 0.0
);

#[derive(Clone, Debug)]
struct ScaleTarget {
    norm: f64,
    rm2: f64,
    inv_win_len: f64,
    trc: [[f64; 4]; PROBE_GROUPS],
    checkpoints: Vec<usize>,
    win_cp_idx: Vec<(usize, usize)>,
    win_rms_t: [Vec<[f64; 4]>; PROBE_GROUPS],
    win_sigma_n: [Vec<[f64; 4]>; PROBE_GROUPS],
    sc_den_sqrt: f64,
    cell_count: f64,
}

/// Precomputed 5-scale 32-probe `MR-IIR` target window cache for one audio channel.
#[derive(Clone, Debug)]
pub struct MrIirTargetCache {
    warmup: usize,
    t_sel: usize,
    has_noise: bool,
    scales: Vec<ScaleTarget>,
}

fn build_win_bounds(m_scale: usize, t_sel: usize) -> (usize, Vec<(usize, usize)>) {
    if t_sel < m_scale {
        return (t_sel, vec![(0, t_sel)]);
    }
    let hop = m_scale / 4;
    let mut bounds = Vec::with_capacity((t_sel - m_scale) / hop + 2);
    let mut start = 0_usize;
    while start + m_scale <= t_sel {
        bounds.push((start, start + m_scale));
        start += hop;
    }
    let last_start = t_sel - m_scale;
    if last_start + hop != start {
        bounds.push((last_start, t_sel));
    }
    (m_scale, bounds)
}

fn build_resonator_coeffs(
    m_scale: usize,
    sr_hz: f64,
    f_max: f64,
) -> (f64, f64, [[f64; 4]; PROBE_GROUPS]) {
    let r_m = (-PI / (m_scale as f64)).exp();
    let rm2 = r_m * r_m;
    let norm = 0.5 * (1.0 - rm2);
    let mut trc = [[0.0_f64; 4]; PROBE_GROUPS];
    for (g, grp) in trc.iter_mut().enumerate() {
        for (l, dst) in grp.iter_mut().enumerate() {
            let p = g * 4 + l;
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            *dst = 2.0 * r_m * (2.0 * PI * f_k / sr_hz).cos();
        }
    }
    (norm, rm2, trc)
}

fn build_checkpoints_and_indices(
    t_sel: usize,
    win_bounds: &[(usize, usize)],
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut checkpoints = Vec::with_capacity(win_bounds.len() * 2 + 2);
    checkpoints.push(0);
    checkpoints.push(t_sel);
    for &(ws, we) in win_bounds {
        checkpoints.push(ws);
        checkpoints.push(we);
    }
    checkpoints.sort_unstable();
    checkpoints.dedup();
    let win_cp_idx = win_bounds
        .iter()
        .filter_map(|&(ws, we)| {
            let cp_ws = checkpoints.binary_search(&ws).ok()?;
            let cp_we = checkpoints.binary_search(&we).ok()?;
            Some((cp_ws, cp_we))
        })
        .collect();
    (checkpoints, win_cp_idx)
}

fn step_group_checkpoints_sq(
    sig: &[f64],
    warmup: usize,
    checkpoints: &[usize],
    norm: f64,
    rm2: f64,
    trc: [f64; 4],
    cp_ps: &mut [[f64; 4]],
) {
    let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
    let (mut y1, mut y2) = ([0.0_f64; 4], [0.0_f64; 4]);
    for &x in &sig[..warmup] {
        let nd = norm * (x - x2);
        for l in 0..4 {
            let b = nd + trc[l] * y1[l] - rm2 * y2[l];
            y2[l] = y1[l];
            y1[l] = b;
        }
        x2 = x1;
        x1 = x;
    }
    let mut acc = [0.0_f64; 4];
    cp_ps[0] = acc;
    let mut cur = warmup;
    for (c_idx, &cp_rel) in checkpoints[1..].iter().enumerate() {
        let end = warmup + cp_rel;
        for &x in &sig[cur..end] {
            let nd = norm * (x - x2);
            for l in 0..4 {
                let b = nd + trc[l] * y1[l] - rm2 * y2[l];
                y2[l] = y1[l];
                y1[l] = b;
                acc[l] += b * b;
            }
            x2 = x1;
            x1 = x;
        }
        cp_ps[c_idx + 1] = acc;
        cur = end;
    }
}

fn compute_group_win_rms(
    sig: &[f64],
    warmup: usize,
    checkpoints: &[usize],
    win_cp_idx: &[(usize, usize)],
    win_scale: (f64, f64, f64),
    trc: [[f64; 4]; PROBE_GROUPS],
    cp_ps: &mut Vec<[f64; 4]>,
) -> ([Vec<[f64; 4]>; PROBE_GROUPS], f64) {
    let (norm, rm2, inv_win_len) = win_scale;
    cp_ps.resize(checkpoints.len(), [0.0; 4]);
    let mut sum_sq = 0.0_f64;
    let rms_groups = core::array::from_fn(|g| {
        step_group_checkpoints_sq(sig, warmup, checkpoints, norm, rm2, trc[g], cp_ps);
        let mut rows = Vec::with_capacity(win_cp_idx.len());
        for &(cp_ws, cp_we) in win_cp_idx {
            let mut r4 = [0.0_f64; 4];
            for l in 0..4 {
                let e = (cp_ps[cp_we][l] - cp_ps[cp_ws][l]).max(0.0);
                let rms = (e * inv_win_len).sqrt();
                r4[l] = rms;
                sum_sq += rms * rms;
            }
            rows.push(r4);
        }
        rows
    });
    (rms_groups, sum_sq)
}

impl MrIirTargetCache {
    /// Precomputes target and optional scale-matched noise resonator window RMS across all 5 scales
    /// and 32 log-spaced probes (`ref[qual.companion.mr-iir]`).
    #[must_use]
    pub fn new(
        target: &[f64],
        noise: Option<&[f64]>,
        warmup: usize,
        sample_rate_hz: f64,
    ) -> Option<Self> {
        let f_max = (0.45 * sample_rate_hz).min(20_000.0);
        if target.len() <= warmup + 16
            || f_max <= 20.0
            || sample_rate_hz < 8_000.0
            || !target.iter().all(|v| v.is_finite())
        {
            return None;
        }
        if let Some(ns) = noise
            && (ns.len() < target.len() || !ns[..target.len()].iter().all(|v| v.is_finite()))
        {
            return None;
        }
        let t_sel = target.len() - warmup;
        let mut cp_ps = Vec::new();
        let mut scales = Vec::with_capacity(SCALES.len());
        for &m_scale in &SCALES {
            let (norm, rm2, trc) = build_resonator_coeffs(m_scale, sample_rate_hz, f_max);
            let (win_len, win_bounds) = build_win_bounds(m_scale, t_sel);
            let (checkpoints, win_cp_idx) = build_checkpoints_and_indices(t_sel, &win_bounds);
            let inv_win_len = 1.0 / (win_len as f64);
            let (win_rms_t, sc_den_sq) = compute_group_win_rms(
                target,
                warmup,
                &checkpoints,
                &win_cp_idx,
                (norm, rm2, inv_win_len),
                trc,
                &mut cp_ps,
            );
            let win_sigma_n = match noise {
                Some(ns) => {
                    compute_group_win_rms(
                        &ns[..target.len()],
                        warmup,
                        &checkpoints,
                        &win_cp_idx,
                        (norm, rm2, inv_win_len),
                        trc,
                        &mut cp_ps,
                    )
                    .0
                }
                None => core::array::from_fn(|_| vec![[0.0_f64; 4]; win_cp_idx.len()]),
            };
            let cell_count = (PROBE_GROUPS * win_cp_idx.len() * 4) as f64;
            scales.push(ScaleTarget {
                norm,
                rm2,
                inv_win_len,
                trc,
                checkpoints,
                win_cp_idx,
                win_rms_t,
                win_sigma_n,
                sc_den_sqrt: sc_den_sq.sqrt(),
                cell_count,
            });
        }
        Some(Self {
            warmup,
            t_sel,
            has_noise: noise.is_some(),
            scales,
        })
    }
}

fn backprop_group(
    y_hat: &[f64],
    warmup: usize,
    scale: &ScaleTarget,
    trc: [f64; 4],
    d_win: &[[f64; 4]],
    dy: &mut [f64],
) {
    let n = y_hat.len();
    let t_sel = n - warmup;
    let n_cp = scale.checkpoints.len();
    let mut cp_w_diff = vec![[0.0_f64; 4]; n_cp];
    for (&(cp_ws, cp_we), &dm) in scale.win_cp_idx.iter().zip(d_win) {
        for l in 0..4 {
            cp_w_diff[cp_ws][l] += dm[l];
            cp_w_diff[cp_we][l] -= dm[l];
        }
    }

    let mut b_bar = vec![[0.0_f64; 4]; t_sel];
    let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
    let (mut y1, mut y2) = ([0.0_f64; 4], [0.0_f64; 4]);
    for &x in &y_hat[..warmup] {
        let nd = scale.norm * (x - x2);
        for l in 0..4 {
            let b = nd + trc[l] * y1[l] - scale.rm2 * y2[l];
            y2[l] = y1[l];
            y1[l] = b;
        }
        x2 = x1;
        x1 = x;
    }

    let mut w_cur = [0.0_f64; 4];
    for c in 0..n_cp.saturating_sub(1) {
        for l in 0..4 {
            w_cur[l] += cp_w_diff[c][l];
        }
        let rel_start = scale.checkpoints[c];
        let rel_end = scale.checkpoints[c + 1];
        for (k, &x) in y_hat[warmup + rel_start..warmup + rel_end]
            .iter()
            .enumerate()
        {
            let nd = scale.norm * (x - x2);
            let bar = &mut b_bar[rel_start + k];
            for l in 0..4 {
                let b = nd + trc[l] * y1[l] - scale.rm2 * y2[l];
                y2[l] = y1[l];
                y1[l] = b;
                bar[l] = 2.0 * w_cur[l] * b;
            }
            x2 = x1;
            x1 = x;
        }
    }

    let (mut lam1, mut lam2) = ([0.0_f64; 4], [0.0_f64; 4]);
    for rel in (0..t_sel).rev() {
        let bb = b_bar[rel];
        let mut s_diff = 0.0_f64;
        for l in 0..4 {
            let lam0 = bb[l] + trc[l] * lam1[l] - scale.rm2 * lam2[l];
            s_diff += lam0 - lam2[l];
            lam2[l] = lam1[l];
            lam1[l] = lam0;
        }
        dy[warmup + rel] += scale.norm * s_diff;
    }
    for dy_i in dy[..warmup].iter_mut().rev() {
        let mut s_diff = 0.0_f64;
        for l in 0..4 {
            let lam0 = trc[l] * lam1[l] - scale.rm2 * lam2[l];
            s_diff += lam0 - lam2[l];
            lam2[l] = lam1[l];
            lam1[l] = lam0;
        }
        *dy_i += scale.norm * s_diff;
    }
}

/// Evaluates normative exact `MR-IIR` score `0.2 * sum_M L_M` (`|ln A - ln A_hat|`, unsmoothed,
/// using scale-matched resonator-filtered noise RMS `sigma_{M,k,h}` and exact zero-floor rules),
/// optionally gating noiseless post-`T60` dry-silent gaps via `gap_mask`.
#[must_use]
pub fn compute_mr_iir_channel_with_gaps(
    cache: &MrIirTargetCache,
    pred: &[f64],
    noise: Option<&[f64]>,
    gap_mask: Option<&DrySilentGapMask>,
) -> f64 {
    let n = cache.warmup + cache.t_sel;
    if pred.len() < n || !pred[..n].iter().all(|v| v.is_finite()) {
        return f64::INFINITY;
    }
    if let Some(ns) = noise
        && (ns.len() < n || !ns[..n].iter().all(|v| v.is_finite()))
    {
        return f64::INFINITY;
    }
    let eff_gaps = if noise.is_some() {
        None
    } else {
        gap_mask.filter(|g| !g.is_empty())
    };
    let y_slice = &pred[..n];
    let inv_scales = 1.0 / (cache.scales.len() as f64);
    let mut total = 0.0_f64;
    for scale in &cache.scales {
        let mut cp_ps = vec![[0.0_f64; 4]; scale.checkpoints.len()];
        let noise_rms_override = noise.map(|ns| {
            compute_group_win_rms(
                &ns[..n],
                cache.warmup,
                &scale.checkpoints,
                &scale.win_cp_idx,
                (scale.norm, scale.rm2, scale.inv_win_len),
                scale.trc,
                &mut cp_ps,
            )
            .0
        });
        let sigma_src = noise_rms_override.as_ref().unwrap_or(&scale.win_sigma_n);

        let mut num_sq = 0.0_f64;
        let mut den_sq = 0.0_f64;
        let mut log_sum = 0.0_f64;
        let mut eval_cells = 0_usize;
        for (g, &trc) in scale.trc.iter().enumerate() {
            step_group_checkpoints_sq(
                y_slice,
                cache.warmup,
                &scale.checkpoints,
                scale.norm,
                scale.rm2,
                trc,
                &mut cp_ps,
            );
            for (w, &(cp_ws, cp_we)) in scale.win_cp_idx.iter().enumerate() {
                let abs_s = cache.warmup + scale.checkpoints[cp_ws];
                let abs_e = cache.warmup + scale.checkpoints[cp_we];
                let gap_span = eff_gaps.and_then(|gm| gm.overlap_span(abs_s, abs_e));
                for l in 0..4 {
                    let sigma = sigma_src[g][w][l].max(0.0);
                    let a_t = scale.win_rms_t[g][w][l].max(sigma);
                    let e = (cp_ps[cp_we][l] - cp_ps[cp_ws][l]).max(0.0);
                    let a_p = (e * scale.inv_win_len).sqrt().max(sigma);
                    if let Some(gm) = eff_gaps
                        && let Some((gs, ge)) = gap_span
                        && ((abs_s >= gs && abs_e <= ge)
                            || a_t <= POST_T60_GAP_AMP_RATIO * gm.active_target_rms)
                    {
                        if (a_p - a_t).abs() > 2.0 * POST_T60_GAP_AMP_RATIO * gm.active_target_rms {
                            return f64::INFINITY;
                        }
                        continue;
                    }
                    let diff = a_t - a_p;
                    num_sq += diff * diff;
                    den_sq += a_t * a_t;
                    if a_t == 0.0 && a_p == 0.0 {
                        // Zero log distance when both floor-clamped magnitudes vanish.
                    } else if a_t == 0.0 || a_p == 0.0 {
                        return f64::INFINITY;
                    } else {
                        log_sum += (a_t.ln() - a_p.ln()).abs();
                    }
                    eval_cells += 1;
                }
            }
        }
        if eval_cells == 0 {
            return f64::INFINITY;
        }
        let sc = if den_sq > 0.0 {
            num_sq.sqrt() / den_sq.sqrt()
        } else if num_sq > 0.0 {
            return f64::INFINITY;
        } else {
            0.0
        };
        total += inv_scales * (sc + log_sum / (eval_cells as f64));
    }
    total
}

/// Evaluates normative exact `MR-IIR` score `0.2 * sum_M L_M` (`|ln A - ln A_hat|`, unsmoothed,
/// using scale-matched resonator-filtered noise RMS `sigma_{M,k,h}` and exact zero-floor rules).
#[must_use]
pub fn compute_mr_iir_channel(
    cache: &MrIirTargetCache,
    pred: &[f64],
    noise: Option<&[f64]>,
) -> f64 {
    compute_mr_iir_channel_with_gaps(cache, pred, noise, None)
}

/// Evaluates smooth differentiable `MR-IIR` loss and accumulates exact reverse-time 2nd-order
/// IIR transpose gradients into `dy_opt`.
pub fn eval_mr_iir_loss_and_adjoint(
    cache: &MrIirTargetCache,
    pred: &[f64],
    mut dy_opt: Option<&mut [f64]>,
) -> f64 {
    let n = cache.warmup + cache.t_sel;
    if pred.len() < n {
        return 0.0;
    }
    let y_slice = &pred[..n];
    let inv_scales = 1.0 / (cache.scales.len() as f64);
    let need_grad = dy_opt.is_some();
    let mut mr_val = 0.0_f64;

    for scale in &cache.scales {
        let m_win = scale.win_cp_idx.len();
        let total_rows = PROBE_GROUPS * m_win;
        let mut cp_ps = vec![[0.0_f64; 4]; scale.checkpoints.len()];
        let mut d_sc = if need_grad {
            vec![[0.0_f64; 4]; total_rows]
        } else {
            Vec::new()
        };
        let mut d_mag = if need_grad {
            vec![[0.0_f64; 4]; total_rows]
        } else {
            Vec::new()
        };

        let mut sc_num_sq = 0.0_f64;
        let mut mag_sum = 0.0_f64;
        for g in 0..PROBE_GROUPS {
            step_group_checkpoints_sq(
                y_slice,
                cache.warmup,
                &scale.checkpoints,
                scale.norm,
                scale.rm2,
                scale.trc[g],
                &mut cp_ps,
            );
            for (w, &(cp_ws, cp_we)) in scale.win_cp_idx.iter().enumerate() {
                let row_idx = g * m_win + w;
                for l in 0..4 {
                    let a_t = scale.win_rms_t[g][w][l];
                    let em = (cp_ps[cp_we][l] - cp_ps[cp_ws][l]).max(0.0);
                    let rms_m = ((em + RMS_SMOOTH_EPS) * scale.inv_win_len).sqrt();
                    let diff = rms_m - a_t;
                    let (w_snr, (e_eff, d_e)) = if cache.has_noise {
                        let sn = scale.win_sigma_n[g][w][l];
                        let en = sn * sn;
                        (snr_weight(a_t * a_t, en), smooth_excess_sq(diff * diff, en))
                    } else {
                        (1.0, (diff * diff, 1.0))
                    };
                    sc_num_sq += w_snr * e_eff;
                    let (dm_v, dm_g) = if a_t > 1.0e-18 {
                        let u = (rms_m / a_t).ln();
                        let r_u = (u * u + LOG_SMOOTH_EPS).sqrt();
                        (
                            w_snr * (r_u - LOG_SMOOTH_EPS.sqrt()),
                            w_snr * (u / r_u) * scale.inv_win_len
                                / (2.0 * rms_m * rms_m * scale.cell_count),
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    mag_sum += dm_v;
                    if need_grad {
                        d_sc[row_idx][l] = (w_snr * d_e * diff) * scale.inv_win_len / (2.0 * rms_m);
                        d_mag[row_idx][l] = dm_g;
                    }
                }
            }
        }

        let sc_root = (sc_num_sq + RMS_SMOOTH_EPS).sqrt();
        let l_sc = if scale.sc_den_sqrt > RMS_SMOOTH_EPS {
            sc_root / scale.sc_den_sqrt
        } else {
            0.0
        };
        let l_mag = mag_sum / scale.cell_count;
        mr_val += inv_scales * (l_sc + l_mag);

        if let Some(dy) = dy_opt.as_deref_mut() {
            let inv_sc_den = if scale.sc_den_sqrt > RMS_SMOOTH_EPS {
                1.0 / (scale.sc_den_sqrt * sc_root)
            } else {
                0.0
            };
            let mut d_win = vec![[0.0_f64; 4]; total_rows];
            for idx in 0..total_rows {
                for l in 0..4 {
                    d_win[idx][l] = inv_scales * (d_sc[idx][l] * inv_sc_den + d_mag[idx][l]);
                }
            }
            for g in 0..PROBE_GROUPS {
                backprop_group(
                    y_slice,
                    cache.warmup,
                    scale,
                    scale.trc[g],
                    &d_win[g * m_win..(g + 1) * m_win],
                    &mut dy[..n],
                );
            }
        }
    }

    mr_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mr_iir_metric_and_reverse_iir_fd_adjoint() {
        let n = 640_usize;
        let target: Vec<f64> = (0..n)
            .map(|t| {
                0.4 * ((t as f64) * 0.031).sin()
                    + 0.35 * ((t as f64) * 0.17).cos()
                    + 0.25 * ((t as f64) * 0.89).sin()
            })
            .collect();
        let pred: Vec<f64> = target
            .iter()
            .enumerate()
            .map(|(t, &v)| 0.88 * v + 0.04 * ((t as f64) * 0.27).sin())
            .collect();
        let cache = MrIirTargetCache::new(&target, None, 64, 48_000.0).unwrap();
        let m_score = compute_mr_iir_channel(&cache, &pred, None);
        assert!(m_score.is_finite() && m_score > 0.0);
        assert!(compute_mr_iir_channel(&cache, &target, None) <= 1.0e-12);

        // Zero-floor +inf rules & resonator-matched noise clamping.
        let zeros = vec![0.0_f64; n];
        let zero_cache = MrIirTargetCache::new(&zeros, None, 64, 48_000.0).unwrap();
        assert_eq!(compute_mr_iir_channel(&zero_cache, &zeros, None), 0.0);
        assert!(compute_mr_iir_channel(&zero_cache, &pred, None).is_infinite());
        let hiss: Vec<f64> = (0..n).map(|i| 1.0e-3 * ((i as f64) * 0.41).sin()).collect();
        let noisy_cache = MrIirTargetCache::new(&zeros, Some(&hiss), 64, 48_000.0).unwrap();
        assert_eq!(compute_mr_iir_channel(&noisy_cache, &zeros, None), 0.0);

        let mut grad = vec![0.0_f64; n];
        let _ = eval_mr_iir_loss_and_adjoint(&cache, &pred, Some(&mut grad));
        let eps = 1.0e-6_f64;
        for idx in [12, 63, 128, 350, 610] {
            let mut pp = pred.clone();
            let mut pm = pred.clone();
            pp[idx] += eps;
            pm[idx] -= eps;
            let fd = (eval_mr_iir_loss_and_adjoint(&cache, &pp, None)
                - eval_mr_iir_loss_and_adjoint(&cache, &pm, None))
                / (2.0 * eps);
            assert!(
                (grad[idx] - fd).abs() <= 1.0e-5 * grad[idx].abs().max(1.0e-4),
                "MR-IIR idx={idx} ana={} fd={fd}",
                grad[idx]
            );
        }

        // Noise-floor SNR weighting + smooth excess softplus matches central FD cleanly.
        let noise_cache = MrIirTargetCache::new(&target, Some(&hiss), 64, 48_000.0).unwrap();
        let mut grad_n = vec![0.0_f64; n];
        let val_n = eval_mr_iir_loss_and_adjoint(&noise_cache, &pred, Some(&mut grad_n));
        assert!(val_n.is_finite() && val_n > 0.0);
        for idx in [128, 350] {
            let mut pp = pred.clone();
            let mut pm = pred.clone();
            pp[idx] += eps;
            pm[idx] -= eps;
            let fd_n = (eval_mr_iir_loss_and_adjoint(&noise_cache, &pp, None)
                - eval_mr_iir_loss_and_adjoint(&noise_cache, &pm, None))
                / (2.0 * eps);
            assert!(
                (grad_n[idx] - fd_n).abs() <= 1.0e-5 * grad_n[idx].abs().max(1.0e-4),
                "MR-IIR noisy idx={idx} ana={} fd={fd_n}",
                grad_n[idx]
            );
        }
    }
}
