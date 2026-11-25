#![forbid(unsafe_code)]
//! 5-Scale Floor-Clamped Multi-Resolution IIR Resonator Bank (`MR-IIR`, `r[qual.companion.mr-iir]`).

mod cache;

pub(crate) use cache::MrIirTargetCache;

use std::f64::consts::PI;

use super::{DrySilentGapMask, MAX_METRIC_SAMPLE_RATE_HZ, POST_T60_GAP_AMP_RATIO};

const SCALES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const NUM_PROBES: usize = 32;
const PROBE_GROUPS: usize = NUM_PROBES / 4;
const SUB_CHUNK: usize = 64;

#[inline]
fn has_tiny_nonzero(sig: &[f64]) -> bool {
    sig.iter().any(|&v| v != 0.0 && v.abs() < 1.0e-100)
}

#[inline]
fn probe_norm_and_trc(
    m_scale: usize,
    sample_rate_hz: f64,
    f_max: f64,
) -> (f64, f64, [[f64; 4]; PROBE_GROUPS]) {
    let r_m = (-PI / (m_scale as f64)).exp();
    let r_m_sq = r_m * r_m;
    let norm = 0.5 * (1.0 - r_m_sq);
    let mut trc = [[0.0_f64; 4]; PROBE_GROUPS];
    for (g, grp) in trc.iter_mut().enumerate() {
        for (l, slot) in grp.iter_mut().enumerate() {
            let p = g * 4 + l;
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            let omega = 2.0 * PI * f_k / sample_rate_hz;
            *slot = 2.0 * r_m * omega.cos();
        }
    }
    (norm, r_m_sq, trc)
}

/// Computes the 5-Scale Floor-Clamped Multi-Resolution IIR Resonator Bank Loss (`MR-IIR`),
/// optionally gating noiseless post-`T60` dry-silent gaps via `gap_mask` while enforcing a strict
/// `-60 dB` residual guard across each excluded gap.
#[must_use]
pub fn compute_mr_iir_loss_with_gaps(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
    gap_mask: Option<&DrySilentGapMask>,
) -> f64 {
    let eff_gaps = if noise.is_some() {
        None
    } else {
        gap_mask.filter(|g| !g.is_empty())
    };
    if eff_gaps.is_none() {
        return compute_mr_iir_loss(target, model, noise, sample_rate_hz, warmup_samples);
    }
    compute_mr_iir_loss_general_with_gaps(
        target,
        model,
        noise,
        sample_rate_hz,
        warmup_samples,
        eff_gaps,
    )
}

/// Computes the 5-Scale Floor-Clamped Multi-Resolution IIR Resonator Bank Loss (`MR-IIR`,
/// $M \in \{256, 512, 1024, 2048, 4096\}$, pole radius $r_M = \exp(-\pi / M)$, hop $H = M / 4$,
/// $K = 32$ log-spaced probes, `ref[qual.companion.mr-iir]`, `ref[qual.besr.finite-timeline]`).
///
/// Returns `f64::INFINITY` on empty, length-mismatched, non-finite, or invalid-rate inputs so
/// invalid evaluations never pass tier gates.
#[must_use]
pub fn compute_mr_iir_loss(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> f64 {
    if noise.is_none()
        && target.len() == model.len()
        && !has_tiny_nonzero(model)
        && let Some(cache) = MrIirTargetCache::new(target, sample_rate_hz, warmup_samples)
    {
        return cache.eval_model(model);
    }
    compute_mr_iir_loss_general(target, model, noise, sample_rate_hz, warmup_samples)
}

pub(crate) fn compute_mr_iir_loss_general(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> f64 {
    compute_mr_iir_loss_general_with_gaps(
        target,
        model,
        noise,
        sample_rate_hz,
        warmup_samples,
        None,
    )
}

fn compute_mr_iir_loss_general_with_gaps(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
    eff_gaps: Option<&DrySilentGapMask>,
) -> f64 {
    if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
        || target.is_empty()
        || target.len() != model.len()
        || warmup_samples >= target.len()
    {
        return f64::INFINITY;
    }
    let n_total = target.len();
    if !target.iter().all(|v| v.is_finite()) || !model.iter().all(|v| v.is_finite()) {
        return f64::INFINITY;
    }
    if let Some(ns) = noise
        && (ns.len() < n_total || !ns[..n_total].iter().all(|v| v.is_finite()))
    {
        return f64::INFINITY;
    }

    let f_max = (0.45 * sample_rate_hz).min(20_000.0);
    if f_max <= 20.0 {
        return f64::INFINITY;
    }
    let t_sel = n_total - warmup_samples;
    let num_full_chunks = t_sel / SUB_CHUNK;
    let mut chunk_t = vec![0.0_f64; num_full_chunks];
    let mut chunk_m = vec![0.0_f64; num_full_chunks];
    let mut chunk_n = vec![0.0_f64; num_full_chunks];
    let mut nz_t = vec![false; num_full_chunks];
    let mut nz_m = vec![false; num_full_chunks];
    let mut nz_n = vec![false; num_full_chunks];
    let mut tail_t = [0.0_f64; 4096];
    let mut tail_m = [0.0_f64; 4096];
    let mut tail_n = [0.0_f64; 4096];

    let mut total_loss = 0.0_f64;

    for &m_scale in &SCALES {
        let hop = m_scale / 4;
        let r_m = (-PI / (m_scale as f64)).exp();
        let r_m_sq = r_m * r_m;
        let norm = 0.5 * (1.0 - r_m_sq);
        let tail_start = t_sel.saturating_sub(m_scale);

        let mut sc_num_sq = 0.0_f64;
        let mut sc_den_sq = 0.0_f64;
        let mut mag_diff_sum = 0.0_f64;
        let mut cell_count = 0usize;
        let mut scale_nz = false;
        let mut scale_nz_n = false;
        let mut scale_nse_sq = 0.0_f64;

        for p in 0..NUM_PROBES {
            let frac = (p as f64) / ((NUM_PROBES - 1) as f64);
            let f_k = 20.0 * (f_max / 20.0).powf(frac);
            let omega = 2.0 * PI * f_k / sample_rate_hz;
            let two_r_cos = 2.0 * r_m * omega.cos();

            let (mut x1_t, mut x2_t, mut y1_t, mut y2_t) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut x1_m, mut x2_m, mut y1_m, mut y2_m) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let (mut x1_n, mut x2_n, mut y1_n, mut y2_n) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

            chunk_t.fill(0.0);
            chunk_m.fill(0.0);
            chunk_n.fill(0.0);
            nz_t.fill(false);
            nz_m.fill(false);
            nz_n.fill(false);

            for i in 0..n_total {
                let xt = target[i];
                let drive_t = xt - x2_t;
                let mut bt = norm * drive_t + two_r_cos * y1_t - r_m_sq * y2_t;
                if drive_t == 0.0
                    && (xt == 0.0 || xt.abs() >= 1.0e-140)
                    && bt.abs() < 1.0e-140
                    && y1_t.abs() < 1.0e-140
                {
                    bt = 0.0;
                }
                x2_t = x1_t;
                x1_t = xt;
                y2_t = y1_t;
                y1_t = bt;

                let xm = model[i];
                let drive_m = xm - x2_m;
                let mut bm = norm * drive_m + two_r_cos * y1_m - r_m_sq * y2_m;
                if drive_m == 0.0
                    && (xm == 0.0 || xm.abs() >= 1.0e-140)
                    && bm.abs() < 1.0e-140
                    && y1_m.abs() < 1.0e-140
                {
                    bm = 0.0;
                }
                x2_m = x1_m;
                x1_m = xm;
                y2_m = y1_m;
                y1_m = bm;

                let xn = noise.map_or(0.0, |ns| ns[i]);
                let drive_n = xn - x2_n;
                let mut bn = norm * drive_n + two_r_cos * y1_n - r_m_sq * y2_n;
                if drive_n == 0.0
                    && (xn == 0.0 || xn.abs() >= 1.0e-140)
                    && bn.abs() < 1.0e-140
                    && y1_n.abs() < 1.0e-140
                {
                    bn = 0.0;
                }
                x2_n = x1_n;
                x1_n = xn;
                y2_n = y1_n;
                y1_n = bn;

                if i >= warmup_samples {
                    let sub_t = xt != 0.0 && xt.abs() < 1.0e-140;
                    let sub_m = xm != 0.0 && xm.abs() < 1.0e-140;
                    let sub_n = xn != 0.0 && xn.abs() < 1.0e-140;
                    let u = i - warmup_samples;
                    let c = u / SUB_CHUNK;
                    if c < num_full_chunks {
                        chunk_t[c] += bt * bt;
                        chunk_m[c] += bm * bm;
                        chunk_n[c] += bn * bn;
                        nz_t[c] |= bt != 0.0 || sub_t;
                        nz_m[c] |= bm != 0.0 || sub_m;
                        nz_n[c] |= bn != 0.0 || sub_n;
                    }
                    if u >= tail_start {
                        let ti = u - tail_start;
                        tail_t[ti] = if bt == 0.0 && sub_t { xt } else { bt };
                        tail_m[ti] = if bm == 0.0 && sub_m { xm } else { bm };
                        tail_n[ti] = if bn == 0.0 && sub_n { xn } else { bn };
                    }
                }
            }

            let mut eval_energies = |e3: (f64, f64, f64),
                                     h3: (bool, bool, bool),
                                     span: (usize, usize, usize)|
             -> Result<(), ()> {
                let (e_t, e_m, e_n) = e3;
                let (ht, hm, hn) = h3;
                let (abs_s, abs_e, w) = span;
                if !e_t.is_finite() || !e_m.is_finite() || !e_n.is_finite() {
                    return Err(());
                }
                let inv_sqrt_len = 1.0 / (w as f64).sqrt();
                let rms_t = e_t.sqrt() * inv_sqrt_len;
                let rms_m = e_m.sqrt() * inv_sqrt_len;
                let sigma = e_n.sqrt() * inv_sqrt_len;
                if let Some(gm) = eff_gaps
                    && gm.overlap_span(abs_s, abs_e).is_some()
                    && rms_t <= POST_T60_GAP_AMP_RATIO * gm.active_target_rms
                {
                    if !rms_m.is_finite()
                        || (ht && (e_t == 0.0 || rms_t == 0.0))
                        || (hm && (e_m == 0.0 || rms_m == 0.0))
                        || (rms_m - rms_t).abs()
                            > 2.0 * POST_T60_GAP_AMP_RATIO * gm.active_target_rms
                    {
                        return Err(());
                    }
                    return Ok(());
                }
                scale_nz |= ht || hm || hn;
                scale_nz_n |= hn;
                scale_nse_sq += e_n;
                if (hn && (e_n == 0.0 || sigma == 0.0))
                    || (((ht && (e_t == 0.0 || rms_t == 0.0))
                        || (hm && (e_m == 0.0 || rms_m == 0.0)))
                        && sigma < 1.6e-162)
                {
                    return Err(());
                }
                let (a_t, a_m) = (rms_t.max(sigma), rms_m.max(sigma));
                if a_t == 0.0 && a_m == 0.0 {
                    cell_count += 1;
                    return Ok(());
                }
                if a_t == 0.0 || a_m == 0.0 || !a_t.is_finite() || !a_m.is_finite() {
                    return Err(());
                }
                let diff = a_t - a_m;
                sc_num_sq += diff * diff;
                sc_den_sq += a_t * a_t;
                mag_diff_sum += (a_t.ln() - a_m.ln()).abs();
                cell_count += 1;
                Ok(())
            };

            let sum_tail = |len: usize,
                            tt: &[f64],
                            tm: &[f64],
                            tn: &[f64]|
             -> (f64, f64, f64, bool, bool, bool) {
                let (mut et, mut em, mut en) = (0.0_f64, 0.0_f64, 0.0_f64);
                let (mut ht, mut hm, mut hn) = (false, false, false);
                for k in 0..len {
                    let (vt, vm, vn) = (tt[k], tm[k], tn[k]);
                    et += vt * vt;
                    em += vm * vm;
                    en += vn * vn;
                    ht |= vt != 0.0;
                    hm |= vm != 0.0;
                    hn |= vn != 0.0;
                }
                (et, em, en, ht, hm, hn)
            };

            if t_sel < m_scale {
                let (et, em, en, ht, hm, hn) = sum_tail(t_sel, &tail_t, &tail_m, &tail_n);
                if eval_energies(
                    (et, em, en),
                    (ht, hm, hn),
                    (warmup_samples, warmup_samples + t_sel, t_sel),
                )
                .is_err()
                {
                    return f64::INFINITY;
                }
            } else {
                let chunks_per_win = m_scale / SUB_CHUNK;
                let chunks_per_hop = hop / SUB_CHUNK;
                let mut c0 = 0usize;
                while (c0 + chunks_per_win) * SUB_CHUNK <= t_sel {
                    let c_end = c0 + chunks_per_win;
                    let et: f64 = chunk_t[c0..c_end].iter().sum();
                    let em: f64 = chunk_m[c0..c_end].iter().sum();
                    let en: f64 = chunk_n[c0..c_end].iter().sum();
                    let ht = nz_t[c0..c_end].iter().any(|&b| b);
                    let hm = nz_m[c0..c_end].iter().any(|&b| b);
                    let hn = nz_n[c0..c_end].iter().any(|&b| b);
                    let abs_s = warmup_samples + c0 * SUB_CHUNK;
                    if eval_energies(
                        (et, em, en),
                        (ht, hm, hn),
                        (abs_s, abs_s + m_scale, m_scale),
                    )
                    .is_err()
                    {
                        return f64::INFINITY;
                    }
                    c0 += chunks_per_hop;
                }
                let start = c0 * SUB_CHUNK;
                let last_start = t_sel - m_scale;
                if last_start + hop != start {
                    let (et, em, en, ht, hm, hn) = sum_tail(m_scale, &tail_t, &tail_m, &tail_n);
                    let abs_s = warmup_samples + last_start;
                    if eval_energies(
                        (et, em, en),
                        (ht, hm, hn),
                        (abs_s, warmup_samples + t_sel, m_scale),
                    )
                    .is_err()
                    {
                        return f64::INFINITY;
                    }
                }
            }
        }

        if !sc_num_sq.is_finite()
            || !sc_den_sq.is_finite()
            || !mag_diff_sum.is_finite()
            || (scale_nz_n && scale_nse_sq == 0.0)
            || (eff_gaps.is_some() && cell_count == 0)
        {
            return f64::INFINITY;
        }
        let l_sc = if sc_den_sq == 0.0 {
            if sc_num_sq == 0.0 && !scale_nz {
                0.0
            } else {
                return f64::INFINITY;
            }
        } else {
            sc_num_sq.sqrt() / sc_den_sq.sqrt()
        };
        let l_mag = if cell_count == 0 {
            0.0
        } else {
            mag_diff_sum / (cell_count as f64)
        };
        total_loss += l_sc + l_mag;
    }

    let out = total_loss / (SCALES.len() as f64);
    if out.is_finite() { out } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    #[test]
    fn test_mr_iir_gap_guard_scores_actively_ringing_long_scale_windows_and_rejects_silent_ringing()
    {
        let sr = 48_000.0_f64;
        let warmup = 64_usize;
        let n = 16_384_usize;
        let mut dry = vec![0.0_f64; n];
        for (i, slot) in dry.iter_mut().enumerate() {
            if !(2_400..14_400).contains(&i) {
                *slot = 0.30 * (TAU * 180.0 * (i as f64) / sr).sin()
                    + 0.20 * (TAU * 1_100.0 * (i as f64) / sr).cos();
            }
        }
        let r_p = 0.992_f64;
        let (ar, ai) = (
            r_p * (TAU * 110.0 / sr).cos(),
            r_p * (TAU * 110.0 / sr).sin(),
        );
        let (mut sr_st, mut si_st) = (0.0_f64, 0.0_f64);
        let mut target = vec![0.0_f64; n];
        for i in 0..n {
            let nsr = ar * sr_st - ai * si_st + dry[i];
            let nsi = ai * sr_st + ar * si_st;
            sr_st = nsr;
            si_st = nsi;
            target[i] = 0.55 * dry[i] + 0.04 * sr_st;
        }
        let gap_mask = DrySilentGapMask::detect(&dry, &target, warmup, sr);
        assert!(!gap_mask.is_empty());

        let mut approx_pred: Vec<f64> = target.iter().map(|&v| 0.95 * v).collect();
        for (i, slot) in approx_pred.iter_mut().enumerate().take(14_400).skip(6_500) {
            *slot = 1.0e-5 * (-((i - 6_500) as f64) / 2_000.0).exp();
        }
        let mr_ok =
            compute_mr_iir_loss_with_gaps(&target, &approx_pred, None, sr, warmup, Some(&gap_mask));
        assert!(
            mr_ok.is_finite() && mr_ok > 0.0,
            "expected finite MR-IIR, got {mr_ok}"
        );

        let mut ring_pred = approx_pred;
        for (i, slot) in ring_pred.iter_mut().enumerate().take(14_000).skip(7_200) {
            *slot = 0.015 * (TAU * 440.0 * (i as f64) / sr).sin();
        }
        let mr_bad =
            compute_mr_iir_loss_with_gaps(&target, &ring_pred, None, sr, warmup, Some(&gap_mask));
        assert!(
            mr_bad.is_infinite(),
            "expected +inf for silent-gap ringing, got {mr_bad}"
        );
    }
}
