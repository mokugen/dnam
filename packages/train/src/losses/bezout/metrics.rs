//! Uncertain-floor & joint-interval `bESR` bounds and duration-weighted `20 ms` `SegESR`
//! evaluation (`ref[qual.besr.uncertain-floor]`, `ref[qual.besr.joint-intervals]`,
//! `ref[qual.companion.seg-esr]`).

use crate::error::TrainError;

use super::super::certification::{
    DrySilentGapMask, POST_T60_GAP_AMP_RATIO, SEG_ESR_UNEXCITED_BAND_AMP_RATIO,
    SEG_ESR_UNEXCITED_BAND_GUARD_RATIO,
};
use super::super::noise_floor::{smooth_excess_sq, snr_weight};
use super::{Bezout7FilterBank, esr_ratio, linear_to_db};

/// Decision under uncertain or interval noise/signal bounds (`ref[qual.besr.uncertain-floor]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BesrIntervalDecision {
    /// Passes at all feasible points including the worst-case bound.
    GuaranteedPass,
    /// Fails even at the most optimistic bound.
    GuaranteedFail,
    /// Endpoints disagree across the uncertainty box.
    Unavailable,
}

/// Exact uncertain noise-floor `bESR` interval `[c_min, c_max]` (`ref[qual.besr.uncertain-floor]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BesrIntervalBounds {
    /// Guaranteed lower bound on qualified `bESR` over `f_lo <= F <= f_hi`.
    pub min_besr_lin: f64,
    /// Guaranteed upper bound on qualified `bESR` over `f_lo <= F <= f_hi`.
    pub max_besr_lin: f64,
    /// Conservative tri-state decision across `[f_lo, f_hi]`.
    pub decision: BesrIntervalDecision,
}

/// Simultaneous rectangular bounds for raw `r(Y, R, F)` and qualified `c(Y, R, F)`
/// over `Y in [sig_lo, sig_hi]`, `R in [err_lo, err_hi]`, `F in [f_lo, f_hi]`
/// (`ref[qual.besr.joint-intervals]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BesrJointIntervalBounds {
    /// Guaranteed lower bound on composite raw `bESR`.
    pub raw_min_lin: f64,
    /// Guaranteed upper bound on composite raw `bESR`.
    pub raw_max_lin: f64,
    /// Guaranteed lower bound on composite noise-qualified `bESR`.
    pub qual_min_lin: f64,
    /// Guaranteed upper bound on composite noise-qualified `bESR`.
    pub qual_max_lin: f64,
    /// Conservative tri-state decision across the joint rectangle.
    pub decision: BesrIntervalDecision,
}

fn band_qualifies_exact(y: f64, r: f64, f: f64, g: f64) -> bool {
    f > 0.0 && (y / f) < (1.0 / g) && (r / f).powi(10) <= 10.0 + 1.0e-12
}

fn validate_nonneg_finite(vals: &[f64], ctx: &'static str) -> Result<(), TrainError> {
    if vals.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(TrainError::NonFiniteInput { context: ctx });
    }
    Ok(())
}

/// Computes exact worst-case `bESR` bounds over an uncertain noise floor interval `[f_lo, f_hi]`
/// checking qualification thresholds and non-finite inputs (`ref[qual.besr.uncertain-floor]`).
pub fn evaluate_uncertain_floor_besr(
    sig_e: [f64; 3],
    err_e: [f64; 3],
    f_lo: [f64; 3],
    f_hi: [f64; 3],
    per_band_ceil_db: [f64; 3],
    composite_ceil_db: f64,
) -> Result<BesrIntervalBounds, TrainError> {
    validate_nonneg_finite(&sig_e, "evaluate_uncertain_floor_besr(sig_e)")?;
    validate_nonneg_finite(&err_e, "evaluate_uncertain_floor_besr(err_e)")?;
    validate_nonneg_finite(&f_lo, "evaluate_uncertain_floor_besr(f_lo)")?;
    validate_nonneg_finite(&f_hi, "evaluate_uncertain_floor_besr(f_hi)")?;
    if !composite_ceil_db.is_finite() || per_band_ceil_db.iter().any(|v| !v.is_finite()) {
        return Err(TrainError::NonFiniteInput {
            context: "evaluate_uncertain_floor_besr(ceil_db)",
        });
    }
    for b in 0..3 {
        if f_lo[b] > f_hi[b] {
            return Err(TrainError::InvalidDimensions {
                context: "evaluate_uncertain_floor_besr(f_lo <= f_hi)",
                expected: 1,
                actual: 0,
            });
        }
    }

    let mut sum_min = 0.0_f64;
    let mut sum_max = 0.0_f64;
    let mut all_pass_worst = true;
    let mut all_pass_best = true;

    for b in 0..3 {
        let lo = f_lo[b];
        let hi = f_hi[b];
        let y = sig_e[b];
        let r = err_e[b];
        let g = 10.0_f64.powf(per_band_ceil_db[b] / 10.0);

        let (r_at_lo, ok_lo) = esr_ratio(y, r, lo);
        let (r_at_hi, ok_hi) = esr_ratio(y, r, hi);
        let q_lo = band_qualifies_exact(y, r, lo, g);
        let q_hi = band_qualifies_exact(y, r, hi, g);

        all_pass_worst &= ok_lo && (q_lo || r_at_lo <= g);
        all_pass_best &= ok_hi && (q_hi || r_at_hi <= g);

        sum_min += if q_hi { 0.0 } else { r_at_hi };
        sum_max += if q_lo { 0.0 } else { r_at_lo };
    }

    let min_besr_lin = sum_min / 3.0;
    let max_besr_lin = sum_max / 3.0;
    let g_comp = 10.0_f64.powf(composite_ceil_db / 10.0);
    let pass_worst = all_pass_worst && max_besr_lin <= g_comp;
    let pass_best = all_pass_best && min_besr_lin <= g_comp;
    let decision = if pass_worst {
        BesrIntervalDecision::GuaranteedPass
    } else if !pass_best {
        BesrIntervalDecision::GuaranteedFail
    } else {
        BesrIntervalDecision::Unavailable
    };
    Ok(BesrIntervalBounds {
        min_besr_lin,
        max_besr_lin,
        decision,
    })
}

/// Evaluates simultaneous rectangular interval bounds over `Y in [sig_lo, sig_hi]`,
/// `R in [err_lo, err_hi]`, and `F in [f_lo, f_hi]` (`ref[qual.besr.joint-intervals]`).
pub fn evaluate_joint_interval_besr(
    sig: ([f64; 3], [f64; 3]),
    err: ([f64; 3], [f64; 3]),
    floor: ([f64; 3], [f64; 3]),
    per_band_ceil_db: [f64; 3],
    composite_ceil_db: f64,
) -> Result<BesrJointIntervalBounds, TrainError> {
    let (sig_lo, sig_hi) = sig;
    let (err_lo, err_hi) = err;
    let (f_lo, f_hi) = floor;
    for slice in [&sig_lo, &sig_hi, &err_lo, &err_hi, &f_lo, &f_hi] {
        validate_nonneg_finite(slice, "evaluate_joint_interval_besr")?;
    }
    if !composite_ceil_db.is_finite() || per_band_ceil_db.iter().any(|v| !v.is_finite()) {
        return Err(TrainError::NonFiniteInput {
            context: "evaluate_joint_interval_besr(ceil_db)",
        });
    }
    for b in 0..3 {
        if sig_lo[b] > sig_hi[b] || err_lo[b] > err_hi[b] || f_lo[b] > f_hi[b] {
            return Err(TrainError::InvalidDimensions {
                context: "evaluate_joint_interval_besr(lo <= hi)",
                expected: 1,
                actual: 0,
            });
        }
    }

    let mut raw_min_sum = 0.0_f64;
    let mut raw_max_sum = 0.0_f64;
    let mut qual_min_sum = 0.0_f64;
    let mut qual_max_sum = 0.0_f64;
    let mut all_pass_worst = true;
    let mut all_pass_best = true;

    for b in 0..3 {
        let g = 10.0_f64.powf(per_band_ceil_db[b] / 10.0);
        let (r_minus, ok_best) = esr_ratio(sig_hi[b], err_lo[b], f_hi[b]);
        let (r_plus, ok_worst) = esr_ratio(sig_lo[b], err_hi[b], f_lo[b]);
        raw_min_sum += r_minus;
        raw_max_sum += r_plus;

        let c_plus = if f_lo[b] == 0.0 || (err_hi[b] / f_lo[b]).powi(10) > 10.0 + 1.0e-12 {
            r_plus
        } else if sig_hi[b] * g < f_lo[b] {
            0.0
        } else {
            esr_ratio(sig_lo[b].max(f_lo[b] / g), err_hi[b], f_lo[b]).0
        };

        let c_minus = if (f_hi[b] > 0.0
            && (err_lo[b] / f_hi[b]).powi(10) <= 10.0 + 1.0e-12
            && sig_lo[b] * g < f_hi[b])
            || band_qualifies_exact(sig_hi[b], err_lo[b], f_hi[b], g)
        {
            0.0
        } else {
            r_minus
        };

        qual_min_sum += c_minus;
        qual_max_sum += c_plus;

        all_pass_worst &= ok_worst && c_plus <= g;
        all_pass_best &= ok_best && c_minus <= g;
    }

    let g_comp = 10.0_f64.powf(composite_ceil_db / 10.0);
    let qual_min_lin = qual_min_sum / 3.0;
    let qual_max_lin = qual_max_sum / 3.0;
    let decision = if all_pass_worst && qual_max_lin <= g_comp {
        BesrIntervalDecision::GuaranteedPass
    } else if !(all_pass_best && qual_min_lin <= g_comp) {
        BesrIntervalDecision::GuaranteedFail
    } else {
        BesrIntervalDecision::Unavailable
    };

    Ok(BesrJointIntervalBounds {
        raw_min_lin: raw_min_sum / 3.0,
        raw_max_lin: raw_max_sum / 3.0,
        qual_min_lin,
        qual_max_lin,
        decision,
    })
}

/// Computes normative duration-weighted `20 ms` (`L = round(fs/50)`) `SegESR` in dB for one channel
/// (`ref[qual.companion.seg-esr]`), optionally gating noiseless post-`T60` dry-silent gaps via `gap_mask`.
#[must_use]
pub fn compute_seg_esr_channel_db_with_gaps(
    pred: &[f64],
    target: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup: usize,
    gap_mask: Option<&DrySilentGapMask>,
) -> f64 {
    let n = pred.len().min(target.len());
    if n <= warmup
        || sample_rate_hz < 8_000.0
        || !sample_rate_hz.is_finite()
        || !pred[..n].iter().all(|v| v.is_finite())
        || !target[..n].iter().all(|v| v.is_finite())
    {
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
    let bank = Bezout7FilterBank::new(sample_rate_hz);
    let y_b = bank.split_3band(&target[..n]);
    let yh_b = bank.split_3band(&pred[..n]);
    let n_b = noise.map(|ns| bank.split_3band(&ns[..n]));

    let active_factor = 10.0_f64.powf(0.6);
    let full_len = (sample_rate_hz / 50.0).round().max(1.0) as usize;
    let mut w_sum = 0.0_f64;
    let mut num_sum = 0.0_f64;
    let mut start = warmup;
    while start < n {
        let end = (start + full_len).min(n);
        let nf = (end - start) as f64;
        let w_f = nf / sample_rate_hz;
        let y_frame_e: f64 = target[start..end].iter().map(|v| v * v).sum();
        let f_frame_e: f64 = noise.map_or(0.0, |ns| ns[start..end].iter().map(|v| v * v).sum());
        if let Some(gm) = eff_gaps
            && gm.overlap_span(start, end).is_some()
            && (y_frame_e / nf).sqrt() <= POST_T60_GAP_AMP_RATIO * gm.active_target_rms
        {
            let r_frame_e: f64 = target[start..end]
                .iter()
                .zip(&pred[start..end])
                .map(|(y, yh)| (yh - y) * (yh - y))
                .sum();
            if (r_frame_e / nf).sqrt() > POST_T60_GAP_AMP_RATIO * gm.active_target_rms {
                return f64::INFINITY;
            }
            start = end;
            continue;
        }
        let is_active = if f_frame_e > 0.0 {
            y_frame_e > active_factor * f_frame_e
        } else {
            y_frame_e > 0.0
        };
        if !is_active {
            let r_frame_e: f64 = target[start..end]
                .iter()
                .zip(&pred[start..end])
                .map(|(y, yh)| (yh - y) * (yh - y))
                .sum();
            let guard_ok = if f_frame_e > 0.0 {
                (r_frame_e / f_frame_e).powi(10) <= 10.0 + 1.0e-12
            } else {
                r_frame_e == 0.0
            };
            if !guard_ok {
                return f64::INFINITY;
            }
            start = end;
            continue;
        }

        let bb_sig_rms = (y_frame_e / nf).sqrt();
        let mut r_f = 0.0_f64;
        let mut excited_bands = 0_usize;
        for b in 0..3 {
            let ey = (y_b[b][start..end].iter().map(|v| v * v).sum::<f64>() / nf).sqrt();
            let ep = (yh_b[b][start..end].iter().map(|v| v * v).sum::<f64>() / nf).sqrt();
            let f_env = n_b.as_ref().map_or(0.0, |nb| {
                nb[b][start..end].iter().map(|v| v * v).sum::<f64>() / nf
            });
            if noise.is_none()
                && f_env == 0.0
                && ey <= SEG_ESR_UNEXCITED_BAND_AMP_RATIO * bb_sig_rms
            {
                if (ep - ey).abs() > SEG_ESR_UNEXCITED_BAND_GUARD_RATIO * bb_sig_rms {
                    return f64::INFINITY;
                }
                continue;
            }
            let denom = ey * ey + f_env;
            if denom > 0.0 {
                let d = ep - ey;
                r_f += ((d * d - f_env).max(0.0)) / denom;
                excited_bands += 1;
            } else if ep > 0.0 {
                return f64::INFINITY;
            }
        }
        if excited_bands == 0 {
            return f64::INFINITY;
        }
        num_sum += w_f * (r_f / (excited_bands as f64));
        w_sum += w_f;
        start = end;
    }
    if w_sum <= 0.0 {
        if eff_gaps.is_some() {
            f64::INFINITY
        } else {
            linear_to_db(0.0)
        }
    } else {
        linear_to_db(num_sum / w_sum)
    }
}

/// Computes normative duration-weighted `20 ms` (`L = round(fs/50)`) `SegESR` in dB for one channel
/// (`ref[qual.companion.seg-esr]`).
#[must_use]
pub fn compute_seg_esr_channel_db(
    pred: &[f64],
    target: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup: usize,
) -> f64 {
    compute_seg_esr_channel_db_with_gaps(pred, target, noise, sample_rate_hz, warmup, None)
}

#[derive(Clone, Debug)]
struct SegEsrFrameTarget {
    start: usize,
    end: usize,
    wf: f64,
    ey: [f64; 3],
    denom: [f64; 3],
    f_env: [f64; 3],
    snr_w: [f64; 3],
}

/// Precomputed 20 ms 3-band Bézout envelope targets (`SegESR`) with optional repeatability noise-floor
/// weights (`ref[qual.companion.seg-esr]`, `ref[qual.eval.thermal-hiss]`).
#[derive(Clone, Debug)]
pub struct SegEsrTargetCache {
    len: usize,
    frames: Vec<SegEsrFrameTarget>,
}

impl SegEsrTargetCache {
    /// Precomputes 20 ms 3-band target RMS envelopes and optional noise-floor SNR weights.
    pub fn new(
        target: &[f64],
        noise: Option<&[f64]>,
        bank: &Bezout7FilterBank,
        sample_rate_hz: f64,
        warmup: usize,
    ) -> Result<Self, TrainError> {
        let n = target.len();
        if n <= warmup
            || sample_rate_hz < 8_000.0
            || !sample_rate_hz.is_finite()
            || !target.iter().all(|v| v.is_finite())
        {
            return Err(TrainError::InvalidDimensions {
                context: "SegEsrTargetCache::new",
                expected: warmup + 1,
                actual: n,
            });
        }
        if let Some(ns) = noise
            && (ns.len() < n || !ns[..n].iter().all(|v| v.is_finite()))
        {
            return Err(TrainError::InvalidDimensions {
                context: "SegEsrTargetCache::new(noise)",
                expected: n,
                actual: ns.len(),
            });
        }
        let y_b = bank.split_3band(target);
        let n_b = noise.map(|ns| bank.split_3band(&ns[..n]));
        let full_len = (sample_rate_hz / 50.0).round().max(1.0) as usize;
        let total_dur = ((n - warmup) as f64) / sample_rate_hz;
        let mut frames = Vec::with_capacity((n - warmup).div_ceil(full_len));

        let mut start = warmup;
        while start < n {
            let end = (start + full_len).min(n);
            let wf = ((end - start) as f64 / sample_rate_hz) / (3.0 * total_dur);
            let mut ey = [0.0_f64; 3];
            let mut denom = [0.0_f64; 3];
            let mut f_env = [0.0_f64; 3];
            let mut snr_w = [1.0_f64; 3];
            for b in 0..3 {
                let ey_sq: f64 = y_b[b][start..end].iter().map(|v| v * v).sum();
                ey[b] = (ey_sq + 1.0e-18).sqrt();
                denom[b] = ey_sq + 1.0e-12;
                let fe = n_b
                    .as_ref()
                    .map_or(0.0, |nb| nb[b][start..end].iter().map(|v| v * v).sum());
                f_env[b] = fe;
                if fe > 0.0 {
                    snr_w[b] = snr_weight(ey_sq, fe);
                }
            }
            frames.push(SegEsrFrameTarget {
                start,
                end,
                wf,
                ey,
                denom,
                f_env,
                snr_w,
            });
            start = end;
        }
        Ok(Self { len: n, frames })
    }
}

/// Evaluates cached differentiable 20 ms `SegESR` envelope loss using [`SegEsrTargetCache`] and
/// accumulates exact adjoint `dL/dy_hat` into `grad_opt`.
pub fn eval_seg_esr_cached_and_adjoint(
    cache: &SegEsrTargetCache,
    bank: &Bezout7FilterBank,
    pred: &[f64],
    grad_opt: Option<&mut [f64]>,
) -> f64 {
    let n = cache.len.min(pred.len());
    if n < cache.len || cache.frames.is_empty() {
        return 0.0;
    }
    let yh_b = bank.split_3band(&pred[..n]);
    let mut loss = 0.0_f64;
    let mut d_bands = grad_opt
        .is_some()
        .then(|| [vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]]);

    for frame in &cache.frames {
        for b in 0..3 {
            let ep_sq: f64 = yh_b[b][frame.start..frame.end].iter().map(|v| v * v).sum();
            let ep = (ep_sq + 1.0e-18).sqrt();
            let diff = ep - frame.ey[b];
            let (e_eff, d_e) = smooth_excess_sq(diff * diff, frame.f_env[b]);
            let w_b = frame.wf * frame.snr_w[b];
            loss += w_b * e_eff / frame.denom[b];
            if let Some(ref mut db) = d_bands {
                let scale = w_b * (2.0 * d_e * diff / (ep * frame.denom[b]));
                for t in frame.start..frame.end {
                    db[b][t] = scale * yh_b[b][t];
                }
            }
        }
    }
    if let (Some(grad), Some(db)) = (grad_opt, d_bands.as_ref()) {
        bank.accum_3band_adjoint(db, &mut grad[..n]);
    }
    loss
}

/// Evaluates differentiable `SegESR` envelope loss and accumulates exact adjoint `dL/dy_hat`.
pub fn eval_seg_esr_loss_and_adjoint(
    pred: &[f64],
    target: &[f64],
    sample_rate_hz: f64,
    warmup: usize,
    grad_opt: Option<&mut [f64]>,
) -> f64 {
    let n = pred.len().min(target.len());
    if n <= warmup || sample_rate_hz < 8_000.0 {
        return 0.0;
    }
    let bank = Bezout7FilterBank::new(sample_rate_hz);
    match SegEsrTargetCache::new(&target[..n], None, &bank, sample_rate_hz, warmup) {
        Ok(cache) => eval_seg_esr_cached_and_adjoint(&cache, &bank, &pred[..n], grad_opt),
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::super::compute_besr_channel;
    use super::*;

    #[test]
    fn test_interval_besr_and_segesr_edge_branches() {
        assert!(matches!(
            evaluate_uncertain_floor_besr(
                [1.0; 3],
                [0.01; 3],
                [0.0; 3],
                [0.0; 3],
                [-20.0; 3],
                f64::NAN,
            ),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            evaluate_uncertain_floor_besr(
                [1.0; 3], [0.01; 3], [2.0; 3], [1.0; 3], [-20.0; 3], -20.0,
            ),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let u_pass = evaluate_uncertain_floor_besr(
            [100.0; 3], [0.01; 3], [0.0; 3], [0.0; 3], [-20.0; 3], -20.0,
        )
        .unwrap();
        assert_eq!(u_pass.decision, BesrIntervalDecision::GuaranteedPass);
        let u_fail = evaluate_uncertain_floor_besr(
            [1.0; 3], [10.0; 3], [0.0; 3], [0.0; 3], [-20.0; 3], -20.0,
        )
        .unwrap();
        assert_eq!(u_fail.decision, BesrIntervalDecision::GuaranteedFail);

        assert!(matches!(
            evaluate_joint_interval_besr(
                ([1.0; 3], [1.0; 3]),
                ([0.01; 3], [0.01; 3]),
                ([0.0; 3], [0.0; 3]),
                [-20.0; 3],
                f64::NAN,
            ),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            evaluate_joint_interval_besr(
                ([2.0; 3], [1.0; 3]),
                ([0.01; 3], [0.01; 3]),
                ([0.0; 3], [0.0; 3]),
                [-20.0; 3],
                -20.0,
            ),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let j_pass = evaluate_joint_interval_besr(
            ([100.0; 3], [120.0; 3]),
            ([0.001; 3], [0.01; 3]),
            ([0.0; 3], [0.0; 3]),
            [-20.0; 3],
            -20.0,
        )
        .unwrap();
        assert_eq!(j_pass.decision, BesrIntervalDecision::GuaranteedPass);
        let j_fail = evaluate_joint_interval_besr(
            ([1.0; 3], [2.0; 3]),
            ([5.0; 3], [10.0; 3]),
            ([0.0; 3], [0.0; 3]),
            [-20.0; 3],
            -20.0,
        )
        .unwrap();
        assert_eq!(j_fail.decision, BesrIntervalDecision::GuaranteedFail);

        assert_eq!(
            compute_seg_esr_channel_db(&[0.0; 8], &[0.0; 8], None, 48_000.0, 16),
            f64::INFINITY
        );
        assert_eq!(
            compute_seg_esr_channel_db(&[0.0; 64], &[0.0; 64], Some(&[0.0; 8]), 48_000.0, 0),
            f64::INFINITY
        );
        let zeros = vec![0.0_f64; 1_024];
        let noise = vec![1.0e-3_f64; 1_024];
        let quiet_ok = vec![1.0e-4_f64; 1_024];
        let quiet_loud = vec![1.0_f64; 1_024];
        assert!(
            compute_seg_esr_channel_db(&quiet_ok, &zeros, Some(&noise), 48_000.0, 0).is_infinite()
        );
        assert_eq!(
            compute_seg_esr_channel_db(&quiet_loud, &zeros, Some(&noise), 48_000.0, 0),
            f64::INFINITY
        );
        assert_eq!(
            compute_seg_esr_channel_db(&quiet_loud, &zeros, None, 48_000.0, 0),
            f64::INFINITY
        );
        assert_eq!(
            eval_seg_esr_loss_and_adjoint(&[0.0; 8], &[0.0; 8], 48_000.0, 16, None),
            0.0
        );
    }

    #[test]
    fn test_interval_besr_and_segesr_guards_and_adjoint() {
        let n = 1_200_usize;
        let target: Vec<f64> = (0..n)
            .map(|t| {
                0.4 * ((t as f64) * 0.025).sin()
                    + 0.35 * ((t as f64) * 0.14).cos()
                    + 0.25 * ((t as f64) * 0.95).sin()
            })
            .collect();
        let pred: Vec<f64> = target
            .iter()
            .enumerate()
            .map(|(i, &x)| 0.9 * x + 0.02 * ((i as f64) * 0.3).cos())
            .collect();
        let mut g_seg = vec![0.0_f64; n];
        let _ = eval_seg_esr_loss_and_adjoint(&pred, &target, 48_000.0, 128, Some(&mut g_seg));
        let eps = 1.0e-6_f64;
        for idx in [20, 150, 500, 990] {
            let (mut pp, mut pm) = (pred.clone(), pred.clone());
            pp[idx] += eps;
            pm[idx] -= eps;
            let fd_s = (eval_seg_esr_loss_and_adjoint(&pp, &target, 48_000.0, 128, None)
                - eval_seg_esr_loss_and_adjoint(&pm, &target, 48_000.0, 128, None))
                / (2.0 * eps);
            assert!((g_seg[idx] - fd_s).abs() <= 1.0e-6 * g_seg[idx].abs().max(1.0e-3));
        }

        let rep = compute_besr_channel(&pred, &target, None, 48_000.0, 128);
        assert!(rep.valid && rep.besr_db < -15.0);
        assert!(compute_seg_esr_channel_db(&pred, &target, None, 48_000.0, 128) < -15.0);

        let iv = evaluate_uncertain_floor_besr(
            [0.5, 100.0, 100.0],
            [1.2, 0.01, 0.01],
            [0.9; 3],
            [1.1; 3],
            [-23.0, -20.0, -15.0],
            -18.0,
        )
        .unwrap();
        assert!(iv.min_besr_lin <= iv.max_besr_lin);
        assert!(matches!(
            evaluate_uncertain_floor_besr(
                [0.5; 3],
                [1.2; 3],
                [f64::NAN; 3],
                [1.1; 3],
                [-23.0; 3],
                -18.0,
            ),
            Err(TrainError::NonFiniteInput { .. })
        ));
        let joint = evaluate_joint_interval_besr(
            ([0.4, 90.0, 90.0], [0.6, 110.0, 110.0]),
            ([1.0, 0.005, 0.005], [1.2, 0.015, 0.015]),
            ([0.9, 0.0, 0.0], [1.1, 0.0, 0.0]),
            [-23.0, -20.0, -15.0],
            -18.0,
        )
        .unwrap();
        assert!(joint.qual_min_lin <= joint.qual_max_lin);

        let silent = vec![0.0_f64; n];
        let nonzero = vec![1.0e-2_f64; n];
        assert!(compute_seg_esr_channel_db(&nonzero, &silent, None, 48_000.0, 128).is_infinite());
        assert_eq!(
            compute_seg_esr_channel_db(&silent, &silent, None, 48_000.0, 128),
            f64::NEG_INFINITY
        );

        // Counterexample 1: point interval failing qualification condition 3 (`R > 10^0.1 F`) still
        // falls back to raw ratio `r <= g`, matching `evaluate_noise_qualified_besr`.
        let uf_pt = evaluate_uncertain_floor_besr(
            [9.0; 3], [1.5; 3], [1.0; 3], [1.0; 3], [-10.0; 3], -10.0,
        )
        .unwrap();
        assert_eq!(uf_pt.decision, BesrIntervalDecision::GuaranteedPass);

        // Counterexample 2: joint rectangle where all points pass via qualification or low raw ratio.
        let j_pass_box = evaluate_joint_interval_besr(
            ([0.0; 3], [12.0; 3]),
            ([1.2; 3], [1.2; 3]),
            ([1.0; 3], [1.0; 3]),
            [-10.0; 3],
            -10.0,
        )
        .unwrap();
        assert_eq!(j_pass_box.decision, BesrIntervalDecision::GuaranteedPass);

        // Noisy cached SegESR matches central FD accurately.
        let bank = Bezout7FilterBank::new(48_000.0);
        let noise_s: Vec<f64> = (0..n).map(|t| 0.02 * ((t as f64) * 0.41).sin()).collect();
        let seg_cache =
            SegEsrTargetCache::new(&target, Some(&noise_s), &bank, 48_000.0, 128).unwrap();
        let mut g_sn = vec![0.0_f64; n];
        let _ = eval_seg_esr_cached_and_adjoint(&seg_cache, &bank, &pred, Some(&mut g_sn));
        let (mut pp, mut pm) = (pred.clone(), pred.clone());
        pp[500] += eps;
        pm[500] -= eps;
        let fd_sn = (eval_seg_esr_cached_and_adjoint(&seg_cache, &bank, &pp, None)
            - eval_seg_esr_cached_and_adjoint(&seg_cache, &bank, &pm, None))
            / (2.0 * eps);
        assert!((g_sn[500] - fd_sn).abs() <= 1.0e-5 * g_sn[500].abs().max(1.0e-4));

        // Unexcited LF band on pure 17 kHz active tone: -55 dB LF transient passes cleanly,
        // while -30 dB spurious LF (> 1e-2 * bb_sig_rms) fails closed with +inf.
        let n_hf = 3_840_usize;
        let tau = core::f64::consts::TAU;
        let tgt_17k: Vec<f64> = (0..n_hf)
            .map(|i| 0.4 * (tau * 17_000.0 * (i as f64) / 48_000.0).sin())
            .collect();
        let mdl_lf_ok: Vec<f64> = tgt_17k
            .iter()
            .enumerate()
            .map(|(i, &y)| y + 0.4 * 1.8e-3 * (tau * 25.0 * (i as f64) / 48_000.0).sin())
            .collect();
        let seg_hf_ok = compute_seg_esr_channel_db(&mdl_lf_ok, &tgt_17k, None, 48_000.0, 960);
        assert!(
            seg_hf_ok.is_finite() && seg_hf_ok < -30.0,
            "seg_hf_ok={seg_hf_ok}"
        );

        let mdl_lf_bad: Vec<f64> = tgt_17k
            .iter()
            .enumerate()
            .map(|(i, &y)| y + 0.4 * 3.0e-2 * (tau * 25.0 * (i as f64) / 48_000.0).sin())
            .collect();
        assert!(
            compute_seg_esr_channel_db(&mdl_lf_bad, &tgt_17k, None, 48_000.0, 960).is_infinite()
        );

        let zero_noise = vec![1.0e-12_f64; n_hf];
        assert!(
            compute_seg_esr_channel_db(&mdl_lf_ok, &tgt_17k, Some(&zero_noise), 48_000.0, 960)
                .is_finite()
        );
    }
}
