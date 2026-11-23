//! 33-band base-10 1/3-octave noise-adjusted `LTAS` certification metric, Welch-33 LTAS
//! loss (`L2Huber`, `HingeMaxDb`, `LogSumExpMinimax`), ultrasonic `[20 kHz, fs/2]` hinge, and
//! exact reverse-mode FFT adjoints (`ref[qual.companion.ltas]`).

mod mode_eval;

use crate::error::TrainError;

use super::bezout::ONE_DB_POWER_FACTOR;
use super::fft::{
    PairedFftWorkspace, Radix2FftPlan, build_hann_window, compute_welch_seg_starts,
    paired_real_backward_adjoint, paired_real_forward_spectra,
};
use super::noise_floor::snr_weight;
use mode_eval::{eval_ltas_mode_terms, nyquist_hinge_loss_and_deriv};

type WelchPowerResult = ([f64; 33], f64, Vec<(Vec<f64>, Vec<f64>)>);

const NYQ_STOPBAND_TAG: u8 = 33;
const _: () = assert!(NYQ_STOPBAND_TAG == 33);

/// Single frequency interval in the canonical 33-band base-10 1/3-octave LTAS partition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtasInterval {
    /// Lower interval edge in Hz.
    pub low_hz: f64,
    /// Upper interval edge in Hz.
    pub high_hz: f64,
    /// Whether `low_hz` is inclusive.
    pub include_low: bool,
    /// Whether `high_hz` is inclusive.
    pub include_high: bool,
}

impl LtasInterval {
    /// Geometric center frequency `sqrt(low_hz * high_hz)` in Hz.
    #[inline]
    #[must_use]
    pub fn center_hz(self) -> f64 {
        (self.low_hz * self.high_hz).sqrt()
    }

    /// Tests whether `f_hz` belongs to this interval.
    #[inline]
    #[must_use]
    pub fn contains(self, f_hz: f64) -> bool {
        let lo_ok = if self.include_low {
            f_hz >= self.low_hz
        } else {
            f_hz > self.low_hz
        };
        let hi_ok = if self.include_high {
            f_hz <= self.high_hz
        } else {
            f_hz < self.high_hz
        };
        lo_ok && hi_ok
    }
}

/// Returns the canonical 33-band base-10 1/3-octave intervals over `[20 Hz, 20,000 Hz]`
/// (`ref[qual.companion.ltas]`: 21 Sub/Low-Mid `[20, 2000)`, 11 HF `[2000, 18000]`, 1 Upper Air `(18000, 20000]`).
#[must_use]
pub fn canonical_ltas_33_intervals_hz() -> [LtasInterval; 33] {
    let edge = |num: i32| -> f64 { 1000.0 * 10.0_f64.powf(f64::from(num) / 20.0) };
    let mut out = [LtasInterval {
        low_hz: 20.0,
        high_hz: 20.0,
        include_low: true,
        include_high: false,
    }; 33];
    let mut idx = 0_usize;

    for j in -17..=2 {
        let lo = if j == -17 { 20.0 } else { edge(2 * j - 1) };
        let hi = edge(2 * j + 1);
        out[idx] = LtasInterval {
            low_hz: lo,
            high_hz: hi,
            include_low: true,
            include_high: false,
        };
        idx += 1;
    }

    out[idx] = LtasInterval {
        low_hz: edge(5),
        high_hz: 2_000.0,
        include_low: true,
        include_high: false,
    };
    idx += 1;
    out[idx] = LtasInterval {
        low_hz: 2_000.0,
        high_hz: edge(7),
        include_low: true,
        include_high: false,
    };
    idx += 1;

    for j in 4..=12 {
        out[idx] = LtasInterval {
            low_hz: edge(2 * j - 1),
            high_hz: edge(2 * j + 1),
            include_low: true,
            include_high: false,
        };
        idx += 1;
    }

    out[idx] = LtasInterval {
        low_hz: edge(25),
        high_hz: 18_000.0,
        include_low: true,
        include_high: true,
    };
    idx += 1;
    out[idx] = LtasInterval {
        low_hz: 18_000.0,
        high_hz: 20_000.0,
        include_low: false,
        include_high: true,
    };
    out
}

/// Returns the 33 geometric center frequencies of [`canonical_ltas_33_intervals_hz`].
#[must_use]
pub fn canonical_ltas_33_centers_hz() -> [f64; 33] {
    canonical_ltas_33_intervals_hz().map(LtasInterval::center_hz)
}

/// Precomputed Welch 33-band LTAS target spectrum and bin mapping for one channel.
#[derive(Clone, Debug)]
pub struct Welch33TargetCache {
    /// Total signal length expected.
    pub len: usize,
    /// Pre-roll warmup sample count.
    pub warmup: usize,
    /// Upper frequency limit in Hz (`min(0.48 * fs, 20_000)` or `20_000` for `fs >= 40 kHz`).
    pub max_freq_hz: f64,
    /// Cached target signal for exact residual `pred - target` power evaluation.
    pub target: Vec<f64>,
    /// Segment length per Welch window.
    pub seg_len: usize,
    /// Window start offsets relative to `warmup`.
    pub seg_starts: Vec<usize>,
    /// Symmetric Hann window vector (`seg_len`).
    pub win: Vec<f64>,
    /// Combined normalization factor `(1 / num_windows) * (1 / sum(win^2))`.
    pub power_scale: f64,
    /// One-sided bin to band tag (`0..33` or `33 = NYQ_STOPBAND_TAG` or `u8::MAX`).
    pub bin_band: Vec<u8>,
    /// Active band mask (`tgt_band_p[b] > leak_floor`).
    pub active: [bool; 33],
    /// Count of active bands.
    pub active_count: usize,
    /// Target band powers (`W`).
    pub tgt_band_p: [f64; 33],
    /// Target ultrasonic power above `max_freq_hz`.
    pub tgt_nyq_p: f64,
    /// Peak target band power across all 33 bands (`max_b tgt_band_p[b]`).
    pub peak_p: f64,
    /// Spectral-leakage exclusion floor (`peak_p * 1.0e-18`).
    pub leak_floor: f64,
    /// Numerical positivity floor for `log` evaluation.
    pub floor: f64,
    /// Per-band SNR attenuation weight `W_snr` (`1.0` when noise floor unavailable).
    pub band_snr_w: [f64; 33],
    /// FFT plan.
    plan: Radix2FftPlan,
}

impl Welch33TargetCache {
    /// Builds a Welch 33-band LTAS target summary on `target[warmup..]`.
    pub fn new(target: &[f64], warmup: usize, sample_rate_hz: f64) -> Result<Self, TrainError> {
        Self::new_with_noise(target, None, warmup, sample_rate_hz)
    }

    /// Builds a Welch 33-band LTAS target summary alongside optional noise-floor SNR band
    /// weights (`ref[qual.eval.thermal-hiss]`).
    pub fn new_with_noise(
        target: &[f64],
        noise: Option<&[f64]>,
        warmup: usize,
        sample_rate_hz: f64,
    ) -> Result<Self, TrainError> {
        if !(8_000.0..=1_536_000.0).contains(&sample_rate_hz) || target.len() <= warmup {
            return Err(TrainError::InvalidDimensions {
                context: "Welch33TargetCache::new",
                expected: warmup + 1,
                actual: target.len(),
            });
        }
        let len = target.len();
        if let Some(ns) = noise
            && (ns.len() < len || !ns[..len].iter().all(|v| v.is_finite()))
        {
            return Err(TrainError::InvalidDimensions {
                context: "Welch33TargetCache::new_with_noise",
                expected: len,
                actual: ns.len(),
            });
        }
        let len_post = len - warmup;
        let max_freq_hz = if sample_rate_hz >= 40_000.0 {
            20_000.0
        } else {
            (0.48 * sample_rate_hz).min(20_000.0)
        };
        let rate_scale = (sample_rate_hz / 48_000.0).ceil().max(1.0) as usize;
        let win_cap = 16_384_usize.saturating_mul(rate_scale.next_power_of_two());
        let n_fft = (win_cap * 2).max((sample_rate_hz.ceil() as usize).next_power_of_two());
        let (seg_len, seg_starts) = compute_welch_seg_starts(len_post, win_cap);
        let (win, inv_win_norm) = build_hann_window(seg_len);
        let intervals = canonical_ltas_33_intervals_hz();
        let bin_band = build_bin_band_map(n_fft, sample_rate_hz, max_freq_hz, &intervals);
        let plan = Radix2FftPlan::new(n_fft)?;
        let inv_wins = 1.0 / (seg_starts.len().max(1) as f64);
        let power_scale = inv_wins * inv_win_norm;

        let mut cache = Self {
            len,
            warmup,
            max_freq_hz,
            target: target.to_vec(),
            seg_len,
            seg_starts,
            win,
            power_scale,
            bin_band,
            active: [false; 33],
            active_count: 0,
            tgt_band_p: [0.0; 33],
            tgt_nyq_p: 0.0,
            peak_p: 0.0,
            leak_floor: 0.0,
            floor: 1.0e-24,
            band_snr_w: [1.0; 33],
            plan,
        };
        let (bp, np, _) = cache.accumulate_signal_powers(target, false);
        cache.tgt_band_p = bp;
        cache.tgt_nyq_p = np;
        let peak_p = bp.iter().copied().fold(0.0_f64, f64::max);
        let leak_floor = peak_p * 1.0e-18;
        cache.peak_p = peak_p;
        cache.leak_floor = leak_floor;
        for b in 0..33 {
            if intervals[b].high_hz <= max_freq_hz + 1.0e-9 && bp[b] > leak_floor {
                cache.active[b] = true;
                cache.active_count += 1;
            }
        }
        cache.floor = (peak_p * 1.0e-15).max(1.0e-24);
        if let Some(ns) = noise {
            let (noise_bp, _, _) = cache.accumulate_signal_powers(&ns[..len], false);
            for b in 0..33 {
                cache.band_snr_w[b] = snr_weight(bp[b], noise_bp[b]);
            }
        }
        Ok(cache)
    }

    fn accumulate_signal_powers(&self, sig: &[f64], retain_spectra: bool) -> WelchPowerResult {
        let n_fft = self.plan.n_fft();
        let half_p1 = n_fft / 2 + 1;
        let n_wins = self.seg_starts.len();
        let mut band_p = [0.0_f64; 33];
        let mut nyq_p = 0.0_f64;
        let mut win_specs = if retain_spectra {
            vec![(vec![0.0_f64; half_p1], vec![0.0_f64; half_p1]); n_wins]
        } else {
            Vec::new()
        };
        let mut ws = PairedFftWorkspace::new(n_fft);
        let (mut r0, mut i0, mut r1, mut i1) = (
            vec![0.0_f64; half_p1],
            vec![0.0_f64; half_p1],
            vec![0.0_f64; half_p1],
            vec![0.0_f64; half_p1],
        );

        let half = n_fft / 2;
        let mut w_idx = 0_usize;
        while w_idx < n_wins {
            let st0 = self.seg_starts[w_idx];
            let seg0 = &sig[self.warmup + st0..self.warmup + st0 + self.seg_len];
            let seg1 = (w_idx + 1 < n_wins).then(|| {
                let st1 = self.seg_starts[w_idx + 1];
                &sig[self.warmup + st1..self.warmup + st1 + self.seg_len]
            });
            paired_real_forward_spectra(
                &self.plan,
                seg0,
                seg1,
                &self.win,
                &mut ws,
                (&mut r0, &mut i0),
                Some((&mut r1, &mut i1)),
            );
            for (k, &tag) in self.bin_band.iter().enumerate() {
                let one_sided = if k == 0 || k == half { 1.0 } else { 2.0 };
                let p_sum = (r0[k] * r0[k] + i0[k] * i0[k])
                    + if seg1.is_some() {
                        r1[k] * r1[k] + i1[k] * i1[k]
                    } else {
                        0.0
                    };
                let c = p_sum * self.power_scale * one_sided;
                if (tag as usize) < 33 {
                    band_p[tag as usize] += c;
                } else if tag == NYQ_STOPBAND_TAG {
                    nyq_p += c;
                }
            }
            if retain_spectra {
                win_specs[w_idx].0.copy_from_slice(&r0);
                win_specs[w_idx].1.copy_from_slice(&i0);
                if w_idx + 1 < n_wins {
                    win_specs[w_idx + 1].0.copy_from_slice(&r1);
                    win_specs[w_idx + 1].1.copy_from_slice(&i1);
                }
            }
            w_idx += 2;
        }
        (band_p, nyq_p, win_specs)
    }
}

fn build_bin_band_map(
    n_fft: usize,
    sample_rate_hz: f64,
    max_freq_hz: f64,
    intervals: &[LtasInterval; 33],
) -> Vec<u8> {
    let half_p1 = n_fft / 2 + 1;
    let mut map = vec![u8::MAX; half_p1];
    let df = sample_rate_hz / (n_fft as f64);
    let stopband_lo_hz = if sample_rate_hz >= 48_000.0 {
        21_600.0
    } else {
        max_freq_hz
    };
    for (k, slot) in map.iter_mut().enumerate() {
        let f_hz = (k as f64) * df;
        if f_hz < 20.0 {
            continue;
        }
        if f_hz > max_freq_hz {
            if f_hz >= stopband_lo_hz {
                *slot = NYQ_STOPBAND_TAG;
            }
            continue;
        }
        for (b, iv) in intervals.iter().enumerate() {
            if iv.high_hz <= max_freq_hz + 1.0e-9 && iv.contains(f_hz) {
                *slot = b as u8;
                break;
            }
        }
    }
    map
}

/// Loss mode for differentiable 33-band Welch LTAS optimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LtasLossMode {
    /// Smooth pseudo-Huber squared dB error averaged across active bands.
    L2Huber,
    /// Soft C2 hinge penalizing deviations exceeding `max_db`.
    HingeMaxDb(f64),
    /// Log-sum-exp smooth minimax approximating `max_b |Delta_b|` over `goal_db`.
    LogSumExpMinimax {
        /// Softmax inverse temperature `beta`.
        beta: f64,
        /// Goal slack margin in dB.
        goal_db: f64,
    },
}

/// Normative 33-band noise-adjusted LTAS report (`ref[qual.companion.ltas]`).
#[derive(Clone, Debug, PartialEq)]
pub struct LtasChannelReport {
    /// Maximum absolute noise-adjusted band deviation `D^adj_{max,c}` (dB).
    pub max_adj_deviation_db: f64,
    /// Whether any excluded band failed the `+1.0 dB` residual guard `R <= 10^{0.1} F`.
    pub excluded_band_guard_failed: bool,
    /// Mask of bands excluded due to target energy at or below noise floor.
    pub excluded_bands: [bool; 33],
    /// Residual power per band for monotone non-regression checks.
    pub residual_band_power: [f64; 33],
}

/// Evaluates normative noise-adjusted 33-band LTAS deviation (`ref[qual.companion.ltas]`).
#[must_use]
pub fn compute_ltas_channel(
    cache: &Welch33TargetCache,
    pred: &[f64],
    noise: Option<&[f64]>,
) -> LtasChannelReport {
    if pred.len() != cache.len {
        return LtasChannelReport {
            max_adj_deviation_db: f64::INFINITY,
            excluded_band_guard_failed: true,
            excluded_bands: [true; 33],
            residual_band_power: [f64::INFINITY; 33],
        };
    }
    let any_nz_res = pred
        .iter()
        .zip(&cache.target)
        .any(|(&m, &t)| (m - t) != 0.0);
    let (pred_p, res_p) = if !any_nz_res {
        (cache.tgt_band_p, [0.0_f64; 33])
    } else {
        let res: Vec<f64> = pred
            .iter()
            .zip(&cache.target)
            .map(|(&p, &t)| p - t)
            .collect();
        (
            cache.accumulate_signal_powers(pred, false).0,
            cache.accumulate_signal_powers(&res, false).0,
        )
    };
    let noise_p = match noise {
        Some(ns) if ns.len() >= cache.len => cache.accumulate_signal_powers(ns, false).0,
        _ => [0.0; 33],
    };

    let intervals = canonical_ltas_33_intervals_hz();
    let mut excluded = [false; 33];
    let mut guard_failed = false;
    let mut max_dev = 0.0_f64;
    for b in 0..33 {
        if intervals[b].high_hz > cache.max_freq_hz + 1.0e-9 {
            excluded[b] = true;
            continue;
        }
        let (py, pm, pr, pf) = (cache.tgt_band_p[b], pred_p[b], res_p[b], noise_p[b]);
        if !py.is_finite() || !pm.is_finite() || !pr.is_finite() || !pf.is_finite() {
            guard_failed = true;
            max_dev = f64::INFINITY;
            continue;
        }
        if pf > 0.0 && py <= 2.0 * pf {
            excluded[b] = true;
            if pr > ONE_DB_POWER_FACTOR * pf {
                guard_failed = true;
                max_dev = f64::INFINITY;
            }
            continue;
        }
        if pf == 0.0 && py <= cache.leak_floor {
            excluded[b] = true;
            let leak_guard = ONE_DB_POWER_FACTOR * py.max(cache.leak_floor);
            if (cache.peak_p == 0.0 && (pr > 0.0 || any_nz_res))
                || (cache.peak_p > 0.0 && pr > leak_guard)
            {
                guard_failed = true;
                max_dev = f64::INFINITY;
            }
            continue;
        }
        let denom_adj = py - pf;
        if pm == 0.0 || denom_adj <= 0.0 {
            max_dev = f64::INFINITY;
        } else {
            let dev = (10.0 * (pm / denom_adj).log10()).abs();
            max_dev = max_dev.max(dev);
        }
    }
    LtasChannelReport {
        max_adj_deviation_db: max_dev,
        excluded_band_guard_failed: guard_failed,
        excluded_bands: excluded,
        residual_band_power: res_p,
    }
}

/// Evaluates differentiable 33-band Welch LTAS and ultrasonic `[20 kHz, fs/2]` hinge in a single
/// fused Welch forward + adjoint FFT pass, accumulating `w_ltas * d_ltas + w_nyquist * d_nyquist`
/// into `grad_opt` and returning `(unweighted_ltas, unweighted_nyquist)`.
pub fn eval_ltas_and_nyquist_fused(
    cache: &Welch33TargetCache,
    pred: &[f64],
    mode: LtasLossMode,
    w_ltas: f64,
    w_nyquist: f64,
    grad_opt: Option<&mut [f64]>,
) -> (f64, f64) {
    if pred.len() != cache.len || cache.seg_starts.is_empty() || (w_ltas <= 0.0 && w_nyquist <= 0.0)
    {
        return (0.0, 0.0);
    }
    let need_grad = grad_opt.is_some();
    let (band_p, nyq_p, win_specs) = cache.accumulate_signal_powers(pred, need_grad);
    let (ltas_loss, mut d_band) = if w_ltas > 0.0 {
        eval_ltas_mode_terms(cache, &band_p, mode, need_grad)
    } else {
        (0.0, [0.0; 33])
    };
    let (nyq_loss, d_nyq) = if w_nyquist > 0.0 {
        nyquist_hinge_loss_and_deriv(cache, nyq_p, 1.0)
    } else {
        (0.0, 0.0)
    };
    if let Some(grad) = grad_opt {
        for d in &mut d_band {
            *d *= w_ltas.max(0.0);
        }
        backprop_welch33_adjoint(cache, &win_specs, &d_band, w_nyquist.max(0.0) * d_nyq, grad);
    }
    (ltas_loss, nyq_loss)
}

/// Evaluates differentiable 33-band Welch LTAS loss (+ optional ultrasonic `[20 kHz, fs/2]` hinge)
/// and accumulates exact reverse-mode FFT adjoint into `grad_opt`.
pub fn eval_ltas_loss_and_adjoint(
    cache: &Welch33TargetCache,
    pred: &[f64],
    mode: LtasLossMode,
    nyquist_hinge_weight: f64,
    grad_opt: Option<&mut [f64]>,
) -> f64 {
    let w_nyq = nyquist_hinge_weight.max(0.0);
    let (ltas_l, nyq_l) = eval_ltas_and_nyquist_fused(cache, pred, mode, 1.0, w_nyq, grad_opt);
    ltas_l + w_nyq * nyq_l
}

/// Evaluates purely the ultrasonic `[20 kHz, fs/2]` excess-energy stopband hinge and its FFT adjoint.
pub fn eval_nyquist_hinge_and_adjoint(
    cache: &Welch33TargetCache,
    pred: &[f64],
    weight: f64,
    grad_opt: Option<&mut [f64]>,
) -> f64 {
    let w = weight.max(0.0);
    if w <= 0.0 {
        return 0.0;
    }
    let (_, nyq_l) =
        eval_ltas_and_nyquist_fused(cache, pred, LtasLossMode::L2Huber, 0.0, w, grad_opt);
    w * nyq_l
}

fn backprop_welch33_adjoint(
    cache: &Welch33TargetCache,
    win_specs: &[(Vec<f64>, Vec<f64>)],
    d_band: &[f64; 33],
    d_nyq: f64,
    grad: &mut [f64],
) {
    let n_fft = cache.plan.n_fft();
    let half = n_fft / 2;
    let half_p1 = half + 1;
    let n_wins = win_specs.len();
    let mut ws = PairedFftWorkspace::new(n_fft);
    let (mut g0_re, mut g0_im, mut g1_re, mut g1_im) = (
        vec![0.0_f64; half_p1],
        vec![0.0_f64; half_p1],
        vec![0.0_f64; half_p1],
        vec![0.0_f64; half_p1],
    );

    let mut w_idx = 0_usize;
    while w_idx < n_wins {
        let has1 = w_idx + 1 < n_wins;
        for (k, &tag) in cache.bin_band.iter().enumerate() {
            let dw = if (tag as usize) < 33 && cache.active[tag as usize] {
                d_band[tag as usize]
            } else if tag == NYQ_STOPBAND_TAG {
                d_nyq
            } else {
                0.0
            };
            if dw == 0.0 {
                g0_re[k] = 0.0;
                g0_im[k] = 0.0;
                g1_re[k] = 0.0;
                g1_im[k] = 0.0;
                continue;
            }
            let one_sided = if k == 0 || k == half { 1.0 } else { 2.0 };
            let c = 2.0 * cache.power_scale * one_sided * dw;
            g0_re[k] = c * win_specs[w_idx].0[k];
            g0_im[k] = c * win_specs[w_idx].1[k];
            if has1 {
                g1_re[k] = c * win_specs[w_idx + 1].0[k];
                g1_im[k] = c * win_specs[w_idx + 1].1[k];
            }
        }
        let b0 = cache.warmup + cache.seg_starts[w_idx];
        let seg_len = cache.seg_len;
        if has1 {
            let b1 = cache.warmup + cache.seg_starts[w_idx + 1];
            let (mut dy0, mut dy1) = (vec![0.0_f64; seg_len], vec![0.0_f64; seg_len]);
            paired_real_backward_adjoint(
                &cache.plan,
                &cache.win,
                (&g0_re, &g0_im),
                Some((&g1_re, &g1_im)),
                &mut ws,
                &mut dy0,
                Some(&mut dy1),
            );
            for (g, d) in grad[b0..b0 + seg_len].iter_mut().zip(&dy0) {
                *g += *d;
            }
            for (g, d) in grad[b1..b1 + seg_len].iter_mut().zip(&dy1) {
                *g += *d;
            }
        } else {
            let mut dy0 = vec![0.0_f64; seg_len];
            paired_real_backward_adjoint(
                &cache.plan,
                &cache.win,
                (&g0_re, &g0_im),
                None,
                &mut ws,
                &mut dy0,
                None,
            );
            for (g, d) in grad[b0..b0 + seg_len].iter_mut().zip(&dy0) {
                *g += *d;
            }
        }
        w_idx += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltas_33_intervals_metric_and_all_three_modes_fd_adjoints() {
        let ivs = canonical_ltas_33_intervals_hz();
        let ctrs = canonical_ltas_33_centers_hz();
        assert_eq!(ivs[0].low_hz, 20.0);
        assert_eq!(ivs[20].high_hz, 2_000.0);
        assert_eq!(ivs[21].low_hz, 2_000.0);
        assert_eq!(ivs[31].high_hz, 18_000.0);
        assert_eq!(ivs[32].high_hz, 20_000.0);
        assert!(ctrs[32] > 18_000.0 && ctrs[32] < 20_000.0);

        let n = 1_600_usize;
        let sr = 48_000.0_f64;
        let tau = core::f64::consts::TAU;
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                let mut s = 0.02 * (tau * 22_200.0 * t).sin();
                for (b, &fc) in ctrs.iter().enumerate() {
                    s += 0.08 * (tau * fc * t + 0.43 * (b as f64)).sin();
                }
                s
            })
            .collect();
        let pred: Vec<f64> = target
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let t = (i as f64) / sr;
                0.85 * x + 0.22 * (tau * 22_200.0 * t + 0.5).sin()
            })
            .collect();
        let cache = Welch33TargetCache::new(&target, 64, sr).unwrap();
        assert_eq!(cache.active_count, 33);
        let rep_eq = compute_ltas_channel(&cache, &target, None);
        assert!(!rep_eq.excluded_band_guard_failed && rep_eq.max_adj_deviation_db < 1.0e-12);
        let rep = compute_ltas_channel(&cache, &pred, None);
        assert!(!rep.excluded_band_guard_failed && rep.max_adj_deviation_db.is_finite());

        // Excluded-band guard must trigger when residual > 10^{0.1} * noise power on py <= 2 * pf.
        let loud_noise: Vec<f64> = target.iter().map(|&v| 1.2 * v).collect();
        let bad_pred: Vec<f64> = target.iter().map(|&v| v * 3.0).collect();
        let rep_excl = compute_ltas_channel(&cache, &bad_pred, Some(&loud_noise));
        assert!(rep_excl.excluded_band_guard_failed && rep_excl.max_adj_deviation_db.is_infinite());

        let eps = 1.0e-5_f64;
        for mode in [
            LtasLossMode::L2Huber,
            LtasLossMode::HingeMaxDb(0.5),
            LtasLossMode::LogSumExpMinimax {
                beta: 4.0,
                goal_db: 0.25,
            },
        ] {
            let mut grad = vec![0.0_f64; n];
            let l_val = eval_ltas_loss_and_adjoint(&cache, &pred, mode, 0.5, Some(&mut grad));
            assert!(l_val.is_finite() && l_val > 0.0);
            for idx in [120, 480, 990] {
                let mut pp = pred.clone();
                let mut pm = pred.clone();
                pp[idx] += eps;
                pm[idx] -= eps;
                let fd = (eval_ltas_loss_and_adjoint(&cache, &pp, mode, 0.5, None)
                    - eval_ltas_loss_and_adjoint(&cache, &pm, mode, 0.5, None))
                    / (2.0 * eps);
                assert!(
                    (grad[idx] - fd).abs() <= 1.0e-5 * grad[idx].abs().max(1.0e-4),
                    "LTAS {mode:?} idx={idx} ana={} fd={fd}",
                    grad[idx]
                );
            }
        }

        let mut nyq_g = vec![0.0_f64; n];
        let nyq_l = eval_nyquist_hinge_and_adjoint(&cache, &pred, 1.0, Some(&mut nyq_g));
        assert!(nyq_l > 0.0);
        let mut pp = pred.clone();
        let mut pm = pred.clone();
        pp[500] += eps;
        pm[500] -= eps;
        let fd_nyq = (eval_nyquist_hinge_and_adjoint(&cache, &pp, 1.0, None)
            - eval_nyquist_hinge_and_adjoint(&cache, &pm, 1.0, None))
            / (2.0 * eps);
        assert!(
            (nyq_g[500] - fd_nyq).abs() <= 1.0e-5 * nyq_g[500].abs().max(1.0e-4),
            "nyq ana={} fd={fd_nyq}",
            nyq_g[500]
        );

        // Noisy fused LTAS + UltrasonicHinge matches separate weighted grads and central FD.
        let noise_cache =
            Welch33TargetCache::new_with_noise(&target, Some(&loud_noise), 64, sr).unwrap();
        let mut gf = vec![0.0_f64; n];
        let (vl, vn) = eval_ltas_and_nyquist_fused(
            &noise_cache,
            &pred,
            LtasLossMode::LogSumExpMinimax {
                beta: 4.0,
                goal_db: 0.2,
            },
            0.1,
            0.25,
            Some(&mut gf),
        );
        let comb = 0.1 * vl + 0.25 * vn;
        assert!(comb.is_finite());
        pp = pred.clone();
        pm = pred.clone();
        pp[500] += eps;
        pm[500] -= eps;
        let (vp_l, vp_n) = eval_ltas_and_nyquist_fused(
            &noise_cache,
            &pp,
            LtasLossMode::LogSumExpMinimax {
                beta: 4.0,
                goal_db: 0.2,
            },
            0.1,
            0.25,
            None,
        );
        let (vm_l, vm_n) = eval_ltas_and_nyquist_fused(
            &noise_cache,
            &pm,
            LtasLossMode::LogSumExpMinimax {
                beta: 4.0,
                goal_db: 0.2,
            },
            0.1,
            0.25,
            None,
        );
        let fd_fused = ((0.1 * vp_l + 0.25 * vp_n) - (0.1 * vm_l + 0.25 * vm_n)) / (2.0 * eps);
        assert!(
            (gf[500] - fd_fused).abs() <= 1.0e-5 * gf[500].abs().max(1.0e-4),
            "fused noisy ana={} fd={fd_fused}",
            gf[500]
        );
    }
}
