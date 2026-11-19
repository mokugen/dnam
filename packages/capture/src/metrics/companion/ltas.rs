#![forbid(unsafe_code)]
//! 33-Band Base-10 1/3-Octave Noise-Adjusted Long-Term Average Spectrum (`LTAS`, `r[qual.companion.ltas]`).

use std::f64::consts::PI;

use super::{MAX_METRIC_SAMPLE_RATE_HZ, ONE_DB_POWER_FACTOR};

/// Single interval in the canonical 33-band base-10 1/3-octave LTAS partition (`ref[qual.companion.ltas]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LtasInterval {
    /// Lower edge in Hz (`binary64` rounded once).
    pub low_hz: f64,
    /// Upper edge in Hz (`binary64` rounded once).
    pub high_hz: f64,
    /// Whether the lower boundary is inclusive (`true` for `[low, ...)`, `false` for `(18000, 20000]`).
    pub include_low: bool,
    /// Whether the upper boundary is inclusive (`true` only at `18,000 Hz` and `20,000 Hz`).
    pub include_high: bool,
}

impl LtasInterval {
    /// Geometric center frequency $\sqrt{f_{\text{low}} f_{\text{high}}}$ in Hz.
    #[inline]
    #[must_use]
    pub fn center_hz(self) -> f64 {
        (self.low_hz * self.high_hz).sqrt()
    }

    /// Tests whether frequency `f_hz` belongs to this interval under `r[qual.companion.ltas]` bullet 1.
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

/// Returns the exact 33-band base-10 1/3-octave partition over `[20 Hz, 20,000 Hz]` anchored at `1,000 Hz`:
/// - `21` Sub/Low-Mid intervals over `[20 Hz, 2,000 Hz)` ($j = -17 \dots 3$, clipped at `20 Hz`, split at `2,000 Hz`)
/// - `11` HF Diagnostic intervals over `[2,000 Hz, 18,000 Hz]` ($j = 3 \dots 13$, `18,000 Hz` inclusive)
/// - `1` Upper Air interval over `(18,000 Hz, 20,000 Hz]` ($j = 13$ above `18 kHz`, `20,000 Hz` inclusive)
///   (`ref[qual.companion.ltas]`).
#[must_use]
pub fn canonical_ltas_33_intervals_hz() -> [LtasInterval; 33] {
    let edge = |j2_plus: i32| -> f64 { 1000.0 * 10.0_f64.powf(f64::from(j2_plus) / 20.0) };
    let mut out = [LtasInterval {
        low_hz: 20.0,
        high_hz: 20.0,
        include_low: true,
        include_high: false,
    }; 33];
    let mut idx = 0usize;

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

    let e5 = edge(5);
    let e7 = edge(7);
    out[idx] = LtasInterval {
        low_hz: e5,
        high_hz: 2_000.0,
        include_low: true,
        include_high: false,
    };
    idx += 1;
    out[idx] = LtasInterval {
        low_hz: 2_000.0,
        high_hz: e7,
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

    let e25 = edge(25);
    out[idx] = LtasInterval {
        low_hz: e25,
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
    debug_assert_eq!(idx + 1, 33);
    out
}

/// Returns the 33 geometric center frequencies (in Hz) of [`canonical_ltas_33_intervals_hz`].
#[must_use]
pub fn canonical_ltas_33_centers_hz() -> [f64; 33] {
    let intervals = canonical_ltas_33_intervals_hz();
    let mut centers = [0.0_f64; 33];
    for (dst, iv) in centers.iter_mut().zip(&intervals) {
        *dst = iv.center_hz();
    }
    centers
}

/// Complete 33-band LTAS evaluation report (`ref[qual.companion.ltas]`).
#[derive(Clone, Debug)]
pub struct LtasEvaluationReport {
    /// Whether the evaluation met preconditions (`Nyquist >= 20 kHz`, non-empty finite signals, valid guard).
    pub available: bool,
    /// Whether any excluded noise-limited band violated the `+1.0 dB` residual guard ($R^{\text{LTAS}} \le 10^{0.1} F^{\text{LTAS}}$).
    pub excluded_band_guard_failed: bool,
    /// Maximum absolute noise-adjusted band deviation $D^{\text{adj}}_{\max}$ in dB (`+inf` on failure).
    pub max_adj_deviation_db: f64,
    /// Unadjusted maximum band deviation in dB across active bands (reported separately per §3).
    pub max_unadjusted_deviation_db: f64,
    /// Number of active (non-excluded) bands evaluated.
    pub active_band_count: usize,
    /// Which of the 33 bands were excluded under the noise-limited band exception.
    pub excluded_bands: [bool; 33],
    /// Raw residual band power $R^{\text{LTAS}}_{b}$ for all 33 bands (used for adjacent-tier excluded-band non-regression).
    pub residual_band_power: [f64; 33],
}

#[derive(Clone, Debug)]
struct WelchPlan {
    n_fft: usize,
    seg_len: usize,
    inv_win_norm: f64,
    inv_wins: f64,
    win: Vec<f64>,
    bitrev_seg: Vec<usize>,
    tw_re: Vec<f64>,
    tw_im: Vec<f64>,
    seg_starts: Vec<usize>,
    k_min: usize,
    bin_band: Vec<u8>,
}

impl WelchPlan {
    fn new(
        sig_len: usize,
        sample_rate_hz: f64,
        max_freq_hz: f64,
        intervals: &[LtasInterval; 33],
    ) -> Self {
        let rate_scale = (sample_rate_hz / 48_000.0).ceil().max(1.0) as usize;
        let win_cap = 16_384usize.saturating_mul(rate_scale.next_power_of_two());
        let n_fft = (win_cap * 2).max((sample_rate_hz.ceil() as usize).next_power_of_two());
        let seg_len = sig_len.min(win_cap);
        let hop = (seg_len / 2).max(1);

        let mut win = vec![1.0_f64; seg_len];
        let mut win_norm = seg_len as f64;
        if seg_len >= 8 {
            win_norm = 0.0;
            for (i, w) in win.iter_mut().enumerate() {
                let v = 0.5 * (1.0 - (2.0 * PI * (i as f64) / ((seg_len - 1) as f64)).cos());
                *w = v;
                win_norm += v * v;
            }
        }
        let inv_win_norm = 1.0 / win_norm.max(1.0e-12);

        let log2_n = n_fft.trailing_zeros();
        let shift = usize::BITS - log2_n;
        let bitrev_seg: Vec<usize> = (0..seg_len).map(|i| i.reverse_bits() >> shift).collect();

        let half_fft = n_fft >> 1;
        let mut tw_re = Vec::with_capacity(half_fft);
        let mut tw_im = Vec::with_capacity(half_fft);
        let base_ang = -2.0 * PI / (n_fft as f64);
        for m in 0..half_fft {
            let (s, c) = (base_ang * (m as f64)).sin_cos();
            tw_re.push(c);
            tw_im.push(s);
        }

        let mut seg_starts = Vec::new();
        let mut start = 0usize;
        while start + seg_len <= sig_len {
            seg_starts.push(start);
            start += hop;
        }
        let last_start = sig_len - seg_len;
        if last_start + hop != start {
            seg_starts.push(last_start);
        }
        let inv_wins = 1.0 / (seg_starts.len() as f64);

        let df = sample_rate_hz / (n_fft as f64);
        let mut k_min = half_fft + 1;
        let mut k_max = 0usize;
        for k in 0..=half_fft {
            let f_hz = (k as f64) * df;
            if f_hz < 20.0 {
                continue;
            }
            if f_hz > max_freq_hz {
                break;
            }
            k_min = k_min.min(k);
            k_max = k_max.max(k);
        }
        let mut bin_band = Vec::new();
        if k_min <= k_max {
            bin_band.resize(k_max - k_min + 1, u8::MAX);
            for k in k_min..=k_max {
                let f_hz = (k as f64) * df;
                for (b, iv) in intervals.iter().enumerate() {
                    if iv.high_hz <= max_freq_hz + 1.0e-9 && iv.contains(f_hz) {
                        bin_band[k - k_min] = b as u8;
                        break;
                    }
                }
            }
        }

        Self {
            n_fft,
            seg_len,
            inv_win_norm,
            inv_wins,
            win,
            bitrev_seg,
            tw_re,
            tw_im,
            seg_starts,
            k_min,
            bin_band,
        }
    }

    fn fft_dit_prescattered(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n_fft;
        for start in (0..n).step_by(2) {
            let (ur, ui) = (re[start], im[start]);
            let (vr, vi) = (re[start + 1], im[start + 1]);
            re[start] = ur + vr;
            im[start] = ui + vi;
            re[start + 1] = ur - vr;
            im[start + 1] = ui - vi;
        }
        if n >= 4 {
            for start in (0..n).step_by(4) {
                let (u0r, u0i) = (re[start], im[start]);
                let (v0r, v0i) = (re[start + 2], im[start + 2]);
                re[start] = u0r + v0r;
                im[start] = u0i + v0i;
                re[start + 2] = u0r - v0r;
                im[start + 2] = u0i - v0i;

                let (u1r, u1i) = (re[start + 1], im[start + 1]);
                let (v1r, v1i) = (im[start + 3], -re[start + 3]);
                re[start + 1] = u1r + v1r;
                im[start + 1] = u1i + v1i;
                re[start + 3] = u1r - v1r;
                im[start + 3] = u1i - v1i;
            }
        }
        let mut len = 8usize;
        while len <= n {
            let half = len >> 1;
            let step = n / len;
            let mut start = 0usize;
            while start < n {
                for k in 0..half {
                    let m = k * step;
                    let (wr, wi) = (self.tw_re[m], self.tw_im[m]);
                    let a = start + k;
                    let b = a + half;
                    let (ur, ui) = (re[a], im[a]);
                    let (br, bi) = (re[b], im[b]);
                    let vr = br * wr - bi * wi;
                    let vi = br * wi + bi * wr;
                    re[a] = ur + vr;
                    im[a] = ui + vi;
                    re[b] = ur - vr;
                    im[b] = ui - vi;
                }
                start += len;
            }
            len <<= 1;
        }
    }

    fn accum_paired_real(
        &self,
        sig_x: &[f64],
        sig_y: Option<&[f64]>,
        diff_y_from_x: bool,
        re: &mut [f64],
        im: &mut [f64],
    ) -> ([f64; 33], [f64; 33]) {
        let mut p_x = [0.0_f64; 33];
        let mut p_y = [0.0_f64; 33];
        if self.bin_band.is_empty() {
            return (p_x, p_y);
        }
        let n = self.n_fft;
        let mask = n - 1;
        for &seg_start in &self.seg_starts {
            re.fill(0.0);
            im.fill(0.0);
            match sig_y {
                None => {
                    for i in 0..self.seg_len {
                        re[self.bitrev_seg[i]] = sig_x[seg_start + i] * self.win[i];
                    }
                }
                Some(sy) if diff_y_from_x => {
                    for i in 0..self.seg_len {
                        let r = self.bitrev_seg[i];
                        let w = self.win[i];
                        let vx = sig_x[seg_start + i];
                        re[r] = vx * w;
                        im[r] = (vx - sy[seg_start + i]) * w;
                    }
                }
                Some(sy) => {
                    for i in 0..self.seg_len {
                        let r = self.bitrev_seg[i];
                        let w = self.win[i];
                        re[r] = sig_x[seg_start + i] * w;
                        im[r] = sy[seg_start + i] * w;
                    }
                }
            }
            self.fft_dit_prescattered(re, im);
            for (idx, &band) in self.bin_band.iter().enumerate() {
                if band >= 33 {
                    continue;
                }
                let k = self.k_min + idx;
                let k_neg = (n - k) & mask;
                let mult = if k == 0 || k * 2 == n { 1.0 } else { 2.0 };
                let (zk_re, zk_im) = (re[k], im[k]);
                if sig_y.is_none() {
                    let mag_sq = (zk_re * zk_re + zk_im * zk_im) * self.inv_win_norm;
                    p_x[usize::from(band)] += mult * mag_sq;
                } else {
                    let (zn_re, zn_im) = (re[k_neg], im[k_neg]);
                    let xr2 = zk_re + zn_re;
                    let xi2 = zk_im - zn_im;
                    let yr2 = zk_im + zn_im;
                    let yi2 = zn_re - zk_re;
                    let mx = 0.25 * (xr2 * xr2 + xi2 * xi2) * self.inv_win_norm;
                    let my = 0.25 * (yr2 * yr2 + yi2 * yi2) * self.inv_win_norm;
                    let b_idx = usize::from(band);
                    p_x[b_idx] += mult * mx;
                    p_y[b_idx] += mult * my;
                }
            }
        }
        for b in 0..33 {
            p_x[b] *= self.inv_wins;
            p_y[b] *= self.inv_wins;
        }
        (p_x, p_y)
    }
}

/// Reusable target-invariant 33-band `LTAS` evaluator (`noise = None`).
#[derive(Clone, Debug)]
pub(crate) struct LtasTargetCache {
    target: Vec<f64>,
    plan: WelchPlan,
    intervals: [LtasInterval; 33],
    max_freq_hz: f64,
    p_y: [f64; 33],
    peak_band_p: f64,
    fft_leak_floor: f64,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl LtasTargetCache {
    #[must_use]
    pub(crate) fn new(target: &[f64], sample_rate_hz: f64, max_freq_hz: f64) -> Option<Self> {
        if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
            || !max_freq_hz.is_finite()
            || max_freq_hz < 20.0
            || sample_rate_hz < 2.0 * max_freq_hz
            || target.is_empty()
            || !target.iter().all(|v| v.is_finite())
        {
            return None;
        }
        let intervals = canonical_ltas_33_intervals_hz();
        let plan = WelchPlan::new(target.len(), sample_rate_hz, max_freq_hz, &intervals);
        let mut re = vec![0.0_f64; plan.n_fft];
        let mut im = vec![0.0_f64; plan.n_fft];
        let (p_y, _) = plan.accum_paired_real(target, None, false, &mut re, &mut im);
        let peak_band_p = p_y.iter().copied().fold(0.0_f64, f64::max);
        let fft_leak_floor = peak_band_p * 1.0e-18;
        Some(Self {
            target: target.to_vec(),
            plan,
            intervals,
            max_freq_hz,
            p_y,
            peak_band_p,
            fft_leak_floor,
            re,
            im,
        })
    }

    #[must_use]
    pub(crate) fn band_powers(&mut self, model: &[f64]) -> Option<([f64; 33], [f64; 33])> {
        if model.len() != self.target.len() || !model.iter().all(|v| v.is_finite()) {
            return None;
        }
        let any_nz_res = model
            .iter()
            .zip(&self.target)
            .any(|(&m, &t)| (m - t) != 0.0);
        if !any_nz_res {
            return Some((self.p_y, self.p_y));
        }
        let (p_m, _) = self
            .plan
            .accum_paired_real(model, None, false, &mut self.re, &mut self.im);
        Some((self.p_y, p_m))
    }

    #[must_use]
    pub(crate) fn eval_model(&mut self, model: &[f64]) -> LtasEvaluationReport {
        if model.len() != self.target.len() || !model.iter().all(|v| v.is_finite()) {
            return fail_unavailable_ltas();
        }
        let any_nz_res = model
            .iter()
            .zip(&self.target)
            .any(|(&m, &t)| (m - t) != 0.0);
        let (p_m, p_r) = if !any_nz_res {
            (self.p_y, [0.0_f64; 33])
        } else {
            self.plan
                .accum_paired_real(model, Some(&self.target), true, &mut self.re, &mut self.im)
        };
        finalize_ltas_report(
            &self.intervals,
            self.max_freq_hz,
            &self.p_y,
            &p_m,
            &p_r,
            &[0.0_f64; 33],
            self.peak_band_p,
            self.fft_leak_floor,
            any_nz_res,
        )
    }
}

fn fail_unavailable_ltas() -> LtasEvaluationReport {
    LtasEvaluationReport {
        available: false,
        excluded_band_guard_failed: false,
        max_adj_deviation_db: f64::INFINITY,
        max_unadjusted_deviation_db: f64::INFINITY,
        active_band_count: 0,
        excluded_bands: [false; 33],
        residual_band_power: [0.0; 33],
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_ltas_report(
    intervals: &[LtasInterval; 33],
    max_freq_hz: f64,
    p_y: &[f64; 33],
    p_m: &[f64; 33],
    p_r: &[f64; 33],
    p_f: &[f64; 33],
    peak_band_p: f64,
    fft_leak_floor: f64,
    any_nz_res: bool,
) -> LtasEvaluationReport {
    let mut max_adj_db = 0.0_f64;
    let mut max_raw_db = 0.0_f64;
    let mut active_count = 0usize;
    let mut excluded_bands = [false; 33];
    let mut guard_failed = false;

    for b in 0..33 {
        if intervals[b].high_hz > max_freq_hz + 1.0e-9 {
            excluded_bands[b] = true;
            continue;
        }
        let (py, pm, pr, pf) = (p_y[b], p_m[b], p_r[b], p_f[b]);
        if !py.is_finite() || !pm.is_finite() || !pr.is_finite() || !pf.is_finite() {
            guard_failed = true;
            max_adj_db = f64::INFINITY;
            max_raw_db = f64::INFINITY;
            continue;
        }

        if pf > 0.0 && py <= 2.0 * pf {
            excluded_bands[b] = true;
            if pr > ONE_DB_POWER_FACTOR * pf {
                guard_failed = true;
                max_adj_db = f64::INFINITY;
            }
            continue;
        }
        if pf == 0.0 && py <= fft_leak_floor {
            excluded_bands[b] = true;
            let leak_guard = ONE_DB_POWER_FACTOR * py.max(fft_leak_floor);
            if (peak_band_p == 0.0 && (pr > 0.0 || any_nz_res))
                || (peak_band_p > 0.0 && pr > leak_guard)
            {
                guard_failed = true;
                max_adj_db = f64::INFINITY;
            }
            continue;
        }

        active_count += 1;
        let denom_adj = py - pf;
        if pm == 0.0 {
            max_adj_db = f64::INFINITY;
            max_raw_db = f64::INFINITY;
        } else {
            let adj_db = (10.0 * (pm / denom_adj).log10()).abs();
            let raw_db = (10.0 * (pm / py).log10()).abs();
            if !adj_db.is_finite() || !raw_db.is_finite() {
                guard_failed = true;
                max_adj_db = f64::INFINITY;
                max_raw_db = f64::INFINITY;
            } else {
                max_adj_db = max_adj_db.max(adj_db);
                max_raw_db = max_raw_db.max(raw_db);
            }
        }
    }

    let available = active_count > 0 && !guard_failed;
    if !available {
        max_adj_db = f64::INFINITY;
        max_raw_db = f64::INFINITY;
    }
    LtasEvaluationReport {
        available,
        excluded_band_guard_failed: guard_failed,
        max_adj_deviation_db: max_adj_db,
        max_unadjusted_deviation_db: max_raw_db,
        active_band_count: active_count,
        excluded_bands,
        residual_band_power: *p_r,
    }
}

/// Evaluates the full 33-Band Base-10 1/3-Octave Noise-Adjusted LTAS report (`ref[qual.companion.ltas]`).
///
/// Enforces `Nyquist >= 20,000 Hz` (`sample_rate_hz >= 40,000 Hz`) when `require_full_20khz` is `true`;
/// when `require_full_20khz` is `false`, evaluates all canonical intervals below `0.48 * sample_rate_hz`
/// so sub-`40 kHz` vintage digital devices can still inspect passband spectral tilt.
#[must_use]
pub fn evaluate_ltas_33band_report(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    require_full_20khz: bool,
) -> LtasEvaluationReport {
    let min_sr = if require_full_20khz {
        40_000.0
    } else {
        8_000.0
    };
    if !(min_sr..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz) {
        return fail_unavailable_ltas();
    }
    let max_freq_hz = if require_full_20khz {
        20_000.0
    } else {
        (0.48 * sample_rate_hz).min(20_000.0)
    };
    evaluate_ltas_33band_report_with_max_freq(target, model, noise, sample_rate_hz, max_freq_hz)
}

/// Evaluates the 33-band Base-10 1/3-Octave Noise-Adjusted LTAS report up to `max_freq_hz`.
#[must_use]
pub fn evaluate_ltas_33band_report_with_max_freq(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    max_freq_hz: f64,
) -> LtasEvaluationReport {
    if !(8_000.0..=MAX_METRIC_SAMPLE_RATE_HZ).contains(&sample_rate_hz)
        || !max_freq_hz.is_finite()
        || max_freq_hz < 20.0
        || sample_rate_hz < 2.0 * max_freq_hz
        || target.is_empty()
        || target.len() != model.len()
        || !target.iter().all(|v| v.is_finite())
        || !model.iter().all(|v| v.is_finite())
    {
        return fail_unavailable_ltas();
    }
    let Some(ns) = noise else {
        return LtasTargetCache::new(target, sample_rate_hz, max_freq_hz)
            .map_or_else(fail_unavailable_ltas, |mut c| c.eval_model(model));
    };

    let n = target.len();
    if ns.len() < n || !ns[..n].iter().all(|v| v.is_finite()) {
        return fail_unavailable_ltas();
    }

    let intervals = canonical_ltas_33_intervals_hz();
    let plan = WelchPlan::new(n, sample_rate_hz, max_freq_hz, &intervals);
    let mut re = vec![0.0_f64; plan.n_fft];
    let mut im = vec![0.0_f64; plan.n_fft];

    let (p_y, p_m) = plan.accum_paired_real(target, Some(model), false, &mut re, &mut im);
    let residual: Vec<f64> = model
        .iter()
        .zip(target.iter())
        .map(|(m, t)| m - t)
        .collect();
    let (p_r, p_f) = plan.accum_paired_real(&residual, Some(&ns[..n]), false, &mut re, &mut im);
    if ns[..n].iter().any(|&v| v != 0.0) && p_f.iter().all(|&v| v == 0.0) {
        return fail_unavailable_ltas();
    }

    let peak_band_p = p_y.iter().copied().fold(0.0_f64, f64::max);
    let fft_leak_floor = peak_band_p * 1.0e-18;
    let any_nz_res = residual.iter().any(|&r| r != 0.0);

    finalize_ltas_report(
        &intervals,
        max_freq_hz,
        &p_y,
        &p_m,
        &p_r,
        &p_f,
        peak_band_p,
        fft_leak_floor,
        any_nz_res,
    )
}

/// Computes `(max_adj_deviation_db, active_band_count)` for the canonical 33-band LTAS (`Nyquist >= 20 kHz`,
/// `ref[qual.companion.ltas]`).
#[must_use]
pub fn compute_ltas_33band_max_deviation_db(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
) -> (f64, usize) {
    let rep = evaluate_ltas_33band_report(target, model, noise, sample_rate_hz, true);
    (rep.max_adj_deviation_db, rep.active_band_count)
}
