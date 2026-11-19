#![forbid(unsafe_code)]
//! Data-Driven `(x, y)` Spectral Split, Crossover & Linear Pole Identification (`spectral_id`).
//!
//! Estimates unknown multi-band crossover/split frequencies, upper bandwidth roll-off, and
//! Stage III Kautz resonances strictly from `(in_l/r, tgt_l/r)` audio waveforms with zero
//! hardcoded fixture priors.

use std::f64::consts::{PI, SQRT_2, TAU};

use dnam_engine::stages::V3MatryoshkaModel;

use super::super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};
use super::super::omp::select_omp_kautz_poles_matryoshka;
use crate::stimulus::fft::{Complex64, fft_radix2_in_place};

const STFT_WIN: usize = 2_048;
const STFT_HOP: usize = 512;
const N_LOG_BANDS: usize = 20;
const F_MIN_HZ: f64 = 45.0;
const F_MAX_HZ: f64 = 13_500.0;

/// Empirically estimated split frequencies and dominant linear poles extracted from `(x, y)`.
#[derive(Clone, Copy, Debug)]
pub(super) struct SpectralSplitEstimate {
    pub(super) f_biamp_hz: f64,
    pub(super) f_lo_hz: f64,
    pub(super) f_hi_hz: f64,
    pub(super) f_top_hz: f64,
    pub(super) emp_poles: [(f64, f64); 4],
}

pub(super) fn butterworth2_pole_rt(fc_hz: f64, fs_hz: f64) -> (f64, f64) {
    let f = fc_hz.clamp(12.0, 0.45 * fs_hz.max(8_000.0));
    let sigma = TAU * f / (fs_hz.max(8_000.0) * SQRT_2);
    (
        (-sigma).exp().clamp(0.05, 0.9994),
        sigma.clamp(0.0, PI - 1e-4),
    )
}

fn default_emp_poles(sr_hz: f64) -> [(f64, f64); 4] {
    [
        butterworth2_pole_rt(120.0, sr_hz),
        butterworth2_pole_rt(600.0, sr_hz),
        butterworth2_pole_rt(2_400.0, sr_hz),
        butterworth2_pole_rt(7_200.0, sr_hz),
    ]
}

fn estimate_emp_poles(dataset: &V3TrainDataset, mono_out: bool, sr_hz: f64) -> [(f64, f64); 4] {
    let fallback = default_emp_poles(sr_hz);
    let Some(st0) = dataset.stems.first() else {
        return fallback;
    };
    let n = st0.usable_len();
    if n < 128 {
        return fallback;
    }
    let w = st0.warmup.min(n / 2);
    let e_in: f64 = st0.in_l[w..n]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    let e_tgt: f64 = st0.tgt_l[w..n]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    if e_in <= 1.0e-8 || e_tgt <= 1.0e-8 {
        return fallback;
    }
    let mut probe =
        V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, mono_out));
    probe.os2x = false;
    probe.tier_os2x = [false; 4];
    probe.max_active_groups = 1;
    probe.active_tier_weights = [1.0, 0.0, 0.0, 0.0];
    let ds_probe = dataset.primary_subwindow_opts(4_096, true, false);
    select_omp_kautz_poles_matryoshka(&mut probe, &ds_probe);
    let mut out = fallback;
    for (slot, &(r, th)) in probe.kautz_poles[..4].iter().enumerate() {
        if r.is_finite() && (0.05..0.9999).contains(&r) && th.is_finite() && (0.0..PI).contains(&th)
        {
            out[slot] = (r, th);
        }
    }
    out
}

fn smooth_3tap(vals: &[f64; N_LOG_BANDS]) -> [f64; N_LOG_BANDS] {
    let mut out = [0.0_f64; N_LOG_BANDS];
    out[0] = 0.75 * vals[0] + 0.25 * vals[1];
    out[N_LOG_BANDS - 1] = 0.25 * vals[N_LOG_BANDS - 2] + 0.75 * vals[N_LOG_BANDS - 1];
    for b in 1..(N_LOG_BANDS - 1) {
        out[b] = 0.25 * vals[b - 1] + 0.50 * vals[b] + 0.25 * vals[b + 1];
    }
    out
}

fn fill_unpopulated_bands(vals: &mut [f64; N_LOG_BANDS], populated: &[bool; N_LOG_BANDS]) {
    let Some(first) = populated.iter().position(|&p| p) else {
        return;
    };
    let Some(last) = populated.iter().rposition(|&p| p) else {
        return;
    };
    let first_val = vals[first];
    let last_val = vals[last];
    for val in vals.iter_mut().take(first) {
        *val = first_val;
    }
    for val in vals.iter_mut().skip(last + 1) {
        *val = last_val;
    }
    let mut i = first;
    while i < last {
        if populated[i] {
            let mut j = i + 1;
            while j <= last && !populated[j] {
                j += 1;
            }
            if j > i + 1 && j <= last {
                let span = (j - i) as f64;
                let (v0, v1) = (vals[i], vals[j]);
                for (offset, val) in vals[(i + 1)..j].iter_mut().enumerate() {
                    let frac = ((offset + 1) as f64) / span;
                    *val = v0 * (1.0 - frac) + v1 * frac;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

impl SpectralSplitEstimate {
    /// Estimates linear poles and level-stratified/coherence split frequencies strictly from `(x, y)`.
    #[must_use]
    pub(super) fn from_dataset(dataset: &V3TrainDataset, mono_out: bool) -> Self {
        let sr_hz = dataset.sample_rate_hz.max(8_000.0);
        let emp_poles = estimate_emp_poles(dataset, mono_out, sr_hz);
        let fallback = Self {
            f_biamp_hz: 320.0,
            f_lo_hz: 220.0,
            f_hi_hz: 1_600.0,
            f_top_hz: 7_500.0,
            emp_poles,
        };
        let Some(st0) = dataset.stems.first() else {
            return fallback;
        };
        let n = st0.usable_len();
        let w = st0.warmup.min(n / 4);
        if n <= w + STFT_WIN {
            return fallback;
        }
        let n_stft = STFT_WIN;
        let hop_stft = STFT_HOP;
        let scan_end = n.min(w + 96_000);
        let mut active_frames: Vec<(usize, f64)> = Vec::new();
        let mut off = w;
        while off + n_stft <= scan_end {
            let mut ss = 0.0_f64;
            for &x in &st0.in_l[off..off + n_stft] {
                let fx = f64::from(x);
                ss += fx * fx;
            }
            let rms = (ss / (n_stft as f64)).sqrt();
            if rms > 1.0e-4 {
                active_frames.push((off, rms));
            }
            off += hop_stft;
        }
        if active_frames.len() < 4 {
            return fallback;
        }
        active_frames.sort_by(|a, b| a.1.total_cmp(&b.1));
        let k_sub = ((active_frames.len() * 25) / 100).max(1);
        let quiet_hi = k_sub;
        let loud_lo = active_frames.len().saturating_sub(k_sub);

        let denom = (n_stft - 1) as f64;
        let win: Vec<f64> = (0..n_stft)
            .map(|j| 0.5 * (1.0 - (TAU * (j as f64) / denom).cos()))
            .collect();
        let n_bins = n_stft / 2 + 1;
        let mut s_xx_all = vec![0.0_f64; n_bins];
        let mut s_yy_all = vec![0.0_f64; n_bins];
        let mut s_yx_re = vec![0.0_f64; n_bins];
        let mut s_yx_im = vec![0.0_f64; n_bins];
        let mut s_xx_q = vec![0.0_f64; n_bins];
        let mut s_yy_q = vec![0.0_f64; n_bins];
        let mut s_xx_l = vec![0.0_f64; n_bins];
        let mut s_yy_l = vec![0.0_f64; n_bins];
        let mut bx = vec![Complex64::new(0.0, 0.0); n_stft];
        let mut by = vec![Complex64::new(0.0, 0.0); n_stft];

        for (idx, &(f_off, _)) in active_frames.iter().enumerate() {
            for j in 0..n_stft {
                let wj = win[j];
                bx[j] = Complex64::new(f64::from(st0.in_l[f_off + j]) * wj, 0.0);
                by[j] = Complex64::new(f64::from(st0.tgt_l[f_off + j]) * wj, 0.0);
            }
            fft_radix2_in_place(&mut bx, false);
            fft_radix2_in_place(&mut by, false);
            let is_q = idx < quiet_hi;
            let is_l = idx >= loud_lo;
            for k in 1..n_bins {
                let px = bx[k].re * bx[k].re + bx[k].im * bx[k].im;
                let py = by[k].re * by[k].re + by[k].im * by[k].im;
                s_xx_all[k] += px;
                s_yy_all[k] += py;
                s_yx_re[k] += by[k].re * bx[k].re + by[k].im * bx[k].im;
                s_yx_im[k] += by[k].im * bx[k].re - by[k].re * bx[k].im;
                if is_q {
                    s_xx_q[k] += px;
                    s_yy_q[k] += py;
                }
                if is_l {
                    s_xx_l[k] += px;
                    s_yy_l[k] += py;
                }
            }
        }

        let f_hi_edge = F_MAX_HZ.min(0.45 * sr_hz);
        let log_span = (f_hi_edge / F_MIN_HZ).ln().max(1.0e-6);
        let freqs: [f64; N_LOG_BANDS] = std::array::from_fn(|b| {
            F_MIN_HZ * (log_span * (b as f64) / ((N_LOG_BANDS - 1) as f64)).exp()
        });
        let max_xx = s_xx_all[1..].iter().copied().fold(0.0_f64, f64::max);
        let bin_eps = (1.0e-12 * max_xx).max(1.0e-24);

        let mut b_px_a = [0.0_f64; N_LOG_BANDS];
        let mut b_py_a = [0.0_f64; N_LOG_BANDS];
        let mut b_plin = [0.0_f64; N_LOG_BANDS];
        let mut b_px_q = [0.0_f64; N_LOG_BANDS];
        let mut b_py_q = [0.0_f64; N_LOG_BANDS];
        let mut b_px_l = [0.0_f64; N_LOG_BANDS];
        let mut b_py_l = [0.0_f64; N_LOG_BANDS];

        let mut accum_band = |b: usize, w_b: f64, k: usize, p_lin_k: f64| {
            b_px_a[b] += w_b * s_xx_all[k];
            b_py_a[b] += w_b * s_yy_all[k];
            b_plin[b] += w_b * p_lin_k;
            b_px_q[b] += w_b * s_xx_q[k];
            b_py_q[b] += w_b * s_yy_q[k];
            b_px_l[b] += w_b * s_xx_l[k];
            b_py_l[b] += w_b * s_yy_l[k];
        };
        for k in 1..n_bins {
            let fk = (k as f64) * sr_hz / (n_stft as f64);
            if fk < F_MIN_HZ || fk > f_hi_edge {
                continue;
            }
            let u = ((fk / F_MIN_HZ).ln() / log_span) * ((N_LOG_BANDS - 1) as f64);
            let b0 = (u.floor() as usize).min(N_LOG_BANDS - 1);
            let frac = (u - (b0 as f64)).clamp(0.0, 1.0);
            let yx_sq = s_yx_re[k] * s_yx_re[k] + s_yx_im[k] * s_yx_im[k];
            let p_lin_k = (yx_sq / (s_xx_all[k] + bin_eps)).min(s_yy_all[k]);
            accum_band(b0, 1.0 - frac, k, p_lin_k);
            if b0 + 1 < N_LOG_BANDS {
                accum_band(b0 + 1, frac, k, p_lin_k);
            }
        }

        let max_band_px = b_px_a.iter().copied().fold(0.0_f64, f64::max);
        if max_band_px <= 1.0e-20 {
            return fallback;
        }
        let band_eps = (1.0e-9 * max_band_px).max(1.0e-24);
        let pop_thresh = 1.0e-5 * max_band_px;
        let mut populated = [false; N_LOG_BANDS];
        let mut g_q = [0.0_f64; N_LOG_BANDS];
        let mut delta_g = [0.0_f64; N_LOG_BANDS];
        let mut g_all = [0.0_f64; N_LOG_BANDS];
        let mut coh = [1.0_f64; N_LOG_BANDS];

        for b in 0..N_LOG_BANDS {
            if b_px_a[b] >= pop_thresh && b_px_q[b] > band_eps && b_px_l[b] > band_eps {
                populated[b] = true;
                let gain_q = ((b_py_q[b] + band_eps) / (b_px_q[b] + band_eps)).max(1.0e-12);
                let gain_l = ((b_py_l[b] + band_eps) / (b_px_l[b] + band_eps)).max(1.0e-12);
                g_q[b] = 10.0 * gain_q.log10();
                delta_g[b] = 10.0 * (gain_l / gain_q).max(1.0e-12).log10();
                g_all[b] = 10.0
                    * ((b_py_a[b] + band_eps) / (b_px_a[b] + band_eps))
                        .max(1.0e-12)
                        .log10();
                coh[b] = (b_plin[b] / (b_py_a[b] + band_eps)).clamp(0.0, 1.0);
            }
        }
        if populated.iter().filter(|&&p| p).count() < 4 {
            return fallback;
        }
        fill_unpopulated_bands(&mut g_q, &populated);
        fill_unpopulated_bands(&mut delta_g, &populated);
        fill_unpopulated_bands(&mut g_all, &populated);
        fill_unpopulated_bands(&mut coh, &populated);

        let g_q_s = smooth_3tap(&g_q);
        let dg_s = smooth_3tap(&delta_g);
        let g_all_s = smooth_3tap(&g_all);
        let coh_s = smooth_3tap(&coh);

        let mut sal = [0.0_f64; N_LOG_BANDS];
        for b in 1..(N_LOG_BANDS - 1) {
            sal[b] = 1.4 * (dg_s[b + 1] - dg_s[b - 1]).abs()
                + 0.45 * (g_q_s[b + 1] - g_q_s[b - 1]).abs()
                + 2.5 * (coh_s[b + 1] - coh_s[b - 1]).abs();
        }

        let mut f_biamp_hz = fallback.f_biamp_hz;
        let mut best_bi = -1.0_f64;
        for b in 1..(N_LOG_BANDS - 1) {
            if (70.0..=1_500.0).contains(&freqs[b]) && sal[b] > best_bi {
                best_bi = sal[b];
                f_biamp_hz = freqs[b];
            }
        }

        let mut f_lo_hz = fallback.f_lo_hz;
        let mut f_hi_hz = fallback.f_hi_hz;
        let mut best_pair = -1.0_f64;
        for b1 in 1..(N_LOG_BANDS - 2) {
            if !(60.0..=750.0).contains(&freqs[b1]) {
                continue;
            }
            for b2 in (b1 + 1)..(N_LOG_BANDS - 1) {
                if !(500.0..=5_500.0).contains(&freqs[b2]) || freqs[b2] < 2.2 * freqs[b1] {
                    continue;
                }
                let sc = sal[b1] + sal[b2];
                if sc > best_pair {
                    best_pair = sc;
                    f_lo_hz = freqs[b1];
                    f_hi_hz = freqs[b2];
                }
            }
        }

        let mid_peak = freqs
            .iter()
            .zip(&g_all_s)
            .filter(|(f, _)| (1_000.0..=3_500.0).contains(*f))
            .map(|(_, &g)| g)
            .fold(f64::NEG_INFINITY, f64::max);
        let f_top_hz = if mid_peak.is_finite() {
            freqs
                .iter()
                .zip(&g_all_s)
                .find(|(f, g)| (2_500.0..=12_000.0).contains(*f) && **g <= mid_peak - 3.0)
                .map_or(fallback.f_top_hz, |(&f, _)| f)
        } else {
            fallback.f_top_hz
        };

        Self {
            f_biamp_hz,
            f_lo_hz,
            f_hi_hz,
            f_top_hz,
            emp_poles,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::canonical_ltas_33_centers_hz;
    use crate::solver::v3_varpro::model_f64::V3TrainStem;

    fn render_multiband_probe(f_lo: f64, f_hi: f64, sr_hz: f64, n: usize) -> V3TrainStem {
        let centers = canonical_ltas_33_centers_hz();
        let a_lo = (-TAU * f_lo / sr_hz).exp();
        let a_hi = (-TAU * f_hi / sr_hz).exp();
        let (mut lp1, mut lp2) = (0.0_f64, 0.0_f64);
        let mut x = vec![0.0_f32; n];
        let mut y = vec![0.0_f32; n];
        for i in 0..n {
            let t = (i as f64) / sr_hz;
            let seg_env = if (i / 4_096) % 2 == 0 { 0.08 } else { 0.85 };
            let mut s = 0.0_f64;
            for (k, &fc) in centers.iter().enumerate() {
                s += (TAU * fc * t + 0.41 * (k as f64)).sin();
            }
            let xi = seg_env * 0.22 * s;
            lp1 = a_lo * lp1 + (1.0 - a_lo) * xi;
            lp2 = a_hi * lp2 + (1.0 - a_hi) * xi;
            let low = lp1;
            let mid = lp2 - lp1;
            let high = xi - lp2;
            let yi = 0.45 * low + (3.2 * mid).tanh() * 0.65 + (6.5 * high).tanh() * 0.30;
            x[i] = xi as f32;
            y[i] = yi as f32;
        }
        V3TrainStem {
            in_l: x.clone(),
            in_r: x,
            tgt_l: y.clone(),
            tgt_r: y,
            warmup: 512,
        }
    }

    #[test]
    fn test_spectral_split_estimate_tracks_diverse_unknown_crossovers() {
        let ds_a = V3TrainDataset::new(
            vec![render_multiband_probe(135.0, 3_100.0, 48_000.0, 24_576)],
            48_000.0,
        );
        let est_a = SpectralSplitEstimate::from_dataset(&ds_a, true);

        let ds_b = V3TrainDataset::new(
            vec![render_multiband_probe(420.0, 1_150.0, 48_000.0, 24_576)],
            48_000.0,
        );
        let est_b = SpectralSplitEstimate::from_dataset(&ds_b, true);

        assert!(
            est_a.f_lo_hz < est_b.f_lo_hz,
            "expected lower f_lo estimate on (135, 3100) vs (420, 1150): {est_a:?} vs {est_b:?}"
        );
        assert!(
            est_a.f_hi_hz > est_b.f_hi_hz,
            "expected higher f_hi estimate on (135, 3100) vs (420, 1150): {est_a:?} vs {est_b:?}"
        );
        assert!(
            est_a
                .emp_poles
                .iter()
                .all(|&(r, th)| r > 0.0 && r < 1.0 && th.is_finite())
        );
    }
}
