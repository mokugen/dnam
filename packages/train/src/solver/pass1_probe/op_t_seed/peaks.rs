//! GCC-PHAT, power-envelope cepstrum delay peak detection, and chunk envelope modulation
//! statistics for `Op-T` residual seeding (`ref[train.pipeline.physics-probe]`).

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;
use crate::losses::Radix2FftPlan;

pub(super) const GCC_WIN_CAP: usize = 16_384;
const _: () = assert!(GCC_WIN_CAP.is_power_of_two());

pub(super) struct GccPeakStats {
    pub peaks: Vec<(f64, f64)>,
    pub top_lag: usize,
    pub phat_z: f64,
    pub dry_rho_at_top_lag: f64,
    pub res_xy_rho_at_top_lag: f64,
    pub excess_mod_cv: f64,
    pub env_corr: f64,
    pub hf_ratio: f64,
}

fn pair_bin_rms(l: &[f64], r: &[f64]) -> f64 {
    let ss: f64 = l.iter().zip(r).map(|(a, b)| a * a + b * b).sum();
    (ss / (2.0 * (l.len() as f64))).sqrt()
}

/// Compute Pearson correlation between two aligned envelope slices, returning `0.0`
/// when either envelope lacks length or normalized variation.
fn pearson_env_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || a.len() != b.len() {
        return 0.0;
    }
    let inv_n = 1.0 / (a.len() as f64);
    let a_mean = a.iter().sum::<f64>() * inv_n;
    let b_mean = b.iter().sum::<f64>() * inv_n;
    let (mut a_var, mut b_var, mut cov) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&av, &bv) in a.iter().zip(b) {
        let da = av - a_mean;
        let db = bv - b_mean;
        a_var += da * da;
        b_var += db * db;
        cov += da * db;
    }
    let a_std = (a_var * inv_n).sqrt();
    let b_std = (b_var * inv_n).sqrt();
    if a_std <= 1.0e-6 * a_mean.max(1.0e-12) || b_std <= 1.0e-6 * b_mean.max(1.0e-12) {
        return 0.0;
    }
    ((cov * inv_n) / (a_std * b_std)).clamp(-1.0, 1.0)
}

fn chunk_env_mod_stats(
    in_l: &[f64],
    in_r: &[f64],
    res_l: &[f64],
    res_r: &[f64],
    start: usize,
    win_n: usize,
) -> (f64, f64) {
    let chunk = 64_usize.min(win_n / 4).max(8);
    let mut dry_env = Vec::new();
    let mut res_env = Vec::new();
    let mut s_pos = start;
    while s_pos + chunk <= start + win_n {
        dry_env.push(pair_bin_rms(
            &in_l[s_pos..s_pos + chunk],
            &in_r[s_pos..s_pos + chunk],
        ));
        res_env.push(pair_bin_rms(
            &res_l[s_pos..s_pos + chunk],
            &res_r[s_pos..s_pos + chunk],
        ));
        s_pos += chunk;
    }
    if dry_env.is_empty() {
        return (0.0, 0.0);
    }
    let inv_m = 1.0 / (dry_env.len() as f64);
    let dry_mean = dry_env.iter().sum::<f64>() * inv_m;
    let res_mean = res_env.iter().sum::<f64>() * inv_m;
    let (mut dry_var, mut res_var) = (0.0_f64, 0.0_f64);
    for (&d, &r) in dry_env.iter().zip(&res_env) {
        let dd = d - dry_mean;
        let dr = r - res_mean;
        dry_var += dd * dd;
        res_var += dr * dr;
    }
    let dry_std = (dry_var * inv_m).sqrt();
    let res_std = (res_var * inv_m).sqrt();
    // Smooth LFOs (<= 16 Hz) remain coherent across adjacent 64-sample chunks (lag-1 autocorr > 0.45),
    // distinguishing true time-varying Op-T modulation from random bin-level noise in filtered audio.
    let res_lag1_ac = if res_env.len() >= 3 && res_var > 1.0e-12 {
        let num: f64 = res_env
            .windows(2)
            .map(|w| (w[0] - res_mean) * (w[1] - res_mean))
            .sum();
        (num / res_var).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let excess_mod_cv = if res_lag1_ac > 0.45 {
        (res_std / res_mean.max(1.0e-12) - dry_std / dry_mean.max(1.0e-12)).max(0.0)
    } else {
        0.0
    };
    // Compare both linear and squared dry envelopes against the residual envelope so coherent
    // nonlinear waveshaper harmonic distortion correlates with amplitude bursts and does not look
    // like an independent chorus LFO.
    let dry_sq_env: Vec<f64> = dry_env.iter().map(|&d| d * d).collect();
    let lag0 = pearson_env_corr(&dry_env, &res_env);
    let sq_lag0 = pearson_env_corr(&dry_sq_env, &res_env);
    let (lag1, sq_lag1) = if dry_env.len() >= 4 {
        let m1 = dry_env.len() - 1;
        (
            pearson_env_corr(&dry_env[..m1], &res_env[1..]),
            pearson_env_corr(&dry_sq_env[..m1], &res_env[1..]),
        )
    } else {
        (lag0, sq_lag0)
    };
    let env_corr = lag0.max(lag1).max(sq_lag0).max(sq_lag1);
    (excess_mod_cv, env_corr)
}

/// Computes FFT GCC-PHAT + power-envelope cepstrum delay candidates `>= 5.0` samples along with
/// MAD-calibrated peak Z-score, dry autocorrelation at `top_lag`, and excess envelope modulation CV.
pub(super) fn detect_gcc_phat_cepstrum_peaks(
    in_l: &[f64],
    in_r: &[f64],
    res_l: &[f64],
    res_r: &[f64],
    start: usize,
    max_lag: usize,
) -> GccPeakStats {
    let n = in_l.len().min(in_r.len()).min(res_l.len()).min(res_r.len());
    let win_n = n.saturating_sub(start).min(GCC_WIN_CAP);
    let lag_lo = MIN_DELAY_CROSSBAR_SAMP as usize;
    let lag_hi = max_lag.min(win_n / 2).max(lag_lo);
    if win_n < lag_lo + 16 {
        return GccPeakStats {
            peaks: Vec::new(),
            top_lag: 0,
            phat_z: 0.0,
            dry_rho_at_top_lag: 0.0,
            res_xy_rho_at_top_lag: 0.0,
            excess_mod_cv: 0.0,
            env_corr: 0.0,
            hf_ratio: 0.0,
        };
    }

    let l_fft = (2 * win_n).next_power_of_two();
    let Ok(plan) = Radix2FftPlan::new(l_fft) else {
        return GccPeakStats {
            peaks: Vec::new(),
            top_lag: 0,
            phat_z: 0.0,
            dry_rho_at_top_lag: 0.0,
            res_xy_rho_at_top_lag: 0.0,
            excess_mod_cv: 0.0,
            env_corr: 0.0,
            hf_ratio: 0.0,
        };
    };

    let dry_mono: Vec<f64> = (0..win_n)
        .map(|t| 0.5 * (in_l[start + t] + in_r[start + t]))
        .collect();
    let res_mono: Vec<f64> = (0..win_n)
        .map(|t| 0.5 * (res_l[start + t] + res_r[start + t]))
        .collect();
    let xm_mean = dry_mono.iter().sum::<f64>() / (win_n as f64);
    let ym_mean = res_mono.iter().sum::<f64>() / (win_n as f64);

    let mut xr = vec![0.0_f64; l_fft];
    let mut xi = vec![0.0_f64; l_fft];
    let mut yr = vec![0.0_f64; l_fft];
    let mut yi = vec![0.0_f64; l_fft];
    for t in 0..win_n {
        xr[t] = dry_mono[t] - xm_mean;
        yr[t] = res_mono[t] - ym_mean;
    }
    plan.fft_in_place(&mut xr, &mut xi);
    plan.fft_in_place(&mut yr, &mut yi);

    let mut avg_p = 0.0_f64;
    let mut phat_re = vec![0.0_f64; l_fft];
    let mut phat_im = vec![0.0_f64; l_fft];
    let mut cep_re = vec![0.0_f64; l_fft];
    let mut cep_im = vec![0.0_f64; l_fft];
    for k in 0..l_fft {
        let cr = yr[k] * xr[k] + yi[k] * xi[k];
        let ci = yi[k] * xr[k] - yr[k] * xi[k];
        let mag = cr.hypot(ci);
        avg_p += mag;
        let pyr = yr[k] * yr[k] + yi[k] * yi[k];
        cep_re[k] = (pyr + 1.0e-12).ln();
        phat_re[k] = cr;
        phat_im[k] = ci;
    }
    let reg = 1.0e-3 * (avg_p / (l_fft as f64)).max(1.0e-12);
    for k in 0..l_fft {
        let den = phat_re[k].hypot(phat_im[k]) + reg;
        phat_re[k] /= den;
        phat_im[k] /= den;
    }
    plan.ifft_in_place(&mut phat_re, &mut phat_im);
    plan.ifft_in_place(&mut cep_re, &mut cep_im);

    let (excess_mod_cv, env_corr) = chunk_env_mod_stats(in_l, in_r, res_l, res_r, start, win_n);

    let (mut hf_res, mut tot_res) = (0.0_f64, 0.0_f64);
    for k in 1..(l_fft / 2) {
        let p = yr[k] * yr[k] + yi[k] * yi[k];
        tot_res += p;
        if k > l_fft / 4 {
            hf_res += p;
        }
    }
    let hf_ratio = hf_res / tot_res.max(1.0e-12);

    let mut phat_mags = Vec::with_capacity(lag_hi - lag_lo + 1);
    let mut scored = Vec::with_capacity(lag_hi - lag_lo + 1);
    let mut top_p = 0.0_f64;
    let mut top_lag = lag_lo;
    for lag in lag_lo..=lag_hi {
        let p_mag = phat_re[lag].abs();
        if p_mag.is_finite() {
            phat_mags.push(p_mag);
            if p_mag > top_p {
                top_p = p_mag;
                top_lag = lag;
            }
        }
        let score = p_mag + 0.35 * cep_re[lag].abs();
        if score.is_finite() {
            scored.push((lag as f64, score));
        }
    }
    phat_mags.sort_by(f64::total_cmp);
    let phat_z = if let Some(&med_p) = phat_mags.get(phat_mags.len() / 2) {
        let sigma_mad = (med_p / 0.674_489_75).max(1.0e-12);
        top_p / sigma_mad
    } else {
        0.0
    };
    let dry_e0: f64 = dry_mono
        .iter()
        .map(|&v| {
            let d = v - xm_mean;
            d * d
        })
        .sum::<f64>()
        .max(1.0e-12);
    let dry_lag_dot: f64 = (top_lag..win_n)
        .map(|t| (dry_mono[t] - xm_mean) * (dry_mono[t - top_lag] - xm_mean))
        .sum();
    let dry_rho_at_top_lag = dry_lag_dot.abs() / dry_e0;
    let res_e0: f64 = res_mono
        .iter()
        .map(|&v| {
            let d = v - ym_mean;
            d * d
        })
        .sum::<f64>()
        .max(1.0e-12);
    let res_lag_dot: f64 = (top_lag..win_n)
        .map(|t| (res_mono[t] - ym_mean) * (dry_mono[t - top_lag] - xm_mean))
        .sum();
    let res_xy_rho_at_top_lag = res_lag_dot.abs() / (dry_e0 * res_e0).sqrt().max(1.0e-12);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for (lag, sc) in scored {
        if peaks.iter().all(|&(pl, _)| (pl - lag).abs() >= 3.0) {
            peaks.push((lag, sc));
            if peaks.len() >= 16 {
                break;
            }
        }
    }
    GccPeakStats {
        peaks,
        top_lag,
        phat_z,
        dry_rho_at_top_lag,
        res_xy_rho_at_top_lag,
        excess_mod_cv,
        env_corr,
        hf_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcc_phat_short_window_and_env_mod_lag1_guard() {
        let tiny = [0.1_f64; 12];
        let stats = detect_gcc_phat_cepstrum_peaks(&tiny, &tiny, &tiny, &tiny, 0, 8);
        assert!(stats.peaks.is_empty());
        assert_eq!(stats.top_lag, 0);
        assert_eq!(stats.res_xy_rho_at_top_lag, 0.0);
        assert_eq!(
            chunk_env_mod_stats(&tiny, &tiny, &tiny, &tiny, 0, 4),
            (0.0, 0.0)
        );
        assert_eq!(pearson_env_corr(&[1.0], &[1.0]), 0.0);
        assert_eq!(pearson_env_corr(&[1.0, 1.0], &[2.0, 3.0]), 0.0);

        let n = 4_096_usize;
        let mut dry = vec![0.0_f64; n];
        let mut rng_state = 0x517c_c1b7_2722_0a95_u64;
        for dst in &mut dry {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *dst = ((rng_state >> 32) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
        }
        let mut echo_res = vec![0.0_f64; n];
        for t in 180..n {
            echo_res[t] = 0.25 * dry[t - 73] - 0.10 * dry[t - 109] + 0.08 * dry[t - 157];
        }
        let echo_stats = detect_gcc_phat_cepstrum_peaks(&dry, &dry, &echo_res, &echo_res, 64, 512);
        assert_eq!(echo_stats.top_lag, 73);
        assert!(
            echo_stats.res_xy_rho_at_top_lag >= 0.40,
            "genuine linear echo must have strong unwhitened rho: {}",
            echo_stats.res_xy_rho_at_top_lag
        );
    }

    #[test]
    fn test_chunk_env_mod_distinguishes_waveshaper_bursts_from_chorus_lfo() {
        let sr = 48_000.0;
        let n_am = 8192_usize;
        let mut am_in = vec![0.0_f64; n_am];
        let mut am_res = vec![0.0_f64; n_am];
        for (t, dst) in am_in.iter_mut().enumerate() {
            let env = 0.08
                + 0.92
                    * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * 3.0 * t as f64 / sr).cos())
                        .powi(2);
            let carrier = (2.0 * std::f64::consts::PI * 440.0 * t as f64 / sr).sin();
            *dst = env * carrier;
        }
        for t in 16..n_am {
            let x = am_in[t - 16];
            am_res[t] = x * x * x;
        }
        let (_am_cv, am_corr) =
            chunk_env_mod_stats(&am_in, &am_in, &am_res, &am_res, 64, n_am - 128);
        assert!(
            am_corr >= 0.70,
            "bursty AM + cubic shaper + 16-sample delay should correlate with dry envelope: env_corr={am_corr}"
        );

        let n_lfo = 48_000_usize;
        let mut lfo_in = vec![0.0_f64; n_lfo];
        let mut lfo_res = vec![0.0_f64; n_lfo];
        let mut rng_state = 0x1234_5678_9abc_def0_u64;
        for t in 0..n_lfo {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 32) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            let lfo = 0.55 + 0.45 * (2.0 * std::f64::consts::PI * 1.2 * t as f64 / sr).sin();
            lfo_in[t] = u;
            lfo_res[t] = u * lfo;
        }
        let (lfo_cv, lfo_corr) =
            chunk_env_mod_stats(&lfo_in, &lfo_in, &lfo_res, &lfo_res, 64, n_lfo - 128);
        assert!(
            lfo_cv > 0.18 && lfo_corr < 0.65,
            "stationary noise + 1.2 Hz chorus should trigger mod stats: cv={lfo_cv}, corr={lfo_corr}"
        );
    }
}
