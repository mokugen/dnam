#![forbid(unsafe_code)]
//! Closed-form sliding-window Wiener lag tracking & weighted `3 x 3` sinusoid LFO identification
//! (`stage4_id/incoherent_rig/lfo_detect.rs`).

use std::f64::consts::{PI, TAU};

use crate::stimulus::fft::{Complex64, fft_radix2_in_place};

use super::super::super::model_f64::V3TrainStem;

const DRY_FIR_TAPS: usize = 24;
const MAX_LAG_SHIFT: usize = 1_024;
const WIN_LEN: usize = 512;
const N_FFT: usize = 2_048;

#[derive(Clone, Copy, Debug)]
pub(super) struct DetectedLfoSpec {
    pub(super) mean_delay_l: f64,
    pub(super) mean_delay_r: f64,
    pub(super) rate_hz: f32,
    pub(super) depth_samples: f32,
    pub(super) stereo_phase: f32,
    pub(super) sign_l: f32,
    pub(super) sign_r: f32,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) r2: f64,
}

struct ChannelLagTrack {
    times_sec: Vec<f64>,
    taus: Vec<f64>,
    weights: Vec<f64>,
    sign: f32,
    tss: f64,
    w_mean: f64,
}

fn solve_sym_linear<const N: usize>(mut a: [[f64; N]; N], b: [f64; N]) -> Option<[f64; N]> {
    for r in 0..N {
        for c in 0..r {
            a[r][c] = a[c][r];
        }
    }
    let mut l = [[0.0_f64; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 1.0e-15 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0.0_f64; N];
    for i in 0..N {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0_f64; N];
    for i in (0..N).rev() {
        let mut s = y[i];
        for k in (i + 1)..N {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// Removes stationary early/dry linear contribution `h[0..DRY_FIR_TAPS]` so comb/chorus peaks at
/// `tau >= 32` stand out cleanly even on multisine or lowpass-filtered signals.
fn subtract_early_dry_response(inp: &[f32], tgt: &[f32], start: usize, avail: usize) -> Vec<f64> {
    let fit_end = (start + avail).min(start + 16_384);
    let mut g = [[0.0_f64; DRY_FIR_TAPS]; DRY_FIR_TAPS];
    let mut rhs = [0.0_f64; DRY_FIR_TAPS];
    for t in start.max(DRY_FIR_TAPS)..fit_end {
        let yt = f64::from(tgt[t]);
        for i in 0..DRY_FIR_TAPS {
            let xi = f64::from(inp[t - i]);
            rhs[i] += yt * xi;
            for j in i..DRY_FIR_TAPS {
                g[i][j] += xi * f64::from(inp[t - j]);
            }
        }
    }
    let mean_diag =
        ((0..DRY_FIR_TAPS).map(|i| g[i][i]).sum::<f64>() / (DRY_FIR_TAPS as f64)).max(1.0e-12);
    let ridge = 1.0e-4 * mean_diag;
    for (i, row) in g.iter_mut().enumerate() {
        row[i] += ridge;
    }
    let h = solve_sym_linear::<DRY_FIR_TAPS>(g, rhs).unwrap_or([0.0; DRY_FIR_TAPS]);

    let mut res = Vec::with_capacity(avail);
    for off in 0..avail {
        let t = start + off;
        let mut pred = 0.0_f64;
        for (k, &hk) in h.iter().enumerate() {
            if t >= k {
                pred += hk * f64::from(inp[t - k]);
            }
        }
        res.push(f64::from(tgt[t]) - pred);
    }
    res
}

fn parabolic_subsample_peak(h: &[Complex64], k: usize, lo: usize, hi: usize) -> f64 {
    if k <= lo || k >= hi {
        return k as f64;
    }
    let ym = h[k - 1].re.abs();
    let y0 = h[k].re.abs();
    let yp = h[k + 1].re.abs();
    let denom = ym - 2.0 * y0 + yp;
    if denom >= -1.0e-18 {
        k as f64
    } else {
        (k as f64) + (0.5 * (ym - yp) / denom).clamp(-0.5, 0.5)
    }
}

struct LagScanCfg<'a> {
    start: usize,
    avail: usize,
    hop: usize,
    sr_hz: f64,
    hann: &'a [f64],
}

fn track_channel_delay_trajectory(
    inp: &[f32],
    tgt: &[f32],
    lag_hint: Option<usize>,
    cfg: &LagScanCfg<'_>,
) -> Option<ChannelLagTrack> {
    let lag_lo = lag_hint.map_or(32, |d| d.saturating_sub(110).max(32));
    let lag_hi = lag_hint.map_or(1_000, |d| (d + 110).min(1_008));
    if lag_hi <= lag_lo + 4 || cfg.avail < WIN_LEN {
        return None;
    }

    let wet_res = subtract_early_dry_response(inp, tgt, cfg.start, cfg.avail);
    let mut bx = vec![Complex64::new(0.0, 0.0); N_FFT];
    let mut by = vec![Complex64::new(0.0, 0.0); N_FFT];
    let mut xc = vec![Complex64::new(0.0, 0.0); N_FFT];

    let mut times_sec = Vec::new();
    let mut taus = Vec::new();
    let mut weights = Vec::new();
    let mut sign_accum = 0.0_f64;

    let x_span = MAX_LAG_SHIFT + WIN_LEN;
    let mut off = 0_usize;
    while off + WIN_LEN <= cfg.avail {
        let base = cfg.start + off;
        for j in 0..x_span {
            let x = (base + j)
                .checked_sub(MAX_LAG_SHIFT)
                .and_then(|idx| inp.get(idx).copied())
                .unwrap_or(0.0);
            bx[j] = Complex64::new(f64::from(x), 0.0);
        }
        bx[x_span..N_FFT].fill(Complex64::new(0.0, 0.0));
        by[..MAX_LAG_SHIFT].fill(Complex64::new(0.0, 0.0));
        let mut e_y = 0.0_f64;
        for j in 0..WIN_LEN {
            let y = wet_res[off + j] * cfg.hann[j];
            e_y += y * y;
            by[MAX_LAG_SHIFT + j] = Complex64::new(y, 0.0);
        }
        by[x_span..N_FFT].fill(Complex64::new(0.0, 0.0));

        if e_y > 1.0e-18 {
            fft_radix2_in_place(&mut bx, false);
            fft_radix2_in_place(&mut by, false);
            let sum_xx: f64 = bx.iter().map(|z| z.norm_sqr()).sum();
            if sum_xx > 1.0e-18 {
                let mean_xx = (sum_xx / (N_FFT as f64)).max(1.0e-24);
                let reg = 0.15 * mean_xx;
                for k in 0..N_FFT {
                    let (xr, xi) = (bx[k].re, bx[k].im);
                    let (yr, yi) = (by[k].re, by[k].im);
                    let inv_d = 1.0 / (xr * xr + xi * xi + reg);
                    xc[k] =
                        Complex64::new((yr * xr + yi * xi) * inv_d, (yi * xr - yr * xi) * inv_d);
                }
                fft_radix2_in_place(&mut xc, true);

                let mut best_k = lag_lo;
                let mut best_abs = -1.0_f64;
                let mut sq_sum = 0.0_f64;
                for k in lag_lo..=lag_hi {
                    let v = xc[k].re.abs();
                    sq_sum += v * v;
                    if v > best_abs {
                        best_abs = v;
                        best_k = k;
                    }
                }
                let rms_h = (sq_sum / ((lag_hi - lag_lo + 1) as f64)).sqrt();
                if best_abs > 2.4 * rms_h && best_abs > 1.0e-9 {
                    let tau_hat = parabolic_subsample_peak(&xc, best_k, lag_lo, lag_hi);
                    let t_mid = ((base as f64) + 0.5 * ((WIN_LEN - 1) as f64)) / cfg.sr_hz;
                    let w = (best_abs / rms_h.max(1.0e-12)).min(32.0);
                    times_sec.push(t_mid);
                    taus.push(tau_hat);
                    weights.push(w);
                    sign_accum += xc[best_k].re * w;
                }
            }
        }
        off += cfg.hop;
    }

    if times_sec.len() < 8 {
        return None;
    }
    let w_sum: f64 = weights.iter().sum();
    if w_sum <= 1.0e-12 {
        return None;
    }
    let w_mean = taus.iter().zip(&weights).map(|(&t, &w)| t * w).sum::<f64>() / w_sum;
    let tss = taus
        .iter()
        .zip(&weights)
        .map(|(&t, &w)| {
            let d = t - w_mean;
            w * d * d
        })
        .sum::<f64>();

    Some(ChannelLagTrack {
        times_sec,
        taus,
        weights,
        sign: if sign_accum >= 0.0 { 1.0 } else { -1.0 },
        tss,
        w_mean,
    })
}

fn fit_channel_sinusoid_at_freq(tr: &ChannelLagTrack, freq_hz: f64) -> ([f64; 3], f64) {
    let w_ang = TAU * freq_hz;
    let mut ata = [[1.0e-9, 0.0, 0.0], [0.0, 1.0e-9, 0.0], [0.0, 0.0, 1.0e-9]];
    let mut atb = [0.0_f64; 3];
    for i in 0..tr.times_sec.len() {
        let wi = tr.weights[i];
        let (s, c) = (w_ang * tr.times_sec[i]).sin_cos();
        let row = [1.0, c, s];
        for r in 0..3 {
            atb[r] += wi * row[r] * tr.taus[i];
            for col in r..3 {
                ata[r][col] += wi * row[r] * row[col];
            }
        }
    }
    let Some(beta) = solve_sym_linear::<3>(ata, atb) else {
        return ([tr.w_mean, 0.0, 0.0], tr.tss);
    };
    let mut sse = 0.0_f64;
    for i in 0..tr.times_sec.len() {
        let (s, c) = (w_ang * tr.times_sec[i]).sin_cos();
        let pred = beta[0] + beta[1] * c + beta[2] * s;
        let err = tr.taus[i] - pred;
        sse += tr.weights[i] * err * err;
    }
    (beta, sse)
}

fn eval_joint_sse(
    tr_l: &ChannelLagTrack,
    tr_r: Option<&ChannelLagTrack>,
    freq_hz: f64,
) -> (f64, [f64; 3], Option<[f64; 3]>) {
    let (beta_l, sse_l) = fit_channel_sinusoid_at_freq(tr_l, freq_hz);
    let (beta_r, sse_r) = match tr_r {
        Some(r) => {
            let (br, sr) = fit_channel_sinusoid_at_freq(r, freq_hz);
            (Some(br), sr)
        }
        None => (None, 0.0),
    };
    (sse_l + sse_r, beta_l, beta_r)
}

fn wrap_pi(mut x: f64) -> f64 {
    while x > PI {
        x -= TAU;
    }
    while x <= -PI {
        x += TAU;
    }
    x
}

/// Identifies modulated delay-line center delay, LFO rate (`Hz`), peak depth (`samples`), and
/// stereo quadrature offset directly from dry/wet stems.
pub(super) fn detect_lfo_modulation(
    stem: &V3TrainStem,
    warmup: usize,
    sr_hz: f64,
    mono_out: bool,
    comb_hint: Option<(usize, usize)>,
) -> Option<DetectedLfoSpec> {
    let start = warmup.max(1_000);
    let end = stem.usable_len().min(start + 96_000);
    let avail = end.saturating_sub(start);
    if avail < 4_096 {
        return None;
    }
    let hop = if avail < 32_768 { 256 } else { 512 };
    let denom = (WIN_LEN - 1) as f64;
    let hann: Vec<f64> = (0..WIN_LEN)
        .map(|j| 0.5 * (1.0 - (TAU * (j as f64) / denom).cos()))
        .collect();

    let cfg = LagScanCfg {
        start,
        avail,
        hop,
        sr_hz,
        hann: &hann,
    };
    let tr_l_opt =
        track_channel_delay_trajectory(&stem.in_l, &stem.tgt_l, comb_hint.map(|(dl, _)| dl), &cfg);
    let tr_r_opt = (!mono_out)
        .then(|| {
            track_channel_delay_trajectory(
                &stem.in_r,
                &stem.tgt_r,
                comb_hint.map(|(_, dr)| dr),
                &cfg,
            )
        })
        .flatten();

    let (primary, secondary) = match (tr_l_opt, tr_r_opt) {
        (Some(l), r) => (l, r),
        (None, Some(r)) => (r, None),
        (None, None) => return None,
    };
    let total_tss = primary.tss + secondary.as_ref().map_or(0.0, |r| r.tss);
    if total_tss <= 1.0e-6 {
        return None;
    }

    let duration_sec = (avail as f64) / sr_hz.max(8_000.0);
    let f_min = (0.35 / duration_sec).clamp(0.20, 4.0).min(8.5);
    let f_max = 9.0_f64;

    let mut best_f = f_min;
    let mut best_sse = f64::INFINITY;
    for fi in 0..160 {
        let u = (fi as f64) / 159.0;
        let f = f_min * (f_max / f_min).powf(u);
        let (sse, _, _) = eval_joint_sse(&primary, secondary.as_ref(), f);
        if sse < best_sse {
            best_sse = sse;
            best_f = f;
        }
    }

    let f_lo_fine = (0.96 * best_f).max(0.15);
    let f_hi_fine = (1.04 * best_f).min(12.0);
    let mut best_beta_l = [primary.w_mean, 0.0, 0.0];
    let mut best_beta_r: Option<[f64; 3]> = None;
    for fi in 0..=16 {
        let f = f_lo_fine + ((fi as f64) / 16.0) * (f_hi_fine - f_lo_fine);
        let (sse, bl, br) = eval_joint_sse(&primary, secondary.as_ref(), f);
        if sse <= best_sse {
            best_sse = sse;
            best_f = f;
            best_beta_l = bl;
            best_beta_r = br;
        }
    }

    let r2 = (1.0 - best_sse / total_tss).clamp(0.0, 1.0);
    let depth_l = best_beta_l[1].hypot(best_beta_l[2]);
    let phi_l = best_beta_l[1].atan2(best_beta_l[2]);
    let (depth_r, phi_r) = match best_beta_r {
        Some(br) => (br[1].hypot(br[2]), br[1].atan2(br[2])),
        None => (0.0, phi_l),
    };
    let raw_depth = if depth_l >= 1.0 && depth_r >= 1.0 {
        0.5 * (depth_l + depth_r)
    } else {
        depth_l.max(depth_r)
    };
    if r2 < 0.30 || !(1.5..=64.0).contains(&raw_depth) {
        return None;
    }

    let mean_l = best_beta_l[0].clamp(12.0, 1_008.0);
    let mean_r = best_beta_r.map_or(mean_l, |br| br[0].clamp(12.0, 1_008.0));
    let safe_margin = (mean_l
        .min(mean_r)
        .min(1_019.0 - mean_l)
        .min(1_019.0 - mean_r)
        - 3.0)
        .clamp(0.0, 64.0);
    let depth_samples = raw_depth.min(safe_margin) as f32;
    if depth_samples < 1.5 {
        return None;
    }

    let stereo_phase = if !mono_out && best_beta_r.is_some() && depth_l >= 1.0 && depth_r >= 1.0 {
        wrap_pi(phi_r - phi_l) as f32
    } else {
        0.0
    };

    Some(DetectedLfoSpec {
        mean_delay_l: mean_l,
        mean_delay_r: mean_r,
        rate_hz: best_f as f32,
        depth_samples,
        stereo_phase,
        sign_l: primary.sign,
        sign_r: secondary.as_ref().map_or(primary.sign, |r| r.sign),
        r2,
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_4;

    use super::*;

    fn make_chorus_stem(n: usize, depth: f64, rate_hz: f64, stereo_phase: f64) -> V3TrainStem {
        let sr_hz = 48_000.0;
        let mut dry = Vec::with_capacity(n);
        let mut seed: u64 = 0x517C_C1B7_2722_0A95;
        for _ in 0..n {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let white = ((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            dry.push((0.25 * white) as f32);
        }
        let read_frac = |buf: &[f32], pos: usize, delay: f64| -> f64 {
            let p = (pos as f64) - delay;
            if p <= 0.0 {
                return 0.0;
            }
            let idx = p.floor() as usize;
            let frac = p - (idx as f64);
            let x0 = f64::from(buf[idx]);
            let x1 = f64::from(buf.get(idx + 1).copied().unwrap_or(0.0));
            x0 + frac * (x1 - x0)
        };
        let mut tgt_l = Vec::with_capacity(n);
        let mut tgt_r = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64) / sr_hz;
            let dl = 340.0 + depth * (TAU * rate_hz * t).sin();
            let dr = 340.0 + depth * (TAU * rate_hz * t + stereo_phase).sin();
            tgt_l.push((0.70 * f64::from(dry[i]) + 0.38 * read_frac(&dry, i, dl)) as f32);
            tgt_r.push((0.70 * f64::from(dry[i]) + 0.38 * read_frac(&dry, i, dr)) as f32);
        }
        V3TrainStem {
            in_l: dry.clone(),
            in_r: dry,
            tgt_l,
            tgt_r,
            warmup: 1_024,
        }
    }

    #[test]
    fn test_lfo_detect_short_and_full_stereo_chorus_vs_static_comb() {
        for &n in &[16_384_usize, 48_000] {
            let stem = make_chorus_stem(n, 45.0, 1.4, FRAC_PI_4);
            let det = detect_lfo_modulation(&stem, 1_024, 48_000.0, false, Some((340, 340)))
                .expect("modulated chorus must be detected");
            assert!(
                (f64::from(det.rate_hz) - 1.4).abs() < 0.15,
                "n={n} rate_hz={} expected ~1.4",
                det.rate_hz
            );
            assert!(
                (f64::from(det.depth_samples) - 45.0).abs() < 6.0,
                "n={n} depth={} expected ~45.0",
                det.depth_samples
            );
            assert!(
                (f64::from(det.stereo_phase) - FRAC_PI_4).abs() < 0.25,
                "n={n} stereo_phase={} expected ~FRAC_PI_4",
                det.stereo_phase
            );
            assert!(det.r2 > 0.85, "n={n} r2={} too low", det.r2);
        }

        let static_stem = make_chorus_stem(16_384, 0.0, 1.4, FRAC_PI_4);
        assert!(
            detect_lfo_modulation(&static_stem, 1_024, 48_000.0, false, Some((340, 340))).is_none(),
            "static LTI comb must not be misclassified as LFO modulated"
        );
    }
}
