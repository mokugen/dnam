#![forbid(unsafe_code)]
//! `4096`-FFT Hann Welch Wiener Impulse-Response Identification, Linearity Gate (`R^2_lin`), and
//! Cabinet-Deconvolved 2-Tap Room Reflection Estimation (`room_lti::wiener_echo`).

use dnam_engine::stages::stage4_tank::v3_fdn::V3Stage4TankParams;

use crate::solver::v3_varpro::model_f64::V3TrainStem;
use crate::stimulus::fft::{Complex64, fft_radix2_in_place};

use super::{
    ChannelEchoFit, ECHO_R2_GATE, MAX_CHANNEL_TAP_L1, MAX_ECHO_LAG, MAX_SCAN_SAMPLES, MIN_ECHO_LAG,
    MIN_TAP_SEP, build_feedforward_tank, eval_2tap_chan,
};

const WIENER_N_FFT: usize = 4_096;
const DECONV_N_FFT: usize = 2_048;
const MAX_IR_LEN: usize = 1_024;

struct WienerChanEstimate {
    fit: Option<ChannelEchoFit>,
    r2_lin: f64,
}

fn estimate_welch_ir(inp: &[f32], tgt: &[f32], start: usize, end: usize) -> Vec<f64> {
    let avail = end.saturating_sub(start);
    let win_len = avail.min(WIENER_N_FFT);
    if win_len < 256 {
        return vec![0.0; MAX_IR_LEN];
    }
    let hop = (win_len / 2).max(256);
    let denom = (win_len.max(2) - 1) as f64;
    let hann: Vec<f64> = (0..win_len)
        .map(|j| 0.5 * (1.0 - (std::f64::consts::TAU * (j as f64) / denom).cos()))
        .collect();

    let mut s_yx = vec![Complex64::new(0.0, 0.0); WIENER_N_FFT];
    let mut s_xx = vec![0.0_f64; WIENER_N_FFT];
    let mut bx = vec![Complex64::new(0.0, 0.0); WIENER_N_FFT];
    let mut by = vec![Complex64::new(0.0, 0.0); WIENER_N_FFT];

    let mut offsets = Vec::new();
    let mut w0 = 0_usize;
    while w0 + win_len <= avail {
        offsets.push(w0);
        w0 += hop;
    }
    if avail > win_len && offsets.last().copied() != Some(avail - win_len) {
        offsets.push(avail - win_len);
    }
    for off in offsets {
        let base = start + off;
        for j in 0..win_len {
            let wj = hann[j];
            bx[j] = Complex64::new(f64::from(inp[base + j]) * wj, 0.0);
            by[j] = Complex64::new(f64::from(tgt[base + j]) * wj, 0.0);
        }
        bx[win_len..].fill(Complex64::new(0.0, 0.0));
        by[win_len..].fill(Complex64::new(0.0, 0.0));
        fft_radix2_in_place(&mut bx, false);
        fft_radix2_in_place(&mut by, false);
        for k in 0..WIENER_N_FFT {
            let (xr, xi) = (bx[k].re, bx[k].im);
            let (yr, yi) = (by[k].re, by[k].im);
            s_yx[k].re += yr * xr + yi * xi;
            s_yx[k].im += yi * xr - yr * xi;
            s_xx[k] += xr * xr + xi * xi;
        }
    }

    let mean_xx = (s_xx.iter().sum::<f64>() / (WIENER_N_FFT as f64)).max(1.0e-24);
    let reg = 1.0e-5 * mean_xx;
    let mut h_fft: Vec<Complex64> = s_yx
        .iter()
        .zip(&s_xx)
        .map(|(yx, &xx)| {
            let d = xx + reg;
            Complex64::new(yx.re / d, yx.im / d)
        })
        .collect();
    fft_radix2_in_place(&mut h_fft, true);
    h_fft.into_iter().take(MAX_IR_LEN).map(|c| c.re).collect()
}

fn deconvolve_room_from_ir(h_w: &[f64], cab_len: usize) -> Vec<f64> {
    let mut bw = vec![Complex64::new(0.0, 0.0); DECONV_N_FFT];
    let mut bh = vec![Complex64::new(0.0, 0.0); DECONV_N_FFT];
    for k in 0..h_w.len().min(DECONV_N_FFT) {
        bw[k] = Complex64::new(h_w[k], 0.0);
    }
    for k in 0..cab_len.min(h_w.len()) {
        bh[k] = Complex64::new(h_w[k], 0.0);
    }
    fft_radix2_in_place(&mut bw, false);
    fft_radix2_in_place(&mut bh, false);

    let mean_p = (bh.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f64>()
        / (DECONV_N_FFT as f64))
        .max(1.0e-24);
    let reg = 5.0e-4 * mean_p;
    let mut hr: Vec<Complex64> = bw
        .iter()
        .zip(&bh)
        .map(|(w, h)| {
            let den = h.re * h.re + h.im * h.im + reg;
            Complex64::new(
                (w.re * h.re + w.im * h.im) / den,
                (w.im * h.re - w.re * h.im) / den,
            )
        })
        .collect();
    // Bandpass-taper `hr` (`450..750 Hz` HP, `16..19.5 kHz` LP) before IFFT so truncated
    // low-frequency cabinet ringing does not bias discrete room reflection lag detection.
    hr[0] = Complex64::new(0.0, 0.0);
    hr[DECONV_N_FFT / 2] = Complex64::new(0.0, 0.0);
    for k in 1..(DECONV_N_FFT / 2) {
        let hz = (k as f64) * 48_000.0 / (DECONV_N_FFT as f64);
        let hp = if hz <= 450.0 {
            0.0
        } else if hz >= 750.0 {
            1.0
        } else {
            0.5 * (1.0 - (std::f64::consts::PI * (hz - 450.0) / 300.0).cos())
        };
        let lp = if hz >= 19_500.0 {
            0.0
        } else if hz <= 16_000.0 {
            1.0
        } else {
            0.5 * (1.0 + (std::f64::consts::PI * (hz - 16_000.0) / 3_500.0).cos())
        };
        let w_bp = hp * lp;
        hr[k].re *= w_bp;
        hr[k].im *= w_bp;
        hr[DECONV_N_FFT - k] = Complex64::new(hr[k].re, -hr[k].im);
    }
    fft_radix2_in_place(&mut hr, true);
    hr.into_iter().map(|c| c.re).collect()
}

fn find_peak_lag(sig: &[f64], lo: usize, hi: usize, exclude: Option<usize>) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for d in lo..=hi.min(sig.len().saturating_sub(1)) {
        if exclude.is_some_and(|ex| ex.abs_diff(d) < MIN_TAP_SEP) {
            continue;
        }
        let v = sig[d].abs();
        if best.is_none_or(|b| v > b.1) {
            best = Some((d, v));
        }
    }
    best.map(|(d, _)| d)
}

fn eval_linear_r2(inp: &[f32], tgt: &[f32], h_w: &[f64], start: usize, n: usize) -> f64 {
    let eval_start = start.max(h_w.len());
    let eval_end = n.min(eval_start + MAX_SCAN_SAMPLES);
    if eval_end <= eval_start + 128 {
        return 0.0;
    }
    let (mut sse, mut y_ss) = (0.0_f64, 0.0_f64);
    for t in eval_start..eval_end {
        let mut pred = 0.0_f64;
        for (k, &hk) in h_w.iter().enumerate() {
            pred += hk * f64::from(inp[t - k]);
        }
        let y = f64::from(tgt[t]);
        let err = y - pred;
        sse += err * err;
        y_ss += y * y;
    }
    if y_ss <= 1.0e-12 {
        0.0
    } else {
        (1.0 - sse / y_ss).clamp(0.0, 1.0)
    }
}

fn analyze_channel(inp: &[f32], tgt: &[f32], warmup: usize) -> Option<WienerChanEstimate> {
    let n = inp.len().min(tgt.len());
    let start = warmup.max(MAX_ECHO_LAG + 16);
    if n <= start + 512 {
        return None;
    }
    let scan_end = n.min(start + 65_536);
    let h_w = estimate_welch_ir(inp, tgt, start, scan_end);
    let r2_lin = eval_linear_r2(inp, tgt, &h_w, start, n);

    let room0 = deconvolve_room_from_ir(&h_w, MIN_ECHO_LAG);
    let d1_init = find_peak_lag(&room0, MIN_ECHO_LAG, MAX_ECHO_LAG, None)?;
    let cab_len = d1_init.saturating_sub(2).clamp(48, 128);
    let room1 = deconvolve_room_from_ir(&h_w, cab_len);
    let d1_c = find_peak_lag(&room1, MIN_ECHO_LAG, MAX_ECHO_LAG, None)?;
    let d2_c = find_peak_lag(&room1, MIN_ECHO_LAG, MAX_ECHO_LAG, Some(d1_c)).unwrap_or_else(|| {
        if d1_c + 64 <= MAX_ECHO_LAG {
            d1_c + 64
        } else {
            d1_c - 64
        }
    });

    let time_end = n.min(start + MAX_SCAN_SAMPLES);
    let mut u_head = vec![0.0_f32; time_end];
    let mut res = vec![0.0_f32; time_end];
    for t in 0..time_end {
        let mut s = 0.0_f64;
        for k in 0..cab_len.min(t + 1) {
            s += h_w[k] * f64::from(inp[t - k]);
        }
        u_head[t] = s as f32;
        res[t] = tgt[t] - u_head[t];
    }
    let mut u_hp = vec![0.0_f32; time_end];
    let mut res_hp = vec![0.0_f32; time_end];
    for t in 2..time_end {
        u_hp[t] = (f64::from(u_head[t]) - 2.0 * f64::from(u_head[t - 1]) + f64::from(u_head[t - 2]))
            as f32;
        res_hp[t] =
            (f64::from(res[t]) - 2.0 * f64::from(res[t - 1]) + f64::from(res[t - 2])) as f32;
    }
    let eval_start = start.max(cab_len + 2);

    let mut best: Option<ChannelEchoFit> = None;
    for dd1 in d1_c.saturating_sub(2).max(MIN_ECHO_LAG)..=(d1_c + 2).min(MAX_ECHO_LAG) {
        for dd2 in d2_c.saturating_sub(2).max(MIN_ECHO_LAG)..=(d2_c + 2).min(MAX_ECHO_LAG) {
            if dd1.abs_diff(dd2) < MIN_TAP_SEP {
                continue;
            }
            if let Some((mut g1, mut g2, r2)) =
                eval_2tap_chan(&u_hp, &res_hp, eval_start, time_end, dd1, dd2)
            {
                let l1 = g1.abs() + g2.abs();
                if l1 > MAX_CHANNEL_TAP_L1 {
                    let sc = MAX_CHANNEL_TAP_L1 / l1;
                    g1 *= sc;
                    g2 *= sc;
                }
                if best.is_none_or(|b| r2 > b.r2) {
                    best = Some(ChannelEchoFit {
                        d1: dd1,
                        d2: dd2,
                        g1,
                        g2,
                        r2,
                    });
                }
            }
        }
    }
    Some(WienerChanEstimate { fit: best, r2_lin })
}

/// Estimates `r2_lin` and discrete `2`-tap room reflections (`64..=960`) blindly from `(in, tgt)`
/// via `4096`-FFT Hann Welch IR deconvolution without relying on baseline Stage II/III predictions.
pub(super) fn detect_wiener_room_2tap(
    stem0: &V3TrainStem,
    warmup: usize,
    mono_out: bool,
) -> (Option<V3Stage4TankParams>, f64) {
    let Some(est_l) = analyze_channel(&stem0.in_l, &stem0.tgt_l, warmup) else {
        return (None, 0.0);
    };
    let est_r = (!mono_out)
        .then(|| analyze_channel(&stem0.in_r, &stem0.tgt_r, warmup))
        .flatten();
    let r2_lin = est_r
        .as_ref()
        .map_or(est_l.r2_lin, |r| est_l.r2_lin.min(r.r2_lin));

    let Some(fit_l) = est_l.fit else {
        return (None, r2_lin);
    };
    let fit_r = est_r.and_then(|e| e.fit);
    let max_r2 = fit_l.r2.max(fit_r.map_or(0.0, |fr| fr.r2));
    if max_r2 < ECHO_R2_GATE {
        return (None, r2_lin);
    }
    (Some(build_feedforward_tank(fit_l, fit_r)), r2_lin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_wiener_room_2tap_recovers_cabinet_convolved_echoes_and_linearity_gate() {
        let n = 16_384;
        let mut seed: u64 = 0x1357_9BDF_2468_ACE0;
        let dry: Vec<f32> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let w = ((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
                let t = (i as f64) / 48_000.0;
                (0.28 * w + 0.16 * (TAU * 110.0 * t).sin()) as f32
            })
            .collect();

        // Synthetic 16-tap decaying LTI cabinet * (1 + 0.24 z^-89 - 0.14 z^-233)
        let mut cab = vec![0.0_f32; n];
        for i in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..16 {
                if i >= k {
                    let h = if k == 0 {
                        0.72
                    } else {
                        0.18 * (-0.35 * (k as f64)).exp() * ((k as f64) * 0.6).cos()
                    };
                    acc += h * f64::from(dry[i - k]);
                }
            }
            cab[i] = acc as f32;
        }
        let mut tgt = vec![0.0_f32; n];
        for i in 0..n {
            let mut y = f64::from(cab[i]);
            if i >= 89 {
                y += 0.24 * f64::from(cab[i - 89]);
            }
            if i >= 233 {
                y -= 0.14 * f64::from(cab[i - 233]);
            }
            tgt[i] = y as f32;
        }
        let stem = V3TrainStem {
            in_l: dry.clone(),
            in_r: dry,
            tgt_l: tgt.clone(),
            tgt_r: tgt,
            warmup: 512,
        };
        let (tank_opt, r2_lin) = detect_wiener_room_2tap(&stem, 512, true);
        assert!(r2_lin > 0.99, "expected r2_lin > 0.99, got {r2_lin}");
        let tank = tank_opt.expect("2-tap Wiener room detector must trigger");
        assert!((87..=91).contains(&tank.delay_lengths[0]));
        assert!((231..=235).contains(&tank.delay_lengths[2]));
        assert!((tank.tap_gains[0] - 0.24).abs() < 0.05);
        assert!((tank.tap_gains[2] - (-0.14)).abs() < 0.05);
    }
}
