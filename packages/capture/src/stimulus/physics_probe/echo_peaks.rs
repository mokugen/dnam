#![forbid(unsafe_code)]
//! Motif Cross-Correlation Peak Scanning, Whitened Lag Detection, and Stereo Echo Extraction
//! (`stimulus::physics_probe::echo_peaks`).

use super::tail::{DecayTailInput, TailDecayStats};

/// Computes root-mean-square energy of `buf` (`0.0` for empty slices).
pub(super) fn rms_energy(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    (buf.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / (buf.len() as f64))
        .sqrt()
}

/// Computes high-frequency first-difference energy ratio $\sqrt{\sum (\Delta x)^2 / (\sum x^2 + \epsilon)}$.
pub(super) fn first_diff_hf_ratio(buf: &[f32]) -> f64 {
    if buf.len() < 4 {
        return 0.0;
    }
    let (mut e, mut de, mut prev) = (0.0_f64, 0.0_f64, f64::from(buf[0]));
    for &s in &buf[1..] {
        let x = f64::from(s);
        e += x * x;
        de += (x - prev) * (x - prev);
        prev = x;
    }
    (de / (e + 1.0e-18)).sqrt()
}

/// Inputs for multi-candidate echo peak extraction across comb, coreset, and tail responses.
pub(super) struct EchoPeakInput<'a> {
    pub tail: &'a DecayTailInput<'a>,
    pub tail_energy_ratio: f64,
    pub env_ms: &'a [f64],
    pub comb_rms: f64,
}

struct PeakScanSpec<'a> {
    motif: &'a [f32],
    motif_e: f64,
    rms_m: f64,
    tgt: &'a [f32],
    start_off: usize,
    max_off: usize,
    c_step: usize,
    lag_base: usize,
    c_th: f64,
    f_th: f64,
    g_th: f64,
    dry_check: Option<(&'a [f32], f64, f64, &'a [f32])>,
}

pub(super) fn corr_at_offset(
    motif: &[f32],
    motif_e: f64,
    tail: &[f32],
    off: usize,
    comb_rms: f64,
    step: usize,
) -> (f64, f64, f64) {
    let (mut dot, mut te) = (0.0_f64, 0.0_f64);
    for i in (0..motif.len()).step_by(step) {
        let (d, t) = (f64::from(motif[i]), f64::from(tail[off + i]));
        dot += d * t;
        te += t * t;
    }
    let scale = step as f64;
    let sc = dot / ((motif_e / scale) * (te + 1.0e-12)).sqrt().max(1.0e-12);
    let gain = ((te * scale / (motif.len() as f64)).sqrt() / comb_rms.max(1.0e-6)).clamp(0.0, 1.5);
    (sc.abs(), sc, gain)
}

fn refine_local_peak(spec: &PeakScanSpec<'_>, lo: usize, hi: usize) -> (usize, f64, f64, f64) {
    let (mut bo, mut ba, mut bs, mut bg) = (lo, -1.0_f64, 0.0, 0.0);
    for o in lo..=hi {
        let (a, s, g) = corr_at_offset(spec.motif, spec.motif_e, spec.tgt, o, spec.rms_m, 1);
        if a > ba {
            ba = a;
            bo = o;
            bs = s;
            bg = g;
        }
    }
    (bo, ba, bs, bg)
}

fn parabolic_peak_offset(spec: &PeakScanSpec<'_>, bo: usize, ba: f64, lo: usize, hi: usize) -> f64 {
    if bo <= lo || bo >= hi {
        return 0.0;
    }
    let y0 = corr_at_offset(spec.motif, spec.motif_e, spec.tgt, bo - 1, spec.rms_m, 1).0;
    let y2 = corr_at_offset(spec.motif, spec.motif_e, spec.tgt, bo + 1, spec.rms_m, 1).0;
    let den = 2.0 * (y0 - 2.0 * ba + y2);
    if den.abs() > 1.0e-12 {
        ((y0 - y2) / den).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

fn scan_peaks_shared(spec: &PeakScanSpec<'_>) -> Vec<(f64, f64, f64, f64)> {
    if spec.start_off + 16 >= spec.max_off {
        return Vec::new();
    }
    let coarse: Vec<(usize, f64, f64)> = (spec.start_off..spec.max_off)
        .step_by(spec.c_step)
        .map(|o| {
            let (a, _, g) = corr_at_offset(spec.motif, spec.motif_e, spec.tgt, o, spec.rms_m, 4);
            (o, a, g)
        })
        .collect();
    let mut out = Vec::new();
    for i in 1..coarse.len().saturating_sub(1) {
        if coarse[i].1 > spec.c_th
            && coarse[i].2 > spec.g_th
            && coarse[i].1 >= coarse[i - 1].1
            && coarse[i].1 >= coarse[i + 1].1
        {
            let lo = coarse[i]
                .0
                .saturating_sub(spec.c_step + 4)
                .max(spec.start_off);
            let hi = (coarse[i].0 + spec.c_step + 4).min(spec.max_off - 1);
            let (bo, ba, bs, bg) = refine_local_peak(spec, lo, hi);
            if ba > spec.f_th && bg > spec.g_th {
                if let Some((dm, de, dr, dt)) = spec.dry_check
                    && bo + dm.len() <= dt.len()
                    && corr_at_offset(dm, de, dt, bo, dr, 1).0 >= 0.55 * ba
                {
                    continue;
                }
                let frac = parabolic_peak_offset(spec, bo, ba, lo, hi);
                let lag = (bo + spec.lag_base) as f64 + frac;
                if out
                    .last()
                    .is_none_or(|p: &(f64, f64, f64, f64)| (lag - p.0).abs() > 720.0)
                {
                    out.push((lag, ba, bs, bg.clamp(0.05, 1.2)));
                    if out.len() >= 4 {
                        break;
                    }
                }
            }
        }
    }
    out
}

pub(super) fn find_motif_peaks_fine(
    comb: &[f32],
    tail: &[f32],
    min_lag: usize,
) -> Vec<(f64, f64, f64, f64)> {
    let mlen = comb.len().min(1_024);
    if mlen < 256 || tail.len() < mlen + 64 {
        return Vec::new();
    }
    let motif = &comb[comb.len() - mlen..];
    let me: f64 = motif
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .max(1.0e-12);
    scan_peaks_shared(&PeakScanSpec {
        motif,
        motif_e: me,
        rms_m: (me / (mlen as f64)).sqrt().max(1.0e-6),
        tgt: tail,
        start_off: min_lag.saturating_sub(mlen),
        max_off: tail.len() - mlen,
        c_step: 4,
        lag_base: mlen,
        c_th: 0.28,
        f_th: 0.32,
        g_th: 0.03,
        dry_check: None,
    })
}

pub(super) fn find_whitened_lag_peaks(
    dry_src: &[f32],
    src: &[f32],
    dst: &[f32],
) -> Vec<(f64, f64, f64, f64)> {
    let n = src.len().min(dst.len()).min(dry_src.len());
    if n < 6_000 {
        return Vec::new();
    }
    let w: Vec<f32> = (1..n).map(|i| src[i] - 0.94 * src[i - 1]).collect();
    let wd: Vec<f32> = (1..n).map(|i| dst[i] - 0.94 * dst[i - 1]).collect();
    let wdry: Vec<f32> = (1..n).map(|i| dry_src[i] - 0.94 * dry_src[i - 1]).collect();
    let (mut bs, mut be) = (0usize, 0.0_f64);
    for st in (0..w.len().saturating_sub(29_800)).step_by(2_048) {
        let e: f64 = w[st..st + 4_096]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        if e > be {
            be = e;
            bs = st;
        }
    }
    let mlen = 4_096.min(w.len().saturating_sub(bs));
    let max_lag = 28_800.min(wd.len().saturating_sub(bs + mlen));
    if be <= 1.0e-8 || mlen < 2_048 || max_lag <= 760 {
        return Vec::new();
    }
    let dry_e: f64 = wdry[bs..bs + mlen]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .max(1.0e-12);
    let dry_rms = (dry_e / (mlen as f64)).sqrt().max(1.0e-6);
    scan_peaks_shared(&PeakScanSpec {
        motif: &w[bs..bs + mlen],
        motif_e: be,
        rms_m: (be / (mlen as f64)).sqrt().max(1.0e-6),
        tgt: &wd[bs..],
        start_off: 720,
        max_off: max_lag,
        c_step: 8,
        lag_base: 0,
        c_th: 0.14,
        f_th: 0.18,
        g_th: 0.02,
        dry_check: Some((&wdry[bs..bs + mlen], dry_e, dry_rms, &wdry[bs..])),
    })
}

fn estimate_cross_feedback(
    fit: &mut TailDecayStats,
    comb_wet_l: &[f32],
    tail_wet_r: Option<&[f32]>,
) {
    let Some(tr) = tail_wet_r.filter(|_| fit.delay_l_samples > 1.0 && fit.bbd_gain_l > 0.02) else {
        return;
    };
    let mlen = comb_wet_l.len().min(1_024);
    let off_r = (fit.delay_l_samples.round() as usize).saturating_sub(mlen);
    if off_r + mlen > tr.len() {
        return;
    }
    let motif = &comb_wet_l[comb_wet_l.len() - mlen..];
    let me: f64 = motif
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .max(1.0e-12);
    let (_, s_lr, g_lr) = corr_at_offset(motif, me, tr, off_r, rms_energy(motif), 1);
    if s_lr.abs() > 0.25 {
        fit.bbd_cross_fb = (s_lr.signum() * (g_lr / fit.bbd_gain_l.max(1.0e-6))).clamp(-0.92, 0.92);
    }
}

fn fallback_envelope_peaks(
    fit: &mut TailDecayStats,
    comb_dry: &[f32],
    env_ms: &[f64],
    comb_rms: f64,
) {
    if !fit.delays_ms.is_empty() || env_ms.len() <= 60 {
        return;
    }
    let motif_len_ms = (comb_dry.len().min(1_024) as f32) / 96.0;
    for m in 35..env_ms.len().saturating_sub(5) {
        let cur = env_ms[m];
        let prev = env_ms[m - 8..m].iter().sum::<f64>() / 8.0;
        if cur > 0.05 * comb_rms
            && cur > 1.35 * prev
            && cur >= env_ms[m - 1]
            && cur >= env_ms[m + 1]
        {
            fit.delays_ms.push((m as f32) + motif_len_ms);
            fit.decay_gains
                .push((cur / comb_rms).clamp(0.0, 1.2) as f32);
            if fit.delays_ms.len() >= 4 {
                break;
            }
        }
    }
}

/// Extracts primary/secondary echo lags, feedback ratios, stereo cross-feedback, and envelope peaks.
pub(super) fn extract_echo_peaks(input: &EchoPeakInput<'_>) -> TailDecayStats {
    let mut fit = TailDecayStats {
        bbd_hf_retain: 1.0,
        ..Default::default()
    };
    if input.env_ms.len() < 40 {
        return fit;
    }
    let t = input.tail;
    let mut peaks_l = find_motif_peaks_fine(t.comb_wet_l, t.tail_wet_l, 720);
    if peaks_l.is_empty() {
        peaks_l = find_motif_peaks_fine(t.comb_dry, t.tail_wet_l, 720);
    }
    let mut peaks_r = t.tail_wet_r.map_or_else(Vec::new, |tr| {
        let pr = find_motif_peaks_fine(t.comb_wet_r.unwrap_or(t.comb_wet_l), tr, 720);
        if pr.is_empty() {
            find_motif_peaks_fine(t.comb_wet_l, tr, 720)
        } else {
            pr
        }
    });
    if peaks_l.is_empty() && input.tail_energy_ratio > 0.04 {
        peaks_l = find_whitened_lag_peaks(t.cs_dry_l, t.cs_wet_l, t.cs_wet_l);
        if peaks_r.is_empty()
            && let Some(csr) = t.cs_wet_r
        {
            peaks_r = find_whitened_lag_peaks(t.cs_dry_l, t.cs_wet_l, csr);
        }
    }
    if let Some(&(lag1, _, _, g1)) = peaks_l.first() {
        fit.delay_l_samples = lag1;
        fit.bbd_gain_l = g1;
        if let Some(&(_, _, _, g2)) = peaks_l.get(1) {
            fit.bbd_fb = (g2 / g1.max(1.0e-6)).clamp(0.0, 0.96);
        }
    }
    if let Some(&(lag_r1, _, _, g_r1)) = peaks_r.first() {
        fit.delay_r_samples = lag_r1;
        fit.bbd_gain_r = g_r1;
    } else if fit.delay_l_samples > 0.0 {
        fit.delay_r_samples = fit.delay_l_samples;
        fit.bbd_gain_r = fit.bbd_gain_l;
    }
    estimate_cross_feedback(&mut fit, t.comb_wet_l, t.tail_wet_r);
    for &(lag, _, _, g) in &peaks_l {
        fit.delays_ms.push((lag / 48.0) as f32);
        fit.decay_gains.push(g as f32);
    }
    fallback_envelope_peaks(&mut fit, t.comb_dry, input.env_ms, input.comb_rms);
    fit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prng_signal(mut state: u64, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            out.push((u - 0.5) * 0.8);
        }
        out
    }

    #[test]
    fn test_rms_energy_and_first_diff_hf_ratio_guards() {
        assert_eq!(rms_energy(&[]), 0.0);
        assert!((rms_energy(&[0.5, -0.5, 0.5, -0.5]) - 0.5).abs() < 1.0e-9);
        assert_eq!(first_diff_hf_ratio(&[0.1, 0.2, 0.3]), 0.0);
        assert_eq!(first_diff_hf_ratio(&[0.4; 16]), 0.0);
        assert!(first_diff_hf_ratio(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) > 1.7);
    }

    #[test]
    fn test_find_motif_peaks_fine_and_whitened_lag_peaks() {
        assert!(find_motif_peaks_fine(&[0.1; 128], &[0.0; 2_048], 720).is_empty());
        assert!(find_motif_peaks_fine(&[0.1; 1_024], &[0.0; 512], 720).is_empty());
        assert!(find_whitened_lag_peaks(&[0.0; 1_000], &[0.0; 1_000], &[0.0; 1_000]).is_empty());
        assert!(find_whitened_lag_peaks(&[0.0; 8_000], &[0.0; 8_000], &[0.0; 8_000]).is_empty());

        let comb_motif = make_prng_signal(0x19_28_37_46, 1_536);
        let tail_len = 8_000;
        let mut tail = vec![0.0_f32; tail_len];
        let lag1 = 1_200_usize;
        let lag2 = 2_800_usize;
        let mlen = 1_024_usize;
        let motif_src = &comb_motif[comb_motif.len() - mlen..];
        for i in 0..mlen {
            tail[lag1 - mlen + i] += 0.65 * motif_src[i];
            tail[lag2 - mlen + i] += 0.35 * motif_src[i];
        }
        let peaks = find_motif_peaks_fine(&comb_motif, &tail, 720);
        assert_eq!(peaks.len(), 2);
        assert!((peaks[0].0 - (lag1 as f64)).abs() < 2.0);
        assert!((peaks[1].0 - (lag2 as f64)).abs() < 2.0);

        // Whitened lag peaks across 36_000 samples with wet echo absent from dry_src
        let dry_full = make_prng_signal(0x55_66_77_88, 36_000);
        let src_burst = make_prng_signal(0x99_AA_BB_CC, 4_500);
        let mut src = vec![0.0_f32; 36_000];
        src[1_000..5_500].copy_from_slice(&src_burst);
        let echo_lag = 1_440_usize;
        let mut dst = src.clone();
        for i in echo_lag..36_000 {
            dst[i] += 0.55 * src[i - echo_lag];
        }
        let w_peaks = find_whitened_lag_peaks(&dry_full, &src, &dst);
        assert!(!w_peaks.is_empty());
        assert!((w_peaks[0].0 - (echo_lag as f64)).abs() < 4.0);
    }

    #[test]
    fn test_extract_echo_peaks_stereo_cross_fb_and_env_fallback() {
        let comb = make_prng_signal(0x88_77_66_55, 1_536);
        let mlen = 1_024_usize;
        let motif = &comb[comb.len() - mlen..];
        let mut tail_l = vec![0.0_f32; 4_800];
        let mut tail_r = vec![0.0_f32; 4_800];
        let lag1 = 1_100_usize;
        let lag2 = 2_400_usize;
        for i in 0..mlen {
            tail_l[lag1 - mlen + i] += 0.7 * motif[i];
            tail_l[lag2 - mlen + i] += 0.35 * motif[i];
            tail_r[lag1 - mlen + i] -= 0.4 * motif[i];
        }
        let env_long = vec![0.1_f64; 100];
        let tail_in = DecayTailInput {
            cs_dry_l: &[0.0; 16],
            cs_wet_l: &[0.0; 16],
            cs_wet_r: None,
            comb_dry: &comb,
            comb_wet_l: &[0.0; 64],
            comb_wet_r: None,
            tail_wet_l: &tail_l,
            tail_wet_r: Some(&tail_r),
            comb_hf_ratio: 0.2,
        };
        let short_fit = extract_echo_peaks(&EchoPeakInput {
            tail: &tail_in,
            tail_energy_ratio: 0.1,
            env_ms: &[0.1; 10],
            comb_rms: 0.2,
        });
        assert_eq!(short_fit.delays_ms.len(), 0);

        let full_in = DecayTailInput {
            cs_dry_l: &comb,
            cs_wet_l: &comb,
            cs_wet_r: Some(&comb),
            comb_dry: &comb,
            comb_wet_l: &comb,
            comb_wet_r: None,
            tail_wet_l: &tail_l,
            tail_wet_r: Some(&tail_r),
            comb_hf_ratio: 0.2,
        };
        let fit = extract_echo_peaks(&EchoPeakInput {
            tail: &full_in,
            tail_energy_ratio: 0.2,
            env_ms: &env_long,
            comb_rms: rms_energy(&comb),
        });
        assert!((fit.delay_l_samples - (lag1 as f64)).abs() < 2.0);
        assert!(fit.bbd_fb > 0.3);
        assert!(fit.bbd_cross_fb < -0.25);
        assert!(!fit.delays_ms.is_empty());

        // Envelope fallback when motif correlation is silent
        let mut env_spike = vec![0.02_f64; 100];
        env_spike[50] = 0.5;
        let blank = vec![0.0_f32; 4_800];
        let env_in = DecayTailInput {
            cs_dry_l: &blank,
            cs_wet_l: &blank,
            cs_wet_r: None,
            comb_dry: &[0.1; 1_024],
            comb_wet_l: &blank,
            comb_wet_r: None,
            tail_wet_l: &blank,
            tail_wet_r: None,
            comb_hf_ratio: 0.1,
        };
        let env_fit = extract_echo_peaks(&EchoPeakInput {
            tail: &env_in,
            tail_energy_ratio: 0.01,
            env_ms: &env_spike,
            comb_rms: 1.0,
        });
        assert!(!env_fit.delays_ms.is_empty());
    }
}
