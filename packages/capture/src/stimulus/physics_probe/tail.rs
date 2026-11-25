#![forbid(unsafe_code)]
//! Zero-Input Decay Tail Analysis (`RT60`, BBD Darkening, Shimmer, Ping-Pong, and LFO Modulation)
//! (`stimulus::physics_probe::tail`).

use super::echo_peaks::{EchoPeakInput, extract_echo_peaks, first_diff_hf_ratio, rms_energy};

#[derive(Default)]
pub(super) struct TailDecayStats {
    pub(super) delays_ms: Vec<f32>,
    pub(super) decay_gains: Vec<f32>,
    pub(super) delay_l_samples: f64,
    pub(super) delay_r_samples: f64,
    pub(super) bbd_gain_l: f64,
    pub(super) bbd_gain_r: f64,
    pub(super) bbd_fb: f64,
    pub(super) bbd_cross_fb: f64,
    pub(super) bbd_lpf_pole: f64,
    pub(super) rt60_ms: f64,
    pub(super) tail_energy_ratio: f64,
    pub(super) bbd_hf_retain: f64,
    pub(super) shimmer_score: f64,
    pub(super) lfo_hz: f64,
    pub(super) lfo_depth: f64,
    pub(super) pingpong_asym: f64,
}

/// Bundled dry/wet coreset, disjoint-comb, and zero-input tail slices for tail physics analysis.
pub(super) struct DecayTailInput<'a> {
    pub(super) cs_dry_l: &'a [f32],
    pub(super) cs_wet_l: &'a [f32],
    pub(super) cs_wet_r: Option<&'a [f32]>,
    pub(super) comb_dry: &'a [f32],
    pub(super) comb_wet_l: &'a [f32],
    pub(super) comb_wet_r: Option<&'a [f32]>,
    pub(super) tail_wet_l: &'a [f32],
    pub(super) tail_wet_r: Option<&'a [f32]>,
    pub(super) comb_hf_ratio: f64,
}

fn ac_couple_tail(buf: &[f32]) -> Vec<f32> {
    const R: f64 = 0.992;
    let mut out: Vec<f64> = buf.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..2 {
        let (mut y, mut xp) = (0.0_f64, 0.0_f64);
        for slot in &mut out {
            let x = *slot;
            y = R * (y + x - xp);
            xp = x;
            *slot = y;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

/// Analyzes the zero-input decay tail (`RT60`, echo delays, BBD darkening, shimmer, LFO, ping-pong).
pub(super) fn analyze_decay_tail(input: &DecayTailInput<'_>) -> TailDecayStats {
    if input.tail_wet_l.len() < 1_024 || input.comb_wet_l.is_empty() {
        return TailDecayStats {
            bbd_hf_retain: 1.0,
            ..Default::default()
        };
    }
    let comb_rms = rms_energy(input.comb_wet_l).max(1.0e-6);
    let tail_ac_l = ac_couple_tail(input.tail_wet_l);
    let tail_ac_r = input.tail_wet_r.map(ac_couple_tail);
    let skip = (20 * 48).min(tail_ac_l.len() / 2);
    let tail_energy_ratio = (rms_energy(&tail_ac_l[skip..]) / comb_rms).clamp(0.0, 10.0);
    let blk = 48_usize;
    let n_ms = tail_ac_l.len() / blk;
    let env_l: Vec<f64> = (0..n_ms)
        .map(|m| rms_energy(&tail_ac_l[m * blk..(m + 1) * blk]))
        .collect();
    let env_r: Vec<f64> = tail_ac_r
        .as_deref()
        .map(|tr| {
            (0..(tr.len() / blk).min(n_ms))
                .map(|m| rms_energy(&tr[m * blk..(m + 1) * blk]))
                .collect()
        })
        .unwrap_or_default();

    let mut s = extract_echo_peaks(&EchoPeakInput {
        tail: input,
        tail_energy_ratio,
        env_ms: &env_l,
        comb_rms,
    });
    s.rt60_ms = estimate_rt60_ms(&env_l, comb_rms);
    s.tail_energy_ratio = tail_energy_ratio;
    let tail_late = &input.tail_wet_l[(60 * blk).min(input.tail_wet_l.len() / 2)..];
    s.bbd_hf_retain = if rms_energy(tail_late) > 5.0e-4 * comb_rms {
        (first_diff_hf_ratio(tail_late) / first_diff_hf_ratio(input.comb_wet_l).max(1.0e-4))
            .clamp(0.0, 4.0)
    } else {
        1.0
    };
    s.bbd_lpf_pole = if tail_energy_ratio > 0.012 && s.bbd_hf_retain < 0.96 {
        ((1.0 - s.bbd_hf_retain.max(0.05)) * 0.88)
            .sqrt()
            .clamp(0.0, 0.94)
    } else {
        0.0
    };
    s.shimmer_score =
        if s.bbd_hf_retain > 1.25 && tail_energy_ratio > 0.02 && input.comb_hf_ratio > 0.01 {
            ((s.bbd_hf_retain - 1.0) * 0.8 + tail_energy_ratio.min(0.5)).clamp(0.0, 2.0)
        } else {
            0.0
        };
    if tail_energy_ratio > 0.005 {
        (s.lfo_hz, s.lfo_depth) = estimate_lfo_modulation(&env_l);
    }
    if env_r.len() == env_l.len() && env_l.len() > 80 {
        let (ds, ts) = (15..env_l.len().min(500)).fold((0.0_f64, 0.0_f64), |(d, t), m| {
            (d + (env_l[m] - env_r[m]).abs(), t + env_l[m] + env_r[m])
        });
        if ts > 1.0e-9 {
            s.pingpong_asym = (ds / ts).clamp(0.0, 1.0);
        }
    }
    s
}

fn estimate_rt60_ms(env_ms: &[f64], comb_rms: f64) -> f64 {
    if env_ms.len() < 120 {
        return 0.0;
    }
    let post_30_peak = env_ms.iter().skip(30).copied().fold(0.0_f64, f64::max);
    if post_30_peak <= 0.012 * comb_rms {
        return 0.0;
    }
    let thresh = (comb_rms * 1.0e-3).max(1.0e-6);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (m, &e) in env_ms.iter().enumerate().skip(30) {
        if e > thresh {
            xs.push((m as f64) * 1.0e-3);
            ys.push(20.0 * e.log10());
        }
    }
    if xs.len() < 40 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&x, &y) in xs.iter().zip(&ys) {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    if den <= 1.0e-12 {
        return 0.0;
    }
    let slope_db_per_s = num / den;
    if slope_db_per_s < -1.0 {
        ((-60.0 / slope_db_per_s) * 1_000.0).clamp(40.0, 30_000.0)
    } else if env_ms.last().copied().unwrap_or(0.0) > 0.02 * comb_rms {
        15_000.0
    } else {
        0.0
    }
}

/// Estimates dominant LFO rate (`Hz`) and modulation depth across `1 ms` RMS envelope bins.
pub(super) fn estimate_lfo_modulation(env_ms: &[f64]) -> (f64, f64) {
    if env_ms.len() < 200 {
        return (0.0, 0.0);
    }
    let mean = (env_ms.iter().sum::<f64>() / (env_ms.len() as f64)).max(1.0e-9);
    let norm: Vec<f64> = env_ms.iter().map(|&e| (e / mean) - 1.0).collect();
    let var: f64 = norm.iter().map(|&x| x * x).sum::<f64>() / (norm.len() as f64);
    let depth = var.sqrt().clamp(0.0, 1.0);
    if depth < 0.04 {
        return (0.0, depth);
    }
    let (mut best_hz, mut best_pow, mut sum_pow, mut count, mut hz) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0usize, 0.4_f64);
    while hz <= 14.0 {
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        let w = 2.0 * std::f64::consts::PI * hz * 1.0e-3;
        for (t, &x) in norm.iter().enumerate() {
            let (s, c) = (w * (t as f64)).sin_cos();
            re += x * c;
            im += x * s;
        }
        let p = re * re + im * im;
        if p > best_pow {
            best_pow = p;
            best_hz = hz;
        }
        sum_pow += p;
        count += 1;
        hz += 0.25;
    }
    let avg_pow = sum_pow / (count.max(1) as f64);
    if best_pow > 2.8 * avg_pow {
        (best_hz, depth)
    } else {
        (0.0, depth)
    }
}

/// Detects short slapback/doubler reflections (`1.5..22 ms`) and stereo chorus LFO modulation.
pub(super) fn detect_short_delay_and_chorus_lfo(
    dc_l: &[f32],
    wc_l: &[f32],
    coreset_wet_r: Option<&[f32]>,
    comb_wet_l: &[f32],
    comb_wet_r: Option<&[f32]>,
) -> (f64, f64, f64, f64) {
    let n = dc_l.len().min(wc_l.len());
    let (mut short_ms, mut short_gain) = (0.0_f64, 0.0_f64);
    if n >= 2_400 {
        let (mut dx, mut dw) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let (mut xx, mut xw, mut ww) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 1..n {
            dx[i] = f64::from(dc_l[i]) - 0.92 * f64::from(dc_l[i - 1]);
            dw[i] = f64::from(wc_l[i]) - 0.92 * f64::from(wc_l[i - 1]);
            xx += dx[i] * dx[i];
            xw += dx[i] * dw[i];
            ww += dw[i] * dw[i];
        }
        let (g0, res_norm) = (
            if xx > 1.0e-12 { xw / xx } else { 0.0 },
            (xx * ww).sqrt().max(1.0e-12),
        );
        let (mut best_lag, mut best_score) = (0usize, 0.0_f64);
        for lag in 72..1_056_usize.min(n / 2) {
            let mut dot = 0.0_f64;
            for idx in (lag + 16..n).step_by(2) {
                dot += (dw[idx] - g0 * dx[idx]) * dx[idx - lag];
            }
            let sc = (dot.abs() * 2.0) / res_norm;
            if sc > best_score {
                best_score = sc;
                best_lag = lag;
            }
        }
        if best_score > 0.045 && best_lag >= 72 {
            short_ms = (best_lag as f64) / 48.0;
            short_gain = best_score.clamp(0.0, 1.5);
        }
    }
    let blk = 48_usize;
    let build_side_env = |wl: &[f32], wr: &[f32]| -> Vec<f64> {
        let m_blocks = wl.len().min(wr.len()) / blk;
        (0..m_blocks)
            .map(|m| {
                let de: f64 = (m * blk..(m + 1) * blk)
                    .map(|i| {
                        let d = f64::from(wl[i]) - f64::from(wr[i]);
                        d * d
                    })
                    .sum();
                (de / (blk as f64)).sqrt()
            })
            .collect()
    };
    let core_side = coreset_wet_r
        .map(|wr| build_side_env(wc_l, wr))
        .unwrap_or_default();
    let single_env = if core_side.len() >= 200 {
        core_side
    } else if let Some(cr) = comb_wet_r {
        let comb_side = build_side_env(comb_wet_l, cr);
        if comb_side.len() > core_side.len() {
            comb_side
        } else {
            core_side
        }
    } else {
        core_side
    };
    let (lfo_hz, lfo_depth) = if single_env.len() >= 200 {
        estimate_lfo_modulation(&single_env)
    } else {
        (0.0, 0.0)
    };
    (short_ms, short_gain, lfo_hz, lfo_depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_rt60_and_lfo_modulation_branches() {
        assert_eq!(estimate_rt60_ms(&[0.5; 60], 1.0), 0.0);
        assert_eq!(estimate_rt60_ms(&[0.005; 150], 1.0), 0.0);
        assert_eq!(estimate_rt60_ms(&vec![0.1; 45], 1.0), 0.0);

        // Exponential decay with 1.0s RT60 (-60 dB/s)
        let exp_env: Vec<f64> = (0..400)
            .map(|m| 0.8 * 10.0_f64.powf(-3.0 * (m as f64) * 1.0e-3))
            .collect();
        let rt60 = estimate_rt60_ms(&exp_env, 1.0);
        assert!((rt60 - 1_000.0).abs() < 40.0);

        // Nearly infinite freeze tail returns 15_000 ms
        let flat_tail = vec![0.3_f64; 180];
        assert_eq!(estimate_rt60_ms(&flat_tail, 1.0), 15_000.0);

        // LFO modulation branches
        assert_eq!(estimate_lfo_modulation(&[1.0; 100]), (0.0, 0.0));
        assert_eq!(estimate_lfo_modulation(&[1.0; 250]).0, 0.0);
        let mod_env: Vec<f64> = (0..500)
            .map(|m| 1.0 + 0.35 * (2.0 * std::f64::consts::PI * 2.5 * (m as f64) * 1.0e-3).sin())
            .collect();
        let (hz, depth) = estimate_lfo_modulation(&mod_env);
        assert!((hz - 2.5).abs() < 0.4);
        assert!(depth > 0.2);

        // Impulsive spike lacks narrowband periodicity
        let mut spike = vec![1.0_f64; 300];
        spike[150] = 5.0;
        let (hz_sp, dep_sp) = estimate_lfo_modulation(&spike);
        assert_eq!(hz_sp, 0.0);
        assert!(dep_sp > 0.04);
    }

    #[test]
    fn test_analyze_decay_tail_dark_shimmer_pingpong_and_short_delay() {
        let short = analyze_decay_tail(&DecayTailInput {
            cs_dry_l: &[0.0; 16],
            cs_wet_l: &[0.0; 16],
            cs_wet_r: None,
            comb_dry: &[0.1; 128],
            comb_wet_l: &[],
            comb_wet_r: None,
            tail_wet_l: &[0.0; 512],
            tail_wet_r: None,
            comb_hf_ratio: 0.2,
        });
        assert_eq!(short.bbd_hf_retain, 1.0);

        // Build comb + long Nyquist-shimmering tail with LFO & R-channel ping-pong asymmetry
        let n_comb = 4_608;
        let comb: Vec<f32> = (0..n_comb)
            .map(|i| 0.4 * ((i as f32) * 0.04).sin())
            .collect();
        let n_tail = 14_400;
        let mut tail_l = vec![0.0_f32; n_tail];
        let mut tail_r = vec![0.0_f32; n_tail];
        for (i, (l, r)) in tail_l.iter_mut().zip(&mut tail_r).enumerate() {
            let env_l =
                0.5 + 0.3 * (2.0 * std::f32::consts::PI * 3.0 * (i as f32) / 48_000.0).sin();
            let hf = if i % 2 == 0 { 0.35 } else { -0.35 };
            *l = env_l * hf;
            *r = 0.1 * hf;
        }
        let shim_stats = analyze_decay_tail(&DecayTailInput {
            cs_dry_l: &comb,
            cs_wet_l: &comb,
            cs_wet_r: None,
            comb_dry: &comb,
            comb_wet_l: &comb,
            comb_wet_r: None,
            tail_wet_l: &tail_l,
            tail_wet_r: Some(&tail_r),
            comb_hf_ratio: 0.15,
        });
        assert!(shim_stats.bbd_hf_retain > 1.25);
        assert!(shim_stats.shimmer_score > 0.0);
        assert!(shim_stats.pingpong_asym > 0.3);
        assert!((shim_stats.lfo_hz - 3.0).abs() < 0.6);

        // Dark LPF tail
        let mut dark_tail = vec![0.0_f32; n_tail];
        let mut lp = 0.0_f32;
        for (i, slot) in dark_tail.iter_mut().enumerate() {
            let x = 0.3 * ((i as f32) * 0.015).sin();
            lp = 0.96 * lp + 0.04 * x;
            *slot = lp;
        }
        let comb_bright: Vec<f32> = (0..n_comb)
            .map(|i| if i % 2 == 0 { 0.4 } else { -0.4 })
            .collect();
        let dark_stats = analyze_decay_tail(&DecayTailInput {
            cs_dry_l: &comb_bright,
            cs_wet_l: &comb_bright,
            cs_wet_r: None,
            comb_dry: &comb_bright,
            comb_wet_l: &comb_bright,
            comb_wet_r: None,
            tail_wet_l: &dark_tail,
            tail_wet_r: None,
            comb_hf_ratio: 0.8,
        });
        assert!(dark_stats.bbd_lpf_pole > 0.3);

        // Short slapback delay + stereo chorus LFO detection
        assert_eq!(
            detect_short_delay_and_chorus_lfo(&[0.0; 100], &[0.0; 100], None, &[], None),
            (0.0, 0.0, 0.0, 0.0)
        );
        let mut dry = vec![0.0_f32; 14_400];
        let mut rng_state = 0x2E_44_66_88_u64;
        for slot in &mut dry {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u = ((rng_state >> 40) as f32) / ((1u32 << 24) as f32);
            *slot = (u - 0.5) * 0.6;
        }
        // Normalize each 48-sample 1 ms block so side-difference RMS reflects pure LFO envelope
        for blk in dry.as_chunks_mut::<48>().0 {
            let rms = (blk.iter().map(|&v| v * v).sum::<f32>() / 48.0)
                .sqrt()
                .max(1.0e-6);
            for s in blk {
                *s = (*s / rms) * 0.25;
            }
        }
        let mut wet_l = dry.clone();
        for i in 360..dry.len() {
            wet_l[i] += 0.45 * dry[i - 360];
        }
        let mut wet_r = wet_l.clone();
        for (i, (&d, r)) in dry.iter().zip(&mut wet_r).enumerate() {
            let lfo =
                0.45 + 0.32 * (2.0 * std::f32::consts::PI * 2.4 * (i as f32) / 48_000.0).sin();
            *r += lfo * d;
        }
        let (short_ms, short_gain, chorus_hz, _) =
            detect_short_delay_and_chorus_lfo(&dry, &wet_l, Some(&wet_r), &[], None);
        assert!((short_ms - 7.5).abs() < 0.5);
        assert!(short_gain > 0.1);
        assert!((chorus_hz - 2.4).abs() < 0.3);

        // Static stereo difference across coreset + differently scaled comb must NOT concatenate
        // into a step discontinuity triggering false chorus LFO detection.
        let static_wet_r: Vec<f32> = wet_l
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let prev = if i > 12 { dry[i - 12] } else { 0.0 };
                v + 0.35 * prev
            })
            .collect();
        let comb_l_scaled: Vec<f32> = dry[..4_608].iter().map(|&v| 2.5 * v).collect();
        let comb_r_scaled: Vec<f32> = comb_l_scaled
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let prev = if i > 18 { comb_l_scaled[i - 18] } else { 0.0 };
                v - 0.40 * prev
            })
            .collect();
        let (_, _, static_lfo_hz, _) = detect_short_delay_and_chorus_lfo(
            &dry,
            &wet_l,
            Some(&static_wet_r),
            &comb_l_scaled,
            Some(&comb_r_scaled),
        );
        assert_eq!(static_lfo_hz, 0.0);

        // Sub--46 dB (`tail_energy_ratio <= 0.005`) oscillating tail noise must NOT trigger LFO detection.
        let mut faint_osc = vec![0.0_f32; n_tail];
        for (i, dst) in faint_osc.iter_mut().enumerate() {
            let env = 0.5 + 0.45 * (2.0 * std::f32::consts::PI * 2.5 * (i as f32) / 48_000.0).sin();
            *dst = 2.0e-4 * env * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let comb_hi = vec![0.3_f32; n_comb];
        let faint_stats = analyze_decay_tail(&DecayTailInput {
            cs_dry_l: &comb_hi,
            cs_wet_l: &comb_hi,
            cs_wet_r: None,
            comb_dry: &comb_hi,
            comb_wet_l: &comb_hi,
            comb_wet_r: None,
            tail_wet_l: &faint_osc,
            tail_wet_r: None,
            comb_hf_ratio: 0.2,
        });
        assert!(faint_stats.tail_energy_ratio <= 0.005);
        assert_eq!(faint_stats.lfo_hz, 0.0);
        assert_eq!(faint_stats.lfo_depth, 0.0);
    }
}
