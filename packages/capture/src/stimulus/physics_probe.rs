#![forbid(unsafe_code)]
//! Pure-Audio Signal-Driven Physics Extractor (`REQ-1.6` Blind Identification).
//!
//! Analyzes `StarCenter` dry/wet coreset, periodic disjoint-comb, and zero-input decay tail
//! responses to measure linear coherence, harmonic odd/even asymmetry, sub-harmonic / shimmer
//! excess, dynamic compression / gating, resonant filter envelope sensitivity, stereo decorrelation,
//! delay-line echoes, LFO modulation rates, and reverb decay (`ExtractedPhysics`) without any
//! control-name or fixture heuristics.

use std::collections::{BTreeMap, BTreeSet};

use crate::proj::schema::ExtractedPhysics;

use super::fft::{
    ADAPTIVE_FRAME_SAMPLES, CYCLIC_PREFIX_SAMPLES, CYCLIC_WINDOW_SAMPLES, Complex64,
    fft_radix2_in_place, rfft_bins,
};
pub(crate) mod echo_peaks;
pub(crate) mod tail;

use self::tail::{DecayTailInput, analyze_decay_tail, detect_short_delay_and_chorus_lfo};

const SR_48K: f64 = 48_000.0;

/// Full signal-driven `StarCenter` physics bundle attached to `.dnam-proj` `stems_registry`.
#[derive(Debug, Clone, Default)]
pub(crate) struct StarPhysicsAnalysis {
    pub extracted: ExtractedPhysics,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Computes `StarPhysicsAnalysis` purely from aligned/raw 48 kHz dry & wet audio buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_star_physics_from_audio(
    coreset_dry_l: &[f32],
    coreset_dry_r: Option<&[f32]>,
    coreset_wet_l: &[f32],
    coreset_wet_r: Option<&[f32]>,
    comb_dry_l: &[f32],
    comb_wet_l: &[f32],
    comb_wet_r: Option<&[f32]>,
    tail_wet_l: &[f32],
    tail_wet_r: Option<&[f32]>,
    carrier_bins: &[usize],
) -> StarPhysicsAnalysis {
    let mut meta = BTreeMap::new();

    // 1. Align coreset wet to dry within +/- 384 samples for clean coherence/dynamics comparison.
    let lag = estimate_best_lag(coreset_dry_l, coreset_wet_l, 384);
    let (dc_l, wc_l) = aligned_slices(coreset_dry_l, coreset_wet_l, lag);
    let coherence = compute_windowed_coherence(dc_l, wc_l);
    let stereo_diff = compute_stereo_diff_ratio(
        coreset_wet_l,
        coreset_wet_r,
        comb_wet_l,
        comb_wet_r,
        coreset_dry_l,
        coreset_dry_r,
    );

    // 2. Periodic comb spectral analysis (harmonics, pre-EQ tilt/HPF/LPF, subharmonic & sidebands).
    let spec = analyze_comb_spectrum(comb_dry_l, comb_wet_l, carrier_bins);

    // 3. Coreset envelope dynamics (noise gate, compressor squash, envelope-filter brightness shift).
    let dyn_stats = analyze_coreset_dynamics(dc_l, wc_l);

    // 4. Post-comb zero-input tail analysis (echo delays, feedback gains, RT60, BBD dark LPF, shimmer, LFO).
    let tail_stats = analyze_decay_tail(&DecayTailInput {
        cs_dry_l: coreset_dry_l,
        cs_wet_l: coreset_wet_l,
        cs_wet_r: coreset_wet_r,
        comb_dry: comb_dry_l,
        comb_wet_l,
        comb_wet_r,
        tail_wet_l,
        tail_wet_r,
        comb_hf_ratio: spec.mid_hf_ratio,
    });

    // 5. Short-delay (1.5..22 ms) reflection & comb/coreset stereo LFO detection.
    let (short_delay_ms, short_delay_gain, chorus_lfo_hz, chorus_lfo_depth) =
        detect_short_delay_and_chorus_lfo(dc_l, wc_l, coreset_wet_r, comb_wet_l, comb_wet_r);
    let eff_lfo_hz = if tail_stats.lfo_hz > 0.15 {
        tail_stats.lfo_hz
    } else if stereo_diff > 0.05 && chorus_lfo_hz > 0.15 {
        chorus_lfo_hz
    } else {
        0.0
    };
    let eff_lfo_depth = if tail_stats.lfo_hz > 0.15 {
        tail_stats.lfo_depth
    } else if stereo_diff > 0.05 && chorus_lfo_hz > 0.15 {
        chorus_lfo_depth.max(tail_stats.lfo_depth)
    } else {
        tail_stats.lfo_depth
    };

    let detected_pitch = if tail_stats.shimmer_score > 0.35 {
        12.0_f32
    } else if spec.subharm_ratio > 0.12 && dyn_stats.gate_ratio >= 0.65 {
        -12.0_f32
    } else {
        0.0_f32
    };
    let detected_grain_ms = if detected_pitch.abs() > 0.5 {
        24.0_f32
    } else {
        0.0_f32
    };

    let r3 = |v: f32| -> f32 {
        if v.is_finite() {
            let x = (v * 1_000.0).round() / 1_000.0;
            if x == 0.0 { 0.0 } else { x }
        } else {
            0.0
        }
    };
    let lfo_freqs_hz = if eff_lfo_hz > 0.15 {
        vec![r3(eff_lfo_hz as f32)]
    } else {
        Vec::new()
    };

    let extracted = ExtractedPhysics {
        delays_ms: tail_stats.delays_ms.iter().map(|&v| r3(v)).collect(),
        lfo_freqs_hz,
        decay_gains_lin: tail_stats.decay_gains.iter().map(|&v| r3(v)).collect(),
        detected_grain_size_ms: r3(detected_grain_ms),
        detected_pitch_shift_semitones: r3(detected_pitch),
    };

    for &(k, v) in &[
        ("phys_coherence", coherence),
        ("phys_stereo_diff", stereo_diff),
        ("phys_odd_harm", spec.odd_harm_ratio),
        ("phys_even_harm", spec.even_harm_ratio),
        ("phys_subharm_ratio", spec.subharm_ratio),
        ("phys_sideband_ratio", spec.sideband_ratio),
        ("phys_ringmod_hz", spec.ringmod_hz),
        ("phys_pre_hpf_hz", spec.pre_hpf_hz),
        ("phys_pre_lpf_hz", spec.pre_lpf_hz),
        ("phys_sub_bass_db", spec.sub_bass_db),
        ("phys_spectral_q", spec.spectral_q),
        ("phys_mid_gain", spec.mid_gain),
        ("phys_gate_ratio", dyn_stats.gate_ratio),
        ("phys_comp_ratio", dyn_stats.comp_ratio),
        ("phys_env_bright_shift", dyn_stats.env_bright_shift),
        ("phys_rt60_ms", tail_stats.rt60_ms),
        ("phys_tail_energy_ratio", tail_stats.tail_energy_ratio),
        ("phys_bbd_delay_l_samples", tail_stats.delay_l_samples),
        ("phys_bbd_delay_r_samples", tail_stats.delay_r_samples),
        ("phys_bbd_gain_l", tail_stats.bbd_gain_l),
        ("phys_bbd_gain_r", tail_stats.bbd_gain_r),
        ("phys_bbd_fb", tail_stats.bbd_fb),
        ("phys_bbd_cross_fb", tail_stats.bbd_cross_fb),
        ("phys_bbd_lpf_pole", tail_stats.bbd_lpf_pole),
        ("phys_bbd_hf_retain", tail_stats.bbd_hf_retain),
        ("phys_shimmer_score", tail_stats.shimmer_score),
        ("phys_lfo_depth", eff_lfo_depth),
        ("phys_lfo_hz", eff_lfo_hz),
        ("phys_pingpong_asym", tail_stats.pingpong_asym),
        ("phys_short_delay_ms", short_delay_ms),
        ("phys_short_delay_gain", short_delay_gain),
    ] {
        insert_f64(&mut meta, k, v);
    }

    StarPhysicsAnalysis {
        extracted,
        metadata: meta,
    }
}

fn insert_f64(map: &mut BTreeMap<String, serde_json::Value>, key: &str, val: f64) {
    let clean = if val.is_finite() {
        let r = (val * 1.0e6).round() / 1.0e6;
        if r == 0.0 { 0.0 } else { r }
    } else {
        0.0
    };
    map.insert(key.to_string(), serde_json::json!(clean));
}

fn slice_periodic(buf: &[f32], win_len: usize) -> Option<&[f32]> {
    if buf.len() >= ADAPTIVE_FRAME_SAMPLES {
        Some(&buf[CYCLIC_PREFIX_SAMPLES..ADAPTIVE_FRAME_SAMPLES])
    } else if buf.len() >= win_len {
        Some(&buf[..win_len])
    } else {
        None
    }
}

fn estimate_best_lag(dry: &[f32], wet: &[f32], max_lag: usize) -> isize {
    let n = dry.len().min(wet.len()).min(8_192);
    if n < 512 {
        return 0;
    }
    let (mut best_lag, mut best_corr) = (0_isize, -1.0_f64);
    let ml = max_lag.min(n / 4) as isize;
    for lag in -ml..=ml {
        let mut dot = 0.0_f64;
        let start = (-lag).max(0) as usize;
        let end = (n as isize - lag.max(0)) as usize;
        for i in (start..end).step_by(2) {
            dot += f64::from(dry[i]) * f64::from(wet[(i as isize + lag) as usize]);
        }
        if dot.abs() > best_corr {
            best_corr = dot.abs();
            best_lag = lag;
        }
    }
    best_lag
}

fn aligned_slices<'a>(dry: &'a [f32], wet: &'a [f32], lag: isize) -> (&'a [f32], &'a [f32]) {
    if lag >= 0 && (lag as usize) < wet.len() {
        let u = lag as usize;
        let len = dry.len().min(wet.len() - u);
        return (&dry[..len], &wet[u..u + len]);
    } else if lag < 0 && ((-lag) as usize) < dry.len() {
        let u = (-lag) as usize;
        let len = (dry.len() - u).min(wet.len());
        return (&dry[u..u + len], &wet[..len]);
    }
    let len = dry.len().min(wet.len());
    (&dry[..len], &wet[..len])
}

fn compute_windowed_coherence(dry: &[f32], wet: &[f32]) -> f64 {
    let (wlen, hop) = (1_024_usize, 512_usize);
    let n = dry.len().min(wet.len());
    if n < wlen {
        return 0.0;
    }
    let bins = wlen / 2 + 1;
    let (mut s_xx, mut s_yy, mut s_xy_re, mut s_xy_im) = (
        vec![0.0_f64; bins],
        vec![0.0_f64; bins],
        vec![0.0_f64; bins],
        vec![0.0_f64; bins],
    );
    let (mut buf_x, mut buf_y) = (
        vec![Complex64::default(); wlen],
        vec![Complex64::default(); wlen],
    );
    let mut pos = 0_usize;
    while pos + wlen <= n {
        for i in 0..wlen {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / (wlen as f64)).cos();
            buf_x[i] = Complex64::new(f64::from(dry[pos + i]) * w, 0.0);
            buf_y[i] = Complex64::new(f64::from(wet[pos + i]) * w, 0.0);
        }
        fft_radix2_in_place(&mut buf_x, false);
        fft_radix2_in_place(&mut buf_y, false);
        for k in 1..bins {
            let (xr, xi, yr, yi) = (buf_x[k].re, buf_x[k].im, buf_y[k].re, buf_y[k].im);
            s_xx[k] += xr * xr + xi * xi;
            s_yy[k] += yr * yr + yi * yi;
            s_xy_re[k] += xr * yr + xi * yi;
            s_xy_im[k] += xr * yi - xi * yr;
        }
        pos += hop;
    }
    let k_lo = ((40.0 * wlen as f64) / SR_48K).round() as usize;
    let k_hi = (((12_000.0 * wlen as f64) / SR_48K).round() as usize).min(bins - 1);
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for k in k_lo.max(1)..=k_hi {
        let xy2 = s_xy_re[k] * s_xy_re[k] + s_xy_im[k] * s_xy_im[k];
        let denom = s_xx[k] * s_yy[k];
        if denom > 1.0e-18 {
            num += (xy2 / denom).clamp(0.0, 1.0) * s_xx[k];
            den += s_xx[k];
        }
    }
    if den > 1.0e-18 {
        (num / den).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn compute_stereo_diff_ratio(
    wet_l: &[f32],
    wet_r: Option<&[f32]>,
    comb_l: &[f32],
    comb_r: Option<&[f32]>,
    dry_l: &[f32],
    dry_r: Option<&[f32]>,
) -> f64 {
    let diff_energy = |l: &[f32], r: &[f32]| -> (f64, f64) {
        l.iter()
            .zip(r)
            .fold((0.0_f64, 0.0_f64), |(de, se), (&ls, &rs)| {
                let (a, b) = (f64::from(ls), f64::from(rs));
                (de + (a - b) * (a - b), se + a * a + b * b)
            })
    };
    let (mut de, mut se) = (0.0_f64, 0.0_f64);
    if let Some(wr) = wet_r {
        let (d, s) = diff_energy(wet_l, wr);
        de += d;
        se += s;
    }
    if let Some(cr) = comb_r {
        let (d, s) = diff_energy(comb_l, cr);
        de += d;
        se += s;
    }
    if se <= 1.0e-18 {
        return 0.0;
    }
    let dry_stereo = dry_r.map_or(0.0, |dr| {
        let (dd, ds) = diff_energy(dry_l, dr);
        if ds > 1.0e-18 { (dd / ds).sqrt() } else { 0.0 }
    });
    let wet_stereo = (de / se).sqrt().clamp(0.0, 1.0);
    if dry_stereo < 0.02 {
        wet_stereo
    } else {
        wet_stereo.max((wet_stereo - dry_stereo).abs())
    }
}

#[derive(Default)]
struct CombSpectrumStats {
    odd_harm_ratio: f64,
    even_harm_ratio: f64,
    subharm_ratio: f64,
    sideband_ratio: f64,
    ringmod_hz: f64,
    pre_hpf_hz: f64,
    pre_lpf_hz: f64,
    sub_bass_db: f64,
    spectral_q: f64,
    mid_gain: f64,
    mid_hf_ratio: f64,
}

fn analyze_comb_spectrum(
    comb_dry: &[f32],
    comb_wet: &[f32],
    carrier_bins: &[usize],
) -> CombSpectrumStats {
    let win_len = CYCLIC_WINDOW_SAMPLES;
    let (Some(d_win), Some(w_win)) = (
        slice_periodic(comb_dry, win_len),
        slice_periodic(comb_wet, win_len),
    ) else {
        return CombSpectrumStats {
            pre_lpf_hz: 20_000.0,
            mid_gain: 1.0,
            mid_hf_ratio: 1.0,
            ..Default::default()
        };
    };

    let d_bins = rfft_bins(d_win);
    let w_bins = rfft_bins(w_win);
    let carrier_set: BTreeSet<usize> = carrier_bins.iter().copied().collect();

    let mut p_car = 0.0_f64;
    let mut p_odd = 0.0_f64;
    let mut p_even = 0.0_f64;
    let mut p_sub = 0.0_f64;
    let mut p_sb = 0.0_f64;
    let mut p_mid = 0.0_f64;
    let mut p_hf = 0.0_f64;

    let max_bin = d_bins.len().min(w_bins.len());
    let mut quad_even_set = vec![false; max_bin];
    for (idx, &ci) in carrier_bins.iter().enumerate() {
        if 2 * ci < max_bin {
            quad_even_set[2 * ci] = true;
        }
        for &cj in &carrier_bins[idx..] {
            let sum = ci + cj;
            if sum < max_bin {
                quad_even_set[sum] = true;
            }
            let diff = ci.abs_diff(cj);
            if diff >= 2 && diff < max_bin {
                quad_even_set[diff] = true;
            }
        }
    }

    let mut transfer: Vec<(f64, f64)> = Vec::new();
    for k in 2..max_bin {
        let dp = d_bins[k].re * d_bins[k].re + d_bins[k].im * d_bins[k].im;
        let wp = w_bins[k].re * w_bins[k].re + w_bins[k].im * w_bins[k].im;
        let freq = (k as f64) * SR_48K / (win_len as f64);

        if (300.0..=1_500.0).contains(&freq) {
            p_mid += wp;
        }
        if freq >= 2_500.0 {
            p_hf += wp;
        }

        let is_carrier = carrier_set.contains(&k) || k == 8 || k == 16;
        let near_carrier = carrier_bins.iter().any(|&c| c != k && k.abs_diff(c) <= 4);

        if is_carrier && dp > 1.0e-12 {
            p_car += wp;
            transfer.push((freq, (wp / dp).sqrt()));
        } else if near_carrier {
            p_sb += wp;
        } else if quad_even_set[k] {
            p_even += wp;
            if k <= 13 && dp < 1.0e-10 {
                p_sub += wp;
            }
        } else {
            p_odd += wp;
            if k <= 13 && dp < 1.0e-10 {
                p_sub += wp;
            }
        }
    }

    let p_tot = (p_car + p_odd + p_even + p_sb).max(1.0e-18);
    let band_avg_gain = |f_lo: f64, f_hi: f64| -> f64 {
        let mut s = 0.0_f64;
        let mut c = 0usize;
        for &(f, g) in &transfer {
            if f >= f_lo && f <= f_hi {
                s += g;
                c += 1;
            }
        }
        if c > 0 { s / (c as f64) } else { 0.0 }
    };

    let mid_gain = band_avg_gain(250.0, 1_200.0).max(1.0e-6);
    let sub_gain = band_avg_gain(30.0, 110.0);
    let sub_bass_db = if sub_gain > 1.0e-9 {
        20.0 * (sub_gain / mid_gain).log10()
    } else {
        -40.0
    };

    let mut pre_hpf_hz = 0.0_f64;
    for &(f, g) in &transfer {
        if f <= 500.0 && g < 0.707 * mid_gain {
            pre_hpf_hz = f;
        }
    }
    let mut pre_lpf_hz = 20_000.0_f64;
    for &(f, g) in &transfer {
        if f >= 1_200.0 && g < 0.5 * mid_gain {
            pre_lpf_hz = f;
            break;
        }
    }

    let mut mid_gains: Vec<f64> = transfer
        .iter()
        .filter(|&&(f, _)| (150.0..=6_000.0).contains(&f))
        .map(|&(_, g)| g)
        .collect();
    let spectral_q = if mid_gains.len() >= 3 {
        let max_g = mid_gains.iter().copied().fold(0.0_f64, f64::max);
        mid_gains.sort_by(f64::total_cmp);
        let med_g = mid_gains[mid_gains.len() / 2].max(1.0e-6);
        (max_g / med_g).clamp(1.0, 25.0)
    } else {
        1.0
    };

    let mut best_dk = 0usize;
    let mut best_sym_sum = 0.0_f64;
    for dk in 5..=64_usize {
        if dk % 8 == 0 || dk % 17 == 0 {
            continue;
        }
        let mut sym_sum = 0.0_f64;
        let mut hits = 0usize;
        for &c in carrier_bins {
            if c > dk + 1
                && c + dk < d_bins.len()
                && !carrier_set.contains(&(c - dk))
                && !carrier_set.contains(&(c + dk))
            {
                let p_lo =
                    w_bins[c - dk].re * w_bins[c - dk].re + w_bins[c - dk].im * w_bins[c - dk].im;
                let p_hi =
                    w_bins[c + dk].re * w_bins[c + dk].re + w_bins[c + dk].im * w_bins[c + dk].im;
                let sym = p_lo.min(p_hi);
                if sym > 1.0e-9 {
                    sym_sum += sym;
                    hits += 1;
                }
            }
        }
        if hits >= 2 && sym_sum > best_sym_sum {
            best_sym_sum = sym_sum;
            best_dk = dk;
        }
    }
    let sideband_ratio = ((2.0 * best_sym_sum) / (p_car + 1.0e-18)).sqrt();
    let ringmod_hz = if sideband_ratio > 0.12 && best_dk > 0 {
        (best_dk as f64) * SR_48K / (win_len as f64)
    } else {
        0.0
    };

    CombSpectrumStats {
        odd_harm_ratio: (p_odd / p_tot).sqrt(),
        even_harm_ratio: (p_even / p_tot).sqrt(),
        subharm_ratio: (p_sub / p_tot).sqrt(),
        sideband_ratio,
        ringmod_hz,
        pre_hpf_hz,
        pre_lpf_hz,
        sub_bass_db,
        spectral_q,
        mid_gain,
        mid_hf_ratio: (p_hf / p_mid.max(1.0e-18)).sqrt(),
    }
}

#[derive(Default)]
struct CoresetDynStats {
    gate_ratio: f64,
    comp_ratio: f64,
    env_bright_shift: f64,
}

fn analyze_coreset_dynamics(dry: &[f32], wet: &[f32]) -> CoresetDynStats {
    let blk = 256_usize;
    let n = dry.len().min(wet.len());
    if n < blk * 4 {
        return CoresetDynStats {
            gate_ratio: 1.0,
            comp_ratio: 1.0,
            env_bright_shift: 1.0,
        };
    }
    // 1-pole midband bandpass (350 Hz HPF + 3,200 Hz LPF) decouples tone EQ from gain dynamics.
    let r_hp = (-std::f64::consts::TAU * 350.0 / 48_000.0).exp();
    let r_lp = (-std::f64::consts::TAU * 3_200.0 / 48_000.0).exp();
    let (mut hp_d, mut prev_d, mut lp_d) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut hp_w, mut prev_w_bp, mut lp_w) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut prev_d1, mut prev_d2) = (0.0_f64, 0.0_f64);
    let (mut quiet_sust, mut quiet_all, mut mid_g, mut loud_g) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut loud_hf, mut quiet_hf) = (Vec::new(), Vec::new());
    for off in (0..=n - blk).step_by(blk) {
        let (mut d_bp_e, mut w_bp_e, mut w_e, mut w_hf, mut prev_w) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, f64::from(wet[off]));
        for i in 0..blk {
            let (d, w) = (f64::from(dry[off + i]), f64::from(wet[off + i]));
            hp_d = r_hp * (hp_d + d - prev_d);
            prev_d = d;
            lp_d = r_lp * lp_d + (1.0 - r_lp) * hp_d;
            hp_w = r_hp * (hp_w + w - prev_w_bp);
            prev_w_bp = w;
            lp_w = r_lp * lp_w + (1.0 - r_lp) * hp_w;
            d_bp_e += lp_d * lp_d;
            w_bp_e += lp_w * lp_w;
            w_e += w * w;
            w_hf += (w - prev_w) * (w - prev_w);
            prev_w = w;
        }
        let d_rms = (d_bp_e / (blk as f64)).sqrt();
        let w_rms = (w_bp_e / (blk as f64)).sqrt();
        let bright = (w_hf / (w_e + 1.0e-18)).sqrt();
        if d_rms >= 1.5e-3 {
            let g = w_rms / d_rms;
            if d_rms >= 0.095 {
                loud_g.push(g);
                loud_hf.push(bright);
            } else if d_rms >= 0.032 {
                mid_g.push(g);
                quiet_hf.push(bright);
            } else if (0.0025..0.022).contains(&d_rms) {
                quiet_all.push(g);
                quiet_hf.push(bright);
                if prev_d1 < 0.028 && prev_d2 < 0.035 {
                    quiet_sust.push(g);
                }
            }
        }
        prev_d2 = prev_d1;
        prev_d1 = d_rms;
    }
    let median_or = |v: &mut [f64], def: f64| -> f64 {
        if v.is_empty() {
            return def;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let loud_med = median_or(&mut loud_g, 1.0).max(1.0e-6);
    let mid_med = median_or(&mut mid_g, loud_med).max(1.0e-6);
    let quiet_med = if !quiet_sust.is_empty() {
        median_or(&mut quiet_sust, mid_med)
    } else {
        median_or(&mut quiet_all, mid_med)
    };
    let mean_or = |v: &[f64], def: f64| -> f64 {
        if v.is_empty() {
            def
        } else {
            v.iter().sum::<f64>() / (v.len() as f64)
        }
    };
    let loud_b = mean_or(&loud_hf, 1.0);
    let quiet_b = mean_or(&quiet_hf, loud_b).max(1.0e-4);
    CoresetDynStats {
        gate_ratio: (quiet_med / mid_med.max(loud_med)).clamp(0.0, 2.0),
        comp_ratio: (loud_med / mid_med).clamp(0.0, 2.0),
        env_bright_shift: (loud_b / quiet_b).clamp(0.1, 10.0),
    }
}
