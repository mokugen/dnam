#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Two-Tier `Schoukens–Pintelon` 4-Regime Excitation Synthesis (`Regime 1..4`):
//! 1. Odd-Random Multisine (`N_per = 16,384`, `CF <= 1.8`, `Law 0.C` mains exclusion)
//! 2. Dynamic Staircase (`-42/-24/-12/-1.5 dBFS`, `Law 0.D` cap) + `>= 250 ms` Silent Decay Gaps
//! 3. Gapless Monophonic Chirp (`Law 0.E`) + Chromatic/Tritone IMD Sweep
//! 4. Topology-Branched Contrast across all `6` `REQ-7.4` modes (`1->1`, `1->2`, `2->2`, `2->1`, `4CM`, `WDW`)
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md` (`ref[train.capture.excitation.codesigned]`).

use std::collections::BTreeSet;
use std::f64::consts::{PI, TAU};

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::U64;

use crate::CANONICAL_TRAINER_SAMPLE_RATE_HZ;
use crate::stimulus::fft::{
    ADC_PEAK_GOVERNOR_FS, Complex64, fft_radix2_in_place, irfft_from_sparse_tones, is_mains_hum_bin,
};
use crate::stimulus::laws::TopologyProbeMode;

/// Canonical Period length ($N_{\text{per}} = 16,384$, $\Delta f = 2.9296875\text{ Hz}$ at `48 kHz`) for Regime 1.
pub const SCHOUKENS_R1_PERIOD_SAMPLES: usize = 16_384;

/// Number of `Regime 1` periods synthesized in Tier-1 transient `SPCM` (`P = 4 >= 3`: Period 0 settled/dropped, Periods `1..=3` analyzed).
pub const SCHOUKENS_R1_SPCM_PERIODS: usize = 4;

/// Maximum permitted linear crest factor ($\text{peak} / \text{RMS} \le 1.8$) on `Regime 1` multisine.
pub const SCHOUKENS_R1_MAX_CREST_FACTOR_LIN: f64 = 1.8;

/// Predictive peak ceiling (`-1.50 dBFS`, `ADC_PEAK_GOVERNOR_FS`) across all four regimes (`Law 0.D`).
pub const SCHOUKENS_PEAK_CEILING_FS: f64 = ADC_PEAK_GOVERNOR_FS;

const SCHOUKENS_R1_25S_LEN: usize = 2 * SCHOUKENS_R1_PERIOD_SAMPLES;
const SCHOUKENS_R2_25S_LEN: usize = 40_800;
const SCHOUKENS_R3_25S_LEN: usize = 26_400;
const SCHOUKENS_R4_25S_LEN: usize = 20_032;

const SCHOUKENS_R1_20S_LEN: usize = SCHOUKENS_R1_PERIOD_SAMPLES;
const SCHOUKENS_R2_20S_LEN: usize = 36_000;
const SCHOUKENS_R3_20S_LEN: usize = 24_000;
const SCHOUKENS_R4_20S_LEN: usize = 19_616;

pub(crate) const SCHOUKENS_SPCM_R1_LEN: usize =
    SCHOUKENS_R1_SPCM_PERIODS * SCHOUKENS_R1_PERIOD_SAMPLES;
pub(crate) const SCHOUKENS_SPCM_R2_BLOCK_LEN: usize = 40_960;
pub(crate) const SCHOUKENS_SPCM_R2_BURST_LEN: usize = 26_560;
pub(crate) const SCHOUKENS_SPCM_R2_GAP_LEN: usize =
    SCHOUKENS_SPCM_R2_BLOCK_LEN - SCHOUKENS_SPCM_R2_BURST_LEN;
pub(crate) const SCHOUKENS_SPCM_R2_LEN: usize = 4 * SCHOUKENS_SPCM_R2_BLOCK_LEN;
pub(crate) const SCHOUKENS_SPCM_R3_LEN: usize = 114_688;
pub(crate) const SCHOUKENS_SPCM_R4_LEN: usize = 87_936;

pub(crate) const SCHOUKENS_R3_25S_MONO_LEN: usize = (SCHOUKENS_R3_25S_LEN * 55) / 100;
pub(crate) const SCHOUKENS_R3_20S_MONO_LEN: usize = (SCHOUKENS_R3_20S_LEN * 55) / 100;
pub(crate) const SCHOUKENS_R3_25S_IMD_LEN: usize = SCHOUKENS_R3_25S_LEN - SCHOUKENS_R3_25S_MONO_LEN;
pub(crate) const SCHOUKENS_R3_20S_IMD_LEN: usize = SCHOUKENS_R3_20S_LEN - SCHOUKENS_R3_20S_MONO_LEN;

/// Canonical condensed `2.5 s` (`120,000`-frame) 4-regime sample offsets `[0, r2, r3, r4, total]`.
pub const SCHOUKENS_CORESET_25S_OFFSETS: [u32; 5] = [
    0,
    SCHOUKENS_R1_25S_LEN as u32,
    (SCHOUKENS_R1_25S_LEN + SCHOUKENS_R2_25S_LEN) as u32,
    (SCHOUKENS_R1_25S_LEN + SCHOUKENS_R2_25S_LEN + SCHOUKENS_R3_25S_LEN) as u32,
    (SCHOUKENS_R1_25S_LEN + SCHOUKENS_R2_25S_LEN + SCHOUKENS_R3_25S_LEN + SCHOUKENS_R4_25S_LEN)
        as u32,
];

/// Canonical condensed `2.0 s` (`96,000`-frame) 4-regime sample offsets `[0, r2, r3, r4, total]`.
pub const SCHOUKENS_CORESET_20S_OFFSETS: [u32; 5] = [
    0,
    SCHOUKENS_R1_20S_LEN as u32,
    (SCHOUKENS_R1_20S_LEN + SCHOUKENS_R2_20S_LEN) as u32,
    (SCHOUKENS_R1_20S_LEN + SCHOUKENS_R2_20S_LEN + SCHOUKENS_R3_20S_LEN) as u32,
    (SCHOUKENS_R1_20S_LEN + SCHOUKENS_R2_20S_LEN + SCHOUKENS_R3_20S_LEN + SCHOUKENS_R4_20S_LEN)
        as u32,
];

/// Canonical `9.0 s` (`432,000`-frame) Tier-1 `SPCM` capture length (`384,000..=576,000`).
pub const SCHOUKENS_SPCM_CANONICAL_FRAMES: usize =
    SCHOUKENS_SPCM_R1_LEN + SCHOUKENS_SPCM_R2_LEN + SCHOUKENS_SPCM_R3_LEN + SCHOUKENS_SPCM_R4_LEN;

/// Canonical `9.0 s` (`432,000`-frame) Tier-1 `SPCM` 4-regime sample offsets `[0, r2, r3, r4, total]`.
pub const SCHOUKENS_SPCM_REGIME_OFFSETS: [u32; 5] = [
    0,
    SCHOUKENS_SPCM_R1_LEN as u32,
    (SCHOUKENS_SPCM_R1_LEN + SCHOUKENS_SPCM_R2_LEN) as u32,
    (SCHOUKENS_SPCM_R1_LEN + SCHOUKENS_SPCM_R2_LEN + SCHOUKENS_SPCM_R3_LEN) as u32,
    SCHOUKENS_SPCM_CANONICAL_FRAMES as u32,
];

const _: () = assert!(SCHOUKENS_R1_PERIOD_SAMPLES.is_power_of_two());
const _: () = assert!(SCHOUKENS_R1_SPCM_PERIODS >= 3);
const _: () = assert!(
    SCHOUKENS_CORESET_25S_OFFSETS[4] == 120_000
        && SCHOUKENS_CORESET_25S_OFFSETS[4].is_multiple_of(16)
        && SCHOUKENS_CORESET_20S_OFFSETS[4] == 96_000
        && SCHOUKENS_CORESET_20S_OFFSETS[4].is_multiple_of(16)
        && SCHOUKENS_SPCM_CANONICAL_FRAMES == 432_000
        && SCHOUKENS_R3_25S_MONO_LEN == 14_520
        && SCHOUKENS_R3_20S_MONO_LEN == 13_200
        && SCHOUKENS_R3_25S_IMD_LEN == 11_880
        && SCHOUKENS_R3_20S_IMD_LEN == 10_800
        && SCHOUKENS_R3_25S_MONO_LEN + SCHOUKENS_R3_20S_MONO_LEN
            < (SCHOUKENS_SPCM_R3_LEN * 55) / 100
        && SCHOUKENS_R3_25S_IMD_LEN + SCHOUKENS_R3_20S_IMD_LEN
            < SCHOUKENS_SPCM_R3_LEN - (SCHOUKENS_SPCM_R3_LEN * 55) / 100
);

/// Deterministic `Regime 1` odd-random multisine bin partition at $N_{\text{per}} = 16,384$.
#[derive(Debug, Clone, PartialEq)]
pub struct OddRandomMultisinePlan {
    /// Period length in samples at `48 kHz` (`16,384`).
    pub period_samples: usize,
    /// Excited odd frequency bins $k \in 2\mathbb{Z} + 1$ with pink spectral amplitudes `(bin, amp)`.
    pub excited_odd_bins: Vec<(usize, f64)>,
    /// Unexcited omitted random-grid odd detection bins (`Law 0.C` / squeal clean) tracking odd nonlinearities.
    pub omitted_odd_bins: Vec<usize>,
    /// Unexcited even harmonic detection bins (`Law 0.C` / squeal clean) tracking even nonlinearities.
    pub unexcited_even_bins: Vec<usize>,
    /// Mains-hum and squeal-masked excluded bins in `2..=nyq_bin` (`Law 0.C`).
    pub masked_bins: BTreeSet<usize>,
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn hann_fade(i: usize, total: usize, ramp: usize) -> f64 {
    if total == 0 || ramp == 0 {
        return 1.0;
    }
    let r = ramp.min(total / 2).max(1);
    if i < r {
        0.5 * (1.0 - (PI * (i as f64) / (r as f64)).cos())
    } else if i + r >= total {
        let rem = total - 1 - i;
        0.5 * (1.0 - (PI * (rem as f64) / (r as f64)).cos())
    } else {
        1.0
    }
}

/// Builds a deterministic odd-random multisine bin partition over `20 Hz..20 kHz` at `N_per = 16,384`,
/// strictly excluding `Law 0.C` mains hum harmonics (`50/60 Hz ± 13.5 Hz`) and squeal masks.
#[must_use]
pub fn build_odd_random_multisine_plan(
    seed_256: &[u8; 256],
    squeal_mask_bins: &BTreeSet<usize>,
) -> OddRandomMultisinePlan {
    let n_per = SCHOUKENS_R1_PERIOD_SAMPLES;
    let bin_min = 7_usize; // 20.51 Hz
    let bin_max = 6_825_usize; // 19,995.1 Hz

    let mut rng = U64::ref_from_bytes(&seed_256[0..8]).map_or(0x5C40_0001_u64, |v| v.get());
    let mut masked_bins = BTreeSet::new();
    let mut candidate_odds = Vec::with_capacity(3_400);
    let mut unexcited_even_bins = Vec::with_capacity(3_400);

    for k in bin_min..=bin_max {
        if is_mains_hum_bin(k, CANONICAL_TRAINER_SAMPLE_RATE_HZ, n_per)
            || squeal_mask_bins.contains(&k)
        {
            masked_bins.insert(k);
            continue;
        }
        if k % 2 == 1 {
            candidate_odds.push(k);
        } else {
            unexcited_even_bins.push(k);
        }
    }

    let mut excited_raw = Vec::with_capacity((candidate_odds.len() * 3) / 4 + 8);
    let mut omitted_odd_bins = Vec::with_capacity(candidate_odds.len() / 4 + 4);
    for group in candidate_odds.chunks(4) {
        // Keep sub-50 Hz odd tones (< bin 17) populated so narrow LF 1/3-oct bands never empty.
        let can_omit = group.len() == 4 && group[0] >= 17;
        if can_omit {
            let omit_idx = (splitmix64(&mut rng) % 4) as usize;
            for (pos, &bin) in group.iter().enumerate() {
                if pos == omit_idx {
                    omitted_odd_bins.push(bin);
                } else {
                    excited_raw.push(bin);
                }
            }
        } else {
            excited_raw.extend_from_slice(group);
        }
    }

    // Pink-tilted excitation amplitudes normalizing total power for crisp wideband BLA SNR.
    let mut excited_odd_bins = Vec::with_capacity(excited_raw.len());
    let mut sq_sum = 0.0_f64;
    let sr_f64 = f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ);
    for &bin in &excited_raw {
        let f_hz = (bin as f64) * sr_f64 / (n_per as f64);
        let a = (1.0 + (f_hz / 2_400.0).powi(2)).powf(-0.25);
        sq_sum += a * a;
        excited_odd_bins.push((bin, a));
    }
    let scale = 0.5 / sq_sum.max(1.0e-18).sqrt();
    for (_, amp) in &mut excited_odd_bins {
        *amp *= scale;
    }

    OddRandomMultisinePlan {
        period_samples: n_per,
        excited_odd_bins,
        omitted_odd_bins,
        unexcited_even_bins,
        masked_bins,
    }
}

/// Synthesizes one `16,384`-sample period of [`OddRandomMultisinePlan`] using Van der Ouderaa /
/// Guillaume iterative clipping (`CF <= 1.8`, peak `<= ADC_PEAK_GOVERNOR_FS`) and zero out-of-set leakage,
/// returning `(period_samples, achieved_crest_factor_linear)`.
#[must_use]
pub fn synthesize_odd_random_multisine_period(
    plan: &OddRandomMultisinePlan,
    seed_256: &[u8; 256],
    channel_salt: u64,
    peak_ceiling_fs: f64,
) -> (Vec<f32>, f64) {
    let n = plan.period_samples;
    if plan.excited_odd_bins.is_empty() || !n.is_power_of_two() || n < 64 {
        return (vec![0.0_f32; n], 1.0);
    }
    let ceiling = if peak_ceiling_fs.is_finite() {
        peak_ceiling_fs.clamp(0.01, ADC_PEAK_GOVERNOR_FS)
    } else {
        ADC_PEAK_GOVERNOR_FS
    };

    let mut rng = U64::ref_from_bytes(&seed_256[8..16]).map_or(0xA11B_C22D_u64, |v| v.get())
        ^ channel_salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);

    let m = plan.excited_odd_bins.len();
    let total_p: f64 = plan
        .excited_odd_bins
        .iter()
        .map(|&(_, a)| a * a)
        .sum::<f64>()
        .max(1.0e-30);
    let mut phases = Vec::with_capacity(m);
    let mut cum_p = 0.0_f64;
    for (idx, &(_, a)) in plan.excited_odd_bins.iter().enumerate() {
        let pk = (a * a) / total_p;
        let schroeder = -TAU * ((idx as f64) * cum_p + 0.5 * (idx as f64) * pk);
        let jitter = (((splitmix64(&mut rng) >> 40) as f64) / ((1u64 << 24) as f64) - 0.5) * 0.35;
        cum_p += pk;
        phases.push((schroeder + jitter).rem_euclid(TAU) - PI);
    }

    let half = n / 2;
    let spec_scale = 0.5 * (n as f64);
    let mut cbuf = vec![Complex64::default(); n];
    let mut best_phases = phases.clone();
    let mut best_cf = f64::INFINITY;

    let synth_cf = |phs: &[f64], buf: &mut [Complex64]| -> (f64, f64) {
        buf.fill(Complex64::default());
        for (&(bin, amp), &ph) in plan.excited_odd_bins.iter().zip(phs) {
            if bin > 0 && bin < half && amp.is_finite() && amp > 0.0 {
                let re = spec_scale * amp * ph.cos();
                let im = spec_scale * amp * ph.sin();
                buf[bin] = Complex64::new(re, im);
                buf[n - bin] = Complex64::new(re, -im);
            }
        }
        fft_radix2_in_place(buf, true);
        let mut peak = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for c in buf.iter() {
            peak = peak.max(c.re.abs());
            sum_sq += c.re * c.re;
        }
        let rms = (sum_sq / (n as f64)).sqrt().max(1.0e-18);
        (peak / rms, rms)
    };

    for iter in 0..40 {
        let (cf, rms) = synth_cf(&phases, &mut cbuf);
        if cf < best_cf {
            best_cf = cf;
            best_phases.clone_from(&phases);
        }
        if cf <= 1.66 {
            break;
        }
        let clip_lim = rms * (1.34 + 0.004 * (iter as f64));
        for c in &mut cbuf {
            c.re = c.re.clamp(-clip_lim, clip_lim);
            c.im = 0.0;
        }
        fft_radix2_in_place(&mut cbuf, false);
        for (idx, &(bin, _)) in plan.excited_odd_bins.iter().enumerate() {
            if bin > 0 && bin < half {
                phases[idx] = cbuf[bin].arg();
            }
        }
    }

    let tones: Vec<(usize, f64, f64)> = plan
        .excited_odd_bins
        .iter()
        .zip(&best_phases)
        .map(|(&(b, a), &ph)| (b, a, ph))
        .collect();
    let raw = irfft_from_sparse_tones(n, &tones);
    let mut peak = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &x in &raw {
        peak = peak.max(x.abs());
        sum_sq += x * x;
    }
    let raw_rms = (sum_sq / (n as f64)).sqrt().max(1.0e-18);
    let achieved_cf = peak / raw_rms;
    let target_peak = (0.72 * ceiling).min(ceiling);
    let gain = if peak > 1.0e-18 {
        target_peak / peak
    } else {
        0.0
    };
    let out = raw.into_iter().map(|x| (x * gain) as f32).collect();
    (out, achieved_cf)
}

const SCHOUKENS_R2_FINAL_GAP_SAMPLES: usize = CANONICAL_TRAINER_SAMPLE_RATE_HZ as usize / 4;

/// Returns canonical `(active_burst_samples, silent_gap_samples)` block specs for condensed `Regime 2`.
#[must_use]
pub(crate) fn regime2_coreset_block_specs(r2_len: usize) -> [(usize, usize); 4] {
    if r2_len == SCHOUKENS_R2_25S_LEN {
        return [
            (3_600, 2_400),
            (3_600, 2_400),
            (4_800, 6_000),
            (6_000, SCHOUKENS_R2_FINAL_GAP_SAMPLES),
        ];
    }
    if r2_len == SCHOUKENS_R2_20S_LEN {
        return [
            (3_000, 2_000),
            (3_000, 2_000),
            (4_000, 5_000),
            (5_000, SCHOUKENS_R2_FINAL_GAP_SAMPLES),
        ];
    }
    let last_gap = SCHOUKENS_R2_FINAL_GAP_SAMPLES.min(r2_len / 3);
    let prefix = r2_len.saturating_sub(last_gap);
    let act4 = (prefix / 5)
        .clamp(16, SCHOUKENS_SPCM_R2_BURST_LEN)
        .min(prefix);
    let pre3 = prefix.saturating_sub(act4);
    let g0 = (pre3 / 10).min(SCHOUKENS_SPCM_R2_GAP_LEN);
    let g1 = (pre3 / 10).min(SCHOUKENS_SPCM_R2_GAP_LEN);
    let g2 = ((pre3 * 22) / 100).min(SCHOUKENS_SPCM_R2_GAP_LEN);
    let act_sum = pre3.saturating_sub(g0 + g1 + g2);
    let a0 = ((act_sum * 3) / 10).min(SCHOUKENS_SPCM_R2_BURST_LEN);
    let a1 = ((act_sum * 3) / 10).min(SCHOUKENS_SPCM_R2_BURST_LEN);
    let a2 = act_sum
        .saturating_sub(a0 + a1)
        .min(SCHOUKENS_SPCM_R2_BURST_LEN);
    [(a0, g0), (a1, g1), (a2, g2), (act4, last_gap)]
}

/// Returns the `(gap_start_local, gap_end_local)` silent decay intervals inside `Regime 2` of length `len`.
#[must_use]
pub fn regime2_gap_ranges(len: usize) -> Vec<(usize, usize)> {
    if len == SCHOUKENS_SPCM_R2_LEN {
        return (0..4)
            .map(|b| {
                let base = b * SCHOUKENS_SPCM_R2_BLOCK_LEN;
                (
                    base + SCHOUKENS_SPCM_R2_BURST_LEN,
                    base + SCHOUKENS_SPCM_R2_BLOCK_LEN,
                )
            })
            .collect();
    }
    if len < 1_024 {
        let last_gap = SCHOUKENS_R2_FINAL_GAP_SAMPLES.min(len / 3);
        return vec![(len.saturating_sub(last_gap), len)];
    }
    let specs = regime2_coreset_block_specs(len);
    let mut out = Vec::with_capacity(4);
    let mut cursor = 0_usize;
    for (act, gap) in specs {
        let g_start = cursor + act;
        let g_end = (g_start + gap).min(len);
        out.push((g_start.min(len), g_end));
        cursor = g_end;
    }
    out
}

/// Synthesizes `Regime 2`: `-42/-24/-12/-1.5 dBFS` broadband staircase bursts interleaved with silent decay gaps.
#[must_use]
pub fn synthesize_regime2_staircase_and_gaps(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let sr = f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ);
    let gaps = regime2_gap_ranges(len);
    let levels_db = [-42.0_f64, -24.0, -12.0, -1.5];
    let mut out = vec![0.0_f32; len];
    let mut active_cursor = 0_usize;
    let harmonics = [
        (82.406_89_f64, 0.42),
        (164.813_78, 0.24),
        (246.941_65, 0.20),
        (440.0, 0.16),
        (987.77, 0.12),
        (2_217.46, 0.10),
        (4_698.64, 0.08),
        (9_397.27, 0.06),
    ];

    for (idx, &(g_start, g_end)) in gaps.iter().enumerate() {
        let start = active_cursor.min(len);
        let end = g_start.min(len);
        if end > start {
            let dur = end - start;
            let level_db = levels_db[idx.min(levels_db.len() - 1)];
            let target_peak = (10.0_f64.powf(level_db / 20.0)).min(ADC_PEAK_GOVERNOR_FS);
            let mut raw = Vec::with_capacity(dur);
            let mut max_raw = 1.0e-12_f64;
            for k in 0..dur {
                let t = (k as f64) / sr;
                let mut s = 0.0_f64;
                for (h_i, &(freq, w)) in harmonics.iter().enumerate() {
                    let ph = (h_i as f64) * 0.61;
                    s += w * (TAU * freq * t + ph).sin();
                }
                let val = hann_fade(k, dur, 48) * s;
                max_raw = max_raw.max(val.abs());
                raw.push(val);
            }
            let scale = target_peak / max_raw;
            for (k, val) in raw.into_iter().enumerate() {
                out[start + k] = (val * scale) as f32;
            }
        }
        active_cursor = g_end;
    }
    out
}

fn synth_mono_log_chirp(dst: &mut [f32], sr: f64) {
    let n = dst.len();
    if n < 2 {
        return;
    }
    let f0 = 55.0_f64;
    let f1 = 16_000.0_f64;
    let ln_r = (f1 / f0).ln();
    let dur = (n as f64) / sr;
    let mono_amp = 0.65 * ADC_PEAK_GOVERNOR_FS;
    for (i, slot) in dst.iter_mut().enumerate() {
        let t = (i as f64) / sr;
        let phase = TAU * f0 * dur * ((t / dur * ln_r).exp() - 1.0) / ln_r;
        *slot = (mono_amp * hann_fade(i, n, 64) * phase.sin()) as f32;
    }
}

fn synth_imd_tritone_sweep(dst: &mut [f32], sr: f64) {
    let n = dst.len();
    if n < 2 {
        return;
    }
    let f_lo = 65.406_39_f64;
    let f_hi = 4_000.0_f64;
    let ln_imd = (f_hi / f_lo).ln();
    let dur = (n as f64) / sr;
    let ratio_tritone = 2.0_f64.powf(6.0 / 12.0);
    let ratio_min2 = 2.0_f64.powf(1.0 / 12.0);
    let imd_amp = 0.85 * ADC_PEAK_GOVERNOR_FS;
    let mut raw = Vec::with_capacity(n);
    let mut peak = 1.0e-12_f64;
    for i in 0..n {
        let t = (i as f64) / sr;
        let ph = TAU * f_lo * dur * ((t / dur * ln_imd).exp() - 1.0) / ln_imd;
        let s = 0.36 * ph.sin()
            + 0.34 * (ph * ratio_tritone + 0.7).sin()
            + 0.30 * (ph * ratio_min2 + 1.4).sin();
        let v = hann_fade(i, n, 64) * s;
        peak = peak.max(v.abs());
        raw.push(v);
    }
    let sc = imd_amp / peak;
    for (i, val) in raw.into_iter().enumerate() {
        dst[i] = (val * sc) as f32;
    }
}

/// Synthesizes `Regime 3`: gapless monophonic log chirp (`55 Hz -> 16 kHz`, `Law 0.E`) + chromatic/tritone IMD sweep.
#[must_use]
pub fn synthesize_regime3_chirp_and_imd(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    let sr = f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ);
    let split = ((len * 55) / 100).max(16).min(len);
    let mut out = vec![0.0_f32; len];

    // Part A [0..split): pure monophonic log chirp (`55 Hz -> 16,000 Hz`, zero polyphonic beating).
    // For Tier-1 SPCM, pack contiguous 2.5s / 2.0s / extended full-band sub-sweeps so condensation
    // preserves the entire 55 Hz -> 16 kHz decade span.
    if len == SCHOUKENS_SPCM_R3_LEN {
        let a0 = SCHOUKENS_R3_25S_MONO_LEN;
        let a1 = a0 + SCHOUKENS_R3_20S_MONO_LEN;
        synth_mono_log_chirp(&mut out[..a0], sr);
        synth_mono_log_chirp(&mut out[a0..a1], sr);
        synth_mono_log_chirp(&mut out[a1..split], sr);
    } else {
        synth_mono_log_chirp(&mut out[..split], sr);
    }

    // Part B [split..len): chromatic/tritone + minor-second dissonant IMD sweep (`65 Hz -> 4,000 Hz`).
    let imd_len = len.saturating_sub(split);
    if imd_len > 0 {
        if len == SCHOUKENS_SPCM_R3_LEN {
            let b0 = split + SCHOUKENS_R3_25S_IMD_LEN;
            let b1 = b0 + SCHOUKENS_R3_20S_IMD_LEN;
            synth_imd_tritone_sweep(&mut out[split..b0], sr);
            synth_imd_tritone_sweep(&mut out[b0..b1], sr);
            synth_imd_tritone_sweep(&mut out[b1..len], sr);
        } else {
            synth_imd_tritone_sweep(&mut out[split..len], sr);
        }
    }
    out
}

fn synth_contrast_burst(start: usize, count: usize, amp: f64, freq_hz: f64) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    let sr = f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ);
    let scale = amp.clamp(0.0, 0.90 * ADC_PEAK_GOVERNOR_FS);
    (0..count)
        .map(|i| {
            let t = ((start + i) as f64) / sr;
            let local_t = ((i % 2_000) as f64) / sr;
            let env = (1.0 - (-local_t / 0.0008).exp()) * (-local_t / 0.024).exp();
            let car =
                0.65 * (TAU * freq_hz * t).sin() + 0.35 * (TAU * (freq_hz * 1.498) * t + 0.3).sin();
            (scale * env * hann_fade(i, count, 32) * car) as f32
        })
        .collect()
}

/// Synthesizes `Regime 4`: explicit topology-contrast excitation across all `6` `REQ-7.4` modes,
/// returning `(in_1, Option<in_2>, Option<aux_in_3>)` capped strictly `<= ADC_PEAK_GOVERNOR_FS`.
#[must_use]
pub fn synthesize_regime4_topology_contrast(
    len: usize,
    topology_mode: TopologyProbeMode,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    if len == 0 {
        return (Vec::new(), None, None);
    }
    let sr = f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ);
    match topology_mode {
        TopologyProbeMode::Mono1To1
        | TopologyProbeMode::MonoToStereo1To2
        | TopologyProbeMode::WetDryWet1To3 => {
            let burst = synth_contrast_burst(0, len, 0.78 * ADC_PEAK_GOVERNOR_FS, 196.0);
            (burst, None, None)
        }
        TopologyProbeMode::TrueStereo2To2 => {
            let mut l = vec![0.0_f32; len];
            let mut r = vec![0.0_f32; len];
            let s = len / 5;
            let b_l = synth_contrast_burst(0, len, 0.72 * ADC_PEAK_GOVERNOR_FS, 185.0);
            let b_w = synth_contrast_burst(11, len, 0.125 * ADC_PEAK_GOVERNOR_FS, 740.0);
            // 0..s: L-only; s..2s: R-only; 2s..3s: Mid (L=R); 3s..4s: Side (L=-R); 4s..len: Hard-Pan L + -18 dB Witness R
            let s1 = s.min(len);
            let s2 = (2 * s).min(len);
            let s3 = (3 * s).min(len);
            let s4 = (4 * s).min(len);
            l[..s1].copy_from_slice(&b_l[..s1]);
            r[s..s2].copy_from_slice(&b_l[s..s2]);
            l[(2 * s)..s3].copy_from_slice(&b_l[(2 * s)..s3]);
            r[(2 * s)..s3].copy_from_slice(&b_l[(2 * s)..s3]);
            for i in (3 * s)..s4 {
                l[i] = b_l[i];
                r[i] = -b_l[i];
            }
            l[(4 * s)..len].copy_from_slice(&b_l[(4 * s)..len]);
            r[(4 * s)..len].copy_from_slice(&b_w[(4 * s)..len]);
            (l, Some(r), None)
        }
        TopologyProbeMode::DualInToMono2To1 => {
            let mut audio = vec![0.0_f32; len];
            let mut key = vec![0.0_f32; len];
            let q = len / 4;
            let a_steady = synth_contrast_burst(0, len, 0.45 * ADC_PEAK_GOVERNOR_FS, 220.0);
            let k_burst = synth_contrast_burst(0, len, 0.80 * ADC_PEAK_GOVERNOR_FS, 110.0);
            // 0..q: Key-Silent / Audio-Steady; q..2q: Key-Loud / Audio-Steady
            // 2q..3q: Key-Burst / Audio-Mid; 3q..len: Key-Steady / Audio-Silent
            let q1 = q.min(len);
            let q2 = (2 * q).min(len);
            let q3 = (3 * q).min(len);
            audio[..q1].copy_from_slice(&a_steady[..q1]);
            for i in q..q2 {
                audio[i] = a_steady[i];
                key[i] =
                    (0.75 * ADC_PEAK_GOVERNOR_FS * (TAU * 130.8 * (i as f64) / sr).sin()) as f32;
            }
            for i in (2 * q)..q3 {
                audio[i] = 0.55 * a_steady[i];
                key[i] = k_burst[i];
            }
            key[(3 * q)..len].copy_from_slice(&k_burst[(3 * q)..len]);
            (audio, Some(key.clone()), Some(key))
        }
        TopologyProbeMode::FourCableMethod4Cm => {
            let mut in1 = vec![0.0_f32; len];
            let mut in3 = vec![0.0_f32; len];
            let t = len / 3;
            let sig1 = synth_contrast_burst(0, len, 0.74 * ADC_PEAK_GOVERNOR_FS, 146.83);
            let sig3 = synth_contrast_burst(0, len, 0.74 * ADC_PEAK_GOVERNOR_FS, 587.33);
            // 0..t: in_1 only; t..2t: in_3 only; 2t..len: simultaneous orthogonal in_1 + in_3
            let t1 = t.min(len);
            let t2 = (2 * t).min(len);
            in1[..t1].copy_from_slice(&sig1[..t1]);
            in3[t..t2].copy_from_slice(&sig3[t..t2]);
            in1[(2 * t)..len].copy_from_slice(&sig1[(2 * t)..len]);
            in3[(2 * t)..len].copy_from_slice(&sig3[(2 * t)..len]);
            (in1, None, Some(in3))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stimulus::fft::rfft_bins;

    #[test]
    fn test_regime1_odd_random_multisine_purity_cf_and_mains_exclusion() {
        let seed = [0x5A_u8; 256];
        let mut mask = BTreeSet::new();
        mask.insert(1_235);
        let plan = build_odd_random_multisine_plan(&seed, &mask);
        assert!(plan.excited_odd_bins.len() > 2_000);
        assert!(!plan.omitted_odd_bins.is_empty());
        assert!(!plan.unexcited_even_bins.is_empty());
        assert!(plan.masked_bins.contains(&1_235));

        let (period, cf) =
            synthesize_odd_random_multisine_period(&plan, &seed, 0, ADC_PEAK_GOVERNOR_FS);
        assert_eq!(period.len(), SCHOUKENS_R1_PERIOD_SAMPLES);
        assert!(cf <= SCHOUKENS_R1_MAX_CREST_FACTOR_LIN, "cf={cf}");
        let peak = period
            .iter()
            .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()));
        assert!(peak <= ADC_PEAK_GOVERNOR_FS + 1.0e-6, "peak={peak}");

        let spec = rfft_bins(&period);
        let max_exc_p = plan
            .excited_odd_bins
            .iter()
            .map(|&(b, _)| spec[b].norm_sqr())
            .fold(0.0_f64, f64::max);
        let max_even_p = plan
            .unexcited_even_bins
            .iter()
            .map(|&b| spec[b].norm_sqr())
            .fold(0.0_f64, f64::max);
        let max_omit_p = plan
            .omitted_odd_bins
            .iter()
            .map(|&b| spec[b].norm_sqr())
            .fold(0.0_f64, f64::max);
        let max_mask_p = plan
            .masked_bins
            .iter()
            .map(|&b| spec[b].norm_sqr())
            .fold(0.0_f64, f64::max);
        let leak_db = 10.0
            * ((max_even_p.max(max_omit_p).max(max_mask_p)) / max_exc_p)
                .max(1e-20)
                .log10();
        assert!(leak_db < -120.0, "out-of-set leakage {leak_db:.1} dB");
    }

    #[test]
    fn test_regimes_2_3_4_and_all_6_topologies() {
        let r2 = synthesize_regime2_staircase_and_gaps(40_800);
        assert_eq!(r2.len(), 40_800);
        assert!(
            r2.iter()
                .all(|v| v.abs() as f64 <= ADC_PEAK_GOVERNOR_FS + 1.0e-6)
        );
        assert!(r2[28_800..40_800].iter().all(|&v| v == 0.0));
        assert_eq!(regime2_gap_ranges(36_000).last(), Some(&(24_000, 36_000)));
        assert_eq!(regime2_gap_ranges(163_840).len(), 4);

        let r3 = synthesize_regime3_chirp_and_imd(26_400);
        assert_eq!(r3.len(), 26_400);
        assert!(
            r3.iter()
                .all(|v| v.abs() as f64 <= ADC_PEAK_GOVERNOR_FS + 1.0e-6)
        );

        for topo in [
            TopologyProbeMode::Mono1To1,
            TopologyProbeMode::MonoToStereo1To2,
            TopologyProbeMode::TrueStereo2To2,
            TopologyProbeMode::DualInToMono2To1,
            TopologyProbeMode::FourCableMethod4Cm,
            TopologyProbeMode::WetDryWet1To3,
        ] {
            let (l, r, aux) = synthesize_regime4_topology_contrast(20_032, topo);
            assert_eq!(l.len(), 20_032);
            match topo {
                TopologyProbeMode::TrueStereo2To2 => {
                    let rv = r.expect("2->2 R");
                    assert_eq!(rv.len(), 20_032);
                    assert!(aux.is_none());
                }
                TopologyProbeMode::DualInToMono2To1 => {
                    assert!(r.is_some());
                    assert!(aux.is_some());
                }
                TopologyProbeMode::FourCableMethod4Cm => {
                    assert!(r.is_none());
                    assert!(aux.is_some());
                }
                _ => {
                    assert!(r.is_none());
                    assert!(aux.is_none());
                }
            }
        }
    }

    #[test]
    fn test_regimes_edge_lengths_hann_fade_and_masks() {
        assert_eq!(hann_fade(0, 0, 0), 1.0);
        assert_eq!(hann_fade(50, 100, 10), 1.0);
        assert!(hann_fade(0, 100, 10) < 0.01 && hann_fade(99, 100, 10) < 0.01);
        assert_eq!(regime2_gap_ranges(0), vec![(0, 0)]);
        assert_eq!(regime2_gap_ranges(600).len(), 1);
        assert_eq!(regime2_gap_ranges(20_000).len(), 4);
        assert!(synthesize_regime3_chirp_and_imd(0).is_empty());
        assert!(
            synthesize_regime4_topology_contrast(0, TopologyProbeMode::Mono1To1)
                .0
                .is_empty()
        );
        let unmasked =
            build_odd_random_multisine_plan(&[0x42_u8; 256], &std::collections::BTreeSet::new());
        assert!(!unmasked.masked_bins.contains(&301));
        assert!(unmasked.masked_bins.contains(&105));
        let masked = build_odd_random_multisine_plan(
            &[0x42_u8; 256],
            &std::collections::BTreeSet::from([301]),
        );
        assert!(masked.masked_bins.contains(&301));
        assert!(!masked.excited_odd_bins.iter().any(|&(b, _)| b == 301));
        assert!(!masked.omitted_odd_bins.contains(&301));
    }
}
