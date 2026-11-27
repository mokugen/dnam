#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! 4-Regime Sub-Sample / Clock-Skew Alignment (`CleanLinear`, `PolarityInverted`,
//! `DispersiveAllPass`, `NonlinearEnvelopeFallback`) and `64`-Tap Kaiser-Sinc Fractional
//! Realignment (`align::regimes`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.alignment.multi-regime]`, `ref[train.capture.invariants]`).

mod causal_check;
mod hilbert_env;
mod hybrid_parabolic;
mod ppm_drift;

use std::f64::consts::PI;

pub use super::calibration::{
    ACOUSTIC_NULL_HOMING_MAX_DB, CALB_STIMULUS_SAMPLES_48K, DecayTailDecision,
    DifferentialPadCalibration, LIVE_DECAY_TAIL_MARGIN_DB, LoopbackCalibrationReport,
    MIC_PAD_PROBE_SAMPLES_48K, NullHomingReport, THERMAL_RECOVERY_MAX_DB, ThermalRecoveryReport,
    calibrate_loopback_calb, evaluate_acoustic_null_homing, evaluate_live_decay_tail,
    evaluate_thermal_recovery_gate, generate_calb_loopback_stimulus, generate_mic_pad_probe_400ms,
    measure_differential_pad_ratio, rescale_padded_stem,
};
pub use hybrid_parabolic::{
    HYBRID_SUBSAMPLE_MAX_DELTA, PARABOLIC_PLATEAU_EPS, constrain_subsample_around_coarse,
    parabolic_peak_offset,
};
pub use ppm_drift::{
    PPM_DEADBAND_MIN, PPM_MAX_ABS, PPM_MIN_SAMPLES, PPM_QUARTER_MIN_COHERENCE, PpmDriftEstimate,
    estimate_ppm_clock_drift,
};

use super::{AlignError, estimate_subsample_lag_and_gain};
use causal_check::guard_lag_causal_onset;
use hilbert_env::compute_analytic_envelopes_stereo_pair;
use hybrid_parabolic::{normalized_corr_at_fractional_lag, refine_fractional_lag_hybrid};

/// Detected physical signal-path regime during sub-sample alignment
/// (`ref[train.capture.alignment.multi-regime]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentRegime {
    /// High linear coherence ($\gamma_{\text{lin}} \ge 0.35$), positive polarity, non-dispersive peak.
    CleanLinear,
    /// High linear coherence ($\gamma_{\text{lin}} \ge 0.35$) with inverted polarity (`signed_gain < 0.0`).
    PolarityInverted,
    /// Dispersive phase/all-pass response (crossover, wah, spring) locked to causal HF wavefront.
    DispersiveAllPass,
    /// Suppressed fundamental linear coherence ($\gamma_{\text{lin}} < 0.35$ from hard fuzz, rectifier,
    /// or octave divider) locked via stereo Jacobi elliptic `H1` Hilbert envelope cross-correlation.
    NonlinearEnvelopeFallback,
}

/// Multi-regime sub-sample latency, polarity, coherence, and clock-skew estimate
/// (`ref[train.capture.alignment.multi-regime]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiRegimeAlignment {
    /// Active physical alignment regime selected for this stem.
    pub regime: AlignmentRegime,
    /// Total sub-sample lag in device-rate samples (`wet` delayed relative to `dry`).
    pub total_lag_samples: f64,
    /// Signed linear gain (or signed RMS gain in `NonlinearEnvelopeFallback`).
    pub signed_gain: f64,
    /// True if wet return is polarity-inverted (`signed_gain < 0.0`).
    pub polarity_inverted: bool,
    /// Normalized broadband linear coherence $\gamma_{\text{lin}} \in [0.0, 1.0]$.
    pub linear_coherence: f64,
    /// Estimated sample-clock skew / drift in parts-per-million (`ppm`, zeroed when gated off).
    pub clock_drift_ppm: f64,
}

#[inline]
fn bessel_i0_f64(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_sq = 0.25 * x * x;
    for k in 1..=24 {
        let kf = f64::from(k);
        term *= half_sq / (kf * kf);
        sum += term;
        if term < sum * 1.0e-16 {
            break;
        }
    }
    sum
}

/// Realigns `wet` by advancing it by `lag_samples + (clock_drift_ppm * 1e-6) * n` using a `64`-tap
/// (`radius = 32`) partition-of-unity Kaiser-sinc interpolator ($\beta = 8.6$) and optional polarity flip.
#[must_use]
pub fn realign_fractional_kaiser64(
    wet: &[f32],
    lag_samples: f64,
    clock_drift_ppm: f64,
    invert_polarity: bool,
) -> Vec<f32> {
    if wet.is_empty() {
        return Vec::new();
    }
    let clean: Vec<f64> = wet
        .iter()
        .map(|&v| if v.is_finite() { f64::from(v) } else { 0.0 })
        .collect();
    let sign = if invert_polarity { -1.0_f64 } else { 1.0_f64 };
    let safe_lag = if lag_samples.is_finite() {
        lag_samples
    } else {
        0.0
    };
    let drift_rate = if clock_drift_ppm.is_finite() {
        (clock_drift_ppm * 1.0e-6).clamp(-0.05, 0.05)
    } else {
        0.0
    };

    if safe_lag.abs() < 1.0e-12 && drift_rate.abs() < 1.0e-15 {
        return clean.into_iter().map(|v| (sign * v) as f32).collect();
    }

    const RADIUS: isize = 32;
    const BETA: f64 = 8.6;
    let inv_i0 = 1.0 / bessel_i0_f64(BETA);
    let n_len = clean.len() as isize;
    let mut out = Vec::with_capacity(clean.len());

    let build_taps = |frac: f64| -> [f64; 64] {
        let mut taps = [0.0_f64; 64];
        let mut sum_w = 0.0_f64;
        for (t, slot) in taps.iter_mut().enumerate() {
            let u = frac - (((t as isize) - RADIUS + 1) as f64);
            let abs_u = u.abs();
            if abs_u >= (RADIUS as f64) {
                continue;
            }
            let sinc = if abs_u < 1.0e-12 {
                1.0
            } else {
                (PI * u).sin() / (PI * u)
            };
            let r = abs_u / (RADIUS as f64);
            let w = sinc * bessel_i0_f64(BETA * (1.0 - r * r).max(0.0).sqrt()) * inv_i0;
            *slot = w;
            sum_w += w;
        }
        let sc = if sum_w.abs() > 1.0e-15 {
            sign / sum_w
        } else {
            0.0
        };
        for slot in &mut taps {
            *slot *= sc;
        }
        taps
    };

    let dot_at = |base: isize, taps: &[f64; 64]| -> f32 {
        let mut acc = 0.0_f64;
        if base >= 0 && base + 64 <= n_len {
            for (&w, &x) in taps.iter().zip(&clean[base as usize..(base as usize + 64)]) {
                acc += w * x;
            }
        } else {
            for (t, &w) in taps.iter().enumerate() {
                let m = base + (t as isize);
                if m >= 0 && m < n_len {
                    acc += w * clean[m as usize];
                }
            }
        }
        if acc.is_finite() { acc as f32 } else { 0.0 }
    };

    if drift_rate.abs() < 1.0e-15 {
        let lag_floor = safe_lag.floor() as isize;
        let taps = build_taps(safe_lag - (lag_floor as f64));
        for n in 0..clean.len() {
            out.push(dot_at((n as isize) + lag_floor - RADIUS + 1, &taps));
        }
        return out;
    }

    let table: Vec<[f64; 64]> = (0..=512).map(|p| build_taps((p as f64) / 512.0)).collect();
    for n in 0..clean.len() {
        let center = (n as f64) + safe_lag + drift_rate * (n as f64);
        let k_base = center.floor() as isize;
        let phase_f = ((center - (k_base as f64)) * 512.0).clamp(0.0, 512.0);
        let p0 = phase_f.floor() as usize;
        let alpha = (phase_f - (p0 as f64)) as f32;
        let y0 = dot_at(k_base - RADIUS + 1, &table[p0]);
        let y1 = dot_at(k_base - RADIUS + 1, &table[(p0 + 1).min(512)]);
        out.push(y0 + alpha * (y1 - y0));
    }
    out
}

fn estimate_leading_edge_polarity(dry: &[f32], wet: &[f32], lag_int: isize) -> bool {
    let peak_dry = dry.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
    if peak_dry <= 1.0e-6 {
        return false;
    }
    let thresh = 0.25 * peak_dry;
    for (i, &d) in dry.iter().enumerate() {
        if d.abs() >= thresh {
            let j = i as isize + lag_int;
            if j >= 0 && (j as usize) < wet.len() {
                let w = wet[j as usize];
                if w.abs() > 1.0e-6 {
                    return (d * w) < 0.0;
                }
            }
        }
    }
    false
}

fn second_diff_hp_wavefront(sig: &[f32]) -> Vec<f32> {
    (0..sig.len())
        .map(|i| {
            let x0 = sig[i];
            let x1 = if i >= 1 { sig[i - 1] } else { 0.0 };
            let x2 = if i >= 2 { sig[i - 2] } else { 0.0 };
            x0 - 1.92 * x1 + 0.9216 * x2
        })
        .collect()
}

fn eighth_diff_wavefront(sig: &[f32]) -> Vec<f32> {
    const BINOM_8: [f32; 9] = [1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0];
    (0..sig.len())
        .map(|i| {
            let mut acc = 0.0_f32;
            for (k, &c) in BINOM_8.iter().enumerate() {
                if i >= k {
                    acc += c * sig[i - k];
                }
            }
            acc
        })
        .collect()
}

fn detect_dispersive_wavefront(
    dry: &[f32],
    wet: &[f32],
    lin_lag: f64,
    lin_coherence: f64,
    lin_gain: f64,
    max_lag_samples: usize,
    tau_coarse_opt: Option<f64>,
) -> Option<(f64, f64, bool)> {
    if dry.len() < 256 {
        return None;
    }
    let hf_dry_full = eighth_diff_wavefront(dry);
    let hf_wet_full = eighth_diff_wavefront(wet);
    let hf_dry = &hf_dry_full[64..hf_dry_full.len() - 64];
    let hf_wet = &hf_wet_full[64..hf_wet_full.len() - 64];
    let dry_e: f64 = dry[64..dry.len() - 64]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    let hf_dry_e: f64 = hf_dry.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    if hf_dry_e > 1.0e-10 * dry_e.max(1.0e-24) {
        let search_span = max_lag_samples.max((lin_lag.abs() as usize) + 16);
        if let Ok(hc) = estimate_subsample_lag_and_gain(hf_dry, hf_wet, search_span) {
            let (hf_lag, hf_coh, hf_gain) =
                refine_fractional_lag_hybrid(hf_dry, hf_wet, hc.total_lag_samples, None);
            let hf_abs = hf_coh.abs();
            let strong_disp = (lin_coherence >= 0.35 && hf_abs >= 0.55 && (lin_lag - hf_lag) > 3.5)
                || (lin_coherence < 0.35 && hf_abs >= 0.30 && hf_abs > lin_coherence + 0.12);
            if strong_disp {
                let inv = if lin_coherence >= 0.50 {
                    lin_gain < 0.0
                } else {
                    hf_coh < 0.0
                };
                let eff_gain = if lin_coherence >= 0.35 {
                    lin_gain
                } else {
                    hf_gain
                };
                let guarded_lag = tau_coarse_opt.map_or(hf_lag, |tc| {
                    let clag = constrain_subsample_around_coarse(hf_lag, 0.0, tc);
                    guard_lag_causal_onset(dry, wet, clag, hf_lag, Some(tc))
                });
                return Some((guarded_lag, eff_gain, inv));
            }
        }
    }
    if lin_coherence >= 0.35 && lin_lag > 4.0 {
        let hp_dry_full = second_diff_hp_wavefront(dry);
        let hp_wet_full = second_diff_hp_wavefront(wet);
        let hp_dry = &hp_dry_full[64..hp_dry_full.len() - 64];
        let hp_wet = &hp_wet_full[64..hp_wet_full.len() - 64];
        let hp_dry_e: f64 = hp_dry.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if hp_dry_e > 1.0e-8 * dry_e.max(1.0e-24) {
            let search_span = max_lag_samples.max((lin_lag.abs() as usize) + 16);
            if let Ok(hc2) = estimate_subsample_lag_and_gain(hp_dry, hp_wet, search_span) {
                let (hp_lag, hp_coh, _) =
                    refine_fractional_lag_hybrid(hp_dry, hp_wet, hc2.total_lag_samples, None);
                if hp_coh.abs() >= 0.36 && hp_lag >= -1.0 && (lin_lag - hp_lag) > 3.5 {
                    let inv = if lin_coherence >= 0.50 {
                        lin_gain < 0.0
                    } else {
                        hp_coh < 0.0
                    };
                    let guarded_lag = tau_coarse_opt.map_or(hp_lag, |tc| {
                        let clag = constrain_subsample_around_coarse(hp_lag, 0.0, tc);
                        guard_lag_causal_onset(dry, wet, clag, hp_lag, Some(tc))
                    });
                    return Some((guarded_lag, lin_gain, inv));
                }
            }
        }
    }
    None
}

/// Estimates sub-sample latency, signed gain, polarity, physical regime (`CleanLinear`,
/// `PolarityInverted`, `DispersiveAllPass`, or `NonlinearEnvelopeFallback`), and `Law 0.A`-gated
/// linear clock drift (`ppm`) with optional causal onset anchor `tau_coarse_opt` (`Delta_tau in
/// [-1.5, +1.5]`) (`ref[train.capture.alignment.multi-regime]`).
pub fn estimate_hybrid_multiregime_alignment(
    dry: &[f32],
    wet: &[f32],
    sample_rate_hz: u32,
    max_lag_samples: usize,
    tau_coarse_opt: Option<f64>,
    law0a_ltv_flag: bool,
) -> Result<MultiRegimeAlignment, AlignError> {
    if sample_rate_hz == 0 {
        return Err(AlignError::InvalidAlignmentInput(
            "sample_rate_hz must be >= 1".to_string(),
        ));
    }
    if dry.len() < 8 || dry.len() != wet.len() {
        return Err(AlignError::InvalidAlignmentInput(
            "dry and wet must have equal length >= 8".to_string(),
        ));
    }
    if dry.iter().any(|v| !v.is_finite()) || wet.iter().any(|v| !v.is_finite()) {
        return Err(AlignError::InvalidAlignmentInput(
            "dry and wet must contain finite samples".to_string(),
        ));
    }
    let tau_coarse_opt = tau_coarse_opt.filter(|v| v.is_finite());
    let inv_len = 1.0 / (dry.len() as f64);
    let mean_d = dry.iter().map(|&x| f64::from(x)).sum::<f64>() * inv_len;
    let mean_w = wet.iter().map(|&x| f64::from(x)).sum::<f64>() * inv_len;
    let dry_centered: Vec<f32> = dry
        .iter()
        .map(|&x| (f64::from(x) - mean_d) as f32)
        .collect();
    let wet_centered: Vec<f32> = wet
        .iter()
        .map(|&x| (f64::from(x) - mean_w) as f32)
        .collect();
    let dry = dry_centered.as_slice();
    let wet = wet_centered.as_slice();

    let coarse = estimate_subsample_lag_and_gain(dry, wet, max_lag_samples)?;
    let (unconstrained_lag, signed_coh, lin_gain) =
        refine_fractional_lag_hybrid(dry, wet, coarse.total_lag_samples, None);
    let lin_coherence = signed_coh.abs().clamp(0.0, 1.0);

    let dispersive_hit = detect_dispersive_wavefront(
        dry,
        wet,
        unconstrained_lag,
        lin_coherence,
        lin_gain,
        max_lag_samples,
        tau_coarse_opt,
    );

    let (regime, chosen_lag, chosen_gain, polarity_inverted) =
        if let Some((hf_lag, eff_gain, inv)) = dispersive_hit {
            (AlignmentRegime::DispersiveAllPass, hf_lag, eff_gain, inv)
        } else if lin_coherence < 0.35 {
            let (env_dry, env_wet) =
                compute_analytic_envelopes_stereo_pair(dry, wet, sample_rate_hz);
            let guarded_init = tau_coarse_opt.map_or(unconstrained_lag, |tc| {
                constrain_subsample_around_coarse(unconstrained_lag, 0.0, tc)
            });
            let (env_lag, raw_env_lag, env_coh) =
                match estimate_subsample_lag_and_gain(&env_dry, &env_wet, max_lag_samples) {
                    Ok(env_est) => {
                        let (elag, ecoh, _) = refine_fractional_lag_hybrid(
                            &env_dry,
                            &env_wet,
                            env_est.total_lag_samples,
                            tau_coarse_opt,
                        );
                        (elag, env_est.total_lag_samples, ecoh.abs())
                    }
                    Err(_) => (guarded_init, unconstrained_lag, 0.0),
                };
            let (cand_lag, cand_raw_lag) = if env_coh > lin_coherence {
                (env_lag, raw_env_lag)
            } else {
                (guarded_init, unconstrained_lag)
            };
            let use_lag = guard_lag_causal_onset(dry, wet, cand_lag, cand_raw_lag, tau_coarse_opt);
            let rms_dry = dry
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let rms_wet = wet
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            let inv = if lin_coherence >= 0.12 {
                signed_coh < 0.0
            } else {
                estimate_leading_edge_polarity(dry, wet, use_lag.round() as isize)
            };
            let mag_gain = if rms_dry > 1.0e-15 {
                rms_wet / rms_dry
            } else {
                0.0
            };
            (
                AlignmentRegime::NonlinearEnvelopeFallback,
                use_lag,
                if inv { -mag_gain } else { mag_gain },
                inv,
            )
        } else {
            let (final_lag, final_gain) = if let Some(tc) = tau_coarse_opt {
                let (glag, _, ggain) =
                    refine_fractional_lag_hybrid(dry, wet, coarse.total_lag_samples, Some(tc));
                let clag = guard_lag_causal_onset(dry, wet, glag, unconstrained_lag, Some(tc));
                let cgain = if (clag - glag).abs() > 1.0e-6 {
                    lin_gain
                } else {
                    ggain
                };
                (clag, cgain)
            } else {
                (unconstrained_lag, lin_gain)
            };
            if final_gain < 0.0 {
                (
                    AlignmentRegime::PolarityInverted,
                    final_lag,
                    final_gain,
                    true,
                )
            } else {
                (AlignmentRegime::CleanLinear, final_lag, final_gain, false)
            }
        };

    let ppm_est = estimate_ppm_clock_drift(dry, wet, max_lag_samples, law0a_ltv_flag);
    let (final_lag, final_gain, final_inv, final_regime, final_lin_coh, final_ppm) =
        if ppm_est.gated_applied {
            let wet_unskewed = realign_fractional_kaiser64(wet, 0.0, ppm_est.applied_ppm, false);
            if regime == AlignmentRegime::DispersiveAllPass {
                let re_est = estimate_subsample_lag_and_gain(dry, &wet_unskewed, max_lag_samples)
                    .ok()
                    .and_then(|cu| {
                        let (ulag, ucoh, ug) = refine_fractional_lag_hybrid(
                            dry,
                            &wet_unskewed,
                            cu.total_lag_samples,
                            None,
                        );
                        detect_dispersive_wavefront(
                            dry,
                            &wet_unskewed,
                            ulag,
                            ucoh.abs().clamp(0.0, 1.0),
                            ug,
                            max_lag_samples,
                            tau_coarse_opt,
                        )
                    });
                if let Some((dlag, dgain, dinv)) = re_est {
                    (
                        dlag,
                        dgain,
                        dinv,
                        AlignmentRegime::DispersiveAllPass,
                        lin_coherence,
                        ppm_est.applied_ppm,
                    )
                } else {
                    (
                        chosen_lag,
                        chosen_gain,
                        polarity_inverted,
                        regime,
                        lin_coherence,
                        0.0,
                    )
                }
            } else {
                let seed_lag = tau_coarse_opt.map_or(ppm_est.lag_at_zero_samples, |tc| {
                    constrain_subsample_around_coarse(ppm_est.lag_at_zero_samples, 0.0, tc)
                });
                let (ref_lag, ref_coh, ref_gain) =
                    refine_fractional_lag_hybrid(dry, &wet_unskewed, seed_lag, tau_coarse_opt);
                if ref_coh.abs() >= 0.35 {
                    let clag = guard_lag_causal_onset(
                        dry,
                        &wet_unskewed,
                        ref_lag,
                        ppm_est.lag_at_zero_samples,
                        tau_coarse_opt,
                    );
                    let (eff_coh, eff_gain) = if (clag - ref_lag).abs() > 1.0e-6 {
                        normalized_corr_at_fractional_lag(
                            dry,
                            &wet_unskewed,
                            ppm_est.lag_at_zero_samples,
                        )
                    } else {
                        (ref_coh, ref_gain)
                    };
                    let inv = eff_gain < 0.0;
                    let prom_regime = if inv {
                        AlignmentRegime::PolarityInverted
                    } else {
                        AlignmentRegime::CleanLinear
                    };
                    (
                        clag,
                        eff_gain,
                        inv,
                        prom_regime,
                        lin_coherence.max(eff_coh.abs().clamp(0.0, 1.0)),
                        ppm_est.applied_ppm,
                    )
                } else {
                    (
                        chosen_lag,
                        chosen_gain,
                        polarity_inverted,
                        regime,
                        lin_coherence,
                        0.0,
                    )
                }
            }
        } else {
            (
                chosen_lag,
                chosen_gain,
                polarity_inverted,
                regime,
                lin_coherence,
                0.0,
            )
        };

    Ok(MultiRegimeAlignment {
        regime: final_regime,
        total_lag_samples: final_lag,
        signed_gain: final_gain,
        polarity_inverted: final_inv,
        linear_coherence: final_lin_coh,
        clock_drift_ppm: final_ppm,
    })
}

/// Estimates sub-sample latency, signed gain, polarity, physical regime, and `Law 0.A`-compliant
/// linear clock drift (`ppm`) with unconstrained causal search (`ref[train.capture.alignment.multi-regime]`).
pub fn estimate_multiregime_alignment(
    dry: &[f32],
    wet: &[f32],
    sample_rate_hz: u32,
    max_lag_samples: usize,
) -> Result<MultiRegimeAlignment, AlignError> {
    estimate_hybrid_multiregime_alignment(dry, wet, sample_rate_hz, max_lag_samples, None, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_multiregime_alignment_tau_coarse_guard_and_ltv_ppm_blocking() {
        let seed = [0x93_u8; 256];
        let (dry, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 48_000, 1, 48_000,
        );
        let wet = realign_fractional_kaiser64(&dry, -37.38, -45.0, true);

        let lti = estimate_hybrid_multiregime_alignment(&dry, &wet, 48_000, 256, Some(37.0), false)
            .unwrap();
        assert_eq!(lti.regime, AlignmentRegime::PolarityInverted);
        assert!(lti.polarity_inverted);
        assert!(
            (lti.total_lag_samples - 37.38).abs() < 0.02,
            "expected lag 37.38 +/- 0.02, got {} (ppm={})",
            lti.total_lag_samples,
            lti.clock_drift_ppm
        );
        assert!(
            (lti.clock_drift_ppm - 45.0).abs() < 4.0,
            "expected ~45 ppm, got {}",
            lti.clock_drift_ppm
        );
        let aligned = realign_fractional_kaiser64(
            &wet,
            lti.total_lag_samples,
            lti.clock_drift_ppm,
            lti.polarity_inverted,
        );
        let (res_e, _, _) =
            refine_fractional_lag_hybrid(&dry[..12_000], &aligned[..12_000], 0.0, None);
        let (res_l, _, _) =
            refine_fractional_lag_hybrid(&dry[36_000..], &aligned[36_000..], 0.0, None);
        assert!(res_e.abs() < 0.02, "leading residual lag {res_e}");
        assert!(res_l.abs() < 0.02, "trailing residual lag {res_l}");

        let ltv = estimate_hybrid_multiregime_alignment(&dry, &wet, 48_000, 256, Some(37.0), true)
            .unwrap();
        assert_eq!(ltv.clock_drift_ppm, 0.0);

        // Full-wave rectifier (`|x|` / `x^2`) on full broadband multi-phrase coreset destroys linear
        // fundamental coherence (< 0.35), triggering Jacobi H1 stereo Hilbert envelope fallback.
        let (dry_24k, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 24_000, 1, 48_000,
        );
        let wet_nl: Vec<f32> = realign_fractional_kaiser64(&dry_24k, -37.38, 0.0, false)
            .into_iter()
            .map(|x| x.abs())
            .collect();
        let nl = estimate_hybrid_multiregime_alignment(
            &dry_24k,
            &wet_nl,
            48_000,
            256,
            Some(37.0),
            false,
        )
        .unwrap();
        assert_eq!(nl.regime, AlignmentRegime::NonlinearEnvelopeFallback);
        assert!(
            (nl.total_lag_samples - 37.38).abs() <= 1.5,
            "expected envelope lag ~37.38, got {}",
            nl.total_lag_samples
        );
    }

    #[test]
    fn test_align_regimes_guards_dispersive_hp_and_ppm_reanchor_paths() {
        assert!(realign_fractional_kaiser64(&[], 5.0, 0.0, false).is_empty());
        assert_eq!(
            realign_fractional_kaiser64(&[1.0, f32::NAN], 0.0, 0.0, true),
            vec![-1.0, 0.0]
        );
        assert!(!estimate_leading_edge_polarity(&[0.0; 16], &[1.0; 16], 0));
        let hp = second_diff_hp_wavefront(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(hp.len(), 4);
        assert!(
            detect_dispersive_wavefront(&[0.0; 32], &[0.0; 32], 0.0, 1.0, 1.0, 32, None).is_none()
        );

        let d8 = [0.0_f32; 8];
        assert!(estimate_hybrid_multiregime_alignment(&d8, &d8, 0, 8, None, false).is_err());
        assert!(
            estimate_hybrid_multiregime_alignment(&d8[..4], &d8[..4], 48_000, 8, None, false)
                .is_err()
        );
        let mut d_nan = [0.0_f32; 16];
        d_nan[3] = f32::NAN;
        assert!(
            estimate_hybrid_multiregime_alignment(&d_nan, &[0.0; 16], 48_000, 8, None, false)
                .is_err()
        );

        // Construct dispersive all-pass chain (low-band delayed +40 samples via cascaded APF)
        // with +40 ppm drift so `detect_dispersive_wavefront` triggers `DispersiveAllPass` and PPM re-anchor.
        let n = 48_000usize;
        let mut dry = vec![0.0_f32; n];
        for (i, s) in dry.iter_mut().enumerate().take(n - 1_024).skip(512) {
            let t = (i as f64) / 48_000.0;
            *s = (0.3 * (std::f64::consts::TAU * 90.0 * t).sin()
                + 0.25
                    * (if i == 2_000 || i == 24_000 || i == 40_000 {
                        1.0
                    } else {
                        0.0
                    })) as f32;
        }
        let mut apf = Vec::with_capacity(n);
        let (mut z1, mut z2) = (0.0_f64, 0.0_f64);
        let a = 0.85_f64;
        for &x in &dry {
            let y1 = -a * f64::from(x) + z1;
            z1 = f64::from(x) + a * y1;
            let y2 = -a * y1 + z2;
            z2 = y1 + a * y2;
            apf.push((0.3 * f64::from(x) + 0.7 * y2) as f32);
        }
        let wet_disp = realign_fractional_kaiser64(&apf, -18.2, -40.0, false);
        let est_disp =
            estimate_hybrid_multiregime_alignment(&dry, &wet_disp, 48_000, 256, None, false)
                .unwrap();
        assert!(est_disp.total_lag_samples.is_finite());

        // Weakly coherent shifted sine across short length hits low-coherence `applied_ppm == 0.0`.
        let short_d: Vec<f32> = (0..1_024)
            .map(|i| 0.2 * ((i as f32) * 0.11).sin())
            .collect();
        let short_w: Vec<f32> = (0..1_024)
            .map(|i| 0.05 * ((i as f32) * 0.37).sin())
            .collect();
        let low =
            estimate_hybrid_multiregime_alignment(&short_d, &short_w, 48_000, 64, None, false)
                .unwrap();
        assert_eq!(low.clock_drift_ppm, 0.0);
    }

    #[test]
    fn test_ppm_drift_causal_onset_clamp_preserves_gain_and_polarity() {
        let seed = [0x93_u8; 256];
        let (dry, _) = crate::stimulus::waveforms::synthesize_a5_musical_transient_coreset(
            &seed, 48_000, 1, 48_000,
        );
        // Causal 1-pole LPF (~4.5 kHz) preserves quarter coherence >= 0.75 for PPM lock while
        // introducing ~1 sample of minimum-phase group delay that Guard 2 clamps back to `tc`.
        let a_lp = 1.0 - (-2.0 * PI * 4_500.0 / 48_000.0).exp();
        let mut lp_st = 0.0_f64;
        let lpf_wet: Vec<f32> = dry
            .iter()
            .map(|&x| {
                lp_st += a_lp * (f64::from(x) - lp_st);
                lp_st as f32
            })
            .collect();

        for &(tc, invert) in &[(0.0_f64, false), (24.0_f64, true)] {
            let wet_ppm = realign_fractional_kaiser64(&lpf_wet, -tc, -45.0, invert);
            let unc =
                estimate_hybrid_multiregime_alignment(&dry, &wet_ppm, 48_000, 128, None, false)
                    .unwrap();
            assert!(
                unc.total_lag_samples > tc + 0.1,
                "unconstrained lag should reflect LPF group delay (> {tc} + 0.1), got {}",
                unc.total_lag_samples
            );
            let est =
                estimate_hybrid_multiregime_alignment(&dry, &wet_ppm, 48_000, 128, Some(tc), false)
                    .unwrap();
            assert_eq!(
                est.total_lag_samples, tc,
                "tc={tc} invert={invert} got lag={}",
                est.total_lag_samples
            );
            assert_eq!(est.polarity_inverted, invert, "tc={tc} invert={invert}");
            assert_eq!(est.signed_gain < 0.0, est.polarity_inverted);
            assert!(
                est.clock_drift_ppm > 20.0 && est.clock_drift_ppm < 60.0,
                "expected positive ~45 ppm lock (within LPF group-delay phrase variance), got {}",
                est.clock_drift_ppm
            );
        }
    }
}
