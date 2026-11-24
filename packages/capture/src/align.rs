#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Wasm32-safe in-memory 12/16/24/32-bit PCM & 32-bit IEEE-754 float WAV buffer codec,
//! canonical `<f4` SHA-256 hashing, and sub-sample loopback lag / polarity alignment (`align`).
//!
//! Operates strictly on `&[u8]`, `Vec<u8>`, and `&[f32]` slices with zero `std::fs` or OS audio I/O.

pub mod calibration;
pub mod regimes;
pub mod wav_codec;
pub mod wav_wire;

use sha2::{Digest, Sha256};
use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::F32;

pub use regimes::{
    ACOUSTIC_NULL_HOMING_MAX_DB, AlignmentRegime, CALB_STIMULUS_SAMPLES_48K, DecayTailDecision,
    DifferentialPadCalibration, HYBRID_SUBSAMPLE_MAX_DELTA, LIVE_DECAY_TAIL_MARGIN_DB,
    LoopbackCalibrationReport, MIC_PAD_PROBE_SAMPLES_48K, MultiRegimeAlignment, NullHomingReport,
    PARABOLIC_PLATEAU_EPS, PPM_DEADBAND_MIN, PPM_MAX_ABS, PPM_MIN_SAMPLES,
    PPM_QUARTER_MIN_COHERENCE, PpmDriftEstimate, THERMAL_RECOVERY_MAX_DB, ThermalRecoveryReport,
    calibrate_loopback_calb, constrain_subsample_around_coarse,
    estimate_hybrid_multiregime_alignment, estimate_multiregime_alignment,
    estimate_ppm_clock_drift, evaluate_acoustic_null_homing, evaluate_live_decay_tail,
    evaluate_thermal_recovery_gate, generate_calb_loopback_stimulus, generate_mic_pad_probe_400ms,
    measure_differential_pad_ratio, parabolic_peak_offset, realign_fractional_kaiser64,
    rescale_padded_stem,
};
pub use wav_codec::{
    decode_wav_bytes, encode_wav_12bit, encode_wav_16bit, encode_wav_24bit, encode_wav_32bit_float,
    encode_wav_32bit_int, encode_wav_with_bit_depth,
};

/// Errors returned by in-memory WAV PCM buffer encoding/decoding or loopback alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignError {
    /// Invalid RIFF/WAVE header, chunk framing, or PCM bit-depth metadata.
    InvalidWav(String),
    /// Invalid alignment stimulus/response slice lengths or zero-energy reference.
    InvalidAlignmentInput(String),
}

impl core::fmt::Display for AlignError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidWav(msg) => write!(f, "invalid WAV buffer: {msg}"),
            Self::InvalidAlignmentInput(msg) => write!(f, "invalid alignment input: {msg}"),
        }
    }
}

impl std::error::Error for AlignError {}

/// Decoded interleaved floating-point PCM WAV buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Channel count (`1` = mono, `2` = stereo, etc.).
    pub channels: u16,
    /// Interleaved floating-point samples.
    pub samples: Vec<f32>,
}

/// Estimated sub-sample round-trip loopback alignment, linear gain, and polarity flag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubSampleAlignment {
    /// Integer peak correlation lag in samples (`wet` delayed relative to `dry`).
    pub integer_lag_samples: isize,
    /// Parabolic fractional lag refinement in `[-0.5, 0.5]` samples.
    pub fractional_lag_samples: f64,
    /// Total sub-sample lag (`integer_lag_samples as f64 + fractional_lag_samples`).
    pub total_lag_samples: f64,
    /// Estimated signed linear loopback gain (`< 0.0` indicates inverted polarity).
    pub signed_gain: f64,
    /// Whether the wet return is polarity-inverted (`signed_gain < 0.0`).
    pub polarity_inverted: bool,
}

/// Computes the canonical IEEE-754 little-endian `<f4` SHA-256 hex digest of `samples`
/// (normalizing `-0.0` and `NaN` to `+0.0`).
#[must_use]
pub fn compute_pcm_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &sample in samples {
        let canonical = if sample == 0.0 || sample.is_nan() {
            0.0_f32
        } else {
            sample
        };
        hasher.update(F32::new(canonical).as_bytes());
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        use core::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    hex
}

/// Estimates sub-sample loopback lag, signed gain, and polarity inversion between `dry` and `wet`
/// using cross-correlation over `[-max_lag_samples, +max_lag_samples]` with 3-point parabolic
/// magnitude refinement (`r[train.capture.sub-sample-align]`).
pub fn estimate_subsample_lag_and_gain(
    dry: &[f32],
    wet: &[f32],
    max_lag_samples: usize,
) -> Result<SubSampleAlignment, AlignError> {
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
    let n = dry.len();
    let max_lag = max_lag_samples.min(n / 2);
    let total_e_dry: f64 = dry.iter().map(|&d| f64::from(d) * f64::from(d)).sum();
    if total_e_dry <= 1.0e-18 {
        return Err(AlignError::InvalidAlignmentInput(
            "dry reference stimulus has near-zero energy".to_string(),
        ));
    }
    let total_e_wet: f64 = wet.iter().map(|&w| f64::from(w) * f64::from(w)).sum();
    if total_e_wet <= 1.0e-18 {
        return Ok(SubSampleAlignment {
            integer_lag_samples: 0,
            fractional_lag_samples: 0.0,
            total_lag_samples: 0.0,
            signed_gain: 0.0,
            polarity_inverted: false,
        });
    }
    let min_ed = 1.0e-6 * total_e_dry;
    let min_ew = 1.0e-6 * total_e_wet;

    let corr_at = |lag: isize| -> (f64, f64, f64) {
        let start = (-lag).max(0) as usize;
        let end = (n as isize - lag).min(n as isize).max(0) as usize;
        if end <= start {
            return (0.0, 0.0, 0.0);
        }
        let wet_start = (start as isize + lag) as usize;
        let mut dot = 0.0_f64;
        let mut e_dry = 0.0_f64;
        let mut e_wet = 0.0_f64;
        for (&d, &w) in dry[start..end].iter().zip(&wet[wet_start..]) {
            let d64 = f64::from(d);
            let w64 = f64::from(w);
            dot += d64 * w64;
            e_dry += d64 * d64;
            e_wet += w64 * w64;
        }
        let norm_abs = if e_dry >= min_ed && e_wet >= min_ew {
            dot.abs() / (e_dry * e_wet).sqrt()
        } else {
            0.0
        };
        (norm_abs, dot, e_dry)
    };

    let mut best_lag: isize = 0;
    let mut best_norm_corr = -1.0_f64;
    let mut best_dot = 0.0_f64;
    let mut best_energy = 0.0_f64;

    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let (norm_abs, dot, e_dry) = corr_at(lag);
        if norm_abs > best_norm_corr {
            best_norm_corr = norm_abs;
            best_lag = lag;
            best_dot = dot;
            best_energy = e_dry;
        }
    }

    let frac = if best_lag > -(max_lag as isize) && best_lag < (max_lag as isize) {
        let y_m1 = corr_at(best_lag - 1).0;
        let y_0 = best_norm_corr;
        let y_p1 = corr_at(best_lag + 1).0;
        regimes::parabolic_peak_offset(y_m1, y_0, y_p1)
    } else {
        0.0
    };

    let signed_gain = if best_energy > 1.0e-18 {
        best_dot / best_energy
    } else {
        0.0
    };
    Ok(SubSampleAlignment {
        integer_lag_samples: best_lag,
        fractional_lag_samples: frac,
        total_lag_samples: (best_lag as f64) + frac,
        signed_gain,
        polarity_inverted: signed_gain < 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_display_and_subsample_lag_edge_cases() {
        assert_eq!(
            compute_pcm_sha256(&[f32::NAN, -0.0, 0.25]),
            compute_pcm_sha256(&[0.0, 0.0, 0.25])
        );
        assert!(format!("{}", AlignError::InvalidWav("x".into())).contains("invalid WAV"));
        assert!(
            format!("{}", AlignError::InvalidAlignmentInput("y".into()))
                .contains("invalid alignment")
        );

        let n = 256;
        let dry: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.23).sin() + 0.5 * (i as f32 * 0.71).cos()) * 0.4)
            .collect();
        let mut wet = vec![0.0_f32; n];
        for i in 5..n {
            wet[i] = -0.8 * dry[i - 5];
        }
        let est = estimate_subsample_lag_and_gain(&dry, &wet, 16).unwrap();
        assert_eq!(est.integer_lag_samples, 5);
        assert!(est.polarity_inverted);
        assert!((est.signed_gain + 0.8).abs() < 0.05);

        // Silent wet, edge impulse (insufficient overlap energy), and invalid lengths/NaN
        let silent = estimate_subsample_lag_and_gain(&dry, &vec![0.0; n], 16).unwrap();
        assert_eq!(silent.total_lag_samples, 0.0);
        assert_eq!(silent.signed_gain, 0.0);

        let mut impulse_d = vec![0.0_f32; 32];
        let mut impulse_w = vec![0.0_f32; 32];
        impulse_d[0] = 1.0;
        impulse_w[31] = 1.0;
        let edge_imp = estimate_subsample_lag_and_gain(&impulse_d, &impulse_w, 4).unwrap();
        assert_eq!(edge_imp.signed_gain, 0.0);

        assert!(estimate_subsample_lag_and_gain(&[0.0; 4], &[0.0; 4], 2).is_err());
        assert!(estimate_subsample_lag_and_gain(&[0.0; 16], &[0.0; 16], 4).is_err());
        assert!(estimate_subsample_lag_and_gain(&[f32::NAN; 16], &[0.1; 16], 4).is_err());
    }
}
