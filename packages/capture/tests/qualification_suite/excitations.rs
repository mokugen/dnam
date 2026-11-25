#![forbid(unsafe_code)]
//! Deterministic unseen holdout waveform synthesis and off-knot intermediate control vector
//! builders for 3-level (`L1` / `L2` / `L3`) qualification (`M25`).

use std::f64::consts::TAU;

use dnam_capture::metrics::canonical_ltas_33_centers_hz;
use dnam_capture::stimulus::TopologyProbeMode;
use dnam_capture::stimulus::waveforms::{
    deterministic_seed_256, synthesize_a5_musical_transient_coreset,
};
use dnam_fixtures::FixtureDeviceMetadata;

/// Unseen multi-control / switch evaluation operating point strictly off Lobatto-3 training knots.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UnseenControlSetting {
    /// Human-readable setting label for diagnostic reports.
    pub label: String,
    /// Full `[range..., switch_idx...]` control vector fed to adapter and `DnamEngine`.
    pub control_values: Vec<f32>,
    /// True when at least one switch moves away from its default index.
    pub flips_switch: bool,
}

fn build_broadband_ltas_burst(num_frames: usize, phase_skew: f64) -> Vec<f32> {
    if num_frames == 0 {
        return Vec::new();
    }
    let centers = canonical_ltas_33_centers_hz();
    let sr = 48_000.0_f64;
    let mut out = Vec::with_capacity(num_frames);
    let mut peak = 0.0_f64;
    let raw: Vec<f64> = (0..num_frames)
        .map(|i| {
            let t = (i as f64) / sr;
            let burst_pos = (i % 6_144) as f64 / 6_144.0;
            let env = 0.25 + 0.75 * (-3.2 * burst_pos).exp();
            let swell = 0.75 + 0.25 * (TAU * 0.9 * t + phase_skew).sin();
            let mut sum = 0.0_f64;
            for (k, &fc) in centers.iter().enumerate() {
                if fc >= 0.45 * sr {
                    continue;
                }
                let ph = ((k + 1) as f64) * core::f64::consts::GOLDEN_RATIO + phase_skew;
                sum += (TAU * fc * t + ph).sin();
            }
            let sample = (sum / 18.0) * env * swell;
            peak = peak.max(sample.abs());
            sample
        })
        .collect();
    let scale = if peak > 1.0e-9 { 0.48 / peak } else { 1.0 };
    for s in raw {
        out.push((s * scale) as f32);
    }
    out
}

/// Builds a deterministic unseen `48 kHz` full-bandwidth holdout stimulus distinct from
/// `Schoukens–Pintelon` training coresets, combining an unseen `A5` musical phrase (`50%`) with
/// a `33`-band broadband burst/swell stream (`50%`) to excite `bESR`, `MR-IIR`, `LTAS`, and `SegESR`.
#[must_use]
pub(crate) fn build_unseen_holdout_excitation_48k(
    seed_tag: u64,
    topology: TopologyProbeMode,
    num_frames: usize,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let stereo_in = matches!(
        topology,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let ch_count = if stereo_in { 2 } else { 1 };
    let half = num_frames / 2;
    let rest = num_frames.saturating_sub(half);
    let seed = deterministic_seed_256(seed_tag);
    let (a5_l, a5_r) = synthesize_a5_musical_transient_coreset(&seed, half, ch_count, 48_000);
    let bb_l = build_broadband_ltas_burst(rest, 0.0);

    let mut dry_l = Vec::with_capacity(num_frames);
    dry_l.extend_from_slice(&a5_l);
    dry_l.extend_from_slice(&bb_l);

    let dry_r = if stereo_in {
        let bb_r = build_broadband_ltas_burst(rest, 1.15);
        let mut r_buf = Vec::with_capacity(num_frames);
        r_buf.extend_from_slice(a5_r.as_deref().unwrap_or(&a5_l));
        r_buf.extend_from_slice(&bb_r);
        Some(r_buf)
    } else {
        None
    };
    (dry_l, dry_r)
}

fn off_knot_interior_coord(default_norm: f32, preferred: f32, fallback: f32) -> f32 {
    if (preferred - default_norm).abs() >= 0.08 {
        preferred
    } else {
        fallback
    }
}

/// Generates unseen interior control vectors strictly off the Lobatto-3 training grid `{-1, 0, +1}`.
#[must_use]
pub(crate) fn build_unseen_control_vectors(
    meta: &FixtureDeviceMetadata,
) -> Vec<UnseenControlSetting> {
    let n_range = meta.controls.len();
    let n_sw = meta.switches.len();
    if n_range == 0 && n_sw == 0 {
        return Vec::new();
    }

    let base_switches: Vec<f32> = meta
        .switches
        .iter()
        .map(|s| f32::from(s.default_index))
        .collect();
    let mut settings = Vec::new();

    if n_range > 0 {
        let specs = [
            ("interior_low", -0.56_f32, -0.34_f32),
            ("interior_high", 0.58_f32, 0.36_f32),
            ("interior_mixed", 0.48_f32, -0.43_f32),
        ];
        for &(name, even_v, odd_v) in &specs {
            let mut vec_vals = Vec::with_capacity(n_range + n_sw);
            for (idx, c) in meta.controls.iter().enumerate() {
                let target = if idx % 2 == 0 { even_v } else { odd_v };
                let alt = if target > 0.0 { 0.27 } else { -0.27 };
                vec_vals.push(off_knot_interior_coord(c.default_norm, target, alt));
            }
            vec_vals.extend_from_slice(&base_switches);
            settings.push(UnseenControlSetting {
                label: name.to_string(),
                control_values: vec_vals,
                flips_switch: false,
            });
        }
    }

    for (sw_idx, sw) in meta.switches.iter().enumerate() {
        if sw.options.len() < 2 {
            continue;
        }
        let alt_opt = (usize::from(sw.default_index) + 1) % sw.options.len();
        let mut vec_vals = Vec::with_capacity(n_range + n_sw);
        for (idx, c) in meta.controls.iter().enumerate() {
            let target = if idx % 2 == 0 { 0.37 } else { -0.39 };
            let alt = if target > 0.0 { 0.24 } else { -0.24 };
            vec_vals.push(off_knot_interior_coord(c.default_norm, target, alt));
        }
        let mut sw_vals = base_switches.clone();
        sw_vals[sw_idx] = alt_opt as f32;
        vec_vals.extend_from_slice(&sw_vals);
        settings.push(UnseenControlSetting {
            label: format!("switch_{}_opt_{alt_opt}", sw.id),
            control_values: vec_vals,
            flips_switch: true,
        });
    }

    settings
}
