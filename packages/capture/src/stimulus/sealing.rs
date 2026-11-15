#![forbid(unsafe_code)]
//! `INPC`/`STMC` Capsule Sealing Helpers, Freeze-Transition Stem Synthesis, Control-Meta
//! Mapping, and `400 ms` Differential Mic-Pad Re-Capture (`stimulus::sealing`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md`
//!   (`ref[train.capture.planner.freeze-transition]`, `ref[train.capture.calib.clean-pad]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

use std::collections::BTreeMap;

use dnam_engine::schema::DnamError;

use crate::CORESET_FP16_FRAMES_PER_STEM;
use crate::align::{
    compute_pcm_sha256, estimate_multiregime_alignment, realign_fractional_kaiser64,
};
use crate::proj::capsules::build_proj_inpc_capsule;
use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::vault::VaultCaptureSpec;

use super::ADAPTIVE_FRAME_SAMPLES;
use super::waveforms::{compute_stmc_psd_88_and_spectral_4096, encode_coreset_12k_fp16};

pub(crate) mod controls;
pub(crate) mod freeze_and_pad;

pub(crate) use controls::{
    TrajectoryInterpSpec, build_controls_meta_map, control_declared_bounds, default_control_vector,
    interpolate_frame_trajectories,
};
pub use freeze_and_pad::recapture_clipped_stem;
pub(crate) use freeze_and_pad::{
    build_freeze_transition_stem_stimulus, normalize_stem_channel_to_48k,
    resample_48k_channels_to_sr,
};

/// `3 * 4,608 = 13,824` samples at `48 kHz` (`12,000` `A5` coreset frames + `1,824` settle zeros).
pub(crate) const CORESET_WITH_SETTLE_48K: usize = 3 * ADAPTIVE_FRAME_SAMPLES;
const SETTLE_PAD_SAMPLES_48K: usize = CORESET_WITH_SETTLE_48K - CORESET_FP16_FRAMES_PER_STEM;

/// Cohesive inputs for encoding `INPC` dry provenance and `STMC` `VaultCaptureSpec` metadata.
pub(crate) struct StemSealRequest<'a> {
    pub stem_id: &'a str,
    pub is_star_center: bool,
    pub seed_256: &'a [u8; 256],
    pub carrier_bins: &'a [usize],
    pub valid_frames: usize,
    pub dry_48k_l: &'a [f32],
    pub dry_48k_r: Option<&'a [f32]>,
    pub wet_48k_l: &'a [f32],
    pub wet_48k_r: Option<&'a [f32]>,
    pub windows: Vec<StmcTrajectoryWindow>,
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    pub user_metadata: BTreeMap<String, serde_json::Value>,
}

struct AlignedCoresetSlices<'a> {
    dry_12k_l: &'a [f32],
    dry_12k_r: Option<&'a [f32]>,
    wet_12k_l: &'a [f32],
    wet_12k_r: Option<&'a [f32]>,
    lag_samples: f64,
    inv_pol: bool,
}

fn steady_silent_dc(dry_48k: &[f32], wet_48k: &[f32], start: usize, end: usize) -> Option<f64> {
    let mut vals = Vec::new();
    for i in start..end {
        if dry_48k[i].abs() < 1.0e-4 {
            vals.push(f64::from(wet_48k[i]));
        }
    }
    if vals.len() < 64 {
        return None;
    }
    let tail = &vals[vals.len().saturating_sub(512)..];
    let mean = tail.iter().sum::<f64>() / (tail.len() as f64);
    if mean.abs() <= 1.0e-3 {
        return None;
    }
    let max_dev = tail.iter().fold(0.0_f64, |m, &v| m.max((v - mean).abs()));
    (max_dev < 0.25 * mean.abs()).then_some(mean)
}

fn estimate_quiet_tail_peak(
    dry_48k: &[f32],
    wet_centered: &[f32],
    n: usize,
    settle_start: usize,
    first_dry_active: usize,
) -> f64 {
    if first_dry_active >= 64 {
        return wet_centered[..first_dry_active]
            .iter()
            .fold(0.0_f64, |m, &w| m.max(f64::from(w).abs()));
    }
    let tail_start = n.saturating_sub(512).max(settle_start);
    let tail_dry_silent =
        tail_start < n && dry_48k[tail_start..n].iter().all(|&x| x.abs() < 1.0e-4);
    if !tail_dry_silent {
        return 0.0;
    }
    let mid = tail_start + (n - tail_start) / 2;
    let tail_late = wet_centered[mid..n]
        .iter()
        .fold(0.0_f64, |m, &w| m.max(f64::from(w).abs()));
    let tail_early = wet_centered[tail_start..mid]
        .iter()
        .fold(0.0_f64, |m, &w| m.max(f64::from(w).abs()));
    if tail_early > 2.0 * tail_late && tail_late < 5.0e-3 {
        0.0
    } else {
        tail_late
    }
}

fn estimate_causal_onset_gap(dry_48k: &[f32], wet_centered: &[f32], quiet_peak: f64) -> f64 {
    let n = dry_48k.len().min(wet_centered.len());
    let dry_peak = dry_48k[..n]
        .iter()
        .fold(0.0_f64, |m, &x| m.max(f64::from(x).abs()));
    let wet_peak = wet_centered[..n]
        .iter()
        .fold(0.0_f64, |m, &x| m.max(f64::from(x).abs()));
    if dry_peak <= 1.0e-4 || wet_peak <= 1.0e-4 {
        return 0.0;
    }
    let dry_thresh = (0.002 * dry_peak).max(2.5e-4);
    let n_dry_idx = dry_48k[..n]
        .iter()
        .position(|&x| f64::from(x).abs() >= dry_thresh)
        .unwrap_or(0);
    let m_init = (n_dry_idx + 6).clamp(4, n);
    let e_d_init: f64 = dry_48k[..m_init]
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum();
    let e_w_init: f64 = wet_centered[..m_init]
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum();
    let g_init = if e_d_init > 1.0e-12 {
        (e_w_init / e_d_init).sqrt()
    } else {
        1.0
    };
    let min_wet_floor = if g_init < 0.30 { 1.0e-4 } else { 2.5e-4 };
    let wet_thresh = (3.5 * quiet_peak).max(min_wet_floor).max(1.0e-4 * wet_peak);
    let n_wet_onset = wet_centered[..n]
        .iter()
        .position(|&x| f64::from(x).abs() >= wet_thresh)
        .unwrap_or(0) as f64;
    (n_wet_onset - (n_dry_idx as f64)).max(0.0)
}

/// Estimates `(bulk_lag_samples, polarity_inverted, dc_offset)` causally across dry & wet `48 kHz` stems.
pub(crate) fn estimate_causal_bulk_alignment(dry_48k: &[f32], wet_48k: &[f32]) -> (f64, bool, f64) {
    let n = dry_48k.len().min(wet_48k.len());
    if n < 64 {
        return (0.0, false, 0.0);
    }
    let settle_start = n.saturating_sub(SETTLE_PAD_SAMPLES_48K);
    let first_dry_active = dry_48k[..n]
        .iter()
        .position(|&x| x.abs() >= 1.0e-3)
        .unwrap_or(0);
    let dc_offset = steady_silent_dc(dry_48k, wet_48k, settle_start, n)
        .or_else(|| {
            (first_dry_active >= 64)
                .then(|| steady_silent_dc(dry_48k, wet_48k, 0, first_dry_active))
                .flatten()
        })
        .unwrap_or(0.0);

    let wet_centered_buf: Vec<f32>;
    let wet_centered: &[f32] = if dc_offset.abs() > 0.0 {
        wet_centered_buf = wet_48k[..n].iter().map(|&w| w - dc_offset as f32).collect();
        &wet_centered_buf
    } else {
        &wet_48k[..n]
    };

    let quiet_peak =
        estimate_quiet_tail_peak(dry_48k, wet_centered, n, settle_start, first_dry_active);
    let causal_gap = estimate_causal_onset_gap(dry_48k, wet_centered, quiet_peak);

    let (raw_lag, raw_inv_pol, lin_coh) =
        match estimate_multiregime_alignment(&dry_48k[..n], wet_centered, 48_000, 1_024) {
            Ok(al) => (
                al.total_lag_samples,
                al.polarity_inverted,
                al.linear_coherence,
            ),
            Err(_) => (0.0, false, 0.0),
        };
    let upper = if lin_coh >= 0.35 {
        causal_gap + 12.0
    } else {
        causal_gap + 2.0
    };
    let (bulk_lag, inv_pol) = if raw_lag > upper {
        let shift = (causal_gap.round().max(0.0) as usize).min(n.saturating_sub(1));
        let mut num = 0.0_f64;
        let mut den_d = 0.0_f64;
        let mut den_w = 0.0_f64;
        for t in 0..n.saturating_sub(shift) {
            let d = f64::from(dry_48k[t]);
            let w = f64::from(wet_centered[t + shift]);
            num += d * w;
            den_d += d * d;
            den_w += w * w;
        }
        let denom = (den_d * den_w).sqrt();
        let causal_inv = denom > 1.0e-12 && (num / denom) < -0.05;
        (causal_gap, causal_inv)
    } else {
        (raw_lag.max(0.0), raw_inv_pol)
    };
    (bulk_lag, inv_pol, dc_offset)
}

fn build_inpc_and_vault_spec(
    mut req: StemSealRequest<'_>,
    slices: &AlignedCoresetSlices<'_>,
) -> Result<(Vec<u8>, VaultCaptureSpec, f64, bool), DnamError> {
    let dry_fp16 = encode_coreset_12k_fp16(slices.dry_12k_l, slices.dry_12k_r);
    let inpc_channels = if slices.dry_12k_r.is_some() {
        2_u16
    } else {
        1_u16
    };

    let mut schedule_320 = [0u8; 320];
    schedule_320[0..4].copy_from_slice(&(req.carrier_bins.len() as u32).to_le_bytes());
    schedule_320[4..8].copy_from_slice(&(req.valid_frames as u32).to_le_bytes());
    for (idx, &b) in req.carrier_bins.iter().take(76).enumerate() {
        let off = 16 + idx * 4;
        schedule_320[off..off + 4].copy_from_slice(&(b as u32).to_le_bytes());
    }
    let inpc_bytes =
        build_proj_inpc_capsule(inpc_channels, req.seed_256, &schedule_320, &dry_fp16)?;

    let wet_fp16 = encode_coreset_12k_fp16(slices.wet_12k_l, slices.wet_12k_r);
    let (psd_88, spec_4096) =
        compute_stmc_psd_88_and_spectral_4096(slices.wet_12k_l, slices.wet_12k_r);

    let provenance = crate::proj::schema::StemProvenance {
        input_pcm_sha256: compute_pcm_sha256(slices.dry_12k_l),
        target_pcm_sha256: compute_pcm_sha256(
            &slices.wet_12k_l[..slices.wet_12k_l.len().min(CORESET_FP16_FRAMES_PER_STEM)],
        ),
        num_frames: CORESET_FP16_FRAMES_PER_STEM as u64,
    };
    req.user_metadata.insert(
        "wet_polarity_inverted".to_string(),
        serde_json::Value::Bool(slices.inv_pol),
    );

    let spec = VaultCaptureSpec {
        stem_id: req.stem_id.to_string(),
        is_star_center: req.is_star_center,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: slices.lag_samples as f32,
        controls_meta: req.controls_meta,
        provenance,
        user_metadata: req.user_metadata,
        extracted_physics: None,
        windows: req.windows,
        band_psd_summary_88: psd_88,
        n_comb_bins: req.carrier_bins.len() as u32,
        spectral_summary_4096: spec_4096,
        wet_coreset_fp16: wet_fp16,
        optional_spcm: None,
    };
    Ok((inpc_bytes, spec, slices.lag_samples, slices.inv_pol))
}

/// Seals `12,000`-frame dry/wet coresets into `(INPC, VaultCaptureSpec)` with optional shared alignment override.
pub(crate) fn seal_coresets_to_inpc_and_vault_with_override(
    req: StemSealRequest<'_>,
    shared_align: Option<(f64, bool)>,
) -> Result<(Vec<u8>, VaultCaptureSpec, f64, bool), DnamError> {
    let dry_12k_l = &req.dry_48k_l[..CORESET_FP16_FRAMES_PER_STEM];
    let dry_12k_r = req.dry_48k_r.map(|r| &r[..CORESET_FP16_FRAMES_PER_STEM]);
    let dry_ref = &req.dry_48k_l[..req.wet_48k_l.len().min(req.dry_48k_l.len())];
    let wet_ref = &req.wet_48k_l[..dry_ref.len()];
    let (lag_samples, inv_pol) = shared_align.unwrap_or_else(|| {
        let (lag, pol, _) = estimate_causal_bulk_alignment(dry_ref, wet_ref);
        (lag, pol)
    });

    let aligned_wet_l = realign_fractional_kaiser64(req.wet_48k_l, lag_samples, 0.0, inv_pol);
    let aligned_wet_r = req
        .wet_48k_r
        .filter(|r| !r.is_empty())
        .map(|r| realign_fractional_kaiser64(r, lag_samples, 0.0, inv_pol));

    let slices = AlignedCoresetSlices {
        dry_12k_l,
        dry_12k_r,
        wet_12k_l: &aligned_wet_l,
        wet_12k_r: aligned_wet_r.as_deref(),
        lag_samples,
        inv_pol,
    };
    build_inpc_and_vault_spec(req, &slices)
}

fn locate_coreset_align_span(dry_48k_l: &[f32]) -> (usize, usize) {
    const RING_GUARD: usize = 128;
    let has_leading_coreset = dry_48k_l.len() >= CORESET_WITH_SETTLE_48K + ADAPTIVE_FRAME_SAMPLES
        && dry_48k_l
            [CORESET_FP16_FRAMES_PER_STEM + RING_GUARD..CORESET_WITH_SETTLE_48K - RING_GUARD]
            .iter()
            .all(|&v| v.abs() < 1.0e-3)
        && dry_48k_l[..CORESET_FP16_FRAMES_PER_STEM]
            .iter()
            .any(|&v| v.abs() > 1.0e-3);
    let has_settle_pad = dry_48k_l.len() >= CORESET_WITH_SETTLE_48K
        && dry_48k_l[dry_48k_l.len() - SETTLE_PAD_SAMPLES_48K + RING_GUARD..]
            .iter()
            .all(|&v| v.abs() < 1.0e-3);
    let tail_span = if has_settle_pad {
        CORESET_WITH_SETTLE_48K
    } else {
        CORESET_FP16_FRAMES_PER_STEM
    };
    let core_start = if has_leading_coreset {
        0
    } else {
        dry_48k_l.len().saturating_sub(tail_span)
    };
    let align_span = if has_leading_coreset {
        CORESET_WITH_SETTLE_48K
    } else {
        tail_span
    };
    (core_start, align_span)
}

/// Aligns a full `48 kHz`-normalized stem on its aperiodic `A5` coreset+settle window FIRST
/// (handling bulk converter/DSP latency without periodic comb aliases) before slicing `12,000` frames,
/// accepting an optional `(lag, inv_pol)` override.
pub(crate) fn align_full_stem_and_seal_vault_with_override(
    req: StemSealRequest<'_>,
    shared_align: Option<(f64, bool)>,
) -> Result<(Vec<u8>, VaultCaptureSpec, f64, bool), DnamError> {
    let (core_start, align_span) = locate_coreset_align_span(req.dry_48k_l);
    let dry_align_slice =
        &req.dry_48k_l[core_start..(core_start + align_span).min(req.dry_48k_l.len())];
    let wet_align_end = req.wet_48k_l.len().min(core_start + dry_align_slice.len());
    let wet_align_slice = if core_start < wet_align_end {
        &req.wet_48k_l[core_start..wet_align_end]
    } else {
        &[]
    };
    let common_len = dry_align_slice.len().min(wet_align_slice.len());
    let (lag_samples, inv_pol) = shared_align.unwrap_or_else(|| {
        let (lag, pol, _) = estimate_causal_bulk_alignment(
            &dry_align_slice[..common_len],
            &wet_align_slice[..common_len],
        );
        (lag, pol)
    });

    let full_aligned_l = realign_fractional_kaiser64(req.wet_48k_l, lag_samples, 0.0, inv_pol);
    let full_aligned_r = req
        .wet_48k_r
        .filter(|r| !r.is_empty())
        .map(|r| realign_fractional_kaiser64(r, lag_samples, 0.0, inv_pol));

    let dry_12k_l = &req.dry_48k_l[core_start..core_start + CORESET_FP16_FRAMES_PER_STEM];
    let dry_12k_r = req
        .dry_48k_r
        .filter(|r| r.len() >= core_start + CORESET_FP16_FRAMES_PER_STEM)
        .map(|r| &r[core_start..core_start + CORESET_FP16_FRAMES_PER_STEM]);

    let wet_start = core_start.min(
        full_aligned_l
            .len()
            .saturating_sub(CORESET_FP16_FRAMES_PER_STEM),
    );
    let wet_end = (wet_start + CORESET_FP16_FRAMES_PER_STEM).min(full_aligned_l.len());
    let wet_12k_l = &full_aligned_l[wet_start..wet_end];
    let wet_12k_r = full_aligned_r.as_deref().map(|r| {
        let ws = core_start.min(r.len().saturating_sub(CORESET_FP16_FRAMES_PER_STEM));
        let we = (ws + CORESET_FP16_FRAMES_PER_STEM).min(r.len());
        &r[ws..we]
    });

    let slices = AlignedCoresetSlices {
        dry_12k_l,
        dry_12k_r,
        wet_12k_l,
        wet_12k_r,
        lag_samples,
        inv_pol,
    };
    build_inpc_and_vault_spec(req, &slices)
}

#[cfg(test)]
mod tests {
    use super::super::waveforms::synthesize_a5_musical_transient_coreset;
    use super::*;

    #[test]
    fn test_steady_silent_dc_and_causal_bulk_alignment_paths() {
        assert!(steady_silent_dc(&[0.0; 16], &[0.1; 16], 0, 16).is_none());
        assert!(steady_silent_dc(&[0.0; 128], &[1.0e-4; 128], 0, 128).is_none());
        let mut noisy = vec![0.1; 128];
        noisy[127] = 0.25;
        assert!(steady_silent_dc(&[0.0; 128], &noisy, 0, 128).is_none());
        let dc_found = steady_silent_dc(&[0.0; 128], &[0.05; 128], 0, 128).unwrap();
        assert!((dc_found - 0.05).abs() < 1.0e-6);

        assert_eq!(
            estimate_causal_bulk_alignment(&[0.0; 16], &[0.0; 16]),
            (0.0, false, 0.0)
        );

        // Delayed + inverted + DC-offset signal with silent lead-in & trailing settle tail
        let seed = [0x42u8; 256];
        let (core, _) = synthesize_a5_musical_transient_coreset(&seed, 8_192, 1, 48_000);
        let mut dry = vec![0.0_f32; 512];
        dry.extend_from_slice(&core);
        dry.resize(dry.len() + SETTLE_PAD_SAMPLES_48K, 0.0);

        let delay = 24_usize;
        let dc = 0.02_f32;
        let mut wet = vec![dc; dry.len()];
        for i in delay..dry.len() {
            wet[i] = dc - 0.8 * dry[i - delay];
        }
        let (lag, inv, dc_est) = estimate_causal_bulk_alignment(&dry, &wet);
        assert!((dc_est - f64::from(dc)).abs() < 1.0e-3);
        assert!(inv);
        assert!((lag - (delay as f64)).abs() < 2.0);
    }

    #[test]
    fn test_seal_coresets_and_align_full_stem_variants() {
        let seed = [0x71u8; 256];
        let (dry_core, r_opt) =
            synthesize_a5_musical_transient_coreset(&seed, CORESET_FP16_FRAMES_PER_STEM, 2, 48_000);
        let dry_r = r_opt.unwrap();
        let meta = BTreeMap::new();
        let win = vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 0.5,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }];

        let (inpc_a, spec_a, _, _) = seal_coresets_to_inpc_and_vault_with_override(
            StemSealRequest {
                stem_id: "seal_a",
                is_star_center: true,
                seed_256: &seed,
                carrier_bins: &[6, 12],
                valid_frames: 2,
                dry_48k_l: &dry_core,
                dry_48k_r: None,
                wet_48k_l: &dry_core,
                wet_48k_r: None,
                windows: win.clone(),
                controls_meta: meta.clone(),
                user_metadata: BTreeMap::new(),
            },
            None,
        )
        .unwrap();
        assert!(!inpc_a.is_empty());
        assert_eq!(spec_a.stem_id, "seal_a");

        // Leading coreset + frame + settle span
        let mut lead_dry = dry_core.clone();
        lead_dry.resize(CORESET_WITH_SETTLE_48K + ADAPTIVE_FRAME_SAMPLES, 0.0);
        let (inpc_lead, spec_lead, _, _) = align_full_stem_and_seal_vault_with_override(
            StemSealRequest {
                stem_id: "seal_lead",
                is_star_center: false,
                seed_256: &seed,
                carrier_bins: &[6, 12],
                valid_frames: 3,
                dry_48k_l: &lead_dry,
                dry_48k_r: Some(&dry_r),
                wet_48k_l: &lead_dry,
                wet_48k_r: Some(&lead_dry),
                windows: win,
                controls_meta: meta,
                user_metadata: BTreeMap::new(),
            },
            None,
        )
        .unwrap();
        assert!(!inpc_lead.is_empty());
        assert_eq!(spec_lead.stem_id, "seal_lead");
    }
}
