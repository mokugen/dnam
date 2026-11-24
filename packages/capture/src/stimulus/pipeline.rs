#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Mode 1 Closed-Loop Streaming Distillation (`train_adaptive_target`), Signal-Measured
//! Endpoint Probing (`REQ-1.6`), and Freeze-Transition Capture (`stimulus::pipeline`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md`
//!   (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.planner.schedule]`,
//!   `ref[train.capture.planner.freeze-transition]`, `ref[train.capture.invariants]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

mod causal_probe;
mod fullband;
pub mod schoukens;
mod stems;
mod sweeps;

pub use causal_probe::{
    build_v3_causal_probe_excitation, estimate_v3_causal_deadtime_and_polarity,
};

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::DnamError;

use crate::proj::bundle::{
    ProjChunkInput, build_dnam_proj_container_with_dnam_bin, export_runtime_dnam,
};
use crate::proj::capsules::CalbChunk64B;
use crate::proj::directory::ProjChunkKind;
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::{
    initialize_backbone_dnam_bin_if_needed, solve_captures_batch_with_progress,
};
use crate::solver::{SolverOptions, TrainerProgressHandle, TrainingPhase, TrainingStatus};
use crate::targets::DistillationTarget;

use super::StimulusConfig;
use super::batch::{build_controls_meta_map, default_control_vector};
use super::fft::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, PEDESTAL_BIN_MONO, build_disjoint_comb_schedule,
    synthesize_cyclic_comb_frame,
};
use super::laws::TopologyProbeMode;
pub use super::null_test::{LiveNullTestReport, run_live_null_test_and_autopatch};
use super::safeguards::CaptureWarning;
use super::sealing::SharedStemAlignment;
use super::waveforms::deterministic_seed_256;
use crate::CORESET_FP32_CANONICAL_25S_FRAMES;
use stems::{render_endpoint_probe_48k, run_single_closed_loop_stem};
use sweeps::{
    SweepCaptureContext, capture_hadamard_diagonal_stems, capture_range_and_switch_stems,
};

/// Options controlling Mode-1 closed-loop adaptive capture and distillation (`train_adaptive_target`).
#[derive(Debug, Clone)]
pub struct AdaptiveTrainingOptions {
    /// Sample rate in Hz (`48_000` canonical; `8_000..=384_000` supported).
    pub sample_rate_hz: u32,
    /// Multi-port / multi-channel probe topology.
    pub topology_mode: TopologyProbeMode,
    /// `--fixed-duration`: disables 4-Gate Cramér-Rao early stopping when `true`.
    pub fixed_duration: bool,
    /// `"Prefer Stationary Stops (Mode B)"` override toggle (`r[train.capture.control-sweeps]`).
    pub prefer_mode_b: bool,
    /// Maximum `96 ms` frames per stem (`default = 16`).
    pub max_frames_per_stem: usize,
    /// Automatically execute the `1.0s` Live Null-Test & `< 5 ms` Auto-Patch at capture end.
    pub run_live_null_test: bool,
    /// Null-test ESR threshold in dB above which the `< 5 ms` sufficient-statistic auto-patch runs.
    pub null_test_autopatch_threshold_db: f32,
    /// Optional Phase-0 `CALB` loopback calibration chunk to embed in `.dnam-proj`.
    pub calb: Option<CalbChunk64B>,
    /// Capture full Tier-1 `9.0 s` (`432,000`-frame) `SPCM` on `StarCenter` ahead of compaction (`default = false`).
    pub use_spcm_tier1: bool,
    /// Condensed coreset frame length (`96_000..=120_000`, `% 16 == 0`; `default = 120_000` / `2.5 s`).
    pub coreset_frames_48k: usize,
}

impl Default for AdaptiveTrainingOptions {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            topology_mode: TopologyProbeMode::Mono1To1,
            fixed_duration: false,
            prefer_mode_b: false,
            max_frames_per_stem: 16,
            run_live_null_test: true,
            null_test_autopatch_threshold_db: -36.0,
            calb: None,
            use_spcm_tier1: false,
            coreset_frames_48k: CORESET_FP32_CANONICAL_25S_FRAMES,
        }
    }
}

/// Complete output of [`train_adaptive_target`].
#[derive(Debug, Clone)]
pub struct AdaptiveTrainingReport {
    /// Validated single-file `.dnam-proj` (`DNPJ`) container bytes.
    pub proj_bytes: Vec<u8>,
    /// Exported runtime `.dnam` (`v2.0`) container bytes.
    pub runtime_dnam_bytes: Vec<u8>,
    /// Accumulated non-blocking hardware safeguard warnings across all stems.
    pub warnings: Vec<CaptureWarning>,
    /// Optional `1.0s` Live Null-Test & `< 5 ms` Auto-Patch report.
    pub null_test: Option<LiveNullTestReport>,
    /// Total stems sealed into the `.dnam-proj` vault.
    pub stem_count: usize,
    /// Number of Hadamard cross-term diagonal sweeps executed (`<= 2` orthogonal diagonals).
    pub hadamard_diagonals_run: usize,
    /// Number of decoupled Hadamard diagonals skipped via `Law 4D` commutator pruning.
    pub hadamard_diagonals_pruned: usize,
}

/// Mode 1 (Streaming Closed-Loop): Drives `target` frame-by-frame via [`AdaptiveStimulusController`],
/// sealing `INPC`, `STMC`, `STAT`, and `WGHT` into a single-file `.dnam-proj` container and
/// optionally executing [`run_live_null_test_and_autopatch`].
pub fn train_adaptive_target(
    manifest: DnamProjManifest,
    target: &mut dyn DistillationTarget,
    options: &AdaptiveTrainingOptions,
) -> Result<AdaptiveTrainingReport, DnamError> {
    train_adaptive_target_with_progress(
        manifest,
        target,
        options,
        &TrainerProgressHandle::default(),
    )
}

/// Variant of [`train_adaptive_target`] emitting coarse capture & solver milestones into
/// [`TrainerProgressHandle`].
pub fn train_adaptive_target_with_progress(
    mut manifest: DnamProjManifest,
    target: &mut dyn DistillationTarget,
    options: &AdaptiveTrainingOptions,
    progress: &TrainerProgressHandle,
) -> Result<AdaptiveTrainingReport, DnamError> {
    if !(8_000..=384_000).contains(&options.sample_rate_hz) {
        return Err(DnamError::InvalidSampleRate(options.sample_rate_hz));
    }
    manifest.stems_registry.clear();
    let mut init_chunks = Vec::new();
    if let Some(calb) = options.calb {
        init_chunks.push(ProjChunkInput {
            kind: ProjChunkKind::Calb,
            chunk_id: 0,
            payload: calb.to_bytes().to_vec(),
        });
    }
    let seeded_dbin = initialize_backbone_dnam_bin_if_needed(None)?;
    let mut proj_bytes =
        build_dnam_proj_container_with_dnam_bin(&mut manifest, &init_chunks, &seeded_dbin)?;

    let seed_256 = deterministic_seed_256(0xACE5_2026_0001);
    let noise_floor_dbfs = options.calb.map_or(-100.0, |c| c.noise_floor_dbfs);
    let base_ctrls = default_control_vector(&manifest);
    let empty_switches = BTreeMap::new();
    let mut all_warnings = Vec::new();

    let empty_mask = BTreeSet::new();
    let probe_sched =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let probe_amps: Vec<(usize, f64)> =
        probe_sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let probe_comb = synthesize_cyclic_comb_frame(
        &probe_amps,
        Some(PEDESTAL_BIN_MONO),
        0.05,
        0.16,
        ADC_PEAK_GOVERNOR_FS,
        &[0.0; 64],
    );
    let mut probe_dry_l_48k = probe_comb.samples;
    probe_dry_l_48k.resize(10 * ADAPTIVE_FRAME_SAMPLES, 0.0);

    let (star_probe_wet_l, star_probe_wet_r) = render_endpoint_probe_48k(
        target,
        &probe_dry_l_48k,
        options.sample_rate_hz,
        options.topology_mode,
        &base_ctrls,
    )?;

    // 1. Capture StarCenter (s = 0)
    progress.emit_phase(TrainingStatus::Running, TrainingPhase::Capture, 20);
    let use_fullband = options.max_frames_per_stem >= 6;
    let star_meta = build_controls_meta_map(&manifest, &base_ctrls, &empty_switches);
    let (
        full_dry_pair,
        full_dry_aux,
        inpc_bytes,
        star_spec,
        star_tilt,
        shared_align_stereo,
        shared_align,
        rep_profiles,
    ) = if use_fullband {
        let star_cap = fullband::capture_fullband_star_center_schoukens(
            target,
            fullband::StarCenterCaptureRequest {
                sample_rate_hz: options.sample_rate_hz,
                topology_mode: options.topology_mode,
                seed_256: &seed_256,
                carrier_bins: &probe_sched.carrier_bins,
                valid_frames: options.max_frames_per_stem.max(6),
                base_ctrls: &base_ctrls,
                star_meta,
                probe_comb_dry_l: &probe_dry_l_48k,
                probe_comb_wet_l: &star_probe_wet_l,
                probe_comb_wet_r: star_probe_wet_r.as_deref(),
                use_spcm_tier1: options.use_spcm_tier1,
                coreset_frames_48k: options.coreset_frames_48k,
                calb_noise_floor_dbfs: noise_floor_dbfs,
            },
        )?;
        let sa = star_cap.shared_align_stereo;
        (
            Some((star_cap.dry_l_48k, star_cap.dry_r_48k)),
            star_cap.aux_in_3_48k,
            star_cap.inpc_bytes,
            star_cap.star_spec,
            star_cap.star_tilt,
            sa,
            Some((sa.lag_samples, sa.invert_l)),
            star_cap.rep_profiles,
        )
    } else {
        let star_cfg = StimulusConfig {
            stem_id: "star_center".to_string(),
            sample_rate_hz: options.sample_rate_hz,
            topology_mode: options.topology_mode,
            control_values: base_ctrls.clone(),
            sweep_end_control_values: None,
            active_control_id: 0,
            is_master_volume_sweep: false,
            fixed_duration: options.fixed_duration,
            max_frames: options.max_frames_per_stem.max(6),
            max_tail_frames: 3,
            noise_floor_dbfs,
            seed_256,
        };
        let (bytes, spec, warns, tilt, star_lag, star_pol) = run_single_closed_loop_stem(
            target,
            star_cfg,
            true,
            star_meta,
            None,
            Some((
                &probe_dry_l_48k,
                &star_probe_wet_l,
                star_probe_wet_r.as_deref(),
                &probe_sched.carrier_bins,
            )),
        )?;
        all_warnings.extend(warns);
        (
            None,
            None,
            bytes,
            spec,
            tilt,
            SharedStemAlignment {
                lag_samples: star_lag,
                clock_drift_ppm: 0.0,
                invert_l: star_pol,
                invert_r: star_pol,
            },
            Some((star_lag, star_pol)),
            None,
        )
    };
    let mut captured_specs = vec![star_spec];

    // 2. Capture 1D Range sweeps/stops & Switch options + Law 4D Hadamard diagonals via `sweeps`.
    let sweep_ctx = SweepCaptureContext {
        manifest: &manifest,
        options,
        base_ctrls: &base_ctrls,
        seed_256: &seed_256,
        probe_sched: &probe_sched,
        probe_dry_l_48k: &probe_dry_l_48k,
        star_probe_wet_l: &star_probe_wet_l,
        full_dry_pair: &full_dry_pair,
        full_dry_aux: &full_dry_aux,
        shared_align_stereo,
        shared_align,
        star_tilt,
        noise_floor_dbfs,
    };
    let sweep_out = capture_range_and_switch_stems(target, &sweep_ctx)?;
    all_warnings.extend(sweep_out.warnings);
    captured_specs.extend(sweep_out.specs);

    let (had_specs, had_warns, hadamard_run, pruned_count) = capture_hadamard_diagonal_stems(
        target,
        &sweep_ctx,
        &sweep_out.active_range_indices,
        &sweep_out.shape_spread_by_ctrl,
    )?;
    all_warnings.extend(had_warns);
    captured_specs.extend(had_specs);

    let stem_count = captured_specs.len();
    progress.emit_phase(TrainingStatus::Running, TrainingPhase::Capture, 380);
    proj_bytes = solve_captures_batch_with_progress(
        &proj_bytes,
        captured_specs,
        Some(&inpc_bytes),
        SolverOptions {
            rep_profiles,
            ..SolverOptions::default()
        },
        &progress.subrange(400, if options.run_live_null_test { 920 } else { 980 }),
    )?;

    let mut runtime_dnam_bytes = export_runtime_dnam(&proj_bytes, None)?;
    let null_test = if options.run_live_null_test {
        progress.emit_phase(TrainingStatus::Running, TrainingPhase::LiveNullTest, 940);
        let (updated_proj, updated_dnam, report) = run_live_null_test_and_autopatch(
            &proj_bytes,
            target,
            options.null_test_autopatch_threshold_db,
        )?;
        proj_bytes = updated_proj;
        runtime_dnam_bytes = updated_dnam;
        Some(report)
    } else {
        None
    };
    progress.emit_phase(TrainingStatus::Completed, TrainingPhase::Completed, 1_000);

    Ok(AdaptiveTrainingReport {
        proj_bytes,
        runtime_dnam_bytes,
        warnings: all_warnings,
        null_test,
        stem_count,
        hadamard_diagonals_run: hadamard_run,
        hadamard_diagonals_pruned: pruned_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::{StemResponse, StemStimulus};
    use std::collections::BTreeMap;

    struct DummyTarget;
    impl DistillationTarget for DummyTarget {
        fn reset_state(&mut self) {}
        fn render_stem(&mut self, s: &StemStimulus) -> Result<StemResponse, DnamError> {
            Ok(StemResponse {
                stem_id: s.stem_id.clone(),
                sample_rate_hz: s.sample_rate_hz,
                out_1: s.in_1.clone(),
                out_2: None,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn test_train_adaptive_target_rejects_invalid_sample_rate() {
        let mut target = DummyTarget;
        let opts = AdaptiveTrainingOptions {
            sample_rate_hz: 4_000,
            ..Default::default()
        };
        assert!(matches!(
            train_adaptive_target(DnamProjManifest::default_reference(), &mut target, &opts),
            Err(DnamError::InvalidSampleRate(4_000))
        ));
    }
}
