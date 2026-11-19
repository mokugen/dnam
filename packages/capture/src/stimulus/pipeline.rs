#![forbid(unsafe_code)]
//! Mode 1 Closed-Loop Streaming Distillation (`train_adaptive_target`), Signal-Measured
//! Endpoint Probing (`REQ-1.6`), and Freeze-Transition Capture (`stimulus::pipeline`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md`
//!   (`ref[train.capture.excitation.codesigned]`, `ref[train.capture.planner.schedule]`,
//!   `ref[train.capture.planner.freeze-transition]`, `ref[train.capture.invariants]`)
//! - `docs/specs/training/iterative_resume_and_caching.md` (`ref[train.cache.proj-vault]`)

mod fullband;
mod stems;

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::{ControlCalibrationRole, ControlKind, DnamError};

use crate::proj::bundle::{
    ProjChunkInput, build_dnam_proj_container_with_cp1_and_compacted, export_runtime_dnam,
};
use crate::proj::capsules::CalbChunk64B;
use crate::proj::directory::{ProjChunkKind, StmcTrajectoryWindow};
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::{initialize_backbone_arena_if_needed, solve_captures_batch_with_progress};
use crate::solver::{SolverOptions, TrainerProgressHandle, TrainingPhase, TrainingStatus};
use crate::targets::{DistillationTarget, StemResponse};

use super::batch::{build_controls_meta_map, control_declared_bounds, default_control_vector};
use super::fft::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, CYCLIC_WINDOW_SAMPLES, PEDESTAL_BIN_MONO,
    build_disjoint_comb_schedule, synthesize_cyclic_comb_frame,
};
use super::laws::{
    HadamardLaw4dTracker, KnobAcousticClass, SweepExecutionMode, TopologyProbeMode,
    build_orthogonal_hadamard_diagonals, classify_knob_from_endpoint_probe,
    compute_hadamard_sfm_ratio, select_sweep_execution_mode,
};
pub use super::null_test::{LiveNullTestReport, run_live_null_test_and_autopatch};
use super::safeguards::CaptureWarning;
use super::waveforms::deterministic_seed_256;
use super::{AdaptiveStimulusController, StimulusConfig};
use stems::{
    capture_freeze_transition_stem_mode1, render_endpoint_probe_48k, run_single_closed_loop_stem,
    spectral_tilt_from_stmc_4096,
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
    let seeded_arena = initialize_backbone_arena_if_needed(None);
    let mut proj_bytes = build_dnam_proj_container_with_cp1_and_compacted(
        &mut manifest,
        &init_chunks,
        &seeded_arena,
        &[],
        false,
    )?;

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
    probe_dry_l_48k.resize(4 * ADAPTIVE_FRAME_SAMPLES, 0.0);

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
    let need_stereo_dry = matches!(
        options.topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let aux_key_mode = matches!(options.topology_mode, TopologyProbeMode::DualInToMono2To1);
    let fullband_probe = if use_fullband {
        let causal_probe = fullband::build_v3_causal_probe_excitation_48k();
        let (pw_l, pw_r) = render_endpoint_probe_48k(
            target,
            &causal_probe,
            options.sample_rate_hz,
            options.topology_mode,
            &base_ctrls,
        )?;
        let (lag_l, inv_l) =
            fullband::estimate_v3_causal_deadtime_and_polarity(&causal_probe, &pw_l);
        let (causal_lag, inv_r) = pw_r.as_deref().map_or((lag_l, inv_l), |r| {
            let (lag_r, pol_r) =
                fullband::estimate_v3_causal_deadtime_and_polarity(&causal_probe, r);
            (lag_l.min(lag_r), pol_r)
        });
        let ti_probe = fullband::build_time_invariance_probe_48k();
        let (ti_wet_l, ti_wet_r) = render_endpoint_probe_48k(
            target,
            &ti_probe,
            options.sample_rate_hz,
            options.topology_mode,
            &base_ctrls,
        )?;
        let contiguous_96k =
            fullband::probe_requires_contiguous_96k_fullband(&ti_wet_l, ti_wet_r.as_deref());
        let rep_profiles = if contiguous_96k {
            Some(fullband::probe_target_repeatability_profiles(
                target,
                options.sample_rate_hz,
                options.topology_mode,
                &base_ctrls,
                (causal_lag, inv_l, inv_r),
            )?)
        } else {
            None
        };
        let dry_pair = fullband::build_fullband_identification_coreset_48k_opts(
            &seed_256,
            need_stereo_dry,
            aux_key_mode,
            options.calb.is_some(),
            contiguous_96k,
        );
        Some((
            (causal_lag, inv_l, inv_r),
            dry_pair,
            contiguous_96k,
            rep_profiles,
        ))
    } else {
        None
    };

    let star_meta = build_controls_meta_map(&manifest, &base_ctrls, &empty_switches);
    let (full_dry_pair, align_stereo_opt, is_contiguous_96k, rep_profiles) = match fullband_probe {
        Some((align_stereo, dry_pair, c96, rep_p)) => {
            (Some(dry_pair), Some(align_stereo), c96, rep_p)
        }
        None => (None, None, false, None),
    };
    let (inpc_bytes, star_spec, star_tilt, shared_align_stereo, shared_align) =
        if let (Some((fb_l, fb_r)), Some(align_stereo)) = (&full_dry_pair, align_stereo_opt) {
            let (causal_lag, inv_l, _) = align_stereo;
            let star_windows = vec![StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: base_ctrls.first().copied().unwrap_or(0.0),
                window_weight: 1.0,
                emr_velocity: 0.0,
            }];
            let (fb_inpc, fb_spec) = fullband::render_and_seal_fullband_stem(
                target,
                fullband::FullbandStemRequest {
                    stem_id: "star_center",
                    sample_rate_hz: options.sample_rate_hz,
                    topology_mode: options.topology_mode,
                    seed_256: &seed_256,
                    carrier_bins: &probe_sched.carrier_bins,
                    valid_frames: options.max_frames_per_stem.max(6),
                    full_dry_l_48k: fb_l,
                    full_dry_r_48k: fb_r.as_deref(),
                    ctrl_start: &base_ctrls,
                    ctrl_end: None,
                    is_star_center: true,
                    windows: star_windows,
                    controls_meta: star_meta,
                    user_metadata: BTreeMap::new(),
                    shared_align: align_stereo,
                    compact_render_frames_48k: None,
                },
            )?;
            let tilt = spectral_tilt_from_stmc_4096(&fb_spec.spectral_summary_4096);
            (
                fb_inpc,
                fb_spec,
                tilt,
                align_stereo,
                Some((causal_lag, inv_l)),
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
                bytes,
                spec,
                tilt,
                (star_lag, star_pol, star_pol),
                Some((star_lag, star_pol)),
            )
        };
    let mut captured_specs = vec![star_spec];

    // 2. Capture FreezeLatch transitions, 1D Range sweeps/stops & Switch options
    let mut active_range_indices = Vec::new();
    let mut shape_spread_by_ctrl = vec![0.0_f64; manifest.base.controls_schema.len()];
    let controls_snapshot = manifest.base.controls_schema.clone();

    let compact_many_knobs =
        (is_contiguous_96k || manifest.base.controls_schema.len() >= 4).then_some(8_192);

    for (p_idx, ctrl) in controls_snapshot.iter().enumerate() {
        if ctrl
            .calibration
            .as_ref()
            .is_some_and(|c| c.role == ControlCalibrationRole::FreezeLatch)
        {
            let freeze_spec = capture_freeze_transition_stem_mode1(
                &manifest,
                target,
                p_idx,
                options,
                &seed_256,
                &probe_sched.carrier_bins,
                shared_align,
            )?;
            captured_specs.push(freeze_spec);
        }

        match ctrl.kind {
            ControlKind::Range => {
                active_range_indices.push(p_idx);
                let (lo, hi) = control_declared_bounds(ctrl);
                let probe_u = if (lo - base_ctrls[p_idx]).abs() >= (hi - base_ctrls[p_idx]).abs() {
                    lo
                } else {
                    hi
                };
                let mut probe_v = base_ctrls.clone();
                probe_v[p_idx] = probe_u;
                let (end_probe_wet, _) = render_endpoint_probe_48k(
                    target,
                    &probe_dry_l_48k,
                    options.sample_rate_hz,
                    options.topology_mode,
                    &probe_v,
                )?;
                let mut ep = classify_knob_from_endpoint_probe(
                    &star_probe_wet_l,
                    &end_probe_wet,
                    ADAPTIVE_FRAME_SAMPLES,
                    &probe_sched,
                );
                if ctrl.calibration.as_ref().is_some_and(|c| {
                    matches!(
                        c.role,
                        ControlCalibrationRole::DelayTimeMs
                            | ControlCalibrationRole::GrainSizeMs
                            | ControlCalibrationRole::LfoRateHz
                    )
                }) {
                    ep.knob_class = KnobAcousticClass::ClassIIIPitchDelayTime;
                    ep.delta_t_mem_ms = ep.delta_t_mem_ms.max(50.0);
                }
                let is_master = ep.is_master_volume_stage;
                let probe_spread = ep.spectral_shape_spread;
                let mode = select_sweep_execution_mode(
                    ep.delta_t_mem_ms,
                    ep.knob_class,
                    options.prefer_mode_b,
                );

                if mode == SweepExecutionMode::ModeAContinuousSweep {
                    let mut start_v = base_ctrls.clone();
                    let mut end_v = base_ctrls.clone();
                    start_v[p_idx] = lo;
                    end_v[p_idx] = hi;
                    let meta = build_controls_meta_map(&manifest, &end_v, &empty_switches);
                    let sweep_windows = vec![
                        StmcTrajectoryWindow {
                            control_id: p_idx as u16,
                            ctrl_kind: 0,
                            active_opt: 0,
                            u_phys: lo,
                            window_weight: 1.0,
                            emr_velocity: 0.1,
                        },
                        StmcTrajectoryWindow {
                            control_id: p_idx as u16,
                            ctrl_kind: 0,
                            active_opt: 0,
                            u_phys: hi,
                            window_weight: 1.0,
                            emr_velocity: 0.1,
                        },
                    ];
                    let (spec, tilt) = if let Some((ref fb_l, ref fb_r)) = full_dry_pair {
                        let (_, fb_spec) = fullband::render_and_seal_fullband_stem(
                            target,
                            fullband::FullbandStemRequest {
                                stem_id: &format!("sweep_ctrl_{p_idx}"),
                                sample_rate_hz: options.sample_rate_hz,
                                topology_mode: options.topology_mode,
                                seed_256: &seed_256,
                                carrier_bins: &probe_sched.carrier_bins,
                                valid_frames: options.max_frames_per_stem.max(6),
                                full_dry_l_48k: fb_l,
                                full_dry_r_48k: fb_r.as_deref(),
                                ctrl_start: &start_v,
                                ctrl_end: Some(&end_v),
                                is_star_center: false,
                                windows: sweep_windows,
                                controls_meta: meta,
                                user_metadata: BTreeMap::new(),
                                shared_align: shared_align_stereo,
                                compact_render_frames_48k: compact_many_knobs,
                            },
                        )?;
                        let fb_tilt = spectral_tilt_from_stmc_4096(&fb_spec.spectral_summary_4096);
                        (fb_spec, fb_tilt)
                    } else {
                        let cfg = StimulusConfig {
                            stem_id: format!("sweep_ctrl_{p_idx}"),
                            sample_rate_hz: options.sample_rate_hz,
                            topology_mode: options.topology_mode,
                            control_values: start_v,
                            sweep_end_control_values: Some(end_v),
                            active_control_id: p_idx as u16,
                            is_master_volume_sweep: is_master,
                            fixed_duration: options.fixed_duration,
                            max_frames: options.max_frames_per_stem.max(6),
                            max_tail_frames: 2,
                            noise_floor_dbfs,
                            seed_256,
                        };
                        let (_, mut cl_spec, warns, cl_tilt, _, _) = run_single_closed_loop_stem(
                            target,
                            cfg,
                            false,
                            meta,
                            shared_align,
                            None,
                        )?;
                        if cl_spec.windows.len() < 2 {
                            cl_spec.windows = sweep_windows;
                        } else {
                            cl_spec.windows[0].u_phys = lo;
                            if let Some(last_w) = cl_spec.windows.last_mut() {
                                last_w.u_phys = hi;
                            }
                        }
                        all_warnings.extend(warns);
                        (cl_spec, cl_tilt)
                    };
                    shape_spread_by_ctrl[p_idx] = (tilt - star_tilt).abs().max(probe_spread);
                    captured_specs.push(spec);
                } else {
                    let mut tilts = Vec::new();
                    let span = (hi - lo).max(1e-6);
                    let mid_u = 0.5 * (lo + hi);
                    let stops = if (base_ctrls[p_idx] - lo).abs() < 0.08 * span {
                        [mid_u, hi]
                    } else if (base_ctrls[p_idx] - hi).abs() < 0.08 * span {
                        [lo, mid_u]
                    } else {
                        [lo, hi]
                    };
                    for (s_idx, &stop_u) in stops.iter().enumerate() {
                        let mut v = base_ctrls.clone();
                        v[p_idx] = stop_u;
                        let meta = build_controls_meta_map(&manifest, &v, &empty_switches);
                        let (spec, tilt) = if let Some((ref fb_l, ref fb_r)) = full_dry_pair {
                            let stop_windows = vec![StmcTrajectoryWindow {
                                control_id: p_idx as u16,
                                ctrl_kind: 0,
                                active_opt: 0,
                                u_phys: stop_u,
                                window_weight: 1.0,
                                emr_velocity: 0.0,
                            }];
                            let (_, fb_spec) = fullband::render_and_seal_fullband_stem(
                                target,
                                fullband::FullbandStemRequest {
                                    stem_id: &format!("stop_ctrl_{p_idx}_{s_idx}"),
                                    sample_rate_hz: options.sample_rate_hz,
                                    topology_mode: options.topology_mode,
                                    seed_256: &seed_256,
                                    carrier_bins: &probe_sched.carrier_bins,
                                    valid_frames: 8,
                                    full_dry_l_48k: fb_l,
                                    full_dry_r_48k: fb_r.as_deref(),
                                    ctrl_start: &v,
                                    ctrl_end: None,
                                    is_star_center: false,
                                    windows: stop_windows,
                                    controls_meta: meta,
                                    user_metadata: BTreeMap::new(),
                                    shared_align: shared_align_stereo,
                                    compact_render_frames_48k: Some(8_192),
                                },
                            )?;
                            let fb_tilt =
                                spectral_tilt_from_stmc_4096(&fb_spec.spectral_summary_4096);
                            (fb_spec, fb_tilt)
                        } else {
                            let cfg = StimulusConfig {
                                stem_id: format!("stop_ctrl_{p_idx}_{s_idx}"),
                                sample_rate_hz: options.sample_rate_hz,
                                topology_mode: options.topology_mode,
                                control_values: v,
                                sweep_end_control_values: None,
                                active_control_id: p_idx as u16,
                                is_master_volume_sweep: is_master,
                                fixed_duration: options.fixed_duration,
                                max_frames: 8,
                                max_tail_frames: 2,
                                noise_floor_dbfs,
                                seed_256,
                            };
                            let (_, cl_spec, warns, cl_tilt, _, _) = run_single_closed_loop_stem(
                                target,
                                cfg,
                                false,
                                meta,
                                shared_align,
                                None,
                            )?;
                            all_warnings.extend(warns);
                            (cl_spec, cl_tilt)
                        };
                        tilts.push(tilt);
                        captured_specs.push(spec);
                    }
                    shape_spread_by_ctrl[p_idx] =
                        (tilts[tilts.len() - 1] - tilts[0]).abs().max(probe_spread);
                }
            }
            ControlKind::Switch => {
                let opts = ctrl.declared_options.clone().unwrap_or_default();
                let def_opt =
                    (base_ctrls[p_idx].round() as usize).min(opts.len().saturating_sub(1));
                for (opt_idx, opt_name) in opts.iter().enumerate() {
                    if opt_idx == def_opt {
                        continue;
                    }
                    let mut sw_map = BTreeMap::new();
                    sw_map.insert(p_idx, opt_name.clone());
                    let mut sw_ctrls = base_ctrls.clone();
                    sw_ctrls[p_idx] = opt_idx as f32;
                    let meta = build_controls_meta_map(&manifest, &sw_ctrls, &sw_map);
                    let sw_windows = vec![StmcTrajectoryWindow {
                        control_id: p_idx as u16,
                        ctrl_kind: 1,
                        active_opt: opt_idx as u8,
                        u_phys: opt_idx as f32,
                        window_weight: 1.0,
                        emr_velocity: 0.0,
                    }];
                    let spec = if let Some((ref fb_l, ref fb_r)) = full_dry_pair {
                        let (_, fb_spec) = fullband::render_and_seal_fullband_stem(
                            target,
                            fullband::FullbandStemRequest {
                                stem_id: &format!("switch_{p_idx}_opt_{opt_idx}"),
                                sample_rate_hz: options.sample_rate_hz,
                                topology_mode: options.topology_mode,
                                seed_256: &seed_256,
                                carrier_bins: &probe_sched.carrier_bins,
                                valid_frames: 6,
                                full_dry_l_48k: fb_l,
                                full_dry_r_48k: fb_r.as_deref(),
                                ctrl_start: &sw_ctrls,
                                ctrl_end: None,
                                is_star_center: false,
                                windows: sw_windows,
                                controls_meta: meta,
                                user_metadata: BTreeMap::new(),
                                shared_align: shared_align_stereo,
                                compact_render_frames_48k: Some(8_192),
                            },
                        )?;
                        fb_spec
                    } else {
                        let cfg = StimulusConfig {
                            stem_id: format!("switch_{p_idx}_opt_{opt_idx}"),
                            sample_rate_hz: options.sample_rate_hz,
                            topology_mode: options.topology_mode,
                            control_values: sw_ctrls,
                            sweep_end_control_values: None,
                            active_control_id: p_idx as u16,
                            is_master_volume_sweep: false,
                            fixed_duration: options.fixed_duration,
                            max_frames: 6,
                            max_tail_frames: 2,
                            noise_floor_dbfs,
                            seed_256,
                        };
                        let (_, mut cl_spec, warns, _, _, _) = run_single_closed_loop_stem(
                            target,
                            cfg,
                            false,
                            meta,
                            shared_align,
                            None,
                        )?;
                        cl_spec.windows = sw_windows;
                        all_warnings.extend(warns);
                        cl_spec
                    };
                    captured_specs.push(spec);
                }
            }
        }
    }

    // 3. Law 4D Commutator-Pruned Orthogonal Hadamard Diagonals (`<= 2` diagonals, ADR-0016 §3.1)
    let total_possible_pairs = if active_range_indices.len() >= 2 {
        (active_range_indices.len() * (active_range_indices.len() - 1)) / 2
    } else {
        0
    };
    let orth_diags =
        build_orthogonal_hadamard_diagonals(&active_range_indices, &shape_spread_by_ctrl);
    let pruned_count = total_possible_pairs.saturating_sub(orth_diags.len());
    let mut hadamard_run = 0usize;

    for (diag_idx, signs) in orth_diags.iter().enumerate() {
        let mut v0 = base_ctrls.clone();
        let mut v1 = base_ctrls.clone();
        let first_p = signs.first().map_or(0, |&(p, _)| p);
        for &(p, sign) in signs {
            let (lo, hi) = control_declared_bounds(&controls_snapshot[p]);
            if sign >= 0.0 {
                v0[p] = lo;
                v1[p] = hi;
            } else {
                v0[p] = hi;
                v1[p] = lo;
            }
        }

        let meta = build_controls_meta_map(&manifest, &v1, &empty_switches);
        let had_windows: Vec<StmcTrajectoryWindow> = signs
            .iter()
            .flat_map(|&(p, _)| {
                [
                    StmcTrajectoryWindow {
                        control_id: p as u16,
                        ctrl_kind: 0,
                        active_opt: 0,
                        u_phys: v0[p],
                        window_weight: 1.0,
                        emr_velocity: 0.1,
                    },
                    StmcTrajectoryWindow {
                        control_id: p as u16,
                        ctrl_kind: 0,
                        active_opt: 0,
                        u_phys: v1[p],
                        window_weight: 1.0,
                        emr_velocity: 0.1,
                    },
                ]
            })
            .collect();
        let spec = if options.fixed_duration
            && let Some((ref fb_l, ref fb_r)) = full_dry_pair
        {
            let (_, fb_spec) = fullband::render_and_seal_fullband_stem(
                target,
                fullband::FullbandStemRequest {
                    stem_id: &format!("hadamard_diag_{diag_idx}"),
                    sample_rate_hz: options.sample_rate_hz,
                    topology_mode: options.topology_mode,
                    seed_256: &seed_256,
                    carrier_bins: &probe_sched.carrier_bins,
                    valid_frames: options.max_frames_per_stem.max(12),
                    full_dry_l_48k: fb_l,
                    full_dry_r_48k: fb_r.as_deref(),
                    ctrl_start: &v0,
                    ctrl_end: Some(&v1),
                    is_star_center: false,
                    windows: had_windows,
                    controls_meta: meta,
                    user_metadata: BTreeMap::new(),
                    shared_align: shared_align_stereo,
                    compact_render_frames_48k: compact_many_knobs,
                },
            )?;
            fb_spec
        } else {
            target.reset_state();
            let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
                stem_id: format!("hadamard_diag_{diag_idx}"),
                sample_rate_hz: options.sample_rate_hz,
                topology_mode: options.topology_mode,
                control_values: v0.clone(),
                sweep_end_control_values: Some(v1.clone()),
                active_control_id: first_p as u16,
                is_master_volume_sweep: false,
                fixed_duration: options.fixed_duration,
                max_frames: options.max_frames_per_stem.max(12),
                max_tail_frames: 1,
                noise_floor_dbfs,
                seed_256,
            });
            let mut law4d = HadamardLaw4dTracker::default();
            let mut sfm_hist: [Option<Vec<f64>>; 2] = [None, None];
            let mut last_resp: Option<StemResponse> = None;
            while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
                if ctrl.is_entering_coreset() {
                    target.reset_state();
                }
                let r = target.render_frame(&frame)?;
                if ctrl.is_in_active_learning() && !options.fixed_duration {
                    let (wl_48k, _) = ctrl.normalize_wet_to_48k(&r);
                    let wet_win = &wl_48k[wl_48k.len().saturating_sub(CYCLIC_WINDOW_SAMPLES)..];
                    let sfm = compute_hadamard_sfm_ratio(wet_win, ctrl.schedule_l(), &mut sfm_hist);
                    if law4d.observe_frame(sfm) {
                        ctrl.request_coreset_handoff();
                    }
                }
                last_resp = Some(r);
            }
            all_warnings.extend_from_slice(ctrl.warnings());
            let (_, mut cl_spec, _, _) =
                ctrl.seal_inpc_and_vault_spec_with_shared_align(false, meta.clone(), shared_align)?;
            cl_spec.windows.clone_from(&had_windows);
            if let Some((ref fb_l, ref fb_r)) = full_dry_pair {
                let (_, fb_spec) = fullband::render_and_seal_fullband_stem(
                    target,
                    fullband::FullbandStemRequest {
                        stem_id: &format!("hadamard_diag_{diag_idx}"),
                        sample_rate_hz: options.sample_rate_hz,
                        topology_mode: options.topology_mode,
                        seed_256: &seed_256,
                        carrier_bins: &probe_sched.carrier_bins,
                        valid_frames: options.max_frames_per_stem.max(12),
                        full_dry_l_48k: fb_l,
                        full_dry_r_48k: fb_r.as_deref(),
                        ctrl_start: &v0,
                        ctrl_end: Some(&v1),
                        is_star_center: false,
                        windows: had_windows,
                        controls_meta: meta,
                        user_metadata: cl_spec.user_metadata.clone(),
                        shared_align: shared_align_stereo,
                        compact_render_frames_48k: compact_many_knobs,
                    },
                )?;
                cl_spec = fb_spec;
            }
            cl_spec
        };
        captured_specs.push(spec);
        hadamard_run += 1;
    }

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
