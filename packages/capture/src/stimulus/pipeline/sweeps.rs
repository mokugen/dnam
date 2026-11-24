#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Range Sweep/Stop, Switch Option, and `Law 4D` Orthogonal Hadamard Diagonal Stem Capturers
//! (`stimulus::pipeline::sweeps`).

use std::collections::BTreeMap;

use dnam_engine::schema::{ControlKind, ControlSchemaDef, DnamError};

use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::VaultCaptureSpec;
use crate::targets::{DistillationTarget, StemResponse};

use super::AdaptiveTrainingOptions;
use super::fullband::{FullbandStemRequest, render_and_seal_fullband_stem};
use super::stems::{
    render_endpoint_probe_48k, run_single_closed_loop_stem, spectral_tilt_from_stmc_4096,
};
use crate::stimulus::batch::{build_controls_meta_map, control_declared_bounds};
use crate::stimulus::fft::{ADAPTIVE_FRAME_SAMPLES, CYCLIC_WINDOW_SAMPLES, DisjointCombSchedule};
use crate::stimulus::laws::{
    HadamardLaw4dTracker, SweepExecutionMode, build_orthogonal_hadamard_diagonals,
    classify_knob_from_endpoint_probe, compute_hadamard_sfm_ratio, select_sweep_execution_mode,
};
use crate::stimulus::safeguards::CaptureWarning;
use crate::stimulus::sealing::SharedStemAlignment;
use crate::stimulus::{AdaptiveStimulusController, StimulusConfig};

/// Shared immutable context passed across 1D Range/Switch and `Law 4D` Hadamard capture helpers.
pub(super) struct SweepCaptureContext<'a> {
    pub manifest: &'a DnamProjManifest,
    pub options: &'a AdaptiveTrainingOptions,
    pub base_ctrls: &'a [f32],
    pub seed_256: &'a [u8; 256],
    pub probe_sched: &'a DisjointCombSchedule,
    pub probe_dry_l_48k: &'a [f32],
    pub star_probe_wet_l: &'a [f32],
    pub full_dry_pair: &'a Option<(Vec<f32>, Option<Vec<f32>>)>,
    pub full_dry_aux: &'a Option<Vec<f32>>,
    pub shared_align_stereo: SharedStemAlignment,
    pub shared_align: Option<(f64, bool)>,
    pub star_tilt: f64,
    pub noise_floor_dbfs: f32,
}

/// Captured 1D sweep/stop/switch outputs returned to the streaming Mode-1 pipeline.
pub(super) struct SweepStageOutput {
    pub specs: Vec<VaultCaptureSpec>,
    pub warnings: Vec<CaptureWarning>,
    pub active_range_indices: Vec<usize>,
    pub shape_spread_by_ctrl: Vec<f64>,
}

fn capture_range_control(
    target: &mut dyn DistillationTarget,
    ctx: &SweepCaptureContext<'_>,
    p_idx: usize,
    ctrl: &ControlSchemaDef,
    out: &mut SweepStageOutput,
) -> Result<(), DnamError> {
    let empty_switches = BTreeMap::new();
    let (lo, hi) = control_declared_bounds(ctrl);
    let probe_u = if (lo - ctx.base_ctrls[p_idx]).abs() >= (hi - ctx.base_ctrls[p_idx]).abs() {
        lo
    } else {
        hi
    };
    let mut probe_v = ctx.base_ctrls.to_vec();
    probe_v[p_idx] = probe_u;
    let (end_probe_wet, _) = render_endpoint_probe_48k(
        target,
        ctx.probe_dry_l_48k,
        ctx.options.sample_rate_hz,
        ctx.options.topology_mode,
        &probe_v,
    )?;
    let ep = classify_knob_from_endpoint_probe(
        ctx.star_probe_wet_l,
        &end_probe_wet,
        ADAPTIVE_FRAME_SAMPLES,
        ctx.probe_sched,
    );
    let is_master = ep.is_master_volume_stage;
    let probe_spread = ep.spectral_shape_spread;
    let mode =
        select_sweep_execution_mode(ep.delta_t_mem_ms, ep.knob_class, ctx.options.prefer_mode_b);

    if mode == SweepExecutionMode::ModeAContinuousSweep {
        let mut start_v = ctx.base_ctrls.to_vec();
        let mut end_v = ctx.base_ctrls.to_vec();
        start_v[p_idx] = lo;
        end_v[p_idx] = hi;
        let meta = build_controls_meta_map(ctx.manifest, &end_v, &empty_switches);
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
        let (spec, tilt) = if let Some((fb_l, fb_r)) = ctx.full_dry_pair {
            let (_, fb_spec) = render_and_seal_fullband_stem(
                target,
                FullbandStemRequest {
                    stem_id: &format!("sweep_ctrl_{p_idx}"),
                    sample_rate_hz: ctx.options.sample_rate_hz,
                    topology_mode: ctx.options.topology_mode,
                    seed_256: ctx.seed_256,
                    carrier_bins: &ctx.probe_sched.carrier_bins,
                    valid_frames: ctx.options.max_frames_per_stem.max(6),
                    full_dry_l_48k: fb_l,
                    full_dry_r_48k: fb_r.as_deref(),
                    aux_in_3_48k: ctx.full_dry_aux.as_deref(),
                    ctrl_start: &start_v,
                    ctrl_end: Some(&end_v),
                    is_star_center: false,
                    windows: sweep_windows,
                    controls_meta: meta,
                    user_metadata: BTreeMap::new(),
                    shared_align: ctx.shared_align_stereo,
                    optional_spcm: None,
                    pre_extracted_physics: None,
                    pre_rendered_aligned_wet: None,
                },
            )?;
            let fb_tilt = spectral_tilt_from_stmc_4096(&fb_spec.spectral_summary_4096);
            (fb_spec, fb_tilt)
        } else {
            let cfg = StimulusConfig {
                stem_id: format!("sweep_ctrl_{p_idx}"),
                sample_rate_hz: ctx.options.sample_rate_hz,
                topology_mode: ctx.options.topology_mode,
                control_values: start_v,
                sweep_end_control_values: Some(end_v),
                active_control_id: p_idx as u16,
                is_master_volume_sweep: is_master,
                fixed_duration: ctx.options.fixed_duration,
                max_frames: ctx.options.max_frames_per_stem.max(6),
                max_tail_frames: 2,
                noise_floor_dbfs: ctx.noise_floor_dbfs,
                seed_256: *ctx.seed_256,
            };
            let (_, mut cl_spec, warns, cl_tilt, _, _) =
                run_single_closed_loop_stem(target, cfg, false, meta, ctx.shared_align, None)?;
            if cl_spec.windows.len() < 2 {
                cl_spec.windows = sweep_windows;
            } else {
                cl_spec.windows[0].u_phys = lo;
                if let Some(last_w) = cl_spec.windows.last_mut() {
                    last_w.u_phys = hi;
                }
            }
            out.warnings.extend(warns);
            (cl_spec, cl_tilt)
        };
        out.shape_spread_by_ctrl[p_idx] = (tilt - ctx.star_tilt).abs().max(probe_spread);
        out.specs.push(spec);
        return Ok(());
    }

    let mut tilts = Vec::new();
    let span = (hi - lo).max(1e-6);
    let mid_u = 0.5 * (lo + hi);
    let stops = if (ctx.base_ctrls[p_idx] - lo).abs() < 0.08 * span {
        [mid_u, hi]
    } else if (ctx.base_ctrls[p_idx] - hi).abs() < 0.08 * span {
        [lo, mid_u]
    } else {
        [lo, hi]
    };
    for (s_idx, &stop_u) in stops.iter().enumerate() {
        let mut v = ctx.base_ctrls.to_vec();
        v[p_idx] = stop_u;
        let meta = build_controls_meta_map(ctx.manifest, &v, &empty_switches);
        let (spec, tilt) = if let Some((fb_l, fb_r)) = ctx.full_dry_pair {
            let stop_windows = vec![StmcTrajectoryWindow {
                control_id: p_idx as u16,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: stop_u,
                window_weight: 1.0,
                emr_velocity: 0.0,
            }];
            let (_, fb_spec) = render_and_seal_fullband_stem(
                target,
                FullbandStemRequest {
                    stem_id: &format!("stop_ctrl_{p_idx}_{s_idx}"),
                    sample_rate_hz: ctx.options.sample_rate_hz,
                    topology_mode: ctx.options.topology_mode,
                    seed_256: ctx.seed_256,
                    carrier_bins: &ctx.probe_sched.carrier_bins,
                    valid_frames: 8,
                    full_dry_l_48k: fb_l,
                    full_dry_r_48k: fb_r.as_deref(),
                    aux_in_3_48k: ctx.full_dry_aux.as_deref(),
                    ctrl_start: &v,
                    ctrl_end: None,
                    is_star_center: false,
                    windows: stop_windows,
                    controls_meta: meta,
                    user_metadata: BTreeMap::new(),
                    shared_align: ctx.shared_align_stereo,
                    optional_spcm: None,
                    pre_extracted_physics: None,
                    pre_rendered_aligned_wet: None,
                },
            )?;
            let fb_tilt = spectral_tilt_from_stmc_4096(&fb_spec.spectral_summary_4096);
            (fb_spec, fb_tilt)
        } else {
            let cfg = StimulusConfig {
                stem_id: format!("stop_ctrl_{p_idx}_{s_idx}"),
                sample_rate_hz: ctx.options.sample_rate_hz,
                topology_mode: ctx.options.topology_mode,
                control_values: v,
                sweep_end_control_values: None,
                active_control_id: p_idx as u16,
                is_master_volume_sweep: is_master,
                fixed_duration: ctx.options.fixed_duration,
                max_frames: 8,
                max_tail_frames: 2,
                noise_floor_dbfs: ctx.noise_floor_dbfs,
                seed_256: *ctx.seed_256,
            };
            let (_, cl_spec, warns, cl_tilt, _, _) =
                run_single_closed_loop_stem(target, cfg, false, meta, ctx.shared_align, None)?;
            out.warnings.extend(warns);
            (cl_spec, cl_tilt)
        };
        tilts.push(tilt);
        out.specs.push(spec);
    }
    out.shape_spread_by_ctrl[p_idx] = (tilts[tilts.len() - 1] - tilts[0]).abs().max(probe_spread);
    Ok(())
}

fn capture_switch_control(
    target: &mut dyn DistillationTarget,
    ctx: &SweepCaptureContext<'_>,
    p_idx: usize,
    ctrl: &ControlSchemaDef,
    out: &mut SweepStageOutput,
) -> Result<(), DnamError> {
    let opts = ctrl.declared_options.clone().unwrap_or_default();
    let def_opt = (ctx.base_ctrls[p_idx].round() as usize).min(opts.len().saturating_sub(1));
    for (opt_idx, opt_name) in opts.iter().enumerate() {
        if opt_idx == def_opt {
            continue;
        }
        let mut sw_map = BTreeMap::new();
        sw_map.insert(p_idx, opt_name.clone());
        let mut sw_ctrls = ctx.base_ctrls.to_vec();
        sw_ctrls[p_idx] = opt_idx as f32;
        let meta = build_controls_meta_map(ctx.manifest, &sw_ctrls, &sw_map);
        let sw_windows = vec![StmcTrajectoryWindow {
            control_id: p_idx as u16,
            ctrl_kind: 1,
            active_opt: opt_idx as u8,
            u_phys: opt_idx as f32,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }];
        let spec = if let Some((fb_l, fb_r)) = ctx.full_dry_pair {
            let (_, fb_spec) = render_and_seal_fullband_stem(
                target,
                FullbandStemRequest {
                    stem_id: &format!("switch_{p_idx}_opt_{opt_idx}"),
                    sample_rate_hz: ctx.options.sample_rate_hz,
                    topology_mode: ctx.options.topology_mode,
                    seed_256: ctx.seed_256,
                    carrier_bins: &ctx.probe_sched.carrier_bins,
                    valid_frames: 6,
                    full_dry_l_48k: fb_l,
                    full_dry_r_48k: fb_r.as_deref(),
                    aux_in_3_48k: ctx.full_dry_aux.as_deref(),
                    ctrl_start: &sw_ctrls,
                    ctrl_end: None,
                    is_star_center: false,
                    windows: sw_windows,
                    controls_meta: meta,
                    user_metadata: BTreeMap::new(),
                    shared_align: ctx.shared_align_stereo,
                    optional_spcm: None,
                    pre_extracted_physics: None,
                    pre_rendered_aligned_wet: None,
                },
            )?;
            fb_spec
        } else {
            let cfg = StimulusConfig {
                stem_id: format!("switch_{p_idx}_opt_{opt_idx}"),
                sample_rate_hz: ctx.options.sample_rate_hz,
                topology_mode: ctx.options.topology_mode,
                control_values: sw_ctrls,
                sweep_end_control_values: None,
                active_control_id: p_idx as u16,
                is_master_volume_sweep: false,
                fixed_duration: ctx.options.fixed_duration,
                max_frames: 6,
                max_tail_frames: 2,
                noise_floor_dbfs: ctx.noise_floor_dbfs,
                seed_256: *ctx.seed_256,
            };
            let (_, mut cl_spec, warns, _, _, _) =
                run_single_closed_loop_stem(target, cfg, false, meta, ctx.shared_align, None)?;
            cl_spec.windows = sw_windows;
            out.warnings.extend(warns);
            cl_spec
        };
        out.specs.push(spec);
    }
    Ok(())
}

/// Captures 1D `Range` sweeps/stops and `Switch` options using signal-measured endpoint probes.
pub(super) fn capture_range_and_switch_stems(
    target: &mut dyn DistillationTarget,
    ctx: &SweepCaptureContext<'_>,
) -> Result<SweepStageOutput, DnamError> {
    let mut out = SweepStageOutput {
        specs: Vec::new(),
        warnings: Vec::new(),
        active_range_indices: Vec::new(),
        shape_spread_by_ctrl: vec![0.0_f64; ctx.manifest.base.controls_schema.len()],
    };
    for (p_idx, ctrl) in ctx.manifest.base.controls_schema.iter().enumerate() {
        match ctrl.kind {
            ControlKind::Range => {
                out.active_range_indices.push(p_idx);
                capture_range_control(target, ctx, p_idx, ctrl, &mut out)?;
            }
            ControlKind::Switch => {
                capture_switch_control(target, ctx, p_idx, ctrl, &mut out)?;
            }
        }
    }
    Ok(out)
}

/// Captures `Law 4D` commutator-pruned orthogonal Hadamard diagonal stems (`<= 2` diagonals).
pub(super) fn capture_hadamard_diagonal_stems(
    target: &mut dyn DistillationTarget,
    ctx: &SweepCaptureContext<'_>,
    active_range_indices: &[usize],
    shape_spread_by_ctrl: &[f64],
) -> Result<(Vec<VaultCaptureSpec>, Vec<CaptureWarning>, usize, usize), DnamError> {
    let controls_snapshot = &ctx.manifest.base.controls_schema;
    let empty_switches = BTreeMap::new();
    let total_possible_pairs = if active_range_indices.len() >= 2 {
        (active_range_indices.len() * (active_range_indices.len() - 1)) / 2
    } else {
        0
    };
    let orth_diags =
        build_orthogonal_hadamard_diagonals(active_range_indices, shape_spread_by_ctrl);
    let pruned_count = total_possible_pairs.saturating_sub(orth_diags.len());
    let mut specs = Vec::with_capacity(orth_diags.len());
    let mut warnings = Vec::new();
    let mut hadamard_run = 0_usize;

    for (diag_idx, signs) in orth_diags.iter().enumerate() {
        let mut v0 = ctx.base_ctrls.to_vec();
        let mut v1 = ctx.base_ctrls.to_vec();
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

        let meta = build_controls_meta_map(ctx.manifest, &v1, &empty_switches);
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
        let spec = if ctx.options.fixed_duration
            && let Some((fb_l, fb_r)) = ctx.full_dry_pair
        {
            let (_, fb_spec) = render_and_seal_fullband_stem(
                target,
                FullbandStemRequest {
                    stem_id: &format!("hadamard_diag_{diag_idx}"),
                    sample_rate_hz: ctx.options.sample_rate_hz,
                    topology_mode: ctx.options.topology_mode,
                    seed_256: ctx.seed_256,
                    carrier_bins: &ctx.probe_sched.carrier_bins,
                    valid_frames: ctx.options.max_frames_per_stem.max(12),
                    full_dry_l_48k: fb_l,
                    full_dry_r_48k: fb_r.as_deref(),
                    aux_in_3_48k: ctx.full_dry_aux.as_deref(),
                    ctrl_start: &v0,
                    ctrl_end: Some(&v1),
                    is_star_center: false,
                    windows: had_windows,
                    controls_meta: meta,
                    user_metadata: BTreeMap::new(),
                    shared_align: ctx.shared_align_stereo,
                    optional_spcm: None,
                    pre_extracted_physics: None,
                    pre_rendered_aligned_wet: None,
                },
            )?;
            fb_spec
        } else {
            target.reset_state();
            let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
                stem_id: format!("hadamard_diag_{diag_idx}"),
                sample_rate_hz: ctx.options.sample_rate_hz,
                topology_mode: ctx.options.topology_mode,
                control_values: v0.clone(),
                sweep_end_control_values: Some(v1.clone()),
                active_control_id: first_p as u16,
                is_master_volume_sweep: false,
                fixed_duration: ctx.options.fixed_duration,
                max_frames: ctx.options.max_frames_per_stem.max(12),
                max_tail_frames: 1,
                noise_floor_dbfs: ctx.noise_floor_dbfs,
                seed_256: *ctx.seed_256,
            });
            let mut law4d = HadamardLaw4dTracker::default();
            let mut sfm_hist: [Option<Vec<f64>>; 2] = [None, None];
            let mut last_resp: Option<StemResponse> = None;
            while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
                if ctrl.is_entering_coreset() {
                    target.reset_state();
                }
                let r = target.render_frame(&frame)?;
                if ctrl.is_in_active_learning() && !ctx.options.fixed_duration {
                    let (wl_48k, _) = ctrl.normalize_wet_to_48k(&r);
                    let wet_win = &wl_48k[wl_48k.len().saturating_sub(CYCLIC_WINDOW_SAMPLES)..];
                    let sfm = compute_hadamard_sfm_ratio(wet_win, ctrl.schedule_l(), &mut sfm_hist);
                    if law4d.observe_frame(sfm) {
                        ctrl.request_coreset_handoff();
                    }
                }
                last_resp = Some(r);
            }
            warnings.extend_from_slice(ctrl.warnings());
            let (_, mut cl_spec, _, _) = ctrl.seal_inpc_and_vault_spec_with_shared_align(
                false,
                meta.clone(),
                ctx.shared_align,
            )?;
            cl_spec.windows.clone_from(&had_windows);
            if let Some((fb_l, fb_r)) = ctx.full_dry_pair {
                let (_, fb_spec) = render_and_seal_fullband_stem(
                    target,
                    FullbandStemRequest {
                        stem_id: &format!("hadamard_diag_{diag_idx}"),
                        sample_rate_hz: ctx.options.sample_rate_hz,
                        topology_mode: ctx.options.topology_mode,
                        seed_256: ctx.seed_256,
                        carrier_bins: &ctx.probe_sched.carrier_bins,
                        valid_frames: ctx.options.max_frames_per_stem.max(12),
                        full_dry_l_48k: fb_l,
                        full_dry_r_48k: fb_r.as_deref(),
                        aux_in_3_48k: ctx.full_dry_aux.as_deref(),
                        ctrl_start: &v0,
                        ctrl_end: Some(&v1),
                        is_star_center: false,
                        windows: had_windows,
                        controls_meta: meta,
                        user_metadata: cl_spec.user_metadata.clone(),
                        shared_align: ctx.shared_align_stereo,
                        optional_spcm: None,
                        pre_extracted_physics: None,
                        pre_rendered_aligned_wet: None,
                    },
                )?;
                cl_spec = fb_spec;
            }
            cl_spec
        };
        specs.push(spec);
        hadamard_run += 1;
    }

    Ok((specs, warnings, hadamard_run, pruned_count))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::stimulus::fft::{
        ADC_PEAK_GOVERNOR_FS, PEDESTAL_BIN_MONO, build_disjoint_comb_schedule,
        synthesize_cyclic_comb_frame,
    };
    use crate::stimulus::pipeline::schoukens::build_schoukens_pintelon_coreset_48k;
    use crate::stimulus::waveforms::deterministic_seed_256;
    use crate::targets::StemStimulus;
    use dnam_engine::schema::{ControlSchemaDef, ControlUiHints};

    struct GainTarget;
    impl DistillationTarget for GainTarget {
        fn reset_state(&mut self) {}
        fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
            let g0 = f64::from(stim.control_values.first().copied().unwrap_or(0.5));
            let g1 = f64::from(stim.control_values.get(1).copied().unwrap_or(0.5));
            let out_1: Vec<f32> = stim
                .in_1
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let c0 = stim
                        .control_trajectories
                        .as_ref()
                        .and_then(|t| t.first())
                        .and_then(|v| v.get(i))
                        .map_or(g0, |&z| f64::from(z));
                    let c1 = stim
                        .control_trajectories
                        .as_ref()
                        .and_then(|t| t.get(1))
                        .and_then(|v| v.get(i))
                        .map_or(g1, |&z| f64::from(z));
                    (0.5 * (0.2 + c0 + c1) * f64::from(x)) as f32
                })
                .collect();
            Ok(StemResponse {
                stem_id: stim.stem_id.clone(),
                sample_rate_hz: stim.sample_rate_hz,
                out_1,
                out_2: None,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    fn mk_range(id: &str, default_val: f32) -> ControlSchemaDef {
        ControlSchemaDef {
            id: id.to_string(),
            name: id.to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: None,
            declared_range: Some([0.0, 1.0]),
            safe_clamped_range: Some([0.0, 1.0]),
            default: serde_json::json!(default_val),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn test_fullband_mode_a_mode_b_stops_and_hadamard_capture_without_solver() {
        let mut manifest = DnamProjManifest::default_reference();
        manifest.base.controls_schema = vec![mk_range("a", 0.02), mk_range("b", 0.98)];
        let seed_256 = deterministic_seed_256(0x7001);
        let probe_sched = build_disjoint_comb_schedule(
            48_000,
            false,
            Some(PEDESTAL_BIN_MONO),
            &BTreeSet::new(),
            &[],
        );
        let amps: Vec<(usize, f64)> = probe_sched.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
        let mut probe_dry = synthesize_cyclic_comb_frame(
            &amps,
            Some(PEDESTAL_BIN_MONO),
            0.05,
            0.16,
            ADC_PEAK_GOVERNOR_FS,
            &[0.0; 64],
        )
        .samples;
        probe_dry.resize(10 * ADAPTIVE_FRAME_SAMPLES, 0.0);

        let mut target = GainTarget;
        let base_ctrls = [0.02_f32, 0.98];
        let (star_probe_wet_l, _) = render_endpoint_probe_48k(
            &mut target,
            &probe_dry,
            48_000,
            crate::stimulus::laws::TopologyProbeMode::Mono1To1,
            &base_ctrls,
        )
        .unwrap();
        let full_dry = build_schoukens_pintelon_coreset_48k(
            &seed_256,
            crate::stimulus::laws::TopologyProbeMode::Mono1To1,
            96_000,
            &BTreeSet::new(),
        )
        .dry_l;
        let pair = Some((full_dry, None));

        // 1. Mode A sweep + Hadamard fixed_duration = true
        let opts_a = AdaptiveTrainingOptions {
            max_frames_per_stem: 6,
            fixed_duration: true,
            prefer_mode_b: false,
            ..Default::default()
        };
        let ctx_a = SweepCaptureContext {
            manifest: &manifest,
            options: &opts_a,
            base_ctrls: &base_ctrls,
            seed_256: &seed_256,
            probe_sched: &probe_sched,
            probe_dry_l_48k: &probe_dry,
            star_probe_wet_l: &star_probe_wet_l,
            full_dry_pair: &pair,
            full_dry_aux: &None,
            shared_align_stereo: SharedStemAlignment {
                lag_samples: 0.0,
                clock_drift_ppm: 0.0,
                invert_l: false,
                invert_r: false,
            },
            shared_align: Some((0.0, false)),
            star_tilt: 0.0,
            noise_floor_dbfs: -100.0,
        };
        let sw_a = capture_range_and_switch_stems(&mut target, &ctx_a).unwrap();
        assert_eq!(sw_a.specs.len(), 2);
        let (had_a, _, run_a, pruned_a) = capture_hadamard_diagonal_stems(
            &mut target,
            &ctx_a,
            &sw_a.active_range_indices,
            &[1.0, 1.0],
        )
        .unwrap();
        assert_eq!((had_a.len(), run_a, pruned_a), (1, 1, 0));

        // 2. Mode B stops (`base_ctrls` near `lo` and `hi`) + Hadamard fixed_duration = false
        let opts_b = AdaptiveTrainingOptions {
            max_frames_per_stem: 6,
            fixed_duration: false,
            prefer_mode_b: true,
            ..Default::default()
        };
        let ctx_b = SweepCaptureContext {
            options: &opts_b,
            ..ctx_a
        };
        let sw_b = capture_range_and_switch_stems(&mut target, &ctx_b).unwrap();
        assert_eq!(sw_b.specs.len(), 4);
        assert_eq!(sw_b.specs[0].stem_id, "stop_ctrl_0_0");
        assert!((sw_b.specs[0].windows[0].u_phys - 0.5).abs() < 1.0e-4);
        assert!((sw_b.specs[3].windows[0].u_phys - 0.5).abs() < 1.0e-4);
        let (had_b, _, run_b, _) = capture_hadamard_diagonal_stems(
            &mut target,
            &ctx_b,
            &sw_b.active_range_indices,
            &[1.0, 1.0],
        )
        .unwrap();
        assert_eq!((had_b.len(), run_b), (1, 1));
    }
}
