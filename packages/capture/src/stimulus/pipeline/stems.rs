#![forbid(unsafe_code)]
//! Endpoint Probing, Closed-Loop Stem Rendering, and Freeze-Transition Helpers
//! (`stimulus::pipeline::stems`).

use std::collections::BTreeMap;

use dnam_engine::schema::{ControlKind, DnamError};

use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::VaultCaptureSpec;
use crate::resampler::{resample_arbitrary_rate_to_48k, resampled_frame_len};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus};

use super::super::batch::{
    build_controls_meta_map, build_freeze_transition_stem_stimulus, resample_48k_channels_to_sr,
};
use super::super::fft::ADAPTIVE_FRAME_SAMPLES;
use super::super::laws::TopologyProbeMode;
use super::super::physics_probe;
use super::super::safeguards::CaptureWarning;
use super::super::sealing::{StemSealRequest, align_full_stem_and_seal_vault_with_override};
use super::super::{AdaptiveStimulusController, StimulusConfig};
use super::AdaptiveTrainingOptions;

pub(super) type StarProbe<'a> = (&'a [f32], &'a [f32], Option<&'a [f32]>, &'a [usize]);
pub(super) type ClosedLoopStemOut = (
    Vec<u8>,
    VaultCaptureSpec,
    Vec<CaptureWarning>,
    f64,
    f64,
    bool,
);

pub(super) fn spectral_tilt_from_stmc_4096(spec_4096: &[u8; 4_096]) -> f64 {
    let mut weighted_e = 0.0_f64;
    let mut total_e = 0.0_f64;
    for i in 1..512 {
        let off = i * 4;
        let m = f64::from(f32::from_le_bytes(
            spec_4096[off..off + 4].try_into().unwrap(),
        ));
        let p = m * m;
        weighted_e += ((i as f64) / 32.0) * p;
        total_e += p;
    }
    weighted_e / (total_e + 1.0e-18)
}

/// Renders the deterministic endpoint probe (`1` `96 ms` comb frame + `9` `96 ms` silent tail
/// frames = `864 ms` tail) and normalizes the wet response to `48 kHz` (`r[train.capture.invariants]`).
pub(super) fn render_endpoint_probe_48k(
    target: &mut dyn DistillationTarget,
    probe_dry_l_48k: &[f32],
    sample_rate_hz: u32,
    topology_mode: TopologyProbeMode,
    control_values: &[f32],
) -> Result<(Vec<f32>, Option<Vec<f32>>), DnamError> {
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let dry_r_48k = need_stereo.then_some(probe_dry_l_48k);
    let aux_48k = need_aux.then_some(probe_dry_l_48k);
    let (in_1, in_2, in_3) = resample_48k_channels_to_sr(sr, probe_dry_l_48k, dry_r_48k, aux_48k);

    target.reset_state();
    let resp = target.render_stem(&StemStimulus {
        stem_id: "endpoint_probe".to_string(),
        sample_rate_hz: sr,
        in_1,
        in_2,
        in_3,
        in_4: None,
        control_values: control_values.to_vec(),
        control_trajectories: None,
        freeze_transition_window: None,
    })?;

    let resp_sr = resp.sample_rate_hz.clamp(8_000, 384_000);
    let to_48k = |ch: &[f32]| -> Vec<f32> {
        if resp_sr == 48_000 {
            ch.to_vec()
        } else {
            let dst = resampled_frame_len(ch.len(), resp_sr, 48_000);
            resample_arbitrary_rate_to_48k(ch, resp_sr, Some(dst))
        }
    };
    let out_l = to_48k(&resp.out_1);
    let out_r = resp.out_2.as_deref().map(to_48k);
    Ok((out_l, out_r))
}

pub(super) fn run_single_closed_loop_stem(
    target: &mut dyn DistillationTarget,
    cfg: StimulusConfig,
    is_star_center: bool,
    controls_meta: BTreeMap<String, serde_json::Value>,
    shared_align: Option<(f64, bool)>,
    star_probe: Option<StarProbe<'_>>,
) -> Result<ClosedLoopStemOut, DnamError> {
    target.reset_state();
    let mut ctrl = AdaptiveStimulusController::new(cfg);
    let mut last_resp: Option<StemResponse> = None;
    while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
        if ctrl.is_entering_coreset() {
            target.reset_state();
        }
        last_resp = Some(target.render_frame(&frame)?);
    }
    let warnings = ctrl.warnings().to_vec();
    let (inpc, mut spec, lag, pol) = ctrl.seal_inpc_and_vault_spec_with_shared_align(
        is_star_center,
        controls_meta,
        shared_align,
    )?;
    if is_star_center
        && let Some((probe_dry_l, probe_wet_l, probe_wet_r, carrier_bins)) = star_probe
    {
        let (cdl, cdr, cwl, cwr) = ctrl.coreset_buffers_48k();
        let comb_len = ADAPTIVE_FRAME_SAMPLES
            .min(probe_dry_l.len())
            .min(probe_wet_l.len());
        let comb_wet_r = probe_wet_r.map(|r| &r[..comb_len.min(r.len())]);
        let tail_wet_l = if probe_wet_l.len() > ADAPTIVE_FRAME_SAMPLES {
            &probe_wet_l[ADAPTIVE_FRAME_SAMPLES..]
        } else {
            &[]
        };
        let tail_wet_r = probe_wet_r.map(|r| {
            if r.len() > ADAPTIVE_FRAME_SAMPLES {
                &r[ADAPTIVE_FRAME_SAMPLES..]
            } else {
                &[][..]
            }
        });
        let phys = physics_probe::extract_star_physics_from_audio(
            cdl,
            cdr,
            cwl,
            cwr,
            &probe_dry_l[..comb_len],
            &probe_wet_l[..comb_len],
            comb_wet_r,
            tail_wet_l,
            tail_wet_r,
            carrier_bins,
        );
        spec.extracted_physics = Some(phys.extracted);
        spec.user_metadata.extend(phys.metadata);
    }
    let tilt = spectral_tilt_from_stmc_4096(&spec.spectral_summary_4096);
    Ok((inpc, spec, warnings, tilt, lag, pol))
}

pub(super) fn capture_freeze_transition_stem_mode1(
    manifest: &DnamProjManifest,
    target: &mut dyn DistillationTarget,
    freeze_ctrl_idx: usize,
    options: &AdaptiveTrainingOptions,
    seed_256: &[u8; 256],
    carrier_bins: &[usize],
    shared_align: Option<(f64, bool)>,
) -> Result<VaultCaptureSpec, DnamError> {
    let stim = build_freeze_transition_stem_stimulus(
        manifest,
        freeze_ctrl_idx,
        options.sample_rate_hz,
        options.topology_mode,
        seed_256,
    );
    target.reset_state();
    let resp = target.render_stem(&stim)?;

    let to_48k = |ch: &[f32], sr: u32| -> Vec<f32> {
        let csr = sr.clamp(8_000, 384_000);
        if csr == 48_000 {
            ch.to_vec()
        } else {
            let dst = resampled_frame_len(ch.len(), csr, 48_000);
            resample_arbitrary_rate_to_48k(ch, csr, Some(dst))
        }
    };
    let dry_48k_l = to_48k(&stim.in_1, stim.sample_rate_hz);
    let dry_48k_r = stim.in_2.as_ref().map(|r| to_48k(r, stim.sample_rate_hz));
    let wet_48k_l = to_48k(&resp.out_1, resp.sample_rate_hz);
    let wet_48k_r = resp.out_2.as_ref().map(|r| to_48k(r, resp.sample_rate_hz));

    let ctrl_def = &manifest.base.controls_schema[freeze_ctrl_idx];
    let is_sw = ctrl_def.kind == ControlKind::Switch;
    let mut sw_map = BTreeMap::new();
    if is_sw && let Some(opt_name) = ctrl_def.declared_options.as_ref().and_then(|o| o.get(1)) {
        sw_map.insert(freeze_ctrl_idx, opt_name.clone());
    }
    let meta = build_controls_meta_map(manifest, &stim.control_values, &sw_map);
    let mut user_meta = BTreeMap::new();
    if let Some([n0, n1]) = stim.freeze_transition_window {
        user_meta.insert(
            "freeze_transition_window".to_string(),
            serde_json::json!([n0, n1]),
        );
    }
    let hi_val = stim
        .control_values
        .get(freeze_ctrl_idx)
        .copied()
        .unwrap_or(1.0);
    let windows = vec![
        StmcTrajectoryWindow {
            control_id: freeze_ctrl_idx as u16,
            ctrl_kind: u8::from(is_sw),
            active_opt: 0,
            u_phys: 0.0,
            window_weight: 1.0,
            emr_velocity: 0.1,
        },
        StmcTrajectoryWindow {
            control_id: freeze_ctrl_idx as u16,
            ctrl_kind: u8::from(is_sw),
            active_opt: u8::from(is_sw),
            u_phys: hi_val,
            window_weight: 1.0,
            emr_velocity: 0.1,
        },
    ];
    let (_, spec, _, _) = align_full_stem_and_seal_vault_with_override(
        StemSealRequest {
            stem_id: &stim.stem_id,
            is_star_center: false,
            seed_256,
            carrier_bins,
            valid_frames: 5,
            dry_48k_l: &dry_48k_l,
            dry_48k_r: dry_48k_r.as_deref(),
            wet_48k_l: &wet_48k_l,
            wet_48k_r: wet_48k_r.as_deref(),
            windows,
            controls_meta: meta,
            user_metadata: user_meta,
        },
        shared_align,
    )?;
    Ok(spec)
}
