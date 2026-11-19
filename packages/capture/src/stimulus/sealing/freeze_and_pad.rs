#![forbid(unsafe_code)]
//! Sample-Rate Normalization, Freeze-Transition Stem Synthesis, and `400 ms` Differential
//! Mic-Pad Clipped Stem Re-Capture (`stimulus::sealing::freeze_and_pad`).

use std::collections::BTreeSet;

use dnam_engine::schema::{ControlKind, DnamError};

use crate::align::{
    DifferentialPadCalibration, compute_pcm_sha256, measure_differential_pad_ratio,
    rescale_padded_stem,
};
use crate::proj::schema::DnamProjManifest;
use crate::proj::vault::{VaultCaptureSpec, replace_capture};
use crate::resampler::{
    resample_48k_to_arbitrary_rate, resample_arbitrary_rate_to_48k, resampled_frame_len,
};
use crate::targets::StemStimulus;
use crate::{CORESET_FP32_FRAMES_PER_STEM, CORESET_FP32_MIN_FRAMES_PER_STEM};

use super::super::laws::TopologyProbeMode;
use super::super::waveforms::{
    ActivePrimitiveKind, compute_stmc_psd_88_and_spectral_4096, encode_coreset_fp32_with_frames,
    synthesize_a5_musical_transient_coreset, synthesize_active_primitive_frame,
};
use super::super::{
    ADAPTIVE_FRAME_SAMPLES, ADC_PEAK_GOVERNOR_FS, PEDESTAL_BIN_MONO, build_disjoint_comb_schedule,
    synthesize_cyclic_comb_frame,
};
use super::controls::{control_declared_bounds, default_control_vector};
use super::{CORESET_A5_FRAMES_48K, CORESET_WITH_SETTLE_48K};

/// Resamples up to three `48 kHz` channels (`l`, optional `r`, optional `aux`) to `sr`.
pub(crate) fn resample_48k_channels_to_sr(
    sr: u32,
    l_48k: &[f32],
    r_48k: Option<&[f32]>,
    aux_48k: Option<&[f32]>,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    if sr == 48_000 {
        return (
            l_48k.to_vec(),
            r_48k.map(<[f32]>::to_vec),
            aux_48k.map(<[f32]>::to_vec),
        );
    }
    let dst_len = resampled_frame_len(l_48k.len(), 48_000, sr);
    let l = resample_48k_to_arbitrary_rate(l_48k, sr, Some(dst_len));
    let r = r_48k.map(|ch| resample_48k_to_arbitrary_rate(ch, sr, Some(dst_len)));
    let aux = aux_48k.map(|ch| resample_48k_to_arbitrary_rate(ch, sr, Some(dst_len)));
    (l, r, aux)
}

/// Normalizes a single captured stem channel from `sr` (`8..=384 kHz`) into `48 kHz`.
pub(crate) fn normalize_stem_channel_to_48k(samples: &[f32], sr: u32) -> Vec<f32> {
    let clamped_sr = sr.clamp(8_000, 384_000);
    if clamped_sr == 48_000 {
        samples.to_vec()
    } else {
        let dst_len = resampled_frame_len(samples.len(), clamped_sr, 48_000);
        resample_arbitrary_rate_to_48k(samples, clamped_sr, Some(dst_len))
    }
}

fn build_freeze_trajectories(
    manifest: &DnamProjManifest,
    freeze_ctrl_idx: usize,
    frame_len: usize,
    n_earliest: usize,
    n_latest: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let base_ctrls = default_control_vector(manifest);
    let ctrl_def = &manifest.base.controls_schema[freeze_ctrl_idx];
    let (unfrozen_val, frozen_val) = match ctrl_def.kind {
        ControlKind::Switch => (0.0_f32, 1.0_f32),
        ControlKind::Range => control_declared_bounds(ctrl_def),
    };
    let mut trajs = Vec::with_capacity(base_ctrls.len());
    for (q, &def_v) in base_ctrls.iter().enumerate() {
        if q == freeze_ctrl_idx {
            let tr: Vec<f32> = (0..frame_len)
                .map(|idx| {
                    if (n_earliest..n_latest).contains(&idx) {
                        frozen_val
                    } else {
                        unfrozen_val
                    }
                })
                .collect();
            trajs.push(tr);
        } else {
            trajs.push(vec![def_v; frame_len]);
        }
    }
    let mut active_ctrls = base_ctrls;
    if let Some(slot) = active_ctrls.get_mut(freeze_ctrl_idx) {
        *slot = frozen_val;
    }
    (active_ctrls, trajs)
}

/// Synthesizes a timed `freeze_latch` sustain-entry -> live play-over -> release [`StemStimulus`]
/// (`r[train.capture.planner.freeze-transition]`).
pub(crate) fn build_freeze_transition_stem_stimulus(
    manifest: &DnamProjManifest,
    freeze_ctrl_idx: usize,
    sample_rate_hz: u32,
    topology_mode: TopologyProbeMode,
    seed_256: &[u8; 256],
) -> StemStimulus {
    let empty_mask = BTreeSet::new();
    let sched_l =
        build_disjoint_comb_schedule(48_000, false, Some(PEDESTAL_BIN_MONO), &empty_mask, &[]);
    let bin_amps_l: Vec<(usize, f64)> = sched_l.carrier_bins.iter().map(|&b| (b, 1.0)).collect();
    let comb_l = synthesize_cyclic_comb_frame(
        &bin_amps_l,
        Some(PEDESTAL_BIN_MONO),
        0.05,
        0.18,
        ADC_PEAK_GOVERNOR_FS,
        &[0.0; 64],
    );
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let (mut coreset_l, mut coreset_r) = synthesize_a5_musical_transient_coreset(
        seed_256,
        CORESET_A5_FRAMES_48K,
        if need_stereo { 2 } else { 1 },
        48_000,
    );
    coreset_l.resize(CORESET_WITH_SETTLE_48K, 0.0);
    if let Some(ref mut cr) = coreset_r {
        cr.resize(CORESET_WITH_SETTLE_48K, 0.0);
    }

    // [0..9,216): unfrozen tank charge; [9,216..18,432): frozen play-over; [18,432..23,040): release;
    // [23,040..36,864): A5 transient coreset + settle.
    let mut wave_l_48k = Vec::with_capacity(8 * ADAPTIVE_FRAME_SAMPLES);
    wave_l_48k.extend_from_slice(&comb_l.samples);
    wave_l_48k.extend_from_slice(&synthesize_active_primitive_frame(
        ActivePrimitiveKind::P4TankDuckingBurst,
        0,
        false,
        ADC_PEAK_GOVERNOR_FS,
    ));
    let n_earliest_48k = wave_l_48k.len();
    wave_l_48k.extend_from_slice(&synthesize_active_primitive_frame(
        ActivePrimitiveKind::P1BranchCutChirp,
        0,
        false,
        ADC_PEAK_GOVERNOR_FS,
    ));
    wave_l_48k.extend_from_slice(&comb_l.samples);
    let n_latest_48k = wave_l_48k.len();
    wave_l_48k.resize(n_latest_48k + ADAPTIVE_FRAME_SAMPLES, 0.0);
    wave_l_48k.extend_from_slice(&coreset_l);

    let wave_r_48k = coreset_r.map(|cr| {
        let prefix_len = wave_l_48k.len() - CORESET_WITH_SETTLE_48K;
        let mut wr = wave_l_48k[..prefix_len].to_vec();
        wr.extend_from_slice(&cr);
        wr
    });
    let aux_48k = need_aux.then(|| wave_l_48k.clone());

    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let (in_1, in_2, in_3) =
        resample_48k_channels_to_sr(sr, &wave_l_48k, wave_r_48k.as_deref(), aux_48k.as_deref());
    let n_earliest = resampled_frame_len(n_earliest_48k, 48_000, sr).min(in_1.len());
    let n_latest = resampled_frame_len(n_latest_48k, 48_000, sr)
        .max(n_earliest + 1)
        .min(in_1.len());

    let (active_ctrls, trajs) =
        build_freeze_trajectories(manifest, freeze_ctrl_idx, in_1.len(), n_earliest, n_latest);

    StemStimulus {
        stem_id: format!("freeze_transition_{freeze_ctrl_idx}"),
        sample_rate_hz: sr,
        in_1,
        in_2,
        in_3,
        in_4: None,
        control_values: active_ctrls,
        control_trajectories: Some(trajs),
        freeze_transition_window: Some([n_earliest, n_latest]),
    }
}

fn recompute_rescaled_capture_stats(
    capture: &mut VaultCaptureSpec,
    num_frames: usize,
    l: &[f32],
    r: Option<&[f32]>,
    pad_gain_linear: f64,
) {
    capture.wet_coreset_fp32 = encode_coreset_fp32_with_frames(num_frames, l, r);
    let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(l, r);
    capture.band_psd_summary_88 = psd_88;
    capture.spectral_summary_4096 = spec_4096;
    capture.provenance.target_pcm_sha256 = compute_pcm_sha256(l);
    capture.pad_gain_f32 = pad_gain_linear as f32;
}

/// Re-normalizes and replaces a clipped stem after the user engages a physical `-10 dB` / `-20 dB`
/// mic/preamp pad (`Law 0.F`), using the `400 ms` `-18 dB` differential reference probe to recover
/// $\hat{G}_{\text{pad}}$ and rescale the re-captured stem by $\hat{G}_{\text{pad}}^{-1}$
/// (`r[train.capture.safeguards.mic-capsule-clip]`, `r[train.cache.proj-vault]`).
pub fn recapture_clipped_stem(
    proj_bytes: &[u8],
    stem_id: &str,
    prev_unpadded_probe_400ms: &[f32],
    new_padded_probe_400ms: &[f32],
    mut padded_capture: VaultCaptureSpec,
) -> Result<(Vec<u8>, DifferentialPadCalibration), DnamError> {
    let pad_cal =
        measure_differential_pad_ratio(prev_unpadded_probe_400ms, new_padded_probe_400ms, 128)
            .map_err(|e| {
                DnamError::SchemaValidationError(format!("mic-pad probe calibration failed: {e}"))
            })?;

    let is_stereo = padded_capture.wet_coreset_fp32.len() >= CORESET_FP32_MIN_FRAMES_PER_STEM * 8;
    let frame_bytes = if is_stereo { 8 } else { 4 };
    let num_frames =
        (padded_capture.wet_coreset_fp32.len() / frame_bytes).max(CORESET_FP32_FRAMES_PER_STEM);
    let decoded: Vec<f32> = padded_capture
        .wet_coreset_fp32
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    let rescaled = rescale_padded_stem(&decoded, pad_cal.pad_gain_linear).map_err(|e| {
        DnamError::SchemaValidationError(format!("mic-pad stem rescale failed: {e}"))
    })?;

    if is_stereo {
        let mut l = Vec::with_capacity(rescaled.len() / 2);
        let mut r = Vec::with_capacity(rescaled.len() / 2);
        for pair in rescaled.as_chunks::<2>().0 {
            l.push(pair[0]);
            r.push(pair[1]);
        }
        recompute_rescaled_capture_stats(
            &mut padded_capture,
            num_frames,
            &l,
            Some(&r),
            pad_cal.pad_gain_linear,
        );
    } else {
        recompute_rescaled_capture_stats(
            &mut padded_capture,
            num_frames,
            &rescaled,
            None,
            pad_cal.pad_gain_linear,
        );
    }

    let updated_proj = replace_capture(proj_bytes, stem_id, padded_capture)?;
    Ok((updated_proj, pad_cal))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dnam_engine::provenance::DefaultCryptoProvider;
    use dnam_engine::schema::{ControlSchemaDef, ControlUiHints};
    use serde_json::json;

    use crate::proj::bundle::{DnamProjView, build_dnam_proj_container};
    use crate::proj::directory::StmcTrajectoryWindow;
    use crate::proj::vault::add_capture_with_options;
    use crate::solver::SolverOptions;

    use super::super::controls::build_controls_meta_map;
    use super::super::{StemSealRequest, seal_coresets_to_inpc_and_vault_with_override};
    use super::*;

    #[test]
    fn test_resample_and_normalize_channels() {
        let dry_48k: Vec<f32> = (0..4_800).map(|i| (i as f32 * 0.02).sin()).collect();
        let (l0, r0, a0) = resample_48k_channels_to_sr(48_000, &dry_48k, Some(&dry_48k), None);
        assert_eq!(l0.len(), 4_800);
        assert_eq!(r0.unwrap().len(), 4_800);
        assert!(a0.is_none());
        assert_eq!(normalize_stem_channel_to_48k(&dry_48k, 48_000).len(), 4_800);

        let (l96, r96, a96) =
            resample_48k_channels_to_sr(96_000, &dry_48k, Some(&dry_48k), Some(&dry_48k));
        assert_eq!(l96.len(), 9_600);
        assert_eq!(r96.unwrap().len(), 9_600);
        assert_eq!(a96.unwrap().len(), 9_600);

        let back = normalize_stem_channel_to_48k(&l96, 96_000);
        assert_eq!(back.len(), 4_800);
        assert!(normalize_stem_channel_to_48k(&dry_48k[..120], 4_000).len() > 120);
    }

    #[test]
    fn test_build_freeze_transition_stem_stimulus_modes() {
        let mut manifest = DnamProjManifest::default_reference();
        manifest.base.controls_schema = vec![
            ControlSchemaDef {
                id: "hold".to_string(),
                name: "Hold".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: json!("off"),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: Some(vec!["off".to_string(), "on".to_string()]),
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "depth".to_string(),
                name: "Depth".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: None,
                declared_range: Some([0.1, 0.9]),
                safe_clamped_range: None,
                default: json!(0.5),
                conditional_min: BTreeMap::new(),
                conditional_max: BTreeMap::new(),
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let seed = [0x3Au8; 256];
        let st_switch = build_freeze_transition_stem_stimulus(
            &manifest,
            0,
            48_000,
            TopologyProbeMode::TrueStereo2To2,
            &seed,
        );
        assert!(st_switch.in_2.is_some());
        assert!(st_switch.in_3.is_none());
        let [w0, w1] = st_switch.freeze_transition_window.unwrap();
        let tr = &st_switch.control_trajectories.unwrap()[0];
        assert!((tr[0] - 0.0).abs() < 1.0e-6);
        assert!((tr[w0] - 1.0).abs() < 1.0e-6);
        assert!((tr[w1] - 0.0).abs() < 1.0e-6);

        let st_4cm = build_freeze_transition_stem_stimulus(
            &manifest,
            1,
            44_100,
            TopologyProbeMode::FourCableMethod4Cm,
            &seed,
        );
        assert_eq!(st_4cm.sample_rate_hz, 44_100);
        assert!(st_4cm.in_2.is_none());
        assert!(st_4cm.in_3.is_some());
        assert!((st_4cm.control_values[1] - 0.9).abs() < 1.0e-6);
    }

    #[test]
    fn test_recapture_clipped_stem_mono_stereo_and_error() {
        let manifest = DnamProjManifest::default_reference();
        let seed = [0x5Cu8; 256];
        let (dry_l, dry_r_opt) = synthesize_a5_musical_transient_coreset(&seed, 12_000, 2, 48_000);
        let dry_r = dry_r_opt.unwrap();
        let meta = build_controls_meta_map(&manifest, &[0.5], &BTreeMap::new());
        let win = vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 0.5,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }];
        let (inpc, spec_mono, _, _) = seal_coresets_to_inpc_and_vault_with_override(
            StemSealRequest {
                stem_id: "clip_stem",
                is_star_center: true,
                seed_256: &seed,
                carrier_bins: &[8, 24],
                valid_frames: 2,
                dry_48k_l: &dry_l,
                dry_48k_r: None,
                wet_48k_l: &dry_l,
                wet_48k_r: None,
                windows: win.clone(),
                controls_meta: meta.clone(),
                user_metadata: BTreeMap::new(),
            },
            Some((0.0, false)),
        )
        .unwrap();

        let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();
        let proj_mono = add_capture_with_options(
            &empty_proj,
            spec_mono.clone(),
            Some(&inpc),
            SolverOptions::preserve_backbone(),
        )
        .unwrap();

        let (probe_unpadded, _) = synthesize_a5_musical_transient_coreset(&seed, 19_200, 1, 48_000);
        let probe_padded: Vec<f32> = probe_unpadded.iter().map(|&v| v * 0.316_227_76).collect();

        assert!(
            recapture_clipped_stem(
                &proj_mono,
                "clip_stem",
                &[0.0; 256],
                &[0.0; 256],
                spec_mono.clone()
            )
            .is_err()
        );

        let (re_mono, cal_mono) = recapture_clipped_stem(
            &proj_mono,
            "clip_stem",
            &probe_unpadded,
            &probe_padded,
            spec_mono,
        )
        .unwrap();
        assert!((cal_mono.pad_gain_linear - 0.316_227_76).abs() < 1.0e-3);
        let crypto = DefaultCryptoProvider::default();
        assert!(DnamProjView::from_bytes(&re_mono, &crypto).is_ok());

        // Stereo stem recapture
        let (inpc_st, spec_stereo, _, _) = seal_coresets_to_inpc_and_vault_with_override(
            StemSealRequest {
                stem_id: "clip_stem",
                is_star_center: true,
                seed_256: &seed,
                carrier_bins: &[8, 24],
                valid_frames: 2,
                dry_48k_l: &dry_l,
                dry_48k_r: Some(&dry_r),
                wet_48k_l: &dry_l,
                wet_48k_r: Some(&dry_r),
                windows: win,
                controls_meta: meta,
                user_metadata: BTreeMap::new(),
            },
            Some((0.0, false)),
        )
        .unwrap();
        let empty_st = build_dnam_proj_container(&manifest, &[]).unwrap();
        let proj_st = add_capture_with_options(
            &empty_st,
            spec_stereo.clone(),
            Some(&inpc_st),
            SolverOptions::preserve_backbone(),
        )
        .unwrap();
        let (re_st, cal_st) = recapture_clipped_stem(
            &proj_st,
            "clip_stem",
            &probe_unpadded,
            &probe_padded,
            spec_stereo,
        )
        .unwrap();
        assert!((cal_st.pad_gain_linear - 0.316_227_76).abs() < 1.0e-3);
        let view_st = DnamProjView::from_bytes(&re_st, &crypto).unwrap();
        assert_eq!(view_st.manifest.stems_registry[0].stem_id, "clip_stem");
    }
}
