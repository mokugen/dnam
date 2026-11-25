#![forbid(unsafe_code)]
//! Fast non-ignored self-tests for the `M25-00` 3-level qualification harness using strictly
//! local synthetic [`BlackBoxModel`] targets (`0` dependencies on `Chain01..Chain18`).

use dnam_capture::metrics::repeatability::TargetRepeatabilityProfile;
use dnam_capture::solver::{ComputeBackendPreference, SolverQualityMode, canonical_besr_warmup};
use dnam_capture::stimulus::TopologyProbeMode;
use dnam_capture::{CORESET_FP32_CANONICAL_25S_FRAMES, DistillationTarget, StemStimulus};
use dnam_engine::schema::DnamError;
use dnam_fixtures::{
    BlackBoxModel, FixtureBitDepth, FixtureControlSpec, FixtureDeviceMetadata, FixtureSwitchSpec,
};

use super::adapter::{
    FixtureDistillationAdapter, build_parametric_manifest_for_model, build_star_manifest_for_model,
    infer_topology_mode,
};
use super::excitations::{build_unseen_control_vectors, build_unseen_holdout_excitation_48k};
use super::harness::{
    EvalMatrixConfig, QualifyConfig, assert_qualification_passed, evaluate_tier_outputs_4metric,
    qualify_fixture_3level,
};

#[derive(Clone, Debug)]
struct SynthLtiStereoModel {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    hist: [f32; 4],
}

impl SynthLtiStereoModel {
    fn new() -> Self {
        Self {
            sample_rate_hz: 48_000,
            bit_depth: FixtureBitDepth::Float32,
            hist: [0.0; 4],
        }
    }
}

impl BlackBoxModel for SynthLtiStereoModel {
    fn reset(&mut self) {
        self.hist = [0.0; 4];
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            id: "Synth-LTI-Stereo",
            name: "Synthetic LTI Stereo Self-Test Model",
            category: "self_test",
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            expected_tail_ms: 2.0,
            supports_tempo_sync: false,
            controls: &[],
            switches: &[],
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = sample_rate_hz;
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.bit_depth = bit_depth;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        _controls: &[f32],
        _switches: &[u8],
    ) {
        let x0 = inputs.first().copied().unwrap_or(0.0);
        let y_l = 0.72 * x0 + 0.14 * self.hist[1];
        let y_r = 0.68 * self.hist[0] - 0.11 * self.hist[2];
        self.hist = [x0, self.hist[0], self.hist[1], self.hist[2]];
        if let Some(o0) = outputs.get_mut(0) {
            *o0 = y_l;
        }
        if let Some(o1) = outputs.get_mut(1) {
            *o1 = y_r;
        }
    }
}

const PARAM_CONTROLS: [FixtureControlSpec; 1] = [FixtureControlSpec {
    id: "level_tilt",
    name: "Level Tilt",
    unit: "%",
    default_norm: 0.0,
    min_physical: 0.0,
    max_physical: 100.0,
}];

const PARAM_SWITCHES: [FixtureSwitchSpec; 1] = [FixtureSwitchSpec {
    id: "voicing",
    name: "Voicing",
    options: &["Flat", "Warm"],
    default_index: 0,
}];

#[derive(Clone, Debug)]
struct SynthParametricDriveModel {
    sample_rate_hz: u32,
    lp_state: f64,
}

impl SynthParametricDriveModel {
    fn new() -> Self {
        Self {
            sample_rate_hz: 48_000,
            lp_state: 0.0,
        }
    }
}

impl BlackBoxModel for SynthParametricDriveModel {
    fn reset(&mut self) {
        self.lp_state = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            id: "Synth-Parametric-Drive",
            name: "Synthetic Parametric Gain + Switch Self-Test Model",
            category: "self_test",
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: FixtureBitDepth::Float32,
            expected_tail_ms: 5.0,
            supports_tempo_sync: false,
            controls: &PARAM_CONTROLS,
            switches: &PARAM_SWITCHES,
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = sample_rate_hz;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let x = f64::from(inputs.first().copied().unwrap_or(0.0));
        let u0 = f64::from(controls.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0));
        let sw = switches.first().copied().unwrap_or(0);
        let gain = 0.65 + 0.25 * u0;
        let drive_in = gain * x;
        let nl = drive_in - 0.04 * drive_in.powi(3);
        let pole = if sw == 1 { 0.45 } else { 0.0 };
        self.lp_state = (1.0 - pole) * nl + pole * self.lp_state;
        if let Some(o0) = outputs.get_mut(0) {
            *o0 = self.lp_state as f32;
        }
    }
}

#[derive(Clone, Debug)]
struct SynthAuxKeyModel {
    sample_rate_hz: u32,
}

impl BlackBoxModel for SynthAuxKeyModel {
    fn reset(&mut self) {}

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            id: "Synth-AuxKey",
            name: "Synthetic Dual-In Mono AuxKey Self-Test",
            category: "self_test",
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: FixtureBitDepth::Float32,
            expected_tail_ms: 1.0,
            supports_tempo_sync: false,
            controls: &PARAM_CONTROLS,
            switches: &PARAM_SWITCHES,
        }
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let x1 = inputs.first().copied().unwrap_or(0.0);
        let x2 = inputs.get(1).copied().unwrap_or(0.0);
        let u0 = controls.first().copied().unwrap_or(0.0);
        let g = if switches.first().copied().unwrap_or(0) == 1 {
            0.5
        } else {
            1.0
        };
        if let Some(o0) = outputs.get_mut(0) {
            *o0 = g * (x1 + 0.25 * (1.0 + u0) * x2);
        }
    }
}

#[test]
fn test_m25_00_adapter_manifest_and_unseen_control_generators() {
    let lti = SynthLtiStereoModel::new();
    assert_eq!(
        infer_topology_mode(&lti),
        TopologyProbeMode::MonoToStereo1To2
    );
    assert!(build_unseen_control_vectors(&lti.metadata()).is_empty());
    let star_manifest = build_star_manifest_for_model(&lti);
    assert!(star_manifest.base.controls_schema.is_empty());
    assert_eq!(star_manifest.base.io_config.output_mode, "stereo");

    let aux = SynthAuxKeyModel {
        sample_rate_hz: 44_100,
    };
    assert_eq!(
        infer_topology_mode(&aux),
        TopologyProbeMode::DualInToMono2To1
    );
    let param_manifest = build_parametric_manifest_for_model(&aux);
    assert_eq!(param_manifest.base.controls_schema.len(), 2);
    assert_eq!(param_manifest.base.io_config.inputs, 2);
    assert_eq!(param_manifest.base.io_config.output_mode, "mono");

    let settings = build_unseen_control_vectors(&aux.metadata());
    assert_eq!(settings.len(), 4);
    assert_eq!(settings.iter().filter(|s| s.flips_switch).count(), 1);
    for s in &settings {
        assert_eq!(s.control_values.len(), 2);
        let u0 = s.control_values[0];
        assert!(u0.abs() > 0.10 && (u0.abs() - 1.0).abs() > 0.10);
    }

    // Verify 44.1 kHz resampling + in_3 Aux-Key routing + switch trajectories
    let mut adapter = FixtureDistillationAdapter::from_model(aux);
    let n = 480;
    let in_1 = vec![0.4_f32; n];
    let in_3 = vec![0.2_f32; n];
    let trajs = vec![vec![1.0_f32; n], vec![1.0_f32; n]];
    let resp = adapter
        .render_stem(&StemStimulus {
            stem_id: "aux_44k1".to_string(),
            sample_rate_hz: 48_000,
            in_1,
            in_2: None,
            in_3: Some(in_3),
            in_4: None,
            control_values: vec![1.0, 1.0],
            control_trajectories: Some(trajs),
            freeze_transition_window: None,
        })
        .unwrap();
    assert_eq!(resp.out_1.len(), n);
    assert!(resp.out_2.is_none());
    let mid_val = resp.out_1[n / 2];
    // sw=1 (g=0.5), u0=1.0 -> 0.5 * (0.4 + 0.5 * 0.2) = 0.25
    assert!((mid_val - 0.25).abs() < 0.02, "mid_val={mid_val}");

    // Holdout stereo vs mono generation check
    let (h_l, h_r) =
        build_unseen_holdout_excitation_48k(99, TopologyProbeMode::TrueStereo2To2, 2_048);
    assert_eq!(h_l.len(), 2_048);
    assert_eq!(h_r.as_ref().map(|v| v.len()), Some(2_048));
    let (h_m, h_mr) = build_unseen_holdout_excitation_48k(99, TopologyProbeMode::Mono1To1, 2_048);
    assert_eq!(h_m.len(), 2_048);
    assert!(h_mr.is_none());

    let can = QualifyConfig::canonical(ComputeBackendPreference::Auto);
    assert_eq!(can.max_frames_per_stem, 16);
    assert_eq!(can.coreset_frames_48k, CORESET_FP32_CANONICAL_25S_FRAMES);
    assert!(can.run_live_null_test);
    assert_eq!(can.holdout_frames_48k, 48_000);
    assert_eq!(
        can.quality_mode,
        SolverQualityMode::BestUntilDiminishingReturns
    );
}

#[test]
fn test_m25_00_harness_l1_l2_cpu_and_negative_falsification() {
    // 1. P == 0 stereo LTI chain qualifies through L1 -> L2 with level3 == None
    let rep_lti = qualify_fixture_3level(
        SynthLtiStereoModel::new,
        QualifyConfig::fast_self_test(ComputeBackendPreference::Cpu),
    );
    assert!(rep_lti.level3.is_none());
    assert_qualification_passed(&rep_lti);

    // 2. Negative falsification (inverted holdout): MUST fail evaluate_tier_outputs_4metric
    let (sig_f32, _) = build_unseen_holdout_excitation_48k(7, TopologyProbeMode::Mono1To1, 4_096);
    let tgt: Vec<f64> = sig_f32.iter().copied().map(f64::from).collect();
    let inv: Vec<f64> = tgt.iter().map(|&v| -v).collect();
    let bad_tiers = [inv.clone(), inv.clone(), inv.clone(), inv];
    let profiles = [TargetRepeatabilityProfile::deterministic()];
    let bad_eval = evaluate_tier_outputs_4metric(
        EvalMatrixConfig {
            level_name: "NEG",
            require_target_floors: true,
            require_monotone: true,
            require_lufs_parity: true,
            sample_rate_hz: 48_000.0,
            warmup: canonical_besr_warmup(tgt.len()),
            zero_dc_ok: true,
        },
        &[tgt],
        None,
        &[bad_tiers],
        &profiles,
        &[[-20.0; 4]],
    );
    assert!(!bad_eval.passes);
    assert!(!bad_eval.failure_reasons.is_empty());

    // 3. Negative falsification (ringing above -60 dB inside dry-silent gap): MUST fail
    let mut dry_gap = vec![0.0_f64; 16_384];
    let mut tgt_gap = vec![0.0_f64; 16_384];
    for i in 0..16_384 {
        if !(2_400..14_400).contains(&i) {
            let s = 0.25 * (core::f64::consts::TAU * 440.0 * (i as f64) / 48_000.0).sin();
            dry_gap[i] = s;
            tgt_gap[i] = s;
        }
    }
    let mut ring = tgt_gap.clone();
    for (i, slot) in ring.iter_mut().enumerate().take(14_000).skip(2_600) {
        *slot = 1.0e-2 * (core::f64::consts::TAU * 880.0 * (i as f64) / 48_000.0).sin();
    }
    let ring_tiers = [ring.clone(), ring.clone(), ring.clone(), ring];
    let ring_eval = evaluate_tier_outputs_4metric(
        EvalMatrixConfig {
            level_name: "NEG_GAP_RING",
            require_target_floors: true,
            require_monotone: true,
            require_lufs_parity: true,
            sample_rate_hz: 48_000.0,
            warmup: 64,
            zero_dc_ok: true,
        },
        &[tgt_gap],
        Some(&[dry_gap]),
        &[ring_tiers],
        &profiles,
        &[[-20.0; 4]],
    );
    assert!(!ring_eval.passes);
    assert!(!ring_eval.failure_reasons.is_empty());
}

#[test]
fn test_m25_00_harness_l1_l2_l3_parametric_cpu() {
    // P == 2 parametric drive+switch chain qualifies strictly through L1 -> L2 -> L3
    let rep_param = qualify_fixture_3level(
        SynthParametricDriveModel::new,
        QualifyConfig::fast_self_test(ComputeBackendPreference::Cpu),
    );
    let l3 = rep_param
        .level3
        .as_ref()
        .expect("L3 must execute when P > 0");
    assert_eq!(l3.len(), 4);
    assert_eq!(l3[0].label, "interior_low");
    assert!(rep_param.level1.lufs_ok.iter().all(|&ok| ok));
    assert!(
        l3.iter().any(|s| s.frozen_ultra_besr_db > -20.0),
        "expected unseen control settings to diverge from frozen StarCenter: {l3:?}"
    );
    assert_qualification_passed(&rep_param);
}

#[test]
fn test_m25_00_backend_preference_cpu_and_wgpu() {
    let cpu_ok = ComputeBackendPreference::Cpu.resolve_runner();
    assert!(cpu_ok.is_ok());

    if cfg!(target_os = "macos") {
        let rep_gpu = qualify_fixture_3level(
            SynthLtiStereoModel::new,
            QualifyConfig::fast_self_test(ComputeBackendPreference::Wgpu),
        );
        assert!(rep_gpu.level3.is_none());
        assert_qualification_passed(&rep_gpu);
    } else if let Err(err) = ComputeBackendPreference::Wgpu.resolve_runner() {
        assert!(matches!(err, DnamError::SchemaValidationError(_)));
    }
}
