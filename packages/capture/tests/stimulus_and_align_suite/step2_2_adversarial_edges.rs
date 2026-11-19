#![forbid(unsafe_code)]
//! Adversarial & edge-case regression suite for Phase 2, Step 2.2:
//! - `Law 0.A` `LtvRegime::EnvelopeSvf` detection, warning, and 2-level (`-18 / -6 dBFS`) pole-zoom
//! - `Law 4D` Hadamard early-stop coreset handoff (`STMC` matched `A5` coreset) & `<= 2` orthogonal diagonals
//! - `REQ-1.6` zero control-name heuristic delay-knob `Mode B` probing (`r[train.capture.invariants]`)
//! - `FreezeLatch` sustain-entry -> play-over -> release transition capture (`r[train.capture.planner.freeze-transition]`)
//! - Mode 2 `96 kHz` Sans-IO batch `Switch` + `Mode B` controls + bulk latency alignment
//! - Simultaneous `Law 0.B` long echo + `Law 0.F` asymmetric fuzz sub-`0 dBFS` capsule clip & Case A replay
//! - Extreme `8 kHz` and `384 kHz` multi-port `4CM` (`in_3` replay/coreset preservation) & `2->2`
//! - `1.0s` Live Null-Test stereo `out_2` sensitivity, `1.0 * STAT` preservation, `96 kHz` rig rate, and rollback
// verify[train.capture.invariants]
// verify[train.capture.planner.freeze-transition]
// verify[train.capture.planner.schedule]
// verify[train.capture.excitation.codesigned]
// verify[train.capture.alignment.multi-regime]
// verify[train.cache.proj-vault]

use std::collections::BTreeMap;

use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, INPC_MONO_BYTES, INPC_STEREO_BYTES, ProjChunkKind,
    decode_proj_inpc_f32, decode_proj_stmc_f32, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::stimulus::{
    AdaptiveStimulusController, AdaptiveTrainingOptions, BatchStimulusOptions, CaptureWarning,
    ExcitationBranch, KnobAcousticClass, LtvRegime, StimulusConfig, TopologyProbeMode,
    build_orthogonal_hadamard_diagonals, classify_knob_acoustic_class, generate_stimulus_batch,
    solve_from_responses, train_adaptive_target,
};
use dnam_capture::{
    CORESET_FP32_FRAMES_PER_STEM, CORESET_FP32_STEREO_BYTES, DistillationTarget, StemResponse,
    StemStimulus, StimulusFrame,
};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlSchemaDef, DnamError,
};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain06EnvelopeResonantSvf,
    Chain08LongBbdEchoAnd30sShimmer, Chain12FullEndToEndStereoRig, Chain12RigMode,
    MicrophoneCapsuleClipDecorator,
};

fn make_range_ctrl(
    id: &str,
    name: &str,
    lo: f32,
    hi: f32,
    def: f32,
    calibration: Option<ControlCalibration>,
) -> ControlSchemaDef {
    let mut c: ControlSchemaDef = serde_json::from_value(serde_json::json!({
        "id": id,
        "name": name,
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [lo, hi],
        "safe_clamped_range": [lo, hi],
        "default": def
    }))
    .unwrap();
    c.calibration = calibration;
    c
}

fn make_switch_ctrl(
    id: &str,
    options: &[&str],
    calibration: Option<ControlCalibration>,
) -> ControlSchemaDef {
    let opts: Vec<String> = options.iter().map(|s| (*s).to_string()).collect();
    let mut c: ControlSchemaDef = serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "switch",
        "declared_options": opts,
        "observed_options": opts,
        "default": options[0]
    }))
    .unwrap();
    c.calibration = calibration;
    c
}

fn mono_resp(stem_id: impl Into<String>, sample_rate_hz: u32, out_1: Vec<f32>) -> StemResponse {
    StemResponse {
        stem_id: stem_id.into(),
        sample_rate_hz,
        out_1,
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    }
}

fn stereo_resp(
    stem_id: impl Into<String>,
    sample_rate_hz: u32,
    out_1: Vec<f32>,
    out_2: Vec<f32>,
) -> StemResponse {
    StemResponse {
        stem_id: stem_id.into(),
        sample_rate_hz,
        out_1,
        out_2: Some(out_2),
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    }
}

fn render_od_batch_96k(
    od: &mut Chain01SymmetricOd,
    stems: &[StemStimulus],
    delay_samples: usize,
) -> Vec<StemResponse> {
    stems
        .iter()
        .map(|s| {
            od.reset();
            let sw = s.control_values.get(2).copied().unwrap_or(0.0).round() as u8;
            let mut out_1 = vec![0.0_f32; delay_samples];
            for &x in &s.in_1[..s.in_1.len() - delay_samples] {
                let mut ob = [0.0_f32];
                od.process_frame(
                    &[x],
                    &mut ob,
                    &[s.control_values[0], s.control_values[1], 0.5],
                    &[sw.min(1)],
                );
                out_1.push(ob[0]);
            }
            mono_resp(&s.stem_id, 96_000, out_1)
        })
        .collect()
}

struct SeparableTwoKnobTarget;

impl DistillationTarget for SeparableTwoKnobTarget {
    fn reset_state(&mut self) {}

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        let mut out_1 = Vec::with_capacity(stim.in_1.len());
        let mut lp = 0.0_f32;
        for (i, &x) in stim.in_1.iter().enumerate() {
            let c0 = stim
                .control_trajectories
                .as_ref()
                .and_then(|t| t.first())
                .and_then(|tr| tr.get(i).copied())
                .or_else(|| stim.control_values.first().copied())
                .unwrap_or(0.5);
            let c1 = stim
                .control_trajectories
                .as_ref()
                .and_then(|t| t.get(1))
                .and_then(|tr| tr.get(i).copied())
                .or_else(|| stim.control_values.get(1).copied())
                .unwrap_or(0.5);
            lp += 0.18 * (x - lp);
            // Strictly additive linear combination of two distinct spectral shapes:
            // large per-knob spectral shape spread (> 0.15), zero cross-product c0 * c1!
            out_1.push((0.30 + 0.55 * c0) * (x - lp) + (0.30 + 0.55 * c1) * lp);
        }
        Ok(mono_resp(&stim.stem_id, stim.sample_rate_hz, out_1))
    }
}

struct Chain08OpaqueTarget {
    bbd: Chain08LongBbdEchoAnd30sShimmer,
}

impl Chain08OpaqueTarget {
    fn new() -> Self {
        Self {
            bbd: Chain08LongBbdEchoAnd30sShimmer::new(),
        }
    }
}

impl DistillationTarget for Chain08OpaqueTarget {
    fn reset_state(&mut self) {
        self.bbd.reset();
    }

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        if let Some([n0, n1]) = stim.freeze_transition_window {
            self.bbd.set_freeze_window(n0, n1);
        }
        let n = stim.in_1.len();
        let (mut out_1, mut out_2) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let time_u = stim
                .control_trajectories
                .as_ref()
                .and_then(|t| t.first())
                .and_then(|tr| tr.get(i).copied())
                .or_else(|| stim.control_values.first().copied())
                .unwrap_or(-1.0);
            let freeze_sw = stim
                .control_trajectories
                .as_ref()
                .and_then(|t| t.get(1))
                .and_then(|tr| tr.get(i).copied())
                .or_else(|| stim.control_values.get(1).copied())
                .unwrap_or(0.0)
                >= 0.5;
            let xr = stim.in_2.as_ref().map_or(stim.in_1[i], |r| r[i]);
            let [yl, yr] = self
                .bbd
                .step_stereo(stim.in_1[i], xr, &[time_u, -0.4, 0.7], freeze_sw);
            out_1.push(yl);
            out_2.push(yr);
        }
        Ok(stereo_resp(
            &stim.stem_id,
            stim.sample_rate_hz,
            out_1,
            out_2,
        ))
    }

    fn render_frame(&mut self, frame: &StimulusFrame) -> Result<StemResponse, DnamError> {
        let n = frame.in_1.len();
        let (mut out_1, mut out_2) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let time_u = frame
                .control_trajectories
                .as_ref()
                .and_then(|t| t.first())
                .and_then(|tr| tr.get(i).copied())
                .or_else(|| frame.control_values.first().copied())
                .unwrap_or(-1.0);
            let freeze_sw = frame.control_values.get(1).copied().unwrap_or(0.0) >= 0.5;
            let xr = frame.in_2.as_ref().map_or(frame.in_1[i], |r| r[i]);
            let [yl, yr] = self
                .bbd
                .step_stereo(frame.in_1[i], xr, &[time_u, -0.4, 0.7], freeze_sw);
            out_1.push(yl);
            out_2.push(yr);
        }
        Ok(stereo_resp(
            format!("frame_{}", frame.frame_index),
            frame.sample_rate_hz,
            out_1,
            out_2,
        ))
    }
}

#[test]
fn test_envelope_svf_ltv_regime_warning_and_two_level_pole_zoom() {
    let mut svf = Chain06EnvelopeResonantSvf::new();
    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        max_frames: 10,
        max_tail_frames: 0,
        fixed_duration: true,
        ..StimulusConfig::default()
    });

    let mut last_resp: Option<StemResponse> = None;
    let mut active_comb_rms = Vec::new();
    while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
        if ctrl.is_in_active_learning() && ctrl.ltv_regime() == LtvRegime::EnvelopeSvf {
            let rms = (frame
                .in_1
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / (frame.in_1.len() as f64))
                .sqrt();
            active_comb_rms.push(rms);
        }
        let mut out_1 = Vec::with_capacity(frame.in_1.len());
        for &x in &frame.in_1 {
            let mut ob = [0.0_f32];
            // High sensitivity + high Q resonance + upward sweep
            svf.process_frame(&[x], &mut ob, &[0.95, 0.90, -0.6], &[0]);
            out_1.push(ob[0]);
        }
        last_resp = Some(mono_resp(format!("f_{}", frame.frame_index), 48_000, out_1));
    }

    assert_eq!(ctrl.ltv_regime(), LtvRegime::EnvelopeSvf);
    assert!(
        ctrl.warnings().iter().any(|w| matches!(
            w,
            CaptureWarning::LtvModulationDetected {
                regime: LtvRegime::EnvelopeSvf,
                ..
            }
        )),
        "expected LtvModulationDetected(EnvelopeSvf), got {:?}",
        ctrl.warnings()
    );
    assert_eq!(
        ctrl.active_branch(),
        ExcitationBranch::Branch1CNonlinearCascade,
        "quiet -18 dBFS pole-zoom frames must not flip EnvelopeSvf into PureLti"
    );
    let min_rms = active_comb_rms
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_rms = active_comb_rms.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        max_rms > 1.6 * min_rms,
        "EnvelopeSvf must alternate between low (-18 dBFS) and high (-6 dBFS) pole-zoom levels: {active_comb_rms:?}"
    );
}

#[test]
fn test_hadamard_law4d_early_stop_seals_matched_a5_coreset_and_orthogonal_diagonals() {
    // 1. Verify P = 4 coupled controls produce <= 2 orthogonal Walsh-Hadamard diagonals
    let diags = build_orthogonal_hadamard_diagonals(&[0, 1, 2, 3], &[0.3, 0.3, 0.3, 0.3]);
    assert_eq!(
        diags.len(),
        2,
        "must cap at <= 2 orthogonal Hadamard diagonals"
    );
    let dot: f32 = diags[0]
        .iter()
        .zip(&diags[1])
        .map(|(&(_, s0), &(_, s1))| s0 * s1)
        .sum();
    assert!(
        dot.abs() < 1.0e-6,
        "Hadamard sign vectors must be orthogonal"
    );

    // 2. Run Mode 1 with max_frames_per_stem = 18 >= 12 on a 2-knob separable target so Law 4D
    //    triggers early-stop at frame 12 AND still captures the A5 coreset into STMC!
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![
        make_range_ctrl("treble_gain", "Treble", 0.0, 1.0, 0.5, None),
        make_range_ctrl("bass_gain", "Bass", 0.0, 1.0, 0.5, None),
    ];
    let mut target = SeparableTwoKnobTarget;
    let rep = train_adaptive_target(
        manifest,
        &mut target,
        &AdaptiveTrainingOptions {
            max_frames_per_stem: 18,
            run_live_null_test: false,
            ..AdaptiveTrainingOptions::default()
        },
    )
    .expect("separable 2-knob train_adaptive_target");
    assert_eq!(rep.hadamard_diagonals_run, 1);

    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(&rep.proj_bytes, &crypto).expect("valid .dnam-proj");
    let inpc = verify_proj_inpc_capsule(view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap())
        .expect("valid INPC");
    let (dry_96k, _) = decode_proj_inpc_f32(&inpc);

    let had_entry = view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "hadamard_diag_0")
        .expect("hadamard_diag_0 stem present");
    let stmc = verify_proj_stmc_capsule(
        view.chunk_slice(ProjChunkKind::Stmc, had_entry.chunk_id)
            .unwrap(),
    )
    .expect("valid hadamard STMC");
    let (wet_96k, _) = decode_proj_stmc_f32(&stmc);
    assert_eq!(wet_96k.len(), CORESET_FP32_FRAMES_PER_STEM);

    // At c0 = c1, SeparableTwoKnobTarget reduces to (0.30 + 0.55*c)*x (scaled identity on A5!);
    // across the diagonal, wet_96k must strongly correlate with INPC dry_96k (would be ~0.0 if comb
    // frames were mistakenly sealed instead of the A5 coreset!).
    let dot_dw: f64 = dry_96k
        .iter()
        .zip(&wet_96k)
        .map(|(&d, &w)| f64::from(d) * f64::from(w))
        .sum();
    let ed: f64 = dry_96k.iter().map(|&d| f64::from(d).powi(2)).sum();
    let ew: f64 = wet_96k.iter().map(|&w| f64::from(w).powi(2)).sum();
    let coh = dot_dw / (ed * ew).sqrt().max(1.0e-18);
    assert!(
        coh > 0.75,
        "hadamard_diag_0 STMC coreset must match INPC A5 coreset (coh={coh:.3})"
    );
    assert!(
        stmc.windows.iter().any(|w| w.control_id == 0)
            && stmc.windows.iter().any(|w| w.control_id == 1),
        "Hadamard trajectory windows must record both coupled controls: {:?}",
        stmc.windows
    );
}

#[test]
fn test_zero_name_heuristic_delay_knob_probing_and_boutique_drive_label() {
    // verify[train.capture.invariants]
    assert_eq!(
        classify_knob_acoustic_class("boutique_drive"),
        KnobAcousticClass::ClassIGainDrive,
        "'boutique_drive' must not match 'out' substring as ClassIVSpatialMixLevel"
    );

    // Opaque knob id "k0" ("Parameter Alpha") with calibration=None controlling Chain08 BBD delay
    // time (40 ms vs 800 ms) MUST automatically select Mode B stationary stops via signal probing!
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    manifest.base.controls_schema = vec![make_range_ctrl(
        "k0",
        "Parameter Alpha",
        -1.0,
        1.0,
        -1.0,
        None,
    )];

    let mut target = Chain08OpaqueTarget::new();
    let rep = train_adaptive_target(
        manifest,
        &mut target,
        &AdaptiveTrainingOptions {
            prefer_mode_b: false,
            max_frames_per_stem: 2,
            run_live_null_test: false,
            ..AdaptiveTrainingOptions::default()
        },
    )
    .expect("zero-heuristic Chain08 delay training");

    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(&rep.proj_bytes, &crypto).expect("valid .dnam-proj");
    let stem_ids: Vec<&str> = view
        .manifest
        .stems_registry
        .iter()
        .map(|s| s.stem_id.as_str())
        .collect();
    assert!(
        stem_ids.contains(&"stop_ctrl_0_0") && stem_ids.contains(&"stop_ctrl_0_1"),
        "opaque delay knob 'k0' must select Mode B stationary stops, got stems {stem_ids:?}"
    );
}

fn build_freeze_latch_manifest() -> DnamProjManifest {
    let freeze_cal = Some(ControlCalibration {
        role: ControlCalibrationRole::FreezeLatch,
        lobatto_physical_anchors: None,
        conditional_physical_anchors: BTreeMap::new(),
        tempo_sync_default_subdivision: None,
    });
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    manifest.base.controls_schema = vec![
        make_range_ctrl("k0", "Mix", -1.0, 1.0, 0.0, None),
        make_switch_ctrl("freeze", &["Normal", "Frozen"], freeze_cal),
    ];
    manifest
}

#[test]
fn test_freeze_latch_transition_capture_mode2() {
    // verify[train.capture.planner.freeze-transition]
    let manifest = build_freeze_latch_manifest();
    let batch = generate_stimulus_batch(
        &manifest,
        &BatchStimulusOptions {
            prefer_mode_b: true,
            ..BatchStimulusOptions::default()
        },
    )
    .expect("Mode 2 batch with FreezeLatch");
    let freeze_stim = batch
        .stems
        .iter()
        .find(|s| s.stem_id == "freeze_transition_1")
        .expect("Mode 2 must schedule freeze_transition_1");
    let [n0, n1] = freeze_stim
        .freeze_transition_window
        .expect("freeze_transition_window populated");
    assert!(n0 > 0 && n1 > n0);

    let mut target = Chain08OpaqueTarget::new();
    let resps: Vec<StemResponse> = batch
        .stems
        .iter()
        .map(|s| {
            target.reset_state();
            target.render_stem(s).unwrap()
        })
        .collect();
    let solved = solve_from_responses(manifest, &[(&batch, &resps)], None)
        .expect("Mode 2 solve with FreezeLatch");
    let crypto = DefaultCryptoProvider::default();
    let v2 = DnamProjView::from_bytes(&solved.proj_bytes, &crypto).unwrap();
    let f_reg = v2
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "freeze_transition_1")
        .expect("freeze_transition_1 in Mode 2 registry");
    assert!(f_reg.user_metadata.contains_key("freeze_transition_window"));
}

#[test]
fn test_freeze_latch_transition_capture_mode1() {
    // verify[train.capture.planner.freeze-transition]
    let manifest = build_freeze_latch_manifest();
    let mut target = Chain08OpaqueTarget::new();
    let rep1 = train_adaptive_target(
        manifest,
        &mut target,
        &AdaptiveTrainingOptions {
            prefer_mode_b: true,
            max_frames_per_stem: 1,
            run_live_null_test: false,
            ..AdaptiveTrainingOptions::default()
        },
    )
    .expect("Mode 1 train with FreezeLatch");
    let crypto = DefaultCryptoProvider::default();
    let v1 = DnamProjView::from_bytes(&rep1.proj_bytes, &crypto).unwrap();
    assert!(
        v1.manifest
            .stems_registry
            .iter()
            .any(|s| s.stem_id == "freeze_transition_1"),
        "Mode 1 must schedule freeze_transition_1"
    );
}

#[test]
fn test_mode2_batch_96khz_switches_mode_b_controls_and_bulk_latency_alignment() {
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![
        make_range_ctrl("drive", "Drive", 0.0, 1.0, 0.5, None),
        make_range_ctrl("tone", "Tone", -1.0, 1.0, 0.0, None),
        make_switch_ctrl("voice", &["Clean", "Crunch", "Lead"], None),
    ];

    let batch = generate_stimulus_batch(
        &manifest,
        &BatchStimulusOptions {
            sample_rate_hz: 96_000,
            prefer_mode_b: true,
            ..BatchStimulusOptions::default()
        },
    )
    .expect("96 kHz Mode 2 batch");
    assert_eq!(batch.sample_rate_hz, 96_000);
    assert!(batch.stems.iter().all(|s| s.sample_rate_hz == 96_000));
    assert!(batch.stems.iter().any(|s| s.stem_id == "switch_2_opt_1"));
    assert!(batch.stems.iter().any(|s| s.stem_id == "switch_2_opt_2"));

    // Render at 96 kHz with +240 samples bulk latency (= +120.0 samples at 48 kHz!) vs 0-delay ref
    let mut od = Chain01SymmetricOd::with_format(96_000, dnam_fixtures::FixtureBitDepth::Float32);
    let resps = render_od_batch_96k(&mut od, &batch.stems, 240);
    let batch_0 = dnam_capture::StemStimulusBatch {
        sample_rate_hz: batch.sample_rate_hz,
        stems: batch
            .stems
            .iter()
            .filter(|s| s.stem_id == "star_center" || s.stem_id == "stop_ctrl_1_0")
            .cloned()
            .collect(),
    };
    let resps_0delay = render_od_batch_96k(&mut od, &batch_0.stems, 0);

    let solved = solve_from_responses(manifest.clone(), &[(&batch, &resps)], None)
        .expect("96 kHz solve_from_responses");
    assert_eq!(solved.stem_count, 7);
    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(&solved.proj_bytes, &crypto).unwrap();

    let stop1_reg = view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "stop_ctrl_1_0")
        .unwrap();
    let stop1_stmc = verify_proj_stmc_capsule(
        view.chunk_slice(ProjChunkKind::Stmc, stop1_reg.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let solved_0 = solve_from_responses(manifest, &[(&batch_0, &resps_0delay)], None).unwrap();
    let view_0 = DnamProjView::from_bytes(&solved_0.proj_bytes, &crypto).unwrap();
    let stop1_0_reg = view_0
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "stop_ctrl_1_0")
        .unwrap();
    let stop1_0_stmc = verify_proj_stmc_capsule(
        view_0
            .chunk_slice(ProjChunkKind::Stmc, stop1_0_reg.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let delta_lag_48k =
        stop1_stmc.fractional_lag_samples_f32 - stop1_0_stmc.fractional_lag_samples_f32;
    assert_eq!(stop1_stmc.windows[0].control_id, 1);
    assert!(
        (delta_lag_48k - 120.0).abs() <= 1.0,
        "expected +120.00 samples 48k lag delta from +240 samples @ 96k, got {delta_lag_48k}"
    );

    let sw2_reg = view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "switch_2_opt_2")
        .unwrap();
    assert_eq!(
        sw2_reg.controls_meta.get("voice"),
        Some(&serde_json::json!("Lead"))
    );
    let sw2_stmc = verify_proj_stmc_capsule(
        view.chunk_slice(ProjChunkKind::Stmc, sw2_reg.chunk_id)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(sw2_stmc.windows[0].control_id, 2);
    assert_eq!(sw2_stmc.windows[0].ctrl_kind, 1);
    assert_eq!(sw2_stmc.windows[0].active_opt, 2);
}

#[test]
fn test_simultaneous_long_echo_and_asymmetric_fuzz_capsule_clip_with_case_a_replay() {
    let mut fuzz = Chain02GatedAsymmetricFuzz::new();
    let mut bbd = Chain08LongBbdEchoAnd30sShimmer::new();
    let mut mic_clip = MicrophoneCapsuleClipDecorator::new(Chain01SymmetricOd::new());
    mic_clip.pos_ceiling_lin = 0.88; // Positive rail never clips
    mic_clip.neg_ceiling_lin = -0.26; // Single-rail sub-0 dBFS asymmetric negative FET clip!

    let mut ctrl = AdaptiveStimulusController::new(StimulusConfig {
        is_master_volume_sweep: false,
        max_frames: 6,
        max_tail_frames: 1,
        ..StimulusConfig::default()
    });
    let mut last_resp: Option<StemResponse> = None;
    let mut emitted_peaks = Vec::new();
    while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
        let pk = frame.in_1.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        emitted_peaks.push(pk);
        let mut clean_mix = Vec::with_capacity(frame.in_1.len());
        for &x in &frame.in_1 {
            let mut fz = [0.0_f32];
            fuzz.process_frame(&[3.5 * x], &mut fz, &[0.95, 0.8, 0.9], &[0]);
            let [el, _] = bbd.step_stereo(fz[0], fz[0], &[-0.7, 0.4, 0.6], false);
            clean_mix.push(0.85 * fz[0] + 0.45 * el);
        }
        let clipped = mic_clip.apply_capsule_and_xmax_fault(&clean_mix);
        last_resp = Some(mono_resp(
            format!("f_{}", frame.frame_index),
            48_000,
            clipped,
        ));
    }

    assert!(
        ctrl.warnings()
            .iter()
            .any(|w| matches!(w, CaptureWarning::MicrophoneCapsuleOrPreampClip { .. })),
        "single-rail asymmetric fuzz + echo capsule clip must trigger Law 0.F, got {:?}",
        ctrl.warnings()
    );
    assert!(
        emitted_peaks.len() >= 2 && emitted_peaks[1] < 0.82 * emitted_peaks[0],
        "Case A boost-clip replay (-3.0 dB = 0.7079x) must immediately step down Frame 1 peak: {emitted_peaks:?}"
    );
}

#[test]
fn test_extreme_8khz_and_384khz_multiport_4cm_and_2to2_with_replay_aux_preservation() {
    // 1. 8 kHz FourCableMethod4Cm with USB dropout on Frame 0 -> replay + Coreset must preserve in_3!
    let mut rig = Chain12FullEndToEndStereoRig::with_format(
        Chain12RigMode::FixedZeroKnob,
        8_000,
        dnam_fixtures::FixtureBitDepth::Float32,
    );
    let mut ctrl_8k = AdaptiveStimulusController::new(StimulusConfig {
        sample_rate_hz: 8_000,
        topology_mode: TopologyProbeMode::FourCableMethod4Cm,
        max_frames: 6,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut resp_8k: Option<StemResponse> = None;
    let mut injected_dropout = false;
    while let Some(fr) = ctrl_8k.next_frame(resp_8k.as_ref()) {
        assert_eq!(fr.sample_rate_hz, 8_000);
        assert_eq!(fr.in_1.len(), 768);
        assert!(
            fr.in_3.as_ref().is_some_and(|aux| aux.len() == 768),
            "4CM must populate in_3 on every frame (including replay & coreset) at 8 kHz"
        );
        let aux = fr.in_3.as_ref().unwrap();
        let (mut out_1, mut out_2) = (Vec::with_capacity(768), Vec::with_capacity(768));
        for i in 0..768 {
            let mut ob = [0.0_f32; 4];
            rig.process_frame(&[fr.in_1[i], aux[i]], &mut ob, &[0.5, 0.5, 0.5], &[]);
            out_1.push(ob[0]);
            out_2.push(ob[1]);
        }
        if !injected_dropout {
            injected_dropout = true;
            // Inject 24-sample USB zero-run dropout at 8 kHz (~144 samples @ 48 kHz >= 32)
            out_1[200..224].fill(0.0);
        }
        resp_8k = Some(stereo_resp(
            format!("f8k_{}", fr.frame_index),
            8_000,
            out_1,
            out_2,
        ));
    }
    assert!(
        ctrl_8k
            .warnings()
            .iter()
            .any(|w| matches!(w, CaptureWarning::ClockSlipOrUsbDropout { .. }))
    );
    let (inpc_8k, spec_8k) = ctrl_8k
        .seal_inpc_and_vault_spec(true, BTreeMap::new())
        .expect("8 kHz seal");
    assert_eq!(inpc_8k.len(), INPC_MONO_BYTES);
    assert_eq!(spec_8k.wet_coreset_fp32.len(), CORESET_FP32_STEREO_BYTES);

    // 2. 384 kHz TrueStereo2To2 (`36,864` samples per 96 ms frame)
    let mut ctrl_384k = AdaptiveStimulusController::new(StimulusConfig {
        sample_rate_hz: 384_000,
        topology_mode: TopologyProbeMode::TrueStereo2To2,
        max_frames: 2,
        max_tail_frames: 0,
        ..StimulusConfig::default()
    });
    let mut resp_384k: Option<StemResponse> = None;
    while let Some(fr) = ctrl_384k.next_frame(resp_384k.as_ref()) {
        assert_eq!(fr.sample_rate_hz, 384_000);
        assert_eq!(fr.in_1.len(), 36_864);
        let in_2 = fr
            .in_2
            .as_ref()
            .expect("2->2 must populate in_2 at 384 kHz");
        assert_eq!(in_2.len(), 36_864);
        let out_1: Vec<f32> = fr.in_1.iter().map(|&x| 0.65 * x).collect();
        let out_2: Vec<f32> = in_2.iter().map(|&x| 0.55 * x).collect();
        resp_384k = Some(stereo_resp(
            format!("f384k_{}", fr.frame_index),
            384_000,
            out_1,
            out_2,
        ));
    }
    let (inpc_384k, spec_384k) = ctrl_384k
        .seal_inpc_and_vault_spec(true, BTreeMap::new())
        .expect("384 kHz seal");
    assert_eq!(inpc_384k.len(), INPC_STEREO_BYTES);
    assert_eq!(spec_384k.wet_coreset_fp32.len(), CORESET_FP32_STEREO_BYTES);
}
