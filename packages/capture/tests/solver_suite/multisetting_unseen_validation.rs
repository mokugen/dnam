#![forbid(unsafe_code)]
//! Anti-Overfitting Black-Box & Multi-Setting Unrecorded-Control Validation Suite
//! (`solver_suite/multisetting_unseen_validation.rs`).
//!
//! Validates two anti-overfitting properties:
//! 1. Blind `v3` identification succeeds when chains are configured at diverse non-default settings
//!    (shifted crossover frequencies, switch voicings, gain/EQ trims).
//! 2. After multi-setting capture (`train_adaptive_target`), exported `.dnam` models running via
//!    `DnamEngine` at unrecorded intermediate multi-knob/switch settings track the actual
//!    `BlackBoxModel` chain rendered at those exact unrecorded settings, outperforming frozen
//!    `StarCenter` while preserving `0-ULP` `StarCenter` invariance.

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use dnam_capture::metrics::{
    TargetRepeatabilityProfile, canonical_ltas_33_centers_hz, certify_4metric_universal,
    verify_monotone_4tier_summaries,
};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkKind, STMC_FLAG_WET_POLARITY_INVERTED,
    export_runtime_dnam, verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    build_v3_causal_probe_excitation, canonical_besr_warmup,
    estimate_v3_causal_deadtime_and_polarity, star_center_control_values,
};
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, realign_fractional_kaiser64,
    resample_arbitrary_rate, resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain13TriBandVcaCompressor, Chain14TriBandCompDistortion,
    Chain15TubeAmpHeadDi,
};
use dnam_train::control::StageStabilityConstraint;
use dnam_train::{
    Pass5EmitConfig, SonaSolverConfig, SonaTrainDataset, TrainRunner, solve_sona_model,
};
use dnam_vm::SonaVm;
use dnam_vm::math::{DefaultMathBackend, ScalarBackend};

struct ValidationAdapter<M: BlackBoxModel> {
    inner: M,
}

impl<M: BlackBoxModel> ValidationAdapter<M> {
    fn from_model(inner: M) -> Self {
        Self { inner }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_internal(
        &mut self,
        stem_id: String,
        stim_sr: u32,
        in_1: &[f32],
        in_2: Option<&[f32]>,
        in_3: Option<&[f32]>,
        in_4: Option<&[f32]>,
        control_values: &[f32],
        control_trajectories: Option<&Vec<Vec<f32>>>,
        freeze_window: Option<[usize; 2]>,
    ) -> Result<StemResponse, DnamError> {
        let dev_sr = self.inner.sample_rate_hz();
        let orig_n = in_1.len();
        if orig_n == 0 {
            let n_out = self.inner.num_outputs().clamp(1, 4);
            return Ok(StemResponse {
                stem_id,
                sample_rate_hz: stim_sr,
                out_1: Vec::new(),
                out_2: (n_out >= 2).then(Vec::new),
                out_3: (n_out >= 3).then(Vec::new),
                out_4: (n_out >= 4).then(Vec::new),
                internal_nodes: BTreeMap::new(),
            });
        }
        let needs_resample = stim_sr != dev_sr;
        let dev_n = if needs_resample {
            resampled_frame_len(orig_n, stim_sr, dev_sr)
        } else {
            orig_n
        };
        let (rs1, rs2, rs3, rs4, rstraj);
        let (dev_in1, dev_in2, dev_in3, dev_in4, dev_trajs, dev_freeze) = if needs_resample {
            rs1 = resample_arbitrary_rate(in_1, stim_sr, dev_sr, Some(dev_n));
            rs2 = in_2.map(|c| resample_arbitrary_rate(c, stim_sr, dev_sr, Some(dev_n)));
            rs3 = in_3.map(|c| resample_arbitrary_rate(c, stim_sr, dev_sr, Some(dev_n)));
            rs4 = in_4.map(|c| resample_arbitrary_rate(c, stim_sr, dev_sr, Some(dev_n)));
            rstraj = control_trajectories.map(|ts| {
                ts.iter()
                    .map(|tr| resample_control_trajectory(tr, dev_n))
                    .collect::<Vec<Vec<f32>>>()
            });
            (
                rs1.as_slice(),
                rs2.as_deref(),
                rs3.as_deref(),
                rs4.as_deref(),
                rstraj.as_ref(),
                freeze_window.map(|w| scale_freeze_window(w, orig_n, dev_n)),
            )
        } else {
            (in_1, in_2, in_3, in_4, control_trajectories, freeze_window)
        };

        let meta = self.inner.metadata();
        let default_controls = meta.default_controls();
        let n_range = meta.controls.len();
        let n_sw = meta.switches.len();
        let freeze_sw_idx = meta
            .switches
            .iter()
            .position(|sw| sw.id == "freeze")
            .unwrap_or(0);
        let mut ctrl_scratch = vec![0.0_f32; n_range];
        let mut sw_scratch = vec![0_u8; n_sw];
        let mut dev_out1 = vec![0.0_f32; dev_n];
        let n_in = self.inner.num_inputs().clamp(1, 4);
        let n_out = self.inner.num_outputs().clamp(1, 4);
        let mut dev_out2 = (n_out >= 2).then(|| vec![0.0_f32; dev_n]);

        for i in 0..dev_n {
            let x1 = dev_in1[i];
            let x2 = dev_in2.and_then(|v| v.get(i).copied()).unwrap_or(x1);
            let x3 = dev_in3.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            let x4 = dev_in4.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            for p in 0..n_range {
                let fb = control_values
                    .get(p)
                    .copied()
                    .or_else(|| default_controls.get(p).copied())
                    .unwrap_or(0.0);
                ctrl_scratch[p] = dev_trajs
                    .and_then(|t| t.get(p))
                    .and_then(|t| t.get(i).copied())
                    .unwrap_or(fb);
            }
            for k in 0..n_sw {
                let fb_sw = meta.switches[k].default_index;
                let max_opt = meta.switches[k].options.len().saturating_sub(1);
                sw_scratch[k] = dev_trajs
                    .and_then(|t| t.get(n_range + k))
                    .and_then(|t| t.get(i).copied())
                    .or_else(|| control_values.get(n_range + k).copied())
                    .map_or(fb_sw, |v| (v.round().max(0.0) as usize).min(max_opt) as u8);
            }
            if let Some([fs, fe]) = dev_freeze
                && let Some(slot) = sw_scratch.get_mut(freeze_sw_idx)
            {
                *slot = if i >= fs && i < fe {
                    1
                } else {
                    meta.switches
                        .get(freeze_sw_idx)
                        .map_or(0, |s| s.default_index)
                };
            }
            let in_f = [x1, x2, x3, x4];
            let mut y = [0.0_f32; 4];
            self.inner
                .process_frame(&in_f[..n_in], &mut y[..n_out], &ctrl_scratch, &sw_scratch);
            dev_out1[i] = y[0];
            if let Some(ref mut c2) = dev_out2 {
                c2[i] = y[1];
            }
        }
        let (out_1, out_2) = if needs_resample {
            (
                resample_arbitrary_rate(&dev_out1, dev_sr, stim_sr, Some(orig_n)),
                dev_out2.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
            )
        } else {
            (dev_out1, dev_out2)
        };
        Ok(StemResponse {
            stem_id,
            sample_rate_hz: stim_sr,
            out_1,
            out_2,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
    }
}

impl<M: BlackBoxModel> DistillationTarget for ValidationAdapter<M> {
    fn reset_state(&mut self) {
        self.inner.reset();
    }
    fn render_stem(&mut self, s: &StemStimulus) -> Result<StemResponse, DnamError> {
        self.reset_state();
        self.render_internal(
            s.stem_id.clone(),
            s.sample_rate_hz,
            &s.in_1,
            s.in_2.as_deref(),
            s.in_3.as_deref(),
            s.in_4.as_deref(),
            &s.control_values,
            s.control_trajectories.as_ref(),
            s.freeze_transition_window,
        )
    }
    fn render_frame(&mut self, f: &StimulusFrame) -> Result<StemResponse, DnamError> {
        self.render_internal(
            format!("frame_{}", f.frame_index),
            f.sample_rate_hz,
            &f.in_1,
            f.in_2.as_deref(),
            f.in_3.as_deref(),
            f.in_4.as_deref(),
            &f.control_values,
            f.control_trajectories.as_ref(),
            None,
        )
    }
}

fn manifest_for_model<M: BlackBoxModel>(model: &M, tag: &str) -> DnamProjManifest {
    let meta = model.metadata();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.metadata.name = format!("{} {tag}", meta.name);
    manifest.base.io_config.inputs = model.num_inputs().clamp(1, 2) as u8;
    manifest.base.io_config.output_mode = if model.num_outputs() >= 2 {
        "stereo".to_string()
    } else {
        "mono".to_string()
    };
    let mut schema = Vec::new();
    for c in meta.controls {
        let def: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": c.id,
            "name": c.name,
            "type": "range",
            "lobatto_degree": 3,
            "declared_range": [-1.0, 1.0],
            "safe_clamped_range": [-1.0, 1.0],
            "default": c.default_norm
        }))
        .unwrap();
        schema.push(def);
    }
    for sw in meta.switches {
        let def: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": sw.id,
            "name": sw.name,
            "type": "switch",
            "declared_options": sw.options,
            "active_options": sw.options,
            "default": sw.options[usize::from(sw.default_index)]
        }))
        .unwrap();
        schema.push(def);
    }
    manifest.base.controls_schema = schema;
    manifest
}

fn build_broadband_dynamic_excitation(n: usize, sr_hz: f64) -> Vec<f32> {
    let centers = canonical_ltas_33_centers_hz();
    let scale = 0.36 / (centers.len() as f64).sqrt();
    (0..n)
        .map(|i| {
            let t = (i as f64) / sr_hz;
            let burst = 0.25 + 0.75 * (-((i % 2_400) as f64) / 720.0).exp();
            let swell = 0.45 + 0.55 * (0.5 - 0.5 * (TAU * 1.9 * t).cos());
            let mut acc = 0.0_f64;
            for (k, &fc) in centers.iter().enumerate() {
                let phi = ((k as f64 + 1.0) * std::f64::consts::GOLDEN_RATIO).fract() * TAU;
                acc += (TAU * fc * t + phi).cos();
            }
            (burst * swell * scale * acc) as f32
        })
        .collect()
}

fn broadband_esr_db(tgt: &[f64], mdl: &[f64], warmup: usize) -> f64 {
    let n = tgt.len().min(mdl.len());
    let mut sig = 0.0_f64;
    let mut err = 0.0_f64;
    for i in warmup..n {
        let d = tgt[i] - mdl[i];
        sig += tgt[i] * tgt[i];
        err += d * d;
    }
    10.0 * (err / sig.max(1.0e-18)).max(1.0e-12).log10()
}

fn verify_blind_nondefault_fit<M: BlackBoxModel>(
    label: &str,
    mut model: M,
    controls: &[f32],
    switches: &[u8],
    min_ultra_besr_db: f64,
) {
    let dry = build_broadband_dynamic_excitation(24_576, 48_000.0);
    model.reset();
    let mut raw_wet = vec![0.0_f32; dry.len()];
    let mut y = [0.0_f32; 2];
    let n_out = model.num_outputs().clamp(1, 2);
    for (i, &x) in dry.iter().enumerate() {
        model.process_frame(&[x], &mut y[..n_out], controls, switches);
        raw_wet[i] = y[0];
    }
    let probe = build_v3_causal_probe_excitation();
    let probe_l: Vec<f32> = probe.iter().map(|v| v[0]).collect();
    model.reset();
    let mut probe_wet = vec![0.0_f32; probe_l.len()];
    for (i, &x) in probe_l.iter().enumerate() {
        model.process_frame(&[x], &mut y[..n_out], controls, switches);
        probe_wet[i] = y[0];
    }
    let (lag, inv) = estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_wet);
    let tgt = realign_fractional_kaiser64(&raw_wet, lag, 0.0, inv);
    let warmup = canonical_besr_warmup(dry.len());
    let dry64: Vec<f64> = dry.iter().map(|&v| f64::from(v)).collect();
    let tgt64: Vec<f64> = tgt.iter().map(|&v| f64::from(v)).collect();
    let ds = SonaTrainDataset {
        sample_rate_hz: 48_000.0,
        warmup_frames: warmup,
        input_l: dry64.clone(),
        input_r: dry64,
        target_l: tgt64.clone(),
        target_r: tgt64.clone(),
        controls: Vec::new(),
        stability_constraint: StageStabilityConstraint::default(),
        control_samples: Vec::new(),
        control_audio_stems: Vec::new(),
        tsyn_entries: Vec::new(),
    };
    let mut cfg = SonaSolverConfig::default();
    cfg.pass2.sample_rate_hz = 48_000.0;
    cfg.pass2.warmup_frames = warmup;
    cfg.pass2.max_epochs = 8;
    cfg.pass5 = Pass5EmitConfig {
        stereo_out: false,
        single_input: true,
        ..Pass5EmitConfig::default()
    };
    let fitted = solve_sona_model(&TrainRunner::cpu(), &ds, &cfg, None)
        .unwrap_or_else(|e| panic!("{label} solve_sona_model failed: {e:?}"));
    let prof = TargetRepeatabilityProfile::deterministic();
    let mut sums = Vec::with_capacity(4);
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let mut vm = SonaVm::<ScalarBackend>::open_dnam_bin_owned(
            &fitted.pass5.stripped_tiers[t_idx],
            tier,
            48_000.0,
        )
        .unwrap();
        let (mut ol, mut or_) = (vec![0.0_f32; dry.len()], vec![0.0_f32; dry.len()]);
        vm.process_block(&dry, &dry, &mut ol, &mut or_).unwrap();
        let mdl64: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
        sums.push(certify_4metric_universal(
            tier, &tgt64, &mdl64, &prof, 48_000.0, warmup,
        ));
    }
    let arr: [_; 4] = sums.try_into().unwrap();
    assert!(
        verify_monotone_4tier_summaries(&arr),
        "{label} blind non-default fit failed 4-tier monotonicity: {arr:#?}"
    );
    assert!(
        arr[3].besr_db <= min_ultra_besr_db,
        "{label} blind non-default Ultra bESR {:.2} dB missed {min_ultra_besr_db:.2} dB",
        arr[3].besr_db
    );
    println!(
        "[BLIND-NONDEFAULT] {label}: bESR=[{:.2}, {:.2}, {:.2}, {:.2}] LTAS(Ultra)={:.3} dB",
        arr[0].besr_db, arr[1].besr_db, arr[2].besr_db, arr[3].besr_db, arr[3].ltas_max_db
    );
}

#[test]
#[ignore = "legacy fixture qualification; superseded by phased per-fixture qualification in M25"]
fn test_blind_identification_at_diverse_non_default_chain_settings() {
    // Non-default Chain-13: 135 Hz / 3,100 Hz crossovers + asymmetric ratios (8:1 / 2:1 / 12:1)
    let norm = |phys: f32, lo: f32, hi: f32| 2.0 * (phys - lo) / (hi - lo) - 1.0;
    verify_blind_nondefault_fit(
        "Chain-13 NonDefault (135Hz/3.1kHz)",
        Chain13TriBandVcaCompressor::new(),
        &[
            norm(135.0, 50.0, 800.0),
            norm(3_100.0, 800.0, 6_000.0),
            -0.35,
            0.25,
            0.20,
            -0.20,
            -0.25,
            0.30,
        ],
        &[2, 0, 3],
        -12.0,
    );

    // Non-default Chain-14: 150 Hz / 2,200 Hz / 11 kHz + B3K mid / Alpha Turbo high voicings
    let lo14 = norm(150.0, 60.0, 600.0);
    let mid14 = norm(2_200.0, 400.0, 3_600.0);
    let hi14 = norm(11_000.0, 2_000.0, 16_000.0);
    verify_blind_nondefault_fit(
        "Chain-14 NonDefault (150Hz/2.2kHz/11kHz B3K+AlphaT)",
        Chain14TriBandCompDistortion::new(),
        &[
            lo14, lo14, mid14, mid14, hi14, 0.35, 0.20, 0.45, -0.15, -0.35, 0.25,
        ],
        &[1, 2],
        -12.0,
    );

    // Non-default Chain-15: Lead switch + high gain + scooped mid + elevated treble/presence
    verify_blind_nondefault_fit(
        "Chain-15 NonDefault (Lead Scoop)",
        Chain15TubeAmpHeadDi::new(),
        &[0.45, -0.30, -0.55, 0.40, 0.35],
        &[1],
        -10.0,
    );
}

fn validate_unrecorded_settings<M: BlackBoxModel>(
    label: &str,
    model: M,
    unrecorded_settings: &[(&str, Vec<f32>)],
) {
    let manifest = manifest_for_model(&model, "MultiSetting");
    let mut adapter = ValidationAdapter::from_model(model);
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: TopologyProbeMode::Mono1To1,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let report = train_adaptive_target(manifest.clone(), &mut adapter, &opts)
        .unwrap_or_else(|e| panic!("{label} train_adaptive_target failed: {e:?}"));

    let star_ctrls = star_center_control_values(&manifest);
    let eval_dry = build_broadband_dynamic_excitation(16_384, 48_000.0);
    let warmup = canonical_besr_warmup(eval_dry.len());
    let crypto = DefaultCryptoProvider::default();
    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto).unwrap();
    let star_entry = proj_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .unwrap();
    let star_stmc = verify_proj_stmc_capsule(
        proj_view
            .chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let star_lag = f64::from(star_stmc.fractional_lag_samples_f32);
    let star_inv = (star_stmc.flags & STMC_FLAG_WET_POLARITY_INVERTED) != 0;

    let mut ultra_dnam = export_runtime_dnam(&report.proj_bytes, Some(Tier::Ultra)).unwrap();
    let lite_dnam = export_runtime_dnam(&report.proj_bytes, Some(Tier::Lite)).unwrap();

    // Check 0-ULP StarCenter invariance across control moves and reset back to StarCenter
    let mut eng_ref =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&ultra_dnam, Tier::Ultra, 48_000)
            .unwrap();
    eng_ref.reset_to_dc_equilibrium();
    for (p, &v) in star_ctrls.iter().enumerate() {
        eng_ref.set_control_immediate(p, v);
    }
    let y_star_ref: Vec<[f32; 2]> = eval_dry[..256]
        .iter()
        .map(|&x| eng_ref.step_frame_sona(x, x, true))
        .collect();

    let mut eng_u =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&ultra_dnam, Tier::Ultra, 48_000)
            .unwrap();
    let mut eng_l =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&lite_dnam, Tier::Lite, 48_000).unwrap();
    // Perturb away then snap back to StarCenter and assert bit-exact output
    for (p, &v) in unrecorded_settings[0].1.iter().enumerate() {
        eng_u.set_control_immediate(p, v);
    }
    eng_u.reset_to_dc_equilibrium();
    for (p, &v) in star_ctrls.iter().enumerate() {
        eng_u.set_control_immediate(p, v);
    }
    for (t, &x) in eval_dry[..256].iter().enumerate() {
        let got = eng_u.step_frame_sona(x, x, true)[0];
        assert_eq!(
            got.to_bits(),
            y_star_ref[t][0].to_bits(),
            "{label} StarCenter repeatability drifted at sample {t}"
        );
    }
    ultra_dnam.clear();

    let prof = TargetRepeatabilityProfile::deterministic();
    let mut failures = Vec::new();
    for (tag, u_unrec) in unrecorded_settings {
        assert_eq!(u_unrec.len(), star_ctrls.len());
        let actual_resp = adapter
            .render_stem(&StemStimulus {
                stem_id: format!("actual_{tag}"),
                sample_rate_hz: 48_000,
                in_1: eval_dry.clone(),
                in_2: None,
                in_3: None,
                in_4: None,
                control_values: u_unrec.clone(),
                control_trajectories: None,
                freeze_transition_window: None,
            })
            .unwrap();
        let actual_aligned: Vec<f64> =
            realign_fractional_kaiser64(&actual_resp.out_1, star_lag, 0.0, star_inv)
                .into_iter()
                .map(f64::from)
                .collect();

        let e_act: f64 = actual_aligned[warmup..].iter().map(|v| v * v).sum();
        for (t_name, tier, eng) in [
            ("Lite", Tier::Lite, &mut eng_l),
            ("Ultra", Tier::Ultra, &mut eng_u),
        ] {
            // Render frozen StarCenter baseline on the same tier as unrecorded evaluation
            eng.reset_to_dc_equilibrium();
            for (p, &v) in star_ctrls.iter().enumerate() {
                eng.set_control_immediate(p, v);
            }
            let frozen_mdl: Vec<f64> = eval_dry
                .iter()
                .map(|&x| f64::from(eng.step_frame_sona(x, x, true)[0]))
                .collect();
            let esr_frozen = broadband_esr_db(&actual_aligned, &frozen_mdl, warmup);
            let frozen_sum = certify_4metric_universal(
                tier,
                &actual_aligned,
                &frozen_mdl,
                &prof,
                48_000.0,
                warmup,
            );
            let e_frz: f64 = frozen_mdl[warmup..].iter().map(|v| v * v).sum();
            let frozen_rms_diff_db = (10.0 * ((e_frz + 1.0e-15) / (e_act + 1.0e-15)).log10()).abs();

            eng.reset_to_dc_equilibrium();
            for (p, &v) in u_unrec.iter().enumerate() {
                eng.set_control_immediate(p, v);
            }
            let unrec_mdl: Vec<f64> = eval_dry
                .iter()
                .map(|&x| f64::from(eng.step_frame_sona(x, x, true)[0]))
                .collect();
            let esr_unrec = broadband_esr_db(&actual_aligned, &unrec_mdl, warmup);
            let sum = certify_4metric_universal(
                tier,
                &actual_aligned,
                &unrec_mdl,
                &prof,
                48_000.0,
                warmup,
            );
            assert!(
                sum.besr_db.is_finite() && sum.ltas_max_db.is_finite(),
                "{label} {tag} {t_name} produced non-finite metrics against actual chain"
            );
            let e_mdl: f64 = unrec_mdl[warmup..].iter().map(|v| v * v).sum();
            let rms_diff_db = (10.0 * ((e_mdl + 1.0e-15) / (e_act + 1.0e-15)).log10()).abs();
            println!(
                "[UNREC-VAL] {label} [{tag}] {t_name}: Unrec ESR={esr_unrec:.2} dB vs Frozen ESR={esr_frozen:.2} dB | bESR={:.2} vs {:.2} dB | dRMS={rms_diff_db:.2} (Frz {frozen_rms_diff_db:.2}) dB | Unrec LTAS={:.3} vs Frozen LTAS={:.3} dB | Unrec MR={:.3} vs Frozen MR={:.3}",
                sum.besr_db,
                frozen_sum.besr_db,
                sum.ltas_max_db,
                frozen_sum.ltas_max_db,
                sum.mr_iir,
                frozen_sum.mr_iir
            );
            if rms_diff_db > 6.0 {
                failures.push(format!(
                    "{label} [{tag}] {t_name} broadband RMS delta exceeded 6.0 dB ceiling: {rms_diff_db:.2} dB"
                ));
            }
            let improves = (esr_unrec + 0.15 < esr_frozen)
                || (sum.besr_db + 0.15 < frozen_sum.besr_db)
                || (sum.mr_iir + 0.005 < frozen_sum.mr_iir
                    && rms_diff_db <= frozen_rms_diff_db + 0.5);
            if !improves {
                failures.push(format!(
                    "{label} [{tag}] {t_name} at unrecorded setting did not improve over frozen StarCenter vs actual chain! (ESR {esr_unrec:.2} vs {esr_frozen:.2} dB, bESR {:.2} vs {:.2} dB, dRMS {rms_diff_db:.2} vs {frozen_rms_diff_db:.2} dB, LTAS {:.3} vs {:.3} dB, MR {:.3} vs {:.3})",
                    sum.besr_db,
                    frozen_sum.besr_db,
                    sum.ltas_max_db,
                    frozen_sum.ltas_max_db,
                    sum.mr_iir,
                    frozen_sum.mr_iir
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{label} unrecorded validation failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "legacy fixture qualification; superseded by phased per-fixture qualification in M25"]
fn test_multisetting_capture_validates_unrecorded_settings_against_actual_chains() {
    validate_unrecorded_settings(
        "Chain-01 SymmetricOD",
        Chain01SymmetricOd::new(),
        &[
            ("offaxis_hi_tone", vec![-0.42, 0.37, 0.25]),
            ("offaxis_hot_dark", vec![0.48, -0.35, -0.22]),
        ],
    );
    validate_unrecorded_settings(
        "Chain-13 TriBandVcaComp",
        Chain13TriBandVcaCompressor::new(),
        &[
            (
                "offaxis_sculpt_ratios",
                vec![
                    -0.45, 0.35, -0.30, 0.35, -0.25, -0.20, 0.30, 0.25, 2.0, 0.0, 2.0,
                ],
            ),
            (
                "offaxis_alt_thresh",
                vec![
                    0.35, -0.35, 0.25, -0.35, 0.20, 0.32, -0.30, -0.25, 0.0, 3.0, 0.0,
                ],
            ),
        ],
    );
    validate_unrecorded_settings(
        "Chain-14 TriBandEntropia",
        Chain14TriBandCompDistortion::new(),
        &[
            (
                "offaxis_b3k_alphat",
                vec![
                    -0.40, -0.40, 0.30, 0.30, 0.25, 0.30, 0.22, 0.42, -0.18, -0.28, 0.24, 1.0, 2.0,
                ],
            ),
            (
                "offaxis_alpha_vmt",
                vec![
                    0.32, 0.32, -0.35, -0.35, -0.25, -0.25, -0.20, -0.30, 0.28, 0.38, -0.22, 2.0,
                    1.0,
                ],
            ),
        ],
    );
    validate_unrecorded_settings(
        "Chain-15 TubeAmpHeadDI",
        Chain15TubeAmpHeadDi::new(),
        &[
            (
                "offaxis_lead_scoop",
                vec![0.40, -0.28, -0.45, 0.35, 0.30, 1.0],
            ),
            (
                "offaxis_crunch_warm",
                vec![-0.38, 0.35, 0.25, -0.32, -0.25, 0.0],
            ),
        ],
    );
}
