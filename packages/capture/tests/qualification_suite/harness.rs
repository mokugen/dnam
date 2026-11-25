#![forbid(unsafe_code)]
//! Strict 3-Level (`L1` / `L2` / `L3`) Universal Black-Box Phased Qualification Harness (`M25`).

#[path = "harness/eval.rs"]
mod eval;

pub(crate) use eval::{
    EvalMatrixConfig, FixtureQualificationReport, Level3SettingReport, assert_qualification_passed,
    evaluate_tier_outputs_4metric, log_level_summary, render_dnam_tiers,
};

use dnam_capture::metrics::certification::universal::certify_4metric_universal;
use dnam_capture::proj::STMC_FLAG_WET_POLARITY_INVERTED;
use dnam_capture::proj::bundle::{DnamProjView, export_runtime_dnam};
use dnam_capture::proj::capsules::{
    decode_proj_inpc_f32, decode_proj_stmc_f32, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::proj::directory::ProjChunkKind;
use dnam_capture::realign_fractional_kaiser64;
use dnam_capture::solver::{
    ComputeBackendPreference, SolverQualityMode, canonical_besr_warmup, resolve_eval_active_len,
    star_center_control_values,
};
use dnam_capture::{
    AdaptiveTrainingOptions, CORESET_FP32_CANONICAL_25S_FRAMES, CORESET_FP32_MIN_FRAMES_PER_STEM,
    DistillationTarget, StemStimulus, train_adaptive_target,
};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::BlackBoxModel;
use dnam_vm::math::DefaultMathBackend;

use super::adapter::{
    FixtureDistillationAdapter, build_parametric_manifest_for_model, build_star_manifest_for_model,
    infer_topology_mode, measure_empirical_repeatability_at_setting,
};
use super::excitations::{build_unseen_control_vectors, build_unseen_holdout_excitation_48k};

/// Extracted `StarCenter` subsample alignment parameters stored in `.dnam-proj`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StarAlignmentParams {
    pub lag_samples: f64,
    pub clock_drift_ppm: f64,
    pub invert_l: bool,
    pub invert_r: bool,
}

/// Execution configuration for 3-level (`L1` / `L2` / `L3`) fixture qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QualifyConfig {
    pub backend: ComputeBackendPreference,
    pub quality_mode: SolverQualityMode,
    pub max_frames_per_stem: usize,
    pub coreset_frames_48k: usize,
    pub run_live_null_test: bool,
    pub holdout_frames_48k: usize,
}

impl QualifyConfig {
    /// Full canonical `M25` qualification configuration (`BestUntilDiminishingReturns`, `120k` coreset).
    #[must_use]
    pub(crate) fn canonical(backend: ComputeBackendPreference) -> Self {
        Self {
            backend,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            max_frames_per_stem: 16,
            coreset_frames_48k: CORESET_FP32_CANONICAL_25S_FRAMES,
            run_live_null_test: true,
            holdout_frames_48k: 48_000,
        }
    }

    /// Fast `ThresholdStop` configuration used by non-ignored `M25-00` synthetic harness self-tests.
    #[must_use]
    pub(crate) fn fast_self_test(backend: ComputeBackendPreference) -> Self {
        Self {
            backend,
            quality_mode: SolverQualityMode::ThresholdStop {
                target_slack_milli_db: 0,
                max_polish_iters: 2,
            },
            max_frames_per_stem: 6,
            coreset_frames_48k: CORESET_FP32_MIN_FRAMES_PER_STEM,
            run_live_null_test: false,
            holdout_frames_48k: 16_384,
        }
    }

    fn into_training_options(
        self,
        topology_mode: dnam_capture::TopologyProbeMode,
    ) -> AdaptiveTrainingOptions {
        AdaptiveTrainingOptions {
            sample_rate_hz: 48_000,
            topology_mode,
            fixed_duration: false,
            prefer_mode_b: false,
            max_frames_per_stem: self.max_frames_per_stem,
            run_live_null_test: self.run_live_null_test,
            null_test_autopatch_threshold_db: -36.0,
            calb: None,
            use_spcm_tier1: false,
            coreset_frames_48k: self.coreset_frames_48k,
            quality_mode: self.quality_mode,
            backend_preference: self.backend,
        }
    }
}

/// Extracts `StarCenter` alignment parameters (`lag`, `ppm`, `invert_l`, `invert_r`) from `.dnam-proj`.
pub(crate) fn extract_star_alignment(proj_bytes: &[u8]) -> Result<StarAlignmentParams, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let proj_view = DnamProjView::from_bytes(proj_bytes, &crypto)?;
    let star_entry = proj_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .ok_or_else(|| {
            DnamError::SchemaValidationError("missing enabled StarCenter stem in manifest".into())
        })?;
    let star_stmc = verify_proj_stmc_capsule(
        proj_view
            .chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
            .ok_or_else(|| {
                DnamError::SchemaValidationError("missing StarCenter STMC chunk".into())
            })?,
    )?;
    let lag_samples = f64::from(star_stmc.fractional_lag_samples_f32);
    let invert_l = (star_stmc.flags & STMC_FLAG_WET_POLARITY_INVERTED) != 0;
    let invert_r = star_entry
        .user_metadata
        .get("wet_r_polarity_inverted")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(invert_l);
    let clock_drift_ppm = star_entry
        .user_metadata
        .get("clock_drift_ppm")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    Ok(StarAlignmentParams {
        lag_samples,
        clock_drift_ppm,
        invert_l,
        invert_r,
    })
}

fn align_live_response(
    out_1: &[f32],
    out_2: Option<&[f32]>,
    n_out: usize,
    align: StarAlignmentParams,
) -> Vec<Vec<f64>> {
    let mut channels = Vec::with_capacity(n_out.clamp(1, 2));
    let al_l = realign_fractional_kaiser64(
        out_1,
        align.lag_samples,
        align.clock_drift_ppm,
        align.invert_l,
    );
    channels.push(al_l.into_iter().map(f64::from).collect());
    if n_out >= 2 {
        let raw_r = out_2.unwrap_or(out_1);
        let al_r = realign_fractional_kaiser64(
            raw_r,
            align.lag_samples,
            align.clock_drift_ppm,
            align.invert_r,
        );
        channels.push(al_r.into_iter().map(f64::from).collect());
    }
    channels
}

fn check_star_center_zero_ulp_invariance(
    proj_bytes: &[u8],
    star_ctrls: &[f32],
    unseen_ctrls: &[f32],
    dry_l: &[f32],
    dry_r: Option<&[f32]>,
    n_out: usize,
) -> Result<bool, DnamError> {
    let mono_only = n_out < 2;
    let ultra_dnam = export_runtime_dnam(proj_bytes, Some(Tier::Ultra))?;
    let mut eng_ref =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&ultra_dnam, Tier::Ultra, 48_000)?;
    eng_ref.reset_to_dc_equilibrium();
    for (p, &v) in star_ctrls.iter().enumerate() {
        eng_ref.set_control_immediate(p, v);
    }
    let mut eng_test =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&ultra_dnam, Tier::Ultra, 48_000)?;
    for (p, &v) in unseen_ctrls.iter().enumerate() {
        eng_test.set_control_immediate(p, v);
    }
    eng_test.reset_to_dc_equilibrium();
    for (p, &v) in star_ctrls.iter().enumerate() {
        eng_test.set_control_immediate(p, v);
    }
    let probe_len = dry_l.len().min(256);
    for t in 0..probe_len {
        let dl = dry_l[t];
        let dr = dry_r.map_or(dl, |r| r[t]);
        let y_ref = eng_ref.step_frame_sona(dl, dr, mono_only);
        let y_got = eng_test.step_frame_sona(dl, dr, mono_only);
        if y_got[0].to_bits() != y_ref[0].to_bits()
            || (!mono_only && y_got[1].to_bits() != y_ref[1].to_bits())
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Runs the strict phased 3-level (`L1 -> L2 -> L3`) qualification pipeline for any [`BlackBoxModel`]
/// factory under the specified [`QualifyConfig`].
pub(crate) fn qualify_fixture_3level<M, F>(
    factory: F,
    config: QualifyConfig,
) -> FixtureQualificationReport
where
    M: BlackBoxModel,
    F: Fn() -> M,
{
    let crypto = DefaultCryptoProvider::default();
    let star_model = factory();
    let meta = star_model.metadata();
    let fixture_id = meta.id.to_string();
    let n_in = star_model.num_inputs().clamp(1, 2);
    let n_out = star_model.num_outputs().clamp(1, 2);
    let num_controls = meta.controls.len() + meta.switches.len();
    let topo = infer_topology_mode(&star_model);

    let star_manifest = build_star_manifest_for_model(&star_model);
    let mut star_adapter = FixtureDistillationAdapter::from_model(star_model);
    let opts = config.into_training_options(topo);
    let l1_train = train_adaptive_target(star_manifest, &mut star_adapter, &opts)
        .unwrap_or_else(|e| panic!("{fixture_id} L1 train_adaptive_target failed: {e:?}"));

    let proj_view = DnamProjView::from_bytes(&l1_train.proj_bytes, &crypto).unwrap();
    let star_align = extract_star_alignment(&l1_train.proj_bytes).unwrap();
    let star_ctrls: Vec<f32> = meta
        .controls
        .iter()
        .map(|c| c.default_norm)
        .chain(meta.switches.iter().map(|s| f32::from(s.default_index)))
        .collect();
    let star_profiles = measure_empirical_repeatability_at_setting(
        &mut star_adapter,
        topo,
        &star_ctrls,
        (
            star_align.lag_samples,
            star_align.clock_drift_ppm,
            star_align.invert_l,
            star_align.invert_r,
        ),
    );
    let is_passive_l1 = star_profiles
        .iter()
        .all(|p| p.band_inc_ratio.iter().all(|&r| r == 0.0));

    // Level 1 (In-Sample Fixed StarCenter)
    let inpc_view =
        verify_proj_inpc_capsule(proj_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
    let (inpc_l, inpc_r) = decode_proj_inpc_f32(&inpc_view);
    let star_entry = proj_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .unwrap();
    let stmc_view = verify_proj_stmc_capsule(
        proj_view
            .chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let (stmc_l, stmc_r) = decode_proj_stmc_f32(&stmc_view);
    let n_common = inpc_l.len().min(stmc_l.len());
    let dry_lr: Vec<[f32; 2]> = (0..n_common)
        .map(|i| [inpc_l[i], inpc_r.as_deref().map_or(inpc_l[i], |r| r[i])])
        .collect();
    let wet_lr: Vec<[f32; 2]> = (0..n_common)
        .map(|i| [stmc_l[i], stmc_r.as_deref().map_or(stmc_l[i], |r| r[i])])
        .collect();
    let n_eval_l1 = resolve_eval_active_len(&proj_view.manifest, &dry_lr, Some(&wet_lr), n_common);
    let warmup_l1 = canonical_besr_warmup(n_eval_l1);
    let mut l1_targets = vec![
        stmc_l[..n_eval_l1]
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<f64>>(),
    ];
    let mut l1_dry = vec![
        inpc_l[..n_eval_l1]
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<f64>>(),
    ];
    if n_out >= 2 {
        let r_sl = stmc_r.as_deref().unwrap_or(&stmc_l);
        l1_targets.push(r_sl[..n_eval_l1].iter().copied().map(f64::from).collect());
        let dry_r_sl = if n_in >= 2 {
            inpc_r.as_deref().unwrap_or(&inpc_l)
        } else {
            &inpc_l
        };
        l1_dry.push(
            dry_r_sl[..n_eval_l1]
                .iter()
                .copied()
                .map(f64::from)
                .collect(),
        );
    }
    let l1_render = render_dnam_tiers(
        &l1_train.proj_bytes,
        n_out,
        &inpc_l[..n_eval_l1],
        (n_in >= 2)
            .then(|| inpc_r.as_deref().map(|r| &r[..n_eval_l1]))
            .flatten(),
        &[],
        warmup_l1,
        is_passive_l1,
    )
    .unwrap();
    let level1 = evaluate_tier_outputs_4metric(
        EvalMatrixConfig {
            level_name: "L1",
            is_holdout_eval: false,
            require_target_floors: true,
            require_monotone: true,
            require_lufs_parity: true,
            sample_rate_hz: 48_000.0,
            warmup: warmup_l1,
            zero_dc_ok: l1_render.zero_dc_ok,
        },
        &l1_targets,
        Some(&l1_dry),
        &l1_render.models_by_channel,
        &star_profiles,
        &l1_render.lufs_by_channel,
    );
    log_level_summary(&fixture_id, &level1, "");

    // Level 2 (Out-of-Sample Untrained Signals at Fixed StarCenter; reuse exact L1 `.dnam`s)
    let (holdout_l, holdout_r) =
        build_unseen_holdout_excitation_48k(0x4D32_355F_4C32_0001, topo, config.holdout_frames_48k);
    let warmup_l2 = canonical_besr_warmup(holdout_l.len());
    let live_l2 = star_adapter
        .render_stem(&StemStimulus {
            stem_id: "qualify_l2_holdout".to_string(),
            sample_rate_hz: 48_000,
            in_1: holdout_l.clone(),
            in_2: holdout_r.clone(),
            in_3: None,
            in_4: None,
            control_values: star_ctrls.clone(),
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap();
    let l2_targets =
        align_live_response(&live_l2.out_1, live_l2.out_2.as_deref(), n_out, star_align);
    let mut l2_dry = vec![
        holdout_l
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<f64>>(),
    ];
    if n_out >= 2 {
        let dry_r_sl = if n_in >= 2 {
            holdout_r.as_deref().unwrap_or(&holdout_l)
        } else {
            &holdout_l
        };
        l2_dry.push(dry_r_sl.iter().copied().map(f64::from).collect());
    }
    let l2_render = render_dnam_tiers(
        &l1_train.proj_bytes,
        n_out,
        &holdout_l,
        holdout_r.as_deref(),
        &[],
        warmup_l2,
        false,
    )
    .unwrap();
    let level2 = evaluate_tier_outputs_4metric(
        EvalMatrixConfig {
            level_name: "L2",
            is_holdout_eval: true,
            require_target_floors: true,
            require_monotone: true,
            require_lufs_parity: false,
            sample_rate_hz: 48_000.0,
            warmup: warmup_l2,
            zero_dc_ok: true,
        },
        &l2_targets,
        Some(&l2_dry),
        &l2_render.models_by_channel,
        &star_profiles,
        &l2_render.lufs_by_channel,
    );
    log_level_summary(&fixture_id, &level2, "");

    // Level 3 (Out-of-Sample Untrained Signals at Unseen Intermediate Control Points; N/A for P == 0)
    let unseen_settings = build_unseen_control_vectors(&meta);
    let level3 = if unseen_settings.is_empty() {
        None
    } else {
        let param_model = factory();
        let param_manifest = build_parametric_manifest_for_model(&param_model);
        let l3_star_ctrls = star_center_control_values(&param_manifest);
        let mut param_adapter = FixtureDistillationAdapter::from_model(param_model);
        let l3_train = train_adaptive_target(param_manifest, &mut param_adapter, &opts)
            .unwrap_or_else(|e| panic!("{fixture_id} L3 train_adaptive_target failed: {e:?}"));
        let l3_align = extract_star_alignment(&l3_train.proj_bytes).unwrap();
        let l3_star_profiles = measure_empirical_repeatability_at_setting(
            &mut param_adapter,
            topo,
            &l3_star_ctrls,
            (
                l3_align.lag_samples,
                l3_align.clock_drift_ppm,
                l3_align.invert_l,
                l3_align.invert_r,
            ),
        );
        let (l3_dry_l, l3_dry_r) = build_unseen_holdout_excitation_48k(
            0x4D32_355F_4C33_0002,
            topo,
            config.holdout_frames_48k,
        );
        let mut l3_dry = vec![
            l3_dry_l
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<f64>>(),
        ];
        if n_out >= 2 {
            let dry_r_sl = if n_in >= 2 {
                l3_dry_r.as_deref().unwrap_or(&l3_dry_l)
            } else {
                &l3_dry_l
            };
            l3_dry.push(dry_r_sl.iter().copied().map(f64::from).collect());
        }
        let warmup_l3 = canonical_besr_warmup(l3_dry_l.len());
        let zero_ulp_ok = check_star_center_zero_ulp_invariance(
            &l3_train.proj_bytes,
            &l3_star_ctrls,
            &unseen_settings[0].control_values,
            &l3_dry_l,
            l3_dry_r.as_deref(),
            n_out,
        )
        .unwrap();

        let frozen_render = render_dnam_tiers(
            &l3_train.proj_bytes,
            n_out,
            &l3_dry_l,
            l3_dry_r.as_deref(),
            &l3_star_ctrls,
            warmup_l3,
            false,
        )
        .unwrap();

        let mut setting_reports = Vec::with_capacity(unseen_settings.len());
        for (idx, setting) in unseen_settings.iter().enumerate() {
            let live_l3 = param_adapter
                .render_stem(&StemStimulus {
                    stem_id: format!("qualify_l3_{}", setting.label),
                    sample_rate_hz: 48_000,
                    in_1: l3_dry_l.clone(),
                    in_2: l3_dry_r.clone(),
                    in_3: None,
                    in_4: None,
                    control_values: setting.control_values.clone(),
                    control_trajectories: None,
                    freeze_transition_window: None,
                })
                .unwrap();
            let l3_targets =
                align_live_response(&live_l3.out_1, live_l3.out_2.as_deref(), n_out, l3_align);
            let prof_for_setting = if setting.flips_switch {
                measure_empirical_repeatability_at_setting(
                    &mut param_adapter,
                    topo,
                    &setting.control_values,
                    (
                        l3_align.lag_samples,
                        l3_align.clock_drift_ppm,
                        l3_align.invert_l,
                        l3_align.invert_r,
                    ),
                )
            } else {
                l3_star_profiles
            };
            let frozen_ultra = certify_4metric_universal(
                Tier::Ultra,
                &l3_targets[0],
                &frozen_render.models_by_channel[0][3],
                &prof_for_setting[0],
                48_000.0,
                warmup_l3,
            )
            .besr_db;
            let l3_render = render_dnam_tiers(
                &l3_train.proj_bytes,
                n_out,
                &l3_dry_l,
                l3_dry_r.as_deref(),
                &setting.control_values,
                warmup_l3,
                false,
            )
            .unwrap();
            let check_zero_ulp = idx == 0;
            let mut eval = evaluate_tier_outputs_4metric(
                EvalMatrixConfig {
                    level_name: &format!("L3:{}", setting.label),
                    is_holdout_eval: true,
                    require_target_floors: false,
                    require_monotone: true,
                    require_lufs_parity: false,
                    sample_rate_hz: 48_000.0,
                    warmup: warmup_l3,
                    zero_dc_ok: !check_zero_ulp || zero_ulp_ok,
                },
                &l3_targets,
                Some(&l3_dry),
                &l3_render.models_by_channel,
                &prof_for_setting,
                &l3_render.lufs_by_channel,
            );
            if check_zero_ulp && !zero_ulp_ok {
                eval.failure_reasons
                    .push("L3 StarCenter 0-ULP invariance failed after control reset".into());
                eval.passes = false;
            }
            log_level_summary(&fixture_id, &eval, &format!(" frzU={frozen_ultra:.2}dB"));
            setting_reports.push(Level3SettingReport {
                label: setting.label.clone(),
                frozen_ultra_besr_db: frozen_ultra,
                eval,
            });
        }
        Some(setting_reports)
    };

    let l3_pass = level3
        .as_ref()
        .is_none_or(|rows| rows.iter().all(|r| r.eval.passes));
    let overall_pass = level1.passes && level2.passes && l3_pass;
    FixtureQualificationReport {
        fixture_id,
        num_controls,
        backend: config.backend,
        level1,
        level2,
        level3,
        overall_pass,
    }
}
