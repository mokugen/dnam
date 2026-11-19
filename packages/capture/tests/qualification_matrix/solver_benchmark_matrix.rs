#![forbid(unsafe_code)]
//! Universal `14`-Chain × `4`-Tier Perceptual Benchmark Matrix (`r[qual.tiers.benchmark-matrix]`).
//!
//! Evaluates all `14` primary single-device / `Amp+Cab` fixtures (`Chain-01..11, 13, 14, 15`)
//! across `[Lite, Full, Pro, Ultra]` using label-free empirical repeatability profiling
//! (`r[qual.besr.noise-subtracted]`, `r[qual.eval.metric-scope]`) with **zero** `ArchetypeFamily`
//! quality carve-outs, plus an optional `DNAM_SMOKE_CHAIN12=1`-gated complex multi-FX smoke rig.

use dnam_capture::metrics::{
    PerceptualCertificationSummary, TargetRepeatabilityProfile, TierQualityThresholds,
    build_repeatability_probe_stream, certify_4metric_universal, certify_holdout_universal,
    verify_cross_tier_loudness_parity, verify_monotone_4tier_summaries,
};
use dnam_capture::proj::{
    DnamProjView, ProjChunkKind, export_runtime_dnam, verify_proj_inpc_capsule,
    verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    build_v3_causal_probe_excitation, canonical_besr_warmup,
    estimate_v3_causal_deadtime_and_polarity, star_center_control_values,
};
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_capture::{DistillationTarget, StemResponse, StemStimulus, realign_fractional_kaiser64};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::BlackBoxModel;
use dnam_fixtures::chains::{
    Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain03ParallelCleanBlend,
    Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom, Chain06EnvelopeResonantSvf,
    Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer, Chain09La2aOpticalCompressor,
    Chain10Oc2OctaveDividerAndRingMod, Chain11BiAmpBassAndOutputIron, Chain12FullEndToEndStereoRig,
    Chain12RigMode, Chain13TriBandVcaCompressor, Chain14TriBandCompDistortion,
    Chain15TubeAmpHeadDi,
};

use super::harness::{
    MatrixAdapter, build_fullband_eval_excitation, build_unseen_holdout_excitation,
    combine_stereo_summary, manifest_for_fixture,
};

#[derive(Clone, Debug)]
struct RowResult {
    label: String,
    tier: Tier,
    summary: PerceptualCertificationSummary,
    holdout_besr_db: f64,
    lu_step: f64,
    lu_pass: bool,
    monotone: bool,
    holdout_pass: bool,
}

fn filter_matches(label: &str) -> bool {
    let Ok(filter) = std::env::var("DNAM_BENCH_FILTER") else {
        return true;
    };
    if filter.trim().is_empty() {
        return true;
    }
    filter.split(',').any(|needle| {
        let trimmed = needle.trim();
        !trimmed.is_empty() && label.contains(trimmed)
    })
}

fn render_fixture_stem<M: BlackBoxModel>(
    adapter: &mut MatrixAdapter<M>,
    tag: &str,
    in_1: Vec<f32>,
    in_2: Option<Vec<f32>>,
    control_values: &[f32],
) -> StemResponse {
    adapter
        .render_stem(&StemStimulus {
            stem_id: tag.to_string(),
            sample_rate_hz: 48_000,
            in_1,
            in_2,
            in_3: None,
            in_4: None,
            control_values: control_values.to_vec(),
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap()
}

fn evaluate_chain<M: BlackBoxModel>(
    label: &str,
    model: M,
    expect_incoherent: bool,
    rows: &mut Vec<RowResult>,
) {
    if !filter_matches(label) {
        return;
    }
    let crypto = DefaultCryptoProvider::default();
    let n_in = model.num_inputs();
    let n_out = model.num_outputs();
    let topo = match (n_in >= 2, n_out >= 2) {
        (true, true) => TopologyProbeMode::TrueStereo2To2,
        (true, false) => TopologyProbeMode::DualInToMono2To1,
        (false, true) => TopologyProbeMode::MonoToStereo1To2,
        (false, false) => TopologyProbeMode::Mono1To1,
    };
    let manifest = manifest_for_fixture(&model, label);
    let mut adapter = MatrixAdapter::from_model(model);
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: topo,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 6,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let report = train_adaptive_target(manifest, &mut adapter, &opts)
        .unwrap_or_else(|e| panic!("{label} train_adaptive_target failed: {e:?}"));

    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto)
        .unwrap_or_else(|e| panic!("{label} invalid .dnam-proj: {e:?}"));
    let _inpc_view =
        verify_proj_inpc_capsule(proj_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
    let star_entry = proj_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .unwrap();
    let _stmc_view = verify_proj_stmc_capsule(
        proj_view
            .chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let mono_only = proj_view.manifest.base.io_config.output_mode == "mono";
    let star_ctrls = star_center_control_values(&proj_view.manifest);

    // Causal alignment probe + empirical target repeatability profile (zero archetype labels).
    let probe_dry = build_v3_causal_probe_excitation();
    let probe_l: Vec<f32> = probe_dry.iter().map(|v| v[0]).collect();
    let probe_r: Vec<f32> = probe_dry.iter().map(|v| v[1]).collect();
    let probe_resp = render_fixture_stem(
        &mut adapter,
        "v3_causal_probe",
        probe_l.clone(),
        (n_in >= 2).then(|| probe_r.clone()),
        &star_ctrls,
    );
    let (eval_lag_l, eval_inv_l) =
        estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_resp.out_1);
    let (eval_lag, eval_inv_r) = probe_resp
        .out_2
        .as_ref()
        .map_or((eval_lag_l, eval_inv_l), |o2| {
            let (lr, ir) = estimate_v3_causal_deadtime_and_polarity(&probe_r, o2);
            (eval_lag_l.min(lr), ir)
        });

    let mut render_aligned = |tag: &str, dry: &[[f32; 2]]| -> (Vec<f64>, Vec<f64>) {
        let h1: Vec<f32> = dry.iter().map(|v| v[0]).collect();
        let h2: Vec<f32> = dry.iter().map(|v| v[1]).collect();
        let resp = render_fixture_stem(
            &mut adapter,
            tag,
            h1,
            (n_in >= 2).then_some(h2),
            &star_ctrls,
        );
        let al_l = realign_fractional_kaiser64(&resp.out_1, eval_lag, 0.0, eval_inv_l);
        let al_r = resp
            .out_2
            .as_ref()
            .map(|v| realign_fractional_kaiser64(v, eval_lag, 0.0, eval_inv_r));
        let tgt_l = al_l.iter().map(|&v| f64::from(v)).collect();
        let tgt_r = al_r
            .as_ref()
            .unwrap_or(&al_l)
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        (tgt_l, tgt_r)
    };

    let (rep_dry, rep_windows, rep_late_w) = build_repeatability_probe_stream(48_000.0);
    let (rep_tgt_l, rep_tgt_r) = render_aligned("repeatability_probe", &rep_dry);
    let prof_l = TargetRepeatabilityProfile::from_probe_response(
        &rep_tgt_l,
        &rep_windows,
        rep_late_w,
        48_000.0,
    );
    let prof_r = if mono_only {
        prof_l
    } else {
        TargetRepeatabilityProfile::from_probe_response(
            &rep_tgt_r,
            &rep_windows,
            rep_late_w,
            48_000.0,
        )
    };
    if expect_incoherent {
        assert!(
            prof_l.band_inc_ratio.iter().any(|&r| r > 0.0)
                || (!mono_only && prof_r.band_inc_ratio.iter().any(|&r| r > 0.0)),
            "{label} incoherent rig yielded zero repeatability floors: L={prof_l:?} R={prof_r:?}"
        );
    } else {
        assert_eq!(
            prof_l.band_inc_ratio, [0.0; 3],
            "{label} L deterministic device yielded non-zero repeatability floor: {prof_l:?}"
        );
        if !mono_only {
            assert_eq!(
                prof_r.band_inc_ratio, [0.0; 3],
                "{label} R deterministic device yielded non-zero repeatability floor: {prof_r:?}"
            );
        }
    }

    let eval_dry = build_fullband_eval_excitation(96_000, 48_000.0);
    let (tgt_l, tgt_r) = render_aligned("fullband_eval", &eval_dry);
    let n_common = eval_dry.len().min(tgt_l.len()).min(tgt_r.len());
    let warmup = canonical_besr_warmup(n_common);

    let holdout_dry = build_unseen_holdout_excitation(48_000, 48_000.0);
    let (hold_tgt_l, hold_tgt_r) = render_aligned("holdout", &holdout_dry);
    let hold_warmup = canonical_besr_warmup(hold_tgt_l.len());

    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut tier_rms = [0.0_f64; 4];
    let mut holdouts = [(0.0_f64, false); 4];
    let mut sums_l = Vec::with_capacity(4);
    let mut sums_r = Vec::with_capacity(4);
    let mut combined_summaries = Vec::with_capacity(4);

    for (t_idx, tier) in tiers.into_iter().enumerate() {
        let tier_dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&tier_dnam, tier, 48_000).unwrap();
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let rendered: Vec<[f32; 2]> = eval_dry[..n_common]
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let sum_sq: f64 = rendered[warmup..n_common]
            .iter()
            .map(|&[l, r]| {
                if mono_only {
                    f64::from(l) * f64::from(l)
                } else {
                    0.5 * (f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r))
                }
            })
            .sum();
        tier_rms[t_idx] = (sum_sq / ((n_common - warmup).max(1) as f64))
            .sqrt()
            .max(1.0e-12);

        let mdl_l: Vec<f64> = rendered.iter().map(|v| f64::from(v[0])).collect();
        let sl =
            certify_4metric_universal(tier, &tgt_l[..n_common], &mdl_l, &prof_l, 48_000.0, warmup);
        sums_l.push(sl.clone());
        let combined = if mono_only {
            sl
        } else {
            let mdl_r: Vec<f64> = rendered.iter().map(|v| f64::from(v[1])).collect();
            let sr = certify_4metric_universal(
                tier,
                &tgt_r[..n_common],
                &mdl_r,
                &prof_r,
                48_000.0,
                warmup,
            );
            sums_r.push(sr.clone());
            combine_stereo_summary(sl, sr)
        };
        combined_summaries.push(combined);

        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let h_rend: Vec<[f32; 2]> = holdout_dry
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let hl: Vec<f64> = h_rend.iter().map(|v| f64::from(v[0])).collect();
        let (db_l, pass_l) =
            certify_holdout_universal(tier, &hold_tgt_l, &hl, &prof_l, 48_000.0, hold_warmup);
        holdouts[t_idx] = if mono_only {
            (db_l, pass_l)
        } else {
            let hr: Vec<f64> = h_rend.iter().map(|v| f64::from(v[1])).collect();
            let (db_r, pass_r) =
                certify_holdout_universal(tier, &hold_tgt_r, &hr, &prof_r, 48_000.0, hold_warmup);
            (db_l.max(db_r), pass_l && pass_r)
        };
    }

    let arr_l: [PerceptualCertificationSummary; 4] = sums_l.try_into().unwrap();
    let monotone = verify_monotone_4tier_summaries(&arr_l)
        && (mono_only || {
            let arr_r: [PerceptualCertificationSummary; 4] = sums_r.try_into().unwrap();
            verify_monotone_4tier_summaries(&arr_r)
        });
    let (lu_step, lu_pass) = verify_cross_tier_loudness_parity(tier_rms);

    for (t_idx, summary) in combined_summaries.into_iter().enumerate() {
        let (holdout_besr_db, holdout_pass) = holdouts[t_idx];
        rows.push(RowResult {
            label: label.to_string(),
            tier: summary.tier,
            summary,
            holdout_besr_db,
            lu_step,
            lu_pass,
            monotone,
            holdout_pass,
        });
    }
}

fn assert_matrix_results(rows: &[RowResult], expected_unfiltered: usize) {
    println!(
        "\n| Chain | Tier | bESR (dB) | LF (dB) | MF (dB) | HF (dB) | LTAS (dB) | MR-IIR | SegESR (dB) | Holdout (dB) | LU Step | Mono | Tgt | Ceil |"
    );
    println!(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |"
    );
    let mut failures = Vec::new();
    for r in rows {
        let s = &r.summary;
        println!(
            "| {} | {:?} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} | {:.3} | {:.2} | {:.2} | {:.4} | {} | {} | {} |",
            r.label,
            r.tier,
            s.besr_db,
            s.band_esr_db[0],
            s.band_esr_db[1],
            s.band_esr_db[2],
            s.ltas_max_db,
            s.mr_iir,
            s.seg_esr_db,
            r.holdout_besr_db,
            r.lu_step,
            r.monotone,
            s.passes_target && r.holdout_pass,
            s.passes_hard_ceiling
        );
        if !(s.passes_target && s.passes_hard_ceiling && r.monotone && r.lu_pass && r.holdout_pass)
        {
            failures.push(format!(
                "{} {:?}: besr={:.2}, bands={:.2?}, ltas={:.3}, mr={:.3}, seg={:.2}, hold={:.2} (ok={}), lu={:.4}, mono={}, tgt={}, ceil={}",
                r.label,
                r.tier,
                s.besr_db,
                s.band_esr_db,
                s.ltas_max_db,
                s.mr_iir,
                s.seg_esr_db,
                r.holdout_besr_db,
                r.holdout_pass,
                r.lu_step,
                r.monotone,
                s.passes_target,
                s.passes_hard_ceiling
            ));
        }
    }
    println!(
        "\nCanonical §4 Universal Summary: {} / {} cells meet full §4 target & ceiling",
        rows.len() - failures.len(),
        rows.len()
    );

    let filter_active = std::env::var("DNAM_BENCH_FILTER")
        .ok()
        .is_some_and(|v| !v.trim().is_empty());
    if filter_active {
        assert!(!rows.is_empty(), "DNAM_BENCH_FILTER matched no chains");
    } else {
        assert_eq!(rows.len(), expected_unfiltered);
    }

    let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
    for r in rows {
        let s = &r.summary;
        assert!(
            valid_db(s.raw_besr_db)
                && valid_db(s.besr_db)
                && s.band_esr_db.iter().copied().all(valid_db)
                && s.ltas_max_db.is_finite()
                && s.mr_iir.is_finite()
                && s.mr_iir >= 0.0
                && valid_db(s.seg_esr_db)
                && valid_db(r.holdout_besr_db),
            "{} {:?} produced invalid canonical §4 metrics",
            r.label,
            r.tier
        );
        assert!(
            s.ltas_excluded_bands.iter().all(|&ex| !ex),
            "{} {:?} unexpectedly excluded LTAS bands across 20 Hz-20 kHz",
            r.label,
            r.tier
        );
    }
    assert!(
        failures.is_empty(),
        "{} / {} benchmark cells missed §4 targets:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "full 14x4 universal single-device & amp+cab qualification matrix; run via moon run capture:qualify"]
fn test_benchmark_matrix_14_chains_4_tiers_telemetry_and_acceptance() {
    let mut rows = Vec::with_capacity(56);
    evaluate_chain(
        "Chain-01 SymmetricOD",
        Chain01SymmetricOd::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-02 GatedFuzz",
        Chain02GatedAsymmetricFuzz::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-03 CleanBlend",
        Chain03ParallelCleanBlend::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-04 TubeAmp4x12",
        Chain04TubeAmpFmv4x12::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-05 PureLTI",
        Chain05PureLtiCabAndRoom::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-06 EnvSVF",
        Chain06EnvelopeResonantSvf::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-07 StereoChorus",
        Chain07StereoChorusAndRotary::new(),
        true,
        &mut rows,
    );
    evaluate_chain(
        "Chain-08 BbdShimmer",
        Chain08LongBbdEchoAnd30sShimmer::new(),
        true,
        &mut rows,
    );
    evaluate_chain(
        "Chain-09 LA2A",
        Chain09La2aOpticalCompressor::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-10 OC2RingMod",
        Chain10Oc2OctaveDividerAndRingMod::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-11 BiAmpBass",
        Chain11BiAmpBassAndOutputIron::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-13 TriBandVcaComp",
        Chain13TriBandVcaCompressor::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-14 TriBandEntropia",
        Chain14TriBandCompDistortion::new(),
        false,
        &mut rows,
    );
    evaluate_chain(
        "Chain-15 TubeAmpHeadDI",
        Chain15TubeAmpHeadDi::new(),
        false,
        &mut rows,
    );
    assert_matrix_results(&rows, 56);
}

#[test]
#[ignore = "optional complex multi-fx smoke rig; run with DNAM_SMOKE_CHAIN12=1"]
fn smoke_chain12_full_end_to_end_stereo_rig() {
    if std::env::var("DNAM_SMOKE_CHAIN12").as_deref() != Ok("1") {
        return;
    }
    let mut rows = Vec::with_capacity(8);
    evaluate_chain(
        "Chain-12 FixedZero",
        Chain12FullEndToEndStereoRig::new(Chain12RigMode::FixedZeroKnob),
        true,
        &mut rows,
    );
    evaluate_chain(
        "Chain-12 Parametric16",
        Chain12FullEndToEndStereoRig::new(Chain12RigMode::Parametric16),
        true,
        &mut rows,
    );
    assert_matrix_results(&rows, 8);
}

#[derive(Clone, Debug)]
struct UnrecRowResult {
    label: String,
    tier: Tier,
    unrec_esr_db: f64,
    frozen_esr_db: f64,
    rms_diff_db: f64,
    frozen_rms_diff_db: f64,
    unrec_besr_db: f64,
    frozen_besr_db: f64,
    unrec_ltas_db: f64,
    frozen_ltas_db: f64,
    unrec_mr_iir: f64,
    frozen_mr_iir: f64,
}

fn build_unrecorded_offaxis_setting(
    manifest: &dnam_capture::proj::DnamProjManifest,
    star_ctrls: &[f32],
) -> Vec<f32> {
    const OFFSETS: [f32; 11] = [
        -0.42, 0.37, -0.31, 0.29, -0.24, 0.41, -0.35, 0.27, -0.22, 0.34, -0.28,
    ];
    let mut unrec = star_ctrls.to_vec();
    let mut moved_first_sw = false;
    for (p, schema) in manifest.base.controls_schema.iter().enumerate() {
        if let Some(opts) = &schema.declared_options
            && !opts.is_empty()
        {
            if !moved_first_sw {
                let def_i = (star_ctrls[p].round().max(0.0) as usize) % opts.len();
                unrec[p] = ((def_i + 1) % opts.len()) as f32;
                moved_first_sw = true;
            }
        } else {
            let delta = OFFSETS[p % OFFSETS.len()];
            let cand = (star_ctrls[p] + delta).clamp(-0.85, 0.85);
            unrec[p] = if (cand - star_ctrls[p]).abs() < 0.20 {
                (star_ctrls[p] - delta).clamp(-0.85, 0.85)
            } else {
                cand
            };
        }
    }
    unrec
}

fn eval_unrecorded_controls_chain<M: BlackBoxModel>(
    label: &str,
    model: M,
    rows: &mut Vec<UnrecRowResult>,
) {
    if !filter_matches(label) {
        return;
    }
    let n_in = model.num_inputs();
    let n_out = model.num_outputs();
    let topo = match (n_in >= 2, n_out >= 2) {
        (true, true) => TopologyProbeMode::TrueStereo2To2,
        (true, false) => TopologyProbeMode::DualInToMono2To1,
        (false, true) => TopologyProbeMode::MonoToStereo1To2,
        (false, false) => TopologyProbeMode::Mono1To1,
    };
    let manifest = manifest_for_fixture(&model, label);
    if manifest.base.controls_schema.is_empty() {
        return;
    }
    let mono_only = manifest.base.io_config.output_mode == "mono";
    let mut adapter = MatrixAdapter::from_model(model);
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: topo,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 6,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let report = train_adaptive_target(manifest.clone(), &mut adapter, &opts)
        .unwrap_or_else(|e| panic!("{label} train_adaptive_target failed: {e:?}"));
    let star_ctrls = star_center_control_values(&manifest);
    let unrec_ctrls = build_unrecorded_offaxis_setting(&manifest, &star_ctrls);

    let probe_dry = build_v3_causal_probe_excitation();
    let probe_l: Vec<f32> = probe_dry.iter().map(|v| v[0]).collect();
    let probe_r: Vec<f32> = probe_dry.iter().map(|v| v[1]).collect();
    let probe_resp = render_fixture_stem(
        &mut adapter,
        "unrec_probe",
        probe_l.clone(),
        (n_in >= 2).then_some(probe_r),
        &unrec_ctrls,
    );
    let (lag_l, inv_l) = estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_resp.out_1);

    let dry_eval = build_fullband_eval_excitation(48_000, 48_000.0);
    let eval_l: Vec<f32> = dry_eval.iter().map(|v| v[0]).collect();
    let eval_r: Vec<f32> = dry_eval.iter().map(|v| v[1]).collect();
    let actual_resp = render_fixture_stem(
        &mut adapter,
        "actual_unrec",
        eval_l,
        (n_in >= 2).then_some(eval_r),
        &unrec_ctrls,
    );
    let actual_l: Vec<f64> = realign_fractional_kaiser64(&actual_resp.out_1, lag_l, 0.0, inv_l)
        .into_iter()
        .map(f64::from)
        .collect();
    let warmup = canonical_besr_warmup(actual_l.len());
    let prof = TargetRepeatabilityProfile::deterministic();
    let esr_fn = |mdl: &[f64]| -> f64 {
        let mut s = 0.0_f64;
        let mut e = 0.0_f64;
        for i in warmup..actual_l.len() {
            let d = actual_l[i] - mdl[i];
            s += actual_l[i] * actual_l[i];
            e += d * d;
        }
        10.0 * (e / s.max(1.0e-18)).max(1.0e-12).log10()
    };

    for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam, tier, 48_000).unwrap();
        eng.reset_to_dc_equilibrium();
        for (p, &v) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, v);
        }
        let frozen_l: Vec<f64> = dry_eval
            .iter()
            .map(|&[dl, dr]| f64::from(eng.step_frame_tora(dl, dr, mono_only)[0]))
            .collect();
        let frozen_esr_db = esr_fn(&frozen_l);
        let frozen_sum =
            certify_4metric_universal(tier, &actual_l, &frozen_l, &prof, 48_000.0, warmup);

        let e_act: f64 = actual_l[warmup..].iter().map(|v| v * v).sum();
        let e_frz: f64 = frozen_l[warmup..].iter().map(|v| v * v).sum();
        let frozen_rms_diff_db = (10.0 * ((e_frz + 1.0e-15) / (e_act + 1.0e-15)).log10()).abs();

        eng.reset_to_dc_equilibrium();
        for (p, &v) in unrec_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, v);
        }
        let unrec_l: Vec<f64> = dry_eval
            .iter()
            .map(|&[dl, dr]| f64::from(eng.step_frame_tora(dl, dr, mono_only)[0]))
            .collect();
        let unrec_esr_db = esr_fn(&unrec_l);
        let e_mdl: f64 = unrec_l[warmup..].iter().map(|v| v * v).sum();
        let rms_diff_db = (10.0 * ((e_mdl + 1.0e-15) / (e_act + 1.0e-15)).log10()).abs();
        let sum = certify_4metric_universal(tier, &actual_l, &unrec_l, &prof, 48_000.0, warmup);
        rows.push(UnrecRowResult {
            label: label.to_string(),
            tier,
            unrec_esr_db,
            frozen_esr_db,
            rms_diff_db,
            frozen_rms_diff_db,
            unrec_besr_db: sum.besr_db,
            frozen_besr_db: frozen_sum.besr_db,
            unrec_ltas_db: sum.ltas_max_db,
            frozen_ltas_db: frozen_sum.ltas_max_db,
            unrec_mr_iir: sum.mr_iir,
            frozen_mr_iir: frozen_sum.mr_iir,
        });
    }
}

#[test]
#[ignore = "multi-setting capture & unrecorded off-axis control validation matrix"]
fn test_multisetting_unrecorded_control_interpolation_matrix() {
    let mut rows = Vec::with_capacity(48);
    eval_unrecorded_controls_chain("Chain-01 SymmetricOD", Chain01SymmetricOd::new(), &mut rows);
    eval_unrecorded_controls_chain(
        "Chain-02 GatedFuzz",
        Chain02GatedAsymmetricFuzz::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-03 CleanBlend",
        Chain03ParallelCleanBlend::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-04 TubeAmp4x12",
        Chain04TubeAmpFmv4x12::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-06 EnvSVF",
        Chain06EnvelopeResonantSvf::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-09 LA2A",
        Chain09La2aOpticalCompressor::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-10 OC2RingMod",
        Chain10Oc2OctaveDividerAndRingMod::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-11 BiAmpBass",
        Chain11BiAmpBassAndOutputIron::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-13 TriBandVcaComp",
        Chain13TriBandVcaCompressor::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-14 TriBandEntropia",
        Chain14TriBandCompDistortion::new(),
        &mut rows,
    );
    eval_unrecorded_controls_chain(
        "Chain-15 TubeAmpHeadDI",
        Chain15TubeAmpHeadDi::new(),
        &mut rows,
    );

    println!(
        "\n| Chain | Tier | Unrec ESR (dB) | Frozen ESR (dB) | dRMS (dB) | Unrec bESR (dB) | Unrec LTAS (dB) | Frozen LTAS (dB) | Unrec MR | Frozen MR | Improves |"
    );
    println!("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |");
    let mut regressions = Vec::new();
    for r in &rows {
        let improves = (r.unrec_esr_db + 0.15 < r.frozen_esr_db)
            || (r.unrec_besr_db + 0.15 < r.frozen_besr_db)
            || (r.unrec_mr_iir + 0.005 < r.frozen_mr_iir
                && r.rms_diff_db <= r.frozen_rms_diff_db + 0.5);
        let level_ok = r.rms_diff_db.is_finite() && r.rms_diff_db <= 6.0;
        println!(
            "| {} | {:?} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
            r.label,
            r.tier,
            r.unrec_esr_db,
            r.frozen_esr_db,
            r.rms_diff_db,
            r.unrec_besr_db,
            r.unrec_ltas_db,
            r.frozen_ltas_db,
            r.unrec_mr_iir,
            r.frozen_mr_iir,
            improves && level_ok
        );
        if !r.unrec_besr_db.is_finite() || !level_ok || !improves {
            regressions.push(format!(
                "{} {:?} (ESR {:.2} vs {:.2}, bESR {:.2} vs {:.2}, dRMS {:.2}, LTAS {:.3} vs {:.3}, MR {:.3} vs {:.3})",
                r.label,
                r.tier,
                r.unrec_esr_db,
                r.frozen_esr_db,
                r.unrec_besr_db,
                r.frozen_besr_db,
                r.rms_diff_db,
                r.unrec_ltas_db,
                r.frozen_ltas_db,
                r.unrec_mr_iir,
                r.frozen_mr_iir
            ));
        }
    }
    assert!(
        regressions.is_empty(),
        "Unrecorded control regressions: {regressions:?}"
    );
}

#[test]
fn test_canonical_tier_thresholds_device_agnostic_and_monotone() {
    // verify[qual.tiers.benchmark-matrix]
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut prev: Option<TierQualityThresholds> = None;
    for tier in tiers {
        let cur = TierQualityThresholds::for_tier(tier);
        assert!(
            cur.composite_besr_db < 0.0
                && cur.per_band_besr_db.iter().all(|&b| b < 0.0)
                && cur.unseen_control_besr_db < 0.0
                && cur.unseen_control_besr_db >= cur.composite_besr_db
                && cur.max_ltas_db > 0.0
                && cur.max_ltas_db <= 0.75
        );
        if let Some(lo) = prev {
            assert!(
                cur.composite_besr_db < lo.composite_besr_db
                    && cur.max_ltas_db < lo.max_ltas_db
                    && cur.unseen_control_besr_db < lo.unseen_control_besr_db
            );
            for (&hi_b, &lo_b) in cur.per_band_besr_db.iter().zip(lo.per_band_besr_db.iter()) {
                assert!(hi_b < lo_b);
            }
        }
        prev = Some(cur);
    }
}
