#![forbid(unsafe_code)]
//! Head-to-head `v2.0` (`DnamEngine`) vs `v3` (`V3CoreInstance` + `VarPro`/Adjoint L-BFGS) benchmark
//! across `[Lite, Full, Pro, Ultra]` on flagship hard nonlinear & dynamic black-box chains.

use std::time::Instant;

use dnam_capture::metrics::{
    ArchetypeFamily, PerceptualCertificationSummary, TierQualityThresholds,
    certify_4metric_signal_chain, evaluate_besr_from_band_energies, split_bezout_3band_energies,
    verify_monotone_4tier_summaries,
};
use dnam_capture::proj::{
    DnamProjView, ProjChunkKind, STMC_FLAG_WET_POLARITY_INVERTED, export_runtime_dnam,
    verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    V3TrainDataset, V3TrainStem, build_v3_causal_probe_excitation, canonical_besr_warmup,
    estimate_v3_causal_deadtime_and_polarity, fit_v3_matryoshka, star_center_control_values,
};
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_capture::{DistillationTarget, realign_fractional_kaiser64};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::stages::v3_core::V3CoreInstance;
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain04TubeAmpFmv4x12,
    Chain06EnvelopeResonantSvf,
};

#[path = "v3_prototype_benchmark/harness.rs"]
mod harness;

use harness::{
    MatrixAdapter, build_fullband_eval_excitation, build_identification_sweep_excitation,
    build_unseen_holdout_excitation, combine_stereo_summary, manifest_for_fixture,
};

struct CompareRow {
    label: String,
    tier: Tier,
    v2_sum: PerceptualCertificationSummary,
    v3_sum: PerceptualCertificationSummary,
    v2_holdout_db: f64,
    v3_holdout_db: f64,
    v3_os2x: bool,
    v2_ms: u128,
    v3_ms: u128,
    v3_monotone: bool,
}

fn benchmark_chain<M: BlackBoxModel>(
    label: &str,
    model: M,
    family: ArchetypeFamily,
    rows: &mut Vec<CompareRow>,
) {
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
        max_frames_per_stem: 3,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let t_v2 = Instant::now();
    let report = train_adaptive_target(manifest, &mut adapter, &opts).unwrap();
    let v2_ms = t_v2.elapsed().as_millis();

    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto).unwrap();
    let _inpc_view =
        verify_proj_inpc_capsule(proj_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
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
    let mono_only = proj_view.manifest.base.io_config.output_mode == "mono";

    let star_ctrls = star_center_control_values(&proj_view.manifest);
    let hold_inv_pol = (stmc_view.flags & STMC_FLAG_WET_POLARITY_INVERTED) != 0;
    let hold_lag = f64::from(stmc_view.fractional_lag_samples_f32);

    let render_aligned =
        |adapter: &mut MatrixAdapter<M>, name: &str, dry: &[[f32; 2]]| -> (Vec<f32>, Vec<f32>) {
            let h1: Vec<f32> = dry.iter().map(|v| v[0]).collect();
            let h2: Vec<f32> = dry.iter().map(|v| v[1]).collect();
            adapter.reset_state();
            let resp = adapter
                .render_internal(
                    name.to_string(),
                    48_000,
                    &h1,
                    (n_in >= 2).then_some(h2.as_slice()),
                    None,
                    None,
                    &star_ctrls,
                    None,
                    None,
                )
                .unwrap();
            let al = realign_fractional_kaiser64(&resp.out_1, hold_lag, 0.0, hold_inv_pol);
            let ar = resp.out_2.as_ref().map_or_else(
                || al.clone(),
                |v| realign_fractional_kaiser64(v, hold_lag, 0.0, hold_inv_pol),
            );
            (al, ar)
        };

    // v3 strictly causal black-box alignment probe (deadtime >= 0, low-frequency polarity).
    let probe_dry = build_v3_causal_probe_excitation();
    let probe_l: Vec<f32> = probe_dry.iter().map(|v| v[0]).collect();
    let probe_r: Vec<f32> = probe_dry.iter().map(|v| v[1]).collect();
    adapter.reset_state();
    let probe_resp = adapter
        .render_internal(
            "v3_causal_probe".to_string(),
            48_000,
            &probe_l,
            (n_in >= 2).then_some(probe_r.as_slice()),
            None,
            None,
            &star_ctrls,
            None,
            None,
        )
        .unwrap();
    let (v3_lag_l, v3_inv_l) =
        estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_resp.out_1);
    let (v3_lag, v3_inv_r) = probe_resp
        .out_2
        .as_ref()
        .map_or((v3_lag_l, v3_inv_l), |o2| {
            let (lr, ir) = estimate_v3_causal_deadtime_and_polarity(&probe_r, o2);
            (v3_lag_l.min(lr), ir)
        });
    let render_v3_causal =
        |adapter: &mut MatrixAdapter<M>, name: &str, dry: &[[f32; 2]]| -> (Vec<f32>, Vec<f32>) {
            let h1: Vec<f32> = dry.iter().map(|v| v[0]).collect();
            let h2: Vec<f32> = dry.iter().map(|v| v[1]).collect();
            adapter.reset_state();
            let resp = adapter
                .render_internal(
                    name.to_string(),
                    48_000,
                    &h1,
                    (n_in >= 2).then_some(h2.as_slice()),
                    None,
                    None,
                    &star_ctrls,
                    None,
                    None,
                )
                .unwrap();
            let al = realign_fractional_kaiser64(&resp.out_1, v3_lag, 0.0, v3_inv_l);
            let ar = resp.out_2.as_ref().map_or_else(
                || al.clone(),
                |v| realign_fractional_kaiser64(v, v3_lag, 0.0, v3_inv_r),
            );
            (al, ar)
        };

    // Unseen holdout target (v2 STMC aligned vs v3 causal deadtime aligned)
    let holdout_dry = build_unseen_holdout_excitation(48_000, 48_000.0);
    let (hold_al_l, hold_al_r) = render_aligned(&mut adapter, "holdout", &holdout_dry);
    let hold_tgt_l: Vec<f64> = hold_al_l.iter().map(|&v| f64::from(v)).collect();
    let hold_tgt_r: Vec<f64> = hold_al_r.iter().map(|&v| f64::from(v)).collect();
    let (v3_hold_al_l, v3_hold_al_r) = render_v3_causal(&mut adapter, "v3_holdout", &holdout_dry);
    let v3_hold_tgt_l: Vec<f64> = v3_hold_al_l.iter().map(|&v| f64::from(v)).collect();
    let v3_hold_tgt_r: Vec<f64> = v3_hold_al_r.iter().map(|&v| f64::from(v)).collect();
    let hold_warmup = canonical_besr_warmup(hold_tgt_l.len());
    let hold_in_l: Vec<f32> = holdout_dry.iter().map(|v| v[0]).collect();
    let hold_in_r: Vec<f32> = holdout_dry.iter().map(|v| v[1]).collect();

    // Full-bandwidth 20 Hz-20 kHz f32 identification & certification signal (96,000 samples across all 33 LTAS bands)
    let eval_dry = build_fullband_eval_excitation(96_000, 48_000.0);
    let (eval_al_l, eval_al_r) = render_aligned(&mut adapter, "fullband_eval", &eval_dry);
    let n_eval = eval_dry.len().min(eval_al_l.len()).min(eval_al_r.len());
    let eval_tgt_l: Vec<f64> = eval_al_l[..n_eval].iter().map(|&v| f64::from(v)).collect();
    let eval_tgt_r: Vec<f64> = eval_al_r[..n_eval].iter().map(|&v| f64::from(v)).collect();
    let (v3_eval_al_l, v3_eval_al_r) =
        render_v3_causal(&mut adapter, "v3_fullband_eval", &eval_dry);
    let v3_eval_tgt_l: Vec<f64> = v3_eval_al_l[..n_eval]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let v3_eval_tgt_r: Vec<f64> = v3_eval_al_r[..n_eval]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let eval_warmup = canonical_besr_warmup(n_eval);
    let eval_in_l: Vec<f32> = eval_dry[..n_eval].iter().map(|v| v[0]).collect();
    let eval_in_r: Vec<f32> = eval_dry[..n_eval].iter().map(|v| v[1]).collect();

    // v3 Identification Dataset: [fullband StarCenter excitation, disjoint 20 Hz-20 kHz sweep]
    let sweep_dry = build_identification_sweep_excitation(48_000, 48_000.0);
    let (sw_al_l, sw_al_r) = render_v3_causal(&mut adapter, "v3_id_sweep", &sweep_dry);
    let fullband_stem = V3TrainStem {
        in_l: eval_in_l.clone(),
        in_r: eval_in_r.clone(),
        tgt_l: v3_eval_al_l[..n_eval].to_vec(),
        tgt_r: v3_eval_al_r[..n_eval].to_vec(),
        warmup: eval_warmup,
    };
    let sw_len = sweep_dry.len().min(sw_al_l.len()).min(sw_al_r.len());
    let sweep_stem = V3TrainStem {
        in_l: sweep_dry[..sw_len].iter().map(|v| v[0]).collect(),
        in_r: sweep_dry[..sw_len].iter().map(|v| v[1]).collect(),
        tgt_l: sw_al_l[..sw_len].to_vec(),
        tgt_r: sw_al_r[..sw_len].to_vec(),
        warmup: canonical_besr_warmup(sw_len),
    };
    let ds =
        V3TrainDataset::new_weighted(vec![(fullband_stem, 0.80), (sweep_stem, 0.20)], 48_000.0);

    let t_v3 = Instant::now();
    let v3_mat = fit_v3_matryoshka(&ds, mono_only);
    let v3_models = v3_mat.to_four_tiers();
    let v3_ms = t_v3.elapsed().as_millis();

    let eval_hold = |tgt_l: &[f64], tgt_r: &[f64], mdl_l: &[f64], mdl_r: &[f64]| -> f64 {
        let (s, e, ns) = split_bezout_3band_energies(tgt_l, mdl_l, None, 48_000.0, hold_warmup);
        let rl = evaluate_besr_from_band_energies(s, e, ns).besr_db;
        if mono_only {
            return rl;
        }
        let (sr, er, nr) = split_bezout_3band_energies(tgt_r, mdl_r, None, 48_000.0, hold_warmup);
        rl.max(evaluate_besr_from_band_energies(sr, er, nr).besr_db)
    };

    let cert_fullband = |tier: Tier,
                         tgt_l: &[f64],
                         tgt_r: &[f64],
                         mdl_l: &[f64],
                         mdl_r: &[f64]|
     -> PerceptualCertificationSummary {
        let sl =
            certify_4metric_signal_chain(family, tier, tgt_l, mdl_l, None, 48_000.0, eval_warmup);
        if mono_only {
            sl
        } else {
            let sr = certify_4metric_signal_chain(
                family,
                tier,
                tgt_r,
                mdl_r,
                None,
                48_000.0,
                eval_warmup,
            );
            combine_stereo_summary(sl, sr)
        }
    };

    let mut v2_sums = Vec::with_capacity(4);
    let mut v3_sums = Vec::with_capacity(4);
    let mut v2_holds = [0.0_f64; 4];
    let mut v3_holds = [0.0_f64; 4];

    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam, tier, 48_000).unwrap();
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let v2_c: Vec<[f32; 2]> = eval_dry[..n_eval]
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let v2_cl: Vec<f64> = v2_c.iter().map(|v| f64::from(v[0])).collect();
        let v2_cr: Vec<f64> = v2_c.iter().map(|v| f64::from(v[1])).collect();
        v2_sums.push(cert_fullband(
            tier,
            &eval_tgt_l,
            &eval_tgt_r,
            &v2_cl,
            &v2_cr,
        ));

        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let v2_h: Vec<[f32; 2]> = holdout_dry
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let v2_hl: Vec<f64> = v2_h.iter().map(|v| f64::from(v[0])).collect();
        let v2_hr: Vec<f64> = v2_h.iter().map(|v| f64::from(v[1])).collect();
        v2_holds[t_idx] = eval_hold(&hold_tgt_l, &hold_tgt_r, &v2_hl, &v2_hr);

        let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&v3_models[t_idx]);
        let (mut o3_l, mut o3_r) = (vec![0.0_f32; n_eval], vec![0.0_f32; n_eval]);
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(
            &eval_in_l,
            if mono_only { &eval_in_l } else { &eval_in_r },
            &mut o3_l,
            &mut o3_r,
        );
        let v3_cl: Vec<f64> = o3_l.iter().map(|&v| f64::from(v)).collect();
        let v3_cr: Vec<f64> = o3_r.iter().map(|&v| f64::from(v)).collect();
        v3_sums.push(cert_fullband(
            tier,
            &v3_eval_tgt_l,
            &v3_eval_tgt_r,
            &v3_cl,
            &v3_cr,
        ));

        let (mut h3_l, mut h3_r) = (
            vec![0.0_f32; hold_in_l.len()],
            vec![0.0_f32; hold_in_r.len()],
        );
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(
            &hold_in_l,
            if mono_only { &hold_in_l } else { &hold_in_r },
            &mut h3_l,
            &mut h3_r,
        );
        let v3_hl: Vec<f64> = h3_l.iter().map(|&v| f64::from(v)).collect();
        let v3_hr: Vec<f64> = h3_r.iter().map(|&v| f64::from(v)).collect();
        v3_holds[t_idx] = eval_hold(&v3_hold_tgt_l, &v3_hold_tgt_r, &v3_hl, &v3_hr);
    }

    let v3_arr: [PerceptualCertificationSummary; 4] = v3_sums.clone().try_into().unwrap();
    let v3_monotone = verify_monotone_4tier_summaries(&v3_arr);
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let s = &v3_sums[t_idx];
        println!(
            "[{label}] {tier:?}: v3 bESR={:+.2} ({:+.1}/{:+.1}/{:+.1}) LTAS={:.2} Hold={:+.2} ms={} mono={} pass={}",
            s.besr_db,
            s.band_esr_db[0],
            s.band_esr_db[1],
            s.band_esr_db[2],
            s.ltas_max_db,
            v3_holds[t_idx],
            v3_ms,
            v3_monotone,
            s.passes_target,
        );
    }

    let mut live_inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&v3_models[0]);
    live_inst.reset_to_dc_equilibrium();
    let live_q = n_eval / 4;
    let live_peak_ref = v3_eval_tgt_l
        .iter()
        .chain(v3_eval_tgt_r.iter())
        .map(|v| v.abs())
        .fold(1.0_f64, f64::max) as f32;
    let mut live_l = vec![0.0_f32; live_q];
    let mut live_r = vec![0.0_f32; live_q];
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        live_inst.switch_tier(&v3_models[t_idx]);
        let start = t_idx * live_q;
        let end = start + live_q;
        live_inst.process_block_stereo(
            &eval_in_l[start..end],
            if mono_only {
                &eval_in_l[start..end]
            } else {
                &eval_in_r[start..end]
            },
            &mut live_l,
            &mut live_r,
        );
        for (&yl, &yr) in live_l.iter().zip(live_r.iter()) {
            assert!(
                yl.is_finite()
                    && yr.is_finite()
                    && yl.abs() <= (live_peak_ref * 4.0).max(16.0)
                    && yr.abs() <= (live_peak_ref * 4.0).max(16.0),
                "{label} {tier:?} live switch_tier produced non-finite or spiking sample: ({yl}, {yr})"
            );
        }
    }

    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        rows.push(CompareRow {
            label: label.to_string(),
            tier,
            v2_sum: v2_sums[t_idx].clone(),
            v3_sum: v3_sums[t_idx].clone(),
            v2_holdout_db: v2_holds[t_idx],
            v3_holdout_db: v3_holds[t_idx],
            v3_os2x: v3_models[t_idx].os2x,
            v2_ms,
            v3_ms,
            v3_monotone,
        });
    }
}

#[test]
fn test_v3_prototype_head_to_head_vs_v2() {
    let mut rows = Vec::with_capacity(16);
    benchmark_chain(
        "Chain-04 TubeAmp4x12",
        Chain04TubeAmpFmv4x12::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    benchmark_chain(
        "Chain-02 GatedFuzz",
        Chain02GatedAsymmetricFuzz::new(),
        ArchetypeFamily::FuzzOctaveChaos,
        &mut rows,
    );
    benchmark_chain(
        "Chain-06 EnvSVF",
        Chain06EnvelopeResonantSvf::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    benchmark_chain(
        "Chain-01 SymmetricOD",
        Chain01SymmetricOd::new(),
        ArchetypeFamily::CleanPedalPreamp,
        &mut rows,
    );

    println!("\n=== v2.0 vs v3 Prototype Head-to-Head Benchmark ===");
    println!(
        "| Chain | Tier | v2 raw bESR | v3 bESR (raw) | Delta (v3-v2) | v3 LF / MF / HF (dB) | v2 LTAS | v3 LTAS | v2 Hold | v3 Hold | OS2x | v2 ms | v3 ms | Mono | Pass |"
    );
    println!(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: | :---: |"
    );
    for r in &rows {
        let delta = r.v3_sum.raw_besr_db - r.v2_sum.raw_besr_db;
        println!(
            "| {} | {:?} | {:+.2} dB | {:+.2} ({:+.2}) dB | {:+.2} dB | {:+.1} / {:+.1} / {:+.1} | {:.2} | {:.2} | {:+.2} | {:+.2} | {} | {} | {} | {} | {} |",
            r.label,
            r.tier,
            r.v2_sum.raw_besr_db,
            r.v3_sum.besr_db,
            r.v3_sum.raw_besr_db,
            delta,
            r.v3_sum.band_esr_db[0],
            r.v3_sum.band_esr_db[1],
            r.v3_sum.band_esr_db[2],
            r.v2_sum.ltas_max_db,
            r.v3_sum.ltas_max_db,
            r.v2_holdout_db,
            r.v3_holdout_db,
            if r.v3_os2x { "2x" } else { "1x" },
            r.v2_ms,
            r.v3_ms,
            r.v3_monotone,
            if r.v3_sum.passes_target {
                "PASS"
            } else {
                "FAIL"
            },
        );
    }

    for r in &rows {
        assert!(
            r.v3_sum.ltas_excluded_bands.iter().all(|&ex| !ex),
            "{} {:?} v3 unexpectedly excluded LTAS bands",
            r.label,
            r.tier
        );
        assert!(
            r.v2_sum.ltas_excluded_bands.iter().all(|&ex| !ex),
            "{} {:?} v2 unexpectedly excluded LTAS bands",
            r.label,
            r.tier
        );
        assert!(
            r.v3_monotone,
            "{} v3 failed strict 7-metric 4-tier monotonicity",
            r.label
        );
        let th = TierQualityThresholds::for_tier(r.tier);
        assert!(
            r.v3_sum.passes_target,
            "{} {:?} failed passes_target (besr={:.2} vs {:.2}, bands={:?} vs {:?}, ltas={:.2} vs {:.2})",
            r.label,
            r.tier,
            r.v3_sum.besr_db,
            th.composite_besr_db,
            r.v3_sum.band_esr_db,
            th.per_band_besr_db,
            r.v3_sum.ltas_max_db,
            th.max_ltas_db,
        );
        assert!(
            r.v3_sum.besr_db <= th.composite_besr_db,
            "{} {:?} besr_db {:.2} > canonical threshold {:.2}",
            r.label,
            r.tier,
            r.v3_sum.besr_db,
            th.composite_besr_db
        );
        assert!(
            r.v3_sum.ltas_max_db <= th.max_ltas_db,
            "{} {:?} ltas_max_db {:.2} > canonical threshold {:.2}",
            r.label,
            r.tier,
            r.v3_sum.ltas_max_db,
            th.max_ltas_db
        );
        assert!(
            r.v3_holdout_db <= th.unseen_control_besr_db,
            "{} {:?} holdout bESR {:.2} > canonical threshold {:.2}",
            r.label,
            r.tier,
            r.v3_holdout_db,
            th.unseen_control_besr_db
        );
    }
}
