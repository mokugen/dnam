#![forbid(unsafe_code)]
//! Comprehensive M24 verification suite (`Cases #1..#26`) for Two-Tier `Schoukens–Pintelon`
//! co-designed excitation (`9.0 s` transient `SPCM` -> `2.0–2.5 s` coreset), 4-regime hybrid
//! sub-sample + `ppm` alignment (`< 0.02` samples), `BLA` / even-odd nonlinear separation /
//! `Law 0.A` variance / ISO 3382 Schroeder `EDC` $T_{60}$, strict `STMC_FLAG_STEREO` recapture,
//! exact integer `MODE_A_SWEEP` knots, and `< 20 MB` `16`-stem `2.5 s` stereo `.dnam-proj` budget.
// verify[train.capture.excitation.codesigned]
// verify[train.capture.alignment.multi-regime]
// verify[container.proj.inpc-capsule]
// verify[container.proj.jcs-manifest]

#[path = "schoukens_pintelon_and_hybrid_align/targets.rs"]
mod targets;

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use dnam_capture::align::{
    AlignmentRegime, constrain_subsample_around_coarse, estimate_hybrid_multiregime_alignment,
    generate_mic_pad_probe_400ms, parabolic_peak_offset, realign_fractional_kaiser64,
};
use dnam_capture::metrics::canonical_ltas_33_intervals_hz;
use dnam_capture::proj::{
    DnamProjView, ExtractedPhysics, PROJ_STAT_BYTES_ULTRA, ProjChunkInput, ProjChunkKind,
    STMC_FLAG_ENABLED, STMC_FLAG_STAR_CENTER, STMC_FLAG_STEREO, StemProvenance, StemRegistryEntry,
    StmcTrajectoryWindow, VaultCaptureSpec, build_dnam_proj_container, build_proj_inpc_capsule,
    build_proj_stmc_capsule, compact_to_capsules, decode_proj_stmc_f32, verify_proj_inpc_capsule,
    verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    build_v3_causal_probe_excitation, estimate_v3_causal_deadtime_and_polarity,
    pass1_hint_from_manifest, resolve_eval_active_len,
};
use dnam_capture::stimulus::fft::rfft_bins;
use dnam_capture::stimulus::pipeline::schoukens::{
    Regime2DecayInput, SCHOUKENS_CORESET_20S_OFFSETS, SCHOUKENS_CORESET_25S_OFFSETS,
    SCHOUKENS_PEAK_CEILING_FS, SCHOUKENS_R1_MAX_CREST_FACTOR_LIN, SCHOUKENS_R1_PERIOD_SAMPLES,
    SCHOUKENS_SPCM_CANONICAL_FRAMES, SCHOUKENS_SPCM_REGIME_OFFSETS, SchoukensAnalysisConfig,
    SpcmHeader64BWire, analyze_schoukens_pintelon_physics, build_schoukens_pintelon_coreset_48k,
    build_schoukens_pintelon_spcm_48k, compute_schroeder_edc_t60_s, condense_spcm_to_coreset,
    condense_spcm_to_coreset_for_topology, regime2_gap_ranges,
};
use dnam_capture::stimulus::waveforms::{
    compute_stmc_psd_88_and_spectral_4096, deterministic_seed_256, encode_coreset_fp32_with_frames,
    synthesize_a5_musical_transient_coreset,
};
use dnam_capture::stimulus::{
    AdaptiveTrainingOptions, TopologyProbeMode, recapture_clipped_stem, train_adaptive_target,
};
use dnam_capture::{
    CORESET_FP32_CANONICAL_25S_FRAMES, CORESET_FP32_MAX_FRAMES_PER_STEM,
    CORESET_FP32_MIN_FRAMES_PER_STEM,
};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;
use dnam_fixtures::{BlackBoxModel, SpringChirpDecorator};
use targets::{
    ConverterRigTarget, IdentityWire, base_manifest, make_range_control, make_switch_control,
};

#[test]
fn test_cases_01_05_fractional_lags_ppm_gates_noise_and_polarity_dc() {
    let seed = deterministic_seed_256(0x2401_0001);
    let exc = build_schoukens_pintelon_coreset_48k(
        &seed,
        TopologyProbeMode::Mono1To1,
        96_000,
        &BTreeSet::new(),
    );
    let dry = &exc.dry_l;

    // Case #1: Sub-sample lag recovery across 0.37, 37.38, and 143.62 samples (< 0.02 samp).
    for &lag in &[0.37_f64, 37.38_f64, 143.62_f64] {
        let wet = realign_fractional_kaiser64(dry, -lag, 0.0, false);
        let est = estimate_hybrid_multiregime_alignment(dry, &wet, 48_000, 256, None, false)
            .expect("fractional lag alignment");
        assert!(
            (est.total_lag_samples - lag).abs() < 0.02,
            "expected lag {lag} +/- 0.02, got {}",
            est.total_lag_samples
        );
    }

    // Case #2: +45 ppm clock drift + 37.38 lag -> < 0.02 samp leading & trailing residual.
    let wet_ppm = realign_fractional_kaiser64(dry, -37.38, -45.0, false);
    let est_ppm =
        estimate_hybrid_multiregime_alignment(dry, &wet_ppm, 48_000, 256, Some(37.0), false)
            .expect("+45 ppm alignment");
    assert!((est_ppm.total_lag_samples - 37.38).abs() < 0.02);
    assert!((est_ppm.clock_drift_ppm - 45.0).abs() < 2.5);
    let aligned_ppm = realign_fractional_kaiser64(
        &wet_ppm,
        est_ppm.total_lag_samples,
        est_ppm.clock_drift_ppm,
        est_ppm.polarity_inverted,
    );
    let lead = estimate_hybrid_multiregime_alignment(
        &dry[..16_384],
        &aligned_ppm[..16_384],
        48_000,
        16,
        None,
        false,
    )
    .expect("leading residual");
    let trail = estimate_hybrid_multiregime_alignment(
        &dry[79_616..],
        &aligned_ppm[79_616..],
        48_000,
        16,
        None,
        false,
    )
    .expect("trailing residual");
    assert!(
        lead.total_lag_samples.abs() < 0.02,
        "leading residual {}",
        lead.total_lag_samples
    );
    assert!(
        trail.total_lag_samples.abs() < 0.02,
        "trailing residual {}",
        trail.total_lag_samples
    );

    // Case #3: PPM blocking gates (LTV, <= 1.0 ppm deadband, > 250 ppm, low coherence).
    let ltv_blocked =
        estimate_hybrid_multiregime_alignment(dry, &wet_ppm, 48_000, 256, Some(37.0), true)
            .unwrap();
    assert_eq!(ltv_blocked.clock_drift_ppm, 0.0);

    let wet_deadband = realign_fractional_kaiser64(dry, -37.38, -0.4, false);
    let deadband_est =
        estimate_hybrid_multiregime_alignment(dry, &wet_deadband, 48_000, 256, Some(37.0), false)
            .unwrap();
    assert_eq!(deadband_est.clock_drift_ppm, 0.0);

    let wet_oor = realign_fractional_kaiser64(dry, -37.38, -320.0, false);
    let oor_est =
        estimate_hybrid_multiregime_alignment(dry, &wet_oor, 48_000, 256, Some(37.0), false)
            .unwrap();
    assert_eq!(oor_est.clock_drift_ppm, 0.0);

    let mut rng_state = 0x5555_AAAA_1234_5678_u64;
    let noisy_low_coh: Vec<f32> = wet_ppm
        .iter()
        .map(|&w| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (((rng_state >> 32) as f64) / (u32::MAX as f64) - 0.5) as f32;
            0.18 * w + 0.9 * u
        })
        .collect();
    let low_coh_est =
        estimate_hybrid_multiregime_alignment(dry, &noisy_low_coh, 48_000, 256, Some(37.0), false)
            .unwrap();
    assert_eq!(low_coh_est.clock_drift_ppm, 0.0);

    // Case #4: -50 dBFS 60 Hz hum + broadband hiss resilience.
    let wet_base = realign_fractional_kaiser64(dry, -37.38, 0.0, false);
    let hum_amp = 10.0_f64.powf(-53.0 / 20.0) * 2.0_f64.sqrt();
    let hiss_amp = 10.0_f64.powf(-53.0 / 20.0) * 12.0_f64.sqrt();
    let mut state_h = 0x9E37_79B9_7F4A_7C15_u64;
    let wet_hum_hiss: Vec<f32> = wet_base
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            state_h = state_h
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            let u = (((state_h >> 32) as f64) / (u32::MAX as f64)) - 0.5;
            let hum = hum_amp * (std::f64::consts::TAU * 60.0 * (i as f64) / 48_000.0).sin();
            (f64::from(w) + hum + hiss_amp * u) as f32
        })
        .collect();
    let hum_est =
        estimate_hybrid_multiregime_alignment(dry, &wet_hum_hiss, 48_000, 256, Some(37.0), false)
            .unwrap();
    assert!((hum_est.total_lag_samples - 37.38).abs() < 0.02);

    // Case #5: Polarity flip + DC bias (+0.08).
    let wet_inv_dc: Vec<f32> = realign_fractional_kaiser64(dry, -37.38, 0.0, true)
        .into_iter()
        .map(|w| w + 0.08)
        .collect();
    let inv_est =
        estimate_hybrid_multiregime_alignment(dry, &wet_inv_dc, 48_000, 256, Some(37.0), false)
            .unwrap();
    assert_eq!(inv_est.regime, AlignmentRegime::PolarityInverted);
    assert!(inv_est.polarity_inverted);
    assert!((inv_est.total_lag_samples - 37.38).abs() < 0.02);
}

#[test]
fn test_cases_06_08_dispersive_spring_nonlinear_rectifier_and_plateau() {
    // Case #6 & #7: SpringChirpDecorator causal tau_coarse lock + DispersiveAllPass regime.
    let probe = build_v3_causal_probe_excitation();
    let probe_l: Vec<f32> = probe.iter().map(|lr| lr[0]).collect();
    let mut spring = SpringChirpDecorator::new(IdentityWire);
    spring.reset();
    let delay = 24_usize;
    let mut delayed_probe_wet = vec![0.0_f32; probe_l.len()];
    for (i, &x) in probe_l.iter().enumerate() {
        let mut out = [0.0_f32];
        spring.process_frame(&[x], &mut out, &[], &[]);
        if i + delay < delayed_probe_wet.len() {
            delayed_probe_wet[i + delay] = (out[0] - 0.75 * x) * 4.0;
        }
    }
    let (tau_coarse, _) = estimate_v3_causal_deadtime_and_polarity(&probe_l, &delayed_probe_wet);
    assert!(
        (tau_coarse - (delay as f64)).abs() <= 2.0,
        "tau_coarse={tau_coarse}"
    );

    // Pure SpringChirpDecorator 16-stage all-pass output + 24.0 sample delay -> DispersiveAllPass.
    let (dry_16k, _) = synthesize_a5_musical_transient_coreset(&[0x42_u8; 256], 16_384, 1, 48_000);
    let delayed_24 = realign_fractional_kaiser64(&dry_16k, -(delay as f64), 0.0, false);
    spring.reset();
    let mut ap_wet = Vec::with_capacity(delayed_24.len());
    for &x in &delayed_24 {
        let mut out = [0.0_f32];
        spring.process_frame(&[x], &mut out, &[], &[]);
        ap_wet.push((out[0] - 0.75 * x) * 4.0);
    }
    let disp = estimate_hybrid_multiregime_alignment(
        &dry_16k,
        &ap_wet,
        48_000,
        256,
        Some(tau_coarse),
        false,
    )
    .expect("DispersiveAllPass alignment");
    assert_eq!(disp.regime, AlignmentRegime::DispersiveAllPass);
    assert!(
        (disp.total_lag_samples - (delay as f64)).abs() <= 1.5,
        "disp lag={}",
        disp.total_lag_samples
    );

    // Case #8a: Hard rectifier / fuzz destroys fundamental coherence (< 0.35) -> NonlinearEnvelopeFallback.
    let wet_nl: Vec<f32> = realign_fractional_kaiser64(&dry_16k, -37.38, 0.0, false)
        .into_iter()
        .map(|x| x.abs() * (2.0 * x.abs() + 0.1))
        .collect();
    let nl =
        estimate_hybrid_multiregime_alignment(&dry_16k, &wet_nl, 48_000, 256, Some(37.0), false)
            .expect("NonlinearEnvelopeFallback");
    assert_eq!(nl.regime, AlignmentRegime::NonlinearEnvelopeFallback);
    assert!((nl.total_lag_samples - 37.38).abs() <= 1.5);

    // Case #8b: Parabolic plateau guard & +/-1.5 tau_coarse bounding.
    assert_eq!(parabolic_peak_offset(0.95, 0.95, 0.95), 0.0);
    assert!((constrain_subsample_around_coarse(42.0, 0.4, 37.0) - 38.5).abs() < 1e-12);
    assert!((constrain_subsample_around_coarse(30.0, -0.4, 37.0) - 35.5).abs() < 1e-12);
    assert_eq!(constrain_subsample_around_coarse(0.1, -0.5, 0.0), 0.0);
}

#[test]
fn test_cases_09_18_schoukens_regimes_bla_even_odd_law0a_edc_and_topologies() {
    let seed = deterministic_seed_256(0x2402_0002);
    // Case #9: CF <= 1.8, -1.5 dBFS ceiling, Law 0.C mains exclusion, 2.0/2.5/9.0 s offsets.
    let spcm =
        build_schoukens_pintelon_spcm_48k(&seed, TopologyProbeMode::Mono1To1, &BTreeSet::new());
    assert_eq!(spcm.dry_l.len(), SCHOUKENS_SPCM_CANONICAL_FRAMES);
    assert_eq!(spcm.regime_offsets, SCHOUKENS_SPCM_REGIME_OFFSETS);
    let r1 = &spcm.r1_period_l;
    let peak = r1.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let rms =
        (r1.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / (r1.len() as f64)).sqrt();
    assert!(peak <= (SCHOUKENS_PEAK_CEILING_FS as f32) + 1e-5);
    assert!((f64::from(peak) / rms) <= SCHOUKENS_R1_MAX_CREST_FACTOR_LIN + 1e-3);

    let bin_hz = 48_000.0 / (SCHOUKENS_R1_PERIOD_SAMPLES as f64);
    for &(b, _) in &spcm.r1_plan.excited_odd_bins {
        let f = (b as f64) * bin_hz;
        for h in 1..=8 {
            assert!((f - 50.0 * (h as f64)).abs() > bin_hz * 1.5);
            assert!((f - 60.0 * (h as f64)).abs() > bin_hz * 1.5);
        }
    }
    let dry_spec = rfft_bins(r1);
    let excited_energy: f64 = spcm
        .r1_plan
        .excited_odd_bins
        .iter()
        .map(|&(b, _)| dry_spec[b].norm_sqr())
        .sum();
    let guard_energy: f64 = spcm
        .r1_plan
        .omitted_odd_bins
        .iter()
        .map(|&b| dry_spec[b].norm_sqr())
        .sum();
    assert!(
        guard_energy / excited_energy.max(1.0e-18) < 1.0e-10,
        "guard leakage {} vs excited {}",
        guard_energy,
        excited_energy
    );

    let core_96k = build_schoukens_pintelon_coreset_48k(
        &seed,
        TopologyProbeMode::Mono1To1,
        96_000,
        &BTreeSet::new(),
    );
    let core_120k = build_schoukens_pintelon_coreset_48k(
        &seed,
        TopologyProbeMode::Mono1To1,
        120_000,
        &BTreeSet::new(),
    );
    assert_eq!(core_96k.regime_offsets, SCHOUKENS_CORESET_20S_OFFSETS);
    assert_eq!(core_120k.regime_offsets, SCHOUKENS_CORESET_25S_OFFSETS);

    // Case #10: BLA 33-band + Even (x^2) vs Odd (x^3) separation.
    let y_even: Vec<f32> = r1
        .iter()
        .map(|&x| (0.8 * f64::from(x) + 0.25 * f64::from(x).powi(2)) as f32)
        .collect();
    let y_odd: Vec<f32> = r1
        .iter()
        .map(|&x| (0.8 * f64::from(x) - 0.30 * f64::from(x).powi(3)) as f32)
        .collect();
    let res_even = analyze_schoukens_pintelon_physics(
        &spcm.r1_plan,
        r1,
        &[&y_even, &y_even],
        None,
        Regime2DecayInput::default(),
        SchoukensAnalysisConfig::default(),
    );
    let res_odd = analyze_schoukens_pintelon_physics(
        &spcm.r1_plan,
        r1,
        &[&y_odd, &y_odd],
        None,
        Regime2DecayInput::default(),
        SchoukensAnalysisConfig::default(),
    );
    assert_eq!(
        res_even.bla_mag_33_db.len(),
        canonical_ltas_33_intervals_hz().len()
    );
    assert!(!res_even.law0a_ltv_flag && !res_odd.law0a_ltv_flag);
    assert!(res_even.even_dist_ratio_db > res_even.odd_dist_ratio_db + 12.0);
    assert!(res_odd.odd_dist_ratio_db > res_odd.even_dist_ratio_db + 12.0);

    // Case #11: Law 0.A LTV detection & unexcited guard-bin fallback.
    let mut y_ltv_p1 = Vec::with_capacity(SCHOUKENS_R1_PERIOD_SAMPLES);
    let mut y_ltv_p2 = Vec::with_capacity(SCHOUKENS_R1_PERIOD_SAMPLES);
    for i in 0..SCHOUKENS_R1_PERIOD_SAMPLES {
        let x = f64::from(r1[i]);
        let g1 = 1.0 + 0.35 * (std::f64::consts::TAU * 2.1 * (i as f64) / 48_000.0).sin();
        let g2 = 1.0
            + 0.35
                * (std::f64::consts::TAU * 2.1 * ((i + SCHOUKENS_R1_PERIOD_SAMPLES) as f64)
                    / 48_000.0)
                    .sin();
        y_ltv_p1.push((g1 * x) as f32);
        y_ltv_p2.push((g2 * x) as f32);
    }
    let res_ltv = analyze_schoukens_pintelon_physics(
        &spcm.r1_plan,
        r1,
        &[&y_ltv_p1, &y_ltv_p2],
        None,
        Regime2DecayInput::default(),
        SchoukensAnalysisConfig::default(),
    );
    assert!(res_ltv.law0a_ltv_flag);
    assert_eq!(res_ltv.noise_var_33_lin.len(), 33);
    assert!(
        res_ltv
            .noise_var_33_lin
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0)
    );
    let total_nv_ltv: f64 = res_ltv.noise_var_33_lin.iter().map(|&v| f64::from(v)).sum();
    assert!(
        total_nv_ltv > 0.0 && total_nv_ltv < 1.0e-5,
        "LTV variance out of expected range: {total_nv_ltv}"
    );

    // Case #12: ISO 3382 Schroeder EDC T60 + stationary-noise rejection + Pass1PhysicsHint precedence.
    let mut gap_buf = vec![0.0_f32; 60_000];
    let t60_true = 1.2_f64;
    let tau = (-6.907_755_278_982_137 / (t60_true * 48_000.0)).exp();
    let mut amp = 0.25_f64;
    for slot in &mut gap_buf[5_000..45_000] {
        *slot = amp as f32;
        amp *= tau;
    }
    let edc_est = compute_schroeder_edc_t60_s(&gap_buf, None, &[(5_000, 45_000)], 48_000, 1.0e-10);
    assert!(
        (f64::from(edc_est) - t60_true).abs() < 0.25,
        "edc_est={edc_est}"
    );
    let mut rng_ns = 0x9988_7766_5544_3322_u64;
    let mut stat_noise = vec![0.0_f32; 40_800];
    for s in &mut stat_noise[28_800..40_800] {
        rng_ns = rng_ns
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s = (0.02 * (((rng_ns >> 32) as f64) / f64::from(u32::MAX) - 0.5)) as f32;
    }
    assert_eq!(
        compute_schroeder_edc_t60_s(
            &stat_noise,
            None,
            &regime2_gap_ranges(40_800),
            48_000,
            1.0e-10
        ),
        0.0
    );

    let mut manifest = base_manifest(true);
    let mut legacy_meta = BTreeMap::new();
    legacy_meta.insert("phys_odd_harm".to_string(), serde_json::json!(0.8));
    legacy_meta.insert("phys_even_harm".to_string(), serde_json::json!(0.8));
    manifest.stems_registry.push(StemRegistryEntry {
        stem_id: "star_center".to_string(),
        chunk_id: 0,
        enabled: true,
        is_star_center: true,
        controls_meta: BTreeMap::new(),
        provenance: StemProvenance {
            input_pcm_sha256: "0".repeat(64),
            target_pcm_sha256: "0".repeat(64),
            num_frames: 120_000,
        },
        user_metadata: legacy_meta,
        extracted_physics: Some(ExtractedPhysics {
            bla_mag_33_db: res_odd.bla_mag_33_db,
            even_dist_ratio_db: -45.0,
            odd_dist_ratio_db: -18.0,
            noise_var_33_lin: res_ltv.noise_var_33_lin,
            edc_t60_s: edc_est,
            law0a_ltv_flag: true,
            coreset_regime_offsets: SCHOUKENS_CORESET_25S_OFFSETS,
            ..Default::default()
        }),
    });
    let hint = pass1_hint_from_manifest(&manifest);
    assert!((hint.rt60_s.unwrap() - f64::from(edc_est)).abs() < 1e-4);
    assert!(hint.thd_ratio.unwrap() > 0.1 && hint.thd_ratio.unwrap() < 0.2);
    assert!(hint.even_odd_asymmetry.unwrap() < -0.5);
    assert!(hint.noise_floor_db.is_some());
    assert_eq!(hint.modulation_rate_hz, Some(1.0));

    // Cases #13-#18: Regimes 2 staircase/gaps, full-band R3 sweeps across SPCM condensation,
    // all 6 REQ-7.4 contrast zones, & independent L/R polarity cross-feed alignment.
    let [_, o2, o3, o4, _] = SCHOUKENS_CORESET_25S_OFFSETS;
    let c25_dry = condense_spcm_to_coreset_for_topology(
        &spcm.dry_l,
        None,
        CORESET_FP32_MAX_FRAMES_PER_STEM,
        TopologyProbeMode::Mono1To1,
    )
    .unwrap()
    .0;
    let r2_pk0 = c25_dry[32_768..32_768 + 3_600]
        .iter()
        .fold(0.0_f32, |m, &v| m.max(v.abs()));
    let r2_pk3 = c25_dry[55_568..61_568]
        .iter()
        .fold(0.0_f32, |m, &v| m.max(v.abs()));
    assert!(
        r2_pk3 > r2_pk0 * 40.0,
        "staircase pk0={r2_pk0} pk3={r2_pk3}"
    );
    assert!(
        c25_dry[(o2 as usize + 28_800)..(o3 as usize)]
            .iter()
            .all(|&v| v == 0.0)
    );
    for (condensed, ro) in [
        (&c25_dry, SCHOUKENS_CORESET_25S_OFFSETS),
        (
            &condense_spcm_to_coreset(&spcm.dry_l, None, CORESET_FP32_MIN_FRAMES_PER_STEM)
                .unwrap()
                .0,
            SCHOUKENS_CORESET_20S_OFFSETS,
        ),
    ] {
        let mut r3_padded = vec![0.0_f32; 32_768];
        let r3_slice = &condensed[ro[2] as usize..ro[3] as usize];
        let take = r3_slice.len().min(r3_padded.len());
        r3_padded[..take].copy_from_slice(&r3_slice[..take]);
        let r3_sp = rfft_bins(&r3_padded);
        let lf_e: f64 = r3_sp[38..205].iter().map(|c| c.norm_sqr()).sum();
        let hf_e: f64 = r3_sp[1_365..10_922].iter().map(|c| c.norm_sqr()).sum();
        assert!(
            lf_e > 1.0 && hf_e > 1.0,
            "full-band R3 missing octaves: lf={lf_e} hf={hf_e}"
        );
    }
    assert_eq!(o4, 99_968);
    for topo in [
        TopologyProbeMode::Mono1To1,
        TopologyProbeMode::MonoToStereo1To2,
        TopologyProbeMode::TrueStereo2To2,
        TopologyProbeMode::DualInToMono2To1,
        TopologyProbeMode::FourCableMethod4Cm,
        TopologyProbeMode::WetDryWet1To3,
    ] {
        let t_exc = build_schoukens_pintelon_coreset_48k(&seed, topo, 120_000, &BTreeSet::new());
        assert_eq!(t_exc.dry_l.len(), 120_000);
        let r4_s = t_exc.regime_offsets[3] as usize;
        let r4_e = t_exc.regime_offsets[4] as usize;
        if topo == TopologyProbeMode::TrueStereo2To2 {
            let r = t_exc.dry_r.as_ref().unwrap();
            let z = (r4_e - r4_s) / 5;
            assert!(r[r4_s..r4_s + z].iter().all(|&v| v == 0.0));
            assert!(
                t_exc.dry_l[r4_s + z..r4_s + 2 * z]
                    .iter()
                    .all(|&v| v == 0.0)
            );
        } else if topo == TopologyProbeMode::DualInToMono2To1 {
            let r = t_exc.dry_r.as_ref().unwrap();
            let q = (r4_e - r4_s) / 4;
            assert!(r[r4_s..r4_s + q].iter().all(|&v| v == 0.0));
            assert!(r[r4_s + q..r4_s + 2 * q].iter().any(|&v| v.abs() > 0.1));
        } else if topo == TopologyProbeMode::FourCableMethod4Cm {
            let a3 = t_exc.aux_in_3.as_ref().unwrap();
            let t = (r4_e - r4_s) / 3;
            assert!(a3[r4_s..r4_s + t].iter().all(|&v| v == 0.0));
            assert!(
                t_exc.dry_l[r4_s + t..r4_s + 2 * t]
                    .iter()
                    .all(|&v| v == 0.0)
            );
        }
    }

    // Independent stereo polarity (out_1 = +x, out_2 = -x) with L/R cross-feed recovers both polarities cleanly.
    let dry_96k = &core_96k.dry_l;
    let delayed_cross = realign_fractional_kaiser64(dry_96k, -12.25, 0.0, false);
    let wet_l: Vec<f32> = delayed_cross.iter().map(|&v| 0.72 * v).collect();
    let wet_r: Vec<f32> = delayed_cross.iter().map(|&v| -0.68 * v).collect();
    let al_l =
        estimate_hybrid_multiregime_alignment(dry_96k, &wet_l, 48_000, 128, Some(12.0), false)
            .unwrap();
    let al_r =
        estimate_hybrid_multiregime_alignment(dry_96k, &wet_r, 48_000, 128, Some(12.0), false)
            .unwrap();
    assert!(!al_l.polarity_inverted && al_r.polarity_inverted);
    let rec_l = realign_fractional_kaiser64(
        &wet_l,
        al_l.total_lag_samples,
        al_l.clock_drift_ppm,
        al_l.polarity_inverted,
    );
    let rec_r = realign_fractional_kaiser64(
        &wet_r,
        al_l.total_lag_samples,
        al_l.clock_drift_ppm,
        al_r.polarity_inverted,
    );
    let dot_l: f64 = dry_96k
        .iter()
        .zip(&rec_l)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let dot_r: f64 = dry_96k
        .iter()
        .zip(&rec_r)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    assert!(
        dot_l > 0.0 && dot_r > 0.0,
        "aligned L & un-inverted R must both positively correlate (dl={dot_l}, dr={dot_r})"
    );
}

#[test]
fn test_cases_19_25_spcm_tier1_compaction_recapture_sweeps_and_e2e() {
    // Cases #19-#25: Mono1To1 E2E (SPCM Tier-1 9.0s -> 2.5s compaction, +45 ppm, Switch stem
    // without 8,192/72k cap, strict STMC_FLAG_STEREO mismatch/length guards + Path-1 recapture).
    let mut mono_m = base_manifest(true);
    mono_m.base.controls_schema = vec![make_switch_control(
        "voicing",
        &["bright", "dark"],
        "bright",
    )];
    let mut mono_t = ConverterRigTarget {
        lag_samples: 37.38,
        clock_drift_ppm: 45.0,
        invert_l: true,
        stereo_out: false,
        ..Default::default()
    };
    let rep = train_adaptive_target(
        mono_m,
        &mut mono_t,
        &AdaptiveTrainingOptions {
            topology_mode: TopologyProbeMode::Mono1To1,
            fixed_duration: true,
            max_frames_per_stem: 6,
            run_live_null_test: false,
            use_spcm_tier1: true,
            coreset_frames_48k: CORESET_FP32_CANONICAL_25S_FRAMES,
            ..AdaptiveTrainingOptions::default()
        },
    )
    .expect("Tier-1 SPCM + 45 ppm + Mono1To1 train_adaptive_target");

    let crypto = DefaultCryptoProvider::default();
    let pre_view = DnamProjView::from_bytes(&rep.proj_bytes, &crypto).unwrap();
    let spcm_slice = pre_view
        .chunk_slice(ProjChunkKind::Spcm, 0)
        .expect("SPCM present when use_spcm_tier1=true");
    assert_eq!(spcm_slice.len(), 64 + SCHOUKENS_SPCM_CANONICAL_FRAMES * 4);
    assert_eq!(
        &spcm_slice[0..4],
        &SpcmHeader64BWire {
            magic: *b"SPCM",
            reserved: [0; 60]
        }
        .magic
    );
    // Persisted SPCM audio must be zero-lag & polarity-aligned (not lagged by 37.38 / inverted).
    let star_inpc_0 =
        verify_proj_inpc_capsule(pre_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
    let (inpc_dry_0, _) = dnam_capture::proj::decode_proj_inpc_f32(&star_inpc_0);
    let spcm_p1: Vec<f32> = spcm_slice[64 + 16_384 * 4..64 + 32_768 * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    let (mut sp_dot, mut sp_ed, mut sp_ew) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&d, &w) in inpc_dry_0[..16_384].iter().zip(&spcm_p1) {
        sp_dot += f64::from(d) * f64::from(w);
        sp_ed += f64::from(d) * f64::from(d);
        sp_ew += f64::from(w) * f64::from(w);
    }
    let sp_coh = sp_dot / (sp_ed * sp_ew).sqrt().max(1.0e-18);
    assert!(
        sp_coh > 0.90,
        "SPCM payload must be aligned & un-inverted (coh={sp_coh})"
    );

    let star_entry = pre_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "star_center")
        .unwrap();
    let ep = star_entry.extracted_physics.as_ref().unwrap();
    assert_eq!(ep.coreset_regime_offsets, SCHOUKENS_CORESET_25S_OFFSETS);
    assert!(!ep.law0a_ltv_flag, "LTI +45 ppm must NOT trip Law 0.A LTV");
    assert_eq!(ep.bla_mag_33_db.len(), 33);

    let stmc_star =
        verify_proj_stmc_capsule(pre_view.chunk_slice(ProjChunkKind::Stmc, 0).unwrap()).unwrap();
    assert!(
        (stmc_star.fractional_lag_samples_f32 - 37.38).abs() < 0.08,
        "lag={}",
        stmc_star.fractional_lag_samples_f32
    );
    assert_eq!(
        star_entry.user_metadata.get("wet_polarity_inverted"),
        Some(&serde_json::json!(true))
    );
    assert_eq!(
        resolve_eval_active_len(&pre_view.manifest, &[[0.0_f32; 2]; 120_000], None, 120_000),
        120_000
    );
    for (idx, entry) in pre_view.manifest.stems_registry.iter().enumerate() {
        let sv = verify_proj_stmc_capsule(
            pre_view
                .chunk_slice(ProjChunkKind::Stmc, entry.chunk_id)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            decode_proj_stmc_f32(&sv).0.len(),
            120_000,
            "stem {} ({idx}) must retain full 120k coreset without 8,192 cap",
            entry.stem_id
        );
    }

    let t0 = Instant::now();
    let compacted = compact_to_capsules(&rep.proj_bytes).expect("compact_to_capsules");
    let compact_ms = t0.elapsed().as_secs_f64() * 1_000.0;
    assert!(
        compact_ms < 2_000.0,
        "compact_to_capsules took {compact_ms:.2} ms"
    );
    let comp_view = DnamProjView::from_bytes(&compacted, &crypto).unwrap();
    assert!(comp_view.chunk_slice(ProjChunkKind::Spcm, 0).is_none());
    let comp_stmc =
        verify_proj_stmc_capsule(comp_view.chunk_slice(ProjChunkKind::Stmc, 0).unwrap()).unwrap();
    assert_eq!(decode_proj_stmc_f32(&comp_stmc).0.len(), 120_000);
    let comp_inpc =
        verify_proj_inpc_capsule(comp_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
    assert_eq!(comp_inpc.num_frames, 120_000);

    // Strict recapture_clipped_stem channel-count/length validation & fast Path-1 update on non-star stem.
    let sw_entry = comp_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.stem_id == "switch_0_opt_1")
        .expect("switch_0_opt_1 stem present");
    let stmc_sw = verify_proj_stmc_capsule(
        comp_view
            .chunk_slice(ProjChunkKind::Stmc, sw_entry.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let (wl_sw, _) = decode_proj_stmc_f32(&stmc_sw);
    let probe = generate_mic_pad_probe_400ms(48_000);
    let padded: Vec<f32> = probe.iter().map(|&v| 0.316_227_76 * v).collect();
    let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(&wl_sw, None);
    let mut bad_spec = VaultCaptureSpec {
        stem_id: "star_center".to_string(),
        is_star_center: true,
        enabled: true,
        is_stereo: false,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        controls_meta: star_entry.controls_meta.clone(),
        provenance: star_entry.provenance.clone(),
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows: stmc_star.windows.clone(),
        band_psd_summary_88: psd_88,
        n_comb_bins: 64,
        spectral_summary_4096: spec_4096,
        wet_coreset_fp32: vec![0u8; 120_000 * 8], // stereo byte count while is_stereo=false
        optional_spcm: None,
    };
    assert!(matches!(
        recapture_clipped_stem(&compacted, "star_center", &probe, &padded, bad_spec.clone()),
        Err(DnamError::SchemaValidationError(_))
    ));
    bad_spec.is_stereo = true; // STMC_FLAG_STEREO mismatch against mono target stem
    assert!(matches!(
        recapture_clipped_stem(&compacted, "star_center", &probe, &padded, bad_spec.clone()),
        Err(DnamError::SchemaValidationError(_))
    ));
    bad_spec.is_stereo = false;
    bad_spec.stem_id = sw_entry.stem_id.clone();
    bad_spec.is_star_center = false;
    let nonzero_short: Vec<u8> = vec![0.15_f32; 48_000]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    bad_spec.wet_coreset_fp32 = nonzero_short; // non-zero payload < 96,000 must fail upfront
    match recapture_clipped_stem(
        &compacted,
        &sw_entry.stem_id,
        &probe,
        &padded,
        bad_spec.clone(),
    ) {
        Err(DnamError::SchemaValidationError(msg)) => {
            assert!(msg.contains("invalid coreset frame count"), "msg={msg}");
        }
        other => panic!("expected invalid coreset frame count rejection, got {other:?}"),
    }
    bad_spec.wet_coreset_fp32 = vec![0.1_f32; 96_008]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect(); // non-multiple of 16 must fail upfront
    assert!(matches!(
        recapture_clipped_stem(&compacted, &sw_entry.stem_id, &probe, &padded, bad_spec.clone()),
        Err(DnamError::SchemaValidationError(msg)) if msg.contains("invalid coreset frame count")
    ));
    let sw_spec = VaultCaptureSpec {
        stem_id: sw_entry.stem_id.clone(),
        is_star_center: false,
        enabled: true,
        is_stereo: false,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: stmc_sw.fractional_lag_samples_f32,
        controls_meta: sw_entry.controls_meta.clone(),
        provenance: sw_entry.provenance.clone(),
        user_metadata: sw_entry.user_metadata.clone(),
        extracted_physics: None,
        windows: stmc_sw.windows.clone(),
        band_psd_summary_88: psd_88,
        n_comb_bins: 64,
        spectral_summary_4096: spec_4096,
        wet_coreset_fp32: stmc_sw.wet_coreset_fp32.to_vec(),
        optional_spcm: None,
    };
    let (_, cal) = recapture_clipped_stem(&compacted, &sw_entry.stem_id, &probe, &padded, sw_spec)
        .expect("matching 2.5s mono recapture on switch stem");
    assert!((cal.pad_attenuation_db - (-10.0)).abs() <= 0.1);

    let dry_exc = build_schoukens_pintelon_spcm_48k(
        &deterministic_seed_256(7),
        TopologyProbeMode::Mono1To1,
        &BTreeSet::new(),
    );
    let (c20, _, o20) =
        condense_spcm_to_coreset(&dry_exc.dry_l, None, CORESET_FP32_MIN_FRAMES_PER_STEM).unwrap();
    assert_eq!(c20.len(), CORESET_FP32_MIN_FRAMES_PER_STEM);
    assert_eq!(o20, SCHOUKENS_CORESET_20S_OFFSETS);
}

#[test]
fn test_case_26_sixteen_stem_25s_stereo_proj_container_under_20mb() {
    let seed = deterministic_seed_256(0x2426);
    let n_frames = CORESET_FP32_MAX_FRAMES_PER_STEM;
    let dry: Vec<f32> = (0..n_frames)
        .map(|i| 0.35 * (std::f32::consts::TAU * 440.0 * (i as f32) / 48_000.0).sin())
        .collect();
    let stereo_fp32 = encode_coreset_fp32_with_frames(n_frames, &dry, Some(&dry));
    let inpc_chunk = build_proj_inpc_capsule(2, &seed, &[0u8; 320], &stereo_fp32).unwrap();
    let (psd_88, spec_4096) = compute_stmc_psd_88_and_spectral_4096(&dry, Some(&dry));

    let mut manifest = base_manifest(false);
    manifest.base.controls_schema = vec![make_range_control("gain", 0.0, 1.0, 0.5)];
    let mut chunks = vec![
        ProjChunkInput {
            kind: ProjChunkKind::Inpc,
            chunk_id: 0,
            payload: inpc_chunk,
        },
        ProjChunkInput {
            kind: ProjChunkKind::Stat,
            chunk_id: 0,
            payload: vec![0u8; PROJ_STAT_BYTES_ULTRA],
        },
    ];
    let win = vec![StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0,
        u_phys: 0.5,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }];
    for stem_idx in 0..16_u16 {
        let stem_flags = if stem_idx == 0 {
            STMC_FLAG_ENABLED | STMC_FLAG_STEREO | STMC_FLAG_STAR_CENTER
        } else {
            STMC_FLAG_ENABLED | STMC_FLAG_STEREO
        };
        let stmc = build_proj_stmc_capsule(
            stem_idx,
            stem_flags,
            1.0,
            0.0,
            &psd_88,
            64,
            &win,
            &spec_4096,
            &stereo_fp32,
        )
        .unwrap();
        chunks.push(ProjChunkInput {
            kind: ProjChunkKind::Stmc,
            chunk_id: u32::from(stem_idx),
            payload: stmc,
        });
        manifest.stems_registry.push(StemRegistryEntry {
            stem_id: if stem_idx == 0 {
                "star_center".to_string()
            } else {
                format!("stem_{stem_idx}")
            },
            chunk_id: u32::from(stem_idx),
            enabled: true,
            is_star_center: stem_idx == 0,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: "0".repeat(64),
                target_pcm_sha256: "0".repeat(64),
                num_frames: n_frames as u64,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        });
    }
    let proj_bytes = build_dnam_proj_container(&manifest, &chunks).unwrap();
    assert!(
        proj_bytes.len() + 262_144 < 20 * 1024 * 1024,
        "16-stem 2.5 s stereo .dnam-proj + 256 KB WGHT/CALB headroom must stay < 20 MB (got {} bytes)",
        proj_bytes.len()
    );
    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(&proj_bytes, &crypto).unwrap();
    assert_eq!(view.manifest.stems_registry.len(), 16);
}
