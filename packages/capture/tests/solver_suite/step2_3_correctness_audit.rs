#![forbid(unsafe_code)]
//! Step 2.3 Correctness & Isomorphism Audit Regressions (`Wide192Accum` narrow-span/mantissa
//! cancellation, subnormal/extreme-exponent `two_prod_dekker`, Boldo-Melquiond `round-to-odd`,
//! `ControlClampDescriptor` arena synchronization for `u_norm != 0` `StarCenter`s, Path-2 `W_ctrl`
//! & `CP1`/`U_int` clearing order, and `Mode 0` shortcut/fallback isomorphism).

use std::collections::BTreeMap;

use dnam_capture::CORESET_FP32_FRAMES_PER_STEM;
use dnam_capture::proj::{
    DnamProjManifest, PROJ_STAT_BYTES_ULTRA, StemProvenance, StmcTrajectoryWindow,
    VaultCaptureSpec, add_capture_with_options, build_dnam_proj_container_with_dnam_bin,
    build_proj_inpc_capsule, export_runtime_dnam, replace_capture_with_options,
    verify_dnam_bin_zero_dc,
};
use dnam_capture::solver::{
    SolverOptions, StageAConfig, StageDAuditOptions, Wide192Accum, accumulate_uplo_u_wide192,
    compute_integrated_lufs, contract_256d_gram_tiled_wide192, finalize_stage_d_and_audit,
    run_stage_a_physics_probe, solve_stage_b_backbone, solve_stage_c_knobs, uplo_u_index,
    uplo_u_packed_len,
};
use dnam_capture::stimulus::{compute_stmc_psd_88_and_spectral_4096, encode_coreset_fp32};
use dnam_engine::container::build_dnam_container_from_dnam_bin;
use dnam_engine::schema::ControlSchemaDef;
use dnam_engine::{DnamEngine, Tier};
use dnam_train::{
    Pass5EmitConfig, SonaModelF64, decode_dnam_bin_to_model, encode_model_to_dnam_bin,
};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;
use dnam_vm::bin::DnamBinView;
use dnam_vm::control::Cp1ContributorSlot;
use dnam_vm::math::DefaultMathBackend;
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{F32, F64};

use super::oracles::{clear_output_readout_weights, inject_controls_and_cp1_into_dbin};

fn populate_sona_teacher_model() -> SonaModelF64 {
    let mut model = SonaModelF64::new(48_000.0);
    model.oversample_upper_tiers_2x = false;
    model.active_fir_taps = 128;
    model.active_kautz_pairs = 32;

    for (q_idx, q) in model.core_quads.iter_mut().enumerate() {
        let (blk, local) = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q_idx];
        for lane in 0..4 {
            let phase = (((blk * 7 + local * 5 + lane * 3 + 1) % 11) as f64 - 5.0) / 5.0;
            q.svf_g[lane] = 0.08 + 0.04 * ((lane + blk) as f64);
            q.svf_k[lane] = 0.65 + 0.10 * (lane as f64);
            q.tone_lp[lane] = 0.35 + 0.08 * phase;
            q.tone_dir[lane] = 0.35;
            q.in_mid_weight[lane] = 0.30 + 0.05 * phase;
            q.in_side_weight[lane] = 0.10 * (-phase);
            q.adaa_drive[lane] = 0.85 + 0.15 * phase;
            q.lru_alpha_rel[lane] = 0.985;
            q.lru_kappa[lane] = 0.08 * phase.abs();
        }
    }
    for (t_idx, tile) in model.tiles.iter_mut().enumerate() {
        for s in 0..4 {
            for d in 0..4 {
                let phase = (((t_idx * 5 + s * 3 + d + 1) % 7) as f64 - 3.0) / 3.0;
                tile.w_odd_4x4[s][d] = 0.025 * phase;
                tile.w_even_4x4[s][d] = 0.015 * (-phase);
            }
        }
    }
    let scales = [0.30_f64, 0.18, 0.10, 0.05];
    for (m, kp) in model.kautz.iter_mut().enumerate() {
        let sc = scales[if m < 8 {
            0
        } else if m < 12 {
            1
        } else if m < 24 {
            2
        } else {
            3
        }];
        let sign = if m % 2 == 0 { 1.0 } else { -0.8 };
        kp.r_low = 0.86 + 0.003 * (m as f64);
        kp.theta = 0.08 + 0.085 * (m as f64);
        kp.b_gamma = 1.0;
        kp.c_re_l = sc * 0.18 * sign;
        kp.c_im_l = -sc * 0.12 * sign;
        kp.c_re_r = sc * 0.18 * sign;
        kp.c_im_r = -sc * 0.12 * sign;
    }

    model.pre_conv_u_gain = [0.75, 0.75];
    model.direct_s2_u_gain = [0.22, 0.22];
    let fir_ranges = [0..32_usize, 32..48, 48..96, 96..128];
    let lane_ranges = [0..16_usize, 16..24, 24..32, 32..40];
    for (t_idx, &sc) in scales.iter().enumerate() {
        for idx in fir_ranges[t_idx].clone() {
            let v = sc * 0.40 * ((((idx * 11 + t_idx * 5 + 3) % 13) as f64) - 6.0) / 6.0;
            model.fir_l[idx] = v;
            model.fir_r[idx] = v;
        }
        for l in lane_ranges[t_idx].clone() {
            for c in 0..6 {
                let p = (((l * 5 + c * 3 + t_idx) % 7) as f64 - 3.0) / 3.0;
                if t_idx == 0 {
                    model.w_ro[l][c] = 0.04 * p;
                }
                model.w_ro[l][6 + c] = sc * 0.16 * p;
            }
        }
    }
    model
}

fn make_range_ctrl(id: &str, lo: f32, hi: f32, def: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [lo, hi],
        "safe_clamped_range": [lo, hi],
        "default": def
    }))
    .unwrap()
}

fn make_spec(
    id: &str,
    is_star: bool,
    meta: BTreeMap<String, serde_json::Value>,
    wet_mono: &[f32],
    windows: Vec<StmcTrajectoryWindow>,
) -> VaultCaptureSpec {
    VaultCaptureSpec {
        stem_id: id.to_string(),
        is_star_center: is_star,
        enabled: true,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        controls_meta: meta,
        provenance: StemProvenance {
            input_pcm_sha256: "0".repeat(64),
            target_pcm_sha256: "1".repeat(64),
            num_frames: CORESET_FP32_FRAMES_PER_STEM as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows,
        band_psd_summary_88: [0u8; 88],
        n_comb_bins: 32,
        spectral_summary_4096: [0u8; 4_096],
        wet_coreset_fp32: encode_coreset_fp32(wet_mono, None),
        optional_spcm: None,
    }
}

#[test]
fn test_wide192_c_and_g_narrow_span_and_f32_mantissa_intra_batch_cancellation() {
    // verify[train.gram.wide192]
    // verify[train.gram.accuracy-contract]
    let n = 64usize;
    let mut feats = vec![[0.0_f32; 256]; n];
    let mut targets = vec![[0.0_f64; 2]; n];
    let weights = vec![1.0_f64; n];

    // 1. Cross-correlation wide_c at <= 65,536x span (2^16):
    // +1.0 + 2^-16 * (2^-16 + 2^-40) - 1.0 - 2^-32 == 2^-56
    let s16 = 2.0_f32.powi(-16);
    feats[0][2] = 1.0;
    targets[0][0] = 1.0;
    feats[1][2] = s16;
    targets[1][0] = 2.0_f64.powi(-16) + 2.0_f64.powi(-40);
    feats[2][2] = -1.0;
    targets[2][0] = 1.0;
    feats[3][2] = -s16;
    targets[3][0] = 2.0_f64.powi(-16);

    // 2. Cross-correlation wide_c at 0-bit exponent span (all magnitudes in [1.0, 2.0)):
    // 1*1 + (1 + 2^-23)*(1 + 2^-40) - 1*(1 + 2^-40) - (1 + 2^-23)*1 == 2^-63
    let u23 = 1.0_f32 + 2.0_f32.powi(-23);
    let t40 = 1.0_f64 + 2.0_f64.powi(-40);
    feats[8][5] = 1.0;
    targets[8][1] = 1.0;
    feats[9][5] = u23;
    targets[9][1] = t40;
    feats[10][5] = -1.0;
    targets[10][1] = t40;
    feats[11][5] = -u23;
    targets[11][1] = 1.0;

    // 3. Off-diagonal Gram wide_g at span = 32 (<< 65,536) with 24-bit f32 significands:
    // 1*1 + (2^-5*(1+2^-23))^2 - 1*1 - 2^-5*(2^-5*(1+2^-22)) == 2^-56
    let a_small = 2.0_f32.powi(-5) * (1.0_f32 + 2.0_f32.powi(-23));
    let b_small = 2.0_f32.powi(-5) * (1.0_f32 + 2.0_f32.powi(-22));
    feats[16][0] = 1.0;
    feats[16][9] = 1.0;
    feats[17][0] = a_small;
    feats[17][9] = a_small;
    feats[18][0] = -1.0;
    feats[18][9] = 1.0;
    feats[19][0] = -2.0_f32.powi(-5);
    feats[19][9] = b_small;

    // Same 24-bit f32 cancellation on diagonal block (col 12 vs col 13 inside same 4-lane block)
    feats[24][12] = 1.0;
    feats[24][13] = 1.0;
    feats[25][12] = a_small;
    feats[25][13] = a_small;
    feats[26][12] = -1.0;
    feats[26][13] = 1.0;
    feats[27][12] = -2.0_f32.powi(-5);
    feats[27][13] = b_small;

    let (g, c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    assert_eq!(c[2][0].to_bits(), (2.0_f64.powi(-56)).to_bits());
    assert_eq!(c[5][1].to_bits(), (2.0_f64.powi(-63)).to_bits());
    assert_eq!(
        g[uplo_u_index(0, 9)].to_bits(),
        (2.0_f64.powi(-56)).to_bits()
    );
    assert_eq!(
        g[uplo_u_index(12, 13)].to_bits(),
        (2.0_f64.powi(-56)).to_bits()
    );
}

#[test]
fn test_wide192_subnormal_extreme_exponent_and_half_ulp_tie_round_to_odd() {
    // verify[train.gram.wide192]
    // 1. Extreme exponent Dekker split (> 1.34e299 where 134_217_729.0 * a would overflow)
    let mut acc_hi = Wide192Accum::zero();
    let a_big = 2.0_f64.powi(996) * (1.0 + 2.0_f64.powi(-30));
    let b_tiny = 2.0_f64.powi(-996) * (1.0 - 2.0_f64.powi(-30));
    acc_hi.add_product(a_big, b_tiny);
    acc_hi.add_f64(-1.0);
    assert_eq!(acc_hi.to_f64().to_bits(), (-2.0_f64.powi(-60)).to_bits());

    // 2. Deep subnormal product + cancellation: +1.0 + (2^-1040 * 2^-20) - 1.0 == 2^-1060
    let mut acc_sub = Wide192Accum::zero();
    acc_sub.add_f64(1.0);
    acc_sub.add_product(2.0_f64.powi(-1040), 2.0_f64.powi(-20));
    acc_sub.add_f64(-1.0);
    assert_eq!(acc_sub.to_f64().to_bits(), (2.0_f64.powi(-1060)).to_bits());

    // 3. Boldo-Melquiond round-to-odd prevents IEEE-754 double-rounding at half-ULP ties
    let mut tie_up = Wide192Accum::zero();
    tie_up.add_f64(1.0);
    tie_up.add_f64(2.0_f64.powi(-53));
    tie_up.add_f64(2.0_f64.powi(-108));
    assert_eq!(
        tie_up.to_f64().to_bits(),
        (1.0 + 2.0_f64.powi(-52)).to_bits()
    );

    let mut tie_down = Wide192Accum::zero();
    tie_down.add_f64(1.0 + 2.0_f64.powi(-52));
    tie_down.add_f64(2.0_f64.powi(-53));
    tie_down.add_f64(-2.0_f64.powi(-108));
    assert_eq!(
        tie_down.to_f64().to_bits(),
        (1.0 + 2.0_f64.powi(-52)).to_bits()
    );

    // 4. Multi-scale triple w * fi * fj without intermediate w * fj underflow/overflow
    let mut fb = [0.0_f32; 256];
    fb[0] = 1.0e35;
    fb[3] = 1.0e-35;
    fb[1] = 1.0e-35;
    fb[4] = 1.0e35;

    let fb64: Vec<f64> = fb[..8].iter().map(|&v| f64::from(v)).collect();
    let mut acc_vec = vec![Wide192Accum::zero(); uplo_u_packed_len(8)];
    let mut acc_c = vec![[Wide192Accum::zero(); 2]; 8];
    accumulate_uplo_u_wide192(
        &mut acc_vec,
        &mut acc_c,
        &fb64,
        [f64::from(1.0e35_f32), 0.0],
        1.0e-300,
    );
    let expected_lo = 1.0e-300_f64 * (f64::from(1.0e35_f32) * f64::from(1.0e-35_f32));
    assert!((acc_vec[uplo_u_index(0, 3)].to_f64() - expected_lo).abs() <= 1.0e-314);
    assert!((acc_c[3][0].to_f64() - expected_lo).abs() <= 1.0e-314);

    let feats = vec![fb; 2];
    let targets = vec![[0.0_f64; 2]; 2];
    let weights = vec![1.0e-300_f64, 1.0e300_f64];
    let (g, _) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    let expected_hi = 1.0e300_f64 * (f64::from(1.0e-35_f32) * f64::from(1.0e35_f32));
    assert!(
        g[uplo_u_index(1, 4)].is_finite() && (g[uplo_u_index(1, 4)] - expected_hi).abs() <= 1.0e285
    );

    let feats_lo_only = vec![fb];
    let (g_lo, _) = contract_256d_gram_tiled_wide192(&feats_lo_only, &[[0.0; 2]], &[1.0e-300], 1);
    assert!((g_lo[uplo_u_index(0, 3)] - expected_lo).abs() <= 1.0e-314);
}

#[test]
#[ignore = "deferred to M23 (differentiable Stage C & repool replacing legacy FD loop)"]
fn test_non_centered_star_control_and_dynamic_w_ctrl_isomorphism_and_repool() {
    // verify[train.pipeline.isomorphism-parity]
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.inputs = 1;
    manifest.base.io_config.output_mode = "mono".to_string();
    manifest.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 8.0)];

    let n = CORESET_FP32_FRAMES_PER_STEM;
    let dry_mono: Vec<f32> = (0..n)
        .map(|i| 0.30 * ((i as f32) * 0.071).sin() + 0.18 * ((i as f32) * 0.233).cos())
        .collect();

    // Populate a valid SONA v3 physical backbone + dynamic W_ctrl row 2 col 0 = 1.25.
    let gt_model = populate_sona_teacher_model();
    let base_gt_dbin = encode_model_to_dnam_bin(&gt_model, Pass5EmitConfig::default()).unwrap();
    let mut gt_slot = Cp1ContributorSlot::zero();
    gt_slot.lanes = 2;
    gt_slot.control_idx = 0;
    gt_slot.aux = 0xFF;
    gt_slot.main_base[0][0] = 0.85;
    gt_slot.emb[0] = 1.0;
    gt_slot.v_base[0] = 0.85;

    let mut gt_manifest = manifest.clone();
    gt_manifest.base.controls_schema[0].safe_clamped_range = Some([6.0, 8.0]);
    let gt_dbin =
        inject_controls_and_cp1_into_dbin(&base_gt_dbin, &gt_manifest, &[gt_slot], |cb| {
            cb.w_ctrl[0][0] = F32::new(0.22);
            cb.w_ctrl[2][0] = F32::new(1.25);
        });
    let mut gt_hdr = gt_manifest.base.clone();
    let gt_dnam = build_dnam_container_from_dnam_bin(&mut gt_hdr, &gt_dbin).expect("valid gt dnam");
    let mut gt_eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&gt_dnam, Tier::Ultra, 48_000).unwrap();

    let render_at = |eng: &mut DnamEngine<'_, DefaultMathBackend>, ctrl_val: f32| -> Vec<f32> {
        eng.reset_to_dc_equilibrium();
        eng.set_control_immediate(0, ctrl_val);
        dry_mono
            .iter()
            .map(|&x| eng.step_frame_sona(x, x, true)[0])
            .collect()
    };
    let wet_star = render_at(&mut gt_eng, 8.0);
    let wet_lo = render_at(&mut gt_eng, 6.0);

    let mut seed_model = decode_dnam_bin_to_model(&base_gt_dbin).unwrap();
    clear_output_readout_weights(&mut seed_model);
    let seed_dbin = encode_model_to_dnam_bin(&seed_model, Pass5EmitConfig::default()).unwrap();
    // Note: seed_proj starts with safe_clamped_range = [0.0, 10.0] (u_norm(8.0) = +0.6), whereas
    // Phase-1 [CLAMP-RANGE] clamps [6.0, 8.0] (u_norm(8.0) = +1.0); vault + Stage B/C/D must sync!
    let seed_proj =
        build_dnam_proj_container_with_dnam_bin(&mut manifest.clone(), &[], &seed_dbin).unwrap();
    let mut schedule_320 = [0u8; 320];
    schedule_320[8..12].copy_from_slice(&1_024_u32.to_le_bytes());
    let inpc = build_proj_inpc_capsule(
        1,
        &[0x21; 256],
        &schedule_320,
        &encode_coreset_fp32(&dry_mono, None),
    )
    .unwrap();

    let make_win = |u_phys: f32| {
        vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }]
    };
    let star_meta = BTreeMap::from([("drive".to_string(), serde_json::json!(8.0))]);
    let lo_meta = BTreeMap::from([("drive".to_string(), serde_json::json!(6.0))]);

    let opts_preserve = SolverOptions {
        stage_a: StageAConfig {
            preserve_existing_backbone: true,
            vp_gn_steps: 1,
        },
        gram_frame_stride: 1,
        clear_dynamic_w_ctrl: false,
        ..SolverOptions::default()
    };
    let proj1 = add_capture_with_options(
        &seed_proj,
        make_spec("star", true, star_meta.clone(), &wet_star, make_win(8.0)),
        Some(&inpc),
        opts_preserve,
    )
    .unwrap();
    let proj2 = add_capture_with_options(
        &proj1,
        make_spec("lo", false, lo_meta, &wet_lo, make_win(6.0)),
        None,
        opts_preserve,
    )
    .unwrap();
    // Force Path 2 re-pool under the newly clamped [6.0, 8.0] range so Stage B harvests u_norm = +1.0
    let proj2 = replace_capture_with_options(
        &proj2,
        "star",
        make_spec("star", true, star_meta.clone(), &wet_star, make_win(8.0)),
        opts_preserve,
    )
    .unwrap();

    let exported = export_runtime_dnam(&proj2, Some(Tier::Ultra)).unwrap();
    let mut dist_eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&exported, Tier::Ultra, 48_000).unwrap();
    let pred_star = render_at(&mut dist_eng, 8.0);
    let err_e: f64 = pred_star[256..]
        .iter()
        .zip(&wet_star[256..])
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    let sig_e: f64 = wet_star[256..]
        .iter()
        .map(|&b| f64::from(b) * f64::from(b))
        .sum();
    let esr_db = 10.0 * (err_e / sig_e).log10();
    assert!(
        esr_db < -50.0,
        "non-centered StarCenter (u_norm = +0.6) isomorphism ESR must be < -50 dB, got {esr_db} dB"
    );

    // Replace StarCenter with clear_dynamic_w_ctrl = true: stale W_ctrl/CP1 must be cleared BEFORE Stage B!
    let wet_scaled: Vec<f32> = dry_mono.iter().map(|&x| 0.42 * x).collect();
    let proj3 = replace_capture_with_options(
        &proj2,
        "star",
        make_spec("star", true, star_meta, &wet_scaled, make_win(8.0)),
        SolverOptions::default(),
    )
    .unwrap();
    let mut lufs = [0.0_f32; 4];
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let b = export_runtime_dnam(&proj3, Some(tier)).unwrap();
        let mut e = DnamEngine::<DefaultMathBackend>::from_bytes_owned(&b, tier, 48_000).unwrap();
        e.reset_to_dc_equilibrium();
        e.set_control_immediate(0, 8.0);
        for &x in &dry_mono[..128] {
            let _ = e.step_frame_sona(x, x, true);
        }
        let out: Vec<[f32; 2]> = dry_mono[128..640]
            .iter()
            .map(|&x| e.step_frame_sona(x, x, true))
            .collect();
        lufs[t_idx] = compute_integrated_lufs(&out, 48_000) as f32;
    }
    assert!(lufs[3] > -55.0);
    for &lu in &lufs {
        assert!((lu - lufs[3]).abs() <= 0.05, "loudness mismatch: {lufs:?}");
    }
}

#[test]
fn test_mode0_shortcut_wdw_binary_flag_and_b_head_guards() {
    // verify[train.pipeline.isomorphism-parity]
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("tone", 0.0, 10.0, 7.5)];
    let mut model = SonaModelF64::new(48_000.0);
    model.oversample_upper_tiers_2x = false;
    model.pre_conv_u_gain = [0.35, 0.25];
    let mut dbin = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap();

    let dry_lr: Vec<[f32; 2]> = (0..512)
        .map(|i| {
            let v = 0.2 * ((i as f32) * 0.11).sin();
            [v, -0.5 * v]
        })
        .collect();
    let _sb = solve_stage_b_backbone(&manifest, &mut dbin, &dry_lr, &dry_lr, 0, 1, false).unwrap();
    let stmc_map = BTreeMap::new();
    let sc = solve_stage_c_knobs(&manifest, &mut dbin, &stmc_map, &dry_lr, 0, false).unwrap();
    let _sd = finalize_stage_d_and_audit(
        &mut manifest,
        &mut dbin,
        &dry_lr,
        Some(&dry_lr),
        None,
        0.0,
        StageDAuditOptions::PATH1_HEAD_ONLY,
    )
    .unwrap();
    assert!(
        sc.cp1_slots
            .iter()
            .all(|s| s.main_base == [[0.0; 8]; 3] && s.v_base == [0.0; 4])
    );
    assert!(verify_dnam_bin_zero_dc(&dbin).is_ok());
    assert!(
        !manifest
            .base
            .compile_clamp_descriptors()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn test_strided_bezout_gram_inner_product_consistency() {
    // verify[train.pipeline.nested-varpro-schur]
    // Strided 12,000-sample skewed-amplitude stem (`gram_frame_stride = 4`):
    // `packed_gram_256` and `STAT` diagonal must stay finite and PSD across stride phases.
    for phase_hit in [true, false] {
        let n = 12_000usize;
        let mut pulse_dry = vec![[0.0_f32; 2]; n];
        let mut pulse_wet = vec![[0.0_f32; 2]; n];
        for i in 512..n {
            let on_stride = (i - 512) % 4 == 0;
            let amp = if on_stride == phase_hit {
                0.75_f32
            } else {
                0.06_f32
            };
            let x = amp * ((i as f32) * 0.13).sin();
            pulse_dry[i] = [x, x];
            pulse_wet[i] = [0.68 * x, 0.68 * x];
        }
        let mut man = DnamProjManifest::default_reference();
        man.base.io_config.inputs = 1;
        man.base.io_config.output_mode = "mono".to_string();
        let mut dbin =
            encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default())
                .unwrap();
        let _ = run_stage_a_physics_probe(
            &man,
            &mut dbin,
            &pulse_dry,
            &pulse_wet,
            None,
            StageAConfig {
                preserve_existing_backbone: false,
                vp_gn_steps: 1,
            },
        );
        let sb =
            solve_stage_b_backbone(&man, &mut dbin, &pulse_dry, &pulse_wet, 0, 4, false).unwrap();
        assert_eq!(sb.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
        assert_eq!(sb.packed_gram_256.len(), uplo_u_packed_len(256));
        let stat_words = <[F64]>::ref_from_bytes(&sb.stat_bytes).unwrap();
        for d in 0..256 {
            let idx = uplo_u_index(d, d);
            let gram_diag = sb.packed_gram_256[idx];
            let stat_diag = stat_words[idx].get();
            assert!(
                gram_diag.is_finite()
                    && gram_diag >= 0.0
                    && stat_diag.to_bits() == gram_diag.to_bits(),
                "strided Gram/STAT diagonal [{d}] must be finite & non-negative: {gram_diag}"
            );
        }
        let sd = finalize_stage_d_and_audit(
            &mut man,
            &mut dbin,
            &pulse_dry,
            Some(&pulse_wet),
            Some(sb.prefix_esr_db),
            sb.elapsed_ms,
            StageDAuditOptions::PATH2_FULL,
        )
        .unwrap();
        assert!(sd.tier_esr_db.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_48k_window_lifting_selects_broadband_segment() {
    // verify[train.pipeline.physics-probe]
    use dnam_capture::stimulus::waveforms::{
        deterministic_seed_256, synthesize_a5_musical_transient_coreset,
    };
    use dnam_fixtures::{BlackBoxModel, Chain05PureLtiCabAndRoom};

    // 48,000-sample (`1.0 s`) Chain-05 capture where `[0..12_000]` is low-note `q1` (`< 1.5 kHz`)
    // and `[36_000..48_000]` is broadband `q4`: Stage A must select the broadband 12k window!
    let seed = deterministic_seed_256(0x48_000_C05);
    let (dry_48k, _) = synthesize_a5_musical_transient_coreset(&seed, 48_000, 1, 48_000);
    let mut cab = Chain05PureLtiCabAndRoom::new();
    cab.reset();
    let ctrls: Vec<f32> = cab
        .metadata()
        .controls
        .iter()
        .map(|c| c.default_norm)
        .collect();
    let sws: Vec<u8> = cab
        .metadata()
        .switches
        .iter()
        .map(|s| s.default_index)
        .collect();
    let mut out_l = vec![0.0_f32; 48_000];
    let mut out_r = vec![0.0_f32; 48_000];
    for (i, &x) in dry_48k.iter().enumerate() {
        let mut y = [0.0_f32; 2];
        cab.process_frame(&[x], &mut y, &ctrls, &sws);
        out_l[i] = y[0];
        out_r[i] = y[1];
    }
    let dry_lr: Vec<[f32; 2]> = dry_48k.iter().map(|&x| [x, x]).collect();
    let wet_lr: Vec<[f32; 2]> = out_l.iter().zip(&out_r).map(|(&l, &r)| [l, r]).collect();
    let mut man48 = DnamProjManifest::default_reference();
    man48.base.io_config.inputs = 1;
    man48.base.io_config.output_mode = "stereo".to_string();
    let mut dbin48 =
        encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default()).unwrap();
    let _ = run_stage_a_physics_probe(
        &man48,
        &mut dbin48,
        &dry_lr,
        &wet_lr,
        None,
        StageAConfig {
            preserve_existing_backbone: false,
            vp_gn_steps: 2,
        },
    );
    let sb48 = solve_stage_b_backbone(&man48, &mut dbin48, &dry_lr, &wet_lr, 0, 4, false).unwrap();
    let sd48 = finalize_stage_d_and_audit(
        &mut man48,
        &mut dbin48,
        &dry_lr,
        Some(&wet_lr),
        Some(sb48.prefix_esr_db),
        sb48.elapsed_ms,
        StageDAuditOptions::PATH2_FULL,
    )
    .unwrap();
    assert!(
        sd48.tier_esr_db[3] < -12.0
            && sd48.tier_esr_db[3] <= sd48.tier_esr_db[2] + 1.0e-3
            && sd48.tier_esr_db[2] <= sd48.tier_esr_db[1] + 1.0e-3
            && sd48.tier_esr_db[1] <= sd48.tier_esr_db[0] + 1.0e-3,
        "48k Chain-05 must achieve < -12 dB monotone 4-tier bESR, got {:?}",
        sd48.tier_esr_db
    );
}

fn synthesize_coreset_dry() -> (Vec<f32>, Vec<[f32; 2]>) {
    use dnam_capture::stimulus::waveforms::{
        deterministic_seed_256, synthesize_a5_musical_transient_coreset,
    };

    let seed = deterministic_seed_256(0x9988_7766);
    let (dry_l, _) = synthesize_a5_musical_transient_coreset(&seed, 12_000, 1, 48_000);
    let dry_lr: Vec<[f32; 2]> = dry_l.iter().map(|&x| [x, x]).collect();
    (dry_l, dry_lr)
}

fn assert_steep_lti_filter_besr(label: &str, dry: &[[f32; 2]], wet: &[[f32; 2]]) {
    let mut man = DnamProjManifest::default_reference();
    man.base.io_config.inputs = 1;
    man.base.io_config.output_mode = "mono".to_string();
    let mut dbin =
        encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default()).unwrap();
    let wet_mono: Vec<f32> = wet.iter().map(|w| w[0]).collect();
    let (_, spec_4096) = compute_stmc_psd_88_and_spectral_4096(&wet_mono, None);
    let sa = run_stage_a_physics_probe(
        &man,
        &mut dbin,
        dry,
        wet,
        Some(&spec_4096),
        StageAConfig {
            preserve_existing_backbone: false,
            vp_gn_steps: 2,
        },
    );
    assert!(
        sa.crossover_hz <= 105.0,
        "{label} high-passed target must lower Stage-0 sub crossover <= 105 Hz, got {}",
        sa.crossover_hz
    );
    let sb = solve_stage_b_backbone(&man, &mut dbin, dry, wet, 0, 4, false).unwrap();
    let sd = finalize_stage_d_and_audit(
        &mut man,
        &mut dbin,
        dry,
        Some(wet),
        Some(sb.prefix_esr_db),
        sb.elapsed_ms,
        StageDAuditOptions::PATH2_FULL,
    )
    .unwrap();
    assert!(
        sd.tier_esr_db[3] < -5.0
            && sd.tier_esr_db[3] <= sd.tier_esr_db[2] + 1.0e-2
            && sd.tier_esr_db[2] <= sd.tier_esr_db[1] + 1.0e-2
            && sd.tier_esr_db[1] <= sd.tier_esr_db[0] + 1.0e-2,
        "{label} must achieve < -5 dB monotone 4-tier bESR, got {:?}",
        sd.tier_esr_db
    );
}

#[test]
fn test_silent_muted_target_besr_safety() {
    // verify[train.pipeline.physics-probe]
    // verify[train.pipeline.nested-varpro-schur]
    // verify[train.pipeline.host-parity-and-export]
    let (_, dry) = synthesize_coreset_dry();
    let zero_wet = vec![[0.0_f32; 2]; 12_000];
    let mut man0 = DnamProjManifest::default_reference();
    man0.base.io_config.inputs = 1;
    man0.base.io_config.output_mode = "mono".to_string();
    let mut dbin0 =
        encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default()).unwrap();
    let _ = run_stage_a_physics_probe(
        &man0,
        &mut dbin0,
        &dry,
        &zero_wet,
        None,
        StageAConfig {
            preserve_existing_backbone: false,
            vp_gn_steps: 1,
        },
    );
    let sb0 = solve_stage_b_backbone(&man0, &mut dbin0, &dry, &zero_wet, 0, 4, false).unwrap();
    let sd0 = finalize_stage_d_and_audit(
        &mut man0,
        &mut dbin0,
        &dry,
        Some(&zero_wet),
        Some(sb0.prefix_esr_db),
        sb0.elapsed_ms,
        StageDAuditOptions::PATH2_FULL,
    )
    .unwrap();
    assert_eq!(sd0.tier_lufs, [-120.0; 4]);
    let view0 = DnamBinView::parse(&dbin0).unwrap();
    let trims = view0.arena0.crossbar.tier_gain_trim.map(F32::get);
    assert_eq!(trims, [1.0_f32; 4]);
}

#[test]
fn test_steep_brickwall_bp_besr_safety() {
    let (dry_l, dry_lr) = synthesize_coreset_dry();
    let mut bp_wet = vec![[0.0_f32; 2]; 12_000];
    let r_hp450 = (-2.0 * std::f64::consts::PI * 450.0 / 48_000.0).exp();
    let r_lp800 = (-2.0 * std::f64::consts::PI * 800.0 / 48_000.0).exp();
    let (mut s_hp8, mut s_lp8) = ([0.0_f64; 8], [0.0_f64; 8]);
    for (i, &x) in dry_l.iter().enumerate() {
        let mut v = f64::from(x);
        for st in &mut s_hp8 {
            let lp = (1.0 - r_hp450) * v + r_hp450 * *st;
            *st = lp;
            v -= lp;
        }
        for st in &mut s_lp8 {
            *st = (1.0 - r_lp800) * v + r_lp800 * *st;
            v = *st;
        }
        let bq = (3.5 * v) as f32;
        bp_wet[i] = [bq, bq];
    }
    assert_steep_lti_filter_besr("brickwall_bp_450_800", &dry_lr, &bp_wet);
}

#[test]
fn test_steep_lti_hp_3500hz_besr_safety() {
    let (dry_l, dry_lr) = synthesize_coreset_dry();
    let mut hp_wet = vec![[0.0_f32; 2]; 12_000];
    let r_hp3500 = (-2.0 * std::f64::consts::PI * 3_500.0 / 48_000.0).exp();
    let mut s_hp2 = [0.0_f64; 2];
    for (i, &x) in dry_l.iter().enumerate() {
        let mut vh = f64::from(x);
        for st in &mut s_hp2 {
            let lp = (1.0 - r_hp3500) * vh + r_hp3500 * *st;
            *st = lp;
            vh -= lp;
        }
        let hq = (1.8 * vh) as f32;
        hp_wet[i] = [hq, hq];
    }
    assert_steep_lti_filter_besr("lti_hp_3500hz", &dry_lr, &hp_wet);
}
