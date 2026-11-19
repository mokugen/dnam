#![forbid(unsafe_code)]
//! Step 2.3 Adversarial & Edge-Case Regression Suite (`dnam_capture::solver`, `dnam_capture::proj`,
//! `dnam_capture::stimulus::null_test`, `dnam_capture::whitebox`, and `dnam_engine`).

#[path = "step2_3_regressions/audit_and_null_test.rs"]
mod audit_and_null_test;

use std::collections::BTreeMap;

use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkKind, StemProvenance, StmcTrajectoryWindow,
    VaultCaptureSpec, add_capture, add_capture_with_options, build_dnam_proj_container_with_cp1,
    build_proj_inpc_capsule, remove_capture_with_options, replace_capture_with_options,
    set_capture_enabled_with_options,
};
use dnam_capture::solver::{
    SolverOptions, StageAConfig, contract_256d_gram_tiled_wide192, run_stage_a_physics_probe,
    uplo_u_index, write_inline_cp1_if_fits,
};
use dnam_capture::stimulus::encode_coreset_fp32;
use dnam_capture::{CORESET_FP32_FRAMES_PER_STEM, DistillationTarget};
use dnam_engine::container::{
    DnamContainerView, build_dnam_container_with_cp1, strip_container_to_tier,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamJsonHeader};
use dnam_engine::stages::weights::{
    ARENA_ULTRA_BYTES, ARENA0_DIRECT_READOUT_OFFSET, ARENA0_TIER_GAIN_TRIM_OFFSET,
    ARENA0_W_CTRL_OFFSET, ARENA1_DIRECT_READOUT_OFFSET, ARENA2_DIRECT_READOUT_OFFSET,
    ARENA3_DIRECT_READOUT_OFFSET, ARENA3_UINT_ULTRA_OFFSET, ARENA3_W_CTRL_ULTRA_OFFSET,
    pack_v3_matryoshka_to_arena,
};
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};
use dnam_engine::{DnamEngine, Tier};

pub(super) fn write_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

pub(super) fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

pub(super) fn make_range_ctrl(id: &str, lo: f32, hi: f32, def: f32) -> ControlSchemaDef {
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

pub(super) fn make_spec(
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
fn test_wide192_tiled_intra_and_cross_batch_cancellation_and_prime_stride() {
    // verify[train.gram.wide192]
    // verify[train.gram.accuracy-contract]
    let tiny_f32 = 2.0_f32.powi(-30);
    let tiny_f64 = f64::from(tiny_f32);
    let big_f32 = 1.0e8_f32;

    let n = 131usize;
    let mut feats = vec![[0.0_f32; 256]; n];
    let mut targets = vec![[0.0_f64; 2]; n];
    let weights = vec![1.0_f64; n];

    // Pair (0, 7) intra-batch [1e8, 2^-30, -1e8]
    feats[0][0] = big_f32;
    feats[0][7] = big_f32;
    feats[1][0] = tiny_f32;
    feats[1][7] = 1.0;
    feats[2][0] = -big_f32;
    feats[2][7] = big_f32;

    // Pair (3, 19) cross-batch [1e8, 2^-30] in batch 0 vs [-1e8] in batch 1
    feats[10][3] = big_f32;
    feats[10][19] = big_f32;
    feats[11][3] = tiny_f32;
    feats[11][19] = 1.0;
    feats[74][3] = -big_f32;
    feats[74][19] = big_f32;

    // Pair (5, 22) intra-batch [1.0, 2^-30, -1.0] + target cross-vector cancellation
    feats[20][5] = 1.0;
    feats[20][22] = 1.0;
    targets[20] = [1.0e8, 1.0];
    feats[21][5] = tiny_f32;
    feats[21][22] = 1.0;
    targets[21] = [1.0, tiny_f64];
    feats[22][5] = 1.0;
    feats[22][22] = -1.0;
    targets[22] = [-1.0e8, 1.0];

    let (g, c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    assert!((g[uplo_u_index(0, 7)] - tiny_f64).abs() <= 1.0e-20);
    assert!((g[uplo_u_index(3, 19)] - tiny_f64).abs() <= 1.0e-20);
    assert!((g[uplo_u_index(5, 22)] - tiny_f64).abs() <= 1.0e-20);
    assert!((c[5][0] - tiny_f64).abs() <= 1.0e-20);
    assert!((c[22][1] - tiny_f64).abs() <= 1.0e-20);

    // Non-divisible prime stride = 7 integral scaling on constant unit feature
    let const_feats = vec![[1.0_f32; 256]; 100];
    let zero_tgt = vec![[0.0_f64; 2]; 100];
    let unit_w = vec![1.0_f64; 100];
    let (g_s7, _) = contract_256d_gram_tiled_wide192(&const_feats, &zero_tgt, &unit_w, 7);
    assert!((g_s7[uplo_u_index(0, 0)] - 100.0).abs() <= 1.0e-11);

    // Normal-scale intra-batch cancellation (max <= 1.0, exponent span 60 > 53 bits):
    // [1.0, 2^-30, -1.0] * [1.0, 2^-30, 1.0] == 2^-60 and target [1.0, 2^-60, -1.0] == 2^-60.
    let two_neg60 = 2.0_f64.powi(-60);
    let mut norm_feats = vec![[0.0_f32; 256]; 8];
    let mut norm_targets = vec![[0.0_f64; 2]; 8];
    let norm_weights = vec![1.0_f64; 8];
    // Off-diagonal tile (col 0 vs col 7) + diagonal tile (col 1 vs col 2) + non-wide cols 3 & 6 + constant col 12
    norm_feats[0][0] = 1.0;
    norm_feats[0][7] = 1.0;
    norm_feats[0][1] = 1.0;
    norm_feats[0][2] = 1.0;
    norm_feats[0][3] = 2.0;
    norm_feats[0][6] = 3.0;
    norm_feats[0][12] = 1.0;
    norm_targets[0] = [1.0, -1.0];

    norm_feats[1][0] = tiny_f32;
    norm_feats[1][7] = tiny_f32;
    norm_feats[1][1] = tiny_f32;
    norm_feats[1][2] = tiny_f32;
    norm_feats[1][12] = 1.0;
    norm_targets[1] = [two_neg60, two_neg60];

    norm_feats[2][0] = -1.0;
    norm_feats[2][7] = 1.0;
    norm_feats[2][1] = -1.0;
    norm_feats[2][2] = 1.0;
    norm_feats[2][12] = 1.0;
    norm_targets[2] = [-1.0, 1.0];

    let (g_norm, c_norm) =
        contract_256d_gram_tiled_wide192(&norm_feats, &norm_targets, &norm_weights, 1);
    assert_eq!(g_norm[uplo_u_index(0, 7)].to_bits(), two_neg60.to_bits());
    assert_eq!(g_norm[uplo_u_index(1, 2)].to_bits(), two_neg60.to_bits());
    assert_eq!(g_norm[uplo_u_index(3, 3)], 4.0);
    assert_eq!(g_norm[uplo_u_index(3, 6)], 6.0);
    assert_eq!(c_norm[12][0].to_bits(), two_neg60.to_bits());
    assert_eq!(c_norm[12][1].to_bits(), two_neg60.to_bits());

    // Weight-driven intra-batch cancellation: w = [1.0, 2^-60, 1.0] with normal-scale unit features
    let mut w_feats = vec![[0.0_f32; 256]; 3];
    w_feats[0][0] = 1.0;
    w_feats[0][7] = 1.0;
    w_feats[1][0] = 1.0;
    w_feats[1][7] = 1.0;
    w_feats[2][0] = -1.0;
    w_feats[2][7] = 1.0;
    let w_targets = vec![[0.0_f64; 2]; 3];
    let raw_w = vec![1.0_f64, two_neg60, 1.0_f64];
    let (g_w, _) = contract_256d_gram_tiled_wide192(&w_feats, &w_targets, &raw_w, 1);
    assert_eq!(g_w[uplo_u_index(0, 7)].to_bits(), two_neg60.to_bits());

    // Partial intra-batch cancellation (`[1.0, 2^-28, -1.0, 2^-9]` -> `2^-18 + 2^-56` bit-exact)
    // and 3-batch normal-scale cross-batch cancellation (`+1.0` in B0, `+2^-60` in B1, `-1.0` in B2).
    let mut adv_feats = vec![[0.0_f32; 256]; 520];
    let adv_targets = vec![[0.0_f64; 2]; 520];
    let adv_weights = vec![1.0_f64; 520];
    adv_feats[0][0] = 1.0;
    adv_feats[0][7] = 1.0;
    adv_feats[1][0] = 2.0_f32.powi(-28);
    adv_feats[1][7] = 2.0_f32.powi(-28);
    adv_feats[2][0] = -1.0;
    adv_feats[2][7] = 1.0;
    adv_feats[3][0] = 2.0_f32.powi(-9);
    adv_feats[3][7] = 2.0_f32.powi(-9);

    adv_feats[0][4] = 1.0;
    adv_feats[0][11] = 1.0;
    adv_feats[256][4] = tiny_f32;
    adv_feats[256][11] = tiny_f32;
    adv_feats[512][4] = -1.0;
    adv_feats[512][11] = 1.0;

    let (g_adv, _) = contract_256d_gram_tiled_wide192(&adv_feats, &adv_targets, &adv_weights, 1);
    let expected_partial = 2.0_f64.powi(-18) + 2.0_f64.powi(-56);
    assert_eq!(
        g_adv[uplo_u_index(0, 7)].to_bits(),
        expected_partial.to_bits()
    );
    assert_eq!(g_adv[uplo_u_index(4, 11)].to_bits(), two_neg60.to_bits());
}

#[test]
fn test_engine_immediate_control_sync_reload_reset_and_tier_gain_trim() {
    // verify[train.pipeline.stage-e-pack]
    // verify[train.pipeline.isomorphism-parity]
    let mut hdr = DnamJsonHeader::default_reference();
    hdr.io_config.inputs = 4;
    hdr.io_config.output_mode = "stereo".to_string();
    hdr.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 0.0)];

    let mut model = V3MatryoshkaModel::identity(false, false);
    for b in 0..3 {
        model.blocks[b].w_in_n[0][0] = 0.25;
        model.blocks[b].w_in_n[1][0] = 0.20;
    }
    for (t_idx, &tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .enumerate()
    {
        let mut ro = V3TierReadout::zeros();
        ro.pre_cab_u_gain = [0.40, 0.35];
        ro.fir_l[0] = 0.12;
        ro.fir_r[0] = 0.12;
        ro.pre_cab_w[0][0] = 0.20;
        ro.pre_cab_w[1][0] = 0.20;
        ro.mask_to_tier(tier);
        model.readouts[t_idx] = ro;
    }

    let mut arena_a = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&model, &mut arena_a);
    // Rows 2 & 12 of unified W_ctrl (`ARENA0_W_CTRL_OFFSET`) map `z_head[0]` into Block 0 FiLM & Tone EQ
    write_f32(&mut arena_a, ARENA0_W_CTRL_OFFSET + 2 * 64, 3.0);
    write_f32(&mut arena_a, ARENA0_W_CTRL_OFFSET + 12 * 64, 3.0);

    let mut slot = Cp1ContributorSlot {
        lanes: 2,
        ctrl_kind: 0,
        control_idx: 0,
        ..Cp1ContributorSlot::zero()
    };
    slot.main_base[0][0] = 1.0;
    slot.emb[0] = 1.0;
    slot.v_base[0] = 1.0;
    write_inline_cp1_if_fits(&mut arena_a, &[slot]);

    let trims = [2.0_f32, 0.5, 1.25, 1.0];
    for (t, &tr) in trims.iter().enumerate() {
        write_f32(&mut arena_a, ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4, tr);
    }

    let dnam_a =
        build_dnam_container_with_cp1(&mut hdr.clone(), &arena_a, &arena_a, false, &[slot])
            .unwrap();

    // 1. Pre-tick immediate sync (samples 0..4, well before sample 16's first 3 kHz control tick!)
    let mut eng_def =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    eng_def.set_control_immediate(0, 0.0);
    let mut y_pre_def = [0.0_f32; 2];
    for _ in 0..4 {
        y_pre_def = eng_def.step_frame_tora(0.35, 0.35, false);
    }

    let mut eng_hi =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    eng_hi.set_control_immediate(0, 10.0);
    let mut y_pre_hi = [0.0_f32; 2];
    for _ in 0..4 {
        y_pre_hi = eng_hi.step_frame_tora(0.35, 0.35, false);
    }
    assert!(
        (y_pre_hi[0] - y_pre_def[0]).abs() > 1.0e-3,
        "set_control_immediate must update current_stage_params before sample 16 tick: hi={y_pre_hi:?} vs def={y_pre_def:?}"
    );

    // Verify reset_controls_to_defaults snaps back to default immediately before sample 16 tick
    let mut eng_reset =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    eng_reset.set_control_immediate(0, 10.0);
    eng_reset.reset_controls_to_defaults();
    let mut eng_fresh_def =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Ultra, 48_000).unwrap();
    for _ in 0..4 {
        assert_eq!(
            eng_reset.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            eng_fresh_def.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            "reset_controls_to_defaults must immediately clear pre-tick control state"
        );
    }

    // 2. reload_model_in_place must clear stale current_stage_params from eng_hi
    let mut arena_b = arena_a.clone();
    arena_b[ARENA0_W_CTRL_OFFSET..ARENA0_W_CTRL_OFFSET + 1_024].fill(0);
    arena_b[ARENA3_W_CTRL_ULTRA_OFFSET..ARENA3_W_CTRL_ULTRA_OFFSET + 256].fill(0);
    write_inline_cp1_if_fits(&mut arena_b, &[]);
    let dnam_b =
        build_dnam_container_with_cp1(&mut hdr.clone(), &arena_b, &arena_b, false, &[]).unwrap();
    eng_hi
        .reload_model_in_place(&dnam_b, Tier::Ultra, 48_000)
        .unwrap();
    let mut eng_b_fresh =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_b, Tier::Ultra, 48_000).unwrap();
    for _ in 0..4 {
        assert_eq!(
            eng_hi.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            eng_b_fresh.step_frame_tora(0.35, 0.35, false)[0].to_bits(),
            "reload_model_in_place leaked stale current_stage_params during pre-tick window"
        );
    }

    // 3. Verify ARENA0_TIER_GAIN_TRIM_OFFSET scales live 1x & 4x 2-port and 4CM outputs + 0 ULP stripping
    let mut untrimmed_arena = arena_a.clone();
    for t in 0..4 {
        write_f32(
            &mut untrimmed_arena,
            ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4,
            1.0,
        );
    }
    let dnam_untrimmed = build_dnam_container_with_cp1(
        &mut hdr.clone(),
        &untrimmed_arena,
        &untrimmed_arena,
        false,
        &[slot],
    )
    .unwrap();

    for (t, (&tier, &tr)) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .zip(&trims)
        .enumerate()
    {
        let mut e_trim =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, tier, 48_000).unwrap();
        let mut e_raw =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_untrimmed, tier, 48_000)
                .unwrap();
        let yt = e_trim.step_frame_tora(0.25, 0.25, false);
        let yr = e_raw.step_frame_tora(0.25, 0.25, false);
        assert!(
            (yt[0] - yr[0] * tr).abs() <= 1.0e-6,
            "tier {t} 1x trim mismatch: got {} vs expected {}",
            yt[0],
            yr[0] * tr
        );

        // 4x offline bounce + 4CM 4-port parity
        e_trim.reset_state();
        e_raw.reset_state();
        e_trim.set_offline_bounce(true);
        e_raw.set_offline_bounce(true);
        let (mut ml_t, mut mr_t, mut sl_t, mut sr_t) = ([0.0_f32; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
        let (mut ml_r, mut mr_r, mut sl_r, mut sr_r) = ([0.0_f32; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
        let inp = [0.25_f32; 4];
        let zero = [0.0_f32; 4];
        e_trim.process_block_4cm(
            &inp, &inp, &zero, &zero, &mut ml_t, &mut mr_t, &mut sl_t, &mut sr_t,
        );
        e_raw.process_block_4cm(
            &inp, &inp, &zero, &zero, &mut ml_r, &mut mr_r, &mut sl_r, &mut sr_r,
        );
        assert!(
            (ml_t[3] - ml_r[3] * tr).abs() <= 1.0e-5,
            "tier {t} 4x 4CM trim mismatch: got {} vs expected {}",
            ml_t[3],
            ml_r[3] * tr
        );
    }

    let s_pro = strip_container_to_tier(&dnam_a, Tier::Pro).unwrap();
    let s_full = strip_container_to_tier(&s_pro, Tier::Full).unwrap();
    let s_lite = strip_container_to_tier(&s_full, Tier::Lite).unwrap();
    let mut e_direct_lite =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_a, Tier::Lite, 48_000).unwrap();
    let mut e_strip_lite =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&s_lite, Tier::Lite, 48_000).unwrap();
    for i in 0..64 {
        let x = 0.2 * ((i as f32) * 0.13).sin();
        let a = e_direct_lite.step_frame_tora(x, x, false);
        let b = e_strip_lite.step_frame_tora(x, x, false);
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert_eq!(a[1].to_bits(), b[1].to_bits());
    }
}

#[test]
fn test_vault_path1_stride_autodetect_w_ctrl_preservation_and_with_options_apis() {
    // verify[train.cache.proj-vault]
    // verify[train.resume.kronecker-reassembly]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("gain", 0.0, 10.0, 5.0)];

    let mut seed_model = V3MatryoshkaModel::identity(false, true);
    for b in 0..3 {
        seed_model.blocks[b].w_in_n[0][0] = 0.25;
        seed_model.blocks[b].w_in_n[1][0] = 0.20;
    }
    let mut seed_arena = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&seed_model, &mut seed_arena);
    let dry: Vec<f32> = (0..CORESET_FP32_FRAMES_PER_STEM)
        .map(|i| 0.35 * ((i as f32) * 0.071).sin() + 0.2 * ((i as f32) * 0.19).cos())
        .collect();
    let dry_lr: Vec<[f32; 2]> = dry.iter().map(|&x| [x, x]).collect();
    let wet_star: Vec<f32> = dry.iter().map(|&x| (1.5 * x).tanh()).collect();
    let wet_lr: Vec<[f32; 2]> = wet_star.iter().map(|&y| [y, y]).collect();
    let _ = run_stage_a_physics_probe(
        &manifest,
        &mut seed_arena,
        &dry_lr,
        &wet_lr,
        None,
        StageAConfig::default(),
    );
    write_f32(&mut seed_arena, ARENA0_W_CTRL_OFFSET + 32, 0.42);
    write_f32(&mut seed_arena, ARENA3_UINT_ULTRA_OFFSET + 12, -0.31);

    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();
    let mut schedule_320 = [0u8; 320];
    schedule_320[8..12].copy_from_slice(&1_024_u32.to_le_bytes());
    let inpc = build_proj_inpc_capsule(
        1,
        &[0x21; 256],
        &schedule_320,
        &encode_coreset_fp32(&dry, None),
    )
    .unwrap();

    let star_spec = make_spec(
        "star_5",
        true,
        BTreeMap::from([("gain".to_string(), serde_json::json!(5.0))]),
        &wet_star,
        vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: 5.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
    );
    let proj_p2 = add_capture_with_options(
        &seed_proj,
        star_spec,
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();

    let wet_hi: Vec<f32> = dry.iter().map(|&x| (3.1 * x).tanh()).collect();
    let off_spec = make_spec(
        "off_10",
        false,
        BTreeMap::from([("gain".to_string(), serde_json::json!(10.0))]),
        &wet_hi,
        vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0xFF,
            u_phys: 10.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }],
    );
    let proj_p1 = add_capture(&proj_p2, off_spec.clone(), None).unwrap();

    let v_p2 = DnamProjView::from_bytes(&proj_p2, &crypto).unwrap();
    let v_p1 = DnamProjView::from_bytes(&proj_p1, &crypto).unwrap();
    let a_p2 =
        DnamContainerView::from_bytes(v_p2.chunk_slice(ProjChunkKind::Wght, 0).unwrap(), &crypto)
            .unwrap()
            .arena_storage
            .as_bytes()
            .to_vec();
    let a_p1 =
        DnamContainerView::from_bytes(v_p1.chunk_slice(ProjChunkKind::Wght, 0).unwrap(), &crypto)
            .unwrap()
            .arena_storage
            .as_bytes()
            .to_vec();

    assert_eq!(read_f32(&a_p1, ARENA0_W_CTRL_OFFSET + 32), 0.42);
    assert_eq!(read_f32(&a_p1, ARENA3_UINT_ULTRA_OFFSET + 12), -0.31);
    for &(hb, sz) in &[
        (ARENA0_DIRECT_READOUT_OFFSET, 320usize),
        (ARENA1_DIRECT_READOUT_OFFSET, 384),
        (ARENA2_DIRECT_READOUT_OFFSET, 448),
        (ARENA3_DIRECT_READOUT_OFFSET, 512),
    ] {
        assert_eq!(&a_p1[hb..hb + sz], &a_p2[hb..hb + sz]);
    }

    let proj_dis = set_capture_enabled_with_options(
        &proj_p1,
        "off_10",
        false,
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let proj_rep = replace_capture_with_options(
        &proj_dis,
        "off_10",
        off_spec,
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let proj_rem =
        remove_capture_with_options(&proj_rep, "off_10", SolverOptions::preserve_backbone())
            .unwrap();
    let v_rem = DnamProjView::from_bytes(&proj_rem, &crypto).unwrap();
    assert_eq!(v_rem.manifest.stems_registry.len(), 1);
}
