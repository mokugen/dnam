#![forbid(unsafe_code)]
//! `Oracle-02` (`P = 6` & `P = 16` `CP1` + Hadamard FM) and `Oracle-03` (zero-WAV `.dnam-proj`
//! vault fast-paths, `SPCM` compaction, HTTP Range streaming, and `[CLAMP-RANGE]` / `[CLAMP-SWITCH]`)
//! split from `oracles.rs` (`< 650` LOC per file).

use std::collections::BTreeMap;

use dnam_capture::CORESET_FP32_FRAMES_PER_STEM;
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, PROJ_FLAG_COMPACTED_CAPSULES, PROJ_FLAG_HAS_RAW_STEMS,
    ProjChunkKind, StmcTrajectoryWindow, add_capture, build_dnam_proj_container,
    build_dnam_proj_container_with_cp1, build_proj_inpc_capsule, compact_to_capsules,
    export_runtime_dnam, parse_proj_manifest_range, remove_capture, replace_capture,
    set_capture_enabled, solve_captures_batch_with_options,
};
use dnam_capture::solver::{SolverOptions, star_center_control_values};
use dnam_capture::stimulus::encode_coreset_fp32;
use dnam_engine::Tier;
use dnam_engine::container::strip_container_to_tier;
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::ControlKind;
use dnam_engine::stages::weights::{
    ARENA_ULTRA_BYTES, ARENA0_W_CTRL_OFFSET, pack_v3_matryoshka_to_arena,
    unpack_v3_matryoshka_from_arena,
};

use super::{
    assert_0_ulp_stereo, build_oracle01_teacher_arena_v3, clear_output_readout_weights, ctrl_meta,
    esr_db_stereo, interleave_lr, make_range_ctrl, make_switch_ctrl, make_vault_spec, pack_dnam,
    render_stereo_dnam, stmc_win, synthesize_rich_coreset_dry, write_f32,
};

fn build_oracle02_teacher_and_seed(
    num_controls: usize,
) -> (Vec<u8>, DnamProjManifest, Vec<Cp1ContributorSlot>, Vec<u8>) {
    let mut arena = build_oracle01_teacher_arena_v3();
    // Populate smooth ControlBus `W_ctrl` couplings (`ARENA0_W_CTRL_OFFSET`):
    // - `z_head[0..3]` (cols 0..3) drive Tone SVF EQ (`rows 12, 13, 14`) & Block 0 FiLM gain (`row 2`)
    // - `ctrl_basis_4[0]` (col 4, Hadamard interaction via `v_base[0]`) drives Tone SVF cutoff (`row 9`)
    let couplings: &[(usize, usize, f32)] = &[
        (2, 0, 0.18),
        (12, 0, 0.22),
        (13, 1, 0.24),
        (14, 2, 0.20),
        (9, 4, 0.22),
    ];
    for &(row, col, val) in couplings {
        write_f32(&mut arena, ARENA0_W_CTRL_OFFSET + row * 64 + col * 4, val);
    }

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    let mut schema = vec![
        make_range_ctrl("drive_asym", -1.0, 1.0, -0.35),
        make_range_ctrl("tone_asym", -1.0, 1.0, 0.40),
        make_switch_ctrl("mode_sw0", &["Clean", "Crunch", "Lead"], 0),
        make_switch_ctrl("bright_sw1", &["Dark", "Normal", "Bright"], 1),
    ];
    for p in 4..num_controls {
        schema.push(make_range_ctrl(&format!("knob_{p}"), 0.0, 10.0, 5.0));
    }
    manifest.base.controls_schema = schema;

    let range_slot = |c_idx: u16, d: usize, l0: f32, l1: f32, vb0: f32| {
        let mut s = Cp1ContributorSlot::zero();
        s.lanes = 2;
        s.control_idx = c_idx;
        s.option_idx = 0xFF;
        s.main_base[0][d] = l0;
        s.main_base[1][d] = l1;
        s.emb = [1.0, 0.0, 0.0];
        s.v_base[0] = vb0;
        s
    };
    let mut teacher_slots = vec![
        range_slot(0, 0, 0.90, 0.35, 0.90),
        range_slot(1, 1, -0.80, 0.30, 0.80),
    ];
    for &(c_idx, opt_idx, m2) in &[
        (2u16, 1u8, 0.75_f32),
        (2, 2, -0.65),
        (3, 0, -0.55),
        (3, 2, 0.60),
    ] {
        let mut sw = Cp1ContributorSlot::zero();
        sw.lanes = 1;
        sw.ctrl_kind = 1;
        sw.control_idx = c_idx;
        sw.option_idx = opt_idx;
        sw.main_base[0][2] = m2;
        sw.emb = [1.0, 0.0, 0.0];
        teacher_slots.push(sw);
    }
    teacher_slots.push(range_slot((num_controls - 1) as u16, 2, 0.60, -0.25, 0.0));

    let mut seed_model = unpack_v3_matryoshka_from_arena(&arena, false);
    for ro in &mut seed_model.readouts {
        clear_output_readout_weights(ro);
    }
    let mut seed_arena = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&seed_model, &mut seed_arena);

    (arena, manifest, teacher_slots, seed_arena)
}

fn run_oracle_02_for_p(num_controls: usize) {
    let crypto = DefaultCryptoProvider::default();
    let (teacher_arena, manifest, teacher_slots, seed_arena) =
        build_oracle02_teacher_and_seed(num_controls);
    let teacher_dnam = pack_dnam(&manifest.base, &teacher_arena, &teacher_slots);

    let dry = synthesize_rich_coreset_dry(CORESET_FP32_FRAMES_PER_STEM);
    let dry_fp32 = encode_coreset_fp32(&dry, None);
    let mut sched_fast = [0u8; 320];
    sched_fast[8..12].copy_from_slice(&2_048_u32.to_le_bytes());
    let inpc = build_proj_inpc_capsule(1, &[0x22; 256], &sched_fast, &dry_fp32).unwrap();

    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();

    let star_vals = star_center_control_values(&manifest);
    assert!((star_vals[0] - (-0.35)).abs() < 1.0e-6 && (star_vals[1] - 0.40).abs() < 1.0e-6);
    assert!((star_vals[2] - 0.0).abs() < 1.0e-6 && (star_vals[3] - 1.0).abs() < 1.0e-6);

    let make_stem = |stem_id: &str,
                     is_star: bool,
                     overrides: &[(usize, f32)],
                     windows: Vec<StmcTrajectoryWindow>| {
        let mut full_ctrls: Vec<(usize, f32)> = star_vals.iter().copied().enumerate().collect();
        let mut meta = BTreeMap::new();
        for (idx, ctrl_def) in manifest.base.controls_schema.iter().enumerate() {
            let v = overrides
                .iter()
                .find_map(|&(i, val)| (i == idx).then_some(val))
                .unwrap_or(star_vals[idx]);
            full_ctrls[idx] = (idx, v);
            match ctrl_def.kind {
                ControlKind::Range => {
                    meta.insert(ctrl_def.id.clone(), serde_json::json!(f64::from(v)));
                }
                ControlKind::Switch => {
                    let name = &ctrl_def.declared_options.as_ref().unwrap()[v.round() as usize];
                    meta.insert(ctrl_def.id.clone(), serde_json::json!(name));
                }
            }
        }
        let wet_lr = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry, &full_ctrls);
        make_vault_spec(
            stem_id,
            is_star,
            meta,
            &dry,
            &interleave_lr(&wet_lr),
            windows,
            None,
        )
    };

    let star_spec = make_stem("star_0", true, &[], vec![stmc_win(0, 0, 0xFF, -0.35)]);
    let proj_star_only = solve_captures_batch_with_options(
        &seed_proj,
        vec![star_spec.clone()],
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let dnam_star_only = export_runtime_dnam(&proj_star_only, Some(Tier::Ultra)).unwrap();
    let star_pairs: Vec<(usize, f32)> = star_vals.iter().copied().enumerate().collect();
    let star_ref_audio = render_stereo_dnam(&dnam_star_only, Tier::Ultra, &dry[..512], &star_pairs);

    let mut batch = vec![star_spec];
    for &(sid, p_idx, val) in &[
        ("c0_lo", 0usize, -1.0_f32),
        ("c0_hi", 0, 1.0),
        ("c1_lo", 1, -1.0),
        ("c1_hi", 1, 1.0),
        ("last_lo", num_controls - 1, 0.0),
        ("last_hi", num_controls - 1, 10.0),
    ] {
        batch.push(make_stem(
            sid,
            false,
            &[(p_idx, val)],
            vec![stmc_win(p_idx as u16, 0, 0xFF, val)],
        ));
    }
    for &(sid, sw_idx, opt) in &[
        ("sw0_crunch", 2usize, 1u8),
        ("sw0_lead", 2, 2),
        ("sw1_dark", 3, 0),
        ("sw1_bright", 3, 2),
    ] {
        batch.push(make_stem(
            sid,
            false,
            &[(sw_idx, f32::from(opt))],
            vec![stmc_win(sw_idx as u16, 1, opt, f32::from(opt))],
        ));
    }
    for &(sid, u0, u1) in &[
        ("had_pp", 1.0_f32, 1.0_f32),
        ("had_pm", 1.0, -1.0),
        ("had_mp", -1.0, 1.0),
        ("had_mm", -1.0, -1.0),
    ] {
        batch.push(make_stem(
            sid,
            false,
            &[(0, u0), (1, u1)],
            vec![stmc_win(0, 0, 0xFF, u0), stmc_win(1, 0, 0xFF, u1)],
        ));
    }

    let proj_multi = solve_captures_batch_with_options(
        &seed_proj,
        batch,
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();
    let view_multi = DnamProjView::from_bytes(&proj_multi, &crypto).unwrap();
    assert_eq!(view_multi.manifest.base.controls_schema.len(), num_controls);

    let dnam_multi_ultra = export_runtime_dnam(&proj_multi, Some(Tier::Ultra)).unwrap();
    let dnam_multi_pro = export_runtime_dnam(&proj_multi, Some(Tier::Pro)).unwrap();
    let stripped_multi_pro = strip_container_to_tier(&dnam_multi_ultra, Tier::Pro).unwrap();

    // 1. 0.000000 ULP StarCenter invariance on multi-knob model
    let star_multi_audio =
        render_stereo_dnam(&dnam_multi_ultra, Tier::Ultra, &dry[..512], &star_pairs);
    assert_0_ulp_stereo(&star_ref_audio, &star_multi_audio, "StarCenter 0 ULP");

    // 2. Multi-knob, non-zero-default switch, P > 8 knob, 2D Hadamard FM + 0 ULP prefix strip
    for test_overrides in [
        vec![(0usize, 1.0_f32), (1, 1.0)],
        vec![(0, 1.0), (1, -1.0)],
        vec![(2, 2.0), (3, 0.0)],
        vec![(num_controls - 1, 10.0)],
    ] {
        let mut full_c = star_pairs.clone();
        for &(i, v) in &test_overrides {
            full_c[i] = (i, v);
        }
        let y_teacher = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry[..2_048], &full_c);
        let y_student = render_stereo_dnam(&dnam_multi_ultra, Tier::Ultra, &dry[..2_048], &full_c);
        let corner_esr = esr_db_stereo(&y_teacher, &y_student);
        assert!(
            corner_esr < -28.0,
            "P={num_controls} corner {test_overrides:?} ESR must beat -28 dB, got {corner_esr:.2} dB"
        );

        let y_pro_strip = render_stereo_dnam(&stripped_multi_pro, Tier::Pro, &dry[..256], &full_c);
        let y_pro_direct = render_stereo_dnam(&dnam_multi_pro, Tier::Pro, &dry[..256], &full_c);
        assert_0_ulp_stereo(&y_pro_strip, &y_pro_direct, "Oracle-02 Pro strip");
    }
}

#[test]
fn test_oracle_02_cp1_arbitrary_p6_and_p16_asymmetric_switches_and_hadamard_fm() {
    // verify[train.pipeline.control-projection-fit]
    // verify[train.pipeline.dispatch-solve]
    run_oracle_02_for_p(6);
    run_oracle_02_for_p(16);
}

#[test]
fn test_oracle_03_zero_wav_dnam_proj_vault_fastpath_and_clamping() {
    // verify[train.pipeline.host-parity-and-export]
    // verify[train.resume.kronecker-reassembly]
    // verify[train.cache.proj-vault]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    manifest.base.controls_schema = vec![
        make_range_ctrl("gain", 0.0, 10.0, 5.0),
        make_switch_ctrl("channel", &["Clean", "Crunch", "Lead"], 0),
    ];
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let dry = synthesize_rich_coreset_dry(CORESET_FP32_FRAMES_PER_STEM);
    let wet_star: Vec<f32> = dry
        .iter()
        .flat_map(|&x| [(1.4 * x).tanh(), (1.2 * x).tanh()])
        .collect();
    let mut spcm_raw = vec![0u8; 64 + 96_000 * 4];
    spcm_raw[0..4].copy_from_slice(b"SPCM");
    spcm_raw[4..6].copy_from_slice(&2u16.to_le_bytes());

    let dry_fp32 = encode_coreset_fp32(&dry, None);
    let mut sched_fast = [0u8; 320];
    sched_fast[8..12].copy_from_slice(&1_024_u32.to_le_bytes());
    let inpc = build_proj_inpc_capsule(1, &[0x44; 256], &sched_fast, &dry_fp32).unwrap();
    let star_spec = make_vault_spec(
        "star_0",
        true,
        ctrl_meta(5.0, "Clean"),
        &dry,
        &wet_star,
        vec![stmc_win(0, 0, 0xFF, 5.0)],
        Some(spcm_raw),
    );
    let proj_with_spcm = add_capture(&empty_proj, star_spec, Some(&inpc)).unwrap();
    assert!(proj_with_spcm.len() > 65_536);

    let view_spcm = DnamProjView::from_bytes(&proj_with_spcm, &crypto).unwrap();
    assert_ne!(
        view_spcm.preamble.lifecycle_flags & PROJ_FLAG_HAS_RAW_STEMS,
        0
    );
    let compacted = compact_to_capsules(&proj_with_spcm).unwrap();
    assert!(compacted.len() < proj_with_spcm.len());
    let view_cmp = DnamProjView::from_bytes(&compacted, &crypto).unwrap();
    assert_eq!(
        view_cmp.preamble.lifecycle_flags & PROJ_FLAG_HAS_RAW_STEMS,
        0
    );
    assert_ne!(
        view_cmp.preamble.lifecycle_flags & PROJ_FLAG_COMPACTED_CAPSULES,
        0
    );
    let prefix_end = (view_cmp.preamble.chunk_count as usize) * 64 + 65_536;
    let range_parsed =
        parse_proj_manifest_range(&compacted[..prefix_end.min(compacted.len())]).unwrap();
    assert_eq!(range_parsed.manifest.stems_registry.len(), 1);
    let stat_payload = view_cmp
        .chunk_slice(ProjChunkKind::Stat, 0)
        .unwrap()
        .to_vec();
    let dnam_star_orig = export_runtime_dnam(&compacted, Some(Tier::Ultra)).unwrap();
    let star_ctrls = [(0, 5.0), (1, 0.0)];
    let audio_star_orig =
        render_stereo_dnam(&dnam_star_orig, Tier::Ultra, &dry[..512], &star_ctrls);

    let wet_off: Vec<f32> = dry
        .iter()
        .flat_map(|&x| [(3.2 * x).tanh(), (2.9 * x).tanh()])
        .collect();
    let off_spec = make_vault_spec(
        "off_lead_9",
        false,
        ctrl_meta(9.0, "Lead"),
        &dry,
        &wet_off,
        vec![stmc_win(0, 0, 0xFF, 9.0)],
        None,
    );
    let proj_added = add_capture(&compacted, off_spec, None).unwrap();
    let added_view = DnamProjView::from_bytes(&proj_added, &crypto).unwrap();
    let added_schema = &added_view.manifest.base.controls_schema;
    assert_eq!(added_schema[0].safe_clamped_range, Some([5.0, 9.0]));
    assert_eq!(
        added_schema[1].active_options,
        Some(vec!["Clean".to_string(), "Lead".to_string()])
    );
    let dnam_added = export_runtime_dnam(&proj_added, Some(Tier::Ultra)).unwrap();
    let audio_star_after_add =
        render_stereo_dnam(&dnam_added, Tier::Ultra, &dry[..512], &star_ctrls);
    assert_0_ulp_stereo(&audio_star_orig, &audio_star_after_add, "Path 1 StarCenter");

    let proj_muted = set_capture_enabled(&proj_added, "off_lead_9", false).unwrap();
    let muted_view = DnamProjView::from_bytes(&proj_muted, &crypto).unwrap();
    let muted_schema = &muted_view.manifest.base.controls_schema;
    assert_eq!(muted_schema[0].safe_clamped_range, Some([5.0, 5.0]));
    assert_eq!(
        muted_schema[1].active_options,
        Some(vec!["Clean".to_string()])
    );
    let proj_removed = remove_capture(&proj_muted, "off_lead_9").unwrap();

    let wet_star_2: Vec<f32> = dry.iter().flat_map(|&x| [0.55 * x, -0.45 * x]).collect();
    let rep_spec = make_vault_spec(
        "star_0",
        true,
        ctrl_meta(3.0, "Crunch"),
        &dry,
        &wet_star_2,
        vec![stmc_win(0, 0, 0xFF, 3.0)],
        None,
    );
    let proj_rep = replace_capture(&proj_removed, "star_0", rep_spec).unwrap();
    let rep_view = DnamProjView::from_bytes(&proj_rep, &crypto).unwrap();
    assert_ne!(
        rep_view.chunk_slice(ProjChunkKind::Stat, 0).unwrap(),
        stat_payload.as_slice()
    );
}
