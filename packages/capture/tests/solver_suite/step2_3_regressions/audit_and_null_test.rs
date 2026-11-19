#![forbid(unsafe_code)]
//! Step 2.3 Stage D `P = 68` `CP1`, 3-Phase Audit, Role-B Live Null-Test Auto-Patch,
//! Black-Box Phase Watermark Z-Score, and Faust / SPICE White-Box Distillation Regressions.

use std::collections::BTreeMap;

use dnam_capture::audit::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkInput, ProjChunkKind, StmcTrajectoryWindow,
    build_dnam_proj_container, build_dnam_proj_container_with_cp1, build_proj_inpc_capsule,
    export_runtime_dnam,
};
use dnam_capture::solver::{
    BuyerWatermarkConfig, compute_integrated_lufs, derive_buyer_phase_seed,
    export_runtime_dnam_watermarked, identify_pole_angles_blackbox_from_audio,
    render_blackbox_probe_response,
};
use dnam_capture::stimulus::{
    AdaptiveTrainingOptions, TopologyProbeMode, encode_coreset_fp32,
    run_live_null_test_and_autopatch, train_adaptive_target,
};
use dnam_capture::whitebox::{FaustRunner, SpiceRunner, parse_faust_json_controls};
use dnam_capture::{CORESET_FP32_FRAMES_PER_STEM, DistillationTarget, StemResponse, StemStimulus};
use dnam_engine::container::DnamContainerView;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, apply_cascade_head_gauge, score_phase_coding_blackbox,
};
use dnam_engine::schema::DnamError;
use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::weights::{
    ARENA_ULTRA_BYTES, ARENA0_DIRECT_READOUT_OFFSET, pack_v3_matryoshka_to_arena,
    unpack_v3_matryoshka_from_arena,
};
use dnam_engine::{DnamEngine, Tier};

use super::{make_range_ctrl, make_spec, read_f32, write_f32};

#[test]
fn test_stage_d_p68_region4_cp1_and_collinear_controls_robustness() {
    // verify[train.pipeline.stage-d-control]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = (0..68)
        .map(|p| make_range_ctrl(&format!("k_{p:02}"), 0.0, 10.0, 5.0))
        .collect();
    let empty_proj = build_dnam_proj_container(&manifest, &[]).unwrap();

    let dry: Vec<f32> = (0..CORESET_FP32_FRAMES_PER_STEM)
        .map(|i| 0.30 * ((i as f32) * 0.063).sin())
        .collect();
    let mut schedule_320 = [0u8; 320];
    schedule_320[8..12].copy_from_slice(&1_024_u32.to_le_bytes());
    let inpc = build_proj_inpc_capsule(
        1,
        &[0x41; 256],
        &schedule_320,
        &encode_coreset_fp32(&dry, None),
    )
    .unwrap();

    let mut star_meta = BTreeMap::new();
    for p in 0..68 {
        star_meta.insert(format!("k_{p:02}"), serde_json::json!(5.0));
    }
    let star_win = vec![StmcTrajectoryWindow {
        control_id: 0,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 5.0,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }];
    let wet_star: Vec<f32> = dry.iter().map(|&x| (1.3 * x).tanh()).collect();
    let proj_star = add_capture_mono(
        &empty_proj,
        "star",
        true,
        star_meta.clone(),
        &wet_star,
        star_win,
        Some(&inpc),
    );

    let mut collinear_meta = star_meta;
    collinear_meta.insert("k_00".to_string(), serde_json::json!(9.5));
    collinear_meta.insert("k_01".to_string(), serde_json::json!(9.5));
    collinear_meta.insert("k_67".to_string(), serde_json::json!(9.5));
    let off_win = vec![StmcTrajectoryWindow {
        control_id: 67,
        ctrl_kind: 0,
        active_opt: 0xFF,
        u_phys: 9.5,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }];
    let wet_off: Vec<f32> = dry.iter().map(|&x| (2.8 * x).tanh()).collect();
    let proj_p68 = add_capture_mono(
        &proj_star,
        "off_collinear_67",
        false,
        collinear_meta,
        &wet_off,
        off_win,
        None,
    );

    let dnam_p68 = export_runtime_dnam(&proj_p68, Some(Tier::Ultra)).unwrap();
    let rt_view = DnamContainerView::from_bytes(&dnam_p68, &crypto).unwrap();
    assert!(
        rt_view.cp1_contributors.len() >= 3,
        "P = 68 controls must pack into Region-4 cp1_contributors stream"
    );

    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_p68, Tier::Ultra, 48_000).unwrap();
    eng.set_control_immediate(0, 9.5);
    eng.set_control_immediate(1, 9.5);
    eng.set_control_immediate(67, 9.5);
    let y = eng.step_frame_tora(0.3, 0.3, false);
    assert!(y[0].is_finite() && y[1].is_finite());
}

fn add_capture_mono(
    proj: &[u8],
    id: &str,
    is_star: bool,
    meta: BTreeMap<String, serde_json::Value>,
    wet_mono: &[f32],
    windows: Vec<StmcTrajectoryWindow>,
    inpc: Option<&[u8]>,
) -> Vec<u8> {
    dnam_capture::proj::add_capture(proj, make_spec(id, is_star, meta, wet_mono, windows), inpc)
        .unwrap()
}

#[test]
fn test_3phase_audit_ultra_modes_5_to_8_stage1_stage4_and_head_only_mode() {
    // verify[train.audit.phase1.clamp-stability]
    // verify[train.audit.phase2.clamp-rail]
    // verify[train.audit.phase3.clamp-head-gain]
    let mut model = V3MatryoshkaModel::identity(false, false);
    model.kautz_poles[0].0 = 1.05;
    model.stage4[0].g_loop = 1.25;
    model.stage4[0].delay_lengths[0] = 1;
    model.stage4[3].delay_lengths[0] = 9_000;
    model.blocks[0].b_n[0] = 9.0;
    model.readouts[3].pre_cab_w[0] = [40.0; 36];

    let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&model, &mut arena);
    let stage4_0_g_loop_off = dnam_engine::stages::weights::ARENA0_STAGE4_TANK_OFFSET + 16;

    let mut m_head = DnamProjManifest::default_reference();
    let mut head_only_arena = arena.clone();
    run_head_only_3phase_arena_audit(&mut m_head, &mut head_only_arena);
    let head_model = unpack_v3_matryoshka_from_arena(&head_only_arena, false);
    assert!(head_model.readouts[3].pre_cab_w[0][0] < 10.0);
    assert_eq!(head_model.kautz_poles[0].0, 1.05);
    assert_eq!(read_f32(&head_only_arena, stage4_0_g_loop_off), 1.25);

    let mut m_full = DnamProjManifest::default_reference();
    run_full_3phase_arena_audit(&mut m_full, &mut arena);
    let full_model = unpack_v3_matryoshka_from_arena(&arena, false);
    assert!(full_model.kautz_poles[0].0 <= 0.9995);
    assert!(full_model.stage4[0].g_loop.abs() <= 0.9990);
    assert!(read_f32(&arena, stage4_0_g_loop_off).abs() <= 0.9990);
    assert!((4..=2_047).contains(&full_model.stage4[0].delay_lengths[0]));
    assert!((4..=2_047).contains(&full_model.stage4[3].delay_lengths[0]));
    assert!(full_model.blocks[0].b_n[0] <= 3.8);
    assert!(full_model.readouts[3].pre_cab_w[0][0] < 10.0);
}

#[derive(Clone)]
struct SimpleTanhTarget {
    gain: f32,
    biquads: [[f32; 2]; 8],
}

impl DistillationTarget for SimpleTanhTarget {
    fn reset_state(&mut self) {
        self.biquads = [[0.0; 2]; 8];
    }

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        const MODES: [(f32, f32, f32); 8] = [
            (0.945, 0.08, 0.18),
            (0.938, 0.19, -0.16),
            (0.930, 0.34, 0.15),
            (0.922, 0.55, -0.14),
            (0.912, 0.84, 0.13),
            (0.902, 1.22, -0.12),
            (0.892, 1.72, 0.11),
            (0.882, 2.32, -0.10),
        ];
        let g = stim.control_values.first().copied().unwrap_or(5.0) * 0.35 + self.gain;
        let mut out_1 = Vec::with_capacity(stim.in_1.len());
        let mut ring = [0.0_f32; 32];
        for (wpos, &x) in stim.in_1.iter().enumerate() {
            let nl = (g * x).tanh();
            let delayed = ring[(wpos + 16) & 31];
            ring[wpos & 31] = nl;
            // Multi-mode resonant cabinet beyond the 16-sample onset delay so all 16 Stage III
            // Kautz poles carry non-trivial phase residues well above the -45 dBFS noise floor.
            let mut res = 0.0_f32;
            for (m_idx, &(r, th, c_re)) in MODES.iter().enumerate() {
                let b_g = (1.0 - r * r).sqrt();
                let (st, cr) = (th.sin(), th.cos());
                let prev_re = self.biquads[m_idx][0];
                let prev_im = self.biquads[m_idx][1];
                let next_re = r * (cr * prev_re - st * prev_im) + b_g * delayed;
                let next_im = r * (st * prev_re + cr * prev_im);
                self.biquads[m_idx] = [next_re, next_im];
                res += c_re * next_re;
            }
            out_1.push(0.35 * nl + res);
        }
        Ok(StemResponse {
            stem_id: stim.stem_id.clone(),
            sample_rate_hz: stim.sample_rate_hz,
            out_1,
            out_2: None,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
    }
}

#[test]
fn test_live_null_test_autopatch_with_role_b_gauge_and_noisy_blackbox_phase_id() {
    // verify[train.pipeline.stage-e-pack]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 6.0)];
    let mut target = SimpleTanhTarget {
        gain: 0.4,
        biquads: [[0.0; 2]; 8],
    };
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
    let trained = train_adaptive_target(manifest, &mut target, &opts).unwrap();

    let view = DnamProjView::from_bytes(&trained.proj_bytes, &crypto).unwrap();
    let wght = view.chunk_slice(ProjChunkKind::Wght, 0).unwrap();
    let rt_view = DnamContainerView::from_bytes(wght, &crypto).unwrap();
    let mut gauge_arena = rt_view.arena_storage.as_bytes().to_vec();
    apply_cascade_head_gauge(&mut gauge_arena, 0, &[[1.35_f32; 4]; 32]).unwrap();
    for i in 0..16 {
        let cur = read_f32(&gauge_arena, ARENA0_DIRECT_READOUT_OFFSET + i * 4);
        write_f32(
            &mut gauge_arena,
            ARENA0_DIRECT_READOUT_OFFSET + i * 4,
            cur * 0.55,
        );
    }
    let mut chunks = Vec::new();
    for (entry, &slice) in view.directory.iter().zip(&view.chunk_slices) {
        let kind = entry.kind().unwrap();
        if kind != ProjChunkKind::Wght && kind != ProjChunkKind::Auth {
            chunks.push(ProjChunkInput {
                kind,
                chunk_id: entry.chunk_id,
                payload: slice.to_vec(),
            });
        }
    }
    let mut patched_manifest = view.manifest.clone();
    let gauged_proj =
        build_dnam_proj_container_with_cp1(&mut patched_manifest, &chunks, &gauge_arena, &[])
            .unwrap();

    let (_, _, null_rep) =
        run_live_null_test_and_autopatch(&gauged_proj, &mut target, -60.0).unwrap();
    assert!(null_rep.reload_in_place_verified);
    assert!(
        null_rep.post_patch_null_esr_db <= null_rep.pre_patch_null_esr_db,
        "Role-B gauge auto-patch must not degrade ESR: pre={} post={}",
        null_rep.pre_patch_null_esr_db,
        null_rep.post_patch_null_esr_db
    );

    // Noisy (-45 dBFS) black-box audio-only pole-angle phase watermark identification
    let wm_cfg = BuyerWatermarkConfig::new([0x77; 32], "tanh_v3", "lic_buyer_03", [0x33; 16]);
    let inn_cfg = BuyerWatermarkConfig::new([0x77; 32], "tanh_v3", "lic_buyer_99", [0x99; 16]);
    let wm_dnam =
        export_runtime_dnam_watermarked(&trained.proj_bytes, &wm_cfg, Some(Tier::Ultra)).unwrap();
    let mut wm_audio = render_blackbox_probe_response(&wm_dnam, 8_192).unwrap();
    for (i, y) in wm_audio.iter_mut().enumerate() {
        *y += 0.0025_f32 * ((i as f32 * 0.713).sin() + 0.5 * (i as f32 * 1.917).cos());
    }
    let (master_angles, est_wm_angles) =
        identify_pole_angles_blackbox_from_audio(&trained.runtime_dnam_bytes, &wm_audio).unwrap();
    let z_target = score_phase_coding_blackbox(
        &master_angles,
        &est_wm_angles,
        &derive_buyer_phase_seed(&wm_cfg),
    )
    .unwrap();
    let z_innocent = score_phase_coding_blackbox(
        &master_angles,
        &est_wm_angles,
        &derive_buyer_phase_seed(&inn_cfg),
    )
    .unwrap();
    assert!(
        z_target >= 3.85,
        "noisy (-45 dBFS) black-box phase Z-score must be >= 3.85, got {z_target}"
    );
    assert!(
        z_innocent.abs() < 3.5,
        "innocent buyer noisy black-box phase |Z| must be < 3.5, got {z_innocent}"
    );
}

fn verify_whitebox_exported_audio(label: &str, proj_bytes: &[u8]) {
    let mut lufs = [0.0_f32; 4];
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let dnam_b = export_runtime_dnam(proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_b, tier, 48_000).unwrap();
        let [s0, s1] = eng.step_frame_tora(0.0, 0.0, false);
        assert!(
            s0.abs() <= 1.0e-7 && s1.abs() <= 1.0e-7,
            "{label} [STAB-ZERO-DC] failed"
        );
        eng.reset_to_dc_equilibrium();
        let mut out = Vec::with_capacity(512);
        for i in 0..640 {
            let x = 0.3 * ((i as f32) * 0.085).sin() + 0.15 * ((i as f32) * 0.23).cos();
            let y = eng.step_frame_tora(x, x, false);
            assert!(y[0].is_finite() && y[1].is_finite());
            if i >= 128 {
                out.push(y);
            }
        }
        lufs[t_idx] = compute_integrated_lufs(&out, 48_000) as f32;
    }
    assert!(
        lufs[3] > -60.0,
        "{label} white-box model output unexpectedly silent: {lufs:?}"
    );
}

#[test]
fn test_tier4_whitebox_faust_runner_varpro_distillation() {
    // verify[train.pipeline.stage-a-poles]
    // verify[train.pipeline.fixtures-matrix]
    let faust_json = r#"{"ui":[{"type":"vgroup","label":"Preamp","items":[{"type":"hslider","label":"Drive","init":0.5,"min":0.0,"max":1.0,"step":0.01},{"type":"hslider","label":"Tone","init":0.5,"min":0.0,"max":1.0,"step":0.01}]}]}"#;
    let parsed_specs = parse_faust_json_controls(faust_json).unwrap();
    let mut faust = FaustRunner::from_faust_json("preamp.dsp", faust_json, 5_000).unwrap();
    let mut m_faust = DnamProjManifest::default_reference();
    m_faust.base.controls_schema = parsed_specs
        .iter()
        .map(|s| make_range_ctrl(&s.label.to_lowercase(), s.min, s.max, s.init))
        .collect();

    let opts_mono = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: TopologyProbeMode::Mono1To1,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let rep_faust = train_adaptive_target(m_faust, &mut faust, &opts_mono).unwrap();
    verify_whitebox_exported_audio("faust", &rep_faust.proj_bytes);
}

#[test]
fn test_tier4_whitebox_spice_runner_varpro_distillation() {
    // verify[train.pipeline.stage-a-poles]
    // verify[train.pipeline.fixtures-matrix]
    let mut spice = SpiceRunner::new("triode_stage.cir", 5_000);
    let mut m_spice = DnamProjManifest::default_reference();
    m_spice.base.io_config.inputs = 2;
    m_spice.base.io_config.output_mode = "stereo".to_string();
    m_spice.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 1.0, 0.5)];
    let opts_stereo = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: TopologyProbeMode::TrueStereo2To2,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let rep_spice = train_adaptive_target(m_spice, &mut spice, &opts_stereo).unwrap();
    verify_whitebox_exported_audio("spice", &rep_spice.proj_bytes);
}
