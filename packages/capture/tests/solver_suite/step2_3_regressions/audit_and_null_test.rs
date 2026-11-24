#![forbid(unsafe_code)]
//! Step 2.3 Stage D `P = 68` `CP1`, 3-Phase Audit, Role-B Live Null-Test Auto-Patch,
//! Black-Box Phase Watermark Z-Score, and Faust / SPICE White-Box Distillation Regressions.

use std::collections::BTreeMap;

use dnam_capture::audit::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkInput, ProjChunkKind, StmcTrajectoryWindow,
    build_dnam_proj_container, build_dnam_proj_container_with_dnam_bin, build_proj_inpc_capsule,
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
use dnam_engine::container::{DnamContainerView, extract_dnam_bin};
use dnam_engine::provenance::{
    DefaultCryptoProvider, apply_cascade_head_gauge, score_phase_coding_blackbox,
};
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};
use dnam_train::{Pass5EmitConfig, SonaModelF64, encode_model_to_dnam_bin};
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    DnamBinBuildInput, DnamBinView, encode_dnam_bin,
};
use dnam_vm::math::DefaultMathBackend;
use zerocopy::byteorder::little_endian::F32;
use zerocopy::{FromBytes, IntoBytes};

use super::{make_range_ctrl, make_spec};

#[test]
fn test_stage_d_p68_region4_cp1_and_collinear_controls_robustness() {
    // verify[train.pipeline.control-projection-fit]
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
    let y = eng.step_frame_sona(0.3, 0.3, false);
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
    let model = SonaModelF64::new(48_000.0);
    let base_dbin = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap();
    let v = DnamBinView::parse(&base_dbin).unwrap();
    let mut a0 = *v.arena0;
    let mut a3 = *v.arena3.unwrap();
    a0.conv_kautz_lite[0].r_low[0] = F32::new(1.05);
    a0.transport_lite.g_loop = F32::new(1.25);
    a0.transport_lite.delay_samples[0] = F32::new(1.0);
    a3.transport_ultra[0].delay_samples[0] = F32::new(60_000.0);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(40.0), F32::new(40.0)];

    let mut dbin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: v.header.reference_sample_rate_hz.get(),
        header_flags: v.header.flags.get(),
        ext_ring_pow2_frames: v.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: v.runtime_meta.ext_ring_lines,
        ext_ring_format: v.runtime_meta.ext_ring_format,
        declared_max_delay_48k: v.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: v.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: v.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: v.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: v.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: v.runtime_meta.content_hash_prefix,
        controls: v.control_descriptors,
        cp1_overflow: v.cp1_overflow,
        tsyn: v.tsyn_entries,
        cext: v.cext,
        arena0: &a0,
        arena1: v.arena1,
        arena2: v.arena2,
        arena3: Some(&a3),
    })
    .unwrap();

    let mut m_head = DnamProjManifest::default_reference();
    let mut head_only_dbin = dbin.clone();
    run_head_only_3phase_arena_audit(&mut m_head, &mut head_only_dbin);
    let head_v = DnamBinView::parse(&head_only_dbin).unwrap();
    assert!(head_v.arena0.linear_readout_lite.direct_s2_u_gain[0].get() < 10.0);
    assert_eq!(head_v.arena0.conv_kautz_lite[0].r_low[0].get(), 1.05);
    assert_eq!(head_v.arena0.transport_lite.g_loop.get(), 1.25);

    let mut m_full = DnamProjManifest::default_reference();
    run_full_3phase_arena_audit(&mut m_full, &mut dbin);
    let full_v = DnamBinView::parse(&dbin).unwrap();
    assert!(full_v.arena0.conv_kautz_lite[0].r_low[0].get() <= 0.9995);
    assert!(full_v.arena0.transport_lite.g_loop.get().abs() <= 0.9990);
    assert!((5.0..=48_000.0).contains(&full_v.arena0.transport_lite.delay_samples[0].get()));
    assert!(
        (5.0..=48_000.0)
            .contains(&full_v.arena3.unwrap().transport_ultra[0].delay_samples[0].get())
    );
    assert!(full_v.arena0.linear_readout_lite.direct_s2_u_gain[0].get() < 10.0);
}

const SIMPLE_TANH_MODE_COUNT: usize = 24;

const fn build_simple_tanh_modes() -> [(f32, f32, f32); SIMPLE_TANH_MODE_COUNT] {
    let mut modes = [(0.0_f32, 0.0_f32, 0.0_f32); SIMPLE_TANH_MODE_COUNT];
    let mut i = 0_usize;
    while i < SIMPLE_TANH_MODE_COUNT {
        let u = (i as f32) / ((SIMPLE_TANH_MODE_COUNT - 1) as f32);
        let r = 0.970_f32 - 0.018_f32 * u;
        let th = 0.080_f32 + 0.090_f32 * (i as f32);
        let mag = 0.22_f32 - 0.04_f32 * u;
        let c_re = if (i & 1) == 0 { mag } else { -mag };
        modes[i] = (r, th, c_re);
        i += 1;
    }
    modes
}

const SIMPLE_TANH_MODES: [(f32, f32, f32); SIMPLE_TANH_MODE_COUNT] = build_simple_tanh_modes();
const _: () = {
    let mut i = 0_usize;
    while i < SIMPLE_TANH_MODE_COUNT {
        let (r, th, c_re) = SIMPLE_TANH_MODES[i];
        assert!(r >= 0.85 && r < 0.99);
        assert!(th > 0.04 && th < 2.8);
        assert!(c_re != 0.0);
        if i > 0 {
            assert!(th > SIMPLE_TANH_MODES[i - 1].1);
        }
        i += 1;
    }
};

#[derive(Clone)]
struct SimpleTanhTarget {
    gain: f32,
    biquads: [[f32; 2]; SIMPLE_TANH_MODE_COUNT],
    ring: [f32; 32],
    ring_pos: usize,
}

impl DistillationTarget for SimpleTanhTarget {
    fn reset_state(&mut self) {
        self.biquads = [[0.0; 2]; SIMPLE_TANH_MODE_COUNT];
        self.ring = [0.0; 32];
        self.ring_pos = 0;
    }

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        let g = stim.control_values.first().copied().unwrap_or(5.0) * 0.35 + self.gain;
        let mut out_1 = Vec::with_capacity(stim.in_1.len());
        for &x in &stim.in_1 {
            let nl = (g * x).tanh();
            let wpos = self.ring_pos;
            let delayed = self.ring[(wpos + 16) & 31];
            self.ring[wpos] = nl;
            self.ring_pos = (wpos + 1) & 31;
            // Multi-mode resonant cabinet beyond the 16-sample onset delay so all 24 Stage III
            // Kautz poles carry non-trivial phase residues well above the -45 dBFS noise floor.
            let mut res = 0.0_f32;
            for (m_idx, &(r, th, c_re)) in SIMPLE_TANH_MODES.iter().enumerate() {
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
    // verify[train.pipeline.host-parity-and-export]
    let crypto = DefaultCryptoProvider::default();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 10.0, 6.0)];
    let mut target = SimpleTanhTarget {
        gain: 0.4,
        biquads: [[0.0; 2]; SIMPLE_TANH_MODE_COUNT],
        ring: [0.0; 32],
        ring_pos: 0,
    };
    let opts = AdaptiveTrainingOptions {
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        ..AdaptiveTrainingOptions::default()
    };
    let trained = train_adaptive_target(manifest, &mut target, &opts).unwrap();

    let view = DnamProjView::from_bytes(&trained.proj_bytes, &crypto).unwrap();
    let wght = view.chunk_slice(ProjChunkKind::Wght, 0).unwrap();
    let gauge_dbin = extract_dnam_bin(wght).unwrap().to_owned();
    let gv = DnamBinView::parse(&gauge_dbin).unwrap();
    let mut arena_buf = Vec::with_capacity(30_720);
    arena_buf.extend_from_slice(gv.arena0.as_bytes());
    arena_buf.extend_from_slice(gv.arena1.unwrap().as_bytes());
    arena_buf.extend_from_slice(gv.arena2.unwrap().as_bytes());
    arena_buf.extend_from_slice(gv.arena3.unwrap().as_bytes());
    apply_cascade_head_gauge(&mut arena_buf, 0, &[[1.35_f32; 4]; 32]).unwrap();
    let (mut ga0, rest) = Arena0LiteWire::read_from_prefix(&arena_buf).unwrap();
    let (ga1, rest) = Arena1FullDeltaWire::read_from_prefix(rest).unwrap();
    let (ga2, rest) = Arena2ProDeltaWire::read_from_prefix(rest).unwrap();
    let (ga3, _) = Arena3UltraDeltaWire::read_from_prefix(rest).unwrap();
    for g in &mut ga0.linear_readout_lite.direct_s2_u_gain {
        *g = F32::new(g.get() * 0.55);
    }
    for tap in &mut ga0.conv_fir_lite.fir_l {
        *tap = F32::new(tap.get() * 0.55);
    }
    let gauged_dbin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: gv.header.reference_sample_rate_hz.get(),
        header_flags: gv.header.flags.get(),
        ext_ring_pow2_frames: gv.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: gv.runtime_meta.ext_ring_lines,
        ext_ring_format: gv.runtime_meta.ext_ring_format,
        declared_max_delay_48k: gv.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: gv.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: gv.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: gv.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: gv.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: gv.runtime_meta.content_hash_prefix,
        controls: gv.control_descriptors,
        cp1_overflow: gv.cp1_overflow,
        tsyn: gv.tsyn_entries,
        cext: gv.cext,
        arena0: &ga0,
        arena1: Some(&ga1),
        arena2: Some(&ga2),
        arena3: Some(&ga3),
    })
    .unwrap();
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
        build_dnam_proj_container_with_dnam_bin(&mut patched_manifest, &chunks, &gauged_dbin)
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
        let [s0, s1] = eng.step_frame_sona(0.0, 0.0, false);
        assert!(
            s0.abs() <= 1.0e-7 && s1.abs() <= 1.0e-7,
            "{label} [STAB-ZERO-DC] failed"
        );
        eng.reset_to_dc_equilibrium();
        let mut out = Vec::with_capacity(512);
        for i in 0..640 {
            let x = 0.3 * ((i as f32) * 0.085).sin() + 0.15 * ((i as f32) * 0.23).cos();
            let y = eng.step_frame_sona(x, x, false);
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
    // verify[train.pipeline.physics-probe]
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
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        ..AdaptiveTrainingOptions::default()
    };
    let rep_faust = train_adaptive_target(m_faust, &mut faust, &opts_mono).unwrap();
    verify_whitebox_exported_audio("faust", &rep_faust.proj_bytes);
}

#[test]
fn test_tier4_whitebox_spice_runner_varpro_distillation() {
    // verify[train.pipeline.physics-probe]
    // verify[train.pipeline.fixtures-matrix]
    let mut spice = SpiceRunner::new("triode_stage.cir", 5_000);
    let mut m_spice = DnamProjManifest::default_reference();
    m_spice.base.io_config.inputs = 2;
    m_spice.base.io_config.output_mode = "stereo".to_string();
    m_spice.base.controls_schema = vec![make_range_ctrl("drive", 0.0, 1.0, 0.5)];
    let opts_stereo = AdaptiveTrainingOptions {
        topology_mode: TopologyProbeMode::TrueStereo2To2,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 2,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        ..AdaptiveTrainingOptions::default()
    };
    let rep_spice = train_adaptive_target(m_spice, &mut spice, &opts_stereo).unwrap();
    verify_whitebox_exported_audio("spice", &rep_spice.proj_bytes);
}
