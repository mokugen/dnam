#![forbid(unsafe_code)]
#![allow(dead_code)]
//! Shared `v2` container and arena fixture builders for engine integration coverage tests.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    build_dnam_container,
};
use dnam_engine::schema::{
    ControlBusConfig, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
    ControlUiHints, DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, PresetDef,
    RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TempoSubdivision, TierProfile,
};
use dnam_engine::stages::stage2_gru::{
    NUM_GRU_BLOCKS, NUM_IN_FEATS_B0, V3ControlBusParams, V3InterBlockToneSvfParams,
};
use dnam_engine::stages::stage3_cab::{V3MatryoshkaModel, V3TierReadout};
use dnam_engine::stages::weights::{
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET, ARENA0_CP1_BASE_OFFSET, ARENA0_CP1_DIR_OFFSET,
    ARENA0_RHO_HEAD_OFFSET, ARENA0_TIER_GAIN_TRIM_OFFSET, ARENA0_UINT_BASE_OFFSET,
    ARENA0_W_CTRL_OFFSET, ARENA3_CP1_ULTRA_OFFSET, ARENA3_UINT_ULTRA_OFFSET,
    ARENA3_W_CTRL_ULTRA_OFFSET, pack_v3_matryoshka_to_arena,
};

pub fn put_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

pub fn build_v2_header() -> DnamJsonHeader {
    let mut cal_anchors = BTreeMap::new();
    cal_anchors.insert("mode=0".to_string(), [20.0_f32, 120.0, 500.0]);
    let mut cond_max = BTreeMap::new();
    cond_max.insert("mode=0".to_string(), 0.85_f32);

    let drive_ctrl = ControlSchemaDef {
        id: "drive".to_string(),
        name: "Drive".to_string(),
        kind: ControlKind::Range,
        unit: Some("dB".to_string()),
        lobatto_degree: Some(2),
        anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
        declared_range: Some([0.0, 10.0]),
        safe_clamped_range: Some([0.5, 9.5]),
        default: serde_json::json!(5.0),
        conditional_min: BTreeMap::new(),
        conditional_max: cond_max,
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        calibration: None,
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    };
    let mode_switch = ControlSchemaDef {
        id: "mode".to_string(),
        name: "Mode".to_string(),
        kind: ControlKind::Switch,
        unit: None,
        lobatto_degree: None,
        anchor_nodes_norm: None,
        declared_range: None,
        safe_clamped_range: None,
        default: serde_json::json!("bright"),
        conditional_min: BTreeMap::new(),
        conditional_max: BTreeMap::new(),
        declared_options: Some(vec!["warm".to_string(), "bright".to_string()]),
        active_options: Some(vec!["warm".to_string(), "bright".to_string()]),
        clamp_warning: None,
        calibration: None,
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    };
    let delay_ctrl = ControlSchemaDef {
        id: "delay_ms".to_string(),
        name: "Delay".to_string(),
        kind: ControlKind::Range,
        unit: Some("ms".to_string()),
        lobatto_degree: Some(2),
        anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
        declared_range: Some([10.0, 800.0]),
        safe_clamped_range: Some([20.0, 500.0]),
        default: serde_json::json!(120.0),
        conditional_min: BTreeMap::new(),
        conditional_max: BTreeMap::new(),
        declared_options: None,
        active_options: None,
        clamp_warning: None,
        calibration: Some(ControlCalibration {
            role: ControlCalibrationRole::DelayTimeMs,
            lobatto_physical_anchors: Some([20.0, 120.0, 500.0]),
            conditional_physical_anchors: cal_anchors,
            tempo_sync_default_subdivision: Some(TempoSubdivision::Quarter),
        }),
        ui_hints: ControlUiHints::default(),
        user_metadata: BTreeMap::new(),
    };
    let mut preset_ctrls = BTreeMap::new();
    preset_ctrls.insert("drive".to_string(), serde_json::json!(7.2));
    preset_ctrls.insert("mode".to_string(), serde_json::json!("warm"));
    preset_ctrls.insert("delay_ms".to_string(), serde_json::json!(240.0));
    let preset = PresetDef {
        id: "tight_lead".to_string(),
        name: "Tight Lead".to_string(),
        description: Some("V2 test preset".to_string()),
        controls: preset_ctrls,
        user_metadata: BTreeMap::new(),
    };

    let mut io_config = IoConfig::default();
    io_config.capabilities.supports_trails = true;
    io_config.capabilities.tempo_sync_capable = true;

    DnamJsonHeader {
        version: "2.0.0".to_string(),
        container_kind: "dnam".to_string(),
        architecture: "DNAM_TORA".to_string(),
        sample_rate_train_hz: 48_000,
        metadata: ModelMetadata {
            name: "V2 Extended Coverage Rig".to_string(),
            author: None,
            description: None,
            license: None,
            created_at: "2026-11-14T00:00:00Z".to_string(),
            input_reference_dbfs: -20.5,
            provenance: DatasetProvenance::default(),
            distribution_policy: None,
            signatures: None,
        },
        user_metadata: BTreeMap::new(),
        io_config,
        slimmable_tiers: SlimmableTiersConfig {
            lite: TierProfile {
                feature_bus_dim: 64,
                channels: 4,
                early_fir_taps: 16,
                kautz_modes: 4,
                tank_lines: 4,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -40.0,
                arena_end_byte_offset: ARENA_OFFSET_LITE_END,
                prefix_crc32c: 0,
            },
            full: TierProfile {
                feature_bus_dim: 128,
                channels: 8,
                early_fir_taps: 32,
                kautz_modes: 8,
                tank_lines: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -45.0,
                arena_end_byte_offset: ARENA_OFFSET_FULL_END,
                prefix_crc32c: 0,
            },
            pro: TierProfile {
                feature_bus_dim: 192,
                channels: 12,
                early_fir_taps: 48,
                kautz_modes: 16,
                tank_lines: 8,
                head_svd_rank: 4,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -48.0,
                arena_end_byte_offset: ARENA_OFFSET_PRO_END,
                prefix_crc32c: 0,
            },
            ultra: TierProfile {
                feature_bus_dim: 256,
                channels: 16,
                early_fir_taps: 64,
                kautz_modes: 32,
                tank_lines: 16,
                head_svd_rank: 8,
                loudness_lufs: -20.5,
                output_trim_db: 0.0,
                validation_esr_db: -51.0,
                arena_end_byte_offset: ARENA_OFFSET_ULTRA_END,
                prefix_crc32c: 0,
            },
        },
        controls_schema: vec![drive_ctrl, mode_switch, delay_ctrl],
        presets: vec![preset],
        control_bus: ControlBusConfig::default(),
        stage0_router: RouterConfig::default(),
        stage1_pre: Stage1PreFxConfig::default(),
        stage2_cascade: Stage2CascadeConfig::default(),
        stage3_cabinet: Stage3CabinetKautzConfig::default(),
        stage4_tank: Stage4SpatialTankConfig::default(),
        delay_histories: Vec::new(),
        tensor_directory: BTreeMap::new(),
    }
}

pub fn populate_v2_arena(linear_s2: bool, mute_s4_taps: bool) -> Vec<u8> {
    let mut model = V3MatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_hz = 180.0;
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;
    model.stage0.clean_sub_mix = 0.25;
    model.stage0.op_p_sub_mix = 0.12;
    model.stage0.gate_thresh = 0.002;
    model.stage0.gate_hyst = 0.001;
    model.stage0.gate_inv_knee = 100.0;
    model.stage0.gate_floor = 0.1;
    model.svf.env_oct_sens = 1.5;
    model.svf.env_comp = 2.0;
    model.tone_svf = V3InterBlockToneSvfParams {
        h0_proj_w: [0.25, -0.15, 0.12, 0.08],
        s1_proj_w: 0.3,
        cab_inject_w: 0.2,
        b1_inject_w: 0.15,
        ..V3InterBlockToneSvfParams::identity()
    };

    for b in 0..NUM_GRU_BLOCKS {
        let blk = &mut model.blocks[b];
        for f in 0..NUM_IN_FEATS_B0 {
            for l in 0..12 {
                blk.w_in_z[f][l] = 0.06;
                blk.w_in_r[f][l] = 0.05;
                blk.w_in_n[f][l] = 0.12 * if ((f + l) & 1) == 0 { 1.0 } else { -1.0 };
            }
        }
        for l in 0..12 {
            blk.b_z[l] = -0.35;
            blk.b_r[l] = 0.25;
            blk.b_n[l] = 0.04;
            blk.k_sag[l] = 0.2;
            blk.k_quad[l] = if linear_s2 { 0.0 } else { 0.12 };
            blk.w_rec_n[l][l] = 0.12;
            if b > 0 {
                blk.w_prev_n[l][l] = 0.22;
            }
        }
    }

    let mut ctrl = V3ControlBusParams::identity();
    ctrl.w_env_fc_oct[0] = 0.35;
    ctrl.w_tone_fc_oct[0] = 0.45;
    ctrl.w_tank_psi[0] = 0.20;
    ctrl.w_film_gain[0][0][0] = 0.25;
    ctrl.w_film_bn[0][0][0] = 0.06;
    model.ctrl = ctrl;

    for t in 0..4 {
        let mut ro = V3TierReadout::default();
        for ch in 0..2 {
            for l in 0..24 {
                ro.pre_cab_w[ch][l] = 0.04 * (((l + ch + t) % 5) as f32 - 2.0);
                ro.direct_s2_w[ch][l] = 0.025 * (((l + t) % 4) as f32 + 0.5);
            }
        }
        for tap in 0..64 {
            ro.fir_l[tap] = 0.03 * (((tap + t) % 5) as f32 - 2.0);
            ro.fir_r[tap] = 0.025 * (((tap + t + 1) % 5) as f32 - 2.0);
        }
        for m in 0..16 {
            ro.kautz_b_gamma[m] = 0.35;
            ro.kautz_c_l[m] = (0.15, -0.08);
            ro.kautz_c_r[m] = (0.12, -0.06);
        }
        model.readouts[t] = ro;
        model.stage4[t].g_loop = if mute_s4_taps { 0.0 } else { 0.72 };
        model.stage4[t].wet_mix = if mute_s4_taps { 0.0 } else { 0.25 };
    }

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    pack_v3_matryoshka_to_arena(&model, &mut arena);

    for off in (ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..ARENA0_CP1_DIR_OFFSET).step_by(4) {
        put_f32(&mut arena, off, 0.25);
    }
    put_f32(&mut arena, ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + 16, 0.90);

    // 3-byte inline CP1 directory at ARENA0_CP1_DIR_OFFSET (`3_044..3_072`): 3 contributors
    arena[ARENA0_CP1_DIR_OFFSET..ARENA0_CP1_DIR_OFFSET + 2].copy_from_slice(&3_u16.to_le_bytes());
    arena[ARENA0_CP1_DIR_OFFSET + 4] = 0;
    arena[ARENA0_CP1_DIR_OFFSET + 5] = 2; // Range, 2 lanes
    arena[ARENA0_CP1_DIR_OFFSET + 6] = 0;
    arena[ARENA0_CP1_DIR_OFFSET + 7] = 1;
    arena[ARENA0_CP1_DIR_OFFSET + 8] = (1 << 4) | 1; // Switch, 1 lane
    arena[ARENA0_CP1_DIR_OFFSET + 9] = 1;
    arena[ARENA0_CP1_DIR_OFFSET + 10] = 0;
    arena[ARENA0_CP1_DIR_OFFSET + 11] = (2 << 4) | 1; // Cross-term, 1 lane
    arena[ARENA0_CP1_DIR_OFFSET + 12] = 2;

    for off in (ARENA0_UINT_BASE_OFFSET..ARENA0_UINT_BASE_OFFSET + 128).step_by(4) {
        put_f32(&mut arena, off, 0.05);
    }
    for off in (ARENA0_CP1_BASE_OFFSET..ARENA0_CP1_BASE_OFFSET + 320).step_by(4) {
        put_f32(&mut arena, off, 0.08);
    }
    for off in (ARENA0_W_CTRL_OFFSET..ARENA0_W_CTRL_OFFSET + 1024).step_by(4) {
        put_f32(&mut arena, off, 0.04);
    }
    for off in (ARENA3_W_CTRL_ULTRA_OFFSET..ARENA3_W_CTRL_ULTRA_OFFSET + 256).step_by(4) {
        put_f32(&mut arena, off, 0.03);
    }
    for off in (ARENA3_UINT_ULTRA_OFFSET..ARENA3_UINT_ULTRA_OFFSET + 384).step_by(4) {
        put_f32(&mut arena, off, 0.04);
    }
    for off in (ARENA3_CP1_ULTRA_OFFSET..ARENA3_CP1_ULTRA_OFFSET + 512).step_by(4) {
        put_f32(&mut arena, off, 0.06);
    }
    for i in 0..8 {
        put_f32(
            &mut arena,
            ARENA0_RHO_HEAD_OFFSET + i * 4,
            0.5 - 0.04 * (i as f32),
        );
    }
    for i in 0..4 {
        put_f32(
            &mut arena,
            ARENA0_TIER_GAIN_TRIM_OFFSET + i * 4,
            1.0 - 0.01 * (i as f32),
        );
    }

    arena
}

pub fn build_v2_extended_container(linear_s2: bool, mute_s4_taps: bool) -> Vec<u8> {
    let mut header = build_v2_header();
    let arena = populate_v2_arena(linear_s2, mute_s4_taps);
    build_dnam_container(&mut header, &arena, &arena, false).unwrap()
}
