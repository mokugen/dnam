#![forbid(unsafe_code)]
#![allow(dead_code)]
//! Shared `v2` container and arena fixture builders for engine integration coverage tests.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    build_dnam_container,
};
use dnam_engine::control::{ConditionalClampRule, ControlClampDescriptor};
use dnam_engine::provenance::{STAGE3_EARLY_FIR_BASES, kautz_soa_byte_offset};
use dnam_engine::schema::{
    ControlBusConfig, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
    ControlUiHints, DatasetProvenance, DnamJsonHeader, IoConfig, ModelMetadata, PresetDef,
    RouterConfig, SlimmableTiersConfig, Stage1PreFxConfig, Stage2CascadeConfig,
    Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TempoSubdivision, TierProfile,
};
use dnam_engine::stages::weights::{
    ARENA0_PIPELINE_FLAGS_OFFSET, ARENA0_RHO_HEAD_OFFSET, ARENA0_STAGE1_EXT_OFFSET,
    ARENA0_STAGE2_EXT_OFFSET, ARENA0_STAGE4_EXT_OFFSET, ARENA0_TIER_GAIN_TRIM_OFFSET,
    ARENA0_W_CTRL_OFFSET, ARENA1_STAGE1_DELTA_PRE_OFFSET, ARENA1_STAGE4_TANK_DELTA_OFFSET,
    ARENA2_CONTROL_BUS_PROJ_DELTA_OFFSET, ARENA3_STAGE4_TANK_SPATIAL_16_OFFSET, PIPE_FLAG_ALL_V2,
    PIPE_MAGIC_HI,
};

pub fn put_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

pub fn write_clamp_descriptor(arena: &mut [u8], slot_idx: usize, desc: &ControlClampDescriptor) {
    let base = 8_512 + slot_idx * 64;
    arena[base] = desc.ctrl_kind;
    arena[base + 1] = desc.lobatto_degree;
    arena[base + 2..base + 4].copy_from_slice(&desc.active_options_mask.to_le_bytes());
    put_f32(arena, base + 4, desc.default_val);
    put_f32(arena, base + 8, desc.safe_min_base);
    put_f32(arena, base + 12, desc.safe_max_base);
    put_f32(arena, base + 16, desc.scale_norm);
    put_f32(arena, base + 20, desc.offset_norm);
    for i in 0..3 {
        put_f32(arena, base + 24 + i * 4, desc.anchor_nodes_norm[i]);
    }
    arena[base + 36..base + 40].copy_from_slice(&desc._reserved.to_le_bytes());
    for r in 0..3 {
        let rb = base + 40 + r * 8;
        let rule = &desc.cond_rules[r];
        arena[rb..rb + 2].copy_from_slice(&rule.switch_ctrl_idx.to_le_bytes());
        arena[rb + 2] = rule.switch_opt_idx;
        arena[rb + 3] = rule.rule_op;
        put_f32(arena, rb + 4, rule.bound_val);
    }
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
    let mut arena = vec![0u8; 64_000];
    arena[ARENA0_PIPELINE_FLAGS_OFFSET..ARENA0_PIPELINE_FLAGS_OFFSET + 4]
        .copy_from_slice(&(PIPE_MAGIC_HI | PIPE_FLAG_ALL_V2).to_le_bytes());

    // Stage 0 router scattering matrix (0..64), dry/wet (116..124), r_ac (124), crossover fc (128)
    let s_mat = [
        [0.0_f32, 0.72, 0.25, 0.08],
        [0.72, 0.0, 0.12, -0.05],
        [0.25, 0.10, 0.40, 0.15],
        [0.08, -0.04, 0.10, 0.50],
    ];
    for r in 0..4 {
        for c in 0..4 {
            put_f32(&mut arena, (r * 4 + c) * 4, s_mat[r][c]);
        }
    }
    put_f32(&mut arena, 116, 0.25);
    put_f32(&mut arena, 120, 0.85);
    put_f32(&mut arena, 124, 0.9993);
    put_f32(&mut arena, 128, 180.0);

    // Stage I poles & extended params (256..488 + 10_752..10_832)
    for lane in 0..4 {
        put_f32(&mut arena, 256 + lane * 4, 0.70 + 0.04 * (lane as f32));
        put_f32(&mut arena, 272 + lane * 4, 0.05);
        put_f32(&mut arena, 288 + lane * 4, 0.35 + 0.15 * (lane as f32));
        put_f32(&mut arena, 304 + lane * 4, 0.45);
        put_f32(&mut arena, 320 + lane * 4, 0.20);
        put_f32(&mut arena, 336 + lane * 4, 0.02);

        put_f32(&mut arena, ARENA1_STAGE1_DELTA_PRE_OFFSET + lane * 4, 0.65);
        put_f32(
            &mut arena,
            ARENA1_STAGE1_DELTA_PRE_OFFSET + 16 + lane * 4,
            0.06,
        );
        put_f32(
            &mut arena,
            ARENA1_STAGE1_DELTA_PRE_OFFSET + 32 + lane * 4,
            0.4 + 0.2 * (lane as f32),
        );
        put_f32(
            &mut arena,
            ARENA1_STAGE1_DELTA_PRE_OFFSET + 48 + lane * 4,
            0.5,
        );
        put_f32(
            &mut arena,
            ARENA1_STAGE1_DELTA_PRE_OFFSET + 64 + lane * 4,
            0.25,
        );
    }
    put_f32(&mut arena, 352, 1.35);
    arena[356..360].copy_from_slice(&2_u32.to_le_bytes());

    let s1_ext: [f32; 32] = [
        0.35, 0.25, 4.0, 0.01, 0.002, 0.0001, 1.2, 0.6, 0.5, 0.04, 0.08, 0.004, 0.15, 0.20, 0.10,
        0.30, 140.0, 0.40, 120.0, 24.0, 0.8, 1.1, 2.5, 8.0, 0.25, 0.45, 450.0, 1.5, 0.7, 0.6, 0.4,
        0.2,
    ];
    for (i, &v) in s1_ext.iter().enumerate() {
        put_f32(&mut arena, ARENA0_STAGE1_EXT_OFFSET + i * 4, v);
    }

    // Stage II cascade poles (4_096..4_288) & extended (512..552)
    for off in (4_096..4_288).step_by(4) {
        put_f32(&mut arena, off, 0.45);
    }
    put_f32(&mut arena, 4_096, 0.985);
    put_f32(&mut arena, 4_100, 0.93);
    put_f32(&mut arena, 4_104, 0.82);
    put_f32(&mut arena, 4_108, 0.992);
    put_f32(&mut arena, 4_272, 0.25);
    put_f32(&mut arena, 4_276, 0.85);
    put_f32(&mut arena, 4_280, 0.96);
    put_f32(&mut arena, 4_284, 0.994);

    let s2_ext = [
        0.85_f32,
        0.45,
        0.20,
        0.25,
        0.30,
        0.20,
        0.15,
        0.25,
        if linear_s2 { 1.0 } else { 0.0 },
        1.15,
    ];
    for (i, &v) in s2_ext.iter().enumerate() {
        put_f32(&mut arena, ARENA0_STAGE2_EXT_OFFSET + i * 4, v);
    }

    let cascade_w_bases = [1_024_usize, 11_264_usize, 22_336_usize, 36_224_usize];
    let cascade_w_counts = [768_usize, 1_536_usize, 2_304_usize, 3_072_usize];
    for tier in 0..4 {
        let cw_base = cascade_w_bases[tier];
        for w in 0..cascade_w_counts[tier] {
            let val = 0.015_f32 * (((w + tier * 5) % 9) as f32 - 4.0);
            put_f32(&mut arena, cw_base + w * 4, val);
        }
        for blk in 0..4 {
            let k_base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                put_f32(&mut arena, k_base + lane * 4, 0.55);
                put_f32(&mut arena, k_base + 16 + lane * 4, 0.05);
                put_f32(&mut arena, k_base + 32 + lane * 4, 0.45);
                put_f32(&mut arena, k_base + 48 + lane * 4, 0.25);
                put_f32(&mut arena, k_base + 64 + lane * 4, 0.15);
                put_f32(&mut arena, k_base + 80 + lane * 4, -0.10);
            }
        }
        let fir_base = STAGE3_EARLY_FIR_BASES[tier];
        for t in 0..16 {
            let v = 0.05_f32 * (((t + tier) % 5) as f32 - 2.0);
            put_f32(&mut arena, fir_base + t * 4, v);
        }
    }

    for &hbase in &[4_928_usize, 18_048_usize, 31_936_usize, 57_600_usize] {
        for i in 0..640 {
            let val = 0.04_f32 * (((i * 11 + 3) % 15) as f32 - 7.0);
            put_f32(&mut arena, hbase + i * 4, val);
        }
    }

    // Stage IV tank + extended BBD / FDN / shimmer + Multi-tier delays
    for (i, &d) in [149.0_f32, 179.0, 211.0, 239.0].iter().enumerate() {
        put_f32(&mut arena, 4_672 + i * 4, d);
    }
    put_f32(&mut arena, 4_688, 0.82);
    put_f32(&mut arena, 4_692, 0.25);
    put_f32(&mut arena, 4_696, 0.85);
    put_f32(&mut arena, 4_700, 0.30);

    let s4_ext: [f32; 17] = [
        if mute_s4_taps { -1.0 } else { 0.7 },
        if mute_s4_taps { 0.0 } else { 0.45 },
        320.0,
        410.0,
        0.65,
        0.45,
        0.75,
        if mute_s4_taps { 0.0 } else { 0.5 },
        512.0,
        733.0,
        997.0,
        1_301.0,
        0.75,
        0.70,
        0.35,
        8.0,
        1.5,
    ];
    for (i, &v) in s4_ext.iter().enumerate() {
        put_f32(&mut arena, ARENA0_STAGE4_EXT_OFFSET + i * 4, v);
    }
    for i in 0..4 {
        put_f32(
            &mut arena,
            ARENA1_STAGE4_TANK_DELTA_OFFSET + i * 4,
            280.0 + (i as f32) * 37.0,
        );
        put_f32(
            &mut arena,
            ARENA2_CONTROL_BUS_PROJ_DELTA_OFFSET + i * 4,
            410.0 + (i as f32) * 53.0,
        );
    }
    for i in 0..8 {
        put_f32(
            &mut arena,
            ARENA3_STAGE4_TANK_SPATIAL_16_OFFSET + i * 4,
            610.0 + (i as f32) * 61.0,
        );
    }

    // Control bus tables (7_488..10_688 and Ultra 60_672..63_424)
    for off in (7_488..7_524).step_by(4) {
        put_f32(&mut arena, off, 0.25);
    }
    put_f32(&mut arena, 7_488 + 16, 0.90);
    // 3-byte inline CP1 directory at 7_524..7_552: 3 contributors
    arena[7_524..7_526].copy_from_slice(&3_u16.to_le_bytes());
    arena[7_528] = 0;
    arena[7_529] = 2; // Range, 2 lanes
    arena[7_530] = 0;
    arena[7_531] = 1;
    arena[7_532] = (1 << 4) | 1; // Switch, 1 lane
    arena[7_533] = 1;
    arena[7_534] = 0;
    arena[7_535] = (2 << 4) | 1; // Cross-term, 1 lane
    arena[7_536] = 2;
    for off in (7_552..8_512).step_by(4) {
        put_f32(&mut arena, off, 0.14);
    }
    for off in (60_928..63_424).step_by(4) {
        put_f32(&mut arena, off, 0.08);
    }
    for off in (ARENA0_W_CTRL_OFFSET..ARENA0_W_CTRL_OFFSET + 1024).step_by(4) {
        put_f32(&mut arena, off, 0.06);
    }
    for off in (60_672..60_928).step_by(4) {
        put_f32(&mut arena, off, 0.05);
    }
    for i in 0..8 {
        put_f32(
            &mut arena,
            ARENA0_RHO_HEAD_OFFSET + i * 4,
            0.5 - 0.04 * (i as f32),
        );
        put_f32(
            &mut arena,
            ARENA0_TIER_GAIN_TRIM_OFFSET + i * 4,
            1.0 - 0.01 * (i as f32),
        );
    }

    let d0 = ControlClampDescriptor {
        ctrl_kind: 0,
        lobatto_degree: 2,
        active_options_mask: 0,
        default_val: 5.0,
        safe_min_base: 0.5,
        safe_max_base: 9.5,
        scale_norm: 0.2,
        offset_norm: -1.0,
        anchor_nodes_norm: [0.0, -1.0, 1.0],
        _reserved: 0,
        cond_rules: [
            ConditionalClampRule {
                switch_ctrl_idx: 1,
                switch_opt_idx: 0,
                rule_op: 1,
                bound_val: 8.5,
            },
            ConditionalClampRule::empty(),
            ConditionalClampRule::empty(),
        ],
    };
    let d1 = ControlClampDescriptor {
        ctrl_kind: 1,
        lobatto_degree: 1,
        active_options_mask: 0b0011,
        default_val: 1.0,
        safe_min_base: 0.0,
        safe_max_base: 1.0,
        scale_norm: 1.0,
        offset_norm: 0.0,
        anchor_nodes_norm: [0.0; 3],
        _reserved: 0,
        cond_rules: [ConditionalClampRule::empty(); 3],
    };
    let d2 = ControlClampDescriptor {
        ctrl_kind: 0,
        lobatto_degree: 2,
        active_options_mask: 0,
        default_val: 120.0,
        safe_min_base: 20.0,
        safe_max_base: 500.0,
        scale_norm: 1.0 / 395.0,
        offset_norm: -1.025,
        anchor_nodes_norm: [0.0, -1.0, 1.0],
        _reserved: 0,
        cond_rules: [ConditionalClampRule::empty(); 3],
    };
    write_clamp_descriptor(&mut arena, 0, &d0);
    write_clamp_descriptor(&mut arena, 1, &d1);
    write_clamp_descriptor(&mut arena, 2, &d2);

    arena
}

pub fn build_v2_extended_container(linear_s2: bool, mute_s4_taps: bool) -> Vec<u8> {
    let mut header = build_v2_header();
    let arena = populate_v2_arena(linear_s2, mute_s4_taps);
    build_dnam_container(&mut header, &arena, &arena, false).unwrap()
}
