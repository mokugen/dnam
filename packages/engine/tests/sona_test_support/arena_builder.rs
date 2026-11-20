#![forbid(unsafe_code)]
#![allow(dead_code)]
//! Shared `SONA` (`v3.0`) container and arena fixture builders for engine integration tests.

use std::collections::BTreeMap;

use dnam_engine::container::{
    ARENA_OFFSET_ULTRA_END, StructuredConditionalBound, StructuredConditionalCurve,
    build_dnam_container,
};
use dnam_engine::graph::sona_core::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_TOTAL_SONA_LANES, SONA_FIR_MAX_TAPS,
    SonaControlBusParams, SonaCrossbarConfig, SonaMatryoshkaModel, SonaTierReadout,
};
use dnam_engine::graph::weights::{
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET, ARENA0_CP1_BASE_OFFSET, ARENA0_CP1_DIR_OFFSET,
    ARENA0_RHO_HEAD_OFFSET, ARENA0_TIER_GAIN_TRIM_OFFSET, ARENA0_UINT_BASE_OFFSET,
    ARENA0_W_CTRL_OFFSET, ARENA3_CP1_ULTRA_OFFSET, ARENA3_UINT_ULTRA_OFFSET,
    ARENA3_W_CTRL_ULTRA_OFFSET, Cp1InlineDirWire, pack_sona_matryoshka_to_arena, write_f32_le,
};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, TempoSubdivision,
};
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::U16;

pub fn put_f32(buf: &mut [u8], off: usize, val: f32) {
    write_f32_le(buf, off, val);
}

pub fn build_sona_header() -> DnamJsonHeader {
    let cal_anchors = vec![StructuredConditionalCurve {
        when: vec![("mode".to_string(), "warm".to_string())],
        anchors: [5.0, 200.0, 800.0],
    }];

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
        conditional_bounds: vec![StructuredConditionalBound {
            when: vec![("mode".to_string(), "warm".to_string())],
            min: None,
            max: Some(8.5),
        }],
        represented_switch_options: None,
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
        conditional_bounds: Vec::new(),
        represented_switch_options: None,
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
        conditional_bounds: Vec::new(),
        represented_switch_options: None,
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
        description: Some("SONA v3.0 test preset".to_string()),
        controls: preset_ctrls,
        user_metadata: BTreeMap::new(),
    };

    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = "SONA Extended Coverage Rig".to_string();
    header.metadata.created_at = "2026-11-18T00:00:00Z".to_string();
    header.io_config.capabilities.supports_trails = true;
    header.io_config.capabilities.tempo_sync_capable = true;
    header.controls_schema = vec![drive_ctrl, mode_switch, delay_ctrl];
    header.presets = vec![preset];
    header
}

pub fn populate_sona_arena(linear_s2: bool, mute_s4_taps: bool) -> Vec<u8> {
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_hz = 180.0;
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;
    model.stage0.clean_sub_mix = 0.25;
    model.stage0.gate_thresh = 0.002;
    model.stage0.gate_hyst = 0.001;
    model.stage0.gate_inv_knee = 100.0;
    model.stage0.gate_floor = 0.1;

    let mut crossbar = SonaCrossbarConfig::identity();
    crossbar.op_p_crossbar_gain = 0.15;
    crossbar.op_t_crossbar_gain = 0.10;
    crossbar.phasor_freq_hz = [0.6, 1.3, 0.0, 0.0];
    crossbar.phasor_stereo_phase_rad = [0.4, 0.8, 0.0, 0.0];
    for g in 0..MAX_SONA_GROUPS {
        crossbar.op_p_to_group[g][0] = 0.08;
        crossbar.phasor_to_group[g][0] = 0.05;
    }
    model.crossbar = crossbar;

    for g in 0..MAX_SONA_GROUPS {
        let grp = &mut model.groups[g];
        for l in 0..4 {
            grp.svf_g[l] = 0.18 + 0.06 * (l as f32);
            grp.svf_k[l] = core::f32::consts::FRAC_1_SQRT_2;
            grp.lru_alpha_rel[l] = 0.90;
            grp.in_x_weight[l] = if l == 0 {
                0.75
            } else {
                0.12 * if ((g + l) & 1) == 0 { 1.0 } else { -1.0 }
            };
            grp.tone_lp[l] = 0.2;
            grp.tone_bp[l] = 0.1;
            grp.tone_dir[l] = 0.7;
            grp.rect_sign[l] = 1.0;
            if linear_s2 {
                grp.adaa_drive[l] = 0.0;
                grp.wavefolder_mix[l] = 0.0;
                grp.sag_depth[l] = 0.0;
                grp.lru_kappa[l] = 0.0;
                grp.lru_alpha_atk[l] = 0.0;
                grp.lru_bias_drive[l] = 0.0;
            } else {
                grp.adaa_drive[l] = 0.65;
                grp.wavefolder_mix[l] = 0.08;
                grp.sag_depth[l] = 0.20;
                grp.lru_kappa[l] = 0.35;
                grp.lru_alpha_atk[l] = 0.05;
                grp.lru_bias_drive[l] = 0.04;
            }
        }
    }

    for tile in &mut model.tiles[..MAX_SONA_TILES] {
        for col in 0..4 {
            for row in 0..4 {
                tile.w_odd_4x4[col][row] = if col == row { 0.14 } else { 0.03 };
                tile.w_even_4x4[col][row] = if !linear_s2 && col == row { 0.04 } else { 0.0 };
                tile.w_lru_4x4[col][row] = if col == row { 0.05 } else { 0.0 };
            }
        }
    }

    let mut ctrl = SonaControlBusParams::identity();
    ctrl.w_svf_g_oct[0] = 0.35;
    ctrl.w_svf_k_delta[0] = 0.15;
    ctrl.w_kautz_oct[0] = 0.12;
    ctrl.w_kautz_damp[0] = 0.10;
    ctrl.w_tank_psi[0] = 0.20;
    ctrl.w_adaa_drive_scale[0][0] = 0.25;
    ctrl.w_lru_bias_shift[0][0] = if linear_s2 { 0.0 } else { 0.06 };
    model.ctrl = ctrl;

    for m in 0..MAX_KAUTZ_POLES {
        model.kautz_poles[m] = (
            0.92 + 0.002 * ((m & 3) as f32),
            0.12 + 0.05 * ((m % 8) as f32),
        );
    }

    for t in 0..4 {
        let mut ro = SonaTierReadout::default();
        for ch in 0..2 {
            for l in 0..MAX_TOTAL_SONA_LANES {
                ro.pre_cab_w_odd[ch][l] = 0.04 * (((l + ch + t) % 5) as f32 - 2.0);
                ro.pre_cab_w_even[ch][l] = if linear_s2 { 0.0 } else { 0.015 };
                ro.pre_cab_w_lru[ch][l] = 0.01;
                ro.direct_w_odd[ch][l] = 0.025 * (((l + t) % 4) as f32 + 0.5);
            }
        }
        for tap in 0..SONA_FIR_MAX_TAPS {
            ro.fir_l[tap] = 0.03 * (((tap + t) % 5) as f32 - 2.0);
            ro.fir_r[tap] = 0.025 * (((tap + t + 1) % 5) as f32 - 2.0);
        }
        for m in 0..MAX_KAUTZ_POLES {
            ro.kautz_b_gamma[m] = 0.35 + 0.02 * (t as f32);
            ro.kautz_c_l[m] = (0.15, -0.08);
            ro.kautz_c_r[m] = (0.12, -0.06);
        }
        model.readouts[t] = ro;
        model.stage4[t].g_loop = if mute_s4_taps { 0.0 } else { 0.72 };
        model.stage4[t].wet_mix = if mute_s4_taps { 0.0 } else { 0.25 };
    }

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    pack_sona_matryoshka_to_arena(&model, &mut arena);

    for off in (ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..ARENA0_CP1_DIR_OFFSET).step_by(4) {
        put_f32(&mut arena, off, 0.25);
    }
    put_f32(&mut arena, ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + 12, 0.999);
    put_f32(&mut arena, ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + 16, 0.90);
    put_f32(&mut arena, ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET + 32, 2.5);

    // 3-byte inline CP1 directory at ARENA0_CP1_DIR_OFFSET via zerocopy Cp1InlineDirWire
    if let Ok((cp1_dir, _)) = Cp1InlineDirWire::mut_from_prefix(&mut arena[ARENA0_CP1_DIR_OFFSET..])
    {
        cp1_dir.cp1_dir_count = U16::new(3);
        cp1_dir.flags = U16::new(0);
        cp1_dir.entries[0] = [0, 2, 0]; // Range, 2 lanes
        cp1_dir.entries[1] = [1, (1 << 4) | 1, 1]; // Switch, 1 lane
        cp1_dir.entries[2] = [0, (2 << 4) | 1, 2]; // Cross-term, 1 lane
    }

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

pub fn build_sona_extended_container(linear_s2: bool, mute_s4_taps: bool) -> Vec<u8> {
    let mut header = build_sona_header();
    let arena = populate_sona_arena(linear_s2, mute_s4_taps);
    build_dnam_container(&mut header, &arena, &arena, false).unwrap()
}

pub fn build_populated_sona_container() -> (DnamJsonHeader, Vec<u8>, Vec<u8>) {
    let mut header = DnamJsonHeader::default_reference();
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;
    model.stage0.clean_sub_mix = 0.25;
    model.crossbar.op_p_crossbar_gain = 0.15;
    model.crossbar.op_t_crossbar_gain = 0.10;
    for g in 0..MAX_SONA_GROUPS {
        model.crossbar.op_p_to_group[g][0] = 0.08;
        let grp = &mut model.groups[g];
        for l in 0..4 {
            grp.svf_g[l] = 0.18 + 0.05 * ((g + l) as f32 * 0.25);
            grp.svf_k[l] = core::f32::consts::FRAC_1_SQRT_2;
            grp.lru_alpha_rel[l] = 0.90;
            grp.in_x_weight[l] = if l == 0 {
                0.70 + 0.04 * (g as f32)
            } else {
                0.14 * if ((g + l) & 1) == 0 { 1.0 } else { -1.0 }
            };
            grp.adaa_drive[l] = 0.65;
            grp.wavefolder_mix[l] = 0.08;
            grp.sag_depth[l] = 0.20;
            grp.lru_kappa[l] = 0.25;
            grp.lru_alpha_atk[l] = 0.05;
            grp.lru_bias_drive[l] = 0.0;
            grp.tone_lp[l] = 0.20;
            grp.tone_bp[l] = 0.10;
            grp.tone_dir[l] = 0.70;
            grp.rect_sign[l] = 1.0;
        }
    }
    for tile in &mut model.tiles[..MAX_SONA_TILES] {
        for col in 0..4 {
            for row in 0..4 {
                tile.w_odd_4x4[col][row] = if col == row { 0.15 } else { 0.03 };
                tile.w_lru_4x4[col][row] = if col == row { 0.05 } else { 0.0 };
            }
        }
    }
    for m in 0..MAX_KAUTZ_POLES {
        model.kautz_poles[m] = (0.90, 0.15 + 0.04 * ((m % 8) as f32));
    }
    for tier in 0..4 {
        let mut ro = SonaTierReadout::default();
        for ch in 0..2 {
            for l in 0..MAX_TOTAL_SONA_LANES {
                ro.pre_cab_w_odd[ch][l] =
                    0.05 * ((tier + 1) as f32) * if (l & 1) == 0 { 1.0 } else { -0.5 };
                ro.pre_cab_w_lru[ch][l] = 0.015 * ((tier + 1) as f32);
                ro.direct_w_odd[ch][l] = 0.03 * ((tier + 1) as f32);
            }
        }
        for tap in 0..SONA_FIR_MAX_TAPS {
            ro.fir_l[tap] = 0.05 * ((tier + 1) as f32) * if tap % 4 == 0 { 1.0 } else { 0.1 };
            ro.fir_r[tap] = 0.045 * ((tier + 1) as f32) * if tap % 4 == 0 { 1.0 } else { 0.1 };
        }
        for m in 0..MAX_KAUTZ_POLES {
            ro.kautz_b_gamma[m] = 0.35 + 0.03 * (tier as f32);
            ro.kautz_c_l[m] = (0.25 * ((tier + 1) as f32), -0.15);
            ro.kautz_c_r[m] = (0.20 * ((tier + 1) as f32), -0.10);
        }
        model.readouts[tier] = ro;
        model.stage4[tier].g_loop = 0.72;
        model.stage4[tier].wet_mix = 0.18 + 0.04 * (tier as f32);
    }
    pack_sona_matryoshka_to_arena(&model, &mut arena);

    let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    (header, arena, container)
}
