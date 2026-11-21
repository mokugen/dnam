#![forbid(unsafe_code)]
#![allow(dead_code)]
//! Shared Layer 1 `SONA` (`v3.0`) container and `.dnam-bin` fixture builders for engine tests and benches.

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{
    DnamPreamble, StructuredConditionalBound, StructuredConditionalCurve,
    build_dnam_container_from_dnam_bin, build_dnam_container_with_dbin, crc32c_update,
    encode_cp1_shared_stream,
};
use dnam_engine::schema::{
    ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    DnamJsonHeader, PresetDef, TempoSubdivision,
};
use dnam_kernels::{TANK_MAX_SIGMA_SAT_FP16, f32_to_fp16};
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3UltraDeltaWire, CONVOLVER_KAUTZ_BLOCKS_PER_TIER, CextDispatchWire, DNBN_FLAG_AUTO_UNITY,
    DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, DNBN_FLAG_WET_POLARITY_INV, DNBN_VERSION_MAJOR,
    DnamBinBuildInput, DnamBinHeaderWire, KautzSoaBlockWire, PIPE_CANONICAL_V3_SONA,
    SONA_CELL_MAGIC, SONA_META_CORE_MODS_PRESENT, SONA_META_OS_EXT_PRESENT, SONA_META_OS2X,
    SONA_META_SENSORY_QUAD, SONA_META_TIER_OS_EXPLICIT, SONA_META_TRANSPORT_LFO,
    SONA_META_TRANSPORT_TAP_GAINS, SonaCrossbarBlockWire, SonaInterBlockTilePairWire,
    TRANSPORT_FLAG_SCHROEDER_ENABLE, TempoSyncEntryWire, TransportQuadWire, encode_dnam_bin,
};
use dnam_vm::control::Cp1ContributorSlot;
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, FromZeros, IntoBytes};

pub fn put_f32(buf: &mut [u8], off: usize, val: f32) {
    let bytes = val.to_le_bytes();
    buf[off..off + 4].copy_from_slice(&bytes);
}

pub fn read_f32(buf: &[u8], off: usize) -> f32 {
    F32::read_from_prefix(&buf[off..]).map_or(0.0, |(v, _)| v.get())
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

fn fill_core_quad(quad: &mut dnam_vm::bin::CoreQuadWire, g: usize, linear_s2: bool) {
    for l in 0..4 {
        quad.svf_g[l] = F32::new(0.18 + 0.05 * (l as f32));
        quad.lru_alpha_rel[l] = F32::new(0.90);
        quad.lru_alpha_atk[l] = F32::new(if linear_s2 { 0.0 } else { 0.05 });
        quad.svf_k_fp16[l] = U16::new(f32_to_fp16(core::f32::consts::FRAC_1_SQRT_2));
        quad.lru_kappa_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.35 }));
        quad.adaa_drive_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.65 }));
        quad.lru_bias_drive_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.04 }));
        quad.tone_lp_fp16[l] = U16::new(f32_to_fp16(0.20));
        quad.tone_bp_fp16[l] = U16::new(f32_to_fp16(0.10));
        quad.tone_dir_fp16[l] = U16::new(f32_to_fp16(0.70));
        quad.sag_depth_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.20 }));
        quad.wavefolder_mix_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.08 }));
        quad.rect_sign_fp16[l] = U16::new(f32_to_fp16(1.0));
        let in_w = if l == 0 {
            0.70 + 0.02 * (g as f32)
        } else {
            0.12 * if ((g + l) & 1) == 0 { 1.0 } else { -1.0 }
        };
        quad.in_mid_weight_fp16[l] = U16::new(f32_to_fp16(in_w));
        quad.in_side_weight_fp16[l] = U16::new(f32_to_fp16(0.5 * in_w));
        quad.lru_stereo_link_fp16[l] = U16::new(f32_to_fp16(0.25));
        quad.lru_duhem_gamma_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 1.5 }));
        quad.lru_signed_mix_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.6 }));
        quad.glu_mix_fp16[l] = U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { 0.25 }));
    }
}

fn fill_tile(tile: &mut SonaInterBlockTilePairWire, linear_s2: bool) {
    for col in 0..4 {
        for row in 0..4 {
            let odd = if col == row { 0.14 } else { 0.03 };
            let even = if !linear_s2 && col == row { 0.04 } else { 0.0 };
            let lru = if col == row { 0.05 } else { 0.0 };
            tile.w_odd_4x4[col][row] = F32::new(odd);
            tile.w_even_4x4_fp16[col][row] = U16::new(f32_to_fp16(even));
            tile.w_lru_4x4_fp16[col][row] = U16::new(f32_to_fp16(lru));
        }
    }
}

fn fill_kautz_block(blk: &mut KautzSoaBlockWire, blk_global: usize) {
    for lane in 0..4 {
        let ch = blk_global * 4 + lane;
        blk.r_low[lane] = F32::new(0.90 + 0.004 * (lane as f32));
        blk.dr[lane] = F32::new(0.02);
        blk.theta[lane] = F32::new(0.10 + 0.05 * ((ch % 8) as f32));
        blk.b_gamma[lane] = F32::new(0.35 + 0.005 * (ch as f32));
        blk.c_re_l[lane] = F32::new(0.15);
        blk.c_im_l[lane] = F32::new(-0.08);
        blk.c_re_r[lane] = F32::new(0.12);
        blk.c_im_r[lane] = F32::new(-0.06);
    }
}

fn fill_transport_quad(tq: &mut TransportQuadWire, tier_idx: usize, mute_s4_taps: bool) {
    let base_delay = 24.0 + 17.0 * (tier_idx as f32);
    for l in 0..4 {
        tq.delay_samples[l] = F32::new(base_delay + 7.0 * (l as f32));
        tq.ramp_spans_samples[l] = F32::new(64.0);
        tq.lfo_depths_fp16[l] = U16::new(f32_to_fp16(if mute_s4_taps { 0.0 } else { 1.5 }));
        tq.input_gains_fp16[l] = U16::new(f32_to_fp16(0.5));
        let tap = if mute_s4_taps { 0.0 } else { 0.25 };
        tq.wet_tap_l_fp16[l] = U16::new(f32_to_fp16(tap));
        tq.wet_tap_r_fp16[l] = U16::new(f32_to_fp16(-tap * 0.8));
        tq.allpass_g_fp16[l] = U16::new(f32_to_fp16(0.35));
        tq.allpass_delays[l] = (13 + l * 6) as u8;
    }
    tq.g_loop = F32::new(if mute_s4_taps { 0.0 } else { 0.72 });
    tq.r_dc = F32::new(dnam_vm::math::TANK_DC_BLOCKER_48K_F32);
    tq.damp_pole = F32::new(0.85);
    tq.lfo_rate_hz = F32::new(1.4 + 0.2 * (tier_idx as f32));
    tq.sigma_knee_fp16 = U16::new(TANK_MAX_SIGMA_SAT_FP16);
    tq.freeze_mix_fp16 = U16::new(f32_to_fp16(if mute_s4_taps { 0.0 } else { 0.25 }));
    tq.flags = TRANSPORT_FLAG_SCHROEDER_ENABLE;
}

fn fill_crossbar_block(cb: &mut SonaCrossbarBlockWire, active_groups: usize, tf: f32) {
    for g in 0..active_groups {
        for l in 0..4 {
            cb.op_p_to_core_fp16[g][l] =
                U16::new(f32_to_fp16(if l == 0 { 0.06 * tf } else { 0.02 }));
            cb.phasor_to_core_fp16[g][l] =
                U16::new(f32_to_fp16(if l == 0 { 0.04 * tf } else { 0.01 }));
        }
    }
    for t in 0..16 {
        cb.opt_to_lr_fp16[t][0] = U16::new(f32_to_fp16(0.05 * tf));
        cb.opt_to_lr_fp16[t][1] = U16::new(f32_to_fp16(-0.04 * tf));
    }
    for ph in 0..4 {
        cb.phasor_stereo_phase_fp16[ph] = U16::new(f32_to_fp16(0.2 * ((ph + 1) as f32)));
    }
}

fn fill_ro_odd_words(w_ro: &mut [U16], lane_count: usize, tier_idx: usize) {
    let tf = (tier_idx + 1) as f32;
    for l in 0..lane_count {
        let base = l * 12;
        let w = 0.04 * tf * (((l + tier_idx) % 5) as f32 - 1.5);
        w_ro[base] = U16::new(f32_to_fp16(w));
        w_ro[base + 1] = U16::new(f32_to_fp16(-w * 0.8));
        w_ro[base + 6] = U16::new(f32_to_fp16(0.025 * tf));
        w_ro[base + 7] = U16::new(f32_to_fp16(0.022 * tf));
    }
}

fn populate_inline_control_bus(
    a0: &mut Arena0LiteWire,
    a3: &mut Arena3UltraDeltaWire,
    controls: &[ControlSchemaDef],
    populate_cp1_inline: bool,
    linear_s2: bool,
) {
    let cb = &mut a0.control_bus_lite;
    cb.lfo_dphi_train = [F32::new(0.002), F32::new(0.005)];
    cb.sigma_drift = F32::new(0.05);
    cb.rho_drift_train = F32::new(0.999);
    cb.lfo_skew_fp16 = [U16::new(f32_to_fp16(0.25)), U16::new(f32_to_fp16(-0.25))];
    cb.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.90));
    cb.ou_to_z_stage_fp16 = [f32_to_fp16(0.05); 4].map(U16::new);
    cb.trust_radius_mh = F32::new(2.5);
    cb.rho_head_and_ctrl_tail[0] = F32::new(2.5);
    for i in 24..32 {
        cb.rho_head_and_ctrl_tail[i] = F32::new(if i % 2 == 0 { 0.05 } else { -0.04 });
    }

    for r in 0..4 {
        for c in 0..8 {
            cb.u_int_base[r][c] = F32::new(0.05);
        }
    }
    for r in 0..16 {
        for c in 0..16 {
            cb.w_ctrl[r][c] = F32::new(0.04);
        }
    }

    for r in 0..16 {
        for c in 0..4 {
            a3.control_bus_ultra_delta.w_ctrl_ultra[r][c] = F32::new(0.18);
        }
    }
    for b in 0..3 {
        for r in 0..4 {
            for c in 0..8 {
                a3.control_bus_ultra_delta.u_int_ultra[b][r][c] = F32::new(0.15);
            }
        }
    }
    for i in 0..160 {
        let sign = if (i & 1) == 0 { 0.25 } else { -0.20 };
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[i] =
            U16::new(f32_to_fp16(if linear_s2 { 0.0 } else { sign }));
    }
    a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[160..164]
        .copy_from_slice(&[f32_to_fp16(0.42); 4].map(U16::new));

    if !populate_cp1_inline || controls.is_empty() {
        return;
    }
    let dir_count = controls.len().min(3);
    cb.cp1_dir.cp1_dir_count = U16::new(dir_count as u16);
    if let (Ok(base_f32), Ok(ultra_f32)) = (
        <[F32]>::mut_from_bytes(&mut cb.cp1_base),
        <[F32]>::mut_from_bytes(&mut a3.control_bus_ultra_delta.cp1_ultra),
    ) {
        let mut b_cur = 0usize;
        let mut u_cur = 0usize;
        for (idx, ctrl) in controls.iter().take(dir_count).enumerate() {
            let is_sw = ctrl.kind == ControlKind::Switch;
            let lanes = if is_sw { 1usize } else { 2usize };
            let kl = ((u8::from(is_sw)) << 4) | (lanes as u8);
            cb.cp1_dir.entries[idx] = [idx as u8, kl, if is_sw { 0 } else { 0xFF }];
            for l in 0..lanes {
                for d in 0..8 {
                    let v = 0.08 * (((idx * 5 + l * 3 + d + 1) % 9) as f32 - 4.0);
                    base_f32[b_cur + l * 8 + d] = F32::new(v);
                    ultra_f32[u_cur + l * 8 + d] = F32::new(0.5 * v);
                }
                base_f32[b_cur + 8 * lanes + l] = F32::new(0.25);
            }
            for d in 0..4 {
                base_f32[b_cur + 9 * lanes + d] = F32::new(0.15 + 0.03 * (d as f32));
                ultra_f32[u_cur + 8 * lanes + d] = F32::new(0.12);
            }
            b_cur += 9 * lanes + 4;
            u_cur += 8 * lanes + 4;
        }
    }
}

pub fn build_sona_wires(
    controls: &[ControlSchemaDef],
    max_tier: Tier,
    linear_s2: bool,
    mute_s4_taps: bool,
    populate_cp1_inline: bool,
) -> (
    Arena0LiteWire,
    Arena1FullDeltaWire,
    Arena2ProDeltaWire,
    Arena3UltraDeltaWire,
) {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    let tier_mask = (1_u16 << (max_tier as usize + 1)) - 1;
    let meta_flags = SONA_META_OS2X
        | SONA_META_TIER_OS_EXPLICIT
        | SONA_META_OS_EXT_PRESENT
        | SONA_META_CORE_MODS_PRESENT
        | SONA_META_TRANSPORT_TAP_GAINS
        | SONA_META_SENSORY_QUAD
        | SONA_META_TRANSPORT_LFO
        | 0b1110;
    a0.crossbar.sense_svf_g = [0.22; 4].map(F32::new);
    a0.crossbar.sense_lru_alpha_rel = [0.90; 4].map(F32::new);
    a0.crossbar.sense_lru_alpha_atk = [0.05; 4].map(F32::new);
    a0.crossbar.sense_svf_k_fp16 = [f32_to_fp16(core::f32::consts::FRAC_1_SQRT_2); 4].map(U16::new);
    a0.crossbar.sense_lru_kappa_fp16 = [f32_to_fp16(0.25); 4].map(U16::new);
    a0.crossbar.sense_in_l_fp16 = [f32_to_fp16(0.5); 4].map(U16::new);
    a0.crossbar.sense_in_r_fp16 = [f32_to_fp16(-0.35); 4].map(U16::new);
    a0.crossbar.sense_tone_lp_fp16 = [f32_to_fp16(0.4); 4].map(U16::new);
    a0.crossbar.sense_tone_bp_fp16 = [f32_to_fp16(0.15); 4].map(U16::new);
    a0.crossbar.sense_tone_hp_fp16 = [f32_to_fp16(0.25); 4].map(U16::new);
    a0.crossbar.sense_vca_w_fp16 = [f32_to_fp16(0.3); 4].map(U16::new);
    a0.crossbar.crossbar_op_gains_fp16 = [f32_to_fp16(0.15), f32_to_fp16(0.10)].map(U16::new);
    a0.crossbar.tier_gain_trim = [1.0, 0.99, 0.98, 0.97].map(F32::new);
    a0.crossbar.pipeline_flags = U32::new(PIPE_CANONICAL_V3_SONA);
    a0.crossbar.sona_meta_flags = U32::new(meta_flags);
    a0.crossbar.phasor_freq_hz = [0.6, 1.3, 0.25, 0.0].map(F32::new);
    a0.crossbar.phasor_init_phase_q32 = [0x1000_0000, 0x2000_0000, 0, 0].map(U32::new);
    a0.crossbar.cell_desc.magic = SONA_CELL_MAGIC;
    a0.crossbar.cell_desc.version_major = U16::new(DNBN_VERSION_MAJOR);
    a0.crossbar.cell_desc.tier_flags = U16::new(tier_mask);
    a0.crossbar.cell_desc.oversample_factor = U16::new(2);
    a0.crossbar.cell_desc.cell_flags = U32::new(meta_flags);

    for (g, quad) in a0.core_quads.iter_mut().enumerate() {
        fill_core_quad(quad, g, linear_s2);
    }
    for tile in &mut a0.tiles {
        fill_tile(tile, linear_s2);
    }
    for t in 0..32 {
        a0.conv_fir_lite.fir_l[t] = F32::new(0.03 * (((t) % 5) as f32 - 2.0));
        a0.conv_fir_lite.fir_r[t] = F32::new(0.025 * (((t + 1) % 5) as f32 - 2.0));
    }
    let mut blk_global = 0usize;
    for blk in &mut a0.conv_kautz_lite {
        fill_kautz_block(blk, blk_global);
        blk_global += 1;
    }
    fill_transport_quad(&mut a0.transport_lite, 0, mute_s4_taps);
    a0.linear_readout_lite.pre_conv_u_gain = [F32::new(0.15), F32::new(-0.12)];
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.10), F32::new(0.08)];
    for k in 0..16 {
        a0.linear_readout_lite.stage_latent_mods_fp16[80 + k] = U16::new(f32_to_fp16(0.08));
    }
    fill_ro_odd_words(&mut a0.linear_readout_lite.w_ro_fp16[..192], 16, 0);
    for k in 0..16 {
        a0.linear_readout_lite.w_ro_fp16[192 + k] = U16::new(f32_to_fp16(0.05));
    }

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    for (i, quad) in a1.core_quads.iter_mut().enumerate() {
        fill_core_quad(quad, 4 + i, linear_s2);
    }
    for tile in &mut a1.tiles {
        fill_tile(tile, linear_s2);
    }
    fill_crossbar_block(&mut a1.crossbar_full, 6, 2.0);
    for t in 0..16 {
        a1.conv_fir_full.fir_l[t] = F32::new(0.02 * (((t + 1) % 5) as f32 - 2.0));
        a1.conv_fir_full.fir_r[t] = F32::new(0.02 * (((t + 2) % 5) as f32 - 2.0));
    }
    fill_kautz_block(&mut a1.conv_kautz_full, blk_global);
    blk_global += 1;
    fill_transport_quad(&mut a1.transport_full, 1, mute_s4_taps);
    fill_ro_odd_words(&mut a1.linear_readout_full.w_ro_and_mods_fp16[..288], 24, 1);

    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    for (i, quad) in a2.core_quads.iter_mut().enumerate() {
        fill_core_quad(quad, 6 + i, linear_s2);
    }
    for tile in &mut a2.tiles {
        fill_tile(tile, linear_s2);
    }
    fill_crossbar_block(&mut a2.crossbar_pro, 8, 3.0);
    for t in 0..48 {
        a2.conv_fir_pro.fir_l[t] = F32::new(0.015 * (((t + 2) % 5) as f32 - 2.0));
        a2.conv_fir_pro.fir_r[t] = F32::new(0.015 * (((t + 3) % 5) as f32 - 2.0));
    }
    for blk in &mut a2.conv_kautz_pro {
        fill_kautz_block(blk, blk_global);
        blk_global += 1;
    }
    fill_transport_quad(&mut a2.transport_pro[0], 2, mute_s4_taps);
    fill_ro_odd_words(&mut a2.linear_readout_pro.w_ro_and_mods_fp16[..384], 32, 2);

    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    for (i, quad) in a3.core_quads.iter_mut().enumerate() {
        fill_core_quad(quad, 8 + i, linear_s2);
    }
    for tile in &mut a3.tiles {
        fill_tile(tile, linear_s2);
    }
    fill_crossbar_block(&mut a3.crossbar_ultra, 10, 4.0);
    for k in 0..8 {
        a3.stage_latent_core_mods_fp16[k * 128] = U16::new(f32_to_fp16(0.25));
    }
    for t in 0..32 {
        a3.conv_fir_ultra.fir_l[t] = F32::new(0.012 * (((t + 3) % 5) as f32 - 2.0));
        a3.conv_fir_ultra.fir_r[t] = F32::new(0.012 * (((t + 4) % 5) as f32 - 2.0));
    }
    for blk in &mut a3.conv_kautz_ultra {
        fill_kautz_block(blk, blk_global);
        blk_global += 1;
    }
    let _ = (blk_global, CONVOLVER_KAUTZ_BLOCKS_PER_TIER);
    fill_transport_quad(&mut a3.transport_ultra[0], 3, mute_s4_taps);
    fill_ro_odd_words(
        &mut a3.linear_readout_ultra.w_ro_and_mods_fp16[..480],
        40,
        3,
    );

    populate_inline_control_bus(&mut a0, &mut a3, controls, populate_cp1_inline, linear_s2);
    (a0, a1, a2, a3)
}

pub fn assemble_arena_bytes(
    a0: &Arena0LiteWire,
    a1: &Arena1FullDeltaWire,
    a2: &Arena2ProDeltaWire,
    a3: &Arena3UltraDeltaWire,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(ARENA_TIER_BYTES[3]);
    out.extend_from_slice(a0.as_bytes());
    out.extend_from_slice(a1.as_bytes());
    out.extend_from_slice(a2.as_bytes());
    out.extend_from_slice(a3.as_bytes());
    out
}

pub fn build_cext_dispatch() -> CextDispatchWire {
    let mut cext = CextDispatchWire::new_zeroed();
    cext.w_ctrl_ext[0][0] = F32::new(0.45);
    cext.w_ctrl_ext[0][4] = F32::new(0.20);
    cext.w_ctrl_ext[14][0] = F32::new(-0.35);
    cext
}

pub fn header_dnbn_flags(header: &DnamJsonHeader) -> u16 {
    let mut flags = 0u16;
    if header.io_config.inputs == 1 {
        flags |= DNBN_FLAG_SINGLE_INPUT;
    }
    if header.io_config.output_mode != "mono" {
        flags |= DNBN_FLAG_STEREO_OUT;
    }
    if header.io_config.capabilities.wet_polarity_inverted {
        flags |= DNBN_FLAG_WET_POLARITY_INV;
    }
    if header.io_config.leveling.auto_unity_enabled {
        flags |= DNBN_FLAG_AUTO_UNITY;
    }
    flags
}

pub fn build_tsyn_entries(header: &DnamJsonHeader) -> Vec<TempoSyncEntryWire> {
    let mut entries = Vec::new();
    for (idx, ctrl) in header.controls_schema.iter().enumerate() {
        let Some(ref cal) = ctrl.calibration else {
            continue;
        };
        let Some(mode) = (match cal.role {
            ControlCalibrationRole::DelayTimeMs | ControlCalibrationRole::GrainSizeMs => Some(0u8),
            ControlCalibrationRole::LfoRateHz => Some(1u8),
            _ => None,
        }) else {
            continue;
        };
        let anchors = cal.lobatto_physical_anchors.unwrap_or([20.0, 120.0, 500.0]);
        let declared = ctrl.declared_range.unwrap_or([anchors[0], anchors[2]]);
        let subdiv = cal
            .tempo_sync_default_subdivision
            .unwrap_or(TempoSubdivision::Quarter);
        entries.push(TempoSyncEntryWire {
            ctrl_idx: U16::new(idx as u16),
            switch_ctrl_idx: U16::new(0),
            switch_opt_idx: 0,
            has_switch_cond: 0,
            mode,
            default_subdiv: subdiv as u8,
            flags: U32::new(0),
            div_ratio: F32::new(1.0),
            bpm_factor: F32::new(1.0),
            lobatto_anchors: anchors.map(F32::new),
            declared_min: F32::new(declared[0]),
            declared_max: F32::new(declared[1]),
        });
    }
    entries
}

pub fn build_dnam_bin_from_header_wires(
    header: &DnamJsonHeader,
    max_tier: Tier,
    arenas: (
        &Arena0LiteWire,
        Option<&Arena1FullDeltaWire>,
        Option<&Arena2ProDeltaWire>,
        Option<&Arena3UltraDeltaWire>,
    ),
    cext: Option<&CextDispatchWire>,
    cp1_overflow: &[Cp1ContributorSlot],
) -> Vec<u8> {
    let (a0, a1, a2, a3) = arenas;
    let descriptors = header
        .compile_clamp_descriptors()
        .expect("valid clamp descriptors");
    let wire_ctrls: Vec<_> = descriptors.iter().map(|d| d.to_wire()).collect();
    let cp1_bytes = if cp1_overflow.is_empty() {
        Vec::new()
    } else {
        encode_cp1_shared_stream(cp1_overflow).expect("valid cp1 overflow")
    };
    let tsyn = build_tsyn_entries(header);
    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: header.sample_rate_train_hz,
        header_flags: header_dnbn_flags(header),
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: header.transport.dc_blocker_pole,
        tail_time_ms: header.io_config.tail_time_ms,
        ro_pole_48k: header.linear_readout.ro_pole_48k,
        switch_xfade_samples: header.transport.switch_xfade_samples,
        content_hash_prefix: [0; 8],
        controls: &wire_ctrls,
        cp1_overflow: &cp1_bytes,
        tsyn: &tsyn,
        cext,
        arena0: a0,
        arena1: (max_tier >= Tier::Full).then_some(a1).flatten(),
        arena2: (max_tier >= Tier::Pro).then_some(a2).flatten(),
        arena3: (max_tier >= Tier::Ultra).then_some(a3).flatten(),
    })
    .expect("valid .dnam-bin")
}

pub fn build_sona_container_from_header_options(
    header: &mut DnamJsonHeader,
    max_tier: Tier,
    linear_s2: bool,
    mute_s4_taps: bool,
    include_cext: bool,
    cp1_overflow: &[Cp1ContributorSlot],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (a0, a1, a2, a3) = build_sona_wires(
        &header.controls_schema,
        max_tier,
        linear_s2,
        mute_s4_taps,
        cp1_overflow.is_empty(),
    );
    let full_arena = assemble_arena_bytes(&a0, &a1, &a2, &a3);
    let cext = include_cext.then(build_cext_dispatch);
    let dbin = build_dnam_bin_from_header_wires(
        header,
        max_tier,
        (&a0, Some(&a1), Some(&a2), Some(&a3)),
        cext.as_ref(),
        cp1_overflow,
    );
    let container = build_dnam_container_from_dnam_bin(header, &dbin).expect("valid hybrid .dnam");
    (
        full_arena[..max_tier.expected_arena_bytes()].to_vec(),
        dbin,
        container,
    )
}

pub fn build_sona_extended_container(linear_s2: bool, mute_s4_taps: bool) -> Vec<u8> {
    let mut header = build_sona_header();
    build_sona_container_from_header_options(
        &mut header,
        Tier::Ultra,
        linear_s2,
        mute_s4_taps,
        true,
        &[],
    )
    .2
}

pub fn build_populated_sona_container() -> (DnamJsonHeader, Vec<u8>, Vec<u8>) {
    let mut header = build_sona_header();
    let (arena, _dbin, container) =
        build_sona_container_from_header_options(&mut header, Tier::Ultra, false, false, true, &[]);
    (header, arena, container)
}

pub fn build_hybrid_container_for_arena(
    header: &mut DnamJsonHeader,
    template_dbin: &[u8],
    plaintext_arena: &[u8],
    stored_arena: &[u8],
    encrypted_arena: bool,
) -> Vec<u8> {
    let (hdr_wire, _) = DnamBinHeaderWire::read_from_prefix(template_dbin).expect("DNBN header");
    let arena0_off = hdr_wire.arena0_offset.get() as usize;
    let mut dbin_prefix = template_dbin[..arena0_off].to_vec();
    let mut patched_hdr = hdr_wire;
    let retained_tier = ARENA_TIER_BYTES
        .iter()
        .position(|&b| b == plaintext_arena.len())
        .expect("canonical arena len");
    let orig_max = hdr_wire.max_present_tier as usize;
    if orig_max <= retained_tier {
        patched_hdr.max_present_tier = retained_tier as u8;
        let expected_mask = (1_u16 << (retained_tier + 1)) - 1;
        patched_hdr.flags = U16::new((patched_hdr.flags.get() & !0x000F) | expected_mask);
    }
    let loop_max = (patched_hdr.max_present_tier as usize).max(retained_tier);
    for t in 0..4 {
        if t <= retained_tier {
            let end = ARENA_TIER_BYTES[t];
            patched_hdr.tier_end_offsets[t] = U32::new((arena0_off + end) as u32);
            patched_hdr.tier_arena_crc32c[t] = U32::new(crc32c_update(0, &plaintext_arena[..end]));
        } else if t <= loop_max {
            let end = ARENA_TIER_BYTES[t];
            patched_hdr.tier_end_offsets[t] = U32::new((arena0_off + end) as u32);
        } else {
            patched_hdr.tier_end_offsets[t] = U32::new(0);
            patched_hdr.tier_arena_crc32c[t] = U32::new(0);
        }
    }
    dbin_prefix[0..64].copy_from_slice(patched_hdr.as_bytes());
    let prefix_crc = crc32c_update(crc32c_update(0, &dbin_prefix[0..20]), &dbin_prefix[24..]);
    patched_hdr.prefix_crc32c = U32::new(prefix_crc);
    dbin_prefix[0..64].copy_from_slice(patched_hdr.as_bytes());

    let container = build_dnam_container_with_dbin(
        header,
        plaintext_arena,
        stored_arena,
        &dbin_prefix,
        encrypted_arena,
    )
    .expect("valid hybrid .dnam for modified arena");
    let preamble = DnamPreamble::from_bytes(&container).expect("valid preamble");
    assert!(preamble.dbin_prefix_bytes > 0);
    container
}

#[cfg(feature = "crypto")]
pub fn build_creator_dau1_container_from_dnam_bin(
    header: &DnamJsonHeader,
    dbin: &[u8],
    creator_seed: &[u8; 32],
    with_dmu1_presets: bool,
) -> (Vec<u8>, Vec<[u8; 32]>, [u8; 64]) {
    use dnam_engine::container::{
        DEFAULT_DMU1_TRAILER_CAPACITY, V3_HEADER_OFFSET, V3_PREAMBLE_SIZE, V3ContainerBuildSpec,
        build_dmu1_mutable_frame, build_v3_dnam_container_with_dbin,
        validate_v3_dnam_layout_with_max_profile,
    };
    use dnam_engine::provenance::{
        Dau1Envelope, RustCryptoProvider, build_creator_manifest_v1,
        build_plaintext_shared_v1_with_dbin, compute_plaintext_prefixes_v1, sha256_digest,
    };
    use dnam_vm::bin::DnamBinView;

    let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(creator_seed);
    let view = DnamBinView::parse(dbin).expect("valid dnam-bin");
    let arena0_off = view.header.arena0_offset.get() as usize;
    let dbin_prefix = &dbin[..arena0_off];
    let arena_all = &dbin[arena0_off..];
    let max_t = view.retained_tier as usize;
    let mut prev = 0usize;
    let mut tier_slices = Vec::with_capacity(max_t + 1);
    for &end in &ARENA_TIER_BYTES[..=max_t] {
        tier_slices.push(&arena_all[prev..end]);
        prev = end;
    }

    let mut hdr_val = serde_json::to_value(header).expect("serialize header");
    if let Some(root) = hdr_val.as_object_mut() {
        root.remove("presets");
        root.remove("user_metadata");
        root.remove("stem_user_metadata");
    }
    let hdr_json = DnamJsonHeader::validate_and_normalize_json_str(&hdr_val.to_string())
        .expect("normalized JCS header");
    let hdr_bytes = hdr_json.into_bytes();

    let mutable_frame = if with_dmu1_presets && !header.presets.is_empty() {
        let mut dmu = serde_json::Map::new();
        dmu.insert(
            "presets".to_string(),
            serde_json::to_value(&header.presets).expect("presets json"),
        );
        let dmu_bytes = serde_json::to_vec(&serde_json::Value::Object(dmu)).expect("dmu bytes");
        Some(build_dmu1_mutable_frame(&dmu_bytes, DEFAULT_DMU1_TRAILER_CAPACITY).expect("dmu1"))
    } else {
        None
    };

    let dummy_dau = Dau1Envelope {
        auth_kind: 0,
        creator_pubkey: creator_pk,
        creator_max_tier: max_t as u8,
        master_prefixes: vec![[0u8; 32]; max_t + 1],
        creator_signature: [0u8; 64],
        marketplace_pubkey: None,
        policy_bytes: Vec::new(),
        buyer_max_tier: None,
        buyer_prefixes: Vec::new(),
        marketplace_signature: None,
    };
    let dummy_auth = dummy_dau.to_bytes().expect("dummy dau");
    let mut out = build_v3_dnam_container_with_dbin(&V3ContainerBuildSpec {
        protection_profile: 0,
        header_bytes: &hdr_bytes,
        shared_bytes: &[],
        auth_bytes: &dummy_auth,
        dbin_prefix_bytes: dbin_prefix,
        tier_slices: &tier_slices,
        mutable_frame_bytes: mutable_frame.as_deref(),
        auth_kind: 0,
    })
    .expect("build creator dau1 container");

    let layout = validate_v3_dnam_layout_with_max_profile(&out, 1).expect("valid dau1 layout");
    let shared_master = build_plaintext_shared_v1_with_dbin(
        &out[..V3_PREAMBLE_SIZE],
        &out[V3_PREAMBLE_SIZE..V3_HEADER_OFFSET],
        &[],
        &out[layout.dbin_extent.0..layout.dbin_extent.0 + layout.dbin_extent.1],
    );
    let hdr_sha = sha256_digest(&hdr_bytes);
    let l_master = compute_plaintext_prefixes_v1(
        b"DNAM-MASTER-PREFIX-V1",
        &hdr_sha,
        &shared_master,
        &tier_slices,
    )
    .expect("master prefixes");
    let manifest = build_creator_manifest_v1(&creator_pk, &hdr_sha, &l_master).expect("manifest");
    let sig = RustCryptoProvider::ed25519_sign(creator_seed, &manifest);
    let signed_auth = Dau1Envelope {
        master_prefixes: l_master.clone(),
        creator_signature: sig,
        ..dummy_dau
    }
    .to_bytes()
    .expect("signed dau1");
    out[layout.auth_extent.0..layout.auth_extent.0 + signed_auth.len()]
        .copy_from_slice(&signed_auth);
    (out, l_master, sig)
}
