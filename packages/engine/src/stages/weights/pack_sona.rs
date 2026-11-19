//! Declarative `zerocopy` `SONA` `v3.0` Matryoshka arena packer & unpacker
//! (`stages/weights/pack_sona.rs`, `#![forbid(unsafe_code)]`, zero manual endian slicing).

mod readout_ctrl;

use dnam_kernels::{FULL_SPEC, LITE_SPEC, PRO_SPEC, ULTRA_SPEC, f32_to_fp16, fp16_to_f32};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, FromZeros, IntoBytes};

use crate::stages::router::V3Stage0Params;
use crate::stages::sona_core::frontend::SonaOversamplerParams;
use crate::stages::sona_core::model::{
    MAX_SONA_GROUPS, SonaColdTierModel, SonaCrossbarConfig, SonaGroupCold, SonaMatryoshkaModel,
    SonaTileCold,
};
use crate::stages::weights::{
    ANTI_ZIPPER_MICRO_RAMP_SCALE_DEFAULT, ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES,
    ARENA_ULTRA_BYTES, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3UltraDeltaWire, PIPE_FLAG_SONA, PIPE_FLAG_STAGE3_EXT, PIPE_FLAG_STAGE4_EXT,
    PIPE_FLAG_UNIFIED_W_CTRL, PIPE_FLAG_V3_MATRYOSHKA, PIPE_MAGIC_HI, SONA_CELL_MAGIC,
    SONA_META_OS_EXT_PRESENT, SONA_META_TIER_OS_EXPLICIT, SonaCellDescWire, SonaCrossbarBlockWire,
    SonaGroupWire, SonaInterBlockTilePairWire, SonaMetaWire, SonaOsAndExtWire,
    SonaPhasorsAndDescWire, Stage0CrossbarWire, V3_META_CORE_MODS_PRESENT, V3_META_OS2X,
    V3_META_STAGE0_EXT, V3_META_STAGE3_PREFIX_OVERRIDES, V3_META_STAGE4_LFO,
    V3_META_STAGE4_TAP_GAINS,
};
use crate::types::Tier;
use readout_ctrl::{
    pack_arena0_ctrl_wire, pack_arena3_ctrl_wire, pack_kautz_blocks, pack_readout_full,
    pack_readout_lite, pack_readout_pro, pack_readout_ultra, pack_stage4_quad_wire,
    rebuild_tier_stage4, tier_pack_flags, unpack_arena0_ctrl_wire, unpack_arena3_ctrl_wire,
    unpack_kautz_blocks, unpack_readout_full, unpack_readout_lite, unpack_readout_pro,
    unpack_readout_ultra, unpack_stage4_quad_wire,
};

/// Writes a little-endian `f32` into `buf[offset..]` via safe `zerocopy` (no-op if out of bounds).
#[inline]
pub fn write_f32_le(buf: &mut [u8], offset: usize, value: f32) {
    if let Some(dst) = buf.get_mut(offset..) {
        let _ = F32::new(value).write_to_prefix(dst);
    }
}

/// Writes a little-endian `u32` into `buf[offset..]` via safe `zerocopy` (no-op if out of bounds).
#[inline]
pub fn write_u32_le(buf: &mut [u8], offset: usize, value: u32) {
    if let Some(dst) = buf.get_mut(offset..) {
        let _ = U32::new(value).write_to_prefix(dst);
    }
}

fn encode_group_wire(g: &SonaGroupCold) -> SonaGroupWire {
    let mut c = *g;
    c.clamp_in_place();
    SonaGroupWire {
        svf_g: c.svf_g.map(F32::new),
        svf_k: c.svf_k.map(F32::new),
        lru_alpha_rel: c.lru_alpha_rel.map(F32::new),
        lru_alpha_atk: c.lru_alpha_atk.map(F32::new),
        lru_kappa: c.lru_kappa.map(F32::new),
        adaa_drive: c.adaa_drive.map(F32::new),
        lru_bias_drive: c.lru_bias_drive.map(F32::new),
        in_x_weight: c.in_x_weight.map(F32::new),
        tone_lp_fp16: c.tone_lp.map(|v| U16::new(f32_to_fp16(v))),
        tone_bp_fp16: c.tone_bp.map(|v| U16::new(f32_to_fp16(v))),
        tone_hp_fp16: c.tone_hp.map(|v| U16::new(f32_to_fp16(v))),
        tone_ap_fp16: c.tone_ap.map(|v| U16::new(f32_to_fp16(v))),
        tone_dir_fp16: c.tone_dir.map(|v| U16::new(f32_to_fp16(v))),
        sag_depth_fp16: c.sag_depth.map(|v| U16::new(f32_to_fp16(v))),
        wavefolder_mix_fp16: c.wavefolder_mix.map(|v| U16::new(f32_to_fp16(v))),
        rect_sign_fp16: c.rect_sign.map(|v| U16::new(f32_to_fp16(v))),
    }
}

fn decode_group_wire(w: &SonaGroupWire) -> SonaGroupCold {
    let mut g = SonaGroupCold {
        svf_g: w.svf_g.map(F32::get),
        svf_k: w.svf_k.map(F32::get),
        lru_alpha_rel: w.lru_alpha_rel.map(F32::get),
        lru_alpha_atk: w.lru_alpha_atk.map(F32::get),
        lru_kappa: w.lru_kappa.map(F32::get),
        adaa_drive: w.adaa_drive.map(F32::get),
        lru_bias_drive: w.lru_bias_drive.map(F32::get),
        in_x_weight: w.in_x_weight.map(F32::get),
        tone_lp: w.tone_lp_fp16.map(|u| fp16_to_f32(u.get())),
        tone_bp: w.tone_bp_fp16.map(|u| fp16_to_f32(u.get())),
        tone_hp: w.tone_hp_fp16.map(|u| fp16_to_f32(u.get())),
        tone_ap: w.tone_ap_fp16.map(|u| fp16_to_f32(u.get())),
        tone_dir: w.tone_dir_fp16.map(|u| fp16_to_f32(u.get())),
        sag_depth: w.sag_depth_fp16.map(|u| fp16_to_f32(u.get())),
        wavefolder_mix: w.wavefolder_mix_fp16.map(|u| fp16_to_f32(u.get())),
        rect_sign: w.rect_sign_fp16.map(|u| fp16_to_f32(u.get())),
    };
    g.clamp_in_place();
    g
}

fn encode_tile_wire(t: &SonaTileCold) -> SonaInterBlockTilePairWire {
    let mut c = *t;
    c.clamp_in_place();
    SonaInterBlockTilePairWire {
        w_odd_4x4: core::array::from_fn(|r| {
            core::array::from_fn(|col| F32::new(c.w_odd_4x4[r][col]))
        }),
        w_even_4x4_fp16: core::array::from_fn(|r| {
            core::array::from_fn(|col| U16::new(f32_to_fp16(c.w_even_4x4[r][col])))
        }),
        w_lru_4x4_fp16: core::array::from_fn(|r| {
            core::array::from_fn(|col| U16::new(f32_to_fp16(c.w_lru_4x4[r][col])))
        }),
    }
}

fn decode_tile_wire(w: &SonaInterBlockTilePairWire) -> SonaTileCold {
    let mut t = SonaTileCold {
        w_odd_4x4: core::array::from_fn(|r| core::array::from_fn(|col| w.w_odd_4x4[r][col].get())),
        w_even_4x4: core::array::from_fn(|r| {
            core::array::from_fn(|col| fp16_to_f32(w.w_even_4x4_fp16[r][col].get()))
        }),
        w_lru_4x4: core::array::from_fn(|r| {
            core::array::from_fn(|col| fp16_to_f32(w.w_lru_4x4_fp16[r][col].get()))
        }),
    };
    t.clamp_in_place();
    t
}

fn encode_crossbar_block(cb: &SonaCrossbarConfig) -> SonaCrossbarBlockWire {
    let mut c = *cb;
    c.clamp_in_place();
    SonaCrossbarBlockWire {
        op_p_to_group_fp16: core::array::from_fn(|g| {
            core::array::from_fn(|l| U16::new(f32_to_fp16(c.op_p_to_group[g][l])))
        }),
        phasor_to_group_fp16: core::array::from_fn(|g| {
            core::array::from_fn(|l| U16::new(f32_to_fp16(c.phasor_to_group[g][l])))
        }),
        opt_to_lr_fp16: core::array::from_fn(|i| {
            core::array::from_fn(|ch| U16::new(f32_to_fp16(c.opt_to_lr[i][ch])))
        }),
        phasor_stereo_phase_fp16: core::array::from_fn(|i| {
            U16::new(if i < 4 {
                f32_to_fp16(c.phasor_stereo_phase_rad[i])
            } else {
                0
            })
        }),
        crossbar_reserved_fp16: [U16::new(0); 8],
    }
}

fn decode_crossbar_block(w: &SonaCrossbarBlockWire, cb: &mut SonaCrossbarConfig) {
    for g in 0..MAX_SONA_GROUPS {
        for l in 0..4 {
            cb.op_p_to_group[g][l] = fp16_to_f32(w.op_p_to_group_fp16[g][l].get());
            cb.phasor_to_group[g][l] = fp16_to_f32(w.phasor_to_group_fp16[g][l].get());
        }
    }
    for i in 0..16 {
        cb.opt_to_lr[i][0] = fp16_to_f32(w.opt_to_lr_fp16[i][0].get());
        cb.opt_to_lr[i][1] = fp16_to_f32(w.opt_to_lr_fp16[i][1].get());
    }
    for i in 0..4 {
        cb.phasor_stereo_phase_rad[i] = fp16_to_f32(w.phasor_stereo_phase_fp16[i].get());
    }
}

fn has_stage3_prefix_overrides(model: &SonaMatryoshkaModel, arena_len: usize) -> bool {
    const TIER_CHECKS: [(usize, usize, usize, usize); 3] = [
        (1, ARENA_FULL_BYTES, 32, 8),
        (2, ARENA_PRO_BYTES, 48, 12),
        (3, ARENA_ULTRA_BYTES, 96, 24),
    ];
    for &(tier_idx, min_bytes, fir_prefix, kautz_prefix) in &TIER_CHECKS {
        if arena_len < min_bytes {
            continue;
        }
        let prev = &model.readouts[tier_idx - 1];
        let curr = &model.readouts[tier_idx];
        for i in 0..fir_prefix {
            if (curr.fir_l[i] - prev.fir_l[i]).abs() > 1.0e-7
                || (curr.fir_r[i] - prev.fir_r[i]).abs() > 1.0e-7
            {
                return true;
            }
        }
        for m in 0..kautz_prefix {
            if (curr.kautz_b_gamma[m] - prev.kautz_b_gamma[m]).abs() > 1.0e-7
                || (curr.kautz_c_l[m].0 - prev.kautz_c_l[m].0).abs() > 1.0e-7
                || (curr.kautz_c_l[m].1 - prev.kautz_c_l[m].1).abs() > 1.0e-7
                || (curr.kautz_c_r[m].0 - prev.kautz_c_r[m].0).abs() > 1.0e-7
                || (curr.kautz_c_r[m].1 - prev.kautz_c_r[m].1).abs() > 1.0e-7
            {
                return true;
            }
        }
    }
    false
}

fn encode_stage0_crossbar(model: &SonaMatryoshkaModel, arena_len: usize) -> Stage0CrossbarWire {
    let mut s0 = model.stage0;
    s0.clamp_in_place();
    let mut cb = *model
        .tier_crossbars
        .as_ref()
        .map_or(&model.crossbar, |cbs| &cbs[0]);
    cb.clamp_in_place();
    let mut os = model.os_params;
    os.clamp_in_place();

    let mut meta_flags = SONA_META_TIER_OS_EXPLICIT
        | SONA_META_OS_EXT_PRESENT
        | V3_META_STAGE4_TAP_GAINS
        | V3_META_STAGE0_EXT
        | V3_META_STAGE4_LFO;
    if has_stage3_prefix_overrides(model, arena_len) {
        meta_flags |= V3_META_STAGE3_PREFIX_OVERRIDES;
    }
    if model.os2x {
        meta_flags |= V3_META_OS2X;
    }
    for t in 0..4 {
        if model.tier_os2x[t] {
            meta_flags |= 1 << t;
        }
    }
    if model.ctrl.core_mods_present
        || model
            .ctrl
            .stage_latent_core_mods_fp16
            .iter()
            .any(|&u| u != 0)
    {
        meta_flags |= V3_META_CORE_MODS_PRESENT;
    }

    let pipe_flags = PIPE_MAGIC_HI
        | PIPE_FLAG_SONA
        | PIPE_FLAG_V3_MATRYOSHKA
        | PIPE_FLAG_UNIFIED_W_CTRL
        | PIPE_FLAG_STAGE3_EXT
        | PIPE_FLAG_STAGE4_EXT;

    let mut dry_wet = 1.0_f32;
    let q0 = model.stage4[0].effective_quad_params(0);
    if q0.wet_mix.abs() > 1.0e-7 || q0.dry_mix != 1.0 {
        dry_wet = q0.dry_mix;
    }

    Stage0CrossbarWire {
        s_router: core::array::from_fn(|r| core::array::from_fn(|c| F32::new(s0.s_router[r][c]))),
        meta: SonaMetaWire {
            pipeline_flags: U32::new(pipe_flags),
            tier_gain_trim: [F32::new(1.0); 4],
            v3_meta_flags: U32::new(meta_flags),
            dry_wet_mix: F32::new(dry_wet),
            r_ac: F32::new(q0.r_dc),
            stage0_gain: s0.gain.map(F32::new),
            stage0_mix_matrix: [
                [F32::new(s0.mix_matrix[0][0]), F32::new(s0.mix_matrix[0][1])],
                [F32::new(s0.mix_matrix[1][0]), F32::new(s0.mix_matrix[1][1])],
            ],
            stage0_xo_hz: F32::new(s0.xo_hz),
            anti_zipper_ramp_scale: F32::new(ANTI_ZIPPER_MICRO_RAMP_SCALE_DEFAULT),
        },
        os_ext: SonaOsAndExtWire {
            os_up_quad_coeff: F32::new(os.up_quad_coeff),
            os_dec_c_prev1: F32::new(os.dec_c_prev1),
            os_dec_c0: F32::new(os.dec_c0),
            os_up_pre_emph: F32::new(os.up_pre_emph),
            os_dec_aa_mix: F32::new(os.dec_aa_mix),
            os_up_aa_mix: F32::new(os.up_aa_mix),
            os_up_zoh_mix: F32::new(os.up_zoh_mix),
            stage0_xo_low_gain: F32::new(s0.xo_low_gain),
            stage0_xo_high_gain: F32::new(s0.xo_high_gain),
            stage0_clean_sub_mix: F32::new(s0.clean_sub_mix),
            stage0_op_p_sub_mix: F32::new(0.0),
            stage0_op_p_lpf_pole: F32::new(0.0),
            stage0_op_p_env_pole: F32::new(0.0),
            stage0_op_p_sub2_mix: F32::new(0.0),
            stage0_op_p_oct_mix: F32::new(0.0),
            stage0_op_p_env_mix: F32::new(0.0),
        },
        cell_phasors: SonaPhasorsAndDescWire {
            cell_desc: SonaCellDescWire {
                magic: SONA_CELL_MAGIC,
                version_major: U16::new(3),
                version_minor: U16::new(0),
                tier_flags: U16::new(tier_pack_flags(model, arena_len)),
                oversample_factor: U16::new(if model.os2x { 2 } else { 1 }),
                cell_flags: U32::new(meta_flags),
            },
            phasor_freq_hz: cb.phasor_freq_hz.map(F32::new),
            phasor_init_phase_q32: cb.phasor_init_phase_q32.map(U32::new),
            gate_and_crossbar_fp16: [
                U16::new(f32_to_fp16(s0.gate_thresh)),
                U16::new(f32_to_fp16(s0.gate_hyst)),
                U16::new(f32_to_fp16(s0.gate_inv_knee)),
                U16::new(f32_to_fp16(s0.gate_floor)),
                U16::new(f32_to_fp16(s0.gate_atk_beta)),
                U16::new(f32_to_fp16(s0.gate_rel_beta)),
                U16::new(f32_to_fp16(cb.op_p_crossbar_gain)),
                U16::new(f32_to_fp16(cb.op_t_crossbar_gain)),
            ],
        },
    }
}

/// Serializes a complete [`SonaMatryoshkaModel`] into a contiguous `64 B`-aligned wire arena (`4,608..30,720 B`).
pub fn pack_sona_matryoshka_to_arena(model: &SonaMatryoshkaModel, arena: &mut [u8]) {
    let arena_len = arena.len();
    let Ok((a0, _)) = Arena0LiteWire::mut_from_prefix(arena) else {
        return;
    };
    *a0 = Arena0LiteWire::new_zeroed();
    a0.stage0 = encode_stage0_crossbar(model, arena_len);

    for g in 0..4 {
        a0.sona_core.groups[g] = encode_group_wire(&model.groups[g]);
    }
    for t in 0..3 {
        a0.sona_core.tiles[t] = encode_tile_wire(&model.tiles[t]);
    }
    for i in 0..32 {
        a0.fir_lite.fir_l[i] = F32::new(model.readouts[0].fir_l[i]);
        a0.fir_lite.fir_r[i] = F32::new(model.readouts[0].fir_r[i]);
    }
    pack_kautz_blocks(
        &model.kautz_poles,
        &model.readouts[0],
        0,
        &mut a0.kautz_lite,
    );
    a0.stage4_lite = pack_stage4_quad_wire(&model.stage4[0].effective_quad_params(0));
    pack_readout_lite(&model.readouts[0], &model.ctrl, &mut a0.readout_lite);
    let mut cb0 = *model
        .tier_crossbars
        .as_ref()
        .map_or(&model.crossbar, |cbs| &cbs[0]);
    cb0.clamp_in_place();
    for g in 0..4 {
        for l in 0..4 {
            a0.readout_lite.stage_latent_mods_fp16[80 + g * 4 + l] =
                U16::new(f32_to_fp16(cb0.op_p_to_group[g][l]));
            a0.readout_lite.w_ro_fp16[192 + g * 4 + l] =
                U16::new(f32_to_fp16(cb0.phasor_to_group[g][l]));
        }
    }
    pack_arena0_ctrl_wire(&model.ctrl, &mut a0.ctrl_base);
    for l in 0..4 {
        a0.ctrl_base.rho_head_and_ctrl_tail[24 + l * 2] = F32::new(cb0.opt_to_lr[l][0]);
        a0.ctrl_base.rho_head_and_ctrl_tail[25 + l * 2] = F32::new(cb0.opt_to_lr[l][1]);
    }

    if arena_len >= ARENA_FULL_BYTES
        && let Ok((a1, _)) = Arena1FullDeltaWire::mut_from_prefix(&mut arena[ARENA_LITE_BYTES..])
    {
        *a1 = Arena1FullDeltaWire::new_zeroed();
        for i in 0..2 {
            a1.sona_core_full.groups[i] = encode_group_wire(&model.groups[4 + i]);
        }
        for i in 0..10 {
            a1.sona_core_full.tiles[i] = encode_tile_wire(&model.tiles[3 + i]);
        }
        let cb1 = model
            .tier_crossbars
            .as_ref()
            .map_or(&model.crossbar, |cbs| &cbs[1]);
        a1.sona_core_full.crossbar = encode_crossbar_block(cb1);
        for i in 0..16 {
            a1.fir_full.fir_l[i] = F32::new(model.readouts[1].fir_l[32 + i]);
            a1.fir_full.fir_r[i] = F32::new(model.readouts[1].fir_r[32 + i]);
        }
        pack_kautz_blocks(
            &model.kautz_poles,
            &model.readouts[1],
            8,
            &mut a1.kautz_full,
        );
        a1.stage4_full = pack_stage4_quad_wire(&model.stage4[1].effective_quad_params(1));
        pack_readout_full(&model.readouts[1], &mut a1.varpro_readout_full);
    }

    if arena_len >= ARENA_PRO_BYTES
        && let Ok((a2, _)) = Arena2ProDeltaWire::mut_from_prefix(&mut arena[ARENA_FULL_BYTES..])
    {
        *a2 = Arena2ProDeltaWire::new_zeroed();
        for i in 0..2 {
            a2.sona_core_pro.groups[i] = encode_group_wire(&model.groups[6 + i]);
        }
        for i in 0..10 {
            a2.sona_core_pro.tiles[i] = encode_tile_wire(&model.tiles[13 + i]);
        }
        let cb2 = model
            .tier_crossbars
            .as_ref()
            .map_or(&model.crossbar, |cbs| &cbs[2]);
        a2.sona_core_pro.crossbar_pro = encode_crossbar_block(cb2);
        for i in 0..48 {
            a2.fir_pro.fir_l[i] = F32::new(model.readouts[2].fir_l[48 + i]);
            a2.fir_pro.fir_r[i] = F32::new(model.readouts[2].fir_r[48 + i]);
        }
        pack_kautz_blocks(
            &model.kautz_poles,
            &model.readouts[2],
            12,
            &mut a2.kautz_pro,
        );
        a2.stage4_pro = [
            pack_stage4_quad_wire(&model.stage4[2].effective_quad_params(2)),
            FromZeros::new_zeroed(),
        ];
        pack_readout_pro(&model.readouts[2], &mut a2.varpro_readout_pro);
    }

    if arena_len >= ARENA_ULTRA_BYTES
        && let Ok((a3, _)) = Arena3UltraDeltaWire::mut_from_prefix(&mut arena[ARENA_PRO_BYTES..])
    {
        *a3 = Arena3UltraDeltaWire::new_zeroed();
        for i in 0..2 {
            a3.sona_core_ultra.groups[i] = encode_group_wire(&model.groups[8 + i]);
        }
        for i in 0..12 {
            a3.sona_core_ultra.tiles[i] = encode_tile_wire(&model.tiles[23 + i]);
        }
        let cb3 = model
            .tier_crossbars
            .as_ref()
            .map_or(&model.crossbar, |cbs| &cbs[3]);
        a3.sona_core_ultra.crossbar_ultra = encode_crossbar_block(cb3);
        a3.sona_core_ultra.stage_latent_core_mods_fp16 =
            model.ctrl.stage_latent_core_mods_fp16.map(U16::new);
        for i in 0..32 {
            a3.fir_ultra.fir_l[i] = F32::new(model.readouts[3].fir_l[96 + i]);
            a3.fir_ultra.fir_r[i] = F32::new(model.readouts[3].fir_r[96 + i]);
        }
        pack_kautz_blocks(
            &model.kautz_poles,
            &model.readouts[3],
            24,
            &mut a3.kautz_ultra,
        );
        a3.stage4_ultra = [
            pack_stage4_quad_wire(&model.stage4[3].effective_quad_params(3)),
            FromZeros::new_zeroed(),
        ];
        pack_readout_ultra(&model.readouts[3], &mut a3.varpro_readout_ultra);
        pack_arena3_ctrl_wire(&model.ctrl, &mut a3.ctrl_ultra);
    }
}

/// Compatibility alias for [`pack_sona_matryoshka_to_arena`].
#[inline]
pub fn pack_v3_matryoshka_to_arena(model: &SonaMatryoshkaModel, arena: &mut [u8]) {
    pack_sona_matryoshka_to_arena(model, arena);
}

fn decode_stage0_and_crossbar(w: &Stage0CrossbarWire, model: &mut SonaMatryoshkaModel) {
    let meta = &w.meta;
    let ext = &w.os_ext;
    let cp = &w.cell_phasors;
    let mf = meta.v3_meta_flags.get();

    model.os2x = (mf & V3_META_OS2X) != 0 || cp.cell_desc.oversample_factor.get() > 1;
    if (mf & SONA_META_TIER_OS_EXPLICIT) != 0 {
        model.tier_os2x = [
            (mf & (1 << 0)) != 0,
            (mf & (1 << 1)) != 0,
            (mf & (1 << 2)) != 0,
            (mf & (1 << 3)) != 0,
        ];
    } else {
        model.tier_os2x = [
            (mf & (1 << 0)) != 0 || LITE_SPEC.oversample_factor > 1,
            (mf & (1 << 1)) != 0 || FULL_SPEC.oversample_factor > 1,
            (mf & (1 << 2)) != 0 || PRO_SPEC.oversample_factor > 1,
            (mf & (1 << 3)) != 0 || ULTRA_SPEC.oversample_factor > 1,
        ];
    }

    if (mf & V3_META_STAGE0_EXT) != 0 || meta.pipeline_flags.get() != 0 {
        let mut s0 = V3Stage0Params {
            s_router: core::array::from_fn(|r| core::array::from_fn(|c| w.s_router[r][c].get())),
            gain: meta.stage0_gain.map(F32::get),
            mix_matrix: [
                [
                    meta.stage0_mix_matrix[0][0].get(),
                    meta.stage0_mix_matrix[0][1].get(),
                ],
                [
                    meta.stage0_mix_matrix[1][0].get(),
                    meta.stage0_mix_matrix[1][1].get(),
                ],
            ],
            xo_hz: meta.stage0_xo_hz.get(),
            xo_low_gain: ext.stage0_xo_low_gain.get(),
            xo_high_gain: ext.stage0_xo_high_gain.get(),
            clean_sub_mix: ext.stage0_clean_sub_mix.get(),
            gate_thresh: fp16_to_f32(cp.gate_and_crossbar_fp16[0].get()),
            gate_hyst: fp16_to_f32(cp.gate_and_crossbar_fp16[1].get()),
            gate_inv_knee: fp16_to_f32(cp.gate_and_crossbar_fp16[2].get()),
            gate_floor: fp16_to_f32(cp.gate_and_crossbar_fp16[3].get()),
            gate_atk_beta: fp16_to_f32(cp.gate_and_crossbar_fp16[4].get()),
            gate_rel_beta: fp16_to_f32(cp.gate_and_crossbar_fp16[5].get()),
        };
        if (mf & V3_META_STAGE0_EXT) == 0 && s0.xo_hz <= 0.0 {
            s0 = V3Stage0Params::identity();
        }
        s0.clamp_in_place();
        model.stage0 = s0;
    }

    let mut os = SonaOversamplerParams {
        up_quad_coeff: ext.os_up_quad_coeff.get(),
        dec_c_prev1: ext.os_dec_c_prev1.get(),
        dec_c0: ext.os_dec_c0.get(),
        dec_c1: 1.0 - ext.os_dec_c_prev1.get() - ext.os_dec_c0.get(),
        up_pre_emph: ext.os_up_pre_emph.get(),
        dec_aa_mix: ext.os_dec_aa_mix.get(),
        up_aa_mix: ext.os_up_aa_mix.get(),
        up_zoh_mix: ext.os_up_zoh_mix.get(),
    };
    if (mf & SONA_META_OS_EXT_PRESENT) != 0 {
        os.clamp_in_place();
    } else if os.up_quad_coeff.abs() < 1e-6 && os.dec_c0.abs() < 1e-6 {
        os = SonaOversamplerParams::default();
    } else {
        os.clamp_in_place();
    }
    model.os_params = os;

    let mut cb = SonaCrossbarConfig::identity();
    cb.phasor_freq_hz = cp.phasor_freq_hz.map(F32::get);
    cb.phasor_init_phase_q32 = cp.phasor_init_phase_q32.map(U32::get);
    cb.op_p_crossbar_gain = fp16_to_f32(cp.gate_and_crossbar_fp16[6].get());
    cb.op_t_crossbar_gain = fp16_to_f32(cp.gate_and_crossbar_fp16[7].get());
    cb.clamp_in_place();
    model.crossbar = cb;
}

/// Unpacks a [`SonaMatryoshkaModel`] from a valid `SONA` `v3.0` wire arena via safe `zerocopy`.
#[must_use]
pub fn unpack_sona_matryoshka_from_arena(arena: &[u8], mono_out: bool) -> SonaMatryoshkaModel {
    let mut model = SonaMatryoshkaModel::identity(false, mono_out);
    let Ok((a0, _)) = Arena0LiteWire::ref_from_prefix(arena) else {
        return model;
    };
    if a0.stage0.cell_phasors.cell_desc.magic != SONA_CELL_MAGIC {
        return model;
    }
    let mf = a0.stage0.meta.v3_meta_flags.get();
    decode_stage0_and_crossbar(&a0.stage0, &mut model);
    unpack_arena0_ctrl_wire(&a0.ctrl_base, &mut model.ctrl);
    for l in 0..4 {
        model.crossbar.opt_to_lr[l] = [
            a0.ctrl_base.rho_head_and_ctrl_tail[24 + l * 2].get(),
            a0.ctrl_base.rho_head_and_ctrl_tail[25 + l * 2].get(),
        ];
    }

    for g in 0..4 {
        model.groups[g] = decode_group_wire(&a0.sona_core.groups[g]);
    }
    for t in 0..3 {
        model.tiles[t] = decode_tile_wire(&a0.sona_core.tiles[t]);
    }
    for i in 0..32 {
        model.readouts[0].fir_l[i] = a0.fir_lite.fir_l[i].get();
        model.readouts[0].fir_r[i] = a0.fir_lite.fir_r[i].get();
    }
    unpack_kautz_blocks(
        &a0.kautz_lite,
        0,
        &mut model.kautz_poles,
        &mut model.readouts[0],
    );
    unpack_readout_lite(&a0.readout_lite, &mut model.readouts[0], &mut model.ctrl);
    for g in 0..4 {
        for l in 0..4 {
            model.crossbar.op_p_to_group[g][l] =
                fp16_to_f32(a0.readout_lite.stage_latent_mods_fp16[80 + g * 4 + l].get());
            model.crossbar.phasor_to_group[g][l] =
                fp16_to_f32(a0.readout_lite.w_ro_fp16[192 + g * 4 + l].get());
        }
    }
    let q0 = unpack_stage4_quad_wire(&a0.stage4_lite);
    model.stage4[0] = rebuild_tier_stage4(Tier::Lite, q0, None, None, None);
    model.crossbar.phasor_stereo_phase_rad = [0.0; 4];
    model.crossbar.clamp_in_place();
    let cb0 = model.crossbar;
    let mut cb1 = cb0;
    let mut cb2 = cb0;
    let mut cb3 = cb0;
    let mut highest_cb = cb0;

    if arena.len() >= ARENA_FULL_BYTES
        && let Ok((a1, _)) = Arena1FullDeltaWire::ref_from_prefix(&arena[ARENA_LITE_BYTES..])
    {
        for i in 0..2 {
            model.groups[4 + i] = decode_group_wire(&a1.sona_core_full.groups[i]);
        }
        for i in 0..10 {
            model.tiles[3 + i] = decode_tile_wire(&a1.sona_core_full.tiles[i]);
        }
        cb1 = cb0;
        decode_crossbar_block(&a1.sona_core_full.crossbar, &mut cb1);
        cb1.clamp_in_place();
        cb2 = cb1;
        cb3 = cb1;
        highest_cb = cb1;
        model.readouts[1] = model.readouts[0].clone();
        for i in 0..16 {
            model.readouts[1].fir_l[32 + i] = a1.fir_full.fir_l[i].get();
            model.readouts[1].fir_r[32 + i] = a1.fir_full.fir_r[i].get();
        }
        unpack_kautz_blocks(
            &a1.kautz_full,
            8,
            &mut model.kautz_poles,
            &mut model.readouts[1],
        );
        let apply_prefix_overrides = (mf & V3_META_STAGE3_PREFIX_OVERRIDES) != 0;
        unpack_readout_full(
            &a1.varpro_readout_full,
            &mut model.readouts[1],
            apply_prefix_overrides,
        );
        let q1 = unpack_stage4_quad_wire(&a1.stage4_full);
        model.stage4[1] = rebuild_tier_stage4(Tier::Full, q0, Some(q1), None, None);
    }

    if arena.len() >= ARENA_PRO_BYTES
        && let Ok((a2, _)) = Arena2ProDeltaWire::ref_from_prefix(&arena[ARENA_FULL_BYTES..])
    {
        for i in 0..2 {
            model.groups[6 + i] = decode_group_wire(&a2.sona_core_pro.groups[i]);
        }
        for i in 0..10 {
            model.tiles[13 + i] = decode_tile_wire(&a2.sona_core_pro.tiles[i]);
        }
        cb2 = cb1;
        decode_crossbar_block(&a2.sona_core_pro.crossbar_pro, &mut cb2);
        cb2.clamp_in_place();
        cb3 = cb2;
        highest_cb = cb2;
        model.readouts[2] = model.readouts[1].clone();
        for i in 0..48 {
            model.readouts[2].fir_l[48 + i] = a2.fir_pro.fir_l[i].get();
            model.readouts[2].fir_r[48 + i] = a2.fir_pro.fir_r[i].get();
        }
        unpack_kautz_blocks(
            &a2.kautz_pro,
            12,
            &mut model.kautz_poles,
            &mut model.readouts[2],
        );
        let apply_prefix_overrides = (mf & V3_META_STAGE3_PREFIX_OVERRIDES) != 0;
        unpack_readout_pro(
            &a2.varpro_readout_pro,
            &mut model.readouts[2],
            apply_prefix_overrides,
        );
        let q1 = model.stage4[1].effective_quad_params(1);
        let q2 = unpack_stage4_quad_wire(&a2.stage4_pro[0]);
        model.stage4[2] = rebuild_tier_stage4(Tier::Pro, q0, Some(q1), Some(q2), None);
    }

    if arena.len() >= ARENA_ULTRA_BYTES
        && let Ok((a3, _)) = Arena3UltraDeltaWire::ref_from_prefix(&arena[ARENA_PRO_BYTES..])
    {
        for i in 0..2 {
            model.groups[8 + i] = decode_group_wire(&a3.sona_core_ultra.groups[i]);
        }
        for i in 0..12 {
            model.tiles[23 + i] = decode_tile_wire(&a3.sona_core_ultra.tiles[i]);
        }
        cb3 = cb2;
        decode_crossbar_block(&a3.sona_core_ultra.crossbar_ultra, &mut cb3);
        cb3.clamp_in_place();
        highest_cb = cb3;
        if (mf & V3_META_CORE_MODS_PRESENT) != 0 {
            model.ctrl.core_mods_present = true;
            model.ctrl.stage_latent_core_mods_fp16 =
                a3.sona_core_ultra.stage_latent_core_mods_fp16.map(U16::get);
        }
        model.readouts[3] = model.readouts[2].clone();
        for i in 0..32 {
            model.readouts[3].fir_l[96 + i] = a3.fir_ultra.fir_l[i].get();
            model.readouts[3].fir_r[96 + i] = a3.fir_ultra.fir_r[i].get();
        }
        unpack_kautz_blocks(
            &a3.kautz_ultra,
            24,
            &mut model.kautz_poles,
            &mut model.readouts[3],
        );
        let apply_prefix_overrides = (mf & V3_META_STAGE3_PREFIX_OVERRIDES) != 0;
        unpack_readout_ultra(
            &a3.varpro_readout_ultra,
            &mut model.readouts[3],
            apply_prefix_overrides,
        );
        let q1 = model.stage4[1].effective_quad_params(1);
        let q2 = model.stage4[2].effective_quad_params(2);
        let q3 = unpack_stage4_quad_wire(&a3.stage4_ultra[0]);
        model.stage4[3] = rebuild_tier_stage4(Tier::Ultra, q0, Some(q1), Some(q2), Some(q3));
        unpack_arena3_ctrl_wire(&a3.ctrl_ultra, &mut model.ctrl);
    }

    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .enumerate()
    {
        model.readouts[t_idx].mask_to_tier(*tier);
    }
    model.ctrl.clamp_in_place();
    model.tier_crossbars = Some([cb0, cb1, cb2, cb3]);
    model.crossbar = highest_cb;
    model
}

/// Compatibility alias for [`unpack_sona_matryoshka_from_arena`].
#[inline]
#[must_use]
pub fn unpack_v3_matryoshka_from_arena(arena: &[u8], mono_out: bool) -> SonaMatryoshkaModel {
    unpack_sona_matryoshka_from_arena(arena, mono_out)
}

/// Unpacks and extracts a single [`SonaColdTierModel`] for `tier` from a serialized wire arena.
#[must_use]
pub fn unpack_sona_tier_from_arena(arena: &[u8], tier: Tier, mono_out: bool) -> SonaColdTierModel {
    unpack_sona_matryoshka_from_arena(arena, mono_out).extract_tier(tier)
}

/// Compatibility alias for [`unpack_sona_tier_from_arena`].
#[inline]
#[must_use]
pub fn unpack_v3_tier_from_arena(arena: &[u8], tier: Tier, mono_out: bool) -> SonaColdTierModel {
    unpack_sona_tier_from_arena(arena, tier, mono_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::stage4_tank::v3_fdn::Stage4ExtFormat;

    #[test]
    fn test_sona_arena0_to_arena3_zerocopy_round_trip() {
        let mut mat = SonaMatryoshkaModel::identity(false, false);
        mat.tier_os2x = [false, false, false, false];
        mat.os_params.up_quad_coeff = 0.0;
        mat.os_params.dec_c0 = 0.0;
        mat.stage0.xo_hz = 250.0;
        mat.stage0.s_router = [
            [0.8, 0.2, 0.9, -0.9],
            [0.1, 0.9, 0.5, 0.5],
            [0.7, 0.0, 0.6, 0.1],
            [0.0, 0.7, -0.1, 0.6],
        ];
        mat.crossbar.phasor_freq_hz = [1.2, 2.4, 3.6, 4.8];
        mat.crossbar.op_p_to_group[2][1] = 0.25;
        mat.crossbar.phasor_to_group[7][3] = -0.375;
        mat.crossbar.opt_to_lr[2] = [0.25, -0.5];
        mat.groups[0].svf_g = [0.12, 0.24, 0.36, 0.48];
        mat.groups[9].adaa_drive = [1.5, 2.0, 2.5, 3.0];
        mat.tiles[0].w_odd_4x4[1][2] = -0.5;
        mat.tiles[34].w_lru_4x4[3][1] = 0.75;
        mat.readouts[0].fir_l[31] = 0.5;
        mat.readouts[1].fir_l[0] = 0.25;
        mat.readouts[1].fir_l[47] = -0.25;
        mat.readouts[2].fir_r[95] = 0.125;
        mat.readouts[3].fir_l[127] = -0.625;
        mat.readouts[3].pre_cab_u_gain = [0.333_333_3, -0.123_456_7];
        mat.readouts[3].kautz_b_gamma[0] = 0.0;
        mat.readouts[3].kautz_c_l[17] = (0.123_456_7, -0.765_432_1);
        mat.readouts[3].kautz_c_l[31] = (0.3, -0.4);
        mat.stage4[2].quads[2].delay_lengths = [263, 293, 331, 367];
        mat.stage4[3].quads[3].ext_format = Stage4ExtFormat::F32;
        mat.stage4[3].quads[3].allpass_delays = [19, 31, 47, 73];
        mat.stage4[3].quads[3].allpass_g = [0.5, -0.5, 0.25, -0.25];
        mat.ctrl.w_adaa_drive_scale[7][9] = 0.625;
        mat.ctrl.sigma_drift = 0.15;
        mat.ctrl.rho_drift_train = 0.995;
        mat.ctrl.trust_radius_mh = 1.75;
        mat.ctrl.w_ctrl[3][11] = -0.42;
        mat.ctrl.w_ctrl_ultra[2][9] = 0.31;
        mat.ctrl.u_int_ultra[2][3][7] = -0.27;
        mat.ctrl.cp1_ultra[123] = 0x5A;
        mat.ctrl.set_core_mod(4, 42, -0.375);
        mat.ctrl.core_mods_present = true;

        let mut arena = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        pack_sona_matryoshka_to_arena(&mat, &mut arena);

        let a0 = Arena0LiteWire::ref_from_prefix(&arena)
            .expect("valid Arena0LiteWire")
            .0;
        assert_eq!(a0.stage0.cell_phasors.cell_desc.magic, *b"SONA");
        assert_eq!(a0.stage0.cell_phasors.cell_desc.version_major.get(), 3);
        assert_eq!(a0.stage0.cell_phasors.cell_desc.version_minor.get(), 0);
        assert_ne!(
            a0.stage0.meta.v3_meta_flags.get() & SONA_META_TIER_OS_EXPLICIT,
            0
        );
        assert_ne!(
            a0.stage0.meta.v3_meta_flags.get() & SONA_META_OS_EXT_PRESENT,
            0
        );
        assert_ne!(
            a0.stage0.meta.v3_meta_flags.get() & V3_META_CORE_MODS_PRESENT,
            0
        );

        let unp = unpack_sona_matryoshka_from_arena(&arena, false);
        assert_eq!(unp.tier_os2x, [false, false, false, false]);
        assert_eq!(unp.os_params.up_quad_coeff, 0.0);
        assert_eq!(unp.os_params.dec_c0, 0.0);
        assert_eq!(unp.stage0.s_router[0][2], 0.0);
        assert_eq!(unp.stage0.s_router[1][3], 0.0);
        assert!((unp.stage0.s_router[2][0] - 0.7).abs() < 1e-6);
        assert!((unp.stage0.xo_hz - 250.0).abs() < 1e-6);
        assert!((unp.crossbar.opt_to_lr[2][0] - 0.25).abs() < 1e-3);
        assert!((unp.groups[0].svf_g[3] - 0.48).abs() < 1e-6);
        assert!((unp.groups[9].adaa_drive[3] - 3.0).abs() < 1e-6);
        assert!((unp.tiles[0].w_odd_4x4[1][2] + 0.5).abs() < 1e-6);
        assert!((unp.tiles[34].w_lru_4x4[3][1] - 0.75).abs() < 1e-3);
        assert!((unp.readouts[0].fir_l[31] - 0.5).abs() < 1e-6);
        assert!((unp.readouts[1].fir_l[0] - 0.25).abs() < 1e-6);
        assert!((unp.readouts[1].fir_l[47] + 0.25).abs() < 1e-6);
        assert!((unp.readouts[2].fir_r[95] - 0.125).abs() < 1e-6);
        assert!((unp.readouts[3].fir_l[127] + 0.625).abs() < 1e-6);
        assert!((unp.readouts[3].pre_cab_u_gain[0] - 0.333_333_3).abs() < 1e-3);
        assert_eq!(unp.readouts[3].kautz_b_gamma[0].to_bits(), 0);
        assert!((unp.readouts[3].kautz_c_l[17].0 - 0.123_456_7).abs() < 1e-3);
        assert_eq!(unp.stage4[2].quads[2].delay_lengths, [263, 293, 331, 367]);
        assert_eq!(unp.stage4[3].quads[3].ext_format, Stage4ExtFormat::F32);
        assert_eq!(unp.stage4[3].quads[3].allpass_delays, [19, 31, 47, 73]);
        assert!((unp.ctrl.w_adaa_drive_scale[7][9] - 0.625).abs() < 1e-3);
        assert!((unp.ctrl.w_ctrl[3][11] + 0.42).abs() < 1e-6);
        assert!((unp.ctrl.w_ctrl_ultra[2][9] - 0.31).abs() < 1e-6);
        assert!((unp.ctrl.u_int_ultra[2][3][7] + 0.27).abs() < 1e-6);
        assert_eq!(unp.ctrl.cp1_ultra[123], 0x5A);
        assert!(unp.ctrl.core_mods_present);
        assert!((unp.ctrl.core_mod(4, 42) + 0.375).abs() < 1e-6);

        // Self-contained Arena 0 Lite-only unpack & exact unstripped-vs-stripped tier equivalence
        let lite_unp = unpack_sona_tier_from_arena(&arena[..ARENA_LITE_BYTES], Tier::Lite, false);
        assert_eq!(lite_unp.tier, Tier::Lite);
        assert!((lite_unp.crossbar.op_p_to_group[2][1] - 0.25).abs() < 1e-3);

        let mut eq_mat = SonaMatryoshkaModel::identity(false, false);
        let mut cbs = [SonaCrossbarConfig::identity(); 4];
        for (idx, cb) in cbs.iter_mut().enumerate() {
            cb.op_p_to_group[0][0] = 0.1 * ((idx + 1) as f32);
            cb.opt_to_lr[0][0] = -0.1 * ((idx + 1) as f32);
            cb.phasor_stereo_phase_rad = [0.25 * (idx as f32); 4];
        }
        eq_mat.tier_crossbars = Some(cbs);
        eq_mat.ctrl.lite_block_scales_present = true;
        eq_mat.ctrl.w_adaa_drive_scale_lite[0][0] = 0.25;
        eq_mat.ctrl.w_adaa_drive_scale[0][0] = 0.9;
        let mut eq_arena = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        pack_sona_matryoshka_to_arena(&eq_mat, &mut eq_arena);
        assert_eq!(
            unpack_sona_tier_from_arena(&eq_arena, Tier::Lite, false),
            unpack_sona_tier_from_arena(&eq_arena[..ARENA_LITE_BYTES], Tier::Lite, false)
        );
        assert_eq!(
            unpack_sona_tier_from_arena(&eq_arena, Tier::Full, false),
            unpack_sona_tier_from_arena(&eq_arena[..ARENA_FULL_BYTES], Tier::Full, false)
        );
        assert_eq!(
            unpack_sona_tier_from_arena(&eq_arena, Tier::Pro, false),
            unpack_sona_tier_from_arena(&eq_arena[..ARENA_PRO_BYTES], Tier::Pro, false)
        );
    }
}
