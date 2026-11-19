//! Lossless `v3` Matryoshka Packing & Unpacking (`Arena 0..3`: `4,608 / 10,240 / 18,432 / 28,672 B`).

pub mod gru;
pub mod readout;

pub use gru::{GRU_FEED_BYTES, GRU_TILE_TRIPLE_BYTES, write_f32_le, write_u32_le};

use crate::stages::router::V3Stage0Params;
use crate::stages::stage2_gru::{NUM_GRU_BLOCKS, V3InterBlockToneSvfParams, V3OversamplerParams};
use crate::stages::stage3_cab::{
    V3ColdTierModel, V3MatryoshkaModel, V3Rate1xCalibration, V3TierReadout,
};
use crate::stages::weights::{
    ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET, ARENA0_DRY_WET_OFFSET,
    ARENA0_FEAT_DET_THRESH_OFFSET, ARENA0_OS_EXT_PARAMS_OFFSET, ARENA0_OS_PARAMS_OFFSET,
    ARENA0_PIPELINE_FLAGS_OFFSET, ARENA0_R_AC_OFFSET, ARENA0_STAGE0_ENV_MIX_OFFSET,
    ARENA0_STAGE0_EXT_OFFSET, ARENA0_STAGE0_PARAMS_OFFSET, ARENA0_STAGE1_PRE_OFFSET,
    ARENA0_TIER_GAIN_TRIM_OFFSET, ARENA0_TONE_SVF_PARAMS_OFFSET, ARENA0_V3_META_OFFSET,
    ARENA2_FEAT_DET_THRESH_OFFSET, ARENA2_STAGE1_PRE_OFFSET, ARENA2_STAGE2_GRU_OFFSET,
    PIPE_FLAG_UNIFIED_W_CTRL, PIPE_FLAG_V3_MATRYOSHKA, PIPE_MAGIC_HI, STAGE4_LFO_OFFSETS,
    STAGE4_TAP_GAINS_OFFSETS, V3_META_STAGE0_EXT, V3_META_STAGE0_ROUTE_CLEAN,
    V3_META_STAGE0_SUB2_PHASE, V3_META_STAGE3_PREFIX_OVERRIDES, V3_META_STAGE4_LFO,
    V3_META_STAGE4_TAP_GAINS, has_pipeline_flag, read_f32_le, read_u32_le,
};
use crate::types::Tier;
use gru::{
    pack_1x_lite_full_gru, pack_2x_pro_ultra_gru, pack_stage1_and_feat, pack_stage4,
    unpack_1x_lite_full_gru, unpack_2x_pro_ultra_gru, unpack_stage1_and_feat, unpack_stage4,
};
use readout::{
    KAUTZ_OFFSETS, STAGE4_OFFSETS, overlay_unflagged_tier_slices, pack_ctrl_params,
    pack_tier_readout_slice, unpack_ctrl_params, unpack_tier_readout_slice,
};

/// Packs a unified 4-tier [`V3MatryoshkaModel`] into a `28,672 B` (`ARENA_ULTRA_BYTES`) buffer.
pub fn pack_v3_matryoshka_to_arena(model: &V3MatryoshkaModel, arena: &mut [u8]) {
    if arena.len() < ARENA_LITE_BYTES {
        return;
    }
    let existing_flags = read_u32_le(arena, ARENA0_PIPELINE_FLAGS_OFFSET);
    let flags = PIPE_MAGIC_HI | existing_flags | PIPE_FLAG_UNIFIED_W_CTRL | PIPE_FLAG_V3_MATRYOSHKA;
    write_u32_le(arena, ARENA0_PIPELINE_FLAGS_OFFSET, flags);
    for t in 0..4 {
        write_f32_le(arena, ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4, 1.0);
    }

    let mut meta: u32 = 0;
    for t in 0..4 {
        if model.tier_os2x[t] {
            meta |= 1u32 << t;
        }
        let b_enc = (model.tier_blocks[t].clamp(1, NUM_GRU_BLOCKS) - 1) as u32;
        meta |= (b_enc & 0x03) << (8 + t * 2);
    }
    if model.os2x {
        meta |= 1u32 << 4;
    }
    if model.mono_out {
        meta |= 1u32 << 5;
    }
    if model.rate_1x_cal.is_some() {
        meta |= 1u32 << 6;
    }
    meta |= V3_META_STAGE3_PREFIX_OVERRIDES
        | V3_META_STAGE4_TAP_GAINS
        | V3_META_STAGE4_LFO
        | V3_META_STAGE0_EXT;
    if model.stage0.op_p_route_clean {
        meta |= V3_META_STAGE0_ROUTE_CLEAN;
    }
    if model.stage0.op_p_sub2_phase {
        meta |= V3_META_STAGE0_SUB2_PHASE;
    }
    write_u32_le(arena, ARENA0_V3_META_OFFSET, meta);
    write_f32_le(arena, ARENA0_STAGE0_EXT_OFFSET, model.stage0.op_p_lpf_pole);
    write_f32_le(
        arena,
        ARENA0_STAGE0_EXT_OFFSET + 4,
        model.stage0.op_p_env_pole,
    );
    write_f32_le(
        arena,
        ARENA0_STAGE0_EXT_OFFSET + 8,
        model.stage0.op_p_sub2_mix,
    );
    write_f32_le(
        arena,
        ARENA0_STAGE0_EXT_OFFSET + 12,
        model.stage0.op_p_oct_mix,
    );
    write_f32_le(
        arena,
        ARENA0_STAGE0_ENV_MIX_OFFSET,
        model.stage0.op_p_env_mix,
    );
    write_f32_le(arena, ARENA0_DRY_WET_OFFSET, 1.0);
    write_f32_le(arena, ARENA0_R_AC_OFFSET, 0.9993);

    let s0 = [
        model.stage0.xo_hz,
        model.stage0.op_p_sub_mix,
        model.stage0.clean_sub_mix,
        model.stage0.gate_thresh,
        model.stage0.gate_hyst,
        model.stage0.gate_inv_knee,
        model.stage0.gate_floor,
    ];
    for (i, &v) in s0.iter().enumerate() {
        write_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + i * 4, v);
    }
    write_f32_le(arena, ARENA0_OS_PARAMS_OFFSET, model.os_params.up_aa_mix);
    write_f32_le(
        arena,
        ARENA0_OS_PARAMS_OFFSET + 4,
        model.os_params.up_pre_emph,
    );
    write_f32_le(
        arena,
        ARENA0_OS_PARAMS_OFFSET + 8,
        model.os_params.dec_aa_mix,
    );
    let os_ext = [
        model.os_params.up_quad_coeff,
        model.os_params.dec_c_prev1,
        model.os_params.dec_c0,
        model.os_params.dec_c1,
        model.os_params.up_zoh_mix,
    ];
    for (i, &v) in os_ext.iter().enumerate() {
        write_f32_le(arena, ARENA0_OS_EXT_PARAMS_OFFSET + i * 4, v);
    }
    write_f32_le(arena, ARENA0_ANTI_ZIPPER_RAMP_SCALE_OFFSET, 0.0625);

    let ts = &model.tone_svf;
    let tone_floats = [
        ts.base_cutoff_hz,
        ts.sample_rate_hz,
        ts.q_res,
        ts.dry_mix,
        ts.lp_mix,
        ts.bp_mix,
        ts.hp_mix,
        ts.h0_proj_w[0],
        ts.h0_proj_w[1],
        ts.h0_proj_w[2],
        ts.h0_proj_w[3],
        ts.s1_proj_w,
        ts.cab_inject_w,
        ts.b1_inject_w,
    ];
    for (i, &v) in tone_floats.iter().enumerate() {
        write_f32_le(arena, ARENA0_TONE_SVF_PARAMS_OFFSET + i * 4, v);
    }

    let (svf_1x, feat_1x, blk_1x) = if let Some(ref cal) = model.rate_1x_cal {
        (cal.svf, cal.feat_params, cal.blocks)
    } else if !model.tier_os2x[0] && model.os2x {
        let mut s = model.svf;
        let mut f = model.feat_params;
        let mut b = model.blocks;
        s.rescale_2x_to_1x();
        f.rescale_2x_to_1x();
        for blk in &mut b {
            blk.rescale_2x_to_1x();
        }
        (s, f, b)
    } else {
        (model.svf, model.feat_params, model.blocks)
    };
    pack_stage1_and_feat(arena, ARENA0_STAGE1_PRE_OFFSET, &svf_1x, &feat_1x);
    write_f32_le(arena, ARENA0_FEAT_DET_THRESH_OFFSET, feat_1x.det_thresh);
    pack_1x_lite_full_gru(arena, &blk_1x);

    if arena.len() >= ARENA_PRO_BYTES {
        pack_stage1_and_feat(
            arena,
            ARENA2_STAGE1_PRE_OFFSET,
            &model.svf,
            &model.feat_params,
        );
        write_f32_le(
            arena,
            ARENA2_FEAT_DET_THRESH_OFFSET,
            model.feat_params.det_thresh,
        );
        pack_2x_pro_ultra_gru(arena, &model.blocks);
        pack_ctrl_params(arena, &model.ctrl);
    } else if arena.len() >= 10_240 {
        pack_ctrl_params(arena, &model.ctrl);
    }

    for (t_idx, &tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .enumerate()
    {
        if arena.len() < tier.expected_arena_bytes() {
            break;
        }
        let k_off = KAUTZ_OFFSETS[t_idx];
        let k_start = t_idx * 4;
        for m in 0..4 {
            let (r, th) = model.kautz_poles[k_start + m];
            write_f32_le(arena, k_off + m * 4, r);
            write_f32_le(arena, k_off + 16 + m * 4, 0.0);
            write_f32_le(arena, k_off + 32 + m * 4, th);
            let bg_stored = model.readouts[t_idx].kautz_b_gamma[k_start + m];
            let bg = if bg_stored > 0.0 {
                bg_stored
            } else {
                crate::stages::stage3_cab::kautz_zoh::cold_b_gamma(r.clamp(0.25, 0.9995))
            };
            write_f32_le(arena, k_off + 48 + m * 4, bg);
        }
        pack_stage4(arena, STAGE4_OFFSETS[t_idx], &model.stage4[t_idx]);
        let tg_off = STAGE4_TAP_GAINS_OFFSETS[t_idx];
        for (l, &g) in model.stage4[t_idx].tap_gains.iter().enumerate() {
            write_f32_le(arena, tg_off + l * 4, g);
        }
        let lfo_off = STAGE4_LFO_OFFSETS[t_idx];
        write_f32_le(arena, lfo_off, model.stage4[t_idx].lfo_rate_hz);
        write_f32_le(arena, lfo_off + 4, model.stage4[t_idx].lfo_depth_samples);
        write_f32_le(arena, lfo_off + 8, model.stage4[t_idx].lfo_stereo_phase);
        pack_tier_readout_slice(arena, tier, &model.readouts[t_idx]);
    }
}

/// Unpacks a `v3` (or zeroed/unflagged legacy test) arena buffer into [`V3MatryoshkaModel`].
#[must_use]
pub fn unpack_v3_matryoshka_from_arena(arena: &[u8], mono_out: bool) -> V3MatryoshkaModel {
    if !has_pipeline_flag(arena, PIPE_FLAG_V3_MATRYOSHKA) {
        let mut model = V3MatryoshkaModel::dual_rate_default(mono_out);
        overlay_unflagged_tier_slices(arena, &mut model);
        return model;
    }

    let meta = read_u32_le(arena, ARENA0_V3_META_OFFSET);
    let mut model =
        V3MatryoshkaModel::identity((meta & (1 << 4)) != 0, mono_out || ((meta & (1 << 5)) != 0));
    for t in 0..4 {
        model.tier_os2x[t] = (meta & (1u32 << t)) != 0;
        let b_enc = ((meta >> (8 + t * 2)) & 0x03) as usize;
        model.tier_blocks[t] = (b_enc + 1).clamp(1, NUM_GRU_BLOCKS);
    }

    model.stage0 = V3Stage0Params {
        xo_hz: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET),
        op_p_sub_mix: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + 4),
        clean_sub_mix: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + 8),
        gate_thresh: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + 12),
        gate_hyst: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + 16),
        gate_inv_knee: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + 20),
        gate_floor: read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET + 24),
        op_p_lpf_pole: if (meta & V3_META_STAGE0_EXT) != 0 {
            read_f32_le(arena, ARENA0_STAGE0_EXT_OFFSET)
        } else {
            V3Stage0Params::identity().op_p_lpf_pole
        },
        op_p_env_pole: if (meta & V3_META_STAGE0_EXT) != 0 {
            read_f32_le(arena, ARENA0_STAGE0_EXT_OFFSET + 4)
        } else {
            0.0
        },
        op_p_sub2_mix: if (meta & V3_META_STAGE0_EXT) != 0 {
            read_f32_le(arena, ARENA0_STAGE0_EXT_OFFSET + 8)
        } else {
            0.0
        },
        op_p_oct_mix: if (meta & V3_META_STAGE0_EXT) != 0 {
            read_f32_le(arena, ARENA0_STAGE0_EXT_OFFSET + 12)
        } else {
            0.0
        },
        op_p_env_mix: if (meta & V3_META_STAGE0_EXT) != 0 {
            read_f32_le(arena, ARENA0_STAGE0_ENV_MIX_OFFSET)
        } else {
            0.0
        },
        op_p_sub2_phase: (meta & V3_META_STAGE0_SUB2_PHASE) != 0,
        op_p_route_clean: (meta & V3_META_STAGE0_ROUTE_CLEAN) != 0,
        ..V3Stage0Params::identity()
    };
    model.stage0.clamp_in_place();
    let ext_quad = read_f32_le(arena, ARENA0_OS_EXT_PARAMS_OFFSET);
    let ext_prev1 = read_f32_le(arena, ARENA0_OS_EXT_PARAMS_OFFSET + 4);
    let ext_c0 = read_f32_le(arena, ARENA0_OS_EXT_PARAMS_OFFSET + 8);
    let ext_c1 = read_f32_le(arena, ARENA0_OS_EXT_PARAMS_OFFSET + 12);
    let ext_zoh = read_f32_le(arena, ARENA0_OS_EXT_PARAMS_OFFSET + 16);
    let has_ext_os = (ext_prev1 + ext_c0 + ext_c1).abs() > 0.5;
    let os_def = V3OversamplerParams::default();
    model.os_params = V3OversamplerParams {
        up_quad_coeff: if has_ext_os {
            ext_quad
        } else {
            os_def.up_quad_coeff
        },
        dec_c_prev1: if has_ext_os {
            ext_prev1
        } else {
            os_def.dec_c_prev1
        },
        dec_c0: if has_ext_os { ext_c0 } else { os_def.dec_c0 },
        dec_c1: if has_ext_os { ext_c1 } else { os_def.dec_c1 },
        up_aa_mix: read_f32_le(arena, ARENA0_OS_PARAMS_OFFSET),
        up_pre_emph: read_f32_le(arena, ARENA0_OS_PARAMS_OFFSET + 4),
        dec_aa_mix: read_f32_le(arena, ARENA0_OS_PARAMS_OFFSET + 8),
        up_zoh_mix: if has_ext_os {
            ext_zoh
        } else {
            os_def.up_zoh_mix
        },
    };
    model.os_params.clamp_in_place();

    let toff = ARENA0_TONE_SVF_PARAMS_OFFSET;
    model.tone_svf = V3InterBlockToneSvfParams {
        base_cutoff_hz: read_f32_le(arena, toff),
        sample_rate_hz: read_f32_le(arena, toff + 4),
        q_res: read_f32_le(arena, toff + 8),
        dry_mix: read_f32_le(arena, toff + 12),
        lp_mix: read_f32_le(arena, toff + 16),
        bp_mix: read_f32_le(arena, toff + 20),
        hp_mix: read_f32_le(arena, toff + 24),
        h0_proj_w: [
            read_f32_le(arena, toff + 28),
            read_f32_le(arena, toff + 32),
            read_f32_le(arena, toff + 36),
            read_f32_le(arena, toff + 40),
        ],
        s1_proj_w: read_f32_le(arena, toff + 44),
        cab_inject_w: read_f32_le(arena, toff + 48),
        b1_inject_w: read_f32_le(arena, toff + 52),
    };
    model.tone_svf.clamp_in_place();

    let has_s0_ext = (meta & V3_META_STAGE0_EXT) != 0;
    let (svf_1x, mut feat_1x) = unpack_stage1_and_feat(arena, ARENA0_STAGE1_PRE_OFFSET);
    if has_s0_ext {
        let dt = read_f32_le(arena, ARENA0_FEAT_DET_THRESH_OFFSET);
        if dt.is_finite() {
            feat_1x.det_thresh = dt.clamp(0.0, 2.0);
        }
    }
    let blk_1x = unpack_1x_lite_full_gru(arena, Tier::Full);
    if arena.len() >= ARENA2_STAGE2_GRU_OFFSET + 3_712 {
        if (meta & (1 << 6)) != 0 || (!model.tier_os2x[0] && model.os2x) {
            model.rate_1x_cal = Some(V3Rate1xCalibration {
                svf: svf_1x,
                feat_params: feat_1x,
                blocks: blk_1x,
            });
        }
        let (svf_2x, mut feat_2x) = unpack_stage1_and_feat(arena, ARENA2_STAGE1_PRE_OFFSET);
        if has_s0_ext {
            feat_2x.det_thresh = if arena.len() >= ARENA2_FEAT_DET_THRESH_OFFSET + 4 {
                let dt2 = read_f32_le(arena, ARENA2_FEAT_DET_THRESH_OFFSET);
                if dt2.is_finite() {
                    dt2.clamp(0.0, 2.0)
                } else {
                    feat_1x.det_thresh
                }
            } else {
                feat_1x.det_thresh
            };
        }
        model.svf = svf_2x;
        model.feat_params = feat_2x;
        model.blocks = unpack_2x_pro_ultra_gru(arena, Tier::Ultra);
    } else {
        model.os2x = model.tier_os2x[0];
        model.rate_1x_cal = None;
        model.svf = svf_1x;
        model.feat_params = feat_1x;
        model.blocks = unpack_1x_lite_full_gru(
            arena,
            if arena.len() >= 10_240 {
                Tier::Full
            } else {
                Tier::Lite
            },
        );
    }
    model.ctrl = unpack_ctrl_params(arena);
    let has_prefix_overrides = (meta & V3_META_STAGE3_PREFIX_OVERRIDES) != 0;

    for (t_idx, &tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .enumerate()
    {
        if arena.len() < tier.expected_arena_bytes() {
            break;
        }
        let k_off = KAUTZ_OFFSETS[t_idx];
        let k_start = t_idx * 4;
        for m in 0..4 {
            let r_low = read_f32_le(arena, k_off + m * 4);
            let dr = read_f32_le(arena, k_off + 16 + m * 4);
            let th = read_f32_le(arena, k_off + 32 + m * 4);
            model.kautz_poles[k_start + m] = (r_low + 0.5 * dr, th);
        }
        model.stage4[t_idx] = unpack_stage4(arena, STAGE4_OFFSETS[t_idx]);
        if (meta & V3_META_STAGE4_TAP_GAINS) != 0 {
            let tg_off = STAGE4_TAP_GAINS_OFFSETS[t_idx];
            for l in 0..4 {
                model.stage4[t_idx].tap_gains[l] = read_f32_le(arena, tg_off + l * 4);
            }
        }
        if (meta & V3_META_STAGE4_LFO) != 0 {
            let lfo_off = STAGE4_LFO_OFFSETS[t_idx];
            model.stage4[t_idx].lfo_rate_hz = read_f32_le(arena, lfo_off);
            model.stage4[t_idx].lfo_depth_samples = read_f32_le(arena, lfo_off + 4);
            model.stage4[t_idx].lfo_stereo_phase = read_f32_le(arena, lfo_off + 8);
        }
        model.stage4[t_idx].clamp_in_place();
        model.readouts[t_idx] = V3TierReadout::zeros();
        unpack_tier_readout_slice(
            arena,
            tier,
            has_prefix_overrides,
            &mut model.readouts[t_idx],
        );
    }
    model
}

/// Unpacks a single executable [`V3ColdTierModel`] directly for `tier` from `arena`.
#[must_use]
pub fn unpack_v3_tier_from_arena(arena: &[u8], tier: Tier, mono_out: bool) -> V3ColdTierModel {
    let mat = unpack_v3_matryoshka_from_arena(arena, mono_out);
    if !has_pipeline_flag(arena, PIPE_FLAG_V3_MATRYOSHKA) {
        return mat.extract_tier(tier);
    }
    let t = tier as usize;
    let tier_os2x = mat.tier_os2x[t];
    let mut cold = mat.extract_tier(tier);
    if !tier_os2x && (arena.len() < 10_240 || t <= 1) {
        let meta = read_u32_le(arena, ARENA0_V3_META_OFFSET);
        let (svf_1x, mut feat_1x) = unpack_stage1_and_feat(arena, ARENA0_STAGE1_PRE_OFFSET);
        if (meta & V3_META_STAGE0_EXT) != 0 {
            let dt = read_f32_le(arena, ARENA0_FEAT_DET_THRESH_OFFSET);
            if dt.is_finite() {
                feat_1x.det_thresh = dt.clamp(0.0, 2.0);
            }
        }
        cold.svf = svf_1x;
        cold.feat_params = feat_1x;
        cold.blocks = unpack_1x_lite_full_gru(
            arena,
            if arena.len() < 10_240 {
                Tier::Lite
            } else {
                tier
            },
        );
    }
    cold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::hot_core::V3HotCore as HotCore;
    use crate::stages::weights::ARENA_ULTRA_BYTES;
    use alloc::vec;
    use proptest::prelude::*;

    #[test]
    fn test_v3_pack_unpack_roundtrip_and_zeroed_fallback() {
        let zero_arena = vec![0u8; ARENA_ULTRA_BYTES];
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let cold = unpack_v3_tier_from_arena(&zero_arena, tier, false);
            let mut core = HotCore::<crate::math::DefaultMathBackend>::from_cold_model(&cold);
            let y = core.step_stereo(0.0, 0.0);
            assert_eq!(y.map(f32::to_bits), [0, 0]);
        }

        let mut mat = V3MatryoshkaModel::dual_rate_default(false);
        mat.stage0.op_p_sub_mix = 0.22;
        mat.stage0.op_p_lpf_pole = 0.93;
        mat.stage0.op_p_env_pole = 0.992;
        mat.stage0.op_p_sub2_mix = 0.45;
        mat.stage0.op_p_oct_mix = 0.625;
        mat.stage0.op_p_env_mix = -0.25;
        mat.stage0.op_p_sub2_phase = true;
        mat.stage0.op_p_route_clean = true;
        mat.feat_params.key_link = 0.875;
        mat.feat_params.det_thresh = 0.22;
        mat.os_params.up_quad_coeff = -0.08;
        mat.os_params.dec_c_prev1 = 0.21;
        mat.os_params.dec_c0 = 0.44;
        mat.os_params.dec_c1 = 0.35;
        mat.os_params.up_zoh_mix = 0.30;
        mat.os_params.clamp_in_place();
        mat.svf.base_g = 0.08;
        mat.svf.atk_alpha = 0.91;
        mat.svf.rel_alpha = 0.992;
        mat.svf.det_lp_pole = 0.82;
        mat.svf.sec2_hp_feed = 0.35;
        mat.svf.stereo_link = 0.75;
        mat.svf.g_min = 0.01;
        mat.svf.g_max = 0.88;
        mat.svf.feat_lp2_mix = 0.9;
        mat.blocks[0].w_in_n[0][0] = 0.75;
        mat.blocks[0].w_rec_n[0][0] = 0.12;
        mat.blocks[0].w_in_n[1][4] = -0.35;
        mat.blocks[0].w_in_n[2][8] = 0.42;
        mat.blocks[1].w_prev_n[0][0] = 0.28;
        mat.tone_svf.b1_inject_w = 0.18;
        mat.kautz_poles[0] = (0.93, 0.44);
        mat.kautz_poles[15] = (0.87, 1.12);
        mat.ctrl.w_film_gain[0][0][0] = 0.3;
        for t in 0..4 {
            mat.readouts[t].pre_cab_w[0][0] = 0.35;
            mat.readouts[t].fir_l[0] = 0.25 + 0.1 * (t as f32);
            mat.readouts[t].fir_l[t * 16] = 0.5;
            mat.readouts[t].kautz_c_l[0] = (0.1 + 0.05 * (t as f32), -0.04);
            mat.readouts[t].kautz_c_l[t * 4] = (0.12, -0.05);
            mat.stage4[t].wet_mix = 0.2;
            mat.stage4[t].tap_gains = [0.7 - 0.1 * (t as f32), -0.45, 0.25, 0.85];
            mat.stage4[t].lfo_rate_hz = 1.1 + 0.35 * (t as f32);
            mat.stage4[t].lfo_depth_samples = 6.0 + 4.0 * (t as f32);
            mat.stage4[t].lfo_stereo_phase = 0.4 * (t as f32) - 0.5;
        }

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        pack_v3_matryoshka_to_arena(&mat, &mut arena);
        let unpacked = unpack_v3_matryoshka_from_arena(&arena, false);
        assert_eq!(unpacked.stage0, mat.stage0);
        assert_eq!(unpacked.feat_params, mat.feat_params);
        assert_eq!(unpacked.os_params, mat.os_params);
        assert_eq!(unpacked.svf, mat.svf);
        assert_eq!(unpacked.tone_svf, mat.tone_svf);
        assert_eq!(unpacked.kautz_poles[15], (0.87, 1.12));
        assert_eq!(unpacked.ctrl.w_film_gain[0][0][0], 0.3);
        for t in 0..4 {
            assert_eq!(unpacked.stage4[t].tap_gains, mat.stage4[t].tap_gains);
            assert_eq!(unpacked.stage4[t].lfo_rate_hz, mat.stage4[t].lfo_rate_hz);
            assert_eq!(
                unpacked.stage4[t].lfo_depth_samples,
                mat.stage4[t].lfo_depth_samples
            );
            assert_eq!(
                unpacked.stage4[t].lfo_stereo_phase,
                mat.stage4[t].lfo_stereo_phase
            );
            assert_eq!(unpacked.readouts[t].fir_l, mat.readouts[t].fir_l);
            assert_eq!(unpacked.readouts[t].kautz_c_l, mat.readouts[t].kautz_c_l);
        }

        let pro_from_full = unpack_v3_tier_from_arena(&arena, Tier::Pro, false);
        let mut pro_arena = vec![0u8; ARENA_PRO_BYTES];
        pack_v3_matryoshka_to_arena(&mat, &mut pro_arena);
        let pro_from_prefix = unpack_v3_tier_from_arena(&pro_arena, Tier::Pro, false);
        assert_eq!(pro_from_prefix.os_params, mat.os_params);
        assert_eq!(pro_from_prefix.svf, pro_from_full.svf);
        assert_eq!(pro_from_prefix.stage4, pro_from_full.stage4);
        assert_eq!(pro_from_prefix.readout.fir_l, pro_from_full.readout.fir_l);

        let lite_from_full = unpack_v3_tier_from_arena(&arena, Tier::Lite, false);
        let lite_from_stripped =
            unpack_v3_tier_from_arena(&arena[..ARENA_LITE_BYTES], Tier::Lite, false);
        assert_eq!(
            lite_from_full.blocks[0].b_n,
            lite_from_stripped.blocks[0].b_n
        );
        assert_eq!(
            lite_from_full.readout.fir_l,
            lite_from_stripped.readout.fir_l
        );
    }

    proptest! {
        #[test]
        fn prop_v3_pack_unpack_audio_equivalence(
            w0 in -1.5_f32..1.5_f32,
            w1 in -1.5_f32..1.5_f32,
            b0 in -2.0_f32..2.0_f32,
        ) {
            let mut mat = V3MatryoshkaModel::dual_rate_default(false);
            mat.blocks[0].w_in_n[0][0] = w0;
            mat.blocks[0].w_in_n[0][4] = w1;
            mat.blocks[0].b_n[0] = b0;
            let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
            pack_v3_matryoshka_to_arena(&mat, &mut arena);
            for tier in [Tier::Lite, Tier::Pro, Tier::Ultra] {
                let c_orig = mat.extract_tier(tier);
                let c_unpk = unpack_v3_tier_from_arena(&arena, tier, false);
                let mut h1 = HotCore::<crate::math::DefaultMathBackend>::from_cold_model(&c_orig);
                let mut h2 = HotCore::<crate::math::DefaultMathBackend>::from_cold_model(&c_unpk);
                let y1 = h1.step_stereo(0.25, -0.2);
                let y2 = h2.step_stereo(0.25, -0.2);
                prop_assert!((y1[0] - y2[0]).abs() < 1.0e-6);
                prop_assert!((y1[1] - y2[1]).abs() < 1.0e-6);
            }
        }
    }
}
