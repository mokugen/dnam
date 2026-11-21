//! Tier crossbars, ZOH group/Kautz/FIR rescaling, and `Arena0..3` crossbar hydration
//! (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OpNPreHoisted};
use zerocopy::byteorder::little_endian::F32;

use crate::bin::{CABINET_KAUTZ_BLOCKS_PER_TIER, DnamBinView, OpNPreBaseWire};
use crate::graph::GraphBackend;
use crate::graph::cabinet::kautz::{
    KautzPrefixOverride, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS, hydrate_kautz_slice_and_zoh,
};
use crate::graph::oversample::SonaOs2xHot;
use crate::graph::zoh::{ZohRateContext, rescale_paralru_lane};
use crate::math::{
    Q32_SCALE_F64, clamp_finite_f32, f32_to_fp16, fp16_to_f32, powf_f64, rescale_tpt_svf_g_zoh,
    round_half_away_f64, sanitize_finite_f32,
};
use crate::types::{Tier, VmError};

use super::super::layout::{
    FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, MAX_SONA_GROUPS,
    MAX_SONA_TILES, SonaHotSlab,
};
use super::wire_decode::{
    HEADER_FLAG_SINGLE_INPUT, HEADER_FLAG_STEREO_OUT, HEADER_FLAG_WET_POLARITY_INV,
    META_CORE_MODS_PRESENT, META_OS_EXT_PRESENT, clamp_finite_fp16, compute_tier_os2x_flags,
    decode_group_wire, decode_tile_wire, fp16_sanitized,
};

const HP_LP_BASE_POLE: f64 = 0.95;

/// Quantizes signed carrier frequency `f_hz` clamped to `[-0.49 * fs_1x, +0.49 * fs_1x]`
/// using precomputed `inv_fs_1x` into signed two's-complement `i32` `Q0.32` increment.
#[inline]
pub(crate) fn signed_phasor_inc_1x_with_inv(freq_hz: f32, fs_1x: f64, inv_fs_1x: f64) -> i32 {
    if !fs_1x.is_finite() || fs_1x <= 0.0 || !inv_fs_1x.is_finite() {
        return 0;
    }
    let lim = (0.49 * fs_1x) as f32;
    let f_eff = clamp_finite_f32(freq_hz, -lim, lim, 0.0);
    let max_i32 = f64::from(i32::MAX);
    let scaled = round_half_away_f64(f64::from(f_eff) * Q32_SCALE_F64 * inv_fs_1x);
    scaled.clamp(-max_i32, max_i32) as i64 as i32
}

fn pack_fir_tiles(dst: &mut [[[f32; 4]; 4]], src: &[f32]) {
    for (t, tile) in dst.iter_mut().enumerate() {
        let base = t * 16;
        for g in 0..4 {
            for l in 0..4 {
                tile[g][l] = src[base + g * 4 + l];
            }
        }
    }
}

pub(super) fn apply_tier_weights_and_rescale<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    tier: Tier,
    zoh: &ZohRateContext,
) -> Result<(), VmError> {
    let t_idx = tier.index();
    let active_groups = tier.simd_groups();
    let active_banks: usize = CABINET_KAUTZ_BLOCKS_PER_TIER[..=t_idx].iter().sum();
    let active_pairs = active_banks * 4;

    let snap = slab.cold().tier_crossbars[t_idx];
    slab.lite.ro_gains_fp16 = snap.ro_gains_fp16;
    slab.lite.phasor_st_cos_fp16 = snap.phasor_st_cos;
    slab.lite.phasor_st_sin_fp16 = snap.phasor_st_sin;
    slab.lite
        .op_p_to_group_lite_fp16
        .copy_from_slice(&snap.op_p_to_group[0..4]);
    slab.full_delta
        .op_p_to_group_full_fp16
        .copy_from_slice(&snap.op_p_to_group[4..6]);
    slab.pro_delta
        .op_p_to_group_pro_fp16
        .copy_from_slice(&snap.op_p_to_group[6..8]);
    slab.ultra_delta
        .op_p_to_group_ultra_fp16
        .copy_from_slice(&snap.op_p_to_group[8..10]);
    slab.lite
        .phasor_to_group_lite_fp16
        .copy_from_slice(&snap.phasor_to_group[0..4]);
    slab.full_delta
        .phasor_to_group_full_fp16
        .copy_from_slice(&snap.phasor_to_group[4..6]);
    slab.pro_delta
        .phasor_to_group_pro_fp16
        .copy_from_slice(&snap.phasor_to_group[6..8]);
    slab.ultra_delta
        .phasor_to_group_ultra_fp16
        .copy_from_slice(&snap.phasor_to_group[8..10]);
    slab.lite
        .opt_to_lr_lite_fp16
        .copy_from_slice(&snap.opt_to_lr[0..4]);
    slab.full_delta
        .opt_to_lr_full_fp16
        .copy_from_slice(&snap.opt_to_lr[4..8]);
    slab.pro_delta
        .opt_to_lr_pro_fp16
        .copy_from_slice(&snap.opt_to_lr[8..12]);
    slab.ultra_delta
        .opt_to_lr_ultra_fp16
        .copy_from_slice(&snap.opt_to_lr[12..16]);
    for g in 0..MAX_SONA_GROUPS {
        *slab.readout_hot_mut(g) = snap.readouts[g];
    }

    let mut any_crossbar = slab.cold().base_op_p_gain != 0.0
        || slab
            .lite
            .carrier_basis_w_fp16
            .iter()
            .flatten()
            .any(|&w| fp16_to_f32(w) != 0.0);
    for g in 0..active_groups {
        any_crossbar |= snap.op_p_to_group[g]
            .iter()
            .chain(snap.phasor_to_group[g].iter())
            .any(|&w| fp16_to_f32(w) != 0.0);
    }
    slab.set_flag(FLAG_CROSSBAR_ACTIVE, any_crossbar);

    for g in 0..MAX_SONA_GROUPS {
        let mut h = slab.cold().base_groups[g];
        for l in 0..4 {
            h.svf_g[l] = rescale_tpt_svf_g_zoh(h.svf_g[l], zoh.rho_core);
            let kap = fp16_to_f32(h.lru_kappa_fp16[l]);
            let (rel, atk, kap_out) =
                rescale_paralru_lane(h.lru_alpha_rel[l], h.lru_alpha_atk[l], kap, zoh.rho_core);
            h.lru_alpha_rel[l] = rel;
            h.lru_alpha_atk[l] = atk;
            h.lru_kappa_fp16[l] = f32_to_fp16(kap_out);
        }
        *slab.group_hot_mut(g) = h;
    }

    slab.lite.hp_lp_pole = (powf_f64(HP_LP_BASE_POLE, zoh.rho_core) as f32).clamp(0.0, 0.99999);
    let cold = slab.cold_mut();
    cold.core_mods_active = cold.core_mods_present && matches!(tier, Tier::Ultra);
    cold.active_kautz_blocks = active_banks;
    cold.active_tank_quads = t_idx + 1;
    cold.hoisted_core_valid = false;

    let (fir_l, fir_r) = (slab.cold().tier_fir_l[t_idx], slab.cold().tier_fir_r[t_idx]);
    pack_fir_tiles(&mut slab.lite.fir_weights_l, &fir_l[0..32]);
    pack_fir_tiles(&mut slab.lite.fir_weights_r, &fir_r[0..32]);
    pack_fir_tiles(&mut slab.full_delta.fir_weights_l_full, &fir_l[32..48]);
    pack_fir_tiles(&mut slab.full_delta.fir_weights_r_full, &fir_r[32..48]);
    pack_fir_tiles(&mut slab.pro_delta.fir_weights_l_pro, &fir_l[48..96]);
    pack_fir_tiles(&mut slab.pro_delta.fir_weights_r_pro, &fir_r[48..96]);
    pack_fir_tiles(&mut slab.ultra_delta.fir_weights_l_ultra, &fir_l[96..128]);
    pack_fir_tiles(&mut slab.ultra_delta.fir_weights_r_ultra, &fir_r[96..128]);

    let (mut c_re_l, mut c_im_l) = ([0.0_f32; MAX_KAUTZ_PAIRS], [0.0_f32; MAX_KAUTZ_PAIRS]);
    let (mut c_re_r, mut c_im_r) = ([0.0_f32; MAX_KAUTZ_PAIRS], [0.0_f32; MAX_KAUTZ_PAIRS]);
    let (mut b_gam, mut banks_tmp) = (
        [0.0_f32; MAX_KAUTZ_PAIRS],
        [Default::default(); MAX_KAUTZ_BANKS],
    );
    for m in 0..active_pairs {
        b_gam[m] = slab.cold().tier_kautz_b_gamma[t_idx][m];
        (c_re_l[m], c_im_l[m]) = slab.cold().tier_kautz_c_l[t_idx][m];
        (c_re_r[m], c_im_r[m]) = slab.cold().tier_kautz_c_r[t_idx][m];
    }
    let ov = KautzPrefixOverride {
        c_re_l: &c_re_l[..active_pairs],
        c_im_l: &c_im_l[..active_pairs],
        c_re_r: &c_re_r[..active_pairs],
        c_im_r: &c_im_r[..active_pairs],
        b_gamma: &b_gam[..active_pairs],
    };
    for b in 0..active_banks {
        banks_tmp[b] = *slab.kautz_bank_mut(b);
    }
    let wire_kautz = slab.cold().wire_kautz;
    let mut slew = slab.cold().kautz_slew;
    hydrate_kautz_slice_and_zoh(
        &wire_kautz[..active_banks],
        &mut banks_tmp[..active_banks],
        &mut slew,
        zoh.rho_vm,
        Some(&ov),
    )?;
    slab.cold_mut().kautz_slew = slew;
    for b in 0..active_banks {
        *slab.kautz_bank_mut(b) = banks_tmp[b];
    }
    Ok(())
}

pub(super) fn populate_crossbar_and_groups<B: GraphBackend>(
    view: &DnamBinView<'_>,
    slab: &mut SonaHotSlab<'_, B>,
) {
    let cb = &view.arena0.crossbar;
    let header_flags = view.header.flags.get();
    let meta_flags = cb.sona_meta_flags.get();
    slab.set_flag(
        FLAG_SINGLE_INPUT,
        (header_flags & HEADER_FLAG_SINGLE_INPUT) != 0,
    );
    slab.set_flag(
        FLAG_WET_POLARITY_INV,
        (header_flags & HEADER_FLAG_WET_POLARITY_INV) != 0,
    );
    slab.set_flag(FLAG_MONO_OUT, (header_flags & HEADER_FLAG_STEREO_OUT) == 0);

    let os_ext = (meta_flags & META_OS_EXT_PRESENT) != 0;
    let mut trims = [1.0_f32; 4];
    for (t, dst) in trims.iter_mut().enumerate() {
        let v = cb.tier_gain_trim[t].get();
        *dst = if v.is_finite() && v > 0.0 {
            v.clamp(0.0625, 16.0)
        } else {
            1.0
        };
    }
    let op_p_gain = clamp_finite_f32(
        fp16_to_f32(cb.gate_and_crossbar_fp16[6].get()),
        -4.0,
        4.0,
        0.0,
    );
    let op_t_gain = clamp_finite_f32(
        fp16_to_f32(cb.gate_and_crossbar_fp16[7].get()),
        -4.0,
        4.0,
        0.0,
    );
    if os_ext {
        for i in 0..3 {
            for j in 0..4 {
                slab.lite.carrier_basis_w_fp16[i][j] =
                    clamp_finite_fp16(cb.carrier_basis_w_fp16[i][j].get(), -8.0, 8.0, 0.0);
            }
        }
    } else {
        slab.lite.carrier_basis_w_fp16 = [[0; 4]; 3];
    }

    let op_n_pre_base = if os_ext {
        OpNPreBaseWire {
            gamma_bc: F32::new(clamp_finite_f32(
                cb.op_n_pre_base.gamma_bc.get(),
                0.0,
                1.0,
                0.0,
            )),
            zoh_hold_depth: F32::new(clamp_finite_f32(
                cb.op_n_pre_base.zoh_hold_depth.get(),
                0.0,
                OP_N_PRE_MAX_ZOH_HOLD_DEPTH,
                0.0,
            )),
            beta_q: F32::new(clamp_finite_f32(
                cb.op_n_pre_base.beta_q.get(),
                0.0,
                OP_N_PRE_MAX_BETA_Q,
                0.0,
            )),
        }
    } else {
        OpNPreBaseWire {
            gamma_bc: F32::new(0.0),
            zoh_hold_depth: F32::new(0.0),
            beta_q: F32::new(0.0),
        }
    };
    slab.os2x_hot = if os_ext {
        SonaOs2xHot::from_crossbar_wire(cb)
    } else {
        SonaOs2xHot::default()
    };

    let cold = slab.cold_mut();
    cold.wire_crossbar = *cb;
    cold.meta_flags = meta_flags;
    cold.tier_os2x = compute_tier_os2x_flags(meta_flags, cb.cell_desc.oversample_factor.get());
    cold.tier_gain_trim = trims;
    cold.base_op_n_pre = op_n_pre_base;
    cold.op_n_pre_hoisted = OpNPreHoisted::from_params(
        op_n_pre_base.gamma_bc.get(),
        op_n_pre_base.zoh_hold_depth.get(),
        op_n_pre_base.beta_q.get(),
        1.0,
    );
    cold.base_op_p_gain = op_p_gain;
    cold.base_op_t_gain = op_t_gain;
    cold.core_mods_present = (meta_flags & META_CORE_MODS_PRESENT) != 0;
    for l in 0..4 {
        cold.base_phasor_freq_hz[l] = sanitize_finite_f32(cb.phasor_freq_hz[l].get());
        cold.base_phasor_init_phase_q32[l] = cb.phasor_init_phase_q32[l].get();
    }
    slab.lite.phasors.phase = cold.base_phasor_init_phase_q32;

    let mut group_wires = [view.arena0.groups[0]; MAX_SONA_GROUPS];
    group_wires[0..4].copy_from_slice(&view.arena0.groups);
    if let Some(a1) = view.arena1 {
        group_wires[4..6].copy_from_slice(&a1.groups);
    }
    if let Some(a2) = view.arena2 {
        group_wires[6..8].copy_from_slice(&a2.groups);
    }
    if let Some(a3) = view.arena3 {
        group_wires[8..10].copy_from_slice(&a3.groups);
    }
    for (g_idx, gw) in group_wires.iter().enumerate() {
        let (hot_g, bits) = decode_group_wire(gw);
        slab.cold_mut().base_groups[g_idx] = hot_g;
        match g_idx {
            0..=3 => slab.lite.rect_sign_mask_lite |= u16::from(bits) << (g_idx * 4),
            4..=5 => slab.full_delta.rect_sign_mask_full |= bits << ((g_idx - 4) * 4),
            6..=7 => slab.pro_delta.rect_sign_mask_pro |= bits << ((g_idx - 6) * 4),
            _ => slab.ultra_delta.rect_sign_mask_ultra |= bits << ((g_idx - 8) * 4),
        }
    }

    let mut tile_wires = [view.arena0.tiles[0]; MAX_SONA_TILES];
    tile_wires[0..3].copy_from_slice(&view.arena0.tiles);
    if let Some(a1) = view.arena1 {
        tile_wires[3..13].copy_from_slice(&a1.tiles);
    }
    if let Some(a2) = view.arena2 {
        tile_wires[13..23].copy_from_slice(&a2.tiles);
    }
    if let Some(a3) = view.arena3 {
        tile_wires[23..35].copy_from_slice(&a3.tiles);
    }
    for (t_idx, tw) in tile_wires.iter().enumerate() {
        *slab.tile_hot_mut(t_idx) = decode_tile_wire(tw);
    }

    for (i, w) in view
        .arena0
        .varpro_readout_lite
        .stage_latent_mods_fp16
        .iter()
        .enumerate()
    {
        let bits = w.get();
        slab.cold_mut().stage_latent_mods_fp16[i] = match i {
            48..64 => clamp_finite_fp16(bits, -1.5, 1.5, 0.0),
            64..80 => clamp_finite_fp16(bits, -2.0, 2.0, 0.0),
            80..96 => clamp_finite_fp16(bits, -8.0, 8.0, 0.0),
            _ => fp16_sanitized(bits),
        };
    }
    if let Some(a3) = view.arena3 {
        for (i, w) in a3
            .control_bus_ultra_delta
            .stage_latent_ctrl_ultra_fp16
            .iter()
            .enumerate()
        {
            let bits = w.get();
            slab.cold_mut().stage_latent_ctrl_ultra_fp16[i] = match i {
                0..80 => clamp_finite_fp16(bits, -1.5, 1.5, 0.0),
                80..160 => clamp_finite_fp16(bits, -2.0, 2.0, 0.0),
                _ => 0,
            };
        }
        for (i, w) in a3.stage_latent_core_mods_fp16.iter().enumerate() {
            slab.cold_mut().stage_latent_core_mods_fp16[i] = if (i & 127) < 120 {
                fp16_sanitized(w.get())
            } else {
                0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rescale_tpt_svf_g_identity_and_oversample() {
        assert_eq!(rescale_tpt_svf_g_zoh(0.25, 1.0), 0.25);
        let g2 = rescale_tpt_svf_g_zoh(1.0, 0.5);
        assert!(
            (g2 - 0.41421357).abs() < 1.0e-5,
            "tan(pi/8) expected ~0.4142, got {g2}"
        );
        assert_eq!(
            signed_phasor_inc_1x_with_inv(-12_000.0, 48_000.0, crate::math::rcp_nr6_f64(48_000.0)),
            -1_073_741_824
        );
    }
}
