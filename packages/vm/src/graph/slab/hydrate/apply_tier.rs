//! Tier crossbars, ZOH `CoreQuad`/`Op-N[Pre]`/`Kautz`/`FIR`/`ro_pole_48k` rescaling, and
//! `Arena0..3` crossbar hydration (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    MAX_PARALRU_R_F32, OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OpNPreHoisted,
};
use zerocopy::byteorder::little_endian::F32;

use crate::bin::{
    CONVOLVER_KAUTZ_BLOCKS_PER_TIER, DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT,
    DNBN_FLAG_WET_POLARITY_INV, DnamBinView, OpNPreBaseWire, SONA_META_CORE_MODS_PRESENT,
    SONA_META_OS_EXT_PRESENT,
};
use crate::graph::GraphBackend;
use crate::graph::convolver::kautz::{
    KautzPrefixOverride, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS, hydrate_kautz_slice_and_zoh,
};
use crate::graph::oversample::SonaOs2xHot;
use crate::graph::zoh::{ZohRateContext, rescale_paralru_lane};
use crate::math::{
    clamp_finite_f32, const_abs_f64, f32_to_fp16, fp16_to_f32, powf_f64, rescale_tpt_svf_g_zoh,
    sanitize_finite_f32,
};
use crate::types::{Tier, VmError};

use super::super::layout::{
    FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, MAX_CORE_QUADS,
    MAX_SONA_TILES, MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, SonaHotSlab,
};
use super::wire_decode::{
    clamp_finite_fp16, compute_tier_os2x_flags, decode_core_quad_wire, decode_tile_wire,
    fp16_sanitized,
};

const RHO_UNITY_EPS_F64: f64 = 1.0e-12;
const RO_POLE_MAX_EFFECTIVE: f32 = 0.99999;

const LITE_QUAD_END: usize = Tier::Lite.simd_groups();
const FULL_QUAD_END: usize = Tier::Full.simd_groups();
const PRO_QUAD_END: usize = Tier::Pro.simd_groups();
const ULTRA_QUAD_END: usize = Tier::Ultra.simd_groups();

const LITE_TANK_END: usize = Tier::Lite.tank_lines();
const FULL_TANK_END: usize = Tier::Full.tank_lines();
const PRO_TANK_END: usize = Tier::Pro.tank_lines();
const ULTRA_TANK_END: usize = Tier::Ultra.tank_lines();

const LITE_FIR_END: usize = Tier::Lite.early_fir_taps();
const FULL_FIR_END: usize = Tier::Full.early_fir_taps();
const PRO_FIR_END: usize = Tier::Pro.early_fir_taps();
const ULTRA_FIR_END: usize = Tier::Ultra.early_fir_taps();

const LITE_TILE_END: usize = 3;
const FULL_TILE_END: usize = 13;
const PRO_TILE_END: usize = 23;
const ULTRA_TILE_END: usize = MAX_SONA_TILES;

const STAGE_LATENT_DEFAULTS_LEN: usize = 8;
const STAGE_LATENT_GLOBAL_MODS_LEN: usize = 5 * 8;
const STAGE_LATENT_BLOCK_MOD_LEN: usize = NUM_SONA_BLOCKS * 4;
const STAGE_LATENT_QUAD_ADAA_START: usize =
    STAGE_LATENT_DEFAULTS_LEN + STAGE_LATENT_GLOBAL_MODS_LEN;
const STAGE_LATENT_QUAD_ADAA_END: usize = STAGE_LATENT_QUAD_ADAA_START + STAGE_LATENT_BLOCK_MOD_LEN;
const STAGE_LATENT_QUAD_LRU_END: usize = STAGE_LATENT_QUAD_ADAA_END + STAGE_LATENT_BLOCK_MOD_LEN;
const STAGE_LATENT_OP_P_TO_CORE_END: usize = STAGE_LATENT_QUAD_LRU_END + STAGE_LATENT_BLOCK_MOD_LEN;

const ULTRA_LATENT_DIM: usize = Tier::Ultra.stage_latent_dim();
const ULTRA_CTRL_ADAA_END: usize = ULTRA_LATENT_DIM * ULTRA_QUAD_END;
const ULTRA_CTRL_LRU_END: usize = ULTRA_CTRL_ADAA_END + ULTRA_LATENT_DIM * ULTRA_QUAD_END;

const STAGE_CORE_MOD_ROW_STRIDE: usize = 128;
const STAGE_CORE_MOD_ROW_MASK: usize = STAGE_CORE_MOD_ROW_STRIDE - 1;
const STAGE_CORE_MOD_ACTIVE_COLS: usize = MAX_TOTAL_SONA_LANES * 3;

const _: () = assert!(
    LITE_QUAD_END == 4
        && FULL_QUAD_END == 6
        && PRO_QUAD_END == 8
        && ULTRA_QUAD_END == MAX_CORE_QUADS
);
const _: () =
    assert!(LITE_TANK_END == 4 && FULL_TANK_END == 8 && PRO_TANK_END == 12 && ULTRA_TANK_END == 16);
const _: () =
    assert!(LITE_FIR_END == 32 && FULL_FIR_END == 48 && PRO_FIR_END == 96 && ULTRA_FIR_END == 128);
const _: () = assert!(ULTRA_TILE_END == 35);
const _: () = assert!(
    STAGE_LATENT_QUAD_ADAA_START == 48
        && STAGE_LATENT_QUAD_ADAA_END == 64
        && STAGE_LATENT_QUAD_LRU_END == 80
        && STAGE_LATENT_OP_P_TO_CORE_END == 96
);
const _: () = assert!(ULTRA_CTRL_ADAA_END == 80 && ULTRA_CTRL_LRU_END == 160);
const _: () = assert!(STAGE_CORE_MOD_ROW_STRIDE.is_power_of_two());
const _: () = assert!(
    STAGE_CORE_MOD_ACTIVE_COLS == 120
        && STAGE_CORE_MOD_ACTIVE_COLS < STAGE_CORE_MOD_ROW_STRIDE
        && ULTRA_LATENT_DIM * STAGE_CORE_MOD_ROW_STRIDE == 1_024
);

/// Computes hoisted [`OpNPreHoisted`] parameters with ZOH hold-depth exponentiation
/// `z_h_eff = z_h^rho_vm` (`0` when `z_h <= 0`, exact `z_h` when `|rho_vm - 1.0| < 1e-12`).
#[inline]
pub(crate) fn compute_op_n_pre_hoisted(
    gamma_bc: f32,
    zoh_hold_depth: f32,
    beta_q: f32,
    rho_vm: f32,
) -> OpNPreHoisted {
    let z_h = clamp_finite_f32(zoh_hold_depth, 0.0, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, 0.0);
    let rho = if rho_vm.is_finite() && rho_vm > 0.0 {
        rho_vm
    } else {
        1.0
    };
    let z_h_eff = if z_h <= 0.0 {
        0.0
    } else if const_abs_f64(f64::from(rho) - 1.0) < RHO_UNITY_EPS_F64 {
        z_h
    } else {
        (powf_f64(f64::from(z_h), f64::from(rho)) as f32).clamp(0.0, OP_N_PRE_MAX_ZOH_HOLD_DEPTH)
    };
    OpNPreHoisted::from_params_with_zoh_depth(gamma_bc, z_h, z_h_eff, beta_q, rho)
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
    let active_core_quads = tier.simd_groups();
    let active_banks: usize = CONVOLVER_KAUTZ_BLOCKS_PER_TIER[..=t_idx].iter().sum();
    let active_pairs = active_banks * 4;

    let snap = slab.cold().tier_crossbars[t_idx];
    slab.lite.ro_gains_fp16 = snap.ro_gains_fp16;
    slab.lite.phasor_st_cos_fp16 = snap.phasor_st_cos;
    slab.lite.phasor_st_sin_fp16 = snap.phasor_st_sin;
    slab.lite
        .op_p_to_core_lite_fp16
        .copy_from_slice(&snap.op_p_to_core[0..LITE_QUAD_END]);
    slab.full_delta
        .op_p_to_core_full_fp16
        .copy_from_slice(&snap.op_p_to_core[LITE_QUAD_END..FULL_QUAD_END]);
    slab.pro_delta
        .op_p_to_core_pro_fp16
        .copy_from_slice(&snap.op_p_to_core[FULL_QUAD_END..PRO_QUAD_END]);
    slab.ultra_delta
        .op_p_to_core_ultra_fp16
        .copy_from_slice(&snap.op_p_to_core[PRO_QUAD_END..ULTRA_QUAD_END]);
    slab.lite
        .phasor_to_core_lite_fp16
        .copy_from_slice(&snap.phasor_to_core[0..LITE_QUAD_END]);
    slab.full_delta
        .phasor_to_core_full_fp16
        .copy_from_slice(&snap.phasor_to_core[LITE_QUAD_END..FULL_QUAD_END]);
    slab.pro_delta
        .phasor_to_core_pro_fp16
        .copy_from_slice(&snap.phasor_to_core[FULL_QUAD_END..PRO_QUAD_END]);
    slab.ultra_delta
        .phasor_to_core_ultra_fp16
        .copy_from_slice(&snap.phasor_to_core[PRO_QUAD_END..ULTRA_QUAD_END]);
    slab.lite
        .opt_to_lr_lite_fp16
        .copy_from_slice(&snap.opt_to_lr[0..LITE_TANK_END]);
    slab.full_delta
        .opt_to_lr_full_fp16
        .copy_from_slice(&snap.opt_to_lr[LITE_TANK_END..FULL_TANK_END]);
    slab.pro_delta
        .opt_to_lr_pro_fp16
        .copy_from_slice(&snap.opt_to_lr[FULL_TANK_END..PRO_TANK_END]);
    slab.ultra_delta
        .opt_to_lr_ultra_fp16
        .copy_from_slice(&snap.opt_to_lr[PRO_TANK_END..ULTRA_TANK_END]);
    for q in 0..MAX_CORE_QUADS {
        *slab.readout_hot_mut(q) = snap.readouts[q];
    }

    let mut any_crossbar = slab.cold().base_op_p_gain != 0.0
        || slab
            .lite
            .carrier_basis_w_fp16
            .iter()
            .flatten()
            .any(|&w| fp16_to_f32(w) != 0.0);
    for q in 0..active_core_quads {
        any_crossbar |= snap.op_p_to_core[q]
            .iter()
            .chain(snap.phasor_to_core[q].iter())
            .any(|&w| fp16_to_f32(w) != 0.0);
    }
    slab.set_flag(FLAG_CROSSBAR_ACTIVE, any_crossbar);

    for q in 0..MAX_CORE_QUADS {
        let mut h = slab.cold().base_core_quads[q];
        for l in 0..4 {
            h.svf_g[l] = rescale_tpt_svf_g_zoh(h.svf_g[l], zoh.rho_core);
            let kap = fp16_to_f32(h.lru_kappa_fp16[l]);
            let (rel, atk, kap_out) =
                rescale_paralru_lane(h.lru_alpha_rel[l], h.lru_alpha_atk[l], kap, zoh.rho_core);
            h.lru_alpha_rel[l] = rel;
            h.lru_alpha_atk[l] = atk;
            h.lru_kappa_fp16[l] = f32_to_fp16(kap_out);
        }
        *slab.core_quad_hot_mut(q) = h;
    }

    let ro = clamp_finite_f32(slab.cold().ro_pole_48k, 0.0, MAX_PARALRU_R_F32, 0.0);
    slab.lite.hp_lp_pole = if ro <= 0.0 {
        0.0
    } else if const_abs_f64(zoh.rho_core - 1.0) < RHO_UNITY_EPS_F64 {
        ro.clamp(0.0, RO_POLE_MAX_EFFECTIVE)
    } else {
        (powf_f64(f64::from(ro), zoh.rho_core) as f32).clamp(0.0, RO_POLE_MAX_EFFECTIVE)
    };
    slab.lite.tank_dry_mix_eff_fp16 = f32_to_fp16(slab.cold().base_tank_dry_mix);
    let cold = slab.cold_mut();
    cold.core_mods_active = cold.core_mods_present && matches!(tier, Tier::Ultra);
    cold.active_kautz_blocks = active_banks;
    cold.active_transport_quads = t_idx + 1;
    cold.hoisted_core_valid = false;

    let (fir_l, fir_r) = (slab.cold().tier_fir_l[t_idx], slab.cold().tier_fir_r[t_idx]);
    pack_fir_tiles(&mut slab.lite.fir_weights_l, &fir_l[0..LITE_FIR_END]);
    pack_fir_tiles(&mut slab.lite.fir_weights_r, &fir_r[0..LITE_FIR_END]);
    pack_fir_tiles(
        &mut slab.full_delta.fir_weights_l_full,
        &fir_l[LITE_FIR_END..FULL_FIR_END],
    );
    pack_fir_tiles(
        &mut slab.full_delta.fir_weights_r_full,
        &fir_r[LITE_FIR_END..FULL_FIR_END],
    );
    pack_fir_tiles(
        &mut slab.pro_delta.fir_weights_l_pro,
        &fir_l[FULL_FIR_END..PRO_FIR_END],
    );
    pack_fir_tiles(
        &mut slab.pro_delta.fir_weights_r_pro,
        &fir_r[FULL_FIR_END..PRO_FIR_END],
    );
    pack_fir_tiles(
        &mut slab.ultra_delta.fir_weights_l_ultra,
        &fir_l[PRO_FIR_END..ULTRA_FIR_END],
    );
    pack_fir_tiles(
        &mut slab.ultra_delta.fir_weights_r_ultra,
        &fir_r[PRO_FIR_END..ULTRA_FIR_END],
    );

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

pub(super) fn populate_crossbar_and_core_quads<B: GraphBackend>(
    view: &DnamBinView<'_>,
    slab: &mut SonaHotSlab<'_, B>,
) {
    let cb = &view.arena0.crossbar;
    let header_flags = view.header.flags.get();
    let meta_flags = cb.sona_meta_flags.get();
    slab.set_flag(
        FLAG_SINGLE_INPUT,
        (header_flags & DNBN_FLAG_SINGLE_INPUT) != 0,
    );
    slab.set_flag(
        FLAG_WET_POLARITY_INV,
        (header_flags & DNBN_FLAG_WET_POLARITY_INV) != 0,
    );
    slab.set_flag(FLAG_MONO_OUT, (header_flags & DNBN_FLAG_STEREO_OUT) == 0);

    let os_ext = (meta_flags & SONA_META_OS_EXT_PRESENT) != 0;
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
        fp16_to_f32(cb.crossbar_op_gains_fp16[0].get()),
        -4.0,
        4.0,
        0.0,
    );
    let op_t_gain = clamp_finite_f32(
        fp16_to_f32(cb.crossbar_op_gains_fp16[1].get()),
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
    cold.op_n_pre_hoisted = compute_op_n_pre_hoisted(
        op_n_pre_base.gamma_bc.get(),
        op_n_pre_base.zoh_hold_depth.get(),
        op_n_pre_base.beta_q.get(),
        1.0,
    );
    cold.base_op_p_gain = op_p_gain;
    cold.base_op_t_gain = op_t_gain;
    cold.core_mods_present = (meta_flags & SONA_META_CORE_MODS_PRESENT) != 0;
    for l in 0..4 {
        cold.base_phasor_freq_hz[l] = sanitize_finite_f32(cb.phasor_freq_hz[l].get());
        cold.base_phasor_init_phase_q32[l] = cb.phasor_init_phase_q32[l].get();
    }
    slab.lite.phasors.phase = cold.base_phasor_init_phase_q32;

    let mut quad_wires = [view.arena0.core_quads[0]; MAX_CORE_QUADS];
    quad_wires[0..LITE_QUAD_END].copy_from_slice(&view.arena0.core_quads);
    if let Some(a1) = view.arena1 {
        quad_wires[LITE_QUAD_END..FULL_QUAD_END].copy_from_slice(&a1.core_quads);
    }
    if let Some(a2) = view.arena2 {
        quad_wires[FULL_QUAD_END..PRO_QUAD_END].copy_from_slice(&a2.core_quads);
    }
    if let Some(a3) = view.arena3 {
        quad_wires[PRO_QUAD_END..ULTRA_QUAD_END].copy_from_slice(&a3.core_quads);
    }
    for (q_idx, qw) in quad_wires.iter().enumerate() {
        let (hot_q, ext_q, bits) = decode_core_quad_wire(qw);
        let cold_mut = slab.cold_mut();
        cold_mut.base_core_quads[q_idx] = hot_q;
        cold_mut.base_core_ext[q_idx] = ext_q;
        if q_idx < LITE_QUAD_END {
            slab.lite.rect_sign_mask_lite |= u16::from(bits) << (q_idx * 4);
        } else if q_idx < FULL_QUAD_END {
            slab.full_delta.rect_sign_mask_full |= bits << ((q_idx - LITE_QUAD_END) * 4);
        } else if q_idx < PRO_QUAD_END {
            slab.pro_delta.rect_sign_mask_pro |= bits << ((q_idx - FULL_QUAD_END) * 4);
        } else {
            slab.ultra_delta.rect_sign_mask_ultra |= bits << ((q_idx - PRO_QUAD_END) * 4);
        }
    }

    let mut tile_wires = [view.arena0.tiles[0]; MAX_SONA_TILES];
    tile_wires[0..LITE_TILE_END].copy_from_slice(&view.arena0.tiles);
    if let Some(a1) = view.arena1 {
        tile_wires[LITE_TILE_END..FULL_TILE_END].copy_from_slice(&a1.tiles);
    }
    if let Some(a2) = view.arena2 {
        tile_wires[FULL_TILE_END..PRO_TILE_END].copy_from_slice(&a2.tiles);
    }
    if let Some(a3) = view.arena3 {
        tile_wires[PRO_TILE_END..ULTRA_TILE_END].copy_from_slice(&a3.tiles);
    }
    for (t_idx, tw) in tile_wires.iter().enumerate() {
        *slab.tile_hot_mut(t_idx) = decode_tile_wire(tw);
    }

    for (i, w) in view
        .arena0
        .linear_readout_lite
        .stage_latent_mods_fp16
        .iter()
        .enumerate()
    {
        let bits = w.get();
        slab.cold_mut().stage_latent_mods_fp16[i] = match i {
            STAGE_LATENT_QUAD_ADAA_START..STAGE_LATENT_QUAD_ADAA_END => {
                clamp_finite_fp16(bits, -1.5, 1.5, 0.0)
            }
            STAGE_LATENT_QUAD_ADAA_END..STAGE_LATENT_QUAD_LRU_END => {
                clamp_finite_fp16(bits, -2.0, 2.0, 0.0)
            }
            STAGE_LATENT_QUAD_LRU_END..STAGE_LATENT_OP_P_TO_CORE_END => {
                clamp_finite_fp16(bits, -8.0, 8.0, 0.0)
            }
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
                0..ULTRA_CTRL_ADAA_END => clamp_finite_fp16(bits, -1.5, 1.5, 0.0),
                ULTRA_CTRL_ADAA_END..ULTRA_CTRL_LRU_END => clamp_finite_fp16(bits, -2.0, 2.0, 0.0),
                _ => 0,
            };
        }
        for (i, w) in a3.stage_latent_core_mods_fp16.iter().enumerate() {
            slab.cold_mut().stage_latent_core_mods_fp16[i] =
                if (i & STAGE_CORE_MOD_ROW_MASK) < STAGE_CORE_MOD_ACTIVE_COLS {
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
    fn test_rescale_tpt_svf_and_op_n_pre_zoh_scaling() {
        assert_eq!(rescale_tpt_svf_g_zoh(0.25, 1.0), 0.25);
        let g2 = rescale_tpt_svf_g_zoh(1.0, 0.5);
        assert!(
            (g2 - 0.41421357).abs() < 1.0e-5,
            "tan(pi/8) expected ~0.4142, got {g2}"
        );

        let unity = compute_op_n_pre_hoisted(0.75, 0.5, 4.0, 1.0);
        assert_eq!(unity.zoh_hold_depth_eff, 0.5);
        let up2x = compute_op_n_pre_hoisted(0.75, 0.5, 4.0, 0.5);
        assert!((up2x.zoh_hold_depth_eff - core::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-5);
        let zero = compute_op_n_pre_hoisted(0.75, 0.0, 4.0, 0.5);
        assert_eq!((zero.zoh_hold_depth_eff, zero.inc_q32), (0.0, 0));
    }
}
