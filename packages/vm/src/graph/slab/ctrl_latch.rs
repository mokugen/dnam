//! `3 kHz` control-bus target latching, gauge-invariant `z_stage` `Op` modulation laws, and
//! `#[inline(never)]` L1 register hoisting (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use core::f32::consts::PI;

use dnam_kernels::{OpNPreHoisted, algebraic_unit_rotor_f32, halve_signed_q32_increments};

use crate::control::bus::laws::{
    eval_block_and_group_adaa_lru, eval_global_stage_mods, eval_lane_core_mods,
    eval_op_n_pre_modulated, eval_op_t_we_eff, eval_phasor_eff_freq_hz, eval_tank_target_psi,
};
use crate::graph::GraphBackend;
use crate::graph::cabinet::kautz::{MAX_KAUTZ_BANKS, retune_kautz_banks_3khz};
use crate::graph::tank::{QuadModSlots, latch_quad_nonslew_targets};
use crate::types::Tier;

use super::hydrate::apply_tier::signed_phasor_inc_1x_with_inv;
use super::layout::SonaHotSlab;
use super::microchunk::core_step::HoistedCoreChunk;

const PSI_SLEW_MAX_RAD: f32 = 0.25;

/// Hoists latched `ControlBus` `(current_stage_params, z_stage)` targets into hot `Op-N[Pre]`,
/// Crossbar gains/phasor increments, and per-lane SONA group modulation arrays (`0` slew step).
#[inline(never)]
pub fn hoist_groups_and_readouts<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    let t = slab.cold().latched_stage_params;
    let z = slab.cold().latched_z_stage;
    let is_ultra = matches!(slab.lite.tier, Tier::Ultra);
    let dims = slab.lite.tier.stage_latent_dim();
    let active_lanes = slab.lite.tier.channels();
    let [global_g_oct, global_k_delta, _, _, _] =
        eval_global_stage_mods(&slab.cold().stage_latent_mods_fp16, &t, &z, dims);

    let (base_op_n, rho_vm) = (slab.cold().base_op_n_pre, slab.cold().zoh.rho_vm as f32);
    let (gamma_bc, zoh_hold, beta_q) = eval_op_n_pre_modulated(
        base_op_n.gamma_bc.get(),
        base_op_n.zoh_hold_depth.get(),
        base_op_n.beta_q.get(),
        &t,
    );
    let hoisted = OpNPreHoisted::from_params(gamma_bc, zoh_hold, beta_q, rho_vm);
    let (fs_vm, inv_fs_vm, base_f) = (
        slab.cold().zoh.f_s_vm,
        slab.cold().zoh.inv_f_s_vm(),
        slab.cold().base_phasor_freq_hz,
    );

    {
        let cold = slab.cold_mut();
        eval_block_and_group_adaa_lru(
            (&t, &z),
            &cold.stage_latent_mods_fp16,
            &cold.stage_latent_ctrl_ultra_fp16,
            is_ultra,
            (&mut cold.group_adaa_scale, &mut cold.group_lru_bias),
        );
        let use_core_mods = is_ultra && cold.core_mods_active;
        eval_lane_core_mods(
            (global_g_oct, global_k_delta),
            &z,
            &cold.stage_latent_core_mods_fp16,
            use_core_mods,
            active_lanes,
            (
                &mut cold.lane_g_scale,
                &mut cold.lane_k_delta,
                &mut cold.lane_wf_delta,
            ),
        );
        cold.op_n_pre_hoisted = hoisted;
    }

    slab.lite.op_n_pre.apply_hoisted(hoisted);

    let mut scaled_1x = [0_i32; 4];
    for l in 0..4 {
        let f_eff = eval_phasor_eff_freq_hz(base_f[l], t[29], fs_vm);
        scaled_1x[l] = signed_phasor_inc_1x_with_inv(f_eff, fs_vm, inv_fs_vm);
    }
    let inc_core = if slab.os2x() {
        halve_signed_q32_increments(scaled_1x)
    } else {
        scaled_1x
    };
    slab.lite.phasors.inc = inc_core.map(|v| v as u32);

    let slots = QuadModSlots::from_targets_32(&t);
    let active_q = slab.cold().active_tank_quads;
    for q in 0..active_q {
        let mut quad = *slab.quad_hot(q);
        {
            let cold = slab.cold_mut();
            let meta = &cold.tank_quad_meta[q];
            let eff = &mut cold.tank_quad_eff[q];
            latch_quad_nonslew_targets(
                &mut quad,
                meta,
                &slots,
                cold.tank_schroeder.get_mut(q),
                eff,
            );
        }
        *slab.quad_hot_mut(q) = quad;
    }

    if !slab.cold().hoisted_core_valid {
        let hc = HoistedCoreChunk::<B>::from_slab(slab);
        let cold = slab.cold_mut();
        cold.hoisted_core = hc;
        cold.hoisted_core_valid = true;
    } else {
        let mut hc = slab.cold().hoisted_core;
        hc.refresh_modulated(slab);
        slab.cold_mut().hoisted_core = hc;
    }
}

fn step_control_tick_3khz_inner<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    immediate_snap: bool,
) {
    let t = slab.cold().latched_stage_params;
    let z = slab.cold().latched_z_stage;
    let dims = slab.lite.tier.stage_latent_dim();
    let [_, _, k_oct, k_damp, tank_psi_delta] =
        eval_global_stage_mods(&slab.cold().stage_latent_mods_fp16, &t, &z, dims);

    let active_banks = slab.cold().active_kautz_blocks;
    let mut banks_tmp = [Default::default(); MAX_KAUTZ_BANKS];
    for b in 0..active_banks {
        banks_tmp[b] = *slab.kautz_bank_mut(b);
    }
    let mut slew = slab.cold().kautz_slew;
    retune_kautz_banks_3khz::<B>(
        &mut banks_tmp[..active_banks],
        active_banks,
        &mut slew,
        k_oct,
        k_damp,
        immediate_snap,
    );
    slab.cold_mut().kautz_slew = slew;
    for b in 0..active_banks {
        *slab.kautz_bank_mut(b) = banks_tmp[b];
    }

    let we_eff = eval_op_t_we_eff(t[16]);
    let active_q = slab.cold().active_tank_quads;
    let retained_q =
        (slab.cold().retained_tier.index() + 1).min(crate::graph::tank::TANK_MAX_QUADS);
    for q in 0..retained_q {
        let mut quad = *slab.quad_hot(q);
        {
            let cold = slab.cold_mut();
            quad.update_we_schmitt(we_eff, cold.tank_ext.get_mut(q));
            if q < active_q {
                let meta = &cold.tank_quad_meta[q];
                let eff = &mut cold.tank_quad_eff[q];
                let target_psi = eval_tank_target_psi(meta.base_psi, tank_psi_delta);
                if immediate_snap {
                    quad.current_psi = target_psi;
                } else {
                    let step_psi =
                        (target_psi - quad.current_psi).clamp(-PSI_SLEW_MAX_RAD, PSI_SLEW_MAX_RAD);
                    quad.current_psi = (quad.current_psi + step_psi).clamp(-PI, PI);
                }
                (quad.cos_psi, quad.sin_psi) = algebraic_unit_rotor_f32(quad.current_psi);
                (eff.cos_psi, eff.sin_psi) = (quad.cos_psi, quad.sin_psi);
            }
        }
        *slab.quad_hot_mut(q) = quad;
    }

    hoist_groups_and_readouts(slab);
}

/// Executes one `3 kHz` boundary control tick (`Kautz` slew, `Tank` slew & `WE` Schmitt latch,
/// plus hot group/readout hoisting).
#[inline(never)]
pub fn step_control_tick_3khz<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    step_control_tick_3khz_inner(slab, false);
}

/// Immediately latches current `ControlBus` targets and snaps `Kautz` and `Tank` slews at init,
/// tier switch, rate re-timing, or deterministic reset.
#[inline(never)]
pub fn latch_immediate_at_init_or_reset<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    if let Some(bus) = slab.cold_mut().ctrl_bus.as_mut() {
        bus.eval_static_and_snap_if_dirty();
        let stage = bus.current_stage_params;
        let zs = bus.z_stage();
        let cold = slab.cold_mut();
        cold.latched_stage_params = stage;
        cold.latched_z_stage = zs;
    }
    step_control_tick_3khz_inner(slab, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::slab::MAX_SONA_GROUPS;
    use crate::math::f32_to_fp16;

    #[test]
    fn test_compute_global_mods_clamps_non_finite_and_bounds() {
        let mut mods = [0_u16; 96];
        mods[8] = f32_to_fp16(4.0);
        let mut t = [0.0_f32; 32];
        t[0] = 2.0;
        t[14] = f32::NAN;
        let z = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let [g_oct, _, _, _, psi] = eval_global_stage_mods(&mods, &t, &z, 4);
        assert_eq!(g_oct, 3.0);
        assert_eq!(psi, 0.0);
        assert_eq!(MAX_SONA_GROUPS, 10);
    }
}
