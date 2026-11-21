//! `3 kHz` control-bus target latching, gauge-invariant `z_stage` `Op` modulation laws, and
//! `#[inline(never)]` L1 register hoisting (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use core::f32::consts::PI;

use dnam_kernels::algebraic_unit_rotor_f32;

use crate::control::bus::laws::{
    eval_block_and_group_adaa_lru, eval_global_stage_mods, eval_lane_core_mods,
    eval_op_n_pre_modulated, eval_phasor_inc_core_q32, eval_phasor_inc_vm_q32,
    eval_transport_dry_mix_stage, eval_transport_target_psi,
};
use crate::graph::GraphBackend;
use crate::graph::convolver::kautz::{MAX_KAUTZ_BANKS, retune_kautz_banks_3khz};
use crate::graph::transport::{TRANSPORT_MAX_QUADS, TransportModSlots, latch_quad_nonslew_targets};
use crate::math::f32_to_fp16;
use crate::types::Tier;

use super::hydrate::apply_tier::compute_op_n_pre_hoisted;
use super::layout::SonaHotSlab;
use super::microchunk::core_step::HoistedCoreChunk;

const PSI_SLEW_MAX_RAD: f32 = 0.25;

/// Hoists latched `ControlBus` `(current_stage_params, eff_z_stage)` targets into hot
/// `Op-N[Pre]`, Crossbar gains/phasor increments, `Transport` dry mix and non-slew targets, and
/// per-lane `CoreQuad` modulation arrays (`0` slew step).
#[inline(never)]
pub fn hoist_core_quads_and_readouts<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
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
    let hoisted = compute_op_n_pre_hoisted(gamma_bc, zoh_hold, beta_q, rho_vm);
    let tank_dry_mix_eff = eval_transport_dry_mix_stage(slab.cold().base_tank_dry_mix, t[22]);
    slab.lite.tank_dry_mix_eff_fp16 = f32_to_fp16(tank_dry_mix_eff);

    let (fs_vm, inv_fs_vm, base_f, os2x) = (
        slab.cold().zoh.f_s_vm as f32,
        slab.cold().zoh.inv_f_s_vm(),
        slab.cold().base_phasor_freq_hz,
        slab.os2x(),
    );

    {
        let cold = slab.cold_mut();
        eval_block_and_group_adaa_lru(
            (&t, &z),
            &cold.stage_latent_mods_fp16,
            &cold.stage_latent_ctrl_ultra_fp16,
            is_ultra,
            (&mut cold.quad_adaa_scale, &mut cold.quad_lru_bias),
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

    let mut inc_core = [0_u32; 4];
    for (l, dst) in inc_core.iter_mut().enumerate() {
        let inc_vm = eval_phasor_inc_vm_q32(base_f[l], t[29], fs_vm, inv_fs_vm);
        *dst = eval_phasor_inc_core_q32(inc_vm, os2x) as u32;
    }
    slab.lite.phasors.inc = inc_core;

    let slots = TransportModSlots::from_targets_32(&t);
    let active_q = slab.cold().active_transport_quads.min(TRANSPORT_MAX_QUADS);
    for q in 0..active_q {
        let mut quad = *slab.transport_quad_hot(q);
        {
            let cold = slab.cold_mut();
            let meta = &cold.transport_quad_meta[q];
            let eff = &mut cold.transport_quad_eff[q];
            latch_quad_nonslew_targets(
                &mut quad,
                meta,
                &slots,
                cold.transport_ext.get_mut(q),
                cold.transport_schroeder.get_mut(q),
                eff,
            );
        }
        *slab.transport_quad_hot_mut(q) = quad;
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

    let active_q = slab.cold().active_transport_quads.min(TRANSPORT_MAX_QUADS);
    for q in 0..active_q {
        let mut quad = *slab.transport_quad_hot(q);
        {
            let cold = slab.cold_mut();
            let meta = &cold.transport_quad_meta[q];
            let eff = &mut cold.transport_quad_eff[q];
            let target_psi = eval_transport_target_psi(meta.base_psi, tank_psi_delta);
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
        *slab.transport_quad_hot_mut(q) = quad;
    }

    hoist_core_quads_and_readouts(slab);
}

/// Executes one `3 kHz` boundary control tick (`Kautz` slew, `TransportQuad` Givens `psi` slew,
/// plus hot `CoreQuad`/readout hoisting).
#[inline(never)]
pub fn step_control_tick_3khz<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    step_control_tick_3khz_inner(slab, false);
}

/// Immediately latches current `ControlBus` targets and `eff_z_stage` and snaps `Kautz` and
/// `Transport` slews at init, tier switch, rate re-timing, or deterministic reset.
#[inline(never)]
pub fn latch_immediate_at_init_or_reset<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    if let Some(bus) = slab.cold_mut().ctrl_bus.as_mut() {
        bus.eval_static_and_snap_if_dirty();
        let stage = bus.current_stage_params;
        let zs = bus.eff_z_stage();
        let cold = slab.cold_mut();
        cold.latched_stage_params = stage;
        cold.latched_z_stage = zs;
    }
    step_control_tick_3khz_inner(slab, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::slab::layout::MAX_CORE_QUADS;

    #[test]
    fn test_compute_global_mods_clamps_non_finite_and_phasor_q32() {
        let mut mods = [0_u16; 96];
        mods[8] = f32_to_fp16(4.0);
        let mut t = [0.0_f32; 32];
        t[0] = 2.0;
        t[14] = f32::NAN;
        let z = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let [g_oct, _, _, _, psi] = eval_global_stage_mods(&mods, &t, &z, 4);
        assert_eq!(g_oct, 3.0);
        assert_eq!(psi, 0.0);
        assert_eq!(MAX_CORE_QUADS, 10);

        let inv_48k = crate::math::rcp_nr6_f64(48_000.0);
        let inc_1x = eval_phasor_inc_vm_q32(-12_000.0, 0.0, 48_000.0, inv_48k);
        assert_eq!(inc_1x, -1_073_741_824);
        assert_eq!(eval_phasor_inc_core_q32(inc_1x, true), -536_870_912);
        assert_eq!(eval_transport_dry_mix_stage(0.75, 0.5), 1.25);
    }
}
