//! Runtime tier switching, host sample-rate re-timing, `4x` bounce toggling, deterministic zero-DC
//! reset, and dual-latch `Op-T` host write locking (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    QUAD_FLAG_HAS_EXT, compute_t1_dc_blocker_coeffs_clamped, halve_signed_q32_increments,
    interpolate_hilbert_h1_alphas_f64,
};

use crate::bin::arena_wire::TANK_FLAG_SCHROEDER_ENABLE;
use crate::control::bus::laws::eval_phasor_eff_freq_hz;
use crate::graph::GraphBackend;
use crate::graph::cabinet::high_sr::CabinetHighSrFirExt;
use crate::graph::tank::{
    QuadEffectiveTargets, QuadHydrateConfig, compute_quad_ext_requirements,
    compute_schroeder_requirements,
};
use crate::graph::zoh::ZohRateContext;
use crate::types::{Tier, VmError};

use super::super::ctrl_latch::latch_immediate_at_init_or_reset;
use super::super::layout::{
    FLAG_COLD_FIR_EXT, FLAG_OS2X, FLAG_XFADE_ACTIVE, MAX_SONA_GROUPS, SonaHotSlab,
    TIER_SWITCH_XFADE_SAMPLES,
};
use super::apply_tier::{apply_tier_weights_and_rescale, signed_phasor_inc_1x_with_inv};

pub(super) fn setup_phasor_increments<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    zoh: &ZohRateContext,
) {
    let inv_fs = zoh.inv_f_s_vm();
    let mut inc_1x = [0_i32; 4];
    for (i, dst) in inc_1x.iter_mut().enumerate() {
        let f_eff = eval_phasor_eff_freq_hz(slab.cold().base_phasor_freq_hz[i], 0.0, zoh.f_s_vm);
        *dst = signed_phasor_inc_1x_with_inv(f_eff, zoh.f_s_vm, inv_fs);
    }
    let inc_core = if zoh.os2x_active {
        halve_signed_q32_increments(inc_1x)
    } else {
        inc_1x
    };
    slab.lite.phasors.inc = inc_core.map(|v| v as u32);
}

fn zero_inactive_tier_states<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    from_tier: Tier,
    to_tier: Tier,
) {
    let from_g = from_tier.simd_groups();
    let to_g = to_tier.simd_groups();
    if to_g > from_g {
        for g in from_g..to_g {
            *slab.state_mut(g) = dnam_kernels::SonaBlockStateStereo8::zero();
        }
    }
    let bank_counts = [2_usize, 3, 6, 8];
    let from_b = bank_counts[from_tier.index()];
    let to_b = bank_counts[to_tier.index()];
    if to_b > from_b {
        for b in from_b..to_b {
            slab.kautz_bank_mut(b).reset();
        }
    }
    if to_tier.index() < from_tier.index() {
        let to_taps = to_tier.early_fir_taps();
        if to_taps <= 32 {
            slab.full_delta.fir_hist_l_full.fill(0.0);
            slab.full_delta.fir_hist_r_full.fill(0.0);
        }
        if to_taps <= 48 {
            slab.pro_delta.fir_hist_l_pro.fill(0.0);
            slab.pro_delta.fir_hist_r_pro.fill(0.0);
        }
        if to_taps <= 96 {
            slab.ultra_delta.fir_hist_l_ultra.fill(0.0);
            slab.ultra_delta.fir_hist_r_ultra.fill(0.0);
        }
    }
}

/// Switches active tier in-place (`0` RT allocs, `0` core reset), preserving `phasors.phase`
/// and starting the deterministic `48`-frame output crossfade when switching across tiers.
pub fn switch_tier<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    to_tier: Tier,
) -> Result<(), VmError> {
    if to_tier.index() > slab.cold().retained_tier.index() {
        return Err(VmError::InvalidTier);
    }
    let from_tier = slab.lite.tier;
    if from_tier == to_tier {
        return Ok(());
    }
    let prev_zoh = slab.cold().zoh;
    let tier_os2x = slab.cold().tier_os2x[to_tier.index()];
    let zoh = ZohRateContext::compute(
        prev_zoh.reference_rate_hz,
        prev_zoh.host_rate_hz,
        to_tier,
        tier_os2x,
        prev_zoh.is_bounce_4x,
    )?;

    let was_os2x = slab.os2x();
    if !was_os2x && zoh.os2x_active {
        let last_in = slab.cold().last_in;
        let last_s2 = slab.cold().last_s2_dir;
        let last_iq = slab.cold().last_op_p_iq_lr;
        let last_feat = slab.cold().last_op_p_feat;
        slab.os2x_hot.warm_seed(last_in, last_s2);
        slab.os2x_hot.op_p_prev_iq_lr = last_iq;
        slab.os2x_hot.op_p_prev_feat = last_feat;
    }
    zero_inactive_tier_states(slab, from_tier, to_tier);

    slab.set_flag(FLAG_OS2X, zoh.os2x_active);
    slab.lite.tier = to_tier;
    slab.cold_mut().zoh = zoh;
    if let Some(bus) = slab.cold_mut().ctrl_bus.as_mut() {
        bus.tier = to_tier;
        bus.is_dirty = true;
    }

    setup_phasor_increments(slab, &zoh);
    apply_tier_weights_and_rescale(slab, to_tier, &zoh)?;
    if slab.cold_fir_ext() {
        let active_taps = to_tier.early_fir_taps();
        let (fl, fr) = (
            slab.cold().tier_fir_l[to_tier.index()],
            slab.cold().tier_fir_r[to_tier.index()],
        );
        let cold = slab.cold_mut();
        cold.cabinet_fir_ext.prepare(
            &fl[..active_taps],
            &fr[..active_taps],
            zoh.f_s_vm,
            zoh.is_bounce_4x,
        )?;
    }

    slab.cold_mut().xfade_rem = TIER_SWITCH_XFADE_SAMPLES;
    slab.set_flag(FLAG_XFADE_ACTIVE, true);
    latch_immediate_at_init_or_reset(slab);
    Ok(())
}

fn retime_with_bounce<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    new_host_rate_hz: f32,
    new_offline: bool,
) -> Result<(), VmError> {
    let tier = slab.lite.tier;
    let tier_os2x = slab.cold().tier_os2x[tier.index()];
    let ref_hz = slab.cold().zoh.reference_rate_hz;
    let zoh = ZohRateContext::compute(ref_hz, new_host_rate_hz, tier, tier_os2x, new_offline)?;

    let req_fir_len = CabinetHighSrFirExt::required_f32_len(zoh.f_s_vm, new_offline);
    if req_fir_len > slab.cold().cabinet_fir_ext.backing_f32_capacity() {
        return Err(VmError::InsufficientBackingCapacity);
    }

    let wires = slab.cold().wire_tank_quads;
    let (decl_max, pow2, dc5) = (
        slab.cold().declared_max_delay_48k,
        slab.cold().ext_ring_pow2_frames,
        slab.cold().dc_blocker_pole_5hz,
    );
    let active_q = slab.cold().retained_tier.index() + 1;
    {
        let cold = slab.cold();
        for (q, wire) in wires.iter().enumerate().take(active_q) {
            let (has_ext, cap) = compute_quad_ext_requirements(wire, q, zoh.rho_vm, decl_max, pow2);
            if has_ext && cold.tank_ext[q].backing_frame_capacity() < cap {
                return Err(VmError::InsufficientBackingCapacity);
            }
            if (wire.flags & TANK_FLAG_SCHROEDER_ENABLE) != 0 {
                let (_, _, _, sch_req) = compute_schroeder_requirements(wire, zoh.rho_vm);
                if cold.tank_schroeder[q].backing_f32_capacity() < sch_req {
                    return Err(VmError::InsufficientBackingCapacity);
                }
            }
        }
    }

    let bounce_changed = slab.cold().offline_bounce != new_offline;
    if bounce_changed {
        let cold = slab.cold_mut();
        cold.offline_bounce = new_offline;
        cold.bounce4x.reset();
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.set_offline_bounce(new_offline);
        }
        slab.os2x_hot.reset();
    }

    slab.set_flag(FLAG_OS2X, zoh.os2x_active);
    let use_cold_fir = req_fir_len > 0;
    slab.set_flag(FLAG_COLD_FIR_EXT, use_cold_fir);

    let (ai, aq) = interpolate_hilbert_h1_alphas_f64(zoh.f_s_vm);
    let rem_hist = slab.cold().rem_hist;
    slab.lite
        .hilbert
        .remap_section_states_from_outputs(ai, aq, &rem_hist);
    let (dc_pole, _) = compute_t1_dc_blocker_coeffs_clamped(zoh.f_s_vm);
    slab.lite.t1_dc.set_pole(dc_pole);

    {
        let cb = slab.cold().wire_crossbar;
        let cold = slab.cold_mut();
        cold.zoh = zoh;
        cold.router.reconfigure_sample_rate(&cb, &zoh);
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.retime_sample_rate(zoh.f_s_vm as f32);
        }
        let active_taps = tier.early_fir_taps();
        let (fl, fr) = (cold.tier_fir_l[tier.index()], cold.tier_fir_r[tier.index()]);
        cold.cabinet_fir_ext.prepare(
            &fl[..active_taps],
            &fr[..active_taps],
            zoh.f_s_vm,
            new_offline,
        )?;
    }

    setup_phasor_increments(slab, &zoh);
    apply_tier_weights_and_rescale(slab, tier, &zoh)?;

    for (q, wire) in wires.iter().enumerate().take(active_q) {
        let cfg = QuadHydrateConfig {
            quad_idx: q,
            fs_vm: zoh.f_s_vm,
            rho_vm: zoh.rho_vm,
            declared_max_delay_48k: decl_max,
            ext_ring_pow2_frames: pow2,
            dc_blocker_pole_5hz: dc5,
        };
        let mut q_hot = *slab.quad_hot(q);
        let inline_opt = if q == 0 {
            Some(&mut slab.lite.tank0_ring)
        } else {
            None
        };
        {
            let cold = &mut slab.cold.as_mut_slice()[0];
            let meta_q = q_hot.reconfigure_sample_rate(
                wire,
                &cfg,
                Some(&mut cold.tank_ext[q]),
                inline_opt,
                Some(&mut cold.tank_schroeder[q]),
            )?;
            if (q_hot.flags & QUAD_FLAG_HAS_EXT) != 0 {
                cold.tank_ext[q].reconfigure_cap(meta_q.cap)?;
            }
            cold.tank_quad_meta[q] = meta_q;
            cold.tank_quad_eff[q] = QuadEffectiveTargets::from_quad_and_meta(&q_hot, &meta_q);
        }
        *slab.quad_hot_mut(q) = q_hot;
    }

    latch_immediate_at_init_or_reset(slab);
    Ok(())
}

/// Re-times all ZOH poles, Hilbert H1 sections (`remap_section_states_from_outputs`), `T1` DC
/// blockers, Cabinet FIR, Kautz, Tank, Router, and `ControlBus` upon a host rate change.
pub fn retime_sample_rate<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    new_host_rate_hz: f32,
) -> Result<(), VmError> {
    let offline = slab.cold().offline_bounce;
    retime_with_bounce(slab, new_host_rate_hz, offline)
}

/// Switches between live `1x` execution and deterministic `4x` offline bounce oversampling.
pub fn set_offline_bounce<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    offline: bool,
) -> Result<(), VmError> {
    if slab.cold().offline_bounce == offline {
        return Ok(());
    }
    let host_hz = slab.cold().zoh.host_rate_hz;
    retime_with_bounce(slab, host_hz, offline)
}

/// Restores deterministic resting zero-DC equilibrium while strictly preserving bit 2
/// (`QUAD_FLAG_HOST_WRITE_LOCK`) across all spatial tank quads.
pub fn reset_to_dc_equilibrium<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    for g in 0..MAX_SONA_GROUPS {
        *slab.state_mut(g) = dnam_kernels::SonaBlockStateStereo8::zero();
    }
    for b in 0..8 {
        slab.kautz_bank_mut(b).reset();
    }
    slab.lite.fir_hist_l.fill(0.0);
    slab.lite.fir_hist_r.fill(0.0);
    slab.full_delta.fir_hist_l_full.fill(0.0);
    slab.full_delta.fir_hist_r_full.fill(0.0);
    slab.pro_delta.fir_hist_l_pro.fill(0.0);
    slab.pro_delta.fir_hist_r_pro.fill(0.0);
    slab.ultra_delta.fir_hist_l_ultra.fill(0.0);
    slab.ultra_delta.fir_hist_r_ultra.fill(0.0);
    slab.os2x_hot.reset();
    slab.lite.hilbert.reset();
    slab.lite.t1.reset();
    slab.lite.t1_dc.reset();
    slab.lite.op_n_pre.held = [0.0; 2];
    slab.lite.op_n_pre.phase_q32 = 0;
    slab.lite.hp_lp_state = [0.0; 2];
    slab.lite.sample_tick = 0;
    slab.lite.phasors.phase = slab.cold().base_phasor_init_phase_q32;
    slab.set_flag(FLAG_XFADE_ACTIVE, false);
    slab.lite.tank0_ring.fill([0; 4]);

    let active_q = slab.cold().retained_tier.index() + 1;
    for q in 0..active_q {
        let mut q_hot = *slab.quad_hot(q);
        {
            let cold = slab.cold_mut();
            q_hot.reset(
                Some(&mut cold.tank_ext[q]),
                None,
                Some(&mut cold.tank_schroeder[q]),
            );
        }
        *slab.quad_hot_mut(q) = q_hot;
    }

    let (mut gate_env, mut gate_gain) = (0.0_f32, 1.0_f32);
    {
        let cold = slab.cold_mut();
        cold.router.reset(&mut gate_env, &mut gate_gain);
        cold.cabinet_fir_ext.reset();
        cold.kautz_slew.reset_slew();
        cold.bounce4x.reset();
        cold.rem_hist.reset();
        cold.xfade_rem = 0;
        cold.xfade_prev_out = [0.0; 2];
        cold.last_in = [0.0; 2];
        cold.last_s2_dir = [0.0; 4];
        cold.last_op_p_iq_lr = [0.0; 4];
        cold.last_op_p_feat = [0.0; 4];
        cold.saturation_flags = 0;
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.reset();
        }
    }
    slab.lite.gate_env = gate_env;
    slab.lite.gate_gain = gate_gain;
    latch_immediate_at_init_or_reset(slab);
}

/// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across all active tank quads without
/// touching bit 0 (`QUAD_FLAG_WE_HOLD_ACTIVE`).
pub fn set_op_t_write_enable<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>, locked: bool) {
    let retained_q =
        (slab.cold().retained_tier.index() + 1).min(crate::graph::tank::TANK_MAX_QUADS);
    for q in 0..retained_q {
        let mut q_hot = *slab.quad_hot(q);
        {
            let cold = slab.cold_mut();
            q_hot.set_host_write_lock(locked, Some(&mut cold.tank_ext[q]));
        }
        *slab.quad_hot_mut(q) = q_hot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_constants_and_xfade_budget() {
        assert_eq!(TIER_SWITCH_XFADE_SAMPLES, 48);
        assert_eq!(MAX_SONA_GROUPS, 10);
    }
}
