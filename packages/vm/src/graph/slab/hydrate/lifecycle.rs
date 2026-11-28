//! Runtime tier switching, host sample-rate re-timing, `4x` bounce toggling, deterministic zero-DC
//! reset, and dual-latch `Op-T` host write locking (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    QUAD_FLAG_HAS_EXT, compute_t1_dc_blocker_coeffs_clamped, interpolate_hilbert_h1_alphas_f64,
};

use crate::bin::CONVOLVER_KAUTZ_BLOCKS_PER_TIER;
use crate::control::bus::laws::{
    eval_op_t_freeze_mix_eff, eval_phasor_inc_core_multirate_q32, eval_phasor_inc_vm_q32,
};
use crate::graph::GraphBackend;
use crate::graph::convolver::high_sr::ConvolverHighSrFirExt;
use crate::graph::convolver::kautz::MAX_KAUTZ_BANKS;
use crate::graph::oversample::RenderMode;
use crate::graph::sensory::{SensoryParamsHot, SensoryQuadCold};
use crate::graph::transport::{
    QuadEffectiveTargets, QuadHydrateConfig, TRANSPORT_MAX_QUADS, TransportModSlots,
    compute_quad_ext_requirements, compute_schroeder_requirements, latch_quad_nonslew_targets,
};
use crate::graph::zoh::ZohRateContext;
use crate::types::{Tier, VmError};

use super::super::ctrl_latch::latch_immediate_at_init_or_reset;
use super::super::layout::{
    FLAG_COLD_FIR_EXT, FLAG_OS2X, FLAG_XFADE_ACTIVE, MAX_CORE_QUADS, SonaHotSlab,
    TIER_SWITCH_XFADE_SAMPLES,
};
use super::apply_tier::apply_tier_weights_and_rescale;

#[inline]
const fn cum_kautz_banks(tier: Tier) -> usize {
    let idx = tier.index();
    let mut sum = 0_usize;
    let mut i = 0_usize;
    while i <= idx {
        sum += CONVOLVER_KAUTZ_BLOCKS_PER_TIER[i];
        i += 1;
    }
    sum
}

pub(super) fn setup_phasor_increments<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    zoh: &ZohRateContext,
) {
    let inv_fs = zoh.inv_f_s_vm();
    let fs_vm_f32 = zoh.f_s_vm as f32;
    let m_core = zoh.m_core;
    let base_freqs = slab.cold().base_phasor_freq_hz;
    let mut inc_core = [0_u32; 4];
    for (i, dst) in inc_core.iter_mut().enumerate() {
        let inc_vm = eval_phasor_inc_vm_q32(base_freqs[i], 0.0, fs_vm_f32, inv_fs);
        *dst = eval_phasor_inc_core_multirate_q32(inc_vm, m_core) as u32;
    }
    slab.lite.phasors.inc = inc_core;
}

fn zero_inactive_tier_states<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    from_tier: Tier,
    to_tier: Tier,
) {
    let from_q = from_tier.simd_groups();
    let to_q = to_tier.simd_groups();
    if to_q > from_q {
        for q in from_q..to_q {
            *slab.state_mut(q) = dnam_kernels::SonaBlockStateStereo8::zero();
        }
    }
    let from_b = cum_kautz_banks(from_tier);
    let to_b = cum_kautz_banks(to_tier);
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
    let zoh = ZohRateContext::compute(
        prev_zoh.reference_rate_hz,
        prev_zoh.host_rate_hz,
        prev_zoh.render_mode,
    )?;

    zero_inactive_tier_states(slab, from_tier, to_tier);

    slab.set_flag(FLAG_OS2X, zoh.m_core >= 2);
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
        cold.convolver_fir_ext.prepare(
            &fl[..active_taps],
            &fr[..active_taps],
            zoh.f_s_vm,
            false,
        )?;
    }

    slab.cold_mut().xfade_rem = TIER_SWITCH_XFADE_SAMPLES;
    slab.set_flag(FLAG_XFADE_ACTIVE, true);
    latch_immediate_at_init_or_reset(slab);
    Ok(())
}

fn retime_with_mode<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    new_host_rate_hz: f32,
    new_mode: RenderMode,
) -> Result<(), VmError> {
    let tier = slab.lite.tier;
    let ref_hz = slab.cold().zoh.reference_rate_hz;
    let zoh = ZohRateContext::compute(ref_hz, new_host_rate_hz, new_mode)?;

    let req_fir_len = ConvolverHighSrFirExt::required_f32_len(zoh.f_s_vm, false);
    if req_fir_len > slab.cold().convolver_fir_ext.backing_f32_capacity() {
        return Err(VmError::InsufficientBackingCapacity);
    }

    let wires = slab.cold().wire_transport_quads;
    let (decl_max, pow2, dc5) = (
        slab.cold().declared_max_delay_48k,
        slab.cold().ext_ring_pow2_frames,
        slab.cold().dc_blocker_pole_5hz,
    );
    let retained_q = (slab.cold().retained_tier.index() + 1).min(TRANSPORT_MAX_QUADS);
    {
        let cold = slab.cold();
        for (q, wire) in wires.iter().enumerate().take(retained_q) {
            let (has_ext, cap) = compute_quad_ext_requirements(wire, q, zoh.rho_vm, decl_max, pow2);
            if has_ext && cold.transport_ext[q].backing_frame_capacity() < cap {
                return Err(VmError::InsufficientBackingCapacity);
            }
            let (_, _, _, sch_req) = compute_schroeder_requirements(wire, zoh.rho_vm);
            if sch_req > 0 && cold.transport_schroeder[q].backing_f32_capacity() < sch_req {
                return Err(VmError::InsufficientBackingCapacity);
            }
        }
    }

    let mode_changed = slab.cold().render_mode != new_mode;
    if mode_changed {
        let cold = slab.cold_mut();
        cold.render_mode = new_mode;
        cold.master_os.reset();
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.set_offline_bounce(matches!(new_mode, RenderMode::Master));
        }
        slab.lite.live_os.reset();
        slab.lite.op_p_prev_iq_lr = [0.0; 4];
        slab.lite.op_p_prev_feat = [0.0; 4];
    }

    slab.set_flag(FLAG_OS2X, zoh.m_core >= 2);
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
        let sensory_cold = SensoryQuadCold::from_crossbar_wire(&cb, zoh.rho_vm);
        let cold = slab.cold_mut();
        cold.zoh = zoh;
        cold.sensory_cold = sensory_cold;
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.retime_sample_rate(zoh.f_s_vm as f32);
        }
        let active_taps = tier.early_fir_taps();
        let (fl, fr) = (cold.tier_fir_l[tier.index()], cold.tier_fir_r[tier.index()]);
        cold.convolver_fir_ext.prepare(
            &fl[..active_taps],
            &fr[..active_taps],
            zoh.f_s_vm,
            false,
        )?;
        slab.sensory_params = SensoryParamsHot::from_cold(&sensory_cold);
    }

    setup_phasor_increments(slab, &zoh);
    apply_tier_weights_and_rescale(slab, tier, &zoh)?;

    for (q, wire) in wires.iter().enumerate().take(retained_q) {
        let cfg = QuadHydrateConfig {
            quad_idx: q,
            fs_vm: zoh.f_s_vm,
            rho_vm: zoh.rho_vm,
            declared_max_delay_48k: decl_max,
            ext_ring_pow2_frames: pow2,
            dc_blocker_pole_5hz: dc5,
        };
        let mut q_hot = *slab.transport_quad_hot(q);
        let inline_opt = if q == 0 {
            Some(&mut slab.lite.transport0_ring)
        } else {
            None
        };
        {
            let cold = &mut slab.cold.as_mut_slice()[0];
            let meta_q = q_hot.reconfigure_sample_rate(
                wire,
                &cfg,
                Some(&mut cold.transport_ext[q]),
                inline_opt,
                Some(&mut cold.transport_schroeder[q]),
            )?;
            if (q_hot.flags & QUAD_FLAG_HAS_EXT) != 0 {
                cold.transport_ext[q].reconfigure_cap(meta_q.cap)?;
            }
            cold.transport_quad_meta[q] = meta_q;
            cold.transport_quad_eff[q] = QuadEffectiveTargets::from_quad_and_meta(&q_hot, &meta_q);
        }
        *slab.transport_quad_hot_mut(q) = q_hot;
    }

    latch_immediate_at_init_or_reset(slab);
    Ok(())
}

/// Re-times all ZOH poles, `SensoryQuad`, Hilbert H1 sections (`remap_section_states_from_outputs`),
/// `T1` DC blockers, `ModalConvolver` FIR/Kautz, `Transport`, and `ControlBus` upon a host rate change.
pub fn retime_sample_rate<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    new_host_rate_hz: f32,
) -> Result<(), VmError> {
    let mode = slab.cold().render_mode;
    retime_with_mode(slab, new_host_rate_hz, mode)
}

/// Switches Core oversampling [`RenderMode`] between `Live` (`2x`) and `Master` (`4x`/`2x` `Box2`),
/// resetting recursive audio/oversampler states to deterministic zero-DC equilibrium.
pub fn set_render_mode<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    mode: RenderMode,
) -> Result<(), VmError> {
    if slab.cold().render_mode == mode {
        return Ok(());
    }
    let host_hz = slab.cold().zoh.host_rate_hz;
    retime_with_mode(slab, host_hz, mode)?;
    reset_to_dc_equilibrium(slab);
    Ok(())
}

/// Adapter switching between [`RenderMode::Live`] (`offline = false`) and [`RenderMode::Master`].
pub fn set_offline_bounce<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    offline: bool,
) -> Result<(), VmError> {
    set_render_mode(
        slab,
        if offline {
            RenderMode::Master
        } else {
            RenderMode::Live
        },
    )
}

/// Restores deterministic resting zero-DC equilibrium while strictly preserving bit 2
/// (`QUAD_FLAG_HOST_WRITE_LOCK`) across all `TransportQuad`s.
pub fn reset_to_dc_equilibrium<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>) {
    for q in 0..MAX_CORE_QUADS {
        *slab.state_mut(q) = dnam_kernels::SonaBlockStateStereo8::zero();
    }
    for b in 0..MAX_KAUTZ_BANKS {
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
    slab.lite.live_os.reset();
    slab.lite.op_p_prev_iq_lr = [0.0; 4];
    slab.lite.op_p_prev_feat = [0.0; 4];
    slab.lite.sensory_quad.reset();
    slab.lite.hilbert.reset();
    slab.lite.t1.reset();
    slab.lite.t1_dc.reset();
    slab.lite.op_n_pre.held = [0.0; 2];
    slab.lite.op_n_pre.phase_q32 = 0;
    slab.lite.hp_lp_state = [0.0; 2];
    slab.lite.sample_tick = 0;
    slab.lite.phasors.phase = slab.cold().base_phasor_init_phase_q32;
    slab.set_flag(FLAG_XFADE_ACTIVE, false);
    slab.lite.transport0_ring.fill([0; 4]);

    let retained_q = (slab.cold().retained_tier.index() + 1).min(TRANSPORT_MAX_QUADS);
    for q in 0..retained_q {
        let mut q_hot = *slab.transport_quad_hot(q);
        {
            let cold = slab.cold_mut();
            q_hot.reset(
                Some(&mut cold.transport_ext[q]),
                None,
                Some(&mut cold.transport_schroeder[q]),
            );
        }
        *slab.transport_quad_hot_mut(q) = q_hot;
    }

    {
        let cold = slab.cold_mut();
        cold.convolver_fir_ext.reset();
        cold.kautz_slew.reset_slew();
        cold.master_os.reset();
        cold.rem_hist.reset();
        cold.xfade_rem = 0;
        cold.xfade_prev_out = [0.0; 2];
        cold.xfade_prev_core = [0.0; 2];
        cold.xfade_prev_wet = [0.0; 2];
        cold.last_in = [0.0; 2];
        cold.last_s2_dir = [0.0; 4];
        cold.last_op_p_iq_lr = [0.0; 4];
        cold.last_op_p_feat = [0.0; 4];
        cold.last_core_lr = [0.0; 2];
        cold.last_transport_wet_lr = [0.0; 2];
        cold.saturation_flags = 0;
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.reset();
        }
    }
    latch_immediate_at_init_or_reset(slab);
}

/// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across retained transport quads and
/// immediately re-latches active `QuadEffectiveTargets` and `PartitionedSchroederBank` gains
/// (`0-ULP` write lock / continuous freeze-hold coordination without control-boundary race).
pub fn set_op_t_write_enable<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>, locked: bool) {
    let slots = TransportModSlots::from_targets_32(&slab.cold().latched_stage_params);
    let active_q = slab.cold().active_transport_quads.min(TRANSPORT_MAX_QUADS);
    let retained_q = (slab.cold().retained_tier.index() + 1).min(TRANSPORT_MAX_QUADS);
    for q in 0..retained_q {
        let mut q_hot = *slab.transport_quad_hot(q);
        {
            let cold = slab.cold_mut();
            let meta = &cold.transport_quad_meta[q];
            let mu_freeze_eff =
                eval_op_t_freeze_mix_eff(meta.base_freeze_mix, slots.op_t_freeze_mix_delta, locked);
            q_hot.set_host_write_lock(locked, cold.transport_ext.get_mut(q), mu_freeze_eff);
            if q < active_q {
                let eff = &mut cold.transport_quad_eff[q];
                latch_quad_nonslew_targets(
                    &mut q_hot,
                    meta,
                    &slots,
                    cold.transport_ext.get_mut(q),
                    cold.transport_schroeder.get_mut(q),
                    eff,
                );
            }
        }
        *slab.transport_quad_hot_mut(q) = q_hot;
    }
}

/// Alias matching `SonaVm::set_quad_write_enable`, updating `QUAD_FLAG_HOST_WRITE_LOCK` and
/// immediately re-latching active transport targets.
#[inline]
pub fn set_quad_write_enable<B: GraphBackend>(slab: &mut SonaHotSlab<'_, B>, locked: bool) {
    set_op_t_write_enable(slab, locked);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::QUAD_FLAG_HOST_WRITE_LOCK;

    /// verify[dsp.op-m.kautz.tier-allocation]
    #[test]
    fn test_lifecycle_constants_and_kautz_cum_banks() {
        assert_eq!(TIER_SWITCH_XFADE_SAMPLES, 48);
        assert_eq!(MAX_CORE_QUADS, 10);
        assert_eq!(cum_kautz_banks(Tier::Lite), 2);
        assert_eq!(cum_kautz_banks(Tier::Full), 3);
        assert_eq!(cum_kautz_banks(Tier::Pro), 6);
        assert_eq!(cum_kautz_banks(Tier::Ultra), 8);
        assert_eq!(
            QUAD_FLAG_HAS_EXT | QUAD_FLAG_HOST_WRITE_LOCK,
            (1 << 1) | (1 << 2)
        );
    }
}
