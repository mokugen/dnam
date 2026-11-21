//! `ControlBus` `slots 12..=25` dispatch, Givens `psi` slewing, continuous `freeze_mix` hold
//! transitions, and `Op-T` modulation target latching (`ref[dsp.ctrl.dispatch.micro-ramp]`).

use core::f32::consts::PI;

use dnam_kernels::{
    QUAD_FLAG_HOST_WRITE_LOCK, TANK_MAX_SIGMA_SAT_F32, algebraic_unit_rotor_f32,
    compute_half_alpha_cubic_f32, compute_op_t_pitch_down_g_loop_eff,
};

use crate::control::bus::laws::{
    eval_crossbar_wet_scale, eval_op_t_freeze_mix_eff, eval_transport_allpass_gains,
    eval_transport_damp_pole, eval_transport_delay_mod_48k, eval_transport_delay_scale,
    eval_transport_g_loop, eval_transport_input_gains, eval_transport_lfo_depths,
    eval_transport_lfo_inc_q32, eval_transport_ramp_rate_and_inc_q32, eval_transport_target_psi,
    eval_transport_window_depth,
};
use crate::math::{clamp_finite_f32, f32_to_fp16, fp16_to_f32};

use super::TRANSPORT_MAX_QUADS;
use super::quad::{MIN_SIGMA_SAT_F32, QuadColdMeta, QuadEffectiveTargets, TransportQuadHotCompact};
use super::ring::{PartitionedSchroederBank, TransportExtRing};

const PSI_SLEW_MAX_RAD: f32 = 0.25;

/// Extracted `ControlBus` `stage_targets[12..=25]` slots governing `Op-T` transport modulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransportModSlots {
    /// Slot 12: `tank_delay_mod` in `48 kHz` reference samples.
    pub tank_delay_mod: f32,
    /// Slot 13: `tank_gain_mod`.
    pub tank_gain_mod: f32,
    /// Slot 14 (plus any `w_tank_psi` sensitivity dot): `tank_psi_delta` in `[-pi, pi]` rad.
    pub tank_psi_delta: f32,
    /// Slot 15: `crossbar_wet_mod` in `[-1.0, 1.0]`.
    pub crossbar_wet_mod: f32,
    /// Slot 16: `op_t_freeze_mix_delta` uniformly micro-ramped (`mu_freeze = clamp(base + t[16], 0, 1)`).
    pub op_t_freeze_mix_delta: f32,
    /// Slot 17: `op_t_delay_scale_oct` in `[-3.0, 3.0]`.
    pub op_t_delay_scale_oct: f32,
    /// Slot 18: `op_t_ramp_rate_hz_delta`.
    pub op_t_ramp_rate_hz_delta: f32,
    /// Slot 19: `op_t_lfo_rate_hz_delta`.
    pub op_t_lfo_rate_hz_delta: f32,
    /// Slot 20: `op_t_lfo_depth_scale`.
    pub op_t_lfo_depth_scale: f32,
    /// Slot 21: `op_t_damp_pole_delta`.
    pub op_t_damp_pole_delta: f32,
    /// Slot 22: `op_t_dry_mix_mod` (stage-level across all tiers via `eval_transport_dry_mix_stage`).
    pub op_t_dry_mix_mod: f32,
    /// Slot 23: `op_t_input_gain_mod`.
    pub op_t_input_gain_mod: f32,
    /// Slot 24: `op_t_window_depth_delta`.
    pub op_t_window_depth_delta: f32,
    /// Slot 25: `op_t_allpass_g_delta`.
    pub op_t_allpass_g_delta: f32,
}

impl TransportModSlots {
    /// Maps a canonical 32-element `ControlBus` `stage_targets` slice (`[12..=25]`) into
    /// [`TransportModSlots`].
    #[inline]
    pub const fn from_targets_32(t: &[f32; 32]) -> Self {
        Self {
            tank_delay_mod: t[12],
            tank_gain_mod: t[13],
            tank_psi_delta: t[14],
            crossbar_wet_mod: t[15],
            op_t_freeze_mix_delta: t[16],
            op_t_delay_scale_oct: t[17],
            op_t_ramp_rate_hz_delta: t[18],
            op_t_lfo_rate_hz_delta: t[19],
            op_t_lfo_depth_scale: t[20],
            op_t_damp_pole_delta: t[21],
            op_t_dry_mix_mod: t[22],
            op_t_input_gain_mod: t[23],
            op_t_window_depth_delta: t[24],
            op_t_allpass_g_delta: t[25],
        }
    }
}

/// Latches all non-slew `Op-T` modulation targets (`slots 12, 13, 15, 16..=25`) at `4`-frame
/// `(sample_tick & 3) == 0` or `3 kHz` cadence without stepping `psi` slew
/// (`ref[dsp.ctrl.dispatch.micro-ramp]`).
pub fn latch_quad_nonslew_targets(
    quad: &mut TransportQuadHotCompact,
    meta: &QuadColdMeta,
    slots: &TransportModSlots,
    ext_ring: Option<&mut TransportExtRing<'_>>,
    schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    eff: &mut QuadEffectiveTargets,
) {
    let was_held = quad.is_full_hold(eff.mu_freeze);
    let mu_freeze = eval_op_t_freeze_mix_eff(
        meta.base_freeze_mix,
        slots.op_t_freeze_mix_delta,
        (quad.flags & QUAD_FLAG_HOST_WRITE_LOCK) != 0,
    );
    let now_held = quad.is_full_hold(mu_freeze);
    if was_held && !now_held {
        quad.clear_hold_offsets(ext_ring);
    }

    quad.freeze_mix_fp16 = f32_to_fp16(mu_freeze);
    quad.g_loop = eval_transport_g_loop(meta.base_g_loop, slots.tank_gain_mod);
    let (ramp_rate, ramp_inc_q32) = eval_transport_ramp_rate_and_inc_q32(
        meta.base_ramp_rate_hz,
        slots.op_t_ramp_rate_hz_delta,
        meta.inv_fs_vm,
    );
    quad.ramp_inc_q32 = ramp_inc_q32;
    let g_loop_eff =
        compute_op_t_pitch_down_g_loop_eff(quad.g_loop, quad.ramp_spans_samples, quad.ramp_inc_q32);
    quad.half_alpha_cubic_fp16 = f32_to_fp16(compute_half_alpha_cubic_f32(g_loop_eff, mu_freeze));

    let (lfo_raw, _) = eval_transport_lfo_depths(
        &meta.base_lfo_depths,
        meta.scale_inv_rho_vm,
        meta.cap,
        mu_freeze,
        slots.op_t_lfo_depth_scale,
    );
    quad.lfo_depths_fp16 = lfo_raw.map(f32_to_fp16);
    eff.lfo_depths_eff = quad.lfo_depths_fp16.map(fp16_to_f32);
    let max_lfo_depth_eff = eff.lfo_depths_eff.into_iter().fold(0.0_f32, f32::max);

    let inj_raw = eval_transport_input_gains(&meta.base_input_gains, slots.op_t_input_gain_mod);
    quad.input_gains_fp16 = inj_raw.map(f32_to_fp16);
    eff.input_gains_eff = quad.input_gains_fp16.map(fp16_to_f32);

    eff.allpass_g_eff =
        eval_transport_allpass_gains(&meta.base_allpass_g, mu_freeze, slots.op_t_allpass_g_delta);
    if let Some(ap) = schroeder {
        ap.set_allpass_g(eff.allpass_g_eff);
    }

    quad.lfo_inc_q32 = eval_transport_lfo_inc_q32(
        meta.base_lfo_rate_hz,
        slots.op_t_lfo_rate_hz_delta,
        max_lfo_depth_eff,
        meta.inv_fs_vm,
    );
    quad.damp_pole = eval_transport_damp_pole(meta.base_damp_pole_zoh, slots.op_t_damp_pole_delta);
    let win_eff =
        eval_transport_window_depth(meta.base_window_depth, slots.op_t_window_depth_delta);
    quad.window_depth_fp16 = f32_to_fp16(win_eff);

    let (oct, delay_scale_pow2) = eval_transport_delay_scale(slots.op_t_delay_scale_oct);
    eff.cap = meta.cap;
    eff.scale_inv_rho_vm = meta.scale_inv_rho_vm;
    eff.tank_delay_mod_48k = eval_transport_delay_mod_48k(slots.tank_delay_mod);
    eff.delay_scale_oct = oct;
    eff.delay_scale_pow2 = delay_scale_pow2;
    eff.g_loop_eff = g_loop_eff;
    eff.sigma_sat = clamp_finite_f32(
        fp16_to_f32(quad.sigma_sat_fp16),
        MIN_SIGMA_SAT_F32,
        TANK_MAX_SIGMA_SAT_F32,
        TANK_MAX_SIGMA_SAT_F32,
    );
    eff.inv_sigma_23 = fp16_to_f32(quad.inv_sigma_23_fp16);
    eff.mu_freeze = mu_freeze;
    eff.half_alpha_cubic = fp16_to_f32(quad.half_alpha_cubic_fp16);
    (eff.cos_psi, eff.sin_psi) = (quad.cos_psi, quad.sin_psi);
    eff.damp_pole_eff = quad.damp_pole;
    eff.wet_scale = eval_crossbar_wet_scale(slots.crossbar_wet_mod);
    eff.window_depth_eff = fp16_to_f32(quad.window_depth_fp16);
    eff.ramp_rate_hz_eff = ramp_rate;
}

/// Applies non-slew `ControlBus` modulation (`slots 12, 13, 15, 16..=25`) uniformly across all
/// `p_q` active transport quads (`slot 22` is stage-level and evaluated externally).
pub fn apply_transport_mod_slots(
    quads: &mut [TransportQuadHotCompact],
    metas: &[QuadColdMeta],
    p_q: usize,
    slots: &TransportModSlots,
    ext_rings: &mut [TransportExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &mut [QuadEffectiveTargets],
) {
    let active_q = p_q
        .min(quads.len())
        .min(metas.len())
        .min(effective.len())
        .min(TRANSPORT_MAX_QUADS);
    for q in 0..active_q {
        latch_quad_nonslew_targets(
            &mut quads[q],
            &metas[q],
            slots,
            ext_rings.get_mut(q),
            schroeders.get_mut(q),
            &mut effective[q],
        );
    }
}

/// Applies `3 kHz` `ControlBus` modulation (`slots 12..=25`), `<= 0.25` rad/tick Givens slewing,
/// continuous `freeze_mix` hold-counter release, and Schroeder gain updates on one quad.
pub fn tick_quad_3khz(
    quad: &mut TransportQuadHotCompact,
    meta: &QuadColdMeta,
    slots: &TransportModSlots,
    ext_ring: Option<&mut TransportExtRing<'_>>,
    schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    eff: &mut QuadEffectiveTargets,
) {
    let target_psi = eval_transport_target_psi(meta.base_psi, slots.tank_psi_delta);
    let step_psi = (target_psi - quad.current_psi).clamp(-PSI_SLEW_MAX_RAD, PSI_SLEW_MAX_RAD);
    quad.current_psi = (quad.current_psi + step_psi).clamp(-PI, PI);
    (quad.cos_psi, quad.sin_psi) = algebraic_unit_rotor_f32(quad.current_psi);

    latch_quad_nonslew_targets(quad, meta, slots, ext_ring, schroeder, eff);
}

/// Applies `3 kHz` `ControlBus` modulation uniformly across all `p_q` active transport quads.
pub fn tick_transport_3khz(
    quads: &mut [TransportQuadHotCompact],
    metas: &[QuadColdMeta],
    p_q: usize,
    slots: &TransportModSlots,
    ext_rings: &mut [TransportExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &mut [QuadEffectiveTargets],
) {
    let active_q = p_q
        .min(quads.len())
        .min(metas.len())
        .min(effective.len())
        .min(TRANSPORT_MAX_QUADS);
    for q in 0..active_q {
        tick_quad_3khz(
            &mut quads[q],
            &metas[q],
            slots,
            ext_rings.get_mut(q),
            schroeders.get_mut(q),
            &mut effective[q],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::const_abs_f32;

    #[test]
    fn test_transport_mod_slots_extraction_and_hold_release_taper() {
        let mut t = [0.0_f32; 32];
        t[12] = 3.5;
        t[15] = 0.5;
        t[16] = -0.5;
        t[22] = -0.25;
        t[25] = 0.4;
        let slots = TransportModSlots::from_targets_32(&t);
        assert_eq!(
            (
                slots.tank_delay_mod,
                slots.crossbar_wet_mod,
                slots.op_t_freeze_mix_delta,
                slots.op_t_dry_mix_mod,
                slots.op_t_allpass_g_delta,
            ),
            (3.5, 0.5, -0.5, -0.25, 0.4)
        );

        let mut quad = TransportQuadHotCompact {
            we_hold_off: [3, 2, 1, 4],
            ..Default::default()
        };
        let meta = QuadColdMeta {
            base_freeze_mix: 1.0,
            base_lfo_rate_hz: 2.0,
            base_lfo_depths: [4.0, 2.0, 0.0, 0.0],
            ..Default::default()
        };
        let mut eff = QuadEffectiveTargets {
            mu_freeze: 1.0,
            ..Default::default()
        };
        latch_quad_nonslew_targets(&mut quad, &meta, &slots, None, None, &mut eff);
        assert_eq!(quad.we_hold_off, [0; 4]);
        assert!(const_abs_f32(eff.mu_freeze - 0.5) < 1.0e-6);
        assert!(const_abs_f32(eff.wet_scale - 1.5) < 1.0e-6);
        assert!(const_abs_f32(eff.allpass_g_eff[0] - 0.2) < 1.0e-5);
        assert!(quad.lfo_inc_q32 > 0);
    }
}
