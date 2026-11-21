//! Canonical `3 kHz` `Op-T` `TransportQuad` (`slots 12..=25`), `Op-P` crossbar phasor (`slot 29`),
//! and `SensoryQuad` VCA bias (`slot 30`) modulation SSOT helpers (`#![no_std]`, `#![forbid(unsafe_code)]`,
//! `ref[dsp.ctrl.dispatch.modulation-laws]`, `ref[dsp.router.modulation-bus]`,
//! `ref[dsp.op-t.window.partition-of-unity]`).

#![forbid(unsafe_code)]

use core::f32::consts::PI;

use dnam_kernels::TANK_MAX_G_LIN_F32;

use crate::graph::transport::quad::DAMP_EFF_MAX;
use crate::graph::transport::ring::TRANSPORT_ALLPASS_G_MAX;
use crate::math::{
    Q32_SCALE_F64, clamp_finite_f32, exp2_signed_oct_f32, floor_f32, round_half_away_f64,
    sanitize_finite_f32,
};

/// Evaluates slot `14` target Givens angle by pre-clamping `tank_psi_delta` to `[-pi, +pi]`
/// BEFORE clamping `base_psi + delta` to `[-pi, +pi]` (`ref[dsp.ctrl.dispatch.modulation-laws]`).
#[inline]
pub fn eval_transport_target_psi(base_psi: f32, tank_psi_delta: f32) -> f32 {
    let base = sanitize_finite_f32(base_psi);
    let delta = clamp_finite_f32(tank_psi_delta, -PI, PI, 0.0);
    clamp_finite_f32(base + delta, -PI, PI, 0.0)
}

/// Evaluates slot `12` (`tank_delay_mod`) in `48 kHz` reference samples, guarding non-finite inputs.
#[inline]
pub fn eval_transport_delay_mod_48k(stage_target_12: f32) -> f32 {
    sanitize_finite_f32(stage_target_12)
}

/// Evaluates `Op-T` loop feedback gain from slot `13`, clamped to `[0.0, TANK_MAX_G_LIN_F32]`.
#[inline]
pub fn eval_transport_g_loop(base_g_loop: f32, stage_target_13: f32) -> f32 {
    clamp_finite_f32(
        base_g_loop + sanitize_finite_f32(stage_target_13),
        0.0,
        TANK_MAX_G_LIN_F32,
        0.0,
    )
}

/// Evaluates slot `17` (`op_t_delay_scale_oct`) as `(oct_clamped, exp2_signed_oct_f32(oct))`.
#[inline]
pub fn eval_transport_delay_scale(stage_target_17: f32) -> (f32, f32) {
    let oct = clamp_finite_f32(stage_target_17, -3.0, 3.0, 0.0);
    (oct, exp2_signed_oct_f32(oct))
}

/// Quantizes a validated `ramp_rate_hz` to signed `i32` `Q32` phase increment (`0 FDIV`).
#[inline]
pub fn quantize_transport_ramp_inc_q32(rate_hz: f32, inv_fs_vm_f64: f64) -> i32 {
    if !rate_hz.is_finite() || !inv_fs_vm_f64.is_finite() || inv_fs_vm_f64 <= 0.0 {
        0
    } else {
        round_half_away_f64(f64::from(rate_hz) * Q32_SCALE_F64 * inv_fs_vm_f64) as i64 as i32
    }
}

/// Evaluates clamped `ramp_rate_hz` (`[-16, +16]`) and signed `i32` `Q32` phase increment (`slot 18`).
#[inline]
pub fn eval_transport_ramp_rate_and_inc_q32(
    base_ramp_rate_hz: f32,
    stage_target_18: f32,
    inv_fs_vm_f64: f64,
) -> (f32, i32) {
    let rate_eff = clamp_finite_f32(
        sanitize_finite_f32(base_ramp_rate_hz) + sanitize_finite_f32(stage_target_18),
        -16.0,
        16.0,
        0.0,
    );
    (
        rate_eff,
        quantize_transport_ramp_inc_q32(rate_eff, inv_fs_vm_f64),
    )
}

/// Quantizes a validated `lfo_rate_hz` to unsigned `u32` `Q32` increment (`0` if rate/depth `<= 0`).
#[inline]
pub fn quantize_transport_lfo_inc_q32(
    rate_hz: f32,
    max_lfo_depth_eff: f32,
    inv_fs_vm_f64: f64,
) -> u32 {
    if rate_hz <= 0.0
        || !rate_hz.is_finite()
        || max_lfo_depth_eff <= 0.0
        || !max_lfo_depth_eff.is_finite()
        || !inv_fs_vm_f64.is_finite()
        || inv_fs_vm_f64 <= 0.0
    {
        0
    } else {
        ((f64::from(rate_hz) * Q32_SCALE_F64 * inv_fs_vm_f64 + 0.5) as u64) as u32
    }
}

/// Converts effective `lfo_rate_hz` (`base + t[19]` clamped `[0, 16]`) to unsigned `u32` `Q32`
/// phase increment, returning `0` when rate or max depth `<= 0.0` (`slot 19`).
#[inline]
pub fn eval_transport_lfo_inc_q32(
    base_lfo_rate_hz: f32,
    stage_target_19: f32,
    max_lfo_depth_eff: f32,
    inv_fs_vm_f64: f64,
) -> u32 {
    let rate_eff = clamp_finite_f32(
        sanitize_finite_f32(base_lfo_rate_hz) + sanitize_finite_f32(stage_target_19),
        0.0,
        16.0,
        0.0,
    );
    quantize_transport_lfo_inc_q32(rate_eff, max_lfo_depth_eff, inv_fs_vm_f64)
}

/// Evaluates `slot 20` (`op_t_lfo_depth_scale`) across all `4` quad delay lanes with passive
/// `(1.0 - mu_freeze)` tapering, returning `(eff_depths, max_lfo_depth_eff)` bounded by
/// `min(512.0, floor(cap / 4))` (`ref[dsp.ctrl.dispatch.modulation-laws]`).
#[inline]
pub fn eval_transport_lfo_depths(
    base_lfo_depths: &[f32; 4],
    scale_inv_rho_vm: f32,
    cap: usize,
    mu_freeze: f32,
    stage_target_20: f32,
) -> ([f32; 4], f32) {
    let _ = scale_inv_rho_vm;
    let lfo_scale = exp2_signed_oct_f32(clamp_finite_f32(stage_target_20, -4.0, 4.0, 0.0));
    let freeze_taper = 1.0 - clamp_finite_f32(mu_freeze, 0.0, 1.0, 0.0);
    let lfo_cap_max = 512.0_f32.min(floor_f32((cap as f32) * 0.25));
    let mut eff = [0.0_f32; 4];
    let mut max_lfo = 0.0_f32;
    for k in 0..4 {
        let d_k = clamp_finite_f32(
            sanitize_finite_f32(base_lfo_depths[k]) * lfo_scale,
            0.0,
            lfo_cap_max,
            0.0,
        ) * freeze_taper;
        eff[k] = d_k;
        max_lfo = max_lfo.max(d_k);
    }
    (eff, max_lfo)
}

/// Evaluates `slot 21` (`op_t_damp_pole_delta`) clamped inside `[0.0, DAMP_EFF_MAX]`.
#[inline]
pub fn eval_transport_damp_pole(base_damp_pole_zoh: f32, stage_target_21: f32) -> f32 {
    clamp_finite_f32(
        sanitize_finite_f32(base_damp_pole_zoh) + sanitize_finite_f32(stage_target_21),
        0.0,
        DAMP_EFF_MAX,
        0.0,
    )
}

/// Evaluates stage-level `Transport` dry pass-through mix from `slot 22` (`op_t_dry_mix_mod`)
/// clamped inside `[-2.0, 2.0]` (`ref[dsp.ctrl.dispatch.modulation-laws]`).
#[inline]
pub fn eval_transport_dry_mix_stage(base_dry_mix: f32, stage_target_22: f32) -> f32 {
    clamp_finite_f32(
        sanitize_finite_f32(base_dry_mix) + sanitize_finite_f32(stage_target_22),
        -2.0,
        2.0,
        0.0,
    )
}

/// Evaluates `slot 23` (`op_t_input_gain_mod`) across all `4` quad injection lanes clamped to `[-4.0, +4.0]`.
#[inline]
pub fn eval_transport_input_gains(base_input_gains: &[f32; 4], stage_target_23: f32) -> [f32; 4] {
    let inj_scale = exp2_signed_oct_f32(clamp_finite_f32(stage_target_23, -4.0, 4.0, 0.0));
    let mut eff = [0.0_f32; 4];
    for k in 0..4 {
        eff[k] = clamp_finite_f32(
            sanitize_finite_f32(base_input_gains[k]) * inj_scale,
            -4.0,
            4.0,
            0.0,
        );
    }
    eff
}

/// Evaluates `slot 24` (`op_t_window_depth_delta`) clamped inside `[0.0, 1.0]`.
#[inline]
pub fn eval_transport_window_depth(base_window_depth: f32, stage_target_24: f32) -> f32 {
    clamp_finite_f32(
        sanitize_finite_f32(base_window_depth) + sanitize_finite_f32(stage_target_24),
        0.0,
        1.0,
        0.0,
    )
}

/// Evaluates `slot 25` (`op_t_allpass_g_delta`) across all `4` Schroeder lanes clamped to
/// `[-TRANSPORT_ALLPASS_G_MAX, +TRANSPORT_ALLPASS_G_MAX]` ahead of `(1.0 - mu_freeze)` taper.
#[inline]
pub fn eval_transport_allpass_gains(
    base_allpass_g: &[f32; 4],
    mu_freeze: f32,
    stage_target_25: f32,
) -> [f32; 4] {
    let delta = sanitize_finite_f32(stage_target_25);
    let freeze_taper = 1.0 - clamp_finite_f32(mu_freeze, 0.0, 1.0, 0.0);
    let mut eff = [0.0_f32; 4];
    for k in 0..4 {
        let g_k = clamp_finite_f32(
            sanitize_finite_f32(base_allpass_g[k]) + delta,
            -TRANSPORT_ALLPASS_G_MAX,
            TRANSPORT_ALLPASS_G_MAX,
            0.0,
        );
        eff[k] = g_k * freeze_taper;
    }
    eff
}

/// Evaluates Crossbar & `Transport` `wet_scale` (`clamp(1.0 + clamp(stage_targets[15], -1.0, 1.0), 0.0, 2.0)`).
#[inline]
pub fn eval_crossbar_wet_scale(stage_target_15: f32) -> f32 {
    clamp_finite_f32(
        1.0 + clamp_finite_f32(stage_target_15, -1.0, 1.0, 0.0),
        0.0,
        2.0,
        1.0,
    )
}

/// Evaluates continuous passive `Op-T` `freeze_mix` (`slot 16`), pinning to `1.0` when
/// `host_write_locked` is active (`QUAD_FLAG_HOST_WRITE_LOCK`).
#[inline]
pub fn eval_op_t_freeze_mix_eff(
    base_freeze_mix: f32,
    stage_target_16: f32,
    host_write_locked: bool,
) -> f32 {
    if host_write_locked {
        return 1.0;
    }
    (clamp_finite_f32(base_freeze_mix, 0.0, 1.0, 0.0) + sanitize_finite_f32(stage_target_16))
        .clamp(0.0, 1.0)
}

/// Evaluates `slot 30` (`sense_vca_gain_delta`) threshold bias clamped inside `[-1.0, 1.0]`.
#[inline]
pub fn eval_sensory_vca_bias_slot30(stage_target_30: f32) -> f32 {
    clamp_finite_f32(stage_target_30, -1.0, 1.0, 0.0)
}

/// Evaluates `slot 29` (`op_p_phasor_rate_oct`) effective frequency clamped to `[-0.49 * fs_vm, +0.49 * fs_vm]`
/// and quantized to signed `i32` `Q0.32` VM-rate phase increment (`0 FDIV`, `ref[dsp.crossbar.top-of-frame]`).
#[inline]
pub fn eval_phasor_inc_vm_q32(
    base_freq_hz: f32,
    slot_29_oct: f32,
    fs_vm_hz: f32,
    inv_fs_vm_f64: f64,
) -> i32 {
    if !fs_vm_hz.is_finite()
        || fs_vm_hz <= 0.0
        || !inv_fs_vm_f64.is_finite()
        || inv_fs_vm_f64 <= 0.0
    {
        return 0;
    }
    let ph_lim = (0.49 * f64::from(fs_vm_hz)) as f32;
    let ph_scale = exp2_signed_oct_f32(clamp_finite_f32(slot_29_oct, -4.0, 4.0, 0.0));
    let f_eff = clamp_finite_f32(
        sanitize_finite_f32(base_freq_hz) * ph_scale,
        -ph_lim,
        ph_lim,
        0.0,
    );
    let max_i32 = f64::from(i32::MAX);
    let scaled = round_half_away_f64(f64::from(f_eff) * Q32_SCALE_F64 * inv_fs_vm_f64);
    scaled.clamp(-max_i32, max_i32) as i64 as i32
}

/// Converts VM-rate signed `Q0.32` phasor increment to core-rate increment (`inc_vm >> 1` when `os2x_active`).
#[inline]
pub const fn eval_phasor_inc_core_q32(inc_vm_q32: i32, os2x_active: bool) -> i32 {
    if os2x_active {
        inc_vm_q32 >> 1
    } else {
        inc_vm_q32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::const_abs_f32;

    /// verify[dsp.ctrl.dispatch.modulation-laws]
    #[test]
    fn test_transport_and_sensory_laws_bounds_and_freeze_taper() {
        // Opposite-sign pre-clamp (`tank_psi_delta` clamped to `[-pi, +pi]` before summing `base_psi`).
        let opposite = eval_transport_target_psi(-2.0, 10.0);
        assert!(const_abs_f32(opposite - (-2.0 + PI)) < 1.0e-5);
        assert_eq!(eval_transport_target_psi(f32::NAN, 0.5), 0.5);
        assert_eq!(eval_transport_target_psi(0.5, f32::NAN), 0.5);

        // Slot 13 g_loop clamps and non-finite base fallback.
        assert!((eval_transport_g_loop(0.95, 0.5) - TANK_MAX_G_LIN_F32).abs() < 1.0e-7);
        assert_eq!(eval_transport_g_loop(f32::INFINITY, 0.1), 0.0);

        // Slot 16 freeze_mix and HOST_WRITE_LOCK override.
        assert_eq!(eval_op_t_freeze_mix_eff(0.25, 0.5, false), 0.75);
        assert_eq!(eval_op_t_freeze_mix_eff(0.25, -0.5, false), 0.0);
        assert_eq!(eval_op_t_freeze_mix_eff(0.8, 0.5, false), 1.0);
        assert_eq!(eval_op_t_freeze_mix_eff(0.0, -1.0, true), 1.0);
        assert_eq!(eval_op_t_freeze_mix_eff(f32::NAN, f32::NAN, false), 0.0);

        // Slot 22 stage dry mix and slot 30 SensoryQuad VCA bias.
        assert_eq!(eval_transport_dry_mix_stage(0.5, 0.25), 0.75);
        assert_eq!(eval_transport_dry_mix_stage(1.5, 1.0), 2.0);
        assert_eq!(eval_transport_dry_mix_stage(-1.5, -1.0), -2.0);
        assert_eq!(eval_transport_dry_mix_stage(f32::NAN, 0.25), 0.25);
        assert_eq!(eval_sensory_vca_bias_slot30(0.45), 0.45);
        assert_eq!(eval_sensory_vca_bias_slot30(2.0), 1.0);
        assert_eq!(eval_sensory_vca_bias_slot30(-2.0), -1.0);
        assert_eq!(eval_sensory_vca_bias_slot30(f32::NAN), 0.0);

        // Slot 29 signed Q0.32 phasor VM/core increment scaling before Nyquist clamping.
        let inv_48k = 1.0 / 48_000.0;
        let inc_vm = eval_phasor_inc_vm_q32(-12_000.0, 0.0, 48_000.0, inv_48k);
        assert_eq!(inc_vm, -1_073_741_824);
        assert_eq!(eval_phasor_inc_core_q32(inc_vm, false), inc_vm);
        assert_eq!(eval_phasor_inc_core_q32(inc_vm, true), inc_vm >> 1);
        let inc_down = eval_phasor_inc_vm_q32(22_500.0, -1.0, 44_100.0, 1.0 / 44_100.0);
        let exp_down = eval_phasor_inc_vm_q32(11_250.0, 0.0, 44_100.0, 1.0 / 44_100.0);
        assert!((inc_down - exp_down).abs() <= 1);
        assert_eq!(eval_phasor_inc_vm_q32(f32::NAN, 1.0, 48_000.0, inv_48k), 0);

        // Slots 12, 15, 17, 18, 19, 20, 21, 23, 24, 25 (with freeze_mix taper).
        assert_eq!(eval_transport_delay_mod_48k(f32::NAN), 0.0);
        assert_eq!(eval_transport_delay_mod_48k(3.5), 3.5);
        assert_eq!(eval_crossbar_wet_scale(0.5), 1.5);
        assert_eq!(eval_crossbar_wet_scale(-2.0), 0.0);
        assert_eq!(eval_crossbar_wet_scale(f32::NAN), 1.0);
        assert_eq!(eval_transport_delay_scale(10.0), (3.0, 8.0));

        let (r_eff, r_q32) = eval_transport_ramp_rate_and_inc_q32(2.0, -4.0, inv_48k);
        assert_eq!(r_eff, -2.0);
        assert!(r_q32 < 0);
        assert_eq!(quantize_transport_ramp_inc_q32(f32::NAN, inv_48k), 0);
        assert_eq!(eval_transport_lfo_inc_q32(0.0, 0.0, 1.0, inv_48k), 0);
        assert_eq!(eval_transport_lfo_inc_q32(1.0, 0.0, 0.0, inv_48k), 0);
        assert!(eval_transport_lfo_inc_q32(1.0, 0.5, 0.5, inv_48k) > 0);
        assert_eq!(quantize_transport_lfo_inc_q32(1.0, f32::NAN, inv_48k), 0);

        let (lfo_eff, max_lfo) =
            eval_transport_lfo_depths(&[10.0, 40.0, 0.0, 0.0], 1.0, 128, 0.0, 2.0);
        assert_eq!(max_lfo, 32.0);
        assert_eq!(lfo_eff, [32.0, 32.0, 0.0, 0.0]);
        let (lfo_half, max_half) =
            eval_transport_lfo_depths(&[10.0, 40.0, 0.0, 0.0], 1.0, 128, 0.5, 2.0);
        assert_eq!((lfo_half, max_half), ([16.0, 16.0, 0.0, 0.0], 16.0));
        let (lfo_frozen, max_frozen) =
            eval_transport_lfo_depths(&[10.0, 40.0, 0.0, 0.0], 1.0, 128, 1.0, 2.0);
        assert_eq!((lfo_frozen, max_frozen), ([0.0; 4], 0.0));

        assert!(const_abs_f32(eval_transport_damp_pole(0.9, 1.0) - DAMP_EFF_MAX) < 1.0e-7);
        assert_eq!(
            eval_transport_input_gains(&[1.0, -2.0, 0.5, 0.0], 2.0),
            [4.0, -4.0, 2.0, 0.0]
        );
        assert_eq!(eval_transport_window_depth(0.5, 1.0), 1.0);
        assert_eq!(
            eval_transport_allpass_gains(&[0.6, -0.6, 0.0, 0.1], 0.0, 0.5),
            [TRANSPORT_ALLPASS_G_MAX, -0.6_f32 + 0.5, 0.5, 0.6]
        );
        assert_eq!(
            eval_transport_allpass_gains(&[0.6, -0.6, 0.0, 0.1], 0.5, 0.5),
            [
                0.5 * TRANSPORT_ALLPASS_G_MAX,
                0.5 * (-0.6_f32 + 0.5),
                0.25,
                0.3
            ]
        );
        assert_eq!(
            eval_transport_allpass_gains(&[0.6, -0.6, 0.0, 0.1], 1.0, 0.5),
            [0.0; 4]
        );
    }
}
