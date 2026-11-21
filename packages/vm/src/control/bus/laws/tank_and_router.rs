//! Canonical `3 kHz` `Op-T` tank (`slots 12..=25`), `Op-P` phasor (`slot 29`), and I/O router
//! gate (`slot 30`) modulation SSOT helpers (`#![no_std]`, `#![forbid(unsafe_code)]`,
//! `ref[dsp.ctrl.dispatch.modulation-laws]`, `ref[dsp.router.modulation-bus]`).

#![forbid(unsafe_code)]

use core::f32::consts::PI;

use dnam_kernels::TANK_MAX_G_LIN_F32;

use crate::graph::tank::quad::DAMP_EFF_MAX;
use crate::graph::tank::ring::TANK_ALLPASS_G_MAX;
use crate::math::{
    Q32_SCALE_F64, clamp_finite_f32, exp2_signed_oct_f32, f32_to_fp16, floor_f32, fp16_to_f32,
    round_half_away_f64, sanitize_finite_f32,
};

/// Evaluates slot `14` target Givens angle by pre-clamping `tank_psi_delta` to `[-pi, +pi]`
/// BEFORE clamping `base_psi + delta` to `[-pi, +pi]` (`parametric_control_bus.md:370`).
#[inline]
pub fn eval_tank_target_psi(base_psi: f32, tank_psi_delta: f32) -> f32 {
    let base = sanitize_finite_f32(base_psi);
    let delta = clamp_finite_f32(tank_psi_delta, -PI, PI, 0.0);
    clamp_finite_f32(base + delta, -PI, PI, 0.0)
}

/// Evaluates slot `12` (`tank_delay_mod`) in `48 kHz` reference samples, guarding non-finite inputs.
#[inline]
pub fn eval_tank_delay_mod_48k(stage_target_12: f32) -> f32 {
    sanitize_finite_f32(stage_target_12)
}

/// Evaluates `Op-T` loop feedback gain from slot `13`, clamped to `[0.0, TANK_MAX_G_LIN_F32]`.
#[inline]
pub fn eval_tank_g_loop(base_g_loop: f32, stage_target_13: f32) -> f32 {
    clamp_finite_f32(
        base_g_loop + sanitize_finite_f32(stage_target_13),
        0.0,
        TANK_MAX_G_LIN_F32,
        0.0,
    )
}

/// Evaluates slot `17` (`op_t_delay_scale_oct`) as `(oct_clamped, exp2_signed_oct_f32(oct))`.
#[inline]
pub fn eval_tank_delay_scale(stage_target_17: f32) -> (f32, f32) {
    let oct = clamp_finite_f32(stage_target_17, -3.0, 3.0, 0.0);
    (oct, exp2_signed_oct_f32(oct))
}

/// Quantizes a validated `ramp_rate_hz` to signed `i32` `Q32` phase increment.
#[inline]
pub fn quantize_tank_ramp_inc_q32(rate_hz: f32, inv_fs_vm_f64: f64) -> i32 {
    if !rate_hz.is_finite() {
        0
    } else {
        round_half_away_f64(f64::from(rate_hz) * Q32_SCALE_F64 * inv_fs_vm_f64) as i64 as i32
    }
}

/// Evaluates clamped `ramp_rate_hz` (`[-16, +16]`) and signed `i32` `Q32` phase increment (`slot 18`).
#[inline]
pub fn eval_tank_ramp_rate_and_inc_q32(
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
        quantize_tank_ramp_inc_q32(rate_eff, inv_fs_vm_f64),
    )
}

/// Quantizes a validated `lfo_rate_hz` to unsigned `u32` `Q32` increment (`0` if rate/depth `<= 0`).
#[inline]
pub fn quantize_tank_lfo_inc_q32(rate_hz: f32, max_lfo_depth_eff: f32, inv_fs_vm_f64: f64) -> u32 {
    if rate_hz <= 0.0
        || !rate_hz.is_finite()
        || max_lfo_depth_eff <= 0.0
        || !max_lfo_depth_eff.is_finite()
    {
        0
    } else {
        ((f64::from(rate_hz) * Q32_SCALE_F64 * inv_fs_vm_f64 + 0.5) as u64) as u32
    }
}

/// Converts effective `lfo_rate_hz` (`base + t[19]` clamped `[0, 16]`) to unsigned `u32` `Q32`
/// phase increment, returning `0` when rate or max depth `<= 0.0` (`slot 19`).
#[inline]
pub fn eval_tank_lfo_inc_q32(
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
    quantize_tank_lfo_inc_q32(rate_eff, max_lfo_depth_eff, inv_fs_vm_f64)
}

/// Evaluates `slot 20` (`op_t_lfo_depth_scale`) across all `4` quad delay lanes, returning
/// `(fp16_bits, eff_f32, max_lfo_depth_eff)` bounded by `min(512.0, floor(cap * 0.25))`.
#[inline]
pub fn eval_tank_lfo_depths(
    base_lfo_depths: [f32; 4],
    stage_target_20: f32,
    cap: usize,
) -> ([u16; 4], [f32; 4], f32) {
    let lfo_scale = exp2_signed_oct_f32(clamp_finite_f32(stage_target_20, -4.0, 4.0, 0.0));
    let lfo_cap_max = 512.0_f32.min(floor_f32((cap as f32) * 0.25));
    let mut fp16 = [0_u16; 4];
    let mut eff = [0.0_f32; 4];
    let mut max_lfo = 0.0_f32;
    for k in 0..4 {
        let d_k = clamp_finite_f32(
            sanitize_finite_f32(base_lfo_depths[k]) * lfo_scale,
            0.0,
            lfo_cap_max,
            0.0,
        );
        fp16[k] = f32_to_fp16(d_k);
        eff[k] = fp16_to_f32(fp16[k]);
        max_lfo = max_lfo.max(eff[k]);
    }
    (fp16, eff, max_lfo)
}

/// Evaluates `slot 21` (`op_t_damp_pole_delta`) clamped inside `[0.0, DAMP_EFF_MAX]`.
#[inline]
pub fn eval_tank_damp_pole(base_damp_pole_zoh: f32, stage_target_21: f32) -> f32 {
    clamp_finite_f32(
        sanitize_finite_f32(base_damp_pole_zoh) + sanitize_finite_f32(stage_target_21),
        0.0,
        DAMP_EFF_MAX,
        0.0,
    )
}

/// Evaluates Quad-0-only `dry_mix` modulation from slot `22` (`op_t_dry_mix_mod`), leaving
/// `quad_idx >= 1` unmodified (`ref[dsp.ctrl.dispatch.modulation-laws]`).
#[inline]
pub fn eval_tank_dry_mix_for_quad(quad_idx: usize, base_dry_mix: f32, stage_target_22: f32) -> f32 {
    if quad_idx == 0 {
        clamp_finite_f32(
            base_dry_mix + sanitize_finite_f32(stage_target_22),
            -2.0,
            2.0,
            1.0,
        )
    } else {
        clamp_finite_f32(base_dry_mix, -2.0, 2.0, 0.0)
    }
}

/// Evaluates `slot 23` (`op_t_input_gain_mod`) across all `4` quad injection lanes as
/// `(fp16_bits, eff_f32)` clamped to `[-4.0, +4.0]`.
#[inline]
pub fn eval_tank_input_gains(
    base_input_gains: [f32; 4],
    stage_target_23: f32,
) -> ([u16; 4], [f32; 4]) {
    let inj_scale = exp2_signed_oct_f32(clamp_finite_f32(stage_target_23, -4.0, 4.0, 0.0));
    let mut fp16 = [0_u16; 4];
    let mut eff = [0.0_f32; 4];
    for k in 0..4 {
        let g_k = clamp_finite_f32(
            sanitize_finite_f32(base_input_gains[k]) * inj_scale,
            -4.0,
            4.0,
            0.0,
        );
        fp16[k] = f32_to_fp16(g_k);
        eff[k] = fp16_to_f32(fp16[k]);
    }
    (fp16, eff)
}

/// Evaluates `slot 24` (`op_t_window_depth_delta`) clamped inside `[0.0, 1.0]`.
#[inline]
pub fn eval_tank_window_depth(base_window_depth: f32, stage_target_24: f32) -> f32 {
    clamp_finite_f32(
        sanitize_finite_f32(base_window_depth) + sanitize_finite_f32(stage_target_24),
        0.0,
        1.0,
        0.0,
    )
}

/// Evaluates `slot 25` (`op_t_allpass_g_delta`) across all `4` Schroeder lanes in
/// `[-TANK_ALLPASS_G_MAX, +TANK_ALLPASS_G_MAX]`.
#[inline]
pub fn eval_tank_allpass_gains(base_allpass_g: [f32; 4], stage_target_25: f32) -> [f32; 4] {
    let delta = sanitize_finite_f32(stage_target_25);
    let mut eff = [0.0_f32; 4];
    for k in 0..4 {
        eff[k] = clamp_finite_f32(
            sanitize_finite_f32(base_allpass_g[k]) + delta,
            -TANK_ALLPASS_G_MAX,
            TANK_ALLPASS_G_MAX,
            0.0,
        );
    }
    eff
}

/// Evaluates `slot 29` (`op_p_phasor_rate_oct`) octave multiplier `2^{clamp(t[29], -4, +4)}`.
#[inline]
pub fn eval_phasor_rate_scale(stage_target_29: f32) -> f32 {
    exp2_signed_oct_f32(clamp_finite_f32(stage_target_29, -4.0, 4.0, 0.0))
}

/// Evaluates `slot 29` effective phasor frequency in Hz: multiplies BEFORE clamping to
/// `[-0.49 * fs_vm, +0.49 * fs_vm]` (`router_and_crossbar_topology.md:294`).
#[inline]
pub fn eval_phasor_eff_freq_hz(base_freq_hz: f32, stage_target_29: f32, fs_vm: f64) -> f32 {
    let ph_lim = (0.49 * fs_vm) as f32;
    let ph_scale = eval_phasor_rate_scale(stage_target_29);
    clamp_finite_f32(
        sanitize_finite_f32(base_freq_hz) * ph_scale,
        -ph_lim,
        ph_lim,
        0.0,
    )
}

/// Evaluates `slot 30` (`router_gate_thresh_delta`) clamped inside `[0.0, max_thresh]`.
#[inline]
pub fn eval_router_gate_thresh(base_thresh: f32, stage_target_30: f32, max_thresh: f32) -> f32 {
    clamp_finite_f32(
        sanitize_finite_f32(base_thresh) + sanitize_finite_f32(stage_target_30),
        0.0,
        max_thresh,
        0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::const_abs_f32;

    #[test]
    fn test_tank_and_router_laws_specs_and_bounds() {
        // Opposite-sign pre-clamp (`tank_psi_delta` clamped to `[-pi, +pi]` before summing `base_psi`).
        let opposite = eval_tank_target_psi(-2.0, 10.0);
        assert!(const_abs_f32(opposite - (-2.0 + PI)) < 1.0e-5);
        assert_eq!(eval_tank_target_psi(f32::NAN, 0.5), 0.5);
        assert_eq!(eval_tank_target_psi(0.5, f32::NAN), 0.5);

        // Slot 29 scales BEFORE Nyquist clamping (96 kHz base pitched -1 oct at 44.1 kHz VM).
        let f_down = eval_phasor_eff_freq_hz(22_500.0, -1.0, 44_100.0);
        assert!(const_abs_f32(f_down - 11_250.0) < 1.0e-2);
        let f_clamped = eval_phasor_eff_freq_hz(48_000.0, 0.0, 44_100.0);
        assert!(const_abs_f32(f_clamped - (0.49 * 44_100.0) as f32) < 1.0e-3);
        assert_eq!(eval_phasor_eff_freq_hz(f32::NAN, 1.0, 48_000.0), 0.0);

        // Slot 30 router gate threshold.
        assert_eq!(eval_router_gate_thresh(0.1, 0.05, 0.5), 0.15);
        assert_eq!(eval_router_gate_thresh(0.1, f32::NAN, 0.5), 0.1);
        assert_eq!(eval_router_gate_thresh(0.4, 0.5, 0.5), 0.5);

        // Slots 12, 17, 20, 21, 23, 24, 25.
        assert_eq!(eval_tank_delay_mod_48k(f32::NAN), 0.0);
        assert_eq!(eval_tank_delay_mod_48k(3.5), 3.5);
        assert_eq!(eval_tank_delay_scale(10.0), (3.0, 8.0));
        let (lfo_fp16, lfo_eff, max_lfo) = eval_tank_lfo_depths([10.0, 40.0, 0.0, 0.0], 2.0, 128);
        assert_eq!(max_lfo, 32.0);
        assert_eq!(lfo_eff, [32.0, 32.0, 0.0, 0.0]);
        assert_eq!(lfo_fp16[0], f32_to_fp16(32.0));
        assert!(const_abs_f32(eval_tank_damp_pole(0.9, 1.0) - DAMP_EFF_MAX) < 1.0e-7);
        let (inj_fp16, inj_eff) = eval_tank_input_gains([1.0, -2.0, 0.5, 0.0], 2.0);
        assert_eq!(inj_eff, [4.0, -4.0, 2.0, 0.0]);
        assert_eq!(inj_fp16[0], f32_to_fp16(4.0));
        assert_eq!(eval_tank_window_depth(0.5, 1.0), 1.0);
        assert_eq!(eval_tank_dry_mix_for_quad(1, f32::NAN, 0.25), 0.0);
        assert_eq!(
            eval_tank_allpass_gains([0.6, -0.6, 0.0, 0.1], 0.5),
            [TANK_ALLPASS_G_MAX, -0.6_f32 + 0.5, 0.5, 0.6]
        );
    }
}
