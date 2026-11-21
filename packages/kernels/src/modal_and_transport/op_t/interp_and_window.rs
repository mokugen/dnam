//! Causal `Op-T` delay-line interpolation (Catmull-Rom Hermite, Lagrange, passive-blended
//! modulated Catmull-Rom, quadratic B-spline / Hermite-Hann partition-of-unity windows), branchless
//! `Q0.32` triangle-folded Hermite-Hann diagonal windowing, Hermite-5 sine LFO evaluation, and
//! pitch-down loop contraction (`ref[dsp.op-t.reader.catmull-rom]`,
//! `ref[dsp.op-t.window.partition-of-unity]`, `ref[dsp.op-t.zero-fdiv]`,
//! `ref[dsp.op-t.ring.q0-32-pointer]`).
//!
//! Strictly 32-bit, `#![no_std]`, `0 FDIV`/`0 UDIV`/`0 FSQRT`/`0 libm`.

use crate::modal_and_transport::kautz::rotor_and_phasor::eval_q32_quadrature_f32;
use crate::simd::{
    F32_INT_EXACT_THRESHOLD, GpuVec4, IEEE754_F32_RSQRT_SEED, KernelVec4, Q32_INV_2POW31_F32,
    Q32_INV_2POW32_F32, TANK_MAX_G_LIN_F32, const_abs_f32, const_taylor_inv_fact_f32,
    flush_subnormal_f32_bits, fp16_to_f32,
};

/// Exact compile-time reciprocal `1 / 6` for 4-point cubic Lagrange interpolation (`0 FDIV`).
const INV_6: f32 = const_taylor_inv_fact_f32(3, false);

/// Exact compile-time reciprocal `1 / 3` for 3-phase barberpole windowing (`0 FDIV`).
const INV_3: f32 = 1.0 / 3.0;

/// Exact compile-time `2 / 3` factor for 3-phase offsets.
const TWO_THIRDS: f32 = 2.0 / 3.0;

/// Binary32 exact-integer threshold `2^23` (`8_388_608.0`).
const INT_FLOOR_THRESHOLD_F32: f32 = F32_INT_EXACT_THRESHOLD;

/// Minimum causal delay floor for standalone single-sample rings (`tau >= 4.0`).
const MIN_CAUSAL_DELAY_SAMPLES: f32 = 4.0;

/// Minimum sawtooth-wrap threshold (`0.25` samples) above which boundary anti-click taper engages.
const BOUNDARY_CLICK_OFFSET_SAMPLES: f32 = 0.25;

/// Linear boundary anti-click slope `(1 / 0.25)` mapping `[0.25, 0.5]` span jumps to `[0.0, 1.0]`.
const BOUNDARY_CLICK_SLOPE: f32 = 4.0;

/// Maximum pitch-down time-warp contraction ratio (`nu_down <= 0.75`).
const MAX_PITCH_DOWN_NU: f32 = 0.75;

/// Near-unity safety floor inside pitch-down `sqrt(max(0, 1 - min(0.9999, nu_down)))`.
const MAX_PITCH_DOWN_U_SUB: f32 = 0.9999;

/// Underflow guard `1e-12` for scalar unfused NR2 `rsqrt` in [`compute_op_t_pitch_down_g_loop_eff`].
const PITCH_DOWN_NR_EPS: f32 = 1.0e-12;

const _: () = assert!(INV_6 > 0.16 && INV_6 < 0.17);
const _: () = assert!(INV_3 > 0.33 && INV_3 < 0.34);
const _: () = assert!(TWO_THIRDS > 0.66 && TWO_THIRDS < 0.67);
const _: () = assert!(MIN_CAUSAL_DELAY_SAMPLES == 4.0);

#[inline(always)]
fn clamp_finite_unit(mu: f32) -> f32 {
    if mu.is_finite() {
        mu.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline(always)]
fn floor_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    if const_abs_f32(x) >= INT_FLOOR_THRESHOLD_F32 {
        return x;
    }
    let truncated = x as i32;
    let as_float = truncated as f32;
    if as_float > x {
        as_float - 1.0
    } else {
        as_float
    }
}

#[inline(always)]
fn frac_unit_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let wrapped = x - floor_f32(x);
    if wrapped >= 1.0 {
        0.0
    } else {
        wrapped.max(0.0)
    }
}

#[inline(always)]
pub(super) const fn valid_pow2_ring_len(len: usize, mask: usize) -> bool {
    if len == 0 || (mask & mask.wrapping_add(1)) != 0 {
        return false;
    }
    match mask.checked_add(1) {
        Some(pow2_len) => pow2_len <= len,
        None => false,
    }
}

#[inline(always)]
fn split_nonneg_delay_f32(delay: f32, mask: usize) -> (usize, f32) {
    if !delay.is_finite() {
        return (0, 0.0);
    }
    let clamped = delay.clamp(0.0, mask as f32);
    let d_int = floor_f32(clamped) as usize;
    let d_frac = (clamped - (d_int as f32)).clamp(0.0, 1.0);
    if d_frac >= 1.0 && d_int < mask {
        (d_int + 1, 0.0)
    } else {
        (d_int, d_frac)
    }
}

#[inline(always)]
fn quadratic_bspline_tap_f32(unit_phase: f32) -> f32 {
    let s = (unit_phase * 3.0).clamp(0.0, 3.0);
    if s < 1.0 {
        0.5 * s * s
    } else if s < 2.0 {
        let u = s - 1.0;
        0.5 + u - u * u
    } else {
        let r = 3.0 - s;
        0.5 * r * r
    }
}

/// Evaluates 4-point cubic Lagrange interpolation weights for nodes `[-1, 0, 1, 2]` at clamped
/// fractional offset `mu in [0.0, 1.0]` with zero runtime divisions (`0 FDIV`).
#[inline(always)]
#[must_use]
pub fn cubic_lagrange_weights_f32(mu: f32) -> [f32; 4] {
    let t = clamp_finite_unit(mu);
    let tm1 = t - 1.0;
    let tp1 = t + 1.0;
    let tm2 = t - 2.0;
    [
        (-INV_6 * t * tm1 * tm2) + 0.0,
        (0.5 * tp1 * tm1 * tm2) + 0.0,
        (-0.5 * tp1 * t * tm2) + 0.0,
        (INV_6 * tp1 * t * tm1) + 0.0,
    ]
}

/// Evaluates the 4-tap Keys/Catmull-Rom cubic Hermite spline in Horner form (`0 FDIV`,
/// `ref[dsp.op-t.reader.catmull-rom]`).
#[inline(always)]
#[must_use]
pub fn catmull_rom_hermite_f32(s_m1: f32, s_0: f32, s_1: f32, s_2: f32, mu: f32) -> f32 {
    let t = clamp_finite_unit(mu);
    let c0 = s_0;
    let c1 = 0.5 * (s_1 - s_m1);
    let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
    let c3 = 1.5 * (s_0 - s_1) + 0.5 * (s_2 - s_m1);
    flush_subnormal_f32_bits(((c3 * t + c2) * t + c1) * t + c0)
}

/// Passive-blended Catmull-Rom reader used inside modulated spatial tank loops
/// (`r_raw = y_lin + half_alpha_cubic * mu * (1 - mu) * y_curv`,
/// `ref[dsp.op-t.reader.catmull-rom]`).
///
/// Setting `half_alpha_cubic = 0.5` yields exact Keys/Catmull-Rom cubic response while `0.0`
/// collapses smoothly to passive linear interpolation for near-unity feedback stability.
#[inline(always)]
#[must_use]
pub fn blend_catmull_rom_modulated_f32(
    s_m1: f32,
    s_0: f32,
    s_1: f32,
    s_2: f32,
    mu: f32,
    half_alpha_cubic: f32,
) -> f32 {
    let t = clamp_finite_unit(mu);
    let omt = 1.0 - t;
    let y_lin = omt * s_0 + t * s_1;
    let ha = if half_alpha_cubic.is_finite() {
        half_alpha_cubic.clamp(0.0, 0.5)
    } else {
        0.0
    };
    let y_curv = omt * (2.0 * s_0 - s_m1 - s_1) + t * (2.0 * s_1 - s_0 - s_2);
    flush_subnormal_f32_bits(y_lin + ha * t * omt * y_curv)
}

/// Computes `half_alpha_cubic = 0.5 * clamp((0.9992 * recipNR2(max(|g_loop_eff|, 0.5)) - 1.0) * 4.0, 0.0, 1.0)`
/// without runtime `FDIV` (`ref[dsp.op-t.reader.catmull-rom]`).
#[inline(always)]
#[must_use]
pub fn compute_half_alpha_cubic_f32(g_loop_eff: f32) -> f32 {
    let mag = if g_loop_eff.is_finite() {
        const_abs_f32(g_loop_eff).max(0.5)
    } else {
        1.0
    };
    let r_inv = GpuVec4::splat(mag).rcp_pos_nr2().to_array()[0];
    let alpha_cubic = ((0.9992 * r_inv - 1.0) * 4.0).clamp(0.0, 1.0);
    0.5 * alpha_cubic
}

/// Reads an `f32` power-of-two delay ring using causal 4-tap Catmull-Rom interpolation.
#[inline(always)]
#[must_use]
pub fn read_catmull_rom_f32(
    ring: &[f32],
    mask: usize,
    write_idx: usize,
    delay_int: usize,
    delay_frac: f32,
) -> f32 {
    if !valid_pow2_ring_len(ring.len(), mask) {
        return 0.0;
    }
    let i_m1 = write_idx.wrapping_add(1).wrapping_sub(delay_int) & mask;
    let i_0 = write_idx.wrapping_sub(delay_int) & mask;
    let i_1 = write_idx.wrapping_sub(delay_int.wrapping_add(1)) & mask;
    let i_2 = write_idx.wrapping_sub(delay_int.wrapping_add(2)) & mask;
    catmull_rom_hermite_f32(ring[i_m1], ring[i_0], ring[i_1], ring[i_2], delay_frac)
}

/// Reads an IEEE-754 `fp16` (`u16`) power-of-two delay ring using causal 4-tap Catmull-Rom
/// interpolation.
#[inline(always)]
#[must_use]
pub fn read_catmull_rom_fp16_u16(
    ring: &[u16],
    mask: usize,
    write_idx: usize,
    delay_int: usize,
    delay_frac: f32,
) -> f32 {
    if !valid_pow2_ring_len(ring.len(), mask) {
        return 0.0;
    }
    let i_m1 = write_idx.wrapping_add(1).wrapping_sub(delay_int) & mask;
    let i_0 = write_idx.wrapping_sub(delay_int) & mask;
    let i_1 = write_idx.wrapping_sub(delay_int.wrapping_add(1)) & mask;
    let i_2 = write_idx.wrapping_sub(delay_int.wrapping_add(2)) & mask;
    catmull_rom_hermite_f32(
        fp16_to_f32(ring[i_m1]),
        fp16_to_f32(ring[i_0]),
        fp16_to_f32(ring[i_1]),
        fp16_to_f32(ring[i_2]),
        delay_frac,
    )
}

/// Branchless `Q0.32` triangle fold `u in [0.0, 1.0]` identical across CPU SIMD and WGSL
/// (`mask = i32(p) >> 31; tri = (p ^ mask) - mask; tri * 2^-31`,
/// `ref[dsp.op-t.window.partition-of-unity]`).
#[inline(always)]
#[must_use]
pub const fn wgsl_triangle_fold_q32(phase_q32: u32) -> f32 {
    let mask = ((phase_q32 as i32) >> 31) as u32;
    let tri = (phase_q32 ^ mask).wrapping_sub(mask);
    (tri as f32) * Q32_INV_2POW31_F32
}

/// Evaluates the zero-at-origin Hermite-5 `Q0.32` sine LFO curve (`0.0` at `0`, `+1.0` at
/// `0x4000_0000`, `0.0` at `0x8000_0000`, `-1.0` at `0xC000_0000`,
/// `ref[dsp.op-t.ring.q0-32-pointer]`).
#[inline(always)]
#[must_use]
pub fn q32_smooth_sin_f32(phase_q32: u32) -> f32 {
    eval_q32_quadrature_f32(phase_q32).1 + 0.0
}

/// Evaluates one lane of the continuous orthogonal `Op-T` Hermite-Hann diagonal window gain
/// `w_k = (1 - mu_win) + mu_win * u^2 * (3 - 2u)` (`ref[dsp.op-t.window.partition-of-unity]`).
#[inline(always)]
#[must_use]
pub fn eval_op_t_diagonal_window_f32(phase_q32: u32, mu_win: f32) -> f32 {
    let u = wgsl_triangle_fold_q32(phase_q32);
    let h = u * u * (3.0 - 2.0 * u);
    let mu = clamp_finite_unit(mu_win);
    (1.0 - mu) + mu * h
}

/// Evaluates all 4 lanes of the continuous orthogonal `Op-T` Hermite-Hann diagonal window in
/// [`KernelVec4`] (`ref[dsp.op-t.window.partition-of-unity]`).
#[inline(always)]
#[must_use]
pub fn eval_op_t_diagonal_window_vec4<V: KernelVec4>(phases_q32: [u32; 4], mu_win: V) -> V {
    let u = V::from_array([
        wgsl_triangle_fold_q32(phases_q32[0]),
        wgsl_triangle_fold_q32(phases_q32[1]),
        wgsl_triangle_fold_q32(phases_q32[2]),
        wgsl_triangle_fold_q32(phases_q32[3]),
    ]);
    let u2 = u.mul(u);
    let h = u2.mul(V::splat(3.0).fms(V::splat(2.0), u));
    let mu = mu_win.clamp(V::zero(), V::splat(1.0));
    V::splat(1.0).sub(mu).fma(mu, h)
}

/// Computes the per-lane boundary-click-clamped window blend `mu_win,k`:
/// `max(base, clamp((|S_k| - 0.25) * 4.0, 0.0, 1.0))` where `S_k = span_samples` when
/// `|ramp_rate_hz| > 1e-6`, else `0.0` (`ref[dsp.op-t.window.partition-of-unity]`).
#[inline(always)]
#[must_use]
pub fn compute_op_t_mu_win_f32(window_depth: f32, span_samples: f32, ramp_rate_hz: f32) -> f32 {
    let base = clamp_finite_unit(window_depth);
    let abs_span = if ramp_rate_hz.is_finite()
        && const_abs_f32(ramp_rate_hz) > 1.0e-6
        && span_samples.is_finite()
    {
        const_abs_f32(span_samples)
    } else {
        0.0
    };
    let click_guard =
        ((abs_span - BOUNDARY_CLICK_OFFSET_SAMPLES) * BOUNDARY_CLICK_SLOPE).clamp(0.0, 1.0);
    base.max(click_guard)
}

/// Computes pitch-down-protected effective tank loop gain `g_loop,eff` using signed `Q0.32`
/// ramp increment `ramp_inc_q32` (positive = increasing delay = pitch-down) and scalar unfused
/// 2-step Newton `rsqrt` with zero `FSQRT` or `FDIV` (`ref[dsp.op-t.window.partition-of-unity]`).
///
/// Negative `ramp_inc_q32` (pitch-up shimmer) yields `nu_down == 0.0`, preserving `g_loop`
/// (clamped to `TANK_MAX_G_LIN_F32`) without attenuation.
#[inline(always)]
#[must_use]
pub fn compute_op_t_pitch_down_g_loop_eff(
    g_loop: f32,
    ramp_spans_samples: [f32; 4],
    ramp_inc_q32: i32,
) -> f32 {
    let g_base = if g_loop.is_finite() {
        g_loop.clamp(0.0, TANK_MAX_G_LIN_F32)
    } else {
        0.0
    };
    let ramp_frac = (ramp_inc_q32 as f32) * Q32_INV_2POW32_F32;
    let mut max_nu = 0.0_f32;
    for &span in &ramp_spans_samples {
        let s = if span.is_finite() { span } else { 0.0 };
        let nu = (s * ramp_frac).clamp(0.0, MAX_PITCH_DOWN_NU);
        max_nu = max_nu.max(nu);
    }
    if max_nu <= 0.0 {
        return g_base;
    }
    let u = (1.0 - max_nu.min(MAX_PITCH_DOWN_U_SUB)).max(0.0);
    let u_safe = u.max(PITCH_DOWN_NR_EPS);
    let h = 0.5 * u_safe;
    let y0 = f32::from_bits(IEEE754_F32_RSQRT_SEED.wrapping_sub(u_safe.to_bits() >> 1));
    let y1 = y0 * (1.5 - h * (y0 * y0));
    let y2 = y1 * (1.5 - h * (y1 * y1));
    let sqrt_u = if u <= 0.0 { 0.0 } else { u * y2 };
    (g_base * sqrt_u).clamp(0.0, TANK_MAX_G_LIN_F32)
}

/// Evaluates the centered `M = 2` Hermite-Hann dual-grain partition-of-unity windows
/// `[H(phi), 1 - H(phi)]`.
#[inline(always)]
#[must_use]
pub fn hermite_hann_partition_of_unity_f32(phase: f32) -> [f32; 2] {
    let p = frac_unit_f32(phase);
    let u = (1.0 - const_abs_f32(2.0 * p - 1.0)).clamp(0.0, 1.0);
    let w0 = u * u * (3.0 - 2.0 * u);
    [w0, 1.0 - w0]
}

/// Evaluates the 3-phase quadratic B-spline partition-of-unity weights for endless `M = 3`
/// barberpole sweeps.
#[inline(always)]
#[must_use]
pub fn bspline_barberpole_3phase_f32(phase: f32) -> [f32; 3] {
    let p0 = frac_unit_f32(phase);
    let p1 = frac_unit_f32(p0 + INV_3);
    let p2 = frac_unit_f32(p0 + TWO_THIRDS);
    [
        quadratic_bspline_tap_f32(p0),
        quadratic_bspline_tap_f32(p1),
        quadratic_bspline_tap_f32(p2),
    ]
}

/// Reads a centered `M = 2` Hermite-Hann dual-grain pitch-shift tap pair from an `f32` delay ring.
#[inline(always)]
#[must_use]
pub fn read_pitch_grain_m2_f32(
    ring: &[f32],
    mask: usize,
    write_idx: usize,
    base_delay: f32,
    grain_span_samples: f32,
    grain_phase: f32,
) -> f32 {
    if !valid_pow2_ring_len(ring.len(), mask) {
        return 0.0;
    }
    let p0 = frac_unit_f32(grain_phase);
    let p1 = frac_unit_f32(p0 + 0.5);
    let [w0, w1] = hermite_hann_partition_of_unity_f32(p0);
    let base = if base_delay.is_finite() {
        base_delay
    } else {
        0.0
    };
    let span = if grain_span_samples.is_finite() {
        grain_span_samples
    } else {
        0.0
    };
    let (d0_int, d0_frac) = split_nonneg_delay_f32(base + span * (p0 - 0.5), mask);
    let (d1_int, d1_frac) = split_nonneg_delay_f32(base + span * (p1 - 0.5), mask);
    let y0 = read_catmull_rom_f32(ring, mask, write_idx, d0_int, d0_frac);
    let y1 = read_catmull_rom_f32(ring, mask, write_idx, d1_int, d1_frac);
    flush_subnormal_f32_bits(w0 * y0 + w1 * y1)
}

/// Reads an endless `M = 3` 3-phase quadratic B-spline barberpole sweep from an `f32` delay ring.
#[inline(always)]
#[must_use]
pub fn read_barberpole_m3_f32(
    ring: &[f32],
    mask: usize,
    write_idx: usize,
    base_delay: f32,
    sweep_span_samples: f32,
    phase: f32,
) -> f32 {
    if !valid_pow2_ring_len(ring.len(), mask) {
        return 0.0;
    }
    let p0 = frac_unit_f32(phase);
    let p1 = frac_unit_f32(p0 + INV_3);
    let p2 = frac_unit_f32(p0 + TWO_THIRDS);
    let w = bspline_barberpole_3phase_f32(p0);
    let base = if base_delay.is_finite() {
        base_delay
    } else {
        0.0
    };
    let span = if sweep_span_samples.is_finite() {
        sweep_span_samples
    } else {
        0.0
    };
    let (d0_int, d0_frac) = split_nonneg_delay_f32(base + span * p0, mask);
    let (d1_int, d1_frac) = split_nonneg_delay_f32(base + span * p1, mask);
    let (d2_int, d2_frac) = split_nonneg_delay_f32(base + span * p2, mask);
    let y0 = read_catmull_rom_f32(ring, mask, write_idx, d0_int, d0_frac);
    let y1 = read_catmull_rom_f32(ring, mask, write_idx, d1_int, d1_frac);
    let y2 = read_catmull_rom_f32(ring, mask, write_idx, d2_int, d2_frac);
    flush_subnormal_f32_bits((w[0] * y0 + w[1] * y1) + w[2] * y2)
}

/// Clamps a commanded `Op-T` transport delay (in samples) to `[4.0, ring_capacity - margin]`.
#[inline(always)]
#[must_use]
pub fn clamp_transport_delay_f32(
    delay_samples: f32,
    ring_capacity: usize,
    lfo_modulated: bool,
) -> f32 {
    let tail_margin = if lfo_modulated { 3 } else { 2 };
    let min_cap = 4 + tail_margin;
    if !delay_samples.is_finite() || ring_capacity < min_cap {
        return MIN_CAUSAL_DELAY_SAMPLES;
    }
    delay_samples.clamp(
        MIN_CAUSAL_DELAY_SAMPLES,
        (ring_capacity - tail_margin) as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{TANK_MAX_G_LIN_BITS, f32_to_fp16};
    use proptest::prelude::*;

    #[test]
    fn test_lagrange_and_catmull_rom_endpoints_reproduction_and_fp16() {
        assert_eq!(cubic_lagrange_weights_f32(0.0), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cubic_lagrange_weights_f32(1.0), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(
            cubic_lagrange_weights_f32(0.5),
            [-0.0625, 0.5625, 0.5625, -0.0625]
        );
        assert_eq!(cubic_lagrange_weights_f32(f32::NAN), [0.0, 1.0, 0.0, 0.0]);

        let ring: [f32; 8] = [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0];
        assert_eq!(read_catmull_rom_f32(&ring, 7, 6, 2, 0.0), 64.0);
        assert_eq!(read_catmull_rom_f32(&ring, 7, 6, 2, 1.0), 27.0);
        assert_eq!(read_catmull_rom_f32(&[], 0, 0, 0, 0.5), 0.0);
        assert_eq!(read_catmull_rom_f32(&ring, 5, 0, 0, 0.5), 0.0);
        assert_eq!(read_catmull_rom_f32(&ring, 15, 0, 0, 0.5), 0.0);
        assert_eq!(read_catmull_rom_f32(&ring, usize::MAX, 0, 0, 0.5), 0.0);

        let mut fp16_ring = [0_u16; 8];
        for (i, slot) in fp16_ring.iter_mut().enumerate() {
            *slot = f32_to_fp16((i as f32) * 0.25);
        }
        assert!((read_catmull_rom_fp16_u16(&fp16_ring, 7, 5, 2, 0.5) - 0.625).abs() < 1.0e-5);
        assert_eq!(read_catmull_rom_fp16_u16(&[], 0, 0, 0, 0.5), 0.0);
        assert_eq!(catmull_rom_hermite_f32(1.0, 2.0, 3.0, 4.0, f32::NAN), 2.0);
    }

    #[test]
    fn test_windows_readers_and_clamp_transport_delay() {
        assert_eq!(hermite_hann_partition_of_unity_f32(0.0), [0.0, 1.0]);
        assert_eq!(hermite_hann_partition_of_unity_f32(0.5), [1.0, 0.0]);
        let b_nan = bspline_barberpole_3phase_f32(f32::NAN);
        assert_eq!(b_nan[0], 0.0);
        assert!((b_nan[0] + b_nan[1] + b_nan[2] - 1.0).abs() < 1.0e-6);

        let dc_ring = [0.625_f32; 32];
        assert!(
            (read_pitch_grain_m2_f32(&dc_ring, 31, 20, 12.0, 6.0, 0.37) - 0.625).abs() < 1.0e-6
        );
        assert!(
            (read_barberpole_m3_f32(&dc_ring, 31, 20, 10.0, 8.0, -0.25) - 0.625).abs() < 1.0e-6
        );
        assert_eq!(read_pitch_grain_m2_f32(&[], 0, 0, 4.0, 2.0, 0.2), 0.0);
        assert_eq!(read_barberpole_m3_f32(&[], 0, 0, 4.0, 2.0, 0.2), 0.0);
        assert!(
            (read_pitch_grain_m2_f32(&dc_ring, 31, 20, f32::NAN, f32::NAN, 0.2) - 0.625).abs()
                < 1.0e-6
        );
        assert!(
            (read_barberpole_m3_f32(&dc_ring, 31, 20, f32::NAN, f32::NAN, 0.2) - 0.625).abs()
                < 1.0e-6
        );
        assert_eq!(floor_f32(f32::NAN), 0.0);
        assert_eq!(floor_f32(1.0e9), 1.0e9);
        assert_eq!(frac_unit_f32(f32::NAN), 0.0);
        assert_eq!(frac_unit_f32(-1.0e-20), 0.0);
        assert_eq!(frac_unit_f32(1.0 - 1.0e-20), 0.0);
        assert_eq!(split_nonneg_delay_f32(f32::NAN, 7), (0, 0.0));
        assert_eq!(split_nonneg_delay_f32(2.0 - 1.0e-9, 7), (2, 0.0));
        assert_eq!(clamp_transport_delay_f32(10.0, 5, false), 4.0);
        assert_eq!(clamp_transport_delay_f32(f32::NAN, 128, false), 4.0);
        assert_eq!(clamp_transport_delay_f32(64.5, 128, false), 64.5);
        assert_eq!(clamp_transport_delay_f32(200.0, 128, false), 126.0);
        assert_eq!(clamp_transport_delay_f32(200.0, 128, true), 125.0);
    }

    /// verify[dsp.op-t.window.partition-of-unity]
    /// verify[dsp.op-t.reader.catmull-rom]
    /// verify[dsp.op-t.ring.q0-32-pointer]
    #[test]
    fn test_ortho_q32_window_sin_pitch_down_and_half_alpha() {
        assert_eq!(wgsl_triangle_fold_q32(0), 0.0);
        assert_eq!(wgsl_triangle_fold_q32(0x4000_0000), 0.5);
        assert_eq!(wgsl_triangle_fold_q32(0x8000_0000), 1.0);
        assert_eq!(eval_op_t_diagonal_window_f32(0x1234_5678, 0.0), 1.0);
        let w_vec = eval_op_t_diagonal_window_vec4(
            [0, 0x8000_0000, 0x4000_0000, 0xC000_0000],
            GpuVec4::splat(1.0),
        )
        .to_array();
        assert_eq!(w_vec, [0.0, 1.0, 0.5, 0.5]);

        assert_eq!(q32_smooth_sin_f32(0), 0.0);
        assert_eq!(q32_smooth_sin_f32(0x4000_0000), 1.0);
        assert_eq!(q32_smooth_sin_f32(0x8000_0000), 0.0);
        assert_eq!(q32_smooth_sin_f32(0xC000_0000), -1.0);

        assert_eq!(compute_op_t_mu_win_f32(0.0, 10.0, 0.0), 0.0);
        assert_eq!(compute_op_t_mu_win_f32(0.0, 0.5, 2.0), 1.0);
        assert!((compute_op_t_mu_win_f32(0.2, 0.375, -1.0) - 0.5).abs() < 1.0e-6);

        // Pitch-up shimmer (negative ramp_inc_q32) preserves 0.995 unattenuated and clamps >= 0.9990.
        let g_up = compute_op_t_pitch_down_g_loop_eff(0.995, [1_200.0; 4], -1_000_000);
        assert_eq!(g_up.to_bits(), 0.995_f32.to_bits());
        let g_clamped = compute_op_t_pitch_down_g_loop_eff(0.9995, [1_200.0; 4], -1_000_000);
        assert_eq!(g_clamped.to_bits(), TANK_MAX_G_LIN_BITS);
        // Pitch-down (positive ramp_inc_q32: +0.125 cycles/sample -> nu_down = 0.75) contracts g_loop.
        let inc_half = 0x2000_0000_i32;
        let g_down = compute_op_t_pitch_down_g_loop_eff(0.995, [6.0, 0.0, 0.0, 0.0], inc_half);
        let expected_down = (0.995_f64 * (1.0 - 0.75_f64).sqrt()) as f32;
        assert!((g_down - expected_down).abs() < 1.0e-5, "g_down={g_down}");

        // half_alpha_cubic == 0.5 matches Catmull-Rom Hermite; 0.0 matches linear interpolation.
        assert_eq!(compute_half_alpha_cubic_f32(0.5), 0.5);
        assert!(compute_half_alpha_cubic_f32(1.0) < 1.0e-6);
        let cr = catmull_rom_hermite_f32(-0.5, 0.25, 0.75, -0.25, 0.37);
        let bl = blend_catmull_rom_modulated_f32(-0.5, 0.25, 0.75, -0.25, 0.37, 0.5);
        assert!((cr - bl).abs() < 1.0e-6);
        let lin = blend_catmull_rom_modulated_f32(-0.5, 0.25, 0.75, -0.25, 0.25, 0.0);
        assert!((lin - 0.375).abs() < 1.0e-6);
    }

    proptest! {
        #[test]
        fn prop_cubic_lagrange_reproduces_cubics_and_partitions_unity(
            mu in -0.5_f32..1.5_f32,
            c0 in -4.0_f32..4.0_f32,
            c1 in -4.0_f32..4.0_f32,
            c2 in -2.0_f32..2.0_f32,
            c3 in -1.0_f32..1.0_f32,
        ) {
            let w = cubic_lagrange_weights_f32(mu);
            prop_assert!(((w[0] + w[1] + w[2] + w[3]) - 1.0).abs() < 1.0e-5);
            let t = mu.clamp(0.0, 1.0);
            let eval_poly = |x: f32| c0 + x * (c1 + x * (c2 + x * c3));
            let s = [eval_poly(-1.0), eval_poly(0.0), eval_poly(1.0), eval_poly(2.0)];
            let interp = w[0] * s[0] + w[1] * s[1] + w[2] * s[2] + w[3] * s[3];
            prop_assert!((interp - eval_poly(t)).abs() < 1.0e-4);
        }

        #[test]
        fn prop_partition_of_unity_windows_hold_on_wrapped_phases(
            phase in -25.0_f32..25.0_f32,
            p_q32 in any::<u32>(),
        ) {
            let [h0, h1] = hermite_hann_partition_of_unity_f32(phase);
            prop_assert!(h0 >= 0.0 && h1 >= 0.0 && (h0 + h1 - 1.0).abs() < 1.0e-6);
            let [b0, b1, b2] = bspline_barberpole_3phase_f32(phase);
            prop_assert!(b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0 && (b0 + b1 + b2 - 1.0).abs() < 1.0e-5);

            let wa = eval_op_t_diagonal_window_f32(p_q32, 1.0);
            let wb = eval_op_t_diagonal_window_f32(p_q32.wrapping_add(0x8000_0000), 1.0);
            prop_assert!((wa + wb - 1.0).abs() <= 1.0e-6, "wa={wa}, wb={wb}");
        }

        #[test]
        fn prop_clamp_transport_delay_enforces_causal_ceiling(
            delay in -100.0_f32..300_000.0_f32,
            cap in 0_usize..=262_144_usize,
            lfo_modulated in prop::bool::ANY,
        ) {
            let clamped = clamp_transport_delay_f32(delay, cap, lfo_modulated);
            let margin = if lfo_modulated { 3 } else { 2 };
            prop_assert!(clamped.is_finite() && clamped >= 4.0);
            if cap < 4 + margin {
                prop_assert_eq!(clamped, 4.0);
            } else {
                prop_assert!(clamped <= (cap - margin) as f32);
            }
        }
    }
}
