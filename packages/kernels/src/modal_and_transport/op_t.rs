//! Causal top-of-frame fractional delay transport (`Op-T`), partition-of-unity granular/barberpole
//! windowing, partitioned Schroeder all-pass diffusion, lossless freeze, and coercive feedback
//! limiting (`ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.op-t.zero-fdiv]`,
//! `ref[repo.code.shared-nostd-kernels]`).
//!
//! All runtime routines are strictly 32-bit, `#![no_std]`, division-free (`0 FDIV` / `0 UDIV`), and
//! `libm`-free (`ref[repo.code.constant-eval-and-derivations]`).

use crate::simd::{KernelVec4, MAX_LOOP_GAIN_F32, flush_subnormal_f32_bits, fp16_to_f32};

/// Exact compile-time reciprocal `1 / 6` for 4-point cubic Lagrange interpolation (`0 FDIV`).
const INV_6: f32 = 1.0 / 6.0;

/// Exact compile-time reciprocal `1 / 3` for 3-phase barberpole windowing (`0 FDIV`).
const INV_3: f32 = 1.0 / 3.0;

/// Exact compile-time `2 / 3` factor for 3-phase offsets and unity-slope Stage IV saturation.
const TWO_THIRDS: f32 = 2.0 / 3.0;

/// Maximum bounded Schroeder all-pass feedback magnitude (`|g| <= 0.75`,
/// `ref[dsp.op-t.diffuser.recurrence]`).
const SCHROEDER_MAX_ABS_GAIN: f32 = 0.75;

/// Binary32 exact-integer threshold `2^23` (`8_388_608.0`), above which all finite `f32` values are
/// exact integers.
const INT_FLOOR_THRESHOLD_F32: f32 = 8_388_608.0;

/// Minimum coercive soft-limit knee floor preventing subnormal or `0 * Inf` singularities.
const LOOP_LIMIT_MIN_SIGMA: f32 = 1.0e-4;

/// Maximum coercive soft-limit ceiling keeping `sigma * inv_sigma` strictly finite.
const LOOP_LIMIT_MAX_SIGMA: f32 = 1.0e12;

/// Stage IV cubic soft-knee linear coefficient `1.5` (`ref[dsp.stage4.loop-saturator]`).
const LOOP_KNEE_THREE_HALVES: f32 = 1.5;

/// Stage IV cubic soft-knee cubic coefficient `0.5` (`ref[dsp.stage4.loop-saturator]`).
const LOOP_KNEE_HALF: f32 = 0.5;

const _: () = assert!(INV_6 > 0.16 && INV_6 < 0.17);
const _: () = assert!(INV_3 > 0.33 && INV_3 < 0.34);
const _: () = assert!(TWO_THIRDS > 0.66 && TWO_THIRDS < 0.67);
const _: () = assert!(SCHROEDER_MAX_ABS_GAIN < MAX_LOOP_GAIN_F32);
const _: () = assert!(LOOP_LIMIT_MIN_SIGMA > 0.0 && LOOP_LIMIT_MIN_SIGMA < LOOP_LIMIT_MAX_SIGMA);

#[inline(always)]
const fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

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
    if abs_f32(x) >= INT_FLOOR_THRESHOLD_F32 {
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
const fn valid_pow2_ring_len(len: usize, mask: usize) -> bool {
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
    let max_delay = mask as f32;
    let clamped = delay.clamp(0.0, max_delay);
    let floored = floor_f32(clamped);
    let d_int = floored as usize;
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
///
/// The returned array `[w_m1, w_0, w_1, w_2]` satisfies `w_m1 + w_0 + w_1 + w_2 == 1.0` and
/// reproduces cubic polynomials `p(k)` at `k = mu` cleanly.
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

/// Reads an `f32` power-of-two delay ring using causal 4-tap Catmull-Rom interpolation
/// (`& mask` indexing, zero `UDIV`/`FDIV`, `ref[dsp.op-t.reader.catmull-rom]`).
///
/// Returns `0.0` without panicking if `ring` is empty or `mask + 1` is not a valid power-of-two
/// prefix within `ring.len()`.
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
/// interpolation (`ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.op-t.guard.fp16-decay]`).
///
/// Returns `0.0` without panicking if `ring` is empty or `mask + 1` is not a valid power-of-two
/// prefix within `ring.len()`.
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

/// Evaluates the centered `M = 2` Hermite-Hann dual-grain partition-of-unity windows
/// `[H(phi), 1 - H(phi)]` (`0 libm`, exact `sum == 1.0`, `ref[dsp.op-t.window.m2-pitch]`).
#[inline(always)]
#[must_use]
pub fn hermite_hann_partition_of_unity_f32(phase: f32) -> [f32; 2] {
    let p = frac_unit_f32(phase);
    let u = (1.0 - abs_f32(2.0 * p - 1.0)).clamp(0.0, 1.0);
    let w0 = u * u * (3.0 - 2.0 * u);
    [w0, 1.0 - w0]
}

/// Evaluates the 3-phase quadratic B-spline partition-of-unity weights for endless `M = 3`
/// barberpole sweeps (`sum == 1.0`, `w_m >= 0.0`, `ref[dsp.op-t.window.m3-barberpole]`).
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

/// Reads a centered `M = 2` Hermite-Hann dual-grain pitch-shift tap pair from an `f32` delay
/// ring (`ref[dsp.op-t.window.m2-pitch]`).
///
/// The two grain heads sweep symmetrically around `base_delay` across signed span
/// `grain_span_samples = nu * G` using phases `phi_0 = frac(grain_phase)` and
/// `phi_1 = frac(phi_0 + 0.5)`.
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

/// Reads an endless `M = 3` 3-phase quadratic B-spline barberpole sweep from an `f32` delay ring
/// (`ref[dsp.op-t.window.m3-barberpole]`).
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

/// Steps a single scalar partitioned Schroeder all-pass diffuser line (`|g| <= 0.75`,
/// `ref[dsp.op-t.diffuser.recurrence]`).
///
/// Executes the division-free coupled recurrence `v = x - a * u`, `y = a * v + u`, storing `v`
/// into `ring[write_idx & mask]` with subnormal flushing. Non-finite inputs are scrubbed to `0.0`.
#[inline(always)]
pub fn step_schroeder_allpass(
    x: f32,
    ring: &mut [f32],
    mask: usize,
    write_idx: usize,
    delay: usize,
    g: f32,
) -> f32 {
    if !valid_pow2_ring_len(ring.len(), mask) {
        return flush_subnormal_f32_bits(if x.is_finite() { x } else { 0.0 });
    }
    let a = if g.is_finite() {
        g.clamp(-SCHROEDER_MAX_ABS_GAIN, SCHROEDER_MAX_ABS_GAIN)
    } else {
        0.0
    };
    let x_clean = if x.is_finite() { x } else { 0.0 };
    let r_idx = write_idx.wrapping_sub(delay) & mask;
    let w_idx = write_idx & mask;
    let delayed = ring[r_idx];
    let v = flush_subnormal_f32_bits(x_clean - a * delayed);
    let y = flush_subnormal_f32_bits(a * v + delayed);
    ring[w_idx] = v;
    y
}

/// Steps 4 parallel Schroeder all-pass diffuser lanes simultaneously in [`KernelVec4`], returning
/// `(y, ring_write)` (`ref[dsp.op-t.diffuser.recurrence]`).
#[inline(always)]
#[must_use]
pub fn step_schroeder_quad_vec4<V: KernelVec4>(x: V, delayed: V, g: V) -> (V, V) {
    let max_g = V::splat(SCHROEDER_MAX_ABS_GAIN);
    let min_g = V::splat(-SCHROEDER_MAX_ABS_GAIN);
    let a = g.clamp(min_g, max_g);
    let ring_write = x.fms(a, delayed).flush_subnormal();
    let y = delayed.fma(a, ring_write).flush_subnormal();
    (y, ring_write)
}

/// Writes `val` to a power-of-two `f32` delay ring only when `freeze == false`
/// (`ref[dsp.stage4.captured-freeze]`).
///
/// Preserves exact `0-ULP` lossless buffer hold during freeze and safely ignores invalid ring/mask
/// combinations.
#[inline(always)]
pub fn write_ring_with_freeze_f32(
    ring: &mut [f32],
    mask: usize,
    write_idx: usize,
    val: f32,
    freeze: bool,
) {
    if !freeze && valid_pow2_ring_len(ring.len(), mask) {
        ring[write_idx & mask] = flush_subnormal_f32_bits(val);
    }
}

/// Applies the Stage IV unity-slope cubic saturator and bounded loop gain (`||G_loop|| < 1.0`,
/// `0 FDIV`, `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).
///
/// Guarantees `|out| <= MAX_LOOP_GAIN_F32 * soft_limit` and contractive small-signal slope
/// `<= max_gain` without any runtime division or branches.
#[inline(always)]
#[must_use]
pub fn coercive_loop_limit_vec4<V: KernelVec4>(x: V, soft_limit: V, max_gain: V) -> V {
    let x_finite = V::select_le(V::splat(-f32::INFINITY), x, x, V::zero());
    let g = max_gain.clamp(V::zero(), V::splat(MAX_LOOP_GAIN_F32));
    let sigma = soft_limit.clamp(
        V::splat(LOOP_LIMIT_MIN_SIGMA),
        V::splat(LOOP_LIMIT_MAX_SIGMA),
    );
    let inv_sigma = sigma.rcp_pos_nr3();
    let v = x_finite
        .mul(V::splat(TWO_THIRDS))
        .mul(inv_sigma)
        .clamp(V::splat(-1.0), V::splat(1.0));
    let v2 = v.mul(v);
    let cubic = V::splat(LOOP_KNEE_THREE_HALVES)
        .fms(V::splat(LOOP_KNEE_HALF), v2)
        .mul(v);
    let sat = sigma.mul(cubic).clamp(sigma.neg(), sigma);
    g.mul(sat).flush_subnormal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, f32_to_fp16};
    use proptest::prelude::*;

    #[test]
    fn test_lagrange_and_catmull_rom_endpoints_reproduction_and_fp16() {
        let w0 = cubic_lagrange_weights_f32(0.0);
        assert_eq!(w0, [0.0, 1.0, 0.0, 0.0]);
        let w1 = cubic_lagrange_weights_f32(1.0);
        assert_eq!(w1, [0.0, 0.0, 1.0, 0.0]);
        let wh = cubic_lagrange_weights_f32(0.5);
        assert_eq!(wh, [-0.0625, 0.5625, 0.5625, -0.0625]);
        assert_eq!(cubic_lagrange_weights_f32(f32::NAN), [0.0, 1.0, 0.0, 0.0]);

        let ring: [f32; 8] = [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0];
        let mask = 7;
        assert_eq!(read_catmull_rom_f32(&ring, mask, 6, 2, 0.0), 64.0);
        assert_eq!(read_catmull_rom_f32(&ring, mask, 6, 2, 1.0), 27.0);
        assert_eq!(read_catmull_rom_f32(&[], 0, 0, 0, 0.5), 0.0);
        assert_eq!(read_catmull_rom_f32(&ring, 5, 0, 0, 0.5), 0.0);
        assert_eq!(read_catmull_rom_f32(&ring, 15, 0, 0, 0.5), 0.0);
        assert_eq!(read_catmull_rom_f32(&ring, usize::MAX, 0, 0, 0.5), 0.0);

        let mut fp16_ring = [0_u16; 8];
        for (i, slot) in fp16_ring.iter_mut().enumerate() {
            *slot = f32_to_fp16((i as f32) * 0.25);
        }
        let interp = read_catmull_rom_fp16_u16(&fp16_ring, 7, 5, 2, 0.5);
        assert!((interp - 0.625).abs() < 1.0e-5);
        assert_eq!(read_catmull_rom_fp16_u16(&[], 0, 0, 0, 0.5), 0.0);
        assert_eq!(catmull_rom_hermite_f32(1.0, 2.0, 3.0, 4.0, f32::NAN), 2.0);
    }

    #[test]
    fn test_windows_readers_schroeder_freeze_and_loop_limiter() {
        let h_start = hermite_hann_partition_of_unity_f32(0.0);
        assert_eq!(h_start, [0.0, 1.0]);
        let h_mid = hermite_hann_partition_of_unity_f32(0.5);
        assert_eq!(h_mid, [1.0, 0.0]);
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

        let mut ap_ring = [0.0_f32; 16];
        let mut energy_in = 0.0_f64;
        let mut energy_out = 0.0_f64;
        for n in 0..512 {
            let x = if n == 0 { 1.0_f32 } else { 0.0_f32 };
            let y = step_schroeder_allpass(x, &mut ap_ring, 15, n, 5, 1.25);
            energy_in += f64::from(x) * f64::from(x);
            energy_out += f64::from(y) * f64::from(y);
        }
        assert!((energy_out - energy_in).abs() < 1.0e-4);
        assert!(step_schroeder_allpass(f32::NAN, &mut ap_ring, 15, 0, 1, f32::NAN).abs() <= 1.0);
        assert_eq!(step_schroeder_allpass(0.25, &mut [], 0, 0, 1, 0.5), 0.25);
        assert_eq!(step_schroeder_allpass(f32::NAN, &mut [], 0, 0, 1, 0.5), 0.0);

        let (yq, wq) = step_schroeder_quad_vec4(
            GpuVec4::from_array([1.0, -1.0, 0.5, 0.0]),
            GpuVec4::from_array([0.2, -0.4, 0.1, 0.8]),
            GpuVec4::from_array([0.5, -0.5, 2.0, -2.0]),
        );
        assert!((wq.to_array()[2] - (0.5 - 0.75 * 0.1)).abs() < 1.0e-6);
        assert!((yq.to_array()[3] - (0.8 - 0.75 * (0.0 + 0.75 * 0.8))).abs() < 1.0e-6);

        let mut freeze_ring = [1.25_f32; 8];
        let before = freeze_ring;
        write_ring_with_freeze_f32(&mut freeze_ring, 7, 3, -99.0, true);
        assert_eq!(freeze_ring.map(f32::to_bits), before.map(f32::to_bits));
        write_ring_with_freeze_f32(&mut freeze_ring, 7, 3, -2.5, false);
        assert_eq!(freeze_ring[3], -2.5);

        let limited = coercive_loop_limit_vec4(
            GpuVec4::from_array([100.0, -100.0, 0.1, f32::NAN]),
            GpuVec4::from_array([0.8, 0.8, 0.8, 0.8]),
            GpuVec4::from_array([1.5, 0.9, 1.0, 0.5]),
        )
        .to_array();
        assert!((limited[0] - MAX_LOOP_GAIN_F32 * 0.8).abs() < 1.0e-5);
        assert!((limited[1] + 0.9 * 0.8).abs() < 1.0e-5);
        assert!(limited[2] > 0.095 && limited[2] < 0.1);
        assert_eq!(limited[3], 0.0);

        assert_eq!(floor_f32(f32::NAN), 0.0);
        assert_eq!(floor_f32(1.0e9), 1.0e9);
        assert_eq!(frac_unit_f32(f32::NAN), 0.0);
        assert_eq!(frac_unit_f32(-1.0e-20), 0.0);
        assert_eq!(frac_unit_f32(1.0 - 1.0e-20), 0.0);
        assert_eq!(split_nonneg_delay_f32(f32::NAN, 7), (0, 0.0));
        assert_eq!(split_nonneg_delay_f32(2.0 - 1.0e-9, 7), (2, 0.0));
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
            let sum = w[0] + w[1] + w[2] + w[3];
            prop_assert!((sum - 1.0).abs() < 1.0e-5);
            let t = mu.clamp(0.0, 1.0);
            let eval_poly = |x: f32| c0 + x * (c1 + x * (c2 + x * c3));
            let s = [
                eval_poly(-1.0),
                eval_poly(0.0),
                eval_poly(1.0),
                eval_poly(2.0),
            ];
            let interp = w[0] * s[0] + w[1] * s[1] + w[2] * s[2] + w[3] * s[3];
            prop_assert!((interp - eval_poly(t)).abs() < 1.0e-4);
        }

        #[test]
        fn prop_partition_of_unity_windows_hold_on_wrapped_phases(phase in -25.0_f32..25.0_f32) {
            let [h0, h1] = hermite_hann_partition_of_unity_f32(phase);
            prop_assert!(h0 >= 0.0 && h1 >= 0.0);
            prop_assert!((h0 + h1 - 1.0).abs() < 1.0e-6);

            let [b0, b1, b2] = bspline_barberpole_3phase_f32(phase);
            prop_assert!(b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0);
            prop_assert!((b0 + b1 + b2 - 1.0).abs() < 1.0e-5);
        }

        #[test]
        fn prop_coercive_loop_limit_is_strictly_passive_and_bounded(
            x in prop::array::uniform4(-100.0_f32..100.0_f32),
            soft_limit in 0.01_f32..4.0_f32,
            max_gain in 0.0_f32..2.0_f32,
        ) {
            let out = coercive_loop_limit_vec4(
                GpuVec4::from_array(x),
                GpuVec4::splat(soft_limit),
                GpuVec4::splat(max_gain),
            )
            .to_array();
            let eff_gain = max_gain.clamp(0.0, MAX_LOOP_GAIN_F32);
            for lane in 0..4 {
                prop_assert!(out[lane].is_finite());
                prop_assert!(out[lane].abs() <= eff_gain * soft_limit + 1.0e-5);
                prop_assert!(out[lane].abs() <= x[lane].abs() + 1.0e-6);
            }
        }
    }
}
