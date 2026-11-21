//! Orthogonal `Op-T` transport-tank quad feedback, unified `sigma_sat` cubic loop saturation, continuous
//! passive `freeze_mix` recirculation + host write-lock holding, partitioned Schroeder diffusion,
//! and non-recirculating stereo wet DC blocker (`ref[dsp.op-t.diffuser.recurrence]`,
//! `ref[dsp.op-t.guard.bounded-history]`, `ref[dsp.op-t.ring.declaration]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`.

use super::interp_and_window::valid_pow2_ring_len;
use crate::simd::{
    DC_BLOCKER_5HZ_48K_F32, KernelVec4, MAX_LOOP_GAIN_F32, TANK_MAX_G_LIN_F32,
    TANK_MAX_SIGMA_SAT_F32, TANK_MAX_V_CEIL_F32, flush_subnormal_f32_bits,
    scrub_and_flush_f32_bits,
};

/// Transport tank (`Op-T`) quad flag bit `1`: quad uses cold external delay storage rather than inline ring
/// (`bit 0` is reserved `0`; `bits 4..7` are reserved `0`).
pub const QUAD_FLAG_HAS_EXT: u8 = 0x02;

/// Transport tank (`Op-T`) quad flag bit `2`: host-owned lock bit freezing feedback & writes cleanly
/// (`bit 0` is reserved `0`).
pub const QUAD_FLAG_HOST_WRITE_LOCK: u8 = 0x04;

/// Transport tank (`Op-T`) quad flag bit `3`: partitioned in-buffer Schroeder all-pass stage enabled.
pub const QUAD_FLAG_SCHROEDER_ENABLE: u8 = 0x08;

/// Maximum finite `fp16` magnitude `65,504.0` used to saturate tank loop writebacks.
pub use crate::simd::FP16_FINITE_MAX_F32;

/// Maximum bounded Schroeder all-pass feedback magnitude (`|g| <= 0.75`).
const SCHROEDER_MAX_ABS_GAIN: f32 = 0.75;

/// Minimum coercive soft-limit knee floor preventing subnormal or `0 * Inf` singularities.
const LOOP_LIMIT_MIN_SIGMA: f32 = 1.0e-4;

/// Maximum coercive soft-limit ceiling keeping `sigma * inv_sigma` strictly finite.
const LOOP_LIMIT_MAX_SIGMA: f32 = 1.0e12;

/// Minimum unified tank loop saturator ceiling `sigma_sat` (`1e-3`).
const MIN_SIGMA_SAT_F32: f32 = 1.0e-3;

/// Compile-time `2 / 3` factor for unity-slope cubic saturators.
const TWO_THIRDS: f32 = 2.0 / 3.0;

/// Default cold-precomputed `2 / (3 * TANK_MAX_SIGMA_SAT_F32)` fallback for `inv_sigma_23`.
const DEFAULT_INV_SIGMA_23_F32: f32 = 2.0 / (3.0 * TANK_MAX_SIGMA_SAT_F32);

/// Defensive upper clamp on `inv_sigma_23` covering `fp16`-quantized `2 / (3 * 1e-3)` headroom.
const MAX_INV_SIGMA_23_F32: f32 = TWO_THIRDS * 1.0e3 + 1.0;

/// Tank cubic soft-knee linear & cubic coefficients (`1.5` and `0.5`).
const LOOP_KNEE_THREE_HALVES: f32 = 1.5;
const LOOP_KNEE_HALF: f32 = 0.5;

/// Maximum runtime 1-pole HF damping pole inside the quad feedback matrix (`[0.0, 0.99995]`).
const TANK_MAX_DAMP_POLE_F32: f32 = 0.999_95;

/// Numeric safety ceiling for `g_inj`, `x_tank`, and feedback state vectors preventing `f32::MAX`
/// overflow to `±Inf` (`<= 2.0 * 1e22 << f32::MAX`) while preserving exact `±65,504.0` saturation.
const TANK_MAX_GINJ_ABS_F32: f32 = 1.0e4;
const TANK_MAX_INJECTED_ABS_F32: f32 = 1.0e18;
const TANK_MAX_STATE_ABS_F32: f32 = 1.0e24;

/// Wet & loop DC blocker minimum (`1e-4`) and maximum (`0.999995`, accommodating ZOH poles up to
/// `0.9999^0.125 = 0.9999875` at `384 kHz`).
const WET_DC_MIN_POLE_F32: f32 = 1.0e-4;
const WET_DC_MAX_POLE_F32: f32 = 0.999_995;

const _: () = assert!(SCHROEDER_MAX_ABS_GAIN < MAX_LOOP_GAIN_F32);
const _: () = assert!(MIN_SIGMA_SAT_F32 > 0.0 && MIN_SIGMA_SAT_F32 < TANK_MAX_SIGMA_SAT_F32);
const _: () =
    assert!(DEFAULT_INV_SIGMA_23_F32 > 0.6 && DEFAULT_INV_SIGMA_23_F32 < MAX_INV_SIGMA_23_F32);
const _: () = assert!(
    (DEFAULT_INV_SIGMA_23_F32 * 1.5 * TANK_MAX_SIGMA_SAT_F32 - 1.0) < 1.0e-6
        && (DEFAULT_INV_SIGMA_23_F32 * 1.5 * TANK_MAX_SIGMA_SAT_F32 - 1.0) > -1.0e-6
);
const _: () =
    assert!((QUAD_FLAG_HAS_EXT | QUAD_FLAG_HOST_WRITE_LOCK | QUAD_FLAG_SCHROEDER_ENABLE) == 0x0E);

#[inline(always)]
fn clamp_finite_range(x: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if x >= lo {
        if x <= hi { x } else { hi }
    } else if x < lo {
        lo
    } else {
        fallback
    }
}

/// Scrubs non-finite lanes (`NaN`/`±Inf`) and clamps magnitudes into `[-max_abs, max_abs]`.
#[inline(always)]
fn scrub_clamp_vec4<V: KernelVec4>(v: V, max_abs: f32) -> V {
    let limit = V::splat(max_abs);
    v.flush_recursive_state().clamp(limit.neg(), limit)
}

/// Shared cubic unity-slope saturator `gain_scale * v * (1.5 - 0.5 * v^2)` with
/// `v = clamp(x * norm_scale, -1, 1)` clamped within `[-clamp_ceil, clamp_ceil]`
/// (`ref[dsp.op-t.guard.bounded-history]`).
#[inline(always)]
fn cubic_soft_saturator_vec4<V: KernelVec4>(
    x: V,
    gain_scale: V,
    clamp_ceil: V,
    norm_scale: V,
) -> V {
    let v = x.mul(norm_scale).clamp(V::splat(-1.0), V::splat(1.0));
    let v2 = v.mul(v);
    let cubic = V::splat(LOOP_KNEE_THREE_HALVES)
        .fms(V::splat(LOOP_KNEE_HALF), v2)
        .mul(v);
    gain_scale.mul(cubic).clamp(clamp_ceil.neg(), clamp_ceil)
}

/// Cohesive 4-lane tank feedback parameters for [`step_op_t_feedback_vec4`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpTFeedbackParams<V: KernelVec4> {
    /// Per-lane input injection weights (`g_inj`).
    pub g_inj: V,
    /// Pitch-down-protected effective loop gain clamped into `[0.0, TANK_MAX_G_LIN_F32]`.
    pub g_loop_eff: f32,
    /// Unified loop saturator ceiling `sigma_sat` clamped into `[1e-3, TANK_MAX_SIGMA_SAT_F32]`.
    pub sigma_sat: f32,
    /// Cold-precomputed `2 / (3 * sigma_sat)` factor eliminating runtime `rcp_pos_nr2`.
    pub inv_sigma_23: f32,
    /// Continuous passive recirculation mix `mu_freeze` in `[0.0, 1.0]`.
    pub mu_freeze: f32,
    /// Cached Givens rotation cosine `cos(psi)`.
    pub cos_psi: f32,
    /// Cached Givens rotation sine `sin(psi)`.
    pub sin_psi: f32,
    /// Effective per-lane 1-pole HF damping pole (`[0.0, 0.99995]`, `0.0` bypasses damping).
    pub damp_pole_eff: f32,
    /// DF-I loop DC blocker pole (`[0.0, 0.999995]`, preserving ZOH poles at up to `384 kHz`).
    pub r_dc: f32,
}

/// Persistent per-quad 4-lane feedback state updated by [`step_op_t_feedback_vec4`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpTFeedbackState<V: KernelVec4> {
    /// Per-lane 1-pole HF damping state (`damp_state`).
    pub damp_state: V,
    /// Loop DF-I DC blocker previous input (`dc_x_prev`).
    pub dc_x_prev: V,
    /// Loop DF-I DC blocker previous output (`dc_y_prev`).
    pub dc_y_prev: V,
}

/// Steps a single scalar partitioned Schroeder all-pass diffuser line (`|g| <= 0.75`).
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
        return scrub_and_flush_f32_bits(x);
    }
    let a = clamp_finite_range(g, -SCHROEDER_MAX_ABS_GAIN, SCHROEDER_MAX_ABS_GAIN, 0.0);
    let x_clean = if x.is_finite() { x } else { 0.0 };
    let r_idx = write_idx.wrapping_sub(delay) & mask;
    let w_idx = write_idx & mask;
    let delayed = ring[r_idx];
    let v = flush_subnormal_f32_bits(x_clean - a * delayed);
    let y = flush_subnormal_f32_bits(a * v + delayed);
    ring[w_idx] = v;
    y
}

/// Steps 4 parallel Schroeder all-pass diffuser lanes simultaneously in [`KernelVec4`].
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

/// Advances one 32-bit `WE` hold offset counter branchlessly and returns `lag_k` in `[1, D_k]`:
/// `lag = select(o < D, D - o, D)`, `o' = select(o + 1 >= D, 0, o + 1)`.
#[inline(always)]
pub fn step_we_hold_lag(offset: &mut u32, delay_rounded: u32) -> u32 {
    let d = if delay_rounded > 0 { delay_rounded } else { 1 };
    let o = *offset;
    let lag = if o < d { d - o } else { d };
    *offset = if o.saturating_add(1) >= d { 0 } else { o + 1 };
    lag
}

/// Compact 16-bit inline-ring variant of [`step_we_hold_lag`].
#[inline(always)]
pub fn step_we_hold_lag_u16(offset: &mut u16, delay_rounded: u32) -> u32 {
    let mut wide = u32::from(*offset);
    let lag = step_we_hold_lag(&mut wide, delay_rounded);
    *offset = wide as u16;
    lag
}

/// Writes `val` to a power-of-two `f32` delay ring only when `(flags & QUAD_FLAG_HOST_WRITE_LOCK) == 0`,
/// guaranteeing exact `0-ULP` ring hold whenever host write lock is active (`ref[dsp.op-t.ring.declaration]`).
#[inline(always)]
pub fn write_ring_with_we_f32(
    ring: &mut [f32],
    mask: usize,
    write_idx: usize,
    val: f32,
    flags: u8,
) {
    if (flags & QUAD_FLAG_HOST_WRITE_LOCK) == 0 && valid_pow2_ring_len(ring.len(), mask) {
        ring[write_idx & mask] = flush_subnormal_f32_bits(val);
    }
}

/// Boolean-freeze wrapper delegating to [`write_ring_with_we_f32`].
#[inline(always)]
pub fn write_ring_with_freeze_f32(
    ring: &mut [f32],
    mask: usize,
    write_idx: usize,
    val: f32,
    freeze: bool,
) {
    let flags = if freeze { QUAD_FLAG_HOST_WRITE_LOCK } else { 0 };
    write_ring_with_we_f32(ring, mask, write_idx, val, flags);
}

/// Applies the tank unity-slope cubic saturator and bounded loop gain (`||G_loop|| < 1.0`,
/// `0 FDIV`, `ref[dsp.op-t.guard.bounded-history]`).
#[inline(always)]
#[must_use]
pub fn coercive_loop_limit_vec4<V: KernelVec4>(x: V, soft_limit: V, max_gain: V) -> V {
    let x_finite = V::select_le(x.abs(), V::splat(f32::MAX), x, V::zero());
    let g = max_gain.clamp(V::zero(), V::splat(MAX_LOOP_GAIN_F32));
    let sigma = soft_limit.clamp(
        V::splat(LOOP_LIMIT_MIN_SIGMA),
        V::splat(LOOP_LIMIT_MAX_SIGMA),
    );
    let inv_sigma = sigma.rcp_pos_nr3();
    let sat =
        cubic_soft_saturator_vec4(x_finite, sigma, sigma, V::splat(TWO_THIRDS).mul(inv_sigma));
    g.mul(sat).flush_subnormal()
}

/// Steps the orthogonal 4-lane `Op-T` transport tank feedback loop with zero `rcp`/`NR2` calls:
/// unified `sigma_sat` cubic saturator (`TANK_MAX_V_CEIL_F32`) -> pairwise 2×2 Givens
/// `R_block2(psi)` -> 4×4 Householder `H_4(1)` + `g_inj * [x_L, x_R, x_L, x_R]` -> 1-pole HF
/// damping -> DF-I loop DC blocker -> continuous `mu_freeze` writeback blend -> `±65,504.0` clamp
/// (`op_t_transport.md` §1.1.1).
#[inline(always)]
#[must_use]
pub fn step_op_t_feedback_vec4<V: KernelVec4>(
    v_read: V,
    r_win: V,
    x_tank_lr: (f32, f32),
    params: &OpTFeedbackParams<V>,
    state: &mut OpTFeedbackState<V>,
) -> V {
    let v_clean = v_read.flush_recursive_state();
    let r_win_clean = scrub_clamp_vec4(r_win, TANK_MAX_STATE_ABS_F32);
    let sigma_sat = clamp_finite_range(
        params.sigma_sat,
        MIN_SIGMA_SAT_F32,
        TANK_MAX_SIGMA_SAT_F32,
        TANK_MAX_SIGMA_SAT_F32,
    );
    let inv_sigma_23 = clamp_finite_range(
        params.inv_sigma_23,
        0.0,
        MAX_INV_SIGMA_23_F32,
        DEFAULT_INV_SIGMA_23_F32,
    );
    let g_eff = clamp_finite_range(params.g_loop_eff, 0.0, TANK_MAX_G_LIN_F32, 0.0);
    let v_loop = cubic_soft_saturator_vec4(
        v_clean,
        V::splat(g_eff * sigma_sat),
        V::splat(TANK_MAX_V_CEIL_F32),
        V::splat(inv_sigma_23),
    );

    let va = v_loop.to_array();
    let rot_swap = V::from_array([-va[1], va[0], -va[3], va[2]]);
    let c = clamp_finite_range(params.cos_psi, -1.0, 1.0, 1.0);
    let s = clamp_finite_range(params.sin_psi, -1.0, 1.0, 0.0);
    let r_givens = v_loop.mul(V::splat(c)).fma(rot_swap, V::splat(s));

    let ra = r_givens.to_array();
    let half_sum = 0.5 * ((ra[0] + ra[1]) + (ra[2] + ra[3]));
    let h_out = r_givens.sub(V::splat(half_sum));

    let g_inj = scrub_clamp_vec4(params.g_inj, TANK_MAX_GINJ_ABS_F32);
    let x_quad = scrub_clamp_vec4(
        V::from_array([x_tank_lr.0, x_tank_lr.1, x_tank_lr.0, x_tank_lr.1]),
        TANK_MAX_INJECTED_ABS_F32,
    );
    let v_fb = h_out.fma(g_inj, x_quad);

    let p_damp = clamp_finite_range(params.damp_pole_eff, 0.0, TANK_MAX_DAMP_POLE_F32, 0.0);
    let damp_prev = scrub_clamp_vec4(state.damp_state, TANK_MAX_STATE_ABS_F32);
    let v_damp_raw = if p_damp == 0.0 {
        v_fb
    } else {
        V::splat(1.0 - p_damp)
            .mul(v_fb)
            .fma(V::splat(p_damp), damp_prev)
    };
    let v_damp = scrub_clamp_vec4(v_damp_raw, TANK_MAX_STATE_ABS_F32);
    state.damp_state = v_damp;

    let r_dc = clamp_finite_range(
        params.r_dc,
        0.0,
        WET_DC_MAX_POLE_F32,
        DC_BLOCKER_5HZ_48K_F32,
    );
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let dc_x_prev = scrub_clamp_vec4(state.dc_x_prev, TANK_MAX_STATE_ABS_F32);
    let dc_y_prev = scrub_clamp_vec4(state.dc_y_prev, TANK_MAX_STATE_ABS_F32);
    let y_dc = scrub_clamp_vec4(
        V::splat(alpha_dc)
            .mul(v_damp.sub(dc_x_prev))
            .fma(V::splat(r_dc), dc_y_prev),
        TANK_MAX_STATE_ABS_F32,
    );
    state.dc_x_prev = v_damp;
    state.dc_y_prev = y_dc;

    let mu = clamp_finite_range(params.mu_freeze, 0.0, 1.0, 0.0);
    let w_ring = if mu == 0.0 {
        y_dc
    } else {
        V::splat(1.0 - mu).mul(y_dc).fma(V::splat(mu), r_win_clean)
    };
    scrub_clamp_vec4(w_ring, FP16_FINITE_MAX_F32).add(V::zero())
}

#[inline(always)]
fn step_wet_dc_channel(pole: f32, alpha: f32, px: &mut f32, py: &mut f32, input: f32) -> f32 {
    let x_clean = scrub_and_flush_f32_bits(input);
    let x_prev = scrub_and_flush_f32_bits(*px);
    let y_prev = scrub_and_flush_f32_bits(*py);
    let y = scrub_and_flush_f32_bits(alpha * (x_clean - x_prev) + pole * y_prev);
    *px = x_clean;
    *py = y;
    y
}

/// Applies the 2×2 stereo wet Givens rotation `(cos_psi, sin_psi)` and steps the non-recirculating
/// `5 Hz` stereo wet Direct-Form I DC blocker (`op_t_transport.md` §1.1.1).
#[inline(always)]
pub fn step_op_t_wet_dc_stereo_f32(
    y_raw_lr: (f32, f32),
    cos_psi: f32,
    sin_psi: f32,
    wet_dc_pole: f32,
    wet_dc_x: &mut [f32; 2],
    wet_dc_y: &mut [f32; 2],
) -> (f32, f32) {
    let yl_in = scrub_and_flush_f32_bits(y_raw_lr.0);
    let yr_in = scrub_and_flush_f32_bits(y_raw_lr.1);
    let c = if cos_psi.is_finite() { cos_psi } else { 1.0 };
    let s = if sin_psi.is_finite() { sin_psi } else { 0.0 };
    let y_wet_l = c * yl_in - s * yr_in;
    let y_wet_r = s * yl_in + c * yr_in;
    let p_wet = clamp_finite_range(
        wet_dc_pole,
        WET_DC_MIN_POLE_F32,
        WET_DC_MAX_POLE_F32,
        DC_BLOCKER_5HZ_48K_F32,
    );
    let alpha_wet = 0.5 * (1.0 + p_wet);
    let out_l = step_wet_dc_channel(
        p_wet,
        alpha_wet,
        &mut wet_dc_x[0],
        &mut wet_dc_y[0],
        y_wet_l,
    );
    let out_r = step_wet_dc_channel(
        p_wet,
        alpha_wet,
        &mut wet_dc_x[1],
        &mut wet_dc_y[1],
        y_wet_r,
    );
    (out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, fp16_to_f32};
    use proptest::prelude::*;

    /// verify[dsp.op-t.guard.bibo-decay]
    /// verify[dsp.tank.loop-saturator]
    /// verify[dsp.tank.quiet-feedback-knee]
    #[test]
    fn test_schroeder_freeze_and_coercive_limiter() {
        assert_eq!(fp16_to_f32(0x7BFF), FP16_FINITE_MAX_F32);
        let mut ap_ring = [0.0_f32; 16];
        let (mut energy_in, mut energy_out) = (0.0_f64, 0.0_f64);
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
    }

    /// verify[dsp.op-t.ring.declaration]
    /// verify[dsp.op-t.guard.bounded-history]
    /// verify[dsp.op-t.guard.max-tree]
    /// verify[dsp.tank.captured-freeze]
    /// verify[dsp.tank.dc-blocker]
    /// verify[dsp.tank.feedback-loop]
    #[test]
    fn test_host_lock_sigma_sat_mu_freeze_and_wet_dc() {
        let mut ring = [3.0_f32; 8];
        write_ring_with_we_f32(&mut ring, 7, 2, -5.0, QUAD_FLAG_HOST_WRITE_LOCK);
        assert_eq!(ring[2], 3.0);
        write_ring_with_we_f32(
            &mut ring,
            7,
            2,
            -5.0,
            QUAD_FLAG_HAS_EXT | QUAD_FLAG_SCHROEDER_ENABLE | 0x01,
        );
        assert_eq!(ring[2], -5.0);

        let mut off16: u16 = 0;
        let lags: [u32; 6] = core::array::from_fn(|_| step_we_hold_lag_u16(&mut off16, 5));
        assert_eq!(lags, [5, 4, 3, 2, 1, 5]);
        let mut stale_off = 9_u32;
        assert_eq!(step_we_hold_lag(&mut stale_off, 5), 5);
        assert_eq!(stale_off, 0);

        // Unified sigma_sat saturator preserves small-signal slope g_eff and caps at g_eff * sigma_sat <= 0.985.
        for sigma in [TANK_MAX_SIGMA_SAT_F32, 0.25] {
            let g_eff = 0.85_f32;
            let p_lim = OpTFeedbackParams {
                g_inj: GpuVec4::zero(),
                g_loop_eff: g_eff,
                sigma_sat: sigma,
                inv_sigma_23: 2.0 / (3.0 * sigma),
                mu_freeze: 0.0,
                cos_psi: 1.0,
                sin_psi: 0.0,
                damp_pole_eff: 0.0,
                r_dc: DC_BLOCKER_5HZ_48K_F32,
            };
            let alpha_dc = 0.5 * (1.0 + p_lim.r_dc);
            let mut st_small = OpTFeedbackState::default();
            let y_small = step_op_t_feedback_vec4(
                GpuVec4::from_array([1.0e-3, -1.0e-3, 1.0e-3, -1.0e-3]),
                GpuVec4::zero(),
                (0.0, 0.0),
                &p_lim,
                &mut st_small,
            )
            .to_array();
            let slope = y_small[0] / (alpha_dc * 1.0e-3);
            assert!(
                (slope - g_eff).abs() < 1.0e-4,
                "sigma={sigma} slope={slope}"
            );

            let mut st_sat = OpTFeedbackState::default();
            let y_sat = step_op_t_feedback_vec4(
                GpuVec4::from_array([100.0, -100.0, 100.0, -100.0]),
                GpuVec4::zero(),
                (0.0, 0.0),
                &p_lim,
                &mut st_sat,
            )
            .to_array();
            let v_loop_sat = y_sat[0] / alpha_dc;
            assert!((v_loop_sat - g_eff * sigma).abs() < 1.0e-6);
            assert!(v_loop_sat <= TANK_MAX_V_CEIL_F32);
        }

        // Feedback stability across 24,000 high-amplitude frames + DC blocking + zero HF damp passthrough.
        let params = OpTFeedbackParams {
            g_inj: GpuVec4::splat(0.5),
            g_loop_eff: TANK_MAX_G_LIN_F32,
            sigma_sat: TANK_MAX_SIGMA_SAT_F32,
            inv_sigma_23: DEFAULT_INV_SIGMA_23_F32,
            mu_freeze: 0.0,
            cos_psi: 0.6,
            sin_psi: 0.8,
            damp_pole_eff: 0.0,
            r_dc: DC_BLOCKER_5HZ_48K_F32,
        };
        let mut fb_st = OpTFeedbackState::default();
        let mut acc = GpuVec4::zero();
        for _ in 0..24_000 {
            acc = step_op_t_feedback_vec4(acc, GpuVec4::zero(), (10.0, 10.0), &params, &mut fb_st);
            for lane in acc.to_array() {
                assert!(lane.is_finite() && lane.abs() <= FP16_FINITE_MAX_F32);
            }
        }
        for lane in acc.to_array() {
            assert!(lane.abs() < 1.0e-3, "DC blocker residual = {lane}");
        }
        for lane in fb_st.damp_state.to_array() {
            assert!((lane - 5.0).abs() < 1.0e-2, "damp_state = {lane}");
        }

        // Continuous mu_freeze writeback blend (1 - mu)*y_dc + mu*r_win + NaN/extreme scrubbing.
        let r_win = GpuVec4::from_array([-0.5, 1.25, -4.0, 65_504.0]);
        let mut st_mu0 = OpTFeedbackState::default();
        let y0 = step_op_t_feedback_vec4(
            GpuVec4::from_array([0.2, -0.3, 0.4, -0.1]),
            GpuVec4::from_array([f32::MAX, f32::NAN, f32::INFINITY, -f32::MAX]),
            (0.3, -0.2),
            &params,
            &mut st_mu0,
        )
        .to_array();
        let mut st_nan = OpTFeedbackState::default();
        let y_nan_mu = step_op_t_feedback_vec4(
            GpuVec4::from_array([0.2, -0.3, 0.4, -0.1]),
            r_win,
            (0.3, -0.2),
            &OpTFeedbackParams {
                mu_freeze: f32::NAN,
                ..params
            },
            &mut st_nan,
        )
        .to_array();
        assert_eq!(y_nan_mu.map(f32::to_bits), y0.map(f32::to_bits));

        let mut st_mu_mid = OpTFeedbackState::default();
        let y_mid = step_op_t_feedback_vec4(
            GpuVec4::from_array([0.2, -0.3, 0.4, -0.1]),
            r_win,
            (0.3, -0.2),
            &OpTFeedbackParams {
                mu_freeze: 0.25,
                ..params
            },
            &mut st_mu_mid,
        )
        .to_array();
        for (idx, lane) in y_mid.into_iter().enumerate() {
            let exp = 0.75 * y0[idx] + 0.25 * r_win.to_array()[idx];
            assert!(
                (lane - exp).abs() < 1.0e-3,
                "idx={idx} lane={lane} exp={exp}"
            );
        }

        let mut st_mu1 = OpTFeedbackState::default();
        let y_mu1 = step_op_t_feedback_vec4(
            GpuVec4::from_array([0.2, -0.3, 0.4, -0.1]),
            r_win,
            (0.3, -0.2),
            &OpTFeedbackParams {
                mu_freeze: 1.5,
                ..params
            },
            &mut st_mu1,
        )
        .to_array();
        assert_eq!(y_mu1.map(f32::to_bits), r_win.to_array().map(f32::to_bits));

        let mut st_scrub = OpTFeedbackState::default();
        let y_scrub = step_op_t_feedback_vec4(
            GpuVec4::splat(0.1),
            GpuVec4::from_array([-0.0, f32::NAN, f32::MAX, -f32::MAX]),
            (0.0, 0.0),
            &OpTFeedbackParams {
                mu_freeze: 1.0,
                ..params
            },
            &mut st_scrub,
        )
        .to_array();
        assert_eq!(y_scrub[0].to_bits(), 0);
        assert_eq!(y_scrub[1].to_bits(), 0);
        assert_eq!(y_scrub[2], FP16_FINITE_MAX_F32);
        assert_eq!(y_scrub[3], -FP16_FINITE_MAX_F32);

        // Wet DC rejects sustained constant offset cleanly.
        let (mut wx, mut wy) = ([0.0_f32; 2], [0.0_f32; 2]);
        let mut wet_out = (0.0, 0.0);
        for _ in 0..24_000 {
            wet_out = step_op_t_wet_dc_stereo_f32(
                (1.0, 0.0),
                0.0,
                1.0,
                DC_BLOCKER_5HZ_48K_F32,
                &mut wx,
                &mut wy,
            );
        }
        assert!(wet_out.0.abs() < 1.0e-5 && wet_out.1.abs() < 1.0e-5);
    }

    /// verify[dsp.tank.feedback-loop]
    /// verify[dsp.tank.loop-saturator]
    #[test]
    fn test_feedback_extreme_inputs_zero_gain_and_poison_recovery() {
        // Extreme finite v_read = [f32::MAX, -f32::MAX, 1e20, -1e20] + 384 kHz ZOH r_dc (0.9999875):
        let extreme = GpuVec4::from_array([f32::MAX, -f32::MAX, 1.0e20, -1.0e20]);
        let params_hi = OpTFeedbackParams {
            g_inj: GpuVec4::zero(),
            g_loop_eff: TANK_MAX_G_LIN_F32,
            sigma_sat: 2.0,
            inv_sigma_23: DEFAULT_INV_SIGMA_23_F32,
            mu_freeze: 0.0,
            cos_psi: 1.0,
            sin_psi: 0.0,
            damp_pole_eff: 0.0,
            r_dc: 0.999_987_5,
        };
        let mut st_hi = OpTFeedbackState::default();
        let out_hi =
            step_op_t_feedback_vec4(extreme, GpuVec4::zero(), (0.0, 0.0), &params_hi, &mut st_hi)
                .to_array();
        let exp_abs = (TANK_MAX_G_LIN_F32 * TANK_MAX_SIGMA_SAT_F32) * (0.5 * (1.0 + 0.999_987_5));
        for (idx, exp) in [exp_abs, -exp_abs, exp_abs, -exp_abs]
            .into_iter()
            .enumerate()
        {
            assert!(
                (out_hi[idx] - exp).abs() < 1.0e-6,
                "out_hi[{idx}] = {}",
                out_hi[idx]
            );
        }

        // Zero feedback gain emits canonical +0.0 without spurious wrap/NaN.
        let params_zero = OpTFeedbackParams {
            g_loop_eff: 0.0,
            damp_pole_eff: 0.5,
            ..params_hi
        };
        let mut st_zero = OpTFeedbackState::default();
        let out_zero = step_op_t_feedback_vec4(
            extreme,
            GpuVec4::from_array([f32::NAN, f32::MAX, -f32::INFINITY, -0.0]),
            (0.0, 0.0),
            &params_zero,
            &mut st_zero,
        )
        .to_array();
        assert_eq!(out_zero.map(f32::to_bits), [0; 4]);

        // Injection `g_inj = 2.0` + consecutive `(±f32::MAX, ∓f32::MAX)` + unclamped Givens
        // saturate cleanly to `±65,504.0` instead of overflowing `±Inf` -> `0.0`.
        let params_inj = OpTFeedbackParams {
            g_inj: GpuVec4::splat(2.0),
            g_loop_eff: 0.0,
            sigma_sat: TANK_MAX_SIGMA_SAT_F32,
            inv_sigma_23: DEFAULT_INV_SIGMA_23_F32,
            mu_freeze: 0.0,
            cos_psi: f32::MAX,
            sin_psi: f32::NEG_INFINITY,
            damp_pole_eff: 0.0,
            r_dc: DC_BLOCKER_5HZ_48K_F32,
        };
        let mut st_inj = OpTFeedbackState::default();
        let out_f0 = step_op_t_feedback_vec4(
            GpuVec4::splat(0.5),
            GpuVec4::zero(),
            (f32::MAX, -f32::MAX),
            &params_inj,
            &mut st_inj,
        )
        .to_array();
        let f = FP16_FINITE_MAX_F32;
        assert_eq!(out_f0, [f, -f, f, -f]);
        let out_f1 = step_op_t_feedback_vec4(
            GpuVec4::splat(0.5),
            GpuVec4::zero(),
            (-f32::MAX, f32::MAX),
            &params_inj,
            &mut st_inj,
        )
        .to_array();
        assert_eq!(out_f1, [-f, f, -f, f]);

        // Poisoned state (`NaN`/`±Inf`/`±f32::MAX`) and `NaN` fields recover finite cleanly.
        let mut st_poison = OpTFeedbackState {
            damp_state: GpuVec4::from_array([f32::NAN, f32::INFINITY, f32::MAX, -f32::MAX]),
            dc_x_prev: GpuVec4::from_array([f32::NAN, f32::NEG_INFINITY, f32::MAX, -f32::MAX]),
            dc_y_prev: GpuVec4::from_array([f32::NAN, f32::INFINITY, -f32::MAX, f32::MAX]),
        };
        let params_poison = OpTFeedbackParams {
            g_inj: GpuVec4::from_array([f32::INFINITY, f32::NAN, 1.0, -1.0]),
            sigma_sat: f32::NAN,
            inv_sigma_23: f32::NAN,
            mu_freeze: f32::NEG_INFINITY,
            damp_pole_eff: 0.5,
            ..params_hi
        };
        let out_poison = step_op_t_feedback_vec4(
            GpuVec4::splat(0.5),
            GpuVec4::splat(f32::NAN),
            (1.0, -1.0),
            &params_poison,
            &mut st_poison,
        )
        .to_array();
        for lane in out_poison {
            assert!(lane.is_finite() && lane.abs() <= FP16_FINITE_MAX_F32);
        }
    }

    proptest! {
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

        #[test]
        fn prop_feedback_sigma_sat_and_mu_freeze_remain_bounded_and_convex(
            v_read in prop::array::uniform4(-16.0_f32..16.0_f32),
            r_win in prop::array::uniform4(-16.0_f32..16.0_f32),
            x_l in -4.0_f32..4.0_f32,
            x_r in -4.0_f32..4.0_f32,
            sigma_sat in 1.0e-3_f32..TANK_MAX_SIGMA_SAT_F32,
            g_loop in 0.0_f32..1.5_f32,
            mu_freeze in 0.0_f32..1.0_f32,
        ) {
            let params0 = OpTFeedbackParams {
                g_inj: GpuVec4::splat(0.5),
                g_loop_eff: g_loop,
                sigma_sat,
                inv_sigma_23: 2.0 / (3.0 * sigma_sat),
                mu_freeze: 0.0,
                cos_psi: 0.8,
                sin_psi: 0.6,
                damp_pole_eff: 0.25,
                r_dc: DC_BLOCKER_5HZ_48K_F32,
            };
            let mut st0 = OpTFeedbackState::default();
            let mut stm = OpTFeedbackState::default();
            let y0 = step_op_t_feedback_vec4(
                GpuVec4::from_array(v_read),
                GpuVec4::from_array(r_win),
                (x_l, x_r),
                &params0,
                &mut st0,
            )
            .to_array();
            let ym = step_op_t_feedback_vec4(
                GpuVec4::from_array(v_read),
                GpuVec4::from_array(r_win),
                (x_l, x_r),
                &OpTFeedbackParams { mu_freeze, ..params0 },
                &mut stm,
            )
            .to_array();
            for lane in 0..4 {
                prop_assert!(ym[lane].is_finite() && ym[lane].abs() <= FP16_FINITE_MAX_F32);
                let exp = (1.0 - mu_freeze) * y0[lane] + mu_freeze * r_win[lane];
                prop_assert!((ym[lane] - exp).abs() < 1.0e-4);
            }
        }
    }
}
