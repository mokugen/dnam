//! Orthogonal spatial-tank quad feedback, partitioned Schroeder diffusion, dual-latch
//! Write-Enable (`WE`) holding, non-recirculating stereo wet DC blocker, and Family 4 ballistic
//! ducking VCA (`ref[dsp.op-t.diffuser.recurrence]`, `ref[dsp.tank.loop-saturator]`,
//! `ref[dsp.tank.quiet-feedback-knee]`, `ref[dsp.tank.captured-freeze]`,
//! `ref[dsp.op-t.ducking.vca]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`.

use super::interp_and_window::valid_pow2_ring_len;
use crate::simd::{
    DC_BLOCKER_5HZ_48K_F32, KernelVec4, MAX_LOOP_GAIN_F32, TANK_MAX_G_LIN_F32, TANK_MAX_V_CEIL_F32,
    flush_subnormal_f32_bits, scrub_and_flush_f32_bits,
};

/// Spatial tank quad flag bit `0`: Schmitt-latched ControlBus Write-Enable hold (`op_t_we_drive`).
pub const QUAD_FLAG_WE_HOLD_ACTIVE: u8 = 0x01;

/// Spatial tank quad flag bit `1`: quad uses cold external delay storage rather than inline ring.
pub const QUAD_FLAG_HAS_EXT: u8 = 0x02;

/// Spatial tank quad flag bit `2`: host-owned lock bit freezing feedback & writes cleanly.
pub const QUAD_FLAG_HOST_WRITE_LOCK: u8 = 0x04;

/// Spatial tank quad flag bit `3`: partitioned in-buffer Schroeder all-pass stage enabled.
pub const QUAD_FLAG_SCHROEDER_ENABLE: u8 = 0x08;

/// Composite dual-latch Write-Enable hold mask (`WE_HOLD_ACTIVE | HOST_WRITE_LOCK == 0x05`).
pub const QUAD_WE_LOCK_MASK: u8 = QUAD_FLAG_WE_HOLD_ACTIVE | QUAD_FLAG_HOST_WRITE_LOCK;

/// Lower Schmitt trigger threshold (`we_eff < 0.45` sets [`QUAD_FLAG_WE_HOLD_ACTIVE`]).
pub const WE_SCHMITT_HOLD_LOW_F32: f32 = 0.45;

/// Upper Schmitt trigger threshold (`we_eff > 0.55` clears [`QUAD_FLAG_WE_HOLD_ACTIVE`]).
pub const WE_SCHMITT_RELEASE_HIGH_F32: f32 = 0.55;

/// Maximum finite `fp16` magnitude `65,504.0` used to saturate tank loop writebacks.
pub const FP16_FINITE_MAX_F32: f32 = 65_504.0;

/// Maximum bounded Schroeder all-pass feedback magnitude (`|g| <= 0.75`).
const SCHROEDER_MAX_ABS_GAIN: f32 = 0.75;

/// Minimum coercive soft-limit knee floor preventing subnormal or `0 * Inf` singularities.
const LOOP_LIMIT_MIN_SIGMA: f32 = 1.0e-4;

/// Maximum coercive soft-limit ceiling keeping `sigma * inv_sigma` strictly finite.
const LOOP_LIMIT_MAX_SIGMA: f32 = 1.0e12;

/// Compile-time `2 / 3` factor for unity-slope cubic saturators.
const TWO_THIRDS: f32 = 2.0 / 3.0;

/// Tank cubic soft-knee linear & cubic coefficients (`1.5` and `0.5`).
const LOOP_KNEE_THREE_HALVES: f32 = 1.5;
const LOOP_KNEE_HALF: f32 = 0.5;

/// Compile-time scaling factor `2 / (3 * V_ceil)` for `0.985` tank loop saturator (`0 FDIV` @ RT).
const TANK_CEIL_NORM_SCALE_F32: f32 = 2.0 / (3.0 * TANK_MAX_V_CEIL_F32);

/// Maximum runtime 1-pole HF damping pole inside the quad feedback matrix (`[0.0, 0.99995]`).
const TANK_MAX_DAMP_POLE_F32: f32 = 0.999_95;

/// Input magnitude clamp before `t = v * inv_sigma` in `step_op_t_feedback_vec4` so
/// `denom = 1 + t^2 <= 1e36 < 0x7EF3_11C3` and `rcp_pos_nr2` never wraps on extreme finite floats.
const KNEE_MAX_ABS_INPUT_F32: f32 = 1.0e14;

/// Numeric safety ceiling for `g_inj`, `x_tank`, and feedback state vectors preventing `f32::MAX`
/// overflow to `±Inf` (`<= 2.0 * 1e22 << f32::MAX`) while preserving exact `±65,504.0` saturation.
const TANK_MAX_GINJ_ABS_F32: f32 = 1.0e4;
const TANK_MAX_INJECTED_ABS_F32: f32 = 1.0e18;
const TANK_MAX_STATE_ABS_F32: f32 = 1.0e24;

/// Wet & loop DC blocker minimum (`1e-4`) and maximum (`0.999995`, accommodating ZOH poles up to
/// `0.9999^0.125 = 0.9999875` at `384 kHz`).
const WET_DC_MIN_POLE_F32: f32 = 1.0e-4;
const WET_DC_MAX_POLE_F32: f32 = 0.999_995;

/// Family 4 VCA gain & release pole bounds (`[0.05, 2.0]` and `[0.0, 0.99999]`).
const VCA_MAX_POLE_F32: f32 = 0.999_99;
const VCA_MIN_SENS_F32: f32 = -8.0;
const VCA_MAX_SENS_F32: f32 = 8.0;
const VCA_MIN_GAIN_F32: f32 = 0.05;
const VCA_MAX_GAIN_F32: f32 = 2.0;

const _: () = assert!(SCHROEDER_MAX_ABS_GAIN < MAX_LOOP_GAIN_F32);
const _: () = assert!(QUAD_WE_LOCK_MASK == 0x05);
const _: () = assert!(WE_SCHMITT_HOLD_LOW_F32 < WE_SCHMITT_RELEASE_HIGH_F32);

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

/// Shared cubic unity-slope saturator `ceil * clamp(x * norm_scale, -1, 1) * (1.5 - 0.5 * v^2)`
/// clamped within `[-ceil, ceil]` (`ref[dsp.tank.loop-saturator]`).
#[inline(always)]
fn cubic_soft_saturator_vec4<V: KernelVec4>(x: V, ceil: V, norm_scale: V) -> V {
    let v = x.mul(norm_scale).clamp(V::splat(-1.0), V::splat(1.0));
    let v2 = v.mul(v);
    let cubic = V::splat(LOOP_KNEE_THREE_HALVES)
        .fms(V::splat(LOOP_KNEE_HALF), v2)
        .mul(v);
    ceil.mul(cubic).clamp(ceil.neg(), ceil)
}

/// Cohesive 4-lane tank feedback parameters for [`step_op_t_feedback_vec4`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpTFeedbackParams<V: KernelVec4> {
    /// Per-lane input injection weights (`g_inj`).
    pub g_inj: V,
    /// Pitch-down-protected effective loop gain clamped into `[0.0, TANK_MAX_G_LIN_F32]`.
    pub g_loop_eff: f32,
    /// Precomputed reciprocal knee width `1 / sigma` (`sigma in [1e-3, 8.0]`).
    pub inv_sigma: f32,
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

/// Updates only bit 0 (`QUAD_FLAG_WE_HOLD_ACTIVE`) via Schmitt hysteresis `(0.45, 0.55)` while
/// preserving all other bits including `QUAD_FLAG_HOST_WRITE_LOCK` (`op_t_transport.md` §1.1.1).
#[inline(always)]
#[must_use]
pub const fn update_we_schmitt_flag(flags: u8, we_eff: f32) -> u8 {
    if we_eff < WE_SCHMITT_HOLD_LOW_F32 {
        flags | QUAD_FLAG_WE_HOLD_ACTIVE
    } else if we_eff > WE_SCHMITT_RELEASE_HIGH_F32 {
        flags & !QUAD_FLAG_WE_HOLD_ACTIVE
    } else {
        flags
    }
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

/// Writes `val` to a power-of-two `f32` delay ring only when `(flags & QUAD_WE_LOCK_MASK) == 0`,
/// guaranteeing exact `0-ULP` ring hold whenever either `WE_HOLD_ACTIVE` or `HOST_WRITE_LOCK` is
/// latched (`ref[dsp.tank.captured-freeze]`).
#[inline(always)]
pub fn write_ring_with_we_f32(
    ring: &mut [f32],
    mask: usize,
    write_idx: usize,
    val: f32,
    flags: u8,
) {
    if (flags & QUAD_WE_LOCK_MASK) == 0 && valid_pow2_ring_len(ring.len(), mask) {
        ring[write_idx & mask] = flush_subnormal_f32_bits(val);
    }
}

/// Legacy boolean-freeze wrapper delegating to [`write_ring_with_we_f32`].
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
/// `0 FDIV`, `ref[dsp.tank.loop-saturator]`, `ref[dsp.tank.quiet-feedback-knee]`).
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
    let sat = cubic_soft_saturator_vec4(x_finite, sigma, V::splat(TWO_THIRDS).mul(inv_sigma));
    g.mul(sat).flush_subnormal()
}

/// Steps the orthogonal 4-lane spatial tank feedback loop:
/// 0.85/0.15 `NR2` quiet-feedback knee -> `0.985` cubic saturator (`TANK_MAX_V_CEIL_F32`) ->
/// pairwise 2×2 Givens `R_block2(psi)` -> 4×4 Householder `H_4(1)` + `g_inj * [x_L, x_R, x_L, x_R]`
/// -> 1-pole HF damping -> DF-I loop DC blocker -> `±65,504.0` clamp (`op_t_transport.md` §1.1.1).
#[inline(always)]
#[must_use]
pub fn step_op_t_feedback_vec4<V: KernelVec4>(
    v_read: V,
    x_tank_lr: (f32, f32),
    params: &OpTFeedbackParams<V>,
    state: &mut OpTFeedbackState<V>,
) -> V {
    let v_clean = v_read.flush_recursive_state();
    let inv_sigma = clamp_finite_range(params.inv_sigma, 0.0, 1.0e4, 0.0);
    let knee_limit = V::splat(KNEE_MAX_ABS_INPUT_F32);
    let t = v_clean
        .clamp(knee_limit.neg(), knee_limit)
        .mul(V::splat(inv_sigma));
    let denom = V::splat(1.0).fma(t, t);
    let w_inv = denom.rcp_pos_nr2();
    let gain_mod = V::splat(0.85).fma(V::splat(0.15), w_inv);
    let g_eff = clamp_finite_range(params.g_loop_eff, 0.0, TANK_MAX_G_LIN_F32, 0.0);
    let v_knee = v_clean.mul(V::splat(g_eff)).mul(gain_mod);

    let v_loop = cubic_soft_saturator_vec4(
        v_knee,
        V::splat(TANK_MAX_V_CEIL_F32),
        V::splat(TANK_CEIL_NORM_SCALE_F32),
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

    let limit = V::splat(FP16_FINITE_MAX_F32);
    y_dc.clamp(limit.neg(), limit)
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
/// `5 Hz` stereo wet Direct-Form I DC blocker (`op_t_transport.md:165-171`).
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

/// Steps the Family 4 ballistic 1-pole envelope ducking VCA on stereo `(y_wdc_l, y_wdc_r)`
/// driven by sidechain `(x_tank_l, x_tank_r)` (`ref[dsp.op-t.ducking.vca]`).
#[inline(always)]
pub fn step_op_t_ducking_vca_f32(
    y_lr: (f32, f32),
    x_tank_lr: (f32, f32),
    vca_pole: f32,
    vca_sens: f32,
    attack_cap: f32,
    vca_env: &mut f32,
) -> (f32, f32) {
    let xl = scrub_and_flush_f32_bits(x_tank_lr.0);
    let xr = scrub_and_flush_f32_bits(x_tank_lr.1);
    let x_sc = 0.5 * f32::from_bits(xl.to_bits() & 0x7FFF_FFFF)
        + 0.5 * f32::from_bits(xr.to_bits() & 0x7FFF_FFFF);
    let p_vca = clamp_finite_range(vca_pole, 0.0, VCA_MAX_POLE_F32, 0.0);
    let att_cap = clamp_finite_range(attack_cap, 0.0, VCA_MAX_POLE_F32, 0.99);
    let env_prev = scrub_and_flush_f32_bits(*vca_env);
    let alpha = if x_sc > env_prev {
        p_vca.min(att_cap)
    } else {
        p_vca
    };
    let env_next = scrub_and_flush_f32_bits(alpha * env_prev + (1.0 - alpha) * x_sc);
    *vca_env = env_next;

    let sens = clamp_finite_range(vca_sens, VCA_MIN_SENS_F32, VCA_MAX_SENS_F32, 0.0);
    let g_vca = if sens == 0.0 {
        1.0
    } else {
        (1.0 + sens * env_next).clamp(VCA_MIN_GAIN_F32, VCA_MAX_GAIN_F32)
    };
    (
        flush_subnormal_f32_bits(y_lr.0 * g_vca),
        flush_subnormal_f32_bits(y_lr.1 * g_vca),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, fp16_to_f32};
    use proptest::prelude::*;

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

    /// verify[dsp.tank.captured-freeze]
    /// verify[dsp.tank.loop-saturator]
    /// verify[dsp.op-t.ducking.vca]
    #[test]
    fn test_ortho_we_dual_latch_feedback_wet_dc_and_vca() {
        // Schmitt dual-latch preserves HOST_WRITE_LOCK and other bits independently.
        let base_flags = QUAD_FLAG_HAS_EXT | QUAD_FLAG_HOST_WRITE_LOCK | QUAD_FLAG_SCHROEDER_ENABLE;
        assert_eq!(update_we_schmitt_flag(base_flags, 0.50), base_flags);
        let latched = update_we_schmitt_flag(base_flags, 0.44);
        assert_eq!(latched, base_flags | QUAD_FLAG_WE_HOLD_ACTIVE);
        assert_eq!(update_we_schmitt_flag(latched, 0.50), latched);
        assert_eq!(update_we_schmitt_flag(latched, 0.56), base_flags);
        assert_eq!(update_we_schmitt_flag(latched, f32::NAN), latched);

        let mut ring = [3.0_f32; 8];
        write_ring_with_we_f32(&mut ring, 7, 2, -5.0, QUAD_FLAG_WE_HOLD_ACTIVE);
        assert_eq!(ring[2], 3.0);
        write_ring_with_we_f32(&mut ring, 7, 2, -5.0, QUAD_FLAG_HAS_EXT);
        assert_eq!(ring[2], -5.0);

        let mut off16: u16 = 0;
        let lags: [u32; 6] = core::array::from_fn(|_| step_we_hold_lag_u16(&mut off16, 5));
        assert_eq!(lags, [5, 4, 3, 2, 1, 5]);
        let mut stale_off = 9_u32;
        assert_eq!(step_we_hold_lag(&mut stale_off, 5), 5);
        assert_eq!(stale_off, 0);

        // Feedback stability across 24,000 high-amplitude frames + DC blocking + zero HF damp passthrough.
        let params = OpTFeedbackParams {
            g_inj: GpuVec4::splat(0.5),
            g_loop_eff: TANK_MAX_G_LIN_F32,
            inv_sigma: 1.0,
            cos_psi: 0.6,
            sin_psi: 0.8,
            damp_pole_eff: 0.0,
            r_dc: DC_BLOCKER_5HZ_48K_F32,
        };
        let mut fb_st = OpTFeedbackState::default();
        let mut acc = GpuVec4::zero();
        for _ in 0..24_000 {
            acc = step_op_t_feedback_vec4(acc, (10.0, 10.0), &params, &mut fb_st);
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

        // Wet DC rejects sustained constant offset and VCA bypasses at unity when sens == 0.0.
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

        let mut vca_env = 0.0_f32;
        let unity =
            step_op_t_ducking_vca_f32((-0.5, 0.25), (2.0, 2.0), 0.999, 0.0, 0.99, &mut vca_env);
        assert_eq!(unity, (-0.5, 0.25));
        assert!(vca_env > 0.0);
        let ducked =
            step_op_t_ducking_vca_f32((1.0, 1.0), (1.0, 1.0), 0.999, -8.0, 0.5, &mut vca_env);
        assert!(ducked.0 >= VCA_MIN_GAIN_F32 && ducked.0 < 1.0);

        // Sidechain with +/-f32::MAX avoids sum overflow and tracks high rather than flushing 0.0.
        let _ = step_op_t_ducking_vca_f32(
            (1.0, 1.0),
            (f32::MAX, -f32::MAX),
            0.5,
            -1.0,
            0.5,
            &mut vca_env,
        );
        assert!(vca_env.is_finite() && vca_env > 1.0e30);

        // Extreme finite v_read = [f32::MAX, -f32::MAX, 1e20, -1e20] + 384 kHz ZOH r_dc (0.9999875):
        let extreme = GpuVec4::from_array([f32::MAX, -f32::MAX, 1.0e20, -1.0e20]);
        let params_hi = OpTFeedbackParams {
            g_inj: GpuVec4::zero(),
            g_loop_eff: TANK_MAX_G_LIN_F32,
            inv_sigma: 1.0e4,
            cos_psi: 1.0,
            sin_psi: 0.0,
            damp_pole_eff: 0.0,
            r_dc: 0.999_987_5,
        };
        let mut st_hi = OpTFeedbackState::default();
        let out_hi =
            step_op_t_feedback_vec4(extreme, (0.0, 0.0), &params_hi, &mut st_hi).to_array();
        let exp_abs = TANK_MAX_V_CEIL_F32 * (0.5 * (1.0 + 0.999_987_5));
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

        // Zero feedback gain must emit 0.0 (never spurious -0.985 from rcp_pos_nr2 wrap/NaN).
        let params_zero = OpTFeedbackParams {
            g_loop_eff: 0.0,
            damp_pole_eff: 0.5,
            ..params_hi
        };
        let mut st_zero = OpTFeedbackState::default();
        let out_zero =
            step_op_t_feedback_vec4(extreme, (0.0, 0.0), &params_zero, &mut st_zero).to_array();
        assert_eq!(out_zero, [0.0; 4]);

        // Injection `g_inj = 2.0` + consecutive `(±f32::MAX, ∓f32::MAX)` frames + unclamped Givens
        // saturate cleanly to `±65,504.0` instead of overflowing `±Inf` -> `0.0`.
        let params_inj = OpTFeedbackParams {
            g_inj: GpuVec4::splat(2.0),
            g_loop_eff: 0.0,
            inv_sigma: 1.0,
            cos_psi: f32::MAX,
            sin_psi: f32::NEG_INFINITY,
            damp_pole_eff: 0.0,
            r_dc: DC_BLOCKER_5HZ_48K_F32,
        };
        let mut st_inj = OpTFeedbackState::default();
        let out_f0 = step_op_t_feedback_vec4(
            GpuVec4::splat(0.5),
            (f32::MAX, -f32::MAX),
            &params_inj,
            &mut st_inj,
        )
        .to_array();
        let f = FP16_FINITE_MAX_F32;
        assert_eq!(out_f0, [f, -f, f, -f]);
        let out_f1 = step_op_t_feedback_vec4(
            GpuVec4::splat(0.5),
            (-f32::MAX, f32::MAX),
            &params_inj,
            &mut st_inj,
        )
        .to_array();
        assert_eq!(out_f1, [-f, f, -f, f]);

        // Poisoned state (`NaN`/`±Inf`/`±f32::MAX`) and `NaN` `g_inj` recover finite cleanly.
        let mut st_poison = OpTFeedbackState {
            damp_state: GpuVec4::from_array([f32::NAN, f32::INFINITY, f32::MAX, -f32::MAX]),
            dc_x_prev: GpuVec4::from_array([f32::NAN, f32::NEG_INFINITY, f32::MAX, -f32::MAX]),
            dc_y_prev: GpuVec4::from_array([f32::NAN, f32::INFINITY, -f32::MAX, f32::MAX]),
        };
        let params_poison = OpTFeedbackParams {
            g_inj: GpuVec4::from_array([f32::INFINITY, f32::NAN, 1.0, -1.0]),
            damp_pole_eff: 0.5,
            ..params_hi
        };
        let out_poison = step_op_t_feedback_vec4(
            GpuVec4::splat(0.5),
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
    }
}
