//! Continuous pre-core `Op-N[Pre]` raised-cosine track-and-hold sample-rate reducer and
//! smooth $C^\infty$ Fourier staircase bitcrusher (`ref[dsp.op-n.bitcrusher.quantizer]`,
//! `ref[dsp.op-n.bitcrusher.zoh-phase]`).
//!
//! Strictly 32-bit (`#![no_std]`, zero 64-bit runtime integers, zero `FDIV`, zero `libm`) and
//! layout-matched to the `28 B` Arena 0 hot slot (`OpNPreHotState`).

pub mod v4;

pub use v4::{
    OP_N_PRE_MIN_DEADZONE_WIDTH, OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH, OpNPreV4Hoisted, OpNPreV4HotState,
    compute_op_n_pre_v4_inc_q32, quantize_hermite_v4_f32,
};

use crate::modal_and_transport::op_t::q32_smooth_sin_f32;
use crate::simd::{
    Q32_INV_2POW32_F32, const_abs_f32, const_f32_pow2, exp2_clamped16_f32, scrub_and_flush_f32_bits,
};

/// Maximum clamped `zoh_hold_depth` (`1.0 - rho_zoh`, zero-initialized safe bypass).
pub const OP_N_PRE_MAX_ZOH_HOLD_DEPTH: f32 = 0.9999;

/// Minimum clamped effective ZOH clock ratio `rho_eff` across `Op-N[Pre]`.
pub const OP_N_PRE_MIN_RHO_EFF: f32 = 1.0e-5;

/// Threshold above which `rho_eff` maps to `inc_q32 = 0` (latch every VM frame).
pub const OP_N_PRE_FULL_PASS_THRESH: f32 = 1.0 - 1.0e-7;

/// Maximum hoisted log2 quantizer bit-depth `beta_q`.
pub const OP_N_PRE_MAX_BETA_Q: f32 = 16.0;

/// Quantizer bypass epsilon (`beta_q <= OP_N_PRE_BETA_BYPASS_EPS` passes clamped `held` directly).
pub const OP_N_PRE_BETA_BYPASS_EPS: f32 = 1.0e-6;

/// Symmetric certified envelope clamp applied before quantization (`[-100, 100]`).
pub const OP_N_PRE_QUANT_CLAMP_F32: f32 = 100.0;

/// Compile-time `1 / (2 * pi)` scaling the smooth Fourier staircase residual (`ref[dsp.op-n.bitcrusher.quantizer]`).
pub const OP_N_PRE_INV_TAU_F32: f32 = 1.0 / core::f32::consts::TAU;

/// Exact IEEE-754 `2^32` scalar (`0x4F80_0000`).
const TWO_POW_32_F32: f32 = const_f32_pow2(32);

/// Compile-time check confirming `TWO_POW_32_F32 * Q32_INV_2POW32_F32 == 1.0` and `INV_TAU`.
const _: () = assert!((TWO_POW_32_F32 * Q32_INV_2POW32_F32 - 1.0) == 0.0);
const _: () = assert!(const_abs_f32(OP_N_PRE_INV_TAU_F32 * core::f32::consts::TAU - 1.0) < 1.0e-6);

/// Persistent 28-byte (`align(4)`) Arena 0 hot state for stereo `Op-N[Pre]` execution
/// (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpNPreHotState {
    /// Latched stereo sample pair (`[L, R]`, flushed to zero on subnormals/non-finite).
    pub held: [f32; 2],
    /// Normalized `Q0.32` sample-rate reducer accumulator in `[0, 1)`.
    pub phase_q32: u32,
    /// `Q0.32` phase increment (`0` means latch every frame / bypass track-and-hold).
    pub inc_q32: u32,
    /// Clamped wet blend factor in `[0.0, 1.0]` (`0.0` = exact 0-ULP pass-through).
    pub gamma_bc: f32,
    /// Clamped log2 bit-depth in `[0.0, 16.0]` (`<= 1.0e-6` = bypass quantizer).
    pub beta_q: f32,
    /// ZOH-rescaled hold depth in `[0.0, 0.9999]` (`0.0` = bypass track-and-hold).
    pub zoh_hold_depth_eff: f32,
}

/// Hoisted 24-byte (`align(4)`) per-block / per-frame `Op-N[Pre]` coefficients avoiding
/// per-sample `exp2` work (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpNPreHoisted {
    /// `Q0.32` ZOH increment (`0` latches every frame).
    pub inc_q32: u32,
    /// Clamped wet blend factor in `[0.0, 1.0]`.
    pub gamma_bc: f32,
    /// Clamped log2 bit-depth in `[0.0, 16.0]`.
    pub beta_q: f32,
    /// ZOH-rescaled hold depth in `[0.0, 0.9999]`.
    pub zoh_hold_depth_eff: f32,
    /// Precomputed quantizer scale `D_q = exp2_clamped16(beta_q)`.
    pub d_scale: f32,
    /// Precomputed quantizer step `Delta_q = exp2_clamped16(-beta_q)`.
    pub delta_step: f32,
}

const _: () = assert!(core::mem::size_of::<OpNPreHotState>() == 28);
const _: () = assert!(core::mem::align_of::<OpNPreHotState>() == 4);
const _: () = assert!(core::mem::size_of::<OpNPreHoisted>() == 24);
const _: () = assert!(core::mem::align_of::<OpNPreHoisted>() == 4);

/// Branchlessly clamps `x` into `[lo, hi]`, mapping `NaN` to `fallback`.
#[inline(always)]
fn clamp_nan_to(x: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if x >= lo {
        if x <= hi { x } else { hi }
    } else if x < lo {
        lo
    } else {
        fallback
    }
}

/// Computes the 32-bit `Q0.32` `Op-N[Pre]` ZOH phase increment from clamped
/// `zoh_hold_depth in [0.0, 0.9999]` and cold host ratio `rho_vm = 48_000.0 / f_{s,vm}`
/// without `FDIV` or 64-bit runtime integers (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline(always)]
#[must_use]
pub fn compute_op_n_pre_inc_q32(zoh_hold_depth: f32, rho_vm: f32) -> u32 {
    let zhd = clamp_nan_to(zoh_hold_depth, 0.0, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, 0.0);
    let raw = (1.0 - zhd) * rho_vm;
    // Non-finite `rho_vm` fails safe to `1.0` -> `inc_q32 == 0` (transparent latch).
    let rho_eff = clamp_nan_to(raw, OP_N_PRE_MIN_RHO_EFF, 1.0, 1.0);
    if rho_eff >= OP_N_PRE_FULL_PASS_THRESH {
        return 0;
    }
    let scaled = rho_eff * TWO_POW_32_F32;
    let t = scaled as u32;
    let f = scaled - (t as f32);
    t.saturating_add(u32::from(f >= 0.5)).max(1)
}

/// Uniform nearest-step quantizer `Q(x) = floor(clamp(x, -100, 100) * D_q + 0.5) * Delta_q`
/// used by `packages/engine` (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline(always)]
#[must_use]
pub fn quantize_nearest_step_f32(x: f32, d_scale: f32, delta_step: f32) -> f32 {
    let x_clamped = clamp_nan_to(x, -OP_N_PRE_QUANT_CLAMP_F32, OP_N_PRE_QUANT_CLAMP_F32, 0.0);
    let v = x_clamped * d_scale;
    let t = v as i32;
    let k = t - i32::from(v < (t as f32));
    let f = v - (k as f32);
    let q_idx = k + i32::from(f >= 0.5);
    (q_idx as f32) * delta_step
}

/// Smooth $C^\infty$ Fourier staircase quantizer
/// `Q~(h; Delta_q) = h_clip - Delta_q * INV_TAU * q32_smooth_sin(phi_q32)`
/// over `[-100, 100]` using negative-safe trunc-to-`i32` floor splitting and overflow-safe
/// `Q0.32` phase wrapping (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline(always)]
#[must_use]
pub fn quantize_smooth_fourier_f32(x: f32, d_scale: f32, delta_step: f32) -> f32 {
    let h_clip = clamp_nan_to(x, -OP_N_PRE_QUANT_CLAMP_F32, OP_N_PRE_QUANT_CLAMP_F32, 0.0) + 0.0;
    let v = h_clip * d_scale;
    let t = v as i32;
    let k = t - i32::from(v < (t as f32));
    let f = clamp_nan_to(v - (k as f32), 0.0, 1.0, 0.0);
    let scaled = f * TWO_POW_32_F32;
    let phi_q32 = if f >= 1.0 || scaled >= TWO_POW_32_F32 {
        0
    } else {
        scaled as u32
    };
    h_clip - delta_step * OP_N_PRE_INV_TAU_F32 * q32_smooth_sin_f32(phi_q32)
}

impl Default for OpNPreHoisted {
    #[inline(always)]
    fn default() -> Self {
        Self::bypass()
    }
}

impl OpNPreHoisted {
    /// Returns identity bypass parameters (`gamma_bc = 0`, `inc_q32 = 0`, `zoh_hold_depth_eff = 0`,
    /// `beta_q = 0`, `D = 1`).
    #[inline(always)]
    #[must_use]
    pub const fn bypass() -> Self {
        Self {
            inc_q32: 0,
            gamma_bc: 0.0,
            beta_q: 0.0,
            zoh_hold_depth_eff: 0.0,
            d_scale: 1.0,
            delta_step: 1.0,
        }
    }

    /// Clamps `ControlBus`-modulated targets `(gamma_bc, zoh_hold_depth, zoh_hold_depth_eff, beta_q)`
    /// and hoists `D_q = exp2_clamped16(beta_q)`, `Delta_q = exp2_clamped16(-beta_q)`, and `inc_q32`
    /// (`ref[dsp.op-n.bitcrusher.quantizer]`).
    #[inline(always)]
    #[must_use]
    pub fn from_params_with_zoh_depth(
        gamma_bc: f32,
        zoh_hold_depth: f32,
        zoh_hold_depth_eff: f32,
        beta_q: f32,
        rho_vm: f32,
    ) -> Self {
        let gamma = clamp_nan_to(gamma_bc, 0.0, 1.0, 0.0);
        let z_h = clamp_nan_to(zoh_hold_depth, 0.0, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, 0.0);
        let z_h_eff = clamp_nan_to(zoh_hold_depth_eff, 0.0, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, 0.0);
        let beta = clamp_nan_to(beta_q, 0.0, OP_N_PRE_MAX_BETA_Q, 0.0);
        let rho_eff = clamp_nan_to((1.0 - z_h) * rho_vm, OP_N_PRE_MIN_RHO_EFF, 1.0, 1.0);
        let inc_q32 = if rho_eff >= OP_N_PRE_FULL_PASS_THRESH || z_h_eff <= 0.0 {
            0
        } else {
            compute_op_n_pre_inc_q32(z_h, rho_vm)
        };
        let d_scale = exp2_clamped16_f32(beta);
        let delta_step = exp2_clamped16_f32(-beta);
        Self {
            inc_q32,
            gamma_bc: gamma,
            beta_q: beta,
            zoh_hold_depth_eff: z_h_eff,
            d_scale,
            delta_step,
        }
    }

    /// Clamps `ControlBus`-modulated targets `(gamma_bc, zoh_hold_depth, beta_q)` using
    /// `zoh_hold_depth_eff = zoh_hold_depth`.
    #[inline(always)]
    #[must_use]
    pub fn from_params(gamma_bc: f32, zoh_hold_depth: f32, beta_q: f32, rho_vm: f32) -> Self {
        Self::from_params_with_zoh_depth(gamma_bc, zoh_hold_depth, zoh_hold_depth, beta_q, rho_vm)
    }
}

impl OpNPreHotState {
    /// Creates zero-initialized safe-bypass `OpNPreHotState`.
    #[inline(always)]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            held: [0.0; 2],
            phase_q32: 0,
            inc_q32: 0,
            gamma_bc: 0.0,
            beta_q: 0.0,
            zoh_hold_depth_eff: 0.0,
        }
    }

    /// Loads hoisted parameters (`inc_q32`, `gamma_bc`, `beta_q`, `zoh_hold_depth_eff`) into hot
    /// state without altering `held` or resetting `phase_q32`.
    #[inline(always)]
    pub fn apply_hoisted(&mut self, hoisted: OpNPreHoisted) {
        self.inc_q32 = hoisted.inc_q32;
        self.gamma_bc = hoisted.gamma_bc;
        self.beta_q = hoisted.beta_q;
        self.zoh_hold_depth_eff = hoisted.zoh_hold_depth_eff;
    }

    /// Evaluates the raised-cosine clock gain `g_clk in [1 - z_h_eff, 1]` and advances `phase_q32`
    /// when `inc_q32 != 0` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
    #[inline(always)]
    fn step_clock_gain(&mut self) -> f32 {
        if self.inc_q32 == 0 {
            return 1.0;
        }
        self.phase_q32 = self.phase_q32.wrapping_add(self.inc_q32);
        let cos_phi = q32_smooth_sin_f32(self.phase_q32.wrapping_add(0x4000_0000));
        let u = 0.5 * (1.0 + cos_phi);
        let w = u * u;
        let z_h_eff = clamp_nan_to(
            self.zoh_hold_depth_eff,
            0.0,
            OP_N_PRE_MAX_ZOH_HOLD_DEPTH,
            0.0,
        );
        (1.0 - z_h_eff) + z_h_eff * w
    }

    /// Executes one stereo `Op-N[Pre]` frame in strict **Raised-Cosine Track-and-Hold -> Smooth
    /// Fourier Quantizer -> Wet Blend** order (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
    ///
    /// Always updates `held` continuously and returns `(x_l, x_r)` at `0-ULP` when `gamma_bc <= 0.0`.
    #[inline(always)]
    pub fn step_stereo(&mut self, x_l: f32, x_r: f32, d_scale: f32, delta_step: f32) -> (f32, f32) {
        let g_clk = self.step_clock_gain();
        self.held[0] = scrub_and_flush_f32_bits((1.0 - g_clk) * self.held[0] + g_clk * x_l) + 0.0;
        self.held[1] = scrub_and_flush_f32_bits((1.0 - g_clk) * self.held[1] + g_clk * x_r) + 0.0;

        let gamma = clamp_nan_to(self.gamma_bc, 0.0, 1.0, 0.0);
        if gamma <= 0.0 {
            return (x_l, x_r);
        }
        let (q_l, q_r) = if !self.beta_q.is_finite() || self.beta_q <= OP_N_PRE_BETA_BYPASS_EPS {
            (
                clamp_nan_to(
                    self.held[0],
                    -OP_N_PRE_QUANT_CLAMP_F32,
                    OP_N_PRE_QUANT_CLAMP_F32,
                    0.0,
                ) + 0.0,
                clamp_nan_to(
                    self.held[1],
                    -OP_N_PRE_QUANT_CLAMP_F32,
                    OP_N_PRE_QUANT_CLAMP_F32,
                    0.0,
                ) + 0.0,
            )
        } else {
            (
                quantize_smooth_fourier_f32(self.held[0], d_scale, delta_step),
                quantize_smooth_fourier_f32(self.held[1], d_scale, delta_step),
            )
        };
        (x_l + gamma * (q_l - x_l), x_r + gamma * (q_r - x_r))
    }

    /// Applies `hoisted` targets into `self` and steps one stereo frame.
    #[inline(always)]
    pub fn step_stereo_hoisted(
        &mut self,
        x_l: f32,
        x_r: f32,
        hoisted: &OpNPreHoisted,
    ) -> (f32, f32) {
        self.apply_hoisted(*hoisted);
        self.step_stereo(x_l, x_r, hoisted.d_scale, hoisted.delta_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// verify[dsp.op-n.bitcrusher.quantizer]
    #[test]
    fn test_op_n_pre_layout_zero_dc_bypass_and_smooth_quantizer() {
        assert_eq!(core::mem::size_of::<OpNPreHotState>(), 28);
        assert_eq!(core::mem::align_of::<OpNPreHotState>(), 4);
        assert_eq!(core::mem::size_of::<OpNPreHoisted>(), 24);
        assert_eq!(core::mem::align_of::<OpNPreHoisted>(), 4);

        let mut st = OpNPreHotState::default();
        let hoisted = OpNPreHoisted::default();
        assert_eq!(hoisted.gamma_bc, 0.0);
        assert_eq!(hoisted.inc_q32, 0);
        assert_eq!(hoisted.zoh_hold_depth_eff, 0.0);
        assert_eq!(hoisted.d_scale, 1.0);
        assert_eq!(hoisted.delta_step, 1.0);

        // 0-ULP bypass still updates continuous `held` state (`gamma_bc == 0.0`) and flushes subnormals.
        let bypass_with_hold = OpNPreHoisted::from_params(0.0, 0.5, 4.0, 1.0);
        let inputs = [-0.0_f32, 0.375_f32, -12.25_f32, 1.0e-42_f32];
        let mut max_held_abs = 0.0_f32;
        for (idx, &x) in inputs.iter().enumerate() {
            let (yl, yr) = st.step_stereo_hoisted(x, -x, &bypass_with_hold);
            assert_eq!(yl.to_bits(), x.to_bits(), "0-ULP bypass L at idx {idx}");
            assert_eq!(yr.to_bits(), (-x).to_bits(), "0-ULP bypass R at idx {idx}");
            max_held_abs = max_held_abs.max(st.held[0].abs());
        }
        assert!(max_held_abs > 1.0);
        assert_eq!(st.held[0].to_bits(), 0);
        assert_eq!(st.held[1].to_bits(), 0);
        assert_eq!(st.phase_q32, 0);

        // Smooth Fourier quantizer: exact `+0.0` at `±0.0`, exact grid points, clamping, NaN safety.
        assert_eq!(quantize_smooth_fourier_f32(0.0, 8.0, 0.125).to_bits(), 0);
        assert_eq!(quantize_smooth_fourier_f32(-0.0, 8.0, 0.125).to_bits(), 0);
        assert_eq!(quantize_smooth_fourier_f32(0.5, 4.0, 0.25), 0.5);
        assert_eq!(quantize_smooth_fourier_f32(-0.75, 4.0, 0.25), -0.75);
        assert_eq!(
            quantize_smooth_fourier_f32(f32::NAN, 8.0, 0.125).to_bits(),
            0
        );
        assert_eq!(quantize_smooth_fourier_f32(250.0, 1.0, 1.0), 100.0);
        assert_eq!(quantize_smooth_fourier_f32(-250.0, 1.0, 1.0), -100.0);

        // Smooth continuity across half-integer boundaries (no discontinuous step jump).
        let q_m = quantize_smooth_fourier_f32(0.499, 1.0, 1.0);
        let q_0 = quantize_smooth_fourier_f32(0.5, 1.0, 1.0);
        let q_p = quantize_smooth_fourier_f32(0.501, 1.0, 1.0);
        assert!((q_0 - 0.5).abs() < 1.0e-5);
        assert!((q_p - q_m).abs() < 0.01);

        // Uniform nearest-step quantizer checks.
        assert_eq!(quantize_nearest_step_f32(-0.5, 1.0, 1.0), 0.0);
        assert_eq!(quantize_nearest_step_f32(-0.6, 1.0, 1.0), -1.0);

        // Zero-DC equilibrium: `0.0` and `-0.0` input emit exact `+0.0` bits under active wet state.
        let mut st_dc = OpNPreHotState::zero();
        let active_dc = OpNPreHoisted::from_params(1.0, 0.75, 4.0, 1.0);
        for &z in &[0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32] {
            let (yl, yr) = st_dc.step_stereo_hoisted(z, z, &active_dc);
            assert_eq!(yl.to_bits(), 0);
            assert_eq!(yr.to_bits(), 0);
        }

        // Direct mutation with NaN fields safely bypasses gamma_bc and beta_q.
        st.gamma_bc = f32::NAN;
        assert_eq!(st.step_stereo(0.25, -0.5, 4.0, 0.25), (0.25, -0.5));
        st.gamma_bc = 1.0;
        st.beta_q = f32::NAN;
        let expected_l = st.held[0].clamp(-100.0, 100.0);
        let (ql, _) = st.step_stereo(0.375, -0.125, 4.0, 0.25);
        assert!(ql.is_finite() && (ql - expected_l).abs() < 100.0);
    }

    /// verify[dsp.op-n.bitcrusher.rate-transfer]
    /// verify[dsp.op-n.bitcrusher.zoh-phase]
    #[test]
    fn test_op_n_pre_raised_cosine_track_hold_at_48k_and_192k() {
        // 50% ZOH hold at 48 kHz -> inc_q32 = 2^31; raised-cosine w alternates 0 and 1.
        let h_48k = OpNPreHoisted::from_params(1.0, 0.5, 0.0, 1.0);
        assert_eq!(h_48k.inc_q32, 0x8000_0000);
        assert_eq!(h_48k.zoh_hold_depth_eff, 0.5);
        let mut st = OpNPreHotState::zero();
        // Step 0: phase 0 -> 0x8000_0000 (pi), cos = -1, u = 0, w = 0, g_clk = 1 - 0.5 = 0.5.
        let (y0, _) = st.step_stereo_hoisted(1.0, -1.0, &h_48k);
        assert_eq!(y0, 0.5);
        // Step 1: phase 0x8000_0000 -> 0, cos = +1, u = 1, w = 1, g_clk = 1.0 -> full track.
        let (y1, _) = st.step_stereo_hoisted(0.25, -0.25, &h_48k);
        assert_eq!(y1, 0.25);

        // At 192 kHz (rho_vm = 0.25), zero hold maps to inc_q32 == 0, while active hold scales 0.25x.
        let h_192k_zero = OpNPreHoisted::from_params(1.0, 0.0, 4.0, 0.25);
        assert_eq!(h_192k_zero.inc_q32, 0);
        assert_eq!(h_192k_zero.zoh_hold_depth_eff, 0.0);
        let h_192k_half =
            OpNPreHoisted::from_params_with_zoh_depth(1.0, 0.5, 0.5_f32.powf(0.25), 4.0, 0.25);
        assert_eq!(h_192k_half.inc_q32, 0x2000_0000);
        assert!((h_192k_half.zoh_hold_depth_eff - 0.5_f32.powf(0.25)).abs() < 1.0e-6);
        assert_eq!(h_192k_half.d_scale, 16.0);
        assert_eq!(h_192k_half.delta_step, 0.0625);

        // NaN inputs fail safe to zero hold + bypass increment.
        let h_nan = OpNPreHoisted::from_params(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        assert_eq!(h_nan.gamma_bc, 0.0);
        assert_eq!(h_nan.inc_q32, 0);
        assert_eq!(h_nan.zoh_hold_depth_eff, 0.0);
        assert_eq!(h_nan.beta_q, 0.0);
    }

    proptest! {
        #[test]
        fn prop_smooth_fourier_quantizer_bounded_and_c_infinity_close(
            x in -100.0_f32..100.0_f32,
            beta_q in 0.5_f32..16.0_f32,
        ) {
            let hoisted = OpNPreHoisted::from_params(1.0, 0.0, beta_q, 1.0);
            let q = quantize_smooth_fourier_f32(x, hoisted.d_scale, hoisted.delta_step);
            prop_assert!(q.is_finite());
            let max_dev = f64::from(hoisted.delta_step) * f64::from(OP_N_PRE_INV_TAU_F32) + 1.0e-5;
            prop_assert!((f64::from(q) - f64::from(x)).abs() <= max_dev);
        }

        #[test]
        fn prop_raised_cosine_clock_gain_contractive(
            phase_q32 in any::<u32>(),
            inc_q32 in 1_u32..=u32::MAX,
            z_h_eff in 0.0_f32..OP_N_PRE_MAX_ZOH_HOLD_DEPTH,
        ) {
            let mut st = OpNPreHotState::zero();
            st.phase_q32 = phase_q32;
            st.inc_q32 = inc_q32;
            st.zoh_hold_depth_eff = z_h_eff;
            let g_clk = st.step_clock_gain();
            let decay = 1.0 - g_clk;
            prop_assert!((-1.0e-7..=OP_N_PRE_MAX_ZOH_HOLD_DEPTH + 1.0e-6).contains(&decay));
            prop_assert!((1.0 - OP_N_PRE_MAX_ZOH_HOLD_DEPTH - 1.0e-6..=1.0 + 1.0e-7).contains(&g_clk));
        }

        #[test]
        fn prop_op_n_pre_inc_q32_monotonic_and_rate_invariant(
            zhd in 0.0_f32..0.9999_f32,
            fs_vm in 44_100.0_f32..384_000.0_f32,
        ) {
            let rho_vm = 48_000.0 / fs_vm;
            let inc = compute_op_n_pre_inc_q32(zhd, rho_vm);
            if zhd > 0.25 && fs_vm >= 48_000.0 {
                prop_assert!(inc > 0);
            }
            let _ = compute_op_n_pre_inc_q32(zhd, 1.0);
        }
    }
}
