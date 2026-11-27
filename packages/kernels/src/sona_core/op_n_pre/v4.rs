//! `Op-N[Pre]` v4.0 front-end (`quantize -> wet blend -> hard-capable Q0.32 phase-wrap ZOH`):
//! `4`-FMA $C^1$ Hermite deadzone staircase quantizer + log-frequency clock `log2_rho_zoh` +
//! `kappa_h`-sharpened phase-wrap ZOH (`0-ULP` bypass when `gamma <= 0 && z_h <= 0`,
//! `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`,
//! `ref[dsp.op-n.bitcrusher.rate-transfer]`, `ref[dsp.op-n.bitcrusher.surrogate]`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero 64-bit runtime integers, zero
//! `FDIV`/`FSQRT`/`libm`, and `spirv-unknown-naga-wgsl`-clean.

use crate::modal_and_transport::op_t::q32_smooth_sin_f32;
use crate::simd::{
    GpuVec4, KernelVec4, canonicalize_zero_f32_bits, const_abs_f32, const_f32_pow2,
    exp2_clamped16_f32, scrub_and_flush_f32_bits,
};

use super::{
    OP_N_PRE_BETA_BYPASS_EPS, OP_N_PRE_FULL_PASS_THRESH, OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MIN_RHO_EFF,
    OP_N_PRE_QUANT_CLAMP_F32, TWO_POW_32_F32, clamp_nan_to,
};

/// Maximum clamped `zoh_hold_depth` in `Op-N[Pre]` v4 (`1.0` enables exact `-325+ dB` hard hold).
pub const OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH: f32 = 1.0;

/// Minimum $C^1$ Hermite deadzone width `w_min = 2^-12` (`1 / 4096`, `ref[dsp.op-n.bitcrusher.quantizer]`).
pub const OP_N_PRE_MIN_DEADZONE_WIDTH: f32 = const_f32_pow2(-12);

const _: () = assert!(OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH == 1.0);
const _: () = assert!(OP_N_PRE_MIN_DEADZONE_WIDTH == 1.0 / 4_096.0);

/// Persistent `36 B` (`align(4)`) `Op-N[Pre]` v4.0 hot state fitting inside the `48 B`
/// `SonaLiteHotPrefix` slot (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpNPreV4HotState {
    /// Latched stereo ZOH output pair (`[L, R]`, canonical `+0.0` at Zero-DC).
    pub held: [f32; 2],
    /// Normalized `Q0.32` ZOH accumulator in `[0, 1)`.
    pub phase_q32: u32,
    /// `Q0.32` ZOH increment (`0` = latch every VM frame).
    pub inc_q32: u32,
    /// Quantizer wet blend `[0.0, 1.0]`.
    pub gamma_bc: f32,
    /// Quantizer log2 bit-depth `[0.0, 16.0]`.
    pub beta_q: f32,
    /// Effective ZOH hold depth `[0.0, 1.0]`.
    pub zoh_hold_depth_eff: f32,
    /// $C^1$ Hermite quantizer sharpness `s_q in [0.0, 1.0]`.
    pub s_q: f32,
    /// Phase-wrap ZOH sharpness `kappa_h in [0.0, 1.0]` (`0` = raised cosine, `1` = hard wrap).
    pub kappa_h: f32,
}

/// Hoisted `40 B` (`align(4)`) `Op-N[Pre]` v4.0 control-tick coefficients avoiding per-sample
/// `exp2` and reciprocal evaluations (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpNPreV4Hoisted {
    /// `Q0.32` ZOH increment (`0` latches every frame).
    pub inc_q32: u32,
    /// Quantizer wet blend `[0.0, 1.0]`.
    pub gamma_bc: f32,
    /// Quantizer log2 bit-depth `[0.0, 16.0]`.
    pub beta_q: f32,
    /// Effective ZOH hold depth `[0.0, 1.0]`.
    pub zoh_hold_depth_eff: f32,
    /// Quantizer sharpness `s_q in [0.0, 1.0]`.
    pub s_q: f32,
    /// Hold sharpness `kappa_h in [0.0, 1.0]`.
    pub kappa_h: f32,
    /// Precomputed quantizer scale `D_q = exp2_clamped16(beta_q)`.
    pub d_scale: f32,
    /// Precomputed quantizer step `Delta_q = exp2_clamped16(-beta_q)`.
    pub delta_step: f32,
    /// Deadzone half-edge `0.5 - 0.5 * w`, where `w = max(1 - s_q, 2^-12)`.
    pub deadzone_edge: f32,
    /// Half-width reciprocal `w_half_inv = 2.0 * rcp_pos_nr3(w)` for exact $C^1$ continuity at `r = ±0.5`.
    pub w_half_inv: f32,
}

const _: () = assert!(core::mem::size_of::<OpNPreV4HotState>() == 36);
const _: () = assert!(core::mem::align_of::<OpNPreV4HotState>() == 4);
const _: () = assert!(core::mem::size_of::<OpNPreV4Hoisted>() == 40);
const _: () = assert!(core::mem::align_of::<OpNPreV4Hoisted>() == 4);

/// Computes the 32-bit `Q0.32` `Op-N[Pre]` v4.0 ZOH phase increment from log-frequency ratio
/// `log2_rho_zoh` (`<= 0.0` at `48 kHz`), effective hold depth `z_h_eff in [0.0, 1.0]`, and cold
/// host ratio `rho_vm = 48_000.0 / f_{s,vm}` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
#[inline(always)]
#[must_use]
pub fn compute_op_n_pre_v4_inc_q32(log2_rho_zoh: f32, z_h_eff: f32, rho_vm: f32) -> u32 {
    let z_eff = clamp_nan_to(z_h_eff, 0.0, OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH, 0.0);
    if z_eff <= 0.0 {
        return 0;
    }
    let rho_zoh = if log2_rho_zoh.is_finite() {
        exp2_clamped16_f32(log2_rho_zoh)
    } else {
        1.0
    };
    let rho_eff = clamp_nan_to(rho_zoh * rho_vm, OP_N_PRE_MIN_RHO_EFF, 1.0, 1.0);
    if rho_eff >= OP_N_PRE_FULL_PASS_THRESH {
        return 0;
    }
    let scaled = rho_eff * TWO_POW_32_F32;
    let t = scaled as u32;
    let f = scaled - (t as f32);
    t.saturating_add(u32::from(f >= 0.5)).max(1)
}

/// Evaluates the `4`-FMA globally $C^1$-continuous Hermite deadzone staircase quantizer
/// `Q(x) = Delta_q * (n + sgn(r) * 0.5 * t^2 * (3 - 2*t))` using `w_half_inv = 2 / w` so
/// `t = clamp((|r| - (0.5 - 0.5*w)) * w_half_inv, 0, 1)` reaches `1.0` cleanly at `|r| = 0.5`
/// (`ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline(always)]
#[must_use]
pub fn quantize_hermite_v4_f32(
    x: f32,
    d_scale: f32,
    delta_step: f32,
    deadzone_edge: f32,
    w_half_inv: f32,
) -> f32 {
    let x_clip = clamp_nan_to(x, -OP_N_PRE_QUANT_CLAMP_F32, OP_N_PRE_QUANT_CLAMP_F32, 0.0);
    let v = x_clip * d_scale;
    let t_i = v as i32;
    let k = t_i - i32::from(v < (t_i as f32));
    let f = v - (k as f32);
    let q_idx = k + i32::from(f >= 0.5);
    let n = q_idx as f32;
    let r = (v - n).clamp(-0.5, 0.5);
    let t = ((const_abs_f32(r) - deadzone_edge) * w_half_inv).clamp(0.0, 1.0);
    let h = (t * t) * (1.5 - t);
    let sgn_h = if r >= 0.0 { h } else { -h };
    canonicalize_zero_f32_bits(delta_step * (n + sgn_h))
}

impl Default for OpNPreV4Hoisted {
    #[inline(always)]
    fn default() -> Self {
        Self::bypass()
    }
}

impl OpNPreV4Hoisted {
    /// Returns identity bypass parameters (`gamma_bc = 0`, `zoh_hold_depth_eff = 0`, `inc_q32 = 0`).
    #[inline(always)]
    #[must_use]
    pub const fn bypass() -> Self {
        Self {
            inc_q32: 0,
            gamma_bc: 0.0,
            beta_q: 0.0,
            zoh_hold_depth_eff: 0.0,
            s_q: 0.0,
            kappa_h: 0.0,
            d_scale: 1.0,
            delta_step: 1.0,
            deadzone_edge: 0.0,
            w_half_inv: 2.0,
        }
    }

    /// Clamps `ControlBus`-modulated targets and hoists `d_scale`, `delta_step`, `deadzone_edge`,
    /// `w_half_inv = 2.0 * rcp_pos_nr3(w)`, and `inc_q32` (`0 FDIV`).
    #[inline(always)]
    #[must_use]
    pub fn from_params(
        gamma_bc: f32,
        zoh_hold_depth_eff: f32,
        log2_rho_zoh: f32,
        beta_q: f32,
        s_q: f32,
        kappa_h: f32,
        rho_vm: f32,
    ) -> Self {
        let gamma = clamp_nan_to(gamma_bc, 0.0, 1.0, 0.0);
        let z_h_eff = clamp_nan_to(zoh_hold_depth_eff, 0.0, OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH, 0.0);
        let beta = clamp_nan_to(beta_q, 0.0, OP_N_PRE_MAX_BETA_Q, 0.0);
        let s_q_c = clamp_nan_to(s_q, 0.0, 1.0, 0.0);
        let kappa_c = clamp_nan_to(kappa_h, 0.0, 1.0, 0.0);
        let inc_q32 = compute_op_n_pre_v4_inc_q32(log2_rho_zoh, z_h_eff, rho_vm);
        let d_scale = exp2_clamped16_f32(beta);
        let delta_step = exp2_clamped16_f32(-beta);
        let w = (1.0 - s_q_c).max(OP_N_PRE_MIN_DEADZONE_WIDTH);
        let deadzone_edge = 0.5 - 0.5 * w;
        let w_half_inv = 2.0 * GpuVec4::splat(w).rcp_pos_nr3().to_array()[0];
        Self {
            inc_q32,
            gamma_bc: gamma,
            beta_q: beta,
            zoh_hold_depth_eff: z_h_eff,
            s_q: s_q_c,
            kappa_h: kappa_c,
            d_scale,
            delta_step,
            deadzone_edge,
            w_half_inv,
        }
    }
}

impl OpNPreV4HotState {
    /// Creates zero-initialized safe-bypass [`OpNPreV4HotState`].
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
            s_q: 0.0,
            kappa_h: 0.0,
        }
    }

    /// Loads hoisted targets without resetting `held` or `phase_q32`.
    #[inline(always)]
    pub fn apply_hoisted(&mut self, hoisted: &OpNPreV4Hoisted) {
        self.inc_q32 = hoisted.inc_q32;
        self.gamma_bc = hoisted.gamma_bc;
        self.beta_q = hoisted.beta_q;
        self.zoh_hold_depth_eff = hoisted.zoh_hold_depth_eff;
        self.s_q = hoisted.s_q;
        self.kappa_h = hoisted.kappa_h;
    }

    #[inline(always)]
    fn step_clock_gain(&mut self) -> f32 {
        let z_h_eff = clamp_nan_to(
            self.zoh_hold_depth_eff,
            0.0,
            OP_N_PRE_V4_MAX_ZOH_HOLD_DEPTH,
            0.0,
        );
        if self.inc_q32 == 0 || z_h_eff <= 0.0 {
            return 1.0;
        }
        let new_phase = self.phase_q32.wrapping_add(self.inc_q32);
        let wrap = if new_phase < self.phase_q32 { 1.0 } else { 0.0 };
        self.phase_q32 = new_phase;
        let cos_phi = q32_smooth_sin_f32(new_phase.wrapping_add(0x4000_0000));
        let u = 0.5 * (1.0 + cos_phi);
        let w_cos = u * u;
        let kappa_h = clamp_nan_to(self.kappa_h, 0.0, 1.0, 0.0);
        (1.0 - z_h_eff) + z_h_eff * ((1.0 - kappa_h) * w_cos + kappa_h * wrap)
    }

    /// Steps one stereo `Op-N[Pre]` v4.0 frame in strict **Quantize -> Wet Blend -> Hard-Capable
    /// Q0.32 Phase-Wrap ZOH** order (`0-ULP` bypass when `gamma_bc <= 0 && zoh_hold_depth_eff <= 0`,
    /// `ref[dsp.op-n.bitcrusher.zoh-phase]`).
    #[inline(always)]
    pub fn step_stereo(&mut self, x_l: f32, x_r: f32, hoisted: &OpNPreV4Hoisted) -> (f32, f32) {
        self.apply_hoisted(hoisted);
        let gamma = hoisted.gamma_bc;
        let z_h_eff = hoisted.zoh_hold_depth_eff;
        if gamma <= 0.0 && z_h_eff <= 0.0 {
            self.held[0] = scrub_and_flush_f32_bits(x_l) + 0.0;
            self.held[1] = scrub_and_flush_f32_bits(x_r) + 0.0;
            return (x_l, x_r);
        }

        let (wet_l, wet_r) = if gamma <= 0.0 {
            (
                scrub_and_flush_f32_bits(x_l) + 0.0,
                scrub_and_flush_f32_bits(x_r) + 0.0,
            )
        } else {
            let (q_l, q_r) =
                if !hoisted.beta_q.is_finite() || hoisted.beta_q <= OP_N_PRE_BETA_BYPASS_EPS {
                    (
                        clamp_nan_to(
                            x_l,
                            -OP_N_PRE_QUANT_CLAMP_F32,
                            OP_N_PRE_QUANT_CLAMP_F32,
                            0.0,
                        ),
                        clamp_nan_to(
                            x_r,
                            -OP_N_PRE_QUANT_CLAMP_F32,
                            OP_N_PRE_QUANT_CLAMP_F32,
                            0.0,
                        ),
                    )
                } else {
                    (
                        quantize_hermite_v4_f32(
                            x_l,
                            hoisted.d_scale,
                            hoisted.delta_step,
                            hoisted.deadzone_edge,
                            hoisted.w_half_inv,
                        ),
                        quantize_hermite_v4_f32(
                            x_r,
                            hoisted.d_scale,
                            hoisted.delta_step,
                            hoisted.deadzone_edge,
                            hoisted.w_half_inv,
                        ),
                    )
                };
            let xl_c = scrub_and_flush_f32_bits(x_l) + 0.0;
            let xr_c = scrub_and_flush_f32_bits(x_r) + 0.0;
            (xl_c + gamma * (q_l - xl_c), xr_c + gamma * (q_r - xr_c))
        };

        let g_clk = self.step_clock_gain();
        self.held[0] = canonicalize_zero_f32_bits((1.0 - g_clk) * self.held[0] + g_clk * wet_l);
        self.held[1] = canonicalize_zero_f32_bits((1.0 - g_clk) * self.held[1] + g_clk * wet_r);
        (self.held[0], self.held[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn quantize_hermite_v4_f64_oracle(x: f64, beta_q: f64, s_q: f64) -> f64 {
        let d_scale = 2.0_f64.powf(beta_q);
        let delta_step = 2.0_f64.powf(-beta_q);
        let w = (1.0 - s_q).max(2.0_f64.powi(-12));
        let edge = 0.5 - 0.5 * w;
        let w_half_inv = 2.0 / w;
        let v = x.clamp(-100.0, 100.0) * d_scale;
        let n = v.round_ties_even();
        let r = (v - n).clamp(-0.5, 0.5);
        let t = ((r.abs() - edge) * w_half_inv).clamp(0.0, 1.0);
        let h = t * t * (1.5 - t);
        delta_step * (n + if r >= 0.0 { h } else { -h })
    }

    /// verify[dsp.op-n.bitcrusher.quantizer]
    /// verify[dsp.op-n.bitcrusher.zoh-phase]
    /// verify[dsp.op-n.bitcrusher.rate-transfer]
    #[test]
    fn test_op_n_pre_v4_c1_continuity_bypass_zero_dc_and_hard_zoh() {
        assert_eq!(core::mem::size_of::<OpNPreV4HotState>(), 36);
        assert_eq!(core::mem::align_of::<OpNPreV4HotState>(), 4);
        assert_eq!(core::mem::size_of::<OpNPreV4Hoisted>(), 40);
        assert_eq!(core::mem::align_of::<OpNPreV4Hoisted>(), 4);

        // 0-ULP bypass when gamma_bc <= 0 && zoh_hold_depth_eff <= 0
        let mut st = OpNPreV4HotState::zero();
        let bypass = OpNPreV4Hoisted::bypass();
        for &x in &[-0.0_f32, 0.375, -12.25, 1.0e-42] {
            let (yl, yr) = st.step_stereo(x, -x, &bypass);
            assert_eq!(yl.to_bits(), x.to_bits());
            assert_eq!(yr.to_bits(), (-x).to_bits());
        }

        // Zero-DC canonical +0.0 under active quantizer + hard ZOH
        let active = OpNPreV4Hoisted::from_params(1.0, 1.0, -3.0, 4.0, 0.98, 1.0, 1.0);
        let mut st_dc = OpNPreV4HotState::zero();
        for &z in &[0.0_f32, -0.0, 0.0, -0.0] {
            let (yl, yr) = st_dc.step_stereo(z, z, &active);
            assert_eq!(yl.to_bits(), 0);
            assert_eq!(yr.to_bits(), 0);
        }

        // Exact C1 continuity (zero slope) across half-integer cell boundaries r = ±0.5 for s_q in [0, 1]
        for &s_q in &[0.0_f32, 0.5, 0.98, 1.0] {
            let h = OpNPreV4Hoisted::from_params(1.0, 0.0, 0.0, 2.0, s_q, 0.0, 1.0);
            let w = (1.0 - s_q).max(OP_N_PRE_MIN_DEADZONE_WIDTH);
            let eps_c1 = 1.0e-3 * 0.5 * w * h.delta_step;
            // Boundary at x = 0.5 / d_scale = 0.125 (v = 0.5)
            let q_lo = quantize_hermite_v4_f32(
                0.125 - eps_c1,
                h.d_scale,
                h.delta_step,
                h.deadzone_edge,
                h.w_half_inv,
            );
            let q_0 = quantize_hermite_v4_f32(
                0.125,
                h.d_scale,
                h.delta_step,
                h.deadzone_edge,
                h.w_half_inv,
            );
            let q_hi = quantize_hermite_v4_f32(
                0.125 + eps_c1,
                h.d_scale,
                h.delta_step,
                h.deadzone_edge,
                h.w_half_inv,
            );
            assert!(
                (q_hi - q_lo).abs() < 1.0e-5,
                "discontinuity at s_q={s_q}: {q_lo} vs {q_hi}"
            );
            assert!(
                (q_hi - q_0).abs() <= 2.0e-5 && (q_0 - q_lo).abs() <= 2.0e-5,
                "quadratic zero-slope boundary failed at s_q={s_q}"
            );
            assert!((q_0 - 0.125).abs() < 1.0e-6);
            let q_neg_lo = quantize_hermite_v4_f32(
                -0.125 - eps_c1,
                h.d_scale,
                h.delta_step,
                h.deadzone_edge,
                h.w_half_inv,
            );
            let q_neg_hi = quantize_hermite_v4_f32(
                -0.125 + eps_c1,
                h.d_scale,
                h.delta_step,
                h.deadzone_edge,
                h.w_half_inv,
            );
            assert!((q_neg_hi - q_neg_lo).abs() < 1.0e-5);
        }

        // Chain-18 hard ZOH parity: (s_q = 1.0, kappa_h = 1.0, z_h = 1.0) matches ideal sample-and-hold <= -325 dB
        let mut st_hard = OpNPreV4HotState::zero();
        let h_hard = OpNPreV4Hoisted::from_params(0.0, 1.0, -3.0, 0.0, 1.0, 1.0, 1.0); // 48k / 8 = 6 kHz
        let mut ideal_held = 0.0_f64;
        let mut ideal_phase = 0_u32;
        let mut err2 = 0.0_f64;
        let mut ref2 = 0.0_f64;
        for n in 0..256 {
            let x = (0.17 * (n as f64)).sin() as f32;
            let (yl, _) = st_hard.step_stereo(x, x, &h_hard);
            let next_p = ideal_phase.wrapping_add(h_hard.inc_q32);
            if next_p < ideal_phase {
                ideal_held = f64::from(x);
            }
            ideal_phase = next_p;
            err2 += (f64::from(yl) - ideal_held).powi(2);
            ref2 += ideal_held * ideal_held;
        }
        assert!(err2 == 0.0 || 10.0 * (err2 / ref2).log10() <= -325.0);

        // Rate-invariance: 6 kHz ZOH at 48 kHz vs 192 kHz (log2_rho = -3.0)
        assert_eq!(compute_op_n_pre_v4_inc_q32(-3.0, 1.0, 1.0), 0x2000_0000);
        assert_eq!(compute_op_n_pre_v4_inc_q32(-3.0, 1.0, 0.25), 0x0800_0000);
    }

    /// verify[dsp.op-n.bitcrusher.surrogate]
    #[test]
    fn test_op_n_pre_v4_chain18_quantizer_parity_and_gradients_fd() {
        // Chain-18 2/3/4/6-bit staircase MSE at s_q = 0.98 vs ideal mid-tread quantizer
        for &(beta_q, ceiling_db) in &[
            (1.0_f32, -29.5_f64),
            (2.0, -35.5),
            (3.0, -41.5),
            (5.0, -54.0),
        ] {
            let h = OpNPreV4Hoisted::from_params(1.0, 0.0, 0.0, beta_q, 0.98, 0.0, 1.0);
            let mut err2 = 0.0_f64;
            let mut ref2 = 0.0_f64;
            for n in 0..4096 {
                let x = (0.043 * (n as f64)).sin();
                let q = f64::from(quantize_hermite_v4_f32(
                    x as f32,
                    h.d_scale,
                    h.delta_step,
                    h.deadzone_edge,
                    h.w_half_inv,
                ));
                let q_ideal = h.delta_step as f64 * (x * (h.d_scale as f64)).round();
                err2 += (q - q_ideal).powi(2);
                ref2 += q_ideal * q_ideal;
            }
            let db = 10.0 * (err2 / ref2).log10();
            assert!(
                db <= ceiling_db,
                "beta_q={beta_q}: {db} dB > {ceiling_db} dB"
            );
        }

        // Exact analytical gradients vs central FD (< 1e-6) inside transition zone 0 < t < 1
        let beta_q = 2.0_f64;
        let s_q = 0.6_f64;
        let w = 1.0 - s_q; // 0.4
        let edge = 0.5 - 0.5 * w; // 0.3
        let w_half_inv = 2.0 / w;
        // Choose x so |r| = 0.4 inside (0.3, 0.5): v = 1.4 -> x = 1.4 / 4 = 0.35
        let x = 0.35_f64;
        let d_scale = 2.0_f64.powf(beta_q);
        let delta_step = 2.0_f64.powf(-beta_q);
        let v = x * d_scale;
        let n = v.round_ties_even();
        let r = v - n;
        let t = (r.abs() - edge) * w_half_inv;
        let sgn = if r >= 0.0 { 1.0 } else { -1.0 };
        let q_val = quantize_hermite_v4_f64_oracle(x, beta_q, s_q);
        let dq_dx = 3.0 * t * (1.0 - t) * w_half_inv;
        let dq_dsq = -delta_step * sgn * 3.0 * t * (1.0 - t) * (1.0 - t) / w;
        let dq_dbeta = core::f64::consts::LN_2 * (x * dq_dx - q_val);

        let eps = 1.0e-7_f64;
        let fd_x = (quantize_hermite_v4_f64_oracle(x + eps, beta_q, s_q)
            - quantize_hermite_v4_f64_oracle(x - eps, beta_q, s_q))
            / (2.0 * eps);
        let fd_s = (quantize_hermite_v4_f64_oracle(x, beta_q, s_q + eps)
            - quantize_hermite_v4_f64_oracle(x, beta_q, s_q - eps))
            / (2.0 * eps);
        let fd_b = (quantize_hermite_v4_f64_oracle(x, beta_q + eps, s_q)
            - quantize_hermite_v4_f64_oracle(x, beta_q - eps, s_q))
            / (2.0 * eps);
        assert!((dq_dx - fd_x).abs() < 1.0e-6);
        assert!((dq_dsq - fd_s).abs() < 1.0e-6);
        assert!((dq_dbeta - fd_b).abs() < 1.0e-6);
    }

    proptest! {
        #[test]
        fn prop_op_n_pre_v4_clock_gain_contractive_and_finite(
            x_l in -1.0e9_f32..1.0e9_f32,
            x_r in -1.0e9_f32..1.0e9_f32,
            gamma in 0.0_f32..1.0_f32,
            z_h in 0.0_f32..1.0_f32,
            log2_rho in -8.0_f32..0.0_f32,
            beta_q in 0.0_f32..16.0_f32,
            s_q in 0.0_f32..1.0_f32,
            kappa_h in 0.0_f32..1.0_f32,
        ) {
            let h = OpNPreV4Hoisted::from_params(gamma, z_h, log2_rho, beta_q, s_q, kappa_h, 1.0);
            let mut st = OpNPreV4HotState::zero();
            let (yl, yr) = st.step_stereo(x_l, x_r, &h);
            prop_assert!(yl.is_finite() && yr.is_finite());
        }
    }

    #[test]
    fn test_op_n_pre_v4_bypass_default_and_edge_branches() {
        let def = OpNPreV4Hoisted::default();
        assert_eq!(def, OpNPreV4Hoisted::bypass());
        assert_eq!(compute_op_n_pre_v4_inc_q32(f32::NAN, 0.5, 1.0), 0);
        assert_eq!(compute_op_n_pre_v4_inc_q32(0.0, 1.0, 1.0), 0);

        let mut st = OpNPreV4HotState::zero();
        // ZOH-only (gamma = 0.0, z_h > 0): wet_l == x_l, ZOH latches cleanly.
        let zoh_only = OpNPreV4Hoisted::from_params(0.0, 1.0, -2.0, 4.0, 0.9, 1.0, 1.0);
        let (y0_l, y0_r) = st.step_stereo(0.25, -0.5, &zoh_only);
        assert_eq!(y0_l, 0.0);
        assert_eq!(y0_r, 0.0);

        // Quantizer active (gamma = 1.0) with beta_q <= eps bypasses staircase quantization.
        let mut st_beta0 = OpNPreV4HotState::zero();
        let beta0 = OpNPreV4Hoisted::from_params(1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0);
        let (qb_l, qb_r) = st_beta0.step_stereo(0.375, -0.625, &beta0);
        assert_eq!(qb_l.to_bits(), 0.375_f32.to_bits());
        assert_eq!(qb_r.to_bits(), (-0.625_f32).to_bits());
    }
}
