//! `200 Hz` sidechain `TPT-LR4` envelope crossover, stereo-linked & auxiliary-key detector, and
//! 4-band asymmetric ballistic envelope bank (`0 FDIV`, `0 FSQRT`, `0 libm`, `!alloc`,
//! `ref[dsp.router.lr4-crossover]`, `ref[dsp.router.stereo-link-detector]`,
//! `ref[dsp.router.ballistic-envelope-bank]`).

#![forbid(unsafe_code)]

use crate::graph::zoh::ballistic_alpha_f32;
use crate::math::{
    clamp_finite_f32, const_abs_f32, const_inv_factorial_f64, flush_recursive_state_f32,
    rcp_nr6_f64, sanitize_finite_f32,
};

/// Default sidechain `TPT-LR4` crossover frequency in Hz (`200.0 Hz`).
pub const DEFAULT_SIDECHAIN_LR4_HZ: f32 = 200.0;
/// Minimum clamped sidechain `TPT-LR4` crossover frequency in Hz (`20.0 Hz`).
pub const MIN_SIDECHAIN_LR4_HZ: f32 = 20.0;
/// Maximum clamped sidechain `TPT-LR4` crossover frequency in Hz (`2,000.0 Hz`).
pub const MAX_SIDECHAIN_LR4_HZ: f32 = 2_000.0;
/// Canonical 4-band attack time constants in seconds (`[Low, Mid/High, Broadband, AuxReturn]`).
pub const BALLISTIC_TAU_ATT_SEC: [f64; 4] = [0.0100, 0.0020, 0.0002, 0.0005];
/// Canonical 4-band release time constants in seconds (`[Low, Mid/High, Broadband, AuxReturn]`).
pub const BALLISTIC_TAU_REL_SEC: [f64; 4] = [0.1500, 0.0400, 0.0150, 0.0500];

const INV_FACT_2: f64 = const_inv_factorial_f64(2);
const INV_FACT_3: f64 = const_inv_factorial_f64(3);
const INV_FACT_4: f64 = const_inv_factorial_f64(4);
const INV_FACT_5: f64 = const_inv_factorial_f64(5);
const INV_FACT_6: f64 = const_inv_factorial_f64(6);
const INV_FACT_7: f64 = const_inv_factorial_f64(7);
const INV_FACT_8: f64 = const_inv_factorial_f64(8);
const INV_FACT_9: f64 = const_inv_factorial_f64(9);

const _: () = assert!(MIN_SIDECHAIN_LR4_HZ < DEFAULT_SIDECHAIN_LR4_HZ);
const _: () = assert!(DEFAULT_SIDECHAIN_LR4_HZ < MAX_SIDECHAIN_LR4_HZ);
const _: () = assert!(INV_FACT_3 > 0.16 && INV_FACT_3 < 0.17);

/// Computes cold `TPT-LR4` coefficients `(a1, a2, a3)` via `binary64` Maclaurin `sin_a / cos_a`
/// and `rcp_nr6_f64` (`0 FDIV`, `0 libm`, `router_and_crossbar_topology.md §9.1`).
#[inline]
fn io_router_lr4_tan(fc_hz: f32, f_s_vm: f64) -> (f32, f32, f32) {
    let fc = f64::from(clamp_finite_f32(
        fc_hz,
        MIN_SIDECHAIN_LR4_HZ,
        MAX_SIDECHAIN_LR4_HZ,
        DEFAULT_SIDECHAIN_LR4_HZ,
    ));
    let fs = if f_s_vm.is_finite() {
        f_s_vm.max(1.0)
    } else {
        48_000.0
    };
    let a = (core::f64::consts::PI * fc * rcp_nr6_f64(fs)).clamp(1.0e-5, 1.4);
    let u2 = a * a;
    let sin_a =
        a * (1.0 - u2 * (INV_FACT_3 - u2 * (INV_FACT_5 - u2 * (INV_FACT_7 - u2 * INV_FACT_9))));
    let cos_a = 1.0 - u2 * (INV_FACT_2 - u2 * (INV_FACT_4 - u2 * (INV_FACT_6 - u2 * INV_FACT_8)));
    let g = sin_a * rcp_nr6_f64(cos_a);
    let a1 = rcp_nr6_f64(1.0 + g * (g + core::f64::consts::SQRT_2));
    let a2 = g * a1;
    let a3 = g * a2;
    (a1 as f32, a2 as f32, a3 as f32)
}

#[inline(always)]
fn step_svf_lp_hp(a1: f32, a2: f32, a3: f32, x: f32, state: &mut [f32; 2]) -> (f32, f32) {
    let v3 = x - state[1];
    let v1 = a1 * state[0] + a2 * v3;
    let lp = state[1] + a2 * state[0] + a3 * v3;
    let hp = x - core::f32::consts::SQRT_2 * v1 - lp;
    state[0] = flush_recursive_state_f32(2.0 * v1 - state[0]);
    state[1] = flush_recursive_state_f32(2.0 * lp - state[1]);
    (lp, hp)
}

#[inline(always)]
fn step_lr4_channel(a1: f32, a2: f32, a3: f32, x: f32, states: &mut [[f32; 2]; 3]) -> (f32, f32) {
    let (lp2, hp2) = step_svf_lp_hp(a1, a2, a3, x, &mut states[0]);
    let (lp4, _) = step_svf_lp_hp(a1, a2, a3, lp2, &mut states[1]);
    let (_, hp4) = step_svf_lp_hp(a1, a2, a3, hp2, &mut states[2]);
    (lp4, hp4)
}

/// Evaluates the programmable stereo-linked & aux-key sidechain detector `(det_l, det_r)` on
/// raw pre-trim inputs without `-6.02 dB` hard-panned peak loss
/// (`ref[dsp.router.stereo-link-detector]`).
#[inline(always)]
pub fn eval_stereo_linked_aux_detector(
    inst_l: f32,
    inst_r: f32,
    aux_mono: f32,
    alpha_link: f32,
    beta_key: f32,
) -> (f32, f32) {
    let a = clamp_finite_f32(alpha_link, 0.0, 1.0, 1.0);
    let b = clamp_finite_f32(beta_key, 0.0, 1.0, 0.0);
    let abs_l = sanitize_finite_f32(const_abs_f32(inst_l));
    let abs_r = sanitize_finite_f32(const_abs_f32(inst_r));
    let e_link = abs_l.max(abs_r);
    let e_key = sanitize_finite_f32(const_abs_f32(aux_mono));
    let one_minus_a = 1.0 - a;
    let one_minus_b = 1.0 - b;
    let det_l = one_minus_b * (one_minus_a * abs_l + a * e_link) + b * e_key;
    let det_r = one_minus_b * (one_minus_a * abs_r + a * e_link) + b * e_key;
    (det_l, det_r)
}

/// 4-band sidechain envelope detector bank driven by a `200 Hz` `TPT-LR4` crossover on raw
/// pre-trim `(inst_l, inst_r)` and aligned auxiliary return inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IoSidechainDetectorBank {
    a1: f32,
    a2: f32,
    a3: f32,
    lr4_l: [[f32; 2]; 3],
    lr4_r: [[f32; 2]; 3],
    att_a: [f32; 4],
    rel_a: [f32; 4],
    env: [f32; 4],
    alpha_link: f32,
    beta_key: f32,
}

impl Default for IoSidechainDetectorBank {
    fn default() -> Self {
        Self::new(48_000.0, DEFAULT_SIDECHAIN_LR4_HZ)
    }
}

impl IoSidechainDetectorBank {
    /// Initializes a zero-state detector bank at `f_s_vm` and sidechain cutoff `fc_hz`.
    pub fn new(f_s_vm: f64, fc_hz: f32) -> Self {
        let mut bank = Self {
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            lr4_l: [[0.0; 2]; 3],
            lr4_r: [[0.0; 2]; 3],
            att_a: [0.0; 4],
            rel_a: [0.0; 4],
            env: [0.0; 4],
            alpha_link: 1.0,
            beta_key: 0.0,
        };
        bank.reconfigure(f_s_vm, fc_hz);
        bank
    }

    /// Recomputes `TPT-LR4` coefficients and 4-band ZOH ballistic alphas while preserving state.
    pub fn reconfigure(&mut self, f_s_vm: f64, fc_hz: f32) {
        let (a1, a2, a3) = io_router_lr4_tan(fc_hz, f_s_vm);
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
        for b in 0..4 {
            self.att_a[b] = ballistic_alpha_f32(f_s_vm, BALLISTIC_TAU_ATT_SEC[b]);
            self.rel_a[b] = ballistic_alpha_f32(f_s_vm, BALLISTIC_TAU_REL_SEC[b]);
        }
    }

    /// Configures cold-path stereo-link (`alpha_link in [0, 1]`) and aux-key (`beta_key in [0, 1]`).
    #[inline]
    pub fn set_link_and_key(&mut self, alpha_link: f32, beta_key: f32) {
        self.alpha_link = clamp_finite_f32(alpha_link, 0.0, 1.0, 1.0);
        self.beta_key = clamp_finite_f32(beta_key, 0.0, 1.0, 0.0);
    }

    /// Resets `TPT-LR4` filter states and all 4 envelope followers to `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        self.lr4_l = [[0.0; 2]; 3];
        self.lr4_r = [[0.0; 2]; 3];
        self.env = [0.0; 4];
    }

    /// Advances the sidechain `LR4` crossover and 4-band asymmetric envelopes by one VM frame.
    #[inline]
    pub fn step(
        &mut self,
        inst_l: f32,
        inst_r: f32,
        ret_l_aligned: f32,
        ret_r_aligned: f32,
    ) -> [f32; 4] {
        let xl = flush_recursive_state_f32(inst_l);
        let xr = flush_recursive_state_f32(inst_r);
        let rl = flush_recursive_state_f32(ret_l_aligned);
        let rr = flush_recursive_state_f32(ret_r_aligned);

        let (lp4_l, hp4_l) = step_lr4_channel(self.a1, self.a2, self.a3, xl, &mut self.lr4_l);
        let (lp4_r, hp4_r) = step_lr4_channel(self.a1, self.a2, self.a3, xr, &mut self.lr4_r);

        let aux_mono = 0.5 * (rl + rr);
        let (det_l, det_r) =
            eval_stereo_linked_aux_detector(xl, xr, aux_mono, self.alpha_link, self.beta_key);

        let inputs = [
            0.5 * (const_abs_f32(lp4_l) + const_abs_f32(lp4_r)),
            0.5 * (const_abs_f32(hp4_l) + const_abs_f32(hp4_r)),
            0.5 * (det_l + det_r),
            0.5 * (const_abs_f32(rl) + const_abs_f32(rr)),
        ];
        for (b, &u) in inputs.iter().enumerate() {
            let diff = u - self.env[b];
            let coeff = if diff > 0.0 {
                self.att_a[b]
            } else {
                self.rel_a[b]
            };
            self.env[b] = flush_recursive_state_f32(self.env[b] + coeff * diff);
        }
        self.env
    }

    /// Returns the current 4-band envelope levels `[low, mid_high, broadband, aux_return]`.
    #[inline]
    pub const fn envelopes(&self) -> [f32; 4] {
        self.env
    }

    /// Returns the prewarped `TPT-LR4` coefficients `(a1, a2, a3)`.
    #[inline]
    pub const fn lr4_coeffs(&self) -> (f32, f32, f32) {
        (self.a1, self.a2, self.a3)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::{const_sin_cos_f64, const_tan_f64};

    #[test]
    fn test_lr4_coeffs_dc_nyquist_and_detector_bank() {
        let mut bank = IoSidechainDetectorBank::new(48_000.0, 200.0);
        let (a1, a2, a3) = bank.lr4_coeffs();
        let g_ref = const_tan_f64(core::f64::consts::PI * 200.0 * rcp_nr6_f64(48_000.0));
        let a1_ref = rcp_nr6_f64(1.0 + g_ref * (g_ref + core::f64::consts::SQRT_2));
        assert!((f64::from(a1) - a1_ref).abs() < 1.0e-6);
        assert!((f64::from(a2) - g_ref * a1_ref).abs() < 1.0e-6);
        assert!((f64::from(a3) - g_ref * g_ref * a1_ref).abs() < 1.0e-6);

        assert_eq!(bank.step(0.0, 0.0, 0.0, 0.0).map(f32::to_bits), [0; 4]);

        for _ in 0..16_000 {
            bank.step(1.0, 1.0, 0.5, -0.5);
        }
        let env_dc = bank.envelopes();
        assert!((env_dc[0] - 1.0).abs() < 1.0e-3);
        assert!(env_dc[1] < 1.0e-3);
        assert!((env_dc[2] - 1.0).abs() < 1.0e-4);
        assert!((env_dc[3] - 0.5).abs() < 1.0e-4);

        bank.reset();
        assert_eq!(bank.envelopes().map(f32::to_bits), [0; 4]);
        for n in 0..48_000 {
            let sign = if (n & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
            bank.step(sign, sign, 0.0, 0.0);
        }
        let env_nyq = bank.envelopes();
        assert!(env_nyq[0] < 1.0e-4);
        assert!((env_nyq[1] - 1.0).abs() < 1.0e-3);

        let (dl, dr) = eval_stereo_linked_aux_detector(1.0, 0.0, 0.25, 1.0, 0.0);
        assert!((dl - 1.0).abs() < 1.0e-7 && (dr - 1.0).abs() < 1.0e-7);
        let (kl, kr) = eval_stereo_linked_aux_detector(1.0, 0.0, 0.25, 0.0, 1.0);
        assert!((kl - 0.25).abs() < 1.0e-7 && (kr - 0.25).abs() < 1.0e-7);

        bank.reconfigure(96_000.0, 250.0);
        bank.set_link_and_key(0.5, 0.25);
        let stepped = bank.step(0.5, -0.25, 0.1, 0.2);
        assert!(stepped.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_sidechain_lr4_allpass_sum_and_non_negative_envelopes(
            fc_hz in 20.0_f32..=2_000.0_f32,
            inst_l in -10.0_f32..=10.0_f32,
            inst_r in -10.0_f32..=10.0_f32,
            ret_l in -10.0_f32..=10.0_f32,
            ret_r in -10.0_f32..=10.0_f32,
        ) {
            let mut bank = IoSidechainDetectorBank::new(48_000.0, fc_hz);
            let mut lr4_state = [[0.0_f32; 2]; 3];
            let (a1, a2, a3) = bank.lr4_coeffs();
            let omega = core::f64::consts::TAU * f64::from(fc_hz) * rcp_nr6_f64(48_000.0);
            let period = ((48_000.0 * rcp_nr6_f64(f64::from(fc_hz))) + 0.5) as usize;
            let warmup = (2 * period).max(512);
            let measure = (2 * period).max(512);
            let mut lp_sq = 0.0_f64;
            let mut hp_sq = 0.0_f64;
            for n in 0..(warmup + measure) {
                let s = const_sin_cos_f64((n as f64) * omega).0 as f32;
                let (lp4, hp4) = step_lr4_channel(a1, a2, a3, s, &mut lr4_state);
                if n >= warmup {
                    lp_sq += f64::from(lp4) * f64::from(lp4);
                    hp_sq += f64::from(hp4) * f64::from(hp4);
                }
                if n < 512 {
                    let env = bank.step(inst_l, inst_r, ret_l, ret_r);
                    for e in env {
                        prop_assert!(e.is_finite() && e >= 0.0);
                    }
                }
            }
            prop_assert!((lp_sq - hp_sq).abs() < 0.05 * (lp_sq + hp_sq + 1.0e-6));
        }
    }
}
