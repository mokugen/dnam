//! Continuous-time Zero-Order Hold (`ZOH`) sample-rate context and parameter conversion helpers
//! (`ref[dsp.zoh.op-m-poles]`, `ref[dsp.zoh.core-paralru]`, `ref[dsp.zoh.ctrl-stride]`).

#![forbid(unsafe_code)]

use crate::math::{
    MAX_PARALRU_R_F32, clamp_finite_f32, const_abs_f64, exp_f64, powf_f64, rcp_nr6_f64,
    round_half_away_f64,
};
use crate::types::{Tier, VmError};

/// Minimum valid reference sample rate in Hz (`44,100 Hz`).
pub const MIN_REF_SAMPLE_RATE_HZ: f64 = 44_100.0;
/// Maximum valid reference sample rate in Hz (`96,000 Hz`).
pub const MAX_REF_SAMPLE_RATE_HZ: f64 = 96_000.0;
/// Minimum valid host sample rate in Hz (`44,100 Hz`).
pub const MIN_HOST_SAMPLE_RATE_HZ: f64 = 44_100.0;
/// Maximum valid host or effective VM sample rate in Hz (`384,000 Hz`).
pub const MAX_HOST_SAMPLE_RATE_HZ: f64 = 384_000.0;
/// Nominal `3 kHz` control-plane rate in Hz.
const CTRL_NOMINAL_RATE_HZ: f64 = 3_000.0;
/// Minimum control-bus decimation stride in VM frames (`ref[dsp.zoh.ctrl-stride]`).
const MIN_CTRL_STRIDE: usize = 8;
/// Offline bounce oversampling factor (`4x`).
const BOUNCE_FACTOR: f64 = 4.0;

const _: () = assert!(MIN_REF_SAMPLE_RATE_HZ <= MAX_REF_SAMPLE_RATE_HZ);
const _: () = assert!(MIN_HOST_SAMPLE_RATE_HZ <= MAX_HOST_SAMPLE_RATE_HZ);
const _: () = assert!(MIN_CTRL_STRIDE == 8);

/// Precomputed dual-rate VM / SONA-core sample-rate context (`0 FDIV`, `0 FSQRT`, `0 libm`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZohRateContext {
    /// Model reference sample rate in Hz (`[44_100, 96_000]`).
    pub reference_rate_hz: f32,
    /// Host callback sample rate in Hz (`[44_100, 384_000]`).
    pub host_rate_hz: f32,
    /// Perimeter VM execution rate in Hz (`host` live, or `4 * host` in `4x` offline bounce).
    pub f_s_vm: f64,
    /// Dimensionless ZOH perimeter rate ratio `reference_rate_hz / f_s_vm`.
    pub rho_vm: f64,
    /// Whether internal `2x` oversampling (`OS2X`) wraps the 4-block SONA core.
    pub os2x_active: bool,
    /// 4-block SONA core execution rate in Hz (`2 * f_s_vm` under `OS2X`, else `f_s_vm`).
    pub f_s_core: f64,
    /// Dimensionless ZOH core rate ratio `reference_rate_hz / f_s_core`.
    pub rho_core: f64,
    /// Control-bus decimation stride `max(8, round_half_away(f_s_vm / 3_000.0))`.
    pub k_ctrl: usize,
    /// Whether `4x` offline bounce oversampling is active.
    pub is_bounce_4x: bool,
}

impl ZohRateContext {
    /// Computes and validates the two-rate [`ZohRateContext`] (`ref[dsp.router.signal-flow]`,
    /// `ref[dsp.zoh.ctrl-stride]`).
    pub fn compute(
        reference_rate_hz: f32,
        host_rate_hz: f32,
        tier: Tier,
        tier_os2x: bool,
        offline_bounce: bool,
    ) -> Result<Self, VmError> {
        if !reference_rate_hz.is_finite() || !host_rate_hz.is_finite() {
            return Err(VmError::InvalidSampleRate);
        }
        let f_ref = f64::from(reference_rate_hz);
        let f_host = f64::from(host_rate_hz);
        if !(MIN_REF_SAMPLE_RATE_HZ..=MAX_REF_SAMPLE_RATE_HZ).contains(&f_ref)
            || !(MIN_HOST_SAMPLE_RATE_HZ..=MAX_HOST_SAMPLE_RATE_HZ).contains(&f_host)
        {
            return Err(VmError::InvalidSampleRate);
        }
        let f_s_vm = if offline_bounce {
            BOUNCE_FACTOR * f_host
        } else {
            f_host
        };
        if !f_s_vm.is_finite() || f_s_vm > MAX_HOST_SAMPLE_RATE_HZ {
            return Err(VmError::InvalidSampleRate);
        }
        let os2x_active = !matches!(tier, Tier::Lite)
            && !offline_bounce
            && tier_os2x
            && (2.0 * f_s_vm <= MAX_HOST_SAMPLE_RATE_HZ);
        let f_s_core = if os2x_active { 2.0 * f_s_vm } else { f_s_vm };

        let rho_vm = if const_abs_f64(f_s_vm - f_ref) <= 1.0e-12 {
            1.0
        } else {
            f_ref * rcp_nr6_f64(f_s_vm)
        };
        let rho_core = if const_abs_f64(f_s_core - f_ref) <= 1.0e-12 {
            1.0
        } else {
            f_ref * rcp_nr6_f64(f_s_core)
        };
        let k_ctrl = (round_half_away_f64(f_s_vm * rcp_nr6_f64(CTRL_NOMINAL_RATE_HZ)) as usize)
            .max(MIN_CTRL_STRIDE);

        Ok(Self {
            reference_rate_hz,
            host_rate_hz,
            f_s_vm,
            rho_vm,
            os2x_active,
            f_s_core,
            rho_core,
            k_ctrl,
            is_bounce_4x: offline_bounce,
        })
    }

    /// Returns `1.0 / f_s_vm` computed via 6-step `binary64` Newton reciprocal (`0 FDIV`).
    #[inline]
    pub fn inv_f_s_vm(&self) -> f64 {
        rcp_nr6_f64(self.f_s_vm)
    }

    /// Returns `1.0 / f_s_core` computed via 6-step `binary64` Newton reciprocal (`0 FDIV`).
    #[inline]
    pub fn inv_f_s_core(&self) -> f64 {
        rcp_nr6_f64(self.f_s_core)
    }
}

impl Default for ZohRateContext {
    /// Returns a nominal `48,000 Hz` live [`Tier::Lite`] context (`rho_vm = 1.0`, `k_ctrl = 16`).
    #[inline]
    fn default() -> Self {
        Self {
            reference_rate_hz: 48_000.0,
            host_rate_hz: 48_000.0,
            f_s_vm: 48_000.0,
            rho_vm: 1.0,
            os2x_active: false,
            f_s_core: 48_000.0,
            rho_core: 1.0,
            k_ctrl: 16,
            is_bounce_4x: false,
        }
    }
}

/// Rescales one `ParaLRU` lane `(alpha_rel, alpha_atk, kappa)` from reference rate to `f_s_core`
/// using `rho_core = f_ref / f_s_core` (`0 FDIV`, `ref[dsp.zoh.core-paralru]`).
#[inline]
pub fn rescale_paralru_lane(
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
    rho_core: f64,
) -> (f32, f32, f32) {
    let r_in = clamp_finite_f32(alpha_rel, 0.0, MAX_PARALRU_R_F32, 0.0);
    let atk_in = clamp_finite_f32(alpha_atk, 0.0, 1.0, 0.0);
    let kap_in = clamp_finite_f32(kappa, 0.0, 1.0, 0.0);
    if !rho_core.is_finite() || rho_core <= 0.0 || const_abs_f64(rho_core - 1.0) <= 1.0e-12 {
        return (r_in, atk_in, kap_in);
    }

    let r = f64::from(r_in);
    let atk = f64::from(atk_in);
    let kap = f64::from(kap_in);
    let beta = 1.0 - atk;
    let kb = kap * beta;
    let a_atk = r * (1.0 - kb);

    let alpha_rel_out = (powf_f64(r, rho_core) as f32).clamp(0.0, MAX_PARALRU_R_F32);
    let one_minus_a = 1.0 - a_atk;
    let ratio = if const_abs_f64(one_minus_a) <= 1.0e-12 {
        rho_core
    } else {
        (1.0 - powf_f64(a_atk, rho_core)) * rcp_nr6_f64(one_minus_a)
    };
    let beta_prime = (beta * ratio).clamp(0.0, 1.0);
    let alpha_atk_out = ((1.0 - beta_prime) as f32).clamp(0.0, 1.0);
    let kappa_out = if beta_prime > 1.0e-12 {
        let num = 1.0 - powf_f64(1.0 - kb, rho_core);
        ((num * rcp_nr6_f64(beta_prime)) as f32).clamp(0.0, 1.0)
    } else {
        kap_in
    };
    (alpha_rel_out, alpha_atk_out, kappa_out)
}

/// Computes continuous-time ZOH ballistic follower coefficient
/// `1 - exp(-1 / (rate_hz * tau_sec))` (`0 FDIV`, `ref[dsp.router.ballistic-envelope-bank]`).
#[inline]
pub fn ballistic_alpha_f32(rate_hz: f64, tau_sec: f64) -> f32 {
    if !rate_hz.is_finite() || rate_hz <= 0.0 || tau_sec.is_nan() {
        return 0.0;
    }
    if tau_sec <= 0.0 {
        return 1.0;
    }
    if !tau_sec.is_finite() {
        return 0.0;
    }
    let prod = rate_hz * tau_sec;
    if !prod.is_finite() {
        return 0.0;
    }
    ((1.0 - exp_f64(-rcp_nr6_f64(prod))) as f32).clamp(0.0, 1.0)
}

/// Rescales an integer sample count `base_48k` by `1 / rho_vm` using half-away-from-zero rounding
/// and clamping to `[min_samples, max_samples]` (`0 FDIV`, `ref[dsp.zoh.op-t-delays]`).
#[inline]
pub fn round_sample_count_zoh(
    base_48k: usize,
    rho_vm: f64,
    min_samples: usize,
    max_samples: usize,
) -> usize {
    let hi = max_samples.max(min_samples);
    if !rho_vm.is_finite() || rho_vm <= 0.0 {
        return base_48k.clamp(min_samples, hi);
    }
    let scaled = (base_48k as f64) * rcp_nr6_f64(rho_vm);
    let rounded = round_half_away_f64(scaled);
    if !rounded.is_finite() || rounded >= (hi as f64) {
        return hi;
    }
    if rounded <= (min_samples as f64) {
        return min_samples;
    }
    (rounded as usize).clamp(min_samples, hi)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_zoh_rate_context_rules_and_bounds() {
        assert_eq!(
            ZohRateContext::compute(44_099.0, 48_000.0, Tier::Full, true, false).unwrap_err(),
            VmError::InvalidSampleRate
        );
        assert_eq!(
            ZohRateContext::compute(96_001.0, 48_000.0, Tier::Full, true, false).unwrap_err(),
            VmError::InvalidSampleRate
        );
        assert_eq!(
            ZohRateContext::compute(48_000.0, 44_099.0, Tier::Full, true, false).unwrap_err(),
            VmError::InvalidSampleRate
        );
        assert_eq!(
            ZohRateContext::compute(48_000.0, 384_001.0, Tier::Full, true, false).unwrap_err(),
            VmError::InvalidSampleRate
        );
        assert_eq!(
            ZohRateContext::compute(48_000.0, f32::NAN, Tier::Full, true, false).unwrap_err(),
            VmError::InvalidSampleRate
        );
        assert_eq!(
            ZohRateContext::compute(48_000.0, 176_400.0, Tier::Ultra, true, true).unwrap_err(),
            VmError::InvalidSampleRate
        );

        let lite = ZohRateContext::compute(48_000.0, 48_000.0, Tier::Lite, true, false).unwrap();
        assert!(!lite.os2x_active);
        assert_eq!(lite.f_s_vm.to_bits(), 48_000.0_f64.to_bits());
        assert_eq!(lite.f_s_core.to_bits(), 48_000.0_f64.to_bits());
        assert_eq!(lite.k_ctrl, 16);

        let full_48 = ZohRateContext::compute(48_000.0, 48_000.0, Tier::Full, true, false).unwrap();
        assert!(full_48.os2x_active);
        assert_eq!(full_48.f_s_core.to_bits(), 96_000.0_f64.to_bits());
        assert!((full_48.rho_core - 0.5).abs() < 1.0e-12);
        assert_eq!(full_48.k_ctrl, 16);

        let sr441 = ZohRateContext::compute(48_000.0, 44_100.0, Tier::Pro, false, false).unwrap();
        assert!(!sr441.os2x_active);
        assert_eq!(sr441.k_ctrl, 15);

        let sr384 = ZohRateContext::compute(48_000.0, 384_000.0, Tier::Ultra, true, false).unwrap();
        assert!(!sr384.os2x_active);
        assert_eq!(sr384.f_s_core.to_bits(), 384_000.0_f64.to_bits());
        assert_eq!(sr384.k_ctrl, 128);

        let bounce96 =
            ZohRateContext::compute(48_000.0, 96_000.0, Tier::Ultra, true, true).unwrap();
        assert!(bounce96.is_bounce_4x);
        assert!(!bounce96.os2x_active);
        assert_eq!(bounce96.f_s_vm.to_bits(), 384_000.0_f64.to_bits());
        assert_eq!(bounce96.f_s_core.to_bits(), 384_000.0_f64.to_bits());
        assert_eq!(bounce96.k_ctrl, 128);
        assert!((bounce96.inv_f_s_vm() * 384_000.0 - 1.0).abs() < 1.0e-12);
        assert!((bounce96.inv_f_s_core() * 384_000.0 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_paralru_ballistic_and_sample_count_zoh() {
        let (r0, a0, k0) = rescale_paralru_lane(0.99, 0.5, 0.4, 1.0);
        assert!((r0 - 0.99).abs() < 1.0e-7);
        assert_eq!(a0.to_bits(), 0.5_f32.to_bits());
        assert!((k0 - 0.4).abs() < 1.0e-7);

        let (r_max_out, a_atk1, k_limit) = rescale_paralru_lane(2.0, 1.0, 0.75, 0.5);
        assert_eq!(r_max_out.to_bits(), MAX_PARALRU_R_F32.to_bits());
        assert_eq!(a_atk1.to_bits(), 1.0_f32.to_bits());
        assert_eq!(k_limit.to_bits(), 0.75_f32.to_bits());

        let (r_nan, a_nan, k_nan) = rescale_paralru_lane(f32::NAN, f32::INFINITY, -1.0, 0.5);
        assert_eq!(r_nan.to_bits(), 0);
        assert_eq!(a_nan.to_bits(), 0);
        assert_eq!(k_nan.to_bits(), 0);

        let alpha_fast = ballistic_alpha_f32(48_000.0, 0.0002);
        let expected_fast = (1.0 - exp_f64(-rcp_nr6_f64(9.6))) as f32;
        assert!((alpha_fast - expected_fast).abs() < 1.0e-7);
        assert_eq!(
            ballistic_alpha_f32(48_000.0, 0.0).to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(ballistic_alpha_f32(0.0, 0.01).to_bits(), 0);
        assert_eq!(ballistic_alpha_f32(48_000.0, f64::INFINITY).to_bits(), 0);

        assert_eq!(
            round_sample_count_zoh(16, 48_000.0 * rcp_nr6_f64(44_100.0), 1, 1024),
            15
        );
        assert_eq!(round_sample_count_zoh(48, 0.5, 16, 1024), 96);
        assert_eq!(round_sample_count_zoh(4, 1.0, 16, 1024), 16);
        assert_eq!(round_sample_count_zoh(2000, 0.5, 1, 1024), 1024);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_paralru_zoh_bounds_and_near_one_continuity(
            alpha_rel in 0.0_f32..=MAX_PARALRU_R_F32,
            alpha_atk in 0.0_f32..=1.0_f32,
            kappa in 0.0_f32..=1.0_f32,
            rho in 0.125_f64..=1.1_f64,
        ) {
            let (r, a, k) = rescale_paralru_lane(alpha_rel, alpha_atk, kappa, rho);
            prop_assert!(r.is_finite() && (0.0..=MAX_PARALRU_R_F32).contains(&r));
            prop_assert!(a.is_finite() && (0.0..=1.0).contains(&a));
            prop_assert!(k.is_finite() && (0.0..=1.0).contains(&k));

            let (r_lim, a_lim, k_lim) = rescale_paralru_lane(MAX_PARALRU_R_F32, 0.0, kappa, rho);
            prop_assert!(r_lim <= MAX_PARALRU_R_F32);
            prop_assert!((0.0..=1.0).contains(&a_lim));
            prop_assert!((0.0..=1.0).contains(&k_lim));
        }
    }
}
