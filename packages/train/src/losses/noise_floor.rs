//! Differentiable repeatability noise-floor SNR weighting and $C^\infty$ smooth excess-residual
//! softplus gating for Pass-2 training (`ref[qual.eval.thermal-hiss]`,
//! `ref[train.pipeline.backbone-trajectory-opt]`).

/// Stability epsilon in denominator of [`snr_weight`] (`1e-12`).
pub const SNR_EPS: f64 = 1.0e-12;
/// Excess-residual softplus smoothing fraction relative to measured noise floor (`0.05`).
pub const EXCESS_TAU_FRAC: f64 = 0.05;
/// Minimum softplus smoothing temperature floor (`1e-12`).
pub const EXCESS_TAU_MIN: f64 = 1.0e-12;

const _: () = assert!(SNR_EPS > 0.0);
const _: () = assert!(EXCESS_TAU_FRAC > 0.0 && EXCESS_TAU_FRAC < 1.0);
const _: () = assert!(EXCESS_TAU_MIN > 0.0);

/// Computes the band/scale SNR attenuation weight
/// `W_snr = max(0, target_energy - noise_energy) / (target_energy + SNR_EPS)`
/// (`ref[qual.eval.thermal-hiss]`). Returns `1.0` when `noise_energy <= 0.0`.
#[inline]
#[must_use]
pub fn snr_weight(target_energy: f64, noise_energy: f64) -> f64 {
    if !target_energy.is_finite() || target_energy < 0.0 {
        return 0.0;
    }
    if !noise_energy.is_finite() || noise_energy <= 0.0 {
        return 1.0;
    }
    let excess = (target_energy - noise_energy).max(0.0);
    (excess / (target_energy + SNR_EPS)).clamp(0.0, 1.0)
}

/// Evaluates the $C^\infty$ smooth excess squared residual above the measured noise floor
/// `tilde(e)^2 = tau * ln(1 + exp((res_sq - noise_energy) / tau))` alongside its exact
/// derivative `d(tilde(e)^2) / d(res_sq) = sigmoid((res_sq - noise_energy) / tau)`, where
/// `tau = max(0.05 * noise_energy, 1e-12)` (`ref[qual.eval.thermal-hiss]`).
///
/// Returns `(res_sq, 1.0)` identically when `noise_energy <= 0.0`.
#[inline]
#[must_use]
pub fn smooth_excess_sq(res_sq: f64, noise_energy: f64) -> (f64, f64) {
    if !noise_energy.is_finite() || noise_energy <= 0.0 {
        return (res_sq, 1.0);
    }
    let tau = (EXCESS_TAU_FRAC * noise_energy).max(EXCESS_TAU_MIN);
    let diff = res_sq - noise_energy;
    let z = diff / tau;
    if z >= 0.0 {
        let em = (-z).exp();
        let val = diff + tau * em.ln_1p();
        let deriv = 1.0 / (1.0 + em);
        (val, deriv)
    } else {
        let ep = z.exp();
        let val = tau * ep.ln_1p();
        let deriv = ep / (1.0 + ep);
        (val, deriv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_floor_zero_identity_snr_weights_and_fd_excess_deriv() {
        // Zero noise preserves exact noiseless identity.
        assert_eq!(snr_weight(1.0e-9, 0.0), 1.0);
        assert_eq!(snr_weight(1.23, -0.5), 1.0);
        assert_eq!(snr_weight(0.0, 0.0), 1.0);
        assert_eq!(smooth_excess_sq(0.42, 0.0), (0.42, 1.0));
        assert_eq!(smooth_excess_sq(0.42, -1.0), (0.42, 1.0));

        // Non-finite target energy safely attenuates to zero.
        assert_eq!(snr_weight(f64::NAN, 1.0e-3), 0.0);
        assert_eq!(snr_weight(-1.0, 1.0e-3), 0.0);

        // Positive noise below/at/above floor.
        assert_eq!(snr_weight(1.0e-4, 2.0e-4), 0.0);
        let w = snr_weight(1.0, 0.25);
        assert!((w - 0.75 / (1.0 + SNR_EPS)).abs() < 1.0e-12);

        // Overflow/underflow safety across extreme regimes.
        let (v_hi, d_hi) = smooth_excess_sq(1.0e6, 1.0e-3);
        assert!(v_hi.is_finite() && (v_hi - (1.0e6 - 1.0e-3)).abs() < 1.0e-9);
        assert!((d_hi - 1.0).abs() < 1.0e-12);

        let (v_lo, d_lo) = smooth_excess_sq(0.0, 1.0);
        assert!(v_lo.is_finite() && v_lo < 1.0e-8 && d_lo < 1.0e-8);

        // Exact central finite-difference derivative check around threshold and knees.
        let noise = 0.08_f64;
        let eps = 1.0e-7_f64;
        for &r_sq in &[0.02, 0.075, 0.08, 0.085, 0.25] {
            let (_val, deriv) = smooth_excess_sq(r_sq, noise);
            let (vp, _) = smooth_excess_sq(r_sq + eps, noise);
            let (vm, _) = smooth_excess_sq(r_sq - eps, noise);
            let fd = (vp - vm) / (2.0 * eps);
            assert!(
                (deriv - fd).abs() < 1.0e-8,
                "FD mismatch at r_sq={r_sq}: analytic={deriv}, fd={fd}"
            );
        }
    }
}
