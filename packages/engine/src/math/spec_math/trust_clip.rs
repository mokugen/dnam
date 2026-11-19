//! Outward-checked 4-D/8-D latent trust-region clipping (`ref[dsp.ctrl.op-n.trust-clip]`,
//! `ref[dsp.op-n.ctrl.trust-clip]`).

use super::ieee_bits::{f32_unbiased_exponent, round_f64_to_f32_bits, widen_f32_bits};

/// Outward step (+1 ULP) on a strictly positive finite `f64` value (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub const fn next_up_pos_f64(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

/// Outward-rounded product of two non-negative `f64` values (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub fn upper_product_f64(left: f64, right: f64) -> f64 {
    if left == 0.0 || right == 0.0 {
        0.0
    } else {
        next_up_pos_f64(left * right)
    }
}

/// Outward-rounded sum of squares $\mathrm{add\text{-}up}(\mathrm{initial}, \sum_i x_i^2)$ (`ref[dsp.ctrl.op-n.trust-clip]`).
#[inline]
pub fn upper_sum_squares_f64(values: &[f64], initial: f64) -> f64 {
    let mut total = initial;
    for &v in values {
        let sq = v * v;
        if sq != 0.0 {
            total = if total == 0.0 {
                sq
            } else {
                next_up_pos_f64(total + sq)
            };
        }
    }
    total
}

#[inline]
fn trust_clip_candidate<const D: usize>(
    x: &[f64; D],
    radius_squared: f64,
    energy_upper: f64,
    scale: f64,
) -> Option<(bool, [u32; D])> {
    let mut out_bits = [0u32; D];
    let mut rounded_f64 = [0.0f64; D];
    for i in 0..D {
        let b = round_f64_to_f32_bits(x[i] * scale)?;
        out_bits[i] = b;
        rounded_f64[i] = widen_f32_bits(b)?;
    }
    let scale_sq_upper = upper_product_f64(scale, scale);
    let ideal_upper = upper_product_f64(scale_sq_upper, energy_upper);
    let rounded_upper = upper_sum_squares_f64(&rounded_f64, 0.0);
    let accepted = ideal_upper <= radius_squared && rounded_upper <= radius_squared;
    Some((accepted, out_bits))
}

#[inline]
const fn inverse_sqrt_seed_f64(energy_upper: f64) -> f64 {
    let bits = energy_upper.to_bits();
    let exponent = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let half_exp = (exponent + 2).div_euclid(2);
    let inv_bits = ((1023 - half_exp) as u64) << 52;
    f64::from_bits(inv_bits)
}

/// Evaluates the 6-step Newton fast path for outward-checked trust clipping (`ref[dsp.ctrl.op-n.trust-clip]`).
pub fn trust_clip_fast_candidate<const D: usize>(
    values_bits: &[u32; D],
    radius_bits: u32,
    epsilon_bits: u32,
) -> Option<(bool, [u32; D], f64)> {
    let mut x = [0.0f64; D];
    for i in 0..D {
        x[i] = widen_f32_bits(values_bits[i])?;
    }
    let radius = widen_f32_bits(radius_bits)?;
    let epsilon = widen_f32_bits(epsilon_bits)?;
    if radius <= 0.0 || epsilon <= 0.0 {
        return None;
    }
    let energy_upper = upper_sum_squares_f64(&x, epsilon);
    let mut inverse = inverse_sqrt_seed_f64(energy_upper);
    for _ in 0..6 {
        let sq = (energy_upper * inverse) * inverse;
        inverse *= 1.5 - 0.5 * sq;
    }
    let guard = 1.0 - f64::from_bits((1023 - 20) << 52);
    let scale = guard.min((radius * inverse) * guard);
    let (accepted, out) = trust_clip_candidate(&x, radius * radius, energy_upper, scale)?;
    Some((accepted, out, scale))
}

/// Evaluates the outward-checked trust-region clip on `D` (`4` or `8`) `f32` bit patterns
/// using the 6-step Newton fast path and 32-step bisection fallback (`ref[dsp.ctrl.op-n.trust-clip]`).
pub fn trust_clip_outward_checked_bits<const D: usize>(
    values_bits: &[u32; D],
    radius_bits: u32,
    epsilon_bits: u32,
) -> Option<([u32; D], f64, f64)> {
    if D != 4 && D != 8 {
        return None;
    }
    let mut x = [0.0f64; D];
    for i in 0..D {
        x[i] = widen_f32_bits(values_bits[i])?;
    }
    let radius = widen_f32_bits(radius_bits)?;
    let epsilon = widen_f32_bits(epsilon_bits)?;
    if radius <= 0.0 || epsilon <= 0.0 {
        return None;
    }
    let ideal_energy_upper = upper_sum_squares_f64(&x, epsilon);
    let radius_squared = radius * radius;

    let (accepted_one, orig_bits) =
        trust_clip_candidate(&x, radius_squared, ideal_energy_upper, 1.0)?;
    if accepted_one {
        return Some((orig_bits, 1.0, 1.0));
    }

    if let Some((true, fast_bits, fast_scale)) =
        trust_clip_fast_candidate(values_bits, radius_bits, epsilon_bits)
    {
        return Some((fast_bits, fast_scale, 1.0));
    }

    trust_clip_reference_bisection_bits(values_bits, radius_bits, epsilon_bits)
}

/// Evaluates the canonical 32-step reference bisection clip returning
/// `(clipped_bits, accepted_scale, initial_high)`.
pub fn trust_clip_reference_bisection_bits<const D: usize>(
    values_bits: &[u32; D],
    radius_bits: u32,
    epsilon_bits: u32,
) -> Option<([u32; D], f64, f64)> {
    if D != 4 && D != 8 {
        return None;
    }
    let mut x = [0.0f64; D];
    for i in 0..D {
        x[i] = widen_f32_bits(values_bits[i])?;
    }
    let radius = widen_f32_bits(radius_bits)?;
    let epsilon = widen_f32_bits(epsilon_bits)?;
    if radius <= 0.0 || epsilon <= 0.0 {
        return None;
    }
    let ideal_energy_upper = upper_sum_squares_f64(&x, epsilon);
    let radius_squared = radius * radius;

    let (accepted_one, orig_bits) =
        trust_clip_candidate(&x, radius_squared, ideal_energy_upper, 1.0)?;
    if accepted_one {
        return Some((orig_bits, 1.0, 1.0));
    }

    let eps_exp_half = f32_unbiased_exponent(epsilon_bits)?.div_euclid(2);
    let mut norm_exp = eps_exp_half;
    for &b in values_bits {
        if let Some(e) = f32_unbiased_exponent(b)
            && e > norm_exp
        {
            norm_exp = e;
        }
    }
    let rad_exp = f32_unbiased_exponent(radius_bits)?;
    let high_exp = rad_exp + 1 - norm_exp;
    let mut high = if high_exp >= 0 {
        1.0
    } else {
        f64::from_bits(((high_exp + 1023) as u64) << 52)
    };
    let initial_high = high;
    let mut low = 0.0_f64;
    let mut result = [0u32; D];
    for i in 0..D {
        result[i] = values_bits[i] & 0x8000_0000;
    }
    for _ in 0..32 {
        let mid = (low + high) * 0.5;
        let (acc, cand) = trust_clip_candidate(&x, radius_squared, ideal_energy_upper, mid)?;
        if acc {
            low = mid;
            result = cand;
        } else {
            high = mid;
        }
    }
    Some((result, low, initial_high))
}

/// Default outward trust-clip regularizer `epsilon` (`f32::MIN_POSITIVE = 2^-126`, `0x0080_0000`).
const DEFAULT_TRUST_CLIP_EPS_BITS: u32 = f32::MIN_POSITIVE.to_bits();

/// Normative squared-radius margin (`1e-9`) for the interior trust-ball fast exit
/// (`ref[dsp.ctrl.op-n.static-and-interior-fastpath]`).
const TRUST_CLIP_INTERIOR_MARGIN_F64: f64 = 1.0e-9;

const _: () = assert!(DEFAULT_TRUST_CLIP_EPS_BITS == 0x0080_0000);

#[inline]
fn trust_clip_outward_nd<const D: usize>(z: &mut [f32; D], rho_max: f32) {
    if !rho_max.is_finite() || rho_max <= 0.0 {
        *z = [0.0; D];
        return;
    }

    let mut norm2 = 0.0_f64;
    let mut all_finite = true;
    for &v in z.iter() {
        if v.is_finite() {
            let v64 = v as f64;
            norm2 += v64 * v64;
        } else {
            all_finite = false;
        }
    }
    let r64 = rho_max as f64;
    if all_finite && norm2.is_finite() && norm2 + TRUST_CLIP_INTERIOR_MARGIN_F64 < r64 * r64 {
        return;
    }

    let in_bits: [u32; D] = core::array::from_fn(|i| {
        if z[i].is_finite() {
            z[i].to_bits()
        } else {
            0.0_f32.to_bits()
        }
    });
    if let Some((out_bits, _, _)) =
        trust_clip_outward_checked_bits(&in_bits, rho_max.to_bits(), DEFAULT_TRUST_CLIP_EPS_BITS)
    {
        for i in 0..D {
            z[i] = f32::from_bits(out_bits[i]);
        }
    } else {
        *z = [0.0; D];
    }
}

/// Applies outward-checked trust-region clipping in place on a 4-D `f32` group (`ref[dsp.ctrl.op-n.trust-clip]`,
/// `ref[dsp.ctrl.op-n.static-and-interior-fastpath]`).
#[inline]
pub fn trust_clip_outward_4d(z: &mut [f32; 4], rho_max: f32) {
    trust_clip_outward_nd(z, rho_max);
}

/// Applies outward-checked trust-region clipping in place on an 8-D `f32` group (`ref[dsp.ctrl.op-n.trust-clip]`,
/// `ref[dsp.ctrl.op-n.static-and-interior-fastpath]`).
#[inline]
pub fn trust_clip_outward_8d(z: &mut [f32; 8], rho_max: f32) {
    trust_clip_outward_nd(z, rho_max);
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::FtzDazGuard;
    use crate::test_support::SplitMix64;

    /// verify[dsp.ctrl.op-n.trust-clip]
    /// verify[dsp.op-n.ctrl.trust-clip]
    #[test]
    fn test_outward_checked_trust_region_clip_4d_and_8d() {
        let _guard = FtzDazGuard::new();
        let eps_bits = 0x0080_0000_u32;
        let radius_bits = 2.5_f32.to_bits();
        let r_f64 = widen_f32_bits(radius_bits).unwrap();

        let inside = [1_u32, 0x8000_0001, 0x3f00_0000, 0xbf00_0000];
        let (out_in, scale_in, _) =
            trust_clip_outward_checked_bits(&inside, radius_bits, eps_bits).unwrap();
        assert_eq!(out_in, inside);
        assert_eq!(scale_in, 1.0);

        let mut rng = SplitMix64::new(0x7257_c119);
        for _ in 0..500 {
            let v4: [u32; 4] = core::array::from_fn(|_| rng.next_f32_range(-20.0, 20.0).to_bits());
            let (fast_acc, fast_out, _) =
                trust_clip_fast_candidate(&v4, radius_bits, eps_bits).unwrap();
            assert!(fast_acc);
            let (out4, scale4, _) =
                trust_clip_outward_checked_bits(&v4, radius_bits, eps_bits).unwrap();
            assert_eq!(out4, if scale4 == 1.0 { v4 } else { fast_out });
            let sq_sum: f64 = out4
                .iter()
                .map(|&b| {
                    let w = widen_f32_bits(b).unwrap();
                    w * w
                })
                .sum();
            assert!(sq_sum <= r_f64 * r_f64);

            let v8: [u32; 8] = core::array::from_fn(|_| rng.next_f32_range(-20.0, 20.0).to_bits());
            let (out8, _, _) = trust_clip_outward_checked_bits(&v8, radius_bits, eps_bits).unwrap();
            let (ref8, _, _) =
                trust_clip_reference_bisection_bits(&v8, radius_bits, eps_bits).unwrap();
            let sq8: f64 = out8
                .iter()
                .map(|&b| {
                    let w = widen_f32_bits(b).unwrap();
                    w * w
                })
                .sum();
            let sq_ref8: f64 = ref8
                .iter()
                .map(|&b| {
                    let w = widen_f32_bits(b).unwrap();
                    w * w
                })
                .sum();
            assert!(sq8 <= r_f64 * r_f64 && sq_ref8 <= r_f64 * r_f64);
        }

        let mut z4 = [3.0_f32, -4.0, 2.0, -1.0];
        trust_clip_outward_4d(&mut z4, 2.5);
        assert!(z4.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() <= 6.25);

        let mut z8 = [3.0_f32, -4.0, 2.0, -1.0, 1.5, -2.5, 0.5, -3.5];
        trust_clip_outward_8d(&mut z8, 2.5);
        assert!(z8.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() <= 6.25);

        // Normative interior fast-exit preserves bit-exact coordinates (`norm2 + 1e-9 < rho^2`).
        let mut z4_fast = [0.5_f32, -0.5, 0.25, -0.25];
        let z4_fast_bits = z4_fast.map(f32::to_bits);
        trust_clip_outward_4d(&mut z4_fast, 2.0);
        assert_eq!(z4_fast.map(f32::to_bits), z4_fast_bits);

        let mut z8_fast = [0.25_f32; 8];
        let z8_fast_bits = z8_fast.map(f32::to_bits);
        trust_clip_outward_8d(&mut z8_fast, 2.0);
        assert_eq!(z8_fast.map(f32::to_bits), z8_fast_bits);

        // Non-positive or non-finite radius safely zeroes the vector.
        for bad_rho in [
            0.0_f32,
            -0.0,
            -1.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut z4_bad = [1.0_f32, -2.0, 3.0, -4.0];
            trust_clip_outward_4d(&mut z4_bad, bad_rho);
            assert_eq!(z4_bad, [0.0; 4]);

            let mut z8_bad = [1.0_f32; 8];
            trust_clip_outward_8d(&mut z8_bad, bad_rho);
            assert_eq!(z8_bad, [0.0; 8]);
        }

        // Non-finite vector entries are sanitized to 0.0 before clipping.
        let mut z4_nan = [f32::NAN, 3.0, f32::INFINITY, f32::NEG_INFINITY];
        trust_clip_outward_4d(&mut z4_nan, 2.0);
        assert!(z4_nan.iter().all(|v| v.is_finite()));
        assert!((z4_nan[1] - 2.0).abs() <= 1.0e-5 && z4_nan[0].to_bits() == 0);

        let mut z8_nan = [f32::NAN, 0.5, f32::INFINITY, -0.5, 0.0, 0.0, 0.0, 0.0];
        trust_clip_outward_8d(&mut z8_nan, 2.0);
        assert_eq!(z8_nan, [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(upper_product_f64(0.0, 5.0), 0.0);
        assert_eq!(upper_product_f64(5.0, 0.0), 0.0);
        assert!(trust_clip_fast_candidate(&[0u32; 4], 0, 0x0080_0000).is_none());
        assert!(trust_clip_fast_candidate(&[0u32; 4], 1.0_f32.to_bits(), 0).is_none());
        assert!(
            trust_clip_outward_checked_bits(&[0u32; 3], 1.0_f32.to_bits(), 0x0080_0000).is_none()
        );
        assert!(trust_clip_outward_checked_bits(&[0u32; 4], 0, 0x0080_0000).is_none());
        assert!(
            trust_clip_reference_bisection_bits(&[0u32; 3], 1.0_f32.to_bits(), 0x0080_0000)
                .is_none()
        );
        assert!(trust_clip_reference_bisection_bits(&[0u32; 4], 0, 0x0080_0000).is_none());
        assert_eq!(
            trust_clip_reference_bisection_bits(&[0u32; 4], 1.0_f32.to_bits(), 0x0080_0000),
            Some(([0u32; 4], 1.0, 1.0))
        );
    }

    proptest! {
        #[test]
        fn prop_outward_trust_clip_strictly_bounded(
            v4 in prop::array::uniform4(-25.0_f32..25.0_f32),
            radius in 0.25_f32..20.0_f32,
        ) {
            let _guard = FtzDazGuard::new();
            let v4_bits = v4.map(f32::to_bits);
            let (out_bits, _, _) = trust_clip_outward_checked_bits(
                &v4_bits,
                radius.to_bits(),
                1.0e-12_f32.to_bits(),
            ).unwrap();
            let sq_sum: f64 = out_bits.iter().map(|&b| {
                let w = widen_f32_bits(b).unwrap();
                w * w
            }).sum();
            let r64 = widen_f32_bits(radius.to_bits()).unwrap();
            prop_assert!(sq_sum <= r64 * r64);
        }
    }
}
