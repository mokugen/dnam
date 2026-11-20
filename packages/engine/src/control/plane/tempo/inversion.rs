//! Physical `f32` calibration curve inversion and coprime FDN prime verification
//! (`ref[dsp.tempo.physical-inversion]`, `ref[dsp.tempo.coprime-fdn]`).

use crate::math::{f32_bits_at_rank, f32_rank_intersection, widen_f32_bits};

/// Evaluates the 3-node physical calibration curve $v(u) = \frac{1}{2} v_{\min} u (u - 1) + v_{\text{mid}}(1 - u^2) + \frac{1}{2} v_{\max} u (u + 1)$
/// (`ref[dsp.tempo.physical-inversion]`).
#[inline]
pub fn eval_3node_calibration_f64(anchors: [f64; 3], u: f64) -> f64 {
    let [lo, mid, hi] = anchors;
    0.5 * lo * u * (u - 1.0) + mid * (1.0 - u * u) + 0.5 * hi * u * (u + 1.0)
}

/// Result of inverting a 3-node physical calibration curve over the admissible `f32` rank set
/// (`ref[dsp.tempo.physical-inversion]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoPhysicalInversionResult {
    pub physical_f32: f32,
    pub normalized_f32: f32,
    pub realized_value: f64,
    pub residual: f64,
    pub search_steps: (u32, u32),
}

/// Inverts a monotone 3-node physical calibration curve `anchors = [v_min, v_mid, v_max]` over the
/// admissible `f32` physical rank interval `[first, end)` for `declared = (decl_lo, decl_hi)` and
/// `safe = (safe_lo, safe_hi)` in at most 32 binary-search comparisons per pass, breaking ties toward
/// the lower physical rank (`ref[dsp.tempo.physical-inversion]`).
pub fn invert_tempo_physical_calibration(
    declared: (f32, f32),
    safe: (f32, f32),
    anchors: [f64; 3],
    target: f64,
    flush_subnormals: bool,
) -> Option<TempoPhysicalInversionResult> {
    if !target.is_finite()
        || !declared.0.is_finite()
        || !declared.1.is_finite()
        || !safe.0.is_finite()
        || !safe.1.is_finite()
        || declared.0 >= declared.1
        || safe.0 > safe.1
    {
        return None;
    }
    let lo_phys = declared.0.max(safe.0);
    let hi_phys = declared.1.min(safe.1);
    if lo_phys > hi_phys {
        return None;
    }
    let (first, end) =
        f32_rank_intersection(&[lo_phys as f64], &[hi_phys as f64], flush_subnormals)?;

    let [v_lo, v_mid, v_hi] = anchors;
    let c1 = 0.5 * (v_hi - v_lo);
    let c2 = 0.5 * (v_hi + v_lo) - v_mid;
    let target_delta = target - v_mid;

    let eval_at_rank = |rank: u64| -> Option<(f32, f32, f64, f64)> {
        let p_bits = f32_bits_at_rank(rank, flush_subnormals)?;
        let low = widen_f32_bits(declared.0.to_bits())?;
        let high = widen_f32_bits(declared.1.to_bits())?;
        let val = widen_f32_bits(p_bits)?;
        let span = high - low;
        let inv_span = crate::math::newton6_reciprocal_f64(span);
        let num = if (low < 0.0) != (high < 0.0) {
            2.0 * val - (low + high)
        } else {
            (val - low) + (val - high)
        };
        let coord = (num * inv_span).clamp(-1.0, 1.0);
        let mut u = f32::from_bits(crate::math::round_f64_to_f32_bits(coord)?);
        if flush_subnormals && (u.to_bits() & 0x7fff_ffff) < 0x0080_0000 {
            u = 0.0;
        }
        let u64_val = widen_f32_bits(u.to_bits())?;
        let delta_v = u64_val * (c1 + c2 * u64_val);
        let v = v_mid + delta_v;
        Some((f32::from_bits(p_bits), u, v, delta_v))
    };

    let (_, _, _, dv_first) = eval_at_rank(first)?;
    let (_, _, _, dv_last) = eval_at_rank(end - 1)?;
    let ascending = dv_last >= dv_first;

    let mut low = first;
    let mut high = end;
    let mut crossing_steps = 0u32;
    while low < high {
        let mid = low + (high - low) / 2;
        let (_, _, _, dv) = eval_at_rank(mid)?;
        crossing_steps += 1;
        let before = if ascending {
            dv < target_delta
        } else {
            dv > target_delta
        };
        if before {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    let cand0 = low.saturating_sub(1).clamp(first, end - 1);
    let cand1 = low.clamp(first, end - 1);
    let (_, _, _, dv0) = eval_at_rank(cand0)?;
    let (_, _, _, dv1) = eval_at_rank(cand1)?;
    let best = if (dv0 - target_delta).abs() <= (dv1 - target_delta).abs() {
        cand0
    } else {
        cand1
    };
    let (_, _, _, chosen_dv) = eval_at_rank(best)?;

    // Second pass: find lowest rank on any constant plateau matching `chosen_dv`
    let mut p_low = first;
    let mut p_high = best;
    let mut plateau_steps = 0u32;
    while p_low < p_high {
        let mid = p_low + (p_high - p_low) / 2;
        let (_, _, _, dv) = eval_at_rank(mid)?;
        plateau_steps += 1;
        let before = if ascending {
            dv < chosen_dv
        } else {
            dv > chosen_dv
        };
        if before {
            p_low = mid + 1;
        } else {
            p_high = mid;
        }
    }
    let (physical_f32, normalized_f32, realized_value, realized_dv) = eval_at_rank(p_low)?;
    Some(TempoPhysicalInversionResult {
        physical_f32,
        normalized_f32,
        realized_value,
        residual: (realized_dv - target_delta).abs(),
        search_steps: (crossing_steps, plateau_steps),
    })
}

/// Prime reference lengths for Spatial Tank FDN lines (`ref[dsp.tempo.coprime-fdn]`).
pub const FDN_COPRIME_BASE_PRIMES: [u32; 8] = [149, 211, 307, 401, 503, 601, 701, 887];

/// Verifies that for any of the 144 `(idx_l, idx_r)` stereo subdivision pairs (`0..12 × 0..12`),
/// the scaled FDN modal delay ratios $\alpha_m = P_m / P_{\text{ref}}$ preserve distinct prime
/// factors (`gcd(P_i, P_j) == 1` within each channel, `ref[dsp.tempo.coprime-fdn]`).
pub const fn verify_coprime_fdn_pair(subdiv_l: usize, subdiv_r: usize) -> bool {
    if subdiv_l >= 12 || subdiv_r >= 12 {
        return false;
    }
    let mut i = 0;
    while i < 8 {
        let mut j = i + 1;
        while j < 8 {
            let mut a = FDN_COPRIME_BASE_PRIMES[i];
            let mut b = FDN_COPRIME_BASE_PRIMES[j];
            while b != 0 {
                let r = a % b;
                a = b;
                b = r;
            }
            if a != 1 {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_inversion_bounds_descending_ftz_and_coprime_guards() {
        assert_eq!(eval_3node_calibration_f64([10.0, 25.0, 50.0], -1.0), 10.0);
        assert_eq!(eval_3node_calibration_f64([10.0, 25.0, 50.0], 0.0), 25.0);
        assert_eq!(eval_3node_calibration_f64([10.0, 25.0, 50.0], 1.0), 50.0);

        assert!(
            invert_tempo_physical_calibration((5.0, 1.0), (1.0, 5.0), [0.0, 1.0, 2.0], 1.0, false)
                .is_none()
        );
        assert!(
            invert_tempo_physical_calibration((0.0, 1.0), (2.0, 3.0), [0.0, 1.0, 2.0], 1.0, false)
                .is_none()
        );

        let desc = invert_tempo_physical_calibration(
            (-1.0, 1.0),
            (-1.0, 1.0),
            [100.0, 50.0, 10.0],
            80.0,
            true,
        )
        .unwrap();
        assert!(desc.physical_f32 < 0.0 && desc.residual < 1e-4);

        let desc_mid = invert_tempo_physical_calibration(
            (-1.0, 1.0),
            (-1.0, 1.0),
            [100.0, 50.0, 10.0],
            50.0,
            true,
        )
        .unwrap();
        assert_eq!(desc_mid.physical_f32, 0.0);
        assert_eq!(desc_mid.normalized_f32, 0.0);

        assert!(!verify_coprime_fdn_pair(12, 0));
        assert!(!verify_coprime_fdn_pair(0, 12));
        assert!(verify_coprime_fdn_pair(0, 11));
    }
}
