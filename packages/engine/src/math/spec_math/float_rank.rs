//! Declared-range normalization and ordered admissible `f32` coordinate ranks
//! (`ref[dsp.ctrl.range.normalize-clamp]`, `ref[dsp.ctrl.range.f32-rank-bounds]`).

use super::{InvalidFloatInput, round_f64_to_f32_bits, widen_f32_bits};

/// Normalizes a physical `f32` value `x` into `[-1.0, +1.0]` against declared `f32` endpoints
/// `low < high` using cold-path `f64` reciprocal arithmetic (`ref[dsp.ctrl.range.normalize-clamp]`).
#[inline]
pub fn normalize_declared_range_bits(
    low_bits: u32,
    high_bits: u32,
    value_bits: u32,
) -> Option<f32> {
    let low = widen_f32_bits(low_bits)?;
    let high = widen_f32_bits(high_bits)?;
    let value = widen_f32_bits(value_bits)?;
    if !(low < high && value >= low && value <= high) {
        return None;
    }
    let span = high - low;
    let reciprocal = 1.0 / span;
    let distance = value - low;
    let product = distance * reciprocal;
    let coord = (2.0 * product - 1.0).clamp(-1.0, 1.0);
    let bits = round_f64_to_f32_bits(coord)?;
    Some(f32::from_bits(bits))
}

const F32_NEG_FIRST_KEY: u64 = 0x0080_0000;
const F32_NEG_NORMAL_END_KEY: u64 = 0x7f7f_ffff;
const F32_NEG_ZERO_KEY: u64 = 0x7fff_ffff;
const F32_POS_ZERO_KEY: u64 = 0x8000_0000;
const F32_POS_NORMAL_START_KEY: u64 = 0x8080_0000;
const F32_NORMALS_PER_SIGN: u64 = 254 * (1 << 23);

/// Returns the total cardinality of admissible `f32` physical knob values (`4,278,190,079` under
/// gradual underflow or `4,261,412,865` under FTZ/DAZ, `ref[dsp.ctrl.range.f32-rank-bounds]`).
#[inline]
pub const fn f32_coordinate_count(flush_subnormals: bool) -> u64 {
    if flush_subnormals {
        2 * F32_NORMALS_PER_SIGN + 1
    } else {
        2 * 255 * (1 << 23) - 1
    }
}

/// Maps an ordered rank `0 <= rank < f32_coordinate_count(flush_subnormals)` to its canonical `f32`
/// bit pattern (`+0.0` at the origin, `ref[dsp.ctrl.range.f32-rank-bounds]`).
#[inline]
pub const fn f32_bits_at_rank(rank: u64, flush_subnormals: bool) -> Option<u32> {
    let count = f32_coordinate_count(flush_subnormals);
    if rank >= count {
        return None;
    }
    let key = if flush_subnormals {
        if rank < F32_NORMALS_PER_SIGN {
            F32_NEG_FIRST_KEY + rank
        } else if rank == F32_NORMALS_PER_SIGN {
            F32_POS_ZERO_KEY
        } else {
            F32_POS_NORMAL_START_KEY + rank - F32_NORMALS_PER_SIGN - 1
        }
    } else {
        let mut k = F32_NEG_FIRST_KEY + rank;
        if k >= F32_NEG_ZERO_KEY {
            k += 1;
        }
        k
    };
    let bits = if (key & F32_POS_ZERO_KEY) != 0 {
        (key ^ F32_POS_ZERO_KEY) as u32
    } else {
        !(key as u32)
    };
    Some(bits)
}

/// Maps a finite `f32` bit pattern to its ordered rank in `[0, f32_coordinate_count(flush_subnormals))`
/// (`ref[dsp.ctrl.range.f32-rank-bounds]`). Returns `Ok(None)` if `bits` is a subnormal when
/// `flush_subnormals == true`, or `Err(InvalidFloatInput)` if `bits` is non-finite.
#[inline]
pub const fn f32_rank_of_bits(
    bits: u32,
    flush_subnormals: bool,
) -> Result<Option<u64>, InvalidFloatInput> {
    if ((bits >> 23) & 0xff) == 0xff {
        return Err(InvalidFloatInput);
    }
    let magnitude = bits & 0x7fff_ffff;
    let key = if magnitude == 0 {
        F32_POS_ZERO_KEY
    } else {
        if flush_subnormals && magnitude < 0x0080_0000 {
            return Ok(None);
        }
        if (bits & 0x8000_0000) != 0 {
            (!bits) as u64
        } else {
            (bits as u64) ^ F32_POS_ZERO_KEY
        }
    };
    let rank = if flush_subnormals {
        if key <= F32_NEG_NORMAL_END_KEY {
            key - F32_NEG_FIRST_KEY
        } else if key == F32_POS_ZERO_KEY {
            F32_NORMALS_PER_SIGN
        } else {
            F32_NORMALS_PER_SIGN + 1 + key - F32_POS_NORMAL_START_KEY
        }
    } else {
        key - F32_NEG_FIRST_KEY - if key >= F32_POS_ZERO_KEY { 1 } else { 0 }
    };
    Ok(Some(rank))
}

/// Binary-searches the rank boundary for a finite `binary64` bound (`upper == false` -> lower bound
/// `first >= value`; `upper == true` -> upper exclusive bound `first > value`, `ref[dsp.ctrl.range.f32-rank-bounds]`).
pub fn f32_rank_boundary_for_f64(value: f64, flush_subnormals: bool, upper: bool) -> Option<u64> {
    if !value.is_finite() {
        return None;
    }
    let mut low = 0u64;
    let mut high = f32_coordinate_count(flush_subnormals);
    while low < high {
        let mid = low + (high - low) / 2;
        let current = widen_f32_bits(f32_bits_at_rank(mid, flush_subnormals)?)?;
        if current < value || (upper && current == value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    Some(low)
}

/// Intersects slices of `binary64` lower and upper bounds onto the inward admissible `f32` rank
/// interval `[first, end)` (`first < end`, `ref[dsp.ctrl.range.f32-rank-bounds]`).
pub fn f32_rank_intersection(
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    flush_subnormals: bool,
) -> Option<(u64, u64)> {
    let count = f32_coordinate_count(flush_subnormals);
    let mut first = 0u64;
    for &lo in lower_bounds {
        let b = f32_rank_boundary_for_f64(lo, flush_subnormals, false)?;
        if b > first {
            first = b;
        }
    }
    let mut end = count;
    for &hi in upper_bounds {
        let b = f32_rank_boundary_for_f64(hi, flush_subnormals, true)?;
        if b < end {
            end = b;
        }
    }
    if first >= end {
        return None;
    }
    Some((first, end))
}

/// Maps any finite `f32` request bit pattern to the nearest admissible `f32` bit pattern in `[first, end)`
/// (ties toward zero, `ref[dsp.ctrl.range.f32-rank-bounds]`).
pub fn f32_nearest_admissible_bits(
    requested_bits: u32,
    first: u64,
    end: u64,
    flush_subnormals: bool,
) -> Option<u32> {
    let count = f32_coordinate_count(flush_subnormals);
    if !(first < end && end <= count) {
        return None;
    }
    let req_f64 = widen_f32_bits(requested_bits)?;
    let at = f32_rank_boundary_for_f64(req_f64, flush_subnormals, false)?;
    let idx0 = at.saturating_sub(1).clamp(first, end - 1);
    let idx1 = at.clamp(first, end - 1);
    let b0 = f32_bits_at_rank(idx0, flush_subnormals)?;
    if idx0 == idx1 {
        return Some(b0);
    }
    let b1 = f32_bits_at_rank(idx1, flush_subnormals)?;
    let v0 = widen_f32_bits(b0)?;
    let v1 = widen_f32_bits(b1)?;
    let d0 = (v0 - req_f64).abs();
    let d1 = (v1 - req_f64).abs();
    if d0 < d1 {
        Some(b0)
    } else if d1 < d0 {
        Some(b1)
    } else if v0.abs() < v1.abs() {
        Some(b0)
    } else if v1.abs() < v0.abs() {
        Some(b1)
    } else if v0 <= v1 {
        Some(b0)
    } else {
        Some(b1)
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::FtzDazGuard;

    /// verify[dsp.ctrl.range.normalize-clamp]
    /// verify[dsp.ctrl.range.f32-rank-bounds]
    /// Verifies `binary64` declared-range normalization (closely spaced `2^30` endpoints & wide `±f32::MAX` spans)
    /// and ordered admissible `f32` rank bounds (`4,278,190,079` vs `4,261,412,865`).
    #[test]
    fn test_declared_range_normalization_and_f32_rank_bounds() {
        let _guard = FtzDazGuard::new();

        // 1. Closely spaced endpoints 2^30 and 2^30 + 128, plus invalid/out-of-range guards
        let low = (1u64 << 30) as f32;
        let high = f32::from_bits(low.to_bits() + 1);
        assert_eq!(
            normalize_declared_range_bits(low.to_bits(), high.to_bits(), low.to_bits()),
            Some(-1.0)
        );
        assert_eq!(
            normalize_declared_range_bits(low.to_bits(), high.to_bits(), high.to_bits()),
            Some(1.0)
        );
        assert_eq!(
            normalize_declared_range_bits(1.0_f32.to_bits(), 0.0_f32.to_bits(), 0.5_f32.to_bits()),
            None
        );

        // 2. Extreme span [-f32::MAX, +f32::MAX]
        let max_bits = 0x7f7f_ffff_u32;
        let min_bits = 0xff7f_ffff_u32;
        assert_eq!(
            normalize_declared_range_bits(min_bits, max_bits, min_bits),
            Some(-1.0)
        );
        assert_eq!(
            normalize_declared_range_bits(min_bits, max_bits, 0),
            Some(0.0)
        );
        assert_eq!(
            normalize_declared_range_bits(min_bits, max_bits, max_bits),
            Some(1.0)
        );

        // 3. Cardinalities, rank round-trips, non-finite/subnormal guards, and signed normals
        assert_eq!(f32_coordinate_count(false), 4_278_190_079);
        assert_eq!(f32_coordinate_count(true), 4_261_412_865);
        assert_eq!(f32_rank_of_bits(0x7fc0_0000, false), Err(InvalidFloatInput));
        assert_eq!(f32_rank_of_bits(0x0000_0001, true), Ok(None));
        for flush in [false, true] {
            let count = f32_coordinate_count(flush);
            assert_eq!(f32_bits_at_rank(0, flush), Some(0xff7f_ffff));
            assert_eq!(f32_bits_at_rank(count - 1, flush), Some(0x7f7f_ffff));
            assert!(f32_bits_at_rank(u64::MAX, flush).is_none());
            let zero_rank = f32_rank_of_bits(0, flush).unwrap().unwrap();
            assert_eq!(f32_rank_of_bits(0x8000_0000, flush), Ok(Some(zero_rank)));
            assert_eq!(f32_bits_at_rank(zero_rank, flush), Some(0));
            for sample in [-1.5_f32, 1.5_f32] {
                let r = f32_rank_of_bits(sample.to_bits(), flush).unwrap().unwrap();
                assert_eq!(f32_bits_at_rank(r, flush), Some(sample.to_bits()));
            }
        }

        // 4. Inward binary64 bound intersection, single-rank clamp, and halfway ties toward zero
        assert!(f32_rank_intersection(&[0.1], &[0.1], false).is_none());
        let (first, end) =
            f32_rank_intersection(&[-2.0, -0.5, -1.0], &[2.0, 0.5, 1.0], false).unwrap();
        let lo_v = widen_f32_bits(f32_bits_at_rank(first, false).unwrap()).unwrap();
        let hi_v = widen_f32_bits(f32_bits_at_rank(end - 1, false).unwrap()).unwrap();
        assert!(lo_v >= -0.5 && hi_v <= 0.5);

        let min_norm_f64 = widen_f32_bits(0x0080_0000).unwrap();
        let (f_ftz, e_ftz) =
            f32_rank_intersection(&[-min_norm_f64], &[min_norm_f64], true).unwrap();
        assert_eq!(
            f32_nearest_admissible_bits(0x0040_0000, f_ftz, e_ftz, true),
            Some(0)
        );
        assert_eq!(
            f32_nearest_admissible_bits(0x8040_0000, f_ftz, e_ftz, true),
            Some(0)
        );
        assert_eq!(
            f32_nearest_admissible_bits(0x0040_0001, f_ftz, e_ftz, true),
            Some(0x0080_0000)
        );
        assert_eq!(
            f32_nearest_admissible_bits(0x8040_0001, f_ftz, e_ftz, true),
            Some(0x8080_0000)
        );
        assert!(f32_nearest_admissible_bits(0, 10, 10, false).is_none());
        let only_100 = f32_bits_at_rank(100, false).unwrap();
        assert_eq!(
            f32_nearest_admissible_bits(0, 100, 101, false),
            Some(only_100)
        );
        assert!(f32_rank_boundary_for_f64(f64::NAN, false, false).is_none());
    }
}
