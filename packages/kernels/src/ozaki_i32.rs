//! Fast-math-immune `3 × 8-bit` `i32` Ozaki mantissa slicing and exact chunked dot-product
//! accumulation (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[repo.code.shared-nostd-kernels]`).
//!
//! Decomposes each normalized `f32` operand into 3 signed `[-128, 128]` slices (`2^-7, 2^-15, 2^-23`)
//! using branchless `1.5 * 2^23` IEEE-754 RNE rounding (zero `libm`, zero `FDIV`, zero `u64`/`f64` on
//! `spirv`) and accumulates up to [`OZAKI_MAX_CHUNK_LEN`] (`1024`) products strictly in `i32`
//! registers before host `f64` diagonal reconstruction.

/// Maximum number of elements per `i32` Ozaki chunk (`1024 * 128 * 128 = 2^24 << i32::MAX`).
pub const OZAKI_MAX_CHUNK_LEN: usize = 1024;

/// IEEE-754 `1.5 * 2^23` (`12_582_912.0`) RNE integer rounding bias (`0x4B40_0000`).
const RNE_MAGIC_F32: f32 = 1.5 * ((1_u32 << 23) as f32);
const RNE_MAGIC_BITS: i32 = RNE_MAGIC_F32.to_bits() as i32;

/// Slice multipliers and exact negative power-of-2 reconstruction weights (`2^7`, `2^15`, `2^23`).
const SLICE0_SCALE_F32: f32 = (1_u32 << 7) as f32;
const SLICE0_INV_F32: f32 = f32::from_bits((127 - 7) << 23);
const SLICE1_SCALE_F32: f32 = (1_u32 << 15) as f32;
const SLICE1_INV_F32: f32 = f32::from_bits((127 - 15) << 23);
const SLICE2_SCALE_F32: f32 = (1_u32 << 23) as f32;

const _: () = assert!((OZAKI_MAX_CHUNK_LEN as u32) * 128 * 128 <= i32::MAX as u32);
const _: () = assert!(RNE_MAGIC_BITS == 0x4B40_0000);
const _: () = assert!(SLICE0_SCALE_F32 * SLICE0_INV_F32 == 1.0);
const _: () = assert!(SLICE1_SCALE_F32 * SLICE1_INV_F32 == 1.0);

#[inline(always)]
fn to_i32_rne_f32(x: f32) -> i32 {
    (x + RNE_MAGIC_F32).to_bits() as i32 - RNE_MAGIC_BITS
}

/// Computes the smallest power-of-two ceiling `scale >= max_abs` and its exact reciprocal
/// `inv_scale = 1.0 / scale` in pure `binary32` bit arithmetic (`(scale, inv_scale)`).
///
/// Returns `(1.0, 1.0)` if `max_abs <= 0.0` or non-finite, and clamps the ceiling exponent to
/// `[1, 253]` (`[2^-126, 2^126]`) so both `scale` and `inv_scale` remain exact normal `f32` powers
/// of two with `scale * inv_scale == 1.0`.
#[inline(always)]
#[must_use]
pub fn pow2_ceil_fp32(max_abs: f32) -> (f32, f32) {
    let bits = max_abs.to_bits();
    if (bits & 0x8000_0000) != 0 || bits == 0 || bits >= 0x7F80_0000 {
        return (1.0, 1.0);
    }
    let ceil_bits = bits.wrapping_add(0x007F_FFFF) & 0x7F80_0000;
    let e = (ceil_bits >> 23).clamp(1, 253);
    let scale = f32::from_bits(e << 23);
    let inv_scale = f32::from_bits((254 - e) << 23);
    (scale, inv_scale)
}

/// Decomposes `val * inv_scale` (normalized to `[-1.0, 1.0]`) into 3 signed `[-128, 128]` slices
/// `[s0, s1, s2]` representing `s0 * 2^-7 + s1 * 2^-15 + s2 * 2^-23`.
///
/// Returns `[0, 0, 0]` if `val` or `inv_scale` is non-finite or produces a non-finite product.
#[inline(always)]
#[must_use]
pub fn slice_f32_mantissa_i8x3(val: f32, inv_scale: f32) -> [i32; 3] {
    if !val.is_finite() || !inv_scale.is_finite() {
        return [0, 0, 0];
    }
    let v_raw = val * inv_scale;
    if !v_raw.is_finite() {
        return [0, 0, 0];
    }
    let v = v_raw.clamp(-1.0, 1.0);
    let s0 = to_i32_rne_f32(v * SLICE0_SCALE_F32);
    let rem1 = v - (s0 as f32) * SLICE0_INV_F32;
    let s1 = to_i32_rne_f32(rem1 * SLICE1_SCALE_F32);
    let rem2 = rem1 - (s1 as f32) * SLICE1_INV_F32;
    let s2 = to_i32_rne_f32(rem2 * SLICE2_SCALE_F32);
    [s0, s1, s2]
}

/// Accumulates one pair of 3-slice mantissa vectors `(sx, sy)` into the 6-word upper-triangle
/// diagonal layout `(d0, d1, d2, d3, d4, reserved_pad)` strictly in `i32` registers.
#[inline(always)]
pub fn accum_ozaki_pair_i32(acc: &mut [i32; 6], sx: [i32; 3], sy: [i32; 3]) {
    acc[0] += sx[0] * sy[0];
    acc[1] += sx[0] * sy[1] + sx[1] * sy[0];
    acc[2] += sx[0] * sy[2] + sx[1] * sy[1] + sx[2] * sy[0];
    acc[3] += sx[1] * sy[2] + sx[2] * sy[1];
    acc[4] += sx[2] * sy[2];
}

/// Accumulates an Ozaki dot-product chunk over `min(x.len(), y.len(), OZAKI_MAX_CHUNK_LEN)` frames
/// strictly in `i32` diagonal accumulators `[d0, d1, d2, d3, d4, 0]`.
#[inline]
#[must_use]
pub fn accum_ozaki_chunk_i32(x: &[f32], y: &[f32], inv_scale_x: f32, inv_scale_y: f32) -> [i32; 6] {
    let mut acc = [0_i32; 6];
    let n = x.len().min(y.len()).min(OZAKI_MAX_CHUNK_LEN);
    for i in 0..n {
        let sx = slice_f32_mantissa_i8x3(x[i], inv_scale_x);
        let sy = slice_f32_mantissa_i8x3(y[i], inv_scale_y);
        accum_ozaki_pair_i32(&mut acc, sx, sy);
    }
    acc
}

/// Reconstructs an `f64` inner product from accumulated 6-word Ozaki diagonal sums on the host.
#[cfg(not(target_arch = "spirv"))]
#[inline]
#[must_use]
pub fn reconstruct_ozaki_dot_f64(sums: &[i32; 6], scale_x: f32, scale_y: f32) -> f64 {
    const fn pow2_neg_f64(shift: u32) -> f64 {
        f64::from_bits((1023_u64 - (shift as u64)) << 52)
    }
    const W0: f64 = pow2_neg_f64(14);
    const W1: f64 = pow2_neg_f64(22);
    const W2: f64 = pow2_neg_f64(30);
    const W3: f64 = pow2_neg_f64(38);
    const W4: f64 = pow2_neg_f64(46);

    let dot_norm = (sums[4] as f64) * W4
        + (sums[3] as f64) * W3
        + (sums[2] as f64) * W2
        + (sums[1] as f64) * W1
        + (sums[0] as f64) * W0;
    dot_norm * f64::from(scale_x) * f64::from(scale_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn slice_reconstruct_f64(s: [i32; 3], scale: f32) -> f64 {
        let w0 = 2.0_f64.powi(-7);
        let w1 = 2.0_f64.powi(-15);
        let w2 = 2.0_f64.powi(-23);
        (f64::from(s[0]) * w0 + f64::from(s[1]) * w1 + f64::from(s[2]) * w2) * f64::from(scale)
    }

    /// verify[train.pipeline.webgpu-scans-and-ozaki]
    #[test]
    fn test_ozaki_pow2_ceil_rne_and_cancellation_immunity() {
        assert_eq!(pow2_ceil_fp32(0.0), (1.0, 1.0));
        assert_eq!(pow2_ceil_fp32(-0.0), (1.0, 1.0));
        assert_eq!(pow2_ceil_fp32(-5.0), (1.0, 1.0));
        assert_eq!(pow2_ceil_fp32(f32::INFINITY), (1.0, 1.0));
        assert_eq!(pow2_ceil_fp32(f32::NEG_INFINITY), (1.0, 1.0));
        assert_eq!(pow2_ceil_fp32(f32::NAN), (1.0, 1.0));

        assert_eq!(pow2_ceil_fp32(1.0), (1.0, 1.0));
        assert_eq!(pow2_ceil_fp32(1.0 + f32::EPSILON), (2.0, 0.5));
        let (sub_s, sub_inv) = pow2_ceil_fp32(f32::from_bits(1));
        assert_eq!(sub_s, 2.0_f32.powi(-126));
        assert_eq!(sub_inv, 2.0_f32.powi(126));
        assert_eq!(sub_s * sub_inv, 1.0);

        let (max_s, max_inv) = pow2_ceil_fp32(f32::MAX);
        assert_eq!(max_s, 2.0_f32.powi(126));
        assert_eq!(max_inv, 2.0_f32.powi(-126));
        assert_eq!(max_s * max_inv, 1.0);

        // RNE rounding boundary verification: ties-to-even over [-2.5, 2.5]
        assert_eq!(to_i32_rne_f32(0.5), 0);
        assert_eq!(to_i32_rne_f32(-0.5), 0);
        assert_eq!(to_i32_rne_f32(1.5), 2);
        assert_eq!(to_i32_rne_f32(-1.5), -2);
        assert_eq!(to_i32_rne_f32(2.5), 2);
        assert_eq!(to_i32_rne_f32(-2.5), -2);
        assert_eq!(to_i32_rne_f32(128.0), 128);
        assert_eq!(to_i32_rne_f32(-128.0), -128);

        assert_eq!(slice_f32_mantissa_i8x3(f32::NAN, 1.0), [0, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(1.0, f32::INFINITY), [0, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(1.0e20, 1.0e25), [0, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(1.0, 1.0), [128, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(-1.0, 1.0), [-128, 0, 0]);

        // Cancellation immunity: naive f32 sum collapses (1e5 + 1.0 - 1e5) -> 0.0, Ozaki -> 1.0 exact
        let x = [100_000.0_f32, 1.0, -100_000.0];
        let y = [1.0_f32, 1.0, 1.0];
        let (sx, inv_sx) = pow2_ceil_fp32(100_000.0);
        let (sy, inv_sy) = pow2_ceil_fp32(1.0);
        let sums = accum_ozaki_chunk_i32(&x, &y, inv_sx, inv_sy);
        assert_eq!(sums[5], 0);
        assert_eq!(reconstruct_ozaki_dot_f64(&sums, sx, sy), 1.0);

        // Chunks longer than OZAKI_MAX_CHUNK_LEN safely truncate at 1024 without i32 overflow
        let big = [1.0_f32; 1200];
        let big_sums = accum_ozaki_chunk_i32(&big, &big, 1.0, 1.0);
        assert_eq!(big_sums[0], 1024 * 128 * 128);
        assert_eq!(reconstruct_ozaki_dot_f64(&big_sums, 1.0, 1.0), 1024.0);
    }

    proptest! {
        #[test]
        fn prop_pow2_ceil_and_slice_i8x3_bounds_and_accuracy(
            max_abs in 1.0e-30_f32..1.0e30_f32,
            frac_x in -1.0_f32..=1.0_f32,
            frac_y in -1.0_f32..=1.0_f32,
        ) {
            let (scale, inv_scale) = pow2_ceil_fp32(max_abs);
            prop_assert!(scale >= max_abs && scale < 2.0 * max_abs);
            prop_assert_eq!(scale * inv_scale, 1.0);

            let val = max_abs * frac_x;
            let s = slice_f32_mantissa_i8x3(val, inv_scale);
            for k in s {
                prop_assert!((-128..=128).contains(&k));
            }
            let recon = slice_reconstruct_f64(s, scale);
            let err = (recon - f64::from(val)).abs();
            // 3 signed slices at 2^-7, 2^-15, 2^-23 reconstruct any |val| <= scale within <= 0.5 * 2^-23 * scale.
            prop_assert!(err <= f64::from(scale) * 2.0_f64.powi(-24) + 1.0e-45);

            // Exact reconstruction for multiples of 2^-23 * scale
            let q_val = (f64::from(s[0]) * 2.0_f64.powi(-7)
                + f64::from(s[1]) * 2.0_f64.powi(-15)
                + f64::from(s[2]) * 2.0_f64.powi(-23))
                * f64::from(scale);
            let s_exact = slice_f32_mantissa_i8x3(q_val as f32, inv_scale);
            prop_assert_eq!(slice_reconstruct_f64(s_exact, scale), q_val);

            // Pair and chunk accumulation parity
            let val_y = max_abs * frac_y;
            let sums = accum_ozaki_chunk_i32(&[val], &[val_y], inv_scale, inv_scale);
            let dot = reconstruct_ozaki_dot_f64(&sums, scale, scale);
            let sy = slice_f32_mantissa_i8x3(val_y, inv_scale);
            let expected_dot = slice_reconstruct_f64(s, scale) * slice_reconstruct_f64(sy, scale);
            prop_assert!((dot - expected_dot).abs() <= 1.0e-12 * f64::from(scale) * f64::from(scale));
        }

        #[test]
        fn prop_ozaki_chunk_dot_matches_f64_oracle(
            xs in prop::collection::vec(-100.0_f32..=100.0_f32, 1..=128),
            ys in prop::collection::vec(-100.0_f32..=100.0_f32, 1..=128),
        ) {
            let n = xs.len().min(ys.len());
            let max_x = xs[..n].iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            let max_y = ys[..n].iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            let (sx, inv_sx) = pow2_ceil_fp32(max_x);
            let (sy, inv_sy) = pow2_ceil_fp32(max_y);
            let sums = accum_ozaki_chunk_i32(&xs[..n], &ys[..n], inv_sx, inv_sy);
            let dot_ozaki = reconstruct_ozaki_dot_f64(&sums, sx, sy);
            let dot_f64: f64 = xs[..n]
                .iter()
                .zip(ys[..n].iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let bound = (n as f64) * 2.1 * f64::from(sx) * f64::from(sy) * 2.0_f64.powi(-24) + 1.0e-12;
            prop_assert!((dot_ozaki - dot_f64).abs() <= bound);
        }
    }
}
