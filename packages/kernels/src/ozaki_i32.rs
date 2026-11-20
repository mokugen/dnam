//! Fast-math-immune `(2^8, 2^16, 2^24)` `i32` Ozaki mantissa slicing and exact chunked
//! dot-product accumulation (`ref[train.pipeline.webgpu-scans-and-ozaki]`,
//! `ref[sdk.rust.shared-nostd-kernels]`).
//!
//! Decomposes each normalized `f32` operand into 3 signed slices (`|s0| <= 256`, `|s1| <= 128`,
//! `|s2| <= 128` at `2^-8, 2^-16, 2^-24`) using branchless `1.5 * 2^23` IEEE-754 RNE rounding
//! (zero `libm`, zero `FDIV`, zero `u64`/`f64` on `spirv`) and accumulates up to
//! [`OZAKI_MAX_CHUNK_LEN`] (`1024`) products strictly in `i32` registers before host `f64` diagonal
//! reconstruction.

/// Maximum number of elements per `i32` Ozaki chunk (`1024`; worst-case `d2` sum
/// `1024 * (256 * 128 + 128 * 128 + 128 * 256) = 83_886_080 <= i32::MAX`).
pub const OZAKI_MAX_CHUNK_LEN: usize = 1024;

/// IEEE-754 `1.5 * 2^23` (`12_582_912.0`) RNE integer rounding bias (`0x4B40_0000`).
const RNE_MAGIC_F32: f32 = 1.5 * ((1_u32 << 23) as f32);
const RNE_MAGIC_BITS: i32 = RNE_MAGIC_F32.to_bits() as i32;

/// Maximum magnitude representable by the `1.5 * 2^23` biased-rounding window (`2^22`).
const RNE_MAX_ABS_F32: f32 = (1_u32 << 22) as f32;

/// Maximum normal Ozaki scaling ceiling (`2^126`), keeping both `scale` and `inv_scale` normal
/// `binary32` powers of two and preventing `scale^2` overflow in `f64` reconstruction.
const MAX_OZAKI_VAL_F32: f32 = f32::from_bits(253 << 23);

/// Slice multipliers and exact negative power-of-2 reconstruction weights (`2^8`, `2^16`, `2^24`).
const SLICE0_SCALE_F32: f32 = (1_u32 << 8) as f32;
const SLICE0_INV_F32: f32 = f32::from_bits((127 - 8) << 23);
const SLICE1_SCALE_F32: f32 = (1_u32 << 16) as f32;
const SLICE1_INV_F32: f32 = f32::from_bits((127 - 16) << 23);
const SLICE2_SCALE_F32: f32 = (1_u32 << 24) as f32;

const OZAKI_MAX_BUCKET_SUM_U32: u32 =
    (OZAKI_MAX_CHUNK_LEN as u32) * (256 * 128 + 128 * 128 + 128 * 256);

const _: () = assert!((OZAKI_MAX_CHUNK_LEN as u32) == 1024);
const _: () = assert!(OZAKI_MAX_BUCKET_SUM_U32 == 83_886_080);
const _: () = assert!(OZAKI_MAX_BUCKET_SUM_U32 <= i32::MAX as u32);
const _: () = assert!(3_u32 * (OZAKI_MAX_CHUNK_LEN as u32) * 256 * 128 <= i32::MAX as u32);
#[cfg(not(target_arch = "spirv"))]
const _: () = assert!(3_i64 * (OZAKI_MAX_CHUNK_LEN as i64) * 256 * 128 <= i32::MAX as i64);
const _: () = assert!(RNE_MAGIC_BITS == 0x4B40_0000);
const _: () = assert!(RNE_MAX_ABS_F32 == 4_194_304.0);
const _: () = assert!(MAX_OZAKI_VAL_F32.to_bits() == (253 << 23));
const _: () = assert!(SLICE0_SCALE_F32 * SLICE0_INV_F32 == 1.0);
const _: () = assert!(SLICE1_SCALE_F32 * SLICE1_INV_F32 == 1.0);
const _: () = assert!(SLICE2_SCALE_F32 == (1_u32 << 24) as f32);

/// Division-free branchless IEEE-754 round-to-nearest-ties-to-even (`f32 -> i32`) guarded against
/// non-finite values and clamped to `[-2^22, +2^22]` (`RNE_MAGIC_F32` exactness window).
#[inline(always)]
#[must_use]
pub fn round_ties_even_f32(x: f32) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    let clamped = x.clamp(-RNE_MAX_ABS_F32, RNE_MAX_ABS_F32);
    (clamped + RNE_MAGIC_F32).to_bits() as i32 - RNE_MAGIC_BITS
}

/// Computes the smallest power-of-two ceiling `scale >= min(max_abs, 2^126)` and its exact
/// reciprocal `inv_scale = 1.0 / scale` in pure `binary32` bit arithmetic (`(scale, inv_scale)`).
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
    let clamped = max_abs.min(MAX_OZAKI_VAL_F32);
    let ceil_bits = clamped.to_bits().wrapping_add(0x007F_FFFF) & 0x7F80_0000;
    let e = (ceil_bits >> 23).clamp(1, 253);
    let scale = f32::from_bits(e << 23);
    let inv_scale = f32::from_bits((254 - e) << 23);
    (scale, inv_scale)
}

/// Decomposes `clamp(val, -2^126, 2^126) * inv_scale` (normalized to `[-1.0, 1.0]`) into 3 signed
/// slices `[s0, s1, s2]` (`|s0| <= 256`, `|s1| <= 128`, `|s2| <= 128`) representing
/// `s0 * 2^-8 + s1 * 2^-16 + s2 * 2^-24`.
///
/// Returns `[0, 0, 0]` if `val` or `inv_scale` is non-finite or produces a non-finite product.
#[inline(always)]
#[must_use]
pub fn slice_f32_mantissa_i8x3(val: f32, inv_scale: f32) -> [i32; 3] {
    if !val.is_finite() || !inv_scale.is_finite() {
        return [0, 0, 0];
    }
    let val_clamped = val.clamp(-MAX_OZAKI_VAL_F32, MAX_OZAKI_VAL_F32);
    let v_raw = val_clamped * inv_scale;
    if !v_raw.is_finite() {
        return [0, 0, 0];
    }
    let v = v_raw.clamp(-1.0, 1.0);
    let s0 = round_ties_even_f32(v * SLICE0_SCALE_F32);
    let rem1 = v - (s0 as f32) * SLICE0_INV_F32;
    let s1 = round_ties_even_f32(rem1 * SLICE1_SCALE_F32);
    let rem2 = rem1 - (s1 as f32) * SLICE1_INV_F32;
    let s2 = round_ties_even_f32(rem2 * SLICE2_SCALE_F32);
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

/// Reconstructs an `f64` inner product from accumulated 6-word Ozaki diagonal sums on the host
/// using `(2^-16, 2^-24, 2^-32, 2^-40, 2^-48)` weights.
#[cfg(not(target_arch = "spirv"))]
#[inline]
#[must_use]
pub fn reconstruct_ozaki_dot_f64(sums: &[i32; 6], scale_x: f32, scale_y: f32) -> f64 {
    const fn pow2_neg_f64(shift: u32) -> f64 {
        f64::from_bits((1023_u64 - (shift as u64)) << 52)
    }
    if !scale_x.is_finite() || !scale_y.is_finite() {
        return 0.0;
    }
    const W0: f64 = pow2_neg_f64(16);
    const W1: f64 = pow2_neg_f64(24);
    const W2: f64 = pow2_neg_f64(32);
    const W3: f64 = pow2_neg_f64(40);
    const W4: f64 = pow2_neg_f64(48);

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
        let w0 = 2.0_f64.powi(-8);
        let w1 = 2.0_f64.powi(-16);
        let w2 = 2.0_f64.powi(-24);
        (f64::from(s[0]) * w0 + f64::from(s[1]) * w1 + f64::from(s[2]) * w2) * f64::from(scale)
    }

    fn chunk_max_abs_finite(vals: &[f32]) -> f32 {
        vals.iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0_f32, |m, v| m.max(v.abs()))
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

        // RNE rounding boundary & range-guard verification
        assert_eq!(round_ties_even_f32(f32::NAN), 0);
        assert_eq!(round_ties_even_f32(f32::INFINITY), 0);
        assert_eq!(round_ties_even_f32(f32::NEG_INFINITY), 0);
        assert_eq!(round_ties_even_f32(0.5), 0);
        assert_eq!(round_ties_even_f32(-0.5), 0);
        assert_eq!(round_ties_even_f32(1.5), 2);
        assert_eq!(round_ties_even_f32(-1.5), -2);
        assert_eq!(round_ties_even_f32(2.5), 2);
        assert_eq!(round_ties_even_f32(-2.5), -2);
        assert_eq!(round_ties_even_f32(256.0), 256);
        assert_eq!(round_ties_even_f32(-256.0), -256);
        assert_eq!(round_ties_even_f32(1.0e10), 4_194_304);
        assert_eq!(round_ties_even_f32(-1.0e10), -4_194_304);

        assert_eq!(slice_f32_mantissa_i8x3(f32::NAN, 1.0), [0, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(1.0, f32::INFINITY), [0, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(1.0e20, 1.0e25), [0, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(1.0, 1.0), [256, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(-1.0, 1.0), [-256, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(f32::MAX, max_inv), [256, 0, 0]);
        assert_eq!(slice_f32_mantissa_i8x3(-f32::MAX, max_inv), [-256, 0, 0]);
        assert_eq!(
            slice_f32_mantissa_i8x3(1.0 - 2.0_f32.powi(-24), 1.0),
            [256, 0, -1]
        );
        assert_eq!(
            slice_f32_mantissa_i8x3(0.5 + 2.0_f32.powi(-24), 1.0),
            [128, 0, 1]
        );
        assert_eq!(reconstruct_ozaki_dot_f64(&[0; 6], f32::NAN, 1.0), 0.0);

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
        assert_eq!(big_sums[0], 1024 * 256 * 256);
        assert_eq!(reconstruct_ozaki_dot_f64(&big_sums, 1.0, 1.0), 1024.0);

        // Non-finite inputs mixed into wide dynamic ranges (2^-60 and 2^40..2^24) scrub cleanly
        let wide_x = [
            2.0_f32.powi(40),
            f32::NAN,
            2.0_f32.powi(24),
            -2.0_f32.powi(40),
        ];
        let wide_y = [1.0_f32, 2.0_f32.powi(24), 2.0_f32.powi(24), 1.0_f32];
        let (wsx, winv_x) = pow2_ceil_fp32(chunk_max_abs_finite(&wide_x));
        let (wsy, winv_y) = pow2_ceil_fp32(chunk_max_abs_finite(&wide_y));
        let wide_sums = accum_ozaki_chunk_i32(&wide_x, &wide_y, winv_x, winv_y);
        assert_eq!(
            reconstruct_ozaki_dot_f64(&wide_sums, wsx, wsy),
            2.0_f64.powi(48)
        );

        let tiny_x = [2.0_f32.powi(-60), f32::NEG_INFINITY, 2.0_f32.powi(-60)];
        let tiny_y = [1.0_f32, 1.0_f32, -1.0_f32];
        let (tsx, tinv_x) = pow2_ceil_fp32(chunk_max_abs_finite(&tiny_x));
        let (tsy, tinv_y) = pow2_ceil_fp32(chunk_max_abs_finite(&tiny_y));
        let tiny_sums = accum_ozaki_chunk_i32(&tiny_x, &tiny_y, tinv_x, tinv_y);
        assert_eq!(reconstruct_ozaki_dot_f64(&tiny_sums, tsx, tsy), 0.0);
    }

    proptest! {
        #[test]
        fn prop_pow2_ceil_and_slice_i8x3_bounds_and_0ulp_top_binade(
            exp in -60_i32..=60_i32,
            mant in 1.0_f32..2.0_f32,
            frac_x in -1.0_f32..=1.0_f32,
            frac_y in -1.0_f32..=1.0_f32,
        ) {
            let max_abs = mant * 2.0_f32.powi(exp);
            let (scale, inv_scale) = pow2_ceil_fp32(max_abs);
            prop_assert!(scale >= max_abs && scale < 2.0 * max_abs);
            prop_assert_eq!(scale * inv_scale, 1.0);

            let val = max_abs * frac_x;
            let s = slice_f32_mantissa_i8x3(val, inv_scale);
            prop_assert!((-256..=256).contains(&s[0]));
            prop_assert!((-128..=128).contains(&s[1]));
            prop_assert!((-128..=128).contains(&s[2]));

            let recon = slice_reconstruct_f64(s, scale);
            let err = (recon - f64::from(val)).abs();
            // 3 signed slices at 2^-8, 2^-16, 2^-24 match within <= 0.5 * 2^-24 * scale
            // and achieve exact 0-ULP reconstruction whenever |val| lies in [0.5 * scale, scale].
            prop_assert!(err <= f64::from(scale) * 2.0_f64.powi(-25) + 1.0e-45);
            if val.abs() >= 0.5 * scale {
                prop_assert_eq!(recon, f64::from(val));
            }

            let val_y = max_abs * frac_y;
            let sums = accum_ozaki_chunk_i32(&[val], &[val_y], inv_scale, inv_scale);
            let dot = reconstruct_ozaki_dot_f64(&sums, scale, scale);
            let sy = slice_f32_mantissa_i8x3(val_y, inv_scale);
            let expected_dot = slice_reconstruct_f64(s, scale) * slice_reconstruct_f64(sy, scale);
            prop_assert!(
                (dot - expected_dot).abs() <= 1.0e-12 * f64::from(scale) * f64::from(scale)
            );
        }

        #[test]
        fn prop_ozaki_chunk_order_0ulp_parity_and_wide_dynamic_range(
            exps_x in prop::collection::vec(-60_i32..=60_i32, 1..=64),
            mants_x in prop::collection::vec(-1.99_f32..=1.99_f32, 1..=64),
            exps_y in prop::collection::vec(-60_i32..=60_i32, 1..=64),
            mants_y in prop::collection::vec(-1.99_f32..=1.99_f32, 1..=64),
            rot in 0_usize..64_usize,
        ) {
            let n = exps_x.len().min(mants_x.len()).min(exps_y.len()).min(mants_y.len());
            let xs: Vec<f32> = (0..n).map(|i| mants_x[i] * 2.0_f32.powi(exps_x[i])).collect();
            let ys: Vec<f32> = (0..n).map(|i| mants_y[i] * 2.0_f32.powi(exps_y[i])).collect();
            let (sx, inv_sx) = pow2_ceil_fp32(chunk_max_abs_finite(&xs));
            let (sy, inv_sy) = pow2_ceil_fp32(chunk_max_abs_finite(&ys));

            let sums_fwd = accum_ozaki_chunk_i32(&xs, &ys, inv_sx, inv_sy);
            let mut xs_rev = xs.clone();
            let mut ys_rev = ys.clone();
            xs_rev.reverse();
            ys_rev.reverse();
            let sums_rev = accum_ozaki_chunk_i32(&xs_rev, &ys_rev, inv_sx, inv_sy);

            let shift = rot % n;
            let mut xs_rot = xs.clone();
            let mut ys_rot = ys.clone();
            xs_rot.rotate_left(shift);
            ys_rot.rotate_left(shift);
            let sums_rot = accum_ozaki_chunk_i32(&xs_rot, &ys_rot, inv_sx, inv_sy);

            // Exact 0-ULP integer bucket parity across arbitrary chunk orders.
            prop_assert_eq!(sums_fwd, sums_rev);
            prop_assert_eq!(sums_fwd, sums_rot);

            let dot_ozaki = reconstruct_ozaki_dot_f64(&sums_fwd, sx, sy);
            let dot_f64: f64 = xs
                .iter()
                .zip(ys.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let bound =
                (n as f64) * 2.1 * f64::from(sx) * f64::from(sy) * 2.0_f64.powi(-25) + 1.0e-30;
            prop_assert!((dot_ozaki - dot_f64).abs() <= bound);
        }
    }
}
