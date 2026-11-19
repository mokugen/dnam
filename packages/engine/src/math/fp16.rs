//! Branchless IEEE-754 `f32` $\leftrightarrow$ `fp16` (`u16`) software conversion primitives,
//! delegating to the canonical [`dnam_kernels::simd::fp16`] codecs.

pub use dnam_kernels::simd::fp16::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_roundtrip_and_special_values() {
        for &val in &[0.0f32, -0.0, 1.0, -1.0, 0.5, -0.25, 0.985, -0.985] {
            let half = f32_to_fp16(val);
            let back = fp16_to_f32(half);
            assert!((back - val).abs() < 1.0e-3);
        }

        let inf_h = f32_to_fp16(f32::INFINITY);
        assert!(fp16_to_f32(inf_h).is_infinite());

        let nan_h = f32_to_fp16(f32::NAN);
        assert!(fp16_to_f32(nan_h).is_nan());

        let ovf_h = f32_to_fp16(100_000.0);
        assert_eq!(ovf_h, 0x7BFF);

        // Rounding near max finite fp16 ([65520.0, 65536.0)) must clamp to 0x7BFF/0xFBFF, never ±Inf
        for &near_max in &[65504.0f32, 65519.0, 65520.0, 65535.0] {
            let hp = f32_to_fp16(near_max);
            let hn = f32_to_fp16(-near_max);
            assert_eq!(hp, 0x7BFF);
            assert_eq!(hn, 0xFBFF);
            assert!(fp16_to_f32(hp).is_finite());
            assert!(fp16_to_f32(hn).is_finite());
        }

        let und_h = f32_to_fp16(1.0e-10);
        assert_eq!(fp16_to_f32(und_h), 0.0);

        let sub_h = f32_to_fp16(3.0e-5);
        assert!(fp16_to_f32(sub_h) > 0.0);
    }
}
