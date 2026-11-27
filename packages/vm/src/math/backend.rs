//! Hardware abstraction traits ([`SimdVec4`] and [`MathBackend`]) for 4-lane `f32` SIMD
//! registers and core division-free math primitives.

pub mod simd_vec4;

use dnam_kernels::{StereoVec8, canonicalize_zero_f32_bits};
pub use simd_vec4::SimdVec4;

use super::{BackendKind, IEEE754_F32_RSQRT_SEED, algebraic_unit_rotor, f32_to_fp16, fp16_to_f32};
use crate::math::crc32c::Crc32cBackend;

/// Flushes `f32` subnormals to signed zero without changing any other bit pattern.
#[inline(always)]
pub const fn flush_subnormal_f32_bits(value: f32) -> f32 {
    dnam_kernels::flush_subnormal_f32_bits(value)
}

/// Scrubs non-finite states (`NaN`/`±Inf`), subnormals, and `-0.0` to canonical `+0.0`.
#[inline(always)]
pub const fn flush_recursive_state_f32(value: f32) -> f32 {
    canonicalize_zero_f32_bits(value)
}

/// Scrubs non-finite lanes (`NaN`/`±Inf`), subnormals, and `-0.0` to `+0.0` across 4 lanes.
#[inline(always)]
pub fn flush_recursive_state_vec4<V: SimdVec4>(value: V) -> V {
    value.scrub_recursive_state()
}

/// Portable scalar 2-step Newton `rsqrt` (`IEEE754_F32_RSQRT_SEED - (v >> 1)`) for a positive `f32`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_f32(value: f32) -> f32 {
    let mut est = f32::from_bits(IEEE754_F32_RSQRT_SEED - (value.to_bits() >> 1));
    let half_v = 0.5 * value;
    for _ in 0..2 {
        let corr = 1.5 - half_v * (est * est);
        est *= corr;
    }
    est
}

/// Portable scalar 2-step Newton `rsqrt` (`IEEE754_F32_RSQRT_SEED - (v >> 1)`) for a stereo pair `(v0, v1)`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
    let mut e0 = f32::from_bits(IEEE754_F32_RSQRT_SEED - (v0.to_bits() >> 1));
    let mut e1 = f32::from_bits(IEEE754_F32_RSQRT_SEED - (v1.to_bits() >> 1));
    let hv0 = 0.5 * v0;
    let hv1 = 0.5 * v1;
    for _ in 0..2 {
        let c0 = 1.5 - hv0 * (e0 * e0);
        let c1 = 1.5 - hv1 * (e1 * e1);
        e0 *= c0;
        e1 *= c1;
    }
    (e0, e1)
}

/// Level 1 Hardware Abstraction: Core `f32` SIMD math primitives across `aarch64`, `x86_64`,
/// `wasm32`, and scalar fallback.
pub trait MathBackend: Copy + Clone + Send + Sync + 'static + Crc32cBackend {
    /// Underlying 128-bit 4-lane vector register type.
    type Vec4: SimdVec4 + dnam_kernels::KernelVec4;
    /// Underlying 8-lane stereo register type (`lanes 0..3 = Left`, `4..7 = Right`).
    type Stereo8: StereoVec8<Half = Self::Vec4>;
    /// Human-readable backend name for telemetry and benchmarks.
    const NAME: &'static str;
    /// Enum identifier for this backend.
    const KIND: BackendKind;

    /// 4x4 matrix-vector FMA tile `acc + sum_{c=0..3} x[c] * mat[c]`.
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: Self::Vec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> Self::Vec4 {
        let xv = Self::Vec4::load(x);
        let c0 = Self::Vec4::load(&mat[0]);
        let c1 = Self::Vec4::load(&mat[1]);
        let c2 = Self::Vec4::load(&mat[2]);
        let c3 = Self::Vec4::load(&mat[3]);
        let p01 = c0.mul_lane::<0>(xv).fma_lane::<1>(c1, xv);
        let p23 = c2.mul_lane::<2>(xv).fma_lane::<3>(c3, xv);
        acc.add(p01.add(p23))
    }

    /// Stereo 4x4 matrix-vector FMA tile using 4 pre-loaded column vectors.
    #[inline(always)]
    fn fma_cols4_stereo(
        acc_l: Self::Vec4,
        acc_r: Self::Vec4,
        cols: [Self::Vec4; 4],
        x_l: Self::Vec4,
        x_r: Self::Vec4,
    ) -> (Self::Vec4, Self::Vec4) {
        let [c0, c1, c2, c3] = cols;
        let p01_l = c0.mul_lane::<0>(x_l).fma_lane::<1>(c1, x_l);
        let p23_l = c2.mul_lane::<2>(x_l).fma_lane::<3>(c3, x_l);
        let p01_r = c0.mul_lane::<0>(x_r).fma_lane::<1>(c1, x_r);
        let p23_r = c2.mul_lane::<2>(x_r).fma_lane::<3>(c3, x_r);
        (acc_l.add(p01_l.add(p23_l)), acc_r.add(p01_r.add(p23_r)))
    }

    /// Stereo 4x4 matrix-vector FMA tile updating `(acc_l, acc_r)` with shared column loads.
    #[inline(always)]
    fn fma_mat4x4_stereo(
        acc_l: Self::Vec4,
        acc_r: Self::Vec4,
        cols: &[[f32; 4]; 4],
        x_l: Self::Vec4,
        x_r: Self::Vec4,
    ) -> (Self::Vec4, Self::Vec4) {
        let c0 = Self::Vec4::load(&cols[0]);
        let c1 = Self::Vec4::load(&cols[1]);
        let c2 = Self::Vec4::load(&cols[2]);
        let c3 = Self::Vec4::load(&cols[3]);
        Self::fma_cols4_stereo(acc_l, acc_r, [c0, c1, c2, c3], x_l, x_r)
    }

    /// Transposes a 4×4 matrix of `Vec4` rows (`[r0, r1, r2, r3]`) into 4 `Vec4` columns.
    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        let a0 = r0.to_array();
        let a1 = r1.to_array();
        let a2 = r2.to_array();
        let a3 = r3.to_array();
        [
            Self::Vec4::from_array([a0[0], a1[0], a2[0], a3[0]]),
            Self::Vec4::from_array([a0[1], a1[1], a2[1], a3[1]]),
            Self::Vec4::from_array([a0[2], a1[2], a2[2], a3[2]]),
            Self::Vec4::from_array([a0[3], a1[3], a2[3], a3[3]]),
        ]
    }

    /// Converts a `Vec4` register directly to 4 IEEE-754 `fp16` (`u16`) values.
    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        Self::f32x4_to_fp16x4(&src.to_array())
    }

    /// Converts 4 IEEE-754 `fp16` (`u16`) values directly into a `Vec4` register.
    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        Self::Vec4::load(&Self::fp16x4_to_f32x4(src))
    }

    /// Converts 4 IEEE-754 `f32` samples to `fp16` (`u16`).
    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        [
            f32_to_fp16(src[0]),
            f32_to_fp16(src[1]),
            f32_to_fp16(src[2]),
            f32_to_fp16(src[3]),
        ]
    }

    /// Converts 4 IEEE-754 `fp16` (`u16`) samples to `f32`.
    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        [
            fp16_to_f32(src[0]),
            fp16_to_f32(src[1]),
            fp16_to_f32(src[2]),
            fp16_to_f32(src[3]),
        ]
    }

    /// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a single positive `f32`.
    #[inline(always)]
    fn rsqrt_nr2_f32(v: f32) -> f32 {
        rsqrt_nr2_scalar_f32(v)
    }

    /// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)`.
    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        rsqrt_nr2_scalar_pair_f32(v0, v1)
    }

    /// Evaluates the division-free algebraic unit-circle phasor `(cos(theta), sin(theta))`.
    #[inline(always)]
    fn fast_unit_phasor(theta_rad: f32) -> (f32, f32) {
        algebraic_unit_rotor(theta_rad)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::scalar::ScalarBackend;

    fn check_flush_subnormal<V: SimdVec4>() {
        let dirty = V::from_array([
            f32::from_bits(1),
            f32::from_bits(0x807f_ffff),
            f32::NAN,
            -0.5,
        ]);
        let flushed = flush_recursive_state_vec4(dirty).to_array();
        assert_eq!(flushed[0].to_bits(), 0);
        assert_eq!(flushed[1].to_bits(), 0);
        assert_eq!(flushed[2].to_bits(), 0);
        assert_eq!(flushed[3].to_bits(), (-0.5_f32).to_bits());

        let neg_zero = V::from_array([-0.0, f32::INFINITY, f32::NEG_INFINITY, 1.5])
            .scrub_recursive_state()
            .to_array();
        assert_eq!(neg_zero[0].to_bits(), 0);
        assert_eq!(neg_zero[1].to_bits(), 0);
        assert_eq!(neg_zero[2].to_bits(), 0);
        assert_eq!(neg_zero[3].to_bits(), 1.5_f32.to_bits());
    }

    pub(crate) fn verify_math_backend_ops<B: MathBackend>() {
        use dnam_kernels::StereoVec8;
        assert!(!B::NAME.is_empty());
        assert!(!B::KIND.name().is_empty());
        assert_eq!(B::crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(B::crc32c_update(0, b"123456789"), 0xE306_9283);
        check_flush_subnormal::<B::Vec4>();
        crate::math::backend::simd_vec4::tests::verify_all_simd_and_kernel_ops::<B::Vec4>();
        let cols = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ];
        let x = [1.0, 2.0, 3.0, 4.0];
        let v = B::fma_mat4x4_vec4(B::Vec4::zero(), &cols, &x);
        assert_eq!(v.to_array(), [1.0, 4.0, 9.0, 16.0]);
        let (sl, sr) = B::fma_mat4x4_stereo(B::Vec4::zero(), B::Vec4::zero(), &cols, v, v);
        assert_eq!(sl.to_array(), [1.0, 8.0, 27.0, 64.0]);
        assert_eq!(sr.to_array(), [1.0, 8.0, 27.0, 64.0]);
        let t = B::transpose_4x4(
            B::Vec4::from_array([1.0, 2.0, 3.0, 4.0]),
            B::Vec4::from_array([5.0, 6.0, 7.0, 8.0]),
            B::Vec4::from_array([9.0, 10.0, 11.0, 12.0]),
            B::Vec4::from_array([13.0, 14.0, 15.0, 16.0]),
        );
        assert_eq!(t[0].to_array(), [1.0, 5.0, 9.0, 13.0]);
        let fp16 = B::vec4_to_fp16x4(B::Vec4::from_array(x));
        assert_eq!(B::fp16x4_to_vec4(&fp16).to_array(), x);
        let fp16_arr = B::f32x4_to_fp16x4(&x);
        assert_eq!(B::fp16x4_to_f32x4(&fp16_arr), x);

        let over = [65_504.0_f32, -65_504.0, 65_520.0, -100_000.0];
        assert_eq!(
            B::vec4_to_fp16x4(B::Vec4::from_array(over)),
            [0x7BFF, 0xFBFF, 0x7BFF, 0xFBFF]
        );
        assert_eq!(B::f32x4_to_fp16x4(&over), [0x7BFF, 0xFBFF, 0x7BFF, 0xFBFF]);
        let special = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7FC0_0000),
            f32::from_bits(0x7F80_0001),
        ];
        assert_eq!(
            B::vec4_to_fp16x4(B::Vec4::from_array(special)),
            [0x7C00, 0xFC00, 0x7E00, 0x7C01]
        );
        {
            let _ftz = crate::types::FtzDazGuard::new();
            let sub = [
                0.0_f32,
                -0.0_f32,
                f32::from_bits(0x3380_0000),
                -f32::from_bits(0x387F_C000),
            ];
            let sub_h = B::vec4_to_fp16x4(B::Vec4::from_array(sub));
            assert_eq!(sub_h, ScalarBackend::f32x4_to_fp16x4(&sub));
            assert_eq!(
                B::fp16x4_to_vec4(&sub_h).to_array().map(f32::to_bits),
                sub.map(f32::to_bits)
            );
        }
        assert!((B::rsqrt_nr2_f32(4.0) - 0.5).abs() < 1.0e-5);
        let (ra, rb) = B::rsqrt_nr2_pair_f32(4.0, 9.0);
        assert!((ra - 0.5).abs() < 1.0e-5 && (rb - (1.0 / 3.0)).abs() < 1.0e-5);
        let (c, s) = B::fast_unit_phasor(0.5);
        assert!((c * c + s * s - 1.0).abs() < 1.0e-6);

        let st = B::Stereo8::from_lr(
            dnam_kernels::KernelVec4::from_array(x),
            dnam_kernels::KernelVec4::from_array(x),
        );
        let (hl, hr) = st.into_lr();
        assert_eq!(
            dnam_kernels::KernelVec4::to_array(hl),
            dnam_kernels::KernelVec4::to_array(hr)
        );
    }

    #[test]
    fn test_backend_default_mat4x4_transpose_and_fp16() {
        verify_math_backend_ops::<ScalarBackend>();
    }

    proptest! {
        #[test]
        fn prop_rsqrt_pair_scalar_matches_single(a in 0.1_f32..100.0, b in 0.1_f32..100.0) {
            let (ra, rb) = rsqrt_nr2_scalar_pair_f32(a, b);
            prop_assert_eq!(ra.to_bits(), rsqrt_nr2_scalar_f32(a).to_bits());
            prop_assert_eq!(rb.to_bits(), rsqrt_nr2_scalar_f32(b).to_bits());
        }
    }
}
