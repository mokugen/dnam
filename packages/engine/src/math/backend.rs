//! Hardware abstraction traits (`SimdVec4` and `MathBackend`) for 4-lane SIMD registers
//! and core division-free math primitives.
//!
//! Higher-level domain modules define their own `<Domain>Backend` traits in their respective
//! modules (`OpMBackend`, `OpNBackend`, `OpTBackend`, `StageBackend`, `EngineBackend`),
//! keeping the dependency DAG strictly one-way (`engine -> stages -> operators -> math`).

pub mod simd_vec4;

use dnam_kernels::{StereoVec8, scrub_and_flush_f32_bits};
pub use simd_vec4::SimdVec4;

use super::{BackendKind, algebraic_unit_rotor, f32_to_fp16, fp16_to_f32};
use crate::math::crc32c::Crc32cBackend;

/// Flushes `f32` subnormals to signed zero without changing any other bit pattern.
#[inline(always)]
pub const fn flush_subnormal_f32_bits(value: f32) -> f32 {
    dnam_kernels::flush_subnormal_f32_bits(value)
}

/// Scrubs non-finite states (`NaN`/`±Inf`) to `+0.0` and flushes subnormals to signed zero.
#[inline(always)]
pub(crate) const fn flush_recursive_state_f32(value: f32) -> f32 {
    scrub_and_flush_f32_bits(value)
}

/// Scrubs non-finite lanes (`NaN`/`±Inf`) to `+0.0` and flushes subnormals to signed zero across 4 lanes.
#[inline(always)]
pub(crate) fn flush_recursive_state_vec4<V: SimdVec4>(value: V) -> V {
    value.scrub_recursive_state()
}

/// Portable scalar 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a single positive `f32`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_f32(value: f32) -> f32 {
    let mut est = f32::from_bits(0x5f37_59df - (value.to_bits() >> 1));
    let half_v = 0.5 * value;
    for _ in 0..2 {
        let corr = 1.5 - (half_v * est) * est;
        est *= corr;
    }
    est
}

/// Portable scalar 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
    let mut e0 = f32::from_bits(0x5f37_59df - (v0.to_bits() >> 1));
    let mut e1 = f32::from_bits(0x5f37_59df - (v1.to_bits() >> 1));
    let hv0 = 0.5 * v0;
    let hv1 = 0.5 * v1;
    for _ in 0..2 {
        let c0 = 1.5 - (hv0 * e0) * e0;
        let c1 = 1.5 - (hv1 * e1) * e1;
        e0 *= c0;
        e1 *= c1;
    }
    (e0, e1)
}

/// Level 2 Hardware Abstraction: Core SIMD math primitives across every architecture
/// (`aarch64`, `x86_64`, `wasm32`, `scalar`).
pub trait MathBackend: Copy + Clone + Send + Sync + 'static + Crc32cBackend {
    /// Underlying 128-bit 4-lane vector register type (also bridged to [`dnam_kernels::KernelVec4`]).
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

    /// Stereo 4x4 matrix-vector FMA tile using 4 pre-loaded column vectors in registers.
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

    /// Converts a `Vec4` register directly to 4 IEEE-754 half-precision `fp16` (`u16`) values.
    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        Self::f32x4_to_fp16x4(&src.to_array())
    }

    /// Converts 4 IEEE-754 half-precision `fp16` (`u16`) values directly into a `Vec4` register.
    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        Self::Vec4::load(&Self::fp16x4_to_f32x4(src))
    }

    /// Converts 4 IEEE-754 single-precision `f32` samples to `fp16` (`u16`).
    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        [
            f32_to_fp16(src[0]),
            f32_to_fp16(src[1]),
            f32_to_fp16(src[2]),
            f32_to_fp16(src[3]),
        ]
    }

    /// Converts 4 IEEE-754 half-precision `fp16` (`u16`) samples to `f32`.
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

    /// Evaluates the division-free algebraic unit-circle phasor `(cos(theta), sin(theta))` (`ref[dsp.op-m.rotor.contractive]`).
    #[inline(always)]
    fn fast_unit_phasor(theta_rad: f32) -> (f32, f32) {
        algebraic_unit_rotor(theta_rad)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::scalar::{ScalarBackend, ScalarVec4};

    fn check_flush_subnormal<V: SimdVec4>() {
        let cases = [
            (
                [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001],
                [0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000],
            ),
            (
                [0x007f_ffff, 0x807f_ffff, 0x0080_0000, 0x8080_0000],
                [0x0000_0000, 0x8000_0000, 0x0080_0000, 0x8080_0000],
            ),
            (
                [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
                [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
            ),
        ];
        for (input, expected) in cases {
            let values = input.map(f32::from_bits);
            assert_eq!(
                values.map(flush_subnormal_f32_bits).map(f32::to_bits),
                expected
            );
            assert_eq!(
                V::from_array(values)
                    .flush_subnormal()
                    .to_array()
                    .map(f32::to_bits),
                expected
            );
        }
    }

    fn check_flush_recursive_all_le_and_fp16<B: MathBackend>() {
        use dnam_kernels::KernelVec4;
        let dirty = [f32::NAN, f32::INFINITY, f32::from_bits(1), -0.75];
        let flushed = <B::Vec4 as KernelVec4>::flush_recursive_state(
            <B::Vec4 as KernelVec4>::from_array(dirty),
        );
        assert_eq!(
            KernelVec4::to_array(flushed).map(f32::to_bits),
            dirty
                .map(dnam_kernels::scrub_and_flush_f32_bits)
                .map(f32::to_bits)
        );
        let a = <B::Vec4 as KernelVec4>::from_array([1.0, 2.0, -3.0, 4.0]);
        let b = <B::Vec4 as KernelVec4>::from_array([1.0, 2.5, -3.0, 4.0]);
        assert!(KernelVec4::all_le(a, b));
        assert!(!KernelVec4::all_le(b, a));
        assert!(!KernelVec4::all_le(
            <B::Vec4 as KernelVec4>::from_array([1.0, f32::NAN, 0.0, 0.0]),
            b,
        ));

        let edge_fp16 = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 100_000.0];
        let hw = B::f32x4_to_fp16x4(&edge_fp16);
        let sw = ScalarBackend::f32x4_to_fp16x4(&edge_fp16);
        assert_eq!(hw[0], sw[0]);
        assert_eq!(hw[1], sw[1]);
        assert_eq!(hw[2], sw[2]);
        assert_eq!(hw[3], sw[3]);
    }

    #[test]
    fn test_flush_subnormal_all_le_and_fp16_preserves_bits() {
        check_flush_subnormal::<ScalarVec4>();
        check_flush_recursive_all_le_and_fp16::<ScalarBackend>();
        check_flush_recursive_all_le_and_fp16::<crate::math::DefaultMathBackend>();
        #[cfg(target_arch = "aarch64")]
        {
            check_flush_subnormal::<crate::math::NeonVec4>();
            check_flush_recursive_all_le_and_fp16::<crate::math::NeonBackend>();
            check_flush_recursive_all_le_and_fp16::<crate::math::ArmSve2Backend>();
            check_flush_recursive_all_le_and_fp16::<crate::math::ArmSmeBackend>();
        }
        #[cfg(target_arch = "x86_64")]
        {
            check_flush_subnormal::<crate::math::X86Vec4>();
            check_flush_recursive_all_le_and_fp16::<crate::math::X86Sse2Backend>();
            check_flush_recursive_all_le_and_fp16::<crate::math::X86Sse42Backend>();
            check_flush_recursive_all_le_and_fp16::<crate::math::X86Avx2FmaBackend>();
        }
        #[cfg(target_arch = "wasm32")]
        {
            check_flush_subnormal::<crate::math::WasmVec4>();
            check_flush_recursive_all_le_and_fp16::<crate::math::WasmSimd128Backend>();
        }
    }

    proptest! {
        #[test]
        fn prop_flush_subnormal_preserves_non_subnormal_bits(bits in any::<u32>()) {
            let expected = if bits & 0x7f80_0000 == 0 {
                bits & 0x8000_0000
            } else {
                bits
            };
            prop_assert_eq!(flush_subnormal_f32_bits(f32::from_bits(bits)).to_bits(), expected);
            let vector = ScalarVec4::splat(f32::from_bits(bits)).flush_subnormal();
            prop_assert_eq!(vector.to_array().map(f32::to_bits), [expected; 4]);
        }
    }

    #[test]
    fn test_math_backend_defaults_and_stereo8() {
        let (c, s) = ScalarBackend::fast_unit_phasor(0.5);
        assert!(c.is_finite() && s.is_finite());
        assert_eq!(ScalarBackend::crc32c(b"123456789"), 0xe3069283);

        let v = ScalarVec4::splat(0.5);
        let mat = [[0.1f32; 4]; 4];
        let _ = ScalarBackend::fma_mat4x4_vec4(v, &mat, &[0.5; 4]);
        assert!((ScalarBackend::rsqrt_nr2_f32(4.0) - 0.5).abs() < 1.0e-5);
        let (p0, p1) = ScalarBackend::rsqrt_nr2_pair_f32(4.0, 9.0);
        assert!((p0 - 0.5).abs() < 1.0e-5);
        assert!((p1 - 1.0 / 3.0).abs() < 1.0e-5);

        let (dl_l, dl_r) = ScalarBackend::fma_mat4x4_stereo(v, v, &mat, v, v);
        assert_eq!(dl_l.to_array(), dl_r.to_array());
        let fp16_v = ScalarBackend::vec4_to_fp16x4(v);
        assert_eq!(fp16_v, ScalarBackend::f32x4_to_fp16x4(&v.to_array()));
        assert_eq!(
            ScalarBackend::fp16x4_to_vec4(&fp16_v).to_array(),
            ScalarBackend::fp16x4_to_f32x4(&fp16_v)
        );

        {
            use dnam_kernels::KernelVec4;
            type DefS8 = <crate::math::DefaultMathBackend as MathBackend>::Stereo8;
            type ScalS8 = <ScalarBackend as MathBackend>::Stereo8;
            let al = [0.25_f32, -0.75, 1.5, -2.25];
            let ar = [-0.5_f32, 1.25, -1.75, 0.5];
            let bl = [0.5_f32, 0.25, -0.5, 1.0];
            let br = [1.0_f32, -0.5, 0.75, -1.25];

            let da = DefS8::load_lr(&al, &ar);
            let db = DefS8::load_lr(&bl, &br);
            let sa = ScalS8::load_lr(&al, &ar);
            let sb = ScalS8::load_lr(&bl, &br);
            let (dl, dr) = da.add(db).into_lr();
            let (sl, sr) = sa.add(sb).into_lr();
            assert_eq!(
                KernelVec4::to_array(dl).map(f32::to_bits),
                KernelVec4::to_array(sl).map(f32::to_bits)
            );
            assert_eq!(
                KernelVec4::to_array(dr).map(f32::to_bits),
                KernelVec4::to_array(sr).map(f32::to_bits)
            );
        }
    }
}
