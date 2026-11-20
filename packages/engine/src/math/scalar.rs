//! Portable 16-byte aligned scalar/auto-vectorized math backend (`ScalarBackend`).
//!
//! Serves both as the universal fallback on any CPU architecture and as the deterministic
//! reference oracle in cross-backend parity tests (`tests/backend_parity.rs`).

use dnam_kernels::InterleavedStereo8;

use super::{BackendKind, MathBackend, SimdVec4};
use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};

/// 16-byte SIMD-aligned 4-lane scalar vector (`#[repr(C, align(16))]`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScalarVec4(pub [f32; 4]);

impl SimdVec4 for ScalarVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self([val; 4])
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0] + a.0[0] * b.0[0],
            self.0[1] + a.0[1] * b.0[1],
            self.0[2] + a.0[2] * b.0[2],
            self.0[3] + a.0[3] * b.0[3],
        ])
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        let s = v.0[LANE];
        Self([
            self.0[0] + a.0[0] * s,
            self.0[1] + a.0[1] * s,
            self.0[2] + a.0[2] * s,
            self.0[3] + a.0[3] * s,
        ])
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0] - a.0[0] * b.0[0],
            self.0[1] - a.0[1] * b.0[1],
            self.0[2] - a.0[2] * b.0[2],
            self.0[3] - a.0[3] * b.0[3],
        ])
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self([
            self.0[0].max(rhs.0[0]),
            self.0[1].max(rhs.0[1]),
            self.0[2].max(rhs.0[2]),
            self.0[3].max(rhs.0[3]),
        ])
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self([
            self.0[0].min(rhs.0[0]),
            self.0[1].min(rhs.0[1]),
            self.0[2].min(rhs.0[2]),
            self.0[3].min(rhs.0[3]),
        ])
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self([
            self.0[0].abs(),
            self.0[1].abs(),
            self.0[2].abs(),
            self.0[3].abs(),
        ])
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self([
            f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(self.0[0].to_bits())),
            f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(self.0[1].to_bits())),
            f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(self.0[2].to_bits())),
            f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(self.0[3].to_bits())),
        ])
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self([
            f32::from_bits(
                crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(self.0[0].to_bits() >> 1),
            ),
            f32::from_bits(
                crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(self.0[1].to_bits() >> 1),
            ),
            f32::from_bits(
                crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(self.0[2].to_bits() >> 1),
            ),
            f32::from_bits(
                crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(self.0[3].to_bits() >> 1),
            ),
        ])
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::flush_recursive_state(self)
    }
}

impl dnam_kernels::KernelVec4 for ScalarVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        <Self as SimdVec4>::splat(val)
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        <Self as SimdVec4>::splat_pair(a, b)
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        <Self as SimdVec4>::from_array(arr)
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        <Self as SimdVec4>::to_array(self)
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        <Self as SimdVec4>::load(src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        <Self as SimdVec4>::store(self, dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        <Self as SimdVec4>::add(self, rhs)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        <Self as SimdVec4>::sub(self, rhs)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        <Self as SimdVec4>::mul(self, rhs)
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        <Self as SimdVec4>::fma(self, a, b)
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        <Self as SimdVec4>::fms(self, a, b)
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        <Self as SimdVec4>::fma_lane::<LANE>(self, a, v)
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        <Self as SimdVec4>::mul_lane::<LANE>(self, v)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        <Self as SimdVec4>::max(self, rhs)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        <Self as SimdVec4>::min(self, rhs)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        <Self as SimdVec4>::abs(self)
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        self.0[0] <= rhs.0[0]
            && self.0[1] <= rhs.0[1]
            && self.0[2] <= rhs.0[2]
            && self.0[3] <= rhs.0[3]
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::pair_swap_neg(self)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        <Self as SimdVec4>::flush_subnormal(self)
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        Self(self.0.map(dnam_kernels::scrub_and_flush_f32_bits))
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        <Self as SimdVec4>::horizontal_sum(self)
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        <Self as SimdVec4>::horizontal_sum_pair(self, rhs)
    }

    #[inline(always)]
    fn dot(self, rhs: Self) -> f32 {
        <Self as SimdVec4>::dot(self, rhs)
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        <Self as SimdVec4>::rcp_pos_seed_bits(self)
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        <Self as SimdVec4>::rsqrt_pos_seed_bits(self)
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        <Self as SimdVec4>::pow2_from_biased_int_bits(self)
    }
}

/// Portable scalar reference backend implementing [`MathBackend`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScalarBackend;

impl Crc32cBackend for ScalarBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        crc32c_software_slice(crc_init, data)
    }
}

impl MathBackend for ScalarBackend {
    type Vec4 = ScalarVec4;
    type Stereo8 = InterleavedStereo8<ScalarVec4>;
    const NAME: &'static str = "Scalar (16B Aligned Reference)";
    const KIND: BackendKind = BackendKind::Scalar;
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_scalar_vec4_ops_and_backend() {
        let a = ScalarVec4::from_array([1.0, -2.0, 3.0, -4.0]);
        let b = ScalarVec4::splat(2.0);
        assert_eq!(a.abs().to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.min(b).to_array(), [1.0, -2.0, 2.0, -4.0]);
        assert_eq!(a.max(b).to_array(), [2.0, 2.0, 3.0, 2.0]);
        assert_eq!(a.add(b).to_array(), [3.0, 0.0, 5.0, -2.0]);
        assert_eq!(a.sub(b).to_array(), [-1.0, -4.0, 1.0, -6.0]);
        assert_eq!(a.mul(b).to_array(), [2.0, -4.0, 6.0, -8.0]);
        assert_eq!(a.fma(b, b).to_array(), [5.0, 2.0, 7.0, 0.0]);
        assert_eq!(a.fms(b, b).to_array(), [-3.0, -6.0, -1.0, -8.0]);
        assert_eq!(a.fma_lane::<1>(b, a).to_array(), [-3.0, -6.0, -1.0, -8.0]);
        assert_eq!(a.horizontal_sum(), -2.0);
        let pos = ScalarVec4::from_array([0.25, 1.0, 2.5, 8.0]);
        assert!((pos.rcp_pos_nr2().to_array()[0] - 4.0).abs() < 1e-4);
        assert!((pos.rcp_pos_nr3().to_array()[2] - 0.4).abs() < 1e-6);
        assert!((pos.rsqrt_pos_nr2().to_array()[0] - 2.0).abs() < 1e-2);
        assert!((pos.rsqrt_pos_nr3().to_array()[1] - 1.0).abs() < 1e-5);
        assert_eq!(ScalarBackend::crc32c_update(0, b"123456789"), 0xe306_9283);
    }

    #[test]
    fn test_scalar_vec4_kernel_vec4_bridge() {
        type K = ScalarVec4;
        let z = <K as dnam_kernels::KernelVec4>::zero();
        assert_eq!(<K as dnam_kernels::KernelVec4>::to_array(z), [0.0; 4]);
        let a = <K as dnam_kernels::KernelVec4>::from_array([1.0, -2.0, 3.0, -4.0]);
        let b = <K as dnam_kernels::KernelVec4>::splat(2.0);
        let p = <K as dnam_kernels::KernelVec4>::splat_pair(1.5, -0.5);
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(p),
            [1.5, -0.5, 1.5, -0.5]
        );
        let mut stored = [0.0_f32; 4];
        <K as dnam_kernels::KernelVec4>::store(a, &mut stored);
        let reloaded = <K as dnam_kernels::KernelVec4>::load(&stored);
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(reloaded),
            [1.0, -2.0, 3.0, -4.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::add(a, b)),
            [3.0, 0.0, 5.0, -2.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::sub(a, b)),
            [-1.0, -4.0, 1.0, -6.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::mul(a, b)),
            [2.0, -4.0, 6.0, -8.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::fma(
                a, b, b
            )),
            [5.0, 2.0, 7.0, 0.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::fms(
                a, b, b
            )),
            [-3.0, -6.0, -1.0, -8.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::fma_lane::<1>(a, b, a)
            ),
            [-3.0, -6.0, -1.0, -8.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::mul_lane::<2>(b, a)
            ),
            [6.0; 4]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::max(a, b)),
            [2.0, 2.0, 3.0, 2.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::min(a, b)),
            [1.0, -2.0, 2.0, -4.0]
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(<K as dnam_kernels::KernelVec4>::abs(a)),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert!(<K as dnam_kernels::KernelVec4>::all_le(
            <K as dnam_kernels::KernelVec4>::splat(-5.0),
            a
        ));
        assert!(!<K as dnam_kernels::KernelVec4>::all_le(a, b));
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::pair_swap_neg(a)
            ),
            [2.0, 1.0, 4.0, 3.0]
        );
        let subn = <K as dnam_kernels::KernelVec4>::from_array([1.0e-40, -1.0e-40, 2.5, -3.5]);
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::flush_subnormal(subn)
            ),
            [0.0, 0.0, 2.5, -3.5]
        );
        let dirty =
            <K as dnam_kernels::KernelVec4>::from_array([f32::NAN, f32::INFINITY, 1.0e-40, -2.5]);
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::flush_recursive_state(dirty)
            ),
            [0.0, 0.0, 0.0, -2.5]
        );
        assert_eq!(<K as dnam_kernels::KernelVec4>::horizontal_sum(a), -2.0);
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::horizontal_sum_pair(a, b),
            (-2.0, 8.0)
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::dot(a, <K as dnam_kernels::KernelVec4>::splat(1.0)),
            -2.0
        );
        let pos = <K as dnam_kernels::KernelVec4>::from_array([0.25, 1.0, 2.5, 8.0]);
        assert!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::rcp_pos_seed_bits(pos)
            )[0]
            .is_finite()
        );
        assert!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::rsqrt_pos_seed_bits(pos)
            )[1]
            .is_finite()
        );
        assert_eq!(
            <K as dnam_kernels::KernelVec4>::to_array(
                <K as dnam_kernels::KernelVec4>::pow2_from_biased_int_bits(z)
            ),
            [1.0; 4]
        );
        assert_eq!(a.dup_low_pair().to_array(), [1.0, -2.0, 1.0, -2.0]);
        assert_eq!(a.dup_high_pair().to_array(), [3.0, -4.0, 3.0, -4.0]);
    }

    proptest! {
        #[test]
        fn prop_scalar_vec4_abs_min_max_match_lanes(
            v0 in -100.0_f32..100.0_f32,
            v1 in -100.0_f32..100.0_f32,
            v2 in -100.0_f32..100.0_f32,
            v3 in -100.0_f32..100.0_f32,
            bound in -50.0_f32..50.0_f32,
        ) {
            let v = ScalarVec4::from_array([v0, v1, v2, v3]);
            let b = ScalarVec4::splat(bound);
            prop_assert_eq!(v.abs().to_array(), [v0.abs(), v1.abs(), v2.abs(), v3.abs()]);
            prop_assert_eq!(
                v.min(b).to_array(),
                [v0.min(bound), v1.min(bound), v2.min(bound), v3.min(bound)]
            );
            prop_assert_eq!(
                v.max(b).to_array(),
                [v0.max(bound), v1.max(bound), v2.max(bound), v3.max(bound)]
            );
            prop_assert!((v.horizontal_sum() - ((v0 + v1) + (v2 + v3))).abs() < 1e-4);
        }
    }
}
