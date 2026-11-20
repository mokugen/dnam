//! [`dnam_kernels::KernelVec4`] implementation for [`super::Sve2Vec4`].

use dnam_kernels::KernelVec4;

use super::Sve2Vec4;
use crate::math::SimdVec4;

impl KernelVec4 for Sve2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(KernelVec4::splat(val))
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        Self(KernelVec4::splat_pair(a, b))
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(KernelVec4::load(src))
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        SimdVec4::store(self.0, dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(KernelVec4::add(self.0, rhs.0))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(KernelVec4::sub(self.0, rhs.0))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(KernelVec4::mul(self.0, rhs.0))
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(KernelVec4::fma(self.0, a.0, b.0))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(KernelVec4::fms(self.0, a.0, b.0))
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        Self(KernelVec4::fma_lane::<LANE>(self.0, a.0, v.0))
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        Self(KernelVec4::mul_lane::<LANE>(self.0, v.0))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(KernelVec4::max(self.0, rhs.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(KernelVec4::min(self.0, rhs.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(KernelVec4::abs(self.0))
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self(KernelVec4::neg(self.0))
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        Self(KernelVec4::select_le(lhs.0, rhs.0, if_true.0, if_false.0))
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        KernelVec4::all_le(self.0, rhs.0)
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
        <Self as SimdVec4>::scrub_recursive_state(self)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sve2_kernel_vec4_ops() {
        let a = <Sve2Vec4 as KernelVec4>::from_array([1.0, 2.0, 3.0, 4.0]);
        let b = <Sve2Vec4 as KernelVec4>::from_array([2.0, 1.0, 4.0, 3.0]);
        assert!(KernelVec4::all_le(a, <Sve2Vec4 as KernelVec4>::splat(4.0)));
        assert_eq!(
            KernelVec4::to_array(KernelVec4::select_le(a, b, a, b)),
            [1.0, 1.0, 3.0, 3.0]
        );
    }
}
