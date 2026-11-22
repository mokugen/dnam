//! ARMv9-A `FEAT_SVE2` vector register ([`Sve2Vec4`]) and stereo pair alias ([`Sve2Stereo8`]).

pub mod backend;
pub mod kernel_vec4;

pub use backend::{ArmSve2Backend, warm_sve2_support_cache};

use super::neon::NeonVec4;
use crate::math::SimdVec4;

/// ARMv9-A `SVE2` 4-lane `f32` vector wrapper executing hardware capability probes while keeping
/// fixed `128-bit` lane semantics for bit-exact `0-ULP` parity with `NEON`, `SSE2`, & `WGSL` on
/// stable Rust (portable vector-length-agnostic intrinsics await stabilization).
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Sve2Vec4(pub NeonVec4);

/// Interleaved `2 × Sve2Vec4` 8-lane stereo vector (`4L + 4R`).
pub type Sve2Stereo8 = dnam_kernels::InterleavedStereo8<Sve2Vec4>;

impl SimdVec4 for Sve2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(NeonVec4::splat(val))
    }

    #[inline(always)]
    fn dup_low_pair(self) -> Self {
        Self(self.0.dup_low_pair())
    }

    #[inline(always)]
    fn dup_high_pair(self) -> Self {
        Self(self.0.dup_high_pair())
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(NeonVec4::load(src))
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        self.0.store(dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(self.0.fma(a.0, b.0))
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        Self(self.0.fma_lane::<LANE>(a.0, v.0))
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        Self(self.0.mul_lane::<LANE>(v.0))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(self.0.fms(a.0, b.0))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        Self(self.0.pair_swap_neg())
    }

    #[inline(always)]
    fn odd_pair_swap_neg(self) -> Self {
        Self(self.0.odd_pair_swap_neg())
    }

    #[inline(always)]
    fn rotate_left_1(self) -> Self {
        Self(self.0.rotate_left_1())
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self(self.0.flush_subnormal())
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        Self(self.0.scrub_recursive_state())
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        self.0.horizontal_sum()
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        self.0.horizontal_sum_pair(rhs.0)
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self(self.0.rcp_pos_seed_bits())
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self(self.0.rsqrt_pos_seed_bits())
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self(self.0.pow2_from_biased_int_bits())
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sve2_vec4_matches_neon_vec4() {
        let a = Sve2Vec4::load(&[1.0, -2.0, 3.5, 0.25]);
        let b = Sve2Vec4::splat(2.0);
        let na = NeonVec4::load(&[1.0, -2.0, 3.5, 0.25]);
        let nb = NeonVec4::splat(2.0);
        assert_eq!(a.fma(b, a).to_array(), na.fma(nb, na).to_array());
        assert_eq!(a.horizontal_sum().to_bits(), na.horizontal_sum().to_bits());
        assert_eq!(a.exp2_clamped5().to_array(), na.exp2_clamped5().to_array());
        crate::math::backend::simd_vec4::tests::verify_all_simd_and_kernel_ops::<Sve2Vec4>();
    }
}
