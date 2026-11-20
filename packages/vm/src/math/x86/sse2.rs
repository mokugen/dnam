//! x86-64 architectural baseline `SSE2` 4-lane `f32` SIMD vector ([`Sse2Vec4`]) and
//! [`X86Sse2Backend`].

use core::arch::x86_64::{
    __m128, _mm_add_epi32, _mm_add_ps, _mm_add_ss, _mm_and_ps, _mm_and_si128, _mm_andnot_ps,
    _mm_andnot_si128, _mm_castps_si128, _mm_castsi128_ps, _mm_cmple_ps, _mm_cmplt_epi32,
    _mm_cvtss_f32, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_movehl_ps, _mm_movelh_ps,
    _mm_movemask_ps, _mm_mul_ps, _mm_or_ps, _mm_or_si128, _mm_set_ps, _mm_set1_epi32, _mm_set1_ps,
    _mm_shuffle_ps, _mm_slli_epi32, _mm_srli_epi32, _mm_storeu_ps, _mm_sub_epi32, _mm_sub_ps,
    _mm_xor_ps,
};
use core::fmt;

use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};
use crate::math::{
    BackendKind, IEEE754_F32_EXP_BIAS, IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, MathBackend,
    SimdVec4,
};

/// x86-64 architectural baseline `SSE2` 128-bit `f32` vector (`__m128`).
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Sse2Vec4(pub __m128);

// SAFETY: `__m128` is a pure 16-byte SIMD value register with no thread affinity.
unsafe impl Send for Sse2Vec4 {}
// SAFETY: `__m128` has immutable value semantics when shared across threads.
unsafe impl Sync for Sse2Vec4 {}

impl fmt::Debug for Sse2Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Sse2Vec4").field(&self.to_array()).finish()
    }
}

impl SimdVec4 for Sse2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: `SSE2` is guaranteed on all `x86_64` CPUs.
        Self(unsafe { _mm_set1_ps(val) })
    }

    #[inline(always)]
    fn dup_low_pair(self) -> Self {
        // SAFETY: `SSE2` is baseline on `x86_64`.
        Self(unsafe { _mm_movelh_ps(self.0, self.0) })
    }

    #[inline(always)]
    fn dup_high_pair(self) -> Self {
        // SAFETY: `SSE2` is baseline on `x86_64`.
        Self(unsafe { _mm_movehl_ps(self.0, self.0) })
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        // SAFETY: `[f32; 4]` has 4 initialized `f32` lanes (`16 B`).
        Self(unsafe { _mm_loadu_ps(src.as_ptr()) })
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        // SAFETY: `[f32; 4]` provides 4 writable `f32` lanes (`16 B`).
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: Baseline `SSE2` intrinsic.
        Self(unsafe { _mm_add_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: Baseline `SSE2` intrinsic.
        Self(unsafe { _mm_sub_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: Baseline `SSE2` intrinsic.
        Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // SAFETY: `SSE2` computes unfused `self + a * b` reliably on baseline targets.
        Self(unsafe { _mm_add_ps(self.0, _mm_mul_ps(a.0, b.0)) })
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // SAFETY: `SSE2` computes unfused `self - a * b` cleanly on baseline targets.
        Self(unsafe { _mm_sub_ps(self.0, _mm_mul_ps(a.0, b.0)) })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: Baseline `SSE2` intrinsic.
        Self(unsafe { _mm_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: Baseline `SSE2` intrinsic.
        Self(unsafe { _mm_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: Clears the IEEE-754 sign bit across all 4 lanes via `SSE2` bitwise AND.
        Self(unsafe {
            _mm_castsi128_ps(_mm_and_si128(
                _mm_castps_si128(self.0),
                _mm_set1_epi32(0x7FFF_FFFF),
            ))
        })
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        // SAFETY: `_mm_shuffle_ps::<0xB1>` swaps `[0,1,2,3] -> [1,0,3,2]` and XOR flips `[0,2]`.
        Self(unsafe {
            let swapped = _mm_shuffle_ps::<0xB1>(self.0, self.0);
            _mm_xor_ps(swapped, _mm_set_ps(0.0, -0.0, 0.0, -0.0))
        })
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        // SAFETY: Signed `i32` comparison matches unsigned magnitude in `[0, 0x7FFF_FFFF]`.
        Self(unsafe {
            let bits = _mm_castps_si128(self.0);
            let abs_bits = _mm_and_si128(bits, _mm_set1_epi32(0x7FFF_FFFF));
            let is_sub = _mm_cmplt_epi32(abs_bits, _mm_set1_epi32(0x0080_0000));
            let signed_zero = _mm_and_si128(bits, _mm_set1_epi32(i32::MIN));
            _mm_castsi128_ps(_mm_or_si128(
                _mm_and_si128(is_sub, signed_zero),
                _mm_andnot_si128(is_sub, bits),
            ))
        })
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        // SAFETY: Scrubs `exp == 0xFF` (`NaN`/`±Inf`) to `+0.0` and subnormals to signed zero.
        Self(unsafe {
            let bits = _mm_castps_si128(self.0);
            let abs_bits = _mm_and_si128(bits, _mm_set1_epi32(0x7FFF_FFFF));
            let is_sub = _mm_cmplt_epi32(abs_bits, _mm_set1_epi32(0x0080_0000));
            let signed_zero = _mm_and_si128(bits, _mm_set1_epi32(i32::MIN));
            let flushed = _mm_or_si128(
                _mm_and_si128(is_sub, signed_zero),
                _mm_andnot_si128(is_sub, bits),
            );
            let is_finite = _mm_cmplt_epi32(abs_bits, _mm_set1_epi32(0x7F80_0000));
            _mm_castsi128_ps(_mm_and_si128(flushed, is_finite))
        })
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // SAFETY: Reduces in exact `(a0+a1)+(a2+a3)` tree order using baseline `SSE2`.
        unsafe {
            let swapped = _mm_shuffle_ps::<0xB1>(self.0, self.0);
            let pair_sums = _mm_add_ps(self.0, swapped);
            let hi = _mm_movehl_ps(pair_sums, pair_sums);
            _mm_cvtss_f32(_mm_add_ss(pair_sums, hi))
        }
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        // SAFETY: `SSE2` 32-bit integer lane subtraction.
        Self(unsafe {
            _mm_castsi128_ps(_mm_sub_epi32(
                _mm_set1_epi32(IEEE754_F32_RCP_SEED as i32),
                _mm_castps_si128(self.0),
            ))
        })
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        // SAFETY: `SSE2` 32-bit logical right shift + subtraction.
        Self(unsafe {
            _mm_castsi128_ps(_mm_sub_epi32(
                _mm_set1_epi32(IEEE754_F32_RSQRT_SEED as i32),
                _mm_srli_epi32::<1>(_mm_castps_si128(self.0)),
            ))
        })
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        // SAFETY: `SSE2` 32-bit exponent bias addition + shift.
        Self(unsafe {
            _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(
                _mm_castps_si128(self.0),
                _mm_set1_epi32(IEEE754_F32_EXP_BIAS as i32),
            )))
        })
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
    }
}

impl dnam_kernels::KernelVec4 for Sse2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        <Self as SimdVec4>::splat(val)
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        <Self as SimdVec4>::splat_pair(a, b)
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
    fn neg(self) -> Self {
        // SAFETY: Flipping the IEEE-754 sign bit via `SSE2` XOR.
        Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        // SAFETY: `SSE2` mask select.
        Self(unsafe {
            let mask = _mm_cmple_ps(lhs.0, rhs.0);
            _mm_or_ps(_mm_and_ps(mask, if_true.0), _mm_andnot_ps(mask, if_false.0))
        })
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        // SAFETY: `_mm_movemask_ps` produces `0b1111` iff all 4 lanes satisfy `<=`.
        unsafe { _mm_movemask_ps(_mm_cmple_ps(self.0, rhs.0)) == 0b1111 }
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

/// x86-64 architectural baseline `SSE2` backend (`MathBackend`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Sse2Backend;

impl Crc32cBackend for X86Sse2Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        crc32c_software_slice(crc_init, data)
    }
}

impl MathBackend for X86Sse2Backend {
    type Vec4 = Sse2Vec4;
    type Stereo8 = dnam_kernels::InterleavedStereo8<Sse2Vec4>;
    const NAME: &'static str = "x86 SSE2";
    const KIND: BackendKind = BackendKind::X86Sse2;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        let mut a = r0.0;
        let mut b = r1.0;
        let mut c = r2.0;
        let mut d = r3.0;
        // SAFETY: Standard `SSE2` `_MM_TRANSPOSE4_PS` register transpose.
        unsafe {
            core::arch::x86_64::_MM_TRANSPOSE4_PS(&mut a, &mut b, &mut c, &mut d);
        }
        [Sse2Vec4(a), Sse2Vec4(b), Sse2Vec4(c), Sse2Vec4(d)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse2_vec4_transpose_and_ops() {
        let a = Sse2Vec4::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.pair_swap_neg().to_array(), [-2.0, 1.0, -4.0, 3.0]);
        assert_eq!(a.horizontal_sum(), 10.0);
        let cols = X86Sse2Backend::transpose_4x4(a, a, a, a);
        assert_eq!(cols[0].to_array(), [1.0; 4]);
    }
}
