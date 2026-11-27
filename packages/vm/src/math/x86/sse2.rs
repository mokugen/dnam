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
    BackendKind, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_BIAS, IEEE754_F32_EXP_MASK,
    IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_BITS, IEEE754_F32_RCP_SEED,
    IEEE754_F32_RSQRT_SEED, IEEE754_F32_SIGN_MASK, MathBackend, SimdVec4,
};

pub(super) const SCRUB_BIAS_SIGNED_I32: i32 =
    ((IEEE754_F32_IMPLICIT_BIT << 1) ^ IEEE754_F32_SIGN_MASK) as i32;
pub(super) const SCRUB_LIMIT_SIGNED_I32: i32 =
    (((IEEE754_F32_EXP_MASK - IEEE754_F32_IMPLICIT_BIT) << 1) ^ IEEE754_F32_SIGN_MASK) as i32;

const _: () = assert!(SCRUB_BIAS_SIGNED_I32 == 0x8100_0000_u32 as i32);
const _: () = assert!(SCRUB_LIMIT_SIGNED_I32 == 0x7E00_0000_i32);

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

impl Sse2Vec4 {
    #[inline(always)]
    pub(super) fn splat_lane<const LANE: usize>(self) -> Self {
        const { assert!(LANE < 4) };
        Self(match LANE {
            0 => _mm_shuffle_ps::<0x00>(self.0, self.0),
            1 => _mm_shuffle_ps::<0x55>(self.0, self.0),
            2 => _mm_shuffle_ps::<0xAA>(self.0, self.0),
            3 => _mm_shuffle_ps::<0xFF>(self.0, self.0),
            _ => unreachable!(),
        })
    }
}

impl SimdVec4 for Sse2Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(_mm_set1_ps(val))
    }

    #[inline(always)]
    fn dup_low_pair(self) -> Self {
        Self(_mm_movelh_ps(self.0, self.0))
    }

    #[inline(always)]
    fn dup_high_pair(self) -> Self {
        Self(_mm_movehl_ps(self.0, self.0))
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
        Self(_mm_add_ps(self.0, rhs.0))
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(_mm_sub_ps(self.0, rhs.0))
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(_mm_mul_ps(self.0, rhs.0))
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // `SSE2` baseline computes unfused `self + a * b`.
        Self(_mm_add_ps(self.0, _mm_mul_ps(a.0, b.0)))
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        self.fma(a, v.splat_lane::<LANE>())
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        self.mul(v.splat_lane::<LANE>())
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // `SSE2` baseline computes unfused `self - a * b`.
        Self(_mm_sub_ps(self.0, _mm_mul_ps(a.0, b.0)))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(_mm_max_ps(self.0, rhs.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(_mm_min_ps(self.0, rhs.0))
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // Clears the IEEE-754 sign bit across all 4 lanes via `SSE2` bitwise AND.
        Self(_mm_castsi128_ps(_mm_and_si128(
            _mm_castps_si128(self.0),
            _mm_set1_epi32(IEEE754_F32_ABS_MASK as i32),
        )))
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        // `_mm_shuffle_ps::<0xB1>` swaps `[0,1,2,3] -> [1,0,3,2]` and XOR flips `[0,2]`.
        let swapped = _mm_shuffle_ps::<0xB1>(self.0, self.0);
        Self(_mm_xor_ps(swapped, _mm_set_ps(0.0, -0.0, 0.0, -0.0)))
    }

    #[inline(always)]
    fn odd_pair_swap_neg(self) -> Self {
        // `_mm_shuffle_ps::<0x1B>` reverses `[0,1,2,3] -> [3,2,1,0]` and XOR flips `[1,3]`.
        let reversed = _mm_shuffle_ps::<0x1B>(self.0, self.0);
        Self(_mm_xor_ps(reversed, _mm_set_ps(-0.0, 0.0, -0.0, 0.0)))
    }

    #[inline(always)]
    fn rotate_left_1(self) -> Self {
        // `_mm_shuffle_ps::<0x39>` rotates `[0,1,2,3] -> [1,2,3,0]`.
        Self(_mm_shuffle_ps::<0x39>(self.0, self.0))
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        // Signed `i32` comparison matches unsigned magnitude in `[0, IEEE754_F32_ABS_MASK]`.
        let bits = _mm_castps_si128(self.0);
        let abs_bits = _mm_and_si128(bits, _mm_set1_epi32(IEEE754_F32_ABS_MASK as i32));
        let is_sub = _mm_cmplt_epi32(abs_bits, _mm_set1_epi32(IEEE754_F32_IMPLICIT_BIT as i32));
        let signed_zero = _mm_and_si128(bits, _mm_set1_epi32(i32::MIN));
        Self(_mm_castsi128_ps(_mm_or_si128(
            _mm_and_si128(is_sub, signed_zero),
            _mm_andnot_si128(is_sub, bits),
        )))
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        // Signed-bias range check keeps normal finite lanes intact while canonicalizing `-0.0`,
        // subnormals, `±Inf`, and `NaN` to `+0.0` (`0x0000_0000`).
        let bits = _mm_castps_si128(self.0);
        let biased = _mm_sub_epi32(
            _mm_slli_epi32::<1>(bits),
            _mm_set1_epi32(SCRUB_BIAS_SIGNED_I32),
        );
        let is_normal = _mm_cmplt_epi32(biased, _mm_set1_epi32(SCRUB_LIMIT_SIGNED_I32));
        Self(_mm_castsi128_ps(_mm_and_si128(bits, is_normal)))
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // Reduces in exact `(a0+a1)+(a2+a3)` tree order using baseline `SSE2`.
        let swapped = _mm_shuffle_ps::<0xB1>(self.0, self.0);
        let pair_sums = _mm_add_ps(self.0, swapped);
        let hi = _mm_movehl_ps(pair_sums, pair_sums);
        _mm_cvtss_f32(_mm_add_ss(pair_sums, hi))
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self(_mm_castsi128_ps(_mm_sub_epi32(
            _mm_set1_epi32(IEEE754_F32_RCP_SEED as i32),
            _mm_castps_si128(self.0),
        )))
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self(_mm_castsi128_ps(_mm_sub_epi32(
            _mm_set1_epi32(IEEE754_F32_RSQRT_SEED as i32),
            _mm_srli_epi32::<1>(_mm_castps_si128(self.0)),
        )))
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self(_mm_castsi128_ps(_mm_slli_epi32::<
            { IEEE754_F32_MANTISSA_BITS as i32 },
        >(_mm_add_epi32(
            _mm_castps_si128(self.0),
            _mm_set1_epi32(IEEE754_F32_EXP_BIAS as i32),
        ))))
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
        Self(_mm_xor_ps(self.0, _mm_set1_ps(-0.0)))
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        let mask = _mm_cmple_ps(lhs.0, rhs.0);
        Self(_mm_or_ps(
            _mm_and_ps(mask, if_true.0),
            _mm_andnot_ps(mask, if_false.0),
        ))
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        _mm_movemask_ps(_mm_cmple_ps(self.0, rhs.0)) == 0b1111
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::pair_swap_neg(self)
    }

    #[inline(always)]
    fn odd_pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::odd_pair_swap_neg(self)
    }

    #[inline(always)]
    fn rotate_left_1(self) -> Self {
        <Self as SimdVec4>::rotate_left_1(self)
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
        core::arch::x86_64::_MM_TRANSPOSE4_PS(&mut a, &mut b, &mut c, &mut d);
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
        assert_eq!(a.odd_pair_swap_neg().to_array(), [4.0, -3.0, 2.0, -1.0]);
        assert_eq!(a.rotate_left_1().to_array(), [2.0, 3.0, 4.0, 1.0]);
        assert_eq!(a.horizontal_sum(), 10.0);
        let cols = X86Sse2Backend::transpose_4x4(a, a, a, a);
        assert_eq!(cols[0].to_array(), [1.0; 4]);
        crate::math::backend::simd_vec4::tests::verify_all_simd_and_kernel_ops::<Sse2Vec4>();
    }
}
