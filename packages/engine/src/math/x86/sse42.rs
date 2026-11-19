//! x86-64 SSE4.2 (`x86-64-v2`) hardware math backend (`Sse42Vec4` & `X86Sse42Backend`).

use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_blendv_ps, _mm_castps_si128,
    _mm_castsi128_ps, _mm_cmple_ps, _mm_crc32_u8, _mm_crc32_u64, _mm_cvtss_f32, _mm_hadd_ps,
    _mm_min_epi32, _mm_movehdup_ps, _mm_movehl_ps, _mm_set1_epi32, _mm_setr_ps, _mm_shuffle_ps,
    _mm_sub_epi32,
};
use core::sync::atomic::{AtomicU8, Ordering};
use dnam_kernels::InterleavedStereo8;

use super::sse2::{Sse2Vec4, X86Sse2Backend, sse2_tracker_exp_scales_f32};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, SimdVec4};

/// 128-bit x86-64 SSE4.2 4-lane `f32` register wrapper (`__m128`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct Sse42Vec4(pub __m128);

#[target_feature(enable = "sse3")]
#[inline]
pub(crate) fn sse3_horizontal_sum(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let pairs = _mm_add_ps(v, shuf);
    let hi = _mm_movehl_ps(pairs, pairs);
    _mm_cvtss_f32(_mm_add_ss(pairs, hi))
}

#[target_feature(enable = "sse3")]
#[inline]
pub(crate) fn sse3_horizontal_sum_pair(lhs: __m128, rhs: __m128) -> (f32, f32) {
    let pairs = _mm_hadd_ps(lhs, rhs);
    let totals = _mm_hadd_ps(pairs, pairs);
    (
        _mm_cvtss_f32(totals),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(totals, totals)),
    )
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn sse41_tracker_exp_scales_hw(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    let max_v = _mm_setr_ps(max_l, max_r, 1.0, 1.0);
    let exp_bits = _mm_and_si128(_mm_castps_si128(max_v), _mm_set1_epi32(0x7f80_0000));
    let su_v = _mm_min_epi32(exp_bits, _mm_set1_epi32(253 << 23));
    let sd_v = _mm_sub_epi32(_mm_set1_epi32(254 << 23), su_v);
    let (sd_f, su_f) = (_mm_castsi128_ps(sd_v), _mm_castsi128_ps(su_v));
    (
        _mm_cvtss_f32(sd_f),
        _mm_cvtss_f32(su_f),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(sd_f, sd_f)),
        _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(su_f, su_f)),
    )
}

/// Evaluates `T1` tracker exponent-split scales using SSE4.1 `_mm_min_epi32` when supported,
/// or SSE2 fallback otherwise.
#[inline]
pub(crate) fn sse42_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    if X86Sse42Backend::is_supported_cached() {
        // SAFETY: Guarded by `X86Sse42Backend::is_supported_cached()`.
        unsafe { sse41_tracker_exp_scales_hw(max_l, max_r) }
    } else {
        sse2_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[target_feature(enable = "sse4.2")]
#[inline]
fn sse42_crc32c_update_hw(crc: u32, data: &[u8]) -> u32 {
    let mut state = u64::from(!crc);
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        state = _mm_crc32_u64(state, u64::from_le_bytes(*chunk));
    }
    let mut crc32 = state as u32;
    for &byte in remainder {
        crc32 = _mm_crc32_u8(crc32, byte);
    }
    !crc32
}

#[inline]
pub(crate) fn cached_cpuid_support(cache: &AtomicU8, detect: fn() -> bool) -> bool {
    let cached = cache.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 2;
    }
    let supported = detect();
    cache.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    supported
}

impl SimdVec4 for Sse42Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(Sse2Vec4::splat(val).0)
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        Self(Sse2Vec4::splat_pair(a, b).0)
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(Sse2Vec4::load(src).0)
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        Self(Sse2Vec4::from_array(arr).0)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        Sse2Vec4(self.0).store(dst);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).add(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).sub(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).mul(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self(Sse2Vec4(self.0).fma(Sse2Vec4(a.0), Sse2Vec4(b.0)).0)
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        Self(
            Sse2Vec4(self.0)
                .fma_lane::<LANE>(Sse2Vec4(a.0), Sse2Vec4(v.0))
                .0,
        )
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        Self(Sse2Vec4(self.0).mul_lane::<LANE>(Sse2Vec4(v.0)).0)
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self(Sse2Vec4(self.0).fms(Sse2Vec4(a.0), Sse2Vec4(b.0)).0)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).max(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self(Sse2Vec4(self.0).min(Sse2Vec4(rhs.0)).0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self(Sse2Vec4(self.0).abs().0)
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        Self(Sse2Vec4(self.0).pair_swap_neg().0)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self(Sse2Vec4(self.0).flush_subnormal().0)
    }

    #[inline]
    fn horizontal_sum(self) -> f32 {
        if X86Sse42Backend::is_supported_cached() {
            // SAFETY: Guarded by `X86Sse42Backend::is_supported_cached()`.
            unsafe { sse3_horizontal_sum(self.0) }
        } else {
            Sse2Vec4(self.0).horizontal_sum()
        }
    }

    #[inline]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        if X86Sse42Backend::is_supported_cached() {
            // SAFETY: Guarded by `X86Sse42Backend::is_supported_cached()`.
            unsafe { sse3_horizontal_sum_pair(self.0, rhs.0) }
        } else {
            Sse2Vec4(self.0).horizontal_sum_pair(Sse2Vec4(rhs.0))
        }
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self(Sse2Vec4(self.0).rcp_pos_seed_bits().0)
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self(Sse2Vec4(self.0).rsqrt_pos_seed_bits().0)
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self(Sse2Vec4(self.0).pow2_from_biased_int_bits().0)
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn sse41_select_le(lhs: __m128, rhs: __m128, if_true: __m128, if_false: __m128) -> __m128 {
    _mm_blendv_ps(if_false, if_true, _mm_cmple_ps(lhs, rhs))
}

impl dnam_kernels::KernelVec4 for Sse42Vec4 {
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
    fn neg(self) -> Self {
        Self(<Sse2Vec4 as dnam_kernels::KernelVec4>::neg(Sse2Vec4(self.0)).0)
    }

    #[inline]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        if X86Sse42Backend::is_supported_cached() {
            // SAFETY: Guarded by `X86Sse42Backend::is_supported_cached()` confirming `SSE4.1`.
            unsafe { Self(sse41_select_le(lhs.0, rhs.0, if_true.0, if_false.0)) }
        } else {
            Self(
                <Sse2Vec4 as dnam_kernels::KernelVec4>::select_le(
                    Sse2Vec4(lhs.0),
                    Sse2Vec4(rhs.0),
                    Sse2Vec4(if_true.0),
                    Sse2Vec4(if_false.0),
                )
                .0,
            )
        }
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        <Sse2Vec4 as dnam_kernels::KernelVec4>::all_le(Sse2Vec4(self.0), Sse2Vec4(rhs.0))
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
        Self(<Sse2Vec4 as dnam_kernels::KernelVec4>::flush_recursive_state(Sse2Vec4(self.0)).0)
    }

    #[inline]
    fn horizontal_sum(self) -> f32 {
        <Self as SimdVec4>::horizontal_sum(self)
    }

    #[inline]
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

/// x86-64 SSE4.2 (`x86-64-v2`) hardware math backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Sse42Backend;

/// Cached `SSE3`/`SSSE3`/`SSE4.1`/`SSE4.2` support state (`0 = uninit`, `1 = no`, `2 = yes`).
static SSE42_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

impl X86Sse42Backend {
    /// Returns the cached `SSE4.2` capability flag with a single relaxed atomic load and zero OS
    /// or `std::arch::is_x86_feature_detected!` / `detect_and_initialize` calls.
    #[inline(always)]
    pub fn is_supported_cached() -> bool {
        SSE42_SUPPORT_CACHE.load(Ordering::Relaxed) == 2
    }

    #[cold]
    fn detect_uncached() -> bool {
        #[cfg(any(test, feature = "std"))]
        {
            std::arch::is_x86_feature_detected!("sse3")
                && std::arch::is_x86_feature_detected!("ssse3")
                && std::arch::is_x86_feature_detected!("sse4.1")
                && std::arch::is_x86_feature_detected!("sse4.2")
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            // SAFETY: `__cpuid` is always supported on 64-bit x86-64 processors.
            unsafe {
                let mask = (1u32 << 0) | (1u32 << 9) | (1u32 << 19) | (1u32 << 20);
                (core::arch::x86_64::__cpuid(1).ecx & mask) == mask
            }
        }
    }

    /// Returns `true` if the host x86-64 CPU supports `SSE3`, `SSSE3`, `SSE4.1`, and `SSE4.2`
    /// (cached in [`SSE42_SUPPORT_CACHE`]).
    #[inline]
    pub fn is_supported() -> bool {
        cached_cpuid_support(&SSE42_SUPPORT_CACHE, Self::detect_uncached)
    }
}

impl Crc32cBackend for X86Sse42Backend {
    #[inline]
    fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
        if Self::is_supported_cached() {
            // SAFETY: Guarded by `Self::is_supported_cached()`.
            unsafe { sse42_crc32c_update_hw(crc, data) }
        } else {
            X86Sse2Backend::crc32c_update(crc, data)
        }
    }
}

impl MathBackend for X86Sse42Backend {
    type Vec4 = Sse42Vec4;
    type Stereo8 = InterleavedStereo8<Sse42Vec4>;
    const NAME: &'static str = "x86-64 SSE4.2 (128-bit __m128 + CRC32C)";
    const KIND: BackendKind = BackendKind::X86Sse42;

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        X86Sse2Backend::rsqrt_nr2_f32(v0)
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        X86Sse2Backend::rsqrt_nr2_pair_f32(v0, v1)
    }
}

#[cfg(test)]
mod tests {
    use super::X86Sse42Backend;

    #[test]
    fn test_sse42_cached_feature_flag_matches_warmed_detection() {
        crate::math::warm_simd_feature_caches();
        assert_eq!(
            X86Sse42Backend::is_supported(),
            X86Sse42Backend::is_supported_cached()
        );
    }
}
