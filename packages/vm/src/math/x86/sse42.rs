//! x86-64 `SSE4.2` vector wrapper ([`Sse42Vec4`]) and hardware Castagnoli `CRC32C`
//! (`_mm_crc32_u64`/`_mm_crc32_u8`) backend ([`X86Sse42Backend`]).

use core::arch::x86_64::{__cpuid, _mm_crc32_u8, _mm_crc32_u64};
use core::sync::atomic::{AtomicU8, Ordering};

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::U64;

use super::sse2::Sse2Vec4;
use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};
use crate::math::{BackendKind, MathBackend, SimdVec4};

const CACHE_UNSET: u8 = 0;
const CACHE_UNSUPPORTED: u8 = 1;
const CACHE_SUPPORTED: u8 = 2;

static SSE42_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(CACHE_UNSET);

/// x86-64 `SSE4.2` 4-lane `f32` vector register wrapper.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Sse42Vec4(pub Sse2Vec4);

#[inline]
fn probe_sse42_support() -> bool {
    if cfg!(target_feature = "sse4.2") {
        return true;
    }
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("sse4.2")
    }
    #[cfg(not(feature = "std"))]
    {
        // SAFETY: CPUID leaf 1 is an unprivileged read instruction guaranteed on all x86_64 CPUs.
        let leaf = unsafe { __cpuid(1) };
        (leaf.ecx & (1 << 20)) != 0
    }
}

/// Pre-warms the lock-free `x86_64` `SSE4.2` capability cache during cold startup.
#[inline]
pub fn warm_sse42_support_cache() {
    if SSE42_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_UNSET {
        let state = if probe_sse42_support() {
            CACHE_SUPPORTED
        } else {
            CACHE_UNSUPPORTED
        };
        SSE42_SUPPORT_CACHE.store(state, Ordering::Relaxed);
    }
}

/// Updates a Castagnoli `CRC32C` digest using `x86_64` `SSE4.2` hardware instructions.
///
/// # Safety
/// Caller must guarantee `SSE4.2` is supported on the executing CPU.
#[target_feature(enable = "sse4.2")]
pub unsafe fn sse42_crc32c_update_hw(crc_init: u32, data: &[u8]) -> u32 {
    let mut crc = u64::from(!crc_init);
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let word = U64::ref_from_bytes(chunk.as_slice())
            .map(U64::get)
            .unwrap_or(0);
        crc = _mm_crc32_u64(crc, word);
    }
    let mut crc32 = crc as u32;
    for &byte in remainder {
        crc32 = _mm_crc32_u8(crc32, byte);
    }
    !crc32
}

impl SimdVec4 for Sse42Vec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self(Sse2Vec4::splat(val))
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
        Self(Sse2Vec4::load(src))
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
        Self(<Sse2Vec4 as dnam_kernels::KernelVec4>::neg(self.0))
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        Self(dnam_kernels::KernelVec4::select_le(
            lhs.0, rhs.0, if_true.0, if_false.0,
        ))
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        dnam_kernels::KernelVec4::all_le(self.0, rhs.0)
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

/// x86-64 `SSE4.2` [`MathBackend`] with hardware Castagnoli `CRC32C`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Sse42Backend;

impl X86Sse42Backend {
    /// Returns `true` if `SSE4.2` is supported on the host CPU (lock-free after warm).
    #[inline]
    pub fn is_supported_cached() -> bool {
        match SSE42_SUPPORT_CACHE.load(Ordering::Relaxed) {
            CACHE_SUPPORTED => true,
            CACHE_UNSUPPORTED => false,
            _ => {
                warm_sse42_support_cache();
                SSE42_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_SUPPORTED
            }
        }
    }
}

impl Crc32cBackend for X86Sse42Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        if Self::is_supported_cached() {
            // SAFETY: `is_supported_cached()` guarantees `SSE4.2` support.
            unsafe { sse42_crc32c_update_hw(crc_init, data) }
        } else {
            crc32c_software_slice(crc_init, data)
        }
    }
}

impl MathBackend for X86Sse42Backend {
    type Vec4 = Sse42Vec4;
    type Stereo8 = dnam_kernels::InterleavedStereo8<Sse42Vec4>;
    const NAME: &'static str = "x86 SSE4.2";
    const KIND: BackendKind = BackendKind::X86Sse42;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        let cols = super::sse2::X86Sse2Backend::transpose_4x4(r0.0, r1.0, r2.0, r3.0);
        [
            Sse42Vec4(cols[0]),
            Sse42Vec4(cols[1]),
            Sse42Vec4(cols[2]),
            Sse42Vec4(cols[3]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse42_backend_identity_and_crc32c() {
        assert_eq!(X86Sse42Backend::NAME, "x86 SSE4.2");
        assert_eq!(X86Sse42Backend::crc32c(b"123456789"), 0xE306_9283);
        let _ = __cpuid(0);
    }
}
