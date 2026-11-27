//! x86-64 `AVX2 + FMA + F16C` 4-lane `f32` vector ([`Avx2FmaVec4`]) and [`X86Avx2FmaBackend`].

pub mod stereo8;

use core::arch::x86_64::{
    __cpuid, __cpuid_count, __m128, __m128i, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
    _mm_cvtph_ps, _mm_cvtps_ph, _mm_fmadd_ps, _mm_fnmadd_ps, _mm_loadl_epi64, _mm_storel_epi64,
    _xgetbv,
};
use core::fmt;
use core::sync::atomic::{AtomicU8, Ordering};

pub use stereo8::Avx2Stereo8;
use zerocopy::IntoBytes;

use super::sse2::Sse2Vec4;
use super::sse42::X86Sse42Backend;
use crate::graph::slab::{
    SonaHotSlab, process_block_sona, process_block_sona_inline, process_tick16_const,
    process_tick16_const_inline, step_control_tick_3khz, step_control_tick_3khz_inline,
    step_frame_4x_sona, step_frame_4x_sona_inline, step_frame_sona, step_frame_sona_inline,
};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{
    BackendKind, FP16_FINITE_MAX_F32, MathBackend, SimdVec4, f32_to_fp16, fp16_to_f32,
};
use crate::types::VmError;

const CACHE_UNSET: u8 = 0;
const CACHE_UNSUPPORTED: u8 = 1;
const CACHE_SUPPORTED: u8 = 2;

static AVX2_FMA_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(CACHE_UNSET);

/// x86-64 `AVX2 + FMA + F16C` 128-bit `f32` lane register (`__m128`).
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2FmaVec4(pub Sse2Vec4);

impl fmt::Debug for Avx2FmaVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Avx2FmaVec4")
            .field(&self.to_array())
            .finish()
    }
}

#[inline]
fn probe_avx2_fma_support() -> bool {
    if cfg!(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    )) {
        return true;
    }
    // CPUID leaf 1 is a safe unprivileged read checking `FMA` (bit 12), `OSXSAVE` (bit 27),
    // `AVX` (bit 28), and `F16C` (bit 29).
    let leaf1 = __cpuid(1);
    let req_ecx = (1 << 12) | (1 << 27) | (1 << 28) | (1 << 29);
    if (leaf1.ecx & req_ecx) != req_ecx {
        return false;
    }
    // SAFETY: OSXSAVE (ECX bit 27) was just verified set above, so XGETBV(0) is valid and unprivileged.
    let xcr0 = unsafe { _xgetbv(0) };
    if (xcr0 & 0x6) != 0x6 {
        return false;
    }
    // CPUID leaf 7 subleaf 0 is a safe unprivileged read checking `AVX2` (EBX bit 5).
    let leaf7 = __cpuid_count(7, 0);
    (leaf7.ebx & (1 << 5)) != 0
}

#[inline(always)]
pub(super) fn is_avx2_fma_hot() -> bool {
    if cfg!(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    )) {
        true
    } else {
        AVX2_FMA_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_SUPPORTED
    }
}

/// Pre-warms the lock-free `x86_64` `AVX2+FMA+F16C` capability cache during cold startup.
#[inline]
pub fn warm_avx2_fma_support_cache() {
    if AVX2_FMA_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_UNSET {
        let state = if probe_avx2_fma_support() {
            CACHE_SUPPORTED
        } else {
            CACHE_UNSUPPORTED
        };
        AVX2_FMA_SUPPORT_CACHE.store(state, Ordering::Relaxed);
    }
}

#[target_feature(enable = "fma")]
#[inline]
unsafe fn fma_m128_hw(acc: __m128, a: __m128, b: __m128) -> __m128 {
    _mm_fmadd_ps(a, b, acc)
}

#[target_feature(enable = "fma")]
#[inline]
unsafe fn fms_m128_hw(acc: __m128, a: __m128, b: __m128) -> __m128 {
    // `- (a * b) + acc == acc - a * b` in single rounding!
    _mm_fnmadd_ps(a, b, acc)
}

#[target_feature(enable = "f16c")]
#[inline]
unsafe fn f32x4_to_fp16x4_hw(src: __m128) -> [u16; 4] {
    let v = Sse2Vec4(src);
    let lanes = dnam_kernels::KernelVec4::to_array(v);
    if !lanes[0].is_finite()
        || !lanes[1].is_finite()
        || !lanes[2].is_finite()
        || !lanes[3].is_finite()
    {
        return lanes.map(f32_to_fp16);
    }
    let clamped = v
        .clamp(
            Sse2Vec4::splat(-FP16_FINITE_MAX_F32),
            Sse2Vec4::splat(FP16_FINITE_MAX_F32),
        )
        .0;
    let packed: __m128i = _mm_cvtps_ph(clamped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let mut out = [0_u16; 4];
    // SAFETY: `out` is a valid 8-byte stack `[u16; 4]` buffer for a 64-bit store.
    unsafe {
        _mm_storel_epi64(out.as_mut_ptr().cast::<__m128i>(), packed);
    }
    out
}

#[target_feature(enable = "f16c")]
#[inline]
unsafe fn fp16x4_to_f32x4_hw(src: &[u16; 4]) -> __m128 {
    // SAFETY: `src` is a valid 8-byte `&[u16; 4]` slice for a 64-bit load.
    let packed = unsafe { _mm_loadl_epi64(src.as_bytes().as_ptr().cast::<__m128i>()) };
    _mm_cvtph_ps(packed)
}

impl SimdVec4 for Avx2FmaVec4 {
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
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed hardware `FMA` support.
            Self(Sse2Vec4(unsafe { fma_m128_hw(self.0.0, a.0.0, b.0.0) }))
        } else {
            Self(self.0.fma(a.0, b.0))
        }
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        self.fma(a, Self(v.0.splat_lane::<LANE>()))
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        self.mul(Self(v.0.splat_lane::<LANE>()))
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed hardware `FMA` support.
            Self(Sse2Vec4(unsafe { fms_m128_hw(self.0.0, a.0.0, b.0.0) }))
        } else {
            Self(self.0.fms(a.0, b.0))
        }
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

impl dnam_kernels::KernelVec4 for Avx2FmaVec4 {
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

/// x86-64 `AVX2 + FMA + F16C` [`MathBackend`] (`X86Avx2FmaBackend`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct X86Avx2FmaBackend;

impl X86Avx2FmaBackend {
    /// Returns `true` if `AVX2 + FMA + F16C` are supported on the host CPU.
    #[inline]
    pub fn is_supported_cached() -> bool {
        match AVX2_FMA_SUPPORT_CACHE.load(Ordering::Relaxed) {
            CACHE_SUPPORTED => true,
            CACHE_UNSUPPORTED => false,
            _ => {
                warm_avx2_fma_support_cache();
                AVX2_FMA_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_SUPPORTED
            }
        }
    }
}

impl Crc32cBackend for X86Avx2FmaBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        X86Sse42Backend::crc32c_update(crc_init, data)
    }
}

impl MathBackend for X86Avx2FmaBackend {
    type Vec4 = Avx2FmaVec4;
    type Stereo8 = Avx2Stereo8;
    const NAME: &'static str = "x86 AVX2";
    const KIND: BackendKind = BackendKind::X86Avx2Fma;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        let cols = super::sse2::X86Sse2Backend::transpose_4x4(r0.0, r1.0, r2.0, r3.0);
        [
            Avx2FmaVec4(cols[0]),
            Avx2FmaVec4(cols[1]),
            Avx2FmaVec4(cols[2]),
            Avx2FmaVec4(cols[3]),
        ]
    }

    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed hardware `F16C` support.
            unsafe { f32x4_to_fp16x4_hw(src.0.0) }
        } else {
            src.to_array().map(f32_to_fp16)
        }
    }

    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed hardware `F16C` support.
            Avx2FmaVec4(Sse2Vec4(unsafe { fp16x4_to_f32x4_hw(src) }))
        } else {
            Self::Vec4::load(&src.map(fp16_to_f32))
        }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        Self::vec4_to_fp16x4(Self::Vec4::load(src))
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        Self::fp16x4_to_vec4(src).to_array()
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn process_block_avx2_hw(
    slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    process_block_sona_inline(slab, in_l, in_r, out_l, out_r)
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn process_tick16_avx2_hw(
    slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>,
    in_l: &[f32; 16],
    in_r: &[f32; 16],
    out_l: &mut [f32; 16],
    out_r: &mut [f32; 16],
) -> Result<(), VmError> {
    process_tick16_const_inline(slab, in_l, in_r, out_l, out_r)
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn step_frame_avx2_hw(
    slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    if slab.cold().offline_bounce {
        step_frame_4x_sona_inline(slab, in_l, in_r)
    } else {
        step_frame_sona_inline(slab, in_l, in_r)
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn step_control_tick_avx2_hw(slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>) {
    step_control_tick_3khz_inline(slab);
}

/// Dispatches `SonaHotSlab::process_block` inside a `#[target_feature(enable = "avx2,fma,f16c")]`
/// block region after checking host support once at entry (falling back safely on non-AVX2 CPUs).
#[inline(always)]
pub(crate) fn dispatch_avx2_process_block(
    slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    if !X86Avx2FmaBackend::is_supported_cached() {
        return process_block_sona(slab, in_l, in_r, out_l, out_r);
    }
    // SAFETY: `is_supported_cached()` verified `AVX2 + FMA + F16C` support at block entry.
    unsafe { process_block_avx2_hw(slab, in_l, in_r, out_l, out_r) }
}

/// Dispatches `process_tick16_const` inside a `#[target_feature(enable = "avx2,fma,f16c")]`
/// tick region after checking host support once at entry (falling back safely on non-AVX2 CPUs).
#[inline(always)]
pub(crate) fn dispatch_avx2_process_tick16(
    slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>,
    in_l: &[f32; 16],
    in_r: &[f32; 16],
    out_l: &mut [f32; 16],
    out_r: &mut [f32; 16],
) -> Result<(), VmError> {
    if !X86Avx2FmaBackend::is_supported_cached() {
        return process_tick16_const(slab, in_l, in_r, out_l, out_r);
    }
    // SAFETY: `is_supported_cached()` verified `AVX2 + FMA + F16C` support at tick entry.
    unsafe { process_tick16_avx2_hw(slab, in_l, in_r, out_l, out_r) }
}

/// Dispatches `step_frame` inside a `#[target_feature(enable = "avx2,fma,f16c")]` frame region
/// after checking host support once at entry (falling back safely on non-AVX2 CPUs).
#[inline(always)]
pub(crate) fn dispatch_avx2_step_frame(
    slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    if !X86Avx2FmaBackend::is_supported_cached() {
        return if slab.cold().offline_bounce {
            step_frame_4x_sona(slab, in_l, in_r)
        } else {
            step_frame_sona(slab, in_l, in_r)
        };
    }
    // SAFETY: `is_supported_cached()` verified `AVX2 + FMA + F16C` support at frame entry.
    unsafe { step_frame_avx2_hw(slab, in_l, in_r) }
}

/// Dispatches `step_control_tick_3khz` inside a `#[target_feature(enable = "avx2,fma,f16c")]`
/// control-tick region after checking host support once at entry (falling back safely on non-AVX2 CPUs).
#[inline(always)]
pub(crate) fn dispatch_avx2_step_control_tick(slab: &mut SonaHotSlab<'_, X86Avx2FmaBackend>) {
    if !X86Avx2FmaBackend::is_supported_cached() {
        step_control_tick_3khz(slab);
        return;
    }
    // SAFETY: `is_supported_cached()` verified `AVX2 + FMA + F16C` support at control-tick entry.
    unsafe { step_control_tick_avx2_hw(slab) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_fma_backend_identity_and_fp16() {
        warm_avx2_fma_support_cache();
        assert_eq!(X86Avx2FmaBackend::NAME, "x86 AVX2");
        assert_eq!(X86Avx2FmaBackend::crc32c(b"123456789"), 0xE306_9283);
        let x = [1.0, -2.5, 0.25, 4.0];
        let v = Avx2FmaVec4::load(&x);
        let h = X86Avx2FmaBackend::vec4_to_fp16x4(v);
        assert_eq!(X86Avx2FmaBackend::fp16x4_to_vec4(&h).to_array(), x);
        crate::math::backend::tests::verify_math_backend_ops::<X86Avx2FmaBackend>();
    }
}
