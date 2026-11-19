//! Unified `x86_64` Hardware Math Library (`X86Avx2FmaBackend`, `X86Sse42Backend`, `X86Sse2Backend`)
//! with runtime CPU feature detection (`std::arch::is_x86_feature_detected!`).
//!
//! Submodules:
//! - [`sse2`]: Baseline `x86-64-v1` (`SSE` + `SSE2` 128-bit `__m128`).
//! - [`sse42`]: `x86-64-v2` (`SSE3` + `SSSE3` + `SSE4.1` + `SSE4.2` hardware `_mm_crc32_u64`),
//!   guarded by `std::arch::is_x86_feature_detected!`.
//! - [`avx2`]: `x86-64-v3` (`AVX` + `AVX2` + `FMA` + `F16C`, 256-bit `_mm256_fmadd_ps` dot products,
//!   `_mm_fmadd_ps` / `_mm_fnmadd_ps`, `_mm_cvtps_ph` / `_mm_cvtph_ps`), guarded by
//!   `std::arch::is_x86_feature_detected!`.

pub mod avx2;
pub mod sse2;
pub mod sse42;
pub mod trainer_f64;

pub use avx2::{Avx2FmaVec4, Avx2Stereo8, X86Avx2FmaBackend};
pub use sse2::{Sse2Vec4, X86Sse2Backend};
pub use sse42::{Sse42Vec4, X86Sse42Backend};
pub use trainer_f64::{Avx2FmaF64Vec2, Sse2F64Vec2};

use super::BackendKind;

/// Alias for the highest-tier x86-64 vector register wrapper ([`Avx2FmaVec4`]).
pub type X86Vec4 = Avx2FmaVec4;
/// Alias for the highest-tier x86-64 stereo 8-lane vector ([`Avx2Stereo8`]).
pub type X86Stereo8 = Avx2Stereo8;
/// Alias for the highest-tier x86-64 hardware math backend ([`X86Avx2FmaBackend`]).
pub type X86SimdBackend = X86Avx2FmaBackend;

/// Pre-warms every runtime-detected `x86_64` SIMD and hardware CRC32C capability cache outside the
/// real-time audio callback so subsequent [`X86Avx2FmaBackend`] and [`X86Sse42Backend`] checks
/// perform only a single relaxed atomic load.
#[inline]
pub fn warm_x86_feature_caches() {
    let _ = X86Avx2FmaBackend::is_supported();
    let _ = X86Sse42Backend::is_supported();
}

/// Detects the highest-capability `x86_64` backend supported by the current CPU at runtime:
/// `X86Avx2Fma` (`x86-64-v3`) -> `X86Sse42` (`x86-64-v2`) -> `X86Sse2` (`x86-64-v1`).
#[inline]
pub fn detect_x86_backend() -> BackendKind {
    warm_x86_feature_caches();
    if X86Avx2FmaBackend::is_supported_cached() {
        BackendKind::X86Avx2Fma
    } else if X86Sse42Backend::is_supported_cached() {
        BackendKind::X86Sse42
    } else {
        BackendKind::X86Sse2
    }
}
