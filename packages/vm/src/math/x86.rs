//! x86-64 (`x86_64`) hardware SIMD backends (`X86Sse2Backend`, `X86Sse42Backend`,
//! `X86Avx2FmaBackend`) and lock-free runtime CPUID capability probes.

pub mod avx2;
pub mod sse2;
pub mod sse42;

pub use avx2::{Avx2FmaVec4, Avx2Stereo8, X86Avx2FmaBackend, warm_avx2_fma_support_cache};
pub use sse2::{Sse2Vec4, X86Sse2Backend};
pub use sse42::{Sse42Vec4, X86Sse42Backend, warm_sse42_support_cache};

use super::BackendKind;

/// Compile-time optimal safe default x86-64 `MathBackend` for baseline `DefaultMathBackend`.
#[cfg(all(
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
))]
pub type X86SimdBackend = X86Avx2FmaBackend;
/// Compile-time optimal safe default x86-64 `Vec4` register.
#[cfg(all(
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
))]
pub type X86Vec4 = Avx2FmaVec4;
/// Compile-time optimal safe default x86-64 `Stereo8` register.
#[cfg(all(
    target_feature = "avx2",
    target_feature = "fma",
    target_feature = "f16c"
))]
pub type X86Stereo8 = Avx2Stereo8;

/// Compile-time optimal safe default x86-64 `MathBackend` (`SSE4.2` baseline).
#[cfg(all(
    target_feature = "sse4.2",
    not(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))
))]
pub type X86SimdBackend = X86Sse42Backend;
/// Compile-time optimal safe default x86-64 `Vec4` register (`SSE4.2`).
#[cfg(all(
    target_feature = "sse4.2",
    not(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))
))]
pub type X86Vec4 = Sse42Vec4;
/// Compile-time optimal safe default x86-64 `Stereo8` register (`SSE4.2`).
#[cfg(all(
    target_feature = "sse4.2",
    not(all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))
))]
pub type X86Stereo8 = dnam_kernels::InterleavedStereo8<Sse42Vec4>;

/// Compile-time optimal safe default x86-64 `MathBackend` (`SSE2` architectural baseline).
#[cfg(not(any(
    target_feature = "sse4.2",
    all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    )
)))]
pub type X86SimdBackend = X86Sse2Backend;
/// Compile-time optimal safe default x86-64 `Vec4` register (`SSE2`).
#[cfg(not(any(
    target_feature = "sse4.2",
    all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    )
)))]
pub type X86Vec4 = Sse2Vec4;
/// Compile-time optimal safe default x86-64 `Stereo8` register (`SSE2`).
#[cfg(not(any(
    target_feature = "sse4.2",
    all(
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    )
)))]
pub type X86Stereo8 = dnam_kernels::InterleavedStereo8<Sse2Vec4>;

/// Pre-warms lock-free x86-64 capability caches (`SSE4.2`, `AVX2+FMA+F16C`) during cold setup.
#[inline]
pub fn warm_x86_feature_caches() {
    sse42::warm_sse42_support_cache();
    avx2::warm_avx2_fma_support_cache();
}

/// Returns the highest-capability `x86_64` math backend supported on the host CPU.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_feature_detection_is_valid_variant() {
        warm_x86_feature_caches();
        let kind = detect_x86_backend();
        assert!(matches!(
            kind,
            BackendKind::X86Sse2 | BackendKind::X86Sse42 | BackendKind::X86Avx2Fma
        ));
    }
}
