//! ARM64 (`aarch64`) hardware SIMD backends (`NeonBackend`, `ArmSve2Backend`, `ArmSmeBackend`)
//! and cached lock-free capability detection (`warm_aarch64_feature_caches`, `detect_aarch64_backend`).

pub mod neon;
pub mod sme;
pub mod sve2;

pub use neon::{NeonBackend, NeonStereo8, NeonVec4, is_crc32_supported_cached};
pub use sme::{ArmSmeBackend, SmeStereo8};
pub use sve2::{ArmSve2Backend, Sve2Stereo8, Sve2Vec4};

use super::BackendKind;

/// Pre-warms lock-free ARM64 capability caches (`CRC32`, `SVE2`, `SME`) during cold setup.
#[inline]
pub fn warm_aarch64_feature_caches() {
    neon::warm_crc32_support_cache();
    sve2::warm_sve2_support_cache();
    sme::warm_sme_support_cache();
}

/// Returns the highest-capability `aarch64` math backend supported on the host CPU.
#[inline]
pub fn detect_aarch64_backend() -> BackendKind {
    warm_aarch64_feature_caches();
    if sme::ArmSmeBackend::is_supported_cached() {
        BackendKind::ArmSme
    } else if sve2::ArmSve2Backend::is_supported_cached() {
        BackendKind::ArmSve2
    } else {
        BackendKind::ArmNeon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aarch64_feature_detection_is_valid_variant() {
        warm_aarch64_feature_caches();
        let kind = detect_aarch64_backend();
        assert!(matches!(
            kind,
            BackendKind::ArmNeon | BackendKind::ArmSve2 | BackendKind::ArmSme
        ));
    }
}
