//! Unified `aarch64` Hardware Math Library (`ArmSmeBackend`, `ArmSve2Backend`, `NeonBackend`)
//! with runtime CPU feature detection.
//!
//! Submodules:
//! - [`neon`]: Baseline 128-bit NEON (`float32x4_t`, `vfmaq_laneq_f32`, hardware `FCVTN`/`FCVTL`, `__crc32cd`).
//! - [`sve2`]: Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`, `z0..z31` + `p0` `vl4`) without NEON fallback.
//! - [`sme`]: ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`) with runtime fallback to [`ArmSve2Backend`] and [`NeonBackend`].

pub mod neon;
pub mod sme;
pub mod sme_f64;
pub mod sve2;
pub mod sve2_f64;
pub mod trainer_f64;

pub use neon::{NeonBackend, NeonVec4};
pub use sme::{ArmSmeBackend, SmeStereo8};
pub use sve2::{ArmSve2Backend, Sve2Stereo8, Sve2Vec4};
pub use sve2_f64::Sve2F64Vec2;
pub use trainer_f64::NeonF64Vec2;

/// Stereo 8-lane vector alias for baseline ARM64 NEON ([`NeonVec4`]).
pub type NeonStereo8 = dnam_kernels::InterleavedStereo8<NeonVec4>;

use super::BackendKind;

/// Pre-warms every runtime-detected `aarch64` SIMD and hardware CRC32C cache outside the
/// real-time audio callback so subsequent [`ArmSmeBackend`], [`ArmSve2Backend`], and `FEAT_CRC32`
/// checks perform only a single relaxed atomic load.
#[inline]
pub fn warm_aarch64_feature_caches() {
    let _ = ArmSmeBackend::is_supported();
    let _ = ArmSmeBackend::is_f64f64_supported();
    let _ = ArmSve2Backend::is_supported();
    let _ = neon::warm_crc32_support_cache();
}

/// Detects the highest-capability `aarch64` backend supported by the current CPU at runtime:
/// `ArmSme` (`FEAT_SME`) -> `ArmSve2` (`FEAT_SVE2`) -> `ArmNeon` (baseline `aarch64`).
#[inline]
pub fn detect_aarch64_backend() -> BackendKind {
    warm_aarch64_feature_caches();
    if ArmSmeBackend::is_supported_cached() {
        BackendKind::ArmSme
    } else if ArmSve2Backend::is_supported_cached() {
        BackendKind::ArmSve2
    } else {
        BackendKind::ArmNeon
    }
}
