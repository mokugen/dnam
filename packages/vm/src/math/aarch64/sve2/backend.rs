//! ARMv9-A `FEAT_SVE2` [`MathBackend`] (`ArmSve2Backend`) and lock-free runtime feature cache.

use core::sync::atomic::{AtomicU8, Ordering};

use super::{Sve2Stereo8, Sve2Vec4};
use crate::math::BackendKind;
use crate::math::MathBackend;
use crate::math::aarch64::neon::NeonBackend;
use crate::math::crc32c::Crc32cBackend;

const CACHE_UNSET: u8 = 0;
const CACHE_UNSUPPORTED: u8 = 1;
const CACHE_SUPPORTED: u8 = 2;

static SVE2_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(CACHE_UNSET);

#[inline]
fn probe_sve2_support() -> bool {
    if cfg!(target_feature = "sve2") {
        return true;
    }
    #[cfg(feature = "std")]
    {
        std::arch::is_aarch64_feature_detected!("sve2")
    }
    #[cfg(all(not(feature = "std"), target_os = "linux"))]
    {
        // Linux traps EL0 `ID_AA64PFR0_EL1` [35:32] (`SVE`) + `ID_AA64ZFR0_EL1` [3:0] (`SVE2`).
        let pfr0: u64;
        // SAFETY: Linux user-space emulates EL0 `ID_AA64PFR0_EL1` feature reads safely.
        unsafe {
            core::arch::asm!("mrs {}, id_aa64pfr0_el1", out(reg) pfr0, options(nomem, nostack, preserves_flags));
        }
        if ((pfr0 >> 32) & 0xF) == 0 {
            return false;
        }
        let zfr0: u64;
        // SAFETY: `SVE` is present above so `ID_AA64ZFR0_EL1` is readable on Linux user-space.
        unsafe {
            core::arch::asm!("mrs {}, id_aa64zfr0_el1", out(reg) zfr0, options(nomem, nostack, preserves_flags));
        }
        (zfr0 & 0xF) >= 1
    }
    #[cfg(not(any(feature = "std", target_os = "linux")))]
    {
        false
    }
}

/// Pre-warms the lock-free `FEAT_SVE2` capability cache before entering hot loops.
#[inline]
pub fn warm_sve2_support_cache() {
    if SVE2_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_UNSET {
        let state = if probe_sve2_support() {
            CACHE_SUPPORTED
        } else {
            CACHE_UNSUPPORTED
        };
        SVE2_SUPPORT_CACHE.store(state, Ordering::Relaxed);
    }
}

/// ARMv9-A `FEAT_SVE2` [`MathBackend`] (`ArmSve2Backend`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSve2Backend;

impl ArmSve2Backend {
    /// Returns `true` if `FEAT_SVE2` is supported on the host core (lock-free after warm).
    #[inline]
    pub fn is_supported_cached() -> bool {
        match SVE2_SUPPORT_CACHE.load(Ordering::Relaxed) {
            CACHE_SUPPORTED => true,
            CACHE_UNSUPPORTED => false,
            _ => {
                warm_sve2_support_cache();
                SVE2_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_SUPPORTED
            }
        }
    }
}

impl Crc32cBackend for ArmSve2Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        NeonBackend::crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSve2Backend {
    type Vec4 = Sve2Vec4;
    type Stereo8 = Sve2Stereo8;
    const NAME: &'static str = "ARMv9 SVE2";
    const KIND: BackendKind = BackendKind::ArmSve2;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        let cols = NeonBackend::transpose_4x4(r0.0, r1.0, r2.0, r3.0);
        [
            Sve2Vec4(cols[0]),
            Sve2Vec4(cols[1]),
            Sve2Vec4(cols[2]),
            Sve2Vec4(cols[3]),
        ]
    }

    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        NeonBackend::vec4_to_fp16x4(src.0)
    }

    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        Sve2Vec4(NeonBackend::fp16x4_to_vec4(src))
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        NeonBackend::f32x4_to_fp16x4(src)
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        NeonBackend::fp16x4_to_f32x4(src)
    }

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        NeonBackend::rsqrt_nr2_f32(v0)
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        NeonBackend::rsqrt_nr2_pair_f32(v0, v1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sve2_backend_identity_and_crc32c() {
        assert_eq!(ArmSve2Backend::NAME, "ARMv9 SVE2");
        assert_eq!(ArmSve2Backend::KIND, BackendKind::ArmSve2);
        assert_eq!(ArmSve2Backend::crc32c(b"123456789"), 0xE306_9283);
        let _ = ArmSve2Backend::is_supported_cached();
        crate::math::backend::tests::verify_math_backend_ops::<ArmSve2Backend>();
    }
}
