//! ARMv9.2-A `FEAT_SME` [`MathBackend`] (`ArmSmeBackend`) and lock-free runtime feature cache.

use core::sync::atomic::{AtomicU8, Ordering};

use super::neon::{NeonBackend, NeonVec4};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend};

const CACHE_UNSET: u8 = 0;
const CACHE_UNSUPPORTED: u8 = 1;
const CACHE_SUPPORTED: u8 = 2;

static SME_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(CACHE_UNSET);

/// Interleaved `2 × NeonVec4` 8-lane stereo vector (`4L + 4R`) backing `ArmSmeBackend`.
pub type SmeStereo8 = dnam_kernels::InterleavedStereo8<NeonVec4>;

#[inline]
fn probe_sme_support() -> bool {
    if cfg!(target_feature = "sme") {
        return true;
    }
    #[cfg(all(not(feature = "std"), target_os = "linux"))]
    {
        // Linux traps EL0 `ID_AA64PFR1_EL1`; bits [27:24] >= 1 denotes `FEAT_SME`.
        let pfr1: u64;
        // SAFETY: Linux user-space safely emulates EL0 `ID_AA64PFR1_EL1` feature reads.
        unsafe {
            core::arch::asm!("mrs {}, id_aa64pfr1_el1", out(reg) pfr1, options(nomem, nostack, preserves_flags));
        }
        ((pfr1 >> 24) & 0xF) >= 1
    }
    #[cfg(not(all(not(feature = "std"), target_os = "linux")))]
    {
        false
    }
}

/// Pre-warms the lock-free `FEAT_SME` capability cache before entering real-time loops.
#[inline]
pub fn warm_sme_support_cache() {
    if SME_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_UNSET {
        let state = if probe_sme_support() {
            CACHE_SUPPORTED
        } else {
            CACHE_UNSUPPORTED
        };
        SME_SUPPORT_CACHE.store(state, Ordering::Relaxed);
    }
}

/// ARMv9.2-A `FEAT_SME` [`MathBackend`] (`ArmSmeBackend`) with lock-free runtime detection and
/// `NEON`-compatible `128-bit`/`Stereo8` lanes preserving `0-ULP` cross-ISA parity on stable Rust.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSmeBackend;

impl ArmSmeBackend {
    /// Returns `true` if `FEAT_SME` is supported on the host core (lock-free after warm).
    #[inline]
    pub fn is_supported_cached() -> bool {
        match SME_SUPPORT_CACHE.load(Ordering::Relaxed) {
            CACHE_SUPPORTED => true,
            CACHE_UNSUPPORTED => false,
            _ => {
                warm_sme_support_cache();
                SME_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_SUPPORTED
            }
        }
    }
}

impl Crc32cBackend for ArmSmeBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        NeonBackend::crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSmeBackend {
    type Vec4 = NeonVec4;
    type Stereo8 = SmeStereo8;
    const NAME: &'static str = "ARM SME";
    const KIND: BackendKind = BackendKind::ArmSme;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        NeonBackend::transpose_4x4(r0, r1, r2, r3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sme_backend_identity_and_crc32c() {
        assert_eq!(ArmSmeBackend::NAME, "ARM SME");
        assert_eq!(ArmSmeBackend::KIND, BackendKind::ArmSme);
        assert_eq!(ArmSmeBackend::crc32c(b"123456789"), 0xE306_9283);
        let _ = ArmSmeBackend::is_supported_cached();
        crate::math::backend::tests::verify_math_backend_ops::<ArmSmeBackend>();
    }
}
