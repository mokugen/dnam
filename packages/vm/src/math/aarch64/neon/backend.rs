//! ARM64 NEON [`MathBackend`] (`NeonBackend`) and cached `FEAT_CRC32` hardware Castagnoli
//! `CRC32C` (`0x1EDC6F41` / reflected `0x82F63B78`) acceleration.

use core::arch::aarch64::{
    __crc32cb, __crc32cd, vreinterpretq_f32_f64, vreinterpretq_f64_f32, vtrn1q_f32, vtrn1q_f64,
    vtrn2q_f32, vtrn2q_f64,
};
use core::sync::atomic::{AtomicU8, Ordering};

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::U64;

use super::{NeonStereo8, NeonVec4};
use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};
use crate::math::{BackendKind, MathBackend};

const CACHE_UNSET: u8 = 0;
const CACHE_UNSUPPORTED: u8 = 1;
const CACHE_SUPPORTED: u8 = 2;

static CRC32_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(CACHE_UNSET);

#[inline]
fn probe_crc32_support() -> bool {
    if cfg!(target_feature = "crc") {
        return true;
    }
    #[cfg(feature = "std")]
    {
        std::arch::is_aarch64_feature_detected!("crc")
    }
    #[cfg(all(not(feature = "std"), target_os = "linux"))]
    {
        // Linux traps EL0 `mrs ID_AA64ISAR0_EL1`; bits [19:16] == 0b0001 denotes `FEAT_CRC32`.
        let isar0: u64;
        // SAFETY: Linux user-space safely emulates EL0 `ID_AA64ISAR0_EL1` feature reads.
        unsafe {
            core::arch::asm!("mrs {}, id_aa64isar0_el1", out(reg) isar0, options(nomem, nostack, preserves_flags));
        }
        ((isar0 >> 16) & 0xF) >= 1
    }
    #[cfg(not(any(feature = "std", target_os = "linux")))]
    {
        false
    }
}

/// Populates the lock-free ARM64 `FEAT_CRC32` capability cache before entering real-time loops.
#[inline]
pub fn warm_crc32_support_cache() {
    if CRC32_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_UNSET {
        let state = if probe_crc32_support() {
            CACHE_SUPPORTED
        } else {
            CACHE_UNSUPPORTED
        };
        CRC32_SUPPORT_CACHE.store(state, Ordering::Relaxed);
    }
}

/// Returns `true` if ARM64 hardware `FEAT_CRC32` is supported on this CPU (lock-free after warm).
#[inline]
pub fn is_crc32_supported_cached() -> bool {
    match CRC32_SUPPORT_CACHE.load(Ordering::Relaxed) {
        CACHE_SUPPORTED => true,
        CACHE_UNSUPPORTED => false,
        _ => {
            warm_crc32_support_cache();
            CRC32_SUPPORT_CACHE.load(Ordering::Relaxed) == CACHE_SUPPORTED
        }
    }
}

/// Updates a Castagnoli `CRC32C` digest using ARM64 `FEAT_CRC32` instructions (`__crc32cd`/`__crc32cb`).
///
/// # Safety
/// Caller must guarantee ARM64 `FEAT_CRC32` is supported on the executing core.
#[target_feature(enable = "crc")]
pub unsafe fn aarch64_crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
    let mut crc = !crc_init;
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        let word = U64::ref_from_bytes(chunk.as_slice())
            .map(|w| w.get())
            .unwrap_or(0);
        crc = __crc32cd(crc, word);
    }
    for &byte in remainder {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

/// ARM64 128-bit NEON math and CRC-32C backend ([`MathBackend`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeonBackend;

impl Crc32cBackend for NeonBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        if is_crc32_supported_cached() {
            // SAFETY: `is_crc32_supported_cached()` guarantees `FEAT_CRC32` availability.
            unsafe { aarch64_crc32c_update(crc_init, data) }
        } else {
            crc32c_software_slice(crc_init, data)
        }
    }
}

impl MathBackend for NeonBackend {
    type Vec4 = NeonVec4;
    type Stereo8 = NeonStereo8;
    const NAME: &'static str = "ARM64 NEON";
    const KIND: BackendKind = BackendKind::ArmNeon;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        // SAFETY: Baseline AArch64 NEON 4x4 matrix transpose via `trn1`/`trn2`.
        unsafe {
            let t0 = vreinterpretq_f64_f32(vtrn1q_f32(r0.0, r1.0));
            let t1 = vreinterpretq_f64_f32(vtrn2q_f32(r0.0, r1.0));
            let t2 = vreinterpretq_f64_f32(vtrn1q_f32(r2.0, r3.0));
            let t3 = vreinterpretq_f64_f32(vtrn2q_f32(r2.0, r3.0));
            [
                NeonVec4(vreinterpretq_f32_f64(vtrn1q_f64(t0, t2))),
                NeonVec4(vreinterpretq_f32_f64(vtrn1q_f64(t1, t3))),
                NeonVec4(vreinterpretq_f32_f64(vtrn2q_f64(t0, t2))),
                NeonVec4(vreinterpretq_f32_f64(vtrn2q_f64(t1, t3))),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::SimdVec4;

    #[test]
    fn test_neon_crc32c_and_transpose_4x4() {
        assert_eq!(NeonBackend::crc32c(b"123456789"), 0xE306_9283);
        let rows = [
            NeonVec4::from_array([1.0, 2.0, 3.0, 4.0]),
            NeonVec4::from_array([5.0, 6.0, 7.0, 8.0]),
            NeonVec4::from_array([9.0, 10.0, 11.0, 12.0]),
            NeonVec4::from_array([13.0, 14.0, 15.0, 16.0]),
        ];
        let cols = NeonBackend::transpose_4x4(rows[0], rows[1], rows[2], rows[3]);
        assert_eq!(cols[0].to_array(), [1.0, 5.0, 9.0, 13.0]);
        assert_eq!(cols[1].to_array(), [2.0, 6.0, 10.0, 14.0]);
        assert_eq!(cols[2].to_array(), [3.0, 7.0, 11.0, 15.0]);
        assert_eq!(cols[3].to_array(), [4.0, 8.0, 12.0, 16.0]);
        crate::math::backend::tests::verify_math_backend_ops::<NeonBackend>();
    }

    proptest! {
        #[test]
        fn prop_neon_crc32c_matches_software_slice(data in proptest::collection::vec(any::<u8>(), 0..256), seed in any::<u32>()) {
            prop_assert_eq!(NeonBackend::crc32c_update(seed, &data), crc32c_software_slice(seed, &data));
        }
    }
}
