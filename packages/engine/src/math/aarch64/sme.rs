//! ARMv9.2-A Scalable Matrix Extension (`FEAT_SME` / `SME2`) hardware math backend (`ArmSmeBackend`).
//!
//! Falls back at runtime to [`ArmSve2Backend`] when `FEAT_SVE2` is present, and to [`NeonBackend`]
//! when `FEAT_SVE2` is absent (such as on Apple M4 processors, which implement `FEAT_SME` + `NEON`
//! without `FEAT_SVE2`).

use super::neon::{NeonBackend, NeonVec4, aarch64_crc32c_update};
use super::sve2::ArmSve2Backend;
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, SimdVec4};

/// ARMv9.2-A Scalable Matrix Extension (`FEAT_SME` / `SME2`) hardware backend for `aarch64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSmeBackend;

impl ArmSmeBackend {
    /// Detects at runtime whether the host `aarch64` CPU supports `FEAT_SME` (cached in an `AtomicU8`).
    #[inline]
    pub fn is_supported() -> bool {
        use core::sync::atomic::{AtomicU8, Ordering};
        static CACHE: AtomicU8 = AtomicU8::new(0);
        let c = CACHE.load(Ordering::Relaxed);
        if c != 0 {
            return c == 2;
        }
        let ok = Self::detect_uncached();
        CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
        ok
    }

    #[cold]
    fn detect_uncached() -> bool {
        #[cfg(all(target_os = "macos", any(test, feature = "std")))]
        {
            Self::sysctl_arm_feature(c"hw.optional.arm.FEAT_SME")
        }
        #[cfg(all(target_os = "linux", any(test, feature = "std")))]
        {
            Self::linux_hwcap2_bit(23)
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            let pfr1: u64;
            // SAFETY: `ID_AA64PFR1_EL1` (`SME` bits [27:24]) is readable at EL1.
            unsafe {
                core::arch::asm!(
                    "mrs {p}, id_aa64pfr1_el1",
                    p = out(reg) pfr1,
                    options(nomem, nostack, preserves_flags),
                );
            }
            ((pfr1 >> 24) & 0xF) >= 1
        }
        #[cfg(all(
            not(any(target_os = "macos", target_os = "linux")),
            any(test, feature = "std")
        ))]
        {
            false
        }
    }

    /// Detects at runtime whether the host `aarch64` CPU supports `FEAT_SME_F64F64`
    /// (double-precision `FMOPA`/`FMOPS` `ZA` tile accumulation, cached in an `AtomicU8`).
    #[inline]
    pub fn is_f64f64_supported() -> bool {
        if !Self::is_supported() {
            return false;
        }
        use core::sync::atomic::{AtomicU8, Ordering};
        static CACHE: AtomicU8 = AtomicU8::new(0);
        let c = CACHE.load(Ordering::Relaxed);
        if c != 0 {
            return c == 2;
        }
        let ok = Self::detect_f64f64_uncached();
        CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
        ok
    }

    #[cold]
    fn detect_f64f64_uncached() -> bool {
        if !Self::is_supported() {
            return false;
        }
        #[cfg(all(target_os = "macos", any(test, feature = "std")))]
        {
            Self::sysctl_arm_feature(c"hw.optional.arm.FEAT_SME_F64F64")
        }
        #[cfg(all(target_os = "linux", any(test, feature = "std")))]
        {
            Self::linux_hwcap2_bit(25)
        }
        #[cfg(not(any(test, feature = "std")))]
        {
            let smfr0: u64;
            // SAFETY: Guarded by `Self::is_supported()` confirming `FEAT_SME`; `ID_AA64SMFR0_EL1`
            // bit [48] (`F64F64`) is readable when `FEAT_SME` is present.
            unsafe {
                core::arch::asm!(
                    ".arch armv8.6-a+sme\nmrs {s}, id_aa64smfr0_el1",
                    s = out(reg) smfr0,
                    options(nomem, nostack, preserves_flags),
                );
            }
            ((smfr0 >> 48) & 1) != 0
        }
        #[cfg(all(
            not(any(target_os = "macos", target_os = "linux")),
            any(test, feature = "std")
        ))]
        {
            false
        }
    }

    #[cfg(all(target_os = "macos", any(test, feature = "std")))]
    fn sysctl_arm_feature(name: &core::ffi::CStr) -> bool {
        type Void = core::ffi::c_void;
        let (mut val, mut size) = (0_i32, core::mem::size_of::<i32>());
        unsafe extern "C" {
            fn sysctlbyname(
                n: *const u8,
                o: *mut Void,
                ol: *mut usize,
                np: *const Void,
                nl: usize,
            ) -> i32;
        }
        // SAFETY: Null-terminated constant sysctl name and valid stack `val`/`size` pointers.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr().cast::<u8>(),
                (&raw mut val).cast::<Void>(),
                &raw mut size,
                core::ptr::null(),
                0,
            )
        };
        rc == 0 && val == 1
    }

    #[cfg(all(target_os = "linux", any(test, feature = "std")))]
    fn linux_hwcap2_bit(bit: u32) -> bool {
        unsafe extern "C" {
            fn getauxval(t: u64) -> u64;
        }
        // SAFETY: `getauxval(AT_HWCAP2 = 26)` is a read-only Linux libc query.
        unsafe { (getauxval(26) & (1_u64 << bit)) != 0 }
    }
}

impl Crc32cBackend for ArmSmeBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        aarch64_crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSmeBackend {
    type Vec4 = NeonVec4;
    const NAME: &'static str = "ARMv9.2-A SME (Scalable Matrix Extension + SVE2/NEON)";
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

    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        if ArmSve2Backend::is_supported() {
            return ArmSve2Backend::f32x4_to_fp16x4(&src.to_array());
        }
        NeonBackend::vec4_to_fp16x4(src)
    }

    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        if ArmSve2Backend::is_supported() {
            return NeonVec4::load(&ArmSve2Backend::fp16x4_to_f32x4(src));
        }
        NeonBackend::fp16x4_to_vec4(src)
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::f32x4_to_fp16x4(src)
        } else {
            NeonBackend::f32x4_to_fp16x4(src)
        }
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::fp16x4_to_f32x4(src)
        } else {
            NeonBackend::fp16x4_to_f32x4(src)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::rsqrt_nr2_f32(v0)
        } else {
            NeonBackend::rsqrt_nr2_f32(v0)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        if ArmSve2Backend::is_supported() {
            ArmSve2Backend::rsqrt_nr2_pair_f32(v0, v1)
        } else {
            NeonBackend::rsqrt_nr2_pair_f32(v0, v1)
        }
    }
}
