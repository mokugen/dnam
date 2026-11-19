//! [`ArmSve2Backend`] detection, `MathBackend` / `Crc32cBackend` implementations, and
//! register-resident SVE2 `T1` tracker exponent scaling.

use core::arch::aarch64::{
    float32x4_t, vandq_u32, vbslq_f32, vceqq_u32, vdupq_n_u32, vreinterpretq_u32_f32,
};
use core::sync::atomic::{AtomicU8, Ordering};

use super::super::neon::aarch64_crc32c_update;
use super::{Sve2Stereo8, Sve2Vec4, from_zreg, to_zreg};
use crate::math::crc32c::Crc32cBackend;
use crate::math::{BackendKind, MathBackend, ScalarBackend, SimdVec4};

/// ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) hardware backend for `aarch64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmSve2Backend;

/// Cached `FEAT_SVE2` support state (`0 = uninitialized`, `1 = unsupported`, `2 = supported`).
static SVE2_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

impl ArmSve2Backend {
    /// Returns the cached `FEAT_SVE2` capability flag with a single relaxed atomic load and zero
    /// OS feature-detection calls (`sysctlbyname`, `getauxval`, or `detect_and_initialize`).
    #[inline(always)]
    pub fn is_supported_cached() -> bool {
        SVE2_SUPPORT_CACHE.load(Ordering::Relaxed) == 2
    }

    /// Queries the OS/CPU once off the hot path when [`SVE2_SUPPORT_CACHE`] is uninitialized.
    #[cold]
    fn detect_uncached() -> bool {
        #[cfg(any(test, feature = "std"))]
        {
            std::arch::is_aarch64_feature_detected!("sve2")
        }
        #[cfg(all(target_os = "linux", not(any(test, feature = "std"))))]
        {
            let (pfr0, zfr0): (u64, u64);
            // SAFETY: `ID_AA64PFR0_EL1` (`SVE` [35:32]) and `ID_AA64ZFR0_EL1` (`S3_0_C0_C4_4`,
            // `SVEver` [3:0]) are emulated at EL0 by Linux. Read `ZFR0` only when `SVE != 0`.
            unsafe {
                core::arch::asm!(
                    "mrs {p}, id_aa64pfr0_el1",
                    p = out(reg) pfr0,
                    options(nomem, nostack, preserves_flags),
                );
                if ((pfr0 >> 32) & 0xF) >= 1 {
                    core::arch::asm!(
                        "mrs {z}, S3_0_C0_C4_4",
                        z = out(reg) zfr0,
                        options(nomem, nostack, preserves_flags),
                    );
                } else {
                    zfr0 = 0;
                }
            }
            (zfr0 & 0xF) >= 1
        }
        #[cfg(all(not(target_os = "linux"), not(any(test, feature = "std"))))]
        {
            cfg!(target_feature = "sve2")
        }
    }

    /// Detects and caches at runtime whether the host `aarch64` CPU supports `FEAT_SVE2`.
    #[inline]
    pub fn is_supported() -> bool {
        let cached = SVE2_SUPPORT_CACHE.load(Ordering::Relaxed);
        if cached != 0 {
            return cached == 2;
        }
        let ok = Self::detect_uncached();
        SVE2_SUPPORT_CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
        ok
    }
}

/// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` in SVE2 `z0..z4`
/// registers when `FEAT_SVE2` is supported at runtime.
#[inline]
pub(crate) fn sve2_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> Option<(f32, f32, f32, f32)> {
    if !ArmSve2Backend::is_supported_cached() {
        return None;
    }
    let max_v = to_zreg([max_l, max_r, 1.0, 1.0]);
    let exp_mask: float32x4_t = zerocopy::transmute!([0x7f80_0000_u32; 4]);
    let cap_253: float32x4_t = zerocopy::transmute!([253_u32 << 23; 4]);
    let two54: float32x4_t = zerocopy::transmute!([254_u32 << 23; 4]);
    let (su_v, sd_v): (float32x4_t, float32x4_t);
    // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
    unsafe {
        core::arch::asm!(
            ".arch armv9-a+sve2",
            "ptrue p0.s, vl4",
            "and z4.d, z0.d, z1.d",
            "umin z4.s, p0/m, z4.s, z2.s",
            "sub z5.s, z3.s, z4.s",
            in("q0") max_v,
            in("q1") exp_mask,
            in("q2") cap_253,
            in("q3") two54,
            out("q4") su_v,
            out("q5") sd_v,
            out("p0") _,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    let (sd, su) = (from_zreg(sd_v), from_zreg(su_v));
    Some((sd[0], su[0], sd[1], su[1]))
}

impl Crc32cBackend for ArmSve2Backend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        aarch64_crc32c_update(crc_init, data)
    }
}

impl MathBackend for ArmSve2Backend {
    type Vec4 = Sve2Vec4;
    type Stereo8 = Sve2Stereo8;
    const NAME: &'static str = "ARMv9-A SVE2 (Scalable Vector Extension 2)";
    const KIND: BackendKind = BackendKind::ArmSve2;

    #[inline]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        if Self::is_supported_cached() {
            let clamped =
                Sve2Vec4::load(src).clamp(Sve2Vec4::splat(-65504.0), Sve2Vec4::splat(65504.0));
            let raw_v = to_zreg(*src);
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2` (`aarch64` NEON baseline).
            let inp = unsafe {
                let raw_bits = vreinterpretq_u32_f32(raw_v);
                let exp_mask = vdupq_n_u32(0x7F80_0000);
                let is_nonfinite = vceqq_u32(vandq_u32(raw_bits, exp_mask), exp_mask);
                vbslq_f32(is_nonfinite, raw_v, to_zreg(clamped.0))
            };
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "fcvt z0.h, p0/m, z0.s",
                    "uzp1 z0.h, z0.h, z0.h",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            let words: [u16; 8] = zerocopy::transmute!(out);
            [words[0], words[1], words[2], words[3]]
        } else {
            ScalarBackend::f32x4_to_fp16x4(src)
        }
    }

    #[inline]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        if Self::is_supported_cached() {
            let raw_in: [u16; 8] = [src[0], src[1], src[2], src[3], 0, 0, 0, 0];
            let inp: float32x4_t = zerocopy::transmute!(raw_in);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "uunpklo z0.s, z0.h",
                    "fcvt z0.s, p0/m, z0.h",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            from_zreg(out)
        } else {
            ScalarBackend::fp16x4_to_f32x4(src)
        }
    }

    #[inline]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        Self::rsqrt_nr2_pair_f32(v0, 1.0).0
    }

    #[inline]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        if Self::is_supported_cached() {
            let v = to_zreg([v0, v1, 1.0, 1.0]);
            let magic: float32x4_t = zerocopy::transmute!([0x5f37_59df_u32; 4]);
            let half = to_zreg([0.5_f32; 4]);
            let three_halves = to_zreg([1.5_f32; 4]);
            let out: float32x4_t;
            // SAFETY: Guarded by `Self::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\n",
                    "fmul z4.s, z0.s, z2.s\n",
                    "lsr z5.s, z0.s, #1\n",
                    "sub z5.s, z1.s, z5.s\n",
                    "fmul z6.s, z4.s, z5.s\n",
                    "fmul z6.s, z6.s, z5.s\n",
                    "fsub z7.s, z3.s, z6.s\n",
                    "fmul z5.s, z5.s, z7.s\n",
                    "fmul z6.s, z4.s, z5.s\n",
                    "fmul z6.s, z6.s, z5.s\n",
                    "fsub z7.s, z3.s, z6.s\n",
                    "fmul z5.s, z5.s, z7.s\n",
                    in("q0") v,
                    in("q1") magic,
                    in("q2") half,
                    in("q3") three_halves,
                    out("q4") _,
                    out("q5") out,
                    out("q6") _,
                    out("q7") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            let arr = from_zreg(out);
            (arr[0], arr[1])
        } else {
            ScalarBackend::rsqrt_nr2_pair_f32(v0, v1)
        }
    }
}
