//! ARM64 NEON (`float32x4_t`) baseline hardware math backend (`NeonVec4` & `NeonBackend`).

use core::arch::aarch64::{
    __crc32cb, __crc32cd, float32x4_t, uint16x4_t, vabsq_f32, vaddq_f32, vaddq_u32, vaddvq_f32,
    vandq_u32, vbicq_u32, vbslq_f32, vceqq_u32, vcleq_f32, vcombine_f32, vdup_n_f32, vdup_n_s32,
    vdupq_n_f32, vdupq_n_u32, vdupq_n_u64, veorq_u32, vfmaq_f32, vfmaq_laneq_f32, vfmsq_f32,
    vget_high_f32, vget_lane_f32, vget_low_f32, vgetq_lane_f32, vld1_u16, vmaxnmq_f32, vminnmq_f32,
    vminq_u32, vminvq_u32, vmul_f32, vmulq_f32, vmulq_laneq_f32, vnegq_f32, vorrq_u32, vpadd_f32,
    vpaddq_f32, vreinterpret_f32_s32, vreinterpret_s32_u32, vreinterpret_u32_f32,
    vreinterpretq_f32_f64, vreinterpretq_f32_u32, vreinterpretq_f64_f32, vreinterpretq_u32_f32,
    vreinterpretq_u32_u64, vrev64q_f32, vset_lane_f32, vsetq_lane_f32, vshlq_n_u32, vshr_n_u32,
    vshrq_n_u32, vsub_f32, vsub_s32, vsubq_f32, vsubq_u32, vtrn1q_f32, vtrn1q_f64, vtrn2q_f32,
    vtrn2q_f64,
};
use core::sync::atomic::{AtomicU8, Ordering};
use dnam_kernels::InterleavedStereo8;

mod sha2;
#[cfg(target_feature = "sha2")]
pub(crate) use sha2::aarch64_sha256_compress;

use crate::math::crc32c::{Crc32cBackend, crc32c_software_slice};
use crate::math::{BackendKind, MathBackend, SimdVec4};

/// 128-bit ARM64 NEON 4-lane `f32` register wrapper (`float32x4_t`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct NeonVec4(pub float32x4_t);

impl SimdVec4 for NeonVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vdupq_n_f32(val)) }
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let pair = vset_lane_f32::<1>(b, vdup_n_f32(a));
            Self(vcombine_f32(pair, pair))
        }
    }

    #[inline(always)]
    fn dup_low_pair(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let pair = vget_low_f32(self.0);
            Self(vcombine_f32(pair, pair))
        }
    }

    #[inline(always)]
    fn dup_high_pair(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let pair = vget_high_f32(self.0);
            Self(vcombine_f32(pair, pair))
        }
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(zerocopy::transmute!(*src))
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let v = vdupq_n_f32(arr[0]);
            let v = vsetq_lane_f32::<1>(arr[1], v);
            let v = vsetq_lane_f32::<2>(arr[2], v);
            Self(vsetq_lane_f32::<3>(arr[3], v))
        }
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = zerocopy::transmute!(self.0);
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            [
                vgetq_lane_f32::<0>(self.0),
                vgetq_lane_f32::<1>(self.0),
                vgetq_lane_f32::<2>(self.0),
                vgetq_lane_f32::<3>(self.0),
            ]
        }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vfmaq_f32(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        // SAFETY: `aarch64` guarantees baseline NEON availability and `LANE < 4`.
        unsafe {
            Self(match LANE {
                0 => vfmaq_laneq_f32::<0>(self.0, a.0, v.0),
                1 => vfmaq_laneq_f32::<1>(self.0, a.0, v.0),
                2 => vfmaq_laneq_f32::<2>(self.0, a.0, v.0),
                _ => vfmaq_laneq_f32::<3>(self.0, a.0, v.0),
            })
        }
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        // SAFETY: `aarch64` guarantees baseline NEON availability and `LANE < 4`.
        unsafe {
            Self(match LANE {
                0 => vmulq_laneq_f32::<0>(self.0, v.0),
                1 => vmulq_laneq_f32::<1>(self.0, v.0),
                2 => vmulq_laneq_f32::<2>(self.0, v.0),
                _ => vmulq_laneq_f32::<3>(self.0, v.0),
            })
        }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vfmsq_f32(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vmaxnmq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vminnmq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let swapped = vrev64q_f32(self.0);
            let mask = vreinterpretq_u32_u64(vdupq_n_u64(0x0000_0000_8000_0000));
            Self(vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(swapped),
                mask,
            )))
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { vaddvq_f32(self.0) }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let p1 = vpaddq_f32(self.0, rhs.0);
            let p2 = vpadd_f32(vget_low_f32(p1), vget_high_f32(p1));
            (vget_lane_f32::<0>(p2), vget_lane_f32::<1>(p2))
        }
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let magic = vdupq_n_u32(crate::math::IEEE754_F32_RCP_SEED);
            let bits = vreinterpretq_u32_f32(self.0);
            Self(vreinterpretq_f32_u32(vsubq_u32(magic, bits)))
        }
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let magic = vdupq_n_u32(crate::math::IEEE754_F32_RSQRT_SEED);
            let bits = vshrq_n_u32::<1>(vreinterpretq_u32_f32(self.0));
            Self(vreinterpretq_f32_u32(vsubq_u32(magic, bits)))
        }
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let biased = vaddq_u32(
                vreinterpretq_u32_f32(self.0),
                vdupq_n_u32(crate::math::IEEE754_F32_EXP_BIAS),
            );
            Self(vreinterpretq_f32_u32(vshlq_n_u32::<23>(biased)))
        }
    }

    #[inline(always)]
    fn rcp_ge1_pair_nr2(self, other: Self) -> (Self, Self) {
        let one = Self::splat(1.0);
        let y0 = self.rcp_pos_seed_bits();
        let y1 = other.rcp_pos_seed_bits();
        let e0 = one.fms(self, y0);
        let e1 = one.fms(other, y1);
        let y0 = y0.fma(y0, e0);
        let y1 = y1.fma(y1, e1);
        let f0 = one.fms(self, y0);
        let f1 = one.fms(other, y1);
        (y0.fma(y0, f0), y1.fma(y1, f1))
    }

    #[inline(always)]
    fn rcp_ge1_quad_nr2(d: [Self; 4]) -> [Self; 4] {
        let one = Self::splat(1.0);
        let y0 = d[0].rcp_pos_seed_bits();
        let y1 = d[1].rcp_pos_seed_bits();
        let y2 = d[2].rcp_pos_seed_bits();
        let y3 = d[3].rcp_pos_seed_bits();
        let e0 = one.fms(d[0], y0);
        let e1 = one.fms(d[1], y1);
        let e2 = one.fms(d[2], y2);
        let e3 = one.fms(d[3], y3);
        let y0 = y0.fma(y0, e0);
        let y1 = y1.fma(y1, e1);
        let y2 = y2.fma(y2, e2);
        let y3 = y3.fma(y3, e3);
        let f0 = one.fms(d[0], y0);
        let f1 = one.fms(d[1], y1);
        let f2 = one.fms(d[2], y2);
        let f3 = one.fms(d[3], y3);
        [
            y0.fma(y0, f0),
            y1.fma(y1, f1),
            y2.fma(y2, f2),
            y3.fma(y3, f3),
        ]
    }

    #[inline(always)]
    fn rsqrt_pos_pair_nr2(a: Self, b: Self) -> (Self, Self) {
        let floor = Self::splat(1.0e-12);
        let half = Self::splat(0.5);
        let three_halves = Self::splat(1.5);
        let da = a.max(floor);
        let db = b.max(floor);
        let hda = da.mul(half);
        let hdb = db.mul(half);
        let mut ya = da.rsqrt_pos_seed_bits();
        let mut yb = db.rsqrt_pos_seed_bits();
        ya = ya.mul(three_halves.fms(hda, ya.mul(ya)));
        yb = yb.mul(three_halves.fms(hdb, yb.mul(yb)));
        ya = ya.mul(three_halves.fms(hda, ya.mul(ya)));
        yb = yb.mul(three_halves.fms(hdb, yb.mul(yb)));
        (ya, yb)
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::flush_recursive_state(self)
    }
}

impl dnam_kernels::KernelVec4 for NeonVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        <Self as SimdVec4>::splat(val)
    }

    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        <Self as SimdVec4>::splat_pair(a, b)
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        <Self as SimdVec4>::from_array(arr)
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        <Self as SimdVec4>::to_array(self)
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
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vnegq_f32(self.0)) }
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vbslq_f32(vcleq_f32(lhs.0, rhs.0), if_true.0, if_false.0)) }
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { vminvq_u32(vcleq_f32(self.0, rhs.0)) != 0 }
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::pair_swap_neg(self)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        <Self as SimdVec4>::flush_subnormal(self)
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let bits = vreinterpretq_u32_f32(self.0);
            let exp_mask = vdupq_n_u32(0x7F80_0000);
            let exp = vandq_u32(bits, exp_mask);
            let is_nonfinite = vceqq_u32(exp, exp_mask);
            let is_sub = vceqq_u32(exp, vdupq_n_u32(0));
            let clear_mask = vorrq_u32(is_nonfinite, vandq_u32(is_sub, vdupq_n_u32(0x007F_FFFF)));
            Self(vreinterpretq_f32_u32(vbicq_u32(bits, clear_mask)))
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        <Self as SimdVec4>::horizontal_sum(self)
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        <Self as SimdVec4>::horizontal_sum_pair(self, rhs)
    }

    #[inline(always)]
    fn dot(self, rhs: Self) -> f32 {
        <Self as SimdVec4>::dot(self, rhs)
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

/// ARM64 NEON hardware math backend targeting Rockchip RK3582 (Cortex-A76/A55) and Apple Silicon.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeonBackend;

#[target_feature(enable = "crc")]
#[inline]
pub(crate) fn arm_crc32c_hw(crc_init: u32, data: &[u8]) -> u32 {
    let mut crc = !crc_init;
    let (chunks, remainder) = data.as_chunks::<8>();
    for chunk in chunks {
        crc = __crc32cd(crc, u64::from_le_bytes(*chunk));
    }
    for &byte in remainder {
        crc = __crc32cb(crc, byte);
    }
    !crc
}

/// Cached `FEAT_CRC32` support state (`0 = uninitialized`, `1 = unsupported`, `2 = supported`).
static CRC32_SUPPORT_CACHE: AtomicU8 = AtomicU8::new(0);

/// Returns the cached `FEAT_CRC32` capability flag with a single relaxed atomic load and zero OS
/// feature-detection calls in hot / real-time CRC paths (`reload_model_in_place`, etc.).
#[inline(always)]
pub(crate) fn is_crc32_supported_cached() -> bool {
    CRC32_SUPPORT_CACHE.load(Ordering::Relaxed) == 2
}

/// Queries `FEAT_CRC32` once off the hot path when [`CRC32_SUPPORT_CACHE`] is uninitialized.
#[cold]
fn detect_crc32_uncached() -> bool {
    #[cfg(any(test, feature = "std"))]
    {
        std::arch::is_aarch64_feature_detected!("crc")
    }
    #[cfg(all(target_os = "linux", not(any(test, feature = "std"))))]
    {
        let isar0: u64;
        // SAFETY: `ID_AA64ISAR0_EL1` is trapped and emulated at EL0 by the Linux kernel.
        unsafe {
            core::arch::asm!(
                "mrs {x}, id_aa64isar0_el1",
                x = out(reg) isar0,
                options(nomem, nostack, preserves_flags),
            );
        }
        ((isar0 >> 16) & 0xF) >= 1
    }
    #[cfg(all(not(target_os = "linux"), not(any(test, feature = "std"))))]
    {
        cfg!(target_feature = "crc")
    }
}

/// Warms [`CRC32_SUPPORT_CACHE`] outside the real-time path and returns whether `FEAT_CRC32` is
/// available on the host `aarch64` CPU.
#[inline]
pub(crate) fn warm_crc32_support_cache() -> bool {
    let cached = CRC32_SUPPORT_CACHE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 2;
    }
    let ok = detect_crc32_uncached();
    CRC32_SUPPORT_CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
    ok
}

#[inline]
pub(crate) fn aarch64_crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
    if is_crc32_supported_cached() {
        // SAFETY: Runtime feature check confirmed `FEAT_CRC32` support.
        unsafe { arm_crc32c_hw(crc_init, data) }
    } else {
        crc32c_software_slice(crc_init, data)
    }
}

/// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` in NEON registers.
#[inline(always)]
pub(crate) fn neon_tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    // SAFETY: `aarch64` guarantees baseline NEON availability.
    unsafe {
        let max_v = NeonVec4::splat_pair(max_l, max_r).0;
        let exp_bits = vandq_u32(vreinterpretq_u32_f32(max_v), vdupq_n_u32(0x7f80_0000));
        let su_v = vminq_u32(exp_bits, vdupq_n_u32(253 << 23));
        let sd_v = vsubq_u32(vdupq_n_u32(254 << 23), su_v);
        let sd_f = vreinterpretq_f32_u32(sd_v);
        let su_f = vreinterpretq_f32_u32(su_v);
        (
            vget_lane_f32::<0>(vget_low_f32(sd_f)),
            vget_lane_f32::<0>(vget_low_f32(su_f)),
            vget_lane_f32::<1>(vget_low_f32(sd_f)),
            vget_lane_f32::<1>(vget_low_f32(su_f)),
        )
    }
}

impl Crc32cBackend for NeonBackend {
    #[inline]
    fn crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
        aarch64_crc32c_update(crc_init, data)
    }
}

impl MathBackend for NeonBackend {
    type Vec4 = NeonVec4;
    type Stereo8 = InterleavedStereo8<NeonVec4>;
    const NAME: &'static str = "ARM64 NEON (128-bit float32x4_t)";
    const KIND: BackendKind = BackendKind::ArmNeon;

    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        // SAFETY: `aarch64` guarantees baseline NEON `trn1`/`trn2` availability.
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

    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        // SAFETY: `aarch64` guarantees baseline NEON `fcvtn`; clamp finite lanes to `[-65504.0, 65504.0]`
        // while keeping `NaN`/`±Inf` uncorrupted via bitwise finite mask blend before conversion.
        unsafe {
            let raw_bits = vreinterpretq_u32_f32(src.0);
            let exp_mask = vdupq_n_u32(0x7F80_0000);
            let is_nonfinite = vceqq_u32(vandq_u32(raw_bits, exp_mask), exp_mask);
            let clamped = src.clamp(NeonVec4::splat(-65504.0), NeonVec4::splat(65504.0));
            let v = vbslq_f32(is_nonfinite, src.0, clamped.0);
            let raw_out: uint16x4_t;
            core::arch::asm!(
                "fcvtn {d:v}.4h, {v:v}.4s",
                v = in(vreg) v,
                d = lateout(vreg) raw_out,
                options(pure, nomem, nostack, preserves_flags),
            );
            zerocopy::transmute!(raw_out)
        }
    }

    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        // SAFETY: `aarch64` guarantees baseline NEON `fcvtl` availability and `src` is valid for 8 bytes.
        unsafe {
            let raw_in = vld1_u16(src.as_ptr());
            let out_v: float32x4_t;
            core::arch::asm!(
                "fcvtl {v:v}.4s, {d:v}.4h",
                d = in(vreg) raw_in,
                v = lateout(vreg) out_v,
                options(pure, nomem, nostack, preserves_flags),
            );
            NeonVec4(out_v)
        }
    }

    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        Self::vec4_to_fp16x4(NeonVec4::load(src))
    }

    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        Self::fp16x4_to_vec4(src).to_array()
    }

    #[inline(always)]
    fn rsqrt_nr2_f32(v0: f32) -> f32 {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let v = vdup_n_f32(v0);
            let magic = vdup_n_s32(0x5f37_59df);
            let half = vdup_n_f32(0.5);
            let three_halves = vdup_n_f32(1.5);
            let hv = vmul_f32(half, v);
            let v_shr = vreinterpret_s32_u32(vshr_n_u32::<1>(vreinterpret_u32_f32(v)));
            let mut est = vreinterpret_f32_s32(vsub_s32(magic, v_shr));
            for _ in 0..2 {
                est = vmul_f32(
                    est,
                    vsub_f32(three_halves, vmul_f32(vmul_f32(hv, est), est)),
                );
            }
            vget_lane_f32::<0>(est)
        }
    }

    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe {
            let v = vset_lane_f32::<1>(v1, vdup_n_f32(v0));
            let magic = vdup_n_s32(0x5f37_59df);
            let half = vdup_n_f32(0.5);
            let three_halves = vdup_n_f32(1.5);
            let hv = vmul_f32(half, v);
            let v_shr = vreinterpret_s32_u32(vshr_n_u32::<1>(vreinterpret_u32_f32(v)));
            let mut est = vreinterpret_f32_s32(vsub_s32(magic, v_shr));
            for _ in 0..2 {
                est = vmul_f32(
                    est,
                    vsub_f32(three_halves, vmul_f32(vmul_f32(hv, est), est)),
                );
            }
            (vget_lane_f32::<0>(est), vget_lane_f32::<1>(est))
        }
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{InterleavedStereo8, StereoVec8};
    use proptest::prelude::*;

    use super::super::{NeonF64Vec2, NeonStereo8};
    use super::*;
    use crate::math::trainer::{ScalarF64Vec2, SimdF64Vec2};
    use crate::math::{ScalarBackend, ScalarVec4};

    #[test]
    fn test_neon_vec4_missed_lanes_and_rcp_ge1() {
        let a_arr = [1.5_f32, -2.5, 3.5, -4.5];
        let b_arr = [2.0_f32, -4.0, 0.5, 8.0];
        let va = <NeonVec4 as SimdVec4>::from_array(a_arr);
        let vb = <NeonVec4 as SimdVec4>::load(&b_arr);

        assert_eq!(
            <NeonVec4 as SimdVec4>::dup_low_pair(va).to_array(),
            [1.5, -2.5, 1.5, -2.5]
        );
        assert_eq!(
            <NeonVec4 as SimdVec4>::dup_high_pair(va).to_array(),
            [3.5, -4.5, 3.5, -4.5]
        );
        assert_eq!(
            <NeonVec4 as SimdVec4>::mul_lane::<0>(va, vb).to_array(),
            [3.0, -5.0, 7.0, -9.0]
        );
        assert_eq!(
            <NeonVec4 as SimdVec4>::mul_lane::<1>(va, vb).to_array(),
            [-6.0, 10.0, -14.0, 18.0]
        );
        assert_eq!(
            <NeonVec4 as SimdVec4>::mul_lane::<2>(va, vb).to_array(),
            [0.75, -1.25, 1.75, -2.25]
        );
        assert_eq!(
            <NeonVec4 as SimdVec4>::mul_lane::<3>(va, vb).to_array(),
            [12.0, -20.0, 28.0, -36.0]
        );

        let d = [
            <NeonVec4 as SimdVec4>::from_array([1.0, 2.0, 4.0, 8.0]),
            <NeonVec4 as SimdVec4>::from_array([1.5, 3.0, 6.0, 10.0]),
            <NeonVec4 as SimdVec4>::from_array([1.25, 2.5, 5.0, 12.5]),
            <NeonVec4 as SimdVec4>::from_array([1.75, 3.5, 7.0, 16.0]),
        ];
        let (p0, p1) = <NeonVec4 as SimdVec4>::rcp_ge1_pair_nr2(d[0], d[1]);
        let quad = <NeonVec4 as SimdVec4>::rcp_ge1_quad_nr2(d);
        for idx in 0..4 {
            let v0 = d[0].to_array()[idx];
            let v1 = d[1].to_array()[idx];
            assert!((p0.to_array()[idx] * v0 - 1.0).abs() < 1.0e-5);
            assert!((p1.to_array()[idx] * v1 - 1.0).abs() < 1.0e-5);
            for k in 0..4 {
                let vk = d[k].to_array()[idx];
                assert!((quad[k].to_array()[idx] * vk - 1.0).abs() < 1.0e-5);
            }
        }

        let ka = <NeonVec4 as dnam_kernels::KernelVec4>::load(&a_arr);
        let kb = <NeonVec4 as dnam_kernels::KernelVec4>::load(&b_arr);
        let mut stored = [0.0_f32; 4];
        <NeonVec4 as dnam_kernels::KernelVec4>::store(ka, &mut stored);
        assert_eq!(stored, a_arr);
        assert_eq!(
            <NeonVec4 as dnam_kernels::KernelVec4>::splat_pair(1.25, -3.75).to_array(),
            [1.25, -3.75, 1.25, -3.75]
        );
        assert_eq!(
            <NeonVec4 as dnam_kernels::KernelVec4>::pair_swap_neg(ka).to_array(),
            [2.5, 1.5, 4.5, 3.5]
        );
        assert_eq!(
            <NeonVec4 as dnam_kernels::KernelVec4>::horizontal_sum(ka),
            -2.0
        );
        assert_eq!(<NeonVec4 as dnam_kernels::KernelVec4>::dot(ka, kb), -21.25);
        assert_eq!(
            <NeonVec4 as dnam_kernels::KernelVec4>::mul_lane::<1>(ka, kb).to_array(),
            [-6.0, 10.0, -14.0, 18.0]
        );
        assert_eq!(
            <NeonVec4 as dnam_kernels::KernelVec4>::mul_lane::<3>(ka, kb).to_array(),
            [12.0, -20.0, 28.0, -36.0]
        );
    }

    #[test]
    fn test_neon_backend_helpers_stereo8_and_f64_vec2() {
        let _ = warm_crc32_support_cache();
        let _ = is_crc32_supported_cached();
        let bytes: [u8; 19] = core::array::from_fn(|i| (i as u8).wrapping_mul(17));
        for len in [0, 1, 7, 8, 9, 19] {
            assert_eq!(
                aarch64_crc32c_update(0x1234_5678, &bytes[..len]),
                crc32c_software_slice(0x1234_5678, &bytes[..len])
            );
        }

        let (sd_l, su_l, sd_r, su_r) = neon_tracker_exp_scales_f32(0.25, 4.0);
        assert_eq!((sd_l * su_l, sd_r * su_r), (1.0, 1.0));
        let (sd_z, su_z, sd_hi, su_hi) = neon_tracker_exp_scales_f32(0.0, 1.0e35);
        assert_eq!(su_z, 0.0);
        assert!(sd_z > 0.0 && (sd_hi * su_hi - 1.0).abs() < 1.0e-6);

        let edge = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 100_000.0];
        assert_eq!(
            NeonBackend::f32x4_to_fp16x4(&edge),
            ScalarBackend::f32x4_to_fp16x4(&edge)
        );
        let fp16_words = NeonBackend::vec4_to_fp16x4(<NeonVec4 as SimdVec4>::from_array([
            0.5, -1.5, 2.25, -0.0,
        ]));
        assert_eq!(
            NeonBackend::fp16x4_to_vec4(&fp16_words).to_array(),
            [0.5, -1.5, 2.25, -0.0]
        );
        assert_eq!(NeonBackend::KIND, BackendKind::ArmNeon);

        let s8_a = NeonStereo8::load_lr(&[1.0, 2.0, 3.0, 4.0], &[-1.0, -2.0, -3.0, -4.0]);
        let s8_b = InterleavedStereo8::<ScalarVec4>::load_lr(
            &[1.0, 2.0, 3.0, 4.0],
            &[-1.0, -2.0, -3.0, -4.0],
        );
        assert_eq!(s8_a.horizontal_sum_lr(), s8_b.horizontal_sum_lr());

        let f64_a = NeonF64Vec2::from_f32_pair(1.5, -2.5);
        let f64_b = NeonF64Vec2::splat(2.0);
        let sf64_a = ScalarF64Vec2::from_f32_pair(1.5, -2.5);
        let sf64_b = ScalarF64Vec2::splat(2.0);
        assert_eq!(
            f64_a.mul_add_ieee(f64_b, f64_b).to_array(),
            sf64_a.mul_add_ieee(sf64_b, sf64_b).to_array()
        );
        assert_eq!(f64_a.dot(f64_b), sf64_a.dot(sf64_b));
    }

    proptest! {
        #[test]
        fn prop_neon_vec4_pairs_lanes_and_rcp_ge1(
            a in prop::array::uniform4(-100.0_f32..100.0_f32),
            b in prop::array::uniform4(1.0_f32..1000.0_f32),
        ) {
            let na = <NeonVec4 as SimdVec4>::from_array(a);
            let nb = <NeonVec4 as SimdVec4>::from_array(b);
            let sa = ScalarVec4::from_array(a);
            let sb = ScalarVec4::from_array(b);
            prop_assert_eq!(
                <NeonVec4 as SimdVec4>::dup_low_pair(na).to_array(),
                [a[0], a[1], a[0], a[1]]
            );
            prop_assert_eq!(
                <NeonVec4 as SimdVec4>::dup_high_pair(na).to_array(),
                [a[2], a[3], a[2], a[3]]
            );
            prop_assert_eq!(
                <NeonVec4 as SimdVec4>::mul_lane::<1>(na, nb).to_array(),
                <ScalarVec4 as SimdVec4>::mul_lane::<1>(sa, sb).to_array()
            );
            prop_assert_eq!(
                <NeonVec4 as SimdVec4>::mul_lane::<3>(na, nb).to_array(),
                <ScalarVec4 as SimdVec4>::mul_lane::<3>(sa, sb).to_array()
            );
            prop_assert_eq!(
                <NeonVec4 as dnam_kernels::KernelVec4>::pair_swap_neg(na).to_array(),
                [-a[1], a[0], -a[3], a[2]]
            );
            let (rp0, rp1) = <NeonVec4 as SimdVec4>::rcp_ge1_pair_nr2(nb, nb);
            let r0 = rp0.to_array();
            let r1 = rp1.to_array();
            for i in 0..4 {
                prop_assert!((f64::from(r0[i]) * f64::from(b[i]) - 1.0).abs() < 1.0e-5);
                prop_assert!((f64::from(r1[i]) * f64::from(b[i]) - 1.0).abs() < 1.0e-5);
            }
        }
    }
}
