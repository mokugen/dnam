//! ARM64 NEON (`float32x4_t`) baseline hardware math backend (`NeonVec4` & `NeonBackend`).

use core::arch::aarch64::{
    __crc32cb, __crc32cd, float32x4_t, uint16x4_t, vabsq_f32, vaddq_f32, vaddq_u32, vaddvq_f32,
    vandq_u32, vcombine_f32, vdup_n_f32, vdup_n_s32, vdupq_n_f32, vdupq_n_u32, vdupq_n_u64,
    veorq_u32, vfmaq_f32, vfmaq_laneq_f32, vfmsq_f32, vget_high_f32, vget_lane_f32, vget_low_f32,
    vgetq_lane_f32, vld1_u16, vmaxnmq_f32, vminnmq_f32, vminq_u32, vmul_f32, vmulq_f32,
    vmulq_laneq_f32, vpadd_f32, vpaddq_f32, vreinterpret_f32_s32, vreinterpret_s32_u32,
    vreinterpret_u32_f32, vreinterpretq_f32_f64, vreinterpretq_f32_u32, vreinterpretq_f64_f32,
    vreinterpretq_u32_f32, vreinterpretq_u32_u64, vrev64q_f32, vset_lane_f32, vsetq_lane_f32,
    vshlq_n_u32, vshr_n_u32, vshrq_n_u32, vsub_f32, vsub_s32, vsubq_f32, vsubq_u32, vtrn1q_f32,
    vtrn1q_f64, vtrn2q_f32, vtrn2q_f64,
};

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

#[inline]
pub(crate) fn aarch64_crc32c_update(crc_init: u32, data: &[u8]) -> u32 {
    if is_crc32_supported() {
        // SAFETY: Runtime feature check confirmed `FEAT_CRC32` support.
        unsafe { arm_crc32c_hw(crc_init, data) }
    } else {
        crc32c_software_slice(crc_init, data)
    }
}

#[inline]
fn is_crc32_supported() -> bool {
    #[cfg(any(test, feature = "std"))]
    {
        std::arch::is_aarch64_feature_detected!("crc")
    }
    #[cfg(not(any(test, feature = "std")))]
    {
        use core::sync::atomic::{AtomicU8, Ordering};
        static CACHE: AtomicU8 = AtomicU8::new(0);
        let c = CACHE.load(Ordering::Relaxed);
        if c != 0 {
            return c == 2;
        }
        let isar0: u64;
        // SAFETY: `ID_AA64ISAR0_EL1` is readable at EL1 (and trapped/emulated by Linux at EL0).
        unsafe {
            core::arch::asm!(
                "mrs {x}, id_aa64isar0_el1",
                x = out(reg) isar0,
                options(nomem, nostack, preserves_flags),
            );
        }
        let ok = ((isar0 >> 16) & 0xF) >= 1;
        CACHE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
        ok
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
        // SAFETY: `aarch64` guarantees baseline NEON `fcvtn` availability; clamp to `[-65504.0, 65504.0]`.
        unsafe {
            let v = src.clamp(NeonVec4::splat(-65504.0), NeonVec4::splat(65504.0));
            let raw_out: uint16x4_t;
            core::arch::asm!(
                "fcvtn {d:v}.4h, {v:v}.4s",
                v = in(vreg) v.0,
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

/// Hardware ARMv8 `FEAT_SHA2` (`SHA256H`, `SHA256H2`, `SHA256SU0`, `SHA256SU1`) block compressor
/// for non-RT container digest verification (`provenance::sha256_digest`).
#[cfg(target_feature = "sha2")]
pub(crate) fn aarch64_sha256_compress(state: &mut [u32; 8], blocks: &[[u8; 64]], k32: &[u32; 64]) {
    use core::arch::aarch64::{
        vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8, vsha256h2q_u32,
        vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
    };
    // SAFETY: Guarded by `#[cfg(target_feature = "sha2")]`; all slice bounds (`8`, `64`, `64`) are fixed at compile time.
    unsafe {
        let mut abcd = vld1q_u32(state.as_ptr());
        let mut efgh = vld1q_u32(state.as_ptr().add(4));

        for block in blocks {
            let abcd_orig = abcd;
            let efgh_orig = efgh;

            let mut s0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr())));
            let mut s1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(16))));
            let mut s2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(32))));
            let mut s3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(48))));

            let mut tmp = vaddq_u32(s0, vld1q_u32(k32.as_ptr()));
            let mut abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            tmp = vaddq_u32(s1, vld1q_u32(k32.as_ptr().add(4)));
            abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            tmp = vaddq_u32(s2, vld1q_u32(k32.as_ptr().add(8)));
            abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            tmp = vaddq_u32(s3, vld1q_u32(k32.as_ptr().add(12)));
            abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            let mut t = 16usize;
            while t < 64 {
                s0 = vsha256su1q_u32(vsha256su0q_u32(s0, s1), s2, s3);
                tmp = vaddq_u32(s0, vld1q_u32(k32.as_ptr().add(t)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

                s1 = vsha256su1q_u32(vsha256su0q_u32(s1, s2), s3, s0);
                tmp = vaddq_u32(s1, vld1q_u32(k32.as_ptr().add(t + 4)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

                s2 = vsha256su1q_u32(vsha256su0q_u32(s2, s3), s0, s1);
                tmp = vaddq_u32(s2, vld1q_u32(k32.as_ptr().add(t + 8)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

                s3 = vsha256su1q_u32(vsha256su0q_u32(s3, s0), s1, s2);
                tmp = vaddq_u32(s3, vld1q_u32(k32.as_ptr().add(t + 12)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);
                t += 16;
            }

            abcd = vaddq_u32(abcd, abcd_orig);
            efgh = vaddq_u32(efgh, efgh_orig);
        }

        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}
