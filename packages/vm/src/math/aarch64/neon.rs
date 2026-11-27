//! ARM64 128-bit NEON 4-lane `f32` vector register ([`NeonVec4`]) and stereo pair alias
//! ([`NeonStereo8`]).

pub mod backend;

use core::arch::aarch64::{
    float32x4_t, uint32x4_t, vabsq_f32, vaddq_f32, vaddq_u32, vandq_u32, vbslq_u32, vcleq_f32,
    vcltq_u32, vdupq_laneq_f64, vdupq_n_f32, vdupq_n_u32, veorq_u32, vextq_f32, vfmaq_f32,
    vfmaq_laneq_f32, vfmsq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vminvq_u32, vmulq_f32,
    vmulq_laneq_f32, vnegq_f32, vreinterpretq_f32_f64, vreinterpretq_f32_u32,
    vreinterpretq_f64_f32, vreinterpretq_u32_f32, vrev64q_f32, vshlq_n_u32, vshrq_n_u32, vst1q_f32,
    vsubq_f32, vsubq_u32,
};
use core::fmt;

pub use backend::{
    NeonBackend, aarch64_crc32c_update, is_crc32_supported_cached, warm_crc32_support_cache,
};

use crate::math::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_BIAS, IEEE754_F32_EXP_MASK, IEEE754_F32_IMPLICIT_BIT,
    IEEE754_F32_MANTISSA_BITS, IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, IEEE754_F32_SIGN_MASK,
    SimdVec4,
};

const SCRUB_SHIFTED_EXP_STEP: u32 = IEEE754_F32_IMPLICIT_BIT << 1;
const SCRUB_SHIFTED_NORMAL_SPAN: u32 = (IEEE754_F32_EXP_MASK - IEEE754_F32_IMPLICIT_BIT) << 1;

const _: () = assert!(SCRUB_SHIFTED_EXP_STEP == 0x0100_0000);
const _: () = assert!(SCRUB_SHIFTED_NORMAL_SPAN == 0xFE00_0000);

/// ARM64 128-bit NEON 4-lane `f32` vector register (`float32x4_t`).
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonVec4(pub float32x4_t);

/// Interleaved `2 × NeonVec4` 8-lane stereo vector (`4L + 4R`).
pub type NeonStereo8 = dnam_kernels::InterleavedStereo8<NeonVec4>;

// SAFETY: `float32x4_t` is a pure 16-byte SIMD value type with no aliasing or thread affinity.
unsafe impl Send for NeonVec4 {}
// SAFETY: `float32x4_t` has immutable value semantics when shared across threads.
unsafe impl Sync for NeonVec4 {}

impl fmt::Debug for NeonVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NeonVec4").field(&self.to_array()).finish()
    }
}

#[inline(always)]
fn sign_pair_flip_mask() -> uint32x4_t {
    // SAFETY: `vld1q_u32` loads 4 aligned stack `u32` sign-flip mask words.
    unsafe {
        let raw = [IEEE754_F32_SIGN_MASK, 0, IEEE754_F32_SIGN_MASK, 0];
        core::arch::aarch64::vld1q_u32(raw.as_ptr())
    }
}

#[inline(always)]
fn odd_sign_pair_flip_mask() -> uint32x4_t {
    // SAFETY: `vld1q_u32` loads 4 aligned stack `u32` sign-flip mask words for lanes `[1, 3]`.
    unsafe {
        let raw = [0, IEEE754_F32_SIGN_MASK, 0, IEEE754_F32_SIGN_MASK];
        core::arch::aarch64::vld1q_u32(raw.as_ptr())
    }
}

impl SimdVec4 for NeonVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        // SAFETY: Baseline AArch64 always supports AdvSIMD (`neon`).
        Self(unsafe { vdupq_n_f32(val) })
    }

    #[inline(always)]
    fn dup_low_pair(self) -> Self {
        // SAFETY: Lane 0 reinterpret-broadcast duplicates `[v0, v1]` across `[v0, v1, v0, v1]`.
        Self(unsafe { vreinterpretq_f32_f64(vdupq_laneq_f64::<0>(vreinterpretq_f64_f32(self.0))) })
    }

    #[inline(always)]
    fn dup_high_pair(self) -> Self {
        // SAFETY: Lane 1 reinterpret-broadcast duplicates `[v2, v3]` across `[v2, v3, v2, v3]`.
        Self(unsafe { vreinterpretq_f32_f64(vdupq_laneq_f64::<1>(vreinterpretq_f64_f32(self.0))) })
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        // SAFETY: `[f32; 4]` has 4 valid initialized elements (`16 B`).
        Self(unsafe { vld1q_f32(src.as_ptr()) })
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        // SAFETY: `[f32; 4]` provides 4 writable elements (`16 B`).
        unsafe { vst1q_f32(dst.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vaddq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vsubq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vmulq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON fused multiply-add.
        Self(unsafe { vfmaq_f32(self.0, a.0, b.0) })
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        // SAFETY: `LANE < 4` is statically verified above for NEON `vfmaq_laneq_f32`.
        Self(unsafe {
            match LANE {
                0 => vfmaq_laneq_f32::<0>(self.0, a.0, v.0),
                1 => vfmaq_laneq_f32::<1>(self.0, a.0, v.0),
                2 => vfmaq_laneq_f32::<2>(self.0, a.0, v.0),
                3 => vfmaq_laneq_f32::<3>(self.0, a.0, v.0),
                _ => unreachable!(),
            }
        })
    }

    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        // SAFETY: `LANE < 4` is statically verified above for NEON `vmulq_laneq_f32`.
        Self(unsafe {
            match LANE {
                0 => vmulq_laneq_f32::<0>(self.0, v.0),
                1 => vmulq_laneq_f32::<1>(self.0, v.0),
                2 => vmulq_laneq_f32::<2>(self.0, v.0),
                3 => vmulq_laneq_f32::<3>(self.0, v.0),
                _ => unreachable!(),
            }
        })
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON fused multiply-subtract (`self - a * b`).
        Self(unsafe { vfmsq_f32(self.0, a.0, b.0) })
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vmaxq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vminq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vabsq_f32(self.0) })
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        // SAFETY: `vrev64q_f32` swaps `[0,1,2,3] -> [1,0,3,2]` and XOR flips sign bits on `[0,2]`.
        Self(unsafe {
            let swapped = vreinterpretq_u32_f32(vrev64q_f32(self.0));
            vreinterpretq_f32_u32(veorq_u32(swapped, sign_pair_flip_mask()))
        })
    }

    #[inline(always)]
    fn odd_pair_swap_neg(self) -> Self {
        // SAFETY: `vrev64q_f32` + `vextq_f32::<2>` reverses `[0,1,2,3] -> [3,2,1,0]` and XOR flips `[1,3]`.
        Self(unsafe {
            let pair_swapped = vrev64q_f32(self.0);
            let reversed = vreinterpretq_u32_f32(vextq_f32::<2>(pair_swapped, pair_swapped));
            vreinterpretq_f32_u32(veorq_u32(reversed, odd_sign_pair_flip_mask()))
        })
    }

    #[inline(always)]
    fn rotate_left_1(self) -> Self {
        // SAFETY: `vextq_f32::<1>(x, x)` extracts `[x1, x2, x3, x0]` in a single NEON register op.
        Self(unsafe { vextq_f32::<1>(self.0, self.0) })
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        // SAFETY: Bitwise IEEE-754 subnormal mask preserves sign bits and normal/NaN/Inf values.
        Self(unsafe {
            let bits = vreinterpretq_u32_f32(self.0);
            let abs_bits = vandq_u32(bits, vdupq_n_u32(IEEE754_F32_ABS_MASK));
            let is_subnormal = vcltq_u32(abs_bits, vdupq_n_u32(IEEE754_F32_IMPLICIT_BIT));
            let signed_zero = vandq_u32(bits, vdupq_n_u32(IEEE754_F32_SIGN_MASK));
            vreinterpretq_f32_u32(vbslq_u32(is_subnormal, signed_zero, bits))
        })
    }

    #[inline(always)]
    fn scrub_recursive_state(self) -> Self {
        // SAFETY: 3-op / 2-constant range check `(bits << 1) - 0x0100_0000 < 0xFE00_0000` keeps
        // normal finite lanes intact while canonicalizing `-0.0`, subnormals, `±Inf`, and `NaN`
        // to `+0.0` (`0x0000_0000`) with zero `FPCR`/`FPSR` interaction.
        Self(unsafe {
            let bits = vreinterpretq_u32_f32(self.0);
            let biased = vsubq_u32(vshlq_n_u32::<1>(bits), vdupq_n_u32(SCRUB_SHIFTED_EXP_STEP));
            let is_normal = vcltq_u32(biased, vdupq_n_u32(SCRUB_SHIFTED_NORMAL_SPAN));
            vreinterpretq_f32_u32(vandq_u32(bits, is_normal))
        })
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        // SAFETY: `vpaddq_f32` matches pairwise `(a0+a1)+(a2+a3)` tree order identically across ISAs.
        unsafe {
            let p1 = core::arch::aarch64::vpaddq_f32(self.0, self.0);
            let p2 = core::arch::aarch64::vpaddq_f32(p1, p1);
            core::arch::aarch64::vgetq_lane_f32::<0>(p2)
        }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        // SAFETY: Interleaved pairwise reduction computes `(self.hsum(), rhs.hsum())` identically.
        unsafe {
            let p1 = core::arch::aarch64::vpaddq_f32(self.0, rhs.0);
            let p2 = core::arch::aarch64::vpaddq_f32(p1, p1);
            (
                core::arch::aarch64::vgetq_lane_f32::<0>(p2),
                core::arch::aarch64::vgetq_lane_f32::<1>(p2),
            )
        }
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        // SAFETY: Baseline AArch64 NEON integer bit arithmetic.
        Self(unsafe {
            vreinterpretq_f32_u32(vsubq_u32(
                vdupq_n_u32(IEEE754_F32_RCP_SEED),
                vreinterpretq_u32_f32(self.0),
            ))
        })
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        // SAFETY: Baseline AArch64 NEON integer bit arithmetic.
        Self(unsafe {
            vreinterpretq_f32_u32(vsubq_u32(
                vdupq_n_u32(IEEE754_F32_RSQRT_SEED),
                vshrq_n_u32::<1>(vreinterpretq_u32_f32(self.0)),
            ))
        })
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        // SAFETY: Baseline AArch64 NEON integer exponent reconstruction.
        Self(unsafe {
            vreinterpretq_f32_u32(vshlq_n_u32::<{ IEEE754_F32_MANTISSA_BITS as i32 }>(
                vaddq_u32(
                    vreinterpretq_u32_f32(self.0),
                    vdupq_n_u32(IEEE754_F32_EXP_BIAS),
                ),
            ))
        })
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
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
        // SAFETY: Baseline AArch64 NEON intrinsic.
        Self(unsafe { vnegq_f32(self.0) })
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        // SAFETY: Baseline AArch64 NEON `fcle` + `bsl` bit-select.
        Self(unsafe {
            let mask = vcleq_f32(lhs.0, rhs.0);
            core::arch::aarch64::vbslq_f32(mask, if_true.0, if_false.0)
        })
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        // SAFETY: `vcleq_f32` yields `u32::MAX` per lane iff `self <= rhs` (`0` on `NaN`).
        unsafe { vminvq_u32(vcleq_f32(self.0, rhs.0)) == u32::MAX }
    }

    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::pair_swap_neg(self)
    }

    #[inline(always)]
    fn odd_pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::odd_pair_swap_neg(self)
    }

    #[inline(always)]
    fn rotate_left_1(self) -> Self {
        <Self as SimdVec4>::rotate_left_1(self)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        <Self as SimdVec4>::flush_subnormal(self)
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        <Self as SimdVec4>::scrub_recursive_state(self)
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::{ScalarVec4, flush_recursive_state_vec4};

    #[test]
    fn test_neon_vec4_matches_scalar_edge_cases() {
        let dirty = [
            f32::from_bits(1),
            f32::from_bits(0x807f_ffff),
            f32::NAN,
            -0.5,
        ];
        let nf = flush_recursive_state_vec4(NeonVec4::load(&dirty)).to_array();
        let sf = flush_recursive_state_vec4(ScalarVec4::load(&dirty)).to_array();
        for i in 0..4 {
            assert_eq!(nf[i].to_bits(), sf[i].to_bits());
        }
        let edge_zeros = [
            -0.0_f32,
            0.0_f32,
            f32::INFINITY,
            f32::from_bits(0x7F80_0001),
        ];
        let nzf = flush_recursive_state_vec4(NeonVec4::load(&edge_zeros)).to_array();
        let szf = flush_recursive_state_vec4(ScalarVec4::load(&edge_zeros)).to_array();
        for i in 0..4 {
            assert_eq!(nzf[i].to_bits(), 0);
            assert_eq!(nzf[i].to_bits(), szf[i].to_bits());
        }
        let v = NeonVec4::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.dup_low_pair().to_array(), [1.0, 2.0, 1.0, 2.0]);
        assert_eq!(v.dup_high_pair().to_array(), [3.0, 4.0, 3.0, 4.0]);
        assert_eq!(v.pair_swap_neg().to_array(), [-2.0, 1.0, -4.0, 3.0]);
        assert!(<NeonVec4 as dnam_kernels::KernelVec4>::all_le(
            v,
            NeonVec4::splat(4.0)
        ));
        assert!(!<NeonVec4 as dnam_kernels::KernelVec4>::all_le(
            NeonVec4::load(&dirty),
            NeonVec4::splat(100.0)
        ));
        crate::math::backend::simd_vec4::tests::verify_all_simd_and_kernel_ops::<NeonVec4>();
    }

    proptest! {
        #[test]
        fn prop_neon_vec4_ops_match_scalar(
            a in proptest::array::uniform4(-4.0_f32..4.0),
            b in proptest::array::uniform4(-4.0_f32..4.0),
            c in proptest::array::uniform4(0.1_f32..8.0),
        ) {
            let na = NeonVec4::load(&a);
            let nb = NeonVec4::load(&b);
            let nc = NeonVec4::load(&c);
            let sa = ScalarVec4::load(&a);
            let sb = ScalarVec4::load(&b);
            let sc = ScalarVec4::load(&c);
            prop_assert_eq!(na.horizontal_sum().to_bits(), sa.horizontal_sum().to_bits());
            for i in 0..4 {
                prop_assert!((na.fma(nb, nc).to_array()[i] - sa.fma(sb, sc).to_array()[i]).abs() < 1.0e-5);
                prop_assert!((na.fms(nb, nc).to_array()[i] - sa.fms(sb, sc).to_array()[i]).abs() < 1.0e-5);
                prop_assert!((na.exp2_clamped5().to_array()[i] - sa.exp2_clamped5().to_array()[i]).abs() < 1.0e-5);
            }
            let rcp_n = nc.rcp_pos_nr2().to_array();
            let rcp_s = sc.rcp_pos_nr2().to_array();
            for i in 0..4 {
                prop_assert!((rcp_n[i] - rcp_s[i]).abs() <= 1.0e-5 * rcp_s[i].abs().max(1.0));
            }
        }
    }
}
