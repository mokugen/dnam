//! Wide 256-bit `AVX2 + FMA` stereo (`4L + 4R`) vector implementation ([`Avx2Stereo8`])
//! wrapping `__m256` (`32 B`, `align(32)`) and implementing [`dnam_kernels::StereoVec8`] with
//! exact `0-ULP` parity against `InterleavedStereo8<Avx2FmaVec4>` (`ref[arch.simd.two-level-traits]`).

mod hw;

use core::arch::x86_64::__m256;
use core::fmt;
use core::mem::{MaybeUninit, align_of, size_of};

use dnam_kernels::{InterleavedStereo8, KernelVec4, StereoVec8};

use super::{Avx2FmaVec4, is_avx2_fma_hot};
use crate::math::SimdVec4;
use crate::math::x86::sse2::Sse2Vec4;
use hw::{
    abs_hw, add_hw, all_le_hw, broadcast_half_hw, flush_recursive_state_hw, flush_subnormal_hw,
    fma_hw, fma_mat4x4_shared_hw, fms_hw, from_lr_hw, horizontal_sum_lr_hw, into_lr_hw, load_lr_hw,
    max_hw, min_hw, mul_hw, neg_hw, rcp_ge1_nr2_hw, rcp_pos_nr2_hw, rcp_pos_nr3_hw,
    rotate_left_1_hw, rsqrt_pos_nr2_hw, rsqrt_pos_nr3_hw, select_le_hw, splat_hw, splat_lr_hw,
    store_lr_hw, sub_hw, zero_hw,
};

/// 256-bit aligned 8-lane stereo vector (`lanes 0..3 = Left`, `4..7 = Right`) backed directly by
/// one hardware `__m256` register (`32 B`, `align(32)`), preserving `0-ULP` parity with
/// `InterleavedStereo8<Avx2FmaVec4>`.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2Stereo8(pub __m256);

const _: () = assert!(size_of::<Avx2Stereo8>() == 32);
const _: () = assert!(align_of::<Avx2Stereo8>() == 32);
const _: () = assert!(size_of::<Avx2Stereo8>() == size_of::<[Sse2Vec4; 2]>());

impl Avx2Stereo8 {
    #[inline(always)]
    const fn from_sse2_pair(pair: [Sse2Vec4; 2]) -> Self {
        // SAFETY: `Avx2Stereo8(__m256)` and `[Sse2Vec4; 2]` (`[__m128; 2]`) are 32-byte POD SIMD
        // lane tuples with identical lane layout (`lanes 0..3 = Left`, `4..7 = Right`).
        unsafe { core::mem::transmute(pair) }
    }

    #[inline(always)]
    const fn into_sse2_pair(self) -> [Sse2Vec4; 2] {
        // SAFETY: `Avx2Stereo8(__m256)` and `[Sse2Vec4; 2]` (`[__m128; 2]`) are 32-byte POD SIMD
        // lane tuples with identical lane layout (`lanes 0..3 = Left`, `4..7 = Right`).
        unsafe { core::mem::transmute(self) }
    }

    #[inline(always)]
    const fn into_interleaved_sse2(self) -> InterleavedStereo8<Sse2Vec4> {
        let [left, right] = self.into_sse2_pair();
        InterleavedStereo8 { left, right }
    }

    #[inline(always)]
    const fn from_interleaved_sse2(v: InterleavedStereo8<Sse2Vec4>) -> Self {
        Self::from_sse2_pair([v.left, v.right])
    }
}

// SAFETY: `__m256` is a pure 32-byte SIMD value register with no aliasing or thread affinity.
unsafe impl Send for Avx2Stereo8 {}
// SAFETY: `__m256` has immutable value semantics when shared across threads.
unsafe impl Sync for Avx2Stereo8 {}

impl fmt::Debug for Avx2Stereo8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (l, r) = self.into_lr();
        f.debug_struct("Avx2Stereo8")
            .field("left", &l.to_array())
            .field("right", &r.to_array())
            .finish()
    }
}

impl StereoVec8 for Avx2Stereo8 {
    type Half = Avx2FmaVec4;

    #[inline(always)]
    fn from_lr(left: Self::Half, right: Self::Half) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                from_lr_hw(left.0.0, right.0.0, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_sse2_pair([left.0, right.0])
        }
    }

    #[inline(always)]
    fn into_lr(self) -> (Self::Half, Self::Half) {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            let (l, r) = unsafe { into_lr_hw(&self) };
            (Avx2FmaVec4(Sse2Vec4(l)), Avx2FmaVec4(Sse2Vec4(r)))
        } else {
            let [l, r] = self.into_sse2_pair();
            (Avx2FmaVec4(l), Avx2FmaVec4(r))
        }
    }

    #[inline(always)]
    fn splat(val: f32) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                splat_hw(val, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(InterleavedStereo8::<Sse2Vec4>::splat(val))
        }
    }

    #[inline(always)]
    fn zero() -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                zero_hw(&mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(InterleavedStereo8::<Sse2Vec4>::zero())
        }
    }

    #[inline(always)]
    fn splat_lr(left: f32, right: f32) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                splat_lr_hw(left, right, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(InterleavedStereo8::<Sse2Vec4>::splat_lr(left, right))
        }
    }

    #[inline(always)]
    fn load_lr(left: &[f32; 4], right: &[f32; 4]) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                load_lr_hw(left, right, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(InterleavedStereo8::<Sse2Vec4>::load_lr(left, right))
        }
    }

    #[inline(always)]
    fn store_lr(self, left: &mut [f32; 4], right: &mut [f32; 4]) {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe { store_lr_hw(&self, left, right) }
        } else {
            self.into_interleaved_sse2().store_lr(left, right);
        }
    }

    #[inline(always)]
    fn rotate_left_1(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                rotate_left_1_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().rotate_left_1())
        }
    }

    #[inline(always)]
    fn broadcast_half(h: Self::Half) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                broadcast_half_hw(h.0.0, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_sse2_pair([h.0, h.0])
        }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                add_hw(&self, &rhs, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .add(rhs.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                sub_hw(&self, &rhs, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .sub(rhs.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                mul_hw(&self, &rhs, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .mul(rhs.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                fma_hw(&self, &a, &b, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .fma(a.into_interleaved_sse2(), b.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                fms_hw(&self, &a, &b, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .fms(a.into_interleaved_sse2(), b.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                max_hw(&self, &rhs, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .max(rhs.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                min_hw(&self, &rhs, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(
                self.into_interleaved_sse2()
                    .min(rhs.into_interleaved_sse2()),
            )
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                abs_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().abs())
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                neg_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().neg())
        }
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                select_le_hw(&lhs, &rhs, &if_true, &if_false, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(InterleavedStereo8::<Sse2Vec4>::select_le(
                lhs.into_interleaved_sse2(),
                rhs.into_interleaved_sse2(),
                if_true.into_interleaved_sse2(),
                if_false.into_interleaved_sse2(),
            ))
        }
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe { all_le_hw(&self, &rhs) }
        } else {
            self.into_interleaved_sse2()
                .all_le(rhs.into_interleaved_sse2())
        }
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                flush_subnormal_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().flush_subnormal())
        }
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                flush_recursive_state_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().flush_recursive_state())
        }
    }

    #[inline(always)]
    fn rcp_ge1_nr2(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                rcp_ge1_nr2_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().rcp_ge1_nr2())
        }
    }

    #[inline(always)]
    fn rcp_pos_nr2(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                rcp_pos_nr2_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().rcp_pos_nr2())
        }
    }

    #[inline(always)]
    fn rcp_pos_nr3(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                rcp_pos_nr3_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().rcp_pos_nr3())
        }
    }

    #[inline(always)]
    fn rsqrt_pos_nr2(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                rsqrt_pos_nr2_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().rsqrt_pos_nr2())
        }
    }

    #[inline(always)]
    fn rsqrt_pos_nr3(self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                rsqrt_pos_nr3_hw(&self, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(self.into_interleaved_sse2().rsqrt_pos_nr3())
        }
    }

    #[inline(always)]
    fn horizontal_sum_lr(self) -> (f32, f32) {
        if is_avx2_fma_hot() {
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe { horizontal_sum_lr_hw(&self) }
        } else {
            self.into_interleaved_sse2().horizontal_sum_lr()
        }
    }

    #[inline(always)]
    fn fma_mat4x4_shared(acc: Self, cols: &[[f32; 4]; 4], x: Self) -> Self {
        if is_avx2_fma_hot() {
            let mut out = MaybeUninit::<Self>::uninit();
            // SAFETY: `is_avx2_fma_hot()` verified pre-warmed `AVX2 + FMA` support.
            unsafe {
                fma_mat4x4_shared_hw(&acc, cols, &x, &mut out);
                out.assume_init()
            }
        } else {
            Self::from_interleaved_sse2(InterleavedStereo8::<Sse2Vec4>::fma_mat4x4_shared(
                acc.into_interleaved_sse2(),
                cols,
                x.into_interleaved_sse2(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use dnam_kernels::InterleavedStereo8;

    fn assert_s8_eq(got: Avx2Stereo8, exp: InterleavedStereo8<Avx2FmaVec4>) {
        let (gl, gr) = got.into_lr();
        let (el, er) = exp.into_lr();
        assert_eq!(
            gl.to_array().map(f32::to_bits),
            el.to_array().map(f32::to_bits)
        );
        assert_eq!(
            gr.to_array().map(f32::to_bits),
            er.to_array().map(f32::to_bits)
        );
    }

    #[test]
    fn test_avx2_stereo8_lr_roundtrip() {
        crate::math::x86::warm_x86_feature_caches();
        let l = [1.0, 2.0, 3.0, 4.0];
        let r = [-1.0, -2.0, -3.0, -4.0];
        let s = Avx2Stereo8::load_lr(&l, &r);
        assert_eq!(s.horizontal_sum_lr(), (10.0, -10.0));
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 4];
        s.abs().store_lr(&mut out_l, &mut out_r);
        assert_eq!(out_l, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out_r, [1.0, 2.0, 3.0, 4.0]);

        let pos = Avx2Stereo8::splat_lr(4.0, 9.0);
        pos.rcp_pos_nr2().store_lr(&mut out_l, &mut out_r);
        assert!((out_l[0] - 0.25).abs() < 1.0e-5 && (out_r[0] - (1.0 / 9.0)).abs() < 1.0e-5);
        pos.rcp_ge1_nr2().store_lr(&mut out_l, &mut out_r);
        assert!((out_l[0] - 0.25).abs() < 1.0e-5 && (out_r[0] - (1.0 / 9.0)).abs() < 1.0e-5);
        s.rotate_left_1().store_lr(&mut out_l, &mut out_r);
        assert_eq!(out_l, [2.0, 3.0, 4.0, 1.0]);
        assert_eq!(out_r, [-2.0, -3.0, -4.0, -1.0]);
        pos.rsqrt_pos_nr3().store_lr(&mut out_l, &mut out_r);
        assert!((out_l[0] - 0.5).abs() < 1.0e-6 && (out_r[0] - (1.0 / 3.0)).abs() < 1.0e-6);

        // Edge cases: all_le with NaN, flush_subnormal, flush_recursive_state, rcp_ge1_nr2 (< 1.0, ±0.0, negative, NaN)
        assert!(!Avx2Stereo8::splat_lr(f32::NAN, 0.0).all_le(Avx2Stereo8::splat(1.0)));
        assert!(!Avx2Stereo8::splat(0.0).all_le(Avx2Stereo8::splat_lr(1.0, f32::NAN)));
        let dirty_l = [f32::from_bits(1), f32::NAN, 1.5, f32::INFINITY];
        let dirty_r = [f32::from_bits(0x807F_FFFF), f32::NEG_INFINITY, -2.5, -0.0];
        assert_s8_eq(
            Avx2Stereo8::load_lr(&dirty_l, &dirty_r).flush_subnormal(),
            InterleavedStereo8::<Avx2FmaVec4>::load_lr(&dirty_l, &dirty_r).flush_subnormal(),
        );
        assert_s8_eq(
            Avx2Stereo8::load_lr(&dirty_l, &dirty_r).flush_recursive_state(),
            InterleavedStereo8::<Avx2FmaVec4>::load_lr(&dirty_l, &dirty_r).flush_recursive_state(),
        );

        let rcp_edge_l = [-5.0, -0.0, 0.0, 0.5];
        let rcp_edge_r = [0.9999999, 1.0, f32::NAN, f32::INFINITY];
        assert_s8_eq(
            Avx2Stereo8::load_lr(&rcp_edge_l, &rcp_edge_r).rcp_ge1_nr2(),
            InterleavedStereo8::<Avx2FmaVec4>::load_lr(&rcp_edge_l, &rcp_edge_r).rcp_ge1_nr2(),
        );
        for val in [-10.0, -0.0, 0.0, 0.25, 0.999, 1.0, f32::NAN] {
            assert_s8_eq(
                Avx2Stereo8::splat(val).rcp_ge1_nr2(),
                InterleavedStereo8::<Avx2FmaVec4>::splat(val).rcp_ge1_nr2(),
            );
        }
    }

    proptest! {
        #[test]
        fn prop_avx2_stereo8_matches_interleaved_zero_ulp(
            al in prop::array::uniform4(-16.0_f32..16.0_f32),
            ar in prop::array::uniform4(-16.0_f32..16.0_f32),
            bl in prop::array::uniform4(-16.0_f32..16.0_f32),
            br in prop::array::uniform4(-16.0_f32..16.0_f32),
            cl in prop::array::uniform4(-16.0_f32..16.0_f32),
            cr in prop::array::uniform4(-16.0_f32..16.0_f32),
            cols in prop::array::uniform4(prop::array::uniform4(-2.0_f32..2.0_f32)),
        ) {
            crate::math::x86::warm_x86_feature_caches();
            type Ref8 = InterleavedStereo8<Avx2FmaVec4>;
            let va = Avx2Stereo8::load_lr(&al, &ar);
            let vb = Avx2Stereo8::load_lr(&bl, &br);
            let vc = Avx2Stereo8::load_lr(&cl, &cr);
            let ra = Ref8::load_lr(&al, &ar);
            let rb = Ref8::load_lr(&bl, &br);
            let rc = Ref8::load_lr(&cl, &cr);

            assert_s8_eq(va.add(vb), ra.add(rb));
            assert_s8_eq(va.sub(vb), ra.sub(rb));
            assert_s8_eq(va.mul(vb), ra.mul(rb));
            assert_s8_eq(va.fma(vb, vc), ra.fma(rb, rc));
            assert_s8_eq(va.fms(vb, vc), ra.fms(rb, rc));
            assert_s8_eq(va.max(vb), ra.max(rb));
            assert_s8_eq(va.min(vb), ra.min(rb));
            assert_s8_eq(va.abs(), ra.abs());
            assert_s8_eq(va.neg(), ra.neg());
            assert_s8_eq(va.rotate_left_1(), ra.rotate_left_1());
            assert_s8_eq(Avx2Stereo8::select_le(va, vb, vc, va), Ref8::select_le(ra, rb, rc, ra));
            prop_assert_eq!(va.all_le(vb), ra.all_le(rb));
            prop_assert_eq!(va.horizontal_sum_lr(), ra.horizontal_sum_lr());
            assert_s8_eq(
                Avx2Stereo8::fma_mat4x4_shared(va, &cols, vb),
                Ref8::fma_mat4x4_shared(ra, &cols, rb),
            );

            assert_s8_eq(va.rcp_ge1_nr2(), ra.rcp_ge1_nr2());
            let pos_v = va.abs().add(Avx2Stereo8::splat(1.0));
            let pos_r = ra.abs().add(Ref8::splat(1.0));
            assert_s8_eq(pos_v.rcp_ge1_nr2(), pos_r.rcp_ge1_nr2());
            assert_s8_eq(pos_v.rcp_pos_nr2(), pos_r.rcp_pos_nr2());
            assert_s8_eq(pos_v.rcp_pos_nr3(), pos_r.rcp_pos_nr3());
            assert_s8_eq(pos_v.rsqrt_pos_nr2(), pos_r.rsqrt_pos_nr2());
            assert_s8_eq(pos_v.rsqrt_pos_nr3(), pos_r.rsqrt_pos_nr3());
        }
    }
}
