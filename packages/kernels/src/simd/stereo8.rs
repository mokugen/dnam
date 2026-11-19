//! Stereo 8-lane (`4L + 4R`) vector abstraction (`StereoVec8`) and portable interleaved
//! `2 × KernelVec4` implementation (`InterleavedStereo8`) (`ref[arch.simd.two-level-traits]`).
//!
//! Preserves exact `0-ULP` parity between wide `StereoVec8` hardware registers (`AVX2+FMA`, `SVE2`,
//! `SME`) and interleaved `2 × KernelVec4` targets (`NEON`, `WASM SIMD128`, `GpuVec4`) by enforcing
//! identical pairwise reduction trees.

use super::KernelVec4;

/// Portable 8-lane stereo SIMD abstraction (`4` Left lanes + `4` Right lanes).
pub trait StereoVec8: Copy + Clone + core::fmt::Debug + Send + Sync + 'static {
    /// Underlying 4-lane single-channel SIMD register type (`KernelVec4`).
    type Half: KernelVec4;

    /// Constructs a stereo vector from separate Left and Right 4-lane halves.
    fn from_lr(left: Self::Half, right: Self::Half) -> Self;

    /// Splits a stereo vector into `(left, right)` 4-lane halves.
    fn into_lr(self) -> (Self::Half, Self::Half);

    /// Broadcasts `val` across all 8 Left and Right lanes.
    #[inline(always)]
    fn splat(val: f32) -> Self {
        let half = Self::Half::splat(val);
        Self::from_lr(half, half)
    }

    /// Returns an 8-lane stereo vector of zeros (`0.0`).
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Broadcasts `left` across the 4 Left lanes and `right` across the 4 Right lanes.
    #[inline(always)]
    fn splat_lr(left: f32, right: f32) -> Self {
        Self::from_lr(Self::Half::splat(left), Self::Half::splat(right))
    }

    /// Loads Left and Right 4-lane slices into a stereo vector.
    #[inline(always)]
    fn load_lr(left: &[f32; 4], right: &[f32; 4]) -> Self {
        Self::from_lr(Self::Half::load(left), Self::Half::load(right))
    }

    /// Stores Left and Right 4-lane halves into slices.
    #[inline(always)]
    fn store_lr(self, left: &mut [f32; 4], right: &mut [f32; 4]) {
        let (l, r) = self.into_lr();
        l.store(left);
        r.store(right);
    }

    /// Broadcasts a single 4-lane vector `h` to both Left and Right halves.
    #[inline(always)]
    fn broadcast_half(h: Self::Half) -> Self {
        Self::from_lr(h, h)
    }

    /// Lane-wise addition (`self + rhs`).
    fn add(self, rhs: Self) -> Self;

    /// Lane-wise subtraction (`self - rhs`).
    fn sub(self, rhs: Self) -> Self;

    /// Lane-wise multiplication (`self * rhs`).
    fn mul(self, rhs: Self) -> Self;

    /// Lane-wise fused multiply-add (`self + a * b`).
    fn fma(self, a: Self, b: Self) -> Self;

    /// Lane-wise fused multiply-subtract (`self - a * b`).
    fn fms(self, a: Self, b: Self) -> Self;

    /// Lane-wise maximum (`max(self, rhs)`).
    fn max(self, rhs: Self) -> Self;

    /// Lane-wise minimum (`min(self, rhs)`).
    fn min(self, rhs: Self) -> Self;

    /// Branchless lane-wise clamp to `[lo, hi]`.
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Lane-wise absolute value (`|self|`).
    fn abs(self) -> Self;

    /// Lane-wise arithmetic negation (`-self`).
    fn neg(self) -> Self;

    /// Lane-wise branchless select: `if lhs <= rhs { if_true } else { if_false }`.
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self;

    /// Returns `true` iff all 8 Left and Right lanes satisfy `self[i] <= rhs[i]` (`false` on `NaN`).
    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        let (sl, sr) = self.into_lr();
        let (rl, rr) = rhs.into_lr();
        sl.all_le(rl) && sr.all_le(rr)
    }

    /// Flushes subnormals to signed zero across all 8 lanes.
    fn flush_subnormal(self) -> Self;

    /// Scrubs non-finite lanes (`NaN`/`±Inf`) to `+0.0` and flushes subnormals to signed zero
    /// across all 8 lanes.
    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        let (l, r) = self.into_lr();
        Self::from_lr(l.flush_recursive_state(), r.flush_recursive_state())
    }

    /// Clamped positive reciprocal `1.0 / max(self, 1.0e-6)` using seed + 2 NR steps.
    fn rcp_pos_nr2(self) -> Self;

    /// Clamped positive reciprocal `1.0 / max(self, 1.0e-6)` using seed + 3 NR steps.
    fn rcp_pos_nr3(self) -> Self;

    /// Clamped positive inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using seed + 2 NR steps.
    fn rsqrt_pos_nr2(self) -> Self;

    /// Clamped positive inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using seed + 3 NR steps.
    fn rsqrt_pos_nr3(self) -> Self;

    /// Reduces Left and Right halves independently using pairwise `(a0+a1)+(a2+a3)` trees.
    #[inline(always)]
    fn horizontal_sum_lr(self) -> (f32, f32) {
        let (l, r) = self.into_lr();
        l.horizontal_sum_pair(r)
    }

    /// Evaluates shared-matrix stereo `4x4` FMA tile `acc + sum_{c=0..3} x[c] * cols[c]`.
    #[inline(always)]
    fn fma_mat4x4_shared(acc: Self, cols: &[[f32; 4]; 4], x: Self) -> Self {
        let c0 = Self::Half::load(&cols[0]);
        let c1 = Self::Half::load(&cols[1]);
        let c2 = Self::Half::load(&cols[2]);
        let c3 = Self::Half::load(&cols[3]);
        let (acc_l, acc_r) = acc.into_lr();
        let (x_l, x_r) = x.into_lr();
        let p01_l = c0.mul_lane::<0>(x_l).fma_lane::<1>(c1, x_l);
        let p23_l = c2.mul_lane::<2>(x_l).fma_lane::<3>(c3, x_l);
        let p01_r = c0.mul_lane::<0>(x_r).fma_lane::<1>(c1, x_r);
        let p23_r = c2.mul_lane::<2>(x_r).fma_lane::<3>(c3, x_r);
        Self::from_lr(acc_l.add(p01_l.add(p23_l)), acc_r.add(p01_r.add(p23_r)))
    }
}

/// Portable interleaved `2 × KernelVec4` stereo 8-lane vector (`left`, `right`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InterleavedStereo8<V: KernelVec4> {
    /// Left-channel 4-lane vector half.
    pub left: V,
    /// Right-channel 4-lane vector half.
    pub right: V,
}

impl<V: KernelVec4> StereoVec8 for InterleavedStereo8<V> {
    type Half = V;

    #[inline(always)]
    fn from_lr(left: Self::Half, right: Self::Half) -> Self {
        Self { left, right }
    }

    #[inline(always)]
    fn into_lr(self) -> (Self::Half, Self::Half) {
        (self.left, self.right)
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            left: self.left.add(rhs.left),
            right: self.right.add(rhs.right),
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            left: self.left.sub(rhs.left),
            right: self.right.sub(rhs.right),
        }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            left: self.left.mul(rhs.left),
            right: self.right.mul(rhs.right),
        }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self {
            left: self.left.fma(a.left, b.left),
            right: self.right.fma(a.right, b.right),
        }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self {
            left: self.left.fms(a.left, b.left),
            right: self.right.fms(a.right, b.right),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self {
            left: self.left.max(rhs.left),
            right: self.right.max(rhs.right),
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self {
            left: self.left.min(rhs.left),
            right: self.right.min(rhs.right),
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self {
            left: self.left.abs(),
            right: self.right.abs(),
        }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            left: self.left.neg(),
            right: self.right.neg(),
        }
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            left: V::select_le(lhs.left, rhs.left, if_true.left, if_false.left),
            right: V::select_le(lhs.right, rhs.right, if_true.right, if_false.right),
        }
    }

    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        V::all_le(self.left, rhs.left) && V::all_le(self.right, rhs.right)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self {
            left: self.left.flush_subnormal(),
            right: self.right.flush_subnormal(),
        }
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        Self {
            left: V::flush_recursive_state(self.left),
            right: V::flush_recursive_state(self.right),
        }
    }

    #[inline(always)]
    fn rcp_pos_nr2(self) -> Self {
        Self {
            left: self.left.rcp_pos_nr2(),
            right: self.right.rcp_pos_nr2(),
        }
    }

    #[inline(always)]
    fn rcp_pos_nr3(self) -> Self {
        Self {
            left: self.left.rcp_pos_nr3(),
            right: self.right.rcp_pos_nr3(),
        }
    }

    #[inline(always)]
    fn rsqrt_pos_nr2(self) -> Self {
        let (left, right) = V::rsqrt_pos_pair_nr2(self.left, self.right);
        Self { left, right }
    }

    #[inline(always)]
    fn rsqrt_pos_nr3(self) -> Self {
        Self {
            left: self.left.rsqrt_pos_nr3(),
            right: self.right.rsqrt_pos_nr3(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::GpuVec4;
    use proptest::prelude::*;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct DefaultTraitStereo8(GpuVec4, GpuVec4);

    impl StereoVec8 for DefaultTraitStereo8 {
        type Half = GpuVec4;

        fn from_lr(left: Self::Half, right: Self::Half) -> Self {
            Self(left, right)
        }
        fn into_lr(self) -> (Self::Half, Self::Half) {
            (self.0, self.1)
        }
        fn add(self, rhs: Self) -> Self {
            Self(self.0.add(rhs.0), self.1.add(rhs.1))
        }
        fn sub(self, rhs: Self) -> Self {
            Self(self.0.sub(rhs.0), self.1.sub(rhs.1))
        }
        fn mul(self, rhs: Self) -> Self {
            Self(self.0.mul(rhs.0), self.1.mul(rhs.1))
        }
        fn fma(self, a: Self, b: Self) -> Self {
            Self(self.0.fma(a.0, b.0), self.1.fma(a.1, b.1))
        }
        fn fms(self, a: Self, b: Self) -> Self {
            Self(self.0.fms(a.0, b.0), self.1.fms(a.1, b.1))
        }
        fn max(self, rhs: Self) -> Self {
            Self(self.0.max(rhs.0), self.1.max(rhs.1))
        }
        fn min(self, rhs: Self) -> Self {
            Self(self.0.min(rhs.0), self.1.min(rhs.1))
        }
        fn abs(self) -> Self {
            Self(self.0.abs(), self.1.abs())
        }
        fn neg(self) -> Self {
            Self(self.0.neg(), self.1.neg())
        }
        fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
            Self(
                GpuVec4::select_le(lhs.0, rhs.0, if_true.0, if_false.0),
                GpuVec4::select_le(lhs.1, rhs.1, if_true.1, if_false.1),
            )
        }
        fn flush_subnormal(self) -> Self {
            Self(self.0.flush_subnormal(), self.1.flush_subnormal())
        }
        fn rcp_pos_nr2(self) -> Self {
            Self(self.0.rcp_pos_nr2(), self.1.rcp_pos_nr2())
        }
        fn rcp_pos_nr3(self) -> Self {
            Self(self.0.rcp_pos_nr3(), self.1.rcp_pos_nr3())
        }
        fn rsqrt_pos_nr2(self) -> Self {
            Self(self.0.rsqrt_pos_nr2(), self.1.rsqrt_pos_nr2())
        }
        fn rsqrt_pos_nr3(self) -> Self {
            Self(self.0.rsqrt_pos_nr3(), self.1.rsqrt_pos_nr3())
        }
    }

    #[test]
    fn test_interleaved_stereo8_ops_and_mat4x4() {
        type S8 = InterleavedStereo8<GpuVec4>;
        let z = S8::zero();
        assert_eq!(z.into_lr(), (GpuVec4::zero(), GpuVec4::zero()));

        let lr = S8::splat_lr(1.5, -2.5);
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 4];
        lr.store_lr(&mut out_l, &mut out_r);
        assert_eq!(out_l, [1.5; 4]);
        assert_eq!(out_r, [-2.5; 4]);

        let bh = S8::broadcast_half(GpuVec4::from_array([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(bh.horizontal_sum_lr(), (10.0, 10.0));

        let a = S8::load_lr(&[1.0, -2.0, 3.0, -4.0], &[0.5, -1.5, 2.5, -3.5]);
        let b = S8::splat_lr(2.0, 4.0);
        assert_eq!(a.add(b).into_lr().0.to_array(), [3.0, 0.0, 5.0, -2.0]);
        assert_eq!(a.sub(b).into_lr().0.to_array(), [-1.0, -4.0, 1.0, -6.0]);
        assert_eq!(a.mul(b).into_lr().1.to_array(), [2.0, -6.0, 10.0, -14.0]);
        assert_eq!(a.fma(b, b).into_lr().0.to_array(), [5.0, 2.0, 7.0, 0.0]);
        assert_eq!(a.fms(b, b).into_lr().0.to_array(), [-3.0, -6.0, -1.0, -8.0]);
        assert_eq!(a.abs().into_lr().0.to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.neg().into_lr().0.to_array(), [-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(
            a.clamp(S8::splat(-1.0), S8::splat(2.0))
                .into_lr()
                .0
                .to_array(),
            [1.0, -1.0, 2.0, -1.0]
        );
        let sel = S8::select_le(a, S8::zero(), S8::splat(10.0), S8::splat(-10.0));
        assert_eq!(sel.into_lr().0.to_array(), [-10.0, 10.0, -10.0, 10.0]);

        assert!(a.all_le(S8::splat(3.5)));
        assert!(!a.all_le(S8::splat(2.9)));
        assert!(!a.all_le(S8::load_lr(&[4.0; 4], &[4.0, 4.0, 2.0, 4.0])));
        assert!(!S8::splat_lr(f32::NAN, 0.0).all_le(S8::splat(1.0)));

        let sub = S8::splat_lr(f32::from_bits(1), f32::from_bits(0x8000_0001));
        let flushed = sub.flush_subnormal();
        assert_eq!(flushed.into_lr().0.to_array().map(f32::to_bits), [0; 4]);
        assert_eq!(
            flushed.into_lr().1.to_array().map(f32::to_bits),
            [0x8000_0000; 4]
        );

        let dirty = S8::load_lr(
            &[f32::from_bits(1), f32::NAN, 1.5, f32::INFINITY],
            &[f32::from_bits(0x8000_0001), f32::NEG_INFINITY, -2.5, 0.0],
        );
        let scrubbed = dirty.flush_recursive_state();
        assert_eq!(
            scrubbed.into_lr().0.to_array().map(f32::to_bits),
            [0, 0, 1.5_f32.to_bits(), 0]
        );
        assert_eq!(
            scrubbed.into_lr().1.to_array().map(f32::to_bits),
            [0x8000_0000, 0, (-2.5_f32).to_bits(), 0]
        );

        let da = DefaultTraitStereo8::from_lr(a.left, a.right);
        let d_dirty = DefaultTraitStereo8::from_lr(dirty.left, dirty.right);
        assert!(da.all_le(DefaultTraitStereo8::splat(3.5)));
        assert!(!da.all_le(DefaultTraitStereo8::splat(2.9)));
        assert_eq!(
            d_dirty.flush_recursive_state().into_lr(),
            scrubbed.into_lr()
        );

        let pos = S8::load_lr(&[1.0, 2.0, 4.0, 8.0], &[0.5, 4.0, 9.0, 16.0]);
        let rcp = pos.rcp_pos_nr2();
        let rsq = pos.rsqrt_pos_nr2();
        let rcp3 = pos.rcp_pos_nr3();
        let rsq3 = pos.rsqrt_pos_nr3();
        assert!((rcp.into_lr().0.to_array()[2] - 0.25).abs() < 1.0e-5);
        assert!((rsq.into_lr().1.to_array()[2] - (1.0 / 3.0)).abs() < 1.0e-3);
        assert!((rcp3.into_lr().0.to_array()[2] - 0.25).abs() < 1.0e-6);
        assert!((rsq3.into_lr().1.to_array()[2] - (1.0 / 3.0)).abs() < 1.0e-6);

        let cols = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ];
        let x = S8::load_lr(&[1.0, 2.0, 3.0, 4.0], &[-1.0, -2.0, -3.0, -4.0]);
        let mv = S8::fma_mat4x4_shared(S8::splat_lr(0.5, -0.5), &cols, x);
        assert_eq!(mv.into_lr().0.to_array(), [1.5, 4.5, 9.5, 16.5]);
        assert_eq!(mv.into_lr().1.to_array(), [-1.5, -4.5, -9.5, -16.5]);
    }

    proptest! {
        #[test]
        fn prop_interleaved_stereo8_matches_lane_algebra(
            l in prop::array::uniform4(-10.0_f32..10.0_f32),
            r in prop::array::uniform4(-10.0_f32..10.0_f32),
            rl in prop::array::uniform4(-10.0_f32..10.0_f32),
            rr in prop::array::uniform4(-10.0_f32..10.0_f32),
        ) {
            type S8 = InterleavedStereo8<GpuVec4>;
            let v = S8::load_lr(&l, &r);
            let rhs = S8::load_lr(&rl, &rr);
            let dv = DefaultTraitStereo8::from_lr(v.left, v.right);
            let drhs = DefaultTraitStereo8::from_lr(rhs.left, rhs.right);
            let expected_le = (0..4).all(|i| l[i] <= rl[i] && r[i] <= rr[i]);
            prop_assert_eq!(v.all_le(rhs), expected_le);
            prop_assert_eq!(dv.all_le(drhs), expected_le);

            let (sum_l, sum_r) = v.horizontal_sum_lr();
            prop_assert_eq!(sum_l, (l[0] + l[1]) + (l[2] + l[3]));
            prop_assert_eq!(sum_r, (r[0] + r[1]) + (r[2] + r[3]));
            let pos = v.abs().add(S8::splat(0.5));
            let inv = pos.rcp_pos_nr2();
            let (inv_l, inv_r) = inv.into_lr();
            for i in 0..4 {
                let d_l = f64::from(l[i].abs() + 0.5);
                let d_r = f64::from(r[i].abs() + 0.5);
                prop_assert!((f64::from(inv_l.to_array()[i]) * d_l - 1.0).abs() < 1.0e-5);
                prop_assert!((f64::from(inv_r.to_array()[i]) * d_r - 1.0).abs() < 1.0e-5);
            }
        }
    }
}
