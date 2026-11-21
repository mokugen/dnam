//! 128-bit 4-lane `f32` SIMD vector abstraction ([`SimdVec4`]) bridged directly to
//! [`dnam_kernels::KernelVec4`].

use crate::math::flush_subnormal_f32_bits;

/// Level 1 Hardware Abstraction: 128-bit 4-lane `f32` SIMD register (`float32x4_t` / `__m128` / `v128`).
pub trait SimdVec4: Copy + Clone + Send + Sync + 'static {
    /// Broadcasts `val` across all 4 lanes.
    fn splat(val: f32) -> Self;

    /// Broadcasts the stereo pair `(a, b)` into `[a, b, a, b]`.
    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        Self::from_array([a, b, a, b])
    }

    /// Duplicate lanes `[0, 1]` across `[0, 1, 2, 3]`.
    #[inline(always)]
    fn dup_low_pair(self) -> Self {
        let a = self.to_array();
        Self::from_array([a[0], a[1], a[0], a[1]])
    }

    /// Duplicate lanes `[2, 3]` across `[0, 1, 2, 3]`.
    #[inline(always)]
    fn dup_high_pair(self) -> Self {
        let a = self.to_array();
        Self::from_array([a[2], a[3], a[2], a[3]])
    }

    /// Loads 4 `f32` lanes from a fixed array.
    fn load(src: &[f32; 4]) -> Self;
    /// Stores 4 `f32` lanes into a fixed array.
    fn store(self, dst: &mut [f32; 4]);
    /// Lane-wise addition (`self + rhs`).
    fn add(self, rhs: Self) -> Self;
    /// Lane-wise subtraction (`self - rhs`).
    fn sub(self, rhs: Self) -> Self;
    /// Lane-wise multiplication (`self * rhs`).
    fn mul(self, rhs: Self) -> Self;
    /// Fused multiply-add (`self + a * b`).
    fn fma(self, a: Self, b: Self) -> Self;

    /// Fused multiply-add with indexed scalar lane broadcast (`self + a * v[LANE]`, `LANE < 4`).
    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        self.fma(a, Self::splat(v.to_array()[LANE]))
    }

    /// Multiplies `self` by an indexed scalar lane broadcast (`self * v[LANE]`, `LANE < 4`).
    #[inline(always)]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        self.mul(Self::splat(v.to_array()[LANE]))
    }

    /// Fused multiply-subtract (`self - a * b`).
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

    /// Swaps adjacent stereo pairs and negates the even lanes (`[-v1, v0, -v3, v2]`).
    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        let a = self.to_array();
        Self::from_array([-a[1], a[0], -a[3], a[2]])
    }

    /// Flushes subnormal lanes to signed zero without altering other bit patterns.
    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self::from_array(self.to_array().map(flush_subnormal_f32_bits))
    }

    /// Scrubs non-finite lanes (`NaN`/`±Inf`) to `+0.0` and flushes subnormals to signed zero.
    fn scrub_recursive_state(self) -> Self;

    /// Horizontal sum across all 4 lanes (`lane0 + lane1 + lane2 + lane3`).
    fn horizontal_sum(self) -> f32;

    /// Simultaneously reduces two 4-lane vectors into `(self.horizontal_sum(), rhs.horizontal_sum())`.
    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        (self.horizontal_sum(), rhs.horizontal_sum())
    }

    /// 4-lane dot product (`sum(self * rhs)`).
    #[inline(always)]
    fn dot(self, rhs: Self) -> f32 {
        self.mul(rhs).horizontal_sum()
    }

    /// Computes IEEE-754 integer-sub reciprocal seed (`0x7EF3_11C3_u32 - bits(self)`) in registers.
    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self::from_array(
            self.to_array().map(|x| {
                f32::from_bits(crate::math::IEEE754_F32_RCP_SEED.wrapping_sub(x.to_bits()))
            }),
        )
    }

    /// Computes IEEE-754 integer-sub inverse-sqrt seed (`IEEE754_F32_RSQRT_SEED - (bits(self) >> 1)`).
    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self::from_array(self.to_array().map(|x| {
            f32::from_bits(crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(x.to_bits() >> 1))
        }))
    }

    /// Converts `k + 12_582_912.0` (`1.5 * 2^23` biased integer bits) into exact `2^k`.
    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self::from_array(self.to_array().map(|x| {
            f32::from_bits(
                x.to_bits().wrapping_add(crate::math::IEEE754_F32_EXP_BIAS)
                    << crate::math::IEEE754_F32_MANTISSA_BITS,
            )
        }))
    }

    /// Fast base-2 exponential `2^clamp(self, -5.0, 5.0)`.
    fn exp2_clamped5(self) -> Self;

    /// Fast positive reciprocal `1.0 / max(self, 1.0)` using 2 Newton–Raphson steps.
    #[inline(always)]
    fn rcp_ge1_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        y1.fma(y1, one.fms(d_safe, y1))
    }

    /// Evaluates two independent `rcp_ge1_nr2` vectors simultaneously.
    #[inline(always)]
    fn rcp_ge1_pair_nr2(self, other: Self) -> (Self, Self) {
        (self.rcp_ge1_nr2(), other.rcp_ge1_nr2())
    }

    /// Evaluates four independent `rcp_ge1_nr2` vectors simultaneously.
    #[inline(always)]
    fn rcp_ge1_quad_nr2(d: [Self; 4]) -> [Self; 4] {
        let one = Self::splat(1.0);
        let d_safe0 = d[0].max(one);
        let d_safe1 = d[1].max(one);
        let d_safe2 = d[2].max(one);
        let d_safe3 = d[3].max(one);
        let mut y0 = d_safe0.rcp_pos_seed_bits();
        let mut y1 = d_safe1.rcp_pos_seed_bits();
        let mut y2 = d_safe2.rcp_pos_seed_bits();
        let mut y3 = d_safe3.rcp_pos_seed_bits();
        y0 = y0.fma(y0, one.fms(d_safe0, y0));
        y1 = y1.fma(y1, one.fms(d_safe1, y1));
        y2 = y2.fma(y2, one.fms(d_safe2, y2));
        y3 = y3.fma(y3, one.fms(d_safe3, y3));
        [
            y0.fma(y0, one.fms(d_safe0, y0)),
            y1.fma(y1, one.fms(d_safe1, y1)),
            y2.fma(y2, one.fms(d_safe2, y2)),
            y3.fma(y3, one.fms(d_safe3, y3)),
        ]
    }

    /// Positive reciprocal `1.0 / max(self, 1.0e-6)` using 2 Newton–Raphson steps.
    #[inline(always)]
    fn rcp_pos_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        y1.fma(y1, one.fms(d_safe, y1))
    }

    /// Positive reciprocal `1.0 / max(self, 1.0e-6)` using 3 Newton–Raphson steps.
    #[inline(always)]
    fn rcp_pos_nr3(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        let y2 = y1.fma(y1, one.fms(d_safe, y1));
        y2.fma(y2, one.fms(d_safe, y2))
    }

    /// Inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using 2 Newton–Raphson steps.
    #[inline(always)]
    fn rsqrt_pos_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-12));
        let half_d = d_safe.mul(Self::splat(0.5));
        let three_halves = Self::splat(1.5);
        let mut y = d_safe.rsqrt_pos_seed_bits();
        y = y.mul(three_halves.fms(half_d, y.mul(y)));
        y.mul(three_halves.fms(half_d, y.mul(y)))
    }

    /// Computes `(rsqrt_pos_nr2(a), rsqrt_pos_nr2(b))` for dual-stream SIMD interleaving.
    #[inline(always)]
    fn rsqrt_pos_pair_nr2(a: Self, b: Self) -> (Self, Self) {
        let eps = Self::splat(1.0e-12);
        let half = Self::splat(0.5);
        let three_halves = Self::splat(1.5);
        let da = a.max(eps);
        let db = b.max(eps);
        let ha = da.mul(half);
        let hb = db.mul(half);
        let mut ya = da.rsqrt_pos_seed_bits();
        let mut yb = db.rsqrt_pos_seed_bits();
        ya = ya.mul(three_halves.fms(ha, ya.mul(ya)));
        yb = yb.mul(three_halves.fms(hb, yb.mul(yb)));
        (
            ya.mul(three_halves.fms(ha, ya.mul(ya))),
            yb.mul(three_halves.fms(hb, yb.mul(yb))),
        )
    }

    /// Inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using 3 Newton–Raphson steps.
    #[inline(always)]
    fn rsqrt_pos_nr3(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-12));
        let half_d = d_safe.mul(Self::splat(0.5));
        let three_halves = Self::splat(1.5);
        let mut y = d_safe.rsqrt_pos_seed_bits();
        y = y.mul(three_halves.fms(half_d, y.mul(y)));
        y = y.mul(three_halves.fms(half_d, y.mul(y)));
        y.mul(three_halves.fms(half_d, y.mul(y)))
    }

    /// Returns a 4-lane vector of zeros (`0.0`).
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Constructs a 4-lane vector from a `[f32; 4]` value.
    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        Self::load(&arr)
    }

    /// Extracts all 4 `f32` lanes as `[f32; 4]`.
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut out = [0.0; 4];
        self.store(&mut out);
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::math::ScalarVec4;
    use dnam_kernels::KernelVec4;
    use proptest::prelude::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct DefaultLaneVec([f32; 4]);

    impl SimdVec4 for DefaultLaneVec {
        fn splat(val: f32) -> Self {
            Self([val; 4])
        }
        fn load(src: &[f32; 4]) -> Self {
            Self(*src)
        }
        fn store(self, dst: &mut [f32; 4]) {
            *dst = self.0;
        }
        fn add(self, rhs: Self) -> Self {
            Self([
                self.0[0] + rhs.0[0],
                self.0[1] + rhs.0[1],
                self.0[2] + rhs.0[2],
                self.0[3] + rhs.0[3],
            ])
        }
        fn sub(self, rhs: Self) -> Self {
            Self([
                self.0[0] - rhs.0[0],
                self.0[1] - rhs.0[1],
                self.0[2] - rhs.0[2],
                self.0[3] - rhs.0[3],
            ])
        }
        fn mul(self, rhs: Self) -> Self {
            Self([
                self.0[0] * rhs.0[0],
                self.0[1] * rhs.0[1],
                self.0[2] * rhs.0[2],
                self.0[3] * rhs.0[3],
            ])
        }
        fn fma(self, a: Self, b: Self) -> Self {
            Self([
                self.0[0] + a.0[0] * b.0[0],
                self.0[1] + a.0[1] * b.0[1],
                self.0[2] + a.0[2] * b.0[2],
                self.0[3] + a.0[3] * b.0[3],
            ])
        }
        fn fms(self, a: Self, b: Self) -> Self {
            Self([
                self.0[0] - a.0[0] * b.0[0],
                self.0[1] - a.0[1] * b.0[1],
                self.0[2] - a.0[2] * b.0[2],
                self.0[3] - a.0[3] * b.0[3],
            ])
        }
        fn max(self, rhs: Self) -> Self {
            Self([
                self.0[0].max(rhs.0[0]),
                self.0[1].max(rhs.0[1]),
                self.0[2].max(rhs.0[2]),
                self.0[3].max(rhs.0[3]),
            ])
        }
        fn min(self, rhs: Self) -> Self {
            Self([
                self.0[0].min(rhs.0[0]),
                self.0[1].min(rhs.0[1]),
                self.0[2].min(rhs.0[2]),
                self.0[3].min(rhs.0[3]),
            ])
        }
        fn abs(self) -> Self {
            Self([
                self.0[0].abs(),
                self.0[1].abs(),
                self.0[2].abs(),
                self.0[3].abs(),
            ])
        }
        fn scrub_recursive_state(self) -> Self {
            Self(self.0.map(dnam_kernels::scrub_and_flush_f32_bits))
        }
        fn horizontal_sum(self) -> f32 {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
        fn exp2_clamped5(self) -> Self {
            Self(<ScalarVec4 as KernelVec4>::exp2_clamped5(ScalarVec4(self.0)).0)
        }
    }

    pub(crate) fn verify_all_simd_vec4_ops<V: SimdVec4>() {
        let a = V::from_array([1.5, -2.5, 3.5, -4.5]);
        let b = V::from_array([2.0, -4.0, 0.5, 8.0]);
        let c = V::from_array([1.0, 1.0, 1.0, 1.0]);
        let mut stored = [0.0; 4];
        V::zero().store(&mut stored);
        assert_eq!(stored, [0.0; 4]);
        assert_eq!(V::splat_pair(1.0, -2.0).to_array(), [1.0, -2.0, 1.0, -2.0]);
        assert_eq!(a.dup_low_pair().to_array(), [1.5, -2.5, 1.5, -2.5]);
        assert_eq!(a.dup_high_pair().to_array(), [3.5, -4.5, 3.5, -4.5]);
        assert_eq!(a.add(b).to_array(), [3.5, -6.5, 4.0, 3.5]);
        assert_eq!(a.sub(b).to_array(), [-0.5, 1.5, 3.0, -12.5]);
        assert_eq!(a.mul(b).to_array(), [3.0, 10.0, 1.75, -36.0]);
        assert_eq!(c.fma(a, b).to_array(), [4.0, 11.0, 2.75, -35.0]);
        assert_eq!(c.fms(a, b).to_array(), [-2.0, -9.0, -0.75, 37.0]);
        assert_eq!(a.mul_lane::<0>(b).to_array(), [3.0, -5.0, 7.0, -9.0]);
        assert_eq!(a.mul_lane::<1>(b).to_array(), [-6.0, 10.0, -14.0, 18.0]);
        assert_eq!(a.mul_lane::<2>(b).to_array(), [0.75, -1.25, 1.75, -2.25]);
        assert_eq!(a.mul_lane::<3>(b).to_array(), [12.0, -20.0, 28.0, -36.0]);
        assert_eq!(c.fma_lane::<0>(a, b).to_array(), [4.0, -4.0, 8.0, -8.0]);
        assert_eq!(c.fma_lane::<1>(a, b).to_array(), [-5.0, 11.0, -13.0, 19.0]);
        assert_eq!(c.fma_lane::<2>(a, b).to_array(), [1.75, -0.25, 2.75, -1.25]);
        assert_eq!(c.fma_lane::<3>(a, b).to_array(), [13.0, -19.0, 29.0, -35.0]);
        assert_eq!(a.max(b).to_array(), [2.0, -2.5, 3.5, 8.0]);
        assert_eq!(a.min(b).to_array(), [1.5, -4.0, 0.5, -4.5]);
        assert_eq!(
            a.clamp(V::splat(-3.0), V::splat(2.5)).to_array(),
            [1.5, -2.5, 2.5, -3.0]
        );
        assert_eq!(a.abs().to_array(), [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(a.pair_swap_neg().to_array(), [2.5, 1.5, 4.5, 3.5]);
        assert_eq!(a.horizontal_sum_pair(b), (-2.0, 6.5));
        assert_eq!(a.dot(b), -21.25);

        let dirty = V::from_array([f32::from_bits(1), f32::NAN, f32::INFINITY, -2.5]);
        assert_eq!(dirty.flush_subnormal().to_array()[0].to_bits(), 0);
        assert_eq!(
            dirty.scrub_recursive_state().to_array(),
            [0.0, 0.0, 0.0, -2.5]
        );

        let d0 = V::from_array([0.25, 1.0, 2.0, 4.0]);
        let d1 = V::from_array([0.5, 1.5, 5.0, 8.0]);
        let seed_r = d0.rcp_pos_seed_bits().to_array();
        let seed_s = d0.rsqrt_pos_seed_bits().to_array();
        assert!(seed_r[0].is_finite() && seed_s[0].is_finite());
        let (p0, p1) = d0.rcp_ge1_pair_nr2(d1);
        let q = V::rcp_ge1_quad_nr2([d0, d1, d0, d1]);
        assert!((p0.to_array()[3] - 0.25).abs() < 1.0e-5);
        assert!((p1.to_array()[3] - 0.125).abs() < 1.0e-5);
        assert_eq!(
            q[0].to_array().map(f32::to_bits),
            p0.to_array().map(f32::to_bits)
        );
        for lane in 0..4 {
            assert!(
                (d0.rcp_pos_nr2().to_array()[lane] - [4.0, 1.0, 0.5, 0.25][lane]).abs() < 1.0e-4
            );
            assert!(
                (d0.rcp_pos_nr3().to_array()[lane] - [4.0, 1.0, 0.5, 0.25][lane]).abs() < 1.0e-5
            );
            assert!(
                (d0.rsqrt_pos_nr2().to_array()[lane] - [2.0, 1.0, 0.70710677, 0.5][lane]).abs()
                    < 1.0e-4
            );
            assert!(
                (d0.rsqrt_pos_nr3().to_array()[lane] - [2.0, 1.0, 0.70710677, 0.5][lane]).abs()
                    < 1.0e-5
            );
        }
        let (rs0, rs1) = V::rsqrt_pos_pair_nr2(d0, d1);
        assert!((rs0.to_array()[0] - 2.0).abs() < 1.0e-4);
        assert!((rs1.to_array()[0] - core::f32::consts::SQRT_2).abs() < 1.0e-4);
        assert_eq!(V::zero().pow2_from_biased_int_bits().to_array(), [1.0; 4]);
        let exp2_v = V::from_array([-1.0, 0.0, 1.0, 2.0])
            .exp2_clamped5()
            .to_array();
        assert!((exp2_v[0] - 0.5).abs() < 1.0e-5 && (exp2_v[3] - 4.0).abs() < 1.0e-4);
    }

    pub(crate) fn verify_all_kernel_vec4_ops<V: KernelVec4>() {
        let a = <V as KernelVec4>::from_array([1.5, -2.5, 3.5, -4.5]);
        let b = <V as KernelVec4>::from_array([2.0, -4.0, 0.5, 8.0]);
        let c = <V as KernelVec4>::splat(1.0);
        let mut stored = [0.0; 4];
        <V as KernelVec4>::splat_pair(1.0, -2.0).store(&mut stored);
        assert_eq!(stored, [1.0, -2.0, 1.0, -2.0]);
        assert_eq!(
            <V as KernelVec4>::add(a, b).to_array(),
            [3.5, -6.5, 4.0, 3.5]
        );
        assert_eq!(
            <V as KernelVec4>::sub(a, b).to_array(),
            [-0.5, 1.5, 3.0, -12.5]
        );
        assert_eq!(
            <V as KernelVec4>::mul(a, b).to_array(),
            [3.0, 10.0, 1.75, -36.0]
        );
        assert_eq!(
            <V as KernelVec4>::fma(c, a, b).to_array(),
            [4.0, 11.0, 2.75, -35.0]
        );
        assert_eq!(
            <V as KernelVec4>::fms(c, a, b).to_array(),
            [-2.0, -9.0, -0.75, 37.0]
        );
        assert_eq!(
            <V as KernelVec4>::mul_lane::<0>(a, b).to_array(),
            [3.0, -5.0, 7.0, -9.0]
        );
        assert_eq!(
            <V as KernelVec4>::mul_lane::<1>(a, b).to_array(),
            [-6.0, 10.0, -14.0, 18.0]
        );
        assert_eq!(
            <V as KernelVec4>::mul_lane::<2>(a, b).to_array(),
            [0.75, -1.25, 1.75, -2.25]
        );
        assert_eq!(
            <V as KernelVec4>::mul_lane::<3>(a, b).to_array(),
            [12.0, -20.0, 28.0, -36.0]
        );
        assert_eq!(
            <V as KernelVec4>::fma_lane::<0>(c, a, b).to_array(),
            [4.0, -4.0, 8.0, -8.0]
        );
        assert_eq!(
            <V as KernelVec4>::fma_lane::<1>(c, a, b).to_array(),
            [-5.0, 11.0, -13.0, 19.0]
        );
        assert_eq!(
            <V as KernelVec4>::fma_lane::<2>(c, a, b).to_array(),
            [1.75, -0.25, 2.75, -1.25]
        );
        assert_eq!(
            <V as KernelVec4>::fma_lane::<3>(c, a, b).to_array(),
            [13.0, -19.0, 29.0, -35.0]
        );
        assert_eq!(
            <V as KernelVec4>::max(a, b).to_array(),
            [2.0, -2.5, 3.5, 8.0]
        );
        assert_eq!(
            <V as KernelVec4>::min(a, b).to_array(),
            [1.5, -4.0, 0.5, -4.5]
        );
        assert_eq!(<V as KernelVec4>::abs(a).to_array(), [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(<V as KernelVec4>::neg(a).to_array(), [-1.5, 2.5, -3.5, 4.5]);
        assert_eq!(
            <V as KernelVec4>::select_le(a, b, a, b).to_array(),
            [1.5, -4.0, 0.5, -4.5]
        );
        assert!(<V as KernelVec4>::all_le(a, <V as KernelVec4>::splat(4.0)));
        assert_eq!(
            <V as KernelVec4>::pair_swap_neg(a).to_array(),
            [2.5, 1.5, 4.5, 3.5]
        );
        let dirty = <V as KernelVec4>::load(&[f32::from_bits(1), f32::NAN, f32::INFINITY, -2.5]);
        assert_eq!(
            <V as KernelVec4>::flush_subnormal(dirty).to_array()[0].to_bits(),
            0
        );
        assert_eq!(
            <V as KernelVec4>::flush_recursive_state(dirty).to_array(),
            [0.0, 0.0, 0.0, -2.5]
        );
        assert_eq!(<V as KernelVec4>::horizontal_sum_pair(a, b), (-2.0, 6.5));
        assert_eq!(<V as KernelVec4>::dot(a, b), -21.25);
        assert!(<V as KernelVec4>::rcp_pos_seed_bits(c).to_array()[0].is_finite());
        assert!(<V as KernelVec4>::rsqrt_pos_seed_bits(c).to_array()[0].is_finite());
        assert_eq!(
            <V as KernelVec4>::pow2_from_biased_int_bits(<V as KernelVec4>::splat(0.0)).to_array(),
            [1.0; 4]
        );
        let exp2_v = <V as KernelVec4>::exp2_clamped5(c).to_array();
        assert!((exp2_v[0] - 2.0).abs() < 1.0e-5);
    }

    pub(crate) fn verify_all_simd_and_kernel_ops<V: SimdVec4 + KernelVec4>() {
        verify_all_simd_vec4_ops::<V>();
        verify_all_kernel_vec4_ops::<V>();
    }

    #[test]
    fn test_simd_vec4_default_methods() {
        verify_all_simd_vec4_ops::<DefaultLaneVec>();
    }

    proptest! {
        #[test]
        fn prop_rcp_and_rsqrt_default_nr_accuracy(
            x in prop::array::uniform4(1.0e-4_f32..1.0e4_f32),
        ) {
            let v = DefaultLaneVec::from_array(x);
            let r2 = v.rcp_pos_nr2().to_array();
            let r3 = v.rcp_pos_nr3().to_array();
            let s2 = v.rsqrt_pos_nr2().to_array();
            let s3 = v.rsqrt_pos_nr3().to_array();
            for i in 0..4 {
                let d = f64::from(x[i]);
                prop_assert!((f64::from(r2[i]) * d - 1.0).abs() < 1.0e-5);
                prop_assert!((f64::from(r3[i]) * d - 1.0).abs() < 1.0e-6);
                let ex = 1.0 / d.sqrt();
                prop_assert!(((f64::from(s2[i]) - ex) / ex).abs() < 1.0e-5);
                prop_assert!(((f64::from(s3[i]) - ex) / ex).abs() < 2.0e-6);
            }
        }
    }
}
