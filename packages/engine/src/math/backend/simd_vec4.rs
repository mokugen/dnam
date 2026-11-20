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

    /// Loads 4 `f32` lanes from a slice.
    fn load(src: &[f32; 4]) -> Self;
    /// Stores 4 `f32` lanes into a slice.
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

    /// Scrubs non-finite lanes (`NaN`/`±Inf`) to `+0.0` and flushes subnormals to signed zero,
    /// delegated by every backend to [`dnam_kernels::KernelVec4::flush_recursive_state`].
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

    /// Computes IEEE-754 integer-sub inverse-sqrt seed (`IEEE754_F32_RSQRT_SEED - (bits(self) >> 1)`) in registers.
    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self::from_array(self.to_array().map(|x| {
            f32::from_bits(crate::math::IEEE754_F32_RSQRT_SEED.wrapping_sub(x.to_bits() >> 1))
        }))
    }

    /// Converts `k + 12_582_912.0` (`1.5 * 2^23` biased integer bits) into exact `2^k` (`k in [-126, 127]`).
    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self::from_array(self.to_array().map(|x| {
            f32::from_bits(x.to_bits().wrapping_add(crate::math::IEEE754_F32_EXP_BIAS) << 23)
        }))
    }

    /// Fast base-2 exponential `2^clamp(self, -5.0, 5.0)` (`< 2.5e-7`, `NaN -> -5.0`),
    /// delegated by every backend to [`dnam_kernels::KernelVec4::exp2_clamped5`].
    fn exp2_clamped5(self) -> Self;

    /// Fast positive reciprocal `1.0 / self` using 2 FMA/FMS Newton-Raphson steps when `self >= 1.0`.
    #[inline(always)]
    fn rcp_ge1_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        y1.fma(y1, one.fms(d_safe, y1))
    }

    /// Evaluates two independent `rcp_ge1_nr2` vectors simultaneously to hide FMA latency.
    #[inline(always)]
    fn rcp_ge1_pair_nr2(self, other: Self) -> (Self, Self) {
        (self.rcp_ge1_nr2(), other.rcp_ge1_nr2())
    }

    /// Evaluates four independent `rcp_ge1_nr2` vectors simultaneously across all SIMD pipes.
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

    /// Positive reciprocal `1.0 / max(self, 1.0e-6)` using 2 FMA/FMS Newton-Raphson steps (`< 6.6e-6`).
    #[inline(always)]
    fn rcp_pos_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        y1.fma(y1, one.fms(d_safe, y1))
    }

    /// Positive reciprocal `1.0 / max(self, 1.0e-6)` using 3 FMA/FMS Newton-Raphson steps (`< 1 ULP`).
    #[inline(always)]
    fn rcp_pos_nr3(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        let y2 = y1.fma(y1, one.fms(d_safe, y1));
        y2.fma(y2, one.fms(d_safe, y2))
    }

    /// Inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using 2 Newton-Raphson FMA/FMS steps.
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

    /// Inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using 3 Newton-Raphson FMA/FMS steps.
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

    /// Converts this 4-lane vector into a `[f32; 4]` array.
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut dst = [0.0; 4];
        self.store(&mut dst);
        dst
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::math::ScalarVec4;

    #[derive(Clone, Copy, Debug)]
    struct DefaultLaneVec([f32; 4]);

    impl SimdVec4 for DefaultLaneVec {
        fn splat(v: f32) -> Self {
            Self([v; 4])
        }
        fn load(s: &[f32; 4]) -> Self {
            Self(*s)
        }
        fn store(self, d: &mut [f32; 4]) {
            *d = self.0;
        }
        fn add(self, r: Self) -> Self {
            Self([
                self.0[0] + r.0[0],
                self.0[1] + r.0[1],
                self.0[2] + r.0[2],
                self.0[3] + r.0[3],
            ])
        }
        fn sub(self, r: Self) -> Self {
            Self([
                self.0[0] - r.0[0],
                self.0[1] - r.0[1],
                self.0[2] - r.0[2],
                self.0[3] - r.0[3],
            ])
        }
        fn mul(self, r: Self) -> Self {
            Self([
                self.0[0] * r.0[0],
                self.0[1] * r.0[1],
                self.0[2] * r.0[2],
                self.0[3] * r.0[3],
            ])
        }
        fn fma(self, a: Self, b: Self) -> Self {
            self.add(a.mul(b))
        }
        fn fms(self, a: Self, b: Self) -> Self {
            self.sub(a.mul(b))
        }
        fn max(self, r: Self) -> Self {
            Self([
                self.0[0].max(r.0[0]),
                self.0[1].max(r.0[1]),
                self.0[2].max(r.0[2]),
                self.0[3].max(r.0[3]),
            ])
        }
        fn min(self, r: Self) -> Self {
            Self([
                self.0[0].min(r.0[0]),
                self.0[1].min(r.0[1]),
                self.0[2].min(r.0[2]),
                self.0[3].min(r.0[3]),
            ])
        }
        fn abs(self) -> Self {
            Self(self.0.map(f32::abs))
        }
        fn pair_swap_neg(self) -> Self {
            Self([-self.0[1], self.0[0], -self.0[3], self.0[2]])
        }
        fn horizontal_sum(self) -> f32 {
            self.0.iter().sum()
        }
        fn exp2_clamped5(self) -> Self {
            <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
        }
        fn scrub_recursive_state(self) -> Self {
            <Self as dnam_kernels::KernelVec4>::flush_recursive_state(self)
        }
    }

    impl dnam_kernels::KernelVec4 for DefaultLaneVec {
        fn splat(val: f32) -> Self {
            <Self as SimdVec4>::splat(val)
        }
        fn splat_pair(a: f32, b: f32) -> Self {
            <Self as SimdVec4>::splat_pair(a, b)
        }
        fn from_array(arr: [f32; 4]) -> Self {
            <Self as SimdVec4>::from_array(arr)
        }
        fn to_array(self) -> [f32; 4] {
            <Self as SimdVec4>::to_array(self)
        }
        fn load(src: &[f32; 4]) -> Self {
            <Self as SimdVec4>::load(src)
        }
        fn store(self, dst: &mut [f32; 4]) {
            <Self as SimdVec4>::store(self, dst);
        }
        fn add(self, rhs: Self) -> Self {
            <Self as SimdVec4>::add(self, rhs)
        }
        fn sub(self, rhs: Self) -> Self {
            <Self as SimdVec4>::sub(self, rhs)
        }
        fn mul(self, rhs: Self) -> Self {
            <Self as SimdVec4>::mul(self, rhs)
        }
        fn fma(self, a: Self, b: Self) -> Self {
            <Self as SimdVec4>::fma(self, a, b)
        }
        fn fms(self, a: Self, b: Self) -> Self {
            <Self as SimdVec4>::fms(self, a, b)
        }
        fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
            <Self as SimdVec4>::fma_lane::<LANE>(self, a, v)
        }
        fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
            <Self as SimdVec4>::mul_lane::<LANE>(self, v)
        }
        fn max(self, rhs: Self) -> Self {
            <Self as SimdVec4>::max(self, rhs)
        }
        fn min(self, rhs: Self) -> Self {
            <Self as SimdVec4>::min(self, rhs)
        }
        fn abs(self) -> Self {
            <Self as SimdVec4>::abs(self)
        }
        fn pair_swap_neg(self) -> Self {
            <Self as SimdVec4>::pair_swap_neg(self)
        }
        fn flush_subnormal(self) -> Self {
            <Self as SimdVec4>::flush_subnormal(self)
        }
        fn horizontal_sum(self) -> f32 {
            <Self as SimdVec4>::horizontal_sum(self)
        }
        fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
            <Self as SimdVec4>::horizontal_sum_pair(self, rhs)
        }
        fn dot(self, rhs: Self) -> f32 {
            <Self as SimdVec4>::dot(self, rhs)
        }
        fn rcp_pos_seed_bits(self) -> Self {
            <Self as SimdVec4>::rcp_pos_seed_bits(self)
        }
        fn rsqrt_pos_seed_bits(self) -> Self {
            <Self as SimdVec4>::rsqrt_pos_seed_bits(self)
        }
        fn pow2_from_biased_int_bits(self) -> Self {
            <Self as SimdVec4>::pow2_from_biased_int_bits(self)
        }
    }

    #[test]
    fn test_simd_vec4_defaults_and_nan_sanitization() {
        let z = ScalarVec4::zero();
        assert_eq!(z.to_array(), [0.0; 4]);

        let arr = [1.0, 2.0, 3.0, 4.0];
        let v = ScalarVec4::from_array(arr);
        assert_eq!(v.dot(v), 30.0);
        assert_eq!(
            v.clamp(ScalarVec4::splat(1.5), ScalarVec4::splat(3.5))
                .to_array(),
            [1.5, 2.0, 3.0, 3.5]
        );

        let dl = DefaultLaneVec([1.0, 2.0, 3.0, 4.0]);
        let lane_src = DefaultLaneVec([0.5, 1.5, 2.5, 3.5]);
        assert_eq!(
            dl.fma_lane::<2>(DefaultLaneVec([10.0; 4]), lane_src).0,
            [26.0, 27.0, 28.0, 29.0]
        );
        assert_eq!(dl.mul_lane::<2>(lane_src).0, [2.5, 5.0, 7.5, 10.0]);
        assert_eq!(
            DefaultLaneVec::splat(f32::NAN).exp2_clamped5().0,
            [0.03125; 4]
        );

        let dl_pos = DefaultLaneVec([0.5, 1.0, 4.0, 0.0]);
        assert_eq!(
            dl_pos.rcp_pos_seed_bits().0[0].to_bits(),
            0x7EF3_11C3_u32.wrapping_sub(0.5_f32.to_bits())
        );
        assert_eq!(
            dl_pos.rsqrt_pos_seed_bits().0[1].to_bits(),
            0x5F37_59DF_u32.wrapping_sub(1.0_f32.to_bits() >> 1)
        );
        assert!((dl_pos.rcp_pos_nr2().0[2] - 0.25).abs() < 1.0e-5);
        assert!((dl_pos.rcp_pos_nr3().0[2] - 0.25).abs() < 1.0e-6);
        assert!((dl_pos.rsqrt_pos_nr2().0[2] - 0.5).abs() < 1.0e-5);
        assert!((dl_pos.rsqrt_pos_nr3().0[2] - 0.5).abs() < 2.0e-6);
    }

    #[test]
    fn test_default_lane_vec_simd_and_kernel_bridge_coverage() {
        use dnam_kernels::KernelVec4;

        let arr = [1.5_f32, -2.5, 3.5, -4.5];
        let rhs_arr = [2.0_f32, 4.0, -1.0, 0.5];
        let a = <DefaultLaneVec as SimdVec4>::from_array(arr);
        let b = <DefaultLaneVec as SimdVec4>::load(&rhs_arr);

        assert_eq!(
            <DefaultLaneVec as SimdVec4>::dup_low_pair(a).0,
            [1.5, -2.5, 1.5, -2.5]
        );
        assert_eq!(
            <DefaultLaneVec as SimdVec4>::dup_high_pair(a).0,
            [3.5, -4.5, 3.5, -4.5]
        );
        assert_eq!(
            <ScalarVec4 as SimdVec4>::dup_low_pair(ScalarVec4(arr)).0,
            [1.5, -2.5, 1.5, -2.5]
        );
        assert_eq!(
            <ScalarVec4 as SimdVec4>::dup_high_pair(ScalarVec4(arr)).0,
            [3.5, -4.5, 3.5, -4.5]
        );

        assert_eq!(<DefaultLaneVec as SimdVec4>::abs(a).0, [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(
            <DefaultLaneVec as SimdVec4>::pair_swap_neg(a).0,
            [2.5, 1.5, 4.5, 3.5]
        );
        assert_eq!(<DefaultLaneVec as SimdVec4>::horizontal_sum(a), -2.0);
        assert_eq!(
            <DefaultLaneVec as SimdVec4>::horizontal_sum_pair(a, b),
            (-2.0, 5.5)
        );
        assert_eq!(<DefaultLaneVec as SimdVec4>::dot(a, b), -12.75);

        let dirty = DefaultLaneVec([f32::NAN, f32::INFINITY, f32::from_bits(1), -1.25]);
        let scrubbed = <DefaultLaneVec as SimdVec4>::scrub_recursive_state(dirty);
        assert_eq!(
            scrubbed.0.map(f32::to_bits),
            [0, 0, 0, (-1.25_f32).to_bits()]
        );
        let flushed = <DefaultLaneVec as SimdVec4>::flush_subnormal(dirty);
        assert!(flushed.0[0].is_nan() && flushed.0[1].is_infinite() && flushed.0[2].to_bits() == 0);

        let d0 = DefaultLaneVec([0.25, 1.0, 2.0, 4.0]);
        let d1 = DefaultLaneVec([0.5, 1.5, 5.0, 8.0]);
        let d2 = DefaultLaneVec([1.0, 3.0, 6.0, 10.0]);
        let d3 = DefaultLaneVec([2.5, 4.5, 7.5, 12.5]);
        let (rp0, rp1) = <DefaultLaneVec as SimdVec4>::rcp_ge1_pair_nr2(d0, d1);
        let rq = <DefaultLaneVec as SimdVec4>::rcp_ge1_quad_nr2([d0, d1, d2, d3]);
        assert!((rp0.0[0] - 1.0).abs() < 1.0e-5);
        assert!((rp1.0[0] - 1.0).abs() < 1.0e-5);
        assert!((rp0.0[3] - 0.25).abs() < 1.0e-5);
        assert!((rp1.0[3] - 0.125).abs() < 1.0e-5);
        for k in 0..4 {
            let expected = [d0, d1, d2, d3][k].0.map(|v| 1.0 / v.max(1.0));
            for lane in 0..4 {
                assert!((rq[k].0[lane] - expected[lane]).abs() < 1.0e-5);
            }
        }

        let ka = <DefaultLaneVec as KernelVec4>::load(&arr);
        let mut stored = [0.0_f32; 4];
        <DefaultLaneVec as KernelVec4>::store(ka, &mut stored);
        assert_eq!(stored, arr);
        assert_eq!(<DefaultLaneVec as KernelVec4>::from_array(arr).0, arr);
        assert_eq!(<DefaultLaneVec as KernelVec4>::to_array(ka), arr);
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::splat_pair(1.25, -3.75).0,
            [1.25, -3.75, 1.25, -3.75]
        );
        let kb = <DefaultLaneVec as KernelVec4>::load(&rhs_arr);
        let kc = <DefaultLaneVec as KernelVec4>::splat(2.0);
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::fms(ka, kb, kc).0,
            [-2.5, -10.5, 5.5, -5.5]
        );
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::fma_lane::<1>(ka, kc, kb).0,
            [9.5, 5.5, 11.5, 3.5]
        );
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::mul_lane::<3>(ka, kb).0,
            [0.75, -1.25, 1.75, -2.25]
        );
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::abs(ka).0,
            [1.5, 2.5, 3.5, 4.5]
        );
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::pair_swap_neg(ka).0,
            [2.5, 1.5, 4.5, 3.5]
        );
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::flush_subnormal(dirty).0[2].to_bits(),
            0
        );
        assert_eq!(<DefaultLaneVec as KernelVec4>::horizontal_sum(ka), -2.0);
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::horizontal_sum_pair(ka, kb),
            (-2.0, 5.5)
        );
        assert_eq!(<DefaultLaneVec as KernelVec4>::dot(ka, kb), -12.75);
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::rcp_pos_seed_bits(d2)
                .0
                .map(f32::to_bits),
            <ScalarVec4 as KernelVec4>::rcp_pos_seed_bits(ScalarVec4(d2.0))
                .0
                .map(f32::to_bits)
        );
        assert_eq!(
            <DefaultLaneVec as KernelVec4>::rsqrt_pos_seed_bits(d2)
                .0
                .map(f32::to_bits),
            <ScalarVec4 as KernelVec4>::rsqrt_pos_seed_bits(ScalarVec4(d2.0))
                .0
                .map(f32::to_bits)
        );
    }

    proptest! {
        #[test]
        fn prop_rcp_and_rsqrt_seed_bits_and_nr2_nr3(
            x0 in 1.0e-5_f32..1.0e5_f32,
            x1 in 1.0e-5_f32..1.0e5_f32,
            x2 in 1.0e-5_f32..1.0e5_f32,
            x3 in 1.0e-5_f32..1.0e5_f32,
        ) {
            type DefVec = <crate::math::DefaultMathBackend as crate::math::MathBackend>::Vec4;
            let xs = [x0, x1, x2, x3];
            let sv = ScalarVec4::from_array(xs);
            let dv = DefVec::from_array(xs);
            prop_assert_eq!(
                sv.rcp_pos_seed_bits().to_array().map(f32::to_bits),
                dv.rcp_pos_seed_bits().to_array().map(f32::to_bits)
            );
            prop_assert_eq!(
                sv.rsqrt_pos_seed_bits().to_array().map(f32::to_bits),
                dv.rsqrt_pos_seed_bits().to_array().map(f32::to_bits)
            );
            let r2 = dv.rcp_pos_nr2().to_array();
            let r3 = dv.rcp_pos_nr3().to_array();
            let g1 = dv.add(DefVec::splat(1.0)).rcp_ge1_nr2().to_array();
            let s2 = dv.rsqrt_pos_nr2().to_array();
            let s3 = dv.rsqrt_pos_nr3().to_array();
            for i in 0..4 {
                let d = f64::from(xs[i]);
                prop_assert!((f64::from(r2[i]) * d - 1.0).abs() < 1.0e-5);
                prop_assert!((f64::from(r3[i]) * d - 1.0).abs() < 1.0e-6);
                let g1_expected = f64::from(xs[i] + 1.0);
                prop_assert!((f64::from(g1[i]) * g1_expected - 1.0).abs() < 1.0e-5);
                let exact_inv_sqrt = d.sqrt().recip();
                prop_assert!(((f64::from(s2[i]) - exact_inv_sqrt) / exact_inv_sqrt).abs() < 1.0e-5);
                prop_assert!(((f64::from(s3[i]) - exact_inv_sqrt) / exact_inv_sqrt).abs() < 2.0e-6);
            }
        }

        #[test]
        fn exp2_clamped5_and_pow2_bits_match_exact_and_default(
            xs in prop::array::uniform4(-8.0_f32..8.0_f32),
        ) {
            type DefVec = <crate::math::DefaultMathBackend as crate::math::MathBackend>::Vec4;
            let sv = ScalarVec4::from_array(xs);
            let dv = DefVec::from_array(xs);
            let sa = sv.exp2_clamped5().to_array();
            let out = dv.exp2_clamped5().to_array();
            for i in 0..4 {
                prop_assert!((sa[i] - out[i]).abs() <= 1.0e-6 * sa[i].abs().max(1.0));
                let exact = f64::from(xs[i].clamp(-5.0, 5.0)).exp2();
                prop_assert!(((f64::from(out[i]) - exact) / exact).abs() < 3.0e-6);
            }
            let biased = ScalarVec4::from_array([-5.0, -1.0, 0.0, 5.0].map(|k| k + 12_582_912.0));
            let biased_d = DefVec::from_array([-5.0, -1.0, 0.0, 5.0].map(|k| k + 12_582_912.0));
            prop_assert_eq!(biased.pow2_from_biased_int_bits().to_array(), [0.03125, 0.5, 1.0, 32.0]);
            prop_assert_eq!(
                biased_d.pow2_from_biased_int_bits().to_array().map(f32::to_bits),
                [0.03125_f32, 0.5, 1.0, 32.0].map(f32::to_bits)
            );
        }
    }
}
