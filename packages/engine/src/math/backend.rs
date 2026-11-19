//! Hardware abstraction traits (`SimdVec4` and `MathBackend`) for 4-lane SIMD registers
//! and core division-free math primitives.
//!
//! Higher-level domain modules define their own `<Domain>Backend` traits in their respective
//! modules (`OpMBackend`, `OpNBackend`, `OpTBackend`, `Stage2Backend`, `Stage3Backend`),
//! keeping the dependency DAG strictly one-way (`engine -> stages -> operators -> math`).

use super::{BackendKind, algebraic_unit_rotor, f32_to_fp16, fp16_to_f32};
use crate::math::crc32c::Crc32cBackend;

/// Flushes `f32` subnormals to signed zero without changing any other bit pattern.
#[inline(always)]
pub const fn flush_subnormal_f32_bits(value: f32) -> f32 {
    let bits = value.to_bits();
    let mask = if bits & 0x7f80_0000 == 0 {
        0x8000_0000
    } else {
        u32::MAX
    };
    f32::from_bits(bits & mask)
}

/// Applies the portable fallback on targets without a native block-level denormal guard.
#[inline(always)]
pub(crate) const fn flush_recursive_state_f32(value: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        value
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        flush_subnormal_f32_bits(value)
    }
}

#[inline(always)]
pub(crate) fn flush_recursive_state_vec4<V: SimdVec4>(value: V) -> V {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        value
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        value.flush_subnormal()
    }
}

/// Portable scalar 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a single positive `f32`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_f32(value: f32) -> f32 {
    let mut est = f32::from_bits(0x5f37_59df - (value.to_bits() >> 1));
    let half_v = 0.5 * value;
    for _ in 0..2 {
        let corr = 1.5 - (half_v * est) * est;
        est *= corr;
    }
    est
}

/// Portable scalar 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)`.
#[inline(always)]
pub fn rsqrt_nr2_scalar_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
    let mut e0 = f32::from_bits(0x5f37_59df - (v0.to_bits() >> 1));
    let mut e1 = f32::from_bits(0x5f37_59df - (v1.to_bits() >> 1));
    let hv0 = 0.5 * v0;
    let hv1 = 0.5 * v1;
    for _ in 0..2 {
        let c0 = 1.5 - (hv0 * e0) * e0;
        let c1 = 1.5 - (hv1 * e1) * e1;
        e0 *= c0;
        e1 *= c1;
    }
    (e0, e1)
}

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
    /// Fast base-2 exponential `2^clamp(self, -5.0, 5.0)` via range reduction + Chebyshev-economized degree-5 polynomial (`< 2.2e-7`).
    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        const C2: f32 = 0.240_221_09;
        const C3: f32 = 0.055_504_11;
        const C4: f32 = 0.009_675_892;
        const C5: f32 = 0.001_333_355_8;
        let x = self.clamp(Self::splat(-5.0), Self::splat(5.0));
        let magic = Self::splat(12_582_912.0);
        let k_biased = x.add(magic);
        let k = k_biased.sub(magic);
        let f = x.sub(k);
        let pow2_k = k_biased.pow2_from_biased_int_bits();
        let poly = Self::splat(C4).fma(Self::splat(C5), f);
        let poly = Self::splat(C3).fma(poly, f);
        let poly = Self::splat(C2).fma(poly, f);
        let poly = Self::splat(core::f32::consts::LN_2).fma(poly, f);
        let poly = Self::splat(1.0).fma(poly, f);
        pow2_k.mul(poly)
    }
    /// Fast positive reciprocal `1.0 / self` using 2 Newton-Raphson FMA/FMS steps (`< 6.6e-6`) when `self >= 1.0`.
    #[inline(always)]
    fn rcp_ge1_nr2(self) -> Self {
        let one = Self::splat(1.0);
        let y = self.rcp_pos_seed_bits();
        let err0 = one.fms(self, y);
        let y = y.fma(y, err0);
        let err1 = one.fms(self, y);
        y.fma(y, err1)
    }
    /// Evaluates two independent `rcp_ge1_nr2` vectors simultaneously to hide FMA latency.
    #[inline(always)]
    fn rcp_ge1_pair_nr2(self, other: Self) -> (Self, Self) {
        (self.rcp_ge1_nr2(), other.rcp_ge1_nr2())
    }
    /// Evaluates four independent `rcp_ge1_nr2` vectors simultaneously across all SIMD pipes.
    #[inline(always)]
    fn rcp_ge1_quad_nr2(d: [Self; 4]) -> [Self; 4] {
        let (y0, y1) = d[0].rcp_ge1_pair_nr2(d[1]);
        let (y2, y3) = d[2].rcp_ge1_pair_nr2(d[3]);
        [y0, y1, y2, y3]
    }
    /// Computes positive reciprocal `1.0 / max(self, 1.0e-6)` using 2 Newton-Raphson FMA/FMS steps (`< 6.6e-6`).
    #[inline(always)]
    fn rcp_pos_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let two = Self::splat(2.0);
        let mut y = d_safe.rcp_pos_seed_bits();
        y = y.mul(two.fms(d_safe, y));
        y.mul(two.fms(d_safe, y))
    }
    /// Computes positive reciprocal `1.0 / max(self, 1.0e-6)` using 3 Newton-Raphson FMA/FMS steps (`< 1 ULP`).
    #[inline(always)]
    fn rcp_pos_nr3(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let two = Self::splat(2.0);
        let mut y = d_safe.rcp_pos_seed_bits();
        y = y.mul(two.fms(d_safe, y));
        y = y.mul(two.fms(d_safe, y));
        y.mul(two.fms(d_safe, y))
    }
    /// Computes inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using 2 Newton-Raphson FMA/FMS steps.
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
        (a.rsqrt_pos_nr2(), b.rsqrt_pos_nr2())
    }
    /// Computes inverse square root `1.0 / sqrt(max(self, 1.0e-12))` using 3 Newton-Raphson FMA/FMS steps.
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

/// Level 2 Hardware Abstraction: Core SIMD math primitives across every architecture
/// (`aarch64`, `x86_64`, `wasm32`, `scalar`).
pub trait MathBackend: Copy + Clone + Send + Sync + 'static + Crc32cBackend {
    /// Underlying 128-bit 4-lane vector register type.
    type Vec4: SimdVec4;
    /// Human-readable backend name for telemetry and benchmarks.
    const NAME: &'static str;
    /// Enum identifier for this backend.
    const KIND: BackendKind;

    /// 4x4 matrix-vector FMA tile `acc + sum_{c=0..3} x[c] * mat[c]`.
    #[inline(always)]
    fn fma_mat4x4_vec4(acc: Self::Vec4, mat: &[[f32; 4]; 4], x: &[f32; 4]) -> Self::Vec4 {
        let xv = Self::Vec4::load(x);
        let c0 = Self::Vec4::load(&mat[0]);
        let c1 = Self::Vec4::load(&mat[1]);
        let c2 = Self::Vec4::load(&mat[2]);
        let c3 = Self::Vec4::load(&mat[3]);
        let p01 = c0.mul_lane::<0>(xv).fma_lane::<1>(c1, xv);
        let p23 = c2.mul_lane::<2>(xv).fma_lane::<3>(c3, xv);
        acc.add(p01.add(p23))
    }

    /// Stereo 4x4 matrix-vector FMA tile using 4 pre-loaded column vectors in registers.
    #[inline(always)]
    fn fma_cols4_stereo(
        acc_l: Self::Vec4,
        acc_r: Self::Vec4,
        cols: [Self::Vec4; 4],
        x_l: Self::Vec4,
        x_r: Self::Vec4,
    ) -> (Self::Vec4, Self::Vec4) {
        let [c0, c1, c2, c3] = cols;
        let p01_l = c0.mul_lane::<0>(x_l).fma_lane::<1>(c1, x_l);
        let p23_l = c2.mul_lane::<2>(x_l).fma_lane::<3>(c3, x_l);
        let p01_r = c0.mul_lane::<0>(x_r).fma_lane::<1>(c1, x_r);
        let p23_r = c2.mul_lane::<2>(x_r).fma_lane::<3>(c3, x_r);
        (acc_l.add(p01_l.add(p23_l)), acc_r.add(p01_r.add(p23_r)))
    }

    /// Stereo 4x4 matrix-vector FMA tile updating `(acc_l, acc_r)` with shared column loads.
    #[inline(always)]
    fn fma_mat4x4_stereo(
        acc_l: Self::Vec4,
        acc_r: Self::Vec4,
        cols: &[[f32; 4]; 4],
        x_l: Self::Vec4,
        x_r: Self::Vec4,
    ) -> (Self::Vec4, Self::Vec4) {
        let c0 = Self::Vec4::load(&cols[0]);
        let c1 = Self::Vec4::load(&cols[1]);
        let c2 = Self::Vec4::load(&cols[2]);
        let c3 = Self::Vec4::load(&cols[3]);
        Self::fma_cols4_stereo(acc_l, acc_r, [c0, c1, c2, c3], x_l, x_r)
    }

    /// Transposes a 4×4 matrix of `Vec4` rows (`[r0, r1, r2, r3]`) into 4 `Vec4` columns.
    #[inline(always)]
    fn transpose_4x4(
        r0: Self::Vec4,
        r1: Self::Vec4,
        r2: Self::Vec4,
        r3: Self::Vec4,
    ) -> [Self::Vec4; 4] {
        let a0 = r0.to_array();
        let a1 = r1.to_array();
        let a2 = r2.to_array();
        let a3 = r3.to_array();
        [
            Self::Vec4::from_array([a0[0], a1[0], a2[0], a3[0]]),
            Self::Vec4::from_array([a0[1], a1[1], a2[1], a3[1]]),
            Self::Vec4::from_array([a0[2], a1[2], a2[2], a3[2]]),
            Self::Vec4::from_array([a0[3], a1[3], a2[3], a3[3]]),
        ]
    }

    /// Converts a `Vec4` register directly to 4 IEEE-754 half-precision `fp16` (`u16`) values.
    #[inline(always)]
    fn vec4_to_fp16x4(src: Self::Vec4) -> [u16; 4] {
        Self::f32x4_to_fp16x4(&src.to_array())
    }

    /// Converts 4 IEEE-754 half-precision `fp16` (`u16`) values directly into a `Vec4` register.
    #[inline(always)]
    fn fp16x4_to_vec4(src: &[u16; 4]) -> Self::Vec4 {
        Self::Vec4::load(&Self::fp16x4_to_f32x4(src))
    }

    /// Converts 4 IEEE-754 single-precision `f32` samples to `fp16` (`u16`).
    #[inline(always)]
    fn f32x4_to_fp16x4(src: &[f32; 4]) -> [u16; 4] {
        [
            f32_to_fp16(src[0]),
            f32_to_fp16(src[1]),
            f32_to_fp16(src[2]),
            f32_to_fp16(src[3]),
        ]
    }

    /// Converts 4 IEEE-754 half-precision `fp16` (`u16`) samples to `f32`.
    #[inline(always)]
    fn fp16x4_to_f32x4(src: &[u16; 4]) -> [f32; 4] {
        [
            fp16_to_f32(src[0]),
            fp16_to_f32(src[1]),
            fp16_to_f32(src[2]),
            fp16_to_f32(src[3]),
        ]
    }

    /// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a single positive `f32` (`0 FSQRT`, `0 FDIV`).
    #[inline(always)]
    fn rsqrt_nr2_f32(v: f32) -> f32 {
        rsqrt_nr2_scalar_f32(v)
    }

    /// Evaluates 2-step Newton `rsqrt` (`0x5f37_59df - (v >> 1)`) for a stereo pair `(v0, v1)` (`0 FSQRT`, `0 FDIV`).
    #[inline(always)]
    fn rsqrt_nr2_pair_f32(v0: f32, v1: f32) -> (f32, f32) {
        rsqrt_nr2_scalar_pair_f32(v0, v1)
    }

    /// Evaluates the division-free algebraic unit-circle phasor `(cos(theta), sin(theta))` (`ref[dsp.op-m.rotor.contractive]`).
    #[inline(always)]
    fn fast_unit_phasor(theta_rad: f32) -> (f32, f32) {
        algebraic_unit_rotor(theta_rad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::{ScalarBackend, ScalarVec4};
    use proptest::prelude::*;

    fn check_flush_subnormal<V: SimdVec4>() {
        let cases = [
            (
                [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001],
                [0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000],
            ),
            (
                [0x007f_ffff, 0x807f_ffff, 0x0080_0000, 0x8080_0000],
                [0x0000_0000, 0x8000_0000, 0x0080_0000, 0x8080_0000],
            ),
            (
                [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
                [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345],
            ),
        ];
        for (input, expected) in cases {
            let values = input.map(f32::from_bits);
            assert_eq!(
                values.map(flush_subnormal_f32_bits).map(f32::to_bits),
                expected
            );
            assert_eq!(
                V::from_array(values)
                    .flush_subnormal()
                    .to_array()
                    .map(f32::to_bits),
                expected
            );
        }
    }

    #[test]
    fn test_flush_subnormal_preserves_bits() {
        check_flush_subnormal::<ScalarVec4>();
        #[cfg(target_arch = "aarch64")]
        check_flush_subnormal::<crate::math::NeonVec4>();
        #[cfg(target_arch = "x86_64")]
        check_flush_subnormal::<crate::math::X86Vec4>();
        #[cfg(target_arch = "wasm32")]
        check_flush_subnormal::<crate::math::WasmVec4>();
    }

    proptest! {
        #[test]
        fn prop_flush_subnormal_preserves_non_subnormal_bits(bits in any::<u32>()) {
            let expected = if bits & 0x7f80_0000 == 0 {
                bits & 0x8000_0000
            } else {
                bits
            };
            prop_assert_eq!(flush_subnormal_f32_bits(f32::from_bits(bits)).to_bits(), expected);
            let vector = ScalarVec4::splat(f32::from_bits(bits)).flush_subnormal();
            prop_assert_eq!(vector.to_array().map(f32::to_bits), [expected; 4]);
        }
    }

    #[test]
    fn test_simd_vec4_defaults() {
        let z = ScalarVec4::zero();
        assert_eq!(z.to_array(), [0.0; 4]);

        let arr = [1.0, 2.0, 3.0, 4.0];
        let v = ScalarVec4::from_array(arr);
        assert_eq!(v.to_array(), arr);

        let d = v.dot(v);
        assert_eq!(d, 1.0 + 4.0 + 9.0 + 16.0);

        let lo = ScalarVec4::splat(1.5);
        let hi = ScalarVec4::splat(3.5);
        let clamped = v.clamp(lo, hi);
        assert_eq!(clamped.to_array(), [1.5, 2.0, 3.0, 3.5]);
    }

    #[test]
    fn test_math_backend_defaults() {
        let (c, s) = ScalarBackend::fast_unit_phasor(0.5);
        assert!(c.is_finite());
        assert!(s.is_finite());

        let crc = ScalarBackend::crc32c(b"123456789");
        assert_eq!(crc, 0xe3069283);

        let fp16 = ScalarBackend::f32x4_to_fp16x4(&[1.0, 2.0, 3.0, 4.0]);
        let back = ScalarBackend::fp16x4_to_f32x4(&fp16);
        assert!((back[0] - 1.0).abs() < 1e-3);

        let v = ScalarVec4::splat(0.5);
        let mat = [[0.1f32; 4]; 4];
        let _ = ScalarBackend::fma_mat4x4_vec4(v, &mat, &[0.5; 4]);

        let r1 = ScalarBackend::rsqrt_nr2_f32(4.0);
        assert!((r1 - 0.5).abs() < 1e-3);
        let (p0, p1) = ScalarBackend::rsqrt_nr2_pair_f32(4.0, 9.0);
        assert!((p0 - 0.5).abs() < 1e-3);
        assert!((p1 - 1.0 / 3.0).abs() < 1e-3);

        #[derive(Clone, Copy)]
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
        }
        let dl = DefaultLaneVec([1.0, 2.0, 3.0, 4.0]);
        let lane_src = DefaultLaneVec([0.5, 1.5, 2.5, 3.5]);
        let res = dl.fma_lane::<2>(DefaultLaneVec([10.0; 4]), lane_src);
        assert_eq!(res.0, [26.0, 27.0, 28.0, 29.0]);
        assert_eq!(dl.mul_lane::<2>(lane_src).0, [2.5, 5.0, 7.5, 10.0]);

        let dl_loaded = DefaultLaneVec::load(&[-1.0, 2.0, -3.0, 4.0]);
        let other = DefaultLaneVec::splat(1.5);
        assert_eq!(dl_loaded.sub(other).0, [-2.5, 0.5, -4.5, 2.5]);
        assert_eq!(
            dl_loaded
                .fms(other, DefaultLaneVec::load(&[2.0, 2.0, 2.0, 2.0]))
                .0,
            [-4.0, -1.0, -6.0, 1.0]
        );
        assert_eq!(dl_loaded.max(other).0, [1.5, 2.0, 1.5, 4.0]);
        assert_eq!(dl_loaded.min(other).0, [-1.0, 1.5, -3.0, 1.5]);
        assert_eq!(dl_loaded.abs().0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dl_loaded.pair_swap_neg().0, [-2.0, -1.0, -4.0, -3.0]);
        assert_eq!(dl_loaded.abs().horizontal_sum(), 10.0);
        let mut stored = [0.0; 4];
        dl_loaded.store(&mut stored);
        assert_eq!(stored, [-1.0, 2.0, -3.0, 4.0]);

        let (dl_l, dl_r) = ScalarBackend::fma_mat4x4_stereo(v, v, &mat, v, v);
        assert_eq!(dl_l.to_array(), dl_r.to_array());
        let fp16_v = ScalarBackend::vec4_to_fp16x4(v);
        assert_eq!(fp16_v, ScalarBackend::f32x4_to_fp16x4(&v.to_array()));
        assert_eq!(
            ScalarBackend::fp16x4_to_vec4(&fp16_v).to_array(),
            ScalarBackend::fp16x4_to_f32x4(&fp16_v)
        );
        let t_s = ScalarBackend::transpose_4x4(
            ScalarVec4::from_array([1.0, 2.0, 3.0, 4.0]),
            ScalarVec4::from_array([5.0, 6.0, 7.0, 8.0]),
            ScalarVec4::from_array([9.0, 10.0, 11.0, 12.0]),
            ScalarVec4::from_array([13.0, 14.0, 15.0, 16.0]),
        );
        assert_eq!(t_s[0].to_array(), [1.0, 5.0, 9.0, 13.0]);
        assert_eq!(t_s[1].to_array(), [2.0, 6.0, 10.0, 14.0]);
        assert_eq!(t_s[2].to_array(), [3.0, 7.0, 11.0, 15.0]);
        assert_eq!(t_s[3].to_array(), [4.0, 8.0, 12.0, 16.0]);
        type DefB = crate::math::DefaultMathBackend;
        let t_d = DefB::transpose_4x4(
            <DefB as MathBackend>::Vec4::from_array([1.0, 2.0, 3.0, 4.0]),
            <DefB as MathBackend>::Vec4::from_array([5.0, 6.0, 7.0, 8.0]),
            <DefB as MathBackend>::Vec4::from_array([9.0, 10.0, 11.0, 12.0]),
            <DefB as MathBackend>::Vec4::from_array([13.0, 14.0, 15.0, 16.0]),
        );
        for k in 0..4 {
            assert_eq!(t_d[k].to_array(), t_s[k].to_array());
        }
        let fp16_d = DefB::vec4_to_fp16x4(<DefB as MathBackend>::Vec4::from_array([
            0.25, -1.5, 12.5, -65504.0,
        ]));
        assert_eq!(
            DefB::fp16x4_to_vec4(&fp16_d).to_array(),
            ScalarBackend::fp16x4_to_vec4(&fp16_d).to_array()
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
        assert!((dl_pos.rsqrt_pos_nr2().0[2] - 0.5).abs() < 2.5e-3);
        assert!((dl_pos.rsqrt_pos_nr3().0[2] - 0.5).abs() < 2.0e-5);
        assert!(dl_pos.rcp_pos_nr2().0[3].is_finite());
        assert!(dl_pos.rsqrt_pos_nr2().0[3].is_finite());
    }

    proptest! {
        #[test]
        fn prop_rcp_and_rsqrt_seed_bits_and_nr2_nr3(
            x0 in 1.0e-5_f32..1.0e5_f32,
            x1 in 1.0e-5_f32..1.0e5_f32,
            x2 in 1.0e-5_f32..1.0e5_f32,
            x3 in 1.0e-5_f32..1.0e5_f32,
        ) {
            type DefVec = <crate::math::DefaultMathBackend as MathBackend>::Vec4;
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
                prop_assert!(((f64::from(s2[i]) - exact_inv_sqrt) / exact_inv_sqrt).abs() < 2.5e-3);
                prop_assert!(((f64::from(s3[i]) - exact_inv_sqrt) / exact_inv_sqrt).abs() < 2.0e-5);
            }
        }

        #[test]
        fn exp2_clamped5_and_pow2_bits_match_exact_and_default(
            xs in prop::array::uniform4(-8.0_f32..8.0_f32),
        ) {
            type DefVec = <crate::math::DefaultMathBackend as MathBackend>::Vec4;
            let sv = ScalarVec4::from_array(xs);
            let dv = DefVec::from_array(xs);
            let sa = sv.exp2_clamped5().to_array();
            let out = dv.exp2_clamped5().to_array();
            for i in 0..4 {
                // Hardware FMA uses 1 rounding vs unfused `ScalarVec4` (`a * b + c`), allowing 1-2 ULP difference.
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
