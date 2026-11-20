//! Portable `#![no_std]` 4-lane (`KernelVec4` / `GpuVec4`) and 8-lane stereo (`StereoVec8`)
//! SIMD abstractions, compile-time mathematical constants, and `fp16` codecs
//! (`ref[arch.simd.two-level-traits]`, `ref[sdk.rust.shared-nostd-kernels]`).
//!
//! All runtime primitives execute with zero `FDIV`, zero `FSQRT`, zero `libm` calls, and zero
//! `u64`/`f64` types so CPU SIMD backends and 32-bit `spirv-unknown-naga-wgsl` shaders evaluate
//! bit-aligned arithmetic.

pub mod const_eval;
pub mod fp16;
pub mod gpu_vec4;
pub mod stereo8;

pub use const_eval::{
    DC_BLOCKER_5HZ_48K_F32, DC_BLOCKER_10HZ_48K_BITS, FULL_SPEC, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, LITE_SPEC, MAX_LOOP_GAIN_F32, MAX_PARALRU_R_F32,
    PRO_SPEC, SonaTierSpec, ULTRA_SPEC,
};
pub use fp16::{
    f32_to_fp16, f32_to_fp16_u32, fp16_to_f32, fp16_u32_to_f32, pack_fp16x2_u32, unpack_fp16x2_u32,
};
pub use gpu_vec4::GpuVec4;
pub use stereo8::{InterleavedStereo8, StereoVec8};

/// IEEE-754 `1.5 * 2^23` (`12_582_912.0`) magic integer-rounding bias (`0x4B40_0000`).
pub(crate) const BIASED_INT_MAGIC_F32: f32 = 1.5 * ((1_u32 << 23) as f32);

/// Chebyshev-economized degree-5 coefficients for `2^f` on `f in [-0.5, 0.5]` (`< 2.5e-7` error;
/// `C1 = ln(2)`, `C2..C5` minimax-adjusted around `ln(2)^k / k!`,
/// `ref[repo.code.constant-eval-and-derivations]`).
pub(crate) const EXP2_C2: f32 = 0.240_221_09;
pub(crate) const EXP2_C3: f32 = 0.055_504_11;
pub(crate) const EXP2_C4: f32 = 0.009_675_892;
pub(crate) const EXP2_C5: f32 = 0.001_333_355_8;

#[inline(always)]
const fn const_abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

#[inline(always)]
const fn const_exp2_poly_f32(f: f32) -> f32 {
    let p = EXP2_C4 + EXP2_C5 * f;
    let p = EXP2_C3 + p * f;
    let p = EXP2_C2 + p * f;
    let p = core::f32::consts::LN_2 + p * f;
    1.0 + p * f
}

#[inline(always)]
const fn const_rcp_nr2_f32(d: f32) -> f32 {
    let y0 = f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(d.to_bits()));
    let y1 = y0 + y0 * (1.0 - d * y0);
    y1 + y1 * (1.0 - d * y1)
}

#[inline(always)]
const fn const_rcp_nr3_f32(d: f32) -> f32 {
    let y2 = const_rcp_nr2_f32(d);
    y2 + y2 * (1.0 - d * y2)
}

#[inline(always)]
const fn const_rsqrt_nr2_f32(d: f32) -> f32 {
    let half_d = 0.5 * d;
    let y0 = f32::from_bits(IEEE754_F32_RSQRT_SEED.wrapping_sub(d.to_bits() >> 1));
    let y1 = y0 * (1.5 - half_d * (y0 * y0));
    y1 * (1.5 - half_d * (y1 * y1))
}

#[inline(always)]
const fn const_rsqrt_nr3_f32(d: f32) -> f32 {
    let half_d = 0.5 * d;
    let y2 = const_rsqrt_nr2_f32(d);
    y2 * (1.5 - half_d * (y2 * y2))
}

const _: () = assert!(BIASED_INT_MAGIC_F32.to_bits() == 0x4B40_0000);
const _: () = assert!(EXP2_C2 > 0.240 && EXP2_C2 < 0.241);
const _: () = assert!(EXP2_C3 > 0.055 && EXP2_C3 < 0.056);
const _: () = assert!(EXP2_C4 > 0.009 && EXP2_C4 < 0.010);
const _: () = assert!(EXP2_C5 > 0.001 && EXP2_C5 < 0.0014);
const _: () =
    assert!(const_abs_f32(const_exp2_poly_f32(-0.5) - core::f32::consts::FRAC_1_SQRT_2) < 2.5e-7);
const _: () = assert!(const_exp2_poly_f32(0.0) == 1.0);
const _: () = assert!(const_abs_f32(const_exp2_poly_f32(0.5) - core::f32::consts::SQRT_2) < 2.5e-7);
const _: () = assert!(const_abs_f32(const_rcp_nr2_f32(3.0) * 3.0 - 1.0) < 1.0e-5);
const _: () = assert!(const_abs_f32(const_rcp_nr3_f32(3.0) * 3.0 - 1.0) < 1.0e-6);
const _: () =
    assert!(const_abs_f32(const_rsqrt_nr2_f32(2.0) - core::f32::consts::FRAC_1_SQRT_2) < 2.5e-3);
const _: () =
    assert!(const_abs_f32(const_rsqrt_nr3_f32(2.0) - core::f32::consts::FRAC_1_SQRT_2) < 1.0e-6);
const _: () = assert!(const_abs_f32(const_rsqrt_nr3_f32(4.0) - 0.5) < 1.0e-6);
const _: () = assert!(const_abs_f32(3.0 * const_rsqrt_nr3_f32(9.0) - 1.0) < 1.0e-6);

/// Flushes `f32` subnormals to signed zero without changing any other bit pattern.
#[inline(always)]
#[must_use]
pub const fn flush_subnormal_f32_bits(value: f32) -> f32 {
    let bits = value.to_bits();
    let mask = if (bits & 0x7F80_0000) == 0 {
        0x8000_0000
    } else {
        u32::MAX
    };
    f32::from_bits(bits & mask)
}

/// Scrubs `NaN` / `±Inf` (`(bits & 0x7F80_0000) == 0x7F80_0000`) to `+0.0` and flushes
/// `f32` subnormals (`(bits & 0x7F80_0000) == 0`) to signed zero (`bits & 0x8000_0000`).
#[inline(always)]
#[must_use]
pub const fn scrub_and_flush_f32_bits(value: f32) -> f32 {
    let bits = value.to_bits();
    let exp = bits & 0x7F80_0000;
    let out_bits = if exp == 0x7F80_0000 {
        0
    } else if exp == 0 {
        bits & 0x8000_0000
    } else {
        bits
    };
    f32::from_bits(out_bits)
}

/// Level 1 Portable 4-Lane `f32` SIMD Register Abstraction (`f32x4` / `GpuVec4`).
pub trait KernelVec4: Copy + Clone + core::fmt::Debug + Send + Sync + 'static {
    /// Broadcasts `val` across all 4 lanes.
    fn splat(val: f32) -> Self;

    /// Returns a 4-lane vector of zeros (`0.0`).
    #[inline(always)]
    fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Broadcasts the stereo pair `(a, b)` into `[a, b, a, b]`.
    #[inline(always)]
    fn splat_pair(a: f32, b: f32) -> Self {
        Self::from_array([a, b, a, b])
    }

    /// Constructs a 4-lane vector from `[f32; 4]`.
    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        Self::load(&arr)
    }

    /// Converts this 4-lane vector into `[f32; 4]`.
    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut dst = [0.0; 4];
        self.store(&mut dst);
        dst
    }

    /// Loads 4 `f32` lanes from `src`.
    fn load(src: &[f32; 4]) -> Self;

    /// Stores 4 `f32` lanes into `dst`.
    fn store(self, dst: &mut [f32; 4]);

    /// Lane-wise addition (`self + rhs`).
    fn add(self, rhs: Self) -> Self;

    /// Lane-wise subtraction (`self - rhs`).
    fn sub(self, rhs: Self) -> Self;

    /// Lane-wise multiplication (`self * rhs`).
    fn mul(self, rhs: Self) -> Self;

    /// Fused multiply-add (`self + a * b`).
    fn fma(self, a: Self, b: Self) -> Self;

    /// Fused multiply-subtract (`self - a * b`).
    fn fms(self, a: Self, b: Self) -> Self;

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
    #[inline(always)]
    fn neg(self) -> Self {
        let a = self.to_array();
        Self::from_array([-a[0], -a[1], -a[2], -a[3]])
    }

    /// Lane-wise branchless select: `if lhs <= rhs { if_true } else { if_false }`.
    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        let l = lhs.to_array();
        let r = rhs.to_array();
        let t = if_true.to_array();
        let f = if_false.to_array();
        Self::from_array([
            if l[0] <= r[0] { t[0] } else { f[0] },
            if l[1] <= r[1] { t[1] } else { f[1] },
            if l[2] <= r[2] { t[2] } else { f[2] },
            if l[3] <= r[3] { t[3] } else { f[3] },
        ])
    }

    /// Returns `true` iff all 4 lanes satisfy `self[i] <= rhs[i]` (returns `false` on `NaN`).
    #[inline(always)]
    fn all_le(self, rhs: Self) -> bool {
        let a = self.to_array();
        let b = rhs.to_array();
        a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2] && a[3] <= b[3]
    }

    /// Swaps adjacent pairs and negates even lanes (`[-a[1], a[0], -a[3], a[2]]`).
    #[inline(always)]
    fn pair_swap_neg(self) -> Self {
        let a = self.to_array();
        Self::from_array([-a[1], a[0], -a[3], a[2]])
    }

    /// Flushes subnormal lanes to signed zero without changing normal/special bit patterns.
    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        let a = self.to_array();
        Self::from_array([
            flush_subnormal_f32_bits(a[0]),
            flush_subnormal_f32_bits(a[1]),
            flush_subnormal_f32_bits(a[2]),
            flush_subnormal_f32_bits(a[3]),
        ])
    }

    /// Scrubs non-finite lanes (`NaN`/`±Inf`) to `+0.0` and flushes subnormals to signed zero.
    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        let a = self.to_array();
        Self::from_array([
            scrub_and_flush_f32_bits(a[0]),
            scrub_and_flush_f32_bits(a[1]),
            scrub_and_flush_f32_bits(a[2]),
            scrub_and_flush_f32_bits(a[3]),
        ])
    }

    /// Horizontal pairwise sum `(a[0] + a[1]) + (a[2] + a[3])` for `0-ULP` cross-ISA tree parity.
    fn horizontal_sum(self) -> f32;

    /// Simultaneously reduces two 4-lane vectors into `(self.horizontal_sum(), rhs.horizontal_sum())`.
    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        (self.horizontal_sum(), rhs.horizontal_sum())
    }

    /// 4-lane dot product (`(self * rhs).horizontal_sum()`).
    #[inline(always)]
    fn dot(self, rhs: Self) -> f32 {
        self.mul(rhs).horizontal_sum()
    }

    /// IEEE-754 integer-sub reciprocal seed (`IEEE754_F32_RCP_SEED - bits(self)`).
    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        let a = self.to_array();
        Self::from_array([
            f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(a[0].to_bits())),
            f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(a[1].to_bits())),
            f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(a[2].to_bits())),
            f32::from_bits(IEEE754_F32_RCP_SEED.wrapping_sub(a[3].to_bits())),
        ])
    }

    /// IEEE-754 integer-sub inverse-sqrt seed (`IEEE754_F32_RSQRT_SEED - (bits(self) >> 1)`).
    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        let a = self.to_array();
        Self::from_array([
            f32::from_bits(IEEE754_F32_RSQRT_SEED.wrapping_sub(a[0].to_bits() >> 1)),
            f32::from_bits(IEEE754_F32_RSQRT_SEED.wrapping_sub(a[1].to_bits() >> 1)),
            f32::from_bits(IEEE754_F32_RSQRT_SEED.wrapping_sub(a[2].to_bits() >> 1)),
            f32::from_bits(IEEE754_F32_RSQRT_SEED.wrapping_sub(a[3].to_bits() >> 1)),
        ])
    }

    /// Converts `k + 12_582_912.0` biased integer bits into exact `2^k` (`k in [-126, 127]`).
    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        let a = self.to_array();
        Self::from_array([
            f32::from_bits(a[0].to_bits().wrapping_add(IEEE754_F32_EXP_BIAS) << 23),
            f32::from_bits(a[1].to_bits().wrapping_add(IEEE754_F32_EXP_BIAS) << 23),
            f32::from_bits(a[2].to_bits().wrapping_add(IEEE754_F32_EXP_BIAS) << 23),
            f32::from_bits(a[3].to_bits().wrapping_add(IEEE754_F32_EXP_BIAS) << 23),
        ])
    }

    /// Fast positive reciprocal `1.0 / self` (`self >= 1.0`) via Chebyshev seed + 2 FMA/FMS NR steps.
    #[inline(always)]
    fn rcp_ge1_nr2(self) -> Self {
        let one = Self::splat(1.0);
        let y0 = self.rcp_pos_seed_bits();
        let err0 = one.fms(self, y0);
        let y1 = y0.fma(y0, err0);
        let err1 = one.fms(self, y1);
        y1.fma(y1, err1)
    }

    /// Clamped positive reciprocal `1.0 / max(self, 1.0e-6)` via seed + 2 FMA/FMS NR steps.
    #[inline(always)]
    fn rcp_pos_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        y1.fma(y1, one.fms(d_safe, y1))
    }

    /// Evaluates four independent `rcp_pos_nr2` vectors simultaneously across SIMD pipes.
    #[inline(always)]
    fn rcp_pos_quad_nr2(d: [Self; 4]) -> [Self; 4] {
        let eps = Self::splat(1.0e-6);
        let one = Self::splat(1.0);
        let d0 = d[0].max(eps);
        let d1 = d[1].max(eps);
        let d2 = d[2].max(eps);
        let d3 = d[3].max(eps);
        let mut y0 = d0.rcp_pos_seed_bits();
        let mut y1 = d1.rcp_pos_seed_bits();
        let mut y2 = d2.rcp_pos_seed_bits();
        let mut y3 = d3.rcp_pos_seed_bits();
        y0 = y0.fma(y0, one.fms(d0, y0));
        y1 = y1.fma(y1, one.fms(d1, y1));
        y2 = y2.fma(y2, one.fms(d2, y2));
        y3 = y3.fma(y3, one.fms(d3, y3));
        [
            y0.fma(y0, one.fms(d0, y0)),
            y1.fma(y1, one.fms(d1, y1)),
            y2.fma(y2, one.fms(d2, y2)),
            y3.fma(y3, one.fms(d3, y3)),
        ]
    }

    /// Clamped positive reciprocal `1.0 / max(self, 1.0e-6)` via seed + 3 FMA/FMS NR steps (`< 1 ULP`).
    #[inline(always)]
    fn rcp_pos_nr3(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-6));
        let one = Self::splat(1.0);
        let y0 = d_safe.rcp_pos_seed_bits();
        let y1 = y0.fma(y0, one.fms(d_safe, y0));
        let y2 = y1.fma(y1, one.fms(d_safe, y1));
        y2.fma(y2, one.fms(d_safe, y2))
    }

    /// Clamped inverse square root `1.0 / sqrt(max(self, 1.0e-12))` via seed + 2 NR steps.
    #[inline(always)]
    fn rsqrt_pos_nr2(self) -> Self {
        let d_safe = self.max(Self::splat(1.0e-12));
        let half_d = d_safe.mul(Self::splat(0.5));
        let three_halves = Self::splat(1.5);
        let mut y = d_safe.rsqrt_pos_seed_bits();
        y = y.mul(three_halves.fms(half_d, y.mul(y)));
        y.mul(three_halves.fms(half_d, y.mul(y)))
    }

    /// Evaluates `(rsqrt_pos_nr2(a), rsqrt_pos_nr2(b))` for dual-stream interleaving.
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

    /// Clamped inverse square root `1.0 / sqrt(max(self, 1.0e-12))` via seed + 3 NR steps.
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

    /// Fast base-2 exponential `2^clamp(self, -5.0, 5.0)` (`< 2.5e-7`, `ref[dsp.op-n.polyadaa.midpoint]`).
    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        let x = self.clamp(Self::splat(-5.0), Self::splat(5.0));
        let magic = Self::splat(BIASED_INT_MAGIC_F32);
        let k_biased = x.add(magic);
        let k = k_biased.sub(magic);
        let f = x.sub(k);
        let pow2_k = k_biased.pow2_from_biased_int_bits();
        let poly = Self::splat(EXP2_C4).fma(Self::splat(EXP2_C5), f);
        let poly = Self::splat(EXP2_C3).fma(poly, f);
        let poly = Self::splat(EXP2_C2).fma(poly, f);
        let poly = Self::splat(core::f32::consts::LN_2).fma(poly, f);
        let poly = Self::splat(1.0).fma(poly, f);
        pow2_k.mul(poly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Copy, Clone, Debug, PartialEq)]
    struct DefaultTraitVec4([f32; 4]);

    impl KernelVec4 for DefaultTraitVec4 {
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
            Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
        }
        fn sub(self, rhs: Self) -> Self {
            Self(core::array::from_fn(|i| self.0[i] - rhs.0[i]))
        }
        fn mul(self, rhs: Self) -> Self {
            Self(core::array::from_fn(|i| self.0[i] * rhs.0[i]))
        }
        fn fma(self, a: Self, b: Self) -> Self {
            self.add(a.mul(b))
        }
        fn fms(self, a: Self, b: Self) -> Self {
            self.sub(a.mul(b))
        }
        fn max(self, rhs: Self) -> Self {
            Self(core::array::from_fn(|i| self.0[i].max(rhs.0[i])))
        }
        fn min(self, rhs: Self) -> Self {
            Self(core::array::from_fn(|i| self.0[i].min(rhs.0[i])))
        }
        fn abs(self) -> Self {
            Self(self.0.map(f32::abs))
        }
        fn horizontal_sum(self) -> f32 {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
    }

    #[test]
    fn test_scrub_flush_const_bounds_and_default_trait_vec4() {
        let sub_bits = [
            0x0000_0000_u32,
            0x8000_0000,
            0x0000_0001,
            0x807F_FFFF,
            0x0080_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0xFF80_0001,
        ];
        let flush_expected = [
            0x0000_0000_u32,
            0x8000_0000,
            0x0000_0000,
            0x8000_0000,
            0x0080_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0xFF80_0001,
        ];
        let scrub_expected = [
            0x0000_0000_u32,
            0x8000_0000,
            0x0000_0000,
            0x8000_0000,
            0x0080_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
        ];
        for (i, &b) in sub_bits.iter().enumerate() {
            let f = f32::from_bits(b);
            assert_eq!(flush_subnormal_f32_bits(f).to_bits(), flush_expected[i]);
            assert_eq!(scrub_and_flush_f32_bits(f).to_bits(), scrub_expected[i]);
        }

        assert!(
            const_abs_f32(const_exp2_poly_f32(-0.5) - core::f32::consts::FRAC_1_SQRT_2) < 2.5e-7
        );
        assert_eq!(const_exp2_poly_f32(0.0), 1.0);
        assert!(const_abs_f32(const_exp2_poly_f32(0.5) - core::f32::consts::SQRT_2) < 2.5e-7);
        assert!(const_abs_f32(const_rcp_nr2_f32(3.0) * 3.0 - 1.0) < 1.0e-5);
        assert!(const_abs_f32(const_rcp_nr3_f32(3.0) * 3.0 - 1.0) < 1.0e-6);
        assert!(
            const_abs_f32(const_rsqrt_nr2_f32(2.0) - core::f32::consts::FRAC_1_SQRT_2) < 2.5e-3
        );
        assert!(
            const_abs_f32(const_rsqrt_nr3_f32(2.0) - core::f32::consts::FRAC_1_SQRT_2) < 1.0e-6
        );

        let dv = DefaultTraitVec4::from_array([1.0, -2.0, 3.0, -4.0]);
        assert_eq!(DefaultTraitVec4::zero().to_array(), [0.0; 4]);
        assert_eq!(
            DefaultTraitVec4::splat_pair(1.5, -2.5).to_array(),
            [1.5, -2.5, 1.5, -2.5]
        );
        assert_eq!(
            dv.clamp(DefaultTraitVec4::splat(-2.5), DefaultTraitVec4::splat(2.0))
                .to_array(),
            [1.0, -2.0, 2.0, -2.5]
        );
        assert_eq!(dv.neg().to_array(), [-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(dv.pair_swap_neg().to_array(), [2.0, 1.0, 4.0, 3.0]);
        assert_eq!(dv.dot(dv), 30.0);
        assert_eq!(dv.horizontal_sum_pair(dv), (-2.0, -2.0));
        assert_eq!(
            dv.mul_lane::<2>(DefaultTraitVec4::splat(2.0)).to_array(),
            [2.0, -4.0, 6.0, -8.0]
        );
        assert_eq!(
            dv.fma_lane::<1>(DefaultTraitVec4::splat(10.0), dv)
                .to_array(),
            [-19.0, -22.0, -17.0, -24.0]
        );
        assert_eq!(
            DefaultTraitVec4::select_le(
                dv,
                DefaultTraitVec4::zero(),
                DefaultTraitVec4::splat(1.0),
                DefaultTraitVec4::splat(-1.0)
            )
            .to_array(),
            [-1.0, 1.0, -1.0, 1.0]
        );
        assert!(dv.all_le(dv));
        assert!(!dv.all_le(DefaultTraitVec4::from_array([1.0, -2.5, 3.0, -4.0])));
        assert!(!DefaultTraitVec4::from_array([1.0, f32::NAN, 3.0, -4.0]).all_le(dv));

        let dirty = DefaultTraitVec4::from_array([
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::NAN,
            f32::INFINITY,
        ]);
        assert_eq!(
            dirty.flush_subnormal().to_array().map(f32::to_bits),
            [0, 0x8000_0000, f32::NAN.to_bits(), f32::INFINITY.to_bits()]
        );
        assert_eq!(
            dirty.flush_recursive_state().to_array().map(f32::to_bits),
            [0, 0x8000_0000, 0, 0]
        );

        let quad = DefaultTraitVec4::rcp_pos_quad_nr2([
            DefaultTraitVec4::splat(1.0),
            DefaultTraitVec4::splat(2.0),
            DefaultTraitVec4::splat(4.0),
            DefaultTraitVec4::splat(8.0),
        ]);
        assert!((quad[2].0[0] - 0.25).abs() < 1.0e-5);
        let (ra, rb) = DefaultTraitVec4::rsqrt_pos_pair_nr2(
            DefaultTraitVec4::splat(4.0),
            DefaultTraitVec4::splat(9.0),
        );
        assert!((ra.0[0] - 0.5).abs() < 2.5e-3);
        assert!((rb.0[0] - 1.0 / 3.0).abs() < 2.5e-3);
        assert_eq!(
            DefaultTraitVec4::from_array([-5.0, -1.0, 0.0, 5.0].map(|k| k + BIASED_INT_MAGIC_F32))
                .pow2_from_biased_int_bits()
                .to_array(),
            [0.03125, 0.5, 1.0, 32.0]
        );
        assert_eq!(
            dv.exp2_clamped5().to_array(),
            GpuVec4::from_array(dv.0).exp2_clamped5().to_array()
        );
        let nan_in = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0];
        assert_eq!(
            DefaultTraitVec4::from_array(nan_in)
                .exp2_clamped5()
                .to_array()
                .map(f32::to_bits),
            GpuVec4::from_array(nan_in)
                .exp2_clamped5()
                .to_array()
                .map(f32::to_bits)
        );
        assert_eq!(
            GpuVec4::from_array(nan_in).exp2_clamped5().to_array(),
            [0.03125, 0.03125, 32.0, 1.0]
        );
    }

    proptest! {
        #[test]
        fn prop_default_trait_vec4_matches_gpu_vec4(
            xs in prop::array::uniform4(1.0e-5_f32..1.0e5_f32),
            exps in prop::array::uniform4(-8.0_f32..8.0_f32),
            rhs in prop::array::uniform4(-10.0_f32..10.0_f32),
        ) {
            let dv = DefaultTraitVec4::from_array(xs);
            let gv = GpuVec4::from_array(xs);
            prop_assert_eq!(dv.rcp_pos_nr2().to_array(), gv.rcp_pos_nr2().to_array());
            prop_assert_eq!(dv.rcp_pos_nr3().to_array(), gv.rcp_pos_nr3().to_array());
            prop_assert_eq!(
                dv.add(DefaultTraitVec4::splat(1.0)).rcp_ge1_nr2().to_array(),
                gv.add(GpuVec4::splat(1.0)).rcp_ge1_nr2().to_array()
            );
            prop_assert_eq!(dv.rsqrt_pos_nr2().to_array(), gv.rsqrt_pos_nr2().to_array());
            prop_assert_eq!(dv.rsqrt_pos_nr3().to_array(), gv.rsqrt_pos_nr3().to_array());

            let ed = DefaultTraitVec4::from_array(exps).exp2_clamped5().to_array();
            let eg = GpuVec4::from_array(exps).exp2_clamped5().to_array();
            prop_assert_eq!(ed, eg);
            prop_assert_eq!(
                DefaultTraitVec4::from_array(exps).all_le(DefaultTraitVec4::from_array(rhs)),
                GpuVec4::from_array(exps).all_le(GpuVec4::from_array(rhs))
            );
        }
    }
}
