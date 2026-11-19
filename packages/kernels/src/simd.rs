//! Portable `#![no_std]` 4-lane (`KernelVec4` / `GpuVec4`) and 8-lane stereo (`StereoVec8`)
//! SIMD abstractions, compile-time mathematical constants, and `fp16` codecs
//! (`ref[arch.simd.two-level-traits]`, `ref[repo.code.shared-nostd-kernels]`).
//!
//! All runtime primitives execute with zero `FDIV`, zero `FSQRT`, zero `libm` calls, and zero
//! `u64`/`f64` types so CPU SIMD backends and 32-bit `spirv-unknown-naga-wgsl` shaders evaluate
//! bit-aligned arithmetic.

pub mod const_eval;
pub mod fp16;
pub mod stereo8;

pub use const_eval::{
    DC_BLOCKER_5HZ_48K_F32, DC_BLOCKER_10HZ_48K_BITS, FULL_SPEC, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_RCP_SEED, IEEE754_F32_RSQRT_SEED, LITE_SPEC, MAX_LOOP_GAIN_F32, MAX_PARALRU_R_F32,
    PRO_SPEC, SonaTierSpec, ULTRA_SPEC,
};
pub use fp16::{
    f32_to_fp16, f32_to_fp16_u32, fp16_to_f32, fp16_u32_to_f32, pack_fp16x2_u32, unpack_fp16x2_u32,
};
pub use stereo8::{InterleavedStereo8, StereoVec8};

/// IEEE-754 `1.5 * 2^23` (`12_582_912.0`) magic integer-rounding bias (`0x4B40_0000`).
const BIASED_INT_MAGIC_F32: f32 = 1.5 * ((1_u32 << 23) as f32);

/// Chebyshev-economized degree-5 coefficients for `2^f` on `f in [-0.5, 0.5]` (`< 2.2e-7` error;
/// `C1 = ln(2)`, `C2..C5` minimax-adjusted around `ln(2)^k / k!`,
/// `ref[repo.code.constant-eval-and-derivations]`).
const EXP2_C2: f32 = 0.240_221_09;
const EXP2_C3: f32 = 0.055_504_11;
const EXP2_C4: f32 = 0.009_675_892;
const EXP2_C5: f32 = 0.001_333_355_8;

const _: () = assert!(BIASED_INT_MAGIC_F32.to_bits() == 0x4B40_0000);

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

    /// Fast base-2 exponential `2^clamp(self, -5.0, 5.0)` (`< 2.2e-7`, `ref[dsp.op-n.polyadaa.midpoint]`).
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

/// Portable `[f32; 4]` vector implementing [`KernelVec4`] cleanly for both scalar CPU and `spirv` WGSL.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GpuVec4(pub [f32; 4]);

impl KernelVec4 for GpuVec4 {
    #[inline(always)]
    fn splat(val: f32) -> Self {
        Self([val; 4])
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0] + a.0[0] * b.0[0],
            self.0[1] + a.0[1] * b.0[1],
            self.0[2] + a.0[2] * b.0[2],
            self.0[3] + a.0[3] * b.0[3],
        ])
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0] - a.0[0] * b.0[0],
            self.0[1] - a.0[1] * b.0[1],
            self.0[2] - a.0[2] * b.0[2],
            self.0[3] - a.0[3] * b.0[3],
        ])
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self([
            self.0[0].max(rhs.0[0]),
            self.0[1].max(rhs.0[1]),
            self.0[2].max(rhs.0[2]),
            self.0[3].max(rhs.0[3]),
        ])
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self([
            self.0[0].min(rhs.0[0]),
            self.0[1].min(rhs.0[1]),
            self.0[2].min(rhs.0[2]),
            self.0[3].min(rhs.0[3]),
        ])
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self([
            f32::from_bits(self.0[0].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(self.0[1].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(self.0[2].to_bits() & 0x7FFF_FFFF),
            f32::from_bits(self.0[3].to_bits() & 0x7FFF_FFFF),
        ])
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    #[inline(always)]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        Self([
            if lhs.0[0] <= rhs.0[0] {
                if_true.0[0]
            } else {
                if_false.0[0]
            },
            if lhs.0[1] <= rhs.0[1] {
                if_true.0[1]
            } else {
                if_false.0[1]
            },
            if lhs.0[2] <= rhs.0[2] {
                if_true.0[2]
            } else {
                if_false.0[2]
            },
            if lhs.0[3] <= rhs.0[3] {
                if_true.0[3]
            } else {
                if_false.0[3]
            },
        ])
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }

    #[inline(always)]
    fn exp2_clamped5(self) -> Self {
        let mut out = [0.0_f32; 4];
        for i in 0..4 {
            let x = self.0[i].clamp(-5.0, 5.0);
            let k_biased = x + BIASED_INT_MAGIC_F32;
            let k = k_biased - BIASED_INT_MAGIC_F32;
            let f = x - k;
            let pow2_k =
                f32::from_bits(k_biased.to_bits().wrapping_add(IEEE754_F32_EXP_BIAS) << 23);
            let poly = EXP2_C4 + EXP2_C5 * f;
            let poly = EXP2_C3 + poly * f;
            let poly = EXP2_C2 + poly * f;
            let poly = core::f32::consts::LN_2 + poly * f;
            let poly = 1.0 + poly * f;
            out[i] = pow2_k * poly;
        }
        Self(out)
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
            self.add(a.mul(b))
        }
        fn fms(self, a: Self, b: Self) -> Self {
            self.sub(a.mul(b))
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
            Self(self.0.map(f32::abs))
        }
        fn horizontal_sum(self) -> f32 {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
    }

    #[test]
    fn test_flush_subnormal_and_kernel_vec4_ops() {
        let bits = [
            0x0000_0000_u32,
            0x8000_0000,
            0x0000_0001,
            0x807F_FFFF,
            0x0080_0000,
            0x7F80_0000,
        ];
        let expected = [
            0x0000_0000_u32,
            0x8000_0000,
            0x0000_0000,
            0x8000_0000,
            0x0080_0000,
            0x7F80_0000,
        ];
        for (i, b) in bits.iter().enumerate() {
            assert_eq!(
                flush_subnormal_f32_bits(f32::from_bits(*b)).to_bits(),
                expected[i]
            );
        }

        let v = GpuVec4::from_array([1.0, -2.0, 3.0, -4.0]);
        let dv = DefaultTraitVec4::from_array([1.0, -2.0, 3.0, -4.0]);
        assert_eq!(GpuVec4::zero().to_array(), [0.0; 4]);
        assert_eq!(
            GpuVec4::splat_pair(1.5, -2.5).to_array(),
            [1.5, -2.5, 1.5, -2.5]
        );
        assert_eq!(
            v.sub(GpuVec4::splat(1.0)).to_array(),
            [0.0, -3.0, 2.0, -5.0]
        );
        assert_eq!(
            v.fma(GpuVec4::splat(2.0), GpuVec4::splat(3.0)).to_array(),
            [7.0, 4.0, 9.0, 2.0]
        );
        assert_eq!(
            v.fms(GpuVec4::splat(2.0), GpuVec4::splat(3.0)).to_array(),
            [-5.0, -8.0, -3.0, -10.0]
        );
        assert_eq!(
            v.clamp(GpuVec4::splat(-2.5), GpuVec4::splat(2.0))
                .to_array(),
            [1.0, -2.0, 2.0, -2.5]
        );
        assert_eq!(
            dv.fms(DefaultTraitVec4::splat(2.0), DefaultTraitVec4::splat(3.0))
                .to_array(),
            [-5.0, -8.0, -3.0, -10.0]
        );
        assert_eq!(v.neg().to_array(), [-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(dv.neg().to_array(), [-1.0, 2.0, -3.0, 4.0]);
        assert_eq!(v.abs().to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.pair_swap_neg().to_array(), [2.0, 1.0, 4.0, 3.0]);
        assert_eq!(v.dot(v), 30.0);
        assert_eq!(v.horizontal_sum_pair(v), (-2.0, -2.0));
        assert_eq!(
            v.mul_lane::<2>(GpuVec4::splat(2.0)).to_array(),
            [2.0, -4.0, 6.0, -8.0]
        );
        assert_eq!(
            v.fma_lane::<1>(GpuVec4::splat(10.0), v).to_array(),
            [-19.0, -22.0, -17.0, -24.0]
        );
        assert_eq!(
            GpuVec4::select_le(
                v,
                GpuVec4::zero(),
                GpuVec4::splat(1.0),
                GpuVec4::splat(-1.0)
            )
            .to_array(),
            [-1.0, 1.0, -1.0, 1.0]
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
        assert_eq!(
            GpuVec4::from_array([f32::from_bits(1), f32::from_bits(0x8000_0001), 1.0, -1.0])
                .flush_subnormal()
                .to_array()
                .map(f32::to_bits),
            [0, 0x8000_0000, 1.0_f32.to_bits(), (-1.0_f32).to_bits()]
        );

        let quad = GpuVec4::rcp_pos_quad_nr2([
            GpuVec4::splat(1.0),
            GpuVec4::splat(2.0),
            GpuVec4::splat(4.0),
            GpuVec4::splat(8.0),
        ]);
        assert!((quad[0].0[0] - 1.0).abs() < 1.0e-5);
        assert!((quad[1].0[0] - 0.5).abs() < 1.0e-5);
        assert!((quad[2].0[0] - 0.25).abs() < 1.0e-5);
        assert!((quad[3].0[0] - 0.125).abs() < 1.0e-5);

        let (ra, rb) = GpuVec4::rsqrt_pos_pair_nr2(GpuVec4::splat(4.0), GpuVec4::splat(9.0));
        assert!((ra.0[0] - 0.5).abs() < 2.5e-3);
        assert!((rb.0[0] - 1.0 / 3.0).abs() < 2.5e-3);
        assert_eq!(
            GpuVec4::from_array([-5.0, -1.0, 0.0, 5.0].map(|k| k + BIASED_INT_MAGIC_F32))
                .pow2_from_biased_int_bits()
                .to_array(),
            [0.03125, 0.5, 1.0, 32.0]
        );
        assert_eq!(
            dv.exp2_clamped5().to_array(),
            GpuVec4::from_array(dv.0).exp2_clamped5().to_array()
        );
    }

    proptest! {
        #[test]
        fn prop_rcp_rsqrt_and_exp2_clamped5_accuracy(
            xs in prop::array::uniform4(1.0e-5_f32..1.0e5_f32),
            exps in prop::array::uniform4(-8.0_f32..8.0_f32),
        ) {
            let v = GpuVec4::from_array(xs);
            let r2 = v.rcp_pos_nr2().to_array();
            let r3 = v.rcp_pos_nr3().to_array();
            let g1 = v.add(GpuVec4::splat(1.0)).rcp_ge1_nr2().to_array();
            let s2 = v.rsqrt_pos_nr2().to_array();
            let s3 = v.rsqrt_pos_nr3().to_array();
            for i in 0..4 {
                let d = f64::from(xs[i]);
                prop_assert!((f64::from(r2[i]) * d - 1.0).abs() < 1.0e-5);
                prop_assert!((f64::from(r3[i]) * d - 1.0).abs() < 1.0e-6);
                prop_assert!((f64::from(g1[i]) * f64::from(xs[i] + 1.0) - 1.0).abs() < 1.0e-5);
                let exact_rsqrt = d.sqrt().recip();
                prop_assert!(((f64::from(s2[i]) - exact_rsqrt) / exact_rsqrt).abs() < 2.5e-3);
                prop_assert!(((f64::from(s3[i]) - exact_rsqrt) / exact_rsqrt).abs() < 2.0e-5);
            }

            let e_gpu = GpuVec4::from_array(exps).exp2_clamped5().to_array();
            let e_def = DefaultTraitVec4::from_array(exps).exp2_clamped5().to_array();
            for i in 0..4 {
                prop_assert_eq!(e_gpu[i], e_def[i]);
                let exact = f64::from(exps[i].clamp(-5.0, 5.0)).exp2();
                prop_assert!(((f64::from(e_gpu[i]) - exact) / exact).abs() < 1.0e-6);
            }
        }
    }
}
