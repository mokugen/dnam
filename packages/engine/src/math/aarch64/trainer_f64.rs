//! AArch64 NEON (`float64x2_t`) double-precision trainer SIMD kernels.

use core::arch::aarch64::{
    float32x2_t, float32x4_t, float64x2_t, vabsq_f64, vaddq_f64, vaddvq_f64, vcvt_f64_f32,
    vcvt_high_f64_f32, vdupq_n_f64, vfmaq_f64, vfmaq_laneq_f64, vfmsq_f64, vget_low_f32,
    vmaxnmq_f64, vminnmq_f64, vmulq_f64, vpaddq_f64, vsubq_f64,
};

use super::neon::NeonBackend;
use crate::math::trainer::{SimdF64Vec2, TrainerMathBackend};

/// 128-bit ARM64 NEON 2-lane `f64` register wrapper (`float64x2_t`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct NeonF64Vec2(pub float64x2_t);

#[inline(always)]
fn f64x2_from_f32x2(a: f32, b: f32) -> float64x2_t {
    let pair: float32x2_t = zerocopy::transmute!([a, b]);
    // SAFETY: `aarch64` guarantees baseline NEON `vcvt_f64_f32` availability.
    unsafe { vcvt_f64_f32(pair) }
}

impl SimdF64Vec2 for NeonF64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vdupq_n_f64(val)) }
    }

    #[inline(always)]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        Self(f64x2_from_f32x2(a, b))
    }

    #[inline(always)]
    fn load(src: &[f64; 2]) -> Self {
        Self(zerocopy::transmute!(*src))
    }

    #[inline(always)]
    fn store(self, dst: &mut [f64; 2]) {
        *dst = zerocopy::transmute!(self.0);
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vaddq_f64(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vsubq_f64(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vmulq_f64(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vfmaq_f64(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 2) };
        // SAFETY: `aarch64` guarantees baseline NEON availability and `LANE < 2`.
        unsafe {
            Self(match LANE {
                0 => vfmaq_laneq_f64::<0>(self.0, a.0, v.0),
                _ => vfmaq_laneq_f64::<1>(self.0, a.0, v.0),
            })
        }
    }

    #[inline(always)]
    fn mul_add_ieee(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability; separate mul + add avoids FMA.
        unsafe { Self(vaddq_f64(self.0, vmulq_f64(a.0, b.0))) }
    }

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vfmsq_f64(self.0, a.0, b.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vmaxnmq_f64(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vminnmq_f64(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // SAFETY: `aarch64` guarantees baseline NEON availability.
        unsafe { Self(vabsq_f64(self.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        // SAFETY: `aarch64` guarantees baseline NEON `vaddvq_f64` availability.
        unsafe { vaddvq_f64(self.0) }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f64, f64) {
        // SAFETY: `aarch64` guarantees baseline NEON `vpaddq_f64` availability.
        let paired = unsafe { vpaddq_f64(self.0, rhs.0) };
        let out: [f64; 2] = zerocopy::transmute!(paired);
        (out[0], out[1])
    }
}

impl TrainerMathBackend for NeonBackend {
    type F64Vec2 = NeonF64Vec2;

    #[inline(always)]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        let n = dst.len().min(a.len()).min(b.len());
        let (dc, dr) = dst[..n].as_chunks_mut::<4>();
        let (ac, ar) = a[..n].as_chunks::<4>();
        let (bc, br) = b[..n].as_chunks::<4>();
        for ((d4, a4), b4) in dc.iter_mut().zip(ac).zip(bc) {
            let va: float32x4_t = zerocopy::transmute!(*a4);
            let vb: float32x4_t = zerocopy::transmute!(*b4);
            // SAFETY: `aarch64` guarantees baseline NEON `vcvt_*_f64_f32` and `vsubq_f64`.
            let (lo, hi) = unsafe {
                let lo = vsubq_f64(
                    vcvt_f64_f32(vget_low_f32(va)),
                    vcvt_f64_f32(vget_low_f32(vb)),
                );
                let hi = vsubq_f64(vcvt_high_f64_f32(va), vcvt_high_f64_f32(vb));
                (lo, hi)
            };
            let lo_arr: [f64; 2] = zerocopy::transmute!(lo);
            let hi_arr: [f64; 2] = zerocopy::transmute!(hi);
            *d4 = [lo_arr[0], lo_arr[1], hi_arr[0], hi_arr[1]];
        }
        for ((d, &av), &bv) in dr.iter_mut().zip(ar).zip(br) {
            *d = f64::from(av) - f64::from(bv);
        }
    }

    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        let mut n = cols_i[0].len().min(cols_j[0].len());
        for k in 1..4 {
            n = n.min(cols_i[k].len()).min(cols_j[k].len());
        }
        // SAFETY: `aarch64` guarantees baseline NEON `float64x2_t` and `vfmaq_laneq_f64`.
        unsafe {
            let z = vdupq_n_f64(0.0);
            let mut acc = [[z, z]; 4];
            for t in 0..n {
                let ci01: float64x2_t = zerocopy::transmute!([cols_i[0][t], cols_i[1][t]]);
                let ci23: float64x2_t = zerocopy::transmute!([cols_i[2][t], cols_i[3][t]]);
                let cj01: float64x2_t = zerocopy::transmute!([cols_j[0][t], cols_j[1][t]]);
                let cj23: float64x2_t = zerocopy::transmute!([cols_j[2][t], cols_j[3][t]]);
                acc[0][0] = vfmaq_laneq_f64::<0>(acc[0][0], cj01, ci01);
                acc[0][1] = vfmaq_laneq_f64::<0>(acc[0][1], cj23, ci01);
                acc[1][0] = vfmaq_laneq_f64::<1>(acc[1][0], cj01, ci01);
                acc[1][1] = vfmaq_laneq_f64::<1>(acc[1][1], cj23, ci01);
                acc[2][0] = vfmaq_laneq_f64::<0>(acc[2][0], cj01, ci23);
                acc[2][1] = vfmaq_laneq_f64::<0>(acc[2][1], cj23, ci23);
                acc[3][0] = vfmaq_laneq_f64::<1>(acc[3][0], cj01, ci23);
                acc[3][1] = vfmaq_laneq_f64::<1>(acc[3][1], cj23, ci23);
            }
            let mut out = [[0.0_f64; 4]; 4];
            for r in 0..4 {
                let lo: [f64; 2] = zerocopy::transmute!(acc[r][0]);
                let hi: [f64; 2] = zerocopy::transmute!(acc[r][1]);
                out[r] = [lo[0], lo[1], hi[0], hi[1]];
            }
            out
        }
    }

    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        let n = cols[0]
            .len()
            .min(cols[1].len())
            .min(cols[2].len())
            .min(cols[3].len());
        let pairs = n / 2;
        // SAFETY: `aarch64` guarantees baseline NEON `vfmaq_f64` and `vpaddq_f64`.
        unsafe {
            let mut a = [vdupq_n_f64(0.0); 10];
            let mut t = 0usize;
            for _ in 0..pairs {
                let c0: float64x2_t = zerocopy::transmute!([cols[0][t], cols[0][t + 1]]);
                let c1: float64x2_t = zerocopy::transmute!([cols[1][t], cols[1][t + 1]]);
                let c2: float64x2_t = zerocopy::transmute!([cols[2][t], cols[2][t + 1]]);
                let c3: float64x2_t = zerocopy::transmute!([cols[3][t], cols[3][t + 1]]);
                a[0] = vfmaq_f64(a[0], c0, c0);
                a[1] = vfmaq_f64(a[1], c0, c1);
                a[2] = vfmaq_f64(a[2], c1, c1);
                a[3] = vfmaq_f64(a[3], c0, c2);
                a[4] = vfmaq_f64(a[4], c1, c2);
                a[5] = vfmaq_f64(a[5], c2, c2);
                a[6] = vfmaq_f64(a[6], c0, c3);
                a[7] = vfmaq_f64(a[7], c1, c3);
                a[8] = vfmaq_f64(a[8], c2, c3);
                a[9] = vfmaq_f64(a[9], c3, c3);
                t += 2;
            }
            let p01: [f64; 2] = zerocopy::transmute!(vpaddq_f64(a[0], a[1]));
            let p23: [f64; 2] = zerocopy::transmute!(vpaddq_f64(a[2], a[3]));
            let p45: [f64; 2] = zerocopy::transmute!(vpaddq_f64(a[4], a[5]));
            let p67: [f64; 2] = zerocopy::transmute!(vpaddq_f64(a[6], a[7]));
            let p89: [f64; 2] = zerocopy::transmute!(vpaddq_f64(a[8], a[9]));
            let mut out = [
                p01[0], p01[1], p23[0], p23[1], p45[0], p45[1], p67[0], p67[1], p89[0], p89[1],
            ];
            if t < n {
                let (x0, x1, x2, x3) = (cols[0][t], cols[1][t], cols[2][t], cols[3][t]);
                out[0] += x0 * x0;
                out[1] += x0 * x1;
                out[2] += x1 * x1;
                out[3] += x0 * x2;
                out[4] += x1 * x2;
                out[5] += x2 * x2;
                out[6] += x0 * x3;
                out[7] += x1 * x3;
                out[8] += x2 * x3;
                out[9] += x3 * x3;
            }
            out
        }
    }

    #[inline]
    fn advance_bezout7_lp_dim_f64(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    ) {
        let bound = dim.min(256);
        let chunks = bound / 4;
        // SAFETY: `aarch64` guarantees baseline NEON; separate `vmulq_f64` + `vaddq_f64` ensures `0-ULP` parity.
        unsafe {
            let rv = vdupq_n_f64(r);
            let omrv = vdupq_n_f64(omr);
            for c in 0..chunks {
                let d = c * 4;
                let fb4: float32x4_t =
                    zerocopy::transmute!([fb[d], fb[d + 1], fb[d + 2], fb[d + 3]]);
                let mut prev0 = vcvt_f64_f32(vget_low_f32(fb4));
                let mut prev1 = vcvt_high_f64_f32(fb4);
                for st in u.iter_mut() {
                    let s0: float64x2_t = zerocopy::transmute!([st[d], st[d + 1]]);
                    let s1: float64x2_t = zerocopy::transmute!([st[d + 2], st[d + 3]]);
                    prev0 = vaddq_f64(vmulq_f64(omrv, prev0), vmulq_f64(rv, s0));
                    prev1 = vaddq_f64(vmulq_f64(omrv, prev1), vmulq_f64(rv, s1));
                    let out0: [f64; 2] = zerocopy::transmute!(prev0);
                    let out1: [f64; 2] = zerocopy::transmute!(prev1);
                    st[d] = out0[0];
                    st[d + 1] = out0[1];
                    st[d + 2] = out1[0];
                    st[d + 3] = out1[1];
                }
            }
        }
        for d in (chunks * 4)..bound {
            let mut prev = f64::from(fb[d]);
            for st in u.iter_mut() {
                prev = omr * prev + r * st[d];
                st[d] = prev;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::aarch64::{ArmSmeBackend, ArmSve2Backend, Sve2F64Vec2};
    use crate::math::trainer::verify_trainer_backend_parity;

    #[test]
    fn test_aarch64_trainer_math_backends_parity() {
        verify_trainer_backend_parity::<NeonBackend>();
        verify_trainer_backend_parity::<ArmSve2Backend>();
        verify_trainer_backend_parity::<ArmSmeBackend>();
        if ArmSmeBackend::is_f64f64_supported() {
            assert!(ArmSmeBackend::is_supported());
        }
        assert_eq!(Sve2F64Vec2::splat(2.5).to_array(), [2.5, 2.5]);
    }
}
