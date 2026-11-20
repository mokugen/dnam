//! Portable scalar/WGSL `[f32; 4]` backend ([`GpuVec4`]) implementing [`KernelVec4`] cleanly
//! across host CPUs and `spirv-unknown-naga-wgsl` compute shaders
//! (`ref[arch.simd.two-level-traits]`, `ref[sdk.rust.shared-nostd-kernels]`).

use super::{KernelVec4, scrub_and_flush_f32_bits};

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
    fn all_le(self, rhs: Self) -> bool {
        self.0[0] <= rhs.0[0]
            && self.0[1] <= rhs.0[1]
            && self.0[2] <= rhs.0[2]
            && self.0[3] <= rhs.0[3]
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        Self([
            scrub_and_flush_f32_bits(self.0[0]),
            scrub_and_flush_f32_bits(self.0[1]),
            scrub_and_flush_f32_bits(self.0[2]),
            scrub_and_flush_f32_bits(self.0[3]),
        ])
    }

    #[inline(always)]
    fn horizontal_sum(self) -> f32 {
        (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
    }
}

#[cfg(test)]
mod tests {
    use super::super::BIASED_INT_MAGIC_F32;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_gpu_vec4_arithmetic_all_le_and_flush_recursive_state() {
        let v = GpuVec4::from_array([1.0, -2.0, 3.0, -4.0]);
        assert_eq!(GpuVec4::zero().to_array(), [0.0; 4]);
        assert_eq!(
            GpuVec4::splat_pair(1.5, -2.5).to_array(),
            [1.5, -2.5, 1.5, -2.5]
        );
        assert_eq!(
            v.add(GpuVec4::splat(1.0)).to_array(),
            [2.0, -1.0, 4.0, -3.0]
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
        assert_eq!(v.neg().to_array(), [-1.0, 2.0, -3.0, 4.0]);
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

        assert!(v.all_le(v));
        assert!(v.all_le(GpuVec4::from_array([1.0, -1.5, 4.0, -4.0])));
        assert!(!v.all_le(GpuVec4::from_array([1.0, -2.1, 3.0, -4.0])));
        assert!(!GpuVec4::from_array([1.0, f32::NAN, 3.0, 4.0]).all_le(GpuVec4::splat(10.0)));
        assert!(!v.all_le(GpuVec4::from_array([10.0, 10.0, f32::NAN, 10.0])));

        let dirty = GpuVec4::from_array([
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::NAN,
            f32::NEG_INFINITY,
        ]);
        assert_eq!(
            dirty.flush_recursive_state().to_array().map(f32::to_bits),
            [0, 0x8000_0000, 0, 0]
        );
        assert_eq!(
            GpuVec4::from_array([f32::INFINITY, -1.25, 2.5, f32::from_bits(0x807F_FFFF)])
                .flush_recursive_state()
                .to_array()
                .map(f32::to_bits),
            [0, (-1.25_f32).to_bits(), 2.5_f32.to_bits(), 0x8000_0000]
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
    }

    proptest! {
        #[test]
        fn prop_gpu_vec4_rcp_rsqrt_exp2_and_all_le(
            xs in prop::array::uniform4(1.0e-5_f32..1.0e5_f32),
            exps in prop::array::uniform4(-8.0_f32..8.0_f32),
            rhs in prop::array::uniform4(-10.0_f32..10.0_f32),
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
            for i in 0..4 {
                let exact = f64::from(exps[i].clamp(-5.0, 5.0)).exp2();
                prop_assert!(((f64::from(e_gpu[i]) - exact) / exact).abs() < 1.0e-6);
            }

            let lhs_v = GpuVec4::from_array(exps);
            let rhs_v = GpuVec4::from_array(rhs);
            let expected = (0..4).all(|i| exps[i] <= rhs[i]);
            prop_assert_eq!(lhs_v.all_le(rhs_v), expected);
        }
    }
}
