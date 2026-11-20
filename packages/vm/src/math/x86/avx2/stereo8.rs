//! Wide 256-bit stereo (`4L + 4R`) vector implementation ([`Avx2Stereo8`]) implementing
//! [`dnam_kernels::StereoVec8`] with exact `0-ULP` parity against interleaved [`super::Avx2FmaVec4`].

use dnam_kernels::{KernelVec4, StereoVec8};

use super::Avx2FmaVec4;

/// 256-bit aligned 8-lane stereo vector (`lanes 0..3 = Left`, `4..7 = Right`), pairing two
/// [`Avx2FmaVec4`] registers so horizontal reductions preserve bit-exact `0-ULP` parity with
/// interleaved `2 × Vec4` targets (`NEON`, `WASM SIMD128`, `GpuVec4`) while retaining hardware
/// `AVX2+FMA+F16C` fused arithmetic (`ref[arch.simd.two-level-traits]`).
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug)]
pub struct Avx2Stereo8 {
    /// Left 4-lane vector (`lanes 0..3`).
    pub left: Avx2FmaVec4,
    /// Right 4-lane vector (`lanes 4..7`).
    pub right: Avx2FmaVec4,
}

impl StereoVec8 for Avx2Stereo8 {
    type Half = Avx2FmaVec4;

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
            left: Avx2FmaVec4::select_le(lhs.left, rhs.left, if_true.left, if_false.left),
            right: Avx2FmaVec4::select_le(lhs.right, rhs.right, if_true.right, if_false.right),
        }
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        Self {
            left: KernelVec4::flush_subnormal(self.left),
            right: KernelVec4::flush_subnormal(self.right),
        }
    }

    #[inline(always)]
    fn flush_recursive_state(self) -> Self {
        Self {
            left: KernelVec4::flush_recursive_state(self.left),
            right: KernelVec4::flush_recursive_state(self.right),
        }
    }

    #[inline(always)]
    fn horizontal_sum_pair(self) -> (f32, f32) {
        (
            KernelVec4::horizontal_sum(self.left),
            KernelVec4::horizontal_sum(self.right),
        )
    }

    #[inline(always)]
    fn rcp_pos_seed_bits(self) -> Self {
        Self {
            left: KernelVec4::rcp_pos_seed_bits(self.left),
            right: KernelVec4::rcp_pos_seed_bits(self.right),
        }
    }

    #[inline(always)]
    fn rsqrt_pos_seed_bits(self) -> Self {
        Self {
            left: KernelVec4::rsqrt_pos_seed_bits(self.left),
            right: KernelVec4::rsqrt_pos_seed_bits(self.right),
        }
    }

    #[inline(always)]
    fn pow2_from_biased_int_bits(self) -> Self {
        Self {
            left: KernelVec4::pow2_from_biased_int_bits(self.left),
            right: KernelVec4::pow2_from_biased_int_bits(self.right),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_stereo8_lr_roundtrip() {
        let l = [1.0, 2.0, 3.0, 4.0];
        let r = [-1.0, -2.0, -3.0, -4.0];
        let s = Avx2Stereo8::load_lr(&l, &r);
        assert_eq!(s.horizontal_sum_pair(), (10.0, -10.0));
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 4];
        s.abs().store_lr(&mut out_l, &mut out_r);
        assert_eq!(out_l, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out_r, [1.0, 2.0, 3.0, 4.0]);
    }
}
