//! Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) hardware math backend
//! (`Sve2Vec4` & `ArmSve2Backend`).
//!
//! Uses `z0..z31` and `p0` (`ptrue p0.s, vl4`) SVE2 instructions directly when `FEAT_SVE2`
//! is detected at runtime via [`std::arch::is_aarch64_feature_detected!`], falling back to
//! [`ScalarVec4`] / [`ScalarBackend`] when `FEAT_SVE2` is absent (zero `NeonBackend` dependency).

use core::arch::aarch64::float32x4_t;
use dnam_kernels::InterleavedStereo8;

mod backend;
pub use backend::ArmSve2Backend;
pub(crate) use backend::sve2_tracker_exp_scales_f32;

use crate::math::{ScalarVec4, SimdVec4};

/// 128-bit 4-lane `f32` vector backed by the low 128 bits (`vl4`) of an SVE2 `z0..z31` register.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sve2Vec4(pub [f32; 4]);

/// SVE2 stereo 8-lane vector (`lanes 0..3 = Left`, `4..7 = Right`, VL-safe `InterleavedStereo8`).
pub type Sve2Stereo8 = InterleavedStereo8<Sve2Vec4>;

#[inline(always)]
pub(super) fn to_zreg(v: [f32; 4]) -> float32x4_t {
    zerocopy::transmute!(v)
}

#[inline(always)]
pub(super) fn from_zreg(v: float32x4_t) -> [f32; 4] {
    zerocopy::transmute!(v)
}

impl SimdVec4 for Sve2Vec4 {
    #[inline]
    fn splat(val: f32) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\ndup z0.s, z0.s[0]",
                    in("s0") val,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([val; 4])
        }
    }

    #[inline]
    fn splat_pair(a: f32, b: f32) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nzip1 z2.s, z0.s, z1.s\ndup z0.d, z2.d[0]",
                    in("s0") a,
                    in("s1") b,
                    out("q2") _,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([a, b, a, b])
        }
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        *dst = self.0;
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfadd z0.s, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).add(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfsub z0.s, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).sub(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfmul z0.s, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).mul(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfmla z0.s, p0/m, z1.s, z2.s",
                    inout("q0") io,
                    in("q1") va,
                    in("q2") vb,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(ScalarVec4(self.0).fma(ScalarVec4(a.0), ScalarVec4(b.0)).0)
        }
    }

    #[inline]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vv) = (to_zreg(self.0), to_zreg(a.0), to_zreg(v.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                match LANE {
                    0 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[0]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    1 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[1]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    2 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[2]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    _ => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.s, z1.s, z2.s[3]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                }
            }
            Self(from_zreg(io))
        } else {
            Self(
                ScalarVec4(self.0)
                    .fma_lane::<LANE>(ScalarVec4(a.0), ScalarVec4(v.0))
                    .0,
            )
        }
    }

    #[inline]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        const { assert!(LANE < 4) };
        if ArmSve2Backend::is_supported_cached() {
            let (va, vv) = (to_zreg(self.0), to_zreg(v.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                match LANE {
                    0 => core::arch::asm!(
                        ".arch armv9-a+sve2",
                        "fmul z0.s, z1.s, z2.s[0]",
                        lateout("q0") out,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    1 => core::arch::asm!(
                        ".arch armv9-a+sve2",
                        "fmul z0.s, z1.s, z2.s[1]",
                        lateout("q0") out,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    2 => core::arch::asm!(
                        ".arch armv9-a+sve2",
                        "fmul z0.s, z1.s, z2.s[2]",
                        lateout("q0") out,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    _ => core::arch::asm!(
                        ".arch armv9-a+sve2",
                        "fmul z0.s, z1.s, z2.s[3]",
                        lateout("q0") out,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                }
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).mul_lane::<LANE>(ScalarVec4(v.0)).0)
        }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfmls z0.s, p0/m, z1.s, z2.s",
                    inout("q0") io,
                    in("q1") va,
                    in("q2") vb,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(ScalarVec4(self.0).fms(ScalarVec4(a.0), ScalarVec4(b.0)).0)
        }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfmaxnm z0.s, p0/m, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).max(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfminnm z0.s, p0/m, z0.s, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).min(ScalarVec4(rhs.0)).0)
        }
    }

    #[inline]
    fn abs(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfabs z0.s, p0/m, z0.s",
                    in("q0") inp,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).abs().0)
        }
    }

    #[inline]
    fn pair_swap_neg(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let mask: float32x4_t = zerocopy::transmute!([0x8000_0000_u32, 0, 0x8000_0000_u32, 0]);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nrevw z0.d, p0/m, z0.d\neor z0.d, z0.d, z1.d",
                    in("q0") inp,
                    in("q1") mask,
                    lateout("q0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([-self.0[1], self.0[0], -self.0[3], self.0[2]])
        }
    }

    #[inline]
    fn horizontal_sum(self) -> f32 {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let out: f32;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfaddv s0, p0, z0.s",
                    in("q0") inp,
                    lateout("s0") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            out
        } else {
            ScalarVec4(self.0).horizontal_sum()
        }
    }

    #[inline]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let (out_l, out_r): (f32, f32);
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.s, vl4\nfaddv s0, p0, z0.s\nfaddv s1, p0, z1.s",
                    in("q0") a,
                    in("q1") b,
                    lateout("s0") out_l,
                    lateout("s1") out_r,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            (out_l, out_r)
        } else {
            (
                ScalarVec4(self.0).horizontal_sum(),
                ScalarVec4(rhs.0).horizontal_sum(),
            )
        }
    }

    #[inline]
    fn rcp_pos_seed_bits(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let magic: float32x4_t = zerocopy::transmute!([crate::math::IEEE754_F32_RCP_SEED; 4]);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nsub z0.s, z1.s, z0.s",
                    in("q0") inp,
                    in("q1") magic,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).rcp_pos_seed_bits().0)
        }
    }

    #[inline]
    fn rsqrt_pos_seed_bits(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let magic: float32x4_t = zerocopy::transmute!([crate::math::IEEE754_F32_RSQRT_SEED; 4]);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nlsr z0.s, z0.s, #1\nsub z0.s, z1.s, z0.s",
                    in("q0") inp,
                    in("q1") magic,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).rsqrt_pos_seed_bits().0)
        }
    }

    #[inline]
    fn pow2_from_biased_int_bits(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let bias: float32x4_t = zerocopy::transmute!([127_u32; 4]);
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "add z0.s, z0.s, z1.s",
                    "lsl z0.s, z0.s, #23",
                    in("q0") inp,
                    in("q1") bias,
                    lateout("q0") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarVec4(self.0).pow2_from_biased_int_bits().0)
        }
    }

    #[inline]
    fn exp2_clamped5(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::exp2_clamped5(self)
    }

    #[inline]
    fn scrub_recursive_state(self) -> Self {
        <Self as dnam_kernels::KernelVec4>::flush_recursive_state(self)
    }
}

impl dnam_kernels::KernelVec4 for Sve2Vec4 {
    #[inline]
    fn splat(val: f32) -> Self {
        <Self as SimdVec4>::splat(val)
    }

    #[inline]
    fn splat_pair(a: f32, b: f32) -> Self {
        <Self as SimdVec4>::splat_pair(a, b)
    }

    #[inline(always)]
    fn from_array(arr: [f32; 4]) -> Self {
        <Self as SimdVec4>::from_array(arr)
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        <Self as SimdVec4>::to_array(self)
    }

    #[inline(always)]
    fn load(src: &[f32; 4]) -> Self {
        <Self as SimdVec4>::load(src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f32; 4]) {
        <Self as SimdVec4>::store(self, dst);
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        <Self as SimdVec4>::add(self, rhs)
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        <Self as SimdVec4>::sub(self, rhs)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        <Self as SimdVec4>::mul(self, rhs)
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        <Self as SimdVec4>::fma(self, a, b)
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        <Self as SimdVec4>::fms(self, a, b)
    }

    #[inline]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        <Self as SimdVec4>::fma_lane::<LANE>(self, a, v)
    }

    #[inline]
    fn mul_lane<const LANE: usize>(self, v: Self) -> Self {
        <Self as SimdVec4>::mul_lane::<LANE>(self, v)
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        <Self as SimdVec4>::max(self, rhs)
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        <Self as SimdVec4>::min(self, rhs)
    }

    #[inline]
    fn abs(self) -> Self {
        <Self as SimdVec4>::abs(self)
    }

    #[inline]
    fn neg(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let mut io = to_zreg(self.0);
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "fneg z0.s, p0/m, z0.s",
                    inout("q0") io,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(<ScalarVec4 as dnam_kernels::KernelVec4>::neg(ScalarVec4(self.0)).0)
        }
    }

    #[inline]
    fn select_le(lhs: Self, rhs: Self, if_true: Self, if_false: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let out: float32x4_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "fcmle p1.s, p0/z, z0.s, z1.s",
                    "sel z0.s, p1, z2.s, z3.s",
                    in("q0") to_zreg(lhs.0),
                    in("q1") to_zreg(rhs.0),
                    in("q2") to_zreg(if_true.0),
                    in("q3") to_zreg(if_false.0),
                    lateout("q0") out,
                    out("p0") _,
                    out("p1") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(
                <ScalarVec4 as dnam_kernels::KernelVec4>::select_le(
                    ScalarVec4(lhs.0),
                    ScalarVec4(rhs.0),
                    ScalarVec4(if_true.0),
                    ScalarVec4(if_false.0),
                )
                .0,
            )
        }
    }

    #[inline]
    fn all_le(self, rhs: Self) -> bool {
        if ArmSve2Backend::is_supported_cached() {
            let cnt: u64;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "fcmle p1.s, p0/z, z0.s, z1.s",
                    "cntp {cnt}, p0, p1.s",
                    in("q0") to_zreg(self.0),
                    in("q1") to_zreg(rhs.0),
                    cnt = out(reg) cnt,
                    out("p0") _,
                    out("p1") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            cnt == 4
        } else {
            <ScalarVec4 as dnam_kernels::KernelVec4>::all_le(ScalarVec4(self.0), ScalarVec4(rhs.0))
        }
    }

    #[inline]
    fn pair_swap_neg(self) -> Self {
        <Self as SimdVec4>::pair_swap_neg(self)
    }

    #[inline(always)]
    fn flush_subnormal(self) -> Self {
        <Self as SimdVec4>::flush_subnormal(self)
    }

    #[inline]
    fn flush_recursive_state(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let mut io = to_zreg(self.0);
            let exp_mask: float32x4_t = zerocopy::transmute!([0x7F80_0000_u32; 4]);
            let sign_mask: float32x4_t = zerocopy::transmute!([0x8000_0000_u32; 4]);
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2",
                    "ptrue p0.s, vl4",
                    "and z1.d, z0.d, z3.d",
                    "cmpeq p1.s, p0/z, z1.s, z3.s",
                    "cmpeq p2.s, p0/z, z1.s, #0",
                    "and z0.s, p2/m, z0.s, z4.s",
                    "eor z0.s, p1/m, z0.s, z0.s",
                    inout("q0") io,
                    out("q1") _,
                    in("q3") exp_mask,
                    in("q4") sign_mask,
                    out("p0") _,
                    out("p1") _,
                    out("p2") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(
                <ScalarVec4 as dnam_kernels::KernelVec4>::flush_recursive_state(ScalarVec4(self.0))
                    .0,
            )
        }
    }

    #[inline]
    fn horizontal_sum(self) -> f32 {
        <Self as SimdVec4>::horizontal_sum(self)
    }

    #[inline]
    fn horizontal_sum_pair(self, rhs: Self) -> (f32, f32) {
        <Self as SimdVec4>::horizontal_sum_pair(self, rhs)
    }

    #[inline]
    fn dot(self, rhs: Self) -> f32 {
        <Self as SimdVec4>::dot(self, rhs)
    }

    #[inline]
    fn rcp_pos_seed_bits(self) -> Self {
        <Self as SimdVec4>::rcp_pos_seed_bits(self)
    }

    #[inline]
    fn rsqrt_pos_seed_bits(self) -> Self {
        <Self as SimdVec4>::rsqrt_pos_seed_bits(self)
    }

    #[inline]
    fn pow2_from_biased_int_bits(self) -> Self {
        <Self as SimdVec4>::pow2_from_biased_int_bits(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{ArmSve2Backend, Sve2Vec4};
    use crate::math::{BackendKind, MathBackend, SimdVec4};

    #[test]
    fn test_sve2_cached_feature_flag_and_vec_ops() {
        crate::math::warm_simd_feature_caches();
        assert_eq!(
            ArmSve2Backend::is_supported(),
            ArmSve2Backend::is_supported_cached()
        );
        assert_eq!(ArmSve2Backend::KIND, BackendKind::ArmSve2);
        let sum = Sve2Vec4::load(&[1.0, 2.0, 3.0, 4.0]).horizontal_sum();
        assert!((sum - 10.0).abs() <= 1e-6);
    }
}
