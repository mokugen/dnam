//! Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) double-precision (`f64`)
//! trainer SIMD kernels (`Sve2F64Vec2` & `TrainerMathBackend` for `ArmSve2Backend`).
//!
//! Executes native Vector-Length-Agnostic (`whilelt p0.d`) SVE2 `f64` loops when `FEAT_SVE2`
//! is detected at runtime, falling back cleanly to [`ScalarF64Vec2`] / [`ScalarBackend`] when
//! `FEAT_SVE2` is absent (zero `NeonBackend` dependency).

use core::arch::aarch64::float64x2_t;

mod backend;

use super::sve2::ArmSve2Backend;
use crate::math::trainer::{ScalarF64Vec2, SimdF64Vec2};

/// 128-bit 2-lane `f64` vector backed by an SVE2 `z0..z31.d` register (`vl2`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sve2F64Vec2(pub [f64; 2]);

#[inline(always)]
pub(super) fn to_zreg(v: [f64; 2]) -> float64x2_t {
    zerocopy::transmute!(v)
}

#[inline(always)]
pub(super) fn from_zreg(v: float64x2_t) -> [f64; 2] {
    zerocopy::transmute!(v)
}

impl SimdF64Vec2 for Sve2F64Vec2 {
    #[inline]
    fn splat(val: f64) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\ndup z1.d, z0.d[0]",
                    in("d0") val,
                    lateout("q1") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([val; 2])
        }
    }

    #[inline]
    fn splat_pair(a: f64, b: f64) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nzip1 z2.d, z0.d, z1.d",
                    in("d0") a,
                    in("d1") b,
                    lateout("q2") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([a, b])
        }
    }

    #[inline]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nzip1 z2.d, z0.d, z1.d\nfcvt z2.d, p0/m, z2.s",
                    in("s0") a,
                    in("s1") b,
                    lateout("q2") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self([f64::from(a), f64::from(b)])
        }
    }

    #[inline(always)]
    fn load(src: &[f64; 2]) -> Self {
        Self(*src)
    }

    #[inline(always)]
    fn store(self, dst: &mut [f64; 2]) {
        *dst = self.0;
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfadd z2.d, z0.d, z1.d",
                    in("q0") a,
                    in("q1") b,
                    lateout("q2") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarF64Vec2(self.0).add(ScalarF64Vec2(rhs.0)).0)
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfsub z2.d, z0.d, z1.d",
                    in("q0") a,
                    in("q1") b,
                    lateout("q2") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarF64Vec2(self.0).sub(ScalarF64Vec2(rhs.0)).0)
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfmul z2.d, z0.d, z1.d",
                    in("q0") a,
                    in("q1") b,
                    lateout("q2") out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarF64Vec2(self.0).mul(ScalarF64Vec2(rhs.0)).0)
        }
    }

    #[inline]
    fn fma(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfmla z0.d, p0/m, z1.d, z2.d",
                    inout("q0") io,
                    in("q1") va,
                    in("q2") vb,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(
                ScalarF64Vec2(self.0)
                    .fma(ScalarF64Vec2(a.0), ScalarF64Vec2(b.0))
                    .0,
            )
        }
    }

    #[inline]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 2) };
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vv) = (to_zreg(self.0), to_zreg(a.0), to_zreg(v.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2` and `LANE < 2`.
            unsafe {
                match LANE {
                    0 => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.d, z1.d, z2.d[0]",
                        inout("q0") io,
                        in("q1") va,
                        in("q2") vv,
                        options(pure, nomem, nostack, preserves_flags),
                    ),
                    _ => core::arch::asm!(
                        ".arch armv9-a+sve2\nfmla z0.d, z1.d, z2.d[1]",
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
                ScalarF64Vec2(self.0)
                    .fma_lane::<LANE>(ScalarF64Vec2(a.0), ScalarF64Vec2(v.0))
                    .0,
            )
        }
    }

    #[inline]
    fn mul_add_ieee(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`;
            // separate `fmul` + `fadd` prevents FMA contraction for `0-ULP` parity.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nfmul z2.d, z1.d, z2.d\nfadd z0.d, z0.d, z2.d",
                    inout("q0") io,
                    in("q1") va,
                    inout("q2") vb => _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(
                ScalarF64Vec2(self.0)
                    .mul_add_ieee(ScalarF64Vec2(a.0), ScalarF64Vec2(b.0))
                    .0,
            )
        }
    }

    #[inline]
    fn fms(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfmls z0.d, p0/m, z1.d, z2.d",
                    inout("q0") io,
                    in("q1") va,
                    in("q2") vb,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(
                ScalarF64Vec2(self.0)
                    .fms(ScalarF64Vec2(a.0), ScalarF64Vec2(b.0))
                    .0,
            )
        }
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, b) = (to_zreg(self.0), to_zreg(rhs.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfmaxnm z0.d, p0/m, z0.d, z1.d",
                    inout("q0") io,
                    in("q1") b,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(ScalarF64Vec2(self.0).max(ScalarF64Vec2(rhs.0)).0)
        }
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let (mut io, b) = (to_zreg(self.0), to_zreg(rhs.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfminnm z0.d, p0/m, z0.d, z1.d",
                    inout("q0") io,
                    in("q1") b,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(io))
        } else {
            Self(ScalarF64Vec2(self.0).min(ScalarF64Vec2(rhs.0)).0)
        }
    }

    #[inline]
    fn abs(self) -> Self {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfabs z1.d, p0/m, z0.d",
                    in("q0") inp,
                    lateout("q1") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(from_zreg(out))
        } else {
            Self(ScalarF64Vec2(self.0).abs().0)
        }
    }

    #[inline]
    fn horizontal_sum(self) -> f64 {
        if ArmSve2Backend::is_supported_cached() {
            let inp = to_zreg(self.0);
            let out: f64;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfaddv d1, p0, z0.d",
                    in("q0") inp,
                    lateout("d1") out,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            out
        } else {
            ScalarF64Vec2(self.0).horizontal_sum()
        }
    }

    #[inline]
    fn horizontal_sum_pair(self, rhs: Self) -> (f64, f64) {
        if ArmSve2Backend::is_supported_cached() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let (out_l, out_r): (f64, f64);
            // SAFETY: Guarded by `ArmSve2Backend::is_supported_cached()` confirming `FEAT_SVE2`.
            unsafe {
                core::arch::asm!(
                    ".arch armv9-a+sve2\nptrue p0.d, vl2\nfaddv d2, p0, z0.d\nfaddv d3, p0, z1.d",
                    in("q0") a,
                    in("q1") b,
                    lateout("d2") out_l,
                    lateout("d3") out_r,
                    out("p0") _,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            (out_l, out_r)
        } else {
            (
                ScalarF64Vec2(self.0).horizontal_sum(),
                ScalarF64Vec2(rhs.0).horizontal_sum(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sve2F64Vec2;
    use crate::math::ArmSve2Backend;
    use crate::math::trainer::SimdF64Vec2;

    #[test]
    fn test_sve2_f64_cached_feature_flag_and_sum() {
        crate::math::warm_simd_feature_caches();
        assert_eq!(
            ArmSve2Backend::is_supported(),
            ArmSve2Backend::is_supported_cached()
        );
        let sum = Sve2F64Vec2::load(&[1.5, 2.5]).horizontal_sum();
        assert!((sum - 4.0).abs() <= 1e-12);
    }
}
