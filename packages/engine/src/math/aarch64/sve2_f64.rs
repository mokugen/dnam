//! Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) double-precision (`f64`)
//! trainer SIMD kernels (`Sve2F64Vec2` & `TrainerMathBackend` for `ArmSve2Backend`).
//!
//! Executes native Vector-Length-Agnostic (`whilelt p0.d`) SVE2 `f64` loops when `FEAT_SVE2`
//! is detected at runtime, falling back cleanly to [`ScalarF64Vec2`] / [`ScalarBackend`] when
//! `FEAT_SVE2` is absent (zero `NeonBackend` dependency).

use core::arch::aarch64::float64x2_t;

use super::sve2::ArmSve2Backend;
use crate::math::ScalarBackend;
use crate::math::trainer::{ScalarF64Vec2, SimdF64Vec2, TrainerMathBackend};

/// 128-bit 2-lane `f64` vector backed by an SVE2 `z0..z31.d` register (`vl2`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sve2F64Vec2(pub [f64; 2]);

#[inline(always)]
fn to_zreg(v: [f64; 2]) -> float64x2_t {
    zerocopy::transmute!(v)
}

#[inline(always)]
fn from_zreg(v: float64x2_t) -> [f64; 2] {
    zerocopy::transmute!(v)
}

impl SimdF64Vec2 for Sve2F64Vec2 {
    #[inline(always)]
    fn splat(val: f64) -> Self {
        if ArmSve2Backend::is_supported() {
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn splat_pair(a: f64, b: f64) -> Self {
        if ArmSve2Backend::is_supported() {
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn from_f32_pair(a: f32, b: f32) -> Self {
        if ArmSve2Backend::is_supported() {
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn fma_lane<const LANE: usize>(self, a: Self, v: Self) -> Self {
        const { assert!(LANE < 2) };
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vv) = (to_zreg(self.0), to_zreg(a.0), to_zreg(v.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2` and `LANE < 2`.
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

    #[inline(always)]
    fn mul_add_ieee(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`;
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

    #[inline(always)]
    fn fms(self, a: Self, b: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, va, vb) = (to_zreg(self.0), to_zreg(a.0), to_zreg(b.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, b) = (to_zreg(self.0), to_zreg(rhs.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        if ArmSve2Backend::is_supported() {
            let (mut io, b) = (to_zreg(self.0), to_zreg(rhs.0));
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn abs(self) -> Self {
        if ArmSve2Backend::is_supported() {
            let inp = to_zreg(self.0);
            let out: float64x2_t;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn horizontal_sum(self) -> f64 {
        if ArmSve2Backend::is_supported() {
            let inp = to_zreg(self.0);
            let out: f64;
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

    #[inline(always)]
    fn horizontal_sum_pair(self, rhs: Self) -> (f64, f64) {
        if ArmSve2Backend::is_supported() {
            let (a, b) = (to_zreg(self.0), to_zreg(rhs.0));
            let (out_l, out_r): (f64, f64);
            // SAFETY: Guarded by `ArmSve2Backend::is_supported()` confirming `FEAT_SVE2`.
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

impl TrainerMathBackend for ArmSve2Backend {
    type F64Vec2 = Sve2F64Vec2;

    #[inline]
    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 || !Self::is_supported() {
            return ScalarBackend::dot_f64(a, b);
        }
        let out: f64;
        // SAFETY: `FEAT_SVE2` verified; `whilelt p0.d` bounds all loads to `< n <= min(a.len(), b.len())`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "mov z0.d, #0",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z1.d}}, p0/z, [{a}, {i}, lsl #3]",
                "ld1d {{z2.d}}, p0/z, [{b}, {i}, lsl #3]",
                "fmla z0.d, p0/m, z1.d, z2.d",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                "ptrue p0.d",
                "faddv d0, p0, z0.d",
                a = in(reg) a.as_ptr(),
                b = in(reg) b.as_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                lateout("d0") out,
                out("q1") _,
                out("q2") _,
                out("p0") _,
                options(readonly, nostack),
            );
        }
        out
    }

    #[inline]
    fn dot_pair_f64(x: &[f64], b0: &[f64], b1: &[f64]) -> (f64, f64) {
        let n = x.len().min(b0.len()).min(b1.len());
        if n == 0 || !Self::is_supported() {
            return ScalarBackend::dot_pair_f64(x, b0, b1);
        }
        let (s0, s1): (f64, f64);
        // SAFETY: `FEAT_SVE2` verified; `whilelt p0.d` bounds all loads to `< n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "mov z0.d, #0",
                "mov z1.d, #0",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z2.d}}, p0/z, [{x}, {i}, lsl #3]",
                "ld1d {{z3.d}}, p0/z, [{b0}, {i}, lsl #3]",
                "ld1d {{z4.d}}, p0/z, [{b1}, {i}, lsl #3]",
                "fmla z0.d, p0/m, z2.d, z3.d",
                "fmla z1.d, p0/m, z2.d, z4.d",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                "ptrue p0.d",
                "faddv d0, p0, z0.d",
                "faddv d1, p0, z1.d",
                x = in(reg) x.as_ptr(),
                b0 = in(reg) b0.as_ptr(),
                b1 = in(reg) b1.as_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                lateout("d0") s0,
                lateout("d1") s1,
                out("q2") _,
                out("q3") _,
                out("q4") _,
                out("p0") _,
                options(readonly, nostack),
            );
        }
        (s0, s1)
    }

    #[inline]
    fn sum_sq_f64(x: &[f64]) -> f64 {
        let n = x.len();
        if n == 0 || !Self::is_supported() {
            return ScalarBackend::sum_sq_f64(x);
        }
        let out: f64;
        // SAFETY: `FEAT_SVE2` verified; `whilelt p0.d` bounds all loads to `< n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "mov z0.d, #0",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z1.d}}, p0/z, [{x}, {i}, lsl #3]",
                "fmla z0.d, p0/m, z1.d, z1.d",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                "ptrue p0.d",
                "faddv d0, p0, z0.d",
                x = in(reg) x.as_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                lateout("d0") out,
                out("q1") _,
                out("p0") _,
                options(readonly, nostack),
            );
        }
        out
    }

    #[inline]
    fn axpy_f64(y: &mut [f64], alpha: f64, x: &[f64]) {
        let n = y.len().min(x.len());
        if n == 0 || !Self::is_supported() {
            ScalarBackend::axpy_f64(y, alpha, x);
            return;
        }
        // SAFETY: `FEAT_SVE2` verified; exclusive `y` and immutable `x` bounded to `< n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "dup z0.d, {alpha_bits}",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z1.d}}, p0/z, [{y}, {i}, lsl #3]",
                "ld1d {{z2.d}}, p0/z, [{x}, {i}, lsl #3]",
                "fmla z1.d, p0/m, z0.d, z2.d",
                "st1d {{z1.d}}, p0, [{y}, {i}, lsl #3]",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                alpha_bits = in(reg) alpha.to_bits(),
                y = in(reg) y.as_mut_ptr(),
                x = in(reg) x.as_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                out("q0") _,
                out("q1") _,
                out("q2") _,
                out("p0") _,
                options(nostack),
            );
        }
    }

    #[inline]
    fn axpy_pair_f64(y: &mut [f64], a0: f64, x0: &[f64], a1: f64, x1: &[f64]) {
        let n = y.len().min(x0.len()).min(x1.len());
        if n == 0 || !Self::is_supported() {
            ScalarBackend::axpy_pair_f64(y, a0, x0, a1, x1);
            return;
        }
        // SAFETY: `FEAT_SVE2` verified; exclusive `y` and immutable `x0`/`x1` bounded to `< n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "dup z0.d, {a0_bits}",
                "dup z1.d, {a1_bits}",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z2.d}}, p0/z, [{y}, {i}, lsl #3]",
                "ld1d {{z3.d}}, p0/z, [{x0}, {i}, lsl #3]",
                "ld1d {{z4.d}}, p0/z, [{x1}, {i}, lsl #3]",
                "fmla z2.d, p0/m, z0.d, z3.d",
                "fmla z2.d, p0/m, z1.d, z4.d",
                "st1d {{z2.d}}, p0, [{y}, {i}, lsl #3]",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                a0_bits = in(reg) a0.to_bits(),
                a1_bits = in(reg) a1.to_bits(),
                y = in(reg) y.as_mut_ptr(),
                x0 = in(reg) x0.as_ptr(),
                x1 = in(reg) x1.as_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                out("q0") _,
                out("q1") _,
                out("q2") _,
                out("q3") _,
                out("q4") _,
                out("p0") _,
                options(nostack),
            );
        }
    }

    #[inline]
    fn scale_f64(x: &mut [f64], scale: f64) {
        let n = x.len();
        if n == 0 || !Self::is_supported() {
            ScalarBackend::scale_f64(x, scale);
            return;
        }
        // SAFETY: `FEAT_SVE2` verified; all active lanes stay strictly within `< x.len()`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "dup z0.d, {scale_bits}",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z1.d}}, p0/z, [{x}, {i}, lsl #3]",
                "fmul z1.d, z1.d, z0.d",
                "st1d {{z1.d}}, p0, [{x}, {i}, lsl #3]",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                scale_bits = in(reg) scale.to_bits(),
                x = in(reg) x.as_mut_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                out("q0") _,
                out("q1") _,
                out("p0") _,
                options(nostack),
            );
        }
    }

    #[inline(always)]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        let n = dst.len().min(a.len()).min(b.len());
        if n == 0 || !Self::is_supported() {
            ScalarBackend::sub_f32_to_f64(dst, a, b);
            return;
        }
        // SAFETY: `FEAT_SVE2` verified; `whilelt p0.d` keeps all `ld1w`/`st1d` inside `0..n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "mov {i}, #0",
                "whilelt p0.d, {i}, {n}",
                "b.none 3f",
                "2:",
                "ld1w {{z0.d}}, p0/z, [{a}, {i}, lsl #2]",
                "ld1w {{z1.d}}, p0/z, [{b}, {i}, lsl #2]",
                "fcvt z0.d, p0/m, z0.s",
                "fcvt z1.d, p0/m, z1.s",
                "fsub z0.d, z0.d, z1.d",
                "st1d {{z0.d}}, p0, [{dst}, {i}, lsl #3]",
                "incd {i}",
                "whilelt p0.d, {i}, {n}",
                "b.first 2b",
                "3:",
                dst = in(reg) dst.as_mut_ptr(),
                a = in(reg) a.as_ptr(),
                b = in(reg) b.as_ptr(),
                n = in(reg) n,
                i = out(reg) _,
                out("q0") _,
                out("q1") _,
                out("p0") _,
                options(nostack),
            );
        }
    }

    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        let mut n = cols_i[0].len().min(cols_j[0].len());
        for k in 1..4 {
            n = n.min(cols_i[k].len()).min(cols_j[k].len());
        }
        if n == 0 || !Self::is_supported() {
            return ScalarBackend::gram_tile_4x4_f64(cols_i, cols_j);
        }
        let mut out = [[0.0_f64; 4]; 4];
        // SAFETY: `FEAT_SVE2` verified; only caller-saved `z0..z7, z16..z31` touched and bounded by `n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "mov z16.d, #0\nmov z17.d, #0\nmov z18.d, #0\nmov z19.d, #0",
                "mov z20.d, #0\nmov z21.d, #0\nmov z22.d, #0\nmov z23.d, #0",
                "mov z24.d, #0\nmov z25.d, #0\nmov z26.d, #0\nmov z27.d, #0",
                "mov z28.d, #0\nmov z29.d, #0\nmov z30.d, #0\nmov z31.d, #0",
                "mov {idx}, #0",
                "whilelt p0.d, {idx}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z0.d}}, p0/z, [{i0}, {idx}, lsl #3]",
                "ld1d {{z1.d}}, p0/z, [{i1}, {idx}, lsl #3]",
                "ld1d {{z2.d}}, p0/z, [{i2}, {idx}, lsl #3]",
                "ld1d {{z3.d}}, p0/z, [{i3}, {idx}, lsl #3]",
                "ld1d {{z4.d}}, p0/z, [{j0}, {idx}, lsl #3]",
                "ld1d {{z5.d}}, p0/z, [{j1}, {idx}, lsl #3]",
                "ld1d {{z6.d}}, p0/z, [{j2}, {idx}, lsl #3]",
                "ld1d {{z7.d}}, p0/z, [{j3}, {idx}, lsl #3]",
                "fmla z16.d, p0/m, z0.d, z4.d\nfmla z17.d, p0/m, z0.d, z5.d",
                "fmla z18.d, p0/m, z0.d, z6.d\nfmla z19.d, p0/m, z0.d, z7.d",
                "fmla z20.d, p0/m, z1.d, z4.d\nfmla z21.d, p0/m, z1.d, z5.d",
                "fmla z22.d, p0/m, z1.d, z6.d\nfmla z23.d, p0/m, z1.d, z7.d",
                "fmla z24.d, p0/m, z2.d, z4.d\nfmla z25.d, p0/m, z2.d, z5.d",
                "fmla z26.d, p0/m, z2.d, z6.d\nfmla z27.d, p0/m, z2.d, z7.d",
                "fmla z28.d, p0/m, z3.d, z4.d\nfmla z29.d, p0/m, z3.d, z5.d",
                "fmla z30.d, p0/m, z3.d, z6.d\nfmla z31.d, p0/m, z3.d, z7.d",
                "incd {idx}",
                "whilelt p0.d, {idx}, {n}",
                "b.first 2b",
                "3:",
                "ptrue p0.d",
                "faddv d0, p0, z16.d\nstr d0, [{dst}, #0]\nfaddv d0, p0, z17.d\nstr d0, [{dst}, #8]",
                "faddv d0, p0, z18.d\nstr d0, [{dst}, #16]\nfaddv d0, p0, z19.d\nstr d0, [{dst}, #24]",
                "faddv d0, p0, z20.d\nstr d0, [{dst}, #32]\nfaddv d0, p0, z21.d\nstr d0, [{dst}, #40]",
                "faddv d0, p0, z22.d\nstr d0, [{dst}, #48]\nfaddv d0, p0, z23.d\nstr d0, [{dst}, #56]",
                "faddv d0, p0, z24.d\nstr d0, [{dst}, #64]\nfaddv d0, p0, z25.d\nstr d0, [{dst}, #72]",
                "faddv d0, p0, z26.d\nstr d0, [{dst}, #80]\nfaddv d0, p0, z27.d\nstr d0, [{dst}, #88]",
                "faddv d0, p0, z28.d\nstr d0, [{dst}, #96]\nfaddv d0, p0, z29.d\nstr d0, [{dst}, #104]",
                "faddv d0, p0, z30.d\nstr d0, [{dst}, #112]\nfaddv d0, p0, z31.d\nstr d0, [{dst}, #120]",
                i0 = in(reg) cols_i[0].as_ptr(),
                i1 = in(reg) cols_i[1].as_ptr(),
                i2 = in(reg) cols_i[2].as_ptr(),
                i3 = in(reg) cols_i[3].as_ptr(),
                j0 = in(reg) cols_j[0].as_ptr(),
                j1 = in(reg) cols_j[1].as_ptr(),
                j2 = in(reg) cols_j[2].as_ptr(),
                j3 = in(reg) cols_j[3].as_ptr(),
                n = in(reg) n,
                dst = in(reg) out.as_mut_ptr(),
                idx = out(reg) _,
                out("q0") _, out("q1") _, out("q2") _, out("q3") _,
                out("q4") _, out("q5") _, out("q6") _, out("q7") _,
                out("q16") _, out("q17") _, out("q18") _, out("q19") _,
                out("q20") _, out("q21") _, out("q22") _, out("q23") _,
                out("q24") _, out("q25") _, out("q26") _, out("q27") _,
                out("q28") _, out("q29") _, out("q30") _, out("q31") _,
                out("p0") _,
                options(nostack),
            );
        }
        out
    }

    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        let n = cols[0]
            .len()
            .min(cols[1].len())
            .min(cols[2].len())
            .min(cols[3].len());
        if n == 0 || !Self::is_supported() {
            return ScalarBackend::gram_sym_block4_f64(cols);
        }
        let mut out = [0.0_f64; 10];
        // SAFETY: `FEAT_SVE2` verified; only caller-saved `z0..z3, z16..z25` touched and bounded by `n`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "mov z16.d, #0\nmov z17.d, #0\nmov z18.d, #0\nmov z19.d, #0\nmov z20.d, #0",
                "mov z21.d, #0\nmov z22.d, #0\nmov z23.d, #0\nmov z24.d, #0\nmov z25.d, #0",
                "mov {idx}, #0",
                "whilelt p0.d, {idx}, {n}",
                "b.none 3f",
                "2:",
                "ld1d {{z0.d}}, p0/z, [{c0}, {idx}, lsl #3]",
                "ld1d {{z1.d}}, p0/z, [{c1}, {idx}, lsl #3]",
                "ld1d {{z2.d}}, p0/z, [{c2}, {idx}, lsl #3]",
                "ld1d {{z3.d}}, p0/z, [{c3}, {idx}, lsl #3]",
                "fmla z16.d, p0/m, z0.d, z0.d\nfmla z17.d, p0/m, z0.d, z1.d",
                "fmla z18.d, p0/m, z1.d, z1.d\nfmla z19.d, p0/m, z0.d, z2.d",
                "fmla z20.d, p0/m, z1.d, z2.d\nfmla z21.d, p0/m, z2.d, z2.d",
                "fmla z22.d, p0/m, z0.d, z3.d\nfmla z23.d, p0/m, z1.d, z3.d",
                "fmla z24.d, p0/m, z2.d, z3.d\nfmla z25.d, p0/m, z3.d, z3.d",
                "incd {idx}",
                "whilelt p0.d, {idx}, {n}",
                "b.first 2b",
                "3:",
                "ptrue p0.d",
                "faddv d0, p0, z16.d\nstr d0, [{dst}, #0]\nfaddv d0, p0, z17.d\nstr d0, [{dst}, #8]",
                "faddv d0, p0, z18.d\nstr d0, [{dst}, #16]\nfaddv d0, p0, z19.d\nstr d0, [{dst}, #24]",
                "faddv d0, p0, z20.d\nstr d0, [{dst}, #32]\nfaddv d0, p0, z21.d\nstr d0, [{dst}, #40]",
                "faddv d0, p0, z22.d\nstr d0, [{dst}, #48]\nfaddv d0, p0, z23.d\nstr d0, [{dst}, #56]",
                "faddv d0, p0, z24.d\nstr d0, [{dst}, #64]\nfaddv d0, p0, z25.d\nstr d0, [{dst}, #72]",
                c0 = in(reg) cols[0].as_ptr(),
                c1 = in(reg) cols[1].as_ptr(),
                c2 = in(reg) cols[2].as_ptr(),
                c3 = in(reg) cols[3].as_ptr(),
                n = in(reg) n,
                dst = in(reg) out.as_mut_ptr(),
                idx = out(reg) _,
                out("q0") _, out("q1") _, out("q2") _, out("q3") _,
                out("q16") _, out("q17") _, out("q18") _, out("q19") _,
                out("q20") _, out("q21") _, out("q22") _, out("q23") _,
                out("q24") _, out("q25") _,
                out("p0") _,
                options(nostack),
            );
        }
        out
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
        if bound == 0 || !Self::is_supported() {
            ScalarBackend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
            return;
        }
        const { assert!(core::mem::size_of::<[f64; 256]>() == 2048) };
        // SAFETY: `FEAT_SVE2` verified; `whilelt` bounds `d < bound <= 256`, non-FMA `fmul`+`fadd` is `0-ULP`.
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sve2",
                "dup z3.d, {omr_bits}",
                "dup z4.d, {r_bits}",
                "mov {d}, #0",
                "whilelt p0.d, {d}, {bound}",
                "b.none 3f",
                "2:",
                "ld1w {{z0.d}}, p0/z, [{fb}, {d}, lsl #2]",
                "fcvt z0.d, p0/m, z0.s",
                "mov {row}, {ubase}",
                "mov {s}, #7",
                "4:",
                "ld1d {{z1.d}}, p0/z, [{row}, {d}, lsl #3]",
                "fmul z2.d, z3.d, z0.d",
                "fmul z1.d, z4.d, z1.d",
                "fadd z0.d, z2.d, z1.d",
                "st1d {{z0.d}}, p0, [{row}, {d}, lsl #3]",
                "add {row}, {row}, #2048",
                "subs {s}, {s}, #1",
                "b.ne 4b",
                "incd {d}",
                "whilelt p0.d, {d}, {bound}",
                "b.first 2b",
                "3:",
                omr_bits = in(reg) omr.to_bits(),
                r_bits = in(reg) r.to_bits(),
                fb = in(reg) fb.as_ptr(),
                ubase = in(reg) u.as_mut_ptr(),
                bound = in(reg) bound,
                d = out(reg) _,
                row = out(reg) _,
                s = out(reg) _,
                out("q0") _,
                out("q1") _,
                out("q2") _,
                out("q3") _,
                out("q4") _,
                out("p0") _,
                options(nostack),
            );
        }
    }
}
