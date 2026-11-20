//! Pure ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`) [`TrainerMathBackend`]
//! implementation for [`super::ArmSve2Backend`].

use core::arch::aarch64::float64x2_t;

use super::super::sve2::ArmSve2Backend;
use super::Sve2F64Vec2;
use crate::math::ScalarBackend;
use crate::math::trainer::TrainerMathBackend;

impl TrainerMathBackend for ArmSve2Backend {
    type F64Vec2 = Sve2F64Vec2;

    #[inline]
    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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

    #[inline]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        let n = dst.len().min(a.len()).min(b.len());
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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
        if n == 0 || !Self::is_supported_cached() {
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
        if bound == 0 || !Self::is_supported_cached() {
            ScalarBackend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
            return;
        }
        const { assert!(core::mem::size_of::<[f64; 256]>() == 2048) };
        let _: float64x2_t = super::to_zreg([0.0, 0.0]);
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

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use proptest::prelude::*;

    use super::*;
    use crate::math::trainer::{ScalarF64Vec2, SimdF64Vec2};
    use crate::math::{BackendKind, warm_simd_feature_caches};

    #[test]
    fn test_sve2_f64_vec2_and_edge_lengths() {
        warm_simd_feature_caches();
        assert_eq!(ArmSve2Backend::trainer_backend_kind(), BackendKind::ArmSve2);

        let a = Sve2F64Vec2::from_array([1.5, -2.5]);
        let b = Sve2F64Vec2::splat_pair(2.0, -4.0);
        let c = Sve2F64Vec2::from_f32_pair(0.5, -1.25);
        let sa = ScalarF64Vec2::from_array([1.5, -2.5]);
        let sb = ScalarF64Vec2::splat_pair(2.0, -4.0);
        let sc = ScalarF64Vec2::from_f32_pair(0.5, -1.25);
        let mut stored = [0.0_f64; 2];
        a.store(&mut stored);
        assert_eq!(stored, [1.5, -2.5]);
        assert_eq!(Sve2F64Vec2::zero().to_array(), [0.0, 0.0]);
        assert_eq!(a.add(b).to_array(), sa.add(sb).to_array());
        assert_eq!(a.sub(b).to_array(), sa.sub(sb).to_array());
        assert_eq!(a.mul(b).to_array(), sa.mul(sb).to_array());
        assert!((a.fma(b, c).horizontal_sum() - sa.fma(sb, sc).horizontal_sum()).abs() <= 1e-12);
        assert!((a.fms(b, c).horizontal_sum() - sa.fms(sb, sc).horizontal_sum()).abs() <= 1e-12);
        assert_eq!(
            a.fma_lane::<0>(b, c).to_array(),
            sa.fma_lane::<0>(sb, sc).to_array()
        );
        assert_eq!(
            a.fma_lane::<1>(b, c).to_array(),
            sa.fma_lane::<1>(sb, sc).to_array()
        );
        assert_eq!(
            a.mul_add_ieee(b, c).to_array(),
            sa.mul_add_ieee(sb, sc).to_array()
        );
        assert_eq!(a.max(b).to_array(), sa.max(sb).to_array());
        assert_eq!(a.min(b).to_array(), sa.min(sb).to_array());
        assert_eq!(
            a.clamp(Sve2F64Vec2::splat(-2.0), Sve2F64Vec2::splat(2.0))
                .to_array(),
            [1.5, -2.0]
        );
        assert_eq!(a.abs().to_array(), [1.5, 2.5]);
        assert_eq!(
            a.horizontal_sum_pair(b),
            (sa.horizontal_sum(), sb.horizontal_sum())
        );
        assert_eq!(a.dot(b), sa.dot(sb));

        for len in [0_usize, 1, 2, 3, 4, 7, 8, 9, 17] {
            let x: Vec<f64> = (0..len)
                .map(|i| 0.29 * ((i as f64) * 0.21).sin() - 0.1)
                .collect();
            let y: Vec<f64> = (0..len)
                .map(|i| -0.37 * ((i as f64) * 0.17).cos() + 0.05)
                .collect();
            let z: Vec<f64> = (0..len)
                .map(|i| 0.41 * ((i as f64) * 0.13).sin() + 0.08)
                .collect();
            let dot_diff = ArmSve2Backend::dot_f64(&x, &y) - ScalarBackend::dot_f64(&x, &y);
            assert!(dot_diff.abs() <= 1e-11);
            let (dp0, dp1) = ArmSve2Backend::dot_pair_f64(&x, &y, &z);
            let (sp0, sp1) = ScalarBackend::dot_pair_f64(&x, &y, &z);
            assert!((dp0 - sp0).abs() <= 1e-11 && (dp1 - sp1).abs() <= 1e-11);
            let sq_diff = ArmSve2Backend::sum_sq_f64(&x) - ScalarBackend::sum_sq_f64(&x);
            assert!(sq_diff.abs() <= 1e-11);

            let (mut ay, mut sy) = (x.clone(), x.clone());
            ArmSve2Backend::axpy_f64(&mut ay, -0.625, &y);
            ScalarBackend::axpy_f64(&mut sy, -0.625, &y);
            assert!(ay.iter().zip(&sy).all(|(u, v)| (u - v).abs() <= 1e-11));

            let (mut apy, mut spy) = (x.clone(), x.clone());
            ArmSve2Backend::axpy_pair_f64(&mut apy, 0.5, &y, -0.75, &z);
            ScalarBackend::axpy_pair_f64(&mut spy, 0.5, &y, -0.75, &z);
            assert!(apy.iter().zip(&spy).all(|(u, v)| (u - v).abs() <= 1e-11));

            ArmSve2Backend::scale_f64(&mut ay, 1.375);
            ScalarBackend::scale_f64(&mut sy, 1.375);
            assert!(ay.iter().zip(&sy).all(|(u, v)| (u - v).abs() <= 1e-11));

            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let (mut ds, mut db) = (vec![0.0_f64; len], vec![0.0_f64; len]);
            ArmSve2Backend::sub_f32_to_f64(&mut db, &x32, &y32);
            ScalarBackend::sub_f32_to_f64(&mut ds, &x32, &y32);
            assert!(db.iter().zip(&ds).all(|(u, v)| u.to_bits() == v.to_bits()));

            let ci = [x.as_slice(), y.as_slice(), z.as_slice(), x.as_slice()];
            let cj = [y.as_slice(), z.as_slice(), x.as_slice(), z.as_slice()];
            let gt_b = ArmSve2Backend::gram_tile_4x4_f64(ci, cj);
            let gt_s = ScalarBackend::gram_tile_4x4_f64(ci, cj);
            for r in 0..4 {
                assert!(
                    gt_b[r]
                        .iter()
                        .zip(&gt_s[r])
                        .all(|(u, v)| (u - v).abs() <= 1e-11)
                );
            }
            let gs_b = ArmSve2Backend::gram_sym_block4_f64(ci);
            let gs_s = ScalarBackend::gram_sym_block4_f64(ci);
            assert!(gs_b.iter().zip(&gs_s).all(|(u, v)| (u - v).abs() <= 1e-11));

            let f0: Vec<[f64; 4]> = (0..len).map(|k| [k as f64 * 0.1, -0.2, 0.3, 0.4]).collect();
            let f1: Vec<[f64; 4]> = (0..len)
                .map(|k| [-0.5, k as f64 * 0.05, 0.15, -0.8])
                .collect();
            let wf: Vec<[f64; 4]> = (0..len)
                .map(|k| [0.4, -0.3, 0.2 + k as f64 * 0.02, -0.6])
                .collect();
            let t4_b = ArmSve2Backend::contract_tile_4x4_ieee_f64(&f0, &wf);
            let t4_s = ScalarBackend::contract_tile_4x4_ieee_f64(&f0, &wf);
            let (t8a_b, t8b_b) = ArmSve2Backend::contract_tile_8x4_ieee_f64(&f0, &f1, &wf);
            let (t8a_s, t8b_s) = ScalarBackend::contract_tile_8x4_ieee_f64(&f0, &f1, &wf);
            for r in 0..4 {
                for col in 0..4 {
                    assert_eq!(t4_b[r][col].to_bits(), t4_s[r][col].to_bits());
                    assert_eq!(t8a_b[r][col].to_bits(), t8a_s[r][col].to_bits());
                    assert_eq!(t8b_b[r][col].to_bits(), t8b_s[r][col].to_bits());
                }
            }
        }
    }

    #[test]
    fn test_sve2_f64_mismatched_lengths_and_bezout7() {
        warm_simd_feature_caches();
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [0.5, -1.0, 1.5];
        let c = [2.0, 3.0];
        assert!((ArmSve2Backend::dot_f64(&a, &b) - 3.0).abs() <= 1e-12);
        let (p0, p1) = ArmSve2Backend::dot_pair_f64(&a, &b, &c);
        assert!((p0 + 1.5).abs() <= 1e-12 && (p1 - 8.0).abs() <= 1e-12);

        let mut dst = [9.0_f64; 5];
        ArmSve2Backend::axpy_f64(&mut dst, 2.0, &b);
        assert_eq!(dst, [10.0, 7.0, 12.0, 9.0, 9.0]);
        ArmSve2Backend::axpy_pair_f64(&mut dst, 1.0, &b, -1.0, &c);
        assert_eq!(dst, [8.5, 3.0, 12.0, 9.0, 9.0]);

        let fb: [f32; 256] = core::array::from_fn(|idx| ((idx as f32) * 0.17).sin() * 0.3);
        for dim in [0_usize, 1, 3, 7, 65, 256, 300] {
            let (mut ub, mut us) = ([[0.0_f64; 256]; 7], [[0.0_f64; 256]; 7]);
            ArmSve2Backend::advance_bezout7_lp_dim_f64(&fb, dim, 0.92, 0.08, &mut ub);
            ScalarBackend::advance_bezout7_lp_dim_f64(&fb, dim, 0.92, 0.08, &mut us);
            for st in 0..7 {
                assert!(
                    ub[st]
                        .iter()
                        .zip(&us[st])
                        .all(|(u, v)| u.to_bits() == v.to_bits())
                );
            }
            let (mut lfb, mut mfb, mut hfb) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
            let (mut lfs, mut mfs, mut hfs) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
            ArmSve2Backend::readout_bezout7_bands_dim_f64(
                &fb, dim, &ub, &ub, &mut lfb, &mut mfb, &mut hfb,
            );
            ScalarBackend::readout_bezout7_bands_dim_f64(
                &fb, dim, &us, &us, &mut lfs, &mut mfs, &mut hfs,
            );
            for d in 0..256 {
                assert_eq!(lfb[d].to_bits(), lfs[d].to_bits());
                assert_eq!(mfb[d].to_bits(), mfs[d].to_bits());
                assert_eq!(hfb[d].to_bits(), hfs[d].to_bits());
            }
        }
    }

    proptest! {
        #[test]
        fn prop_sve2_f64_trainer_matches_scalar_across_lengths(
            xs in prop::collection::vec(-5.0_f64..5.0_f64, 0..25),
            ys in prop::collection::vec(-5.0_f64..5.0_f64, 0..25),
        ) {
            warm_simd_feature_caches();
            let n = xs.len().min(ys.len());
            let d_dot = ArmSve2Backend::dot_f64(&xs[..n], &ys[..n]);
            let s_dot = ScalarBackend::dot_f64(&xs[..n], &ys[..n]);
            prop_assert!((d_dot - s_dot).abs() <= 1.0e-11 * s_dot.abs().max(1.0));
            let x32: Vec<f32> = xs[..n].iter().map(|&v| v as f32).collect();
            let y32: Vec<f32> = ys[..n].iter().map(|&v| v as f32).collect();
            let (mut db, mut ds) = (vec![0.0; n], vec![0.0; n]);
            ArmSve2Backend::sub_f32_to_f64(&mut db, &x32, &y32);
            ScalarBackend::sub_f32_to_f64(&mut ds, &x32, &y32);
            for i in 0..n {
                prop_assert_eq!(db[i].to_bits(), ds[i].to_bits());
            }
        }
    }
}
