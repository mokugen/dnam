//! Raw Streaming SVE (`smstart sm` .. `smstop sm`) and `FEAT_SME_F64F64` (`FMOPA` `ZA`) kernels
//! for [`super::sme_f64`].

/// Evaluates `<a, b>` over `n > 0` elements inside Streaming SVE mode without `FEAT_SME_FA64`.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()` is true and `a`/`b` are valid for `n`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_dot_f64(a: *const f64, b: *const f64, n: usize) -> f64 {
    let mut out = 0.0_f64;
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "ptrue p1.d, vl1",
            "st1d {{z0.d}}, p1, [{out}]",
            "smstop sm",
            a = in(reg) a,
            b = in(reg) b,
            n = in(reg) n,
            out = in(reg) (&raw mut out),
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
    out
}

/// Evaluates `(<x, b0>, <x, b1>)` over `n > 0` elements inside Streaming SVE mode.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()` is true and slices are valid for `n`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_dot_pair_f64(
    x: *const f64,
    b0: *const f64,
    b1: *const f64,
    n: usize,
) -> (f64, f64) {
    let mut out = [0.0_f64; 2];
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "ptrue p1.d, vl1",
            "st1d {{z0.d}}, p1, [{out0}]",
            "st1d {{z1.d}}, p1, [{out1}]",
            "smstop sm",
            x = in(reg) x,
            b0 = in(reg) b0,
            b1 = in(reg) b1,
            n = in(reg) n,
            out0 = in(reg) (&raw mut out[0]),
            out1 = in(reg) (&raw mut out[1]),
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
    (out[0], out[1])
}

/// Evaluates `sum(x^2)` over `n > 0` elements inside Streaming SVE mode.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()` is true and `x` is valid for `n`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_sum_sq_f64(x: *const f64, n: usize) -> f64 {
    let mut out = 0.0_f64;
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "ptrue p1.d, vl1",
            "st1d {{z0.d}}, p1, [{out}]",
            "smstop sm",
            x = in(reg) x,
            n = in(reg) n,
            out = in(reg) (&raw mut out),
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
    out
}

/// Executes `y[i] += alpha * x[i]` over `0..n` inside Streaming SVE mode.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()`, exclusive `y`, and valid `x` for `n`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_axpy_f64(y: *mut f64, alpha: f64, x: *const f64, n: usize) {
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "smstop sm",
            alpha_bits = in(reg) alpha.to_bits(),
            y = in(reg) y,
            x = in(reg) x,
            n = in(reg) n,
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
}

/// Executes `y[i] += a0 * x0[i] + a1 * x1[i]` over `0..n` inside Streaming SVE mode.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()`, exclusive `y`, and valid `x0`/`x1`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_axpy_pair_f64(
    y: *mut f64,
    a0: f64,
    x0: *const f64,
    a1: f64,
    x1: *const f64,
    n: usize,
) {
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "smstop sm",
            a0_bits = in(reg) a0.to_bits(),
            a1_bits = in(reg) a1.to_bits(),
            y = in(reg) y,
            x0 = in(reg) x0,
            x1 = in(reg) x1,
            n = in(reg) n,
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
}

/// Executes `x[i] *= scale` over `0..n` inside Streaming SVE mode.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()` and exclusive valid `x` for `n`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_scale_f64(x: *mut f64, scale: f64, n: usize) {
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "smstop sm",
            scale_bits = in(reg) scale.to_bits(),
            x = in(reg) x,
            n = in(reg) n,
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
}

/// Computes `dst[i] = f64::from(a[i]) - f64::from(b[i])` (`0-ULP`) inside Streaming SVE mode.
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()`, exclusive `dst`, and valid `a`/`b`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_sub_f32_to_f64(
    dst: *mut f64,
    a: *const f32,
    b: *const f32,
    n: usize,
) {
    // SAFETY: Caller verified `FEAT_SME` and slice bounds `0..n`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "smstop sm",
            dst = in(reg) dst,
            a = in(reg) a,
            b = in(reg) b,
            n = in(reg) n,
            i = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
}

/// Steps 7 cascaded 1-pole Bézout stages across `bound` (`<= 256`) features (`0-ULP`).
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_supported()` and `bound <= 256`.
#[inline(never)]
pub(super) unsafe fn sme_ssve_advance_bezout7(
    fb: *const f32,
    bound: usize,
    r: f64,
    omr: f64,
    u_base: *mut [f64; 256],
) {
    const { assert!(core::mem::size_of::<[f64; 256]>() == 2048) };
    // SAFETY: Caller verified `FEAT_SME` and `bound <= 256`; non-FMA `fmul`+`fadd` preserves `0-ULP`.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme",
            "smstart sm",
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
            "smstop sm",
            omr_bits = in(reg) omr.to_bits(),
            r_bits = in(reg) r.to_bits(),
            fb = in(reg) fb,
            ubase = in(reg) u_base,
            bound = in(reg) bound,
            d = out(reg) _,
            row = out(reg) _,
            s = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
}

/// Evaluates `4 x 4` outer-product Gram contraction across `pairs * 2` samples in `ZA` (`za0.d..za3.d`).
///
/// # Safety
/// Caller must guarantee `ArmSmeBackend::is_f64f64_supported()`, `pairs > 0`, and valid pointers
/// for `pairs * 2` elements on every column pointer.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn sme_za_gram_tile_4x4_f64(
    i0: *const f64,
    i1: *const f64,
    i2: *const f64,
    i3: *const f64,
    j0: *const f64,
    j1: *const f64,
    j2: *const f64,
    j3: *const f64,
    pairs: usize,
) -> [[f64; 4]; 4] {
    let mut out = [[0.0_f64; 4]; 4];
    let dst = out.as_mut_ptr().cast::<u8>();
    let rem_pairs = pairs;
    let idx = 0_usize;
    // SAFETY: Caller verified `FEAT_SME` + `FEAT_SME_F64F64` and `pairs * 2` bounds.
    unsafe {
        core::arch::asm!(
            ".arch armv8.6-a+sme+sme-f64f64",
            "smstart",
            "zero {{za}}",
            "ptrue p0.d, vl2",
            "2:",
            "ld1d {{z0.d}}, p0/z, [{i0}, {idx}, lsl #3]",
            "ld1d {{z1.d}}, p0/z, [{i1}, {idx}, lsl #3]",
            "ld1d {{z2.d}}, p0/z, [{i2}, {idx}, lsl #3]",
            "ld1d {{z3.d}}, p0/z, [{i3}, {idx}, lsl #3]",
            "ld1d {{z4.d}}, p0/z, [{j0}, {idx}, lsl #3]",
            "ld1d {{z5.d}}, p0/z, [{j1}, {idx}, lsl #3]",
            "ld1d {{z6.d}}, p0/z, [{j2}, {idx}, lsl #3]",
            "ld1d {{z7.d}}, p0/z, [{j3}, {idx}, lsl #3]",
            "trn1 z16.d, z0.d, z1.d\ntrn1 z17.d, z2.d, z3.d",
            "trn1 z18.d, z4.d, z5.d\ntrn1 z19.d, z6.d, z7.d",
            "trn2 z20.d, z0.d, z1.d\ntrn2 z21.d, z2.d, z3.d",
            "trn2 z22.d, z4.d, z5.d\ntrn2 z23.d, z6.d, z7.d",
            "fmopa za0.d, p0/m, p0/m, z16.d, z18.d",
            "fmopa za1.d, p0/m, p0/m, z16.d, z19.d",
            "fmopa za2.d, p0/m, p0/m, z17.d, z18.d",
            "fmopa za3.d, p0/m, p0/m, z17.d, z19.d",
            "fmopa za0.d, p0/m, p0/m, z20.d, z22.d",
            "fmopa za1.d, p0/m, p0/m, z20.d, z23.d",
            "fmopa za2.d, p0/m, p0/m, z21.d, z22.d",
            "fmopa za3.d, p0/m, p0/m, z21.d, z23.d",
            "add {idx}, {idx}, #2",
            "subs {pairs}, {pairs}, #1",
            "b.ne 2b",
            "mov w12, #0",
            "st1d {{za0h.d[w12, 0]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za1h.d[w12, 0]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za0h.d[w12, 1]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za1h.d[w12, 1]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za2h.d[w12, 0]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za3h.d[w12, 0]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za2h.d[w12, 1]}}, p0, [{dst}]\nadd {dst}, {dst}, #16",
            "st1d {{za3h.d[w12, 1]}}, p0, [{dst}]",
            "smstop",
            i0 = in(reg) i0,
            i1 = in(reg) i1,
            i2 = in(reg) i2,
            i3 = in(reg) i3,
            j0 = in(reg) j0,
            j1 = in(reg) j1,
            j2 = in(reg) j2,
            j3 = in(reg) j3,
            idx = inout(reg) idx => _,
            pairs = inout(reg) rem_pairs => _,
            dst = inout(reg) dst => _,
            out("w12") _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _, out("q7") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
            out("q16") _, out("q17") _, out("q18") _, out("q19") _, out("q20") _, out("q21") _, out("q22") _, out("q23") _,
            out("q24") _, out("q25") _, out("q26") _, out("q27") _, out("q28") _, out("q29") _, out("q30") _, out("q31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _, out("p6") _, out("p7") _,
            out("p8") _, out("p9") _, out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _, out("p15") _,
            options(nostack),
        );
    }
    out
}
