#![forbid(unsafe_code)]

//! Workgroup-uniform `256 -> scalar` scalar, batch-6, and non-commutative `TptAdjointElem`
//! binary-tree reductions (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::wg_barrier;
use crate::abi::{ADJOINT_AUX_FLOATS, ADJOINT_SCAN_SCRATCH_FLOATS, TILE_FRAMES};
use dnam_kernels::{TptAdjointElem, canonicalize_zero_f32_bits, compose_tpt_adjoint_scan};
#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
use spirv_std::arch::subgroup_f_add;

const HALF_TILE: usize = TILE_FRAMES / 2;

#[inline(always)]
fn store_dual_adj(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    k: usize,
    a: TptAdjointElem,
    gb: [f32; 2],
) {
    let slot = k & 127;
    scan_scratch[slot] = a.t00;
    scan_scratch[HALF_TILE + slot] = a.t01;
    scan_scratch[2 * HALF_TILE + slot] = a.t10;
    scan_scratch[3 * HALF_TILE + slot] = a.t11;
    scan_scratch[4 * HALF_TILE + slot] = a.g0;
    scan_scratch[5 * HALF_TILE + slot] = a.g1;
    scan_scratch[6 * HALF_TILE + slot] = gb[0];
    scan_scratch[7 * HALF_TILE + slot] = gb[1];
}

#[inline(always)]
fn load_dual_adj(
    scan_scratch: &[f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    k: usize,
) -> (TptAdjointElem, [f32; 2]) {
    let slot = k & 127;
    (
        TptAdjointElem {
            t00: scan_scratch[slot],
            t01: scan_scratch[HALF_TILE + slot],
            t10: scan_scratch[2 * HALF_TILE + slot],
            t11: scan_scratch[3 * HALF_TILE + slot],
            g0: scan_scratch[4 * HALF_TILE + slot],
            g1: scan_scratch[5 * HALF_TILE + slot],
        },
        [
            scan_scratch[6 * HALF_TILE + slot],
            scan_scratch[7 * HALF_TILE + slot],
        ],
    )
}

#[inline(always)]
fn store_quad_g(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    k: usize,
    gl: [f32; 2],
    gr: [f32; 2],
    kgl: [f32; 2],
    kgr: [f32; 2],
) {
    let slot = k & 127;
    scan_scratch[slot] = gl[0];
    scan_scratch[HALF_TILE + slot] = gl[1];
    scan_scratch[2 * HALF_TILE + slot] = gr[0];
    scan_scratch[3 * HALF_TILE + slot] = gr[1];
    scan_scratch[4 * HALF_TILE + slot] = kgl[0];
    scan_scratch[5 * HALF_TILE + slot] = kgl[1];
    scan_scratch[6 * HALF_TILE + slot] = kgr[0];
    scan_scratch[7 * HALF_TILE + slot] = kgr[1];
}

#[inline(always)]
fn load_quad_g(
    scan_scratch: &[f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    k: usize,
) -> ([f32; 2], [f32; 2], [f32; 2], [f32; 2]) {
    let slot = k & 127;
    (
        [scan_scratch[slot], scan_scratch[HALF_TILE + slot]],
        [
            scan_scratch[2 * HALF_TILE + slot],
            scan_scratch[3 * HALF_TILE + slot],
        ],
        [
            scan_scratch[4 * HALF_TILE + slot],
            scan_scratch[5 * HALF_TILE + slot],
        ],
        [
            scan_scratch[6 * HALF_TILE + slot],
            scan_scratch[7 * HALF_TILE + slot],
        ],
    )
}

#[inline(always)]
fn matmul2(
    a00: f32,
    a01: f32,
    a10: f32,
    a11: f32,
    b00: f32,
    b01: f32,
    b10: f32,
    b11: f32,
) -> (f32, f32, f32, f32) {
    (
        a00 * b00 + a01 * b10,
        a00 * b01 + a01 * b11,
        a10 * b00 + a11 * b10,
        a10 * b01 + a11 * b11,
    )
}

/// Paired non-commutative `256 -> 1` binary-tree reduction (`9` barriers) for `(elem_a, elem_b)`
/// sharing identical `2x2` matrix `T` (e.g. Block 0 PHS `sum_tpt` and `sum_k`), returning the exact
/// full-tile pair on `lid & 255 == 0`.
#[inline]
pub fn tpt_adjoint_dual_reduce_256(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem_a: TptAdjointElem,
    elem_b: TptAdjointElem,
) -> (TptAdjointElem, TptAdjointElem) {
    let i = lid & 255;
    let k = i >> 1;
    let is_even = (i & 1) == 0;

    if !is_even {
        store_dual_adj(scan_scratch, k, elem_a, [elem_b.g0, elem_b.g1]);
    }
    wg_barrier();

    let mut cur_a = elem_a;
    let mut cur_gb = [elem_b.g0, elem_b.g1];
    if is_even {
        let (sa, sgb) = load_dual_adj(scan_scratch, k);
        cur_gb = [
            cur_a.t00 * sgb[0] + cur_a.t01 * sgb[1] + cur_gb[0],
            cur_a.t10 * sgb[0] + cur_a.t11 * sgb[1] + cur_gb[1],
        ];
        cur_a = compose_tpt_adjoint_scan(cur_a, sa);
        store_dual_adj(scan_scratch, k, cur_a, cur_gb);
    }
    wg_barrier();

    let mut stride = 1_usize;
    while stride < HALF_TILE {
        let mask = (stride << 1) - 1;
        if is_even && (k & mask) == 0 {
            let (sa, sgb) = load_dual_adj(scan_scratch, k + stride);
            cur_gb = [
                cur_a.t00 * sgb[0] + cur_a.t01 * sgb[1] + cur_gb[0],
                cur_a.t10 * sgb[0] + cur_a.t11 * sgb[1] + cur_gb[1],
            ];
            cur_a = compose_tpt_adjoint_scan(cur_a, sa);
            store_dual_adj(scan_scratch, k, cur_a, cur_gb);
        }
        wg_barrier();
        stride <<= 1;
    }

    (
        cur_a,
        TptAdjointElem {
            t00: cur_a.t00,
            t01: cur_a.t01,
            t10: cur_a.t10,
            t11: cur_a.t11,
            g0: cur_gb[0],
            g1: cur_gb[1],
        },
    )
}

/// Constant-`M` `256 -> 1` binary-tree reduction (`9` barriers, `4,096 B` SRAM) reducing all 4
/// `(sum_tpt_l, sum_tpt_r, sum_k_l, sum_k_r)` accumulators simultaneously for `!is_phs` lanes
/// and computing exact `T^valid_len` via register squaring; results are exact on `lid & 255 == 0`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn tpt_adjoint_const_m_reduce_256_lr(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    valid_len: usize,
    a1: f32,
    a2: f32,
    a3: f32,
    g_l: [f32; 2],
    g_r: [f32; 2],
    kg_l: [f32; 2],
    kg_r: [f32; 2],
) -> (TptAdjointElem, TptAdjointElem, [f32; 2], [f32; 2]) {
    let i = lid & 255;
    let k = i >> 1;
    let is_even = (i & 1) == 0;
    let two_a2 = a2 + a2;
    let two_a3 = a3 + a3;
    let mut c00 = (a1 + a1) - 1.0;
    let mut c01 = two_a2;
    let mut c10 = -two_a2;
    let mut c11 = 1.0 - two_a3;

    let (mut p00, mut p01, mut p10, mut p11) = if (valid_len & 1) != 0 {
        (c00, c01, c10, c11)
    } else {
        (1.0, 0.0, 0.0, 1.0)
    };

    if !is_even {
        store_quad_g(scan_scratch, k, g_l, g_r, kg_l, kg_r);
    }
    wg_barrier();

    let mut cur_gl = g_l;
    let mut cur_gr = g_r;
    let mut cur_kgl = kg_l;
    let mut cur_kgr = kg_r;
    if is_even {
        let (sl, sr, skl, skr) = load_quad_g(scan_scratch, k);
        cur_gl = [
            c00 * sl[0] + c01 * sl[1] + cur_gl[0],
            c10 * sl[0] + c11 * sl[1] + cur_gl[1],
        ];
        cur_gr = [
            c00 * sr[0] + c01 * sr[1] + cur_gr[0],
            c10 * sr[0] + c11 * sr[1] + cur_gr[1],
        ];
        cur_kgl = [
            c00 * skl[0] + c01 * skl[1] + cur_kgl[0],
            c10 * skl[0] + c11 * skl[1] + cur_kgl[1],
        ];
        cur_kgr = [
            c00 * skr[0] + c01 * skr[1] + cur_kgr[0],
            c10 * skr[0] + c11 * skr[1] + cur_kgr[1],
        ];
        store_quad_g(scan_scratch, k, cur_gl, cur_gr, cur_kgl, cur_kgr);
    }
    (c00, c01, c10, c11) = matmul2(c00, c01, c10, c11, c00, c01, c10, c11);
    if (valid_len & 2) != 0 {
        (p00, p01, p10, p11) = matmul2(p00, p01, p10, p11, c00, c01, c10, c11);
    }
    wg_barrier();

    let mut stride = 1_usize;
    let mut pow_bit = 4_usize;
    while stride < HALF_TILE {
        let mask = (stride << 1) - 1;
        if is_even && (k & mask) == 0 {
            let (sl, sr, skl, skr) = load_quad_g(scan_scratch, k + stride);
            cur_gl = [
                c00 * sl[0] + c01 * sl[1] + cur_gl[0],
                c10 * sl[0] + c11 * sl[1] + cur_gl[1],
            ];
            cur_gr = [
                c00 * sr[0] + c01 * sr[1] + cur_gr[0],
                c10 * sr[0] + c11 * sr[1] + cur_gr[1],
            ];
            cur_kgl = [
                c00 * skl[0] + c01 * skl[1] + cur_kgl[0],
                c10 * skl[0] + c11 * skl[1] + cur_kgl[1],
            ];
            cur_kgr = [
                c00 * skr[0] + c01 * skr[1] + cur_kgr[0],
                c10 * skr[0] + c11 * skr[1] + cur_kgr[1],
            ];
            store_quad_g(scan_scratch, k, cur_gl, cur_gr, cur_kgl, cur_kgr);
        }
        (c00, c01, c10, c11) = matmul2(c00, c01, c10, c11, c00, c01, c10, c11);
        if (valid_len & pow_bit) != 0 {
            (p00, p01, p10, p11) = matmul2(p00, p01, p10, p11, c00, c01, c10, c11);
        }
        wg_barrier();
        stride <<= 1;
        pow_bit <<= 1;
    }

    (
        TptAdjointElem {
            t00: p00,
            t01: p01,
            t10: p10,
            t11: p11,
            g0: cur_gl[0],
            g1: cur_gl[1],
        },
        TptAdjointElem {
            t00: p00,
            t01: p01,
            t10: p10,
            t11: p11,
            g0: cur_gr[0],
            g1: cur_gr[1],
        },
        cur_kgl,
        cur_kgr,
    )
}

/// Deterministically reduces a `256`-thread `f32` sum inside `aux[0..128]` with optional W3C
/// WebGPU `subgroup_f_add` acceleration, broadcasting the scalar sum to all invocations.
#[inline]
pub fn reduce_sum_256_in_aux(
    aux: &mut [f32; ADJOINT_AUX_FLOATS],
    lid: usize,
    sg_lid: u32,
    val: f32,
) -> f32 {
    #[cfg(all(target_arch = "spirv", feature = "subgroups"))]
    let contrib = {
        let sg_sum = subgroup_f_add(val);
        if sg_lid == 0 { sg_sum } else { 0.0 }
    };
    #[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
    let contrib = {
        let _ = sg_lid;
        val
    };

    let i = lid & 255;
    if i < ADJOINT_AUX_FLOATS {
        aux[i & 127] = contrib;
    }
    wg_barrier();
    if i >= ADJOINT_AUX_FLOATS {
        aux[(i - ADJOINT_AUX_FLOATS) & 127] += contrib;
    }
    wg_barrier();

    let mut stride = ADJOINT_AUX_FLOATS >> 1;
    while stride > 0 {
        if i < stride {
            let dst = i & 127;
            let src = (i + stride).min(ADJOINT_AUX_FLOATS - 1);
            aux[dst] += aux[src];
        }
        wg_barrier();
        stride >>= 1;
    }

    let total = canonicalize_zero_f32_bits(aux[0]);
    wg_barrier();
    total
}

/// Deterministically reduces `6` scalar `f32` accumulators simultaneously in `scan_scratch[0..1536]`
/// using `8` uniform barriers (with `subgroup_f_add` pre-reduction under `subgroups`).
#[inline]
pub fn reduce_batch6_in_scratch(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    sg_lid: u32,
    vals: [f32; 6],
) -> [f32; 6] {
    #[cfg(all(target_arch = "spirv", feature = "subgroups"))]
    let contribs = {
        let s0 = subgroup_f_add(vals[0]);
        let s1 = subgroup_f_add(vals[1]);
        let s2 = subgroup_f_add(vals[2]);
        let s3 = subgroup_f_add(vals[3]);
        let s4 = subgroup_f_add(vals[4]);
        let s5 = subgroup_f_add(vals[5]);
        if sg_lid == 0 {
            [s0, s1, s2, s3, s4, s5]
        } else {
            [0.0; 6]
        }
    };
    #[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
    let contribs = {
        let _ = sg_lid;
        vals
    };

    let i = lid & 255;
    for k in 0..6 {
        let slot = (k * TILE_FRAMES + i).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
        scan_scratch[slot] = contribs[k];
    }
    wg_barrier();

    let mut stride = TILE_FRAMES >> 1;
    while stride > 0 {
        if i < stride {
            for k in 0..6 {
                let dst = (k * TILE_FRAMES + i).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
                let src = (k * TILE_FRAMES + i + stride).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
                scan_scratch[dst] += scan_scratch[src];
            }
        }
        wg_barrier();
        stride >>= 1;
    }

    let totals = [
        canonicalize_zero_f32_bits(scan_scratch[0]),
        canonicalize_zero_f32_bits(scan_scratch[TILE_FRAMES]),
        canonicalize_zero_f32_bits(scan_scratch[2 * TILE_FRAMES]),
        canonicalize_zero_f32_bits(scan_scratch[3 * TILE_FRAMES]),
        canonicalize_zero_f32_bits(scan_scratch[4 * TILE_FRAMES]),
        canonicalize_zero_f32_bits(scan_scratch[5 * TILE_FRAMES]),
    ];
    wg_barrier();
    totals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpt_adjoint_const_m_reduce_mat_pow_exact() {
        let mut scratch = [0.0_f32; ADJOINT_SCAN_SCRATCH_FLOATS];
        let a1 = 0.85_f32;
        let a2 = 0.12_f32;
        let a3 = 0.04_f32;
        let base = TptAdjointElem::from_step_grad(a1, a2, a3, 0.0, 0.0);

        for valid_len in [0_usize, 1, 13, 88, 256] {
            let mut ref_mat = TptAdjointElem::identity();
            for _ in 0..valid_len {
                ref_mat = compose_tpt_adjoint_scan(ref_mat, base);
            }
            let (got_l, got_r, _, _) = tpt_adjoint_const_m_reduce_256_lr(
                &mut scratch,
                0,
                valid_len,
                a1,
                a2,
                a3,
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            );
            assert!((got_l.t00 - ref_mat.t00).abs() < 1.0e-5, "len={valid_len}");
            assert!((got_l.t01 - ref_mat.t01).abs() < 1.0e-5, "len={valid_len}");
            assert!((got_l.t10 - ref_mat.t10).abs() < 1.0e-5, "len={valid_len}");
            assert!((got_l.t11 - ref_mat.t11).abs() < 1.0e-5, "len={valid_len}");
            assert_eq!(got_l.t00, got_r.t00);
        }
    }
}
