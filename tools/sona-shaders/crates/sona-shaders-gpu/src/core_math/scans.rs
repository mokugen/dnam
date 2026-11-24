#![forbid(unsafe_code)]

//! Workgroup-uniform `256`-wide associative `TPT-SVF` forward/adjoint scans, `ParaLRU` linear
//! reverse suffix scans, and deterministic `256 -> scalar` reductions in `scan_scratch` / `aux`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{
    ADJOINT_AUX_FLOATS, ADJOINT_SCAN_SCRATCH_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS, TILE_FRAMES,
};
use dnam_kernels::{
    LinearScanElem, TptAdjointElem, TptScanElem, apply_linear_scan, apply_tpt_adjoint,
    apply_tpt_init, canonicalize_zero_f32_bits, compose_linear_scan, compose_paralru_adjoint_scan,
    compose_tpt_adjoint_scan, compose_tpt_scan,
};
#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
use spirv_std::arch::subgroup_f_add;
#[cfg(target_arch = "spirv")]
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;

const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(FORWARD_SCAN_SCRATCH_FLOATS == 6 * TILE_FRAMES);
const _: () = assert!(ADJOINT_SCAN_SCRATCH_FLOATS == 6 * TILE_FRAMES);
const _: () = assert!(ADJOINT_AUX_FLOATS == TILE_FRAMES / 2);

/// Issues a workgroup memory + execution barrier on `spirv` targets (`no-op` on host CPU).
#[inline(always)]
pub fn wg_barrier() {
    #[cfg(target_arch = "spirv")]
    workgroup_memory_barrier_with_group_sync();
}

#[inline(always)]
fn store_tpt_elem(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    idx: usize,
    e: TptScanElem,
) {
    let i = idx & 255;
    scan_scratch[i] = e.m00;
    scan_scratch[TILE_FRAMES + i] = e.m01;
    scan_scratch[2 * TILE_FRAMES + i] = e.m10;
    scan_scratch[3 * TILE_FRAMES + i] = e.m11;
    scan_scratch[4 * TILE_FRAMES + i] = e.b0;
    scan_scratch[5 * TILE_FRAMES + i] = e.b1;
}

#[inline(always)]
fn load_tpt_elem(scan_scratch: &[f32; FORWARD_SCAN_SCRATCH_FLOATS], idx: usize) -> TptScanElem {
    let i = idx & 255;
    TptScanElem {
        m00: scan_scratch[i],
        m01: scan_scratch[TILE_FRAMES + i],
        m10: scan_scratch[2 * TILE_FRAMES + i],
        m11: scan_scratch[3 * TILE_FRAMES + i],
        b0: scan_scratch[4 * TILE_FRAMES + i],
        b1: scan_scratch[5 * TILE_FRAMES + i],
    }
}

#[inline(always)]
fn store_tpt_adj_elem(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    idx: usize,
    e: TptAdjointElem,
) {
    let i = idx & 255;
    scan_scratch[i] = e.t00;
    scan_scratch[TILE_FRAMES + i] = e.t01;
    scan_scratch[2 * TILE_FRAMES + i] = e.t10;
    scan_scratch[3 * TILE_FRAMES + i] = e.t11;
    scan_scratch[4 * TILE_FRAMES + i] = e.g0;
    scan_scratch[5 * TILE_FRAMES + i] = e.g1;
}

#[inline(always)]
fn load_tpt_adj_elem(
    scan_scratch: &[f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    idx: usize,
) -> TptAdjointElem {
    let i = idx & 255;
    TptAdjointElem {
        t00: scan_scratch[i],
        t01: scan_scratch[TILE_FRAMES + i],
        t10: scan_scratch[2 * TILE_FRAMES + i],
        t11: scan_scratch[3 * TILE_FRAMES + i],
        g0: scan_scratch[4 * TILE_FRAMES + i],
        g1: scan_scratch[5 * TILE_FRAMES + i],
    }
}

#[inline(always)]
fn store_linear_elem(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    offset_floats: usize,
    idx: usize,
    e: LinearScanElem,
) {
    let i = idx & 255;
    let a_idx = (offset_floats + i).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
    let b_idx = (offset_floats + TILE_FRAMES + i).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
    scan_scratch[a_idx] = e.a;
    scan_scratch[b_idx] = e.b;
}

#[inline(always)]
fn load_linear_elem(
    scan_scratch: &[f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    offset_floats: usize,
    idx: usize,
) -> LinearScanElem {
    let i = idx & 255;
    let a_idx = (offset_floats + i).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
    let b_idx = (offset_floats + TILE_FRAMES + i).min(ADJOINT_SCAN_SCRATCH_FLOATS - 1);
    LinearScanElem {
        a: scan_scratch[a_idx],
        b: scan_scratch[b_idx],
    }
}

const SUBTILE_FRAMES: usize = TILE_FRAMES / 2;

#[inline(always)]
fn store_tpt_elem_sub(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    base_floats: usize,
    idx: usize,
    e: TptScanElem,
) {
    let i = idx & 127;
    scan_scratch[base_floats + i] = e.m00;
    scan_scratch[base_floats + SUBTILE_FRAMES + i] = e.m01;
    scan_scratch[base_floats + 2 * SUBTILE_FRAMES + i] = e.m10;
    scan_scratch[base_floats + 3 * SUBTILE_FRAMES + i] = e.m11;
    scan_scratch[base_floats + 4 * SUBTILE_FRAMES + i] = e.b0;
    scan_scratch[base_floats + 5 * SUBTILE_FRAMES + i] = e.b1;
}

#[inline(always)]
fn load_tpt_elem_sub(
    scan_scratch: &[f32; FORWARD_SCAN_SCRATCH_FLOATS],
    base_floats: usize,
    idx: usize,
) -> TptScanElem {
    let i = idx & 127;
    TptScanElem {
        m00: scan_scratch[base_floats + i],
        m01: scan_scratch[base_floats + SUBTILE_FRAMES + i],
        m10: scan_scratch[base_floats + 2 * SUBTILE_FRAMES + i],
        m11: scan_scratch[base_floats + 3 * SUBTILE_FRAMES + i],
        b0: scan_scratch[base_floats + 4 * SUBTILE_FRAMES + i],
        b1: scan_scratch[base_floats + 5 * SUBTILE_FRAMES + i],
    }
}

/// Runs simultaneous `128`-frame `O(log2 128)` Hillis-Steele `TPT-SVF` exclusive scans for `L`
/// (`lid in 0..128`, `scan_scratch[0..768]`) and `R` (`lid in 128..256`, `scan_scratch[768..1536]`),
/// writing prior `(s_prev_out_0, s_prev_out_1)` and returning the inclusive sub-tile end state.
#[inline]
pub fn tpt_exclusive_scan_128_pair(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    lid: usize,
    sub_valid_len: usize,
    elem: TptScanElem,
    init: [f32; 2],
    s_prev_out_0: &mut f32,
    s_prev_out_1: &mut f32,
) -> [f32; 2] {
    let is_r = (lid >> 7) != 0;
    let i = lid & 127;
    let base = if is_r { 6 * SUBTILE_FRAMES } else { 0 };
    let mut cur = elem;
    store_tpt_elem_sub(scan_scratch, base, i, cur);
    wg_barrier();

    let mut step = 1_usize;
    while step < SUBTILE_FRAMES {
        let next = if i >= step {
            let pred = load_tpt_elem_sub(scan_scratch, base, i - step);
            compose_tpt_scan(pred, cur)
        } else {
            cur
        };
        wg_barrier();
        store_tpt_elem_sub(scan_scratch, base, i, next);
        cur = next;
        wg_barrier();
        step <<= 1;
    }

    let [p0, p1] = if i == 0 {
        init
    } else {
        let prev_elem = load_tpt_elem_sub(scan_scratch, base, i - 1);
        apply_tpt_init(prev_elem, init)
    };
    *s_prev_out_0 = p0;
    *s_prev_out_1 = p1;

    let end_state = if sub_valid_len > 0 {
        let last = if sub_valid_len <= SUBTILE_FRAMES {
            sub_valid_len - 1
        } else {
            SUBTILE_FRAMES - 1
        };
        apply_tpt_init(load_tpt_elem_sub(scan_scratch, base, last), init)
    } else {
        init
    };
    wg_barrier();
    end_state
}

#[inline(always)]
fn store_linear_sub(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    base: usize,
    idx: usize,
    e: LinearScanElem,
) {
    let i = idx & 127;
    scan_scratch[base + i] = e.a;
    scan_scratch[base + SUBTILE_FRAMES + i] = e.b;
}

#[inline(always)]
fn load_linear_sub(
    scan_scratch: &[f32; FORWARD_SCAN_SCRATCH_FLOATS],
    base: usize,
    idx: usize,
) -> LinearScanElem {
    let i = idx & 127;
    LinearScanElem {
        a: scan_scratch[base + i],
        b: scan_scratch[base + SUBTILE_FRAMES + i],
    }
}

/// Runs simultaneous `128`-frame `O(log2 128)` Hillis-Steele forward linear scans for `L`
/// (`lid in 0..128`, `scan_scratch[0..256]`) and `R` (`lid in 128..256`, `scan_scratch[256..512]`),
/// returning `(inclusive_state_t, subtile_end_state)`.
#[inline]
pub fn linear_forward_exclusive_scan_128_pair(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    lid: usize,
    sub_valid_len: usize,
    elem: LinearScanElem,
    v_init: f32,
) -> (f32, f32) {
    let is_r = (lid >> 7) != 0;
    let i = lid & 127;
    let base = if is_r { 2 * SUBTILE_FRAMES } else { 0 };
    let mut cur = elem;
    store_linear_sub(scan_scratch, base, i, cur);
    wg_barrier();

    let mut step = 1_usize;
    while step < SUBTILE_FRAMES {
        let next = if i >= step {
            let pred = load_linear_sub(scan_scratch, base, i - step);
            compose_linear_scan(pred, cur)
        } else {
            cur
        };
        wg_barrier();
        store_linear_sub(scan_scratch, base, i, next);
        cur = next;
        wg_barrier();
        step <<= 1;
    }

    let v_incl = apply_linear_scan(cur, v_init);
    let v_end = if sub_valid_len > 0 {
        let last = if sub_valid_len <= SUBTILE_FRAMES {
            sub_valid_len - 1
        } else {
            SUBTILE_FRAMES - 1
        };
        apply_linear_scan(load_linear_sub(scan_scratch, base, last), v_init)
    } else {
        v_init
    };
    wg_barrier();
    (v_incl, v_end)
}

/// Runs a `256`-wide `O(log2 256)` Hillis-Steele inclusive `TPT-SVF` scan in `scan_scratch`,
/// writing each invocation's exclusive prior state `(s_prev_out_0, s_prev_out_1)` from `init`
/// and returning the inclusive end-of-tile state (`valid_len - 1`, or `init` if `valid_len == 0`).
#[inline]
pub fn tpt_exclusive_scan_256(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    lid: usize,
    valid_len: usize,
    elem: TptScanElem,
    init: [f32; 2],
    s_prev_out_0: &mut f32,
    s_prev_out_1: &mut f32,
) -> [f32; 2] {
    let i = lid & 255;
    let mut cur = elem;
    store_tpt_elem(scan_scratch, i, cur);
    wg_barrier();

    let mut step = 1_usize;
    while step < TILE_FRAMES {
        let next = if i >= step {
            let pred = load_tpt_elem(scan_scratch, i - step);
            compose_tpt_scan(pred, cur)
        } else {
            cur
        };
        wg_barrier();
        store_tpt_elem(scan_scratch, i, next);
        cur = next;
        wg_barrier();
        step <<= 1;
    }

    let [p0, p1] = if i == 0 {
        init
    } else {
        let prev_elem = load_tpt_elem(scan_scratch, i - 1);
        apply_tpt_init(prev_elem, init)
    };
    *s_prev_out_0 = p0;
    *s_prev_out_1 = p1;

    let end_state = if valid_len > 0 {
        let last = if valid_len <= TILE_FRAMES {
            valid_len - 1
        } else {
            TILE_FRAMES - 1
        };
        apply_tpt_init(load_tpt_elem(scan_scratch, last), init)
    } else {
        init
    };
    wg_barrier();
    end_state
}

/// Runs an `8`-step reverse suffix doubling scan over `TptAdjointElem` in `scan_scratch`,
/// writing `psi_t = dL_{future > t} / ds_t` into `(psi_out_0, psi_out_1)` and returning the
/// full-tile reverse summary `[0]`.
#[inline]
pub fn tpt_adjoint_reverse_suffix_256(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem: TptAdjointElem,
    adj_s_out: [f32; 2],
    psi_out_0: &mut f32,
    psi_out_1: &mut f32,
) -> TptAdjointElem {
    let i = lid & 255;
    let mut cur = elem;
    store_tpt_adj_elem(scan_scratch, i, cur);
    wg_barrier();

    let mut step = 1_usize;
    while step < TILE_FRAMES {
        let next = if i + step < TILE_FRAMES {
            let succ = load_tpt_adj_elem(scan_scratch, i + step);
            compose_tpt_adjoint_scan(cur, succ)
        } else {
            cur
        };
        wg_barrier();
        store_tpt_adj_elem(scan_scratch, i, next);
        cur = next;
        wg_barrier();
        step <<= 1;
    }

    let after_elem = if i + 1 < TILE_FRAMES {
        load_tpt_adj_elem(scan_scratch, i + 1)
    } else {
        TptAdjointElem::identity()
    };
    let [psi0, psi1] = apply_tpt_adjoint(after_elem, adj_s_out);
    *psi_out_0 = psi0;
    *psi_out_1 = psi1;
    let summary = load_tpt_adj_elem(scan_scratch, 0);
    wg_barrier();
    summary
}

/// Simultaneously runs `L` (`scan_scratch[0..512]`) and `R` (`scan_scratch[512..1024]`) reverse
/// `ParaLRU` suffix scans in `8` shared barriers, returning `(summary_l, summary_r, lam_after_l, lam_after_r)`.
#[inline]
pub fn linear_adjoint_reverse_suffix_256_lr(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem_l: LinearScanElem,
    elem_r: LinearScanElem,
    adj_v_out_l: f32,
    adj_v_out_r: f32,
) -> (LinearScanElem, LinearScanElem, f32, f32) {
    let i = lid & 255;
    let mut cur_l = elem_l;
    let mut cur_r = elem_r;
    store_linear_elem(scan_scratch, 0, i, cur_l);
    store_linear_elem(scan_scratch, 2 * TILE_FRAMES, i, cur_r);
    wg_barrier();

    let mut step = 1_usize;
    while step < TILE_FRAMES {
        let (next_l, next_r) = if i + step < TILE_FRAMES {
            let succ_l = load_linear_elem(scan_scratch, 0, i + step);
            let succ_r = load_linear_elem(scan_scratch, 2 * TILE_FRAMES, i + step);
            (
                compose_paralru_adjoint_scan(cur_l, succ_l),
                compose_paralru_adjoint_scan(cur_r, succ_r),
            )
        } else {
            (cur_l, cur_r)
        };
        wg_barrier();
        store_linear_elem(scan_scratch, 0, i, next_l);
        store_linear_elem(scan_scratch, 2 * TILE_FRAMES, i, next_r);
        cur_l = next_l;
        cur_r = next_r;
        wg_barrier();
        step <<= 1;
    }

    let (after_l, after_r) = if i + 1 < TILE_FRAMES {
        (
            load_linear_elem(scan_scratch, 0, i + 1),
            load_linear_elem(scan_scratch, 2 * TILE_FRAMES, i + 1),
        )
    } else {
        (LinearScanElem::identity(), LinearScanElem::identity())
    };
    let lam_after_l = apply_linear_scan(after_l, adj_v_out_l);
    let lam_after_r = apply_linear_scan(after_r, adj_v_out_r);
    let summary_l = load_linear_elem(scan_scratch, 0, 0);
    let summary_r = load_linear_elem(scan_scratch, 2 * TILE_FRAMES, 0);
    wg_barrier();
    (summary_l, summary_r, lam_after_l, lam_after_r)
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
