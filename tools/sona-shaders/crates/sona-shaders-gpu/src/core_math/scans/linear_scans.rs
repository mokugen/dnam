#![forbid(unsafe_code)]

//! Workgroup-uniform `128`/`256`-wide linear affine forward & reverse-suffix `ParaLRU` scans
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::wg_barrier;
use crate::abi::{ADJOINT_SCAN_SCRATCH_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS, TILE_FRAMES};
use dnam_kernels::{
    LinearScanElem, apply_linear_scan, compose_linear_scan, compose_paralru_adjoint_scan,
};

const SUBTILE_FRAMES: usize = TILE_FRAMES / 2;

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

/// Runs simultaneous `256`-frame `O(log2 256)` Hillis-Steele inclusive forward linear scans for
/// `L` (`scan_scratch[0..512]`) and `R` (`scan_scratch[512..1024]`) (`4,096 B` SRAM), returning
/// `(v_incl_l, v_incl_r)` at frame `lid & 255`.
#[inline]
pub fn linear_forward_inclusive_scan_256_lr(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem_l: LinearScanElem,
    elem_r: LinearScanElem,
    v_init_l: f32,
    v_init_r: f32,
) -> (f32, f32) {
    let i = lid & 255;
    let mut cur_l = elem_l;
    let mut cur_r = elem_r;
    store_linear_elem(scan_scratch, 0, i, cur_l);
    store_linear_elem(scan_scratch, 2 * TILE_FRAMES, i, cur_r);
    wg_barrier();

    let mut step = 1_usize;
    while step < TILE_FRAMES {
        let (next_l, next_r) = if i >= step {
            let pred_l = load_linear_elem(scan_scratch, 0, i - step);
            let pred_r = load_linear_elem(scan_scratch, 2 * TILE_FRAMES, i - step);
            (
                compose_linear_scan(pred_l, cur_l),
                compose_linear_scan(pred_r, cur_r),
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

    (
        apply_linear_scan(cur_l, v_init_l),
        apply_linear_scan(cur_r, v_init_r),
    )
}

/// Simultaneously runs `L` (`scan_scratch[0..512]`) and `R` (`scan_scratch[512..1024]`) reverse
/// `ParaLRU` suffix scans in `8` doubling steps, returning
/// `(summary_l, summary_r, lam_after_l, lam_after_r, after_l_a, after_r_a)`.
#[inline]
pub fn linear_adjoint_reverse_suffix_256_lr(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem_l: LinearScanElem,
    elem_r: LinearScanElem,
    adj_v_out_l: f32,
    adj_v_out_r: f32,
) -> (LinearScanElem, LinearScanElem, f32, f32, f32, f32) {
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
    (
        summary_l,
        summary_r,
        lam_after_l,
        lam_after_r,
        after_l.a,
        after_r.a,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scan_128_and_256_zero_identity_boundaries() {
        let mut fwd_s = [0.0_f32; FORWARD_SCAN_SCRATCH_FLOATS];
        let (vi, ve) = linear_forward_exclusive_scan_128_pair(
            &mut fwd_s,
            0,
            0,
            LinearScanElem { a: 0.75, b: 0.25 },
            1.5,
        );
        assert!((vi - (0.75 * 1.5 + 0.25)).abs() < 1.0e-6);
        assert_eq!(ve, 1.5);

        let mut adj_s = [0.0_f32; ADJOINT_SCAN_SCRATCH_FLOATS];
        let (vl, vr) = linear_forward_inclusive_scan_256_lr(
            &mut adj_s,
            0,
            LinearScanElem { a: 0.5, b: 0.25 },
            LinearScanElem { a: 0.8, b: -0.1 },
            2.0,
            -1.0,
        );
        assert!((vl - 1.25).abs() < 1.0e-6);
        assert!((vr - (-0.9)).abs() < 1.0e-6);

        let (sl, sr, ll, lr, al, ar) = linear_adjoint_reverse_suffix_256_lr(
            &mut adj_s,
            255,
            LinearScanElem { a: 0.6, b: 0.3 },
            LinearScanElem { a: 0.4, b: -0.2 },
            1.0,
            1.0,
        );
        assert_eq!(ll, 1.0);
        assert_eq!(lr, 1.0);
        assert_eq!(al, 1.0);
        assert_eq!(ar, 1.0);
        let _ = (sl, sr);
    }
}
