#![forbid(unsafe_code)]

//! Workgroup-uniform `256`-wide `TPT-SVF` constant-`M` & general scans (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::wg_barrier;
use crate::abi::{ADJOINT_SCAN_SCRATCH_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS, TILE_FRAMES};
use dnam_kernels::{
    TptAdjointElem, TptScanElem, apply_tpt_adjoint, apply_tpt_init, compose_tpt_adjoint_scan,
    compose_tpt_scan,
};

const SUBTILE_FRAMES: usize = TILE_FRAMES / 2;

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

#[inline(always)]
fn square_mat2(c00: f32, c01: f32, c10: f32, c11: f32) -> (f32, f32, f32, f32) {
    (
        c00 * c00 + c01 * c10,
        c00 * c01 + c01 * c11,
        c10 * c00 + c11 * c10,
        c10 * c01 + c11 * c11,
    )
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

/// Runs simultaneous `128`-frame constant-`M` register-squaring `TPT-SVF` scans for `L`
/// (`lid in 0..128`, `scan_scratch[0..256]`) and `R` (`lid in 128..256`, `scan_scratch[256..512]`),
/// holding `M^(2^k)` in 4 registers (`2,048 B` SRAM) and returning the inclusive sub-tile end state.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn tpt_const_m_scan_128_pair(
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    lid: usize,
    sub_valid_len: usize,
    a1: f32,
    a2: f32,
    a3: f32,
    u_my: f32,
    active128: bool,
    init: [f32; 2],
    s_prev_out_0: &mut f32,
    s_prev_out_1: &mut f32,
) -> [f32; 2] {
    let is_r = (lid >> 7) != 0;
    let i = lid & 127;
    let base = if is_r { 2 * SUBTILE_FRAMES } else { 0 };
    let two_a2 = a2 + a2;
    let two_a3 = a3 + a3;
    let mut c00 = (a1 + a1) - 1.0;
    let mut c01 = -two_a2;
    let mut c10 = two_a2;
    let mut c11 = 1.0 - two_a3;

    let mut cur0 = if active128 { two_a2 * u_my } else { 0.0 };
    let mut cur1 = if active128 { two_a3 * u_my } else { 0.0 };
    if i == 0 && sub_valid_len > 0 {
        cur0 += c00 * init[0] + c01 * init[1];
        cur1 += c10 * init[0] + c11 * init[1];
    }
    scan_scratch[base + i] = cur0;
    scan_scratch[base + SUBTILE_FRAMES + i] = cur1;
    wg_barrier();

    let mut step = 1_usize;
    while step < SUBTILE_FRAMES {
        let (next0, next1) = if i >= step {
            let pred_i = (i - step) & 127;
            let pb0 = scan_scratch[base + pred_i];
            let pb1 = scan_scratch[base + SUBTILE_FRAMES + pred_i];
            (c00 * pb0 + c01 * pb1 + cur0, c10 * pb0 + c11 * pb1 + cur1)
        } else {
            (cur0, cur1)
        };
        wg_barrier();
        scan_scratch[base + i] = next0;
        scan_scratch[base + SUBTILE_FRAMES + i] = next1;
        cur0 = next0;
        cur1 = next1;
        (c00, c01, c10, c11) = square_mat2(c00, c01, c10, c11);
        wg_barrier();
        step <<= 1;
    }

    if i == 0 {
        *s_prev_out_0 = init[0];
        *s_prev_out_1 = init[1];
    } else {
        let prev = (i - 1) & 127;
        *s_prev_out_0 = scan_scratch[base + prev];
        *s_prev_out_1 = scan_scratch[base + SUBTILE_FRAMES + prev];
    }

    let end_state = if sub_valid_len > 0 {
        let last = if sub_valid_len <= SUBTILE_FRAMES {
            sub_valid_len - 1
        } else {
            SUBTILE_FRAMES - 1
        };
        [
            scan_scratch[base + last],
            scan_scratch[base + SUBTILE_FRAMES + last],
        ]
    } else {
        init
    };
    wg_barrier();
    end_state
}

/// Runs simultaneous `256`-frame constant-`M` register-squaring `TPT-SVF` forward scans for `L`
/// (`scan_scratch[0..512]`) and `R` (`scan_scratch[512..1024]`) in one pass (`4,096 B` SRAM),
/// returning `(s_prev_l, s_prev_r)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn tpt_const_m_scan_256_lr(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    valid_len: usize,
    a1: f32,
    a2: f32,
    a3: f32,
    u_l: f32,
    u_r: f32,
    is_active: bool,
    init_l: [f32; 2],
    init_r: [f32; 2],
) -> ([f32; 2], [f32; 2]) {
    let i = lid & 255;
    let two_a2 = a2 + a2;
    let two_a3 = a3 + a3;
    let mut c00 = (a1 + a1) - 1.0;
    let mut c01 = -two_a2;
    let mut c10 = two_a2;
    let mut c11 = 1.0 - two_a3;

    let mut cur_l0 = if is_active { two_a2 * u_l } else { 0.0 };
    let mut cur_l1 = if is_active { two_a3 * u_l } else { 0.0 };
    let mut cur_r0 = if is_active { two_a2 * u_r } else { 0.0 };
    let mut cur_r1 = if is_active { two_a3 * u_r } else { 0.0 };

    if i == 0 && valid_len > 0 {
        cur_l0 += c00 * init_l[0] + c01 * init_l[1];
        cur_l1 += c10 * init_l[0] + c11 * init_l[1];
        cur_r0 += c00 * init_r[0] + c01 * init_r[1];
        cur_r1 += c10 * init_r[0] + c11 * init_r[1];
    }

    scan_scratch[i] = cur_l0;
    scan_scratch[TILE_FRAMES + i] = cur_l1;
    scan_scratch[2 * TILE_FRAMES + i] = cur_r0;
    scan_scratch[3 * TILE_FRAMES + i] = cur_r1;
    wg_barrier();

    let mut step = 1_usize;
    while step < TILE_FRAMES {
        let (nl0, nl1, nr0, nr1) = if i >= step {
            let p = (i - step) & 255;
            let pl0 = scan_scratch[p];
            let pl1 = scan_scratch[TILE_FRAMES + p];
            let pr0 = scan_scratch[2 * TILE_FRAMES + p];
            let pr1 = scan_scratch[3 * TILE_FRAMES + p];
            (
                c00 * pl0 + c01 * pl1 + cur_l0,
                c10 * pl0 + c11 * pl1 + cur_l1,
                c00 * pr0 + c01 * pr1 + cur_r0,
                c10 * pr0 + c11 * pr1 + cur_r1,
            )
        } else {
            (cur_l0, cur_l1, cur_r0, cur_r1)
        };
        wg_barrier();
        scan_scratch[i] = nl0;
        scan_scratch[TILE_FRAMES + i] = nl1;
        scan_scratch[2 * TILE_FRAMES + i] = nr0;
        scan_scratch[3 * TILE_FRAMES + i] = nr1;
        cur_l0 = nl0;
        cur_l1 = nl1;
        cur_r0 = nr0;
        cur_r1 = nr1;
        (c00, c01, c10, c11) = square_mat2(c00, c01, c10, c11);
        wg_barrier();
        step <<= 1;
    }

    let (s_prev_l, s_prev_r) = if i == 0 {
        (init_l, init_r)
    } else {
        let prev = (i - 1) & 255;
        (
            [scan_scratch[prev], scan_scratch[TILE_FRAMES + prev]],
            [
                scan_scratch[2 * TILE_FRAMES + prev],
                scan_scratch[3 * TILE_FRAMES + prev],
            ],
        )
    };
    wg_barrier();
    (s_prev_l, s_prev_r)
}

/// Runs an `8`-step forward exclusive Hillis-Steele `256`-wide `TPT-SVF` scan over `TptScanElem`
/// in `scan_scratch[0..1536]` for a single channel (`L` or `R`), returning `(s1_prev, s2_prev)`.
#[inline]
pub fn tpt_forward_exclusive_scan_256(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem: TptScanElem,
    init: [f32; 2],
) -> [f32; 2] {
    let i = lid & 255;
    let mut cur = elem;
    scan_scratch[i] = cur.m00;
    scan_scratch[TILE_FRAMES + i] = cur.m01;
    scan_scratch[2 * TILE_FRAMES + i] = cur.m10;
    scan_scratch[3 * TILE_FRAMES + i] = cur.m11;
    scan_scratch[4 * TILE_FRAMES + i] = cur.b0;
    scan_scratch[5 * TILE_FRAMES + i] = cur.b1;
    wg_barrier();

    let mut step = 1_usize;
    while step < TILE_FRAMES {
        let next = if i >= step {
            let p = (i - step) & 255;
            let pred = TptScanElem {
                m00: scan_scratch[p],
                m01: scan_scratch[TILE_FRAMES + p],
                m10: scan_scratch[2 * TILE_FRAMES + p],
                m11: scan_scratch[3 * TILE_FRAMES + p],
                b0: scan_scratch[4 * TILE_FRAMES + p],
                b1: scan_scratch[5 * TILE_FRAMES + p],
            };
            compose_tpt_scan(pred, cur)
        } else {
            cur
        };
        wg_barrier();
        scan_scratch[i] = next.m00;
        scan_scratch[TILE_FRAMES + i] = next.m01;
        scan_scratch[2 * TILE_FRAMES + i] = next.m10;
        scan_scratch[3 * TILE_FRAMES + i] = next.m11;
        scan_scratch[4 * TILE_FRAMES + i] = next.b0;
        scan_scratch[5 * TILE_FRAMES + i] = next.b1;
        cur = next;
        wg_barrier();
        step <<= 1;
    }

    let s_prev = if i == 0 {
        init
    } else {
        let p = (i - 1) & 255;
        let prev_elem = TptScanElem {
            m00: scan_scratch[p],
            m01: scan_scratch[TILE_FRAMES + p],
            m10: scan_scratch[2 * TILE_FRAMES + p],
            m11: scan_scratch[3 * TILE_FRAMES + p],
            b0: scan_scratch[4 * TILE_FRAMES + p],
            b1: scan_scratch[5 * TILE_FRAMES + p],
        };
        apply_tpt_init(prev_elem, init)
    };
    wg_barrier();
    s_prev
}

/// Runs an `8`-step reverse suffix doubling scan over `TptAdjointElem` in `scan_scratch`,
/// returning `(summary_0, after_elem_i)` where `after_elem_i` is the exclusive future suffix
/// `[i + 1 .. 256)` (`TptAdjointElem::identity()` at `i == 255`).
#[inline]
pub fn tpt_adjoint_reverse_suffix_with_after_256(
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem: TptAdjointElem,
) -> (TptAdjointElem, TptAdjointElem) {
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
    let summary = load_tpt_adj_elem(scan_scratch, 0);
    wg_barrier();
    (summary, after_elem)
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
    let (summary, after_elem) = tpt_adjoint_reverse_suffix_with_after_256(scan_scratch, lid, elem);
    let [psi0, psi1] = apply_tpt_adjoint(after_elem, adj_s_out);
    *psi_out_0 = psi0;
    *psi_out_1 = psi1;
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpt_const_m_scan_128_pair_matches_exclusive_scan() {
        let a1 = 0.88_f32;
        let a2 = 0.14_f32;
        let a3 = 0.03_f32;
        let two_a2 = a2 + a2;
        let two_a3 = a3 + a3;
        let m00 = (a1 + a1) - 1.0;
        let m01 = -two_a2;
        let m10 = two_a2;
        let m11 = 1.0 - two_a3;
        let init_l = [0.25_f32, -0.15];
        let init_r = [-0.10_f32, 0.30];

        for (lid, init, u0) in [(0_usize, init_l, 0.08_f32), (128_usize, init_r, -0.11_f32)] {
            for valid_len in [0_usize, 1] {
                let mut ref_scratch = [0.0_f32; FORWARD_SCAN_SCRATCH_FLOATS];
                let mut cm_scratch = [0.0_f32; FORWARD_SCAN_SCRATCH_FLOATS];
                let elem = TptScanElem {
                    m00,
                    m01,
                    m10,
                    m11,
                    b0: two_a2 * u0,
                    b1: two_a3 * u0,
                };
                let (mut ref_sp0, mut ref_sp1) = (0.0_f32, 0.0_f32);
                let ref_end = tpt_exclusive_scan_128_pair(
                    &mut ref_scratch,
                    lid,
                    valid_len,
                    elem,
                    init,
                    &mut ref_sp0,
                    &mut ref_sp1,
                );
                let (mut cm_sp0, mut cm_sp1) = (0.0_f32, 0.0_f32);
                let cm_end = tpt_const_m_scan_128_pair(
                    &mut cm_scratch,
                    lid,
                    valid_len,
                    a1,
                    a2,
                    a3,
                    u0,
                    valid_len > 0,
                    init,
                    &mut cm_sp0,
                    &mut cm_sp1,
                );
                assert_eq!([cm_sp0, cm_sp1], init);
                assert_eq!([ref_sp0, ref_sp1], init);
                assert!((cm_end[0] - ref_end[0]).abs() < 1.0e-6);
                assert!((cm_end[1] - ref_end[1]).abs() < 1.0e-6);
                if valid_len == 0 {
                    assert_eq!(cm_end, init);
                } else {
                    let exp = [
                        m00 * init[0] + m01 * init[1] + two_a2 * u0,
                        m10 * init[0] + m11 * init[1] + two_a3 * u0,
                    ];
                    assert!((cm_end[0] - exp[0]).abs() < 1.0e-6);
                    assert!((cm_end[1] - exp[1]).abs() < 1.0e-6);
                }
            }
        }
    }
}
