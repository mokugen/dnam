#![forbid(unsafe_code)]

//! `adjoint_scan_256` `pass_mode = 1`: level-2 `O(log2 num_tiles)` reverse suffix scan in
//! `256`-tile chunks across `Slot 7` (`tile_adj_boundaries`), resolving exact cross-tile
//! `(adj_s1_out, adj_s2_out)` and `adj_v_out` boundary sensitivities with zero BPTT truncation
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{ADJOINT_SCAN_SCRATCH_FLOATS, AdjointScanUniforms, TILE_FRAMES};
use crate::core_math::{
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, linear_adjoint_reverse_suffix_256_lr, load_vec4, store_vec4,
    tpt_adjoint_reverse_suffix_256,
};
use dnam_kernels::{LinearScanElem, TptAdjointElem, apply_linear_scan, apply_tpt_adjoint};
use spirv_std::glam::Vec4;

/// Executes `adjoint_scan_256` `pass_mode = 1` across `num_tiles` in reverse `256`-tile blocks.
#[inline]
pub fn run_pass1(
    q_global: usize,
    lid: usize,
    uniforms: &AdjointScanUniforms,
    tile_adj_boundaries: &mut [Vec4],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
) {
    let i = lid & 255;
    let num_tiles = uniforms.num_tiles as usize;
    if num_tiles == 0 {
        return;
    }
    let num_blocks = ((num_tiles - 1) >> 8) + 1;

    for c in 0..4 {
        let mut carry_v_l = 0.0_f32;
        let mut carry_v_r = 0.0_f32;
        let mut carry_s_l = [0.0_f32; 2];
        let mut carry_s_r = [0.0_f32; 2];

        for b_rev in 0..num_blocks {
            let b = (num_blocks - 1) - b_rev;
            let tile_idx = b * TILE_FRAMES + i;
            let in_range = tile_idx < num_tiles;
            let slot = (q_global * num_tiles + tile_idx) * TILE_ADJ_BOUNDARY_VEC4_STRIDE + c * 4;

            let (lru_elem_l, lru_elem_r, tpt_elem_l, tpt_elem_r) = if in_range {
                let s0 = load_vec4(tile_adj_boundaries, slot);
                let s1 = load_vec4(tile_adj_boundaries, slot + 1);
                let s2 = load_vec4(tile_adj_boundaries, slot + 2);
                (
                    LinearScanElem { a: s2.x, b: s2.y },
                    LinearScanElem { a: s2.z, b: s2.w },
                    TptAdjointElem {
                        t00: s0.x,
                        t01: s0.y,
                        t10: s0.z,
                        t11: s0.w,
                        g0: s1.x,
                        g1: s1.y,
                    },
                    TptAdjointElem {
                        t00: s0.x,
                        t01: s0.y,
                        t10: s0.z,
                        t11: s0.w,
                        g0: s1.z,
                        g1: s1.w,
                    },
                )
            } else {
                (
                    LinearScanElem::identity(),
                    LinearScanElem::identity(),
                    TptAdjointElem::identity(),
                    TptAdjointElem::identity(),
                )
            };

            let (full_v_l, full_v_r, adj_v_out_l, adj_v_out_r) =
                linear_adjoint_reverse_suffix_256_lr(
                    scan_scratch,
                    i,
                    lru_elem_l,
                    lru_elem_r,
                    carry_v_l,
                    carry_v_r,
                );
            if in_range {
                store_vec4(
                    tile_adj_boundaries,
                    slot + 2,
                    Vec4::new(adj_v_out_l, adj_v_out_r, 0.0, 0.0),
                );
            }
            carry_v_l = apply_linear_scan(full_v_l, carry_v_l);
            carry_v_r = apply_linear_scan(full_v_r, carry_v_r);

            let mut adj_s1_l_out = 0.0_f32;
            let mut adj_s2_l_out = 0.0_f32;
            let full_s_l = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                tpt_elem_l,
                carry_s_l,
                &mut adj_s1_l_out,
                &mut adj_s2_l_out,
            );

            let mut adj_s1_r_out = 0.0_f32;
            let mut adj_s2_r_out = 0.0_f32;
            let full_s_r = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                tpt_elem_r,
                carry_s_r,
                &mut adj_s1_r_out,
                &mut adj_s2_r_out,
            );

            if in_range {
                store_vec4(
                    tile_adj_boundaries,
                    slot,
                    Vec4::new(adj_s1_l_out, adj_s2_l_out, adj_s1_r_out, adj_s2_r_out),
                );
            }
            carry_s_l = apply_tpt_adjoint(full_s_l, carry_s_l);
            carry_s_r = apply_tpt_adjoint(full_s_r, carry_s_r);
        }
    }
}
