#![forbid(unsafe_code)]

//! `adjoint_scan_256` `pass_mode = 1`: level-2 `O(log2 num_tiles)` reverse suffix scan in
//! `256`-tile chunks across `11`-Vec4-per-lane `Slot 7` (`tile_adj_boundaries`), seeded by and
//! exporting into double-buffered `24`-Vec4 wave header (`[0..tile_base)`) with cross-operator
//! `ParaLRU -> Pass B -> Pass A` Pred-Corr PHS coupling injection
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{
    ADJOINT_SCAN_SCRATCH_FLOATS, AdjointScanUniforms, BLOCK_IDX_DIRECT_LINEAR,
    EXIT_ADJ_BOUNDARY_VEC4_STRIDE, TILE_FRAMES,
};
use crate::core_math::{
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_LANE_VEC4_STRIDE, get_lane,
    linear_adjoint_reverse_suffix_256_lr, load_vec4, store_vec4, tpt_adjoint_reverse_suffix_256,
};
use dnam_kernels::{LinearScanElem, TptAdjointElem, apply_linear_scan, apply_tpt_adjoint};
use spirv_std::glam::Vec4;

/// Executes `adjoint_scan_256` `pass_mode = 1` across `num_tiles` in reverse `256`-tile blocks.
#[inline]
pub fn run_pass1(
    q_wave: usize,
    tile_base: usize,
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
    let is_phs = uniforms.block_idx != BLOCK_IDX_DIRECT_LINEAR && (uniforms.block_idx & 0x3) == 0;
    let num_blocks = ((num_tiles - 1) >> 8) + 1;
    let exit_base = q_wave * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;
    let in_exit_vl = load_vec4(tile_adj_boundaries, exit_base + 4);
    let in_exit_vr = load_vec4(tile_adj_boundaries, exit_base + 5);

    let mut out_entry_vl = [0.0_f32; 4];
    let mut out_entry_vr = [0.0_f32; 4];
    let mut out_entry_rprev_l = [0.0_f32; 4];
    let mut out_entry_rprev_r = [0.0_f32; 4];

    for c in 0..4 {
        let s_exit = load_vec4(tile_adj_boundaries, exit_base + c);
        let mut carry_s_l = [s_exit.x, s_exit.y];
        let mut carry_s_r = [s_exit.z, s_exit.w];
        let mut carry_v_l = get_lane(in_exit_vl, c);
        let mut carry_v_r = get_lane(in_exit_vr, c);
        let s_lin_exit = if is_phs {
            load_vec4(tile_adj_boundaries, exit_base + 8 + c)
        } else {
            Vec4::ZERO
        };
        let mut carry_s_lin_l = [s_lin_exit.x, s_lin_exit.y];
        let mut carry_s_lin_r = [s_lin_exit.z, s_lin_exit.w];

        for b_rev in 0..num_blocks {
            let b = (num_blocks - 1) - b_rev;
            let tile_idx = b * TILE_FRAMES + i;
            let in_range = tile_idx < num_tiles;
            let slot = tile_base
                + (q_wave * num_tiles + tile_idx) * TILE_ADJ_BOUNDARY_VEC4_STRIDE
                + c * TILE_ADJ_LANE_VEC4_STRIDE;

            let (lru_elem_l, lru_elem_r, s0, s1, s4, s5_kvs) = if in_range {
                let v0 = load_vec4(tile_adj_boundaries, slot);
                let v1 = load_vec4(tile_adj_boundaries, slot + 1);
                let v2 = load_vec4(tile_adj_boundaries, slot + 2);
                let v4 = load_vec4(tile_adj_boundaries, slot + 4);
                let v5 = load_vec4(tile_adj_boundaries, slot + 5);
                (
                    LinearScanElem { a: v2.x, b: v2.y },
                    LinearScanElem { a: v2.z, b: v2.w },
                    v0,
                    v1,
                    v4,
                    v5,
                )
            } else {
                (
                    LinearScanElem::identity(),
                    LinearScanElem::identity(),
                    Vec4::ZERO,
                    Vec4::ZERO,
                    Vec4::ZERO,
                    Vec4::ZERO,
                )
            };

            let (full_v_l, full_v_r, adj_v_out_l, adj_v_out_r, _, _) =
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

            let (tpt_elem_l, tpt_elem_r) = if in_range {
                (
                    TptAdjointElem {
                        t00: s0.x,
                        t01: s0.y,
                        t10: s0.z,
                        t11: s0.w,
                        g0: s1.x + s5_kvs.x * adj_v_out_l,
                        g1: s1.y + s5_kvs.y * adj_v_out_l,
                    },
                    TptAdjointElem {
                        t00: s4.x,
                        t01: s4.y,
                        t10: s4.z,
                        t11: s4.w,
                        g0: s1.z + s5_kvs.z * adj_v_out_r,
                        g1: s1.w + s5_kvs.w * adj_v_out_r,
                    },
                )
            } else {
                (TptAdjointElem::identity(), TptAdjointElem::identity())
            };

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

            if is_phs {
                let (lin_elem_l, lin_elem_r) = if in_range {
                    let s6 = load_vec4(tile_adj_boundaries, slot + 6);
                    let s7 = load_vec4(tile_adj_boundaries, slot + 7);
                    let s8 = load_vec4(tile_adj_boundaries, slot + 8);
                    let s9 = load_vec4(tile_adj_boundaries, slot + 9);
                    let s10 = load_vec4(tile_adj_boundaries, slot + 10);
                    (
                        TptAdjointElem {
                            t00: s6.x,
                            t01: s6.y,
                            t10: s6.z,
                            t11: s6.w,
                            g0: s7.x
                                + s8.x * adj_v_out_l
                                + s9.x * adj_s1_l_out
                                + s10.x * adj_s2_l_out,
                            g1: s7.y
                                + s8.y * adj_v_out_l
                                + s9.y * adj_s1_l_out
                                + s10.y * adj_s2_l_out,
                        },
                        TptAdjointElem {
                            t00: s6.x,
                            t01: s6.y,
                            t10: s6.z,
                            t11: s6.w,
                            g0: s7.z
                                + s8.z * adj_v_out_r
                                + s9.z * adj_s1_r_out
                                + s10.z * adj_s2_r_out,
                            g1: s7.w
                                + s8.w * adj_v_out_r
                                + s9.w * adj_s1_r_out
                                + s10.w * adj_s2_r_out,
                        },
                    )
                } else {
                    (TptAdjointElem::identity(), TptAdjointElem::identity())
                };

                let mut adj_s1_lin_l_out = 0.0_f32;
                let mut adj_s2_lin_l_out = 0.0_f32;
                let full_lin_l = tpt_adjoint_reverse_suffix_256(
                    scan_scratch,
                    i,
                    lin_elem_l,
                    carry_s_lin_l,
                    &mut adj_s1_lin_l_out,
                    &mut adj_s2_lin_l_out,
                );

                let mut adj_s1_lin_r_out = 0.0_f32;
                let mut adj_s2_lin_r_out = 0.0_f32;
                let full_lin_r = tpt_adjoint_reverse_suffix_256(
                    scan_scratch,
                    i,
                    lin_elem_r,
                    carry_s_lin_r,
                    &mut adj_s1_lin_r_out,
                    &mut adj_s2_lin_r_out,
                );

                if in_range {
                    store_vec4(
                        tile_adj_boundaries,
                        slot + 6,
                        Vec4::new(
                            adj_s1_lin_l_out,
                            adj_s2_lin_l_out,
                            adj_s1_lin_r_out,
                            adj_s2_lin_r_out,
                        ),
                    );
                }
                carry_s_lin_l = apply_tpt_adjoint(full_lin_l, carry_s_lin_l);
                carry_s_lin_r = apply_tpt_adjoint(full_lin_r, carry_s_lin_r);
            }
        }

        if i == 0 {
            store_vec4(
                tile_adj_boundaries,
                exit_base + 12 + c,
                Vec4::new(carry_s_l[0], carry_s_l[1], carry_s_r[0], carry_s_r[1]),
            );
            store_vec4(
                tile_adj_boundaries,
                exit_base + 20 + c,
                if is_phs {
                    Vec4::new(
                        carry_s_lin_l[0],
                        carry_s_lin_l[1],
                        carry_s_lin_r[0],
                        carry_s_lin_r[1],
                    )
                } else {
                    Vec4::ZERO
                },
            );
            out_entry_vl[c] = carry_v_l;
            out_entry_vr[c] = carry_v_r;
            let tile0_slot3 = tile_base
                + q_wave * num_tiles * TILE_ADJ_BOUNDARY_VEC4_STRIDE
                + c * TILE_ADJ_LANE_VEC4_STRIDE
                + 3;
            let tile0_s3 = load_vec4(tile_adj_boundaries, tile0_slot3);
            out_entry_rprev_l[c] = tile0_s3.z;
            out_entry_rprev_r[c] = tile0_s3.w;
        }
    }

    if i == 0 {
        store_vec4(
            tile_adj_boundaries,
            exit_base + 16,
            Vec4::from(out_entry_vl),
        );
        store_vec4(
            tile_adj_boundaries,
            exit_base + 17,
            Vec4::from(out_entry_vr),
        );
        store_vec4(
            tile_adj_boundaries,
            exit_base + 18,
            Vec4::from(out_entry_rprev_l),
        );
        store_vec4(
            tile_adj_boundaries,
            exit_base + 19,
            Vec4::from(out_entry_rprev_r),
        );
    }
}
