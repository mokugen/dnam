#![forbid(unsafe_code)]

//! Exact 3-pass `256`-wide reverse-BPTT adjoint kernel (`adjoint_scan_256_impl`) executing
//! inside `14,848 B` (`14.5 KiB`) workgroup SRAM across `pass_mode` `0`, `1`, and `2`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod common;
pub mod pass0;
pub mod pass1;
pub mod pass2;
pub mod pullback;

use crate::abi::{
    ADJOINT_AUX_FLOATS, ADJOINT_S_PREV2_VEC4S, ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S,
    ADJOINT_V_TRAJ2_VEC2S, AdjointScanUniforms,
};
use spirv_std::glam::{UVec3, Vec2, Vec4};

/// Executes one pass (`pass_mode` in `{0, 1, 2}`) of the exact `adjoint_scan_256` BPTT kernel
/// inside `14,848 B` SRAM with zero `f32` atomics and zero truncation across tiles.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn adjoint_scan_256_impl(
    gid: UVec3,
    wg_id: UVec3,
    num_wg: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &AdjointScanUniforms,
    core_inputs: &[Vec4],
    quad_params: &[Vec4],
    tile_checkpoints: &[Vec4],
    d_loss_d_features: &[Vec4],
    d_loss_d_inputs: &mut [Vec4],
    param_grads: &mut [Vec4],
    tile_adj_boundaries: &mut [Vec4],
    feature_outputs: &[Vec4],
    s_prev2: &mut [Vec4; ADJOINT_S_PREV2_VEC4S],
    u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    aux: &mut [f32; ADJOINT_AUX_FLOATS],
) {
    let _ = gid;
    let lid = (lid_u32 as usize) & 255;
    let quads_in_block = (num_wg.x as usize).max(1);
    let q_local = wg_id.x as usize;
    let tile = wg_id.y as usize;
    let stem = wg_id.z as usize;
    let q_global = stem * quads_in_block + q_local;
    let num_stems = (num_wg.z as usize).max(uniforms.num_stems as usize).max(1);
    let total_q = quads_in_block * num_stems;

    match uniforms.pass_mode {
        0 => pass0::run_pass0(
            q_global,
            tile,
            lid,
            uniforms,
            core_inputs,
            quad_params,
            tile_checkpoints,
            d_loss_d_features,
            tile_adj_boundaries,
            feature_outputs,
            s_prev2,
            u_tone2,
            v_traj2,
            scan_scratch,
        ),
        1 => pass1::run_pass1(q_global, lid, uniforms, tile_adj_boundaries, scan_scratch),
        _ => pass2::run_pass2(
            q_global,
            total_q,
            tile,
            lid,
            sg_lid,
            uniforms,
            core_inputs,
            quad_params,
            tile_checkpoints,
            d_loss_d_features,
            d_loss_d_inputs,
            param_grads,
            tile_adj_boundaries,
            feature_outputs,
            s_prev2,
            u_tone2,
            v_traj2,
            scan_scratch,
            aux,
        ),
    }
}
