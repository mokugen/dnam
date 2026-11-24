#![cfg_attr(target_arch = "spirv", no_std)]
#![forbid(unsafe_code)]

//! Single-source SPIR-V compute entry points for SONA v3.0 WebGPU training
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod abi;
mod adjoint_core;
mod core_math;
mod fused_core;
mod gabor_bezout;
mod inner_opt;
mod ozaki_gram;
mod prescreen;
mod reduce;
mod tile_mix;

use abi::{
    ADJOINT_AUX_FLOATS, ADJOINT_S_PREV2_VEC4S, ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S,
    ADJOINT_V_TRAJ2_VEC2S, AdjointScanUniforms, FORWARD_AUX_SCRATCH_FLOATS,
    FORWARD_LANE_BUF_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS, FusedScanUniforms, GaborBezoutUniforms,
    InnerOptUniforms, OzakiGramUniforms, PrescreenUniforms, TILE_MIX_REDUCE_A_FLOATS,
    TILE_MIX_REDUCE_B_FLOATS, TileMixUniforms,
};
use adjoint_core::adjoint_scan_256_impl;
use fused_core::fused_core_scan_256_impl;
use gabor_bezout::{
    GABOR_BUF_A_VEC4S, GABOR_BUF_B_VEC4S, GABOR_BUF_C_VEC2S, GABOR_REDUCE_FLOATS,
    gabor_bezout_grad_impl,
};
use inner_opt::{
    INNER_OPT_DOT_FLOATS, INNER_OPT_Q_FLOATS, INNER_OPT_SCALAR_FLOATS, inner_opt_step_impl,
};
use ozaki_gram::{OZAKI_REDUCE_I32_WORDS, ozaki_gram_rhs_i32_impl};
use prescreen::{
    PRESCREEN_ERR_SCRATCH_FLOATS, PRESCREEN_SCAN_SCRATCH_FLOATS, PRESCREEN_TGT_SCRATCH_FLOATS,
    prescreen_population_impl,
};
use spirv_std::glam::{UVec3, Vec2, Vec4};
use spirv_std::spirv;
use tile_mix::tile_mix_fwd_adj_impl;

#[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn prescreen_population(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &PrescreenUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] target_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] seed_params: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] seed_losses: &mut [f32],
    #[spirv(workgroup)] scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    #[spirv(workgroup)] err_scratch: &mut [f32; PRESCREEN_ERR_SCRATCH_FLOATS],
    #[spirv(workgroup)] tgt_scratch: &mut [f32; PRESCREEN_TGT_SCRATCH_FLOATS],
) {
    prescreen_population_impl(
        wg_id,
        lid,
        0,
        uniforms,
        input_audio,
        target_audio,
        seed_params,
        seed_losses,
        scan_scratch,
        err_scratch,
        tgt_scratch,
    );
}

#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn prescreen_population(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(subgroup_local_invocation_id)] sg_lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &PrescreenUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] target_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] seed_params: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] seed_losses: &mut [f32],
    #[spirv(workgroup)] scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    #[spirv(workgroup)] err_scratch: &mut [f32; PRESCREEN_ERR_SCRATCH_FLOATS],
    #[spirv(workgroup)] tgt_scratch: &mut [f32; PRESCREEN_TGT_SCRATCH_FLOATS],
) {
    prescreen_population_impl(
        wg_id,
        lid,
        sg_lid,
        uniforms,
        input_audio,
        target_audio,
        seed_params,
        seed_losses,
        scan_scratch,
        err_scratch,
        tgt_scratch,
    );
}

/// Forward `CoreQuad` tile scan (`fused_core_scan_256`). Both `portable` and `subgroups` builds
/// share identical workgroup-barrier sub-tile scans inside `15,360 B` SRAM.
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn fused_core_scan_256(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(num_workgroups)] num_wg: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &FusedScanUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] core_inputs: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] quad_params: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tile_checkpoints: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] feature_outputs: &mut [Vec4],
    #[spirv(workgroup)] scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    #[spirv(workgroup)] lane_buf: &mut [f32; FORWARD_LANE_BUF_FLOATS],
    #[spirv(workgroup)] aux_scratch: &mut [f32; FORWARD_AUX_SCRATCH_FLOATS],
) {
    fused_core_scan_256_impl(
        UVec3::ZERO,
        wg_id,
        num_wg,
        lid,
        uniforms,
        core_inputs,
        quad_params,
        tile_checkpoints,
        feature_outputs,
        scan_scratch,
        lane_buf,
        aux_scratch,
    );
}

#[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn adjoint_scan_256(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(num_workgroups)] num_wg: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &AdjointScanUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] core_inputs: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] quad_params: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tile_checkpoints: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] d_loss_d_features: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] d_loss_d_inputs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] param_grads: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] tile_adj_boundaries: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] feature_outputs: &[Vec4],
    #[spirv(workgroup)] s_prev2: &mut [Vec4; ADJOINT_S_PREV2_VEC4S],
    #[spirv(workgroup)] u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    #[spirv(workgroup)] v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    #[spirv(workgroup)] scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    #[spirv(workgroup)] aux: &mut [f32; ADJOINT_AUX_FLOATS],
) {
    adjoint_scan_256_impl(
        UVec3::ZERO,
        wg_id,
        num_wg,
        lid,
        0,
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
    );
}

#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn adjoint_scan_256(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(num_workgroups)] num_wg: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(subgroup_local_invocation_id)] sg_lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &AdjointScanUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] core_inputs: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] quad_params: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tile_checkpoints: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] d_loss_d_features: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] d_loss_d_inputs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] param_grads: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] tile_adj_boundaries: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] feature_outputs: &[Vec4],
    #[spirv(workgroup)] s_prev2: &mut [Vec4; ADJOINT_S_PREV2_VEC4S],
    #[spirv(workgroup)] u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    #[spirv(workgroup)] v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    #[spirv(workgroup)] scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    #[spirv(workgroup)] aux: &mut [f32; ADJOINT_AUX_FLOATS],
) {
    adjoint_scan_256_impl(
        UVec3::ZERO,
        wg_id,
        num_wg,
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
    );
}

#[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn tile_mix_fwd_adj(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &TileMixUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    block_inputs_or_d_loss_d_inputs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] tile_descriptors: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] feature_outputs: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] tile_weight_grads: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] d_loss_d_features: &mut [Vec4],
    #[spirv(workgroup)] reduce_a: &mut [f32; TILE_MIX_REDUCE_A_FLOATS],
    #[spirv(workgroup)] reduce_b: &mut [f32; TILE_MIX_REDUCE_B_FLOATS],
) {
    tile_mix_fwd_adj_impl(
        wg_id,
        lid,
        0,
        uniforms,
        block_inputs_or_d_loss_d_inputs,
        tile_descriptors,
        feature_outputs,
        tile_weight_grads,
        d_loss_d_features,
        reduce_a,
        reduce_b,
    );
}

#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn tile_mix_fwd_adj(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(subgroup_local_invocation_id)] sg_lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &TileMixUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    block_inputs_or_d_loss_d_inputs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] tile_descriptors: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] feature_outputs: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] tile_weight_grads: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] d_loss_d_features: &mut [Vec4],
    #[spirv(workgroup)] reduce_a: &mut [f32; TILE_MIX_REDUCE_A_FLOATS],
    #[spirv(workgroup)] reduce_b: &mut [f32; TILE_MIX_REDUCE_B_FLOATS],
) {
    tile_mix_fwd_adj_impl(
        wg_id,
        lid,
        sg_lid,
        uniforms,
        block_inputs_or_d_loss_d_inputs,
        tile_descriptors,
        feature_outputs,
        tile_weight_grads,
        d_loss_d_features,
        reduce_a,
        reduce_b,
    );
}

#[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gabor_bezout_grad(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &GaborBezoutUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pred_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] target_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] bezout_taps: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] d_loss_d_pred: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] band_losses: &mut [Vec4],
    #[spirv(workgroup)] buf_a: &mut [Vec4; GABOR_BUF_A_VEC4S],
    #[spirv(workgroup)] buf_b: &mut [Vec4; GABOR_BUF_B_VEC4S],
    #[spirv(workgroup)] buf_c: &mut [Vec2; GABOR_BUF_C_VEC2S],
    #[spirv(workgroup)] reduce_buf: &mut [f32; GABOR_REDUCE_FLOATS],
) {
    gabor_bezout_grad_impl(
        wg_id,
        lid,
        0,
        uniforms,
        pred_audio,
        target_audio,
        bezout_taps,
        d_loss_d_pred,
        band_losses,
        buf_a,
        buf_b,
        buf_c,
        reduce_buf,
    );
}

#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn gabor_bezout_grad(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(subgroup_local_invocation_id)] sg_lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &GaborBezoutUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pred_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] target_audio: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] bezout_taps: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] d_loss_d_pred: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] band_losses: &mut [Vec4],
    #[spirv(workgroup)] buf_a: &mut [Vec4; GABOR_BUF_A_VEC4S],
    #[spirv(workgroup)] buf_b: &mut [Vec4; GABOR_BUF_B_VEC4S],
    #[spirv(workgroup)] buf_c: &mut [Vec2; GABOR_BUF_C_VEC2S],
    #[spirv(workgroup)] reduce_buf: &mut [f32; GABOR_REDUCE_FLOATS],
) {
    gabor_bezout_grad_impl(
        wg_id,
        lid,
        sg_lid,
        uniforms,
        pred_audio,
        target_audio,
        bezout_taps,
        d_loss_d_pred,
        band_losses,
        buf_a,
        buf_b,
        buf_c,
        reduce_buf,
    );
}

#[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn ozaki_gram_rhs_i32(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &OzakiGramUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] features: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] targets: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] col_scales: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] ozaki_buckets: &mut [i32],
    #[spirv(workgroup)] reduce_i32: &mut [i32; OZAKI_REDUCE_I32_WORDS],
) {
    ozaki_gram_rhs_i32_impl(
        wg_id,
        lid,
        0,
        uniforms,
        features,
        targets,
        col_scales,
        ozaki_buckets,
        reduce_i32,
    );
}

#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn ozaki_gram_rhs_i32(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(subgroup_local_invocation_id)] sg_lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &OzakiGramUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] features: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] targets: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] col_scales: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] ozaki_buckets: &mut [i32],
    #[spirv(workgroup)] reduce_i32: &mut [i32; OZAKI_REDUCE_I32_WORDS],
) {
    ozaki_gram_rhs_i32_impl(
        wg_id,
        lid,
        sg_lid,
        uniforms,
        features,
        targets,
        col_scales,
        ozaki_buckets,
        reduce_i32,
    );
}

#[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn inner_opt_step(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &InnerOptUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] theta_candidates: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] grad_candidates: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] param_bounds_and_ste_mask: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] adamw_m_v_or_lbfgs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] search_dir: &mut [f32],
    #[spirv(workgroup)] scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    #[spirv(workgroup)] dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
    #[spirv(workgroup)] lbfgs_scalars: &mut [f32; INNER_OPT_SCALAR_FLOATS],
) {
    inner_opt_step_impl(
        wg_id,
        lid,
        0,
        uniforms,
        theta_candidates,
        grad_candidates,
        param_bounds_and_ste_mask,
        adamw_m_v_or_lbfgs,
        search_dir,
        scratch_q,
        dot_scratch,
        lbfgs_scalars,
    );
}

#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
#[allow(clippy::too_many_arguments)]
#[spirv(compute(threads(256, 1, 1)))]
pub fn inner_opt_step(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_index)] lid: u32,
    #[spirv(subgroup_local_invocation_id)] sg_lid: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] uniforms: &InnerOptUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] theta_candidates: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] grad_candidates: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] param_bounds_and_ste_mask: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] adamw_m_v_or_lbfgs: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] search_dir: &mut [f32],
    #[spirv(workgroup)] scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    #[spirv(workgroup)] dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
    #[spirv(workgroup)] lbfgs_scalars: &mut [f32; INNER_OPT_SCALAR_FLOATS],
) {
    inner_opt_step_impl(
        wg_id,
        lid,
        sg_lid,
        uniforms,
        theta_candidates,
        grad_candidates,
        param_bounds_and_ste_mask,
        adamw_m_v_or_lbfgs,
        search_dir,
        scratch_q,
        dot_scratch,
        lbfgs_scalars,
    );
}
