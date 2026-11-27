//! Deterministic CPU SIMD reference runners for `SONA v4.0` `fused_core_scan_256` (including
//! `Streamed` `12 × Vec4` checkpoint chaining) and 3-pass-equivalent reverse BPTT
//! `adjoint_scan_256` (`ref[train.pipeline.webgpu-scans-and-ozaki]`,
//! `ref[train.pipeline.isomorphism-parity]`).

mod adjoint;
mod adjoint_math;
mod forward;
mod streamed;

pub use adjoint::run_adjoint_scan_cpu;
pub use forward::run_fused_core_scan_cpu;
pub use streamed::run_fused_core_scan_planned_cpu;

use dnam_kernels::{
    CoreQuadV4Block0StateStereo8, CoreQuadV4ParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4,
    MIN_TPT_G, SonaBlockStateStereo8, StereoVec8, canonicalize_zero_f32_bits,
};

use adjoint_math::{QuadLaneParams, TileGradAccum};

use crate::error::TrainError;
use crate::gpu::abi::{QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, quad_param_slots};
use crate::runner::types::unpack_core_quad_params;

#[inline(always)]
pub(crate) fn canon_vec4(v: GpuVec4) -> [f32; 4] {
    v.to_array().map(canonicalize_zero_f32_bits)
}

pub(crate) fn read_checkpoint(
    ckpts: &[[f32; 4]],
    q_global: usize,
    num_tiles: usize,
    tile: usize,
) -> CoreQuadV4Block0StateStereo8<InterleavedStereo8<GpuVec4>> {
    let base = (q_global * num_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    CoreQuadV4Block0StateStereo8 {
        block: SonaBlockStateStereo8 {
            svf_s1: InterleavedStereo8::from_lr(
                GpuVec4::from_array(ckpts[base]),
                GpuVec4::from_array(ckpts[base + 1]),
            ),
            svf_s2: InterleavedStereo8::from_lr(
                GpuVec4::from_array(ckpts[base + 2]),
                GpuVec4::from_array(ckpts[base + 3]),
            ),
            lru_v: InterleavedStereo8::from_lr(
                GpuVec4::from_array(ckpts[base + 4]),
                GpuVec4::from_array(ckpts[base + 5]),
            ),
            adaa_prev: InterleavedStereo8::from_lr(
                GpuVec4::from_array(ckpts[base + 6]),
                GpuVec4::from_array(ckpts[base + 7]),
            ),
        },
        svf_lin_s1: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 8]),
            GpuVec4::from_array(ckpts[base + 9]),
        ),
        svf_lin_s2: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 10]),
            GpuVec4::from_array(ckpts[base + 11]),
        ),
    }
}

pub(crate) fn write_checkpoint(
    ckpts: &mut [[f32; 4]],
    q_global: usize,
    num_tiles: usize,
    tile: usize,
    state: &CoreQuadV4Block0StateStereo8<InterleavedStereo8<GpuVec4>>,
) {
    let base = (q_global * num_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let (s1_l, s1_r) = state.block.svf_s1.into_lr();
    let (s2_l, s2_r) = state.block.svf_s2.into_lr();
    let (v_l, v_r) = state.block.lru_v.into_lr();
    let (ap_l, ap_r) = state.block.adaa_prev.into_lr();
    let (sl1_l, sl1_r) = state.svf_lin_s1.into_lr();
    let (sl2_l, sl2_r) = state.svf_lin_s2.into_lr();
    ckpts[base] = canon_vec4(s1_l);
    ckpts[base + 1] = canon_vec4(s1_r);
    ckpts[base + 2] = canon_vec4(s2_l);
    ckpts[base + 3] = canon_vec4(s2_r);
    ckpts[base + 4] = canon_vec4(v_l);
    ckpts[base + 5] = canon_vec4(v_r);
    ckpts[base + 6] = canon_vec4(ap_l);
    ckpts[base + 7] = canon_vec4(ap_r);
    ckpts[base + 8] = canon_vec4(sl1_l);
    ckpts[base + 9] = canon_vec4(sl1_r);
    ckpts[base + 10] = canon_vec4(sl2_l);
    ckpts[base + 11] = canon_vec4(sl2_r);
}

pub(super) fn unpack_all_quads(
    quad_params: &[[f32; 4]],
    total_q: usize,
) -> Result<Vec<CoreQuadV4ParamsStereo4<GpuVec4>>, TrainError> {
    let needed = total_q * QUAD_PARAMS_VEC4_STRIDE;
    if quad_params.len() < needed {
        return Err(TrainError::InvalidDimensions {
            context: "quad_params",
            expected: needed,
            actual: quad_params.len(),
        });
    }
    let mut out = Vec::with_capacity(total_q);
    for q in 0..total_q {
        out.push(unpack_core_quad_params(
            &quad_params[q * QUAD_PARAMS_VEC4_STRIDE..(q + 1) * QUAD_PARAMS_VEC4_STRIDE],
        )?);
    }
    Ok(out)
}

pub(super) fn write_lane_param_grads(
    out_grads: &mut [[f32; 4]],
    pg_base: usize,
    c: usize,
    lp: &QuadLaneParams,
    acc: &TileGradAccum,
    is_phs: bool,
) {
    for slot in 0..QUAD_PARAMS_VEC4_STRIDE {
        out_grads[pg_base + slot][c] = acc.slots[slot];
    }
    let (g_svf_k, g_svf_k_nl, g_svf_g) = if is_phs {
        (
            if lp.k >= 0.0 {
                acc.slots[quad_param_slots::SVF_K]
            } else {
                0.0
            },
            if lp.svf_k_nl >= 0.0 {
                acc.slots[quad_param_slots::SVF_K_NL]
            } else {
                0.0
            },
            (if lp.svf_g >= MIN_TPT_G {
                acc.slots[quad_param_slots::SVF_G]
            } else {
                0.0
            }) + 2.0 * acc.duhem_two_svf_g,
        )
    } else {
        let r0 = [acc.slots[0], acc.slots[1], acc.slots[2]];
        let g_k = acc.svf_k_tone - lp.a2 * (r0[0] * lp.a1 + r0[1] * lp.a2 + r0[2] * lp.a3);
        let g_g_tpt = if lp.svf_g >= MIN_TPT_G {
            let denom_g_term = 2.0 * lp.a2 + lp.k * lp.a1;
            let da1_dg = -lp.a1 * denom_g_term;
            let da2_dg = lp.a1 - lp.a2 * denom_g_term;
            let da3_dg = (lp.a2 + lp.a2) - lp.a3 * denom_g_term;
            r0[0] * da1_dg + r0[1] * da2_dg + r0[2] * da3_dg
        } else {
            0.0
        };
        (g_k, 0.0, g_g_tpt + 2.0 * acc.duhem_two_svf_g)
    };
    out_grads[pg_base + quad_param_slots::SVF_K][c] = g_svf_k;
    out_grads[pg_base + quad_param_slots::SVF_G][c] = g_svf_g;
    out_grads[pg_base + quad_param_slots::SVF_K_NL][c] = g_svf_k_nl;
}
