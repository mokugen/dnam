//! Deterministic CPU SIMD reference runner for `SONA v4.0` `fused_core_scan_256` with
//! `12 × Vec4` checkpointing (`s_lin` included on Block `0`), `tile_start > 0` resume support,
//! and exact `+0.0` Zero-DC canonicalization (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{
    CoreQuadV4Block0StateStereo8, GpuVec4, InterleavedStereo8, KernelVec4, StereoVec8,
    step_core_quad_v4_phs_stereo8, step_core_quad_v4_stereo8,
};

use super::adjoint_math::flush_zero;
use super::{canon_vec4, read_checkpoint, unpack_all_quads, write_checkpoint};
use crate::error::TrainError;
use crate::gpu::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE,
    FUSED_FLAG_FIXED_POINT_SCAN, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES, block_quad_count,
    resolve_quad_idx,
};
use crate::runner::types::{FusedCoreScanInput, FusedCoreScanOutput};

/// Deterministic CPU reference implementation of `SONA v4.0` `fused_core_scan_256`.
pub fn run_fused_core_scan_cpu(
    input: &FusedCoreScanInput<'_>,
) -> Result<FusedCoreScanOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_tiles = (u.num_tiles as usize).max(1);
    let num_quads = u.num_quads as usize;
    let num_stems = u.num_stems as usize;
    let total_q = num_quads * num_stems;
    if num_frames == 0 || num_quads == 0 || num_stems == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256",
            expected: 1,
            actual: 0,
        });
    }
    let needed_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
    if input.core_inputs.len() < needed_in {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256(core_inputs)",
            expected: needed_in,
            actual: input.core_inputs.len(),
        });
    }
    let params = unpack_all_quads(input.quad_params, total_q)?;
    let mut tile_checkpoints = vec![[0.0_f32; 4]; total_q * num_tiles * TILE_CKPT_VEC4_STRIDE];
    if let Some(seed) = input.initial_checkpoints {
        let n = seed.len().min(tile_checkpoints.len());
        tile_checkpoints[..n].copy_from_slice(&seed[..n]);
    }
    let mut feature_outputs = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];

    let max_tiles = num_frames.div_ceil(TILE_FRAMES).min(num_tiles);
    let tile_start = (u.tile_start as usize).min(max_tiles);
    let resuming = tile_start > 0 && tile_start < num_tiles;
    let is_direct = u.block_idx == BLOCK_IDX_DIRECT_LINEAR;
    let raw_block = if is_direct {
        BLOCK_IDX_DIRECT_LINEAR
    } else {
        u.block_idx & !FUSED_FLAG_FIXED_POINT_SCAN
    };
    let is_phs = !is_direct && (raw_block & 3) == 0;
    let wg_x = if is_direct {
        num_quads
    } else {
        block_quad_count((raw_block & 3) as usize, num_quads)
    };

    for stem in 0..num_stems {
        for q_local in 0..wg_x {
            let quad_idx = resolve_quad_idx(raw_block, q_local, num_quads);
            if quad_idx >= num_quads {
                continue;
            }
            let q_global = stem * num_quads + quad_idx;
            let mut state = if resuming {
                read_checkpoint(&tile_checkpoints, q_global, num_tiles, tile_start)
            } else {
                CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero()
            };
            for tile in tile_start..max_tiles {
                if !(resuming && tile == tile_start) {
                    write_checkpoint(&mut tile_checkpoints, q_global, num_tiles, tile, &state);
                }
                let base_frame = tile * TILE_FRAMES;
                let end_frame = (base_frame + TILE_FRAMES).min(num_frames);
                for t in base_frame..end_frame {
                    let in_idx = (q_global * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                    let u_vec = InterleavedStereo8::from_lr(
                        GpuVec4::from_array(input.core_inputs[in_idx].map(flush_zero)),
                        GpuVec4::from_array(input.core_inputs[in_idx + 1].map(flush_zero)),
                    );
                    let out = if is_phs {
                        step_core_quad_v4_phs_stereo8(u_vec, &mut state, &params[q_global])
                    } else {
                        step_core_quad_v4_stereo8(u_vec, &mut state.block, &params[q_global])
                    };
                    let (vl, vr) = out.lru_state.into_lr();
                    let (ol, or) = out.o_tilde.into_lr();
                    let (el, er) = out.e_tilde.into_lr();
                    let f_idx = (q_global * num_frames + t) * FEATURE_VEC4_STRIDE;
                    feature_outputs[f_idx] = canon_vec4(vl);
                    feature_outputs[f_idx + 1] = canon_vec4(vr);
                    feature_outputs[f_idx + 2] = canon_vec4(ol);
                    feature_outputs[f_idx + 3] = canon_vec4(or);
                    feature_outputs[f_idx + 4] = canon_vec4(el);
                    feature_outputs[f_idx + 5] = canon_vec4(er);
                }
            }
        }
    }
    Ok(FusedCoreScanOutput {
        tile_checkpoints,
        feature_outputs,
    })
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        CoreQuadV4ParamsStereo4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
        SonaToneFoldedBlock0, SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4,
        prewarp_tan_pi_f_over_fs_f32,
    };

    use super::*;
    use crate::gpu::abi::{FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE, quad_param_slots};
    use crate::runner::types::pack_core_quad_params;

    #[test]
    fn test_v4_forward_zero_dc_direct_linear_and_fixed_point_flag_parity() {
        let g = GpuVec4::splat(prewarp_tan_pi_f_over_fs_f32(0.1));
        let k = GpuVec4::splat(0.75);
        let params = CoreQuadV4ParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneFoldedBlock0::from_weights(&SonaToneWeights4 {
                lp: GpuVec4::splat(0.4),
                bp: GpuVec4::splat(0.25),
                hp: GpuVec4::splat(-0.1),
                ap: GpuVec4::splat(0.05),
                dir: GpuVec4::splat(0.3),
            }),
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.9),
                GpuVec4::splat(0.35),
                GpuVec4::splat(0.2),
                GpuVec4::splat(0.15),
                GpuVec4::splat(0.45),
                GpuVec4::splat(0.2),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::splat(0.8),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            g0: GpuVec4::splat(0.9),
            g_inf: GpuVec4::splat(1.2),
            d_pre: GpuVec4::splat(0.3),
            se_w_sq: GpuVec4::splat(0.4),
            se_d_bus: GpuVec4::splat(0.25),
            adaa_drive: GpuVec4::splat(1.5),
            lru_bias_drive: GpuVec4::splat(0.2),
            lin_mix: GpuVec4::splat(0.1),
            wavefolder_mix: GpuVec4::splat(0.25),
            glu_mix: GpuVec4::splat(0.35),
            out_gain: GpuVec4::splat(1.0),
        };
        let packed = pack_core_quad_params(&params);
        let n = TILE_FRAMES + 16;
        let zero_in = vec![[-0.0_f32; 4]; n * CORE_INPUT_VEC4_STRIDE];
        let zero_out = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: n as u32,
                num_tiles: 2,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &zero_in,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        assert!(
            zero_out
                .feature_outputs
                .iter()
                .all(|v| v.iter().all(|x| x.to_bits() == 0))
        );

        let mut core_in = vec![[0.0_f32; 4]; n * CORE_INPUT_VEC4_STRIDE];
        for t in 0..n {
            core_in[t * 2] = [(t as f32 * 0.08).sin() * 0.3; 4];
            core_in[t * 2 + 1] = [(t as f32 * 0.11).cos() * 0.3; 4];
        }
        let u_lin = FusedScanUniforms {
            num_frames: n as u32,
            num_tiles: 2,
            num_quads: 1,
            num_stems: 1,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx: BLOCK_IDX_DIRECT_LINEAR,
        };
        let dir = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u_lin,
            core_inputs: &core_in,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        assert_eq!(dir.tile_checkpoints[8..12], [[0.0_f32; 4]; 4]);

        let fp = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                block_idx: FUSED_FLAG_FIXED_POINT_SCAN,
                ..u_lin
            },
            core_inputs: &core_in,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        let nom = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                block_idx: 0,
                ..u_lin
            },
            core_inputs: &core_in,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        assert_eq!(fp.feature_outputs, nom.feature_outputs);
        let _ = (QUAD_PARAMS_VEC4_STRIDE, quad_param_slots::SVF_G);
    }
}
