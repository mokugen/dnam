#![forbid(unsafe_code)]
//! `SONA v4.0` CPU <-> `wgpu` (`Portable` + `Subgroups`) hardware parity suite covering:
//! - Full `28`-slot `CoreQuadV4ParamsStereo4` forward/adjoint parity (Block 0 Pred-Corr PHS,
//!   Blocks 1..3 3-tap folded SVF, `BLOCK_IDX_DIRECT_LINEAR`) (`verify[dsp.op-n.core.tpt-tone-basis]`)
//! - Multi-chunk `block_idx = 0` `24`-Vec4 `EXIT_ADJ_BOUNDARY_VEC4_STRIDE` (`adj_s_lin_out` carry)
//! - `SE-Bus` fast path vs `Slot 5` `b_all` staging (`lin_mix != 0`, tiny `|out_gain|`)
//! - `AlgADAA1(4096)` & `|r_unclamped| > 4096` straight-through mask across `255 <-> 256` tile seam
//!   (`verify[dsp.op-n.polyadaa.clamp]`, `verify[dsp.op-n.polyadaa.rail-crossing]`,
//!   `verify[dsp.op-n.wavefolder.cubic-c2]`)
//! - `[3, 3, 2, 2]` `37`-tile cascade (`verify[dsp.op-n.core.interblock-tiles]`),
//!   `FUSED_FLAG_FIXED_POINT_SCAN`, `Prescreen v4`, and central-FD (`verify[train.pipeline.isomorphism-parity]`,
//!   `verify[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{
    CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, MAX_ALG_R, SOFT_RECT_EPS,
    SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL, SonaParaLruDuhemCoeffs4, SonaToneFolded3,
    SonaToneFoldedBlock0, SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4,
    prewarp_tan_pi_f_over_fs_f32,
};
use dnam_train::{
    AdjointScanInput, CpuRunner, FusedCoreScanInput, PrescreenInput, WaveCoreAdjointInput,
    WaveCoreCascadeInput,
    abi::{
        AdjointScanUniforms, BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE,
        EXIT_ADJ_BOUNDARY_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FUSED_FLAG_FIXED_POINT_SCAN,
        FusedScanUniforms, PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms, QUAD_PARAMS_VEC4_STRIDE,
        TILE_CKPT_VEC4_STRIDE, TILE_FRAMES, quad_param_slots,
    },
    pack_core_quad_params,
};

use super::support::{acquire_variants, build_wave_descriptors};

fn assert_slice_close(expected: &[[f32; 4]], actual: &[[f32; 4]], rel_tol: f32, label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label}: len mismatch");
    for (idx, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        for c in 0..4 {
            let tol = rel_tol * e[c].abs().max(1.0);
            let diff = (e[c] - a[c]).abs();
            assert!(
                diff <= tol && a[c].is_finite(),
                "{label}[{idx}][{c}]: exp={} got={} diff={diff} tol={tol}",
                e[c],
                a[c]
            );
        }
    }
}

fn make_v4_quad(
    block_idx: usize,
    k_nl: [f32; 4],
    lin_mix: [f32; 4],
    out_gain: [f32; 4],
) -> [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE] {
    let g = GpuVec4::from_array([
        prewarp_tan_pi_f_over_fs_f32(0.05),
        prewarp_tan_pi_f_over_fs_f32(0.11),
        prewarp_tan_pi_f_over_fs_f32(0.18),
        prewarp_tan_pi_f_over_fs_f32(0.26),
    ]);
    let k = GpuVec4::from_array([0.45, 0.75, 1.05, 1.35]);
    let w = SonaToneWeights4 {
        lp: GpuVec4::from_array([0.45, 0.25, -0.15, 0.12]),
        bp: GpuVec4::from_array([0.22, 0.36, 0.24, -0.18]),
        hp: GpuVec4::from_array([-0.14, 0.18, 0.32, 0.28]),
        ap: GpuVec4::from_array([0.09, -0.07, 0.14, -0.06]),
        dir: GpuVec4::from_array([0.21, 0.12, -0.05, 0.19]),
    };
    let tone = if block_idx == 0 {
        SonaToneFoldedBlock0::from_weights(&w)
    } else {
        SonaToneFoldedBlock0::from_folded3(&SonaToneFolded3::from_weights(&w, k))
    };
    pack_core_quad_params(&CoreQuadV4ParamsStereo4 {
        svf: TptSvfCoeffs4::from_gk(g, k),
        tone,
        lru: SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::from_array([0.94, 0.88, 0.80, 0.72]),
            GpuVec4::from_array([0.28, 0.42, 0.58, 0.74]),
            GpuVec4::from_array([0.18, 0.35, 0.55, 0.78]),
            GpuVec4::from_array([0.14, 0.26, 0.42, 0.60]),
            GpuVec4::from_array([0.20, 0.38, 0.52, 0.68]),
            GpuVec4::from_array([0.08, 0.22, 0.38, 0.56]),
        ),
        two_svf_g: g.add(g),
        svf_k_nl: GpuVec4::from_array(k_nl),
        rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
        g0: GpuVec4::from_array([0.85, 1.10, 0.90, 1.25]),
        g_inf: GpuVec4::from_array([1.35, 0.82, 1.20, 0.92]),
        d_pre: GpuVec4::from_array([0.45, 0.75, 0.30, 0.55]),
        se_w_sq: GpuVec4::from_array([0.55, 0.30, 0.45, 0.25]),
        se_d_bus: GpuVec4::from_array([0.40, 0.25, 0.50, 0.35]),
        adaa_drive: GpuVec4::from_array([1.6, 2.2, 1.9, 1.4]),
        lru_bias_drive: GpuVec4::from_array([0.18, -0.24, 0.28, -0.15]),
        lin_mix: GpuVec4::from_array(lin_mix),
        wavefolder_mix: GpuVec4::from_array([0.15, 0.40, 0.65, 0.85]),
        glu_mix: GpuVec4::from_array([0.55, -0.45, 0.65, -0.35]),
        out_gain: GpuVec4::from_array(out_gain),
    })
}

fn fill_stereo(n: usize, seed: f32, amp: f32) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0_f32; 4]; n * CORE_INPUT_VEC4_STRIDE];
    for t in 0..n {
        let tf = t as f32;
        out[t * 2] = [
            amp * (0.07 * tf + seed).sin(),
            amp * (0.11 * tf - seed).cos(),
            amp * (0.05 * tf + 0.3).sin(),
            -amp * (0.13 * tf).cos(),
        ];
        out[t * 2 + 1] = [
            -amp * (0.09 * tf + 0.2).cos(),
            amp * (0.06 * tf + seed).sin(),
            amp * (0.12 * tf).cos(),
            amp * (0.08 * tf - 0.4).sin(),
        ];
    }
    out
}

#[test]
fn test_v4_multi_chunk_phs_exit_boundary_and_se_bus_staging_parity() {
    // verify[dsp.op-n.core.tpt-tone-basis]
    // verify[train.pipeline.isomorphism-parity]
    let variants = acquire_variants();
    if variants.is_empty() {
        return;
    }
    let cpu = CpuRunner::detect();
    let n = 2 * TILE_FRAMES;
    let core_in = fill_stereo(n, 0.15, 0.45);
    let dl_feat: Vec<[f32; 4]> = (0..n * FEATURE_VEC4_STRIDE)
        .map(|i| {
            let s = (i as f32) * 0.17;
            [
                0.2 * s.sin(),
                -0.15 * s.cos(),
                0.18 * (0.5 * s).cos(),
                -0.12 * s.sin(),
            ]
        })
        .collect();

    for &(k_nl, lin_mix, out_gain, tag) in &[
        ([0.0; 4], [0.0; 4], [1.0, 0.9, -0.8, 1.1], "b0_knl0_fast_se"),
        (
            [1.2, 0.4, 2.0, 0.8],
            [0.0; 4],
            [1.0, 0.9, -0.8, 1.1],
            "b0_knl_pos_fast_se",
        ),
        (
            [1.5, 0.7, 0.0, 2.2],
            [0.25, 0.0, 0.15, 0.4],
            [0.0, 1.0e-7, 0.9, -1.1],
            "b0_staged_b_all",
        ),
    ] {
        let qp = make_v4_quad(0, k_nl, lin_mix, out_gain);
        let fwd_cpu = cpu
            .fused_core_scan_256(&FusedCoreScanInput {
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
                core_inputs: &core_in,
                quad_params: &qp,
                initial_checkpoints: None,
            })
            .unwrap();
        let adj_cpu = cpu
            .adjoint_scan_256(&AdjointScanInput {
                uniforms: AdjointScanUniforms {
                    num_frames: n as u32,
                    num_tiles: 2,
                    num_quads: 1,
                    num_stems: 1,
                    pass_mode: 2,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: 0,
                },
                core_inputs: &core_in,
                quad_params: &qp,
                tile_checkpoints: &fwd_cpu.tile_checkpoints,
                d_loss_d_features: &dl_feat,
                feature_outputs: &fwd_cpu.feature_outputs,
                exit_adj_boundary: None,
            })
            .unwrap();

        for (variant, gpu) in &variants {
            let label = format!("{tag}/{variant:?}");
            let fwd_gpu = gpu
                .fused_core_scan_256(&FusedCoreScanInput {
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
                    core_inputs: &core_in,
                    quad_params: &qp,
                    initial_checkpoints: None,
                })
                .unwrap();
            assert_slice_close(
                &fwd_cpu.feature_outputs,
                &fwd_gpu.feature_outputs,
                2.0e-4,
                &format!("fwd {label}"),
            );
            assert_slice_close(
                &fwd_cpu.tile_checkpoints,
                &fwd_gpu.tile_checkpoints,
                2.0e-4,
                &format!("ckpt {label}"),
            );

            let fp_gpu = gpu
                .fused_core_scan_256(&FusedCoreScanInput {
                    uniforms: FusedScanUniforms {
                        num_frames: n as u32,
                        num_tiles: 2,
                        num_quads: 1,
                        num_stems: 1,
                        tile_start: 0,
                        soft_rect_eps: SOFT_RECT_EPS,
                        tiny_crossing_eps: TINY_CROSSING_EPS,
                        block_idx: FUSED_FLAG_FIXED_POINT_SCAN,
                    },
                    core_inputs: &core_in,
                    quad_params: &qp,
                    initial_checkpoints: None,
                })
                .unwrap();
            assert_slice_close(
                &fwd_cpu.feature_outputs,
                &fp_gpu.feature_outputs,
                5.0e-4,
                &format!("fp {label}"),
            );

            let adj_gpu = gpu
                .adjoint_scan_256(&AdjointScanInput {
                    uniforms: AdjointScanUniforms {
                        num_frames: n as u32,
                        num_tiles: 2,
                        num_quads: 1,
                        num_stems: 1,
                        pass_mode: 2,
                        soft_rect_eps: SOFT_RECT_EPS,
                        tiny_crossing_eps: TINY_CROSSING_EPS,
                        block_idx: 0,
                    },
                    core_inputs: &core_in,
                    quad_params: &qp,
                    tile_checkpoints: &fwd_gpu.tile_checkpoints,
                    d_loss_d_features: &dl_feat,
                    feature_outputs: &fwd_gpu.feature_outputs,
                    exit_adj_boundary: None,
                })
                .unwrap();
            assert_slice_close(
                &adj_cpu.d_loss_d_inputs,
                &adj_gpu.d_loss_d_inputs,
                5.0e-4,
                &format!("du {label}"),
            );
            assert_slice_close(
                &adj_cpu.param_grads,
                &adj_gpu.param_grads,
                1.0e-3,
                &format!("pg {label}"),
            );

            // Multi-chunk 24-Vec4 EXIT_ADJ parity: out_entry[12..24] -> in_exit[0..12]
            let c1 = gpu
                .adjoint_scan_256(&AdjointScanInput {
                    uniforms: AdjointScanUniforms {
                        num_frames: TILE_FRAMES as u32,
                        num_tiles: 1,
                        num_quads: 1,
                        num_stems: 1,
                        pass_mode: 2,
                        soft_rect_eps: SOFT_RECT_EPS,
                        tiny_crossing_eps: TINY_CROSSING_EPS,
                        block_idx: 0,
                    },
                    core_inputs: &core_in[TILE_FRAMES * 2..],
                    quad_params: &qp,
                    tile_checkpoints: &fwd_gpu.tile_checkpoints[TILE_CKPT_VEC4_STRIDE..],
                    d_loss_d_features: &dl_feat[TILE_FRAMES * FEATURE_VEC4_STRIDE..],
                    feature_outputs: &fwd_gpu.feature_outputs[TILE_FRAMES * FEATURE_VEC4_STRIDE..],
                    exit_adj_boundary: None,
                })
                .unwrap();
            let mut seed_c0 = vec![[0.0_f32; 4]; EXIT_ADJ_BOUNDARY_VEC4_STRIDE];
            seed_c0[0..12].copy_from_slice(&c1.entry_adj_boundary[12..24]);
            let c0 = gpu
                .adjoint_scan_256(&AdjointScanInput {
                    uniforms: AdjointScanUniforms {
                        num_frames: TILE_FRAMES as u32,
                        num_tiles: 1,
                        num_quads: 1,
                        num_stems: 1,
                        pass_mode: 2,
                        soft_rect_eps: SOFT_RECT_EPS,
                        tiny_crossing_eps: TINY_CROSSING_EPS,
                        block_idx: 0,
                    },
                    core_inputs: &core_in[..TILE_FRAMES * 2],
                    quad_params: &qp,
                    tile_checkpoints: &fwd_gpu.tile_checkpoints[..TILE_CKPT_VEC4_STRIDE],
                    d_loss_d_features: &dl_feat[..TILE_FRAMES * FEATURE_VEC4_STRIDE],
                    feature_outputs: &fwd_gpu.feature_outputs[..TILE_FRAMES * FEATURE_VEC4_STRIDE],
                    exit_adj_boundary: Some(&seed_c0),
                })
                .unwrap();
            assert_slice_close(
                &adj_gpu.d_loss_d_inputs[..TILE_FRAMES * 2],
                &c0.d_loss_d_inputs,
                5.0e-5,
                &format!("chunked24 du {label}"),
            );
            assert_slice_close(
                &adj_gpu.param_grads[..QUAD_PARAMS_VEC4_STRIDE],
                &c0.param_grads,
                1.0e-4,
                &format!("chunked24 pg0 {label}"),
            );
        }
    }
}

#[test]
fn test_v4_alg_adaa_clamp_seam_37_tile_cascade_and_prescreen_parity() {
    // verify[dsp.op-n.polyadaa.clamp]
    // verify[dsp.op-n.polyadaa.rail-crossing]
    // verify[dsp.op-n.wavefolder.cubic-c2]
    // verify[dsp.op-n.core.interblock-tiles]
    // verify[train.pipeline.webgpu-scans-and-ozaki]
    let variants = acquire_variants();
    if variants.is_empty() {
        return;
    }
    let cpu = CpuRunner::detect();

    // 1. Tile seam (frame 255 saturated |r| > 4096 -> frame 256 linear |r| <= 4096) straight-through mask
    let mut qp_seam = make_v4_quad(1, [0.0; 4], [0.0; 4], [1.0; 4]);
    qp_seam[quad_param_slots::TONE_C_LP] = [0.0; 4];
    qp_seam[quad_param_slots::TONE_C_BP] = [0.0; 4];
    qp_seam[quad_param_slots::TONE_C_KBP] = [0.0; 4];
    qp_seam[quad_param_slots::TONE_C_U] = [1.0; 4];
    qp_seam[quad_param_slots::G0] = [2.5; 4];
    qp_seam[quad_param_slots::G_INF] = [2.5; 4];
    qp_seam[quad_param_slots::SE_W_SQ] = [0.0; 4];
    qp_seam[quad_param_slots::SE_D_BUS] = [0.0; 4];
    qp_seam[quad_param_slots::ADAA_DRIVE] = [32.0; 4];
    qp_seam[quad_param_slots::LRU_BIAS_DRIVE] = [0.0; 4];
    qp_seam[quad_param_slots::GLU_MIX] = [0.0; 4];
    let n_seam = TILE_FRAMES + 4;
    let mut in_seam = vec![[0.0_f32; 4]; n_seam * 2];
    in_seam[255 * 2] = [100.0; 4]; // r_unclamped = 8000 > MAX_ALG_R (4096)
    in_seam[256 * 2] = [0.01; 4]; // linear seam consumer!
    let mut dl_seam = vec![[0.0_f32; 4]; n_seam * FEATURE_VEC4_STRIDE];
    dl_seam[256 * FEATURE_VEC4_STRIDE + 2] = [1.0; 4]; // loss strictly on o_tilde_L[256]
    let fwd_seam_c = cpu
        .fused_core_scan_256(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: n_seam as u32,
                num_tiles: 2,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: BLOCK_IDX_DIRECT_LINEAR,
            },
            core_inputs: &in_seam,
            quad_params: &qp_seam,
            initial_checkpoints: None,
        })
        .unwrap();
    let adj_seam_c = cpu
        .adjoint_scan_256(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: n_seam as u32,
                num_tiles: 2,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: BLOCK_IDX_DIRECT_LINEAR,
            },
            core_inputs: &in_seam,
            quad_params: &qp_seam,
            tile_checkpoints: &fwd_seam_c.tile_checkpoints,
            d_loss_d_features: &dl_seam,
            feature_outputs: &fwd_seam_c.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();
    assert_eq!(adj_seam_c.d_loss_d_inputs[255 * 2], [0.0; 4]);
    assert_eq!(
        adj_seam_c.param_grads[quad_param_slots::ADAA_DRIVE],
        [0.0; 4]
    );
    assert_eq!(adj_seam_c.param_grads[quad_param_slots::G0], [0.0; 4]);
    let _ = MAX_ALG_R;

    for (variant, gpu) in &variants {
        let fwd_g = gpu
            .fused_core_scan_256(&FusedCoreScanInput {
                uniforms: FusedScanUniforms {
                    num_frames: n_seam as u32,
                    num_tiles: 2,
                    num_quads: 1,
                    num_stems: 1,
                    tile_start: 0,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: BLOCK_IDX_DIRECT_LINEAR,
                },
                core_inputs: &in_seam,
                quad_params: &qp_seam,
                initial_checkpoints: None,
            })
            .unwrap();
        let adj_g = gpu
            .adjoint_scan_256(&AdjointScanInput {
                uniforms: AdjointScanUniforms {
                    num_frames: n_seam as u32,
                    num_tiles: 2,
                    num_quads: 1,
                    num_stems: 1,
                    pass_mode: 2,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: BLOCK_IDX_DIRECT_LINEAR,
                },
                core_inputs: &in_seam,
                quad_params: &qp_seam,
                tile_checkpoints: &fwd_g.tile_checkpoints,
                d_loss_d_features: &dl_seam,
                feature_outputs: &fwd_g.feature_outputs,
                exit_adj_boundary: None,
            })
            .unwrap();
        assert_eq!(
            adj_g.d_loss_d_inputs[255 * 2],
            [0.0; 4],
            "GPU seam clamp {variant:?}"
        );
        assert_eq!(
            adj_g.param_grads[quad_param_slots::ADAA_DRIVE],
            [0.0; 4],
            "GPU ADAA_DRIVE clamp {variant:?}"
        );
        assert_slice_close(
            &adj_seam_c.d_loss_d_inputs,
            &adj_g.d_loss_d_inputs,
            5.0e-4,
            &format!("seam du {variant:?}"),
        );
    }

    // 2. [3,3,2,2] 37-tile full cascade + central-FD verification on v4 slots
    let num_quads = 10_usize;
    let active_tiles = 37_usize;
    let n_cas = 300_usize;
    let mut quad_params = Vec::with_capacity(num_quads * QUAD_PARAMS_VEC4_STRIDE);
    for q in 0..num_quads {
        let blk = SONA_V4_CORE_QUAD_BLOCK_AND_LOCAL[q].0;
        let knl = if blk == 0 {
            [0.35, 0.15, 0.85, 0.5]
        } else {
            [0.0; 4]
        };
        quad_params.extend_from_slice(&make_v4_quad(
            blk,
            knl,
            [0.08, 0.15, 0.0, 0.12],
            [0.95, 1.05, -0.90, 0.85],
        ));
    }
    let tile_desc = build_wave_descriptors(active_tiles, 0.04);
    let mut base_in = vec![[0.0_f32; 4]; num_quads * n_cas * CORE_INPUT_VEC4_STRIDE];
    for q in 0..num_quads {
        let slice = fill_stereo(n_cas, 0.25 * (q as f32 + 1.0), 0.35);
        let q_off = q * n_cas * CORE_INPUT_VEC4_STRIDE;
        base_in[q_off..q_off + slice.len()].copy_from_slice(&slice);
    }
    let fwd_c = cpu
        .run_wave_core_cascade(&WaveCoreCascadeInput {
            num_frames: n_cas,
            num_quads,
            num_stems: 1,
            active_tiles,
            write_checkpoints: true,
            fixed_point_scan: false,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            base_core_inputs: &base_in,
            quad_params: &quad_params,
            tile_descriptors: &tile_desc,
        })
        .unwrap();
    let dl_cas = vec![[0.12_f32, -0.08, 0.15, -0.10]; fwd_c.feature_outputs.len()];
    let adj_c = cpu
        .run_wave_core_adjoint(&WaveCoreAdjointInput {
            num_frames: n_cas,
            num_quads,
            num_stems: 1,
            active_tiles,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            core_inputs: &fwd_c.core_inputs,
            quad_params: &quad_params,
            tile_checkpoints: &fwd_c.tile_checkpoints,
            d_loss_d_features: &dl_cas,
            feature_outputs: &fwd_c.feature_outputs,
            tile_descriptors: &tile_desc,
        })
        .unwrap();

    for (variant, gpu) in &variants {
        let fwd_g = gpu
            .run_wave_core_cascade(&WaveCoreCascadeInput {
                num_frames: n_cas,
                num_quads,
                num_stems: 1,
                active_tiles,
                write_checkpoints: true,
                fixed_point_scan: false,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                base_core_inputs: &base_in,
                quad_params: &quad_params,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        let adj_g = gpu
            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                num_frames: n_cas,
                num_quads,
                num_stems: 1,
                active_tiles,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                core_inputs: &fwd_g.core_inputs,
                quad_params: &quad_params,
                tile_checkpoints: &fwd_g.tile_checkpoints,
                d_loss_d_features: &dl_cas,
                feature_outputs: &fwd_g.feature_outputs,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        assert_slice_close(
            &fwd_c.feature_outputs,
            &fwd_g.feature_outputs,
            5.0e-4,
            &format!("cas fwd {variant:?}"),
        );
        assert_slice_close(
            &adj_c.d_loss_d_inputs,
            &adj_g.d_loss_d_inputs,
            1.0e-3,
            &format!("cas du {variant:?}"),
        );
        assert_slice_close(
            &adj_c.param_grads,
            &adj_g.param_grads,
            1.0e-3,
            &format!("cas pg {variant:?}"),
        );
    }

    // 3. Central FD check on quad 8 (Block 0) & quad 7 (Block 3) v4 slots
    let num_tiles = n_cas.div_ceil(TILE_FRAMES);
    let eps = 2.0e-3_f32;
    for &(q_test, slot) in &[
        (8_usize, quad_param_slots::TONE_C_KBP),
        (8, quad_param_slots::G0),
        (8, quad_param_slots::SE_W_SQ),
        (7, quad_param_slots::TONE_C_LP),
        (7, quad_param_slots::LIN_MIX),
        (7, quad_param_slots::OUT_GAIN),
    ] {
        let eval_cas = |delta: f32| -> f64 {
            let mut qp = quad_params.clone();
            qp[q_test * QUAD_PARAMS_VEC4_STRIDE + slot][1] += delta;
            let o = cpu
                .run_wave_core_cascade(&WaveCoreCascadeInput {
                    num_frames: n_cas,
                    num_quads,
                    num_stems: 1,
                    active_tiles,
                    write_checkpoints: false,
                    fixed_point_scan: false,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    base_core_inputs: &base_in,
                    quad_params: &qp,
                    tile_descriptors: &tile_desc,
                })
                .unwrap();
            o.feature_outputs
                .iter()
                .zip(dl_cas.iter())
                .map(|(a, b)| {
                    (0..4)
                        .map(|j| f64::from(a[j]) * f64::from(b[j]))
                        .sum::<f64>()
                })
                .sum()
        };
        let fd = (eval_cas(eps) - eval_cas(-eps)) / f64::from(2.0 * eps);
        let got: f64 = (0..num_tiles)
            .map(|t| {
                f64::from(
                    adj_c.param_grads[(q_test * num_tiles + t) * QUAD_PARAMS_VEC4_STRIDE + slot][1],
                )
            })
            .sum();
        assert!(
            (got - fd).abs() <= 5.0e-3 * fd.abs().max(1.0),
            "Cascade FD q={q_test} slot={slot}: got={got} fd={fd}"
        );
    }

    // 4. Prescreen v4 (Slot 13 = sag_depth, ParaLRU Duhem v4, AlgADAA1) CPU <-> GPU parity
    let mut seed_params = vec![0.0_f32; 2 * PRESCREEN_SEED_PARAM_STRIDE];
    for s in 0..2 {
        let b = s * PRESCREEN_SEED_PARAM_STRIDE;
        seed_params[b] = 0.08 + 0.04 * s as f32;
        seed_params[b + 1] = 0.75;
        seed_params[b + 2] = 0.45;
        seed_params[b + 3] = 0.25;
        seed_params[b + 4] = -0.10;
        seed_params[b + 5] = 0.20;
        seed_params[b + 6] = 0.92;
        seed_params[b + 7] = 0.35;
        seed_params[b + 8] = 0.30;
        seed_params[b + 9] = 0.25;
        seed_params[b + 10] = 0.40;
        seed_params[b + 11] = 0.15;
        seed_params[b + 12] = if s == 0 { 1.0 } else { -1.0 };
        seed_params[b + 13] = if s == 0 { 0.45 } else { -0.65 }; // sag_depth
        seed_params[b + 14] = 2.4;
        seed_params[b + 15] = 0.35;
    }
    let pre_in: Vec<[f32; 2]> = (0..64)
        .map(|i| {
            [
                0.5 * (0.19 * i as f32).sin(),
                -0.4 * (0.13 * i as f32).cos(),
            ]
        })
        .collect();
    let pre_tgt: Vec<[f32; 2]> = (0..64)
        .map(|i| {
            [
                0.2 * (0.19 * i as f32).sin(),
                -0.15 * (0.13 * i as f32).cos(),
            ]
        })
        .collect();
    let pre_cpu = cpu
        .prescreen_population(&PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 2,
                num_frames: 64,
                num_quads: 1,
                num_stems: 1,
                sample_rate_hz: 48_000.0,
                eps_floor: 1.0e-5,
                _pad0: 0,
                _pad1: 0,
            },
            input_audio: &pre_in,
            target_audio: &pre_tgt,
            seed_params: &seed_params,
        })
        .unwrap();
    for (variant, gpu) in &variants {
        let pre_gpu = gpu
            .prescreen_population(&PrescreenInput {
                uniforms: PrescreenUniforms {
                    num_seeds: 2,
                    num_frames: 64,
                    num_quads: 1,
                    num_stems: 1,
                    sample_rate_hz: 48_000.0,
                    eps_floor: 1.0e-5,
                    _pad0: 0,
                    _pad1: 0,
                },
                input_audio: &pre_in,
                target_audio: &pre_tgt,
                seed_params: &seed_params,
            })
            .unwrap();
        for (i, (&c, &g)) in pre_cpu.iter().zip(pre_gpu.iter()).enumerate() {
            assert!(
                (c - g).abs() <= 1.0e-5,
                "Prescreen v4 [{i}] ({variant:?}): cpu={c} gpu={g}"
            );
        }
    }
}
