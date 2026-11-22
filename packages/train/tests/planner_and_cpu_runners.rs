#![forbid(unsafe_code)]
//! Integration suite for the 2-D memory planner (`MemoryBudgetProfile`), embedded WGSL manifest
//! SHA-256 verification, browser `enable subgroups;` transform, and deterministic [`CpuRunner`]
//! across all 6 compute kernels (`verify[train.pipeline.webgpu-scans-and-ozaki]`,
//! `verify[train.pipeline.isomorphism-parity]`).

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
    SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
};
use dnam_train::{
    ActiveComputeBackend, AdjointScanInput, BROWSER_WEBGPU_BUDGET_BYTES, CPU_WASM_BUDGET_BYTES,
    CoreScanWorkload, CpuRunner, DESKTOP_CAPSULE_BUDGET_BYTES, ExecutionMode, FusedCoreScanInput,
    GaborBezoutInput, InnerOptInput, KernelId, MAX_WEBGPU_BINDING_BYTES, MemoryBudgetProfile,
    OzakiGramInput, PrescreenInput, STREAMED_CKPT_EXTRA_TILES, STREAMING_HOST_DEFAULT_BUDGET_BYTES,
    STREAMING_HOST_MAX_TENSOR_BYTES, ShaderVariant, TrainError, TrainRunner,
    abi::{
        AdjointScanUniforms, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
        GaborBezoutUniforms, InnerOptUniforms, MAX_WORKGROUP_SRAM_BYTES, OZAKI_WORDS_PER_PAIR,
        OzakiGramUniforms, PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms, TILE_FRAMES,
        quad_param_slots,
    },
    embedded_wgsl, pack_core_quad_params, plan_core_scan_execution, prepare_for_browser_webgpu,
    uplo_u_packed_len, verify_embedded_shader_digests,
};
use proptest::prelude::*;

fn sample_quad_params(stem: usize, quad: usize) -> [[f32; 4]; 24] {
    let f = (stem * 2 + quad) as f32;
    let fc = [0.03 + 0.01 * f, 0.08 + 0.01 * f, 0.17 + 0.01 * f, 0.29];
    let g = GpuVec4::from_array(fc.map(prewarp_tan_pi_f_over_fs_f32));
    let k = GpuVec4::from_array([0.6 + 0.05 * f, 0.85, 1.15, 1.45]);
    let p = CoreQuadParamsStereo4 {
        svf: TptSvfCoeffs4::from_gk(g, k),
        tone: SonaToneWeights4 {
            lp: GpuVec4::from_array([0.42, 0.22, -0.14, 0.12]),
            bp: GpuVec4::from_array([0.24, 0.38, 0.28, -0.18]),
            hp: GpuVec4::from_array([-0.14, 0.16, 0.32, 0.42]),
            ap: GpuVec4::from_array([0.11, -0.09, 0.14, 0.08]),
            dir: GpuVec4::from_array([0.19, 0.14, -0.08, 0.24]),
        },
        lru: SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::from_array([0.91, 0.85, 0.77, 0.67]),
            GpuVec4::from_array([0.26, 0.41, 0.56, 0.72]),
            GpuVec4::from_array([0.16, 0.33, 0.58, 0.80]),
            GpuVec4::from_array([0.12, 0.27, 0.44, 0.61]),
            GpuVec4::from_array([0.21, 0.39, 0.54, 0.69]),
            GpuVec4::from_array([0.06, 0.24, 0.42, 0.60]),
        ),
        two_svf_g: g.add(g),
        svf_k_nl: GpuVec4::zero(),
        rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
        sag_depth: GpuVec4::from_array([0.17, 0.31, 0.42, 0.20]),
        adaa_drive: GpuVec4::from_array([1.15, 1.60, 2.05, 1.32]),
        lru_bias_drive: GpuVec4::from_array([0.14, -0.21, 0.28, -0.16]),
        wavefolder_mix: GpuVec4::from_array([0.15, 0.35, 0.65, 0.90]),
        glu_mix: GpuVec4::from_array([0.60, -0.42, 0.74, -0.48]),
    };
    pack_core_quad_params(&p)
}

#[test]
fn test_planner_profiles_wgsl_manifest_and_browser_helper() {
    // verify[train.pipeline.webgpu-scans-and-ozaki]
    verify_embedded_shader_digests().unwrap();
    for id in KernelId::ALL {
        assert!(id.expected_sram_bytes() <= MAX_WORKGROUP_SRAM_BYTES);
        let p_wgsl = embedded_wgsl(id, ShaderVariant::Portable);
        let s_wgsl = embedded_wgsl(id, ShaderVariant::Subgroups);
        assert!(p_wgsl.contains(id.entry_point()));
        assert!(s_wgsl.contains(id.entry_point()));
        let browser = prepare_for_browser_webgpu(s_wgsl, ShaderVariant::Subgroups);
        if s_wgsl.contains("subgroupAdd") {
            assert!(browser.starts_with("enable subgroups;\n"));
        }
    }

    let large = CoreScanWorkload {
        num_frames: 192_000,
        num_quads: 8,
        num_stems: 16,
        include_adjoint: true,
    };
    let b_plan = plan_core_scan_execution(large, MemoryBudgetProfile::BrowserWebGpu, None).unwrap();
    assert_eq!(b_plan.mode, ExecutionMode::Streamed);
    assert!(b_plan.footprints.max_single_binding_bytes <= MAX_WEBGPU_BINDING_BYTES);
    assert!(b_plan.footprints.peak_total_bytes <= BROWSER_WEBGPU_BUDGET_BYTES);
    assert_eq!(
        b_plan.ckpt_slot_tiles,
        b_plan.time_block_tiles + STREAMED_CKPT_EXTRA_TILES
    );

    let w_plan =
        plan_core_scan_execution(large, MemoryBudgetProfile::CpuWasmFallback, None).unwrap();
    assert!(w_plan.footprints.peak_total_bytes <= CPU_WASM_BUDGET_BYTES);

    let d_plan =
        plan_core_scan_execution(large, MemoryBudgetProfile::DesktopCapsule, None).unwrap();
    assert!(d_plan.footprints.peak_total_bytes <= DESKTOP_CAPSULE_BUDGET_BYTES);

    let h_plan = plan_core_scan_execution(
        large,
        MemoryBudgetProfile::StreamingHost {
            total_budget_bytes: STREAMING_HOST_DEFAULT_BUDGET_BYTES,
            max_tensor_bytes: STREAMING_HOST_MAX_TENSOR_BYTES,
        },
        None,
    )
    .unwrap();
    assert!(h_plan.footprints.peak_total_bytes <= STREAMING_HOST_DEFAULT_BUDGET_BYTES);

    assert!(matches!(
        plan_core_scan_execution(large, MemoryBudgetProfile::BrowserWebGpu, Some(32)),
        Err(TrainError::BindingExceedsCap { .. })
    ));
}

#[test]
fn test_cpu_runner_all_six_kernels_end_to_end() {
    // verify[train.pipeline.isomorphism-parity]
    // verify[train.pipeline.webgpu-scans-and-ozaki]
    let runner = TrainRunner::cpu();
    assert!(matches!(
        runner.active_backend(),
        ActiveComputeBackend::CpuSimd { .. }
    ));

    let num_frames = 540_usize;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let num_quads = 2_usize;
    let num_stems = 2_usize;
    let total_q = num_quads * num_stems;
    let mut quad_params = Vec::with_capacity(total_q * 24);
    for s in 0..num_stems {
        for q in 0..num_quads {
            quad_params.extend_from_slice(&sample_quad_params(s, q));
        }
    }
    let mut core_inputs = vec![[0.0_f32; 4]; total_q * num_frames * CORE_INPUT_VEC4_STRIDE];
    for t in 0..(total_q * num_frames) {
        let ph = (t as f32) * 0.041;
        core_inputs[t * 2] = [
            0.4 * ph.sin(),
            0.3 * (1.2 * ph).cos(),
            0.25,
            -0.35 * ph.sin(),
        ];
        core_inputs[t * 2 + 1] = [0.35 * ph.cos(), -0.25 * ph.sin(), 0.4, 0.2];
    }

    let u = FusedScanUniforms {
        num_frames: num_frames as u32,
        num_tiles: num_tiles as u32,
        num_quads: num_quads as u32,
        num_stems: num_stems as u32,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx: 0,
    };
    let res = runner
        .fused_core_scan_256(&FusedCoreScanInput {
            uniforms: u,
            core_inputs: &core_inputs,
            quad_params: &quad_params,
            initial_checkpoints: None,
        })
        .unwrap();

    let mut forced_stream = plan_core_scan_execution(
        CoreScanWorkload {
            num_frames,
            num_quads,
            num_stems,
            include_adjoint: false,
        },
        MemoryBudgetProfile::DesktopCapsule,
        None,
    )
    .unwrap();
    forced_stream.mode = ExecutionMode::Streamed;
    forced_stream.stem_group = 1;
    forced_stream.time_block_frames = TILE_FRAMES;
    forced_stream.time_block_tiles = 1;
    let streamed = runner
        .run_fused_core_scan_planned(
            &FusedCoreScanInput {
                uniforms: u,
                core_inputs: &core_inputs,
                quad_params: &quad_params,
                initial_checkpoints: None,
            },
            &forced_stream,
        )
        .unwrap();
    for (a, b) in res.feature_outputs.iter().zip(&streamed.feature_outputs) {
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }

    let d_loss = vec![[0.25_f32, -0.15, 0.3, -0.2]; total_q * num_frames * FEATURE_VEC4_STRIDE];
    let adj = runner
        .adjoint_scan_256(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: num_quads as u32,
                num_stems: num_stems as u32,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_inputs,
            quad_params: &quad_params,
            tile_checkpoints: &res.tile_checkpoints,
            d_loss_d_features: &d_loss,
            feature_outputs: &res.feature_outputs,
        })
        .unwrap();
    assert_eq!(adj.d_loss_d_inputs.len(), core_inputs.len());
    assert!(
        adj.param_grads[quad_param_slots::SVF_K]
            .iter()
            .any(|x| *x != 0.0)
    );

    // Prescreen, Gabor Mode A & B, Ozaki, and InnerOpt
    let audio_len = 256_usize;
    let in_a: Vec<[f32; 2]> = (0..audio_len)
        .map(|t| [((t as f32) * 0.09).sin() * 0.5, 0.25])
        .collect();
    let tg_a: Vec<[f32; 2]> = (0..audio_len)
        .map(|t| [((t as f32) * 0.09).cos() * 0.3, 0.1])
        .collect();
    let mut seeds = vec![0.0_f32; PRESCREEN_SEED_PARAM_STRIDE];
    seeds[0] = 0.08;
    seeds[1] = 0.8;
    seeds[2] = 0.5;
    seeds[6] = 0.85;
    seeds[12] = 1.0;
    seeds[14] = 1.3;
    let pre = runner
        .prescreen_population(&PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 1,
                num_frames: audio_len as u32,
                num_quads: 1,
                num_stems: 1,
                sample_rate_hz: 48_000.0,
                eps_floor: 1.0e-5,
                _pad0: 0,
                _pad1: 0,
            },
            input_audio: &in_a,
            target_audio: &tg_a,
            seed_params: &seeds,
        })
        .unwrap();
    assert_eq!(pre.len(), 1);

    let gabor = runner
        .gabor_bezout_grad(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: audio_len as u32,
                fft_size: 64,
                hop_size: 64,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 0.7,
                w_hf: 0.4,
                eps_noise: 1.0e-5,
            },
            pred_audio: &in_a,
            target_audio: &tg_a,
            bezout_taps: &vec![[0.5, 1.0, 1.0, 1.0]; 64],
        })
        .unwrap();
    assert_eq!(gabor.band_losses.len(), 4);

    let cpu = CpuRunner::scalar();
    let feats: Vec<f32> = (0..3 * 128).map(|i| ((i as f32) * 0.11).sin()).collect();
    let scales = [[1.0_f32, 1.0_f32]; 3];
    let oz = cpu
        .ozaki_gram_rhs_i32(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                num_frames: 128,
                feature_dim: 3,
                target_channels: 0,
                chunk_len: 1024,
                num_chunks: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            features: &feats,
            targets: &[],
            col_scales: &scales,
            seed_buckets: None,
        })
        .unwrap();
    assert_eq!(
        oz.ozaki_buckets.len(),
        uplo_u_packed_len(3) * OZAKI_WORDS_PER_PAIR
    );

    let opt = runner
        .inner_opt_step(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: 4,
                history_m: 4,
                step_idx: 0,
                line_search_iter: 0,
                step_size: 0.5,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                _pad: 0,
            },
            theta: &[1.0, 2.0, -1.5, 0.5],
            grad: &[0.5, 1.0, -0.5, 0.25],
            param_bounds: &[[-3.0, 3.0]; 4],
            lbfgs_s_y: &[[0.0; 4]; 20],
            search_dir: &[0.0; 4],
        })
        .unwrap();
    assert!(opt.theta[0] < 1.0);
}

proptest! {
    #[test]
    fn prop_planner_desktop_and_browser_invariants(
        num_frames in 1usize..96_000,
        num_quads in 1usize..8,
        num_stems in 1usize..12,
        include_adjoint in any::<bool>(),
    ) {
        let wl = CoreScanWorkload { num_frames, num_quads, num_stems, include_adjoint };
        let plan = plan_core_scan_execution(wl, MemoryBudgetProfile::BrowserWebGpu, None).unwrap();
        prop_assert!(plan.footprints.max_single_binding_bytes <= MAX_WEBGPU_BINDING_BYTES);
        prop_assert!(plan.footprints.peak_total_bytes <= BROWSER_WEBGPU_BUDGET_BYTES);
        if matches!(plan.mode, ExecutionMode::Streamed) {
            prop_assert_eq!(plan.ckpt_slot_tiles, plan.time_block_tiles + STREAMED_CKPT_EXTRA_TILES);
        }
    }
}
