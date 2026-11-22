#![forbid(unsafe_code)]
//! Shared test fixtures and validation-guard helpers for `tests/cpu_wgpu_parity.rs`.

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
    SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
};
use dnam_train::{
    ActiveComputeBackend, AdjointScanInput, CoreScanWorkload, ExecutionMode, FusedCoreScanInput,
    FusedCoreScanOutput, GaborBezoutInput, InnerOptInput, MemoryBudgetProfile, OzakiGramInput,
    PrescreenInput, ShaderVariant, TrainRunner, WgpuRunner,
    abi::{
        AdjointScanUniforms, FusedScanUniforms, GaborBezoutUniforms, InnerOptUniforms,
        OzakiGramUniforms, PrescreenUniforms, quad_param_slots,
    },
    pack_core_quad_params, plan_core_scan_execution,
};

pub(crate) fn build_quad(stem: usize, quad: usize, zero_g: bool) -> [[f32; 4]; 24] {
    let f = (stem * 2 + quad) as f32;
    let fc = [0.02 + 0.015 * f, 0.07 + 0.012 * f, 0.16 + 0.01 * f, 0.28];
    let g = GpuVec4::from_array(fc.map(prewarp_tan_pi_f_over_fs_f32));
    let k = GpuVec4::from_array([0.55 + 0.05 * f, 0.85, 1.15, 1.45]);
    let p = CoreQuadParamsStereo4 {
        svf: TptSvfCoeffs4::from_gk(g, k),
        tone: SonaToneWeights4 {
            lp: GpuVec4::from_array([0.45, 0.20, -0.15, 0.10]),
            bp: GpuVec4::from_array([0.25, 0.40, 0.30, -0.20]),
            hp: GpuVec4::from_array([-0.15, 0.15, 0.35, 0.45]),
            ap: GpuVec4::from_array([0.10, -0.10, 0.15, 0.10]),
            dir: GpuVec4::from_array([0.20, 0.15, -0.10, 0.25]),
        },
        lru: SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::from_array([0.92, 0.86, 0.78, 0.68]),
            GpuVec4::from_array([0.25, 0.40, 0.55, 0.75]),
            GpuVec4::from_array([0.15, 0.35, 0.60, 0.85]),
            GpuVec4::from_array([0.12, 0.28, 0.45, 0.65]),
            GpuVec4::from_array([0.20, 0.40, 0.55, 0.70]),
            GpuVec4::from_array([0.05, 0.25, 0.45, 0.65]),
        ),
        two_svf_g: g.add(g),
        svf_k_nl: GpuVec4::zero(),
        rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
        sag_depth: GpuVec4::from_array([0.18, 0.32, 0.45, 0.22]),
        adaa_drive: GpuVec4::from_array([1.15, 1.65, 2.10, 1.35]),
        lru_bias_drive: GpuVec4::from_array([0.15, -0.22, 0.30, -0.18]),
        wavefolder_mix: GpuVec4::from_array([0.05, 0.35, 0.70, 0.95]),
        glu_mix: GpuVec4::from_array([0.65, -0.45, 0.80, -0.55]),
    };
    let mut packed = pack_core_quad_params(&p);
    if zero_g {
        packed[quad_param_slots::SVF_G] = [0.0; 4];
    }
    packed
}

pub(crate) fn acquire_variants() -> Vec<(ShaderVariant, WgpuRunner)> {
    let mut out = Vec::new();
    match WgpuRunner::try_new_with_variant(ShaderVariant::Portable) {
        Ok(p) => out.push((ShaderVariant::Portable, p)),
        Err(err) if cfg!(target_os = "macos") => {
            panic!("Metal adapter required on macOS: {err}")
        }
        Err(_) => return out,
    }
    if let Ok(sg) = WgpuRunner::try_new_with_variant(ShaderVariant::Subgroups) {
        out.push((ShaderVariant::Subgroups, sg));
    } else if cfg!(target_os = "macos") {
        panic!("Apple Silicon Metal adapter must support ShaderVariant::Subgroups");
    }
    out
}

pub(crate) fn verify_auto_detect_and_wgpu_validation_guards(
    fwd_in: FusedCoreScanInput<'_>,
    fwd_cpu: &FusedCoreScanOutput,
    d_loss: &[[f32; 4]],
) {
    let fwd_u = fwd_in.uniforms;
    let num_frames = fwd_u.num_frames as usize;
    let num_quads = fwd_u.num_quads as usize;
    let num_stems = fwd_u.num_stems as usize;
    let auto = TrainRunner::auto_detect();
    if cfg!(target_os = "macos") {
        assert!(matches!(
            auto.active_backend(),
            ActiveComputeBackend::Wgpu { .. }
        ));
    }
    assert!(!format!("{auto:?}").is_empty());
    let TrainRunner::Wgpu(ref wr) = auto else {
        return;
    };
    assert!(!format!("{:?}", wr.context()).is_empty());
    let _ = wr.context().supports_subgroups();
    let cpu_adj_in = AdjointScanInput {
        uniforms: AdjointScanUniforms {
            num_frames: fwd_u.num_frames,
            num_tiles: fwd_u.num_tiles,
            num_quads: fwd_u.num_quads,
            num_stems: fwd_u.num_stems,
            pass_mode: 2,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx: 0,
        },
        core_inputs: fwd_in.core_inputs,
        quad_params: fwd_in.quad_params,
        tile_checkpoints: &fwd_cpu.tile_checkpoints,
        d_loss_d_features: d_loss,
        feature_outputs: &fwd_cpu.feature_outputs,
    };
    let res_plan = plan_core_scan_execution(
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
    assert_eq!(res_plan.mode, ExecutionMode::ResidentBatched);
    let planned_res = auto
        .run_fused_core_scan_planned(&fwd_in, &res_plan)
        .unwrap();
    assert_eq!(
        planned_res.feature_outputs.len(),
        fwd_cpu.feature_outputs.len()
    );
    assert!(auto.adjoint_scan_256(&cpu_adj_in).is_ok());
    assert!(auto.fused_core_scan_256(&fwd_in).is_ok());
    assert!(
        auto.prescreen_population(&PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 1,
                num_frames: 16,
                num_quads: 1,
                num_stems: 1,
                sample_rate_hz: 48_000.0,
                eps_floor: 1.0e-5,
                _pad0: 0,
                _pad1: 0,
            },
            input_audio: &[[0.1, 0.2]; 16],
            target_audio: &[[0.1, 0.2]; 16],
            seed_params: &[0.05; 16],
        })
        .is_ok()
    );
    assert!(
        auto.gabor_bezout_grad(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: 16,
                fft_size: 16,
                hop_size: 16,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 0.7,
                w_hf: 0.4,
                eps_noise: 1.0e-5,
            },
            pred_audio: &[[0.1, 0.2]; 16],
            target_audio: &[[0.1, 0.2]; 16],
            bezout_taps: &[[1.0; 4]; 16],
        })
        .is_ok()
    );
    assert!(
        auto.inner_opt_step(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: 4,
                history_m: 1,
                step_idx: 0,
                line_search_iter: 0,
                step_size: 0.2,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                _pad: 0,
            },
            theta: &[0.2; 4],
            grad: &[0.1; 4],
            param_bounds: &[[-1.0, 1.0]; 4],
            lbfgs_s_y: &[[0.0; 4]; 8],
            search_dir: &[0.0; 4],
        })
        .is_ok()
    );
    assert!(
        auto.reconstruct_ozaki_gram_rhs_f64(
            &auto
                .ozaki_gram_rhs_i32(&OzakiGramInput {
                    uniforms: OzakiGramUniforms {
                        num_frames: 4,
                        feature_dim: 1,
                        target_channels: 0,
                        chunk_len: 1024,
                        num_chunks: 1,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                    features: &[1.0; 4],
                    targets: &[],
                    col_scales: &[[1.0, 1.0]],
                    seed_buckets: None,
                })
                .unwrap(),
            1,
            0,
            &[[1.0, 1.0]],
        )
        .is_ok()
    );

    // Exercise WgpuRunner validation error arms across all 5 GPU dispatch methods.
    assert!(
        wr.fused_core_scan_256(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: 0,
                ..fwd_u
            },
            ..fwd_in
        })
        .is_err()
    );
    assert!(
        wr.fused_core_scan_256(&FusedCoreScanInput {
            core_inputs: &[],
            ..fwd_in
        })
        .is_err()
    );
    assert!(
        wr.adjoint_scan_256(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: 0,
                ..cpu_adj_in.uniforms
            },
            ..cpu_adj_in
        })
        .is_err()
    );
    assert!(
        wr.adjoint_scan_256(&AdjointScanInput {
            core_inputs: &[],
            ..cpu_adj_in
        })
        .is_err()
    );
    assert!(
        wr.gabor_bezout_grad(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: 0,
                fft_size: 0,
                hop_size: 4,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 1.0,
                w_hf: 1.0,
                eps_noise: 1.0e-5,
            },
            pred_audio: &[],
            target_audio: &[],
            bezout_taps: &[],
        })
        .is_err()
    );
    assert!(
        wr.gabor_bezout_grad(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: 64,
                fft_size: 64,
                hop_size: 32,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 1.0,
                w_hf: 1.0,
                eps_noise: 1.0e-5,
            },
            pred_audio: &[[0.0; 2]; 64],
            target_audio: &[[0.0; 2]; 64],
            bezout_taps: &[[1.0; 4]; 64],
        })
        .is_err()
    );
    assert!(
        wr.ozaki_gram_rhs_i32(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                num_frames: 0,
                feature_dim: 1,
                target_channels: 0,
                chunk_len: 1024,
                num_chunks: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            features: &[],
            targets: &[],
            col_scales: &[],
            seed_buckets: None,
        })
        .is_err()
    );
    assert!(
        wr.ozaki_gram_rhs_i32(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                num_frames: 4,
                feature_dim: 2,
                target_channels: 0,
                chunk_len: 1024,
                num_chunks: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            features: &[1.0; 8],
            targets: &[],
            col_scales: &[[1.0, 1.0]; 2],
            seed_buckets: Some(&[0]),
        })
        .is_err()
    );
    assert!(
        wr.inner_opt_step(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: 0,
                history_m: 1,
                step_idx: 0,
                line_search_iter: 0,
                step_size: 0.1,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                _pad: 0,
            },
            theta: &[],
            grad: &[],
            param_bounds: &[],
            lbfgs_s_y: &[],
            search_dir: &[],
        })
        .is_err()
    );
}
