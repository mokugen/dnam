#![forbid(unsafe_code)]
//! Integration suite verifying bit-exact and fixed-precision CPU <-> `wgpu 30.0.2` hardware
//! parity across both `ShaderVariant::Portable` and `ShaderVariant::Subgroups`, `< 1 ms` startup
//! probes, Zero-DC `+0.0`, `0-ULP` Ozaki Gram/RHS + `target_channels == 0` guard, `Streamed` <->
//! `ResidentBatched` equivalence, and `TrainRunner::auto_detect()` (`verify[train.pipeline.webgpu-scans-and-ozaki]`,
//! `verify[train.pipeline.isomorphism-parity]`).

#[cfg(feature = "gpu-wgpu")]
mod gpu_parity {
    use dnam_kernels::{
        CoreQuadParamsStereo4, GpuVec4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
        SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4, accum_ozaki_chunk_i32, pow2_ceil_fp32,
        prewarp_tan_pi_f_over_fs_f32, reconstruct_ozaki_dot_f64,
    };
    use dnam_train::{
        ActiveComputeBackend, AdjointScanInput, CoreScanWorkload, CpuRunner, ExecutionMode,
        FusedCoreScanInput, GaborBezoutInput, InnerOptInput, MemoryBudgetProfile, OzakiGramInput,
        PrescreenInput, ShaderVariant, TrainRunner, WgpuRunner,
        abi::{
            AdjointScanUniforms, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
            GaborBezoutUniforms, InnerOptUniforms, OZAKI_WORDS_PER_PAIR, OzakiGramUniforms,
            PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
            quad_param_slots,
        },
        pack_core_quad_params, plan_core_scan_execution, uplo_u_index, uplo_u_packed_len,
    };
    use proptest::prelude::*;

    fn build_quad(stem: usize, quad: usize, zero_g: bool) -> [[f32; 4]; 24] {
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

    fn acquire_variants() -> Vec<(ShaderVariant, WgpuRunner)> {
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

    #[test]
    fn test_wgpu_startup_probe_ozaki_prescreen_gabor_and_opt_parity() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        // verify[train.pipeline.isomorphism-parity]
        let runners = acquire_variants();
        if runners.is_empty() {
            return;
        }
        let cpu = CpuRunner::detect();

        for (variant, gpu) in &runners {
            let label = variant.label();
            assert!(
                gpu.context().max_workgroup_storage_bytes() >= 16_384,
                "{label} SRAM limit too small"
            );
            assert!(
                gpu.context().startup_probe_elapsed().as_millis() < 250,
                "{label} startup probe unusually slow: {:?}",
                gpu.context().startup_probe_elapsed()
            );

            // 1. 0-ULP Ozaki Gram + RHS + target_channels == 0 guard
            let num_frames = 1_500_usize;
            let dim = 4_usize;
            let tc = 2_usize;
            let mut features = vec![0.0_f32; dim * num_frames];
            let mut targets = vec![0.0_f32; tc * num_frames];
            let mut scales = vec![[0.0_f32; 2]; dim + tc];
            for c in 0..dim {
                let mut m = 0.0_f32;
                for t in 0..num_frames {
                    let v = ((t as f32) * 0.037 * ((c + 1) as f32)).sin() * (1.2 + c as f32);
                    features[c * num_frames + t] = v;
                    m = m.max(v.abs());
                }
                let (s, is) = pow2_ceil_fp32(m);
                scales[c] = [s, is];
            }
            for ch in 0..tc {
                let mut m = 0.0_f32;
                for t in 0..num_frames {
                    let v = ((t as f32) * 0.071 + (ch as f32)).cos() * 2.1;
                    targets[ch * num_frames + t] = v;
                    m = m.max(v.abs());
                }
                scales[dim + ch] = [m, 0.0];
            }
            let oz_in = OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    num_frames: num_frames as u32,
                    feature_dim: dim as u32,
                    target_channels: tc as u32,
                    chunk_len: 1024,
                    num_chunks: 2,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
                features: &features,
                targets: &targets,
                col_scales: &scales,
                seed_buckets: None,
            };
            let oz_cpu = cpu.ozaki_gram_rhs_i32(&oz_in).unwrap();
            let oz_gpu = gpu.ozaki_gram_rhs_i32(&oz_in).unwrap();
            assert_eq!(
                oz_gpu.ozaki_buckets, oz_cpu.ozaki_buckets,
                "Ozaki 0-ULP ({label})"
            );
            let (g_c, r_c) = cpu
                .reconstruct_ozaki_gram_rhs_f64(&oz_cpu, dim, tc, &scales)
                .unwrap();
            let (g_g, r_g) = gpu
                .reconstruct_ozaki_gram_rhs_f64(&oz_gpu, dim, tc, &scales)
                .unwrap();
            assert_eq!(g_c, g_g);
            assert_eq!(r_c, r_g);

            // target_channels == 0 sentinel guard
            let sentinel = 0x5A5A_5A5A_i32;
            let gp = uplo_u_packed_len(3);
            let seed = vec![sentinel; (gp + 2) * OZAKI_WORDS_PER_PAIR];
            let z_in = OzakiGramInput {
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
                features: &features[..3 * 128],
                targets: &[],
                col_scales: &scales[..3],
                seed_buckets: Some(&seed),
            };
            let z_cpu = cpu.ozaki_gram_rhs_i32(&z_in).unwrap();
            let z_gpu = gpu.ozaki_gram_rhs_i32(&z_in).unwrap();
            assert_eq!(z_gpu.ozaki_buckets, z_cpu.ozaki_buckets);
            assert_eq!(
                &z_gpu.ozaki_buckets[gp * OZAKI_WORDS_PER_PAIR..],
                &vec![sentinel; 2 * OZAKI_WORDS_PER_PAIR]
            );

            // 2. prescreen_population parity
            let pn = 300;
            let in_a: Vec<[f32; 2]> = (0..pn)
                .map(|t| [((t as f32) * 0.11).sin() * 0.5, 0.3])
                .collect();
            let tg_a: Vec<[f32; 2]> = (0..pn)
                .map(|t| [((t as f32) * 0.09).cos() * 0.2, 0.15])
                .collect();
            let mut seed_p = vec![0.0_f32; 2 * PRESCREEN_SEED_PARAM_STRIDE];
            for i in 0..2 {
                let b = i * PRESCREEN_SEED_PARAM_STRIDE;
                seed_p[b] = 0.04 + 0.05 * (i as f32);
                seed_p[b + 1] = 0.75;
                seed_p[b + 2] = 0.4;
                seed_p[b + 6] = 0.85;
                seed_p[b + 7] = 0.35;
                seed_p[b + 12] = 1.0;
                seed_p[b + 14] = 1.4;
            }
            let p_in = PrescreenInput {
                uniforms: PrescreenUniforms {
                    num_seeds: 2,
                    num_frames: pn as u32,
                    num_quads: 1,
                    num_stems: 1,
                    sample_rate_hz: 48_000.0,
                    eps_floor: 1.0e-5,
                    _pad0: 0,
                    _pad1: 0,
                },
                input_audio: &in_a,
                target_audio: &tg_a,
                seed_params: &seed_p,
            };
            let pre_c = cpu.prescreen_population(&p_in).unwrap();
            let pre_g = gpu.prescreen_population(&p_in).unwrap();
            for (c, g) in pre_c.iter().zip(&pre_g) {
                assert!(
                    (c - g).abs() <= 1.0e-4 * c.abs().max(1.0),
                    "prescreen {label}"
                );
            }

            // 3. gabor_bezout_grad Mode A & Mode B parity
            let g_in_a = GaborBezoutInput {
                uniforms: GaborBezoutUniforms {
                    num_frames: 270,
                    fft_size: 0,
                    hop_size: 4,
                    num_stems: 1,
                    w_lf: 1.1,
                    w_mf: 0.75,
                    w_hf: 0.45,
                    eps_noise: 1.0e-5,
                },
                pred_audio: &in_a[..270],
                target_audio: &tg_a[..270],
                bezout_taps: &[
                    [0.25, -0.35, 0.45, 1.0],
                    [0.35, 0.20, -0.35, 1.0],
                    [0.25, 0.30, 0.15, 1.0],
                    [0.15, -0.15, -0.25, 1.0],
                ],
            };
            let gba_c = cpu.gabor_bezout_grad(&g_in_a).unwrap();
            let gba_g = gpu.gabor_bezout_grad(&g_in_a).unwrap();
            for (c, g) in gba_c.band_losses.iter().zip(&gba_g.band_losses) {
                assert!((c[3] - g[3]).abs() <= 1.0e-4 * c[3].abs().max(1.0));
            }
            for (c, g) in gba_c.d_loss_d_pred.iter().zip(&gba_g.d_loss_d_pred) {
                assert!((c[0] - g[0]).abs() <= 1.0e-4);
                assert!((c[1] - g[1]).abs() <= 1.0e-4);
            }

            let g_in = GaborBezoutInput {
                uniforms: GaborBezoutUniforms {
                    num_frames: 128,
                    fft_size: 64,
                    hop_size: 64,
                    num_stems: 1,
                    w_lf: 1.0,
                    w_mf: 0.7,
                    w_hf: 0.4,
                    eps_noise: 1.0e-5,
                },
                pred_audio: &in_a[..128],
                target_audio: &tg_a[..128],
                bezout_taps: &vec![[0.5, 1.0, 1.0, 1.0]; 64],
            };
            let gb_c = cpu.gabor_bezout_grad(&g_in).unwrap();
            let gb_g = gpu.gabor_bezout_grad(&g_in).unwrap();
            for (c, g) in gb_c.band_losses.iter().zip(&gb_g.band_losses) {
                assert!((c[3] - g[3]).abs() <= 1.0e-4 * c[3].abs().max(1.0));
            }
            for (c, g) in gb_c.d_loss_d_pred.iter().zip(&gb_g.d_loss_d_pred) {
                assert!((c[0] - g[0]).abs() <= 1.0e-4);
                assert!((c[1] - g[1]).abs() <= 1.0e-4);
            }

            // 4. inner_opt_step parity
            let mut th_c = vec![1.5_f32; 32];
            let mut th_g = th_c.clone();
            let grad = vec![0.4_f32; 32];
            let bnd = vec![[-1.0_f32, 1.0]; 32];
            let sy = vec![[0.0_f32; 4]; 5 * 32];
            let dir = vec![0.0_f32; 32];
            let o_c = cpu
                .inner_opt_step(&InnerOptInput {
                    uniforms: InnerOptUniforms {
                        num_params: 32,
                        history_m: 4,
                        step_idx: 0,
                        line_search_iter: 0,
                        step_size: 0.5,
                        damping_lambda: 1.0e-4,
                        min_bound_eps: 1.0e-4,
                        _pad: 0,
                    },
                    theta: &th_c,
                    grad: &grad,
                    param_bounds: &bnd,
                    lbfgs_s_y: &sy,
                    search_dir: &dir,
                })
                .unwrap();
            let o_g = gpu
                .inner_opt_step(&InnerOptInput {
                    uniforms: InnerOptUniforms {
                        num_params: 32,
                        history_m: 4,
                        step_idx: 0,
                        line_search_iter: 0,
                        step_size: 0.5,
                        damping_lambda: 1.0e-4,
                        min_bound_eps: 1.0e-4,
                        _pad: 0,
                    },
                    theta: &th_g,
                    grad: &grad,
                    param_bounds: &bnd,
                    lbfgs_s_y: &sy,
                    search_dir: &dir,
                })
                .unwrap();
            th_c = o_c.theta;
            th_g = o_g.theta;
            for (c, g) in th_c.iter().zip(&th_g) {
                assert!((c - g).abs() <= 1.0e-5);
            }
        }
    }

    #[test]
    fn test_wgpu_fused_core_streamed_zero_dc_and_adjoint_svf_gk_parity() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        // verify[train.pipeline.isomorphism-parity]
        let runners = acquire_variants();
        if runners.is_empty() {
            return;
        }
        let cpu = CpuRunner::detect();

        let num_frames = 600_usize;
        let num_tiles = num_frames.div_ceil(TILE_FRAMES);
        let num_quads = 2_usize;
        let num_stems = 2_usize;
        let total_q = num_quads * num_stems;
        let mut quad_params = Vec::with_capacity(total_q * 24);
        for s in 0..num_stems {
            for q in 0..num_quads {
                quad_params.extend_from_slice(&build_quad(s, q, false));
            }
        }
        let mut core_inputs = vec![[0.0_f32; 4]; total_q * num_frames * CORE_INPUT_VEC4_STRIDE];
        for s in 0..num_stems {
            for q in 0..num_quads {
                let qg = s * num_quads + q;
                for t in 0..num_frames {
                    let ph = (t as f32) * 0.097 + (qg as f32) * 0.41;
                    let idx = (qg * num_frames + t) * 2;
                    core_inputs[idx] = [
                        0.45 * ph.sin() + 0.15 * (2.3 * ph).cos(),
                        0.38 * (1.4 * ph).cos(),
                        0.42 * ph.cos() - 0.10 * (3.1 * ph).sin(),
                        0.30 * (1.9 * ph).cos() + 0.12,
                    ];
                    core_inputs[idx + 1] = [
                        0.40 * (1.1 * ph).cos(),
                        0.48 * ph.sin() + 0.12,
                        0.35 * (2.1 * ph).cos() - 0.14 * ph.sin(),
                        -0.25 * (1.3 * ph).sin(),
                    ];
                }
            }
        }

        let fwd_u = FusedScanUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_quads: num_quads as u32,
            num_stems: num_stems as u32,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx: 0,
        };
        let fwd_in = FusedCoreScanInput {
            uniforms: fwd_u,
            core_inputs: &core_inputs,
            quad_params: &quad_params,
            initial_checkpoints: None,
        };
        let fwd_cpu = cpu.fused_core_scan_256(&fwd_in).unwrap();
        let peak = fwd_cpu
            .feature_outputs
            .iter()
            .flat_map(|v| v.iter().copied())
            .fold(1.0_f32, |m, x| m.max(x.abs()));
        let tol = 1.0e-4 * peak;

        let mut d_loss = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];
        for (idx, item) in d_loss.iter_mut().enumerate() {
            let ph = (idx as f32) * 0.017;
            *item = [
                0.35 * ph.sin(),
                0.30 * (1.3 * ph).cos(),
                0.28 * ph.cos(),
                -0.22 * ph.sin(),
            ];
        }

        for (variant, gpu) in &runners {
            let label = variant.label();
            let fwd_gpu = gpu.fused_core_scan_256(&fwd_in).unwrap();
            for (c, g) in fwd_cpu.feature_outputs.iter().zip(&fwd_gpu.feature_outputs) {
                for l in 0..4 {
                    assert!(
                        (c[l] - g[l]).abs() <= tol,
                        "fused_core_scan_256 ({label}) mismatch: cpu={} gpu={} tol={tol}",
                        c[l],
                        g[l]
                    );
                }
            }
            // Verify single-step frame t=0 stays within 1.5 ULP
            for l in 0..4 {
                let c_bits = fwd_cpu.feature_outputs[0][l].to_bits();
                let g_bits = fwd_gpu.feature_outputs[0][l].to_bits();
                assert!(
                    c_bits.abs_diff(g_bits) <= 2,
                    "single-step ULP ({label}): cpu={} gpu={}",
                    fwd_cpu.feature_outputs[0][l],
                    fwd_gpu.feature_outputs[0][l]
                );
            }

            // Streamed <-> ResidentBatched 0-ULP bit identity on GPU
            let mut plan = plan_core_scan_execution(
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
            plan.mode = ExecutionMode::Streamed;
            plan.stem_group = 1;
            plan.time_block_frames = TILE_FRAMES;
            plan.time_block_tiles = 1;
            let streamed_gpu = gpu.run_fused_core_scan_planned(&fwd_in, &plan).unwrap();
            for (a, b) in fwd_gpu
                .feature_outputs
                .iter()
                .zip(&streamed_gpu.feature_outputs)
            {
                assert_eq!(
                    a.map(f32::to_bits),
                    b.map(f32::to_bits),
                    "Streamed != Resident ({label})"
                );
            }
            for (a, b) in fwd_gpu
                .tile_checkpoints
                .iter()
                .zip(&streamed_gpu.tile_checkpoints)
            {
                assert_eq!(
                    a.map(f32::to_bits),
                    b.map(f32::to_bits),
                    "Streamed ckpt != Resident ({label})"
                );
            }

            // Pure -0.0 inputs + negative weights Zero-DC +0.0 verification
            let neg_in = vec![[-0.0_f32; 4]; core_inputs.len()];
            let mut neg_params = quad_params.clone();
            for q in 0..total_q {
                neg_params[q * 24 + quad_param_slots::TONE_LP] = [-0.7; 4];
                neg_params[q * 24 + quad_param_slots::GLU_MIX] = [-0.6; 4];
            }
            let z_out = gpu
                .fused_core_scan_256(&FusedCoreScanInput {
                    uniforms: fwd_u,
                    core_inputs: &neg_in,
                    quad_params: &neg_params,
                    initial_checkpoints: None,
                })
                .unwrap();
            assert!(
                z_out
                    .feature_outputs
                    .iter()
                    .all(|v| v.iter().all(|x| x.to_bits() == 0)),
                "Zero-DC -0.0 leak ({label})"
            );

            // 3-pass adjoint GPU vs CPU parity (including SVF_K and SVF_G >= MIN_TPT_G)
            let adj_in = AdjointScanInput {
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
                tile_checkpoints: &fwd_gpu.tile_checkpoints,
                d_loss_d_features: &d_loss,
                feature_outputs: &fwd_gpu.feature_outputs,
            };
            let adj_cpu = cpu.adjoint_scan_256(&adj_in).unwrap();
            let adj_gpu = gpu.adjoint_scan_256(&adj_in).unwrap();
            let adj_peak = adj_cpu
                .d_loss_d_inputs
                .iter()
                .flat_map(|v| v.iter().copied())
                .fold(1.0_f32, |m, x| m.max(x.abs()));
            for (c, g) in adj_cpu.d_loss_d_inputs.iter().zip(&adj_gpu.d_loss_d_inputs) {
                for l in 0..4 {
                    assert!(
                        (c[l] - g[l]).abs() <= 5.0e-4 * adj_peak,
                        "adjoint d_inputs ({label}) mismatch: cpu={} gpu={}",
                        c[l],
                        g[l]
                    );
                }
            }
            for (slot, (c, g)) in adj_cpu
                .param_grads
                .iter()
                .zip(&adj_gpu.param_grads)
                .enumerate()
            {
                for l in 0..4 {
                    assert!(
                        (c[l] - g[l]).abs() <= 5.0e-3 * c[l].abs().max(1.0),
                        "adjoint param_grads ({label}) slot={slot} lane={l}: cpu={} gpu={}",
                        c[l],
                        g[l]
                    );
                }
            }

            // Explicit SVF_G == 0.0 (< MIN_TPT_G) zero-guard parity across CPU and GPU.
            let mut zero_g_params = Vec::with_capacity(total_q * 24);
            for s in 0..num_stems {
                for q in 0..num_quads {
                    zero_g_params.extend_from_slice(&build_quad(s, q, true));
                }
            }
            let zero_g_in = AdjointScanInput {
                quad_params: &zero_g_params,
                ..adj_in
            };
            let zg_cpu = cpu.adjoint_scan_256(&zero_g_in).unwrap();
            let zg_gpu = gpu.adjoint_scan_256(&zero_g_in).unwrap();
            for tile_block in 0..(total_q * num_tiles) {
                let svf_g_idx = tile_block * 24 + quad_param_slots::SVF_G;
                assert_eq!(zg_cpu.param_grads[svf_g_idx], [0.0; 4]);
                assert_eq!(zg_gpu.param_grads[svf_g_idx], [0.0; 4]);
            }
        }

        // Verify TrainRunner::auto_detect selects hardware Wgpu on Apple Silicon Metal and covers
        // Debug/getters, ResidentBatched planned mode, and validation guards on WgpuRunner.
        let auto = TrainRunner::auto_detect();
        if cfg!(target_os = "macos") {
            assert!(matches!(
                auto.active_backend(),
                ActiveComputeBackend::Wgpu { .. }
            ));
        }
        assert!(!format!("{auto:?}").is_empty());
        if let TrainRunner::Wgpu(ref wr) = auto {
            assert!(!format!("{:?}", wr.context()).is_empty());
            let _ = wr.context().supports_subgroups();
            let cpu_adj_in = AdjointScanInput {
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
                tile_checkpoints: &fwd_cpu.tile_checkpoints,
                d_loss_d_features: &d_loss,
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
        let _ = (
            uplo_u_index(0, 0),
            TILE_CKPT_VEC4_STRIDE,
            accum_ozaki_chunk_i32,
            reconstruct_ozaki_dot_f64,
        );
    }

    proptest! {
        #[test]
        fn prop_cpu_gpu_single_chunk_ozaki_zero_ulp(
            xs in prop::collection::vec(-3.0_f32..3.0_f32, 8..64),
            ys in prop::collection::vec(-3.0_f32..3.0_f32, 8..64),
        ) {
            let Ok(gpu) = WgpuRunner::try_new() else {
                return Ok(());
            };
            let cpu = CpuRunner::detect();
            let n = xs.len().min(ys.len());
            let mut feats = Vec::with_capacity(2 * n);
            feats.extend_from_slice(&xs[..n]);
            feats.extend_from_slice(&ys[..n]);
            let mx = xs[..n].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let my = ys[..n].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let scales = [[mx, 0.0], [my, 0.0]];
            let inp = OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    num_frames: n as u32,
                    feature_dim: 2,
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
            };
            let c_out = cpu.ozaki_gram_rhs_i32(&inp).unwrap();
            let g_out = gpu.ozaki_gram_rhs_i32(&inp).unwrap();
            prop_assert_eq!(g_out.ozaki_buckets, c_out.ozaki_buckets);
        }
    }
}
