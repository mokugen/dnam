#![forbid(unsafe_code)]
//! M22 Pillar 3 GPU <-> CPU parity & persistent-arena tests (`ShaderVariant::Portable` &
//! `ShaderVariant::Subgroups`):
//! 1. `tile_mix_fwd_adj` all 4 modes (`0..=3`, multi-stem `6`-quad DAG).
//! 2. 2D split-64 `ozaki_gram_rhs_split64` (`D=384, tc=2`, readback `<= 1.2 MiB`, `0-ULP` parity).
//! 3. Multi-candidate `(1, num_candidates, 1)` `inner_opt_step` (`AdamW` + `grad_clip_norm` + `ste_fp16`
//!    across transforms `0..=5`, and `L-SSBroyden` `m=16` multi-step `tau_k` parity).
//! 4. High-`alpha_rel = 0.999` cross-block streamed `adjoint_scan_256` seam + `FUSED_FLAG_FIXED_POINT_SCAN`.
//! 5. Ultra `10`-quad `35`-tile `run_wave_core_cascade` + `run_wave_core_adjoint` (resident +
//!    `> WAVE_STREAMING_MAX_FRAMES` streamed), GPU central-FD tile weight check, and `GpuTrainArena`
//!    zero-growth reuse + `DESKTOP_CAPSULE_BUDGET_BYTES` invariant.

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS, pow2_ceil_fp32};
use dnam_train::{
    AdjointScanInput, CpuRunner, DESKTOP_CAPSULE_BUDGET_BYTES, FusedCoreScanInput, InnerOptInput,
    TileMixInput, WAVE_STREAMING_MAX_FRAMES, WaveCoreAdjointInput, WaveCoreCascadeInput,
    abi::{
        AdjointScanUniforms, BLOCK_IDX_DIRECT_LINEAR, BLOCK_WAVE_ALL,
        COORD_TRANSFORM_ASINH_BOX_NONNEG, COORD_TRANSFORM_ASINH_BOX_SYM, COORD_TRANSFORM_BOX,
        COORD_TRANSFORM_COMPLEMENT_LOG, COORD_TRANSFORM_FREE, COORD_TRANSFORM_LOG_BOX,
        CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE, FEATURE_VEC4_STRIDE,
        FUSED_FLAG_FIXED_POINT_SCAN, FusedScanUniforms, INNER_OPT_MODE_LSSBROYDEN,
        INNER_OPT_MODE_STE_FP16, InnerOptUniforms, QUAD_PARAMS_VEC4_STRIDE, STE_FP16_META_BIT,
        TILE_CKPT_VEC4_STRIDE, TILE_FRAMES, TILE_MIX_DIR_FORWARD, TILE_MIX_DIR_PARTIAL_WEIGHTS,
        TILE_MIX_DIR_PULLBACK, TILE_MIX_DIR_REDUCE_WEIGHTS, TILE_WEIGHT_GRAD_FLOATS,
        TileMixUniforms, quad_param_slots,
    },
    uplo_u_packed_len,
};

use super::support::{
    acquire_variants, assert_vec4_close, build_quad, build_quad_with_k_nl, build_wave_descriptors,
    fill_core_inputs,
};

#[test]
fn test_wgpu_tile_mix_all_four_modes_and_ozaki_split64_high_dim_384() {
    let runners = acquire_variants();
    if runners.is_empty() {
        return;
    }
    let cpu = CpuRunner::detect();

    // 1. tile_mix_fwd_adj across all 4 modes (num_quads=6, active_tiles=13, num_stems=2, frames=300)
    let num_frames = 300_usize;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let num_quads = 6_usize;
    let active_tiles = 13_usize;
    let num_stems = 2_usize;
    let total_q = num_quads * num_stems;
    let base_in = fill_core_inputs(total_q, num_frames, 0.35);
    let desc = build_wave_descriptors(active_tiles, 0.08);
    let mut feats = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];
    for (i, f) in feats.iter_mut().enumerate() {
        let p = (i as f32) * 0.031;
        *f = [0.4 * p.sin(), -0.3 * p.cos(), 0.25 * (0.6 * p).sin(), 0.2];
    }
    let dl_in = fill_core_inputs(total_q, num_frames, 0.22);
    let dl_feat = vec![[0.12_f32, -0.09, 0.07, -0.05]; feats.len()];

    // 2. High-dim Ozaki D=384, tc=2 split-64 compact readback check (<= 1.2 MiB)
    let oz_frames = 1_500_usize;
    let dim = 384_usize;
    let tc = 2_usize;
    let expected_pairs = uplo_u_packed_len(dim) + dim * tc;
    assert!(
        expected_pairs * 4 * 4 <= 1_258_291,
        "split-64 readback > 1.2 MiB"
    );
    let mut oz_feat = vec![0.0_f32; dim * oz_frames];
    let mut oz_tgt = vec![0.0_f32; tc * oz_frames];
    let mut oz_scales = vec![[0.0_f32; 2]; dim + tc];
    for c in 0..dim {
        let mut m = 0.0_f32;
        let step = 0.017 + 0.0003 * (c as f32);
        for t in 0..oz_frames {
            let v = ((t as f32) * step + (c as f32) * 0.13).sin() * 1.4;
            oz_feat[c * oz_frames + t] = v;
            m = m.max(v.abs());
        }
        let (s, is) = pow2_ceil_fp32(m);
        oz_scales[c] = [s, is];
    }
    for ch in 0..tc {
        let mut m = 0.0_f32;
        for t in 0..oz_frames {
            let v = ((t as f32) * 0.043 + (ch as f32) * 0.5).cos() * 1.8;
            oz_tgt[ch * oz_frames + t] = v;
            m = m.max(v.abs());
        }
        let (s, is) = pow2_ceil_fp32(m);
        oz_scales[dim + ch] = [s, is];
    }
    let s64_cpu = cpu
        .ozaki_gram_rhs_split64(&oz_feat, &oz_tgt, oz_frames, dim, tc, &oz_scales)
        .unwrap();
    let (gram_c, rhs_c) = cpu
        .reconstruct_ozaki_split64_f64(&s64_cpu, dim, tc, &oz_scales)
        .unwrap();

    for (variant, gpu) in &runners {
        let label = variant.label();
        let u_fwd = TileMixUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_active_tiles: active_tiles as u32,
            dir_mode: TILE_MIX_DIR_FORWARD,
            num_stems: num_stems as u32,
            block_wave: 1,
            num_quads: num_quads as u32,
            _pad1: 0,
        };
        let fwd_c = cpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: u_fwd,
                block_inputs_or_d_loss_d_inputs: &base_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &[],
                d_loss_d_features: &[],
            })
            .unwrap();
        let fwd_g = gpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: u_fwd,
                block_inputs_or_d_loss_d_inputs: &base_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &[],
                d_loss_d_features: &[],
            })
            .unwrap();
        assert_vec4_close(
            &fwd_c.block_inputs_or_d_loss_d_inputs,
            &fwd_g.block_inputs_or_d_loss_d_inputs,
            1.0e-5,
            &format!("tile_mix fwd ({label})"),
        );

        let u_pb = TileMixUniforms {
            dir_mode: TILE_MIX_DIR_PULLBACK,
            block_wave: BLOCK_WAVE_ALL,
            ..u_fwd
        };
        let pb_c = cpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: u_pb,
                block_inputs_or_d_loss_d_inputs: &dl_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &[],
                d_loss_d_features: &dl_feat,
            })
            .unwrap();
        let pb_g = gpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: u_pb,
                block_inputs_or_d_loss_d_inputs: &dl_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &[],
                d_loss_d_features: &dl_feat,
            })
            .unwrap();
        assert_vec4_close(
            &pb_c.d_loss_d_features,
            &pb_g.d_loss_d_features,
            1.0e-5,
            &format!("tile_mix pullback ({label})"),
        );

        let u_part = TileMixUniforms {
            dir_mode: TILE_MIX_DIR_PARTIAL_WEIGHTS,
            block_wave: BLOCK_WAVE_ALL,
            ..u_fwd
        };
        let part_c = cpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: u_part,
                block_inputs_or_d_loss_d_inputs: &dl_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &[],
                d_loss_d_features: &dl_feat,
            })
            .unwrap();
        let part_g = gpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: u_part,
                block_inputs_or_d_loss_d_inputs: &dl_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &[],
                d_loss_d_features: &dl_feat,
            })
            .unwrap();
        let red_u = TileMixUniforms {
            dir_mode: TILE_MIX_DIR_REDUCE_WEIGHTS,
            ..u_part
        };
        let red_c = cpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: red_u,
                block_inputs_or_d_loss_d_inputs: &dl_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &part_c.tile_weight_grads,
                d_loss_d_features: &dl_feat,
            })
            .unwrap();
        let red_g = gpu
            .tile_mix_fwd_adj(&TileMixInput {
                uniforms: red_u,
                block_inputs_or_d_loss_d_inputs: &dl_in,
                tile_descriptors: &desc,
                feature_outputs: &feats,
                tile_weight_grads: &part_g.tile_weight_grads,
                d_loss_d_features: &dl_feat,
            })
            .unwrap();
        for (i, (&vc, &vg)) in red_c.tile_weight_grads[..active_tiles * TILE_WEIGHT_GRAD_FLOATS]
            .iter()
            .zip(&red_g.tile_weight_grads[..active_tiles * TILE_WEIGHT_GRAD_FLOATS])
            .enumerate()
        {
            assert!(
                (vc - vg).abs() <= 5.0e-5 * vc.abs().max(1.0),
                "tile_mix weight grad ({label}) idx={i}: cpu={vc} gpu={vg}"
            );
        }

        let s64_g = gpu
            .ozaki_gram_rhs_split64(&oz_feat, &oz_tgt, oz_frames, dim, tc, &oz_scales)
            .unwrap();
        assert_eq!(s64_g.num_pairs, expected_pairs);
        assert_eq!(s64_g.split64_words.len(), expected_pairs * 4);
        assert_eq!(
            s64_g.split64_words, s64_cpu.split64_words,
            "Ozaki 384D split64 ({label})"
        );
        let (gram_g, rhs_g) = gpu
            .reconstruct_ozaki_split64_f64(&s64_g, dim, tc, &oz_scales)
            .unwrap();
        assert_eq!(gram_g, gram_c);
        assert_eq!(rhs_g, rhs_c);
    }

    // Multi-batch (`> 16 * 1024 = 16,384` frames) split-64 Ozaki parity exercising
    // `OZAKI_FLAG_ACCUMULATE_SPLIT64` across multiple batches on both Portable & Subgroups.
    let mb_frames = 17_000_usize;
    let mb_dim = 8_usize;
    let mb_tc = 2_usize;
    let mb_feat: Vec<f32> = (0..mb_dim * mb_frames)
        .map(|k| ((k as f32) * 0.023 + 0.1).sin() * 1.1)
        .collect();
    let mb_tgt: Vec<f32> = (0..mb_tc * mb_frames)
        .map(|k| ((k as f32) * 0.037 + 0.4).cos() * 1.3)
        .collect();
    let (mb_s, mb_is) = pow2_ceil_fp32(1.5);
    let mb_scales = vec![[mb_s, mb_is]; mb_dim + mb_tc];
    let mb_cpu = cpu
        .ozaki_gram_rhs_split64(&mb_feat, &mb_tgt, mb_frames, mb_dim, mb_tc, &mb_scales)
        .unwrap();
    for (variant, gpu) in &runners {
        let mb_gpu = gpu
            .ozaki_gram_rhs_split64(&mb_feat, &mb_tgt, mb_frames, mb_dim, mb_tc, &mb_scales)
            .unwrap();
        assert_eq!(
            mb_gpu.split64_words,
            mb_cpu.split64_words,
            "Multi-batch split64 Ozaki ({})",
            variant.label()
        );
    }
}

#[test]
fn test_wgpu_inner_opt_multi_candidate_adamw_lssb_ste_fp16_and_streamed_seams() {
    let runners = acquire_variants();
    if runners.is_empty() {
        return;
    }
    let cpu = CpuRunner::detect();

    // 1. Multi-candidate (1, 4, 1) AdamW + grad_clip_norm + ste_fp16 across all 6 transforms (0..=5)
    let n_p = 12_usize;
    let num_cands = 4_usize;
    let bounds: Vec<[f32; 4]> = (0..n_p)
        .map(|p| {
            let tid = (p % 6) as u32;
            let meta = f32::from_bits(tid | STE_FP16_META_BIT);
            match tid {
                COORD_TRANSFORM_FREE => [-2.0, 2.0, 1.0, meta],
                COORD_TRANSFORM_BOX => [-1.5, 1.5, 1.0, meta],
                COORD_TRANSFORM_LOG_BOX => [0.01, 10.0, 1.0, meta],
                COORD_TRANSFORM_COMPLEMENT_LOG => [1.0e-4, 0.5, 1.0, meta],
                COORD_TRANSFORM_ASINH_BOX_SYM => [-8.0, 8.0, 0.5, meta],
                COORD_TRANSFORM_ASINH_BOX_NONNEG => [0.0, 8.0, 0.5, meta],
                _ => [-1.0, 1.0, 1.0, meta],
            }
        })
        .collect();
    let theta0: Vec<f32> = (0..num_cands * n_p)
        .map(|i| 0.35 * ((i as f32) * 0.29).sin())
        .collect();
    let grad0: Vec<f32> = (0..num_cands * n_p)
        .map(|i| 2.5 * ((i as f32) * 0.41 + 0.3).cos())
        .collect();
    let adam_state0 = vec![[0.0_f32; 4]; num_cands * n_p];
    let dir0 = vec![0.0_f32; num_cands * n_p];
    let u_adam = InnerOptUniforms {
        num_params: n_p as u32,
        history_m: 1,
        step_idx: 1,
        mode_flags: INNER_OPT_MODE_STE_FP16,
        step_size: 0.08,
        damping_lambda: 1.0e-4,
        min_bound_eps: 1.0e-4,
        grad_clip_norm: 1.5,
    };
    let adam_c = cpu
        .inner_opt_step(&InnerOptInput {
            uniforms: u_adam,
            num_candidates: num_cands,
            theta: &theta0,
            grad: &grad0,
            param_bounds: &bounds,
            lbfgs_s_y: &adam_state0,
            search_dir: &dir0,
        })
        .unwrap();

    // 2. High-alpha_rel = 0.999 cross-block streamed adjoint seam + Fixed-Point Scan parity
    let num_frames = 600_usize;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let inputs = fill_core_inputs(1, num_frames, 0.42);
    let mut qp = build_quad(0, 0, false);
    qp[quad_param_slots::LRU_ALPHA_REL] = [0.999; 4];
    let fwd_res = cpu
        .fused_core_scan_256(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: BLOCK_IDX_DIRECT_LINEAR,
            },
            core_inputs: &inputs,
            quad_params: &qp,
            initial_checkpoints: None,
        })
        .unwrap();
    assert!(
        fwd_res.tile_checkpoints[TILE_CKPT_VEC4_STRIDE..]
            .iter()
            .flat_map(|v| v.iter())
            .any(|&x| x.abs() > 1.0e-3),
        "BLOCK_IDX_DIRECT_LINEAR forward checkpoints must be non-zero"
    );
    let dlf = vec![[0.14_f32, -0.11, 0.09, -0.07]; fwd_res.feature_outputs.len()];
    let adj_res = cpu
        .adjoint_scan_256(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: BLOCK_IDX_DIRECT_LINEAR,
            },
            core_inputs: &inputs,
            quad_params: &qp,
            tile_checkpoints: &fwd_res.tile_checkpoints,
            d_loss_d_features: &dlf,
            feature_outputs: &fwd_res.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();

    let mut qp_phs = build_quad_with_k_nl(0, 0, [0.35, 0.65, 0.95, 1.25], false);
    qp_phs[quad_param_slots::LRU_GAMMA_DUHEM] = [0.15; 4];
    let fwd_phs_cpu = cpu
        .fused_core_scan_256(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &inputs,
            quad_params: &qp_phs,
            initial_checkpoints: None,
        })
        .unwrap();

    for (variant, gpu) in &runners {
        let label = variant.label();
        let adam_g = gpu
            .inner_opt_step(&InnerOptInput {
                uniforms: u_adam,
                num_candidates: num_cands,
                theta: &theta0,
                grad: &grad0,
                param_bounds: &bounds,
                lbfgs_s_y: &adam_state0,
                search_dir: &dir0,
            })
            .unwrap();
        for (i, (&c, &g)) in adam_c.theta.iter().zip(&adam_g.theta).enumerate() {
            assert!(
                (c - g).abs() <= 1.0e-5,
                "AdamW STE ({label}) i={i}: {c} vs {g}"
            );
        }

        // Multi-step L-SSBroyden (m=16, 4 candidates, steps 0..=2)
        let m = 16_usize;
        let mut tc = theta0.clone();
        let mut tg = theta0.clone();
        let mut syc = vec![[0.0_f32; 4]; num_cands * (m + 1) * n_p];
        let mut syg = syc.clone();
        let mut dc = vec![0.0_f32; num_cands * n_p];
        let mut dg = dc.clone();
        for step in 0..3_u32 {
            let gc: Vec<f32> = tc
                .iter()
                .enumerate()
                .map(|(k, &v)| 0.6 * v + 0.1 * ((k + step as usize) as f32).sin())
                .collect();
            let u_lss = InnerOptUniforms {
                num_params: n_p as u32,
                history_m: m as u32,
                step_idx: step,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN,
                step_size: 0.4,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                grad_clip_norm: 0.0,
            };
            let oc = cpu
                .inner_opt_step(&InnerOptInput {
                    uniforms: u_lss,
                    num_candidates: num_cands,
                    theta: &tc,
                    grad: &gc,
                    param_bounds: &bounds,
                    lbfgs_s_y: &syc,
                    search_dir: &dc,
                })
                .unwrap();
            let og = gpu
                .inner_opt_step(&InnerOptInput {
                    uniforms: u_lss,
                    num_candidates: num_cands,
                    theta: &tg,
                    grad: &gc,
                    param_bounds: &bounds,
                    lbfgs_s_y: &syg,
                    search_dir: &dg,
                })
                .unwrap();
            tc = oc.theta;
            tg = og.theta;
            syc = oc.lbfgs_s_y;
            syg = og.lbfgs_s_y;
            dc = oc.search_dir;
            dg = og.search_dir;
            for (i, (&c, &g)) in tc.iter().zip(&tg).enumerate() {
                assert!(
                    (c - g).abs() <= 2.0e-5,
                    "L-SSBroyden ({label}) s={step} i={i}: {c} vs {g}"
                );
            }
        }

        // Cross-block streamed adjoint (chunk1 [256..600] -> chunk0 [0..256]) at alpha_rel = 0.999
        let run_adj_slice = |f_start: usize, f_end: usize, exit_seed: Option<&[[f32; 4]]>| {
            let frames = f_end - f_start;
            let tiles = frames.div_ceil(TILE_FRAMES);
            let t_start = f_start / TILE_FRAMES;
            gpu.adjoint_scan_256(&AdjointScanInput {
                uniforms: AdjointScanUniforms {
                    num_frames: frames as u32,
                    num_tiles: tiles as u32,
                    num_quads: 1,
                    num_stems: 1,
                    pass_mode: 2,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: BLOCK_IDX_DIRECT_LINEAR,
                },
                core_inputs: &inputs
                    [f_start * CORE_INPUT_VEC4_STRIDE..f_end * CORE_INPUT_VEC4_STRIDE],
                quad_params: &qp,
                tile_checkpoints: &fwd_res.tile_checkpoints
                    [t_start * TILE_CKPT_VEC4_STRIDE..(t_start + tiles) * TILE_CKPT_VEC4_STRIDE],
                d_loss_d_features: &dlf[f_start * FEATURE_VEC4_STRIDE..f_end * FEATURE_VEC4_STRIDE],
                feature_outputs: &fwd_res.feature_outputs
                    [f_start * FEATURE_VEC4_STRIDE..f_end * FEATURE_VEC4_STRIDE],
                exit_adj_boundary: exit_seed,
            })
            .unwrap()
        };
        let gpu_res = run_adj_slice(0, num_frames, None);
        let c1 = run_adj_slice(TILE_FRAMES, num_frames, None);
        let mut seed_c0 = vec![[0.0_f32; 4]; EXIT_ADJ_BOUNDARY_VEC4_STRIDE];
        seed_c0[0..8].copy_from_slice(&c1.entry_adj_boundary[8..16]);
        let c0 = run_adj_slice(0, TILE_FRAMES, Some(&seed_c0));
        let du_streamed: Vec<[f32; 4]> = c0
            .d_loss_d_inputs
            .into_iter()
            .chain(c1.d_loss_d_inputs)
            .collect();
        assert_vec4_close(
            &gpu_res.d_loss_d_inputs,
            &du_streamed,
            1.0e-5,
            &format!("alpha_rel=0.999 gpu resident vs streamed du ({label})"),
        );
        assert_vec4_close(
            &adj_res.d_loss_d_inputs,
            &du_streamed,
            5.0e-4,
            &format!("alpha_rel=0.999 cpu vs gpu streamed du ({label})"),
        );

        // Fixed-point Block 0 PHS + Block 1 linear forward scan parity
        let fp_b0 = gpu
            .fused_core_scan_256(&FusedCoreScanInput {
                uniforms: FusedScanUniforms {
                    num_frames: num_frames as u32,
                    num_tiles: num_tiles as u32,
                    num_quads: 1,
                    num_stems: 1,
                    tile_start: 0,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: FUSED_FLAG_FIXED_POINT_SCAN,
                },
                core_inputs: &inputs,
                quad_params: &qp_phs,
                initial_checkpoints: None,
            })
            .unwrap();
        assert!(
            fwd_phs_cpu
                .feature_outputs
                .iter()
                .flat_map(|v| v.iter())
                .any(|&x| x.abs() > 1.0e-3),
            "Block 0 PHS fixed_point reference must be non-zero"
        );
        assert_vec4_close(
            &fwd_phs_cpu.feature_outputs,
            &fp_b0.feature_outputs,
            1.0e-4,
            &format!("fixed_point Block 0 PHS ({label})"),
        );

        let in2: Vec<[f32; 4]> = inputs.iter().chain(&inputs).copied().collect();
        let mut lin_q = qp_phs;
        lin_q[quad_param_slots::SVF_K_NL] = [0.0; 4];
        let p2: Vec<[f32; 4]> = qp_phs.iter().chain(&lin_q).copied().collect();
        let cpu_b1_ref = cpu
            .fused_core_scan_256(&FusedCoreScanInput {
                uniforms: FusedScanUniforms {
                    num_frames: num_frames as u32,
                    num_tiles: num_tiles as u32,
                    num_quads: 2,
                    num_stems: 1,
                    tile_start: 0,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: 1,
                },
                core_inputs: &in2,
                quad_params: &p2,
                initial_checkpoints: None,
            })
            .unwrap();
        assert!(
            cpu_b1_ref
                .feature_outputs
                .iter()
                .flat_map(|v| v.iter())
                .any(|&x| x.abs() > 1.0e-3),
            "Block 1 linear fixed_point reference must be non-zero"
        );
        let fp_b1 = gpu
            .fused_core_scan_256(&FusedCoreScanInput {
                uniforms: FusedScanUniforms {
                    num_frames: num_frames as u32,
                    num_tiles: num_tiles as u32,
                    num_quads: 2,
                    num_stems: 1,
                    tile_start: 0,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: 1 | FUSED_FLAG_FIXED_POINT_SCAN,
                },
                core_inputs: &in2,
                quad_params: &p2,
                initial_checkpoints: None,
            })
            .unwrap();
        assert_vec4_close(
            &cpu_b1_ref.feature_outputs,
            &fp_b1.feature_outputs,
            1.0e-4,
            &format!("fixed_point Block 1 linear ({label})"),
        );
    }
}

#[test]
fn test_wgpu_ultra_wave_cascade_adjoint_resident_streamed_and_arena_reuse() {
    let runners = acquire_variants();
    if runners.is_empty() {
        return;
    }
    let cpu = CpuRunner::detect();
    let num_quads = 10_usize;
    let active_tiles = 35_usize;
    let mut quad_params = Vec::with_capacity(num_quads * QUAD_PARAMS_VEC4_STRIDE);
    for q in 0..num_quads {
        let mut qp = build_quad(0, q, false);
        if matches!(q, 0 | 4 | 6 | 8) {
            qp[quad_param_slots::SVF_K_NL] = [0.15 + 0.05 * (q as f32); 4];
        }
        quad_params.extend_from_slice(&qp);
    }
    let tile_desc = build_wave_descriptors(active_tiles, 0.05);

    // 1. Resident Ultra (300 frames, 10 quads, 35 tiles) CPU parity + GPU central-FD tile weight verification
    let res_frames = 300_usize;
    let res_in = fill_core_inputs(num_quads, res_frames, 0.30);
    let fwd_cpu = cpu
        .run_wave_core_cascade(&WaveCoreCascadeInput {
            num_frames: res_frames,
            num_quads,
            num_stems: 1,
            active_tiles,
            write_checkpoints: true,
            fixed_point_scan: false,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            base_core_inputs: &res_in,
            quad_params: &quad_params,
            tile_descriptors: &tile_desc,
        })
        .unwrap();
    let dlf_res = vec![[0.14_f32, -0.09, 0.07, -0.05]; fwd_cpu.feature_outputs.len()];
    let adj_cpu = cpu
        .run_wave_core_adjoint(&WaveCoreAdjointInput {
            num_frames: res_frames,
            num_quads,
            num_stems: 1,
            active_tiles,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            core_inputs: &fwd_cpu.core_inputs,
            quad_params: &quad_params,
            tile_checkpoints: &fwd_cpu.tile_checkpoints,
            d_loss_d_features: &dlf_res,
            feature_outputs: &fwd_cpu.feature_outputs,
            tile_descriptors: &tile_desc,
        })
        .unwrap();

    // 2. Streamed Ultra (> WAVE_STREAMING_MAX_FRAMES = 16_384, non-multiple of 256)
    let str_frames = WAVE_STREAMING_MAX_FRAMES + 116;
    let str_in = fill_core_inputs(num_quads, str_frames, 0.22);
    let fwd_str_c = cpu
        .run_wave_core_cascade(&WaveCoreCascadeInput {
            num_frames: str_frames,
            num_quads,
            num_stems: 1,
            active_tiles,
            write_checkpoints: true,
            fixed_point_scan: false,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            base_core_inputs: &str_in,
            quad_params: &quad_params,
            tile_descriptors: &tile_desc,
        })
        .unwrap();
    let dlf_str = vec![[0.11_f32, -0.08, 0.05, -0.04]; fwd_str_c.feature_outputs.len()];
    let adj_str_c = cpu
        .run_wave_core_adjoint(&WaveCoreAdjointInput {
            num_frames: str_frames,
            num_quads,
            num_stems: 1,
            active_tiles,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            core_inputs: &fwd_str_c.core_inputs,
            quad_params: &quad_params,
            tile_checkpoints: &fwd_str_c.tile_checkpoints,
            d_loss_d_features: &dlf_str,
            feature_outputs: &fwd_str_c.feature_outputs,
            tile_descriptors: &tile_desc,
        })
        .unwrap();

    for (variant, gpu) in &runners {
        let label = variant.label();
        let fwd_g = gpu
            .run_wave_core_cascade(&WaveCoreCascadeInput {
                num_frames: res_frames,
                num_quads,
                num_stems: 1,
                active_tiles,
                write_checkpoints: true,
                fixed_point_scan: false,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                base_core_inputs: &res_in,
                quad_params: &quad_params,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        assert_vec4_close(
            &fwd_cpu.feature_outputs,
            &fwd_g.feature_outputs,
            2.0e-4,
            &format!("Ultra wave cascade resident ({label})"),
        );

        let adj_g = gpu
            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                num_frames: res_frames,
                num_quads,
                num_stems: 1,
                active_tiles,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                core_inputs: &fwd_g.core_inputs,
                quad_params: &quad_params,
                tile_checkpoints: &fwd_g.tile_checkpoints,
                d_loss_d_features: &dlf_res,
                feature_outputs: &fwd_g.feature_outputs,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        assert_vec4_close(
            &adj_cpu.d_loss_d_inputs,
            &adj_g.d_loss_d_inputs,
            5.0e-4,
            &format!("Ultra wave adjoint du resident ({label})"),
        );

        // GPU central FD check on tile 0 w_odd[0][0]
        let eps = 2.0e-3_f32;
        let eval_gpu_loss = |delta: f32| -> f64 {
            let mut d = tile_desc.clone();
            d[1][0] += delta;
            let o = gpu
                .run_wave_core_cascade(&WaveCoreCascadeInput {
                    num_frames: res_frames,
                    num_quads,
                    num_stems: 1,
                    active_tiles,
                    write_checkpoints: false,
                    fixed_point_scan: false,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    base_core_inputs: &res_in,
                    quad_params: &quad_params,
                    tile_descriptors: &d,
                })
                .unwrap();
            o.feature_outputs
                .iter()
                .zip(&dlf_res)
                .map(|(a, b)| {
                    (0..4)
                        .map(|k| f64::from(a[k]) * f64::from(b[k]))
                        .sum::<f64>()
                })
                .sum()
        };
        let fd = (eval_gpu_loss(eps) - eval_gpu_loss(-eps)) / f64::from(2.0 * eps);
        let got = f64::from(adj_g.tile_weight_grads[0]);
        assert!(
            (got - fd).abs() <= 5.0e-3 * fd.abs().max(1.0),
            "GPU Ultra tile[0].w_odd[0][0] FD ({label}): got={got} fd={fd}"
        );

        // Streamed > 16,384-frame Ultra wave cascade + adjoint
        let fwd_str_g = gpu
            .run_wave_core_cascade(&WaveCoreCascadeInput {
                num_frames: str_frames,
                num_quads,
                num_stems: 1,
                active_tiles,
                write_checkpoints: true,
                fixed_point_scan: false,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                base_core_inputs: &str_in,
                quad_params: &quad_params,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        assert_vec4_close(
            &fwd_str_c.feature_outputs,
            &fwd_str_g.feature_outputs,
            5.0e-4,
            &format!("Ultra wave cascade streamed ({label})"),
        );

        let adj_str_g = gpu
            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                num_frames: str_frames,
                num_quads,
                num_stems: 1,
                active_tiles,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                core_inputs: &fwd_str_g.core_inputs,
                quad_params: &quad_params,
                tile_checkpoints: &fwd_str_g.tile_checkpoints,
                d_loss_d_features: &dlf_str,
                feature_outputs: &fwd_str_g.feature_outputs,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        assert_vec4_close(
            &adj_str_c.d_loss_d_inputs,
            &adj_str_g.d_loss_d_inputs,
            1.0e-3,
            &format!("Ultra wave adjoint du streamed ({label})"),
        );
        assert_vec4_close(
            &adj_str_c.param_grads,
            &adj_str_g.param_grads,
            1.0e-3,
            &format!("Ultra wave adjoint param_grads streamed ({label})"),
        );
        assert_eq!(
            adj_str_c.tile_weight_grads.len(),
            adj_str_g.tile_weight_grads.len()
        );
        for (i, (&exp, &got)) in adj_str_c
            .tile_weight_grads
            .iter()
            .zip(adj_str_g.tile_weight_grads.iter())
            .enumerate()
        {
            let tol = 1.0e-3 * exp.abs().max(1.0);
            assert!(
                (exp - got).abs() <= tol,
                "Ultra wave adjoint tile_weight_grads[{i}] streamed ({label}): exp={exp} got={got}"
            );
        }

        // Arena budget cap & zero-growth reuse invariant after warmup
        let warmed = gpu.arena_allocated_bytes();
        assert!(warmed > 0 && warmed <= DESKTOP_CAPSULE_BUDGET_BYTES);
        let _ = gpu
            .run_wave_core_cascade(&WaveCoreCascadeInput {
                num_frames: res_frames,
                num_quads,
                num_stems: 1,
                active_tiles,
                write_checkpoints: true,
                fixed_point_scan: false,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                base_core_inputs: &res_in,
                quad_params: &quad_params,
                tile_descriptors: &tile_desc,
            })
            .unwrap();
        assert_eq!(
            gpu.arena_allocated_bytes(),
            warmed,
            "GpuTrainArena must allocate 0 extra bytes on smaller/equal re-dispatch ({label})"
        );
    }
}
