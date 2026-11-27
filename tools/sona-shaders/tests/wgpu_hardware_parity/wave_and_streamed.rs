#![forbid(unsafe_code)]

//! Wave-mapped `BLOCK_QUAD_SLOTS` Ultra (`num_quads = 10`) parity, high-persistence
//! `alpha_rel = 0.999` multi-tile adjoint FD, and `Slot 9` cross-block streamed adjoint tests.

use super::core_and_adjoint::eval_cpu_features_loss;
use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, build_core_quad_bundle, cpu_params_from_packed,
    dispatch_adjoint_3pass, dispatch_kernel, read_vec4_vec,
};
use dnam_kernels::{
    CoreQuadV4Block0StateStereo8, GpuVec4, InterleavedStereo8, KernelVec4, SOFT_RECT_EPS,
    SonaBlockStateStereo8, StereoVec8, TINY_CROSSING_EPS, step_core_quad_v4_phs_stereo8,
    step_core_quad_v4_stereo8,
};
use sona_shaders::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE,
    FEATURE_VEC4_STRIDE, FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE,
    TILE_FRAMES, block_quad_count, quad_param_slots, resolve_quad_idx,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

pub fn verify_alpha_rel_0999_multitile_and_streamed_adjoint(
    ctx: &GpuContext,
    variant: ShaderVariant,
) {
    let label = variant.label();
    let num_frames = 512_usize;
    let num_tiles = 2_usize;
    let mut inputs = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
    for t in 0..num_frames {
        let ph = (t as f32) * 0.043;
        inputs[t * 2] = [
            0.40 * ph.sin(),
            0.35 * (1.2 * ph).cos(),
            0.28 * (0.8 * ph).sin(),
            0.24 * (1.5 * ph).cos(),
        ];
        inputs[t * 2 + 1] = [
            0.33 * (0.9 * ph).cos(),
            0.38 * (1.1 * ph).sin(),
            0.25 * (1.3 * ph).cos(),
            0.20 * (0.7 * ph).sin(),
        ];
    }

    let (_, mut packed) = build_core_quad_bundle(0, 0);
    packed[quad_param_slots::LRU_ALPHA_REL] = [0.999; 4];
    packed[quad_param_slots::LRU_ALPHA_ATK] = [0.85; 4];
    packed[quad_param_slots::LRU_KAPPA] = [0.55; 4];
    packed[quad_param_slots::LRU_GAMMA_DUHEM] = [0.40; 4];
    packed[quad_param_slots::LRU_STEREO_LINK] = [0.25; 4];
    packed[quad_param_slots::D_PRE] = [2.5; 4];
    let cpu_p = cpu_params_from_packed(&packed);

    let ckpt_init = vec![[0.0_f32; 4]; num_tiles * TILE_CKPT_VEC4_STRIDE];
    let feat_init = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
    let fwd_u = FusedScanUniforms {
        num_frames: num_frames as u32,
        num_tiles: num_tiles as u32,
        num_quads: 1,
        num_stems: 1,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx: BLOCK_IDX_DIRECT_LINEAR,
    };
    let (fwd_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        fwd_u.as_bytes(),
        &[
            (false, as_byte_slice(&inputs)),
            (false, as_byte_slice(&packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let ckpts = read_vec4_vec(&fwd_out[0]);
    let feats = read_vec4_vec(&fwd_out[1]);

    let mut d_loss = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
    for (idx, item) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.011;
        *item = [
            0.22 * ph.sin(),
            0.19 * (1.3 * ph).cos(),
            -0.17 * (0.7 * ph).sin(),
            0.21 * (1.7 * ph).cos(),
        ];
    }

    let res_adj = dispatch_adjoint_3pass(
        ctx,
        variant,
        num_frames,
        num_tiles,
        1,
        1,
        1,
        BLOCK_IDX_DIRECT_LINEAR,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs),
        as_byte_slice(&packed),
        as_byte_slice(&ckpts),
        as_byte_slice(&d_loss),
        as_byte_slice(&feats),
        None,
    );
    let got_du_res = res_adj.du;
    let got_pg_res = res_adj.param_grads;

    // Central FD parity on alpha_rel = 0.999 across seam t in {254, 255, 256, 257}.
    let eps = 2.0e-3_f32;
    let params_slice = [cpu_p];
    for &t in &[0_usize, 17, 254, 255, 256, 257, 511] {
        let idx = t * CORE_INPUT_VEC4_STRIDE;
        for lane in [0_usize, 2] {
            let mut pp = inputs.clone();
            let mut pm = inputs.clone();
            pp[idx][lane] += eps;
            pm[idx][lane] -= eps;
            let fd = (eval_cpu_features_loss(&pp, &params_slice, &d_loss, 0, num_frames)
                - eval_cpu_features_loss(&pm, &params_slice, &d_loss, 0, num_frames))
                / ((pp[idx][lane] - pm[idx][lane]) as f64);
            let got = got_du_res[idx][lane] as f64;
            let tol = 1.0e-3 * fd.abs().max(1.0);
            assert!(
                (got - fd).abs() <= tol,
                "alpha_rel=0.999 FD mismatch ({label}) t={t} lane={lane}: got={got} fd={fd}"
            );
        }
    }

    // 2-chunk cross-block streamed adjoint: Chunk 1 [256..512] -> Slot 7 header out_entry[8..16] -> Chunk 0 [0..256].
    let c1 = dispatch_adjoint_3pass(
        ctx,
        variant,
        TILE_FRAMES,
        1,
        1,
        1,
        1,
        BLOCK_IDX_DIRECT_LINEAR,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs[TILE_FRAMES * 2..]),
        as_byte_slice(&packed),
        as_byte_slice(&ckpts[TILE_CKPT_VEC4_STRIDE..]),
        as_byte_slice(&d_loss[TILE_FRAMES * FEATURE_VEC4_STRIDE..]),
        as_byte_slice(&feats[TILE_FRAMES * FEATURE_VEC4_STRIDE..]),
        None,
    );
    let mut seed_c0 = vec![[0.0_f32; 4]; EXIT_ADJ_BOUNDARY_VEC4_STRIDE];
    seed_c0[0..12].copy_from_slice(&c1.exit_boundary[12..24]);
    let c0 = dispatch_adjoint_3pass(
        ctx,
        variant,
        TILE_FRAMES,
        1,
        1,
        1,
        1,
        BLOCK_IDX_DIRECT_LINEAR,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs[..TILE_FRAMES * 2]),
        as_byte_slice(&packed),
        as_byte_slice(&ckpts[..TILE_CKPT_VEC4_STRIDE]),
        as_byte_slice(&d_loss[..TILE_FRAMES * FEATURE_VEC4_STRIDE]),
        as_byte_slice(&feats[..TILE_FRAMES * FEATURE_VEC4_STRIDE]),
        Some(as_byte_slice(&seed_c0)),
    );

    let mut du_streamed = Vec::with_capacity(c0.du.len() + c1.du.len());
    du_streamed.extend_from_slice(&c0.du);
    du_streamed.extend_from_slice(&c1.du);
    for (idx, (&a, &b)) in got_du_res.iter().zip(&du_streamed).enumerate() {
        for lane in 0..4 {
            let tol = 1.0e-5 * a[lane].abs().max(b[lane].abs()).max(1.0);
            assert!(
                (a[lane] - b[lane]).abs() <= tol,
                "cross-block streamed du mismatch ({label}) idx={idx} lane={lane}: res={} stream={}",
                a[lane],
                b[lane]
            );
        }
    }
    let pg_streamed: Vec<[f32; 4]> = c0.param_grads.into_iter().chain(c1.param_grads).collect();
    for (slot, (&res_v, &str_v)) in got_pg_res.iter().zip(&pg_streamed).enumerate() {
        for lane in 0..4 {
            let tol = 1.0e-4 * res_v[lane].abs().max(1.0);
            assert!(
                (res_v[lane] - str_v[lane]).abs() <= tol,
                "cross-block streamed param_grad mismatch ({label}) slot={slot} lane={lane}: res={} stream={}",
                res_v[lane],
                str_v[lane]
            );
        }
    }
}

pub fn verify_canonical_4wave_ultra_hierarchy_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let num_frames = 300_usize;
    let num_tiles = 2_usize;
    let num_quads = 10_usize;
    let num_stems = 1_usize;

    let mut core_inputs = vec![[0.0_f32; 4]; num_quads * num_frames * CORE_INPUT_VEC4_STRIDE];
    let mut quad_params_packed = Vec::with_capacity(num_quads * QUAD_PARAMS_VEC4_STRIDE);
    let mut quad_params_cpu = Vec::with_capacity(num_quads);
    for q in 0..num_quads {
        for t in 0..num_frames {
            let ph = (t as f32) * 0.067 + (q as f32) * 0.29;
            let base = (q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
            core_inputs[base] = [
                0.4 * ph.sin(),
                0.35 * (1.2 * ph).cos(),
                0.3 * (0.7 * ph).sin(),
                0.25 * (1.5 * ph).cos(),
            ];
            core_inputs[base + 1] = [
                0.36 * (0.9 * ph).cos(),
                0.31 * (1.3 * ph).sin(),
                0.27 * (0.8 * ph).cos(),
                0.22 * (1.4 * ph).sin(),
            ];
        }
        let (_, mut packed) = build_core_quad_bundle(0, q);
        if matches!(q, 0 | 4 | 8) {
            packed[quad_param_slots::SVF_K_NL] = [0.25 + 0.1 * (q as f32); 4];
        }
        quad_params_cpu.push(cpu_params_from_packed(&packed));
        quad_params_packed.extend_from_slice(&packed);
    }

    let mut gpu_ckpts = vec![[0.0_f32; 4]; num_quads * num_tiles * TILE_CKPT_VEC4_STRIDE];
    let mut gpu_feats = vec![[0.0_f32; 4]; num_quads * num_frames * FEATURE_VEC4_STRIDE];
    for block_idx in 0..4_u32 {
        let wg_x = block_quad_count(block_idx as usize, num_quads);
        let u = FusedScanUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_quads: num_quads as u32,
            num_stems: num_stems as u32,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx,
        };
        let (out, _) = dispatch_kernel(
            ctx,
            KernelId::FusedCoreScan256,
            variant,
            (wg_x as u32, 1, 1),
            u.as_bytes(),
            &[
                (false, as_byte_slice(&core_inputs)),
                (false, as_byte_slice(&quad_params_packed)),
                (true, as_byte_slice(&gpu_ckpts)),
                (true, as_byte_slice(&gpu_feats)),
            ],
        );
        gpu_ckpts = read_vec4_vec(&out[0]);
        gpu_feats = read_vec4_vec(&out[1]);
    }

    // Verify all 10 quads match CPU reference step_core_quad_v4[_phs]_stereo8 ([3, 3, 2, 2]).
    for q in 0..num_quads {
        let is_phs = matches!(q, 0 | 4 | 8);
        let mut st_phs = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        for t in 0..num_frames {
            let in_b = (q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
            let u = InterleavedStereo8::from_lr(
                GpuVec4::from_array(core_inputs[in_b]),
                GpuVec4::from_array(core_inputs[in_b + 1]),
            );
            let out = if is_phs {
                step_core_quad_v4_phs_stereo8(u, &mut st_phs, &quad_params_cpu[q])
            } else {
                step_core_quad_v4_stereo8(u, &mut st, &quad_params_cpu[q])
            };
            let (vl, vr) = out.lru_state.into_lr();
            let f_b = (q * num_frames + t) * FEATURE_VEC4_STRIDE;
            for l in 0..4 {
                assert!(
                    (gpu_feats[f_b][l] - vl.to_array()[l]).abs() < 1.0e-4,
                    "4-wave Ultra ({label}) q={q} t={t} L mismatch"
                );
                assert!(
                    (gpu_feats[f_b + 1][l] - vr.to_array()[l]).abs() < 1.0e-4,
                    "4-wave Ultra ({label}) q={q} t={t} R mismatch"
                );
            }
        }
    }

    let mut d_loss = vec![[0.0_f32; 4]; gpu_feats.len()];
    for (idx, w) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.013;
        *w = [
            0.2 * ph.sin(),
            0.15 * ph.cos(),
            -0.18 * ph.sin(),
            0.21 * ph.cos(),
        ];
    }
    let mut got_du = vec![[0.0_f32; 4]; core_inputs.len()];
    for block_idx in (0..4_u32).rev() {
        let wg_x = block_quad_count(block_idx as usize, num_quads);
        let du_b = dispatch_adjoint_3pass(
            ctx,
            variant,
            num_frames,
            num_tiles,
            num_quads,
            num_stems,
            wg_x,
            block_idx,
            SOFT_RECT_EPS,
            TINY_CROSSING_EPS,
            as_byte_slice(&core_inputs),
            as_byte_slice(&quad_params_packed),
            as_byte_slice(&gpu_ckpts),
            as_byte_slice(&d_loss),
            as_byte_slice(&gpu_feats),
            None,
        )
        .du;
        for q_local in 0..wg_x {
            let q = resolve_quad_idx(block_idx, q_local, num_quads);
            assert!(q < num_quads);
            let in_s = q * num_frames * CORE_INPUT_VEC4_STRIDE;
            let in_e = in_s + num_frames * CORE_INPUT_VEC4_STRIDE;
            got_du[in_s..in_e].copy_from_slice(&du_b[in_s..in_e]);
        }
    }

    let eps = 3.0e-3_f32;
    for q in [0_usize, 5, 7, 3] {
        let is_phs = q == 0;
        let idx = (q * num_frames + 255) * CORE_INPUT_VEC4_STRIDE;
        let eval_q = |ci: &[[f32; 4]]| -> f64 {
            if is_phs {
                let mut st = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
                let mut loss = 0.0_f64;
                for t in 0..num_frames {
                    let ib = (q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                    let u = InterleavedStereo8::from_lr(
                        GpuVec4::from_array(ci[ib]),
                        GpuVec4::from_array(ci[ib + 1]),
                    );
                    let out = step_core_quad_v4_phs_stereo8(u, &mut st, &quad_params_cpu[q]);
                    let (vl, vr) = out.lru_state.into_lr();
                    let (ol, or) = out.o_tilde.into_lr();
                    let (el, er) = out.e_tilde.into_lr();
                    let arrs = [
                        vl.to_array(),
                        vr.to_array(),
                        ol.to_array(),
                        or.to_array(),
                        el.to_array(),
                        er.to_array(),
                    ];
                    let wb = (q * num_frames + t) * FEATURE_VEC4_STRIDE;
                    for (fi, a) in arrs.iter().enumerate() {
                        for l in 0..4 {
                            loss += (a[l] as f64) * (d_loss[wb + fi][l] as f64);
                        }
                    }
                }
                loss
            } else {
                eval_cpu_features_loss(ci, &quad_params_cpu, &d_loss, q, num_frames)
            }
        };
        let mut pp = core_inputs.clone();
        let mut pm = core_inputs.clone();
        pp[idx][0] += eps;
        pm[idx][0] -= eps;
        let fd = (eval_q(&pp) - eval_q(&pm)) / ((pp[idx][0] - pm[idx][0]) as f64);
        let g = got_du[idx][0] as f64;
        assert!(
            (g - fd).abs() <= 4.0e-3 * fd.abs().max(1.0),
            "4-wave Ultra ({label}) adjoint FD mismatch q={q}: got={g} fd={fd}"
        );
    }
}
