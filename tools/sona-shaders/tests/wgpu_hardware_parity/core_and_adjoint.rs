#![forbid(unsafe_code)]

//! Hardware parity and Zero-DC tests for `fused_core_scan_256` and exact 3-pass multi-tile
//! reverse-BPTT finite-difference checks for `adjoint_scan_256`.

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, build_core_quad_bundle, cpu_params_from_packed,
    dispatch_adjoint_3pass, dispatch_kernel, read_vec4_vec,
};
use dnam_kernels::{
    CoreQuadV4ParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, SOFT_RECT_EPS,
    SonaBlockStateStereo8, StereoVec8, TINY_CROSSING_EPS, step_core_quad_v4_stereo8,
};
use sona_shaders::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
    QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, quad_param_slots,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

const NUM_FRAMES: usize = 600;
const NUM_TILES: usize = 3;
const NUM_QUADS: usize = 2;
const NUM_STEMS: usize = 2;
const TOTAL_Q: usize = NUM_QUADS * NUM_STEMS;

fn make_core_inputs(num_frames: usize, zero_mode: bool) -> Vec<[f32; 4]> {
    let mut core_inputs = vec![[0.0_f32; 4]; TOTAL_Q * num_frames * CORE_INPUT_VEC4_STRIDE];
    for s in 0..NUM_STEMS {
        for q in 0..NUM_QUADS {
            let q_global = s * NUM_QUADS + q;
            for t in 0..num_frames {
                let base = (q_global * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                if zero_mode {
                    let z = if (t & 1) == 0 { 0.0_f32 } else { -0.0_f32 };
                    core_inputs[base] = [z; 4];
                    core_inputs[base + 1] = [z; 4];
                } else {
                    let ph = (t as f32) * 0.097 + (s as f32) * 0.63 + (q as f32) * 0.31;
                    core_inputs[base] = [
                        0.45 * ph.sin() + 0.15 * (2.3 * ph).cos(),
                        0.38 * (1.4 * ph).cos(),
                        0.52 * (0.7 * ph).sin() - 0.10 * (3.1 * ph).sin(),
                        0.30 * (1.9 * ph).cos() + 0.12,
                    ];
                    core_inputs[base + 1] = [
                        0.42 * (1.1 * ph).cos(),
                        0.50 * (0.85 * ph).sin() + 0.12,
                        0.35 * (2.1 * ph).cos() - 0.14 * ph.sin(),
                        0.48 * (1.3 * ph).sin(),
                    ];
                }
            }
        }
    }
    core_inputs
}

pub(super) fn eval_cpu_features_loss(
    core_inputs: &[[f32; 4]],
    params_cpu: &[CoreQuadV4ParamsStereo4<GpuVec4>],
    weights: &[[f32; 4]],
    q_global: usize,
    num_frames: usize,
) -> f64 {
    let mut state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut loss = 0.0_f64;
    for t in 0..num_frames {
        let in_base = (q_global * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(core_inputs[in_base]),
            GpuVec4::from_array(core_inputs[in_base + 1]),
        );
        let out = step_core_quad_v4_stereo8(u, &mut state, &params_cpu[q_global]);
        let (vl, vr) = out.lru_state.into_lr();
        let (ol, or) = out.o_tilde.into_lr();
        let (el, er) = out.e_tilde.into_lr();
        let feats = [
            vl.to_array(),
            vr.to_array(),
            ol.to_array(),
            or.to_array(),
            el.to_array(),
            er.to_array(),
        ];
        let w_base = (q_global * num_frames + t) * FEATURE_VEC4_STRIDE;
        for f_idx in 0..6 {
            for lane in 0..4 {
                loss += (feats[f_idx][lane] as f64) * (weights[w_base + f_idx][lane] as f64);
            }
        }
    }
    loss
}

pub fn verify_fused_core_scan_256_parity_and_zero_dc(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let core_inputs = make_core_inputs(NUM_FRAMES, false);
    let mut quad_params_packed = Vec::with_capacity(TOTAL_Q * QUAD_PARAMS_VEC4_STRIDE);
    let mut quad_params_cpu = Vec::with_capacity(TOTAL_Q);
    for s in 0..NUM_STEMS {
        for q in 0..NUM_QUADS {
            let (cpu_p, gpu_p) = build_core_quad_bundle(s, q);
            quad_params_cpu.push(cpu_p);
            quad_params_packed.extend_from_slice(&gpu_p);
        }
    }

    let ckpt_init = vec![[0.0_f32; 4]; TOTAL_Q * NUM_TILES * TILE_CKPT_VEC4_STRIDE];
    let feat_init = vec![[0.0_f32; 4]; TOTAL_Q * NUM_FRAMES * FEATURE_VEC4_STRIDE];
    let uniforms = FusedScanUniforms {
        num_frames: NUM_FRAMES as u32,
        num_tiles: NUM_TILES as u32,
        num_quads: NUM_QUADS as u32,
        num_stems: NUM_STEMS as u32,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx: BLOCK_IDX_DIRECT_LINEAR,
    };

    let (out, _elapsed) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (NUM_QUADS as u32, 1, NUM_STEMS as u32),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let gpu_ckpts = read_vec4_vec(&out[0]);
    let gpu_feats = read_vec4_vec(&out[1]);

    for q_global in 0..TOTAL_Q {
        let mut state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut cpu_feats = Vec::with_capacity(NUM_FRAMES * FEATURE_VEC4_STRIDE);
        let mut cpu_ckpts = Vec::with_capacity(NUM_TILES * TILE_CKPT_VEC4_STRIDE);
        let mut peak = 1.0_f32;

        for t in 0..NUM_FRAMES {
            if t % 256 == 0 {
                let (s1_l, s1_r) = state.svf_s1.into_lr();
                let (s2_l, s2_r) = state.svf_s2.into_lr();
                let (v_l, v_r) = state.lru_v.into_lr();
                let (ap_l, ap_r) = state.adaa_prev.into_lr();
                cpu_ckpts.push(s1_l.to_array());
                cpu_ckpts.push(s1_r.to_array());
                cpu_ckpts.push(s2_l.to_array());
                cpu_ckpts.push(s2_r.to_array());
                cpu_ckpts.push(v_l.to_array());
                cpu_ckpts.push(v_r.to_array());
                cpu_ckpts.push(ap_l.to_array());
                cpu_ckpts.push(ap_r.to_array());
                cpu_ckpts.push([0.0; 4]);
                cpu_ckpts.push([0.0; 4]);
                cpu_ckpts.push([0.0; 4]);
                cpu_ckpts.push([0.0; 4]);
            }
            let in_base = (q_global * NUM_FRAMES + t) * CORE_INPUT_VEC4_STRIDE;
            let u = InterleavedStereo8::from_lr(
                GpuVec4::from_array(core_inputs[in_base]),
                GpuVec4::from_array(core_inputs[in_base + 1]),
            );
            let out = step_core_quad_v4_stereo8(u, &mut state, &quad_params_cpu[q_global]);
            let (vl, vr) = out.lru_state.into_lr();
            let (ol, or) = out.o_tilde.into_lr();
            let (el, er) = out.e_tilde.into_lr();
            for arr in [
                vl.to_array(),
                vr.to_array(),
                ol.to_array(),
                or.to_array(),
                el.to_array(),
                er.to_array(),
            ] {
                for x in arr {
                    peak = peak.max(x.abs());
                }
                cpu_feats.push(arr);
            }
        }

        let tol = 1.0e-4 * peak;
        for t in 0..NUM_FRAMES {
            for slot in 0..FEATURE_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
                let cpu = cpu_feats[t * FEATURE_VEC4_STRIDE + slot];
                let gpu = gpu_feats[idx];
                for lane in 0..4 {
                    let diff = (gpu[lane] - cpu[lane]).abs();
                    assert!(
                        diff <= tol,
                        "fused_core_scan_256 ({label}) feature mismatch q={q_global} t={t} slot={slot} lane={lane}: \
                         gpu={} cpu={} diff={diff} tol={tol}",
                        gpu[lane],
                        cpu[lane]
                    );
                }
            }
        }
        for tile in 0..NUM_TILES {
            for c_slot in 0..TILE_CKPT_VEC4_STRIDE {
                let idx = (q_global * NUM_TILES + tile) * TILE_CKPT_VEC4_STRIDE + c_slot;
                let cpu = cpu_ckpts[tile * TILE_CKPT_VEC4_STRIDE + c_slot];
                let gpu = gpu_ckpts[idx];
                for lane in 0..4 {
                    let diff = (gpu[lane] - cpu[lane]).abs();
                    assert!(
                        diff <= tol,
                        "fused_core_scan_256 ({label}) ckpt mismatch q={q_global} tile={tile} slot={c_slot} lane={lane}: \
                         gpu={} cpu={} diff={diff}",
                        gpu[lane],
                        cpu[lane]
                    );
                }
            }
        }
    }

    // Exact +0.0 Zero-DC invariance check across pure -0.0 zero inputs & negative weights.
    let neg_zero_inputs = vec![[-0.0_f32; 4]; core_inputs.len()];
    let mut neg_weight_params = quad_params_packed.clone();
    neg_weight_params[quad_param_slots::TONE_C_LP] = [-0.8, -0.5, -0.4, -0.3];
    neg_weight_params[quad_param_slots::TONE_C_BP] = [-0.4, -0.3, -0.2, -0.1];
    neg_weight_params[quad_param_slots::ADAA_DRIVE] = [-1.5, -1.2, -1.1, -1.3];
    neg_weight_params[quad_param_slots::LRU_BIAS_DRIVE] = [-0.6, -0.4, -0.5, -0.3];
    neg_weight_params[quad_param_slots::GLU_MIX] = [-0.7, -0.5, -0.8, -0.6];

    for zero_in in [make_core_inputs(NUM_FRAMES, true), neg_zero_inputs] {
        let (z_out, _elapsed) = dispatch_kernel(
            ctx,
            KernelId::FusedCoreScan256,
            variant,
            (NUM_QUADS as u32, 1, NUM_STEMS as u32),
            uniforms.as_bytes(),
            &[
                (false, as_byte_slice(&zero_in)),
                (false, as_byte_slice(&neg_weight_params)),
                (true, as_byte_slice(&ckpt_init)),
                (true, as_byte_slice(&feat_init)),
            ],
        );
        for (buf_idx, bytes) in z_out.iter().enumerate() {
            for (i, v) in read_vec4_vec(bytes).iter().enumerate() {
                for (lane, &x) in v.iter().enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        0,
                        "Zero-DC ({label}) violation buf={buf_idx} vec={i} lane={lane}: {:#010x}",
                        x.to_bits()
                    );
                }
            }
        }
    }

    // Resume invariance check: tile_start = 1 with pre-seeded tile_checkpoints matches full scan bit-for-bit.
    let mut resume_u = uniforms;
    resume_u.tile_start = 1;
    let (res_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (NUM_QUADS as u32, 1, NUM_STEMS as u32),
        resume_u.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&gpu_ckpts)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let res_ckpts = read_vec4_vec(&res_out[0]);
    let res_feats = read_vec4_vec(&res_out[1]);
    for q_global in 0..TOTAL_Q {
        for tile in 1..NUM_TILES {
            for slot in 0..TILE_CKPT_VEC4_STRIDE {
                let idx = (q_global * NUM_TILES + tile) * TILE_CKPT_VEC4_STRIDE + slot;
                assert_eq!(
                    res_ckpts[idx].map(f32::to_bits),
                    gpu_ckpts[idx].map(f32::to_bits),
                    "tile_start=1 checkpoint resume mismatch ({label}) q={q_global} tile={tile} slot={slot}"
                );
            }
        }
        for t in 256..NUM_FRAMES {
            for slot in 0..FEATURE_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
                assert_eq!(
                    res_feats[idx].map(f32::to_bits),
                    gpu_feats[idx].map(f32::to_bits),
                    "tile_start=1 feature resume mismatch ({label}) q={q_global} t={t} slot={slot}"
                );
            }
        }
    }
}

fn cpu_params_with_raw_slot(
    packed: &[[f32; 4]],
    slot: usize,
    lane: usize,
    val: f32,
) -> CoreQuadV4ParamsStereo4<GpuVec4> {
    let mut patched = packed.to_vec();
    patched[slot][lane] = val;
    let mut params = cpu_params_from_packed(&patched);
    let override_vec = GpuVec4::from_array(patched[slot]);
    match slot {
        quad_param_slots::SVF_A1 => params.svf.a1 = override_vec,
        quad_param_slots::SVF_A2 => params.svf.a2 = override_vec,
        quad_param_slots::SVF_A3 => params.svf.a3 = override_vec,
        _ => {}
    }
    params
}

fn verify_adjoint_pass1_multiblock_fd(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let num_frames = 260 * 256;
    let num_tiles = 260_usize;
    let mut inputs = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
    for t in 0..num_frames {
        let ph = (t as f32) * 0.029;
        inputs[t * CORE_INPUT_VEC4_STRIDE] = [
            0.35 * ph.sin(),
            0.30 * (1.2 * ph).cos(),
            0.25 * (0.7 * ph).sin(),
            0.20 * (1.5 * ph).cos(),
        ];
        inputs[t * CORE_INPUT_VEC4_STRIDE + 1] = [
            0.32 * (0.9 * ph).cos(),
            0.28 * (1.1 * ph).sin(),
            0.22 * (1.4 * ph).cos(),
            0.18 * (0.8 * ph).sin(),
        ];
    }
    let (cpu_p, packed_p) = build_core_quad_bundle(0, 0);
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
            (false, as_byte_slice(&packed_p)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );

    let mut d_loss = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
    for t in 250 * 256..num_frames {
        for slot in 0..FEATURE_VEC4_STRIDE {
            let ph = ((t * FEATURE_VEC4_STRIDE + slot) as f32) * 0.013;
            d_loss[t * FEATURE_VEC4_STRIDE + slot] = [
                0.25 * ph.sin(),
                0.22 * (1.3 * ph).cos(),
                0.20 * (0.7 * ph).sin(),
                0.18 * (1.7 * ph).cos(),
            ];
        }
    }

    let got_du = dispatch_adjoint_3pass(
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
        as_byte_slice(&packed_p),
        &fwd_out[0],
        as_byte_slice(&d_loss),
        &fwd_out[1],
        None,
    )
    .du;
    let params_slice = [cpu_p];
    let eps = 3.0e-3_f32;
    for &t in &[
        17_usize,
        255 * 256 - 1,
        256 * 256,
        256 * 256 + 17,
        259 * 256 + 200,
    ] {
        let idx = t * CORE_INPUT_VEC4_STRIDE;
        let mut p_plus = inputs.clone();
        let mut p_minus = inputs.clone();
        p_plus[idx][0] += eps;
        p_minus[idx][0] -= eps;
        let delta = (p_plus[idx][0] - p_minus[idx][0]) as f64;
        let lp = eval_cpu_features_loss(&p_plus, &params_slice, &d_loss, 0, num_frames);
        let lm = eval_cpu_features_loss(&p_minus, &params_slice, &d_loss, 0, num_frames);
        let fd = (lp - lm) / delta;
        let g = got_du[idx][0] as f64;
        assert!(
            (g - fd).abs() <= 5.0e-3 * fd.abs().max(1.0),
            "adjoint pass1 multiblock ({label}) FD mismatch t={t}: got={g} fd={fd}"
        );
    }
}

pub fn verify_adjoint_scan_256_fd_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let core_inputs = make_core_inputs(NUM_FRAMES, false);
    let mut quad_params_packed = Vec::with_capacity(TOTAL_Q * QUAD_PARAMS_VEC4_STRIDE);
    let mut quad_params_cpu = Vec::with_capacity(TOTAL_Q);
    for s in 0..NUM_STEMS {
        for q in 0..NUM_QUADS {
            let (cpu_p, gpu_p) = build_core_quad_bundle(s, q);
            quad_params_cpu.push(cpu_p);
            quad_params_packed.extend_from_slice(&gpu_p);
        }
    }

    let ckpt_init = vec![[0.0_f32; 4]; TOTAL_Q * NUM_TILES * TILE_CKPT_VEC4_STRIDE];
    let feat_init = vec![[0.0_f32; 4]; TOTAL_Q * NUM_FRAMES * FEATURE_VEC4_STRIDE];
    let fwd_uniforms = FusedScanUniforms {
        num_frames: NUM_FRAMES as u32,
        num_tiles: NUM_TILES as u32,
        num_quads: NUM_QUADS as u32,
        num_stems: NUM_STEMS as u32,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx: BLOCK_IDX_DIRECT_LINEAR,
    };
    let (fwd_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (NUM_QUADS as u32, 1, NUM_STEMS as u32),
        fwd_uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let tile_checkpoints = &fwd_out[0];
    let feature_outputs = &fwd_out[1];

    let mut d_loss_d_features = vec![[0.0_f32; 4]; TOTAL_Q * NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for q_global in 0..TOTAL_Q {
        for t in 0..NUM_FRAMES {
            for slot in 0..FEATURE_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
                let base = (idx as f32) * 0.017;
                d_loss_d_features[idx] = [
                    0.35 * base.sin(),
                    0.30 * (1.3 * base).cos(),
                    0.28 * (0.7 * base).sin(),
                    0.32 * (1.9 * base).cos(),
                ];
            }
        }
    }

    let adj_res = dispatch_adjoint_3pass(
        ctx,
        variant,
        NUM_FRAMES,
        NUM_TILES,
        NUM_QUADS,
        NUM_STEMS,
        NUM_QUADS,
        BLOCK_IDX_DIRECT_LINEAR,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&core_inputs),
        as_byte_slice(&quad_params_packed),
        tile_checkpoints,
        as_byte_slice(&d_loss_d_features),
        feature_outputs,
        None,
    );
    let got_du = adj_res.du;
    let got_pgrad = adj_res.param_grads;

    // Check input sensitivities at interior frames, cross-tile seams (255<->256, 511<->512), & tail end (599).
    let eps_u = 3.0e-3_f32;
    let mut worst_input_rel = 0.0_f64;
    let mut input_failures = Vec::new();
    for q_global in [0_usize, 1, 3] {
        for &t in &[0_usize, 17, 254, 255, 256, 257, 510, 511, 512, 598, 599] {
            for ch in 0..CORE_INPUT_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * CORE_INPUT_VEC4_STRIDE + ch;
                for lane in [0_usize, 2] {
                    let mut p_plus = core_inputs.clone();
                    let mut p_minus = core_inputs.clone();
                    p_plus[idx][lane] += eps_u;
                    p_minus[idx][lane] -= eps_u;
                    let delta = (p_plus[idx][lane] - p_minus[idx][lane]) as f64;
                    let lp = eval_cpu_features_loss(
                        &p_plus,
                        &quad_params_cpu,
                        &d_loss_d_features,
                        q_global,
                        NUM_FRAMES,
                    );
                    let lm = eval_cpu_features_loss(
                        &p_minus,
                        &quad_params_cpu,
                        &d_loss_d_features,
                        q_global,
                        NUM_FRAMES,
                    );
                    let fd = (lp - lm) / delta;
                    let g = got_du[idx][lane] as f64;
                    let rel = (g - fd).abs() / fd.abs().max(1.0);
                    worst_input_rel = worst_input_rel.max(rel);
                    let tol = 3.0e-3 * fd.abs().max(1.0);
                    if (g - fd).abs() > tol {
                        input_failures.push(format!(
                            "q={q_global} t={t} ch={ch} lane={lane}: got={g} fd={fd} rel={rel}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        input_failures.is_empty(),
        "adjoint ({label}) input FD mismatches (worst_rel={worst_input_rel:.6e}): {input_failures:?}"
    );

    // Check summed parameter gradients across all tiles against CPU FD on differentiable parameters.
    let diff_slots: &[(usize, f32)] = &[
        (quad_param_slots::SVF_A1, 1.0e-4),
        (quad_param_slots::SVF_A2, 1.0e-4),
        (quad_param_slots::SVF_A3, 1.0e-4),
        (quad_param_slots::SVF_K, 2.0e-3),
        (quad_param_slots::SVF_G, 1.0e-4),
        (quad_param_slots::TONE_C_LP, 2.5e-4),
        (quad_param_slots::TONE_C_BP, 2.5e-4),
        (quad_param_slots::TONE_C_KBP, 2.5e-4),
        (quad_param_slots::TONE_C_U, 2.5e-4),
        (quad_param_slots::LRU_ALPHA_REL, 2.0e-4),
        (quad_param_slots::LRU_ALPHA_ATK, 2.0e-4),
        (quad_param_slots::LRU_KAPPA, 2.0e-4),
        (quad_param_slots::LRU_GAMMA_DUHEM, 2.0e-4),
        (quad_param_slots::LRU_SIGNED_MIX, 2.0e-4),
        (quad_param_slots::LRU_STEREO_LINK, 5.0e-4),
        (quad_param_slots::G0, 5.0e-4),
        (quad_param_slots::G_INF, 5.0e-4),
        (quad_param_slots::D_PRE, 5.0e-4),
        (quad_param_slots::SE_W_SQ, 5.0e-4),
        (quad_param_slots::SE_D_BUS, 5.0e-4),
        (quad_param_slots::ADAA_DRIVE, 5.0e-4),
        (quad_param_slots::LRU_BIAS_DRIVE, 5.0e-4),
        (quad_param_slots::LIN_MIX, 1.0e-4),
        (quad_param_slots::WAVEFOLDER_MIX, 1.0e-4),
        (quad_param_slots::GLU_MIX, 5.0e-4),
        (quad_param_slots::OUT_GAIN, 5.0e-4),
    ];

    for q_global in [0_usize, 3] {
        let mut cpu_packed_base = vec![[0.0_f32; 4]; QUAD_PARAMS_VEC4_STRIDE];
        cpu_packed_base.copy_from_slice(
            &quad_params_packed
                [q_global * QUAD_PARAMS_VEC4_STRIDE..(q_global + 1) * QUAD_PARAMS_VEC4_STRIDE],
        );

        for &(slot, eps) in diff_slots {
            for lane in [0_usize, 1, 2, 3] {
                let mut g_gpu = 0.0_f64;
                for tile in 0..NUM_TILES {
                    let p_idx = (q_global * NUM_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE + slot;
                    g_gpu += got_pgrad[p_idx][lane] as f64;
                }

                let base_val = cpu_packed_base[slot][lane];
                let (val_plus, val_minus) = if (slot == quad_param_slots::WAVEFOLDER_MIX
                    || slot == quad_param_slots::LIN_MIX)
                    && lane == 0
                {
                    (base_val + eps, base_val)
                } else if slot == quad_param_slots::WAVEFOLDER_MIX && lane == 3 {
                    (base_val, base_val - eps)
                } else {
                    (base_val + eps, base_val - eps)
                };
                let delta = (val_plus - val_minus) as f64;
                let mut p_cpu = quad_params_cpu.clone();
                p_cpu[q_global] = cpu_params_with_raw_slot(&cpu_packed_base, slot, lane, val_plus);
                let lp = eval_cpu_features_loss(
                    &core_inputs,
                    &p_cpu,
                    &d_loss_d_features,
                    q_global,
                    NUM_FRAMES,
                );
                p_cpu[q_global] = cpu_params_with_raw_slot(&cpu_packed_base, slot, lane, val_minus);
                let lm = eval_cpu_features_loss(
                    &core_inputs,
                    &p_cpu,
                    &d_loss_d_features,
                    q_global,
                    NUM_FRAMES,
                );
                let fd = (lp - lm) / delta;
                let tol = 5.0e-3 * fd.abs().max(1.0);
                assert!(
                    (g_gpu - fd).abs() <= tol,
                    "adjoint ({label}) param FD mismatch q={q_global} slot={slot} lane={lane}: \
                     got={g_gpu} fd={fd} tol={tol}"
                );
            }
        }
        for tile in 0..NUM_TILES {
            let p_idx = (q_global * NUM_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE
                + quad_param_slots::SVF_K_NL;
            assert_eq!(got_pgrad[p_idx].map(f32::to_bits), [0; 4]);
        }
    }

    verify_adjoint_pass1_multiblock_fd(ctx, variant);
    super::wave_and_streamed::verify_alpha_rel_0999_multitile_and_streamed_adjoint(ctx, variant);
    super::wave_and_streamed::verify_canonical_4wave_ultra_hierarchy_parity(ctx, variant);
}
