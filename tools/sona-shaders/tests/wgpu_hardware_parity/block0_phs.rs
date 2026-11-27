#![forbid(unsafe_code)]

//! Hardware parity, `0-ULP` linear limit check, and central finite-difference adjoint
//! verification for Block 0 linearly-implicit `PHS` `TPT-SVF` (`uniforms.block_idx == 0`) across
//! unsaturated (`s1^2 < 64`) and saturated (`E = 64`, `gamma = 0`) stereo regimes
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, build_core_quad_bundle_with_k_nl,
    cpu_params_from_packed, dispatch_adjoint_3pass, dispatch_kernel, read_vec4_vec,
};
use dnam_kernels::{
    CoreQuadV4Block0StateStereo8, CoreQuadV4ParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4,
    SOFT_RECT_EPS, StereoVec8, TINY_CROSSING_EPS, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
    step_core_quad_v4_phs_stereo8, step_core_quad_v4_stereo8,
};
use sona_shaders::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FUSED_FLAG_FIXED_POINT_SCAN, FusedScanUniforms,
    QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, quad_param_slots, resolve_quad_idx,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

const NUM_FRAMES: usize = 600;
const NUM_TILES: usize = 3;
const NUM_QUADS: usize = 6;
const WAVE0_WG_X: usize = 2;
const WAVE0_QUADS: [usize; WAVE0_WG_X] = [0, 4];
const NUM_STEMS: usize = 1;
const TOTAL_Q: usize = NUM_QUADS * NUM_STEMS;

fn make_asymmetric_phs_inputs(num_frames: usize) -> Vec<[f32; 4]> {
    let mut core_inputs = vec![[0.0_f32; 4]; TOTAL_Q * num_frames * CORE_INPUT_VEC4_STRIDE];
    for q in 0..NUM_QUADS {
        for t in 0..num_frames {
            let base = (q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
            let ph = (t as f32) * 0.085 + (q as f32) * 0.41;
            // Drive L strongly enough for |s1_L| >= 8.0 (E_max = 64.0) while keeping R unsaturated,
            // guaranteeing distinct independent stereo T_L != T_R Jacobians across tile seams.
            let drive_l = 26.0 + 6.0 * (0.23 * ph).cos();
            core_inputs[base] = [
                drive_l * ph.sin(),
                6.5 * (1.25 * ph).cos(),
                14.0 * (0.75 * ph).sin(),
                12.5 * (1.55 * ph).cos(),
            ];
            core_inputs[base + 1] = [
                0.35 * (1.1 * ph).cos(),
                0.65 * (0.85 * ph).sin() + 0.12,
                0.75 * (1.9 * ph).cos() - 0.10 * ph.sin(),
                0.85 * (1.3 * ph).sin(),
            ];
        }
    }
    core_inputs
}

fn build_phs_quad_bundles() -> (Vec<CoreQuadV4ParamsStereo4<GpuVec4>>, Vec<[f32; 4]>) {
    let k_nl_sets = [
        [0.05_f32, 0.45, 1.10, 3.25],
        [0.0_f32; 4],
        [0.0_f32; 4],
        [0.0_f32; 4],
        [0.0_f32, 0.35, 0.85, 2.10],
        [0.0_f32; 4],
    ];
    let mut cpu_params = Vec::with_capacity(TOTAL_Q);
    let mut packed_params = Vec::with_capacity(TOTAL_Q * QUAD_PARAMS_VEC4_STRIDE);
    for (q, k_nl) in k_nl_sets.into_iter().enumerate() {
        let profile_idx = if q == 4 { 2 } else { q % 4 };
        let (mut cpu_p, mut gpu_p) = build_core_quad_bundle_with_k_nl(0, profile_idx, k_nl);
        if q == 0 {
            let mut g_arr = gpu_p[quad_param_slots::SVF_G];
            g_arr[0] = prewarp_tan_pi_f_over_fs_f32(0.04);
            let k_low = [0.48_f32, 0.55, 0.75, 1.05];
            let svf =
                TptSvfCoeffs4::from_gk(GpuVec4::from_array(g_arr), GpuVec4::from_array(k_low));
            gpu_p[quad_param_slots::SVF_A1] = svf.a1.to_array();
            gpu_p[quad_param_slots::SVF_A2] = svf.a2.to_array();
            gpu_p[quad_param_slots::SVF_A3] = svf.a3.to_array();
            gpu_p[quad_param_slots::SVF_G] = g_arr;
            gpu_p[quad_param_slots::SVF_K] = k_low;
            cpu_p = cpu_params_from_packed(&gpu_p);
        }
        cpu_params.push(cpu_p);
        packed_params.extend_from_slice(&gpu_p);
    }
    (cpu_params, packed_params)
}

fn eval_cpu_phs_loss(
    core_inputs: &[[f32; 4]],
    params_cpu: &[CoreQuadV4ParamsStereo4<GpuVec4>],
    weights: &[[f32; 4]],
    q_global: usize,
    num_frames: usize,
) -> (f64, bool, bool) {
    let mut state = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut loss = 0.0_f64;
    let mut saw_saturated = false;
    let mut saw_unsaturated_nonzero = false;
    for t in 0..num_frames {
        let (s1_l, _) = state.svf_lin_s1.into_lr();
        let s1_sq = s1_l.to_array()[0] * s1_l.to_array()[0];
        if s1_sq >= 64.0 {
            saw_saturated = true;
        } else if s1_sq > 1.0 {
            saw_unsaturated_nonzero = true;
        }
        let in_base = (q_global * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(core_inputs[in_base]),
            GpuVec4::from_array(core_inputs[in_base + 1]),
        );
        let out = step_core_quad_v4_phs_stereo8(u, &mut state, &params_cpu[q_global]);
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
    (loss, saw_saturated, saw_unsaturated_nonzero)
}

fn verify_k_nl_zero_linear_parity(ctx: &GpuContext, variant: ShaderVariant, inputs: &[[f32; 4]]) {
    let (_cpu_p, packed_zero) = build_core_quad_bundle_with_k_nl(0, 0, [0.0; 4]);
    let single_in = &inputs[..NUM_FRAMES * CORE_INPUT_VEC4_STRIDE];
    let ckpt_init = vec![[0.0_f32; 4]; NUM_TILES * TILE_CKPT_VEC4_STRIDE];
    let feat_init = vec![[0.0_f32; 4]; NUM_FRAMES * FEATURE_VEC4_STRIDE];
    let u0 = FusedScanUniforms {
        num_frames: NUM_FRAMES as u32,
        num_tiles: NUM_TILES as u32,
        num_quads: 1,
        num_stems: 1,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx: 0,
    };
    let (out_phs, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u0.as_bytes(),
        &[
            (false, as_byte_slice(single_in)),
            (false, as_byte_slice(&packed_zero)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let dual_in = [single_in, single_in].concat();
    let u1 = FusedScanUniforms {
        num_quads: 2,
        block_idx: 1,
        ..u0
    };
    let (out_lin, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u1.as_bytes(),
        &[
            (false, as_byte_slice(&dual_in)),
            (false, as_byte_slice(&[packed_zero, packed_zero].concat())),
            (
                true,
                as_byte_slice(&vec![[0.0_f32; 4]; 2 * NUM_TILES * TILE_CKPT_VEC4_STRIDE]),
            ),
            (
                true,
                as_byte_slice(&vec![[0.0_f32; 4]; 2 * NUM_FRAMES * FEATURE_VEC4_STRIDE]),
            ),
        ],
    );
    let (_, packed_nz) = build_core_quad_bundle_with_k_nl(0, 0, [0.45, 1.10, 2.50, 4.00]);
    let (out_lin_nz, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u1.as_bytes(),
        &[
            (false, as_byte_slice(&dual_in)),
            (false, as_byte_slice(&[packed_nz, packed_nz].concat())),
            (
                true,
                as_byte_slice(&vec![[0.0_f32; 4]; 2 * NUM_TILES * TILE_CKPT_VEC4_STRIDE]),
            ),
            (
                true,
                as_byte_slice(&vec![[0.0_f32; 4]; 2 * NUM_FRAMES * FEATURE_VEC4_STRIDE]),
            ),
        ],
    );
    assert_eq!(out_lin_nz[0], out_lin[0]);
    assert_eq!(out_lin_nz[1], out_lin[1]);
    let d_loss_lin = vec![[0.1_f32; 4]; 2 * NUM_FRAMES * FEATURE_VEC4_STRIDE];
    let pg_lin_v = dispatch_adjoint_3pass(
        ctx,
        variant,
        NUM_FRAMES,
        NUM_TILES,
        2,
        1,
        1,
        1,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&dual_in),
        as_byte_slice(&[packed_nz, packed_nz].concat()),
        &out_lin_nz[0],
        as_byte_slice(&d_loss_lin),
        &out_lin_nz[1],
        None,
    )
    .param_grads;
    for tile in 0..NUM_TILES {
        let idx = (NUM_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE + quad_param_slots::SVF_K_NL;
        assert_eq!(pg_lin_v[idx].map(f32::to_bits), [0; 4]);
    }
    let feats_phs = read_vec4_vec(&out_phs[1]);
    let feats_lin_all = read_vec4_vec(&out_lin[1]);
    let feats_lin = &feats_lin_all[NUM_FRAMES * FEATURE_VEC4_STRIDE..];
    let mut state_phs = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut state_lin = dnam_kernels::SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let cpu_zero = cpu_params_from_packed(&packed_zero);
    for t in 0..NUM_FRAMES {
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(single_in[t * 2]),
            GpuVec4::from_array(single_in[t * 2 + 1]),
        );
        let op = step_core_quad_v4_phs_stereo8(u, &mut state_phs, &cpu_zero);
        let ol = step_core_quad_v4_stereo8(u, &mut state_lin, &cpu_zero);
        assert_eq!(
            op.lru_state.into_lr().0.to_array().map(f32::to_bits),
            ol.lru_state.into_lr().0.to_array().map(f32::to_bits)
        );
        assert_eq!(
            op.o_tilde.into_lr().1.to_array().map(f32::to_bits),
            ol.o_tilde.into_lr().1.to_array().map(f32::to_bits)
        );
        assert_eq!(
            op.e_tilde.into_lr().0.to_array().map(f32::to_bits),
            ol.e_tilde.into_lr().0.to_array().map(f32::to_bits)
        );
        // Sequential Block 0 PHS (k_nl == 0) vs O(log2 N) Blelloch Block 1 scan agrees within
        // 32-bit associative reassociation tolerance (`1e-4 * max(1, |x|)`), while CPU steps match 0-ULP.
        for s in 0..FEATURE_VEC4_STRIDE {
            let idx = t * FEATURE_VEC4_STRIDE + s;
            for lane in 0..4 {
                let a = feats_phs[idx][lane];
                let b = feats_lin[idx][lane];
                let tol = 1.0e-4 * a.abs().max(b.abs()).max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "k_nl==0 GPU parity mismatch t={t} s={s} lane={lane}: phs={a} lin={b}"
                );
            }
        }
    }
}

pub fn verify_block0_phs_core_and_adjoint_fd(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let core_inputs = make_asymmetric_phs_inputs(NUM_FRAMES);
    verify_k_nl_zero_linear_parity(ctx, variant, &core_inputs);

    let (quad_params_cpu, quad_params_packed) = build_phs_quad_bundles();
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
        block_idx: 0,
    };
    let (fwd_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (WAVE0_WG_X as u32, 1, NUM_STEMS as u32),
        fwd_uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let gpu_ckpts = read_vec4_vec(&fwd_out[0]);
    let gpu_feats = read_vec4_vec(&fwd_out[1]);

    // Check forward parity against step_core_quad_v4_phs_stereo8 and Zero-DC at block_idx = 0.
    for (ql, &q_global) in WAVE0_QUADS.iter().enumerate() {
        assert_eq!(resolve_quad_idx(0, ql, NUM_QUADS), q_global);
        let mut state = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let mut cpu_feats = Vec::with_capacity(NUM_FRAMES * FEATURE_VEC4_STRIDE);
        let mut peak = 1.0_f32;
        for t in 0..NUM_FRAMES {
            let in_base = (q_global * NUM_FRAMES + t) * CORE_INPUT_VEC4_STRIDE;
            let u = InterleavedStereo8::from_lr(
                GpuVec4::from_array(core_inputs[in_base]),
                GpuVec4::from_array(core_inputs[in_base + 1]),
            );
            let out = step_core_quad_v4_phs_stereo8(u, &mut state, &quad_params_cpu[q_global]);
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
                let cpu = cpu_feats[t * FEATURE_VEC4_STRIDE + slot];
                let g = gpu_feats[(q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot];
                for lane in 0..4 {
                    assert!(
                        (g[lane] - cpu[lane]).abs() <= tol,
                        "PHS ({label}) fwd mismatch q={q_global} t={t} slot={slot} lane={lane}: gpu={} cpu={} peak={peak} tol={tol}",
                        g[lane],
                        cpu[lane]
                    );
                }
            }
        }
    }

    // Verify FUSED_FLAG_FIXED_POINT_SCAN Passive Picard (Block 0) + Damped-DEER (Block 0 & Block 1) <= 1e-6 parity.
    let mut fp_u0 = fwd_uniforms;
    fp_u0.block_idx = FUSED_FLAG_FIXED_POINT_SCAN;
    let (fp_out0, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (WAVE0_WG_X as u32, 1, NUM_STEMS as u32),
        fp_u0.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let fp_feats0 = read_vec4_vec(&fp_out0[1]);
    for &q_global in &WAVE0_QUADS {
        for t in 0..NUM_FRAMES {
            for slot in 0..FEATURE_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
                for lane in 0..4 {
                    let ref_val = fp_feats0[idx][lane];
                    let got_val = gpu_feats[idx][lane];
                    let diff = (ref_val - got_val).abs();
                    let tol = 1.0e-5 * ref_val.abs().max(got_val.abs()).max(1.0);
                    assert!(
                        diff <= tol,
                        "Fixed-point PHS/DEER ({label}) mismatch q={q_global} t={t} slot={slot} lane={lane}: diff={diff} tol={tol}"
                    );
                }
            }
        }
    }
    let lin_seq_u = FusedScanUniforms {
        block_idx: 1,
        ..fwd_uniforms
    };
    let (lin_seq_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (2, 1, NUM_STEMS as u32),
        lin_seq_u.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let lin_fp_u = FusedScanUniforms {
        block_idx: 1 | FUSED_FLAG_FIXED_POINT_SCAN,
        ..fwd_uniforms
    };
    let (lin_fp_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (2, 1, NUM_STEMS as u32),
        lin_fp_u.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let lin_seq_f = read_vec4_vec(&lin_seq_out[1]);
    let lin_fp_f = read_vec4_vec(&lin_fp_out[1]);
    for q_global in [1_usize, 5] {
        for t in 0..NUM_FRAMES {
            for slot in 0..FEATURE_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
                for lane in 0..4 {
                    let ref_val = lin_fp_f[idx][lane];
                    let got_val = lin_seq_f[idx][lane];
                    let diff = (ref_val - got_val).abs();
                    let tol = 1.0e-5 * ref_val.abs().max(got_val.abs()).max(1.0);
                    assert!(
                        diff <= tol,
                        "Fixed-point Linear Damped-DEER ({label}) mismatch q={q_global} t={t} slot={slot} lane={lane}: diff={diff} tol={tol}"
                    );
                }
            }
        }
    }

    // Zero-DC canonical +0.0 verification on block_idx = 0 with nonzero svf_k_nl.
    let neg_zero_in = vec![[-0.0_f32; 4]; core_inputs.len()];
    let (zero_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (WAVE0_WG_X as u32, 1, NUM_STEMS as u32),
        fwd_uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&neg_zero_in)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    for buf in zero_out {
        for v in read_vec4_vec(&buf) {
            assert_eq!(v.map(f32::to_bits), [0; 4]);
        }
    }

    // Verify tile_start = 1 resume bit-identity on block_idx = 0.
    let mut resume_u = fwd_uniforms;
    resume_u.tile_start = 1;
    let (res_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (WAVE0_WG_X as u32, 1, NUM_STEMS as u32),
        resume_u.as_bytes(),
        &[
            (false, as_byte_slice(&core_inputs)),
            (false, as_byte_slice(&quad_params_packed)),
            (true, as_byte_slice(&gpu_ckpts)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let res_feats = read_vec4_vec(&res_out[1]);
    for &q_global in &WAVE0_QUADS {
        for t in 256..NUM_FRAMES {
            for slot in 0..FEATURE_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
                assert_eq!(
                    res_feats[idx].map(f32::to_bits),
                    gpu_feats[idx].map(f32::to_bits)
                );
            }
        }
    }

    let mut d_loss_d_features = vec![[0.0_f32; 4]; TOTAL_Q * NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for idx in 0..d_loss_d_features.len() {
        let base = (idx as f32) * 0.023;
        d_loss_d_features[idx] = [
            0.30 * base.sin(),
            0.25 * (1.2 * base).cos(),
            0.28 * (0.7 * base).sin(),
            0.31 * (1.6 * base).cos(),
        ];
    }

    let (_, saw_sat, saw_unsat) = eval_cpu_phs_loss(
        &core_inputs,
        &quad_params_cpu,
        &d_loss_d_features,
        0,
        NUM_FRAMES,
    );
    assert!(
        saw_sat && saw_unsat,
        "PHS test must exercise both saturated and unsaturated E[n] regimes"
    );

    let phs_adj = dispatch_adjoint_3pass(
        ctx,
        variant,
        NUM_FRAMES,
        NUM_TILES,
        NUM_QUADS,
        NUM_STEMS,
        WAVE0_WG_X,
        0,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&core_inputs),
        as_byte_slice(&quad_params_packed),
        &fwd_out[0],
        as_byte_slice(&d_loss_d_features),
        &fwd_out[1],
        None,
    );
    let got_du = phs_adj.du;
    let got_pgrad = phs_adj.param_grads;

    for &q_global in &WAVE0_QUADS {
        for &t in &[0_usize, 19, 254, 255, 256, 257, 510, 511, 512, 598, 599] {
            for ch in 0..CORE_INPUT_VEC4_STRIDE {
                let idx = (q_global * NUM_FRAMES + t) * CORE_INPUT_VEC4_STRIDE + ch;
                for lane in [0_usize, 1, 2, 3] {
                    // Scale step to drive amplitude (`26..32` on `ch=0 lane=0`): `1e-2` avoids `f32` FD
                    // cancellation on `600`-frame PHS; `t=0` (`u=0`) uses `4e-3` to stay inside the `z=0` knee.
                    let eps_u = if ch == 0 && lane == 0 {
                        if t == 0 && quad_params_cpu[q_global].svf_k_nl.to_array()[lane] != 0.0 {
                            4.0e-3_f32
                        } else {
                            1.0e-2_f32
                        }
                    } else {
                        1.0e-3_f32
                    };
                    let mut p_plus = core_inputs.clone();
                    let mut p_minus = core_inputs.clone();
                    p_plus[idx][lane] += eps_u;
                    p_minus[idx][lane] -= eps_u;
                    let delta = (p_plus[idx][lane] - p_minus[idx][lane]) as f64;
                    let (lp, _, _) = eval_cpu_phs_loss(
                        &p_plus,
                        &quad_params_cpu,
                        &d_loss_d_features,
                        q_global,
                        NUM_FRAMES,
                    );
                    let (lm, _, _) = eval_cpu_phs_loss(
                        &p_minus,
                        &quad_params_cpu,
                        &d_loss_d_features,
                        q_global,
                        NUM_FRAMES,
                    );
                    let fd = (lp - lm) / delta;
                    let g = got_du[idx][lane] as f64;
                    let tol = 6.0e-3 * fd.abs().max(1.0);
                    assert!(
                        (g - fd).abs() <= tol,
                        "PHS ({label}) input FD mismatch q={q_global} t={t} ch={ch} lane={lane}: got={g} fd={fd}"
                    );
                }
            }
        }
    }

    // High-drive lane 0 reaches |s1|=8 saturation; smaller FD steps avoid curvature/kink drift.
    let phs_slots: &[(usize, f32, f32)] = &[
        (quad_param_slots::SVF_K_NL, 2.0e-5, 5.0e-4),
        (quad_param_slots::SVF_K, 2.0e-3, 2.0e-3),
        (quad_param_slots::SVF_G, 1.5e-4, 1.0e-3),
        (quad_param_slots::TONE_C_LP, 2.0e-3, 2.5e-4),
        (quad_param_slots::TONE_C_BP, 5.0e-4, 5.0e-4),
        (quad_param_slots::TONE_C_KBP, 1.0e-3, 1.0e-3),
        (quad_param_slots::TONE_C_U, 5.0e-4, 2.5e-4),
        (quad_param_slots::ADAA_DRIVE, 2.0e-3, 2.0e-3),
        (quad_param_slots::GLU_MIX, 2.0e-3, 2.0e-3),
        (quad_param_slots::OUT_GAIN, 2.0e-3, 2.0e-3),
    ];
    for &q_global in &WAVE0_QUADS {
        let base_slice = &quad_params_packed
            [q_global * QUAD_PARAMS_VEC4_STRIDE..(q_global + 1) * QUAD_PARAMS_VEC4_STRIDE];
        for zero_slot in [
            quad_param_slots::SVF_A1,
            quad_param_slots::SVF_A2,
            quad_param_slots::SVF_A3,
        ] {
            for tile in 0..NUM_TILES {
                let p_idx = (q_global * NUM_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE + zero_slot;
                assert_eq!(got_pgrad[p_idx].map(f32::to_bits), [0; 4]);
            }
        }

        for &(slot, eps_l0, eps_l13) in phs_slots {
            for lane in 0..4 {
                let eps = if lane == 0 { eps_l0 } else { eps_l13 };
                let mut g_gpu = 0.0_f64;
                for tile in 0..NUM_TILES {
                    let p_idx = (q_global * NUM_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE + slot;
                    g_gpu += got_pgrad[p_idx][lane] as f64;
                }
                let mut p_plus_packed = base_slice.to_vec();
                let mut p_minus_packed = base_slice.to_vec();
                let v = base_slice[slot][lane];
                let (vp, vm) = if slot == quad_param_slots::SVF_K_NL && v == 0.0 {
                    (v + eps, 0.0)
                } else {
                    (v + eps, v - eps)
                };
                p_plus_packed[slot][lane] = vp;
                p_minus_packed[slot][lane] = vm;
                let delta = (vp - vm) as f64;

                let mut p_cpu = quad_params_cpu.clone();
                p_cpu[q_global] = cpu_params_from_packed(&p_plus_packed);
                let (lp, _, _) = eval_cpu_phs_loss(
                    &core_inputs,
                    &p_cpu,
                    &d_loss_d_features,
                    q_global,
                    NUM_FRAMES,
                );
                p_cpu[q_global] = cpu_params_from_packed(&p_minus_packed);
                let (lm, _, _) = eval_cpu_phs_loss(
                    &core_inputs,
                    &p_cpu,
                    &d_loss_d_features,
                    q_global,
                    NUM_FRAMES,
                );
                let fd = (lp - lm) / delta;
                let tol = 8.0e-3 * fd.abs().max(1.0);
                assert!(
                    (g_gpu - fd).abs() <= tol,
                    "PHS ({label}) param FD mismatch q={q_global} slot={slot} lane={lane}: got={g_gpu} fd={fd} tol={tol}"
                );
                if slot == quad_param_slots::SVF_K_NL {
                    assert!(
                        g_gpu.abs() > 1.0e-3,
                        "expected nonzero SVF_K_NL gradient on PHS q={q_global} lane={lane}, got {g_gpu} (fd={fd})"
                    );
                }
            }
        }
    }
}
