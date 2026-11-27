#![forbid(unsafe_code)]

//! Hardware parity, `-0.0` Zero-DC, resume, and central finite-difference adjoint tests for:
//! - `kappa * (1 - signed_mix) == 0` `O(log2 N)` `ParaLRU` fast-path (`kappa == 0.0` & `signed_mix == 1.0`)
//! - `8`-thread `L/R x 4-lane` coupled `ParaLRU` split + `FUSED_FLAG_FIXED_POINT_SCAN`
//! - Constant-`M` register-squaring `TPT-SVF` forward & adjoint scans (`Blocks 1..3`)
//! - `pass0` binary-tree reductions (`tpt_adjoint_const_m_reduce_256_lr` & `tpt_adjoint_dual_reduce_256`)

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, build_core_quad_bundle_with_k_nl,
    cpu_params_from_packed, dispatch_adjoint_3pass, dispatch_kernel, read_vec4_vec,
};
use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, SOFT_RECT_EPS,
    SonaBlockStateStereo8, SonaParaLruDuhemCoeffs4, StereoVec8, TINY_CROSSING_EPS,
    step_core_quad_phs_stereo8, step_core_quad_stereo8,
};
use sona_shaders::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FUSED_FLAG_FIXED_POINT_SCAN, FusedScanUniforms,
    QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, quad_param_slots,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

const NUM_FRAMES: usize = 600;
const NUM_TILES: usize = 3;

fn make_inputs(num_frames: usize) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
    for t in 0..num_frames {
        let ph = (t as f32) * 0.081;
        out[t * 2] = [
            0.55 * ph.sin() + 0.14 * (2.1 * ph).cos(),
            0.42 * (1.3 * ph).cos(),
            -0.48 * (0.75 * ph).sin() + 0.11,
            0.36 * (1.7 * ph).cos(),
        ];
        out[t * 2 + 1] = [
            0.44 * (1.15 * ph).cos(),
            0.51 * (0.85 * ph).sin() - 0.12,
            0.39 * (1.9 * ph).cos(),
            -0.46 * (1.25 * ph).sin(),
        ];
    }
    out
}

fn build_mixed_decoupled_bundle(
    k_nl: [f32; 4],
) -> (
    CoreQuadParamsStereo4<GpuVec4>,
    [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE],
) {
    let (mut cpu_p, mut packed) = build_core_quad_bundle_with_k_nl(0, 0, k_nl);
    // Lane 0: kappa == 0.0 (decoupled, nonzero g_kappa); Lane 1: coupled;
    // Lane 2: signed_mix == 1.0 (decoupled even with kappa > 0); Lane 3: coupled.
    let kappa = [0.0_f32, 0.40, 0.55, 0.70];
    let signed_mix = [0.20_f32, 0.50, 1.00, 0.70];
    let alpha_rel = [0.91_f32, 0.87, 0.82, 0.76];
    let alpha_atk = [0.25_f32, 0.35, 0.45, 0.60];
    let gamma_duhem = [0.15_f32, 0.25, 0.30, 0.45];
    let stereo_link = [0.18_f32, 0.32, 0.48, 0.62];

    packed[quad_param_slots::LRU_ALPHA_REL] = alpha_rel;
    packed[quad_param_slots::LRU_ALPHA_ATK] = alpha_atk;
    packed[quad_param_slots::LRU_KAPPA] = kappa;
    packed[quad_param_slots::LRU_GAMMA_DUHEM] = gamma_duhem;
    packed[quad_param_slots::LRU_SIGNED_MIX] = signed_mix;
    packed[quad_param_slots::LRU_STEREO_LINK] = stereo_link;
    cpu_p.lru = SonaParaLruDuhemCoeffs4::from_clamped(
        GpuVec4::from_array(alpha_rel),
        GpuVec4::from_array(alpha_atk),
        GpuVec4::from_array(kappa),
        GpuVec4::from_array(gamma_duhem),
        GpuVec4::from_array(signed_mix),
        GpuVec4::from_array(stereo_link),
    );
    (cpu_p, packed)
}

fn eval_cpu_loss_block(
    core_inputs: &[[f32; 4]],
    params_cpu: &CoreQuadParamsStereo4<GpuVec4>,
    weights: &[[f32; 4]],
    is_phs: bool,
) -> f64 {
    let mut state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut loss = 0.0_f64;
    for t in 0..NUM_FRAMES {
        let in_base = t * CORE_INPUT_VEC4_STRIDE;
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(core_inputs[in_base]),
            GpuVec4::from_array(core_inputs[in_base + 1]),
        );
        let out = if is_phs {
            step_core_quad_phs_stereo8(u, &mut state, params_cpu)
        } else {
            step_core_quad_stereo8(u, &mut state, params_cpu)
        };
        let (vl, vr) = out.lru_state.into_lr();
        let (ol, or) = out.adaa_odd.into_lr();
        let (el, er) = out.adaa_even.into_lr();
        let feats = [
            vl.to_array(),
            vr.to_array(),
            ol.to_array(),
            or.to_array(),
            el.to_array(),
            er.to_array(),
        ];
        let w_base = t * FEATURE_VEC4_STRIDE;
        for f_idx in 0..FEATURE_VEC4_STRIDE {
            for lane in 0..4 {
                loss += (feats[f_idx][lane] as f64) * (weights[w_base + f_idx][lane] as f64);
            }
        }
    }
    loss
}

pub fn verify_paralru_fastpath_and_const_m_scans(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let inputs = make_inputs(NUM_FRAMES);
    let (cpu_p, packed_p) = build_mixed_decoupled_bundle([0.0; 4]);
    let dual_packed = [packed_p, packed_p].concat();
    let dual_inputs = [inputs.as_slice(), inputs.as_slice()].concat();

    let ckpt_init = vec![[0.0_f32; 4]; 2 * NUM_TILES * TILE_CKPT_VEC4_STRIDE];
    let feat_init = vec![[0.0_f32; 4]; 2 * NUM_FRAMES * FEATURE_VEC4_STRIDE];

    // 1. Forward parity on Block 1 (constant-M + mixed decoupled/coupled lanes), fixed-point flag,
    //    tile_start = 1 resume, and -0.0 Zero-DC.
    let u_lin = FusedScanUniforms {
        num_frames: NUM_FRAMES as u32,
        num_tiles: NUM_TILES as u32,
        num_quads: 2,
        num_stems: 1,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx: 1,
    };
    let (fwd_lin, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u_lin.as_bytes(),
        &[
            (false, as_byte_slice(&dual_inputs)),
            (false, as_byte_slice(&dual_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let gpu_feats = read_vec4_vec(&fwd_lin[1]);

    let mut cpu_state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut cpu_feats = Vec::with_capacity(NUM_FRAMES * FEATURE_VEC4_STRIDE);
    let mut peak = 1.0_f32;
    for t in 0..NUM_FRAMES {
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(inputs[t * 2]),
            GpuVec4::from_array(inputs[t * 2 + 1]),
        );
        let out = step_core_quad_stereo8(u, &mut cpu_state, &cpu_p);
        let (vl, vr) = out.lru_state.into_lr();
        let (ol, or) = out.adaa_odd.into_lr();
        let (el, er) = out.adaa_even.into_lr();
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
            let gpu = gpu_feats[(NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot];
            let cpu = cpu_feats[t * FEATURE_VEC4_STRIDE + slot];
            for lane in 0..4 {
                assert!(
                    (gpu[lane] - cpu[lane]).abs() <= tol,
                    "mixed ParaLRU Block 1 ({label}) t={t} slot={slot} lane={lane}: gpu={} cpu={}",
                    gpu[lane],
                    cpu[lane]
                );
            }
        }
    }

    // Fixed-point flag parity (decoupled lanes take O(log2 N) fast path, coupled take DEER).
    let u_fp = FusedScanUniforms {
        block_idx: 1 | FUSED_FLAG_FIXED_POINT_SCAN,
        ..u_lin
    };
    let (fwd_fp, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u_fp.as_bytes(),
        &[
            (false, as_byte_slice(&dual_inputs)),
            (false, as_byte_slice(&dual_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let fp_feats = read_vec4_vec(&fwd_fp[1]);
    for t in 0..NUM_FRAMES {
        for slot in 0..FEATURE_VEC4_STRIDE {
            let idx = (NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
            for lane in 0..4 {
                let diff = (fp_feats[idx][lane] - gpu_feats[idx][lane]).abs();
                assert!(
                    diff <= 1.0e-5 * peak,
                    "fixed-point mixed ParaLRU ({label}) t={t} slot={slot} lane={lane}: diff={diff}"
                );
            }
        }
    }

    // tile_start = 1 resume bit-exactness.
    let mut u_res = u_lin;
    u_res.tile_start = 1;
    let (res_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u_res.as_bytes(),
        &[
            (false, as_byte_slice(&dual_inputs)),
            (false, as_byte_slice(&dual_packed)),
            (true, &fwd_lin[0]),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    let res_feats = read_vec4_vec(&res_out[1]);
    for t in 256..NUM_FRAMES {
        for slot in 0..FEATURE_VEC4_STRIDE {
            let idx = (NUM_FRAMES + t) * FEATURE_VEC4_STRIDE + slot;
            assert_eq!(
                res_feats[idx].map(f32::to_bits),
                gpu_feats[idx].map(f32::to_bits),
                "tile_start=1 resume ({label}) mismatch t={t} slot={slot}"
            );
        }
    }

    // -0.0 Zero-DC invariance.
    let neg_zero_in = vec![[-0.0_f32; 4]; dual_inputs.len()];
    let (z_out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u_lin.as_bytes(),
        &[
            (false, as_byte_slice(&neg_zero_in)),
            (false, as_byte_slice(&dual_packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    for v in read_vec4_vec(&z_out[1]) {
        assert_eq!(v.map(f32::to_bits), [0; 4]);
    }

    // 2. Central FD adjoint parity on Block 1 (!is_phs) and Block 0 (is_phs) with mixed decoupled lanes.
    let mut d_loss = vec![[0.0_f32; 4]; NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for (idx, row) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.017;
        *row = [
            0.33 * ph.sin(),
            0.29 * (1.2 * ph).cos(),
            0.27 * (0.8 * ph).sin(),
            0.31 * (1.5 * ph).cos(),
        ];
    }
    let dual_d_loss = [d_loss.as_slice(), d_loss.as_slice()].concat();

    for (block_idx, is_phs, k_nl) in [
        (1_u32, false, [0.0; 4]),
        (0_u32, true, [0.15, 0.35, 0.70, 1.1]),
    ] {
        let (cpu_b, packed_b) = build_mixed_decoupled_bundle(k_nl);
        let fwd_b_u = FusedScanUniforms {
            num_frames: NUM_FRAMES as u32,
            num_tiles: NUM_TILES as u32,
            num_quads: 2,
            num_stems: 1,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx,
        };
        let dual_pb = [packed_b, packed_b].concat();
        let (fwd_b, _) = dispatch_kernel(
            ctx,
            KernelId::FusedCoreScan256,
            variant,
            (1, 1, 1),
            fwd_b_u.as_bytes(),
            &[
                (false, as_byte_slice(&dual_inputs)),
                (false, as_byte_slice(&dual_pb)),
                (true, as_byte_slice(&ckpt_init)),
                (true, as_byte_slice(&feat_init)),
            ],
        );
        let adj_b = dispatch_adjoint_3pass(
            ctx,
            variant,
            NUM_FRAMES,
            NUM_TILES,
            2,
            1,
            1,
            block_idx,
            SOFT_RECT_EPS,
            TINY_CROSSING_EPS,
            as_byte_slice(&dual_inputs),
            as_byte_slice(&dual_pb),
            &fwd_b[0],
            as_byte_slice(&dual_d_loss),
            &fwd_b[1],
            None,
        );
        let q_off = usize::from(!is_phs);
        let in_off = q_off * NUM_FRAMES * CORE_INPUT_VEC4_STRIDE;

        for &t in &[0_usize, 19, 255, 256, 511, 512, 599] {
            for ch in 0..2 {
                let idx = in_off + t * CORE_INPUT_VEC4_STRIDE + ch;
                for lane in 0..4 {
                    let eps_u = 2.0e-3_f32;
                    let mut pp = inputs.clone();
                    let mut pm = inputs.clone();
                    pp[t * 2 + ch][lane] += eps_u;
                    pm[t * 2 + ch][lane] -= eps_u;
                    let delta = (pp[t * 2 + ch][lane] - pm[t * 2 + ch][lane]) as f64;
                    let lp = eval_cpu_loss_block(&pp, &cpu_b, &d_loss, is_phs);
                    let lm = eval_cpu_loss_block(&pm, &cpu_b, &d_loss, is_phs);
                    let fd = (lp - lm) / delta;
                    let g = adj_b.du[idx][lane] as f64;
                    let tol_du = 5.0e-3 * fd.abs().max(1.0);
                    assert!(
                        (g - fd).abs() <= tol_du,
                        "block={block_idx} ({label}) input FD mismatch t={t} ch={ch} lane={lane}: got={g} fd={fd}"
                    );
                }
            }
        }

        // Verify g_kappa is nonzero on lane 0 (kappa == 0.0, signed_mix < 1.0) and FD-accurate across all 4 lanes.
        for &(slot, eps) in &[
            (quad_param_slots::LRU_KAPPA, 1.5e-3_f32),
            (quad_param_slots::LRU_ALPHA_REL, 1.5e-3),
            (quad_param_slots::SVF_K, 2.0e-3),
            (quad_param_slots::SVF_G, 2.0e-4),
        ] {
            for lane in 0..4 {
                let mut g_gpu = 0.0_f64;
                for tile in 0..NUM_TILES {
                    let p_idx = (q_off * NUM_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE + slot;
                    g_gpu += adj_b.param_grads[p_idx][lane] as f64;
                }
                if slot == quad_param_slots::LRU_KAPPA && lane == 0 {
                    assert!(
                        g_gpu.abs() > 1.0e-4,
                        "expected nonzero g_kappa on kappa==0.0 && signed_mix<1.0 lane 0, got {g_gpu}"
                    );
                }
                let base_v = packed_b[slot][lane];
                let (vp, vm) = if (slot == quad_param_slots::LRU_KAPPA && lane == 0)
                    || (slot == quad_param_slots::LRU_SIGNED_MIX && lane == 2)
                {
                    (base_v + eps, base_v)
                } else {
                    (base_v + eps, base_v - eps)
                };
                let mut pp = packed_b;
                let mut pm = packed_b;
                pp[slot][lane] = vp;
                pm[slot][lane] = vm;
                let lp =
                    eval_cpu_loss_block(&inputs, &cpu_params_from_packed(&pp), &d_loss, is_phs);
                let lm =
                    eval_cpu_loss_block(&inputs, &cpu_params_from_packed(&pm), &d_loss, is_phs);
                let fd = (lp - lm) / ((vp - vm) as f64);
                let tol_p = 5.0e-3 * fd.abs().max(1.0);
                assert!(
                    (g_gpu - fd).abs() <= tol_p,
                    "block={block_idx} ({label}) param FD mismatch slot={slot} lane={lane}: got={g_gpu} fd={fd}"
                );
            }
        }
    }

    // 3. Multi-tile (8-tile, 2048-frame) low-fc Block 1 (!is_phs) overdrive central-FD parity
    //    covering cross-tile TPT-SVF cancellation matching `test_case_11` q=1 regimes.
    {
        const LOW_FC_FRAMES: usize = 2048;
        const LOW_FC_TILES: usize = 8;
        let inputs2 = make_inputs(LOW_FC_FRAMES);
        let (_, mut packed2) = build_mixed_decoupled_bundle([0.0; 4]);
        let g0 = 0.04_f32;
        let k0 = 0.25_f32;
        let a1 = 1.0_f32 / (1.0 + g0 * (g0 + k0));
        let a2 = g0 * a1;
        let a3 = g0 * a2;
        packed2[quad_param_slots::SVF_A1][0] = a1;
        packed2[quad_param_slots::SVF_A2][0] = a2;
        packed2[quad_param_slots::SVF_A3][0] = a3;
        packed2[quad_param_slots::SVF_K][0] = k0;
        packed2[quad_param_slots::SVF_TWO_K][0] = 2.0 * k0;
        packed2[quad_param_slots::SVF_G][0] = g0;
        packed2[quad_param_slots::TWO_SVF_G][0] = 2.0 * g0;
        let dual_in2 = [inputs2.as_slice(), inputs2.as_slice()].concat();
        let dual_p2 = [packed2, packed2].concat();
        let ckpt2 = vec![[0.0_f32; 4]; 2 * LOW_FC_TILES * TILE_CKPT_VEC4_STRIDE];
        let feat2 = vec![[0.0_f32; 4]; 2 * LOW_FC_FRAMES * FEATURE_VEC4_STRIDE];
        let mut dl2 = vec![[0.0_f32; 4]; LOW_FC_FRAMES * FEATURE_VEC4_STRIDE];
        for t in 0..LOW_FC_FRAMES {
            let scale = 0.35_f32;
            let ph = (t as f32) * 0.037;
            for f in 0..FEATURE_VEC4_STRIDE {
                dl2[t * FEATURE_VEC4_STRIDE + f] = [
                    scale * ph.sin(),
                    scale * ph.cos(),
                    0.5 * scale,
                    -0.5 * scale,
                ];
            }
        }
        let dual_dl2 = [dl2.as_slice(), dl2.as_slice()].concat();
        let fwd_u2 = FusedScanUniforms {
            num_frames: LOW_FC_FRAMES as u32,
            num_tiles: LOW_FC_TILES as u32,
            num_quads: 2,
            num_stems: 1,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx: 1,
        };
        let (fwd2, _) = dispatch_kernel(
            ctx,
            KernelId::FusedCoreScan256,
            variant,
            (1, 1, 1),
            fwd_u2.as_bytes(),
            &[
                (false, as_byte_slice(&dual_in2)),
                (false, as_byte_slice(&dual_p2)),
                (true, as_byte_slice(&ckpt2)),
                (true, as_byte_slice(&feat2)),
            ],
        );
        let adj2 = dispatch_adjoint_3pass(
            ctx,
            variant,
            LOW_FC_FRAMES,
            LOW_FC_TILES,
            2,
            1,
            1,
            1,
            SOFT_RECT_EPS,
            TINY_CROSSING_EPS,
            as_byte_slice(&dual_in2),
            as_byte_slice(&dual_p2),
            &fwd2[0],
            as_byte_slice(&dual_dl2),
            &fwd2[1],
            None,
        );
        let eval_gpu_u = |ins: &[[f32; 4]], p: &[[[f32; 4]; QUAD_PARAMS_VEC4_STRIDE]]| -> f64 {
            let di = [ins, ins].concat();
            let dp = [p[0], p[0]].concat();
            let (fo, _) = dispatch_kernel(
                ctx,
                KernelId::FusedCoreScan256,
                variant,
                (1, 1, 1),
                fwd_u2.as_bytes(),
                &[
                    (false, as_byte_slice(&di)),
                    (false, as_byte_slice(&dp)),
                    (true, as_byte_slice(&ckpt2)),
                    (true, as_byte_slice(&feat2)),
                ],
            );
            let gf = read_vec4_vec(&fo[1]);
            let mut loss = 0.0_f64;
            for i in 0..LOW_FC_FRAMES {
                let wb = i * FEATURE_VEC4_STRIDE;
                let gb = (LOW_FC_FRAMES + i) * FEATURE_VEC4_STRIDE;
                for f_idx in 0..FEATURE_VEC4_STRIDE {
                    for lane in 0..4 {
                        loss += (gf[gb + f_idx][lane] as f64) * (dl2[wb + f_idx][lane] as f64);
                    }
                }
            }
            loss
        };
        for &t in &[0_usize, 255, 256, 511, 512, 1023, 1024, 1536, 2047] {
            let eps_u = 2.0e-3_f32;
            let mut in_p = inputs2.clone();
            let mut in_m = inputs2.clone();
            in_p[t * 2][0] += eps_u;
            in_m[t * 2][0] -= eps_u;
            let fd_gpu = (eval_gpu_u(&in_p, &[packed2]) - eval_gpu_u(&in_m, &[packed2]))
                / ((in_p[t * 2][0] - in_m[t * 2][0]) as f64);
            let du_gpu = adj2.du[(LOW_FC_FRAMES + t) * CORE_INPUT_VEC4_STRIDE][0] as f64;
            let tol_u = 5.0e-3 * fd_gpu.abs().max(1.0);
            assert!(
                (du_gpu - fd_gpu).abs() <= tol_u,
                "low-fc 8-tile Block 1 ({label}) input t={t} lane=0 FD mismatch: got={du_gpu} fd={fd_gpu}"
            );
        }
        let set_gk_coupled =
            |p: &mut [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE], slot: usize, val: f32| {
                p[slot][0] = val;
                if slot == quad_param_slots::SVF_G || slot == quad_param_slots::SVF_K {
                    let g = p[quad_param_slots::SVF_G][0];
                    let k = p[quad_param_slots::SVF_K][0];
                    let a1 = 1.0_f32 / (1.0 + g * (g + k));
                    let a2 = g * a1;
                    let a3 = g * a2;
                    p[quad_param_slots::SVF_A1][0] = a1;
                    p[quad_param_slots::SVF_A2][0] = a2;
                    p[quad_param_slots::SVF_A3][0] = a3;
                    p[quad_param_slots::SVF_TWO_K][0] = 2.0 * k;
                    p[quad_param_slots::TWO_SVF_G][0] = 2.0 * g;
                }
            };
        for &(slot, eps) in &[
            (quad_param_slots::SVF_A1, 3.0e-5_f32),
            (quad_param_slots::TWO_SVF_G, 5.0e-4_f32),
            (quad_param_slots::SVF_K, 1.0e-3_f32),
            (quad_param_slots::LRU_ALPHA_REL, 1.5e-3_f32),
        ] {
            let mut pp = packed2;
            let mut pm = packed2;
            set_gk_coupled(&mut pp, slot, packed2[slot][0] + eps);
            set_gk_coupled(&mut pm, slot, packed2[slot][0] - eps);
            let fd = (eval_gpu_u(&inputs2, &[pp]) - eval_gpu_u(&inputs2, &[pm]))
                / ((pp[slot][0] - pm[slot][0]) as f64);
            let mut sum_gpu = 0.0_f64;
            for tile in 0..LOW_FC_TILES {
                let idx = (LOW_FC_TILES + tile) * QUAD_PARAMS_VEC4_STRIDE + slot;
                sum_gpu += adj2.param_grads[idx][0] as f64;
            }
            let tol = 5.0e-3 * fd.abs().max(1.0);
            assert!(
                (sum_gpu - fd).abs() <= tol,
                "low-fc 8-tile Block 1 ({label}) slot={slot} lane=0 FD mismatch: got={sum_gpu} fd={fd}"
            );
        }
    }
}
