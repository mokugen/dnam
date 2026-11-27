#![forbid(unsafe_code)]

//! SONA v4.0 hardware parity & finite-difference verification (`ref[dsp.op-n.polyadaa.clamp]`,
//! `ref[dsp.op-n.core.interblock-tiles]`, `ref[dsp.op-n.core.tpt-tone-basis]`):
//! - `|r_curr_unclamped| > 4096.0` straight-through pre-clamp mask across `255 <-> 256` tile seams
//! - `Slot 8` `SE-Bus` + `g_pre` + `lin_mix` + `out_gain` + `glu_mix` FD on fast paths & `Slot 5` fallback
//! - Native `[0, 1]` `adaa_even` and high-`r` GL4 / Rational-Möbius `AlgADAA1` backward FD
//! - `Pred-Corr PHS` (`c_kbp != 0` & `k_nl == 0` parity across `255 <-> 256` seams)

use super::core_and_adjoint::eval_cpu_features_loss;
use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, build_core_quad_bundle_with_k_nl,
    cpu_params_from_packed, dispatch_adjoint_3pass, dispatch_kernel, read_vec4_vec,
};
use dnam_kernels::{
    CoreQuadV4Block0StateStereo8, CoreQuadV4ParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4,
    MAX_ALG_R, SOFT_RECT_EPS, SonaBlockStateStereo8, StereoVec8, TINY_CROSSING_EPS, TptSvfCoeffs4,
    step_core_quad_v4_phs_stereo8, step_core_quad_v4_stereo8,
};
use sona_shaders::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
    QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, quad_param_slots,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

const NUM_FRAMES: usize = 512;
const NUM_TILES: usize = 2;

type FwdDispatchResult = (Vec<[f32; 4]>, Vec<[f32; 4]>, Vec<u8>, Vec<u8>);
type SeModeSpec = (&'static str, [f32; 4], [f32; 4], [f32; 4]);

fn dispatch_fwd_single(
    ctx: &GpuContext,
    variant: ShaderVariant,
    inputs: &[[f32; 4]],
    packed: &[[f32; 4]],
    block_idx: u32,
) -> FwdDispatchResult {
    let num_frames = inputs.len() / CORE_INPUT_VEC4_STRIDE;
    let num_tiles = num_frames.div_ceil(256);
    let ckpt_init = vec![[0.0_f32; 4]; num_tiles * TILE_CKPT_VEC4_STRIDE];
    let feat_init = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
    let u = FusedScanUniforms {
        num_frames: num_frames as u32,
        num_tiles: num_tiles as u32,
        num_quads: 1,
        num_stems: 1,
        tile_start: 0,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        block_idx,
    };
    let (out, _) = dispatch_kernel(
        ctx,
        KernelId::FusedCoreScan256,
        variant,
        (1, 1, 1),
        u.as_bytes(),
        &[
            (false, as_byte_slice(inputs)),
            (false, as_byte_slice(packed)),
            (true, as_byte_slice(&ckpt_init)),
            (true, as_byte_slice(&feat_init)),
        ],
    );
    (
        read_vec4_vec(&out[0]),
        read_vec4_vec(&out[1]),
        out[0].clone(),
        out[1].clone(),
    )
}

fn eval_phs_loss_f64(
    inputs: &[[f64; 4]],
    packed: &[[f32; 4]],
    weights: &[[f32; 4]],
    num_frames: usize,
) -> f64 {
    let g = packed[quad_param_slots::SVF_G].map(f64::from);
    let k = packed[quad_param_slots::SVF_K].map(f64::from);
    let k_nl = packed[quad_param_slots::SVF_K_NL].map(f64::from);
    let c_lp = packed[quad_param_slots::TONE_C_LP].map(f64::from);
    let c_bp = packed[quad_param_slots::TONE_C_BP].map(f64::from);
    let c_kbp = packed[quad_param_slots::TONE_C_KBP].map(f64::from);
    let c_u = packed[quad_param_slots::TONE_C_U].map(f64::from);
    let a_rel = packed[quad_param_slots::LRU_ALPHA_REL].map(f64::from);
    let a_atk = packed[quad_param_slots::LRU_ALPHA_ATK].map(f64::from);
    let kappa = packed[quad_param_slots::LRU_KAPPA].map(f64::from);
    let gamma = packed[quad_param_slots::LRU_GAMMA_DUHEM].map(f64::from);
    let eta = packed[quad_param_slots::LRU_SIGNED_MIX].map(f64::from);
    let slink = packed[quad_param_slots::LRU_STEREO_LINK].map(f64::from);
    let rect_sign = packed[quad_param_slots::RECT_SIGN].map(f64::from);
    let g0 = packed[quad_param_slots::G0].map(f64::from);
    let g_inf = packed[quad_param_slots::G_INF].map(f64::from);
    let d_pre = packed[quad_param_slots::D_PRE].map(f64::from);
    let se_w_sq = packed[quad_param_slots::SE_W_SQ].map(f64::from);
    let se_d_bus = packed[quad_param_slots::SE_D_BUS].map(f64::from);
    let adaa_drive = packed[quad_param_slots::ADAA_DRIVE].map(f64::from);
    let lru_bias = packed[quad_param_slots::LRU_BIAS_DRIVE].map(f64::from);
    let lin_mix = packed[quad_param_slots::LIN_MIX].map(f64::from);
    let wf_mix = packed[quad_param_slots::WAVEFOLDER_MIX].map(f64::from);
    let glu_mix = packed[quad_param_slots::GLU_MIX].map(f64::from);
    let out_gain = packed[quad_param_slots::OUT_GAIN].map(f64::from);

    let mut s1_a = [[0.0_f64; 4]; 2];
    let mut s2_a = [[0.0_f64; 4]; 2];
    let mut s1_b = [[0.0_f64; 4]; 2];
    let mut s2_b = [[0.0_f64; 4]; 2];
    let mut lru_v = [[0.0_f64; 4]; 2];
    let mut adaa_prev = [[0.0_f64; 4]; 2];
    let eps_rect = f64::from(SOFT_RECT_EPS);
    let mut loss = 0.0_f64;

    for t in 0..num_frames {
        let mut tone = [[0.0_f64; 4]; 2];
        let mut bp_arr = [[0.0_f64; 4]; 2];
        let mut v_raw = [[0.0_f64; 4]; 2];
        for ch in 0..2 {
            let u_ch = inputs[t * 2 + ch];
            for l in 0..4 {
                let u = u_ch[l];
                let a1_0 = 1.0 / (1.0 + g[l] * (g[l] + k[l]));
                let a2_0 = g[l] * a1_0;
                let a3_0 = g[l] * a2_0;
                let v3_a = u - s2_a[ch][l];
                let v1_lin = a1_0 * s1_a[ch][l] + a2_0 * v3_a;
                let v2_a = s2_a[ch][l] + a2_0 * s1_a[ch][l] + a3_0 * v3_a;
                s1_a[ch][l] = 2.0 * v1_lin - s1_a[ch][l];
                s2_a[ch][l] = 2.0 * v2_a - s2_a[ch][l];

                let e = (v1_lin * v1_lin).min(64.0);
                let k_eff = k[l] + k_nl[l] * e;
                let a1_b = 1.0 / (1.0 + g[l] * (g[l] + k_eff));
                let a2_b = g[l] * a1_b;
                let a3_b = g[l] * a2_b;
                let v3_b = u - s2_b[ch][l];
                let bp = a1_b * s1_b[ch][l] + a2_b * v3_b;
                let lp = s2_b[ch][l] + a2_b * s1_b[ch][l] + a3_b * v3_b;
                s1_b[ch][l] = 2.0 * bp - s1_b[ch][l];
                s2_b[ch][l] = 2.0 * lp - s2_b[ch][l];

                let tn = c_lp[l] * lp + (c_bp[l] + c_kbp[l] * k_eff) * bp + c_u[l] * u;
                tone[ch][l] = tn;
                bp_arr[ch][l] = bp;

                let flux_abs = (2.0 * g[l] * bp).abs();
                let rho = 1.0 - a_atk[l] / (1.0 + gamma[l] * flux_abs);
                let d = a_rel[l] * lru_v[ch][l];
                let z = rect_sign[l] * tn - kappa[l] * (1.0 - eta[l]) * d;
                let zb = z.clamp(-1.0, 1.0);
                let psi1 = zb * (1.0 - zb * zb / 3.0);
                let tr = (z / (2.0 * eps_rect)).clamp(0.0, 1.0);
                let phi = eps_rect * tr * tr * tr * (2.0 - tr) + (z - 2.0 * eps_rect).max(0.0);
                let phi_eta = if eta[l] <= 0.0 {
                    phi
                } else {
                    (1.0 - eta[l]) * phi + eta[l] * psi1
                };
                let vn = (1.0 - eta[l] * rho) * d + rho * phi_eta;
                lru_v[ch][l] = vn;
                v_raw[ch][l] = vn;
            }
        }
        let mut v_eff = [[0.0_f64; 4]; 2];
        let mut g_pre = [[0.0_f64; 4]; 2];
        let mut a_odd_wf = [[0.0_f64; 4]; 2];
        let mut a_even = [[0.0_f64; 4]; 2];
        for l in 0..4 {
            let vm = 0.5 * (v_raw[0][l] + v_raw[1][l]);
            v_eff[0][l] = v_raw[0][l] + slink[l] * (vm - v_raw[0][l]);
            v_eff[1][l] = v_raw[1][l] + slink[l] * (vm - v_raw[1][l]);
        }
        for ch in 0..2 {
            let mut p_bus = 0.0_f64;
            for l in 0..4 {
                p_bus += se_w_sq[l] * v_eff[ch][l].abs().min(1.0e12);
            }
            let mut bounded_odd = [0.0_f64; 4];
            let mut g_se = [0.0_f64; 4];
            for l in 0..4 {
                let de = d_pre[l] * v_eff[ch][l] * v_eff[ch][l];
                g_pre[ch][l] = (g0[l] + g_inf[l] * de) / (1.0 + de);
                let r_c = ((g_pre[ch][l] * adaa_drive[l]) * tone[ch][l]
                    + lru_bias[l] * v_eff[ch][l])
                    .clamp(-4096.0, 4096.0);
                let r_p = adaa_prev[ch][l];
                adaa_prev[ch][l] = r_c;
                let sp = (1.0 + r_p * r_p).sqrt();
                let sn = (1.0 + r_c * r_c).sqrt();
                let alg_o = if r_p == 0.0 && r_c == 0.0 {
                    0.0
                } else {
                    (((r_c + r_p) / (sn + sp)) * (1.0 + 0.5 / (sn * sp))).clamp(-1.0, 1.0)
                };
                let alg_e = if r_p == 0.0 && r_c == 0.0 {
                    0.0
                } else if (r_c - r_p).abs() > 1.0e-7 * (1.0 + r_c.abs().max(r_p.abs())) {
                    (((r_c - r_c / sn) - (r_p - r_p / sp)) / (r_c - r_p)).clamp(0.0, 1.0)
                } else {
                    let tp = (r_p * r_p) / (sp + 1.0);
                    let tn = (r_c * r_c) / (sn + 1.0);
                    let ts = tn + tp;
                    ((2.0 * ts + ts * ts + tn * tp * (1.0 + ts) + r_c * r_p)
                        / ((sn + sp) * sn * sp))
                        .clamp(0.0, 1.0)
                };
                let wf_o = if (r_c - r_p).abs() > 1.0e-9 {
                    let wp = -0.125 * (1.0 - r_p.clamp(-1.0, 1.0).powi(2)).powi(4);
                    let wn = -0.125 * (1.0 - r_c.clamp(-1.0, 1.0).powi(2)).powi(4);
                    (wn - wp) / (r_c - r_p)
                } else {
                    let m = 0.5 * (r_p + r_c);
                    if m.abs() < 1.0 {
                        m * (1.0 - m * m).powi(3)
                    } else {
                        0.0
                    }
                };
                a_odd_wf[ch][l] = (1.0 - wf_mix[l]) * alg_o + wf_mix[l] * wf_o;
                a_even[ch][l] = alg_e;
                g_se[l] = 1.0 / (1.0 + se_d_bus[l] * p_bus);
                bounded_odd[l] = g_se[l] * a_odd_wf[ch][l];
            }
            let wb = t * FEATURE_VEC4_STRIDE;
            for l in 0..4 {
                let o_t = out_gain[l]
                    * g_se[l]
                    * ((1.0 - lin_mix[l]) * a_odd_wf[ch][l]
                        + lin_mix[l] * g_pre[ch][l] * tone[ch][l]);
                let rot = bounded_odd[(l + 1) & 3];
                let e_t = out_gain[l]
                    * (1.0 - lin_mix[l])
                    * (g_se[l] * a_even[ch][l] + glu_mix[l] * bounded_odd[l] * rot);
                loss += v_eff[ch][l] * f64::from(weights[wb + ch][l])
                    + o_t * f64::from(weights[wb + 2 + ch][l])
                    + e_t * f64::from(weights[wb + 4 + ch][l]);
            }
        }
    }
    loss
}

fn eval_phs_loss_single(
    inputs: &[[f32; 4]],
    cpu_p: &CoreQuadV4ParamsStereo4<GpuVec4>,
    weights: &[[f32; 4]],
    num_frames: usize,
) -> f64 {
    let mut st = CoreQuadV4Block0StateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut loss = 0.0_f64;
    for t in 0..num_frames {
        let ib = t * CORE_INPUT_VEC4_STRIDE;
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(inputs[ib]),
            GpuVec4::from_array(inputs[ib + 1]),
        );
        let out = step_core_quad_v4_phs_stereo8(u, &mut st, cpu_p);
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
        let wb = t * FEATURE_VEC4_STRIDE;
        for (fi, a) in arrs.iter().enumerate() {
            for l in 0..4 {
                loss += (a[l] as f64) * (weights[wb + fi][l] as f64);
            }
        }
    }
    loss
}

fn verify_preclamp_seam_mask(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let (_, mut packed) = build_core_quad_bundle_with_k_nl(0, 0, [0.0; 4]);
    // Pure direct tone pass-through (c_lp=0, c_bp=0, c_kbp=0, c_u=1) + g0=g_inf=2.5 + adaa_drive=32.0
    // lets us force exact |r_unclamped| > 4096 at t=255 and t=256 independently.
    packed[quad_param_slots::TONE_C_LP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_BP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_KBP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_U] = [1.0; 4];
    packed[quad_param_slots::G0] = [2.5; 4];
    packed[quad_param_slots::G_INF] = [2.5; 4];
    packed[quad_param_slots::D_PRE] = [0.0; 4];
    packed[quad_param_slots::LRU_ALPHA_REL] = [0.5; 4];
    packed[quad_param_slots::LRU_BIAS_DRIVE] = [0.0; 4];
    packed[quad_param_slots::ADAA_DRIVE] = [32.0; 4];
    packed[quad_param_slots::LIN_MIX] = [0.0; 4];
    packed[quad_param_slots::WAVEFOLDER_MIX] = [0.0, 0.25, 0.5, 1.0];
    packed[quad_param_slots::GLU_MIX] = [1.0; 4];
    packed[quad_param_slots::OUT_GAIN] = [1.0; 4];
    let cpu_p = cpu_params_from_packed(&packed);

    // g_pre * adaa_drive = 2.5 * 32.0 = 80.0 => |r_unclamped| > 4096 whenever |u| > 51.2.
    let mut inputs = vec![[0.0_f32; 4]; NUM_FRAMES * CORE_INPUT_VEC4_STRIDE];
    for t in 0..NUM_FRAMES {
        let ph = (t as f32) * 0.05;
        let base = if t == 255 || t == 256 {
            // 80.0 * 60.0 = 4800.0 > 4096.0 (saturated!).
            [60.0_f32, -65.0, 58.0, 72.0]
        } else if t == 254 || t == 257 {
            // 80.0 * 10.0 = 800.0 (unclamped high-r).
            [10.0_f32, -12.0, 14.0, -9.0]
        } else {
            [0.4 * ph.sin(), 0.3 * ph.cos(), -0.35 * ph.sin(), 0.25]
        };
        inputs[t * 2] = base;
        inputs[t * 2 + 1] = base.map(|v| -0.9 * v);
    }
    const {
        assert!(2.5 * 32.0 * 60.0 > MAX_ALG_R);
    }

    let (_ckpts, gpu_feats, ckpt_bytes, feat_bytes) =
        dispatch_fwd_single(ctx, variant, &inputs, &packed, BLOCK_IDX_DIRECT_LINEAR);

    let mut st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    for t in 0..NUM_FRAMES {
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(inputs[t * 2]),
            GpuVec4::from_array(inputs[t * 2 + 1]),
        );
        let out = step_core_quad_v4_stereo8(u, &mut st, &cpu_p);
        let (ol, _) = out.o_tilde.into_lr();
        let (el, _) = out.e_tilde.into_lr();
        for l in 0..4 {
            assert!((gpu_feats[t * 6 + 2][l] - ol.to_array()[l]).abs() < 1.0e-5);
            assert!((gpu_feats[t * 6 + 4][l] - el.to_array()[l]).abs() < 1.0e-5);
        }
    }

    let mut d_loss = vec![[0.0_f32; 4]; NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for (idx, row) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.021;
        *row = [
            0.3 * ph.sin(),
            -0.25 * ph.cos(),
            0.35 * (1.1 * ph).sin(),
            0.28 * (0.7 * ph).cos(),
        ];
    }

    let adj = dispatch_adjoint_3pass(
        ctx,
        variant,
        NUM_FRAMES,
        NUM_TILES,
        1,
        1,
        1,
        BLOCK_IDX_DIRECT_LINEAR,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs),
        as_byte_slice(&packed),
        &ckpt_bytes,
        as_byte_slice(&d_loss),
        &feat_bytes,
        None,
    );

    let params_slice = [cpu_p];
    for &t in &[17_usize, 254, 255, 256, 257] {
        let eps = if t == 255 || t == 256 { 1.0e-2 } else { 2.0e-3 };
        for lane in 0..4 {
            let mut pp = inputs.clone();
            let mut pm = inputs.clone();
            pp[t * 2][lane] += eps;
            pm[t * 2][lane] -= eps;
            let fd = (eval_cpu_features_loss(&pp, &params_slice, &d_loss, 0, NUM_FRAMES)
                - eval_cpu_features_loss(&pm, &params_slice, &d_loss, 0, NUM_FRAMES))
                / ((pp[t * 2][lane] - pm[t * 2][lane]) as f64);
            let got = adj.du[t * 2][lane] as f64;
            let tol = 3.0e-3 * fd.abs().max(1.0);
            assert!(
                (got - fd).abs() <= tol,
                "preclamp ({label}) du FD mismatch t={t} lane={lane}: got={got} fd={fd}"
            );
        }
    }
}

fn verify_se_bus_and_glu_modes(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let mut inputs = vec![[0.0_f32; 4]; NUM_FRAMES * CORE_INPUT_VEC4_STRIDE];
    for t in 0..NUM_FRAMES {
        let ph = (t as f32) * 0.071;
        inputs[t * 2] = [
            0.42 * ph.sin(),
            -0.38 * (1.2 * ph).cos(),
            0.35 * (0.8 * ph).sin(),
            0.29 * (1.4 * ph).cos(),
        ];
        inputs[t * 2 + 1] = [
            -0.36 * (0.9 * ph).cos(),
            0.44 * (1.1 * ph).sin(),
            -0.31 * (1.3 * ph).cos(),
            0.39 * (0.7 * ph).sin(),
        ];
    }

    // Exercise Fast Path 1 (glu_mix == 0, out_gain[1] == 0), Fast Path 2 (lin_mix == 0, glu_mix != 0),
    // and General Slot 5 Fallback (glu_mix != 0 AND lin_mix != 0 AND tiny/zero out_gain).
    let modes: [SeModeSpec; 3] = [
        (
            "fast_path_glu_zero",
            [0.0, 0.25, 0.55, 0.85],
            [0.0; 4],
            [1.0, 0.0, -1.5, 0.75],
        ),
        (
            "fast_path_lin_zero",
            [0.0; 4],
            [0.8, -0.65, 1.2, -0.9],
            [1.1, -1.35, 0.8, -1.5],
        ),
        (
            "slot5_fallback_general",
            [0.2, 0.0, 0.55, 0.9],
            [0.75, -0.5, 1.1, -0.8],
            [0.0, -1.2, 1.0e-7, 1.75],
        ),
    ];

    let mut d_loss = vec![[0.0_f32; 4]; NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for (idx, row) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.013;
        *row = [
            0.25 * ph.sin(),
            0.22 * (1.1 * ph).cos(),
            -0.27 * (0.8 * ph).sin(),
            0.31 * (1.4 * ph).cos(),
        ];
    }

    for (mode_name, lin_mix, glu_mix, out_gain) in modes {
        let (_, mut packed) = build_core_quad_bundle_with_k_nl(0, 0, [0.0; 4]);
        packed[quad_param_slots::LIN_MIX] = lin_mix;
        packed[quad_param_slots::GLU_MIX] = glu_mix;
        packed[quad_param_slots::OUT_GAIN] = out_gain;
        let cpu_p = cpu_params_from_packed(&packed);

        let (_ckpts, _feats, ckpt_bytes, feat_bytes) =
            dispatch_fwd_single(ctx, variant, &inputs, &packed, BLOCK_IDX_DIRECT_LINEAR);
        let adj = dispatch_adjoint_3pass(
            ctx,
            variant,
            NUM_FRAMES,
            NUM_TILES,
            1,
            1,
            1,
            BLOCK_IDX_DIRECT_LINEAR,
            SOFT_RECT_EPS,
            TINY_CROSSING_EPS,
            as_byte_slice(&inputs),
            as_byte_slice(&packed),
            &ckpt_bytes,
            as_byte_slice(&d_loss),
            &feat_bytes,
            None,
        );

        let params_slice = [cpu_p];
        for &t in &[17_usize, 255, 256, 511] {
            let mut pp = inputs.clone();
            let mut pm = inputs.clone();
            pp[t * 2][0] += 2.0e-3;
            pm[t * 2][0] -= 2.0e-3;
            let fd = (eval_cpu_features_loss(&pp, &params_slice, &d_loss, 0, NUM_FRAMES)
                - eval_cpu_features_loss(&pm, &params_slice, &d_loss, 0, NUM_FRAMES))
                / ((pp[t * 2][0] - pm[t * 2][0]) as f64);
            let got = adj.du[t * 2][0] as f64;
            let tol = 3.0e-3 * fd.abs().max(1.0);
            assert!(
                (got - fd).abs() <= tol,
                "{mode_name} ({label}) du FD mismatch t={t}: got={got} fd={fd}"
            );
        }

        for &(slot, eps) in &[
            (quad_param_slots::G0, 1.0e-3_f32),
            (quad_param_slots::G_INF, 1.0e-3_f32),
            (quad_param_slots::D_PRE, 1.0e-3_f32),
            (quad_param_slots::SE_W_SQ, 1.0e-3_f32),
            (quad_param_slots::SE_D_BUS, 1.0e-3_f32),
            (quad_param_slots::LIN_MIX, 5.0e-4_f32),
            (quad_param_slots::GLU_MIX, 1.0e-3_f32),
            (quad_param_slots::OUT_GAIN, 1.0e-3_f32),
        ] {
            for lane in 0..4 {
                let mut pp = packed;
                let mut pm = packed;
                let v = packed[slot][lane];
                let (vp, vm) = if slot == quad_param_slots::LIN_MIX && v == 0.0 {
                    (v + eps, 0.0)
                } else {
                    (v + eps, v - eps)
                };
                pp[slot][lane] = vp;
                pm[slot][lane] = vm;
                let fd = (eval_cpu_features_loss(
                    &inputs,
                    &[cpu_params_from_packed(&pp)],
                    &d_loss,
                    0,
                    NUM_FRAMES,
                ) - eval_cpu_features_loss(
                    &inputs,
                    &[cpu_params_from_packed(&pm)],
                    &d_loss,
                    0,
                    NUM_FRAMES,
                )) / ((vp - vm) as f64);
                let mut g_gpu = 0.0_f64;
                for tile in 0..NUM_TILES {
                    g_gpu += adj.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + slot][lane] as f64;
                }
                let tol = 3.0e-3 * fd.abs().max(1.0);
                assert!(
                    (g_gpu - fd).abs() <= tol,
                    "{mode_name} ({label}) param FD mismatch slot={slot} lane={lane}: got={g_gpu} fd={fd}"
                );
            }
        }
    }
}

fn verify_high_r_gl4_mobius_and_native_even(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let (_, mut packed) = build_core_quad_bundle_with_k_nl(0, 0, [0.0; 4]);
    packed[quad_param_slots::TONE_C_LP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_BP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_KBP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_U] = [1.0; 4];
    packed[quad_param_slots::ADAA_DRIVE] = [12.0, 20.0, 28.0, 30.0];
    packed[quad_param_slots::LIN_MIX] = [0.0; 4];
    packed[quad_param_slots::WAVEFOLDER_MIX] = [0.0, 0.3, 0.7, 1.0];
    packed[quad_param_slots::GLU_MIX] = [1.5, -1.25, 1.75, -1.5];
    packed[quad_param_slots::OUT_GAIN] = [1.0; 4];
    let cpu_p = cpu_params_from_packed(&packed);

    let mut inputs = vec![[0.0_f32; 4]; NUM_FRAMES * CORE_INPUT_VEC4_STRIDE];
    for t in 0..NUM_FRAMES {
        let ph = (t as f32) * 0.03;
        // Same-sign large positive values & slow GL4 narrow spans as well as wide Möbius spans (5..600).
        let slow = 18.0 + 4.0 * ph.sin();
        let wide = 4.0 + 35.0 * (0.5 * ph).cos().abs();
        inputs[t * 2] = [slow, wide, -slow, -wide];
        inputs[t * 2 + 1] = [wide, -slow, wide, -slow];
    }

    let (_ckpts, gpu_feats, ckpt_bytes, feat_bytes) =
        dispatch_fwd_single(ctx, variant, &inputs, &packed, BLOCK_IDX_DIRECT_LINEAR);

    let mut st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    for t in 0..NUM_FRAMES {
        let u = InterleavedStereo8::from_lr(
            GpuVec4::from_array(inputs[t * 2]),
            GpuVec4::from_array(inputs[t * 2 + 1]),
        );
        let out = step_core_quad_v4_stereo8(u, &mut st, &cpu_p);
        let (el, _) = out.e_tilde.into_lr();
        for l in 0..4 {
            assert!((gpu_feats[t * 6 + 4][l] - el.to_array()[l]).abs() < 1.0e-4);
        }
    }

    let mut d_loss = vec![[0.0_f32; 4]; NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for (idx, row) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.015;
        *row = [
            0.2 * ph.sin(),
            0.25 * ph.cos(),
            -0.22 * ph.sin(),
            0.24 * ph.cos(),
        ];
    }
    let adj = dispatch_adjoint_3pass(
        ctx,
        variant,
        NUM_FRAMES,
        NUM_TILES,
        1,
        1,
        1,
        BLOCK_IDX_DIRECT_LINEAR,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs),
        as_byte_slice(&packed),
        &ckpt_bytes,
        as_byte_slice(&d_loss),
        &feat_bytes,
        None,
    );

    let params_slice = [cpu_p];
    for &t in &[12_usize, 255, 256, 340] {
        for lane in 0..4 {
            let mut pp = inputs.clone();
            let mut pm = inputs.clone();
            // Scale step to high-amplitude input (`|u| in 4..39`) so `f32` `F_even(r)` ULP cancellation stays < 1e-4.
            let eps_u = 1.5e-2_f32;
            pp[t * 2][lane] += eps_u;
            pm[t * 2][lane] -= eps_u;
            let fd = (eval_cpu_features_loss(&pp, &params_slice, &d_loss, 0, NUM_FRAMES)
                - eval_cpu_features_loss(&pm, &params_slice, &d_loss, 0, NUM_FRAMES))
                / ((pp[t * 2][lane] - pm[t * 2][lane]) as f64);
            let got = adj.du[t * 2][lane] as f64;
            let tol = 3.0e-3 * fd.abs().max(1.0);
            assert!(
                (got - fd).abs() <= tol,
                "high-r GL4/Mobius ({label}) du FD mismatch t={t} lane={lane}: got={got} fd={fd}"
            );
        }
    }
}

fn verify_pred_corr_phs_kbp_and_knl_zero(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let (_, mut packed) = build_core_quad_bundle_with_k_nl(0, 0, [0.4, 1.1, 2.2, 3.6]);
    packed[quad_param_slots::TONE_C_KBP] = [0.75, -0.60, 0.90, -0.85];
    let cpu_p = cpu_params_from_packed(&packed);

    let mut inputs = vec![[0.0_f32; 4]; NUM_FRAMES * CORE_INPUT_VEC4_STRIDE];
    for t in 0..NUM_FRAMES {
        let ph = (t as f32) * 0.067;
        inputs[t * 2] = [
            1.4 * ph.sin(),
            0.9 * (1.2 * ph).cos(),
            -1.1 * (0.7 * ph).sin(),
            0.8 * (1.5 * ph).cos(),
        ];
        inputs[t * 2 + 1] = [
            -(0.9 * ph).cos(),
            1.3 * (1.1 * ph).sin(),
            0.85 * (1.3 * ph).cos(),
            -0.95 * ph.sin(),
        ];
    }

    let (_ckpts, gpu_feats, ckpt_bytes, feat_bytes) =
        dispatch_fwd_single(ctx, variant, &inputs, &packed, 0);
    let mut d_loss = vec![[0.0_f32; 4]; NUM_FRAMES * FEATURE_VEC4_STRIDE];
    for (idx, row) in d_loss.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.017;
        *row = [
            0.22 * ph.sin(),
            0.2 * ph.cos(),
            -0.19 * ph.sin(),
            0.24 * ph.cos(),
        ];
    }
    let adj = dispatch_adjoint_3pass(
        ctx,
        variant,
        NUM_FRAMES,
        NUM_TILES,
        1,
        1,
        1,
        0,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs),
        as_byte_slice(&packed),
        &ckpt_bytes,
        as_byte_slice(&d_loss),
        &feat_bytes,
        None,
    );

    let _ = gpu_feats;
    for &t in &[15_usize, 255, 256, 400] {
        let mut pp = inputs.clone();
        let mut pm = inputs.clone();
        pp[t * 2][1] += 2.0e-3;
        pm[t * 2][1] -= 2.0e-3;
        let fd = (eval_phs_loss_single(&pp, &cpu_p, &d_loss, NUM_FRAMES)
            - eval_phs_loss_single(&pm, &cpu_p, &d_loss, NUM_FRAMES))
            / ((pp[t * 2][1] - pm[t * 2][1]) as f64);
        let got = adj.du[t * 2][1] as f64;
        let tol = 4.0e-3 * fd.abs().max(1.0);
        assert!(
            (got - fd).abs() <= tol,
            "Pred-Corr PHS c_kbp ({label}) du FD mismatch t={t}: got={got} fd={fd}"
        );
    }
    for &(slot, eps) in &[
        (quad_param_slots::SVF_K, 1.0e-3_f32),
        (quad_param_slots::SVF_K_NL, 1.0e-3_f32),
        (quad_param_slots::SVF_G, 5.0e-4_f32),
        (quad_param_slots::TONE_C_KBP, 1.0e-3_f32),
    ] {
        for lane in 0..4 {
            let mut pp = packed;
            let mut pm = packed;
            pp[slot][lane] += eps;
            pm[slot][lane] -= eps;
            let fd =
                (eval_phs_loss_single(&inputs, &cpu_params_from_packed(&pp), &d_loss, NUM_FRAMES)
                    - eval_phs_loss_single(
                        &inputs,
                        &cpu_params_from_packed(&pm),
                        &d_loss,
                        NUM_FRAMES,
                    ))
                    / ((pp[slot][lane] - pm[slot][lane]) as f64);
            let mut g_gpu = 0.0_f64;
            for tile in 0..NUM_TILES {
                g_gpu += adj.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + slot][lane] as f64;
            }
            let tol = 4.0e-3 * fd.abs().max(1.0);
            assert!(
                (g_gpu - fd).abs() <= tol,
                "Pred-Corr PHS c_kbp ({label}) param FD mismatch slot={slot} lane={lane}: got={g_gpu} fd={fd}"
            );
        }
    }
}

fn verify_pred_corr_phs_3tile_alpha_rel_0999_tail_loss(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let num_frames = 768_usize;
    let num_tiles = 3_usize;
    let (_, mut packed) = build_core_quad_bundle_with_k_nl(0, 0, [0.5, 1.2, 2.0, 3.0]);
    let g_arr = [0.04_f32; 4];
    let k_arr = [0.25_f32; 4];
    let svf = TptSvfCoeffs4::from_gk(GpuVec4::from_array(g_arr), GpuVec4::from_array(k_arr));
    packed[quad_param_slots::SVF_A1] = svf.a1.to_array();
    packed[quad_param_slots::SVF_A2] = svf.a2.to_array();
    packed[quad_param_slots::SVF_A3] = svf.a3.to_array();
    packed[quad_param_slots::SVF_G] = g_arr;
    packed[quad_param_slots::SVF_K] = k_arr;
    packed[quad_param_slots::TONE_C_LP] = [1.0; 4];
    packed[quad_param_slots::TONE_C_BP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_KBP] = [0.0; 4];
    packed[quad_param_slots::TONE_C_U] = [0.0; 4];
    packed[quad_param_slots::LRU_ALPHA_REL] = [0.999; 4];
    let cpu_p = cpu_params_from_packed(&packed);

    let mut inputs = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
    for t in 0..num_frames {
        let ph = (t as f32) * 0.053;
        inputs[t * 2] = [
            3.5 * ph.sin(),
            3.0 * (1.1 * ph).cos(),
            -2.8 * (0.8 * ph).sin(),
            3.2 * (1.4 * ph).cos(),
        ];
        inputs[t * 2 + 1] = [
            -3.1 * (0.9 * ph).cos(),
            2.9 * (1.2 * ph).sin(),
            3.4 * (0.7 * ph).cos(),
            -2.7 * (1.3 * ph).sin(),
        ];
    }

    let (_ckpts, _gpu_feats, ckpt_bytes, feat_bytes) =
        dispatch_fwd_single(ctx, variant, &inputs, &packed, 0);

    // Zero loss on tiles 0 and 1 so all tile-0 sensitivity routes via tile-1 cross-tile carry.
    let mut d_loss = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
    for t in 512..num_frames {
        for fi in 0..FEATURE_VEC4_STRIDE {
            let ph = ((t * FEATURE_VEC4_STRIDE + fi) as f32) * 0.019;
            d_loss[t * FEATURE_VEC4_STRIDE + fi] = [
                0.30 * ph.sin(),
                -0.25 * (1.2 * ph).cos(),
                0.28 * (0.7 * ph).sin(),
                0.22 * (1.5 * ph).cos(),
            ];
        }
    }

    let adj = dispatch_adjoint_3pass(
        ctx,
        variant,
        num_frames,
        num_tiles,
        1,
        1,
        1,
        0,
        SOFT_RECT_EPS,
        TINY_CROSSING_EPS,
        as_byte_slice(&inputs),
        as_byte_slice(&packed),
        &ckpt_bytes,
        as_byte_slice(&d_loss),
        &feat_bytes,
        None,
    );

    let inputs_f64: Vec<[f64; 4]> = inputs.iter().map(|r| r.map(f64::from)).collect();
    let eps = 1.0e-6_f64;
    for &t in &[200_usize, 255] {
        for ch in 0..2 {
            let idx = t * 2 + ch;
            let mut max_fd = 0.0_f64;
            for lane in 0..4 {
                let mut pp = inputs_f64.clone();
                let mut pm = inputs_f64.clone();
                pp[idx][lane] += eps;
                pm[idx][lane] -= eps;
                let fd = (eval_phs_loss_f64(&pp, &packed, &d_loss, num_frames)
                    - eval_phs_loss_f64(&pm, &packed, &d_loss, num_frames))
                    / (2.0 * eps);
                max_fd = max_fd.max(fd.abs());
                let got = adj.du[idx][lane] as f64;
                let tol = 3.0e-3 * fd.abs().max(5.0e-2);
                assert!(
                    (got - fd).abs() <= tol,
                    "Pred-Corr PHS 3-tile alpha_rel=0.999 tail-loss ({label}) du FD mismatch t={t} ch={ch} lane={lane}: got={got} fd={fd}"
                );
            }
            assert!(
                max_fd > 1.0e-3,
                "expected nonzero tail-loss sensitivity at t={t} ch={ch}: max_fd={max_fd}"
            );
        }
    }
    let _ = cpu_p;
}

pub fn verify_v4_adjoint_and_clamp_suite(ctx: &GpuContext, variant: ShaderVariant) {
    verify_preclamp_seam_mask(ctx, variant);
    verify_se_bus_and_glu_modes(ctx, variant);
    verify_high_r_gl4_mobius_and_native_even(ctx, variant);
    verify_pred_corr_phs_kbp_and_knl_zero(ctx, variant);
    verify_pred_corr_phs_3tile_alpha_rel_0999_tail_loss(ctx, variant);
}
