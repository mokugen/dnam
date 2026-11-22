//! Block 0 linearly-implicit Port-Hamiltonian (`PHS`) `TPT-SVF` scalar forward recomputation
//! and exact transposed-Jacobian reverse-time adjoint pullback matching WGSL `adjoint_scan_256`
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_TPT_G, SVF_PHS_E_MAX, TptAdjointElem, step_tpt_svf_phs_vec4,
};

use crate::gpu::abi::quad_param_slots;

use super::{FwdFrameState, LaneParams, ReverseCarry, TileGradAccum};

/// Per-frame scalar Block 0 PHS state and coefficients evaluated at `s1_prev`.
#[derive(Copy, Clone, Default)]
pub(in crate::cpu::core_and_adjoint) struct PhsFrameTerms {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k_eff: f32,
    pub e: f32,
    pub gamma: f32,
    pub taps: [f32; 4],
}

/// Evaluates one Block 0 scalar PHS `TPT-SVF` forward step via `dnam_kernels::step_tpt_svf_phs_vec4`,
/// updating `(s1, s2)` and returning `[lp, bp, hp, ap]`.
#[inline(always)]
pub(in crate::cpu::core_and_adjoint) fn step_tpt_svf_phs_scalar(
    u: f32,
    s1: &mut f32,
    s2: &mut f32,
    lp: &LaneParams,
) -> [f32; 4] {
    let mut s1_v = GpuVec4::splat(*s1);
    let mut s2_v = GpuVec4::splat(*s2);
    let out = step_tpt_svf_phs_vec4(
        GpuVec4::splat(u),
        &mut s1_v,
        &mut s2_v,
        GpuVec4::splat(lp.two_svf_g * 0.5),
        GpuVec4::splat(lp.k),
        GpuVec4::splat(lp.svf_k_nl),
    );
    *s1 = s1_v.to_array()[0];
    *s2 = s2_v.to_array()[0];
    [
        out.lp.to_array()[0],
        out.bp.to_array()[0],
        out.hp.to_array()[0],
        out.ap.to_array()[0],
    ]
}

/// Evaluates scalar Block 0 PHS frame coefficients `(a1, a2, a3, k_eff, e, gamma)` alongside
/// `0-ULP` forward taps `[lp, bp, hp, ap]` anchored at `(s1_prev, s2_prev)`.
#[inline(always)]
pub(in crate::cpu::core_and_adjoint) fn eval_phs_frame_terms(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    lp: &LaneParams,
) -> PhsFrameTerms {
    let g_safe = (lp.two_svf_g * 0.5).max(MIN_TPT_G);
    let k_safe = lp.k.max(0.0);
    let k_nl_safe = lp.svf_k_nl.max(0.0);
    let s1_sq = s1_prev * s1_prev;
    let e = s1_sq.min(SVF_PHS_E_MAX);
    let gamma = if s1_sq < SVF_PHS_E_MAX {
        (k_nl_safe + k_nl_safe) * s1_prev
    } else {
        0.0
    };
    let k_eff = k_safe + k_nl_safe * e;
    let denom = 1.0 + g_safe * (g_safe + k_eff);
    let a1 = GpuVec4::splat(denom).rcp_ge1_nr2().to_array()[0];
    let a2 = g_safe * a1;
    let a3 = g_safe * a2;
    let mut s1_tmp = s1_prev;
    let mut s2_tmp = s2_prev;
    let taps = step_tpt_svf_phs_scalar(u, &mut s1_tmp, &mut s2_tmp, lp);
    PhsFrameTerms {
        a1,
        a2,
        a3,
        k_eff,
        e,
        gamma,
        taps,
    }
}

#[inline(always)]
fn build_tpt_phs_elem(
    p: &PhsFrameTerms,
    adj_v1: f32,
    adj_v2: f32,
    k_eff_loc_bar: f32,
) -> TptAdjointElem {
    let two_a2 = p.a2 + p.a2;
    let two_a3 = p.a3 + p.a3;
    let v1_gamma = p.taps[1] * p.gamma;
    TptAdjointElem {
        t00: (p.a1 + p.a1 - 1.0) - two_a2 * v1_gamma,
        t01: two_a2 - two_a3 * v1_gamma,
        t10: -two_a2,
        t11: 1.0 - two_a3,
        g0: p.a1 * adj_v1 + p.a2 * adj_v2 + k_eff_loc_bar * p.gamma,
        g1: -p.a2 * adj_v1 + (1.0 - p.a3) * adj_v2,
    }
}

/// Executes the reverse-time Block 0 PHS `2×2` TPT pullback for `(L, R)` on a single frame,
/// updating `carry.adj_s_out_{l,r}`, accumulating `SVF_K`, `SVF_K_NL`, `SVF_G` (`tpt` term), and
/// returning `(du_l, du_r)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(in crate::cpu::core_and_adjoint) fn step_phs_tpt_reverse(
    fwd: &FwdFrameState,
    lp: &LaneParams,
    phs_l: &PhsFrameTerms,
    phs_r: &PhsFrameTerms,
    adj_v1: [f32; 2],
    adj_v2: [f32; 2],
    adj_hp: [f32; 2],
    adj_ap: [f32; 2],
    adj_u_nonrec: [f32; 2],
    carry: &mut ReverseCarry,
    acc: &mut TileGradAccum,
) -> (f32, f32) {
    let psi_l = carry.adj_s_out_l;
    let psi_r = carry.adj_s_out_r;
    let tot_v1_l = adj_v1[0] + 2.0 * psi_l[0];
    let tot_v2_l = adj_v2[0] + 2.0 * psi_l[1];
    let tot_v1_r = adj_v1[1] + 2.0 * psi_r[0];
    let tot_v2_r = adj_v2[1] + 2.0 * psi_r[1];

    let v1_l = phs_l.taps[1];
    let v1_r = phs_r.taps[1];
    let k_loc_l = (-adj_hp[0] - 2.0 * adj_ap[0]) * v1_l
        - v1_l * (phs_l.a2 * adj_v1[0] + phs_l.a3 * adj_v2[0]);
    let k_loc_r = (-adj_hp[1] - 2.0 * adj_ap[1]) * v1_r
        - v1_r * (phs_r.a2 * adj_v1[1] + phs_r.a3 * adj_v2[1]);

    let k_eff_bar_l = k_loc_l - 2.0 * v1_l * (phs_l.a2 * psi_l[0] + phs_l.a3 * psi_l[1]);
    let k_eff_bar_r = k_loc_r - 2.0 * v1_r * (phs_r.a2 * psi_r[0] + phs_r.a3 * psi_r[1]);
    acc.slots[quad_param_slots::SVF_K] += k_eff_bar_l + k_eff_bar_r;
    acc.slots[quad_param_slots::SVF_K_NL] += k_eff_bar_l * phs_l.e + k_eff_bar_r * phs_r.e;

    let two_g_safe = 2.0 * (lp.two_svf_g * 0.5).max(MIN_TPT_G);
    let v3_l = fwd.u_l - fwd.s2_prev_l;
    let v3_r = fwd.u_r - fwd.s2_prev_r;
    let g_tpt_l = tot_v2_l * v1_l
        + (phs_l.a1 * tot_v1_l + phs_l.a2 * tot_v2_l) * (v3_l - (two_g_safe + phs_l.k_eff) * v1_l);
    let g_tpt_r = tot_v2_r * v1_r
        + (phs_r.a1 * tot_v1_r + phs_r.a2 * tot_v2_r) * (v3_r - (two_g_safe + phs_r.k_eff) * v1_r);
    acc.slots[quad_param_slots::SVF_G] += g_tpt_l + g_tpt_r;

    let du_l = adj_u_nonrec[0] + phs_l.a2 * tot_v1_l + phs_l.a3 * tot_v2_l;
    let du_r = adj_u_nonrec[1] + phs_r.a2 * tot_v1_r + phs_r.a3 * tot_v2_r;
    carry.adj_s_out_l = dnam_kernels::apply_tpt_adjoint(
        build_tpt_phs_elem(phs_l, adj_v1[0], adj_v2[0], k_loc_l),
        psi_l,
    );
    carry.adj_s_out_r = dnam_kernels::apply_tpt_adjoint(
        build_tpt_phs_elem(phs_r, adj_v1[1], adj_v2[1], k_loc_r),
        psi_r,
    );
    (du_l, du_r)
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        CoreQuadParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, SonaBlockStateStereo8,
        SonaParaLruDuhemCoeffs4, SonaToneWeights4, StereoVec8, TptSvfCoeffs4,
        prewarp_tan_pi_f_over_fs_f32, step_core_quad_phs_stereo8,
    };

    use crate::cpu::core_and_adjoint::{run_adjoint_scan_cpu, run_fused_core_scan_cpu};
    use crate::gpu::abi::{
        AdjointScanUniforms, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
        QUAD_PARAMS_VEC4_STRIDE, TILE_FRAMES, quad_param_slots,
    };
    use crate::runner::types::{AdjointScanInput, FusedCoreScanInput, pack_core_quad_params};
    use dnam_kernels::{SOFT_RECT_EPS, SVF_PHS_E_MAX, TINY_CROSSING_EPS};

    #[test]
    fn test_cpu_block0_phs_saturated_asymmetric_central_fd_and_linear_limit() {
        let g = GpuVec4::from_array([0.04, 0.11, 0.19, 0.30].map(prewarp_tan_pi_f_over_fs_f32));
        let k = GpuVec4::from_array([0.50, 0.30, 0.65, 1.10]);
        let k_nl = GpuVec4::from_array([0.02, 0.55, 1.40, 3.80]);
        let params = CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::from_array([0.45, 0.20, -0.15, 0.10]),
                bp: GpuVec4::from_array([0.30, 0.40, 0.25, -0.20]),
                hp: GpuVec4::from_array([-0.18, 0.15, 0.35, 0.40]),
                ap: GpuVec4::from_array([0.12, -0.10, 0.15, 0.10]),
                dir: GpuVec4::from_array([0.15, 0.15, -0.10, 0.22]),
            },
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::from_array([0.88, 0.82, 0.75, 0.66]),
                GpuVec4::from_array([0.25, 0.40, 0.55, 0.70]),
                GpuVec4::from_array([0.15, 0.30, 0.50, 0.75]),
                GpuVec4::from_array([0.12, 0.25, 0.40, 0.55]),
                GpuVec4::from_array([0.20, 0.35, 0.50, 0.65]),
                GpuVec4::from_array([0.10, 0.25, 0.40, 0.55]),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: k_nl,
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: GpuVec4::from_array([0.15, 0.25, 0.35, 0.20]),
            adaa_drive: GpuVec4::from_array([1.2, 1.5, 1.8, 1.3]),
            lru_bias_drive: GpuVec4::from_array([0.12, -0.18, 0.22, -0.14]),
            wavefolder_mix: GpuVec4::from_array([0.15, 0.35, 0.65, 0.90]),
            glu_mix: GpuVec4::from_array([0.50, -0.35, 0.65, -0.40]),
        };
        let packed = pack_core_quad_params(&params);

        let num_frames = 540_usize;
        let num_tiles = num_frames.div_ceil(TILE_FRAMES);
        let mut core_inputs = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
        for t in 0..num_frames {
            let ph = (t as f32) * 0.087;
            let amp_l0 = 26.0 + 6.0 * (0.21 * ph).cos();
            let amp_l = 9.0 + 3.0 * (0.21 * ph).cos();
            core_inputs[t * 2] = [
                amp_l0 * ph.sin(),
                amp_l * (1.2 * ph).cos(),
                0.7 * amp_l * (0.8 * ph).sin(),
                0.5 * amp_l * (1.4 * ph).cos(),
            ];
            core_inputs[t * 2 + 1] = [
                0.35 * (0.9 * ph).cos(),
                0.42 * ph.sin(),
                0.30,
                -0.25 * ph.sin(),
            ];
        }
        let mut d_loss = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
        for (idx, item) in d_loss.iter_mut().enumerate() {
            let ph = (idx as f32) * 0.017;
            *item = [
                0.3 * ph.sin(),
                0.25 * ph.cos(),
                -0.2 * ph.sin(),
                0.35 * ph.cos(),
            ];
        }

        let u_phs = FusedScanUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_quads: 1,
            num_stems: 1,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx: 0,
        };
        let fwd = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u_phs,
            core_inputs: &core_inputs,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        let adj = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_inputs,
            quad_params: &packed,
            tile_checkpoints: &fwd.tile_checkpoints,
            d_loss_d_features: &d_loss,
            feature_outputs: &fwd.feature_outputs,
        })
        .unwrap();

        // Blocks 1..3 ignore nonzero svf_k_nl in buffer and emit exact 0.0 SVF_K_NL grad.
        let mut u_lin = u_phs;
        u_lin.block_idx = 1;
        let fwd_lin = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u_lin,
            core_inputs: &core_inputs,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        let mut packed_zero = packed;
        packed_zero[quad_param_slots::SVF_K_NL] = [0.0; 4];
        let fwd_lin_zero = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u_lin,
            core_inputs: &core_inputs,
            quad_params: &packed_zero,
            initial_checkpoints: None,
        })
        .unwrap();
        assert_eq!(
            fwd_lin
                .tile_checkpoints
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>(),
            fwd_lin_zero
                .tile_checkpoints
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            fwd_lin
                .feature_outputs
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>(),
            fwd_lin_zero
                .feature_outputs
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>()
        );
        assert_ne!(fwd_lin.feature_outputs, fwd.feature_outputs);
        let adj_lin = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 1,
            },
            core_inputs: &core_inputs,
            quad_params: &packed,
            tile_checkpoints: &fwd_lin.tile_checkpoints,
            d_loss_d_features: &d_loss,
            feature_outputs: &fwd_lin.feature_outputs,
        })
        .unwrap();
        for tile in 0..num_tiles {
            assert_eq!(
                adj_lin.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + quad_param_slots::SVF_K_NL]
                    .map(f32::to_bits),
                [0; 4]
            );
        }

        let eval_loss = |in_buf: &[[f32; 4]], p: &CoreQuadParamsStereo4<GpuVec4>| -> f64 {
            let mut st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            let mut sum = 0.0_f64;
            for t in 0..num_frames {
                let u = InterleavedStereo8::from_lr(
                    GpuVec4::from_array(in_buf[t * 2]),
                    GpuVec4::from_array(in_buf[t * 2 + 1]),
                );
                let out = step_core_quad_phs_stereo8(u, &mut st, p);
                let (vl, vr) = out.lru_state.into_lr();
                let (ol, or) = out.adaa_odd.into_lr();
                let (el, er) = out.adaa_even.into_lr();
                let f = [
                    vl.to_array(),
                    vr.to_array(),
                    ol.to_array(),
                    or.to_array(),
                    el.to_array(),
                    er.to_array(),
                ];
                for s in 0..6 {
                    for l in 0..4 {
                        sum += f64::from(f[s][l]) * f64::from(d_loss[t * 6 + s][l]);
                    }
                }
            }
            sum
        };

        let mut check_st = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
        let (mut saw_sat, mut saw_unsat) = (false, false);
        for t in 0..num_frames {
            let (s1_l, _) = check_st.svf_s1.into_lr();
            let s1_sq = s1_l.to_array()[0] * s1_l.to_array()[0];
            if s1_sq >= SVF_PHS_E_MAX {
                saw_sat = true;
            } else if s1_sq > 0.1 {
                saw_unsat = true;
            }
            let u = InterleavedStereo8::from_lr(
                GpuVec4::from_array(core_inputs[t * 2]),
                GpuVec4::from_array(core_inputs[t * 2 + 1]),
            );
            let _ = step_core_quad_phs_stereo8(u, &mut check_st, &params);
        }
        assert!(
            saw_sat && saw_unsat,
            "CPU Block 0 PHS test must exercise both saturated (s1^2 >= 64) and unsaturated regimes"
        );

        for tile in 0..num_tiles {
            for z_slot in [
                quad_param_slots::SVF_A1,
                quad_param_slots::SVF_A2,
                quad_param_slots::SVF_A3,
                quad_param_slots::SVF_TWO_K,
            ] {
                assert_eq!(
                    adj.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + z_slot].map(f32::to_bits),
                    [0; 4]
                );
            }
        }

        for t in [17_usize, 255, 256, 512] {
            for ch in 0..2 {
                for lane in 0..4 {
                    let eps_u = if ch == 0 && lane == 0 {
                        4.0e-3_f32
                    } else {
                        1.0e-3_f32
                    };
                    let mut ip = core_inputs.clone();
                    let mut im = core_inputs.clone();
                    ip[t * 2 + ch][lane] += eps_u;
                    im[t * 2 + ch][lane] -= eps_u;
                    let fd = (eval_loss(&ip, &params) - eval_loss(&im, &params))
                        / f64::from(ip[t * 2 + ch][lane] - im[t * 2 + ch][lane]);
                    let got = f64::from(adj.d_loss_d_inputs[t * 2 + ch][lane]);
                    assert!(
                        (got - fd).abs() <= 6.0e-3 * fd.abs().max(1.0),
                        "input FD mismatch t={t} ch={ch} lane={lane}: got={got} fd={fd}"
                    );
                }
            }
        }

        for slot in [
            quad_param_slots::SVF_K_NL,
            quad_param_slots::SVF_K,
            quad_param_slots::SVF_G,
        ] {
            for lane in 0..4 {
                let eps = match slot {
                    quad_param_slots::SVF_K_NL => {
                        if lane == 0 {
                            1.0e-4_f32
                        } else {
                            1.0e-3_f32
                        }
                    }
                    quad_param_slots::SVF_G => {
                        if lane == 0 {
                            5.0e-4_f32
                        } else {
                            1.0e-3_f32
                        }
                    }
                    _ => 2.0e-3_f32,
                };
                let got: f64 = (0..num_tiles)
                    .map(|tile| {
                        f64::from(adj.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + slot][lane])
                    })
                    .sum();
                if slot == quad_param_slots::SVF_K_NL {
                    assert!(
                        got.abs() > 1.0e-3,
                        "expected nonzero SVF_K_NL gradient on lane {lane}, got {got}"
                    );
                }
                let mut pp = params;
                let mut pm = params;
                match slot {
                    quad_param_slots::SVF_K_NL => {
                        let mut a = pp.svf_k_nl.to_array();
                        a[lane] += eps;
                        pp.svf_k_nl = GpuVec4::from_array(a);
                        let mut b = pm.svf_k_nl.to_array();
                        b[lane] -= eps;
                        pm.svf_k_nl = GpuVec4::from_array(b);
                    }
                    quad_param_slots::SVF_K => {
                        let mut a = pp.svf.k.to_array();
                        a[lane] += eps;
                        pp.svf = TptSvfCoeffs4::from_gk(g, GpuVec4::from_array(a));
                        let mut b = pm.svf.k.to_array();
                        b[lane] -= eps;
                        pm.svf = TptSvfCoeffs4::from_gk(g, GpuVec4::from_array(b));
                    }
                    _ => {
                        let mut ga = g.to_array();
                        ga[lane] += eps;
                        let g_p = GpuVec4::from_array(ga);
                        pp.svf = TptSvfCoeffs4::from_gk(g_p, k);
                        pp.two_svf_g = g_p.add(g_p);
                        let mut gb = g.to_array();
                        gb[lane] -= eps;
                        let g_m = GpuVec4::from_array(gb);
                        pm.svf = TptSvfCoeffs4::from_gk(g_m, k);
                        pm.two_svf_g = g_m.add(g_m);
                    }
                }
                let fd = (eval_loss(&core_inputs, &pp) - eval_loss(&core_inputs, &pm))
                    / f64::from(2.0 * eps);
                assert!(
                    (got - fd).abs() <= 8.0e-3 * fd.abs().max(1.0),
                    "slot={slot} lane={lane} got={got} fd={fd}"
                );
            }
        }
    }
}
