//! Causal Pass-2 forward audio evaluation and reverse-time adjoint assembly across
//! `Op-N[Pre]`, causal `CoreQuad` + `SONA_INTERBLOCK_TILES` cascade, stereo `ModalConvolver`
//! (`FIR + Kautz`), and 4-lane `Op-T` recurrent transport.

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};
use dnam_vm::{SONA_INTERBLOCK_TILES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER};

use crate::error::TrainError;
use crate::gpu::abi::{
    AdjointScanUniforms, QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
    quad_param_slots::ADAA_DRIVE,
};
use crate::runner::{AdjointScanInput, TrainRunner, pack_core_quad_params};

use super::super::model::cascade::{
    CausalCascadeOutput, run_causal_core_cascade_stereo, sanitize_unit_f64,
};
use super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64};
use super::adjoints::{eval_kautz_modal_channel_and_adjoint, eval_op_n_pre_and_adjoint};
use super::op_t_quad::{
    OpTQuadAdjointGrads, OpTQuadAdjointSlices, eval_op_t_quad_stereo_and_adjoint,
};

/// Forward intermediates cached during Pass-2 evaluation for exact reverse-time BPTT.
pub(super) struct Pass2EvalBuffers {
    pub ux_l: Vec<f64>,
    pub ux_r: Vec<f64>,
    pub onset_pre_l: Vec<f64>,
    pub onset_pre_r: Vec<f64>,
    pub x_tank_l: Vec<f64>,
    pub x_tank_r: Vec<f64>,
    pub pred_l: Vec<f64>,
    pub pred_r: Vec<f64>,
    pub cascade: CausalCascadeOutput,
}

pub(super) struct Pass2ParamGrads {
    pub d_gamma_bc: f64,
    pub d_beta_q: f64,
    pub d_adaa: Vec<[f64; 4]>,
    pub d_w_ro_direct: Vec<[[f64; 4]; 4]>, // [q][lane][6..10 -> 0..4]
    pub d_direct_s2_u: [f64; 2],
    pub d_kautz: Vec<[f64; 6]>, // [r_low, theta, c_re_l, c_im_l, c_re_r, c_im_r]
    pub d_op_t: OpTQuadAdjointGrads,
}

#[inline]
fn delay16_slice(sig: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; sig.len()];
    if sig.len() > 16 {
        out[16..].copy_from_slice(&sig[..(sig.len() - 16)]);
    }
    out
}

pub(super) fn eval_model_pass2_audio(
    runner: &TrainRunner,
    model: &SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    active_quads: usize,
) -> Result<Pass2EvalBuffers, TrainError> {
    let n = in_l.len().min(in_r.len());
    let q_count = active_quads.clamp(1, 10);

    let (ux_l, _, _) = eval_op_n_pre_and_adjoint(
        &in_l[..n],
        model.op_n_pre_gamma_bc,
        model.op_n_pre_beta_q,
        None,
        None,
    );
    let (ux_r, _, _) = eval_op_n_pre_and_adjoint(
        &in_r[..n],
        model.op_n_pre_gamma_bc,
        model.op_n_pre_beta_q,
        None,
        None,
    );

    let rho_core = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let cascade =
        run_causal_core_cascade_stereo(runner, model, &ux_l, &ux_r, q_count, rho_core, true)?;

    let (pc_l, pc_r) = (model.pre_conv_u_gain[0], model.pre_conv_u_gain[1]);
    let (dir_l, dir_r) = (model.direct_s2_u_gain[0], model.direct_s2_u_gain[1]);
    let mut pre_conv_l = vec![0.0_f64; n];
    let mut pre_conv_r = vec![0.0_f64; n];
    let mut x_tank_l = vec![0.0_f64; n];
    let mut x_tank_r = vec![0.0_f64; n];

    for t in 0..n {
        let mut bus_l = pc_l * ux_l[t];
        let mut bus_r = pc_r * ux_r[t];
        let mut tank_l = dir_l * ux_l[t];
        let mut tank_r = dir_r * ux_r[t];
        for q in 0..q_count {
            let fbase = (q * n + t) * 6;
            let v_eff_l = cascade.feature_outputs[fbase];
            let v_eff_r = cascade.feature_outputs[fbase + 1];
            let odd_l = cascade.feature_outputs[fbase + 2];
            let odd_r = cascade.feature_outputs[fbase + 3];
            let even_l = cascade.feature_outputs[fbase + 4];
            let even_r = cascade.feature_outputs[fbase + 5];
            for k in 0..4 {
                let r = &model.w_ro[q * 4 + k];
                bus_l += r[0] * f64::from(odd_l[k])
                    + r[2] * f64::from(even_l[k])
                    + r[4] * f64::from(v_eff_l[k]);
                bus_r += r[1] * f64::from(odd_r[k])
                    + r[3] * f64::from(even_r[k])
                    + r[5] * f64::from(v_eff_r[k]);
                tank_l += r[6] * f64::from(odd_l[k])
                    + r[8] * f64::from(even_l[k])
                    + r[10] * f64::from(v_eff_l[k]);
                tank_r += r[7] * f64::from(odd_r[k])
                    + r[9] * f64::from(even_r[k])
                    + r[11] * f64::from(v_eff_r[k]);
            }
        }
        pre_conv_l[t] = bus_l;
        pre_conv_r[t] = bus_r;
        x_tank_l[t] = tank_l;
        x_tank_r[t] = tank_r;
    }

    let active_fir = model.active_fir_taps.min(NUM_FIR_TAPS_ULTRA);
    if active_fir > 0 {
        for t in 0..n {
            let (mut fl, mut fr) = (0.0_f64, 0.0_f64);
            for k in 0..active_fir.min(t + 1) {
                fl += model.fir_l[k] * pre_conv_l[t - k];
                fr += model.fir_r[k] * pre_conv_r[t - k];
            }
            x_tank_l[t] += fl;
            x_tank_r[t] += fr;
        }
    }

    let onset_pre_l = delay16_slice(&pre_conv_l);
    let onset_pre_r = delay16_slice(&pre_conv_r);
    let active_k = model.active_kautz_pairs.min(model.kautz.len());
    if active_k > 0 {
        let (kl, _) = eval_kautz_modal_channel_and_adjoint(
            &onset_pre_l,
            &model.kautz[..active_k],
            false,
            None,
            None,
        );
        let (kr, _) = eval_kautz_modal_channel_and_adjoint(
            &onset_pre_r,
            &model.kautz[..active_k],
            true,
            None,
            None,
        );
        for t in 0..n {
            x_tank_l[t] += kl[t];
            x_tank_r[t] += kr[t];
        }
    }

    let dry_mix = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let mut pred_l: Vec<f64> = x_tank_l.iter().map(|&v| dry_mix * v).collect();
    let mut pred_r: Vec<f64> = x_tank_r.iter().map(|&v| dry_mix * v).collect();
    let tq = &model.transport_quads[0];
    let op_t_active = tq
        .input_gains
        .iter()
        .chain(&tq.wet_tap_l)
        .chain(&tq.wet_tap_r)
        .any(|&v| v.abs() > 1.0e-9);
    if op_t_active {
        let (tl, tr, _) =
            eval_op_t_quad_stereo_and_adjoint(&x_tank_l, &x_tank_r, tq, model.sample_rate_hz, None);
        for t in 0..n {
            pred_l[t] += tl[t];
            pred_r[t] += tr[t];
        }
    }

    Ok(Pass2EvalBuffers {
        ux_l,
        ux_r,
        onset_pre_l,
        onset_pre_r,
        x_tank_l,
        x_tank_r,
        pred_l,
        pred_r,
        cascade,
    })
}

pub(super) struct Pass2AdjInput<'a> {
    pub model: &'a SonaModelF64,
    pub in_l: &'a [f64],
    pub in_r: &'a [f64],
    pub buf: &'a Pass2EvalBuffers,
    pub gl: &'a [f64],
    pub gr: &'a [f64],
    pub q_opt: usize,
    pub k_opt: usize,
}

pub(super) fn assemble_pass2_gradients(
    runner: &TrainRunner,
    inp: &Pass2AdjInput<'_>,
) -> Result<Pass2ParamGrads, TrainError> {
    let Pass2AdjInput {
        model,
        in_l,
        in_r,
        buf,
        gl,
        gr,
        q_opt,
        k_opt,
    } = *inp;
    let n = buf.ux_l.len();
    let q_count = (buf.cascade.feature_outputs.len() / (n * 6)).clamp(1, 10);
    let active_tiles_topo = match q_count {
        0..=4 => 3_usize,
        5..=6 => 13_usize,
        7..=8 => 23_usize,
        _ => 35_usize,
    };
    let num_tiles = n.div_ceil(TILE_FRAMES);
    let dry_mix = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);

    let mut gx_l: Vec<f64> = gl.iter().map(|&v| dry_mix * v).collect();
    let mut gx_r: Vec<f64> = gr.iter().map(|&v| dry_mix * v).collect();
    let tq = &model.transport_quads[0];
    let op_t_active = tq
        .input_gains
        .iter()
        .chain(&tq.wet_tap_l)
        .chain(&tq.wet_tap_r)
        .any(|&v| v.abs() > 1.0e-9);
    let (_, _, d_op_t) = if op_t_active {
        eval_op_t_quad_stereo_and_adjoint(
            &buf.x_tank_l,
            &buf.x_tank_r,
            tq,
            model.sample_rate_hz,
            Some(OpTQuadAdjointSlices {
                d_out_l: gl,
                d_out_r: gr,
                grad_x_l: Some(&mut gx_l),
                grad_x_r: Some(&mut gx_r),
            }),
        )
    } else {
        (Vec::new(), Vec::new(), OpTQuadAdjointGrads::default())
    };

    // ModalConvolver (Kautz + Early FIR) reverse pass into pre_conv_bus.
    let mut gx_pre_l = vec![0.0_f64; n];
    let mut gx_pre_r = vec![0.0_f64; n];
    let active_fir = model.active_fir_taps.min(NUM_FIR_TAPS_ULTRA);
    if active_fir > 0 {
        for t in 0..n {
            let (gl_t, gr_t) = (gx_l[t], gx_r[t]);
            for k in 0..active_fir.min(t + 1) {
                gx_pre_l[t - k] += model.fir_l[k] * gl_t;
                gx_pre_r[t - k] += model.fir_r[k] * gr_t;
            }
        }
    }

    let active_k = model.active_kautz_pairs.min(model.kautz.len());
    let mut d_kautz = vec![[0.0_f64; 6]; k_opt];
    if active_k > 0 {
        let mut g_onset_l = vec![0.0_f64; n];
        let mut g_onset_r = vec![0.0_f64; n];
        let (_, k_gl) = eval_kautz_modal_channel_and_adjoint(
            &buf.onset_pre_l,
            &model.kautz[..active_k],
            false,
            Some(&gx_l),
            Some(&mut g_onset_l),
        );
        let (_, k_gr) = eval_kautz_modal_channel_and_adjoint(
            &buf.onset_pre_r,
            &model.kautz[..active_k],
            true,
            Some(&gx_r),
            Some(&mut g_onset_r),
        );
        for k in 0..k_opt.min(active_k) {
            d_kautz[k] = [
                k_gl[k][0] + k_gr[k][0],
                k_gl[k][1] + k_gr[k][1],
                k_gl[k][3],
                k_gl[k][4],
                k_gr[k][3],
                k_gr[k][4],
            ];
        }
        for t in 16..n {
            gx_pre_l[t - 16] += g_onset_l[t];
            gx_pre_r[t - 16] += g_onset_r[t];
        }
    }

    // Readout linear weight gradients + initial CoreQuad feature adjoints.
    let mut d_w_ro_direct = vec![[[0.0_f64; 4]; 4]; q_opt];
    let mut d_direct_s2_u = [0.0_f64; 2];
    let mut gx_ux_l: Vec<f64> = (0..n)
        .map(|t| {
            d_direct_s2_u[0] += gx_l[t] * buf.ux_l[t];
            d_direct_s2_u[1] += gx_r[t] * buf.ux_r[t];
            model.direct_s2_u_gain[0] * gx_l[t] + model.pre_conv_u_gain[0] * gx_pre_l[t]
        })
        .collect();
    let mut gx_ux_r: Vec<f64> = (0..n)
        .map(|t| model.direct_s2_u_gain[1] * gx_r[t] + model.pre_conv_u_gain[1] * gx_pre_r[t])
        .collect();

    let mut d_feats = vec![[0.0_f32; 4]; q_count * n * 6];
    for q in 0..q_count {
        for t in 0..n {
            let fbase = (q * n + t) * 6;
            let (gl_t, gr_t) = (gx_l[t], gx_r[t]);
            let (pl_t, pr_t) = (gx_pre_l[t], gx_pre_r[t]);
            let odd_l = buf.cascade.feature_outputs[fbase + 2];
            let odd_r = buf.cascade.feature_outputs[fbase + 3];
            let even_l = buf.cascade.feature_outputs[fbase + 4];
            let even_r = buf.cascade.feature_outputs[fbase + 5];
            for k in 0..4 {
                let r = &model.w_ro[q * 4 + k];
                if q < q_opt {
                    d_w_ro_direct[q][k][0] += gl_t * f64::from(odd_l[k]);
                    d_w_ro_direct[q][k][1] += gr_t * f64::from(odd_r[k]);
                    d_w_ro_direct[q][k][2] += gl_t * f64::from(even_l[k]);
                    d_w_ro_direct[q][k][3] += gr_t * f64::from(even_r[k]);
                }
                d_feats[fbase][k] = (pl_t * r[4] + gl_t * r[10]) as f32;
                d_feats[fbase + 1][k] = (pr_t * r[5] + gr_t * r[11]) as f32;
                d_feats[fbase + 2][k] = (pl_t * r[0] + gl_t * r[6]) as f32;
                d_feats[fbase + 3][k] = (pr_t * r[1] + gr_t * r[7]) as f32;
                d_feats[fbase + 4][k] = (pl_t * r[2] + gl_t * r[8]) as f32;
                d_feats[fbase + 5][k] = (pr_t * r[3] + gr_t * r[9]) as f32;
            }
        }
    }

    // Reverse topological scan across CoreQuads & InterBlockTiles -> M/S input -> Op-N[Pre].
    let mut d_adaa = vec![[0.0_f64; 4]; q_opt];
    let rho_core = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let ckpt_per_quad = num_tiles * TILE_CKPT_VEC4_STRIDE;
    for &q in SONA_ULTRA_QUAD_TRAVERSAL_ORDER.iter().rev() {
        if q >= q_count {
            continue;
        }
        let qp = pack_core_quad_params(
            &model.core_quads[q]
                .rescaled_for_rho(rho_core)
                .to_kernel_params(),
        );
        let adj = runner.adjoint_scan_256(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: n as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &buf.cascade.core_inputs[(q * n * 2)..((q + 1) * n * 2)],
            quad_params: &qp,
            tile_checkpoints: &buf.cascade.tile_checkpoints
                [(q * ckpt_per_quad)..((q + 1) * ckpt_per_quad)],
            d_loss_d_features: &d_feats[(q * n * 6)..((q + 1) * n * 6)],
            feature_outputs: &buf.cascade.feature_outputs[(q * n * 6)..((q + 1) * n * 6)],
        })?;
        if q < q_opt {
            for tile in 0..num_tiles {
                let g = adj.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + ADAA_DRIVE];
                for k in 0..4 {
                    d_adaa[q][k] += f64::from(g[k]);
                }
            }
        }

        // Propagate input adjoints back into ux_l/ux_r (via M/S) and upstream tile source quads.
        for t in 0..n {
            let gin_l = adj.d_loss_d_inputs[t * 2];
            let gin_r = adj.d_loss_d_inputs[t * 2 + 1];
            for d in 0..4 {
                let wm = sanitize_unit_f64(model.core_quads[q].in_mid_weight[d], -16.0, 16.0, 0.0);
                let ws = sanitize_unit_f64(model.core_quads[q].in_side_weight[d], -16.0, 16.0, 0.0);
                let gl_d = f64::from(gin_l[d]);
                let gr_d = f64::from(gin_r[d]);
                let d_mid = wm * (gl_d + gr_d);
                let d_side = ws * (gl_d - gr_d);
                gx_ux_l[t] += 0.5 * (d_mid + d_side);
                gx_ux_r[t] += 0.5 * (d_mid - d_side);
            }
            for (tile_idx, &(src_q, dst_q)) in SONA_INTERBLOCK_TILES[..active_tiles_topo]
                .iter()
                .enumerate()
            {
                if dst_q == q && src_q < q_count {
                    let tile = &model.tiles[tile_idx];
                    let sbase = (src_q * n + t) * 6;
                    for d in 0..4 {
                        let gl_d = f64::from(gin_l[d]);
                        let gr_d = f64::from(gin_r[d]);
                        for s in 0..4 {
                            let wo = sanitize_unit_f64(tile.w_odd_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                            let we =
                                sanitize_unit_f64(tile.w_even_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                            let wl = sanitize_unit_f64(tile.w_lru_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                            d_feats[sbase][s] += wl * (gl_d as f32);
                            d_feats[sbase + 1][s] += wl * (gr_d as f32);
                            d_feats[sbase + 2][s] += wo * (gl_d as f32);
                            d_feats[sbase + 3][s] += wo * (gr_d as f32);
                            d_feats[sbase + 4][s] += we * (gl_d as f32);
                            d_feats[sbase + 5][s] += we * (gr_d as f32);
                        }
                    }
                }
            }
        }
    }

    let (_, g_gl, g_bl) = eval_op_n_pre_and_adjoint(
        &in_l[..n],
        model.op_n_pre_gamma_bc,
        model.op_n_pre_beta_q,
        Some(&gx_ux_l),
        None,
    );
    let (_, g_gr, g_br) = eval_op_n_pre_and_adjoint(
        &in_r[..n],
        model.op_n_pre_gamma_bc,
        model.op_n_pre_beta_q,
        Some(&gx_ux_r),
        None,
    );

    Ok(Pass2ParamGrads {
        d_gamma_bc: g_gl + g_gr,
        d_beta_q: g_bl + g_br,
        d_adaa,
        d_w_ro_direct,
        d_direct_s2_u,
        d_kautz,
        d_op_t,
    })
}
