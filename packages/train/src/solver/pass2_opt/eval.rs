//! Causal Pass-2 forward audio evaluation and reverse-time adjoint assembly across `SensoryQuad`,
//! `Op-P` + signed `Q0.32` `Crossbar`, `Op-N[Pre]`, causal `CoreQuad` + `SONA_INTERBLOCK_TILES`
//! cascade, ZOH `ro_pole_48k` readout, stereo `ModalConvolver` (`FIR + Kautz`), and `1..=4`
//! `Op-T` recurrent transport quads (`ref[train.pipeline.backbone-trajectory-opt]`).

pub(super) mod core_adj;

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;
use dnam_vm::SONA_ULTRA_QUAD_TRAVERSAL_ORDER;

use crate::error::TrainError;
use crate::gpu::abi::{
    AdjointScanUniforms, FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE,
    TILE_FRAMES,
};
use crate::runner::{AdjointScanInput, TrainRunner, pack_core_quad_params};

use super::super::model::cascade::{
    CascadeCrossbarSpec, CausalCascadeOutput, run_causal_core_cascade_with_crossbar,
    sanitize_unit_f64,
};
use super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64};
use super::adjoints::{eval_kautz_modal_channel_and_adjoint, eval_ro_pole_hp_and_adjoint};
use super::causal_loop::{
    CausalLoopBuffers, assemble_causal_loop_adjoint, eval_causal_loop_forward,
    is_op_t_crossbar_coupled,
};
use super::crossbar::{CrossbarForwardTrace, assemble_crossbar_adjoint, eval_crossbar_forward};
use super::op_n_pre::eval_op_n_pre_and_adjoint;
use super::op_t_quad::{OpTQuadAdjointSlices, eval_op_t_bank_stereo_and_adjoint};
use super::sensory::eval_sensory_quad_and_adjoint;
use core_adj::{
    CoreInputBackpropCtx, CoreInputBackpropOut, active_opt_quads_for_core_quads,
    active_tiles_for_quads, add_tile_param_grads_to_quad, backprop_core_input_frame,
    finalize_core_quad_zoh,
};

pub(super) use core_adj::Pass2FullGrads as Pass2ParamGrads;

/// Forward intermediates cached during Pass-2 evaluation for exact reverse-time BPTT.
pub(super) struct Pass2EvalBuffers {
    pub u_sense_l: Vec<f64>,
    pub u_sense_r: Vec<f64>,
    pub cb_trace: CrossbarForwardTrace,
    pub ux_l: Vec<f64>,
    pub ux_r: Vec<f64>,
    pub ro_hp_l: Vec<f64>,
    pub ro_hp_r: Vec<f64>,
    pub onset_pre_l: Vec<f64>,
    pub onset_pre_r: Vec<f64>,
    pub x_tank_l: Vec<f64>,
    pub x_tank_r: Vec<f64>,
    pub pred_l: Vec<f64>,
    pub pred_r: Vec<f64>,
    pub cascade: CausalCascadeOutput,
    pub causal_loop: Option<CausalLoopBuffers>,
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

    if is_op_t_crossbar_coupled(model, q_count) {
        let cl = eval_causal_loop_forward(model, &in_l[..n], &in_r[..n], q_count, true);
        let cb_dummy = CrossbarForwardTrace {
            n: 0,
            q_count,
            sample_rate_hz: model.sample_rate_hz,
            include_stereo_phase: q_count > 4,
            os2x: false,
            active: false,
            dc_pole: 0.0,
            dc_gain: 1.0,
            alpha_i: [0.0; 6],
            alpha_q: [0.0; 6],
            iq_lr: Vec::new(),
            op_p_feat: Vec::new(),
            cb_add_l: Vec::new(),
            cb_add_r: Vec::new(),
        };
        return Ok(Pass2EvalBuffers {
            u_sense_l: Vec::new(),
            u_sense_r: Vec::new(),
            cb_trace: cb_dummy,
            ux_l: Vec::new(),
            ux_r: Vec::new(),
            ro_hp_l: Vec::new(),
            ro_hp_r: Vec::new(),
            onset_pre_l: Vec::new(),
            onset_pre_r: Vec::new(),
            x_tank_l: Vec::new(),
            x_tank_r: Vec::new(),
            pred_l: cl.pred_l.clone(),
            pred_r: cl.pred_r.clone(),
            cascade: CausalCascadeOutput {
                core_inputs: Vec::new(),
                tile_checkpoints: Vec::new(),
                feature_outputs: Vec::new(),
            },
            causal_loop: Some(cl),
        });
    }

    let (u_sense_l, u_sense_r, _) =
        eval_sensory_quad_and_adjoint(model, &in_l[..n], &in_r[..n], None, None);
    let include_stereo_phase = q_count > 4;
    let cb_trace =
        eval_crossbar_forward(model, &u_sense_l, &u_sense_r, q_count, include_stereo_phase);

    let (ux_l, _) = eval_op_n_pre_and_adjoint(
        &u_sense_l,
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        model.sample_rate_hz,
        None,
        None,
    );
    let (ux_r, _) = eval_op_n_pre_and_adjoint(
        &u_sense_r,
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        model.sample_rate_hz,
        None,
        None,
    );

    let rho_core = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let (ro_hp_l, _) = eval_ro_pole_hp_and_adjoint(&ux_l, model.ro_pole_48k, rho_core, None, None);
    let (ro_hp_r, _) = eval_ro_pole_hp_and_adjoint(&ux_r, model.ro_pole_48k, rho_core, None, None);

    let cb_add = cb_trace
        .active
        .then_some((cb_trace.cb_add_l.as_slice(), cb_trace.cb_add_r.as_slice()));
    let cascade = run_causal_core_cascade_with_crossbar(
        runner,
        model,
        &ux_l,
        &ux_r,
        CascadeCrossbarSpec {
            active_quads: q_count,
            rho_core,
            write_checkpoints: true,
            crossbar_add: cb_add,
        },
    )?;

    let (pc_l, pc_r) = (model.pre_conv_u_gain[0], model.pre_conv_u_gain[1]);
    let (pc_hp_l, pc_hp_r) = (model.pre_conv_hp_gain[0], model.pre_conv_hp_gain[1]);
    let (dir_l, dir_r) = (model.direct_s2_u_gain[0], model.direct_s2_u_gain[1]);
    let (dir_hp_l, dir_hp_r) = (model.direct_s2_hp_gain[0], model.direct_s2_hp_gain[1]);
    let mut pre_conv_l = vec![0.0_f64; n];
    let mut pre_conv_r = vec![0.0_f64; n];
    let mut x_tank_l = vec![0.0_f64; n];
    let mut x_tank_r = vec![0.0_f64; n];

    for t in 0..n {
        let mut bus_l = pc_l * ux_l[t] + pc_hp_l * ro_hp_l[t];
        let mut bus_r = pc_r * ux_r[t] + pc_hp_r * ro_hp_r[t];
        let mut tank_l = dir_l * ux_l[t] + dir_hp_l * ro_hp_l[t];
        let mut tank_r = dir_r * ux_r[t] + dir_hp_r * ro_hp_r[t];
        for q in 0..q_count {
            let fbase = (q * n + t) * FEATURE_VEC4_STRIDE;
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
    let active_opt = active_opt_quads_for_core_quads(q_count);
    let (tl, tr, _) = eval_op_t_bank_stereo_and_adjoint(
        &x_tank_l,
        &x_tank_r,
        &model.transport_quads,
        active_opt,
        model.sample_rate_hz,
        None,
    );
    for t in 0..n {
        pred_l[t] += tl[t];
        pred_r[t] += tr[t];
    }

    Ok(Pass2EvalBuffers {
        u_sense_l,
        u_sense_r,
        cb_trace,
        ux_l,
        ux_r,
        ro_hp_l,
        ro_hp_r,
        onset_pre_l,
        onset_pre_r,
        x_tank_l,
        x_tank_r,
        pred_l,
        pred_r,
        cascade,
        causal_loop: None,
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
    if let Some(ref cl) = buf.causal_loop {
        return Ok(assemble_causal_loop_adjoint(model, in_l, in_r, cl, gl, gr));
    }

    let n = buf.ux_l.len();
    let q_count = if n > 0 {
        (buf.cascade.feature_outputs.len() / (n * FEATURE_VEC4_STRIDE)).clamp(1, 10)
    } else {
        q_opt.clamp(1, 10)
    };
    let active_tiles_topo = active_tiles_for_quads(q_count);
    let active_opt = active_opt_quads_for_core_quads(q_count);
    let active_k = model.active_kautz_pairs.min(model.kautz.len());
    let mut grads = Pass2ParamGrads::with_dims(q_opt.max(q_count), k_opt.max(active_k));
    if n == 0 {
        return Ok(grads);
    }

    let num_tiles = n.div_ceil(TILE_FRAMES);
    let dry_mix = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    if model.tank_dry_mix.is_finite() && model.tank_dry_mix.abs() < 2.0 {
        for t in 0..n {
            grads.d_tank_dry_mix += gl[t] * buf.x_tank_l[t] + gr[t] * buf.x_tank_r[t];
        }
    }

    let mut gx_l: Vec<f64> = gl.iter().map(|&v| dry_mix * v).collect();
    let mut gx_r: Vec<f64> = gr.iter().map(|&v| dry_mix * v).collect();
    let (_, _, d_op_t) = eval_op_t_bank_stereo_and_adjoint(
        &buf.x_tank_l,
        &buf.x_tank_r,
        &model.transport_quads,
        active_opt,
        model.sample_rate_hz,
        Some(OpTQuadAdjointSlices {
            d_out_l: gl,
            d_out_r: gr,
            grad_x_l: Some(&mut gx_l),
            grad_x_r: Some(&mut gx_r),
        }),
    );
    grads.op_t = d_op_t;

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
        for k in 0..active_k.min(grads.d_kautz.len()) {
            grads.d_kautz[k] = [
                k_gl[k][0] + k_gr[k][0],
                k_gl[k][1] + k_gr[k][1],
                k_gl[k][2] + k_gr[k][2],
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

    let mut d_hp_l = vec![0.0_f64; n];
    let mut d_hp_r = vec![0.0_f64; n];
    let mut gx_ux_l = vec![0.0_f64; n];
    let mut gx_ux_r = vec![0.0_f64; n];
    for t in 0..n {
        let (gl_t, gr_t) = (gx_l[t], gx_r[t]);
        let (pl_t, pr_t) = (gx_pre_l[t], gx_pre_r[t]);
        grads.d_direct_s2_u[0] += gl_t * buf.ux_l[t];
        grads.d_direct_s2_u[1] += gr_t * buf.ux_r[t];
        grads.d_direct_s2_hp[0] += gl_t * buf.ro_hp_l[t];
        grads.d_direct_s2_hp[1] += gr_t * buf.ro_hp_r[t];
        grads.d_pre_conv_u[0] += pl_t * buf.ux_l[t];
        grads.d_pre_conv_u[1] += pr_t * buf.ux_r[t];
        grads.d_pre_conv_hp[0] += pl_t * buf.ro_hp_l[t];
        grads.d_pre_conv_hp[1] += pr_t * buf.ro_hp_r[t];

        gx_ux_l[t] = model.direct_s2_u_gain[0] * gl_t + model.pre_conv_u_gain[0] * pl_t;
        gx_ux_r[t] = model.direct_s2_u_gain[1] * gr_t + model.pre_conv_u_gain[1] * pr_t;
        d_hp_l[t] = model.direct_s2_hp_gain[0] * gl_t + model.pre_conv_hp_gain[0] * pl_t;
        d_hp_r[t] = model.direct_s2_hp_gain[1] * gr_t + model.pre_conv_hp_gain[1] * pr_t;
    }

    let mut d_feats = vec![[0.0_f32; 4]; q_count * n * FEATURE_VEC4_STRIDE];
    for q in 0..q_count {
        for t in 0..n {
            let fbase = (q * n + t) * FEATURE_VEC4_STRIDE;
            let (gl_t, gr_t) = (gx_l[t], gx_r[t]);
            let (pl_t, pr_t) = (gx_pre_l[t], gx_pre_r[t]);
            let v_eff_l = buf.cascade.feature_outputs[fbase];
            let v_eff_r = buf.cascade.feature_outputs[fbase + 1];
            let odd_l = buf.cascade.feature_outputs[fbase + 2];
            let odd_r = buf.cascade.feature_outputs[fbase + 3];
            let even_l = buf.cascade.feature_outputs[fbase + 4];
            let even_r = buf.cascade.feature_outputs[fbase + 5];
            for k in 0..4 {
                let row_idx = q * 4 + k;
                let r = &model.w_ro[row_idx];
                let dw = &mut grads.d_w_ro[row_idx];
                dw[0] += pl_t * f64::from(odd_l[k]);
                dw[1] += pr_t * f64::from(odd_r[k]);
                dw[2] += pl_t * f64::from(even_l[k]);
                dw[3] += pr_t * f64::from(even_r[k]);
                dw[4] += pl_t * f64::from(v_eff_l[k]);
                dw[5] += pr_t * f64::from(v_eff_r[k]);
                dw[6] += gl_t * f64::from(odd_l[k]);
                dw[7] += gr_t * f64::from(odd_r[k]);
                dw[8] += gl_t * f64::from(even_l[k]);
                dw[9] += gr_t * f64::from(even_r[k]);
                dw[10] += gl_t * f64::from(v_eff_l[k]);
                dw[11] += gr_t * f64::from(v_eff_r[k]);

                d_feats[fbase][k] = (pl_t * r[4] + gl_t * r[10]) as f32;
                d_feats[fbase + 1][k] = (pr_t * r[5] + gr_t * r[11]) as f32;
                d_feats[fbase + 2][k] = (pl_t * r[0] + gl_t * r[6]) as f32;
                d_feats[fbase + 3][k] = (pr_t * r[1] + gr_t * r[7]) as f32;
                d_feats[fbase + 4][k] = (pl_t * r[2] + gl_t * r[8]) as f32;
                d_feats[fbase + 5][k] = (pr_t * r[3] + gr_t * r[9]) as f32;
            }
        }
    }

    let rho_core = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let ckpt_per_quad = num_tiles * TILE_CKPT_VEC4_STRIDE;
    let mut d_cb_l = if buf.cb_trace.active {
        vec![[0.0_f64; 4]; q_count * n]
    } else {
        Vec::new()
    };
    let mut d_cb_r = if buf.cb_trace.active {
        vec![[0.0_f64; 4]; q_count * n]
    } else {
        Vec::new()
    };

    for &q in SONA_ULTRA_QUAD_TRAVERSAL_ORDER.iter().rev() {
        if q >= q_count {
            continue;
        }
        let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
        let qp = pack_core_quad_params(
            &model.core_quads[q]
                .rescaled_for_rho(rho_core)
                .to_kernel_params_for_block(block_idx),
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
                block_idx,
            },
            core_inputs: &buf.cascade.core_inputs[(q * n * 2)..((q + 1) * n * 2)],
            quad_params: &qp,
            tile_checkpoints: &buf.cascade.tile_checkpoints
                [(q * ckpt_per_quad)..((q + 1) * ckpt_per_quad)],
            d_loss_d_features: &d_feats
                [(q * n * FEATURE_VEC4_STRIDE)..((q + 1) * n * FEATURE_VEC4_STRIDE)],
            feature_outputs: &buf.cascade.feature_outputs
                [(q * n * FEATURE_VEC4_STRIDE)..((q + 1) * n * FEATURE_VEC4_STRIDE)],
        })?;
        for tile in 0..num_tiles {
            add_tile_param_grads_to_quad(
                &model.core_quads[q],
                block_idx,
                &adj.param_grads,
                tile * QUAD_PARAMS_VEC4_STRIDE,
                &mut grads.core_quads[q],
            );
        }
        finalize_core_quad_zoh(&model.core_quads[q], rho_core, &mut grads.core_quads[q]);

        for t in 0..n {
            let gin_l = adj.d_loss_d_inputs[t * 2];
            let gin_r = adj.d_loss_d_inputs[t * 2 + 1];
            let cb_opt = if buf.cb_trace.active {
                Some((d_cb_l.as_mut_slice(), d_cb_r.as_mut_slice()))
            } else {
                None
            };
            let (add_l, add_r) = backprop_core_input_frame(
                &CoreInputBackpropCtx {
                    model,
                    q,
                    t,
                    n,
                    q_count,
                    active_tiles: active_tiles_topo,
                    ux_l_t: buf.ux_l[t],
                    ux_r_t: buf.ux_r[t],
                    feature_outputs: &buf.cascade.feature_outputs,
                },
                gin_l,
                gin_r,
                CoreInputBackpropOut {
                    d_feats: &mut d_feats,
                    d_quad: &mut grads.core_quads[q],
                    d_tiles: &mut grads.tiles,
                    d_cb_opt: cb_opt,
                },
            );
            gx_ux_l[t] += add_l;
            gx_ux_r[t] += add_r;
        }
    }

    let (_, d_ro_l) = eval_ro_pole_hp_and_adjoint(
        &buf.ux_l,
        model.ro_pole_48k,
        rho_core,
        Some(&d_hp_l),
        Some(&mut gx_ux_l),
    );
    let (_, d_ro_r) = eval_ro_pole_hp_and_adjoint(
        &buf.ux_r,
        model.ro_pole_48k,
        rho_core,
        Some(&d_hp_r),
        Some(&mut gx_ux_r),
    );
    grads.d_ro_pole_48k = d_ro_l + d_ro_r;

    let mut d_s0_l = vec![0.0_f64; n];
    let mut d_s0_r = vec![0.0_f64; n];
    let (_, op_nl) = eval_op_n_pre_and_adjoint(
        &buf.u_sense_l,
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        model.sample_rate_hz,
        Some(&gx_ux_l),
        Some(&mut d_s0_l),
    );
    let (_, op_nr) = eval_op_n_pre_and_adjoint(
        &buf.u_sense_r,
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        model.sample_rate_hz,
        Some(&gx_ux_r),
        Some(&mut d_s0_r),
    );
    grads.op_n_pre = op_nl;
    grads.op_n_pre.accumulate(op_nr);

    let mut d_u_sense_l = vec![0.0_f64; n];
    let mut d_u_sense_r = vec![0.0_f64; n];
    if buf.cb_trace.active {
        grads.crossbar = assemble_crossbar_adjoint(
            model,
            &buf.cb_trace,
            &d_cb_l,
            &d_cb_r,
            Some(&mut d_u_sense_l),
            Some(&mut d_u_sense_r),
        );
    }
    for t in 0..n {
        d_u_sense_l[t] += d_s0_l[t];
        d_u_sense_r[t] += d_s0_r[t];
    }
    let (_, _, sense_grads) = eval_sensory_quad_and_adjoint(
        model,
        &in_l[..n],
        &in_r[..n],
        Some((&d_u_sense_l, &d_u_sense_r)),
        None,
    );
    grads.sensory = sense_grads;

    Ok(grads)
}
