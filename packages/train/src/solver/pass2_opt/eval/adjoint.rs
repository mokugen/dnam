//! Reverse-time Pass-2 adjoint assembly across `Op-T` recurrent transport, `ModalConvolver`
//! (`FIR + Kautz`), `2x` downsampler/upsampler, `W_ro` / `LinearReadout` (`ro_pole_48k`),
//! `CoreQuad` + `InterBlockTile` cascade, `Op-N[Pre]`, `Op-P` Crossbar, and `SensoryQuad`
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;

use crate::error::TrainError;
use crate::gpu::abi::{CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, TILE_FRAMES};
use crate::runner::TrainRunner;
use crate::runner::types::WaveCoreAdjointInput;

use super::super::super::model::cascade::{
    downsample_lane_2x_adjoint_f64, pack_wave_quads_and_tiles, upsample_lane_2x_adjoint_f64,
};
use super::super::super::model::v3_core_bridge::{
    V3_QUAD_PARAMS_VEC4_STRIDE, run_v3_wave_core_adjoint,
};
use super::super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64};
use super::super::adjoints::{eval_kautz_modal_channel_and_adjoint, eval_ro_pole_hp_and_adjoint};
use super::super::causal_loop::assemble_causal_loop_adjoint;
use super::super::crossbar::assemble_crossbar_adjoint;
use super::super::op_n_pre::eval_op_n_pre_and_adjoint;
use super::super::op_t_quad::{OpTQuadAdjointSlices, eval_op_t_bank_stereo_and_adjoint};
use super::super::sensory::eval_sensory_quad_and_adjoint;
use super::Pass2EvalBuffers;
use super::core_adj::{
    Pass2FullGrads as Pass2ParamGrads, active_opt_quads_for_core_quads, active_tiles_for_quads,
    add_tile_param_grads_to_quad, add_wave_tile_weight_grads, backprop_local_core_input_frame,
    finalize_core_quad_zoh,
};

use super::super::super::model::cascade::sanitize_unit_f64;

/// Inputs to [`assemble_pass2_gradients`].
#[derive(Debug)]
pub struct Pass2AdjInput<'a> {
    /// Model evaluated during forward pass.
    pub model: &'a SonaModelF64,
    /// Left input waveform.
    pub in_l: &'a [f64],
    /// Right input waveform.
    pub in_r: &'a [f64],
    /// Cached forward buffers from [`super::eval_model_pass2_audio`].
    pub buf: &'a Pass2EvalBuffers,
    /// Upstream loss gradient wrt left predicted waveform.
    pub gl: &'a [f64],
    /// Upstream loss gradient wrt right predicted waveform.
    pub gr: &'a [f64],
    /// Active core-quad count (`1..=10`).
    pub q_opt: usize,
    /// Active Kautz pair count (`0..=32`).
    pub k_opt: usize,
}

/// Assembles complete Pass-2 reverse-time parameter gradients across all `SONA` stages.
pub fn assemble_pass2_gradients(
    _runner: &TrainRunner,
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
    let c_ux_l_ref: &[f64] = if buf.os2x { &buf.core_ux_l } else { &buf.ux_l };
    let c_ux_r_ref: &[f64] = if buf.os2x { &buf.core_ux_r } else { &buf.ux_r };
    let n_core = c_ux_l_ref.len();
    let q_count = if n_core > 0 {
        (buf.cascade.feature_outputs.len() / (n_core * FEATURE_VEC4_STRIDE)).clamp(1, 10)
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

    let num_tiles = n_core.div_ceil(TILE_FRAMES);
    let dry_mix = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let wet_scale = sanitize_unit_f64(model.op_t_wet_scale, 0.0, 2.0, 1.0);
    if model.tank_dry_mix.is_finite() && model.tank_dry_mix.abs() < 2.0 {
        for t in 0..n {
            grads.d_tank_dry_mix += gl[t] * buf.x_tank_l[t] + gr[t] * buf.x_tank_r[t];
        }
    }
    if model.op_t_wet_scale.is_finite() && model.op_t_wet_scale >= 0.0 && model.op_t_wet_scale < 2.0
    {
        for t in 0..n {
            grads.d_op_t_wet_scale += gl[t] * buf.op_t_wet_l[t] + gr[t] * buf.op_t_wet_r[t];
        }
    }

    let mut gx_l: Vec<f64> = gl.iter().map(|&v| dry_mix * v).collect();
    let mut gx_r: Vec<f64> = gr.iter().map(|&v| dry_mix * v).collect();
    let gl_wet: Vec<f64> = gl.iter().map(|&v| wet_scale * v).collect();
    let gr_wet: Vec<f64> = gr.iter().map(|&v| wet_scale * v).collect();
    let (_, _, d_op_t) = eval_op_t_bank_stereo_and_adjoint(
        &buf.x_tank_l,
        &buf.x_tank_r,
        &model.transport_quads,
        active_opt,
        model.sample_rate_hz,
        model.dc_blocker_pole_5hz,
        Some(OpTQuadAdjointSlices {
            d_out_l: &gl_wet,
            d_out_r: &gr_wet,
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

    let (g_core_dir_l, g_core_dir_r, g_core_pre_l, g_core_pre_r) = if buf.os2x {
        (
            downsample_lane_2x_adjoint_f64(&gx_l, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_adjoint_f64(&gx_r, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_adjoint_f64(&gx_pre_l, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_adjoint_f64(&gx_pre_r, model.os_dec_c_prev1, model.os_dec_c0),
        )
    } else {
        (gx_l, gx_r, gx_pre_l, gx_pre_r)
    };

    let mut d_hp_l = vec![0.0_f64; n_core];
    let mut d_hp_r = vec![0.0_f64; n_core];
    let mut g_core_ux_l = vec![0.0_f64; n_core];
    let mut g_core_ux_r = vec![0.0_f64; n_core];
    for t in 0..n_core {
        let (gl_t, gr_t) = (g_core_dir_l[t], g_core_dir_r[t]);
        let (pl_t, pr_t) = (g_core_pre_l[t], g_core_pre_r[t]);
        grads.d_direct_s2_u[0] += gl_t * c_ux_l_ref[t];
        grads.d_direct_s2_u[1] += gr_t * c_ux_r_ref[t];
        grads.d_direct_s2_hp[0] += gl_t * buf.core_ro_hp_l[t];
        grads.d_direct_s2_hp[1] += gr_t * buf.core_ro_hp_r[t];
        grads.d_pre_conv_u[0] += pl_t * c_ux_l_ref[t];
        grads.d_pre_conv_u[1] += pr_t * c_ux_r_ref[t];
        grads.d_pre_conv_hp[0] += pl_t * buf.core_ro_hp_l[t];
        grads.d_pre_conv_hp[1] += pr_t * buf.core_ro_hp_r[t];

        g_core_ux_l[t] = model.direct_s2_u_gain[0] * gl_t + model.pre_conv_u_gain[0] * pl_t;
        g_core_ux_r[t] = model.direct_s2_u_gain[1] * gr_t + model.pre_conv_u_gain[1] * pr_t;
        d_hp_l[t] = model.direct_s2_hp_gain[0] * gl_t + model.pre_conv_hp_gain[0] * pl_t;
        d_hp_r[t] = model.direct_s2_hp_gain[1] * gr_t + model.pre_conv_hp_gain[1] * pr_t;
    }

    let mut d_feats = vec![[0.0_f32; 4]; q_count * n_core * FEATURE_VEC4_STRIDE];
    for q in 0..q_count {
        for t in 0..n_core {
            let fbase = (q * n_core + t) * FEATURE_VEC4_STRIDE;
            let (gl_t, gr_t) = (g_core_dir_l[t], g_core_dir_r[t]);
            let (pl_t, pr_t) = (g_core_pre_l[t], g_core_pre_r[t]);
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

    let rho_1x = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let rho_core = if buf.os2x { 0.5 * rho_1x } else { rho_1x };
    let mut d_cb_l = if buf.cb_trace.active {
        vec![[0.0_f64; 4]; q_count * n_core]
    } else {
        Vec::new()
    };
    let mut d_cb_r = if buf.cb_trace.active {
        vec![[0.0_f64; 4]; q_count * n_core]
    } else {
        Vec::new()
    };

    let (quad_params, tile_descriptors) =
        pack_wave_quads_and_tiles(model, q_count, active_tiles_topo, rho_core);
    let wave_adj = run_v3_wave_core_adjoint(&WaveCoreAdjointInput {
        num_frames: n_core,
        num_quads: q_count,
        num_stems: 1,
        active_tiles: active_tiles_topo,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        core_inputs: &buf.cascade.core_inputs,
        quad_params: &quad_params,
        tile_checkpoints: &buf.cascade.tile_checkpoints,
        d_loss_d_features: &d_feats,
        feature_outputs: &buf.cascade.feature_outputs,
        tile_descriptors: &tile_descriptors,
    })?;
    add_wave_tile_weight_grads(
        model,
        q_count,
        active_tiles_topo,
        &wave_adj.tile_weight_grads,
        &mut grads.tiles,
    );

    for q in 0..q_count {
        let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
        for tile in 0..num_tiles {
            add_tile_param_grads_to_quad(
                &model.core_quads[q],
                block_idx,
                &wave_adj.param_grads,
                (q * num_tiles + tile) * V3_QUAD_PARAMS_VEC4_STRIDE,
                &mut grads.core_quads[q],
            );
        }
        finalize_core_quad_zoh(&model.core_quads[q], rho_core, &mut grads.core_quads[q]);

        for t in 0..n_core {
            let in_idx = (q * n_core + t) * CORE_INPUT_VEC4_STRIDE;
            let gin_l = wave_adj.d_loss_d_inputs[in_idx];
            let gin_r = wave_adj.d_loss_d_inputs[in_idx + 1];
            let cb_opt = if buf.cb_trace.active {
                Some((d_cb_l.as_mut_slice(), d_cb_r.as_mut_slice()))
            } else {
                None
            };
            let (add_l, add_r) = backprop_local_core_input_frame(
                model,
                q,
                t,
                n_core,
                c_ux_l_ref[t],
                c_ux_r_ref[t],
                gin_l,
                gin_r,
                &mut grads.core_quads[q],
                cb_opt,
            );
            g_core_ux_l[t] += add_l;
            g_core_ux_r[t] += add_r;
        }
    }

    let (_, d_ro_l) = eval_ro_pole_hp_and_adjoint(
        c_ux_l_ref,
        model.ro_pole_48k,
        rho_core,
        Some(&d_hp_l),
        Some(&mut g_core_ux_l),
    );
    let (_, d_ro_r) = eval_ro_pole_hp_and_adjoint(
        c_ux_r_ref,
        model.ro_pole_48k,
        rho_core,
        Some(&d_hp_r),
        Some(&mut g_core_ux_r),
    );
    grads.d_ro_pole_48k = d_ro_l + d_ro_r;

    let (gx_ux_l, gx_ux_r) = if buf.os2x {
        (
            upsample_lane_2x_adjoint_f64(&g_core_ux_l, model.os_up_quad_coeff),
            upsample_lane_2x_adjoint_f64(&g_core_ux_r, model.os_up_quad_coeff),
        )
    } else {
        (g_core_ux_l, g_core_ux_r)
    };

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
