//! Causal Pass-2 forward audio evaluation and reverse-time adjoint assembly across `SensoryQuad`,
//! `Op-P` + signed `Q0.32` `Crossbar`, `Op-N[Pre]`, causal `CoreQuad` + `SONA_INTERBLOCK_TILES`
//! cascade, ZOH `ro_pole_48k` readout, stereo `ModalConvolver` (`FIR + Kautz`), and `1..=4`
//! `Op-T` recurrent transport quads (`ref[train.pipeline.backbone-trajectory-opt]`).

pub(super) mod core_adj;

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;

use crate::error::TrainError;
use crate::gpu::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE, TILE_FRAMES,
};
use crate::runner::TrainRunner;
use crate::runner::types::WaveCoreAdjointInput;

use super::super::model::cascade::{
    CascadeCrossbarSpec, CausalCascadeOutput, downsample_lane_2x_adjoint_f64,
    downsample_lane_2x_f64, pack_wave_quads_and_tiles, run_causal_core_cascade_with_crossbar,
    sanitize_unit_f64, upsample_lane_2x_adjoint_f64, upsample_stereo_2x_f64,
};
use super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64};
use super::adjoints::{eval_kautz_modal_channel_and_adjoint, eval_ro_pole_hp_and_adjoint};
use super::causal_loop::{
    CausalLoopBuffers, assemble_causal_loop_adjoint, eval_causal_loop_forward,
    is_op_t_crossbar_coupled,
};
use super::crossbar::{
    CrossbarForwardTrace, assemble_crossbar_adjoint, eval_crossbar_forward,
    eval_crossbar_forward_os2x,
};
use super::op_n_pre::eval_op_n_pre_and_adjoint;
use super::op_t_quad::{OpTQuadAdjointSlices, eval_op_t_bank_stereo_and_adjoint};
use super::sensory::eval_sensory_quad_and_adjoint;
use core_adj::{
    active_opt_quads_for_core_quads, active_tiles_for_quads, add_tile_param_grads_to_quad,
    add_wave_tile_weight_grads, backprop_local_core_input_frame, finalize_core_quad_zoh,
};

pub(super) use core_adj::Pass2FullGrads as Pass2ParamGrads;

/// Forward intermediates cached during Pass-2 evaluation for exact reverse-time BPTT.
pub(super) struct Pass2EvalBuffers {
    pub os2x: bool,
    pub u_sense_l: Vec<f64>,
    pub u_sense_r: Vec<f64>,
    pub cb_trace: CrossbarForwardTrace,
    pub ux_l: Vec<f64>,
    pub ux_r: Vec<f64>,
    pub core_ux_l: Vec<f64>,
    pub core_ux_r: Vec<f64>,
    pub core_ro_hp_l: Vec<f64>,
    pub core_ro_hp_r: Vec<f64>,
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
            os2x: false,
            u_sense_l: Vec::new(),
            u_sense_r: Vec::new(),
            cb_trace: cb_dummy,
            ux_l: Vec::new(),
            ux_r: Vec::new(),
            core_ux_l: Vec::new(),
            core_ux_r: Vec::new(),
            core_ro_hp_l: Vec::new(),
            core_ro_hp_r: Vec::new(),
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
    let os2x = model.oversample_upper_tiers_2x && q_count > 4;
    let cb_trace = if os2x {
        eval_crossbar_forward_os2x(model, &u_sense_l, &u_sense_r, q_count, include_stereo_phase)
    } else {
        eval_crossbar_forward(model, &u_sense_l, &u_sense_r, q_count, include_stereo_phase)
    };

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

    let (core_ux_l, core_ux_r) = if os2x {
        upsample_stereo_2x_f64(&ux_l, &ux_r, model.os_up_quad_coeff)
    } else {
        (Vec::new(), Vec::new())
    };
    let (c_ux_l_ref, c_ux_r_ref): (&[f64], &[f64]) = if os2x {
        (&core_ux_l, &core_ux_r)
    } else {
        (&ux_l, &ux_r)
    };
    let n_core = c_ux_l_ref.len();

    let rho_1x = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let rho_core = if os2x { 0.5 * rho_1x } else { rho_1x };
    let (core_ro_hp_l, _) =
        eval_ro_pole_hp_and_adjoint(c_ux_l_ref, model.ro_pole_48k, rho_core, None, None);
    let (core_ro_hp_r, _) =
        eval_ro_pole_hp_and_adjoint(c_ux_r_ref, model.ro_pole_48k, rho_core, None, None);

    let cb_add = cb_trace
        .active
        .then_some((cb_trace.cb_add_l.as_slice(), cb_trace.cb_add_r.as_slice()));
    let cascade = run_causal_core_cascade_with_crossbar(
        runner,
        model,
        c_ux_l_ref,
        c_ux_r_ref,
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
    let mut core_pre_l = vec![0.0_f64; n_core];
    let mut core_pre_r = vec![0.0_f64; n_core];
    let mut core_dir_l = vec![0.0_f64; n_core];
    let mut core_dir_r = vec![0.0_f64; n_core];

    for t in 0..n_core {
        let mut bus_l = pc_l * c_ux_l_ref[t] + pc_hp_l * core_ro_hp_l[t];
        let mut bus_r = pc_r * c_ux_r_ref[t] + pc_hp_r * core_ro_hp_r[t];
        let mut tank_l = dir_l * c_ux_l_ref[t] + dir_hp_l * core_ro_hp_l[t];
        let mut tank_r = dir_r * c_ux_r_ref[t] + dir_hp_r * core_ro_hp_r[t];
        for q in 0..q_count {
            let fbase = (q * n_core + t) * FEATURE_VEC4_STRIDE;
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
        core_pre_l[t] = bus_l;
        core_pre_r[t] = bus_r;
        core_dir_l[t] = tank_l;
        core_dir_r[t] = tank_r;
    }

    let (pre_conv_l, pre_conv_r, mut x_tank_l, mut x_tank_r) = if os2x {
        (
            downsample_lane_2x_f64(&core_pre_l, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_f64(&core_pre_r, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_f64(&core_dir_l, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_f64(&core_dir_r, model.os_dec_c_prev1, model.os_dec_c0),
        )
    } else {
        (core_pre_l, core_pre_r, core_dir_l, core_dir_r)
    };

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
        os2x,
        u_sense_l,
        u_sense_r,
        cb_trace,
        ux_l,
        ux_r,
        core_ux_l,
        core_ux_r,
        core_ro_hp_l,
        core_ro_hp_r,
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
    let wave_adj = runner.run_wave_core_adjoint(&WaveCoreAdjointInput {
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
                (q * num_tiles + tile) * QUAD_PARAMS_VEC4_STRIDE,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::snap_model_fp16;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};
    use dnam_vm::{SonaVm, Tier, bin::DnamBinView, math::DefaultMathBackend};

    #[test]
    fn test_pass2_os2x_vm_parity_all_tiers_and_central_fd_q6() {
        let runner = TrainRunner::cpu();
        let n = 128_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n).map(|t| 0.42 * ((t as f64) * 0.075).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| 0.36 * ((t as f64) * 0.125).cos()).collect();
        let in_l32: Vec<f32> = in_l.iter().map(|&v| v as f32).collect();
        let in_r32: Vec<f32> = in_r.iter().map(|&v| v as f32).collect();

        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = true;
        model.sense_in_l = [0.6, -0.3, 0.2, 0.1];
        model.sense_in_r = [-0.2, 0.65, 0.15, -0.25];
        model.sense_vca_w = [1.1, -0.7, 0.5, 0.3];
        model.crossbar_op_gains = [0.55, 0.0];
        model.phasor_freq_hz = [110.0, -230.0, 410.0, -780.0];
        model.phasor_stereo_phase = [0.2, -0.3, 0.15, -0.1];
        model.carrier_basis_w[0] = [0.4, -0.25, 0.18, 0.12];
        model.op_p_to_core[0] = [0.25, -0.15, 0.1, 0.08];
        model.op_p_to_core[5] = [-0.18, 0.22, 0.12, -0.05];
        model.phasor_to_core[0] = [0.22, 0.15, -0.12, 0.1];
        model.phasor_to_core[5] = [0.16, -0.2, 0.14, 0.08];
        model.op_n_pre_gamma_bc = 0.35;
        model.ro_pole_48k = 0.92;
        model.direct_s2_u_gain = [0.5, 0.45];
        model.direct_s2_hp_gain = [0.15, 0.12];
        model.pre_conv_u_gain = [0.22, 0.18];
        model.pre_conv_hp_gain = [0.08, -0.06];
        model.active_fir_taps = 4;
        model.fir_l[0] = 0.28;
        model.fir_l[1] = -0.09;
        model.fir_r[0] = 0.25;
        model.fir_r[1] = -0.07;
        model.active_kautz_pairs = 4;
        model.kautz[0].r_low = 0.7;
        model.kautz[0].theta = 0.42;
        model.kautz[0].b_gamma = 0.6;
        model.kautz[0].c_re_l = 0.15;
        model.kautz[0].c_re_r = 0.14;
        for q in 0..10 {
            model.core_quads[q].adaa_drive = [1.3, 0.9, 1.1, 0.8];
            model.w_ro[q * 4][6] = 0.12;
            model.w_ro[q * 4][7] = 0.11;
            model.w_ro[q * 4][0] = 0.09;
            model.w_ro[q * 4][1] = 0.08;
        }
        model.tiles[0].w_odd_4x4[0][0] = 0.25;
        model.tiles[4].w_odd_4x4[0][0] = 0.22;
        model.tiles[14].w_odd_4x4[0][0] = 0.18;
        model.tiles[24].w_odd_4x4[0][0] = 0.15;
        snap_model_fp16(&mut model);

        let bin = encode_model_to_dnam_bin(
            &model,
            Pass5EmitConfig {
                stereo_out: true,
                single_input: false,
                ..Pass5EmitConfig::default()
            },
        )
        .unwrap();
        let view = DnamBinView::parse(&bin).unwrap();
        for (tier, q_count) in [
            (Tier::Lite, 4),
            (Tier::Full, 6),
            (Tier::Pro, 8),
            (Tier::Ultra, 10),
        ] {
            let mut vm =
                SonaVm::<DefaultMathBackend>::from_view_owned(&view, tier, sr as f32, false)
                    .unwrap();
            let (mut vm_l, mut vm_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
            vm.process_block(&in_l32, &in_r32, &mut vm_l, &mut vm_r)
                .unwrap();
            let buf = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, q_count).unwrap();
            assert_eq!(buf.os2x, q_count > 4);
            let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
            for t in 16..n {
                let (vl, vr) = (f64::from(vm_l[t]), f64::from(vm_r[t]));
                sig_sq += vl * vl + vr * vr;
                err_sq += (buf.pred_l[t] - vl).powi(2) + (buf.pred_r[t] - vr).powi(2);
            }
            let err_db = 10.0 * (err_sq / sig_sq.max(1.0e-18)).max(1.0e-16).log10();
            assert!(err_db < -60.0, "tier {tier:?} VM parity err={err_db} dB");
        }

        // Central-FD gradient parity for open-loop 2x at q_count = 6 across all stages.
        let gl: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.21).cos()).collect();
        let gr: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.17).sin()).collect();
        let buf = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, 6).unwrap();
        let grads = assemble_pass2_gradients(
            &runner,
            &Pass2AdjInput {
                model: &model,
                in_l: &in_l,
                in_r: &in_r,
                buf: &buf,
                gl: &gl,
                gr: &gr,
                q_opt: 6,
                k_opt: 4,
            },
        )
        .unwrap();
        let eval_fn = |m: &SonaModelF64| -> f64 {
            let b = eval_model_pass2_audio(&runner, m, &in_l, &in_r, 6).unwrap();
            b.pred_l.iter().zip(&gl).map(|(x, g)| x * g).sum::<f64>()
                + b.pred_r.iter().zip(&gr).map(|(x, g)| x * g).sum::<f64>()
        };
        let eps = 1.0e-5_f64;
        type FdCheck = (&'static str, f64, fn(&mut SonaModelF64, f64));
        let checks: [FdCheck; 7] = [
            ("sense_vca_w[0]", grads.sensory.d_vca_w[0], |m, d| {
                m.sense_vca_w[0] += d
            }),
            ("op_n_pre_gamma_bc", grads.op_n_pre.d_gamma_bc, |m, d| {
                m.op_n_pre_gamma_bc += d
            }),
            (
                "phasor_freq_hz[1]",
                grads.crossbar.d_phasor_freq_hz[1],
                |m, d| m.phasor_freq_hz[1] += d,
            ),
            ("ro_pole_48k", grads.d_ro_pole_48k, |m, d| {
                m.ro_pole_48k += d
            }),
            ("w_ro[20][6]", grads.d_w_ro[20][6], |m, d| {
                m.w_ro[20][6] += d
            }),
            (
                "tiles[4].odd[0][0]",
                grads.tiles[4].w_odd_4x4[0][0],
                |m, d| m.tiles[4].w_odd_4x4[0][0] += d,
            ),
            (
                "core_quads[5].adaa[0]",
                grads.core_quads[5].adaa_drive[0],
                |m, d| m.core_quads[5].adaa_drive[0] += d,
            ),
        ];
        for (name, ana, mut_fn) in checks {
            let step_eps = if name.contains("tiles") {
                1.0e-3_f64
            } else {
                eps
            };
            let (mut mp, mut mm) = (model.clone(), model.clone());
            mut_fn(&mut mp, step_eps);
            mut_fn(&mut mm, -step_eps);
            let fd = (eval_fn(&mp) - eval_fn(&mm)) / (2.0 * step_eps);
            let tol = if name.contains("core_quads") || name.contains("sense_vca") {
                2.0e-2
            } else {
                1.0e-4
            };
            let scale = ana.abs().max(fd.abs()).max(1.0e-3);
            assert!(
                (ana - fd).abs() < tol * scale,
                "os2x {name}: ana={ana} fd={fd}"
            );
        }
    }
}
