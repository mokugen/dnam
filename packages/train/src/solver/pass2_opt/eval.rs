//! Causal Pass-2 forward audio evaluation and reverse-time adjoint assembly across `SensoryQuad`,
//! `Op-P` + signed `Q0.32` `Crossbar`, `Op-N[Pre]`, causal `CoreQuad` + `SONA_INTERBLOCK_TILES`
//! cascade, ZOH `ro_pole_48k` readout, stereo `ModalConvolver` (`FIR + Kautz`), and `1..=4`
//! `Op-T` recurrent transport quads (`ref[train.pipeline.backbone-trajectory-opt]`).

mod adjoint;
pub(super) mod core_adj;

use crate::error::TrainError;
use crate::gpu::abi::FEATURE_VEC4_STRIDE;
use crate::runner::TrainRunner;

use super::super::model::cascade::{
    CascadeCrossbarSpec, CausalCascadeOutput, downsample_lane_2x_f64,
    run_causal_core_cascade_with_crossbar, sanitize_unit_f64, upsample_stereo_2x_f64,
};
use super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64};
use super::adjoints::{eval_kautz_modal_channel_and_adjoint, eval_ro_pole_hp_and_adjoint};
use super::causal_loop::{CausalLoopBuffers, eval_causal_loop_forward, is_op_t_crossbar_coupled};
use super::crossbar::{CrossbarForwardTrace, eval_crossbar_forward, eval_crossbar_forward_os2x};
use super::op_n_pre::eval_op_n_pre_and_adjoint;
use super::op_t_quad::eval_op_t_bank_stereo_and_adjoint;
use super::sensory::eval_sensory_quad_and_adjoint;
use core_adj::active_opt_quads_for_core_quads;

pub use adjoint::{Pass2AdjInput, assemble_pass2_gradients};
pub use core_adj::Pass2FullGrads as Pass2ParamGrads;

/// Forward intermediates cached during Pass-2 evaluation for exact reverse-time BPTT.
pub struct Pass2EvalBuffers {
    pub(super) os2x: bool,
    pub(super) u_sense_l: Vec<f64>,
    pub(super) u_sense_r: Vec<f64>,
    pub(super) cb_trace: CrossbarForwardTrace,
    pub(super) ux_l: Vec<f64>,
    pub(super) ux_r: Vec<f64>,
    pub(super) core_ux_l: Vec<f64>,
    pub(super) core_ux_r: Vec<f64>,
    pub(super) core_ro_hp_l: Vec<f64>,
    pub(super) core_ro_hp_r: Vec<f64>,
    pub(super) onset_pre_l: Vec<f64>,
    pub(super) onset_pre_r: Vec<f64>,
    pub(super) x_tank_l: Vec<f64>,
    pub(super) x_tank_r: Vec<f64>,
    pub(super) op_t_wet_l: Vec<f64>,
    pub(super) op_t_wet_r: Vec<f64>,
    pub(crate) pred_l: Vec<f64>,
    pub(crate) pred_r: Vec<f64>,
    pub(super) cascade: CausalCascadeOutput,
    pub(super) causal_loop: Option<CausalLoopBuffers>,
}

impl Pass2EvalBuffers {
    /// Returns the predicted left channel waveform.
    #[must_use]
    pub fn pred_l(&self) -> &[f64] {
        &self.pred_l
    }

    /// Returns the predicted right channel waveform.
    #[must_use]
    pub fn pred_r(&self) -> &[f64] {
        &self.pred_r
    }
}

impl core::fmt::Debug for Pass2EvalBuffers {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pass2EvalBuffers")
            .field("os2x", &self.os2x)
            .field("frames", &self.pred_l.len())
            .finish()
    }
}

#[inline]
fn delay16_slice(sig: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; sig.len()];
    if sig.len() > 16 {
        out[16..].copy_from_slice(&sig[..(sig.len() - 16)]);
    }
    out
}

/// Evaluates causal Pass-2 forward audio across all `SONA` subsystems, caching [`Pass2EvalBuffers`].
pub fn eval_model_pass2_audio(
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
            op_t_wet_l: Vec::new(),
            op_t_wet_r: Vec::new(),
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
    let wet_scale = sanitize_unit_f64(model.op_t_wet_scale, 0.0, 2.0, 1.0);
    let mut pred_l: Vec<f64> = x_tank_l.iter().map(|&v| dry_mix * v).collect();
    let mut pred_r: Vec<f64> = x_tank_r.iter().map(|&v| dry_mix * v).collect();
    let active_opt = active_opt_quads_for_core_quads(q_count);
    let (op_t_wet_l, op_t_wet_r, _) = eval_op_t_bank_stereo_and_adjoint(
        &x_tank_l,
        &x_tank_r,
        &model.transport_quads,
        active_opt,
        model.sample_rate_hz,
        model.dc_blocker_pole_5hz,
        None,
    );
    for t in 0..n {
        pred_l[t] += wet_scale * op_t_wet_l[t];
        pred_r[t] += wet_scale * op_t_wet_r[t];
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
        op_t_wet_l,
        op_t_wet_r,
        pred_l,
        pred_r,
        cascade,
        causal_loop: None,
    })
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
