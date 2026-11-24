//! Frame-interleaved exact causal forward evaluator and reverse-time BPTT facade over
//! [`forward`] and [`adjoint`] when `Op-T` Top-of-Frame crossbar feedback closes a live recurrent
//! loop (`ref[train.pipeline.backbone-trajectory-opt]`, `ref[dsp.op-t.ring.declaration]`).

mod adjoint;
mod forward;
mod os2x;

pub(crate) use adjoint::assemble_causal_loop_adjoint;
pub(crate) use forward::eval_causal_loop_forward;

use super::super::model::SonaModelF64;
use super::adjoints::{KautzChannelStepper, RoPoleResolved};
use super::crossbar::CrossbarForwardTrace;
use super::eval::core_adj::active_opt_quads_for_core_quads;
use super::op_n_pre::OpNPreResolved;
use super::op_t_quad::forward::OpTQuadStepState;
use super::op_t_quad::is_op_t_quad_active;

/// Cached per-frame intermediates from [`eval_causal_loop_forward`] for exact reverse BPTT.
pub(crate) struct CausalLoopBuffers {
    pub q_count: usize,
    pub os2x: bool,
    pub active_opt_quads: usize,
    pub active_k: usize,
    pub active_fir: usize,
    pub op_t_gain: f64,
    pub op_t_gain_open: bool,
    pub dry_mix: f64,
    pub dry_mix_open: bool,
    pub wet_scale: f64,
    pub wet_scale_open: bool,
    pub rho_core: f64,
    pub pred_l: Vec<f64>,
    pub pred_r: Vec<f64>,
    pub op_t_wet_l: Vec<f64>,
    pub op_t_wet_r: Vec<f64>,
    pub s0_l: Vec<f64>,
    pub s0_r: Vec<f64>,
    pub ux_l: Vec<f64>,
    pub ux_r: Vec<f64>,
    pub core_ux_l: Vec<f64>,
    pub core_ux_r: Vec<f64>,
    pub onset_pre_l: Vec<f64>,
    pub onset_pre_r: Vec<f64>,
    pub x_tank_l: Vec<f64>,
    pub x_tank_r: Vec<f64>,
    pub cb_trace: CrossbarForwardTrace,
    pub core_inputs: Vec<[f32; 4]>,
    pub tile_checkpoints: Vec<[f32; 4]>,
    pub feature_outputs: Vec<[f32; 4]>,
    pub op_n_resolved: OpNPreResolved,
    pub op_n_held_l: Vec<f64>,
    pub op_n_held_r: Vec<f64>,
    pub op_n_gclk_l: Vec<f64>,
    pub op_n_dg_l: Vec<f64>,
    pub op_n_gclk_r: Vec<f64>,
    pub op_n_dg_r: Vec<f64>,
    pub ro_resolved: RoPoleResolved,
    pub ro_lp_l: Vec<f64>,
    pub ro_lp_r: Vec<f64>,
    pub kautz_l: KautzChannelStepper,
    pub kautz_r: KautzChannelStepper,
    pub opt_active: [bool; 4],
    pub opt_fwd: Vec<Option<OpTQuadStepState>>,
    pub v_reads_hist: Vec<[[f64; 4]; 4]>,
}

/// Returns `true` when Top-of-Frame `Op-T -> opt_to_lr -> s0` crossbar path has either non-zero
/// `crossbar_op_gains[1]` or non-zero active `opt_to_lr` taps with an active `TransportQuad`
/// (preserving escape gradients from one-sided zero states `(0, !=0)` and `(!=0, 0)`).
#[inline]
#[must_use]
pub(crate) fn is_op_t_crossbar_coupled(model: &SonaModelF64, active_quads: usize) -> bool {
    let op_t_gain = model.crossbar_op_gains[1];
    let has_gain = op_t_gain.is_finite() && op_t_gain.abs() > 1.0e-12;
    let q_count = active_quads.clamp(1, 10);
    let active_opt = active_opt_quads_for_core_quads(q_count);
    let any_tap = model.opt_to_lr[..(active_opt * 4)]
        .iter()
        .any(|[wl, wr]| wl.abs() > 1.0e-12 || wr.abs() > 1.0e-12);
    let any_quad = (0..active_opt).any(|q| is_op_t_quad_active(&model.transport_quads[q]));
    (has_gain || any_tap) && any_quad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::abi::FEATURE_VEC4_STRIDE;
    use crate::runner::TrainRunner;
    use crate::solver::model::TransportQuadF64;
    use crate::solver::model::cascade::run_causal_core_cascade_stereo;
    use crate::solver::pass2_opt::op_n_pre::eval_op_n_pre_and_adjoint;
    use crate::solver::pass2_opt::op_t_quad::{
        OpTQuadAdjointSlices, eval_op_t_quad_stereo_and_adjoint,
    };

    fn base_coupled_model(sr_hz: f64) -> SonaModelF64 {
        let mut m = SonaModelF64::new(sr_hz);
        m.sense_in_l = [0.65, -0.35, 0.25, 0.15];
        m.sense_in_r = [-0.25, 0.70, 0.20, -0.30];
        m.sense_vca_w = [1.4, -0.9, 0.6, 0.45];
        m.phasor_freq_hz = [-140.0, 260.0, -420.0, 810.0];
        m.phasor_init_phase_q32 = [0x1000_0000, 0x3500_0000, 0x7000_0000, 0xB000_0000];
        m.crossbar_op_gains = [0.65, 0.55];
        m.carrier_basis_w = [
            [0.5, 0.3, -0.2, 0.25],
            [-0.25, 0.45, 0.3, -0.2],
            [0.15, -0.2, 0.4, 0.3],
        ];
        m.op_p_to_core[0] = [0.22, -0.18, 0.14, -0.10];
        m.phasor_to_core[0] = [0.18, 0.15, -0.12, 0.09];
        m.opt_to_lr[0] = [0.45, -0.25];
        m.opt_to_lr[1] = [-0.20, 0.38];
        m.op_n_pre_gamma_bc = 0.55;
        m.op_n_pre_zoh_hold_depth = 0.32;
        m.op_n_pre_beta_q = 2.4;
        m.ro_pole_48k = 0.86;
        m.pre_conv_u_gain = [0.42, 0.35];
        m.pre_conv_hp_gain = [0.28, -0.22];
        m.direct_s2_u_gain = [0.32, 0.28];
        m.direct_s2_hp_gain = [-0.18, 0.24];
        m.core_quads[0].svf_g = [0.18, 0.32, 0.55, 0.90];
        m.core_quads[0].svf_k = [0.85, 1.1, 0.65, 1.35];
        m.core_quads[0].svf_k_nl = [0.5, 0.3, 0.2, 0.15];
        m.core_quads[0].adaa_drive = [2.4, 1.6, 1.2, 0.9];
        m.core_quads[0].tone_dir = [0.6, 0.35, 0.25, -0.2];
        m.core_quads[0].tone_bp = [0.3, -0.25, 0.4, 0.2];
        m.core_quads[1].svf_g = [0.22, 0.48, 0.85, 1.4];
        m.core_quads[1].adaa_drive = [1.5, 2.0, 0.8, 1.1];
        m.tiles[0].w_odd_4x4[0][0] = 0.38;
        m.tiles[0].w_even_4x4[1][2] = -0.22;
        m.tiles[0].w_lru_4x4[2][1] = 0.28;
        for l in 0..4 {
            m.w_ro[l] = [
                0.35, -0.2, 0.18, 0.14, 0.12, -0.1, 0.45, 0.38, -0.2, 0.16, 0.15, -0.12,
            ];
            m.w_ro[4 + l] = [
                0.15, 0.18, -0.1, 0.12, 0.08, 0.06, 0.22, -0.18, 0.14, 0.1, 0.09, 0.11,
            ];
        }
        m.active_fir_taps = 4;
        m.fir_l[..4].copy_from_slice(&[0.5, 0.25, -0.12, 0.08]);
        m.fir_r[..4].copy_from_slice(&[0.45, -0.2, 0.15, -0.06]);
        m.active_kautz_pairs = 2;
        for (idx, kp) in m.kautz[..2].iter_mut().enumerate() {
            kp.r_low = 0.82 - 0.1 * (idx as f64);
            kp.theta = 0.45 + 0.5 * (idx as f64);
            kp.b_gamma = 0.85;
            kp.c_re_l = 0.35;
            kp.c_im_l = -0.2;
            kp.c_re_r = 0.3;
            kp.c_im_r = 0.25;
        }
        m.tank_dry_mix = 0.72;
        m.transport_quads[0] = TransportQuadF64 {
            delay_samples: [6.4, 8.8, 11.3, 14.7],
            ramp_spans_samples: [1.5, 2.2, 1.1, 1.8],
            g_loop: 0.62,
            r_dc: 0.985,
            damp_pole: 0.28,
            lfo_rate_hz: 2.5,
            ramp_rate_hz: 1.75,
            lfo_depths: [0.4, 0.3, 0.5, 0.25],
            input_gains: [0.5, 0.35, -0.3, 0.22],
            wet_tap_l: [0.45, 0.2, -0.35, 0.15],
            wet_tap_r: [0.18, -0.42, 0.24, 0.32],
            allpass_g: [0.28, -0.22, 0.3, -0.18],
            allpass_delays: [3, 5, 7, 11],
            fb_psi: 0.28,
            sigma_knee: 0.58,
            freeze_mix: 0.15,
            window_depth: 0.3,
            scat_psi: 0.24,
            ..Default::default()
        };
        m
    }

    #[test]
    fn test_causal_loop_zero_gain_matches_dag_and_live_feedback_fd_parity() {
        let n = 160_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 0.4 * ((t as f64) * 0.11).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| -0.35 * ((t as f64) * 0.17).cos()).collect();
        let gl: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.23).cos()).collect();
        let gr: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.19).sin()).collect();
        let obj = |pl: &[f64], pr: &[f64]| -> f64 {
            pl.iter().zip(&gl).map(|(a, b)| a * b).sum::<f64>()
                + pr.iter().zip(&gr).map(|(a, b)| a * b).sum::<f64>()
        };

        let mut dag_m = base_coupled_model(48_000.0);
        dag_m.crossbar_op_gains = [0.0, 0.0];
        dag_m.opt_to_lr = [[0.0; 2]; 16];
        dag_m.sense_vca_w = [0.0; 4];
        dag_m.op_p_to_core = [[0.0; 4]; 10];
        dag_m.phasor_to_core = [[0.0; 4]; 10];
        dag_m.pre_conv_hp_gain = [0.0; 2];
        dag_m.direct_s2_hp_gain = [0.0; 2];
        dag_m.active_kautz_pairs = 0;
        assert!(!is_op_t_crossbar_coupled(&dag_m, 2));

        let cl_buf = eval_causal_loop_forward(&dag_m, &in_l, &in_r, 2, true);
        let cl_grad = assemble_causal_loop_adjoint(&dag_m, &in_l, &in_r, &cl_buf, &gl, &gr);

        let runner = TrainRunner::cpu();
        let (ux_l, _) = eval_op_n_pre_and_adjoint(
            &in_l,
            dag_m.op_n_pre_gamma_bc,
            dag_m.op_n_pre_zoh_hold_depth,
            dag_m.op_n_pre_beta_q,
            dag_m.sample_rate_hz,
            None,
            None,
        );
        let (ux_r, _) = eval_op_n_pre_and_adjoint(
            &in_r,
            dag_m.op_n_pre_gamma_bc,
            dag_m.op_n_pre_zoh_hold_depth,
            dag_m.op_n_pre_beta_q,
            dag_m.sample_rate_hz,
            None,
            None,
        );
        let cas =
            run_causal_core_cascade_stereo(&runner, &dag_m, &ux_l, &ux_r, 2, 1.0, true).unwrap();
        let mut pre_l = vec![0.0_f64; n];
        let mut pre_r = vec![0.0_f64; n];
        let mut xt_l = vec![0.0_f64; n];
        let mut xt_r = vec![0.0_f64; n];
        for t in 0..n {
            pre_l[t] = dag_m.pre_conv_u_gain[0] * ux_l[t];
            pre_r[t] = dag_m.pre_conv_u_gain[1] * ux_r[t];
            xt_l[t] = dag_m.direct_s2_u_gain[0] * ux_l[t];
            xt_r[t] = dag_m.direct_s2_u_gain[1] * ux_r[t];
            for q in 0..2 {
                let fb = (q * n + t) * FEATURE_VEC4_STRIDE;
                for k in 0..4 {
                    let r = &dag_m.w_ro[q * 4 + k];
                    pre_l[t] += r[0] * f64::from(cas.feature_outputs[fb + 2][k])
                        + r[2] * f64::from(cas.feature_outputs[fb + 4][k])
                        + r[4] * f64::from(cas.feature_outputs[fb][k]);
                    pre_r[t] += r[1] * f64::from(cas.feature_outputs[fb + 3][k])
                        + r[3] * f64::from(cas.feature_outputs[fb + 5][k])
                        + r[5] * f64::from(cas.feature_outputs[fb + 1][k]);
                    xt_l[t] += r[6] * f64::from(cas.feature_outputs[fb + 2][k])
                        + r[8] * f64::from(cas.feature_outputs[fb + 4][k])
                        + r[10] * f64::from(cas.feature_outputs[fb][k]);
                    xt_r[t] += r[7] * f64::from(cas.feature_outputs[fb + 3][k])
                        + r[9] * f64::from(cas.feature_outputs[fb + 5][k])
                        + r[11] * f64::from(cas.feature_outputs[fb + 1][k]);
                }
            }
            for k in 0..4.min(t + 1) {
                xt_l[t] += dag_m.fir_l[k] * pre_l[t - k];
                xt_r[t] += dag_m.fir_r[k] * pre_r[t - k];
            }
        }
        let (tl, tr, _) = eval_op_t_quad_stereo_and_adjoint(
            &xt_l,
            &xt_r,
            &dag_m.transport_quads[0],
            48_000.0,
            None,
        );
        for t in 0..n {
            assert!((cl_buf.pred_l[t] - (dag_m.tank_dry_mix * xt_l[t] + tl[t])).abs() < 1.0e-5);
            assert!((cl_buf.pred_r[t] - (dag_m.tank_dry_mix * xt_r[t] + tr[t])).abs() < 1.0e-5);
        }
        let mut dag_gx_l: Vec<f64> = gl.iter().map(|&v| dag_m.tank_dry_mix * v).collect();
        let mut dag_gx_r: Vec<f64> = gr.iter().map(|&v| dag_m.tank_dry_mix * v).collect();
        let (_, _, dag_opt) = eval_op_t_quad_stereo_and_adjoint(
            &xt_l,
            &xt_r,
            &dag_m.transport_quads[0],
            48_000.0,
            Some(OpTQuadAdjointSlices {
                d_out_l: &gl,
                d_out_r: &gr,
                grad_x_l: Some(&mut dag_gx_l),
                grad_x_r: Some(&mut dag_gx_r),
            }),
        );
        assert!((cl_grad.op_t[0].d_delay_samples[0] - dag_opt.d_delay_samples[0]).abs() < 1.0e-5);

        let mut failures = Vec::new();
        // 2a. Pure-f64 coupled loop verification (CoreQuad readouts 0 so zero f32 rounding jitter) at < 1e-4.
        for &sr in &[48_000.0_f64, 96_000.0_f64] {
            let mut m = base_coupled_model(sr);
            m.w_ro = [[0.0; 12]; 40];
            assert!(is_op_t_crossbar_coupled(&m, 2));
            let b = eval_causal_loop_forward(&m, &in_l, &in_r, 2, true);
            let g = assemble_causal_loop_adjoint(&m, &in_l, &in_r, &b, &gl, &gr);
            let eps = 1.0e-6_f64;
            let eval_m = |cand: &SonaModelF64| {
                let cb = eval_causal_loop_forward(cand, &in_l, &in_r, 2, false);
                obj(&cb.pred_l, &cb.pred_r)
            };
            let mut check_exact = |name: &str, ana: f64, edit: &dyn Fn(&mut SonaModelF64, f64)| {
                let (mut mp, mut mm) = (m.clone(), m.clone());
                edit(&mut mp, eps);
                edit(&mut mm, -eps);
                let fd = (eval_m(&mp) - eval_m(&mm)) / (2.0 * eps);
                let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                if (ana - fd).abs() >= 1.0e-4 * scale {
                    failures.push(format!("f64 sr={sr} {name}: ana={ana} fd={fd}"));
                }
            };
            check_exact("opt_to_lr[0][0]", g.crossbar.d_opt_to_lr[0][0], &|c, d| {
                c.opt_to_lr[0][0] += d
            });
            check_exact(
                "crossbar_op_gains[1]",
                g.crossbar.d_crossbar_op_gains[1],
                &|c, d| {
                    c.crossbar_op_gains[1] += d;
                },
            );
            check_exact("delay_samples[0]", g.op_t[0].d_delay_samples[0], &|c, d| {
                c.transport_quads[0].delay_samples[0] += d;
            });
            check_exact("sense_vca_w[0]", g.sensory.d_vca_w[0], &|c, d| {
                c.sense_vca_w[0] += d
            });
            check_exact("op_n_pre_gamma_bc", g.op_n_pre.d_gamma_bc, &|c, d| {
                c.op_n_pre_gamma_bc += d;
            });
            check_exact("ro_pole_48k", g.d_ro_pole_48k, &|c, d| c.ro_pole_48k += d);
            check_exact("kautz[0].b_gamma", g.d_kautz[0][2], &|c, d| {
                c.kautz[0].b_gamma += d
            });
            check_exact("tank_dry_mix", g.d_tank_dry_mix, &|c, d| {
                c.tank_dry_mix += d
            });
        }

        // 2b. Full mixed f32/f64 coupled loop verification at 48 kHz & 96 kHz with active
        // CoreQuads, PolyADAA1, svf_k_nl, InterBlockTile, Crossbar, and Op-T.
        for &sr in &[48_000.0_f64, 96_000.0_f64] {
            let mut m = base_coupled_model(sr);
            m.core_quads[0].lru_alpha_rel = [0.85; 4];
            m.core_quads[1].lru_alpha_rel = [0.85; 4];
            m.opt_to_lr[0] = [1.25, -0.95];
            m.opt_to_lr[1] = [-0.80, 1.15];
            m.sense_vca_w = [3.5, -2.2, 1.6, 1.1];
            m.op_p_to_core[0] = [0.65, -0.50, 0.40, -0.30];
            assert!(is_op_t_crossbar_coupled(&m, 2));
            let b = eval_causal_loop_forward(&m, &in_l, &in_r, 2, true);
            let g = assemble_causal_loop_adjoint(&m, &in_l, &in_r, &b, &gl, &gr);
            let eps = 2.0e-3_f64;
            let eval_m = |cand: &SonaModelF64| {
                let cb = eval_causal_loop_forward(cand, &in_l, &in_r, 2, false);
                obj(&cb.pred_l, &cb.pred_r)
            };
            let mut check_eps =
                |name: &str, ana: f64, step_eps: f64, edit: &dyn Fn(&mut SonaModelF64, f64)| {
                    let (mut mp, mut mm) = (m.clone(), m.clone());
                    edit(&mut mp, step_eps);
                    edit(&mut mm, -step_eps);
                    let fd = (eval_m(&mp) - eval_m(&mm)) / (2.0 * step_eps);
                    let scale = ana.abs().max(fd.abs()).max(0.5);
                    if (ana - fd).abs() >= 1.5e-2 * scale {
                        failures.push(format!("full sr={sr} {name}: ana={ana} fd={fd}"));
                    }
                };
            check_eps(
                "opt_to_lr[0][0]",
                g.crossbar.d_opt_to_lr[0][0],
                eps,
                &|c, d| c.opt_to_lr[0][0] += d,
            );
            check_eps(
                "crossbar_op_gains[1]",
                g.crossbar.d_crossbar_op_gains[1],
                eps,
                &|c, d| {
                    c.crossbar_op_gains[1] += d;
                },
            );
            check_eps(
                "delay_samples[0]",
                g.op_t[0].d_delay_samples[0],
                eps,
                &|c, d| {
                    c.transport_quads[0].delay_samples[0] += d;
                },
            );
            check_eps(
                "adaa_drive[0]",
                g.core_quads[0].adaa_drive[0],
                eps,
                &|c, d| {
                    c.core_quads[0].adaa_drive[0] += d;
                },
            );
            check_eps("svf_g[0]", g.core_quads[0].svf_g[0], 1.0e-3, &|c, d| {
                c.core_quads[0].svf_g[0] += d;
            });
            check_eps("op_n_pre_gamma_bc", g.op_n_pre.d_gamma_bc, eps, &|c, d| {
                c.op_n_pre_gamma_bc += d;
            });
            check_eps("ro_pole_48k", g.d_ro_pole_48k, eps, &|c, d| {
                c.ro_pole_48k += d
            });
            check_eps("kautz[0].b_gamma", g.d_kautz[0][2], eps, &|c, d| {
                c.kautz[0].b_gamma += d
            });
            check_eps("tank_dry_mix", g.d_tank_dry_mix, eps, &|c, d| {
                c.tank_dry_mix += d
            });
            check_eps(
                "tiles[0].w_odd[0][0]",
                g.tiles[0].w_odd_4x4[0][0],
                eps,
                &|c, d| {
                    c.tiles[0].w_odd_4x4[0][0] += d;
                },
            );
            check_eps(
                "in_mid_weight[0]",
                g.core_quads[0].in_mid_weight[0],
                eps,
                &|c, d| {
                    c.core_quads[0].in_mid_weight[0] += d;
                },
            );
            check_eps(
                "op_p_to_core[0][0]",
                g.crossbar.d_op_p_to_core[0][0],
                eps,
                &|c, d| {
                    c.op_p_to_core[0][0] += d;
                },
            );
            check_eps(
                "phasor_to_core[0][0]",
                g.crossbar.d_phasor_to_core[0][0],
                eps,
                &|c, d| {
                    c.phasor_to_core[0][0] += d;
                },
            );

            // Check sense_vca_w[0] with op_p_to_core = 0 so HilbertH1 path avoids T1 step jumps.
            let mut m_vca = m.clone();
            m_vca.op_p_to_core = [[0.0; 4]; 10];
            let bv = eval_causal_loop_forward(&m_vca, &in_l, &in_r, 2, true);
            let gv = assemble_causal_loop_adjoint(&m_vca, &in_l, &in_r, &bv, &gl, &gr);
            let (mut vp, mut vm) = (m_vca.clone(), m_vca);
            vp.sense_vca_w[0] += eps;
            vm.sense_vca_w[0] -= eps;
            let fd_vca = (eval_m(&vp) - eval_m(&vm)) / (2.0 * eps);
            let scale_vca = gv.sensory.d_vca_w[0].abs().max(fd_vca.abs()).max(0.5);
            if (gv.sensory.d_vca_w[0] - fd_vca).abs() >= 1.5e-2 * scale_vca {
                failures.push(format!(
                    "full sr={sr} sense_vca_w[0]: ana={} fd={fd_vca}",
                    gv.sensory.d_vca_w[0]
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "FD mismatches:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn test_causal_loop_one_sided_zero_coupling_escapes_match_central_fd() {
        let n = 128_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 0.4 * ((t as f64) * 0.11).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| -0.35 * ((t as f64) * 0.17).cos()).collect();
        let gl: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.23).cos()).collect();
        let gr: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.19).sin()).collect();
        let obj = |pl: &[f64], pr: &[f64]| -> f64 {
            pl.iter().zip(&gl).map(|(a, b)| a * b).sum::<f64>()
                + pr.iter().zip(&gr).map(|(a, b)| a * b).sum::<f64>()
        };
        let eps = 1.0e-6_f64;

        let mut both_zero = base_coupled_model(48_000.0);
        both_zero.w_ro = [[0.0; 12]; 40];
        both_zero.crossbar_op_gains[1] = 0.0;
        both_zero.opt_to_lr = [[0.0; 2]; 16];
        assert!(!is_op_t_crossbar_coupled(&both_zero, 2));

        // Case A: crossbar_op_gains[1] != 0, opt_to_lr == 0 -> non-zero d_opt_to_lr matches FD.
        let mut gain_only = both_zero.clone();
        gain_only.crossbar_op_gains[1] = 0.25;
        assert!(is_op_t_crossbar_coupled(&gain_only, 2));
        let buf_a = eval_causal_loop_forward(&gain_only, &in_l, &in_r, 2, true);
        let grad_a = assemble_causal_loop_adjoint(&gain_only, &in_l, &in_r, &buf_a, &gl, &gr);
        let (mut mp, mut mm) = (gain_only.clone(), gain_only);
        mp.opt_to_lr[0][0] += eps;
        mm.opt_to_lr[0][0] -= eps;
        let fd_a = (obj(
            &eval_causal_loop_forward(&mp, &in_l, &in_r, 2, false).pred_l,
            &eval_causal_loop_forward(&mp, &in_l, &in_r, 2, false).pred_r,
        ) - obj(
            &eval_causal_loop_forward(&mm, &in_l, &in_r, 2, false).pred_l,
            &eval_causal_loop_forward(&mm, &in_l, &in_r, 2, false).pred_r,
        )) / (2.0 * eps);
        let ana_a = grad_a.crossbar.d_opt_to_lr[0][0];
        assert!(ana_a.abs() > 1.0e-4, "expected non-zero escape d_opt_to_lr");
        assert!(
            (ana_a - fd_a).abs() < 1.0e-4 * ana_a.abs().max(fd_a.abs()).max(1.0e-3),
            "ana={ana_a} fd={fd_a}"
        );

        // Case B: crossbar_op_gains[1] == 0, opt_to_lr != 0 -> non-zero d_crossbar_op_gains[1] matches FD.
        let mut tap_only = both_zero;
        tap_only.opt_to_lr[0] = [0.22, -0.18];
        assert!(is_op_t_crossbar_coupled(&tap_only, 2));
        let buf_b = eval_causal_loop_forward(&tap_only, &in_l, &in_r, 2, true);
        let grad_b = assemble_causal_loop_adjoint(&tap_only, &in_l, &in_r, &buf_b, &gl, &gr);
        let (mut mp, mut mm) = (tap_only.clone(), tap_only);
        mp.crossbar_op_gains[1] += eps;
        mm.crossbar_op_gains[1] -= eps;
        let fd_b = (obj(
            &eval_causal_loop_forward(&mp, &in_l, &in_r, 2, false).pred_l,
            &eval_causal_loop_forward(&mp, &in_l, &in_r, 2, false).pred_r,
        ) - obj(
            &eval_causal_loop_forward(&mm, &in_l, &in_r, 2, false).pred_l,
            &eval_causal_loop_forward(&mm, &in_l, &in_r, 2, false).pred_r,
        )) / (2.0 * eps);
        let ana_b = grad_b.crossbar.d_crossbar_op_gains[1];
        assert!(
            ana_b.abs() > 1.0e-4,
            "expected non-zero escape d_crossbar_op_gains[1]"
        );
        assert!(
            (ana_b - fd_b).abs() < 1.0e-4 * ana_b.abs().max(fd_b.abs()).max(1.0e-3),
            "ana={ana_b} fd={fd_b}"
        );
    }

    #[test]
    fn test_causal_loop_os2x_vm_parity_and_central_fd_q6() {
        use crate::solver::model::snap_model_fp16;
        use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};
        use dnam_vm::{SonaVm, Tier, bin::DnamBinView, math::DefaultMathBackend};

        let n = 128_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n).map(|t| 0.4 * ((t as f64) * 0.08).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| 0.35 * ((t as f64) * 0.13).cos()).collect();
        let gl: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.17).cos()).collect();
        let gr: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.19).sin()).collect();

        let mut model = base_coupled_model(sr);
        model.oversample_upper_tiers_2x = true;
        model.active_fir_taps = 4;
        model.fir_l[0] = 0.25;
        model.fir_l[1] = -0.08;
        model.fir_r[0] = 0.22;
        model.fir_r[1] = -0.06;
        model.active_kautz_pairs = 2;
        model.kautz[0].r_low = 0.72;
        model.kautz[0].theta = 0.4;
        model.kautz[0].b_gamma = 0.5;
        model.kautz[0].c_re_l = 0.2;
        model.kautz[0].c_re_r = 0.18;
        model.core_quads[4].adaa_drive = [1.4, 1.1, 0.9, 0.7];
        model.core_quads[5].svf_g = [0.45, 0.65, 0.85, 1.1];
        model.tiles[4].w_odd_4x4[0][0] = 0.25;
        model.w_ro[20][6] = 0.14;
        model.w_ro[20][7] = 0.12;
        model.w_ro[20][0] = 0.11;
        model.w_ro[20][1] = 0.09;
        model.phasor_stereo_phase = [0.2, -0.3, 0.15, -0.25];
        snap_model_fp16(&mut model);

        // Check pre-Pass-3 2x VM parity on Tier::Full (6 quads, 12 Kautz / clipped active).
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
        let mut vm =
            SonaVm::<DefaultMathBackend>::from_view_owned(&view, Tier::Full, sr as f32, false)
                .unwrap();
        let in_l32: Vec<f32> = in_l.iter().map(|&v| v as f32).collect();
        let in_r32: Vec<f32> = in_r.iter().map(|&v| v as f32).collect();
        let (mut vm_l, mut vm_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        vm.process_block(&in_l32, &in_r32, &mut vm_l, &mut vm_r)
            .unwrap();

        let buf = eval_causal_loop_forward(&model, &in_l, &in_r, 6, true);
        assert!(buf.os2x);
        assert_eq!(buf.core_ux_l.len(), 2 * n);
        let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
        for t in 16..n {
            let (vl, vr) = (f64::from(vm_l[t]), f64::from(vm_r[t]));
            sig_sq += vl * vl + vr * vr;
            err_sq += (buf.pred_l[t] - vl).powi(2) + (buf.pred_r[t] - vr).powi(2);
        }
        let err_db = 10.0 * (err_sq / sig_sq.max(1.0e-18)).max(1.0e-16).log10();
        assert!(
            err_db < -60.0,
            "2x Full closed-loop VM mismatch {err_db} dB"
        );

        // 1) Pure-f64 2x upsample/decimate/ro_pole/Op-T/Op-N[Pre] closed-loop FD parity to < 1e-7.
        let mut p64 = model.clone();
        p64.w_ro = [[0.0; 12]; 40];
        p64.direct_s2_u_gain = [0.65, 0.55];
        p64.pre_conv_u_gain = [0.35, 0.25];
        p64.opt_to_lr[0] = [0.85, -0.55];
        let b64 = eval_causal_loop_forward(&p64, &in_l, &in_r, 6, true);
        let g64 = assemble_causal_loop_adjoint(&p64, &in_l, &in_r, &b64, &gl, &gr);
        let eval_m = |m: &SonaModelF64| -> f64 {
            let b = eval_causal_loop_forward(m, &in_l, &in_r, 6, false);
            b.pred_l.iter().zip(&gl).map(|(x, g)| x * g).sum::<f64>()
                + b.pred_r.iter().zip(&gr).map(|(x, g)| x * g).sum::<f64>()
        };
        let eps64 = 1.0e-6_f64;
        for (name, ana, mut_fn) in [
            (
                "f64 opt_to_lr[0][0]",
                g64.crossbar.d_opt_to_lr[0][0],
                (|m, d| m.opt_to_lr[0][0] += d) as fn(&mut SonaModelF64, f64),
            ),
            ("f64 ro_pole_48k", g64.d_ro_pole_48k, |m, d| {
                m.ro_pole_48k += d
            }),
            ("f64 op_n_pre_gamma_bc", g64.op_n_pre.d_gamma_bc, |m, d| {
                m.op_n_pre_gamma_bc += d
            }),
        ] {
            let (mut mp, mut mm) = (p64.clone(), p64.clone());
            mut_fn(&mut mp, eps64);
            mut_fn(&mut mm, -eps64);
            let fd = (eval_m(&mp) - eval_m(&mm)) / (2.0 * eps64);
            assert!(
                (ana - fd).abs() < 1.0e-5 * ana.abs().max(fd.abs()).max(1.0e-3),
                "os2x {name}: ana={ana} fd={fd}"
            );
        }

        // 2) Full mixed f32/f64 Central-FD parity across q_count=6 (2x) matching line-346 eps.
        let grad = assemble_causal_loop_adjoint(&model, &in_l, &in_r, &buf, &gl, &gr);
        let eps = 2.0e-3_f64;
        type FdCheck = (&'static str, f64, fn(&mut SonaModelF64, f64));
        let checks: [FdCheck; 6] = [
            (
                "opt_to_lr[0][0]",
                grad.crossbar.d_opt_to_lr[0][0],
                |m, d| m.opt_to_lr[0][0] += d,
            ),
            ("op_n_pre_gamma_bc", grad.op_n_pre.d_gamma_bc, |m, d| {
                m.op_n_pre_gamma_bc += d
            }),
            ("ro_pole_48k", grad.d_ro_pole_48k, |m, d| m.ro_pole_48k += d),
            ("w_ro[20][6]", grad.d_w_ro[20][6], |m, d| m.w_ro[20][6] += d),
            (
                "tiles[4].odd[0][0]",
                grad.tiles[4].w_odd_4x4[0][0],
                |m, d| m.tiles[4].w_odd_4x4[0][0] += d,
            ),
            (
                "core_quads[5].adaa[0]",
                grad.core_quads[5].adaa_drive[0],
                |m, d| m.core_quads[5].adaa_drive[0] += d,
            ),
        ];
        for (name, ana, mut_fn) in checks {
            let (mut mp, mut mm) = (model.clone(), model.clone());
            mut_fn(&mut mp, eps);
            mut_fn(&mut mm, -eps);
            let fd = (eval_m(&mp) - eval_m(&mm)) / (2.0 * eps);
            let scale = ana.abs().max(fd.abs()).max(0.25);
            assert!(
                (ana - fd).abs() < 2.0e-2 * scale,
                "os2x closed-loop {name}: ana={ana} fd={fd}"
            );
        }
    }
}
