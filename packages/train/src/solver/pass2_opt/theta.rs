//! Flattened Pass-2 optimization vector layout ([`ThetaLayout`]) mapping `Op-N[Pre]`, Block 0
//! `svf_k_nl`, `CoreQuad` ADAA/readouts, Kautz poles/residues, and `Op-T` delays/`scat_psi`/taps.

use core::f64::consts::PI;

use dnam_kernels::{MAX_SVF_K_NL, TANK_MAX_G_LIN_F32};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;

use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::model::SonaModelF64;
use super::eval::Pass2ParamGrads;

/// Continuous optimization layout bridging [`SonaModelF64`] and Pass-2 L-BFGS / STE QAT solvers.
pub(super) struct ThetaLayout {
    pub q_opt: usize,
    pub k_opt: usize,
    pub op_t_enabled: bool,
    pub max_tau: f64,
}

impl ThetaLayout {
    /// Packs the active Pass-2 subspace from `m` alongside matching box bounds.
    pub(super) fn pack(&self, m: &SonaModelF64) -> (Vec<f64>, Vec<ParamBoundSpec>) {
        let g_max = f64::from(TANK_MAX_G_LIN_F32);
        let k_nl_max = f64::from(MAX_SVF_K_NL);
        let mut x = Vec::with_capacity(112);
        let mut b = Vec::with_capacity(112);

        x.push(m.op_n_pre_gamma_bc);
        b.push(ParamBoundSpec::Box { lo: 0.0, hi: 1.0 });
        x.push(m.op_n_pre_beta_q);
        b.push(ParamBoundSpec::Box { lo: 0.0, hi: 16.0 });

        for q in 0..self.q_opt {
            let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0;
            for k in 0..4 {
                x.push(m.core_quads[q].adaa_drive[k]);
                b.push(ParamBoundSpec::Box {
                    lo: -32.0,
                    hi: 32.0,
                });
                for col in 6..10 {
                    x.push(m.w_ro[q * 4 + k][col]);
                    b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });
                }
            }
            if is_block0 {
                for k in 0..4 {
                    x.push(m.core_quads[q].svf_k_nl[k]);
                    b.push(ParamBoundSpec::Box {
                        lo: 0.0,
                        hi: k_nl_max,
                    });
                }
            }
        }
        x.push(m.direct_s2_u_gain[0]);
        b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });
        x.push(m.direct_s2_u_gain[1]);
        b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });

        for k in 0..self.k_opt {
            let kp = &m.kautz[k];
            x.push(kp.r_low);
            b.push(ParamBoundSpec::Box {
                lo: 0.25,
                hi: 0.999,
            });
            x.push(kp.theta);
            b.push(ParamBoundSpec::Box {
                lo: 1.0e-4,
                hi: PI - 1.0e-4,
            });
            x.push(kp.c_re_l);
            b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });
            x.push(kp.c_im_l);
            b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });
            x.push(kp.c_re_r);
            b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });
            x.push(kp.c_im_r);
            b.push(ParamBoundSpec::Box { lo: -8.0, hi: 8.0 });
        }

        if self.op_t_enabled {
            let tq = &m.transport_quads[0];
            for k in 0..4 {
                x.push(tq.delay_samples[k]);
                b.push(ParamBoundSpec::Box {
                    lo: 5.0,
                    hi: self.max_tau,
                });
            }
            x.push(tq.g_loop);
            b.push(ParamBoundSpec::Box { lo: 0.0, hi: g_max });
            x.push(tq.damp_pole);
            b.push(ParamBoundSpec::Box { lo: 0.0, hi: 0.999 });
            x.push(tq.scat_psi);
            b.push(ParamBoundSpec::Box { lo: -PI, hi: PI });
            for k in 0..4 {
                x.push(tq.input_gains[k]);
                b.push(ParamBoundSpec::Box { lo: -2.0, hi: 2.0 });
            }
            for &wt in &[
                tq.wet_tap_l[0],
                tq.wet_tap_l[2],
                tq.wet_tap_r[1],
                tq.wet_tap_r[3],
            ] {
                x.push(wt);
                b.push(ParamBoundSpec::Box { lo: -2.0, hi: 2.0 });
            }
        }
        (x, b)
    }

    /// Clamps and unpacks `cand` into `m`, enforcing full Table-4 stability bounds on exit.
    pub(super) fn unpack(&self, cand: &[f64], m: &mut SonaModelF64) {
        #[inline]
        fn clamp_finite(v: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        }
        let g_max = f64::from(TANK_MAX_G_LIN_F32);
        let k_nl_max = f64::from(MAX_SVF_K_NL);
        let mut p = 0_usize;
        m.op_n_pre_gamma_bc = clamp_finite(cand[p], 0.0, 1.0, 0.0);
        m.op_n_pre_beta_q = clamp_finite(cand[p + 1], 0.0, 16.0, 0.0);
        p += 2;

        for q in 0..self.q_opt {
            let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0;
            for k in 0..4 {
                m.core_quads[q].adaa_drive[k] = clamp_finite(cand[p], -32.0, 32.0, 0.0);
                p += 1;
                for col in 6..10 {
                    m.w_ro[q * 4 + k][col] = clamp_finite(cand[p], -8.0, 8.0, 0.0);
                    p += 1;
                }
            }
            if is_block0 {
                for k in 0..4 {
                    m.core_quads[q].svf_k_nl[k] = clamp_finite(cand[p], 0.0, k_nl_max, 0.0);
                    p += 1;
                }
            }
        }
        m.direct_s2_u_gain = [
            clamp_finite(cand[p], -8.0, 8.0, 0.0),
            clamp_finite(cand[p + 1], -8.0, 8.0, 0.0),
        ];
        p += 2;

        for k in 0..self.k_opt {
            let kp = &mut m.kautz[k];
            kp.r_low = clamp_finite(cand[p], 0.25, 0.999, 0.5);
            kp.dr = clamp_finite(kp.dr, 0.0, (0.9995 - kp.r_low).max(0.0), 0.0);
            kp.theta = clamp_finite(cand[p + 1], 1.0e-4, PI - 1.0e-4, 0.1);
            kp.c_re_l = clamp_finite(cand[p + 2], -8.0, 8.0, 0.0);
            kp.c_im_l = clamp_finite(cand[p + 3], -8.0, 8.0, 0.0);
            kp.c_re_r = clamp_finite(cand[p + 4], -8.0, 8.0, 0.0);
            kp.c_im_r = clamp_finite(cand[p + 5], -8.0, 8.0, 0.0);
            p += 6;
        }

        if self.op_t_enabled {
            let tq = &mut m.transport_quads[0];
            for k in 0..4 {
                tq.delay_samples[k] = clamp_finite(cand[p], 5.0, self.max_tau, 5.0);
                p += 1;
            }
            tq.g_loop = clamp_finite(cand[p], 0.0, g_max, 0.0);
            tq.damp_pole = clamp_finite(cand[p + 1], 0.0, 0.999, 0.0);
            tq.scat_psi = clamp_finite(cand[p + 2], -PI, PI, 0.0);
            p += 3;
            for k in 0..4 {
                tq.input_gains[k] = clamp_finite(cand[p], -2.0, 2.0, 0.0);
                p += 1;
            }
            tq.wet_tap_l[0] = clamp_finite(cand[p], -2.0, 2.0, 0.0);
            tq.wet_tap_l[2] = clamp_finite(cand[p + 1], -2.0, 2.0, 0.0);
            tq.wet_tap_r[1] = clamp_finite(cand[p + 2], -2.0, 2.0, 0.0);
            tq.wet_tap_r[3] = clamp_finite(cand[p + 3], -2.0, 2.0, 0.0);
        }
        m.clamp_all_stability_bounds(true);
    }

    /// Writes `grads` into `out` in the exact coordinate order produced by [`Self::pack`].
    pub(super) fn write_grad(&self, grads: &Pass2ParamGrads, out: &mut [f64]) {
        let mut p = 0_usize;
        out[p] = grads.d_gamma_bc;
        out[p + 1] = grads.d_beta_q;
        p += 2;

        for q in 0..self.q_opt {
            let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0;
            for k in 0..4 {
                out[p] = grads.d_adaa[q][k];
                p += 1;
                for c in 0..4 {
                    out[p] = grads.d_w_ro_direct[q][k][c];
                    p += 1;
                }
            }
            if is_block0 {
                for k in 0..4 {
                    out[p] = grads.d_svf_k_nl[q][k];
                    p += 1;
                }
            }
        }
        out[p] = grads.d_direct_s2_u[0];
        out[p + 1] = grads.d_direct_s2_u[1];
        p += 2;

        for k in 0..self.k_opt {
            out[p..p + 6].copy_from_slice(&grads.d_kautz[k]);
            p += 6;
        }
        if self.op_t_enabled {
            for k in 0..4 {
                out[p] = grads.d_op_t.d_delay_samples[k];
                p += 1;
            }
            out[p] = grads.d_op_t.d_g_loop;
            out[p + 1] = grads.d_op_t.d_damp_pole;
            out[p + 2] = grads.d_op_t.d_scat_psi;
            p += 3;
            for k in 0..4 {
                out[p] = grads.d_op_t.d_input_gains[k];
                p += 1;
            }
            out[p] = grads.d_op_t.d_wet_tap_l[0];
            out[p + 1] = grads.d_op_t.d_wet_tap_l[2];
            out[p + 2] = grads.d_op_t.d_wet_tap_r[1];
            out[p + 3] = grads.d_op_t.d_wet_tap_r[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::{CompositeLossConfig, CompositeLossEvaluator};
    use crate::runner::TrainRunner;
    use crate::solver::pass2_opt::eval::{
        Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio,
    };

    #[test]
    fn test_theta_layout_roundtrip_and_block0_k_nl_and_scat_psi_fd_parity() {
        let runner = TrainRunner::cpu();
        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].svf_g = [0.18, 0.25, 0.42, 0.65];
        model.core_quads[0].svf_k_nl = [0.6, 0.4, 0.25, 0.15];
        model.core_quads[0].adaa_drive = [2.2, 1.5, 1.8, 1.2];
        model.core_quads[0].tone_lp = [0.4, 0.2, -0.1, 0.3];
        model.core_quads[0].tone_bp = [0.3, 0.4, 0.2, -0.2];
        model.core_quads[0].in_mid_weight = [1.0; 4];
        model.core_quads[0].in_side_weight = [0.8; 4];
        model.core_quads[1].svf_k_nl = [1.5; 4];
        model.core_quads[1].in_mid_weight = [0.6; 4];
        model.core_quads[1].tone_bp = [0.25; 4];
        model.w_ro[0][6..10].copy_from_slice(&[0.5, -0.3, 0.2, 0.15]);
        model.w_ro[1][6..10].copy_from_slice(&[0.35, 0.45, -0.15, 0.25]);
        model.w_ro[4][6..10].copy_from_slice(&[0.25, 0.25, 0.1, 0.1]);
        model.transport_quads[0].delay_samples = [7.25, 9.6, 12.4, 15.1];
        model.transport_quads[0].g_loop = 0.65;
        model.transport_quads[0].damp_pole = 0.3;
        model.transport_quads[0].scat_psi = 0.48;
        model.transport_quads[0].allpass_g = [0.3, -0.25, 0.35, -0.2];
        model.transport_quads[0].allpass_delays = [3, 5, 7, 11];
        model.transport_quads[0].input_gains = [0.5, 0.4, -0.3, 0.25];
        model.transport_quads[0].wet_tap_l = [0.6, 0.15, -0.4, 0.1];
        model.transport_quads[0].wet_tap_r = [0.1, -0.55, 0.2, 0.35];

        let layout = ThetaLayout {
            q_opt: 2,
            k_opt: 0,
            op_t_enabled: true,
            max_tau: 128.0,
        };
        let (theta, bounds) = layout.pack(&model);
        assert_eq!(theta.len(), bounds.len());
        // 2 (op_n_pre) + 2*20 (adaa+w_ro) + 4 (q=0 Block0 svf_k_nl; q=1 omitted) + 2 (direct_s2_u) + 15 (op_t incl scat_psi) = 63.
        assert_eq!(theta.len(), 63);

        let mut unpacked = model.clone();
        layout.unpack(&theta, &mut unpacked);
        assert_eq!(unpacked.core_quads[0].svf_k_nl, [0.6, 0.4, 0.25, 0.15]);
        // Blocks 1..=3 svf_k_nl are forced to 0 on unpack stability clamp.
        assert_eq!(unpacked.core_quads[1].svf_k_nl, [0.0; 4]);
        assert!((unpacked.transport_quads[0].scat_psi - 0.48).abs() < 1.0e-12);

        let n = 128_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 1.8 * ((t as f64) * 0.13).sin()).collect();
        let in_r: Vec<f64> = (0..n)
            .map(|t| 1.4 * ((t as f64) * 0.09 + 0.3).cos())
            .collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.4 * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.35 * v).collect();
        let loss_eval = CompositeLossEvaluator::new(
            &tgt_l,
            &tgt_r,
            CompositeLossConfig {
                sample_rate_hz: 48_000.0,
                warmup_frames: 16,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();

        model.clamp_all_stability_bounds(true);
        let buf = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, 2).unwrap();
        let mut gl = vec![0.0; n];
        let mut gr = vec![0.0; n];
        let _ = loss_eval
            .eval_and_adjoint(&buf.pred_l, &buf.pred_r, Some(&mut gl), Some(&mut gr))
            .unwrap();
        let grads = assemble_pass2_gradients(
            &runner,
            &Pass2AdjInput {
                model: &model,
                in_l: &in_l,
                in_r: &in_r,
                buf: &buf,
                gl: &gl,
                gr: &gr,
                q_opt: layout.q_opt,
                k_opt: layout.k_opt,
            },
        )
        .unwrap();
        assert_eq!(grads.d_svf_k_nl[1], [0.0; 4]);

        let (theta0, _) = layout.pack(&model);
        let mut grad_vec = vec![0.0_f64; theta0.len()];
        layout.write_grad(&grads, &mut grad_vec);

        let eval_at = |idx: usize, delta: f64| -> f64 {
            let mut t = theta0.clone();
            t[idx] += delta;
            let mut m = model.clone();
            layout.unpack(&t, &mut m);
            let b = eval_model_pass2_audio(&runner, &m, &in_l, &in_r, 2).unwrap();
            loss_eval
                .eval_and_adjoint(&b.pred_l, &b.pred_r, None, None)
                .unwrap()
                .total
        };

        // Index 2 + 20 = 22 is Block 0 svf_k_nl[0]; index 2 + 40 + 4 + 2 + 4 + 2 = 54 is Op-T scat_psi.
        for &(idx, eps) in &[(22_usize, 1.0e-4_f64), (54_usize, 1.0e-5_f64)] {
            let fd = (eval_at(idx, eps) - eval_at(idx, -eps)) / (2.0 * eps);
            let tol = 3.0e-2 * fd.abs().max(1.0);
            assert!(
                grad_vec[idx].abs() > 1.0e-5,
                "expected nonzero gradient at theta[{idx}], got {}",
                grad_vec[idx]
            );
            assert!(
                (grad_vec[idx] - fd).abs() <= tol,
                "ThetaLayout FD mismatch idx={idx}: adj={} fd={fd} tol={tol}",
                grad_vec[idx]
            );
        }

        let mut m_nan = model.clone();
        layout.unpack(&vec![f64::NAN; theta0.len()], &mut m_nan);
        assert!(m_nan.op_n_pre_gamma_bc.is_finite());
        assert!(m_nan.kautz[0].r_low.is_finite() && m_nan.kautz[0].dr.is_finite());
        assert!(m_nan.transport_quads[0].scat_psi.is_finite());
    }
}
