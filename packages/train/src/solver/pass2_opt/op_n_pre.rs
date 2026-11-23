//! Native $C^\infty$ 3-parameter `Op-N[Pre]` raised-cosine track-and-hold + Fourier staircase
//! quantizer surrogate and exact reverse-time BPTT adjoints matching
//! `dnam_kernels::sona_core::op_n_pre` (`ref[dsp.op-n.bitcrusher.zoh-phase]`,
//! `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::{LN_2, TAU};

use dnam_kernels::{
    OP_N_PRE_BETA_BYPASS_EPS, OP_N_PRE_FULL_PASS_THRESH, OP_N_PRE_MAX_BETA_Q,
    OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OP_N_PRE_MIN_RHO_EFF, OP_N_PRE_QUANT_CLAMP_F32,
};

/// Hard safety limit enforcing `Op-N[Pre]` quantizer clipping `[-100, 100]`.
const OP_N_CLIP_F64: f64 = OP_N_PRE_QUANT_CLAMP_F32 as f64;
const _: () = assert!(OP_N_CLIP_F64 == 100.0);

/// Parameter gradients returned by [`eval_op_n_pre_and_adjoint`] (unscaled wrt `N`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpNPreGrads {
    /// Gradient wrt `op_n_pre_gamma_bc` in `[0.0, 1.0]`.
    pub d_gamma_bc: f64,
    /// Unscaled gradient wrt `op_n_pre_zoh_hold_depth` in `[0.0, 0.9999]`.
    pub d_zoh_hold_depth: f64,
    /// Gradient wrt `op_n_pre_beta_q` in `[0.0, 16.0]`.
    pub d_beta_q: f64,
}

impl OpNPreGrads {
    /// Element-wise accumulation of another channel's gradients into `self`.
    #[inline]
    pub fn accumulate(&mut self, rhs: Self) {
        self.d_gamma_bc += rhs.d_gamma_bc;
        self.d_zoh_hold_depth += rhs.d_zoh_hold_depth;
        self.d_beta_q += rhs.d_beta_q;
    }
}

/// Single-frame forward cache passed into [`OpNPreResolved::step_reverse_frame`].
#[derive(Clone, Copy, Debug)]
pub(super) struct OpNPreFrameSample {
    pub dy: f64,
    pub xt: f64,
    pub ht: f64,
    pub h_prev: f64,
    pub g_clk: f64,
    pub dg_dzh: f64,
}

/// Pre-resolved `Op-N[Pre]` frame-step parameters shared across batch evaluation and
/// frame-interleaved `CausalLoop` execution.
#[derive(Clone, Copy, Debug)]
pub(super) struct OpNPreResolved {
    gamma: f64,
    z_h: f64,
    z_open: bool,
    rho_eff: f64,
    d_rho_dzh: f64,
    clock_active: bool,
    bypass_beta: bool,
    beta_open: bool,
    d_scale: f64,
    delta_step: f64,
    inv_tau: f64,
}

impl OpNPreResolved {
    #[must_use]
    pub(super) fn new(
        gamma_bc: f64,
        zoh_hold_depth: f64,
        beta_q: f64,
        sample_rate_hz: f64,
    ) -> Self {
        let sr = if sample_rate_hz.is_finite() && sample_rate_hz > 0.0 {
            sample_rate_hz
        } else {
            48_000.0
        };
        let rho_vm = 48_000.0 / sr;
        let gamma = if gamma_bc.is_finite() {
            gamma_bc.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let z_max = f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH);
        let z_h = if zoh_hold_depth.is_finite() {
            zoh_hold_depth.clamp(0.0, z_max)
        } else {
            0.0
        };
        let z_open = zoh_hold_depth.is_finite() && zoh_hold_depth >= 0.0 && zoh_hold_depth <= z_max;
        let min_rho = f64::from(OP_N_PRE_MIN_RHO_EFF);
        let rho_raw = (1.0 - z_h) * rho_vm;
        let rho_eff = rho_raw.clamp(min_rho, 1.0);
        let d_rho_dzh = if z_open && rho_raw > min_rho && rho_raw < 1.0 {
            -rho_vm
        } else {
            0.0
        };
        let clock_active = z_h > 0.0 && rho_raw < f64::from(OP_N_PRE_FULL_PASS_THRESH);
        let bypass_beta = !beta_q.is_finite() || beta_q <= f64::from(OP_N_PRE_BETA_BYPASS_EPS);
        let beta = if bypass_beta {
            0.0
        } else {
            beta_q.clamp(0.0, f64::from(OP_N_PRE_MAX_BETA_Q))
        };
        let d_scale = beta.exp2();
        let delta_step = 1.0 / d_scale;
        let beta_open = beta_q.is_finite()
            && beta_q > f64::from(OP_N_PRE_BETA_BYPASS_EPS)
            && beta_q <= f64::from(OP_N_PRE_MAX_BETA_Q);
        Self {
            gamma,
            z_h,
            z_open,
            rho_eff,
            d_rho_dzh,
            clock_active,
            bypass_beta,
            beta_open,
            d_scale,
            delta_step,
            inv_tau: 1.0 / TAU,
        }
    }

    #[inline]
    pub(super) fn step_forward_frame(
        &self,
        t: usize,
        xt: f64,
        held_prev: &mut f64,
    ) -> (f64, f64, f64, f64) {
        let (g_clk, dg_dzh) = if self.clock_active {
            // `OpNPreHotState::step_clock_gain` advances phase_q32 *before* reading `cos(phi)`.
            let phi = TAU * ((t + 1) as f64) * self.rho_eff;
            let (sin_p, cos_p) = phi.sin_cos();
            let u = 0.5 * (1.0 + cos_p);
            let w = u * u;
            let g = (1.0 - self.z_h) + self.z_h * w;
            let dg = if self.z_open {
                (w - 1.0) - self.z_h * u * sin_p * TAU * ((t + 1) as f64) * self.d_rho_dzh
            } else {
                0.0
            };
            (g, dg)
        } else {
            (1.0, 0.0)
        };
        let h_t = (1.0 - g_clk) * *held_prev + g_clk * xt;
        *held_prev = h_t;
        let q_t = if self.gamma <= 0.0 {
            xt
        } else if self.bypass_beta {
            h_t.clamp(-OP_N_CLIP_F64, OP_N_CLIP_F64)
        } else {
            let hc = h_t.clamp(-OP_N_CLIP_F64, OP_N_CLIP_F64);
            hc - (self.delta_step * self.inv_tau) * (TAU * self.d_scale * hc).sin()
        };
        let y_t = (1.0 - self.gamma) * xt + self.gamma * q_t;
        (y_t, h_t, g_clk, dg_dzh)
    }

    #[inline]
    pub(super) fn step_reverse_frame(
        &self,
        frame: OpNPreFrameSample,
        adj_h_next: &mut f64,
        grads: &mut OpNPreGrads,
    ) -> f64 {
        let OpNPreFrameSample {
            dy,
            xt,
            ht,
            h_prev,
            g_clk,
            dg_dzh,
        } = frame;
        let (q_t, dq_dh, dq_dbeta) = if self.bypass_beta {
            let hc = ht.clamp(-OP_N_CLIP_F64, OP_N_CLIP_F64);
            let dh = if ht.abs() < OP_N_CLIP_F64 { 1.0 } else { 0.0 };
            (hc, dh, 0.0)
        } else {
            let hc = ht.clamp(-OP_N_CLIP_F64, OP_N_CLIP_F64);
            let (s, c) = (TAU * self.d_scale * hc).sin_cos();
            let qt = hc - (self.delta_step * self.inv_tau) * s;
            let dh = if ht.abs() < OP_N_CLIP_F64 {
                1.0 - c
            } else {
                0.0
            };
            let db = if self.beta_open {
                LN_2 * ((self.delta_step * self.inv_tau) * s - hc * c)
            } else {
                0.0
            };
            (qt, dh, db)
        };
        grads.d_gamma_bc += dy * (q_t - xt);
        grads.d_beta_q += dy * self.gamma * dq_dbeta;
        let adj_h = dy * self.gamma * dq_dh + *adj_h_next;
        grads.d_zoh_hold_depth += adj_h * (xt - h_prev) * dg_dzh;
        *adj_h_next = (1.0 - g_clk) * adj_h;
        dy * (1.0 - self.gamma) + adj_h * g_clk
    }
}

/// Evaluates single-channel `Op-N[Pre]` (1-indexed raised-cosine track-and-hold `zoh_hold_depth` ->
/// `beta_q <= 1e-6` bypass or smooth $C^\infty$ Fourier staircase -> `gamma_bc` wet blend) and
/// optionally accumulates reverse-time BPTT parameter and input gradients.
pub fn eval_op_n_pre_and_adjoint(
    input: &[f64],
    gamma_bc: f64,
    zoh_hold_depth: f64,
    beta_q: f64,
    sample_rate_hz: f64,
    d_out_opt: Option<&[f64]>,
    mut grad_in_opt: Option<&mut [f64]>,
) -> (Vec<f64>, OpNPreGrads) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), OpNPreGrads::default());
    }
    let resolved = OpNPreResolved::new(gamma_bc, zoh_hold_depth, beta_q, sample_rate_hz);
    let need_adj = d_out_opt.is_some();
    let mut y = vec![0.0_f64; n];
    let mut held_hist = if need_adj {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    let mut g_clk_hist = if need_adj {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    let mut dg_dzh_hist = if need_adj {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };

    let mut held_prev = 0.0_f64;
    for (t, &xt) in input.iter().enumerate() {
        let (y_t, h_t, g_clk, dg_dzh) = resolved.step_forward_frame(t, xt, &mut held_prev);
        y[t] = y_t;
        if need_adj {
            held_hist.push(h_t);
            g_clk_hist.push(g_clk);
            dg_dzh_hist.push(dg_dzh);
        }
    }

    let mut grads = OpNPreGrads::default();
    if let Some(d_out) = d_out_opt {
        let mut adj_h_next = 0.0_f64;
        for t in (0..n).rev() {
            let h_prev = if t > 0 { held_hist[t - 1] } else { 0.0 };
            let g_in = resolved.step_reverse_frame(
                OpNPreFrameSample {
                    dy: d_out[t],
                    xt: input[t],
                    ht: held_hist[t],
                    h_prev,
                    g_clk: g_clk_hist[t],
                    dg_dzh: dg_dzh_hist[t],
                },
                &mut adj_h_next,
                &mut grads,
            );
            if let Some(gx) = grad_in_opt.as_deref_mut() {
                gx[t] += g_in;
            }
        }
    }

    (y, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::{OP_N_PRE_MAX_BETA_Q, OpNPreHotState};

    #[test]
    fn test_op_n_pre_vm_bypass_parity_and_3param_central_fd() {
        let n = 160_usize;
        let sr = 96_000.0_f64;
        let x: Vec<f64> = (0..n)
            .map(|t| 0.48 * ((t as f64) * 0.11).sin() + 0.22 * ((t as f64) * 0.27).cos())
            .collect();
        let d_out: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.17).cos()).collect();
        let obj = |v: &[f64]| -> f64 { v.iter().zip(&d_out).map(|(a, b)| a * b).sum() };

        // 1. VM beta_q <= 1e-6 bypass parity (no Fourier ripple added when beta_q == 0.0).
        let (y_byp, _) = eval_op_n_pre_and_adjoint(&x, 0.75, 0.42, 0.0, 48_000.0, None, None);
        let mut kern = OpNPreHotState::zero();
        let hoisted = dnam_kernels::OpNPreHoisted::from_params(0.75, 0.42, 0.0, 1.0);
        kern.apply_hoisted(hoisted);
        let mut max_vm_diff = 0.0_f64;
        for (t, &xt) in x.iter().enumerate() {
            let (yl, _) =
                kern.step_stereo(xt as f32, xt as f32, hoisted.d_scale, hoisted.delta_step);
            max_vm_diff = max_vm_diff.max((f64::from(yl) - y_byp[t]).abs());
        }
        assert!(max_vm_diff < 5.0e-5, "bypass VM parity diff={max_vm_diff}");

        // 2. Exact central-FD checks on all 3 Op-N[Pre] parameters + input frames at 96 kHz and 48 kHz.
        for &(gamma, zoh, beta, rate) in &[(0.65, 0.48, 2.75, sr), (0.55, 0.62, 3.1, 48_000.0)] {
            let mut gx = vec![0.0_f64; n];
            let (_, g) =
                eval_op_n_pre_and_adjoint(&x, gamma, zoh, beta, rate, Some(&d_out), Some(&mut gx));

            let eps = 1.0e-6_f64;
            let fd_g =
                (obj(&eval_op_n_pre_and_adjoint(&x, gamma + eps, zoh, beta, rate, None, None).0)
                    - obj(&eval_op_n_pre_and_adjoint(
                        &x,
                        gamma - eps,
                        zoh,
                        beta,
                        rate,
                        None,
                        None,
                    )
                    .0))
                    / (2.0 * eps);
            let fd_z =
                (obj(&eval_op_n_pre_and_adjoint(&x, gamma, zoh + eps, beta, rate, None, None).0)
                    - obj(&eval_op_n_pre_and_adjoint(
                        &x,
                        gamma,
                        zoh - eps,
                        beta,
                        rate,
                        None,
                        None,
                    )
                    .0))
                    / (2.0 * eps);
            let fd_b =
                (obj(&eval_op_n_pre_and_adjoint(&x, gamma, zoh, beta + eps, rate, None, None).0)
                    - obj(&eval_op_n_pre_and_adjoint(
                        &x,
                        gamma,
                        zoh,
                        beta - eps,
                        rate,
                        None,
                        None,
                    )
                    .0))
                    / (2.0 * eps);

            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[43] += eps;
            xm[43] -= eps;
            let fd_x43 =
                (obj(&eval_op_n_pre_and_adjoint(&xp, gamma, zoh, beta, rate, None, None).0)
                    - obj(&eval_op_n_pre_and_adjoint(&xm, gamma, zoh, beta, rate, None, None).0))
                    / (2.0 * eps);

            for (name, ana, fd) in [
                ("gamma_bc", g.d_gamma_bc, fd_g),
                ("zoh_hold_depth", g.d_zoh_hold_depth, fd_z),
                ("beta_q", g.d_beta_q, fd_b),
                ("x43", gx[43], fd_x43),
            ] {
                let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                assert!(
                    (ana - fd).abs() < 1.0e-4 * scale,
                    "rate={rate} {name}: ana={ana} fd={fd}"
                );
            }
            assert!(g.d_zoh_hold_depth.abs() > 1.0e-4);
        }
        let _ = OP_N_PRE_MAX_BETA_Q;
    }
}
