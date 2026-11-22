//! Differentiable forward and exact reverse-time adjoint evaluators for `Op-N[Pre]` Fourier
//! surrogate, stereo complex conjugate Kautz modal banks, and `Op-T` recurrent transport
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::{LN_2, TAU};

use dnam_kernels::{OP_N_PRE_MAX_BETA_Q, OP_N_PRE_QUANT_CLAMP_F32, TANK_MAX_G_LIN_F32};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;

use super::super::model::KautzPairF64;
use super::op_t_quad::{
    blend_catmull_rom_with_derivs, cubic_soft_sat_and_deriv, half_alpha_cubic_and_deriv,
};

/// Hard safety limit enforcing smooth Fourier `Op-N[Pre]` input clipping.
const OP_N_CLIP_F64: f64 = OP_N_PRE_QUANT_CLAMP_F32 as f64;
/// Minimum active Kautz pole radius enforced by `dnam-vm` runtime hydration.
const KAUTZ_R_MIN: f64 = 0.25;
/// Maximum stable Kautz pole radius enforced by `dnam-vm` runtime hydration.
const KAUTZ_R_MAX: f64 = 0.9995;

const _: () = assert!(OP_N_CLIP_F64 > 0.0);
const _: () = assert!(KAUTZ_R_MIN > 0.0 && KAUTZ_R_MIN < KAUTZ_R_MAX && KAUTZ_R_MAX < 1.0);

/// Differentiable $C^\infty$ Fourier smooth quantizer surrogate consistent with
/// `dnam_kernels::quantize_smooth_fourier_f32`:
/// `y[t] = (1 - gamma) * x[t] + gamma * (x_c - (step / 2pi) * sin(2pi * d_scale * x_c))`.
pub fn eval_op_n_pre_and_adjoint(
    x: &[f64],
    gamma_bc: f64,
    beta_q: f64,
    d_out_opt: Option<&[f64]>,
    grad_x_opt: Option<&mut [f64]>,
) -> (Vec<f64>, f64, f64) {
    let gamma = gamma_bc.clamp(0.0, 1.0);
    let beta = beta_q.clamp(0.0, f64::from(OP_N_PRE_MAX_BETA_Q));
    let d_scale = beta.exp2();
    let step = 1.0 / d_scale;
    let inv_tau = 1.0 / TAU;

    let mut y = vec![0.0_f64; x.len()];
    let mut g_gamma = 0.0_f64;
    let mut g_beta = 0.0_f64;

    for (t, &xt) in x.iter().enumerate() {
        let xc = xt.clamp(-OP_N_CLIP_F64, OP_N_CLIP_F64);
        let phase = TAU * d_scale * xc;
        let (s, c) = phase.sin_cos();
        let q_term = xc - (step * inv_tau) * s;
        let yt = (1.0 - gamma) * xt + gamma * q_term;
        y[t] = yt;

        if let Some(d_out) = d_out_opt {
            let dy = d_out[t];
            g_gamma += dy * (q_term - xt);
            let dq_dbeta = LN_2 * ((step * inv_tau) * s - xc * c);
            g_beta += dy * gamma * dq_dbeta;
        }
    }
    if let (Some(d_out), Some(gx)) = (d_out_opt, grad_x_opt) {
        for t in 0..x.len() {
            let xt = x[t];
            let xc = xt.clamp(-OP_N_CLIP_F64, OP_N_CLIP_F64);
            let dq_dx = if xt.abs() <= OP_N_CLIP_F64 {
                1.0 - (TAU * d_scale * xc).cos()
            } else {
                0.0
            };
            gx[t] += d_out[t] * ((1.0 - gamma) + gamma * dq_dx);
        }
    }
    (y, g_gamma, g_beta)
}

/// Evaluates one stereo channel (`is_right = false -> L`, `true -> R`) of complex conjugate Kautz
/// resonators `kautz[0..count]` using exact runtime readout `2.0 * (c_re * sr - c_im * si)`,
/// returning reverse-time gradients `[d_r, d_theta, d_b_gamma, d_c_re, d_c_im]`.
pub fn eval_kautz_modal_channel_and_adjoint(
    u: &[f64],
    pairs: &[KautzPairF64],
    is_right: bool,
    d_out_opt: Option<&[f64]>,
    grad_u_opt: Option<&mut [f64]>,
) -> (Vec<f64>, Vec<[f64; 5]>) {
    let n = u.len();
    let m = pairs.len();
    let mut y = vec![0.0_f64; n];
    let mut sr_hist = vec![vec![0.0_f64; n]; m];
    let mut si_hist = vec![vec![0.0_f64; n]; m];

    for (k, pair) in pairs.iter().enumerate() {
        let r_raw = pair.r_low + 0.5 * pair.dr;
        if !r_raw.is_finite() || r_raw <= 0.0 || !pair.b_gamma.is_finite() || pair.b_gamma <= 0.0 {
            continue;
        }
        let r = r_raw.clamp(KAUTZ_R_MIN, KAUTZ_R_MAX);
        let (st, ct) = pair.theta.sin_cos();
        let (ar, ai) = (r * ct, r * st);
        let bg = pair.b_gamma;
        let (c_re, c_im) = if is_right {
            (pair.c_re_r, pair.c_im_r)
        } else {
            (pair.c_re_l, pair.c_im_l)
        };
        let (mut sr, mut si) = (0.0_f64, 0.0_f64);
        for t in 0..n {
            let nr = ar * sr - ai * si + bg * u[t];
            let ni = ai * sr + ar * si;
            sr = nr;
            si = ni;
            sr_hist[k][t] = sr;
            si_hist[k][t] = si;
            y[t] += 2.0 * (c_re * sr - c_im * si);
        }
    }

    let mut pair_grads = vec![[0.0_f64; 5]; m];
    if let Some(d_out) = d_out_opt {
        let mut gu_local = vec![0.0_f64; n];
        for (k, pair) in pairs.iter().enumerate() {
            let r_raw = pair.r_low + 0.5 * pair.dr;
            if !r_raw.is_finite()
                || r_raw <= 0.0
                || !pair.b_gamma.is_finite()
                || pair.b_gamma <= 0.0
            {
                continue;
            }
            let r = r_raw.clamp(KAUTZ_R_MIN, KAUTZ_R_MAX);
            let (st, ct) = pair.theta.sin_cos();
            let (ar, ai) = (r * ct, r * st);
            let (c_re, c_im) = if is_right {
                (pair.c_re_r, pair.c_im_r)
            } else {
                (pair.c_re_l, pair.c_im_l)
            };
            let (mut lam_r, mut lam_i) = (0.0_f64, 0.0_f64);
            let (mut d_ar, mut d_ai, mut d_bg, mut d_cre, mut d_cim) = (0.0, 0.0, 0.0, 0.0, 0.0);

            for t in (0..n).rev() {
                let dt2 = 2.0 * d_out[t];
                d_cre += dt2 * sr_hist[k][t];
                d_cim -= dt2 * si_hist[k][t];
                let adj_r = dt2 * c_re + lam_r;
                let adj_i = -dt2 * c_im + lam_i;
                gu_local[t] += pair.b_gamma * adj_r;
                d_bg += adj_r * u[t];
                let prev_r = if t > 0 { sr_hist[k][t - 1] } else { 0.0 };
                let prev_i = if t > 0 { si_hist[k][t - 1] } else { 0.0 };
                d_ar += adj_r * prev_r + adj_i * prev_i;
                d_ai += adj_i * prev_r - adj_r * prev_i;
                lam_r = ar * adj_r + ai * adj_i;
                lam_i = -ai * adj_r + ar * adj_i;
            }
            let d_r = d_ar * ct + d_ai * st;
            let d_theta = -d_ar * r * st + d_ai * r * ct;
            pair_grads[k] = [d_r, d_theta, d_bg, d_cre, d_cim];
        }
        if let Some(gu) = grad_u_opt {
            for (dst, src) in gu.iter_mut().zip(&gu_local) {
                *dst += *src;
            }
        }
    }
    (y, pair_grads)
}

/// Evaluates the Left-channel complex conjugate Kautz modal bank via [`eval_kautz_modal_channel_and_adjoint`].
#[inline]
pub fn eval_kautz_modal_and_adjoint(
    u: &[f64],
    pairs: &[KautzPairF64],
    d_out_opt: Option<&[f64]>,
    grad_u_opt: Option<&mut [f64]>,
) -> (Vec<f64>, Vec<[f64; 5]>) {
    eval_kautz_modal_channel_and_adjoint(u, pairs, false, d_out_opt, grad_u_opt)
}

/// Evaluates causal single-lane fractional `Op-T` recurrent delay/damping/feedback transport
/// (`tau >= 5.0`) with passive-blended 4-tap Catmull-Rom interpolation, `sigma_sat` cubic
/// loop soft-knee, and `5 Hz` loop + wet DC blockers, returning `[d_tau, d_g_loop, d_damp, d_wet]`.
pub fn eval_op_t_transport_and_adjoint(
    x: &[f64],
    tau_samples: f64,
    g_loop: f64,
    damp_pole: f64,
    wet_tap: f64,
    d_out_opt: Option<&[f64]>,
    grad_x_opt: Option<&mut [f64]>,
) -> (Vec<f64>, [f64; 4]) {
    let n = x.len();
    if n < 8 {
        return (vec![0.0; n], [0.0; 4]);
    }
    let tau = tau_samples.clamp(
        MIN_DELAY_CROSSBAR_SAMP,
        (n.saturating_sub(3) as f64).max(MIN_DELAY_CROSSBAR_SAMP),
    );
    let gl = g_loop.clamp(0.0, f64::from(TANK_MAX_G_LIN_F32));
    let dp = damp_pole.clamp(0.0, 0.9995);
    let one_m_dp = 1.0 - dp;
    let sigma_sat = 0.984_375_f64;
    let inv_sigma_23 = 2.0 / (3.0 * sigma_sat);
    let r_dc = f64::from(TANK_DC_BLOCKER_48K_F32);
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let (half_alpha, d_half_dg) = half_alpha_cubic_and_deriv(gl, 0.0);

    let dk = tau.floor() as usize;
    let mu = (tau - (dk as f64)).clamp(0.0, 1.0);
    let omt = 1.0 - mu;

    let mut w_ring = vec![0.0_f64; n];
    let mut v_read = vec![0.0_f64; n];
    let mut v_fb = vec![0.0_f64; n];
    let mut v_damp = vec![0.0_f64; n];
    let mut y_dc = vec![0.0_f64; n];
    let mut d_tau_vr = vec![0.0_f64; n];
    let mut d_half_v = vec![0.0_f64; n];
    let mut d_loop_dv = vec![0.0_f64; n];
    let mut d_loop_dg = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];

    let mut wet_raw_prev = 0.0_f64;
    let mut wet_out_prev = 0.0_f64;
    for t in 0..n {
        let s_m1 = if t + 1 >= dk { w_ring[t + 1 - dk] } else { 0.0 };
        let s_0 = if t >= dk { w_ring[t - dk] } else { 0.0 };
        let s_1 = if t > dk { w_ring[t - dk - 1] } else { 0.0 };
        let s_2 = if t >= dk + 2 { w_ring[t - dk - 2] } else { 0.0 };
        let (vr, d_mu, _) = blend_catmull_rom_with_derivs(s_m1, s_0, s_1, s_2, mu, half_alpha);
        let y_curv = omt * (2.0 * s_0 - s_m1 - s_1) + mu * (2.0 * s_1 - s_0 - s_2);
        v_read[t] = vr;
        d_tau_vr[t] = d_mu;
        d_half_v[t] = mu * omt * y_curv;

        let (v_loop, dv, dg) = cubic_soft_sat_and_deriv(vr, gl * sigma_sat, inv_sigma_23);
        d_loop_dv[t] = dv;
        d_loop_dg[t] = dg * sigma_sat;
        let vfb = x[t] + v_loop;
        v_fb[t] = vfb;

        let dp_prev = if t > 0 { v_damp[t - 1] } else { 0.0 };
        let vd = one_m_dp * vfb + dp * dp_prev;
        v_damp[t] = vd;

        let y_prev = if t > 0 { y_dc[t - 1] } else { 0.0 };
        let ydc = alpha_dc * (vd - dp_prev) + r_dc * y_prev;
        y_dc[t] = ydc;
        w_ring[t] = ydc;

        let w_raw = wet_tap * vr;
        let w_out = alpha_dc * (w_raw - wet_raw_prev) + r_dc * wet_out_prev;
        wet_raw_prev = w_raw;
        wet_out_prev = w_out;
        y[t] = w_out;
    }

    let mut grads = [0.0_f64; 4]; // [d_tau, d_g_loop, d_damp_pole, d_wet_tap]
    if let Some(d_out) = d_out_opt {
        let mut adj_w = vec![0.0_f64; n];
        let mut gx_local = vec![0.0_f64; n];
        let mut adj_out_next = 0.0_f64;
        let mut adj_raw_next = 0.0_f64;
        let mut adj_ydc_next = 0.0_f64;
        let mut adj_vd_next_dc = 0.0_f64;
        let mut adj_vd_next_dp = 0.0_f64;
        let ac = half_alpha * mu * omt;
        let w_taps = [
            -ac * omt,
            omt + ac * (2.0 * omt - mu),
            mu + ac * (2.0 * mu - omt),
            -ac * mu,
        ];

        for t in (0..n).rev() {
            let d_yo = d_out[t] + r_dc * adj_out_next;
            adj_out_next = d_yo;
            let d_wraw = alpha_dc * d_yo + adj_raw_next;
            adj_raw_next = -alpha_dc * d_yo;
            grads[3] += d_wraw * v_read[t];

            let adj_ydc = adj_w[t] + r_dc * adj_ydc_next;
            adj_ydc_next = adj_ydc;
            let adj_vd = alpha_dc * adj_ydc + adj_vd_next_dc + dp * adj_vd_next_dp;
            adj_vd_next_dc = -alpha_dc * adj_ydc;
            adj_vd_next_dp = adj_vd;

            let dp_prev = if t > 0 { v_damp[t - 1] } else { 0.0 };
            grads[2] += adj_vd * (dp_prev - v_fb[t]);
            let adj_vfb = one_m_dp * adj_vd;
            gx_local[t] += adj_vfb;

            grads[1] += adj_vfb * d_loop_dg[t];
            let d_vr = d_wraw * wet_tap + adj_vfb * d_loop_dv[t];
            grads[0] += d_vr * d_tau_vr[t];
            grads[1] += d_vr * d_half_v[t] * d_half_dg;

            if t + 1 >= dk {
                adj_w[t + 1 - dk] += d_vr * w_taps[0];
            }
            if t >= dk {
                adj_w[t - dk] += d_vr * w_taps[1];
            }
            if t > dk {
                adj_w[t - dk - 1] += d_vr * w_taps[2];
            }
            if t >= dk + 2 {
                adj_w[t - dk - 2] += d_vr * w_taps[3];
            }
        }
        if let Some(gx) = grad_x_opt {
            for (dst, src) in gx.iter_mut().zip(&gx_local) {
                *dst += *src;
            }
        }
    }
    (y, grads)
}

#[cfg(test)]
mod tests {
    use super::super::super::model::TransportQuadF64;
    use super::super::op_t_quad::{OpTQuadAdjointSlices, eval_op_t_quad_stereo_and_adjoint};
    use super::*;

    fn assert_fd_close(name: &str, ana: f64, fd: f64) {
        let scale = ana.abs().max(fd.abs()).max(1.0e-3);
        assert!(
            (ana - fd).abs() <= 5.0e-5 * scale,
            "{name} adjoint mismatch: ana={ana}, fd={fd}"
        );
    }

    #[test]
    fn test_op_n_pre_kautz_stereo_and_op_t_quad_adjoints_match_central_fd() {
        let x: Vec<f64> = (0..128).map(|t| 0.55 * ((t as f64) * 0.13).sin()).collect();
        let x_r: Vec<f64> = (0..128)
            .map(|t| -0.40 * ((t as f64) * 0.19).cos())
            .collect();
        let d_out: Vec<f64> = (0..128).map(|t| ((t as f64) * 0.29).cos()).collect();
        let d_out_r: Vec<f64> = (0..128).map(|t| ((t as f64) * 0.17).sin()).collect();
        let eps = 1.0e-6_f64;
        let obj = |yv: &[f64]| yv.iter().zip(&d_out).map(|(a, b)| a * b).sum::<f64>();
        let obj_lr = |yl: &[f64], yr: &[f64]| -> f64 {
            yl.iter().zip(&d_out).map(|(a, b)| a * b).sum::<f64>()
                + yr.iter().zip(&d_out_r).map(|(a, b)| a * b).sum::<f64>()
        };

        // 1. Op-N[Pre] FD parity.
        let mut gx = vec![0.0_f64; 128];
        let (_, gg, gb) = eval_op_n_pre_and_adjoint(&x, 0.65, 2.5, Some(&d_out), Some(&mut gx));
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[37] += eps;
        xm[37] -= eps;
        assert_fd_close(
            "op_n_x",
            gx[37],
            (obj(&eval_op_n_pre_and_adjoint(&xp, 0.65, 2.5, None, None).0)
                - obj(&eval_op_n_pre_and_adjoint(&xm, 0.65, 2.5, None, None).0))
                / (2.0 * eps),
        );
        assert_fd_close(
            "op_n_gamma",
            gg,
            (obj(&eval_op_n_pre_and_adjoint(&x, 0.65 + eps, 2.5, None, None).0)
                - obj(&eval_op_n_pre_and_adjoint(&x, 0.65 - eps, 2.5, None, None).0))
                / (2.0 * eps),
        );
        assert_fd_close(
            "op_n_beta",
            gb,
            (obj(&eval_op_n_pre_and_adjoint(&x, 0.65, 2.5 + eps, None, None).0)
                - obj(&eval_op_n_pre_and_adjoint(&x, 0.65, 2.5 - eps, None, None).0))
                / (2.0 * eps),
        );

        // 2. Kautz Right-channel (c_re_r, c_im_r, r, theta, b_gamma) FD parity.
        let pairs = [KautzPairF64 {
            r_low: 0.88,
            dr: 0.04,
            theta: 0.37,
            b_gamma: 0.75,
            c_re_l: 0.5,
            c_im_l: -0.3,
            c_re_r: -0.45,
            c_im_r: 0.25,
        }];
        let mut gu = vec![0.0_f64; 128];
        let (_, kr_grads) =
            eval_kautz_modal_channel_and_adjoint(&x_r, &pairs, true, Some(&d_out_r), Some(&mut gu));
        let obj_r = |yv: &[f64]| yv.iter().zip(&d_out_r).map(|(a, b)| a * b).sum::<f64>();
        let mut p_cr = pairs;
        p_cr[0].c_re_r += eps;
        let mut p_cm = pairs;
        p_cm[0].c_re_r -= eps;
        assert_fd_close(
            "kautz_cre_r",
            kr_grads[0][3],
            (obj_r(&eval_kautz_modal_channel_and_adjoint(&x_r, &p_cr, true, None, None).0)
                - obj_r(&eval_kautz_modal_channel_and_adjoint(&x_r, &p_cm, true, None, None).0))
                / (2.0 * eps),
        );
        let mut p_ci = pairs;
        p_ci[0].c_im_r += eps;
        let mut p_cd = pairs;
        p_cd[0].c_im_r -= eps;
        assert_fd_close(
            "kautz_cim_r",
            kr_grads[0][4],
            (obj_r(&eval_kautz_modal_channel_and_adjoint(&x_r, &p_ci, true, None, None).0)
                - obj_r(&eval_kautz_modal_channel_and_adjoint(&x_r, &p_cd, true, None, None).0))
                / (2.0 * eps),
        );

        // 3. Single-lane Op-T (Catmull-Rom + sigma_sat + 5Hz DC) FD parity.
        let (_, t1_g) =
            eval_op_t_transport_and_adjoint(&x, 9.35, 0.65, 0.25, 0.8, Some(&d_out), None);
        assert_fd_close(
            "op_t1_tau",
            t1_g[0],
            (obj(&eval_op_t_transport_and_adjoint(&x, 9.35 + eps, 0.65, 0.25, 0.8, None, None).0)
                - obj(&eval_op_t_transport_and_adjoint(
                    &x,
                    9.35 - eps,
                    0.65,
                    0.25,
                    0.8,
                    None,
                    None,
                )
                .0))
                / (2.0 * eps),
        );
        assert_fd_close(
            "op_t1_gloop",
            t1_g[1],
            (obj(&eval_op_t_transport_and_adjoint(&x, 9.35, 0.65 + eps, 0.25, 0.8, None, None).0)
                - obj(&eval_op_t_transport_and_adjoint(
                    &x,
                    9.35,
                    0.65 - eps,
                    0.25,
                    0.8,
                    None,
                    None,
                )
                .0))
                / (2.0 * eps),
        );

        // 4. Full 4-lane Op-T quad with Schroeder + freeze_mix + Givens + sigma_knee FD parity.
        let tq = TransportQuadF64 {
            delay_samples: [7.25, 9.6, 12.4, 15.1],
            ramp_spans_samples: [0.0; 4],
            g_loop: 0.72,
            r_dc: f64::from(TANK_DC_BLOCKER_48K_F32),
            damp_pole: 0.3,
            lfo_rate_hz: 0.0,
            ramp_rate_hz: 0.0,
            lfo_depths: [0.0; 4],
            input_gains: [0.6, 0.5, -0.4, 0.3],
            wet_tap_l: [0.7, 0.2, -0.5, 0.1],
            wet_tap_r: [0.15, -0.65, 0.25, 0.45],
            allpass_g: [0.35, -0.25, 0.4, -0.3],
            allpass_delays: [3, 5, 7, 11],
            fb_psi: 0.4,
            sigma_knee: 0.8,
            freeze_mix: 0.25,
            window_depth: 0.15,
            scat_psi: 0.35,
            ext_f32: false,
        };
        let (mut gxl, mut gxr) = (vec![0.0; 128], vec![0.0; 128]);
        let (_, _, qg) = eval_op_t_quad_stereo_and_adjoint(
            &x,
            &x_r,
            &tq,
            48_000.0,
            Some(OpTQuadAdjointSlices {
                d_out_l: &d_out,
                d_out_r: &d_out_r,
                grad_x_l: Some(&mut gxl),
                grad_x_r: Some(&mut gxr),
            }),
        );
        let eval_tq = |cand: &TransportQuadF64| {
            let (yl, yr, _) = eval_op_t_quad_stereo_and_adjoint(&x, &x_r, cand, 48_000.0, None);
            obj_lr(&yl, &yr)
        };
        let mut p = tq;
        p.delay_samples[0] += eps;
        let mut m = tq;
        m.delay_samples[0] -= eps;
        assert_fd_close(
            "quad_delay0",
            qg.d_delay_samples[0],
            (eval_tq(&p) - eval_tq(&m)) / (2.0 * eps),
        );
        p = tq;
        p.g_loop += eps;
        m = tq;
        m.g_loop -= eps;
        assert_fd_close(
            "quad_gloop",
            qg.d_g_loop,
            (eval_tq(&p) - eval_tq(&m)) / (2.0 * eps),
        );
        p = tq;
        p.damp_pole += eps;
        m = tq;
        m.damp_pole -= eps;
        assert_fd_close(
            "quad_damp",
            qg.d_damp_pole,
            (eval_tq(&p) - eval_tq(&m)) / (2.0 * eps),
        );
        p = tq;
        p.input_gains[1] += eps;
        m = tq;
        m.input_gains[1] -= eps;
        assert_fd_close(
            "quad_inj1",
            qg.d_input_gains[1],
            (eval_tq(&p) - eval_tq(&m)) / (2.0 * eps),
        );
        p = tq;
        p.wet_tap_r[2] += eps;
        m = tq;
        m.wet_tap_r[2] -= eps;
        assert_fd_close(
            "quad_wetr2",
            qg.d_wet_tap_r[2],
            (eval_tq(&p) - eval_tq(&m)) / (2.0 * eps),
        );
        p = tq;
        p.scat_psi += eps;
        m = tq;
        m.scat_psi -= eps;
        assert_fd_close(
            "quad_scat_psi",
            qg.d_scat_psi,
            (eval_tq(&p) - eval_tq(&m)) / (2.0 * eps),
        );
    }
}
