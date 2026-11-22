//! Exact 4-lane stereo `Op-T` recurrent transport forward + reverse-time BPTT adjoint
//! matching `dnam_kernels` Catmull-Rom 4-tap interpolation, Hermite windowing, Schroeder
//! allpass diffusion, `sigma_sat` cubic feedback saturation, pairwise Givens + Householder `H_4`,
//! `5 Hz` loop & wet DC blockers, and continuous `freeze_mix` (`ref[dsp.op-t.diffuser.recurrence]`).

use dnam_kernels::{TANK_MAX_G_LIN_F32, TANK_MAX_V_CEIL_F32};

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;

use super::super::model::TransportQuadF64;

/// Reverse-time adjoint gradients accumulated by [`eval_op_t_quad_stereo_and_adjoint`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpTQuadAdjointGrads {
    /// Gradients wrt `delay_samples[0..4]`.
    pub d_delay_samples: [f64; 4],
    /// Gradient wrt `g_loop`.
    pub d_g_loop: f64,
    /// Gradient wrt `damp_pole`.
    pub d_damp_pole: f64,
    /// Gradients wrt `input_gains[0..4]`.
    pub d_input_gains: [f64; 4],
    /// Gradients wrt `wet_tap_l[0..4]`.
    pub d_wet_tap_l: [f64; 4],
    /// Gradients wrt `wet_tap_r[0..4]`.
    pub d_wet_tap_r: [f64; 4],
}

#[inline]
pub(super) fn blend_catmull_rom_with_derivs(
    s_m1: f64,
    s_0: f64,
    s_1: f64,
    s_2: f64,
    mu: f64,
    half_alpha: f64,
) -> (f64, f64, [f64; 4]) {
    let omt = 1.0 - mu;
    let y_lin = omt * s_0 + mu * s_1;
    let y_curv = omt * (2.0 * s_0 - s_m1 - s_1) + mu * (2.0 * s_1 - s_0 - s_2);
    let ac = half_alpha * mu * omt;
    let val = y_lin + ac * y_curv;
    let d_ylin = s_1 - s_0;
    let d_ycurv = 3.0 * (s_1 - s_0) + s_m1 - s_2;
    let d_mu = d_ylin + half_alpha * ((1.0 - 2.0 * mu) * y_curv + mu * omt * d_ycurv);
    let w_taps = [
        -ac * omt,
        omt + ac * (2.0 * omt - mu),
        mu + ac * (2.0 * mu - omt),
        -ac * mu,
    ];
    (val, d_mu, w_taps)
}

#[inline]
pub(super) fn half_alpha_cubic_and_deriv(g_eff: f64, mu_freeze: f64) -> (f64, f64) {
    let g_clamped = g_eff.max(0.5);
    let raw = (0.9992 / g_clamped - 1.0) * 4.0;
    let omt = 1.0 - mu_freeze;
    if raw <= 0.0 {
        (0.0, 0.0)
    } else if raw >= 1.0 {
        (0.5 * omt, 0.0)
    } else {
        let d_half = if g_eff > 0.5 {
            0.5 * omt * (-4.0 * 0.9992 / (g_eff * g_eff))
        } else {
            0.0
        };
        (0.5 * omt * raw, d_half)
    }
}

#[inline]
pub(super) fn cubic_soft_sat_and_deriv(
    x: f64,
    gain_scale: f64,
    inv_sigma_23: f64,
) -> (f64, f64, f64) {
    let v_ceil = f64::from(TANK_MAX_V_CEIL_F32);
    let scaled = x * inv_sigma_23;
    let u = scaled.clamp(-1.0, 1.0);
    let poly = (1.5 - 0.5 * u * u) * u;
    let raw = gain_scale * poly;
    if raw.abs() >= v_ceil {
        return (raw.clamp(-v_ceil, v_ceil), 0.0, 0.0);
    }
    let d_x = if scaled.abs() < 1.0 {
        gain_scale * inv_sigma_23 * 1.5 * (1.0 - u * u)
    } else {
        0.0
    };
    (raw, d_x, poly)
}

#[inline]
fn eval_hermite_window_weight(t: usize, k: usize, tq: &TransportQuadF64, sr_hz: f64) -> f64 {
    let base = tq.window_depth.clamp(0.0, 1.0);
    let omt_freeze = 1.0 - tq.freeze_mix.clamp(0.0, 1.0);
    let mu_win = if tq.ramp_rate_hz.abs() > 1.0e-6 {
        let tapered_span = tq.ramp_spans_samples[k].abs() * omt_freeze;
        base.max(((tapered_span - 0.25) * 4.0).clamp(0.0, 1.0))
    } else {
        base
    };
    if mu_win <= 0.0 {
        return 1.0;
    }
    let sr = sr_hz.max(8_000.0);
    let phase = ((t as f64) * tq.ramp_rate_hz / sr).rem_euclid(1.0);
    let fold = if phase <= 0.5 {
        2.0 * phase
    } else {
        2.0 * (1.0 - phase)
    };
    let h = fold * fold * (3.0 - 2.0 * fold);
    (1.0 - mu_win) + mu_win * h
}

struct QuadFrameHist {
    w_ring: Vec<[f64; 4]>,
    ap_write: Vec<[f64; 4]>,
    r_raw: Vec<[f64; 4]>,
    r_win: Vec<[f64; 4]>,
    v_read: Vec<[f64; 4]>,
    v_fb: Vec<[f64; 4]>,
    v_damp: Vec<[f64; 4]>,
    y_dc: Vec<[f64; 4]>,
    d_tau_vread: Vec<[f64; 4]>,
    d_loop_dv: Vec<[f64; 4]>,
    d_loop_dg: Vec<[f64; 4]>,
    d_half_v: Vec<[f64; 4]>,
    win: Vec<[f64; 4]>,
}

/// Optional reverse-time adjoint buffers supplied to [`eval_op_t_quad_stereo_and_adjoint`].
#[derive(Debug)]
pub struct OpTQuadAdjointSlices<'a> {
    /// Upstream Left gradient wrt `Op-T` wet output.
    pub d_out_l: &'a [f64],
    /// Upstream Right gradient wrt `Op-T` wet output.
    pub d_out_r: &'a [f64],
    /// Optional accumulator for gradients wrt Left tank input `x_l`.
    pub grad_x_l: Option<&'a mut [f64]>,
    /// Optional accumulator for gradients wrt Right tank input `x_r`.
    pub grad_x_r: Option<&'a mut [f64]>,
}

/// Evaluates the full 4-lane stereo `Op-T` recurrent transport block (`[Catmull-Rom -> Window ->
/// Schroeder -> sigma_sat -> Givens -> H_4 -> inj -> Damp -> 5Hz Loop DC -> freeze_mix -> wet
/// Givens + 5Hz DC]`) and computes exact reverse-time BPTT adjoints.
pub fn eval_op_t_quad_stereo_and_adjoint(
    x_l: &[f64],
    x_r: &[f64],
    tq: &TransportQuadF64,
    sample_rate_hz: f64,
    adj: Option<OpTQuadAdjointSlices<'_>>,
) -> (Vec<f64>, Vec<f64>, OpTQuadAdjointGrads) {
    let n = x_l.len().min(x_r.len());
    if n < 8 {
        return (vec![0.0; n], vec![0.0; n], OpTQuadAdjointGrads::default());
    }
    let max_tau = (n.saturating_sub(3) as f64).max(MIN_DELAY_CROSSBAR_SAMP);
    let mut i_int = [0_usize; 4];
    let mut mu_k = [0.0_f64; 4];
    for k in 0..4 {
        let tau = tq.delay_samples[k].clamp(MIN_DELAY_CROSSBAR_SAMP, max_tau);
        let d_fl = tau.floor();
        i_int[k] = d_fl as usize;
        mu_k[k] = (tau - d_fl).clamp(0.0, 1.0);
    }

    let g_eff = tq.g_loop.clamp(0.0, f64::from(TANK_MAX_G_LIN_F32));
    let mu_freeze = tq.freeze_mix.clamp(0.0, 1.0);
    let sigma_sat = tq.sigma_knee.clamp(1.0e-3, 0.984_375);
    let inv_sigma_23 = 2.0 / (3.0 * sigma_sat);
    let dp = tq.damp_pole.clamp(0.0, 0.9995);
    let one_m_dp = 1.0 - dp;
    let r_dc = tq.r_dc.clamp(0.0, 0.999_995);
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let (sin_psi, cos_psi) = tq.fb_psi.sin_cos();
    let (half_alpha, d_half_dg) = half_alpha_cubic_and_deriv(g_eff, mu_freeze);
    let sch_active = tq.schroeder_enable && mu_freeze < 1.0 - 1.0e-6;
    let ap_delays = tq.allpass_delays.map(usize::from);
    let ap_g = tq
        .allpass_g
        .map(|g| g.clamp(-0.75, 0.75) * (1.0 - mu_freeze));

    let mut hist = QuadFrameHist {
        w_ring: vec![[0.0; 4]; n],
        ap_write: vec![[0.0; 4]; n],
        r_raw: vec![[0.0; 4]; n],
        r_win: vec![[0.0; 4]; n],
        v_read: vec![[0.0; 4]; n],
        v_fb: vec![[0.0; 4]; n],
        v_damp: vec![[0.0; 4]; n],
        y_dc: vec![[0.0; 4]; n],
        d_tau_vread: vec![[0.0; 4]; n],
        d_loop_dv: vec![[0.0; 4]; n],
        d_loop_dg: vec![[0.0; 4]; n],
        d_half_v: vec![[0.0; 4]; n],
        win: vec![[1.0; 4]; n],
    };
    let mut out_l = vec![0.0_f64; n];
    let mut out_r = vec![0.0_f64; n];

    let mut wet_raw_prev = [0.0_f64; 2];
    let mut wet_out_prev = [0.0_f64; 2];
    for t in 0..n {
        let mut r_raw_t = [0.0_f64; 4];
        let mut r_win_t = [0.0_f64; 4];
        let mut v_read_t = [0.0_f64; 4];
        let mut v_loop_t = [0.0_f64; 4];
        for k in 0..4 {
            let dk = i_int[k];
            let s_m1 = if t + 1 >= dk {
                hist.w_ring[t + 1 - dk][k]
            } else {
                0.0
            };
            let s_0 = if t >= dk { hist.w_ring[t - dk][k] } else { 0.0 };
            let s_1 = if t > dk {
                hist.w_ring[t - dk - 1][k]
            } else {
                0.0
            };
            let s_2 = if t >= dk + 2 {
                hist.w_ring[t - dk - 2][k]
            } else {
                0.0
            };
            let (vr, d_mu, _) =
                blend_catmull_rom_with_derivs(s_m1, s_0, s_1, s_2, mu_k[k], half_alpha);
            let omt = 1.0 - mu_k[k];
            let y_curv = omt * (2.0 * s_0 - s_m1 - s_1) + mu_k[k] * (2.0 * s_1 - s_0 - s_2);
            hist.d_half_v[t][k] = mu_k[k] * omt * y_curv;
            hist.d_tau_vread[t][k] = d_mu;
            r_raw_t[k] = vr;

            let w_k = eval_hermite_window_weight(t, k, tq, sample_rate_hz);
            hist.win[t][k] = w_k;
            r_win_t[k] = w_k * vr;

            if sch_active && ap_delays[k] > 0 {
                let d_ap = ap_delays[k];
                let ap_del = if t >= d_ap {
                    hist.ap_write[t - d_ap][k]
                } else {
                    0.0
                };
                let ap_w = r_win_t[k] - ap_g[k] * ap_del;
                hist.ap_write[t][k] = ap_w;
                v_read_t[k] = ap_del + ap_g[k] * ap_w;
            } else {
                v_read_t[k] = r_win_t[k];
            }
            let (vl, dv, dg) =
                cubic_soft_sat_and_deriv(v_read_t[k], g_eff * sigma_sat, inv_sigma_23);
            v_loop_t[k] = vl;
            hist.d_loop_dv[t][k] = dv;
            hist.d_loop_dg[t][k] = dg * sigma_sat;
        }
        hist.r_raw[t] = r_raw_t;
        hist.r_win[t] = r_win_t;
        hist.v_read[t] = v_read_t;

        let r_giv = [
            cos_psi * v_loop_t[0] - sin_psi * v_loop_t[1],
            sin_psi * v_loop_t[0] + cos_psi * v_loop_t[1],
            cos_psi * v_loop_t[2] - sin_psi * v_loop_t[3],
            sin_psi * v_loop_t[2] + cos_psi * v_loop_t[3],
        ];
        let half_sum = 0.5 * (r_giv[0] + r_giv[1] + r_giv[2] + r_giv[3]);
        let xq = [x_l[t], x_r[t], x_l[t], x_r[t]];
        let mut wr_t = [0.0_f64; 4];
        for k in 0..4 {
            let vfb = (r_giv[k] - half_sum) + tq.input_gains[k] * xq[k];
            hist.v_fb[t][k] = vfb;
            let dp_prev = if t > 0 { hist.v_damp[t - 1][k] } else { 0.0 };
            let vd = one_m_dp * vfb + dp * dp_prev;
            hist.v_damp[t][k] = vd;
            let vd_prev = dp_prev;
            let y_prev = if t > 0 { hist.y_dc[t - 1][k] } else { 0.0 };
            let ydc = alpha_dc * (vd - vd_prev) + r_dc * y_prev;
            hist.y_dc[t][k] = ydc;
            wr_t[k] = (1.0 - mu_freeze) * ydc + mu_freeze * r_win_t[k];
        }
        hist.w_ring[t] = wr_t;

        let (mut wl_sum, mut wr_sum) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            wl_sum += tq.wet_tap_l[k] * v_read_t[k];
            wr_sum += tq.wet_tap_r[k] * v_read_t[k];
        }
        let wl_raw = cos_psi * wl_sum - sin_psi * wr_sum;
        let wr_raw = sin_psi * wl_sum + cos_psi * wr_sum;
        let yl = alpha_dc * (wl_raw - wet_raw_prev[0]) + r_dc * wet_out_prev[0];
        let yr = alpha_dc * (wr_raw - wet_raw_prev[1]) + r_dc * wet_out_prev[1];
        wet_raw_prev = [wl_raw, wr_raw];
        wet_out_prev = [yl, yr];
        out_l[t] = yl;
        out_r[t] = yr;
    }

    let mut grads = OpTQuadAdjointGrads::default();
    if let Some(mut adj_slices) = adj {
        let dl = adj_slices.d_out_l;
        let dr = adj_slices.d_out_r;
        let mut adj_w = vec![[0.0_f64; 4]; n];
        let mut adj_ap_w = vec![[0.0_f64; 4]; n];
        let mut adj_ydc_next = [0.0_f64; 4];
        let mut adj_vd_next_dc = [0.0_f64; 4];
        let mut adj_vd_next_dp = [0.0_f64; 4];
        let mut adj_out_wet_next = [0.0_f64; 2];
        let mut adj_raw_wet_next = [0.0_f64; 2];

        for t in (0..n).rev() {
            let d_yl = dl[t] + r_dc * adj_out_wet_next[0];
            let d_yr = dr[t] + r_dc * adj_out_wet_next[1];
            adj_out_wet_next = [d_yl, d_yr];

            let d_wl_raw = alpha_dc * d_yl + adj_raw_wet_next[0];
            let d_wr_raw = alpha_dc * d_yr + adj_raw_wet_next[1];
            adj_raw_wet_next = [-alpha_dc * d_yl, -alpha_dc * d_yr];

            let d_wl_sum = cos_psi * d_wl_raw + sin_psi * d_wr_raw;
            let d_wr_sum = -sin_psi * d_wl_raw + cos_psi * d_wr_raw;

            let mut adj_vread = [0.0_f64; 4];
            let mut adj_ydc = [0.0_f64; 4];
            let mut adj_vd = [0.0_f64; 4];
            let mut adj_vfb = [0.0_f64; 4];
            let xq = [x_l[t], x_r[t], x_l[t], x_r[t]];

            for k in 0..4 {
                grads.d_wet_tap_l[k] += d_wl_sum * hist.v_read[t][k];
                grads.d_wet_tap_r[k] += d_wr_sum * hist.v_read[t][k];
                adj_vread[k] += d_wl_sum * tq.wet_tap_l[k] + d_wr_sum * tq.wet_tap_r[k];

                let aw = adj_w[t][k];
                adj_ydc[k] = (1.0 - mu_freeze) * aw + r_dc * adj_ydc_next[k];
                adj_ydc_next[k] = adj_ydc[k];

                adj_vd[k] = alpha_dc * adj_ydc[k] + adj_vd_next_dc[k] + dp * adj_vd_next_dp[k];
                adj_vd_next_dc[k] = -alpha_dc * adj_ydc[k];
                adj_vd_next_dp[k] = adj_vd[k];

                let dp_prev = if t > 0 { hist.v_damp[t - 1][k] } else { 0.0 };
                grads.d_damp_pole += adj_vd[k] * (dp_prev - hist.v_fb[t][k]);
                adj_vfb[k] = one_m_dp * adj_vd[k];
                grads.d_input_gains[k] += adj_vfb[k] * xq[k];
            }
            if let Some(ref mut gxl) = adj_slices.grad_x_l {
                gxl[t] += adj_vfb[0] * tq.input_gains[0] + adj_vfb[2] * tq.input_gains[2];
            }
            if let Some(ref mut gxr) = adj_slices.grad_x_r {
                gxr[t] += adj_vfb[1] * tq.input_gains[1] + adj_vfb[3] * tq.input_gains[3];
            }

            // Householder H_4(1)^T == H_4(1), Givens R(psi)^T == R(-psi).
            let half_adj = 0.5 * (adj_vfb[0] + adj_vfb[1] + adj_vfb[2] + adj_vfb[3]);
            let a_rg = [
                adj_vfb[0] - half_adj,
                adj_vfb[1] - half_adj,
                adj_vfb[2] - half_adj,
                adj_vfb[3] - half_adj,
            ];
            let a_vloop = [
                cos_psi * a_rg[0] + sin_psi * a_rg[1],
                -sin_psi * a_rg[0] + cos_psi * a_rg[1],
                cos_psi * a_rg[2] + sin_psi * a_rg[3],
                -sin_psi * a_rg[2] + cos_psi * a_rg[3],
            ];

            for k in 0..4 {
                grads.d_g_loop += a_vloop[k] * hist.d_loop_dg[t][k];
                let d_vr = adj_vread[k] + a_vloop[k] * hist.d_loop_dv[t][k];

                let mut d_rwin = mu_freeze * adj_w[t][k];
                if sch_active && ap_delays[k] > 0 {
                    let d_ap_w = ap_g[k] * d_vr + adj_ap_w[t][k];
                    d_rwin += d_ap_w;
                    let d_ap = ap_delays[k];
                    if t >= d_ap {
                        adj_ap_w[t - d_ap][k] += d_vr - ap_g[k] * d_ap_w;
                    }
                } else {
                    d_rwin += d_vr;
                }

                let d_rraw = d_rwin * hist.win[t][k];
                grads.d_delay_samples[k] += d_rraw * hist.d_tau_vread[t][k];
                grads.d_g_loop += d_rraw * hist.d_half_v[t][k] * d_half_dg;

                let dk = i_int[k];
                let mu = mu_k[k];
                let omt = 1.0 - mu;
                let ac = half_alpha * mu * omt;
                let w_taps = [
                    -ac * omt,
                    omt + ac * (2.0 * omt - mu),
                    mu + ac * (2.0 * mu - omt),
                    -ac * mu,
                ];
                if t + 1 >= dk {
                    adj_w[t + 1 - dk][k] += d_rraw * w_taps[0];
                }
                if t >= dk {
                    adj_w[t - dk][k] += d_rraw * w_taps[1];
                }
                if t > dk {
                    adj_w[t - dk - 1][k] += d_rraw * w_taps[2];
                }
                if t >= dk + 2 {
                    adj_w[t - dk - 2][k] += d_rraw * w_taps[3];
                }
            }
        }
    }

    (out_l, out_r, grads)
}
