//! Exact 4-lane stereo `Op-T` recurrent transport forward + reverse-time BPTT adjoint
//! matching `dnam_kernels` Catmull-Rom 4-tap interpolation, Hermite windowing, Schroeder
//! allpass diffusion, `sigma_sat` cubic feedback saturation, pairwise Givens + Householder `H_4`,
//! `5 Hz` loop & wet DC blockers, and continuous `freeze_mix` (`ref[dsp.op-t.diffuser.recurrence]`).

use core::f64::consts::PI;

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
    /// Gradient wrt `scat_psi`.
    pub d_scat_psi: f64,
    /// Gradients wrt `input_gains[0..4]`.
    pub d_input_gains: [f64; 4],
    /// Gradients wrt `wet_tap_l[0..4]`.
    pub d_wet_tap_l: [f64; 4],
    /// Gradients wrt `wet_tap_r[0..4]`.
    pub d_wet_tap_r: [f64; 4],
}

#[inline(always)]
fn j_even_f64(x: [f64; 4]) -> [f64; 4] {
    [-x[1], x[0], -x[3], x[2]]
}

#[inline(always)]
fn j_odd_f64(y: [f64; 4]) -> [f64; 4] {
    [y[3], -y[2], y[1], -y[0]]
}

#[inline(always)]
fn apply_so4_even_f64(x: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    let jx = j_even_f64(x);
    [
        c * x[0] + s * jx[0],
        c * x[1] + s * jx[1],
        c * x[2] + s * jx[2],
        c * x[3] + s * jx[3],
    ]
}

#[inline(always)]
fn apply_so4_odd_f64(y: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    let jy = j_odd_f64(y);
    [
        c * y[0] + s * jy[0],
        c * y[1] + s * jy[1],
        c * y[2] + s * jy[2],
        c * y[3] + s * jy[3],
    ]
}

#[inline(always)]
fn apply_so4_even_transpose_f64(g: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    apply_so4_even_f64(g, c, -s)
}

#[inline(always)]
fn apply_so4_odd_transpose_f64(g: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    apply_so4_odd_f64(g, c, -s)
}

#[inline(always)]
fn so4_theta_grad_f64(g_v_read: [f64; 4], v_read: [f64; 4], g_x1: [f64; 4], x1: [f64; 4]) -> f64 {
    let jo = j_odd_f64(v_read);
    let je = j_even_f64(x1);
    (0..4).map(|k| g_v_read[k] * jo[k] + g_x1[k] * je[k]).sum()
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
    x1: Vec<[f64; 4]>,
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
/// SO(4) R_even -> Schroeder -> SO(4) R_odd -> sigma_sat -> Givens -> H_4 -> inj -> Damp -> 5Hz
/// Loop DC -> freeze_mix -> wet Givens + 5Hz DC]`) and computes exact reverse-time BPTT adjoints.
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
    let full_hold = mu_freeze >= 1.0 - 1.0e-6;
    let sigma_sat = tq.sigma_knee.clamp(1.0e-3, 0.984_375);
    let inv_sigma_23 = 2.0 / (3.0 * sigma_sat);
    let dp = tq.damp_pole.clamp(0.0, 0.9995);
    let one_m_dp = 1.0 - dp;
    let r_dc = tq.r_dc.clamp(0.0, 0.999_995);
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let (sin_psi, cos_psi) = tq.fb_psi.sin_cos();
    let scat_open = tq.scat_psi.is_finite() && tq.scat_psi.abs() <= PI;
    let scat_clamped = if tq.scat_psi.is_finite() {
        tq.scat_psi.clamp(-PI, PI)
    } else {
        0.0
    };
    let theta_eff = if full_hold {
        0.0
    } else {
        (1.0 - mu_freeze) * scat_clamped
    };
    let (s_scat, c_scat) = theta_eff.sin_cos();
    let (half_alpha, d_half_dg) = half_alpha_cubic_and_deriv(g_eff, mu_freeze);
    let ap_delays = tq.allpass_delays.map(usize::from);
    let ap_g = tq
        .allpass_g
        .map(|g| g.clamp(-0.75, 0.75) * (1.0 - mu_freeze));

    let mut hist = QuadFrameHist {
        w_ring: vec![[0.0; 4]; n],
        ap_write: vec![[0.0; 4]; n],
        r_raw: vec![[0.0; 4]; n],
        r_win: vec![[0.0; 4]; n],
        x1: vec![[0.0; 4]; n],
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
        let mut y1_t = [0.0_f64; 4];
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
        }
        let x1_t = if full_hold {
            r_win_t
        } else {
            apply_so4_even_f64(r_win_t, c_scat, s_scat)
        };
        for k in 0..4 {
            if !full_hold && ap_delays[k] > 0 {
                let d_ap = ap_delays[k];
                let ap_del = if t >= d_ap {
                    hist.ap_write[t - d_ap][k]
                } else {
                    0.0
                };
                let ap_w = x1_t[k] - ap_g[k] * ap_del;
                hist.ap_write[t][k] = ap_w;
                y1_t[k] = ap_del + ap_g[k] * ap_w;
            } else {
                y1_t[k] = x1_t[k];
            }
        }
        let v_read_t = if full_hold {
            y1_t
        } else {
            apply_so4_odd_f64(y1_t, c_scat, s_scat)
        };
        for k in 0..4 {
            let (vl, dv, dg) =
                cubic_soft_sat_and_deriv(v_read_t[k], g_eff * sigma_sat, inv_sigma_23);
            v_loop_t[k] = vl;
            hist.d_loop_dv[t][k] = dv;
            hist.d_loop_dg[t][k] = dg * sigma_sat;
        }
        hist.r_raw[t] = r_raw_t;
        hist.r_win[t] = r_win_t;
        hist.x1[t] = x1_t;
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
        let mut d_theta_eff = 0.0_f64;

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

            let mut g_v_read_t = [0.0_f64; 4];
            for k in 0..4 {
                grads.d_g_loop += a_vloop[k] * hist.d_loop_dg[t][k];
                g_v_read_t[k] = adj_vread[k] + a_vloop[k] * hist.d_loop_dv[t][k];
            }

            let g_y1_t = if full_hold {
                g_v_read_t
            } else {
                apply_so4_odd_transpose_f64(g_v_read_t, c_scat, s_scat)
            };
            let mut g_x1_t = [0.0_f64; 4];
            for k in 0..4 {
                if !full_hold && ap_delays[k] > 0 {
                    let d_ap_w = ap_g[k] * g_y1_t[k] + adj_ap_w[t][k];
                    g_x1_t[k] = d_ap_w;
                    let d_ap = ap_delays[k];
                    if t >= d_ap {
                        adj_ap_w[t - d_ap][k] += g_y1_t[k] - ap_g[k] * d_ap_w;
                    }
                } else {
                    g_x1_t[k] = g_y1_t[k];
                }
            }
            if !full_hold {
                d_theta_eff += so4_theta_grad_f64(g_v_read_t, hist.v_read[t], g_x1_t, hist.x1[t]);
            }
            let g_r_win_scat = if full_hold {
                g_x1_t
            } else {
                apply_so4_even_transpose_f64(g_x1_t, c_scat, s_scat)
            };

            for k in 0..4 {
                let d_rwin = mu_freeze * adj_w[t][k] + g_r_win_scat[k];
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
        grads.d_scat_psi = if !full_hold && scat_open {
            (1.0 - mu_freeze) * d_theta_eff
        } else {
            0.0
        };
    }

    (out_l, out_r, grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_quad_loss(
        x_l: &[f64],
        x_r: &[f64],
        tq: &TransportQuadF64,
        w_l: &[f64],
        w_r: &[f64],
    ) -> f64 {
        let (yl, yr, _) = eval_op_t_quad_stereo_and_adjoint(x_l, x_r, tq, 48_000.0, None);
        yl.iter()
            .zip(yr.iter())
            .enumerate()
            .map(|(i, (&a, &b))| a * w_l[i] + b * w_r[i])
            .sum()
    }

    #[test]
    fn test_op_t_quad_so4_scat_psi_bptt_and_full_hold_fd_parity() {
        let n = 128_usize;
        let x_l: Vec<f64> = (0..n).map(|i| 0.4 * ((i as f64) * 0.17).sin()).collect();
        let x_r: Vec<f64> = (0..n)
            .map(|i| 0.35 * ((i as f64) * 0.13 + 0.4).cos())
            .collect();
        let w_l: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.11).cos()).collect();
        let w_r: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.09).sin()).collect();

        for ap_delays in [[3, 5, 7, 11], [0, 0, 0, 0]] {
            let tq = TransportQuadF64 {
                delay_samples: [7.25, 9.6, 12.4, 15.1],
                ramp_spans_samples: [0.0; 4],
                g_loop: 0.72,
                r_dc: 0.999,
                damp_pole: 0.3,
                lfo_rate_hz: 0.0,
                ramp_rate_hz: 0.0,
                lfo_depths: [0.0; 4],
                input_gains: [0.6, 0.5, -0.4, 0.3],
                wet_tap_l: [0.7, 0.2, -0.5, 0.1],
                wet_tap_r: [0.15, -0.65, 0.25, 0.45],
                allpass_g: [0.35, -0.25, 0.4, -0.3],
                allpass_delays: ap_delays,
                fb_psi: 0.4,
                scat_psi: 0.55,
                sigma_knee: 0.8,
                freeze_mix: 0.25,
                window_depth: 0.15,
                ext_f32: false,
            };
            let (_, _, grads) = eval_op_t_quad_stereo_and_adjoint(
                &x_l,
                &x_r,
                &tq,
                48_000.0,
                Some(OpTQuadAdjointSlices {
                    d_out_l: &w_l,
                    d_out_r: &w_r,
                    grad_x_l: None,
                    grad_x_r: None,
                }),
            );
            let eps = 1.0e-6_f64;
            let mut p = tq;
            p.scat_psi += eps;
            let mut m = tq;
            m.scat_psi -= eps;
            let fd = (eval_quad_loss(&x_l, &x_r, &p, &w_l, &w_r)
                - eval_quad_loss(&x_l, &x_r, &m, &w_l, &w_r))
                / (2.0 * eps);
            assert!(
                grads.d_scat_psi.abs() > 1.0e-3,
                "expected nonzero d_scat_psi for ap_delays={ap_delays:?}"
            );
            assert!(
                (grads.d_scat_psi - fd).abs() < 1.0e-5,
                "scat_psi FD mismatch ap={ap_delays:?}: adj={} fd={}",
                grads.d_scat_psi,
                fd
            );
        }

        // Full hold (mu_freeze = 1.0) freezes scattering and zeroes d_scat_psi.
        let tq_hold = TransportQuadF64 {
            delay_samples: [7.25, 9.6, 12.4, 15.1],
            g_loop: 0.72,
            r_dc: 0.999,
            damp_pole: 0.3,
            input_gains: [0.6, 0.5, -0.4, 0.3],
            wet_tap_l: [0.7, 0.2, -0.5, 0.1],
            wet_tap_r: [0.15, -0.65, 0.25, 0.45],
            allpass_g: [0.35, -0.25, 0.4, -0.3],
            allpass_delays: [3, 5, 7, 11],
            fb_psi: 0.4,
            scat_psi: 0.9,
            sigma_knee: 0.8,
            freeze_mix: 1.0,
            window_depth: 0.0,
            ..TransportQuadF64::default()
        };
        let (yl_h, yr_h, gh) = eval_op_t_quad_stereo_and_adjoint(
            &x_l,
            &x_r,
            &tq_hold,
            48_000.0,
            Some(OpTQuadAdjointSlices {
                d_out_l: &w_l,
                d_out_r: &w_r,
                grad_x_l: None,
                grad_x_r: None,
            }),
        );
        let mut tq_hold_z = tq_hold;
        tq_hold_z.scat_psi = 0.0;
        let (yl_hz, yr_hz, _) =
            eval_op_t_quad_stereo_and_adjoint(&x_l, &x_r, &tq_hold_z, 48_000.0, None);
        assert_eq!(yl_h, yl_hz);
        assert_eq!(yr_h, yr_hz);
        assert_eq!(gh.d_scat_psi, 0.0);
    }
}
