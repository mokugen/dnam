//! Exact reverse-time BPTT for all 38 continuous `TransportQuad` parameters (delays, LFO
//! modulation, grain ramps, Hermite-Hann diagonal window, `SO(4)` scattering, Schroeder allpass,
//! pitch-down `g_loop_eff`, cubic `sigma_sat`, Givens `fb_psi`, HF damping, loop DC pole `r_dc`,
//! continuous `freeze_mix`, injection/wet taps) and frame-stepped crossbar feedback
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::TAU;

use super::forward::OpTQuadStepState;
use super::{
    OpTQuadAdjointGrads, apply_so4_even_transpose_f64, apply_so4_odd_transpose_f64,
    so4_theta_grad_f64,
};

/// Reverse-time per-quad BPTT accumulators supporting both batch and frame-interleaved
/// [`super::super::causal_loop`] stepping.
pub(crate) struct OpTQuadReverseState {
    pub adj_w: Vec<[f64; 4]>,
    pub adj_ap_w: Vec<[f64; 4]>,
    pub adj_ydc_next: [f64; 4],
    pub adj_vd_next_dc: [f64; 4],
    pub adj_vd_next_dp: [f64; 4],
    pub adj_out_wet_next: [f64; 2],
    pub adj_raw_wet_next: [f64; 2],
    pub d_theta_scat: f64,
    pub d_g_loop_eff: f64,
    pub d_half_alpha: f64,
    pub d_r_dc_eff: f64,
    pub d_damp_eff: f64,
    pub grads: OpTQuadAdjointGrads,
}

impl OpTQuadReverseState {
    pub(crate) fn new(n_frames: usize) -> Self {
        Self {
            adj_w: vec![[0.0_f64; 4]; n_frames],
            adj_ap_w: vec![[0.0_f64; 4]; n_frames],
            adj_ydc_next: [0.0; 4],
            adj_vd_next_dc: [0.0; 4],
            adj_vd_next_dp: [0.0; 4],
            adj_out_wet_next: [0.0; 2],
            adj_raw_wet_next: [0.0; 2],
            d_theta_scat: 0.0,
            d_g_loop_eff: 0.0,
            d_half_alpha: 0.0,
            d_r_dc_eff: 0.0,
            d_damp_eff: 0.0,
            grads: OpTQuadAdjointGrads::default(),
        }
    }

    /// Backprops frame `t` End-of-Frame wet output + loop writeback, returning
    /// `(d_v_read_end, d_r_win_freeze, d_x_l, d_x_r)`.
    pub(crate) fn reverse_end_of_frame_write(
        &mut self,
        fwd: &OpTQuadStepState,
        t: usize,
        x_l: f64,
        x_r: f64,
        d_out_l: f64,
        d_out_r: f64,
    ) -> ([f64; 4], [f64; 4], f64, f64) {
        let p = &fwd.p;
        let tr = &fwd.frames[t];
        let top = &tr.top;

        let d_yl = d_out_l + p.wet_dc_pole * self.adj_out_wet_next[0];
        let d_yr = d_out_r + p.wet_dc_pole * self.adj_out_wet_next[1];
        self.adj_out_wet_next = [d_yl, d_yr];

        let d_wl_raw = p.alpha_wet_dc * d_yl + self.adj_raw_wet_next[0];
        let d_wr_raw = p.alpha_wet_dc * d_yr + self.adj_raw_wet_next[1];
        self.adj_raw_wet_next = [-p.alpha_wet_dc * d_yl, -p.alpha_wet_dc * d_yr];

        let d_wl_sum = p.cos_psi * d_wl_raw + p.sin_psi * d_wr_raw;
        let d_wr_sum = -p.sin_psi * d_wl_raw + p.cos_psi * d_wr_raw;
        if p.fb_psi_open {
            self.grads.d_fb_psi += d_wl_raw * (-p.sin_psi * tr.wl_sum - p.cos_psi * tr.wr_sum)
                + d_wr_raw * (p.cos_psi * tr.wl_sum - p.sin_psi * tr.wr_sum);
        }

        let mut d_vr_end = [0.0_f64; 4];
        let mut d_rwin_freeze = [0.0_f64; 4];
        let mut adj_vfb = [0.0_f64; 4];
        let xq = [x_l, x_r, x_l, x_r];
        let one_m_dp = 1.0 - p.damp_eff;

        for k in 0..4 {
            if p.wet_tap_l_open[k] {
                self.grads.d_wet_tap_l[k] += d_wl_sum * top.v_read[k];
            }
            if p.wet_tap_r_open[k] {
                self.grads.d_wet_tap_r[k] += d_wr_sum * top.v_read[k];
            }
            d_vr_end[k] += d_wl_sum * p.wet_tap_l[k] + d_wr_sum * p.wet_tap_r[k];

            let aw = self.adj_w[t][k];
            if p.mu_freeze_open {
                self.grads.d_freeze_mix += aw * (top.r_win[k] - tr.y_dc[k]);
            }
            d_rwin_freeze[k] = p.mu_freeze * aw;

            let adj_ydc = p.taper * aw + p.r_dc_eff * self.adj_ydc_next[k];
            self.adj_ydc_next[k] = adj_ydc;
            let vd_prev = if t > 0 {
                fwd.frames[t - 1].v_damp[k]
            } else {
                0.0
            };
            let ydc_prev = if t > 0 {
                fwd.frames[t - 1].y_dc[k]
            } else {
                0.0
            };
            self.d_r_dc_eff += adj_ydc * (0.5 * (tr.v_damp[k] - vd_prev) + ydc_prev);

            let adj_vd = p.alpha_loop_dc * adj_ydc
                + self.adj_vd_next_dc[k]
                + p.damp_eff * self.adj_vd_next_dp[k];
            self.adj_vd_next_dc[k] = -p.alpha_loop_dc * adj_ydc;
            self.adj_vd_next_dp[k] = adj_vd;

            self.d_damp_eff += adj_vd * (vd_prev - tr.v_fb[k]);
            adj_vfb[k] = one_m_dp * adj_vd;
            if p.input_gain_open[k] {
                self.grads.d_input_gains[k] += adj_vfb[k] * xq[k];
            }
        }
        let d_x_l = adj_vfb[0] * p.input_gains[0] + adj_vfb[2] * p.input_gains[2];
        let d_x_r = adj_vfb[1] * p.input_gains[1] + adj_vfb[3] * p.input_gains[3];

        let half_adj = 0.5 * (adj_vfb[0] + adj_vfb[1] + adj_vfb[2] + adj_vfb[3]);
        let a_rg = [
            adj_vfb[0] - half_adj,
            adj_vfb[1] - half_adj,
            adj_vfb[2] - half_adj,
            adj_vfb[3] - half_adj,
        ];
        if p.fb_psi_open {
            self.grads.d_fb_psi += a_rg[0] * (-p.sin_psi * tr.v_loop[0] - p.cos_psi * tr.v_loop[1])
                + a_rg[1] * (p.cos_psi * tr.v_loop[0] - p.sin_psi * tr.v_loop[1])
                + a_rg[2] * (-p.sin_psi * tr.v_loop[2] - p.cos_psi * tr.v_loop[3])
                + a_rg[3] * (p.cos_psi * tr.v_loop[2] - p.sin_psi * tr.v_loop[3]);
        }
        let a_vloop = [
            p.cos_psi * a_rg[0] + p.sin_psi * a_rg[1],
            -p.sin_psi * a_rg[0] + p.cos_psi * a_rg[1],
            p.cos_psi * a_rg[2] + p.sin_psi * a_rg[3],
            -p.sin_psi * a_rg[2] + p.cos_psi * a_rg[3],
        ];

        for k in 0..4 {
            self.d_g_loop_eff += a_vloop[k] * tr.d_loop_dg_eff[k];
            if p.sigma_open {
                self.grads.d_sigma_knee += a_vloop[k] * tr.d_loop_dsigma[k];
            }
            d_vr_end[k] += a_vloop[k] * tr.d_loop_dv[k];
        }
        (d_vr_end, d_rwin_freeze, d_x_l, d_x_r)
    }

    /// Backprops Top-of-Frame read `v_read` (end + crossbar feedback) through `R_odd`,
    /// Schroeder allpass, `R_even`, Hermite-Hann diagonal window, and time-varying Catmull-Rom.
    pub(crate) fn reverse_top_of_frame_read(
        &mut self,
        fwd: &OpTQuadStepState,
        t: usize,
        d_v_read_total: [f64; 4],
        d_r_win_freeze: [f64; 4],
    ) {
        let p = &fwd.p;
        let top = &fwd.frames[t].top;
        let tf = t as f64;
        let ramp_active = p.ramp_rate_hz.abs() > 1.0e-6;

        let g_y1_t = apply_so4_odd_transpose_f64(d_v_read_total, p.c_scat, p.s_scat);
        let mut g_x1_t = [0.0_f64; 4];
        for k in 0..4 {
            let d_ap = p.ap_delays[k];
            if d_ap > 0 {
                let g_ap = p.ap_g_eff[k];
                let d_ap_v = g_ap * g_y1_t[k] + self.adj_ap_w[t][k];
                g_x1_t[k] = d_ap_v;
                if t >= d_ap {
                    self.adj_ap_w[t - d_ap][k] += g_y1_t[k] - g_ap * d_ap_v;
                }
                let d_ap_g_eff = g_y1_t[k] * top.ap_v[k] - d_ap_v * top.ap_delayed[k];
                if p.ap_g_open[k] {
                    self.grads.d_allpass_g[k] += d_ap_g_eff * p.taper;
                }
                if p.mu_freeze_open {
                    let ag_base = p.ap_g_eff[k] / p.taper.max(1.0e-12);
                    self.grads.d_freeze_mix += -d_ap_g_eff * ag_base;
                }
            } else {
                g_x1_t[k] = g_y1_t[k];
            }
        }
        self.d_theta_scat += so4_theta_grad_f64(d_v_read_total, top.v_read, g_x1_t, top.x1);
        let g_r_win_scat = apply_so4_even_transpose_f64(g_x1_t, p.c_scat, p.s_scat);

        for k in 0..4 {
            let d_rwin = d_r_win_freeze[k] + g_r_win_scat[k];
            let d_win_w = d_rwin * top.r_raw[k];
            let d_h_win = d_win_w * top.mu_win[k];
            if p.ramp_rate_open {
                self.grads.d_ramp_rate_hz +=
                    d_h_win * top.dh_du_tri[k] * top.du_dphi_tri[k] * (tf / p.sr);
            }
            let d_mu_win = d_win_w * (top.h_win[k] - 1.0);
            let cg = top.click_guard_raw[k].clamp(0.0, 1.0);
            let wd_dominates = p.window_depth >= cg;
            let cg_dominates = cg > p.window_depth;
            if wd_dominates && p.window_depth_open {
                self.grads.d_window_depth += d_mu_win;
            } else if cg_dominates
                && ramp_active
                && top.click_guard_raw[k] > 0.0
                && top.click_guard_raw[k] < 1.0
            {
                let sgn = if p.span_tap[k] >= 0.0 { 1.0 } else { -1.0 };
                let d_span_tap_cg = d_mu_win * 4.0 * sgn;
                if p.span_open[k] {
                    self.grads.d_ramp_spans_samples[k] += d_span_tap_cg * p.scale * p.taper;
                }
                if p.mu_freeze_open {
                    self.grads.d_freeze_mix += -d_span_tap_cg * p.span_hyd[k];
                }
            }

            let d_rraw = d_rwin * top.win_w[k];
            self.d_half_alpha += d_rraw * top.d_half_v[k];
            if top.tau_open[k] {
                let d_tau = d_rraw * top.d_tau_vread[k];
                if p.d_open[k] {
                    self.grads.d_delay_samples[k] += d_tau * p.scale;
                }
                if p.lfo_d_open[k] {
                    self.grads.d_lfo_depths[k] += d_tau * p.scale * p.taper * top.s_lfo[k];
                }
                if p.lfo_rate_open {
                    self.grads.d_lfo_rate_hz +=
                        d_tau * p.lfo_d_eff[k] * top.c_lfo[k] * TAU * (tf / p.sr);
                }
                if p.span_open[k] {
                    self.grads.d_ramp_spans_samples[k] +=
                        d_tau * p.scale * p.taper * (top.phi_ramp[k] - 0.5);
                }
                if p.ramp_rate_open {
                    self.grads.d_ramp_rate_hz += d_tau * p.span_tap[k] * (tf / p.sr);
                }
                if p.mu_freeze_open {
                    let lfo_un = p.lfo_d_eff[k] / p.taper.max(1.0e-12);
                    self.grads.d_freeze_mix +=
                        -d_tau * (lfo_un * top.s_lfo[k] + p.span_hyd[k] * (top.phi_ramp[k] - 0.5));
                }
            }

            let dk = top.d_int[k];
            let mu = top.mu_frac[k];
            let omt = 1.0 - mu;
            let ac = p.half_alpha * mu * omt;
            let w_taps = [
                -ac * omt,
                omt + ac * (2.0 * omt - mu),
                mu + ac * (2.0 * mu - omt),
                -ac * mu,
            ];
            if t + 1 >= dk && t + 1 - dk < self.adj_w.len() {
                self.adj_w[t + 1 - dk][k] += d_rraw * w_taps[0];
            }
            if t >= dk {
                self.adj_w[t - dk][k] += d_rraw * w_taps[1];
            }
            if t > dk {
                self.adj_w[t - dk - 1][k] += d_rraw * w_taps[2];
            }
            if t >= dk + 2 {
                self.adj_w[t - dk - 2][k] += d_rraw * w_taps[3];
            }
        }
    }

    pub(crate) fn finalize(mut self, fwd: &OpTQuadStepState) -> OpTQuadAdjointGrads {
        let p = &fwd.p;
        self.d_g_loop_eff += self.d_half_alpha * p.d_half_dg_eff;
        if p.mu_freeze_open {
            self.grads.d_freeze_mix += self.d_half_alpha * p.d_half_d_mu;
        }
        self.grads.d_g_loop += self.d_g_loop_eff * p.dg_loop_eff_dg_base;
        let d_max_nu = self.d_g_loop_eff * p.d_g_loop_eff_d_max_nu;
        if d_max_nu != 0.0 {
            let ramp_frac = p.ramp_rate_hz / p.sr;
            for k in 0..4 {
                let w = p.max_nu_arg_ties[k];
                if w > 0.0 {
                    if p.span_open[k] {
                        self.grads.d_ramp_spans_samples[k] += d_max_nu * w * p.scale * ramp_frac;
                    }
                    if p.ramp_rate_open {
                        self.grads.d_ramp_rate_hz += d_max_nu * w * (p.span_hyd[k] / p.sr);
                    }
                }
            }
        }
        if p.scat_open {
            self.grads.d_scat_psi += self.d_theta_scat * p.taper;
        }
        if p.mu_freeze_open {
            self.grads.d_freeze_mix += -self.d_theta_scat * p.scat_psi;
        }
        self.grads.d_r_dc += self.d_r_dc_eff * p.dr_dc_eff;
        self.grads.d_damp_pole += self.d_damp_eff * p.ddamp_eff;
        self.grads
    }
}
