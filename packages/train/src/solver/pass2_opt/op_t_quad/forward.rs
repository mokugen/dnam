//! Exact frame-stepped `Op-T` `TransportQuad` forward evaluator with ZOH pole/delay scaling,
//! time-varying LFO + grain-ramp `tau(t)` modulation, Hermite-Hann diagonal windowing, fused
//! `SO(4)` + Schroeder allpass diffusion, pitch-down `g_loop_eff`, cubic `sigma_sat`, Givens
//! rotation, Householder mixing, loop damping, loop DC blocking, `freeze_mix` recirculation,
//! and 5 Hz stereo wet DC blocking (`ref[dsp.op-t.reader.catmull-rom]`).

use core::f64::consts::{PI, TAU};

use dnam_kernels::{TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_F32, TANK_MAX_V_CEIL_F32};
use dnam_vm::math::{DC_BLOCKER_5HZ_48K_F32, TANK_DC_BLOCKER_48K_F32};

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;

use super::super::super::model::TransportQuadF64;
use super::{
    apply_so4_even_f64, apply_so4_odd_f64, blend_catmull_rom_with_derivs, cubic_soft_sat_and_deriv,
    half_alpha_cubic_and_deriv,
};

const R_DC_MIN: f64 = 0.95;
const R_DC_MAX: f64 = dnam_kernels::MAX_PARALRU_R_F32 as f64;
const DAMP_MAX: f64 = 0.9995;
const DAMP_EFF_MAX: f64 = 0.999_95;
const LOOP_DC_ZOH_MIN: f64 = 1.0e-4;
const LOOP_DC_ZOH_MAX: f64 = 0.999_995;
const SIGMA_MIN: f64 = 1.0e-3;
const FREEZE_MAX: f64 = 0.999;
const MAX_PITCH_DOWN_NU: f64 = 0.75;
const MAX_PITCH_DOWN_U_SUB: f64 = 0.9999;
const _: () = assert!(
    R_DC_MIN < R_DC_MAX
        && DAMP_MAX < DAMP_EFF_MAX
        && LOOP_DC_ZOH_MIN < LOOP_DC_ZOH_MAX
        && FREEZE_MAX < 1.0
);

/// ZOH-resolved effective constants and analytic sensitivity multipliers for one `TransportQuad`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OpTResolvedParams {
    pub sr: f64,
    pub scale: f64,
    pub max_tau: f64,
    pub d_eff: [f64; 4],
    pub d_open: [bool; 4],
    pub span_hyd: [f64; 4],
    pub span_tap: [f64; 4],
    pub span_open: [bool; 4],
    pub lfo_d_eff: [f64; 4],
    pub lfo_d_open: [bool; 4],
    pub lfo_rate_hz: f64,
    pub lfo_rate_open: bool,
    pub ramp_rate_hz: f64,
    pub ramp_rate_open: bool,
    pub lfo_phase_off_q16: [u16; 4],
    pub ramp_phase_off_q16: [u16; 4],
    pub mu_freeze: f64,
    pub mu_freeze_open: bool,
    pub taper: f64,
    pub g_loop_eff: f64,
    pub dg_loop_eff_dg_base: f64,
    pub d_g_loop_eff_d_max_nu: f64,
    pub max_nu_arg_ties: [f64; 4],
    pub half_alpha: f64,
    pub d_half_dg_eff: f64,
    pub d_half_d_mu: f64,
    pub sigma_sat: f64,
    pub sigma_open: bool,
    pub inv_sigma_23: f64,
    pub r_dc_eff: f64,
    pub dr_dc_eff: f64,
    pub damp_eff: f64,
    pub ddamp_eff: f64,
    pub wet_dc_pole: f64,
    pub alpha_loop_dc: f64,
    pub alpha_wet_dc: f64,
    pub cos_psi: f64,
    pub sin_psi: f64,
    pub fb_psi_open: bool,
    pub c_scat: f64,
    pub s_scat: f64,
    pub scat_psi: f64,
    pub scat_open: bool,
    pub window_depth: f64,
    pub window_depth_open: bool,
    pub lfo_static: bool,
    pub s_lfo_const: [f64; 4],
    pub c_lfo_const: [f64; 4],
    pub ramp_static: bool,
    pub phi_ramp_const: [f64; 4],
    pub du_dphi_const: [f64; 4],
    pub h_win_const: [f64; 4],
    pub dh_du_const: [f64; 4],
    pub click_guard_raw_const: [f64; 4],
    pub mu_win_const: [f64; 4],
    pub win_w_const: [f64; 4],
    pub input_gains: [f64; 4],
    pub input_gain_open: [bool; 4],
    pub wet_tap_l: [f64; 4],
    pub wet_tap_l_open: [bool; 4],
    pub wet_tap_r: [f64; 4],
    pub wet_tap_r_open: [bool; 4],
    pub ap_delays: [usize; 4],
    pub ap_g_eff: [f64; 4],
    pub ap_g_open: [bool; 4],
}

impl OpTResolvedParams {
    pub(crate) fn from_quad_with_wet_dc(
        tq: &TransportQuadF64,
        sample_rate_hz: f64,
        n_frames: usize,
        wet_dc_pole_48k: f64,
    ) -> Self {
        let sr = if sample_rate_hz.is_finite() && sample_rate_hz > 0.0 {
            sample_rate_hz
        } else {
            48_000.0
        };
        let rho_vm = 48_000.0 / sr;
        let scale = 1.0 / rho_vm;
        let exact_48k = (rho_vm - 1.0).abs() <= 1.0e-12;
        let tau_min = MIN_DELAY_CROSSBAR_SAMP;
        let max_tau = (n_frames as f64).max(tau_min + 4.0).min(48_000.0 * scale);
        let max_span = (max_tau - tau_min).max(0.0);

        let freeze_raw = if tq.freeze_mix.is_finite() {
            tq.freeze_mix
        } else {
            0.0
        };
        let mu_freeze = freeze_raw.clamp(0.0, FREEZE_MAX);
        let mu_freeze_open =
            tq.freeze_mix.is_finite() && tq.freeze_mix >= 0.0 && tq.freeze_mix < FREEZE_MAX;
        let taper = 1.0 - mu_freeze;

        let mut d_eff = [tau_min; 4];
        let mut d_open = [false; 4];
        let mut span_hyd = [0.0_f64; 4];
        let mut span_tap = [0.0_f64; 4];
        let mut span_open = [false; 4];
        let mut lfo_d_eff = [0.0_f64; 4];
        let mut lfo_d_open = [false; 4];
        let mut ap_delays = [0_usize; 4];
        let mut ap_g_eff = [0.0_f64; 4];
        let mut ap_g_open = [false; 4];

        for k in 0..4 {
            let d_w = if tq.delay_samples[k].is_finite() {
                tq.delay_samples[k].clamp(tau_min, 48_000.0)
            } else {
                tau_min
            };
            let d_s = d_w * scale;
            d_eff[k] = d_s.clamp(tau_min, max_tau);
            d_open[k] = tq.delay_samples[k].is_finite()
                && tq.delay_samples[k] > tau_min
                && tq.delay_samples[k] < 48_000.0
                && d_s > tau_min
                && d_s < max_tau;

            let sp_w = if tq.ramp_spans_samples[k].is_finite() {
                tq.ramp_spans_samples[k].clamp(0.0, 48_000.0)
            } else {
                0.0
            };
            let sp_s = sp_w * scale;
            span_hyd[k] = sp_s.clamp(0.0, max_span);
            span_tap[k] = span_hyd[k] * taper;
            span_open[k] = tq.ramp_spans_samples[k].is_finite()
                && tq.ramp_spans_samples[k] >= 0.0
                && tq.ramp_spans_samples[k] < 48_000.0
                && sp_s < max_span;

            let ld_w = if tq.lfo_depths[k].is_finite() {
                tq.lfo_depths[k].clamp(0.0, 512.0)
            } else {
                0.0
            };
            lfo_d_eff[k] = ld_w * scale * taper;
            lfo_d_open[k] =
                tq.lfo_depths[k].is_finite() && tq.lfo_depths[k] >= 0.0 && tq.lfo_depths[k] < 512.0;

            if tq.allpass_delays[k] > 0 {
                ap_delays[k] = ((f64::from(tq.allpass_delays[k]) * scale).round() as usize).max(1);
            }
            let ag = if tq.allpass_g[k].is_finite() {
                tq.allpass_g[k].clamp(-0.75, 0.75)
            } else {
                0.0
            };
            ap_g_eff[k] = ag * taper;
            ap_g_open[k] = tq.allpass_g[k].is_finite() && tq.allpass_g[k].abs() < 0.75;
        }

        let lfo_rate_hz = if tq.lfo_rate_hz.is_finite() {
            tq.lfo_rate_hz.clamp(0.0, 16.0)
        } else {
            0.0
        };
        let lfo_rate_open =
            tq.lfo_rate_hz.is_finite() && tq.lfo_rate_hz >= 0.0 && tq.lfo_rate_hz < 16.0;
        let ramp_rate_hz = if tq.ramp_rate_hz.is_finite() {
            tq.ramp_rate_hz.clamp(-16.0, 16.0)
        } else {
            0.0
        };
        let ramp_rate_open = tq.ramp_rate_hz.is_finite() && tq.ramp_rate_hz.abs() < 16.0;

        let g_max = f64::from(TANK_MAX_G_LIN_F32);
        let g_loop = if tq.g_loop.is_finite() {
            tq.g_loop.clamp(0.0, g_max)
        } else {
            0.0
        };
        let g_loop_open = tq.g_loop.is_finite() && tq.g_loop >= 0.0 && tq.g_loop < g_max;

        let ramp_frac = ramp_rate_hz / sr;
        let mut nu_arr = [0.0_f64; 4];
        let mut max_nu = 0.0_f64;
        for k in 0..4 {
            let nu = (span_hyd[k] * ramp_frac).clamp(0.0, MAX_PITCH_DOWN_NU);
            nu_arr[k] = nu;
            if nu > max_nu {
                max_nu = nu;
            }
        }
        let mut max_nu_arg_ties = [0.0_f64; 4];
        if max_nu > 0.0 && max_nu < MAX_PITCH_DOWN_NU {
            let tie_cnt = nu_arr
                .iter()
                .filter(|&&nu| (nu - max_nu).abs() <= 1.0e-15)
                .count();
            if tie_cnt > 0 {
                let w = 1.0 / (tie_cnt as f64);
                for k in 0..4 {
                    if (nu_arr[k] - max_nu).abs() <= 1.0e-15 {
                        max_nu_arg_ties[k] = w;
                    }
                }
            }
        }
        let sqrt_u = if max_nu <= 0.0 {
            1.0
        } else {
            (1.0 - max_nu.min(MAX_PITCH_DOWN_U_SUB)).max(0.0).sqrt()
        };
        let g_eff_raw = g_loop * sqrt_u;
        let g_loop_eff = g_eff_raw.clamp(0.0, g_max);
        let dg_loop_eff_dg_base = if g_loop_open && g_eff_raw < g_max {
            sqrt_u
        } else {
            0.0
        };
        let d_g_loop_eff_d_max_nu =
            if max_nu > 0.0 && max_nu < MAX_PITCH_DOWN_NU && sqrt_u > 1.0e-9 && g_eff_raw < g_max {
                -0.5 * g_loop / sqrt_u
            } else {
                0.0
            };

        let (half_alpha, d_half_dg_eff) = half_alpha_cubic_and_deriv(g_loop_eff, mu_freeze);
        let alpha_cubic_only = if taper > 1.0e-12 {
            half_alpha / taper
        } else {
            let g_clamped = g_loop_eff.max(0.5);
            0.5 * ((0.9992_f64 / g_clamped - 1.0) * 4.0).clamp(0.0, 1.0)
        };
        let d_half_d_mu = if mu_freeze_open {
            -alpha_cubic_only
        } else {
            0.0
        };

        let sig_max = f64::from(TANK_MAX_SIGMA_SAT_F32);
        let sigma_sat = if tq.sigma_knee.is_finite() {
            tq.sigma_knee.clamp(SIGMA_MIN, sig_max)
        } else {
            sig_max
        };
        let sigma_open =
            tq.sigma_knee.is_finite() && tq.sigma_knee > SIGMA_MIN && tq.sigma_knee < sig_max;
        let inv_sigma_23 = 2.0 / (3.0 * sigma_sat);

        let r_dc_fallback = f64::from(TANK_DC_BLOCKER_48K_F32);
        let r_dc_ref = if tq.r_dc.is_finite() {
            tq.r_dc.clamp(R_DC_MIN, R_DC_MAX)
        } else {
            r_dc_fallback
        };
        let (r_dc_eff, dr_dc_eff) = if exact_48k {
            let d = if tq.r_dc.is_finite() && tq.r_dc > R_DC_MIN && tq.r_dc < R_DC_MAX {
                1.0
            } else {
                0.0
            };
            (r_dc_ref.clamp(LOOP_DC_ZOH_MIN, LOOP_DC_ZOH_MAX), d)
        } else {
            let r_e = r_dc_ref
                .powf(rho_vm)
                .clamp(LOOP_DC_ZOH_MIN, LOOP_DC_ZOH_MAX);
            let d = if tq.r_dc.is_finite()
                && tq.r_dc > R_DC_MIN
                && tq.r_dc < R_DC_MAX
                && r_e < LOOP_DC_ZOH_MAX
            {
                rho_vm * r_dc_ref.powf(rho_vm - 1.0)
            } else {
                0.0
            };
            (r_e, d)
        };

        let damp_ref = if tq.damp_pole.is_finite() {
            tq.damp_pole.clamp(0.0, DAMP_MAX)
        } else {
            0.0
        };
        let (damp_eff, ddamp_eff) = if damp_ref <= 1.0e-7 {
            (0.0, 0.0)
        } else if exact_48k {
            let d = if tq.damp_pole.is_finite() && tq.damp_pole > 1.0e-7 && tq.damp_pole < DAMP_MAX
            {
                1.0
            } else {
                0.0
            };
            (damp_ref.clamp(0.0, DAMP_EFF_MAX), d)
        } else {
            let d_e = damp_ref.powf(rho_vm).clamp(0.0, DAMP_EFF_MAX);
            let d = if tq.damp_pole.is_finite()
                && tq.damp_pole > 1.0e-7
                && tq.damp_pole < DAMP_MAX
                && d_e < DAMP_EFF_MAX
            {
                rho_vm * damp_ref.powf(rho_vm - 1.0)
            } else {
                0.0
            };
            (d_e, d)
        };

        let wet_dc_base =
            if wet_dc_pole_48k.is_finite() && wet_dc_pole_48k > 0.8 && wet_dc_pole_48k < 1.0 {
                wet_dc_pole_48k
            } else {
                f64::from(DC_BLOCKER_5HZ_48K_F32)
            };
        let wet_dc_pole = if exact_48k {
            wet_dc_base.clamp(LOOP_DC_ZOH_MIN, LOOP_DC_ZOH_MAX)
        } else {
            wet_dc_base
                .powf(rho_vm)
                .clamp(LOOP_DC_ZOH_MIN, LOOP_DC_ZOH_MAX)
        };

        let fb_psi = if tq.fb_psi.is_finite() {
            tq.fb_psi.clamp(-PI, PI)
        } else {
            0.0
        };
        let (sin_psi, cos_psi) = fb_psi.sin_cos();
        let fb_psi_open = tq.fb_psi.is_finite() && tq.fb_psi.abs() < PI;

        let scat_psi = if tq.scat_psi.is_finite() {
            tq.scat_psi.clamp(-PI, PI)
        } else {
            0.0
        };
        let scat_eff = scat_psi * taper;
        let (s_scat, c_scat) = scat_eff.sin_cos();
        let scat_open = tq.scat_psi.is_finite() && tq.scat_psi.abs() < PI;

        let window_depth = if tq.window_depth.is_finite() {
            tq.window_depth.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let window_depth_open =
            tq.window_depth.is_finite() && tq.window_depth >= 0.0 && tq.window_depth < 1.0;

        let lfo_static = lfo_rate_hz == 0.0;
        let ramp_static = ramp_rate_hz == 0.0;
        let mut s_lfo_const = [0.0_f64; 4];
        let mut c_lfo_const = [1.0_f64; 4];
        let mut phi_ramp_const = [0.0_f64; 4];
        let mut du_dphi_const = [2.0_f64; 4];
        let mut h_win_const = [0.0_f64; 4];
        let mut dh_du_const = [0.0_f64; 4];
        let click_guard_raw_const = [-1.0_f64; 4];
        let mu_win_const = [window_depth; 4];
        let mut win_w_const = [1.0_f64; 4];
        for k in 0..4 {
            let lfo_ph = f64::from(tq.lfo_phase_off_q16[k]) / 65_536.0;
            (s_lfo_const[k], c_lfo_const[k]) = (TAU * lfo_ph).sin_cos();
            let phi_r = (f64::from(tq.ramp_phase_off_q16[k]) / 65_536.0).rem_euclid(1.0);
            let (u_t, du_dp) = if phi_r <= 0.5 {
                (2.0 * phi_r, 2.0)
            } else {
                (2.0 * (1.0 - phi_r), -2.0)
            };
            let h_w = u_t * u_t * (3.0 - 2.0 * u_t);
            phi_ramp_const[k] = phi_r;
            du_dphi_const[k] = du_dp;
            h_win_const[k] = h_w;
            dh_du_const[k] = 6.0 * u_t * (1.0 - u_t);
            win_w_const[k] = (1.0 - window_depth) + window_depth * h_w;
        }

        let input_gains = tq.input_gains.map(|v| {
            if v.is_finite() {
                v.clamp(-2.0, 2.0)
            } else {
                0.0
            }
        });
        let input_gain_open = tq.input_gains.map(|v| v.is_finite() && v.abs() < 2.0);
        let wet_tap_l = tq.wet_tap_l.map(|v| {
            if v.is_finite() {
                v.clamp(-2.0, 2.0)
            } else {
                0.0
            }
        });
        let wet_tap_l_open = tq.wet_tap_l.map(|v| v.is_finite() && v.abs() < 2.0);
        let wet_tap_r = tq.wet_tap_r.map(|v| {
            if v.is_finite() {
                v.clamp(-2.0, 2.0)
            } else {
                0.0
            }
        });
        let wet_tap_r_open = tq.wet_tap_r.map(|v| v.is_finite() && v.abs() < 2.0);

        Self {
            sr,
            scale,
            max_tau,
            d_eff,
            d_open,
            span_hyd,
            span_tap,
            span_open,
            lfo_d_eff,
            lfo_d_open,
            lfo_rate_hz,
            lfo_rate_open,
            ramp_rate_hz,
            ramp_rate_open,
            lfo_phase_off_q16: tq.lfo_phase_off_q16,
            ramp_phase_off_q16: tq.ramp_phase_off_q16,
            mu_freeze,
            mu_freeze_open,
            taper,
            g_loop_eff,
            dg_loop_eff_dg_base,
            d_g_loop_eff_d_max_nu,
            max_nu_arg_ties,
            half_alpha,
            d_half_dg_eff,
            d_half_d_mu,
            sigma_sat,
            sigma_open,
            inv_sigma_23,
            r_dc_eff,
            dr_dc_eff,
            damp_eff,
            ddamp_eff,
            wet_dc_pole,
            alpha_loop_dc: 0.5 * (1.0 + r_dc_eff),
            alpha_wet_dc: 0.5 * (1.0 + wet_dc_pole),
            cos_psi,
            sin_psi,
            fb_psi_open,
            c_scat,
            s_scat,
            scat_psi,
            scat_open,
            window_depth,
            window_depth_open,
            lfo_static,
            s_lfo_const,
            c_lfo_const,
            ramp_static,
            phi_ramp_const,
            du_dphi_const,
            h_win_const,
            dh_du_const,
            click_guard_raw_const,
            mu_win_const,
            win_w_const,
            input_gains,
            input_gain_open,
            wet_tap_l,
            wet_tap_l_open,
            wet_tap_r,
            wet_tap_r_open,
            ap_delays,
            ap_g_eff,
            ap_g_open,
        }
    }
}

/// Top-of-Frame per-quad read outputs and cached modulation geometry for exact BPTT.
#[derive(Clone, Copy, Default)]
pub(crate) struct OpTTopRead {
    pub v_read: [f64; 4],
    pub r_win: [f64; 4],
    pub x1: [f64; 4],
    pub r_raw: [f64; 4],
    pub ap_delayed: [f64; 4],
    pub ap_v: [f64; 4],
    pub d_int: [usize; 4],
    pub mu_frac: [f64; 4],
    pub d_tau_vread: [f64; 4],
    pub d_half_v: [f64; 4],
    pub tau_open: [bool; 4],
    pub win_w: [f64; 4],
    pub h_win: [f64; 4],
    pub mu_win: [f64; 4],
    pub click_guard_raw: [f64; 4],
    pub du_dphi_tri: [f64; 4],
    pub dh_du_tri: [f64; 4],
    pub s_lfo: [f64; 4],
    pub c_lfo: [f64; 4],
    pub phi_ramp: [f64; 4],
}

#[derive(Clone, Copy, Default)]
pub(crate) struct OpTEndFrameTrace {
    pub top: OpTTopRead,
    pub v_loop: [f64; 4],
    pub d_loop_dv: [f64; 4],
    pub d_loop_dg_eff: [f64; 4],
    pub d_loop_dsigma: [f64; 4],
    pub v_fb: [f64; 4],
    pub v_damp: [f64; 4],
    pub y_dc: [f64; 4],
    pub wl_sum: f64,
    pub wr_sum: f64,
}

/// Stateful per-quad streaming evaluator used both in standalone batch calls and inside
/// frame-coupled [`super::super::causal_loop`] execution.
pub(crate) struct OpTQuadStepState {
    pub p: OpTResolvedParams,
    pub w_ring: Vec<[f64; 4]>,
    pub ap_buf: Vec<[f64; 4]>,
    pub damp_state: [f64; 4],
    pub dc_x_prev: [f64; 4],
    pub dc_y_prev: [f64; 4],
    pub wet_raw_prev: [f64; 2],
    pub wet_out_prev: [f64; 2],
    pub frames: Vec<OpTEndFrameTrace>,
    pub t_idx: usize,
}

impl OpTQuadStepState {
    pub(crate) fn new(p: OpTResolvedParams, n_frames: usize, record_trace: bool) -> Self {
        Self {
            p,
            w_ring: Vec::with_capacity(n_frames),
            ap_buf: Vec::with_capacity(n_frames),
            damp_state: [0.0; 4],
            dc_x_prev: [0.0; 4],
            dc_y_prev: [0.0; 4],
            wet_raw_prev: [0.0; 2],
            wet_out_prev: [0.0; 2],
            frames: if record_trace {
                Vec::with_capacity(n_frames)
            } else {
                Vec::new()
            },
            t_idx: 0,
        }
    }

    pub(crate) fn step_top_of_frame_read(&self) -> OpTTopRead {
        let t = self.t_idx;
        let tf = t as f64;
        let p = &self.p;
        let mut tr = OpTTopRead::default();
        let ramp_active = p.ramp_rate_hz.abs() > 1.0e-6;

        for k in 0..4 {
            let (s_lfo, c_lfo) = if p.lfo_static {
                (p.s_lfo_const[k], p.c_lfo_const[k])
            } else {
                let lfo_ph =
                    tf * p.lfo_rate_hz / p.sr + f64::from(p.lfo_phase_off_q16[k]) / 65_536.0;
                (TAU * lfo_ph).sin_cos()
            };
            let (phi_ramp, h_win, dh_du, du_dphi, cg_raw, mu_win, win_w) = if p.ramp_static {
                (
                    p.phi_ramp_const[k],
                    p.h_win_const[k],
                    p.dh_du_const[k],
                    p.du_dphi_const[k],
                    p.click_guard_raw_const[k],
                    p.mu_win_const[k],
                    p.win_w_const[k],
                )
            } else {
                let phi_r = (tf * p.ramp_rate_hz / p.sr
                    + f64::from(p.ramp_phase_off_q16[k]) / 65_536.0)
                    .rem_euclid(1.0);
                let (u_tri, du_d) = if phi_r <= 0.5 {
                    (2.0 * phi_r, 2.0)
                } else {
                    (2.0 * (1.0 - phi_r), -2.0)
                };
                let hw = u_tri * u_tri * (3.0 - 2.0 * u_tri);
                let dhw = 6.0 * u_tri * (1.0 - u_tri);
                let abs_span = if ramp_active {
                    p.span_tap[k].abs()
                } else {
                    0.0
                };
                let cg_r = (abs_span - 0.25) * 4.0;
                let cg = cg_r.clamp(0.0, 1.0);
                let mu_w = p.window_depth.max(cg);
                let ww = (1.0 - mu_w) + mu_w * hw;
                (phi_r, hw, dhw, du_d, cg_r, mu_w, ww)
            };
            let tau_raw = p.d_eff[k] + p.lfo_d_eff[k] * s_lfo + p.span_tap[k] * (phi_ramp - 0.5);
            let tau = tau_raw.clamp(MIN_DELAY_CROSSBAR_SAMP, p.max_tau);
            let dk = (tau.floor() as usize).max(MIN_DELAY_CROSSBAR_SAMP as usize);
            let mu_frac = (tau - (dk as f64)).clamp(0.0, 1.0);
            let omt = 1.0 - mu_frac;

            let s_m1 = if t + 1 >= dk && t + 1 - dk < self.w_ring.len() {
                self.w_ring[t + 1 - dk][k]
            } else {
                0.0
            };
            let s_0 = if t >= dk { self.w_ring[t - dk][k] } else { 0.0 };
            let s_1 = if t > dk {
                self.w_ring[t - dk - 1][k]
            } else {
                0.0
            };
            let s_2 = if t >= dk + 2 {
                self.w_ring[t - dk - 2][k]
            } else {
                0.0
            };
            let (rr, d_mu, _) =
                blend_catmull_rom_with_derivs(s_m1, s_0, s_1, s_2, mu_frac, p.half_alpha);
            let y_curv = omt * (2.0 * s_0 - s_m1 - s_1) + mu_frac * (2.0 * s_1 - s_0 - s_2);

            tr.r_raw[k] = rr;
            tr.r_win[k] = win_w * rr;
            tr.d_int[k] = dk;
            tr.mu_frac[k] = mu_frac;
            tr.d_tau_vread[k] = d_mu;
            tr.d_half_v[k] = mu_frac * omt * y_curv;
            tr.tau_open[k] = tau_raw > MIN_DELAY_CROSSBAR_SAMP && tau_raw < p.max_tau;
            tr.win_w[k] = win_w;
            tr.h_win[k] = h_win;
            tr.mu_win[k] = mu_win;
            tr.click_guard_raw[k] = cg_raw;
            tr.du_dphi_tri[k] = du_dphi;
            tr.dh_du_tri[k] = dh_du;
            tr.s_lfo[k] = s_lfo;
            tr.c_lfo[k] = c_lfo;
            tr.phi_ramp[k] = phi_ramp;
        }

        let x1 = apply_so4_even_f64(tr.r_win, p.c_scat, p.s_scat);
        tr.x1 = x1;
        let mut y1 = [0.0_f64; 4];
        for k in 0..4 {
            let d_ap = p.ap_delays[k];
            if d_ap > 0 {
                let del = if t >= d_ap {
                    self.ap_buf[t - d_ap][k]
                } else {
                    0.0
                };
                let g_ap = p.ap_g_eff[k];
                let v_ap = x1[k] - g_ap * del;
                tr.ap_delayed[k] = del;
                tr.ap_v[k] = v_ap;
                y1[k] = del + g_ap * v_ap;
            } else {
                y1[k] = x1[k];
            }
        }
        tr.v_read = apply_so4_odd_f64(y1, p.c_scat, p.s_scat);
        tr
    }

    pub(crate) fn step_end_of_frame_write(
        &mut self,
        top: &OpTTopRead,
        x_l: f64,
        x_r: f64,
    ) -> (f64, f64) {
        let p = &self.p;
        self.ap_buf.push(top.ap_v);

        let v_ceil = f64::from(TANK_MAX_V_CEIL_F32);
        let gain_scale = p.g_loop_eff * p.sigma_sat;
        let mut v_loop = [0.0_f64; 4];
        let mut d_loop_dv = [0.0_f64; 4];
        let mut d_loop_dg_eff = [0.0_f64; 4];
        let mut d_loop_dsigma = [0.0_f64; 4];
        for k in 0..4 {
            let (vl, dv, dgs) = cubic_soft_sat_and_deriv(top.v_read[k], gain_scale, p.inv_sigma_23);
            let scaled = top.v_read[k] * p.inv_sigma_23;
            let v_norm = scaled.clamp(-1.0, 1.0);
            let cubic = v_norm * (1.5 - 0.5 * v_norm * v_norm);
            let raw_sat = gain_scale * cubic;
            v_loop[k] = vl;
            d_loop_dv[k] = dv;
            d_loop_dg_eff[k] = dgs * p.sigma_sat;
            d_loop_dsigma[k] = if raw_sat.abs() < v_ceil {
                p.g_loop_eff * v_norm * v_norm * v_norm
            } else {
                0.0
            };
        }

        let r_giv = [
            p.cos_psi * v_loop[0] - p.sin_psi * v_loop[1],
            p.sin_psi * v_loop[0] + p.cos_psi * v_loop[1],
            p.cos_psi * v_loop[2] - p.sin_psi * v_loop[3],
            p.sin_psi * v_loop[2] + p.cos_psi * v_loop[3],
        ];
        let half_sum = 0.5 * (r_giv[0] + r_giv[1] + r_giv[2] + r_giv[3]);
        let xq = [x_l, x_r, x_l, x_r];
        let one_m_dp = 1.0 - p.damp_eff;
        let mut v_fb = [0.0_f64; 4];
        let mut v_damp = [0.0_f64; 4];
        let mut y_dc = [0.0_f64; 4];
        let mut wr_t = [0.0_f64; 4];
        for k in 0..4 {
            let fb = (r_giv[k] - half_sum) + p.input_gains[k] * xq[k];
            let dp_prev = self.damp_state[k];
            let vd = one_m_dp * fb + p.damp_eff * dp_prev;
            let ydc = p.alpha_loop_dc * (vd - self.dc_x_prev[k]) + p.r_dc_eff * self.dc_y_prev[k];
            self.damp_state[k] = vd;
            self.dc_x_prev[k] = vd;
            self.dc_y_prev[k] = ydc;
            v_fb[k] = fb;
            v_damp[k] = vd;
            y_dc[k] = ydc;
            wr_t[k] = p.taper * ydc + p.mu_freeze * top.r_win[k];
        }
        self.w_ring.push(wr_t);

        let (mut wl_sum, mut wr_sum) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            wl_sum += p.wet_tap_l[k] * top.v_read[k];
            wr_sum += p.wet_tap_r[k] * top.v_read[k];
        }
        let wl_raw = p.cos_psi * wl_sum - p.sin_psi * wr_sum;
        let wr_raw = p.sin_psi * wl_sum + p.cos_psi * wr_sum;
        let yl =
            p.alpha_wet_dc * (wl_raw - self.wet_raw_prev[0]) + p.wet_dc_pole * self.wet_out_prev[0];
        let yr =
            p.alpha_wet_dc * (wr_raw - self.wet_raw_prev[1]) + p.wet_dc_pole * self.wet_out_prev[1];
        self.wet_raw_prev = [wl_raw, wr_raw];
        self.wet_out_prev = [yl, yr];

        if !self.frames.capacity().eq(&0) {
            self.frames.push(OpTEndFrameTrace {
                top: *top,
                v_loop,
                d_loop_dv,
                d_loop_dg_eff,
                d_loop_dsigma,
                v_fb,
                v_damp,
                y_dc,
                wl_sum,
                wr_sum,
            });
        }
        self.t_idx += 1;
        (yl, yr)
    }

    /// Evaluates causal wet stereo output `(yl, yr)` for `top` without mutating quad state.
    #[inline]
    pub(crate) fn preview_wet_output(&self, top: &OpTTopRead) -> (f64, f64) {
        let p = &self.p;
        let (mut wl_sum, mut wr_sum) = (0.0_f64, 0.0_f64);
        for k in 0..4 {
            wl_sum += p.wet_tap_l[k] * top.v_read[k];
            wr_sum += p.wet_tap_r[k] * top.v_read[k];
        }
        let wl_raw = p.cos_psi * wl_sum - p.sin_psi * wr_sum;
        let wr_raw = p.sin_psi * wl_sum + p.cos_psi * wr_sum;
        let yl =
            p.alpha_wet_dc * (wl_raw - self.wet_raw_prev[0]) + p.wet_dc_pole * self.wet_out_prev[0];
        let yr =
            p.alpha_wet_dc * (wr_raw - self.wet_raw_prev[1]) + p.wet_dc_pole * self.wet_out_prev[1];
        (yl, yr)
    }
}
