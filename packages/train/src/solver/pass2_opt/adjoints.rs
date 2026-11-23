//! Differentiable forward and exact reverse-time adjoint evaluators for stereo complex conjugate
//! Kautz modal banks, `LinearReadout` leak-pole HP (`ro_pole_48k`), and legacy 1-lane `Op-T`
//! helpers (`ref[train.pipeline.backbone-trajectory-opt]`).

use dnam_kernels::TANK_MAX_G_LIN_F32;
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;

use crate::control::stability::{MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP};

use super::super::model::KautzPairF64;
use super::op_t_quad::{
    blend_catmull_rom_with_derivs, cubic_soft_sat_and_deriv, half_alpha_cubic_and_deriv,
};

/// Minimum active Kautz pole radius enforced by `dnam-vm` runtime hydration.
const KAUTZ_R_MIN: f64 = 0.25;
/// Maximum stable Kautz pole radius enforced by `dnam-vm` runtime hydration.
const KAUTZ_R_MAX: f64 = 0.9995;

const _: () = assert!(KAUTZ_R_MIN > 0.0 && KAUTZ_R_MIN < KAUTZ_R_MAX && KAUTZ_R_MAX < 1.0);

/// Pre-resolved ZOH `LinearReadout` leak pole `p = ro_pole_48k^rho_core` and sensitivity `dp/dro`.
#[derive(Clone, Copy, Debug)]
pub(super) struct RoPoleResolved {
    pub p: f64,
    pub one_m_p: f64,
    pub dp_dro: f64,
}

impl RoPoleResolved {
    #[must_use]
    pub(super) fn new(ro_pole_48k: f64, rho_core: f64) -> Self {
        let rho = if rho_core.is_finite() && rho_core > 0.0 {
            rho_core.clamp(0.125, 8.0)
        } else {
            1.0
        };
        let ro_clamped = if ro_pole_48k.is_finite() && ro_pole_48k > 0.0 {
            ro_pole_48k.clamp(0.0, MAX_RO_POLE_48K_F64)
        } else {
            0.0
        };
        let (p, dp_dro) = if ro_clamped <= 0.0 {
            (
                0.0,
                if ro_pole_48k.is_finite() && ro_pole_48k >= 0.0 && (rho - 1.0).abs() <= 1.0e-12 {
                    1.0
                } else {
                    0.0
                },
            )
        } else {
            let p_eff = ro_clamped.powf(rho).clamp(0.0, MAX_RO_POLE_48K_F64);
            let dp = if ro_pole_48k < MAX_RO_POLE_48K_F64 {
                rho * ro_clamped.powf(rho - 1.0)
            } else {
                0.0
            };
            (p_eff, dp)
        };
        Self {
            p,
            one_m_p: 1.0 - p,
            dp_dro,
        }
    }

    #[inline]
    pub(super) fn step_forward_frame(self, u_t: f64, lp_prev: &mut f64) -> (f64, f64) {
        let lp_t = self.p * *lp_prev + self.one_m_p * u_t;
        *lp_prev = lp_t;
        (u_t - lp_t, lp_t)
    }

    #[inline]
    pub(super) fn step_reverse_frame(
        self,
        dy: f64,
        u_t: f64,
        lp_prev: f64,
        adj_lp_next: &mut f64,
        d_p_acc: &mut f64,
    ) -> f64 {
        let adj_lp = -dy + self.p * *adj_lp_next;
        *d_p_acc += adj_lp * (lp_prev - u_t);
        *adj_lp_next = adj_lp;
        dy + self.one_m_p * adj_lp
    }
}

/// Evaluates the 1-pole `LinearReadout` leak high-pass `hp[t] = u[t] - lp[t]` for ZOH-scaled
/// `p = ro_pole_48k^rho_core` matching `SonaVm`'s `step_core_subsample` and returns
/// `(hp, d_loss/d_ro_pole_48k)` while accumulating into `grad_u_opt`.
pub(super) fn eval_ro_pole_hp_and_adjoint(
    u: &[f64],
    ro_pole_48k: f64,
    rho_core: f64,
    d_out_opt: Option<&[f64]>,
    mut grad_u_opt: Option<&mut [f64]>,
) -> (Vec<f64>, f64) {
    let n = u.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let resolved = RoPoleResolved::new(ro_pole_48k, rho_core);
    let mut hp = vec![0.0_f64; n];
    let mut lp_hist = if d_out_opt.is_some() {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };
    let mut lp_prev = 0.0_f64;
    for (t, &ut) in u.iter().enumerate() {
        let (hp_t, lp_t) = resolved.step_forward_frame(ut, &mut lp_prev);
        hp[t] = hp_t;
        if d_out_opt.is_some() {
            lp_hist.push(lp_t);
        }
    }

    let mut d_p = 0.0_f64;
    if let Some(d_out) = d_out_opt {
        let mut adj_lp_next = 0.0_f64;
        for t in (0..n).rev() {
            let lp_p = if t > 0 { lp_hist[t - 1] } else { 0.0 };
            let gu_t =
                resolved.step_reverse_frame(d_out[t], u[t], lp_p, &mut adj_lp_next, &mut d_p);
            if let Some(gu) = grad_u_opt.as_deref_mut() {
                gu[t] += gu_t;
            }
        }
    }
    (hp, d_p * resolved.dp_dro)
}

#[derive(Clone, Copy)]
struct KautzResolvedPair {
    active: bool,
    r: f64,
    st: f64,
    ct: f64,
    ar: f64,
    ai: f64,
    bg: f64,
    c_re: f64,
    c_im: f64,
}

impl KautzResolvedPair {
    fn new(pair: &KautzPairF64, is_right: bool) -> Self {
        let r_raw = pair.r_low + 0.5 * pair.dr;
        if !r_raw.is_finite() || r_raw <= 0.0 || !pair.b_gamma.is_finite() || pair.b_gamma <= 0.0 {
            return Self {
                active: false,
                r: 0.0,
                st: 0.0,
                ct: 1.0,
                ar: 0.0,
                ai: 0.0,
                bg: 0.0,
                c_re: 0.0,
                c_im: 0.0,
            };
        }
        let r = r_raw.clamp(KAUTZ_R_MIN, KAUTZ_R_MAX);
        let (st, ct) = pair.theta.sin_cos();
        let (c_re, c_im) = if is_right {
            (pair.c_re_r, pair.c_im_r)
        } else {
            (pair.c_re_l, pair.c_im_l)
        };
        Self {
            active: true,
            r,
            st,
            ct,
            ar: r * ct,
            ai: r * st,
            bg: pair.b_gamma,
            c_re,
            c_im,
        }
    }
}

/// Stateful per-frame Kautz modal channel evaluator with optional state trajectory recording.
pub(super) struct KautzChannelStepper {
    pairs: Vec<KautzResolvedPair>,
    sr: Vec<f64>,
    si: Vec<f64>,
    sr_hist: Vec<f64>,
    si_hist: Vec<f64>,
}

impl KautzChannelStepper {
    #[must_use]
    pub(super) fn new(pairs: &[KautzPairF64], is_right: bool, cap_frames: usize) -> Self {
        let m = pairs.len();
        let resolved = pairs
            .iter()
            .map(|p| KautzResolvedPair::new(p, is_right))
            .collect();
        Self {
            pairs: resolved,
            sr: vec![0.0_f64; m],
            si: vec![0.0_f64; m],
            sr_hist: Vec::with_capacity(cap_frames * m),
            si_hist: Vec::with_capacity(cap_frames * m),
        }
    }

    #[inline]
    pub(super) fn step_forward_frame(&mut self, u_t: f64, record_trace: bool) -> f64 {
        let mut y_t = 0.0_f64;
        for (k, p) in self.pairs.iter().enumerate() {
            if !p.active {
                if record_trace {
                    self.sr_hist.push(0.0);
                    self.si_hist.push(0.0);
                }
                continue;
            }
            let nr = p.ar * self.sr[k] - p.ai * self.si[k] + p.bg * u_t;
            let ni = p.ai * self.sr[k] + p.ar * self.si[k];
            self.sr[k] = nr;
            self.si[k] = ni;
            if record_trace {
                self.sr_hist.push(nr);
                self.si_hist.push(ni);
            }
            y_t += 2.0 * (p.c_re * nr - p.c_im * ni);
        }
        y_t
    }
}

/// Reverse-time frame-by-frame adjoint accumulator paired with [`KautzChannelStepper`].
pub(super) struct KautzChannelReverseState {
    lam_r: Vec<f64>,
    lam_i: Vec<f64>,
    d_ar: Vec<f64>,
    d_ai: Vec<f64>,
    d_bg: Vec<f64>,
    d_cre: Vec<f64>,
    d_cim: Vec<f64>,
}

impl KautzChannelReverseState {
    #[must_use]
    pub(super) fn new(m: usize) -> Self {
        Self {
            lam_r: vec![0.0_f64; m],
            lam_i: vec![0.0_f64; m],
            d_ar: vec![0.0_f64; m],
            d_ai: vec![0.0_f64; m],
            d_bg: vec![0.0_f64; m],
            d_cre: vec![0.0_f64; m],
            d_cim: vec![0.0_f64; m],
        }
    }

    #[inline]
    pub(super) fn step_reverse_frame(
        &mut self,
        fwd: &KautzChannelStepper,
        t: usize,
        u_t: f64,
        d_out_t: f64,
    ) -> f64 {
        let m = fwd.pairs.len();
        let dt2 = 2.0 * d_out_t;
        let mut gu_t = 0.0_f64;
        let cur_base = t * m;
        for (k, p) in fwd.pairs.iter().enumerate() {
            if !p.active {
                continue;
            }
            let cur_r = fwd.sr_hist[cur_base + k];
            let cur_i = fwd.si_hist[cur_base + k];
            let (prev_r, prev_i) = if t > 0 {
                let prev_base = (t - 1) * m;
                (fwd.sr_hist[prev_base + k], fwd.si_hist[prev_base + k])
            } else {
                (0.0, 0.0)
            };
            self.d_cre[k] += dt2 * cur_r;
            self.d_cim[k] -= dt2 * cur_i;
            let adj_r = dt2 * p.c_re + self.lam_r[k];
            let adj_i = -dt2 * p.c_im + self.lam_i[k];
            gu_t += p.bg * adj_r;
            self.d_bg[k] += adj_r * u_t;
            self.d_ar[k] += adj_r * prev_r + adj_i * prev_i;
            self.d_ai[k] += adj_i * prev_r - adj_r * prev_i;
            self.lam_r[k] = p.ar * adj_r + p.ai * adj_i;
            self.lam_i[k] = -p.ai * adj_r + p.ar * adj_i;
        }
        gu_t
    }

    #[must_use]
    pub(super) fn finalize(&self, fwd: &KautzChannelStepper) -> Vec<[f64; 5]> {
        fwd.pairs
            .iter()
            .enumerate()
            .map(|(k, p)| {
                if !p.active {
                    [0.0; 5]
                } else {
                    let d_r = self.d_ar[k] * p.ct + self.d_ai[k] * p.st;
                    let d_theta = -self.d_ar[k] * p.r * p.st + self.d_ai[k] * p.r * p.ct;
                    [d_r, d_theta, self.d_bg[k], self.d_cre[k], self.d_cim[k]]
                }
            })
            .collect()
    }
}

/// Evaluates one stereo channel (`is_right = false -> L`, `true -> R`) of complex conjugate Kautz
/// resonators `kautz[0..count]` using exact runtime readout `2.0 * (c_re * sr - c_im * si)`,
/// returning reverse-time gradients `[d_r, d_theta, d_b_gamma, d_c_re, d_c_im]`.
pub fn eval_kautz_modal_channel_and_adjoint(
    u: &[f64],
    pairs: &[KautzPairF64],
    is_right: bool,
    d_out_opt: Option<&[f64]>,
    mut grad_u_opt: Option<&mut [f64]>,
) -> (Vec<f64>, Vec<[f64; 5]>) {
    let n = u.len();
    let m = pairs.len();
    let record = d_out_opt.is_some();
    let mut stepper = KautzChannelStepper::new(pairs, is_right, if record { n } else { 0 });
    let mut y = vec![0.0_f64; n];
    for t in 0..n {
        y[t] = stepper.step_forward_frame(u[t], record);
    }

    let Some(d_out) = d_out_opt else {
        return (y, vec![[0.0_f64; 5]; m]);
    };
    let mut rev = KautzChannelReverseState::new(m);
    for t in (0..n).rev() {
        let gu_t = rev.step_reverse_frame(&stepper, t, u[t], d_out[t]);
        if let Some(gu) = grad_u_opt.as_deref_mut() {
            gu[t] += gu_t;
        }
    }
    (y, rev.finalize(&stepper))
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

        // 1. ro_pole_48k ZOH highpass FD parity at 48 kHz and 96 kHz (`rho = 1.0, 0.5`).
        for &(ro, rho) in &[(0.92, 1.0), (0.85, 0.5)] {
            let mut gu_hp = vec![0.0_f64; 128];
            let (_, d_ro) =
                eval_ro_pole_hp_and_adjoint(&x, ro, rho, Some(&d_out), Some(&mut gu_hp));
            let fd_ro = (obj(&eval_ro_pole_hp_and_adjoint(&x, ro + eps, rho, None, None).0)
                - obj(&eval_ro_pole_hp_and_adjoint(&x, ro - eps, rho, None, None).0))
                / (2.0 * eps);
            assert_fd_close("ro_pole_48k", d_ro, fd_ro);
            let (mut up, mut um) = (x.clone(), x.clone());
            up[31] += eps;
            um[31] -= eps;
            let fd_u31 = (obj(&eval_ro_pole_hp_and_adjoint(&up, ro, rho, None, None).0)
                - obj(&eval_ro_pole_hp_and_adjoint(&um, ro, rho, None, None).0))
                / (2.0 * eps);
            assert_fd_close("ro_hp_u31", gu_hp[31], fd_u31);
        }

        // 2. Kautz Right-channel (c_re_r, c_im_r, r_low, theta, b_gamma) FD parity.
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
        for (label, idx, bump) in [
            ("kautz_r_low", 0_usize, 0_usize),
            ("kautz_theta", 1, 1),
            ("kautz_b_gamma", 2, 2),
        ] {
            let (mut pp, mut pm) = (pairs, pairs);
            match bump {
                0 => {
                    pp[0].r_low += eps;
                    pm[0].r_low -= eps;
                }
                1 => {
                    pp[0].theta += eps;
                    pm[0].theta -= eps;
                }
                _ => {
                    pp[0].b_gamma += eps;
                    pm[0].b_gamma -= eps;
                }
            }
            assert_fd_close(
                label,
                kr_grads[0][idx],
                (obj_r(&eval_kautz_modal_channel_and_adjoint(&x_r, &pp, true, None, None).0)
                    - obj_r(&eval_kautz_modal_channel_and_adjoint(&x_r, &pm, true, None, None).0))
                    / (2.0 * eps),
            );
        }
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
            ..TransportQuadF64::default()
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
