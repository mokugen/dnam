//! Exact 38-parameter 4-lane `Op-T` transport quad evaluator & reverse-time BPTT facade over
//! [`forward`] and [`adjoint`], supporting `1..=4` parallel `TransportQuad`s (`ref[dsp.op-t.ring.declaration]`,
//! `ref[train.pipeline.backbone-trajectory-opt]`).

pub(crate) mod adjoint;
pub(crate) mod forward;

use dnam_kernels::{TANK_MAX_G_LIN_F32, TANK_MAX_V_CEIL_F32};

use super::super::model::TransportQuadF64;
use adjoint::OpTQuadReverseState;
use forward::{OpTQuadStepState, OpTResolvedParams};

/// Exact `38`-parameter gradients for one [`TransportQuadF64`] in canonical wire/theta order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpTQuadAdjointGrads {
    /// Gradient wrt base delay lengths in 48 kHz samples (`4` lanes).
    pub d_delay_samples: [f64; 4],
    /// Gradient wrt ramp span widths in 48 kHz samples (`4` lanes).
    pub d_ramp_spans_samples: [f64; 4],
    /// Gradient wrt raw feedback loop gain.
    pub d_g_loop: f64,
    /// Gradient wrt loop DC blocker pole (`48 kHz`).
    pub d_r_dc: f64,
    /// Gradient wrt loop damping pole (`48 kHz`).
    pub d_damp_pole: f64,
    /// Gradient wrt modulation LFO frequency (`Hz`).
    pub d_lfo_rate_hz: f64,
    /// Gradient wrt grain/delay ramp frequency (`Hz`).
    pub d_ramp_rate_hz: f64,
    /// Gradient wrt LFO depth (`4` lanes).
    pub d_lfo_depths: [f64; 4],
    /// Gradient wrt loop injection gains (`4` lanes).
    pub d_input_gains: [f64; 4],
    /// Gradient wrt embedded Schroeder allpass coefficients (`4` lanes).
    pub d_allpass_g: [f64; 4],
    /// Gradient wrt Hadamard/SO(4) feedback rotation angle.
    pub d_fb_psi: f64,
    /// Gradient wrt cubic soft-saturator knee parameter.
    pub d_sigma_knee: f64,
    /// Gradient wrt freeze mix `[0, 0.999]`.
    pub d_freeze_mix: f64,
    /// Gradient wrt dual-tap window depth `[0, 1]`.
    pub d_window_depth: f64,
    /// Gradient wrt read/write paraunitary scattering angle.
    pub d_scat_psi: f64,
    /// Gradient wrt left wet output tap weights (`4` lanes).
    pub d_wet_tap_l: [f64; 4],
    /// Gradient wrt right wet output tap weights (`4` lanes).
    pub d_wet_tap_r: [f64; 4],
}

/// Optional reverse-time adjoint inputs and gradient destinations for `Op-T`.
#[derive(Debug)]
pub struct OpTQuadAdjointSlices<'a> {
    /// Reverse sensitivity wrt left wet output `[0..n]`.
    pub d_out_l: &'a [f64],
    /// Reverse sensitivity wrt right wet output `[0..n]`.
    pub d_out_r: &'a [f64],
    /// Optional accumulator for reverse sensitivity wrt left input `[0..n]`.
    pub grad_x_l: Option<&'a mut [f64]>,
    /// Optional accumulator for reverse sensitivity wrt right input `[0..n]`.
    pub grad_x_r: Option<&'a mut [f64]>,
}

/// Returns `true` when any delay/feedback/tap path on `tq` is active.
#[inline]
#[must_use]
pub(crate) fn is_op_t_quad_active(tq: &TransportQuadF64) -> bool {
    let any_inj_or_wet = tq
        .input_gains
        .iter()
        .chain(&tq.wet_tap_l)
        .chain(&tq.wet_tap_r)
        .any(|&v| v.abs() > 1.0e-12);
    any_inj_or_wet || tq.g_loop.abs() > 1.0e-12
}

/// Returns `true` when `tq` has both a non-zero input injection gain and a non-zero stereo wet
/// output tap, which is necessary and sufficient for non-zero forward wet output.
#[inline]
#[must_use]
pub(crate) fn has_wet_op_t_output(tq: &TransportQuadF64) -> bool {
    let has_in = tq.input_gains.iter().any(|&v| v.abs() > 1.0e-12);
    let has_wet = tq
        .wet_tap_l
        .iter()
        .chain(&tq.wet_tap_r)
        .any(|&v| v.abs() > 1.0e-12);
    has_in && has_wet
}

#[inline]
pub(super) fn j_even_f64(x: [f64; 4]) -> [f64; 4] {
    [-x[1], x[0], -x[3], x[2]]
}

#[inline]
pub(super) fn j_odd_f64(y: [f64; 4]) -> [f64; 4] {
    [y[3], -y[2], y[1], -y[0]]
}

#[inline]
pub(super) fn apply_so4_even_f64(x: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    if (c - 1.0).abs() <= 1.0e-15 && s == 0.0 {
        return x;
    }
    let j = j_even_f64(x);
    core::array::from_fn(|k| c * x[k] + s * j[k])
}

#[inline]
pub(super) fn apply_so4_odd_f64(y: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    if (c - 1.0).abs() <= 1.0e-15 && s == 0.0 {
        return y;
    }
    let j = j_odd_f64(y);
    core::array::from_fn(|k| c * y[k] + s * j[k])
}

#[inline]
pub(super) fn apply_so4_even_transpose_f64(g: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    apply_so4_even_f64(g, c, -s)
}

#[inline]
pub(super) fn apply_so4_odd_transpose_f64(g: [f64; 4], c: f64, s: f64) -> [f64; 4] {
    apply_so4_odd_f64(g, c, -s)
}

#[inline]
pub(super) fn so4_theta_grad_f64(
    g_v_read: [f64; 4],
    v_read: [f64; 4],
    g_x1: [f64; 4],
    x1: [f64; 4],
) -> f64 {
    let jo = j_odd_f64(v_read);
    let je = j_even_f64(x1);
    (0..4).map(|k| g_v_read[k] * jo[k] + g_x1[k] * je[k]).sum()
}

/// Evaluates blended Catmull-Rom delay interpolation (`half_alpha in [0, 0.5]`) and returns
/// `(value, d_val/d_mu, d_val/d_half_alpha)`.
#[inline]
pub(super) fn blend_catmull_rom_with_derivs(
    s_m1: f64,
    s_0: f64,
    s_1: f64,
    s_2: f64,
    mu: f64,
    half_alpha: f64,
) -> (f64, f64, f64) {
    let t = mu.clamp(0.0, 1.0);
    let omt = 1.0 - t;
    let ha = half_alpha.clamp(0.0, 0.5);
    let y_lin = omt * s_0 + t * s_1;
    let y_curv = omt * (2.0 * s_0 - s_m1 - s_1) + t * (2.0 * s_1 - s_0 - s_2);
    let val = y_lin + ha * t * omt * y_curv;
    let d_ycurv = (s_1 - s_0) * 2.0 + (s_m1 - s_2) + (s_1 - s_0);
    let d_mu = (s_1 - s_0) + ha * ((omt - t) * y_curv + t * omt * d_ycurv);
    let d_ha = t * omt * y_curv;
    (val, d_mu, d_ha)
}

/// Evaluates `half_alpha_cubic` and its derivative wrt `g_eff`.
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
        let half_alpha = 0.5 * omt * raw;
        let d_g = if g_eff > 0.5 && g_eff < f64::from(TANK_MAX_G_LIN_F32) {
            -0.5 * omt * 4.0 * 0.9992 / (g_eff * g_eff)
        } else {
            0.0
        };
        (half_alpha, d_g)
    }
}

/// Evaluates `cubic_soft_saturator` and its derivatives wrt `(x, gain_scale)`.
#[inline]
pub(super) fn cubic_soft_sat_and_deriv(
    x: f64,
    gain_scale: f64,
    inv_sigma_23: f64,
) -> (f64, f64, f64) {
    let v_ceil = f64::from(TANK_MAX_V_CEIL_F32);
    let scaled = x * inv_sigma_23;
    let v = scaled.clamp(-1.0, 1.0);
    let cubic = v * (1.5 - 0.5 * v * v);
    let raw = gain_scale * cubic;
    if raw.abs() >= v_ceil {
        (raw.clamp(-v_ceil, v_ceil), 0.0, 0.0)
    } else {
        let dv = if scaled.abs() < 1.0 {
            gain_scale * 1.5 * (1.0 - v * v) * inv_sigma_23
        } else {
            0.0
        };
        (raw, dv, cubic)
    }
}

/// Evaluates one `TransportQuadF64` over stereo `(x_l, x_r)` with optional reverse BPTT across
/// all `38` quad parameters (`d_v_read_cb` zero).
pub fn eval_op_t_quad_stereo_and_adjoint(
    x_l: &[f64],
    x_r: &[f64],
    tq: &TransportQuadF64,
    sample_rate_hz: f64,
    adj: Option<OpTQuadAdjointSlices<'_>>,
) -> (Vec<f64>, Vec<f64>, OpTQuadAdjointGrads) {
    eval_op_t_quad_with_cb(x_l, x_r, tq, sample_rate_hz, adj, None)
}

pub(crate) fn eval_op_t_quad_with_cb(
    x_l: &[f64],
    x_r: &[f64],
    tq: &TransportQuadF64,
    sample_rate_hz: f64,
    adj: Option<OpTQuadAdjointSlices<'_>>,
    d_v_read_cb: Option<&[[f64; 4]]>,
) -> (Vec<f64>, Vec<f64>, OpTQuadAdjointGrads) {
    let n = x_l.len().min(x_r.len());
    if n == 0 {
        return (Vec::new(), Vec::new(), OpTQuadAdjointGrads::default());
    }
    let resolved = OpTResolvedParams::from_quad(tq, sample_rate_hz, n);
    let mut fwd = OpTQuadStepState::new(resolved, n, adj.is_some());
    let mut out_l = Vec::with_capacity(n);
    let mut out_r = Vec::with_capacity(n);
    for t in 0..n {
        let top = fwd.step_top_of_frame_read();
        let (yl, yr) = fwd.step_end_of_frame_write(&top, x_l[t], x_r[t]);
        out_l.push(yl);
        out_r.push(yr);
    }

    let Some(mut adj_s) = adj else {
        return (out_l, out_r, OpTQuadAdjointGrads::default());
    };
    let mut rev = OpTQuadReverseState::new(n);
    for t in (0..n).rev() {
        let (d_vr_end, d_rwin_freeze, gx_l, gx_r) = rev.reverse_end_of_frame_write(
            &fwd,
            t,
            x_l[t],
            x_r[t],
            adj_s.d_out_l[t],
            adj_s.d_out_r[t],
        );
        if let Some(ref mut gxl) = adj_s.grad_x_l {
            gxl[t] += gx_l;
        }
        if let Some(ref mut gxr) = adj_s.grad_x_r {
            gxr[t] += gx_r;
        }
        let d_vr_tot = if let Some(cb) = d_v_read_cb {
            core::array::from_fn(|k| d_vr_end[k] + cb[t][k])
        } else {
            d_vr_end
        };
        rev.reverse_top_of_frame_read(&fwd, t, d_vr_tot, d_rwin_freeze);
    }
    let grads = rev.finalize(&fwd);
    (out_l, out_r, grads)
}

/// Evaluates all `1..=active_quads` `TransportQuad`s in parallel (`Op-T` End-of-Frame summation)
/// with optional joint reverse BPTT into `[OpTQuadAdjointGrads; 4]` and `(grad_x_l, grad_x_r)`.
pub(crate) fn eval_op_t_bank_stereo_and_adjoint(
    x_l: &[f64],
    x_r: &[f64],
    quads: &[TransportQuadF64; 4],
    active_quads: usize,
    sample_rate_hz: f64,
    mut adj_opt: Option<OpTQuadAdjointSlices<'_>>,
) -> (Vec<f64>, Vec<f64>, [OpTQuadAdjointGrads; 4]) {
    let n = x_l.len().min(x_r.len());
    let mut out_l = vec![0.0_f64; n];
    let mut out_r = vec![0.0_f64; n];
    let mut all_grads = [OpTQuadAdjointGrads::default(); 4];
    for q in 0..active_quads.min(4) {
        let active = if adj_opt.is_some() {
            is_op_t_quad_active(&quads[q])
        } else {
            has_wet_op_t_output(&quads[q])
        };
        if !active {
            continue;
        }
        let adj = adj_opt.as_mut().map(|a| OpTQuadAdjointSlices {
            d_out_l: a.d_out_l,
            d_out_r: a.d_out_r,
            grad_x_l: a.grad_x_l.as_deref_mut().map(|s| &mut s[..n]),
            grad_x_r: a.grad_x_r.as_deref_mut().map(|s| &mut s[..n]),
        });
        let (ql, qr, qg) =
            eval_op_t_quad_stereo_and_adjoint(x_l, x_r, &quads[q], sample_rate_hz, adj);
        for t in 0..n {
            out_l[t] += ql[t];
            out_r[t] += qr[t];
        }
        all_grads[q] = qg;
    }
    (out_l, out_r, all_grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_fd(name: &str, ana: f64, fd: f64) {
        let scale = ana.abs().max(fd.abs()).max(1.0e-3);
        assert!(
            (ana - fd).abs() < 1.0e-4 * scale,
            "{name} FD mismatch: ana={ana} fd={fd}"
        );
    }

    #[test]
    fn test_op_t_quad_all_38_params_zoh_and_cb_fd_parity() {
        let n = 160_usize;
        let in_l: Vec<f64> = (0..n).map(|t| 0.45 * ((t as f64) * 0.13).sin()).collect();
        let in_r: Vec<f64> = (0..n).map(|t| -0.35 * ((t as f64) * 0.19).cos()).collect();
        let d_out_l: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.29).cos()).collect();
        let d_out_r: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.17).sin()).collect();
        let obj = |yl: &[f64], yr: &[f64]| -> f64 {
            yl.iter().zip(&d_out_l).map(|(a, b)| a * b).sum::<f64>()
                + yr.iter().zip(&d_out_r).map(|(a, b)| a * b).sum::<f64>()
        };

        let tq = TransportQuadF64 {
            delay_samples: [7.3, 9.7, 13.2, 17.6],
            ramp_spans_samples: [2.5, 4.1, 1.8, 3.2],
            g_loop: 0.72,
            r_dc: 0.985,
            damp_pole: 0.30,
            lfo_rate_hz: 3.5,
            ramp_rate_hz: 2.25,
            lfo_depths: [0.6, 0.45, 0.75, 0.35],
            input_gains: [0.55, 0.42, -0.38, 0.27],
            wet_tap_l: [0.65, 0.22, -0.45, 0.18],
            wet_tap_r: [0.14, -0.58, 0.26, 0.41],
            allpass_g: [0.35, -0.25, 0.40, -0.30],
            allpass_delays: [3, 5, 7, 11],
            lfo_phase_off_q16: [0, 16_384, 32_768, 49_152],
            ramp_phase_off_q16: [0, 32_768, 16_384, 49_152],
            fb_psi: 0.38,
            sigma_knee: 0.60,
            freeze_mix: 0.25,
            window_depth: 0.40,
            scat_psi: 0.32,
            ext_f32: false,
        };

        for &sr in &[48_000.0_f64, 96_000.0_f64] {
            let mut gx_l = vec![0.0_f64; n];
            let mut gx_r = vec![0.0_f64; n];
            let (_, _, g) = eval_op_t_quad_stereo_and_adjoint(
                &in_l,
                &in_r,
                &tq,
                sr,
                Some(OpTQuadAdjointSlices {
                    d_out_l: &d_out_l,
                    d_out_r: &d_out_r,
                    grad_x_l: Some(&mut gx_l),
                    grad_x_r: Some(&mut gx_r),
                }),
            );
            let eps = 1.0e-6_f64;
            let eval_c = |cand: &TransportQuadF64| {
                let (yl, yr, _) = eval_op_t_quad_stereo_and_adjoint(&in_l, &in_r, cand, sr, None);
                obj(&yl, &yr)
            };

            let check_param = |name: &str, ana: f64, edit: &dyn Fn(&mut TransportQuadF64, f64)| {
                let (mut cp, mut cm) = (tq, tq);
                edit(&mut cp, eps);
                edit(&mut cm, -eps);
                let fd = (eval_c(&cp) - eval_c(&cm)) / (2.0 * eps);
                assert_fd(&format!("sr={sr} {name}"), ana, fd);
            };

            for k in 0..4 {
                check_param("delay", g.d_delay_samples[k], &|c, d| {
                    c.delay_samples[k] += d
                });
                check_param("span", g.d_ramp_spans_samples[k], &|c, d| {
                    c.ramp_spans_samples[k] += d
                });
                check_param("lfo_d", g.d_lfo_depths[k], &|c, d| c.lfo_depths[k] += d);
                check_param("inj", g.d_input_gains[k], &|c, d| c.input_gains[k] += d);
                check_param("ap_g", g.d_allpass_g[k], &|c, d| c.allpass_g[k] += d);
                check_param("wet_l", g.d_wet_tap_l[k], &|c, d| c.wet_tap_l[k] += d);
                check_param("wet_r", g.d_wet_tap_r[k], &|c, d| c.wet_tap_r[k] += d);
            }
            check_param("g_loop", g.d_g_loop, &|c, d| c.g_loop += d);
            check_param("r_dc", g.d_r_dc, &|c, d| c.r_dc += d);
            check_param("damp_pole", g.d_damp_pole, &|c, d| c.damp_pole += d);
            check_param("lfo_rate", g.d_lfo_rate_hz, &|c, d| c.lfo_rate_hz += d);
            check_param("ramp_rate", g.d_ramp_rate_hz, &|c, d| c.ramp_rate_hz += d);
            check_param("fb_psi", g.d_fb_psi, &|c, d| c.fb_psi += d);
            check_param("sigma_knee", g.d_sigma_knee, &|c, d| c.sigma_knee += d);
            check_param("freeze_mix", g.d_freeze_mix, &|c, d| c.freeze_mix += d);
            check_param("window_depth", g.d_window_depth, &|c, d| {
                c.window_depth += d
            });
            check_param("scat_psi", g.d_scat_psi, &|c, d| c.scat_psi += d);

            let (mut xlp, mut xlm) = (in_l.clone(), in_l.clone());
            xlp[29] += eps;
            xlm[29] -= eps;
            let (pl, pr, _) = eval_op_t_quad_stereo_and_adjoint(&xlp, &in_r, &tq, sr, None);
            let (ml, mr, _) = eval_op_t_quad_stereo_and_adjoint(&xlm, &in_r, &tq, sr, None);
            assert_fd(
                &format!("sr={sr} x_l[29]"),
                gx_l[29],
                (obj(&pl, &pr) - obj(&ml, &mr)) / (2.0 * eps),
            );
        }

        // Verify multi-quad bank summation + multi-quad joint FD parity.
        let mut q2 = tq;
        q2.delay_samples = [19.5, 23.1, 29.4, 31.7];
        let mut qs = [TransportQuadF64::default(); 4];
        qs[0] = tq;
        qs[1] = q2;
        let mut bg_x_l = vec![0.0_f64; n];
        let mut bg_x_r = vec![0.0_f64; n];
        let (bl, br, bg) = eval_op_t_bank_stereo_and_adjoint(
            &in_l,
            &in_r,
            &qs,
            2,
            48_000.0,
            Some(OpTQuadAdjointSlices {
                d_out_l: &d_out_l,
                d_out_r: &d_out_r,
                grad_x_l: Some(&mut bg_x_l),
                grad_x_r: Some(&mut bg_x_r),
            }),
        );
        assert_eq!(bl.len(), n);
        assert_eq!(br.len(), n);
        let eps_b = 1.0e-6_f64;
        for qi in 0..2 {
            let (mut qsp, mut qsm) = (qs, qs);
            qsp[qi].delay_samples[0] += eps_b;
            qsm[qi].delay_samples[0] -= eps_b;
            let (pl, pr, _) =
                eval_op_t_bank_stereo_and_adjoint(&in_l, &in_r, &qsp, 2, 48_000.0, None);
            let (ml, mr, _) =
                eval_op_t_bank_stereo_and_adjoint(&in_l, &in_r, &qsm, 2, 48_000.0, None);
            let fd = (obj(&pl, &pr) - obj(&ml, &mr)) / (2.0 * eps_b);
            assert_fd(
                &format!("bank q={qi} delay[0]"),
                bg[qi].d_delay_samples[0],
                fd,
            );
        }
        assert!(
            bg[1].d_delay_samples.iter().map(|v| v.abs()).sum::<f64>() > 1.0e-3,
            "q2 delay grads unexpectedly tiny: {:?}",
            bg[1].d_delay_samples
        );

        // Dormant quad with g_loop > 0 but zero input_gains/wet_tap must skip forward work & output 0.
        let dormant = TransportQuadF64 {
            g_loop: 0.25,
            delay_samples: [29.0, 43.0, 59.0, 73.0],
            ..TransportQuadF64::default()
        };
        assert!(!has_wet_op_t_output(&dormant));
        assert!(is_op_t_quad_active(&dormant));
        let dormant_bank = [dormant; 4];
        let (dl, dr, _) =
            eval_op_t_bank_stereo_and_adjoint(&in_l, &in_r, &dormant_bank, 4, 48_000.0, None);
        assert!(dl.iter().chain(&dr).all(|&v| v == 0.0));
    }
}
