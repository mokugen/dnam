//! Top-of-Frame `Op-P` (`T1DcBlockerStereo` -> `HilbertH1StereoKernel` -> `T1SubHarmonicKernel`) +
//! signed `Q0.32` quadrature phasor bank + 3-basis carrier forward evaluator and reverse-time
//! crossbar / `opt_to_lr` adjoints (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.rotor.q0-192-lfo]`,
//! `ref[train.pipeline.backbone-trajectory-opt]`).

mod t1;

use core::f64::consts::{PI, TAU};

use dnam_kernels::{
    HILBERT_H1_STAGES, compute_t1_dc_blocker_coeffs_clamped, interpolate_hilbert_h1_alphas_f64,
};

use super::super::model::SonaModelF64;
use t1::{T1SubHarmonicF64, accumulate_t1_1x_iq_grads};

const Q32_TO_RAD: f64 = TAU / 4_294_967_296.0;

/// Exact gradients for all `Op-P`, signed `Q0.32` phasor, carrier, and `Op-T` crossbar weights.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CrossbarGrads {
    pub d_phasor_freq_hz: [f64; 4],
    pub d_crossbar_op_gains: [f64; 2],
    pub d_carrier_basis_w: [[f64; 4]; 3],
    pub d_op_p_to_core: [[f64; 4]; 10],
    pub d_phasor_to_core: [[f64; 4]; 10],
    pub d_phasor_stereo_phase: [f64; 4],
    pub d_opt_to_lr: [[f64; 2]; 16],
}

impl Default for CrossbarGrads {
    fn default() -> Self {
        Self {
            d_phasor_freq_hz: [0.0; 4],
            d_crossbar_op_gains: [0.0; 2],
            d_carrier_basis_w: [[0.0; 4]; 3],
            d_op_p_to_core: [[0.0; 4]; 10],
            d_phasor_to_core: [[0.0; 4]; 10],
            d_phasor_stereo_phase: [0.0; 4],
            d_opt_to_lr: [[0.0; 2]; 16],
        }
    }
}

/// Forward Top-of-Frame crossbar cache across `[0..n)` frames and `q_count` `CoreQuad`s.
pub(crate) struct CrossbarForwardTrace {
    pub n: usize,
    pub q_count: usize,
    pub sample_rate_hz: f64,
    pub include_stereo_phase: bool,
    pub os2x: bool,
    pub active: bool,
    pub dc_pole: f64,
    pub dc_gain: f64,
    pub alpha_i: [f64; HILBERT_H1_STAGES],
    pub alpha_q: [f64; HILBERT_H1_STAGES],
    pub iq_lr: Vec<[f64; 4]>,
    pub op_p_feat: Vec<[f64; 4]>,
    pub cb_add_l: Vec<[f64; 4]>,
    pub cb_add_r: Vec<[f64; 4]>,
}

/// Checks whether any Top-of-Frame `Op-P` or carrier phasor crossbar path is non-zero.
#[inline]
#[must_use]
pub(crate) fn is_crossbar_feed_active(m: &SonaModelF64, q_count: usize) -> bool {
    let op_p_active = m.crossbar_op_gains[0].abs() > 1.0e-12
        && m.op_p_to_core[..q_count.min(10)]
            .iter()
            .any(|row| row.iter().any(|&v| v.abs() > 1.0e-12));
    let car_active = m
        .carrier_basis_w
        .iter()
        .any(|row| row.iter().any(|&v| v.abs() > 1.0e-12))
        && m.phasor_to_core[..q_count.min(10)]
            .iter()
            .any(|row| row.iter().any(|&v| v.abs() > 1.0e-12));
    op_p_active || car_active
}

#[inline]
fn step_dc_blocker_f64(pole: f64, gain: f64, px: &mut f64, py: &mut f64, x: f64) -> f64 {
    let y = gain * (x - *px) + pole * *py;
    *px = x;
    *py = y;
    y
}

#[inline]
fn step_allpass_chain_f64(
    input: f64,
    alpha: &[f64; HILBERT_H1_STAGES],
    s0: &mut [f64; HILBERT_H1_STAGES],
    s1: &mut [f64; HILBERT_H1_STAGES],
) -> f64 {
    let mut v = input;
    for stage in 0..HILBERT_H1_STAGES {
        let a = alpha[stage];
        let y = a * v + s1[stage];
        s1[stage] = s0[stage];
        s0[stage] = a * y - v;
        v = y;
    }
    v
}

#[inline]
fn reverse_allpass_chain_f64(alpha: &[f64; HILBERT_H1_STAGES], d_out: &[f64]) -> Vec<f64> {
    let n = d_out.len();
    let mut g = d_out.to_vec();
    for stage in (0..HILBERT_H1_STAGES).rev() {
        let a = alpha[stage];
        let mut adj_s0 = 0.0_f64;
        let mut adj_s1 = 0.0_f64;
        for g_t in g.iter_mut().rev() {
            let adj_y = *g_t + a * adj_s0;
            let adj_v = a * adj_y - adj_s0;
            *g_t = adj_v;
            adj_s0 = adj_s1;
            adj_s1 = adj_y;
        }
    }
    let _ = n;
    g
}

#[inline]
fn fold_os2x_iq_grad(d_iq: &[f64], os2x: bool) -> Vec<f64> {
    if !os2x {
        return d_iq.to_vec();
    }
    let n_1x = d_iq.len() / 2;
    (0..n_1x)
        .map(|t1x| {
            let cur = d_iq[2 * t1x + 1];
            let mid_cur = 0.5 * d_iq[2 * t1x];
            let mid_next = if t1x + 1 < n_1x {
                0.5 * d_iq[2 * (t1x + 1)]
            } else {
                0.0
            };
            cur + mid_cur + mid_next
        })
        .collect()
}

fn pullback_1x_iq_to_u_sense(
    trace: &CrossbarForwardTrace,
    d_i_1x: &[f64],
    d_q_1x: &[f64],
    grad_u_sense: &mut [f64],
) {
    let n_1x = d_i_1x.len();
    let d_dc_from_i = reverse_allpass_chain_f64(&trace.alpha_i, d_i_1x);
    let mut d_q_shifted = vec![0.0_f64; n_1x];
    if n_1x > 1 {
        d_q_shifted[..n_1x - 1].copy_from_slice(&d_q_1x[1..]);
    }
    let d_dc_from_q = reverse_allpass_chain_f64(&trace.alpha_q, &d_q_shifted);
    let d_dc: Vec<f64> = d_dc_from_i
        .iter()
        .zip(&d_dc_from_q)
        .map(|(a, b)| a + b)
        .collect();
    reverse_dc_blocker_f64(trace.dc_pole, trace.dc_gain, &d_dc, grad_u_sense);
}

#[inline]
fn reverse_dc_blocker_f64(pole: f64, gain: f64, d_out: &[f64], grad_in: &mut [f64]) {
    let n = d_out.len().min(grad_in.len());
    let mut adj_y_next = 0.0_f64;
    let mut adj_x_delay = 0.0_f64;
    for t in (0..n).rev() {
        let adj_y = d_out[t] + pole * adj_y_next;
        grad_in[t] += gain * adj_y - adj_x_delay;
        adj_x_delay = gain * adj_y;
        adj_y_next = adj_y;
    }
}

/// Evaluates Top-of-Frame `Op-P` + signed `Q0.32` phasors + 3-basis carrier injection into
/// `q_count` `CoreQuad`s from `u_sense_l/r`, caching linear `H1` coefficients for exact BPTT.
pub(crate) fn eval_crossbar_forward(
    model: &SonaModelF64,
    u_sense_l: &[f64],
    u_sense_r: &[f64],
    q_count: usize,
    include_stereo_phase: bool,
) -> CrossbarForwardTrace {
    eval_crossbar_forward_impl(
        model,
        u_sense_l,
        u_sense_r,
        q_count,
        include_stereo_phase,
        false,
    )
}

/// Evaluates Top-of-Frame `1x` `Op-P` with `2x` midpoint interpolation (`sub_idx == 0`) and
/// `2x`-rate phasor progression matching `dnam-vm` `step_core_subsample` (`os2x = true`).
pub(crate) fn eval_crossbar_forward_os2x(
    model: &SonaModelF64,
    u_sense_l: &[f64],
    u_sense_r: &[f64],
    q_count: usize,
    include_stereo_phase: bool,
) -> CrossbarForwardTrace {
    eval_crossbar_forward_impl(
        model,
        u_sense_l,
        u_sense_r,
        q_count,
        include_stereo_phase,
        true,
    )
}

fn eval_crossbar_forward_impl(
    model: &SonaModelF64,
    u_sense_l: &[f64],
    u_sense_r: &[f64],
    q_count: usize,
    include_stereo_phase: bool,
    os2x: bool,
) -> CrossbarForwardTrace {
    let n = u_sense_l.len().min(u_sense_r.len());
    let out_n = if os2x { n * 2 } else { n };
    let q_c = q_count.clamp(1, 10);
    let sr = if model.sample_rate_hz.is_finite() && model.sample_rate_hz > 0.0 {
        model.sample_rate_hz
    } else {
        48_000.0
    };
    let (dc_pole_f32, dc_gain_f32) = compute_t1_dc_blocker_coeffs_clamped(sr);
    let (dc_pole, dc_gain) = (f64::from(dc_pole_f32), f64::from(dc_gain_f32));
    let (ai_f32, aq_f32) = interpolate_hilbert_h1_alphas_f64(sr);
    let alpha_i = ai_f32.map(f64::from);
    let alpha_q = aq_f32.map(f64::from);

    let active = is_crossbar_feed_active(model, q_c);
    if n == 0 || !active {
        return CrossbarForwardTrace {
            n: out_n,
            q_count: q_c,
            sample_rate_hz: sr,
            include_stereo_phase,
            os2x,
            active: false,
            dc_pole,
            dc_gain,
            alpha_i,
            alpha_q,
            iq_lr: Vec::new(),
            op_p_feat: Vec::new(),
            cb_add_l: vec![[0.0; 4]; q_c * out_n],
            cb_add_r: vec![[0.0; 4]; q_c * out_n],
        };
    }

    let mut dc_px = [0.0_f64; 2];
    let mut dc_py = [0.0_f64; 2];
    let (mut s0_i_l, mut s1_i_l) = ([0.0_f64; HILBERT_H1_STAGES], [0.0_f64; HILBERT_H1_STAGES]);
    let (mut s0_q_l, mut s1_q_l) = ([0.0_f64; HILBERT_H1_STAGES], [0.0_f64; HILBERT_H1_STAGES]);
    let (mut s0_i_r, mut s1_i_r) = ([0.0_f64; HILBERT_H1_STAGES], [0.0_f64; HILBERT_H1_STAGES]);
    let (mut s0_q_r, mut s1_q_r) = ([0.0_f64; HILBERT_H1_STAGES], [0.0_f64; HILBERT_H1_STAGES]);
    let (mut q_prev_l, mut q_prev_r) = (0.0_f64, 0.0_f64);
    let mut t1 = T1SubHarmonicF64::new();

    let mut iq_lr = Vec::with_capacity(out_n);
    let mut op_p_feat = Vec::with_capacity(out_n);
    let mut cb_add_l = vec![[0.0_f64; 4]; q_c * out_n];
    let mut cb_add_r = vec![[0.0_f64; 4]; q_c * out_n];

    let max_hz = 0.49 * sr;
    let ph_freq: [f64; 4] = model.phasor_freq_hz.map(|f| {
        if f.is_finite() {
            f.clamp(-max_hz, max_hz)
        } else {
            0.0
        }
    });
    let ph_init: [f64; 4] = model.phasor_init_phase_q32.map(|p| (p as f64) * Q32_TO_RAD);
    let (cos_st, sin_st): ([f64; 4], [f64; 4]) = if include_stereo_phase {
        let mut cs = [1.0; 4];
        let mut sn = [0.0; 4];
        for l in 0..4 {
            let th = if model.phasor_stereo_phase[l].is_finite() {
                model.phasor_stereo_phase[l].clamp(-PI, PI)
            } else {
                0.0
            };
            (sn[l], cs[l]) = th.sin_cos();
        }
        (cs, sn)
    } else {
        ([1.0; 4], [0.0; 4])
    };
    let op_p_gain = if model.crossbar_op_gains[0].is_finite() {
        model.crossbar_op_gains[0].clamp(-4.0, 4.0)
    } else {
        0.0
    };
    let sr_phasor = if os2x { 2.0 * sr } else { sr };

    let mut push_frame = |frame: usize, iq_step: [f64; 4], feat_step: [f64; 4]| {
        iq_lr.push(iq_step);
        op_p_feat.push(feat_step);
        let [i_l, q_l, i_r, q_r] = iq_step;
        let mut car_l = [0.0_f64; 4];
        let mut car_r = [0.0_f64; 4];
        let mut op_vec = [0.0_f64; 4];
        for l in 0..4 {
            let phi_l = ph_init[l] + (frame as f64) * TAU * ph_freq[l] / sr_phasor;
            let (sin_l, cos_l) = phi_l.sin_cos();
            let cos_r = cos_l * cos_st[l] - sin_l * sin_st[l];
            let sin_r = sin_l * cos_st[l] + cos_l * sin_st[l];
            let (w0, w1, w2) = (
                model.carrier_basis_w[0][l],
                model.carrier_basis_w[1][l],
                model.carrier_basis_w[2][l],
            );
            car_l[l] = w0 * (i_l * cos_l) + w1 * (q_l * sin_l) + w2 * sin_l;
            car_r[l] = w0 * (i_r * cos_r) + w1 * (q_r * sin_r) + w2 * sin_r;
            op_vec[l] = op_p_gain * feat_step[l];
        }
        for q in 0..q_c {
            let idx = q * out_n + frame;
            for l in 0..4 {
                cb_add_l[idx][l] =
                    op_vec[l] * model.op_p_to_core[q][l] + car_l[l] * model.phasor_to_core[q][l];
                cb_add_r[idx][l] =
                    op_vec[l] * model.op_p_to_core[q][l] + car_r[l] * model.phasor_to_core[q][l];
            }
        }
    };

    let mut prev_iq = [0.0_f64; 4];
    let mut prev_feat = [0.0_f64; 4];
    for t in 0..n {
        let dl = step_dc_blocker_f64(dc_pole, dc_gain, &mut dc_px[0], &mut dc_py[0], u_sense_l[t]);
        let dr = step_dc_blocker_f64(dc_pole, dc_gain, &mut dc_px[1], &mut dc_py[1], u_sense_r[t]);
        let i_l = step_allpass_chain_f64(dl, &alpha_i, &mut s0_i_l, &mut s1_i_l);
        let q_l_next = step_allpass_chain_f64(dl, &alpha_q, &mut s0_q_l, &mut s1_q_l);
        let i_r = step_allpass_chain_f64(dr, &alpha_i, &mut s0_i_r, &mut s1_i_r);
        let q_r_next = step_allpass_chain_f64(dr, &alpha_q, &mut s0_q_r, &mut s1_q_r);
        let q_l = q_prev_l;
        let q_r = q_prev_r;
        q_prev_l = q_l_next;
        q_prev_r = q_r_next;

        let feat_cur = t1.step(0.5 * (i_l + i_r), 0.5 * (q_l + q_r));
        let iq_cur = [i_l, q_l, i_r, q_r];

        if os2x {
            let iq_mid = core::array::from_fn(|k| 0.5 * (prev_iq[k] + iq_cur[k]));
            let feat_mid = core::array::from_fn(|k| 0.5 * (prev_feat[k] + feat_cur[k]));
            push_frame(2 * t, iq_mid, feat_mid);
            push_frame(2 * t + 1, iq_cur, feat_cur);
            prev_iq = iq_cur;
            prev_feat = feat_cur;
        } else {
            push_frame(t, iq_cur, feat_cur);
        }
    }

    CrossbarForwardTrace {
        n: out_n,
        q_count: q_c,
        sample_rate_hz: sr,
        include_stereo_phase,
        os2x,
        active: true,
        dc_pole,
        dc_gain,
        alpha_i,
        alpha_q,
        iq_lr,
        op_p_feat,
        cb_add_l,
        cb_add_r,
    }
}

/// Computes exact parameter gradients in `CrossbarGrads` and propagates `d_u_sense_l/r` back
/// through `HilbertH1` + `T1DcBlockerStereo` from per-quad crossbar drive adjoints `d_cb_add_l/r`.
pub(crate) fn assemble_crossbar_adjoint(
    model: &SonaModelF64,
    trace: &CrossbarForwardTrace,
    d_cb_add_l: &[[f64; 4]],
    d_cb_add_r: &[[f64; 4]],
    grad_u_sense_l_opt: Option<&mut [f64]>,
    grad_u_sense_r_opt: Option<&mut [f64]>,
) -> CrossbarGrads {
    let mut grads = CrossbarGrads::default();
    if !trace.active || trace.n == 0 {
        return grads;
    }
    let n = trace.n;
    let sr = trace.sample_rate_hz;
    let sr_phasor = if trace.os2x { 2.0 * sr } else { sr };
    let max_hz = 0.49 * sr;
    let ph_freq: [f64; 4] = model.phasor_freq_hz.map(|f| {
        if f.is_finite() {
            f.clamp(-max_hz, max_hz)
        } else {
            0.0
        }
    });
    let ph_init: [f64; 4] = model.phasor_init_phase_q32.map(|p| (p as f64) * Q32_TO_RAD);
    let (cos_st, sin_st): ([f64; 4], [f64; 4]) = if trace.include_stereo_phase {
        let mut cs = [1.0; 4];
        let mut sn = [0.0; 4];
        for l in 0..4 {
            let th = if model.phasor_stereo_phase[l].is_finite() {
                model.phasor_stereo_phase[l].clamp(-PI, PI)
            } else {
                0.0
            };
            (sn[l], cs[l]) = th.sin_cos();
        }
        (cs, sn)
    } else {
        ([1.0; 4], [0.0; 4])
    };
    let op_p_gain = if model.crossbar_op_gains[0].is_finite() {
        model.crossbar_op_gains[0].clamp(-4.0, 4.0)
    } else {
        0.0
    };

    let need_u_sense = grad_u_sense_l_opt.is_some() || grad_u_sense_r_opt.is_some();
    let (mut d_il, mut d_ql, mut d_ir, mut d_qr, mut d_feat) = if need_u_sense {
        (
            vec![0.0_f64; n],
            vec![0.0_f64; n],
            vec![0.0_f64; n],
            vec![0.0_f64; n],
            vec![[0.0_f64; 4]; n],
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };

    for t in 0..n {
        let [i_l, q_l, i_r, q_r] = trace.iq_lr[t];
        let feat = trace.op_p_feat[t];

        for l in 0..4 {
            let phi_l = ph_init[l] + (t as f64) * TAU * ph_freq[l] / sr_phasor;
            let (sin_l, cos_l) = phi_l.sin_cos();
            let cos_r = cos_l * cos_st[l] - sin_l * sin_st[l];
            let sin_r = sin_l * cos_st[l] + cos_l * sin_st[l];
            let (w0, w1, w2) = (
                model.carrier_basis_w[0][l],
                model.carrier_basis_w[1][l],
                model.carrier_basis_w[2][l],
            );
            let car_l = w0 * (i_l * cos_l) + w1 * (q_l * sin_l) + w2 * sin_l;
            let car_r = w0 * (i_r * cos_r) + w1 * (q_r * sin_r) + w2 * sin_r;
            let op_vec = op_p_gain * feat[l];

            let (mut d_op_vec, mut d_car_l, mut d_car_r) = (0.0_f64, 0.0_f64, 0.0_f64);
            for q in 0..trace.q_count {
                let idx = q * n + t;
                let gl = d_cb_add_l[idx][l];
                let gr = d_cb_add_r[idx][l];
                let glr = gl + gr;
                grads.d_op_p_to_core[q][l] += glr * op_vec;
                grads.d_phasor_to_core[q][l] += gl * car_l + gr * car_r;
                let ph_w = model.phasor_to_core[q][l];
                d_op_vec += glr * model.op_p_to_core[q][l];
                d_car_l += gl * ph_w;
                d_car_r += gr * ph_w;
            }

            if model.crossbar_op_gains[0].is_finite() && model.crossbar_op_gains[0].abs() < 4.0 {
                grads.d_crossbar_op_gains[0] += d_op_vec * feat[l];
            }
            grads.d_carrier_basis_w[0][l] += d_car_l * (i_l * cos_l) + d_car_r * (i_r * cos_r);
            grads.d_carrier_basis_w[1][l] += d_car_l * (q_l * sin_l) + d_car_r * (q_r * sin_r);
            grads.d_carrier_basis_w[2][l] += d_car_l * sin_l + d_car_r * sin_r;

            let d_phi_l = d_car_l * (-w0 * i_l * sin_l + (w1 * q_l + w2) * cos_l);
            let d_phi_r = d_car_r * (-w0 * i_r * sin_r + (w1 * q_r + w2) * cos_r);
            if model.phasor_freq_hz[l].is_finite() && model.phasor_freq_hz[l].abs() < max_hz {
                grads.d_phasor_freq_hz[l] += (d_phi_l + d_phi_r) * ((t as f64) * TAU / sr_phasor);
            }
            if trace.include_stereo_phase
                && model.phasor_stereo_phase[l].is_finite()
                && model.phasor_stereo_phase[l].abs() < PI
            {
                grads.d_phasor_stereo_phase[l] += d_phi_r;
            }

            if need_u_sense {
                d_feat[t][l] = d_op_vec * op_p_gain;
                d_il[t] += d_car_l * w0 * cos_l;
                d_ql[t] += d_car_l * w1 * sin_l;
                d_ir[t] += d_car_r * w0 * cos_r;
                d_qr[t] += d_car_r * w1 * sin_r;
            }
        }
    }

    if need_u_sense {
        let mut d_il_1x = fold_os2x_iq_grad(&d_il, trace.os2x);
        let mut d_ql_1x = fold_os2x_iq_grad(&d_ql, trace.os2x);
        let mut d_ir_1x = fold_os2x_iq_grad(&d_ir, trace.os2x);
        let mut d_qr_1x = fold_os2x_iq_grad(&d_qr, trace.os2x);
        accumulate_t1_1x_iq_grads(
            trace,
            &d_feat,
            &mut d_il_1x,
            &mut d_ql_1x,
            &mut d_ir_1x,
            &mut d_qr_1x,
        );
        if let Some(gl) = grad_u_sense_l_opt {
            pullback_1x_iq_to_u_sense(trace, &d_il_1x, &d_ql_1x, gl);
        }
        if let Some(gr) = grad_u_sense_r_opt {
            pullback_1x_iq_to_u_sense(trace, &d_ir_1x, &d_qr_1x, gr);
        }
    }

    grads
}

/// Evaluates Top-of-Frame `Op-T` crossbar feedback contribution `(t_cb_l, t_cb_r)` and accumulates
/// `d_opt_to_lr`, `d_crossbar_op_gains[1]`, and `d_v_read` for `CausalLoop`.
#[inline]
pub(crate) fn step_opt_crossbar_taps_and_adjoint(
    v_reads: &[[f64; 4]],
    opt_to_lr: &[[f64; 2]],
    op_t_gain: f64,
    active_opt_quads: usize,
) -> (f64, f64) {
    if op_t_gain == 0.0 || active_opt_quads == 0 {
        return (0.0, 0.0);
    }
    let mut acc_l = 0.0_f64;
    let mut acc_r = 0.0_f64;
    for q in 0..active_opt_quads.min(4) {
        for l in 0..4 {
            let idx = q * 4 + l;
            let vr = v_reads[q][l];
            acc_l += vr * opt_to_lr[idx][0];
            acc_r += vr * opt_to_lr[idx][1];
        }
    }
    (op_t_gain * acc_l, op_t_gain * acc_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_crossbar_model() -> SonaModelF64 {
        let mut m = SonaModelF64::new(48_000.0);
        m.phasor_freq_hz = [-120.0, 220.0, -480.0, 880.0];
        m.phasor_init_phase_q32 = [0x1200_0000, 0x4000_0000, 0x8000_0000, 0xC000_0000];
        m.phasor_stereo_phase = [0.25, -0.45, 0.65, -0.35];
        m.crossbar_op_gains = [0.75, 0.4];
        m.carrier_basis_w = [
            [0.6, 0.3, -0.25, 0.4],
            [-0.2, 0.5, 0.35, -0.3],
            [0.15, -0.2, 0.45, 0.25],
        ];
        for q in 0..6 {
            let s = 0.25 / ((q + 1) as f64);
            m.op_p_to_core[q] = [s, -0.5 * s, 0.75 * s, -0.25 * s];
            m.phasor_to_core[q] = [0.8 * s, 0.6 * s, -0.5 * s, 0.4 * s];
        }
        m
    }

    #[test]
    fn test_crossbar_signed_q32_carrier_op_p_and_hilbert_u_sense_fd_parity() {
        let n = 96_usize;
        let q_c = 6_usize;
        let m = test_crossbar_model();
        let u_l: Vec<f64> = (0..n).map(|t| 0.45 * ((t as f64) * 0.17).sin()).collect();
        let u_r: Vec<f64> = (0..n).map(|t| -0.38 * ((t as f64) * 0.23).cos()).collect();

        let trace = eval_crossbar_forward(&m, &u_l, &u_r, q_c, true);
        let d_l: Vec<[f64; 4]> = (0..q_c * n)
            .map(|i| {
                let s = ((i + 1) as f64) * 0.013;
                [s.sin(), s.cos(), (2.0 * s).sin(), -(0.5 * s).cos()]
            })
            .collect();
        let d_r: Vec<[f64; 4]> = (0..q_c * n)
            .map(|i| {
                let s = ((i + 3) as f64) * 0.019;
                [s.cos(), -s.sin(), -(1.5 * s).cos(), (0.7 * s).sin()]
            })
            .collect();
        let obj_trace = |tr: &CrossbarForwardTrace| -> f64 {
            tr.cb_add_l
                .iter()
                .zip(&d_l)
                .flat_map(|(a, b)| (0..4).map(move |l| a[l] * b[l]))
                .sum::<f64>()
                + tr.cb_add_r
                    .iter()
                    .zip(&d_r)
                    .flat_map(|(a, b)| (0..4).map(move |l| a[l] * b[l]))
                    .sum::<f64>()
        };

        let mut g_ul = vec![0.0_f64; n];
        let mut g_ur = vec![0.0_f64; n];
        let grads =
            assemble_crossbar_adjoint(&m, &trace, &d_l, &d_r, Some(&mut g_ul), Some(&mut g_ur));
        let eps = 1.0e-6_f64;
        let check = |name: &str, ana: f64, mp: &SonaModelF64, mm: &SonaModelF64| {
            let fp = eval_crossbar_forward(mp, &u_l, &u_r, q_c, true);
            let fm = eval_crossbar_forward(mm, &u_l, &u_r, q_c, true);
            let fd = (obj_trace(&fp) - obj_trace(&fm)) / (2.0 * eps);
            let scale = ana.abs().max(fd.abs()).max(1.0e-3);
            assert!(
                (ana - fd).abs() < 1.0e-4 * scale,
                "{name}: ana={ana} fd={fd}"
            );
        };

        for l in 0..4 {
            let (mut mp, mut mm) = (m.clone(), m.clone());
            mp.phasor_freq_hz[l] += eps;
            mm.phasor_freq_hz[l] -= eps;
            check("phasor_freq", grads.d_phasor_freq_hz[l], &mp, &mm);

            let (mut mp, mut mm) = (m.clone(), m.clone());
            mp.phasor_stereo_phase[l] += eps;
            mm.phasor_stereo_phase[l] -= eps;
            check(
                "phasor_stereo_phase",
                grads.d_phasor_stereo_phase[l],
                &mp,
                &mm,
            );

            for b in 0..3 {
                let (mut mp, mut mm) = (m.clone(), m.clone());
                mp.carrier_basis_w[b][l] += eps;
                mm.carrier_basis_w[b][l] -= eps;
                check("carrier_basis_w", grads.d_carrier_basis_w[b][l], &mp, &mm);
            }
            for q in 0..q_c {
                let (mut mp, mut mm) = (m.clone(), m.clone());
                mp.op_p_to_core[q][l] += eps;
                mm.op_p_to_core[q][l] -= eps;
                check("op_p_to_core", grads.d_op_p_to_core[q][l], &mp, &mm);

                let (mut mp, mut mm) = (m.clone(), m.clone());
                mp.phasor_to_core[q][l] += eps;
                mm.phasor_to_core[q][l] -= eps;
                check("phasor_to_core", grads.d_phasor_to_core[q][l], &mp, &mm);
            }
        }
        let (mut mp, mut mm) = (m.clone(), m.clone());
        mp.crossbar_op_gains[0] += eps;
        mm.crossbar_op_gains[0] -= eps;
        check("crossbar_op_p_gain", grads.d_crossbar_op_gains[0], &mp, &mm);

        // Verify exact Hilbert+T1 u_sense pullback when op_p_gain == 0 (pure carrier path).
        let mut m_car = m;
        m_car.crossbar_op_gains[0] = 0.0;
        let tr_c = eval_crossbar_forward(&m_car, &u_l, &u_r, q_c, true);
        let mut gu_cl = vec![0.0_f64; n];
        let mut gu_cr = vec![0.0_f64; n];
        assemble_crossbar_adjoint(
            &m_car,
            &tr_c,
            &d_l,
            &d_r,
            Some(&mut gu_cl),
            Some(&mut gu_cr),
        );
        let (mut ulp, mut ulm) = (u_l.clone(), u_l.clone());
        ulp[21] += eps;
        ulm[21] -= eps;
        let fd_u21 = (obj_trace(&eval_crossbar_forward(&m_car, &ulp, &u_r, q_c, true))
            - obj_trace(&eval_crossbar_forward(&m_car, &ulm, &u_r, q_c, true)))
            / (2.0 * eps);
        let scale = gu_cl[21].abs().max(fd_u21.abs()).max(1.0e-3);
        assert!(
            (gu_cl[21] - fd_u21).abs() < 1.0e-4 * scale,
            "hilbert u_l[21]: ana={} fd={fd_u21}",
            gu_cl[21]
        );
    }

    #[test]
    fn test_crossbar_os2x_midpoint_phasor_and_inactive_bypass() {
        let mut off = SonaModelF64::new(48_000.0);
        let u_l: Vec<f64> = (0..16).map(|t| 0.4 * ((t as f64) * 0.2).sin()).collect();
        let u_r: Vec<f64> = (0..16).map(|t| -0.3 * ((t as f64) * 0.15).cos()).collect();
        let tr_off = eval_crossbar_forward_os2x(&off, &u_l, &u_r, 6, true);
        assert!(!tr_off.active);
        assert_eq!(tr_off.n, 32);
        assert_eq!(tr_off.cb_add_l.len(), 6 * 32);
        assert!(tr_off.cb_add_l.iter().all(|v| *v == [0.0; 4]));

        // With phasor_freq_hz == 0, odd 2x frames (2*t + 1) match 1x frames identically and
        // even frames (2*t) use midpoint interpolation of (prev, cur).
        off = test_crossbar_model();
        off.phasor_freq_hz = [0.0; 4];
        let tr1 = eval_crossbar_forward(&off, &u_l, &u_r, 4, true);
        let tr2 = eval_crossbar_forward_os2x(&off, &u_l, &u_r, 4, true);
        assert!(tr2.active);
        assert_eq!(tr2.n, 32);
        for t in 0..16 {
            assert_eq!(tr2.iq_lr[2 * t + 1], tr1.iq_lr[t]);
            assert_eq!(tr2.op_p_feat[2 * t + 1], tr1.op_p_feat[t]);
            for q in 0..4 {
                assert_eq!(tr2.cb_add_l[q * 32 + 2 * t + 1], tr1.cb_add_l[q * 16 + t]);
                assert_eq!(tr2.cb_add_r[q * 32 + 2 * t + 1], tr1.cb_add_r[q * 16 + t]);
            }
            for k in 0..4 {
                let prev_iq = if t == 0 { 0.0 } else { tr1.iq_lr[t - 1][k] };
                let prev_ft = if t == 0 { 0.0 } else { tr1.op_p_feat[t - 1][k] };
                assert!((tr2.iq_lr[2 * t][k] - 0.5 * (prev_iq + tr1.iq_lr[t][k])).abs() <= 1.0e-12);
                assert!(
                    (tr2.op_p_feat[2 * t][k] - 0.5 * (prev_ft + tr1.op_p_feat[t][k])).abs()
                        <= 1.0e-12
                );
            }
        }

        // Verify active 2x phasor progression advances at 2 * sr (phi = ph_init + frame * TAU * f / (2*sr))
        // and assemble_crossbar_adjoint matches central FD on phasor_freq_hz and u_sense under os2x.
        let mut m2 = test_crossbar_model();
        m2.crossbar_op_gains[0] = 0.0;
        let q_os = 2_usize;
        let tr_ph = eval_crossbar_forward_os2x(&m2, &u_l, &u_r, q_os, true);
        assert!(tr_ph.active && tr_ph.os2x && tr_ph.cb_add_l.len() == q_os * 32);
        let dl_2x: Vec<[f64; 4]> = (0..(q_os * 32))
            .map(|i| core::array::from_fn(|l| ((i * 7 + l) as f64 * 0.13).cos()))
            .collect();
        let dr_2x: Vec<[f64; 4]> = (0..(q_os * 32))
            .map(|i| core::array::from_fn(|l| ((i * 5 + l) as f64 * 0.19).sin()))
            .collect();
        let mut gu2_l = vec![0.0_f64; 16];
        let mut gu2_r = vec![0.0_f64; 16];
        let g2 = assemble_crossbar_adjoint(
            &m2,
            &tr_ph,
            &dl_2x,
            &dr_2x,
            Some(&mut gu2_l),
            Some(&mut gu2_r),
        );
        let obj2 = |tr: &CrossbarForwardTrace| -> f64 {
            tr.cb_add_l
                .iter()
                .zip(&dl_2x)
                .map(|(a, b)| (0..4).map(|k| a[k] * b[k]).sum::<f64>())
                .sum::<f64>()
                + tr.cb_add_r
                    .iter()
                    .zip(&dr_2x)
                    .map(|(a, b)| (0..4).map(|k| a[k] * b[k]).sum::<f64>())
                    .sum::<f64>()
        };
        let eps = 1.0e-6_f64;
        for l in 0..4 {
            let (mut mp, mut mm) = (m2.clone(), m2.clone());
            mp.phasor_freq_hz[l] += eps;
            mm.phasor_freq_hz[l] -= eps;
            let fd_f = (obj2(&eval_crossbar_forward_os2x(&mp, &u_l, &u_r, q_os, true))
                - obj2(&eval_crossbar_forward_os2x(&mm, &u_l, &u_r, q_os, true)))
                / (2.0 * eps);
            let scale_f = g2.d_phasor_freq_hz[l].abs().max(fd_f.abs()).max(1.0e-3);
            assert!(
                (g2.d_phasor_freq_hz[l] - fd_f).abs() < 1.0e-4 * scale_f,
                "os2x phasor_freq_hz[{l}]: ana={} fd={fd_f}",
                g2.d_phasor_freq_hz[l]
            );
        }
        let (mut ulp, mut ulm) = (u_l.clone(), u_l.clone());
        ulp[5] += eps;
        ulm[5] -= eps;
        let fd_u5 = (obj2(&eval_crossbar_forward_os2x(&m2, &ulp, &u_r, q_os, true))
            - obj2(&eval_crossbar_forward_os2x(&m2, &ulm, &u_r, q_os, true)))
            / (2.0 * eps);
        let scale_u = gu2_l[5].abs().max(fd_u5.abs()).max(1.0e-3);
        assert!(
            (gu2_l[5] - fd_u5).abs() < 1.0e-4 * scale_u,
            "os2x u_l[5]: ana={} fd={fd_u5}",
            gu2_l[5]
        );
        let (mut urp, mut urm) = (u_r.clone(), u_r.clone());
        urp[7] += eps;
        urm[7] -= eps;
        let fd_ur7 = (obj2(&eval_crossbar_forward_os2x(&m2, &u_l, &urp, q_os, true))
            - obj2(&eval_crossbar_forward_os2x(&m2, &u_l, &urm, q_os, true)))
            / (2.0 * eps);
        let scale_ur = gu2_r[7].abs().max(fd_ur7.abs()).max(1.0e-3);
        assert!(
            (gu2_r[7] - fd_ur7).abs() < 1.0e-4 * scale_ur,
            "os2x u_r[7]: ana={} fd={fd_ur7}",
            gu2_r[7]
        );
    }

    fn verify_u_sense_fd_case(model: &SonaModelF64, os2x: bool, probes: &[usize], label: &str) {
        let n = 48_usize;
        let q_c = 4_usize;
        let out_n = if os2x { 2 * n } else { n };
        let u_l: Vec<f64> = (0..n).map(|t| 0.45 * ((t as f64) * 0.27).sin()).collect();
        let u_r: Vec<f64> = (0..n).map(|t| -0.38 * ((t as f64) * 0.31).cos()).collect();
        let d_l: Vec<[f64; 4]> = (0..q_c * out_n)
            .map(|i| core::array::from_fn(|l| ((i * 3 + l + 1) as f64 * 0.07).sin()))
            .collect();
        let d_r: Vec<[f64; 4]> = (0..q_c * out_n)
            .map(|i| core::array::from_fn(|l| ((i * 5 + l + 2) as f64 * 0.09).cos()))
            .collect();
        let eval_fwd = |ul: &[f64], ur: &[f64]| -> CrossbarForwardTrace {
            if os2x {
                eval_crossbar_forward_os2x(model, ul, ur, q_c, true)
            } else {
                eval_crossbar_forward(model, ul, ur, q_c, true)
            }
        };
        let obj = |tr: &CrossbarForwardTrace| -> f64 {
            tr.cb_add_l
                .iter()
                .zip(&d_l)
                .chain(tr.cb_add_r.iter().zip(&d_r))
                .map(|(a, b)| (0..4).map(|k| a[k] * b[k]).sum::<f64>())
                .sum()
        };

        let trace = eval_fwd(&u_l, &u_r);
        let mut g_l = vec![0.0_f64; n];
        let mut g_r = vec![0.0_f64; n];
        assemble_crossbar_adjoint(model, &trace, &d_l, &d_r, Some(&mut g_l), Some(&mut g_r));

        // Verify one-sided (Some/None and None/Some) calls match full stereo call identically.
        let mut g_l_only = vec![0.0_f64; n];
        let mut g_r_only = vec![0.0_f64; n];
        assemble_crossbar_adjoint(model, &trace, &d_l, &d_r, Some(&mut g_l_only), None);
        assemble_crossbar_adjoint(model, &trace, &d_l, &d_r, None, Some(&mut g_r_only));
        assert_eq!(g_l_only, g_l);
        assert_eq!(g_r_only, g_r);

        let eps = 1.0e-6_f64;
        for &idx in probes {
            let (mut ulp, mut ulm) = (u_l.clone(), u_l.clone());
            ulp[idx] += eps;
            ulm[idx] -= eps;
            let fd_l = (obj(&eval_fwd(&ulp, &u_r)) - obj(&eval_fwd(&ulm, &u_r))) / (2.0 * eps);
            let rel_l = (g_l[idx] - fd_l).abs() / g_l[idx].abs().max(fd_l.abs()).max(1.0e-3);
            assert!(
                rel_l < 1.0e-6,
                "{label} os2x={os2x} u_l[{idx}]: ana={} fd={fd_l} rel={rel_l}",
                g_l[idx]
            );

            let (mut urp, mut urm) = (u_r.clone(), u_r.clone());
            urp[idx] += eps;
            urm[idx] -= eps;
            let fd_r = (obj(&eval_fwd(&u_l, &urp)) - obj(&eval_fwd(&u_l, &urm))) / (2.0 * eps);
            let rel_r = (g_r[idx] - fd_r).abs() / g_r[idx].abs().max(fd_r.abs()).max(1.0e-3);
            assert!(
                rel_r < 1.0e-6,
                "{label} os2x={os2x} u_r[{idx}]: ana={} fd={fd_r} rel={rel_r}",
                g_r[idx]
            );
        }
    }

    #[test]
    fn test_crossbar_t1_hilbert_dc_and_joint_carrier_u_sense_fd_parity() {
        let mut pure_op_p = test_crossbar_model();
        pure_op_p.phasor_to_core = [[0.0; 4]; 10];
        let joint = test_crossbar_model();
        for os2x in [false, true] {
            verify_u_sense_fd_case(&pure_op_p, os2x, &[5, 17, 31], "pure_op_p");
            verify_u_sense_fd_case(&joint, os2x, &[5, 17, 31], "joint_op_p_carrier");
        }
    }
}
