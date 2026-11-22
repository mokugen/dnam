//! Pass-1 universal black-box physics & linear-modal identification probe
//! (`< 0.5 ms` CPU `Op-P` quadrature probe, iterative Steiglitz–McBride `(B, A)` +
//! Durand–Kerner pole solver, stereo Cauchy/Szegő Kautz OMP `8/12/24/32`, regularized FIR
//! `32/48/96/128`, stereo comb `Op-T` ID, and zero-init `Op-N[Pre]`,
//! `ref[train.pipeline.physics-probe]`).

use core::f64::consts::{PI, TAU};

use dnam_kernels::{HilbertH1StereoKernel, PhasorQ32Bank4, T1DcBlockerStereo, T1SubHarmonicKernel};

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;
use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_adaptive_ridge;

use super::model::{KautzPairF64, NUM_FIR_TAPS_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, SonaModelF64};

/// Summary returned by the `< 0.5 ms` CPU `Op-P` quadrature & pitch probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpPProbeSummary {
    /// Detected narrowband dominant carrier frequency in Hz (`0.0` if broadband).
    pub dominant_freq_hz: f64,
    /// Normalized pitch periodicity confidence `[0.0, 1.0]`.
    pub pitch_confidence: f64,
    /// Mean stereo quadrature analytic energy `mean(I^2 + Q^2)`.
    pub analytic_energy: f64,
    /// Detected sub-harmonic octaver activity ratio `[0.0, 1.0]`.
    pub subharmonic_activity: f64,
}

/// Complete Pass-1 initialization diagnostic report.
#[derive(Clone, Debug, PartialEq)]
pub struct Pass1ProbeReport {
    /// CPU `Op-P` quadrature summary on dry stereo excitation.
    pub op_p: OpPProbeSummary,
    /// Count of recovered complex conjugate Kautz modal pairs (`<= 32`).
    pub recovered_kautz_pairs: usize,
    /// Dominant `Op-T` feedback lag detected in samples (`>= 5.0`, or `0.0` if non-resonant).
    pub dominant_transport_delay_samples: f64,
    /// Estimated `Op-T` feedback loop gain `[0.0, 0.99]`.
    pub estimated_loop_gain: f64,
    /// Initial normalized stereo time-domain residual energy ratio vs target.
    pub initial_esr_lin: f64,
}

/// Executes the `< 0.5 ms` CPU `Op-P` probe across up to `4,096` dry stereo samples using
/// `dnam-kernels` `T1DcBlockerStereo`, `HilbertH1StereoKernel`, `T1SubHarmonicKernel`, and
/// `PhasorQ32Bank4` (`ref[train.pipeline.physics-probe]`).
#[must_use]
pub fn run_cpu_op_p_probe(
    input_l: &[f64],
    input_r: &[f64],
    sample_rate_hz: f64,
) -> OpPProbeSummary {
    let n = input_l.len().min(input_r.len()).min(4_096);
    if n < 32 {
        return OpPProbeSummary {
            dominant_freq_hz: 0.0,
            pitch_confidence: 0.0,
            analytic_energy: 0.0,
            subharmonic_activity: 0.0,
        };
    }
    let mut dc = T1DcBlockerStereo::new();
    let mut hilb = HilbertH1StereoKernel::new();
    let mut t1 = T1SubHarmonicKernel::new();
    let mut phasor = PhasorQ32Bank4::from_freqs_hz(
        [110.0, 220.0, 440.0, 880.0],
        1.0 / (sample_rate_hz.max(8_000.0) as f32),
    );

    let mut e_analytic = 0.0_f64;
    let mut e_sub = 0.0_f64;
    let mut dphase_cos_acc = 0.0_f64;
    let mut dphase_sin_acc = 0.0_f64;
    let mut prev_re = 1.0_f64;
    let mut prev_im = 0.0_f64;

    for i in 0..n {
        let (cl, cr) = dc.step_stereo(input_l[i] as f32, input_r[i] as f32);
        let ((il, ql), (ir, qr)) = hilb.step_stereo(cl, cr);
        let im = 0.5 * (il + ir);
        let qm = 0.5 * (ql + qr);
        let feat = t1.step(im, qm);
        let (p_cos, _p_sin) = phasor.step_quadrature();
        let _carrier_probe = f64::from(p_cos[0]);

        let il64 = f64::from(il);
        let ql64 = f64::from(ql);
        let ir64 = f64::from(ir);
        let qr64 = f64::from(qr);
        e_analytic += 0.5 * (il64 * il64 + ql64 * ql64 + ir64 * ir64 + qr64 * qr64);
        e_sub += f64::from(feat[0] * feat[0] + feat[1] * feat[1]);

        let im64 = f64::from(im);
        let qm64 = f64::from(qm);
        let mid_norm = (im64 * im64 + qm64 * qm64).sqrt().max(1.0e-12);
        let cur_re = im64 / mid_norm;
        let cur_im = qm64 / mid_norm;
        if i > 0 {
            dphase_cos_acc += cur_re * prev_re + cur_im * prev_im;
            dphase_sin_acc += cur_im * prev_re - cur_re * prev_im;
        }
        prev_re = cur_re;
        prev_im = cur_im;
    }
    let mean_an = e_analytic / (n as f64);
    let mean_sub = e_sub / (n as f64);
    let r_vec = (dphase_cos_acc * dphase_cos_acc + dphase_sin_acc * dphase_sin_acc).sqrt()
        / ((n - 1) as f64);
    let pitch_confidence = r_vec.clamp(0.0, 1.0);
    let omega = dphase_sin_acc.atan2(dphase_cos_acc).abs();
    let dominant_freq_hz = if pitch_confidence > 0.35 {
        omega * sample_rate_hz / TAU
    } else {
        0.0
    };
    OpPProbeSummary {
        dominant_freq_hz,
        pitch_confidence,
        analytic_energy: mean_an,
        subharmonic_activity: (mean_sub / mean_an.max(1.0e-12)).clamp(0.0, 1.0),
    }
}

fn solve_yule_walker_ar(sig: &[f64], order: usize) -> Vec<f64> {
    let n = sig.len();
    if order == 0 || n <= order + 4 {
        return Vec::new();
    }
    let packed = order * (order + 1) / 2;
    let mut g_u = vec![0.0_f64; packed];
    let mut rhs = vec![0.0_f64; order];
    for c in 0..order {
        let col_off = c * (c + 1) / 2;
        for r in 0..=c {
            let mut dot = 0.0_f64;
            for t in order..n {
                dot += sig[t - r - 1] * sig[t - c - 1];
            }
            if r == c {
                dot += 1.0e-6;
            }
            g_u[col_off + r] = dot;
        }
        let mut rd = 0.0_f64;
        for t in order..n {
            rd += -sig[t] * sig[t - c - 1];
        }
        rhs[c] = rd;
    }
    solve_uplo_adaptive_ridge(order, &g_u, &rhs, &[], None, 1.0e-6)
        .map(|(v, _)| v)
        .unwrap_or_else(|_| vec![0.0; order])
}

fn filter_all_pole_inverse(x: &[f64], a: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; x.len()];
    for t in 0..x.len() {
        let mut acc = x[t];
        for (k, &ak) in a.iter().enumerate() {
            if t > k {
                acc -= ak * y[t - k - 1];
            }
        }
        y[t] = acc;
    }
    y
}

#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cdiv(num: (f64, f64), den: (f64, f64)) -> (f64, f64) {
    let d2 = (den.0 * den.0 + den.1 * den.1).max(1.0e-30);
    (
        (num.0 * den.0 + num.1 * den.1) / d2,
        (num.1 * den.0 - num.0 * den.1) / d2,
    )
}

fn find_monic_roots(a: &[f64]) -> Vec<(f64, f64)> {
    let m = a.len();
    let mut roots: Vec<(f64, f64)> = (0..m)
        .map(|k| {
            let ang = TAU * ((k as f64) + 0.37) / (m as f64);
            (0.85 * ang.cos(), 0.85 * ang.sin())
        })
        .collect();
    for _ in 0..56 {
        let mut max_step = 0.0_f64;
        for i in 0..m {
            let zi = roots[i];
            let mut pval = (1.0_f64, 0.0_f64);
            for &ak in a {
                pval = cmul(pval, zi);
                pval.0 += ak;
            }
            let mut prod = (1.0_f64, 0.0_f64);
            for (j, &zj) in roots.iter().enumerate() {
                if i != j {
                    prod = cmul(prod, (zi.0 - zj.0, zi.1 - zj.1));
                }
            }
            let delta = cdiv(pval, prod);
            roots[i] = (zi.0 - delta.0, zi.1 - delta.1);
            max_step = max_step.max(delta.0 * delta.0 + delta.1 * delta.1);
        }
        if max_step < 1.0e-18 {
            break;
        }
    }
    roots
}

fn stabilize_ar_coeffs(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut roots = find_monic_roots(a);
    let needs_stabilize = roots
        .iter()
        .any(|&(re, im)| re * re + im * im >= 0.996 * 0.996);
    if !needs_stabilize {
        return a.to_vec();
    }
    let mut coeffs = vec![(0.0_f64, 0.0_f64); a.len() + 1];
    coeffs[0] = (1.0, 0.0);
    for r in &mut roots {
        let mag = (r.0 * r.0 + r.1 * r.1).sqrt();
        if mag >= 0.995 {
            let s = 0.995 / mag.max(1.0e-12);
            r.0 *= s;
            r.1 *= s;
        }
    }
    for (k, &root) in roots.iter().enumerate() {
        for j in (1..=k + 1).rev() {
            let prod = cmul(coeffs[j - 1], root);
            coeffs[j] = (coeffs[j].0 - prod.0, coeffs[j].1 - prod.1);
        }
    }
    coeffs[1..].iter().map(|c| c.0).collect()
}

fn solve_arx_prefiltered_step(x_f: &[f64], y_f: &[f64], m: usize) -> Option<Vec<f64>> {
    let n = x_f.len().min(y_f.len());
    if n <= m + 4 {
        return None;
    }
    let dim = 2 * m + 1;
    let packed = dim * (dim + 1) / 2;
    let mut g_u = vec![0.0_f64; packed];
    let mut rhs = vec![0.0_f64; dim];
    let mut phi = vec![0.0_f64; dim];
    for t in m..n {
        for k in 0..m {
            phi[k] = -y_f[t - k - 1];
        }
        for j in 0..=m {
            phi[m + j] = x_f[t - j];
        }
        let yt = y_f[t];
        for c in 0..dim {
            let col_off = c * (c + 1) / 2;
            let pc = phi[c];
            rhs[c] += yt * pc;
            for r in 0..=c {
                g_u[col_off + r] += phi[r] * pc;
            }
        }
    }
    for c in 0..dim {
        g_u[c * (c + 1) / 2 + c] += 1.0e-6;
    }
    let (sol, _) = solve_uplo_adaptive_ridge(dim, &g_u, &rhs, &[], None, 1.0e-6).ok()?;
    let a_next: Vec<f64> = sol[..m].to_vec();
    if a_next.iter().all(|v| v.is_finite()) {
        Some(stabilize_ar_coeffs(&a_next))
    } else {
        None
    }
}

/// Identifies `(B(z), A(z))` via iterative Steiglitz–McBride prefiltering over `(input, target)`
/// and extracts strictly stable, deduplicated upper-half-plane `(r, theta)` pole pairs via Durand–Kerner.
#[must_use]
pub fn steiglitz_mcbride_poles(
    input: &[f64],
    target: &[f64],
    order: usize,
    iters: usize,
) -> Vec<(f64, f64)> {
    let n = input.len().min(target.len());
    if order < 2 || n < order + 8 {
        return Vec::new();
    }
    let in_e: f64 = input[..n].iter().map(|v| v * v).sum();
    let mut a = if in_e > 1.0e-12 {
        let m = order.min(12);
        let mut cur_a = vec![0.0_f64; m];
        for _ in 0..iters.clamp(1, 6) {
            let xf = filter_all_pole_inverse(&input[..n], &cur_a);
            let yf = filter_all_pole_inverse(&target[..n], &cur_a);
            if xf
                .iter()
                .chain(&yf)
                .any(|v| !v.is_finite() || v.abs() > 1.0e6)
            {
                break;
            }
            if let Some(next_a) = solve_arx_prefiltered_step(&xf, &yf, m) {
                let diff: f64 = cur_a
                    .iter()
                    .zip(&next_a)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0_f64, f64::max);
                cur_a = next_a;
                if diff < 1.0e-5 {
                    break;
                }
            } else {
                break;
            }
        }
        cur_a
    } else {
        let m = order.min(24);
        let mut cur_a = solve_yule_walker_ar(&target[..n], m);
        for _ in 0..iters.min(6) {
            let w = filter_all_pole_inverse(&target[..n], &cur_a);
            if w.iter().any(|v| !v.is_finite() || v.abs() > 1.0e6) {
                break;
            }
            let next = solve_yule_walker_ar(&w, m);
            if next.iter().all(|v| v.is_finite()) {
                cur_a = stabilize_ar_coeffs(&next);
            }
        }
        cur_a
    };
    if a.is_empty() {
        return Vec::new();
    }
    a = stabilize_ar_coeffs(&a);
    let roots = find_monic_roots(&a);
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for (re, im) in roots {
        if !re.is_finite() || !im.is_finite() || im < -1.0e-6 {
            continue;
        }
        let r = (re * re + im * im).sqrt().clamp(0.0, 0.9995);
        let theta = im.abs().atan2(re).clamp(1.0e-4, PI - 1.0e-4);
        if !pairs
            .iter()
            .any(|&(pr, pt)| (pr - r).abs() < 1.0e-3 && (pt - theta).abs() < 1.0e-3)
        {
            pairs.push((r, theta));
        }
    }
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
    pairs
}

/// Evaluates one unit-excitation complex conjugate Kautz basis waveform `[0..n]` for `(r, theta)`.
#[must_use]
pub fn synth_kautz_unit_mode(r: f64, theta: f64, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = input.len();
    let (mut out_re, mut out_im) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    let ar = r * theta.cos();
    let ai = r * theta.sin();
    let (mut sr, mut si) = (0.0_f64, 0.0_f64);
    for t in 0..n {
        let next_r = ar * sr - ai * si + input[t];
        let next_i = ai * sr + ar * si;
        sr = next_r;
        si = next_i;
        out_re[t] = sr;
        out_im[t] = si;
    }
    (out_re, out_im)
}

fn solve_kautz_2x2(res: &[f64], mr: &[f64], mi: &[f64]) -> (f64, f64) {
    let n = res.len().min(mr.len()).min(mi.len());
    let (mut s_rr, mut s_ri, mut s_ii, mut r0, mut r1) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for t in 0..n {
        let m_r = mr[t];
        let m_i = mi[t];
        let y = res[t];
        s_rr += m_r * m_r;
        s_ri += m_r * m_i;
        s_ii += m_i * m_i;
        r0 += y * m_r;
        r1 -= y * m_i;
    }
    let g00 = 4.0 * s_rr + 1.0e-6;
    let g01 = -4.0 * s_ri;
    let g11 = 4.0 * s_ii + 1.0e-6;
    let rhs0 = 2.0 * r0;
    let rhs1 = 2.0 * r1;
    let det = g00 * g11 - g01 * g01;
    if !det.is_finite() || det <= 1.0e-18 {
        return (0.0, 0.0);
    }
    let c_re = ((g11 * rhs0 - g01 * rhs1) / det).clamp(-8.0, 8.0);
    let c_im = ((-g01 * rhs0 + g00 * rhs1) / det).clamp(-8.0, 8.0);
    (c_re, c_im)
}

fn fit_early_fir_taps(input: &[f64], target: &[f64], num_taps: usize) -> [f64; NUM_FIR_TAPS_ULTRA] {
    let mut taps = [0.0_f64; NUM_FIR_TAPS_ULTRA];
    let p = num_taps.min(NUM_FIR_TAPS_ULTRA);
    let n = input.len().min(target.len());
    if p == 0 || n <= p {
        return taps;
    }
    let packed = p * (p + 1) / 2;
    let mut g_u = vec![0.0_f64; packed];
    let mut rhs = vec![0.0_f64; p];
    for c in 0..p {
        let col_off = c * (c + 1) / 2;
        for r in 0..=c {
            let mut dot = 0.0_f64;
            for t in p..n {
                dot += input[t - r] * input[t - c];
            }
            if r == c {
                dot += 1.0e-5;
            }
            g_u[col_off + r] = dot;
        }
        let mut rd = 0.0_f64;
        for t in p..n {
            rd += target[t] * input[t - c];
        }
        rhs[c] = rd;
    }
    if let Ok((sol, _)) = solve_uplo_adaptive_ridge(p, &g_u, &rhs, &[], None, 1.0e-5) {
        for (i, &w) in sol.iter().enumerate() {
            taps[i] = w.clamp(-8.0, 8.0);
        }
    }
    taps
}

fn detect_transport_comb_delay(residual: &[f64], max_lag: usize) -> (f64, f64) {
    let n = residual.len();
    if n < 12 {
        return (0.0, 0.0);
    }
    let mean = residual.iter().sum::<f64>() / (n as f64);
    let centered: Vec<f64> = residual.iter().map(|&v| v - mean).collect();
    let lag_hi = max_lag.min(n / 2).max(6);
    let e0: f64 = centered.iter().map(|v| v * v).sum::<f64>().max(1.0e-12);
    let mut best_lag = 0_usize;
    let mut best_rho = 0.0_f64;
    let lag_lo = MIN_DELAY_CROSSBAR_SAMP as usize;
    for lag in lag_lo..=lag_hi {
        let mut dot = 0.0_f64;
        for t in lag..n {
            dot += centered[t] * centered[t - lag];
        }
        let rho = dot.abs() / e0;
        if rho > best_rho {
            best_rho = rho;
            best_lag = lag;
        }
    }
    if best_rho > 0.25 && best_lag >= lag_lo {
        (best_lag as f64, best_rho.clamp(0.0, 0.98))
    } else {
        (0.0, 0.0)
    }
}

fn estimate_comb_wet_tap(
    target: &[f64],
    residual: &[f64],
    start: usize,
    d: usize,
    g_init: f64,
) -> f64 {
    let n = target.len().min(residual.len());
    let mut comb = vec![0.0_f64; n];
    let (mut num_wet, mut den_wet) = (0.0_f64, 0.0_f64);
    for t in 0..n {
        let base_t = target[t] - residual[t];
        let tap = if t >= d { comb[t - d] } else { 0.0 };
        comb[t] = base_t + g_init * tap;
        if t >= start {
            num_wet += residual[t] * tap;
            den_wet += tap * tap;
        }
    }
    if den_wet <= 1.0e-12 {
        0.0
    } else {
        (num_wet / den_wet).clamp(-0.90, 0.90)
    }
}

/// Runs Pass-1 universal physics probe + pole/FIR/comb initialization from aligned stereo inputs
/// and targets, initializing a valid `SonaModelF64` (`Op-N[Pre]` zeroed, `tau >= 5.0`).
pub fn run_pass1_probe(
    input_l: &[f64],
    input_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    sample_rate_hz: f64,
    max_fir_taps: usize,
    max_kautz_pairs: usize,
) -> Result<(Pass1ProbeReport, SonaModelF64), TrainError> {
    let n = input_l
        .len()
        .min(input_r.len())
        .min(target_l.len())
        .min(target_r.len());
    if n < 64 {
        return Err(TrainError::InvalidDimensions {
            context: "run_pass1_probe",
            expected: 64,
            actual: n,
        });
    }

    let op_p = run_cpu_op_p_probe(&input_l[..n], &input_r[..n], sample_rate_hz);
    let mut model = SonaModelF64::new(sample_rate_hz);
    if op_p.dominant_freq_hz > 20.0 {
        model.phasor_freq_hz[0] = op_p.dominant_freq_hz;
    }

    let fir_n = max_fir_taps.clamp(32, NUM_FIR_TAPS_ULTRA);
    model.fir_l = fit_early_fir_taps(&input_l[..n], &target_l[..n], fir_n);
    model.fir_r = fit_early_fir_taps(&input_r[..n], &target_r[..n], fir_n);
    model.active_fir_taps = fir_n;

    let mut res_l = vec![0.0_f64; n];
    let mut res_r = vec![0.0_f64; n];
    for t in 0..n {
        let (mut fl, mut fr) = (0.0_f64, 0.0_f64);
        for k in 0..fir_n.min(t + 1) {
            fl += model.fir_l[k] * input_l[t - k];
            fr += model.fir_r[k] * input_r[t - k];
        }
        res_l[t] = target_l[t] - fl;
        res_r[t] = target_r[t] - fr;
    }

    let start = fir_n.min(n / 4);
    let tgt_l_e: f64 = target_l[start..n].iter().map(|v| v * v).sum::<f64>();
    let tgt_r_e: f64 = target_r[start..n].iter().map(|v| v * v).sum::<f64>();
    let tgt_tot_e = (tgt_l_e + tgt_r_e).max(1.0e-12);
    let res_l_e: f64 = res_l[start..n].iter().map(|v| v * v).sum();
    let res_r_e: f64 = res_r[start..n].iter().map(|v| v * v).sum();

    let mut poles: Vec<(f64, f64)> = Vec::new();
    if res_l_e > 1.0e-6 * tgt_tot_e {
        poles.extend(steiglitz_mcbride_poles(
            &input_l[start..n],
            &res_l[start..n],
            12,
            4,
        ));
    }
    if res_r_e > 1.0e-6 * tgt_tot_e {
        for &(r, th) in &steiglitz_mcbride_poles(&input_r[start..n], &res_r[start..n], 12, 4) {
            if !poles
                .iter()
                .any(|&(pr, pt)| (pr - r).abs() < 1.0e-3 && (pt - th).abs() < 1.0e-3)
            {
                poles.push((r, th));
            }
        }
    }
    poles.sort_by(|x, y| y.0.total_cmp(&x.0));

    let k_limit = max_kautz_pairs.min(NUM_KAUTZ_PAIRS_ULTRA).min(poles.len());
    let mut onset_l = vec![0.0_f64; n];
    let mut onset_r = vec![0.0_f64; n];
    if n > 16 {
        onset_l[16..n].copy_from_slice(&input_l[..(n - 16)]);
        onset_r[16..n].copy_from_slice(&input_r[..(n - 16)]);
    }
    let mut res_k_l = res_l.clone();
    let mut res_k_r = res_r.clone();
    let mut pair_idx = 0_usize;
    for &(r, theta) in &poles {
        if pair_idx >= k_limit {
            break;
        }
        if r <= 0.0 || !r.is_finite() {
            continue;
        }
        let r_eff = r.clamp(0.25, 0.9995);
        let (mr_l, mi_l) = synth_kautz_unit_mode(r_eff, theta, &onset_l);
        let (mr_r, mi_r) = synth_kautz_unit_mode(r_eff, theta, &onset_r);
        let (c_re_l, c_im_l) =
            solve_kautz_2x2(&res_k_l[start..n], &mr_l[start..n], &mi_l[start..n]);
        let (c_re_r, c_im_r) =
            solve_kautz_2x2(&res_k_r[start..n], &mr_r[start..n], &mi_r[start..n]);

        let before_e: f64 = res_k_l[start..n]
            .iter()
            .zip(&res_k_r[start..n])
            .map(|(a, b)| a * a + b * b)
            .sum();
        let mut new_e = 0.0_f64;
        for t in start..n {
            let dl = res_k_l[t] - 2.0 * (c_re_l * mr_l[t] - c_im_l * mi_l[t]);
            let dr = res_k_r[t] - 2.0 * (c_re_r * mr_r[t] - c_im_r * mi_r[t]);
            new_e += dl * dl + dr * dr;
        }
        if new_e < before_e * (1.0 - 1.0e-4) {
            for t in 0..n {
                res_k_l[t] -= 2.0 * (c_re_l * mr_l[t] - c_im_l * mi_l[t]);
                res_k_r[t] -= 2.0 * (c_re_r * mr_r[t] - c_im_r * mi_r[t]);
            }
            model.kautz[pair_idx] = KautzPairF64 {
                r_low: r_eff,
                dr: 0.0,
                theta,
                b_gamma: 1.0,
                c_re_l,
                c_im_l,
                c_re_r,
                c_im_r,
            };
            pair_idx += 1;
        }
    }
    model.active_kautz_pairs = pair_idx;

    // Stereo transport comb & feedback detection (`tau >= 5.0` samples across both L and R).
    let res_k_l_e: f64 = res_k_l[start..n].iter().map(|v| v * v).sum();
    let res_k_r_e: f64 = res_k_r[start..n].iter().map(|v| v * v).sum();
    let res_tot_e = res_k_l_e + res_k_r_e;
    let max_lag = ((n - start) / 2).min(4_096);
    let (lag_l, rho_l) = if res_k_l_e > 1.0e-3 * tgt_tot_e {
        detect_transport_comb_delay(&res_k_l[start..n], max_lag)
    } else {
        (0.0, 0.0)
    };
    let (lag_r, rho_r) = if res_k_r_e > 1.0e-3 * tgt_tot_e {
        detect_transport_comb_delay(&res_k_r[start..n], max_lag)
    } else {
        (0.0, 0.0)
    };
    let (delay_samp, loop_g) = if rho_r > rho_l && lag_r >= MIN_DELAY_CROSSBAR_SAMP {
        (lag_r, rho_r)
    } else {
        (lag_l, rho_l)
    };
    if delay_samp >= MIN_DELAY_CROSSBAR_SAMP {
        let g_init = loop_g.clamp(0.0, 0.85);
        let delay_l = if lag_l >= MIN_DELAY_CROSSBAR_SAMP {
            lag_l
        } else {
            delay_samp
        };
        let delay_r = if lag_r >= MIN_DELAY_CROSSBAR_SAMP {
            lag_r
        } else {
            delay_samp
        };
        let d_l = (delay_l.round() as usize).min(n.saturating_sub(1));
        let d_r = (delay_r.round() as usize).min(n.saturating_sub(1));
        let wl = estimate_comb_wet_tap(target_l, &res_k_l, start, d_l, g_init);
        let wr = estimate_comb_wet_tap(target_r, &res_k_r, start, d_r, g_init);
        let wl_init = if wl.abs() >= 0.05 {
            wl
        } else if lag_l >= MIN_DELAY_CROSSBAR_SAMP {
            0.35
        } else {
            0.0
        };
        let wr_init = if wr.abs() >= 0.05 {
            wr
        } else if lag_r >= MIN_DELAY_CROSSBAR_SAMP {
            0.35
        } else {
            0.0
        };
        model.transport_quads[0].delay_samples = [delay_l, delay_r, delay_l, delay_r];
        model.transport_quads[0].g_loop = g_init;
        model.transport_quads[0].input_gains = [0.5, 0.5, -0.5, -0.5];
        model.transport_quads[0].wet_tap_l = [wl_init, 0.0, -wl_init, 0.0];
        model.transport_quads[0].wet_tap_r = [0.0, wr_init, 0.0, -wr_init];
        model.transport_quads[0].ext_f32 = delay_l > 120.0 || delay_r > 120.0;
    }

    model.op_n_pre_gamma_bc = 0.0;
    model.op_n_pre_zoh_hold_depth = 0.0;
    model.op_n_pre_beta_q = 0.0;
    model.pre_conv_u_gain = if model.active_fir_taps > 0 || model.active_kautz_pairs > 0 {
        [1.0, 1.0]
    } else {
        [0.0, 0.0]
    };
    model.direct_s2_u_gain = [0.0, 0.0];

    Ok((
        Pass1ProbeReport {
            op_p,
            recovered_kautz_pairs: pair_idx,
            dominant_transport_delay_samples: delay_samp,
            estimated_loop_gain: loop_g,
            initial_esr_lin: res_tot_e / tgt_tot_e,
        },
        model,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass1_probe_recovers_fir_resonant_poles_and_comb_delay() {
        let n = 1_024_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                0.4 * (TAU * 220.0 * t).sin()
                    + 0.3 * (TAU * 1_100.0 * t).cos()
                    + if i == 40 { 0.8 } else { 0.0 }
            })
            .collect();
        let in_r = in_l.clone();

        let mut tgt_l = vec![0.0_f64; n];
        let (r_true, th_true) = (0.92_f64, 0.28_f64);
        let (mr, mi) = synth_kautz_unit_mode(r_true, th_true, &in_l);
        for t in 0..n {
            let fir = 0.6 * in_l[t] + if t >= 3 { 0.35 * in_l[t - 3] } else { 0.0 };
            let modal = 0.05 * mr[t] - 0.02 * mi[t];
            let comb = if t >= 12 { 0.25 * tgt_l[t - 12] } else { 0.0 };
            tgt_l[t] = fir + modal + comb;
        }
        let tgt_r = tgt_l.clone();

        let (rep, mdl) = run_pass1_probe(&in_l, &in_r, &tgt_l, &tgt_r, sr, 32, 8).unwrap();
        assert!(rep.op_p.analytic_energy > 0.0);
        assert!(rep.recovered_kautz_pairs > 0);
        assert!(mdl.fir_l[0] > 0.45 && mdl.fir_l[3] > 0.2);
        assert_eq!(mdl.op_n_pre_gamma_bc, 0.0);
        assert_eq!(mdl.op_n_pre_beta_q, 0.0);
        assert!(rep.initial_esr_lin < 0.25);

        // Steiglitz-McBride recovers plant poles cleanly under heavily colored input excitation.
        let mut col_in = vec![0.0_f64; n];
        let (mut s_c, mut seed) = (0.0_f64, 0x9E37_79B9_7F4A_7C15_u64);
        for v in &mut col_in {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let w = ((seed >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5;
            s_c = 0.85 * s_c + w;
            *v = s_c;
        }
        let mut plant_y = vec![0.0_f64; n];
        let a1 = -2.0 * 0.90 * (0.45_f64).cos();
        let a2 = 0.90 * 0.90;
        for t in 2..n {
            plant_y[t] = col_in[t] - a1 * plant_y[t - 1] - a2 * plant_y[t - 2];
        }
        let sm_poles = steiglitz_mcbride_poles(&col_in, &plant_y, 2, 4);
        assert!(!sm_poles.is_empty());
        assert!((sm_poles[0].0 - 0.90).abs() < 0.05, "r={:?}", sm_poles[0]);
        assert!((sm_poles[0].1 - 0.45).abs() < 0.05, "th={:?}", sm_poles[0]);

        // Right-channel-only late delayed comb recovery (`42` > `32` early FIR taps, zero Left comb).
        let mut right_only_tgt = vec![0.0_f64; n];
        let left_flat: Vec<f64> = col_in.iter().map(|&v| 0.5 * v).collect();
        for t in 0..n {
            right_only_tgt[t] = 0.5 * col_in[t]
                + if t >= 42 {
                    0.65 * right_only_tgt[t - 42]
                } else {
                    0.0
                };
        }
        let (rep_r, _) =
            run_pass1_probe(&col_in, &col_in, &left_flat, &right_only_tgt, sr, 32, 8).unwrap();
        assert_eq!(rep_r.dominant_transport_delay_samples, 42.0);
        assert!(
            rep_r.estimated_loop_gain > 0.35,
            "g={}",
            rep_r.estimated_loop_gain
        );
    }
}
