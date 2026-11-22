//! Pass-1 early FIR lifting, $\mathcal{H}_2$-normalized Kautz basis synthesis, log-octave
//! candidate pole generation, and joint stereo `2K × 2K` Cholesky modal fitting
//! (`ref[train.pipeline.physics-probe]`, `ref[dsp.op-m.init.log-octave-h2]`).

use core::f64::consts::{PI, TAU};

use crate::linalg::cholesky::solve_uplo_adaptive_ridge;
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::super::model::{KautzPairF64, NUM_FIR_TAPS_ULTRA, NUM_KAUTZ_PAIRS_ULTRA};

/// Minimum Kautz pole radius admitted by Pass-1 modal initialization.
const MIN_PASS1_KAUTZ_R: f64 = 0.25;
/// Maximum Kautz pole radius admitted by Pass-1 modal initialization (caps peak Q below `+42 dB`).
const MAX_PASS1_KAUTZ_R: f64 = 0.992;
/// Minimum modal residue norm magnitude required to retain a Kautz pair.
const MIN_KAUTZ_RESIDUE_NORM: f64 = 5.0e-4;
/// Maximum probe window frames used for Pass-1 joint Kautz Cholesky fitting.
const MAX_PASS1_KAUTZ_PROBE_FRAMES: usize = 24_000;

/// Computes the $\mathcal{H}_2$-normalized input gain $b_\gamma = \sqrt{1 - r_{\text{eff}}^2}$
/// for a complex-conjugate Kautz pair (`ref[dsp.op-m.init.log-octave-h2]`).
#[inline]
#[must_use]
pub(super) fn h2_norm_b_gamma(r_eff: f64) -> f64 {
    (1.0 - r_eff * r_eff).clamp(1.0e-4, 1.0).sqrt()
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

/// Fits a trace-scaled ridge-regularized early FIR filter (`<= 128` taps) over `(input, target)`.
pub(crate) fn fit_early_fir_taps(
    input: &[f64],
    target: &[f64],
    num_taps: usize,
) -> [f64; NUM_FIR_TAPS_ULTRA] {
    let mut taps = [0.0_f64; NUM_FIR_TAPS_ULTRA];
    let p = num_taps.min(NUM_FIR_TAPS_ULTRA);
    let n = input.len().min(target.len());
    if p == 0 || n <= p {
        return taps;
    }
    let packed = uplo_u_packed_len(p);
    let mut g_u = vec![0.0_f64; packed];
    let mut rhs = vec![0.0_f64; p];
    let mut diag_sum = 0.0_f64;
    for c in 0..p {
        let col_off = c * (c + 1) / 2;
        for r in 0..=c {
            let mut dot = 0.0_f64;
            for t in p..n {
                dot += input[t - r] * input[t - c];
            }
            g_u[col_off + r] = dot;
            if r == c {
                diag_sum += dot.max(0.0);
            }
        }
        let mut rd = 0.0_f64;
        for t in p..n {
            rd += target[t] * input[t - c];
        }
        rhs[c] = rd;
    }
    let mean_diag = (diag_sum / (p as f64)).max(0.0);
    let ridge = (1.0e-4 * mean_diag).max(1.0e-5);
    for c in 0..p {
        g_u[uplo_u_index(c, c)] += ridge;
    }
    if let Ok((sol, _)) = solve_uplo_adaptive_ridge(p, &g_u, &rhs, &[], None, ridge) {
        for (i, &w) in sol.iter().enumerate() {
            taps[i] = if w.is_finite() {
                w.clamp(-4.0, 4.0)
            } else {
                0.0
            };
        }
    }
    taps
}

/// Evaluates `fir` (`active_taps` length) over `input` and returns `target - fir(input)`.
pub(crate) fn residual_after_fir(
    input: &[f64],
    target: &[f64],
    fir: &[f64; NUM_FIR_TAPS_ULTRA],
    active_taps: usize,
) -> Vec<f64> {
    let n = input.len().min(target.len());
    let taps = active_taps.min(NUM_FIR_TAPS_ULTRA);
    let mut res = vec![0.0_f64; n];
    for t in 0..n {
        let mut y_fir = 0.0_f64;
        for k in 0..taps.min(t + 1) {
            y_fir += fir[k] * input[t - k];
        }
        res[t] = target[t] - y_fir;
    }
    res
}

/// Appends universal log-octave $\mathcal{H}_2$ Kautz candidate poles (`[70 Hz .. 0.38*fs]`) up to
/// `k_cap`, skipping angles already within `0.025 rad` of an existing candidate
/// (`ref[dsp.op-m.init.log-octave-h2]`).
pub(crate) fn append_log_octave_h2_candidates(
    poles: &mut Vec<(f64, f64)>,
    sample_rate_hz: f64,
    k_cap: usize,
) {
    if poles.len() >= k_cap || k_cap == 0 {
        return;
    }
    let sr = sample_rate_hz.max(8_000.0);
    let f_lo = 70.0_f64;
    let f_hi = (0.38 * sr).max(f_lo * 2.0);
    let ratio = f_hi / f_lo;
    let n_grid = k_cap.max(NUM_KAUTZ_PAIRS_ULTRA);
    for idx in 0..n_grid {
        if poles.len() >= k_cap {
            break;
        }
        let frac = if n_grid <= 1 {
            0.0
        } else {
            (idx as f64) / ((n_grid - 1) as f64)
        };
        let fc_hz = f_lo * ratio.powf(frac);
        let theta = (TAU * fc_hz / sr).clamp(0.005, PI - 0.005);
        let r = if fc_hz < 1_500.0 { 0.90 } else { 0.86 };
        if !poles.iter().any(|&(_, pt)| (pt - theta).abs() < 0.025) {
            poles.push((r, theta));
        }
    }
}

/// Borrowed inputs for [`fit_joint_kautz_bank`].
pub(crate) struct KautzFitInputs<'a> {
    pub candidates: &'a [(f64, f64)],
    pub input_l: &'a [f64],
    pub input_r: &'a [f64],
    pub res_in_l: &'a [f64],
    pub res_in_r: &'a [f64],
    pub start: usize,
    pub k_limit: usize,
}

/// Fits a joint `2K × 2K` regularized stereo Kautz bank on the `t >= 16` onset residual using
/// $\mathcal{H}_2$-normalized columns (`2 * b_gamma * mr`, `-2 * b_gamma * mi`), sorts surviving pairs
/// by descending stereo modal output energy, and returns `(accepted_pairs, res_l, res_r)`.
pub(crate) fn fit_joint_kautz_bank(
    inp: &KautzFitInputs<'_>,
) -> (Vec<KautzPairF64>, Vec<f64>, Vec<f64>) {
    let (candidates, input_l, input_r, res_in_l, res_in_r, start, k_limit) = (
        inp.candidates,
        inp.input_l,
        inp.input_r,
        inp.res_in_l,
        inp.res_in_r,
        inp.start,
        inp.k_limit,
    );
    let n = input_l
        .len()
        .min(input_r.len())
        .min(res_in_l.len())
        .min(res_in_r.len());
    let k_cand = candidates.len().min(k_limit).min(NUM_KAUTZ_PAIRS_ULTRA);
    if k_cand == 0 || n <= start + 8 {
        return (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec());
    }

    let fit_end = n.min(start + MAX_PASS1_KAUTZ_PROBE_FRAMES);
    let m = fit_end - start;
    let mut onset_fit_l = vec![0.0_f64; fit_end];
    let mut onset_fit_r = vec![0.0_f64; fit_end];
    if fit_end > 16 {
        onset_fit_l[16..fit_end].copy_from_slice(&input_l[..(fit_end - 16)]);
        onset_fit_r[16..fit_end].copy_from_slice(&input_r[..(fit_end - 16)]);
    }

    let mut sanitized: Vec<(f64, f64, f64)> = Vec::with_capacity(k_cand);
    for &(r, theta) in candidates.iter().take(k_cand) {
        if r.is_finite() && theta.is_finite() && r > 0.0 {
            let r_eff = r.clamp(MIN_PASS1_KAUTZ_R, MAX_PASS1_KAUTZ_R);
            let th_eff = theta.clamp(0.005, PI - 0.005);
            let b_gamma = h2_norm_b_gamma(r_eff);
            sanitized.push((r_eff, th_eff, b_gamma));
        }
    }
    let k_act = sanitized.len();
    if k_act == 0 {
        return (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec());
    }

    let dim = 2 * k_act;
    let mut cols_l = Vec::with_capacity(dim);
    let mut cols_r = Vec::with_capacity(dim);
    for &(r_eff, th_eff, b_gamma) in &sanitized {
        let scale = 2.0 * b_gamma;
        let (mr_l, mi_l) = synth_kautz_unit_mode(r_eff, th_eff, &onset_fit_l);
        let (mr_r, mi_r) = synth_kautz_unit_mode(r_eff, th_eff, &onset_fit_r);
        cols_l.push(
            mr_l[start..fit_end]
                .iter()
                .map(|&v| scale * v)
                .collect::<Vec<_>>(),
        );
        cols_l.push(
            mi_l[start..fit_end]
                .iter()
                .map(|&v| -scale * v)
                .collect::<Vec<_>>(),
        );
        cols_r.push(
            mr_r[start..fit_end]
                .iter()
                .map(|&v| scale * v)
                .collect::<Vec<_>>(),
        );
        cols_r.push(
            mi_r[start..fit_end]
                .iter()
                .map(|&v| -scale * v)
                .collect::<Vec<_>>(),
        );
    }

    let solve_side = |cols: &[Vec<f64>], res: &[f64]| -> Vec<f64> {
        let mut g_u = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut rhs = vec![0.0_f64; dim];
        let mut diag_sum = 0.0_f64;
        for c in 0..dim {
            let col_off = c * (c + 1) / 2;
            let cc = &cols[c];
            for r in 0..=c {
                let cr = &cols[r];
                let dot: f64 = (0..m).map(|t| cr[t] * cc[t]).sum();
                g_u[col_off + r] = dot;
                if r == c {
                    diag_sum += dot.max(0.0);
                }
            }
            rhs[c] = (0..m).map(|t| cc[t] * res[start + t]).sum();
        }
        let mean_diag = (diag_sum / (dim as f64)).max(0.0);
        let ridge = (2.0e-3 * mean_diag).max(1.0e-4);
        for c in 0..dim {
            g_u[uplo_u_index(c, c)] += ridge;
        }
        solve_uplo_adaptive_ridge(dim, &g_u, &rhs, &[], None, ridge)
            .map(|(sol, _)| {
                sol.into_iter()
                    .map(|v| {
                        if v.is_finite() {
                            v.clamp(-2.5, 2.5)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .unwrap_or_else(|_| vec![0.0; dim])
    };

    let coef_l = solve_side(&cols_l, res_in_l);
    let coef_r = solve_side(&cols_r, res_in_r);

    let mut ranked: Vec<(f64, KautzPairF64)> = Vec::with_capacity(k_act);
    for (idx, &(r_eff, th_eff, b_gamma)) in sanitized.iter().enumerate() {
        let cl_re = coef_l[2 * idx];
        let cl_im = coef_l[2 * idx + 1];
        let cr_re = coef_r[2 * idx];
        let cr_im = coef_r[2 * idx + 1];
        if cl_re.hypot(cl_im) + cr_re.hypot(cr_im) <= MIN_KAUTZ_RESIDUE_NORM {
            continue;
        }
        let e_l: f64 = (0..m)
            .map(|t| {
                let y = cl_re * cols_l[2 * idx][t] + cl_im * cols_l[2 * idx + 1][t];
                y * y
            })
            .sum();
        let e_r: f64 = (0..m)
            .map(|t| {
                let y = cr_re * cols_r[2 * idx][t] + cr_im * cols_r[2 * idx + 1][t];
                y * y
            })
            .sum();
        ranked.push((
            e_l + e_r,
            KautzPairF64 {
                r_low: r_eff,
                dr: 0.0,
                theta: th_eff,
                b_gamma,
                c_re_l: cl_re,
                c_im_l: cl_im,
                c_re_r: cr_re,
                c_im_r: cr_im,
            },
        ));
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let accepted: Vec<KautzPairF64> = ranked.into_iter().map(|(_, p)| p).collect();
    if accepted.is_empty() {
        return (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec());
    }

    let mut onset_full_l = vec![0.0_f64; n];
    let mut onset_full_r = vec![0.0_f64; n];
    if n > 16 {
        onset_full_l[16..n].copy_from_slice(&input_l[..(n - 16)]);
        onset_full_r[16..n].copy_from_slice(&input_r[..(n - 16)]);
    }
    let mut out_l = res_in_l[..n].to_vec();
    let mut out_r = res_in_r[..n].to_vec();
    for pair in &accepted {
        let (mr_l, mi_l) = synth_kautz_unit_mode(pair.r_low, pair.theta, &onset_full_l);
        let (mr_r, mi_r) = synth_kautz_unit_mode(pair.r_low, pair.theta, &onset_full_r);
        let scale = 2.0 * pair.b_gamma;
        for t in 0..n {
            out_l[t] -= scale * (pair.c_re_l * mr_l[t] - pair.c_im_l * mi_l[t]);
            out_r[t] -= scale * (pair.c_re_r * mr_r[t] - pair.c_im_r * mi_r[t]);
        }
    }

    let before_e: f64 = res_in_l[start..n]
        .iter()
        .zip(&res_in_r[start..n])
        .map(|(a, b)| a * a + b * b)
        .sum();
    let after_e: f64 = out_l[start..n]
        .iter()
        .zip(&out_r[start..n])
        .map(|(a, b)| a * a + b * b)
        .sum();
    if after_e < before_e * 0.995 {
        (accepted, out_l, out_r)
    } else {
        (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec())
    }
}
