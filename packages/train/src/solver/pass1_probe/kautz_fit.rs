//! Pass-1 $\mathcal{H}_2$-normalized Kautz basis synthesis, sub-bass to `0.9992` high-Q candidate
//! generation, and 2-pass OMP + Steiglitz–McBride stereo `2K × 2K` Cholesky modal fitting
//! (`ref[train.pipeline.physics-probe]`, `ref[dsp.op-m.init.log-octave-h2]`,
//! `ref[dsp.op-m.kautz.pole-extraction]`).

use core::f64::consts::{LN_10, PI, TAU};

use crate::linalg::cholesky::solve_uplo_adaptive_ridge;
use crate::losses::certification::{GAP_PRERING_TRIM_MAX_SAMPLES, MIN_POST_T60_GAP_SAMPLES};
use crate::losses::{DRY_SILENCE_EPS, MIN_DRY_SILENT_GAP_SECS, POST_T60_GAP_AMP_RATIO};
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::super::model::{KautzPairF64, NUM_KAUTZ_PAIRS_ULTRA};
use super::kautz_varpro::select_multiregime_blocks;
use super::steiglitz_mcbride::steiglitz_mcbride_poles;

// Re-export Early FIR utilities for Pass-3 `pre_conv_proj` callers.
pub(crate) use super::fir_toeplitz::{fit_early_fir_taps, residual_after_fir};

/// Minimum Kautz pole radius admitted by Pass-1 modal initialization.
pub(super) const MIN_PASS1_KAUTZ_R: f64 = 0.25;
/// Maximum Kautz pole radius admitted by Pass-1 modal initialization (`ref[dsp.op-m.kautz.pole-extraction]`).
pub(crate) const MAX_PASS1_KAUTZ_R: f64 = 0.9992;
/// Minimum audible frequency boundary (Hz) for Pass-1/2 Kautz poles (`ref[qual.companion.ltas]`).
pub(crate) const MIN_AUDIBLE_KAUTZ_HZ: f64 = 20.0;
/// Maximum audible frequency ceiling (Hz) for Pass-1/2 Kautz poles (`ref[qual.companion.ltas]`).
pub(crate) const MAX_AUDIBLE_KAUTZ_HZ: f64 = 19_500.0;
/// Maximum Nyquist fraction for Pass-1/2 Kautz poles.
pub(crate) const MAX_AUDIBLE_KAUTZ_NYQ_FRAC: f64 = 0.42;
const MIN_PHYS_KAUTZ_R_CEIL: f64 = 0.95;
const PHYS_DECAY_MARGIN_FACTOR: f64 = 2.0;
const MIN_PHYS_DECAY_BUDGET_SECS: f64 = 0.025;
const MIN_BOUNDARY_POLE_R: f64 = 0.85;
/// Minimum modal residue norm magnitude required to retain a Kautz pair.
const MIN_KAUTZ_RESIDUE_NORM: f64 = 5.0e-4;
/// Maximum probe window frames used for Pass-1 joint Kautz Cholesky fitting.
const MAX_PASS1_KAUTZ_PROBE_FRAMES: usize = 24_000;
/// Maximum BIBO steady-state resonant peak gain `2 * b_gamma * |c| / (1 - r)` per Pass-1 Kautz pair (`< RAIL_OUTPUT_CEIL = 100.0`).
const MAX_PASS1_KAUTZ_PEAK_GAIN: f64 = 48.0;
const _: () = assert!(
    MAX_PASS1_KAUTZ_PEAK_GAIN > 0.0
        && MAX_PASS1_KAUTZ_PEAK_GAIN < 100.0
        && MIN_AUDIBLE_KAUTZ_HZ >= 20.0
        && MAX_AUDIBLE_KAUTZ_HZ < 20_000.0
        && MAX_AUDIBLE_KAUTZ_NYQ_FRAC > 0.35
        && MAX_AUDIBLE_KAUTZ_NYQ_FRAC < 0.49
        && MIN_PHYS_KAUTZ_R_CEIL > 0.90
        && MIN_PHYS_KAUTZ_R_CEIL < MAX_PASS1_KAUTZ_R
        && PHYS_DECAY_MARGIN_FACTOR >= 1.5
        && MIN_PHYS_DECAY_BUDGET_SECS > 0.0
        && MIN_BOUNDARY_POLE_R > 0.5
        && MIN_BOUNDARY_POLE_R < MIN_PHYS_KAUTZ_R_CEIL
);

/// Returns canonical audible `(theta_min, theta_max)` bounds for Kautz modal pairs at `sample_rate_hz`.
#[inline]
#[must_use]
pub(crate) fn audible_kautz_theta_bounds(sample_rate_hz: f64) -> (f64, f64) {
    let sr = if sample_rate_hz.is_finite() && sample_rate_hz >= 8_000.0 {
        sample_rate_hz
    } else {
        48_000.0
    };
    let th_min = (TAU * MIN_AUDIBLE_KAUTZ_HZ / sr).clamp(0.0025, 0.25);
    let f_hi = MAX_AUDIBLE_KAUTZ_HZ.min(MAX_AUDIBLE_KAUTZ_NYQ_FRAC * sr);
    let th_max = (TAU * f_hi / sr).clamp(th_min + 0.05, PI - 0.05);
    (th_min, th_max)
}

/// Estimates a universal data-driven upper Kautz pole-radius ceiling from dry-silent gap `T60`
/// decay spans across both stereo channels so short-decay systems never hallucinate `0.9992`
/// phantom tails into silent gaps while genuinely long-tail systems keep `MAX_PASS1_KAUTZ_R`.
#[must_use]
pub(crate) fn estimate_physical_kautz_r_ceil(
    input_l: &[f64],
    input_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    warmup: usize,
    sample_rate_hz: f64,
) -> f64 {
    let sr = if sample_rate_hz.is_finite() && sample_rate_hz >= 8_000.0 {
        sample_rate_hz
    } else {
        return MAX_PASS1_KAUTZ_R;
    };
    let min_gap = (sr * MIN_DRY_SILENT_GAP_SECS).round().max(1.0) as usize;
    let mut saw_gap = false;
    let mut max_decay_samples = 0_usize;

    for (dry, tgt) in [(input_l, target_l), (input_r, target_r)] {
        let n = dry.len().min(tgt.len());
        if n <= warmup {
            continue;
        }
        let (mut act_sum, mut act_cnt) = (0.0_f64, 0_usize);
        for i in warmup..n {
            let (d, y) = (dry[i], tgt[i]);
            if !d.is_finite() || !y.is_finite() {
                return MAX_PASS1_KAUTZ_R;
            }
            if d.abs() > DRY_SILENCE_EPS {
                act_sum += y;
                act_cnt += 1;
            }
        }
        if act_cnt == 0 {
            continue;
        }
        let act_mean = act_sum / (act_cnt as f64);
        let mut act_sq = 0.0_f64;
        for i in warmup..n {
            if dry[i].abs() > DRY_SILENCE_EPS {
                let d_ac = tgt[i] - act_mean;
                act_sq += d_ac * d_ac;
            }
        }
        let act_rms = (act_sq / (act_cnt as f64)).sqrt();
        if !act_rms.is_finite() || act_rms <= 1.0e-12 {
            continue;
        }
        let tau60 = POST_T60_GAP_AMP_RATIO * act_rms;
        let mut i = warmup;
        while i < n {
            if dry[i].abs() <= DRY_SILENCE_EPS {
                let s_dry = i;
                while i < n && dry[i].abs() <= DRY_SILENCE_EPS {
                    i += 1;
                }
                let e_dry = i;
                if e_dry - s_dry >= min_gap {
                    saw_gap = true;
                    let tail_e = if e_dry < n {
                        e_dry
                            .saturating_sub(GAP_PRERING_TRIM_MAX_SAMPLES)
                            .max(s_dry)
                    } else {
                        e_dry
                    };
                    let half = MIN_POST_T60_GAP_SAMPLES;
                    let span = 2 * half;
                    let tail_s = tail_e.saturating_sub(span);
                    let gap_dc = if tail_s >= s_dry && tail_e - tail_s == span {
                        let mid = tail_s + half;
                        let m1 = tgt[tail_s..mid].iter().sum::<f64>() / (half as f64);
                        let m2 = tgt[mid..tail_e].iter().sum::<f64>() / (half as f64);
                        let (mut lo_v, mut hi_v) = (f64::INFINITY, f64::NEG_INFINITY);
                        for &v in &tgt[tail_s..tail_e] {
                            lo_v = lo_v.min(v);
                            hi_v = hi_v.max(v);
                        }
                        let p2p = hi_v - lo_v;
                        if (m1 - m2).abs() <= 1.0e-5 * act_rms && p2p <= 0.1 * tau60 {
                            m2
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let mut e_end = e_dry;
                    if e_dry < n {
                        let ring_lo = e_dry
                            .saturating_sub(GAP_PRERING_TRIM_MAX_SAMPLES)
                            .max(s_dry);
                        if let Some(first_ring) =
                            (ring_lo..e_dry).find(|&j| (tgt[j] - gap_dc).abs() > tau60)
                        {
                            e_end = first_ring;
                        }
                    }
                    let last_above = (s_dry..e_end)
                        .rev()
                        .find(|&j| (tgt[j] - gap_dc).abs() > tau60);
                    let s_t60 = last_above.map_or(s_dry, |idx| idx + 1);
                    if e_end.saturating_sub(s_t60) < MIN_POST_T60_GAP_SAMPLES {
                        return MAX_PASS1_KAUTZ_R;
                    }
                    max_decay_samples = max_decay_samples.max(s_t60 - s_dry);
                }
            } else {
                i += 1;
            }
        }
    }

    if !saw_gap {
        return MAX_PASS1_KAUTZ_R;
    }
    let budget = ((max_decay_samples as f64) * PHYS_DECAY_MARGIN_FACTOR)
        .max(MIN_PHYS_DECAY_BUDGET_SECS * sr);
    (-(3.0 * LN_10) / budget)
        .exp()
        .clamp(MIN_PHYS_KAUTZ_R_CEIL, MAX_PASS1_KAUTZ_R)
}

fn kautz_sol_within_bibo_bounds(sol: &[f64], sanitized: &[(f64, f64, f64)]) -> bool {
    if sol.len() < 2 * sanitized.len() || sol.iter().any(|&v| !v.is_finite() || v.abs() > 4.0) {
        return false;
    }
    for (idx, &(r_eff, _, b_gamma)) in sanitized.iter().enumerate() {
        let c_mag = sol[2 * idx].hypot(sol[2 * idx + 1]);
        let peak_gain = 2.0 * b_gamma * c_mag / (1.0 - r_eff).max(1.0e-6);
        if !peak_gain.is_finite() || peak_gain > MAX_PASS1_KAUTZ_PEAK_GAIN {
            return false;
        }
    }
    true
}

fn project_kautz_sol_bibo(sol: &[f64], sanitized: &[(f64, f64, f64)]) -> Vec<f64> {
    let mut out: Vec<f64> = sol
        .iter()
        .map(|&v| {
            if v.is_finite() {
                v.clamp(-4.0, 4.0)
            } else {
                0.0
            }
        })
        .collect();
    for (idx, &(r_eff, _, b_gamma)) in sanitized.iter().enumerate() {
        if 2 * idx + 1 >= out.len() {
            break;
        }
        let c_mag = out[2 * idx].hypot(out[2 * idx + 1]);
        let one_minus_r = (1.0 - r_eff).max(1.0e-6);
        let peak_gain = 2.0 * b_gamma * c_mag / one_minus_r;
        if peak_gain > MAX_PASS1_KAUTZ_PEAK_GAIN {
            let scale =
                MAX_PASS1_KAUTZ_PEAK_GAIN * one_minus_r / (2.0 * b_gamma * c_mag).max(1.0e-12);
            out[2 * idx] *= scale;
            out[2 * idx + 1] *= scale;
        }
    }
    out
}

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

/// Complex-plane separation guard rejecting collinear duplicate poles (`|z1 - z2| < 0.012`).
#[inline]
#[must_use]
pub(crate) fn poles_too_close(r1: f64, th1: f64, r2: f64, th2: f64) -> bool {
    (r1 * th1.cos() - r2 * th2.cos()).hypot(r1 * th1.sin() - r2 * th2.sin()) < 0.012
}

/// Appends universal log-octave $\mathcal{H}_2$ Kautz candidate poles (`[40 Hz .. 0.38*fs]`
/// with band radii `0.975 / 0.92 / 0.86`) up to `k_cap`, skipping poles already close to an
/// existing candidate (`ref[dsp.op-m.init.log-octave-h2]`).
pub(crate) fn append_log_octave_h2_candidates(
    poles: &mut Vec<(f64, f64)>,
    sample_rate_hz: f64,
    k_cap: usize,
) {
    if poles.len() >= k_cap || k_cap == 0 {
        return;
    }
    let sr = sample_rate_hz.max(8_000.0);
    let f_lo = 40.0_f64;
    let f_hi = (0.38 * sr).min(MAX_AUDIBLE_KAUTZ_HZ).max(f_lo * 2.0);
    let (th_min, th_max) = audible_kautz_theta_bounds(sr);
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
        let theta = (TAU * fc_hz / sr).clamp(th_min, th_max);
        let r = if fc_hz < 180.0 {
            0.975
        } else if fc_hz < 1_500.0 {
            0.92
        } else {
            0.86
        };
        if !poles.iter().any(|&(pr, pt)| {
            poles_too_close(pr, pt, r, theta)
                || ((pt - theta).abs() < 0.025 && (pr - r).abs() < 0.12)
        }) {
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
    pub sample_rate_hz: f64,
    pub r_max: f64,
    pub lock_prefix_lens: [usize; 2],
}

type JointKautzSurvivor = (KautzPairF64, Vec<f64>, Vec<f64>, f64, u8);

fn solve_joint_kautz_candidates(
    sanitized: &[(f64, f64, f64)],
    inp: &KautzFitInputs<'_>,
    blocks: &[(usize, usize)],
    lock_lens: [usize; 2],
) -> (Vec<KautzPairF64>, Vec<f64>, Vec<f64>, f64, [usize; 2]) {
    let (input_l, input_r, res_in_l, res_in_r, start) = (
        inp.input_l,
        inp.input_r,
        inp.res_in_l,
        inp.res_in_r,
        inp.start,
    );
    let n = input_l
        .len()
        .min(input_r.len())
        .min(res_in_l.len())
        .min(res_in_r.len());
    let fit_end = blocks.last().map_or(start, |&(_, be)| be.min(n));
    let k_act = sanitized.len();
    let m: usize = blocks.iter().map(|&(bs, be)| be.saturating_sub(bs)).sum();
    if k_act == 0 || fit_end <= start || m == 0 {
        return (
            Vec::new(),
            res_in_l[..n].to_vec(),
            res_in_r[..n].to_vec(),
            f64::INFINITY,
            [0, 0],
        );
    }

    let mut onset_fit_l = vec![0.0_f64; fit_end];
    let mut onset_fit_r = vec![0.0_f64; fit_end];
    if fit_end > 16 {
        onset_fit_l[16..fit_end].copy_from_slice(&input_l[..(fit_end - 16)]);
        onset_fit_r[16..fit_end].copy_from_slice(&input_r[..(fit_end - 16)]);
    }

    let dim = 2 * k_act;
    let mut cols_l = Vec::with_capacity(dim);
    let mut cols_r = Vec::with_capacity(dim);
    let gather_blocks = |sig: &[f64], sign_scale: f64| -> Vec<f64> {
        let mut out = Vec::with_capacity(m);
        for &(bs, be) in blocks {
            for &v in &sig[bs..be] {
                out.push(sign_scale * v);
            }
        }
        out
    };
    for &(r_eff, th_eff, b_gamma) in sanitized {
        let scale = 2.0 * b_gamma;
        let (mr_l, mi_l) = synth_kautz_unit_mode(r_eff, th_eff, &onset_fit_l);
        let (mr_r, mi_r) = synth_kautz_unit_mode(r_eff, th_eff, &onset_fit_r);
        cols_l.push(gather_blocks(&mr_l, scale));
        cols_l.push(gather_blocks(&mi_l, -scale));
        cols_r.push(gather_blocks(&mr_r, scale));
        cols_r.push(gather_blocks(&mi_r, -scale));
    }

    let res_seg_l = gather_blocks(res_in_l, 1.0);
    let res_seg_r = gather_blocks(res_in_r, 1.0);
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
            rhs[c] = (0..m).map(|t| cc[t] * res[t]).sum();
        }
        let mean_diag = (diag_sum / (dim as f64)).max(0.0);
        // Compact refined pole banks (<= 8 pairs) need tighter ridge than dense 32-pole grids.
        let _ = uplo_u_index(0, 0);
        let (rel_ridge, min_ridge, max_ridge) = if sanitized.len() <= 8 {
            (1.0e-5, 1.0e-8, 1.0e-2)
        } else {
            (1.0e-3, 1.0e-6, 1.0e-1)
        };
        let mut ridge = (rel_ridge * mean_diag).clamp(min_ridge, max_ridge);
        let mut best_sol = vec![0.0_f64; dim];
        for _retry in 0..7 {
            if let Ok((sol, used)) = solve_uplo_adaptive_ridge(dim, &g_u, &rhs, &[], None, ridge) {
                if kautz_sol_within_bibo_bounds(&sol, sanitized) {
                    best_sol = sol;
                    break;
                }
                best_sol = project_kautz_sol_bibo(&sol, sanitized);
                ridge = (used * 10.0)
                    .max(ridge * 10.0)
                    .min(mean_diag.max(1.0) * 1.0e3);
            } else {
                break;
            }
        }
        best_sol
    };

    let coef_l = solve_side(&cols_l, &res_seg_l);
    let coef_r = solve_side(&cols_r, &res_seg_r);

    let mut survivors: Vec<JointKautzSurvivor> = Vec::with_capacity(k_act);
    for (idx, &(r_eff, th_eff, b_gamma)) in sanitized.iter().enumerate() {
        let cl_re = coef_l[2 * idx];
        let cl_im = coef_l[2 * idx + 1];
        let cr_re = coef_r[2 * idx];
        let cr_im = coef_r[2 * idx + 1];
        if cl_re.hypot(cl_im) + cr_re.hypot(cr_im) <= MIN_KAUTZ_RESIDUE_NORM {
            continue;
        }
        let contrib_l: Vec<f64> = (0..m)
            .map(|t| cl_re * cols_l[2 * idx][t] + cl_im * cols_l[2 * idx + 1][t])
            .collect();
        let contrib_r: Vec<f64> = (0..m)
            .map(|t| cr_re * cols_r[2 * idx][t] + cr_im * cols_r[2 * idx + 1][t])
            .collect();
        let self_e: f64 = contrib_l.iter().map(|v| v * v).sum::<f64>()
            + contrib_r.iter().map(|v| v * v).sum::<f64>();
        let bucket = if idx < lock_lens[0] {
            0_u8
        } else if idx < lock_lens[1] {
            1_u8
        } else {
            2_u8
        };
        survivors.push((
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
            contrib_l,
            contrib_r,
            self_e,
            bucket,
        ));
    }
    if survivors.is_empty() {
        return (
            Vec::new(),
            res_in_l[..n].to_vec(),
            res_in_r[..n].to_vec(),
            f64::INFINITY,
            [0, 0],
        );
    }

    // Order surviving pairs into nested tier prefixes (`0..8` Lite, `0..12` Full, `0..24` Pro,
    // `0..32` Ultra) strictly bucket-by-bucket (`0`: locked Lite, `1`: locked Full, `2`: upper)
    // by non-negative scalar OMP projection gain `<rem, c>^2 / ||c||^2`.
    let mut rem_l = res_seg_l;
    let mut rem_r = res_seg_r;
    let mut used = vec![false; survivors.len()];
    let mut accepted = Vec::with_capacity(survivors.len());
    let mut bucket_counts = [0_usize; 3];
    for target_bucket in 0..=2_u8 {
        loop {
            let mut best_i = None;
            let mut best_alpha = 0.0_f64;
            let mut best_gain = -1.0_f64;
            for (i, (_, cl, cr, self_e, bucket)) in survivors.iter().enumerate() {
                if used[i] || *bucket != target_bucket {
                    continue;
                }
                let dot: f64 = (0..m).map(|t| rem_l[t] * cl[t] + rem_r[t] * cr[t]).sum();
                let (alpha, gain) = if *self_e > 1.0e-18 {
                    let a = dot / self_e;
                    (a, a * dot)
                } else {
                    (0.0, 0.0)
                };
                if gain > best_gain {
                    best_gain = gain;
                    best_alpha = alpha;
                    best_i = Some(i);
                }
            }
            let Some(pick) = best_i else {
                break;
            };
            used[pick] = true;
            if best_alpha != 0.0 {
                for t in 0..m {
                    rem_l[t] -= best_alpha * survivors[pick].1[t];
                    rem_r[t] -= best_alpha * survivors[pick].2[t];
                }
            }
            accepted.push(survivors[pick].0);
            bucket_counts[target_bucket as usize] += 1;
        }
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
    let after_e: f64 = out_l[start..n]
        .iter()
        .zip(&out_r[start..n])
        .map(|(a, b)| a * a + b * b)
        .sum();
    let kept_lock0 = bucket_counts[0];
    let kept_lock1 = kept_lock0 + bucket_counts[1];
    (accepted, out_l, out_r, after_e, [kept_lock0, kept_lock1])
}

/// Fits a 2-pass OMP + Steiglitz–McBride refined `2K × 2K` joint stereo Kautz bank (up to `32`
/// pairs, `r_low in [0.25, r_max]`, `dr = 0.0`) on the `t >= 16` onset residual
/// (`ref[train.pipeline.physics-probe]`, `ref[dsp.op-m.kautz.pole-extraction]`).
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
    let k_cap = k_limit.min(NUM_KAUTZ_PAIRS_ULTRA);
    let k_cand = candidates.len().min(k_cap);
    if k_cand == 0 || n <= start + 8 {
        return (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec());
    }

    let blocks = select_multiregime_blocks(res_in_l, res_in_r, start, MAX_PASS1_KAUTZ_PROBE_FRAMES);
    let fit_end = blocks.last().map_or(start, |&(_, be)| be.min(n));
    let before_e: f64 = res_in_l[start..n]
        .iter()
        .zip(&res_in_r[start..n])
        .map(|(a, b)| a * a + b * b)
        .sum();
    if before_e <= 1.0e-18 || blocks.is_empty() {
        return (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec());
    }

    let (th_min, th_max) = audible_kautz_theta_bounds(inp.sample_rate_hz);
    let r_cap = inp.r_max.clamp(MIN_PASS1_KAUTZ_R, MAX_PASS1_KAUTZ_R);
    let sanitize = |r: f64, theta: f64| -> Option<(f64, f64, f64)> {
        if r.is_finite()
            && theta.is_finite()
            && r > 0.0
            && theta <= th_max
            && !(theta <= th_min + 1.0e-4 && r < MIN_BOUNDARY_POLE_R)
        {
            let r_eff = r.clamp(MIN_PASS1_KAUTZ_R, r_cap);
            let th_eff = theta.clamp(th_min, th_max);
            Some((r_eff, th_eff, h2_norm_b_gamma(r_eff)))
        } else {
            None
        }
    };

    let lock_req0 = inp.lock_prefix_lens[0]
        .min(inp.lock_prefix_lens[1])
        .min(k_cand);
    let lock_req1 = inp.lock_prefix_lens[1].min(k_cand);
    let mut base_poles: Vec<(f64, f64, f64)> = Vec::with_capacity(k_cand);
    let (mut base_lock0, mut base_lock1) = (0_usize, 0_usize);
    for (idx, &(r, th)) in candidates.iter().take(k_cand).enumerate() {
        if let Some(s) = sanitize(r, th) {
            if idx < lock_req0 {
                base_lock0 += 1;
            }
            if idx < lock_req1 {
                base_lock1 += 1;
            }
            base_poles.push(s);
        }
    }
    if base_poles.is_empty() {
        return (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec());
    }

    // Pass A (OMP Stage 1): Solve joint Cholesky across initial candidate bank.
    let (acc_a, out_al, out_ar, err_a, kept_locks) =
        solve_joint_kautz_candidates(&base_poles, inp, &blocks, [base_lock0, base_lock1]);
    let mut best = if err_a < before_e * 0.995 {
        Some((acc_a.clone(), out_al, out_ar, err_a))
    } else {
        None
    };

    // Pass B (OMP Stage 2 + Steiglitz–McBride residual refinement): keep top modes from Pass A,
    // extract missed poles only when Pass A leaves substantial (> 5%) residual energy, and
    // re-solve joint 2K x 2K Cholesky cleanly.
    if err_a > 0.05 * before_e && k_cap >= 2 && !acc_a.is_empty() && fit_end > 24 {
        let keep_n = acc_a.len().min((k_cap / 2).max(kept_locks[1]).max(1));
        let mut onset_sm_l = vec![0.0_f64; fit_end];
        let mut onset_sm_r = vec![0.0_f64; fit_end];
        onset_sm_l[16..fit_end].copy_from_slice(&input_l[..(fit_end - 16)]);
        onset_sm_r[16..fit_end].copy_from_slice(&input_r[..(fit_end - 16)]);

        let (_, res_mid_l, res_mid_r, _, _) = solve_joint_kautz_candidates(
            &acc_a[..keep_n]
                .iter()
                .map(|p| (p.r_low, p.theta, p.b_gamma))
                .collect::<Vec<_>>(),
            inp,
            &blocks,
            [kept_locks[0].min(keep_n), kept_locks[1].min(keep_n)],
        );
        let (sm_s, sm_e) = blocks
            .iter()
            .copied()
            .max_by(|&(s1, e1), &(s2, e2)| {
                let en1: f64 = res_mid_l[s1..e1]
                    .iter()
                    .zip(&res_mid_r[s1..e1])
                    .map(|(a, b)| a * a + b * b)
                    .sum();
                let en2: f64 = res_mid_l[s2..e2]
                    .iter()
                    .zip(&res_mid_r[s2..e2])
                    .map(|(a, b)| a * a + b * b)
                    .sum();
                en1.total_cmp(&en2)
            })
            .unwrap_or((start, fit_end));
        let sm_len = sm_e.min(sm_s + 16_384);
        let sm_order = (k_cap - keep_n).clamp(2, 16);
        let mut combined: Vec<(f64, f64, f64)> = acc_a[..keep_n]
            .iter()
            .map(|p| (p.r_low, p.theta, p.b_gamma))
            .collect();
        for (r, th) in steiglitz_mcbride_poles(
            &onset_sm_l[sm_s..sm_len],
            &res_mid_l[sm_s..sm_len],
            sm_order,
            4,
        )
        .into_iter()
        .chain(steiglitz_mcbride_poles(
            &onset_sm_r[sm_s..sm_len],
            &res_mid_r[sm_s..sm_len],
            sm_order,
            4,
        )) {
            if combined.len() >= k_cap {
                break;
            }
            if let Some(s) = sanitize(r, th)
                && !combined
                    .iter()
                    .any(|&(pr, pt, _)| poles_too_close(pr, pt, s.0, s.1))
            {
                combined.push(s);
            }
        }
        for s in base_poles.drain(..) {
            if combined.len() >= k_cap {
                break;
            }
            if !combined
                .iter()
                .any(|&(pr, pt, _)| poles_too_close(pr, pt, s.0, s.1))
            {
                combined.push(s);
            }
        }
        if combined.len() > acc_a.len().min(keep_n) {
            let (acc_b, out_bl, out_br, err_b, _) = solve_joint_kautz_candidates(
                &combined,
                inp,
                &blocks,
                [kept_locks[0].min(keep_n), kept_locks[1].min(keep_n)],
            );
            let cur_best_err = best.as_ref().map_or(before_e * 0.995, |b| b.3);
            if err_b < cur_best_err && !acc_b.is_empty() {
                best = Some((acc_b, out_bl, out_br, err_b));
            }
        }
    }

    if let Some((acc, out_l, out_r, _)) = best {
        (acc, out_l, out_r)
    } else {
        (Vec::new(), res_in_l[..n].to_vec(), res_in_r[..n].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kautz_2pass_omp_recovers_r0999_and_50hz_modes() {
        let n = 8_192_usize;
        let sr = 48_000.0_f64;
        let mut x = vec![0.0_f64; n];
        let mut seed = 0x5555_AAAA_1111_9999_u64;
        for v in &mut x {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = 2.0 * (((seed >> 33) as f64) / ((1_u64 << 31) as f64)) - 1.0;
        }

        let mut onset_x = vec![0.0_f64; n];
        onset_x[16..].copy_from_slice(&x[..(n - 16)]);

        let (r1, th1) = (0.9990_f64, TAU * 1_400.0 / sr);
        let (r2, th2) = (0.995_f64, TAU * 50.0 / sr);
        let (mr1, mi1) = synth_kautz_unit_mode(r1, th1, &onset_x);
        let (mr2, mi2) = synth_kautz_unit_mode(r2, th2, &onset_x);
        let s1 = 2.0 * h2_norm_b_gamma(r1);
        let s2 = 2.0 * h2_norm_b_gamma(r2);
        let mut y = vec![0.0_f64; n];
        for t in 0..n {
            y[t] = s1 * (0.4 * mr1[t] - 0.15 * mi1[t]) + s2 * (-0.35 * mr2[t] + 0.2 * mi2[t]);
        }

        let mut cands = vec![(r1, th1)];
        append_log_octave_h2_candidates(&mut cands, sr, 24);
        let (accepted, res_l, _) = fit_joint_kautz_bank(&KautzFitInputs {
            candidates: &cands,
            input_l: &x,
            input_r: &x,
            res_in_l: &y,
            res_in_r: &y,
            start: 32,
            k_limit: 24,
            sample_rate_hz: sr,
            r_max: MAX_PASS1_KAUTZ_R,
            lock_prefix_lens: [0, 0],
        });
        assert!(!accepted.is_empty());
        assert!(accepted.iter().any(|p| p.r_low > 0.997));
        let y_e: f64 = y[32..].iter().map(|v| v * v).sum();
        let r_e: f64 = res_l[32..].iter().map(|v| v * v).sum();
        assert!(
            r_e / y_e < 0.05,
            "2-pass Kautz OMP residual ratio={}",
            r_e / y_e
        );

        // Unexcited r = 0.9992 pole on a short ill-conditioned window obeys the BIBO peak-gain bound.
        let short_n = 256_usize;
        let short_x: Vec<f64> = (0..short_n)
            .map(|i| 0.25 * (TAU * 220.0 * (i as f64) / sr).sin())
            .collect();
        let short_y: Vec<f64> = short_x.iter().map(|&v| 0.8 * v).collect();
        let mut cands_bibo = vec![(0.9992_f64, TAU * 13_300.0 / sr)];
        append_log_octave_h2_candidates(&mut cands_bibo, sr, 16);
        let (acc_bibo, _, _) = fit_joint_kautz_bank(&KautzFitInputs {
            candidates: &cands_bibo,
            input_l: &short_x,
            input_r: &short_x,
            res_in_l: &short_y,
            res_in_r: &short_y,
            start: 16,
            k_limit: 16,
            sample_rate_hz: sr,
            r_max: MAX_PASS1_KAUTZ_R,
            lock_prefix_lens: [0, 0],
        });
        for p in &acc_bibo {
            let peak_l = 2.0 * p.b_gamma * p.c_re_l.hypot(p.c_im_l) / (1.0 - p.r_low).max(1.0e-6);
            assert!(
                peak_l <= MAX_PASS1_KAUTZ_PEAK_GAIN + 1.0e-9,
                "kautz peak_l={peak_l} exceeded {MAX_PASS1_KAUTZ_PEAK_GAIN}"
            );
        }

        // Physical r_max estimation: continuous or long-tail stays MAX_PASS1_KAUTZ_R, while short-decay
        // gaps derive a bounded ceiling < 0.9985; ultrasonic (> 19.5 kHz) candidates are rejected.
        assert_eq!(
            estimate_physical_kautz_r_ceil(&x, &x, &y, &y, 32, sr),
            MAX_PASS1_KAUTZ_R
        );
        let mut dry_g = vec![0.0_f64; 16_384];
        let mut tgt_short = vec![0.0_f64; 16_384];
        let mut tgt_long = vec![0.0_f64; 16_384];
        for i in 0..16_384 {
            if !(2_400..12_000).contains(&i) {
                let s = 0.3 * (TAU * 350.0 * (i as f64) / sr).sin();
                dry_g[i] = s;
                tgt_short[i] = s;
                tgt_long[i] = s;
            } else {
                tgt_short[i] = if i < 2_600 { 1.0e-2 } else { 0.0 };
                tgt_long[i] = 0.02 * (-((i - 2_400) as f64) / 3_500.0).exp();
            }
        }
        let r_short =
            estimate_physical_kautz_r_ceil(&dry_g, &dry_g, &tgt_short, &tgt_short, 32, sr);
        let r_long = estimate_physical_kautz_r_ceil(&dry_g, &dry_g, &tgt_long, &tgt_long, 32, sr);
        assert!(
            (MIN_PHYS_KAUTZ_R_CEIL..0.9985).contains(&r_short),
            "r_short={r_short}"
        );
        let tgt_short_dc: Vec<f64> = tgt_short.iter().map(|&v| v + 0.08).collect();
        let r_short_dc =
            estimate_physical_kautz_r_ceil(&dry_g, &dry_g, &tgt_short_dc, &tgt_short_dc, 32, sr);
        assert!(
            (MIN_PHYS_KAUTZ_R_CEIL..0.9985).contains(&r_short_dc),
            "r_short_dc={r_short_dc}"
        );
        assert_eq!(r_long, MAX_PASS1_KAUTZ_R);

        let th_ultra = TAU * 22_000.0 / sr;
        let lock_cands = vec![(0.95_f64, TAU * 250.0 / sr), (0.9992, th_ultra), (r1, th1)];
        let (acc_locked, _, _) = fit_joint_kautz_bank(&KautzFitInputs {
            candidates: &lock_cands,
            input_l: &x,
            input_r: &x,
            res_in_l: &y,
            res_in_r: &y,
            start: 32,
            k_limit: 8,
            sample_rate_hz: sr,
            r_max: r_short,
            lock_prefix_lens: [1, 2],
        });
        assert!(
            acc_locked
                .iter()
                .all(|p| p.theta < th_ultra && p.r_low <= r_short + 1.0e-12)
        );
        if !acc_locked.is_empty() {
            assert!(
                (acc_locked[0].theta - TAU * 250.0 / sr).abs() < 1.0e-3
                    || acc_locked[0].theta == th1
            );
        }
    }
}
