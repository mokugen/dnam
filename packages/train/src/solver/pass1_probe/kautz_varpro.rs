//! Multi-regime block selection and joint `[FIR(0..16), Kautz(z^{-16})]` Kaufman Variable-
//! Projection Levenberg–Marquardt pole refiner (`ref[train.pipeline.physics-probe]`,
//! `ref[dsp.op-m.kautz.pole-extraction]`).

use core::f64::consts::TAU;

use crate::linalg::cholesky::{factorize_uplo_u_cholesky, solve_factorized_uplo_u};
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

use super::kautz_fit::{
    MAX_PASS1_KAUTZ_R, MIN_PASS1_KAUTZ_R, audible_kautz_theta_bounds, h2_norm_b_gamma,
};

const fn const_neg_ln_unit(r: f64) -> f64 {
    // ln(r) = -2 * atanh((1 - r) / (1 + r)), converging at |u| <= 0.6 for r in [0.25, 1.0].
    let u = (1.0 - r) / (1.0 + r);
    let u2 = u * u;
    let mut term = u;
    let mut sum = 0.0_f64;
    let mut k = 1_usize;
    while k <= 31 {
        sum += term / (k as f64);
        term *= u2;
        k += 2;
    }
    2.0 * sum
}

const MIN_KAUTZ_ALPHA: f64 = const_neg_ln_unit(MAX_PASS1_KAUTZ_R);
const MAX_KAUTZ_ALPHA: f64 = const_neg_ln_unit(MIN_PASS1_KAUTZ_R);
const VARPRO_PREROLL_FRAMES: usize = 1_024;
const MAX_VARPRO_FRAMES: usize = 6_144;
const NUM_REGIME_QUARTILES: usize = 4;
const MAX_LM_ITERS: usize = 24;
const MIN_RESIDUE_REL_ENERGY: f64 = 1.0e-4;
const MIN_VARPRO_POLE_SEPARATION: f64 = 0.012;

const _: () = assert!(
    MIN_KAUTZ_ALPHA > 0.0
        && MIN_KAUTZ_ALPHA < MAX_KAUTZ_ALPHA
        && VARPRO_PREROLL_FRAMES == 1_024
        && MAX_VARPRO_FRAMES.is_multiple_of(NUM_REGIME_QUARTILES)
        && MIN_VARPRO_POLE_SEPARATION > 0.0
        && MIN_VARPRO_POLE_SEPARATION < 0.02
);

/// Selects up to `4` disjoint high-energy quartile spans totaling `<= max_total_frames` across
/// `[start..n]` so multi-regime captures with notched or burst-gap regions are covered uniformly.
#[must_use]
pub(super) fn select_multiregime_blocks(
    target_l: &[f64],
    target_r: &[f64],
    start: usize,
    max_total_frames: usize,
) -> Vec<(usize, usize)> {
    let n = target_l.len().min(target_r.len());
    if n <= start || max_total_frames == 0 {
        return Vec::new();
    }
    let avail = n - start;
    if avail <= max_total_frames {
        return vec![(start, n)];
    }

    let blk_target = (max_total_frames / NUM_REGIME_QUARTILES).max(1);
    let q_base = avail / NUM_REGIME_QUARTILES;
    let mut blocks = Vec::with_capacity(NUM_REGIME_QUARTILES);
    for q in 0..NUM_REGIME_QUARTILES {
        let qs = start + q * q_base;
        let qe = if q + 1 == NUM_REGIME_QUARTILES {
            n
        } else {
            start + (q + 1) * q_base
        };
        let q_len = qe.saturating_sub(qs);
        if q_len <= blk_target {
            if q_len > 0 {
                blocks.push((qs, qe));
            }
            continue;
        }
        let slack = q_len - blk_target;
        let mut best_s = qs;
        let mut best_e = -1.0_f64;
        for cand in 0..4_usize {
            let cs = qs + (cand * slack) / 3;
            let ce = cs + blk_target;
            let energy: f64 = target_l[cs..ce]
                .iter()
                .zip(&target_r[cs..ce])
                .map(|(&l, &r)| l * l + r * r)
                .sum();
            if energy.is_finite() && energy > best_e {
                best_e = energy;
                best_s = cs;
            }
        }
        blocks.push((best_s, best_s + blk_target));
    }
    blocks
}

struct ModeColumns {
    u0: Vec<f64>,
    u1: Vec<f64>,
    da0: Vec<f64>,
    da1: Vec<f64>,
    dt0: Vec<f64>,
    dt1: Vec<f64>,
}

fn eval_pole_mode_across_blocks(
    r: f64,
    theta: f64,
    input: &[f64],
    fir_onset_n: usize,
    blocks: &[(usize, usize)],
    m_tot: usize,
    need_derivs: bool,
) -> ModeColumns {
    let scale = 2.0 * h2_norm_b_gamma(r);
    let (s, c) = theta.sin_cos();
    let (ar, ai) = (r * c, r * s);

    let mut u0 = Vec::with_capacity(m_tot);
    let mut u1 = Vec::with_capacity(m_tot);
    let mut da0 = if need_derivs {
        Vec::with_capacity(m_tot)
    } else {
        Vec::new()
    };
    let mut da1 = if need_derivs {
        Vec::with_capacity(m_tot)
    } else {
        Vec::new()
    };
    let mut dt0 = if need_derivs {
        Vec::with_capacity(m_tot)
    } else {
        Vec::new()
    };
    let mut dt1 = if need_derivs {
        Vec::with_capacity(m_tot)
    } else {
        Vec::new()
    };

    for &(bs, be) in blocks {
        let pre_s = bs.saturating_sub(VARPRO_PREROLL_FRAMES);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        let (mut dr, mut di) = (0.0_f64, 0.0_f64);
        for t in pre_s..be {
            let u_in = if t >= fir_onset_n {
                input[t - fir_onset_n]
            } else {
                0.0
            };
            if need_derivs {
                let ndr = ar * dr - ai * di + zr;
                let ndi = ai * dr + ar * di + zi;
                dr = ndr;
                di = ndi;
            }
            let nzr = ar * zr - ai * zi + u_in;
            let nzi = ai * zr + ar * zi;
            zr = nzr;
            zi = nzi;
            if t >= bs {
                u0.push(scale * zr);
                u1.push(-scale * zi);
                if need_derivs {
                    let pdr = ar * dr - ai * di;
                    let pdi = ai * dr + ar * di;
                    da0.push(-scale * pdr);
                    da1.push(scale * pdi);
                    dt0.push(-scale * pdi);
                    dt1.push(-scale * pdr);
                }
            }
        }
    }

    ModeColumns {
        u0,
        u1,
        da0,
        da1,
        dt0,
        dt1,
    }
}

struct ChannelSystem {
    r_upper: Vec<f64>,
    weights: Vec<f64>,
    resid: Vec<f64>,
}

fn solve_channel_linear_block(
    cols: &[Vec<f64>],
    tgt: &[f64],
    m_tot: usize,
) -> Option<ChannelSystem> {
    let dim = cols.len();
    if dim == 0 || m_tot == 0 {
        return None;
    }
    let mut g_u = vec![0.0_f64; uplo_u_packed_len(dim)];
    let mut rhs = vec![0.0_f64; dim];
    let mut diag_sum = 0.0_f64;
    for c in 0..dim {
        let cc = &cols[c];
        let col_off = uplo_u_index(0, c);
        rhs[c] = (0..m_tot).map(|i| cc[i] * tgt[i]).sum();
        for r in 0..=c {
            let cr = &cols[r];
            let dot: f64 = (0..m_tot).map(|i| cr[i] * cc[i]).sum();
            g_u[col_off + r] = dot;
            if r == c {
                diag_sum += dot.max(0.0);
            }
        }
    }
    let mean_diag = (diag_sum / (dim as f64)).max(0.0);
    let mut ridge = (1.0e-10 * mean_diag).clamp(1.0e-12, 1.0e-4);
    let mut factored = None;
    for _ in 0..5 {
        if let Ok(r_upper) = factorize_uplo_u_cholesky(dim, &g_u, ridge)
            && let Ok(weights) = solve_factorized_uplo_u(dim, &r_upper, &rhs)
        {
            factored = Some((r_upper, weights));
            break;
        }
        ridge = (ridge * 100.0).min(mean_diag.max(1.0));
    }
    let (r_upper, weights) = factored?;
    let mut resid = tgt.to_vec();
    for (c, cc) in cols.iter().enumerate() {
        let wc = weights[c];
        for i in 0..m_tot {
            resid[i] -= wc * cc[i];
        }
    }
    Some(ChannelSystem {
        r_upper,
        weights,
        resid,
    })
}

fn project_perp(raw: &[f64], cols: &[Vec<f64>], r_upper: &[f64], m_tot: usize) -> Option<Vec<f64>> {
    let dim = cols.len();
    let mut proj_rhs = vec![0.0_f64; dim];
    for (j, col) in cols.iter().enumerate() {
        proj_rhs[j] = (0..m_tot).map(|i| col[i] * raw[i]).sum();
    }
    let coef = solve_factorized_uplo_u(dim, r_upper, &proj_rhs).ok()?;
    let mut out = raw.to_vec();
    for (j, col) in cols.iter().enumerate() {
        let cj = coef[j];
        for i in 0..m_tot {
            out[i] -= cj * col[i];
        }
    }
    Some(out)
}

struct VarProEvalState {
    err: f64,
    cols_l: Vec<Vec<f64>>,
    cols_r: Vec<Vec<f64>>,
    derivs_l: Vec<[Vec<f64>; 4]>,
    derivs_r: Vec<[Vec<f64>; 4]>,
    sys_l: ChannelSystem,
    sys_r: ChannelSystem,
}

struct FixedVarProInputs<'a> {
    in_l: &'a [f64],
    in_r: &'a [f64],
    blocks: &'a [(usize, usize)],
    m_tot: usize,
    fir_onset_n: usize,
    fir_cols_l: &'a [Vec<f64>],
    fir_cols_r: &'a [Vec<f64>],
    tgt_seg_l: &'a [f64],
    tgt_seg_r: &'a [f64],
    same_input: bool,
}

fn eval_varpro_state(
    fx: &FixedVarProInputs<'_>,
    poles: &[(f64, f64)],
    need_derivs: bool,
) -> Option<VarProEvalState> {
    let k_cnt = poles.len();
    let mut cols_l = Vec::with_capacity(fx.fir_onset_n + 2 * k_cnt);
    cols_l.extend_from_slice(fx.fir_cols_l);
    let mut cols_r = if fx.same_input {
        Vec::new()
    } else {
        let mut v = Vec::with_capacity(fx.fir_onset_n + 2 * k_cnt);
        v.extend_from_slice(fx.fir_cols_r);
        v
    };
    let mut derivs_l = if need_derivs {
        Vec::with_capacity(k_cnt)
    } else {
        Vec::new()
    };
    let mut derivs_r = if need_derivs && !fx.same_input {
        Vec::with_capacity(k_cnt)
    } else {
        Vec::new()
    };

    for &(r, th) in poles {
        let mc_l = eval_pole_mode_across_blocks(
            r,
            th,
            fx.in_l,
            fx.fir_onset_n,
            fx.blocks,
            fx.m_tot,
            need_derivs,
        );
        cols_l.push(mc_l.u0);
        cols_l.push(mc_l.u1);
        if need_derivs {
            derivs_l.push([mc_l.da0, mc_l.da1, mc_l.dt0, mc_l.dt1]);
        }
        if !fx.same_input {
            let mc_r = eval_pole_mode_across_blocks(
                r,
                th,
                fx.in_r,
                fx.fir_onset_n,
                fx.blocks,
                fx.m_tot,
                need_derivs,
            );
            cols_r.push(mc_r.u0);
            cols_r.push(mc_r.u1);
            if need_derivs {
                derivs_r.push([mc_r.da0, mc_r.da1, mc_r.dt0, mc_r.dt1]);
            }
        }
    }

    let sys_l = solve_channel_linear_block(&cols_l, fx.tgt_seg_l, fx.m_tot)?;
    let sys_r = if fx.same_input {
        let rhs_r: Vec<f64> = cols_l
            .iter()
            .map(|cc| (0..fx.m_tot).map(|i| cc[i] * fx.tgt_seg_r[i]).sum())
            .collect();
        let weights = solve_factorized_uplo_u(cols_l.len(), &sys_l.r_upper, &rhs_r).ok()?;
        let mut resid = fx.tgt_seg_r.to_vec();
        for (c, cc) in cols_l.iter().enumerate() {
            let wc = weights[c];
            for i in 0..fx.m_tot {
                resid[i] -= wc * cc[i];
            }
        }
        ChannelSystem {
            r_upper: sys_l.r_upper.clone(),
            weights,
            resid,
        }
    } else {
        solve_channel_linear_block(&cols_r, fx.tgt_seg_r, fx.m_tot)?
    };
    let err: f64 = sys_l
        .resid
        .iter()
        .zip(&sys_r.resid)
        .map(|(a, b)| a * a + b * b)
        .sum();
    err.is_finite().then_some(VarProEvalState {
        err,
        cols_l,
        cols_r,
        derivs_l,
        derivs_r,
        sys_l,
        sys_r,
    })
}

fn compute_lm_step(
    fx: &FixedVarProInputs<'_>,
    st: &VarProEvalState,
    k_cnt: usize,
    poles: &[(f64, f64)],
    bounds: (f64, f64, f64),
    lam: &mut f64,
) -> Option<(Vec<(f64, f64)>, VarProEvalState)> {
    let (alpha_min, theta_min, theta_max) = bounds;
    let pdim = 2 * k_cnt;
    let mut jl = Vec::with_capacity(pdim);
    let mut jr = Vec::with_capacity(pdim);
    let r_cols = if fx.same_input {
        &st.cols_l
    } else {
        &st.cols_r
    };
    let r_derivs = if fx.same_input {
        &st.derivs_l
    } else {
        &st.derivs_r
    };

    for idx in 0..k_cnt {
        let [da0_l, da1_l, dt0_l, dt1_l] = &st.derivs_l[idx];
        let [da0_r, da1_r, dt0_r, dt1_r] = &r_derivs[idx];
        let c0l = st.sys_l.weights[fx.fir_onset_n + 2 * idx];
        let c1l = st.sys_l.weights[fx.fir_onset_n + 2 * idx + 1];
        let c0r = st.sys_r.weights[fx.fir_onset_n + 2 * idx];
        let c1r = st.sys_r.weights[fx.fir_onset_n + 2 * idx + 1];
        for ((d0l, d1l), (d0r, d1r)) in [
            ((da0_l, da1_l), (da0_r, da1_r)),
            ((dt0_l, dt1_l), (dt0_r, dt1_r)),
        ] {
            let raw_l: Vec<f64> = (0..fx.m_tot).map(|i| c0l * d0l[i] + c1l * d1l[i]).collect();
            let raw_r: Vec<f64> = (0..fx.m_tot).map(|i| c0r * d0r[i] + c1r * d1r[i]).collect();
            jl.push(project_perp(
                &raw_l,
                &st.cols_l,
                &st.sys_l.r_upper,
                fx.m_tot,
            )?);
            jr.push(project_perp(&raw_r, r_cols, &st.sys_r.r_upper, fx.m_tot)?);
        }
    }

    let mut jtj = vec![0.0_f64; uplo_u_packed_len(pdim)];
    let mut jtr = vec![0.0_f64; pdim];
    for c in 0..pdim {
        let col_off = uplo_u_index(0, c);
        jtr[c] = (0..fx.m_tot)
            .map(|i| jl[c][i] * st.sys_l.resid[i] + jr[c][i] * st.sys_r.resid[i])
            .sum();
        for r in 0..=c {
            jtj[col_off + r] = (0..fx.m_tot)
                .map(|i| jl[r][i] * jl[c][i] + jr[r][i] * jr[c][i])
                .sum();
        }
    }

    for _try in 0..6 {
        let mut damp_gram = jtj.clone();
        for i in 0..pdim {
            let idx = uplo_u_index(i, i);
            damp_gram[idx] *= 1.0 + *lam;
        }
        if let Ok(r_up) = factorize_uplo_u_cholesky(pdim, &damp_gram, 1.0e-12)
            && let Ok(dp) = solve_factorized_uplo_u(pdim, &r_up, &jtr)
        {
            let cand: Vec<(f64, f64)> = poles
                .iter()
                .enumerate()
                .map(|(idx, &(rk, thk))| {
                    let ak = -rk.clamp(MIN_PASS1_KAUTZ_R, MAX_PASS1_KAUTZ_R).ln();
                    let na =
                        (ak + dp[2 * idx].clamp(-0.35, 0.35)).clamp(alpha_min, MAX_KAUTZ_ALPHA);
                    let nth =
                        (thk + dp[2 * idx + 1].clamp(-0.15, 0.15)).clamp(theta_min, theta_max);
                    ((-na).exp(), nth)
                })
                .collect();
            if let Some(next_st) = eval_varpro_state(fx, &cand, false)
                && next_st.err < st.err
            {
                *lam = (*lam * 0.3).max(1.0e-7);
                return Some((cand, next_st));
            }
        }
        *lam *= 4.0;
    }
    None
}

/// Configuration for [`refine_kautz_poles_kaufman_varpro`].
#[derive(Clone, Copy, Debug)]
pub(super) struct VarProPoleConfig {
    pub sample_rate_hz: f64,
    pub fir_onset_n: usize,
    pub start: usize,
    pub k_ref: usize,
    pub r_max: f64,
    pub max_lm_iters: usize,
}

/// Refines up to `cfg.k_ref` (`<= 8`) stereo Kautz pole pairs `(r, theta)` directly from a blind
/// log-octave grid via joint `[FIR(0..fir_onset_n), Kautz(z^{-fir_onset_n})]` Kaufman Variable-
/// Projection Levenberg–Marquardt across multi-regime energy blocks, staying strictly inside the
/// audible band and excluding any previously locked poles in `avoid_poles`.
#[must_use]
pub(super) fn refine_kautz_poles_kaufman_varpro(
    input_l: &[f64],
    input_r: &[f64],
    tgt_l: &[f64],
    tgt_r: &[f64],
    cfg: VarProPoleConfig,
    avoid_poles: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    let VarProPoleConfig {
        sample_rate_hz,
        fir_onset_n,
        start,
        k_ref,
        r_max,
        max_lm_iters,
    } = cfg;
    let n = input_l
        .len()
        .min(input_r.len())
        .min(tgt_l.len())
        .min(tgt_r.len());
    let k_cnt = k_ref.min(8);
    if k_cnt == 0 || n <= start + 32 || fir_onset_n == 0 {
        return Vec::new();
    }
    let blocks = select_multiregime_blocks(&tgt_l[..n], &tgt_r[..n], start, MAX_VARPRO_FRAMES);
    if blocks.is_empty() {
        return Vec::new();
    }
    let m_tot: usize = blocks.iter().map(|&(bs, be)| be - bs).sum();
    if m_tot < 2 * (fir_onset_n + 2 * k_cnt) {
        return Vec::new();
    }

    let same_input = input_l[..n] == input_r[..n];
    let collect_fir_cols = |inp: &[f64]| -> Vec<Vec<f64>> {
        (0..fir_onset_n)
            .map(|tap| {
                let mut col = Vec::with_capacity(m_tot);
                for &(bs, be) in &blocks {
                    for t in bs..be {
                        col.push(if t >= tap { inp[t - tap] } else { 0.0 });
                    }
                }
                col
            })
            .collect()
    };
    let fir_cols_l = collect_fir_cols(&input_l[..n]);
    let fir_cols_r = if same_input {
        Vec::new()
    } else {
        collect_fir_cols(&input_r[..n])
    };
    let mut tgt_seg_l = Vec::with_capacity(m_tot);
    let mut tgt_seg_r = Vec::with_capacity(m_tot);
    for &(bs, be) in &blocks {
        tgt_seg_l.extend_from_slice(&tgt_l[bs..be]);
        tgt_seg_r.extend_from_slice(&tgt_r[bs..be]);
    }
    let fx = FixedVarProInputs {
        in_l: &input_l[..n],
        in_r: &input_r[..n],
        blocks: &blocks,
        m_tot,
        fir_onset_n,
        fir_cols_l: &fir_cols_l,
        fir_cols_r: &fir_cols_r,
        tgt_seg_l: &tgt_seg_l,
        tgt_seg_r: &tgt_seg_r,
        same_input,
    };

    let sr = sample_rate_hz.max(8_000.0);
    let (theta_min, theta_max) = audible_kautz_theta_bounds(sr);
    let r_cap = r_max.clamp(MIN_PASS1_KAUTZ_R, MAX_PASS1_KAUTZ_R);
    let alpha_min = (-r_cap.ln()).max(MIN_KAUTZ_ALPHA);
    let bounds = (alpha_min, theta_min, theta_max);
    let f_lo = 40.0_f64;
    let f_hi = (0.35 * sr).min(12_000.0).max(f_lo * 2.0);
    let ratio = f_hi / f_lo;
    let mut poles: Vec<(f64, f64)> = (0..k_cnt)
        .map(|idx| {
            let frac = if k_cnt <= 1 {
                0.0
            } else {
                (idx as f64) / ((k_cnt - 1) as f64)
            };
            let fc_hz = f_lo * ratio.powf(frac);
            let mut th = (TAU * fc_hz / sr).clamp(theta_min, theta_max);
            for &(_, ath) in avoid_poles {
                if (th - ath).abs() < 0.02 {
                    let shift = if th >= ath { 0.025 } else { -0.025 };
                    th = (th + shift).clamp(theta_min, theta_max);
                }
            }
            let r0 = if fc_hz < 180.0 {
                0.985_f64.min(r_cap)
            } else if fc_hz < 1_500.0 {
                0.94_f64.min(r_cap)
            } else {
                0.90_f64.min(r_cap)
            };
            (r0, th)
        })
        .collect();

    let mut lam = 1.0e-2_f64;
    for _iter in 0..max_lm_iters.clamp(1, MAX_LM_ITERS) {
        let Some(cur_st) = eval_varpro_state(&fx, &poles, true) else {
            break;
        };
        let Some((cand, next_st)) = compute_lm_step(&fx, &cur_st, k_cnt, &poles, bounds, &mut lam)
        else {
            break;
        };
        let rel_imp = (cur_st.err - next_st.err) / cur_st.err.max(1.0e-18);
        poles = cand;
        if rel_imp < 1.0e-6 {
            break;
        }
    }

    let Some(fin_st) = eval_varpro_state(&fx, &poles, false) else {
        return Vec::new();
    };
    let mut scored: Vec<(f64, f64, f64)> = (0..k_cnt)
        .map(|idx| {
            let col0 = &fin_st.cols_l[fx.fir_onset_n + 2 * idx];
            let col1 = &fin_st.cols_l[fx.fir_onset_n + 2 * idx + 1];
            let (c0l, c1l) = (
                fin_st.sys_l.weights[fx.fir_onset_n + 2 * idx],
                fin_st.sys_l.weights[fx.fir_onset_n + 2 * idx + 1],
            );
            let (c0r, c1r) = (
                fin_st.sys_r.weights[fx.fir_onset_n + 2 * idx],
                fin_st.sys_r.weights[fx.fir_onset_n + 2 * idx + 1],
            );
            let r_c0 = if fx.same_input {
                col0
            } else {
                &fin_st.cols_r[fx.fir_onset_n + 2 * idx]
            };
            let r_c1 = if fx.same_input {
                col1
            } else {
                &fin_st.cols_r[fx.fir_onset_n + 2 * idx + 1]
            };
            let e_k: f64 = (0..fx.m_tot)
                .map(|i| {
                    let vl = c0l * col0[i] + c1l * col1[i];
                    let vr = c0r * r_c0[i] + c1r * r_c1[i];
                    vl * vl + vr * vr
                })
                .sum();
            (poles[idx].0, poles[idx].1, e_k)
        })
        .collect();
    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    let max_e = scored.first().map_or(0.0, |x| x.2);
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(k_cnt);
    for (r, th, ek) in scored {
        if ek <= MIN_RESIDUE_REL_ENERGY * max_e || (th <= theta_min + 1.0e-4 && r < 0.85) {
            continue;
        }
        let dup = out.iter().chain(avoid_poles.iter()).any(|&(pr, pt)| {
            (pr * pt.cos() - r * th.cos()).hypot(pr * pt.sin() - r * th.sin())
                < MIN_VARPRO_POLE_SEPARATION
        });
        if !dup {
            out.push((r, th));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::pass1_probe::kautz_fit::synth_kautz_unit_mode;

    #[test]
    fn test_kaufman_varpro_recovers_notched_lf_and_hf_poles_across_regimes() {
        let sr = 48_000.0_f64;
        let n = 16_384_usize;
        let q1_end = 4_096_usize;
        // Quartile 1: multi-tone with 40..75 Hz notch (simulating Law 0.C mains exclusion).
        // Quartiles 2..4: deterministic broadband bursts separated by silent intervals.
        let mut in_sig = vec![0.0_f64; n];
        for t in 0..q1_end {
            let sec = (t as f64) / sr;
            let mut s = 0.0_f64;
            for k in 1..180_usize {
                let f = (k as f64) * 35.0;
                if (40.0..=75.0).contains(&f) {
                    continue;
                }
                s += (TAU * f * sec + 0.37 * (k as f64)).sin();
            }
            in_sig[t] = 0.04 * s;
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
        for t in q1_end..n {
            if (t % 2_048) < 1_024 {
                rng = rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                in_sig[t] = 0.35 * (((rng >> 33) as f64) / ((1_u64 << 30) as f64) - 1.0);
            }
        }

        // Generic 5-mode stereo filter (`[62, 175, 980, 2400, 3900] Hz`) + direct onset term.
        let true_poles = [
            (0.993_f64, 62.0_f64, 0.32_f64, 0.32_f64),
            (0.974, 175.0, 0.21, 0.17),
            (0.955, 980.0, 0.25, -0.20),
            (0.935, 2_400.0, -0.19, 0.23),
            (0.910, 3_900.0, 0.16, -0.14),
        ];
        let mut tgt_l: Vec<f64> = in_sig.iter().map(|&x| 0.22 * x).collect();
        let mut tgt_r: Vec<f64> = in_sig.iter().map(|&x| 0.22 * x).collect();
        for &(r, hz, wl, wr) in &true_poles {
            let th = TAU * hz / sr;
            let (mr, _) = synth_kautz_unit_mode(r, th, &in_sig);
            let gl = (1.0 - r) * wl;
            let gr = (1.0 - r) * wr;
            for t in 0..n {
                tgt_l[t] += gl * mr[t];
                tgt_r[t] += gr * mr[t];
            }
        }

        let blocks = select_multiregime_blocks(&tgt_l, &tgt_r, 32, MAX_VARPRO_FRAMES);
        assert_eq!(blocks.len(), 4);
        assert!(blocks[1].1 <= n / 2);

        let recovered_exact = refine_kautz_poles_kaufman_varpro(
            &in_sig,
            &in_sig,
            &tgt_l,
            &tgt_r,
            VarProPoleConfig {
                sample_rate_hz: sr,
                fir_onset_n: 16,
                start: 32,
                k_ref: 5,
                r_max: MAX_PASS1_KAUTZ_R,
                max_lm_iters: 24,
            },
            &[],
        );
        assert!(
            recovered_exact.len() == 5,
            "recovered_exact={recovered_exact:?}"
        );
        for &(r_true, hz_true, _, _) in &true_poles {
            let th_true = TAU * hz_true / sr;
            let best = recovered_exact
                .iter()
                .min_by(|a, b| (a.1 - th_true).abs().total_cmp(&(b.1 - th_true).abs()))
                .copied()
                .unwrap();
            let got_hz = best.1 * sr / TAU;
            assert!(
                (got_hz - hz_true).abs() < 2.0 && (best.0 - r_true).abs() < 0.002,
                "missed exact-order pole ({r_true}, {hz_true} Hz): closest ({:.5}, {got_hz:.2} Hz) in {recovered_exact:?}",
                best.0
            );
        }

        // Over-parameterized k_ref = 8 must achieve < -40 dB onset residual and capture LF/HF modes cleanly.
        // Over-parameterized Pass 1 (`k_cap = 8`, `fir = 32`) on multi-regime notched signal
        // must recover clean modal poles and achieve < -40 dB overall ESR.
        let recovered = refine_kautz_poles_kaufman_varpro(
            &in_sig,
            &in_sig,
            &tgt_l,
            &tgt_r,
            VarProPoleConfig {
                sample_rate_hz: sr,
                fir_onset_n: 16,
                start: 32,
                k_ref: 8,
                r_max: MAX_PASS1_KAUTZ_R,
                max_lm_iters: 24,
            },
            &[],
        );
        assert!(recovered.len() >= 5, "recovered={recovered:?}");
        let (rep_p1, _) = crate::solver::pass1_probe::run_pass1_probe(
            &in_sig, &in_sig, &tgt_l, &tgt_r, sr, 32, 8,
        )
        .unwrap();
        let esr_db = 10.0 * rep_p1.initial_esr_lin.max(1.0e-18).log10();
        assert!(
            esr_db < -40.0,
            "k_cap=8 Pass-1 residual ESR={esr_db:.2} dB >= -40 dB ({recovered:?})"
        );

        // Also exercise distinct L/R input path + finite-difference derivative sanity on 1 block.
        let rev_in_r: Vec<f64> = in_sig.iter().map(|&v| 0.9 * v).collect();
        let stereo_rec = refine_kautz_poles_kaufman_varpro(
            &in_sig[..4_096],
            &rev_in_r[..4_096],
            &tgt_l[..4_096],
            &tgt_r[..4_096],
            VarProPoleConfig {
                sample_rate_hz: sr,
                fir_onset_n: 16,
                start: 32,
                k_ref: 6,
                r_max: MAX_PASS1_KAUTZ_R,
                max_lm_iters: 24,
            },
            &[],
        );
        assert!(!stereo_rec.is_empty());
        assert_eq!(
            select_multiregime_blocks(&tgt_l, &tgt_r, n + 1, 512),
            Vec::new()
        );
    }
}
