#![forbid(unsafe_code)]
//! Lag-compensated causal `8`-tap FIR Variable-Projection + Gauss-Newton Stage 0 (`Op-P`)
//! parameter identification across body (`stem0`) and monophonic sweep (`stem1`) waveforms.

use dnam_engine::stages::router::{V3Stage0Hot, V3Stage0Params};

use super::super::model_f64::{V3TrainDataset, V3TrainStem};
use crate::solver::canonical_besr_warmup;

pub(super) const MAX_PROBE_BODY_SAMPLES: usize = 16_384;
pub(super) const FIR_TAPS: usize = 8;

#[derive(Clone, Debug)]
pub(super) struct SubOctFeatures {
    pub x_g: Vec<f64>,
    pub abs_x: Vec<f64>,
    pub env: Vec<f64>,
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    pub tgt: Vec<f64>,
    pub valid_mask: Vec<bool>,
}

/// Solves `A * x = b` via partial-pivoted Gaussian elimination.
pub(super) fn solve_small_linear<const N: usize>(
    mut a: [[f64; N]; N],
    mut b: [f64; N],
) -> Option<[f64; N]> {
    for col in 0..N {
        let mut piv_row = col;
        let mut piv_abs = a[col][col].abs();
        for (row, r_vals) in a.iter().enumerate().skip(col + 1) {
            let v = r_vals[col].abs();
            if v > piv_abs {
                piv_abs = v;
                piv_row = row;
            }
        }
        if !piv_abs.is_finite() || piv_abs < 1.0e-12 {
            return None;
        }
        if piv_row != col {
            a.swap(col, piv_row);
            b.swap(col, piv_row);
        }
        let inv_piv = 1.0 / a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] * inv_piv;
            if factor == 0.0 {
                continue;
            }
            for c in col..N {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0_f64; N];
    for col in (0..N).rev() {
        let mut sum = b[col];
        for c in (col + 1)..N {
            sum -= a[col][c] * x[c];
        }
        x[col] = sum / a[col][col];
        if !x[col].is_finite() {
            return None;
        }
    }
    Some(x)
}

pub(super) fn ridge_ls_r2<const N: usize>(
    rows: &[[f64; N]],
    tgt: &[f64],
) -> Option<([f64; N], f64)> {
    let m = rows.len().min(tgt.len());
    if m <= N + 8 {
        return None;
    }
    let mut mean_y = 0.0_f64;
    for &y in &tgt[..m] {
        mean_y += y;
    }
    mean_y /= m as f64;
    let mut tss = 0.0_f64;
    for &y in &tgt[..m] {
        let d = y - mean_y;
        tss += d * d;
    }
    if !tss.is_finite() || tss <= 1.0e-12 {
        return None;
    }

    let mut xtx = [[0.0_f64; N]; N];
    let mut xty = [0.0_f64; N];
    for (row, &y) in rows[..m].iter().zip(&tgt[..m]) {
        for r in 0..N {
            xty[r] += row[r] * y;
            for c in r..N {
                xtx[r][c] += row[r] * row[c];
            }
        }
    }
    for r in 0..N {
        for c in 0..r {
            xtx[r][c] = xtx[c][r];
        }
        xtx[r][r] += 1.0e-8 * (m as f64).max(1.0);
    }

    let w = solve_small_linear(xtx, xty)?;
    let mut sse = 0.0_f64;
    for (row, &y) in rows[..m].iter().zip(&tgt[..m]) {
        let mut pred = 0.0_f64;
        for c in 0..N {
            pred += row[c] * w[c];
        }
        let err = y - pred;
        sse += err * err;
    }
    Some((w, (1.0 - sse / tss).clamp(0.0, 1.0)))
}

fn append_chan_features(
    dst: &mut SubOctFeatures,
    in_sig: &[f32],
    tgt_sig: &[f32],
    warmup: usize,
    max_cap: usize,
    base_p: V3Stage0Params,
) {
    let n_total = in_sig.len().min(tgt_sig.len());
    let start = warmup.min(n_total);
    let end = n_total.min(start + max_cap);
    if end <= start + FIR_TAPS + 8 {
        return;
    }

    let mut hot = V3Stage0Hot::from_cold(&base_p);
    let base_idx = dst.tgt.len();
    for i in 0..end {
        let [u_avg, s1_avg, s2_avg, oct_avg, env_avg] =
            hot.step_op_p_4x_components_ch(0, in_sig[i]);
        if i >= start {
            let rel = dst.tgt.len() - base_idx;
            dst.x_g.push(f64::from(u_avg));
            dst.abs_x.push(f64::from(oct_avg));
            dst.env.push(f64::from(env_avg));
            dst.s1.push(f64::from(s1_avg));
            dst.s2.push(f64::from(s2_avg));
            dst.tgt.push(f64::from(tgt_sig[i]));
            dst.valid_mask.push(rel >= FIR_TAPS);
        }
    }
}

pub(super) fn collect_stem_pair_features(
    stem: &V3TrainStem,
    warmup_override: Option<usize>,
    max_cap: usize,
    lpf_pole: f32,
    env_pole: f32,
    sub2_phase: bool,
    dst: &mut SubOctFeatures,
) {
    let mut base_p = V3Stage0Params::identity();
    base_p.op_p_lpf_pole = lpf_pole;
    base_p.op_p_env_pole = env_pole;
    base_p.op_p_sub2_phase = sub2_phase;
    let warmup = warmup_override
        .unwrap_or_else(|| stem.warmup.max(canonical_besr_warmup(stem.usable_len())))
        .min(stem.usable_len());
    append_chan_features(dst, &stem.in_l, &stem.tgt_l, warmup, max_cap, base_p);
    if stem.in_r != stem.in_l || stem.tgt_r != stem.tgt_l {
        append_chan_features(dst, &stem.in_r, &stem.tgt_r, warmup, max_cap, base_p);
    }
}

fn empty_features() -> SubOctFeatures {
    SubOctFeatures {
        x_g: Vec::new(),
        abs_x: Vec::new(),
        env: Vec::new(),
        s1: Vec::new(),
        s2: Vec::new(),
        tgt: Vec::new(),
        valid_mask: Vec::new(),
    }
}

fn collect_stem_features(
    stem: &V3TrainStem,
    warmup_override: Option<usize>,
    max_cap: usize,
    lpf_pole: f32,
    env_pole: f32,
    sub2_phase: bool,
) -> Option<SubOctFeatures> {
    let mut feat = empty_features();
    collect_stem_pair_features(
        stem,
        warmup_override,
        max_cap,
        lpf_pole,
        env_pole,
        sub2_phase,
        &mut feat,
    );
    (feat.tgt.len() >= 128).then_some(feat)
}

fn eval_fir_tanh_sse(feat: &SubOctFeatures, h: &[f64; FIR_TAPS], theta: &[f64; 5]) -> f64 {
    let n = feat.tgt.len();
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let u = theta[0] * feat.x_g[i]
            + theta[1] * feat.s1[i]
            + theta[2] * feat.s2[i]
            + theta[3] * feat.abs_x[i]
            + theta[4] * feat.env[i];
        z[i] = u.tanh();
    }
    let mut sse = 0.0_f64;
    for i in FIR_TAPS..n {
        if !feat.valid_mask[i] {
            continue;
        }
        let mut pred = 0.0_f64;
        for (k, &hk) in h.iter().enumerate() {
            pred += hk * z[i - k];
        }
        let err = pred - feat.tgt[i];
        sse += err * err;
    }
    sse
}

fn project_causal_fir(feat: &SubOctFeatures, theta: &[f64; 5]) -> Option<[f64; FIR_TAPS]> {
    let n = feat.tgt.len();
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let u = theta[0] * feat.x_g[i]
            + theta[1] * feat.s1[i]
            + theta[2] * feat.s2[i]
            + theta[3] * feat.abs_x[i]
            + theta[4] * feat.env[i];
        z[i] = u.tanh();
    }
    let mut rows = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in FIR_TAPS..n {
        if !feat.valid_mask[i] {
            continue;
        }
        let r: [f64; FIR_TAPS] = std::array::from_fn(|k| z[i - k]);
        rows.push(r);
        ys.push(feat.tgt[i]);
    }
    ridge_ls_r2(&rows, &ys).map(|(h, _)| h)
}

fn fit_gauss_newton_params(feat: &SubOctFeatures) -> Option<([f64; 5], f64)> {
    let n = feat.tgt.len();
    let mut best_init: Option<(usize, [f64; 5], f64)> = None;
    for lag in 0..FIR_TAPS {
        let mut rows = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in FIR_TAPS..n {
            if !feat.valid_mask[i] {
                continue;
            }
            let j = i - lag;
            rows.push([
                feat.x_g[j],
                feat.s1[j],
                feat.s2[j],
                feat.abs_x[j],
                feat.env[j],
            ]);
            ys.push(feat.tgt[i]);
        }
        if let Some((w, r2)) = ridge_ls_r2(&rows, &ys)
            && best_init.as_ref().is_none_or(|b| r2 > b.2)
        {
            best_init = Some((lag, w, r2));
        }
    }
    let (best_lag, w_init, _) = best_init?;
    let mut h = [0.0_f64; FIR_TAPS];
    h[best_lag] = 1.0;
    let mut theta = [
        (w_init[0] * 1.4).max(0.05),
        w_init[1] * 1.4,
        w_init[2] * 1.4,
        w_init[3] * 1.4,
        w_init[4] * 1.4,
    ];
    let mut best_sse = eval_fir_tanh_sse(feat, &h, &theta);
    if !best_sse.is_finite() {
        return None;
    }

    let mut th_buf = vec![0.0_f64; n];
    let mut sech2_buf = vec![0.0_f64; n];
    for _ in 0..4 {
        if let Some(h_new) = project_causal_fir(feat, &theta) {
            let sse_h = eval_fir_tanh_sse(feat, &h_new, &theta);
            if sse_h.is_finite() && sse_h <= best_sse {
                h = h_new;
                best_sse = sse_h;
            }
        }
        for _ in 0..4 {
            for i in 0..n {
                let u = theta[0] * feat.x_g[i]
                    + theta[1] * feat.s1[i]
                    + theta[2] * feat.s2[i]
                    + theta[3] * feat.abs_x[i]
                    + theta[4] * feat.env[i];
                let th = u.tanh();
                th_buf[i] = th;
                sech2_buf[i] = 1.0 - th * th;
            }
            let mut jtj = [[0.0_f64; 5]; 5];
            let mut jtr = [0.0_f64; 5];
            for i in FIR_TAPS..n {
                if !feat.valid_mask[i] {
                    continue;
                }
                let mut pred = 0.0_f64;
                let mut jac = [0.0_f64; 5];
                for (k, &hk) in h.iter().enumerate() {
                    let j = i - k;
                    pred += hk * th_buf[j];
                    let g = hk * sech2_buf[j];
                    jac[0] += g * feat.x_g[j];
                    jac[1] += g * feat.s1[j];
                    jac[2] += g * feat.s2[j];
                    jac[3] += g * feat.abs_x[j];
                    jac[4] += g * feat.env[j];
                }
                let res = feat.tgt[i] - pred;
                for r in 0..5 {
                    jtr[r] += jac[r] * res;
                    for c in r..5 {
                        jtj[r][c] += jac[r] * jac[c];
                    }
                }
            }
            let mut diag_max = 1.0e-6_f64;
            for r in 0..5 {
                diag_max = diag_max.max(jtj[r][r]);
                for c in 0..r {
                    jtj[r][c] = jtj[c][r];
                }
            }
            let damp = 1.0e-3 * diag_max;
            for (r, row) in jtj.iter_mut().enumerate() {
                row[r] += damp;
            }
            let Some(delta) = solve_small_linear(jtj, jtr) else {
                break;
            };
            let mut accepted = false;
            for &alpha in &[1.0_f64, 0.5, 0.25] {
                let mut cand = [0.0_f64; 5];
                for k in 0..5 {
                    cand[k] = theta[k] + alpha * delta[k];
                }
                cand[0] = cand[0].max(0.05);
                let sse_try = eval_fir_tanh_sse(feat, &h, &cand);
                if sse_try.is_finite() && sse_try < best_sse {
                    theta = cand;
                    best_sse = sse_try;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                break;
            }
        }
    }
    Some((theta, best_sse))
}

pub(super) fn fit_stage0_params_from_dataset(dataset: &V3TrainDataset) -> Option<V3Stage0Params> {
    const LPF_COARSE: [f32; 6] = [0.84, 0.90, 0.94, 0.953, 0.968, 0.982];
    const ENV_COARSE: [f32; 5] = [0.94, 0.968, 0.980, 0.988, 0.994];

    let (coarse_stem, coarse_warmup, coarse_cap) = if let Some(stem1) = dataset.stems.get(1) {
        (stem1, Some(192), 5_952)
    } else {
        (dataset.stems.first()?, None, MAX_PROBE_BODY_SAMPLES)
    };
    let mut best_coarse: Option<(f64, f32, f32, bool)> = None;
    for &sub2_phase in &[false, true] {
        for &lpf in &LPF_COARSE {
            for &env in &ENV_COARSE {
                if let Some(feat) = collect_stem_features(
                    coarse_stem,
                    coarse_warmup,
                    coarse_cap,
                    lpf,
                    env,
                    sub2_phase,
                ) && let Some((_, sse)) = fit_gauss_newton_params(&feat)
                    && best_coarse.as_ref().is_none_or(|b| sse < b.0)
                {
                    best_coarse = Some((sse, lpf, env, sub2_phase));
                }
            }
        }
    }
    let (_, mut best_lpf, mut best_env, mut best_phase) = best_coarse?;

    if let Some(stem0) = dataset.stems.first() {
        let mut phase_best: Option<(f64, bool)> = None;
        for &ph in &[false, true] {
            if let Some(feat) = collect_stem_features(stem0, None, 8_192, best_lpf, best_env, ph)
                && let Some((_, sse)) = fit_gauss_newton_params(&feat)
                && phase_best.as_ref().is_none_or(|b| sse < b.0)
            {
                phase_best = Some((sse, ph));
            }
        }
        if let Some((_, ph)) = phase_best {
            best_phase = ph;
        }

        let mut med_lpf_best: Option<(f64, f32)> = None;
        for dl in -20_i32..=20 {
            let lpf = (best_lpf + 0.0005 * (dl as f32)).clamp(0.75, 0.995);
            if let Some(feat) = collect_stem_features(stem0, None, 8_192, lpf, best_env, best_phase)
                && let Some((_, sse)) = fit_gauss_newton_params(&feat)
                && med_lpf_best.as_ref().is_none_or(|b| sse < b.0)
            {
                med_lpf_best = Some((sse, lpf));
            }
        }
        if let Some((_, m_lpf)) = med_lpf_best {
            best_lpf = m_lpf;
        }

        let mut med_env_best: Option<(f64, f32)> = None;
        for de in -4_i32..=4 {
            let env = (best_env + 0.002 * (de as f32)).clamp(0.85, 0.999);
            if let Some(feat) = collect_stem_features(stem0, None, 8_192, best_lpf, env, best_phase)
                && let Some((_, sse)) = fit_gauss_newton_params(&feat)
                && med_env_best.as_ref().is_none_or(|b| sse < b.0)
            {
                med_env_best = Some((sse, env));
            }
        }
        if let Some((_, m_env)) = med_env_best {
            best_env = m_env;
        }

        let mut fine_lpf_best: Option<(f64, f32)> = None;
        for dl in -16_i32..=16 {
            let lpf = (best_lpf + 0.00004 * (dl as f32)).clamp(0.75, 0.995);
            if let Some(feat) =
                collect_stem_features(stem0, None, 49_152, lpf, best_env, best_phase)
                && let Some((_, sse)) = fit_gauss_newton_params(&feat)
                && fine_lpf_best.as_ref().is_none_or(|b| sse < b.0)
            {
                fine_lpf_best = Some((sse, lpf));
            }
        }
        if let Some((_, f_lpf)) = fine_lpf_best {
            best_lpf = f_lpf;
        }
    }

    let stem0 = dataset.stems.first()?;
    let final_feat = collect_stem_features(stem0, None, 49_152, best_lpf, best_env, best_phase)?;
    let (w, _) = fit_gauss_newton_params(&final_feat)?;
    let inv_dir = 1.0 / w[0].max(0.05);
    let mut s1 = w[1] * inv_dir;
    let mut s2 = w[2] * inv_dir;
    let mut oct = w[3] * inv_dir;
    let mut env_mix = w[4] * inv_dir;

    let ratio_sub = (s1.abs().max(s2.abs()) / 1.98).max(1.0);
    let ratio_oct = (oct.abs().max(env_mix.abs()) / 3.98).max(1.0);
    let scale = 1.0 / ratio_sub.max(ratio_oct);
    s1 *= scale;
    s2 *= scale;
    oct *= scale;
    env_mix *= scale;

    let mut out = V3Stage0Params::identity();
    out.op_p_lpf_pole = best_lpf;
    out.op_p_env_pole = best_env;
    out.op_p_sub_mix = s1 as f32;
    out.op_p_sub2_mix = s2 as f32;
    out.op_p_oct_mix = oct as f32;
    out.op_p_env_mix = if env_mix.abs() < 1.0e-6 {
        1.0e-6
    } else {
        env_mix as f32
    };
    out.op_p_sub2_phase = best_phase;
    out.clamp_in_place();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_lag_compensated_fir_stage0_fit_recovers_delayed_suboctave_and_octavia() {
        let n = 6_144;
        let dry: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let env = 0.60 + 0.40 * (TAU * 2.7 * t).sin();
                (0.42 * env * (TAU * 135.0 * t).sin()) as f32
            })
            .collect();
        let mut planted = V3Stage0Params::identity();
        planted.op_p_lpf_pole = 0.94;
        planted.op_p_env_pole = 0.985;
        planted.op_p_sub_mix = 1.15;
        planted.op_p_sub2_mix = 0.55;
        planted.op_p_oct_mix = 1.40;
        planted.op_p_env_mix = -0.60;
        let mut hot = V3Stage0Hot::from_cold(&planted);
        let nl: Vec<f64> = dry
            .iter()
            .map(|&x| {
                let [dl, _, _, _] = hot.step_stereo(x, x);
                (0.85 * f64::from(dl)).tanh()
            })
            .collect();
        // Apply 4-tap group-delayed minimum-phase filter on output
        const H: [f64; 4] = [0.10, 0.35, 0.45, 0.15];
        let tgt: Vec<f32> = (0..n)
            .map(|i| {
                let mut acc = 0.0_f64;
                for (k, &hk) in H.iter().enumerate() {
                    if i >= k {
                        acc += hk * nl[i - k];
                    }
                }
                acc as f32
            })
            .collect();
        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: dry.clone(),
                in_r: dry,
                tgt_l: tgt.clone(),
                tgt_r: tgt,
                warmup: 256,
            }],
            48_000.0,
        );
        let fitted = fit_stage0_params_from_dataset(&ds).expect("lag-compensated Stage 0 fit");
        assert!(
            fitted.op_p_sub_mix.abs() > 0.45,
            "sub_mix={}",
            fitted.op_p_sub_mix
        );
        assert!(
            fitted.op_p_oct_mix.abs() > 0.45,
            "oct_mix={}",
            fitted.op_p_oct_mix
        );
    }
}
