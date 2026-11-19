#![forbid(unsafe_code)]
//! `uplo='U'` Bézout-Weighted Matryoshka VarPro Cholesky and Progressive-Slot Kautz OMP.

use std::f64::consts::PI;

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_GROUPS_PER_BLOCK, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES,
    NUM_GRU_BLOCKS, tier_block_lanes_array, tier_fir_segments, tier_kautz_banks,
    tier_total_gru_lanes,
};

use crate::solver::backends::TrainerBackend;
use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index};

mod als;
mod cache;

use super::model_f64::{Bezout7FilterBank, V3TierReadoutF64};
pub(super) use als::{als_update_pre_cab_chan, solve_tier_output_chan, solve_tier_output_k_chan};
pub use als::{solve_matryoshka_varpro_cholesky, solve_matryoshka_varpro_cholesky_with_backend};
pub(super) use cache::{
    PreSplitStemChan, SubspaceGramData, assemble_tier_band_grams, eval_bezout_loss_from_gram,
    sym_uplo_matvec,
};

pub const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
pub const KAUTZ_DELAY_SAMPLES: usize = 16;

/// Linear readout basis dimension (`45, 73, 101, 129` for `Lite..Ultra`).
#[inline]
#[must_use]
pub const fn linear_basis_dim(tier: Tier) -> usize {
    linear_basis_dim_with_k(tier, tier_kautz_banks(tier) * 4)
}

/// Linear readout basis dimension capped at `k_keep` complex Kautz poles.
#[inline]
#[must_use]
pub const fn linear_basis_dim_with_k(tier: Tier, k_keep: usize) -> usize {
    let f = tier_fir_segments(tier) * 16;
    9 + tier_total_gru_lanes(tier) + f + 2 * k_keep
}

/// Active per-block GRU lane counts for `max_groups` (`[4,4,4]`, `[8,4,4]`, or `[12,8,4]`).
#[inline]
#[must_use]
pub(super) fn active_block_lanes(max_groups: usize) -> [usize; NUM_GRU_BLOCKS] {
    let g_cnt = max_groups.clamp(1, MAX_GROUPS);
    std::array::from_fn(|b| g_cnt.saturating_sub(b).clamp(1, MAX_GROUPS_PER_BLOCK[b]) * 4)
}

pub(super) fn normalize_pre_cab_chan(
    r: &mut V3TierReadoutF64,
    prev: Option<&V3TierReadoutF64>,
    tau: usize,
    ch: usize,
) {
    let c_t = tier_block_lanes_array(ALL_TIERS[tau]);
    let mut sq = r.pre_cab_u_gain[ch].powi(2) + r.pre_cab_hp_gain[ch].powi(2);
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        let c_b = c_t[b];
        sq += r.pre_cab_w[ch][base..base + c_b]
            .iter()
            .map(|w| w * w)
            .sum::<f64>();
        r.pre_cab_w[ch][base + c_b..base + MAX_BLOCK_LANES].fill(0.0);
        r.direct_s2_w[ch][base + c_b..base + MAX_BLOCK_LANES].fill(0.0);
    }
    if sq <= 1.0e-12 {
        return;
    }
    let mut scale = sq.sqrt();
    if let Some(p) = prev {
        let c_p = tier_block_lanes_array(ALL_TIERS[tau.saturating_sub(1)]);
        let mut dot = r.pre_cab_u_gain[ch] * p.pre_cab_u_gain[ch]
            + r.pre_cab_hp_gain[ch] * p.pre_cab_hp_gain[ch];
        for b in 0..NUM_GRU_BLOCKS {
            let base = b * MAX_BLOCK_LANES;
            let c_pb = c_p[b];
            dot += r.pre_cab_w[ch][base..base + c_pb]
                .iter()
                .zip(&p.pre_cab_w[ch][base..base + c_pb])
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }
        if dot < 0.0 {
            scale = -scale;
        }
    }
    let inv = 1.0 / scale;
    r.pre_cab_u_gain[ch] *= inv;
    r.pre_cab_hp_gain[ch] *= inv;
    for w in r.pre_cab_w[ch].iter_mut() {
        *w *= inv;
    }
    let fir = if ch == 0 { &mut r.fir_l } else { &mut r.fir_r };
    fir.iter_mut().for_each(|v| *v *= scale);
    let kc = if ch == 0 {
        &mut r.kautz_c_l
    } else {
        &mut r.kautz_c_r
    };
    kc.iter_mut().for_each(|c| *c = (c.0 * scale, c.1 * scale));
}

pub(super) fn normalize_pre_cab_gauge(readouts: &mut [V3TierReadoutF64; 4], active_ch: usize) {
    for ch in 0..active_ch {
        for tau in 0..4 {
            let prev = (tau > 0).then(|| readouts[tau - 1].clone());
            normalize_pre_cab_chan(&mut readouts[tau], prev.as_ref(), tau, ch);
        }
    }
}

pub(super) fn eval_s2_from_readout(
    r: &V3TierReadoutF64,
    ch: usize,
    c_t: [usize; NUM_GRU_BLOCKS],
    u_dec: &[f64],
    hp_dec: &[f64],
    h_dec: &[[f64; MAX_TOTAL_GRU_LANES]],
    out_s2: &mut Vec<f64>,
) {
    let n = u_dec.len();
    out_s2.resize(n, 0.0);
    for t in 0..n {
        let mut s2 = r.pre_cab_u_gain[ch] * u_dec[t] + r.pre_cab_hp_gain[ch] * hp_dec[t];
        let hd = &h_dec[t];
        for b in 0..NUM_GRU_BLOCKS {
            let base = b * MAX_BLOCK_LANES;
            for l in 0..c_t[b] {
                s2 += r.pre_cab_w[ch][base + l] * hd[base + l];
            }
        }
        out_s2[t] = s2;
    }
}

pub(super) fn apply_fixed_cabinet_f64<B: TrainerBackend>(
    sig: &[f64],
    fir: &[f64],
    kautz_pre6: &[(f64, f64, f64, f64, f64, f64)],
    kc: &[(f64, f64)],
    out: &mut [f64],
) {
    let n = sig.len().min(out.len());
    let f_t = fir.len();
    let k_t = kautz_pre6.len().min(kc.len()).min(MAX_KAUTZ_POLES);
    let mut fir_rev = [0.0_f64; 64];
    let fast_fir = f_t > 0 && f_t <= fir_rev.len();
    if fast_fir {
        for (k, &w) in fir.iter().rev().enumerate() {
            fir_rev[k] = w;
        }
    }
    let mut s_re = [0.0_f64; MAX_KAUTZ_POLES];
    let mut s_im = [0.0_f64; MAX_KAUTZ_POLES];
    let mut out_re = [0.0_f64; MAX_KAUTZ_POLES];
    let mut out_im = [0.0_f64; MAX_KAUTZ_POLES];
    for t in 0..n {
        let mut v = if fast_fir && t + 1 >= f_t {
            B::dot_f64(&fir_rev[..f_t], &sig[t + 1 - f_t..=t])
        } else {
            let mut acc = 0.0_f64;
            for k in 0..f_t.min(t + 1) {
                acc += fir[k] * sig[t - k];
            }
            acc
        };
        let x_del = if t >= KAUTZ_DELAY_SAMPLES {
            sig[t - KAUTZ_DELAY_SAMPLES]
        } else {
            0.0
        };
        v += B::step_kautz_forward_frame(
            x_del,
            &kautz_pre6[..k_t],
            &kc[..k_t],
            &mut s_re[..k_t],
            &mut s_im[..k_t],
            &mut out_re[..k_t],
            &mut out_im[..k_t],
        );
        out[t] = v;
    }
}

pub(super) fn kautz_pre_tuples(poles: &[(f64, f64)]) -> Vec<(f64, f64, f64)> {
    poles
        .iter()
        .map(|&(rr, tt)| {
            let r = rr.clamp(0.25, 0.9985);
            let th = tt.clamp(0.0, PI - 1.0e-3);
            let bg = (1.0 - r * r).max(1.0e-7).sqrt();
            (r * th.cos(), r * th.sin(), bg)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tier_columns_with_kautz_cap<B: TrainerBackend>(
    tier: Tier,
    poles: &[(f64, f64)],
    k_keep: usize,
    tap_dec: &[[f64; 7]],
    u_dec: &[f64],
    hp_dec: &[f64],
    h_dec: &[[f64; MAX_TOTAL_GRU_LANES]],
    s2: &[f64],
) -> Vec<Vec<f64>> {
    let n = s2.len();
    let c_t = tier_block_lanes_array(tier);
    let total_h = tier_total_gru_lanes(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = (tier_kautz_banks(tier) * 4).min(k_keep);
    let p_dim = 9 + total_h + f_t + 2 * k_t;
    let mut cols = vec![vec![0.0_f64; n]; p_dim];
    for i in 0..7 {
        let col = &mut cols[i][..n];
        for (t, taps) in tap_dec[..n].iter().enumerate() {
            col[t] = taps[i];
        }
    }
    cols[7][..n].copy_from_slice(&u_dec[..n]);
    cols[8][..n].copy_from_slice(&hp_dec[..n]);
    let mut idx = 9;
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for l in 0..c_t[b] {
            let lane = base + l;
            let col = &mut cols[idx][..n];
            for (t, h_row) in h_dec[..n].iter().enumerate() {
                col[t] = h_row[lane];
            }
            idx += 1;
        }
    }
    for k in 0..f_t {
        if k < n {
            cols[idx + k][k..n].copy_from_slice(&s2[..n - k]);
        }
    }
    let k_base = 9 + total_h + f_t;
    if k_t > 0 {
        let pre = kautz_pre_tuples(&poles[..k_t]);
        B::filter_pole_bank_columns(s2, KAUTZ_DELAY_SAMPLES, &pre, &mut cols, k_base);
    }
    cols
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tier_columns<B: TrainerBackend>(
    tier: Tier,
    poles: &[(f64, f64)],
    tap_dec: &[[f64; 7]],
    u_dec: &[f64],
    hp_dec: &[f64],
    h_dec: &[[f64; MAX_TOTAL_GRU_LANES]],
    s2: &[f64],
) -> Vec<Vec<f64>> {
    build_tier_columns_with_kautz_cap::<B>(
        tier,
        poles,
        tier_kautz_banks(tier) * 4,
        tap_dec,
        u_dec,
        hp_dec,
        h_dec,
        s2,
    )
}

pub(super) fn unpack_theta_k_to_readout(
    tier: Tier,
    ch: usize,
    k_keep: usize,
    theta: &[f64],
    r: &mut V3TierReadoutF64,
) {
    let c_t = tier_block_lanes_array(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = (tier_kautz_banks(tier) * 4).min(k_keep);
    if ch == 0 {
        r.svf_direct_taps_l[..7].copy_from_slice(&theta[..7]);
    } else {
        r.svf_direct_taps_r[..7].copy_from_slice(&theta[..7]);
    }
    r.direct_s2_u_gain[ch] = theta[7];
    r.direct_s2_hp_gain[ch] = theta[8];
    let mut idx = 9;
    r.direct_s2_w[ch].fill(0.0);
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for l in 0..c_t[b] {
            r.direct_s2_w[ch][base + l] = theta[idx];
            idx += 1;
        }
    }
    let fir = if ch == 0 { &mut r.fir_l } else { &mut r.fir_r };
    fir.fill(0.0);
    fir[..f_t].copy_from_slice(&theta[idx..idx + f_t]);
    idx += f_t;
    let kc = if ch == 0 {
        &mut r.kautz_c_l
    } else {
        &mut r.kautz_c_r
    };
    kc.fill((0.0, 0.0));
    for m in 0..k_t {
        kc[m] = (theta[idx + 2 * m], theta[idx + 2 * m + 1]);
    }
}

pub(super) fn unpack_theta_to_readout(
    tier: Tier,
    ch: usize,
    theta: &[f64],
    r: &mut V3TierReadoutF64,
) {
    unpack_theta_k_to_readout(tier, ch, tier_kautz_banks(tier) * 4, theta, r);
}

pub(super) fn pack_readout_to_theta(tier: Tier, ch: usize, r: &V3TierReadoutF64) -> Vec<f64> {
    let c_t = tier_block_lanes_array(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = tier_kautz_banks(tier) * 4;
    let mut theta = Vec::with_capacity(linear_basis_dim(tier));
    let taps = if ch == 0 {
        &r.svf_direct_taps_l
    } else {
        &r.svf_direct_taps_r
    };
    theta.extend_from_slice(&taps[..7]);
    theta.push(r.direct_s2_u_gain[ch]);
    theta.push(r.direct_s2_hp_gain[ch]);
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for l in 0..c_t[b] {
            theta.push(r.direct_s2_w[ch][base + l]);
        }
    }
    let fir = if ch == 0 { &r.fir_l } else { &r.fir_r };
    theta.extend_from_slice(&fir[..f_t]);
    let kc = if ch == 0 { &r.kautz_c_l } else { &r.kautz_c_r };
    for &(re, im) in &kc[..k_t] {
        theta.push(re);
        theta.push(im);
    }
    theta
}

pub(super) fn map_prev_tier_anchor(
    tau: usize,
    prev_r: &V3TierReadoutF64,
    ch: usize,
) -> Vec<(usize, f64)> {
    let prev_t = ALL_TIERS[tau - 1];
    let cur_t = ALL_TIERS[tau];
    let cp = tier_block_lanes_array(prev_t);
    let cc = tier_block_lanes_array(cur_t);
    let fp = tier_fir_segments(prev_t) * 16;
    let fc = tier_fir_segments(cur_t) * 16;
    let kp = tier_kautz_banks(prev_t) * 4;
    let pt = pack_readout_to_theta(prev_t, ch, prev_r);
    let mut out = Vec::with_capacity(pt.len());
    for (i, &val) in pt.iter().enumerate().take(9) {
        out.push((i, val));
    }
    let mut pidx = 9;
    let mut dst_off = 9;
    for b in 0..NUM_GRU_BLOCKS {
        for l in 0..cp[b] {
            out.push((dst_off + l, pt[pidx]));
            pidx += 1;
        }
        dst_off += cc[b];
    }
    let f_base = 9 + tier_total_gru_lanes(cur_t);
    for k in 0..fp {
        out.push((f_base + k, pt[pidx]));
        pidx += 1;
    }
    let k_base = f_base + fc;
    for m in 0..2 * kp {
        out.push((k_base + m, pt[pidx + m]));
    }
    out
}

pub(super) fn split_append_3band(
    bank: &Bezout7FilterBank,
    sig: &[f64],
    warmup: usize,
    sq_bw: [f64; 3],
    out_w: &mut Vec<f64>,
) {
    let w = warmup.min(sig.len());
    let mut bands = [Vec::new(), Vec::new(), Vec::new()];
    bank.split_3band(sig, &mut bands);
    let seg = sig.len().saturating_sub(w);
    let start = out_w.len();
    out_w.resize(start + 3 * seg, 0.0);
    for (b, band) in bands.iter().enumerate() {
        let s = sq_bw[b];
        let dst = &mut out_w[start + b * seg..start + (b + 1) * seg];
        for (d, &v) in dst.iter_mut().zip(&band[w..sig.len()]) {
            *d = s * v;
        }
    }
}

pub(super) struct StemChanTraj {
    pub(super) tap: Vec<[f64; 7]>,
    pub(super) u: Vec<f64>,
    pub(super) hp: Vec<f64>,
    pub(super) h: Vec<[f64; MAX_TOTAL_GRU_LANES]>,
}

pub(super) fn solve_uplo_with_bypass_and_anchor(
    p_dim: usize,
    gram: &[f64],
    rhs: &[[f64; 2]],
    bypass_cols: &[(usize, f64)],
    anchor: &[(usize, f64)],
    anchor_scale: f64,
) -> Vec<f64> {
    let mut trace = 0.0_f64;
    for j in 0..p_dim {
        trace += gram[uplo_u_index(j, j)];
    }
    let mean_diag = (trace / (p_dim.max(1) as f64)).max(1.0e-12);
    let mut g_work = gram.to_vec();
    let mut r_work = rhs.to_vec();
    for &(idx, scale) in bypass_cols {
        if idx < p_dim && scale > 0.0 {
            g_work[uplo_u_index(idx, idx)] += scale * mean_diag;
        }
    }
    if anchor_scale > 0.0 && !anchor.is_empty() {
        let lam_a = anchor_scale * mean_diag;
        for &(idx, val) in anchor {
            if idx < p_dim {
                g_work[uplo_u_index(idx, idx)] += lam_a;
                r_work[idx][0] += lam_a * val;
            }
        }
    }
    let mults: &[f64] = if bypass_cols.is_empty() && anchor.is_empty() {
        &[
            1.0e-12, 1.0e-10, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-3, 1.0e-1,
        ]
    } else {
        &[1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-3, 1.0e-1]
    };
    for &mult in mults {
        let ridge = mult * mean_diag
            + if bypass_cols.is_empty() {
                1.0e-15
            } else {
                1.0e-11
            };
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(&g_work, &r_work, ridge) {
            return sol.into_iter().map(|c| c[0]).collect();
        }
    }
    vec![0.0; p_dim]
}

fn readout_theta_feasible(
    theta: &[f64],
    p_dim: usize,
    total_h: usize,
    dir_w_budget_sq: f64,
) -> bool {
    const LIN_CAP: f64 = 64.0;
    let lin_end = 9.min(theta.len()).min(p_dim);
    let max_lin = theta[..lin_end].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let w_start = 9.min(theta.len()).min(p_dim);
    let w_end = (9 + total_h).min(theta.len()).min(p_dim);
    let dir_w_sq: f64 = theta[w_start..w_end].iter().map(|v| v * v).sum();
    dir_w_sq <= dir_w_budget_sq && max_lin <= LIN_CAP
}

/// Solves a tier linear readout via `solve_uplo_with_bypass_and_anchor` while enforcing an exact
/// trust region on `svf_direct_taps`, `direct_s2_u_gain`, `direct_s2_hp_gain`, and `direct_s2_w`
/// (`<= 240.0 - ||pre_cab_w||_2^2`) so Stage III FIR/Kautz weights compensate instead of relying
/// on collinear cancellations that Phase-3 `[CLAMP-HEAD-GAIN]` would later truncate.
pub(super) fn solve_tier_readout_uplo_bounded(
    p_dim: usize,
    total_h: usize,
    pre_cab_w_sq: f64,
    gram: &[f64],
    rhs: &[[f64; 2]],
    bypass_cols: &[(usize, f64)],
    anchor_spec: Option<(&[(usize, f64)], f64)>,
) -> Vec<f64> {
    const LIN_CAP: f64 = 64.0;
    const BRACKET_MUS: [f64; 9] = [
        1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 1.0e2, 1.0e3, 1.0e4,
    ];

    let (anchor, anchor_scale) = anchor_spec.unwrap_or((&[], 0.0));
    let theta =
        solve_uplo_with_bypass_and_anchor(p_dim, gram, rhs, bypass_cols, anchor, anchor_scale);
    let dir_w_budget_sq = (240.0 - pre_cab_w_sq).max(64.0);
    if readout_theta_feasible(&theta, p_dim, total_h, dir_w_budget_sq) {
        return theta;
    }

    let anchor_w_sq: f64 = anchor
        .iter()
        .filter(|&&(idx, _)| idx >= 9 && idx < 9 + total_h)
        .map(|&(_, v)| v * v)
        .sum();
    let anchor_lin: f64 = anchor
        .iter()
        .filter(|&&(idx, _)| idx < 9)
        .fold(0.0_f64, |m, &(_, v)| m.max(v.abs()));
    let eff_anchor_scale = if anchor_w_sq <= dir_w_budget_sq && anchor_lin <= LIN_CAP {
        anchor_scale
    } else {
        0.0
    };

    let pen_end = (9 + total_h).min(p_dim);
    let eval_mu = |mu: f64| -> Vec<f64> {
        let mut aug = Vec::with_capacity(bypass_cols.len() + pen_end);
        aug.extend_from_slice(bypass_cols);
        for idx in 0..pen_end {
            aug.push((idx, mu));
        }
        solve_uplo_with_bypass_and_anchor(p_dim, gram, rhs, &aug, anchor, eff_anchor_scale)
    };

    let mut mu_lo = 0.0_f64;
    let mut mu_hi = *BRACKET_MUS.last().unwrap_or(&1.0e4);
    let mut best_theta: Option<Vec<f64>> = None;
    for &mu in &BRACKET_MUS {
        let cand = eval_mu(mu);
        if readout_theta_feasible(&cand, p_dim, total_h, dir_w_budget_sq) {
            mu_hi = mu;
            best_theta = Some(cand);
            break;
        }
        mu_lo = mu;
    }

    let Some(mut feasible_theta) = best_theta else {
        return eval_mu(mu_hi);
    };
    for _ in 0..8 {
        let mu_mid = if mu_lo > 0.0 {
            (mu_lo * mu_hi).sqrt()
        } else {
            0.5 * mu_hi
        };
        let cand = eval_mu(mu_mid);
        if readout_theta_feasible(&cand, p_dim, total_h, dir_w_budget_sq) {
            mu_hi = mu_mid;
            feasible_theta = cand;
        } else {
            mu_lo = mu_mid;
        }
    }
    feasible_theta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::uplo_u_packed_len;

    #[test]
    fn test_solve_tier_readout_uplo_bounded_trust_region_compensates_collinear_cols() {
        let p_dim = 12_usize;
        let total_h = 2_usize;
        // Build nearly collinear direct columns (7 = u, 8 = hp, 9..11 = direct_s2_w) while column 11
        // (Stage III FIR tap 0) cleanly reproduces target y = 0.5 * x.
        let n = 64_usize;
        let mut gram = vec![0.0_f64; uplo_u_packed_len(p_dim)];
        let mut rhs = vec![[0.0_f64; 2]; p_dim];
        for t in 0..n {
            let x = (0.13 * (t as f64)).sin();
            let mut row = [0.0_f64; 12];
            row[7] = x;
            row[8] = x + 1.0e-5 * ((0.31 * (t as f64)).cos());
            row[9] = x - 1.0e-5 * ((0.27 * (t as f64)).sin());
            row[10] = x + 1.0e-5 * ((0.19 * (t as f64)).sin());
            row[11] = 0.99 * x;
            // Target requires large collinear cancellation unless regularized toward col 11.
            let y = 0.5 * x
                + 2.5e-4 * ((0.31 * (t as f64)).cos())
                + 2.5e-4 * ((0.27 * (t as f64)).sin());
            for j in 0..p_dim {
                rhs[j][0] += row[j] * y;
                for i in 0..=j {
                    gram[uplo_u_index(i, j)] += row[i] * row[j];
                }
            }
        }
        let mut excessive_anchor = vec![(8_usize, 120.0), (9_usize, 30.0)];
        let unconstrained =
            solve_uplo_with_bypass_and_anchor(p_dim, &gram, &rhs, &[], &excessive_anchor, 1.0e-2);
        assert!(
            !readout_theta_feasible(&unconstrained, p_dim, total_h, 240.0),
            "synthetic collinear system should violate raw unconstrained norm cap"
        );

        let bounded = solve_tier_readout_uplo_bounded(
            p_dim,
            total_h,
            0.0,
            &gram,
            &rhs,
            &[],
            Some((&excessive_anchor, 1.0e-2)),
        );
        assert!(readout_theta_feasible(&bounded, p_dim, total_h, 240.0));
        assert!(
            bounded[11].abs() > 0.25,
            "Stage III col 11 should compensate when direct_s2 cols are damped: {:?}",
            bounded
        );

        excessive_anchor.clear();
        let well_cond =
            solve_tier_readout_uplo_bounded(p_dim, total_h, 0.0, &gram, &rhs, &[], None);
        assert!(readout_theta_feasible(&well_cond, p_dim, total_h, 240.0));
    }
}
