//! Subspace helpers for `CP1` ANOVA/FM regression (`solve_ols_ridge_uplo`, symmetric deflated
//! power-iteration factorization, and exact self-term-subtracted FM pairwise features).

use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::control::cp1::Cp1ContributorSlot;
use dnam_vm::control::descriptor::ControlClampDescriptor;

use crate::linalg::cholesky::solve_uplo_adaptive_ridge;

use super::super::Cp1ControlSample;

pub(super) const ACTIVE_DISPATCH_SLOTS: usize = STAGE_TARGET_SLOTS - 1;
pub(super) const RANK_BASE_FM: usize = 4;
pub(super) const RANK_STATIC_LATENT: usize = 8;
pub(super) const RANK_ULTRA_LATENT: usize = 4;

pub(super) fn solve_ols_ridge_uplo(
    design: &[Vec<f64>],
    targets: &[Vec<f64>],
    reg: f64,
) -> Vec<Vec<f64>> {
    let n_obs = design.len();
    if n_obs == 0 || design[0].is_empty() || targets.is_empty() || targets[0].is_empty() {
        return Vec::new();
    }
    let p = design[0].len();
    let k = targets[0].len();
    let packed_len = p * (p + 1) / 2;
    let mut g_u = vec![0.0_f64; packed_len];
    for c in 0..p {
        let col_off = c * (c + 1) / 2;
        for r in 0..=c {
            let mut acc = 0.0_f64;
            for row in design {
                acc += row[r] * row[c];
            }
            g_u[col_off + r] = acc;
        }
    }
    let mut weights = vec![vec![0.0_f64; k]; p];
    let mut rhs = vec![0.0_f64; p];
    for target_col in 0..k {
        rhs.fill(0.0);
        for (i, row) in design.iter().enumerate() {
            let y = targets[i][target_col];
            for r in 0..p {
                rhs[r] += row[r] * y;
            }
        }
        if let Ok((sol, _ridge)) =
            solve_uplo_adaptive_ridge(p, &g_u, &rhs, &[], None, reg.max(1.0e-9))
        {
            for r in 0..p {
                weights[r][target_col] = sol[r];
            }
        }
    }
    weights
}

const BASE_DISPATCH_SLOTS: usize = 16;
const EXT_DISPATCH_SLOTS: usize = ACTIVE_DISPATCH_SLOTS - BASE_DISPATCH_SLOTS;

const _: () = assert!(EXT_DISPATCH_SLOTS == 15);

/// Extracts up to `max_rank` orthonormal eigenvectors from symmetric PSD `gram` (`m x m`) via
/// deterministic deflated symmetric power iteration.
fn extract_top_gram_eigenvectors(gram: &[Vec<f64>], max_rank: usize) -> Vec<Vec<f64>> {
    let m = gram.len();
    if m == 0 || max_rank == 0 {
        return Vec::new();
    }
    let mut work_gram = gram.to_vec();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_rank.min(m));
    for k in 0..max_rank.min(m) {
        let mut best_col = 0_usize;
        let mut best_diag = 0.0_f64;
        for (c, row) in work_gram.iter().enumerate() {
            if row[c] > best_diag {
                best_diag = row[c];
                best_col = c;
            }
        }
        if best_diag <= 1.0e-18 {
            break;
        }
        let mut v = work_gram[best_col].clone();
        for _iter in 0..48 {
            let mut next = vec![0.0_f64; m];
            for r in 0..m {
                let mut acc = 0.0_f64;
                for c in 0..m {
                    acc += work_gram[r][c] * v[c];
                }
                next[r] = acc;
            }
            for prev in &basis[..k] {
                let proj: f64 = next.iter().zip(prev).map(|(a, b)| a * b).sum();
                for (dst, &p) in next.iter_mut().zip(prev) {
                    *dst -= proj * p;
                }
            }
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= 1.0e-15 {
                v.fill(0.0);
                break;
            }
            let mut max_diff = 0.0_f64;
            for (dst, val) in v.iter_mut().zip(next) {
                let unit = val / norm;
                max_diff = max_diff.max((*dst - unit).abs());
                *dst = unit;
            }
            if max_diff <= 1.0e-13 {
                break;
            }
        }
        let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm <= 1.0e-12 {
            break;
        }
        for dst in &mut v {
            *dst /= v_norm;
        }

        let mut cv = vec![0.0_f64; m];
        for r in 0..m {
            cv[r] = work_gram[r].iter().zip(&v).map(|(g, x)| g * x).sum();
        }
        let lambda: f64 = v.iter().zip(&cv).map(|(a, b)| a * b).sum();
        for r in 0..m {
            for c in 0..m {
                work_gram[r][c] -= lambda * v[r] * v[c];
            }
        }
        basis.push(v);
    }
    basis
}

#[inline]
fn accum_symmetric_gram(rows: &[Vec<f64>], col_offset: usize, ncols: usize) -> Vec<Vec<f64>> {
    let mut gram = vec![vec![0.0_f64; ncols]; ncols];
    for row in rows {
        for j in 0..ncols {
            let rj = row[col_offset + j];
            if rj == 0.0 {
                continue;
            }
            for i in 0..=j {
                let prod = row[col_offset + i] * rj;
                gram[i][j] += prod;
                if i != j {
                    gram[j][i] += prod;
                }
            }
        }
    }
    gram
}

/// Factors `[N x 31]` target matrix via deterministic deflated symmetric power iteration into
/// latent coordinates `Z[N x R]` (bounded to `0.4 * trust_radius_mh`) and dispatch weights `W[31 x R]`.
pub(super) fn factor_dispatch_subspace<const R: usize>(
    targets: &[Vec<f64>],
    trust_radius_mh: f32,
) -> (Vec<[f64; R]>, [[f64; R]; ACTIVE_DISPATCH_SLOTS]) {
    let n_obs = targets.len();
    let gram = accum_symmetric_gram(targets, 0, ACTIVE_DISPATCH_SLOTS);
    let basis = extract_top_gram_eigenvectors(&gram, R);

    let z_cap = 0.4 * f64::from(trust_radius_mh).max(0.25);
    let mut z_latents = vec![[0.0_f64; R]; n_obs];
    let mut w_disp = [[0.0_f64; R]; ACTIVE_DISPATCH_SLOTS];
    for (k, eigvec) in basis.iter().enumerate() {
        let mut max_abs = 0.0_f64;
        for (i, row) in targets.iter().enumerate() {
            let dot: f64 = row[..ACTIVE_DISPATCH_SLOTS]
                .iter()
                .zip(eigvec)
                .map(|(a, b)| a * b)
                .sum();
            z_latents[i][k] = dot;
            max_abs = max_abs.max(dot.abs());
        }
        let scale = if max_abs > 1.0e-12 {
            max_abs / z_cap
        } else {
            1.0
        };
        for z_row in &mut z_latents {
            z_row[k] /= scale;
        }
        for s_idx in 0..ACTIVE_DISPATCH_SLOTS {
            w_disp[s_idx][k] = eigvec[s_idx] * scale;
        }
    }
    (z_latents, w_disp)
}

/// Factors `[N x 31]` targets preserving `0-ULP` `SO(4)` buyer-watermark gauge invariance:
/// columns `0..4` (`z_head`) hold strictly zero weights on extended slots `16..31` (`w_disp[16..31][0..4] == 0.0`),
/// while unrotated `ctrl_basis_4` columns `4..8` carry up to `4` `"CEXT"` subspace directions plus any
/// remaining base-slot directions (`ref[dsp.ctrl.dispatch.matrix]`, `ref[sec.watermark.orthogonal-gauge]`).
pub(super) fn factor_gauge_invariant_dispatch_subspace(
    targets: &[Vec<f64>],
    trust_radius_mh: f32,
) -> (
    Vec<[f64; RANK_STATIC_LATENT]>,
    [[f64; RANK_STATIC_LATENT]; ACTIVE_DISPATCH_SLOTS],
) {
    let n_obs = targets.len();
    let z_cap = 0.4 * f64::from(trust_radius_mh).max(0.25);
    let mut z_latents = vec![[0.0_f64; RANK_STATIC_LATENT]; n_obs];
    let mut w_disp = [[0.0_f64; RANK_STATIC_LATENT]; ACTIVE_DISPATCH_SLOTS];
    if n_obs == 0 {
        return (z_latents, w_disp);
    }

    let g_ext = accum_symmetric_gram(targets, BASE_DISPATCH_SLOTS, EXT_DISPATCH_SLOTS);
    let ext_basis = extract_top_gram_eigenvectors(&g_ext, 4);
    let r_ext = ext_basis.len().min(4);

    let mut rem_base: Vec<Vec<f64>> = targets
        .iter()
        .map(|row| row[..BASE_DISPATCH_SLOTS].to_vec())
        .collect();

    for (k, eigvec) in ext_basis.iter().take(r_ext).enumerate() {
        let col = 4 + k;
        let mut max_abs = 0.0_f64;
        for (i, row) in targets.iter().enumerate() {
            let dot: f64 = row[BASE_DISPATCH_SLOTS..ACTIVE_DISPATCH_SLOTS]
                .iter()
                .zip(eigvec)
                .map(|(a, b)| a * b)
                .sum();
            z_latents[i][col] = dot;
            max_abs = max_abs.max(dot.abs());
        }
        let scale = if max_abs > 1.0e-12 {
            max_abs / z_cap
        } else {
            1.0
        };
        let mut z_norm_sq = 0.0_f64;
        for z_row in &mut z_latents {
            z_row[col] /= scale;
            z_norm_sq += z_row[col] * z_row[col];
        }
        for j in 0..EXT_DISPATCH_SLOTS {
            w_disp[BASE_DISPATCH_SLOTS + j][col] = eigvec[j] * scale;
        }
        if z_norm_sq > 1.0e-18 {
            for r in 0..BASE_DISPATCH_SLOTS {
                let mut dot_r = 0.0_f64;
                for i in 0..n_obs {
                    dot_r += rem_base[i][r] * z_latents[i][col];
                }
                let alpha = dot_r / z_norm_sq;
                w_disp[r][col] = alpha;
                for i in 0..n_obs {
                    rem_base[i][r] -= alpha * z_latents[i][col];
                }
            }
        }
    }

    let rem_rank = RANK_STATIC_LATENT.saturating_sub(r_ext);
    let g_base = accum_symmetric_gram(&rem_base, 0, BASE_DISPATCH_SLOTS);
    let base_basis = extract_top_gram_eigenvectors(&g_base, rem_rank);
    for (idx, eigvec) in base_basis.iter().enumerate() {
        let col = if idx < 4 { idx } else { 4 + r_ext + (idx - 4) };
        if col >= RANK_STATIC_LATENT {
            continue;
        }
        let mut max_abs = 0.0_f64;
        for (i, row) in rem_base.iter().enumerate() {
            let dot: f64 = row.iter().zip(eigvec).map(|(a, b)| a * b).sum();
            z_latents[i][col] = dot;
            max_abs = max_abs.max(dot.abs());
        }
        let scale = if max_abs > 1.0e-12 {
            max_abs / z_cap
        } else {
            1.0
        };
        for z_row in &mut z_latents {
            z_row[col] /= scale;
        }
        for r in 0..BASE_DISPATCH_SLOTS {
            w_disp[r][col] = eigvec[r] * scale;
        }
    }

    (z_latents, w_disp)
}

pub(super) fn contributor_features_matrix(
    slots: &[Cp1ContributorSlot],
    norm_table: &[Vec<f64>],
    descs: &[ControlClampDescriptor],
    samples: &[Cp1ControlSample],
) -> Vec<Vec<f64>> {
    samples
        .iter()
        .enumerate()
        .map(|(i, samp)| {
            slots
                .iter()
                .map(|sl| {
                    let c = usize::from(sl.control_idx);
                    if sl.ctrl_kind == 0 {
                        let u_p = norm_table[i][c] as f32;
                        let lob = descs[c].eval_lobatto_basis(u_p);
                        f64::from(sl.emb[0] * lob[0] + sl.emb[1] * lob[1] + sl.emb[2] * lob[2])
                    } else if samp.switch_options.get(c).copied().unwrap_or(0) == sl.aux {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

pub(super) fn slot_cooccurrence_mask(phi: &[Vec<f64>], n_slots: usize) -> Vec<bool> {
    (0..n_slots)
        .map(|si| {
            phi.iter().any(|row| {
                row[si].abs() > 1.0e-9
                    && row
                        .iter()
                        .enumerate()
                        .any(|(sj, &v)| sj != si && v.abs() > 1.0e-9)
            })
        })
        .collect()
}

pub(super) fn eval_fm_pairs_with_self_sub(
    slots: &[Cp1ContributorSlot],
    norm_table: &[Vec<f64>],
    descs: &[ControlClampDescriptor],
    samples: &[Cp1ControlSample],
    use_ultra_v: bool,
) -> Vec<Vec<f64>> {
    let phi = contributor_features_matrix(slots, norm_table, descs, samples);
    let n_obs = samples.len();
    let mut out = vec![vec![0.0_f64; RANK_BASE_FM]; n_obs];
    for (i, phi_row) in phi.iter().enumerate() {
        let mut sum_v = [0.0_f64; RANK_BASE_FM];
        let mut sum_v_sq = [0.0_f64; RANK_BASE_FM];
        for (si, sl) in slots.iter().enumerate() {
            let z_c = phi_row[si];
            let v_arr = if use_ultra_v { &sl.v_ultra } else { &sl.v_base };
            for f in 0..RANK_BASE_FM {
                let term = f64::from(v_arr[f]) * z_c;
                sum_v[f] += term;
                sum_v_sq[f] += term * term;
            }
        }
        for f in 0..RANK_BASE_FM {
            out[i][f] = 0.5 * (sum_v[f] * sum_v[f] - sum_v_sq[f]);
        }
    }
    out
}

fn eval_fm_rss(fm: &[Vec<f64>], u: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
    let k_dim = targets.first().map_or(0, Vec::len);
    let mut rss = 0.0_f64;
    for (fm_row, tgt_row) in fm.iter().zip(targets) {
        for (d, &tv) in tgt_row.iter().enumerate().take(k_dim) {
            let mut pred = 0.0_f64;
            for f in 0..RANK_BASE_FM {
                pred += fm_row[f] * u.get(f).and_then(|r| r.get(d)).copied().unwrap_or(0.0);
            }
            let e = tv - pred;
            rss += e * e;
        }
    }
    rss
}

/// Refines Rank-4 FM slot factor weights (`v_base` or `v_ultra`) and readout `U` via alternating
/// least squares with exact self-term-subtracted coordinate updates, guaranteeing monotone RSS
/// reduction over the initial fixed `v` solve.
pub(super) fn fit_fm_als_u_and_v(
    slots: &mut [Cp1ContributorSlot],
    norm_table: &[Vec<f64>],
    descs: &[ControlClampDescriptor],
    samples: &[Cp1ControlSample],
    targets: &[Vec<f64>],
    reg: f64,
    use_ultra_v: bool,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_slots = slots.len();
    let phi = contributor_features_matrix(slots, norm_table, descs, samples);
    let cooccurs = slot_cooccurrence_mask(&phi, n_slots);
    for (sl, &co) in slots.iter_mut().zip(&cooccurs) {
        if !co {
            if use_ultra_v {
                sl.v_ultra = [0.0; RANK_BASE_FM];
            } else {
                sl.v_base = [0.0; RANK_BASE_FM];
            }
        }
    }
    let fm_reg = reg.max(1.0e-5);
    let v_reg = 1.0e-4_f64;
    let init_fm = eval_fm_pairs_with_self_sub(slots, norm_table, descs, samples, use_ultra_v);
    let init_u = solve_ols_ridge_uplo(&init_fm, targets, fm_reg);
    let init_rss = eval_fm_rss(&init_fm, &init_u, targets);
    if n_slots < 2 || samples.is_empty() || targets.is_empty() || targets[0].is_empty() {
        return (init_fm, init_u);
    }

    let n_obs = samples.len();
    let k_dim = targets[0].len();
    let orig_v: Vec<[f32; RANK_BASE_FM]> = slots
        .iter()
        .map(|sl| if use_ultra_v { sl.v_ultra } else { sl.v_base })
        .collect();
    let mut v_work: Vec<[f64; RANK_BASE_FM]> = slots
        .iter()
        .enumerate()
        .map(|(si, sl)| {
            if !cooccurs[si] {
                return [0.0; RANK_BASE_FM];
            }
            let arr = if use_ultra_v { &sl.v_ultra } else { &sl.v_base };
            core::array::from_fn(|f| f64::from(arr[f]))
        })
        .collect();
    let mut sum_v = vec![[0.0_f64; RANK_BASE_FM]; n_obs];
    let mut fm_work = vec![vec![0.0_f64; RANK_BASE_FM]; n_obs];
    let recompute_fm =
        |vw: &[[f64; RANK_BASE_FM]], sv: &mut [[f64; RANK_BASE_FM]], fm_buf: &mut [Vec<f64>]| {
            for i in 0..n_obs {
                let mut s = [0.0_f64; RANK_BASE_FM];
                let mut s2 = [0.0_f64; RANK_BASE_FM];
                for si in 0..n_slots {
                    let ph = phi[i][si];
                    for f in 0..RANK_BASE_FM {
                        let t = vw[si][f] * ph;
                        s[f] += t;
                        s2[f] += t * t;
                    }
                }
                sv[i] = s;
                for f in 0..RANK_BASE_FM {
                    fm_buf[i][f] = 0.5 * (s[f] * s[f] - s2[f]);
                }
            }
        };

    recompute_fm(&v_work, &mut sum_v, &mut fm_work);
    let mut err = vec![vec![0.0_f64; k_dim]; n_obs];

    for _sweep in 0..6 {
        let u = solve_ols_ridge_uplo(&fm_work, targets, fm_reg);
        for i in 0..n_obs {
            for d in 0..k_dim {
                let mut pred = 0.0_f64;
                for f in 0..RANK_BASE_FM {
                    pred += fm_work[i][f] * u[f][d];
                }
                err[i][d] = targets[i][d] - pred;
            }
        }
        for si in 0..n_slots {
            if !cooccurs[si] {
                continue;
            }
            for f in 0..RANK_BASE_FM {
                let u_sq: f64 = u[f].iter().map(|&uv| uv * uv).sum();
                if u_sq <= 1.0e-18 {
                    continue;
                }
                let v_old = v_work[si][f];
                let mut a_sq = 0.0_f64;
                let mut b_dot = 0.0_f64;
                for i in 0..n_obs {
                    let ph = phi[i][si];
                    if ph == 0.0 {
                        continue;
                    }
                    let rest = sum_v[i][f] - v_old * ph;
                    let ai = ph * rest;
                    if ai == 0.0 {
                        continue;
                    }
                    a_sq += ai * ai;
                    let eu: f64 = (0..k_dim).map(|d| err[i][d] * u[f][d]).sum();
                    b_dot += ai * eu;
                }
                let denom = a_sq * u_sq + v_reg;
                if denom <= 1.0e-18 {
                    continue;
                }
                let step = (b_dot - v_reg * v_old) / denom;
                let v_new = (v_old + step).clamp(-4.0, 4.0);
                let dv = v_new - v_old;
                if dv.abs() <= 1.0e-12 {
                    continue;
                }
                v_work[si][f] = v_new;
                for i in 0..n_obs {
                    let ph = phi[i][si];
                    if ph == 0.0 {
                        continue;
                    }
                    let rest = sum_v[i][f] - v_old * ph;
                    let ai = ph * rest;
                    sum_v[i][f] += dv * ph;
                    fm_work[i][f] += dv * ai;
                    for d in 0..k_dim {
                        err[i][d] -= dv * ai * u[f][d];
                    }
                }
            }
        }
        recompute_fm(&v_work, &mut sum_v, &mut fm_work);
    }

    let cand_v_f32: Vec<[f32; RANK_BASE_FM]> =
        v_work.iter().map(|row| row.map(|x| x as f32)).collect();
    for (sl, vf) in slots.iter_mut().zip(&cand_v_f32) {
        if use_ultra_v {
            sl.v_ultra = *vf;
        } else {
            sl.v_base = *vf;
        }
    }
    let cand_fm = eval_fm_pairs_with_self_sub(slots, norm_table, descs, samples, use_ultra_v);
    let cand_u = solve_ols_ridge_uplo(&cand_fm, targets, fm_reg);
    let cand_rss = eval_fm_rss(&cand_fm, &cand_u, targets);
    if cand_rss <= init_rss {
        (cand_fm, cand_u)
    } else {
        for (sl, vf) in slots.iter_mut().zip(&orig_v) {
            if use_ultra_v {
                sl.v_ultra = *vf;
            } else {
                sl.v_base = *vf;
            }
        }
        (init_fm, init_u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauge_invariant_dispatch_subspace_keeps_cext_cols04_zero_and_full_rank() {
        // Case 1: Independent rank-4 CEXT (16..31) + rank-4 base (0..16) -> zero reconstruction error.
        let mut t_joint = Vec::new();
        for i in 0..24 {
            let mut row = vec![0.0_f64; ACTIVE_DISPATCH_SLOTS];
            for k in 0..4 {
                let a = (((i * (k + 1)) % 9) as f64 - 4.0) * 0.25;
                let b = (((i * (k + 5)) % 7) as f64 - 3.0) * 0.2;
                row[k] = a + 0.5 * b;
                row[4 + k] = -0.3 * a;
                row[16 + k] = b;
                row[20 + k] = 0.4 * b;
            }
            t_joint.push(row);
        }
        let (z, w) = factor_gauge_invariant_dispatch_subspace(&t_joint, 2.5);
        for row in &w[16..] {
            assert_eq!(&row[..4], &[0.0; 4]);
        }
        for (i, row) in t_joint.iter().enumerate() {
            for r in 0..ACTIVE_DISPATCH_SLOTS {
                let pred: f64 = (0..RANK_STATIC_LATENT).map(|c| z[i][c] * w[r][c]).sum();
                assert!(
                    (pred - row[r]).abs() < 1.0e-8,
                    "i={i} r={r}: {pred} vs {}",
                    row[r]
                );
            }
        }

        // Case 2: Zero CEXT -> base uses all 8 columns (0 rank loss).
        let mut t_base_only = Vec::new();
        for i in 0..24 {
            let mut row = vec![0.0_f64; ACTIVE_DISPATCH_SLOTS];
            for k in 0..8 {
                let f = ((i as f64 + 1.0) * (k as f64 + 1.0) * 0.37).sin();
                row[k] = f;
                row[8 + k] = 0.5 * f;
            }
            t_base_only.push(row);
        }
        let (zb, wb) = factor_gauge_invariant_dispatch_subspace(&t_base_only, 2.5);
        for row in &wb[16..] {
            assert_eq!(*row, [0.0; RANK_STATIC_LATENT]);
        }
        for (i, row) in t_base_only.iter().enumerate() {
            for r in 0..16 {
                let pred: f64 = (0..RANK_STATIC_LATENT).map(|c| zb[i][c] * wb[r][c]).sum();
                assert!(
                    (pred - row[r]).abs() < 1.0e-8,
                    "rank-8 base i={i} r={r}: {pred} vs {}",
                    row[r]
                );
            }
        }
    }
}
