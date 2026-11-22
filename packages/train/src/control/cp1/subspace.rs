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
            if r == c {
                acc += reg.max(1.0e-9);
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

/// Factors `[N x 31]` target matrix via deterministic deflated symmetric power iteration into
/// latent coordinates `Z[N x R]` (bounded to `0.4 * trust_radius_mh`) and dispatch weights `W[31 x R]`.
pub(super) fn factor_dispatch_subspace<const R: usize>(
    targets: &[Vec<f64>],
    trust_radius_mh: f32,
) -> (Vec<[f64; R]>, [[f64; R]; ACTIVE_DISPATCH_SLOTS]) {
    let n_obs = targets.len();
    let mut gram = [[0.0_f64; ACTIVE_DISPATCH_SLOTS]; ACTIVE_DISPATCH_SLOTS];
    for row in targets {
        for j in 0..ACTIVE_DISPATCH_SLOTS {
            let rj = row[j];
            if rj == 0.0 {
                continue;
            }
            for i in 0..=j {
                let prod = row[i] * rj;
                gram[i][j] += prod;
                if i != j {
                    gram[j][i] += prod;
                }
            }
        }
    }

    let mut work_gram = gram;
    let mut basis = [[0.0_f64; ACTIVE_DISPATCH_SLOTS]; R];
    for k in 0..R {
        let mut best_col = 0_usize;
        let mut best_diag = 0.0_f64;
        for c in 0..ACTIVE_DISPATCH_SLOTS {
            if work_gram[c][c] > best_diag {
                best_diag = work_gram[c][c];
                best_col = c;
            }
        }
        if best_diag <= 1.0e-18 {
            break;
        }
        let mut v = work_gram[best_col];
        for _iter in 0..20 {
            let mut next = [0.0_f64; ACTIVE_DISPATCH_SLOTS];
            for r in 0..ACTIVE_DISPATCH_SLOTS {
                let mut acc = 0.0_f64;
                for c in 0..ACTIVE_DISPATCH_SLOTS {
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
                v = [0.0; ACTIVE_DISPATCH_SLOTS];
                break;
            }
            for (dst, val) in v.iter_mut().zip(next) {
                *dst = val / norm;
            }
        }
        let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm <= 1.0e-12 {
            break;
        }
        for dst in &mut v {
            *dst /= v_norm;
        }
        basis[k] = v;

        let mut cv = [0.0_f64; ACTIVE_DISPATCH_SLOTS];
        for r in 0..ACTIVE_DISPATCH_SLOTS {
            cv[r] = work_gram[r].iter().zip(&v).map(|(g, x)| g * x).sum();
        }
        let lambda: f64 = v.iter().zip(&cv).map(|(a, b)| a * b).sum();
        for r in 0..ACTIVE_DISPATCH_SLOTS {
            for c in 0..ACTIVE_DISPATCH_SLOTS {
                work_gram[r][c] -= lambda * v[r] * v[c];
            }
        }
    }

    let z_cap = 0.4 * f64::from(trust_radius_mh).max(0.25);
    let mut z_latents = vec![[0.0_f64; R]; n_obs];
    let mut w_disp = [[0.0_f64; R]; ACTIVE_DISPATCH_SLOTS];
    for k in 0..R {
        let mut max_abs = 0.0_f64;
        for (i, row) in targets.iter().enumerate() {
            let dot: f64 = row[..ACTIVE_DISPATCH_SLOTS]
                .iter()
                .zip(&basis[k])
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
            w_disp[s_idx][k] = basis[k][s_idx] * scale;
        }
    }
    (z_latents, w_disp)
}

pub(super) fn eval_fm_pairs_with_self_sub(
    slots: &[Cp1ContributorSlot],
    norm_table: &[Vec<f64>],
    descs: &[ControlClampDescriptor],
    samples: &[Cp1ControlSample],
    use_ultra_v: bool,
) -> Vec<Vec<f64>> {
    let n_obs = samples.len();
    let mut out = vec![vec![0.0_f64; RANK_BASE_FM]; n_obs];
    for (i, samp) in samples.iter().enumerate() {
        let mut sum_v = [0.0_f64; RANK_BASE_FM];
        let mut sum_v_sq = [0.0_f64; RANK_BASE_FM];
        for sl in slots {
            let c = usize::from(sl.control_idx);
            let z_c = if sl.ctrl_kind == 0 {
                let u_p = norm_table[i][c] as f32;
                let lob = descs[c].eval_lobatto_basis(u_p);
                f64::from(sl.emb[0] * lob[0] + sl.emb[1] * lob[1] + sl.emb[2] * lob[2])
            } else if samp.switch_options.get(c).copied().unwrap_or(0) == sl.aux {
                1.0
            } else {
                0.0
            };
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
