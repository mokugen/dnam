//! Main-effect `psi(u)` basis evaluation, `M = 5` deterministic ChaCha20/Cholesky ridge
//! regression, and candidate disagreement + leverage variance scoring for `PANAMA`
//! (`ref[train.audit.report.prescription]`).

use sha2::{Digest, Sha256};
use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::U32;

use crate::error::TrainError;
use crate::linalg::{factorize_uplo_u_cholesky, solve_factorized_uplo_u};
use crate::wide_gram::uplo_u_index;

use super::candidates::{AxisKind, VaryingAxis, round_half_away_i64};
use super::chacha::ChaCha20Stream;
use super::{PANAMA_ENSEMBLE_SIZE, PANAMA_GAMMA_LEV, PANAMA_LAMBDA_CP1, PANAMA_LAMBDA_MULTS};

pub(super) fn compute_main_effect_dim(axes: &[VaryingAxis]) -> usize {
    axes.iter()
        .map(|ax| match &ax.kind {
            AxisKind::Range { desc, .. } => usize::from(desc.lobatto_degree).clamp(1, 3),
            AxisKind::Switch {
                sorted_active_opts,
                star_center_opt,
                ..
            } => sorted_active_opts
                .iter()
                .filter(|&&o| o != *star_center_opt)
                .count(),
        })
        .sum()
}

pub(super) fn eval_main_effect_psi(axes: &[VaryingAxis], var_vals: &[f64]) -> Vec<f64> {
    let mut psi = Vec::with_capacity(compute_main_effect_dim(axes));
    for (ax, &val) in axes.iter().zip(var_vals.iter()) {
        match &ax.kind {
            AxisKind::Range {
                desc,
                u_star_norm,
                star_lobatto,
                ..
            } => {
                let deg = usize::from(desc.lobatto_degree).clamp(1, 3);
                let u_p = ((val as f32) * desc.scale_norm + desc.offset_norm).clamp(-1.0, 1.0);
                if desc.lobatto_degree == 0 || desc.ctrl_kind != 0 {
                    psi.push(f64::from(u_p - *u_star_norm));
                } else {
                    let b = desc.eval_lobatto_basis(u_p);
                    for d in 0..deg {
                        psi.push(f64::from(b[d] - star_lobatto[d]));
                    }
                }
            }
            AxisKind::Switch {
                sorted_active_opts,
                star_center_opt,
                ..
            } => {
                let opt_val = round_half_away_i64(val).clamp(0, i64::from(u8::MAX)) as u8;
                for &opt in sorted_active_opts {
                    if opt != *star_center_opt {
                        psi.push(if opt_val == opt { 1.0 } else { 0.0 });
                    }
                }
            }
        }
    }
    psi
}

pub(super) struct EnsembleFitCtx<'a> {
    pub axes_dim: usize,
    pub anchor_gram: &'a [f64],
    pub anchor_rhs: &'a [f64],
    pub interior_indices: &'a [usize],
    pub psi_rows: &'a [Vec<f64>],
    pub targets: &'a [[f64; 31]],
    pub w_sens: &'a [f64; 31],
    pub stems_key: &'a str,
}

pub(super) fn fit_ensemble_members(
    ctx: &EnsembleFitCtx<'_>,
) -> Result<Vec<Vec<[f64; 31]>>, TrainError> {
    let axes_dim = ctx.axes_dim;
    let packed_len = ctx.anchor_gram.len();
    let n_int = ctx.interior_indices.len();
    let mut members = Vec::with_capacity(PANAMA_ENSEMBLE_SIZE);

    for (m, &mult) in PANAMA_LAMBDA_MULTS.iter().enumerate() {
        let mut g_m = ctx.anchor_gram.to_vec();
        let mut r_m = ctx.anchor_rhs.to_vec();
        if n_int > 0 {
            let mut hasher = Sha256::new();
            hasher.update(b"DNAM-PANAMA-V1:");
            hasher.update(ctx.stems_key.as_bytes());
            hasher.update(U32::new(m as u32).as_bytes());
            let key: [u8; 32] = hasher.finalize().into();
            let mut stream = ChaCha20Stream::new(key, [0_u8; 12], 0);
            let mut counts = vec![0_usize; n_int];
            for _ in 0..n_int {
                let draw = stream.draw_uniform_index(n_int);
                counts[draw] += 1;
            }
            for (&row_idx, &cnt) in ctx.interior_indices.iter().zip(counts.iter()) {
                if cnt == 0 {
                    continue;
                }
                let wt = cnt as f64;
                let psi = &ctx.psi_rows[row_idx];
                let tgt = &ctx.targets[row_idx];
                for j in 0..axes_dim {
                    let pj = psi[j] * wt;
                    for (i, &pi) in psi.iter().enumerate().take(j + 1) {
                        g_m[uplo_u_index(i, j)] += pi * pj;
                    }
                }
                for r in 0..31 {
                    let tr = tgt[r] * wt;
                    if tr != 0.0 {
                        let base = r * axes_dim;
                        for c in 0..axes_dim {
                            r_m[base + c] += psi[c] * tr;
                        }
                    }
                }
            }
        }
        debug_assert_eq!(g_m.len(), packed_len);
        let r_factor = factorize_uplo_u_cholesky(axes_dim, &g_m, mult * PANAMA_LAMBDA_CP1)?;
        let mut weights_by_col = vec![[0.0_f64; 31]; axes_dim];
        for r in 0..31 {
            if ctx.w_sens[r] == 0.0 {
                continue;
            }
            let rhs_slice = &r_m[r * axes_dim..(r + 1) * axes_dim];
            let solved = solve_factorized_uplo_u(axes_dim, &r_factor, rhs_slice)?;
            for (c, &val) in solved.iter().enumerate() {
                weights_by_col[c][r] = val;
            }
        }
        members.push(weights_by_col);
    }

    Ok(members)
}

pub(super) fn score_candidate_sigma2(
    axes: &[VaryingAxis],
    cand_var: &[f64],
    members: &[Vec<[f64; 31]>],
    w_sens: &[f64; 31],
    r0_factor: &[f64],
) -> Result<f64, TrainError> {
    let psi_c = eval_main_effect_psi(axes, cand_var);
    let m_f64 = PANAMA_ENSEMBLE_SIZE as f64;
    let mut pred = [[0.0_f64; 31]; PANAMA_ENSEMBLE_SIZE];
    let mut mean_t = [0.0_f64; 31];
    for (m, weights) in members.iter().enumerate() {
        for (c, &pc) in psi_c.iter().enumerate() {
            if pc == 0.0 {
                continue;
            }
            for r in 0..31 {
                pred[m][r] += weights[c][r] * pc;
            }
        }
        for r in 0..31 {
            mean_t[r] += pred[m][r];
        }
    }
    for mt in &mut mean_t {
        *mt /= m_f64;
    }

    let mut ens_var = 0.0_f64;
    for row in &pred {
        for r in 0..31 {
            let d = w_sens[r] * (row[r] - mean_t[r]);
            ens_var += d * d;
        }
    }
    ens_var /= (PANAMA_ENSEMBLE_SIZE - 1) as f64;

    let lev_vec = solve_factorized_uplo_u(psi_c.len(), r0_factor, &psi_c)?;
    let lev = psi_c
        .iter()
        .zip(lev_vec.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .max(0.0);
    Ok(ens_var + PANAMA_GAMMA_LEV * lev)
}

#[cfg(test)]
mod tests {
    use dnam_vm::bin::{ControlClampDescriptorWire, DnamBinView};
    use dnam_vm::control::descriptor::compute_scale_offset_norm;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    use crate::control::panama::{PanamaControlDesc, PanamaControlKind};
    use crate::solver::model::SonaModelF64;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};

    use super::super::candidates::build_varying_axes;
    use super::*;

    #[test]
    fn test_eval_main_effect_psi_vanishes_at_star_center_and_ensemble_fit_determinism() {
        let model = SonaModelF64::new(48_000.0);
        let bytes = encode_model_to_dnam_bin(
            &model,
            Pass5EmitConfig {
                calibrate_lufs_trims: false,
                ..Pass5EmitConfig::default()
            },
        )
        .unwrap();
        let mut view = DnamBinView::parse(&bytes).unwrap();
        let (scale, offset) = compute_scale_offset_norm(0.0, 10.0);
        let mut d0 = ControlClampDescriptorWire::new_zeroed();
        d0.lobatto_degree = 3;
        d0.default_val = F32::new(5.0);
        d0.safe_min_base = F32::new(0.0);
        d0.safe_max_base = F32::new(10.0);
        d0.scale_norm = F32::new(scale);
        d0.offset_norm = F32::new(offset);
        d0.anchor_nodes_norm = [F32::new(0.0), F32::new(-1.0), F32::new(1.0)];

        let mut d1 = ControlClampDescriptorWire::new_zeroed();
        d1.ctrl_kind = 1;
        d1.lobatto_degree = 1;
        d1.active_options_mask = U16::new(0b11);

        let descs_wire = [d0, d1];
        view.control_descriptors = &descs_wire;

        let ctrls = vec![
            PanamaControlDesc {
                control_id: "a_gain".into(),
                wire_ctrl_idx: 0,
                kind: PanamaControlKind::Range {
                    safe_min: 0.0,
                    safe_max: 10.0,
                    star_center: 5.0,
                },
            },
            PanamaControlDesc {
                control_id: "b_mode".into(),
                wire_ctrl_idx: 1,
                kind: PanamaControlKind::Switch {
                    active_options: vec![0, 1],
                    declared_option_names: vec!["clean".into(), "hot".into()],
                    star_center_option: 0,
                },
            },
        ];
        let (axes, _) = build_varying_axes(&view, &ctrls).unwrap();
        assert_eq!(compute_main_effect_dim(&axes), 4);
        assert_eq!(eval_main_effect_psi(&axes, &[5.0, 0.0]), vec![0.0; 4]);
        let non_star = eval_main_effect_psi(&axes, &[10.0, 1.0]);
        assert!(non_star[0].abs() > 0.5 && non_star[3] == 1.0);
    }
}
