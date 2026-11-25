//! Ultra `W_ctrl_ultra` + Rank-8 FM delta fitting split from [`super`] (`< 900` LOC ceiling).

use dnam_vm::bin::{Arena0ControlBusWire, Arena3ControlBusWire, CextDispatchWire};
use dnam_vm::control::cp1::Cp1ContributorSlot;
use dnam_vm::control::descriptor::ControlClampDescriptor;
use zerocopy::byteorder::little_endian::F32;

use super::super::Cp1ControlSample;
use super::super::subspace::{
    ACTIVE_DISPATCH_SLOTS, RANK_BASE_FM, RANK_STATIC_LATENT, RANK_ULTRA_LATENT,
    factor_dispatch_subspace, fit_fm_als_u_and_v, solve_ols_ridge_uplo,
};
use super::Cp1FitConfig;

const UINT_ULTRA_BLOCKS: usize = 3;
const ULTRA_SUBSPACE_MIN_REL_ERR: f64 = 1.0e-2;
const _: () = assert!(ULTRA_SUBSPACE_MIN_REL_ERR > 0.0 && ULTRA_SUBSPACE_MIN_REL_ERR < 0.25);

pub(super) struct UltraFitInputs<'a> {
    pub basis_cols: &'a [(usize, usize)],
    pub design_main: &'a [Vec<f64>],
    pub norm_table: &'a [Vec<f64>],
    pub descs: &'a [ControlClampDescriptor],
    pub samples: &'a [Cp1ControlSample],
    pub target_31: &'a [Vec<f64>],
    pub target_u_delta_31: &'a [Vec<f64>],
    pub z_base_target: &'a [Vec<f64>],
    pub z_base_hat: &'a [[f64; 8]],
    pub fm_pair_base: Option<&'a [Vec<f64>]>,
    pub config: Cp1FitConfig,
    pub a0_cb: &'a Arena0ControlBusWire,
}

pub(super) struct UltraFitOutputs<'a> {
    pub slots: &'a mut [Cp1ContributorSlot],
    pub a3_cb: &'a mut Arena3ControlBusWire,
    pub cext_wire: &'a mut CextDispatchWire,
    pub need_cext: &'a mut bool,
}

pub(super) fn fit_ultra_dispatch_and_fm(inp: &UltraFitInputs<'_>, out: &mut UltraFitOutputs<'_>) {
    let n_obs = inp.samples.len();
    let has_explicit_ultra_delta = inp
        .target_u_delta_31
        .iter()
        .any(|row| row.iter().any(|&v| v.abs() > 1.0e-7));
    let has_active_fm = inp
        .fm_pair_base
        .is_some_and(|rows| rows.iter().any(|row| row.iter().any(|&v| v.abs() > 1.0e-7)));
    let mut target_u_res = vec![vec![0.0_f64; ACTIVE_DISPATCH_SLOTS]; n_obs];
    let (mut tgt_sq, mut subspace_err_sq) = (0.0_f64, 0.0_f64);
    for i in 0..n_obs {
        for r in 0..ACTIVE_DISPATCH_SLOTS {
            let mut base_pred = 0.0_f64;
            let mut sub_pred = 0.0_f64;
            let col_start = if r < 16 { 0 } else { 4 };
            for c in col_start..RANK_STATIC_LATENT {
                let w = if r < 16 {
                    inp.a0_cb.w_ctrl[r][c].get()
                } else {
                    out.cext_wire.w_ctrl_ext[r - 16][c].get()
                };
                let wf = f64::from(w);
                base_pred += inp.z_base_hat[i][c] * wf;
                sub_pred += inp.z_base_target[i][c] * wf;
            }
            let raw_t = inp.target_31[i][r];
            let res = (raw_t + inp.target_u_delta_31[i][r]) - base_pred;
            let sub_res = raw_t - sub_pred;
            tgt_sq += raw_t * raw_t;
            subspace_err_sq += sub_res * sub_res;
            target_u_res[i][r] = res;
        }
    }

    if !has_explicit_ultra_delta
        && !has_active_fm
        && subspace_err_sq <= ULTRA_SUBSPACE_MIN_REL_ERR * tgt_sq.max(1.0e-18)
    {
        out.cext_wire.w_ctrl_ultra_ext[15].fill(F32::new(0.0));
        return;
    }

    let (z_u4_target, w_ultra_31) =
        factor_dispatch_subspace::<RANK_ULTRA_LATENT>(&target_u_res, inp.config.trust_radius_mh);
    let mut z_u8_target = vec![vec![0.0_f64; 8]; n_obs];
    for i in 0..n_obs {
        z_u8_target[i][4..8].copy_from_slice(&z_u4_target[i]);
    }

    let w_u = solve_ols_ridge_uplo(inp.design_main, &z_u8_target, inp.config.l2_reg);
    for (col, &(si, l)) in inp.basis_cols.iter().enumerate() {
        if let Some(row) = w_u.get(col) {
            for d in 0..8 {
                out.slots[si].main_ultra[l][d] = row[d] as f32;
            }
        }
    }

    if let Some(fm_base) = inp.fm_pair_base {
        let mut z_u8_res = vec![vec![0.0_f64; 8]; n_obs];
        let mut z_base_res = vec![vec![0.0_f64; 8]; n_obs];
        let mut z_joint_target = Vec::with_capacity(n_obs);
        for i in 0..n_obs {
            for d in 0..8 {
                let mut u_lin = 0.0_f64;
                for (col, &(si, l)) in inp.basis_cols.iter().enumerate() {
                    u_lin += inp.design_main[i][col] * f64::from(out.slots[si].main_ultra[l][d]);
                }
                z_u8_res[i][d] = z_u8_target[i][d] - u_lin;
                z_base_res[i][d] = inp.z_base_target[i][d] - inp.z_base_hat[i][d];
            }
            let mut row = Vec::with_capacity(12);
            row.extend_from_slice(&z_base_res[i]);
            row.extend_from_slice(&z_u8_res[i][4..8]);
            z_joint_target.push(row);
        }
        let (fm_ultra, _) = fit_fm_als_u_and_v(
            out.slots,
            inp.norm_table,
            inp.descs,
            inp.samples,
            &z_joint_target,
            inp.config.l2_reg,
            true,
        );
        let mut fm_joint = Vec::with_capacity(n_obs);
        for i in 0..n_obs {
            let mut row = Vec::with_capacity(2 * RANK_BASE_FM);
            row.extend_from_slice(&fm_base[i]);
            row.extend_from_slice(&fm_ultra[i]);
            fm_joint.push(row);
        }

        let fm_reg = inp.config.l2_reg.max(1.0e-5);
        let u0_fit = solve_ols_ridge_uplo(&fm_ultra, &z_base_res, fm_reg);
        let u12_fit = solve_ols_ridge_uplo(&fm_joint, &z_u8_res, fm_reg);
        let _ = UINT_ULTRA_BLOCKS;
        for f in 0..RANK_BASE_FM {
            if let Some(r0) = u0_fit.get(f) {
                for d in 0..8 {
                    out.a3_cb.u_int_ultra[0][f][d] = F32::new(r0[d] as f32);
                }
            }
            if let Some(r1) = u12_fit.get(f) {
                for d in 0..8 {
                    out.a3_cb.u_int_ultra[1][f][d] = F32::new(r1[d] as f32);
                }
            }
            if let Some(r2) = u12_fit.get(RANK_BASE_FM + f) {
                for d in 0..8 {
                    out.a3_cb.u_int_ultra[2][f][d] = F32::new(r2[d] as f32);
                }
            }
        }
    }

    out.cext_wire.w_ctrl_ultra_ext[15].fill(F32::new(0.0));
    for (s_idx, row) in w_ultra_31.iter().enumerate() {
        for c in 0..RANK_ULTRA_LATENT {
            let w = row[c] as f32;
            if s_idx < 16 {
                out.a3_cb.w_ctrl_ultra[s_idx][c] = F32::new(w);
            } else {
                if w.abs() > 1.0e-7 {
                    *out.need_cext = true;
                }
                out.cext_wire.w_ctrl_ultra_ext[s_idx - 16][c] = F32::new(w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromZeros;

    #[test]
    fn test_ultra_dispatch_skips_noise_residuals_unless_explicit_delta_present() {
        let descs = vec![ControlClampDescriptor::range(
            -1.0,
            1.0,
            0.0,
            3,
            [0.0, -1.0, 1.0],
        )];
        let mut a0_cb = Arena0ControlBusWire::new_zeroed();
        a0_cb.w_ctrl[0][0] = F32::new(1.0);
        let z_base_hat = vec![
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0; 8],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let samples = [-1.0_f32, 0.0, 1.0]
            .into_iter()
            .map(|u| Cp1ControlSample {
                phys_controls: vec![u],
                switch_options: vec![0],
                stage_targets_base: [0.0; 32],
                stage_targets_ultra_delta: None,
            })
            .collect::<Vec<_>>();
        let design_main = vec![vec![-1.0], vec![0.0], vec![1.0]];
        let norm_table = vec![vec![-1.0], vec![0.0], vec![1.0]];
        let mut t31 = vec![vec![0.0; ACTIVE_DISPATCH_SLOTS]; 3];
        t31[0][0] = -1.0 + 1.0e-3;
        t31[1][0] = 1.0e-3;
        t31[2][0] = 1.0 - 1.0e-3;

        let zero_u = vec![vec![0.0; ACTIVE_DISPATCH_SLOTS]; 3];
        let mut slots = vec![Cp1ContributorSlot::default()];
        let mut a3_cb = Arena3ControlBusWire::new_zeroed();
        let mut cext_wire = CextDispatchWire::new_zeroed();
        let mut need_cext = false;

        fit_ultra_dispatch_and_fm(
            &UltraFitInputs {
                basis_cols: &[(0, 0)],
                design_main: &design_main,
                norm_table: &norm_table,
                descs: &descs,
                samples: &samples,
                target_31: &t31,
                target_u_delta_31: &zero_u,
                z_base_target: &z_base_hat.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                z_base_hat: &z_base_hat,
                fm_pair_base: None,
                config: Cp1FitConfig::default(),
                a0_cb: &a0_cb,
            },
            &mut UltraFitOutputs {
                slots: &mut slots,
                a3_cb: &mut a3_cb,
                cext_wire: &mut cext_wire,
                need_cext: &mut need_cext,
            },
        );
        assert!(
            a3_cb.w_ctrl_ultra.iter().flatten().all(|w| w.get() == 0.0),
            "tiny base residual must leave w_ctrl_ultra zero"
        );
        assert_eq!(slots[0].main_ultra, [[0.0; 8]; 3]);

        // All-zero fm_pair_base with non-zero Lobatto residual (`z_base_hat != z_base_target`)
        // but zero subspace truncation error must still skip noisy Ultra residual fitting.
        let zero_fm = vec![vec![0.0_f64; RANK_BASE_FM]; 3];
        let z_sub_exact: Vec<Vec<f64>> = t31
            .iter()
            .map(|r| vec![r[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect();
        let z_hat_noisy = [
            [-0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        fit_ultra_dispatch_and_fm(
            &UltraFitInputs {
                basis_cols: &[(0, 0)],
                design_main: &design_main,
                norm_table: &norm_table,
                descs: &descs,
                samples: &samples,
                target_31: &t31,
                target_u_delta_31: &zero_u,
                z_base_target: &z_sub_exact,
                z_base_hat: &z_hat_noisy,
                fm_pair_base: Some(&zero_fm),
                config: Cp1FitConfig::default(),
                a0_cb: &a0_cb,
            },
            &mut UltraFitOutputs {
                slots: &mut slots,
                a3_cb: &mut a3_cb,
                cext_wire: &mut cext_wire,
                need_cext: &mut need_cext,
            },
        );
        assert!(
            a3_cb.w_ctrl_ultra.iter().flatten().all(|w| w.get() == 0.0),
            "all-zero fm_pair_base with zero subspace error must leave w_ctrl_ultra zero"
        );
        assert_eq!(slots[0].main_ultra, [[0.0; 8]; 3]);

        // Active non-zero fm_pair_base must still populate Ultra FM/dispatch weights.
        let active_fm = vec![
            vec![0.4, -0.2, 0.0, 0.0],
            vec![0.0; RANK_BASE_FM],
            vec![0.35, 0.15, 0.0, 0.0],
        ];
        fit_ultra_dispatch_and_fm(
            &UltraFitInputs {
                basis_cols: &[(0, 0)],
                design_main: &design_main,
                norm_table: &norm_table,
                descs: &descs,
                samples: &samples,
                target_31: &t31,
                target_u_delta_31: &zero_u,
                z_base_target: &z_sub_exact,
                z_base_hat: &z_hat_noisy,
                fm_pair_base: Some(&active_fm),
                config: Cp1FitConfig::default(),
                a0_cb: &a0_cb,
            },
            &mut UltraFitOutputs {
                slots: &mut slots,
                a3_cb: &mut a3_cb,
                cext_wire: &mut cext_wire,
                need_cext: &mut need_cext,
            },
        );
        assert!(
            a3_cb.w_ctrl_ultra[0].iter().any(|w| w.get().abs() > 1.0e-3),
            "active non-zero fm_pair_base must populate Ultra dispatch weights"
        );

        // Explicit Ultra delta must still fit non-zero Ultra weights.
        let mut a3_cb_exp = Arena3ControlBusWire::new_zeroed();
        let mut explicit_u = vec![vec![0.0; ACTIVE_DISPATCH_SLOTS]; 3];
        explicit_u[0][1] = -0.35;
        explicit_u[2][1] = 0.35;
        fit_ultra_dispatch_and_fm(
            &UltraFitInputs {
                basis_cols: &[(0, 0)],
                design_main: &design_main,
                norm_table: &norm_table,
                descs: &descs,
                samples: &samples,
                target_31: &t31,
                target_u_delta_31: &explicit_u,
                z_base_target: &z_base_hat.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                z_base_hat: &z_base_hat,
                fm_pair_base: None,
                config: Cp1FitConfig::default(),
                a0_cb: &a0_cb,
            },
            &mut UltraFitOutputs {
                slots: &mut slots,
                a3_cb: &mut a3_cb_exp,
                cext_wire: &mut cext_wire,
                need_cext: &mut need_cext,
            },
        );
        assert!(
            a3_cb_exp.w_ctrl_ultra[1]
                .iter()
                .any(|w| w.get().abs() > 1.0e-3),
            "explicit delta must populate w_ctrl_ultra[1]"
        );
    }
}
