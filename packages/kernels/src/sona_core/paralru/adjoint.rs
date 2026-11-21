//! Reverse-BPTT `O(log2 N)` parallel suffix adjoint scan for decoupled `1D` scalar `ParaLRU`
//! (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::scans::{LinearScanElem, apply_linear_scan, compose_linear_scan};
use super::{ParaLruParams, soft_rect_zero_dc_f32};

/// Forms the reverse-BPTT linear adjoint scan element `(J_n, J_n * adj_v_n)` at step `n`.
#[inline(always)]
#[must_use]
pub fn paralru_adjoint_step_elem(
    u: f32,
    v_prev: f32,
    params: ParaLruParams,
    adj_v: f32,
) -> LinearScanElem {
    let z = u - params.kappa * params.alpha_rel * v_prev;
    let (_, phi_prime) = soft_rect_zero_dc_f32(z);
    let one_minus_atk = 1.0 - params.alpha_atk;
    let jac = params.alpha_rel * (1.0 - params.kappa * one_minus_atk * phi_prime);
    LinearScanElem {
        a: jac,
        b: jac * adj_v,
    }
}

/// Composes two reverse-BPTT `ParaLRU` adjoint elements (`earlier_in_reverse` then `later_in_reverse`).
#[inline(always)]
#[must_use]
pub fn compose_paralru_adjoint_scan(
    later_in_reverse: LinearScanElem,
    earlier_in_reverse: LinearScanElem,
) -> LinearScanElem {
    compose_linear_scan(earlier_in_reverse, later_in_reverse)
}

/// Reverse-BPTT gradients and cross-tile summary computed over an `N`-sample `ParaLRU` tile.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParaLruTileGrads<const N: usize> {
    pub adj_u: [f32; N],
    pub adj_v_init: f32,
    pub adj_alpha_rel: f32,
    pub adj_alpha_atk: f32,
    pub adj_kappa: f32,
    pub lambda: [f32; N],
    /// Level-2 cross-tile reverse adjoint summary (`suffix[0]`, `(a_tile, b_tile)`).
    pub tile_summary: LinearScanElem,
}

/// In-place `O(log2 N)` Hillis-Steele doubling inclusive suffix scan (`step = 1, 2, 4, ... < N`).
#[inline]
pub(super) fn parallel_suffix_scan<T: Copy, const N: usize>(
    elems: &mut [T; N],
    combine: impl Fn(T, T) -> T,
) {
    let mut step = 1_usize;
    while step < N {
        let limit = N - step;
        let mut i = 0_usize;
        while i < limit {
            elems[i] = combine(elems[i], elems[i + step]);
            i += 1;
        }
        step <<= 1;
    }
}

/// Executes the `O(log2 N)` backward linear adjoint suffix scan over `valid_len.min(N)` active
/// frames of a `ParaLRU` tile with incoming boundary sensitivity `adj_v_out` (`0` truncated-BPTT
/// loss across `256`-sample tile boundaries), returning gradients and `tile_summary`.
#[inline]
#[must_use]
pub fn run_paralru_adjoint_scan_partial_tile<const N: usize>(
    u: &[f32; N],
    v_traj: &[f32; N],
    valid_len: usize,
    v_init: f32,
    params: ParaLruParams,
    adj_v: &[f32; N],
    adj_v_out: f32,
) -> ParaLruTileGrads<N> {
    let mut adj_u = [0.0_f32; N];
    let mut lambda = [0.0_f32; N];
    let active = valid_len.min(N);
    if active == 0 {
        return ParaLruTileGrads {
            adj_u,
            adj_v_init: adj_v_out,
            adj_alpha_rel: 0.0,
            adj_alpha_atk: 0.0,
            adj_kappa: 0.0,
            lambda,
            tile_summary: LinearScanElem::identity(),
        };
    }

    let mut adj_elems = [LinearScanElem::identity(); N];
    for i in 0..active {
        let v_prev = if i == 0 { v_init } else { v_traj[i - 1] };
        adj_elems[i] = paralru_adjoint_step_elem(u[i], v_prev, params, adj_v[i]);
    }
    parallel_suffix_scan(&mut adj_elems, compose_paralru_adjoint_scan);

    let one_minus_atk = 1.0 - params.alpha_atk;
    let mut adj_alpha_rel = 0.0_f32;
    let mut adj_alpha_atk = 0.0_f32;
    let mut adj_kappa = 0.0_f32;

    for i in 0..active {
        let v_prev = if i == 0 { v_init } else { v_traj[i - 1] };
        let suffix_after = if i + 1 < active {
            adj_elems[i + 1]
        } else {
            LinearScanElem::identity()
        };
        let lam_i = adj_v[i] + apply_linear_scan(suffix_after, adj_v_out);
        lambda[i] = lam_i;

        let z = u[i] - params.kappa * params.alpha_rel * v_prev;
        let (phi, phi_prime) = soft_rect_zero_dc_f32(z);
        adj_u[i] = lam_i * one_minus_atk * phi_prime;
        adj_alpha_rel += lam_i * v_prev * (1.0 - params.kappa * one_minus_atk * phi_prime);
        adj_alpha_atk -= lam_i * phi;
        adj_kappa -= lam_i * one_minus_atk * params.alpha_rel * v_prev * phi_prime;
    }

    let tile_summary = adj_elems[0];
    ParaLruTileGrads {
        adj_u,
        adj_v_init: apply_linear_scan(tile_summary, adj_v_out),
        adj_alpha_rel,
        adj_alpha_atk,
        adj_kappa,
        lambda,
        tile_summary,
    }
}

/// Executes the `O(log2 N)` backward linear adjoint suffix scan over all `N` frames of a `ParaLRU`
/// tile with incoming cross-tile sensitivity `adj_v_out`.
#[inline]
#[must_use]
pub fn run_paralru_adjoint_scan_tile<const N: usize>(
    u: &[f32; N],
    v_traj: &[f32; N],
    v_init: f32,
    params: ParaLruParams,
    adj_v: &[f32; N],
    adj_v_out: f32,
) -> ParaLruTileGrads<N> {
    run_paralru_adjoint_scan_partial_tile(u, v_traj, N, v_init, params, adj_v, adj_v_out)
}

#[cfg(test)]
mod tests {
    use super::super::step_paralru_vec4;
    use super::*;
    use crate::simd::{GpuVec4, KernelVec4, MAX_PARALRU_R_F32};
    use proptest::prelude::*;

    fn soft_rect_f64(z: f64) -> f64 {
        let eps = super::super::SOFT_RECT_EPS as f64;
        0.5 * (z + (z * z + eps * eps).sqrt() - eps)
    }

    fn paralru_loss_f64(
        u: &[f64],
        v_init: f64,
        alpha_rel: f64,
        alpha_atk: f64,
        kappa: f64,
        adj_v: &[f64],
        inject: Option<(usize, f64)>,
    ) -> f64 {
        let mut v = v_init;
        let mut loss = 0.0;
        for (idx, &ui) in u.iter().enumerate() {
            let z = ui - kappa * alpha_rel * v;
            v = alpha_rel * v + (1.0 - alpha_atk) * soft_rect_f64(z);
            if let Some((t_inj, delta)) = inject
                && idx == t_inj
            {
                v += delta;
            }
            loss += adj_v[idx] * v;
        }
        loss
    }

    #[test]
    fn test_paralru_adjoint_edge_cases_and_mixed_sign_relative_fd() {
        let clamped = ParaLruParams::clamped(2.0, -0.5, 1.5);
        let empty = run_paralru_adjoint_scan_tile::<0>(&[], &[], 0.0, clamped, &[], 0.4);
        assert_eq!(empty.adj_v_init, 0.4);
        assert_eq!(empty.tile_summary, LinearScanElem::identity());
        let partial0 = run_paralru_adjoint_scan_partial_tile::<4>(
            &[1.0; 4], &[1.0; 4], 0, 0.0, clamped, &[1.0; 4], -0.3,
        );
        assert_eq!(partial0.adj_v_init, -0.3);

        let step_adj =
            paralru_adjoint_step_elem(0.5, 0.2, ParaLruParams::clamped(0.95, 0.4, 0.7), 1.0);
        let id = LinearScanElem::identity();
        assert_eq!(compose_paralru_adjoint_scan(step_adj, id), step_adj);
        assert_eq!(compose_paralru_adjoint_scan(id, step_adj), step_adj);

        let det_u = [1.5_f32, -2.0, 0.8, -0.6, 2.4, -1.2, 0.5, -1.8];
        let det_adj = [0.7_f32, 0.4, 0.9, 0.3, 0.6, 0.5, 0.8, 0.2];
        let det_v0 = -1.25_f32;
        let det_p = ParaLruParams::clamped(0.85, 0.3, 0.6);
        let mut det_traj = [0.0_f32; 8];
        let mut det_st = GpuVec4::splat(det_v0);
        for (idx, &ui) in det_u.iter().enumerate() {
            det_traj[idx] = step_paralru_vec4(
                GpuVec4::splat(ui),
                &mut det_st,
                GpuVec4::splat(det_p.alpha_rel),
                GpuVec4::splat(det_p.alpha_atk),
                GpuVec4::splat(det_p.kappa),
            )
            .to_array()[0];
        }
        let det_g =
            run_paralru_adjoint_scan_tile::<8>(&det_u, &det_traj, det_v0, det_p, &det_adj, 0.0);
        let (du64, da64) = (det_u.map(f64::from), det_adj.map(f64::from));
        let (dv0, dar, daa, dkp) = (
            det_v0 as f64,
            det_p.alpha_rel as f64,
            det_p.alpha_atk as f64,
            det_p.kappa as f64,
        );
        let eps = 1.0e-6_f64;
        let rel_ok = |got: f32, fd: f64| {
            let abs_err = (got as f64 - fd).abs();
            let tol = 1.0e-4 * fd.abs() + 2.0e-8;
            assert!(
                abs_err <= tol,
                "got={got}, fd={fd}, abs_err={abs_err}, tol={tol}"
            );
        };
        rel_ok(
            det_g.adj_v_init,
            (paralru_loss_f64(&du64, dv0 + eps, dar, daa, dkp, &da64, None)
                - paralru_loss_f64(&du64, dv0 - eps, dar, daa, dkp, &da64, None))
                / (2.0 * eps),
        );
        rel_ok(
            det_g.adj_alpha_rel,
            (paralru_loss_f64(&du64, dv0, dar + eps, daa, dkp, &da64, None)
                - paralru_loss_f64(&du64, dv0, dar - eps, daa, dkp, &da64, None))
                / (2.0 * eps),
        );
        rel_ok(
            det_g.adj_alpha_atk,
            (paralru_loss_f64(&du64, dv0, dar, daa + eps, dkp, &da64, None)
                - paralru_loss_f64(&du64, dv0, dar, daa - eps, dkp, &da64, None))
                / (2.0 * eps),
        );
        rel_ok(
            det_g.adj_kappa,
            (paralru_loss_f64(&du64, dv0, dar, daa, dkp + eps, &da64, None)
                - paralru_loss_f64(&du64, dv0, dar, daa, dkp - eps, &da64, None))
                / (2.0 * eps),
        );
        for t in 0..8 {
            let (mut up, mut um) = (du64, du64);
            up[t] += eps;
            um[t] -= eps;
            rel_ok(
                det_g.adj_u[t],
                (paralru_loss_f64(&up, dv0, dar, daa, dkp, &da64, None)
                    - paralru_loss_f64(&um, dv0, dar, daa, dkp, &da64, None))
                    / (2.0 * eps),
            );
            rel_ok(
                det_g.lambda[t],
                (paralru_loss_f64(&du64, dv0, dar, daa, dkp, &da64, Some((t, eps)))
                    - paralru_loss_f64(&du64, dv0, dar, daa, dkp, &da64, Some((t, -eps))))
                    / (2.0 * eps),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_paralru_adjoint_scan_matches_finite_differences(
            u_vec in proptest::collection::vec(-4.0_f32..4.0_f32, 16),
            adj_vec in proptest::collection::vec(-1.0_f32..1.0_f32, 16),
            v_init in prop_oneof![-3.0_f32..-0.1_f32, 0.1_f32..3.0_f32],
            alpha_rel in 0.1_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.05_f32..0.95_f32,
            kappa in 0.0_f32..0.95_f32,
            adj_out in -1.0_f32..1.0_f32,
        ) {
            let mut u = [0.0_f32; 16];
            let mut adj_v = [0.0_f32; 16];
            u.copy_from_slice(&u_vec);
            adj_v.copy_from_slice(&adj_vec);
            let params = ParaLruParams::clamped(alpha_rel, alpha_atk, kappa);

            let mut v_traj = [0.0_f32; 16];
            let mut v_state = GpuVec4::splat(v_init);
            let mut min_abs_z = f32::MAX;
            for (idx, &ui) in u.iter().enumerate() {
                let v_prev = v_state.to_array()[0];
                min_abs_z = min_abs_z.min((ui - params.kappa * params.alpha_rel * v_prev).abs());
                v_traj[idx] = step_paralru_vec4(
                    GpuVec4::splat(ui),
                    &mut v_state,
                    GpuVec4::splat(params.alpha_rel),
                    GpuVec4::splat(params.alpha_atk),
                    GpuVec4::splat(params.kappa),
                )
                .to_array()[0];
            }

            let grads =
                run_paralru_adjoint_scan_tile::<16>(&u, &v_traj, v_init, params, &adj_v, adj_out);
            let mut u_first = [0.0_f32; 8];
            let mut u_second = [0.0_f32; 8];
            let mut v_first = [0.0_f32; 8];
            let mut v_second = [0.0_f32; 8];
            let mut a_first = [0.0_f32; 8];
            let mut a_second = [0.0_f32; 8];
            u_first.copy_from_slice(&u[..8]);
            u_second.copy_from_slice(&u[8..]);
            v_first.copy_from_slice(&v_traj[..8]);
            v_second.copy_from_slice(&v_traj[8..]);
            a_first.copy_from_slice(&adj_v[..8]);
            a_second.copy_from_slice(&adj_v[8..]);
            let g1 = run_paralru_adjoint_scan_tile::<8>(
                &u_second,
                &v_second,
                v_first[7],
                params,
                &a_second,
                adj_out,
            );
            let g0 = run_paralru_adjoint_scan_tile::<8>(
                &u_first,
                &v_first,
                v_init,
                params,
                &a_first,
                g1.adj_v_init,
            );
            prop_assert!((g0.adj_v_init - grads.adj_v_init).abs() < 1.0e-5);
            prop_assert!(
                ((g0.adj_alpha_rel + g1.adj_alpha_rel) - grads.adj_alpha_rel).abs() < 1.0e-4
            );
            prop_assert!(
                ((g0.adj_alpha_atk + g1.adj_alpha_atk) - grads.adj_alpha_atk).abs() < 1.0e-4
            );
            prop_assert!(((g0.adj_kappa + g1.adj_kappa) - grads.adj_kappa).abs() < 1.0e-4);

            let u64: Vec<f64> = u.iter().map(|&v| v as f64).collect();
            let mut adj64: Vec<f64> = adj_v.iter().map(|&v| v as f64).collect();
            adj64[15] += adj_out as f64;
            let (v0_64, ar_64, aa_64, kp_64) = (
                v_init as f64,
                params.alpha_rel as f64,
                params.alpha_atk as f64,
                params.kappa as f64,
            );
            let eps = 1.0e-6_f64;

            let fd_v0 = (paralru_loss_f64(&u64, v0_64 + eps, ar_64, aa_64, kp_64, &adj64, None)
                - paralru_loss_f64(&u64, v0_64 - eps, ar_64, aa_64, kp_64, &adj64, None))
                / (2.0 * eps);
            let fd_ar = (paralru_loss_f64(&u64, v0_64, ar_64 + eps, aa_64, kp_64, &adj64, None)
                - paralru_loss_f64(&u64, v0_64, ar_64 - eps, aa_64, kp_64, &adj64, None))
                / (2.0 * eps);
            let fd_aa = (paralru_loss_f64(&u64, v0_64, ar_64, aa_64 + eps, kp_64, &adj64, None)
                - paralru_loss_f64(&u64, v0_64, ar_64, aa_64 - eps, kp_64, &adj64, None))
                / (2.0 * eps);
            let fd_kp = (paralru_loss_f64(&u64, v0_64, ar_64, aa_64, kp_64 + eps, &adj64, None)
                - paralru_loss_f64(&u64, v0_64, ar_64, aa_64, kp_64 - eps, &adj64, None))
                / (2.0 * eps);

            let lam_abs_sum: f64 = grads.lambda.iter().map(|&l| (l as f64).abs()).sum();
            let v_abs_max: f64 = v_traj
                .iter()
                .fold((v_init as f64).abs(), |acc, &x| acc.max((x as f64).abs()));
            let check_fd = |got: f32, fd: f64, scale: f64| -> Result<(), TestCaseError> {
                let err = (got as f64 - fd).abs();
                prop_assert!(err < 1.0e-3, "got={got}, fd={fd}, err={err}");
                let cond = scale / fd.abs().max(1.0e-12);
                if min_abs_z >= 0.02 && fd.abs() >= 0.05 && cond <= 20.0 {
                    prop_assert!(
                        err / fd.abs() < 1.0e-4,
                        "rel_err={}, got={got}, fd={fd}",
                        err / fd.abs()
                    );
                }
                Ok(())
            };

            check_fd(grads.adj_v_init, fd_v0, (grads.lambda[0] as f64).abs())?;
            check_fd(grads.adj_alpha_rel, fd_ar, lam_abs_sum * v_abs_max)?;
            check_fd(grads.adj_alpha_atk, fd_aa, lam_abs_sum * 4.0)?;
            check_fd(grads.adj_kappa, fd_kp, lam_abs_sum * v_abs_max)?;

            for t in 0..16 {
                let mut u_p = u64.clone();
                let mut u_m = u64.clone();
                u_p[t] += eps;
                u_m[t] -= eps;
                let fd_u = (paralru_loss_f64(&u_p, v0_64, ar_64, aa_64, kp_64, &adj64, None)
                    - paralru_loss_f64(&u_m, v0_64, ar_64, aa_64, kp_64, &adj64, None))
                    / (2.0 * eps);
                check_fd(grads.adj_u[t], fd_u, (grads.lambda[t] as f64).abs())?;

                let fd_lam = (paralru_loss_f64(
                    &u64, v0_64, ar_64, aa_64, kp_64, &adj64, Some((t, eps)),
                ) - paralru_loss_f64(
                    &u64, v0_64, ar_64, aa_64, kp_64, &adj64, Some((t, -eps)),
                )) / (2.0 * eps);
                check_fd(grads.lambda[t], fd_lam, lam_abs_sum)?;
            }
        }
    }
}
