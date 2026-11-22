//! High-precision `f64` linear algebra, dense `uplo='U'` Cholesky, Schur/EMR/LTAS-KKT
//! VarPro solvers, and bounded two-loop / whitened L-BFGS optimizers.

pub mod cholesky;
pub mod lbfgs;
pub mod spectral_kkt;
pub mod varpro;

pub use cholesky::{
    MIN_CHOLESKY_RIDGE, checked_uplo_u_packed_len, factorize_uplo_u_cholesky,
    solve_factorized_uplo_u, solve_factorized_uplo_u_2ch, solve_uplo_adaptive_ridge,
    solve_uplo_norm_bounded, solve_uplo_u_cholesky, solve_uplo_u_cholesky_2ch, symv_uplo_u,
};
pub use lbfgs::{
    LbfgsConfig, LbfgsResult, ParamBoundSpec, TwoLoopLbfgs, WhitenedLbfgs, step_runner_inner_opt,
};
pub use spectral_kkt::{
    EmrMmConfig, EmrMmResult, EmrSpectralProblem, LtasKktConfig, LtasKktResult,
    solve_emr_mm_cholesky, solve_ltas_newton_kkt,
};
pub use varpro::{
    NestedTierPrefixResult, assemble_ozaki_normal_equations, solve_nested_tier_prefix_schur,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::TrainRunner;

    #[test]
    fn test_linalg_ozaki_schur_and_lbfgs_integration() {
        let runner = TrainRunner::cpu();
        let frames = 64_usize;
        let dim = 4_usize;
        let mut feats = vec![0.0_f32; dim * frames];
        let mut tgts = vec![0.0_f32; 2 * frames];
        for c in 0..dim {
            for t in 0..frames {
                feats[c * frames + t] = ((t as f32 + 1.0) * (0.11 + 0.07 * c as f32)).sin();
            }
        }
        for t in 0..frames {
            tgts[t] = 0.6 * feats[t] - 0.3 * feats[2 * frames + t];
            tgts[frames + t] = -0.4 * feats[frames + t] + 0.5 * feats[3 * frames + t];
        }
        let (gram, rhs) =
            assemble_ozaki_normal_equations(&runner, &feats, &tgts, frames, dim, 2, false).unwrap();
        let rhs_l: Vec<f64> = (0..dim).map(|i| rhs[i][0]).collect();
        let rhs_r: Vec<f64> = (0..dim).map(|i| rhs[i][1]).collect();
        let schur =
            solve_nested_tier_prefix_schur(&gram, &rhs_l, &rhs_r, [1, 2, 3, 4], 1.0e-8, 1.0e-7)
                .unwrap();
        assert!((schur.joint_prefix_weights[3][0][0] - 0.6).abs() < 1.0e-3);
        assert!((schur.joint_prefix_weights[3][3][1] - 0.5).abs() < 1.0e-3);

        let wl = WhitenedLbfgs::from_uplo_u(dim, &gram, &[0.0; 4], 1.0e-8).unwrap();
        let opt_res = wl
            .minimize(LbfgsConfig::default(), |w, mut grad_opt| {
                let gw = symv_uplo_u(dim, &gram, w);
                let mut loss = 0.0_f64;
                for i in 0..dim {
                    loss += 0.5 * w[i] * gw[i] - rhs_l[i] * w[i];
                    if let Some(g) = grad_opt.as_deref_mut() {
                        g[i] = gw[i] - rhs_l[i];
                    }
                }
                Ok(loss)
            })
            .unwrap();
        assert!((opt_res.best_params[0] - 0.6).abs() < 1.0e-3);
    }
}
