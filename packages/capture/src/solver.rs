#![forbid(unsafe_code)]
//! `v3` Analytical Variable-Projection (`VarPro`), `uplo='U'` Cholesky & Noise-Subtracted `B-ESR` Solver.

pub mod backends;
pub mod progress;
pub mod stage_a_probe;
pub mod stage_b_backbone;
pub mod stage_c_knobs;
pub mod stage_d_audit;
pub mod stages;
pub mod v3_varpro;
pub mod wide_gram;

pub use backends::{
    DefaultTrainerBackend, DynamicTrainer, GaborOperatorBackend, GramCholeskyBackend,
    KautzCascadeBackend, ScalarTrainerBackend, TrainerBackend, TrainerTask,
};
pub use progress::{
    AtomicTrainerProgress, PROGRESS_MAGIC_DNPR, PROGRESS_WIRE_BYTES, PROGRESS_WIRE_WORDS,
    TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus,
};
pub use stage_a_probe::{
    StageAConfig, StageAReport, has_populated_v3_backbone, run_stage_a_physics_probe,
};
pub use stage_b_backbone::{
    StageBOutputs, StageBSolveConfig, WeightedStemRef, compute_v3_stat_chunk_bytes,
    patch_v3_readouts_weighted_cholesky, serialize_v3_stat_chunk_ultra, solve_stage_b_backbone,
    solve_stage_b_backbone_with_progress,
};
pub use stage_c_knobs::{
    StageCKnobsOutput, decode_coreset_lr, solve_stage_c_knobs, star_center_control_values,
};
pub use stage_d_audit::{
    BuyerWatermarkConfig, StageDAuditOptions, StageDSolveOutput, compute_integrated_lufs,
    derive_buyer_gauge_params, derive_buyer_phase_seed, derive_buyer_tardos_codebook,
    export_runtime_dnam_watermarked, extract_arena_pole_angles_16, extract_arena_pole_angles_34,
    finalize_stage_d_and_audit, identify_pole_angles_blackbox_from_audio,
    render_blackbox_probe_response,
};
pub use stages::write_inline_cp1_if_fits;
pub use v3_varpro::{
    V3TrainDataset, V3TrainStem, build_v3_causal_probe_excitation,
    estimate_v3_causal_deadtime_and_polarity, fit_v3_core_four_tiers,
    fit_v3_core_four_tiers_with_backend, fit_v3_matryoshka, fit_v3_matryoshka_fast,
    fit_v3_matryoshka_fast_with_backend, fit_v3_matryoshka_with_backend,
};
pub use wide_gram::{Wide192Accum, accumulate_uplo_u_wide192, contract_256d_gram_tiled_wide192};

/// Quality vs. wall-clock stopping policy for the `v3` Matryoshka solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverQualityMode {
    /// Default: optimize every tier and candidate until relative improvement reaches genuine
    /// diminishing returns (`|Δloss| / |loss| < ε` streak), selecting the highest-fidelity model.
    #[default]
    BestUntilDiminishingReturns,
    /// Stop early once a candidate clears `target_slack_milli_db` (`0` = canonical §4 gates,
    /// `500` = `+0.50 dB` looser preview) or reaches `max_polish_iters`.
    ThresholdStop {
        /// Extra slack in milli-dB (`1_000 == 1.0 dB`).
        target_slack_milli_db: u32,
        /// Per-tier polish iteration ceiling (`8..=64`).
        max_polish_iters: usize,
    },
}

/// Tuning options for the `v3` 4-Stage Analytical Pipeline (`Stage A..D`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverOptions {
    /// Stage A anchor-stem lifting configuration (`preserve_existing_backbone` for white-box oracles).
    pub stage_a: StageAConfig,
    /// Stride when contracting `256 × 256` `Wide192Accum` Gram outer products (`default = 4`).
    pub gram_frame_stride: usize,
    /// Clears `W_ctrl` and `U_int` during Stage C unless preserving a white-box dynamic backbone.
    pub clear_dynamic_w_ctrl: bool,
    /// Convergence vs. early-threshold stopping policy across Stage B and Stage IV.
    pub quality_mode: SolverQualityMode,
    /// Optional empirical left/right repeatability profiles measured strictly from `(x, y)` probes.
    pub rep_profiles: Option<[crate::metrics::TargetRepeatabilityProfile; 2]>,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            stage_a: StageAConfig::default(),
            gram_frame_stride: 4,
            clear_dynamic_w_ctrl: true,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            rep_profiles: None,
        }
    }
}

impl SolverOptions {
    /// Returns white-box oracle options that preserve an existing populated `28,672 B` physical
    /// backbone (`Stage 0..IV` + `W_ctrl`) while solving linear readouts, `CP1`, `U_int`, `STAT`, and trims.
    #[must_use]
    pub const fn preserve_backbone() -> Self {
        Self {
            stage_a: StageAConfig {
                preserve_existing_backbone: true,
                vp_gn_steps: 1,
            },
            gram_frame_stride: 4,
            clear_dynamic_w_ctrl: false,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            rep_profiles: None,
        }
    }
}

use dnam_engine::Tier;

/// Computes canonical post-warmup offset matching Stage D certification gates (`512` on `>= 2,048` frames).
#[must_use]
pub const fn canonical_besr_warmup(n_frames: usize) -> usize {
    if n_frames > 64 {
        let q = n_frames / 4;
        if q < 512 { q } else { 512 }
    } else {
        0
    }
}

/// Returns the canonical feature-bus dimension $N_T \in \{64, 128, 192, 256\}$ for `tier`.
#[must_use]
pub const fn tier_feature_dimension(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 64,
        Tier::Full => 128,
        Tier::Pro => 192,
        Tier::Ultra => 256,
    }
}

/// Returns the packed upper-triangular length $N(N + 1)/2$ for a symmetric $N \times N$ Gram matrix
/// (`ref[container.proj.stat-uplo-u]`), avoiding `dim * (dim + 1)` intermediate overflow on 32-bit targets.
#[must_use]
pub const fn uplo_u_packed_len(dim: usize) -> usize {
    if dim.is_multiple_of(2) {
        (dim / 2).saturating_mul(dim.saturating_add(1))
    } else {
        dim.saturating_mul(dim.div_ceil(2))
    }
}

#[inline]
const fn checked_uplo_u_packed_len(dim: usize) -> Option<usize> {
    if dim.is_multiple_of(2) {
        match dim.checked_add(1) {
            Some(d1) => (dim / 2).checked_mul(d1),
            None => None,
        }
    } else {
        dim.checked_mul(dim.div_ceil(2))
    }
}

/// Maps upper-triangular coordinates `(i, j)` with `0 <= i <= j < dim` to their column-major
/// `uplo='U'` packed index $i + j(j + 1)/2$.
#[must_use]
pub const fn uplo_u_index(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    i + if j.is_multiple_of(2) {
        (j / 2) * (j + 1)
    } else {
        j * j.div_ceil(2)
    }
}

/// Accumulates a weighted feature outer product $\mathbf{G} \mathrel{+}= w \boldsymbol{\phi} \boldsymbol{\phi}^\top$
/// and cross-correlation $\mathbf{C} \mathrel{+}= w \boldsymbol{\phi} \mathbf{y}^\top$ in `f64`
/// (`ref[train.gram.accuracy-contract]`, `ref[container.proj.stat-uplo-u]`).
pub fn accumulate_uplo_u_gram_and_cross(
    packed_gram: &mut [f64],
    cross_corr: &mut [[f64; 2]],
    features: &[f64],
    target_lr: [f64; 2],
    weight: f64,
) {
    let n = features.len();
    assert_eq!(Some(packed_gram.len()), checked_uplo_u_packed_len(n));
    assert_eq!(cross_corr.len(), n);
    if !weight.is_finite()
        || weight <= 0.0
        || !target_lr[0].is_finite()
        || !target_lr[1].is_finite()
        || !features.iter().all(|v| v.is_finite())
    {
        return;
    }
    for j in 0..n {
        let w_fj = weight * features[j];
        let base = uplo_u_index(0, j);
        for i in 0..=j {
            packed_gram[base + i] += w_fj * features[i];
        }
        cross_corr[j][0] += w_fj * target_lr[0];
        cross_corr[j][1] += w_fj * target_lr[1];
    }
}

pub(crate) fn factorize_uplo_u_cholesky(
    dim: usize,
    packed_gram: &[f64],
    ridge_lambda: f64,
) -> Result<Vec<f64>, &'static str> {
    if Some(packed_gram.len()) != checked_uplo_u_packed_len(dim) {
        return Err("packed_gram length mismatch");
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 || !packed_gram.iter().all(|v| v.is_finite())
    {
        return Err("non-finite solver input");
    }
    let lambda = ridge_lambda.max(1.0e-12);
    let mut r_upper = packed_gram.to_vec();
    for j in 0..dim {
        r_upper[uplo_u_index(j, j)] += lambda;
    }

    // Upper-triangular Cholesky: G + lambda*I = R^T R (4-way unrolled column dot products)
    for j in 0..dim {
        let col_j = uplo_u_index(0, j);
        for i in 0..j {
            let col_i = uplo_u_index(0, i);
            let (ci_chunks, ci_rem) = r_upper[col_i..col_i + i].as_chunks::<4>();
            let (cj_chunks, cj_rem) = r_upper[col_j..col_j + i].as_chunks::<4>();
            let (mut d0, mut d1, mut d2, mut d3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for (a, b) in ci_chunks.iter().zip(cj_chunks) {
                d0 += a[0] * b[0];
                d1 += a[1] * b[1];
                d2 += a[2] * b[2];
                d3 += a[3] * b[3];
            }
            let mut dot = (d0 + d1) + (d2 + d3);
            for (&r_ki, &r_kj) in ci_rem.iter().zip(cj_rem) {
                dot += r_ki * r_kj;
            }
            r_upper[col_j + i] = (r_upper[col_j + i] - dot) / r_upper[col_i + i];
        }
        let mut diag = r_upper[col_j + j];
        for &r_kj in &r_upper[col_j..col_j + j] {
            diag -= r_kj * r_kj;
        }
        if !diag.is_finite() || diag <= 0.0 {
            return Err("matrix is not positive definite");
        }
        r_upper[col_j + j] = diag.sqrt();
    }
    Ok(r_upper)
}

pub(crate) fn solve_factorized_uplo_u_2ch(
    n: usize,
    r_upper: &[f64],
    cross_corr: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, &'static str> {
    if cross_corr.len() != n
        || !cross_corr
            .iter()
            .all(|c| c[0].is_finite() && c[1].is_finite())
    {
        return Err("non-finite solver input");
    }

    // Forward substitution: R^T Y = C
    let mut y = vec![[0.0_f64; 2]; n];
    for i in 0..n {
        let col_i = uplo_u_index(0, i);
        let mut s0 = cross_corr[i][0];
        let mut s1 = cross_corr[i][1];
        for (&r_ki, yk) in r_upper[col_i..col_i + i].iter().zip(&y[..i]) {
            s0 -= r_ki * yk[0];
            s1 -= r_ki * yk[1];
        }
        let inv_diag = 1.0 / r_upper[col_i + i];
        y[i][0] = s0 * inv_diag;
        y[i][1] = s1 * inv_diag;
    }

    // Back substitution: R W = Y (column-sweep updates)
    let mut w = y;
    for k in (0..n).rev() {
        let col_k = uplo_u_index(0, k);
        let inv_diag = 1.0 / r_upper[col_k + k];
        let wk0 = w[k][0] * inv_diag;
        let wk1 = w[k][1] * inv_diag;
        w[k] = [wk0, wk1];
        for (wi, &r_ik) in w[..k].iter_mut().zip(&r_upper[col_k..col_k + k]) {
            wi[0] -= r_ik * wk0;
            wi[1] -= r_ik * wk1;
        }
    }

    if !w.iter().all(|r| r[0].is_finite() && r[1].is_finite()) {
        return Err("non-finite solver output");
    }
    Ok(w)
}

/// Solves the regularized symmetric positive-definite linear system
/// $(\mathbf{G} + \lambda \mathbf{I}) \mathbf{W} = \mathbf{C}$ from a `uplo='U'` packed Gram matrix
/// via Cholesky factorization (`ref[train.pipeline.stage-b-harvest]`, `ref[train.resume.kronecker-reassembly]`).
pub fn solve_uplo_u_cholesky_2ch(
    packed_gram: &[f64],
    cross_corr: &[[f64; 2]],
    ridge_lambda: f64,
) -> Result<Vec<[f64; 2]>, &'static str> {
    let n = cross_corr.len();
    let r_upper = factorize_uplo_u_cholesky(n, packed_gram, ridge_lambda)?;
    solve_factorized_uplo_u_2ch(n, &r_upper, cross_corr)
}

#[inline]
fn eval_bezout_degree7_poly(h: f64) -> f64 {
    let c = crate::metrics::BEZOUT7_LP_COEFFS;
    let u = if h.is_finite() {
        h.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if u <= 0.5 {
        let u2 = u * u;
        let u4 = u2 * u2;
        u4 * (c[0] + c[1] * u + c[2] * u2 + c[3] * u * u2)
    } else {
        let v = 1.0 - u;
        let v2 = v * v;
        let v4 = v2 * v2;
        1.0 - v4 * (c[0] + c[1] * v + c[2] * v2 + c[3] * v * v2)
    }
}

/// Evaluates the canonical degree-7 Binomial/Bézout 3-band partition
/// $B_7(h) = 35 h^4 - 84 h^5 + 70 h^6 - 20 h^7$ ($B_7(h) + B_7(1 - h) \equiv 1$) across ordered
/// normalized responses $h_1 = (1 - s)^2 \le h_2 = 1 - s^2$ and returns `(h_low, h_mid, h_high)`
/// (`ref[qual.besr.bezout-crossover]`).
#[must_use]
pub fn bezout_three_band_weights(s: f64) -> (f64, f64, f64) {
    let s_clamped = if s.is_finite() {
        s.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let one_minus_s = 1.0 - s_clamped;
    let l1 = eval_bezout_degree7_poly(one_minus_s * one_minus_s);
    let l2 = eval_bezout_degree7_poly(1.0 - s_clamped * s_clamped);
    let h_low = l1;
    let h_mid = (l2 - l1).max(0.0);
    let h_high = eval_bezout_degree7_poly(s_clamped * s_clamped);
    (h_low, h_mid, h_high)
}

/// Computes the canonical §1.2 sum-denominator noise-subtracted Error-to-Signal Ratio
/// $\max(0, R - F) / (Y + F)$ with exact zero-denominator semantics (`0.0` when $Y + F = 0, R = 0$;
/// `+inf` when $Y + F = 0, R > 0$ or on invalid inputs; `ref[qual.besr.noise-subtracted]`,
/// `ref[qual.besr.zero-denominator]`).
#[must_use]
pub fn compute_noise_subtracted_besr(
    err_energy: f64,
    target_energy: f64,
    noise_energy: f64,
) -> f64 {
    if !err_energy.is_finite()
        || !target_energy.is_finite()
        || !noise_energy.is_finite()
        || err_energy < 0.0
        || target_energy < 0.0
        || noise_energy < 0.0
    {
        return f64::INFINITY;
    }
    let denom = target_energy + noise_energy;
    if denom.is_finite() {
        if denom > 0.0 {
            (err_energy - noise_energy).max(0.0) / denom
        } else if err_energy == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        let half_denom = 0.5 * target_energy + 0.5 * noise_energy;
        (0.5 * err_energy - 0.5 * noise_energy).max(0.0) / half_denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_cholesky_solver_and_error_paths() {
        assert!(solve_uplo_u_cholesky_2ch(&[1.0], &[[1.0, 2.0], [3.0, 4.0]], 1.0e-6).is_err());
        assert!(solve_uplo_u_cholesky_2ch(&[-10.0], &[[1.0, 2.0]], 1.0e-6).is_err());

        let mut g = vec![0.0_f64; uplo_u_packed_len(2)];
        let mut c = vec![[0.0_f64; 2]; 2];
        accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &[1.0, 0.0], [2.5, -1.5], 0.0);
        accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &[1.0, 0.0], [2.5, -1.5], f64::NAN);
        accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &[f64::NAN, 0.0], [2.5, -1.5], 1.0);
        assert_eq!(g, vec![0.0, 0.0, 0.0]);
        assert!(solve_uplo_u_cholesky_2ch(&[1.0], &[[f64::NAN, 0.0]], 1.0e-6).is_err());
        assert!(solve_uplo_u_cholesky_2ch(&[1.0], &[[1.0, 0.0]], f64::NAN).is_err());
        assert!(solve_uplo_u_cholesky_2ch(&[0.0], &[[1.0e305, 0.0]], 1.0e-12).is_err());
        assert_eq!(bezout_three_band_weights(f64::NAN), (1.0, 0.0, 0.0));

        assert_eq!(compute_noise_subtracted_besr(0.0, 0.0, 0.0), 0.0);
        assert!(compute_noise_subtracted_besr(1.0e-15, 0.0, 0.0).is_infinite());
        assert!(compute_noise_subtracted_besr(-1.0, 1.0, 0.0).is_infinite());
        assert!((compute_noise_subtracted_besr(0.011, 0.999, 0.001) - 0.01).abs() <= 1.0e-15);
    }

    proptest! {
        #[test]
        fn prop_bezout_identity_sums_to_unity(s in 0.0_f64..=1.0_f64) {
            let (low, mid, high) = bezout_three_band_weights(s);
            prop_assert!(low >= 0.0 && mid >= 0.0 && high >= 0.0);
            prop_assert!(((low + mid + high) - 1.0).abs() <= 1.0e-14);
        }

        #[test]
        fn prop_uplo_u_cholesky_recovers_exact_linear_weights(
            w00 in -5.0_f64..5.0_f64,
            w01 in -5.0_f64..5.0_f64,
            w10 in -5.0_f64..5.0_f64,
            w11 in -5.0_f64..5.0_f64,
        ) {
            let mut g = vec![0.0_f64; uplo_u_packed_len(2)];
            let mut c = vec![[0.0_f64; 2]; 2];
            let basis = [[1.0_f64, 0.2_f64], [-0.3_f64, 1.1_f64], [0.7_f64, -0.8_f64]];
            for row in basis {
                let y0 = row[0] * w00 + row[1] * w10;
                let y1 = row[0] * w01 + row[1] * w11;
                accumulate_uplo_u_gram_and_cross(&mut g, &mut c, &row, [y0, y1], 1.0);
            }
            let solved = solve_uplo_u_cholesky_2ch(&g, &c, 1.0e-12).expect("SPD system should solve");
            prop_assert!((solved[0][0] - w00).abs() <= 1.0e-7);
            prop_assert!((solved[0][1] - w01).abs() <= 1.0e-7);
            prop_assert!((solved[1][0] - w10).abs() <= 1.0e-7);
            prop_assert!((solved[1][1] - w11).abs() <= 1.0e-7);
        }
    }
}
