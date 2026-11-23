#![forbid(unsafe_code)]
//! `SONA v3.0` Capture Solver Orchestration & Canonical `dnam-train` / Bézout Re-exports.

pub mod progress;
pub mod stage_a_probe;
pub mod stage_b_backbone;
pub mod stage_c_knobs;
pub mod stage_d_audit;
pub mod stages;

pub use crate::stimulus::pipeline::{
    build_v3_causal_probe_excitation, estimate_v3_causal_deadtime_and_polarity,
};
pub use dnam_train::{
    CpuRunner, Pass5EmitConfig, SonaModelF64, SonaSolveResult, SonaSolverConfig, SonaTrainDataset,
    TrainRunner, Wide192Accum, accumulate_uplo_u_wide192, contract_256d_gram_tiled_wide192,
    decode_dnam_bin_to_model, encode_model_to_dnam_bin, solve_sona_model, solve_uplo_u_cholesky,
    solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len,
};
pub use progress::{
    AtomicTrainerProgress, PROGRESS_MAGIC_DNPR, PROGRESS_WIRE_BYTES, PROGRESS_WIRE_WORDS,
    TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus,
};
pub use stage_a_probe::{
    StageAConfig, StageAReport, has_populated_v3_backbone, pass1_hint_from_manifest,
    run_stage_a_physics_probe, run_stage_a_physics_probe_with_result,
};
pub use stage_b_backbone::{
    StageBBackboneSpec, StageBOutputs, StageBSolveConfig, WeightedStemRef,
    compute_v3_stat_chunk_bytes, serialize_v3_stat_chunk_ultra, solve_stage_b_backbone,
    solve_stage_b_backbone_with_pass1, solve_stage_b_backbone_with_progress,
};
pub use stage_c_knobs::{
    StageCKnobsOutput, decode_coreset_lr, solve_stage_c_knobs, star_center_control_values,
};
pub use stage_d_audit::{
    BuyerWatermarkConfig, StageDAuditOptions, StageDSolveOutput, compute_integrated_lufs,
    derive_buyer_gauge_params, derive_buyer_phase_seed, derive_buyer_tardos_codebook,
    export_runtime_dnam_watermarked, extract_arena_pole_angles_16, extract_arena_pole_angles_32,
    extract_arena_pole_angles_34, finalize_stage_d_and_audit,
    identify_pole_angles_blackbox_from_audio, render_blackbox_probe_response,
};
pub use stages::write_inline_cp1_if_fits;

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
    /// Returns white-box oracle options that preserve an existing populated `30,720 B` physical
    /// `Arena0` backbone while solving linear readouts, `CP1`, `STAT`, and output trims.
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
