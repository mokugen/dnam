//! Deterministic CPU reference runners for 2D split-64 `ozaki_gram_rhs_i32` (`ozaki`) and
//! multi-candidate `(1, num_candidates, 1)` `inner_opt_step` (`inner_opt`)
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

#[path = "ozaki_and_opt/inner_opt.rs"]
mod inner_opt;
#[path = "ozaki_and_opt/ozaki.rs"]
mod ozaki;

pub use inner_opt::run_inner_opt_step_cpu;
pub use ozaki::{
    reconstruct_ozaki_gram_rhs_f64, reconstruct_ozaki_split64_f64, run_ozaki_gram_rhs_i32_cpu,
    run_ozaki_gram_rhs_split64_cpu,
};

#[cfg(test)]
mod tests {
    use dnam_kernels::{pow2_ceil_fp32, reconstruct_ozaki_dot_f64};
    use proptest::prelude::*;

    use super::*;
    use crate::error::TrainError;
    use crate::gpu::abi::{
        COORD_TRANSFORM_ASINH_BOX_NONNEG, COORD_TRANSFORM_ASINH_BOX_SYM, COORD_TRANSFORM_BOX,
        COORD_TRANSFORM_COMPLEMENT_LOG, COORD_TRANSFORM_FREE, COORD_TRANSFORM_LOG_BOX,
        GPU_OZAKI_MAX_CHUNK_LEN, INNER_OPT_MODE_LINE_SEARCH_TRIAL, INNER_OPT_MODE_LSSBROYDEN,
        INNER_OPT_MODE_RESET_STATE, INNER_OPT_MODE_STE_FP16, InnerOptUniforms,
        LSSBROYDEN_MAX_HISTORY_M, OZAKI_FLAG_ACCUMULATE_SPLIT64, OZAKI_PAIR_TILE_ROWS_DEFAULT,
        OZAKI_SPLIT64_WORDS_PER_PAIR, OZAKI_WORDS_PER_PAIR, OzakiGramUniforms, STE_FP16_META_BIT,
    };
    use crate::runner::types::{InnerOptInput, OzakiGramInput, OzakiSplit64Output};
    use crate::wide_gram::uplo_u_index;

    #[test]
    fn test_cpu_ozaki_gram_split64_and_target_channels_zero_guard() {
        let num_frames = 1280_usize;
        let dim = 4_usize;
        let tc = 2_usize;
        let mut features = vec![0.0_f32; dim * num_frames];
        let mut targets = vec![0.0_f32; tc * num_frames];
        let mut col_scales = vec![[0.0_f32; 2]; dim + tc];
        for c in 0..dim {
            let mut m = 0.0_f32;
            for t in 0..num_frames {
                let v = ((t as f32) * 0.03 * ((c + 1) as f32)).sin();
                features[c * num_frames + t] = v;
                m = m.max(v.abs());
            }
            let (s, inv) = pow2_ceil_fp32(m);
            col_scales[c] = [s, inv];
        }
        for ch in 0..tc {
            let mut m = 0.0_f32;
            for t in 0..num_frames {
                let v = ((t as f32) * 0.05 + (ch as f32)).cos();
                targets[ch * num_frames + t] = v;
                m = m.max(v.abs());
            }
            let (s, inv) = pow2_ceil_fp32(m);
            col_scales[dim + ch] = [s, inv];
        }

        let u0 = OzakiGramUniforms {
            num_frames: num_frames as u32,
            feature_dim: dim as u32,
            target_channels: tc as u32,
            chunk_len: GPU_OZAKI_MAX_CHUNK_LEN as u32,
            num_chunks: 2,
            pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
            reduce_pass: 0,
            _pad: 0,
        };
        let p0 = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
            uniforms: u0,
            features: &features,
            targets: &targets,
            col_scales: &col_scales,
            seed_buckets: None,
        })
        .unwrap();
        let p1 = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                reduce_pass: 1,
                ..u0
            },
            features: &features,
            targets: &targets,
            col_scales: &col_scales,
            seed_buckets: Some(&p0.ozaki_buckets),
        })
        .unwrap();
        let (gram, rhs) = reconstruct_ozaki_gram_rhs_f64(&p1, dim, tc, &col_scales).unwrap();
        let split64 =
            run_ozaki_gram_rhs_split64_cpu(&features, &targets, num_frames, dim, tc, &col_scales)
                .unwrap();
        let p1_split64: Vec<u32> = p1.ozaki_buckets[..split64.split64_words.len()]
            .iter()
            .map(|&v| v as u32)
            .collect();
        assert_eq!(p1_split64, split64.split64_words);
        let (gram_s64, rhs_s64) =
            reconstruct_ozaki_split64_f64(&split64, dim, tc, &col_scales).unwrap();
        for (&a, &b) in gram_s64.iter().zip(&gram) {
            assert!((a - b).abs() <= 1.0e-12 * b.abs().max(1.0));
        }
        for (row_a, row_b) in rhs_s64.iter().zip(&rhs) {
            for (&a, &b) in row_a.iter().zip(row_b) {
                assert!((a - b).abs() <= 1.0e-12 * b.abs().max(1.0));
            }
        }

        // Accumulate flag doubles split64 words cleanly.
        let p1_acc = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                reduce_pass: 1,
                _pad: OZAKI_FLAG_ACCUMULATE_SPLIT64,
                ..u0
            },
            features: &features,
            targets: &targets,
            col_scales: &col_scales,
            seed_buckets: Some(&p1.ozaki_buckets),
        })
        .unwrap();
        let (g_acc, _) = reconstruct_ozaki_split64_f64(
            &OzakiSplit64Output {
                split64_words: p1_acc.ozaki_buckets[..split64.split64_words.len()]
                    .iter()
                    .map(|&v| v as u32)
                    .collect(),
                num_pairs: split64.num_pairs,
            },
            dim,
            tc,
            &col_scales,
        )
        .unwrap();
        assert!((g_acc[0] - 2.0 * gram[0]).abs() <= 1.0e-12);

        // target_channels == 0 sentinel guard.
        let gram_pairs = dim * (dim + 1) / 2;
        let sentinel = 0x1357_2468_i32;
        let seed_len =
            gram_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR + (gram_pairs + 2) * 2 * OZAKI_WORDS_PER_PAIR;
        let seed = vec![sentinel; seed_len];
        let zt = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                target_channels: 0,
                ..u0
            },
            features: &features,
            targets: &[],
            col_scales: &col_scales[..dim],
            seed_buckets: Some(&seed),
        })
        .unwrap();
        let tail =
            gram_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR + gram_pairs * 2 * OZAKI_WORDS_PER_PAIR;
        assert!(zt.ozaki_buckets[tail..].iter().all(|&v| v == sentinel));
        let _ = reconstruct_ozaki_dot_f64(&[0; 6], 1.0, 1.0);
    }

    #[test]
    fn test_cpu_inner_opt_lssbroyden_m32_and_adamw_ste_all_transforms() {
        let num_cands = 2_usize;
        let n = 6_usize;
        let m = LSSBROYDEN_MAX_HISTORY_M;
        let bounds = [
            [
                -2.0_f32,
                2.0,
                1.0,
                f32::from_bits(COORD_TRANSFORM_FREE | STE_FP16_META_BIT),
            ],
            [
                0.1,
                2.0,
                1.0,
                f32::from_bits(COORD_TRANSFORM_BOX | STE_FP16_META_BIT),
            ],
            [
                20.0,
                2000.0,
                1.0,
                f32::from_bits(COORD_TRANSFORM_LOG_BOX | STE_FP16_META_BIT),
            ],
            [
                1e-4,
                0.2,
                1.0,
                f32::from_bits(COORD_TRANSFORM_COMPLEMENT_LOG | STE_FP16_META_BIT),
            ],
            [
                -3.0,
                3.0,
                0.5,
                f32::from_bits(COORD_TRANSFORM_ASINH_BOX_SYM | STE_FP16_META_BIT),
            ],
            [
                0.0,
                5.0,
                0.5,
                f32::from_bits(COORD_TRANSFORM_ASINH_BOX_NONNEG | STE_FP16_META_BIT),
            ],
        ];
        let mut theta = vec![0.4_f32; num_cands * n];
        let grad = vec![0.25_f32; num_cands * n];
        let mut state = vec![[0.0_f32; 4]; num_cands * n];
        let dir = vec![0.0_f32; num_cands * n];

        let adam_res = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: n as u32,
                history_m: 0,
                step_idx: 0,
                mode_flags: INNER_OPT_MODE_STE_FP16,
                step_size: 0.05,
                damping_lambda: 0.0,
                min_bound_eps: 1.0e-4,
                grad_clip_norm: 0.5,
            },
            num_candidates: num_cands,
            theta: &theta,
            grad: &grad,
            param_bounds: &bounds,
            lbfgs_s_y: &state,
            search_dir: &dir,
            diag_precond: None,
        })
        .unwrap();
        assert_eq!(adam_res.theta.len(), num_cands * n);

        // L-SSBroyden multi-candidate step + line search trial + reset.
        state = vec![[0.0_f32; 4]; num_cands * (m + 1) * n];
        let free_bounds = [[-2.0_f32, 2.0, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)]; 6];
        let u0 = InnerOptUniforms {
            num_params: n as u32,
            history_m: m as u32,
            step_idx: 0,
            mode_flags: INNER_OPT_MODE_LSSBROYDEN,
            step_size: 1.0,
            damping_lambda: 1.0e-4,
            min_bound_eps: 1.0e-4,
            grad_clip_norm: 0.0,
        };
        let s0 = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: u0,
            num_candidates: num_cands,
            theta: &theta,
            grad: &grad,
            param_bounds: &free_bounds,
            lbfgs_s_y: &state,
            search_dir: &dir,
            diag_precond: None,
        })
        .unwrap();
        let trial = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: InnerOptUniforms {
                mode_flags: INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_LINE_SEARCH_TRIAL,
                step_size: 0.5,
                ..u0
            },
            num_candidates: num_cands,
            theta: &s0.theta,
            grad: &grad,
            param_bounds: &free_bounds,
            lbfgs_s_y: &s0.lbfgs_s_y,
            search_dir: &s0.search_dir,
            diag_precond: None,
        })
        .unwrap();
        assert_eq!(trial.lbfgs_s_y, s0.lbfgs_s_y);

        let reset = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: InnerOptUniforms {
                mode_flags: INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_RESET_STATE,
                ..u0
            },
            num_candidates: num_cands,
            theta: &trial.theta,
            grad: &grad,
            param_bounds: &free_bounds,
            lbfgs_s_y: &trial.lbfgs_s_y,
            search_dir: &trial.search_dir,
            diag_precond: None,
        })
        .unwrap();
        assert_eq!(reset.lbfgs_s_y[0], [0.0; 4]);

        theta.truncate(1);
        assert!(matches!(
            run_inner_opt_step_cpu(&InnerOptInput {
                uniforms: u0,
                num_candidates: num_cands,
                theta: &theta,
                grad: &grad,
                param_bounds: &free_bounds,
                lbfgs_s_y: &state,
                search_dir: &dir,
                diag_precond: None,
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn prop_ozaki_split64_and_i32_match_exact_f64_dot_products(
            num_frames in 1_usize..=48,
            feature_dim in 1_usize..=4,
            target_channels in 0_usize..=2,
            col_scale_exp in -4_i32..=3,
            raw_vals in proptest::collection::vec(-8.0_f32..8.0_f32, 320),
        ) {
            let mut features = vec![0.0_f32; feature_dim * num_frames];
            let mut targets = vec![0.0_f32; target_channels * num_frames];
            let mut raw_idx = 0_usize;
            for col in 0..feature_dim {
                let col_mult = 2.0_f32.powi(col_scale_exp + (col as i32) - 1);
                for frame in 0..num_frames {
                    features[col * num_frames + frame] = raw_vals[raw_idx % raw_vals.len()] * col_mult;
                    raw_idx += 1;
                }
            }
            for tc in 0..target_channels {
                let tc_mult = 2.0_f32.powi(col_scale_exp - (tc as i32));
                for frame in 0..num_frames {
                    targets[tc * num_frames + frame] = raw_vals[raw_idx % raw_vals.len()] * tc_mult;
                    raw_idx += 1;
                }
            }

            let mut col_scales = Vec::with_capacity(feature_dim + target_channels);
            for col in 0..feature_dim {
                let slice = &features[col * num_frames..(col + 1) * num_frames];
                let max_abs = slice.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
                let (s, inv) = pow2_ceil_fp32(max_abs);
                col_scales.push([s, inv]);
            }
            for tc in 0..target_channels {
                let slice = &targets[tc * num_frames..(tc + 1) * num_frames];
                let max_abs = slice.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
                let (s, inv) = pow2_ceil_fp32(max_abs);
                col_scales.push([s, inv]);
            }

            let split64 = run_ozaki_gram_rhs_split64_cpu(
                &features,
                &targets,
                num_frames,
                feature_dim,
                target_channels,
                &col_scales,
            )
            .unwrap();
            let (gram_s64, rhs_s64) = reconstruct_ozaki_split64_f64(
                &split64,
                feature_dim,
                target_channels,
                &col_scales,
            )
            .unwrap();

            let chunk_len = 16_usize;
            let num_chunks = num_frames.div_ceil(chunk_len);
            let u0 = OzakiGramUniforms {
                num_frames: num_frames as u32,
                feature_dim: feature_dim as u32,
                target_channels: target_channels as u32,
                chunk_len: chunk_len as u32,
                num_chunks: num_chunks as u32,
                pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
                reduce_pass: 0,
                _pad: 0,
            };
            let p0 = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
                uniforms: u0,
                features: &features,
                targets: &targets,
                col_scales: &col_scales,
                seed_buckets: None,
            })
            .unwrap();
            let p1 = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    reduce_pass: 1,
                    ..u0
                },
                features: &features,
                targets: &targets,
                col_scales: &col_scales,
                seed_buckets: Some(&p0.ozaki_buckets),
            })
            .unwrap();
            let p1_split64: Vec<u32> = p1.ozaki_buckets[..split64.split64_words.len()]
                .iter()
                .map(|&w| w as u32)
                .collect();
            prop_assert_eq!(p1_split64, split64.split64_words);
            let (gram_i32, rhs_i32) = reconstruct_ozaki_gram_rhs_f64(
                &p1,
                feature_dim,
                target_channels,
                &col_scales,
            )
            .unwrap();

            for i in 0..feature_dim {
                let col_x = &features[i * num_frames..(i + 1) * num_frames];
                let norm_x = col_x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
                prop_assert!(gram_s64[uplo_u_index(i, i)] >= 0.0);
                for j in i..feature_dim {
                    let col_y = &features[j * num_frames..(j + 1) * num_frames];
                    let norm_y = col_y.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
                    let exact: f64 = col_x
                        .iter()
                        .zip(col_y.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let idx = uplo_u_index(i, j);
                    let tol_exact = 1e-5 * (norm_x * norm_y + 1e-12);
                    prop_assert!((gram_s64[idx] - exact).abs() <= tol_exact);
                    let tol_parity = 1e-12 * (gram_s64[idx].abs() + norm_x * norm_y + 1.0);
                    prop_assert!((gram_s64[idx] - gram_i32[idx]).abs() <= tol_parity);
                }
                for tc in 0..target_channels {
                    let col_t = &targets[tc * num_frames..(tc + 1) * num_frames];
                    let norm_t = col_t.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
                    let exact: f64 = col_x
                        .iter()
                        .zip(col_t.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let tol_exact = 1e-5 * (norm_x * norm_t + 1e-12);
                    prop_assert!((rhs_s64[i][tc] - exact).abs() <= tol_exact);
                    let tol_parity = 1e-12 * (rhs_s64[i][tc].abs() + norm_x * norm_t + 1.0);
                    prop_assert!((rhs_s64[i][tc] - rhs_i32[i][tc]).abs() <= tol_parity);
                }
            }
        }
    }
}
