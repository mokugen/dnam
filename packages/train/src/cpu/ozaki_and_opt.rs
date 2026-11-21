//! Deterministic CPU reference runners for `ozaki_gram_rhs_i32` (`0-ULP` integer-sliced
//! accumulation + `Wide192Accum` `f64` host reconstruction) and `inner_opt_step` (`m <= 8`
//! bounded L-BFGS + Armijo backtracking) (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{accum_ozaki_chunk_i32, pow2_ceil_fp32, reconstruct_ozaki_dot_f64};

use crate::error::TrainError;
use crate::gpu::abi::{GPU_OZAKI_MAX_CHUNK_LEN, LBFGS_MAX_HISTORY_M, OZAKI_WORDS_PER_PAIR};
use crate::runner::types::{InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput};
use crate::wide_gram::{Wide192Accum, uplo_u_index, uplo_u_packed_len};

#[inline(always)]
fn resolve_col_inv_scale(entry: [f32; 2]) -> f32 {
    if entry[1] > 0.0 && entry[1].is_finite() {
        entry[1]
    } else {
        pow2_ceil_fp32(entry[0]).1
    }
}

#[inline(always)]
fn resolve_col_scale(entry: [f32; 2]) -> f32 {
    if entry[1] > 0.0 && entry[1].is_finite() {
        entry[0]
    } else {
        pow2_ceil_fp32(entry[0]).0
    }
}

/// Deterministic CPU reference implementation of `ozaki_gram_rhs_i32`.
///
/// Enforces the `target_channels == 0` guard: any trailing buffer pairs beyond
/// `feature_dim * (feature_dim + 1) / 2` remain untouched when `target_channels == 0`.
pub fn run_ozaki_gram_rhs_i32_cpu(
    input: &OzakiGramInput<'_>,
) -> Result<OzakiGramOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let dim = u.feature_dim as usize;
    let tc = u.target_channels as usize;
    let chunk_len = (u.chunk_len as usize).clamp(1, GPU_OZAKI_MAX_CHUNK_LEN);
    let num_chunks = (u.num_chunks as usize).max(num_frames.div_ceil(chunk_len));

    if num_frames == 0 || dim == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32",
            expected: 1,
            actual: 0,
        });
    }
    if input.features.len() < dim * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(features)",
            expected: dim * num_frames,
            actual: input.features.len(),
        });
    }
    if tc > 0 && input.targets.len() < tc * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(targets)",
            expected: tc * num_frames,
            actual: input.targets.len(),
        });
    }
    if input.col_scales.len() < dim + tc {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(col_scales)",
            expected: dim + tc,
            actual: input.col_scales.len(),
        });
    }

    let gram_pairs = uplo_u_packed_len(dim);
    let active_pairs = gram_pairs + dim * tc;
    let min_words = active_pairs * num_chunks * OZAKI_WORDS_PER_PAIR;
    let mut ozaki_buckets = match input.seed_buckets {
        Some(seed) => {
            if seed.len() < min_words {
                return Err(TrainError::InvalidDimensions {
                    context: "ozaki_gram_rhs_i32(seed_buckets)",
                    expected: min_words,
                    actual: seed.len(),
                });
            }
            seed.to_vec()
        }
        None => vec![0_i32; min_words],
    };
    let num_pairs = ozaki_buckets.len() / (num_chunks * OZAKI_WORDS_PER_PAIR);

    let mut pair_idx = 0_usize;
    for i in 0..dim {
        let isx = resolve_col_inv_scale(input.col_scales[i]);
        for j in i..dim {
            let isy = resolve_col_inv_scale(input.col_scales[j]);
            for ch in 0..num_chunks {
                let s = ch * chunk_len;
                let e = (s + chunk_len).min(num_frames);
                let words = accum_ozaki_chunk_i32(
                    &input.features[i * num_frames + s..i * num_frames + e],
                    &input.features[j * num_frames + s..j * num_frames + e],
                    isx,
                    isy,
                );
                let dst = (pair_idx * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                ozaki_buckets[dst..dst + OZAKI_WORDS_PER_PAIR].copy_from_slice(&words);
            }
            pair_idx += 1;
        }
    }
    for i in 0..dim {
        let isx = resolve_col_inv_scale(input.col_scales[i]);
        for t_ch in 0..tc {
            let isy = resolve_col_inv_scale(input.col_scales[dim + t_ch]);
            for ch in 0..num_chunks {
                let s = ch * chunk_len;
                let e = (s + chunk_len).min(num_frames);
                let words = accum_ozaki_chunk_i32(
                    &input.features[i * num_frames + s..i * num_frames + e],
                    &input.targets[t_ch * num_frames + s..t_ch * num_frames + e],
                    isx,
                    isy,
                );
                let dst = (pair_idx * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                ozaki_buckets[dst..dst + OZAKI_WORDS_PER_PAIR].copy_from_slice(&words);
            }
            pair_idx += 1;
        }
    }

    Ok(OzakiGramOutput {
        ozaki_buckets,
        num_chunks,
        num_pairs,
    })
}

/// Reconstructs column-major packed `uplo='U'` `f64` Gram matrix and RHS vectors from `i32`
/// Ozaki chunk buckets using [`Wide192Accum`].
pub fn reconstruct_ozaki_gram_rhs_f64(
    output: &OzakiGramOutput,
    feature_dim: usize,
    target_channels: usize,
    col_scales: &[[f32; 2]],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
    if col_scales.len() < feature_dim + target_channels {
        return Err(TrainError::InvalidDimensions {
            context: "reconstruct_ozaki_gram_rhs_f64(col_scales)",
            expected: feature_dim + target_channels,
            actual: col_scales.len(),
        });
    }
    let gram_len = uplo_u_packed_len(feature_dim);
    let needed =
        (gram_len + feature_dim * target_channels) * output.num_chunks * OZAKI_WORDS_PER_PAIR;
    if output.ozaki_buckets.len() < needed {
        return Err(TrainError::InvalidDimensions {
            context: "reconstruct_ozaki_gram_rhs_f64(ozaki_buckets)",
            expected: needed,
            actual: output.ozaki_buckets.len(),
        });
    }

    let mut gram_u = vec![0.0_f64; gram_len];
    let mut rhs = vec![vec![0.0_f64; target_channels]; feature_dim];
    let mut pair_idx = 0_usize;

    for i in 0..feature_dim {
        let sx = resolve_col_scale(col_scales[i]);
        for j in i..feature_dim {
            let sy = resolve_col_scale(col_scales[j]);
            let mut acc = Wide192Accum::zero();
            for ch in 0..output.num_chunks {
                let b = (pair_idx * output.num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                let mut words = [0_i32; OZAKI_WORDS_PER_PAIR];
                words.copy_from_slice(&output.ozaki_buckets[b..b + OZAKI_WORDS_PER_PAIR]);
                acc.add_f64(reconstruct_ozaki_dot_f64(&words, sx, sy));
            }
            gram_u[uplo_u_index(i, j)] = acc.to_f64();
            pair_idx += 1;
        }
    }
    for i in 0..feature_dim {
        let sx = resolve_col_scale(col_scales[i]);
        for tc in 0..target_channels {
            let sy = resolve_col_scale(col_scales[feature_dim + tc]);
            let mut acc = Wide192Accum::zero();
            for ch in 0..output.num_chunks {
                let b = (pair_idx * output.num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                let mut words = [0_i32; OZAKI_WORDS_PER_PAIR];
                words.copy_from_slice(&output.ozaki_buckets[b..b + OZAKI_WORDS_PER_PAIR]);
                acc.add_f64(reconstruct_ozaki_dot_f64(&words, sx, sy));
            }
            rhs[i][tc] = acc.to_f64();
            pair_idx += 1;
        }
    }
    Ok((gram_u, rhs))
}

#[inline(always)]
fn clamp_box(cand: f32, bounds: [f32; 2], min_eps: f32) -> f32 {
    let lo = bounds[0] + min_eps;
    let hi = bounds[1] - min_eps;
    if lo <= hi {
        cand.clamp(lo, hi)
    } else {
        0.5 * (bounds[0] + bounds[1])
    }
}

/// Deterministic CPU reference implementation of single-workgroup `(1, 1, 1)` `inner_opt_step`
/// (`m <= 8` bounded L-BFGS two-loop recursion and Armijo line search backtracking).
pub fn run_inner_opt_step_cpu(input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
    let u = &input.uniforms;
    let n = u.num_params as usize;
    let m = u.history_m as usize;
    if n == 0 || m > LBFGS_MAX_HISTORY_M {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step",
            expected: LBFGS_MAX_HISTORY_M,
            actual: m,
        });
    }
    let need_sy = (m + 1) * n;
    if input.theta.len() < n
        || input.grad.len() < n
        || input.param_bounds.len() < n
        || input.lbfgs_s_y.len() < need_sy
        || input.search_dir.len() < n
    {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step(buffers)",
            expected: need_sy,
            actual: input.lbfgs_s_y.len(),
        });
    }

    let mut theta = input.theta[..n].to_vec();
    let mut lbfgs_s_y = input.lbfgs_s_y[..need_sy].to_vec();
    let mut search_dir = input.search_dir[..n].to_vec();
    let step_sz = u.step_size;
    let damping = u.damping_lambda;
    let min_eps = u.min_bound_eps;

    if u.line_search_iter > 0 {
        for p in 0..n {
            theta[p] = clamp_box(
                lbfgs_s_y[m * n + p][0] + step_sz * search_dir[p],
                input.param_bounds[p],
                min_eps,
            );
        }
        return Ok(InnerOptOutput {
            theta,
            lbfgs_s_y,
            search_dir,
        });
    }

    if u.step_idx > 0 && m > 0 {
        let (mut ys, mut ss, mut yy) = (0.0_f32, 0.0_f32, 0.0_f32);
        for p in 0..n {
            let s = theta[p] - lbfgs_s_y[m * n + p][0];
            let y = input.grad[p] - lbfgs_s_y[m * n + p][1];
            ys += s * y;
            ss += s * s;
            yy += y * y;
        }
        let slot = ((u.step_idx - 1) as usize) % m;
        let valid = ys.is_finite() && ys > 1.0e-12 * (ss + yy) && ys > 0.0;
        for p in 0..n {
            lbfgs_s_y[slot * n + p] = if valid {
                [
                    theta[p] - lbfgs_s_y[m * n + p][0],
                    input.grad[p] - lbfgs_s_y[m * n + p][1],
                    0.0,
                    0.0,
                ]
            } else {
                [0.0; 4]
            };
        }
    }

    let mut q = input.grad[..n].to_vec();
    let used = if u.step_idx == 0 || m == 0 {
        0
    } else {
        (u.step_idx as usize).min(m)
    };
    let mut alphas = vec![0.0_f32; used];
    let mut rhos = vec![0.0_f32; used];
    let mut found_h0 = false;
    let mut h0_scale = 1.0 / (1.0 + damping);

    for age in 0..used {
        let slot = ((u.step_idx as usize) - 1 - age) % m;
        let (mut sy, mut sq, mut yy) = (0.0_f32, 0.0_f32, 0.0_f32);
        for p in 0..n {
            let s = lbfgs_s_y[slot * n + p][0];
            let y = lbfgs_s_y[slot * n + p][1];
            sy += s * y;
            sq += s * q[p];
            if !found_h0 {
                yy += y * y;
            }
        }
        if sy > 0.0 {
            let r = 1.0 / sy;
            let a = r * sq;
            if !found_h0 {
                h0_scale = sy / (yy + damping.max(1.0e-12));
                found_h0 = true;
            }
            for p in 0..n {
                q[p] -= a * lbfgs_s_y[slot * n + p][1];
            }
            alphas[age] = a;
            rhos[age] = r;
        }
    }
    for v in &mut q {
        *v *= h0_scale;
    }
    for rev in 0..used {
        let age = used - 1 - rev;
        let slot = ((u.step_idx as usize) - 1 - age) % m;
        let mut yq = 0.0_f32;
        for p in 0..n {
            yq += lbfgs_s_y[slot * n + p][1] * q[p];
        }
        let coeff = alphas[age] - rhos[age] * yq;
        for p in 0..n {
            q[p] += lbfgs_s_y[slot * n + p][0] * coeff;
        }
    }

    let gd: f32 = input.grad[..n]
        .iter()
        .zip(&q)
        .map(|(g, qv)| g * (-qv))
        .sum();
    let fallback = !gd.is_finite() || gd >= 0.0 || q.iter().any(|v| !v.is_finite());
    for p in 0..n {
        let dir = if fallback {
            -input.grad[p] / (1.0 + damping)
        } else {
            -q[p]
        };
        lbfgs_s_y[m * n + p] = [
            theta[p],
            input.grad[p],
            if used > 0 { 1.0 } else { 0.0 },
            0.0,
        ];
        search_dir[p] = dir;
        theta[p] = clamp_box(theta[p] + step_sz * dir, input.param_bounds[p], min_eps);
    }

    Ok(InnerOptOutput {
        theta,
        lbfgs_s_y,
        search_dir,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::gpu::abi::{InnerOptUniforms, OzakiGramUniforms};

    #[test]
    fn test_cpu_ozaki_guard_recon_and_lbfgs_step() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        let n = 128;
        let dim = 3;
        let mut feats = vec![0.0_f32; dim * n];
        for c in 0..dim {
            for t in 0..n {
                feats[c * n + t] = ((t as f32) * 0.05 * ((c + 1) as f32)).sin();
            }
        }
        let gram_pairs = uplo_u_packed_len(dim);
        let sentinel = 0x5A5A_5A5A_i32;
        let seed_buckets = vec![sentinel; (gram_pairs + 2) * OZAKI_WORDS_PER_PAIR];
        let scales = vec![[1.0_f32, 1.0_f32]; dim];
        let out = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                num_frames: n as u32,
                feature_dim: dim as u32,
                target_channels: 0,
                chunk_len: 1024,
                num_chunks: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            features: &feats,
            targets: &[],
            col_scales: &scales,
            seed_buckets: Some(&seed_buckets),
        })
        .unwrap();
        assert_eq!(
            &out.ozaki_buckets[gram_pairs * OZAKI_WORDS_PER_PAIR..],
            &vec![sentinel; 2 * OZAKI_WORDS_PER_PAIR]
        );
        let (gram, rhs) = reconstruct_ozaki_gram_rhs_f64(&out, dim, 0, &scales).unwrap();
        assert_eq!(gram.len(), gram_pairs);
        assert!(rhs.iter().all(Vec::is_empty));

        // L-BFGS convergence + backtrack invariance check
        let m = 4;
        let np = 12;
        let mut theta = vec![2.0_f32; np];
        let mut sy = vec![[0.0_f32; 4]; (m + 1) * np];
        let mut dir = vec![0.0_f32; np];
        let bounds = vec![[-1.0_f32, 1.0_f32]; np];
        let target: Vec<f32> = (0..np).map(|i| -0.4 + 0.05 * (i as f32)).collect();
        let eval_fn = |x: &[f32]| -> (f32, Vec<f32>) {
            let g: Vec<f32> = x.iter().zip(&target).map(|(xi, ti)| xi - ti).collect();
            let f = 0.5 * g.iter().map(|v| v * v).sum::<f32>();
            (f, g)
        };
        let (mut f_cur, mut grad) = eval_fn(&theta);
        for step in 0..8_u32 {
            let mut alpha = if step == 1 { 4.0 } else { 1.0 };
            for ls in 0..5_u32 {
                let res = run_inner_opt_step_cpu(&InnerOptInput {
                    uniforms: InnerOptUniforms {
                        num_params: np as u32,
                        history_m: m as u32,
                        step_idx: step,
                        line_search_iter: ls,
                        step_size: alpha,
                        damping_lambda: 1.0e-4,
                        min_bound_eps: 1.0e-4,
                        _pad: 0,
                    },
                    theta: &theta,
                    grad: &grad,
                    param_bounds: &bounds,
                    lbfgs_s_y: &sy,
                    search_dir: &dir,
                })
                .unwrap();
                if ls > 0 {
                    assert_eq!(res.lbfgs_s_y, sy);
                    assert_eq!(res.search_dir, dir);
                }
                let (fn_val, gn) = eval_fn(&res.theta);
                theta = res.theta;
                sy = res.lbfgs_s_y;
                dir = res.search_dir;
                if fn_val <= f_cur {
                    f_cur = fn_val;
                    grad = gn;
                    break;
                }
                alpha *= 0.5;
            }
        }
        assert!(f_cur < 0.01);

        // Cover input dimension & validation error branches + inverted-bounds/NaN fallback.
        let oz_zero = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
            uniforms: OzakiGramUniforms {
                num_frames: 0,
                feature_dim: 2,
                target_channels: 0,
                chunk_len: 1024,
                num_chunks: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            features: &[],
            targets: &[],
            col_scales: &scales,
            seed_buckets: None,
        });
        assert!(matches!(oz_zero, Err(TrainError::InvalidDimensions { .. })));

        for (bad_feats, bad_tc, bad_tgts, bad_scales, bad_seed) in [
            (&[][..], 0_usize, &[][..], &scales[..], None),
            (&feats[..], 1, &[][..], &scales[..], None),
            (&feats[..], 0, &[][..], &[][..], None),
            (&feats[..], 0, &[][..], &scales[..], Some(&[0_i32; 1][..])),
        ] {
            let err = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    num_frames: n as u32,
                    feature_dim: dim as u32,
                    target_channels: bad_tc as u32,
                    chunk_len: 1024,
                    num_chunks: 1,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
                features: bad_feats,
                targets: bad_tgts,
                col_scales: bad_scales,
                seed_buckets: bad_seed,
            });
            assert!(matches!(err, Err(TrainError::InvalidDimensions { .. })));
        }
        assert!(matches!(
            reconstruct_ozaki_gram_rhs_f64(&out, dim, 5, &scales),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let short_out = OzakiGramOutput {
            ozaki_buckets: vec![0; 2],
            num_chunks: 1,
            num_pairs: 1,
        };
        assert!(matches!(
            reconstruct_ozaki_gram_rhs_f64(&short_out, dim, 0, &scales),
            Err(TrainError::InvalidDimensions { .. })
        ));

        // inner_opt_step validation errors, inverted box bounds (`lo > hi`), & NaN gradient fallback.
        let base_opt_u = InnerOptUniforms {
            num_params: 2,
            history_m: 1,
            step_idx: 1,
            line_search_iter: 0,
            step_size: 0.5,
            damping_lambda: 1.0e-4,
            min_bound_eps: 2.0,
            _pad: 0,
        };
        assert!(matches!(
            run_inner_opt_step_cpu(&InnerOptInput {
                uniforms: InnerOptUniforms {
                    num_params: 0,
                    ..base_opt_u
                },
                theta: &[0.0; 2],
                grad: &[0.0; 2],
                param_bounds: &[[-1.0, 1.0]; 2],
                lbfgs_s_y: &[[0.0; 4]; 4],
                search_dir: &[0.0; 2],
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            run_inner_opt_step_cpu(&InnerOptInput {
                uniforms: base_opt_u,
                theta: &[0.0; 1],
                grad: &[0.0; 2],
                param_bounds: &[[-1.0, 1.0]; 2],
                lbfgs_s_y: &[[0.0; 4]; 4],
                search_dir: &[0.0; 2],
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let fallback = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: base_opt_u,
            theta: &[0.25, 0.25],
            grad: &[f32::NAN, 1.0],
            param_bounds: &[[-1.0, 1.0]; 2],
            lbfgs_s_y: &[[0.0; 4]; 4],
            search_dir: &[0.0; 2],
        })
        .unwrap();
        assert_eq!(fallback.theta[1], 0.0);
    }

    proptest! {
        #[test]
        fn prop_ozaki_f64_reconstruction_accuracy(
            vals_x in prop::collection::vec(-2.0_f32..2.0_f32, 16..64),
            vals_y in prop::collection::vec(-2.0_f32..2.0_f32, 16..64),
        ) {
            let n = vals_x.len().min(vals_y.len());
            let mut feats = Vec::with_capacity(2 * n);
            feats.extend_from_slice(&vals_x[..n]);
            feats.extend_from_slice(&vals_y[..n]);
            let max_x = vals_x[..n].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let max_y = vals_y[..n].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let scales = [[max_x, 0.0], [max_y, 0.0]];
            let out = run_ozaki_gram_rhs_i32_cpu(&OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    num_frames: n as u32,
                    feature_dim: 2,
                    target_channels: 0,
                    chunk_len: 1024,
                    num_chunks: 1,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
                features: &feats,
                targets: &[],
                col_scales: &scales,
                seed_buckets: None,
            }).unwrap();
            let (gram, _) = reconstruct_ozaki_gram_rhs_f64(&out, 2, 0, &scales).unwrap();
            let (exact, l1) = vals_x[..n].iter().zip(&vals_y[..n]).fold((0.0_f64, 0.0_f64), |(s, a), (&x, &y)| {
                let p = f64::from(x) * f64::from(y);
                (s + p, a + p.abs())
            });
            prop_assert!((gram[uplo_u_index(0, 1)] - exact).abs() <= 1.0e-5 * l1.max(1.0));
        }
    }
}
