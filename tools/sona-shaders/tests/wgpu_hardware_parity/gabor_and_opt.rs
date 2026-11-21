#![forbid(unsafe_code)]

//! Pure-`f64` Gabor/Bézout Mode A & Mode B finite-difference verification (`n in {16,32,64,128,256}`
//! and `num_stems == 1, num_windows > 1`) plus single-workgroup `(1, 1, 1)` strided global `m <= 8`
//! L-BFGS optimization parity over `num_params > 256` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, dispatch_kernel, read_f32_vec, read_vec2_vec,
    read_vec4_vec,
};
use sona_shaders::abi::{GaborBezoutUniforms, InnerOptUniforms};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

fn run_gabor_mode_b_case(
    ctx: &GpuContext,
    variant: ShaderVariant,
    fft_size: usize,
    num_stems: usize,
    num_windows: usize,
) {
    let label = variant.label();
    let hop = fft_size;
    let num_frames = fft_size * num_windows;
    let mut pred = vec![[0.0_f32; 2]; num_stems * num_frames];
    let mut tgt = vec![[0.0_f32; 2]; num_stems * num_frames];
    for s in 0..num_stems {
        for t in 0..num_frames {
            let ph = (t as f32) + (s as f32) * 3.7;
            pred[s * num_frames + t] = [
                0.55 * (0.21 * ph).sin() + 0.30 * (1.35 * ph).cos() + 0.22 * (2.55 * ph).sin(),
                0.45 * (0.21 * ph).cos() + 0.28 * (1.42 * ph).sin() + 0.18 * (2.63 * ph).cos(),
            ];
            tgt[s * num_frames + t] = [
                0.35 * (0.21 * ph).sin() + 0.24 * (1.35 * ph).cos() + 0.16 * (2.55 * ph).sin(),
                0.25 * (0.23 * ph).cos() + 0.22 * (1.42 * ph).sin() + 0.14 * (2.63 * ph).cos(),
            ];
        }
    }
    let mut taps = vec![[0.0_f32; 4]; fft_size];
    for k in 0..fft_size {
        let win = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * k as f32 / fft_size as f32).cos());
        let lf = if k <= (fft_size / 8).max(1) || k >= fft_size - (fft_size / 8).max(1) {
            1.0
        } else {
            0.0
        };
        let mf = if ((fft_size / 8).max(1) + 1..fft_size / 3).contains(&k)
            || (fft_size - fft_size / 3..=fft_size - (fft_size / 8).max(1) - 1).contains(&k)
        {
            1.0
        } else {
            0.0
        };
        let hf = if (fft_size / 3..=fft_size - fft_size / 3).contains(&k) {
            1.0
        } else {
            0.0
        };
        taps[k] = [win, lf, mf, hf];
    }
    let uniforms = GaborBezoutUniforms {
        num_frames: num_frames as u32,
        fft_size: fft_size as u32,
        hop_size: hop as u32,
        num_stems: num_stems as u32,
        w_lf: 1.0,
        w_mf: 0.7,
        w_hf: 0.4,
        eps_noise: 1.0e-5,
    };
    let d_init = vec![[0.0_f32; 2]; num_stems * num_frames];
    let bl_init = vec![[0.0_f32; 4]; num_stems * num_windows];
    let (out, _) = dispatch_kernel(
        ctx,
        KernelId::GaborBezoutGrad,
        variant,
        (num_windows as u32, num_stems as u32, 1),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&pred)),
            (false, as_byte_slice(&tgt)),
            (false, as_byte_slice(&taps)),
            (true, as_byte_slice(&d_init)),
            (true, as_byte_slice(&bl_init)),
        ],
    );
    let got_grad = read_vec2_vec(&out[0]);
    let got_losses = read_vec4_vec(&out[1]);

    let pred_f64: Vec<[f64; 2]> = pred.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let eval_dft_f64 = |p_buf: &[[f64; 2]]| -> Vec<f64> {
        let mut out_l = vec![0.0_f64; num_stems * num_windows];
        for s in 0..num_stems {
            for w in 0..num_windows {
                let base = s * num_frames + w * hop;
                let mut el = [0.0_f64; 3];
                let mut tl = [0.0_f64; 3];
                for k in 0..fft_size {
                    let (mut erl, mut eil, mut err, mut eir) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                    let (mut trl, mut til, mut trr, mut tir) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                    for n in 0..fft_size {
                        let theta =
                            -2.0 * std::f64::consts::PI * ((k * n) as f64) / (fft_size as f64);
                        let (sin_w, cos_w) = theta.sin_cos();
                        let wv = taps[n][0] as f64;
                        let xl = (p_buf[base + n][0] - (tgt[base + n][0] as f64)) * wv;
                        let xr = (p_buf[base + n][1] - (tgt[base + n][1] as f64)) * wv;
                        let ytl = (tgt[base + n][0] as f64) * wv;
                        let ytr = (tgt[base + n][1] as f64) * wv;
                        erl += xl * cos_w;
                        eil += xl * sin_w;
                        err += xr * cos_w;
                        eir += xr * sin_w;
                        trl += ytl * cos_w;
                        til += ytl * sin_w;
                        trr += ytr * cos_w;
                        tir += ytr * sin_w;
                    }
                    let em2 = erl * erl + eil * eil + err * err + eir * eir;
                    let tm2 = trl * trl + til * til + trr * trr + tir * tir;
                    for b in 0..3 {
                        el[b] += em2 * (taps[k][b + 1] as f64);
                        tl[b] += tm2 * (taps[k][b + 1] as f64);
                    }
                }
                let eps = uniforms.eps_noise as f64;
                out_l[s * num_windows + w] = 1.0 * el[0] / (tl[0] + eps)
                    + 0.7 * el[1] / (tl[1] + eps)
                    + 0.4 * el[2] / (tl[2] + eps);
            }
        }
        out_l
    };

    let ref_losses = eval_dft_f64(&pred_f64);
    for i in 0..ref_losses.len() {
        assert!(
            (got_losses[i][3] as f64 - ref_losses[i]).abs()
                <= 2.0e-4 * ref_losses[i].abs().max(1.0),
            "Mode B fft={fft_size} stems={num_stems} wins={num_windows} ({label}) loss mismatch {i}: got={} exp={}",
            got_losses[i][3],
            ref_losses[i]
        );
    }

    let eps = 1.0e-6_f64;
    let mut sample_indices = Vec::new();
    for s in 0..num_stems {
        for w in 0..num_windows {
            sample_indices.push(s * num_frames + w * hop + (fft_size / 2).min(fft_size - 1));
        }
    }
    for idx in sample_indices {
        let slot = (idx / num_frames) * num_windows + ((idx % num_frames) / hop);
        for ch in 0..2 {
            let mut pp = pred_f64.clone();
            let mut pm = pred_f64.clone();
            pp[idx][ch] += eps;
            pm[idx][ch] -= eps;
            let fd = (eval_dft_f64(&pp)[slot] - eval_dft_f64(&pm)[slot]) / (2.0 * eps);
            let g = got_grad[idx][ch] as f64;
            assert!(
                (g - fd).abs() <= 2.0e-4 * fd.abs().max(1.0),
                "Mode B fft={fft_size} stems={num_stems} ({label}) FD mismatch idx={idx} ch={ch}: got={g} fd={fd}"
            );
        }
    }
}

pub fn verify_gabor_bezout_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();

    // --- Mode A: Causal Bezout FIR across 300 frames (2 tiles: 256 + 44) & 2 stems ---
    {
        let num_frames = 300_usize;
        let num_stems = 2_usize;
        let num_windows = 2_usize;
        let num_taps = 5_usize;
        let mut pred = vec![[0.0_f32; 2]; num_stems * num_frames];
        let mut tgt = vec![[0.0_f32; 2]; num_stems * num_frames];
        for s in 0..num_stems {
            for t in 0..num_frames {
                let x = (t as f32) * 0.13 + (s as f32) * 0.7;
                pred[s * num_frames + t] = [0.5 * x.sin(), 0.4 * (1.7 * x).cos()];
                tgt[s * num_frames + t] = [0.3 * (0.8 * x).sin(), 0.25 * (1.2 * x).cos()];
            }
        }
        let taps = vec![
            [0.25_f32, -0.35, 0.45, 1.0],
            [0.30, 0.15, -0.40, 1.0],
            [0.25, 0.40, 0.20, 1.0],
            [0.15, -0.10, -0.15, 1.0],
            [0.05, -0.10, 0.05, 1.0],
        ];
        let uniforms = GaborBezoutUniforms {
            num_frames: num_frames as u32,
            fft_size: 0,
            hop_size: num_taps as u32,
            num_stems: num_stems as u32,
            w_lf: 1.2,
            w_mf: 0.8,
            w_hf: 0.5,
            eps_noise: 1.0e-5,
        };

        let pred_f64: Vec<[f64; 2]> = pred.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
        let eval_cpu_mode_a_f64 = |p_buf: &[[f64; 2]]| -> Vec<f64> {
            let mut tile_losses = vec![0.0_f64; num_stems * num_windows];
            for s in 0..num_stems {
                for w in 0..num_windows {
                    let base = w * 256;
                    let valid = (num_frames - base).min(256);
                    let mut el = [0.0_f64; 3];
                    let mut tl = [0.0_f64; 3];
                    for n in 0..valid {
                        let mut y = [[0.0_f64; 2]; 3];
                        let mut to = [[0.0_f64; 2]; 3];
                        for k in 0..num_taps.min(n + 1) {
                            let idx = s * num_frames + base + n - k;
                            let err_l = p_buf[idx][0] - (tgt[idx][0] as f64);
                            let err_r = p_buf[idx][1] - (tgt[idx][1] as f64);
                            for b in 0..3 {
                                let h = taps[k][b] as f64;
                                y[b][0] += h * err_l;
                                y[b][1] += h * err_r;
                                to[b][0] += h * (tgt[idx][0] as f64);
                                to[b][1] += h * (tgt[idx][1] as f64);
                            }
                        }
                        for b in 0..3 {
                            el[b] += y[b][0] * y[b][0] + y[b][1] * y[b][1];
                            tl[b] += to[b][0] * to[b][0] + to[b][1] * to[b][1];
                        }
                    }
                    let eps = uniforms.eps_noise as f64;
                    tile_losses[s * num_windows + w] = 1.2 * el[0] / (tl[0] + eps)
                        + 0.8 * el[1] / (tl[1] + eps)
                        + 0.5 * el[2] / (tl[2] + eps);
                }
            }
            tile_losses
        };

        let expected = eval_cpu_mode_a_f64(&pred_f64);
        let d_init = vec![[0.0_f32; 2]; num_stems * num_frames];
        let bl_init = vec![[0.0_f32; 4]; num_stems * num_windows];
        let (out, _) = dispatch_kernel(
            ctx,
            KernelId::GaborBezoutGrad,
            variant,
            (num_windows as u32, num_stems as u32, 1),
            uniforms.as_bytes(),
            &[
                (false, as_byte_slice(&pred)),
                (false, as_byte_slice(&tgt)),
                (false, as_byte_slice(&taps)),
                (true, as_byte_slice(&d_init)),
                (true, as_byte_slice(&bl_init)),
            ],
        );
        let got_grad = read_vec2_vec(&out[0]);
        let got_losses = read_vec4_vec(&out[1]);
        for i in 0..expected.len() {
            assert!((got_losses[i][3] as f64 - expected[i]).abs() < 1.0e-5);
        }

        let eps = 1.0e-6_f64;
        for &idx in &[0, 7, 251, 255, 256, 280, 310, 575] {
            let slot = (idx / num_frames) * num_windows + ((idx % num_frames) / 256);
            for ch in 0..2 {
                let mut pp = pred_f64.clone();
                let mut pm = pred_f64.clone();
                pp[idx][ch] += eps;
                pm[idx][ch] -= eps;
                let fd =
                    (eval_cpu_mode_a_f64(&pp)[slot] - eval_cpu_mode_a_f64(&pm)[slot]) / (2.0 * eps);
                let g = got_grad[idx][ch] as f64;
                assert!(
                    (g - fd).abs() <= 1.0e-4 * fd.abs().max(1.0),
                    "Mode A ({label}) FD mismatch idx={idx} ch={ch}: got={g} fd={fd}"
                );
            }
        }
    }

    // --- Mode B: Gabor FFT across all supported radix-2 sizes & single-stem multi-window ---
    for &fft_size in &[16_usize, 32, 64, 128, 256] {
        run_gabor_mode_b_case(ctx, variant, fft_size, 2, 2);
    }
    run_gabor_mode_b_case(ctx, variant, 64, 1, 3);
}

fn cpu_lbfgs_step_f64(
    theta: &mut [f64],
    grad: &[f64],
    bounds: &[[f32; 2]],
    lbfgs_s_y: &mut [[f64; 4]],
    search_dir: &mut [f64],
    u: &InnerOptUniforms,
) {
    let m = u.history_m as usize;
    let step_idx = u.step_idx;
    let line_search_iter = u.line_search_iter;
    let step_sz = u.step_size as f64;
    let damping = u.damping_lambda as f64;
    let min_eps = u.min_bound_eps as f64;
    let n = theta.len();
    let clamp_b = |cand: f64, b: [f32; 2]| -> f64 {
        let lo_e = (b[0] as f64) + min_eps;
        let hi_e = (b[1] as f64) - min_eps;
        cand.clamp(lo_e, hi_e)
    };
    if line_search_iter > 0 {
        for p in 0..n {
            theta[p] = clamp_b(lbfgs_s_y[m * n + p][0] + step_sz * search_dir[p], bounds[p]);
        }
        return;
    }

    if step_idx > 0 && m > 0 {
        let mut ys = 0.0_f64;
        let mut ss = 0.0_f64;
        let mut yy = 0.0_f64;
        for p in 0..n {
            let s = theta[p] - lbfgs_s_y[m * n + p][0];
            let y = grad[p] - lbfgs_s_y[m * n + p][1];
            ys += s * y;
            ss += s * s;
            yy += y * y;
        }
        let slot = ((step_idx - 1) as usize) % m;
        let valid = ys.is_finite() && ys > 1.0e-12 * (ss + yy) && ys > 0.0;
        for p in 0..n {
            lbfgs_s_y[slot * n + p] = if valid {
                [
                    theta[p] - lbfgs_s_y[m * n + p][0],
                    grad[p] - lbfgs_s_y[m * n + p][1],
                    0.0,
                    0.0,
                ]
            } else {
                [0.0; 4]
            };
        }
    }

    let mut q = grad.to_vec();
    let used = if step_idx == 0 || m == 0 {
        0
    } else {
        (step_idx as usize).min(m)
    };
    let mut alphas = vec![0.0_f64; used];
    let mut rhos = vec![0.0_f64; used];
    let mut found_h0 = false;
    let mut h0_scale = 1.0 / (1.0 + damping);

    for age in 0..used {
        let slot = ((step_idx as usize) - 1 - age) % m;
        let mut sy = 0.0_f64;
        let mut sq = 0.0_f64;
        let mut yy = 0.0_f64;
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
        let slot = ((step_idx as usize) - 1 - age) % m;
        let mut yq = 0.0_f64;
        for p in 0..n {
            yq += lbfgs_s_y[slot * n + p][1] * q[p];
        }
        let beta = rhos[age] * yq;
        let coeff = alphas[age] - beta;
        for p in 0..n {
            q[p] += lbfgs_s_y[slot * n + p][0] * coeff;
        }
    }

    let gd: f64 = grad.iter().zip(q.iter()).map(|(g, qv)| g * (-qv)).sum();
    let fallback = !gd.is_finite() || gd >= 0.0 || q.iter().any(|v| !v.is_finite());
    for p in 0..n {
        let dir = if fallback {
            -grad[p] / (1.0 + damping)
        } else {
            -q[p]
        };
        lbfgs_s_y[m * n + p] = [theta[p], grad[p], if used > 0 { 1.0 } else { 0.0 }, 0.0];
        search_dir[p] = dir;
        theta[p] = clamp_b(theta[p] + step_sz * dir, bounds[p]);
    }
}

pub fn verify_inner_opt_step_lbfgs_and_bounds(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();

    // Case 1: Bounded 30-param L-BFGS hitting box constraints & Armijo backtracking.
    {
        let num_params = 30_usize;
        let history_m = 8_usize;
        let min_eps = 1.0e-4_f32;
        let mut theta = vec![2.5_f32; num_params];
        let mut bounds = vec![[-3.0_f32, 3.0_f32]; num_params];
        bounds[0] = [-0.25, 0.25];
        bounds[1] = [-0.15, 0.15];
        let mut lbfgs_s_y = vec![[0.0_f32; 4]; (history_m + 1) * num_params];
        let mut search_dir = vec![0.0_f32; num_params];

        let target: Vec<f32> = (0..num_params).map(|i| -0.6 + 0.05 * (i as f32)).collect();
        let eval_obj = |x: &[f32]| -> (f32, Vec<f32>) {
            let mut f = 0.0_f32;
            let mut g = vec![0.0_f32; num_params];
            for i in 0..num_params {
                let w = 1.0 + 0.5 * (i as f32);
                let d = x[i] - target[i];
                f += 0.5 * w * d * d;
                g[i] = w * d;
            }
            (f, g)
        };

        let (mut f_cur, mut grad) = eval_obj(&theta);
        let initial_f = f_cur;
        let mut verified_backtrack = false;

        for step in 0..24_u32 {
            let mut alpha_step = if step == 1 { 4.0_f32 } else { 1.0_f32 };
            for ls in 0..8_u32 {
                let uniforms = InnerOptUniforms {
                    num_params: num_params as u32,
                    history_m: history_m as u32,
                    step_idx: step,
                    line_search_iter: ls,
                    step_size: alpha_step,
                    damping_lambda: 1.0e-4,
                    min_bound_eps: min_eps,
                    _pad: 0,
                };
                let (out, _) = dispatch_kernel(
                    ctx,
                    KernelId::InnerOptStep,
                    variant,
                    (1, 1, 1),
                    uniforms.as_bytes(),
                    &[
                        (true, as_byte_slice(&theta)),
                        (false, as_byte_slice(&grad)),
                        (false, as_byte_slice(&bounds)),
                        (true, as_byte_slice(&lbfgs_s_y)),
                        (true, as_byte_slice(&search_dir)),
                    ],
                );
                let next_theta = read_f32_vec(&out[0]);
                let next_sy = read_vec4_vec(&out[1]);
                let next_dir = read_f32_vec(&out[2]);

                for i in 0..num_params {
                    assert!(
                        next_theta[i] >= bounds[i][0] + min_eps - 1.0e-6
                            && next_theta[i] <= bounds[i][1] - min_eps + 1.0e-6,
                        "inner_opt_step ({label}) violated box bounds at {i}: {}",
                        next_theta[i]
                    );
                }
                if ls > 0 {
                    assert_eq!(
                        next_sy, lbfgs_s_y,
                        "Armijo ls > 0 mutated lbfgs_s_y ({label})"
                    );
                    assert_eq!(
                        next_dir, search_dir,
                        "Armijo ls > 0 mutated search_dir ({label})"
                    );
                    verified_backtrack = true;
                }

                let (f_try, g_try) = eval_obj(&next_theta);
                theta = next_theta;
                lbfgs_s_y = next_sy;
                search_dir = next_dir;
                if f_try <= f_cur {
                    f_cur = f_try;
                    grad = g_try;
                    break;
                }
                alpha_step *= 0.5;
            }
        }

        assert!(
            verified_backtrack,
            "expected at least one Armijo backtrack trial ({label})"
        );
        assert!(
            f_cur < 0.05 * initial_f,
            "L-BFGS ({label}) did not converge: {f_cur}/{initial_f}"
        );
        assert!((theta[0] - (bounds[0][0] + min_eps)).abs() < 1.0e-4);
        assert!((theta[1] - (bounds[1][0] + min_eps)).abs() < 1.0e-4);
    }

    // Case 2: Large num_params = 600 (> 256, 3 strided tiles) exact CPU f64 L-BFGS parity on (1, 1, 1).
    let n = 600_usize;
    let m = 4_usize;
    let min_eps = 1.0e-4_f32;
    let damping = 1.0e-4_f32;
    let mut theta_gpu: Vec<f32> = (0..n).map(|i| 1.25 - 0.002 * ((i % 25) as f32)).collect();
    let mut theta_cpu: Vec<f64> = theta_gpu.iter().map(|&v| v as f64).collect();
    let mut bounds = vec![[-3.0_f32, 3.0_f32]; n];
    bounds[0] = [-0.20, 0.20];
    bounds[300] = [-0.15, 0.15];
    let mut sy_gpu = vec![[0.0_f32; 4]; (m + 1) * n];
    let mut sy_cpu = vec![[0.0_f64; 4]; (m + 1) * n];
    let mut dir_gpu = vec![0.0_f32; n];
    let mut dir_cpu = vec![0.0_f64; n];

    let target: Vec<f64> = (0..n).map(|i| -0.45 + 0.03 * ((i % 21) as f64)).collect();
    let eval_f64 = |x: &[f64]| -> (f64, Vec<f64>) {
        let mut f = 0.0_f64;
        let mut g = vec![0.0_f64; n];
        for i in 0..n {
            let w = 1.0 + 0.08 * ((i % 19) as f64);
            let d = x[i] - target[i];
            f += 0.5 * w * d * d;
            g[i] = w * d;
        }
        (f, g)
    };
    let eval_f32 = |x: &[f32]| -> (f32, Vec<f32>) {
        let mut f = 0.0_f32;
        let mut g = vec![0.0_f32; n];
        for i in 0..n {
            let w = 1.0 + 0.08 * ((i % 19) as f32);
            let d = x[i] - (target[i] as f32);
            f += 0.5 * w * d * d;
            g[i] = w * d;
        }
        (f, g)
    };

    let (mut f_gpu, mut g_gpu) = eval_f32(&theta_gpu);
    let (_f_c, mut g_cpu) = eval_f64(&theta_cpu);
    let init_f = f_gpu;
    let mut saw_backtrack = false;

    for step in 0..12_u32 {
        let mut alpha_step = if step == 1 { 4.0_f32 } else { 1.0_f32 };
        for ls in 0..6_u32 {
            let u = InnerOptUniforms {
                num_params: n as u32,
                history_m: m as u32,
                step_idx: step,
                line_search_iter: ls,
                step_size: alpha_step,
                damping_lambda: damping,
                min_bound_eps: min_eps,
                _pad: 0,
            };
            let (out, _) = dispatch_kernel(
                ctx,
                KernelId::InnerOptStep,
                variant,
                (1, 1, 1),
                u.as_bytes(),
                &[
                    (true, as_byte_slice(&theta_gpu)),
                    (false, as_byte_slice(&g_gpu)),
                    (false, as_byte_slice(&bounds)),
                    (true, as_byte_slice(&sy_gpu)),
                    (true, as_byte_slice(&dir_gpu)),
                ],
            );
            theta_gpu = read_f32_vec(&out[0]);
            sy_gpu = read_vec4_vec(&out[1]);
            dir_gpu = read_f32_vec(&out[2]);

            cpu_lbfgs_step_f64(
                &mut theta_cpu,
                &g_cpu,
                &bounds,
                &mut sy_cpu,
                &mut dir_cpu,
                &u,
            );

            for p in 0..n {
                assert!(
                    (theta_gpu[p] as f64 - theta_cpu[p]).abs() < 1.0e-4,
                    "600-param L-BFGS ({label}) step={step} ls={ls} p={p} theta gpu={} cpu={}",
                    theta_gpu[p],
                    theta_cpu[p]
                );
                assert!(
                    (dir_gpu[p] as f64 - dir_cpu[p]).abs() < 2.0e-4,
                    "600-param L-BFGS ({label}) step={step} ls={ls} p={p} dir gpu={} cpu={}",
                    dir_gpu[p],
                    dir_cpu[p]
                );
            }
            if ls > 0 {
                saw_backtrack = true;
            }
            let (f_try, g_try) = eval_f32(&theta_gpu);
            if f_try <= f_gpu {
                f_gpu = f_try;
                g_gpu = g_try;
                let (_fc, gc) = eval_f64(&theta_cpu);
                g_cpu = gc;
                break;
            }
            alpha_step *= 0.5;
        }
    }

    assert!(
        saw_backtrack,
        "expected Armijo backtrack in 600-param run ({label})"
    );
    assert!(
        f_gpu < 0.05 * init_f,
        "600-param L-BFGS did not converge: {f_gpu}/{init_f}"
    );
    assert!((theta_gpu[300] - (bounds[300][0] + min_eps)).abs() < 1.0e-4);
}
