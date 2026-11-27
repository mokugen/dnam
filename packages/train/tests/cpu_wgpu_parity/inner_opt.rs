#![forbid(unsafe_code)]
//! Hardware parity tests for `inner_opt_step` per-family diagonal preconditioning
//! (`INNER_OPT_MODE_DIAG_PRECOND`, `@binding(6) diag_precond`), `RESET_STATE`, `age == 0`
//! `y^T H_0 y`, middle `H_0 * q` scaling, damping fallback, `STE_FP16`, multi-candidate dispatch,
//! and `diag_precond: None` backward compatibility (`verify[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_train::{
    CpuRunner, InnerOptInput,
    abi::{
        COORD_TRANSFORM_ASINH_BOX_SYM, COORD_TRANSFORM_BOX, COORD_TRANSFORM_FREE,
        INNER_OPT_MODE_DIAG_PRECOND, INNER_OPT_MODE_LSSBROYDEN, INNER_OPT_MODE_RESET_STATE,
        INNER_OPT_MODE_STE_FP16, InnerOptUniforms, STE_FP16_META_BIT,
    },
};

use super::support::acquire_variants;

#[test]
fn test_wgpu_inner_opt_diag_precond_multi_family_and_fallback_parity() {
    let runners = acquire_variants();
    if runners.is_empty() {
        return;
    }
    let cpu = CpuRunner::detect();

    let num_cands = 3_usize;
    let n = 24_usize;
    let m = 6_usize;

    let diag_precond: Vec<f32> = (0..n)
        .map(|p| match p % 4 {
            0 => 1.0e-10,
            1 => 1.0e-4,
            2 => 1.0,
            _ => 1.0e2,
        })
        .collect();

    let bounds: Vec<[f32; 4]> = (0..n)
        .map(|p| match p % 3 {
            0 => [-10.0, 10.0, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)],
            1 => [
                -2.0,
                2.0,
                1.0,
                f32::from_bits(COORD_TRANSFORM_BOX | STE_FP16_META_BIT),
            ],
            _ => [
                -4.0,
                4.0,
                0.5,
                f32::from_bits(COORD_TRANSFORM_ASINH_BOX_SYM | STE_FP16_META_BIT),
            ],
        })
        .collect();

    let theta0: Vec<f32> = (0..num_cands * n)
        .map(|i| 0.35 * ((i as f32) * 0.23).sin() + 0.1)
        .collect();

    for (variant, gpu) in &runners {
        let label = variant.label();

        // Subtest A: Multi-candidate L-SSBroyden with RESET_STATE, multi-step history (age==0 y^T H_0 y,
        // middle H_0 * q scaling), STE_FP16, and diag_precond spanning [1e-10, 1e2].
        let mut tc = theta0.clone();
        let mut tg = theta0.clone();
        let mut syc = vec![[0.0_f32; 4]; num_cands * (m + 1) * n];
        let mut syg = syc.clone();
        let mut dc = vec![0.0_f32; num_cands * n];
        let mut dg = dc.clone();

        for (step, extra_flags) in [
            (0_u32, INNER_OPT_MODE_RESET_STATE),
            (0_u32, 0),
            (1_u32, 0),
            (2_u32, INNER_OPT_MODE_STE_FP16),
            (3_u32, INNER_OPT_MODE_STE_FP16),
        ] {
            let grad: Vec<f32> = tc
                .iter()
                .enumerate()
                .map(|(k, &v)| 0.45 * v + 0.15 * ((k + step as usize) as f32).cos())
                .collect();
            let u = InnerOptUniforms {
                num_params: n as u32,
                history_m: m as u32,
                step_idx: step,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND | extra_flags,
                step_size: 0.2,
                damping_lambda: 1.0e-4,
                min_bound_eps: 1.0e-4,
                grad_clip_norm: 0.0,
            };
            let oc = cpu
                .inner_opt_step(&InnerOptInput {
                    uniforms: u,
                    num_candidates: num_cands,
                    theta: &tc,
                    grad: &grad,
                    param_bounds: &bounds,
                    lbfgs_s_y: &syc,
                    search_dir: &dc,
                    diag_precond: Some(&diag_precond),
                })
                .unwrap();
            let og = gpu
                .inner_opt_step(&InnerOptInput {
                    uniforms: u,
                    num_candidates: num_cands,
                    theta: &tg,
                    grad: &grad,
                    param_bounds: &bounds,
                    lbfgs_s_y: &syg,
                    search_dir: &dg,
                    diag_precond: Some(&diag_precond),
                })
                .unwrap();
            tc = oc.theta;
            tg = og.theta;
            syc = oc.lbfgs_s_y;
            syg = og.lbfgs_s_y;
            dc = oc.search_dir;
            dg = og.search_dir;

            for i in 0..num_cands * n {
                let tol = 2.0e-5 * dc[i].abs().max(1.0);
                assert!(
                    (tc[i] - tg[i]).abs() <= tol && (dc[i] - dg[i]).abs() <= tol,
                    "diag_precond ({label}) step={step} extra={extra_flags:#x} i={i}: \
                     th=({},{}) dir=({},{})",
                    tc[i],
                    tg[i],
                    dc[i],
                    dg[i]
                );
            }
        }

        // Subtest B: Force fallback path (`cur_g` vs corrupted non-finite `lbfgs_s_y` history) and
        // verify `-cur_g * h0_p * inv_damping` matches between CPU and GPU.
        let mut sy_bad = vec![[0.0_f32; 4]; (m + 1) * n];
        sy_bad[0] = [1.0, 1.0, 0.0, 0.0];
        sy_bad[0][0] = f32::NAN;
        let grad_fb: Vec<f32> = (0..n).map(|i| 0.25 + 0.05 * (i as f32)).collect();
        let u_fb = InnerOptUniforms {
            num_params: n as u32,
            history_m: m as u32,
            step_idx: 1,
            mode_flags: INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND,
            step_size: 0.1,
            damping_lambda: 0.25,
            min_bound_eps: 1.0e-4,
            grad_clip_norm: 0.0,
        };
        let fb_c = cpu
            .inner_opt_step(&InnerOptInput {
                uniforms: u_fb,
                num_candidates: 1,
                theta: &theta0[..n],
                grad: &grad_fb,
                param_bounds: &bounds,
                lbfgs_s_y: &sy_bad,
                search_dir: &vec![0.0_f32; n],
                diag_precond: Some(&diag_precond),
            })
            .unwrap();
        let fb_g = gpu
            .inner_opt_step(&InnerOptInput {
                uniforms: u_fb,
                num_candidates: 1,
                theta: &theta0[..n],
                grad: &grad_fb,
                param_bounds: &bounds,
                lbfgs_s_y: &sy_bad,
                search_dir: &vec![0.0_f32; n],
                diag_precond: Some(&diag_precond),
            })
            .unwrap();
        for i in 0..n {
            assert!(
                (fb_c.search_dir[i] - fb_g.search_dir[i]).abs() <= 1.0e-5,
                "fallback ({label}) i={i}: {} vs {}",
                fb_c.search_dir[i],
                fb_g.search_dir[i]
            );
        }

        // Subtest C: `diag_precond: None` backward compatibility.
        let none_c = cpu
            .inner_opt_step(&InnerOptInput {
                uniforms: InnerOptUniforms {
                    mode_flags: INNER_OPT_MODE_LSSBROYDEN,
                    ..u_fb
                },
                num_candidates: 1,
                theta: &theta0[..n],
                grad: &grad_fb,
                param_bounds: &bounds,
                lbfgs_s_y: &vec![[0.0; 4]; (m + 1) * n],
                search_dir: &vec![0.0_f32; n],
                diag_precond: None,
            })
            .unwrap();
        let none_g = gpu
            .inner_opt_step(&InnerOptInput {
                uniforms: InnerOptUniforms {
                    mode_flags: INNER_OPT_MODE_LSSBROYDEN,
                    ..u_fb
                },
                num_candidates: 1,
                theta: &theta0[..n],
                grad: &grad_fb,
                param_bounds: &bounds,
                lbfgs_s_y: &vec![[0.0; 4]; (m + 1) * n],
                search_dir: &vec![0.0_f32; n],
                diag_precond: None,
            })
            .unwrap();
        for i in 0..n {
            assert!((none_c.theta[i] - none_g.theta[i]).abs() <= 1.0e-5);
        }

        // Subtest D: Positive/negative subnormal FTZ parity (`< f32::MIN_POSITIVE` falls back to `1.0`)
        // alongside NaN/Inf/negative/out-of-range clamping across CPU and Metal `wgpu`.
        let edge_dp: Vec<f32> = (0..n)
            .map(|i| match i % 8 {
                0 => 1.0e-40_f32,
                1 => -1.0e-40_f32,
                2 => f32::NAN,
                3 => f32::INFINITY,
                4 => -5.0,
                5 => 1.0e-20,
                6 => 1.0e20,
                _ => f32::MIN_POSITIVE,
            })
            .collect();
        let u_edge = InnerOptUniforms {
            mode_flags: INNER_OPT_MODE_LSSBROYDEN
                | INNER_OPT_MODE_DIAG_PRECOND
                | INNER_OPT_MODE_RESET_STATE,
            ..u_fb
        };
        let edge_c = cpu
            .inner_opt_step(&InnerOptInput {
                uniforms: u_edge,
                num_candidates: 1,
                theta: &theta0[..n],
                grad: &grad_fb,
                param_bounds: &bounds,
                lbfgs_s_y: &vec![[0.0; 4]; (m + 1) * n],
                search_dir: &vec![0.0_f32; n],
                diag_precond: Some(&edge_dp),
            })
            .unwrap();
        let edge_g = gpu
            .inner_opt_step(&InnerOptInput {
                uniforms: u_edge,
                num_candidates: 1,
                theta: &theta0[..n],
                grad: &grad_fb,
                param_bounds: &bounds,
                lbfgs_s_y: &vec![[0.0; 4]; (m + 1) * n],
                search_dir: &vec![0.0_f32; n],
                diag_precond: Some(&edge_dp),
            })
            .unwrap();
        assert_eq!(edge_c.search_dir[0], -grad_fb[0]);
        for i in 0..n {
            assert_eq!(
                edge_c.search_dir[i].to_bits(),
                edge_g.search_dir[i].to_bits(),
                "edge/subnormal FTZ parity ({label}) i={i}: cpu={} ({:#x}) vs gpu={} ({:#x})",
                edge_c.search_dir[i],
                edge_c.search_dir[i].to_bits(),
                edge_g.search_dir[i],
                edge_g.search_dir[i].to_bits()
            );
        }
    }
}
