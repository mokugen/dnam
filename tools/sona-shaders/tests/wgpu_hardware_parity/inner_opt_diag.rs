#![forbid(unsafe_code)]

//! Multi-family diagonal `H_0` preconditioning parity tests (`1e-8 .. 1e2`) for
//! `inner_opt_step` (`INNER_OPT_MODE_DIAG_PRECOND`, `@binding(6)`).

use super::gabor_and_opt::cpu_lssbroyden_step_f64;
use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, dispatch_kernel, read_f32_vec, read_vec4_vec,
};
use sona_shaders::abi::{
    COORD_TRANSFORM_FREE, INNER_OPT_MODE_DIAG_PRECOND, INNER_OPT_MODE_LSSBROYDEN,
    INNER_OPT_MODE_RESET_STATE, InnerOptUniforms,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

pub fn verify_inner_opt_diag_precond(ctx: &GpuContext, variant: ShaderVariant, label: &str) {
    // Case 4: Multi-family diagonal preconditioning (`1e-8 .. 1e2`) RESET_STATE, age==0, H_0*q, fallback.
    let n_dp = 16_usize;
    let m_dp = 4_usize;
    let diag_p: Vec<f32> = (0..n_dp)
        .map(|i| match i % 4 {
            0 => 1.0e-8,
            1 => 1.0e-3,
            2 => 1.0,
            _ => 1.0e2,
        })
        .collect();
    let bounds_dp = vec![[-100.0_f32, 100.0, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)]; n_dp];
    let mut th_gpu_dp: Vec<f32> = (0..n_dp).map(|i| 0.4 + 0.05 * (i as f32)).collect();
    let mut th_cpu_dp: Vec<f64> = th_gpu_dp.iter().map(|&v| v as f64).collect();
    let mut sy_gpu_dp = vec![[0.0_f32; 4]; (m_dp + 1) * n_dp];
    let mut sy_cpu_dp = vec![[0.0_f64; 4]; (m_dp + 1) * n_dp];
    let mut dir_gpu_dp = vec![0.0_f32; n_dp];
    let mut dir_cpu_dp = vec![0.0_f64; n_dp];
    let g_dp_f32: Vec<f32> = (0..n_dp).map(|i| 0.3 - 0.03 * (i as f32)).collect();

    for (step, flags) in [
        (
            0_u32,
            INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND | INNER_OPT_MODE_RESET_STATE,
        ),
        (
            0_u32,
            INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND,
        ),
        (
            1_u32,
            INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND,
        ),
        (
            2_u32,
            INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND,
        ),
    ] {
        let g_step_f32: Vec<f32> = g_dp_f32
            .iter()
            .enumerate()
            .map(|(i, &g)| g * (1.0 - 0.1 * (step as f32)) + 0.02 * (i as f32))
            .collect();
        let g_step_f64: Vec<f64> = g_step_f32.iter().map(|&v| v as f64).collect();
        let u = InnerOptUniforms {
            num_params: n_dp as u32,
            history_m: m_dp as u32,
            step_idx: step,
            mode_flags: flags,
            step_size: 0.25,
            damping_lambda: 1.0e-4,
            min_bound_eps: 1.0e-4,
            grad_clip_norm: 0.0,
        };
        let (out, _) = dispatch_kernel(
            ctx,
            KernelId::InnerOptStep,
            variant,
            (1, 1, 1),
            u.as_bytes(),
            &[
                (true, as_byte_slice(&th_gpu_dp)),
                (false, as_byte_slice(&g_step_f32)),
                (false, as_byte_slice(&bounds_dp)),
                (true, as_byte_slice(&sy_gpu_dp)),
                (true, as_byte_slice(&dir_gpu_dp)),
                (false, as_byte_slice(&diag_p)),
            ],
        );
        th_gpu_dp = read_f32_vec(&out[0]);
        sy_gpu_dp = read_vec4_vec(&out[1]);
        dir_gpu_dp = read_f32_vec(&out[2]);
        cpu_lssbroyden_step_f64(
            0,
            &mut th_cpu_dp,
            &g_step_f64,
            &bounds_dp,
            &mut sy_cpu_dp,
            &mut dir_cpu_dp,
            Some(&diag_p),
            &u,
        );
        for p in 0..n_dp {
            let tol = 1.0e-4 * dir_cpu_dp[p].abs().max(1.0);
            assert!(
                (dir_gpu_dp[p] as f64 - dir_cpu_dp[p]).abs() <= tol,
                "diag_precond ({label}) step={step} flags={flags:#x} p={p}: gpu={} cpu={}",
                dir_gpu_dp[p],
                dir_cpu_dp[p]
            );
        }
    }
}
