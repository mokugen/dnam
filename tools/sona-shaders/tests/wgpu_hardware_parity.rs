#![forbid(unsafe_code)]

//! End-to-end real `wgpu` Metal/hardware execution parity and finite-difference verification across
//! both `portable` and `subgroups` variants of all six SONA v3.0 compute kernels
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[train.pipeline.isomorphism-parity]`).

#[path = "wgpu_hardware_parity/core_and_adjoint.rs"]
mod core_and_adjoint;
#[path = "wgpu_hardware_parity/gabor_and_opt.rs"]
mod gabor_and_opt;
#[path = "wgpu_hardware_parity/harness.rs"]
mod harness;
#[path = "wgpu_hardware_parity/ozaki_and_prescreen.rs"]
mod ozaki_and_prescreen;

use harness::init_gpu_context;

/// verify[train.pipeline.webgpu-scans-and-ozaki]
#[test]
fn test_wgpu_ozaki_gram_rhs_i32_and_startup_probes() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        ozaki_and_prescreen::verify_startup_probes_and_ozaki_parity(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
/// verify[train.pipeline.isomorphism-parity]
#[test]
fn test_wgpu_fused_core_scan_256_hardware_parity_and_zero_dc() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        core_and_adjoint::verify_fused_core_scan_256_parity_and_zero_dc(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
#[test]
fn test_wgpu_adjoint_scan_256_three_pass_finite_difference_parity() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        core_and_adjoint::verify_adjoint_scan_256_fd_parity(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
#[test]
fn test_wgpu_prescreen_population_hardware_parity() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        ozaki_and_prescreen::verify_prescreen_population_parity(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
#[test]
fn test_wgpu_gabor_bezout_grad_modes_a_and_b_finite_difference() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        gabor_and_opt::verify_gabor_bezout_parity(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
#[test]
fn test_wgpu_inner_opt_step_lbfgs_and_bounds() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        gabor_and_opt::verify_inner_opt_step_lbfgs_and_bounds(&ctx, variant);
    }
}
