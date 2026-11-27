#![forbid(unsafe_code)]

//! End-to-end real `wgpu` Metal/hardware execution parity and finite-difference verification across
//! both `portable` and `subgroups` variants of all seven SONA v3.0 compute kernels
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[train.pipeline.isomorphism-parity]`).

#[path = "wgpu_hardware_parity/block0_phs.rs"]
mod block0_phs;
#[path = "wgpu_hardware_parity/core_and_adjoint.rs"]
mod core_and_adjoint;
#[path = "wgpu_hardware_parity/gabor_and_opt.rs"]
mod gabor_and_opt;
#[path = "wgpu_hardware_parity/harness.rs"]
mod harness;
#[path = "wgpu_hardware_parity/inner_opt_diag.rs"]
mod inner_opt_diag;
#[path = "wgpu_hardware_parity/ozaki_and_prescreen.rs"]
mod ozaki_and_prescreen;
#[path = "wgpu_hardware_parity/paralru_fastpath.rs"]
mod paralru_fastpath;
#[path = "wgpu_hardware_parity/tile_mix.rs"]
mod tile_mix;
#[path = "wgpu_hardware_parity/v4_adjoint_and_clamp.rs"]
mod v4_adjoint_and_clamp;
#[path = "wgpu_hardware_parity/wave_and_streamed.rs"]
mod wave_and_streamed;

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
fn test_wgpu_block0_phs_core_and_adjoint_fd_parity() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        block0_phs::verify_block0_phs_core_and_adjoint_fd(&ctx, variant);
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

/// verify[train.pipeline.webgpu-scans-and-ozaki]
#[test]
fn test_wgpu_tile_mix_fwd_adj_ultra_dag_parity_and_determinism() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        tile_mix::verify_tile_mix_ultra_dag_parity(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
/// verify[dsp.op-n.core.matryoshka-blocks]
#[test]
fn test_wgpu_paralru_fastpath_and_const_m_scans() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        paralru_fastpath::verify_paralru_fastpath_and_const_m_scans(&ctx, variant);
    }
}

/// verify[train.pipeline.webgpu-scans-and-ozaki]
/// verify[dsp.op-n.polyadaa.clamp]
/// verify[dsp.op-n.core.interblock-tiles]
#[test]
fn test_wgpu_v4_adjoint_clamp_se_bus_and_pred_corr_phs() {
    let Some(ctx) = init_gpu_context() else {
        return;
    };
    for variant in ctx.variants() {
        v4_adjoint_and_clamp::verify_v4_adjoint_and_clamp_suite(&ctx, variant);
    }
}
