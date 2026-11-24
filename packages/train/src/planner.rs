//! 2-D (`time_block × stem_group`) VRAM and WebGPU storage binding planner enforcing the strict
//! `<= 128 MiB` single-binding cap and profile total-memory ceilings across browser, WASM fallback,
//! desktop capsule, and streaming host solves (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::mem::size_of;

use crate::error::TrainError;
use crate::gpu::abi::{
    CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE, FEATURE_VEC4_STRIDE,
    QUAD_PARAMS_VEC4_STRIDE, TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
};

/// Maximum frames per chunk in wave-batched `run_wave_core_cascade` / `run_wave_core_adjoint`
/// (`64` tiles = `16,384` frames; keeps Ultra 10-quad `feature_outputs` `<= 15.0 MiB` <= `128 MiB`).
pub const WAVE_STREAMING_MAX_FRAMES: usize = 64 * TILE_FRAMES;

/// Strict WebGPU storage buffer single-binding cap (`128 MiB`).
pub const MAX_WEBGPU_BINDING_BYTES: u64 = 128 * 1024 * 1024;
/// Combined host+device memory ceiling in browser WebGPU workers (`256 MiB`).
pub const BROWSER_WEBGPU_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
/// CPU SIMD fallback budget ceiling in WASM/resource-constrained workers (`64 MiB`).
pub const CPU_WASM_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// Desktop capsule solve memory ceiling (`512 MiB`).
pub const DESKTOP_CAPSULE_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
/// Default native host streaming memory ceiling (`4 GiB`).
pub const STREAMING_HOST_DEFAULT_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Default streaming single-tensor cap (`1 GiB`).
pub const STREAMING_HOST_MAX_TENSOR_BYTES: u64 = 1024 * 1024 * 1024;
/// Number of extra checkpoint tile slots needed per streamed time block (`+1` lead dummy resume tile + `+1` exit tile).
pub const STREAMED_CKPT_EXTRA_TILES: usize = 2;

const VEC4_BYTES: u64 = size_of::<[f32; 4]>() as u64;

const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(WAVE_STREAMING_MAX_FRAMES == 16_384);
const _: () = assert!(
    10 * WAVE_STREAMING_MAX_FRAMES * FEATURE_VEC4_STRIDE * size_of::<[f32; 4]>() == 15_728_640
);
const _: () = assert!(STREAMED_CKPT_EXTRA_TILES == 2);
const _: () = assert!(MAX_WEBGPU_BINDING_BYTES == 134_217_728);
const _: () = assert!(CPU_WASM_BUDGET_BYTES < BROWSER_WEBGPU_BUDGET_BYTES);
const _: () = assert!(BROWSER_WEBGPU_BUDGET_BYTES < DESKTOP_CAPSULE_BUDGET_BYTES);

/// Target memory envelope driving 2-D batching vs streaming selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBudgetProfile {
    /// Browser WebGPU worker (`<= 128 MiB` per binding, `<= 256 MiB` total).
    BrowserWebGpu,
    /// CPU SIMD fallback (`< 64 MiB` total workspace).
    CpuWasmFallback,
    /// Desktop capsule trainer (`<= 128 MiB` per binding, `<= 512 MiB` total).
    DesktopCapsule,
    /// Multi-minute native capture streaming host (`4..8 GiB` total, configurable tensor cap).
    StreamingHost {
        /// Total combined device + host workspace ceiling in bytes.
        total_budget_bytes: u64,
        /// Per-tensor storage buffer ceiling in bytes.
        max_tensor_bytes: u64,
    },
}

impl MemoryBudgetProfile {
    /// Returns `(default_total_budget_bytes, default_binding_cap_bytes)` for this profile.
    #[inline]
    #[must_use]
    pub const fn limits(self) -> (u64, u64) {
        match self {
            Self::BrowserWebGpu => (BROWSER_WEBGPU_BUDGET_BYTES, MAX_WEBGPU_BINDING_BYTES),
            Self::CpuWasmFallback => (CPU_WASM_BUDGET_BYTES, CPU_WASM_BUDGET_BYTES),
            Self::DesktopCapsule => (DESKTOP_CAPSULE_BUDGET_BYTES, MAX_WEBGPU_BINDING_BYTES),
            Self::StreamingHost {
                total_budget_bytes,
                max_tensor_bytes,
            } => (total_budget_bytes, max_tensor_bytes),
        }
    }
}

/// Selected forward/adjoint execution schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Entire workload executes in one resident `(num_quads, 1, num_stems)` GPU pass.
    ResidentBatched,
    /// Workload streams over tiled `(time_block_frames, stem_group)` slices bounded to binding caps.
    Streamed,
}

/// Input dimensions describing a `fused_core_scan_256` / `adjoint_scan_256` workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreScanWorkload {
    /// Frames per stem.
    pub num_frames: usize,
    /// Active `CoreQuad` count (`1..=16`).
    pub num_quads: usize,
    /// Training stem count.
    pub num_stems: usize,
    /// Whether adjoint backward buffers (`d_loss_d_features`, `d_loss_d_inputs`, `param_grads`, `tile_adj_boundaries`) are allocated together.
    pub include_adjoint: bool,
}

/// Per-buffer byte footprints computed for one scheduled dispatch slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreScanBindingFootprints {
    /// Storage size of `core_inputs` in bytes.
    pub core_inputs_bytes: u64,
    /// Storage size of `quad_params` in bytes.
    pub quad_params_bytes: u64,
    /// Storage size of `tile_checkpoints` in bytes (includes `STREAMED_CKPT_EXTRA_TILES` in `Streamed` mode).
    pub tile_checkpoints_bytes: u64,
    /// Storage size of `feature_outputs` in bytes.
    pub feature_outputs_bytes: u64,
    /// Storage size of adjoint scratch/output buffers when `include_adjoint == true`.
    pub adjoint_workspace_bytes: u64,
    /// Single largest storage buffer binding across the planned dispatch.
    pub max_single_binding_bytes: u64,
    /// Peak combined binding footprint across active workspace buffers.
    pub peak_total_bytes: u64,
}

/// Concrete 2-D `(time_block × stem_group)` execution schedule emitted by [`plan_core_scan_execution`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreScanPlan {
    /// Selected execution strategy (`ResidentBatched` or `Streamed`).
    pub mode: ExecutionMode,
    /// Number of stems processed per streamed group (`1..=num_stems`).
    pub stem_group: usize,
    /// Frames processed per time block (`<= num_frames`, multiple of `256` whenever streamed).
    pub time_block_frames: usize,
    /// Tiles processed per time block (`ceil(time_block_frames / 256)`).
    pub time_block_tiles: usize,
    /// Checkpoint tile slots provisioned per dispatch (`time_block_tiles + 2` in `Streamed` mode).
    pub ckpt_slot_tiles: usize,
    /// Effective per-binding limit enforced during planning.
    pub effective_binding_cap_bytes: u64,
    /// Effective combined total memory limit enforced during planning.
    pub effective_total_budget_bytes: u64,
    /// Resulting buffer byte footprints.
    pub footprints: CoreScanBindingFootprints,
}

fn evaluate_slice_footprints(
    workload: CoreScanWorkload,
    stem_group: usize,
    time_block_frames: usize,
    mode: ExecutionMode,
) -> CoreScanBindingFootprints {
    let total_q = (stem_group as u64).saturating_mul(workload.num_quads as u64);
    let frames = time_block_frames as u64;
    let time_tiles = time_block_frames.div_ceil(TILE_FRAMES) as u64;
    let ckpt_tiles = match mode {
        ExecutionMode::ResidentBatched => time_tiles,
        ExecutionMode::Streamed => time_tiles.saturating_add(STREAMED_CKPT_EXTRA_TILES as u64),
    };

    let core_inputs_bytes = total_q
        .saturating_mul(frames)
        .saturating_mul(CORE_INPUT_VEC4_STRIDE as u64)
        .saturating_mul(VEC4_BYTES);
    let quad_params_bytes = total_q
        .saturating_mul(QUAD_PARAMS_VEC4_STRIDE as u64)
        .saturating_mul(VEC4_BYTES);
    let tile_checkpoints_bytes = total_q
        .saturating_mul(ckpt_tiles)
        .saturating_mul(TILE_CKPT_VEC4_STRIDE as u64)
        .saturating_mul(VEC4_BYTES);
    let feature_outputs_bytes = total_q
        .saturating_mul(frames)
        .saturating_mul(FEATURE_VEC4_STRIDE as u64)
        .saturating_mul(VEC4_BYTES);

    let (adjoint_workspace_bytes, adj_max) = if workload.include_adjoint {
        let param_grads = total_q
            .saturating_mul(ckpt_tiles)
            .saturating_mul(QUAD_PARAMS_VEC4_STRIDE as u64)
            .saturating_mul(VEC4_BYTES);
        let adj_bounds = total_q
            .saturating_mul(time_tiles)
            .saturating_mul(TILE_ADJ_BOUNDARY_VEC4_STRIDE as u64)
            .saturating_mul(VEC4_BYTES);
        let exit_bounds = total_q
            .saturating_mul(EXIT_ADJ_BOUNDARY_VEC4_STRIDE as u64)
            .saturating_mul(VEC4_BYTES);
        let slot7_bounds = adj_bounds.saturating_add(exit_bounds);
        let total = feature_outputs_bytes
            .saturating_add(core_inputs_bytes)
            .saturating_add(param_grads)
            .saturating_add(slot7_bounds);
        let m = feature_outputs_bytes
            .max(core_inputs_bytes)
            .max(param_grads)
            .max(slot7_bounds);
        (total, m)
    } else {
        (0, 0)
    };

    let max_single_binding_bytes = core_inputs_bytes
        .max(quad_params_bytes)
        .max(tile_checkpoints_bytes)
        .max(feature_outputs_bytes)
        .max(adj_max);
    let peak_total_bytes = core_inputs_bytes
        .saturating_add(quad_params_bytes)
        .saturating_add(tile_checkpoints_bytes)
        .saturating_add(feature_outputs_bytes)
        .saturating_add(adjoint_workspace_bytes);

    CoreScanBindingFootprints {
        core_inputs_bytes,
        quad_params_bytes,
        tile_checkpoints_bytes,
        feature_outputs_bytes,
        adjoint_workspace_bytes,
        max_single_binding_bytes,
        peak_total_bytes,
    }
}

fn offending_buffer_name(fp: &CoreScanBindingFootprints, cap: u64) -> (&'static str, u64) {
    if fp.feature_outputs_bytes > cap {
        ("feature_outputs", fp.feature_outputs_bytes)
    } else if fp.core_inputs_bytes > cap {
        ("core_inputs", fp.core_inputs_bytes)
    } else if fp.tile_checkpoints_bytes > cap {
        ("tile_checkpoints", fp.tile_checkpoints_bytes)
    } else if fp.quad_params_bytes > cap {
        ("quad_params", fp.quad_params_bytes)
    } else if fp.adjoint_workspace_bytes > cap {
        ("adjoint_workspace", fp.adjoint_workspace_bytes)
    } else {
        ("peak_total_workspace", fp.peak_total_bytes)
    }
}

/// Computes a 2-D `(time_block × stem_group)` execution schedule respecting `profile` memory
/// limits and `adapter_max_storage_binding_bytes` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).
pub fn plan_core_scan_execution(
    workload: CoreScanWorkload,
    profile: MemoryBudgetProfile,
    adapter_max_storage_binding_bytes: Option<u64>,
) -> Result<CoreScanPlan, TrainError> {
    if workload.num_frames == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "plan_core_scan_execution(num_frames)",
            expected: 1,
            actual: 0,
        });
    }
    if workload.num_quads == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "plan_core_scan_execution(num_quads)",
            expected: 1,
            actual: 0,
        });
    }
    if workload.num_stems == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "plan_core_scan_execution(num_stems)",
            expected: 1,
            actual: 0,
        });
    }

    let (profile_total, profile_binding_cap) = profile.limits();
    let effective_binding_cap_bytes = match adapter_max_storage_binding_bytes {
        Some(limit) => profile_binding_cap.min(limit),
        None => profile_binding_cap,
    };
    let effective_total_budget_bytes = profile_total;

    let res_fp = evaluate_slice_footprints(
        workload,
        workload.num_stems,
        workload.num_frames,
        ExecutionMode::ResidentBatched,
    );
    if res_fp.max_single_binding_bytes <= effective_binding_cap_bytes
        && res_fp.peak_total_bytes <= effective_total_budget_bytes
    {
        let total_tiles = workload.num_frames.div_ceil(TILE_FRAMES);
        return Ok(CoreScanPlan {
            mode: ExecutionMode::ResidentBatched,
            stem_group: workload.num_stems,
            time_block_frames: workload.num_frames,
            time_block_tiles: total_tiles,
            ckpt_slot_tiles: total_tiles,
            effective_binding_cap_bytes,
            effective_total_budget_bytes,
            footprints: res_fp,
        });
    }

    let total_tiles = workload.num_frames.div_ceil(TILE_FRAMES).max(1);
    let mut stem_group = workload.num_stems;
    loop {
        let mut block_tiles = total_tiles;
        loop {
            let block_frames = (block_tiles * TILE_FRAMES).min(workload.num_frames);
            let fp = evaluate_slice_footprints(
                workload,
                stem_group,
                block_frames,
                ExecutionMode::Streamed,
            );
            if fp.max_single_binding_bytes <= effective_binding_cap_bytes
                && fp.peak_total_bytes <= effective_total_budget_bytes
            {
                return Ok(CoreScanPlan {
                    mode: ExecutionMode::Streamed,
                    stem_group,
                    time_block_frames: block_frames,
                    time_block_tiles: block_tiles,
                    ckpt_slot_tiles: block_tiles + STREAMED_CKPT_EXTRA_TILES,
                    effective_binding_cap_bytes,
                    effective_total_budget_bytes,
                    footprints: fp,
                });
            }
            if block_tiles == 1 {
                break;
            }
            block_tiles = (block_tiles / 2).max(1);
        }
        if stem_group == 1 {
            break;
        }
        stem_group = (stem_group / 2).max(1);
    }

    let min_frames = TILE_FRAMES.min(workload.num_frames);
    let min_fp = evaluate_slice_footprints(workload, 1, min_frames, ExecutionMode::Streamed);
    let (buf, bytes) = offending_buffer_name(&min_fp, effective_binding_cap_bytes);
    let cap = if min_fp.max_single_binding_bytes > effective_binding_cap_bytes {
        effective_binding_cap_bytes
    } else {
        effective_total_budget_bytes
    };
    Err(TrainError::BindingExceedsCap {
        buffer: buf,
        bytes,
        cap_bytes: cap,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_planner_resident_and_streamed_profiles() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        let small = CoreScanWorkload {
            num_frames: 600,
            num_quads: 4,
            num_stems: 2,
            include_adjoint: false,
        };
        let plan_res = plan_core_scan_execution(small, MemoryBudgetProfile::BrowserWebGpu, None)
            .expect("small workload must stay resident");
        assert_eq!(plan_res.mode, ExecutionMode::ResidentBatched);
        assert_eq!(plan_res.stem_group, 2);
        assert_eq!(plan_res.time_block_frames, 600);

        let large = CoreScanWorkload {
            num_frames: 192_000,
            num_quads: 8,
            num_stems: 16,
            include_adjoint: true,
        };
        let plan_stream = plan_core_scan_execution(large, MemoryBudgetProfile::BrowserWebGpu, None)
            .expect("large workload must stream within 128 MiB binding / 256 MiB total");
        assert_eq!(plan_stream.mode, ExecutionMode::Streamed);
        assert_eq!(plan_stream.time_block_frames % TILE_FRAMES, 0);
        assert_eq!(
            plan_stream.ckpt_slot_tiles,
            plan_stream.time_block_tiles + STREAMED_CKPT_EXTRA_TILES
        );
        assert!(plan_stream.footprints.max_single_binding_bytes <= MAX_WEBGPU_BINDING_BYTES);
        assert!(plan_stream.footprints.peak_total_bytes <= BROWSER_WEBGPU_BUDGET_BYTES);

        let cpu_plan = plan_core_scan_execution(large, MemoryBudgetProfile::CpuWasmFallback, None)
            .expect("large workload must fit < 64 MiB CPU fallback");
        assert!(cpu_plan.footprints.peak_total_bytes <= CPU_WASM_BUDGET_BYTES);

        let host_plan = plan_core_scan_execution(
            large,
            MemoryBudgetProfile::StreamingHost {
                total_budget_bytes: STREAMING_HOST_DEFAULT_BUDGET_BYTES,
                max_tensor_bytes: STREAMING_HOST_MAX_TENSOR_BYTES,
            },
            Some(MAX_WEBGPU_BINDING_BYTES),
        )
        .expect("streaming host plan must succeed");
        assert!(host_plan.footprints.max_single_binding_bytes <= MAX_WEBGPU_BINDING_BYTES);

        let tiny_cap =
            plan_core_scan_execution(small, MemoryBudgetProfile::DesktopCapsule, Some(64));
        assert!(matches!(
            tiny_cap,
            Err(TrainError::BindingExceedsCap { .. })
        ));

        assert!(matches!(
            plan_core_scan_execution(
                CoreScanWorkload {
                    num_frames: 0,
                    ..small
                },
                MemoryBudgetProfile::DesktopCapsule,
                None
            ),
            Err(TrainError::InvalidDimensions { .. })
        ));
        for bad in [
            CoreScanWorkload {
                num_quads: 0,
                ..small
            },
            CoreScanWorkload {
                num_stems: 0,
                ..small
            },
        ] {
            assert!(matches!(
                plan_core_scan_execution(bad, MemoryBudgetProfile::DesktopCapsule, None),
                Err(TrainError::InvalidDimensions { .. })
            ));
        }
        // Cover offending_buffer_name branches (feature_outputs, core_inputs, tile_checkpoints, quad_params, adjoint_workspace, peak_total)
        let mut fp = evaluate_slice_footprints(
            CoreScanWorkload {
                include_adjoint: true,
                ..small
            },
            1,
            16,
            ExecutionMode::Streamed,
        );
        assert_eq!(
            offending_buffer_name(&fp, fp.feature_outputs_bytes - 1).0,
            "feature_outputs"
        );
        fp.feature_outputs_bytes = 0;
        assert_eq!(
            offending_buffer_name(&fp, fp.core_inputs_bytes - 1).0,
            "core_inputs"
        );
        fp.core_inputs_bytes = 0;
        assert_eq!(
            offending_buffer_name(&fp, fp.tile_checkpoints_bytes - 1).0,
            "tile_checkpoints"
        );
        fp.tile_checkpoints_bytes = 0;
        assert_eq!(
            offending_buffer_name(&fp, fp.quad_params_bytes - 1).0,
            "quad_params"
        );
        fp.quad_params_bytes = 0;
        assert_eq!(
            offending_buffer_name(&fp, fp.adjoint_workspace_bytes - 1).0,
            "adjoint_workspace"
        );
        assert_eq!(
            offending_buffer_name(&fp, fp.peak_total_bytes).0,
            "peak_total_workspace"
        );
        assert!(matches!(
            plan_core_scan_execution(
                small,
                MemoryBudgetProfile::StreamingHost {
                    total_budget_bytes: 128,
                    max_tensor_bytes: MAX_WEBGPU_BINDING_BYTES,
                },
                None
            ),
            Err(TrainError::BindingExceedsCap {
                buffer: "peak_total_workspace",
                ..
            })
        ));
    }

    proptest! {
        #[test]
        fn prop_planner_enforces_binding_and_budget_bounds(
            num_frames in 1usize..120_000,
            num_quads in 1usize..8,
            num_stems in 1usize..12,
            include_adjoint in any::<bool>(),
        ) {
            let workload = CoreScanWorkload {
                num_frames,
                num_quads,
                num_stems,
                include_adjoint,
            };
            let plan = plan_core_scan_execution(
                workload,
                MemoryBudgetProfile::BrowserWebGpu,
                Some(MAX_WEBGPU_BINDING_BYTES),
            ).expect("planner must find a valid schedule under 128 MiB / 256 MiB");
            prop_assert!(plan.footprints.max_single_binding_bytes <= MAX_WEBGPU_BINDING_BYTES);
            prop_assert!(plan.footprints.peak_total_bytes <= BROWSER_WEBGPU_BUDGET_BYTES);
            if matches!(plan.mode, ExecutionMode::Streamed) {
                prop_assert!(
                    plan.time_block_frames == workload.num_frames
                        || plan.time_block_frames.is_multiple_of(TILE_FRAMES)
                );
                prop_assert_eq!(
                    plan.ckpt_slot_tiles,
                    plan.time_block_tiles + STREAMED_CKPT_EXTRA_TILES
                );
            }
        }
    }
}
