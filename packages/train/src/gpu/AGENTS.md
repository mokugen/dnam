# `dnam-train::gpu` — WGSL Shader Manifest & Native `wgpu` Compute Map

Scoped discovery guide for [`src/gpu.rs`](../gpu.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]` and `zerocopy` ABI structs; consumes checked-in `.wgsl` artifacts from [`tools/sona-shaders`](../../../../tools/sona-shaders/AGENTS.md) with feature-gated `gpu-wgpu` native execution.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`abi.rs`](abi.rs) | `zerocopy` uniform/parameter WGSL ABI layouts (`QUAD_PARAMS_VEC4_STRIDE`, `TILE_FRAMES`, strides) | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`shaders.rs`](shaders.rs) | Embedded `.wgsl` strings, SHA-256 `manifest.json` verification & browser WebGPU asset prefix helper | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`context.rs`](context.rs) | `WgpuContext` adapter/device discovery, subgroup feature probing & `ozaki_gram_rhs_i32` startup probe | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`dispatch.rs`](dispatch.rs), [`ops.rs`](ops.rs) | Feature-gated `WgpuRunner` command encoders, `<= 128 MiB` storage buffers & `Streamed` chunking | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
