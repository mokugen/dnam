# `packages/train/src` — Rust Implementation Map (`dnam-train`)

Parent discovery map: [`../AGENTS.md`](../AGENTS.md). Strictly enforces `#![deny(unsafe_code)]` at crate root, restricting `unsafe` intrinsics solely to `math/aarch64/` and `math/x86/` with `#[forbid(unsafe_code)]` everywhere else.

| Subsystem | Files | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| **Root & Errors** | [`lib.rs`](lib.rs), [`error.rs`](error.rs) | [`rust_sdk.md`](../../../docs/specs/sdks/rust_sdk.md) |
| **`f64` SIMD (`TrainerMathBackend`)** | [`math.rs`](math.rs), [`trainer.rs`](math/trainer.rs), [`f64_vec2.rs`](math/trainer/f64_vec2.rs), [`aarch64.rs`](math/aarch64.rs), [`x86.rs`](math/x86.rs), [`wasm.rs`](math/wasm.rs) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **`Wide192Accum` & `uplo='U'` Gram** | [`wide_gram.rs`](wide_gram.rs), [`tiled.rs`](wide_gram/tiled.rs), [`cross_only.rs`](wide_gram/cross_only.rs) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **2-D VRAM Planner** | [`planner.rs`](planner.rs) (`<= 128 MiB` binding cap, `ResidentBatched` / `Streamed`) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **Unified Runner** | [`runner.rs`](runner.rs), [`types.rs`](runner/types.rs) (`TrainRunner::auto_detect`, I/O bundles) | [`rust_sdk.md`](../../../docs/specs/sdks/rust_sdk.md) |
| **CPU Reference Runner** | [`cpu.rs`](cpu.rs), [`core_and_adjoint.rs`](cpu/core_and_adjoint.rs), [`adjoint_math.rs`](cpu/core_and_adjoint/adjoint_math.rs), [`adaa_derivs.rs`](cpu/core_and_adjoint/adjoint_math/adaa_derivs.rs), [`gabor_and_prescreen.rs`](cpu/gabor_and_prescreen.rs), [`ozaki_and_opt.rs`](cpu/ozaki_and_opt.rs) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| **`wgpu` Compute & WGSL** | [`gpu.rs`](gpu.rs), [`abi.rs`](gpu/abi.rs), [`shaders.rs`](gpu/shaders.rs), [`context.rs`](gpu/context.rs), [`dispatch.rs`](gpu/dispatch.rs), [`ops.rs`](gpu/ops.rs) | [`webgpu_and_varpro_solver.md`](../../../docs/specs/training/webgpu_and_varpro_solver.md) |
