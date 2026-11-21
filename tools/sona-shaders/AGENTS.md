# `tools/sona-shaders` — Offline AOT Shader Generator & ABI Map

Isolated standalone Rust tool workspace compiling [`dnam-kernels`](../../packages/kernels/AGENTS.md) compute shaders offline via `cargo-gpu-install` + `spirv-builder` (`spirv-unknown-naga-wgsl`) into `naga`-validated `.wgsl` files (`ref[train.pipeline.webgpu-scans-and-ozaki]`). Run `moon check sona-shaders` and `moon run sona-shaders:check-drift`.

| Module / Path | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`src/abi.rs`](src/abi.rs) | 32 B `zerocopy` uniforms, SRAM ceilings, `24 × Vec4` `CoreQuad` parameter layout | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/manifest.rs`](src/manifest.rs) | 6-kernel registry, expected binding specs, `ShaderManifest`, `prepare_for_browser_webgpu` | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/validate.rs`](src/validate.rs) | `naga` parse/validate, `<= 15,872 B` SRAM check, zero 64-bit check, ABI enforcement | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/compiler.rs`](src/compiler.rs) | Portable + `subgroups` AOT build, SHA-256 manifest emission, `check_drift()` | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
| [`src/bin/sona_shaders_gen.rs`](src/bin/sona_shaders_gen.rs) | CLI (`build` / `check-drift`) writing `wgsl/` and `wgsl/subgroups/` | [`rust_sdk.md`](../../docs/specs/sdks/rust_sdk.md) |
| [`crates/sona-shaders-gpu/`](crates/sona-shaders-gpu/src/lib.rs) | `#![no_std]` SPIR-V compute kernels sharing `dnam-kernels` primitives | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`tests/naga_validation_and_abi.rs`](tests/naga_validation_and_abi.rs) | ABI layout, SRAM budget, manifest SHA-256, and Naga validation tests | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
