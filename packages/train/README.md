# `dnam-train` — SONA v3.0 `f64` SIMD, `Wide192Accum`, & CPU / WebGPU Training Runners

**`dnam-train`** (`packages/train/`) implements double-precision solver linear algebra, 192-bit expansion Gram contraction, 2-D GPU memory planning, and bit-accurate CPU and `wgpu` compute execution for **Dynamic Neural Audio Modeler (DNAM)** SONA v3.0 models. It depends only on [`dnam-kernels`](../kernels/README.md), [`dnam-vm`](../vm/README.md), `zerocopy`, `sha2`, and optional `wgpu`.

---

## Core Capabilities

- **`f64` Multi-ISA `TrainerMathBackend`:** Extends [`MathBackend`](../vm/README.md) with [`SimdF64Vec2`](src/math/trainer/f64_vec2.rs) and [`TrainerMathBackend`](src/math/trainer.rs) across ARM NEON, SVE2, SME, x86 SSE2, SSE4.2, AVX2/FMA, WASM SIMD128, and portable scalar paths while preserving strict non-FMA IEEE-754 parity on tile contractions and Bézout cascades.
- **`Wide192Accum` Upper-Triangular Gram Contraction:** Combines `ScalarBackend`/`DefaultMathBackend` L1D register-tiled micro-batches with 3-limb Priest/Dekker expansion (`Wide192Accum`) and Boldo–Melquiond round-to-odd collapse, eliminating catastrophic cancellation and canonicalizing `-0.0` to `+0.0`.
- **2-D VRAM Planner (`planner`):** Plans `ResidentBatched` or checkpointed `Streamed` core scans across `BrowserWebGpu`, `CpuWasmFallback`, `DesktopCapsule`, and `StreamingHost` budgets while keeping every storage binding within the WebGPU binding limit.
- **Deterministic CPU & Verified `wgpu` Compute Runners:** Embeds all twelve Naga-validated `.wgsl` artifacts from `tools/sona-shaders` via `include_str!`, verifies SHA-256 digests and SRAM limits, executes sub-millisecond `ozaki_gram_rhs_i32` startup probes, selects subgroup vs portable shaders automatically, and provides `TrainRunner::auto_detect()` parity across `prescreen_population`, `fused_core_scan_256`, `adjoint_scan_256`, `gabor_bezout_grad`, `ozaki_gram_rhs_i32`, and `inner_opt_step`.

---

## Quality Gates

```bash
moon check train
cargo check -p dnam-train --no-default-features
cargo check -p dnam-train --no-default-features --target wasm32-unknown-unknown
moon run train:coverage
```
