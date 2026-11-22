# `dnam-train` — SONA v3.0 `f64` SIMD, WebGPU Runners, VarPro Solver & `.dnam-bin` Export

**`dnam-train`** (`packages/train/`) implements double-precision solver linear algebra, 192-bit expansion Gram contraction, 2-D GPU memory planning, bit-accurate CPU and `wgpu` compute execution, perceptual loss evaluation, 3-phase control-space safety auditing, and the 5-pass universal black-box SONA v3.0 system-identification pipeline. It depends solely on [`dnam-kernels`](../kernels/README.md), [`dnam-vm`](../vm/README.md), `zerocopy`, `sha2`, and optional `wgpu`.

---

## Core Capabilities

- **`f64` Multi-ISA `TrainerMathBackend` & `Wide192Accum`:** Extends [`MathBackend`](../vm/README.md) with [`SimdF64Vec2`](src/math/trainer/f64_vec2.rs) across ARM NEON/SVE2/SME, x86 SSE2/SSE4.2/AVX2, WASM SIMD128, and scalar paths, paired with 3-limb Priest/Dekker [`Wide192Accum`](src/wide_gram.rs) `uplo='U'` normal-equation contraction.
- **2-D VRAM Planner & Verified CPU/`wgpu` Runners:** Embeds all twelve Naga-validated `.wgsl` artifacts from `tools/sona-shaders` via `include_str!`, plans `ResidentBatched` / checkpointed `Streamed` scans under the WebGPU binding cap, and runs parity-checked `TrainRunner::auto_detect()` execution across CPU and GPU.
- **`linalg`, Perceptual `losses`, & Control Safety:** Implements `uplo='U'` Cholesky, nested 4-tier Schur VarPro, 1/12-octave `EMR` `MM-Cholesky`, 33-band `LTAS` Newton-KKT, constrained two-loop/whitened L-BFGS, differentiable losses (`bESR`, `SegESR`, `MR-IIR`, `SpecOT`, `IF+GD`, `Gabor`, `LTAS`), Phase 1–3 control clamping, `CP1`/`CEXT`/`TSYN` fitting, and 4-band Wiener recovery.
- **5-Pass Universal Solver & `.dnam-bin` Export:** Executes Pass 1 (`Op-P`, early FIR, Steiglitz–McBride + Kautz OMP, comb ID), Pass 2 (`BestUntilDiminishingReturns` L-BFGS trajectory optimization), Pass 3 (nested 4-tier Schur VarPro + conditioned `ro_pole_48k` sweep + `LTAS` KKT), Pass 4 (`CP1`), and Pass 5 (`zerocopy` `Arena0..3` serialization, LUFS trim calibration within `<= 0.05 LU`, monotone tier certification, `STAT` digest, and 256 B `DNPR` seqlock telemetry).

---

## Quality Gates

```bash
moon check train
cargo check -p dnam-train --no-default-features
cargo check -p dnam-train --no-default-features --target wasm32-unknown-unknown
moon run train:coverage
```
