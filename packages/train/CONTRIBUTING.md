# Contributing to `dnam-train` (`packages/train/`)

This guide defines the safety, crate-boundary, `wgpu`, testing, and coverage rules for **`dnam-train`** ([`README.md`](README.md)). All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Dependency Isolation

Consult [`AGENTS.md`](AGENTS.md) for module routing:

- **`src/lib.rs`:** Enforces `#![deny(unsafe_code)]` and `#![warn(unreachable_pub)]`.
- **Approved `unsafe` Scope (`ref[sdk.rust.safe-rust-and-boundaries]`):** `#[allow(unsafe_code)]` is restricted exclusively to hardware SIMD intrinsics inside `src/math/{aarch64,x86}/**` (with module-level `#[allow(unsafe_code)]` in `src/math.rs`). All other modules (`wide_gram`, `planner`, `runner`, `cpu`, `gpu`, `error`, `wasm`, `tests/`) enforce `#[forbid(unsafe_code)]`.
- **Forbidden Dependencies (`ref[sdk.rust.safe-rust-and-boundaries]`):** Never depend on `dnam-engine`, `dnam-capture`, `dnam-fixtures` (even in `[dev-dependencies]`), `sona-shaders` crate directly (excluded build tool pulling `cargo-gpu`/`naga`), `serde`, `ndarray`, `nalgebra`, `rustfft`, `realfft`, `clap`, or `cpal`.

---

## 2. Numerical, `wgpu`, & Structural Invariants

1. **Zero-ULP IEEE-754 & `Wide192Accum` Determinism:** Hardware `TrainerMathBackend` tile contractions and Bézout lowpass cascades use strict non-FMA arithmetic for bit-exact parity against `ScalarBackend`. Canonical SHA-256 input-gram hashing (`DNAM-GRAM-INPUT-V3`) uses declarative `zerocopy` byteorder structs rather than manual `to_le_bytes` slicing.
2. **Checked-In AOT `.wgsl` Consumption & Startup Probes (`ref[train.pipeline.webgpu-scans-and-ozaki]`):** Embed all twelve `.wgsl` artifacts and `manifest.json` from `../../tools/sona-shaders/wgsl/` via `include_str!`. Always verify SHA-256 digests, workgroup SRAM limits (`15,872 B`), and the sub-millisecond `ozaki_gram_rhs_i32` startup probe before dispatching GPU kernels, falling back transparently to `CpuRunner` if unavailable.
3. **2-D VRAM Planning:** Keep every GPU storage buffer binding within `min(128 MiB, device_max_storage_binding_size)` (for non-streaming profiles), falling back automatically from `ResidentBatched` to checkpointed `Streamed` execution.
4. **Modularity & Coverage (`ref[sdk.rust.modular-layout-and-proptest]`):** Zero `mod.rs` files, zero `coverage(off)`, zero `macro_rules!` in `src/math/**`, every `.rs` file below 900 lines (targeting 250–650), and inline `#[cfg(test)] mod tests` with `proptest!` suites co-located at the bottom of source files.

---

## 3. Quality Gates

```bash
moon check train
cargo check -p dnam-train --no-default-features
cargo check -p dnam-train --no-default-features --target wasm32-unknown-unknown
moon run train:coverage
moon check tests
```

- [ ] `moon check train` (`format-check`, `lint`, `typecheck`, `build`, `test`) and `--no-default-features` / `wasm32` builds pass cleanly.
- [ ] `moon run train:coverage` achieves at least 95% line coverage across `packages/train/src/`.
- [ ] `moon check tests` passes all boundary, `unsafe`, `zerocopy`, and hygiene guardrails.
