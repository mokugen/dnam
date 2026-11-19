# Contributing to `dnam-kernels` (`packages/kernels/`)

This guide defines the safety, numerical, `#![no_std]` / `SPIR-V`, testing, and coverage requirements for **`dnam-kernels`** ([`README.md`](README.md)). All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Module Routing

Consult [`AGENTS.md`](AGENTS.md) for the file routing table:

- **`src/lib.rs`:** Crate root enforcing `#![cfg_attr(not(test), no_std)]` and `#![forbid(unsafe_code)]`.
- **`src/simd.rs` & `src/simd/{const_eval,fp16,gpu_vec4,stereo8}.rs`:** `KernelVec4`, `GpuVec4`, `StereoVec8`, `InterleavedStereo8<V>`, compile-time `const fn` seeds/poles, branchless `fp16` codecs, and subnormal/NaN bit flushing.
- **`src/ozaki_i32.rs`:** `pow2_ceil_fp32` and fast-math-immune `(2^8, 2^16, 2^24)` `3 × 8-bit` `i32` Ozaki slicing primitives.
- **`src/sona_core.rs` & `src/sona_core/{tpt_svf,paralru/{scans,adjoint},poly_adaa/wavefolder}.rs`:** Passive `2×2` ZDF `TPT-SVF`, contractive `1D` `ParaLRU` exact `0-ULP` forward tiles + `O(log2 N)` linear reverse-adjoint suffix scans, and pure-`f32` `PolyADAA1` + `C²` wavefolder.
- **`src/modal_and_transport.rs` & `src/modal_and_transport/{kautz/rotor_and_phasor,op_t,op_p}.rs`:** Contractive `f32` rotor, `4`-lane `Q0.32` quadrature phasor bank, 4-lane/stereo `Kautz` resonators (`0.9995` ceiling) & associative complex modal scans, Catmull-Rom `f32`/`fp16` readers, `clamp_transport_delay_f32`, `M=2`/`M=3` pitch/barberpole windowing, partitioned Schroeder diffusers, `0-ULP` freeze ring writes, coercive loop limiter, and `5 Hz` upstream DC blocker + `H1` Hilbert + `4`-feature `T1` `Op-P` kernels.
- **`tests/sona_kernel_scans_and_parity.rs`:** Cross-module associative scan, Ozaki, and stereo parity integration tests.

---

## 2. Mandatory Engineering & `SPIR-V` / Real-Time Invariants

1. **Zero `unsafe` & Zero External Dependencies (`ref[repo.code.type-safety-and-imports]`):**
   - `dnam-kernels` must compile with `#![cfg_attr(not(test), no_std)]`, `#![forbid(unsafe_code)]`, zero external dependencies, and zero `alloc` / `Vec` usage in production code.
2. **Strictly `32-bit` Non-Test Arithmetic (`ref[repo.code.shared-nostd-kernels]`):**
   - Non-test code (`src/**/*.rs` outside `#[cfg(test)]`) may use **only** `f32`, `i32`, `u32`, `u16`, `u8`, `bool`, `usize`, and fixed arrays/slices—never `f64`, `i64`, `u64`, `i128`, or `u128`.
   - Zero runtime `FDIV` (`/`), `FSQRT`, `IDIV`/`UDIV`/`SDIV`, `libm` calls (`.sin()`, `.cos()`, `.tan()`, `.sqrt()`, `.exp()`, `.ln()`, `.powf()`), or `.rem_euclid()` in non-test code; use `KernelVec4`/`GpuVec4` `NR2` reciprocals and `rsqrt_pos_nr2()`, reciprocal multiplication, and power-of-two `& mask` wrapping.
3. **Explicit Implementations, `Debug` & File Ceilings (`ref[repo.code.explicit-simd-and-coverage]`, `ref[repo.code.modular-decomposition-and-tests]`):**
   - Zero `macro_rules!`, zero `#[cfg_attr(coverage, coverage(off))]`, and `#[derive(Debug)]` on every `pub struct` and `pub enum`.
   - Keep every `.rs` source file strictly `< 900` lines (`~250–650` lines), with zero `mod.rs` files and zero `\`-newline code golf.
4. **Universal Black-Box & Compile-Time Constant Rules (`ref[repo.code.universal-black-box]`, `ref[repo.code.constant-eval-and-derivations]`):**
   - Implement general mathematical basis operators only (zero device/brand circuit heuristics or bypass flags); derive polynomial/filter constants at compile time or document their derivation in `docs/specs/`.

---

## 3. Co-Located Tests (`proptest!`), Coverage (≥ 95%) & Quality Gates

- **4-Tier Testing Discipline:** Co-locate leaf unit tests and `proptest!` suites inside inline `#[cfg(test)] mod tests { ... }` at the bottom of each submodule (with `verify[...]` citations matching canonical rules in `docs/specs/`), keep facade inter-unit tests in parent modules, and keep cross-module parity/scan suites in `tests/sona_kernel_scans_and_parity.rs` (`never` read `docs/` files in Rust tests).

```bash
moon check kernels
moon run kernels:coverage
moon check tests
```

---

## 4. Pre-Commit Checklist

- [ ] Every modified `.rs` module passes `moon check kernels` (`fmt`, `clippy -D warnings`, `check`, `build --release`, `test`) with zero `f64`/`i64`/`u64`/`FDIV`/`FSQRT`/`libm` in non-test code.
- [ ] `moon run kernels:coverage` achieves `>= 95%` instrumented line coverage across `packages/kernels/src/`.
- [ ] `moon check tests` passes all `AGENTS.md` byte budgets (`< 2,500 B`), `verify[...]` / `ref[...]` checks, and Markdown link integrity checks.
