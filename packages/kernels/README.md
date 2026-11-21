# `dnam-kernels` — Shared `#![no_std]` 32-bit SONA DSP & Ozaki Kernel Crate

**`dnam-kernels`** (`packages/kernels/`) is the single-source `#![no_std]`, `#![forbid(unsafe_code)]`, zero-dependency 32-bit arithmetic core for **Dynamic Neural Audio Modeler (DNAM)** SONA v3.0. Every mathematical primitive in this crate compiles identically for multi-ISA CPU SIMD execution inside [`dnam-engine`](../engine/README.md) (Rust 1.100.0) and offline AOT `cargo-gpu` `spirv-unknown-naga-wgsl` `.wgsl` compute shader emission (`tools/sona-shaders`, [`ADR-0022`](../../docs/adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

## Core Design Invariants

- **`#![no_std]`, `#![forbid(unsafe_code)]`, Zero External Crates:** Depends on zero external runtime crates (`proptest` is `[dev-dependencies]` only) and contains zero `unsafe` blocks or overrides.
- **Strictly 32-bit SPIR-V / WGSL-Clean Runtime Arithmetic:** Non-test code uses only `f32`, `i32`, `u32`, `u16`, `u8`, `bool`, `usize`, fixed arrays, and slices—zero `f64`, `i64`/`u64`/`i128`/`u128`, `alloc`, `Vec`, FDIV, FSQRT, IDIV/UDIV, `libm`, `rem_euclid`, `coverage(off)`, or `macro_rules!` boilerplate.
- **Compile-Time Constant Derivations:** Seeds, polynomial coefficients, filter poles, and IEEE-754 boundary bit masks are derived via `const fn` evaluators and checked with compile-time `const _: () = assert!(...)` invariants ([`src/simd/const_eval.rs`](src/simd/const_eval.rs)).

---

## Module Architecture

| Module | Scope & Key Primitives |
| :--- | :--- |
| **[`src/lib.rs`](src/lib.rs)** | Crate root enforcing `#![cfg_attr(not(test), no_std)]` and `#![forbid(unsafe_code)]`. |
| **[`src/simd.rs`](src/simd.rs)** | Core SIMD & conversion traits/helpers: `KernelVec4`, `StereoVec8`, `InterleavedStereo8<V>`, IEEE-754 `f32` to/from fp16 (`u16` / `u32` / packed fp16x2), `flush_subnormal_f32_bits` / `scrub_and_flush_f32_bits`, and submodules [`const_eval.rs`](src/simd/const_eval.rs), [`op_constants.rs`](src/simd/op_constants.rs), [`fp16.rs`](src/simd/fp16.rs), [`gpu_vec4.rs`](src/simd/gpu_vec4.rs) (`GpuVec4`), and [`stereo8.rs`](src/simd/stereo8.rs). |
| **[`src/ozaki_i32.rs`](src/ozaki_i32.rs)** | Fast-math-immune 3 × 8-bit integer mantissa Ozaki slicing (`pow2_ceil_fp32`, `i32` dot contractions) for WebGPU Gram/RHS accumulation. |
| **[`src/sona_core.rs`](src/sona_core.rs)** | 4-Block Matryoshka SONA state-space core primitives: [`core_quad.rs`](src/sona_core/core_quad.rs) (`CoreQuadParamsStereo4`, `couple_ms_input_stereo8`, `step_core_quad_stereo8`), [`tpt_svf.rs`](src/sona_core/tpt_svf.rs) (passive 2×2 ZDF TPT-SVF step & associative scan compose), [`paralru.rs`](src/sona_core/paralru.rs), [`scans.rs`](src/sona_core/paralru/scans.rs), [`adjoint.rs`](src/sona_core/paralru/adjoint.rs), [`duhem_scans.rs`](src/sona_core/paralru/duhem_scans.rs), [`duhem_adjoint.rs`](src/sona_core/paralru/duhem_adjoint.rs) (contractive & Selective-Duhem ParaLRU scans), [`op_n_pre.rs`](src/sona_core/op_n_pre.rs) (28 B smooth `Op-N[Pre]` ZOH + nearest-step bitcrusher), plus [`poly_adaa.rs`](src/sona_core/poly_adaa.rs) and branchless [`wavefolder.rs`](src/sona_core/poly_adaa/wavefolder.rs). |
| **[`src/modal_and_transport.rs`](src/modal_and_transport.rs)** | Crossbar, Convolver & Transport modal/transport primitives: [`kautz.rs`](src/modal_and_transport/kautz.rs) & [`rotor_and_phasor.rs`](src/modal_and_transport/kautz/rotor_and_phasor.rs) (pure-`f32` contractive unit rotor, `PhasorQ32Bank4`, `KautzQuadState`, `eval_kautz_sweep_contraction_f32` 0.9995 ceiling, `ComplexModalScanElem`), [`op_t.rs`](src/modal_and_transport/op_t.rs) ([`interp_and_window.rs`](src/modal_and_transport/op_t/interp_and_window.rs), [`quad_ortho.rs`](src/modal_and_transport/op_t/quad_ortho.rs) with unified `sigma_sat` cubic limiter & continuous `freeze_mix`), and [`op_p.rs`](src/modal_and_transport/op_p.rs) ([`hilbert_stereo.rs`](src/modal_and_transport/op_p/hilbert_stereo.rs), [`anchors.rs`](src/modal_and_transport/op_p/hilbert_stereo/anchors.rs), [`dc_and_tracker.rs`](src/modal_and_transport/op_p/dc_and_tracker.rs) with 28 B 2-stage `T1SubHarmonicKernel`, [`carrier_3basis.rs`](src/modal_and_transport/op_p/carrier_3basis.rs)). |
| **[`tests/sona_kernel_scans_and_parity.rs`](tests/sona_kernel_scans_and_parity.rs)** | Cross-module associative scan, Ozaki exactness, stereo parity, and `proptest!` verification suites. |

---

## Quickstart & Quality Gates

Run all tasks from the monorepo root via `moon`:

```bash
# Format check, clippy (-D warnings), typecheck, release build, and unit/proptest/integration tests
moon check kernels

# Individual tasks
moon run kernels:format-check
moon run kernels:lint
moon run kernels:typecheck
moon run kernels:build
moon run kernels:test
moon run kernels:coverage      # >= 95% instrumented line coverage
```

---

## License

Licensed under the [DNAM Source-Available & Branding License 1.0.0 (`../../LICENSE`)](../../LICENSE).
