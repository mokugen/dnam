# `packages/kernels` (`dnam-kernels`) — Shared Kernel Map

`#![no_std]`, `#![forbid(unsafe_code)]`, zero-dependency `32-bit` DSP, scan, and Ozaki kernels shared by [`dnam-vm`](../vm/AGENTS.md), [`dnam-engine`](../engine/AGENTS.md), and `tools/sona-shaders`. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`README.md`](README.md); run `moon check kernels` & `moon run kernels:coverage`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | Crate root (`#![no_std]`, `#![forbid(unsafe_code)]`) | [`project_and_documentation_quality.md`](../../docs/specs/architecture/project_and_documentation_quality.md) |
| [`src/simd.rs`](src/simd.rs), [`simd/`](src/simd/AGENTS.md) | `KernelVec4`, `GpuVec4`, `StereoVec8`, `const fn`, `exp2_clamped16`, `fp16` | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
| [`src/ozaki_i32.rs`](src/ozaki_i32.rs) | `pow2_ceil_fp32` & `3×8-bit` `i32` Ozaki slices | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`src/sona_core.rs`](src/sona_core.rs), [`sona_core/`](src/sona_core/AGENTS.md), [`core_quad.rs`](src/sona_core/core_quad.rs), [`op_n_pre.rs`](src/sona_core/op_n_pre.rs) | `CoreQuad`, `28 B` `Op-N[Pre]`, `TPT-SVF`, `PolyADAA1`/`C²`, `ParaLRU` + [`duhem_scans.rs`](src/sona_core/paralru/duhem_scans.rs) / [`duhem_adjoint.rs`](src/sona_core/paralru/duhem_adjoint.rs) | [`op_n_paralru_and_polyadaa.md`](../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`src/modal_and_transport.rs`](src/modal_and_transport.rs), [`modal_and_transport/`](src/modal_and_transport/AGENTS.md), [`dc_and_tracker.rs`](src/modal_and_transport/op_p/dc_and_tracker.rs), [`op_t/`](src/modal_and_transport/op_t/) | Rotor, `Kautz`, `28 B` 2-stage `T1SubHarmonicKernel`, `Op-T` `sigma_sat` cubic limiter, continuous `freeze_mix`, `QUAD_FLAG_*` (bit `0` reserved `0`) | [`op_m_modal_kautz.md`](../../docs/specs/dsp/op_m_modal_kautz.md), [`op_t_transport.md`](../../docs/specs/dsp/op_t_transport.md) |
| [`tests/sona_kernel_scans_and_parity.rs`](tests/sona_kernel_scans_and_parity.rs) | Scans, Ozaki, stereo parity & `proptest!` | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
