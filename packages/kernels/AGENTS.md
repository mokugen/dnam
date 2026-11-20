# `packages/kernels` (`dnam-kernels`) — Shared Kernel Map

`#![no_std]`, `#![forbid(unsafe_code)]`, zero-dependency `32-bit` DSP, scan, and Ozaki kernels shared by [`dnam-engine`](../engine/AGENTS.md) and `tools/sona-shaders`. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`README.md`](README.md); run `moon check kernels` & `moon run kernels:coverage`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | Crate root (`#![no_std]`, `#![forbid(unsafe_code)]`) | [`project_and_documentation_quality.md`](../../docs/specs/architecture/project_and_documentation_quality.md) |
| [`simd.rs`](src/simd.rs), [`const_eval.rs`](src/simd/const_eval.rs), [`op_constants.rs`](src/simd/op_constants.rs), [`fp16.rs`](src/simd/fp16.rs), [`gpu_vec4.rs`](src/simd/gpu_vec4.rs), [`stereo8.rs`](src/simd/stereo8.rs) | `KernelVec4`, `GpuVec4`, `StereoVec8`, `const fn`, `exp2_clamped16`, `fp16` | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
| [`src/ozaki_i32.rs`](src/ozaki_i32.rs) | `pow2_ceil_fp32` & `3×8-bit` `i32` Ozaki slices | [`webgpu_and_varpro_solver.md`](../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`sona_core.rs`](src/sona_core.rs), [`tpt_svf.rs`](src/sona_core/tpt_svf.rs), [`paralru.rs`](src/sona_core/paralru.rs) ([`scans.rs`](src/sona_core/paralru/scans.rs), [`adjoint.rs`](src/sona_core/paralru/adjoint.rs)), [`poly_adaa.rs`](src/sona_core/poly_adaa.rs), [`wavefolder.rs`](src/sona_core/poly_adaa/wavefolder.rs), [`op_n_pre.rs`](src/sona_core/op_n_pre.rs) | `TPT-SVF` scan, `1D` `ParaLRU` fwd/adj, `PolyADAA1` & `C²`, `Op-N[Pre]` | [`op_n_paralru_and_polyadaa.md`](../../docs/specs/dsp/op_n_paralru_and_polyadaa.md) |
| [`modal_and_transport.rs`](src/modal_and_transport.rs), [`modal_and_transport/`](src/modal_and_transport/AGENTS.md) | Rotor, `Q0.32`, `Kautz`, ortho `Op-T`/`WE`, stereo `H1`, `T1`, 3-basis carrier | [`op_m_modal_kautz.md`](../../docs/specs/dsp/op_m_modal_kautz.md), [`op_t_transport.md`](../../docs/specs/dsp/op_t_transport.md) |
| [`tests/sona_kernel_scans_and_parity.rs`](tests/sona_kernel_scans_and_parity.rs) | Scans, Ozaki, stereo parity & `proptest!` | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
