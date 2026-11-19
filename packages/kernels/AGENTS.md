# `packages/kernels` (`dnam-kernels`) — Shared Kernel Map

`#![no_std]`, `#![forbid(unsafe_code)]`, zero-dependency `32-bit` DSP, scan, and Ozaki kernels shared by [`dnam-engine`](../engine/AGENTS.md) and `tools/sona-shaders`. See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`README.md`](README.md); run `moon check kernels` & `moon run kernels:coverage`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | Crate root (`#![no_std]`, `#![forbid(unsafe_code)]`) | [`project_and_documentation_quality.md`](../../docs/specs/architecture/project_and_documentation_quality.md) |
| [`simd.rs`](src/simd.rs), [`const_eval.rs`](src/simd/const_eval.rs), [`fp16.rs`](src/simd/fp16.rs), [`gpu_vec4.rs`](src/simd/gpu_vec4.rs), [`stereo8.rs`](src/simd/stereo8.rs) | `KernelVec4`, `GpuVec4`, `StereoVec8`, `const fn`, `fp16` | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
| [`src/ozaki_i32.rs`](src/ozaki_i32.rs) | `pow2_ceil_fp32` & `3×8-bit` `i32` Ozaki slices | [`four_stage_pipeline.md`](../../docs/specs/training/four_stage_pipeline.md) |
| [`sona_core.rs`](src/sona_core.rs), [`tpt_svf.rs`](src/sona_core/tpt_svf.rs), [`paralru.rs`](src/sona_core/paralru.rs) ([`scans.rs`](src/sona_core/paralru/scans.rs), [`adjoint.rs`](src/sona_core/paralru/adjoint.rs)), [`poly_adaa.rs`](src/sona_core/poly_adaa.rs), [`wavefolder.rs`](src/sona_core/poly_adaa/wavefolder.rs) | `TPT-SVF` scan, `1D` `ParaLRU` exact fwd & linear adjoint, `PolyADAA1` & `C²` | [`op_n_commuted_cascade.md`](../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`modal_and_transport.rs`](src/modal_and_transport.rs), [`kautz.rs`](src/modal_and_transport/kautz.rs), [`rotor_and_phasor.rs`](src/modal_and_transport/kautz/rotor_and_phasor.rs), [`op_t.rs`](src/modal_and_transport/op_t.rs), [`op_p.rs`](src/modal_and_transport/op_p.rs) | Rotor, `Q0.32`, `Kautz`, `Op-T` clamp/readers, `5 Hz` `T1` + `4`-ch `Op-P` | [`op_m_modal_kautz.md`](../../docs/specs/dsp/op_m_modal_kautz.md), [`op_t_transport.md`](../../docs/specs/dsp/op_t_transport.md) |
| [`tests/sona_kernel_scans_and_parity.rs`](tests/sona_kernel_scans_and_parity.rs) | Scans, Ozaki, stereo parity & `proptest!` | [`sona_architecture.md`](../../docs/specs/architecture/sona_architecture.md) |
