# `dnam-kernels::simd` — Portable `32-bit` SIMD & `const_eval` Map

Submodule map for [`simd.rs`](../simd.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, `0` `f64`/`i64`/`u64`/`FDIV`/`FSQRT`/`libm`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`const_eval.rs`](const_eval.rs), [`tiers.rs`](const_eval/tiers.rs), [`fixed_point.rs`](const_eval/fixed_point.rs) | Pure-`u32` `Q60` `const fn`, `HB15` taps & v3/v4 `SonaTierSpec` | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`log_vca.rs`](log_vca.rs) | `fast_log2_*`, `exp2_clamped5_f32` & `UnifiedLogKneeVca` (`[-30..+24 dB]`) | [`router_and_crossbar_topology.md`](../../../../docs/specs/dsp/router_and_crossbar_topology.md) |
| [`op_constants.rs`](op_constants.rs) | `exp2_clamped16_f32`, `os2x_midpoint_*`, `TANK_MAX_*`, `HERMITE5_SIN_*` | [`op_t_transport.md`](../../../../docs/specs/dsp/op_t_transport.md) |
| [`fp16.rs`](fp16.rs) | Branchless `f32` ↔ `fp16` (`u16`/`u32`/`fp16x2`) codecs | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
| [`gpu_vec4.rs`](gpu_vec4.rs), [`stereo8.rs`](stereo8.rs) | `GpuVec4` (`NR2`/`NR3`), `StereoVec8`, `InterleavedStereo8` | [`sona_architecture.md`](../../../../docs/specs/architecture/sona_architecture.md) |
