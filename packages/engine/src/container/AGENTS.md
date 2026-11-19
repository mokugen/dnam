# `src/container` — `.dnam` Runtime Binary Framing & Continuous-Time ZOH

Implements the `#![no_std]` + `alloc` runtime `.dnam` parser, builder, `CRC32C`, `DMU1`/`DMUE` mutable suffix framing, and Continuous-Time ZOH sample-rate rescaling (`#[forbid(unsafe_code)]`). Project container (`.dnam-proj` / `DNPJ`) types live in `packages/capture/src/proj/`.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`preamble.rs`](preamble.rs), [`wire.rs`](wire.rs) ([`structs.rs`](wire/structs.rs), [`layout.rs`](wire/layout.rs), [`mutable_frame.rs`](wire/layout/mutable_frame.rs), [`jcs.rs`](wire/jcs.rs), [`slices_and_cp1.rs`](wire/slices_and_cp1.rs), [`delay_and_stage3.rs`](wire/slices_and_cp1/delay_and_stage3.rs), [`conditional.rs`](wire/conditional.rs)) | `zerocopy` `v3.0` wire structs, 64B preamble, 128B directory, JCS header, `DMU1`/`DMUE`, `CP1`, `delay_histories` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`builder.rs`](builder.rs), [`view.rs`](view.rs) ([`auth.rs`](view/auth.rs), [`sealed.rs`](view/sealed.rs)), [`strip.rs`](strip.rs) | `DnamContainerView`, `DAU1`/policy verification, `build_dnam_container`, `DMU1` in-place updates, stripping | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`zoh.rs`](zoh.rs) | `ZohRescaledParams` (44.1–192 kHz poles, delays, LFO invariance, FIR resampling) + `proptest` | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for container verification guidance.
