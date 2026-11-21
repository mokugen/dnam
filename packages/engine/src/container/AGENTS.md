# `src/container` — `.dnam` Layer 1 Binary Framing & `.dnam-bin` Bridge

Implements the `#![no_std]` + `alloc` runtime `.dnam` parser, builder, `CRC32C`, `DMU1`/`DMUE` mutable suffix framing, and zero-copy Layer 0 `.dnam-bin` extraction (`extract_dnam_bin`, `#[forbid(unsafe_code)]`). Project container (`.dnam-proj` / `DNPJ`) types live in `packages/capture/src/proj/`, and ZOH sample-rate rescaling lives in `packages/vm/src/graph/zoh.rs`.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`preamble.rs`](preamble.rs), [`wire.rs`](wire.rs) ([`structs.rs`](wire/structs.rs), [`layout.rs`](wire/layout.rs), [`directory.rs`](wire/layout/directory.rs), [`mutable_frame.rs`](wire/layout/mutable_frame.rs), [`jcs.rs`](wire/jcs.rs), [`slices_and_cp1.rs`](wire/slices_and_cp1.rs), [`delay_and_convolver.rs`](wire/slices_and_cp1/delay_and_convolver.rs), [`conditional.rs`](wire/conditional.rs)) | `zerocopy` v3.0 wire structs, 64B preamble, 128B directory (`dbin_offset`/`dbin_prefix_bytes`), JCS header, `DMU1`/`DMUE`, `CP1`, `delay_histories` + `proptest` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`builder.rs`](builder.rs), [`view.rs`](view.rs) ([`auth.rs`](view/auth.rs), [`dbin.rs`](view/dbin.rs), [`sealed.rs`](view/sealed.rs)), [`strip.rs`](strip.rs) | `DnamContainerView`, `extract_dnam_bin`, `DAU1`/policy verification, `build_dnam_container*`, `DMU1` in-place updates, stripping + `proptest` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for container verification guidance.
