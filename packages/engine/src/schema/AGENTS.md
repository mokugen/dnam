# `src/schema` — `#![no_std]` + `alloc` JSON Header, Control Schema & Error Types

Implements `#![no_std]` + `alloc` `serde` / `serde_json` parsing and validation (`#[forbid(unsafe_code)]`) for runtime `.dnam` JSON headers, UI power-skew gesture mapping, and compilation into 64-byte [`ControlClampDescriptor`](../../../vm/src/control/descriptor.rs) cache lines. (`.dnam-proj` manifest schema types live in `packages/capture/src/proj/schema.rs`.)

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`header.rs`](header.rs) ([`clamp_compiler.rs`](header/clamp_compiler.rs), [`unknown_keys.rs`](header/unknown_keys.rs)) | `DnamJsonHeader`, `compile_clamp_descriptors` & unknown-key validation | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`topology.rs`](topology.rs), [`io_and_tiers.rs`](io_and_tiers.rs), [`json_wire.rs`](json_wire.rs) | `ModalConvolverConfig`, `LinearReadoutConfig`, `TransportConfig`, `IoConfig`, `TierProfile`, `TensorSliceEntry` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`controls.rs`](controls.rs), [`ui_hints.rs`](ui_hints.rs), [`tempo_subdiv.rs`](tempo_subdiv.rs) | `ControlSchemaDef`, `ControlUiHints`, midpoint power-skew (`norm_to_physical` / `physical_to_norm`), `TempoSubdivision` + `proptest` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md), [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`signatures.rs`](signatures.rs) | `ModelSignatures`, `CreatorSignature`, `MarketplaceReceipt`, `AuthorizedDevice`, `RoyaltySplit` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`error.rs`](error.rs) | `DnamError` (`core::error::Error`) and cold-path `f64` math wrappers | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for `#![no_std]` + `alloc` `serde` rules.
