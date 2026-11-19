# `src/stages/weights` — Arena 0..3 Directory & `v3` Packers

Implements the 64B-aligned `Arena 0..3` byte directory (`weights.rs`) and Matryoshka binary codecs (`#[forbid(unsafe_code)]`).

| Module | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../weights.rs`](../weights.rs) | `Arena 0..3` byte offsets (`4,608`..`28,672 B`), flags & directory view | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`pack_v3.rs`](pack_v3.rs) | Top-level `v3` `pack_v3_matryoshka_to_arena` & `unpack_*` entrypoints | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`pack_v3/gru.rs`](pack_v3/gru.rs) | Stage I `SVF`, Stage II `GRU` tiles, `Stage IV` tank & `rate_1x_cal` codec | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`pack_v3/readout.rs`](pack_v3/readout.rs) | Stage III Early FIR, Kautz poles/residues, readouts & `ControlBusParams` | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for arena invariants and tier boundaries.
