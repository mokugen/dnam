# `src/stages/weights` — Arena 0..3 Directory & `SONA` `v3.0` Packers

Implements the `64 B`-aligned declarative `zerocopy` `Arena 0..3` wire structs (`weights.rs`) and `SONA` Matryoshka binary codecs (`#[forbid(unsafe_code)]`).

| Module | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../weights.rs`](../weights.rs) | `Arena0LiteWire..Arena3UltraDeltaWire` (`4,608`..`30,720 B`), flags & directory offsets | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`pack_sona.rs`](pack_sona.rs) | Top-level `SONA` `pack_sona_matryoshka_to_arena` & `unpack_*` entrypoints | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`pack_sona/readout_ctrl.rs`](pack_sona/readout_ctrl.rs) | Stage III Early FIR + prefix overrides, Kautz poles/residues, readouts & `ControlBusParams` | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for arena invariants and tier boundaries.
