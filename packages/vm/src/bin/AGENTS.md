# `dnam-vm::bin` — Zero-Copy `.dnam-bin` Wire, Validator & Builder Map

Scoped discovery guide for [`src/bin.rs`](../bin.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Strictly `#![forbid(unsafe_code)]`, `--no-default-features` `!alloc`-capable, declarative `zerocopy` structs with `64 B`-aligned tier prefixes.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`header.rs`](header.rs) | `64 B` `"DNBN"` & `"DBRM"` headers (`ro_pole_48k`, `TSYN`/`CEXT`/`CP1` wire structs) | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`arena_wire.rs`](arena_wire.rs) | `Arena 0..3` (`4,608..30,720 B`), `CoreQuadWire`, `CrossbarWire`, `TransportQuadWire`, `LinearReadout*Wire` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`view.rs`](view.rs), [`validate.rs`](view/validate.rs), [`arena.rs`](view/validate/arena.rs) | `DnamBinView` (`< 50 ns` `bind_prevalidated` & CRC-32C `parse`), `REQ-5.6` `strip_dnam_bin_to_tier` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`builder.rs`](builder.rs) | Deterministic `!alloc` `encode_dnam_bin_into` & owned `encode_dnam_bin` | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
