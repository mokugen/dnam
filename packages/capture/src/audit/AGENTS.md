# `dnam-capture` Audit Submodules (`packages/capture/src/audit/`)

Routing map for manifest-level Phase-1 clamp domains (`audit`). Parent facade (`.dnam-bin` 3-Phase stability, rail burst, and `+6 dB` readout head-gain sweeps delegated to `dnam-train`): [`../audit.rs`](../audit.rs).

| Module | Responsibility | Canonical Specs |
| :--- | :--- | :--- |
| [`manifest.rs`](manifest.rs) | Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-CORNER]` (`StructuredConditionalBound`) manifest recomputation | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
