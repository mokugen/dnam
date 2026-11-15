# `dnam-capture` Audit Submodules (`packages/capture/src/audit/`)

Routing map for manifest-level Phase-1 clamp domains and arena-level 3-phase safety sweeps (`audit`). Parent facade: [`../audit.rs`](../audit.rs).

| Module | Responsibility | Canonical Specs |
| :--- | :--- | :--- |
| [`manifest.rs`](manifest.rs) | Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-CORNER]` manifest clamp recomputation | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
| [`arena.rs`](arena.rs) | 3-phase arena stability, `Stage IV EXT`, `PolyADAA` rail lockup, and `+6.0 dB` head-gain audit | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
