# `dnam-train::control` — Phase 1..3 Control Audits, `CP1`, & 4-Band Wiener Map

Scoped discovery guide for [`src/control.rs`](../control.rs) (parent [`src/AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]` and `zerocopy 0.8` wire packing across unobserved control clamping, `CP1`/`CEXT`/`TSYN` fitting, and dynamic head-gain audits.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`clamp.rs`](clamp.rs), [`clamp/phase1.rs`](clamp/phase1.rs), [`clamp/phase3.rs`](clamp/phase3.rs) | Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]` (`<= 3` cond rules) & Phase-3 `+6 dB` idempotent head-gain clamp | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
| [`stability.rs`](stability.rs), [`stability/rail_and_arena.rs`](stability/rail_and_arena.rs) | Core/crossbar/Kautz/transport stability bounds, stop-gradient corner barrier, `Arena0LiteWire` invariants & Phase-2 `[CLAMP-RAIL]` burst | [`control_space_audit.md`](../../../../docs/specs/training/control_space_audit.md) |
| [`cp1.rs`](cp1.rs), [`cp1/fit.rs`](cp1/fit.rs), [`cp1/subspace.rs`](cp1/subspace.rs), [`cp1/wire.rs`](cp1/wire.rs) | `CP1` Lobatto ANOVA/SVD/Rank-4..8 FM & 31-slot `"CEXT"` dispatch fitting, `"CP1\0"` overflow & `"TSYN"` builders | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md), [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`wiener.rs`](wiener.rs) | 4-band TPT-SVF energy gate (`5%`), phase-decorrelated delay recovery & two-pass 32-slot Wiener Jacobian | [`webgpu_and_varpro_solver.md`](../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
