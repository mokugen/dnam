# `dnam-train::solver::pass5_emit` — Tier Packing, Certification & Monotonic Repair

Scoped guide for [`../pass5_emit.rs`](../pass5_emit.rs) (parent [`../AGENTS.md`](../AGENTS.md)). Enforces `#[forbid(unsafe_code)]`, `Arena0..3` / `PrefixOverride` packing, `<= 0.05 LU` trim calibration, stereo 4-metric certification, and non-destructive adjacent-tier monotone repair.

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../pass5_emit.rs`](../pass5_emit.rs) | `run_pass5_emit_and_certify` orchestration (`LUFS`, rail burst, `4`-tier report) | [`webgpu_and_varpro_solver.md`](../../../../../docs/specs/training/webgpu_and_varpro_solver.md) |
| [`pack.rs`](pack.rs), [`blocks.rs`](pack/blocks.rs), [`readouts.rs`](pack/readouts.rs) | Encode `SonaModelF64` (`Arena0..3` + `PrefixOverride`) into validated `.dnam-bin` | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`unpack.rs`](unpack.rs) | Safe `zerocopy` decoding of `.dnam-bin` back into `SonaModelF64` | [`dnam_binary_format.md`](../../../../../docs/specs/containers/dnam_binary_format.md) |
| [`certify.rs`](certify.rs) | `render_all_tiers` via `SonaVm`, `<= 0.05 LU` trims, & `zero_tier_delta_weights` | [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
| [`monotone.rs`](monotone.rs), [`helpers.rs`](monotone/helpers.rs), [`stage0.rs`](monotone/stage0.rs), [`stage_b.rs`](monotone/stage_b.rs) | Non-destructive `Stage 0` (`f32` `w_ro`) & `Stage A/B/C/D` monotone repair | [`psychoacoustic_quality_and_tiers.md`](../../../../../docs/specs/architecture/psychoacoustic_quality_and_tiers.md) |
