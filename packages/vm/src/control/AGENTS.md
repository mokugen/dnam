# `dnam-vm::control` — `3 kHz` `ControlBus`, `CP1` & `40 B` `TempoSync` Map

Scoped discovery guide for [`src/control.rs`](../control.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Enforces `#[forbid(unsafe_code)]`, `--no-default-features` `!alloc` operation, `0` `FDIV`/`FSQRT`/`libm`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`descriptor.rs`](descriptor.rs) | `ControlClampDescriptor`, 3-slot conditional intersection, Lobatto basis | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`exact_int.rs`](exact_int.rs) | `9-word S` + `18-word D` `ExactIntFactorAccumulator` (`0-ULP` `StarCenter`) | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`cp1.rs`](cp1.rs) | `"CP1\0"` overflow + inline `cp1_dir` decoder & `U_int` FM matrices | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`tempo.rs`](tempo.rs) | `40 B` `"TSYN"` `PeriodMs`/`FrequencyHz`, Lobatto inversion, `Q64`/`Q32` phase | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`bus.rs`](bus.rs), [`load.rs`](bus/load.rs), [`tick.rs`](bus/tick.rs), [`dispatch.rs`](bus/dispatch.rs), [`laws.rs`](bus/laws.rs), [`tank_and_router.rs`](bus/laws/tank_and_router.rs) | `!alloc` `ControlBusBacking`, `stage_targets[0..32]` + `"CEXT"`, slot 16 snap, `z_stage` + tank/router laws | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
