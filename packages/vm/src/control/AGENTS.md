# `dnam-vm::control` — `3 kHz` `ControlBus`, `CP1` & `40 B` `TempoSync` Map

Scoped discovery guide for [`src/control.rs`](../control.rs) (parent [`AGENTS.md`](../../AGENTS.md)). Enforces `#[forbid(unsafe_code)]`, `--no-default-features` `!alloc` operation, `0` `FDIV`/`FSQRT`/`libm`.

| Module / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`descriptor.rs`](descriptor.rs) | `ControlClampDescriptor`, 3-slot conditional intersection, Lobatto basis | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`exact_int.rs`](exact_int.rs) | `9-word S` + `18-word D` `ExactIntFactorAccumulator` (`0-ULP` `StarCenter`) | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`cp1.rs`](cp1.rs) | `"CP1\0"` overflow + inline `cp1_dir` decoder & `U_int` FM matrices | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |
| [`tempo.rs`](tempo.rs) | `40 B` `"TSYN"` `PeriodMs`/`FrequencyHz`, Lobatto inversion, `Q64`/`Q32` phase | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`bus.rs`](bus.rs), [`load.rs`](bus/load.rs) | `!alloc` `ControlBusBacking`, `lfo_skew` (`[-1.0, 1.0]`), `ou_to_z_stage` → `eff_z_stage()`, `base_tank_dry_mix()`, `stage_targets[0..32]` + `"CEXT"` (`ro_pole_48k` in [`slab`](../graph/slab/AGENTS.md)) | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`tick.rs`](bus/tick.rs), [`dispatch.rs`](bus/dispatch.rs) | Skewed `Q0.192` LFO (`q2`), sensory `e[0..4]` → `q3`, OU drift, uniform `32`-slot `ControlBus` micro-ramping | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`laws.rs`](bus/laws.rs), [`transport_and_sensory.rs`](bus/laws/transport_and_sensory.rs) | `eff_z_stage` laws, `Op-T` slots `12..=25` (`freeze_mix` slot `16`, `tank_dry_mix` slot `22`), `Op-P` `slot 29`, `SensoryQuad` `slot 30` VCA bias | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
