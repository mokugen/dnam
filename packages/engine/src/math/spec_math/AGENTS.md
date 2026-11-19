# `src/math/spec_math` — Bit-Exact & Finite-Precision Math Submodules

Organizes the normative bit-exact kernels re-exported by [`../spec_math.rs`](../spec_math.rs):

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`saturation.rs`](saturation.rs) | `fp16` telemetry & `±1.0e9` hard-safety ceiling | [`sona_architecture.md`](../../../../../docs/specs/architecture/sona_architecture.md) |
| [`ieee_bits.rs`](ieee_bits.rs) | Strict bit-exact `f32`/`f64` rounding & finite bounds | [`parametric_control_bus.md`](../../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`trust_clip.rs`](trust_clip.rs) | Ellipsoidal trust-region clip & Stage IV loop bounds | [`control_space_audit.md`](../../../../../docs/specs/training/control_space_audit.md) |
| [`pitch_log2.rs`](pitch_log2.rs) | Monotone pitch `log2` & time-warp profile | [`parametric_control_bus.md`](../../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`float_rank.rs`](float_rank.rs) | Admissible `f32` rank-space intersection | [`parametric_control_bus.md`](../../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`rotor_q192.rs`](rotor_q192.rs) | Contractive Horner rotor & `Q0.192` phase accumulator | [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for zero-`libm` / zero-`coverage(off)` guardrails.