# `src/math/spec_math` — Bit-Exact & Finite-Precision Math Submodules

Organizes the normative bit-exact kernels re-exported by [`../spec_math.rs`](../spec_math.rs):

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`rotor_q192.rs`](rotor_q192.rs) | Quadrant-reduced Horner contractive rotor, adaptive ULP guard & `Q0.192` phase accumulator | [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`float_rank.rs`](float_rank.rs) | `binary64` declared-range normalization & ordered admissible `f32` rank-space intersection | [`parametric_control_bus.md`](../../../../../docs/specs/dsp/parametric_control_bus.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for zero-`libm` / zero-`coverage(off)` guardrails.
