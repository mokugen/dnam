# `v3_core/` — L1D SIMD `v3` Matryoshka Core Map

See [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) for zero-stall real-time invariants (`#[forbid(unsafe_code)]`, co-located `#[cfg(test)] mod tests`).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../v3_core.rs`](../v3_core.rs) | `V3HotCore` & 4-tier slimmable GRU dispatch. | [`tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md) |
| [`instance.rs`](instance.rs) | `V3CoreInstance` runtime wrapper & scalar/`aarch64` block entrypoints. | [`tora_architecture.md`](../../../../../docs/specs/architecture/tora_architecture.md) |
| [`cascade.rs`](cascade.rs) | Dual-rate `2x` oversampler & 3-block Matryoshka GRU cascade. | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`svf.rs`](svf.rs) | Envelope-following dual-section TPT SVF & nonlinear tap bank. | [`router_and_stages_topology.md`](../../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`stage3.rs`](stage3.rs) | Stage III FIR + Kautz modal cabinet & `V3MatryoshkaModel`. | [`op_m_modal_kautz.md`](../../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`math_pade.rs`](math_pade.rs) | `tanh` / `sigmoid` Padé approximations & Newton reciprocal helpers. | [`op_n_commuted_cascade.md`](../../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
