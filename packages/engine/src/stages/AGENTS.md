# `src/stages` — TORA Multi-Stage Rig Topology & Matryoshka Core

Implements the unified 5-stage physical Matryoshka core and matrix routing primitives (`#[forbid(unsafe_code)]`; see [`CONTRIBUTING.md`](../../CONTRIBUTING.md)).

## 1. Submodule Map

| Stage / File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`hot_core.rs`](hot_core.rs) ([`instance.rs`](hot_core/instance.rs), [`lifecycle.rs`](hot_core/lifecycle.rs), [`post_cond.rs`](hot_core/post_cond.rs)) | L1D-resident `V3HotCore` execution, tier switching & ZOH lifecycle | [`tora_architecture.md`](../../../../docs/specs/architecture/tora_architecture.md) |
| [`router.rs`](router.rs) ([`v3_stage0.rs`](router/v3_stage0.rs)), [`topology.rs`](topology.rs) ([`router_and_head.rs`](topology/router_and_head.rs), [`zoh_fir_onset.rs`](topology/zoh_fir_onset.rs)) | Stage 0 8-port router, ZOH/FIR onset & clean crossover | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`stage1_svf.rs`](stage1_svf.rs) ([`coeffs.rs`](stage1_svf/coeffs.rs), [`audio.rs`](stage1_svf/audio.rs)), [`stage1_pre.rs`](stage1_pre.rs) | Stage I Env-SVF & pre-dynamics input conditioning | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`stage2_gru.rs`](stage2_gru.rs), [`stage2_gru/AGENTS.md`](stage2_gru/AGENTS.md) | Stage II 3-block tapered `GRU`, `FiLM`, Tone `SVF` & kernels | [`op_n_commuted_cascade.md`](../../../../docs/specs/dsp/op_n_commuted_cascade.md) |
| [`stage3_cab.rs`](stage3_cab.rs) ([`cold.rs`](stage3_cab/cold.rs), [`kautz_zoh.rs`](stage3_cab/kautz_zoh.rs)), [`stage3_cabinet.rs`](stage3_cabinet.rs) | Stage III Early FIR & Kautz modal banks (`ZOH` + 3 kHz retune) | [`op_m_modal_kautz.md`](../../../../docs/specs/dsp/op_m_modal_kautz.md) |
| [`stage4_tank.rs`](stage4_tank.rs) ([`v3_fdn.rs`](stage4_tank/v3_fdn.rs)), [`head.rs`](head.rs) | Stage IV Spatial FDN tank & Mid/Side Kronecker helpers | [`router_and_stages_topology.md`](../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`weights.rs`](weights.rs), [`weights/AGENTS.md`](weights/AGENTS.md) | 64B-aligned `Arena 0..3` directory & `v3` binary packing | [`dnam_binary_format.md`](../../../../docs/specs/containers/dnam_binary_format.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for stage chaining and memory layout.
