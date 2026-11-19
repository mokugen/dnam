# `stage0_id/` — Stage 0 (`Op-P`) Sub-Octave Identification Map

See [`../../../../CONTRIBUTING.md`](../../../../CONTRIBUTING.md) and [`../../../../../../docs/specs/dsp/router_and_stages_topology.md`](../../../../../../docs/specs/dsp/router_and_stages_topology.md).

| File | Scope | Spec |
| :--- | :--- | :--- |
| [`../stage0_id.rs`](../stage0_id.rs) | Sub-harmonic variance `ΔR²` gate (`0..8` lag) and end-to-end 4-tier acceptance. | [`router_and_stages_topology.md`](../../../../../../docs/specs/dsp/router_and_stages_topology.md) |
| [`fit.rs`](fit.rs) | Causal `8`-tap FIR Variable-Projection + damped Gauss-Newton `Op-P` solver across body and sweep stems. | [`four_stage_pipeline.md`](../../../../../../docs/specs/training/four_stage_pipeline.md) |
