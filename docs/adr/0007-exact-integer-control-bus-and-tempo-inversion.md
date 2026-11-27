---
id: ADR-0007
title: "Centered ANOVA + Rank-R FM Control Bus, Exact-Integer Accumulator & Physical Tempo Inversion"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/parametric_control_bus.md
requirements:
  - REQ-2.4
  - REQ-2.5
---

# ADR-0007: Centered ANOVA + Rank-`R` FM Control Bus, Exact-Integer Accumulator & Physical Tempo Inversion

This record justifies the control-bus and tempo-sync formulations in [`../specs/dsp/parametric_control_bus.md`](../specs/dsp/parametric_control_bus.md) fulfilling [`REQ-2.4`, `REQ-2.5`](../requirements/system_goals_and_requirements.md).

## 1. Context: Why Naive Conditioning Architectures & $P \le 16$ Control Buses Fail

| Naive Architecture | Structural Failure Mode |
| :--- | :--- |
| **1. Audio-Rate (`48 kHz`) Input Concatenation / FiLM** | Re-evaluates knob embeddings on every audio sample (`+1,536 MACs/sample`), wasting `> 40%` of the embedded Cortex-A76 cycle budget even when knobs are stationary. |
| **2. Unconstrained Dense MLP Hypernetwork** | Overfits sparse `Star`/`Corner` stem grids (`15–30` stems) and hallucinates unstable pole radii (`r > 1.0`) or negative delays at unvisited multi-knob corners. |
| **3. High-Degree Uniform Polynomials** | Fitting degree-3 polynomials to a control captured at only `1` or `2` settings triggers severe Runge oscillation across interior knob positions. |
| **4. Unfactored Pairwise Tensor Products** | Full pairwise interaction tables scale as `O(P²)` in memory and `O(P)` per control update, hitting a hard `P ≤ 16` wall on multi-channel studio rigs (`P = 20..64` controls). |

## 2. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Canonical Self-Contained `v3.0` `CP1` Arbitrary-`P` Control Projections (`ref[arch.pillars.cp1-projections]`, `ref[container.dnam.cp1-profile]`, `ref[dsp.op-n.ctrl.incremental-update]`)** | Embed base control descriptors, inline `cp1_dir`, `U_int`, `W_ctrl`, and Lite `W_readout` wholly in Arena 0 (`Arena0LiteWire`, placing overflow `CP1` in `.dnam-bin` `shared_controls` when `dbin_prefix_bytes > 0` with `Region 4 == 0 B`, or in `Region 4` `shared_extent` otherwise), supporting arbitrary `P` controls in `O(1)` per moved control with zero real-time heap allocations on `reload_model_in_place` and lossless `strip_container_to_tier(Lite)` (`REQ-5.6`). | Fixed 16-slot in-arena basis arrays hit a hard control-count ceiling (`P <= 8..16`) or alias slots modulo 16 on multi-channel studio rigs. |
| **Switch `default_val` Zero-Baseline & Hadamard FM Identification (`ref[dsp.ctrl.switch.option-contributors]`, [`ADR-0016`](0016-codesigned-disjoint-comb-stimulus-and-null-homing-capture.md))** | Anchor each discrete `switch` at its signed `default_val` (`StarCenter` emits `0.0`) so only non-default captured options emit `1`-lane `CP1` records, and identify Rank-4/8 FM interaction factors (`U_int^base[4, 8]` and `U_int,Delta[3, 4, 8]`) from 2 orthogonal `3.0 s` Hadamard diagonal sweeps with automatic 1D commutator interaction pruning (`Theorem 3.5`). | Non-zero switch baselines shift `StarCenter` mode-0 audio; full `P(P-1)/2` pairwise captures explode capture time quadratically in `P`. |
| **Automation Timing & Tie Precedence (`ref[dsp.ctrl.op-m.timing]`)** | Apply host parameter updates at the `3 kHz` control boundary (`n mod K_ctrl == 0`), coalescing in timestamp order. On exact-timestamp ties: DAW automation wins over live Studio/MIDI; between live Studio and MIDI, last-received wins. | Sample-rate re-projection of `W_ctrl` on every host event or arbitrary array-order tie breaking. |
| **Exact-Integer Sufficient-Statistic Accumulator (`ref[dsp.ctrl.accum.cp1-scaling]`, `ref[dsp.ctrl.accum.exact-int]`)** | Accumulate materialized finite `f32` main effects and Rank-`R` interaction pair statistics (`ExactIntFactorAccumulator`) exactly in fixed-width integers (updating pair sums via two shifted 24-bit significand products) and feed the exact-real oracle into trust-region clipping before narrowing to `f32`. | Naive `f32` `(∑ v_p z_p)² - ∑ v_p² z_p²` suffers catastrophic cancellation and drifts by `> 0.001` after `10,000` closed three-step control cycles. |
| **4-D Groupwise Outward-Checked Trust-Region Clip & Unified 32-Slot `W_ctrl` + `"CEXT"` Dispatch (`ref[dsp.ctrl.op-n.trust-clip]`, [`ADR-0022`](0022-sona-state-space-operator-and-webgpu-architecture.md))** | Apply outward-checked `f64` trust-region clipping independently to each 4-D block (`z_head[0..4]`, `z_stage[0..4]`, `z_head[4..8]`, `z_stage[4..8]`), evaluate uniform micro-ramping across `stage_targets[0..32]` (incl. slot 16 `freeze_mix`, slot 22 `op_t_dry_mix_mod`, slot 30 `sense_vca_gain_delta`) via `16 × 16` `W_ctrl` + optional `"CEXT"` over `q16 = [z_head[0..4], ctrl_basis_4[0..4], skewed q2 (lfo_skew_fp16), SensoryQuad q3]` plus `eff_z_stage = static_z_stage + w_ou * ou_drift_state` (and `S_router` `S_{ret->send} = 0` isolated in host shell). | Joint 8-D/16-D norm clipping couples head and stage coordinates and breaks Mechanism A `Q_buyer` gauge invariance; sparse tank-only or 16-slot dispatch omits Crossbar, Core, `freeze_mix`, and extended transport modulation. |
| **Tempo Range, Transport Lock & Reachable Physical Inversion (`REQ-2.5`, `ref[dsp.tempo.q64-q32-phase]`, `ref[dsp.tempo.physical-inversion]`)** | Support `20–400 BPM` inclusive across host, MIDI (`24 PPQN`), and tap sources; invert calibrated Lobatto polynomials over reachable `f32` physical knob coordinates (with inward `f32` rank intersection of `f64` safety bounds), optional octave folding, and `Q64`/`Q32` phase tracking. | Solving quadratic roots in normalized real coordinates can select unreachable `f32` values or wrong branches; averaging raw frame counts across sample-rate changes corrupts BPM (`120 -> 160 BPM`). |

## 3. Evaluated R&D Candidate: Fixed-Tree Control Accumulator

Maintains a balanced binary tree over $P \le 32$ controls where internal nodes store $S_v = S_L + S_R$ and $I_v = I_L + I_R + S_L \odot S_R$, updating in $\mathcal{O}(\log_2 P)$ steps per control move without subtraction cancellation (superseded by the $\mathcal{O}(1)$ `CP1` + `ExactIntFactorAccumulator` architecture supporting arbitrary `P ∈ [0, 65535]`).

## Amendment Appendix

- **2026-11-20T02:07:00-05:00:** Amended so control and tempo laws run zero-JSON inside `#![no_std]` `!alloc` VM environments and keep autonomous modulators ticking on zero-knob rigs (`ADR-0020`, `ADR-0022`).
- **2026-11-20T23:02:00-05:00:** Amended to replace fixed LFO skew, fallback envelope poles, OU drift target indices, and un-ramped exceptions with learned, uniformly ramped laws (`ADR-0022`).
- **2026-11-27T14:00:00Z:** Amended for `ControlBus v4 (GRA)`, keeping `ExactIntMainAccumulator` on `SO(4)`-isolated `z_head[0..4]`, `CP1`, and `TSYN` while reclaiming `6,336 B` of dead `W_ctrl`/`CEXT`/`U_int` planes for Zero-Anchored `SwiGLU + Padé-[1/1]`, Anchored `1.5-Entmax` Prototypes, `6`-edge and per-quad `Op-T` soft routing, bilinear/inertia dynamic gates, `4`-frame micro-ramping (`K_ctrl(44.1k) = 16`), live `TSYN` phase-locking, zero-cost static idle, and `O(P + M)` BIBO bounds ([`ADR-0024`](0024-sona-v4-kernel-hb-fir-perimeter-and-control-bus-gra.md), [`ADR-0025`](0025-unified-runtime-and-non-causal-trainer-attention.md)).
