---
id: ADR-0006
title: "Model-Declared fp16 Delay Histories, Centered Granular Pitch & Spatial Tank Feedback Guard"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/op_t_transport.md
  - docs/specs/training/control_space_audit.md
requirements:
  - REQ-1.4
---

# ADR-0006: Model-Declared `fp16` Delay Histories, Centered Granular Pitch & Spatial Tank Feedback Guard

This record justifies the delay history, pitch windowing, and feedback stability rules in [`../specs/dsp/op_t_transport.md`](../specs/dsp/op_t_transport.md) and [`../specs/training/control_space_audit.md`](../specs/training/control_space_audit.md) fulfilling [`REQ-1.4`](../requirements/system_goals_and_requirements.md).

## 1. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Model-Declared Delay Histories (`ref[dsp.op-t.ring.capacity]`, `ref[dsp.op-t.ring.q0-32-pointer]`)** | Cold-allocated 64B-aligned `fp16`/`f32` histories with arbitrary-length branchless wrap, integer-plus-`Q0.32` delay pointers, inward representable `f32` reader bounds, and host memory-ceiling admission. | Fixed power-of-two ring assumption wastes up to `50%` RAM or truncates `20 BPM` dotted-half (`9 s`) delays; unguarded `f32` fraction rounding can carry across fractional safe bounds. |
| **Granular Pitch Shifter (`±24` st, `ref[dsp.op-t.window.m2-pitch]`)** | Scope the `0.25` sample/sample slew limit to continuous base-delay changes; use the centered grain-duration construction (`D_center ≥ 7,205` samples for a `100 ms` grain at `48 kHz` and `+24` st) with causal floor `τ_m ≥ 5.0` across 4-frame crossbar/`TransportQuad` micro-chunks (`≥ 4.0` standalone). | Applying `0.25` sample/sample slew to grain-head phase warp caps pitch shift well below `+24` semitones (`3.0` samples/sample warp rate). |
| **Transport & Delay History Feedback Guard (`ref[dsp.op-t.guard.bounded-history]`, `ref[dsp.tank.loop-saturator]`, `ref[dsp.tank.quiet-feedback-knee]`)** | Compact `216 B` pure-wet `TransportQuad`s use unified `sigma_sat` cubic limiting (`α = 0`, `TANK_MAX_SIGMA_SAT_FP16 = 0x3BE0`, precomputed `inv_sigma_23`, `V_ceil = 0.985`), continuous passive `freeze_mix` (`slot 16`) + host `WE` write lock, and stage-level `tank_dry_mix` (`slot 22`); standalone delay lines retain generalized rational knee `K(u)` and grouped max-tree bounded-history guard `clamp(v_i[n], -M_n, M_n)`. | Even within the `0.25` slew limit at `3 kHz` control rate, a time-varying Catmull-Rom loop at `g = 0.999` has a period multiplier in `(1.02, 1.03)` without the bounded-history feedback guard. |

## 2. Evaluated R&D Candidates

1. **Bounded Hermite Reader & Passive Projected-State Delay:**
   - *Bounded Hermite Slope Clamp:* For $d_0 = x_1 - x_0$ and raw centered differences $m_0 = (x_1 - x_{-1})/2$, $m_1 = (x_2 - x_0)/2$, the bounded slope clamp $T(a, b) = \frac{1}{2}\big(\mathrm{clamp}(a + b, -3|d_0|, 3|d_0|) + \mathrm{clamp}(a - b, -|d_0|, |d_0|)\big)$ enforces $p(\mu) \in [\min(x_0, x_1), \max(x_0, x_1)]$.
   - *Passive Projected-State Update:* Projects history into orthogonal Legendre modes $z_m[n] = \langle \mathbf{x}_n, \mathbf{q}_m(\mu_{n-1}) \rangle$ so time-varying delay sweeps remain contractive (`‖z[n]‖₂ ≤ ‖x_n‖₂`).
2. **Octave/Fractional Fixed-Point Pitch Table:** Decomposes $p_{\text{st}} = 12 q + r$ with 257-entry table $P[idx] = \lfloor 2^{idx/256} \cdot 2^{30} + 0.5 \rfloor$ and cubic correction, achieving $< 0.002$ cents error without runtime `exp2f`.

## Amendment Appendix

- **2026-11-20T02:07:00-05:00:** Amended to make transport loops continuously differentiable for black-box solver discovery without discrete mode switches or bespoke freeze forks (`ADR-0022`).
- **2026-11-20T23:02:00-05:00:** Amended to eliminate asymmetric quad special cases and non-differentiable Schmitt/knee heuristics while fitting active state into `216 B` (`ADR-0022`).
- **2026-11-27T14:00:00Z:** Amended for `Op-T v4` adding per-quad modulation routing and per-tap readout, passive convex `M_fb`, bipolar `g_loop ∈ [-0.9990, +0.9990]`, causal lower-triangular `M_series`, `alpha_sum_fb`, `±64 Hz` ramp, modulatable `ramp_spans_samples`, wired 3-phase B-spline, `lfo_rate_ratio`, quadrature AM, passive shimmer (`V_ceil = 4.0`), unitary `H_4(1)` freeze, `0.25` samp/samp delay slew, and `64`-frame inline ring ([`ADR-0024`](0024-sona-v4-kernel-hb-fir-perimeter-and-control-bus-gra.md)).
