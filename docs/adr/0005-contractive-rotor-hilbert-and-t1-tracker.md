---
id: ADR-0005
title: "Quadrant-Reduced Contractive Rotor, H1 Hilbert Lattice, T1 Tracker & Cabinet ZOH"
status: accepted
supersedes: []
superseded_by: null
specs:
  - docs/specs/dsp/op_m_modal_kautz.md
requirements:
  - REQ-1.2
  - REQ-1.3
---

# ADR-0005: Quadrant-Reduced Contractive Rotor, `H1` Hilbert Lattice, `T1` Tracker & Cabinet ZOH

This record justifies the modal, Hilbert, and tracker formulations in [`../specs/dsp/op_m_modal_kautz.md`](../specs/dsp/op_m_modal_kautz.md) fulfilling [`REQ-1.2`, `REQ-1.3`](../requirements/system_goals_and_requirements.md).

## 1. Decision & Counterexample Matrix

| Mechanism | Canonical Selection | Counterexample to Naive Formulation |
| :--- | :--- | :--- |
| **Contractive Rotor & LFO Phase (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.q0-192-lfo]`)** | Quadrant reduction to `[0, π/4]`, degree-13/14 Horner `f64` reference and pure-32-bit `algebraic_unit_rotor_f32` (`‖u_rot‖₂ < 1`) with exact zero-increment hold, `Q0.192` LFO phase accumulator, and learned `T_2` harmonic skew (`lfo_skew_fp16`). | Unreduced `[-π/2, π/2]` polynomial suffers cancellation near `π/2`; `f32` rounded rotor state update stalls into a fixed point for slow LFO increments (`Δθ = 2^-30 rad`). |
| **Family 2 `H1` Hilbert Pair (`ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`, `ref[dsp.op-m.hilbert.rate-remap]`)** | 6-section all-pass pair with 15-anchor rate table (`HILBERT-H1-ANCHORS`, `< 0.10°` @ `48 kHz`, `<= 0.19°` at 15 anchors, `<= 0.34°` across `[44.1, 96]` & `[176.4, 384]` kHz and `f >= 30 Hz` in `(96, 176.4)` kHz, `<= 0.67°` @ `20 Hz` near `136.2 kHz` over `20 Hz–20 kHz`), exact ties-to-even interpolation, and history-consistent two-sample section state initialization on rate transitions. | Naive `f64` linear interpolation rounds halfway rates to the wrong `f32` neighbor; copying transposed state cells across rate changes breaks constant-input equilibrium. |
| **Family 1 `T1` Sub-Octave Tracker (`ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.toggle-sheet]`)** | Upstream `5 Hz` TPT DC blocker (`T1-COEFFICIENT-VECTORS`), pure-`f32` `rsqrt_pos_nr3` magnitude normalization, `tracker_crosses_branch_cut_f32` guard, and 2-stage cancellation-free half/quarter-angle smooth quadratures `env_gate * [cos(theta/2), sin(theta/2), cos(theta/4), sin(theta/4)]` into crossbar `Op-P` lanes `0..4`. | Testing crossings on the normalized sheet phasor misses or invents toggles when `I` flushes under FTZ; pausing tracker state during local mute loses divider phase parity. |
| **`ModalConvolver` Sample-Rate Conversion (`ref[dsp.op-m.kautz.rate-convert]`, `ref[dsp.zoh.cabinet-fir-onset]`)** | Integer multiples (`48, 96, 192, 384 kHz`) use exact sparse delay `H_ref(z^q)`; live `44.1–48 kHz` uses the 64-tap anchored difference family; `4×` (`176.4–384 kHz`) uses the 6-point Lagrange difference mapping; other non-integer rates use Catmull-Rom FIR rescaling. | 4-tap Catmull-Rom at `44.1 kHz` attenuates `20 kHz` by `4.51 dB`; forward `f32` summation of unit-DC taps yields `0x3F800002` without difference-form anchoring. |

## 2. Evaluated R&D Candidates & LFO Detection Profiles

1. **Quarter-Wave Fixed-Point Rotor Table:** 513-entry `u32` table $T[m] = \lfloor \sin(m\pi/1024) \cdot 2^{30} + 0.5 \rfloor$ with linear interpolation $S(u) = (1 - 2s)\big(((2^{16}-f)T[idx] + f T[idx+1] + 2^{15}) \gg 16\big)2^{-30}$ and unit-circle correction $\delta = \big((3 \cdot 2^{29} - (\tilde{c}^2 + \tilde{s}^2)2^{-31}) \gg 1\big)2^{-30}$.
2. **Repeated-Window Cubic-Nuisance & Helmert LFO Detection:** Evaluates harmonic regression over $M$ repeated windows against a cubic drift baseline (`1, τ, τ², τ³`) via exact $F(2, \nu)$ variance ratios or orthonormal Helmert contrasts $z_k[n] = \big(\sum_{j=1}^k y^{(j)}[n] - k y^{(k+1)}[n]\big) / \sqrt{k(k+1)}$.

## Amendment Appendix

- **2026-11-20T02:07:00-05:00:** Amended to maintain sub-degree quadrature accuracy up to `192 kHz` and support stereo and silent-input carrier modulation (`ADR-0022`).
- **2026-11-20T23:02:00-05:00:** Amended to eliminate octave-divider step aliasing, prevent quarter-angle norm drift, and keep LFO harmonic skew learned (`ADR-0022`).
