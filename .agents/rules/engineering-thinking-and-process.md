---
trigger: always_on
---

# Engineering Thinking, Problem-Solving Strategy & Execution Discipline

These cognitive and process standards are mandatory for every research, design, debugging, implementation, and qualification task in DNAM:

## Part I — How to Think (First-Principles & Adversarial Engineering)

1. **Derive Before Coding & Isolate the Lowest Broken Layer:**
   - Before editing code or tuning parameters, derive the mathematical, physical, or hardware constraints first (expressiveness ceiling, rank/nullspace, delay-bandwidth product, stability bounds, precision floor, cache/SRAM footprint).
   - Always diagnose failures bottom-up across five ordered layers and **never patch layer `N` to hide a defect in layer `< N`**:
     1. *Data / Signal / Alignment* (lag, polarity, subsample shift, clock drift, warmup, noise floor)
     2. *Objective / Gradient / Adjoint* (finite-difference accuracy, sign errors, loss-term conflicts)
     3. *Optimizer / Conditioning* (parameter transforms, Hessian curvature, dead zones, initialization)
     4. *Model / Operator Expressiveness* (representation limits, static vs. modulated paths, topology depth)
     5. *Numerical Precision & Hardware* (`f32`/`fp16` cancellation or overflow, L1D spills, SIMD/GPU occupancy, SRAM)

2. **Falsify Cheaply Before Investing Heavily (`< 30 s` Micro-Experiments):**
   - Never test an unproven mathematical idea for the first time by refactoring multiple crates and running a multi-minute build or capture.
   - Prototype and stress-test new formulas, error bounds, or solver ideas in a `< 30 s` isolated unit test or short numerical script using the **exact target arithmetic** (`f32`, NR2/NR3 approximations, `fp16` quantization) rather than idealized `f64` math.
   - Invert your own hypothesis first: ask *"What is the fastest experiment that would prove my hypothesis **wrong**?"* and run that check before building on top of it.

3. **Adversarially Red-Team Your Own Design Before Implementing:**
   - Before committing to any operator, wire layout, or algorithm, actively try to break it across five axes:
     - *Extremes & Degeneracies:* `0`, `-0.0`, `±100` (`REQ-7.1`), `±1e9` (`REQ-7.2`), subnormals/`NaN`, equal endpoints (`r_n == r_p`), silent segments, and zeroed weights at `StarCenter`.
     - *Physics, Sign & Causality:* Gain/sag polarity, group delay and phase dispersion, stage ordering, and sample-rate invariance (`44.1..192 kHz`).
     - *Cross-Cutting Invariants:* Determinism, live tier switching (`REQ-5.4`), `4×` offline bounce parity (`REQ-5.5`), `f32 → fp16` watermark survival (`REQ-8.5`), `64 KiB` L1D (`REQ-6.1`), `15.5 KiB` GPU SRAM (`REQ-4.2`), and `0 FDIV/FSQRT/IDIV/libm` (`REQ-6.2`).
     - *Optimization Geometry:* Check gradient conditioning, transform saturation, Hessian nullspaces, and hard-clamp dead zones—an operator that works forward but poisons gradients is broken.
     - *Honest Cost Accounting:* Measure actual hot-loop bytes, cold-RAM spills, instructions, and GPU barriers; never ignore spills or move goalposts.

4. **Zoom Out When Complexity Accretes (Structural Simplicity Over Local Patches):**
   - When fixing a failure demands special cases, compensating penalties, or multi-file workarounds, step back one architectural level and ask *what wrong abstraction, missing invariance, or dead indirection is making the simple solution hard*.
   - Prefer unifying algebraic identities and deleting dead code/bytes over bolting on extra stages. Actively reject sunk-cost attachment: archive or discard a failing branch cleanly rather than piling patch `#3` onto patch `#2`.

5. **Epistemic Rigor & Zero Rationalization:**
   - Strictly distinguish **measured facts** (direct command/test output) from **analytical derivations** and **untested assumptions**. Never present an unmeasured estimate as verified.
   - Never shrug off a small unexpected numerical discrepancy ("probably rounding" or "close enough"): in DSP and adjoint code, unexpected residuals almost always expose a real sign, phase, index, cancellation, or parity bug.

## Part II — How to Execute (Anti-Loop, Spec-First, Atomic Commits & Tiered Gates)

6. **Oracle Ceiling First & Max-2-Attempt Circuit Breaker:**
   - Prove via a standalone oracle or synthetic self-distillation that the VM clears the target gate by `>= 6 dB` before tuning any optimizer or loss weight.
   - Log `(Hypothesis | < 60 s Diagnostic | Predicted Quantitative Delta)` before every non-trivial fix.
   - **Hard 2-Strike Stop:** If **2 consecutive attempts** fail to achieve the predicted delta (or regress another metric), **STOP editing immediately**, run the 5-layer root-cause audit above, and escalate structural blockers to the user instead of looping.
   - **Banned Smells:** Zero fixture/control-name heuristics (`REQ-1.6`/`1.9`), zero penalty-stacking over broken gradients, zero multi-minute qualification captures used as a printf debugger, and zero loosening of spec thresholds.

7. **Spec-First ADRs & Canonical Documentation Hygiene (`docs/CONTRIBUTING.md`):**
   - Update `docs/adr/` (`docs/templates/adr_template.md`, sequential `ADR-NNNN`, descriptive candidate names, `git mv NNNN-superseded-*.md` or terse `## Amendment Appendix`) and SSOT `docs/specs/` **before** changing contracts.
   - Run `moon check tests` after every doc or `AGENTS.md` edit (clean-room separation in the 16 domain specs, zero raw `$` in Markdown tables, GitHub MathJax escapes, strict inline code pill ban in `docs/specs/`, prose `<= 650` chars / list `<= 240` chars, standalone `r[...]` / `verify[...]` rules, single root `AGENTS.md` `< 5,000 B` budget with zero nested `AGENTS.md` files, and required `<pkg>/CONTRIBUTING.md` guides).

8. **Small Atomic Commits & Persistent Task Ledger:**
   - Keep each commit to **1 cohesive concern/layer** (`~150–450` net prod LOC, `<= 600` max excluding generated `.wgsl`; every file `< 900` LOC per `modular-decomposition.md`).
   - Run `moon check <touched-pkgs>`, `moon check tests` (when docs/structure change), and `>= 95%` coverage on touched crates, then commit **immediately** with `git commit -s`. Never accumulate sprawling cross-crate dirty trees.
   - Maintain a persistent `task_ledger.md` artifact on multi-step campaigns (locked decisions, atomic commit queue + SHAs, exact numeric metric ledger, 2-strike attempt log, and failed-hypothesis graveyard).

9. **Fail-Fast Tiered Verification (`T0 → T1 → T2 → T3 → T4`):**
   - Climb from cheapest to most expensive verification; never jump to slow all-tier captures until lower tiers pass:
     - **`T0` (`< 5 s`):** Unit + `proptest!` micro-oracles, central FD (`rel_err < 1e-6`), `CPU ↔ WebGPU` parity, `REQ-7.2 ±1e9` fuzz, `0 FDIV/FSQRT/IDIV/libm` disassembly.
     - **`T1` (`< 30 s`):** Single-tier `Lite` `L1` fast-loop canary (`QualifyConfig::fast_self_test`).
     - **`T2`:** Nested `Lite ⊂ Full ⊂ Pro ⊂ Ultra` `L1 → L2` monotonicity, `STAT uplo='U'`, live tier switching (`REQ-5.4`), `4×` bounce parity (`REQ-5.5`), L1D & WCET budgets (`REQ-6.1`).
     - **`T3`:** Stratified fixture waves running `L1` first (`DNAM_DIAG_STOP_AFTER_L2`) → `L2` → `L3`, committing cleanly every 1–2 certified fixtures.
     - **`T4`:** Full `CPU ↔ wgpu`, multi-SR (`44.1..192 kHz`), `REQ-8.5` watermark survival, `>= 95%` coverage, `moon check --all`, and Dual Clean-Room Critic sign-off.
