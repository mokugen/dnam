---
id: ADR-0021
title: "TORA v3 Unified Zero-DC Matryoshka GRU, Direct-Register 5-Stage Topology, L1D-Resident Tier Sizing & Full-Bandwidth fp32 Capsules"
status: accepted
version: "1.0.0"
date: "2026-11-16"
supersedes:
  - ADR-0004
  - ADR-0008
  - ADR-0011
  - ADR-0017
superseded_by: null
specs:
  - ../specs/architecture/tora_architecture.md
  - ../specs/architecture/psychoacoustic_quality_and_tiers.md
  - ../specs/dsp/op_n_commuted_cascade.md
  - ../specs/dsp/op_m_modal_kautz.md
  - ../specs/dsp/router_and_stages_topology.md
  - ../specs/dsp/parametric_control_bus.md
  - ../specs/containers/dnam_binary_format.md
  - ../specs/containers/project_container_format.md
  - ../specs/training/four_stage_pipeline.md
  - ../specs/training/iterative_resume_and_caching.md
  - ../specs/training/control_space_audit.md
  - ../specs/security/watermarking_and_marketplace.md
  - ../specs/sdks/typescript_sdk.md
requirements:
  - REQ-1.1
  - REQ-1.5
  - REQ-2.1
  - REQ-3.1
  - REQ-3.3
  - REQ-4.1
  - REQ-4.2
  - REQ-4.3
  - REQ-5.1
  - REQ-5.2
  - REQ-5.4
  - REQ-5.6
  - REQ-6.1
  - REQ-6.2
  - REQ-6.3
  - REQ-7.1
  - REQ-7.4
  - REQ-8.5
---

# ADR-0021: TORA `v3` Unified Zero-DC Matryoshka `GRU`, Direct-Register 5-Stage Topology, L1D-Resident Tier Sizing & Full-Bandwidth `fp32` Capsules

This architectural decision record supersedes [`ADR-0004`](0004-division-free-polyadaa-and-c2-wavefolder.md), [`ADR-0008`](0008-per-stem-kronecker-statistics-and-wide-gram.md), [`ADR-0011`](0011-end-to-end-stage-wiring-and-arena-slice-topology.md), and [`ADR-0017`](0017-pure-rust-analytical-varpro-solver-and-planar-decoupling.md), unifying DNAM around the **TORA `v3` Direct-Register Matryoshka Architecture** across [`packages/engine/`](../../packages/engine/README.md) and [`packages/capture/`](../../packages/capture/README.md).

---

## 1. Context & Problem Statement

Practical multi-rig validation against the 13-chain black-box fixture matrix (`REQ-1.1`–`REQ-1.5`), embedded pedalboard WCET/L1D constraints (`REQ-6.1`–`REQ-6.3`), and unattended multi-stem Vault editing (`REQ-3.1`–`REQ-3.3`) exposed five structural bottlenecks in the TORA `v2` compromises recorded in `ADR-0004`, `ADR-0008`, `ADR-0011`, and `ADR-0017`:

1. **`256-D` `feature_bus_scratch` Scatter/Gather & `Role B` Overhead:** Materializing `feature_bus_scratch: [f32; 256]` and folding `Role B` readout blocks (`b = 1..K_t-1`) created excessive memory traffic (`4,608 B`+ output-head slices and extra `Arena 0..3` weight bloat), cache misses on `64 KiB` ARM Cortex-A76 L1D, and unnecessary coupling between watermarks and nonlinear trajectories.
2. **Stage II Fixed `PolyADAA` + `Theorem 1` Basis Memory Ceiling:** Fixed-drive `PolyADAA` banks (`0.25x..16x`) paired with `K_t <= 192` commuted LTI blocks struggled on high-order dynamic amp/compressor coupling and extreme waveshaping (`Chain-02 GatedFuzz`, `Chain-06 EnvSVF`) while demanding large readout projections.
3. **High Per-Instance Working Set & Conditional Stage-Bypass Complexity:** Keeping `~14.8 KiB` transient kernel scratch inside each engine instance inflated serial multi-pedalboard working sets past `64 KiB` L1D when running multiple chained instances, while per-stage bypass flags introduced branch divergence and uneven WCET profiles.
4. **Short `250 ms` `18 kHz` `fp16` Coresets & `192-Bit` Feature Gram Bloat:** Truncating `.dnam-proj` `INPC`/`STMC` coresets to `250 ms` (`12,000` frames at `18 kHz` band-limited `fp16`) dropped `18–20 kHz` harmonic air (`LTAS` bands `31–33`) and multi-second delay/envelope context needed for raw-audio-free `Path 2` re-pooling (`REQ-3.3`, `REQ-5.4`).
5. **Revised Embedded Pedalboard Serial Budget (`REQ-6.1`):** Practical pedalboards commonly chain up to `14` serial effects per core on `1× RK3582 Cortex-A76 @ 2.4 GHz` at an `80%` real-time scheduling cap (`16,667 ns/sample` usable budget including LV2 host dispatch and multi-instance cache spill), requiring `>= 14x Lite`, `>= 8x Full`, `>= 4x Pro`, and `>= 2x Ultra` with all DSP stages unconditionally active.

---

## 2. Considered Options

| Candidate Architecture | Stage II Nonlinear Core | Feature Transport & Scratch | Tier Sizing & Embedded Serial Budget | Capsule Storage & Solver |
| :--- | :--- | :--- | :--- | :--- |
| **TORA `v2` (`ADR-0004` / `0008` / `0011` / `0017`)** | `1× PolyADAA` + `Role A/B` `Theorem 1` LTI-commuted cascade | `256-D` `feature_bus_scratch` + Kronecker Mid/Side head | Fixed `64/128/192/256-D` (`10x / 6x / 2x` floors); per-instance scratch in heap state | `250 ms` `18 kHz` `fp16` coresets + `192-bit` wide-significand Gram + `5`-stage `MM-Cholesky` |
| **TORA `v3` Direct-Register Matryoshka Architecture (Selected)** | **Zero-DC Tapered Matryoshka `GRU` (`[1,1,1]`, `[2,1,1]`, `[2,2,1]`, `[3,2,1]`) @ `1x` (`Lite`/`Full`) & `2x` (`Pro`/`Ultra`) + Inter-Block `3 kHz` `TPT-SVF` + `3 kHz` `FiLM`** | **Direct SIMD-register `pre_cab` & `direct` `VarPro` linear accumulators (`0` `256-D` scatter/gather, `0` bypass flags); stack-allocated `V3KernelScratch`** | **Inverted L1D-Resident Max-Quality Sizing (`<= 4.5 / 10 / 18 / 28 KiB` arenas): `>= 14x Lite`, `>= 8x Full`, `>= 4x Pro`, `>= 2x Ultra` on `1× A76 @ 80%` RT cap** | **`2.0–2.5 s` (`96k–120k` frames) `20 Hz–20 kHz` `fp32` `INPC`/`STMC` + `STAT` `StarCenter` trajectory + `4`-stage Physics Probe / Windowed-BPTT / `FiLM+CP1` / `f32` Polish (`< 6 s/chain`)** |

---

## 3. Decision Outcome

We adopt **TORA `v3` Direct-Register Matryoshka Architecture**:

1. **Stage II Zero-DC Matryoshka `GRU` Cascade (`ref[dsp.op-n.stage2.sub-stages]`):**
   - **Tapered Pyramid & Continuous-Time ZOH Gate Scaling:** Replaces `PolyADAA` + `Role A/B` blocks with a 3-block feed-forward `4`-lane SIMD `GRU` pyramid (`Lite [1,1,1] = 12 ch @ 1x`, `Full [2,1,1] = 16 ch @ 1x`, `Pro [2,2,1] = 20 ch @ 2x`, `Ultra [3,2,1] = 24 ch @ 2x` / `4x` bounce). Storing continuous-time gate retention $1 - z_{\text{eff}}(q) = (1 - z_{1\times})^{1/q}$ lets `Lite` and `Full` run natively at `1x` (`48 kHz`) using `V3Rate1xCalibration` (`up_zoh_mix` + low-delay `poly3` decimation) nested inside `2x` (`96 kHz`) `Pro` and `Ultra`.
   - **Division-Free In-Register `NR2` Padé Activations & Analytical Zero-DC Equilibrium:** Evaluates $\tanh$ and $\sigma$ via a degree-5/4 rational Padé approximation using `SimdVec4::rcp_ge1_nr2` (`2` Newton-Raphson iterations in vector registers: `< 4.5e-5` relative error, `-87 dB`, `0-ULP` odd symmetry, $\tanh(0) = +0.0$, `0` GPR spills, `0` `FDIV`, `0` `libm`) and applies an analytical Zero-DC bias shift $b_n$ guaranteeing $h_{\text{eq}} = 0$ at rest (`ref[dsp.op-n.commuted.dc-equilibrium]`).
   - **Stereo `L + R` Tile Interleaving, `3 kHz` `FiLM` & Inter-Block `TPT-SVF`:** Steps `Left` (`Vec4`) and `Right` (`Vec4`) side by side sharing loaded `4×4` weight tiles across dual NEON FMA pipes (`step_stage2_stereo`), applies `3 kHz` control-modulated per-block `FiLM` gain/shift + Zero-DC $b_n$, and steps a `3 kHz`-decimated inter-block 3-band `TPT-SVF` tone stack (`Block 0 -> Tone SVF -> Block 1 -> Block 2`).

2. **Direct-Register 5-Stage Topology (`0` Bypass Flags, `0` `feature_bus_scratch`, `ref[dsp.router.signal-flow]`):**
   - Executes all five stages unconditionally on every stereo sample: `Stage 0 (8-Port Router + LR4 + Op-P + Gate) -> 3 kHz ControlBus -> Stage I (2-Sec ZDF Env-SVF + Pre-Bus) -> Stage II (Zero-DC Matryoshka GRU + Tone SVF) -> Stage III (Contiguous 16/32/48/64 FIR + 4/8/12/16 Shared Kautz Pairs) -> Stage IV (4/8/8/16-Line fp16 FDN + 1/2/4/4 Taps)`.
   - Accumulates `pre_cab` and `direct` linear `VarPro` readouts directly in SIMD registers and evaluates unclipped linear paths up to $\pm 100.0$ (`+40 dBFS`, `REQ-7.1`) while keeping `GRU` states bounded in `[-2.5, 2.5]` and Stage IV `fp16` writes $\le \pm 65,504.0$.

3. **Inverted L1D-Resident Max-Quality Tier Sizing (`ref[arch.tiers.hardware-budget]`):**
   - Evicts transient `~14.8 KiB` kernel scratch onto the audio callback stack (`0 B/instance`), places cold `3 kHz` control-bus params at the trailing tail of the hot struct, uses `SimdVec4::rsqrt_pos_pair_nr2`, and guards `FtzDazGuard::drop` with `cur_fpcr != self.prev_fpcr` to eliminate redundant `msr fpcr` stalls and reduce timing jitter.
   - Maximizes Stage II (`[1,1,1] / [2,1,1] / [2,2,1] / [3,2,1]`), Stage III (`16/32/48/64` FIR + `4/8/12/16` Kautz pairs), and Stage IV (`4/8/8/16`-line FDN + `1/2/4/4` taps) inside the `64 KiB` Cortex-A76 L1D cache and `<= 4,608 / 10,240 / 18,432 / 28,672 B` `64 B`-aligned `.dnam` arenas, verified at `158.64 / 196.97 / 452.70 / 531.14 ns/frame` Desktop NEON (`<= 220 / 300 / 500 / 700 ns`) and `>= 14x Lite / >= 8x Full / >= 4x Pro / >= 2x Ultra` safe serial pedalboard capacity on `1× RK3582 Cortex-A76 @ 2.4 GHz` at `80%` RT cap (`REQ-6.1`).

4. **Full-Bandwidth `2.0–2.5 s` `fp32` `.dnam-proj` Capsules & `< 6 s/chain` 4-Stage Solver (`ref[train.pipeline.topology]`):**
   - Replaces `250 ms` `18 kHz` `fp16` coresets with **`2.0–2.5 s` (`96,000–120,000` frames at `48 kHz`) full-bandwidth (`20 Hz–20 kHz`) `fp32` `INPC`/`STMC` capsules** (`~384–480 KiB/stem`, `< 20 MB` `REQ-5.4` project vault) plus `STAT` cached `StarCenter` Stage B trajectories and `uplo='U'` `f64` readout Gram stats.
   - Decomposes the solver into `Stage A (Physics Probe) -> Stage B (Joint 4-Tier Parallel Windowed-BPTT Backbone + 2-Pass Kautz OMP + Fused Radix-4 Welch LTAS Adjoint) -> Stage C (Nested 4-Tier VarPro Cholesky) -> Stage D (Parallel Multi-Knob FiLM+ToneSVF+CP1 + 3-Phase Audit) -> Stage E (f32 Readout Polish, ISS/Schur Stability Audit & Pack)`, pairing `< 150 ms` `Path 1` incremental vault edits (`0-ULP` `StarCenter` invariance) with raw-audio-free `Path 2` capsule re-pooling (`ref[train.resume.two-tier-execution]`).

5. **Lossless `0-ULP` `GRU` Hidden-Lane Sign-Flip Watermarking (`ref[sec.watermark.reciprocal-gauge]`):**
   - Replaces `Role B` cascade gauges with an algebraically exact **`0-ULP` odd-activation `GRU` hidden-lane sign-flip gauge** ($s_i \in \lbrace -1, +1 \rbrace$ applied symmetrically to hidden-lane input rows, recurrent rows/columns, Zero-DC bias shift, and downstream readouts since $\tanh(-x) = -\tanh(x)$ and $h_{\text{eq}} = 0$) combined with Stage III Kautz reciprocal/phase gauges, `SO(4)` `CP1` rotation, and EMR-masked Coupled Tardos codes surviving `strip_container_to_tier`.

---

## 4. Carried-Forward Invariants from Superseded ADRs

Superseding `ADR-0004`, `ADR-0008`, `ADR-0011`, and `ADR-0017` retires their obsolete `v2` internal topologies (`PolyADAA` cascade, `Role B` blocks, `256-D` feature bus scatter/gather, `Wide192` Kronecker head solve, `250 ms` `fp16` coresets), while explicitly carrying forward into TORA `v3`:
- **From `ADR-0004`:** Family 5 ZOH Bitcrusher nearest-step quantizer (halfway ties to $+\infty$), `2^32` Hz integer ZOH phase accumulator, `u128` reciprocal rate transfer (`ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`), and the $C^2$ cubed-envelope wavefolder (`ref[dsp.op-n.wavefolder.cubic-c2]`) used in Stage I/pre-processing.
- **From `ADR-0008` & `ADR-0017`:** Column-major upper-triangular (`uplo = 'U'`) `f64` Gram formulation, compensated wide-accumulator determinism (`ref[train.gram.accuracy-contract]`), `33`-band `LTAS` KKT regularization, `[STAB-ZERO-DC]` export enforcement, and `< 64 MiB` browser worker bounds (`REQ-4.3`).
- **From `ADR-0011`:** Stage 0–IV signal-chain integration contracts, partial-preset default restoration and disjoint-interval crossfading (`ref[dsp.ctrl.switch.conditional-batch]`), `SpilloverTrails` vs. `HardBypass` state machine (`ref[arch.topology.spillover-trails]`), captured hardware freeze (`ref[dsp.stage4.captured-freeze]`), and clamped control-bus modulation bounds on `theta_0..theta_15` (`ref[dsp.ctrl.dispatch.stage-laws]`).
