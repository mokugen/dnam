---
id: spec.dsp.op-n
title: "Op-N: Universal Nonlinear, Hysteretic & Commuted Cascade Operator"
status: canonical
version: "1.0.0"
domain: dsp
rule_prefixes:
  - "r[dsp.op-n."
requirements:
  - REQ-1.1
  - REQ-6.2
adrs:
  - ADR-0019
  - ADR-0021
---

# Op-N: Universal Nonlinear, Hysteretic & Matryoshka `GRU` Operator

This specification defines **Op-N (Nonlinear)** ([`packages/engine/src/operators/op_n.rs`](../../../packages/engine/README.md), [`packages/capture/src/`](../../../packages/capture/README.md)), the universal operator kernel governing TORA `v3` Stage II Zero-DC Matryoshka `GRU` (`[1,1,1] / [2,1,1] / [2,2,1] / [3,2,1]` at `1×`/`2×` ZOH scaling), division-free in-register `NR2` rational Padé `tanh`/`sigmoid` via `SimdVec4::rcp_ge1_nr2`, inter-block `3 kHz` `TPT-SVF` + `FiLM`, `C²` cubic-envelope wavefolding, 2D quadrature phase-space hysteresis, fixed-hertz ZOH sample-rate reduction and bitcrushing (`Family 5`), pre-stage `PolyADAA` reference profiles, resting Zero-DC equilibrium startup, and the `3 kHz` polynomial control engine across **Dynamic Neural Audio Modeler (DNAM)** ([`tora_architecture.md`](../architecture/tora_architecture.md), [`ADR-0021`](../../adr/0021-unified-matryoshka-gru-and-direct-register-stages.md)).

---

## 1. TORA `v3` `NR2` Rational Padé Activations & Reference `PolyADAA` Profile

### 1.1 Pre-Activation Clamping & `NR2` Padé Argument Bounding

`r[dsp.op-n.polyadaa.clamp]`

In TORA `v3` Stage II (`GRU`), pre-activations bind finite operands cleanly before `SimdVec4` Padé evaluation (`tanh_pade5_vec4` / `sigmoid_pade5_vec4`), clamping linear gain paths only at the `±1e9` safety boundary (`REQ-7.1`). For reference Stage I / legacy `PolyADAA` shaping with drive scale $d \in [0.25, 1.0, 4.0, 16.0]$, form $r_n = \frac{2}{3} d \cdot x[n]$ and clamped $u_n = \mathrm{clamp}(r_n, -1.0, +1.0)$, retaining unclamped $r_{n-1}$.

### 1.2 Division-Free In-Register `NR2` Rational Padé (`rcp_ge1_nr2`) & Midpoint Factorization

`r[dsp.op-n.polyadaa.midpoint]`

To evaluate $\tanh(x)$ and $\sigma(x)$ in 4-lane SIMD registers with **zero `FDIV`, zero `libm`, zero LUT gathers, zero GPR spills, exact `0-ULP` odd symmetry, and exact $\tanh(+0.0) = +0.0$**:
1. Clamp $|x|$ to $R = 4$ via the closed-form $C^1$-clamped rational Padé-`5/4` pair $P_5(x)/Q_4(x) = x(1 + n_1 x^2 + n_2 x^4)/(1 + d_1 x^2 + d_2 x^4)$ solved analytically from the `x^5` Maclaurin series matching of $\tanh(x)$ together with endpoint pinning $P_5(R) = Q_4(R)$ and $C^1$ zero-slope saturation $(P_5/Q_4)'(R) = 0$:
   $$d_1 - n_1 = \frac{1}{3}, \qquad d_2 - n_2 - \frac{1}{3}d_1 = -\frac{2}{15}, \qquad R + R^3 n_1 + R^5 n_2 = 1 + R^2 d_1 + R^4 d_2, \qquad 1 + 3R^2 n_1 + 5R^4 n_2 = 2R d_1 + 4R^3 d_2$$
   At $R = 4$, this yields the exact rationals $(n_1, n_2, d_1, d_2) = (61/600,\ 11/19200,\ 87/200,\ 47/3840)$ evaluated at compile time (`ref[repo.code.constant-eval-and-derivations]`).
2. Evaluate $Q(x^2)^{-1}$ via **`SimdVec4::rcp_ge1_nr2`**: subtract normalized IEEE-754 `binary32` bits from the Chebyshev minimax reciprocal seed `0x7EF3_11C3` (`0x7E80_0000 +` $\mathrm{round}\big((2\sqrt{6} - 4)\cdot 2^{23}\big)$) (derived by balancing the relative seed error of $(1 + f)^{-1}$ across $f \in [0, 1)$) in vector registers (`0` `.to_array()`/`from_array()` GPR transfers) and execute `2` Newton-Raphson refinement steps (`NR2`, relative error $< 3.5 \times 10^{-6}$). Likewise, inverse-square-root helpers centralize the global two-binade minimax seeds (`0x5F37_59DF` for `f32`, `0x5FE6_EB50_C7B5_37A9` for `f64`).
3. For reference interior `PolyADAA` midpoint endpoints $u_n, u_{n-1} \in [-1, 1]$ with $m_n = \frac{1}{2}(u_n + u_{n-1})$ and $d_n = \frac{1}{2}(u_n - u_{n-1})$:
$$\mathcal{A}_{\text{odd}}[n] = m_n \Big(1.5 - 0.5\big(m_n^2 + d_n^2\big)\Big), \qquad \mathcal{A}_{\text{even}}[n] = \left(m_n^2 + \frac{1}{3}d_n^2\right) - \left(m_n^4 + 2 m_n^2 d_n^2 + \frac{1}{5}d_n^4\right)$$

#### Saturated-Curve Rail Crossing (6-Step Division-Free Reciprocal)

`r[dsp.op-n.polyadaa.rail-crossing]`

When consecutive raw inputs $a = \min(r_{n-1}, r_n) < b = \max(r_{n-1}, r_n)$ strictly cross $\pm 1$, partition the interval length $b - a$ into interior length $\ell_0 = \max(0, \min(b, 1) - \max(a, -1))$ and rail-plateau lengths $\ell_- = \max(0, \min(b, -1) - a)$, $\ell_+ = \max(0, b - \max(a, 1))$:
$$A_{\text{odd,sat}} = \big(\ell_0 \mathcal{A}_{\text{odd}} + \ell_+ - \ell_-\big) \cdot y, \qquad A_{\text{even,sat}} = \big(\ell_0 \mathcal{A}_{\text{even}}\big) \cdot y$$
Compute $y \approx (b - a)^{-1}$ without hardware division (`0 FDIV`) by widening $a, b$ bit-preservingly to `f64`, seeding $y_0 = 2^{-E_d - 1}$ from the unbiased exponent $E_d$ of $d = \mathrm{fl}_{64}(b - a)$, and executing 6 Newton reciprocal steps $y_{k+1} = \mathrm{fl}_{64}(y_k \cdot \mathrm{fl}_{64}(2 - \mathrm{fl}_{64}(d \cdot y_k)))$, guaranteeing absolute error $< 2^{-44}$ in `f64` and $< 2^{-24} + 2^{-44}$ after `f32` rounding.

`r[dsp.op-n.polyadaa.stereo-and-rail-guard]` **Highest-Drive Rail-Crossing Fast Path & Packed Stereo `V1A`/`V1B` Execution ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)):**
1. **Highest-Drive-Lane Rail Guard:** Because the drive bank $d_{\text{bank}} = [0.25\times, 1.0\times, 4.0\times, 16.0\times]$ is strictly increasing, if the highest active lane (`lane 3 = 16.0x` in `V1A` or `lane 2 = 4.0x` in `V1B`) satisfies $|r_{n-1}| \le 1.0$ and $|r_n| \le 1.0$, all lower-drive lanes are guaranteed interior and skip the `f64` 6-step Newton rail-crossing path with a single scalar comparison.
2. **Interleaved Stereo `V1A` & Packed 4-Lane `[l_1x, l_4x, r_1x, r_4x]` `V1B`:** Sub-Stage 2.1 (`step_stereo_pair`) interleaves Left and Right 4-drive `B::Vec4` registers to hide FMA latency, whereas Sub-Stage 2.2 (`step_stereo_lanes_1_2_lr`) packs only the two downstream-consumed `1.0x` and `4.0x` drive lanes (`lanes 1..3`) for Left and Right into a single 4-lane `B::Vec4` vector `[l_1x, l_4x, r_1x, r_4x]` and updates `left.{u_prev, a_prev, m_dc, last_a_even}[1..3]` and `right.{u_prev, a_prev, m_dc, last_a_even}[1..3]`, cutting `V1B` SIMD polynomial evaluations in half while preserving both channels' canonical 4-drive lane layout.

### 1.3 Un-Delayed Linear Separation & $1\times$ Passive B-Spline Blend

1. `r[dsp.op-n.polyadaa.linear-residual]` **Un-Delayed Linear Residual Path:** Evaluate $\alpha_c x_c[n]$ directly without ADAA filter delay, preserving unity passband gain within $< 10^{-12}\text{ dB}$ up to $22\text{ kHz}$ at $f_s = 48\text{ kHz}$.
2. `r[dsp.op-n.polyadaa.bspline-ac]` **Passive Quadratic B-Spline Blend ($0.875, 0.125$) + AC Coupling ($R_{\text{ac}} = 0.9993$ at $48\text{ kHz}$):**
   $$\tilde{y}_{\text{nl}}[n] = 0.875 \mathcal{A}[n] + 0.125 \mathcal{A}[n-1], \qquad m_{\text{dc}}[n] = 0.9993 m_{\text{dc}}[n-1] + 0.0007 \tilde{y}_{\text{nl}}[n]$$
   $$y[n] = \alpha_c x[n] + \big(\tilde{y}_{\text{nl}}[n] - m_{\text{dc}}[n]\big)$$
3. `r[dsp.op-n.polyadaa.drive-bank]` **Multi-Drive Bank (`last_a_odd` & `last_a_even`):** Evaluated in parallel over fixed drive scales $d_{\text{bank}} = [0.25\times, 1.0\times, 4.0\times, 16.0\times]$, caching both odd (`last_a_odd: [f32; 4]`) and even (`last_a_even: [f32; 4]`) interval-average antiderivative outputs per channel for downstream cascade and feature-bus projection.

---

## 2. Wavefolder, 2D Hysteresis & ZOH Bitcrusher

### 2.1 Wavefolder Rail Regularity and Candidates

`r[dsp.op-n.wavefolder.cubic-c2]`

To ensure $C^2$ continuity at the $\pm 1$ rails (vanishing value, first derivative, and second derivative at $\bar u = \pm 1$), `Op-N` uses the cubed-envelope odd and even wavefolders with $\bar u = \mathrm{clamp}(u, -1, 1)$:
$$w_{\text{odd},3}(u) = \bar u\big(1 - \bar u^2\big)^3, \qquad w_{\text{even},3}(u) = \bar u^2\big(1 - \bar u^2\big)^3$$

For consecutive clamped inputs $u_0, u_1 \in [-1, 1]$ with $m = (u_0 + u_1)/2$ and $d = (u_1 - u_0)/2$, define the division-free interval-average basis polynomials $Q_j(m, d)$ for $j \in [1, 8]$:
$$Q_j(m,d) = \sum_{k=0}^{\lfloor j/2\rfloor} \binom{j}{2k} m^{j-2k} d^{2k} (2k+1)^{-1}$$
$$A_{\text{odd},3} = Q_1 - 3Q_3 + 3Q_5 - Q_7, \qquad A_{\text{even},3} = Q_2 - 3Q_4 + 3Q_6 - Q_8, \qquad \hat w_{\text{odd},3}[n] = u_1 + A_{\text{odd},3} - m$$

### 2.2 2D Quadrature Phase-Space Hysteresis

`r[dsp.op-n.hysteresis.quadrature]`

Magnetic core transformer hysteresis and optocoupler memory are modeled via phase-space coordinates $(s_{\text{re}}[n], s_{\text{im}}[n])$:
$$h_{\text{hyst}}[n] = s_{\text{re}}[n] \cdot \big(s_{\text{re}}[n] - s_{\text{re}}[n-1]\big) + \kappa_{\text{quad}} \cdot s_{\text{re}}[n] s_{\text{im}}[n]$$

### 2.3 Family 5: Precomputed-Reciprocal ZOH Sample-Rate Reducer & Bitcrusher

1. `r[dsp.op-n.bitcrusher.quantizer]` **Uniform Nearest-Step Quantizer (Halfway Ties to $+\infty$):** For integer bit depth $B \in [2, 16]$, precompute $\Delta = 2^{-(B-1)}$ and $D = 2^{B-1}$ on the cold path:
   $$Q_B(x) = \lfloor xD + 0.5 \rfloor \Delta$$
   In `f32` over the certified `[-100, 100]` envelope, compute $v = xD$, truncate $v$ toward zero to `i32` $t$, form floor $k = t - \mathbf{1}[v < t]$ and residual $f = v - k$, and return $(k + \mathbf{1}[f \ge 0.5])\Delta$ (avoiding negative truncation errors and `0.5f32` pre-addition rounding overflow).
2. `r[dsp.op-n.bitcrusher.zoh-phase]` **Fixed-Hertz Integer ZOH Phase Profile (`0 FDIV`):** Using $F = 2^{32}$ units per Hz, prepare $D = \mathrm{round}_{\text{even}}(F f_s)$ and $I = \mathrm{round}_{\text{even}}(F f_{\text{target}})$ ($0 \le I \le D$). Advance unsigned 64-bit phase $P \in [0, D)$ per sample:
   $$A = P + I, \qquad b = \mathbf{1}[A \ge D], \qquad P \leftarrow A - bD$$
   When $b = 1$, latch $Q_B(x[n])$; at full rate ($I = D$), every sample latches ($b = 1$) while $P$ is unchanged; at zero rate ($I = 0$, exposed only when observed in the signed control domain), hold the last latched sample and phase.
3. `r[dsp.op-n.bitcrusher.rate-transfer]` **Division-Free Effective-Rate Phase Transfer:** Precompute $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$ as a `u64` on the cold path. On a sample-rate change to $D_{\text{new}}$, form $N = P_{\text{old}} D_{\text{new}} < 2^{102}$ in `u128`, compute $q_0 = \lfloor N R / 2^{104} \rfloor$ via shifts, correct remainder $r_0 = N - q_0 D_{\text{old}}$, round ties-to-even, and clamp to $D_{\text{new}} - 1$ so imminent carries remain imminent:
   $$P_{\text{new}} = \min\left(\mathrm{round}_{\text{even}}\left(\frac{P_{\text{old}}D_{\text{new}}}{D_{\text{old}}}\right), D_{\text{new}}-1\right), \qquad q_0 = (h \mathbin{\mathrm{<<}} 24) \mathbin{\mathrm{or}} \big((m \mathbin{\mathrm{and}} M) \mathbin{\mathrm{>>}} 40\big)$$
   where $N = N_{\text{hi}}2^{64} + N_{\text{lo}}$, $a = N_{\text{lo}}R$, $c = N_{\text{hi}}R$, $M = 2^{64} - 1$, $m = (a \mathbin{\mathrm{>>}} 64) + (c \mathbin{\mathrm{and}} M)$, and $h = (c \mathbin{\mathrm{>>}} 64) + (m \mathbin{\mathrm{>>}} 64)$, bounding fixed-rate frequency error to $|f_s(I/D) - f_{\text{target}}| \le 2^{-32}\text{ Hz}$.
4. `r[dsp.op-n.bitcrusher.surrogate]` **Forward-Faithful Training Surrogate:** Return the hard runtime quantizer output in the forward pass with grouped stop-gradient $\hat q(x) = \mathrm{sg}[Q_B(x)] + (x - \mathrm{sg}[x])$, or use the 4-term Fourier backward surrogate during warmup:
   $$q_4(x) = x - \frac{\Delta}{\pi}\sum_{k=1}^{4}\frac{(-1)^{k-1}}{k}\sin\left(\frac{2\pi k x}{\Delta}\right)$$

---

## 3. Exact LTI Basis Commutativity Theorem (Theorem 1)

### 3.1 Formal Theorem Statement & Proof

`r[dsp.op-n.commuted.theorem1]`

While TORA `v2` applied **Theorem 1** across a fixed-drive `PolyADAA` basis and LTI filter bank (`y_comm[n] = Sum_k L_k(u_nl[n] * W_k)`), **TORA `v3` promotes Stage II to a Direct-Register Matryoshka `GRU` + `3 kHz` `TPT-SVF` architecture** while retaining Theorem 1's causal linear-operator property for linear `VarPro` `pre_cab` and `direct` output-readout collapse (`3 kHz` control weight folding into direct SIMD registers without `256-D` `feature_bus_scratch`):

**Theorem 1 (Exact LTI Basis Commutativity).**  
Let $\mathbf{u}[n] \in \mathbb{R}^P$ be bounded per-channel block activations or inputs ($\sup_{n, p} |u_p[n]| \le U_{\max} < \infty$), $\lbrace \mathcal{L}_k \rbrace_{k=0}^{K-1}$ be causal, stable LTI filters with impulse responses $h_k \in \ell_1(\mathbb{Z})$, and $\lbrace \mathbf{W}_k \rbrace_{k=0}^{K-1} \subset \mathbb{R}^{P \times C}$ be constant real projection matrices. Then for every channel $c \in [0, C-1]$ and sample $n \in \mathbb{Z}$:
$$\sum_{k=0}^{K-1} \Big(\mathcal{L}_k\big(\mathbf{u}\big)[n]\Big) \mathbf{W}_k \equiv \sum_{k=0}^{K-1} \mathcal{L}_k\Big(\mathbf{u} \mathbf{W}_k\Big)[n]$$

*Proof.* Because $h_k \in \ell_1(\mathbb{Z})$ and $\mathbf{u}$ is bounded, the double sum converges absolutely, permitting summation interchange by Fubini's theorem so $\mathbf{u}\mathbf{W}_k$ filters directly on $C$ SIMD lanes. $\blacksquare$

### 3.2 Causality Audit & Elimination of Acausal Lookahead

`r[dsp.op-n.commuted.causality]`

All delay offsets and `poly3` decimation / Stage II–III interfaces in TORA `v3` are strictly causal ($d \ge 0$), preserving sample-accurate alignment across `1×` (`Lite`/`Full`) and `2×` (`Pro`/`Ultra`) paths.

### 3.3 Commutator Perturbation Under $3\text{ kHz}$ Control-Rate Modulation

`r[dsp.op-n.commuted.project-then-filter]`

Under time-varying control modulation at the `3 kHz` boundary (`FiLM`, Zero-DC $b_n$, `TPT-SVF` coefficients, and linear `VarPro` readouts), all coefficient deltas micro-ramp across $K_{\text{ctrl}}$ samples, matching forward evaluation in [`packages/capture/`](../../../packages/capture/README.md) with zero train/runtime divergence:
$$\Delta_{\text{comm}}[n] \le \frac{B_u \nu_{\text{ctrl}}}{f_s} \cdot \frac{r_k}{1 - r_k}$$

### 3.4 Boundary Conditions, Resting DC Equilibrium & Pop-Free Startup

`r[dsp.op-n.commuted.dc-equilibrium]`

Given Stage II Zero-DC Matryoshka `GRU` resting bias construction ($h_t = (1 - z_t) \odot h_{t-1} + z_t \odot n_t$), setting candidate rest offset $b_n = -W_{n} u_{\text{rest}} - b_{hn}$ (with $h_{\text{eq}} = 0$ at rest when user input $x = 0$) paired with `NR2` odd-symmetric $\tanh(+0.0) = +0.0$ guarantees exact `0.0f32` DC rest equilibrium without transient startup pops, and any auxiliary tone/AC filter initializes cleanly to zero rest:
1. **Stage II Zero-DC `GRU` Hidden State:** $h_{\text{init}} = 0.0\text{f32}$ (`12 / 16 / 20 / 24` ch per channel).
2. **`3-Band TPT-SVF` & AC Filters:** Initialize integrator states $s_1 = s_2 = 0.0\text{f32}$ at resting zero equilibrium.

---

## 4. Stage II Zero-DC Matryoshka `GRU` & Inter-Block Tone Stack

`r[dsp.op-n.stage2.sub-stages]`

Stage II models vacuum tube preamps, asymmetric drives, power-amp compression, and interactive tone stacks via a **3-block tapered feed-forward Zero-DC Matryoshka `GRU` pyramid** (`Block 0 -> Inter-Block Tone TPT-SVF -> Block 1 -> Block 2`) running natively at **`1×` (`48 kHz`) in `Lite` (`[1,1,1] = 12 ch`) and `Full` (`[2,1,1] = 16 ch`)** and **`2×` (`96 kHz`, `4×` bounce) in `Pro` (`[2,2,1] = 20 ch`) and `Ultra` (`[3,2,1] = 24 ch`)**:
1. **Continuous-Time ZOH Gate Scaling & `1×` Aliasing Guard:** Stores continuous-time update-gate retention rates $1 - z_{\text{eff}}(q) = (1 - z_{1\times})^{1/q}$ so `Lite` and `Full` run natively at `1×` using `V3Rate1xCalibration` (`up_zoh_mix` + low-delay `poly3` decimation) while sharing prefix weights with `2×` `Pro` and `Ultra`.
2. **Bounded ISS / Finite State Invariant:** Every `GRU` hidden lane stays bounded in `[-2.5, 2.5]` with spectral contraction under zero input, while linear `pre_cab` and `direct` `VarPro` bypass/readout paths carry unclipped audio signals across `±100.0` (`+40 dBFS`, `REQ-7.1`).

### 4.1 Four-Filter Tone Basis (Refinement 3)

`r[dsp.op-n.stage2.tone-basis]`

Between `Block 0` and `Block 1` of the Stage II Matryoshka `GRU`, TORA `v3` steps a **3-band Trapezoidal State-Variable Filter (`TPT-SVF`) tone stack** (`Lowpass`, `Bandpass`, `Highpass`) whose normalized frequencies and gains refresh at `3 kHz` (micro-ramped over $K_{\text{ctrl}}$ samples, `0` per-sample transcendental or `FDIV` calls), replacing the `v2` `[Direct, LPF_80, LPF_400, HPF_1200]` fixed-pole bank while retaining the exact 4-basis structure at rest:
$$H_{\text{tone},c}(z) = g_{\text{lp}} L_{\text{svf}}(z) + g_{\text{bp}} B_{\text{svf}}(z) + g_{\text{hp}} H_{\text{svf}}(z) + g_{\text{dir}}$$

### 4.2 Stereo Basis, Role A Inter-Stage Coupling & Surjective Role B Feature-Bus Folding

`r[dsp.op-n.stage2.stereo-coupling]`

Stage II preserves independent Left and Right recurrent state trajectories alongside optional cross-channel stereo coupling, retiring legacy `v2` `Role A / Role B` `stage2_cascade_w` block folding in favor of direct-register `pre_cab` (`Stage II -> Stage III`) and `direct` (`Stage II -> Out`) `VarPro` stereo linear accumulators (`0` intermediate scatter/gather copies into `feature_bus_scratch`).
- `r[dsp.op-n.stage2.conv6-simd]` **Stereo `L + R` `4×4` Tile Interleaving (`step_stage2_stereo`) & `NR2` Vectorization ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0021`](../../adr/0021-unified-matryoshka-gru-and-direct-register-stages.md)):** `Stage2Backend` steps `Left` (`SimdVec4`) and `Right` (`SimdVec4`) hidden blocks side by side inside `step_stage2_stereo`, loading each `4×4` recurrent and input weight tile once from `Arena 0..3` into vector registers and dual-issuing `L` and `R` FMAs across both SIMD FMA pipes before applying `NR2` Padé `tanh`/`sigmoid` with zero NEON-to-GPR spills.

### 4.3 $3\text{ kHz}$ Bilinear Rank-$R$ Stage Latent Folding

`r[dsp.op-n.stage2.latent-folding]`

At each $3\text{ kHz}$ control boundary, stage latents $\mathbf{z}_{\text{stage}} \in \mathbb{R}^{D_{\text{stage}}}$ ($D_{\text{stage}} = 4$ in `Lite/Full/Pro`, $8$ in `Ultra`) modulate Stage II's per-block **`FiLM` gain/shift vectors**, **Zero-DC $b_n$ shift**, **inter-block `TPT-SVF` cutoff/band gains**, and **direct `VarPro` linear readouts** (`Arena 3` holds `Ultra` extensions):
$$\boldsymbol{\gamma}_{\text{film}}(\mathbf{z}_{\text{stage}}) = \boldsymbol{\gamma}_0 + \sum_{r=1}^{D_{\text{stage}}} z_{\text{stage}, r} \Delta\boldsymbol{\gamma}_r, \qquad \mathbf{W}_{\text{ro}}(\mathbf{z}_{\text{stage}}) = \mathbf{W}_{\text{ro},0} + \sum_{r=1}^{D_{\text{stage}}} z_{\text{stage}, r} \mathbf{W}_{\text{ro}, r}$$

---

## 5. `Op-N[Ctrl]`: 3 kHz Algebraic Polynomial & $O(1)$ Sufficient-Statistic Engine

1. `r[dsp.op-n.ctrl.incremental-update]` **Incremental Sufficient-Statistic Update:** When contributor $p \in \mathcal{I}_{\text{moved}}$ changes from $u_{p,\text{old}}$ to $u_{p,\text{new}}$, update the exact-integer main-effect and Rank-$R$ pair-statistic accumulators (`64 B` in `Lite..Pro`, `128 B` in `Ultra`; see [`parametric_control_bus.md` §2.3](parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling)):
   $$\Delta\mathbf{z}_{\text{main}} = \mathbf{W}_{\text{main}, p} \cdot \big(\boldsymbol{\psi}_p(u_{p, \text{new}}) - \boldsymbol{\psi}_p(u_{p, \text{old}})\big)$$
2. `r[dsp.op-n.ctrl.lobatto-basis]` **Node-Anchored Control Basis:** Evaluate the persisted observed anchor nodes and switch contributor slices ([`parametric_control_bus.md` §2.2](parametric_control_bus.md#22-cardinality-bounded-node-anchored-newtonlobatto-basis)).
3. `r[dsp.op-n.ctrl.chebyshev-lfo]` **Chebyshev LFO Waveshaping:** Shape quadrature rotor states via $T_1(x) = x$, $T_2(x) = 2x^2 - 1$, $T_3(x) = 4x^3 - 3x$, $T_4(x) = 8x^4 - 8x^2 + 1$.
4. `r[dsp.op-n.ctrl.fm-interaction]` **Rank-4/8 FM Bilinear Interaction:** Evaluate interaction factors from scalar embeddings $z_p = \mathbf{w}_{\text{proj},p}^\top\boldsymbol{\psi}_p(u_p)$ and mix via $\mathbf{U}_{\text{int}}$ (`8×4` base in `Arena 0` + three `8×4` `Ultra` delta blocks in `Arena 3`).
5. `r[dsp.op-n.ctrl.trust-clip]` **Groupwise Trust-Region Clipping:** Apply the outward-checked trust-region clip independently to $\mathbf{z}_{\text{head}}$ and $\mathbf{z}_{\text{stage}}$ ([`parametric_control_bus.md` §1.2](parametric_control_bus.md#outward-checked-trust-region-clip)).
