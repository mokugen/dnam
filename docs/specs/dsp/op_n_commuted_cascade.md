---
id: spec.dsp.op-n
title: "Op-N: Universal PolyADAA1, 1D Scalar ParaLRU & 4-Block Matryoshka SONA Core Operator"
status: canonical
version: "3.0.0"
domain: dsp
rule_prefixes:
  - "r[dsp.op-n."
requirements:
  - REQ-1.1
  - REQ-6.2
adrs:
  - ADR-0019
  - ADR-0022
---

# Op-N: Universal `PolyADAA1`, `1D` Scalar `ParaLRU` & `4`-Block Matryoshka `SONA` Core Operator

This specification defines **Op-N (Nonlinear & Parallel State-Space)** (shared `#![no_std]` `32-bit` kernels in `packages/kernels/` `dnam-kernels`, runtime implementation in [`packages/engine/src/operators/op_n.rs`](../../../packages/engine/README.md), and identification in [`packages/capture/src/`](../../../packages/capture/README.md)), retiring coupled `GRU` (`stage2_gru`) and normatively specifying the **`SONA` `v3.0` 4-Block Matryoshka Core** (`LiteSpec [1,1,1,1]`, `FullSpec [2,2,1,1]`, `ProSpec [3,2,2,1]`, `UltraSpec [4,3,2,1]` locked as compile-time `const` profiles at `Gate 0b`), **decoupled `1D` scalar `ParaLRU`** ($K = 3$ scalar Newton scans forward / $1$ linear scan backward on GPU), **pure-`f32` divided-difference `PolyADAA1`** (`(m_n, d_n)` cubic odd + quartic even + `C²` wavefolder + batched `rcp_pos_quad_nr2` rail-crossing correction, `0` `FDIV`, analytical `0.000000` DC equilibrium), lower-triangular `4×4` inter-block spatial mixing, **`StereoVec8` + `2× SimdVec4` `0-ULP` parity**, ZOH bitcrusher (`Family 5`), and **`Op-N[Ctrl]`** across **Dynamic Neural Audio Modeler (DNAM)** ([`sona_architecture.md`](../architecture/sona_architecture.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

## 1. Pure-`f32` Divided-Difference `PolyADAA1` & Antialiased Activations

### 1.1 Pre-Activation Clamping & Interior/Rail Argument Bounding

`r[dsp.op-n.polyadaa.clamp]`

Inside every `SONA` core block, linear signal paths clamp only at the `±1e9` safety boundary (`REQ-7.1`). For `PolyADAA1` shaping with per-lane drive scale $d$, form raw pre-activation $r_n = \frac{2}{3} d \cdot u_n^{\mathrm{raw}}$ (retaining the previous unclamped endpoint $r_{n-1}$) and interior clamped coordinates $u_n = \mathrm{clamp}(r_n, -1.0, +1.0)$, preserving exact odd symmetry and $g_{\mathrm{odd}}(+0.0) = +0.0$.

### 1.2 Pure-`f32` Divided-Difference `PolyADAA1`, `rcp_pos_quad_nr2` & Midpoint Factorization

`r[dsp.op-n.polyadaa.midpoint]`

To evaluate antialiased nonlinear activations in 4-lane SIMD and WebGPU `32-bit` registers with **zero `FDIV`, zero `libm`, zero LUT gathers, zero GPR spills, exact `0-ULP` odd symmetry, and exact $f(+0.0) = +0.0$**:
1. **Interior `PolyADAA1` Midpoint / Half-Difference Form (`6` FMAs):** For interior clamped endpoints $u_n, u_{n-1} \in [-1, 1]$ with midpoint $m_n = \frac{1}{2}(u_n + u_{n-1})$ and half-difference $d_n = \frac{1}{2}(u_n - u_{n-1})$:
   $$\mathcal{A}_{\text{odd}}[n] = m_n \Big(1.5 - 0.5\big(m_n^2 + d_n^2\big)\Big), \qquad \mathcal{A}_{\text{even}}[n] = \left(m_n^2 + \frac{1}{3}d_n^2\right) - \left(m_n^4 + 2 m_n^2 d_n^2 + \frac{1}{5}d_n^4\right)$$
   At $r_n = r_{n-1} = u \in [-1, 1]$ ($d_n = 0$), this reduces identically to the memoryless cubic/quartic curves $g_{\mathrm{odd}}(u) = 1.5u - 0.5u^3$ and $g_{\mathrm{even}}(u) = u^2 - u^4$ with exact zero slope at the rail boundaries ($g_{\mathrm{odd}}'(\pm 1) = 0$).
2. **Chebyshev Minimax `rcp_pos_nr2` / `rcp_ge1_nr2` & Two-Binade `rsqrt` Seeds (`const_eval`):** For normalized IEEE-754 `binary32` reciprocals (`rcp_pos_nr2`, `rcp_pos_quad_nr2`, and `rcp_ge1_nr2` in `dnam-kernels`), subtract the normalized significand bits from the compile-time Chebyshev minimax seed `0x7EF3_11C3` (`0x7E80_0000 + round((2*sqrt(6) - 4) * 2^23)`, derived by balancing the relative seed error of $(1 + f)^{-1}$ across $f \in [0, 1)$) in vector registers (`0` GPR transfers) and execute `2` Newton-Raphson refinement steps (`NR2`, relative error $< 3.5 \times 10^{-6}$). Inverse-square-root helpers centralize the global two-binade minimax seeds (`0x5F37_59DF` for `f32`, `0x5FE6_EB50_C7B5_37A9` for `f64`), alongside the documented $C^1$-clamped rational Padé-`[5/4]` constants $(n_1, n_2, d_1, d_2) = (61/600, 11/19200, 87/200, 47/3840)$ solved from $d_1 - n_1 = 1/3$, $d_2 - n_2 - d_1/3 = -2/15$, $P_5(R) = Q_4(R)$, and $(P_5/Q_4)'(R) = 0$ at $R = 4$ (`ref[repo.code.constant-eval-and-derivations]`).

#### Saturated-Curve Rail Crossing (Pure-`f32` Batched `rcp_pos_quad_nr2`)

`r[dsp.op-n.polyadaa.rail-crossing]`

When consecutive raw inputs $a = \min(r_{n-1}, r_n) < b = \max(r_{n-1}, r_n)$ strictly cross $\pm 1$, partition the interval length $b - a$ into interior length $\ell_0 = \max(0, \min(b, 1) - \max(a, -1))$ and rail-plateau lengths $\ell_- = \max(0, \min(b, -1) - a)$, $\ell_+ = \max(0, b - \max(a, 1))$, evaluating $\mathcal{A}_{\text{odd}}, \mathcal{A}_{\text{even}}$ over the clamped interior interval $[\max(a, -1), \min(b, 1)]$:
$$A_{\text{odd,sat}} = \big(\ell_0 \mathcal{A}_{\text{odd}} + \ell_+ - \ell_-\big) \cdot y, \qquad A_{\text{even,sat}} = \big(\ell_0 \mathcal{A}_{\text{even}}\big) \cdot y$$
Compute $y \approx (b - a)^{-1}$ directly in **pure `f32`** via `rcp_pos_nr2(max(b - a, 2^-12))` without hardware division (`0 FDIV`, `0` `0/0` NaNs; unconditionally accurate in `binary32` because $g_{\mathrm{odd}}'(\pm 1) = 0$ keeps divided differences across $\pm 1$ $C^1$-smooth with bounded condition number), using 6-step `f64` Newton reciprocals solely as an offline verification oracle.

`r[dsp.op-n.polyadaa.stereo-and-rail-guard]` **Amortized Fast-Path `PolyADAA1` & Batched `rcp_pos_quad_nr2` Stereo Execution ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
1. **Interior `6`-FMA Fast Path (`|u| <= 1`):** Whenever all lanes of a `SimdVec4` / `StereoVec8` register satisfy $|r_{n-1}| \le 1.0$ and $|r_n| \le 1.0$, `PolyADAA1` evaluates strictly via `6` branch-free `SimdVec4` FMAs (`0` reciprocal evaluations).
2. **Batched `rcp_pos_quad_nr2` Rail-Crossing Path:** Whenever any lane in the stereo pair crosses $\pm 1$, batch the rail-crossing interval reciprocals across Left/Right odd, even, and $C^2$ wavefolder vectors via `SimdVec4::rcp_pos_quad_nr2` (`0` GPR transfers, `0` per-lane branches, identical execution in `dnam-kernels` CPU and WGSL shaders).

### 1.3 Un-Delayed Linear Separation & $1\times$ Passive B-Spline Blend

1. `r[dsp.op-n.polyadaa.linear-residual]` **Un-Delayed Linear Residual Path:** Evaluate $\alpha_c x_c[n]$ directly without ADAA filter delay, preserving unity passband gain within $< 10^{-12}\text{ dB}$ up to $22\text{ kHz}$ at $f_s = 48\text{ kHz}$.
2. `r[dsp.op-n.polyadaa.bspline-ac]` **Passive Quadratic B-Spline Blend ($0.875, 0.125$) + AC Coupling ($R_{\text{ac}} = 0.9993$ at $48\text{ kHz}$):**
   $$\tilde{y}_{\text{nl}}[n] = 0.875 \mathcal{A}[n] + 0.125 \mathcal{A}[n-1], \qquad m_{\text{dc}}[n] = 0.9993 m_{\text{dc}}[n-1] + 0.0007 \tilde{y}_{\text{nl}}[n]$$
   $$y[n] = \alpha_c x[n] + \big(\tilde{y}_{\text{nl}}[n] - m_{\text{dc}}[n]\big)$$
3. `r[dsp.op-n.polyadaa.drive-bank]` **Multi-Drive `PolyADAA1` Lane Bank (`odd` & `even` + wavefolder):** Evaluated in parallel across `PolyADAA1` lanes in each `SONA` core block (and reference multi-drive profile $d_{\text{bank}} = [0.25\times, 1.0\times, 4.0\times, 16.0\times]$), retaining previous endpoints `(r_prev, u_prev)` for exact `1×` (`48 kHz`) and `2×` (`96 kHz`) interval-average antialiasing (`last_a_odd`, `last_a_even`).

---

## 2. Wavefolder, 2D Hysteresis & ZOH Bitcrusher

### 2.1 Wavefolder Rail Regularity and Candidates

`r[dsp.op-n.wavefolder.cubic-c2]`

To ensure $C^2$ continuity at the $\pm 1$ rails (vanishing value, first derivative, and second derivative at $\bar u = \pm 1$), `Op-N` pairs `PolyADAA1` odd/even saturation with the cubed-envelope odd and even wavefolders (`eval_adaa_cubic_c2_wavefolder_odd_f32` in `dnam-kernels`) using $\bar u = \mathrm{clamp}(u, -1, 1)$:
$$w_{\text{odd},3}(u) = \bar u\big(1 - \bar u^2\big)^3, \qquad w_{\text{even},3}(u) = \bar u^2\big(1 - \bar u^2\big)^3$$

For consecutive clamped inputs $u_0, u_1 \in [-1, 1]$ with $m = (u_0 + u_1)/2$ and $d = (u_1 - u_0)/2$, define the division-free interval-average basis polynomials $Q_j(m, d)$ for $j \in [1, 8]$ (using batched `rcp_pos_quad_nr2` on rail-crossing segments):
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

`SONA` `v3.0` applies **Theorem 1** to collapse linear `3 kHz` `VarPro` readout subspace projections (`pre_cab` feeding Stage III and `direct` feeding stereo output) directly inside SIMD registers across the 4-block `SONA` core (`TPT-SVF -> ParaLRU -> PolyADAA1`) with zero intermediate `256-D` `feature_bus_scratch` gather buffers:

**Theorem 1 (Exact LTI Basis Commutativity).**  
Let $\mathbf{u}[n] \in \mathbb{R}^P$ be bounded per-channel block activations or inputs ($\sup_{n, p} |u_p[n]| \le U_{\max} < \infty$), $\lbrace \mathcal{L}_k \rbrace_{k=0}^{K-1}$ be causal, stable LTI filters with impulse responses $h_k \in \ell_1(\mathbb{Z})$, and $\lbrace \mathbf{W}_k \rbrace_{k=0}^{K-1} \subset \mathbb{R}^{P \times C}$ be constant real projection matrices. Then for every channel $c \in [0, C-1]$ and sample $n \in \mathbb{Z}$:
$$\sum_{k=0}^{K-1} \Big(\mathcal{L}_k\big(\mathbf{u}\big)[n]\Big) \mathbf{W}_k \equiv \sum_{k=0}^{K-1} \mathcal{L}_k\Big(\mathbf{u} \mathbf{W}_k\Big)[n]$$

*Proof.* Because $h_k \in \ell_1(\mathbb{Z})$ and $\mathbf{u}$ is bounded, the double sum converges absolutely, permitting summation interchange by Fubini's theorem so $\mathbf{u}\mathbf{W}_k$ filters directly on $C$ SIMD lanes. $\blacksquare$

### 3.2 Causality Audit & Elimination of Acausal Lookahead

`r[dsp.op-n.commuted.causality]`

All `Op-T` Top-of-Frame Crossbar reads ($\tau \ge 4.0$ samples), `TPT-SVF` / `ParaLRU` / `PolyADAA1` states, `poly3` decimation filters, and Stage III / output interfaces in `SONA` `v3.0` are strictly causal ($d \ge 0$), preserving sample-accurate alignment across `1×` and `2×` (`4×` bounce) execution paths.

### 3.3 Commutator Perturbation Under $3\text{ kHz}$ Control-Rate Modulation

`r[dsp.op-n.commuted.project-then-filter]`

Under time-varying control modulation at the `3 kHz` boundary (`TPT-SVF` prewarped `(a1, a2, a3)` and tap gains, `ParaLRU` `(alpha_rel, alpha_atk, kappa)`, `PolyADAA1` drive/bias, and `fp16` `W_readout` subspace collapse), all active coefficient deltas micro-ramp across $K_{\text{ctrl}}$ samples, matching GPU/CPU forward evaluation in [`packages/capture/`](../../../packages/capture/README.md) with zero train/runtime divergence:
$$\Delta_{\text{comm}}[n] \le \frac{B_u \nu_{\text{ctrl}}}{f_s} \cdot \frac{r_k}{1 - r_k}$$

### 3.4 Boundary Conditions, Resting DC Equilibrium & Pop-Free Startup

`r[dsp.op-n.commuted.dc-equilibrium]`

With decoupled `1D` scalar `ParaLRU` recurrence ($v_i[n] = \alpha_{\text{rel},i} v_i[n-1] + (1 - \alpha_{\text{atk},i})\phi_{\text{rect}}(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1])$ where $\phi_{\text{rect}}(0) = 0$) paired with analytical zero-DC resting offset centering at `StarCenter` ($u_{\text{eq}} = 0.0\text{f32}$), odd/even `PolyADAA1` ($g_{\mathrm{odd}}(+0.0) = +0.0$, $g_{\mathrm{even}}(+0.0) = +0.0$), and zero-initialized `TPT-SVF` integrators, every recursive lane settles at analytical **`0.000000` DC equilibrium (`h_eq == +0.0f32`, `b_head == 0.0`)** with zero startup pops:
1. **`4`-Block `SONA` `ParaLRU` State:** $v_{\text{init}} = +0.0\text{f32}$ (`16 / 24 / 32 / 40` scalar lanes per channel across `Lite / Full / Pro / Ultra`).
2. **`TPT-SVF` & AC Filters:** Initialize integrator states $s_1 = s_2 = +0.0\text{f32}$ at resting zero equilibrium.

---

## 4. `4`-Block Matryoshka `SONA` Core (`TPT-SVF` $\to$ `ParaLRU` $\to$ `PolyADAA1`)

`r[dsp.op-n.stage2.sub-stages]`

`SONA` `v3.0` permanently retires the coupled `GRU` (`stage2_gru`) and models preamps, asymmetric drives, diodes, power-amp sag, crossovers, and resonant sweeps via a **4-block Matryoshka `SONA` State-Space Core** (`Block 0 -> Block 1 -> Block 2 -> Block 3`). Each block composes a passive **`2×2` ZDF `TPT-SVF` bank** $\to$ **decoupled `1D` scalar `ParaLRU`** $\to$ **pure-`f32` divided-difference `PolyADAA1`** (plus $C^2$ wavefolder lanes), monomorphized over **compile-time `const` `SonaTierSpec` profiles locked at `Gate 0b`** (`0` capture-time tier resizing):
- **`LiteSpec` (`[1, 1, 1, 1]`):** `16` lanes + `4` `TPT-SVF` modes @ `1×` `PolyADAA1` (or `[1, 1, 1, 0]` @ `2×` OS locked at `Gate 0b` inside `<= 1,000 ns` A76 / `<= 4.5 KiB` `SonaHotSlab`).
- **`FullSpec` (`[2, 2, 1, 1]`):** `24` lanes + `6` `TPT-SVF` modes @ `2×` OS (`96 kHz`) + `PolyADAA1` (`<= 7.8 KiB`).
- **`ProSpec` (`[3, 2, 2, 1]`):** `32` lanes + `8` `TPT-SVF` modes @ `2×` OS (`96 kHz`) + `PolyADAA1` (`<= 15.5 KiB`).
- **`UltraSpec` (`[4, 3, 2, 1]`):** `40` lanes + `10–12` `TPT-SVF` modes @ `2×` OS (`4×` Bounce) + `PolyADAA1` (`<= 30.0 KiB`).

1. **Decoupled `1D` Scalar `ParaLRU` Recurrence & $K=3$ Parallel Scan:**
   Each scalar lane $i$ updates strictly independently inside a block via a single decoupled self-rectification scalar $\kappa_i \in [0, 1]$, attack/release rates $\alpha_{\text{atk},i} \in [0, 1)$, $\alpha_{\text{rel},i} \in (0, r_{\max}]$ ($r_{\max} \le 0.999995$), and monotone bounded rectification $\phi_{\text{rect}}$ ($\phi_{\text{rect}}(0) = 0$, $0 \le \phi_{\text{rect}}'(z) \le 1$):
   $$v_i[n] = \alpha_{\text{rel},i} v_i[n-1] + (1 - \alpha_{\text{atk},i}) \phi_{\text{rect}}\big(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1]\big)$$
   Because every lane's Jacobian satisfies:
   $$J_i[n] = \frac{\partial v_i[n]}{\partial v_i[n-1]} = \alpha_{\text{rel},i}\Big(1 - \kappa_i (1 - \alpha_{\text{atk},i}) \phi_{\text{rect}}'\big(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1]\big)\Big) \in \big[\alpha_{\text{rel},i}(1 - \kappa_i(1 - \alpha_{\text{atk},i})), \alpha_{\text{rel},i}\big] \subset (0, r_{\max}] \subset (0, 1)$$
   `ParaLRU` is unconditionally contractive without post-hoc ISS clamping (`0` `w_rec *= 0.5`), executes in `6` `SimdVec4` operations on CPU, and evaluates on WebGPU via **$K = 3$ parallel scalar Newton Blelloch scans forward** and **$1$ parallel linear Blelloch scan backward** inside `16 KiB` `var<workgroup>` SRAM (`2 KiB` for `workgroup_size(256)`).
2. **Continuous-Time ZOH Retention & `1×` Antialiasing:** At effective rate ratio $\rho_{\text{sr}} = 48000 / f_{s,\text{eff}}$, rescale decay parameters via $\alpha_{\text{rel},i}(f_{s,\text{eff}}) = \alpha_{\text{rel},i,\text{ref}}^{\rho_{\text{sr}}}$ and $\alpha_{\text{atk},i}(f_{s,\text{eff}}) = \alpha_{\text{atk},i,\text{ref}}^{\rho_{\text{sr}}}$ alongside analytical `TPT-SVF` prewarping $g_m(f_{s,\text{eff}}) = \tan(\pi f_{c,m} / f_{s,\text{eff}})$ (`PolyADAA1` provides `1×` aliasing protection when `Lite` executes at `1×`, while `2×` OS provides `-90 dB` alias immunity on `Full..Ultra` with causal `poly3` decimation).
3. **Bounded State & Unclipped Envelope Invariant:** Every `ParaLRU` lane remains unconditionally BIBO-bounded under all finite inputs (`REQ-7.2`), while direct-register `VarPro` linear readout paths (`pre_cab` and `direct`) carry unclipped audio across `±100.0` (`+40 dBFS`, `REQ-7.1`).

### 4.1 Four-Filter Tone Basis (Refinement 3)

`r[dsp.op-n.stage2.tone-basis]`

Each passive `2×2` ZDF `TPT-SVF` mode inside the 4-block `SONA` core exposes all four orthogonal filter responses simultaneously (`LP = v_2`, `BP = v_1`, `HP = u - k v_1 - v_2`, `AllPass = u - 2 k v_1`; coefficients refreshed at `3 kHz` via analytical ZOH prewarping with zero audio-rate `FDIV` or `libm`), spanning arbitrary multi-band crossovers, resonant tone stacks, and `4–12`-pole phasers strictly through learned mixing weights (`REQ-1.9`):
$$H_{\text{tone},c}(z) = g_{\text{lp}} L_{\text{svf}}(z) + g_{\text{bp}} B_{\text{svf}}(z) + g_{\text{hp}} H_{\text{svf}}(z) + g_{\text{ap}} A_{\text{svf}}(z) + g_{\text{dir}}$$

### 4.2 Stereo Basis, Lower-Triangular `4×4` Inter-Block Mixing & `StereoVec8` Parity

`r[dsp.op-n.stage2.stereo-coupling]`

Inter-lane mixing across the 4-block `SONA` core occurs strictly **between blocks** via **lower-triangular `4×4` spatial mixing tiles** (`Lite ⊂ Full ⊂ Pro ⊂ Ultra` prefix nesting, preserving independent L/R trajectories alongside cross-channel stereo coupling), while direct-register `VarPro` linear readouts (`pre_cab` feeding Stage III and `direct` feeding stereo output) accumulate directly in SIMD registers with zero intermediate `feature_bus_scratch` materialization.

- `r[dsp.op-n.stage2.conv6-simd]` **Wide-ISA `StereoVec8` Dispatch (`AVX2+FMA`, `SVE2`, `SME`) + Interleaved `2× SimdVec4` (`NEON`, `WASM SIMD128`) & `0-ULP` Parity ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):** `SONA`'s core (`process_tick16_const`) steps `Left` (`4` lanes) and `Right` (`4` lanes) simultaneously in `256-bit` / scalable `StereoVec8` registers (`AVX2+FMA`, `SVE2`, `SME`) or interleaved `2× SimdVec4` (`NEON`, `WASM SIMD128`) inside `16`-frame sub-block register-hoisted loops (`TPT-SVF` coeffs, `ParaLRU` scalars, `PolyADAA1` states, and `4×4` inter-block tiles pinned in vector registers with `0` GPR spills), enforcing identical fixed `Vec4`-pair reduction trees for exact `0-ULP` cross-ISA parity.

### 4.3 $3\text{ kHz}$ Bilinear Rank-$R$ Stage Latent Folding

`r[dsp.op-n.stage2.latent-folding]`

At each $3\text{ kHz}$ control boundary, stage latents $\mathbf{z}_{\text{stage}} \in \mathbb{R}^{D_{\text{stage}}}$ ($D_{\text{stage}} = 4$ in `Lite/Full/Pro`, $8$ in `Ultra`) modulate per-block **`TPT-SVF` prewarped `(g, k)` and tap weights**, **`ParaLRU` `(alpha_rel, alpha_atk, kappa)` and drive/bias scaling**, and **`fp16` `W_readout` direct `VarPro` linear readouts** (`Arena 3` holds `Ultra` extensions; `0` extra `48 kHz` MACs when controls are static):
$$\boldsymbol{\gamma}_{\text{stage}}(\mathbf{z}_{\text{stage}}) = \boldsymbol{\gamma}_0 + \sum_{r=1}^{D_{\text{stage}}} z_{\text{stage}, r} \Delta\boldsymbol{\gamma}_r, \qquad \mathbf{W}_{\text{ro}}(\mathbf{z}_{\text{stage}}) = \mathbf{W}_{\text{ro},0} + \sum_{r=1}^{D_{\text{stage}}} z_{\text{stage}, r} \mathbf{W}_{\text{ro}, r}$$

---

## 5. `Op-N[Ctrl]`: 3 kHz Algebraic Polynomial & $O(1)$ Sufficient-Statistic Engine

1. `r[dsp.op-n.ctrl.incremental-update]` **Incremental Sufficient-Statistic Update:** When contributor $p \in \mathcal{I}_{\text{moved}}$ changes from $u_{p,\text{old}}$ to $u_{p,\text{new}}$, update the exact-integer main-effect and Rank-$R$ pair-statistic accumulators (`64 B` in `Lite..Pro`, `128 B` in `Ultra`; self-contained in `Arena 0` for `Lite` strip safety; see [`parametric_control_bus.md` §2.3](parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling)):
   $$\Delta\mathbf{z}_{\text{main}} = \mathbf{W}_{\text{main}, p} \cdot \big(\boldsymbol{\psi}_p(u_{p, \text{new}}) - \boldsymbol{\psi}_p(u_{p, \text{old}})\big)$$
2. `r[dsp.op-n.ctrl.lobatto-basis]` **Node-Anchored Control Basis:** Evaluate the persisted observed anchor nodes and switch contributor slices ([`parametric_control_bus.md` §2.2](parametric_control_bus.md#22-cardinality-bounded-node-anchored-newtonlobatto-basis)).
3. `r[dsp.op-n.ctrl.chebyshev-lfo]` **Chebyshev LFO Waveshaping:** Shape quadrature rotor states via $T_1(x) = x$, $T_2(x) = 2x^2 - 1$, $T_3(x) = 4x^3 - 3x$, $T_4(x) = 8x^4 - 8x^2 + 1$.
4. `r[dsp.op-n.ctrl.fm-interaction]` **Rank-4/8 FM Bilinear Interaction:** Evaluate interaction factors from scalar embeddings $z_p = \mathbf{w}_{\text{proj},p}^\top\boldsymbol{\psi}_p(u_p)$ and mix via $\mathbf{U}_{\text{int}}$ (`8×4` base in `Arena 0` + three `8×4` `Ultra` delta blocks in `Arena 3`).
5. `r[dsp.op-n.ctrl.trust-clip]` **Groupwise Trust-Region Clipping:** Apply the outward-checked trust-region clip independently to $\mathbf{z}_{\text{head}}$ and $\mathbf{z}_{\text{stage}}$ ([`parametric_control_bus.md` §1.2](parametric_control_bus.md#outward-checked-trust-region-clip)).
