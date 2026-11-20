---
id: spec.dsp.op-n
title: "Op-N: Contractive ParaLRU, Divided-Difference PolyADAA1, Wavefolder & Bitcrusher"
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

# Op-N: Contractive `ParaLRU`, Divided-Difference `PolyADAA1`, Wavefolder & Bitcrusher

This specification defines Op-N (Nonlinear and Parallel State-Space) for Dynamic Neural Audio Modeler (DNAM) SONA v3.0 ([sona_architecture.md](../architecture/sona_architecture.md), [REQ-1.1, REQ-6.2](../../requirements/system_goals_and_requirements.md), [ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)), covering four interlocking primitives:

1. The 4-block Matryoshka SONA core across the four tier profiles with lower-triangular 4×4 inter-block mixing and 0-ULP parity across 8-lane and interleaved 4-lane SIMD execution.
2. Decoupled 1-D scalar `ParaLRU` with exact in-SRAM forward tiles and $O(\log_2 N)$ affine inclusive reverse Blelloch suffix scans across 256-sample GPU tiles.
3. Pure `f32` divided-difference `PolyADAA1` and cubic $C^2$ wavefolder saturation with batched 2-step Newton rail-crossing correction and zero-DC equilibrium.
4. Precomputed-reciprocal ZOH bitcrusher (`Family 5`) and `Op-N[Ctrl]` control-plane operators.

---

## 1. Pure-`f32` Divided-Difference `PolyADAA1` & Antialiased Activations

### 1.1 Pre-Activation Clamping, Core Sag Scaling & Standalone Reference

`r[dsp.op-n.polyadaa.clamp]`

Inside every SONA core block, linear signal paths clamp only at the ±1e9 safety boundary (`REQ-7.1`). Both `PolyADAA1` and the cubic $C^2$ wavefolder store the prior unclamped endpoint $r_{n-1}$ and pre-clamp endpoints via $\bar{r}_n = \mathrm{clamp}(r_n, -1.0, +1.0)$ before polynomial evaluation so out-of-range inputs never yield `Inf - Inf` NaNs. Whenever $\max(|r_{n-1}|, |r_n|) \le 0.0$, explicit zero masking writes `0x0000_0000` (+0.0) so odd symmetry holds without -0.0 residues.

| Evaluation Context | Pre-Activation Drive Formula | Parameter Domains & Scaling Semantics |
| :--- | :--- | :--- |
| SONA Core (4-Lane Stereo Group) | `r_n = u_tone[n] * d_adaa * g_sag + b_lru * v_lru[n]` | `g_sag = max(0.05, 1.0 - d_sag * v_lru)`; `sag_depth` in `[-8.0, 8.0]`, `adaa_drive` in `[-32.0, 32.0]`, `lru_bias_drive` in `[-16.0, 16.0]` |
| Normalized Kernel Evaluation | Normalized pre-scaled `(u_prev, u_curr)` inputs | Drive `(2 / 3) * d` pre-multiplied upstream in `adaa_drive`; the ADAA kernel evaluates endpoints directly |

### 1.2 Pure-`f32` Division-Free Primitives & Midpoint Factorization

`r[dsp.op-n.polyadaa.midpoint]`

To guarantee 0-ULP bit-exact reproducibility across CPU scalar/SIMD and 32-bit GPU compute targets without `FDIV`, `FSQRT`, standard math library calls, lookup-table gathers, or register spills, all nonlinear and transcendental helpers execute in pure IEEE-754 `binary32`.

At every recursive state write across `TPT-SVF`, `ParaLRU`, `Op-P` filters, Kautz banks, and `Op-T` diffusers, recursive state subnormal flushing clears IEEE-754 subnormals (`(bits & 0x7F80_0000) == 0`) via `bits & 0x8000_0000`, while zero-equilibrium nonlinear outputs emit canonical +0.0 (`0x0000_0000`).

For division-free Newton-Raphson reciprocals (2-step, 3-step, >= 1.0 fast path, and batched 4-pair evaluation), inputs clamp at $d_{\text{safe}} = \max(d, 10^{-6})$ except when $d \ge 1.0$ holds structurally.

Letting `bits(x)` denote the 32-bit unsigned IEEE-754 bit pattern of `x`, seeding `y_0` from `0x7EF3_11C3 - bits(d_safe)` (`IEEE754_F32_RCP_SEED`) and running two FMA steps yields relative error below $3.5 \times 10^{-6}$, while three steps yield sub-1-ULP accuracy:

$$y_{k+1} = y_k + y_k (1 - d_{\text{safe}} y_k) = \mathrm{fma}\big(y_k, \mathrm{fma}(-d_{\text{safe}}, y_k, 1.0), y_k\big)$$

Inverse square roots (2-step and 3-step Newton-Raphson) clamp inputs at $d_{\text{safe}} = \max(d, 10^{-12})$, form $h = 0.5 d_{\text{safe}}$, seed `y_0` from `0x5F37_59DF - (bits(d_safe) >> 1)` (`IEEE754_F32_RSQRT_SEED`, or `0x5FE6_EB50_C7B5_37A9` offline in `f64`), and run two or three refinement steps:

$$y_{k+1} = y_k \cdot \big(1.5 - h \cdot y_k^2\big)$$

Clamped base-2 exponential octave scaling over `[-5.0, +5.0]` uses one canonical degree-5 minimax Horner approximation across both SIMD math and filter pre-scaling:

| Kernel Operation | Input Clamp | Exponent Split & Bit Injection | Degree-5 Minimax Horner Coefficients (`f32` Hex) |
| :--- | :--- | :--- | :--- |
| Clamped Base-2 Exponential | `[-5.0, +5.0]` | Nearest int `k` via `BIASED_INT_MAGIC_F32 = 0x4B40_0000`, fractional `f = x - k` in `[-0.5, 0.5]`, scale factor `(k + 127) << 23` | 1.0 (`0x3F80_0000`), `LN_2 = 0.6931471824645996` (`0x3F31_7218`), `EXP2_C2 = 0.24022109` (`0x3E75_FC84`), `EXP2_C3 = 0.05550411` (`0x3D63_5847`), `EXP2_C4 = 0.009675892` (`0x3C1E_87A2`), `EXP2_C5 = 0.0013333558` (`0x3AAE_C3FF`) |

Software `fp16` serialization extracts sign `(bits >> 16) & 0x8000`, exponent `(bits >> 23) & 0xFF`, and mantissa `bits & 0x007F_FFFF` strictly via `u32` bit manipulation without clamping infinity or NaN prior to classification (`new_exp = exp - 112`):

| IEEE-754 Class | Encode Condition (`f32` -> `u16`) | Encoded `u16` Bit Pattern | Decode (`fp16` -> `f32`) |
| :--- | :--- | :--- | :--- |
| 1. Infinity | `exp == 0xFF && mant == 0` | `sign \| 0x7C00` | `(sign << 16) \| 0x7F80_0000` |
| 2. Payload `NaN` | `exp == 0xFF && mant != 0` | `sign \| 0x7C00 \| max(mant >> 13, 1)` | `(sign << 16) \| 0x7F80_0000 \| (mant16 << 13)` |
| 3. Overflow | `new_exp >= 31` or RNE carry into exponent 31 | Saturate to `sign \| 0x7BFF` (±65,504.0) | Normal exponent 30 decode (±65,504.0) |
| 4. Zero / Tiny | `new_exp < -10` (magnitude below 2^-25) | `sign` (signed ±0.0) | `sign << 16` |
| 5. Subnormal | `-10 <= new_exp <= 0` | Shift `0x0080_0000 \| mant` right by `14 - new_exp` with ties-to-even | Normalize leading `mant16` bit into `binary32` |
| 6. Normal | `1 <= new_exp <= 30` | `(new_exp << 10) \| rounded(mant >> 13)`; overflow saturates to `sign \| 0x7BFF` | `(sign << 16) \| ((exp16 + 112) << 23) \| (mant16 << 13)` |

Interior `PolyADAA1` evaluates six FMAs over pre-clamped endpoints $u_n, u_{n-1} \in [-1, 1]$ using midpoint $m_n = \frac{1}{2}(u_n + u_{n-1})$ and half-difference $d_n = \frac{1}{2}(u_n - u_{n-1})$:

$$\mathcal{A}_{\text{odd}}[n] = m_n \Big(1.5 - 0.5\big(m_n^2 + d_n^2\big)\Big), \qquad \mathcal{A}_{\text{even}}[n] = \left(m_n^2 + \frac{1}{3}d_n^2\right) - \left(m_n^4 + 2 m_n^2 d_n^2 + \frac{1}{5}d_n^4\right)$$

When $r_n = r_{n-1} = u \in [-1, 1]$, $d_n = 0$ and these collapse to $g_{\mathrm{odd}}(u) = 1.5u - 0.5u^3$ and $g_{\mathrm{even}}(u) = u^2 - u^4$ with zero derivative at the rails, $g_{\mathrm{odd}}'(\pm 1) = 0$. Offline Padé-[5/4] reference constants $(n_1, n_2, d_1, d_2) = (61/600, 11/19200, 87/200, 47/3840)$ solve $d_1 - n_1 = 1/3$, $d_2 - n_2 - d_1/3 = -2/15$, $P_5(R) = Q_4(R)$, and $(P_5/Q_4)'(R) = 0$ at $R = 4$.

### 1.3 Saturated-Curve Rail Crossing (Pure-`f32` Batched Newton Reciprocal)

`r[dsp.op-n.polyadaa.rail-crossing]`

When consecutive inputs $a = \min(r_{n-1}, r_n) < b = \max(r_{n-1}, r_n)$ cross $\pm 1$, the span $b - a$ partitions into interior length $\ell_0 = \max(0, \min(b, 1) - \max(a, -1))$ and plateau lengths $\ell_- = \max(0, \min(b, -1) - a)$ and $\ell_+ = \max(0, b - \max(a, 1))$, evaluating $\mathcal{A}_{\text{odd}}, \mathcal{A}_{\text{even}}$ on $[\max(a, -1), \min(b, 1)]$:

$$A_{\text{odd,sat}} = \big(\ell_0 \mathcal{A}_{\text{odd}} + \ell_+ - \ell_-\big) \cdot y, \qquad A_{\text{even,sat}} = \big(\ell_0 \mathcal{A}_{\text{even}}\big) \cdot y$$

Whenever $|r_n - r_{n-1}| \le 10^{-6}$ (`TINY_CROSSING_EPS = 1.0e-6`), the kernel returns the algebraic midpoint result directly to avoid zero-span division. Otherwise, the ADAA kernel computes $y \approx (b - a)^{-1}$ in pure 32-bit float via a 2-step Newton reciprocal of $\max(b - a, \varepsilon_{\text{span}})$ with floor $\varepsilon_{\text{span}} = 10^{-6}$ (`RAIL_CROSSING_MIN_SPAN`). Because $g_{\mathrm{odd}}'(\pm 1) = 0$, crossing differences remain $C^1$ smooth with bounded condition number in `binary32`, reserving 6-step `f64` Newton reciprocals solely for offline test oracles.

### 1.4 Amortized Fast-Path `PolyADAA1` & Batched Newton Stereo Execution

`r[dsp.op-n.polyadaa.stereo-and-rail-guard]`

Whenever all lanes of a 4-lane or 8-lane stereo SIMD register satisfy $|r_{n-1}| \le 1.0$ and $|r_n| \le 1.0$, the interior `PolyADAA1` fast path evaluates branchlessly with zero reciprocal calls.

If any lane crosses $\pm 1$, the rail-crossing path evaluates 2-step Newton reciprocals across the 8-lane stereo vector or batches dual 4-lane vector pairs without register spills ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

### 1.5 Un-Delayed Linear Separation & 1× Passive B-Spline Blend

#### 1.5.1 Un-Delayed Linear Residual Path

`r[dsp.op-n.polyadaa.linear-residual]`

The linear residual path evaluates $\alpha_c x_c[n]$ directly with zero ADAA delay, preserving unity passband gain within $10^{-12}$ dB up to 22 kHz at 48 kHz.

#### 1.5.2 Passive Quadratic B-Spline Blend (0.875, 0.125) + AC Coupling

`r[dsp.op-n.polyadaa.bspline-ac]`

Nonlinear outputs pass through passive quadratic B-spline smoothing with weights 0.875 and 0.125, followed by 1-pole AC coupling at $R_{\text{ac}} = 0.9993$ (at 48 kHz) before summing with the un-delayed linear residual:

$$\tilde{y}_{\text{nl}}[n] = 0.875 \mathcal{A}[n] + 0.125 \mathcal{A}[n-1], \qquad m_{\text{dc}}[n] = 0.9993 m_{\text{dc}}[n-1] + 0.0007 \tilde{y}_{\text{nl}}[n]$$
$$y[n] = \alpha_c x[n] + \big(\tilde{y}_{\text{nl}}[n] - m_{\text{dc}}[n]\big)$$

#### 1.5.3 Multi-Drive `PolyADAA1` Lane Bank

`r[dsp.op-n.polyadaa.drive-bank]`

SONA core blocks evaluate `PolyADAA1` lanes in parallel alongside the reference multi-drive profile $d_{\text{bank}} = [0.25, 1.0, 4.0, 16.0]$, retaining prior endpoints `(r_prev, u_prev)` and antialiasing states (`last_a_odd`, `last_a_even`) for 1× (48 kHz) and 2× (96 kHz) interval integration.

---

## 2. Wavefolder, 2D Hysteresis & ZOH Bitcrusher

### 2.1 Wavefolder Rail Regularity and Candidates

`r[dsp.op-n.wavefolder.cubic-c2]`

To preserve $C^2$ continuity at $\pm 1$ with vanishing value, slope, and curvature at $\bar u = \pm 1$, `Op-N` pairs `PolyADAA1` with cubed-envelope odd and even wavefolders, convex-blended via $\mathcal{A}_{\text{odd}} \leftarrow (1 - \mu_{\text{wf}})\mathcal{A}_{\text{odd}} + \mu_{\text{wf}} w_{\text{odd,ADAA}}$ for $\mu_{\text{wf}} \in [0, 1]$ using pre-clamped inputs $\bar u = \mathrm{clamp}(u, -1, 1)$:

$$w_{\text{odd},3}(u) = \bar u\big(1 - \bar u^2\big)^3, \qquad w_{\text{even},3}(u) = \bar u^2\big(1 - \bar u^2\big)^3$$

For consecutive clamped endpoints $\bar{u}_{n-1}, \bar{u}_n \in [-1, 1]$ with $m = (\bar{u}_{n-1} + \bar{u}_n)/2$, $d = (\bar{u}_n - \bar{u}_{n-1})/2$, $s = m^2 + d^2$, and $\omega = 1 - s$, division-free evaluation emits +0.0 (`0x0000_0000`) when $\max(|u_{n-1}|, |u_n|) \le 0.0$ and otherwise computes:

$$A_{\text{odd},3} = m \cdot \omega \cdot \big(\omega^2 + 4 m^2 d^2\big), \qquad A_{\text{even},3} = (Q_2 - Q_8) - 3(Q_4 - Q_6)$$
$$Q_2 = m^2 + \frac{1}{3}d^2, \qquad Q_4 = m^4 + 2 m^2 d^2 + \frac{1}{5}d^4, \qquad Q_6 = m^6 + 5 m^4 d^2 + 3 m^2 d^4 + \frac{1}{7}d^6$$
$$Q_8 = m^8 + \frac{28}{3} m^6 d^2 + 14 m^4 d^4 + 4 m^2 d^6 + \frac{1}{9}d^8$$

When $\max(|u_{n-1}|, |u_n|) > 1.0$, rail-crossing integration multiplies interior span $\ell_0 = \max(0, \min(\max(u_{n-1}, u_n), 1) - \max(\min(u_{n-1}, u_n), -1))$ by $A_{\text{odd/even},3}$ and scales by a 2-step Newton reciprocal of $\max(\mathrm{distance}, \varepsilon_{\text{span}})$ (`RAIL_CROSSING_MIN_SPAN`), taking the interior midpoint whenever `distance <= TINY_CROSSING_EPS` (`1.0e-6`) and emitting 0.0 outside `[-1, 1]`.

---

### 2.2 2D Quadrature Phase-Space Hysteresis

`r[dsp.op-n.hysteresis.quadrature]`

Magnetic core transformer hysteresis and optocoupler memory evolve over phase-space coordinates $(s_{\text{re}}[n], s_{\text{im}}[n])$:

$$h_{\text{hyst}}[n] = s_{\text{re}}[n] \cdot \big(s_{\text{re}}[n] - s_{\text{re}}[n-1]\big) + \kappa_{\text{quad}} \cdot s_{\text{re}}[n] s_{\text{im}}[n]$$

---

### 2.3 Family 5: Precomputed-Reciprocal ZOH Sample-Rate Reducer & Bitcrusher

#### 2.3.1 Uniform Nearest-Step Quantizer (Halfway Ties to +∞)

`r[dsp.op-n.bitcrusher.quantizer]`

For integer bit depth $B \in [2, 16]$, cold preparation computes step $\Delta = 2^{-(B-1)}$ and scale $D = 2^{B-1}$:

$$Q_B(x) = \lfloor xD + 0.5 \rfloor \Delta$$

Over the certified [-100, 100] signal envelope in `f32`, the kernel computes $v = xD$, truncates toward zero to `i32` $t$, forms integer floor $k = t - \mathbf{1}[v < t]$ and residual $f = v - k$, and returns $(k + \mathbf{1}[f \ge 0.5])\Delta$ without negative truncation bias or pre-addition rounding overflow.

#### 2.3.2 Fixed-Hertz Integer ZOH Phase Profile

`r[dsp.op-n.bitcrusher.zoh-phase]`

Using $F = 2^{32}$ units per Hz, cold preparation computes $D = \mathrm{round}_{\text{even}}(F f_s)$ and $I = \mathrm{round}_{\text{even}}(F f_{\text{target}})$ with $0 \le I \le D$, stepping unsigned 64-bit phase $P \in [0, D)$ once per sample:

$$A = P + I, \qquad b = \mathbf{1}[A \ge D], \qquad P \leftarrow A - bD$$

Carry $b = 1$ latches $Q_B(x[n])$; at full rate $I = D$, every sample latches while $P$ stays constant; at zero rate $I = 0$ (active only when explicitly observed), the last latched value and phase hold unchanged.

#### 2.3.3 Division-Free Effective-Rate Phase Transfer

`r[dsp.op-n.bitcrusher.rate-transfer]`

Precomputing `u64` reciprocal $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$ on the cold path lets sample-rate transitions form $N = P_{\text{old}} D_{\text{new}} < 2^{102}$ in 128-bit unsigned integer arithmetic, evaluate quotient $q_0 = \lfloor N R / 2^{104} \rfloor$ via bit shifts, refine remainder $r_0 = N - q_0 D_{\text{old}}$, round ties-to-even, and clamp to $D_{\text{new}} - 1$ so imminent carries stay imminent:

$$P_{\text{new}} = \min\left(\mathrm{round}_{\text{even}}\left(\frac{P_{\text{old}}D_{\text{new}}}{D_{\text{old}}}\right), D_{\text{new}}-1\right), \qquad q_0 = (h \ll 24) \mathbin{\mathrm{or}} \big((m \mathbin{\mathrm{and}} M) \gg 40\big)$$

Here $N = N_{\text{hi}}2^{64} + N_{\text{lo}}$, $a = N_{\text{lo}}R$, $c = N_{\text{hi}}R$, $M = 2^{64} - 1$, $m = (a \gg 64) + (c \mathbin{\mathrm{and}} M)$, and $h = (c \gg 64) + (m \gg 64)$, keeping frequency error $|f_s(I/D) - f_{\text{target}}| \le 2^{-32}$ Hz.

#### 2.3.4 Forward-Faithful Training Surrogate

`r[dsp.op-n.bitcrusher.surrogate]`

Training returns exact quantized values on the forward pass with straight-through gradients $\hat q(x) = \mathrm{sg}[Q_B(x)] + (x - \mathrm{sg}[x])$, or substitutes a 4-term Fourier backward surrogate during warmup:

$$q_4(x) = x - \frac{\Delta}{\pi}\sum_{k=1}^{4}\frac{(-1)^{k-1}}{k}\sin\left(\frac{2\pi k x}{\Delta}\right)$$

---

## 3. Exact LTI Basis Commutativity Theorem (Theorem 1)

### 3.1 Formal Theorem Statement & Proof

`r[dsp.op-n.commuted.theorem1]`

SONA v3.0 applies Theorem 1 to fold linear 3 kHz VarPro readout projections (`pre_cab` and `direct`) directly inside SIMD registers across the 4-block core with zero intermediate scratch gather buffers.

**Theorem 1 (Exact LTI Basis Commutativity).** Let $\mathbf{u}[n] \in \mathbb{R}^P$ be bounded per-channel block activations satisfying $\sup_{n, p} |u_p[n]| \le U_{\max} < \infty$, let $\lbrace \mathcal{L}_k \rbrace_{k=0}^{K-1}$ be causal, stable LTI filters with impulse responses $h_k \in \ell_1(\mathbb{Z})$, and let $\lbrace \mathbf{W}_k \rbrace_{k=0}^{K-1} \subset \mathbb{R}^{P \times C}$ be constant projection matrices. Then for every output channel $c \in [0, C-1]$ and sample $n \in \mathbb{Z}$:

$$\sum_{k=0}^{K-1} \Big(\mathcal{L}_k\big(\mathbf{u}\big)[n]\Big) \mathbf{W}_k \equiv \sum_{k=0}^{K-1} \mathcal{L}_k\Big(\mathbf{u} \mathbf{W}_k\Big)[n]$$

*Proof.* Absolute convergence follows from $h_k \in \ell_1(\mathbb{Z})$ and boundedness of $\mathbf{u}$, allowing summation interchange by Fubini's theorem so $\mathbf{u}\mathbf{W}_k$ filters directly across $C$ SIMD lanes. $\blacksquare$

---

### 3.2 Causality Audit & Elimination of Acausal Lookahead

`r[dsp.op-n.commuted.causality]`

All Top-of-Frame Crossbar `Op-T` reads where $\tau \ge 4.0$ samples, `TPT-SVF` / `ParaLRU` / `PolyADAA1` updates, `poly3` decimation filters, and cabinet interfaces in SONA v3.0 are strictly causal with $d \ge 0$, maintaining sample-accurate phase alignment across 1× and 2× (4× bounce) paths.

---

### 3.3 Commutator Perturbation Under 3 kHz Control-Rate Modulation

`r[dsp.op-n.commuted.project-then-filter]`

Under 3 kHz control modulation of `TPT-SVF` coefficients, `ParaLRU` poles, `PolyADAA1` drive/bias, and `fp16` readout matrices, active parameter deltas micro-ramp across $K_{\text{ctrl}}$ samples, matching GPU and CPU forward evaluation with bounded commutator residual:

$$\Delta_{\text{comm}}[n] \le \frac{B_u \nu_{\text{ctrl}}}{f_s} \cdot \frac{r_k}{1 - r_k}$$

---

### 3.4 Boundary Conditions, Resting DC Equilibrium & Pop-Free Startup

`r[dsp.op-n.commuted.dc-equilibrium]`

With decoupled 1-D `ParaLRU` recurrence $v_i[n] = \alpha_{\text{rel},i} v_i[n-1] + (1 - \alpha_{\text{atk},i})\phi_{\text{rect}}(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1])$ satisfying $\phi_{\text{rect}}(0) = 0$, analytical zero-DC resting bias centering at `StarCenter` where $u_{\text{eq}} = 0.0$, `PolyADAA1` zero equilibrium satisfying $g_{\mathrm{odd}}(+0.0) = g_{\mathrm{even}}(+0.0) = +0.0$, and zero-initialized integrators, all recursive state settles at exact 0.000000 DC equilibrium (`h_eq == +0.0`, `b_head == 0.0`):

1. Core `ParaLRU` lanes initialize at $v_{\text{init}} = +0.0$ across 16, 24, 32, or 40 lanes per channel from Lite through Ultra.
2. `TPT-SVF` and AC coupling filters initialize states $s_1 = s_2 = +0.0$ at idle equilibrium.

---

## 4. 4-Block Matryoshka SONA Core (`TPT-SVF`, `ParaLRU` & `PolyADAA1`)

`r[dsp.op-n.core.matryoshka-blocks]`

SONA v3.0 models nonlinear preamps, diodes, sag, crossovers, and sweeps via a 4-block Matryoshka state-space core sorted across 10 logical 4-lane groups `[(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (0,2), (2,1), (0,3), (1,2)]` and 35 causal lower-triangular inter-block mixing tiles. Each block chains passive 2×2 `TPT-SVF`, decoupled 1-D scalar `ParaLRU`, and pure `f32` `PolyADAA1` (plus `C²` wavefolder lanes) specialized over the four tier profiles:

| Tier Profile | Groups & Lanes | Inter-Block Tiles | Core Rate | `TPT-SVF` Quads / Modes | Cabinet Slice (FIR / Kautz) | Tank Quads (`Lines`) | Hot Working Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lite `[1, 1, 1, 1]` | 4 groups (16 lanes) | 3 | 1× `PolyADAA1` | 4 quads (16 modes) | 32 taps / 8 pairs | 1 quad (4 lines) | <= 4,608 B |
| Full `[2, 2, 1, 1]` | 6 groups (24 lanes) | 13 | 2× OS (96 kHz) | 6 quads (24 modes) | 48 taps / 12 pairs | 2 quads (8 lines) | <= 7,936 B |
| Pro `[3, 2, 2, 1]` | 8 groups (32 lanes) | 23 | 2× OS (96 kHz) | 8 quads (32 modes) | 96 taps / 24 pairs | 3 quads (12 lines) | <= 15,872 B |
| Ultra `[4, 3, 2, 1]` | 10 groups (40 lanes) | 35 | 2× OS (4× Bounce) | 10 quads (40 modes) | 128 taps / 32 pairs | 4 quads (16 lines) | <= 28,672 B |

With `SOFT_RECT_EPS` $\varepsilon = 10^{-3}$ and `SOFT_RECT_LARGE_Z = 1.0e15`, the zero-DC soft rectifier sets $z_{\text{safe}} = \min(|z|, 10^{15})$, $q = z_{\text{safe}}^2 + \varepsilon^2$, evaluates $r_{\text{inv}} = q^{-1/2}$ via a 3-step Newton inverse square root, $r_{\text{rad}} = q \cdot r_{\text{inv}}$, and evaluates $w_\varepsilon = (r_{\text{rad}} + \varepsilon)^{-1}$ and $w_z = (r_{\text{rad}} + z_{\text{safe}})^{-1}$ via 3-step Newton reciprocals across three cancellation-free branches for $\phi(z) = \frac{1}{2}(z + \sqrt{z^2 + \varepsilon^2} - \varepsilon)$:

- Zero branch for $|z| \le 0.0$: emits canonical +0.0 (`0x0000_0000`) with derivative 0.5.
- Positive branch for $z > 0$: evaluates $\phi(z) = \frac{1}{2}(z_{\text{safe}} + z_{\text{safe}}^2 w_\varepsilon)$ and $\phi'(z) = \frac{1}{2}(1 + z_{\text{safe}} r_{\text{inv}})$.
- Negative branch for $z < 0$: uses rationalized evaluation to avoid square-root cancellation, saturating at $(-\varepsilon/2, 0.0)$ when $|z| > 10^{15}$:

$$\phi(z) = -\frac{\varepsilon}{2} (z_{\text{safe}} w_z)(1 + z_{\text{safe}} w_\varepsilon), \qquad \phi'(z) = \frac{\varepsilon^2 r_{\text{inv}}}{2(r_{\text{rad}} + z_{\text{safe}})}$$

Every `ParaLRU` lane clamps $\alpha_{\text{rel}} \in [0.0, r_{\max}]$ with `MAX_PARALRU_R_F32 = 0x3F7F_FFAC` (0.999995), $\alpha_{\text{atk}} \in [0.0, 1.0]$, and $\kappa \in [0.0, 1.0]$, updating with subnormal state flush:

$$v_i[n] = \alpha_{\text{rel},i} v_i[n-1] + (1 - \alpha_{\text{atk},i}) \phi\big(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1]\big)$$
$$J_i[n] = \frac{\partial v_i[n]}{\partial v_i[n-1]} = \alpha_{\text{rel},i}\Big(1 - \kappa_i (1 - \alpha_{\text{atk},i}) \phi'\big(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1]\big)\Big) \in \big[\alpha_{\text{rel},i}(1 - \kappa_i(1 - \alpha_{\text{atk},i})), \alpha_{\text{rel},i}\big] \subset [0, r_{\max}]$$

Because $J_i[n] \in [0, r_{\max}] \subset [0, 1)$, `ParaLRU` is unconditionally contractive without post-hoc ISS clamping. Because $\phi$ has a sharp $\varepsilon = 10^{-3}$ active-set bend that stalls time-parallel forward Newton scans, WebGPU training runs forward tiles serially per decoupled lane in GPU workgroup shared SRAM (<= 15.0 KiB, [webgpu_and_varpro_solver.md](../training/webgpu_and_varpro_solver.md)) across both full and padded-tail tiles (matching CPU 4-lane scalar lane 0 for 0-ULP parity), while `TPT-SVF`, `PolyADAA1`, and wavefolder run 256-thread time-parallel.

For backward gradients inside the reverse-time adjoint scan tile, recomputing `v_traj[0..N]` from checkpoint $v_{\text{init}}$ gives linear reverse-time scan elements `(J_i, J_i * adj_v[i])` over $N + 1$ inclusive suffix slots starting from sentinel `suffix[N] = (1.0, 0.0)`:

$$e_i = (a_i, b_i) = \big(J_i,\ J_i \bar{v}_i\big), \qquad e_{\mathrm{id}} = (1.0, 0.0)$$
$$\mathrm{compose}(e_{\mathrm{first}}, e_{\mathrm{second}}) = \big(e_{\mathrm{second}}.a \cdot e_{\mathrm{first}}.a,\ e_{\mathrm{second}}.a \cdot e_{\mathrm{first}}.b + e_{\mathrm{second}}.b\big)$$
$$\mathtt{suffix}[N] = (1.0, 0.0), \qquad \mathtt{suffix}[i] = \mathrm{compose}\big(\mathtt{suffix}[i+1],\ e_i\big)$$

Because `suffix[i+1]` precedes `e_i` in reverse time, composition sets $\mathtt{suffix}[i].b = J_i \cdot \mathtt{suffix}[i+1].b + J_i \bar{v}_i$. Propagating incoming boundary adjoint $\bar{v}_{\text{out}}$ (`adj_v_out`) from `tile + 1` alongside `tile_summary = suffix[0]` yields exact full-stem BPTT gradients across slow sag poles:

$$\lambda_i = \bar{v}_i + \mathtt{suffix}[i+1].a \cdot \bar{v}_{\text{out}} + \mathtt{suffix}[i+1].b, \qquad \nabla_{v_{\text{init}}} L = \mathtt{suffix}[0].a \cdot \bar{v}_{\text{out}} + \mathtt{suffix}[0].b$$
$$\nabla_{u_i} L = \lambda_i (1 - \alpha_{\text{atk}})\phi'(z_i), \qquad \nabla_{\alpha_{\text{rel}}} L = \sum_{i=0}^{N-1} \lambda_i v_{i-1}\big(1 - \kappa(1 - \alpha_{\text{atk}})\phi'(z_i)\big)$$
$$\nabla_{\alpha_{\text{atk}}} L = -\sum_{i=0}^{N-1} \lambda_i \phi(z_i), \qquad \nabla_{\kappa} L = -\sum_{i=0}^{N-1} \lambda_i \alpha_{\text{rel}}(1 - \alpha_{\text{atk}})\phi'(z_i) v_{i-1}$$

```text
// Inclusive Reverse-Time Adjoint Suffix Scan Kernel (N + 1 Slots per Lane)
1. Recompute v_traj[0..N] in workgroup SRAM from saved tile-start checkpoint v_init.
2. Form local reverse-scan elements for i in 0..N: e[i] = { a: J[i], b: J[i] * adj_v[i] }.
3. Run O(log2 N) inclusive reverse Blelloch suffix scan over N + 1 slots with compose(suffix[i+1], e[i]):
      suffix[N] = { a: 1.0, b: 0.0 }
      suffix[i].a = e[i].a * suffix[i+1].a
      suffix[i].b = e[i].a * suffix[i+1].b + e[i].b
4. Propagate incoming boundary adjoint adj_v_out from tile + 1:
      lambda[i] = adj_v[i] + suffix[i+1].a * adj_v_out + suffix[i+1].b
      adj_v_init = suffix[0].a * adj_v_out + suffix[0].b   // passed backward as adj_v_out of tile - 1
```

For ZOH sample-rate transfer at $\rho_{\text{sr}} = f_{s,\text{ref}} / f_{s,\text{eff}}$, `TPT-SVF` clamps $g_{\text{ref}} \in [10^{-5}, 64.0]$ while `ParaLRU` rescales $(\alpha_{\text{rel}}', \alpha_{\text{atk}}', \kappa')$ with $\beta = 1 - \alpha_{\text{atk}}$ and $a_{\text{atk}} = \alpha_{\text{rel}}(1 - \kappa\beta)$:

$$g_{\text{host}} = \mathrm{clamp}\Big(\tan\big(\rho_{\text{sr}} \arctan(g_{\text{ref}})\big), 10^{-5}, 64.0\Big)$$
$$\alpha_{\text{rel}}' = \mathrm{clamp}\big(\alpha_{\text{rel}}^{\rho_{\text{sr}}}, 0.0, r_{\max}\big), \qquad \beta' = \mathrm{clamp}\left(\beta \frac{1 - a_{\text{atk}}^{\rho_{\text{sr}}}}{1 - a_{\text{atk}}}, 0.0, 1.0\right), \qquad \alpha_{\text{atk}}' = 1 - \beta'$$

The ratio $(1 - a_{\text{atk}}^{\rho_{\text{sr}}}) / (1 - a_{\text{atk}})$ approaches $\rho_{\text{sr}}$ continuously as $a_{\text{atk}} \to 1$, and $\kappa' = \mathrm{clamp}((1 - (1 - \kappa\beta)^{\rho_{\text{sr}}}) / \beta', 0.0, 1.0)$ when $\beta' > 10^{-12}$ (retaining $\kappa$ otherwise). Every lane stays unconditionally BIBO-stable (`REQ-7.2`), while direct VarPro readouts (`pre_cab` and `direct`) carry unclipped ±100.0 (+40 dBFS) headroom (`REQ-7.1`).

---

### 4.1 Passive 2×2 ZDF `TPT-SVF` Four-Filter Tone Basis & Associative Scans

`r[dsp.op-n.core.tpt-tone-basis]`

Each 2×2 ZDF `TPT-SVF` mode clamps $g \in [10^{-5}, 64.0]$ and $k \in [0.0, 8.0]$, where $f_c = (f_s/\pi)\arctan(g)$, evaluates division-free $a_1 = (1 + g(g + k))^{-1}$ via a 2-step Newton reciprocal for arguments $\ge 1$ (within 1 ULP of 3-step refinement), $a_2 = g a_1$, and $a_3 = g a_2$, and updates subnormal-flushed states $s_1, s_2$:

$$v_3 = u - s_2, \qquad v_1 = a_1 s_1 + a_2 v_3, \qquad v_2 = s_2 + a_2 s_1 + a_3 v_3$$
$$\mathrm{bp} = v_1, \qquad \mathrm{lp} = v_2, \qquad \mathrm{hp} = u - k v_1 - v_2, \qquad \mathrm{ap} = u - 2 k v_1$$
$$s_1' = \mathrm{flush}(2 v_1 - s_1), \qquad s_2' = \mathrm{flush}(2 v_2 - s_2)$$

Linear tap mix $y_{\text{tone}} = w_{\text{lp}} \mathrm{lp} + w_{\text{bp}} \mathrm{bp} + w_{\text{hp}} \mathrm{hp} + w_{\text{ap}} \mathrm{ap} + w_{\text{dir}} u$ (`binary16` weights in `[-16.0, 16.0]`) discovers arbitrary crossovers, tone stacks, and phasers solely via data (`REQ-1.9`). In matrix form $\mathbf{s}_{n+1} = \mathbf{A}_{\text{tpt}}\mathbf{s}_n + \mathbf{b}_{\text{tpt}} u_n$:

$$(\mathbf{A}_{\text{tpt}})_{11} = 2 a_1 - 1, \quad (\mathbf{A}_{\text{tpt}})_{12} = -2 a_2, \quad (\mathbf{A}_{\text{tpt}})_{21} = 2 a_2, \quad (\mathbf{A}_{\text{tpt}})_{22} = 1 - 2 a_3, \quad b_1 = 2 a_2, \quad b_2 = 2 a_3, \quad \Vert \mathbf{A}_{\text{tpt}} \Vert_2 \le 1$$

Forward and backward WebGPU scans compose via $(\mathbf{A}_2, \mathbf{b}_2) \circ (\mathbf{A}_1, \mathbf{b}_1) = (\mathbf{A}_2 \mathbf{A}_1, \mathbf{A}_2 \mathbf{b}_1 + \mathbf{b}_2)$ and reverse transpose $(\mathbf{A}_1^\top, \mathbf{g}_1) \circ_{\text{rev}} (\mathbf{A}_2^\top, \mathbf{g}_2) = (\mathbf{A}_1^\top \mathbf{A}_2^\top, \mathbf{A}_1^\top \mathbf{g}_2 + \mathbf{g}_1)$.

---

### 4.2 Stereo Basis & Lower-Triangular 4×4 Inter-Block Mixing

`r[dsp.op-n.core.interblock-tiles]`

Inter-lane mixing across the 4-block SONA core runs strictly between blocks through lower-triangular 4×4 spatial mixing tiles with strict `Lite < Full < Pro < Ultra` prefix nesting. Direct-register VarPro linear readouts (`pre_cab` feeding the cabinet and `direct` feeding stereo output) accumulate in vector registers without intermediate scratch gather buffers.

---

### 4.3 Wide-ISA 8-Lane Stereo & Interleaved 4-Lane SIMD Execution

`r[dsp.op-n.core.stereo-simd]`

Inside 16-frame register-hoisted sub-blocks, SONA steps Left (4 lanes) and Right (4 lanes) simultaneously in 256-bit or scalable 8-lane stereo SIMD registers or interleaved 4-lane 128-bit vector pairs. Pinning filter coefficients, `ParaLRU` parameters, `PolyADAA1` history, and mixing tiles in SIMD registers yields 0-ULP cross-ISA parity with fixed 4-lane pair reduction trees ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

### 4.4 3 kHz Gauge-Invariant `z_stage` Control Sensitivity & Static `W_ro,0` Readout

`r[dsp.op-n.core.control-sensitivity]`

`SO(4)` buyer gauge `Q_buyer` rotates only `z_head[0..4]` along with columns 0..4 of `main_base`, `u_int_base`, `w_ctrl`, and `u_int_ultra` Block 0.

Unrotated `z_stage` spans `ctrl_basis_4` for Lite..Pro (`D_s = 4`) and adds `z_stage_ultra` for Ultra (`D_s = 8`, columns 4..8 and Blocks 1..2).

The readout tier block stores one static base matrix `W_ro,0` (sized 16×12, 24×12, 32×12, or 40×12 across tiers) with zero per-latent matrices or wire biases. At each 3 kHz boundary, `z_head` stays inside `W_ctrl` (`f32`) while `stage_latent_*` half-precision sensitivity tables index strictly unrotated `z_stage`, keeping 0-ULP `SO(4)` watermark invariance.

| Modulation Reference | Scope & Canonical Definition |
| :--- | :--- |
| Sensitivity Laws (SSOT) | `stage_targets[0..16]`, `stage_latent_mods_fp16[0..80]`, Ultra `stage_latent_ctrl_ultra_fp16[0..256]`, and `SONA_META_CORE_MODS_PRESENT`-gated `stage_latent_core_mods_fp16[0..1024]` ([parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws)) |
| Container Layout | 64-byte-aligned binary arena field offsets, alignment, and tier sizes ([dnam_binary_format.md](../containers/dnam_binary_format.md)) |

---

## 5. `Op-N[Ctrl]`: 3 kHz Algebraic Polynomial & `O(1)` Sufficient-Statistic Engine

### 5.1 Incremental Sufficient-Statistic Update

`r[dsp.op-n.ctrl.incremental-update]`

When control $p \in \mathcal{I}_{\text{moved}}$ moves from $u_{p,\text{old}}$ to $u_{p,\text{new}}$, the control engine updates exact-integer main-effect and rank $R$ pair-statistic accumulators (64 B in Lite..Pro, 128 B in Ultra, self-contained in Arena 0; [parametric_control_bus.md §2.3](parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling)):

$$\Delta\mathbf{z}_{\text{main}} = \mathbf{W}_{\text{main}, p} \cdot \big(\boldsymbol{\psi}_p(u_{p, \text{new}}) - \boldsymbol{\psi}_p(u_{p, \text{old}})\big)$$

### 5.2 Node-Anchored Control Basis

`r[dsp.op-n.ctrl.lobatto-basis]`

Basis evaluation interpolates persisted observed anchor nodes and switch contributor slices as defined in [parametric_control_bus.md §2.2](parametric_control_bus.md#22-cardinality-bounded-node-anchored-newtonlobatto-basis).

### 5.3 Chebyshev LFO Waveshaping

`r[dsp.op-n.ctrl.chebyshev-lfo]`

Quadrature LFO states are shaped through Chebyshev polynomials $T_1(x) = x$, $T_2(x) = 2x^2 - 1$, $T_3(x) = 4x^3 - 3x$, and $T_4(x) = 8x^4 - 8x^2 + 1$.

### 5.4 Rank-4/8 FM Bilinear Interaction

`r[dsp.op-n.ctrl.fm-interaction]`

Pairwise FM interactions combine scalar embeddings $z_p = \mathbf{w}_{\text{proj},p}^\top\boldsymbol{\psi}_p(u_p)$ through $\mathbf{U}_{\text{int}}$ (an 8×4 base matrix in Arena 0 plus three 8×4 delta blocks in Arena 3).

### 5.5 Groupwise Trust-Region Clipping

`r[dsp.op-n.ctrl.trust-clip]`

Outward-checked trust-region clipping runs independently on `z_head[0..4]`, `z_head_ultra[0..4]`, `z_mod[0..4]`, and `z_mod_ultra[0..4]` per [parametric_control_bus.md §1.2.1](parametric_control_bus.md#121-outward-checked-trust-region-clip).
