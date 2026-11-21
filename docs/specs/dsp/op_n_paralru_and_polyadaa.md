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
4. Smooth raised-cosine ZOH & Fourier staircase quantization (`Op-N[Pre]`, Family 5) and `Op-N[Ctrl]` control-plane operators.

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

To guarantee 0.000000 ULP parity across same-FMA-class CPU backends (and across all CPU backends on explicitly unfused `FP_CONTRACT OFF` paths) plus `<= 1.5` ULP/step agreement between hardware-FMA and non-FMA CPU or 32-bit GPU targets ([sona_architecture.md §1.5.3](../architecture/sona_architecture.md#153-single-library-per-architecture-runtime-simd-dispatch--0000000-ulp-parity)), all nonlinear and transcendental helpers execute in pure IEEE-754 `binary32` without `FDIV`, `FSQRT`, or external math libraries.

At every recursive state write across `TPT-SVF`, `ParaLRU`, `Op-P` filters, Kautz banks, and `Op-T` diffusers, recursive state subnormal flushing clears IEEE-754 subnormals (`(bits & 0x7F80_0000) == 0`) via `bits & 0x8000_0000`, while zero-equilibrium nonlinear outputs emit canonical +0.0 (`0x0000_0000`).

For division-free Newton-Raphson reciprocals (2-step, 3-step, >= 1.0 fast path, and batched 4-pair evaluation), inputs clamp at $d_{\text{safe}} = \max(d, 10^{-6})$ except when $d \ge 1.0$ holds structurally.

Letting `bits(x)` denote the 32-bit unsigned IEEE-754 bit pattern of `x`, seeding `y_0` from `0x7EF3_11C3 - bits(d_safe)` (`IEEE754_F32_RCP_SEED`) and running two FMA steps yields relative error below $3.5 \times 10^{-6}$, while three steps yield sub-1-ULP accuracy:

$$y_{k+1} = y_k + y_k (1 - d_{\text{safe}} y_k) = \mathrm{fma}\big(y_k, \mathrm{fma}(-d_{\text{safe}}, y_k, 1.0), y_k\big)$$

Inverse square roots (2-step and 3-step Newton-Raphson) clamp inputs at $d_{\text{safe}} = \max(d, 10^{-12})$, form $h = 0.5 d_{\text{safe}}$, seed `y_0` from `0x5F37_59DF - (bits(d_safe) >> 1)` (`IEEE754_F32_RSQRT_SEED`, or `0x5FE6_EB50_C7B5_37A9` offline in `f64`), and run two or three refinement steps:

$$y_{k+1} = y_k \cdot \big(1.5 - h \cdot y_k^2\big)$$

Vector SIMD `rsqrt_pos_nr2` / `rsqrt_pos_nr3` steps evaluate `y = y * (1.5).fms(h, y * y)` (`acc.fms(a, b) = fma(-a, b, acc)`), whereas host standalone `rsqrt_nr2_pair_f32` evaluates unfused `y * (1.5 - h * (y * y))` under `FP_CONTRACT OFF`.

Clamped base-2 exponential scaling uses one canonical degree-5 minimax Horner decomposition across both octave/filter modulations (`exp2_clamped5` on `[-5.0, +5.0]`) and wide quantizer scaling (`exp2_clamped16` on `[-16.0, +16.0]`):

| Kernel Operation | Input Clamp | Exponent Split & Bit Injection | Degree-5 Minimax Horner Coefficients (`f32` Hex) |
| :--- | :--- | :--- | :--- |
| `exp2_clamped5` (Octave / Filter) | `[-5.0, +5.0]` | Nearest int `k = round_ties_even(clamp(x, -5.0, +5.0))` via `BIASED_INT_MAGIC_F32 = 0x4B40_0000`, `f = x - k` in `[-0.5, +0.5]`, scale `(k + 127) << 23` | 1.0 (`0x3F80_0000`), `LN_2 = 0.6931471824645996` (`0x3F31_7218`), `EXP2_C2 = 0.24022109` (`0x3E75_FC84`), `EXP2_C3 = 0.05550411` (`0x3D63_5847`), `EXP2_C4 = 0.009675892` (`0x3C1E_87A2`), `EXP2_C5 = 0.0013333558` (`0x3AAE_C3FF`) |
| `exp2_clamped16` (`Op-N[Pre]` Scale / Step) | `[-16.0, +16.0]` | Same nearest-int split `k = round_ties_even(clamp(x, -16.0, +16.0))` via `BIASED_INT_MAGIC_F32 = 0x4B40_0000`, `f = x - k` in `[-0.5, +0.5]`, scale `(k + 127) << 23` | Exact same 6 degree-5 minimax Horner `f32` coefficients above (`0x3F80_0000` through `0x3AAE_C3FF`) |

Software `fp16` serialization extracts sign `(bits >> 16) & 0x8000`, exponent `(bits >> 23) & 0xFF`, and mantissa `bits & 0x007F_FFFF` strictly via `u32` bit manipulation without clamping infinity or NaN prior to classification (`new_exp = exp - 112`):

| IEEE-754 Class | Encode Condition (`f32` -> `u16`) | Encoded `u16` Bit Pattern | Decode (`fp16` -> `f32`) |
| :--- | :--- | :--- | :--- |
| 1. Infinity | `exp == 0xFF && mant == 0` | `sign \| 0x7C00` | `(sign << 16) \| 0x7F80_0000` |
| 2. Payload `NaN` | `exp == 0xFF && mant != 0` | `sign \| 0x7C00 \| max(mant >> 13, 1)` | `(sign << 16) \| 0x7F80_0000 \| (mant16 << 13)` |
| 3. Overflow | `new_exp >= 31` or RNE carry into exponent 31 | Saturate to `sign \| 0x7BFF` (±65,504.0) | Normal exponent 30 decode (±65,504.0) |
| 4. Zero / Tiny | `new_exp < -10` (magnitude below 2^-25) | `sign` (signed ±0.0) | `sign << 16` |
| 5. Subnormal | `-10 <= new_exp <= 0` | Shift `0x0080_0000 \| mant` right by `14 - new_exp` with ties-to-even | Normalize leading `mant16` bit into `binary32` |
| 6. Normal | `1 <= new_exp <= 30` | `(new_exp << 10) \| rounded(mant >> 13)`; overflow saturates to `sign \| 0x7BFF` | `(sign << 16) \| ((exp16 + 112) << 23) \| (mant16 << 13)` |

Interior `PolyADAA1` evaluates four interior FMAs (five including rail-crossing) over pre-clamped endpoints $u_n, u_{n-1} \in [-1, 1]$ using midpoint $m_n = \frac{1}{2}(u_n + u_{n-1})$ and half-difference $d_n = \frac{1}{2}(u_n - u_{n-1})$:

$$\mathcal{A}_{\text{odd}}[n] = m_n \Big(1.5 - 0.5\big(m_n^2 + d_n^2\big)\Big), \qquad \mathcal{A}_{\text{even}}[n] = \left(m_n^2 + \frac{1}{3}d_n^2\right) - \left(m_n^4 + 2 m_n^2 d_n^2 + \frac{1}{5}d_n^4\right)$$

```text
// Exact PolyADAA1 Interior & Rail-Crossing SIMD Evaluation (acc.fma(a,b) = fma(a,b,acc), acc.fms(a,b) = fma(-a,b,acc))
m2 = m * m; d2 = d * d; m4 = m2 * m2; d4 = d2 * d2
raw_odd    = m * (1.5).fms(0.5, m2 + d2)                   // canonical +0.0 when max(|u_prev|, |u_curr|) <= 0.0
lin_even   = m2.fma(1.0 / 3.0, d2)                         // 1/3 = 0x3EAA_AAAB
quar_even  = (m4.fma(2.0 * m2, d2)).fma(1.0 / 5.0, d4)     // 1/5 = 0x3E4C_CCCD
raw_even   = lin_even - quar_even                          // canonical +0.0 when max(|u_prev|, |u_curr|) <= 0.0
```

When $r_n = r_{n-1} = u \in [-1, 1]$, $d_n = 0$ and these collapse to $g_{\mathrm{odd}}(u) = 1.5u - 0.5u^3$ and $g_{\mathrm{even}}(u) = u^2 - u^4$ with zero derivative at the rails, $g_{\mathrm{odd}}'(\pm 1) = 0$. Offline Padé-[5/4] reference constants $(n_1, n_2, d_1, d_2) = (61/600, 11/19200, 87/200, 47/3840)$ solve $d_1 - n_1 = 1/3$, $d_2 - n_2 - d_1/3 = -2/15$, $P_5(R) = Q_4(R)$, and $(P_5/Q_4)'(R) = 0$ at $R = 4$.

### 1.3 Saturated-Curve Rail Crossing (Pure-`f32` Batched Newton Reciprocal)

`r[dsp.op-n.polyadaa.rail-crossing]`

When consecutive inputs $a = \min(r_{n-1}, r_n) < b = \max(r_{n-1}, r_n)$ cross $\pm 1$, the span $b - a$ partitions into interior length $\ell_0 = \max(0, \min(b, 1) - \max(a, -1))$ and plateau lengths $\ell_- = \max(0, \min(b, -1) - a)$ and $\ell_+ = \max(0, b - \max(a, 1))$, evaluating $\mathcal{A}_{\text{odd}}, \mathcal{A}_{\text{even}}$ on $[\max(a, -1), \min(b, 1)]$:

$$A_{\text{odd,sat}} = \big(\ell_0 \mathcal{A}_{\text{odd}} + \ell_+ - \ell_-\big) \cdot y, \qquad A_{\text{even,sat}} = \big(\ell_0 \mathcal{A}_{\text{even}}\big) \cdot y$$

```text
odd_cross  = ((above - below).fma(inside, odd_int)) * inv  // inv = 2-step NR reciprocal of max(distance, 1.0e-6)
even_cross = (inside * even_int) * inv
```

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

In standalone 1× `Op-N` reference kernels, nonlinear outputs pass through passive quadratic B-spline smoothing with weights 0.875 and 0.125, followed by 1-pole AC coupling at $R_{\text{ac}} = 0.9993$ (48 kHz) before summing with the un-delayed linear residual:

$$\tilde{y}_{\text{nl}}[n] = 0.875 \mathcal{A}[n] + 0.125 \mathcal{A}[n-1], \qquad m_{\text{dc}}[n] = 0.9993 m_{\text{dc}}[n-1] + 0.0007 \tilde{y}_{\text{nl}}[n]$$
$$y[n] = \alpha_c x[n] + \big(\tilde{y}_{\text{nl}}[n] - m_{\text{dc}}[n]\big)$$

This (0.875, 0.125) passive B-spline blend and `R_ac` AC coupling apply strictly to standalone 1× `Op-N` reference kernels and do NOT run inside the 4-block SONA core (§4). Inside the core, each 4-lane `CoreQuad` passes `(A_odd, A_even, lru_state)` directly into lower-triangular inter-block tiles and `LinearReadout` (`W_ro,0`).

#### 1.5.3 Multi-Drive `PolyADAA1` Lane Bank

`r[dsp.op-n.polyadaa.drive-bank]`

Standalone 1× reference `PolyADAA1` multi-drive kernels using profile $d_{\text{bank}} = [0.25, 1.0, 4.0, 16.0]$ retain prior endpoints `(r_prev, u_prev)` and antialiasing states `(last_a_odd, last_a_even)`, whereas each 4-lane SONA `CoreQuad` stores `adaa_r_prev` inside its 128 B stereo state (`[svf_s1, svf_s2, lru_v, adaa_r_prev]`) and evaluates midpoint fallback directly when `|r_n - r_{n-1}| <= 1e-6` (§1.3).

---

## 2. Wavefolder, 2D Hysteresis & ZOH Bitcrusher

### 2.1 Wavefolder Rail Regularity and Candidates

`r[dsp.op-n.wavefolder.cubic-c2]`

To preserve $C^2$ continuity at $\pm 1$ with vanishing value, slope, and curvature at $\bar u = \pm 1$, `Op-N` pairs `PolyADAA1` with cubed-envelope odd and even wavefolders evaluated by one branchless single-precision kernel running identically across CPU SIMD targets and 32-bit GPU training shaders.

Inside every 4-block SONA core group, `wavefolder_mix` with $\mu_{\text{wf}} \in [0, 1]$ convexly blends **only the odd harmonic channel** via $\mathcal{A}_{\text{odd}} \leftarrow (1 - \mu_{\text{wf}})\mathcal{A}_{\text{odd}} + \mu_{\text{wf}} w_{\text{odd,ADAA}}$ while passing `PolyADAA1`'s $\mathcal{A}_{\text{even}}$ unmodified, whereas standalone wavefolder kernels emit both $(A_{\text{odd},3}, A_{\text{even},3})$ using pre-clamped inputs $\bar u = \mathrm{clamp}(u, -1, 1)$:

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

#### 2.3.1 Continuous `Op-N[Pre]` Front-End, Zero-Init Safety & Uniform Nearest-Step Quantizer

`r[dsp.op-n.bitcrusher.quantizer]`

At the Top-of-Frame Crossbar, continuous pre-core conditioning executes through the 3-parameter `Op-N[Pre]` front-end configured by a 12-byte little-endian base wire descriptor at Arena 0 `Crossbar[180..192)` (`base_gamma_bc`, `base_zoh_hold_depth`, and `base_beta_q`, modulated at 3 kHz by `ControlBus` slots 26..28) and backed by the 28-byte 4-byte-aligned `OpNPreHotState` with zero 64-bit runtime integers in the audio callback:

| Region / Offset | Bytes | Fields & Types | Clamped Domain, Zero-Init & Control-Bus Invariant |
| :--- | :---: | :--- | :--- |
| Crossbar Wire 180..192 | 12 | `gamma_bc: f32le`, `zoh_hold_depth: f32le`, `beta_q: f32le` | Base wire `[0.0, 1.0]`, `[0.0, 0.9999]`, `[0.0, 16.0]` (log2 bit-depth above 0; 0 = bypass); slots 26..28 |
| Active-Hot State 0..16 | 16 | `held: f32[2]`, `phase_q32: u32`, `inc_q32: u32` | `OpNPreHotState` prefix: unclipped L/R track-and-hold states plus 32-bit `Q0.32` ZOH accumulator and increment |
| Active-Hot State 16..28 | 12 | `gamma_bc: f32`, `beta_q: f32`, `zoh_hold_depth_eff: f32` | Effective wet blend `[0.0, 1.0]`, hoisted log2 quantizer scale `[0.0, 16.0]`, and ZOH-rescaled hold depth `[0.0, 0.9999]` |

Encoding `zoh_hold_depth = 1.0 - rho_zoh` in `[0.0, 0.9999]` rather than `rho_zoh` directly enforces a critical zero-initialization identity invariant: zero-filled arenas (`zoh_hold_depth == 0.0`) decode `rho_zoh = 1.0` (ZOH bypass), so `g_clk == 1.0` passes every sample transparently instead of latching zero permanently.

Positioned immediately after `Op-T` crossbar addition and ahead of `OS2X` upsampling, `Op-N[Pre]` executes once per VM frame at $f_{s,\text{vm}}$ (`f_{s,host}` in real-time, or `4 * f_{s,host}` in 4× offline bounce where internal `OS2X` is bypassed), so `OS2X` never doubles its rate.

Cold setup defines `rho_vm = 48_000.0 / f_{s,vm}` with zero audio-thread `FDIV`. Micro-ramped `ControlBus` slots 26..28 (`op_n_pre_{gamma_bc,zoh_hold,beta_q}_delta`) update `OpNPreHotState` across cold load, `K_ctrl` ticks, and per-frame VM execution:

- **Clamped targets:** `gamma_bc = clamp(base_gamma_bc + t[26], 0.0, 1.0)`, `z_h = clamp(base_zoh_hold_depth + t[27], 0.0, 0.9999)`, `beta_q = clamp(base_beta_q + t[28], 0.0, 16.0)`.
- **Effective ZOH rate & depth:** `rho_zoh = 1.0 - z_h` yields `rho_eff = clamp(rho_zoh * rho_vm, 1.0e-5, 1.0)` and `z_h_eff = (z_h <= 0.0) ? 0.0 : clamp(z_h^rho_vm, 0.0, 0.9999)`.
- **Scaled increment:** `inc_q32 = (rho_eff >= 1.0 - 1.0e-7 || z_h_eff <= 0.0) ? 0 : max(1, round(rho_eff * 2^32))`, scaling proportionally by `rho_vm` without 64-bit integers.

Per VM sub-block hoist or frame update, `beta_q` prepares `D_q = exp2_clamped16(beta_q)` and `Delta_q = exp2_clamped16(-beta_q)` via `exp2_clamped16` rather than `exp2_clamped5` (§1.2, nearest-integer split, IEEE-754 exponent injection, and degree-5 Horner polynomial on `[-0.5, +0.5]`), incurring zero division inside the 28-byte `OpNPreHotState`.

Over the certified [-100, 100] signal envelope in f32, integer floor splitting computes `h_clip = clamp(held_c, -100.0, 100.0)`, `v = h_clip * D_q`, truncates toward zero to signed i32 `t = v as i32`, forms `k = t - (v < t as f32) as i32` with clamped fractional residual `f = clamp(v - k as f32, 0.0, 1.0)`, wraps `phi_q32` into u32 [0, 2^32) with zero overflow, and drives `q32_smooth_sin(phi_q32)` using compile-time `INV_TAU = 1.0 / (2 * pi)`:

$$\widetilde{Q}(h; \Delta_q) = h_{\text{clip}} - \Delta_q \mathrm{INV-TAU} \cdot \mathrm{q32-smooth-sin}(\phi_{q32})$$

#### 2.3.2 Raised-Cosine Clocked Track-and-Hold & Smooth Fourier Quantizer

`r[dsp.op-n.bitcrusher.zoh-phase]`

Executing once per VM frame at $f_{s,\text{vm}}$ ahead of `OS2X` (`f_{s,host}` in real-time, or `4 * f_{s,host}` in 4× offline bounce with `OS2X` bypassed), stereo `Op-N[Pre]` combines a raised-cosine track-and-hold with a smooth Fourier staircase quantizer using strictly 32-bit runtime integers:

**Raised-Cosine Clock Gain** sets `g_clk = 1.0` when `inc_q32 == 0`, otherwise advancing `phase_q32 = phase_q32.wrapping_add(inc_q32)`, evaluating `cos_phi = q32_smooth_sin(phase_q32.wrapping_add(0x4000_0000))`, `u = 0.5 * (1.0 + cos_phi)`, `w = u * u`, and `g_clk = (1.0 - z_h_eff) + z_h_eff * w`.

**Continuous State Update & Bypass** always updates unclipped `held_c = flush((1.0 - g_clk) * held_c + g_clk * x_c)` per channel and returns `(x_L, x_R)` unchanged (0-ULP pass-through preserving tracked history) whenever `gamma_bc <= 0.0`.

**Smooth Fourier Quantizer & Wet Blend** sets `q_c = clamp(held_c, -100.0, 100.0)` when `beta_q <= 1.0e-6`, otherwise evaluating $\widetilde{Q}(held_c)$ via signed i32 floor-splitting and `q32_smooth_sin` (§2.3.1), and emits `y_c = x_c + gamma_bc * (q_c - x_c)`.

$$w(\phi) = \left(\frac{1 + \cos(2\pi\phi)}{2}\right)^2, \qquad g_{\mathrm{clk}}[n] = (1 - z_{h,\mathrm{eff}}) + z_{h,\mathrm{eff}} w(\phi[n]), \qquad h_c[n] = (1 - g_{\mathrm{clk}}[n]) h_c[n-1] + g_{\mathrm{clk}}[n] x_c[n]$$

When `inc_q32 == 0` (such as `zoh_hold_depth == 0.0` at 48 kHz), `g_clk == 1.0` tracks every input sample cleanly with Zero-DC (`0.0 -> 0.0`) and unconditional BIBO contractivity (`0 <= 1 - g_clk <= 0.9999 < 1`). For standalone fixed-Hertz exact-rate tracking using $F = 2^{32}$ units per Hz, cold preparation computes $D = \mathrm{round}_{\text{even}}(F f_s)$ and $I = \mathrm{round}_{\text{even}}(F f_{\text{target}})$ with $0 \le I \le D$, stepping phase $P \in [0, D)$ via $A = P + I$, carry $b = \mathbf{1}[A \ge D]$, and $P \leftarrow A - bD$.

#### 2.3.3 Division-Free `Q0.32` & Fixed-Hertz Effective-Rate Phase Transfer

`r[dsp.op-n.bitcrusher.rate-transfer]`

Across host sample-rate changes or 4× offline bounce transitions, internal `OS2X` never doubles the `Op-N[Pre]` rate because execution runs ahead of upsampling at $f_{s,\text{vm}}$. Normalized 32-bit `phase_q32` encodes a dimensionless fraction in $[0, 1)$ of one hold period and transfers across rates unchanged without 64-bit integers, while cold and 3 kHz setup rescale `inc_q32` and `z_h_eff` proportionally by $\rho_{\text{vm}} = 48{,}000 / f_{s,\text{vm}}$.

For standalone fixed-Hertz exact-rate tracking, precomputing `u64` reciprocal $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$ on the cold path lets sample-rate transitions form $N = P_{\text{old}} D_{\text{new}} < 2^{102}$ in 128-bit unsigned integer arithmetic, evaluate quotient $q_0 = \lfloor N R / 2^{104} \rfloor$ via bit shifts, refine remainder $r_0 = N - q_0 D_{\text{old}}$, round ties-to-even, and clamp to $D_{\text{new}} - 1$ so imminent carries stay imminent:

$$P_{\text{new}} = \min\left(\mathrm{round}_{\text{even}}\left(\frac{P_{\text{old}}D_{\text{new}}}{D_{\text{old}}}\right), D_{\text{new}}-1\right), \qquad q_0 = (h \ll 24) \mathbin{\mathrm{or}} \big((m \mathbin{\mathrm{and}} M) \gg 40\big)$$

Here $N = N_{\text{hi}}2^{64} + N_{\text{lo}}$, $a = N_{\text{lo}}R$, $c = N_{\text{hi}}R$, $M = 2^{64} - 1$, $m = (a \gg 64) + (c \mathbin{\mathrm{and}} M)$, and $h = (c \gg 64) + (m \gg 64)$, keeping frequency error $|f_s(I/D) - f_{\text{target}}| \le 2^{-32}$ Hz.

#### 2.3.4 Exact $C^\infty$ Training & 1-D Affine Track-and-Hold Gradients

`r[dsp.op-n.bitcrusher.surrogate]`

Because raised-cosine track-and-hold is a 1-D scalar affine recurrence, with $0 \le 1 - g_{\text{clk}}[n] < 1$ , and $\widetilde{Q}(h)$ is $C^\infty$ differentiable with zero discontinuous jumps, WebGPU training executes exact forward and reverse adjoint passes using the shared 1-D affine `LinearScanElem` scan across $(x, z_h, \omega_{\text{clk}}, \Delta_q, \gamma_{\text{bc}})$ with zero straight-through estimators (`STE`).

---

## 3. Exact LTI Basis Commutativity Theorem (Theorem 1)

### 3.1 Formal Theorem Statement & Proof

`r[dsp.op-n.commuted.theorem1]`

SONA v3.0 applies Theorem 1 to fold `LinearReadout` 3 kHz linear projections (`pre_conv` and `direct_s2`) directly inside SIMD registers across the 4-block core with zero intermediate scratch gather buffers.

**Theorem 1 (Exact LTI Basis Commutativity).** Let $\mathbf{u}[n] \in \mathbb{R}^P$ be bounded per-channel block activations satisfying $\sup_{n, p} |u_p[n]| \le U_{\max} < \infty$, let $\lbrace \mathcal{L}_k \rbrace_{k=0}^{K-1}$ be causal, stable LTI filters with impulse responses $h_k \in \ell_1(\mathbb{Z})$, and let $\lbrace \mathbf{W}_k \rbrace_{k=0}^{K-1} \subset \mathbb{R}^{P \times C}$ be constant projection matrices. Then for every output channel $c \in [0, C-1]$ and sample $n \in \mathbb{Z}$:

$$\sum_{k=0}^{K-1} \Big(\mathcal{L}_k\big(\mathbf{u}\big)[n]\Big) \mathbf{W}_k \equiv \sum_{k=0}^{K-1} \mathcal{L}_k\Big(\mathbf{u} \mathbf{W}_k\Big)[n]$$

*Proof.* Absolute convergence follows from $h_k \in \ell_1(\mathbb{Z})$ and boundedness of $\mathbf{u}$, allowing summation interchange by Fubini's theorem so $\mathbf{u}\mathbf{W}_k$ filters directly across $C$ SIMD lanes. $\blacksquare$

---

### 3.2 Causality Audit & Elimination of Acausal Lookahead

`r[dsp.op-n.commuted.causality]`

All Top-of-Frame Crossbar `Op-T` reads where $\tau_{\min} = 5.0$ samples across 4-frame causal micro-chunks (and $\tau \ge 4.0$ on standalone single-sample rings), `TPT-SVF` / `ParaLRU` / `PolyADAA1` updates, `poly3` decimation filters, and `ModalConvolver` interfaces in SONA v3.0 are strictly causal with $d \ge 0$, maintaining sample-accurate phase alignment across 1× and 2× (4× bounce) paths.

---

### 3.3 Commutator Perturbation Under 3 kHz Control-Rate Modulation

`r[dsp.op-n.commuted.project-then-filter]`

Under 3 kHz control modulation of core filters, drive nonlinearities, and readout matrices, ControlBus advances all 32 slots in `current_stage_params` uniformly per VM frame across `K_ctrl(f_{s,vm}) = max(8, round(f_{s,vm} / 3_000.0))` (including slot 16 `freeze_mix_drive` and slot 30 `sense_vca_gain_delta`), while recursive slews (Kautz `prev_oct` and Op-T `current_psi`) update strictly when `ctrl_sub_counter == 0`.

Hot `CoreQuad`, crossbar, `Op-N[Pre]`, and `Op-T` registers latch `(current_stage_params, eff_z_stage)` strictly when `(sample_tick & 3) == 0 || ctrl_sub_counter == 0`, entering 4-frame micro-chunks only when `(sample_tick & 3) == 0 && rem_frames >= 4 && rem_tick >= 4` and otherwise peeling 1 frame while holding latched registers across `(sample_tick & 3) in 1..4`.

Because both `ControlBus` increments and register-latch boundaries depend solely on sample phase, `process(1) * N == process(4) * (N/4) == process(16) * (N/16)` holds at 0 ULP across every sample rate (including 44.1 kHz with `K_ctrl = 15`) even under active control sweeps, bounding the commutator residual against continuous evaluation:

$$\Delta_{\text{comm}}[n] \le \frac{B_u \nu_{\text{ctrl}}}{f_s} \cdot \frac{r_k}{1 - r_k}$$

---

### 3.4 Boundary Conditions, Resting DC Equilibrium & Pop-Free Startup

`r[dsp.op-n.commuted.dc-equilibrium]`

With decoupled 1-D `ParaLRU` recurrence $v_i[n] = \alpha_{\text{rel},i} v_i[n-1] + (1 - \alpha_{\text{atk},i})\phi_{\text{rect}}(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1])$ satisfying $\phi_{\text{rect}}(0) = 0$, analytical zero-DC resting bias centering at `StarCenter` where $u_{\text{eq}} = 0.0$, `PolyADAA1` zero equilibrium satisfying $g_{\mathrm{odd}}(+0.0) = g_{\mathrm{even}}(+0.0) = +0.0$, and zero-initialized integrators, all recursive state settles at exact 0.000000 DC equilibrium (`h_eq == +0.0`, `b_head == 0.0`):

1. Core `ParaLRU` lanes initialize at $v_{\text{init}} = +0.0$ across 16, 24, 32, or 40 lanes per channel from Lite through Ultra.
2. `TPT-SVF` and AC coupling filters initialize states $s_1 = s_2 = +0.0$ at idle equilibrium.

---

## 4. 4-Block Matryoshka `CoreQuad` Stack (`TPT-SVF`, Selective-Duhem `ParaLRU`, `PolyADAA1` & `GLU`)

`r[dsp.op-n.core.matryoshka-blocks]`

SONA v3.0 models nonlinear preamps, magnetic hysteresis, stereo-linked bus compression, crossovers, and sweeps via a 4-block Matryoshka state-space core sorted across 10 logical 4-lane CoreQuads `[(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (0,2), (2,1), (0,3), (1,2)]` and 35 causal lower-triangular inter-block mixing tiles. Each CoreQuad chains M/S input coupling (`in_mid_weight_fp16`, `in_side_weight_fp16`), passive 2×2 TPT-SVF, Selective-Duhem stereo-linked 1-D ParaLRU, pure f32 PolyADAA1 plus C² wavefolder, and bounded cross-lane GLU across the four tier profiles:

| Tier Profile | `CoreQuad`s & Lanes | Inter-Block Tiles | Core Rate | `TPT-SVF` Quads / Modes | `ModalConvolver` (FIR / Kautz) | `TransportQuad`s (`Lines`) | Hot Working Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lite `[1, 1, 1, 1]` | 4 quads (16 lanes) | 3 | 1× `PolyADAA1` | 4 quads (16 modes) | 32 taps / 8 pairs | 1 quad (4 lines) | <= 4,608 B |
| Full `[2, 2, 1, 1]` | 6 quads (24 lanes) | 13 | 2× OS (96 kHz) | 6 quads (24 modes) | 48 taps / 12 pairs | 2 quads (8 lines) | <= 7,936 B |
| Pro `[3, 2, 2, 1]` | 8 quads (32 lanes) | 23 | 2× OS (96 kHz) | 8 quads (32 modes) | 96 taps / 24 pairs | 3 quads (12 lines) | <= 15,872 B |
| Ultra `[4, 3, 2, 1]` | 10 quads (40 lanes) | 35 | 2× OS (4× Bounce) | 10 quads (40 modes) | 128 taps / 32 pairs | 4 quads (16 lines) | <= 28,672 B |

With `SOFT_RECT_EPS` $\varepsilon = 10^{-3}$ and `SOFT_RECT_LARGE_Z = 1.0e15`, the zero-DC soft rectifier sets $z_{\text{safe}} = \min(|z|, 10^{15})$, $q = z_{\text{safe}}^2 + \varepsilon^2$, evaluates $r_{\text{inv}} = q^{-1/2}$ via a 3-step Newton inverse square root, $r_{\text{rad}} = q \cdot r_{\text{inv}}$, and evaluates $w_\varepsilon = (r_{\text{rad}} + \varepsilon)^{-1}$ and $w_z = (r_{\text{rad}} + z_{\text{safe}})^{-1}$ via 3-step Newton reciprocals across three cancellation-free branches for $\phi_{\text{rect}}(z) = \frac{1}{2}(z + \sqrt{z^2 + \varepsilon^2} - \varepsilon)$:

- Zero branch for $|z| \le 0.0$: emits canonical +0.0 (`0x0000_0000`) with derivative 0.5.
- Positive branch for $z > 0$: evaluates $\phi_{\text{rect}}(z) = \frac{1}{2}(z_{\text{safe}} + z_{\text{safe}}^2 w_\varepsilon)$ and $\phi_{\text{rect}}'(z) = \frac{1}{2}(1 + z_{\text{safe}} r_{\text{inv}})$.
- Negative branch for $z < 0$: uses rationalized evaluation to avoid square-root cancellation, saturating at $(-\varepsilon/2, 0.0)$ when $|z| > 10^{15}$:

$$\phi_{\text{rect}}(z) = -\frac{\varepsilon}{2} (z_{\text{safe}} w_z)(1 + z_{\text{safe}} w_\varepsilon), \qquad \phi_{\text{rect}}'(z) = \frac{\varepsilon^2 r_{\text{inv}}}{2(r_{\text{rad}} + z_{\text{safe}})}$$

Every `ParaLRU` lane clamps $\alpha_{\text{rel}} \in [0.0, r_{\max}]$ with `MAX_PARALRU_R_F32 = 0x3F7F_FFAC` (0.999995), $\alpha_{\text{atk}} \in [0.0, 1.0]$, $\kappa \in [0.0, 1.0]$, stereo link $\lambda \in [0.0, 1.0]$, Duhem derivative gate $\gamma_{\text{duhem}} \in [0.0, 64.0]$, and signed blend $\eta \in [0.0, 1.0]$. Reusing the `TPT-SVF` flux increment $\Delta s_{2,i}[n] = 2 g_i \mathrm{BP}_i[n]$ in registers with zero extra state (`Mamba` / Duhem gating):

$$\rho_i[n] = \mathrm{clamp}\big((1 - \alpha_{\text{atk},i}) + \gamma_{\text{duhem},i} \lvert 2 g_i \cdot \mathrm{BP}_i[n] \rvert, 0.0, 1.0\big)$$
$$d_i[n] = \alpha_{\text{rel},i} v_i[n-1], \qquad z_i[n] = s_i u_{\text{tone},i}[n] - \kappa_i (1 - \eta_i) d_i[n]$$
$$\bar{z}_i[n] = \mathrm{clamp}\big(z_i[n], -1.0, 1.0\big), \qquad \psi_1\big(z_i[n]\big) = \bar{z}_i[n]\left(1.0 - \frac{1}{3}\bar{z}_i[n]^2\right)$$
$$\phi_{\eta,i}\big(z_i[n]\big) = (1 - \eta_i)\phi_{\text{rect}}\big(z_i[n]\big) + \eta_i\psi_1\big(z_i[n]\big)$$
$$v_i[n] = \mathrm{flush}\Big(\big(1 - \eta_i \rho_i[n]\big) d_i[n] + \rho_i[n] \phi_{\eta,i}\big(z_i[n]\big)\Big)$$
$$J_i[n] = \frac{\partial v_i[n]}{\partial v_i[n-1]} = \alpha_{\text{rel},i}\Big((1 - \eta_i \rho_i[n]) - \rho_i[n] \kappa_i (1 - \eta_i)\phi_{\eta,i}'(z_i[n])\Big) \in \big[(1 - \rho_i[n])\alpha_{\text{rel},i}, \alpha_{\text{rel},i}\big] \subset [0, 1)$$

Because $J_i[n] \in [(1 - \rho_i[n])\alpha_{\text{rel},i}, \alpha_{\text{rel},i}] \subset [0, 1)$, Selective-Duhem ParaLRU is unconditionally contractive without post-hoc ISS clamping. At $\gamma_{\text{duhem}} = 0, \eta = 0$, it reduces bit-for-bit to the scalar diode-RC rectifier, while $\eta = 1, \alpha_{\text{atk}} = 1, \gamma_{\text{duhem}} > 0$ holds coercive magnetic remanence across extrema as $\Delta s_2 \to 0$.

WebGPU training runs joint 4-thread `(L, R)` in-SRAM stepping (<= 15.0 KiB, [webgpu_and_varpro_solver.md](../training/webgpu_and_varpro_solver.md)) so stereo linking $\lambda_i$ requires zero cross-workgroup sync and matches same-FMA-class CPU 4-lane execution at 0-ULP parity (<= 1.5 ULP/step across FMA vs non-FMA CPU/GPU targets).

Each CoreQuad applies audio-rate stereo linking $v_{\mathrm{eff},c,i} = v_{c,i} + \lambda_i(v_{\mathrm{mid},i} - v_{c,i})$, where $v_{\mathrm{mid},i} = 0.5(v_{L,i} + v_{R,i})$ , right after updating `(v_L, v_R)`. Downstream `sag_gain = max(0.05, (1.0).fms(sag_depth, v_eff))` and `lru_bias_eff` consume `v_eff`, preserving 0° image drift at $\lambda_i = 1$ . Effective bias `b_lru_eff` (`b_lru_base = base_lru_bias_drive`) drives PolyADAA1 via `r_curr = ((tone_in * d_adaa_eff) * sag_gain).fma(b_lru_eff, v_eff)`, where $s_{\mathrm{rect}}$ denotes `rect_sign`:

$$b_{\mathrm{lru,eff}} = \mathrm{clamp}\big(b_{\mathrm{lru,base}} + s_{\mathrm{rect}} \cdot \theta_{\mathrm{lru}}, -16.0, +16.0\big)$$

Mechanism A2 watermarking flips `in_mid_weight`, `in_side_weight`, `b_lru_base`, and `rect_sign` by lane sign $s_l \in \lbrace -1, +1 \rbrace$ (keeping $|2 g \cdot \mathrm{BP}_l|$ and $\rho_l$ bit-identical and $v_l$ invariant at 0 ULP across $\eta \in [0, 1]$ ) and flips `glu_mix[l]` by $s_l \cdot s_{(l+1)\bmod 4}$, preserving 0-ULP forward audio equivalence across static and modulated states ([parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws)).

For backward gradients inside the reverse-time adjoint scan tile, recomputing `(v_L, v_R)` from checkpoints gives linear reverse-time scan elements `(J_i, J_i * adj_v[i])` over $N + 1$ inclusive suffix slots starting from sentinel `suffix[N] = (1.0, 0.0)`:

$$e_i = (a_i, b_i) = \big(J_i, J_i \bar{v}_i\big), \qquad e_{\mathrm{id}} = (1.0, 0.0)$$
$$\mathrm{compose}(e_{\mathrm{first}}, e_{\mathrm{second}}) = \big(e_{\mathrm{second}}.a \cdot e_{\mathrm{first}}.a, e_{\mathrm{second}}.a \cdot e_{\mathrm{first}}.b + e_{\mathrm{second}}.b\big)$$
$$\mathtt{suffix}[N] = (1.0, 0.0), \qquad \mathtt{suffix}[i] = \mathrm{compose}\big(\mathtt{suffix}[i+1], e_i\big)$$

Because stereo linking transforms `(v_L, v_R) -> (v_eff,L, v_eff,R)` outside the recurrence state `v`, the backward pass backpropagates `adj_v` through the linear stereo-link transpose prior to the suffix scan, propagates $\bar{v}_{\text{out}}$ across tiles, and propagates $\nabla_{\rho_i} L$ back into $\nabla_{\mathrm{BP}_i} L$ through the Selective-Duhem gate:

$$\bar{v}_{c} \mathrel{+}= \big(1 - 0.5\lambda\big)\bar{v}_{\mathrm{eff},c} + 0.5\lambda\bar{v}_{\mathrm{eff},\bar{c}}, \qquad \lambda_i = \bar{v}_i + \mathtt{suffix}[i+1].a \cdot \bar{v}_{\text{out}} + \mathtt{suffix}[i+1].b$$
$$\nabla_{\rho_i} L = \lambda_i \big(\phi_{\eta,i}(z_i) - \eta_i d_i\big), \qquad \nabla_{\mathrm{BP}_i} L \mathrel{+}= \nabla_{\rho_i} L \cdot \mathbf{1}_{\rho \in (0, 1)} \cdot 2 g_i \gamma_{\text{duhem},i} \mathrm{sign}(\mathrm{BP}_i)$$
$$\nabla_{\alpha_{\text{rel}}} L = \sum_{i=0}^{N-1} \lambda_i v_{i-1}\Big((1 - \eta_i \rho_i) - \rho_i \kappa_i (1 - \eta_i)\phi_{\eta,i}'(z_i)\Big), \qquad \nabla_{\gamma_{\text{duhem}}} L = \sum_{i=0}^{N-1} \nabla_{\rho_i} L \cdot \mathbf{1}_{\rho \in (0, 1)} \cdot |2 g_i \mathrm{BP}_i|$$

```text
// Inclusive Reverse-Time Adjoint Suffix Scan Kernel (N + 1 Slots per Lane, Joint 4-Thread L/R Workgroup)
1. Recompute v_L,traj[0..N] & v_R,traj[0..N] in workgroup SRAM from saved tile-start checkpoints v_init.
2. Propagate downstream GLU/wavefolder adjoints through stereo-link transpose -> adj_v[c][i].
3. Form local reverse-scan elements for i in 0..N: e[i] = { a: J[i], b: J[i] * adj_v[i] }.
4. Run O(log2 N) inclusive reverse Blelloch suffix scan over N + 1 slots with compose(suffix[i+1], e[i]):
      suffix[N] = { a: 1.0, b: 0.0 }
      suffix[i].a = e[i].a * suffix[i+1].a
      suffix[i].b = e[i].a * suffix[i+1].b + e[i].b
5. Propagate incoming boundary adjoint adj_v_out from tile + 1 and backpropagate Selective-Duhem adj_rho -> adj_BP:
      lambda[i] = adj_v[i] + suffix[i+1].a * adj_v_out + suffix[i+1].b
      adj_v_init = suffix[0].a * adj_v_out + suffix[0].b   // passed backward as adj_v_out of tile - 1
```

For ZOH sample-rate transfer at $\rho_{\text{sr}} = f_{s,\text{ref}} / f_{s,\text{eff}}$, cold helpers `rescale_tpt_svf_g_zoh` and `powf_f64` ([router_and_crossbar_topology.md §9.1](router_and_crossbar_topology.md#91-causal-crossbar-sensoryquad-core-tpt-svf--kautz-conversion)) clamp $g_{\text{ref}} \in [10^{-5}, 64.0]$ for `TPT-SVF` while `ParaLRU` rescales $(\alpha_{\text{rel}}', \alpha_{\text{atk}}', \kappa')$ with $\beta = 1 - \alpha_{\text{atk}}$ and $a_{\text{atk}} = \alpha_{\text{rel}}(1 - \kappa\beta)$:

$$g_{\text{host}} = \mathrm{clamp}\Big(\tan\big(\mathrm{clamp}(\rho_{\text{sr}} \arctan(g_{\text{ref}}),\ 10^{-5}\pi,\ 0.495\pi)\big),\ 10^{-5},\ 64.0\Big)$$
$$\alpha_{\text{rel}}' = \mathrm{clamp}\big(\alpha_{\text{rel}}^{\rho_{\text{sr}}}, 0.0, r_{\max}\big), \qquad \beta' = \mathrm{clamp}\left(\beta \frac{1 - a_{\text{atk}}^{\rho_{\text{sr}}}}{1 - a_{\text{atk}}}, 0.0, 1.0\right), \qquad \alpha_{\text{atk}}' = 1 - \beta'$$

The ratio $(1 - a_{\text{atk}}^{\rho_{\text{sr}}}) / (1 - a_{\text{atk}})$ approaches $\rho_{\text{sr}}$ continuously as $a_{\text{atk}} \to 1$, and $\kappa' = \mathrm{clamp}((1 - (1 - \kappa\beta)^{\rho_{\text{sr}}}) / \beta', 0.0, 1.0)$ when $\beta' > 10^{-12}$ (retaining $\kappa$ otherwise). Every lane stays unconditionally BIBO-stable (`REQ-7.2`), while direct `LinearReadout` sums (`pre_conv` and `direct`) carry unclipped ±100.0 (+40 dBFS) headroom (`REQ-7.1`).

---

### 4.1 Passive 2×2 ZDF `TPT-SVF` Four-Filter Tone Basis & Associative Scans

`r[dsp.op-n.core.tpt-tone-basis]`

Each 2×2 ZDF `TPT-SVF` mode clamps $g \in [10^{-5}, 64.0]$ and $k \in [0.0, 8.0]$, where $f_c = (f_s/\pi)\arctan(g)$, evaluates division-free $a_1 = (1 + g(g + k))^{-1}$ via a 2-step Newton reciprocal for arguments $\ge 1$ (within 1 ULP of 3-step refinement), $a_2 = g a_1$, and $a_3 = g a_2$, and updates subnormal-flushed states $s_1, s_2$:

$$v_3 = u - s_2, \qquad v_1 = a_1 s_1 + a_2 v_3, \qquad v_2 = s_2 + a_2 s_1 + a_3 v_3$$
$$\mathrm{bp} = v_1, \qquad \mathrm{lp} = v_2, \qquad \mathrm{hp} = u - k v_1 - v_2, \qquad \mathrm{ap} = u - 2 k v_1$$
$$s_1' = \mathrm{flush}(2 v_1 - s_1), \qquad s_2' = \mathrm{flush}(2 v_2 - s_2)$$

Linear tap mix `y_tone` (`binary16` weights in [-16.0, 16.0]) evaluates left-to-right chained vector FMA `((((lp * w_lp).fma(bp, w_bp)).fma(hp, w_hp)).fma(ap, w_ap)).fma(u, w_dir)` where receiver-first `acc.fma(a, b) = fma(a, b, acc)`, lowering to hardware IEEE-754 FMA on FMA SIMD backends and `acc + (a * b)` on non-FMA scalar, WebAssembly SIMD128, and GPU compute targets (`<= 1.5` ULP/step per [sona_architecture.md §1.5.3](../architecture/sona_architecture.md#153-single-library-per-architecture-runtime-simd-dispatch--0000000-ulp-parity)) to discover arbitrary crossovers, tone stacks, and phasers solely via data (`REQ-1.9`). In matrix form $\mathbf{s}_{n+1} = \mathbf{A}_{\text{tpt}}\mathbf{s}_n + \mathbf{b}_{\text{tpt}} u_n$:

$$(\mathbf{A}_{\text{tpt}})_{11} = 2 a_1 - 1, \quad (\mathbf{A}_{\text{tpt}})_{12} = -2 a_2, \quad (\mathbf{A}_{\text{tpt}})_{21} = 2 a_2, \quad (\mathbf{A}_{\text{tpt}})_{22} = 1 - 2 a_3, \quad b_1 = 2 a_2, \quad b_2 = 2 a_3, \quad \Vert \mathbf{A}_{\text{tpt}} \Vert_2 \le 1$$

Forward and backward WebGPU scans compose via $(\mathbf{A}_2, \mathbf{b}_2) \circ (\mathbf{A}_1, \mathbf{b}_1) = (\mathbf{A}_2 \mathbf{A}_1, \mathbf{A}_2 \mathbf{b}_1 + \mathbf{b}_2)$ and reverse transpose $(\mathbf{A}_1^\top, \mathbf{g}_1) \circ_{\text{rev}} (\mathbf{A}_2^\top, \mathbf{g}_2) = (\mathbf{A}_1^\top \mathbf{A}_2^\top, \mathbf{A}_1^\top \mathbf{g}_2 + \mathbf{g}_1)$.

---

### 4.2 Stereo M/S Input, Lower-Triangular Inter-Block Mixing & Bounded `GLU`

`r[dsp.op-n.core.interblock-tiles]`

Inter-lane mixing across the 4-block `CoreQuad` stack visits active quads strictly in block-major `SONA_BLOCK_QUAD_SLOTS` order (`Block 0 -> 1 -> 2 -> 3`, matching `[0, 4, 6, 8, 1, 5, 9, 2, 7, 3]` at Ultra) with strict tier prefix nesting:

| Block (`blk`) | Active Count (`B_{blk}` at Lite / Full / Pro / Ultra) | `SONA_BLOCK_QUAD_SLOTS[blk]` (`-` unused) |
| :--- | :---: | :--- |
| Block 0 | 1 / 2 / 3 / 4 | `[0, 4, 6, 8]` |
| Block 1 | 1 / 2 / 2 / 3 | `[1, 5, 9, -]` |
| Block 2 | 1 / 1 / 2 / 2 | `[2, 7, -, -]` |
| Block 3 | 1 / 1 / 1 / 1 | `[3, -, -, -]` |

```text
u_mid = 0.5 * (u_L + u_R);  u_side = 0.5 * (u_L - u_R)
u_acc = 0.0
for prev_blk = 0..blk, s = 0..B_{prev_blk}, src_g = SONA_BLOCK_QUAD_SLOTS[prev_blk][s]:
  if edge_in_tier_prefix(src_g, dst_g, T_CAP): // Lite (T_CAP = 3): prev_blk == blk - 1 only; Full..Ultra: all prev_blk < blk
    u_acc = step_tile_odd(u_acc, w_odd_4x4, A_odd_wf[src_g])
    u_acc = step_tile_even(u_acc, w_even_4x4_fp16, A_even_glu[src_g])
    u_acc = step_tile_lru(u_acc, w_lru_4x4_fp16, v_eff[src_g])
u_{g,L}[n] = (u_acc_L + in_mid_weight * u_mid + in_side_weight * u_side) + ((op_p_to_core * (op_p_gain * T1)) + (phasor_to_core * car_L))
u_{g,R}[n] = (u_acc_R + in_mid_weight * u_mid - in_side_weight * u_side) + ((op_p_to_core * (op_p_gain * T1)) + (phasor_to_core * car_R))

// Bounded GLU evaluated after wavefolder_mix with +0.0 canonicalization:
A_odd_wf = (1 - mu_wf) * A_odd + mu_wf * wf_odd
A_even_glu[i] = canonical_zero_pos(A_even[i] + glu_mix[i] * (A_odd_wf[i] * A_odd_wf[(i + 1) & 3]))

Active CoreQuads in SONA_BLOCK_QUAD_SLOTS order -> wire LinearReadout W_ro,0[g] (g in 0..N_g, l = 4 * g + local_lane)
  --[ (acc.fma(ro_odd, A_odd_wf)).fma(ro_even, A_even_glu).fma(ro_lru, v_eff) + (a_0+a_1)+(a_2+a_3) ]--> (pre_conv, direct_s2)
```

For each destination CoreQuad `g = SONA_BLOCK_QUAD_SLOTS[blk][d]`, the inter-block loop evaluates edges inside tier prefix `0..T_CAP` (`T_CAP` in {3, 13, 23, 35}) in `odd -> even -> lru` order into `u_acc`, adding voltage-normalized M/S input drive `(in_mid_weight * u_mid ± in_side_weight * u_side)` plus crossbar `(op_p_to_core, phasor_to_core)` terms.

Lite (`T_CAP = 3`) executes strictly adjacent-block edges `prev_blk == blk - 1`, whereas Full, Pro, and Ultra evaluate all causal edges `prev_blk < blk`. Inside each quad after `wavefolder_mix`, cross-lane GLU multiplies adjacent odd channels `A_odd_wf[i] * A_odd_wf[(i+1)&3]` weighted by `glu_mix_fp16` in [-4.0, 4.0] onto A_even with +0.0 canonicalization, encoding bilinear multiplicative gating without overflow (|A_odd_wf| <= 1.0).

Every inter-block tile (`w_odd_4x4`, `w_even_4x4_fp16`, `w_lru_4x4_fp16`) uses column-major `[src_lane][dst_lane]` storage (source lane 0..4 by destination lane 0..4), evaluating `p01 = (c0 * x_0).fma(c1, x_1)`, `p23 = (c2 * x_2).fma(c3, x_3)`, and unfused `u_acc = u_acc + (p01 + p23)` per [watermarking_and_marketplace.md §1.1](../security/watermarking_and_marketplace.md#mechanism-a1-reciprocal-kautz-gauge--mechanism-a2-0-ulp-40-lane-sona-sign-flip-gauge).

Binary LinearReadout `W_ro,0` (`w_ro_fp16` or `w_ro_and_mods_fp16`) serializes lanes `l = 4 * g + local_lane` by tier-introduction global quad g in 0..N_g and unpacks per CoreQuad. Runtime execution visits active quads in block-major `SONA_BLOCK_QUAD_SLOTS` order ([0, 4, 6, 8, 1, 5, 9, 2, 7, 3] at Ultra) and accumulates `(acc.fma(ro_odd, A_odd_wf)).fma(ro_even, A_even_glu).fma(ro_lru, v_eff)`, adding downstream `ro_pole_48k` readout damping before routing pre_conv to ModalConvolver and direct_s2 to Transport.

Unlike standalone 1× `Op-N` reference kernels (§1.5.2), the 4-block `CoreQuad` stack omits the (0.875, 0.125) passive B-spline blend and `R_ac` AC coupling, passing `(A_odd_wf, A_even_glu, v_eff)` directly into downstream tiles and `LinearReadout` weights.

---

### 4.3 Wide-ISA 8-Lane Stereo & Interleaved 4-Lane SIMD Execution

`r[dsp.op-n.core.stereo-simd]`

Inside 16-frame register-hoisted sub-blocks, SONA steps Left (4 lanes) and Right (4 lanes) simultaneously in 256-bit or scalable 8-lane stereo SIMD registers or interleaved 4-lane 128-bit vector pairs. Pinning filter coefficients, `ParaLRU` parameters, `PolyADAA1` history, and mixing tiles in SIMD registers yields 0-ULP parity across same-FMA-class 8-lane stereo and interleaved 4-lane backends with fixed 4-lane pair reduction trees (`<= 1.5` ULP/step across FMA vs non-FMA targets per [sona_architecture.md §1.5.3](../architecture/sona_architecture.md#153-single-library-per-architecture-runtime-simd-dispatch--0000000-ulp-parity), [ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

### 4.4 3 kHz Gauge-Invariant `z_stage` Control Sensitivity & Static `W_ro,0` Readout

`r[dsp.op-n.core.control-sensitivity]`

`SO(4)` buyer gauge `Q_buyer` rotates only `z_head[0..4]` along with columns 0..4 of `main_base`, `u_int_base`, `w_ctrl`, and `u_int_ultra` Block 0.

Unrotated `z_stage` spans `ctrl_basis_4` for Lite..Pro (`D_s = 4`) and adds `z_stage_ultra` for Ultra (`D_s = 8`, columns 4..8 and Blocks 1..2).

The readout tier block stores one static base matrix `W_ro,0` (sized 16×12, 24×12, 32×12, or 40×12 across tiers) with zero per-latent matrices or wire biases. At each 3 kHz boundary, `z_head` stays inside `W_ctrl` (`f32`) while `stage_latent_*` half-precision sensitivity tables index strictly unrotated `z_stage`, keeping 0-ULP `SO(4)` watermark invariance.

| Modulation Reference | Scope & Canonical Definition |
| :--- | :--- |
| Sensitivity Laws (SSOT) | `stage_targets[0..32]` (0..15 base + 16..31 optional `"CEXT"`), `stage_latent_mods_fp16[0..80]`, Ultra `stage_latent_ctrl_ultra_fp16[0..256]`, and `SONA_META_CORE_MODS_PRESENT`-gated `stage_latent_core_mods_fp16[0..1024]` ([parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws)) |
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

### 5.3 Skewed Quadrature LFO Waveshaping (Second-Harmonic Chebyshev)

`r[dsp.op-n.ctrl.chebyshev-lfo]`

Quadrature LFO states evaluate `(c_j, s_j)` per oscillator and apply learned second-degree Chebyshev harmonic skew `T_2(x) = 2x^2 - 1` weighted by `beta_j = lfo_skew_fp16[j]` in `[-1.0, 1.0]`, emitting `[c_0 + beta_0 * (2*c_0^2 - 1), s_0, c_1 + beta_1 * (2*c_1^2 - 1), s_1]` into control vector `q2`, while higher-order Chebyshev terms $T_3(x) = 4x^3 - 3x$ and $T_4(x) = 8x^4 - 8x^2 + 1$ remain available in standalone control kernels.

### 5.4 Rank-4/8 FM Bilinear Interaction

`r[dsp.op-n.ctrl.fm-interaction]`

Pairwise FM interactions combine scalar embeddings $z_p = \mathbf{w}_{\text{proj},p}^\top\boldsymbol{\psi}_p(u_p)$ through $\mathbf{U}_{\text{int}}$, a 4×8 base matrix in Arena 0 (`[4][8]`, 4 FM factors by 8 latent coordinates) plus three 4×8 delta blocks in Arena 3.

### 5.5 Groupwise Trust-Region Clipping

`r[dsp.op-n.ctrl.trust-clip]`

Outward-checked trust-region clipping runs independently on `z_head[0..4]`, `ctrl_basis_4[0..4]`, `z_head_ultra[0..4]`, and `z_stage_ultra[0..4]` via `trust_clip_outward_4d` using scalar `trust_radius_mh` per [parametric_control_bus.md §1.2.1](parametric_control_bus.md#121-outward-checked-trust-region-clip).
