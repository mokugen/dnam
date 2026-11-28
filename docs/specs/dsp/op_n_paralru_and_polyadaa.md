---
id: spec.dsp.op-n
title: "Op-N: Contractive ParaLRU, Divided-Difference PolyADAA1, Wavefolder & Bitcrusher"
status: canonical
version: "4.0.0"
domain: dsp
rule_prefixes:
  - "r[dsp.op-n."
requirements:
  - REQ-1.1
  - REQ-1.9
  - REQ-6.2
  - REQ-7.1
  - REQ-7.2
  - REQ-8.5
adrs:
  - ADR-0019
  - ADR-0022
  - ADR-0024
  - ADR-0026
---

# Op-N: Contractive `ParaLRU`, Divided-Difference `PolyADAA1`, Wavefolder & Bitcrusher

This specification defines Op-N (Nonlinear and Parallel State-Space) for Dynamic Neural Audio Modeler (DNAM) SONA v4.0 ([sona_architecture.md](../architecture/sona_architecture.md), [REQ-1.1, REQ-6.2](../../requirements/system_goals_and_requirements.md), [ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md), [ADR-0024](../../adr/0024-sona-v4-kernel-hb-fir-perimeter-and-control-bus-gra.md), [ADR-0026](../../adr/0026-core-only-multirate-offline-and-crossbar-latency-invariance.md)), covering four interlocking primitives:

1. CoreOversampler (Live 2× / Master up to 4× + `Box2`) 4-block Matryoshka core (`[1,1,1,1]`, `[2,2,1,1]`, `[2,2,2,2]`, `[3,3,2,2]`, max 7 outgoing tiles/quad) with Quad-Major execution.
2. Smooth non-saturating `ParaLRU` (C² quartic `soft_rect`), 3-tap folded `TPT-SVF`, Predictor-Corrector passive PHS (Block 0), and 2-block partial-fraction LR4 scans.
3. Clamped `AlgADAA1(0.5)` (`MAX_ALG_R = 4096.0`, NR3) with GL/Möbius backward adjoints, `g_pre(v²)` + SE-Bus + GLU, explicit `rect_sign_mask`, and 1× `PolyADAA1` references.
4. Op-N[Pre] v4 (`quantize -> ZOH`: 4-FMA Hermite quantizer + hard-capable `Q0.32` wrap ZOH) and `Op-N[Ctrl]` (`ControlBus` v4 GRA).

---

## 1. Pure-`f32` Divided-Difference `PolyADAA1` & Antialiased Activations

### 1.1 Pre-Activation Clamping, Core Sag Scaling & Standalone Reference

`r[dsp.op-n.polyadaa.clamp]`

Inside every SONA v4.0 CoreQuad block, linear signal paths clamp only at the ±1e9 safety boundary (`REQ-7.1`, `REQ-7.2`). `AlgADAA1(0.5)` clamps `r_curr = clamp(g_pre * d_adaa * u_tone + b_bias, -4096.0, +4096.0)` (`MAX_ALG_R = 4096.0`, where cubic 4096³ = 6.87e10 << 3.40e38) **before** storing `adaa_prev`, eliminating `f32` overflow under ±1e9 inputs.

Core sag scaling replaces `max(0.05, 1.0 - d_sag * v_lru)` with $C^\infty$ bounded pre-ADAA gain $g_{\mathrm{pre}}(v_{\mathrm{eff}}^2)$ using $e_i = \min(v_{\mathrm{eff},i}^2, 10^6)$ and `g0_i`, `g_inf_i` in `[0.0, 2.5]`—realizing a 12 dB/oct noise gate and B+ sag drive with zero kink under Duhem `eta > 0`.

Lane polarity uses explicit `u16` `rect_sign_mask` (+1.0 / -1.0, never -0.0 in `f32`), emitting canonical +0.0 (`0x0000_0000`) whenever $\max(|r_{n-1}|, |r_n|) \le 0.0$.

| Evaluation Context | Pre-Activation Drive Formula | Parameter Domains & Scaling Semantics |
| :--- | :--- | :--- |
| SONA Core (`AlgADAA1(0.5)`) | `r_n = clamp(g_pre * d_adaa * u_tone + b_bias, -4096.0, +4096.0)` | `g_pre = (g0 + g_inf * d_pre * e) * rcp_ge1_nr2(1.0 + d_pre * e)` (`g0, g_inf` in `[0, 2.5]`); `d_adaa` in `[-32.0, 32.0]`, `b_bias` in `[-16.0, 16.0]` |
| Standalone Reference (`PolyADAA1`) | Pre-clamp endpoints `bar_r = clamp(r_n, -1.0, +1.0)` | Drive `(2 / 3) * d` pre-multiplied upstream; standalone 1× kernels evaluate piecewise polynomial ADAA1 directly |

### 1.2 Pure-`f32` Division-Free Primitives & Midpoint Factorization

`r[dsp.op-n.polyadaa.midpoint]`

To guarantee 0.000000 ULP parity across same-FMA-class CPU backends (and across all CPU backends on explicitly unfused `FP_CONTRACT OFF` paths) plus `<= 1.5` ULP/step agreement between hardware-FMA and non-FMA CPU or 32-bit GPU targets ([sona_architecture.md §1.5.3](../architecture/sona_architecture.md#153-single-library-per-architecture-runtime-simd-dispatch--0000000-ulp-parity)), all nonlinear and transcendental helpers execute in pure IEEE-754 `binary32` without `FDIV`, `FSQRT`, or external math libraries.

At every recursive state write across `TPT-SVF`, `ParaLRU`, `Op-P` filters, Kautz banks, and `Op-T` diffusers, recursive state subnormal flushing clears IEEE-754 subnormals (`(bits & 0x7F80_0000) == 0`) via `bits & 0x8000_0000`, while zero-equilibrium nonlinear outputs emit canonical +0.0 (`0x0000_0000`).

For division-free Newton-Raphson reciprocals (`rcp_pos_nr2`, 3-step, >= 1.0 fast path `rcp_ge1_nr2`, and batched 4-pair evaluation), inputs clamp at $d_{\text{safe}} = \max(d, 10^{-6})$ except when $d \ge 1.0$ holds structurally under `rcp_ge1_nr2`.

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
| `fast_log2_f32` (Log-Knee VCA) | Positive finite `f32` (`x > 0`) | Exponent `E` + normalized mantissa `m in [1, 2)` split (`log2(1 + f)` via degree-4 minimax polynomial) | 0.0079 oct (0.047 dB) max error across full dynamic range; zero `FDIV`, `FSQRT`, or `libm` |

Software `fp16` serialization extracts sign `(bits >> 16) & 0x8000`, exponent `(bits >> 23) & 0xFF`, and mantissa `bits & 0x007F_FFFF` strictly via `u32` bit manipulation without clamping infinity or NaN prior to classification (`new_exp = exp - 112`):

| IEEE-754 Class | Encode Condition (`f32` -> `u16`) | Encoded `u16` Bit Pattern | Decode (`fp16` -> `f32`) |
| :--- | :--- | :--- | :--- |
| 1. Infinity | `exp == 0xFF && mant == 0` | `sign \| 0x7C00` | `(sign << 16) \| 0x7F80_0000` |
| 2. Payload `NaN` | `exp == 0xFF && mant != 0` | `sign \| 0x7C00 \| max(mant >> 13, 1)` | `(sign << 16) \| 0x7F80_0000 \| (mant16 << 13)` |
| 3. Overflow | `new_exp >= 31` or RNE carry into exponent 31 | Saturate to `sign \| 0x7BFF` (±65,504.0) | Normal exponent 30 decode (±65,504.0) |
| 4. Zero / Tiny | `new_exp < -10` (magnitude below 2^-25) | `sign` (signed ±0.0) | `sign << 16` |
| 5. Subnormal | `-10 <= new_exp <= 0` | Shift `0x0080_0000 \| mant` right by `14 - new_exp` with ties-to-even | Normalize leading `mant16` bit into `binary32` |
| 6. Normal | `1 <= new_exp <= 30` | `(new_exp << 10) \| rounded(mant >> 13)`; overflow saturates to `sign \| 0x7BFF` | `(sign << 16) \| ((exp16 + 112) << 23) \| (mant16 << 13)` |

Inside CoreQuad blocks, clamped `AlgADAA1(0.5)` evaluates $f(r) = r / \sqrt{1 + r^2}$ over clamped endpoints $r_p, r_n \in [-4096.0, +4096.0]$ using 3-step Newton-Raphson NR3 primitives (`rsqrt_pos_nr3` and `rcp_ge1_nr3`). Precomputation forms `q_k = 1 + r_k^2`, `inv_s_k = rsqrt_pos_nr3(q_k)`, `s_k = q_k * inv_s_k`, `t_k = r_k^2 * rcp_ge1_nr3(s_k + 1)`, and `inv_sum = rcp_ge1_nr3(s_n + s_p)`.

`AlgADAA1(0.5)` `A_even` natively spans [0.0, 1.0] directly without a runtime 4.0× scale factor, whereas legacy `PolyADAA1` `A_even` in [0.0, 0.25] uses 4.0× normalization to map onto [0.0, 1.0] (16× better `w_even` conditioning, max odd err <= 7.59e-8, max even err <= 1.79e-7 across [-1e9, +1e9]):

$$\mathcal{A}_{\text{odd}}^{\text{alg}}(r_p, r_n) = (r_n + r_p) \mathrm{invsum} \cdot \big(1.0 + 0.5 \mathrm{invs}_n \mathrm{invs}_p\big)$$
$$\mathcal{A}_{\text{even}}^{\text{alg}}(r_p, r_n) = \mathrm{clamp}\Big(\big(2(t_n + t_p) + (t_n + t_p)^2 + t_n t_p (1 + t_n + t_p) + r_n r_p\big) \mathrm{invsum} \mathrm{invs}_n \mathrm{invs}_p, 0.0, 1.0\Big)$$

For standalone 1× reference kernels, `PolyADAA1` evaluates four interior FMAs (five including rail-crossing) over pre-clamped endpoints $u_n, u_{n-1} \in [-1, 1]$ using midpoint $m_n = \frac{1}{2}(u_n + u_{n-1})$ and half-difference $d_n = \frac{1}{2}(u_n - u_{n-1})$:

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

### 1.3 Cancellation-Free `AlgADAA1` Backward & Saturated-Curve Rail Crossing

`r[dsp.op-n.polyadaa.rail-crossing]`

For `AlgADAA1(0.5)` backward sensitivity inside GPU adjoint scans, direct finite differences suffer 253× cancellation at `r = 100` and sign flips at `r = 1e4`. Evaluating strictly positive $P(r_p, r_n) = 1 + s_n s_p - r_n r_p$ for $r_n r_p \le 0$ or $1 + (1 + r_n^2 + r_p^2)/(s_n s_p + r_n r_p)$ for $r_n r_p > 0$ gives exact `deven` with relative error $\le 4.08 \times 10^{-7}$, while `dodd` branches on `prod = r_n * r_p`, `lo = min(|r_p|, |r_n|)`, and `hi = max(|r_p|, |r_n|)`:

1. When `prod >= 0 && lo > 2.0 && |r_n - r_p| <= 0.15 * hi`: 4-point linear Gauss-Legendre on $[0, 1]$ of $1.5 t (1 + (r_p + t(r_n - r_p))^2)^{-5/2}$.
2. When `prod >= 0 && lo > 2.0 && |r_n - r_p| > 0.15 * hi`: 5-point Rational-Möbius Gauss-Legendre using $r(u) = \frac{(lo+1)(hi+1)}{hi+1 - u(hi-lo)} - 1$ (zero `FDIV`, zero `exp`/`log`).
3. Otherwise (`prod <= 0 || lo <= 2.0`): exact algebraic form (<= 8.68e-6 max relative error across 578 exhaustive pairs in `[-4096, 4096]²`, zero sign flips).

For standalone 1× `PolyADAA1` reference kernels when consecutive inputs $a = \min(r_{n-1}, r_n) < b = \max(r_{n-1}, r_n)$ cross $\pm 1$, the span $b - a$ partitions into interior length $\ell_0 = \max(0, \min(b, 1) - \max(a, -1))$ and plateau lengths $\ell_- = \max(0, \min(b, -1) - a)$ and $\ell_+ = \max(0, b - \max(a, 1))$, evaluating $\mathcal{A}_{\text{odd}}, \mathcal{A}_{\text{even}}$ on $[\max(a, -1), \min(b, 1)]$:

$$A_{\text{odd,sat}} = \big(\ell_0 \mathcal{A}_{\text{odd}} + \ell_+ - \ell_-\big) \cdot y, \qquad A_{\text{even,sat}} = \big(\ell_0 \mathcal{A}_{\text{even}}\big) \cdot y$$

```text
odd_cross  = ((above - below).fma(inside, odd_int)) * inv  // inv = 2-step NR reciprocal of max(distance, 1.0e-6)
even_cross = (inside * even_int) * inv
```

Whenever $|r_n - r_{n-1}| \le 10^{-6}$ (`TINY_CROSSING_EPS = 1.0e-6`), the kernel returns the algebraic midpoint result directly to avoid zero-span division. Otherwise, the ADAA kernel computes $y \approx (b - a)^{-1}$ in pure 32-bit float via a 2-step Newton reciprocal of $\max(b - a, \varepsilon_{\text{span}})$ with floor $\varepsilon_{\text{span}} = 10^{-6}$ (`RAIL_CROSSING_MIN_SPAN`). Because $g_{\mathrm{odd}}'(\pm 1) = 0$, crossing differences remain $C^1$ smooth with bounded condition number in `binary32`, reserving 6-step `f64` Newton reciprocals solely for offline test oracles.

### 1.4 Amortized Fast-Path `AlgADAA1` & `PolyADAA1` + Wavefolder Execution

`r[dsp.op-n.polyadaa.stereo-and-rail-guard]`

Inside `CoreQuad` blocks, fused `AlgADAA1 + Wavefolder` evaluates `all_lanes_le_one` fast-path checks (`|r_{n-1}| <= 1.0` and `|r_n| <= 1.0`) directly across 4-lane and 8-lane stereo SIMD registers, dropping dead `_wf_even` evaluation because `wavefolder_mix` blends only the odd channel.

For standalone `PolyADAA1` kernels, whenever all lanes satisfy $|r_{n-1}| \le 1.0$ and $|r_n| \le 1.0$, the interior fast path evaluates branchlessly with zero reciprocal calls, otherwise evaluating batched 2-step Newton reciprocals across the 8-lane stereo vector or dual 4-lane vector pairs without register spills ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

### 1.5 Un-Delayed Linear Separation & 1× Passive B-Spline Blend

#### 1.5.1 Un-Delayed Linear Residual Path & `lin_mix`

`r[dsp.op-n.polyadaa.linear-residual]`

Inside CoreQuad blocks, bounded linear bypass weight $\mu_{\mathrm{lin}} \in [0, 1]$ (`lin_mix`) blends `A_odd_wf` with `g_pre * u_tone` before post-ADAA SE-Bus (`o_pre = (1 - mu_lin) * A_odd_wf + mu_lin * g_pre * u_tone`) while zeroing the even channel by `(1 - mu_lin)`. Under unified 2× oversampling (96 kHz), 0.5 2×-sample ADAA1 delay equals 0.25 1×-sample (-1.41 dB at 20 kHz), and `lin_mix` polarizes toward 0 or 1 lanes to avoid single-lane mixed-phase blends.

For standalone 1× reference kernels, the linear residual path evaluates $\alpha_c x_c[n]$ directly with zero ADAA delay, preserving unity passband gain within $10^{-12}$ dB up to 22 kHz at 48 kHz.

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

Inside every 4-block SONA core group, `wavefolder_mix` with $\mu_{\text{wf}} \in [0, 1]$ convexly blends **only the odd harmonic channel** via $\mathcal{A}_{\text{odd}} \leftarrow (1 - \mu_{\text{wf}})\mathcal{A}_{\text{odd}} + \mu_{\text{wf}} w_{\text{odd,ADAA}}$ (with lane sign polarity stored in explicit `u16` `rect_sign_mask` rather than bit 15 of `wavefolder_mix_fp16`) while passing `AlgADAA1`'s $\mathcal{A}_{\text{even}}$ unmodified, whereas standalone wavefolder kernels emit both $(A_{\text{odd},3}, A_{\text{even},3})$ using pre-clamped inputs $\bar u = \mathrm{clamp}(u, -1, 1)$:

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

At the Top-of-Frame Crossbar, continuous pre-core conditioning executes `Op-N[Pre]` v4 strictly in **`quantize -> ZOH -> post-DAC CoreQuad SVF`** order ahead of `CoreOversampler`, configured by a base wire descriptor (`Crossbar[180..192)` plus `s_q` / `kappa_h` / `log2_rho_zoh` controls) and a 4-byte-aligned `OpNPreHotState` with zero 64-bit runtime integers:

| Region / Offset | Bytes | Fields & Types | Clamped Domain, Zero-Init & Control-Bus Invariant |
| :--- | :---: | :--- | :--- |
| Crossbar Wire 180..192 | 12 | `gamma_bc: f32le`, `zoh_hold_depth: f32le`, `beta_q: u16le`, `kappa_h: u16le` | Base wire `[0.0, 1.0]`, `[0.0, 0.9999]` / `log2_rho_zoh`, `[0.0, 16.0]` `fp16` (0 = bypass), and hold sharpness `kappa_h` in `[0.0, 1.0]` `fp16`; slots 26..28 |
| Active-Hot State 0..16 | 16 | `held: f32[2]`, `phase_q32: u32`, `inc_q32: u32` | `OpNPreHotState` prefix: unclipped L/R ZOH latch states plus 32-bit `Q0.32` ZOH accumulator and increment |
| Active-Hot State 16..28+ | 12+ | `gamma_bc: f32`, `beta_q: f32`, `zoh_hold_depth_eff: f32`, `s_q`, `kappa_h` | Effective wet blend `[0.0, 1.0]`, hoisted log2 scale `[0.0, 16.0]`, ZOH hold depth `[0.0, 0.9999]`, quantizer sharpness `s_q`, and hold sharpness `kappa_h` |

Encoding `zoh_hold_depth = 1.0 - rho_zoh` in `[0.0, 0.9999]` and `log2_rho_zoh` enforces zero-init pass-through (`g_clk == 1.0`) while keeping clock scaling rate-invariant across 44.1–192 kHz. Positioned after Op-T addition and ahead of `CoreOversampler`, Op-N[Pre] v4 replaces the 1-harmonic Fourier quantizer with the **4-FMA $C^1$ Hermite deadzone staircase quantizer** (zero FDIV, faster than sine evaluation).

For sharpness $s_q \in [0, 1]$, deadzone width $w = \max(1 - s_q, 2^{-12})$, pre-multiplied inverse width $w_{\text{inv}} = \mathrm{rcp}(w)$, and step $\Delta_q = 2^{-\mathrm{clamp}(\beta_q, -16, 16)}$ over $x_{\text{clip}} = \mathrm{clamp}(x_c, -100.0, +100.0)$:

$$n = \mathrm{round}(x_{\text{clip}} D_q), \qquad r = x_{\text{clip}} D_q - n \in [-0.5, 0.5], \qquad t = \mathrm{clamp}\big((|r| - (0.5 - 0.5 w)) w_{\text{inv}}, 0, 1\big)$$
$$Q(x_c) = \Delta_q \Big(n + \mathrm{sgn}(r) \cdot 0.5 t^2(3 - 2t)\Big)$$

#### 2.3.2 Hard-Capable `Q0.32` Phase-Wrap ZOH & Quantize-First Latch

`r[dsp.op-n.bitcrusher.zoh-phase]`

Executing once per VM frame at $f_{s,\text{vm}}$ ahead of `CoreOversampler`, `Op-N[Pre]` v4 applies `Q(x_c)` first (`q_c = clamp(x_c, -100.0, 100.0)` when `beta_q <= 1e-6`, else 4-FMA $C^1$ Hermite staircase), blends wet quantized signal `q_wet = x_c + gamma_bc * (q_c - x_c)`, and feeds the **decoupled hard-capable `Q0.32` phase-wrap ZOH** using strictly 32-bit runtime integers:

- **Log-Frequency Clock:** `rho_eff = exp2_clamped16(log2_rho_zoh) * rho_vm` maps across 44.1–192 kHz into `inc_q32 = (rho_eff >= 1.0 - 1.0e-7 || z_h_eff <= 0.0) ? 0 : max(1, round(rho_eff * 2^32))`.
- **Phase-Wrap Hold Latch:** Wrapping `phase_q32` sets `wrap = (new_phase < phase_q32) ? 1.0 : 0.0` alongside soft raised-cosine edge `w(phi) = 0.25 * (1 + cos(2*pi*phi))^2`.
- **Independent Hold Sharpness:** Blending `kappa_h` in `[0, 1]` yields `g_clk = (inc_q32 == 0) ? 1.0 : ((1 - z_h_eff) + z_h_eff * ((1 - kappa_h) * w(phi) + kappa_h * wrap))` (0-ULP bypass when `gamma_bc <= 0.0 && z_h_eff <= 0.0`).

$$g_{\mathrm{clk}}[n] = (1 - z_{h,\mathrm{eff}}) + z_{h,\mathrm{eff}} \big((1 - \kappa_h) w(\phi[n]) + \kappa_h \cdot \mathbf{1}_{\mathrm{wrap}}[n]\big), \qquad h_c[n] = (1 - g_{\mathrm{clk}}[n]) h_c[n-1] + g_{\mathrm{clk}}[n] q_{\mathrm{wet},c}[n]$$

When `inc_q32 == 0`, `g_clk == 1.0` tracks every sample cleanly with exact Zero-DC (`0.0 -> 0.0`) and unconditional BIBO contractivity (`0 <= 1 - g_clk < 1`). Standalone fixed-Hertz exact-rate tracking retains $F = 2^{32}$ units per Hz with $D = \mathrm{round}_{\text{even}}(F f_s)$ and $I = \mathrm{round}_{\text{even}}(F f_{\text{target}})$.

#### 2.3.3 Division-Free `Q0.32` & Log-Frequency Effective-Rate Phase Transfer

`r[dsp.op-n.bitcrusher.rate-transfer]`

Across host sample rates from 44.1 kHz through 192 kHz in both Live and Master render modes, Op-N[Pre] always executes at `f_{s,vm} == f_{s,host}` ahead of `CoreOversampler` upsampling. Normalized 32-bit `phase_q32` transfers unchanged across sample rates, while `log2_rho_zoh` maps via `exp2_clamped16(log2_rho_zoh) * rho_vm` directly to `inc_q32` so ZOH cutoff frequencies remain invariant across 44.1–192 kHz with zero 64-bit integers in the audio callback.

For standalone fixed-Hertz exact-rate tracking, precomputing `u64` reciprocal $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$ on the cold path lets sample-rate transitions form $N = P_{\text{old}} D_{\text{new}} < 2^{102}$ in 128-bit unsigned integer arithmetic, evaluate quotient $q_0 = \lfloor N R / 2^{104} \rfloor$ via bit shifts, refine remainder $r_0 = N - q_0 D_{\text{old}}$, round ties-to-even, and clamp to $D_{\text{new}} - 1$ so imminent carries stay imminent:

$$P_{\text{new}} = \min\left(\mathrm{round}_{\text{even}}\left(\frac{P_{\text{old}}D_{\text{new}}}{D_{\text{old}}}\right), D_{\text{new}}-1\right), \qquad q_0 = (h \ll 24) \mathbin{\mathrm{or}} \big((m \mathbin{\mathrm{and}} M) \gg 40\big)$$

Here $N = N_{\text{hi}}2^{64} + N_{\text{lo}}$, $a = N_{\text{lo}}R$, $c = N_{\text{hi}}R$, $M = 2^{64} - 1$, $m = (a \gg 64) + (c \mathbin{\mathrm{and}} M)$, and $h = (c \gg 64) + (m \gg 64)$, keeping frequency error $|f_s(I/D) - f_{\text{target}}| \le 2^{-32}$ Hz.

#### 2.3.4 Exact $C^1$ Hermite & Phase-Wrap ZOH Training Gradients

`r[dsp.op-n.bitcrusher.surrogate]`

Because the 4-FMA $C^1$ Hermite deadzone quantizer is $C^1$ differentiable in $(x, \beta_q, s_q)$ and the phase-wrap ZOH is a 1-D scalar affine recurrence with $0 \le 1 - g_{\text{clk}}[n] < 1$, WebGPU training evaluates exact forward and reverse adjoint passes using the shared 1-D affine `LinearScanElem` scan across $(x, z_h, \kappa_h, \log_2\rho_{\text{zoh}}, \beta_q, s_q, \gamma_{\text{bc}})$ with zero straight-through estimators (`STE`), achieving -30.6..-55.3 dB at smooth `(0.98, 1.0)` and -325+ dB at exact `(1.0, 1.0)`.

---

## 3. Exact LTI Basis Commutativity Theorem (Theorem 1)

### 3.1 Formal Theorem Statement & Proof

`r[dsp.op-n.commuted.theorem1]`

SONA v4.0 applies Theorem 1 to fold `LinearReadout` (`W_ro,0` + additive `W_ro_head` deltas) directly inside SIMD registers across the 4-block core with zero intermediate scratch gather buffers.

**Theorem 1 (Exact LTI Basis Commutativity).** Let $\mathbf{u}[n] \in \mathbb{R}^P$ be bounded per-channel block activations satisfying $\sup_{n, p} |u_p[n]| \le U_{\max} < \infty$, let $\lbrace \mathcal{L}_k \rbrace_{k=0}^{K-1}$ be causal, stable LTI filters with impulse responses $h_k \in \ell_1(\mathbb{Z})$, and let $\lbrace \mathbf{W}_k \rbrace_{k=0}^{K-1} \subset \mathbb{R}^{P \times C}$ be constant projection matrices. Then for every output channel $c \in [0, C-1]$ and sample $n \in \mathbb{Z}$:

$$\sum_{k=0}^{K-1} \Big(\mathcal{L}_k\big(\mathbf{u}\big)[n]\Big) \mathbf{W}_k \equiv \sum_{k=0}^{K-1} \mathcal{L}_k\Big(\mathbf{u} \mathbf{W}_k\Big)[n]$$

*Proof.* Absolute convergence follows from $h_k \in \ell_1(\mathbb{Z})$ and boundedness of $\mathbf{u}$, allowing summation interchange by Fubini's theorem so $\mathbf{u}\mathbf{W}_k$ filters directly across $C$ SIMD lanes. $\blacksquare$

---

### 3.2 Causality Audit & Elimination of Acausal Lookahead

`r[dsp.op-n.commuted.causality]`

All Top-of-Frame Crossbar Op-T reads (`tau_eff >= 42.0` so `tau_ring_cb = tau_eff - D_os >= 5.0`), self-feedback `TransportQuad` reads (`tau_min = 5.0`), standalone rings (`tau >= 4.0`), TPT-SVF/ParaLRU/AlgADAA1 updates, the 2-mode `CoreOversampler` (Live D_os 7/0; Master D_os 37/31/0 with `Box2` composite quadrature + 2× midpoint `lin_mix`), and ModalConvolver v4 interfaces are strictly causal with $d \ge 0$, maintaining sample-accurate phase alignment across Live and Master paths.

---

### 3.3 Commutator Perturbation Under 3 kHz Control-Rate Modulation

`r[dsp.op-n.commuted.project-then-filter]`

Under 3 kHz control modulation of core filters, drive nonlinearities, routing gates, and readout deltas, `ControlBus` v4 uses 4-frame aligned stride $K_{\text{ctrl}}(f_{s,\text{vm}}) = 4 \cdot \max(2, \mathrm{round}(f_{s,\text{vm}} / 12{,}000.0))$ (`K_ctrl(44.1 kHz) = 16`, `gcd(16, 4) = 4`, eliminating 196 Hz beat jitter and scalar peeling) and micro-ramps `[stage_targets(32), z_stage(8), z_head(8), g_edge(6), op_t_q(16)]` once per 4-frame micro-chunk (`alpha_chunk = 4 / K_ctrl = 0.25`), while recursive slews (Kautz `prev_oct` and Op-T `current_psi`) update strictly when `ctrl_sub_counter == 0`.

Hot `CoreQuad`, crossbar, `Op-N[Pre]`, and `Op-T` registers latch strictly at 4-frame chunk boundaries (`(sample_tick & 3) == 0 || ctrl_sub_counter == 0`), guaranteeing 0-ULP equivalence `process(1) * N == process(4) * (N/4) == process(16) * (N/16)` across every sample rate including 44.1 kHz (`K_ctrl = 16`) with zero tile rewrites:

$$\Delta_{\text{comm}}[n] \le \frac{B_u \nu_{\text{ctrl}}}{f_s} \cdot \frac{r_k}{1 - r_k}$$

---

### 3.4 Boundary Conditions, Resting DC Equilibrium & Pop-Free Startup

`r[dsp.op-n.commuted.dc-equilibrium]`

With decoupled 1-D `ParaLRU` recurrence using the $C^2$ quartic spline `soft_rect` satisfying $\phi_{\text{rect}}(0) = +0.0$, analytical zero-DC resting bias centering at `StarCenter` where $u_{\text{eq}} = +0.0$, `AlgADAA1(0.5)` zero equilibrium satisfying $\mathcal{A}_{\mathrm{odd}}^{\mathrm{alg}}(+0.0, +0.0) = \mathcal{A}_{\mathrm{even}}^{\mathrm{alg}}(+0.0, +0.0) = +0.0$, and zero-initialized integrators, all recursive state settles at exact 0.000000 DC equilibrium (`h_eq == +0.0`, `b_head == 0.0`):

1. Core `ParaLRU` lanes initialize at $v_{\text{init}} = +0.0$ across 16, 24, 32, or 40 lanes per channel from Lite through Ultra.
2. `TPT-SVF` and AC coupling filters initialize states $s_1 = s_2 = +0.0$ at idle equilibrium.

---

## 4. 4-Block Matryoshka `CoreQuad` Stack (`TPT-SVF`, Selective-Duhem `ParaLRU`, `PolyADAA1` & `GLU`)

`r[dsp.op-n.core.matryoshka-blocks]`

SONA v4.0 models nonlinear preamps, resonant diode filters, magnetic hysteresis, stereo-linked bus compression, crossovers, and sweeps via a rebalanced 4-block Matryoshka state-space core sorted across 10 logical 4-lane CoreQuads `[(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2)]` and 37 causal lower-triangular inter-block mixing tiles (`MAX_OUTGOING_PER_QUAD = 7`).

Each CoreQuad chains M/S input coupling (`in_mid_weight_fp16`, `in_side_weight_fp16`), 3-tap folded TPT-SVF (Predictor-Corrector passive PHS on Block 0 and linear ZDF TPT-SVF on Blocks 1..3), and smooth Selective-Duhem stereo-linked ParaLRU with C² quartic spline `soft_rect`.

Nonlinear drive applies `g_pre(v²)` and clamped AlgADAA1(0.5) (`MAX_ALG_R = 4096.0`, NR3) with `lin_mix` (evaluated at exact 2× midpoint `0.5 * (x_{2n} + x_{2n+1})` inside Master 4× CoreQuad blocks alongside `Box2` 2-point sub-sample averaging on 4×->2× writes), C² wavefolder (`rect_sign_mask`), post-ADAA rank-1 SE-Bus, bounded GLU, and micro-ramped `out_gain`:

| Tier Profile | `CoreQuad`s & Lanes | Inter-Block Tiles | Core Rate | `TPT-SVF` Quads / Modes | `ModalConvolver` (FIR / `ErSparse` / Kautz) | `TransportQuad`s (`Lines`) | Hot L1D Working Set |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lite `[1, 1, 1, 1]` | 4 quads (16 lanes) | 5 | Live 2× / Master 4× | 4 quads (16 modes) | 32 taps / 8 `ErSparse` / 8 pairs | 1 quad (4 lines, 64-fr inline) | 4,432 B (`14× = 60.59 KiB <= 64 KiB`) |
| Full `[2, 2, 1, 1]` | 6 quads (24 lanes) | 13 | Live 2× / Master 4× | 6 quads (24 modes) | 48 taps / 16 `ErSparse` / 12 pairs | 2 quads (8 lines) | <= 8,192 B (`8× <= 64 KiB`) |
| Pro `[2, 2, 2, 2]` | 8 quads (32 lanes) | 24 | Live 2× / Master 4× | 8 quads (32 modes) | 96 taps / 16 `ErSparse` / 24 pairs | 3 quads (12 lines) | <= 16,384 B (`4× <= 64 KiB`) |
| Ultra `[3, 3, 2, 2]` | 10 quads (40 lanes) | 37 | Live 2× / Master 4× | 10 quads (40 modes) | 128 taps / 16 `ErSparse` / 32 pairs | 4 quads (16 lines) | <= 32,768 B (`2× <= 64 KiB`) |

Replacing the 84-op square-root rectifier (6 `NR3` chains/quad) with a **C² quartic spline `soft_rect`** (12–14 NEON ops, zero `rsqrt` or `rcp`, -70c/quad modeled CPU, zero negative leak, $C^2$ continuous curvature for `L-SSBroyden`) sets $t = \mathrm{clamp}(z / (2\epsilon), 0, 1)$ with `SOFT_RECT_EPS` $\varepsilon = 10^{-3}$:

$$\phi_{\text{rect}}(z) = \epsilon t^3(2 - t) + \max(z - 2\epsilon, 0), \qquad \phi_{\text{rect}}'(z) = t^2(3 - 2t) \in [0, 1]$$

Every `ParaLRU` lane clamps $\alpha_{\text{rel}} \in [0.0, r_{\max}]$ (`MAX_PARALRU_R_F32 = 0x3F7F_FFAC`), $\alpha_{\text{atk}} \in [0.0, 1.0]$ in continuous-time log coordinates where ZOH scaling is exact scalar $\lambda \mapsto \rho\lambda$, $\kappa \in [0.0, 1.0]$, stereo link $\lambda \in [0.0, 1.0]$, Duhem gate $\gamma_{\text{duhem}} \in [0.0, 64.0]$, and signed blend $\eta \in [0.0, 1.0]$. Replacing hard-clamped `rho` with the **smooth non-saturating `ParaLRU` ODE** (zero gradient collapse):

$$\rho_i[n] = 1.0 - \alpha_{\text{atk},i}\cdot\mathrm{rcp-ge1-nr2}\big(1.0 + \gamma_{\text{duhem},i} \lvert 2 g_i \cdot \mathrm{BP}_i[n] \rvert\big)$$
$$d_i[n] = \alpha_{\text{rel},i} v_i[n-1], \qquad z_i[n] = s_i u_{\text{tone},i}[n] - \kappa_i (1 - \eta_i) d_i[n]$$
$$\bar{z}_i[n] = \mathrm{clamp}\big(z_i[n], -1.0, 1.0\big), \qquad \psi_1\big(z_i[n]\big) = \bar{z}_i[n]\left(1.0 - \frac{1}{3}\bar{z}_i[n]^2\right)$$
$$\phi_{\eta,i}\big(z_i[n]\big) = (1 - \eta_i)\phi_{\text{rect}}\big(z_i[n]\big) + \eta_i\psi_1\big(z_i[n]\big)$$
$$v_i[n] = \mathrm{flush}\Big(\big(1 - \eta_i \rho_i[n]\big) d_i[n] + \rho_i[n] \phi_{\eta,i}\big(z_i[n]\big)\Big)$$
$$J_i[n] = \frac{\partial v_i[n]}{\partial v_i[n-1]} = \alpha_{\text{rel},i}\Big((1 - \eta_i \rho_i[n]) - \rho_i[n] \kappa_i (1 - \eta_i)\phi_{\eta,i}'(z_i[n])\Big) \in \big[(1 - \rho_i[n])\alpha_{\text{rel},i}, \alpha_{\text{rel},i}\big] \subset [0, 1)$$

Because $J_i[n] \in [(1 - \rho_i[n])\alpha_{\text{rel},i}, \alpha_{\text{rel},i}] \subset [0, 1)$, Selective-Duhem `ParaLRU` is unconditionally contractive without post-hoc ISS clamping. At $\gamma_{\text{duhem}} = 0, \eta = 0$, it reduces bit-for-bit to the scalar diode-RC rectifier, while $\eta = 1, \alpha_{\text{atk}} = 1, \gamma_{\text{duhem}} > 0$ holds coercive magnetic remanence across extrema as $\Delta s_2 \to 0$.

WebGPU training executes an **8-thread L/R × 4-lane `ParaLRU` split** in SRAM (<= 15.0 KiB, eliminating 97%–99.6% thread starvation) with an exact $O(\log_2 N)$ associative scan fast path whenever $\kappa(1 - \eta) == 0$ (feedforward and pure-Duhem lanes).

Audio-rate stereo linking $v_{\mathrm{eff},c,i} = v_{c,i} + \lambda_i(v_{\mathrm{mid},i} - v_{c,i})$ feeds $C^\infty$ bounded pre-ADAA gain $g_{\mathrm{pre}}(v_{\mathrm{eff}}^2)$ and clamped AlgADAA1 (`r_curr = clamp(g_pre * d_adaa_eff * tone_in + b_lru_eff, -4096.0, +4096.0)`), where $s_{\mathrm{rect}}$ denotes explicit `u16` `rect_sign_mask` (replacing bit-15 XOR of `wavefolder_mix_fp16` so `f32` never carries -0.0):

$$b_{\mathrm{lru,eff}} = \mathrm{clamp}\big(b_{\mathrm{lru,base}} + s_{\mathrm{rect}} \cdot \theta_{\mathrm{lru}}, -16.0, +16.0\big)$$

Mechanism A2 watermarking flips `in_mid_weight`, `in_side_weight`, `b_lru_base`, `rect_sign_mask`, and `W_ro_head_odd` cleanly by lane sign $s_l \in \lbrace -1, +1 \rbrace$ while keeping $|2 g \cdot \mathrm{BP}_l|$, $\rho_l$, $g_{\mathrm{pre}}(v_{\mathrm{eff}}^2)$, $p_{\mathrm{bus}}$, and Block 0 PHS damping $k_{\mathrm{nl}} \min(v_{1,\mathrm{lin}}^2, 64.0)$ strictly even-symmetric (0-ULP A2 invariant). For backward gradients, `adj_rho -> adj_BP` propagates smoothly through $\partial\rho_i / \partial |\mathrm{BP}_i|$ with zero hard clipping mask `1_{rho in (0, 1)}`.

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

### 4.1 3-Tap Folded `TPT-SVF` Tone, Predictor-Corrector `PHS` & 2-Block `LR4`

`r[dsp.op-n.core.tpt-tone-basis]`

Each 2×2 ZDF TPT-SVF mode clamps $g \in [10^{-5}, 64.0]$, optimized in $\log(f_c / f_s)$ and $\log k$, with cutoff relation $f_c = (f_s/\pi)\arctan(g)$ and effective linear damping $k \in [0.0, 8.0]$. To eliminate 80 redundant zero Hessian eigenvalues at Ultra while saving 4 vector ops/quad, Blocks 1..3 collapse the 5-tap basis into **3 orthogonal folded coefficients (`c_lp, c_bp, c_u`)**, whereas Block 0 retains `(c_lp, c_bp, c_kbp, c_u)`.

On Block 0 `CoreQuad`s `{0, 4, 8}`, nonlinear diode-bandpass damping $k_{\mathrm{nl}} \in [0.0, 16.0]$ executes as a **1-step Predictor-Corrector Passive PHS**: Pass A steps the linear TPT-SVF to obtain $v_{1,\mathrm{lin}}[n]$, and Pass B runs passive LTV TPT-SVF with $E[n] = \min(v_{1,\mathrm{lin}}[n]^2, 64.0)$ and $k_{\mathrm{eff}}[n] = k + k_{\mathrm{nl}} E[n]$, guaranteeing $\Vert \mathbf{M}_n \Vert_2 \le 1$ and $O(\log_2 N)$ 256-thread GPU parallelism.

Because $H_{\text{LR4-Mid}}(s) = [H_{\text{HP2}}(s/\omega_{\text{lo}})\cdot H_{\text{LP2}}(s/\omega_{\text{hi}})]^2$ factors into two distinct complex-conjugate pole pairs, partial-fraction decomposition realizes $H_{\text{HP2}} \cdot H_{\text{LP2}}$ as a 2-lane parallel TPT-SVF sum in Block 0 squared in Block 1 (-285.69 dB exactness), leaving Block 2 for drive and Block 3 for post-tilt.

On Blocks 1..3 where $k_{\mathrm{nl}} = 0$, GPU forward scans execute constant-M register-squaring stereo scans (`M^(2^k)` in 4 registers, scanning both L and R in one pass inside 4,096 B <= 6,144 B SRAM), while `pass0` binary-tree reductions save 216 barriers per tile (596 -> ~220) inside <= 15.0 KiB SRAM.

---

### 4.2 Stereo M/S Input, 6-Edge Routing, Post-ADAA `SE-Bus`, `GLU` & Split `W_ro_head`

`r[dsp.op-n.core.interblock-tiles]`

Inter-lane mixing across the 4-block `CoreQuad` stack visits active quads strictly in block-major `SONA_BLOCK_QUAD_SLOTS` order (`Block 0 -> 1 -> 2 -> 3`, matching `[0, 4, 8, 1, 5, 9, 2, 6, 3, 7]` at Ultra) with strict tier prefix nesting (5 / 13 / 24 / 37 tiles, `MAX_OUTGOING_PER_QUAD = 7`) and 6 scalar routing gates `g_edge[0..6]` without rewriting tiles at 12 kHz:

| Block (`blk`) | Active Count (`B_{blk}` at Lite / Full / Pro / Ultra) | `SONA_BLOCK_QUAD_SLOTS[blk]` (`-` unused) |
| :--- | :---: | :--- |
| Block 0 | 1 / 2 / 2 / 3 | `[0, 4, 8, -]` |
| Block 1 | 1 / 2 / 2 / 3 | `[1, 5, 9, -]` |
| Block 2 | 1 / 1 / 2 / 2 | `[2, 6, -, -]` |
| Block 3 | 1 / 1 / 2 / 2 | `[3, 7, -, -]` |

```text
u_mid = 0.5 * (u_L + u_R);  u_side = 0.5 * (u_L - u_R)
u_acc_pair[0..6] = 0.0  // 6 causal block-pair accumulators: (0->1, 0->2, 0->3, 1->2, 1->3, 2->3)
for prev_blk = 0..blk, s = 0..B_{prev_blk}, src_g = SONA_BLOCK_QUAD_SLOTS[prev_blk][s]:
  if edge_in_tier_prefix(src_g, dst_g, T_CAP): // Lite (T_CAP=5): adj + (0,2),(1,3); Full..Ultra: 13, 24, 37
    u_acc_pair[pair_idx(prev_blk, blk)] += tile_odd(w_odd, o_tilde[src_g]) + tile_even(w_even_fp16, e_tilde[src_g]) + tile_lru(w_lru_fp16, v_eff[src_g])
u_acc = sum_{prev_blk < blk} g_edge[pair_idx(prev_blk, blk)] * u_acc_pair[pair_idx(prev_blk, blk)]
u_{g,L/R}[n] = (u_acc_L/R + in_mid_weight * u_mid ± in_side_weight * u_side) + ((op_p_to_core * T1_feat) + (phasor_to_core * car_L/R))

// Bounded Pre-Gain -> AlgADAA1 -> Post-ADAA Rank-1 SE-Bus -> Bounded GLU -> out_gain:
A_odd_wf  = (1 - mu_wf) * A_odd + mu_wf * wf_odd
p_bus     = sum_{j=0..3} w_sq[j] * min(|v_eff[j]|, 1e6)
g_se      = rcp_ge1_nr2(1.0 + d_bus * p_bus)                 // (0, 1], +0 B WebGPU SRAM via Slot 8
o_pre     = (1 - mu_lin) * A_odd_wf + mu_lin * g_pre * u_tone
o_tilde   = g_out * g_se * o_pre
e_tilde   = canonical_zero_pos(g_out * (1 - mu_lin) * (g_se * A_even + mu_glu * (g_se * A_odd_wf) * rotl1(g_se * A_odd_wf)))
```

For each destination CoreQuad `g = SONA_BLOCK_QUAD_SLOTS[blk][d]`, the inter-block loop evaluates edges inside tier prefix `0..T_CAP` (`T_CAP` in `{5, 13, 24, 37}`, `MAX_OUTGOING_PER_QUAD = 7`) and scales the 6 block-pair tile sums directly by micro-ramped `g_edge[0..6]` without ever rewriting tile matrices at 12 kHz.

Inside each quad, post-ADAA rank-1 SE-Bus computes `g_se[i] = rcp_ge1_nr2(1.0 + d_bus[i] * p_bus)` from `v_eff` (+0 B WebGPU SRAM via `Slot 8` pullback), followed by bounded GLU evaluated strictly on `g_se * A_odd_wf` (never unbounded `o_pre`) and micro-ramped `out_gain` in `[-4.0, +4.0]`.

Every inter-block tile uses column-major `[src_lane][dst_lane]` storage decoded once per 8 subsamples in Quad-Major order (8× fewer `fp16` decodes). Binary LinearReadout (`W_ro,0` plus additive split `W_ro_head` `odd` and `even_lru` deltas) traverses quads in block-major sequence `[0, 4, 8, 1, 5, 9, 2, 6, 3, 7]` and accumulates `(pre_conv, direct_s2)` ahead of ModalConvolver and Op-T.

Unlike standalone 1× `Op-N` reference kernels (§1.5.2), the 4-block `CoreQuad` stack omits the (0.875, 0.125) passive B-spline blend and `R_ac` AC coupling, passing `(o_tilde, e_tilde, v_eff)` directly into downstream tiles and `LinearReadout` weights.

---

### 4.3 Quad-Major 8-Subsample Register-Resident Execution & Wide-ISA SIMD

`r[dsp.op-n.core.stereo-simd]`

Inside each 4-frame / 8-subsample micro-chunk, SONA executes Quad-Major in topological order: load `state[q]` once, decode `fp16` `CoreQuadHot`, `ReadoutHot`, and outgoing `TileHot` coefficients **once per 8 subsamples** (8× fewer `fp16` tile decodes at Ultra: 2,240 -> 280, zero cold spills).

Each active quad steps all 8 subsamples in 256-bit 8-lane stereo SIMD registers or interleaved 4-lane 128-bit vector pairs (with 3-op subnormal scrubbing and target-ISA block-entry dispatch), and stores `state[q]` once. Fixed 4-lane pair reduction trees guarantee 0-ULP parity across same-FMA-class targets (`<= 1.5` ULP/step across FMA vs non-FMA targets per [sona_architecture.md §1.5.3](../architecture/sona_architecture.md#153-single-library-per-architecture-runtime-simd-dispatch--0000000-ulp-parity), [ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0024](../../adr/0024-sona-v4-kernel-hb-fir-perimeter-and-control-bus-gra.md)).

---

### 4.4 4-Frame Gauge-Invariant `z_stage` Sensitivity & Split `W_ro_head`

`r[dsp.op-n.core.control-sensitivity]`

`SO(4)` buyer gauge `Q_buyer` rotates only `z_head[0..4]` along with watermarked main and interaction blocks, while `z_stage[0..8]`, `z_head[0..8]`, and the 6 edge gates `g_edge[0..6]` micro-ramp once per 4-frame micro-chunk (`alpha_chunk = 0.25`).

The readout block stores static base matrix `W_ro,0` plus additive split `W_ro_head` (`odd` and `even_lru` deltas) so Mechanism A2 flips `W_ro_head_odd` cleanly and StarCenter-muted lanes activate without bias drift (~1e-6 `f32` invariance; [parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws), [dnam_binary_format.md](../containers/dnam_binary_format.md)).

---

## 5. `Op-N[Ctrl]`: 3 kHz `ControlBus` v4 (`GRA`) Algebraic Polynomial & Sufficient-Statistic Engine

### 5.1 Incremental Sufficient-Statistic Update

`r[dsp.op-n.ctrl.incremental-update]`

When control $p \in \mathcal{I}_{\text{moved}}$ moves from $u_{p,\text{old}}$ to $u_{p,\text{new}}$, Op-N[Ctrl] updates `ExactIntMainAccumulator` on SO(4)-isolated `z_head[0..4]` (preserving 0-ULP StarCenter and buyer watermark isolation) and builds unrotated `CP1` basis `h_0 = W_in * xi` in $\mathbb{R}^D$, with $D = 8$ Lite or $16$ Full–Ultra, inside reclaimed Arena 0 space per [parametric_control_bus.md §2.3](parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling):

$$\Delta\mathbf{z}_{\text{main}} = \mathbf{W}_{\text{main}, p} \cdot \big(\boldsymbol{\psi}_p(u_{p, \text{new}}) - \boldsymbol{\psi}_p(u_{p, \text{old}})\big)$$

### 5.2 Node-Anchored Control Basis & Zero-Anchored `SwiGLU + Padé-[1/1]`

`r[dsp.op-n.ctrl.lobatto-basis]`

Basis evaluation interpolates persisted observed anchor nodes and switch contributor slices (`xi == +0.0` at StarCenter), feeding Zero-Anchored SwiGLU, with $H = 8$ Lite or $16$ Full–Ultra, and a Padé-[1/1] rational head (`recip_nr3_pos_vec4`) followed by anchored 1.5-Entmax residual prototypes (`M = 8` Lite–Pro or 16 Ultra, `m = 0` implicit StarCenter anchor, `O(P + M)` global BIBO) per [parametric_control_bus.md §2.2](parametric_control_bus.md#22-cardinality-bounded-node-anchored-newtonlobatto-basis) and [§3.1.2](parametric_control_bus.md#312-controlbus-v4-gated-rational-attention-gra-pipeline).

### 5.3 Skewed Quadrature LFO Waveshaping (Second-Harmonic Chebyshev)

`r[dsp.op-n.ctrl.chebyshev-lfo]`

Quadrature LFO states evaluate `(c_j, s_j)` per oscillator with asymmetric inertia slew and live `TSYN` lock, applying learned second-degree Chebyshev harmonic skew `T_2(x) = 2x^2 - 1` weighted by `beta_j = lfo_skew_fp16[j]` in `[-1.0, 1.0]` into control vector `q2`, while higher-order Chebyshev terms $T_3(x)$ and $T_4(x)$ remain available in standalone control kernels.

### 5.4 Rank-4/8 FM Bilinear Interaction

`r[dsp.op-n.ctrl.fm-interaction]`

Pairwise FM interactions and SwiGLU bilinear couplings (`(v, +g)` / `(v, -g)` symmetric pairs reproducing `v * g` to 1.776e-15) combine scalar embeddings over `U_int^base` (128 B on SO4-isolated `z_head[0..4]`) and `W_g`/`W_v` within the unified ControlBus v4 (`GRA`) pipeline ([parametric_control_bus.md §3.1.2](parametric_control_bus.md#312-controlbus-v4-gated-rational-attention-gra-pipeline)).

### 5.5 Groupwise Trust-Region Clipping

`r[dsp.op-n.ctrl.trust-clip]`

Outward-checked trust-region clipping runs independently on `SO(4)`-isolated `z_head[0..4]` via `trust_clip_outward_4d`, while `GRA` hidden vector `h_1 = trust_clip(h_0 + W_o * a)` enforces $\Vert \mathbf{h}_1 \Vert_2 \le R_{\text{trust}}$ to certify global `O(P + M)` convex-hull BIBO bounds per [parametric_control_bus.md §1.2.1](parametric_control_bus.md#121-outward-checked-trust-region-clip) and [§3.1.3](parametric_control_bus.md#313-global-op--m-convex-hull-bibo-stability).
