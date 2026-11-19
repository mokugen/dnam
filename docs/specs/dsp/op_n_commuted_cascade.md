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

This specification defines **Op-N (Nonlinear & Parallel State-Space)** (shared `#![no_std]` `32-bit` kernels in `packages/kernels/` `dnam-kernels`, runtime implementation in [`packages/engine/src/operators/op_n.rs`](../../../packages/engine/README.md), and identification in [`packages/capture/src/`](../../../packages/capture/README.md)), retiring coupled `GRU` (`stage2_gru`) and normatively specifying the **`SONA` `v3.0` 4-Block Matryoshka Core** (`LiteSpec [1,1,1,1]`, `FullSpec [2,2,1,1]`, `ProSpec [3,2,2,1]`, `UltraSpec [4,3,2,1]` locked as compile-time `const` profiles at `Gate 0b`), **decoupled `1D` scalar `ParaLRU`** (exact in-SRAM `step_paralru` forward recurrence / $O(\log_2 N)$ `LinearScanElem` reverse-adjoint Blelloch scan across `256`-sample WebGPU tiles with exact cross-tile boundary propagation), **pure-`f32` divided-difference `PolyADAA1`** (`(m_n, d_n)` cubic odd + quartic even + `C²` wavefolder + batched `rcp_pos_quad_nr2` rail-crossing correction, `0` `FDIV`, analytical `0.000000` DC equilibrium), lower-triangular `4×4` inter-block spatial mixing, **`StereoVec8` + `2× SimdVec4` `0-ULP` parity**, ZOH bitcrusher (`Family 5`), and **`Op-N[Ctrl]`** across **Dynamic Neural Audio Modeler (DNAM)** ([`sona_architecture.md`](../architecture/sona_architecture.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

## 1. Pure-`f32` Divided-Difference `PolyADAA1` & Antialiased Activations

### 1.1 Pre-Activation Clamping & Interior/Rail Argument Bounding

`r[dsp.op-n.polyadaa.clamp]`

Inside every `SONA` core block, linear signal paths clamp only at the `±1e9` safety boundary (`REQ-7.1`). Inside `SONA` group execution (`step_sona_group_stereo8`), drive pre-modulation applies dynamic sag $g_{\text{sag}} = \max(0.05, 1.0 - d_{\text{sag}} \cdot v_{\text{lru}})$ with `sag_depth` quantized in `binary16` over `[-8.0, 8.0]`, forming raw pre-activation $r_n = u_{\text{tone}}[n] \cdot d_{\text{adaa}} \cdot g_{\text{sag}} + b_{\text{lru}} \cdot v_{\text{lru}}[n]$ (`adaa_drive` clamped to `[-32.0, 32.0]`, `lru_bias_drive` clamped to `[-16.0, 16.0]`; reference fixed bank uses $r_n = \frac{2}{3} d \cdot u_n^{\mathrm{raw}}$). Both `PolyADAA1` and the $C^2$ wavefolder retain the previous unclamped endpoint $r_{n-1}$ and **pre-clamp endpoints to $\bar{r}_n = \mathrm{clamp}(r_n, -1.0, +1.0)$ BEFORE polynomial evaluation** so infinite or out-of-range intermediates never yield `Inf - Inf = NaN`, preserving exact odd symmetry ($f(-x) = -f(x)$) and clearing `-0.0` at zero via explicit `0x0000_0000` (`+0.0`) output masking whenever $\max(|r_{n-1}|, |r_n|) \le 0.0$.

### 1.2 Pure-`f32` Division-Free Primitives, `PolyADAA1`, `rcp_pos_quad_nr2` & Midpoint Factorization

`r[dsp.op-n.polyadaa.midpoint]`

To guarantee **0-ULP bit-exact clean-room reproducibility across C, C++, Rust, Python, TypeScript, and `32-bit` WGSL** with **zero `FDIV`, zero `FSQRT`, zero `libm`, zero LUT gathers, zero GPR spills, and `wasm32`/`FTZ`-safe boundary behavior**:

1. **Subnormal State Flushing (`flush_recursive_state_f32`):** At every recursive state write (`TPT-SVF` $s_1, s_2$, `ParaLRU` $v$, `Op-P` all-pass/DC states, `Kautz` states, and `Op-T` diffusers), flush IEEE-754 subnormals (`|x| < 2^-126`, i.e., `(bits & 0x7F80_0000) == 0`) to zero (`bits & 0x8000_0000`, while nonlinear zero-equilibrium masks emit canonical `+0.0 = 0x0000_0000`).
2. **Chebyshev Minimax FMA Newton Reciprocal (`rcp_pos_nr2/3`, `rcp_ge1_nr2`, `rcp_pos_quad_nr2`):**
   For normalized `binary32` positive reciprocal $1 / d$, clamp $d_{\text{safe}} = \max(d, 10^{-6})$ (`rcp_ge1_nr2` omits the $10^{-6}$ floor when $d \ge 1.0$ holds structurally), subtract the `binary32` significand bits from compile-time Chebyshev minimax seed `IEEE754_F32_RCP_SEED = 0x7EF3_11C3` (`((2 * 127 - 1) << 23) + round((2*sqrt(6) - 4) * 2^23)`), form `y_0 = from_bits(0x7EF3_11C3 - to_bits(d_safe))`, and run $N_{\text{iter}} \in \lbrace 2, 3 \rbrace$ **cancellation-safe FMA Newton-Raphson iterations** (`y = fma(y, fma(-d_safe, y, 1.0), y)`, equivalent to $y + y(1 - d_{\text{safe}} y)$, avoiding catastrophic cancellation in $2 - d_{\text{safe}} y$):
   $$y_{k+1} = y_k + y_k (1 - d_{\text{safe}} y_k) = \mathrm{fma}\big(y_k, \mathrm{fma}(-d_{\text{safe}}, y_k, 1.0), y_k\big)$$
   `nr2` (`2` iterations) achieves relative error $< 3.5 \times 10^{-6}$; `nr3` (`3` iterations) achieves $< 1\text{ ULP}$ across normal `binary32` ranges.
3. **Two-Binade Minimax `rsqrt_pos_nr2/3`:** Clamp $d_{\text{safe}} = \max(d, 10^{-12})$, form $h = 0.5 d_{\text{safe}}$, compute seed `y_0 = from_bits(0x5F37_59DF - (to_bits(d_safe) >> 1))` (`IEEE754_F32_RSQRT_SEED = 0x5F37_59DF`, `0x5FE6_EB50_C7B5_37A9` for offline `f64`), and run `2` (`nr2`) or `3` (`nr3`) Halley/Newton steps:
   $$y_{k+1} = y_k \cdot \big(1.5 - h \cdot y_k^2\big)$$
4. **Degree-5 Minimax `2^x` (`exp2_clamped5` & `exp2_signed_oct_f32`):**
   - *Centered Round-Magic Kernel (`dnam-kernels::simd::exp2_clamped5`):* Clamp $x \in [-5.0, +5.0]$, extract nearest integer $k$ via IEEE-754 biased rounding magic `0x4B40_0000` ($1.5 \cdot 2^{23}$), compute fractional residual $f = x - k \in [-0.5, +0.5]$ and exact power-of-two factor `2^k = from_bits((to_bits(k_biased) + 127) << 23)`, and evaluate Horner degree-5 polynomial with `[1.0, LN_2 = 0.6931471824645996, EXP2_C2 = 0.24022109, EXP2_C3 = 0.05550411, EXP2_C4 = 0.009675892, EXP2_C5 = 0.0013333558]`.
   - *Floor `[0, 1]` Octave Kernel (`stage1_svf::exp2_signed_oct_f32` & Normal Extension `[-126, +127]`):* Using integer floor $n = \lfloor x \rfloor$ on $[-5.0, +5.0]$ (extending to $x \in [-126.0, +127.0]$ so $127 + n \in [1, 254]$ stays strictly normal across `binary32`), $f = \mathrm{clamp}(x - n, 0.0, 1.0)$ evaluates Hermite-matched `[C1 = 0.6931472, C2 = 0.2402265, C3 = 0.055712965, C4 = 0.009011465, C5 = 0.001901882]` multiplied by `from_bits(((127 + n) as u32) << 23)`.
5. **Software IEEE-754 `binary16` (`fp16`) Round-to-Nearest-Even Codec (`simd/fp16.rs`):**
   All `binary16` parameter encodes/decodes operate strictly via `u32` bit manipulation (**never** clamping `Inf`/`NaN` to finite `65504.0` prior to classification): extract `sign = (bits >> 16) & 0x8000`, `exp = ((bits >> 23) & 0xFF) as i32`, `mant = bits & 0x007F_FFFF`. Map `exp == 0xFF` with `mant == 0` to `sign | 0x7C00` (`±Inf`), and `mant != 0` to `sign | 0x7C00 | max(mant >> 13, 1)` (`NaN` payload preserved). Set `new_exp = exp - 112`. If `new_exp >= 31`, saturate finite overflow to `sign | 0x7BFF` (`±65504.0`). If `new_exp <= 0`, emit `sign` (`±0.0`) for `new_exp < -10`, else right-shift `0x0080_0000 | mant` by `shift = 14 - new_exp` with ties-to-even rounding (`round_bit = bit (shift - 1)`, `sticky`, odd tie-break). For normal values (`1 <= new_exp <= 30`), round `mant >> 13` at bit `12` with sticky mask `0x0FFF`, saturating mantissa-carry overflow at `exp = 31` to `sign | 0x7BFF`. Decoding (`fp16_u32_to_f32`) mirrors every IEEE-754 class (`±0.0`, normalized `fp16` subnormals, normals `(exp + 112) << 23`, `±Inf`, and payload-preserving `NaN`).
6. **Interior `PolyADAA1` Midpoint / Half-Difference Form (`6` FMAs):** For interior clamped endpoints $u_n, u_{n-1} \in [-1, 1]$ with midpoint $m_n = \frac{1}{2}(u_n + u_{n-1})$ and half-difference $d_n = \frac{1}{2}(u_n - u_{n-1})$:
   $$\mathcal{A}_{\text{odd}}[n] = m_n \Big(1.5 - 0.5\big(m_n^2 + d_n^2\big)\Big), \qquad \mathcal{A}_{\text{even}}[n] = \left(m_n^2 + \frac{1}{3}d_n^2\right) - \left(m_n^4 + 2 m_n^2 d_n^2 + \frac{1}{5}d_n^4\right)$$
   At $r_n = r_{n-1} = u \in [-1, 1]$ ($d_n = 0$), this reduces identically to the memoryless cubic/quartic curves $g_{\mathrm{odd}}(u) = 1.5u - 0.5u^3$ and $g_{\mathrm{even}}(u) = u^2 - u^4$ with exact zero slope at the rail boundaries ($g_{\mathrm{odd}}'(\pm 1) = 0$). (Offline Padé-`[5/4]` reference constants: $(n_1, n_2, d_1, d_2) = (61/600, 11/19200, 87/200, 47/3840)$ solved from $d_1 - n_1 = 1/3$, $d_2 - n_2 - d_1/3 = -2/15$, $P_5(R) = Q_4(R)$, and $(P_5/Q_4)'(R) = 0$ at $R = 4$, `ref[repo.code.constant-eval-and-derivations]`).

#### Saturated-Curve Rail Crossing (Pure-`f32` Batched `rcp_pos_quad_nr2`)

`r[dsp.op-n.polyadaa.rail-crossing]`

When consecutive raw inputs $a = \min(r_{n-1}, r_n) < b = \max(r_{n-1}, r_n)$ strictly cross $\pm 1$, partition the interval length $b - a$ into interior length $\ell_0 = \max(0, \min(b, 1) - \max(a, -1))$ and rail-plateau lengths $\ell_- = \max(0, \min(b, -1) - a)$, $\ell_+ = \max(0, b - \max(a, 1))$, evaluating $\mathcal{A}_{\text{odd}}, \mathcal{A}_{\text{even}}$ over the pre-clamped interior interval $[\max(a, -1), \min(b, 1)]$:
$$A_{\text{odd,sat}} = \big(\ell_0 \mathcal{A}_{\text{odd}} + \ell_+ - \ell_-\big) \cdot y, \qquad A_{\text{even,sat}} = \big(\ell_0 \mathcal{A}_{\text{even}}\big) \cdot y$$
When $|r_n - r_{n-1}| \le 10^{-6}$ (`TINY_CROSSING_EPS = 1.0e-6`), select the algebraic interior/plateau midpoint result directly without reciprocal division (`0/0` NaN immunity). Otherwise compute $y \approx (b - a)^{-1}$ directly in **pure `f32`** via `rcp_pos_nr2(max(b - a, 1.0e-6))` (`RAIL_CROSSING_MIN_SPAN = 1.0e-6` denominator floor in `dnam-kernels::simd::rcp_pos_nr2`, exposed via `eval_poly_adaa1_*` such as `eval_poly_adaa1_f32` / `eval_poly_adaa1_vec4` / `eval_poly_adaa1_stereo8`, without hardware division; unconditionally accurate in `binary32` because $g_{\mathrm{odd}}'(\pm 1) = 0$ keeps divided differences across $\pm 1$ $C^1$-smooth with bounded condition number), using 6-step `f64` Newton reciprocals solely as an offline verification oracle.

`r[dsp.op-n.polyadaa.stereo-and-rail-guard]` **Amortized Fast-Path `PolyADAA1` & Batched `rcp_pos_quad_nr2` Stereo Execution ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
1. **Interior `6`-FMA Fast Path (`|u| <= 1`):** Whenever all lanes of a `SimdVec4` / `StereoVec8` register satisfy $|r_{n-1}| \le 1.0$ and $|r_n| \le 1.0$, `PolyADAA1` evaluates strictly via `6` branch-free `SimdVec4` FMAs (`0` reciprocal evaluations).
2. **Batched `rcp_pos_quad_nr2` Rail-Crossing Path:** Whenever any lane in the stereo pair crosses $\pm 1$, batch the 4 independent segment reciprocal lanes (`[L_odd, L_wf, R_odd, R_wf]` / batched channel spans) via `SimdVec4::rcp_pos_quad_nr2` (`0` GPR transfers, `0` per-lane branches, identical execution in `dnam-kernels` CPU and WGSL shaders).

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

To ensure $C^2$ continuity at the $\pm 1$ rails (vanishing value, first derivative, and second derivative at $\bar u = \pm 1$), `Op-N` pairs `PolyADAA1` (`eval_poly_adaa1_*`) odd/even saturation with the cubed-envelope odd and even wavefolders (`eval_c2_wavefolder_adaa_*` — e.g., `eval_c2_wavefolder_adaa_f32`, `eval_c2_wavefolder_adaa_vec4`, `eval_c2_wavefolder_adaa_stereo8` in `dnam-kernels`, convex-blended into odd output via $\mathcal{A}_{\text{odd}} \leftarrow (1 - \mu_{\text{wf}})\mathcal{A}_{\text{odd}} + \mu_{\text{wf}} w_{\text{odd,ADAA}}$ for $\mu_{\text{wf}} \in [0, 1]$, plus reference offset $\hat w_{\text{odd},3}[n] = u_1 + A_{\text{odd},3} - m$ in `wavefolder_and_bitcrusher.rs`) using pre-clamped endpoints $\bar u = \mathrm{clamp}(u, -1, 1)$ prior to polynomial evaluation (`Inf - Inf = NaN` immunity):
$$w_{\text{odd},3}(u) = \bar u\big(1 - \bar u^2\big)^3, \qquad w_{\text{even},3}(u) = \bar u^2\big(1 - \bar u^2\big)^3$$

For consecutive pre-clamped inputs $\bar{u}_{n-1}, \bar{u}_n \in [-1, 1]$ with $m = (\bar{u}_{n-1} + \bar{u}_n)/2$, $d = (\bar{u}_n - \bar{u}_{n-1})/2$, $s = m^2 + d^2$, and $\omega = 1 - s$, evaluate the exact division-free interior FMA factorization (returning `+0.0 = 0x0000_0000` when $\max(|u_{n-1}|, |u_n|) \le 0.0$):
$$A_{\text{odd},3} = m \cdot \omega \cdot \big(\omega^2 + 4 m^2 d^2\big), \qquad A_{\text{even},3} = (Q_2 - Q_8) - 3(Q_4 - Q_6)$$
$$Q_2 = m^2 + \frac{1}{3}d^2, \qquad Q_4 = m^4 + 2 m^2 d^2 + \frac{1}{5}d^4, \qquad Q_6 = m^6 + 5 m^4 d^2 + 3 m^2 d^4 + \frac{1}{7}d^6$$
$$Q_8 = m^8 + \frac{28}{3} m^6 d^2 + 14 m^4 d^4 + 4 m^2 d^6 + \frac{1}{9}d^8$$
Whenever $\max(|u_{n-1}|, |u_n|) > 1.0$, compute the rail-crossing average over active length $\ell_0 = \max(0, \min(\max(u_{n-1}, u_n), 1) - \max(\min(u_{n-1}, u_n), -1))$ via $\ell_0 A_{\text{odd/even},3} \cdot$ `rcp_pos_nr2(max(distance, RAIL_CROSSING_MIN_SPAN))` (selecting the interior midpoint when `distance <= TINY_CROSSING_EPS` (`1.0e-6`), with `RAIL_CROSSING_MIN_SPAN = 1.0e-6`, and emitting `0.0` outside `[-1, 1]`).

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

`SONA` `v3.0` permanently retires the coupled `GRU` (`stage2_gru`) and models preamps, asymmetric drives, diodes, power-amp sag, crossovers, and resonant sweeps via a **4-block Matryoshka `SONA` State-Space Core** (`Block 0 -> Block 1 -> Block 2 -> Block 3`, with `10` logical `4`-lane groups sorted as `[(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (0,2), (2,1), (0,3), (1,2)]` and `35` causal lower-triangular inter-block mixing tiles). Each block composes a passive **`2×2` ZDF `TPT-SVF` bank** $\to$ **decoupled `1D` scalar `ParaLRU`** $\to$ **pure-`f32` divided-difference `PolyADAA1`** (plus $C^2$ wavefolder lanes), monomorphized over **compile-time `const` `SonaTierSpec` profiles locked at `Gate 0b`** (`0` capture-time tier resizing):
- **`LiteSpec` (`[1, 1, 1, 1]`):** `4` groups (`16` lanes, `3` inter-block tiles), `1×` `PolyADAA1`, `svf_quads=4` (`16` scalar `TPT-SVF` modes), `fir_taps=32`, `kautz_pairs=8`, `tank_quads=1` (`opt_lines=4`, `<= 4,608 B` `SonaHotSlab`).
- **`FullSpec` (`[2, 2, 1, 1]`):** `6` groups (`24` lanes, `13` inter-block tiles), `2×` OS (`96 kHz`), `svf_quads=6` (`24` scalar `TPT-SVF` modes), `fir_taps=48`, `kautz_pairs=12`, `tank_quads=2` (`opt_lines=8`, `<= 7,936 B`).
- **`ProSpec` (`[3, 2, 2, 1]`):** `8` groups (`32` lanes, `23` inter-block tiles), `2×` OS (`96 kHz`), `svf_quads=8` (`32` scalar `TPT-SVF` modes), `fir_taps=96`, `kautz_pairs=24`, `tank_quads=3` (`opt_lines=12`, `<= 15,872 B`).
- **`UltraSpec` (`[4, 3, 2, 1]`):** `10` groups (`40` lanes, `35` inter-block tiles), `2×` OS (`4×` Bounce), `svf_quads=10` (`40` scalar `TPT-SVF` modes), `fir_taps=128`, `kautz_pairs=32`, `tank_quads=4` (`opt_lines=16`, `<= 30,720 B`).

1. **Decoupled `1D` Scalar `ParaLRU` Recurrence, Zero-DC Soft Rectifier, Exact Forward Tile & $O(\log_2 N)$ Linear Reverse-Adjoint Scans:**
   - *Cancellation-Free Zero-DC Soft Rectifier (`soft_rect_zero_dc` in `dnam-kernels` / WGSL):* With `SOFT_RECT_EPS` $\varepsilon = 10^{-3}$ and `SOFT_RECT_LARGE_Z = 1.0e15`, let $z_{\text{safe}} = \min(|z|, 10^{15})$, $q = z_{\text{safe}}^2 + \varepsilon^2$, $r_{\text{inv}} =$ `rsqrt_pos_nr3(q)`, $r_{\text{rad}} = q \cdot r_{\text{inv}}$, $w_\varepsilon =$ `rcp_pos_nr3(r_rad + eps)`, and $w_z =$ `rcp_pos_nr3(r_rad + z_safe)` to evaluate $\phi(z) = \frac{1}{2}(z + \sqrt{z^2 + \varepsilon^2} - \varepsilon)$:
     - Zero guard ($|z| \le 0.0$): emit exact `+0.0 = 0x0000_0000` (`phi = +0.0`, `phi' = 0.5`).
     - Positive branch ($z > 0$): $\phi(z) = \frac{1}{2}(z_{\text{safe}} + z_{\text{safe}}^2 w_\varepsilon)$ (`|z|` for $|z| > 10^{15}$), $\phi'(z) = \frac{1}{2}(1 + z_{\text{safe}} r_{\text{inv}})$.
     - Rationalized negative branch ($z < 0$): avoiding $(z + \sqrt{z^2+\varepsilon^2} - \varepsilon)$ cancellation:
       $$\phi(z) = -\frac{\varepsilon}{2} (z_{\text{safe}} w_z)(1 + z_{\text{safe}} w_\varepsilon), \qquad \phi'(z) = \frac{\varepsilon^2 r_{\text{inv}}}{2(r_{\text{rad}} + z_{\text{safe}})}$$
       saturating to $(-\varepsilon/2, 0.0)$ when $|z| > 10^{15}$.
   - *Scalar Contractive Recurrence & Parameter Clamps:* Each lane clamps $\alpha_{\text{rel}} \in [0.0, r_{\max}]$ where `MAX_PARALRU_R_F32 = 0x3F7F_FFAC` ($\approx 0.999995$), $\alpha_{\text{atk}} \in [0.0, 1.0]$, and $\kappa \in [0.0, 1.0]$; evaluates $d = \alpha_{\text{rel}} v[n-1]$ and $\phi_n = \phi(u[n] - \kappa d)$; and updates with subnormal flush:
     $$v_i[n] = \alpha_{\text{rel},i} v_i[n-1] + (1 - \alpha_{\text{atk},i}) \phi\big(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1]\big)$$
     Because every lane's Jacobian satisfies:
     $$J_i[n] = \frac{\partial v_i[n]}{\partial v_i[n-1]} = \alpha_{\text{rel},i}\Big(1 - \kappa_i (1 - \alpha_{\text{atk},i}) \phi'\big(u_i[n] - \kappa_i \alpha_{\text{rel},i} v_i[n-1]\big)\Big) \in \big[\alpha_{\text{rel},i}(1 - \kappa_i(1 - \alpha_{\text{atk},i})), \alpha_{\text{rel},i}\big] \subset [0, r_{\max}] \subset [0, 1)$$
     `ParaLRU` is unconditionally contractive without post-hoc ISS clamping (`0` `w_rec *= 0.5`).
   - *Exact In-SRAM Forward Tile (`run_paralru_forward_tile`) + $O(\log_2 N)$ Linear Reverse-Adjoint Scan (`run_paralru_adjoint_scan_tile`):* Because $f_u(v)$ is strictly convex with a sharp $\varepsilon = 10^{-3}$ active-set kink and non-zero $\kappa(1-\alpha_{\text{atk}})$, few-step time-parallel forward Newton scans stagnate at active-set switchbacks (yielding multi-ULP to $> 10.0$ errors on sustained drives); therefore `SONA` evaluates `ParaLRU` forward exactly and runs parallel Blelloch suffix scans on its exact linear adjoint:
     1. **Step 1 — Exact Lane-Parallel In-SRAM Forward Recurrence (`run_paralru_forward_tile`, `0-ULP` CPU Identity):** Each decoupled scalar lane steps `step_paralru` over the `u[0..N]` tile in registers/SRAM (delegating directly to `step_paralru_vec4` lane `0` on CPU for `0-ULP` parity with SIMD runtime and `<= 1.5 ULP`/step under WGSL FMA contraction, requiring only `1x` `soft_rect_zero_dc` cost and `0` intra-tile barriers; partial tail tiles with `valid_len < N` hold state and zero-pad `i >= valid_len` via `run_paralru_forward_partial_tile`). On WebGPU, `TPT-SVF` forward (`TptScanElem`) and `PolyADAA1` / $C^2$ wavefolder remain `256`-thread time-parallel across frames within `<= 15.0 KiB` `var<workgroup>` SRAM (`four_stage_pipeline.md` §3.2).
     2. **Step 2 — Exact $O(\log_2 N)$ Reverse `LinearScanElem` Adjoint Suffix Scan (`run_paralru_adjoint_scan_tile` with Cross-Tile `adj_v_out`):** Given converged forward trajectory `v_traj[0..N]` (recomputed in-tile from the saved tile-start checkpoint), sample-local Jacobian `J_i = alpha_rel * (1 - kappa * (1 - alpha_atk) * phi'(z_i))` at $z_i = u_i - \kappa \alpha_{\text{rel}} v_{i-1}$ ($v_{-1} = v_{\text{init}}$) is `100%` linear and associative. Form sample-local elements `elem_i = LinearScanElem { a: J_i, b: J_i * adj_v_i }` with identity `(1.0, 0.0)` and ordered composition `compose_linear_scan(first, second) = LinearScanElem { a: second.a * first.a, b: second.a * first.b + second.b }` (`second(first(v))`), and execute an $O(\log_2 N)$ reverse-time suffix scan: in reverse time, `suffix[i+1]` precedes `elem_i`, so **`suffix[i] = compose_linear_scan(suffix[i+1], elem_i)`** with `suffix[N] = LinearScanElem::identity()`, multiplying `suffix[i+1]` by `elem_i.a = J_i` (`b_i = J_i * suffix[i+1].b + J_i * adj_v_i`). Passing the incoming boundary sensitivity **`adj_v_out: f32`** from `tile + 1` (plus `tile_summary = suffix[0]`) yields exact unbroken full-stem BPTT gradients (`0` `5.33 ms` truncation across slow release/sag poles):
        $$\lambda_i = \bar{v}_i + \mathtt{suffix}[i+1].a \cdot \bar{v}_{\text{out}} + \mathtt{suffix}[i+1].b, \qquad \nabla_{v_{\text{init}}} L = \mathtt{suffix}[0].a \cdot \bar{v}_{\text{out}} + \mathtt{suffix}[0].b$$
        $$\nabla_{u_i} L = \lambda_i (1 - \alpha_{\text{atk}})\phi'(z_i), \qquad \nabla_{\alpha_{\text{rel}}} L = \sum_{i=0}^{N-1} \lambda_i v_{i-1}\big(1 - \kappa(1 - \alpha_{\text{atk}})\phi'(z_i)\big)$$
        $$\nabla_{\alpha_{\text{atk}}} L = -\sum_{i=0}^{N-1} \lambda_i \phi(z_i), \qquad \nabla_{\kappa} L = -\sum_{i=0}^{N-1} \lambda_i \alpha_{\text{rel}}(1 - \alpha_{\text{atk}})\phi'(z_i) v_{i-1}$$

2. **Continuous-Time ZOH Rate Transfer ($\rho_{\text{sr}} = f_{s,\text{ref}} / f_{s,\text{eff}}$) & Retention Invariance:**
   - **`TPT-SVF`:** Clamp $g_{\text{ref}} \in [10^{-5}, 64.0]$ and rescale via exact ZOH cutoff invariance:
     $$g_{\text{host}} = \mathrm{clamp}\Big(\tan\big(\rho_{\text{sr}} \arctan(g_{\text{ref}})\big), 10^{-5}, 64.0\Big)$$
   - **`ParaLRU`:** Erase legacy `rate_1x_cal` dual-rate override; full continuous-time ZOH step-response matching over $\rho_{\text{sr}} = f_{s,\text{ref}} / f_{s,\text{eff}}$ defines the single canonical invariant $(\alpha_{\text{rel}}', \alpha_{\text{atk}}', \kappa')$ for $\beta = 1 - \alpha_{\text{atk}}$ and $a_{\text{atk}} = \alpha_{\text{rel}}(1 - \kappa \beta)$:
     $$\alpha_{\text{rel}}' = \mathrm{clamp}\big(\alpha_{\text{rel}}^{\rho_{\text{sr}}}, 0.0, r_{\max}\big), \qquad \beta' = \mathrm{clamp}\left(\beta \frac{1 - a_{\text{atk}}^{\rho_{\text{sr}}}}{1 - a_{\text{atk}}}, 0.0, 1.0\right), \qquad \alpha_{\text{atk}}' = 1 - \beta'$$
     where $(1 - a_{\text{atk}}^{\rho_{\text{sr}}}) / (1 - a_{\text{atk}}) \to \rho_{\text{sr}}$ continuously as $a_{\text{atk}} \to 1$, and $\kappa' = \mathrm{clamp}\big((1 - (1 - \kappa\beta)^{\rho_{\text{sr}}}) / \beta', 0.0, 1.0\big)$ when $\beta' > 10^{-12}$ (else $\kappa' = \kappa$). `PolyADAA1` provides `1×` aliasing protection when `Lite` executes unconditionally at `1×`, while `2×` OS provides `-90 dB` alias immunity on `Full..Ultra` with causal `poly3` / elliptic halfband decimation.
3. **Bounded State & Unclipped Envelope Invariant:** Every `ParaLRU` lane remains unconditionally BIBO-bounded under all finite inputs (`REQ-7.2`), while direct-register `VarPro` linear readout paths (`pre_cab` and `direct`) carry unclipped audio across `±100.0` (`+40 dBFS`, `REQ-7.1`).

### 4.1 Passive `2×2` ZDF `TPT-SVF` Four-Filter Tone Basis & Associative Scans

`r[dsp.op-n.stage2.tone-basis]`

Each passive `2×2` ZDF `TPT-SVF` mode inside the 4-block `SONA` core (`dnam-kernels::sona_core::tpt_svf`) operates over clamped physical domains $g \in [10^{-5}, 64.0]$, $k \in [0.0, 8.0]$ ($f_c = \frac{f_s}{\pi}\arctan(g)$), computing division-free admittance $a_1 =$ `rcp_ge1_nr2(1 + g * (g + k))` (equivalent within `1 ULP` to `rcp_pos_nr3`), $a_2 = g a_1$, $a_3 = g a_2$, and evaluating trapezoidal ZDF integrators with subnormal-flushed states $s_1, s_2$:
$$v_3 = u - s_2, \qquad v_1 = a_1 s_1 + a_2 v_3, \qquad v_2 = s_2 + a_2 s_1 + a_3 v_3$$
$$\mathrm{bp} = v_1, \qquad \mathrm{lp} = v_2, \qquad \mathrm{hp} = u - k v_1 - v_2, \qquad \mathrm{ap} = u - 2 k v_1$$
$$s_1' = \mathrm{flush}(2 v_1 - s_1), \qquad s_2' = \mathrm{flush}(2 v_2 - s_2)$$
The learned linear tone tap mix $y_{\text{tone}} = w_{\text{lp}} \mathrm{lp} + w_{\text{bp}} \mathrm{bp} + w_{\text{hp}} \mathrm{hp} + w_{\text{ap}} \mathrm{ap} + w_{\text{dir}} u$ (`tone_*` weights quantized in `binary16` over `[-16.0, 16.0]`) spans arbitrary multi-band crossovers, resonant tone stacks, and `4–12`-pole phasers strictly via data-driven coefficients (`REQ-1.9`). Equivalently in linear state-space matrix form $\mathbf{s}_{n+1} = \mathbf{A}_{\text{tpt}}\mathbf{s}_n + \mathbf{b}_{\text{tpt}} u_n$:
$$(\mathbf{A}_{\text{tpt}})_{11} = 2 a_1 - 1, \quad (\mathbf{A}_{\text{tpt}})_{12} = -2 a_2, \quad (\mathbf{A}_{\text{tpt}})_{21} = 2 a_2, \quad (\mathbf{A}_{\text{tpt}})_{22} = 1 - 2 a_3, \quad b_1 = 2 a_2, \quad b_2 = 2 a_3, \quad \Vert \mathbf{A}_{\text{tpt}} \Vert_2 \le 1$$
Forward and backward WebGPU scans compose via $(\mathbf{A}_2, \mathbf{b}_2) \circ (\mathbf{A}_1, \mathbf{b}_1) = (\mathbf{A}_2 \mathbf{A}_1, \mathbf{A}_2 \mathbf{b}_1 + \mathbf{b}_2)$ (and reverse transpose $(\mathbf{A}_1^\top, \mathbf{g}_1) \circ_{\text{rev}} (\mathbf{A}_2^\top, \mathbf{g}_2) = (\mathbf{A}_1^\top \mathbf{A}_2^\top, \mathbf{A}_1^\top \mathbf{g}_2 + \mathbf{g}_1)$).

### 4.2 Stereo Basis, Lower-Triangular `4×4` Inter-Block Mixing & `StereoVec8` Parity

`r[dsp.op-n.stage2.stereo-coupling]`

Inter-lane mixing across the 4-block `SONA` core occurs strictly **between blocks** via **lower-triangular `4×4` spatial mixing tiles** (`Lite ⊂ Full ⊂ Pro ⊂ Ultra` prefix nesting, preserving independent L/R trajectories alongside cross-channel stereo coupling), while direct-register `VarPro` linear readouts (`pre_cab` feeding Stage III and `direct` feeding stereo output) accumulate directly in SIMD registers with zero intermediate `feature_bus_scratch` materialization.

- `r[dsp.op-n.stage2.conv6-simd]` **Wide-ISA `StereoVec8` Dispatch (`AVX2+FMA`, `SVE2`, `SME`) + Interleaved `2× SimdVec4` (`NEON`, `WASM SIMD128`) & `0-ULP` Parity ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):** `SONA`'s core (`process_tick16_const`) steps `Left` (`4` lanes) and `Right` (`4` lanes) simultaneously in `256-bit` / scalable `StereoVec8` registers (`AVX2+FMA`, `SVE2`, `SME`) or interleaved `2× SimdVec4` (`NEON`, `WASM SIMD128`) inside `16`-frame sub-block register-hoisted loops (`TPT-SVF` coeffs, `ParaLRU` scalars, `PolyADAA1` states, and `4×4` inter-block tiles pinned in vector registers with `0` GPR spills), enforcing identical fixed `Vec4`-pair reduction trees for exact `0-ULP` cross-ISA parity.

### 4.3 $3\text{ kHz}$ Gauge-Invariant `z_stage` Control Sensitivity & Static `W_ro,0` Readout

`r[dsp.op-n.stage2.latent-folding]`

`SO(4)` buyer gauge `Q_buyer` rotates ONLY `z_head[0..4]` (plus `cols 0..4` of `main_base`, `u_int_base`, `w_ctrl`, and `u_int_ultra` Block `0`), while **`z_stage[0..D_s]`** (`D_s = 4` `ctrl_basis_4[0..4]` for `Lite..Pro`; `D_s = 8` `[ctrl_basis_4[0..4], z_stage_ultra[0..4]]` for `Ultra`) is unrotated (`cols 4..8` / `u_int_ultra` Blocks `1..2`). `VarProReadout*Wire` stores **ONLY one static base matrix `W_ro,0`** (`16×12` / `24×12` / `32×12` / `40×12`, zero per-latent `W_{ro,1..D_h}` wire matrices and zero `b_a` wire biases). At each $3\text{ kHz}$ control boundary (`0` extra `48 kHz` MACs when controls are static), `z_head` (`[z_head[0..4], z_head_ultra[0..4]]`) stays inside `W_ctrl` (`f32`), while ALL `stage_latent_*` `fp16` additive sensitivity tables index strictly unrotated **`z_stage[0..D_s]`** (preserving `0-ULP` `SO(4)` buyer-watermark invariance):
1. **`stage_targets[0..16]`** via `16 × 16` `W_ctrl` (+ `16 × 4` `W_ctrl_ultra` on `Ultra`).
2. **Global Scalar & Base `[4][4]` Sensitivities (`VarProReadoutLiteWire.stage_latent_mods_fp16[0..80]`):** `[0..8]` stores `default_knobs[8]` (resting physical control defaults in `[0.0, 1.0]^8` for standalone `Arena 0` preset/audit inspection; `z_stage == 0` at `StarCenter`, adding `0.0` at runtime); `[8..48]` (`w_svf_g_oct[8]`, `w_svf_k_delta[8]`, `w_kautz_oct[8]`, `w_kautz_damp[8]`, `w_tank_psi[8]`) sums over `k in 0..D_s-1` against `z_stage[k]` into `svf_g_oct`, `svf_k_delta`, `kautz_oct_delta`, `kautz_damp_delta`, and `tank_psi_delta` ([`parametric_control_bus.md` §3.1](parametric_control_bus.md#31-stage-parameter-dispatch-matrix-mathbfw_textctrl)). On `Lite..Pro` (`D_s = 4`, no `Arena 3` dispatch), `[48..80]` (`w_adaa_drive_scale[4][4]`, `w_lru_bias_shift[4][4]`) yields per-block `block_adaa_oct[b] = clamp(stage_targets[2 + b] + sum_{k=0..3} w_adaa_drive_scale[k][b] * z_stage[k], -1.5, +1.5)` and `block_lru_bias[b] = clamp(stage_targets[6 + b] + sum_{k=0..3} w_lru_bias_shift[k][b] * z_stage[k], -2.0, +2.0)` for `b in 0..4` (shared across all groups inside block `b`).
3. **Superseding `Ultra` `[8][10]` Group Table (`Arena3ControlBusWire.stage_latent_ctrl_ultra_fp16[0..160]`):** On `Ultra` (`D_s = 8`), `w_adaa_drive_scale[8][10]` (`[0..80]`) + `w_lru_bias_shift[8][10]` (`[80..160]`; `[160..256]` reserved `0`) **supersedes (replaces, never adds to)** `Arena 0`'s `[4][4]` subset, computing per-group `group_adaa_oct[g] = clamp(stage_targets[2 + block_of(g)] + sum_{k=0..7} w_adaa_drive_scale[k][g] * z_stage[k], -1.5, +1.5)` and `group_lru_bias[g] = clamp(stage_targets[6 + block_of(g)] + sum_{k=0..7} w_lru_bias_shift[k][g] * z_stage[k], -2.0, +2.0)` for `g in 0..10` (`block_of(g) = SONA_GROUP_BLOCK_AND_LOCAL[g].0`: `{0, 4, 6, 8} -> 0`, `{1, 5, 9} -> 1`, `{2, 7} -> 2`, `{3} -> 3`).
4. **Per-Lane `SonaCoreUltraDeltaWire.stage_latent_core_mods_fp16[0..1024]` (`V3_META_CORE_MODS_PRESENT` = `0x0000_0200`):** Row-major `[k in 0..8][col in 0..128]`. When `V3_META_CORE_MODS_PRESENT == 1` on `Ultra` (`D_s = 8`), for each lane `l in 0..40`:
   - `delta_g_oct[l] = sum_{k=0..7} core_mods[k][l] * z_stage[k]` $\to$ `g_oct_lane[l] = clamp(svf_g_oct + delta_g_oct[l], -3.0, +3.0)` (`g_scale[l] = 2^{g_oct_lane[l]}` multiplying `svf_g[l]` clamped to `[1e-5, 64.0]`);
   - `delta_k[l] = sum_{k=0..7} core_mods[k][40 + l] * z_stage[k]` $\to$ `svf_k_eff[l] = clamp(svf_k[l] + svf_k_delta + delta_k[l], 0.0, 8.0)`;
   - `delta_wf[l] = sum_{k=0..7} core_mods[k][80 + l] * z_stage[k]` $\to$ `wf_mix_eff[l] = clamp(wavefolder_mix[l] + delta_wf[l], 0.0, 1.0)`;
   - `col in 120..128` is reserved `0x0000`.
   When `V3_META_CORE_MODS_PRESENT == 0`, skip `[0..1024]` (`0` MACs).

---

## 5. `Op-N[Ctrl]`: 3 kHz Algebraic Polynomial & $O(1)$ Sufficient-Statistic Engine

1. `r[dsp.op-n.ctrl.incremental-update]` **Incremental Sufficient-Statistic Update:** When contributor $p \in \mathcal{I}_{\text{moved}}$ changes from $u_{p,\text{old}}$ to $u_{p,\text{new}}$, update the exact-integer main-effect and Rank-$R$ pair-statistic accumulators (`64 B` in `Lite..Pro`, `128 B` in `Ultra`; self-contained in `Arena 0` for `Lite` strip safety; see [`parametric_control_bus.md` §2.3](parametric_control_bus.md#23-o1-additive-latent-sufficient-statistic-control-scaling)):
   $$\Delta\mathbf{z}_{\text{main}} = \mathbf{W}_{\text{main}, p} \cdot \big(\boldsymbol{\psi}_p(u_{p, \text{new}}) - \boldsymbol{\psi}_p(u_{p, \text{old}})\big)$$
2. `r[dsp.op-n.ctrl.lobatto-basis]` **Node-Anchored Control Basis:** Evaluate the persisted observed anchor nodes and switch contributor slices ([`parametric_control_bus.md` §2.2](parametric_control_bus.md#22-cardinality-bounded-node-anchored-newtonlobatto-basis)).
3. `r[dsp.op-n.ctrl.chebyshev-lfo]` **Chebyshev LFO Waveshaping:** Shape quadrature rotor states via $T_1(x) = x$, $T_2(x) = 2x^2 - 1$, $T_3(x) = 4x^3 - 3x$, $T_4(x) = 8x^4 - 8x^2 + 1$.
4. `r[dsp.op-n.ctrl.fm-interaction]` **Rank-4/8 FM Bilinear Interaction:** Evaluate interaction factors from scalar embeddings $z_p = \mathbf{w}_{\text{proj},p}^\top\boldsymbol{\psi}_p(u_p)$ and mix via $\mathbf{U}_{\text{int}}$ (`8×4` base in `Arena 0` + three `8×4` `Ultra` delta blocks in `Arena 3`).
5. `r[dsp.op-n.ctrl.trust-clip]` **Groupwise Trust-Region Clipping:** Apply the outward-checked trust-region clip independently to $\mathbf{z}_{\text{head}}$ and $\mathbf{z}_{\text{stage}}$ ([`parametric_control_bus.md` §1.2](parametric_control_bus.md#outward-checked-trust-region-clip)).
