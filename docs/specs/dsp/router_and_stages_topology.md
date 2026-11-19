---
id: spec.dsp.router
title: "Stage 0 Router, Top-of-Frame Causal Crossbar, 4-Block SONA Topology & Direct-Register VarPro Head"
status: canonical
version: "3.0.0"
domain: dsp
rule_prefixes:
  - "r[dsp.router."
  - "r[dsp.stage1."
  - "r[dsp.stage2."
  - "r[dsp.stage3."
  - "r[dsp.stage4."
  - "r[dsp.head."
  - "r[dsp.zoh."
requirements:
  - REQ-1.5
  - REQ-7.4
adrs:
  - ADR-0006
  - ADR-0019
  - ADR-0022
---

# Stage 0 Router, Top-of-Frame Causal Crossbar, `4`-Block `SONA` Topology & Direct-Register `VarPro` Head

This specification defines the **Stage 0 8-Port Matrix Router**, the **Top-of-Frame Causal Operator Crossbar (`Op-P` + `4× Q0.32` Carriers + `Op-T` Causal Reads)**, the **4-Block Matryoshka `SONA` State-Space Core (`TPT-SVF -> ParaLRU -> PolyADAA1`)** executed via `process_tick16_const` over a contiguous **`SonaHotSlab`** (`<= 4.5 / 7.8 / 15.5 / 30.0 KiB`) and compile-time **`const` `SonaTierSpec`**, **Stage III Cabinet (`32..128` FIR + `8..32` Kautz)**, **Coercive `Op-T` Feedback Loop (`||G_loop|| < 1`)**, **Direct-Register `VarPro` Mid/Side Readouts**, and **Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability (`44.1–384 kHz`, `2×` OS & `4×` Bounce)** across **Dynamic Neural Audio Modeler (DNAM)** `SONA` `v3.0` ([`packages/engine/src/engine/`](../../../packages/engine/README.md), [`packages/capture/src/`](../../../packages/capture/README.md), [`sona_architecture.md`](../architecture/sona_architecture.md), [`REQ-7.4`](../../requirements/system_goals_and_requirements.md)). For architectural rationales and counterexamples, see [`../../adr/0006-model-declared-delays-and-tank-feedback-guard.md`](../../adr/0006-model-declared-delays-and-tank-feedback-guard.md) and [`../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md).

---

## 0. Detailed Signal Flow & Stage Boundaries

`r[dsp.router.signal-flow]`

```mermaid
flowchart TD
    subgraph ControlBusGroup ["3 kHz Unified Control Plane (Arena 0 Self-Contained + Ultra Delta)"]
        CtrlSrc["Arena 0 Descriptors, CP1, U_int, W_ctrl & fp16 W_readout"] --> OpN["Op-N[Ctrl]: Exact-Integer Sufficient-Statistic Engine<br/>+ 3 kHz TPT-SVF, ParaLRU & Direct VarPro Readout Collapse"]
        SendDetector["Instrument / Independent Aux Detector"] --> SendMod["Send-Safe Dynamic Vector m_send"]
        DownDetector["Downstream Instrument / Return / Aux Detector"] --> DownMod["Downstream Dynamic Vector m_down"]
        OpN & SendMod --> SendBus["Send Dispatch Rows 0..4<br/>16x16 Base + Ultra 16x4 Delta + Micro-Ramp"]
        OpN & DownMod --> DownBus["Downstream Dispatch Rows 5..15<br/>16x16 Base + Ultra 16x4 Delta + Micro-Ramp"]
    end

    In0["Port In 0: Inst In L"] & In1["Port In 1: Inst In R"] --> PreRouter["Stage 0: Instrument & Send Routing<br/>• Mechanical Normaling<br/>• Voltage-Normalized Mid/Side"]
    In2["Port In 2: FX Return L"] & In3["Port In 3: FX Return R / Aux Key"] --> DownRouter["Stage 0: Return & Downstream Routing<br/>• Parallel-Path Sub-Sample Alignment<br/>• Direct Return-to-Send Block Zero"]
    PreRouter -.-> SendDetector
    PreRouter & DownRouter -.-> DownDetector

    PreRouter --> S1["Stage I / Top-of-Frame Causal Crossbar<br/>• Input-Only Op-P (Hilbert H1 + T1 -1/-2 Oct)<br/>• 4x Q0.32 Quadrature Phasors (0.05 Hz..20 kHz)<br/>• Causal Op-T Ring Reads (tau >= 4, M=1/2/3)"]
    S1 --> Out0["Port Out 0: Send L<br/>(External Amp 1)"]
    S1 -->|"Dual-Amp Modes"| Out1["Port Out 1: Send R / WDW Dry"]
    DryTap["WDW Dry Selector<br/>PreSpatialCore / DirectDry"] -->|"WDW"| Out1
    S1 --> DownRouter

    DownRouter -->|"200 Hz TPT LR4 Low: Clean Bypass"| S3
    DownRouter -->|"200 Hz TPT LR4 High: Core Path"| S2["Stage II: 4-Block Matryoshka SONA Core<br/>TPT-SVF -> ParaLRU -> PolyADAA1<br/>(Lite/Full/Pro/Ultra const SonaTierSpec)"]
    S2 -->|"pre_cab register path"| S3["Stage III: Cabinet & Modals<br/>(32/48/96/128 FIR + 8/12/24/32 Kautz Pairs)"]
    S3 --> S4["Stage IV: Coercive Op-T Feedback Write<br/>(||G_loop|| < 1, Givens+Householder, Schroeder, 0-ULP Freeze)"]

    DownBus --> DownRouter & S1 & S2 & S3 & S4
    SendBus --> PreRouter & S1

    S1 & S2 & S3 -->|"direct VarPro linear registers"| DirectOut["Direct-Register Stereo Combine<br/>& Calibrated Tier Gain Trim"]
    S4 --> DirectOut
    S4 -. "z^-tau (tau >= 4)" .-> S1
    DownBus --> DirectOut
    DirectOut --> Out2["Port Out 2: Main L / WDW Wet L"]
    DirectOut --> Out3["Port Out 3: Main R / WDW Wet R"]
```

---

## 1. Stage 0: 8-Port Matrix Router (`REQ-7.4`)

`r[dsp.router.ports]`

| Port Identifier | Signal Variable | Physical & Logical Routing Role |
| :--- | :--- | :--- |
| **`Port In 0 (Inst In L)`** | `x_{inst,L}[n]` | Primary mono or Left instrument input signal. |
| **`Port In 1 (Inst In R)`** | `x_{inst,R}[n]` | Secondary Right instrument input signal; mechanically normaled to `Port In 0` when unplugged. |
| **`Port In 2 (FX Return L)`** | `x_{ret,L}[n]` | External amplifier effects loop return for Amp 1 / Left channel. |
| **`Port In 3 (FX Return R / Aux Key)`** | `x_{ret,R}[n]` or `x_{aux}[n]` | External amplifier effects loop return for Amp 2 / Right channel OR dedicated auxiliary sidechain detector key; mechanically normaled to `Port In 2` when unplugged in stereo return mode. |
| **`Port Out 0 (Send L)`** | `y_{send,L}[n]` | Pre-loop preamp send driving Amp 1 / Left external amplifier front-end (or optional mono send in WDW). |
| **`Port Out 1 (Send R / WDW Dry)`** | `y_{send,R}[n]` or `y_{dry}[n]` | Right external send in dual-amp modes; dedicated center dry output in WDW (`PreSpatialCore` or `DirectDry`). |
| **`Port Out 2 (Main Out L)`** | `y_{out,L}[n]` or `y_{wet,L}[n]` | Main output Left; wet-only Left in WDW. |
| **`Port Out 3 (Main Out R)`** | `y_{out,R}[n]` or `y_{wet,R}[n]` | Main output Right; wet-only Right in WDW. |

#### WDW Dry-Tap and Wet-Output Contract

`r[dsp.router.wdw-contract]`

| `WdwDryTap` Mode | Center Dry on `Port Out 1` | Wet-Only L/R on `Ports Out 2/3` | Default Applicability |
| :--- | :--- | :--- | :--- |
| **`PreSpatialCore`** | Mid contribution `m_c` of Top-of-Frame Crossbar + `SONA` Core + Stage III | Coercive `Op-T` spatial loop contribution `m_w ± α s_w` | Default for WDW-capable full-rig models |
| **`DirectDry`** | Voltage-normalized Mid `0.5(x_L + x_R)` of instrument input (or active aligned return) | Effect-only model response relative to the selected direct source | Default for standalone wet-effect or external-center-amp rigs |

For `PreSpatialCore`, partition the non-recirculating core $(m_c, s_c)$ from the Stage IV `Op-T` spatial contribution $(m_w, s_w)$ with width $\alpha$:
$$y_L = m_c + m_w + \alpha(s_c + s_w), \qquad y_R = m_c + m_w - \alpha(s_c + s_w)$$
$$y_{\text{dry}} = m_c, \qquad y_{\text{wet},L} = m_w + \alpha s_w, \qquad y_{\text{wet},R} = m_w - \alpha s_w$$

#### $4 \times 4$ Scattering Matrix $\mathbf{S}_{\mathrm{router}}$ & Return-to-Send Isolation

`r[dsp.router.return-to-send-isolation]`

```text
                  [ S_00  S_01 |   0     0   ]   [ S_{inst->send}    0_{2x2}    ]
S_router =        [ S_10  S_11 |   0     0   ] = [                              ]
                  [------------+-------------]   [                              ]
                  [ S_20  S_21 | S_22  S_23  ]   [ S_{inst->down}  S_{ret->down}]
                  [ S_30  S_31 | S_32  S_33  ]
```

$$\mathbf{y}_{\mathrm{send}}[n] = \mathbf{S}_{\mathrm{inst} \to \mathrm{send}} \mathbf{x}_{\mathrm{inst}}[n], \qquad \mathbf{u}_{\mathrm{down}}[n] = \mathbf{S}_{\mathrm{inst} \to \mathrm{down}} \mathbf{x}_{\mathrm{inst}}[n] + \mathbf{S}_{\mathrm{ret} \to \mathrm{down}} \mathbf{x}_{\mathrm{ret, aligned}}[n]$$

1. **Direct-Path Isolation:** Enforcing $\mathbf{S}_{\mathrm{ret} \to \mathrm{send}} \equiv \mathbf{0}_{2 \times 2}$ guarantees open-loop return ratio $\mathbf{T}_{\mathrm{loop}}(z) = \mathbf{S}_{\mathrm{ret} \to \mathrm{send}} \mathbf{H}_{\mathrm{amp}}(z) \equiv \mathbf{0}_{2 \times 2}$ and $\det(\mathbf{I} - \mathbf{T}_{\mathrm{loop}}(z)) \equiv 1$.
2. **Send-Safe Control & Detector Partition:** Pre-loop crossbar and router send gains (`θ_0..θ_4`) consume only the send-safe dynamic vector $\mathbf{m}_{\text{send}}$ (derived from instrument inputs and any independent auxiliary key), never return-derived samples or detector histories; downstream stages (`θ_5..θ_15`) consume $\mathbf{m}_{\text{down}}$ ([`parametric_control_bus.md` §3.1](parametric_control_bus.md#31-stage-parameter-dispatch-matrix-mathbfw_textctrl)).

#### Parallel-Path Sub-Sample Alignment

`r[dsp.router.parallel-alignment]`

| `AlignmentMode` | Interpolation Kernel | Host-Rate Centering Offset (`G`) | Selection & Offline Bounce Rule |
| :--- | :--- | :---: | :--- |
| **`FullBand64Tap`** | 64-tap Kaiser-derived windowed sinc (`β = 8.6`) | `31` host samples (`0.65 ms` @ `48 kHz`) | Default for full-band 4CM/WDW, training, certification, and `4×` offline bounce |
| **`LowLatency16Tap`** | 16-tap Kaiser-derived windowed sinc (`β = 6.5`) | `7` host samples (`0.15 ms` @ `48 kHz`) | Selectable low-latency live monitoring mode |
| **`MinLatency4Tap`** | Centered 4-tap Catmull-Rom cubic Hermite | `1` host sample (`0.021 ms` @ `48 kHz`) | Low-band alignment or minimum-latency live monitoring mode |

For $K \in \lbrace 16, 64 \rbrace$ with $G = (K - 2)/2$, $R = K/2$, and $s_k = k - G - \mu$, prepare the normalized zero-ended Kaiser-derived sinc weights on the cold path:
$$w_{\beta}(s) = \frac{I_0\big(\beta\sqrt{1 - (s/R)^2}\big) - 1}{I_0(\beta) - 1} \quad (|s| \le R), \qquad h_k(\mu) = \frac{\mathrm{sinc}(s_k) w_{\beta}(s_k)}{\sum_{j=0}^{K-1}\mathrm{sinc}(s_j) w_{\beta}(s_j)}$$
For oversampling factor $M \in \lbrace 1, 4 \rbrace$, measured effective-rate arrivals $a_i$, $A = \max_i a_i$, and optional common host-sample pad $p \ge 0$, set $\Delta_i = A - a_i + M p$, $J_i = \lfloor \Delta_i \rfloor$, $\mu_i = \Delta_i - J_i$, and $B = (M - 1)G$:
$$y_i[n] = \sum_{k=0}^{K-1} h_k(\mu_i) x_i[n - B - J_i - k], \qquad d_{\mathrm{eff},i} = M G + \Delta_i$$
At `4×` bounce, `FullBand64Tap` applies $B = 93$ plus its `31`-effective-sample center to preserve a `31`-host-sample alignment offset (`43` host samples combined nominal PDC with the `12`-host-sample oversampler; minimum `157` effective-rate samples of ring history per channel). `MinLatency4Tap` ($G = 1$) uses the 4-tap Catmull-Rom basis ($\sum_{k=-1}^2 c_k(\mu) \equiv 1.0$, `0.00 dB` exact DC gain):
$$c_{-1}(\mu) = -0.5\mu(1 - \mu)^2, \quad c_0(\mu) = 1 - 0.5\mu^2(5 - 3\mu), \quad c_1(\mu) = 0.5\mu(1 + 4\mu - 3\mu^2), \quad c_2(\mu) = 0.5\mu^2(\mu - 1)$$

#### Stage 0 LR4 Clean/Drive Crossover

`r[dsp.router.lr4-crossover]`

When parallel sub-bass bypass is active, split the selected downstream source using a 4th-order Linkwitz–Riley (`LR4`, default $f_c = 200\text{ Hz}$) crossover realized as two cascaded 2nd-order Butterworth **trapezoidal state-variable filter (TPT SVF)** sections ($Q = 1/\sqrt{2}$). Prepare cold-path coefficients in `f64` and store as `f32`:
$$g = \tan\left(\frac{\pi f_c}{f_{s,\text{eff}}}\right), \qquad k = \sqrt{2}, \qquad a_1 = \frac{1}{1 + g(g + k)}, \qquad a_2 = g a_1, \qquad a_3 = g a_2$$
Each SVF section updates integrator states $(s_1, s_2)$ without real-time division:
$$v_3 = x - s_2, \qquad v_1 = a_1 s_1 + a_2 v_3, \qquad v_2 = s_2 + a_2 s_1 + a_3 v_3$$
$$\ell = v_2, \qquad h = x - k v_1 - v_2, \qquad s_1' = 2 v_1 - s_1, \qquad s_2' = 2 v_2 - s_2$$
Route $L_4 = L_2^2$ to the Stage III clean bypass and $H_4 = H_2^2$ to the `SONA` core (`|L₄(e^{jω}) + H₄(e^{jω})| ≡ 1`, `-6.02 dB` and equal phase at $f_c$).

---

## 2. Mid/Side Encoding, Output Head & Stereo-Linked Dynamics

### 2.1 Drive-Invariant Voltage-Normalized Mid/Side Input Transformation

`r[dsp.router.mid-side-input]`

To preserve `0.00 dB` drive-threshold parity between mono (`1 → 2`) and centered stereo (`2 → 2`, $x_L = x_R = x$) inputs (`REQ-7.4`), Stage 0 uses **Voltage-Normalized Mid/Side Encoding** rather than orthonormal $1/\sqrt{2}$ power scaling:
$$x_{\mathrm{mid}}[n] = \frac{1}{2}\big(x_L[n] + x_R[n]\big), \qquad x_{\mathrm{side}}[n] = \frac{1}{2}\big(x_L[n] - x_R[n]\big), \qquad x_L[n] = x_{\mathrm{mid}}[n] + x_{\mathrm{side}}[n], \qquad x_R[n] = x_{\mathrm{mid}}[n] - x_{\mathrm{side}}[n]$$

### 2.2 Canonical Mid/Side Output Head Factorization

`r[dsp.head.mid-side-factorization]`

At each `3 kHz` control tick, the $D_h \in \lbrace 4, 8 \rbrace$ control modes collapse the `fp16` `W_readout` subspace (`VarProReadoutSubspaceWire`) directly into Mid (`Slice 0`) and Side (`Slice 1`) linear readout vectors (`pre_cab` and `direct`) over the unconditional `SONA` direct-register state (`0` `256-D` `feature_bus_scratch` scatter/gather):
$$\mathbf{w}_a[n] = \mathbf{w}_{a,0} + \sum_{d=1}^{D_h} z_{\mathrm{head},d}[n] \mathbf{w}_{a,d}, \qquad b_a[n] = b_{a,0} + \sum_{d=1}^{D_h} z_{\mathrm{head},d}[n] b_{a,d}$$
$$\mathbf{w}_{\mathrm{mid}} \triangleq \frac{1}{2}\big(\mathbf{w}_L + \mathbf{w}_R\big), \qquad \mathbf{w}_{\mathrm{side}} \triangleq \frac{1}{2}\big(\mathbf{w}_L - \mathbf{w}_R\big), \qquad b_{\mathrm{mid}} \triangleq \frac{1}{2}\big(b_L + b_R\big), \qquad b_{\mathrm{side}} \triangleq \frac{1}{2}\big(b_L - b_R\big)$$
$$y_{\mathrm{mid}}[n] = \mathbf{w}_{\mathrm{mid}}^\top \boldsymbol{\Phi}[n] + b_{\mathrm{mid}}, \qquad y_{\mathrm{side}}'[n] = \alpha_{\mathrm{width}}[n] \big(\mathbf{w}_{\mathrm{side}}^\top \boldsymbol{\Phi}[n] + b_{\mathrm{side}}\big)$$
$$y_L[n] = y_{\mathrm{mid}}[n] + y_{\mathrm{side}}'[n], \qquad y_R[n] = y_{\mathrm{mid}}[n] - y_{\mathrm{side}}'[n]$$

1. **Mono (`1 → 1`, `2 → 1`) Execution:** Evaluates only Slice 0 ($y_{\mathrm{mid}}[n]$), preserving dual-mic (`SM57 + R121`) and ping-pong delay sums without dropping the right channel while skipping Slice 1 dot-products.
2. **Mid Reconstruction Bound:** For matching state and features, $|m - \hat{m}| \le 2 \mathrm{ulp}(M) + 2 F_{\min}$ where $M = \max(|y_{\mathrm{mid}}|, |y_{\mathrm{side}}'|)$.
3. **Stereo Width Folding ($\alpha_{\mathrm{width}} \in [0.0, 2.0]$):** When static over a control block, fold $\alpha_{\mathrm{width}}$ into $\tilde{\mathbf{w}}_{\mathrm{side}} = \alpha_{\mathrm{width}}\mathbf{w}_{\mathrm{side}}$ and $\tilde{b}_{\mathrm{side}} = \alpha_{\mathrm{width}}b_{\mathrm{side}}$ at the `3 kHz` boundary (`0` extra audio-rate multiplies); when automated, micro-ramp $\alpha_{\mathrm{width}}[n]$ per sample.

### 2.3 Topology-Adaptive Even/Odd SIMD Lane Partitioning

`r[dsp.router.topology-dispatch]`

| Parameter / Feature | Mono (`1 → 1`) | Stereo-to-Mono (`2 → 1`) | Mono-to-Stereo (`1 → 2`) | True Stereo (`2 → 2`) |
| :--- | :---: | :---: | :---: | :---: |
| **Engine Entrypoint** | `process_mono` | `process_stereo_to_mono` | `process_mono_to_stereo` | `process_stereo` |
| **Crossbar & 4-Block `SONA` Core** | Normaled L/R paths & `4×4` coupling retained | Full stereo L/R paths & coupling | Normaled L/R paths & `4×4` coupling retained | Full stereo L/R paths & coupling |
| **Stage III Early FIR & Modals** | Both L/R FIR histories & modal states advance | Both L/R FIR histories & modal states advance | Both L/R FIR histories & modal states advance | Both L/R FIR histories & modal states advance |
| **Output Head Slices** | Slice 0 (`w_mid`) only | Slice 0 (`w_mid`) only | Slices 0 & 1 (`w_mid, w_side'`) | Slices 0 & 1 (`w_mid, w_side'`) |

### 2.4 Programmable Stereo-Linked & Aux-Key Sidechain Envelope Detector

`r[dsp.router.stereo-link-detector]`

The sidechain detector blends peak-linked stereo detection ($\alpha_{\mathrm{link}} \in [0.0, 1.0]$) and external auxiliary keying ($\beta_{\mathrm{key}} \in [0.0, 1.0]$) with zero `-6.02 dB` hard-panned peak loss:
$$e_{\mathrm{link}}[n] = \max\big(|x_L[n]|, |x_R[n]|\big), \qquad e_{\mathrm{key}}[n] = |x_{\mathrm{aux}}[n]|$$
$$e_{\mathrm{det}, c}[n] = (1 - \beta_{\mathrm{key}})\Big((1 - \alpha_{\mathrm{link}})|x_c[n]| + \alpha_{\mathrm{link}} e_{\mathrm{link}}[n]\Big) + \beta_{\mathrm{key}} e_{\mathrm{key}}[n], \qquad c \in \lbrace L, R \rbrace$$
At full link ($\alpha_{\mathrm{link}} = 1.0$) with a shared positive normal gain $G[n] > 0$ and normal finite inputs/products ($u = 2^{-24}$), the stereo image angle error satisfies:
$$\Vert\widehat{\mathbf{y}} - G\mathbf{x}\Vert_2 \le u\Vert G\mathbf{x}\Vert_2, \qquad |\sin(\Delta\theta)| \le \frac{u}{1 - u}, \qquad |\Delta\theta| < 0.00000342^\circ$$
Audio-rate ballistic envelope smoothing prepares $\alpha_{\mathrm{att}} = 1 - \exp(-1/(f_s\tau_{\mathrm{att}}))$ and $\alpha_{\mathrm{rel}} = 1 - \exp(-1/(f_s\tau_{\mathrm{rel}}))$ on the cold path and updates $v_{\mathrm{env},c}[n] = v_{\mathrm{env},c}[n-1] + \mathrm{select}(\Delta e_c[n] > 0, \alpha_{\mathrm{att}}, \alpha_{\mathrm{rel}})\Delta e_c[n]$ branchlessly.

### 2.5 4-Band Asymmetric Ballistic Envelope Bank & Aux Key Detector

`r[dsp.router.ballistic-envelope-bank]`

| Band ID | Frequency Range | Ballistic Attack (`τ_att`) | Ballistic Release (`τ_rel`) | Physical Circuit Role |
| :--- | :--- | :---: | :---: | :--- |
| **Band 0 (Low)** | `< 250 Hz` | `10.0 ms` | `150.0 ms` | Power supply B+ rail sag, cone excursion pumping, transformer saturation. |
| **Band 1 (Mid)** | `250–2000 Hz` | `2.0 ms` | `40.0 ms` | Dynamic auto-wah envelope, speech formant modulation, optical compression. |
| **Band 2 (High)** | `> 2000 Hz` | `0.2 ms` | `15.0 ms` | Pick transient attack detection, noise gate onset, sizzle ducking. |
| **Band 3 (Aux Key)** | Wideband / Sidechain | `0.5 ms` | `50.0 ms` | Driven from `Port In 3 (Aux Key)` or stereo-linked instrument sum. |

---

## 3. Stage I / Top-of-Frame Causal Crossbar (`Op-P`, `Q0.32` Carriers & `Op-T` Reads)

`r[dsp.stage1.pre-bus]`

| Crossbar Kernel | State Allocation | Operational Modes & Physical Roles |
| :--- | :--- | :--- |
| **`Op-T[Crossbar]` (Causal Reads, `tau >= 4`)** | Inline `SonaHotSlab` `fp16` ring (`Quad 0`) + cold power-of-two `ext_lines` (`a in {2, 4}`, [`op_t_transport.md` §1.1](op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)) | • **`M = 1`:** Standard delay, chorus, TZF & `0-ULP` freeze hold.<br/>• **`M = 2`:** Centered Hermite-Hann pitch grains (`±24 st`, Family 3).<br/>• **`M = 3`:** Endless barberpole sweep (Family 6). Feeds into `SONA` core (`Reverb -> Fuzz`, `Chorus -> Amp`). |
| **`Op-P` (Input-Only `Hilbert` + `T1`)** | `HilbertH1Lattice6Sec` + pure-`f32` `T1SubHarmonicTracker` (`dnam-kernels`, [`op_m_modal_kautz.md` §2–§3](op_m_modal_kautz.md#2-family-2-hilbert-ssb-lattice--carrier-modulation)) | • **`M-1`:** 6-section `H1` all-pass pair for SSB / analytic mixing.<br/>• **`M-2`:** `T1` sub-octave divider (`f_0/2`, `f_0/4`, half-angle cos/sin).<br/>Precomputable once on CPU (`< 0.5 ms`) before WebGPU batch upload. |
| **`4× Q0.32` Quadrature Phasors** | `4`-lane `u32` wrapping phase + degree-5 Hermite quadrature (`0.05 Hz..20 kHz`) | Slow LFOs (chorus/rotary), dual polyrhythmic tempo sync (`REQ-2.5`), and audio-rate AM / ring-modulation carriers (`220 Hz` bilinear cross-products). |
| **`Op-N[Pre]` (Dynamics & ZOH)** | Fixed-hertz ZOH quantizer (`0 FDIV`, [`op_n_commuted_cascade.md` §2.3](op_n_commuted_cascade.md#23-family-5-precomputed-reciprocal-zoh-sample-rate-reducer--bitcrusher)) | Hysteretic gate, auto-swell, ZOH sample-rate reducer + nearest-step bitcrusher (`Family 5`), and pre-core linear mix into `SONA` Block 0 and direct `VarPro` readouts. |

On every stereo frame, the Top-of-Frame Causal Operator Crossbar executes unconditionally before the 4-block `SONA` core (`0` stage-bypass flags), combining dry router inputs, causal `Op-T` tap reads ($\tau \ge 4.0$), `Op-P` sub-harmonic/Hilbert channels, and `4× Q0.32` quadrature carriers (`0.05 Hz..20 kHz`) via learned mixing weights into `SONA` Block 0 and the direct-register `VarPro` readout accumulators (§7).

---

## 4. Stage II: `4`-Block Matryoshka `SONA` Core (`TPT-SVF` $\to$ `ParaLRU` $\to$ `PolyADAA1`)

`r[dsp.stage2.commuted-cascade]`

Stage II retires coupled `GRU` (`stage2_gru`) and models vacuum-tube preamps, asymmetric drives, `LR4` / multi-band crossovers, diode/optical dynamics, phasers, and power-amp sag via the **Unconditional `4`-Block Matryoshka `SONA` State-Space Core** (`Block 0 -> Block 1 -> Block 2 -> Block 3`, each composing `TPT-SVF -> 1D Scalar ParaLRU -> Pure-f32 Divided-Difference PolyADAA1` with lower-triangular `4×4` inter-block spatial mixing; [`op_n_commuted_cascade.md` §3–§4](op_n_commuted_cascade.md#3-exact-lti-basis-commutativity-theorem-theorem-1)):

1. **Compile-Time `const` `SonaTierSpec` Shapes (Locked at `Gate 0b`, `0` Capture-Time Tier Resizing):**
   - **`LiteSpec` (`[1, 1, 1, 1]`):** `16` lanes + `4` `TPT-SVF` modes @ `1×` `PolyADAA1` (or `[1, 1, 1, 0]` @ `2×` OS locked at `Gate 0b` inside `<= 1,000 ns` A76 / `<= 4.5 KiB` `SonaHotSlab`).
   - **`FullSpec` (`[2, 2, 1, 1]`):** `24` lanes + `6` `TPT-SVF` modes @ `2×` OS (`96 kHz`) + `PolyADAA1` (`<= 7.8 KiB`).
   - **`ProSpec` (`[3, 2, 2, 1]`):** `32` lanes + `8` `TPT-SVF` modes @ `2×` OS (`96 kHz`) + `PolyADAA1` (`<= 15.5 KiB`).
   - **`UltraSpec` (`[4, 3, 2, 1]`):** `40` lanes + `10–12` `TPT-SVF` modes @ `2×` OS (`4×` offline bounce) + `PolyADAA1` (`<= 30.0 KiB`).
2. **`process_tick16_const` `16`-Frame Register Hoisting & `StereoVec8` / `2× SimdVec4` `0-ULP` Parity:** `SonaInstance` executes `16` host frames (`32` @ `2×` OS) per sub-block loop (`process_tick16_const`), pinning `TPT-SVF` `(a1, a2, a3)`, `ParaLRU` `(alpha_rel, alpha_atk, kappa)`, `FIR`, and `KautzModalBank<4>` constants inside `StereoVec8` (`AVX2+FMA`, `SVE2`, `SME`) or interleaved `2× SimdVec4` (`NEON`, `WASM SIMD128`) registers with `0` stack spills and `0-ULP` parity.
3. **Pure-`f32` `PolyADAA1`, `TPT-SVF` Passivity & Zero-DC Equilibrium:** Every block evaluates passive `2×2` ZDF `TPT-SVF` modes ($\Vert\mathbf{A}_{\text{tpt}}\Vert_2 \le 1$), strictly contractive scalar `ParaLRU` ($J_i[n] \in (0, r_{\max}]$), and fast-path `PolyADAA1` + batched `rcp_pos_quad_nr2` with analytical `0.000000` DC equilibrium (`h_eq == +0.0`), while linear `pre_cab` and `direct` `VarPro` readouts carry unclipped audio across `±100.0` (`+40 dBFS`, `REQ-7.1`).

---

## 5. Stage III: Contiguous SIMD Early FIR (`32..128` Taps) + Shared Kautz Modal Bank (`8..32` Pairs)

`r[dsp.stage3.cabinet-bank]`

Stage III executes between the `SONA` core and end-of-frame `Op-T` writes, combining independent L/R early FIR segments (`32 / 48 / 96 / 128` taps/ch across `Lite / Full / Pro / Ultra`) with shared-pole `KautzModalBank<4>` pairs (`8 / 12 / 24 / 32` pairs = `16 / 24 / 48 / 64` real poles, `2 / 3 / 6 / 8` 4-mode blocks, driven at $x_a[n-16]$ at `48 kHz`) and independent L/R complex readouts hoisted across `16`-frame sub-blocks inside `process_tick16_const` ([`op_m_modal_kautz.md` §4](op_m_modal_kautz.md#4-stage-iii--family-6-shared-kautz-modal-bank)).

`r[dsp.stage3.mirrored-fir-simd]` **Double-Mirrored Reverse-Order FIR History & Contiguous `dot16_stereo` Evaluation ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)):** To eliminate per-tap modulo index wrapping (`(pos - k) & 127`) and scalar gather loads across up to `128`-tap early FIR filters, each channel maintains a double-mirrored reverse-order circular buffer `fir_rev_history: [f32; 256]` where each incoming sample writes to both `rev` and `rev + 128` for `rev = (128 - (self.fir_write_ptr & 127)) & 127`. Because `[rev + tap_offset .. rev + tap_offset + 16]` is always contiguous in memory for every 16-tap slice (`tap_offset ∈ {0, 16, .., 112}`), `Stage3Backend::dot16_stereo` loads contiguous `B::Vec4` / `StereoVec8` vectors hoisted across `16`-frame sub-blocks and reduces Left and Right dot products simultaneously via `SimdVec4::horizontal_sum_pair`.

---

## 6. Stage IV: Coercive `Op-T` Spatial & Feedback Loop (`Op-T[Tank]`)

`r[dsp.stage4.spatial-tank]`

| Stage IV Component | Architectural & Mathematical Specification |
| :--- | :--- |
| **Physical Structure & Causal Feedback** | `M` delay lines (`1` quad / `M = 4` Lite, `2` quads / `M = 8` Full, `3–4` quads / `M = 12–16` Pro, `4` quads / `M = 16` Ultra) read in the Top-of-Frame Crossbar (`tau >= 4`) and written back at frame end with inline `SonaHotSlab` `fp16` (`Quad 0`) or cold power-of-two `ext_lines` (`<= 48,000` ref samples @ `48 kHz` + `4×` bounce headroom, `a in {2, 4}`, `O(1)` `& ext_mask`), `Q0.32` degree-5 Hermite quadrature LFO + convex-blended 4-tap Catmull-Rom fractional reads, and coercive bounded-history feedback guard ([Op-T §1](op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)). |
| **Partitioned Schroeder All-Pass Diffusers** | Non-overlapping in-buffer segments (`PartitionedSchroederDiffuserBank`, scaling to `8` stages on `Ultra`) with rate-converted integer lengths and `∣a_diff,m∣ ≤ 0.75` ([Op-T §3](op_t_transport.md#31-memory-partitioning--rate-aware-capacity)). |
| **`O(M)` Givens + Householder Mixing `U(ψ, v)`** | Block-diagonal `2×2` Givens rotation `R(ψ)` (`∣Δψ∣ ≤ 0.05 rad/tick`) paired with unit-norm Householder reflection `H(v) = I_M - 2vv^T` (`‖U^T U - I‖_∞ < 1e-14` in `f64`, `< 1e-6` in `f32` across `M ∈ {4, 8, 12, 16}`). |
| **Input & Readout DC Conditioning** | Passband-normalized `5 Hz` DC blocker (`0x3F7FD51F` @ `48 kHz`) applied at loop input injection and non-recirculating loop readouts (§6.2). |
| **Blended Feedback Knee & Loop Saturator** | Per-line blended knee `K(u)` (`g_lin ≤ 0.9990`, §6.4) followed by unity-slope cubic saturator (`V_ceil,m ≤ 0.9850`, §6.3, `‖G_loop‖ < 1`). |

**`Q0.32` Degree-5 Hermite Quadrature LFO Derivation (`ref[repo.code.constant-eval-and-derivations]`):** On each quadrant $u \in [0, 1]$, the unique odd-degree polynomial $P(u) = c_1 u - c_3 u^3 + c_5 u^5$ satisfying the quarter-sine $C^1$ boundary conditions $P(0) = 0,\ P'(0) = \pi/2,\ P''(0) = 0,\ P(1) = 1,\ P'(1) = 0$ solves to $c_1 = \pi/2,\ c_3 = \pi - 5/2,\ c_5 = \pi/2 - 3/2$ via compile-time `const_hermite5_quarter_sine_coeffs()`. For long-delay models (`> 1,024` frames), `ext_lines` is sized and allocated solely in cold container loading so all RT steps wrap solely via `& ext_mask` (`0 UDIV`/`FDIV`/`FSQRT`/`libm`).

### 6.1 Captured Freeze and Modeled Controls

`r[dsp.stage4.captured-freeze]`

Hardware freeze is modeled when present and captured in the training stems ([`capture_and_alignment.md` §1.3](../training/capture_and_alignment.md#freeze-transition-capture)); unobserved freeze options remain clamped under `ref[train.audit.phase1.clamp-switch]`. When a captured hardware freeze (`freeze == true`) locks ring writes, `Op-T` stops all new delay-ring overwrites and holds integer-lag reads at **`0 ULP` with zero `fp16` re-quantization hiss buildup**, while the Top-of-Frame Crossbar, `SONA` core, Stage III, and non-recirculating readouts continue processing live play-over instrument audio cleanly without damping the held loop state.

### 6.2 Stage IV Passband-Normalized DC Blocker

`r[dsp.stage4.dc-blocker]`

For pole $0 \le R_{\text{dc}} < 1$, the passband-normalized DC blocker is applied at `Op-T` loop input injection and non-recirculating readouts:
$$H_{\text{dc}}(z) = \frac{\alpha_{\text{dc}}(1 - z^{-1})}{1 - R_{\text{dc}}z^{-1}}, \qquad \alpha_{\text{dc}} = \frac{1 + R_{\text{dc}}}{2}, \qquad R_{\text{dc,ref}}(48\text{ kHz}) = \frac{\cos(2\pi \cdot 5 / 48000)}{1 + \sin(2\pi \cdot 5 / 48000)} \approx \mathtt{0x3F7FD51F}$$
At other effective rates, convert $R_{\text{dc}}(f_{s,\text{eff}}) = R_{\text{dc,ref}}^{\rho_{\text{sr}}}$ on the cold path (§9).

### 6.3 Stage IV Normalized Loop Saturator

`r[dsp.stage4.loop-saturator]`

For finite ceiling $0 < V_{\text{ceil}} \le 0.985$ (`≤ 0x3F7C28F5` in `f32`, since `0x3F7C28F6 ≈ 0.9850000143` exceeds `0.985`), the unity-slope memoryless cubic saturator is:
$$v = \mathrm{clamp}\left(\frac{2u}{3V_{\text{ceil}}},\ -1,\ 1\right), \qquad S(u) = \mathrm{clamp}\Big(V_{\text{ceil}} v(1.5 - 0.5 v^2),\ -V_{\text{ceil}},\ V_{\text{ceil}}\Big)$$
The explicit outer clamp to $\pm V_{\text{ceil}}$ prevents `f32` intermediate rounding (e.g., at `v = 0x3F7FF4D1`) from exceeding `0x3F7C28F5`.

### 6.4 Stage IV Blended Quiet-Feedback Knee

`r[dsp.stage4.quiet-feedback-knee]`

Each loop line applies the blended pre-saturator feedback knee with quiet slope $0 \le g_{\text{lin}} \le 0.9990$ (`≤ 0x3F7FBE76` in `f32`, since `0x3F7FBE77` exceeds `0.9990`), blend factor $\alpha \in [0, 1]$, and knee width $\sigma > 0$:
$$K(u) = g_{\text{lin}} u \left((1 - \alpha) + \frac{\alpha}{1 + (u/\sigma)^2}\right), \qquad K'(0) = g_{\text{lin}}, \qquad |K(u)| \le g_{\text{lin}}|u|$$
At $\alpha = 0$, $K(u) = g_{\text{lin}} u$ is purely linear; at $\alpha = 1$, $K(u)$ is the rational knee. Evaluate $(1 + (u/\sigma)^2)^{-1}$ without audio-loop division via cold-path prepared $1/\sigma$ and 6 exponent-seeded `f64` Newton reciprocal steps (`pre-rounding relative error < 2^-40`).

`r[dsp.stage4.simd-tank-loop]` **4-Lane SIMD Loop Conditioning, Saturation & Orthogonal Mixing ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)):** `Stage4SpatialTank::step_stereo_with_readouts` vectorizes the 4-line feedback loop in `B::Vec4` SIMD registers:
1. **4-Lane Passband-Normalized DC Blocker:** Updates the 4-line `dc_x_prev` and `dc_y_prev` states in `B::Vec4` via `y = alpha_dc * (x - x_prev) + r_dc * y_prev` + `flush_recursive_state_vec4`.
2. **4-Lane Blended Knee & Cubic Loop Saturator (`coercive_saturate_4ch`):** Evaluates $(1 + (u \cdot \sigma^{-1})^2)^{-1}$ across all 4 lines in `B::Vec4` using cached `inv_sigma_knee` and 3-step `f32` Newton reciprocal refinement (with 6-step `f64` `coercive_saturate` on the single-line path), followed by 4-lane cubic saturation clamped to $\pm V_{\text{ceil}}$.
3. **4-Lane Givens Rotation & Householder Reflection:** Caches `(cos_psi, sin_psi)` whenever the slewed Givens angle is stationary (`|target_psi - current_psi| <= 1e-7`), applies the $2\times 2$ block-diagonal Givens rotation in a single 4-lane FMA via `SimdVec4::pair_swap_neg([x0, x1, x2, x3]) -> [-x1, x0, -x3, x2]`, and evaluates the 4-channel Householder reflection $y - 2(v^\top y)v$ using `SimdVec4::horizontal_sum` and `SimdVec4::fma`.

---

## 7. Unified 256-D 4-Tier Kronecker Output Head

`r[dsp.head.feature-bus-slots]`

In `SONA` `v3.0` (`ADR-0022`), the legacy `256-D` `feature_bus_scratch` scatter/gather buffer is completely eliminated (`0` `256-D` bus copies) in favor of **zero-copy direct-register `VarPro` linear readouts** (`pre_cab` and `direct` plus `fp16` `W_readout` subspace in `Arena 0..3`) accumulated inline in SIMD vector registers alongside the compile-time **`const` `SonaTierSpec`**, contiguous Stage III FIR/Kautz slices (`32 / 48 / 96 / 128` FIR taps and `8 / 12 / 24 / 32` Kautz pairs), and `Op-T` quads (`1 / 2 / 3–4 / 4` quads) inside a zero-spill **`SonaHotSlab`**:

| Tier (`const` `SonaTierSpec`) | `4`-Block Matryoshka `SONA` Core (`TPT-SVF` @ Rate) | Stage III Kautz (`8-Vec SoA`) | Stage III FIR (`L/R`) | `Op-T` Quads (`M` Lines) | Target `SonaHotSlab` (`N × Slab ≤ 64 KiB` L1D) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Lite` (`LiteSpec`)** | `[1, 1, 1, 1]` (`16` lanes + `4` `SVF`) @ `1×` (or `[1,1,1,0]` @ `2×`) | `8` pairs (`2` blks = `256 B`) | `32` taps (`256 B`) | `1` quad (`M = 4` inline `fp16` + cold `ext`) | **`≤ 4.5 KiB`** (`14× ≤ 63 KiB`) |
| **`Full` (`FullSpec`)** | `[2, 2, 1, 1]` (`24` lanes + `6` `SVF`) @ `2×` OS | `12` pairs (`3` blks = `384 B`) | `48` taps (`384 B`) | `2` quads (`M = 8` + Schroeder + `Q0` inline) | **`≤ 7.8 KiB`** (`8× ≤ 62.4 KiB`) |
| **`Pro` (`ProSpec`)** | `[3, 2, 2, 1]` (`32` lanes + `8` `SVF`) @ `2×` OS | `24` pairs (`6` blks = `768 B`) | `96` taps (`768 B`) | `3–4` quads (`M = 12–16` + Schroeder, `fp16∣f32`) | **`≤ 15.5 KiB`** (`4× ≤ 62 KiB`) |
| **`Ultra` (`UltraSpec`)** | `[4, 3, 2, 1]` (`40` lanes + `10–12` `SVF`) @ `2×` (`4×` Bounce) | `32` pairs (`8` blks = `1,024 B`) | `128` taps (`1,024 B`) | `4` quads (`M = 16` + `8`-stage Schroeder, `fp16∣f32`) | **`≤ 30.0 KiB`** (`2× ≤ 60 KiB`) |

`r[dsp.head.tier-equivalence]` **Effective-Tier Equivalence:** For identical external inputs, control values, initial active-tier state, and backend, a container stripped to tier `T` produces bit-identical `SONA` core trajectories, L/R FIR histories, and retained readout outputs to the unstripped container running at effective tier `T` (with all tier `> T` weights and `Ultra` control extensions gated off, while retaining full `Arena 0` parametric control conditioning).

#### Block-Diagonal Lower-Triangular Staging (Refinement 11)

`r[dsp.head.lower-triangular]`

```text
W_head = [ W_lite           0             0             0           ]
         [ W_full,lite     W_full,delta  0             0           ]
         [ W_pro,lite      W_pro,full    W_pro,delta   0           ]
         [ W_ultra,lite    W_ultra,full  W_ultra,pro   W_ultra,delta ]
```

`r[dsp.head.precollapse-and-zerocopy]` **Contiguous `SonaHotSlab`, `process_tick16_const` Register Hoisting & `3 kHz` `VarPro` Readout Collapse ([`ADR-0019`](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):**
1. **Contiguous Zero-Spill `SonaHotSlab` (`[0..ActiveHotBytes)`) + `0 B` Transient Scratch:** Hot `48 kHz` audio weights and states are packed contiguously in strict execution order at `[0..ActiveHotBytes)` inside a tier-exact `64 B`-aligned `SonaHotSlab` (`<= 4.5 / 7.8 / 15.5 / 30.0 KiB`, `0 B` per-instance transient heap scratch, audited via release disassembly to eliminate stack spills prior to `Gate 0b`), while `P`-sized active controls + `ControlBusWire` + `fp16` `W_readout` (`VarProReadoutSubspaceWire`) reside at the cold tail.
2. **`process_tick16_const` `16`-Frame Sub-Block Register Hoisting & `3 kHz` Collapse:** Audio callbacks run `process_tick16_const` over `16` host stereo frames (`32` @ `2×` OS), hoisting `TPT-SVF` `(a1, a2, a3)`, `ParaLRU` `(alpha_rel, alpha_atk, kappa)`, `FIR` taps, and `KautzModalBank<4>` constants/states in SIMD registers. At `3 kHz` boundaries where `is_dirty` or dynamic modulation updates latents, `ControlBus` collapses `TPT-SVF` prewarped coeffs, `ParaLRU` parameters, and `fp16` `W_readout` into `pre_cab` / `direct` `VarPro` linear registers (`0` `256-D` bus scatter/gather, `0` extra `48 kHz` MACs when controls are static).
3. **Passive Post-Stage-3 4-Band Partition-of-Unity Continuity:** When passive post-Stage-3 4-band log-octave equalization is active (`f_sub = 250 Hz` 1-pole TPT + `f_mid = 1,800 Hz`, `k = 1.25` Cytomic SVF derived via `const_sin_cos_f64` and `tan_prewarp_f32` + `recip_nr3_pos_f32`), its components `b_sub = lp_sub`, `b_lm = v2 - lp_sub`, `b_pres = k·v1`, and `b_hi = x - k·v1 - v2` satisfy `b_sub + b_lm + b_pres + b_hi ≡ x` (`0-ULP` bypass at `StarCenter`, bounded passive BIBO gain in `[-4.5, +3.0]` octaves).

---

## 8. Tier-Aware Static & Dynamic Modulation Routing

### 8.1 Bus Architecture

`r[dsp.router.modulation-bus]`

$$\mathbf{q}_{16}[k] = \big[\mathbf{z}_{\text{ctrl}}^{(8)}[k], \ \mathbf{m}_{\text{dyn}}^{(8)}[k]\big]^\top \in \mathbb{R}^{16}$$

1. **Static Control Latents:** `Lite/Full/Pro` expose `4` head + `4` stage coordinates (`Rank-4` FM) self-contained in `Arena 0`; `Ultra` adds `+4` head (`modes 5..8`) + `+4` stage coordinates (`Rank-8` FM) and a `16 × 4` stage-dispatch delta $\mathbf{W}_{\text{ctrl},\Delta}^{\text{Ultra}}$ in `Arena 3` ([`parametric_control_bus.md` §3.1](parametric_control_bus.md#31-stage-parameter-dispatch-matrix-mathbfw_textctrl)).
2. **Autonomous Dynamic Modulators $\mathbf{m}_{\text{dyn}} \in \mathbb{R}^8$ (`float32x4_t[2]`):**
   - $m_0 = z_{\mathrm{re}, \text{lfo0}}[k]$, $m_1 = z_{\mathrm{im}, \text{lfo0}}[k]$ (Quadrature LFO 0 In-Phase & $90^\circ$ rotors).
   - $m_2 = T_{\text{shape}}(z_{\mathrm{re}, \text{lfo0}}[k])$, $m_3 = T_{\text{shape}}(z_{\mathrm{re}, \text{lfo1}}[k])$ (Chebyshev shaped LFO 0 & 1).
   - $m_4 = e_{\text{low}}[k]$ ($< 250\text{ Hz}$), $m_5 = e_{\text{mid}}[k]$ ($250\text{–}2000\text{ Hz}$), $m_6 = e_{\text{high}}[k]$ ($> 2000\text{ Hz}$), $m_7 = e_{\text{aux}}[k]$ (`Port In 3` Aux Key).

---

## 9. Continuous-Time Zero-Order Hold (ZOH) Sample-Rate Adaptability (44.1–96 kHz & 4× Bounce)

Containers store discrete-time parameters anchored at $f_{s,\text{train}} = 48,000\text{ Hz}$. For host rate $f_{s,\text{host}} \in [44,100, 96,000]\text{ Hz}$ and oversampling factor $M_{\text{os}} \in \lbrace 1, 2, 4 \rbrace$, let $f_{s,\text{eff}} = M_{\text{os}} f_{s,\text{host}}$ (`44.1..384 kHz`, with `2×` OS active on `Full..Ultra` and evaluated on `Lite` at `Gate 0b`, promoting to `4×` on offline bounce) and $\rho_{\text{sr}} = 48000 / f_{s,\text{eff}}$:

1. `r[dsp.zoh.op-m-poles]` **Stage I (`Op-M[Pre]`), Core `TPT-SVF` Prewarping & Stage III (`Op-M[Modal]`) Conversion:**
   $$g_{\text{host}} = \tan\left(\frac{\pi f_c}{f_{s,\text{eff}}}\right), \qquad r_c(f_{s,\text{eff}}) = \big(r_{\text{ref},c}\big)^{\rho_{\text{sr}}} \in \big[0.25^{\rho_{\text{sr}}},\ 0.9995^{\rho_{\text{sr}}}\big], \qquad \theta_c(f_{s,\text{eff}}) = \mathrm{clamp}\Big(\rho_{\text{sr}} \theta_c(f_{s,\text{train}}),\ 10^{-4},\ \pi - 10^{-4}\Big)$$
   At $f_{s,\text{eff}} = 48,000\text{ Hz}$, use the validated serialized coefficients directly. At each `3 kHz` control boundary, `TPT-SVF` (`theta_5, theta_6`), `Op-M` (`theta_2..4`), and Stage III (`theta_9, theta_10`) dynamically retune their prewarped filter gains, delays, and Kautz modal angles/gates without division or `libm`.
2. `r[dsp.zoh.router-detector]` **Stage 0 Relative Alignment Latency & `Op-M[Ctrl]` Detector Poles:**
   $$\delta_{\text{align},ch}^{\text{samples}}(f_{s,\text{eff}}) = \frac{\delta_{\text{align},ch}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{sr}}}, \qquad p_{\text{det},b}(f_{s,\text{eff}}) = \big(p_{\text{det},b}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
3. `r[dsp.zoh.stage2-op-n]` **4-Block `SONA` Core (`TPT-SVF` Analytical Prewarping & `ParaLRU` Retention) & Family 5 ZOH Phase (`0.999995` Upper Clamp):**
   $$\alpha_{\text{rel},i}(f_{s,\text{eff}}) = \big(\alpha_{\text{rel},i,\text{ref}}\big)^{\rho_{\text{sr}}}, \qquad \alpha_{\text{atk},i}(f_{s,\text{eff}}) = \big(\alpha_{\text{atk},i,\text{ref}}\big)^{\rho_{\text{sr}}}, \qquad R_{\text{ac}}(f_{s,\text{eff}}) = \big(R_{\text{ac}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
   Inside the `SONA` core, `TPT-SVF` modes transfer via analytical bilinear ZOH frequency prewarping ($g_{\text{host}} = \tan(\pi f_c / f_{s,\text{eff}})$), `Full..Ultra` execute at `2×` OS (`4×` on offline bounce) with causal `poly3` decimation on `1×` outputs, and `Lite` runs at `1×` (`PolyADAA1`) or `2×` per the `Gate 0b` lock. Real poles rescaled via `rescale_real_pole` clamp to $[0.0, 0.999995]$ ($Q \approx 100{,}000$), preserving sub-`5 Hz` AC blockers through `384 kHz`, alongside $\Delta s_{\text{re}}[n] = \rho_{\text{sr}}^{-1}(s_{\text{re}}[n] - s_{\text{re}}[n-1])$.
4. `r[dsp.zoh.op-t-delays]` **`Op-T` Delays, All-Pass Diffusers, Crossfades & DC Blocker:**
   $$\tau_{0,m}^{\text{samples}}(f_{s,\text{eff}}) = \max\left(4.0,\ \frac{\tau_{0,m}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right), \qquad D_{\text{ap},m}(f_{s,\text{eff}}) = \max\left(1,\ \mathrm{round}\left(\frac{D_{\text{ap},m}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)\right)$$
   $$L_{\text{xfade}}(f_{s,\text{eff}}) = \max\left(16,\ \mathrm{round}\left(\frac{L_{\text{xfade}}(f_{s,\text{train}})}{\rho_{\text{sr}}}\right)\right), \qquad R_{\text{dc}}(f_{s,\text{eff}}) = \big(R_{\text{dc}}(f_{s,\text{train}})\big)^{\rho_{\text{sr}}}$$
5. `r[dsp.zoh.ctrl-stride]` **Control-Bus Decimation Stride $K_{\text{ctrl}}$, Tick Ratio $\rho_{\text{ctrl}}$ & Micro-Ramp Step $\alpha_{\text{ramp}}$:**
   $$K_{\text{ctrl}}(f_{s,\text{eff}}) = \max\left(8,\ \mathrm{round}\left(\frac{f_{s,\text{eff}}}{3000.0}\right)\right), \qquad \rho_{\text{ctrl}} = \frac{f_{s,\text{train}} \cdot K_{\text{ctrl}}(f_{s,\text{eff}})}{f_{s,\text{eff}} \cdot K_{\text{ctrl}}(f_{s,\text{train}})}, \qquad \alpha_{\text{ramp}}(f_{s,\text{eff}}) = \frac{1.0}{K_{\text{ctrl}}(f_{s,\text{eff}})}$$
   $$\Delta\phi_{\text{lfo},j}(f_{s,\text{eff}}) = \rho_{\text{ctrl}} \Delta\phi_{\text{lfo},j}(f_{s,\text{train}}), \qquad R_{\text{env},b}(f_{s,\text{eff}}) = \big(R_{\text{env},b}(f_{s,\text{train}})\big)^{\rho_{\text{ctrl}}}$$
6. `r[dsp.zoh.stage3-fir-onset]` **Stage III FIR & Modal Onset Conversion (`Up to 8 Host Samples` Budget):**
   - **Integer Multiples ($f_{s,\text{eff}} = q \cdot 48000, q \in \lbrace 1, 2, 4, 8 \rbrace$):** Evaluate exact sparse FIR delays $x_a[n - q(16t + j)]$ with integer modal onset $16q$ (`0` added latency; [`op_m_modal_kautz.md` §4.1](op_m_modal_kautz.md#41-stage-iii-architecture--tier-allocation)).
   - **Live `44.1–48 kHz` First-Tap Continuous Difference Family:** With $\lambda = (48000 - f)/3900$ and $\widetilde p_j = \mathrm{RN}_{32}(\lambda p_j)$, evaluate $x[n-1] + \sum_{j=0}^{62} \widetilde p_j (x[n-j] - x[n-j-1])$ (`< 0.0177` ideal complex error through `20 kHz`, exact unit DC gain on constant normal histories).
   - **`4×` Bounce (`176.4–384 kHz`) Six-Point Lagrange Difference Mapping:** For tap $k \ge 1$ with exact integer decomposition $d = k f / 48000 = m + \mu$ ($m = \lfloor d \rfloor \ge 3$, $\mu = r/D$), anchor at $a = 2$ ($\mu \le 1/2$) or $a = 3$ ($\mu > 1/2$) with residual coefficients $p_j = \mathrm{RN}_{32}(\sum_{i=0}^j h_i(\mu) - \mathbf{1}_{j \ge a})$:
     $$y_{\text{tap},k}[n] = x[n - m - a + 2] + \sum_{j=0}^4 p_j \big(x[n - m + 2 - j] - x[n - m + 1 - j]\big)$$
     With integer scales $c_i = (-1, 5, -10, 10, -5, 1)$ at nodes $v_i = (-2, -1, 0, 1, 2, 3)$ and exact integer rate ratio $f = M 2^{e-52}$, $D = 48000 \cdot 2^{52-e}$, $k M = m D + r$:
     $$Q_j = \sum_{i=0}^j c_i \prod_{\ell \ne i}(r - v_\ell D) - 120 D^5 \mathbf{1}_{j \ge a}, \qquad p_j = \mathrm{RN}_{32}\left(\frac{Q_j}{120 D^5}\right)$$
     This causal 6-point kernel has ideal complex error $< 0.00064$ through `20 kHz` ($(5/1024)(2\pi \cdot 20000/176400)^6 + 2^{-23} + 10 \cdot 2^{-150} < 0.00064$), full-band gain $< 1.391$, and max ring lookback `251 / 379 / 763 / 1,019` samples/ch (`252 / 380 / 764 / 1,020` history slots across `32 / 48 / 96 / 128` taps/ch for `Lite..Ultra`). Other non-integer effective rates fail fast.
