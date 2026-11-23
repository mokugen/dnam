---
id: spec.dsp.router
title: "Pure 2×2 VM Boundary, SensoryQuad, Causal Crossbar, SONA Signal Graph & ZOH Rate Adaptation"
status: canonical
version: "3.0.0"
domain: dsp
rule_prefixes:
  - "r[dsp.router."
  - "r[dsp.crossbar."
  - "r[dsp.core."
  - "r[dsp.cabinet."
  - "r[dsp.tank."
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

# Pure 2×2 VM Boundary, `SensoryQuad`, Causal Crossbar, SONA Signal Graph & ZOH Rate Adaptation

This specification defines the clean boundary between the outer Host-Shell 8-Port Patchbay (4CM, WDW, parallel latency alignment, M/S output width) and the pure 2-in / 2-out DSP virtual machine: learned 4-lane `SensoryQuad` (`Op-M/N[Sense]`), Top-of-Frame Causal Operator Crossbar, and the 4-Block Matryoshka `CoreQuad` stack executing in 4-frame causal micro-chunks inside 16-frame register-hoisted sub-blocks over a cache-compact active-hot working set (4,544 to 14,592 B inside 4,608 to 28,672 B L1D ceilings across the four tier profiles).

It also specifies `ModalConvolver` (`modal_convolver` FIR + Shared Kautz Bank), symmetric coercive Op-T `Transport` (`transport`), direct-register `LinearReadout` (`ro_pole_48k` tilt/shelf basis), and continuous-time Zero-Order Hold (ZOH) sample-rate adaptation across 44.1–384 kHz.

For architectural background and derivations, see [sona_architecture.md](../architecture/sona_architecture.md), [REQ-7.4](../../requirements/system_goals_and_requirements.md), [ADR-0006](../../adr/0006-model-declared-delays-and-tank-feedback-guard.md), [ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), and [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md).

---

## 1. Signal Graph Overview

`r[dsp.router.signal-flow]`

```mermaid
flowchart TD
    subgraph ControlPlaneGroup ["3 kHz Unified Control Plane (Arena 0 Self-Contained + Optional CEXT + Ultra Delta)"]
        CtrlSrc["Arena 0 Descriptors, CP1, U_int, W_ctrl, CEXT & Static W_ro,0"] --> OpN["Op-N[Ctrl]: Exact-Integer Sufficient Statistics<br/>+ ou_to_z_stage Drift, z_stage Sensitivities & Static W_ro,0"]
        EnvLfo["SensoryQuad e[0..4] (q3) + 2x Q0.192 Skewed LFOs (q2)"] --> CtrlDispatch["Unified 32-Row Control Dispatch (q16 -> stage_targets[0..32])<br/>16x16 W_ctrl (+ CEXT + Ultra Deltas) & Uniform 32-Slot Micro-Ramp"]
        OpN --> CtrlDispatch
    end

    In0["Port In 0: Inst In L"] & In1["Port In 1: Inst In R"] --> PreRouter["Host-Shell Patchbay: Send & Normaling<br/>(Outside Pure 2x2 VM; Chained VM Instances)"]
    In2["Port In 2: FX Return L"] & In3["Port In 3: FX Return R / Aux Key"] --> DownRouter["Host-Shell Patchbay: Return & Align<br/>(Parallel Alignment & 4CM/WDW Routing)"]

    PreRouter --> Out0["Port Out 0: Send L<br/>(External Amp 1)"]
    PreRouter -->|"Dual-Amp Modes"| Out1["Port Out 1: Send R / WDW Dry"]
    DryTap["WDW Dry Selector<br/>(PreSpatialCore / DirectDry)"] -->|"WDW"| Out1

    PreRouter & DownRouter -->|"2x2 Stereo L/R"| Sense["SensoryQuad Op-M/N[Sense] @ f_s,vm<br/>• 4x TPT-SVF + Full-Wave ParaLRU -> q3<br/>• Smooth Input VCA g_in (Zero LR4/Gate Heuristic)"]
    Sense -.-> EnvLfo
    Sense -->|"g_in * (x_L, x_R)"| Crossbar["Crossbar & Op-N[Pre] @ f_s,vm<br/>• Op-T Read v_read (tau_min = 5.0) + Stereo Op-P (H1 + 2-Stage T1)<br/>• s0 = g_in * x + op_t_gain * sum(opt_to_lr * v_read)<br/>• Smooth Op-N[Pre] (Raised-Cosine ZOH + Fourier Q) -> u_1x"]
    Crossbar -->|"OS2X Up if os2x_active"| Core["4-Block CoreQuads @ f_s,core<br/>• PhasorQ32Bank4 (inc_core = inc_vm >> 1 under OS2X)<br/>• M/S Input, TPT-SVF -> Selective-Duhem ParaLRU -> PolyADAA1 + Wavefolder + GLU<br/>• hp_lp(ro_pole_48k) -> (pre_conv, direct_s2)"]
    Core -->|"OS2X Down -> pre_conv @ f_s,vm"| Conv["ModalConvolver Early FIR + Kautz @ f_s,vm -> conv_out"]
    Core -->|"OS2X Down -> direct_s2 @ f_s,vm"| Tank["Symmetric Pure-Wet Transport @ f_s,vm<br/>• x_tank = (direct_s2_l + conv_l, direct_s2_r + conv_r)<br/>• Stage tank_dry_mix + P_Q Wet Quads (sigma_sat + freeze_mix) -> Xfade -> eff_trim -> +/-1.0e9 Safety"]
    Conv -->|"conv_out @ f_s,vm"| Tank

    CtrlDispatch --> Sense & Crossbar & Core & Conv & Tank

    Tank --> DirectOut["Post-VM Host-Shell Patchbay<br/>• Host M/S Width & Multi-Port Routing"]
    Tank -. "z^-tau (tau_min = 5.0)" .-> Crossbar
    DirectOut --> Out2["Port Out 2: Main L / WDW Wet L"]
    DirectOut --> Out3["Port Out 3: Main R / WDW Wet R"]
```

SONA executes a two-rate pure 2-in / 2-out virtual machine loop separating perimeter stages at `f_{s,vm}` from oversampled core stages at `f_{s,core}`, wrapped externally by the host shell's multi-port patchbay and stereo M/S width control. In live real-time execution across 44.1–384 kHz, `f_{s,vm} = f_{s,host}`; in 4× offline bounce, an outer polyphase oversampler sets `f_{s,vm} = 4 * f_{s,host}` and disables internal `OS2X`.

Rate ratios follow `rho_vm = 48_000.0 / f_{s,vm}` and `K_ctrl(f_{s,vm}) = max(8, round(f_{s,vm} / 3_000.0))` (half-away-from-zero `floor(x + 0.5)`). Runtime core oversampling engages iff `os2x_active = (tier > 0) && !offline_bounce && tier_os2x[tier] && (2.0 * f_{s,vm} <= 384_000.0)`, setting `f_{s,core} = 2 * f_{s,vm}` when active (else `f_{s,vm}`) and `rho_core = 48_000.0 / f_{s,core}`. Every VM frame executes strictly in the order below:

| Step | Stage & Rate | Exact Per-VM-Frame Operation |
| :---: | :--- | :--- |
| 1 | `ControlBus` (`f_{s,vm}`) | Run `tick_3khz()` strictly at `ctrl_sub_counter == 0` (latching `SensoryQuad` `e[0..4]` into `q3` and updating `eff_z_stage = static_z_stage + ou_to_z_stage * ou_drift_state`), compute linear `delta[r]` uniformly across all slots 0..32 (incl. slots 16 and 30), advance `current_stage_params[r] += delta[r]`, and latch registers when `(sample_tick & 3) == 0` or `ctrl_sub_counter == 0` |
| 2 | `SensoryQuad` (`f_{s,vm}`) | Inside the pure 2×2 VM, mix `(x_L, x_R)` via learned `sense_in_l/r`, step 4-lane passive `TPT-SVF` + weighted `(LP, BP, HP)` mix -> full-wave `ParaLRU` rectifier `e[0..4]`, and scale input audio uniformly by smooth VCA `g_in = clamp(1.0 + delta_30 + sum(w_vca * e), 0.0, 2.0)` (`0°` stereo shift) |
| 3 | `Op-T` Read & `Op-P` (`f_{s,vm}`) | Read Top-of-Frame `Op-T` delays `v_read` at `tau_min = 5.0` VM frames and step stereo `Op-P` on gated `(g_in * x_L, g_in * x_R)` (`HilbertH1StereoKernel`, stereo 5 Hz `T1` DC, 2-stage cancellation-free mid `T1` via `rsqrt_pos_nr3`) |
| 4 | `Op-T` Crossbar Tap (`f_{s,vm}`) | Form `(s0_L, s0_R) = g_in * (x_L, x_R) + op_t_gain * sum_line(opt_to_lr * v_read)` ahead of `Op-N[Pre]` |
| 5 | Smooth `Op-N[Pre]` (`f_{s,vm}`) | Step raised-cosine clocked track-and-hold (`g_clk = (1 - z_h_eff) + z_h_eff * w(phi)`, 28 B state) -> early `gamma_bc <= 0` bypass -> smooth Fourier staircase (`q32_smooth_sin`) -> wet blend into `u_1x` ahead of `OS2X` |
| 6 | `OS2X` Upsample (`f_{s,vm} -> f_{s,core}`) | Upsample `u_1x` 2× for the 4-Block `CoreQuad` cascade only when `os2x_active` is true; otherwise bypass via `u_c = u_1x` |
| 7 | 4-Block `CoreQuad`s (`f_{s,core}`) | Step `PhasorQ32Bank4` at `inc_core = inc_vm >> 1` under `OS2X` else `inc_vm`, midpoint `Op-P` `(I_L, Q_L, I_R, Q_R)` & `T1[0..4]`, M/S per-quad coupling, `TPT-SVF`, Selective-Duhem stereo-linked `ParaLRU`, `PolyADAA1` + wavefolder + bounded `GLU`, and `hp_lp(ro_pole_48k)` readout into `(pre_conv, direct_s2)` |
| 8 | `OS2X` Downsample (`f_{s,core} -> f_{s,vm}`) | Decimate `(pre_conv, direct_s2)` back to `f_{s,vm}` when `os2x_active` is true, or pass through unchanged |
| 9 | `ModalConvolver` (`f_{s,vm}`) | Run lag-ordered `Early FIR` and shared-pole Kautz bank on `pre_conv` at `f_{s,vm}`, producing `conv_out` |
| 10 | `Transport` & Output (`f_{s,vm}`) | Feed `x_tank = (direct_s2_l + conv_l, direct_s2_r + conv_r)` into End-of-Frame symmetric `Transport` (`tank_dry_mix_eff * x_tank + sum P_Q` pure-wet quads with unified `sigma_sat` cubic limiter & continuous `freeze_mix`), apply `mono_out`, 48-frame linear tier crossfade, polarity-signed `eff_trim`, and ±1.0e9 safety ceiling; outer host shell wraps M/S width and multi-port patchbay |

---

## 2. Outer Host-Shell 8-Port Patchbay, `SensoryQuad` & Pure 2×2 VM Coupling

Multi-port hardware routing (4CM/7CM send/return matrices, `ParallelAlignLine`, `WDW` tapping, and post-VM M/S output width) resides cleanly in the outer host engine shell composing pure 2-in / 2-out VM instances, while per-instance dynamics sensing and stereo coupling run unconditionally inside the VM via `SensoryQuad` and `CoreQuad` Mid/Side projection.

### 2.1 Port Allocation & Physical Signal Roles

`r[dsp.router.ports]`

| Port Identifier | Signal Variable | Physical & Logical Host-Shell Role |
| :--- | :--- | :--- |
| Port In 0 (Inst In L) | `x_inst,L[n]` | Primary mono or Left instrument input into the outer patchbay |
| Port In 1 (Inst In R) | `x_inst,R[n]` | Secondary Right input; normaled to Port In 0 when unplugged |
| Port In 2 (FX Return L) | `x_ret,L[n]` | External amplifier effects return for Amp 1 / Left channel |
| Port In 3 (FX Return R / Aux Key) | `x_ret,R[n]` or `x_aux[n]` | External return for Amp 2 / Right or host sidechain key; normaled to Port In 2 |
| Port Out 0 (Send L) | `y_send,L[n]` | Pre-loop preamp send driving Amp 1 or optional mono WDW send |
| Port Out 1 (Send R / WDW Dry) | `y_send,R[n]` or `y_dry[n]` | Right send in dual-amp rigs or center dry feed in WDW (`PreSpatialCore` / `DirectDry`) |
| Port Out 2 (Main Out L) | `y_out,L[n]` or `y_wet,L[n]` | Main Left output, or wet-only Left in WDW mode |
| Port Out 3 (Main Out R) | `y_out,R[n]` or `y_wet,R[n]` | Main Right output, or wet-only Right in WDW mode |

### 2.2 Host-Shell 4×4 Scattering Matrix & Return-to-Send Isolation

`r[dsp.router.return-to-send-isolation]`

```text
                  [ S_00  S_01 |   0     0   ]   [ S_{inst->send}    0_{2x2}    ]
S_router =        [ S_10  S_11 |   0     0   ] = [                              ]
                  [------------+-------------]   [                              ]
                  [ S_20  S_21 | S_22  S_23  ]   [ S_{inst->down}  S_ret->down  ]
                  [ S_30  S_31 | S_32  S_33  ]
```

$$\mathbf{y}_{\mathrm{send}}[n] = \mathbf{S}_{\mathrm{inst} \to \mathrm{send}} \mathbf{x}_{\mathrm{inst}}[n], \qquad \mathbf{u}_{\mathrm{down}}[n] = \mathbf{S}_{\mathrm{inst} \to \mathrm{down}} \mathbf{x}_{\mathrm{inst}}[n] + \mathbf{S}_{\mathrm{ret} \to \mathrm{down}} \mathbf{x}_{\mathrm{ret, aligned}}[n]$$

Zeroing $\mathbf{S}_{\mathrm{ret} \to \mathrm{send}} \equiv \mathbf{0}_{2 \times 2}$ in the host-shell patchbay guarantees open-loop return ratio $\mathbf{T}_{\mathrm{loop}}(z) = \mathbf{S}_{\mathrm{ret} \to \mathrm{send}} \mathbf{H}_{\mathrm{amp}}(z) \equiv \mathbf{0}_{2 \times 2}$ and $\det(\mathbf{I} - \mathbf{T}_{\mathrm{loop}}(z)) \equiv 1$, eliminating external feedback loops through hardware effects loops while each wrapped VM receives clean stereo `(in_L, in_R)` (`stage_targets[0..32]` fed uniformly by $\mathbf{q}_{16}$; [parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws)).

### 2.3 Drive-Invariant Voltage-Normalized Mid/Side Coupling (`CoreQuad` & `REQ-7.4`)

`r[dsp.router.mid-side-input]`

To maintain exact 0.00 dB drive-threshold parity between mono (1 → 2) and centered stereo (2 → 2) routing where $x_L = x_R = x$ (`REQ-7.4`), SONA uses voltage-normalized Mid/Side decomposition (`x_mid = 0.5*(L + R)`, `x_side = 0.5*(L - R)`) both at `CoreQuad` input coupling and in post-VM host stereo width:

$$u_{\mathrm{mid}}[n] = \frac{1}{2}\big(u_L[n] + u_R[n]\big), \qquad u_{\mathrm{side}}[n] = \frac{1}{2}\big(u_L[n] - u_R[n]\big), \qquad u_L[n] = u_{\mathrm{mid}}[n] + u_{\mathrm{side}}[n], \qquad u_R[n] = u_{\mathrm{mid}}[n] - u_{\mathrm{side}}[n]$$

Each `CoreQuad` $g$ projects `(u_mid, u_side)` across dual 4-lane weights (`in_mid_weight_fp16`, `in_side_weight_fp16`), guaranteeing exact 0.00 dB parity when `u_side == 0.0` while recovering independent L/R when `w_side == w_mid` or learning full elliptical M/S filtering and M/S saturation directly from data.

### 2.4 Topology-Adaptive Even/Odd SIMD Lane Partitioning

`r[dsp.router.topology-dispatch]`

| Feature | Mono (1 → 1) | Stereo-to-Mono (2 → 1) | Mono-to-Stereo (1 → 2) | True Stereo (2 → 2) |
| :--- | :---: | :---: | :---: | :---: |
| Channel Topology | `1 -> 1` | `2 -> 1` | `1 -> 2` | `2 -> 2` |
| `SensoryQuad`, Crossbar & `CoreQuad`s | Normaled L/R paths & 4×4 coupling retained | Full stereo L/R paths, `w_in_l/r` keying & M/S coupling | Normaled L/R paths & 4×4 coupling retained | Full stereo L/R paths & M/S coupling |
| `ModalConvolver` FIR & Kautz | Both L/R histories & modes advance | Both L/R histories & modes advance | Both L/R histories & modes advance | Both L/R histories & modes advance |
| Output Finalization | Mono fold `0.5*(y_L + y_R)` inside VM | Mono fold `0.5*(y_L + y_R)` inside VM | Stereo output (host-shell M/S width `(y_L, y_R)`) | Stereo output (host-shell M/S width `(y_L, y_R)`) |

### 2.5 Stereo-Linked Dynamics Imaging (`SensoryQuad` VCA & `ParaLRU` Link)

`r[dsp.router.stereo-link-detector]`

To prevent stereo image drift (REQ-7.4), both the learned SensoryQuad pre-core input VCA `g_in[n]` (shared scalar gain applied uniformly to `(x_L, x_R)`) and audio-rate CoreQuad ParaLRU stereo linking, where $\lambda_i \in [0, 1]$ mixes channel state toward `v_mid = 0.5*(v_L + v_R)` inside `sag_gain` and `lru_bias_drive`, eliminate unlinked dual-mono pan shifts without fixed peak-detector constants:

$$e_{\mathrm{link}}[n] = \max\big(|x_L[n]|, |x_R[n]|\big), \qquad e_{\mathrm{key}}[n] = |x_{\mathrm{aux}}[n]|, \qquad v_{\mathrm{eff},c,i}[n] = v_{c,i}[n] + \lambda_i\big(v_{\mathrm{mid},i}[n] - v_{c,i}[n]\big)$$

With shared positive gain $G[n] > 0$ (`g_in[n]` on normal binary32 inputs with roundoff $u = 2^{-24}$ ), stereo image angle drift obeys:

$$\Vert\widehat{\mathbf{y}} - G\mathbf{x}\Vert_2 \le u\Vert G\mathbf{x}\Vert_2, \qquad |\sin(\Delta\theta)| \le \frac{u}{1 - u}, \qquad |\Delta\theta| < 0.00000342^\circ$$

### 2.6 `SensoryQuad` (`Op-M/N[Sense]`) Sensing Bank & Input VCA

`r[dsp.router.ballistic-envelope-bank]`

At input, `SensoryQuad` runs four learned sensing lanes at $f_{s,\mathrm{vm}}$ using 2×2 `ZDF` `TPT-SVF` and contractive `ParaLRU` primitives (48 B `SensoryQuad` hot state plus Crossbar offsets 0..64, 88..120, and 240..256), with zero fixed runtime crossover cutoffs or preset envelope timings:

| Step | SensoryQuad Operation (`k` in 0..4 @ `f_{s,vm}`) | Clamped Wire Domain & ZOH Law |
| :--- | :--- | :--- |
| 1. Stereo Key Mix | `u_sense,k = w_in_L,k * x_L + w_in_R,k * x_R` | `sense_in_l_fp16`, `sense_in_r_fp16` in `[-4.0, 4.0]` (0.0 default; expresses Mid, Side, or R-channel aux key with 0 mode bits) |
| 2. Passive `TPT-SVF` | Step 4-lane `ZDF TPT-SVF`, form `s_sense,k = c_lp,k * LP_k + c_bp,k * BP_k + c_hp,k * HP_k` | `sense_svf_g` `[1e-5, 64.0]` (`rescale_tpt_svf_g_zoh`, fallback 0.25); `sense_svf_k` `[0.0, 8.0]` (`1/sqrt(2)`); `sense_tone_{lp,bp,hp}` `[-16.0, 16.0]` |
| 3. Full-Wave `ParaLRU` | Feed `abs(s_sense,k)` into 4-lane contractive `ParaLRU`, emitting non-negative `e_k[n] >= 0` | `sense_lru_alpha_rel` `[0.0, 0x3F7F_FFAC]` (`^rho_vm`); `sense_lru_alpha_atk`, `sense_lru_kappa` `[0.0, 1.0]` (`alpha_atk^rho_vm`, `kappa` scalar or coupled per `SensoryQuad` `ParaLRU`) |
| 4. `ControlBus` `q3` & VCA | Sample `e[0..4]` into `env_states[0..4]` (`q3`) at `K_ctrl` ticks; scale input by smooth VCA `g_in[n]` | `g_in = clamp(1.0 + clamp(t[30], -1.0, 1.0) + sum(w_vca,k * e_k), 0.0, 2.0)`; `sense_vca_w` `[-512.0, 512.0]` (0-ULP 1.0 when zero-init) |

$$s_{\text{sense},k}[n] = c_{\text{lp},k}\mathrm{LP}_k[n] + c_{\text{bp},k}\mathrm{BP}_k[n] + c_{\text{hp},k}\mathrm{HP}_k[n], \qquad g_{\mathrm{in}}[n] = \mathrm{clamp}\left(1.0 + \Delta_{\text{ctrl},30} + \sum_{k=0}^{3} w_{\text{vca},k} e_k[n], 0.0, 2.0\right)$$

Crossbar `SONA_META_SENSORY_QUAD` (bit 17) decodes all `SensoryQuad` wire fields unconditionally even when zeroed, because zero weights yield `e_k == 0.0`, `g_in == 1.0`, and exact 0-ULP pass-through identity. Standalone `ControlBus` invocations without `SensoryQuad` initialize `env_states = [0.0; 4]` deterministically.

### 2.7 Tier-Aware Static & Dynamic Modulation Routing

`r[dsp.router.modulation-bus]`

$$\mathbf{q}_{16}[k] = \big[\mathbf{q}_0^\top,\ \mathbf{q}_1^\top,\ \mathbf{q}_2^\top,\ \mathbf{q}_3^\top\big]^\top \in \mathbb{R}^{16}$$

Every tier evaluates base rows 0..16 (slots 0 through 15) of the 16×16 matrix `W_ctrl` unconditionally over `[q0 = z_head[0..4], q1 = ctrl_basis_4[0..4], q2 = skewed quadrature LFOs, q3 = env_states[0..4]]` (plus `qu = z_stage_ultra[0..4]` through 16×4 `W_ctrl_ultra` in Arena 3 on Ultra), and evaluates optional unwatermarked `"CEXT"` active extended slots 16..31 (slots 16 through 30; row 15 / slot 31 and columns 0..4 fixed at +0.0, defaulting to +0.0 otherwise).

Optional Ornstein-Uhlenbeck drift projects through learned `ou_to_z_stage_fp16[4]` (and Arena 3 `ou_to_z_stage_ultra_fp16[4]` at `stage_latent_ctrl_ultra_fp16[160..164]`) directly into runtime `eff_z_stage = static_z_stage + w_ou * ou_drift_state` whenever `live_analog_drift_enabled && !offline_bounce`.

Control slots 0..32 micro-ramp uniformly across `stage_targets[0..32]` (`svf_g_oct`, `svf_k_delta`, per-block drive/bias, Kautz formant/damping, Transport delay/gain/angle, `crossbar_wet_mod`, slot 16 continuous `freeze_mix_drive`, continuous Op-T / Op-N[Pre] / Op-P parameters, slot 22 `op_t_dry_mix_mod`, and slot 30 `sense_vca_gain_delta` in [-1.0, 1.0]; [parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws)).

Autonomous modulators $\mathbf{m}_{\text{dyn}} \in \mathbb{R}^8$ supply learned 2nd-harmonic skewed quadrature oscillators `q2 = [cos(phi_0) + beta_0*(2*cos^2(phi_0) - 1), sin(phi_0), cos(phi_1) + beta_1*(2*cos^2(phi_1) - 1), sin(phi_1)]` (`beta_j = lfo_skew_fp16[j] in [-1, 1]`) paired with `SensoryQuad` envelopes `env_states[0..4]` in `q3`.

---

## 3. Host-Shell Parallel Alignment, WDW & Smooth Crossbar Entry

### 3.1 Host-Shell WDW Dry-Tap and Wet-Output Contract

`r[dsp.router.wdw-contract]`

| WDW Dry-Tap Mode | Center Dry on Port Out 1 | Wet-Only L/R on Ports Out 2/3 | Host-Shell Applicability |
| :--- | :--- | :--- | :--- |
| `PreSpatialCore` | Mid `m_c` of Crossbar + `CoreQuad`s + `ModalConvolver` | Coercive Op-T `Transport` output `m_w ± alpha*s_w` | Full-rig host chaining (`DirectDry` around wet VM instance) |
| `DirectDry` | Voltage-normalized Mid `0.5*(x_L + x_R)` input or aligned return | Effect-only VM response relative to direct source | Wet effects or external center rigs in host patchbay |

For `PreSpatialCore`, partitioning the non-recirculating core $(m_c, s_c)$ from spatial `Transport` contribution $(m_w, s_w)$ with host-shell stereo width $\alpha$ satisfies:

$$y_L = m_c + m_w + \alpha(s_c + s_w), \qquad y_R = m_c + m_w - \alpha(s_c + s_w)$$
$$y_{\text{dry}} = m_c, \qquad y_{\text{wet},L} = m_w + \alpha s_w, \qquad y_{\text{wet},R} = m_w - \alpha s_w$$

### 3.2 Host-Shell Parallel-Path Sub-Sample Alignment

`r[dsp.router.parallel-alignment]`

| Alignment Mode | Interpolation Kernel | Host Center Offset (G) | Selection & Offline Bounce Rule |
| :--- | :--- | :---: | :--- |
| `FullBand64Tap` | 64-tap Kaiser windowed sinc (beta = 8.6) | 31 samples (0.65 ms at 48 kHz) | Default for full-band 4CM/WDW in host shell, training, and 4× bounce |
| `LowLatency16Tap` | 16-tap Kaiser windowed sinc (beta = 6.5) | 7 samples (0.15 ms at 48 kHz) | Selectable low-latency live monitoring |
| `MinLatency4Tap` | Centered 4-tap Catmull-Rom cubic Hermite | 1 sample (0.021 ms at 48 kHz) | Low-band alignment or minimum-latency monitoring |

For $K \in \lbrace 16, 64 \rbrace$ with $G = (K - 2)/2$, $R = K/2$, and $s_k = k - G - \mu$, cold setup normalizes zero-ended Kaiser sinc weights:

$$w_{\beta}(s) = \frac{I_0\big(\beta\sqrt{1 - (s/R)^2}\big) - 1}{I_0(\beta) - 1} \quad (|s| \le R), \qquad h_k(\mu) = \frac{\mathrm{sinc}(s_k) w_{\beta}(s_k)}{\sum_{j=0}^{K-1}\mathrm{sinc}(s_j) w_{\beta}(s_j)}$$

Given oversampling factor $M \in \lbrace 1, 4 \rbrace$, arrivals $a_i$, $A = \max_i a_i$, and optional common pad $p \ge 0$, setting $\Delta_i = A - a_i + M p$, $J_i = \lfloor \Delta_i \rfloor$, $\mu_i = \Delta_i - J_i$, and $B = (M - 1)G$ aligns channels via:

$$y_i[n] = \sum_{k=0}^{K-1} h_k(\mu_i) x_i[n - B - J_i - k], \qquad d_{\mathrm{eff},i} = M G + \Delta_i$$

At 4× bounce, `FullBand64Tap` uses $B = 93$ plus its 31-sample center to hold the 31-host-sample alignment offset (43 host samples nominal combined PDC with the 12-sample oversampler, needing at least 157 effective-rate samples of history per channel). `MinLatency4Tap` with $G = 1$ uses 4-tap Catmull-Rom with exact 0.00 dB DC sum $\sum_{k=-1}^2 c_k(\mu) \equiv 1.0$:

$$c_{-1}(\mu) = -0.5\mu(1 - \mu)^2, \quad c_0(\mu) = 1 - 0.5\mu^2(5 - 3\mu), \quad c_1(\mu) = 0.5\mu(1 + 4\mu - 3\mu^2), \quad c_2(\mu) = 0.5\mu^2(\mu - 1)$$

### 3.3 Continuous Pre-Core Input VCA & Zero-Crossover Boundary

`r[dsp.router.lr4-crossover]`

SONA v3.0 contains no fixed runtime crossover or discrete hysteretic noise gate inside the pure 2×2 VM. `SensoryQuad` (§2.6) computes `(g_in * x_L, g_in * x_R)` directly from four learned `TPT-SVF` plus full-wave `ParaLRU` lanes (`g_in = clamp(1.0 + clamp(t[30], -1.0, 1.0) + sum(w_vca * e), 0.0, 2.0)`).

Crossbar bytes 124..128 store `anti_zipper_ramp_scale` as the 48 kHz reference per-sample micro-ramp step fraction over 0.0 < s <= 1.0, defaulting to 1.0 / 16.0 = 0.0625 when non-finite or at most 1.0e-7. Multi-band expansion, gating, ducking, and sidechain keying are learned directly from continuous weights across `sense_in_l/r`, `sense_svf_g/k`, `sense_tone_{lp,bp,hp}`, `sense_lru_*`, and `sense_vca_w`, plus optional slot 30 modulation, with zero hardcoded frequency constants.

---

## 4. Top-of-Frame Causal Operator Crossbar

`r[dsp.crossbar.top-of-frame]`

| Crossbar Kernel | State Allocation | Operational Modes & Physical Roles |
| :--- | :--- | :--- |
| `SensoryQuad` (`Op-M/N[Sense]`) | 48 B hot `SensoryQuad` state | Learned 4-lane `TPT-SVF` + full-wave `ParaLRU` feeding `ControlBus` `q3` + uniform input VCA `g_in` (0° stereo shift) |
| `Op-T[Crossbar]` (`tau_min = 5.0`) | Inline `fp16` Quad 0 (1,024 B) + 216 B hot state + cold ext buffers | Unified 4-lane delay, chorus, TZF, `freeze_mix` overdub/hold, ±24 st Hermite-Hann grains, and barberpole sweeps at `f_{s,vm}` |
| `Op-P` (`Hilbert` + 3-Basis + 2-Stage `T1`) | True-stereo `H1` (248 B) + stereo 5 Hz `T1` DC (24 B) + 2-stage mid `T1` (28 B) + 24 B 3-basis | < 0.10 deg (0.0976 deg) at 48 kHz; <= 0.19 deg at 15 anchors; <= 0.34 deg in [44.1, 96] & [176.4, 384] kHz and `f >= 30 Hz` in (96, 176.4) kHz (<= 0.67 deg at 20 Hz near 136.2 kHz); SSB/carrier, cancellation-free `rsqrt_pos_nr3` T1, and 2× `OS2X` midpoint interp |
| 4× `Q0.32` Quadrature Phasors | 4-lane `u32` phase + Hermite-5 sin/cos (`[-0.49 * fs, +0.49 * fs]`) | Chorus/rotary LFOs, tempo sync (`REQ-2.5`), AM/ring-mod carriers (`inc_core = inc_vm >> 1` under `OS2X` else `inc_vm`; slot 29 oct mod) |
| Smooth `Op-N[Pre]` (Front-End) | 12 B wire (`Crossbar[180..192)`) + 28 B hot state (0 64-bit RT ints) | Raised-cosine track-and-hold -> `gamma_bc <= 0` bypass -> smooth Fourier staircase (`q32_smooth_sin`) -> wet blend at `f_{s,vm}` ahead of `OS2X` |

On every VM frame at $f_{s,\mathrm{vm}}$, the Top-of-Frame Crossbar executes sequentially before OS2X and the 4-block CoreQuad stack. First SensoryQuad steps on `(x_L, x_R)` to compute `e[0..4]` and scaled `(x_g,L, x_g,R) = g_in * (x_L, x_R)`; next Op-T reads causal delays $v_{\mathrm{read}}$ with $\tau_{\min} = 5.0$ VM frames and stereo Op-P steps on `(x_g,L, x_g,R)`.

Next, `Op-T` crossbar taps add into the scaled input to form `(s0_L, s0_R)`. Immediately after that addition at $f_{s,\mathrm{vm}}$, smooth `Op-N[Pre]` processes `(s0_L, s0_R)` into core input `u_1x` ahead of `OS2X`.

Cold initialization sets the Op-P tracker to `prev_w1_re = 1.0`, `prev_w1_im = 0.0`, `sheet = +1`, `sheet2 = +1`, `prev_valid = false`, and `(prev_re, prev_im) = (1.0, 0.0)`, with SensoryQuad, 5 Hz DC blocker, and H1 states zeroed. At 3 kHz ticks, slot 29 `op_p_phasor_rate_oct` scales each phasor:

$$f_{\mathrm{eff}} = \mathrm{clamp}\big(f_{\mathrm{base}} \cdot 2^{\mathrm{clamp}(t_{29},\ -4.0,\ +4.0)},\ -0.49 f_{s,\mathrm{vm}},\ +0.49 f_{s,\mathrm{vm}}\big)$$

Converting $f_{\mathrm{eff}}$ to signed `Q0.32` `i32` `inc_vm = round_half_away((f64(f_eff) * 2^32) * inv_fs_vm) as i64 as i32` using symmetric `round_half_away(x) = (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5)` supports negative and positive carrier frequencies without +0.5 floor bias. Offline 4× bounce sets $f_{s,\mathrm{vm}} = 4 f_{s,\mathrm{host}}$ and disables internal oversampling, scaling phase increments automatically by $1 / f_{s,\mathrm{vm}}$.

Each `PhasorQ32Bank4` step executes in evaluate-then-advance order (`eval(phase)` first, then `phase = wrapping_add(phase, inc as u32)`), stepping at `inc_core = inc_vm >> 1` (arithmetic right shift) under `OS2X` and `inc_vm` otherwise.

In Crossbar Coupling Block bytes 224..240, `phasor_stereo_phase_fp16[0..4]` encodes right-minus-left relative stereo phase offsets in radians clamped to $[-\pi, +\pi]$ (Lite defaults `[0.0; 4]`), while slots 4..8 at 232..240 are reserved `0x0000`.

Pure-32-bit `algebraic_unit_rotor_f32` ([op_m_modal_kautz.md §1.2.4](op_m_modal_kautz.md#124-q0192-control-lfo--4-lane-q032-crossbar-phasors)) converts `phasor_stereo_phase_fp16[0..4]` (radians) into `(c_st, s_st)`, rotating Right rotor state from Left via `cos_R = cos_L * c_st - sin_L * s_st` and `sin_R = sin_L * c_st + cos_L * s_st`. Bytes 156..180 store `carrier_basis_w_fp16` as `[w0[4], w1[4], w2[4]]`, driving stereo projection for $c \in \lbrace L, R \rbrace$:

$$\mathrm{car}_c[p] = w_{\mathrm{scale}} \big(w_0[p] I_c \cos_c[p] + w_1[p] Q_c \sin_c[p] + w_2[p] \sin_c[p]\big)$$

Here $w_{\mathrm{scale}} = \mathrm{clamp}(1.0 + \mathrm{clamp}(t_{15}, -1.0, 1.0), 0.0, 2.0)$, `op_p_gain = base_op_p_gain * wet_scale`, and `op_t_gain = base_op_t_gain * wet_scale`. Elementwise per-lane `l` in 0..4 crossbar injection adds `w_opP[g][l] = op_p_to_core[g][l]` and `w_ph[g][l] = phasor_to_core[g][l]` into `CoreQuad` `g`:

$$u_{g,c}[l] \mathrel{+}= w_{\mathrm{opP}}[g][l] \cdot \big(g_{\mathrm{opP}} \cdot T_1[l]\big) + w_{\mathrm{ph}}[g][l] \cdot \mathrm{car}_c[l], \qquad c \in \lbrace L, R \rbrace, \quad l \in \lbrace 0, 1, 2, 3 \rbrace$$

Sequential execution at $f_{s,\mathrm{vm}}$ first performs Op-T crossbar tap addition `s0_L = g_in * x_L + op_t_gain * sum_line(opt_to_lr_L[line] * v_read[line])` and `s0_R = g_in * x_R + op_t_gain * sum_line(opt_to_lr_R[line] * v_read[line])`. It then passes `(s0_L, s0_R)` through smooth `Op-N[Pre]` to produce `u_1x` ahead of `OS2X`.

---

## 5. 4-Block Matryoshka SONA Core

`r[dsp.core.matryoshka-cascade]`

The 4-block Matryoshka core connects `Block 0 -> Block 1 -> Block 2 -> Block 3` across 10 logical 4-lane `CoreQuad`s sorted in `SONA_CORE_QUAD_BLOCK_AND_LOCAL = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (0,2), (2,1), (0,3), (1,2)]` and 35 causal lower-triangular 4×4 inter-block tiles ([op_n_paralru_and_polyadaa.md §3–§4](op_n_paralru_and_polyadaa.md#3-exact-lti-basis-commutativity-theorem-theorem-1)).

While binary `W_ro,0` stores lane `l = 4 * g + local_lane` indexed by tier-introduction global quad g in 0..N_g (unpacked per CoreQuad), runtime block-major execution (`SONA_BLOCK_QUAD_SLOTS = [[0, 4, 6, 8], [1, 5, 9, -], [2, 7, -, -], [3, -, -, -]]`, using hyphen for unused slots) traverses active global quads and accumulates readout at Ultra strictly in sequence [0, 4, 6, 8, 1, 5, 9, 2, 7, 3]:

| SONA Block | Global `CoreQuad` Indices (`-` = unused) | Active Tiers |
| :---: | :--- | :--- |
| Block 0 | 0, 4, 6, 8 | Lite (0), Full (+4), Pro (+6), Ultra (+8) |
| Block 1 | 1, 5, 9, - | Lite (1), Full (+5), Ultra (+9) |
| Block 2 | 2, 7, -, - | Lite (2), Pro (+7) |
| Block 3 | 3, -, -, - | Lite (3) |

| Tier Profile | `CoreQuad`s & Lanes | Inter-Block Tiles | Core Rate | `TPT-SVF` Modes | Active-Hot / Ceiling (Wire Arena) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lite `[1, 1, 1, 1]` | 4 quads (16 lanes) | 3 | 1× `PolyADAA1` | 4 quads (16 modes) | 4,544 B <= 4,608 B (4,608 B wire) |
| Full `[2, 2, 1, 1]` | 6 quads (24 lanes) | 13 | 2× OS (96 kHz) | 6 quads (24 modes) | 7,552 B <= 7,936 B (7,936 B wire) |
| Pro `[3, 2, 2, 1]` | 8 quads (32 lanes) | 23 | 2× OS (96 kHz) | 8 quads (32 modes) | 11,136 B <= 15,872 B (15,872 B wire) |
| Ultra `[4, 3, 2, 1]` | 10 quads (40 lanes) | 35 | 2× OS (4× Bounce) | 10 quads (40 modes) | 14,592 B <= 28,672 B (30,720 B wire) |

Cold init and `reset()` clear dynamic modulator state (`lfo_phases_q192`, `env_states`, `ou_drift_state`, skipping OU PRNG steps) with skewed unity cosine `q2 = [1.0 + beta_0, 0.0, 1.0 + beta_1, 0.0]`, keep static_z_stage separate from runtime `eff_z_stage`, evaluate static dispatch ahead of frame 0 (`ctrl_sub_counter = 0`), and snap `current_stage_params = stage_targets` with zero deltas:

```text
Init/Reset: lfo_phases_q192 = [[0; 3]; 2], lfo_phasors = [[1.0, 0.0], [1.0, 0.0]], env_states = [0.0; 4], ou_drift_state = [0.0; 4]
Dispatch:   q2 = [1.0 + beta_0, 0.0, 1.0 + beta_1, 0.0], q3 = [0.0; 4] -> stage_targets[0..32], current_stage_params = stage_targets, deltas = [0.0; 32]
Activity:   has_internal_modulators = ((lfo_angle_bits[0] & 0x7FFF_FFFF) != 0) || ((lfo_angle_bits[1] & 0x7FFF_FFFF) != 0) || any_nonzero_f32_bits(W_ctrl[:, 8..16], W_ctrl_ext[:, 8..16])
```

At frame 0 and every K_ctrl (`max(8, round(f_{s,vm} / 3_000.0))`) VM frames (`ctrl_sub_counter == 0`), ControlBus runs `tick_3khz()`. On each K_ctrl tick, every slot r in 0..32 (including slot 16 `freeze_mix_drive` and slot 30 `sense_vca_gain_delta`) computes linear ramp slope `delta[r] = (stage_targets[r] - current_stage_params[r]) * alpha_ramp(f_{s,vm})`.

Every VM frame advances `current_stage_params[r]` uniformly by delta[r], clamping inside [min(prev, target), max(prev, target)] and snapping to `stage_targets[r]` when `ctrl_sub_counter + 1 >= K_ctrl`. Recursive slew states—the Kautz 3 kHz formant `prev_oct` and Op-T `current_psi` slew capped at 0.25 rad/tick—update strictly at `ctrl_sub_counter == 0`.

All remaining modulated `CoreQuad`, crossbar, `Op-N[Pre]`, and `Op-T` hot registers latch `current_stage_params` and `eff_z_stage` strictly when `(sample_tick & 3) == 0 || ctrl_sub_counter == 0`, skipped when `deltas == 0 && !dirty`.

The audio callback enters a 4-frame causal micro-chunk strictly when `(sample_tick & 3) == 0 && rem_frames >= 4 && rem_tick >= 4`, peeling 1 frame at a time otherwise while holding latched registers across `(sample_tick & 3)` in 1..4. Under active `OS2X`, each 4-frame VM micro-chunk executes 8 core sub-frames (`2 * K_ctrl` sub-frames per control tick).

Because every active Op-T tap enforces $\tau_{\min} = 5.0$ (`i_m1 = write_idx + 1 - delay_int <= write_idx - 4`), every Catmull-Rom read stays strictly older than all 4 VM frames in the micro-chunk. Output therefore depends exclusively on monotonic `sample_tick`, guaranteeing 0-ULP `process(1) x N == process(4) x (N/4) == process(16) x (N/16)` across arbitrary caller buffer splits.

When `os2x_active` is true on Full through Ultra, where $f_{s,\mathrm{core}} = 2 f_{s,\mathrm{vm}}$, a 256 B common oversampler hot block (32 B params, 144 B 2× state, 16 B `op_p_prev_iq_lr`, 16 B `op_p_prev_feat`, 48 B pad) lets `Op-P` compute perimeter features once per VM frame and midpoint-interpolate `0.5 * (prev + curr)` on sub-frame 0 and curr on sub-frame 1. When `os2x_active` is false—including Lite, live perimeter rates above 192 kHz, or 4× offline bounce—`u_c = u_1x` enters the Core directly.

Runtime tier transitions preserve state via `tier_gain_trim` in [0.0625, 16.0] and execute `warm_seed` on a 1×-to-2× switch, initializing the 2× upsampler and decimator to exact DC steady state with zero runtime `FDIV`:

$$u_{\mathrm{raw,prev}} = u_{\mathrm{prev1}} = u_{\mathrm{prev2}} = x_{\mathrm{last}}, \qquad s_{\mathrm{up},j}[c] = x_{\mathrm{pair}}[c] \cdot \mathrm{INV}_{\mathrm{up},j}[c] \quad (j \in \lbrace 0, 1 \rbrace)$$
$$d_{\mathrm{prev1}} = r_{\mathrm{last}}, \qquad s_{\mathrm{dec}}[\mathrm{st}][c] = r_{\mathrm{last}}[c] \cdot \mathrm{INV}_{\mathrm{dec}}[\mathrm{st}]$$

Here `in_pair = [last_in[0], last_in[1], last_in[0], last_in[1]]` pairs with `HB_A0` for c in 0..2 and `HB_A1` for c in 2..4, while `last_s2_dir = [pre_conv_L, pre_conv_R, direct_s2_L, direct_s2_R]` sets `dec_prev1 = last_s2_dir` alongside `up_raw_prev = up_prev1 = up_prev2 = last_in`.

State initialization multiplies exclusively by compile-time `f32` inverse constants: `INV_UP_HB_S0[c] = 1.0 / (1.0 + HB_A*[0])`, `INV_UP_HB_S1[c] = 1.0 / (1.0 + HB_A*[1])`, and `INV_DEC_HB[st] = 1.0 / (1.0 + [HB_A0[0], HB_A0[1], HB_A1[0], HB_A1[1]][st])`.

Non-finite samples set `INVALID_NONFINITE` and flush to 0.0, while finite inputs exceeding `max(certified_limit, CERTIFIED_INPUT_ENVELOPE_MIN_F32)` (±100.0 / +40 dBFS) set `OUTSIDE_CERTIFIED_INPUT` telemetry and pass unclipped; only magnitudes exceeding `SAFETY_CEILING_F32` (±1.0e9) clamp to `[-SAFETY_CEILING_F32, SAFETY_CEILING_F32]` with `INPUT_SATURATED`, `INTERNAL_SATURATED`, or `OUTPUT_SATURATED`.

On real-time 2× oversampling (Full through Ultra), the 2× stereo half-band oversampler cascades a 9th-order elliptic half-band all-pass filter (`HB_A0` and `HB_A1`) over clamped weights (`[up_quad_coeff, dec_c_prev1, dec_c0, dec_c1, up_pre_emph]`):

$$A_\alpha(z) = \frac{\alpha + z^{-1}}{1 + \alpha z^{-1}}, \qquad H_{\mathrm{HB}}(z) = \frac{1}{2}\Big(A_{\alpha_{0,0}}(z^2) A_{\alpha_{0,1}}(z^2) + z^{-1} A_{\alpha_{1,0}}(z^2) A_{\alpha_{1,1}}(z^2)\Big)$$

Specifically, the half-band filters fix `HB_A0 = [0.079866426, 0.54535365]` and `HB_A1 = [0.28382934, 0.8344119]`.

Oversampler parameters clamp `up_quad_coeff` inside [-0.20, 0.05], `dec_c_prev1` inside [0.0, 0.28], `dec_c0` inside [0.0, 0.55], `up_pre_emph` inside [-0.28, 1.0], and `[dec_aa_mix, up_aa_mix, up_zoh_mix]` inside [0.0, 1.0], deriving `dec_c1 = 1 - dec_c_prev1 - dec_c0`.

| Oversampler Field | Clamped Domain | Default When `SONA_META_OS_EXT_PRESENT == 0` |
| :--- | :---: | :---: |
| `up_quad_coeff` | `[-0.20, 0.05]` | -0.125 |
| `dec_c_prev1`, `dec_c0`, `dec_c1` | `[0.0, 0.28]`, `[0.0, 0.55]`, `1 - dec_c_prev1 - dec_c0` | 0.15, 0.50, 0.35 |
| `up_pre_emph` | `[-0.28, 1.0]` | 0.0 |
| `dec_aa_mix`, `up_aa_mix`, `up_zoh_mix` | `[0.0, 1.0]` | 0.0, 0.0, 0.0 |

When `SONA_META_OS_EXT_PRESENT == 0`, decoders apply the oversampler defaults above and also force `carrier_basis_w_fp16 = [0; 12]` and `Op-N[Pre]` `(gamma_bc, zoh_hold_depth, beta_q) = (0.0, 0.0, 0.0)` to bypass both stages cleanly.

Upsampling (`1× -> 2×` on `u_1x` after `Op-N[Pre]`) computes pre-emphasized $x_p$ (`up_raw_prev' = x`), second difference $d_2$, ZOH blend $c_{p1}$, polynomial midpoint $x_{0,\mathrm{poly}}$, and $x_{1,\mathrm{poly}} = x_p$, updating `up_prev2' = up_prev1` and `up_prev1' = x_p`:

$$x_p = (1 - p_{\mathrm{pre}}) x + p_{\mathrm{pre}} x_{\mathrm{prev}}, \qquad d_2 = x_p - 2 x_1 + x_2, \qquad c_{p1} = 0.5(1 - \mu_{\mathrm{zoh}})$$
$$x_{0,\mathrm{poly}} = (1 - c_{p1}) x_p + c_{p1} x_1 + q_{\mathrm{up}} d_2, \qquad x_{1,\mathrm{poly}} = x_p$$

Here $x_{\mathrm{prev}} =$ `up_raw_prev`, $(x_1, x_2) =$ `(up_prev1, up_prev2)`, $p_{\mathrm{pre}} =$ `up_pre_emph`, $\mu_{\mathrm{zoh}} =$ `up_zoh_mix`, and $q_{\mathrm{up}} =$ `up_quad_coeff`.

Both upsampling and decimation step 2-stage all-pass cascades via `s_new = flush(u - alpha * s_old)` and `y = flush(alpha * s_new + s_old)` (`s_old' = s_new`). In upsampling, `HB_A0` steps on $x_p$ using `s_even` state to emit $x_{0,\mathrm{hb}} = \mathrm{HB}_{A0}(x_p)$, and `HB_A1` steps on $x_p$ using `s_odd` state to emit $x_{1,\mathrm{hb}} = \mathrm{HB}_{A1}(x_p)$, blending via $\mu_{\mathrm{up,aa}} =$ `up_aa_mix`:

$$s_{\mathrm{new}} = \mathrm{flush}(u - \alpha s_{\mathrm{old}}), \qquad y = \mathrm{flush}(\alpha s_{\mathrm{new}} + s_{\mathrm{old}})$$
$$x_{0,\mathrm{out}} = x_{0,\mathrm{poly}} + \mu_{\mathrm{up,aa}} (x_{0,\mathrm{hb}} - x_{0,\mathrm{poly}}), \qquad x_{1,\mathrm{out}} = x_{1,\mathrm{poly}} + \mu_{\mathrm{up,aa}} (x_{1,\mathrm{hb}} - x_{1,\mathrm{poly}})$$

Decimation from 2× to 1× returns `(pre_conv, direct_s2)` back to $f_{s,\mathrm{vm}}$; it steps `HB_A0` on `y[2n + 1]` and `HB_A1` on `y[2n]` to form $x_{\mathrm{hb}}[n] = \mathrm{flush}(0.5(b_0 + b_1))$ and blends via $\alpha_{\mathrm{aa}} =$ `dec_aa_mix` with the unity-DC 3-tap `poly3` FIR:

$$x_{\mathrm{poly}}[n] = c_{\mathrm{prev1}} y[2n - 1] + c_0 y[2n] + c_1 y[2n + 1], \qquad x_{\mathrm{out}}[n] = x_{\mathrm{poly}}[n] + \alpha_{\mathrm{aa}} \big(x_{\mathrm{hb}}[n] - x_{\mathrm{poly}}[n]\big)$$

On isolated 4× offline bounce, $f_{s,\mathrm{vm}} = 4 f_{s,\mathrm{host}}$ and internal `OS2X` is bypassed so $f_{s,\mathrm{core}} = f_{s,\mathrm{vm}}$. The 4× polyphase oversampler wraps the VM and cascades two linear-phase dyadic-rational half-band FIR stages: a 19-tap 1×/2× filter (`K_1 = 5`, `HB_2X_TAPS = [35, -405, 2268, -8820, 39690] / 65536`, 9 high-rate delay) and an 11-tap 2×/4× filter (`K_2 = 3`, `HB_4X_TAPS = [3, -25, 150] / 256`, 5 high-rate delay), yielding `OFFLINE_BOUNCE_PDC_SAMPLES = 12` host samples (43 combined PDC with `FullBand64Tap`):

$$y_{\uparrow}[2n] = \sum_{m=0}^{K-1} w_m \Big(x[n - m] + x[n - (2K - 1 - m)]\Big), \qquad y_{\uparrow}[2n + 1] = x[n - (K - 1)]$$
$$x_{\downarrow}[n] = 0.5 y[2n - (2K - 1)] + \frac{1}{2} \sum_{m=0}^{K-1} w_m \Big(y[2n - 2m] + y[2n - 2(2K - 1 - m)]\Big)$$

Across all tiers, passive `TPT-SVF` with $\Vert\mathbf{A}_{\text{tpt}}\Vert_2 \le 1$, contractive `ParaLRU` satisfying $J_i \in [0, r_{\max}]$, and fast-path `PolyADAA1` maintain analytical 0.000000 DC equilibrium (`h_eq == +0.0`), while linear `pre_conv` and `direct_s2` readouts carry ±100.0 (+40 dBFS) unclipped range (`REQ-7.1`).

---

## 6. `ModalConvolver`: Lag-Ordered Early FIR & Shared Kautz Bank

`r[dsp.cabinet.fir-kautz-bank]`

`ModalConvolver` (`modal_convolver`) executes between `OS2X` decimation and the end-of-frame `Transport` stage. It pairs independent L/R early FIR slices (32, 48, 96, or 128 taps/ch) with shared-pole 4-pair stereo Kautz conjugate pairs (8, 12, 24, or 32 pairs = 16, 24, 48, or 64 real poles, driven at $x_a[n-16]$ at 48 kHz) hoisted across 4-frame micro-chunks ([op_m_modal_kautz.md §4](op_m_modal_kautz.md#4-shared-pole-kautz-modal-bank)).

### 6.1 Lag-Ordered Early FIR History & Contiguous Micro-Chunk Evaluation

`r[dsp.cabinet.mirrored-fir-simd]`

When `f_{s,vm} == 48,000` Hz, each channel stores single-copy lag-ordered history `[x[n-1], x[n-2], ..., x[n-N]]` directly in the hot working set (32, 48, 96, or 128 taps/ch: 256 B Lite, +128 B Full, +384 B Pro, +256 B Ultra across L and R). On every 4-frame micro-chunk, it copies those `N` samples plus 4 incoming frames onto stack scratch `work[N + 4]` (at most 528 B/ch at Ultra) for branchless SIMD convolution and one shift back ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)).

Whenever `f_{s,vm} != 48,000` Hz (live off-48 kHz or 4× bounce), execution activates the cold 1,024-frame double-mirrored extension ring (2,048 floats per channel, `mask = 1,023`) for both FIR convolution and `onset_delay_samples` lookback, while tier upgrades zero-fill newly unlocked `[N_old..N_new)` history slots.

---

## 7. Coercive `Op-T` Symmetric `Transport` Stage

`r[dsp.tank.feedback-loop]`

| Transport Stage Component (`transport`) | Architectural & Mathematical Specification |
| :--- | :--- |
| Physical Structure & Causal Feedback | `M = 4 * P_Q` lines (`P_Q` in `{1, 2, 3, 4}`) read at Crossbar (`tau_min = 5.0`, 4-frame micro-chunks) and written back at frame end per 128 B serialized / 216 B `align(8)` active-hot symmetric pure-wet `TransportQuad` using 128-frame `fp16` inline Quad 0 (1,024 B) or power-of-two cold external buffers, Hermite-5 `Q0.32` LFO (`lfo_depth_samples <= 512.0`), unified ramp/windowing, Catmull-Rom interpolation, and bounded feedback ([op_t_transport.md §1.1](op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)) |
| Partitioned Schroeder Diffusers | `4 quads × 4 lanes = 16` partitioned per-lane Schroeder all-pass stages wrapped in register-only `SO(4)` paraunitary scattering `Q(scat_psi) = R_odd * R_even` across Quads 0..3 using lane-major SoA 4-frame micro-chunking (up to 16,384 samples at 384 kHz, `abs(a_diff) <= 0.75`), stepped before tap weight and loop write with `(1 - mu_freeze)` tapering of `allpass_g` and `scat_psi` ([op_t_transport.md §3.1](op_t_transport.md#31-memory-partitioning--rate-aware-capacity)) |
| Stage Dry Mix & Givens + Householder `H_4(1)` | Each quad applies pairwise 2×2 Givens `R(psi)` (`abs(d_psi) <= 0.25 rad/tick`) and 4×4 Householder `H_4(1) = I_4 - 0.5*1*1^T`, then sums symmetrically `y_tank,c = g_dry_eff * x_tank,c + w_scale * sum_q y_wet,q,c` across `P_Q` active quads |
| Input & Readout DC Conditioning | Passband-normalized 5 Hz DC blocker (`0x3F7F_D51F` at 48 kHz) at loop write and non-recirculating stereo wet readout (§7.2) |
| Unified Loop Limiter & Continuous Freeze | Cold-precomputed `inv_sigma_23` and `sigma_sat = clamp(sigma_knee, 1.0e-3, 0.984375)` (`0x3BE0`, `TANK_MAX_G_LIN_BITS = 0x3F7F_BE76`, 0.9990) drive the unity-slope cubic saturator (`‖G_loop‖ < 1`, zero feedback `NR2`, §7.3–§7.4) paired with continuous passive `freeze_mix` (§7.1) |

On each quadrant $u \in [0, 1]$, the unique odd polynomial $P(u) = c_1 u - c_3 u^3 + c_5 u^5$ satisfying $P(0) = 0$, $P'(0) = \pi/2$, $P''(0) = 0$, $P(1) = 1$, and $P'(1) = 0$ solves analytically to $c_1 = \pi/2$, $c_3 = \pi - 5/2$, and $c_5 = \pi/2 - 3/2$. Cold external buffer sizing activates when `q > 0 || (quad_wire_flags[0] & 0x02) != 0 || max_k(wire_delay_48k[0][k] * scale) > 120.0 || ceil(max_k((delay + 0.5 * abs(ramp_span) + lfo_depth) * scale)) + 6 > 128 || ext_ring_pow2_frames >= 8` ([op_t_transport.md §1.1.1](op_t_transport.md#111-ring-declaration-unified-4-lane-quad-wire--active-hot-state)) so audio paths use only bitmask `& (cap - 1)` with zero division or standard math library calls.

### 7.1 Continuous Passive `freeze_mix` & Host Write Lock

`r[dsp.tank.captured-freeze]`

Sustained hold transitions captured in training stems ([capture_and_alignment.md §1.3](../training/capture_and_alignment.md#sustain-hold-and-write-enable-transition-capture)) calibrate naturally onto uniformly micro-ramped `ControlBus` slot 16 (`op_t_freeze_mix_delta`), while unobserved actuator ranges or options clamp via non-stop 3-phase auditing ([control_space_audit.md](../training/control_space_audit.md)) with zero dedicated container freeze flags and zero host freeze APIs.

Each quad forms effective recirculation ratio `mu_freeze = clamp(base_freeze_mix[q] + t[16], 0.0, 1.0)` (forced to 1.0 whenever host `QUAD_FLAG_HOST_WRITE_LOCK` bit 2 is set), tapers `half_alpha_cubic`, `lfo_depths`, `ramp_spans`, Schroeder `allpass_g`, and `scat_psi` by 1 - mu_freeze, and writes passive blend w_ring = (1 - mu_freeze) * y_dc + mu_freeze * r_win into both fp16 and f32 rings. At mu_freeze >= 1.0 - 1.0e-6, `we_hold_off` holds ring addresses bit-for-bit at 0 ULP with zero re-quantization hiss while Crossbar, CoreQuads, ModalConvolver, and non-recirculating readouts continue processing live audio.

### 7.2 `Transport` Passband-Normalized DC Blocker

`r[dsp.tank.dc-blocker]`

For pole $0 \le R_{\text{dc}} < 1$, the passband-normalized DC blocker protects both loop feedback injection and non-recirculating stereo readouts:

$$H_{\text{dc}}(z) = \frac{\alpha_{\text{dc}}(1 - z^{-1})}{1 - R_{\text{dc}}z^{-1}}, \qquad \alpha_{\text{dc}} = \frac{1 + R_{\text{dc}}}{2}, \qquad R_{\text{dc,ref}}(48\text{ kHz}) = \frac{\cos(2\pi \cdot 5 / 48000)}{1 + \sin(2\pi \cdot 5 / 48000)} \approx \mathtt{0x3F7FD51F}$$

Off-48 kHz sample rates compute $R_{\text{dc}}(f_{s,\text{vm}}) = R_{\text{dc,ref}}^{\rho_{\text{vm}}}$ on the cold path (§9).

### 7.3 Unified `sigma_sat` Cubic Loop Saturator

`r[dsp.tank.loop-saturator]`

With ceiling $\sigma_{\text{sat}} = \mathrm{clamp}(\sigma_{\text{knee}}, 10^{-3}, 0.984375)$ (`TANK_MAX_SIGMA_SAT_FP16 = 0x3BE0`, non-finite defaulting to 0.984375), `TANK_MAX_V_CEIL_BITS = 0x3F7C_28F5` (0.985), slope $0 \le g_{\text{lin}} \le 0.9990$ (`TANK_MAX_G_LIN_BITS = 0x3F7F_BE76`), and cold-precomputed `inv_sigma_23` equal to $2 / (3 \sigma_{\text{sat}})$, each `TransportQuad` evaluates one memoryless unity-slope cubic saturator with zero feedback-loop division or reciprocal steps:

$$v = \mathrm{clamp}\big(u \cdot \mathrm{inv-sigma-23}, -1, 1\big), \qquad S(u) = \mathrm{clamp}\Big(g_{\text{lin}} \sigma_{\text{sat}} v(1.5 - 0.5 v^2), -V_{\text{ceil}}, V_{\text{ceil}}\Big)$$

The outer $\pm V_{\text{ceil}}$ clamp ensures intermediate rounding preserves passive contraction $\Vert G_{\text{loop}} \Vert < 1$ with unity small-signal gain $g_{\text{lin}}$.

### 7.4 Generalized Quiet-Feedback Knee Law

`r[dsp.tank.quiet-feedback-knee]`

Standalone operators and analytical audits additionally support the parameterized rational feedback knee with slope $0 \le g_{\text{lin}} \le 0.9990$ (`TANK_MAX_G_LIN_BITS = 0x3F7F_BE76`), blend $\alpha \in [0, 1]$, and knee scale $\sigma \in [10^{-3}, 8.0]$:

$$K(u) = g_{\text{lin}} u \left((1 - \alpha) + \frac{\alpha}{1 + (u/\sigma)^2}\right), \qquad K'(0) = g_{\text{lin}}, \qquad |K(u)| \le g_{\text{lin}}|u|$$

In compact `TransportQuad` instances, feedback normalization sets $\alpha = 0$ (linear small-signal slope $g_{\text{lin}}$ into the unified $\sigma_{\text{sat}}$ cubic limiter of §7.3 with zero feedback `NR2` reciprocal), whereas standalone kernels support arbitrary $\alpha \in [0, 1]$ using a 2-step Newton reciprocal of `1 + (u * inv_sigma)^2` (or 6-step `f64` Newton reciprocals with `rel_err < 2^-40` on single-line paths).

### 7.5 4-Lane SIMD Loop Conditioning, Saturation & Orthogonal Mixing

`r[dsp.tank.simd-loop]`

Inside 4-lane 32-bit float SIMD registers ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)), each `TransportQuad` updates 4-lane loop DC blockers with recursive subnormal flushing and evaluates the unified $\sigma_{\text{sat}}$ cubic saturator using cold-precomputed `inv_sigma_23` (`TANK_MAX_SIGMA_SAT_FP16 = 0x3BE0`, zero loop reciprocal).

Whenever the slewed Givens angle stays stationary (`abs(target_psi - current_psi) <= 1e-7`), the quad reuses cached `(cos_psi, sin_psi)` (otherwise recomputed via pure-32-bit `algebraic_unit_rotor_f32`), rotates pairwise via lane swap-and-negate, and applies 4×4 Householder reflection via horizontal lane reduction and fused multiply-add.

---

## 8. Direct-Register `LinearReadout`

`r[dsp.head.direct-register-readout]`

SONA accumulates `pre_conv` and `direct_s2` `LinearReadout` sums directly in SIMD registers ([ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)) via packed `[u16; 8]` half-precision readout gains (16 B) inside an execution-ordered active-hot L1D working set (`N × ActiveHot <= 64 KiB` plus the shared 256 B = 4 CL `OS2X` hot block on Full through Ultra, with zero intermediate scratch feature-bus gather buffers):

| Tier Profile | SONA Core (`TPT-SVF` @ Rate) | `ModalConvolver` (FIR + Kautz) | `TransportQuad` (`Op-T`) | Active-Hot L1D vs Ceiling (`N× <= 64 KiB` & Wire) |
| :--- | :---: | :---: | :---: | :---: |
| Lite | `[1, 1, 1, 1]` (4 `CoreQuad`s / 16 lanes @ 1×) | 8 pairs (256 B wire) + 32 taps (256 B) | 1 quad (216 B + 1,024 B inline + ext) | 4,544 B (71 CL <= 4,608 B; 14× = 63,616 B; 4,608 B wire) |
| Full | `[2, 2, 1, 1]` (6 `CoreQuad`s / 24 lanes @ 2× OS) | 12 pairs (384 B wire) + 48 taps (384 B) | 2 quads + Schroeder | 7,552 B (118 CL <= 7,936 B; 8× = 60,416 B; 7,936 B wire) |
| Pro | `[3, 2, 2, 1]` (8 `CoreQuad`s / 32 lanes @ 2× OS) | 24 pairs (768 B wire) + 96 taps (768 B) | 3 quads + `fp16`/`f32` | 11,136 B (174 CL <= 15,872 B; 4× = 44,544 B; 15,872 B wire) |
| Ultra | `[4, 3, 2, 1]` (10 `CoreQuad`s / 40 lanes @ 2× / 4×) | 32 pairs (1,024 B wire) + 128 taps (1,024 B) | 4 quads (4 × 4 = 16 Schroeder stages) | 14,592 B (228 CL <= 28,672 B; 2× = 29,184 B; 30,720 B wire) |

### 8.1 Canonical Mid/Side Output Head Factorization

`r[dsp.head.mid-side-factorization]`

At load and tier switch, the tier `LinearReadout` block unpacks static `fp16` base matrix `W_ro,0` (16×12, 24×12, 32×12, or 40×12 across tiers) storing Left and Right channel readout vectors $\mathbf{w}_{L,0}$ and $\mathbf{w}_{R,0}$. At 3 kHz control boundaries, these Left and Right channel vectors hoist directly into stereo register pairs and accumulate directly into Left and Right `pre_conv` and `direct_s2` outputs over SONA state $\boldsymbol{\Phi}[n]$.

The active-hot slab allocates 12 B (three 32-bit floats) for 1-pole tilt/shelf readout conditioning via `hp_lp_pole` and stereo `hp_lp_state[2]`. Cold setup decodes `"DBRM"` `[44..48)` `ro_pole_48k` ([0.0, 0x3F7F_FFAC]) via ZOH scaling (`p_ro = 0.0` when `ro_pole_48k <= 0.0`, else `clamp(ro_pole_48k^rho_core, 0.0, 0.99999)`).

Input `u_c` for $c \in \lbrace L, R \rbrace$ is the core input sample after Op-N[Pre] and OS2X upsampling when active, feeding `ro_gains_fp16[8]` (gains rg 0..8); because $\mathrm{LP}_p(u) = u - \mathrm{HP}_p(u)$, $\lbrace u, \mathrm{HP}_p(u) \rbrace$ spans all 1st-order low-shelf, high-shelf, tilt, and all-pass blends:

$$\mathrm{lp}_c' = \mathrm{flush}\big(\mathrm{lp}_c + (1 - p_{\mathrm{ro}})(u_c - \mathrm{lp}_c)\big), \qquad \mathrm{hp}_c = u_c - \mathrm{lp}_c'$$
$$\mathrm{pre}_L = \mathrm{pre}_L + \big((rg_0 u_L) + (rg_2 \mathrm{hp}_L)\big), \qquad \mathrm{pre}_R = \mathrm{pre}_R + \big((rg_1 u_R) + (rg_3 \mathrm{hp}_R)\big)$$
$$\mathrm{dir}_L = \mathrm{dir}_L + \big((rg_4 u_L) + (rg_6 \mathrm{hp}_L)\big), \qquad \mathrm{dir}_R = \mathrm{dir}_R + \big((rg_5 u_R) + (rg_7 \mathrm{hp}_R)\big)$$

Following paired horizontal reduction of `pre_conv_acc` and `dir_acc`, those direct `u` and `hp` sums evaluate in unfused IEEE-754 `binary32` (`FP_CONTRACT OFF`).

Meanwhile `z_head` stays inside `W_ctrl` while stage sensitivity tables (`stage_latent_mods_fp16`, `stage_latent_ctrl_ultra_fp16`, `stage_latent_core_mods_fp16`) index strictly `z_stage` for 0-ULP `SO(4)` invariance ([parametric_control_bus.md §3.1](parametric_control_bus.md#31-operator-parameter-dispatch-matrix--sensitivity-laws)).

After OS2X decimates `(pre_conv, direct_s2)` back to $f_{s,\mathrm{vm}}$ when active, ModalConvolver evaluates `pre_conv` at $f_{s,\mathrm{vm}}$ to emit `(conv_l, conv_r)` and feeds `x_tank = (direct_s2_l + conv_l, direct_s2_r + conv_r)` into the symmetric Transport stage combining stage-level `tank_dry_mix_eff` (slot 22) and P_Q pure-wet TransportQuad outputs:

$$g_{\mathrm{dry,eff}} = \mathrm{clamp}\big(\mathrm{tank-dry-mix} + t_{22}, -2.0, +2.0\big), \qquad y_{\mathrm{tank},c} = g_{\mathrm{dry,eff}} x_{\mathrm{tank},c} + w_{\mathrm{scale}} \sum_{q=0}^{P_Q - 1} y_{\mathrm{wet},q,c}, \qquad c \in \lbrace L, R \rbrace$$

Stage-level `tank_dry_mix_fp16` in the Arena 0 Control Plane decodes `0x0000` verbatim as 0.0 and falls back to 1.0 on non-finite values, yielding `y_tank,c` directly before `mono_out` folding.

```text
Tier Crossfade (init/reset: prev = (0.0, 0.0), xfade_rem = 0; tier switch sets xfade_rem = 48; unfused binary32 FP_CONTRACT OFF):
  Per output frame:
    if xfade_rem > 0:
      a = f32(xfade_rem) * (1.0 / 48.0);  prev_mid = 0.5 * (prev_L + prev_R)
      if mono_out: out_L = prev_mid * a + out_L * (1.0 - a);  out_R = out_L
      else:        out_c = prev_c * a + out_c * (1.0 - a)
      xfade_rem -= 1
    if xfade_rem == 0:
      prev = (out_L, out_R)   // O(1) sub-block equivalent: update prev = (out_L[len - 1], out_R[len - 1]) iff post-loop xfade_rem == 0 && len > 0
```

Right after `mono_out` and tier crossfade, the pure 2×2 VM multiplies by polarity-signed trim `eff_trim = (wet_polarity_inv ? -1.0 : +1.0) * clamp(tier_gain_trim[tier], 0.0625, 16.0)` controlled by `WET_POLARITY_INV` and applies ±1.0e9 safety clamping on `(y_L[n], y_R[n])`. Outer host-shell post-VM M/S width adjusts stereo spread on `(y_L[n], y_R[n])` after VM exit:

$$y_{\mathrm{mid}}[n] = \frac{1}{2}\big(y_L[n] + y_R[n]\big), \qquad y_{\mathrm{side}}[n] = \frac{1}{2}\big(y_L[n] - y_R[n]\big)$$
$$y_L'[n] = y_{\mathrm{mid}}[n] + \alpha_{\mathrm{width}}[n] y_{\mathrm{side}}[n], \qquad y_R'[n] = y_{\mathrm{mid}}[n] - \alpha_{\mathrm{width}}[n] y_{\mathrm{side}}[n]$$

Mono graphs (1 → 1 and 2 → 1) fold `0.5 * (y_L + y_R)` inside the VM while keeping internal dual-mic and ping-pong state active and bounding Mid error by $|m - \hat{m}| \le 2\mathrm{ulp}(M) + 2F_{\min}$ for $M = \max(|y_{\mathrm{mid}}|, |\alpha_{\mathrm{width}} y_{\mathrm{side}}|)$.

### 8.2 Effective-Tier Equivalence

`r[dsp.head.tier-equivalence]`

For identical external inputs, controls, initial tier state, and execution target, a container stripped to tier `T` produces bit-identical SONA core states, L/R FIR histories, and readout outputs to the full container executed at tier `T` with higher-tier weights gated off and Arena 0 controls preserved.

### 8.3 Block-Diagonal Lower-Triangular Staging

`r[dsp.head.lower-triangular]`

```text
W_head = [ W_lite           0             0             0           ]
         [ W_full,lite     W_full,delta  0             0           ]
         [ W_pro,lite      W_pro,full    W_pro,delta   0           ]
         [ W_ultra,lite    W_ultra,full  W_ultra,pro   W_ultra,delta ]
```

### 8.4 Contiguous Hot Working Set, Register Hoisting & 3 kHz Readout Collapse

`r[dsp.head.precollapse-and-zerocopy]`

Hot weights and audio states pack contiguously across `[0..ActiveHotBytes)` inside the 64-byte-aligned hot working set (4,544 / 7,552 / 11,136 / 14,592 B actual inside 4,608 / 7,936 / 15,872 / 28,672 B ceilings, including the 256 B `OS2X` block on Full+ and 16 B half-precision `ro_gains_fp16[8]` with 0 B transient heap allocation), keeping `P` control definitions, Arena 0 and Arena 3 control-bus sections, `"TSYN"`, optional `"CEXT"`, and static `W_ro,0` at the cold tail ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

Modulated hot registers refresh strictly when `(sample_tick & 3) == 0 || ctrl_sub_counter == 0` (holding latched registers across 1-frame peels while 4-frame causal micro-chunks execute 8 sub-frames under active `OS2X`), keeping all `TPT-SVF`, `ParaLRU`, FIR, and 4-pair stereo Kautz constants hoisted in SIMD registers.

At every `K_ctrl` control boundary when active (`num_controls > 0`, `tsyn_count > 0`, active modulators, dirty controls, or drift enabled), the control bus evaluates `stage_targets[0..32]` (32 single-precision base plus `"CEXT"` targets) and `SONA_META_CORE_MODS_PRESENT` / unrotated `z_stage` sensitivity tables without extra `f_{s,vm}` MACs when static.

---

## 9. Continuous-Time ZOH Rate Adaptation

Containers anchor parameters at $f_{s,\mathrm{train}} = 48{,}000$ Hz. For host sample rate $f_{s,\mathrm{host}} \in [44{,}100, 384{,}000]$ Hz, VM perimeter blocks step at `f_{s,vm}` (`f_{s,host}` live, or `4 * f_{s,host}` in 4× offline bounce where internal `OS2X` is bypassed), giving `rho_vm = 48_000.0 / f_{s,vm}`.

Internal 2× oversampling wraps only the 4-Block SONA Core iff `os2x_active = (tier > 0) && !offline_bounce && tier_os2x[tier] && (2.0 * f_{s,vm} <= 384_000.0)` (`f_{s,core} = 2 * f_{s,vm}`, else `f_{s,core} = f_{s,vm}`), giving `rho_core = 48_000.0 / f_{s,core}`. Live 44.1–96 kHz runs 2× RT on Full+ or 4× bounce, live 96.001–192 kHz runs 1× or 2× up to a 384 kHz core ceiling, live > 192–384 kHz runs 1×, and Lite always runs 1×.

### 9.1 Causal Crossbar, `SensoryQuad`, Core `TPT-SVF` & Kautz Conversion

`r[dsp.zoh.op-m-poles]`

$$g_{\text{core}} = \tan\left(\frac{\pi f_c}{f_{s,\text{core}}}\right), \qquad r_c(f_{s,\text{vm}}) = \big(r_{\text{ref},c}\big)^{\rho_{\text{vm}}} \in \big[0.25^{\rho_{\text{vm}}},\ 0.9995^{\rho_{\text{vm}}}\big], \qquad \theta_c(f_{s,\text{vm}}) = \mathrm{clamp}\Big(\rho_{\text{vm}} \theta_c(f_{s,\text{train}}),\ 10^{-4},\ \pi - 10^{-4}\Big)$$

Core TPT-SVF prewarps at $f_{s,\mathrm{core}}$, whereas SensoryQuad (§2.6), `Op-N[Pre]` (`z_h_eff = powf_f64(zoh_hold_depth, rho_vm)`), and ModalConvolver Kautz poles adapt at $f_{s,\mathrm{vm}}$. At 48 kHz 1×, where $f_{s,\mathrm{vm}} = f_{s,\mathrm{core}} = 48{,}000$ Hz, stored parameters run directly; at `K_ctrl` ticks, TPT-SVF, Kautz, Transport, and `crossbar_wet_mod` parameters retune dynamically without division or standard math library calls.

All cold setup and ZOH rate conversions across host alignment, SensoryQuad, TPT-SVF, ParaLRU, `Op-N[Pre]`, `Op-T`, `ro_pole_48k`, Kautz poles, and ControlBus execute deterministic math-library-free 64-bit and 32-bit routines without FDIV or FSQRT:

```text
// Cold-Boundary Math-Library-Free Transcendental Reference (FP_CONTRACT OFF, 0 FDIV, 0 FSQRT)
rcp_nr6_f64(d > 0):
  inv = f64_from_bits((bits(d) & (1<<63)) | (u64(clamp(1022 - (i32((bits(d)>>52) & 0x7FF) - 1023), 1, 2046)) << 52))
  repeat 6 times: inv = inv * (2.0 - d * inv)
  rcp_signed_f64(x) = (x >= 0.0) ? rcp_nr6_f64(max(x, 1.0e-15)) : -rcp_nr6_f64(max(-x, 1.0e-15))

sqrt_f64(x):
  if x.is_nan(): return NaN; if x <= 0.0: return 0.0; if x == +inf: return +inf
  (sx, s_out) = (x < f64_from_bits(0x0010_0000_0000_0000)) ? (x * 2^54, 2^-27) : (x, 1.0)
  z = f64_from_bits(0x5FE6_EB50_C7B5_37A9 - (bits(sx) >> 1))
  repeat 6 times: z = z * (1.5 - 0.5 * sx * z * z)
  y = sx * z; return (y + 0.5 * (sx - y * y) * z) * s_out

ln_f64(x > 0):
  k = i64((bits(x) >> 52) & 0x7FF) - 1023
  m = f64_from_bits((bits(x) & 0x000F_FFFF_FFFF_FFFF) | (1023 << 52))
  if m > SQRT_2 (0x3FF6_A09E_667F_3BCD): m *= 0.5; k += 1
  u = (m - 1.0) * rcp_nr6_f64(m + 1.0); u2 = u * u; term = u; sum = u
  for n in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]: term *= u2; sum += term * (1.0 / f64(n))
  return 2.0 * sum + f64(k) * LN_2 (0x3FE6_2E42_FEFA_39EF)

exp_f64(x):
  if x < -708.0: return 0.0; if x > 709.0: return +inf
  k = i64(round_half_away(x * LOG2_E (0x3FF7_1547_652B_82FE))); r = x - f64(k) * LN_2
  term = 1.0; sum = 1.0
  for i in 1..=14: term *= r * (1.0 / f64(i)); sum += term
  return sum * f64_from_bits(u64(k + 1023) << 52)

powf_f64(b, e):
  if b <= 0.0: return 0.0
  if |e - 1.0| < 1.0e-15: return b
  if |e| < 1.0e-15: return 1.0
  return exp_f64(e * ln_f64(b))
  // ZOH real poles clamp input/output to [1e-4, 0.999995] (or operator bounds) via f32(powf_f64(f64(p), rho))
  // ParaLRU ratios evaluate (1 - a_atk^rho) / (1 - a_atk) in f64 via powf_f64 + rcp_nr6_f64 (returning f32(rho) if |1 - a_atk| <= 1e-12)

cold_atan_pos_f64(g > 0):
  quad_sin_cos(x): k = i64(round_half_away(x * 2/pi)); r = x - f64(k) * (pi/2)
    s_loc = sum_{j=0..10} (-1)^j r^(2j+1)/(2j+1)! ; c_loc = sum_{j=0..10} (-1)^j r^(2j)/(2j)!
    rotate by (k & 3) mod 4 -> (s, c)
  [lo, hi] = [0.0, pi/2 - 1e-7]
  repeat 52 times: mid = 0.5 * (lo + hi); (s, c) = quad_sin_cos(mid); if s < g * c: lo = mid else: hi = mid
  return 0.5 * (lo + hi)

prewarp_tan_pi_f_over_fs_f32(r):
  rc = clamp(r, 1e-5, 0.495); t = (rc <= 0.25) ? rc : (0.5 - rc); x = PI_F32 * t; x2 = x * x
  s = x * (1 + x2 * (-1/6 + x2 * (1/120 + x2 * (-1/5040 + x2 * (1/362880 - x2 * (1/39916800))))))
  c = 1 + x2 * (-0.5 + x2 * (1/24 + x2 * (-1/720 + x2 * (1/40320 - x2 * (1/3628800)))))
  rcp_nr3(d) = seed(0x7EEF_FFFF - bits(d)) then 3x inv = inv * (2.0 - d * inv)
  return (rc <= 0.25) ? (s * rcp_nr3(c)) : (c * rcp_nr3(s))

rescale_tpt_svf_g_zoh(g_ref, rho_sr):
  g_pos = is_finite(g_ref) ? clamp(g_ref, 1e-5, 64.0) : 0.1
  if |rho_sr - 1.0| <= 1e-9: return g_pos
  r_eff = f32(clamp(cold_atan_pos_f64(f64(g_pos)) * (1/pi) * rho_sr, 1e-5, 0.495))   // tan(0.495*pi) = 63.6567 <= 64.0; avoids > pi/2 wrap at 44.1 kHz
  return prewarp_tan_pi_f_over_fs_f32(r_eff)

kaiser_sinc (all 1/k! and INV_M_SQ[m] = 1/(m*m) are compile-time f64 constants for host alignment, 0 runtime FDIV):
  Kaiser I0(z): h2 = 0.25 * z * z; term = 1.0; sum = 1.0; for m in 1..=28: term *= h2 * INV_M_SQ[m]; sum += term
  Sinc(x): if |x| < 1e-12: return 1.0
           k = i64(round_half_away(x)); u = pi * (x - f64(k)); u2 = u * u
           sin_u = u * (1.0 - u2 * (1/6 - u2 * (1/120 - u2 * (1/5040 - u2 * (1/362880 - u2 * (1/39916800))))))
           return ((-1)^k * sin_u) * rcp_signed_f64(pi * x)
```

### 9.2 Host-Shell Alignment Latency & `SensoryQuad` Poles

`r[dsp.zoh.router-detector]`

$$\delta_{\text{align},ch}^{\text{samples}}(f_{s,\text{vm}}) = \frac{\delta_{\text{align},ch}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{vm}}}, \qquad \alpha_{\text{rel},k}' = \mathrm{clamp}\big(\alpha_{\text{rel},k}^{\rho_{\text{vm}}}, 0, r_{\max}\big), \qquad g_{\text{sense},k}' = \tan\big(\rho_{\text{vm}}\arctan(g_{\text{sense},k})\big)$$

### 9.3 4-Block Matryoshka Core `TPT-SVF` & `ParaLRU` ZOH Adaptation

`r[dsp.zoh.core-paralru]`

For each `ParaLRU` lane at $f_{s,\mathrm{core}}$ with $\beta = 1 - \alpha_{\text{atk}}$, $a_{\text{atk}} = \alpha_{\text{rel}}(1 - \kappa\beta)$, and $r_{\max} =$ `MAX_PARALRU_R_F32` (`0x3F7F_FFAC`):

$$\alpha_{\text{rel}}' = \mathrm{clamp}\big(\alpha_{\text{rel}}^{\rho_{\text{core}}}, 0, r_{\max}\big), \qquad \beta' = \mathrm{clamp}\left(\beta \frac{1 - a_{\text{atk}}^{\rho_{\text{core}}}}{1 - a_{\text{atk}}}, 0, 1\right), \qquad \alpha_{\text{atk}}' = 1 - \beta'$$
$$\kappa' = \mathrm{clamp}\left(\frac{1 - (1 - \kappa\beta)^{\rho_{\text{core}}}}{\beta'}, 0, 1\right) \ (\beta' > 10^{-12}, \text{ else } \kappa), \qquad R_{\text{ac}}(f_{s,\text{core}}) = \big(R_{\text{ac}}(f_{s,\text{train}})\big)^{\rho_{\text{core}}}$$

Core TPT-SVF modes rescale via analytical tangent prewarping $g_{\mathrm{core}} = \tan(\pi f_c / f_{s,\mathrm{core}})$, and LinearReadout conditioning adapts `"DBRM"` `ro_pole_48k` (0.0 when `<= 0.0`, else `clamp(ro_pole_48k^rho_core, 0.0, 0.99999)`). Real-pole ZOH rescaling pre-clamps nonzero inputs to `[1.0e-4, 0.999995]`, returning p_clamped unchanged when $|\rho - 1| < 10^{-12}$ and otherwise returning `clamp(p_clamped^rho, 1.0e-4, 0.999995)`.

This preserves sub-5 Hz AC blockers across 44.1–384 kHz, including the upstream 5 Hz `T1` DC blocker in `Op-P`, while the `T1` tracker itself is rate-invariant because it computes half-angle features directly from instantaneous normalized phasor components `(u_re, u_im)` at $f_{s,\mathrm{vm}}$.

### 9.4 `Op-T` Delays, All-Pass Diffusers, Crossfades & DC Blocker

`r[dsp.zoh.op-t-delays]`

$$\tau_{0,m}^{\text{samples}}(f_{s,\text{vm}}) = \max\left(\tau_{\min},\ \frac{\tau_{0,m}^{\text{samples}}(f_{s,\text{train}})}{\rho_{\text{vm}}}\right), \qquad D_{\text{ap},m}(f_{s,\text{vm}}) = \max\left(1,\ \mathrm{round}\left(\frac{D_{\text{ap},m}(f_{s,\text{train}})}{\rho_{\text{vm}}}\right)\right)$$
$$L_{\text{xfade}}(f_{s,\text{vm}}) = \max\left(16,\ \mathrm{round}\left(\frac{L_{\text{xfade}}(f_{s,\text{train}})}{\rho_{\text{vm}}}\right)\right), \qquad R_{\text{dc}}(f_{s,\text{vm}}) = \big(R_{\text{dc}}(f_{s,\text{train}})\big)^{\rho_{\text{vm}}}$$

Here $\tau_{\min} = 5.0$ for Top-of-Frame crossbar and `Transport` quads stepped in 4-frame causal micro-chunks, or $\tau_{\min} = 4.0$ on standalone single-sample rings. Fields `ramp_spans_samples`, `lfo_depths_fp16`, and 48 kHz `tank_delay_mod` (`stage_targets[12]`) rescale identically by `1 / rho_vm` into VM frames, while 48 kHz `u8` Schroeder delays `allpass_delays[4]` expand up to 16,384 samples at 384 kHz cleanly without bypass.

### 9.5 Control-Bus Decimation Stride, Tick Ratio & Micro-Ramp Step

`r[dsp.zoh.ctrl-stride]`

$$K_{\mathrm{ctrl}}(f_{s,\mathrm{vm}}) = \max\left(8,\ \mathrm{round}\left(\frac{f_{s,\mathrm{vm}}}{3000.0}\right)\right), \qquad \rho_{\mathrm{ctrl}} = \frac{f_{s,\mathrm{train}} \cdot K_{\mathrm{ctrl}}(f_{s,\mathrm{vm}})}{f_{s,\mathrm{vm}} \cdot K_{\mathrm{ctrl}}(f_{s,\mathrm{train}})}$$
$$\alpha_{\mathrm{ramp}}(f_{s,\mathrm{vm}}) = \mathrm{clamp}\left(s_{\mathrm{ramp}} \cdot \frac{16.0}{K_{\mathrm{ctrl}}(f_{s,\mathrm{vm}})},\ 0.0,\ 1.0\right), \qquad \Delta\phi_{\mathrm{lfo},j}(f_{s,\mathrm{vm}}) = \rho_{\mathrm{ctrl}} \Delta\phi_{\mathrm{lfo},j}(f_{s,\mathrm{train}})$$

Here $\mathrm{round}(x)$ denotes symmetric half-away-from-zero rounding `round_half_away(x) = (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5)`, which avoids +0.5 floor bias on negative values while matching `floor(x + 0.5)` on positive `round(f_{s,vm} / 3_000.0)` (16 at 48 kHz and 15 at 44.1 kHz). Also $s_{\mathrm{ramp}} =$ `anti_zipper_ramp_scale` defaults to 0.0625 when `<= 1.0e-7` or non-finite per [parametric_control_bus.md §3.2](parametric_control_bus.md#32-division-free-linear-micro-ramping).

During full-graph VM execution, the 3 kHz `ControlBus` samples the four per-VM-frame `SensoryQuad` `ParaLRU` envelopes `[e_0, e_1, e_2, e_3]` (§2.6) directly at each `K_ctrl` boundary after `rho_vm` ZOH scaling (§9.1–§9.2), while standalone `ControlBus` invocations initialize `env_states = [0.0; 4]` deterministically.

At `f_{s,vm}`, `tick_3khz()` runs strictly when `ctrl_sub_counter == 0` whenever `num_controls > 0`, `tsyn_count > 0`, internal modulators/LFOs are active, controls are dirty, or analog drift is enabled, stepping all 32 slots of `current_stage_params` uniformly on every VM frame and latching registers strictly at `(sample_tick & 3) == 0 || ctrl_sub_counter == 0`.

### 9.6 `ModalConvolver` FIR & Modal Onset Conversion

`r[dsp.zoh.cabinet-fir-onset]`

`ModalConvolver` FIR and modal onset conversion executes at `f_{s,vm}` within an 8-host-sample budget in four mutually exclusive priority cases:

| Case | Rate Domain (`f_{s,vm}`) | FIR Tap & Onset Rule |
| :--- | :--- | :--- |
| Case (a) | Integer multiples `q * 48,000 Hz`, `q in {1, 2, 4, 8}` | Sparse delays `x_a[n - q * k]` for `k = 16t + j` and modal onset `16 * q` ([op_m_modal_kautz.md §4.1](op_m_modal_kautz.md#41-modalconvolver-architecture--tier-allocation)) |
| Case (b) | Live 1× `M_os = 1`, 44,100 <= `f_{s,vm}` < 48,000 | Evaluates `b_0 * x[n] + b_1 * y_{tap1}[n] + sum_{k=0}^{out_len - 1} b_out[k] * x[n - k]` with 64-tap fractional difference filter using 63 adjacent difference terms; tail taps `b[2..active_base]` resample via Case (d) Catmull-Rom |
| Case (c) | 4× offline bounce `M_os = 4`, `f_{s,vm}` in 176,400–384,000 Hz | Causal 6-point Lagrange difference mapping (192 and 384 kHz match Case (a) identically) |
| Case (d) | All remaining pairs including 88.2 kHz | DC-normalized 4-tap Catmull-Rom FIR tap rescaling and modal onset `round(16 / rho_vm)` |

In Case (a), tap index $k = 16t + j$ steps across 16-tap SIMD segment index $t \in \lbrace 0, \dots, N_{\text{fir}}/16 - 1 \rbrace$ and intra-segment offset $j \in \lbrace 0, \dots, 15 \rbrace$.

In Case (b), cold live first-tap difference preparation computes $\lambda = (48{,}000 - f_{s,\text{vm}}) / 3{,}900$ and prefix sums $p_0 = t_0$, $p_j = (\sum_{i=0}^j t_i) - 1.0$ over $j = 1 \dots 62$ from widened `f64` constants in `GAIN_BOUNDED_FIRST_TAP_64_BITS[0..64]`, then rounds $\widetilde p_j = \mathrm{RN}_{32}(\lambda p_j)$ to `f32` and evaluates $y_{\text{tap1}}[n] = x[n-1] + \sum_{j=0}^{62} \widetilde p_j (x[n-j] - x[n-j-1])$.

Thus Case (b) evaluates `b_0 * x[n] + b_1 * y_{tap1}[n] + sum_{k=0}^{out_len - 1} b_out[k] * x[n - k]` (tap-1 complex error under 0.0177 through 20 kHz), where `b_1` scales the 64-tap fractional difference filter `y_{tap1}[n]` using 63 adjacent difference terms over `x[n]..x[n - 63]`.

Base taps `b[2..active_base]` masked as `[0.0, 0.0, b_2..b_{N-1}, 0.0]` of length `active_base + 1` run Case (d) with `out_len = active_base` to emit `b_out[0..out_len]`, retaining `b_out[0] * x[n]` and nonzero `b_out[1] * x[n - 1]` from $k = 0$ so Case (d) DC normalization $S_{\mathrm{out}}$ is preserved:

```text
GAIN_BOUNDED_FIRST_TAP_64_BITS[0..64] = [
  0x3d6c9268, 0x3f83725f, 0xbdf0a4f1, 0x3dadb502, 0xbd9da17d, 0x3d95380c, 0xbd949c5f, 0x3d91792e,
  0xbd91ce40, 0x3d8dd068, 0xbd8beae0, 0x3d853dac, 0xbd80189f, 0x3d6c7fa0, 0xbd5ba083, 0x3d4231cb,
  0xbd2bf99b, 0x3d0e88f9, 0xbcea4a93, 0x3cac58c1, 0xbc714876, 0x3bf4c5f0, 0xbae22743, 0xbb962d07,
  0x3c1634d0, 0xbc65add4, 0x3c8b7e54, 0xbca4d3dc, 0x3caf0e76, 0xbcb982b1, 0x3cb5d3c2, 0xbcb2e27a,
  0x3ca3932b, 0xbc963f77, 0x3c7dc51e, 0xbc565369, 0x3c1fd21c, 0xbbe7f768, 0x3b775394, 0xbaa98a21,
  0xbacf4513, 0x3b66e3c0, 0xbbbaaf04, 0x3be253a0, 0xbc0770ed, 0x3c0ea839, 0xbc181cd3, 0x3c14420d,
  0xbc133bf5, 0x3c074843, 0xbbfe526f, 0x3bdd6c26, 0xbbc62d29, 0x3ba31dc4, 0xbb8b1039, 0x3b568017,
  0xbb2d823b, 0x3af63e3f, 0xbabad495, 0x3a6ae4c8, 0xba231f37, 0x39a60fed, 0xb943301a, 0x386dc990
]
```

All 64 little-endian words in `GAIN_BOUNDED_FIRST_TAP_64_BITS[0..64]` (ending with `0x386dc990` at index 63) have SHA-256 `befa94d57b6450d5533034cb06c7b4d1b7c96fd738eca0cc6ab35586845250d5` and `f32` accumulator sum `0x3f80_0002`. Difference-coefficient synthesis consumes prefixes 0..63 over indices 0..62, whose leading 63 `u32le` words have SHA-256 `44744b3b9f80f8bc9ff88a98e298485916368b0370ab7153d8d0fda0c4cca4e4`.

In Case (d), cold FIR rescaling computes `out_len = clamp(ceil(active_base / rho_vm), active_base, 1024)`, or `out_len = active_base` in Case (b) over `[0.0, 0.0, b_2..b_{N-1}, 0.0]` of length `in_len = active_base + 1`. For each output tap index $n \in [0, L_{\mathrm{out}})$:

$$u = n \rho_{\mathrm{vm}}, \qquad i = \lfloor u \rfloor, \qquad a = \mathrm{f32}(u - i)$$
$$w_{-1} = -0.5 a^3 + a^2 - 0.5 a, \qquad w_0 = 1.5 a^3 - 2.5 a^2 + 1.0$$
$$w_1 = -1.5 a^3 + 2.0 a^2 + 0.5 a, \qquad w_2 = 0.5 a^3 - 0.5 a^2$$
$$\widetilde{b}_{\mathrm{out}}[n] = \rho_{\mathrm{vm}} \sum_{k=-1}^{2} w_k \cdot b_{\mathrm{in}}\big[\mathrm{clamp}(i + k,\ 0,\ L_{\mathrm{in}} - 1)\big]$$
$$m_{\mathrm{dc}} = \frac{S_{\mathrm{in}} \cdot \mathrm{sign}(S_{\mathrm{out}})}{|S_{\mathrm{out}}|}, \qquad b_{\mathrm{out}}[n] = m_{\mathrm{dc}} \widetilde{b}_{\mathrm{out}}[n] \quad (|S_{\mathrm{out}}| > 10^{-9} \text{ and } |S_{\mathrm{in}}| > 10^{-9})$$

Here $L_{\mathrm{out}} =$ `out_len`, $L_{\mathrm{in}} =$ `in_len`, $S_{\mathrm{in}} = \sum_m b_{\mathrm{in}}[m]$, and $S_{\mathrm{out}} = \sum_n \widetilde{b}_{\mathrm{out}}[n]$.

In Case (c), for tap $k \ge 1$ with exact integer decomposition $d = k f / 48000 = m + \mu$ where $m = \lfloor d \rfloor \ge 3$ and $\mu = r/D$, anchor at $a = 2$ when $\mu \le 1/2$ or $a = 3$ when $\mu > 1/2$ with scales $c_i = (-1, 5, -10, 10, -5, 1)$ at nodes $v_i = (-2, -1, 0, 1, 2, 3)$, $f = M 2^{e-52}$, $D = 48000 \cdot 2^{52-e}$, and $k M = m D + r$:

$$y_{\text{tap},k}[n] = x[n - m - a + 2] + \sum_{j=0}^4 p_j \big(x[n - m + 2 - j] - x[n - m + 1 - j]\big)$$
$$Q_j = \sum_{i=0}^j c_i \prod_{\ell \ne i}(r - v_\ell D) - 120 D^5 \mathbf{1}_{j \ge a}, \qquad p_j = \mathrm{RN}_{32}\left(\frac{Q_j}{120 D^5}\right)$$

This 6-point kernel bounds complex error below 0.00064 through 20 kHz with max ring lookback 251, 379, 763, and 1,019 samples/ch across Lite..Ultra (252, 380, 764, and 1,020 history slots).
