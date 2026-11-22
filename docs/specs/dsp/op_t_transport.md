---
id: spec.dsp.op-t
title: "Op-T: Universal Transport, Granular Windowing & All-Pass Diffusion Operator"
status: canonical
version: "3.0.0"
domain: dsp
rule_prefixes:
  - "r[dsp.op-t."
requirements:
  - REQ-1.4
  - REQ-6.2
adrs:
  - ADR-0006
  - ADR-0019
  - ADR-0022
---

# Op-T: Universal Transport, Granular Windowing & All-Pass Diffusion Operator

This specification defines Op-T (Transport), governing causal top-of-frame fractional delay reads (with $\tau_{\min} = 5.0$ samples across 4-frame crossbar/`TransportQuad` micro-chunks and $\tau[n] \ge 4.0$ on standalone single-sample rings), pitch-warped granular windows, barberpole phase sweeps, bounded-history coercive feedback loops, continuous passive `freeze_mix` + host `WE` RAM write lock, and fused `SO(4)` paraunitary Schroeder all-pass diffusion across Dynamic Neural Audio Modeler (DNAM) SONA v3.0 ([sona_architecture.md](../architecture/sona_architecture.md), [REQ-1.4, REQ-6.2](../../requirements/system_goals_and_requirements.md)).

Every SONA Op-T read executes inside a single unified continuous 4-lane Top-of-Frame Causal Operator Crossbar pipeline before the 4-block Matryoshka core, while coercive loop writes update delay rings at frame end. This unified placement expresses pre-drive chorus, rotary modulation, and reverb-into-fuzz chains alongside post-core tape/BBD echoes, granular pitch shifters, barberpole sweeps, and shimmer holds strictly via learned weights ([ADR-0006](../../adr/0006-model-declared-delays-and-tank-feedback-guard.md), [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

## 1. Model-Declared Ring Buffer Addressing & Stateless Catmull-Rom Reader

### 1.1 Exact-Sized Delay Histories and Branchless Wrap

#### 1.1.1 Ring Declaration, Unified 4-Lane Quad Wire & Active-Hot State

`r[dsp.op-t.ring.declaration]`

Every active Op-T delay history declares its maximum reference delay $D_{\mathrm{ref}} \in \mathbb{N}^+$, storage stride `sample_format` (`"fp16"` for 2 B or `"f32"` for 4 B in 30 s+ ultra-low-floor tails), tier ownership (Lite through Ultra), and owner metadata covering all admitted delay, granular, and crossfade read paths.

Op-T separates standalone arbitrary-length delay lines with block max-tree feedback guards (§1.1.4 and §1.3) from the unified symmetric pure-wet 4-lane `TransportQuad`s (`transport`), which allocate 1, 2, 3, or 4 quads across Lite..Ultra for 4, 8, 12, or 16 delay lines without discrete mode bits or freeze-latch flags.

```text
+---------------------------------------------------------------------------------------------------+
| OP-T 64-BYTE-ALIGNED PARTITIONED DELAY & DIFFUSER RING BUFFER                                     |
+---------------------------------------------------------------------------------------------------+
|  [0 .. L_main - 1]: Main Delay Ring (L_main = ceil(D_ref * f_eff / 48000) + 2 samples)            |
|  [L_main .. L_main + D_ap - 1]: Partitioned Schroeder All-Pass Segment (D_ap samples)             |
|  [.. B bytes]: 64-Byte Cache-Line Padding (B = 64 * ceil(a * L / 64), not ring state)             |
+---------------------------------------------------------------------------------------------------+
```

In compact `TransportQuad`s, Quad 0 places short lines (such as primes `[53, 67, 79, 89]`) inline inside the 64-byte-aligned hot working set as 128 frame-major `fp16` vectors (1,024 B = 16 cache lines), requiring zero heap allocations when `max_delay <= 120` and `max_delay + ceil(lfo_depth) + 6 <= 128`.

Longer lines (up to 48,000 reference samples at 48 kHz and 524,288 power-of-two capacity covering 8× rate headroom up to 384 kHz) allocate power-of-two cold external delay buffers during load and wrap via bitmask `& (cap - 1)` with zero division.

| Tier Profile | Quads | Delay Lines (M) | Inline Quad 0 (Hot Working Set) | Cold External Buffer Format |
| :--- | :---: | :---: | :--- | :--- |
| Lite | 1 | 4 | 128 frames `fp16` (1,024 B) | Optional `"fp16"` (2 B) power-of-two |
| Full | 2 | 8 | 128 frames `fp16` (1,024 B) | Optional `"fp16"` (2 B) power-of-two |
| Pro | 3 | 12 | 128 frames `fp16` (1,024 B) | Optional `"fp16"` or `"f32"` (2 B or 4 B) |
| Ultra | 4 | 16 | 128 frames `fp16` (1,024 B) | Optional `"fp16"` or `"f32"` (2 B or 4 B) |

Each `TransportQuad` is serialized in the tier arena as a 128-byte little-endian descriptor (bytes 0..128) encoding 32-bit float delays and ramp spans (preventing `fp16` overflow at high rates), loop/damping poles, half-precision modulation gains, continuous `freeze_mix_fp16`, 48 kHz Schroeder delays, and mode-free flags:

| Wire Offset | Bytes | LE Type | Field Name | Clean-Room Role & Cold-Load Constraint |
| :---: | :---: | :--- | :--- | :--- |
| 0..16 | 16 | `f32le[4]` | `delay_samples[4]` | 48 kHz base per-lane delays (`[5.0, 48,000.0]` on crossbar/`TransportQuad`s) |
| 16..32 | 16 | `f32le[4]` | `ramp_spans_samples[4]` | 48 kHz base ramp spans per lane (`f32` avoids `fp16` overflow at 384 kHz) |
| 32..56 | 24 | 6 × `f32le` | `g_loop`, `r_dc`, `damp_pole`, `lfo_rate_hz`, `ramp_rate_hz`, `reserved_52_56` | Loop feedback, 5 Hz DC pole, HF damp pole, LFO/ramp rates, and reserved 0.0 at 52..56 |
| 56..112 | 56 | 7 × `u16le[4]` | `lfo_depths_fp16[4]`, `lfo_phase_off_q16[4]`, `ramp_phase_off_q16[4]`, `input_gains_fp16[4]`, `wet_tap_l_fp16[4]`, `wet_tap_r_fp16[4]`, `allpass_g_fp16[4]` | Per-lane LFO depths/offsets, ramp phase offsets, injection gains, L/R wet tap gains, and Schroeder `allpass_g` in `[-0.75, 0.75]` |
| 112..122 | 10 | 5 × `u16le` | `fb_psi_fp16`, `sigma_knee_fp16`, `freeze_mix_fp16`, `window_depth_fp16`, `scat_psi_fp16` | Givens rotation, unified loop limiter ceiling, passive `freeze_mix`, window depth, and `SO(4)` paraunitary scattering angle `[-pi, +pi]` (default 0.0) |
| 122..126 | 4 | `u8[4]` | `allpass_delays[4]` | 48 kHz base Schroeder lengths <= 255; expanded into rate-scaled integer host buffers on load up to 16,384 samples at 384 kHz |
| 126..127 | 1 | `u8` | `flags` | Bit 0 = 0 reserved, bit 1 = `ext_f32` (`0 = fp16`, `1 = f32` ring precision for this quad; logical OR sets cold sizing `ext_ring_format`); bits 2..7 strictly 0 (zero discrete stage-bypass, mode, or freeze bits) |
| 127..128 | 1 | `u8` | `_pad` | Reserved pad byte; must equal `0x00` |

At runtime, each pure-wet active `TransportQuad` occupies a 216-byte 8-byte-aligned hot state block (bytes 0..216) inside the cache-locked working set using 2-state Direct-Form I DC blockers, precomputed `inv_sigma_23`, continuous `freeze_mix`, and host write-lock flags:

| Hot Offset | Bytes | Type | Active-Hot Fields | Runtime Invariant |
| :--- | :---: | :--- | :--- | :--- |
| 0..64 | 64 | 16 × `f32` | `dc_x_prev[4]`, `dc_y_prev[4]`, `damp_state[4]`, `wet_dc_x[2]`, `wet_dc_y[2]` | 2-state Direct-Form I loop DC blockers, per-lane HF damp state, and stereo wet DC blockers |
| 64..96 | 32 | 8 × `f32` | `delay_samples[4]`, `ramp_spans_samples[4]` | Effective-rate base delays and ramp spans |
| 96..144 | 48 | 5 × 32-bit + 7 × `f32` | `write_ptr: u32`, `lfo_phase_q32: u32`, `lfo_inc_q32: u32`, `ramp_phase_q32: u32`, `ramp_inc_q32: i32`, `g_loop`, `r_dc`, `damp_pole`, `current_psi`, `cos_psi`, `sin_psi`, `wet_dc_pole` | Backward write cursor, LFO/ramp accumulators, loop/wet DC and damp poles, and cached Givens `(cos, sin)` |
| 144..216 | 72 | `u16`/`u8` + pad | `we_hold_off[4]: u16` (inline hold offset; 32-bit counters in cold storage when `HAS_EXT`), `lfo_phase_off_q16[4]`, `ramp_phase_off_q16[4]`, `lfo_depths_fp16[4]`, `input_gains_fp16[4]`, `wet_tap_l_fp16[4]`, `wet_tap_r_fp16[4]`, 6 scalar `fp16` (`window_depth_fp16`, `freeze_mix_fp16`, `sigma_sat_fp16`, `inv_sigma_23_fp16`, `half_alpha_cubic_fp16`, `base_psi_fp16`), `flags: u8`, `_pad8: u8`, `_pad16: u16` | Hold offsets, per-lane Q16/`fp16` gains, scalar `fp16` params (`inv_sigma_23 = 2/(3*sigma_sat)`), and `flags` (`bit 0 = 0` reserved, `bit 1 = HAS_EXT`, `bit 2 = HOST_WRITE_LOCK`, bits 3..7 = 0) |

All active crossbar and `TransportQuad` blocks execute at `f_{s,vm}` (`rho_vm = 48_000.0 / f_{s,vm}`) and enforce the causal delay floor `tau_min = 5.0` samples (`delay_int >= 5`) at Top-of-Frame so 4-tap Catmull-Rom accesses remain at least 4 samples behind the newest frame in each 4-frame causal micro-chunk (`i_m1 = write_idx + 1 - delay_int <= write_idx - 4`). This preserves 0-ULP equivalence between 1-frame and 4-frame micro-chunk stepping (`ref[dsp.op-t.ring.q0-32-pointer]`), while standalone single-sample rings allow `tau >= 4.0`.

At rate ratio $\rho_{\mathrm{vm}} = 48{,}000 / f_{s,\mathrm{vm}}$ (`scale = 1.0 / rho_vm`), each active quad $q \in [0, P_Q)$ deterministically computes external buffer selection `has_ext[q]` (hot `flags` bit 1 `HAS_EXT`) and power-of-two ring capacity `cap[q]` from wire fields, `declared_max_delay_48k`, and `ext_ring_pow2_frames`:

```text
req_frames[q] = ceil(max_k((wire_delay_48k[q][k] + 0.5 * abs(wire_ramp_span_48k[q][k]) + wire_lfo_depth_48k[q][k]) * scale)) + 6
dbrm_pow2_cap = (ext_ring_pow2_frames >= 7) ? (1 << ext_ring_pow2_frames) : 0
decl_scaled   = ceil(f64(declared_max_delay_48k) * scale) + 8
has_ext[q]    = (q > 0) || ((quad_wire_flags[0] & 0x02) != 0) || (max_k(wire_delay_48k[0][k] * scale) > 120.0) || (req_frames[0] > 128) || (ext_ring_pow2_frames >= 8)
cap[q]        = !has_ext[q] ? 128 : clamp(next_pow2(max(128, req_frames[q], decl_scaled, dbrm_pow2_cap)), 128, 524_288)
```

When `!has_ext[0]`, Quad 0 uses the 128-frame inline `fp16` hot ring (0 B cold storage). Whenever `has_ext[q]` is true (always for expansion quads `q >= 1` on Full..Ultra, or when Quad 0 exceeds inline limits or selects `ext_f32`), cold load provisions `cap[q]` power-of-two frames in element format `(quad_wire_flags[q] & 0x02) != 0 ? f32 : fp16` (failing closed if caller-provided fixed arenas are undersized).

Active `TransportQuad` blocks rescale and clamp runtime parameters at cold load using `rho_vm = 48_000.0 / f_{s,vm}` against each quad's resolved `(has_ext[q], cap[q])`:

| Parameter | Clamped Cold-Load Domain | Notes & Invariants |
| :--- | :--- | :--- |
| `delay_samples[0..4]` | `[5.0, max_tau]` at `f_{s,vm}` (`[4.0, max_tau]` standalone) | ZOH-rescaled by `1 / rho_vm` from 48 kHz `[5.0, 48,000.0]`; `max_tau = cap - 3` if modulated else `cap - 2` |
| `ramp_spans_samples[0..4]` | `[0.0, max_span]` samples | ZOH-rescaled by `1 / rho_vm` on cold load; `max_span = max(0.0, max_tau - 5.0)` |
| `allpass_delays[0..4]` | `[0, 16,384]` samples | 48 kHz `u8 <= 255`; 0 sets `A_k` to unity with `y_1,k = x_1,k` and zero cold ring accesses while `R_even`/`R_odd` still run whenever not in full hold, with `[0; 4]` allocating 0 B cold ring RAM; nonzero lengths expand via `max(1, round(D_48k / rho_vm))` |
| `allpass_g_fp16[0..4]` | `[-0.75, 0.75]` | Per-lane Schroeder all-pass coefficients tapered by `(1 - mu_freeze)` during feedback recirculation |
| `g_loop` | `[0.0, MAX_LOOP_GAIN_F32]` | Clamped by `TANK_MAX_G_LIN_BITS` (`0x3F7F_BE76`, 0.9990) and pitch-down energy limit (§2) |
| `sigma_knee_fp16` | `[1.0e-3, 0.984375]` (`0x3BE0`) | Unified loop saturator ceiling `sigma_sat` (non-finite fallback `0.984375 = 0x3BE0`) with cold `inv_sigma_23 = 2 / (3 * sigma_sat)` |
| `freeze_mix_fp16` | `[0.0, 1.0]` | Base continuous passive recirculation mix (non-finite fallback 0.0); stage-level `tank_dry_mix_fp16` resides in Arena 0 Control Plane |
| `base_psi_fp16` | `[-pi, pi]` rad | Base Givens angle; `target_psi = clamp(base_psi + tank_psi_delta, -pi, +pi)` (clamped, never wrapped mod `2*pi`), slewed `<= 0.25` rad/tick via `algebraic_unit_rotor_f32` |
| `scat_psi_fp16` | `[-pi, pi]` rad | Base SO(4) paraunitary scattering angle with non-finite fallback 0.0 stored in cold quad metadata outside the 216-byte active-hot state block; evaluates effective angle `(1 - freeze_mix) * base_scat_psi` from unlocked continuous `freeze_mix = clamp(base_freeze_mix + t[16], 0.0, 1.0)` via `algebraic_unit_rotor_f32` into cached `(cos_scat, sin_scat)` on cold load, reset, and register latches `(sample_tick & 3) == 0 || ctrl_sub_counter == 0` during slot-16 ramps so releasing `HOST_WRITE_LOCK` never retains stale `(1.0, 0.0)` |
| `input_gains_fp16[0..4]` | `[-2.0, 2.0]` | Per-lane input injection weights into 4-line Householder reflection |
| `wet_tap_l_fp16`, `wet_tap_r_fp16` | `[-2.0, 2.0]` | Stereo per-lane L/R wet tap weights |
| `r_dc` | `[0.95, 0.9999]` | Finite inputs clamp to `[0.95, 0.9999]` before ZOH `r_dc^rho_vm`; only non-finite inputs fall back to `0x3F7F_D51F` (5 Hz at 48 kHz) |
| `damp_pole` | `[0.0, 0.9995]` at 48 kHz; `[0.0, 0.99995]` eff | Cold ZOH `(base <= 1e-7) ? 0.0 : clamp(base^rho_vm, 0.0, 0.99995)`; 3 kHz `damp_pole_eff = clamp(damp_pole_zoh + op_t_damp_pole_delta, 0.0, 0.99995)` |
| `lfo_rate_hz`, `ramp_rate_hz` | `[0.0, 16.0]` Hz / `[-16.0, 16.0]` Hz | Advances `u32` `lfo_inc_q32` LFO quadrature and signed `i32` `ramp_inc_q32` ramp accumulator |
| `lfo_depths_fp16[0..4]` | `[0.0, min(512.0, floor(cap / 4) as f32)]` | ZOH-rescales by `1 / rho_vm` on cold load before clamping to `[0.0, min(512.0, floor(cap / 4) as f32)]` |
| `window_depth_fp16` | `[0.0, 1.0]` | Base Hermite-Hann diagonal window blend before boundary-click safety clamp (§2) |

Calling quad hot-state `reset()` restores deterministic zero-state resting equilibrium across all hot and external quad buffers as specified below:

| Quad `reset()` Field | Reset Value | Scope & Lock Semantics |
| :--- | :--- | :--- |
| `dc_x_prev`, `dc_y_prev`, `damp_state` | 0 (`[0.0; 4]`) | Zeroes loop DF-I DC blocker and per-lane HF damping state |
| `wet_dc_x`, `wet_dc_y` | 0 (`[0.0; 2]`) | Zeroes stereo non-recirculating wet DF-I DC blocker state |
| `write_ptr`, `lfo_phase_q32`, `ramp_phase_q32` | 0 | Resets backward write cursor and LFO/ramp phase accumulators |
| `current_psi` | `base_psi` | Snaps slewed Givens angle directly to `base_psi` |
| `(cos_psi, sin_psi)` | `algebraic_unit_rotor_f32(base_psi)` | Recomputes cached Givens rotor and cold effective `(cos_scat, sin_scat)` at `(1 - freeze_mix) * base_scat_psi` from unlocked continuous `freeze_mix = clamp(base_freeze_mix + t[16], 0.0, 1.0)` directly via pure-32-bit `algebraic_unit_rotor_f32` |
| `we_hold_off` | `[0; 4]` | Clears inline/cold hold counters |
| `flags` (`HOST_WRITE_LOCK`) | Preserved on `reset()` | Preserves bit 2 (`HOST_WRITE_LOCK`) on `reset()`; full VM re-init clears all lock bits |
| Inline, External & Schroeder Rings | 0.0 | Zero-fills main delay ring and Schroeder all-pass buffer (`ap_wr = 0`) |

Inside each active transport quad, `lfo_phase_q32` and `ramp_phase_q32` execute in evaluate-then-advance order, quantizing increments via cold 6-step Newton reciprocal `inv_fs_vm = 1.0 / f_{s,vm}` as `lfo_inc_q32 = (rate <= 0.0 || max_k(lfo_depth_eff[k]) <= 0.0) ? 0 : ((f64(rate) * 2^32 * inv_fs_vm + 0.5) as u64) as u32` and signed `ramp_inc_q32 = round_half_away(f64(ramp_rate) * 2^32 * inv_fs_vm) as i64 as i32`, where `round_half_away(x) = (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5)`.

Per-lane LFO output $s_{\phi,k}$ evaluates the zero-at-origin Hermite-5 **sine** curve `q32_smooth_sin(lfo_phase_q32 + (u32(lfo_phase_off_q16[k]) << 16))` (hitting 0.0 at phase 0, +1.0 at `0x4000_0000`, 0.0 at `0x8000_0000`, and -1.0 at `0xC000_0000`), while sawtooth phase $\phi_{\mathrm{ramp},k} \in [0, 1)$ evaluates at `ramp_phase_q32 + (u32(ramp_phase_off_q16[k]) << 16)` ahead of wrapping-adding `lfo_inc_q32` and signed `ramp_inc_q32`.

Each VM frame combines cold ZOH-rescaled `delay_samples[k]`, clamped octave scaling, LFO and ramp sweeps `ramp_spans_samples[k] * (phi_ramp,k - 0.5)`, and `tank_delay_mod` (`stage_targets[12]` in 48 kHz samples converted via `1 / rho_vm`), clamped inside $[5.0, \tau_{\max}]$:

$$\tau_k[n] = \mathrm{clamp}\left(D_k \cdot 2^{\mathrm{clamp}(\Delta_{\mathrm{oct}}, -3.0, +3.0)} + d_{\mathrm{lfo},k} s_{\phi,k} + S_{\mathrm{span},k}\big(\phi_{\mathrm{ramp},k} - 0.5\big) + \Delta_{\mathrm{ctrl,48k}} \cdot \frac{1}{\rho_{\mathrm{vm}}},\ 5.0,\ \tau_{\max}\right)$$

Here $\Delta_{\mathrm{oct}} =$ `stage_targets[17]` (`op_t_delay_scale_oct`) clamps before exponentiation as `2^clamp(stage_targets[17], -3.0, +3.0)`, while $\Delta_{\mathrm{ctrl,48k}} =$ `stage_targets[12]` rescales 48 kHz samples into VM frames. Upper bound $\tau_{\max} = \mathtt{cap} - 3$ applies when modulated (`lfo_depths`, `ramp_spans`, `|stage_targets[17]| > 1.0e-6`, or $|\Delta_{\mathrm{ctrl,48k}}| > 10^{-6}$ nonzero), and $\tau_{\max} = \mathtt{cap} - 2$ otherwise.

For each lane $\tau_k = D_{\mathrm{int},k} + \mu_k$, modulated Catmull-Rom readout blends linear and cubic interpolation via `half_alpha_cubic_fp16`, denoted $\alpha_{\mathrm{half}}[g]$, computed at cold load and 3 kHz ticks from `g_loop,eff` using a 2-step Newton-Raphson reciprocal of `max(|g_loop,eff|, 0.5)` and scaled by `(1.0 - mu_freeze)` so passive freeze reaches lossless non-expansive hold at $\mu_{\mathrm{freeze}} = 1.0$:

$$r_{\mathrm{inv}} = \mathrm{recipNR2}\big(\max(|g_{\mathrm{loop,eff}}|, 0.5)\big), \qquad \alpha_{\mathrm{cubic}} = \mathrm{clamp}\big((0.9992 \cdot r_{\mathrm{inv}} - 1.0) \cdot 4.0, 0.0, 1.0\big), \qquad \alpha_{\mathrm{half}}[g, \mu_{\mathrm{freeze}}] = 0.5 (1 - \mu_{\mathrm{freeze}}) \alpha_{\mathrm{cubic}}$$
$$y_{\mathrm{lin}} = (1 - \mu)s[0] + \mu s[1], \qquad y_{\mathrm{curv}} = (1 - \mu)\big(2s[0] - s[-1] - s[1]\big) + \mu\big(2s[1] - s[0] - s[2]\big)$$
$$r_{\mathrm{raw},k}[n] = y_{\mathrm{lin}} + \alpha_{\mathrm{half}}[g, \mu_{\mathrm{freeze}}] \cdot \mu (1 - \mu) y_{\mathrm{curv}}$$

The unified Top-of-Frame 4-lane pipeline then multiplies each lane by its continuous diagonal window gain $w_k \in [0, 1]$ (§2) and steps the fused `SO(4)` matrix Schroeder diffuser **before** tap weighting and loop feedback write across Quads 0..3 up to 16 stages at Ultra (§3.1):

$$\mathbf{H}(z;\ \theta_{\mathrm{eff}}, \mathbf{a}, \mathbf{D}) = \mathbf{R}_{\mathrm{odd}}(\theta_{\mathrm{eff}}) \mathrm{diag}(A_0, A_1, A_2, A_3) \mathbf{R}_{\mathrm{even}}(\theta_{\mathrm{eff}})$$

Here $\theta_{\mathrm{eff}} = (1 - \mu_{\mathrm{freeze}})\theta_{\mathrm{scat,base}}$ and each lane uses feedback $(1 - \mu_{\mathrm{freeze}}) a_{\mathrm{diff},k}$.

Whenever not in full hold, the fused diffuser executes unconditionally from learned weights with zero discrete stage-bypass flags, producing automatic 0-ULP identity when `scat_psi == 0.0` and `allpass_delays == [0; 4]`.

On full hold (`mu_freeze >= 1.0 - 1.0e-6` or bit 2 `HOST_WRITE_LOCK` set), the entire fused diffuser including `R_even` and `R_odd` bypasses cleanly (`v_read,k = r_win,k`), freezing the Schroeder buffer and `ap_wr` at 0 ULP.

When active, any lane with zero `allpass_delays` evaluates $A_k(z) = 1$ with zero cold ring accesses while `R_even` and `R_odd` execute; when all four delays are 0, `H(z)` collapses to lossless memoryless `SO(4)` rotation, preserving 0-ULP identity at zero scattering angle.

Feedback conditioning applies cold-precomputed `inv_sigma_23 = 2 / (3 * sigma_sat)` and `sigma_sat = clamp(sigma_knee, 1.0e-3, 0.984375)` (`TANK_MAX_SIGMA_SAT_FP16 = 0x3BE0`) in one unity-slope cubic saturator (`TANK_MAX_V_CEIL_BITS = 0x3F7C_28F5`, 0.985, zero loop `NR2` reciprocal) ahead of pairwise 2×2 Givens `R_block2(psi)` and unitary 4×4 Householder reflection $\mathbf{H}_4(\mathbf{1}) = \mathbf{I}_4 - 0.5\mathbf{1}\mathbf{1}^\top$ plus input injection.

On adjacent 2×2 lane pairs `(0, 1)` and `(2, 3)`, `R_block2(psi)` acts on $\mathbf{v} = [v_0, v_1, v_2, v_3]^\top$ via `v * cos(psi) + [-v_1, v_0, -v_3, v_2] * sin(psi)`:

$$\mathbf{R}_{\mathrm{block2}}(\psi)\mathbf{v} = \big[v_0\cos\psi - v_1\sin\psi,\ v_0\sin\psi + v_1\cos\psi,\ v_2\cos\psi - v_3\sin\psi,\ v_2\sin\psi + v_3\cos\psi\big]^\top$$

Immediately after $\mathbf{v}_{\mathrm{fb}}$ and before the loop DF-I DC blocker, where $\alpha_{\mathrm{dc}} = 0.5(1 + r_{\mathrm{dc}})$, per-lane 1-pole HF damping applies `v_damp[j] = (1 - damp_pole_eff) * v_fb[j] + damp_pole_eff * damp_state[j]` and `damp_state'[j] = flush(v_damp[j])`. Setting `damp_pole_eff == 0.0` yields identity pass-through into `y_dc[n]`, passive recirculation blends `w_ring = (1 - mu_freeze) * y_dc + mu_freeze * r_win` into both `fp16` and `f32` rings, and writes saturate to ±65,504.0 on `fp16` lines:

$$\mathbf{v}_{\mathrm{loop}} = \mathcal{S}_{\mathrm{ceil}}\big(\mathbf{v}_{\mathrm{read}}; g_{\mathrm{loop,eff}}, \sigma_{\mathrm{sat}}\big), \qquad \mathbf{v}_{\mathrm{fb}} = \mathbf{H}_4(\mathbf{1})\mathbf{R}_{\mathrm{block2}}(\psi)\mathbf{v}_{\mathrm{loop}} + \mathbf{g}_{\mathrm{inj}} \odot [x_L, x_R, x_L, x_R]^\top$$
$$\mathbf{v}_{\mathrm{damp}}[n] = (1 - p_{\mathrm{damp}})\mathbf{v}_{\mathrm{fb}}[n] + p_{\mathrm{damp}}\mathbf{d}_{\mathrm{state}}[n-1], \qquad \mathbf{d}_{\mathrm{state}}[n] = \mathrm{flush}\big(\mathbf{v}_{\mathrm{damp}}[n]\big)$$
$$y_{\mathrm{dc}}[n] = \mathrm{flush}\Big(\alpha_{\mathrm{dc}}\big(v_{\mathrm{damp}}[n] - v_{\mathrm{damp}}[n-1]\big) + r_{\mathrm{dc}} y_{\mathrm{dc}}[n-1]\Big), \qquad \mathbf{w}_{\mathrm{ring}}[n] = (1 - \mu_{\mathrm{freeze}})\mathbf{y}_{\mathrm{dc}}[n] + \mu_{\mathrm{freeze}}\mathbf{r}_{\mathrm{win}}[n]$$

Here $\mathcal{S}_{\mathrm{ceil}}(u; g, \sigma_{\mathrm{sat}}) = \mathrm{clamp}(g \sigma_{\mathrm{sat}} \cdot v(1.5 - 0.5 v^2), -V_{\mathrm{ceil}}, V_{\mathrm{ceil}})$ with $v = \mathrm{clamp}(u \cdot \mathrm{inv-sigma-23}, -1, 1)$, $\mathrm{inv-sigma-23} = 2/(3\sigma_{\mathrm{sat}})$, $V_{\mathrm{ceil}} \approx 0.985$ (`TANK_MAX_V_CEIL_BITS = 0x3F7C_28F5`), $p_{\mathrm{damp}} =$ `damp_pole_eff`, and $\mathbf{d}_{\mathrm{state}} =$ `damp_state`.

All `P_Q` active `TransportQuad` blocks receive `x_tank = (direct_s2_l + conv_l, direct_s2_r + conv_r)` with $x_L = x_{\mathrm{tank},L}$ and $x_R = x_{\mathrm{tank},R}$. On each pure-wet quad $q \in [0, P_Q)$, weighted taps form stereo sum $y_{\text{raw},L} = \sum_k w_{L,k} v_k$ and $y_{\text{raw},R} = \sum_k w_{R,k} v_k$, rotate through unitary 2×2 Givens matrix $\mathbf{R}(\psi)$, and step the 5 Hz stereo wet Direct-Form I DC blocker into pure-wet output $y_{\mathrm{wet},q,c}$:

$$p_{\mathrm{wet}} = \mathrm{clamp}\big(p_{\mathrm{dc,base}}^{\rho_{\mathrm{vm}}}, 10^{-4}, 0.999995\big), \qquad \alpha_{\mathrm{wet}} = \frac{1 + p_{\mathrm{wet}}}{2}, \qquad y_{\mathrm{wdc},c}[n] = \mathrm{flush}\Big(\alpha_{\mathrm{wet}}\big(y_{\mathrm{wet},c}[n] - y_{\mathrm{wet},c}[n-1]\big) + p_{\mathrm{wet}} y_{\mathrm{wdc},c}[n-1]\Big)$$

Here `wet_dc_pole`, denoted $p_{\mathrm{wet}}$, reads valid metadata `dc_blocker_pole_5hz` in `(0.8, 1.0)` as `p_dc_base` (else `0x3F7F_D51F`) and keeps it verbatim when `|rho_vm - 1.0| < 1e-12`, otherwise evaluating cold `powf_f64` per [router_and_crossbar_topology.md §9.1](router_and_crossbar_topology.md#91-causal-crossbar-sensoryquad-core-tpt-svf--kautz-conversion).

Stage-level dry pass-through `tank_dry_mix_fp16` (slot 22, decoded verbatim with `0x0000` -> 0.0 and non-finite -> 1.0) combines symmetrically with the sum of all `P_Q` pure-wet `TransportQuad` outputs:

$$y_{\text{rot},L} = \cos(\psi) y_{\text{raw},L} - \sin(\psi) y_{\text{raw},R}, \qquad y_{\text{rot},R} = \sin(\psi) y_{\text{raw},L} + \cos(\psi) y_{\text{raw},R}$$
$$g_{\mathrm{dry,eff}} = \mathrm{clamp}\big(\mathrm{tank-dry-mix} + t_{22}, -2.0, +2.0\big), \qquad y_{\mathrm{tank},c} = g_{\mathrm{dry,eff}} x_{\mathrm{tank},c} + w_{\mathrm{scale}} \sum_{q=0}^{P_Q - 1} y_{\mathrm{wet},q,c}, \qquad c \in \lbrace L, R \rbrace$$

`ControlBus` slots 12..14 and 16..25 apply uniformly across every active quad (`q in 0..P_Q`), while slot 22 (`op_t_dry_mix_mod`) modulates the stage-level `g_dry_eff` scalar cleanly across all four tiers.

The `Transport` stage outputs $y_{\mathrm{tank},c}$ into `mono_out` folding, the 48-frame linear tier crossfade (`rem` counting down from 48 to 1 with `a = f32(rem) * (1.0 / 48.0)` per [router_and_crossbar_topology.md §8.1](router_and_crossbar_topology.md#81-canonical-midside-output-head-factorization)), polarity-signed `eff_trim` (`WET_POLARITY_INV`), and ±1.0e9 VM safety clamping ahead of host-shell M/S output width.

For Givens angle updates, 3 kHz ticks form `target_psi = clamp(base_psi + tank_psi_delta, -pi, +pi)`, strictly clamped and never wrapped modulo $2\pi$, slew `current_psi` by at most 0.25 rad/tick, and evaluate `(cos_psi, sin_psi)` via pure-32-bit `algebraic_unit_rotor_f32` ([op_m_modal_kautz.md §1.2.4](op_m_modal_kautz.md#124-q0192-control-lfo--4-lane-q032-crossbar-phasors)).

Continuous passive `freeze_mix` (`mu_freeze = clamp(base_freeze_mix + t[16], 0.0, 1.0)` via uniformly micro-ramped slot 16 `op_t_freeze_mix_delta`, or forced to 1.0 whenever low-level host `HOST_WRITE_LOCK` bit 2 is set in `flags`) governs sound-on-sound recirculation vs 0-ULP ring hold without discrete Schmitt hysteresis.

Across (0, 1), tapering `lfo_depths`, `ramp_spans`, `half_alpha_cubic`, Schroeder `allpass_g`, and `scat_psi` by `(1 - mu_freeze)` keeps writeback `w_ring = (1 - mu_freeze) * y_dc + mu_freeze * r_win` non-expansive across both fp16 and f32 rings. Whenever full hold engages (`mu_freeze >= 1.0 - 1.0e-6` or host write lock bit 2 set), phase increments halt while live input processing and wet readout (wet taps, Givens $\psi$, wet DC, stage dry mix) continue executing:

$$D_k = \mathrm{clamp}\big(\mathrm{round}(D_{\mathrm{delay}}[k]), 5, \mathtt{cap} - 2\big), \qquad \mathrm{lag}_k = \mathrm{select}(o_k < D_k, D_k - o_k, D_k), \qquad o_k' = \mathrm{select}(o_k + 1 \ge D_k, 0, o_k + 1)$$

During full hold, each lane $k \in \lbrace 0, 1, 2, 3 \rbrace$ reads integer lag `(write_ptr + lag_k) & mask` using `o_k = we_hold_off[k]` at $w_k = 1.0$ (or cold 32-bit counters when `HAS_EXT` bit 1 is set) and advances $o_k'$ branchlessly. All four `we_hold_off` counters reset to 0 on `reset()` and on hold release.

| Hold Condition (`mu_freeze >= 1.0 - 1.0e-6` or `HOST_WRITE_LOCK`) | Hold Behavior & Precision Invariant |
| :--- | :--- |
| LFO & Ramp Phase Accumulators | `lfo_phase_q32` and `ramp_phase_q32` hold with zero increment |
| Main Ring Read & Window | Reads `(write_ptr + lag_k) & mask` at `w_k = 1.0`; advances `we_hold_off[k]` branchlessly |
| `SO(4)` Schroeder All-Pass Diffuser | Bypasses fused `SO(4)` diffuser evaluation (`v_read,k = r_frozen,k`); lane-major SoA Schroeder buffer and `ap_wr` freeze at 0-ULP |
| Feedback & Ring Writeback | Skips main ring `write_ptr` writes, loop feedback, HF damping (`damp_state`), and loop DC (`dc_x_prev`, `dc_y_prev`) |
| Non-Recirculating Wet Readout | Continues `wet_tap_l/r`, Givens `psi`, wet DC blocker, and stage `g_dry_eff * x_tank` |

#### 1.1.2 Cold-Path Capacity Sizing

`r[dsp.op-t.ring.capacity]`

At effective sample rate $f_{\mathrm{eff}} = M 2^{e-52}$, cold initialization computes exact integer ceiling capacity $D_{\mathrm{eff}}$ in 128-bit unsigned integer arithmetic without floating-point rounding:

$$D_{\mathrm{eff}} = \left\lceil \frac{D_{\mathrm{ref}} f_{\mathrm{eff}}}{48000} \right\rceil, \qquad L = D_{\mathrm{eff}} + 2, \qquad B = 64 \left\lceil \frac{aL}{64} \right\rceil$$

The 2-sample guard preserves the oldest 4-tap Catmull-Rom neighbor while maintaining $\tau[n] \ge 5.0$ on Top-of-Frame crossbar/transport quads (and $\tau[n] \ge 4.0$ on standalone single-sample rings). Crossbar reads finish across all active heads before end-of-frame writes update the ring.

#### 1.1.3 Host Memory-Ceiling Admission

`r[dsp.op-t.ring.admission]`

Total instance and crossfade buffer requirements must pass host memory ceiling checks on the cold path before activation, returning an explicit error on overflow without truncating delay history or dropping tails.

#### 1.1.4 Forward-Wrap Standalone Delay & Backward-Wrap Transport Quad Addressing

`r[dsp.op-t.ring.branchless-wrap]`

Standalone delay lines and compact transport quads implement complementary division-free ring addressing while preserving identical chronological neighborhood ordering `[s[-1], s[0], s[1], s[2]]` at lags `[D_int - 1, D_int, D_int + 1, D_int + 2]`:

| Mode & Structure | Write Cursor Advance | Read Index Formula | Tap Offset `o` for `s[-1], s[0], s[1], s[2]` | Causal `D_int` Bounds |
| :--- | :--- | :--- | :--- | :--- |
| Forward-Wrap Standalone Delay | `w' = (w + 1) - (L & -[w + 1 == L])` | `i = w - D_int + o; r = i + (L & -[i < 0])` | `[+1, 0, -1, -2]` (newer +1, older -2) | `4 <= D_int <= L - 2` |
| Backward-Wrap Transport Quad | `ring[wr] = sat_write(w_ring); wr' = (wr - 1) & mask` | `r = (wr + D_int + o) & mask` | `[-1, 0, +1, +2]` (newer -1, older +2) | `D_int in [5, cap - 2]` (`cap - 3` modulated) |

For standalone delay lines with write cursor $0 \le w < L$, integer delay $4 \le d \le L - 2$, and tap offset $o \in \lbrace +1, 0, -1, -2 \rbrace$ yielding signed offset $i = w - d + o \in [-L, L)$:

$$r = i + \big(L \mathbin{\mathrm{AND}} -[i < 0]\big), \qquad w' = (w + 1) - \big(L \mathbin{\mathrm{AND}} -[w + 1 = L]\big)$$

In compact transport quads, the End-of-Frame write `ring[wr] = sat_write(w_ring)` executes strictly *before* decrementing `(wr - 1) & mask`, placing sample lag $k \ge 1$ at `(wr + k) & mask` (`mask = cap - 1`). Reading `(wr + D_int + o) & mask` inside `[5, cap - 3]` keeps every 4-tap Catmull-Rom load at least 4 samples behind the write cursor, preventing 4-frame micro-chunk collisions while unmodulated `D_int = cap - 2` forces zero fraction so tap `o = +2` carries zero weight.

#### 1.1.5 Integer-Plus-`Q0.32` Delay Pointer & Causal Reader Bounds

`r[dsp.op-t.ring.q0-32-pointer]`

Standalone delay positions store an integer sample lag $D_{\mathrm{int}}$ plus an unsigned `Q0.32` fractional word rather than a single `f32` total delay, preventing sub-sample quantization drift on multi-second lines. Converting only the fractional part via $\mu = \mathtt{f32}(\mathrm{frac}) \cdot 2^{-32} \in [0.0, 1.0)$ carries any rounding overflow 1.0 to $(D_{\mathrm{int}} + 1, 0.0)$ and clamps total lag strictly inside `[4.0, L - 2.0]` for standalone rings.

For Top-of-Frame crossbar and transport quads executed in 4-frame causal micro-chunks, clamping `delay_int >= 5` (`tau_min = 5.0`) keeps `i_m1 = write_idx + 1 - delay_int <= write_idx - 4` on every step, preserving exact 0-ULP equivalence between 1-frame and 4-frame micro-chunk stepping.

#### 1.1.6 `fp16` Write Saturation Telemetry

`r[dsp.op-t.ring.fp16-saturation]`

Finite delay writes into 2 B (`fp16`) rings exceeding ±65,504.0 (`±FP16_MAX_FINITE_F32`) saturate to ±65,504.0 (`0x7BFF` / `0xFBFF`) and latch `FP16_SATURATED` ([sona_architecture.md §1.2](../architecture/sona_architecture.md#finite-precision-range-and-saturation-policy)).

---

### 1.2 Stateless 4-Tap Catmull-Rom Hermite Cubic Delay Reader

#### 1.2.1 Horner Cubic Evaluation & Passivity Invariants

`r[dsp.op-t.reader.catmull-rom]`

Executed at the Top-of-Frame Causal Crossbar, standalone Catmull-Rom reading splits $\tau[n] \ge 4.0$ into integer lag $D_{\mathrm{int}}[n] = \lfloor \tau[n] \rfloor$ and fraction $\mu[n] = \tau[n] - D_{\mathrm{int}}[n] \in [0.0, 1.0)$, loading the four neighbor samples `[s[-1], s[0], s[1], s[2]]` before writing frame $n$:

$$s[-1] = \mathrm{ring}\big[\mathrm{addr}(n - D_{\mathrm{int}}[n] + 1)\big], \qquad s[0] = \mathrm{ring}\big[\mathrm{addr}(n - D_{\mathrm{int}}[n])\big]$$
$$s[1] = \mathrm{ring}\big[\mathrm{addr}(n - D_{\mathrm{int}}[n] - 1)\big], \qquad s[2] = \mathrm{ring}\big[\mathrm{addr}(n - D_{\mathrm{int}}[n] - 2)\big]$$

Horner evaluation computes the cubic spline with zero `FDIV` operations:

$$d(\tau[n]) = \big((c_3' \mu[n] + c_2') \mu[n] + c_1'\big) \mu[n] + c_0'$$
$$c_3' = -0.5 s[-1] + 1.5 s[0] - 1.5 s[1] + 0.5 s[2], \qquad c_2' = s[-1] - 2.5 s[0] + 2.0 s[1] - 0.5 s[2]$$
$$c_1' = -0.5 s[-1] + 0.5 s[1], \qquad c_0' = s[0]$$

| Invariant | Mathematical Property |
| :--- | :--- |
| 1. Endpoint Exactness | At `μ = 0`, `d(τ) = s[0]`; at `μ = 1`, `c₃' + c₂' + c₁' + c₀' = s[1]`; at full freeze hold (`mu_freeze >= 1.0 - 1e-6` or `HOST_WRITE_LOCK` bit 2 set) with integer lag, exact 0-ULP hold |
| 2. C¹ Spline Continuity | Matches first derivative `0.5(s[2] - s[0])` across integer lag steps, giving over 15× lower LFO jerk than Thiran |
| 3. Stationary Energy Passivity | Satisfies `‖H_μ‖_∞ <= 1` for every fixed fractional phase `μ` |
| 4. Bounded Peak Kernel Norm | Instantaneous `L₁` norm is `1 + μ(1 - μ) <= 1.25` (max at `μ = 0.5`); feedback loops enforce decay via §1.3 |

#### 1.2.2 Contiguous 4-Lane Half-to-Single Stereo Fast Path

`r[dsp.op-t.reader.contiguous-fp16x4]`

In partitioned stereo delay lines (with per-channel ring capacity `L_ch = L_ring / 2`), whenever the four Catmull-Rom taps do not straddle ring index 0 (`p_0 >= 2` and `p_0 + 1 < L_ch`), stereo delay reading loads contiguous 4-element `fp16` tuples per channel and converts them via 4-lane half-to-single float vector conversion or portable bit-level decoding. Wrapped per-tap loads execute only on boundary frames, while 4 B `f32` rings use direct 4-lane loads ([ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)).

---

### 1.3 Bounded-History Feedback Guard & Loop Conditioning

#### 1.3.1 Coercive Loop Limiter, Soft-Knee & Cubic Saturator

`r[dsp.op-t.guard.bounded-history]`

Because time-varying Catmull-Rom interpolation can reach 1.25× instantaneous gain, standalone block max-tree guards and compact `TransportQuad`s enforce strict contractivity (`||G_loop|| < 1`) using complementary loop-conditioning paths while skipping ring overwrites whenever full freeze hold (`mu_freeze >= 1.0 - 1.0e-6` or `HOST_WRITE_LOCK` bit 2) is active:

In standalone delay lines with block max-tree guards, active writes clamp against the live binary max-tree bound $M_n$ and apply a coercive 4-lane cubic loop limiter with division-free reciprocal $r_\sigma = \sigma^{-1}$ computed via a 3-step Newton reciprocal:

$$w_i[n] = x_i[n] + g_i[n] \cdot \mathrm{clamp}\big(v_i[n], -M_n, M_n\big), \qquad 0 \le g_i[n] \le g_{\max} < 1$$
$$v_{\text{knee}} = \mathrm{clamp}\left(\frac{2}{3} x_{\text{finite}} r_\sigma, -1, 1\right), \qquad y_{\text{lim}} = \mathrm{flush}\Big(g \cdot \mathrm{clamp}\big(\sigma v_{\text{knee}}(1.5 - 0.5 v_{\text{knee}}^2), -\sigma, \sigma\big)\Big)$$

In compact TransportQuads, end-of-frame feedback clamps `g_loop` by `0x3F7F_BE76` (0.9990) and evaluates the unified `sigma_sat` cubic saturator (`TANK_MAX_SIGMA_SAT_FP16 = 0x3BE0`, cold-precomputed `inv_sigma_23 = 2 / (3 * sigma_sat)`, `TANK_MAX_V_CEIL_BITS = 0x3F7C_28F5` = 0.985, zero loop NR2 reciprocal) before Givens/Householder mixing (§1.1.1).

#### 1.3.2 Grouped Max-Tree Tracking

`r[dsp.op-t.guard.max-tree]`

The block max-tree guard partitions standalone delay rings into contiguous blocks of at most $H = 64$ samples, maintaining block maxima in a binary max-tree paired with an active-block running maximum.

#### 1.3.3 Division-Free Incremental `(block_idx, in_block_offset)` Counters

`r[dsp.op-t.guard.incremental-counters]`

To eliminate hardware integer division (`pos / block_size` or `pos % block_size == 0`) on arbitrary ring lengths, the block max-tree guard keeps pre-allocated `block_indices` and `in_block_offsets` counters per line and advances branchlessly (`offset + 1 == block_size` or `pos == 0`, [ADR-0019](../../adr/0019-simd-kernel-architecture-and-realtime-execution-optimizations.md)).

#### 1.3.4 Driven BIBO & Unforced Geometric Decay

`r[dsp.op-t.guard.bibo-decay]`

Under bounded input $|x_i[n]| \le B$, history magnitude stays bounded by $C = \max(M_0, B / (1 - g_{\max}))$; under zero input $x_i[n] = 0$, $M_n$ decays by at least $g_{\max}$ every $L_{\max} + H$ frames regardless of read-head velocity.

#### 1.3.5 Directed `fp16` Decay Rule

`r[dsp.op-t.guard.fp16-decay]`

Capping loop feedback at `0x3F7F_BE76` (`TANK_MAX_G_LIN_BITS`, 0.9990) and rounding `fp16` delay writes toward zero forces subnormal ring contents to decay strictly to zero without nonzero `fp16` limit cycles.

---

### 1.4 Dual-Mode Runtime Delay Transitions (`Range` Slew vs. `Switch` Crossfade)

`r[dsp.op-t.transition.dual-mode]`

| Control Event | Transition Law | Physical & Acoustic Role |
| :--- | :--- | :--- |
| Continuous `range` Knob Sweep | 1-pole exponential delay slew bounded at 0.25 samples/sample | Models smooth analog BBD or tape Doppler pitch glide |
| Discrete `switch` Delay Jump | Hold endpoints and crossfade over `max(16, round(128 / rho_vm))` frames (128 at 48 kHz) | Prevents clicks and unmodeled pitch sweeps on discrete jumps |

For crossfade step $k \in \lbrace 0, \dots, L_{\text{xfade}} - 1 \rbrace$, cold preparation synthesizes `alpha_xf[k]` without `FDIV` or external math libraries using the Cody-Waite degree-13/14 rotor `algebraic_unit_rotor_f32` ([op_m_modal_kautz.md §1.2.4](op_m_modal_kautz.md#124-q0192-control-lfo--4-lane-q032-crossbar-phasors)) and blends endpoints in unfused `binary32` (`FP_CONTRACT OFF`):

```text
inv_denom   = f32(rcp_nr6_f64(f64(L_xfade - 1)))
w[k]        = f32(k) * inv_denom                                         // w[0] == 0.0 -> alpha_xf[0] == 0.0
alpha_xf[k] = 0.5 * (1.0 - algebraic_unit_rotor_f32(w[k] * PI_F32).0)  // PI_F32 = 0x4049_0FDB, FP_CONTRACT OFF
d[n]        = (1.0 - alpha_xf[k]) * d_old[n] + alpha_xf[k] * d_new[n]    // unfused binary32; snaps to d_new once k + 1 >= L_xfade
```

While a fade runs, its two endpoints stay frozen and newer accepted delay changes coalesce into one pending target governed by [host automation timing](parametric_control_bus.md#112-host-automation-timing), latching on the first sample after completion while honoring conditional batch bounds ([parametric_control_bus.md §2.4.2](parametric_control_bus.md#242-conditional-batch-safety)).

---

## 2. Unified Continuous 4-Lane Windowing & Partition-of-Unity Regimes

`r[dsp.op-t.window.partition-of-unity]`

Every Op-T `TransportQuad` executes one single Top-of-Frame 4-lane pipeline (`Catmull-Rom -> Diagonal Window w_k -> Fused SO(4) Paraunitary Schroeder Diffuser`) continuously across all lanes without discrete mode selectors or conditional branching. Continuous parameterization spans static delays, chorus, TZF, passive `freeze_mix` overdub and `WE` RAM hold, dual- or quad-grain Hermite-Hann pitch/shimmer, staggered barberpole sweeps, colorless paraunitary FIR FFM diffusion, and coupled dispersive spring diffusion seamlessly:

```mermaid
flowchart LR
    Read["4-Lane Catmull-Rom Read (tau_min = 5.0)"] --> Fold["32-Bit Signed Tri Fold -> Hermite-Hann Diagonal w_k"]
    Fold --> Sch["Fused SO(4) Paraunitary Schroeder Diffuser"]
    Sch --> Out["Pure-Wet Readout + Coercive Loop (freeze_mix & Host Lock Gated)"]
```

For each lane $k \in \lbrace 0, 1, 2, 3 \rbrace$, adding the 16-bit ramp phase offset `ramp_phase_off_q16[k]` shifted into `Q0.32` forms per-lane phase $\phi_k = \mathrm{frac}(\phi_{\mathrm{ramp}} + \Delta\phi_{\mathrm{ramp},k})$. Interpreting the 32-bit unsigned phase word `p` as signed `p_signed` yields a branchless triangle fold identical on CPU SIMD and 32-bit GPU compute shaders:

$$\mathrm{mask} = p_{\mathrm{signed}} \gg 31, \qquad \mathrm{tri} = (p \oplus \mathrm{mask}) -_{\mathrm{u32}} \mathrm{mask}, \qquad u_k = \mathrm{f32}(\mathrm{tri}) \cdot 2^{-31} \in [0, 1]$$
$$h_k = u_k^2\big(3 - 2u_k\big), \qquad w_k = \big(1 - \mu_{\mathrm{win},k}\big) + \mu_{\mathrm{win},k} h_k$$

To eliminate boundary clicks when any nonzero ramp sweep wraps across a nonzero span, cold and 3 kHz setup evaluate per-lane $\mu_{\mathrm{win},k}$ directly in 4-lane registers from scalar `window_depth_fp16` and effective-rate sawtooth wrap jump $S_k$ in samples, where $S_k$ equals `ramp_spans_samples[k]` when $|f_{\mathrm{ramp,eff}}| > 10^{-6}$ Hz and 0.0 otherwise:

$$\mu_{\mathrm{win},k} = \max\Big(\mu_{\mathrm{win,base}},\ \mathrm{clamp}\big((|S_k| - 0.25) \cdot 4.0,\ 0.0,\ 1.0\big)\Big)$$

To prevent energy accumulation under downward pitch time-stretching, downward slope $\nu_{\mathrm{down},k}$ is positive when reading backward or slower to lower pitch and negative before clamping on upward pitch shimmers, leaving `g_loop` unattenuated on pitch-up shimmers:

$$\nu_{\mathrm{down},k} = \mathrm{clamp}\left(\frac{S_{\mathrm{span},k} \cdot f_{\mathrm{ramp,eff}}}{f_{s,\mathrm{vm}}},\ 0.0,\ 0.75\right)$$
$$g_{\mathrm{loop,eff}} = \min\big(g_{\mathrm{loop}},\ \mathtt{0x3F7FBE76}\big) \cdot \min_k \sqrt{\max\big(0.0,\ 1.0 - \min(0.9999,\ \nu_{\mathrm{down},k})\big)}$$

Cold and 3 kHz setup evaluate the square root without audio-loop `FSQRT` or `FDIV` via unfused scalar 2-step Newton inverse square root (`FP_CONTRACT OFF`):

```text
u_safe = max(u, 1.0e-12); h = 0.5 * u_safe; y0 = f32_bits(0x5F37_59DF - (bits(u_safe) >> 1))
y1 = y0 * (1.5 - h * (y0 * y0)); y2 = y1 * (1.5 - h * (y1 * y1))
sqrt_u = (u <= 0.0) ? 0.0 : (u * y2)
```

| Continuous Regime | `ramp_rate_hz` & `ramp_spans_samples` | `ramp_phase_off_q16[4]` & `window_depth` | Partition-of-Unity & Hold Behavior |
| :--- | :--- | :--- | :--- |
| Delay, Chorus, TZF & Passive Hold | `ramp_rate_hz == 0.0`, `ramp_spans == 0.0` | `window_depth == 0.0` (`S_k == 0.0`) | Diagonal gain `w_k == 1.0` across all 4 lanes; full hold (`mu_freeze >= 1.0 - 1e-6` or `HOST_WRITE_LOCK` bit 2) bypasses Schroeder (`v_read,k = r_frozen,k`) and locks writes at 0 ULP |
| Hermite-Hann Pitch & Shimmer | Nonzero sweep across grain span `G` | Complementary offsets (`[0, 0.5]` pairs or `[0, 0.25, 0.5, 0.75]`) | Exact `H(phi_A) + H(phi_B) == 1.000000` plus boundary and pitch-down clamps |
| Continuous Barberpole Sweep | Staggered sawtooth `ramp_spans != 0` | Quarter-turn or staggered offsets | Boundary clamp tapers wrap edges to zero; standalone 3-phase `B_2` sum `== 1.0` |

### 2.1 Delay, Chorus, Through-Zero Flanging & `WE` RAM Hold

`r[dsp.op-t.window.m1-delay]`

When `ramp_rate_hz == 0.0`, `ramp_spans_samples == 0.0`, and `window_depth == 0.0`, diagonal window weights simplify to $w_k \equiv 1.0$ with zero cosine LUTs. Anchoring tap 0 at static reference delay $\tau_{\text{ref}} = 64.0$ samples (1.33 ms at 48 kHz) while sweeping line 1 over $\tau_1[n] \in [16.0, 112.0]$ samples achieves Through-Zero Flanging with deep cancellation at $\Delta\tau = 0.0$ ms.

Engaging full passive hold (`mu_freeze >= 1.0 - 1e-6` or `HOST_WRITE_LOCK` bit 2) freezes `lfo_phase_q32`, `ramp_phase_q32`, main ring `write_ptr`, and Schroeder `ap_wr` at 0 ULP while bypassing Schroeder evaluation (`v_read,k = r_frozen,k`).

### 2.2 Centered Hermite-Hann Granular Pitch Shifter (`Family 3`)

`r[dsp.op-t.window.m2-pitch]`

For pitch offset $\Delta p \in [-24, +24]$ semitones, time-warp rate $\nu = 1 - 2^{\Delta p / 12} \in [-3.0, 0.75]$ derives from the [control-bus pitch profile](parametric_control_bus.md#51-pitch-ratio-and-time-warp-profile). Base-delay slew limits (0.25 samples/sample) govern center delay $D_c$ while complementary 2-grain or 4-grain `ramp_phase_off_q16` offsets sweep across even integer grain duration $G$:

$$\phi_A[n] = \mathrm{frac}(n/G), \qquad \phi_B[n] = \mathrm{frac}(\phi_A[n] + 1/2), \qquad \tau_j[n] = D_c + \nu G\big(\phi_j[n] - 1/2\big) \in \big[D_c - |\nu|G/2, D_c + |\nu|G/2\big]$$

Evaluating the branchless 32-bit triangle fold yields $u_A = 1 - |2\phi_A - 1| \in [0, 1]$ and complementary Hermite-Hann windows satisfying exact sum invariance and pitch-down loop contraction $g_{\text{loop,eff}}$:

$$H(\phi_A) = u_A^2\big(3 - 2u_A\big), \qquad H(\phi_B) = 1.0 - H(\phi_A), \qquad H(\phi_A) + H(\phi_B) \equiv 1.000000000000$$

For a 100 ms grain at 48 kHz with $G = 4{,}800$ and +24 semitones where $\nu = -3$, the modulated delay reaches $7{,}205 + 3 \cdot 4{,}800 / 2 = 14{,}405$ samples. Because $\tau_{\max} = \mathtt{cap} - 3$ when modulated, satisfying the 5-sample crossbar causal floor requires $D_c \ge 7{,}205$ samples and ring capacity $\ge 14{,}408$ samples (power-of-two `cap >= 16,384` in transport quads).

### 2.3 Continuous Barberpole Frequency Sweep (`Family 6`)

`r[dsp.op-t.window.m3-barberpole]`

Continuous barberpole modulation drives staggered sawtooth `ramp_spans_samples` across lanes 0..3 with staggered `ramp_phase_off_q16` offsets, where the boundary-click safety clamp automatically forces $\mu_{\mathrm{win},k} \to 1.0$ whenever $|S_k| \ge 0.5$ so wrap boundaries fade smoothly to zero. Standalone 3-head sweeps spaced by $1/3$ cycle apply precomputed raised-cosine or 3-piece cardinal quadratic B-spline windows without runtime `FDIV` or standard math library calls:

$$\sum_{m=0}^2 B_2\big(3 \cdot \mathrm{frac}(\phi + m/3)\big) \equiv 1.000000000000$$

Here $B_2(t) = 0.5 t^2$ on $[0, 1)$, $B_2(t) = -t^2 + 3t - 1.5$ on $[1, 2)$, and $B_2(t) = 0.5(3 - t)^2$ on $[2, 3)$.

---

## 3. Fused `SO(4)` Paraunitary Scattering & SoA Partitioned Schroeder All-Pass Diffusers

### 3.1 Memory Partitioning & Rate-Aware Capacity

`r[dsp.op-t.diffuser.partition]`

Per-lane tank diffusers execute as a fused `SO(4)` paraunitary matrix all-pass operator $\mathbf{H}(z;\ \theta_{\mathrm{eff}}, \mathbf{a}, \mathbf{D})$ across Quads `0..P_Q` up to 16 stages at Ultra (`|a_diff| <= 0.75`, `scat_psi` in `[-pi, +pi]`) before tap weighting and loop write, while standalone diffusers use 8-length coprime reference cascades (§3.2).

Active `TransportQuad` diffusers store cold rings in lane-major Structure-of-Arrays layout `[lane * cap_per_lane + idx]` across `4 * cap_per_lane` 32-bit floats (`cap_per_lane` power-of-two `>= max_delay + 1` whenever nonzero delays exist, or 0 for 0 B cold ring RAM strictly when `allpass_delays == [0; 4]`).

Under full hold (`mu_freeze >= 1.0 - 1.0e-6` or bit 2 `HOST_WRITE_LOCK`), execution bypasses `H(z)` cleanly (`v_read = r_win`); otherwise `H(z)` executes unconditionally purely from learned weights with zero discrete bypass flags.

When active, any lane with zero `allpass_delays[k]` sets $A_k(z) = 1$ with zero cold ring accesses while surrounding Givens rotations run.

Setting all four delays to 0 yields a lossless memoryless `SO(4)` rotation, preserving 0-ULP identity when $\theta_{\mathrm{eff}} = 0$.

Nonzero reference delays rescale via $D_{\text{ap},m}(f_{s,\text{vm}}) = \max(1, \mathrm{round}(D_{\text{ap},m}(48000) / \rho_{\text{vm}}))$ up to 16,384 samples ([router_and_crossbar_topology.md §9](router_and_crossbar_topology.md#9-continuous-time-zoh-rate-adaptation)); 223 samples at 48 kHz scale to 1,784 samples at 384 kHz, while `allpass_g` and `scat_psi` taper continuously by `1 - mu_freeze`.

During 4-frame causal micro-chunk stepping (`process_microchunk_4`), active execution when not in full hold (`!full_hold`) branches cleanly over three delay regimes matching 1-frame `step_vec4` at 0 ULP:

| Active Delay Regime (`!full_hold`) | Execution & Ring Pointer Advance (`cap_per_lane` power of 2 or 0) |
| :--- | :--- |
| **`allpass_delays == [0; 4]` (`cap_per_lane == 0`)** | Apply `R_even(theta_eff)` then `R_odd(theta_eff)` across all 4 frames in registers with zero ring memory accesses and no `ap_wr` update, guarding `cap_per_lane > 0` before forming `mask = cap_per_lane - 1` |
| **Nonzero `D_ap,k >= 4` (`cap_per_lane >= 8`)** | Execute 4-frame causal SoA `step_chunk4`: apply `R_even` via 4×4 SIMD transpose; per nonzero lane bind `r_idx = (ap_wr - D_k) & mask` and `write_pos = ap_wr`, load 16 B contiguously at `r_idx` if `r_idx + 4 <= cap_per_lane` else 2-piece wrap gather, step 4 recurrence frames, store 16 B contiguously at `write_pos` if `write_pos + 4 <= cap_per_lane` else 2-piece wrap store; apply `R_odd`, transpose back to frame-major, and advance `ap_wr' = (ap_wr + 4) & mask` |
| **Any `1 <= D_ap,k < 4`** | Fallback to 4 sequential 1-frame `step_vec4` calls advancing `ap_wr' = (ap_wr + 1) & mask` per frame (0-ULP block-size invariant) |

### 3.2 `SO(4)` Paraunitary Butterfly Algebra, Recurrence & Reverse-Time Adjoint

`r[dsp.op-t.diffuser.recurrence]`

Using `algebraic_unit_rotor_f32` to evaluate $(c_s, s_s)$ at effective angle $\theta_{\mathrm{eff}} = (1 - \mu_{\mathrm{freeze}})\theta_{\mathrm{scat,base}}$, define orthogonal Givens generators $\mathbf{J}_{\mathrm{even}}(\mathbf{x}) = [-x_1,\ x_0,\ -x_3,\ x_2]^\top$ and $\mathbf{J}_{\mathrm{odd}}(\mathbf{y}) = [y_3,\ -y_2,\ y_1,\ -y_0]^\top$:

$$\mathbf{x}_1 = \mathbf{R}_{\mathrm{even}}(\theta_{\mathrm{eff}})\ \mathbf{r}_{\mathrm{win}} = c_s \mathbf{r}_{\mathrm{win}} + s_s \mathbf{J}_{\mathrm{even}}(\mathbf{r}_{\mathrm{win}}), \qquad \mathbf{v}_{\mathrm{read}} = \mathbf{R}_{\mathrm{odd}}(\theta_{\mathrm{eff}})\ \mathbf{y}_1 = c_s \mathbf{y}_1 + s_s \mathbf{J}_{\mathrm{odd}}(\mathbf{y}_1)$$

Product $\mathbf{Q}(\theta) = \mathbf{R}_{\mathrm{odd}}(\theta)\mathbf{R}_{\mathrm{even}}(\theta) \in SO(4)$ is the 4×4 orthogonal circulant matrix with first row $[c_s^2,\ -c_s s_s,\ s_s^2,\ c_s s_s]$ commuting with the 4×4 Householder reflection $\mathbf{H}_4(\mathbf{1})$, giving determinant $+1$ and unit spectral norm $\Vert \mathbf{H}(e^{j\omega}) \Vert_2 \equiv 1$ across all $a_k \in [-0.75, 0.75]$, $D_k \ge 0$, and $\theta \in [-\pi, \pi]$.

It has period $\pi$ via $\mathbf{Q}(\theta + \pi) = \mathbf{Q}(\theta)$ with fundamental optimization domain $[-\pi/2, \pi/2]$, yields 0-ULP identity at $\theta = 0$, produces orthonormal $\pm 0.5$ Hadamard scattering at $\theta = \pi/4$, realizes a Schlecht–Dal Santo paraunitary FIR filter feedback matrix when `allpass_g == [0; 4]` with positive delays, and forms matrix-coupled dispersive chains when `allpass_g != 0` and $\theta \neq 0$.

Between $\mathbf{x}_1$ and $\mathbf{y}_1$, the 48 kHz standalone reference lengths `[19, 31, 47, 73, 103, 139, 179, 223]` are pairwise coprime; off-48 kHz rates keep those exact rounded lengths. Each active lane $m$ with $D_{\text{ap},m} > 0$ steps converted delay $D_{\text{ap},m}$ and clamped coefficient $a_{\text{diff},m} \in [-0.75, +0.75]$ (tapered by `1 - mu_freeze` on active `TransportQuad`s):

$$v_{\text{ap}, m}[n] = x_{\text{in}}[n] - a_{\text{diff}, m} u_{\text{ap}, m}[n - D_{\text{ap}, m}]$$
$$y_{\text{ap}, m}[n] = a_{\text{diff}, m} v_{\text{ap}, m}[n] + u_{\text{ap}, m}[n - D_{\text{ap}, m}], \qquad u_{\text{ap}, m}[n] = v_{\text{ap}, m}[n]$$

For reverse-time BPTT given $\mathbf{g}_{\mathrm{out}} = \nabla_{\mathbf{v}_{\mathrm{read}}}\mathcal{L}$, exact transposes use $\mathbf{R}_{\bullet}(-\theta_{\mathrm{eff}})$ with `(c_s, -s_s)` and contribute into both $\theta_{\mathrm{scat,base}}$ and the slot-16 $\mu_{\mathrm{freeze}}$ pullback whenever the diffuser is active:

$$\mathbf{g}_{\mathbf{y}_1} = \mathbf{R}_{\mathrm{odd}}(-\theta_{\mathrm{eff}})\ \mathbf{g}_{\mathrm{out}}, \qquad \mathbf{g}_{\mathbf{r}_{\mathrm{win}}} = \mathbf{R}_{\mathrm{even}}(-\theta_{\mathrm{eff}})\ \mathbf{g}_{\mathbf{x}_1}$$
$$\frac{\partial\mathcal{L}}{\partial\theta_{\mathrm{eff}}} = \mathbf{g}_{\mathrm{out}}^\top \mathbf{J}_{\mathrm{odd}}(\mathbf{v}_{\mathrm{read}}) + \mathbf{g}_{\mathbf{x}_1}^\top \mathbf{J}_{\mathrm{even}}(\mathbf{x}_1), \qquad \frac{\partial\mathcal{L}}{\partial\theta_{\mathrm{scat,base}}} \mathrel{+}= (1 - \mu_{\mathrm{freeze}})\frac{\partial\mathcal{L}}{\partial\theta_{\mathrm{eff}}}, \qquad \frac{\partial\mathcal{L}}{\partial\mu_{\mathrm{freeze}}} \mathrel{+}= -\theta_{\mathrm{scat,base}}\frac{\partial\mathcal{L}}{\partial\theta_{\mathrm{eff}}}$$

Here $\mathbf{g}_{\mathbf{x}_1}$ and $\partial\mathcal{L}/\partial a_{\mathrm{diff},m}$ propagate backward through each lane's Schroeder all-pass recurrence. For fixed $a_{\text{diff}}$ and $D_{\text{ap}}$, $|H_{\text{ap}}(e^{i\omega})|^2 \equiv 1$ across all frequencies; for bounded $|x_{\text{in}}[n]| \le A$ and $|a_{\text{diff}}| \le 0.75$, impulse peak gain stays bounded by $1 + 2|a_{\text{diff}}| \le 2.5A$ and internal state by $\max(V_0, 4A)$.

---

## 4. 1-Pole Ballistic Envelope Ducking VCA (`Family 4`)

`r[dsp.op-t.ducking.vca]`

In the SONA VM, compact 128 B wire / 216 B active-hot TransportQuads execute strictly pure-wet without per-quad VCA state (§1.1.1), routing ducking, gating, and sidechain expansion uniformly through SensoryQuad plus `ControlBus` slots 15 and 23 while stage `tank_dry_mix_fp16` (slot 22) blends direct pass-through ahead of `mono_out`.

Standalone generalized ducking VCA operators implement the 5-parameter `(tau_attack, tau_release, theta_duck, kappa_duck, g_duck_min)` formulation over `x_sidechain = 0.5 * (abs(in_l) + abs(in_r))`:

| Standalone Parameter | Clamped Domain / Floor | Physical Role |
| :--- | :--- | :--- |
| `tau_attack` | >= 0.5 ms (`alpha_att` in `[0.5, 0.9999]`) | Fast attack envelope coefficient on sidechain peaks |
| `tau_release` | >= 20.0 ms (`alpha_rel` in `[0.5, 0.99999]`) | Smooth release envelope recovery coefficient |
| `theta_duck` | `[0.0, 2.0]` | Sidechain ducking threshold |
| `kappa_duck` | `[0.0, 8.0]` | Compression attenuation slope |
| `g_duck_min` | `[0.05, 1.0]` | Minimum wet gain floor preventing total blackout |

$$e_{\text{duck}}[n] = \alpha_{\text{duck}}[n] e_{\text{duck}}[n-1] + (1 - \alpha_{\text{duck}}[n]) |x_{\text{sidechain}}[n]|$$
$$g_{\text{duck}}[n] = \mathrm{clamp}\Big(1.0 - \kappa_{\text{duck}}\big(e_{\text{duck}}[n] - \theta_{\text{duck}}\big), g_{\text{duck,min}}, 1.0\Big), \qquad y_{\text{wet,ducked}}[n] = g_{\text{duck}}[n] \cdot y_{\text{wet}}[n]$$

---

## 5. Strict Zero-`FDIV`, Zero-`FSQRT` & Zero Standard-Math-Library Verification

`r[dsp.op-t.zero-fdiv]`

| Operation | Naive Formulation | SONA Division-Free Factorization |
| :--- | :--- | :--- |
| Ring buffer wrap | `idx = pos % L_ring` (`IDIV`) | Branchless `i + (L & -[i < 0])` (standalone) or `& (cap - 1)` (`TransportQuad`) |
| Catmull-Rom read | Recursive Thiran all-pass (`FDIV`) | Stateless Horner cubic polynomial + 4-lane half-to-single / `f32` fast path (§1.2) |
| Passive `freeze_mix` & `WE` hold | Re-write `fp16` loop every sample | Continuous `mu_freeze` blend and host `HOST_WRITE_LOCK` increment `we_hold_off[4]` at 0 ULP (§1.1.1) |
| Unified 4-lane windowing | Real-time `cos(pi * phi)` + mode branching | Branchless 32-bit signed tri fold + Hermite `u^2 * (3 - 2u)` and B-spline `B_2` (§2) |
| Boundary & pitch-down guards | Per-sample `FDIV` / `FSQRT` | 3 kHz / cold computation of `S_k` safety clamp and `g_loop,eff` (§2) |
| Schroeder all-pass | Direct-form rational division | Coupled state recurrence `v = x - a*u`, `y = a*v + u` (§3.2) |
| Base-delay slew | Per-sample rate division | Precomputed 1-pole slew capped at 0.25 samples/sample (§1.4) |
| Switch crossfade | Real-time `cos(pi * k / L)` | Cached 6-step `f64` Newton reciprocal of `L_xfade - 1` + `algebraic_unit_rotor_f32` cosine rotor (§1.4) |
| Max-tree block index | `pos / block_size` (`UDIV`) | Incremental `(block_idx, in_block_offset)` counters (§1.3.3) |
