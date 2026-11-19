# `src/control` — $3\text{ kHz}$ Parametric Control Bus & Non-Stop 3-Phase Clamping

Implements the $3\text{ kHz}$ Control Bus (`#[forbid(unsafe_code)]`): 64-byte binary [`ControlClampDescriptor`](descriptor.rs) cache lines, anti-zipper slewing, Node-Anchored Lobatto HDMR + Rank-4 Factorization Machine, autonomous Chebyshev LFOs, envelope followers, and trust-region soft-clipping.

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`descriptor.rs`](descriptor.rs) | 64-byte `ControlClampDescriptor` / `ControlClampDescriptorWire` (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`) | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`bus.rs`](bus.rs) ([`tick.rs`](bus/tick.rs), [`tank_mod.rs`](bus/tick/tank_mod.rs), [`simd_and_clip.rs`](bus/tick/simd_and_clip.rs), [`cp1_and_uint.rs`](bus/cp1_and_uint.rs)) | `ControlBus<B>`, `tick_3khz`, `CP1` (`576 B` base / `2,112 B` Ultra), `U_int` | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`plane.rs`](plane.rs) ([`stride_and_ingress.rs`](plane/stride_and_ingress.rs), [`exact_int.rs`](plane/exact_int.rs), [`tempo.rs`](plane/tempo.rs), [`inversion.rs`](plane/tempo/inversion.rs)) | Stride alignment, `Q0.192` accumulators, `4768`-bit exact summations, tempo sync, and physical calibration inversion | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |
| [`reference.rs`](reference.rs) | Reference `f64` Lobatto, Rank-4 FM, Chebyshev LFO, and trust-region kernels | [`parametric_control_bus.md`](../../../../docs/specs/dsp/parametric_control_bus.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for control-basis and verification guidance.
