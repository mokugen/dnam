# Contributing to `dnam-vm` (`packages/vm/`)

This guide defines the safety, `#![no_std]` / `!alloc`, `zerocopy`, testing, and coverage requirements for **`dnam-vm`** ([`README.md`](README.md)). All contributions must also comply with the root [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`CLA.md`](../../CLA.md) (`git commit -s`).

---

## 1. Crate Architecture & Dependency Isolation

Consult [`AGENTS.md`](AGENTS.md) for the full module routing map:

- **`src/lib.rs`:** Crate root enforcing `#![cfg_attr(not(any(feature = "std", test)), no_std)]`, `#![deny(unsafe_code)]`, and `#![warn(unreachable_pub)]`.
- **`Cargo.toml` Dependencies:** Strictly limited to [`dnam-kernels`](../kernels/CONTRIBUTING.md) and `zerocopy` (`proptest` in `[dev-dependencies]` only). Zero `serde`, JSON parsers, filesystem/network crates, `crypto`, `wgpu`, or `64-bit` trainer traits.
- **Approved `unsafe` Scope (`ref[sdk.rust.safe-rust-and-boundaries]`):** `#[allow(unsafe_code)]` is restricted solely to hardware SIMD intrinsics under `src/math/{aarch64,x86}/**` and `FtzDazGuard` CSR registers in `src/types/fp_env.rs`. Both `src/bin/**` and `src/control/**` enforce `#[forbid(unsafe_code)]`.

---

## 2. Real-Time, `!alloc`, & `zerocopy` Invariants

1. **`--no-default-features` (`!std` & `!alloc`):** Every public VM container parser (`DnamBinView::parse`, `strip_dnam_bin_to_tier`) and `ControlBus::from_dnam_bin` + `ControlBusBacking::from_slices` must compile cleanly under `cargo check -p dnam-vm --no-default-features` and `wasm32-unknown-unknown` with zero heap allocation or `Drop` glue in real-time loops.
2. **Zero Stalls & Zero Transcendentals (`REQ-6.2`):** Non-test `src/**/*.rs` must emit zero hardware `FDIV`, `FSQRT`, `IDIV`/`UDIV`, `libm` relocations, or float `.rem_euclid()` calls.
3. **Declarative `zerocopy` Wire Structs (`ref[sdk.rust.zerocopy-wire-structs]`):** All `.dnam-bin` (`"DNBN"`/`"DBRM"`/`"TSYN"`/`"CEXT"`/`"CP1\0"`) and nested `Arena 0..3` structs derive `FromBytes`, `IntoBytes`, `KnownLayout`, `Immutable` (and `Unaligned` when cast from unaligned slices), verified with `const _: () = assert!(...)` layout assertions and zero manual `from_le_bytes`/`to_le_bytes` slicing.
4. **Explicit SIMD & LOC Ceilings (`ref[sdk.rust.explicit-simd-and-coverage]`):** Zero `macro_rules!` SIMD wrappers, zero `#[cfg_attr(coverage, coverage(off))]`, zero `mod.rs` files, `< 700` lines target per `.rs` file (`< 900` guardrail ceiling), and inline `#[cfg(test)] mod tests` strictly at the bottom of source files.

---

## 3. Quality Gates & Pre-Commit Checklist

```bash
moon check vm
cargo check -p dnam-vm --no-default-features
cargo check -p dnam-vm --no-default-features --features alloc
cargo check -p dnam-vm --no-default-features --target wasm32-unknown-unknown
moon run vm:coverage
moon check tests
```

- [ ] Every modified file passes `moon check vm` (`fmt`, `clippy -D warnings`, `check`, `build --release`, `test`) and bare `--no-default-features` checks.
- [ ] `moon run vm:coverage` achieves `>= 95%` instrumented line coverage across `packages/vm/src/`.
- [ ] `moon check tests` passes all `AGENTS.md` budgets, `zerocopy`, `#![no_std]`, and clean-label guardrails.
