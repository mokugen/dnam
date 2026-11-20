//! `dnam-engine` — Pure Rust `#![no_std]` + `alloc` Real-Time Stereo Neural Audio Modeling & Differentiable DSP Core.
//!
//! Unifies the State-Space & Operator Neural Architecture (**SONA v3.0**) of **Dynamic Neural Audio Modeler (DNAM)**:
//! 1. **Operators ([`operators`]):** Op-T (Transport / Elastic Buffer), Op-M (Symplectic Modal / Kautz Bank),
//!    and Op-N (Multi-Drive PolyADAA Commuted Cascade / Waveshaping).
//! 2. **Graph ([`graph`]):** 8-Port I/O Router & Causal Crossbar, Unconditional `SONA` Matryoshka Core
//!    ([`SonaHotSlab`] / [`SonaInstance`]), Early FIR + Kautz Cabinet, and Spatial `Op-T` FDN Tank.
//! 3. **Control Bus ([`control`]):** $3\text{ kHz}$ Envelopes/LFOs + Node-Anchored Lobatto HDMR & Rank-4/8
//!    Factorization Machine with 128-byte CCR table, division-free 16-sample micro-ramping,
//!    and `z_stage` sensitivity conditioning.
//!
//! ## Core Real-Time Invariants
//! - **100% `#![no_std]` + `alloc` Foundation:** Zero OS calls, zero locks, and **zero heap allocations**
//!   inside [`DnamEngine::process_block`], [`DnamEngine::process_block_in_place`],
//!   [`DnamEngine::process_mono_to_stereo`], [`DnamEngine::process_mono`], and [`DnamEngine::reload_model_in_place`].
//! - **Zero `FSQRT`, `FDIV`, or `libm` Calls** in $48\text{ kHz}$ audio and $3\text{ kHz}$ control loops.
//! - **Total Lite State RAM:** Audited via [`TOTAL_LITE_STATE_RAM_BYTES`].
//! - **Block Floating-Point Environment ([`FtzDazGuard`]):** Selects round-to-nearest-even and
//!   non-trapping exceptions and `FPCR.FZ` with conditional `FIZ` (`aarch64`) or `MXCSR`
//!   `FTZ | DAZ` (`x86_64`) on block entry, then restores the caller's control register. On x86-64,
//!   sticky flags are cleared on entry and restored on exit; on AArch64, the separate `FPSR` status
//!   is not saved. Portable targets flush recursive writebacks. Direct per-sample and control-tick
//!   callers hold a guard once per batch.

#![no_std]
#![deny(unsafe_code)]
#![warn(unreachable_pub)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

#[forbid(unsafe_code)]
pub mod container;
#[forbid(unsafe_code)]
pub mod control;
#[forbid(unsafe_code)]
pub mod dynamic;
#[forbid(unsafe_code)]
pub mod engine;
#[allow(unsafe_code)]
pub mod ffi;
#[forbid(unsafe_code)]
pub mod graph;
#[forbid(unsafe_code)]
pub mod host;
pub mod math;
#[forbid(unsafe_code)]
pub mod memory;
#[forbid(unsafe_code)]
pub mod operators;
#[forbid(unsafe_code)]
pub mod provenance;
#[forbid(unsafe_code)]
pub mod schema;
#[cfg(test)]
#[forbid(unsafe_code)]
pub(crate) mod test_support;
pub mod types;

pub use container::strip_container_to_tier;
pub use dynamic::DynamicDnamEngine;
pub use engine::{DnamAudioIo, DnamEngine, OFFLINE_BOUNCE_PDC_SAMPLES};
pub use graph::{
    SonaColdTierModel, SonaHotSlab, SonaInstance, SonaMatryoshkaModel, SonaTierReadout,
    TOTAL_LITE_STATE_RAM_BYTES, sona_process_block_default, sona_process_block_infallible_default,
};
pub use types::{DnamTelemetrySnapshot, FtzDazGuard, Tier};
