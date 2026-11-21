//! # Dynamic Neural Audio Modeler (DNAM) Layer 1 `.dnam` Shell & Provenance Engine (`dnam-engine`)
//!
//! Provides `#![no_std]` (`alloc`) `.dnam` / `.dnam-seal` / `.dnam-prev` container parsing,
//! cryptographic provenance (`DAU1` / `DNSL` / `Ed25519` / HPKE), Layer 0 `.dnam-bin` extraction
//! ([`extract_dnam_bin`]), Layer 1 [`HostShellPatchbay`] multi-port routing, storefront
//! [`PreviewGuardState`] probe defense, real-time [`DnamEngine`] / [`DynamicDnamEngine`] wrappers
//! over [`dnam_vm::SonaVm`], wait-free [`host`] concurrency, and `15` `dnam_engine_*` C-ABI
//! functions (`#![deny(unsafe_code)]`, `#[forbid(unsafe_code)]` on all safe submodules).

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

#[forbid(unsafe_code)]
pub mod container;
#[forbid(unsafe_code)]
pub mod dynamic;
#[forbid(unsafe_code)]
pub mod engine;
#[allow(unsafe_code)]
pub mod ffi;
#[forbid(unsafe_code)]
pub mod host;
#[forbid(unsafe_code)]
pub mod memory;
pub mod provenance;
#[forbid(unsafe_code)]
pub mod schema;
#[cfg(test)]
#[forbid(unsafe_code)]
pub(crate) mod test_support;
#[forbid(unsafe_code)]
pub mod types;

pub use container::{extract_dnam_bin, strip_container_to_tier};
pub use dynamic::DynamicDnamEngine;
pub use engine::{
    DnamAudioIo, DnamEngine, EngineBackend, HostShellPatchbay, OFFLINE_BOUNCE_PDC_SAMPLES,
};
pub use provenance::PreviewGuardState;
pub use types::{DnamTelemetrySnapshot, FtzDazGuard, Tier};
