//! Cold `SONA` Matryoshka model & readout re-exports (`stage3_cab/cold.rs`).
//!
//! Canonical definitions live in [`crate::stages::sona_core::model`] (`ref[repo.code.modular-decomposition-and-tests]`).

pub use crate::stages::sona_core::model::{
    SonaColdTierModel, SonaMatryoshkaModel, SonaTierReadout, V3ColdTierModel, V3MatryoshkaModel,
    V3TierReadout,
};
