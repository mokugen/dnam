//! Spatial `Op-T` FDN Tank with Givens+Householder Mixing (`graph/tank/fdn.rs`).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_crossbar_topology.md` (`ref[dsp.tank.feedback-loop]`,
//! `ref[dsp.tank.dc-blocker]`, `ref[dsp.tank.loop-saturator]`, `ref[dsp.tank.quiet-feedback-knee]`).

pub mod fdn;

pub use fdn::{
    SONA_TANK_MAX_LINES, SONA_TANK_QUADS, STAGE4_SCHROEDER_MAX_DELAY,
    STAGE4_SCHROEDER_REF_DELAYS_48K, SpatialTankHot, SpatialTankParams, TankExtFormat,
    TankQuadParams, V3_TANK_CAP, V3_TANK_LINES, V3_TANK_MAX_DECLARED_DELAY, V3_TANK_MAX_EXT_CAP,
    V3_TANK_MAX_LFO_DEPTH, V3_TANK_RING_STRIDE,
};
