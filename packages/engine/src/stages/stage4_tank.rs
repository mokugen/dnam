//! Stage IV: Unitary Spatial `Op-T` FDN Tank with Givens+Householder Mixing (`v3_fdn`).
//!
//! Canonical Reference: `docs/specs/dsp/router_and_stages_topology.md` (`ref[dsp.stage4.spatial-tank]`,
//! `ref[dsp.stage4.dc-blocker]`, `ref[dsp.stage4.loop-saturator]`, `ref[dsp.stage4.quiet-feedback-knee]`).

pub mod v3_fdn;

pub use v3_fdn::{
    SONA_TANK_MAX_LINES, SONA_TANK_QUADS, STAGE4_SCHROEDER_MAX_DELAY,
    STAGE4_SCHROEDER_REF_DELAYS_48K, Stage4ExtFormat, Stage4QuadParams, V3_TANK_CAP, V3_TANK_LINES,
    V3_TANK_MAX_DECLARED_DELAY, V3_TANK_MAX_EXT_CAP, V3_TANK_MAX_LFO_DEPTH, V3_TANK_RING_STRIDE,
    V3Stage4TankHot, V3Stage4TankParams,
};
