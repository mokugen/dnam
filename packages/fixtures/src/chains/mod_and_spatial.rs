//! Envelope-Modulated Resonant SVF (`Chain-06`), Stereo Chorus & Rotary (`Chain-07`),
//! Long BBD Echo + `30s` Shimmer FDN with Timed Freeze (`Chain-08`), and Barberpole Sweep (`Chain-17`).

#![forbid(unsafe_code)]

pub mod barberpole;
pub mod bbd_shimmer_fdn;
pub mod chorus_rotary;
pub mod envelope_svf;

pub use barberpole::Chain17BarberpoleSweep;
pub use bbd_shimmer_fdn::Chain08LongBbdEchoAnd30sShimmer;
pub use chorus_rotary::{CANONICAL_SUBDIVISION_RATIOS, Chain07StereoChorusAndRotary};
pub use envelope_svf::Chain06EnvelopeResonantSvf;
