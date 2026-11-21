//! Core execution types (`Tier`, `FtzDazGuard`, `TELEMETRY_CONTROL_SLOTS`, and
//! `DnamTelemetrySnapshot`).

pub use dnam_vm::{FtzDazGuard, Tier};

/// Number of leading physical control slots mirrored into C-ABI telemetry snapshots (`16`).
pub const TELEMETRY_CONTROL_SLOTS: usize = 16;

const _: () = assert!(TELEMETRY_CONTROL_SLOTS == 16);

/// Maps a [`crate::provenance::VerificationStatus`] to its `u8` C-ABI code (`0`, `1`, `2`).
#[inline]
pub fn verification_status_code(status: &crate::provenance::VerificationStatus) -> u8 {
    match status {
        crate::provenance::VerificationStatus::Unsigned => 0,
        crate::provenance::VerificationStatus::CreatorVerified { .. } => 1,
        crate::provenance::VerificationStatus::MarketplaceVerified { .. } => 2,
    }
}

/// Lock-free, copyable `#[repr(C)]` Layer 1 telemetry snapshot for 60 fps UI visualizers
/// (`packages/ui`, `packages/studio`, and `packages/js`).
#[forbid(unsafe_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamTelemetrySnapshot {
    /// Active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
    pub active_tier: u8,
    /// Highest retained tier available in the loaded container (`0..=3`).
    pub max_available_tier: u8,
    /// Host/CPU maximum supported tier ceiling (`0..=3`).
    pub max_supported_tier: u8,
    /// `1` if stereo output head is active, `0` if mono output mode.
    pub stereo_out: u8,
    /// `1` if the engine is currently bypassed, `0` otherwise.
    pub bypassed: u8,
    /// `1` if `4x` offline bounce oversampling is active, `0` otherwise.
    pub offline_bounce: u8,
    /// `1` if host-direct crossbar / `Op-T` parameter lock is enabled, `0` otherwise.
    pub host_write_lock: u8,
    /// Provenance status code (`0 = Unsigned`, `1 = CreatorVerified`, `2 = MarketplaceVerified`).
    pub verification_status: u8,
    /// Bitmask of controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// Current host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}})$.
    pub control_decimation_host: u16,
    /// Active processing latency in host-rate samples.
    pub latency_samples: u32,
    /// Current host callback sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Accumulated finite-precision overflow/saturation flags (`ref[arch.fp.range-saturation]`).
    pub saturation_flags: u32,
    /// Host shell parallel dry mix gain.
    pub mix_dry_gain: f32,
    /// Host shell parallel wet mix gain.
    pub mix_wet_gain: f32,
    /// Active Mid-Side stereo width multiplier (`[0.0, 2.0]`).
    pub stereo_width: f32,
    /// Current host tempo in BPM.
    pub tempo_bpm: f32,
    /// Peak `Op-N` normalized drive $|u_c| \in [0, 1]$ in the latest block.
    pub max_op_n_drive: f32,
    /// Peak `Op-M` modal pole radius $r_c[n] \le 0.9995$ in the latest block.
    pub max_pole_radius: f32,
    /// Peak Stage IV feedback loop gain $g_{\text{eff},m} \le 0.9990$ in the latest block.
    pub max_effective_loop_gain: f32,
    /// Reported $T_{60}$ decay tail time in milliseconds.
    pub tail_time_ms: f32,
    /// Current collapsed SVD head coordinates $\mathbf{z}_{\text{head}} \in \mathbb{R}^4$ ($\mathbf{z}_{0..3}$).
    pub z_head: [f32; 4],
    /// Current 3-band envelope follower states (`[Low, Mid, High, 0.0]`).
    pub env_states: [f32; 4],
    /// Current 4-harmonic Chebyshev LFO waves ($T_1..T_4$ for LFO 0 and LFO 1).
    pub lfo_harmonics: [f32; 8],
    /// Current clamped physical control values across the leading 16 C-ABI telemetry slots.
    pub clamped_controls: [f32; TELEMETRY_CONTROL_SLOTS],
}

const _: () = assert!(core::mem::size_of::<DnamTelemetrySnapshot>() == 184);
const _: () = assert!(core::mem::align_of::<DnamTelemetrySnapshot>() == 4);

impl Default for DnamTelemetrySnapshot {
    #[inline]
    fn default() -> Self {
        Self {
            active_tier: Tier::Lite as u8,
            max_available_tier: Tier::Lite as u8,
            max_supported_tier: Tier::Ultra as u8,
            stereo_out: 1,
            bypassed: 0,
            offline_bounce: 0,
            host_write_lock: 0,
            verification_status: 0,
            active_clamp_mask: 0,
            control_decimation_host: 16,
            latency_samples: 0,
            sample_rate_hz: 48_000,
            saturation_flags: 0,
            mix_dry_gain: 0.0,
            mix_wet_gain: 1.0,
            stereo_width: 1.0,
            tempo_bpm: 120.0,
            max_op_n_drive: 0.0,
            max_pole_radius: 0.0,
            max_effective_loop_gain: 0.0,
            tail_time_ms: 0.0,
            z_head: [0.0; 4],
            env_states: [0.0; 4],
            lfo_harmonics: [0.0; 8],
            clamped_controls: [0.0; TELEMETRY_CONTROL_SLOTS],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_conversions_and_ftz_daz_guard() {
        assert_eq!(Tier::from_u8(0), Tier::Lite);
        assert_eq!(Tier::from_u8(1), Tier::Full);
        assert_eq!(Tier::from_u8(2), Tier::Pro);
        assert_eq!(Tier::from_u8(3), Tier::Ultra);
        assert_eq!(Tier::from_u8(99), Tier::Ultra);

        assert_eq!(Tier::Lite.as_str(), "lite");
        assert_eq!(Tier::Full.as_str(), "full");
        assert_eq!(Tier::Pro.as_str(), "pro");
        assert_eq!(Tier::Ultra.as_str(), "ultra");

        assert_eq!(Tier::from_str_name("LITE"), Some(Tier::Lite));
        assert_eq!(Tier::from_str_name("Full"), Some(Tier::Full));
        assert_eq!(Tier::from_str_name("pro"), Some(Tier::Pro));
        assert_eq!(Tier::from_str_name("ultra"), Some(Tier::Ultra));
        assert_eq!(Tier::from_str_name("invalid"), None);

        assert_eq!(Tier::Lite.channels(), 16);
        assert_eq!(Tier::Full.channels(), 24);
        assert_eq!(Tier::Pro.channels(), 32);
        assert_eq!(Tier::Ultra.channels(), 40);

        assert_eq!(Tier::Lite.active_channels(), 16);
        assert_eq!(Tier::Ultra.active_channels(), 40);

        assert_eq!(Tier::Lite.simd_groups(), 4);
        assert_eq!(Tier::Full.simd_groups(), 6);
        assert_eq!(Tier::Pro.simd_groups(), 8);
        assert_eq!(Tier::Ultra.simd_groups(), 10);

        assert_eq!(Tier::Lite.tank_groups(), 1);
        assert_eq!(Tier::Full.tank_groups(), 2);
        assert_eq!(Tier::Pro.tank_groups(), 3);
        assert_eq!(Tier::Ultra.tank_groups(), 4);

        assert_eq!(Tier::Lite.tank_lines(), 4);
        assert_eq!(Tier::Full.tank_lines(), 8);
        assert_eq!(Tier::Pro.tank_lines(), 12);
        assert_eq!(Tier::Ultra.tank_lines(), 16);

        assert_eq!(Tier::Lite.svd_control_modes(), 4);
        assert_eq!(Tier::Pro.svd_control_modes(), 4);
        assert_eq!(Tier::Ultra.svd_control_modes(), 8);
        assert_eq!(Tier::Lite.fm_interaction_rank(), 4);
        assert_eq!(Tier::Full.fm_interaction_rank(), 4);
        assert_eq!(Tier::Pro.fm_interaction_rank(), 4);
        assert_eq!(Tier::Ultra.fm_interaction_rank(), 8);

        assert_eq!(Tier::Lite.expected_arena_bytes(), 4_608);
        assert_eq!(Tier::Full.expected_arena_bytes(), 7_936);
        assert_eq!(Tier::Pro.expected_arena_bytes(), 15_872);
        assert_eq!(Tier::Ultra.expected_arena_bytes(), 30_720);

        assert_eq!(Tier::Lite.expected_cache_lines(), 72);
        assert_eq!(Tier::Full.expected_cache_lines(), 124);
        assert_eq!(Tier::Pro.expected_cache_lines(), 248);
        assert_eq!(Tier::Ultra.expected_cache_lines(), 480);

        let guard1 = FtzDazGuard::default();
        let guard2 = FtzDazGuard::new();
        drop(guard2);
        drop(guard1);
    }
}
