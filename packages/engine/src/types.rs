//! Core execution types (`Tier`, `FtzDazGuard`, and `DnamTelemetrySnapshot`).

mod fp_env;

pub use fp_env::FtzDazGuard;

use crate::control::TELEMETRY_CONTROL_SLOTS;

/// Slimmable execution tier (`Lite`, `Full`, `Pro`, or `Ultra`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// **`Lite` ($C=4$ channels, $M=4$ tank lines, $D_h=4$ SVD modes, `10,752 B` `Arena 0`):**
    /// Mix-Ready / Live Grade (`168` cache lines).
    Lite = 0,
    /// **`Full` ($C=8$ channels, $M=8$ tank lines, $D_h=4$ SVD modes, `22,336 B` `Arena 0..1`):**
    /// True Studio Quality (`349` cache lines).
    Full = 1,
    /// **`Pro` ($C=12$ channels, $M=8$ tank lines, $D_h=4$ SVD modes, `36,224 B` `Arena 0..2`):**
    /// Mastering / Reference Grade (`566` cache lines, `8 + 4` SIMD loop).
    Pro = 2,
    /// **`Ultra` ($C=16$ channels, $M=16$ tank lines, $D_h=8$ SVD modes, `64,000 B` `Arena 0..3`):**
    /// Unconstrained Workstation Max (`1,000` cache lines, Rank-8 FM + 16-line Householder/Givens tank).
    Ultra = 3,
}

impl Tier {
    /// Converts a raw `u8` (`0`, `1`, `2`, `3`) into a [`Tier`], clamping values $\ge 3$ to [`Tier::Ultra`].
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Lite,
            1 => Self::Full,
            2 => Self::Pro,
            _ => Self::Ultra,
        }
    }

    /// Returns the canonical lowercase string identifier (`"lite"`, `"full"`, `"pro"`, `"ultra"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lite => "lite",
            Self::Full => "full",
            Self::Pro => "pro",
            Self::Ultra => "ultra",
        }
    }

    /// Parses a tier name (case-insensitive `"lite"`, `"full"`, `"pro"`, `"ultra"`).
    #[inline]
    pub fn from_str_name(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("lite") {
            Some(Self::Lite)
        } else if s.eq_ignore_ascii_case("full") {
            Some(Self::Full)
        } else if s.eq_ignore_ascii_case("pro") {
            Some(Self::Pro)
        } else if s.eq_ignore_ascii_case("ultra") {
            Some(Self::Ultra)
        } else {
            None
        }
    }

    /// Returns the number of active channels ($C \in \{4, 8, 12, 16\}$).
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Lite => 4,
            Self::Full => 8,
            Self::Pro => 12,
            Self::Ultra => 16,
        }
    }

    /// Alias for `channels`.
    #[inline]
    pub const fn active_channels(self) -> usize {
        self.channels()
    }

    /// Returns the number of 4-channel SIMD tier groups (`1`, `2`, `3`, or `4`).
    #[inline]
    pub const fn simd_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full => 2,
            Self::Pro => 3,
            Self::Ultra => 4,
        }
    }

    /// Returns the number of 4-line SIMD groups in Stage IV tank (`1` for `Lite`, `2` for `Full`/`Pro`, `4` for `Ultra`).
    #[inline]
    pub const fn tank_groups(self) -> usize {
        match self {
            Self::Lite => 1,
            Self::Full | Self::Pro => 2,
            Self::Ultra => 4,
        }
    }

    /// Returns the number of active Stage IV tank delay lines ($M \in \{4, 8, 8, 16\}$).
    #[inline]
    pub const fn tank_lines(self) -> usize {
        self.tank_groups() * 4
    }

    /// Returns the active SVD control subspace rank ($D_h = 4$ for `Lite`/`Full`/`Pro`, $D_h = 8$ for `Ultra`).
    #[inline]
    pub const fn svd_control_modes(self) -> usize {
        match self {
            Self::Lite | Self::Full | Self::Pro => 4,
            Self::Ultra => 8,
        }
    }

    /// Returns the active Factorization Machine (FM) interaction rank ($R = 4$ for `Lite`/`Full`/`Pro`, $R = 8$ for `Ultra`).
    #[inline]
    pub const fn fm_interaction_rank(self) -> usize {
        match self {
            Self::Lite | Self::Full | Self::Pro => 4,
            Self::Ultra => 8,
        }
    }

    /// Returns the cumulative arena byte end offset (`10,752`, `22,336`, `36,224`, or `64,000`).
    #[inline]
    pub const fn expected_arena_bytes(self) -> usize {
        match self {
            Self::Lite => 4_608,
            Self::Full => 10_240,
            Self::Pro => 18_432,
            Self::Ultra => 28_672,
        }
    }

    /// Returns the cumulative number of 64-byte cache lines (`168`, `349`, `566`, or `1,000`).
    #[inline]
    pub const fn expected_cache_lines(self) -> usize {
        self.expected_arena_bytes() / 64
    }
}

/// Lock-free, copyable `#[repr(C)]` telemetry snapshot for 60 fps UI visualizers
/// (`ToraTelemetry.tsx`, `packages/plugin`, and `packages/js`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamTelemetrySnapshot {
    /// Active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
    pub active_tier: u8,
    /// `1` if stereo output head is active, `0` if mono output mode.
    pub stereo_out: u8,
    /// `1` if the engine is currently bypassed, `0` otherwise.
    pub bypassed: u8,
    /// `1` if Stage IV infinite freeze is currently engaged, `0` otherwise.
    pub is_frozen: u8,
    /// Bitmask of controls currently clamped by `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, or `[CLAMP-STABILITY]`.
    pub active_clamp_mask: u16,
    /// Current host control-bus decimation stride $K_{\text{ctrl}}(f_{s,\text{host}})$.
    pub control_decimation_host: u16,
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
    /// Accumulated finite-precision overflow/saturation flags (`ref[arch.fp.range-saturation]`).
    pub overflow_or_saturation_flags: u32,
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

        assert_eq!(Tier::Lite.channels(), 4);
        assert_eq!(Tier::Full.channels(), 8);
        assert_eq!(Tier::Pro.channels(), 12);
        assert_eq!(Tier::Ultra.channels(), 16);

        assert_eq!(Tier::Lite.active_channels(), 4);
        assert_eq!(Tier::Ultra.active_channels(), 16);

        assert_eq!(Tier::Lite.simd_groups(), 1);
        assert_eq!(Tier::Full.simd_groups(), 2);
        assert_eq!(Tier::Pro.simd_groups(), 3);
        assert_eq!(Tier::Ultra.simd_groups(), 4);

        assert_eq!(Tier::Lite.tank_groups(), 1);
        assert_eq!(Tier::Full.tank_groups(), 2);
        assert_eq!(Tier::Pro.tank_groups(), 2);
        assert_eq!(Tier::Ultra.tank_groups(), 4);

        assert_eq!(Tier::Lite.tank_lines(), 4);
        assert_eq!(Tier::Full.tank_lines(), 8);
        assert_eq!(Tier::Pro.tank_lines(), 8);
        assert_eq!(Tier::Ultra.tank_lines(), 16);

        assert_eq!(Tier::Lite.svd_control_modes(), 4);
        assert_eq!(Tier::Pro.svd_control_modes(), 4);
        assert_eq!(Tier::Ultra.svd_control_modes(), 8);
        assert_eq!(Tier::Lite.fm_interaction_rank(), 4);
        assert_eq!(Tier::Full.fm_interaction_rank(), 4);
        assert_eq!(Tier::Pro.fm_interaction_rank(), 4);
        assert_eq!(Tier::Ultra.fm_interaction_rank(), 8);

        assert_eq!(Tier::Lite.expected_arena_bytes(), 4_608);
        assert_eq!(Tier::Full.expected_arena_bytes(), 10_240);
        assert_eq!(Tier::Pro.expected_arena_bytes(), 18_432);
        assert_eq!(Tier::Ultra.expected_arena_bytes(), 28_672);

        assert_eq!(Tier::Lite.expected_cache_lines(), 72);
        assert_eq!(Tier::Full.expected_cache_lines(), 160);
        assert_eq!(Tier::Pro.expected_cache_lines(), 288);
        assert_eq!(Tier::Ultra.expected_cache_lines(), 448);

        let guard1 = FtzDazGuard::default();
        let guard2 = FtzDazGuard::new();
        drop(guard2);
        drop(guard1);
    }
}
