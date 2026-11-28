//! Core execution types (`Tier`, `SonaTier`, `CacheLine64`, `ColdBuf`, `ColdSlice`, `VmError`,
//! and `FtzDazGuard`).

#[forbid(unsafe_code)]
pub mod cold_buf;
#[forbid(unsafe_code)]
pub mod error;
#[allow(unsafe_code)]
mod fp_env;

pub use cold_buf::{ColdBuf, ColdSlice};
pub use error::VmError;
pub use fp_env::FtzDazGuard;

use core::mem::{align_of, size_of};

use dnam_kernels::{FULL_SPEC, LITE_SPEC, PRO_SPEC, SonaTierSpec, ULTRA_SPEC};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Slimmable `SONA v4.0` execution tier (`Lite`, `Full`, `Pro`, or `Ultra`).
#[forbid(unsafe_code)]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// `Lite`: `16` channels, `4` tank lines, `D_h = 4`, `5,376 B` (`84` cache lines).
    Lite = 0,
    /// `Full`: `24` channels, `8` tank lines, `D_h = 4`, `8,768 B` (`137` cache lines).
    Full = 1,
    /// `Pro`: `32` channels, `12` tank lines, `D_h = 4`, `17,088 B` (`267` cache lines).
    Pro = 2,
    /// `Ultra`: `40` channels, `16` tank lines, `D_h = 8`, `30,272 B` (`473` cache lines).
    Ultra = 3,
}

/// Canonical `SONA` tier alias exported at the VM root.
pub type SonaTier = Tier;

#[forbid(unsafe_code)]
impl Tier {
    /// All four tiers ordered monotonically from `Lite` (`0`) to `Ultra` (`3`).
    pub const ALL: [Self; 4] = [Self::Lite, Self::Full, Self::Pro, Self::Ultra];

    /// Converts a raw `u8` (`0..=3`) into [`Tier`], clamping values `>= 3` to [`Tier::Ultra`].
    #[inline]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Lite,
            1 => Self::Full,
            2 => Self::Pro,
            _ => Self::Ultra,
        }
    }

    /// Strictly decodes a raw `u8` (`0..=3`) into `Some(Tier)`, rejecting values `>= 4`.
    #[inline]
    pub const fn try_from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Lite),
            1 => Some(Self::Full),
            2 => Some(Self::Pro),
            3 => Some(Self::Ultra),
            _ => None,
        }
    }

    /// Returns the zero-based tier index (`0..=3`).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
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

    /// Returns the canonical compile-time [`SonaTierSpec`] from `dnam-kernels`.
    #[inline]
    pub const fn spec(self) -> SonaTierSpec {
        match self {
            Self::Lite => LITE_SPEC,
            Self::Full => FULL_SPEC,
            Self::Pro => PRO_SPEC,
            Self::Ultra => ULTRA_SPEC,
        }
    }

    /// Returns the number of active `SONA` core channels (`16`, `24`, `32`, or `40`).
    #[inline]
    pub const fn channels(self) -> usize {
        self.spec().lanes()
    }

    /// Alias for [`Self::channels`].
    #[inline]
    pub const fn active_channels(self) -> usize {
        self.channels()
    }

    /// Returns the number of 4-channel SIMD tier groups (`4`, `6`, `8`, or `10`).
    #[inline]
    pub const fn simd_groups(self) -> usize {
        self.channels() / 4
    }

    /// Returns the number of Stage IV tank quads (`1`, `2`, `3`, or `4`).
    #[inline]
    pub const fn tank_groups(self) -> usize {
        self.spec().tank_quads()
    }

    /// Returns the number of active Stage IV tank delay lines (`4`, `8`, `12`, or `16`).
    #[inline]
    pub const fn tank_lines(self) -> usize {
        self.tank_groups() * 4
    }

    /// Returns the cumulative Cabinet early FIR tap count (`32`, `48`, `96`, or `128`).
    #[inline]
    pub const fn early_fir_taps(self) -> usize {
        self.spec().fir_taps
    }

    /// Returns the cumulative Cabinet shared Kautz pole pair count (`8`, `12`, `24`, or `32`).
    #[inline]
    pub const fn kautz_pairs(self) -> usize {
        self.spec().kautz_pairs
    }

    /// Returns the active SVD control subspace rank (`4` for `Lite..=Pro`, `8` for `Ultra`).
    #[inline]
    pub const fn svd_control_modes(self) -> usize {
        match self {
            Self::Lite | Self::Full | Self::Pro => 4,
            Self::Ultra => 8,
        }
    }

    /// Returns the active stage-latent coordinate dimension (`4` for `Lite..=Pro`, `8` for `Ultra`).
    #[inline]
    pub const fn stage_latent_dim(self) -> usize {
        self.svd_control_modes()
    }

    /// Returns the active Factorization Machine interaction rank (`4` for `Lite..=Pro`, `8` for `Ultra`).
    #[inline]
    pub const fn fm_interaction_rank(self) -> usize {
        self.svd_control_modes()
    }

    /// Returns the cumulative `SONA v4.0` arena byte footprint (`5,376`, `8,768`, `17,088`, or `30,272`).
    #[inline]
    pub const fn expected_arena_bytes(self) -> usize {
        crate::bin::ARENA_TIER_BYTES[self.index()]
    }

    /// Alias for [`Self::expected_arena_bytes`].
    #[inline]
    pub const fn arena_bytes(self) -> usize {
        self.expected_arena_bytes()
    }

    /// Returns the cumulative number of `64 B` cache lines (`84`, `137`, `267`, or `473`).
    #[inline]
    pub const fn expected_cache_lines(self) -> usize {
        self.expected_arena_bytes() / 64
    }
}

/// Fixed `64 B` cache-line-aligned backing unit for `#![no_std]` / `!alloc` arena storage.
#[forbid(unsafe_code)]
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct CacheLine64(pub [u8; 64]);

const _: () = assert!(size_of::<CacheLine64>() == 64);
const _: () = assert!(align_of::<CacheLine64>() == 64);

impl Default for CacheLine64 {
    #[inline]
    fn default() -> Self {
        Self([0; 64])
    }
}

#[forbid(unsafe_code)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_conversions_and_cache_line_layout() {
        assert_eq!(Tier::from_u8(0), Tier::Lite);
        assert_eq!(Tier::from_u8(1), Tier::Full);
        assert_eq!(Tier::from_u8(2), Tier::Pro);
        assert_eq!(Tier::from_u8(3), Tier::Ultra);
        assert_eq!(Tier::from_u8(99), Tier::Ultra);
        assert_eq!(Tier::try_from_u8(2), Some(Tier::Pro));
        assert_eq!(Tier::try_from_u8(4), None);

        assert_eq!(Tier::Lite.index(), 0);
        assert_eq!(Tier::Ultra.index(), 3);
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
        assert_eq!(Tier::Ultra.active_channels(), 40);
        assert_eq!(Tier::Pro.simd_groups(), 8);
        assert_eq!(Tier::Lite.tank_lines(), 4);
        assert_eq!(Tier::Full.tank_lines(), 8);
        assert_eq!(Tier::Pro.tank_lines(), 12);
        assert_eq!(Tier::Ultra.tank_lines(), 16);

        assert_eq!(Tier::Lite.early_fir_taps(), 32);
        assert_eq!(Tier::Full.early_fir_taps(), 48);
        assert_eq!(Tier::Pro.early_fir_taps(), 96);
        assert_eq!(Tier::Ultra.early_fir_taps(), 128);
        assert_eq!(Tier::Lite.kautz_pairs(), 8);
        assert_eq!(Tier::Ultra.kautz_pairs(), 32);

        assert_eq!(Tier::Pro.svd_control_modes(), 4);
        assert_eq!(Tier::Ultra.stage_latent_dim(), 8);
        assert_eq!(Tier::Ultra.fm_interaction_rank(), 8);

        assert_eq!(Tier::Lite.arena_bytes(), 5_376);
        assert_eq!(Tier::Full.arena_bytes(), 8_768);
        assert_eq!(Tier::Pro.arena_bytes(), 17_088);
        assert_eq!(Tier::Ultra.arena_bytes(), 30_272);
        assert_eq!(Tier::Lite.expected_cache_lines(), 84);
        assert_eq!(Tier::Ultra.expected_cache_lines(), 473);

        let cl = CacheLine64::default();
        assert_eq!(cl.0, [0u8; 64]);
        let guard = FtzDazGuard::default();
        drop(guard);
    }
}
