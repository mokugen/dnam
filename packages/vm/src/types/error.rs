//! Allocation-free VM container and control error types (`VmError`).

use core::fmt;

/// Allocation-free error returned by `.dnam-bin` parsing, stripping, builder encoding,
/// `CP1` decoding, and `ControlBus` validation (`#[forbid(unsafe_code)]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmError {
    /// Input buffer is shorter than the fixed `128 B` `"DNBN"` + `"DBRM"` prefix.
    TruncatedPrefix,
    /// `"DNBN"` or `"DBRM"` magic tag does not match.
    InvalidMagic,
    /// Unsupported binary container version (expected `v3.0`).
    UnsupportedVersion,
    /// Reserved byte, flag bit (including evicted freeze bits `5..=6`), or metadata size is non-zero / invalid.
    InvalidHeaderFlagsOrReserved,
    /// `max_present_tier` or requested `target_tier` is out of range or exceeds retained bytes.
    InvalidTier,
    /// Declared `reference_sample_rate_hz` is outside `[44_100..=96_000]` or host rate is invalid.
    InvalidSampleRate,
    /// Region offsets violate `128 <= shared_controls <= tsyn <= cext <= arena0_offset`, `64 B` alignment, or bounds.
    InvalidRegionLayout,
    /// Non-zero byte encountered inside an alignment padding region between prefix tables and `arena0_offset`.
    NonZeroAlignmentPadding,
    /// File length does not exactly match `tier_end_offsets[t]` for any valid `t <= max_present_tier`.
    LengthTierMismatch,
    /// Castagnoli `CRC-32C` over `[0..20) || [24..arena0_offset)` failed.
    PrefixCrcMismatch,
    /// Castagnoli `CRC-32C` over `[arena0_offset..tier_end_offsets[t])` failed.
    TierArenaCrcMismatch,
    /// Arena 0 `pipeline_flags`, `sona_meta_flags`, `SonaCellDescWire`, or active `TankQuadWire` reserved bits failed validation.
    InvalidArenaHeader,
    /// `ControlClampDescriptorWire` failed kind, bitmask, finite-bounds, or `cond_rules` validation.
    InvalidControlDescriptor,
    /// `"CP1\0"` overflow stream or inline `cp1_dir` failed framing, alignment, or capacity validation.
    InvalidCp1Stream,
    /// `"TSYN"` entry failed index, mode, finite-positive ratio, or `declared_min <= declared_max` validation.
    InvalidTempoSyncEntry,
    /// Caller-provided mutable backing slice is too small for the loaded `.dnam-bin` control/tempo count or output buffer.
    InsufficientBackingCapacity,
    /// `ExactIntFactorAccumulator` contributor index exceeded bounds or input float was non-finite.
    InvalidExactAccumulatorInput,
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::TruncatedPrefix => "truncated .dnam-bin prefix (< 128 B)",
            Self::InvalidMagic => "invalid .dnam-bin or DBRM magic bytes",
            Self::UnsupportedVersion => "unsupported .dnam-bin container version (expected v3.0)",
            Self::InvalidHeaderFlagsOrReserved => {
                "invalid .dnam-bin header flags or non-zero reserved field"
            }
            Self::InvalidTier => "invalid or unavailable SONA tier index",
            Self::InvalidSampleRate => "invalid reference or host sample rate",
            Self::InvalidRegionLayout => "invalid or unaligned .dnam-bin prefix region layout",
            Self::NonZeroAlignmentPadding => "non-zero byte in .dnam-bin alignment padding",
            Self::LengthTierMismatch => {
                ".dnam-bin length does not match any retained tier end offset"
            }
            Self::PrefixCrcMismatch => ".dnam-bin prefix CRC-32C verification failed",
            Self::TierArenaCrcMismatch => {
                ".dnam-bin cumulative tier arena CRC-32C verification failed"
            }
            Self::InvalidArenaHeader => {
                "invalid Arena 0 pipeline/cell descriptor or tank reserved bits"
            }
            Self::InvalidControlDescriptor => "invalid ControlClampDescriptorWire field or bounds",
            Self::InvalidCp1Stream => "invalid CP1 inline directory or overflow stream",
            Self::InvalidTempoSyncEntry => "invalid 40 B TSYN tempo-sync entry",
            Self::InsufficientBackingCapacity => "insufficient caller backing capacity",
            Self::InvalidExactAccumulatorInput => {
                "invalid exact-integer accumulator contributor or value"
            }
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VmError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_error_display_variants() {
        let all = [
            VmError::TruncatedPrefix,
            VmError::InvalidMagic,
            VmError::UnsupportedVersion,
            VmError::InvalidHeaderFlagsOrReserved,
            VmError::InvalidTier,
            VmError::InvalidSampleRate,
            VmError::InvalidRegionLayout,
            VmError::NonZeroAlignmentPadding,
            VmError::LengthTierMismatch,
            VmError::PrefixCrcMismatch,
            VmError::TierArenaCrcMismatch,
            VmError::InvalidArenaHeader,
            VmError::InvalidControlDescriptor,
            VmError::InvalidCp1Stream,
            VmError::InvalidTempoSyncEntry,
            VmError::InsufficientBackingCapacity,
            VmError::InvalidExactAccumulatorInput,
        ];
        struct CountSink(usize);
        impl fmt::Write for CountSink {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.0 += s.len();
                Ok(())
            }
        }
        for err in all {
            use fmt::Write;
            let mut sink = CountSink(0);
            write!(sink, "{err}").unwrap();
            assert!(sink.0 > 0);
            #[cfg(feature = "alloc")]
            assert!(!alloc::format!("{err}").is_empty());
        }
    }
}
