#![forbid(unsafe_code)]
//! Declarative `#[repr(C)]` `zerocopy` RIFF/WAVE wire headers (`align::wav_wire`).

use core::mem::{align_of, offset_of, size_of};

use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Standard RIFF/WAVE header (`12 B` at byte `0..12`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct RiffHeaderWire {
    /// FourCC `"RIFF"`.
    pub riff_id: [u8; 4],
    /// Remaining byte count from byte `8` to EOF.
    pub riff_size: U32,
    /// FourCC `"WAVE"`.
    pub wave_id: [u8; 4],
}

const _: () = assert!(size_of::<RiffHeaderWire>() == 12);
const _: () = assert!(align_of::<RiffHeaderWire>() == 1);
const _: () = assert!(offset_of!(RiffHeaderWire, riff_size) == 4);
const _: () = assert!(offset_of!(RiffHeaderWire, wave_id) == 8);

/// Generic RIFF sub-chunk header (`8 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct RiffChunkHeaderWire {
    /// FourCC chunk identifier (e.g., `b"fmt "`, `b"data"`).
    pub chunk_id: [u8; 4],
    /// Unpadded payload length in bytes.
    pub chunk_len: U32,
}

const _: () = assert!(size_of::<RiffChunkHeaderWire>() == 8);
const _: () = assert!(align_of::<RiffChunkHeaderWire>() == 1);
const _: () = assert!(offset_of!(RiffChunkHeaderWire, chunk_len) == 4);

/// Base `16 B` `WAVEFORMATEX` / `PCMWAVEFORMAT` `"fmt "` body.
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct WavFmtBaseWire {
    /// Wave format tag (`1` = PCM, `3` = IEEE Float, `0xFFFE` = `WAVE_FORMAT_EXTENSIBLE`).
    pub audio_format: U16,
    /// Interleaved channel count (`>= 1`).
    pub channels: U16,
    /// Sample rate in Hz (`>= 1`).
    pub sample_rate_hz: U32,
    /// Average bytes per second (`sample_rate_hz * block_align`).
    pub avg_bytes_per_sec: U32,
    /// Frame stride in bytes (`channels * container_bytes`).
    pub block_align: U16,
    /// Bits per sample (`12`, `16`, `24`, or `32`).
    pub bits_per_sample: U16,
}

const _: () = assert!(size_of::<WavFmtBaseWire>() == 16);
const _: () = assert!(align_of::<WavFmtBaseWire>() == 1);
const _: () = assert!(offset_of!(WavFmtBaseWire, avg_bytes_per_sec) == 8);
const _: () = assert!(offset_of!(WavFmtBaseWire, bits_per_sample) == 14);

/// Extended `40 B` `WAVE_FORMAT_EXTENSIBLE` `"fmt "` body (`0xFFFE`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct WavFmtExtensibleWire {
    /// Leading `16 B` base format descriptor (`audio_format = 0xFFFE`).
    pub base: WavFmtBaseWire,
    /// Extension byte size (`>= 22`).
    pub cb_size: U16,
    /// Significant bits per sample (`wValidBitsPerSample`).
    pub valid_bits_per_sample: U16,
    /// Speaker position bitmask.
    pub channel_mask: U32,
    /// Base GUID sub-format tag (`1` = PCM, `3` = IEEE Float).
    pub sub_format_tag: U16,
    /// Remaining `14 B` `KSDATAFORMAT_SUBTYPE` GUID tail.
    pub sub_format_tail: [u8; 14],
}

const _: () = assert!(size_of::<WavFmtExtensibleWire>() == 40);
const _: () = assert!(align_of::<WavFmtExtensibleWire>() == 1);
const _: () = assert!(offset_of!(WavFmtExtensibleWire, cb_size) == 16);
const _: () = assert!(offset_of!(WavFmtExtensibleWire, valid_bits_per_sample) == 18);
const _: () = assert!(offset_of!(WavFmtExtensibleWire, channel_mask) == 20);
const _: () = assert!(offset_of!(WavFmtExtensibleWire, sub_format_tag) == 24);
const _: () = assert!(offset_of!(WavFmtExtensibleWire, sub_format_tail) == 26);

/// Standard 44-byte RIFF/WAVE prefix (`RIFF` + `fmt ` 16B + `data` chunk header).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct WavPcmHeader44Wire {
    /// `12 B` RIFF/WAVE preamble.
    pub riff: RiffHeaderWire,
    /// `"fmt "` chunk header (`chunk_len = 16`).
    pub fmt_hdr: RiffChunkHeaderWire,
    /// `16 B` base format record.
    pub fmt_body: WavFmtBaseWire,
    /// `"data"` chunk header.
    pub data_hdr: RiffChunkHeaderWire,
}

const _: () = assert!(size_of::<WavPcmHeader44Wire>() == 44);
const _: () = assert!(align_of::<WavPcmHeader44Wire>() == 1);
const _: () = assert!(offset_of!(WavPcmHeader44Wire, fmt_hdr) == 12);
const _: () = assert!(offset_of!(WavPcmHeader44Wire, fmt_body) == 20);
const _: () = assert!(offset_of!(WavPcmHeader44Wire, data_hdr) == 36);

/// `68 B` `WAVE_FORMAT_EXTENSIBLE` RIFF/WAVE prefix (`RIFF` + `fmt ` 40B + `data` chunk header).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct WavExtensibleHeader68Wire {
    /// `12 B` RIFF/WAVE preamble.
    pub riff: RiffHeaderWire,
    /// `"fmt "` chunk header (`chunk_len = 40`).
    pub fmt_hdr: RiffChunkHeaderWire,
    /// `40 B` extensible format record.
    pub fmt_body: WavFmtExtensibleWire,
    /// `"data"` chunk header.
    pub data_hdr: RiffChunkHeaderWire,
}

const _: () = assert!(size_of::<WavExtensibleHeader68Wire>() == 68);
const _: () = assert!(align_of::<WavExtensibleHeader68Wire>() == 1);
const _: () = assert!(offset_of!(WavExtensibleHeader68Wire, fmt_body) == 20);
const _: () = assert!(offset_of!(WavExtensibleHeader68Wire, data_hdr) == 60);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_wire_roundtrip() {
        let hdr = WavPcmHeader44Wire {
            riff: RiffHeaderWire {
                riff_id: *b"RIFF",
                riff_size: U32::new(36),
                wave_id: *b"WAVE",
            },
            fmt_hdr: RiffChunkHeaderWire {
                chunk_id: *b"fmt ",
                chunk_len: U32::new(16),
            },
            fmt_body: WavFmtBaseWire {
                audio_format: U16::new(1),
                channels: U16::new(2),
                sample_rate_hz: U32::new(48_000),
                avg_bytes_per_sec: U32::new(192_000),
                block_align: U16::new(4),
                bits_per_sample: U16::new(16),
            },
            data_hdr: RiffChunkHeaderWire {
                chunk_id: *b"data",
                chunk_len: U32::new(0),
            },
        };
        let bytes = hdr.as_bytes();
        let parsed = WavPcmHeader44Wire::ref_from_bytes(bytes).unwrap();
        assert_eq!(parsed.fmt_body.sample_rate_hz.get(), 48_000);
        assert_eq!(parsed.fmt_body.channels.get(), 2);
    }
}
