#![forbid(unsafe_code)]
//! `64 B` `DNWB` Binary Pipe Protocol, GPL-Isolated `FaustRunner` & `SpiceRunner`
//! (`PCHIP` Variable-Step `4x` Interpolation + `309`-Tap Kaiser-Sinc `4x -> 1x` Decimation),
//! and Analytical Rust `f64` Reference Twins (`whitebox`).
//!
//! Canonical Reference: `ADR-0018`, `ADR-0023`, and `docs/specs/training/webgpu_and_varpro_solver.md`
//! (`r[train.pipeline.fixtures-matrix]`).

mod faust;
mod ipc;
mod runners;
mod session;
mod wire;

pub use faust::{FaustControlSpec, FaustRunner, parse_faust_json_controls};
pub use ipc::{
    execute_dnwb_subprocess_request, read_dnwb_stem_response, serve_dnwb_single_request,
    serve_dnwb_single_request_with_switches, write_dnwb_postcard_stem_response,
    write_dnwb_stem_response,
};
pub use runners::{
    SPICE_CAUSAL_DELAY_SAMPLES, SpiceRunner, format_spice_tran_directives,
    format_spice_tran_directives_for_rate, kaiser_sinc_decimate_4x_to_48khz,
    pchip_resample_to_192khz, pchip_resample_to_rate,
};
pub use session::{DnwbWorkerSession, serve_dnwb_stream, serve_dnwb_stream_with_switches};
pub use wire::{DnwbHeader64BWire, DnwbPostcardRequestMeta, DnwbPostcardResponseMeta};

use std::io::{Read, Write};

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, IntoBytes};

use crate::targets::StemStimulus;

/// Magic bytes for the `64 B`-aligned White-Box Distillation Pipe Protocol (`b"DNWB"`).
pub const DNWB_MAGIC: [u8; 4] = *b"DNWB";

/// Current `DNWB` wire protocol version (`1`).
pub const DNWB_VERSION: u16 = 1;

/// Maximum permitted `DNWB` payload byte length (`64 MiB`) to prevent OOM allocation on corrupt frames.
pub const MAX_DNWB_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum SPICE solver timestep at `48 kHz` ($\Delta t_{\max} = 1 / 192{,}000\text{ s} \approx 5.208333\times 10^{-6}\text{ s}$).
pub const SPICE_MAXSTEP_SEC: f64 = 1.0 / 192_000.0;

pub(crate) const DNWB_FLAG_STRUCTURED_V1: u16 = 1 << 0;
pub(crate) const DNWB_FLAG_HAS_TRAJECTORIES: u16 = 1 << 1;
pub(crate) const DNWB_FLAG_HAS_FREEZE_WINDOW: u16 = 1 << 2;
/// Bit `3`: closed ephemeral `#![no_std]` Rust ↔ Rust control metadata encoded via `postcard` (`ADR-0023`).
pub const DNWB_FLAG_POSTCARD_CTRL: u16 = 1 << 3;
pub(crate) const DNWB_CHANNEL_MASK_SHIFT: u16 = 8;
pub(crate) const DNWB_KNOWN_REQ_FLAGS: u16 = DNWB_FLAG_STRUCTURED_V1
    | DNWB_FLAG_HAS_TRAJECTORIES
    | DNWB_FLAG_HAS_FREEZE_WINDOW
    | DNWB_FLAG_POSTCARD_CTRL
    | (0x0f << DNWB_CHANNEL_MASK_SHIFT);

/// Command opcodes for [`DnwbHeader64B`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum DnwbCommand {
    /// Reset internal circuit/DSP state to DC equilibrium (`0`).
    Reset = 0,
    /// Render a full batch stem (`1`).
    RenderStem = 1,
    /// Render a single streaming frame without resetting state (`2`).
    RenderFrame = 2,
}

impl DnwbCommand {
    /// Converts a wire `u16` opcode to [`DnwbCommand`].
    #[must_use]
    pub const fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Self::Reset),
            1 => Some(Self::RenderStem),
            2 => Some(Self::RenderFrame),
            _ => None,
        }
    }
}

/// Errors emitted by the `DNWB` binary protocol, `FaustRunner`, or `SpiceRunner`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DnwbError {
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u16),
    InvalidCommand(u16),
    NonZeroReservedBytes,
    PayloadTooLarge { actual: u64, max: u64 },
    TruncatedPacket { expected: usize, actual: usize },
    InvalidStimulusLayout(String),
    InvalidJsonMetadata(String),
    WatchdogTimeout { timeout_ms: u64 },
    IoFailure(String),
}

/// `64 B`-aligned binary header for out-of-process white-box distillation IPC (`b"DNWB"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DnwbHeader64B {
    pub magic: [u8; 4],
    pub version: u16,
    pub command: DnwbCommand,
    pub sample_rate_hz: u32,
    pub num_frames: u32,
    pub num_input_channels: u16,
    pub num_output_channels: u16,
    pub num_controls: u16,
    pub num_switches: u16,
    pub num_internal_nodes: u16,
    pub flags: u16,
    pub payload_byte_len: u64,
}

impl DnwbHeader64B {
    /// Serializes this header into an exact `64 B` little-endian array (`bytes[36..64]` zero-padded).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let wire = DnwbHeader64BWire {
            magic: self.magic,
            version: U16::new(self.version),
            command: U16::new(self.command as u16),
            sample_rate_hz: U32::new(self.sample_rate_hz),
            num_frames: U32::new(self.num_frames),
            num_input_channels: U16::new(self.num_input_channels),
            num_output_channels: U16::new(self.num_output_channels),
            num_controls: U16::new(self.num_controls),
            num_switches: U16::new(self.num_switches),
            num_internal_nodes: U16::new(self.num_internal_nodes),
            flags: U16::new(self.flags),
            payload_byte_len: U64::new(self.payload_byte_len),
            reserved: [0; 28],
        };
        let mut out = [0u8; 64];
        out.copy_from_slice(wire.as_bytes());
        out
    }

    /// Parses and validates a `64 B` little-endian `DNWB` header, enforcing zeroed reserved bytes
    /// and the `MAX_DNWB_PAYLOAD_BYTES` OOM guard.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnwbError> {
        let Ok((wire, _)) = DnwbHeader64BWire::ref_from_prefix(bytes) else {
            return Err(DnwbError::TruncatedPacket {
                expected: 64,
                actual: bytes.len(),
            });
        };
        if wire.magic != DNWB_MAGIC {
            return Err(DnwbError::InvalidMagic(wire.magic));
        }
        let version = wire.version.get();
        if version != DNWB_VERSION {
            return Err(DnwbError::UnsupportedVersion(version));
        }
        let cmd_raw = wire.command.get();
        let command = DnwbCommand::from_u16(cmd_raw).ok_or(DnwbError::InvalidCommand(cmd_raw))?;
        if wire.reserved.iter().any(|&b| b != 0) {
            return Err(DnwbError::NonZeroReservedBytes);
        }
        let payload_byte_len = wire.payload_byte_len.get();
        if payload_byte_len > MAX_DNWB_PAYLOAD_BYTES {
            return Err(DnwbError::PayloadTooLarge {
                actual: payload_byte_len,
                max: MAX_DNWB_PAYLOAD_BYTES,
            });
        }
        Ok(Self {
            magic: wire.magic,
            version,
            command,
            sample_rate_hz: wire.sample_rate_hz.get(),
            num_frames: wire.num_frames.get(),
            num_input_channels: wire.num_input_channels.get(),
            num_output_channels: wire.num_output_channels.get(),
            num_controls: wire.num_controls.get(),
            num_switches: wire.num_switches.get(),
            num_internal_nodes: wire.num_internal_nodes.get(),
            flags: wire.flags.get(),
            payload_byte_len,
        })
    }
}

/// Writes a `DNWB` stimulus request (`64 B` header + `stem_id` + controls + switches + all active
/// dry audio channels `in_1..in_4` + optional `[P][N]` trajectories & freeze window) to `writer`.
pub fn write_dnwb_stimulus_request<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    stimulus: &StemStimulus,
) -> Result<DnwbHeader64B, DnwbError> {
    write_dnwb_stimulus_request_with_switches(writer, command, stimulus, &[])
}

/// Writes a `DNWB` stimulus request including discrete switch states (`switches`).
pub fn write_dnwb_stimulus_request_with_switches<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
) -> Result<DnwbHeader64B, DnwbError> {
    wire::write_dnwb_stimulus_request_dyn(writer, command, stimulus, switches, false)
}

/// Writes a `DNWB` stimulus request encoding closed Rust ↔ Rust control metadata with `postcard`
/// (`DNWB_FLAG_POSTCARD_CTRL`, `ADR-0023`) alongside `zerocopy` `[F32]` audio/trajectory slabs.
pub fn write_dnwb_postcard_stimulus_request<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
) -> Result<DnwbHeader64B, DnwbError> {
    wire::write_dnwb_stimulus_request_dyn(writer, command, stimulus, switches, true)
}

/// Reads a `DNWB` stimulus request from `reader` (discarding optional discrete switches).
pub fn read_dnwb_stimulus_request<R: Read>(
    reader: &mut R,
) -> Result<(DnwbHeader64B, StemStimulus), DnwbError> {
    let (hdr, stim, _switches) = read_dnwb_stimulus_request_with_switches(reader)?;
    Ok((hdr, stim))
}

/// Reads a `DNWB` stimulus request and discrete switch vector from `reader`.
pub fn read_dnwb_stimulus_request_with_switches<R: Read>(
    reader: &mut R,
) -> Result<(DnwbHeader64B, StemStimulus, Vec<u8>), DnwbError> {
    wire::read_dnwb_stimulus_request_dyn(reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_dnwb_protocol_roundtrip_postcard_and_structured_guards() {
        assert_eq!(DnwbCommand::from_u16(0), Some(DnwbCommand::Reset));
        assert_eq!(DnwbCommand::from_u16(1), Some(DnwbCommand::RenderStem));
        assert_eq!(DnwbCommand::from_u16(2), Some(DnwbCommand::RenderFrame));
        assert_eq!(DnwbCommand::from_u16(99), None);

        assert!(DnwbHeader64B::from_bytes(&[0u8; 12]).is_err());
        let mut bad_magic = [0u8; 64];
        bad_magic[0..4].copy_from_slice(b"XXXX");
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_magic),
            Err(DnwbError::InvalidMagic(_))
        ));
        let mut bad_ver = [0u8; 64];
        bad_ver[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_ver[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_ver),
            Err(DnwbError::UnsupportedVersion(99))
        ));
        let mut bad_cmd = [0u8; 64];
        bad_cmd[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_cmd[4..6].copy_from_slice(&DNWB_VERSION.to_le_bytes());
        bad_cmd[6..8].copy_from_slice(&77u16.to_le_bytes());
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_cmd),
            Err(DnwbError::InvalidCommand(77))
        ));

        let mut bad_rsvd = [0u8; 64];
        bad_rsvd[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_rsvd[4..6].copy_from_slice(&DNWB_VERSION.to_le_bytes());
        bad_rsvd[6..8].copy_from_slice(&1u16.to_le_bytes());
        bad_rsvd[63] = 0xAA;
        assert_eq!(
            DnwbHeader64B::from_bytes(&bad_rsvd),
            Err(DnwbError::NonZeroReservedBytes)
        );

        let mut bad_oom = [0u8; 64];
        bad_oom[0..4].copy_from_slice(&DNWB_MAGIC);
        bad_oom[4..6].copy_from_slice(&DNWB_VERSION.to_le_bytes());
        bad_oom[6..8].copy_from_slice(&1u16.to_le_bytes());
        bad_oom[28..36].copy_from_slice(&(MAX_DNWB_PAYLOAD_BYTES + 1).to_le_bytes());
        assert!(matches!(
            DnwbHeader64B::from_bytes(&bad_oom),
            Err(DnwbError::PayloadTooLarge { .. })
        ));

        let stim = StemStimulus {
            stem_id: String::from("stem_4ch_96k"),
            sample_rate_hz: 96_000,
            in_1: vec![0.1, -0.2, 0.3, -0.4],
            in_2: Some(vec![-0.1, 0.2, -0.3, 0.4]),
            in_3: Some(vec![0.05, 0.05, -0.05, -0.05]),
            in_4: Some(vec![0.0, 0.1, 0.0, -0.1]),
            control_values: vec![0.25, -0.5],
            control_trajectories: Some(vec![vec![-0.5, -0.2, 0.2, 0.5], vec![0.4, 0.2, 0.0, -0.2]]),
            freeze_transition_window: Some([1, 3]),
        };
        let switches = vec![1u8, 2u8];
        let mut buf = Vec::new();
        let hdr = write_dnwb_stimulus_request_with_switches(
            &mut buf,
            DnwbCommand::RenderStem,
            &stim,
            &switches,
        )
        .unwrap();
        assert_eq!(hdr.num_input_channels, 4);
        assert_eq!(hdr.num_switches, 2);

        let (hdr_rt, stim_rt, sw_rt) =
            read_dnwb_stimulus_request_with_switches(&mut Cursor::new(buf.clone())).unwrap();
        assert_eq!(hdr, hdr_rt);
        assert_eq!(stim_rt, stim);
        assert_eq!(sw_rt, switches);

        // Postcard control envelope roundtrip
        let mut pc_buf = Vec::new();
        let pc_hdr = write_dnwb_postcard_stimulus_request(
            &mut pc_buf,
            DnwbCommand::RenderStem,
            &stim,
            &switches,
        )
        .unwrap();
        assert_ne!(pc_hdr.flags & DNWB_FLAG_POSTCARD_CTRL, 0);
        let (pc_hdr_rt, pc_stim_rt, pc_sw_rt) =
            read_dnwb_stimulus_request_with_switches(&mut Cursor::new(&pc_buf)).unwrap();
        assert_eq!(pc_hdr, pc_hdr_rt);
        assert_eq!(pc_stim_rt, stim);
        assert_eq!(pc_sw_rt, switches);

        assert!(read_dnwb_stimulus_request(&mut Cursor::new(&buf[..68])).is_err());

        let mut bad_pc_flags = pc_buf;
        bad_pc_flags[26..28].copy_from_slice(&(DNWB_FLAG_POSTCARD_CTRL).to_le_bytes());
        assert!(read_dnwb_stimulus_request(&mut Cursor::new(bad_pc_flags)).is_err());

        assert!(matches!(
            wire::read_u16_le(&[0u8; 1], &mut 0),
            Err(DnwbError::TruncatedPacket { .. })
        ));
        assert!(matches!(
            wire::read_u32_le(&[0u8; 3], &mut 0),
            Err(DnwbError::TruncatedPacket { .. })
        ));
        assert!(matches!(
            wire::read_u64_le(&[0u8; 7], &mut 0),
            Err(DnwbError::TruncatedPacket { .. })
        ));
        assert!(matches!(
            wire::read_f32_vec(&[0u8; 3], &mut 0, 1),
            Err(DnwbError::TruncatedPacket { .. })
        ));
    }
}
