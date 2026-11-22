#![forbid(unsafe_code)]
//! Declarative `zerocopy` `DnwbHeader64BWire` & `postcard` Control Envelope Codecs (`whitebox::wire`).

use core::mem::{align_of, offset_of, size_of};
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use zerocopy::byteorder::little_endian::{F32, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::{
    DNWB_CHANNEL_MASK_SHIFT, DNWB_FLAG_HAS_FREEZE_WINDOW, DNWB_FLAG_HAS_TRAJECTORIES,
    DNWB_FLAG_POSTCARD_CTRL, DNWB_FLAG_STRUCTURED_V1, DNWB_KNOWN_REQ_FLAGS, DNWB_MAGIC,
    DNWB_VERSION, DnwbCommand, DnwbError, DnwbHeader64B, MAX_DNWB_PAYLOAD_BYTES,
};
use crate::targets::StemStimulus;

/// Declarative `64 B` wire struct for `DNWB` white-box pipe framing (`ref[sdk.rust.zerocopy-wire-structs]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DnwbHeader64BWire {
    /// Magic `"DNWB"`.
    pub magic: [u8; 4],
    /// Framing version (`1`).
    pub version: U16,
    /// Command opcode (`0 = Reset`, `1 = RenderStem`, `2 = RenderFrame`).
    pub command: U16,
    /// Sample rate in Hz.
    pub sample_rate_hz: U32,
    /// Number of sample frames.
    pub num_frames: U32,
    /// Active input channel count (`0..=4`).
    pub num_input_channels: U16,
    /// Active output channel count (`0..=4`).
    pub num_output_channels: U16,
    /// Continuous control parameter count.
    pub num_controls: U16,
    /// Discrete switch count.
    pub num_switches: U16,
    /// Internal node trace count.
    pub num_internal_nodes: U16,
    /// `DNWB_FLAG_*` bitmask and bits `8..=11` channel mask.
    pub flags: U16,
    /// Byte length of payload immediately following this `64 B` header.
    pub payload_byte_len: U64,
    /// Zeroed alignment tail (`28 B`).
    pub reserved: [u8; 28],
}

const _: () = assert!(size_of::<DnwbHeader64BWire>() == 64);
const _: () = assert!(align_of::<DnwbHeader64BWire>() == 1);
const _: () = assert!(offset_of!(DnwbHeader64BWire, sample_rate_hz) == 8);
const _: () = assert!(offset_of!(DnwbHeader64BWire, num_input_channels) == 16);
const _: () = assert!(offset_of!(DnwbHeader64BWire, flags) == 26);
const _: () = assert!(offset_of!(DnwbHeader64BWire, payload_byte_len) == 28);
const _: () = assert!(offset_of!(DnwbHeader64BWire, reserved) == 36);

/// Variable-length closed Rust ↔ Rust `DNWB` request control metadata serialized with `postcard` (`ADR-0023`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DnwbPostcardRequestMeta {
    /// Stem identifier string.
    pub stem_id: String,
    /// Static continuous control values.
    pub control_values: Vec<f32>,
    /// Active discrete switch indices.
    pub switch_options: Vec<u8>,
    /// True if per-frame control trajectories follow the audio channels.
    pub has_trajectories: bool,
    /// Optional hold/freeze transition window `[start, end)`.
    pub freeze_transition_window: Option<[u64; 2]>,
}

/// Variable-length closed Rust ↔ Rust `DNWB` response metadata serialized with `postcard` (`ADR-0023`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DnwbPostcardResponseMeta {
    /// Stem identifier string.
    pub stem_id: String,
    /// Ordered `(node_name, num_samples)` descriptors for trailing `[F32]` traces.
    pub internal_node_meta: Vec<(String, u32)>,
}

#[inline]
pub(super) fn checked_add_usize(a: usize, b: usize) -> Result<usize, DnwbError> {
    a.checked_add(b)
        .ok_or_else(|| DnwbError::InvalidStimulusLayout(String::from("byte length overflow")))
}

#[inline]
pub(super) fn checked_mul_usize(a: usize, b: usize) -> Result<usize, DnwbError> {
    a.checked_mul(b).ok_or_else(|| {
        DnwbError::InvalidStimulusLayout(String::from("byte multiplication overflow"))
    })
}

pub(super) fn read_u16_le(buf: &[u8], off: &mut usize) -> Result<u16, DnwbError> {
    let end = checked_add_usize(*off, 2)?;
    let Some(slice) = buf.get(*off..end) else {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    };
    let val = U16::ref_from_bytes(slice)
        .map_err(|_| DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        })?
        .get();
    *off = end;
    Ok(val)
}

pub(super) fn read_u32_le(buf: &[u8], off: &mut usize) -> Result<u32, DnwbError> {
    let end = checked_add_usize(*off, 4)?;
    let Some(slice) = buf.get(*off..end) else {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    };
    let val = U32::ref_from_bytes(slice)
        .map_err(|_| DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        })?
        .get();
    *off = end;
    Ok(val)
}

pub(super) fn read_u64_le(buf: &[u8], off: &mut usize) -> Result<u64, DnwbError> {
    let end = checked_add_usize(*off, 8)?;
    let Some(slice) = buf.get(*off..end) else {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    };
    let val = U64::ref_from_bytes(slice)
        .map_err(|_| DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        })?
        .get();
    *off = end;
    Ok(val)
}

pub(super) fn ensure_finite_f32_slice(xs: &[f32], field: &str) -> Result<(), DnwbError> {
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "non-finite f32 sample in DNWB {field}"
        )));
    }
    Ok(())
}

pub(super) fn push_f32_slice_wire(dst: &mut Vec<u8>, xs: &[f32]) {
    for &x in xs {
        dst.extend_from_slice(F32::new(x).as_bytes());
    }
}

pub(super) fn read_f32_vec(
    buf: &[u8],
    off: &mut usize,
    count: usize,
) -> Result<Vec<f32>, DnwbError> {
    let byte_len = checked_mul_usize(count, 4)?;
    let end = checked_add_usize(*off, byte_len)?;
    let Some(slice) = buf.get(*off..end) else {
        return Err(DnwbError::TruncatedPacket {
            expected: end,
            actual: buf.len(),
        });
    };
    let words = <[F32]>::ref_from_bytes(slice).map_err(|_| DnwbError::TruncatedPacket {
        expected: end,
        actual: buf.len(),
    })?;
    let mut out = Vec::with_capacity(count);
    for w in words {
        let val = w.get();
        if !val.is_finite() {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "non-finite f32 sample in DNWB payload",
            )));
        }
        out.push(val);
    }
    *off = end;
    Ok(out)
}

struct ValidatedRequestChannels<'a> {
    n_frames: usize,
    opt_channels: [Option<&'a Vec<f32>>; 3],
    flags: u16,
    n_in_ch: usize,
}

fn validate_request_channels_and_trajectories<'a>(
    stimulus: &'a StemStimulus,
    switches: &[u8],
) -> Result<ValidatedRequestChannels<'a>, DnwbError> {
    let n_frames = stimulus.in_1.len();
    if n_frames > u32::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "num_frames exceeds u32::MAX",
        )));
    }
    if stimulus.stem_id.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "stem_id byte length exceeds u16::MAX",
        )));
    }
    if stimulus.control_values.len() > u16::MAX as usize || switches.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "control or switch count exceeds u16::MAX",
        )));
    }
    ensure_finite_f32_slice(&stimulus.control_values, "control_values")?;
    ensure_finite_f32_slice(&stimulus.in_1, "in_1")?;

    let opt_channels = [
        stimulus.in_2.as_ref(),
        stimulus.in_3.as_ref(),
        stimulus.in_4.as_ref(),
    ];
    let mut ch_mask: u16 = 1 << 0;
    let mut n_in_ch: usize = 1;
    for (idx, ch_opt) in opt_channels.iter().enumerate() {
        if let Some(ch) = ch_opt {
            if ch.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "input channel {} length {} != in_1 length {}",
                    idx + 2,
                    ch.len(),
                    n_frames
                )));
            }
            ensure_finite_f32_slice(ch, "optional input channel")?;
            ch_mask |= 1 << (idx + 1);
            n_in_ch += 1;
        }
    }
    let mut flags = DNWB_FLAG_STRUCTURED_V1 | (ch_mask << DNWB_CHANNEL_MASK_SHIFT);
    if let Some(ref trajs) = stimulus.control_trajectories {
        if trajs.len() > u16::MAX as usize || trajs.len() != stimulus.control_values.len() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "control_trajectories len {} != control_values len {}",
                trajs.len(),
                stimulus.control_values.len()
            )));
        }
        for (p, t) in trajs.iter().enumerate() {
            if t.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "control_trajectories[{p}] len {} != num_frames {n_frames}",
                    t.len()
                )));
            }
            ensure_finite_f32_slice(t, "control_trajectories")?;
        }
        flags |= DNWB_FLAG_HAS_TRAJECTORIES;
    }
    if let Some([start, end]) = stimulus.freeze_transition_window {
        if start > end || end > n_frames {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
            )));
        }
        flags |= DNWB_FLAG_HAS_FREEZE_WINDOW;
    }
    Ok(ValidatedRequestChannels {
        n_frames,
        opt_channels,
        flags,
        n_in_ch,
    })
}

pub(super) fn write_dnwb_stimulus_request_dyn(
    writer: &mut dyn Write,
    command: DnwbCommand,
    stimulus: &StemStimulus,
    switches: &[u8],
    use_postcard: bool,
) -> Result<DnwbHeader64B, DnwbError> {
    let validated = validate_request_channels_and_trajectories(stimulus, switches)?;
    let n_frames = validated.n_frames;
    let opt_channels = validated.opt_channels;
    let mut flags = validated.flags;
    let n_in_ch = validated.n_in_ch;
    let mut payload = Vec::new();
    if use_postcard {
        flags |= DNWB_FLAG_POSTCARD_CTRL;
        let meta = DnwbPostcardRequestMeta {
            stem_id: stimulus.stem_id.clone(),
            control_values: stimulus.control_values.clone(),
            switch_options: switches.to_vec(),
            has_trajectories: stimulus.control_trajectories.is_some(),
            freeze_transition_window: stimulus
                .freeze_transition_window
                .map(|[s, e]| [s as u64, e as u64]),
        };
        let encoded = postcard::to_allocvec(&meta)
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?;
        let meta_len = u32::try_from(encoded.len()).map_err(|_| {
            DnwbError::InvalidStimulusLayout(String::from("postcard metadata exceeds u32::MAX"))
        })?;
        payload.extend_from_slice(U32::new(meta_len).as_bytes());
        payload.extend_from_slice(&encoded);
        push_f32_slice_wire(&mut payload, &stimulus.in_1);
        for ch_opt in opt_channels.into_iter().flatten() {
            push_f32_slice_wire(&mut payload, ch_opt);
        }
        if let Some(ref trajs) = stimulus.control_trajectories {
            for t in trajs {
                push_f32_slice_wire(&mut payload, t);
            }
        }
    } else {
        let stem_bytes = stimulus.stem_id.as_bytes();
        payload.extend_from_slice(U16::new(stem_bytes.len() as u16).as_bytes());
        payload.extend_from_slice(stem_bytes);
        push_f32_slice_wire(&mut payload, &stimulus.control_values);
        payload.extend_from_slice(switches);
        push_f32_slice_wire(&mut payload, &stimulus.in_1);
        for ch_opt in opt_channels.into_iter().flatten() {
            push_f32_slice_wire(&mut payload, ch_opt);
        }
        if let Some(ref trajs) = stimulus.control_trajectories {
            payload.extend_from_slice(U16::new(trajs.len() as u16).as_bytes());
            for t in trajs {
                push_f32_slice_wire(&mut payload, t);
            }
        }
        if let Some([start, end]) = stimulus.freeze_transition_window {
            payload.extend_from_slice(U64::new(start as u64).as_bytes());
            payload.extend_from_slice(U64::new(end as u64).as_bytes());
        }
    }

    let payload_byte_len = payload.len() as u64;
    if payload_byte_len > MAX_DNWB_PAYLOAD_BYTES {
        return Err(DnwbError::PayloadTooLarge {
            actual: payload_byte_len,
            max: MAX_DNWB_PAYLOAD_BYTES,
        });
    }

    let hdr = DnwbHeader64B {
        magic: DNWB_MAGIC,
        version: DNWB_VERSION,
        command,
        sample_rate_hz: stimulus.sample_rate_hz,
        num_frames: n_frames as u32,
        num_input_channels: n_in_ch as u16,
        num_output_channels: 0,
        num_controls: stimulus.control_values.len() as u16,
        num_switches: switches.len() as u16,
        num_internal_nodes: 0,
        flags,
        payload_byte_len,
    };

    writer
        .write_all(&hdr.to_bytes())
        .and_then(|()| writer.write_all(&payload))
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    Ok(hdr)
}

fn resolve_input_channel_mask(hdr: &DnwbHeader64B) -> Result<u16, DnwbError> {
    let n_frames = hdr.num_frames as usize;
    let ch_mask = if hdr.flags & DNWB_FLAG_STRUCTURED_V1 != 0 {
        let mask = (hdr.flags >> DNWB_CHANNEL_MASK_SHIFT) & 0x0f;
        if mask.count_ones() as u16 != hdr.num_input_channels {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "channel mask popcount {} != num_input_channels {}",
                mask.count_ones(),
                hdr.num_input_channels
            )));
        }
        mask
    } else {
        if (hdr.flags >> DNWB_CHANNEL_MASK_SHIFT) & 0x0f != 0 {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "channel mask bits set without DNWB_FLAG_STRUCTURED_V1",
            )));
        }
        match hdr.num_input_channels {
            0 => 0,
            1 => 0b0001,
            2 => 0b0011,
            3 => 0b0111,
            4 => 0b1111,
            n => {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "unsupported legacy num_input_channels {n} (max 4)"
                )));
            }
        }
    };
    if (ch_mask & 1 == 0) && (hdr.num_input_channels > 0 || n_frames > 0) {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "primary input channel in_1 bit missing in request channel mask",
        )));
    }
    Ok(ch_mask)
}

struct InputChannelSlabs {
    in_1: Vec<f32>,
    in_2: Option<Vec<f32>>,
    in_3: Option<Vec<f32>>,
    in_4: Option<Vec<f32>>,
}

fn read_input_channel_slabs(
    payload: &[u8],
    off: &mut usize,
    ch_mask: u16,
    n_frames: usize,
) -> Result<InputChannelSlabs, DnwbError> {
    let in_1 = if ch_mask & (1 << 0) != 0 {
        read_f32_vec(payload, off, n_frames)?
    } else {
        Vec::new()
    };
    let in_2 = if ch_mask & (1 << 1) != 0 {
        Some(read_f32_vec(payload, off, n_frames)?)
    } else {
        None
    };
    let in_3 = if ch_mask & (1 << 2) != 0 {
        Some(read_f32_vec(payload, off, n_frames)?)
    } else {
        None
    };
    let in_4 = if ch_mask & (1 << 3) != 0 {
        Some(read_f32_vec(payload, off, n_frames)?)
    } else {
        None
    };
    Ok(InputChannelSlabs {
        in_1,
        in_2,
        in_3,
        in_4,
    })
}

pub(super) fn read_dnwb_stimulus_request_dyn(
    reader: &mut dyn Read,
) -> Result<(DnwbHeader64B, StemStimulus, Vec<u8>), DnwbError> {
    let mut hdr_buf = [0u8; 64];
    reader
        .read_exact(&mut hdr_buf)
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    let hdr = DnwbHeader64B::from_bytes(&hdr_buf)?;
    if hdr.flags & !DNWB_KNOWN_REQ_FLAGS != 0 {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "unknown DNWB request flags {:#06x}",
            hdr.flags
        )));
    }
    if (hdr.flags & DNWB_FLAG_POSTCARD_CTRL) != 0 && (hdr.flags & DNWB_FLAG_STRUCTURED_V1) == 0 {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "DNWB_FLAG_POSTCARD_CTRL requires DNWB_FLAG_STRUCTURED_V1",
        )));
    }
    let payload_len =
        usize::try_from(hdr.payload_byte_len).map_err(|_| DnwbError::PayloadTooLarge {
            actual: hdr.payload_byte_len,
            max: MAX_DNWB_PAYLOAD_BYTES,
        })?;
    let mut payload = vec![0u8; payload_len];
    reader
        .read_exact(&mut payload)
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;

    let mut off = 0usize;
    let n_ctrl = hdr.num_controls as usize;
    let n_sw = hdr.num_switches as usize;
    let n_frames = hdr.num_frames as usize;
    let ch_mask = resolve_input_channel_mask(&hdr)?;

    if (hdr.flags & DNWB_FLAG_POSTCARD_CTRL) != 0 {
        let meta_len = read_u32_le(&payload, &mut off)? as usize;
        let meta_end = checked_add_usize(off, meta_len)?;
        let Some(meta_slice) = payload.get(off..meta_end) else {
            return Err(DnwbError::TruncatedPacket {
                expected: meta_end,
                actual: payload.len(),
            });
        };
        let meta: DnwbPostcardRequestMeta = postcard::from_bytes(meta_slice)
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?;
        off = meta_end;
        if meta.control_values.len() != n_ctrl || meta.switch_options.len() != n_sw {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "postcard control/switch count ({}, {}) != header ({n_ctrl}, {n_sw})",
                meta.control_values.len(),
                meta.switch_options.len()
            )));
        }
        ensure_finite_f32_slice(&meta.control_values, "control_values")?;
        let slabs = read_input_channel_slabs(&payload, &mut off, ch_mask, n_frames)?;
        let control_trajectories = if meta.has_trajectories {
            let mut trajs = Vec::with_capacity(n_ctrl);
            for _ in 0..n_ctrl {
                trajs.push(read_f32_vec(&payload, &mut off, n_frames)?);
            }
            Some(trajs)
        } else {
            None
        };
        let freeze_transition_window = match meta.freeze_transition_window {
            Some([s, e]) => {
                let start = usize::try_from(s).map_err(|_| {
                    DnwbError::InvalidStimulusLayout(String::from("freeze window start overflow"))
                })?;
                let end = usize::try_from(e).map_err(|_| {
                    DnwbError::InvalidStimulusLayout(String::from("freeze window end overflow"))
                })?;
                if start > end || end > n_frames {
                    return Err(DnwbError::InvalidStimulusLayout(format!(
                        "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
                    )));
                }
                Some([start, end])
            }
            None => None,
        };
        if off != payload.len() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "trailing unconsumed stimulus bytes: consumed {off} of {}",
                payload.len()
            )));
        }
        return Ok((
            hdr,
            StemStimulus {
                stem_id: meta.stem_id,
                sample_rate_hz: hdr.sample_rate_hz,
                in_1: slabs.in_1,
                in_2: slabs.in_2,
                in_3: slabs.in_3,
                in_4: slabs.in_4,
                control_values: meta.control_values,
                control_trajectories,
                freeze_transition_window,
            },
            meta.switch_options,
        ));
    }

    let stem_id = if hdr.flags & DNWB_FLAG_STRUCTURED_V1 != 0 {
        let id_len = read_u16_le(&payload, &mut off)? as usize;
        let end = checked_add_usize(off, id_len)?;
        let Some(slice) = payload.get(off..end) else {
            return Err(DnwbError::TruncatedPacket {
                expected: end,
                actual: payload.len(),
            });
        };
        let s = std::str::from_utf8(slice)
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?
            .to_string();
        off = end;
        s
    } else {
        String::from("dnwb_stem")
    };

    let control_values = read_f32_vec(&payload, &mut off, n_ctrl)?;
    let sw_end = checked_add_usize(off, n_sw)?;
    let Some(sw_slice) = payload.get(off..sw_end) else {
        return Err(DnwbError::TruncatedPacket {
            expected: sw_end,
            actual: payload.len(),
        });
    };
    let switches = sw_slice.to_vec();
    off = sw_end;

    let slabs = read_input_channel_slabs(&payload, &mut off, ch_mask, n_frames)?;

    let control_trajectories = if hdr.flags & DNWB_FLAG_HAS_TRAJECTORIES != 0 {
        let p_cnt = read_u16_le(&payload, &mut off)? as usize;
        if p_cnt != n_ctrl {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "control_trajectories count {p_cnt} != num_controls {n_ctrl}"
            )));
        }
        let mut trajs = Vec::with_capacity(p_cnt);
        for _ in 0..p_cnt {
            trajs.push(read_f32_vec(&payload, &mut off, n_frames)?);
        }
        Some(trajs)
    } else {
        None
    };

    let freeze_transition_window = if hdr.flags & DNWB_FLAG_HAS_FREEZE_WINDOW != 0 {
        let start_u64 = read_u64_le(&payload, &mut off)?;
        let end_u64 = read_u64_le(&payload, &mut off)?;
        let start = usize::try_from(start_u64).map_err(|_| {
            DnwbError::InvalidStimulusLayout(String::from("freeze window start overflow"))
        })?;
        let end = usize::try_from(end_u64).map_err(|_| {
            DnwbError::InvalidStimulusLayout(String::from("freeze window end overflow"))
        })?;
        if start > end || end > n_frames {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "invalid freeze_transition_window [{start}, {end}] for num_frames {n_frames}"
            )));
        }
        Some([start, end])
    } else {
        None
    };

    if off != payload.len() {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "trailing unconsumed stimulus bytes: consumed {off} of {}",
            payload.len()
        )));
    }

    Ok((
        hdr,
        StemStimulus {
            stem_id,
            sample_rate_hz: hdr.sample_rate_hz,
            in_1: slabs.in_1,
            in_2: slabs.in_2,
            in_3: slabs.in_3,
            in_4: slabs.in_4,
            control_values,
            control_trajectories,
            freeze_transition_window,
        },
        switches,
    ))
}
