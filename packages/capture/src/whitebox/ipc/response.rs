#![forbid(unsafe_code)]
//! `DNWB` Stem Response Framing (`out_1..out_4` + `internal_nodes`) with `zerocopy` & `postcard`
//! (`whitebox::ipc::response`).

use std::collections::BTreeMap;
use std::io::{Read, Write};

use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::{U16, U32};

use super::super::wire::{
    DnwbPostcardResponseMeta, checked_add_usize, checked_mul_usize, ensure_finite_f32_slice,
    push_f32_slice_wire, read_f32_vec, read_u16_le, read_u32_le,
};
use super::super::{
    DNWB_CHANNEL_MASK_SHIFT, DNWB_FLAG_POSTCARD_CTRL, DNWB_FLAG_STRUCTURED_V1, DNWB_MAGIC,
    DNWB_VERSION, DnwbCommand, DnwbError, DnwbHeader64B, MAX_DNWB_PAYLOAD_BYTES,
};
use crate::targets::StemResponse;

/// Writes a `DNWB` stem response (`64 B` header + `stem_id` + `out_1..out_4` + `internal_nodes`) to `writer`.
pub fn write_dnwb_stem_response<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    response: &StemResponse,
) -> Result<DnwbHeader64B, DnwbError> {
    write_dnwb_stem_response_dyn(writer, command, response, false)
}

/// Writes a `DNWB` stem response using `postcard` control/node metadata (`DNWB_FLAG_POSTCARD_CTRL`, `ADR-0023`).
pub fn write_dnwb_postcard_stem_response<W: Write>(
    writer: &mut W,
    command: DnwbCommand,
    response: &StemResponse,
) -> Result<DnwbHeader64B, DnwbError> {
    write_dnwb_stem_response_dyn(writer, command, response, true)
}

fn write_dnwb_stem_response_dyn(
    writer: &mut dyn Write,
    command: DnwbCommand,
    response: &StemResponse,
    use_postcard: bool,
) -> Result<DnwbHeader64B, DnwbError> {
    let n_frames = response.out_1.len();
    if n_frames > u32::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "response num_frames exceeds u32::MAX",
        )));
    }
    let stem_bytes = response.stem_id.as_bytes();
    if stem_bytes.len() > u16::MAX as usize || response.internal_nodes.len() > u16::MAX as usize {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "response stem_id or internal_nodes count exceeds u16::MAX",
        )));
    }

    ensure_finite_f32_slice(&response.out_1, "out_1")?;
    let opt_outputs = [
        (1u16, response.out_2.as_ref()),
        (2u16, response.out_3.as_ref()),
        (3u16, response.out_4.as_ref()),
    ];
    let mut ch_mask: u16 = 1 << 0;
    let mut n_out_ch: usize = 1;
    for (bit, ch_opt) in opt_outputs {
        if let Some(ch) = ch_opt {
            if ch.len() != n_frames {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "output channel {} length {} != out_1 length {n_frames}",
                    bit + 1,
                    ch.len()
                )));
            }
            ensure_finite_f32_slice(ch, "optional output channel")?;
            ch_mask |= 1 << bit;
            n_out_ch += 1;
        }
    }
    for (name, trace) in &response.internal_nodes {
        if name.len() > u16::MAX as usize || trace.len() > u32::MAX as usize {
            return Err(DnwbError::InvalidStimulusLayout(String::from(
                "internal node name or trace length overflow",
            )));
        }
        ensure_finite_f32_slice(trace, "internal_nodes")?;
    }

    let mut flags = DNWB_FLAG_STRUCTURED_V1 | (ch_mask << DNWB_CHANNEL_MASK_SHIFT);
    let mut payload = Vec::new();
    if use_postcard {
        flags |= DNWB_FLAG_POSTCARD_CTRL;
        let node_meta: Vec<(String, u32)> = response
            .internal_nodes
            .iter()
            .map(|(k, v)| (k.clone(), v.len() as u32))
            .collect();
        let meta = DnwbPostcardResponseMeta {
            stem_id: response.stem_id.clone(),
            internal_node_meta: node_meta,
        };
        let encoded = postcard::to_allocvec(&meta)
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?;
        let meta_len = u32::try_from(encoded.len()).map_err(|_| {
            DnwbError::InvalidStimulusLayout(String::from("postcard metadata exceeds u32::MAX"))
        })?;
        payload.extend_from_slice(U32::new(meta_len).as_bytes());
        payload.extend_from_slice(&encoded);
        push_f32_slice_wire(&mut payload, &response.out_1);
        for (_, ch_opt) in opt_outputs {
            if let Some(ch) = ch_opt {
                push_f32_slice_wire(&mut payload, ch);
            }
        }
        for trace in response.internal_nodes.values() {
            push_f32_slice_wire(&mut payload, trace);
        }
    } else {
        let mut cap = checked_add_usize(2, stem_bytes.len())?;
        let out_samples = checked_mul_usize(n_out_ch, n_frames)?;
        cap = checked_add_usize(cap, checked_mul_usize(out_samples, 4)?)?;
        for (name, trace) in &response.internal_nodes {
            let entry_hdr = checked_add_usize(6, name.len())?;
            let entry_data = checked_mul_usize(trace.len(), 4)?;
            cap = checked_add_usize(cap, checked_add_usize(entry_hdr, entry_data)?)?;
        }
        payload.reserve(cap);
        payload.extend_from_slice(U16::new(stem_bytes.len() as u16).as_bytes());
        payload.extend_from_slice(stem_bytes);
        push_f32_slice_wire(&mut payload, &response.out_1);
        for (_, ch_opt) in opt_outputs {
            if let Some(ch) = ch_opt {
                push_f32_slice_wire(&mut payload, ch);
            }
        }
        for (name, trace) in &response.internal_nodes {
            let nb = name.as_bytes();
            payload.extend_from_slice(U16::new(nb.len() as u16).as_bytes());
            payload.extend_from_slice(nb);
            payload.extend_from_slice(U32::new(trace.len() as u32).as_bytes());
            push_f32_slice_wire(&mut payload, trace);
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
        sample_rate_hz: response.sample_rate_hz,
        num_frames: n_frames as u32,
        num_input_channels: 0,
        num_output_channels: n_out_ch as u16,
        num_controls: 0,
        num_switches: 0,
        num_internal_nodes: response.internal_nodes.len() as u16,
        flags,
        payload_byte_len,
    };

    writer
        .write_all(&hdr.to_bytes())
        .and_then(|()| writer.write_all(&payload))
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    Ok(hdr)
}

/// Reads a `DNWB` stem response (`out_1..out_4` + `internal_nodes`) from `reader`.
pub fn read_dnwb_stem_response<R: Read>(
    reader: &mut R,
) -> Result<(DnwbHeader64B, StemResponse), DnwbError> {
    read_dnwb_stem_response_dyn(reader)
}

fn read_dnwb_stem_response_dyn(
    reader: &mut dyn Read,
) -> Result<(DnwbHeader64B, StemResponse), DnwbError> {
    let mut hdr_buf = [0u8; 64];
    reader
        .read_exact(&mut hdr_buf)
        .map_err(|e| DnwbError::IoFailure(e.to_string()))?;
    let hdr = DnwbHeader64B::from_bytes(&hdr_buf)?;
    const KNOWN_RESP_FLAGS: u16 =
        DNWB_FLAG_STRUCTURED_V1 | DNWB_FLAG_POSTCARD_CTRL | (0x0f << DNWB_CHANNEL_MASK_SHIFT);
    if hdr.flags & DNWB_FLAG_STRUCTURED_V1 == 0 || (hdr.flags & !KNOWN_RESP_FLAGS != 0) {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "invalid or unsupported DNWB response flags {:#06x}",
            hdr.flags
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
    let n_frames = hdr.num_frames as usize;
    let ch_mask = (hdr.flags >> DNWB_CHANNEL_MASK_SHIFT) & 0x0f;
    if ch_mask.count_ones() as u16 != hdr.num_output_channels {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "output channel mask popcount {} != num_output_channels {}",
            ch_mask.count_ones(),
            hdr.num_output_channels
        )));
    }
    if (ch_mask & 1 == 0) && (hdr.num_output_channels > 0 || n_frames > 0) {
        return Err(DnwbError::InvalidStimulusLayout(String::from(
            "primary output channel out_1 bit missing in response channel mask",
        )));
    }

    if (hdr.flags & DNWB_FLAG_POSTCARD_CTRL) != 0 {
        let meta_len = read_u32_le(&payload, &mut off)? as usize;
        let meta_end = checked_add_usize(off, meta_len)?;
        let Some(meta_slice) = payload.get(off..meta_end) else {
            return Err(DnwbError::TruncatedPacket {
                expected: meta_end,
                actual: payload.len(),
            });
        };
        let meta: DnwbPostcardResponseMeta = postcard::from_bytes(meta_slice)
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?;
        off = meta_end;
        if meta.internal_node_meta.len() != hdr.num_internal_nodes as usize {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "postcard internal_node_meta count {} != num_internal_nodes {}",
                meta.internal_node_meta.len(),
                hdr.num_internal_nodes
            )));
        }
        let out_1 = if ch_mask & (1 << 0) != 0 {
            read_f32_vec(&payload, &mut off, n_frames)?
        } else {
            Vec::new()
        };
        let out_2 = if ch_mask & (1 << 1) != 0 {
            Some(read_f32_vec(&payload, &mut off, n_frames)?)
        } else {
            None
        };
        let out_3 = if ch_mask & (1 << 2) != 0 {
            Some(read_f32_vec(&payload, &mut off, n_frames)?)
        } else {
            None
        };
        let out_4 = if ch_mask & (1 << 3) != 0 {
            Some(read_f32_vec(&payload, &mut off, n_frames)?)
        } else {
            None
        };
        let mut internal_nodes = BTreeMap::new();
        for (name, len_u32) in meta.internal_node_meta {
            let trace = read_f32_vec(&payload, &mut off, len_u32 as usize)?;
            if internal_nodes.insert(name.clone(), trace).is_some() {
                return Err(DnwbError::InvalidStimulusLayout(format!(
                    "duplicate internal_nodes key '{name}' in DNWB response"
                )));
            }
        }
        if off != payload.len() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "trailing unconsumed response bytes: consumed {off} of {}",
                payload.len()
            )));
        }
        return Ok((
            hdr,
            StemResponse {
                stem_id: meta.stem_id,
                sample_rate_hz: hdr.sample_rate_hz,
                out_1,
                out_2,
                out_3,
                out_4,
                internal_nodes,
            },
        ));
    }

    let id_len = read_u16_le(&payload, &mut off)? as usize;
    let id_end = checked_add_usize(off, id_len)?;
    let Some(id_slice) = payload.get(off..id_end) else {
        return Err(DnwbError::TruncatedPacket {
            expected: id_end,
            actual: payload.len(),
        });
    };
    let stem_id = std::str::from_utf8(id_slice)
        .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?
        .to_string();
    off = id_end;

    let out_1 = if ch_mask & (1 << 0) != 0 {
        read_f32_vec(&payload, &mut off, n_frames)?
    } else {
        Vec::new()
    };
    let out_2 = if ch_mask & (1 << 1) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };
    let out_3 = if ch_mask & (1 << 2) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };
    let out_4 = if ch_mask & (1 << 3) != 0 {
        Some(read_f32_vec(&payload, &mut off, n_frames)?)
    } else {
        None
    };

    let mut internal_nodes = BTreeMap::new();
    for _ in 0..hdr.num_internal_nodes {
        let name_len = read_u16_le(&payload, &mut off)? as usize;
        let name_end = checked_add_usize(off, name_len)?;
        let Some(name_slice) = payload.get(off..name_end) else {
            return Err(DnwbError::TruncatedPacket {
                expected: name_end,
                actual: payload.len(),
            });
        };
        let name = std::str::from_utf8(name_slice)
            .map_err(|e| DnwbError::InvalidStimulusLayout(e.to_string()))?
            .to_string();
        off = name_end;
        let trace_len = read_u32_le(&payload, &mut off)? as usize;
        let trace = read_f32_vec(&payload, &mut off, trace_len)?;
        if internal_nodes.insert(name.clone(), trace).is_some() {
            return Err(DnwbError::InvalidStimulusLayout(format!(
                "duplicate internal_nodes key '{name}' in DNWB response"
            )));
        }
    }

    if off != payload.len() {
        return Err(DnwbError::InvalidStimulusLayout(format!(
            "trailing unconsumed response bytes: consumed {off} of {}",
            payload.len()
        )));
    }

    Ok((
        hdr,
        StemResponse {
            stem_id,
            sample_rate_hz: hdr.sample_rate_hz,
            out_1,
            out_2,
            out_3,
            out_4,
            internal_nodes,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_postcard_response_roundtrip_and_guards() {
        let mut internal_nodes = BTreeMap::new();
        internal_nodes.insert(String::from("node_a"), vec![0.25, -0.25, 0.5, -0.5]);
        let resp = StemResponse {
            stem_id: String::from("stem_pc_resp"),
            sample_rate_hz: 96_000,
            out_1: vec![0.2, -0.4, 0.6, -0.8],
            out_2: Some(vec![-0.2, 0.4, -0.6, 0.8]),
            out_3: None,
            out_4: None,
            internal_nodes,
        };
        let mut wire = Vec::new();
        let hdr =
            write_dnwb_postcard_stem_response(&mut wire, DnwbCommand::RenderStem, &resp).unwrap();
        assert_ne!(hdr.flags & DNWB_FLAG_POSTCARD_CTRL, 0);
        let (hdr_rt, resp_rt) = read_dnwb_stem_response(&mut Cursor::new(&wire)).unwrap();
        assert_eq!(hdr_rt, hdr);
        assert_eq!(resp_rt, resp);

        let mut bad_nodes_hdr = hdr;
        bad_nodes_hdr.num_internal_nodes = 9;
        let mut bad_nodes_wire = Vec::from(bad_nodes_hdr.to_bytes());
        bad_nodes_wire.extend_from_slice(&wire[64..]);
        assert!(read_dnwb_stem_response(&mut Cursor::new(bad_nodes_wire)).is_err());
    }
}
