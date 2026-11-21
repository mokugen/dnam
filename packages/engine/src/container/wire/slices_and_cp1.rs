//! Model-Declared Delay Histories, `ModalConvolver` Separate `64 B`-Aligned Slices, and `CP1` Packed Control
//! Projections (`ref[container.dnam.delay-histories]`, `ref[container.dnam.load-validation]`,
//! `ref[container.dnam.cabinet-slices]`, `ref[container.dnam.cp1-profile]`).

pub mod delay_and_convolver;

pub use delay_and_convolver::{
    DeclaredDelayHistory, DelayHistoryEntry, parse_canonical_reference_samples,
    validate_convolver_tier_slices, validate_delay_histories, validate_delay_history_entries,
};

use alloc::string::ToString;
use alloc::vec::Vec;
use core::mem::{align_of, size_of};

use dnam_vm::control::Cp1ContributorSlot;
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::layout::ContainerWireError;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};

#[inline]
fn read_finite_f32_words_rt(bytes: &[u8]) -> Result<&[F32], ContainerWireError> {
    let words = <[F32]>::ref_from_bytes(bytes)
        .map_err(|_| ContainerWireError::SchemaStatic("invalid f32 slice"))?;
    for w in words {
        if !w.get().is_finite() {
            return Err(ContainerWireError::SchemaStatic("nonfinite coefficient"));
        }
    }
    Ok(words)
}

#[inline]
fn read_finite_f32_words(bytes: &[u8]) -> Result<&[F32], DnamError> {
    read_finite_f32_words_rt(bytes).map_err(ContainerWireError::into_dnam_error)
}

/// Wire header (`8 B`) for a self-describing `CP1\0` stream in Region 4 `shared_extent`
/// (`ref[container.dnam.cp1-profile]`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Cp1StreamHeaderWire {
    pub magic: [u8; 4],
    pub count: U16,
    pub flags: u8,
    pub reserved: u8,
}

const _: () =
    assert!(size_of::<Cp1StreamHeaderWire>() == 8 && align_of::<Cp1StreamHeaderWire>() == 1);

/// Wire directory entry (`4 B`) inside a `CP1\0` stream (`ref[container.dnam.cp1-profile]`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Cp1StreamDirEntryWire {
    pub control_idx: U16,
    pub raw_kl: u8,
    pub aux: u8,
}

const _: () =
    assert!(size_of::<Cp1StreamDirEntryWire>() == 4 && align_of::<Cp1StreamDirEntryWire>() == 1);

/// Packed `4 B` contributor record descriptor header (`ref[container.dnam.cp1-profile]`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Cp1ContributorRecordHeaderWire {
    pub control_idx: U16,
    pub kind_and_lanes: u8,
    pub aux: u8,
}

const _: () = assert!(
    size_of::<Cp1ContributorRecordHeaderWire>() == 4
        && align_of::<Cp1ContributorRecordHeaderWire>() == 1
);

/// Magic bytes for a packed `CP1\0` control projection contributor stream in `shared_extent`.
pub const CP1_MAGIC: [u8; 4] = *b"CP1\0";

/// Returns the byte length of a single `CP1` contributor record with `lanes` (`1..=3`):
/// `4 * (9 * lanes + 4)` in `Arena 0` (`ultra == false`) or `4 * (8 * lanes + 4)` in `Arena 3`
/// (`ultra == true`, `ref[container.dnam.cp1-profile]`).
#[inline]
pub const fn cp1_contributor_record_bytes(lanes: usize, ultra_delta: bool) -> Option<usize> {
    if lanes < 1 || lanes > 3 {
        None
    } else if ultra_delta {
        Some(4 * (8 * lanes + 4))
    } else {
        Some(4 * (9 * lanes + 4))
    }
}

/// Decoded `CP1` contributor record (`ref[container.dnam.cp1-profile]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Cp1DecodedRecord {
    pub lanes: usize,
    pub base_main: Vec<f32>,
    pub embedding: Vec<f32>,
    pub base_factors: [f32; 4],
    pub extra_main: Vec<f32>,
    pub extra_factors: Option<[f32; 4]>,
}

/// Decodes a stream of packed `CP1` contributor records from `base_bytes` (`Arena 0`) and optional
/// `ultra_delta_bytes` (`Arena 3`) for the given sequence of contributor lane counts (`ref[container.dnam.cp1-profile]`).
pub fn decode_cp1_contributor_stream(
    base_bytes: &[u8],
    ultra_delta_bytes: Option<&[u8]>,
    contributor_lanes: &[usize],
) -> Result<Vec<Cp1DecodedRecord>, DnamError> {
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for &lanes in contributor_lanes {
        expected_base += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
    }
    reject_if(
        base_bytes.len() != expected_base,
        "incorrect base tensor length",
    )?;
    if let Some(ub) = ultra_delta_bytes
        && ub.len() != expected_ultra
    {
        return Err(DnamError::SchemaValidationError(
            "incorrect delta tensor length".to_string(),
        ));
    }

    let read_f32_vec = |bytes: &[u8]| -> Result<Vec<f32>, DnamError> {
        Ok(read_finite_f32_words(bytes)?
            .iter()
            .map(|w| w.get())
            .collect())
    };

    let mut records = Vec::with_capacity(contributor_lanes.len());
    let mut b_off = 0usize;
    let mut u_off = 0usize;
    for &lanes in contributor_lanes {
        let b_len = cp1_contributor_record_bytes(lanes, false).unwrap();
        let b_vals = read_f32_vec(&base_bytes[b_off..b_off + b_len])?;
        b_off += b_len;
        let main_end = 8 * lanes;
        let base_main = b_vals[..main_end].to_vec();
        let embedding = b_vals[main_end..main_end + lanes].to_vec();
        let base_factors = [
            b_vals[main_end + lanes],
            b_vals[main_end + lanes + 1],
            b_vals[main_end + lanes + 2],
            b_vals[main_end + lanes + 3],
        ];
        let (extra_main, extra_factors) = if let Some(ub) = ultra_delta_bytes {
            let u_len = cp1_contributor_record_bytes(lanes, true).unwrap();
            let u_vals = read_f32_vec(&ub[u_off..u_off + u_len])?;
            u_off += u_len;
            (
                u_vals[..main_end].to_vec(),
                Some([
                    u_vals[main_end],
                    u_vals[main_end + 1],
                    u_vals[main_end + 2],
                    u_vals[main_end + 3],
                ]),
            )
        } else {
            (Vec::new(), None)
        };
        records.push(Cp1DecodedRecord {
            lanes,
            base_main,
            embedding,
            base_factors,
            extra_main,
            extra_factors,
        });
    }
    Ok(records)
}

/// Decodes a stream of packed `CP1` contributor records directly into a caller-supplied pre-allocated
/// [`Cp1ContributorSlot`] slice without heap allocation (`ref[container.dnam.cp1-profile]`).
pub fn decode_cp1_contributor_stream_into_slice(
    base_bytes: &[u8],
    ultra_delta_bytes: Option<&[u8]>,
    contributor_lanes: &[usize],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, DnamError> {
    reject_if(
        contributor_lanes.len() > out_slots.len(),
        "contributor count exceeds slot capacity",
    )?;
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for &lanes in contributor_lanes {
        expected_base += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
    }
    reject_if(
        base_bytes.len() < expected_base,
        "incorrect base tensor length",
    )?;
    if let Some(ub) = ultra_delta_bytes
        && ub.len() < expected_ultra
    {
        return Err(DnamError::SchemaValidationError(
            "incorrect delta tensor length".to_string(),
        ));
    }

    let base_words = read_finite_f32_words(&base_bytes[..expected_base])?;
    let ultra_words = match ultra_delta_bytes {
        Some(ub) => Some(read_finite_f32_words(&ub[..expected_ultra])?),
        None => None,
    };

    let mut b_word_off = 0usize;
    let mut u_word_off = 0usize;
    for (slot_idx, &lanes) in contributor_lanes.iter().enumerate() {
        let b_words = 9 * lanes + 4;
        let b_rec = &base_words[b_word_off..b_word_off + b_words];
        b_word_off += b_words;

        let mut slot = Cp1ContributorSlot::zero();
        slot.lanes = lanes as u8;
        slot.control_idx = slot_idx as u16;
        slot.aux = 0xFF;

        for k in 0..lanes {
            for r in 0..8 {
                slot.main_base[k][r] = b_rec[k * 8 + r].get();
            }
            slot.emb[k] = b_rec[8 * lanes + k].get();
        }
        let v_start = 9 * lanes;
        for f in 0..4 {
            slot.v_base[f] = b_rec[v_start + f].get();
        }

        if let Some(uw) = ultra_words {
            let u_words = 8 * lanes + 4;
            let u_rec = &uw[u_word_off..u_word_off + u_words];
            u_word_off += u_words;
            for k in 0..lanes {
                for r in 0..8 {
                    slot.main_ultra[k][r] = u_rec[k * 8 + r].get();
                }
            }
            let uv_start = 8 * lanes;
            for f in 0..4 {
                slot.v_ultra[f] = u_rec[uv_start + f].get();
            }
            slot.has_ultra = true;
        }
        out_slots[slot_idx] = slot;
    }
    Ok(contributor_lanes.len())
}

/// Zero-drop-glue (`Copy` error) `CP1\0` stream decoder for real-time container reloaders.
pub(crate) fn decode_cp1_shared_stream_into_slice_rt(
    cp1_slice: &[u8],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, ContainerWireError> {
    let Ok((hdr, after_hdr)) = Cp1StreamHeaderWire::ref_from_prefix(cp1_slice) else {
        return Ok(0);
    };
    if hdr.magic != CP1_MAGIC {
        return Ok(0);
    }
    let count = hdr.count.get() as usize;
    if count == 0 {
        return Ok(0);
    }
    if hdr.reserved != 0 {
        return Err(ContainerWireError::SchemaStatic(
            "invalid CP1 shared stream header",
        ));
    }
    let has_ultra = (hdr.flags & 1) != 0;
    let Ok((dir_entries, payload_bytes)) =
        <[Cp1StreamDirEntryWire]>::ref_from_prefix_with_elems(after_hdr, count)
    else {
        return Err(ContainerWireError::SchemaStatic(
            "invalid CP1 shared stream header",
        ));
    };
    if count > out_slots.len() {
        return Err(ContainerWireError::SchemaStatic(
            "invalid CP1 shared stream header",
        ));
    }
    let mut expected_base = 0usize;
    let mut expected_ultra = 0usize;
    for entry in dir_entries {
        let lanes = (entry.raw_kl & 0x03) as usize;
        expected_base += cp1_contributor_record_bytes(lanes, false).ok_or(
            ContainerWireError::SchemaStatic("invalid contributor lane count"),
        )?;
        if has_ultra {
            expected_ultra += cp1_contributor_record_bytes(lanes, true).unwrap();
        }
    }
    if payload_bytes.len() < expected_base + expected_ultra {
        return Err(ContainerWireError::SchemaStatic(
            "truncated CP1 shared stream payload",
        ));
    }
    let base_bytes = &payload_bytes[..expected_base];
    let ultra_bytes = &payload_bytes[expected_base..expected_base + expected_ultra];
    let read_f32_at = |bytes: &[u8], idx: usize| -> Result<f32, ContainerWireError> {
        let off = idx * 4;
        let words = read_finite_f32_words_rt(&bytes[off..off + 4])?;
        Ok(words[0].get())
    };
    let mut b_off = 0usize;
    let mut u_off = 0usize;
    for (out_slot, dir_entry) in out_slots.iter_mut().take(count).zip(dir_entries.iter()) {
        let control_idx = dir_entry.control_idx.get();
        let raw_kl = dir_entry.raw_kl;
        let lanes = (raw_kl & 0x03) as usize;
        let ctrl_kind = (raw_kl >> 2) & 0x03;
        let aux = dir_entry.aux;
        let b_len = cp1_contributor_record_bytes(lanes, false).unwrap();
        let b_slice = &base_bytes[b_off..b_off + b_len];
        b_off += b_len;

        let mut slot = Cp1ContributorSlot::zero();
        slot.lanes = lanes as u8;
        slot.ctrl_kind = ctrl_kind;
        slot.control_idx = control_idx;
        slot.aux = aux;
        for k in 0..lanes {
            for r in 0..8 {
                slot.main_base[k][r] = read_f32_at(b_slice, k * 8 + r)?;
            }
            slot.emb[k] = read_f32_at(b_slice, 8 * lanes + k)?;
        }
        let v_start = 9 * lanes;
        for f in 0..4 {
            slot.v_base[f] = read_f32_at(b_slice, v_start + f)?;
        }
        if has_ultra {
            let u_len = cp1_contributor_record_bytes(lanes, true).unwrap();
            let u_slice = &ultra_bytes[u_off..u_off + u_len];
            u_off += u_len;
            for k in 0..lanes {
                for r in 0..8 {
                    slot.main_ultra[k][r] = read_f32_at(u_slice, k * 8 + r)?;
                }
            }
            let uv_start = 8 * lanes;
            for f in 0..4 {
                slot.v_ultra[f] = read_f32_at(u_slice, uv_start + f)?;
            }
            slot.has_ultra = true;
        }
        *out_slot = slot;
    }
    Ok(count)
}

/// Decodes a self-describing `CP1\0` contributor stream from `shared_extent` into `out_slots`
/// without heap allocation (`ref[container.dnam.cp1-profile]`).
///
/// Framing: `b"CP1\0"` (`4 B`), `count: u16le` (`2 B`), `flags: u8` (`1 B`, bit 0 = `has_ultra`),
/// `reserved: u8` (`1 B`), followed by `count * 4 B` contributor directory entries
/// `(control_idx: u16le, raw_kl: u8, aux: u8)`, followed by packed `base_bytes` and optional `ultra_bytes`.
pub fn decode_cp1_shared_stream_into_slice(
    cp1_slice: &[u8],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, DnamError> {
    decode_cp1_shared_stream_into_slice_rt(cp1_slice, out_slots)
        .map_err(ContainerWireError::into_dnam_error)
}

/// Encodes a slice of [`Cp1ContributorSlot`] entries into a self-describing
/// `CP1\0` stream suitable for Region 4 `shared_extent` (`ref[container.dnam.cp1-profile]`).
pub fn encode_cp1_shared_stream(slots: &[Cp1ContributorSlot]) -> Result<Vec<u8>, DnamError> {
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    reject_if(
        slots.len() > u16::MAX as usize,
        "CP1 contributor count exceeds u16::MAX",
    )?;
    let has_ultra = slots.iter().any(|s| s.has_ultra);
    let mut base_len = 0usize;
    let mut ultra_len = 0usize;
    for slot in slots {
        let lanes = slot.lanes as usize;
        base_len += cp1_contributor_record_bytes(lanes, false)
            .or_schema_err("invalid contributor lane count")?;
        if has_ultra {
            ultra_len += cp1_contributor_record_bytes(lanes, true).unwrap();
        }
    }
    let hdr = Cp1StreamHeaderWire {
        magic: CP1_MAGIC,
        count: U16::new(slots.len() as u16),
        flags: u8::from(has_ultra),
        reserved: 0,
    };
    let mut out = Vec::with_capacity(
        size_of::<Cp1StreamHeaderWire>()
            + slots.len() * size_of::<Cp1StreamDirEntryWire>()
            + base_len
            + ultra_len,
    );
    out.extend_from_slice(hdr.as_bytes());
    for slot in slots {
        let raw_kl = ((slot.ctrl_kind & 0x03) << 2) | (slot.lanes & 0x03);
        let entry = Cp1StreamDirEntryWire {
            control_idx: U16::new(slot.control_idx),
            raw_kl,
            aux: slot.aux,
        };
        out.extend_from_slice(entry.as_bytes());
    }
    let push_f32 = |buf: &mut Vec<u8>, val: f32| -> Result<(), DnamError> {
        reject_if(!val.is_finite(), "nonfinite coefficient")?;
        buf.extend_from_slice(F32::new(val).as_bytes());
        Ok(())
    };
    for slot in slots {
        let lanes = slot.lanes as usize;
        for k in 0..lanes {
            for r in 0..8 {
                push_f32(&mut out, slot.main_base[k][r])?;
            }
        }
        for k in 0..lanes {
            push_f32(&mut out, slot.emb[k])?;
        }
        for f in 0..4 {
            push_f32(&mut out, slot.v_base[f])?;
        }
    }
    if has_ultra {
        for slot in slots {
            let lanes = slot.lanes as usize;
            for k in 0..lanes {
                for r in 0..8 {
                    push_f32(&mut out, slot.main_ultra[k][r])?;
                }
            }
            for f in 0..4 {
                push_f32(&mut out, slot.v_ultra[f])?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use alloc::vec;
    use serde_json::json;

    /// verify[container.dnam.delay-histories]
    /// Verifies canonical decimal parsing (`u64` without leading zeros) and `delay_histories` validation.
    #[test]
    fn test_model_declared_delay_histories_validation() {
        assert_eq!(
            parse_canonical_reference_samples("432000").unwrap(),
            432_000
        );
        assert_eq!(
            parse_canonical_reference_samples("18446744073709551615").unwrap(),
            u64::MAX
        );
        for bad in ["", "0", "04", "+4", "4.0", "18446744073709551616"] {
            assert!(parse_canonical_reference_samples(bad).is_err());
        }

        let valid_entries = vec![
            json!({
                "id": "transport.quad0.line0",
                "owner": "transport",
                "kind": "fractional",
                "introduced_tier": "lite",
                "sample_format": "fp16",
                "reference_samples": {
                    "lite": "432000",
                    "full": "432000",
                    "pro": "432000",
                    "ultra": "432000"
                }
            }),
            json!({
                "id": "transport.ap.0",
                "owner": "transport",
                "kind": "allpass",
                "introduced_tier": "pro",
                "sample_format": "f32",
                "reference_samples": {
                    "pro": "149",
                    "ultra": "149"
                }
            }),
            json!({
                "id": "pre.left",
                "owner": "crossbar",
                "kind": "fractional",
                "introduced_tier": "lite",
                "sample_format": "fp16",
                "reference_samples": {
                    "lite": "480",
                    "full": "480",
                    "pro": "960",
                    "ultra": "960"
                }
            }),
        ];
        let parsed = validate_delay_histories(&valid_entries, 3).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].owner, "transport");
        assert_eq!(parsed[1].sample_format, "f32");
        assert_eq!(parsed[2].owner, "crossbar");
        assert_eq!(parsed[2].reference_samples_by_tier, [480, 480, 960, 960]);
        assert_eq!(parsed[1].reference_samples_by_tier, [0, 0, 149, 149]);

        // Unsupported sample_format ("f64") is rejected
        let bad_fmt = vec![json!({
            "id": "transport.bad_fmt",
            "owner": "transport",
            "kind": "fractional",
            "introduced_tier": "lite",
            "sample_format": "f64",
            "reference_samples": { "lite": "64" }
        })];
        assert!(validate_delay_histories(&bad_fmt, 0).is_err());

        // Legacy owner values / legacy "stage" key are rejected
        let bad_legacy_key = vec![json!({
            "id": "tank.legacy",
            "stage": "op_t_tank",
            "kind": "fractional",
            "introduced_tier": "lite",
            "sample_format": "fp16",
            "reference_samples": { "lite": "64" }
        })];
        assert!(validate_delay_histories(&bad_legacy_key, 0).is_err());

        for legacy_owner in ["op_t_tank", "spatial_tank"] {
            let bad_legacy_owner = vec![json!({
                "id": "tank.legacy_val",
                "owner": legacy_owner,
                "kind": "fractional",
                "introduced_tier": "lite",
                "sample_format": "fp16",
                "reference_samples": { "lite": "64" }
            })];
            assert!(validate_delay_histories(&bad_legacy_owner, 0).is_err());
        }

        // Fractional delay < 4 is rejected
        let too_short = vec![json!({
            "id": "pre.left",
            "owner": "crossbar",
            "kind": "fractional",
            "introduced_tier": "lite",
            "sample_format": "fp16",
            "reference_samples": { "lite": "3" }
        })];
        assert!(validate_delay_histories(&too_short, 0).is_err());
    }

    /// verify[container.dnam.load-validation]
    /// verify[container.dnam.cabinet-slices]
    /// verify[container.dnam.cp1-profile]
    /// Verifies `ModalConvolver` separate `64 B`-aligned slices and `CP1` packed control projection records.
    #[test]
    fn test_convolver_slices_and_cp1_packed_projection_validation() {
        // 1. ModalConvolver slices across Lite, Full, Pro, Ultra:
        //    FIR [256, 128, 384, 256] B, Kautz [2, 1, 3, 2] * 128 B = [256, 128, 384, 256] B
        for (tier, fir_len, blocks) in [
            (0u8, 256usize, 2usize),
            (1, 128, 1),
            (2, 384, 3),
            (3, 256, 2),
        ] {
            let kautz_len = blocks * 128;
            let arena_bytes = vec![0u8; fir_len + kautz_len + 64];
            validate_convolver_tier_slices(tier, &arena_bytes, 0, fir_len, fir_len, kautz_len)
                .unwrap();
            // Wrong FIR or Kautz length rejected
            assert!(
                validate_convolver_tier_slices(
                    tier,
                    &arena_bytes,
                    0,
                    fir_len / 2,
                    fir_len,
                    kautz_len
                )
                .is_err()
            );
            // Overlapping slices rejected
            assert!(
                validate_convolver_tier_slices(tier, &arena_bytes, 0, fir_len, 64, kautz_len)
                    .is_err()
            );
            // Unaligned offset rejected
            assert!(
                validate_convolver_tier_slices(
                    tier,
                    &arena_bytes,
                    32,
                    fir_len,
                    fir_len + 64,
                    kautz_len
                )
                .is_err()
            );
        }

        // 2. CP1 contributor sizes: 4*(9k+4) B in Arena 0, 4*(8k+4) B in Arena 3
        assert_eq!(cp1_contributor_record_bytes(1, false), Some(52));
        assert_eq!(cp1_contributor_record_bytes(2, false), Some(88));
        assert_eq!(cp1_contributor_record_bytes(3, false), Some(124));
        assert_eq!(cp1_contributor_record_bytes(1, true), Some(48));
        assert_eq!(cp1_contributor_record_bytes(2, true), Some(80));
        assert_eq!(cp1_contributor_record_bytes(3, true), Some(112));
        assert_eq!(cp1_contributor_record_bytes(0, false), None);
        assert_eq!(cp1_contributor_record_bytes(4, false), None);

        let lanes = [3usize, 1, 2];
        let base_total: usize = lanes
            .iter()
            .map(|&k| cp1_contributor_record_bytes(k, false).unwrap())
            .sum();
        let ultra_total: usize = lanes
            .iter()
            .map(|&k| cp1_contributor_record_bytes(k, true).unwrap())
            .sum();
        let base_bytes = vec![0u8; base_total];
        let ultra_bytes = vec![0u8; ultra_total];
        let decoded =
            decode_cp1_contributor_stream(&base_bytes, Some(&ultra_bytes), &lanes).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].lanes, 3);
        assert_eq!(decoded[0].base_main.len(), 24);
        assert_eq!(decoded[0].embedding.len(), 3);
        assert!(decoded[0].extra_factors.is_some());
        let decoded_no_ultra = decode_cp1_contributor_stream(&base_bytes, None, &lanes).unwrap();
        assert!(decoded_no_ultra[0].extra_factors.is_none());

        let mut slots = [Cp1ContributorSlot::zero(); 4];
        let n_slots = decode_cp1_contributor_stream_into_slice(
            &base_bytes,
            Some(&ultra_bytes),
            &lanes,
            &mut slots,
        )
        .unwrap();
        assert_eq!(n_slots, 3);
        assert!(slots[0].has_ultra);

        let mut shared_stream = Vec::new();
        let hdr = Cp1StreamHeaderWire {
            magic: CP1_MAGIC,
            count: U16::new(2),
            flags: 1, // has_ultra = true
            reserved: 0,
        };
        let e0 = Cp1StreamDirEntryWire {
            control_idx: U16::new(0),
            raw_kl: 3,
            aux: 0xFF,
        };
        let e1 = Cp1StreamDirEntryWire {
            control_idx: U16::new(1),
            raw_kl: (2 << 2) | 1,
            aux: 2,
        };
        shared_stream.extend_from_slice(hdr.as_bytes());
        shared_stream.extend_from_slice(e0.as_bytes());
        shared_stream.extend_from_slice(e1.as_bytes());
        shared_stream.resize(8 + 8 + (124 + 52) + (112 + 48), 0);
        let n_shared = decode_cp1_shared_stream_into_slice(&shared_stream, &mut slots).unwrap();
        assert_eq!(n_shared, 2);
        assert_eq!(slots[1].aux, 2);
        assert_eq!(
            decode_cp1_shared_stream_into_slice(&[], &mut slots).unwrap(),
            0
        );

        assert!(encode_cp1_shared_stream(&[]).unwrap().is_empty());
        let mut s0 = Cp1ContributorSlot::zero();
        s0.lanes = 2;
        s0.ctrl_kind = 0;
        s0.control_idx = 5;
        s0.aux = 0xFF;
        s0.main_base[0][0] = 0.75;
        s0.emb[1] = -0.25;
        s0.v_base[0] = 0.5;
        s0.main_ultra[1][3] = 0.125;
        s0.v_ultra[2] = -0.5;
        s0.has_ultra = true;
        let mut s1 = Cp1ContributorSlot::zero();
        s1.lanes = 1;
        s1.ctrl_kind = 2;
        s1.control_idx = 6;
        s1.aux = 7;
        s1.has_ultra = true;
        let encoded = encode_cp1_shared_stream(&[s0, s1]).unwrap();
        let mut rt_slots = [Cp1ContributorSlot::zero(); 4];
        let rt_n = decode_cp1_shared_stream_into_slice(&encoded, &mut rt_slots).unwrap();
        assert_eq!(rt_n, 2);
        assert_eq!(rt_slots[0].control_idx, 5);
        assert_eq!(rt_slots[0].main_base[0][0], 0.75);
        assert_eq!(rt_slots[0].emb[1], -0.25);
        assert_eq!(rt_slots[0].main_ultra[1][3], 0.125);
        assert_eq!(rt_slots[1].ctrl_kind, 2);
        assert_eq!(rt_slots[1].aux, 7);

        let mut bad_slot = s0;
        bad_slot.v_base[0] = f32::NAN;
        assert!(encode_cp1_shared_stream(&[bad_slot]).is_err());
    }
}
