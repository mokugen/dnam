//! `zerocopy` wire serialization for inline/`"CP1\0"` contributor streams and validated `"TSYN"`
//! tempo-sync rows (`ref[container.dnam.cp1-profile]`, `ref[dsp.tempo.physical-inversion]`).

use dnam_vm::bin::{
    Arena0ControlBusWire, Arena3ControlBusWire, CP1_STREAM_FLAG_HAS_ULTRA, CP1_STREAM_MAGIC,
    Cp1InlineDirWire, Cp1StreamDirEntryWire, Cp1StreamHeaderWire, MAX_NUM_CONTROLS,
    TSYN_FLAG_CLAMP_ONLY, TempoSyncEntryWire,
};
use dnam_vm::control::MAX_EXACT_CONTRIBUTORS;
use dnam_vm::control::cp1::Cp1ContributorSlot;
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromZeros, IntoBytes};

use crate::error::TrainError;

const INLINE_MAX_SLOTS: usize = 8;
const INLINE_BASE_BYTES: usize = 576;
const INLINE_ULTRA_BYTES: usize = 2_112;

/// Canonical quarter-note (`1/4`) tempo-sync subdivision index (`ref[dsp.tempo.physical-inversion]`).
pub const TSYN_SUBDIV_QUARTER: u8 = 4;

fn serialize_cp1_slot_base(slot: &Cp1ContributorSlot, out: &mut Vec<F32>) {
    let lanes = usize::from(slot.lanes).min(3);
    // Layout: `main_base[0..k][0..8]` contiguous, followed by `emb[0..k]`, followed by `v_base[0..4]`.
    for l in 0..lanes {
        for d in 0..8 {
            out.push(F32::new(slot.main_base[l][d]));
        }
    }
    for l in 0..lanes {
        out.push(F32::new(slot.emb[l]));
    }
    for f in 0..4 {
        out.push(F32::new(slot.v_base[f]));
    }
}

fn serialize_cp1_slot_ultra(slot: &Cp1ContributorSlot, out: &mut Vec<F32>) {
    let lanes = usize::from(slot.lanes).min(3);
    for l in 0..lanes {
        for d in 0..8 {
            out.push(F32::new(slot.main_ultra[l][d]));
        }
    }
    for f in 0..4 {
        out.push(F32::new(slot.v_ultra[f]));
    }
}

/// Packs `slots` either inline inside [`Arena0ControlBusWire`]/[`Arena3ControlBusWire`] (`<= 8`
/// slots within `576 B` / `2,112 B`) or into an exact zero-padded-free `"CP1\0"` overflow stream.
pub fn pack_cp1_slots_into_wire(
    slots: &[Cp1ContributorSlot],
    num_controls: usize,
    has_ultra: bool,
    a0_cb: &mut Arena0ControlBusWire,
    a3_cb: &mut Arena3ControlBusWire,
) -> Result<Vec<u8>, TrainError> {
    a0_cb.cp1_dir = Cp1InlineDirWire::new_zeroed();
    a0_cb.cp1_base.fill(0);
    a3_cb.cp1_ultra.fill(0);

    if num_controls > MAX_NUM_CONTROLS || slots.len() > MAX_EXACT_CONTRIBUTORS {
        return Err(TrainError::InvalidControlSpec {
            context: "pack_cp1_slots_into_wire",
            detail: "CP1 control or contributor count exceeds VM wire ceiling",
        });
    }

    for s in slots {
        let lanes = usize::from(s.lanes);
        if usize::from(s.control_idx) >= num_controls
            || s.ctrl_kind > 2
            || !(1..=3).contains(&s.lanes)
            || (s.ctrl_kind == 1 && s.lanes != 1)
            || (s.ctrl_kind == 2 && usize::from(s.aux) >= num_controls)
        {
            return Err(TrainError::InvalidControlSpec {
                context: "pack_cp1_slots_into_wire",
                detail: "CP1 contributor slot violates VM wire invariants",
            });
        }
        let weights_ok = s.main_base[..lanes].iter().flatten().all(|v| v.is_finite())
            && s.emb[..lanes].iter().all(|v| v.is_finite())
            && s.v_base.iter().all(|v| v.is_finite())
            && (!has_ultra
                || (s.main_ultra[..lanes]
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite())
                    && s.v_ultra.iter().all(|v| v.is_finite())));
        if !weights_ok {
            return Err(TrainError::NonFiniteInput {
                context: "pack_cp1_slots_into_wire",
            });
        }
    }

    let base_bytes: usize = slots
        .iter()
        .map(|s| 4 * (9 * usize::from(s.lanes) + 4))
        .sum();
    let ultra_bytes: usize = if has_ultra {
        slots
            .iter()
            .map(|s| 4 * (8 * usize::from(s.lanes) + 4))
            .sum()
    } else {
        0
    };

    if slots.len() <= INLINE_MAX_SLOTS
        && base_bytes <= INLINE_BASE_BYTES
        && ultra_bytes <= INLINE_ULTRA_BYTES
    {
        a0_cb.cp1_dir.cp1_dir_count = U16::new(slots.len() as u16);
        a0_cb.cp1_dir.flags = U16::new(0);
        let mut base_off = 0_usize;
        let mut ultra_off = 0_usize;
        for (i, s) in slots.iter().enumerate() {
            a0_cb.cp1_dir.entries[i] = [
                s.control_idx as u8,
                ((s.ctrl_kind & 0x0F) << 4) | (s.lanes & 0x0F),
                s.aux,
            ];
            let mut bw = Vec::with_capacity(9 * usize::from(s.lanes) + 4);
            serialize_cp1_slot_base(s, &mut bw);
            let bslice = bw.as_bytes();
            a0_cb.cp1_base[base_off..base_off + bslice.len()].copy_from_slice(bslice);
            base_off += bslice.len();
            if has_ultra {
                let mut uw = Vec::with_capacity(8 * usize::from(s.lanes) + 4);
                serialize_cp1_slot_ultra(s, &mut uw);
                let uslice = uw.as_bytes();
                a3_cb.cp1_ultra[ultra_off..ultra_off + uslice.len()].copy_from_slice(uslice);
                ultra_off += uslice.len();
            }
        }
        return Ok(Vec::new());
    }

    let dir_bytes = slots.len() * 4;
    let total_bytes = 8 + dir_bytes + base_bytes + ultra_bytes;
    let mut stream = Vec::with_capacity(total_bytes);
    let hdr = Cp1StreamHeaderWire {
        magic: CP1_STREAM_MAGIC,
        count: U16::new(slots.len() as u16),
        flags: if has_ultra {
            CP1_STREAM_FLAG_HAS_ULTRA
        } else {
            0
        },
        reserved: 0,
    };
    stream.extend_from_slice(hdr.as_bytes());
    for s in slots {
        let entry = Cp1StreamDirEntryWire {
            control_idx: U16::new(s.control_idx),
            raw_kl: ((s.ctrl_kind & 0b11) << 2) | (s.lanes & 0b11),
            aux: s.aux,
        };
        stream.extend_from_slice(entry.as_bytes());
    }
    let mut bw = Vec::new();
    for s in slots {
        bw.clear();
        serialize_cp1_slot_base(s, &mut bw);
        stream.extend_from_slice(bw.as_bytes());
    }
    if has_ultra {
        let mut uw = Vec::new();
        for s in slots {
            uw.clear();
            serialize_cp1_slot_ultra(s, &mut uw);
            stream.extend_from_slice(uw.as_bytes());
        }
    }
    Ok(stream)
}

/// Builds and validates a canonical `40 B` [`TempoSyncEntryWire`] row with optional switch condition
/// and caller-selected subdivision (`default_subdiv < 12`, e.g. `TSYN_SUBDIV_QUARTER = 4`).
pub fn build_tsyn_entry_wire_with_cond(
    ctrl_idx: u16,
    is_freq_hz: bool,
    subdiv_cond: (u8, Option<(u16, u8)>),
    rates: (f32, f32),
    lobatto_anchors: [f32; 3],
    declared_bounds: (f32, f32),
    clamp_only: bool,
) -> Result<TempoSyncEntryWire, TrainError> {
    let (default_subdiv, switch_cond) = subdiv_cond;
    let (div_ratio, bpm_factor) = rates;
    if default_subdiv >= 12
        || !div_ratio.is_finite()
        || div_ratio <= 0.0
        || !bpm_factor.is_finite()
        || bpm_factor <= 0.0
        || !lobatto_anchors.iter().all(|v| v.is_finite())
        || !declared_bounds.0.is_finite()
        || !declared_bounds.1.is_finite()
        || declared_bounds.0 > declared_bounds.1
    {
        return Err(TrainError::InvalidControlSpec {
            context: "build_tsyn_entry_wire_with_cond",
            detail: "invalid TSYN subdivision, bounds, rate factors, or anchors",
        });
    }
    let (switch_ctrl_idx, switch_opt_idx, has_switch_cond) = match switch_cond {
        Some((sw_c, sw_opt)) => (U16::new(sw_c), sw_opt.min(15), 1_u8),
        None => (U16::new(0), 0_u8, 0_u8),
    };
    Ok(TempoSyncEntryWire {
        ctrl_idx: U16::new(ctrl_idx),
        switch_ctrl_idx,
        switch_opt_idx,
        has_switch_cond,
        mode: u8::from(is_freq_hz),
        default_subdiv,
        flags: U32::new(u32::from(clamp_only) * TSYN_FLAG_CLAMP_ONLY),
        div_ratio: F32::new(div_ratio),
        bpm_factor: F32::new(bpm_factor),
        lobatto_anchors: lobatto_anchors.map(F32::new),
        declared_min: F32::new(declared_bounds.0),
        declared_max: F32::new(declared_bounds.1),
    })
}

/// Builds a canonical `40 B` quarter-note (`1/4`) unconditional [`TempoSyncEntryWire`] row.
pub fn build_tsyn_entry_wire(
    ctrl_idx: u16,
    is_freq_hz: bool,
    div_ratio: f32,
    bpm_factor: f32,
    lobatto_anchors: [f32; 3],
    declared_bounds: (f32, f32),
    clamp_only: bool,
) -> Result<TempoSyncEntryWire, TrainError> {
    build_tsyn_entry_wire_with_cond(
        ctrl_idx,
        is_freq_hz,
        (TSYN_SUBDIV_QUARTER, None),
        (div_ratio, bpm_factor),
        lobatto_anchors,
        declared_bounds,
        clamp_only,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::control::cp1::decode_cp1_contributors;

    #[test]
    fn test_cp1_multi_lane_inline_and_overflow_exact_roundtrip_and_tsyn_validation() {
        let mut slots = Vec::new();
        for (c, lanes) in [(0_u16, 1_u8), (1, 2), (2, 3)] {
            let mut s = Cp1ContributorSlot::zero();
            s.control_idx = c;
            s.ctrl_kind = 0;
            s.lanes = lanes;
            s.aux = 0xFF;
            s.has_ultra = true;
            for l in 0..usize::from(lanes) {
                for d in 0..8 {
                    s.main_base[l][d] =
                        ((c as f32) + 1.0) * 0.1 + (l as f32) * 0.02 + (d as f32) * 0.003;
                    s.main_ultra[l][d] = -0.5 * s.main_base[l][d];
                }
                s.emb[l] = 0.25 * ((l + 1) as f32);
            }
            s.v_base = [0.1, -0.2, 0.3, -0.4];
            s.v_ultra = [-0.15, 0.25, -0.35, 0.45];
            slots.push(s);
        }

        // Inline path (3 slots with 1, 2, 3 lanes).
        let mut a0_cb = Arena0ControlBusWire::new_zeroed();
        let mut a3_cb = Arena3ControlBusWire::new_zeroed();
        let ov_inline = pack_cp1_slots_into_wire(&slots, 3, true, &mut a0_cb, &mut a3_cb).unwrap();
        assert!(ov_inline.is_empty());
        let mut decoded = [Cp1ContributorSlot::zero(); 12];
        let n = decode_cp1_contributors(&a0_cb, Some(&a3_cb), &ov_inline, 3, &mut decoded).unwrap();
        assert_eq!(n, slots.len());
        for (expected, actual) in slots.iter().zip(&decoded[..n]) {
            assert_eq!(actual.control_idx, expected.control_idx);
            assert_eq!(actual.lanes, expected.lanes);
            assert_eq!(actual.main_base, expected.main_base);
            assert_eq!(actual.emb, expected.emb);
            assert_eq!(actual.v_base, expected.v_base);
            assert_eq!(actual.main_ultra, expected.main_ultra);
            assert_eq!(actual.v_ultra, expected.v_ultra);
        }

        // Force overflow `"CP1\0"` path (`9` slots > `8` inline cap).
        let mut slots9 = Vec::new();
        for c in 0..9_u16 {
            let mut s = slots[(c as usize) % 3];
            s.control_idx = c;
            slots9.push(s);
        }
        let ov_stream = pack_cp1_slots_into_wire(&slots9, 9, true, &mut a0_cb, &mut a3_cb).unwrap();
        assert!(ov_stream.starts_with(&CP1_STREAM_MAGIC));
        let n9 =
            decode_cp1_contributors(&a0_cb, Some(&a3_cb), &ov_stream, 9, &mut decoded).unwrap();
        assert_eq!(n9, 9);
        for (expected, actual) in slots9.iter().zip(&decoded[..n9]) {
            assert_eq!(actual.main_base, expected.main_base);
            assert_eq!(actual.emb, expected.emb);
            assert_eq!(actual.v_base, expected.v_base);
            assert_eq!(actual.main_ultra, expected.main_ultra);
            assert_eq!(actual.v_ultra, expected.v_ultra);
        }

        let row = build_tsyn_entry_wire_with_cond(
            0,
            true,
            (TSYN_SUBDIV_QUARTER, Some((1, 2))),
            (1.0, 1.0),
            [-1.0, 0.0, 1.0],
            (0.1, 10.0),
            false,
        )
        .unwrap();
        assert_eq!(row.default_subdiv, TSYN_SUBDIV_QUARTER);
        assert_eq!(row.has_switch_cond, 1);
        assert_eq!(row.switch_opt_idx, 2);
        assert!(matches!(
            build_tsyn_entry_wire(0, true, -1.0, 1.0, [-1.0, 0.0, 1.0], (0.0, 1.0), false),
            Err(TrainError::InvalidControlSpec { .. })
        ));
        assert!(matches!(
            pack_cp1_slots_into_wire(&slots, MAX_NUM_CONTROLS + 1, false, &mut a0_cb, &mut a3_cb),
            Err(TrainError::InvalidControlSpec { .. })
        ));
        let mut nan_slot = slots[0];
        nan_slot.main_base[0][0] = f32::NAN;
        assert!(matches!(
            pack_cp1_slots_into_wire(&[nan_slot], 3, false, &mut a0_cb, &mut a3_cb),
            Err(TrainError::NonFiniteInput { .. })
        ));
    }
}
