//! `CP1` contributor decoder (`cp1_dir`/`cp1_base`/`cp1_ultra` and `"CP1\0"` overflow stream)
//! and `U_int` matrix evaluation (`ref[container.dnam.cp1-profile]`, `ref[dsp.ctrl.accum.anova-fm]`).

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::F32;

use crate::bin::{
    Arena0ControlBusWire, Arena3ControlBusWire, CP1_STREAM_FLAG_HAS_ULTRA, CP1_STREAM_MAGIC,
    Cp1StreamDirEntryWire, Cp1StreamHeaderWire, MAX_NUM_CONTROLS,
};
use crate::types::VmError;

/// Decoded `CP1` control contributor (`Range`, `Switch`, or `BilinearCrossTerm`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cp1ContributorSlot {
    /// Target physical control index (`< num_controls`).
    pub control_idx: u16,
    /// Contributor kind (`0 = Range`, `1 = Switch`, `2 = BilinearCrossTerm`).
    pub ctrl_kind: u8,
    /// Active basis lanes `k` (`1..=3`).
    pub lanes: u8,
    /// Option index (`Switch`), partner control index (`BilinearCrossTerm`), or `0xFF`.
    pub aux: u8,
    /// Whether this slot carries an active Ultra delta slice.
    pub has_ultra: bool,
    /// Base `W_main[3][8]` projection (`row-major [lane][latent_dim]`).
    pub main_base: [[f32; 8]; 3],
    /// Embedded latent weight `emb[3]` for FM contributor `z_c`.
    pub emb: [f32; 3],
    /// Base rank-4 FM interaction vector `v_base[4]`.
    pub v_base: [f32; 4],
    /// Ultra `W_main,Delta[3][8]` projection (`row-major [lane][latent_dim]`).
    pub main_ultra: [[f32; 8]; 3],
    /// Ultra rank-4 FM delta vector `v_ultra[4]`.
    pub v_ultra: [f32; 4],
}

impl Cp1ContributorSlot {
    /// Returns a zeroed contributor slot.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            control_idx: 0,
            ctrl_kind: 0,
            lanes: 0,
            aux: 0xFF,
            has_ultra: false,
            main_base: [[0.0; 8]; 3],
            emb: [0.0; 3],
            v_base: [0.0; 4],
            main_ultra: [[0.0; 8]; 3],
            v_ultra: [0.0; 4],
        }
    }
}

fn read_base_slice(slot: &mut Cp1ContributorSlot, bytes: &[u8]) -> Result<(), VmError> {
    let lanes = usize::from(slot.lanes);
    let words = <[F32]>::ref_from_bytes(bytes).map_err(|_| VmError::InvalidCp1Stream)?;
    if words.len() != 9 * lanes + 4 {
        return Err(VmError::InvalidCp1Stream);
    }
    let mut idx = 0usize;
    for l in 0..lanes {
        for d in 0..8 {
            let val = words[idx].get();
            if !val.is_finite() {
                return Err(VmError::InvalidCp1Stream);
            }
            slot.main_base[l][d] = val;
            idx += 1;
        }
    }
    for l in 0..lanes {
        let val = words[idx].get();
        if !val.is_finite() {
            return Err(VmError::InvalidCp1Stream);
        }
        slot.emb[l] = val;
        idx += 1;
    }
    for f in 0..4 {
        let val = words[idx].get();
        if !val.is_finite() {
            return Err(VmError::InvalidCp1Stream);
        }
        slot.v_base[f] = val;
        idx += 1;
    }
    Ok(())
}

fn read_ultra_slice(slot: &mut Cp1ContributorSlot, bytes: &[u8]) -> Result<(), VmError> {
    let lanes = usize::from(slot.lanes);
    let words = <[F32]>::ref_from_bytes(bytes).map_err(|_| VmError::InvalidCp1Stream)?;
    if words.len() != 8 * lanes + 4 {
        return Err(VmError::InvalidCp1Stream);
    }
    let mut idx = 0usize;
    for l in 0..lanes {
        for d in 0..8 {
            let val = words[idx].get();
            if !val.is_finite() {
                return Err(VmError::InvalidCp1Stream);
            }
            slot.main_ultra[l][d] = val;
            idx += 1;
        }
    }
    for f in 0..4 {
        let val = words[idx].get();
        if !val.is_finite() {
            return Err(VmError::InvalidCp1Stream);
        }
        slot.v_ultra[f] = val;
        idx += 1;
    }
    slot.has_ultra = true;
    Ok(())
}

/// Decodes `CP1` contributors into `out_slots` (`#[forbid(unsafe_code)]`, zero allocation).
///
/// Whenever `cp1_overflow` is non-empty, `"CP1\0"` takes canonical precedence over Arena 0
/// inline `cp1_dir`; Ultra delta slices are populated iff `arena3_cb.is_some()`.
pub fn decode_cp1_contributors(
    arena0_cb: &Arena0ControlBusWire,
    arena3_cb: Option<&Arena3ControlBusWire>,
    cp1_overflow: &[u8],
    num_controls: usize,
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, VmError> {
    if !cp1_overflow.is_empty() {
        let (hdr, rem) = Cp1StreamHeaderWire::read_from_prefix(cp1_overflow)
            .map_err(|_| VmError::InvalidCp1Stream)?;
        if hdr.magic != CP1_STREAM_MAGIC
            || (hdr.flags & !CP1_STREAM_FLAG_HAS_ULTRA) != 0
            || hdr.reserved != 0
        {
            return Err(VmError::InvalidCp1Stream);
        }
        let count = hdr.count.get() as usize;
        if count > crate::control::MAX_EXACT_CONTRIBUTORS {
            return Err(VmError::InvalidCp1Stream);
        }
        if out_slots.len() < count {
            return Err(VmError::InsufficientBackingCapacity);
        }
        let dir_len = count.checked_mul(4).ok_or(VmError::InvalidCp1Stream)?;
        let (dir_bytes, payload) = rem
            .split_at_checked(dir_len)
            .ok_or(VmError::InvalidCp1Stream)?;
        let entries = <[Cp1StreamDirEntryWire]>::ref_from_bytes(dir_bytes)
            .map_err(|_| VmError::InvalidCp1Stream)?;
        let mut cursor = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            if (entry.raw_kl & 0xF0) != 0 {
                return Err(VmError::InvalidCp1Stream);
            }
            let control_idx = entry.control_idx.get();
            let ctrl_kind = (entry.raw_kl >> 2) & 0b11;
            let lanes = entry.raw_kl & 0b11;
            if usize::from(control_idx) >= num_controls
                || ctrl_kind > 2
                || !(1..=3).contains(&lanes)
                || (ctrl_kind == 1 && lanes != 1)
            {
                return Err(VmError::InvalidCp1Stream);
            }
            if ctrl_kind == 2 && usize::from(entry.aux) >= num_controls {
                return Err(VmError::InvalidCp1Stream);
            }
            let slice_bytes = 4 * (9 * usize::from(lanes) + 4);
            let next_cursor = cursor
                .checked_add(slice_bytes)
                .ok_or(VmError::InvalidCp1Stream)?;
            let slice = payload
                .get(cursor..next_cursor)
                .ok_or(VmError::InvalidCp1Stream)?;
            let mut slot = Cp1ContributorSlot::zero();
            slot.control_idx = control_idx;
            slot.ctrl_kind = ctrl_kind;
            slot.lanes = lanes;
            slot.aux = entry.aux;
            read_base_slice(&mut slot, slice)?;
            cursor = next_cursor;
            out_slots[i] = slot;
        }
        let stream_has_ultra = (hdr.flags & CP1_STREAM_FLAG_HAS_ULTRA) != 0;
        if stream_has_ultra {
            for slot in &mut out_slots[..count] {
                let slice_bytes = 4 * (8 * usize::from(slot.lanes) + 4);
                let next_cursor = cursor
                    .checked_add(slice_bytes)
                    .ok_or(VmError::InvalidCp1Stream)?;
                let slice = payload
                    .get(cursor..next_cursor)
                    .ok_or(VmError::InvalidCp1Stream)?;
                if arena3_cb.is_some() {
                    read_ultra_slice(slot, slice)?;
                }
                cursor = next_cursor;
            }
        }
        if cursor != payload.len() {
            return Err(VmError::InvalidCp1Stream);
        }
        return Ok(count);
    }

    let count = arena0_cb.cp1_dir.cp1_dir_count.get() as usize;
    if count > 8 || arena0_cb.cp1_dir.flags.get() != 0 {
        return Err(VmError::InvalidCp1Stream);
    }
    if out_slots.len() < count {
        return Err(VmError::InsufficientBackingCapacity);
    }
    let mut base_cursor = 0usize;
    let mut ultra_cursor = 0usize;
    for i in 0..count {
        let [c_idx, kl, aux] = arena0_cb.cp1_dir.entries[i];
        let ctrl_kind = (kl >> 4) & 0x0F;
        let lanes = kl & 0x0F;
        if usize::from(c_idx) >= num_controls
            || ctrl_kind > 2
            || !(1..=3).contains(&lanes)
            || (ctrl_kind == 1 && lanes != 1)
        {
            return Err(VmError::InvalidCp1Stream);
        }
        if ctrl_kind == 2 && usize::from(aux) >= num_controls {
            return Err(VmError::InvalidCp1Stream);
        }
        let base_bytes = 4 * (9 * usize::from(lanes) + 4);
        if base_cursor + base_bytes > arena0_cb.cp1_base.len() {
            return Err(VmError::InvalidCp1Stream);
        }
        let mut slot = Cp1ContributorSlot::zero();
        slot.control_idx = u16::from(c_idx);
        slot.ctrl_kind = ctrl_kind;
        slot.lanes = lanes;
        slot.aux = aux;
        read_base_slice(
            &mut slot,
            &arena0_cb.cp1_base[base_cursor..base_cursor + base_bytes],
        )?;
        base_cursor += base_bytes;

        if let Some(a3) = arena3_cb {
            let ultra_bytes = 4 * (8 * usize::from(lanes) + 4);
            if ultra_cursor + ultra_bytes > a3.cp1_ultra.len() {
                return Err(VmError::InvalidCp1Stream);
            }
            read_ultra_slice(
                &mut slot,
                &a3.cp1_ultra[ultra_cursor..ultra_cursor + ultra_bytes],
            )?;
            ultra_cursor += ultra_bytes;
        }
        out_slots[i] = slot;
    }
    Ok(count)
}

/// Counts distinct physical controls represented in `slots` without allocation.
#[inline]
pub fn count_distinct_contributor_controls(slots: &[Cp1ContributorSlot]) -> usize {
    let mut seen = [0_u64; 4];
    for slot in slots {
        let idx = usize::from(slot.control_idx);
        if idx < MAX_NUM_CONTROLS {
            seen[idx >> 6] |= 1_u64 << (idx & 63);
        }
    }
    seen.iter().map(|w| w.count_ones() as usize).sum()
}

/// Evaluates $(U_{\text{int}}^{\text{base}})^\top \mathbf{z}_{\text{fm}}^{\text{base}}$ into
/// 8 base latent coordinates in sequential `binary32` (`FP_CONTRACT OFF`).
#[inline]
pub fn eval_uint_base_4x8(u_int_base: &[[f32; 8]; 4], z_fm_base: &[f32; 4]) -> [f32; 8] {
    let mut out = [0.0_f32; 8];
    for d in 0..8 {
        let mut acc = 0.0_f32;
        for f in 0..4 {
            acc += u_int_base[f][d] * z_fm_base[f];
        }
        out[d] = acc;
    }
    out
}

/// Evaluates the 3 Ultra $4 \times 8$ FM interaction blocks (`[0]` onto base `0..8` from `z_fm_ultra`,
/// `[1]` onto ultra `8..16` from `z_fm_base`, `[2]` onto ultra `8..16` from `z_fm_ultra`).
#[inline]
pub fn eval_uint_ultra_delta_3x4x8(
    u_int_ultra: &[[[f32; 8]; 4]; 3],
    z_fm_base: &[f32; 4],
    z_fm_ultra: &[f32; 4],
) -> ([f32; 8], [f32; 8]) {
    let mut delta_base = [0.0_f32; 8];
    let mut ultra_8 = [0.0_f32; 8];
    for d in 0..8 {
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut acc2 = 0.0_f32;
        for f in 0..4 {
            acc0 += u_int_ultra[0][f][d] * z_fm_ultra[f];
            acc1 += u_int_ultra[1][f][d] * z_fm_base[f];
            acc2 += u_int_ultra[2][f][d] * z_fm_ultra[f];
        }
        delta_base[d] = acc0;
        ultra_8[d] = acc1 + acc2;
    }
    (delta_base, ultra_8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromZeros, IntoBytes};

    /// verify[dsp.ctrl.accum.cp1-scaling]
    /// verify[dsp.ctrl.ledger.memory-compute]
    /// verify[dsp.ctrl.switch.option-contributors]
    /// verify[dsp.op-n.ctrl.incremental-update]
    #[test]
    fn test_inline_cp1_decode_and_distinct_count() {
        let mut a0_cb = Arena0ControlBusWire::new_zeroed();
        a0_cb.cp1_dir.cp1_dir_count = zerocopy::byteorder::little_endian::U16::new(2);
        a0_cb.cp1_dir.entries[0] = [0, 1, 0xFF];
        a0_cb.cp1_dir.entries[1] = [1, 1, 0xFF];
        let mut slice0 = [F32::new(0.0); 13];
        slice0[0] = F32::new(0.5);
        slice0[8] = F32::new(1.0);
        slice0[9] = F32::new(0.25);
        a0_cb.cp1_base[..52].copy_from_slice(slice0.as_bytes());
        a0_cb.cp1_base[52..104].copy_from_slice(slice0.as_bytes());

        let mut out = [Cp1ContributorSlot::zero(); 4];
        let n = decode_cp1_contributors(&a0_cb, None, &[], 2, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out[0].main_base[0][0], 0.5);
        assert_eq!(out[0].v_base[0], 0.25);
        assert_eq!(count_distinct_contributor_controls(&out[..n]), 2);

        // Test "CP1\0" overflow with CP1_STREAM_FLAG_HAS_ULTRA, cross-term + switch, and uint_ultra.
        let a3_cb = Arena3ControlBusWire::new_zeroed();
        let mut stream = [0u8; 8 + 4 + (9 + 4) * 4 + (8 + 4) * 4];
        let hdr = Cp1StreamHeaderWire {
            magic: CP1_STREAM_MAGIC,
            count: zerocopy::byteorder::little_endian::U16::new(1),
            flags: CP1_STREAM_FLAG_HAS_ULTRA,
            reserved: 0,
        };
        hdr.write_to_prefix(&mut stream[0..8]).unwrap();
        let ent = Cp1StreamDirEntryWire {
            control_idx: zerocopy::byteorder::little_endian::U16::new(0),
            raw_kl: (2 << 2) | 1,
            aux: 1,
        };
        ent.write_to_prefix(&mut stream[8..12]).unwrap();
        let mut base_words = [F32::new(0.0); 13];
        base_words[0] = F32::new(0.75);
        base_words[9] = F32::new(0.5);
        stream[12..64].copy_from_slice(base_words.as_bytes());
        let mut ultra_words = [F32::new(0.0); 12];
        ultra_words[0] = F32::new(-0.25);
        ultra_words[8] = F32::new(0.125);
        stream[64..112].copy_from_slice(ultra_words.as_bytes());

        let m = decode_cp1_contributors(&a0_cb, Some(&a3_cb), &stream, 2, &mut out).unwrap();
        assert_eq!(m, 1);
        assert!(out[0].has_ultra);
        assert_eq!(out[0].main_ultra[0][0], -0.25);
        assert_eq!(out[0].v_ultra[0], 0.125);

        let mut u_int_ultra = [[[0.0_f32; 8]; 4]; 3];
        u_int_ultra[0][0][0] = 2.0;
        u_int_ultra[1][0][1] = 3.0;
        u_int_ultra[2][0][1] = 4.0;
        let (db, du) =
            eval_uint_ultra_delta_3x4x8(&u_int_ultra, &[1.0, 0.0, 0.0, 0.0], &[0.5, 0.0, 0.0, 0.0]);
        assert_eq!(db[0], 1.0);
        assert_eq!(du[1], 5.0);

        assert_eq!(
            decode_cp1_contributors(&a0_cb, Some(&a3_cb), &stream, 2, &mut []).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
    }
}
