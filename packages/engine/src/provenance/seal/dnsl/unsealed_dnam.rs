//! Helpers for validating `DNPL` `dbin_prefix` extents and assembling unsealed `v3.0` Open `.dnam`
//! containers (`ref[sec.drm.protected-inner-framing]`, `ref[container.dnam.open-layout]`).

use alloc::vec::Vec;

use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::{U16, U32, U64};

use super::{
    UnsealedDnslInner, align64_usize, extract_mutable_json_bytes, merge_mutable_fields, pad_to_64,
};
use crate::container::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DEFAULT_DMU1_TRAILER_CAPACITY, DNAM_MAGIC,
    DcrcAuthRecord, DmueFooterWire, DnamDirectoryWire, DnamPreambleWire, FLAG_HAS_SIGNATURES,
    FLAG_STEREO_OUT, align64_u64, build_dmu1_mutable_frame, crc32c_update,
};
use crate::schema::{DnamError, DnamJsonHeader, reject_if};

pub(crate) fn extract_verified_logical_dbin_prefix(
    logical_prefix: &[u8],
    dir: &DnamDirectoryWire,
    l_hdr_len: usize,
    shared_bytes: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let hdr_end = 192usize
        .checked_add(l_hdr_len)
        .ok_or(DnamError::CryptoVerificationFailed(
            "logical header end overflow",
        ))?;
    let hdr_aligned = align64_usize(hdr_end);
    let pad =
        logical_prefix
            .get(hdr_end..hdr_aligned)
            .ok_or(DnamError::CryptoVerificationFailed(
                "truncated logical JSON header padding",
            ))?;
    reject_if(
        pad.iter().any(|&b| b != 0),
        "non-zero logical JSON header alignment padding in DNPL",
    )?;
    let l_dbin_len = usize::try_from(dir.dbin_prefix_bytes.get())
        .map_err(|_| DnamError::CryptoVerificationFailed("logical dbin_prefix length overflow"))?;
    let l_dbin_off = dir.dbin_offset.get();
    if l_dbin_len == 0 {
        reject_if(
            l_dbin_off != 0 || logical_prefix.len() != hdr_aligned,
            "unexpected DNPL logical_prefix dbin_offset or trailing bytes",
        )?;
        return Ok(Vec::new());
    }
    reject_if(
        dir.shared_binary_length.get() != 0 || !shared_bytes.is_empty(),
        "DNPL shared_extent must be empty when dbin_prefix_bytes > 0",
    )?;
    reject_if(
        (l_dbin_len & 63) != 0 || (l_dbin_off & 63) != 0,
        "misaligned DNPL logical dbin_prefix offset or length",
    )?;
    let auth_aligned = align64_u64(
        dir.authentication_envelope_offset
            .get()
            .saturating_add(dir.authentication_envelope_length.get()),
    )
    .ok_or(DnamError::CryptoVerificationFailed(
        "logical auth end overflow",
    ))?;
    let dbin_end = l_dbin_off
        .checked_add(l_dbin_len as u64)
        .and_then(align64_u64)
        .ok_or(DnamError::CryptoVerificationFailed(
            "logical dbin end overflow",
        ))?;
    reject_if(
        l_dbin_off < auth_aligned || dir.tier_start.get() != dbin_end,
        "inconsistent DNPL logical dbin_offset or tier_start",
    )?;
    let expected_end =
        hdr_aligned
            .checked_add(l_dbin_len)
            .ok_or(DnamError::CryptoVerificationFailed(
                "logical_prefix dbin end overflow",
            ))?;
    reject_if(
        logical_prefix.len() != expected_end,
        "DNPL logical_prefix length mismatch against dbin_prefix_bytes",
    )?;
    Ok(logical_prefix[hdr_aligned..expected_end].to_vec())
}

pub(crate) fn assemble_unsealed_open_dnam(
    unsealed: &UnsealedDnslInner,
) -> Result<Vec<u8>, DnamError> {
    let mut header = DnamJsonHeader::from_json_slice(&unsealed.header_json_bytes)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    if let Some(ref mut_val) = unsealed.mutable_json {
        merge_mutable_fields(&mut header, mut_val)?;
    }
    let mut_json = extract_mutable_json_bytes(&header)?;
    let dmu1 = build_dmu1_mutable_frame(&mut_json, DEFAULT_DMU1_TRAILER_CAPACITY)?;

    let auth_len = 64usize;
    let hdr_off = 192u64;
    let hdr_len = unsealed.header_json_bytes.len() as u64;
    let shared_off = align64_u64(hdr_off + hdr_len).ok_or(DnamError::CryptoVerificationFailed(
        "DNSL unseal shared_off overflow",
    ))?;
    let shared_len = unsealed.shared_bytes.len() as u64;
    let auth_off = align64_u64(shared_off + shared_len).ok_or(
        DnamError::CryptoVerificationFailed("DNSL unseal auth_off overflow"),
    )?;
    let dbin_len = unsealed.dbin_prefix.len() as u64;
    let dbin_off = if dbin_len > 0 {
        align64_u64(auth_off + auth_len as u64).ok_or(DnamError::CryptoVerificationFailed(
            "DNSL unseal dbin_off overflow",
        ))?
    } else {
        0
    };
    let tier_start = align64_u64(if dbin_len > 0 {
        dbin_off + dbin_len
    } else {
        auth_off + auth_len as u64
    })
    .ok_or(DnamError::CryptoVerificationFailed(
        "DNSL unseal tier_start overflow",
    ))?;
    let mut tier_ends = [U64::new(0); 4];
    for t in 0..=(unsealed.retained_tier as usize) {
        tier_ends[t] = U64::new(tier_start + unsealed.tier_ends[t] as u64);
    }
    let preamble = DnamPreambleWire {
        magic: DNAM_MAGIC,
        version_major: U16::new(CONTAINER_VERSION_MAJOR),
        version_minor: U16::new(CONTAINER_VERSION_MINOR),
        protection_profile: 0,
        auth_kind: 0,
        maximum_tier: unsealed.retained_tier as u8,
        reserved_flag: 0,
        sample_rate_train_hz: U32::new(48_000),
        reserved_pad: [0u8; 48],
    };
    let dir = DnamDirectoryWire {
        immutable_header_offset: U64::new(hdr_off),
        immutable_header_length: U64::new(hdr_len),
        shared_binary_offset: U64::new(shared_off),
        shared_binary_length: U64::new(shared_len),
        authentication_envelope_offset: U64::new(auth_off),
        authentication_envelope_length: U64::new(auth_len as u64),
        dbin_offset: U64::new(dbin_off),
        dbin_prefix_bytes: U64::new(dbin_len),
        tier_start: U64::new(tier_start),
        original_tier_ends: tier_ends,
        reserved_tail: [0u8; 24],
    };
    let total_end = (tier_start as usize) + unsealed.arena_bytes.len();
    let mut out = Vec::with_capacity(total_end + dmu1.len() + 16);
    out.extend_from_slice(preamble.as_bytes());
    out.extend_from_slice(dir.as_bytes());
    out.extend_from_slice(&unsealed.header_json_bytes);
    pad_to_64(&mut out);
    out.extend_from_slice(&unsealed.shared_bytes);
    pad_to_64(&mut out);
    out.resize(out.len() + 64, 0);
    if !unsealed.dbin_prefix.is_empty() {
        out.extend_from_slice(&unsealed.dbin_prefix);
        pad_to_64(&mut out);
    }
    let mut crc = 0_u32;
    let mut crc_tiers = [0_u32; 4];
    let mut prev_t = 0_usize;
    for t in 0..=(unsealed.retained_tier as usize) {
        let end_t = unsealed.tier_ends[t];
        crc = crc32c_update(crc, &unsealed.arena_bytes[prev_t..end_t]);
        crc_tiers[t] = crc;
        prev_t = end_t;
    }
    let last_crc = crc_tiers[unsealed.retained_tier as usize];
    for slot in crc_tiers
        .iter_mut()
        .skip((unsealed.retained_tier as usize) + 1)
    {
        *slot = last_crc;
    }
    let mut flags_hint = 0_u16;
    if header.io_config.output_mode != "mono" {
        flags_hint |= FLAG_STEREO_OUT;
    }
    if header.metadata.signatures.is_some() {
        flags_hint |= FLAG_HAS_SIGNATURES;
    }
    let crc_dbin_prefix = if dbin_len > 0 {
        crc32c_update(0, &out[dbin_off as usize..(dbin_off + dbin_len) as usize])
    } else {
        0
    };
    let dcrc = DcrcAuthRecord {
        crc_prefix: crc32c_update(0, &out[..auth_off as usize]),
        crc_tiers,
        num_controls: header.controls_schema.len().min(u16::MAX as usize) as u16,
        flags_hint,
        tail_time_ms_bits: header.io_config.tail_time_ms.to_bits(),
        crc_dbin_prefix,
    };
    out[auth_off as usize..auth_off as usize + 64].copy_from_slice(&dcrc.to_bytes());
    out.extend_from_slice(&unsealed.arena_bytes);
    out.extend_from_slice(&dmu1);
    let dmue = DmueFooterWire {
        magic: *b"DMUE",
        frame_len: U64::new(dmu1.len() as u64),
        reserved: U32::new(0),
    };
    out.extend_from_slice(dmue.as_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_verified_logical_dbin_prefix_rejects_corruption() {
        let mut dir = DnamDirectoryWire {
            immutable_header_offset: U64::new(192),
            immutable_header_length: U64::new(2),
            shared_binary_offset: U64::new(256),
            shared_binary_length: U64::new(0),
            authentication_envelope_offset: U64::new(256),
            authentication_envelope_length: U64::new(64),
            dbin_offset: U64::new(320),
            dbin_prefix_bytes: U64::new(128),
            tier_start: U64::new(448),
            original_tier_ends: [U64::new(0); 4],
            reserved_tail: [0u8; 24],
        };
        let mut prefix = alloc::vec![0u8; 256 + 128];
        prefix[192..194].copy_from_slice(b"{}");
        prefix[256..384].fill(0xAA);
        let ok = extract_verified_logical_dbin_prefix(&prefix, &dir, 2, &[]).unwrap();
        assert_eq!(ok.len(), 128);

        // Non-empty shared_bytes alongside dbin_prefix fails closed.
        assert!(extract_verified_logical_dbin_prefix(&prefix, &dir, 2, &[1]).is_err());
        // Non-zero header padding fails closed.
        prefix[200] = 1;
        assert!(extract_verified_logical_dbin_prefix(&prefix, &dir, 2, &[]).is_err());
        prefix[200] = 0;
        // Length overflow or mismatched tier_start fails closed.
        dir.tier_start = U64::new(512);
        assert!(extract_verified_logical_dbin_prefix(&prefix, &dir, 2, &[]).is_err());
    }
}
