#![forbid(unsafe_code)]
//! Layer 0 `.dnam-bin` (`"DNBN"`) compilation, transport delay-history synchronization, and
//! resting `[STAB-ZERO-DC]` verification for `proj::bundle` (`ref[container.proj.wght-and-export]`).

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{encode_cp1_shared_stream, extract_dnam_bin};
use dnam_engine::schema::{DelayHistoryEntry, DnamError, DnamJsonHeader};
use dnam_kernels::{MAX_PARALRU_R_F32, fp16_to_f32};
use dnam_vm::SonaVm;
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3UltraDeltaWire, DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, DnamBinBuildInput,
    DnamBinView, TRANSPORT_FLAG_EXT_F32, TransportQuadWire, encode_dnam_bin,
};
use dnam_vm::control::Cp1ContributorSlot;
use dnam_vm::math::DefaultMathBackend;
use sha2::{Digest, Sha256};
use zerocopy::{FromBytes, IntoBytes};

use super::super::schema::DnamProjManifest;

#[inline]
pub(super) fn map_vm_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

fn quad_max_samples_48k(quad: &TransportQuadWire) -> u64 {
    let mut max_req = 0_u64;
    for lane in 0..4 {
        let d = quad.delay_samples[lane].get().max(0.0);
        let depth = fp16_to_f32(quad.lfo_depths_fp16[lane].get()).abs();
        let req = (d + depth).ceil().max(4.0) as u64 + 6;
        max_req = max_req.max(req);
    }
    max_req
}

/// Synchronizes `header.delay_histories` (`owner: "transport"`) from a validated `.dnam-bin`
/// image so engine hosts pre-allocate extended ring buffers without RT allocation
/// (`ref[dsp.op-t.ring.declaration]`, `ref[container.dnam.delay-histories]`).
pub(crate) fn sync_manifest_transport_delay_history(
    header: &mut DnamJsonHeader,
    dnam_bin_bytes: &[u8],
) {
    header.delay_histories.retain(|d| d.owner != "transport");
    let Ok(view) = DnamBinView::parse(dnam_bin_bytes) else {
        return;
    };
    let max_tier = view.retained_tier.index();
    let tier_names = ["lite", "full", "pro", "ultra"];
    let quads: [Option<&TransportQuadWire>; 4] = [
        Some(&view.arena0.transport_lite),
        view.arena1.map(|a| &a.transport_full),
        view.arena2.map(|a| &a.transport_pro[0]),
        view.arena3.map(|a| &a.transport_ultra[0]),
    ];

    for (q_idx, q_opt) in quads.iter().enumerate() {
        let Some(quad) = q_opt else {
            continue;
        };
        let req = quad_max_samples_48k(quad);
        let use_f32 = (quad.flags & TRANSPORT_FLAG_EXT_F32) != 0;
        if req <= 128 && !use_f32 {
            continue;
        }
        let mut refs = BTreeMap::new();
        for tier_name in tier_names.iter().take(max_tier + 1).skip(q_idx) {
            refs.insert((*tier_name).to_string(), format!("{req}"));
        }
        header.delay_histories.push(DelayHistoryEntry {
            id: format!("transport.quad{q_idx}"),
            owner: "transport".to_string(),
            kind: "fractional".to_string(),
            introduced_tier: tier_names[q_idx].to_string(),
            sample_format: if use_f32 { "f32" } else { "fp16" }.to_string(),
            reference_samples: refs,
        });
    }
}

/// Verifies `[STAB-ZERO-DC]` resting output invariance (`<= 1.0e-5`, `-100 dBFS`) on `SonaVm`
/// across retained `Lite` (and `Ultra` when present) tiers of a `.dnam-bin` program.
pub fn verify_dnam_bin_zero_dc(dnam_bin_bytes: &[u8]) -> Result<(), DnamError> {
    let view = DnamBinView::parse(dnam_bin_bytes).map_err(map_vm_err)?;
    let check_tiers: &[Tier] = if view.retained_tier == Tier::Ultra {
        &[Tier::Lite, Tier::Ultra]
    } else {
        &[Tier::Lite]
    };
    for &tier in check_tiers {
        let mut vm = SonaVm::<DefaultMathBackend>::from_view_owned(&view, tier, 48_000.0, false)
            .map_err(map_vm_err)?;
        vm.reset();
        for _ in 0..4 {
            let (yl, yr) = vm.step_frame(0.0, 0.0).map_err(map_vm_err)?;
            if !yl.is_finite() || !yr.is_finite() || yl.abs() > 1.0e-5 || yr.abs() > 1.0e-5 {
                return Err(DnamError::SchemaValidationError(format!(
                    "WGHT {tier:?} violates [STAB-ZERO-DC]: resting DC output ({yl:.2e}, {yr:.2e}) > 1e-5"
                )));
            }
        }
    }
    Ok(())
}

/// Verifies `[STAB-ZERO-DC]` on an open runtime `.dnam` container (`WGHT` chunk).
pub fn verify_wght_zero_dc(wght_dnam: &[u8]) -> Result<(), DnamError> {
    let dbin = extract_dnam_bin(wght_dnam)?;
    verify_dnam_bin_zero_dc(dbin)
}

/// Compiles raw `SONA v3.0` `Arena 0..3` bytes (`4,608 / 7,936 / 15,872 / 30,720 B`) and `CP1`
/// slots into a validated Layer 0 `.dnam-bin` (`"DNBN"`) image (or returns a pre-encoded `.dnam-bin`).
pub(crate) fn compile_arena_and_cp1_to_dnam_bin(
    manifest: &DnamProjManifest,
    arena_or_dbin: &[u8],
    cp1_contributors: &[Cp1ContributorSlot],
) -> Result<Vec<u8>, DnamError> {
    if arena_or_dbin.len() >= 4 && &arena_or_dbin[0..4] == b"DNBN" {
        let _ = DnamBinView::parse(arena_or_dbin).map_err(map_vm_err)?;
        return Ok(arena_or_dbin.to_vec());
    }
    if arena_or_dbin.len() < ARENA_TIER_BYTES[0] {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_TIER_BYTES[0],
            actual: arena_or_dbin.len(),
        });
    }
    let a0 = Arena0LiteWire::ref_from_bytes(&arena_or_dbin[..ARENA_TIER_BYTES[0]])
        .map_err(|_| DnamError::SchemaValidationError("invalid Arena0LiteWire".to_string()))?;
    let a1 = if arena_or_dbin.len() == ARENA_TIER_BYTES[1]
        || arena_or_dbin.len() == ARENA_TIER_BYTES[2]
        || arena_or_dbin.len() == ARENA_TIER_BYTES[3]
    {
        Some(
            Arena1FullDeltaWire::ref_from_bytes(
                &arena_or_dbin[ARENA_TIER_BYTES[0]..ARENA_TIER_BYTES[1]],
            )
            .map_err(|_| {
                DnamError::SchemaValidationError("invalid Arena1FullDeltaWire".to_string())
            })?,
        )
    } else if arena_or_dbin.len() == ARENA_TIER_BYTES[0] {
        None
    } else {
        return Err(DnamError::SchemaValidationError(format!(
            "arena length {} must match canonical SONA tier end ({:?})",
            arena_or_dbin.len(),
            ARENA_TIER_BYTES
        )));
    };
    let a2 = (arena_or_dbin.len() >= ARENA_TIER_BYTES[2])
        .then(|| {
            Arena2ProDeltaWire::ref_from_bytes(
                &arena_or_dbin[ARENA_TIER_BYTES[1]..ARENA_TIER_BYTES[2]],
            )
        })
        .transpose()
        .map_err(|_| DnamError::SchemaValidationError("invalid Arena2ProDeltaWire".to_string()))?;
    let a3 = (arena_or_dbin.len() >= ARENA_TIER_BYTES[3])
        .then(|| {
            Arena3UltraDeltaWire::ref_from_bytes(
                &arena_or_dbin[ARENA_TIER_BYTES[2]..ARENA_TIER_BYTES[3]],
            )
        })
        .transpose()
        .map_err(|_| {
            DnamError::SchemaValidationError("invalid Arena3UltraDeltaWire".to_string())
        })?;

    let compiled_descs = manifest.base.compile_clamp_descriptors()?;
    let wire_descs: Vec<_> = compiled_descs.iter().map(|d| d.to_wire()).collect();
    let cp1_overflow = if cp1_contributors.is_empty() {
        Vec::new()
    } else {
        encode_cp1_shared_stream(cp1_contributors)?
    };

    let mut quads: Vec<&TransportQuadWire> = vec![&a0.transport_lite];
    if let Some(q1) = a1 {
        quads.push(&q1.transport_full);
    }
    if let Some(q2) = a2 {
        quads.push(&q2.transport_pro[0]);
    }
    if let Some(q3) = a3 {
        quads.push(&q3.transport_ultra[0]);
    }
    let any_ext = quads
        .iter()
        .any(|q| (q.flags & TRANSPORT_FLAG_EXT_F32) != 0);
    let max_req = quads
        .iter()
        .map(|q| quad_max_samples_48k(q))
        .max()
        .unwrap_or(16)
        .clamp(16, 524_288) as u32;
    let max_delay_48k = quads
        .iter()
        .flat_map(|q| q.delay_samples.iter().map(|d| d.get()))
        .fold(16.0_f32, f32::max)
        .ceil()
        .clamp(16.0, 48_000.0) as u32;
    let ext_pow2 = if any_ext || max_req > 128 {
        (max_req.max(256).next_power_of_two().trailing_zeros() as u8).clamp(8, 19)
    } else {
        0
    };
    let (ext_lines, ext_fmt) = if ext_pow2 > 0 {
        (16_u8, u8::from(any_ext))
    } else {
        (0_u8, 0_u8)
    };

    let mut hasher = Sha256::new();
    hasher.update(b"DNAM-BIN-CONTENT-V3\0");
    hasher.update(a0.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let mut prefix8 = [0_u8; 8];
    prefix8.copy_from_slice(&digest[..8]);

    let mut header_flags = 0_u16;
    if manifest.base.io_config.output_mode != "mono" {
        header_flags |= DNBN_FLAG_STEREO_OUT;
    }
    if manifest.base.io_config.inputs == 1 {
        header_flags |= DNBN_FLAG_SINGLE_INPUT;
    }
    let dc_pole = a0.transport_lite.r_dc.get().clamp(0.95, 0.9999);
    let tail_ms = manifest.base.io_config.tail_time_ms.clamp(0.0, 60_000.0);
    let ro_pole = manifest
        .base
        .linear_readout
        .ro_pole_48k
        .clamp(0.0, MAX_PARALRU_R_F32);

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: manifest.base.sample_rate_train_hz,
        header_flags,
        ext_ring_pow2_frames: ext_pow2,
        ext_ring_lines: ext_lines,
        ext_ring_format: ext_fmt,
        declared_max_delay_48k: max_delay_48k,
        dc_blocker_pole_5hz: dc_pole,
        tail_time_ms: tail_ms,
        ro_pole_48k: ro_pole,
        switch_xfade_samples: 64,
        content_hash_prefix: prefix8,
        controls: &wire_descs,
        cp1_overflow: &cp1_overflow,
        tsyn: &[],
        cext: None,
        arena0: a0,
        arena1: a1,
        arena2: a2,
        arena3: a3,
    })
    .map_err(map_vm_err)
}
