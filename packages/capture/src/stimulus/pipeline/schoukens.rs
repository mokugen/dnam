#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Two-Tier `Schoukens–Pintelon` Excitation Builder, `9.0 s -> 2.5 s` Condenser, and `SPCM`
//! `zerocopy` Serializer (`stimulus::pipeline::schoukens`).
//!
//! Canonical References:
//! - `docs/specs/training/capture_and_alignment.md` (`ref[train.capture.excitation.codesigned]`)
//! - `docs/specs/containers/project_container_format.md` (`ref[container.proj.inpc-capsule]`)

pub mod analysis;
pub mod edc;
pub mod regimes;

use core::mem::{align_of, size_of};
use std::collections::BTreeSet;

use dnam_engine::schema::DnamError;
use zerocopy::byteorder::little_endian::F32;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::stimulus::laws::TopologyProbeMode;
use crate::{CORESET_FP32_MAX_FRAMES_PER_STEM, CORESET_FP32_MIN_FRAMES_PER_STEM};

pub use analysis::{
    LAW0A_MIN_PERIOD_COHERENCE, LAW0A_REL_MISMATCH_FLOOR, Regime2DecayInput,
    SchoukensAnalysisConfig, SchoukensPhysicsResult, analyze_schoukens_pintelon_physics,
};
pub use edc::compute_schroeder_edc_t60_s;
pub use regimes::{
    OddRandomMultisinePlan, SCHOUKENS_CORESET_20S_OFFSETS, SCHOUKENS_CORESET_25S_OFFSETS,
    SCHOUKENS_PEAK_CEILING_FS, SCHOUKENS_R1_MAX_CREST_FACTOR_LIN, SCHOUKENS_R1_PERIOD_SAMPLES,
    SCHOUKENS_R1_SPCM_PERIODS, SCHOUKENS_SPCM_CANONICAL_FRAMES, SCHOUKENS_SPCM_REGIME_OFFSETS,
    build_odd_random_multisine_plan, regime2_gap_ranges, synthesize_odd_random_multisine_period,
    synthesize_regime2_staircase_and_gaps, synthesize_regime3_chirp_and_imd,
    synthesize_regime4_topology_contrast,
};
use regimes::{
    SCHOUKENS_R3_20S_IMD_LEN, SCHOUKENS_R3_20S_MONO_LEN, SCHOUKENS_R3_25S_IMD_LEN,
    SCHOUKENS_R3_25S_MONO_LEN, SCHOUKENS_SPCM_R2_BLOCK_LEN, SCHOUKENS_SPCM_R2_BURST_LEN,
    regime2_coreset_block_specs,
};

/// Declarative `64 B` wire header preceding raw `fp32` samples inside `.dnam-proj` `SPCM` chunks.
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SpcmHeader64BWire {
    /// Magic `"SPCM"`.
    pub magic: [u8; 4],
    /// Zeroed `60 B` header pad (`reserved`).
    pub reserved: [u8; 60],
}

const _: () = assert!(size_of::<SpcmHeader64BWire>() == 64);
const _: () = assert!(align_of::<SpcmHeader64BWire>() == 1);

/// Complete 4-regime `Schoukens–Pintelon` excitation bundle (`9.0 s` `SPCM` or `2.0–2.5 s` coreset).
#[derive(Debug, Clone, PartialEq)]
pub struct SchoukensExcitation {
    /// Primary `48 kHz` dry stimulus (`in_1`).
    pub dry_l: Vec<f32>,
    /// Optional companion Right / Aux-Key `48 kHz` dry stimulus (`in_2`).
    pub dry_r: Option<Vec<f32>>,
    /// Optional auxiliary/return `48 kHz` stimulus (`in_3`) for `2->1` / `4CM`.
    pub aux_in_3: Option<Vec<f32>>,
    /// Deterministic `Regime 1` odd-random bin partition.
    pub r1_plan: OddRandomMultisinePlan,
    /// Single-period (`16,384`-sample) `Regime 1` dry waveform on `L`.
    pub r1_period_l: Vec<f32>,
    /// Monotonic sample boundaries `[0, r2_start, r3_start, r4_start, total_frames]`.
    pub regime_offsets: [u32; 5],
    /// Global sample `(start, end)` silent decay intervals inside `Regime 2`.
    pub r2_gap_ranges_global: Vec<(usize, usize)>,
    /// True if generated as full `8.0–12.0 s` transient `SPCM`; false for `2.0–2.5 s` coreset.
    pub is_spcm_tier1: bool,
}

fn assemble_excitation(
    seed_256: &[u8; 256],
    topology_mode: TopologyProbeMode,
    squeal_mask_bins: &BTreeSet<usize>,
    offsets: [u32; 5],
    r1_periods: usize,
    is_spcm_tier1: bool,
) -> SchoukensExcitation {
    let r1_plan = build_odd_random_multisine_plan(seed_256, squeal_mask_bins);
    let (r1_period_l, _) =
        synthesize_odd_random_multisine_period(&r1_plan, seed_256, 0, SCHOUKENS_PEAK_CEILING_FS);
    let r1_period_r_opt = (topology_mode == TopologyProbeMode::TrueStereo2To2).then(|| {
        synthesize_odd_random_multisine_period(&r1_plan, seed_256, 1, SCHOUKENS_PEAK_CEILING_FS).0
    });

    let r1_len = offsets[1] as usize;
    let r2_len = (offsets[2] - offsets[1]) as usize;
    let r3_len = (offsets[3] - offsets[2]) as usize;
    let r4_len = (offsets[4] - offsets[3]) as usize;
    let total = offsets[4] as usize;

    let mut dry_l = Vec::with_capacity(total);
    for _ in 0..r1_periods {
        dry_l.extend_from_slice(&r1_period_l);
    }
    dry_l.truncate(r1_len);
    dry_l.extend_from_slice(&synthesize_regime2_staircase_and_gaps(r2_len));
    dry_l.extend_from_slice(&synthesize_regime3_chirp_and_imd(r3_len));

    let (r4_l, r4_r_opt, r4_aux_opt) = synthesize_regime4_topology_contrast(r4_len, topology_mode);
    let prefix_len = dry_l.len();
    dry_l.extend_from_slice(&r4_l);

    let r2_global: Vec<(usize, usize)> = regime2_gap_ranges(r2_len)
        .into_iter()
        .map(|(s, e)| (offsets[1] as usize + s, offsets[1] as usize + e))
        .collect();

    let dry_r = match topology_mode {
        TopologyProbeMode::TrueStereo2To2 => {
            let mut r = Vec::with_capacity(total);
            if let Some(ref p_r) = r1_period_r_opt {
                for _ in 0..r1_periods {
                    r.extend_from_slice(p_r);
                }
            }
            r.truncate(r1_len);
            r.extend_from_slice(&dry_l[r1_len..prefix_len]);
            if let Some(r4_r) = r4_r_opt {
                r.extend_from_slice(&r4_r);
            }
            Some(r)
        }
        TopologyProbeMode::DualInToMono2To1 => {
            let mut r = vec![0.0_f32; prefix_len];
            if let Some(ref r4_r) = r4_r_opt {
                r.extend_from_slice(r4_r);
            }
            Some(r)
        }
        _ => None,
    };

    let aux_in_3 = match topology_mode {
        TopologyProbeMode::DualInToMono2To1 => {
            let mut a = vec![0.0_f32; prefix_len];
            if let Some(r4_a) = r4_aux_opt {
                a.extend_from_slice(&r4_a);
            }
            Some(a)
        }
        TopologyProbeMode::FourCableMethod4Cm => {
            let mut a = Vec::with_capacity(total);
            a.extend_from_slice(&dry_l[..prefix_len]);
            if let Some(r4_a) = r4_aux_opt {
                a.extend_from_slice(&r4_a);
            }
            Some(a)
        }
        _ => None,
    };

    SchoukensExcitation {
        dry_l,
        dry_r,
        aux_in_3,
        r1_plan,
        r1_period_l,
        regime_offsets: offsets,
        r2_gap_ranges_global: r2_global,
        is_spcm_tier1,
    }
}

/// Computes monotonic `[0, r2_start, r3_start, r4_start, total]` offsets for `coreset_frames` in `96_000..=120_000`.
#[must_use]
pub fn coreset_offsets_for_frames(coreset_frames: usize) -> [u32; 5] {
    let total = (coreset_frames.clamp(
        CORESET_FP32_MIN_FRAMES_PER_STEM,
        CORESET_FP32_MAX_FRAMES_PER_STEM,
    )) & !15;
    if total == CORESET_FP32_MAX_FRAMES_PER_STEM {
        return SCHOUKENS_CORESET_25S_OFFSETS;
    }
    if total == CORESET_FP32_MIN_FRAMES_PER_STEM {
        return SCHOUKENS_CORESET_20S_OFFSETS;
    }
    let r1 = SCHOUKENS_R1_PERIOD_SAMPLES;
    let rem = total - r1;
    let r2 = (rem * 45) / 100;
    let r3 = (rem * 30) / 100;
    [
        0,
        r1 as u32,
        (r1 + r2) as u32,
        (r1 + r2 + r3) as u32,
        total as u32,
    ]
}

/// Synthesizes full Tier-1 `9.0 s` (`432,000`-frame) 4-regime `Schoukens–Pintelon` excitation for transient `SPCM`.
#[must_use]
pub fn build_schoukens_pintelon_spcm_48k(
    seed_256: &[u8; 256],
    topology_mode: TopologyProbeMode,
    squeal_mask_bins: &BTreeSet<usize>,
) -> SchoukensExcitation {
    assemble_excitation(
        seed_256,
        topology_mode,
        squeal_mask_bins,
        SCHOUKENS_SPCM_REGIME_OFFSETS,
        SCHOUKENS_R1_SPCM_PERIODS,
        true,
    )
}

/// Synthesizes condensed Tier-2 `2.0–2.5 s` (`96,000–120,000`-frame, canonical `120,000`) 4-regime `Schoukens–Pintelon` excitation.
#[must_use]
pub fn build_schoukens_pintelon_coreset_48k(
    seed_256: &[u8; 256],
    topology_mode: TopologyProbeMode,
    coreset_frames: usize,
    squeal_mask_bins: &BTreeSet<usize>,
) -> SchoukensExcitation {
    let offsets = coreset_offsets_for_frames(coreset_frames);
    let r1_periods = (offsets[1] as usize) / SCHOUKENS_R1_PERIOD_SAMPLES;
    assemble_excitation(
        seed_256,
        topology_mode,
        squeal_mask_bins,
        offsets,
        r1_periods.max(1),
        false,
    )
}

fn push_condense_channel(
    spcm: &[f32],
    offsets: [u32; 5],
    topology_mode: TopologyProbeMode,
    out: &mut Vec<f32>,
) {
    let r1_len = offsets[1] as usize;
    let r2_len = (offsets[2] - offsets[1]) as usize;
    let r3_len = (offsets[3] - offsets[2]) as usize;
    let r4_len = (offsets[4] - offsets[3]) as usize;

    // Regime 1: skip Period 0 [0..16_384), copy settled periods [16_384 .. 16_384 + r1_len)
    let r1_start = SCHOUKENS_R1_PERIOD_SAMPLES;
    out.extend_from_slice(&spcm[r1_start..r1_start + r1_len]);

    // Regime 2: proportionally extract burst + trailing decay gap from all 4 SPCM Regime 2 blocks,
    // matching `regime2_coreset_block_specs(r2_len)` so `regime2_gap_ranges` aligns identically.
    let spcm_r2_start = SCHOUKENS_SPCM_REGIME_OFFSETS[1] as usize;
    let block_specs = regime2_coreset_block_specs(r2_len);
    for (b, &(act_take, gap_take)) in block_specs.iter().enumerate() {
        let b_mid = spcm_r2_start + b * SCHOUKENS_SPCM_R2_BLOCK_LEN + SCHOUKENS_SPCM_R2_BURST_LEN;
        out.extend_from_slice(&spcm[b_mid - act_take..b_mid + gap_take]);
    }

    // Regime 3: 55% monophonic chirp + 45% tritone IMD sweep from full-band SPCM sub-sweeps
    let spcm_r3_start = SCHOUKENS_SPCM_REGIME_OFFSETS[2] as usize;
    let spcm_r3_len =
        (SCHOUKENS_SPCM_REGIME_OFFSETS[3] - SCHOUKENS_SPCM_REGIME_OFFSETS[2]) as usize;
    let spcm_r3_split = (spcm_r3_len * 55) / 100;
    let c_mono = ((r3_len * 55) / 100).min(spcm_r3_split);
    let c_imd = r3_len
        .saturating_sub(c_mono)
        .min(spcm_r3_len - spcm_r3_split);
    let mono_src = if c_mono <= SCHOUKENS_R3_20S_MONO_LEN {
        spcm_r3_start + SCHOUKENS_R3_25S_MONO_LEN
    } else {
        spcm_r3_start
    };
    out.extend_from_slice(&spcm[mono_src..mono_src + c_mono]);
    let imd_base = spcm_r3_start + spcm_r3_split;
    let imd_src = if c_imd <= SCHOUKENS_R3_20S_IMD_LEN {
        imd_base + SCHOUKENS_R3_25S_IMD_LEN
    } else {
        imd_base
    };
    out.extend_from_slice(&spcm[imd_src..imd_src + c_imd]);

    // Regime 4: proportionally take slices across topology-matched SPCM Regime 4 contrast zones
    let spcm_r4_start = SCHOUKENS_SPCM_REGIME_OFFSETS[3] as usize;
    let spcm_r4_len =
        (SCHOUKENS_SPCM_REGIME_OFFSETS[4] - SCHOUKENS_SPCM_REGIME_OFFSETS[3]) as usize;
    let zones = match topology_mode {
        TopologyProbeMode::DualInToMono2To1 => 4_usize,
        TopologyProbeMode::FourCableMethod4Cm => 3_usize,
        _ => 5_usize,
    };
    let src_zone = spcm_r4_len / zones;
    let dst_base = r4_len / zones;
    for z in 0..zones {
        let take = if z + 1 == zones {
            r4_len - dst_base * (zones - 1)
        } else {
            dst_base
        };
        let s0 = spcm_r4_start + z * src_zone;
        out.extend_from_slice(&spcm[s0..s0 + take]);
    }
}

/// Condensed `(dry_or_wet_l, Option<dry_or_wet_r>, coreset_regime_offsets)` output of
/// [`condense_spcm_to_coreset`].
pub type CondensedCoresetTuple = (Vec<f32>, Option<Vec<f32>>, [u32; 5]);

/// Condenses aligned `9.0 s` (`432,000`-frame) Tier-1 `SPCM` recordings into the canonical
/// `2.0–2.5 s` (`96,000–120,000`-frame) 4-regime coreset (dropping `Period 0`), respecting
/// `topology_mode`'s Regime 4 zone count (`3`, `4`, or `5`).
pub fn condense_spcm_to_coreset_for_topology(
    spcm_l: &[f32],
    spcm_r: Option<&[f32]>,
    target_frames: usize,
    topology_mode: TopologyProbeMode,
) -> Result<CondensedCoresetTuple, DnamError> {
    if spcm_l.len() < SCHOUKENS_SPCM_CANONICAL_FRAMES {
        return Err(DnamError::BufferTooSmall {
            expected: SCHOUKENS_SPCM_CANONICAL_FRAMES,
            actual: spcm_l.len(),
        });
    }
    if let Some(r) = spcm_r
        && r.len() < SCHOUKENS_SPCM_CANONICAL_FRAMES
    {
        return Err(DnamError::BufferTooSmall {
            expected: SCHOUKENS_SPCM_CANONICAL_FRAMES,
            actual: r.len(),
        });
    }
    let offsets = coreset_offsets_for_frames(target_frames);
    let total = offsets[4] as usize;
    let mut out_l = Vec::with_capacity(total);
    push_condense_channel(spcm_l, offsets, topology_mode, &mut out_l);
    let out_r = spcm_r.map(|r| {
        let mut buf = Vec::with_capacity(total);
        push_condense_channel(r, offsets, topology_mode, &mut buf);
        buf
    });
    Ok((out_l, out_r, offsets))
}

/// Condenses aligned `9.0 s` (`432,000`-frame) Tier-1 `SPCM` recordings into the canonical
/// `2.0–2.5 s` (`96,000–120,000`-frame) 4-regime coreset (dropping `Period 0`), returning
/// `(condensed_l, Option<condensed_r>, coreset_regime_offsets)`.
pub fn condense_spcm_to_coreset(
    spcm_l: &[f32],
    spcm_r: Option<&[f32]>,
    target_frames: usize,
) -> Result<CondensedCoresetTuple, DnamError> {
    condense_spcm_to_coreset_for_topology(
        spcm_l,
        spcm_r,
        target_frames,
        TopologyProbeMode::TrueStereo2To2,
    )
}

/// Encodes mono or interleaved stereo `48 kHz` audio into a `.dnam-proj` `SPCM` payload
/// (`SpcmHeader64BWire` + IEEE-754 little-endian `f32` PCM samples).
#[must_use]
pub fn encode_spcm_chunk(ch_l: &[f32], ch_r: Option<&[f32]>) -> Vec<u8> {
    let n = ch_r.map_or(ch_l.len(), |r| ch_l.len().min(r.len()));
    let hdr = SpcmHeader64BWire {
        magic: *b"SPCM",
        reserved: [0; 60],
    };
    let mut out = Vec::with_capacity(64 + n * if ch_r.is_some() { 8 } else { 4 });
    out.extend_from_slice(hdr.as_bytes());
    let mut words = Vec::with_capacity(n * if ch_r.is_some() { 2 } else { 1 });
    for i in 0..n {
        let l = ch_l[i];
        words.push(F32::new(if l.is_finite() { l } else { 0.0 }));
        if let Some(r_sl) = ch_r {
            let r = r_sl[i];
            words.push(F32::new(if r.is_finite() { r } else { 0.0 }));
        }
    }
    out.extend_from_slice(words.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spcm_and_coreset_builders_condense_and_encode_spcm_chunk() {
        let seed = [0x42_u8; 256];
        let spcm = build_schoukens_pintelon_spcm_48k(
            &seed,
            TopologyProbeMode::TrueStereo2To2,
            &BTreeSet::new(),
        );
        assert!(spcm.is_spcm_tier1);
        assert_eq!(spcm.dry_l.len(), SCHOUKENS_SPCM_CANONICAL_FRAMES);
        assert_eq!(
            spcm.dry_r.as_ref().unwrap().len(),
            SCHOUKENS_SPCM_CANONICAL_FRAMES
        );
        assert_eq!(spcm.regime_offsets, SCHOUKENS_SPCM_REGIME_OFFSETS);
        assert_eq!(spcm.r2_gap_ranges_global.len(), 4);

        let spcm_chunk = encode_spcm_chunk(&spcm.dry_l, spcm.dry_r.as_deref());
        let (hdr, _) = SpcmHeader64BWire::ref_from_prefix(&spcm_chunk).unwrap();
        assert_eq!(&hdr.magic, b"SPCM");
        assert_eq!(spcm_chunk.len(), 64 + SCHOUKENS_SPCM_CANONICAL_FRAMES * 8);

        let (c25_l, c25_r, off25) =
            condense_spcm_to_coreset(&spcm.dry_l, spcm.dry_r.as_deref(), 120_000).unwrap();
        assert_eq!(off25, SCHOUKENS_CORESET_25S_OFFSETS);
        assert_eq!(c25_l.len(), 120_000);
        assert_eq!(c25_r.unwrap().len(), 120_000);
        // R1 settled Period 1 == Period 2 in condensed 2.5 s coreset
        assert_eq!(&c25_l[..16_384], &c25_l[16_384..32_768]);

        let (c20_l, _, off20) = condense_spcm_to_coreset(&spcm.dry_l, None, 96_000).unwrap();
        assert_eq!(off20, SCHOUKENS_CORESET_20S_OFFSETS);
        assert_eq!(c20_l.len(), 96_000);

        let direct25 = build_schoukens_pintelon_coreset_48k(
            &seed,
            TopologyProbeMode::Mono1To1,
            120_000,
            &BTreeSet::new(),
        );
        assert!(!direct25.is_spcm_tier1);
        assert_eq!(direct25.dry_l.len(), 120_000);
        assert_eq!(direct25.regime_offsets, SCHOUKENS_CORESET_25S_OFFSETS);

        // Intermediate target_frames (108,000), NaN/Inf SPCM encoding sanitization, BufferTooSmall guards,
        // and all 6 topology modes in `condense_spcm_to_coreset_for_topology`.
        assert!(condense_spcm_to_coreset(&spcm.dry_l[..1_000], None, 96_000).is_err());
        assert!(condense_spcm_to_coreset(&spcm.dry_l, Some(&spcm.dry_l[..1_000]), 96_000).is_err());
        let sanitized = encode_spcm_chunk(&[f32::NAN, 0.25], Some(&[0.5, f32::INFINITY]));
        assert_eq!(sanitized.len(), 64 + 16);

        for topo in [
            TopologyProbeMode::Mono1To1,
            TopologyProbeMode::MonoToStereo1To2,
            TopologyProbeMode::TrueStereo2To2,
            TopologyProbeMode::DualInToMono2To1,
            TopologyProbeMode::FourCableMethod4Cm,
            TopologyProbeMode::WetDryWet1To3,
        ] {
            let (c108_l, c108_r, off108) = condense_spcm_to_coreset_for_topology(
                &spcm.dry_l,
                spcm.dry_r.as_deref(),
                108_000,
                topo,
            )
            .unwrap();
            assert_eq!(off108[4] as usize, 108_000);
            assert_eq!(c108_l.len(), 108_000);
            assert_eq!(c108_r.unwrap().len(), 108_000);
        }
    }
}
