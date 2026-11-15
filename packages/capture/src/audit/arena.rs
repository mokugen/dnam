#![forbid(unsafe_code)]
//! Arena-Level 3-Phase Stability, Rail-Lockup & Head-Gain Audit (`dnam_capture::audit::arena`).

use std::collections::BTreeSet;

use dnam_engine::provenance::{STAGE3_KAUTZ_BASES, STAGE3_KAUTZ_BLOCKS_PER_TIER};
use dnam_engine::stages::weights::{PIPE_FLAG_STAGE4_EXT, has_pipeline_flag};

use crate::proj::schema::{AuditReport, DnamProjManifest};

use super::{
    audit_phase2_polyadaa_rail_lockup, clamp_head_spectral_gain_linear,
    compute_phase2_corner_barrier_loss, read_arena_f32, verify_dynamic_stability_excursion,
    write_arena_f32,
};

const STAGE4_EXT_BOUNDS: [(usize, f32, f32); 17] = [
    (4_704, 0.0, 4.0),
    (4_708, 0.0, 4.0),
    (4_712, 0.0, 48_000.0),
    (4_716, 0.0, 48_000.0),
    (4_720, -0.995, 0.995),
    (4_724, -0.995, 0.995),
    (4_728, 0.0, 0.9999),
    (4_732, 0.0, 4.0),
    (4_736, 0.0, 48_000.0),
    (4_740, 0.0, 48_000.0),
    (4_744, 0.0, 48_000.0),
    (4_748, 0.0, 48_000.0),
    (4_752, -0.995, 0.995),
    (4_756, 0.0, 0.9999),
    (4_760, 0.0, 1.0),
    (4_764, 0.0, 2_048.0),
    (4_768, 0.0, 50.0),
];

fn clamp_pole_dr_pair(
    arena_64k: &mut [u8],
    low_off: usize,
    dr_off: usize,
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    let r_low = read_arena_f32(arena_64k, low_off);
    let dr = read_arena_f32(arena_64k, dr_off);
    if r_low == 0.0 && dr == 0.0 {
        return;
    }
    let r_high = r_low + dr.max(0.0);
    let (r_c, _, _, _, ok) = verify_dynamic_stability_excursion(r_high, 0.0, 16.0, 0.0, 0.85, 0.5);
    if !ok || !r_low.is_finite() || !dr.is_finite() || r_low < 0.0 {
        let clamped_low = if r_low.is_finite() {
            r_low.clamp(0.0, r_c)
        } else {
            0.0
        };
        let clamped_dr = if dr.is_finite() {
            dr.clamp(0.0, (r_c - clamped_low).max(0.0))
        } else {
            0.0
        };
        write_arena_f32(arena_64k, low_off, clamped_low);
        write_arena_f32(arena_64k, dr_off, clamped_dr);
        phase1_set.insert("CLAMP-STABILITY".to_string());
        phase3_set.insert("CLAMP-DYNAMIC".to_string());
    }
}

fn clamp_stage1_and_stage3_poles(
    arena_64k: &mut [u8],
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    for &s1_base in &[256usize, 10_752] {
        if s1_base + 32 <= arena_64k.len() {
            for lane in 0..4 {
                clamp_pole_dr_pair(
                    arena_64k,
                    s1_base + lane * 4,
                    s1_base + 16 + lane * 4,
                    phase1_set,
                    phase3_set,
                );
            }
        }
    }

    for (&base, &blocks) in STAGE3_KAUTZ_BASES.iter().zip(&STAGE3_KAUTZ_BLOCKS_PER_TIER) {
        for blk in 0..blocks {
            let b_off = base + blk * 128;
            if b_off + 128 > arena_64k.len() {
                continue;
            }
            for lane in 0..4 {
                clamp_pole_dr_pair(
                    arena_64k,
                    b_off + lane * 4,
                    b_off + (4 + lane) * 4,
                    phase1_set,
                    phase3_set,
                );
            }
        }
    }
}

fn clamp_stage4_tank_and_ext(
    arena_64k: &mut [u8],
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    if arena_64k.len() >= 4_716 {
        let stage4_ext = has_pipeline_flag(arena_64k, PIPE_FLAG_STAGE4_EXT);
        for l in 0..4 {
            let d = read_arena_f32(arena_64k, 4_672 + l * 4);
            if d != 0.0 && (!d.is_finite() || !(4.0..=255.0).contains(&d)) {
                let cd = if d.is_finite() {
                    d.clamp(4.0, 255.0)
                } else {
                    4.0
                };
                write_arena_f32(arena_64k, 4_672 + l * 4, cd);
                phase1_set.insert("CLAMP-STABILITY".to_string());
            }
            if !stage4_ext || l == 0 {
                let a_diff = read_arena_f32(arena_64k, 4_700 + l * 4);
                if a_diff != 0.0 && (!a_diff.is_finite() || a_diff.abs() > 0.75) {
                    let c_ad = if a_diff.is_finite() {
                        a_diff.clamp(-0.75, 0.75)
                    } else {
                        0.0
                    };
                    write_arena_f32(arena_64k, 4_700 + l * 4, c_ad);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
            }
        }
        if stage4_ext && arena_64k.len() >= 4_772 {
            for &(off, lo, hi) in &STAGE4_EXT_BOUNDS {
                let v = read_arena_f32(arena_64k, off);
                if !v.is_finite() || v < lo || v > hi {
                    let clamped = if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
                    write_arena_f32(arena_64k, off, clamped);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
            }
        }
        let g_loop = read_arena_f32(arena_64k, 4_688);
        if g_loop != 0.0 {
            let (_, _, g_c, _, ok) =
                verify_dynamic_stability_excursion(0.95, 0.0, 64.0, 0.0, g_loop, 0.5);
            if !ok {
                write_arena_f32(arena_64k, 4_688, g_c);
                phase1_set.insert("CLAMP-STABILITY".to_string());
                phase3_set.insert("CLAMP-DYNAMIC".to_string());
            }
        }
    }

    for &(d_base, count, max_d) in &[
        (17_792usize, 4usize, 1_024.0_f32),
        (34_496, 4, 1_024.0),
        (48_896, 8, 2_048.0),
    ] {
        if d_base + count * 4 <= arena_64k.len() {
            for k in 0..count {
                let d = read_arena_f32(arena_64k, d_base + k * 4);
                if d != 0.0 && (!d.is_finite() || !(4.0..=max_d).contains(&d)) {
                    let cd = if d.is_finite() {
                        d.clamp(4.0, max_d)
                    } else {
                        4.0
                    };
                    write_arena_f32(arena_64k, d_base + k * 4, cd);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
            }
        }
    }
}

fn run_phase2_rail_and_barrier_audit(
    arena_64k: &mut [u8],
    phase1_set: &mut BTreeSet<String>,
    phase2_warns: &mut Vec<String>,
) {
    let g_now = read_arena_f32(arena_64k, 4_688).clamp(0.0, 0.9990);
    let tau_now = read_arena_f32(arena_64k, 4_672).max(4.0);
    let barrier =
        compute_phase2_corner_barrier_loss(0.9990, f64::from(g_now), 0.5, f64::from(tau_now));
    if barrier >= 1.0e4 {
        phase1_set.insert("CLAMP-STABILITY".to_string());
    }
    let (occ, safe_scale, locked) = audit_phase2_polyadaa_rail_lockup(arena_64k, 3.0, 0.25);
    if locked && arena_64k.len() >= 4_136 {
        let cur_b = read_arena_f32(arena_64k, 4_112).max(0.1);
        let cur_br = read_arena_f32(arena_64k, 4_116);
        write_arena_f32(arena_64k, 4_112, (cur_b * safe_scale).clamp(0.1, 4.0));
        if cur_br > 0.0 {
            write_arena_f32(arena_64k, 4_116, (cur_br * safe_scale).clamp(0.1, 4.0));
        }
        phase2_warns.push(format!(
            "[CLAMP-RAIL] PolyADAA corner rail occupancy {:.1}% > 95.0%; scaled V1A beta by {:.3}",
            occ * 100.0,
            safe_scale
        ));
    }
}

fn clamp_head_mode_slice_6db(
    arena_64k: &mut [u8],
    m_off: usize,
    phase3_set: &mut BTreeSet<String>,
) {
    if m_off + 512 > arena_64k.len() {
        return;
    }
    let mut l2_sq_l = 0.0_f32;
    let mut l2_sq_r = 0.0_f32;
    let mut any_non_finite = false;
    for i in 0..64 {
        let wl = read_arena_f32(arena_64k, m_off + i * 4);
        let wr = read_arena_f32(arena_64k, m_off + 256 + i * 4);
        if !wl.is_finite() || !wr.is_finite() {
            any_non_finite = true;
        } else {
            l2_sq_l += wl * wl;
            l2_sq_r += wr * wr;
        }
    }
    let rms_gain = (l2_sq_l.max(l2_sq_r) / 16.0).sqrt();
    let (scale, was_clamped) = clamp_head_spectral_gain_linear(rms_gain);
    if was_clamped || any_non_finite {
        for i in 0..128 {
            let off = m_off + i * 4;
            let v = read_arena_f32(arena_64k, off);
            write_arena_f32(arena_64k, off, if v.is_finite() { v * scale } else { 0.0 });
        }
        phase3_set.insert("CLAMP-HEAD-GAIN".to_string());
    }
}

fn clamp_head_modes_6db(arena_64k: &mut [u8], phase3_set: &mut BTreeSet<String>) {
    const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];
    for &h_base in &HEAD_BASES {
        for mode in 1..5 {
            clamp_head_mode_slice_6db(arena_64k, h_base + mode * 512, phase3_set);
        }
    }
    if arena_64k.len() >= 49_408 + 8_192 {
        for t in 0..4 {
            for mode in 0..4 {
                clamp_head_mode_slice_6db(arena_64k, 49_408 + t * 2_048 + mode * 512, phase3_set);
            }
        }
    }
    phase3_set.insert("NUMERIC-ENVELOPE".to_string());
}

fn run_3phase_arena_audit_with_mode(
    manifest: &mut DnamProjManifest,
    arena_64k: &mut [u8],
    clamp_backbone: bool,
) {
    let mut phase1_set: BTreeSet<String> = manifest
        .audit_report
        .as_ref()
        .map(|r| r.phase1_clamps.iter().cloned().collect())
        .unwrap_or_default();
    let mut phase2_warns: Vec<String> = manifest
        .audit_report
        .as_ref()
        .map(|r| r.phase2_rail_warnings.clone())
        .unwrap_or_default();
    let mut phase3_set: BTreeSet<String> = manifest
        .audit_report
        .as_ref()
        .map(|r| r.phase3_clamps.iter().cloned().collect())
        .unwrap_or_default();
    let prescriptions = manifest
        .audit_report
        .as_ref()
        .map(|r| r.prescriptions.clone())
        .unwrap_or_default();

    if clamp_backbone {
        phase2_warns.clear();
        clamp_stage1_and_stage3_poles(arena_64k, &mut phase1_set, &mut phase3_set);
        clamp_stage4_tank_and_ext(arena_64k, &mut phase1_set, &mut phase3_set);
        run_phase2_rail_and_barrier_audit(arena_64k, &mut phase1_set, &mut phase2_warns);
    }

    clamp_head_modes_6db(arena_64k, &mut phase3_set);

    manifest.audit_report = Some(AuditReport {
        phase1_clamps: phase1_set.into_iter().collect(),
        phase2_rail_warnings: phase2_warns,
        phase3_clamps: phase3_set.into_iter().collect(),
        prescriptions,
    });
}

/// Runs the complete Phase-1/2/3 stability, corner-barrier, `2,048`-sample `PolyADAA` rail-lockup,
/// dynamic excursion, and `+6.0 dB` head-gain audit in-place over `arena_64k`, updating
/// `manifest.audit_report` (`ref[train.audit.phase1.clamp-stability]`,
/// `ref[train.audit.phase2.corner-barrier]`, `ref[train.audit.phase2.clamp-rail]`,
/// `ref[train.audit.phase3.clamp-dynamic]`, `ref[train.audit.phase3.clamp-head-gain]`).
pub fn run_full_3phase_arena_audit(manifest: &mut DnamProjManifest, arena_64k: &mut [u8]) {
    run_3phase_arena_audit_with_mode(manifest, arena_64k, true);
}

/// Runs Phase-3 `[CLAMP-HEAD-GAIN]` (`Modes 1..8`) and `[NUMERIC-ENVELOPE]` without mutating an
/// already-frozen `StarCenter` physical backbone (`Path 1` fast-path or `preserve_backbone()`).
pub fn run_head_only_3phase_arena_audit(manifest: &mut DnamProjManifest, arena_64k: &mut [u8]) {
    run_3phase_arena_audit_with_mode(manifest, arena_64k, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::stages::weights::{ARENA0_PIPELINE_FLAGS_OFFSET, PIPE_MAGIC_HI};

    #[test]
    fn test_head_only_preserves_backbone_and_phase2_warnings() {
        let mut arena = vec![0u8; 64_000];
        write_arena_f32(&mut arena, 256, 1.05); // Stage I pole out of bounds (must stay untouched)
        write_arena_f32(&mut arena, 4_928 + 512, 20.0); // Dynamic Mode 1 head out of bounds
        write_arena_f32(&mut arena, 49_408, f32::NAN); // Ultra Mode 5 non-finite lane

        let mut manifest = DnamProjManifest::default_reference();
        manifest.audit_report = Some(AuditReport {
            phase1_clamps: vec!["CLAMP-RANGE".to_string()],
            phase2_rail_warnings: vec!["prior-rail-note".to_string()],
            phase3_clamps: vec![],
            prescriptions: vec!["keep-me".to_string()],
        });

        run_head_only_3phase_arena_audit(&mut manifest, &mut arena);
        assert_eq!(read_arena_f32(&arena, 256), 1.05);
        assert!(read_arena_f32(&arena, 4_928 + 512) < 20.0);
        assert_eq!(read_arena_f32(&arena, 49_408), 0.0);

        let ar = manifest.audit_report.unwrap();
        assert_eq!(ar.phase2_rail_warnings, vec!["prior-rail-note".to_string()]);
        assert_eq!(ar.prescriptions, vec!["keep-me".to_string()]);
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-HEAD-GAIN"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "NUMERIC-ENVELOPE"));
    }

    #[test]
    fn test_full_arena_audit_stage4_ext_nonfinite_and_rail_scaling() {
        let mut arena = vec![0u8; 64_000];
        let flags_off = ARENA0_PIPELINE_FLAGS_OFFSET;
        arena[flags_off..flags_off + 4]
            .copy_from_slice(&(PIPE_MAGIC_HI | PIPE_FLAG_STAGE4_EXT).to_le_bytes());

        // Stage I non-finite pole + Ultra delay rails + PolyADAA rail lockup betas
        write_arena_f32(&mut arena, 256, f32::NAN);
        write_arena_f32(&mut arena, 10_752, 0.99);
        write_arena_f32(&mut arena, 10_752 + 16, 0.05);
        write_arena_f32(&mut arena, 17_792, f32::NAN);
        write_arena_f32(&mut arena, 48_896, 5_000.0);
        write_arena_f32(&mut arena, 4_112, 8.0);
        write_arena_f32(&mut arena, 4_116, 2.0);

        // Stage IV tank: lane 0 a_diff still clamped; lane 1 overlaps extended table (`4_704..4_772`)
        write_arena_f32(&mut arena, 4_700, 0.9);
        write_arena_f32(&mut arena, 4_704, 3.5); // In-bounds when STAGE4_EXT ([0, 4]) even though > 0.75
        write_arena_f32(&mut arena, 4_708, 9.0); // Out of bounds ([0, 4]) -> clamped to 4.0
        write_arena_f32(&mut arena, 4_720, -1.5); // Out of bounds ([-0.995, 0.995]) -> -0.995
        write_arena_f32(&mut arena, 4_768, f32::NAN); // Non-finite -> 0.0

        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut arena);

        assert_eq!(read_arena_f32(&arena, 256), 0.0);
        assert!(read_arena_f32(&arena, 10_752) + read_arena_f32(&arena, 10_752 + 16) <= 0.9996);
        assert_eq!(read_arena_f32(&arena, 17_792), 4.0);
        assert_eq!(read_arena_f32(&arena, 48_896), 2_048.0);
        assert_eq!(read_arena_f32(&arena, 4_700), 0.75);
        assert_eq!(read_arena_f32(&arena, 4_704), 3.5);
        assert_eq!(read_arena_f32(&arena, 4_708), 4.0);
        assert_eq!(read_arena_f32(&arena, 4_720), -0.995);
        assert_eq!(read_arena_f32(&arena, 4_768), 0.0);
        assert!(read_arena_f32(&arena, 4_112) < 8.0);
        assert!(read_arena_f32(&arena, 4_116) < 2.0);

        let ar = manifest.audit_report.unwrap();
        assert!(ar.phase1_clamps.iter().any(|t| t == "CLAMP-STABILITY"));
        assert!(!ar.phase2_rail_warnings.is_empty());
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-DYNAMIC"));
    }

    #[test]
    fn test_full_arena_audit_non_ext_stage4_adiff_clamp() {
        let mut arena = vec![0u8; 64_000];
        // Without PIPE_FLAG_STAGE4_EXT, all 4 a_diff lanes at 4_700..4_716 are clamped to +-0.75
        write_arena_f32(&mut arena, 4_704, -1.25);
        write_arena_f32(&mut arena, 4_708, f32::NAN);

        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut arena);

        assert_eq!(read_arena_f32(&arena, 4_704), -0.75);
        assert_eq!(read_arena_f32(&arena, 4_708), 0.0);
    }
}
