#![forbid(unsafe_code)]
//! Arena-Level 3-Phase Stability, GRU Bounded-State, Schur/SVF & Readout-Gain Audit over `v3`
//! `28,672 B` Matryoshka Arenas (`dnam_capture::audit::arena`).

mod recovery;

pub(crate) use recovery::{
    ensure_gru_block_zero_input_contraction, ensure_model_overdrive_silence_recovery,
};

use std::collections::BTreeSet;
use std::f32::consts::PI;

use dnam_engine::provenance::STAGE3_KAUTZ_BASES;
use dnam_engine::stages::stage1_svf::SVF_W_MAX;
use dnam_engine::stages::stage2_gru::{NUM_GRU_BLOCKS, V3ControlBusParams, V3GruBlockCold};
use dnam_engine::stages::stage3_cab::{V3MatryoshkaModel, V3TierReadout};
use dnam_engine::stages::stage4_tank::v3_fdn::V3_TANK_MAX_DECLARED_DELAY;
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, ARENA0_CP1_BASE_OFFSET, ARENA0_DIRECT_READOUT_OFFSET,
    ARENA0_STAGE4_TANK_OFFSET, ARENA0_TIER_GAIN_TRIM_OFFSET, ARENA0_UINT_BASE_OFFSET,
    ARENA0_W_CTRL_OFFSET, ARENA1_DIRECT_READOUT_OFFSET, ARENA1_STAGE4_TANK_OFFSET,
    ARENA2_DIRECT_READOUT_OFFSET, ARENA2_FEAT_DET_THRESH_OFFSET, ARENA2_STAGE4_TANK_OFFSET,
    ARENA3_CP1_ULTRA_END, ARENA3_DIRECT_READOUT_OFFSET, ARENA3_STAGE4_TANK_OFFSET,
    ARENA3_UINT_ULTRA_OFFSET, ARENA3_W_CTRL_ULTRA_OFFSET, PIPE_FLAG_V3_MATRYOSHKA,
    STAGE4_LFO_OFFSETS, STAGE4_TAP_GAINS_OFFSETS, has_pipeline_flag, pack_v3_matryoshka_to_arena,
    read_u32_le, unpack_v3_matryoshka_from_arena,
};
#[cfg(test)]
use dnam_engine::stages::weights::{ARENA_ULTRA_BYTES, ARENA0_STAGE3_KAUTZ_OFFSET};

use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::schema::{AuditReport, DnamProjManifest};

use super::{
    audit_phase2_gru_state_saturation, clamp_head_spectral_gain_linear,
    compute_phase2_corner_barrier_loss, read_arena_f32, verify_dynamic_stability_excursion,
    write_arena_f32,
};

const READOUT_BASES: [usize; 4] = [
    ARENA0_DIRECT_READOUT_OFFSET,
    ARENA1_DIRECT_READOUT_OFFSET,
    ARENA2_DIRECT_READOUT_OFFSET,
    ARENA3_DIRECT_READOUT_OFFSET,
];
const READOUT_SIZES: [usize; 4] = [320, 384, 448, 512];
const STAGE4_BASES: [usize; 4] = [
    ARENA0_STAGE4_TANK_OFFSET,
    ARENA1_STAGE4_TANK_OFFSET,
    ARENA2_STAGE4_TANK_OFFSET,
    ARENA3_STAGE4_TANK_OFFSET,
];

#[inline]
fn sanitize_f32_slice(slice: &mut [f32], lo: f32, hi: f32, fallback: f32) -> bool {
    let mut dirty = false;
    for v in slice.iter_mut() {
        if !v.is_finite() {
            *v = fallback;
            dirty = true;
        } else if *v < lo || *v > hi {
            *v = v.clamp(lo, hi);
            dirty = true;
        }
    }
    dirty
}

fn audit_svf_and_tone_svf(
    model: &mut V3MatryoshkaModel,
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    let mut svf_dirty = false;
    let check_svf = |svf: &mut dnam_engine::stages::stage1_svf::V3Stage1SvfParams, d: &mut bool| {
        if !svf.g_min.is_finite() || svf.g_min <= 0.0 || svf.g_min > SVF_W_MAX {
            svf.g_min = if svf.g_min.is_finite() {
                svf.g_min.clamp(1.0e-4, SVF_W_MAX)
            } else {
                0.02
            };
            *d = true;
        }
        if !svf.g_max.is_finite() || svf.g_max < svf.g_min || svf.g_max > SVF_W_MAX {
            svf.g_max = if svf.g_max.is_finite() {
                svf.g_max.clamp(svf.g_min, SVF_W_MAX)
            } else {
                SVF_W_MAX
            };
            *d = true;
        }
        if !svf.base_g.is_finite()
            || svf.base_g < 1.0e-4
            || svf.base_g > dnam_engine::stages::stage1_svf::SVF_W_MAX
        {
            svf.base_g = if svf.base_g.is_finite() {
                svf.base_g
                    .clamp(1.0e-4, dnam_engine::stages::stage1_svf::SVF_W_MAX)
            } else {
                0.02
            };
            *d = true;
        }
        if !svf.damp_k1.is_finite() || svf.damp_k1 < 0.05 || svf.damp_k1 > 4.0 {
            svf.damp_k1 = if svf.damp_k1.is_finite() {
                svf.damp_k1.clamp(0.05, 4.0)
            } else {
                0.707
            };
            *d = true;
        }
        if !svf.damp_k2.is_finite() || svf.damp_k2 < 0.05 || svf.damp_k2 > 4.0 {
            svf.damp_k2 = if svf.damp_k2.is_finite() {
                svf.damp_k2.clamp(0.05, 4.0)
            } else {
                0.707
            };
            *d = true;
        }
        svf.clamp_in_place();
    };

    check_svf(&mut model.svf, &mut svf_dirty);
    if let Some(ref mut cal) = model.rate_1x_cal {
        check_svf(&mut cal.svf, &mut svf_dirty);
    }

    let ts = &mut model.tone_svf;
    if !ts.sample_rate_hz.is_finite() || !(8_000.0..=384_000.0).contains(&ts.sample_rate_hz) {
        ts.sample_rate_hz = 96_000.0;
        svf_dirty = true;
    }
    let max_fc = 0.49 * ts.sample_rate_hz;
    if !ts.base_cutoff_hz.is_finite() || !(10.0..=max_fc).contains(&ts.base_cutoff_hz) {
        ts.base_cutoff_hz = if ts.base_cutoff_hz.is_finite() {
            ts.base_cutoff_hz.clamp(10.0, max_fc)
        } else {
            900.0
        };
        svf_dirty = true;
    }
    if !ts.q_res.is_finite() || !(0.1..=16.0).contains(&ts.q_res) {
        ts.q_res = if ts.q_res.is_finite() {
            ts.q_res.clamp(0.1, 16.0)
        } else {
            0.707
        };
        svf_dirty = true;
    }
    if sanitize_f32_slice(&mut ts.h0_proj_w, -16.0, 16.0, 0.0)
        || !ts.dry_mix.is_finite()
        || !ts.lp_mix.is_finite()
        || !ts.bp_mix.is_finite()
        || !ts.hp_mix.is_finite()
    {
        for f in [
            &mut ts.dry_mix,
            &mut ts.lp_mix,
            &mut ts.bp_mix,
            &mut ts.hp_mix,
            &mut ts.s1_proj_w,
            &mut ts.cab_inject_w,
            &mut ts.b1_inject_w,
        ] {
            if !f.is_finite() {
                *f = 0.0;
            }
        }
        svf_dirty = true;
    }
    if svf_dirty {
        phase1_set.insert("CLAMP-STABILITY".to_string());
        phase3_set.insert("CLAMP-DYNAMIC".to_string());
    }
}

fn audit_gru_blocks(
    blocks: &mut [V3GruBlockCold; NUM_GRU_BLOCKS],
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    let mut dirty = false;
    for (b_idx, blk) in blocks.iter_mut().enumerate() {
        for mat in [&mut blk.w_in_z, &mut blk.w_in_r, &mut blk.w_in_n] {
            for row in mat.iter_mut() {
                dirty |= sanitize_f32_slice(row, -128.0, 128.0, 0.0);
            }
        }
        for mat in [
            &mut blk.w_prev_z,
            &mut blk.w_prev_r,
            &mut blk.w_prev_n,
            &mut blk.w_rec_z,
            &mut blk.w_rec_r,
            &mut blk.w_rec_n,
        ] {
            for row in mat.iter_mut() {
                dirty |= sanitize_f32_slice(row, -128.0, 128.0, 0.0);
            }
        }
        dirty |= sanitize_f32_slice(&mut blk.b_z, -8.0, 8.2, 0.0);
        dirty |= sanitize_f32_slice(&mut blk.b_r, -8.0, 8.2, 0.0);
        dirty |= sanitize_f32_slice(&mut blk.b_n, -3.8, 3.8, 0.0);
        dirty |= sanitize_f32_slice(&mut blk.k_sag, 0.0, 8.0, 0.0);
        dirty |= sanitize_f32_slice(&mut blk.k_quad, -0.45, 0.45, 0.0);
        blk.clamp_in_place();
        dirty |= recovery::ensure_gru_block_zero_input_contraction(b_idx, blk);
    }
    if dirty {
        phase1_set.insert("CLAMP-STABILITY".to_string());
        phase3_set.insert("CLAMP-DYNAMIC".to_string());
    }
}

fn audit_kautz_and_stage4(
    model: &mut V3MatryoshkaModel,
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    let mut stab_dirty = false;
    let mut dyn_dirty = false;
    for &mut (ref mut r, ref mut theta) in &mut model.kautz_poles {
        if !r.is_finite() || !theta.is_finite() {
            *r = 0.0;
            *theta = 0.25;
            stab_dirty = true;
            dyn_dirty = true;
            continue;
        }
        if *r != 0.0 {
            let (_, _, _, _, ok) =
                verify_dynamic_stability_excursion(r.abs(), 0.0, 16.0, 0.0, 0.85, 0.5);
            if !ok || *r < 0.0 || *r > 0.9995 {
                *r = r.clamp(0.0, 0.9995);
                stab_dirty = true;
                dyn_dirty = true;
            }
        }
        if !(0.0..=PI - 1.0e-4).contains(theta) {
            *theta = theta.clamp(0.0, PI - 1.0e-4);
            stab_dirty = true;
        }
    }
    let max_d = V3_TANK_MAX_DECLARED_DELAY;
    for s4 in &mut model.stage4 {
        for d in &mut s4.delay_lengths {
            if !(4..=max_d).contains(d) {
                *d = (*d).clamp(4, max_d);
                stab_dirty = true;
            }
        }
        if !s4.g_loop.is_finite() || s4.g_loop.abs() > 0.9990 {
            s4.g_loop = if s4.g_loop.is_finite() {
                s4.g_loop.clamp(-0.9990, 0.9990)
            } else {
                0.0
            };
            stab_dirty = true;
            dyn_dirty = true;
        }
        if !s4.sigma_knee.is_finite() || s4.sigma_knee <= 0.0 {
            s4.sigma_knee = 1.0;
            stab_dirty = true;
        }
        if !s4.r_dc.is_finite() || !(0.0..=0.99995).contains(&s4.r_dc) {
            s4.r_dc = 0.999_345_5;
            stab_dirty = true;
        }
        for f in [
            &mut s4.base_psi,
            &mut s4.inject_gain,
            &mut s4.dry_mix,
            &mut s4.wet_mix,
        ] {
            if !f.is_finite() {
                *f = 0.0;
                stab_dirty = true;
            }
        }
        for tg in &mut s4.tap_gains {
            if !tg.is_finite() || !(-2.0..=2.0).contains(tg) {
                *tg = if tg.is_finite() {
                    tg.clamp(-2.0, 2.0)
                } else {
                    0.5
                };
                stab_dirty = true;
            }
        }
        if !s4.lfo_rate_hz.is_finite() || !(0.0..=16.0).contains(&s4.lfo_rate_hz) {
            s4.lfo_rate_hz = if s4.lfo_rate_hz.is_finite() {
                s4.lfo_rate_hz.clamp(0.0, 16.0)
            } else {
                0.0
            };
            stab_dirty = true;
            dyn_dirty = true;
        }
        if !s4.lfo_depth_samples.is_finite() || !(0.0..=64.0).contains(&s4.lfo_depth_samples) {
            s4.lfo_depth_samples = if s4.lfo_depth_samples.is_finite() {
                s4.lfo_depth_samples.clamp(0.0, 64.0)
            } else {
                0.0
            };
            stab_dirty = true;
            dyn_dirty = true;
        }
        if !s4.lfo_stereo_phase.is_finite() || !(-PI..=PI).contains(&s4.lfo_stereo_phase) {
            s4.lfo_stereo_phase = if s4.lfo_stereo_phase.is_finite() {
                s4.lfo_stereo_phase.clamp(-PI, PI)
            } else {
                0.0
            };
            stab_dirty = true;
        }
        let barrier = compute_phase2_corner_barrier_loss(
            0.9990,
            f64::from(s4.g_loop.abs().clamp(0.0, 0.9990)),
            0.5,
            4.0,
        );
        if barrier >= 1.0e4 {
            stab_dirty = true;
        }
        s4.clamp_in_place();
    }
    if stab_dirty {
        phase1_set.insert("CLAMP-STABILITY".to_string());
    }
    if dyn_dirty {
        phase3_set.insert("CLAMP-DYNAMIC".to_string());
    }
}

fn audit_readout_and_ctrl(
    readouts: &mut [V3TierReadout; 4],
    ctrl: &mut V3ControlBusParams,
    phase3_set: &mut BTreeSet<String>,
) {
    let mut head_clamped = false;
    for ro in readouts.iter_mut() {
        let mut nf_dirty = false;
        nf_dirty |= sanitize_f32_slice(&mut ro.svf_direct_taps_l, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.svf_direct_taps_r, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.pre_cab_u_gain, -256.0, 256.0, 1.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.pre_cab_hp_gain, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.direct_s2_u_gain, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.direct_s2_hp_gain, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.fir_l, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.fir_r, -256.0, 256.0, 0.0);
        nf_dirty |= sanitize_f32_slice(&mut ro.kautz_b_gamma, -16.0, 16.0, 0.0);
        for c_arr in [&mut ro.kautz_c_l, &mut ro.kautz_c_r] {
            for (re, im) in c_arr.iter_mut() {
                if !re.is_finite() {
                    *re = 0.0;
                    nf_dirty = true;
                }
                if !im.is_finite() {
                    *im = 0.0;
                    nf_dirty = true;
                }
            }
        }
        let mut l2_sq_l = 0.0_f32;
        let mut l2_sq_r = 0.0_f32;
        for l in 0..ro.pre_cab_w[0].len() {
            for ch in 0..2 {
                if !ro.pre_cab_w[ch][l].is_finite() {
                    ro.pre_cab_w[ch][l] = 0.0;
                    nf_dirty = true;
                }
                if !ro.direct_s2_w[ch][l].is_finite() {
                    ro.direct_s2_w[ch][l] = 0.0;
                    nf_dirty = true;
                }
            }
            l2_sq_l += ro.pre_cab_w[0][l] * ro.pre_cab_w[0][l]
                + ro.direct_s2_w[0][l] * ro.direct_s2_w[0][l];
            l2_sq_r += ro.pre_cab_w[1][l] * ro.pre_cab_w[1][l]
                + ro.direct_s2_w[1][l] * ro.direct_s2_w[1][l];
        }
        let rms_gain = (l2_sq_l.max(l2_sq_r) / 64.0).sqrt();
        let (scale, clamped) = clamp_head_spectral_gain_linear(rms_gain);
        if clamped || nf_dirty {
            for ch in 0..2 {
                for l in 0..ro.pre_cab_w[ch].len() {
                    ro.pre_cab_w[ch][l] *= scale;
                    ro.direct_s2_w[ch][l] *= scale;
                }
            }
            head_clamped = true;
        }
    }

    for arr in [
        &mut ctrl.w_env_fc_oct,
        &mut ctrl.w_env_k,
        &mut ctrl.w_tone_fc_oct,
        &mut ctrl.w_tone_k,
        &mut ctrl.w_tank_psi,
    ] {
        head_clamped |= sanitize_f32_slice(arr, -16.0, 16.0, 0.0);
    }
    for eq in &mut ctrl.w_tone_eq {
        head_clamped |= sanitize_f32_slice(eq, -16.0, 16.0, 0.0);
    }
    for blk in &mut ctrl.w_film_gain {
        for lane in blk {
            head_clamped |= sanitize_f32_slice(lane, -8.0, 8.0, 0.0);
        }
    }
    for blk in &mut ctrl.w_film_bn {
        for lane in blk {
            head_clamped |= sanitize_f32_slice(lane, -8.0, 8.0, 0.0);
        }
    }
    if head_clamped {
        phase3_set.insert("CLAMP-HEAD-GAIN".to_string());
    }
}

fn sanitize_raw_float_range(
    arena: &mut [u8],
    range: core::ops::Range<usize>,
    phase3_set: &mut BTreeSet<String>,
) {
    let end = range.end.min(arena.len());
    let mut off = range.start;
    let mut fixed = false;
    while off + 4 <= end {
        let v = read_arena_f32(arena, off);
        if !v.is_finite() {
            write_arena_f32(arena, off, 0.0);
            fixed = true;
        }
        off += 4;
    }
    if fixed {
        phase3_set.insert("CLAMP-HEAD-GAIN".to_string());
    }
}

fn clamp_raw_tier_trims_and_ctrl_ranges(arena: &mut [u8], phase3_set: &mut BTreeSet<String>) {
    if arena.len() >= ARENA0_TIER_GAIN_TRIM_OFFSET + 16 {
        for t in 0..4 {
            let off = ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4;
            let raw = read_arena_f32(arena, off);
            if !raw.is_finite() || !(0.0625..=16.0).contains(&raw) {
                let c = if raw.is_finite() && raw > 0.0 {
                    raw.clamp(0.0625, 16.0)
                } else {
                    1.0
                };
                write_arena_f32(arena, off, c);
                phase3_set.insert("CLAMP-HEAD-GAIN".to_string());
            }
        }
    }
    sanitize_raw_float_range(
        arena,
        ARENA0_UINT_BASE_OFFSET..ARENA0_CP1_BASE_OFFSET,
        phase3_set,
    );
    sanitize_raw_float_range(
        arena,
        ARENA0_CP1_BASE_OFFSET..ARENA0_W_CTRL_OFFSET,
        phase3_set,
    );
    sanitize_raw_float_range(arena, ARENA0_W_CTRL_OFFSET..ARENA_LITE_BYTES, phase3_set);
    if arena.len() >= ARENA3_CP1_ULTRA_END {
        sanitize_raw_float_range(
            arena,
            ARENA3_W_CTRL_ULTRA_OFFSET..ARENA3_UINT_ULTRA_OFFSET,
            phase3_set,
        );
        sanitize_raw_float_range(
            arena,
            ARENA3_UINT_ULTRA_OFFSET..ARENA3_CP1_ULTRA_END,
            phase3_set,
        );
    }
}

fn audit_unflagged_raw_arena(
    arena: &mut [u8],
    clamp_backbone: bool,
    phase1_set: &mut BTreeSet<String>,
    phase3_set: &mut BTreeSet<String>,
) {
    if clamp_backbone {
        for &base in &STAGE3_KAUTZ_BASES {
            if base + 128 > arena.len() {
                continue;
            }
            for m in 0..4 {
                let r = read_arena_f32(arena, base + m * 4);
                let th = read_arena_f32(arena, base + 16 + m * 4);
                if !r.is_finite() || r < 0.0 || r > 0.9995 {
                    let cr = if r.is_finite() {
                        r.clamp(0.0, 0.9995)
                    } else {
                        0.0
                    };
                    write_arena_f32(arena, base + m * 4, cr);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                    phase3_set.insert("CLAMP-DYNAMIC".to_string());
                }
                if !th.is_finite() || !(0.0..=PI - 1.0e-4).contains(&th) {
                    let ct = if th.is_finite() {
                        th.clamp(0.0, PI - 1.0e-4)
                    } else {
                        0.25
                    };
                    write_arena_f32(arena, base + 16 + m * 4, ct);
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
            }
        }
        let max_d = V3_TANK_MAX_DECLARED_DELAY as u32;
        for &base in &STAGE4_BASES {
            if base + 64 > arena.len() {
                continue;
            }
            for l in 0..4 {
                let off = base + l * 4;
                let d = read_u32_le(arena, off);
                if d < 4 || d > max_d {
                    let cd = d.clamp(4, max_d);
                    arena[off..off + 4].copy_from_slice(&cd.to_le_bytes());
                    phase1_set.insert("CLAMP-STABILITY".to_string());
                }
            }
            let g_loop = read_arena_f32(arena, base + 16);
            if !g_loop.is_finite() || g_loop.abs() > 0.9990 {
                let cg = if g_loop.is_finite() {
                    g_loop.clamp(-0.9990, 0.9990)
                } else {
                    0.0
                };
                write_arena_f32(arena, base + 16, cg);
                phase1_set.insert("CLAMP-STABILITY".to_string());
                phase3_set.insert("CLAMP-DYNAMIC".to_string());
            }
        }
    }

    for (&base, &size) in READOUT_BASES.iter().zip(&READOUT_SIZES) {
        if base + size > arena.len() {
            continue;
        }
        let w_floats = (size - 96) / 4;
        let mut l2_sq = 0.0_f32;
        let mut nf = false;
        for i in 0..w_floats {
            let off = base + 96 + i * 4;
            let v = read_arena_f32(arena, off);
            if !v.is_finite() {
                nf = true;
            } else {
                l2_sq += v * v;
            }
        }
        let rms = (l2_sq / 64.0).sqrt();
        let (scale, clamped) = clamp_head_spectral_gain_linear(rms);
        if clamped || nf {
            for i in 0..w_floats {
                let off = base + 96 + i * 4;
                let v = read_arena_f32(arena, off);
                write_arena_f32(arena, off, if v.is_finite() { v * scale } else { 0.0 });
            }
            phase3_set.insert("CLAMP-HEAD-GAIN".to_string());
        }
    }
    clamp_raw_tier_trims_and_ctrl_ranges(arena, phase3_set);
}

fn run_3phase_arena_audit_with_mode(
    manifest: &mut DnamProjManifest,
    arena: &mut [u8],
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

    let is_v3_flagged =
        arena.len() >= ARENA_LITE_BYTES && has_pipeline_flag(arena, PIPE_FLAG_V3_MATRYOSHKA);
    if is_v3_flagged {
        let mut orig_trims = [0u8; 16];
        orig_trims.copy_from_slice(
            &arena[ARENA0_TIER_GAIN_TRIM_OFFSET..ARENA0_TIER_GAIN_TRIM_OFFSET + 16],
        );
        let orig_arena = (!clamp_backbone).then(|| arena.to_vec());
        let mono_out = manifest.base.io_config.output_mode == "mono";
        let mut model = unpack_v3_matryoshka_from_arena(arena, mono_out);
        if clamp_backbone {
            phase2_warns.clear();
            audit_svf_and_tone_svf(&mut model, &mut phase1_set, &mut phase3_set);
            audit_gru_blocks(&mut model.blocks, &mut phase1_set, &mut phase3_set);
            if let Some(ref mut cal) = model.rate_1x_cal {
                audit_gru_blocks(&mut cal.blocks, &mut phase1_set, &mut phase3_set);
            }
            audit_kautz_and_stage4(&mut model, &mut phase1_set, &mut phase3_set);
        }
        audit_readout_and_ctrl(&mut model.readouts, &mut model.ctrl, &mut phase3_set);
        pack_v3_matryoshka_to_arena(&model, arena);
        arena[ARENA0_TIER_GAIN_TRIM_OFFSET..ARENA0_TIER_GAIN_TRIM_OFFSET + 16]
            .copy_from_slice(&orig_trims);
        if let Some(ref orig) = orig_arena {
            let copy_range = |dst: &mut [u8], start: usize, end: usize| {
                let lim = end.min(dst.len()).min(orig.len());
                if start < lim {
                    dst[start..lim].copy_from_slice(&orig[start..lim]);
                }
            };
            copy_range(arena, 96, 216);
            copy_range(arena, 256, 2_368);
            copy_range(arena, 4_608, 5_440);
            copy_range(arena, 10_240, 14_080);
            copy_range(arena, 18_432, 19_456);
            for &k_off in &STAGE3_KAUTZ_BASES {
                copy_range(arena, k_off, k_off + 64);
            }
            for &s4_off in &STAGE4_BASES {
                copy_range(arena, s4_off, s4_off + 64);
            }
            for &tg_off in &STAGE4_TAP_GAINS_OFFSETS {
                copy_range(arena, tg_off, tg_off + 16);
            }
            for &lfo_off in &STAGE4_LFO_OFFSETS {
                copy_range(arena, lfo_off, lfo_off + 12);
            }
            copy_range(
                arena,
                ARENA2_FEAT_DET_THRESH_OFFSET,
                ARENA2_FEAT_DET_THRESH_OFFSET + 4,
            );
        }
        clamp_raw_tier_trims_and_ctrl_ranges(arena, &mut phase3_set);
        if clamp_backbone {
            let (occ, safe_scale, locked) = audit_phase2_gru_state_saturation(arena, 3.0);
            if locked {
                phase2_warns.push(format!(
                    "[CLAMP-RAIL] GRU corner drive occupancy {:.1}% > 95.0%; recommended drive scale {:.3}",
                    occ * 100.0,
                    safe_scale
                ));
            }
        }
    } else {
        if clamp_backbone {
            phase2_warns.clear();
        }
        audit_unflagged_raw_arena(arena, clamp_backbone, &mut phase1_set, &mut phase3_set);
    }

    zero_arena_b_head_dc_bias(arena);
    phase3_set.insert("NUMERIC-ENVELOPE".to_string());

    manifest.audit_report = Some(AuditReport {
        phase1_clamps: phase1_set.into_iter().collect(),
        phase2_rail_warnings: phase2_warns,
        phase3_clamps: phase3_set.into_iter().collect(),
        prescriptions,
    });
}

/// Runs the complete `v3` Phase-1/2/3 stability, corner-barrier, GRU state saturation, dynamic
/// excursion, and `+6.0 dB` head-gain audit in-place over a `v3` `arena` (`4,608..=28,672 B`),
/// updating `manifest.audit_report` (`ref[train.audit.phase1.clamp-stability]`,
/// `ref[train.audit.phase2.corner-barrier]`, `ref[train.audit.phase2.clamp-rail]`,
/// `ref[train.audit.phase3.clamp-dynamic]`, `ref[train.audit.phase3.clamp-head-gain]`).
pub fn run_full_3phase_arena_audit(manifest: &mut DnamProjManifest, arena: &mut [u8]) {
    run_3phase_arena_audit_with_mode(manifest, arena, true);
}

/// Runs Phase-3 `[CLAMP-HEAD-GAIN]` and `[NUMERIC-ENVELOPE]` over a `v3` `arena` without mutating
/// the frozen `StarCenter` Stage I/II/III/IV backbone (`Path 1` fast-path / `preserve_backbone()`).
pub fn run_head_only_3phase_arena_audit(manifest: &mut DnamProjManifest, arena: &mut [u8]) {
    run_3phase_arena_audit_with_mode(manifest, arena, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_only_preserves_v3_backbone_and_phase2_warnings() {
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let mut model = V3MatryoshkaModel::identity(true, false);
        model.kautz_poles[0] = (0.985, 0.22);
        model.readouts[0].pre_cab_w[0][0] = 20.0;
        model.readouts[3].direct_s2_w[0][0] = f32::NAN;
        pack_v3_matryoshka_to_arena(&model, &mut arena);
        write_arena_f32(&mut arena, ARENA0_STAGE3_KAUTZ_OFFSET, 1.05); // Deliberately out-of-bounds raw backbone

        let mut manifest = DnamProjManifest::default_reference();
        manifest.audit_report = Some(AuditReport {
            phase1_clamps: vec!["CLAMP-RANGE".to_string()],
            phase2_rail_warnings: vec!["prior-rail-note".to_string()],
            phase3_clamps: vec![],
            prescriptions: vec!["keep-me".to_string()],
        });

        run_head_only_3phase_arena_audit(&mut manifest, &mut arena);
        let unpacked = unpack_v3_matryoshka_from_arena(&arena, false);
        assert!(unpacked.readouts[0].pre_cab_w[0][0] < 20.0);
        assert!(unpacked.readouts[3].direct_s2_w[0][0].is_finite());

        let ar = manifest.audit_report.unwrap();
        assert_eq!(ar.phase2_rail_warnings, vec!["prior-rail-note".to_string()]);
        assert_eq!(ar.prescriptions, vec!["keep-me".to_string()]);
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-HEAD-GAIN"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "NUMERIC-ENVELOPE"));
    }

    #[test]
    fn test_full_v3_arena_audit_clamps_nonfinite_poles_delays_and_trims() {
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let mut model = V3MatryoshkaModel::identity(true, false);
        model.kautz_poles[0] = (1.03, 0.25);
        model.kautz_poles[1] = (f32::NAN, 0.5);
        model.stage4[0].delay_lengths[0] = 2;
        model.stage4[3].delay_lengths[1] = 60_000;
        model.stage4[0].g_loop = -1.25;
        model.blocks[0].b_n[0] = 9.0;
        pack_v3_matryoshka_to_arena(&model, &mut arena);
        write_arena_f32(&mut arena, ARENA0_TIER_GAIN_TRIM_OFFSET, 0.92);
        write_arena_f32(&mut arena, ARENA0_TIER_GAIN_TRIM_OFFSET + 4, 25.0);

        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut arena);

        let out = unpack_v3_matryoshka_from_arena(&arena, false);
        assert!(out.kautz_poles[0].0 <= 0.9995);
        assert_eq!(out.kautz_poles[1].0, 0.0);
        assert!(out.stage4[0].delay_lengths[0] >= 4);
        assert!(out.stage4[3].delay_lengths[1] <= V3_TANK_MAX_DECLARED_DELAY);
        assert!(out.stage4[0].g_loop.abs() <= 0.9990);
        assert!(out.blocks[0].b_n[0] <= 3.8);
        assert!((read_arena_f32(&arena, ARENA0_TIER_GAIN_TRIM_OFFSET) - 0.92).abs() < 1e-6);
        assert!(read_arena_f32(&arena, ARENA0_TIER_GAIN_TRIM_OFFSET + 4) <= 16.0);

        let ar = manifest.audit_report.unwrap();
        assert!(ar.phase1_clamps.iter().any(|t| t == "CLAMP-STABILITY"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-DYNAMIC"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-HEAD-GAIN"));
    }

    #[test]
    fn test_full_v3_arena_sanitizes_svf_lfo_ctrl_and_unflagged_raw_arenas() {
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let mut model = V3MatryoshkaModel::identity(true, false);
        model.svf.g_min = f32::NAN;
        model.svf.g_max = -1.0;
        model.svf.base_g = f32::INFINITY;
        model.svf.damp_k1 = f32::NAN;
        model.svf.damp_k2 = 9.0;
        model.tone_svf.sample_rate_hz = 100.0;
        model.tone_svf.base_cutoff_hz = f32::NAN;
        model.tone_svf.q_res = f32::NAN;
        model.tone_svf.dry_mix = f32::NAN;
        model.tone_svf.h0_proj_w = [f32::NAN, 40.0, 0.0, 0.0];
        let mut cal = dnam_engine::stages::stage3_cab::V3Rate1xCalibration {
            svf: model.svf,
            feat_params: model.feat_params,
            blocks: model.blocks,
        };
        cal.svf.g_min = 99.0;
        cal.svf.base_g = 99.0;
        cal.svf.damp_k1 = 0.0;
        cal.svf.damp_k2 = f32::NAN;
        cal.blocks[0].w_in_n[0][0] = f32::NAN;
        model.rate_1x_cal = Some(cal);
        model.kautz_poles[0] = (0.95, 10.0);
        model.stage4[0].g_loop = f32::NAN;
        model.stage4[0].sigma_knee = -1.0;
        model.stage4[0].r_dc = 2.0;
        model.stage4[0].wet_mix = f32::NAN;
        model.stage4[0].tap_gains = [f32::NAN, 5.0, -5.0, 0.5];
        model.stage4[0].lfo_rate_hz = f32::NAN;
        model.stage4[0].lfo_depth_samples = 99.0;
        model.stage4[0].lfo_stereo_phase = f32::NAN;
        model.stage4[1].lfo_rate_hz = 25.0;
        model.stage4[1].lfo_depth_samples = f32::NAN;
        model.stage4[1].lfo_stereo_phase = 5.0;
        model.ctrl.w_env_fc_oct[0] = f32::NAN;
        model.ctrl.w_film_gain[0][0][0] = 25.0;
        pack_v3_matryoshka_to_arena(&model, &mut arena);
        write_arena_f32(&mut arena, ARENA0_TIER_GAIN_TRIM_OFFSET, f32::NAN);
        write_arena_f32(&mut arena, ARENA0_UINT_BASE_OFFSET, f32::NAN);
        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut arena);
        let out = unpack_v3_matryoshka_from_arena(&arena, false);
        assert!(out.svf.g_min.is_finite() && out.svf.base_g.is_finite());
        assert!(out.tone_svf.base_cutoff_hz.is_finite() && out.tone_svf.q_res.is_finite());
        assert!(out.stage4[0].lfo_rate_hz.is_finite() && out.stage4[0].lfo_depth_samples <= 64.0);
        assert!(out.stage4[1].lfo_rate_hz <= 16.0 && out.stage4[1].lfo_stereo_phase.abs() <= PI);
        assert!(read_arena_f32(&arena, ARENA0_UINT_BASE_OFFSET).is_finite());

        // Unflagged raw arena audit paths (`clamp_backbone = true` and `false`).
        let mut raw_arena = vec![0u8; ARENA_ULTRA_BYTES];
        write_arena_f32(&mut raw_arena, STAGE3_KAUTZ_BASES[0], f32::NAN);
        write_arena_f32(&mut raw_arena, STAGE3_KAUTZ_BASES[0] + 16, 10.0);
        write_arena_f32(&mut raw_arena, STAGE3_KAUTZ_BASES[1], 1.5);
        write_arena_f32(&mut raw_arena, STAGE3_KAUTZ_BASES[1] + 16, f32::NAN);
        write_arena_f32(&mut raw_arena, STAGE4_BASES[0] + 16, f32::NAN);
        write_arena_f32(&mut raw_arena, STAGE4_BASES[1] + 16, 2.5);
        write_arena_f32(&mut raw_arena, READOUT_BASES[0] + 96, 30.0);
        run_head_only_3phase_arena_audit(&mut manifest, &mut raw_arena);
        run_full_3phase_arena_audit(&mut manifest, &mut raw_arena);
        assert!(read_arena_f32(&raw_arena, STAGE4_BASES[0] + 16).is_finite());
        assert!(read_arena_f32(&raw_arena, READOUT_BASES[0] + 96) < 30.0);
    }
}
