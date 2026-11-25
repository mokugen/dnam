#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! 100% data-driven universal spectral control bus (`TPT-SVF` partition-of-unity `LP`/`BP`/`HP` +
//! `tone_dir` across dedicated `Arena 0` `PolyADAA1` lanes into `pre_conv` with zero unmodulated
//! bypass) for parametric EQ, tone, and voicing controls discovered strictly from off-center
//! `STMC` wet audio divergence (`REQ-1.6`, `REQ-1.9`).

use std::collections::BTreeMap;

use dnam_engine::schema::DnamError;
use dnam_train::losses::{Welch33TargetCache, canonical_ltas_33_centers_hz};
use dnam_train::{
    CoreQuadF64, InterBlockTileF64, NUM_CORE_QUADS_ULTRA, Pass3VarProOptions, Pass3VarProReport,
    Pass5EmitConfig, SonaModelF64, StereoTrainingPairs, TrainRunner, encode_model_to_dnam_bin,
    run_pass3_varpro_readout_with_options,
};
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SONA_INTERBLOCK_TILES, SonaVm, Tier};
use zerocopy::FromBytes;

use crate::proj::STMC_FLAG_STEREO;
use crate::proj::directory::StmcHeader128BWire;
use crate::proj::schema::DnamProjManifest;
use crate::solver::{canonical_besr_warmup, decode_coreset_lr};

const MIN_PIVOT_HZ: f64 = 300.0;
const MAX_PIVOT_HZ: f64 = 6_500.0;
const MIN_SPECTRAL_SHAPE_SPREAD_DB: f64 = 0.75;
const DEDICATED_ADAA_DRIVE: f64 = 1.0 / 32.0;
const DEDICATED_PRE_DIR_WEIGHT: f64 = -24.0;
const DEDICATED_PRE_BAND_WEIGHT: f64 = 48.0;
const SPECTRAL_BUS_STAR_ESR_MAX_DB: f64 = -20.0;
const PURE_SPECTRAL_BUS_SUFFICIENT_ESR_DB: f64 = -48.0;
const HYBRID_OS2X_TRIGGER_ESR_DB: f64 = -38.0;
const HYBRID_OS2X_MIN_IMPROVE_DB: f64 = 2.0;
const HYBRID_OS2X_MAX_LITE_REGRESS_DB: f64 = 0.5;
const PURE_BUS_PARSIMONY_MARGIN_DB: f64 = 1.0;

const _: () = assert!(MIN_PIVOT_HZ < MAX_PIVOT_HZ);
const _: () = assert!(
    DEDICATED_ADAA_DRIVE > 0.0
        && DEDICATED_ADAA_DRIVE <= 0.125
        && DEDICATED_PRE_DIR_WEIGHT < 0.0
        && DEDICATED_PRE_DIR_WEIGHT >= -64.0
        && DEDICATED_PRE_BAND_WEIGHT == -2.0 * DEDICATED_PRE_DIR_WEIGHT
        && DEDICATED_PRE_BAND_WEIGHT <= 64.0
        && PURE_SPECTRAL_BUS_SUFFICIENT_ESR_DB < HYBRID_OS2X_TRIGGER_ESR_DB
        && HYBRID_OS2X_TRIGGER_ESR_DB < -34.0
        && -34.0 < SPECTRAL_BUS_STAR_ESR_MAX_DB
        && HYBRID_OS2X_MIN_IMPROVE_DB > 0.0
        && HYBRID_OS2X_MAX_LITE_REGRESS_DB > 0.0
        && PURE_BUS_PARSIMONY_MARGIN_DB > 0.0
);

#[inline]
fn map_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

fn mid_channel_f64(lr: &[[f32; 2]]) -> Vec<f64> {
    lr.iter()
        .map(|&[l, r]| 0.5 * (f64::from(l) + f64::from(r)))
        .collect()
}

fn eval_slice_divergence(
    slice_mid: &[f64],
    star_db: &[f64; 33],
    peak_star_db: f64,
    centers: &[f64; 33],
) -> Option<(f64, f64, f64)> {
    let n = slice_mid.len();
    if n < 512 {
        return None;
    }
    let warmup = canonical_besr_warmup(n).min(n / 4);
    let cache = Welch33TargetCache::new(slice_mid, warmup, 48_000.0).ok()?;
    let mut act_bands = Vec::with_capacity(33);
    for b in 0..33 {
        if centers[b] >= 60.0 && centers[b] <= 14_000.0 && star_db[b] > peak_star_db - 45.0 {
            let db = 10.0 * cache.tgt_band_p[b].max(1.0e-24).log10();
            act_bands.push((b, db - star_db[b]));
        }
    }
    if act_bands.len() < 4 {
        return None;
    }
    let mean_delta = act_bands.iter().map(|&(_, d)| d).sum::<f64>() / (act_bands.len() as f64);
    let residuals: Vec<(usize, f64)> = act_bands
        .iter()
        .map(|&(b, d)| (b, d - mean_delta))
        .collect();
    let rms_db =
        (residuals.iter().map(|&(_, r)| r * r).sum::<f64>() / (residuals.len() as f64)).sqrt();
    if rms_db < MIN_SPECTRAL_SHAPE_SPREAD_DB {
        return Some((rms_db, 0.0, 0.0));
    }
    let (mut sum_w, mut sum_w_logf) = (0.0_f64, 0.0_f64);
    for pair in residuals.windows(2) {
        let (b0, r0) = pair[0];
        let (b1, r1) = pair[1];
        let w = (r1 - r0).powi(2);
        sum_w += w;
        sum_w_logf += w * 0.5 * (centers[b0].ln() + centers[b1].ln());
    }
    Some((rms_db, sum_w, sum_w_logf))
}

fn compute_star_slice_db(
    star_mid: &[f64],
    range: core::ops::Range<usize>,
) -> Option<([f64; 33], f64)> {
    let slice = star_mid.get(range)?;
    let n = slice.len();
    if n < 512 {
        return None;
    }
    let warmup = canonical_besr_warmup(n).min(n / 4);
    let cache = Welch33TargetCache::new(slice, warmup, 48_000.0).ok()?;
    let db: [f64; 33] = core::array::from_fn(|b| 10.0 * cache.tgt_band_p[b].max(1.0e-24).log10());
    let peak = db.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((db, peak))
}

/// Detects whether enabled non-`StarCenter` stems in `stmc_by_id` exhibit statistically
/// significant gain-normalized spectral-shape divergence (`>= 0.75 dB` RMS across active Welch-33
/// bands) vs `star_wet_lr`, returning the slope-weighted log-frequency pivot in `[300, 6500] Hz`.
/// Strictly audio-driven with zero fixture, device, or control-name heuristics (`REQ-1.6`, `REQ-1.9`).
#[must_use]
pub(crate) fn detect_offcenter_spectral_pivot_hz(
    manifest: &DnamProjManifest,
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    star_wet_lr: &[[f32; 2]],
    active_frames: usize,
) -> Option<f64> {
    if manifest.base.controls_schema.is_empty() || star_wet_lr.len() < 1_024 {
        return None;
    }
    let star_n = if active_frames == 0 {
        star_wet_lr.len()
    } else {
        active_frames.min(star_wet_lr.len())
    };
    if star_n < 1_024 {
        return None;
    }
    let star_mid = mid_channel_f64(&star_wet_lr[..star_n]);
    let centers = canonical_ltas_33_centers_hz();
    type StarSliceLtas = ([f64; 33], f64);
    let mut star_slice_cache: BTreeMap<(usize, usize), Option<StarSliceLtas>> = BTreeMap::new();

    let mut max_rms_db = 0.0_f64;
    let (mut total_w, mut total_w_logf) = (0.0_f64, 0.0_f64);
    for stem in manifest
        .stems_registry
        .iter()
        .filter(|s| s.enabled && !s.is_star_center)
    {
        let Some(stmc_bytes) = stmc_by_id.get(&stem.chunk_id) else {
            continue;
        };
        if stmc_bytes.len() <= 4_800 {
            continue;
        }
        let flags =
            StmcHeader128BWire::ref_from_prefix(stmc_bytes).map_or(0, |(h, _)| h.flags.get());
        let ch = if (flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let wet_lr = decode_coreset_lr(&stmc_bytes[4_800..], ch);
        let n_s = if active_frames == 0 {
            wet_lr.len()
        } else {
            active_frames.min(wet_lr.len())
        }
        .min(star_n);
        if n_s < 512 {
            continue;
        }
        let stem_mid = mid_channel_f64(&wet_lr[..n_s]);
        let mut ranges = Vec::with_capacity(3);
        ranges.push(0..n_s);
        if n_s >= 12_288 {
            let third = n_s / 3;
            ranges.push(0..third);
            ranges.push((n_s - third)..n_s);
        }
        for r in ranges {
            let star_slice = star_slice_cache
                .entry((r.start, r.end))
                .or_insert_with(|| compute_star_slice_db(&star_mid, r.clone()));
            let Some((star_db, peak_star_db)) = star_slice.as_ref() else {
                continue;
            };
            if let Some((rms_db, sw, sw_logf)) =
                eval_slice_divergence(&stem_mid[r], star_db, *peak_star_db, &centers)
            {
                max_rms_db = max_rms_db.max(rms_db);
                total_w += sw;
                total_w_logf += sw_logf;
            }
        }
    }

    if max_rms_db < MIN_SPECTRAL_SHAPE_SPREAD_DB || total_w <= 1.0e-12 {
        return None;
    }
    let pivot = (total_w_logf / total_w).exp();
    pivot
        .is_finite()
        .then(|| pivot.clamp(MIN_PIVOT_HZ, MAX_PIVOT_HZ))
}

fn eval_vm_lite_ultra_esr_db(
    model: &SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    warmup: usize,
) -> Result<[f64; 2], DnamError> {
    let (in_l, in_r, tgt_l, tgt_r) = (pairs.in_l, pairs.in_r, pairs.target_l, pairs.target_r);
    let n = in_l.len().min(in_r.len()).min(tgt_l.len()).min(tgt_r.len());
    let bin = encode_model_to_dnam_bin(
        model,
        Pass5EmitConfig {
            calibrate_lufs_trims: false,
            ..Pass5EmitConfig::default()
        },
    )
    .map_err(map_err)?;
    let in_l32: Vec<f32> = in_l[..n].iter().map(|&v| v as f32).collect();
    let in_r32: Vec<f32> = in_r[..n].iter().map(|&v| v as f32).collect();
    let mut esrs = [0.0_f64; 2];
    for (idx, tier) in [Tier::Lite, Tier::Ultra].into_iter().enumerate() {
        let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(
            &bin,
            tier,
            model.sample_rate_hz as f32,
        )
        .map_err(map_err)?;
        let (mut ol, mut or) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        vm.process_block(&in_l32, &in_r32, &mut ol, &mut or)
            .map_err(map_err)?;
        let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
        for t in warmup..n {
            let dl = tgt_l[t] - f64::from(ol[t]);
            let dr = tgt_r[t] - f64::from(or[t]);
            sig_sq += tgt_l[t] * tgt_l[t] + tgt_r[t] * tgt_r[t];
            err_sq += dl * dl + dr * dr;
        }
        esrs[idx] = if sig_sq <= 1.0e-18 {
            if err_sq <= 1.0e-18 { -120.0 } else { 0.0 }
        } else {
            10.0 * (err_sq / sig_sq).max(1.0e-16).log10()
        };
    }
    Ok(esrs)
}

fn select_quietest_lane_in_quad(model: &SonaModelF64, q: usize) -> usize {
    let mut best_l = 3_usize;
    let mut best_score = f64::INFINITY;
    for l in (0..4).rev() {
        let lane_idx = q * 4 + l;
        let w_sum: f64 = model.w_ro[lane_idx].iter().map(|v| v.abs()).sum();
        let score = model.core_quads[q].adaa_drive[l].abs()
            + model.core_quads[q].wavefolder_mix[l].abs()
            + model.core_quads[q].lru_kappa[l].abs()
            + w_sum;
        if score < best_score {
            best_score = score;
            best_l = l;
        }
    }
    best_l
}

fn configure_spectral_bus_lanes(
    cand: &mut SonaModelF64,
    pivot_hz: f64,
    clear_non_bus_core: bool,
) -> u64 {
    cand.pre_conv_u_gain = [0.0, 0.0];
    cand.pre_conv_hp_gain = [0.0, 0.0];
    cand.direct_s2_u_gain = [0.0, 0.0];
    cand.direct_s2_hp_gain = [0.0, 0.0];
    cand.active_fir_taps = cand.active_fir_taps.max(Tier::Ultra.early_fir_taps());

    if clear_non_bus_core {
        cand.core_quads.fill(CoreQuadF64::default());
        cand.tiles.fill(InterBlockTileF64::default());
        cand.op_p_to_core = [[0.0; 4]; NUM_CORE_QUADS_ULTRA];
        cand.phasor_to_core = [[0.0; 4]; NUM_CORE_QUADS_ULTRA];
        cand.w_ro = [[0.0; 12]; 40];
    } else {
        for row in &mut cand.w_ro {
            row[0..6].fill(0.0);
        }
    }

    let mut lane_mask = if clear_non_bus_core { u64::MAX } else { 0_u64 };
    let svf_g = (core::f64::consts::PI * pivot_hz.clamp(MIN_PIVOT_HZ, MAX_PIVOT_HZ)
        / cand.sample_rate_hz.max(8_000.0))
    .tan()
    .clamp(1.0e-4, 4.0);

    for q in 0..4 {
        let l_q = if clear_non_bus_core {
            3
        } else {
            select_quietest_lane_in_quad(cand, q)
        };
        lane_mask |= 1_u64 << (q * 4 + l_q);

        if !clear_non_bus_core {
            for (tile_idx, &(src_q, dst_q)) in SONA_INTERBLOCK_TILES.iter().enumerate() {
                if tile_idx >= cand.tiles.len() {
                    break;
                }
                if src_q == q {
                    cand.tiles[tile_idx].w_odd_4x4[l_q].fill(0.0);
                    cand.tiles[tile_idx].w_even_4x4[l_q].fill(0.0);
                    cand.tiles[tile_idx].w_lru_4x4[l_q].fill(0.0);
                }
                if dst_q == q {
                    for row in 0..4 {
                        cand.tiles[tile_idx].w_odd_4x4[row][l_q] = 0.0;
                        cand.tiles[tile_idx].w_even_4x4[row][l_q] = 0.0;
                        cand.tiles[tile_idx].w_lru_4x4[row][l_q] = 0.0;
                    }
                }
            }
            cand.op_p_to_core[q][l_q] = 0.0;
            cand.phasor_to_core[q][l_q] = 0.0;
        }

        let cq = &mut cand.core_quads[q];
        cq.svf_g[l_q] = svf_g;
        cq.svf_k[l_q] = core::f64::consts::SQRT_2;
        cq.svf_k_nl[l_q] = 0.0;
        cq.sag_depth[l_q] = 0.0;
        // Keep ParaLRU strictly zero (`v_eff == +0.0`) on dedicated linear EQ bus lanes so
        // additive `stage_targets[6..=9]` bias never amplifies envelope DC through `w_ro = ±48`.
        cq.lru_alpha_rel[l_q] = 0.0;
        cq.lru_alpha_atk[l_q] = 1.0;
        cq.lru_kappa[l_q] = 0.0;
        cq.lru_bias_drive[l_q] = 0.0;
        cq.wavefolder_mix[l_q] = 0.0;
        cq.glu_mix[l_q] = 0.0;
        cq.lru_duhem_gamma[l_q] = 0.0;
        cq.lru_signed_mix[l_q] = 0.0;
        cq.lru_stereo_link[l_q] = 0.0;
        cq.rect_sign[l_q] = 1.0;
        cq.in_mid_weight[l_q] = 1.0;
        cq.in_side_weight[l_q] = 1.0;
        cq.adaa_drive[l_q] = DEDICATED_ADAA_DRIVE;
        cq.tone_lp[l_q] = if q == 1 { 1.0 } else { 0.0 };
        cq.tone_bp[l_q] = if q == 2 {
            core::f64::consts::SQRT_2
        } else {
            0.0
        };
        cq.tone_hp[l_q] = if q == 3 { 1.0 } else { 0.0 };
        cq.tone_ap[l_q] = 0.0;
        cq.tone_dir[l_q] = if q == 0 { 1.0 } else { 0.0 };

        cand.w_ro[q * 4 + l_q] = [0.0; 12];
        let w_pre = if q == 0 {
            DEDICATED_PRE_DIR_WEIGHT
        } else {
            DEDICATED_PRE_BAND_WEIGHT
        };
        cand.w_ro[q * 4 + l_q][0] = w_pre;
        cand.w_ro[q * 4 + l_q][1] = w_pre;
    }

    lane_mask
}

fn solve_spectral_bus_candidate(
    runner: &TrainRunner,
    base: &SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    warmup: usize,
    pivot_hz: f64,
    clear_non_bus_core: bool,
    os2x: bool,
) -> Result<(SonaModelF64, [f64; 2]), DnamError> {
    let mut cand = base.clone();
    cand.oversample_upper_tiers_2x = os2x;
    let lane_mask = configure_spectral_bus_lanes(&mut cand, pivot_hz, clear_non_bus_core);
    cand.tier_overrides = [None, None, None, None];
    cand.prefix_override_needed = [false; 4];

    let _rep: Pass3VarProReport = run_pass3_varpro_readout_with_options(
        runner,
        &mut cand,
        pairs,
        warmup,
        Pass3VarProOptions {
            suppress_direct_s2_linear: true,
            suppress_post_core_lane_mask: lane_mask,
        },
    )
    .map_err(map_err)?;

    let esrs = eval_vm_lite_ultra_esr_db(&cand, pairs, warmup)?;
    Ok((cand, esrs))
}

#[inline]
fn esrs_below(esrs: [f64; 2], max_db: f64) -> bool {
    esrs[0].is_finite() && esrs[1].is_finite() && esrs[0] <= max_db && esrs[1] <= max_db
}

#[inline]
fn should_adopt_os2x(cand_2x: [f64; 2], cur: [f64; 2]) -> bool {
    cand_2x[1].is_finite()
        && cand_2x[0].is_finite()
        && (!cur[1].is_finite()
            || (cand_2x[1] < cur[1] - HYBRID_OS2X_MIN_IMPROVE_DB
                && cand_2x[0] <= cur[0] + HYBRID_OS2X_MAX_LITE_REGRESS_DB))
}

/// Configures a 4-block `Arena 0` partition-of-unity `TPT-SVF` + `tone_dir` spectral control bus
/// routed strictly into `pre_conv` (`w_ro[l][0..2]`) using a two-stage parsimonious solver:
/// Stage 1 solves a clean `1×` pure `pre_conv` bus; Stage 2 falls back to hybrid nonlinear +
/// spectral bus only when heavy nonlinearity exceeds pure bus (`REQ-1.6`, `REQ-1.9`).
pub(super) fn try_apply_spectral_control_bus_pass3(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    warmup: usize,
    pivot_hz: f64,
) -> Result<bool, DnamError> {
    let (cand1, c1) =
        solve_spectral_bus_candidate(runner, model, pairs, warmup, pivot_hz, true, false)?;
    if esrs_below(c1, PURE_SPECTRAL_BUS_SUFFICIENT_ESR_DB) {
        *model = cand1;
        return Ok(true);
    }

    let (cand2_1x, c2_1x) =
        solve_spectral_bus_candidate(runner, model, pairs, warmup, pivot_hz, false, false)?;
    let (mut best_hybrid, mut best_hybrid_esr) = (cand2_1x, c2_1x);
    if model.oversample_upper_tiers_2x
        && (!c2_1x[1].is_finite() || c2_1x[1] > HYBRID_OS2X_TRIGGER_ESR_DB)
    {
        let (cand2_2x, c2_2x) =
            solve_spectral_bus_candidate(runner, model, pairs, warmup, pivot_hz, false, true)?;
        if should_adopt_os2x(c2_2x, best_hybrid_esr) {
            best_hybrid = cand2_2x;
            best_hybrid_esr = c2_2x;
        }
    }

    let prefer_pure = c1[0].is_finite()
        && c1[1].is_finite()
        && (!esrs_below(best_hybrid_esr, SPECTRAL_BUS_STAR_ESR_MAX_DB)
            || (c1[1] <= best_hybrid_esr[1] + PURE_BUS_PARSIMONY_MARGIN_DB
                && c1[0] <= best_hybrid_esr[0] + PURE_BUS_PARSIMONY_MARGIN_DB));
    let (chosen, chosen_esr) = if prefer_pure {
        (cand1, c1)
    } else {
        (best_hybrid, best_hybrid_esr)
    };

    if esrs_below(chosen_esr, SPECTRAL_BUS_STAR_ESR_MAX_DB) {
        *model = chosen;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{
        evaluate_besr_from_band_energies, evaluate_ltas_33band_report,
        try_split_bezout_3band_energies,
    };
    use crate::proj::directory::{StmcTrajectoryWindow, build_proj_stmc_capsule};
    use crate::proj::schema::{StemProvenance, StemRegistryEntry};
    use crate::proj::{STMC_FLAG_ENABLED, STMC_FLAG_STAR_CENTER, STMC_FLAG_STEREO};
    use dnam_train::control::{
        StageTargetBounds, StageTargetSolveConfig, StageTargetSolveInput,
        solve_point_stage_targets_adjoint,
    };
    use dnam_train::{
        CompositeLossConfig, CompositeLossWeights, SsBroydenConfig, run_pass1_probe,
        run_pass3_varpro_readout,
    };
    use dnam_vm::control::STAGE_TARGET_SLOTS;
    use std::f64::consts::TAU;
    use zerocopy::IntoBytes;

    fn make_stem_entry(id: &str, chunk_id: u32, is_star: bool) -> StemRegistryEntry {
        StemRegistryEntry {
            stem_id: id.to_string(),
            chunk_id,
            enabled: true,
            is_star_center: is_star,
            controls_meta: BTreeMap::new(),
            provenance: StemProvenance {
                input_pcm_sha256: "0".repeat(64),
                target_pcm_sha256: "1".repeat(64),
                num_frames: 4_096,
            },
            user_metadata: BTreeMap::new(),
            extracted_physics: None,
        }
    }

    fn make_stmc_payload(chunk_id: u16, flags: u16, wet_lr: &[[f32; 2]]) -> Vec<u8> {
        let frames = 96_000_usize;
        let mut raw = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let [l, r] = wet_lr[i % wet_lr.len()];
            raw.push(l);
            raw.push(r);
        }
        let win = [StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 1,
            active_opt: chunk_id as u8,
            u_phys: f32::from(chunk_id),
            window_weight: 1.0,
            emr_velocity: 0.0,
        }];
        build_proj_stmc_capsule(
            chunk_id,
            flags,
            1.0,
            0.0,
            &[0; 88],
            0,
            &win,
            &[0; 4_096],
            raw.as_bytes(),
        )
        .expect("build test stmc")
    }

    #[test]
    fn test_detect_offcenter_spectral_pivot_and_pass3_spectral_bus() {
        let n = 4_096_usize;
        let sr = 48_000.0_f64;
        let centers = canonical_ltas_33_centers_hz();
        let dry_lr: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                let mut s = 0.0_f64;
                for (k, &fc) in centers.iter().enumerate() {
                    s += (TAU * fc * t + 0.31 * (k as f64)).sin();
                }
                let x = (0.045 * s) as f32;
                [x, x]
            })
            .collect();
        let star_wet: Vec<[f32; 2]> = dry_lr.iter().map(|&[l, r]| [0.8 * l, 0.8 * r]).collect();
        let gain_wet: Vec<[f32; 2]> = dry_lr.iter().map(|&[l, r]| [0.35 * l, 0.35 * r]).collect();

        // 1-pole lowpass around ~1.2 kHz creates significant spectral-shape divergence vs flat.
        let alpha = (-TAU * 1_200.0 / sr).exp() as f32;
        let mut lp_state = 0.0_f32;
        let lp_wet: Vec<[f32; 2]> = dry_lr
            .iter()
            .map(|&[l, _]| {
                lp_state = (1.0 - alpha) * 0.8 * l + alpha * lp_state;
                [lp_state, lp_state]
            })
            .collect();

        let mut manifest = DnamProjManifest::default_reference();
        let sw_def: dnam_engine::schema::ControlSchemaDef =
            serde_json::from_value(serde_json::json!({
                "id": "voicing",
                "name": "Voicing",
                "type": "switch",
                "declared_options": ["Flat", "Dark"],
                "active_options": ["Flat", "Dark"],
                "default": "Flat"
            }))
            .unwrap();
        manifest.base.controls_schema.push(sw_def);
        manifest.stems_registry = vec![
            make_stem_entry("star", 0, true),
            make_stem_entry("gain_only", 1, false),
        ];
        let mut stmc_by_id = BTreeMap::new();
        stmc_by_id.insert(
            0,
            make_stmc_payload(
                0,
                STMC_FLAG_ENABLED | STMC_FLAG_STAR_CENTER | STMC_FLAG_STEREO,
                &star_wet,
            ),
        );
        stmc_by_id.insert(
            1,
            make_stmc_payload(1, STMC_FLAG_ENABLED | STMC_FLAG_STEREO, &gain_wet),
        );
        assert_eq!(
            detect_offcenter_spectral_pivot_hz(&manifest, &stmc_by_id, &star_wet, n),
            None,
            "pure broadband gain change must not trigger spectral pivot"
        );

        stmc_by_id.insert(
            1,
            make_stmc_payload(1, STMC_FLAG_ENABLED | STMC_FLAG_STEREO, &lp_wet),
        );
        let pivot = detect_offcenter_spectral_pivot_hz(&manifest, &stmc_by_id, &star_wet, n)
            .expect("1-pole LP voicing shift must yield spectral pivot");
        assert!(
            (MIN_PIVOT_HZ..=MAX_PIVOT_HZ).contains(&pivot),
            "pivot={pivot}"
        );

        let in_l: Vec<f64> = dry_lr.iter().map(|&[l, _]| f64::from(l)).collect();
        let in_r: Vec<f64> = dry_lr.iter().map(|&[_, r]| f64::from(r)).collect();
        let tgt_l: Vec<f64> = star_wet.iter().map(|&[l, _]| f64::from(l)).collect();
        let tgt_r: Vec<f64> = star_wet.iter().map(|&[_, r]| f64::from(r)).collect();

        let runner = TrainRunner::cpu();
        let mut model = run_pass1_probe(&in_l, &in_r, &tgt_l, &tgt_r, sr, 128, 32)
            .map_or_else(|_| SonaModelF64::new(sr), |p| p.1);
        model.oversample_upper_tiers_2x = false;
        let _ = run_pass3_varpro_readout(&runner, &mut model, &in_l, &in_r, &tgt_l, &tgt_r, 128);

        let pairs = StereoTrainingPairs {
            in_l: &in_l,
            in_r: &in_r,
            target_l: &tgt_l,
            target_r: &tgt_r,
        };
        let applied = try_apply_spectral_control_bus_pass3(&runner, &mut model, &pairs, 128, pivot)
            .expect("spectral control bus pass3 should succeed");
        assert!(
            applied,
            "expected spectral control bus commit on flat target"
        );
        assert_eq!(model.direct_s2_u_gain, [0.0, 0.0]);
        assert_eq!(model.pre_conv_u_gain, [0.0, 0.0]);
        let [esr_l, esr_u] = eval_vm_lite_ultra_esr_db(&model, &pairs, 128).unwrap();
        assert!(esr_l < -40.0 && esr_u < -40.0, "esr=({esr_l}, {esr_u})");

        // Verify exact zero-bypass weights and zeroed ParaLRU across dedicated Arena 0 bus lanes.
        let mut bus_w = [0.0_f64; 4];
        for q in 0..4 {
            let l_q = (0..4)
                .find(|&l| model.w_ro[q * 4 + l][0].abs() > 0.5)
                .expect("dedicated bus lane must carry nonzero w_ro[0]");
            assert_eq!(model.core_quads[q].lru_alpha_rel[l_q], 0.0);
            assert_eq!(model.core_quads[q].lru_alpha_atk[l_q], 1.0);
            bus_w[q] = model.w_ro[q * 4 + l_q][0];
        }
        assert_eq!(
            bus_w,
            [
                DEDICATED_PRE_DIR_WEIGHT,
                DEDICATED_PRE_BAND_WEIGHT,
                DEDICATED_PRE_BAND_WEIGHT,
                DEDICATED_PRE_BAND_WEIGHT,
            ]
        );

        let bin_base = encode_model_to_dnam_bin(
            &model,
            Pass5EmitConfig {
                calibrate_lufs_trims: false,
                ..Pass5EmitConfig::default()
            },
        )
        .unwrap();
        let in_l32: Vec<f32> = dry_lr.iter().map(|lr| lr[0]).collect();

        // Sweeping block 3 ADAA slot t[5] alters HF tilt smoothly via SVF HP lane.
        let mut st_dark = [0.0_f32; STAGE_TARGET_SLOTS];
        st_dark[5] = -1.2;
        let mut vm_dark =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_base, Tier::Lite, sr as f32)
                .unwrap();
        vm_dark.reset_with_latched_stage_targets(&st_dark, &[0.0; 8]);
        let (mut dl, mut dr) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        vm_dark
            .process_block(&in_l32, &in_l32, &mut dl, &mut dr)
            .unwrap();
        let mut star_sq = 0.0_f64;
        let mut diff_sq = 0.0_f64;
        for t in 128..n {
            star_sq += tgt_l[t] * tgt_l[t];
            diff_sq += (tgt_l[t] - f64::from(dl[t])).powi(2);
        }
        let dark_diff_esr = 10.0 * (diff_sq / star_sq.max(1.0e-18)).max(1.0e-16).log10();
        assert!(
            dark_diff_esr > -20.0,
            "t[5] modulation must alter response relative to StarCenter (got {dark_diff_esr})"
        );

        // Switch voicing check: pole = 0.45 1-pole LP target over 1,024 samples with adjoint solve.
        let mut pole45_wet = vec![[0.0_f32; 2]; n];
        let mut st45 = 0.0_f64;
        for (i, &[dl, _]) in dry_lr.iter().enumerate() {
            st45 = 0.55 * (0.8 * f64::from(dl)) + 0.45 * st45;
            pole45_wet[i] = [st45 as f32, st45 as f32];
        }
        stmc_by_id.insert(
            1,
            make_stmc_payload(1, STMC_FLAG_ENABLED | STMC_FLAG_STEREO, &pole45_wet),
        );
        let pivot45 = detect_offcenter_spectral_pivot_hz(&manifest, &stmc_by_id, &star_wet, n)
            .expect("pole=0.45 LP switch must yield spectral pivot");
        assert!(
            pivot45 >= 3_000.0,
            "expected >= 3 kHz pivot45, got {pivot45}"
        );

        // Non-stationary LF-then-HF broadband gain change (>= 12,288 frames) must return None.
        let n_ns = 12_288_usize;
        let ns_star: Vec<[f32; 2]> = (0..n_ns)
            .map(|i| {
                let t = (i as f64) / sr;
                let s = if i < n_ns / 3 {
                    0.25 * (TAU * 120.0 * t).sin() + 0.20 * (TAU * 280.0 * t).sin()
                } else {
                    0.25 * (TAU * 2_400.0 * t).sin() + 0.20 * (TAU * 6_000.0 * t).sin()
                } as f32;
                [s, s]
            })
            .collect();
        let ns_gain: Vec<[f32; 2]> = ns_star.iter().map(|&[l, r]| [0.4 * l, 0.4 * r]).collect();
        let mut ns_stmc = BTreeMap::new();
        ns_stmc.insert(
            0,
            make_stmc_payload(
                0,
                STMC_FLAG_ENABLED | STMC_FLAG_STAR_CENTER | STMC_FLAG_STEREO,
                &ns_star,
            ),
        );
        ns_stmc.insert(
            1,
            make_stmc_payload(1, STMC_FLAG_ENABLED | STMC_FLAG_STEREO, &ns_gain),
        );
        assert_eq!(
            detect_offcenter_spectral_pivot_hz(&manifest, &ns_stmc, &ns_star, n_ns),
            None
        );
        let mut model45 = run_pass1_probe(&in_l, &in_r, &tgt_l, &tgt_r, sr, 128, 32)
            .map_or_else(|_| SonaModelF64::new(sr), |p| p.1);
        model45.oversample_upper_tiers_2x = false;
        assert!(
            try_apply_spectral_control_bus_pass3(&runner, &mut model45, &pairs, 128, pivot45)
                .unwrap()
        );

        let n_adj = 1_024_usize;
        let pole45_l: Vec<f64> = pole45_wet[..n_adj]
            .iter()
            .map(|lr| f64::from(lr[0]))
            .collect();
        let pole45_r: Vec<f64> = pole45_wet[..n_adj]
            .iter()
            .map(|lr| f64::from(lr[1]))
            .collect();
        let cfg = StageTargetSolveConfig {
            active_slots: (0..31).collect(),
            tier: Tier::Ultra,
            warmup_frames: 128,
            loss_weights: CompositeLossWeights::from_config(&CompositeLossConfig {
                w_besr: 0.45,
                w_mr_iir: 0.0,
                w_spec_ot: 0.0,
                w_gabor: 0.0,
                w_if_gd: 0.0,
                w_ltas: 0.15,
                w_seg_esr: 0.0,
                w_nyquist: 0.0,
                w_mse: 0.40,
                ..CompositeLossConfig::default()
            }),
            barrier_weight: 0.0,
            l2_reg: 1.0e-6,
            opt: SsBroydenConfig {
                max_iters: 20,
                max_line_search: 5,
                rel_tol: 1.0e-5,
                patience: 3,
                ..SsBroydenConfig::default()
            },
            vm_micro_polish_iters: 0,
            ..StageTargetSolveConfig::default()
        };
        let bounds = StageTargetBounds::canonical(false, 16.0);
        let rep = solve_point_stage_targets_adjoint(
            &StageTargetSolveInput {
                runner: &runner,
                base: &model45,
                in_l: &in_l[..n_adj],
                in_r: &in_r[..n_adj],
                target_l: &pole45_l,
                target_r: &pole45_r,
                init_targets: &[0.0; 32],
                z_stage: &[0.0; 8],
                bounds: &bounds,
            },
            &cfg,
            None,
        )
        .expect("pole=0.45 adjoint solve should succeed");

        let bin45 = encode_model_to_dnam_bin(
            &model45,
            Pass5EmitConfig {
                calibrate_lufs_trims: false,
                ..Pass5EmitConfig::default()
            },
        )
        .unwrap();
        let mut st32 = [0.0_f32; STAGE_TARGET_SLOTS];
        for (dst, &src) in st32[..STAGE_TARGET_SLOTS - 1]
            .iter_mut()
            .zip(&rep.stage_targets)
        {
            *dst = if src.is_finite() { src as f32 } else { 0.0 };
        }
        let tgt_full: Vec<f64> = pole45_wet.iter().map(|lr| f64::from(lr[0])).collect();
        for tier in [Tier::Lite, Tier::Ultra] {
            let mut vm45 =
                SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin45, tier, sr as f32).unwrap();
            vm45.reset_with_latched_stage_targets(&st32, &[0.0; 8]);
            let (mut ol, mut or) = (vec![0.0_f32; n], vec![0.0_f32; n]);
            vm45.process_block(&in_l32, &in_l32, &mut ol, &mut or)
                .unwrap();
            let mdl_f64: Vec<f64> = ol.into_iter().map(f64::from).collect();
            let (se, ee, ne) = try_split_bezout_3band_energies(&tgt_full, &mdl_f64, None, sr, 128)
                .expect("3-band energies");
            let besr_rep = evaluate_besr_from_band_energies(se, ee, ne);
            let ltas_rep =
                evaluate_ltas_33band_report(&tgt_full[128..], &mdl_f64[128..], None, sr, true);
            assert!(
                besr_rep.besr_db < -24.0 && ltas_rep.max_adj_deviation_db < 0.5,
                "expected zero-bypass pre_conv bus on {tier:?} to solve pole=0.45 LP cleanly, got bESR={:.2} dB, LTAS={:.3} dB",
                besr_rep.besr_db,
                ltas_rep.max_adj_deviation_db
            );
        }
    }

    #[test]
    fn test_spectral_bus_hybrid_fallback_on_heavy_nonlinear_target() {
        let n = 4_096_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                0.16 * (TAU * 130.0 * t).sin()
                    + 0.12 * (TAU * 310.0 * t).sin()
                    + 0.08 * (TAU * 720.0 * t).sin()
                    + 0.05 * (TAU * 1_480.0 * t).sin()
            })
            .collect();
        let in_r = in_l.clone();
        let tgt_l: Vec<f64> = in_l.iter().map(|&x| 0.65 * (4.0 * x).tanh()).collect();
        let tgt_r = tgt_l.clone();

        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 64;
        model.core_quads[0].tone_dir = [1.0, 0.0, 0.0, 0.0];
        model.core_quads[0].adaa_drive[0] = 3.0;
        model.core_quads[0].in_mid_weight = [1.0; 4];
        model.core_quads[0].in_side_weight = [1.0; 4];

        let runner = TrainRunner::cpu();
        let pairs = StereoTrainingPairs {
            in_l: &in_l,
            in_r: &in_r,
            target_l: &tgt_l,
            target_r: &tgt_r,
        };
        let applied =
            try_apply_spectral_control_bus_pass3(&runner, &mut model, &pairs, 128, 1_000.0)
                .expect("hybrid fallback should succeed");
        assert!(applied, "expected Stage 2 hybrid spectral bus commit");
        assert_eq!(model.direct_s2_u_gain, [0.0, 0.0]);
        assert_eq!(model.pre_conv_u_gain, [0.0, 0.0]);
        assert!(
            (model.core_quads[0].adaa_drive[0] - 3.0).abs() < 1.0e-9,
            "Stage 2 hybrid must preserve non-bus nonlinear lane adaa_drive"
        );
        let [_esr_l, esr_u] = eval_vm_lite_ultra_esr_db(&model, &pairs, 128).unwrap();
        assert!(esr_u < -20.0, "esr_u={esr_u}");
    }
}
