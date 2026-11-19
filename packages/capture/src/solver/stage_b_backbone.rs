#![forbid(unsafe_code)]
//! Stage B (`Joint 4-Tier Matryoshka Backbone` & `256-D` `uplo='U'` `STAT` Contraction, `solver::stage_b_backbone`).
//!
//! Canonical Reference: `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.stage-b-harvest]`).

use dnam_engine::Tier;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::DnamError;
use dnam_engine::stages::V3CoreInstance;
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, ARENA_ULTRA_BYTES, ARENA0_STAGE0_PARAMS_OFFSET, PIPE_FLAG_V3_MATRYOSHKA,
    has_pipeline_flag, pack_v3_matryoshka_to_arena, read_f32_le, unpack_v3_matryoshka_from_arena,
    unpack_v3_tier_from_arena,
};

mod whitebox;

use crate::proj::PROJ_STAT_BYTES_ULTRA;
use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::capsules::verify_proj_stat_uplo_u;
use crate::proj::schema::DnamProjManifest;
use crate::solver::progress::{
    TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus,
};
use crate::solver::stage_a_probe::has_populated_v3_backbone;
use crate::solver::v3_varpro::{
    V3TrainDataset, V3TrainStem, fit_v3_matryoshka, fit_v3_matryoshka_fast,
};
use crate::solver::wide_gram::contract_256d_gram_tiled_wide192;
use crate::solver::{SolverQualityMode, canonical_besr_warmup, uplo_u_packed_len};

/// Borrowed stereo stem reference and regression weight for fast weighted readout patching.
#[derive(Debug, Clone, Copy)]
pub struct WeightedStemRef<'a> {
    /// Dry stereo input waveform at `48 kHz`.
    pub dry_lr: &'a [[f32; 2]],
    /// Target wet stereo waveform at `48 kHz`.
    pub wet_lr: &'a [[f32; 2]],
    /// Non-negative stem weight in the `uplo='U'` normal equations.
    pub weight: f64,
}

/// Fullband multi-regime body slice `[0..72_000)` before the `[72_000..96_000)` sweep.
const FULLBAND_BODY_LEN: usize = 72_000;
const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

/// Output summary and `uplo='U'` sufficient statistics from [`solve_stage_b_backbone`].
#[derive(Debug, Clone)]
pub struct StageBOutputs {
    /// Wall-clock Stage B duration in milliseconds.
    pub elapsed_ms: f32,
    /// Canonical `267,264 B` `uplo='U'` `STAT` chunk payload (`PROJ_STAT_BYTES_ULTRA`).
    pub stat_bytes: Vec<u8>,
    /// Packed upper-triangular `256 × 256` `StarCenter` Gram matrix (`32,896` elements).
    pub packed_gram_256: Vec<f64>,
    /// `256 × 2` stereo `StarCenter` feature-target cross-correlations.
    pub cross_256: [[f64; 2]; 256],
    /// Post-warmup `[Lite, Full, Pro, Ultra]` `StarCenter` ESR in dB.
    pub prefix_esr_db: [f32; 4],
}

/// Configuration for [`solve_stage_b_backbone_with_progress`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageBSolveConfig {
    /// Frame stride when contracting `256 × 256` `uplo='U'` Gram statistics (`default = 4`).
    pub gram_frame_stride: usize,
    /// Preserve pre-populated white-box `v3` backbone weights while solving readouts.
    pub preserve_existing_backbone: bool,
    /// Optimization convergence vs. threshold-stop quality policy.
    pub quality_mode: SolverQualityMode,
    /// Optional empirical left/right repeatability profiles measured strictly from `(x, y)` probes.
    pub rep_profiles: Option<[crate::metrics::TargetRepeatabilityProfile; 2]>,
}

impl Default for StageBSolveConfig {
    fn default() -> Self {
        Self {
            gram_frame_stride: 4,
            preserve_existing_backbone: false,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            rep_profiles: None,
        }
    }
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

/// Serializes `(packed_gram_256, cross_256)` into the canonical `267,264 B` `uplo='U'` `STAT` chunk
/// (`PROJ_STAT_BYTES_ULTRA`, `ref[container.proj.stat-uplo-u]`).
#[must_use]
pub fn serialize_v3_stat_chunk_ultra(
    packed_gram_256: &[f64],
    cross_256: &[[f64; 2]; 256],
) -> Vec<u8> {
    let n_pack = uplo_u_packed_len(256);
    let mut out = Vec::with_capacity(PROJ_STAT_BYTES_ULTRA);
    for idx in 0..n_pack {
        let v = packed_gram_256.get(idx).copied().unwrap_or(0.0);
        let clean = if v.is_finite() { v } else { 0.0 };
        out.extend_from_slice(&clean.to_le_bytes());
    }
    for &[cl, cr] in cross_256 {
        let vl = if cl.is_finite() { cl } else { 0.0 };
        let vr = if cr.is_finite() { cr } else { 0.0 };
        out.extend_from_slice(&vl.to_le_bytes());
        out.extend_from_slice(&vr.to_le_bytes());
    }
    out
}

fn eval_tier_esr_db(
    arena: &[u8],
    tier: Tier,
    mono_out: bool,
    dual_in: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> f32 {
    let n = dry_lr.len().min(wet_lr.len());
    if n == 0 {
        return 0.0;
    }
    let cold = unpack_v3_tier_from_arena(arena, tier, mono_out);
    let keep_in_r = !mono_out || dual_in || cold.feat_params.key_link > 0.0;
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&cold);
    inst.reset_to_dc_equilibrium();
    let in_l: Vec<f32> = dry_lr[..n].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr[..n]
        .iter()
        .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
        .collect();
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    inst.process_block_stereo(&in_l, &in_r, &mut out_l, &mut out_r);

    let warmup = canonical_besr_warmup(n).min(n.saturating_sub(1));
    let mut sig_sq = 0.0_f64;
    let mut err_sq = 0.0_f64;
    for t in warmup..n {
        let tl = f64::from(wet_lr[t][0]);
        let dl = tl - f64::from(out_l[t]);
        sig_sq += tl * tl;
        err_sq += dl * dl;
        if !mono_out {
            let tr = f64::from(wet_lr[t][1]);
            let dr = tr - f64::from(out_r[t]);
            sig_sq += tr * tr;
            err_sq += dr * dr;
        }
    }
    if sig_sq <= 1.0e-18 {
        if err_sq <= 1.0e-18 { -120.0 } else { 0.0 }
    } else {
        (10.0 * (err_sq / sig_sq).max(1.0e-16).log10()) as f32
    }
}

fn eval_all_four_tier_esr_db(
    arena: &[u8],
    mono_out: bool,
    dual_in: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> [f32; 4] {
    ALL_TIERS.map(|t| eval_tier_esr_db(arena, t, mono_out, dual_in, dry_lr, wet_lr))
}

fn build_star_dataset(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    mono_out: bool,
    dual_in: bool,
    active_samples: usize,
) -> (V3TrainDataset, bool) {
    let n = active_samples.min(dry_lr.len()).min(wet_lr.len());
    if n == 0 {
        return (V3TrainDataset::new(Vec::new(), 48_000.0), false);
    }
    let keep_in_r = !mono_out || dual_in;
    let slice_stem = |lo: usize, hi: usize| -> V3TrainStem {
        let len = hi - lo;
        V3TrainStem {
            in_l: dry_lr[lo..hi].iter().map(|lr| lr[0]).collect(),
            in_r: dry_lr[lo..hi]
                .iter()
                .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
                .collect(),
            tgt_l: wet_lr[lo..hi].iter().map(|lr| lr[0]).collect(),
            tgt_r: wet_lr[lo..hi]
                .iter()
                .map(|lr| if mono_out { lr[0] } else { lr[1] })
                .collect(),
            warmup: canonical_besr_warmup(len),
        }
    };
    let has_sweep = n >= 76_096
        && dry_lr[71_100..71_900]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4)
        && dry_lr[FULLBAND_BODY_LEN..n]
            .iter()
            .any(|&[l, r]| l.abs() > 1.0e-4 || r.abs() > 1.0e-4);
    if has_sweep {
        let body_active_end = dry_lr[..FULLBAND_BODY_LEN]
            .iter()
            .rposition(|&[l, r]| l.abs() > 1.0e-6 || r.abs() > 1.0e-6)
            .map_or(FULLBAND_BODY_LEN, |idx| (idx + 1).min(FULLBAND_BODY_LEN));
        let body = slice_stem(0, body_active_end);
        let sweep = slice_stem(FULLBAND_BODY_LEN, n);
        let mut ds = V3TrainDataset::new_weighted(vec![(body, 0.80), (sweep, 0.20)], 48_000.0);
        ds.full_cert_stem = Some(slice_stem(0, n));
        (ds, true)
    } else if n >= 48_000 {
        let tail_active = (0..n).rfind(|&idx| {
            dry_lr[idx][0].abs() > 1.0e-5
                || dry_lr[idx][1].abs() > 1.0e-5
                || wet_lr[idx][0].abs() > 1.0e-5
                || wet_lr[idx][1].abs() > 1.0e-5
        });
        let n_fit = tail_active.map_or(n, |last| (last + 4_096).clamp(48_000, n));
        let stem = slice_stem(0, n_fit);
        (V3TrainDataset::new(vec![stem], 48_000.0), true)
    } else {
        let stem = slice_stem(0, n);
        (
            V3TrainDataset::new_weighted_opts(vec![(stem, 1.0)], 48_000.0, false, false),
            false,
        )
    }
}

fn fill_256d_basis_row(
    row: &mut [f32; 256],
    io_lr: [f32; 4],
    dl_hist: &[f32; 16],
    y_hist_l: &[f32; 96],
    y_hist_r: &[f32; 96],
    inst: &V3CoreInstance<DefaultMathBackend>,
) {
    let [dl, dr, yl, yr] = io_lr;
    let hot = inst.hot_core();
    row[0] = dl;
    row[1] = dr;
    row[2..6].copy_from_slice(&hot.last_s2_dir);

    const MATRYOSHKA_BG: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)];
    let mut cursor = 6usize;
    for &(b, g) in &MATRYOSHKA_BG {
        row[cursor..cursor + 4].copy_from_slice(&hot.state_l.h[b][g]);
        cursor += 4;
    }
    let mut active_r_cursor = 30usize;
    for &(b, g) in &MATRYOSHKA_BG {
        row[active_r_cursor..active_r_cursor + 4].copy_from_slice(&hot.state_r.h[b][g]);
        active_r_cursor += 4;
    }
    row[54..62].copy_from_slice(&dl_hist[..8]);
    row[62] = yl;
    row[63] = yr;
    for (dst, &v) in row[64..160].iter_mut().zip(y_hist_l.iter()) {
        *dst = v;
    }
    for (dst, &v) in row[160..256].iter_mut().zip(y_hist_r.iter()) {
        *dst = v;
    }
}

fn extract_star_256d_gram(
    arena: &[u8],
    mono_out: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_samples: usize,
    stride: usize,
) -> (Vec<f64>, [[f64; 2]; 256]) {
    let n = active_samples.min(dry_lr.len()).min(wet_lr.len());
    if n == 0 {
        return (vec![0.0; uplo_u_packed_len(256)], [[0.0; 2]; 256]);
    }
    let step = stride.max(1);
    let warmup = canonical_besr_warmup(n).min(n.saturating_sub(1));
    let cold = unpack_v3_tier_from_arena(arena, Tier::Ultra, mono_out);
    let keep_in_r = !mono_out || cold.feat_params.key_link > 0.0;
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&cold);
    inst.reset_to_dc_equilibrium();

    let cap = (n - warmup).div_ceil(step).max(1);
    let mut feats = Vec::with_capacity(cap);
    let mut tgts = Vec::with_capacity(cap);
    let mut dl_hist = [0.0_f32; 16];
    let mut y_hist_l = [0.0_f32; 96];
    let mut y_hist_r = [0.0_f32; 96];

    for t in 0..n {
        let dl = dry_lr[t][0];
        let dr = if keep_in_r { dry_lr[t][1] } else { dl };
        let [yl, yr] = inst.step_stereo(dl, dr);
        dl_hist.rotate_right(1);
        dl_hist[0] = dl;
        y_hist_l.rotate_right(1);
        y_hist_l[0] = yl;
        y_hist_r.rotate_right(1);
        y_hist_r[0] = yr;

        if t >= warmup && (t - warmup).is_multiple_of(step) {
            let mut row = [0.0_f32; 256];
            fill_256d_basis_row(
                &mut row,
                [dl, dr, yl, yr],
                &dl_hist,
                &y_hist_l,
                &y_hist_r,
                &inst,
            );
            feats.push(row);
            let tl = f64::from(wet_lr[t][0]);
            let tr = if mono_out {
                tl
            } else {
                f64::from(wet_lr[t][1])
            };
            tgts.push([tl, tr]);
        }
    }
    let weights = vec![step as f64; feats.len()];
    let (gram_g, cross_vec) = contract_256d_gram_tiled_wide192(&feats, &tgts, &weights, 1);
    let mut cross_256 = [[0.0_f64; 2]; 256];
    for (dst, &src) in cross_256.iter_mut().zip(cross_vec.iter()) {
        *dst = src;
    }
    (gram_g, cross_256)
}

/// Serializes `arena`'s `Ultra` trajectory Gram statistics (`267,264 B` `uplo='U'` `STAT`) directly without re-solving.
pub fn compute_v3_stat_chunk_bytes(
    arena: &[u8],
    mono_out: bool,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_samples: usize,
    stride: usize,
) -> Result<Vec<u8>, DnamError> {
    if arena.len() < ARENA_ULTRA_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_ULTRA_BYTES,
            actual: arena.len(),
        });
    }
    let (packed_gram_256, cross_256) =
        extract_star_256d_gram(arena, mono_out, dry_lr, wet_lr, active_samples, stride);
    let stat_bytes = serialize_v3_stat_chunk_ultra(&packed_gram_256, &cross_256);
    let _ = verify_proj_stat_uplo_u(&stat_bytes)?;
    Ok(stat_bytes)
}

/// Fast weighted exact-`f32` `uplo='U'` Cholesky readout patch across all 4 tiers without nonlinear L-BFGS.
pub fn patch_v3_readouts_weighted_cholesky(
    model: &mut dnam_engine::stages::V3MatryoshkaModel,
    stems: &[WeightedStemRef<'_>],
) {
    whitebox::patch_v3_readouts_weighted_cholesky(model, stems);
}

/// Executes Stage B (`Joint 4-Tier Matryoshka Backbone` fit or white-box preservation) and
/// contracts the canonical `267,264 B` `uplo='U'` `STAT` chunk (`r[train.pipeline.stage-b-harvest]`).
pub fn solve_stage_b_backbone(
    manifest: &DnamProjManifest,
    arena: &mut [u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_frames_hint: usize,
    gram_frame_stride: usize,
    preserve_existing_backbone: bool,
) -> Result<StageBOutputs, DnamError> {
    solve_stage_b_backbone_with_progress(
        manifest,
        arena,
        dry_lr,
        wet_lr,
        active_frames_hint,
        StageBSolveConfig {
            gram_frame_stride,
            preserve_existing_backbone,
            quality_mode: SolverQualityMode::BestUntilDiminishingReturns,
            rep_profiles: None,
        },
        &TrainerProgressHandle::default(),
    )
}

/// Variant of [`solve_stage_b_backbone`] accepting [`StageBSolveConfig`] and a hybrid
/// [`TrainerProgressHandle`] (`AtomicTrainerProgress` seqlock + coarse callback).
pub fn solve_stage_b_backbone_with_progress(
    manifest: &DnamProjManifest,
    arena: &mut [u8],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    active_frames_hint: usize,
    config: StageBSolveConfig,
    progress: &TrainerProgressHandle,
) -> Result<StageBOutputs, DnamError> {
    let gram_frame_stride = config.gram_frame_stride;
    let preserve_existing_backbone = config.preserve_existing_backbone;
    if arena.len() < ARENA_ULTRA_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_ULTRA_BYTES,
            actual: arena.len(),
        });
    }
    if arena.len() < ARENA_LITE_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena.len(),
        });
    }

    let timer = now_ms_token();
    let mono_out = manifest.base.io_config.output_mode == "mono";
    let dual_in = manifest.base.io_config.inputs >= 2;
    let avail = dry_lr.len().min(wet_lr.len());
    let active_samples = if active_frames_hint == 0 {
        avail
    } else {
        active_frames_hint.min(avail)
    };
    let has_sweep = active_samples >= 76_096
        && dry_lr[71_100..71_900]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4)
        && dry_lr[FULLBAND_BODY_LEN..active_samples]
            .iter()
            .any(|&[l, r]| l.abs() > 1.0e-4 || r.abs() > 1.0e-4);
    let eval_len = if has_sweep {
        dry_lr[..FULLBAND_BODY_LEN]
            .iter()
            .rposition(|&[l, r]| l.abs() > 1.0e-6 || r.abs() > 1.0e-6)
            .map_or(FULLBAND_BODY_LEN, |idx| (idx + 1).min(FULLBAND_BODY_LEN))
    } else {
        active_samples
    };

    if active_samples > 0 {
        if preserve_existing_backbone && has_populated_v3_backbone(arena) {
            let init_esr = eval_all_four_tier_esr_db(
                arena,
                mono_out,
                dual_in,
                &dry_lr[..eval_len],
                &wet_lr[..eval_len],
            );
            if init_esr.iter().any(|&e| !e.is_finite() || e > -85.0) {
                let mut existing = unpack_v3_matryoshka_from_arena(arena, mono_out);
                whitebox::solve_preserved_backbone_f32_readouts(
                    &mut existing,
                    &dry_lr[..active_samples],
                    &wet_lr[..active_samples],
                );
                pack_v3_matryoshka_to_arena(&existing, arena);
                zero_arena_b_head_dc_bias(arena);
            }
        } else {
            let prev_xo_hz = has_pipeline_flag(arena, PIPE_FLAG_V3_MATRYOSHKA)
                .then(|| read_f32_le(arena, ARENA0_STAGE0_PARAMS_OFFSET))
                .filter(|xo| (20.0..=2_000.0).contains(xo));
            let (mut ds, use_full) =
                build_star_dataset(dry_lr, wet_lr, mono_out, dual_in, active_samples);
            ds.rep_profiles = config.rep_profiles;
            let ds = ds
                .with_quality_mode(config.quality_mode)
                .with_progress(progress.subrange(0, 920));
            let mut fitted = if use_full {
                fit_v3_matryoshka(&ds, mono_out)
            } else {
                fit_v3_matryoshka_fast(&ds, mono_out)
            };
            let stage0_active = fitted.stage0.op_p_sub_mix.abs() > 1.0e-6
                || fitted.stage0.op_p_sub2_mix.abs() > 1.0e-6
                || fitted.stage0.op_p_oct_mix.abs() > 1.0e-6
                || fitted.stage0.op_p_env_mix.abs() > 1.0e-6
                || fitted.stage0.clean_sub_mix.abs() > 1.0e-6
                || fitted.stage0.op_p_route_clean;
            if !stage0_active && let Some(xo_hz) = prev_xo_hz {
                fitted.stage0.xo_hz = xo_hz;
            }
            pack_v3_matryoshka_to_arena(&fitted, arena);
            zero_arena_b_head_dc_bias(arena);
        }
    }

    let prefix_esr_db = eval_all_four_tier_esr_db(
        arena,
        mono_out,
        dual_in,
        &dry_lr[..eval_len],
        &wet_lr[..eval_len],
    );
    if progress.is_active() {
        progress.emit(&TrainerProgressSnapshot {
            status: TrainingStatus::Running,
            phase: TrainingPhase::StageBBackbone,
            active_tier: Some(Tier::Ultra),
            progress_permille: 1_000,
            tier_besr_db: prefix_esr_db,
            ..Default::default()
        });
    }
    let (packed_gram_256, cross_256) = extract_star_256d_gram(
        arena,
        mono_out,
        dry_lr,
        wet_lr,
        active_samples,
        gram_frame_stride,
    );
    let stat_bytes = serialize_v3_stat_chunk_ultra(&packed_gram_256, &cross_256);
    let _ = verify_proj_stat_uplo_u(&stat_bytes)?;

    Ok(StageBOutputs {
        elapsed_ms: elapsed_ms_since(timer),
        stat_bytes,
        packed_gram_256,
        cross_256,
        prefix_esr_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::stages::V3MatryoshkaModel;
    use std::f64::consts::TAU;

    #[test]
    fn test_stage_b_backbone_fast_solve_preserve_and_stat_roundtrip() {
        let mut small_arena = vec![0u8; ARENA_LITE_BYTES];
        let manifest = DnamProjManifest::default_reference();
        assert!(
            solve_stage_b_backbone(&manifest, &mut small_arena, &[], &[], 0, 1, false).is_err()
        );

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let centers = crate::metrics::companion::canonical_ltas_33_centers_hz();
        let n = 4_096usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let mut sum = 0.0_f64;
                for (k, &hz) in centers.iter().enumerate() {
                    sum += (TAU * hz * t + 0.37 * (k as f64)).sin();
                }
                let s = (0.045 * sum) as f32;
                [s, s]
            })
            .collect();
        let wet: Vec<[f32; 2]> = dry
            .iter()
            .map(|&[l, r]| [0.75 * (2.2 * l).tanh(), 0.75 * (2.2 * r).tanh()])
            .collect();

        let out = solve_stage_b_backbone(&manifest, &mut arena, &dry, &wet, n, 2, false)
            .expect("stage b fresh fast solve should succeed");
        assert_eq!(out.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
        assert_eq!(verify_proj_stat_uplo_u(&out.stat_bytes).unwrap(), 256);
        assert!(out.prefix_esr_db[3] < -12.0, "got {:?}", out.prefix_esr_db);

        // White-box preservation path: zeroed readouts should be recovered while keeping nonlinear blocks.
        let mut preserved_model = unpack_v3_matryoshka_from_arena(&arena, false);
        let saved_w_in = preserved_model.blocks[0].w_in_n;
        preserved_model.readouts[0].pre_cab_u_gain = [0.0; 2];
        preserved_model.readouts[0].pre_cab_w = [[0.0; 36]; 2];
        preserved_model.readouts[0].fir_l = [0.0; 64];
        preserved_model.readouts[0].svf_direct_taps_l = [0.0; 8];
        pack_v3_matryoshka_to_arena(&preserved_model, &mut arena);

        let out_pres = solve_stage_b_backbone(&manifest, &mut arena, &dry, &wet, n, 2, true)
            .expect("stage b preserve solve should succeed");
        let after = unpack_v3_matryoshka_from_arena(&arena, false);
        assert_eq!(after.blocks[0].w_in_n, saved_w_in);
        assert!(out_pres.prefix_esr_db[3] < -12.0);
        let _ = V3MatryoshkaModel::identity(true, false);
    }
}
