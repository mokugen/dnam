#![forbid(unsafe_code)]
//! Stage D (`3-Phase Audit`, `LUFS` Inter-Tier Loudness Parity, `[STAB-ZERO-DC]` Packing &
//! `3-Layer Buyer Watermarking`, `solver::stage_d_audit`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.stage-e-pack]`)
//! - `docs/specs/security/watermarking_and_marketplace.md` (`ref[sec.watermark.reciprocal-gauge]`)

mod trims;
pub mod watermark;

pub use watermark::{
    BuyerWatermarkConfig, derive_buyer_gauge_params, derive_buyer_phase_seed,
    derive_buyer_tardos_codebook, export_runtime_dnam_watermarked, extract_arena_pole_angles_16,
    extract_arena_pole_angles_34, identify_pole_angles_blackbox_from_audio,
    render_blackbox_probe_response,
};

use std::f64::consts::PI;

use dnam_engine::container::build_dnam_container_with_cp1;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::schema::DnamError;
use dnam_engine::stages::V3CoreInstance;
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, pack_v3_matryoshka_to_arena, unpack_v3_matryoshka_from_arena,
    unpack_v3_tier_from_arena,
};
use dnam_engine::{DnamEngine, Tier};

use crate::audit::arena::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
use crate::proj::bundle::{sync_manifest_stage4_delay_history, zero_arena_b_head_dc_bias};
use crate::proj::schema::{DnamProjManifest, SolverTelemetrySummary};
use crate::solver::canonical_besr_warmup;
use crate::solver::stage_c_knobs::star_center_control_values;

use trims::{
    ALL_TIERS, StageDCertCache, apply_allpass_and_kautz_phase_rotation, blend_cold_readouts,
    calibrate_trims_matching_ultra_lufs, eval_trimmed_tier_esr, fine_tune_monotone_trims,
    measure_tier_loudness_db, pair_has_monotone_cushion, pair_is_strict_monotone,
    read_tier_gain_trims, solve_exact_tier_trim, write_tier_gain_trims,
};

/// Options controlling Stage D 3-Phase audit depth and inter-tier loudness calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageDAuditOptions {
    /// `true` on Path 2 (`StarCenter` re-pool); `false` on Path 1 (`0-ULP` `StarCenter` invariance).
    pub full_backbone_audit: bool,
    /// `true` on Path 2 to calibrate `<= 0.05 LU` inter-tier `tier_gain_trim` multipliers.
    pub calibrate_tier_trims: bool,
}

impl StageDAuditOptions {
    /// Path 2 full backbone audit with inter-tier loudness trim calibration.
    pub const PATH2_FULL: Self = Self {
        full_backbone_audit: true,
        calibrate_tier_trims: true,
    };
    /// Path 1 incremental head-only audit preserving existing trims and `StarCenter` outputs.
    pub const PATH1_HEAD_ONLY: Self = Self {
        full_backbone_audit: false,
        calibrate_tier_trims: false,
    };
}

/// Output summary from Stage D finalization & 3-Phase audit (`r[train.pipeline.stage-e-pack]`).
#[derive(Debug, Clone, PartialEq)]
pub struct StageDSolveOutput {
    /// Wall-clock Stage D execution time in milliseconds.
    pub elapsed_ms: f32,
    /// Post-warmup `[Lite, Full, Pro, Ultra]` `StarCenter` ESR in dB.
    pub tier_esr_db: [f32; 4],
    /// Post-warmup `[Lite, Full, Pro, Ultra]` `StarCenter` integrated loudness in LUFS.
    pub tier_lufs: [f64; 4],
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

#[derive(Clone, Copy)]
struct BiquadK {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl BiquadK {
    fn stage1_shelf(sr_hz: f64) -> Self {
        let f0 = 1_681.974_450_955_533_f64;
        let db = 3.999_843_853_973_347_f64;
        let q = 0.707_175_236_955_419_6_f64;
        let k = (PI * f0 / sr_hz).tan();
        let v_h = 10.0_f64.powf(db / 20.0);
        let v_b = v_h.powf(0.499_666_774_154_541_6);
        let norm = 1.0 / (1.0 + k / q + k * k);
        Self {
            b0: (v_h + v_b * k / q + k * k) * norm,
            b1: 2.0 * (k * k - v_h) * norm,
            b2: (v_h - v_b * k / q + k * k) * norm,
            a1: 2.0 * (k * k - 1.0) * norm,
            a2: (1.0 - k / q + k * k) * norm,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn stage2_rlb_hp(sr_hz: f64) -> Self {
        let f0 = 38.135_470_876_024_44_f64;
        let q = 0.500_327_037_323_877_3_f64;
        let k = (PI * f0 / sr_hz).tan();
        let norm = 1.0 / (1.0 + k / q + k * k);
        Self {
            b0: norm,
            b1: -2.0 * norm,
            b2: norm,
            a1: 2.0 * (k * k - 1.0) * norm,
            a2: (1.0 - k / q + k * k) * norm,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline(always)]
    fn step(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Computes ITU-R BS.1770-4 K-weighted integrated loudness in LUFS (`r[train.pipeline.stage-e-pack]`).
#[must_use]
pub fn compute_integrated_lufs(samples_lr: &[[f32; 2]], sample_rate_hz: u32) -> f64 {
    if samples_lr.is_empty() {
        return -120.0;
    }
    let sr = f64::from(sample_rate_hz.clamp(8_000, 384_000));
    let mut s1_l = BiquadK::stage1_shelf(sr);
    let mut s1_r = BiquadK::stage1_shelf(sr);
    let mut s2_l = BiquadK::stage2_rlb_hp(sr);
    let mut s2_r = BiquadK::stage2_rlb_hp(sr);

    let mut powers = Vec::with_capacity(samples_lr.len());
    let mut total_e = 0.0_f64;
    for &[l, r] in samples_lr {
        let fl = if l.is_finite() { f64::from(l) } else { 0.0 };
        let fr = if r.is_finite() { f64::from(r) } else { 0.0 };
        let yl = s2_l.step(s1_l.step(fl));
        let yr = s2_r.step(s1_r.step(fr));
        let p = 0.5 * (yl * yl + yr * yr);
        powers.push(p);
        total_e += p;
    }

    let win = ((0.400 * sr).round() as usize).max(1);
    let hop = ((0.100 * sr).round() as usize).max(1);
    if powers.len() < win {
        let mean_p = total_e / (powers.len() as f64);
        return if mean_p <= 1.0e-12 {
            -120.0
        } else {
            (-0.691 + 10.0 * mean_p.log10()).clamp(-120.0, 20.0)
        };
    }

    let mut block_powers = Vec::new();
    let abs_floor = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let mut start = 0usize;
    while start + win <= powers.len() {
        let sum: f64 = powers[start..start + win].iter().sum();
        let mean_b = sum / (win as f64);
        if mean_b > abs_floor {
            block_powers.push(mean_b);
        }
        start += hop;
    }
    if block_powers.is_empty() {
        let mean_p = total_e / (powers.len() as f64);
        return if mean_p <= 1.0e-12 {
            -120.0
        } else {
            (-0.691 + 10.0 * mean_p.log10()).clamp(-120.0, 20.0)
        };
    }
    let rel_mean: f64 = block_powers.iter().sum::<f64>() / (block_powers.len() as f64);
    let rel_thresh = rel_mean * 0.1; // -10 LU
    let (mut kept_sum, mut kept_cnt) = (0.0_f64, 0usize);
    for &b in &block_powers {
        if b >= rel_thresh {
            kept_sum += b;
            kept_cnt += 1;
        }
    }
    let gated = if kept_cnt > 0 {
        kept_sum / (kept_cnt as f64)
    } else {
        rel_mean
    };
    (-0.691 + 10.0 * gated.max(1.0e-12).log10()).clamp(-120.0, 20.0)
}

fn render_tier_star_center(
    manifest: &DnamProjManifest,
    arena_untrimmed: &[u8],
    untrimmed_dnam: Option<&[u8]>,
    star_ctrls: &[f32],
    tier: Tier,
    mono_out: bool,
    dry_lr: &[[f32; 2]],
) -> Vec<[f32; 2]> {
    if let Some(dnam_bytes) = untrimmed_dnam
        && let Ok(mut eng) =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(dnam_bytes, tier, 48_000)
    {
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        return dry_lr
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_out))
            .collect();
    }
    let dual_in = manifest.base.io_config.inputs >= 2;
    let n = dry_lr.len();
    let cold = unpack_v3_tier_from_arena(arena_untrimmed, tier, mono_out);
    let keep_in_r = !mono_out || dual_in || cold.feat_params.key_link > 0.0;
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&cold);
    inst.reset_to_dc_equilibrium();
    let in_l: Vec<f32> = dry_lr.iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr
        .iter()
        .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
        .collect();
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    inst.process_block_stereo(&in_l, &in_r, &mut out_l, &mut out_r);
    (0..n)
        .map(|i| [out_l[i], if mono_out { out_l[i] } else { out_r[i] }])
        .collect()
}

struct CrossRateCandidate {
    esr: f32,
    trim: f32,
    arena: Vec<u8>,
    rendered: Vec<[f32; 2]>,
}

struct StageDRepairCtx<'a> {
    manifest: &'a mut DnamProjManifest,
    arena: &'a mut [u8],
    rendered: &'a mut [Vec<[f32; 2]>; 4],
    trims: &'a mut [f32; 4],
    raw_esr: &'a mut [f32; 4],
    ultra_db: &'a mut f64,
    wet_slice: &'a [[f32; 2]],
    dry_slice: &'a [[f32; 2]],
    star_ctrls: &'a [f32],
    mono_out: bool,
    warmup: usize,
    use_broadband_rms: bool,
    calibrate_trims: bool,
}

impl StageDRepairCtx<'_> {
    fn render_candidate_tier_arena(
        &mut self,
        model: &dnam_engine::stages::V3MatryoshkaModel,
        tier: Tier,
    ) -> (Vec<u8>, Vec<[f32; 2]>) {
        let mut cand_arena = self.arena.to_vec();
        pack_v3_matryoshka_to_arena(model, &mut cand_arena);
        run_head_only_3phase_arena_audit(self.manifest, &mut cand_arena);
        write_tier_gain_trims(&mut cand_arena, [1.0; 4]);
        let mut cand_hdr = self.manifest.base.clone();
        sync_manifest_stage4_delay_history(&mut cand_hdr, &cand_arena);
        let cand_dnam =
            build_dnam_container_with_cp1(&mut cand_hdr, &cand_arena, &cand_arena, false, &[]).ok();
        let cand_out = render_tier_star_center(
            self.manifest,
            &cand_arena,
            cand_dnam.as_deref(),
            self.star_ctrls,
            tier,
            self.mono_out,
            self.dry_slice,
        );
        (cand_arena, cand_out)
    }

    fn repair_same_rate_inversions(&mut self) {
        let mut model = unpack_v3_matryoshka_from_arena(self.arena, self.mono_out);
        let n_len = self
            .rendered
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(self.wet_slice.len());
        if n_len <= self.warmup + 64 {
            return;
        }
        let mut cert = StageDCertCache::new(self.wet_slice, n_len, self.warmup, self.mono_out);
        let (mut sums_l, mut sums_r, _, _, _) =
            cert.eval_all_tiers_and_monotone(self.rendered, *self.trims, self.use_broadband_rms);

        for tau in 1..4 {
            let prev_l = &sums_l[tau - 1];
            let prev_r = sums_r.as_ref().map(|r| &r[tau - 1]);
            let cur_l = &sums_l[tau];
            let cur_r = sums_r.as_ref().map(|r| &r[tau]);
            if pair_is_strict_monotone(prev_l, cur_l, prev_r, cur_r)
                || model.tier_os2x[tau] != model.tier_os2x[tau - 1]
            {
                continue;
            }
            let orig_passes = cur_l.passes_target && cur_r.is_none_or(|r| r.passes_target);
            let orig_tau = model.readouts[tau].clone();
            let prev_ro = model.readouts[tau - 1].clone();
            let alphas: &[f32] = if tau == 3 {
                // Keep Ultra distinct from Pro unless zero-blend is the only way to repair a genuine regression.
                &[0.95, 0.88, 0.75, 0.55, 0.35, 0.15, 0.04]
            } else {
                &[0.95, 0.88, 0.75, 0.55, 0.35, 0.15, 0.04, 0.0]
            };
            let mut repaired = false;
            for &alpha in alphas {
                model.readouts[tau] =
                    blend_cold_readouts(&orig_tau, &prev_ro, alpha, ALL_TIERS[tau]);
                let (cand_arena, cand_out) =
                    self.render_candidate_tier_arena(&model, ALL_TIERS[tau]);
                let cand_trim = if !self.calibrate_trims {
                    self.trims[tau]
                } else if tau == 3 {
                    1.0
                } else {
                    solve_exact_tier_trim(
                        &cand_out[self.warmup..],
                        *self.ultra_db,
                        self.mono_out,
                        self.use_broadband_rms,
                    )
                };
                let (sl, sr) = cert.eval_tier(&cand_out, cand_trim, ALL_TIERS[tau]);
                let cand_passes = sl.passes_target && sr.as_ref().is_none_or(|r| r.passes_target);
                if (!orig_passes || cand_passes)
                    && pair_is_strict_monotone(prev_l, &sl, prev_r, sr.as_ref())
                {
                    let cand_esr = sl
                        .raw_besr_db
                        .max(sr.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db))
                        as f32;
                    self.arena.copy_from_slice(&cand_arena);
                    model = unpack_v3_matryoshka_from_arena(self.arena, self.mono_out);
                    self.rendered[tau] = cand_out;
                    self.raw_esr[tau] = cand_esr;
                    self.trims[tau] = cand_trim;
                    sums_l[tau] = sl;
                    if let (Some(rv), Some(s_r)) = (sums_r.as_mut(), sr) {
                        rv[tau] = s_r;
                    }
                    if tau == 3 && self.calibrate_trims {
                        *self.ultra_db = measure_tier_loudness_db(
                            &self.rendered[3][self.warmup..],
                            self.mono_out,
                            self.use_broadband_rms,
                        );
                        for t in 0..3 {
                            self.trims[t] = solve_exact_tier_trim(
                                &self.rendered[t][self.warmup..],
                                *self.ultra_db,
                                self.mono_out,
                                self.use_broadband_rms,
                            );
                            let (tl, tr) =
                                cert.eval_tier(&self.rendered[t], self.trims[t], ALL_TIERS[t]);
                            self.raw_esr[t] = tl
                                .raw_besr_db
                                .max(tr.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db))
                                as f32;
                            sums_l[t] = tl;
                            if let (Some(rv), Some(s_r)) = (sums_r.as_mut(), tr) {
                                rv[t] = s_r;
                            }
                        }
                    }
                    repaired = true;
                    break;
                }
            }
            if !repaired {
                model.readouts[tau] = orig_tau;
            }
        }
    }

    fn repair_cross_rate_1x_inversions(&mut self) {
        const THETA_GRID: [f32; 7] = [0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.65];
        let n_len = self
            .rendered
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(self.wet_slice.len());
        if n_len <= self.warmup + 64 {
            return;
        }
        let mut cert = StageDCertCache::new(self.wet_slice, n_len, self.warmup, self.mono_out);
        let (mut sums_l, mut sums_r, _, _, _) =
            cert.eval_all_tiers_and_monotone(self.rendered, *self.trims, self.use_broadband_rms);

        for target_tau in [1_usize, 0] {
            let hi_l = &sums_l[target_tau + 1];
            let hi_r = sums_r.as_ref().map(|r| &r[target_tau + 1]);
            if pair_is_strict_monotone(
                &sums_l[target_tau],
                hi_l,
                sums_r.as_ref().map(|r| &r[target_tau]),
                hi_r,
            ) {
                continue;
            }
            let orig_passes = sums_l[target_tau].passes_target
                && sums_r.as_ref().is_none_or(|r| r[target_tau].passes_target);
            let mut model = unpack_v3_matryoshka_from_arena(self.arena, self.mono_out);
            let orig_r = model.readouts[target_tau].clone();
            let mut best_ok: Option<(
                CrossRateCandidate,
                crate::metrics::certification::PerceptualCertificationSummary,
                Option<crate::metrics::certification::PerceptualCertificationSummary>,
            )> = None;
            for &theta in &THETA_GRID {
                model.readouts[target_tau] = orig_r.clone();
                apply_allpass_and_kautz_phase_rotation(
                    &mut model.readouts[target_tau],
                    ALL_TIERS[target_tau],
                    theta,
                );
                let (cand_arena, cand_out) =
                    self.render_candidate_tier_arena(&model, ALL_TIERS[target_tau]);
                let cand_trim = if self.calibrate_trims {
                    solve_exact_tier_trim(
                        &cand_out[self.warmup..],
                        *self.ultra_db,
                        self.mono_out,
                        self.use_broadband_rms,
                    )
                } else {
                    self.trims[target_tau]
                };
                let (sl, sr) = cert.eval_tier(&cand_out, cand_trim, ALL_TIERS[target_tau]);
                let cand_passes = sl.passes_target && sr.as_ref().is_none_or(|r| r.passes_target);
                if (!orig_passes || cand_passes)
                    && pair_is_strict_monotone(&sl, hi_l, sr.as_ref(), hi_r)
                {
                    let cand_esr = sl
                        .raw_besr_db
                        .max(sr.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db))
                        as f32;
                    best_ok = Some((
                        CrossRateCandidate {
                            esr: cand_esr,
                            trim: cand_trim,
                            arena: cand_arena,
                            rendered: cand_out,
                        },
                        sl,
                        sr,
                    ));
                    break;
                }
            }
            if let Some((best, sl, sr)) = best_ok {
                self.arena.copy_from_slice(&best.arena);
                self.rendered[target_tau] = best.rendered;
                self.trims[target_tau] = best.trim;
                self.raw_esr[target_tau] = best.esr;
                sums_l[target_tau] = sl;
                if let (Some(rv), Some(s_r)) = (sums_r.as_mut(), sr) {
                    rv[target_tau] = s_r;
                }
            }
        }
    }
}

/// Finalizes Stage D (`3-Phase Audit`, `[STAB-ZERO-DC]`, `<= 0.05 LU` loudness trims, and solver
/// telemetry, `r[train.pipeline.stage-e-pack]`).
pub fn finalize_stage_d_and_audit(
    manifest: &mut DnamProjManifest,
    arena: &mut [u8],
    dry_lr: &[[f32; 2]],
    star_wet_lr: Option<&[[f32; 2]]>,
    fallback_esr: Option<[f32; 4]>,
    prior_solve_ms: f32,
    audit_opts: StageDAuditOptions,
) -> Result<StageDSolveOutput, DnamError> {
    if arena.len() < ARENA_LITE_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena.len(),
        });
    }
    let timer = now_ms_token();

    if audit_opts.full_backbone_audit {
        run_full_3phase_arena_audit(manifest, arena);
    } else {
        run_head_only_3phase_arena_audit(manifest, arena);
    }

    let mono_out = manifest.base.io_config.output_mode == "mono";
    let raw_n_eval = star_wet_lr.map_or(dry_lr.len(), |w| dry_lr.len().min(w.len()));
    let has_sweep = raw_n_eval >= 76_096
        && dry_lr[71_100..71_900]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4)
        && dry_lr[72_000..raw_n_eval]
            .iter()
            .any(|&[l, r]| l.abs() > 1.0e-4 || r.abs() > 1.0e-4);
    const BS1770_MIN_GATED_SAMPLES_48K: usize = 19_200;
    let use_broadband_rms =
        has_sweep || raw_n_eval >= 76_096 || raw_n_eval < BS1770_MIN_GATED_SAMPLES_48K;
    let n_eval = if has_sweep {
        dry_lr[..72_000]
            .iter()
            .rposition(|&[l, r]| l.abs() > 1.0e-6 || r.abs() > 1.0e-6)
            .map_or(72_000, |idx| (idx + 1).min(72_000))
    } else {
        raw_n_eval
    };
    let warmup = canonical_besr_warmup(n_eval).min(n_eval.saturating_sub(1));

    let mut trims = read_tier_gain_trims(arena);
    let mut tier_lufs = [-120.0_f64; 4];
    let mut tier_esr_db = fallback_esr.unwrap_or([0.0; 4]);

    if n_eval > warmup + 8 {
        let mut arena_untrimmed = arena.to_vec();
        write_tier_gain_trims(&mut arena_untrimmed, [1.0; 4]);
        let mut hdr_copy = manifest.base.clone();
        sync_manifest_stage4_delay_history(&mut hdr_copy, &arena_untrimmed);
        let untrimmed_dnam = build_dnam_container_with_cp1(
            &mut hdr_copy,
            &arena_untrimmed,
            &arena_untrimmed,
            false,
            &[],
        )
        .ok();
        let star_ctrls = star_center_control_values(manifest);
        let mut rendered: [Vec<[f32; 2]>; 4] = ALL_TIERS.map(|t| {
            render_tier_star_center(
                manifest,
                &arena_untrimmed,
                untrimmed_dnam.as_deref(),
                &star_ctrls,
                t,
                mono_out,
                &dry_lr[..n_eval],
            )
        });

        if let Some(wet_slice) = star_wet_lr {
            let mut ultra_db =
                measure_tier_loudness_db(&rendered[3][warmup..], mono_out, use_broadband_rms);
            let n_cert = n_eval.min(wet_slice.len());
            let has_s4_cascade = {
                let m = unpack_v3_matryoshka_from_arena(arena, mono_out);
                (0..3).any(|t| {
                    m.stage4[t].delay_lengths != m.stage4[3].delay_lengths
                        || (m.stage4[t].wet_mix - m.stage4[3].wet_mix).abs() > 1.0e-3
                        || (m.stage4[t].g_loop - m.stage4[3].g_loop).abs() > 1.0e-3
                })
            };
            let (mut raw_esr, mut is_monotone) = if n_cert > warmup + 64 {
                let mut cert = StageDCertCache::new(wet_slice, n_cert, warmup, mono_out);
                let (sums_u_l, sums_u_r, esr_untrimmed, lu_untrimmed, mono_untrimmed) =
                    cert.eval_all_tiers_and_monotone(&rendered, trims, use_broadband_rms);
                let is_incoherent_s4 = esr_untrimmed[3] > -16.0;
                let cush_untrimmed = mono_untrimmed
                    && (is_incoherent_s4
                        || has_s4_cascade
                        || (0..3).all(|t| {
                            pair_has_monotone_cushion(
                                &sums_u_l[t],
                                &sums_u_l[t + 1],
                                sums_u_r.as_ref().map(|r| &r[t]),
                                sums_u_r.as_ref().map(|r| &r[t + 1]),
                            )
                        }));
                let lu_gate = if is_incoherent_s4 || has_s4_cascade {
                    0.048
                } else {
                    0.025
                };
                if lu_untrimmed <= lu_gate && cush_untrimmed {
                    // Untrimmed tiers already satisfy cushioned 7-metric monotonicity and LU parity.
                    (esr_untrimmed, true)
                } else if audit_opts.calibrate_tier_trims
                    && let Some((cand_trims, cand_esr, udb, mono_ok)) =
                        calibrate_trims_matching_ultra_lufs(
                            &rendered,
                            wet_slice,
                            mono_out,
                            warmup,
                            use_broadband_rms,
                        )
                {
                    trims = cand_trims;
                    ultra_db = udb;
                    (cand_esr, mono_ok)
                } else {
                    (esr_untrimmed, mono_untrimmed)
                }
            } else if audit_opts.calibrate_tier_trims
                && let Some((cand_trims, cand_esr, udb, mono_ok)) =
                    calibrate_trims_matching_ultra_lufs(
                        &rendered,
                        wet_slice,
                        mono_out,
                        warmup,
                        use_broadband_rms,
                    )
            {
                trims = cand_trims;
                ultra_db = udb;
                (cand_esr, mono_ok)
            } else {
                let esr: [f32; 4] = std::array::from_fn(|t| {
                    eval_trimmed_tier_esr(&rendered[t], wet_slice, trims[t], mono_out, warmup)
                });
                let ok = (0..3).all(|t| esr[t + 1] <= esr[t]);
                (esr, ok)
            };

            if audit_opts.full_backbone_audit && !is_monotone && !has_s4_cascade {
                let mut repair_ctx = StageDRepairCtx {
                    manifest,
                    arena,
                    rendered: &mut rendered,
                    trims: &mut trims,
                    raw_esr: &mut raw_esr,
                    ultra_db: &mut ultra_db,
                    wet_slice,
                    dry_slice: &dry_lr[..n_eval],
                    star_ctrls: &star_ctrls,
                    mono_out,
                    warmup,
                    use_broadband_rms,
                    calibrate_trims: audit_opts.calibrate_tier_trims,
                };
                repair_ctx.repair_same_rate_inversions();
                if audit_opts.calibrate_tier_trims {
                    is_monotone = fine_tune_monotone_trims(
                        repair_ctx.rendered,
                        wet_slice,
                        repair_ctx.trims,
                        repair_ctx.raw_esr,
                        mono_out,
                        warmup,
                    );
                } else if n_cert > warmup + 64 {
                    let mut cert = StageDCertCache::new(wet_slice, n_cert, warmup, mono_out);
                    let (_, _, esr_chk, _, mono_chk) = cert.eval_all_tiers_and_monotone(
                        repair_ctx.rendered,
                        *repair_ctx.trims,
                        use_broadband_rms,
                    );
                    *repair_ctx.raw_esr = esr_chk;
                    is_monotone = mono_chk;
                } else {
                    is_monotone =
                        (0..3).all(|t| repair_ctx.raw_esr[t + 1] <= repair_ctx.raw_esr[t]);
                }
                if !is_monotone {
                    repair_ctx.repair_cross_rate_1x_inversions();
                }
            }
            if audit_opts.calibrate_tier_trims {
                write_tier_gain_trims(arena, trims);
            }
        }

        for t in 0..4 {
            let trimmed_out: Vec<[f32; 2]> = rendered[t][warmup..]
                .iter()
                .map(|&[l, r]| [l * trims[t], r * trims[t]])
                .collect();
            tier_lufs[t] = compute_integrated_lufs(&trimmed_out, 48_000);
            if let Some(wet_slice) = star_wet_lr {
                tier_esr_db[t] =
                    eval_trimmed_tier_esr(&rendered[t], wet_slice, trims[t], mono_out, warmup);
            }
        }
    }

    zero_arena_b_head_dc_bias(arena);
    let stage_d_ms = elapsed_ms_since(timer);
    let active_stem_count = manifest.stems_registry.iter().filter(|s| s.enabled).count() as u32;
    manifest.solver_telemetry = Some(SolverTelemetrySummary {
        active_stem_count,
        total_solve_ms: (prior_solve_ms + stage_d_ms).max(0.0),
        tier_esr_db,
    });

    Ok(StageDSolveOutput {
        elapsed_ms: stage_d_ms,
        tier_esr_db,
        tier_lufs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::stages::V3MatryoshkaModel;
    use dnam_engine::stages::weights::{ARENA_ULTRA_BYTES, pack_v3_matryoshka_to_arena};
    use std::f64::consts::TAU;

    #[test]
    fn test_lufs_and_stage_d_audit_path1_vs_path2() {
        assert_eq!(compute_integrated_lufs(&[], 48_000), -120.0);
        let n = 24_000usize; // 0.5 s > 400 ms BS.1770 gating window
        let tone: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.25 * (TAU * 997.0 * (i as f64) / 48_000.0).sin() as f32;
                [s, s]
            })
            .collect();
        let lufs = compute_integrated_lufs(&tone, 48_000);
        assert!(lufs > -22.0 && lufs < -10.0, "unexpected LUFS {lufs}");

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let mat = V3MatryoshkaModel::identity(true, false);
        pack_v3_matryoshka_to_arena(&mat, &mut arena);
        let mut manifest = DnamProjManifest::default_reference();

        let d_out = finalize_stage_d_and_audit(
            &mut manifest,
            &mut arena,
            &tone[..2_048],
            Some(&tone[..2_048]),
            None,
            1.5,
            StageDAuditOptions::PATH2_FULL,
        )
        .expect("stage d path 2 should succeed");
        assert!(d_out.tier_lufs.iter().all(|v| v.is_finite()));
        assert!(manifest.solver_telemetry.is_some());

        let head_only = finalize_stage_d_and_audit(
            &mut manifest,
            &mut arena,
            &tone[..2_048],
            None,
            Some([-20.0; 4]),
            0.5,
            StageDAuditOptions::PATH1_HEAD_ONLY,
        )
        .expect("stage d path 1 should succeed");
        assert_eq!(head_only.tier_esr_db, [-20.0; 4]);
    }
}
