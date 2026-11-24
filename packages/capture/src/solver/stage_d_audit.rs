#![forbid(unsafe_code)]
//! Stage D (`3-Phase Audit`, `LUFS` Inter-Tier Loudness Parity, `[STAB-ZERO-DC]` Certification &
//! `3-Layer Buyer Watermarking`, `solver::stage_d_audit`).
//!
//! Canonical References:
//! - `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.host-parity-and-export]`)
//! - `docs/specs/security/watermarking_and_marketplace.md` (`ref[sec.watermark.reciprocal-gauge]`)

pub mod watermark;

pub use watermark::{
    BuyerWatermarkConfig, derive_buyer_gauge_params, derive_buyer_phase_seed,
    derive_buyer_tardos_codebook, export_runtime_dnam_watermarked, extract_arena_pole_angles_16,
    extract_arena_pole_angles_32, extract_arena_pole_angles_34,
    identify_pole_angles_blackbox_from_audio, render_blackbox_probe_response,
};

use dnam_engine::schema::DnamError;
use dnam_train::{
    Pass5EmitConfig, compute_itu_bs1770_lufs, decode_dnam_bin_to_model, run_pass5_emit_and_certify,
};
use dnam_vm::bin::{DNBN_FLAG_STEREO_OUT, DnamBinView};
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use crate::audit::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
use crate::metrics::{evaluate_besr_from_band_energies, try_split_bezout_3band_energies};
use crate::proj::bundle::sync_manifest_transport_delay_history;
use crate::proj::schema::{DnamProjManifest, SolverTelemetrySummary};
use crate::proj::verify_dnam_bin_zero_dc;
use crate::solver::canonical_besr_warmup;

const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

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

/// Output summary from Stage D finalization & 3-Phase audit (`ref[train.pipeline.host-parity-and-export]`).
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

#[inline]
fn map_stage_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

/// Computes ITU-R BS.1770-4 K-weighted integrated loudness in LUFS (`ref[train.pipeline.host-parity-and-export]`).
#[must_use]
pub fn compute_integrated_lufs(samples_lr: &[[f32; 2]], sample_rate_hz: u32) -> f64 {
    if samples_lr.is_empty() {
        return -120.0;
    }
    let sr = f64::from(sample_rate_hz.clamp(8_000, 384_000));
    let l: Vec<f64> = samples_lr
        .iter()
        .map(|lr| {
            if lr[0].is_finite() {
                f64::from(lr[0])
            } else {
                0.0
            }
        })
        .collect();
    let r: Vec<f64> = samples_lr
        .iter()
        .map(|lr| {
            if lr[1].is_finite() {
                f64::from(lr[1])
            } else {
                0.0
            }
        })
        .collect();
    compute_itu_bs1770_lufs(&l, &r, sr, 0)
}

fn eval_rendered_esr_db(
    out_l: &[f32],
    out_r: &[f32],
    wet_lr: &[[f32; 2]],
    mono_out: bool,
    warmup: usize,
) -> f32 {
    let n = out_l.len().min(out_r.len()).min(wet_lr.len());
    if n <= warmup {
        return 0.0;
    }
    let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
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

fn eval_tier_3band_besr_db(
    out_l: &[f32],
    out_r: &[f32],
    wet_lr: &[[f32; 2]],
    mono_out: bool,
    warmup: usize,
) -> f32 {
    let n = out_l.len().min(out_r.len()).min(wet_lr.len());
    if n <= warmup + 8 {
        return eval_rendered_esr_db(out_l, out_r, wet_lr, mono_out, warmup);
    }
    let tgt_l: Vec<f64> = wet_lr[..n].iter().map(|lr| f64::from(lr[0])).collect();
    let mdl_l: Vec<f64> = out_l[..n].iter().map(|&v| f64::from(v)).collect();
    if let Some((sl, el, nl)) =
        try_split_bezout_3band_energies(&tgt_l, &mdl_l, None, 48_000.0, warmup)
    {
        let rep_l = evaluate_besr_from_band_energies(sl, el, nl);
        if rep_l.valid {
            if mono_out {
                return rep_l.besr_db as f32;
            }
            let tgt_r: Vec<f64> = wet_lr[..n].iter().map(|lr| f64::from(lr[1])).collect();
            let mdl_r: Vec<f64> = out_r[..n].iter().map(|&v| f64::from(v)).collect();
            if let Some((sr, er, nr)) =
                try_split_bezout_3band_energies(&tgt_r, &mdl_r, None, 48_000.0, warmup)
            {
                let rep_r = evaluate_besr_from_band_energies(sr, er, nr);
                if rep_r.valid {
                    return rep_l.besr_db.max(rep_r.besr_db) as f32;
                }
            }
        }
    }
    eval_rendered_esr_db(out_l, out_r, wet_lr, mono_out, warmup)
}

struct VmEvalInputs<'a> {
    dry_lr: &'a [[f32; 2]],
    star_wet_lr: Option<&'a [[f32; 2]]>,
    len: usize,
    warmup: usize,
    mono_out: bool,
    dual_in: bool,
}

fn eval_tiers_via_vm(
    dnam_bin: &[u8],
    inputs: &VmEvalInputs<'_>,
    tier_lufs: &mut [f64; 4],
    tier_esr_db: &mut [f32; 4],
) -> Result<(), DnamError> {
    let VmEvalInputs {
        dry_lr,
        star_wet_lr,
        len,
        warmup,
        mono_out,
        dual_in,
    } = *inputs;
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let keep_in_r = dual_in || !mono_out;
    let in_l: Vec<f32> = dry_lr[..len].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = dry_lr[..len]
        .iter()
        .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
        .collect();
    let max_t = view.retained_tier;
    for (t_idx, &tier) in ALL_TIERS.iter().enumerate() {
        let eff_tier = if tier.index() <= max_t.index() {
            tier
        } else {
            max_t
        };
        let mut vm = SonaVm::<DefaultMathBackend>::from_view_owned(
            &view,
            eff_tier,
            48_000.0,
            mono_out && !dual_in,
        )
        .map_err(map_stage_err)?;
        let (mut out_l, mut out_r) = (vec![0.0_f32; len], vec![0.0_f32; len]);
        vm.process_block(&in_l, &in_r, &mut out_l, &mut out_r)
            .map_err(map_stage_err)?;
        let post: Vec<[f32; 2]> = out_l[warmup..]
            .iter()
            .zip(&out_r[warmup..])
            .map(|(&l, &r)| [l, if mono_out { l } else { r }])
            .collect();
        tier_lufs[t_idx] = compute_integrated_lufs(&post, 48_000);
        if let Some(wet_slice) = star_wet_lr {
            tier_esr_db[t_idx] =
                eval_tier_3band_besr_db(&out_l, &out_r, &wet_slice[..len], mono_out, warmup);
        }
    }
    Ok(())
}

/// Input slices and options for [`finalize_stage_d_and_audit_with_panama_rows`].
#[derive(Clone, Copy, Debug)]
pub struct StageDFinalizeInputs<'a> {
    /// Aligned dry reference frames `[L, R]`.
    pub dry_lr: &'a [[f32; 2]],
    /// Optional aligned `StarCenter` target wet frames `[L, R]` (required for `Path 2` trims).
    pub star_wet_lr: Option<&'a [[f32; 2]]>,
    /// Fallback per-tier `bESR` values when no audio evaluation window is supplied.
    pub fallback_esr: Option<[f32; 4]>,
    /// Prior Stage A/B/C cumulative solver wall time in milliseconds.
    pub prior_solve_ms: f32,
    /// Stage D audit mode (`PATH2_FULL` vs `PATH1_HEAD_ONLY`).
    pub audit_opts: StageDAuditOptions,
    /// Pass-4 solved stationary & synchronous `MODE_A_SWEEP` sample rows for `PANAMA`.
    pub panama_sample_rows: &'a [dnam_train::PanamaSampleRow],
}

/// Finalizes Stage D (`3-Phase Audit`, `LUFS` trim calibration on Path 2, closed-form
/// `[CLAMP-DYNAMIC]` + 256-frame extreme-corner `[STAB-ZERO-DC]` certification, `< 10 ms` `PANAMA`
/// corner recommendations, transport delay history synchronization, and zero-DC check)
/// (`ref[train.pipeline.host-parity-and-export]`, `ref[train.audit.report.prescription]`).
pub fn finalize_stage_d_and_audit_with_panama_rows(
    manifest: &mut DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    inputs: StageDFinalizeInputs<'_>,
) -> Result<StageDSolveOutput, DnamError> {
    let dry_lr = inputs.dry_lr;
    let star_wet_lr = inputs.star_wet_lr;
    let fallback_esr = inputs.fallback_esr;
    let prior_solve_ms = inputs.prior_solve_ms;
    let audit_opts = inputs.audit_opts;
    let panama_sample_rows = inputs.panama_sample_rows;

    let view_pre = DnamBinView::parse(dnam_bin).map_err(|_| DnamError::BufferTooSmall {
        expected: 128,
        actual: dnam_bin.len(),
    })?;
    let prev_stereo = (view_pre.header.flags.get() & DNBN_FLAG_STEREO_OUT) != 0;
    let timer = now_ms_token();

    if audit_opts.full_backbone_audit {
        run_full_3phase_arena_audit(manifest, dnam_bin);
    } else {
        run_head_only_3phase_arena_audit(manifest, dnam_bin);
    }

    let mono_out = manifest.base.io_config.output_mode == "mono";
    let dual_in = manifest.base.io_config.inputs >= 2;
    let raw_n_eval = star_wet_lr.map_or(dry_lr.len(), |w| dry_lr.len().min(w.len()));
    let has_sweep = raw_n_eval >= 76_096
        && dry_lr[71_100..71_900]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4)
        && dry_lr[72_000..raw_n_eval]
            .iter()
            .any(|&[l, r]| l.abs() > 1.0e-4 || r.abs() > 1.0e-4);
    let body_cap = if has_sweep { 72_000 } else { raw_n_eval };
    let last_active_dry = dry_lr[..body_cap]
        .iter()
        .rposition(|&[l, r]| l.abs() > 1.0e-6 || r.abs() > 1.0e-6)
        .map_or(body_cap, |idx| (idx + 1).min(body_cap));
    let min_eval = canonical_besr_warmup(body_cap).saturating_add(64);
    let tail_wet_silent = star_wet_lr.is_none_or(|wet| {
        wet[last_active_dry..body_cap]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4)
    });
    let n_eval = if has_sweep
        || (body_cap.saturating_sub(last_active_dry) >= body_cap / 2
            && last_active_dry > min_eval
            && tail_wet_silent)
    {
        last_active_dry
    } else {
        body_cap
    };
    let warmup = canonical_besr_warmup(n_eval).min(n_eval.saturating_sub(1));

    let mut tier_lufs = [-120.0_f64; 4];
    let mut tier_esr_db = fallback_esr.unwrap_or([0.0; 4]);

    if audit_opts.full_backbone_audit
        && let Some(wet_slice) = star_wet_lr
        && n_eval > warmup + 8
    {
        let mut model = decode_dnam_bin_to_model(dnam_bin).map_err(map_stage_err)?;
        let keep_in_r = dual_in || !mono_out;
        let in_l: Vec<f64> = dry_lr[..n_eval].iter().map(|lr| f64::from(lr[0])).collect();
        let in_r: Vec<f64> = dry_lr[..n_eval]
            .iter()
            .map(|lr| f64::from(if keep_in_r { lr[1] } else { lr[0] }))
            .collect();
        let tgt_l: Vec<f64> = wet_slice[..n_eval]
            .iter()
            .map(|lr| f64::from(lr[0]))
            .collect();
        let tgt_r: Vec<f64> = wet_slice[..n_eval]
            .iter()
            .map(|lr| f64::from(if mono_out { lr[0] } else { lr[1] }))
            .collect();

        let emit_cfg = Pass5EmitConfig {
            stereo_out: prev_stereo,
            single_input: !dual_in && !prev_stereo,
            calibrate_lufs_trims: audit_opts.calibrate_tier_trims,
            ..Default::default()
        };
        let res =
            run_pass5_emit_and_certify(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, warmup, emit_cfg)
                .map_err(map_stage_err)?;
        *dnam_bin = res.dnam_bin;
        tier_lufs = res.tier_lufs;
        for (t, rep) in res.tier_reports.iter().enumerate() {
            tier_esr_db[t] = rep.besr_db as f32;
        }
        let body_warmup = canonical_besr_warmup(body_cap).min(body_cap.saturating_sub(1));
        if n_eval < body_cap && !has_sweep && body_cap > body_warmup + 8 {
            eval_tiers_via_vm(
                dnam_bin,
                &VmEvalInputs {
                    dry_lr,
                    star_wet_lr,
                    len: body_cap,
                    warmup: body_warmup,
                    mono_out,
                    dual_in,
                },
                &mut tier_lufs,
                &mut tier_esr_db,
            )?;
        }
    } else {
        let eval_len = if has_sweep { n_eval } else { body_cap };
        let eval_warmup = canonical_besr_warmup(eval_len).min(eval_len.saturating_sub(1));
        if eval_len > eval_warmup + 8 {
            eval_tiers_via_vm(
                dnam_bin,
                &VmEvalInputs {
                    dry_lr,
                    star_wet_lr,
                    len: eval_len,
                    warmup: eval_warmup,
                    mono_out,
                    dual_in,
                },
                &mut tier_lufs,
                &mut tier_esr_db,
            )?;
        }
    }

    crate::audit::audit_dynamic_corners_and_zero_dc(
        manifest,
        dnam_bin,
        audit_opts.full_backbone_audit,
    )?;
    let _ = crate::audit::append_panama_prescriptions(manifest, dnam_bin, panama_sample_rows)?;
    sync_manifest_transport_delay_history(&mut manifest.base, dnam_bin);
    verify_dnam_bin_zero_dc(dnam_bin)?;

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

/// Finalizes Stage D (`3-Phase Audit`, `LUFS` trim calibration on Path 2, transport delay history
/// synchronization, and `[STAB-ZERO-DC]` verification) (`ref[train.pipeline.host-parity-and-export]`).
pub fn finalize_stage_d_and_audit(
    manifest: &mut DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    dry_lr: &[[f32; 2]],
    star_wet_lr: Option<&[[f32; 2]]>,
    fallback_esr: Option<[f32; 4]>,
    prior_solve_ms: f32,
    audit_opts: StageDAuditOptions,
) -> Result<StageDSolveOutput, DnamError> {
    finalize_stage_d_and_audit_with_panama_rows(
        manifest,
        dnam_bin,
        StageDFinalizeInputs {
            dry_lr,
            star_wet_lr,
            fallback_esr,
            prior_solve_ms,
            audit_opts,
            panama_sample_rows: &[],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;
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

        let mut dnam_bin = initialize_backbone_dnam_bin_if_needed(None).expect("neutral dnam-bin");
        let mut manifest = DnamProjManifest::default_reference();

        let d_out = finalize_stage_d_and_audit(
            &mut manifest,
            &mut dnam_bin,
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
            &mut dnam_bin,
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
