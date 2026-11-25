#![forbid(unsafe_code)]
//! 4-Tier (`Lite`, `Full`, `Pro`, `Ultra`) VM render & 4-metric certification evaluator for `M25`.

use dnam_capture::metrics::MAX_TIER_LOUDNESS_STEP_LU;
use dnam_capture::metrics::certification::universal::{
    certify_4metric_universal, certify_4metric_universal_with_dry_gaps, certify_holdout_universal,
};
use dnam_capture::metrics::certification::{
    PerceptualCertificationSummary, verify_monotone_4tier_summaries,
};
use dnam_capture::metrics::repeatability::TargetRepeatabilityProfile;
use dnam_capture::proj::bundle::export_runtime_dnam;
use dnam_capture::solver::{ComputeBackendPreference, compute_integrated_lufs};
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};
use dnam_vm::math::DefaultMathBackend;

pub(crate) const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

/// Per-tier, per-channel certification result for a single qualification level.
#[derive(Clone, Debug)]
pub(crate) struct TierChannelReport {
    pub tier: Tier,
    pub channel: usize,
    pub summary: PerceptualCertificationSummary,
    pub holdout_besr_db: f64,
    pub holdout_pass: bool,
    pub lufs: f64,
}

/// Complete 4-tier evaluation report for one qualification level (`L1`, `L2`, or `L3` setting).
#[derive(Clone, Debug)]
pub(crate) struct LevelEvaluationReport {
    pub level_name: String,
    pub tiers_by_channel: Vec<[TierChannelReport; 4]>,
    pub monotone_ok: Vec<bool>,
    pub lufs_step_lu: Vec<f64>,
    pub lufs_ok: Vec<bool>,
    pub zero_dc_ok: bool,
    pub passes: bool,
    pub failure_reasons: Vec<String>,
}

/// Per-setting report inside Level 3 multi-control/switch qualification.
#[derive(Clone, Debug)]
pub(crate) struct Level3SettingReport {
    pub label: String,
    pub frozen_ultra_besr_db: f64,
    pub eval: LevelEvaluationReport,
}

/// Aggregated `L1` / `L2` / `L3` qualification outcome for a fixture & compute backend.
#[derive(Clone, Debug)]
pub(crate) struct FixtureQualificationReport {
    pub fixture_id: String,
    pub num_controls: usize,
    pub backend: ComputeBackendPreference,
    pub level1: LevelEvaluationReport,
    pub level2: LevelEvaluationReport,
    pub level3: Option<Vec<Level3SettingReport>>,
    pub overall_pass: bool,
}

/// Parameters controlling [`evaluate_tier_outputs_4metric`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct EvalMatrixConfig<'a> {
    pub level_name: &'a str,
    pub require_target_floors: bool,
    pub require_monotone: bool,
    pub require_lufs_parity: bool,
    pub sample_rate_hz: f64,
    pub warmup: usize,
    pub zero_dc_ok: bool,
}

pub(crate) struct TierRenderOutput {
    pub models_by_channel: Vec<[Vec<f64>; 4]>,
    pub lufs_by_channel: Vec<[f64; 4]>,
    pub zero_dc_ok: bool,
}

pub(crate) fn render_dnam_tiers(
    proj_bytes: &[u8],
    n_out: usize,
    dry_l: &[f32],
    dry_r: Option<&[f32]>,
    ctrls: &[f32],
    warmup: usize,
    check_zero_dc_passive: bool,
) -> Result<TierRenderOutput, DnamError> {
    let ch_out = n_out.clamp(1, 2);
    let mono_only = ch_out < 2;
    let n = dry_l.len();
    let mut models_by_channel: Vec<[Vec<f64>; 4]> = (0..ch_out)
        .map(|_| std::array::from_fn(|_| Vec::with_capacity(n)))
        .collect();
    let mut lufs_by_channel: Vec<[f64; 4]> = vec![[0.0; 4]; ch_out];
    let mut zero_dc_ok = true;

    for (t_idx, &tier) in ALL_TIERS.iter().enumerate() {
        let tier_dnam = export_runtime_dnam(proj_bytes, Some(tier))?;
        let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned(&tier_dnam, tier, 48_000)?;
        eng.reset_to_dc_equilibrium();
        for (p, &v) in ctrls.iter().enumerate() {
            eng.set_control_immediate(p, v);
        }
        if check_zero_dc_passive {
            for _ in 0..256 {
                let yz = eng.step_frame_sona(0.0, 0.0, mono_only);
                if !yz[0].is_finite()
                    || yz[0].abs() >= 1.0e-5
                    || (!mono_only && (!yz[1].is_finite() || yz[1].abs() >= 1.0e-5))
                {
                    zero_dc_ok = false;
                }
            }
            eng.reset_to_dc_equilibrium();
            for (p, &v) in ctrls.iter().enumerate() {
                eng.set_control_immediate(p, v);
            }
        }
        let mut lufs_frames: Vec<Vec<[f32; 2]>> =
            vec![Vec::with_capacity(n.saturating_sub(warmup)); ch_out];
        for i in 0..n {
            let dl = dry_l[i];
            let dr = dry_r.map_or(dl, |r| r[i]);
            let y = eng.step_frame_sona(dl, dr, mono_only);
            for c in 0..ch_out {
                models_by_channel[c][t_idx].push(f64::from(y[c]));
                if i >= warmup {
                    lufs_frames[c].push([y[c], y[c]]);
                }
            }
        }
        for c in 0..ch_out {
            lufs_by_channel[c][t_idx] = compute_integrated_lufs(&lufs_frames[c], 48_000);
        }
    }

    Ok(TierRenderOutput {
        models_by_channel,
        lufs_by_channel,
        zero_dc_ok,
    })
}

/// Evaluates `[Lite, Full, Pro, Ultra]` outputs across every output channel under canonical 4-metric
/// certification, monotonicity, cross-tier loudness parity, and zero-DC stability.
#[must_use]
pub(crate) fn evaluate_tier_outputs_4metric(
    cfg: EvalMatrixConfig<'_>,
    targets_by_channel: &[Vec<f64>],
    dry_by_channel: Option<&[Vec<f64>]>,
    models_by_channel: &[[Vec<f64>; 4]],
    profiles: &[TargetRepeatabilityProfile],
    lufs_by_channel: &[[f64; 4]],
) -> LevelEvaluationReport {
    let n_ch = targets_by_channel.len().min(models_by_channel.len());
    let mut tiers_by_channel = Vec::with_capacity(n_ch);
    let mut monotone_ok = Vec::with_capacity(n_ch);
    let mut lufs_step_lu = Vec::with_capacity(n_ch);
    let mut lufs_ok = Vec::with_capacity(n_ch);
    let mut failure_reasons = Vec::new();
    if !cfg.zero_dc_ok {
        failure_reasons.push(format!(
            "{}: zero-DC / silence stability failed",
            cfg.level_name
        ));
    }

    for c in 0..n_ch {
        let prof = profiles.get(c).unwrap_or(&profiles[0]);
        let tgt = &targets_by_channel[c];
        let mut reports: Vec<TierChannelReport> = Vec::with_capacity(4);
        for t_idx in 0..4 {
            let tier = ALL_TIERS[t_idx];
            let mdl = &models_by_channel[c][t_idx];
            let summary = if let Some(drys) = dry_by_channel
                && let Some(dry_ch) = drys.get(c)
            {
                certify_4metric_universal_with_dry_gaps(
                    tier,
                    dry_ch,
                    tgt,
                    mdl,
                    prof,
                    cfg.sample_rate_hz,
                    cfg.warmup,
                )
            } else {
                certify_4metric_universal(tier, tgt, mdl, prof, cfg.sample_rate_hz, cfg.warmup)
            };
            let (holdout_besr_db, holdout_pass) = if dry_by_channel.is_some() {
                (summary.besr_db, summary.passes_target)
            } else {
                certify_holdout_universal(tier, tgt, mdl, prof, cfg.sample_rate_hz, cfg.warmup)
            };
            let incoherent = prof.band_inc_ratio.iter().any(|&r| r > 0.0)
                || summary.qualified_band_mask.iter().any(|&q| q);
            let tier_pass = if cfg.require_target_floors {
                summary.passes_target && (incoherent || holdout_pass)
            } else if incoherent {
                holdout_pass
            } else {
                holdout_pass && summary.passes_hard_ceiling
            };
            if !tier_pass {
                failure_reasons.push(format!(
                    "{} ch{c} {:?}: besr={:.2} dB holdout={:.2} dB ltas={:.3} dB mr={:.3} seg={:.2} (passes_target={} ceiling={} holdout_ok={})",
                    cfg.level_name,
                    tier,
                    summary.besr_db,
                    holdout_besr_db,
                    summary.ltas_max_db,
                    summary.mr_iir,
                    summary.seg_esr_db,
                    summary.passes_target,
                    summary.passes_hard_ceiling,
                    holdout_pass
                ));
            }
            reports.push(TierChannelReport {
                tier,
                channel: c,
                summary,
                holdout_besr_db,
                holdout_pass,
                lufs: lufs_by_channel.get(c).map_or(0.0, |row| row[t_idx]),
            });
        }
        let arr: [TierChannelReport; 4] = reports.try_into().expect("4 tiers");
        let summaries: [PerceptualCertificationSummary; 4] = [
            arr[0].summary.clone(),
            arr[1].summary.clone(),
            arr[2].summary.clone(),
            arr[3].summary.clone(),
        ];
        let mono = verify_monotone_4tier_summaries(&summaries);
        if cfg.require_monotone && !mono {
            failure_reasons.push(format!(
                "{} ch{c}: 4-tier monotone non-regression failed (raw={:?}, ltas={:?}, mr={:?}, seg={:?})",
                cfg.level_name,
                [
                    arr[0].summary.raw_besr_db,
                    arr[1].summary.raw_besr_db,
                    arr[2].summary.raw_besr_db,
                    arr[3].summary.raw_besr_db,
                ],
                [
                    arr[0].summary.ltas_max_db,
                    arr[1].summary.ltas_max_db,
                    arr[2].summary.ltas_max_db,
                    arr[3].summary.ltas_max_db,
                ],
                [
                    arr[0].summary.mr_iir,
                    arr[1].summary.mr_iir,
                    arr[2].summary.mr_iir,
                    arr[3].summary.mr_iir,
                ],
                [
                    arr[0].summary.seg_esr_db,
                    arr[1].summary.seg_esr_db,
                    arr[2].summary.seg_esr_db,
                    arr[3].summary.seg_esr_db,
                ]
            ));
        }
        let row_lufs = lufs_by_channel.get(c).copied().unwrap_or([0.0; 4]);
        let mut max_step = 0.0_f64;
        let mut lufs_valid = row_lufs.iter().all(|v| v.is_finite());
        for t in 0..3 {
            let d = (row_lufs[t + 1] - row_lufs[t]).abs();
            if !d.is_finite() {
                lufs_valid = false;
            } else {
                max_step = max_step.max(d);
            }
        }
        let step_ok = lufs_valid && max_step <= MAX_TIER_LOUDNESS_STEP_LU;
        if cfg.require_lufs_parity && !step_ok {
            failure_reasons.push(format!(
                "{} ch{c}: LUFS parity step {max_step:.4} > {MAX_TIER_LOUDNESS_STEP_LU:.2}",
                cfg.level_name
            ));
        }
        monotone_ok.push(mono);
        lufs_step_lu.push(max_step);
        lufs_ok.push(step_ok);
        tiers_by_channel.push(arr);
    }

    let passes = failure_reasons.is_empty();
    LevelEvaluationReport {
        level_name: cfg.level_name.to_string(),
        tiers_by_channel,
        monotone_ok,
        lufs_step_lu,
        lufs_ok,
        zero_dc_ok: cfg.zero_dc_ok,
        passes,
        failure_reasons,
    }
}

pub(crate) fn log_level_summary(fixture_id: &str, rep: &LevelEvaluationReport, suffix: &str) {
    for row in &rep.tiers_by_channel {
        let ch = row[0].channel;
        println!(
            "[M25-QUAL] {fixture_id} [{}] ch{} tiers=[{:?},{:?},{:?},{:?}] bESR=[{:.2}, {:.2}, {:.2}, {:.2}]dB hold=[{:.2}({}),{:.2}({}),{:.2}({}),{:.2}({})] LTAS=[{:.3}, {:.3}, {:.3}, {:.3}]dB MR=[{:.3}, {:.3}, {:.3}, {:.3}] LUFS=[{:.2},{:.2},{:.2},{:.2}](d={:.4},ok={}) mono={} zdc={} pass={}{}",
            rep.level_name,
            ch,
            row[0].tier,
            row[1].tier,
            row[2].tier,
            row[3].tier,
            row[0].summary.besr_db,
            row[1].summary.besr_db,
            row[2].summary.besr_db,
            row[3].summary.besr_db,
            row[0].holdout_besr_db,
            row[0].holdout_pass,
            row[1].holdout_besr_db,
            row[1].holdout_pass,
            row[2].holdout_besr_db,
            row[2].holdout_pass,
            row[3].holdout_besr_db,
            row[3].holdout_pass,
            row[0].summary.ltas_max_db,
            row[1].summary.ltas_max_db,
            row[2].summary.ltas_max_db,
            row[3].summary.ltas_max_db,
            row[0].summary.mr_iir,
            row[1].summary.mr_iir,
            row[2].summary.mr_iir,
            row[3].summary.mr_iir,
            row[0].lufs,
            row[1].lufs,
            row[2].lufs,
            row[3].lufs,
            rep.lufs_step_lu[ch],
            rep.lufs_ok[ch],
            rep.monotone_ok[ch],
            rep.zero_dc_ok,
            rep.passes,
            suffix,
        );
    }
}

/// Asserts that `report` passed every applicable level (`L1`, `L2`, and `L3` when `P > 0`),
/// formatting all diagnostic failures clearly.
pub(crate) fn assert_qualification_passed(report: &FixtureQualificationReport) {
    if report.overall_pass {
        return;
    }
    let mut all_msgs = Vec::new();
    all_msgs.extend(report.level1.failure_reasons.iter().cloned());
    all_msgs.extend(report.level2.failure_reasons.iter().cloned());
    if let Some(ref l3) = report.level3 {
        for s in l3 {
            for r in &s.eval.failure_reasons {
                all_msgs.push(format!("[{}] {r}", s.label));
            }
        }
    }
    panic!(
        "Fixture {} ({:?}, P={}) failed 3-level qualification:\n{}",
        report.fixture_id,
        report.backend,
        report.num_controls,
        all_msgs.join("\n")
    );
}
