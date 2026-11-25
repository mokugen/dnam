//! 4-tier quality floor contracts and zero-slack adjacent-tier monotone non-regression checks
//! (`ref[qual.companion.suite]`, `ref[qual.tiers.benchmark-matrix]`).

use dnam_vm::Tier;

use crate::error::TrainError;

use super::bezout::{
    compute_besr_channel, compute_seg_esr_channel_db_with_gaps, evaluate_noise_qualified_besr,
};
use super::ltas::{Welch33TargetCache, compute_ltas_channel};
use super::mr_iir::{MrIirTargetCache, compute_mr_iir_channel_with_gaps};

/// Schroeder `T60` (`-60 dB`) amplitude completion threshold $10^{-60/20} = 10^{-3}$ relative to
/// active target RMS used to detect dead-silent post-decay interiors inside dry-silent gaps.
pub const POST_T60_GAP_AMP_RATIO: f64 = 1.0 / 1_000.0;

/// Relative target-band amplitude threshold ($10^{-3} = -60\text{ dB}$ vs frame broadband RMS) below
/// which a noiseless `SegESR` band in an active 20 ms frame is treated as an unexcited crossover skirt.
pub const SEG_ESR_UNEXCITED_BAND_AMP_RATIO: f64 = 1.0 / 1_000.0;

/// Strict residual guard ($10^{-2} = -40\text{ dB}$ vs frame broadband RMS) enforced on unexcited
/// noiseless `SegESR` bands (`|E_b(y_hat) - E_b(y)| <= 10^-2 E_bb(y)`).
pub const SEG_ESR_UNEXCITED_BAND_GUARD_RATIO: f64 = 1.0 / 100.0;

/// Minimum contiguous dry-silence duration (`20 ms`, matching one `SegESR` frame) eligible for
/// post-`T60` gap detection inside multi-regime stimuli such as `Schoukens–Pintelon` `Regime 2`.
pub const MIN_DRY_SILENT_GAP_SECS: f64 = 1.0 / 50.0;

/// Exact/numeric silence floor on dry stimulus samples (`|dry| <= DRY_SILENCE_EPS`).
pub const DRY_SILENCE_EPS: f64 = 1.0e-9;

pub(crate) const MIN_POST_T60_GAP_SAMPLES: usize = 256;
pub(crate) const GAP_PRERING_TRIM_MAX_SAMPLES: usize = 64;

const _: () = assert!(
    POST_T60_GAP_AMP_RATIO > 0.0
        && POST_T60_GAP_AMP_RATIO <= 1.0e-3
        && SEG_ESR_UNEXCITED_BAND_AMP_RATIO == POST_T60_GAP_AMP_RATIO
        && SEG_ESR_UNEXCITED_BAND_GUARD_RATIO == 10.0 * SEG_ESR_UNEXCITED_BAND_AMP_RATIO
        && MIN_DRY_SILENT_GAP_SECS > 0.0
        && DRY_SILENCE_EPS > 0.0
        && MIN_POST_T60_GAP_SAMPLES == 256
        && GAP_PRERING_TRIM_MAX_SAMPLES == 64
        && GAP_PRERING_TRIM_MAX_SAMPLES < MIN_POST_T60_GAP_SAMPLES
);

/// Post-`T60` dead-silent `[start, end)` sample intervals detected purely from `(dry, target)` inside
/// `>= 20 ms` dry-silent gaps where target output has decayed below `-60 dB` of active RMS.
///
/// Used exclusively on noiseless (`noise == None`) captures so `f32` IIR sub-`-100 dBFS` DC-blocker /
/// modal tails inside zero-excitation `Regime 2` decay gaps do not trigger spurious log-underflow
/// penalties while enforcing a strict `-60 dB` residual ceiling throughout every gap.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrySilentGapMask {
    /// Sorted disjoint `[start, end)` sample spans inside `[warmup, n)`.
    pub spans: Vec<(usize, usize)>,
    /// Post-warmup target RMS computed across active (`|dry| > DRY_SILENCE_EPS`) samples.
    pub active_target_rms: f64,
}

impl DrySilentGapMask {
    /// Detects post-`T60` dead-silent intervals strictly from aligned `(dry, target)` pairs.
    #[must_use]
    pub fn detect(dry: &[f64], target: &[f64], warmup: usize, sample_rate_hz: f64) -> Self {
        let n = dry.len().min(target.len());
        if n <= warmup
            || !(8_000.0..=1_536_000.0).contains(&sample_rate_hz)
            || !sample_rate_hz.is_finite()
        {
            return Self::default();
        }
        let mut active_sq_sum = 0.0_f64;
        let mut active_count = 0_usize;
        for i in warmup..n {
            let d = dry[i];
            let y = target[i];
            if !d.is_finite() || !y.is_finite() {
                return Self::default();
            }
            if d.abs() > DRY_SILENCE_EPS {
                active_sq_sum += y * y;
                active_count += 1;
            }
        }
        if active_count == 0 || !active_sq_sum.is_finite() {
            return Self::default();
        }
        let active_target_rms = (active_sq_sum / (active_count as f64)).sqrt();
        if !active_target_rms.is_finite() || active_target_rms <= 1.0e-12 {
            return Self::default();
        }

        let min_gap = (sample_rate_hz * MIN_DRY_SILENT_GAP_SECS).round().max(1.0) as usize;
        let t60_thresh = POST_T60_GAP_AMP_RATIO * active_target_rms;
        let mut spans = Vec::new();
        let mut i = warmup;
        while i < n {
            if dry[i].abs() <= DRY_SILENCE_EPS {
                let s_dry = i;
                while i < n && dry[i].abs() <= DRY_SILENCE_EPS {
                    i += 1;
                }
                let e_dry = i;
                if e_dry - s_dry >= min_gap {
                    let mut e_end = e_dry;
                    if e_dry < n {
                        let ring_lo = e_dry
                            .saturating_sub(GAP_PRERING_TRIM_MAX_SAMPLES)
                            .max(s_dry);
                        if let Some(first_ring) =
                            (ring_lo..e_dry).find(|&j| target[j].abs() > t60_thresh)
                        {
                            e_end = first_ring;
                        }
                    }
                    let mut last_above = None;
                    for j in (s_dry..e_end).rev() {
                        if target[j].abs() > t60_thresh {
                            last_above = Some(j);
                            break;
                        }
                    }
                    let s_t60 = last_above.map_or(s_dry, |idx| idx + 1);
                    if e_end.saturating_sub(s_t60) >= MIN_POST_T60_GAP_SAMPLES {
                        spans.push((s_t60, e_end));
                    }
                }
            } else {
                i += 1;
            }
        }
        Self {
            spans,
            active_target_rms,
        }
    }

    /// Returns `true` when no post-`T60` dead-silent gap spans were detected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
            || !self.active_target_rms.is_finite()
            || self.active_target_rms <= 0.0
    }

    /// Returns the first span `(s, e)` overlapping half-open `[start, end)`, if any.
    #[must_use]
    pub fn overlap_span(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        if start >= end {
            return None;
        }
        self.spans
            .iter()
            .copied()
            .find(|&(s, e)| end > s && start < e)
    }
}

/// Normative 4-tier quality benchmark floors (`ref[qual.tiers.benchmark-matrix]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierQualityFloors {
    /// Target tier.
    pub tier: Tier,
    /// Composite `bESR` ceiling in dB (`-18 / -23 / -30 / -34`).
    pub composite_besr_db: f64,
    /// Per-band `[LF, MF, HF]` ceilings in dB.
    pub per_band_besr_db: [f64; 3],
    /// Maximum absolute `LTAS` deviation in dB (`0.75 / 0.35 / 0.15 / 0.10`).
    pub max_ltas_db: f64,
    /// Unseen multi-control `bESR` ceiling in dB (`-16 / -21 / -27 / -32`).
    pub unseen_control_besr_db: f64,
}

impl TierQualityFloors {
    /// Returns the normative benchmark floors for `tier`.
    #[must_use]
    pub const fn for_tier(tier: Tier) -> Self {
        match tier {
            Tier::Lite => Self {
                tier,
                composite_besr_db: -18.0,
                per_band_besr_db: [-23.0, -20.0, -15.0],
                max_ltas_db: 0.75,
                unseen_control_besr_db: -16.0,
            },
            Tier::Full => Self {
                tier,
                composite_besr_db: -23.0,
                per_band_besr_db: [-27.0, -25.0, -20.0],
                max_ltas_db: 0.35,
                unseen_control_besr_db: -21.0,
            },
            Tier::Pro => Self {
                tier,
                composite_besr_db: -30.0,
                per_band_besr_db: [-34.0, -31.0, -26.0],
                max_ltas_db: 0.15,
                unseen_control_besr_db: -27.0,
            },
            Tier::Ultra => Self {
                tier,
                composite_besr_db: -34.0,
                per_band_besr_db: [-38.0, -35.0, -30.0],
                max_ltas_db: 0.10,
                unseen_control_besr_db: -32.0,
            },
        }
    }
}

/// Complete 4-metric certification summary for one rendered tier channel (`ref[qual.companion.suite]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierCertificationReport {
    /// Evaluated tier.
    pub tier: Tier,
    /// Raw composite `bESR` in dB.
    pub besr_db: f64,
    /// Per-band `bESR` `[LF, MF, HF]` in dB.
    pub band_besr_db: [f64; 3],
    /// 5-scale 32-probe `MR-IIR` loss.
    pub mr_iir: f64,
    /// Maximum 33-band noise-adjusted `LTAS` deviation in dB.
    pub ltas_max_db: f64,
    /// Duration-weighted `20 ms` `SegESR` in dB.
    pub seg_esr_db: f64,
    /// Excluded-band mask under `ref[qual.companion.ltas]`.
    pub excluded_bands: [bool; 33],
    /// Residual Welch band power per band (`W`) for excluded-band monotone checks.
    pub excluded_band_residuals: [f64; 33],
    /// Whether this report meets its [`TierQualityFloors`].
    pub meets_floor: bool,
}

/// Evaluates all 4 canonical metrics (`bESR`, `MR-IIR`, `LTAS`, `SegESR`) for a single tier output
/// channel, optionally gating noiseless post-`T60` dry-silent gaps via `gap_mask`.
pub fn certify_tier_audio_with_gaps(
    tier: Tier,
    pred: &[f64],
    target: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup: usize,
    gap_mask: Option<&DrySilentGapMask>,
) -> Result<TierCertificationReport, TrainError> {
    let n = pred.len().min(target.len());
    if n <= warmup {
        return Err(TrainError::InvalidDimensions {
            context: "certify_tier_audio",
            expected: warmup + 1,
            actual: n,
        });
    }
    let eff_gaps = if noise.is_some() {
        None
    } else {
        gap_mask.filter(|g| !g.is_empty())
    };
    let besr_rep = compute_besr_channel(&pred[..n], &target[..n], noise, sample_rate_hz, warmup);
    let floors = TierQualityFloors::for_tier(tier);
    let nq =
        evaluate_noise_qualified_besr(&besr_rep, floors.per_band_besr_db, floors.composite_besr_db);
    let mr_iir = MrIirTargetCache::new(&target[..n], noise, warmup, sample_rate_hz)
        .map_or(f64::INFINITY, |c| {
            compute_mr_iir_channel_with_gaps(&c, &pred[..n], noise, eff_gaps)
        });
    let ltas_cache = Welch33TargetCache::new(&target[..n], warmup, sample_rate_hz)?;
    let ltas_rep = compute_ltas_channel(&ltas_cache, &pred[..n], noise);
    let seg_esr_db = compute_seg_esr_channel_db_with_gaps(
        &pred[..n],
        &target[..n],
        noise,
        sample_rate_hz,
        warmup,
        eff_gaps,
    );
    let meets_floor = nq.passes
        && !ltas_rep.excluded_band_guard_failed
        && ltas_rep.max_adj_deviation_db <= floors.max_ltas_db;

    Ok(TierCertificationReport {
        tier,
        besr_db: besr_rep.besr_db,
        band_besr_db: besr_rep.band_esr_db,
        mr_iir,
        ltas_max_db: ltas_rep.max_adj_deviation_db,
        seg_esr_db,
        excluded_bands: ltas_rep.excluded_bands,
        excluded_band_residuals: ltas_rep.residual_band_power,
        meets_floor,
    })
}

/// Evaluates all 4 canonical metrics (`bESR`, `MR-IIR`, `LTAS`, `SegESR`) for a single tier output channel.
pub fn certify_tier_audio(
    tier: Tier,
    pred: &[f64],
    target: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup: usize,
) -> Result<TierCertificationReport, TrainError> {
    certify_tier_audio_with_gaps(tier, pred, target, noise, sample_rate_hz, warmup, None)
}

/// Checks monotone non-regression for a single adjacent tier pair `(lo, hi)` across
/// composite/per-band `bESR`, `MR-IIR`, max `LTAS`, excluded-band `LTAS` residuals, and `SegESR`
/// (`ref[qual.tiers.benchmark-matrix]`), rejecting any `NaN` unconditionally.
#[must_use]
pub fn is_tier_pair_monotone(lo: &TierCertificationReport, hi: &TierCertificationReport) -> bool {
    if lo.besr_db.is_nan()
        || hi.besr_db.is_nan()
        || lo.mr_iir.is_nan()
        || hi.mr_iir.is_nan()
        || lo.ltas_max_db.is_nan()
        || hi.ltas_max_db.is_nan()
        || lo.seg_esr_db.is_nan()
        || hi.seg_esr_db.is_nan()
        || hi.besr_db > lo.besr_db
        || hi.mr_iir > lo.mr_iir
        || hi.ltas_max_db > lo.ltas_max_db
        || hi.seg_esr_db > lo.seg_esr_db
        || hi
            .band_besr_db
            .iter()
            .zip(&lo.band_besr_db)
            .any(|(h, l)| h.is_nan() || l.is_nan() || h > l)
    {
        return false;
    }
    for b in 0..33 {
        if lo.excluded_bands[b]
            && (!hi.excluded_band_residuals[b].is_finite()
                || !lo.excluded_band_residuals[b].is_finite()
                || hi.excluded_band_residuals[b] > lo.excluded_band_residuals[b])
        {
            return false;
        }
    }
    true
}

/// Checks zero-slack adjacent-tier monotone non-regression (`Lite <= Full <= Pro <= Ultra`) across
/// composite/per-band `bESR`, `MR-IIR`, max `LTAS`, excluded-band `LTAS` residuals, and `SegESR`
/// (`ref[qual.tiers.benchmark-matrix]`).
#[must_use]
pub fn check_adjacent_tier_monotone(reports: &[TierCertificationReport; 4]) -> bool {
    const EXPECTED: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    for (idx, rep) in reports.iter().enumerate() {
        if rep.tier != EXPECTED[idx]
            || rep.besr_db.is_nan()
            || rep.mr_iir.is_nan()
            || rep.ltas_max_db.is_nan()
            || rep.seg_esr_db.is_nan()
            || rep.band_besr_db.iter().any(|v| v.is_nan())
        {
            return false;
        }
    }
    (0..3).all(|i| is_tier_pair_monotone(&reports[i], &reports[i + 1]))
}

#[cfg(test)]
mod tests {
    use super::super::ltas::canonical_ltas_33_centers_hz;
    use super::*;

    #[test]
    fn test_4metric_tier_certification_and_monotone_checks() {
        let n = 1_200_usize;
        let sr = 48_000.0_f64;
        let tau = core::f64::consts::TAU;
        let centers = canonical_ltas_33_centers_hz();
        let tgt_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                let mut s = 0.02 * (tau * 22_200.0 * t).sin();
                for (b, &fc) in centers.iter().enumerate() {
                    s += 0.07 * (tau * fc * t + 0.37 * (b as f64)).sin();
                }
                s
            })
            .collect();

        let scales = [0.04_f64, 0.02, 0.008, 0.002];
        let mut reports = Vec::with_capacity(4);
        for (tier, sc) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
            .into_iter()
            .zip(scales)
        {
            let p: Vec<f64> = tgt_l.iter().map(|&v| v * (1.0 - sc)).collect();
            reports.push(certify_tier_audio(tier, &p, &tgt_l, None, 48_000.0, 64).unwrap());
        }
        let mut arr: [TierCertificationReport; 4] = reports.try_into().unwrap();
        assert!(arr[3].meets_floor);
        assert!(check_adjacent_tier_monotone(&arr));

        arr[0].excluded_bands[0] = true;
        arr[0].excluded_band_residuals[0] = 1.0e-6;
        arr[1].excluded_band_residuals[0] = 2.0e-6;
        assert!(!check_adjacent_tier_monotone(&arr));
    }

    #[test]
    fn test_dry_silent_gap_mask_detection_and_guards() {
        let sr = 48_000.0_f64;
        let tau = core::f64::consts::TAU;
        let n = 16_384_usize;
        let warmup = 64_usize;

        // Burst -> 250 ms exact-zero gap -> burst.
        let mut dry = vec![0.0_f64; n];
        let mut target = vec![0.0_f64; n];
        for i in 0..n {
            if !(2_400..14_400).contains(&i) {
                let s = 0.25 * (tau * 440.0 * (i as f64) / sr).sin()
                    + 0.15 * (tau * 1_650.0 * (i as f64) / sr).cos();
                dry[i] = s;
                target[i] = 0.9 * s;
            }
        }
        let mask = DrySilentGapMask::detect(&dry, &target, warmup, sr);
        assert_eq!(mask.spans, vec![(2_400, 14_400)]);
        assert!(mask.active_target_rms > 0.15);

        // Sub--100 dBFS IIR/DC tail inside exact-zero gap fails without gap mask, passes with it.
        let mut pred_dc_tail = target.clone();
        for (i, slot) in pred_dc_tail.iter_mut().enumerate().take(14_400).skip(2_400) {
            let decay = (-((i - 2_400) as f64) / 2_000.0).exp();
            *slot = 2.0e-6 * decay;
        }
        let rep_raw = certify_tier_audio(Tier::Ultra, &pred_dc_tail, &target, None, sr, warmup)
            .expect("raw cert");
        assert!(rep_raw.seg_esr_db.is_infinite());

        let rep_gap = certify_tier_audio_with_gaps(
            Tier::Ultra,
            &pred_dc_tail,
            &target,
            None,
            sr,
            warmup,
            Some(&mask),
        )
        .expect("gap cert");
        assert!(rep_gap.seg_esr_db.is_finite());
        assert!(rep_gap.mr_iir.is_finite() && rep_gap.mr_iir < 0.05);
        assert!(rep_gap.meets_floor);

        // Unstable ringing above -60 dB inside dry-silent gap MUST fail SegESR & MR-IIR with inf.
        let mut pred_ring = target.clone();
        for (i, slot) in pred_ring.iter_mut().enumerate().take(14_000).skip(2_600) {
            *slot = 1.0e-2 * (tau * 880.0 * (i as f64) / sr).sin();
        }
        let rep_ring = certify_tier_audio_with_gaps(
            Tier::Ultra,
            &pred_ring,
            &target,
            None,
            sr,
            warmup,
            Some(&mask),
        )
        .expect("ring cert");
        assert!(rep_ring.seg_esr_db.is_infinite());
        assert!(rep_ring.mr_iir.is_infinite());

        // Sustained reverb tail (> -60 dB in gap) leaves gap mask empty.
        let mut reverb_tgt = target.clone();
        for (i, slot) in reverb_tgt.iter_mut().enumerate().take(14_400).skip(2_400) {
            *slot = 0.02 * (tau * 330.0 * (i as f64) / sr).sin();
        }
        let rev_mask = DrySilentGapMask::detect(&dry, &reverb_tgt, warmup, sr);
        assert!(rev_mask.is_empty());

        // Muted target (`target == 0` under active dry) keeps mask empty and rejects nonzero pred.
        let muted = vec![0.0_f64; n];
        let muted_mask = DrySilentGapMask::detect(&dry, &muted, warmup, sr);
        assert!(muted_mask.is_empty());
        let rep_muted = certify_tier_audio_with_gaps(
            Tier::Ultra,
            &pred_dc_tail,
            &muted,
            None,
            sr,
            warmup,
            Some(&muted_mask),
        )
        .expect("muted cert");
        assert!(rep_muted.seg_esr_db.is_infinite() && rep_muted.mr_iir.is_infinite());

        // Explicit noise trace bypasses gap_mask cleanly.
        let noise: Vec<f64> = (0..n)
            .map(|i| 1.0e-4 * (tau * 3_111.0 * (i as f64) / sr).sin())
            .collect();
        let rep_noise = certify_tier_audio_with_gaps(
            Tier::Ultra,
            &pred_dc_tail,
            &target,
            Some(&noise),
            sr,
            warmup,
            Some(&mask),
        )
        .expect("noise cert");
        assert!(rep_noise.mr_iir.is_finite());

        // Trailing Kaiser-64 boundary pre-ringing inside [e_dry - 28 .. e_dry) trims cleanly instead
        // of collapsing the dead-silent interior span to empty.
        let mut prering_tgt = target.clone();
        for (i, slot) in prering_tgt
            .iter_mut()
            .enumerate()
            .take(14_400)
            .skip(14_400 - 28)
        {
            let phase = (i - (14_400 - 28)) as f64;
            *slot = 5.0e-3 * (-0.08 * (28.0 - phase)).exp() * (0.7 * phase).cos();
        }
        let prering_mask = DrySilentGapMask::detect(&dry, &prering_tgt, warmup, sr);
        assert_eq!(prering_mask.spans, vec![(2_400, 14_400 - 28)]);

        // Active > -60 dB tail throughout the entire dry-silent gap leaves spans empty.
        let mut genuine_tail_tgt = target.clone();
        for (i, slot) in genuine_tail_tgt
            .iter_mut()
            .enumerate()
            .take(14_400)
            .skip(2_400)
        {
            *slot = 5.0e-3 * (tau * 110.0 * (i as f64) / sr).sin();
        }
        let genuine_tail_mask = DrySilentGapMask::detect(&dry, &genuine_tail_tgt, warmup, sr);
        assert!(genuine_tail_mask.spans.is_empty());
    }
}
