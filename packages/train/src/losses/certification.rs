//! 4-tier quality floor contracts and zero-slack adjacent-tier monotone non-regression checks
//! (`ref[qual.companion.suite]`, `ref[qual.tiers.benchmark-matrix]`).

use dnam_vm::Tier;

use crate::error::TrainError;

use super::bezout::{
    compute_besr_channel, compute_seg_esr_channel_db, evaluate_noise_qualified_besr,
};
use super::ltas::{Welch33TargetCache, compute_ltas_channel};
use super::mr_iir::{MrIirTargetCache, compute_mr_iir_channel};

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

/// Evaluates all 4 canonical metrics (`bESR`, `MR-IIR`, `LTAS`, `SegESR`) for a single tier output channel.
pub fn certify_tier_audio(
    tier: Tier,
    pred: &[f64],
    target: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup: usize,
) -> Result<TierCertificationReport, TrainError> {
    let n = pred.len().min(target.len());
    if n <= warmup {
        return Err(TrainError::InvalidDimensions {
            context: "certify_tier_audio",
            expected: warmup + 1,
            actual: n,
        });
    }
    let besr_rep = compute_besr_channel(&pred[..n], &target[..n], noise, sample_rate_hz, warmup);
    let floors = TierQualityFloors::for_tier(tier);
    let nq =
        evaluate_noise_qualified_besr(&besr_rep, floors.per_band_besr_db, floors.composite_besr_db);
    let mr_iir = MrIirTargetCache::new(&target[..n], noise, warmup, sample_rate_hz)
        .map_or(f64::INFINITY, |c| {
            compute_mr_iir_channel(&c, &pred[..n], noise)
        });
    let ltas_cache = Welch33TargetCache::new(&target[..n], warmup, sample_rate_hz)?;
    let ltas_rep = compute_ltas_channel(&ltas_cache, &pred[..n], noise);
    let seg_esr_db =
        compute_seg_esr_channel_db(&pred[..n], &target[..n], noise, sample_rate_hz, warmup);
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
}
