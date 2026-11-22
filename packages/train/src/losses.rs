//! Composable differentiable perceptual training losses (`bESR`, `MR-IIR`, `SpecOT`, `Gabor`,
//! `IF+GD`, `LTAS`, `SegESR`, ultrasonic stopband) and canonical 4-tier quality certification
//! (`ref[qual.companion.suite]`, `ref[qual.tiers.benchmark-matrix]`, `ref[train.pipeline.backbone-trajectory-opt]`).

pub mod bezout;
pub mod fft;
pub mod ltas;
pub mod mr_iir;
pub mod phase_ot;
pub mod spectral;

use dnam_vm::Tier;

use crate::error::TrainError;

pub use bezout::{
    BESR_WORST_BAND_OFFSET_DB, BesrChannelReport, BesrIntervalBounds, BesrIntervalDecision,
    BesrJointIntervalBounds, Bezout7FilterBank, NoiseQualification, ONE_DB_POWER_FACTOR,
    compute_besr_channel, compute_seg_esr_channel_db, eval_seg_esr_loss_and_adjoint,
    evaluate_joint_interval_besr, evaluate_noise_qualified_besr, evaluate_uncertain_floor_besr,
};
pub use fft::{Radix2FftPlan, build_hann_window, compute_welch_seg_starts};
pub use ltas::{
    LtasChannelReport, LtasInterval, LtasLossMode, Welch33TargetCache,
    canonical_ltas_33_centers_hz, canonical_ltas_33_intervals_hz, compute_ltas_channel,
    eval_ltas_loss_and_adjoint, eval_nyquist_hinge_and_adjoint,
};
pub use mr_iir::{MrIirTargetCache, compute_mr_iir_channel, eval_mr_iir_loss_and_adjoint};
pub use phase_ot::{
    IfGdTargetCache, SpecOtTargetCache, eval_if_gd_loss_and_adjoint, eval_spec_ot_loss_and_adjoint,
};
pub use spectral::{
    AnalyticGaborBank, GaborBandKernel, GaborTargetCache, K_GABOR, eval_gabor_loss_and_adjoint,
    eval_runner_gabor_bezout_bridge,
};

/// Multi-objective weights composing the SONA v3.0 perceptual loss function
/// (`ref[train.pipeline.backbone-trajectory-opt]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositeLossConfig {
    /// Sample rate in Hz (`48_000.0` canonical).
    pub sample_rate_hz: f64,
    /// Pre-roll warmup frames excluded from score accumulation.
    pub warmup_frames: usize,
    /// Degree-7 Bézout `bESR` weight.
    pub w_besr: f64,
    /// 5-scale 32-probe `MR-IIR` weight.
    pub w_mr_iir: f64,
    /// 1-D spectral optimal transport (`SpecOT`) weight.
    pub w_spec_ot: f64,
    /// 33-band constant-$Q$ `Gabor` envelope/tilt weight.
    pub w_gabor: f64,
    /// Instantaneous frequency + group delay (`IF+GD`) weight.
    pub w_if_gd: f64,
    /// 33-band Welch `LTAS` loss weight.
    pub w_ltas: f64,
    /// Welch `LTAS` mode.
    pub ltas_mode: LtasLossMode,
    /// `20 ms` segmental `SegESR` envelope hinge weight.
    pub w_seg_esr: f64,
    /// Ultrasonic `[20 kHz, fs/2]` excess-energy stopband weight.
    pub w_nyquist: f64,
    /// STFT FFT length for `SpecOT` and `IF+GD`.
    pub stft_fft_size: usize,
}

impl Default for CompositeLossConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000.0,
            warmup_frames: 64,
            w_besr: 1.0,
            w_mr_iir: 0.35,
            w_spec_ot: 0.25,
            w_gabor: 0.35,
            w_if_gd: 0.25,
            w_ltas: 0.40,
            ltas_mode: LtasLossMode::LogSumExpMinimax {
                beta: 4.0,
                goal_db: 0.20,
            },
            w_seg_esr: 0.20,
            w_nyquist: 0.15,
            stft_fft_size: 256,
        }
    }
}

/// Per-component breakdown returned alongside total composite scalar loss.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CompositeLossBreakdown {
    /// Weighted sum over all enabled objective terms.
    pub total: f64,
    /// Unweighted Bézout 3-band `bESR` component.
    pub besr: f64,
    /// Unweighted 5-scale `MR-IIR` component.
    pub mr_iir: f64,
    /// Unweighted `SpecOT` Wasserstein component.
    pub spec_ot: f64,
    /// Unweighted 33-band `Gabor` component.
    pub gabor: f64,
    /// Unweighted `IF+GD` phase coherence component.
    pub if_gd: f64,
    /// Unweighted 33-band Welch `LTAS` component.
    pub ltas: f64,
    /// Unweighted `SegESR` component.
    pub seg_esr: f64,
    /// Unweighted ultrasonic `[20 kHz, fs/2]` hinge component.
    pub nyquist: f64,
}

/// Reusable stereo composite loss & reverse-mode time-domain adjoint evaluator.
#[derive(Clone, Debug)]
pub struct CompositeLossEvaluator {
    config: CompositeLossConfig,
    len: usize,
    bank: Bezout7FilterBank,
    gabor_bank: AnalyticGaborBank,
    target_l: Vec<f64>,
    target_r: Vec<f64>,
    ch_l: ChannelCachesWrapper,
    ch_r: ChannelCachesWrapper,
}

#[derive(Clone, Debug)]
struct ChannelCachesWrapper {
    bezout_target_e: [f64; 3],
    mr_iir: Option<MrIirTargetCache>,
    ltas: Option<Welch33TargetCache>,
    gabor: GaborTargetCache,
    spec_ot: Option<SpecOtTargetCache>,
    if_gd: Option<IfGdTargetCache>,
}

impl ChannelCachesWrapper {
    fn from_channel(
        target: &[f64],
        cfg: &CompositeLossConfig,
        bank: &Bezout7FilterBank,
        gabor_bank: &AnalyticGaborBank,
    ) -> Result<Self, TrainError> {
        let w = cfg.warmup_frames.min(target.len());
        let mut tmp = [Vec::new(), Vec::new(), Vec::new()];
        let bezout_target_e = bank.split_3band_and_energies(target, w, &mut tmp);
        let mr_iir = (cfg.w_mr_iir > 0.0)
            .then(|| MrIirTargetCache::new(target, None, w, cfg.sample_rate_hz))
            .flatten();
        let ltas = (cfg.w_ltas > 0.0 || cfg.w_nyquist > 0.0)
            .then(|| Welch33TargetCache::new(target, w, cfg.sample_rate_hz).ok())
            .flatten();
        let gabor = GaborTargetCache::new(target, w, gabor_bank);
        let spec_ot = (cfg.w_spec_ot > 0.0)
            .then(|| SpecOtTargetCache::new(target, w, cfg.sample_rate_hz, cfg.stft_fft_size).ok())
            .flatten();
        let if_gd = (cfg.w_if_gd > 0.0)
            .then(|| IfGdTargetCache::new(target, w, cfg.sample_rate_hz, cfg.stft_fft_size).ok())
            .flatten();
        Ok(Self {
            bezout_target_e,
            mr_iir,
            ltas,
            gabor,
            spec_ot,
            if_gd,
        })
    }
}

impl CompositeLossEvaluator {
    /// Builds target caches for a stereo reference capsule pair `(target_l, target_r)`.
    pub fn new(
        target_l: &[f64],
        target_r: &[f64],
        config: CompositeLossConfig,
    ) -> Result<Self, TrainError> {
        let len = target_l.len();
        if len == 0 || target_r.len() != len {
            return Err(TrainError::InvalidDimensions {
                context: "CompositeLossEvaluator::new",
                expected: len.max(1),
                actual: target_r.len(),
            });
        }
        let bank = Bezout7FilterBank::new(config.sample_rate_hz);
        let gabor_bank = AnalyticGaborBank::new(config.sample_rate_hz);
        let ch_l = ChannelCachesWrapper::from_channel(target_l, &config, &bank, &gabor_bank)?;
        let ch_r = ChannelCachesWrapper::from_channel(target_r, &config, &bank, &gabor_bank)?;
        Ok(Self {
            config,
            len,
            bank,
            gabor_bank,
            target_l: target_l.to_vec(),
            target_r: target_r.to_vec(),
            ch_l,
            ch_r,
        })
    }

    fn eval_channel(
        &self,
        pred: &[f64],
        target: &[f64],
        caches: &ChannelCachesWrapper,
        mut grad_opt: Option<&mut [f64]>,
    ) -> CompositeLossBreakdown {
        let c = &self.config;
        let w = c.warmup_frames.min(self.len);
        let mut local_g = vec![0.0_f64; self.len];
        let accum = |scale: f64, g_dst: Option<&mut [f64]>, src: &[f64]| {
            if let Some(dst) = g_dst {
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d += scale * s;
                }
            }
        };

        let mut besr = 0.0;
        if c.w_besr > 0.0 {
            local_g.fill(0.0);
            besr = self.bank.eval_besr_loss_and_adjoint(
                pred,
                target,
                caches.bezout_target_e,
                [1.0e-6; 3],
                w,
                grad_opt.is_some().then_some(&mut local_g),
            );
            accum(c.w_besr, grad_opt.as_deref_mut(), &local_g);
        }

        let mut mr_iir = 0.0;
        if c.w_mr_iir > 0.0
            && let Some(ref mc) = caches.mr_iir
        {
            local_g.fill(0.0);
            mr_iir =
                eval_mr_iir_loss_and_adjoint(mc, pred, grad_opt.is_some().then_some(&mut local_g));
            accum(c.w_mr_iir, grad_opt.as_deref_mut(), &local_g);
        }

        let mut gabor = 0.0;
        if c.w_gabor > 0.0 {
            local_g.fill(0.0);
            gabor = eval_gabor_loss_and_adjoint(
                &self.gabor_bank,
                &caches.gabor,
                pred,
                grad_opt.is_some().then_some(&mut local_g),
            );
            accum(c.w_gabor, grad_opt.as_deref_mut(), &local_g);
        }

        let mut spec_ot = 0.0;
        if c.w_spec_ot > 0.0
            && let Some(ref sc) = caches.spec_ot
        {
            local_g.fill(0.0);
            spec_ot =
                eval_spec_ot_loss_and_adjoint(sc, pred, grad_opt.is_some().then_some(&mut local_g));
            accum(c.w_spec_ot, grad_opt.as_deref_mut(), &local_g);
        }

        let mut if_gd = 0.0;
        if c.w_if_gd > 0.0
            && let Some(ref ic) = caches.if_gd
        {
            local_g.fill(0.0);
            if_gd =
                eval_if_gd_loss_and_adjoint(ic, pred, grad_opt.is_some().then_some(&mut local_g));
            accum(c.w_if_gd, grad_opt.as_deref_mut(), &local_g);
        }

        let mut ltas = 0.0;
        let mut nyquist = 0.0;
        if (c.w_ltas > 0.0 || c.w_nyquist > 0.0)
            && let Some(ref lc) = caches.ltas
        {
            if c.w_ltas > 0.0 {
                local_g.fill(0.0);
                ltas = eval_ltas_loss_and_adjoint(
                    lc,
                    pred,
                    c.ltas_mode,
                    0.0,
                    grad_opt.is_some().then_some(&mut local_g),
                );
                accum(c.w_ltas, grad_opt.as_deref_mut(), &local_g);
            }
            if c.w_nyquist > 0.0 {
                local_g.fill(0.0);
                nyquist = eval_nyquist_hinge_and_adjoint(
                    lc,
                    pred,
                    1.0,
                    grad_opt.is_some().then_some(&mut local_g),
                );
                accum(c.w_nyquist, grad_opt.as_deref_mut(), &local_g);
            }
        }

        let mut seg_esr = 0.0;
        if c.w_seg_esr > 0.0 {
            local_g.fill(0.0);
            seg_esr = eval_seg_esr_loss_and_adjoint(
                pred,
                target,
                c.sample_rate_hz,
                w,
                grad_opt.is_some().then_some(&mut local_g),
            );
            accum(c.w_seg_esr, grad_opt, &local_g);
        }

        let total = c.w_besr * besr
            + c.w_mr_iir * mr_iir
            + c.w_spec_ot * spec_ot
            + c.w_gabor * gabor
            + c.w_if_gd * if_gd
            + c.w_ltas * ltas
            + c.w_seg_esr * seg_esr
            + c.w_nyquist * nyquist;
        CompositeLossBreakdown {
            total,
            besr,
            mr_iir,
            spec_ot,
            gabor,
            if_gd,
            ltas,
            seg_esr,
            nyquist,
        }
    }

    /// Evaluates stereo composite loss and optionally fills exact reverse-mode gradients `[d_l, d_r]`.
    pub fn eval_and_adjoint(
        &self,
        pred_l: &[f64],
        pred_r: &[f64],
        mut grad_l_opt: Option<&mut [f64]>,
        mut grad_r_opt: Option<&mut [f64]>,
    ) -> Result<CompositeLossBreakdown, TrainError> {
        if pred_l.len() != self.len || pred_r.len() != self.len {
            return Err(TrainError::InvalidDimensions {
                context: "CompositeLossEvaluator::eval_and_adjoint",
                expected: self.len,
                actual: pred_l.len().min(pred_r.len()),
            });
        }
        let mut bl = self.eval_channel(
            pred_l,
            &self.target_l,
            &self.ch_l,
            grad_l_opt.as_deref_mut(),
        );
        let br = self.eval_channel(
            pred_r,
            &self.target_r,
            &self.ch_r,
            grad_r_opt.as_deref_mut(),
        );
        if let Some(gl) = grad_l_opt {
            for v in &mut gl[..self.len] {
                *v *= 0.5;
            }
        }
        if let Some(gr) = grad_r_opt {
            for v in &mut gr[..self.len] {
                *v *= 0.5;
            }
        }
        bl.total = 0.5 * (bl.total + br.total);
        bl.besr = 0.5 * (bl.besr + br.besr);
        bl.mr_iir = 0.5 * (bl.mr_iir + br.mr_iir);
        bl.spec_ot = 0.5 * (bl.spec_ot + br.spec_ot);
        bl.gabor = 0.5 * (bl.gabor + br.gabor);
        bl.if_gd = 0.5 * (bl.if_gd + br.if_gd);
        bl.ltas = 0.5 * (bl.ltas + br.ltas);
        bl.seg_esr = 0.5 * (bl.seg_esr + br.seg_esr);
        bl.nyquist = 0.5 * (bl.nyquist + br.nyquist);
        Ok(bl)
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
    use super::*;

    #[test]
    fn test_composite_loss_fd_and_4metric_tier_monotonicity() {
        let n = 1_200_usize;
        let sr = 48_000.0_f64;
        let tau = core::f64::consts::TAU;
        let centers = canonical_ltas_33_centers_hz();
        let synth = |phase: f64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let t = (i as f64) / sr;
                    let mut s = 0.02 * (tau * 22_200.0 * t + phase).sin();
                    for (b, &fc) in centers.iter().enumerate() {
                        s += 0.07 * (tau * fc * t + phase + 0.37 * (b as f64)).sin();
                    }
                    s
                })
                .collect()
        };
        let tgt_l = synth(0.0);
        let tgt_r = synth(0.7);
        let pred_l: Vec<f64> = tgt_l
            .iter()
            .enumerate()
            .map(|(i, &v)| 0.88 * v + 0.20 * (tau * 22_200.0 * ((i as f64) / sr)).sin())
            .collect();
        let pred_r: Vec<f64> = tgt_r
            .iter()
            .enumerate()
            .map(|(i, &v)| 0.90 * v - 0.18 * (tau * 22_200.0 * ((i as f64) / sr)).cos())
            .collect();

        let eval =
            CompositeLossEvaluator::new(&tgt_l, &tgt_r, CompositeLossConfig::default()).unwrap();
        let (mut gl, mut gr) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let bd = eval
            .eval_and_adjoint(&pred_l, &pred_r, Some(&mut gl), Some(&mut gr))
            .unwrap();
        assert!(
            bd.total.is_finite()
                && bd.besr > 0.0
                && bd.mr_iir > 0.0
                && bd.spec_ot > 0.0
                && bd.ltas > 0.0
                && bd.nyquist > 0.0
        );

        let eps = 1.0e-6_f64;
        for idx in [150, 500, 900] {
            let (mut ppl, mut pml) = (pred_l.clone(), pred_l.clone());
            ppl[idx] += eps;
            pml[idx] -= eps;
            let fd_l = (eval
                .eval_and_adjoint(&ppl, &pred_r, None, None)
                .unwrap()
                .total
                - eval
                    .eval_and_adjoint(&pml, &pred_r, None, None)
                    .unwrap()
                    .total)
                / (2.0 * eps);
            assert!(
                (gl[idx] - fd_l).abs() <= 1.0e-5 * gl[idx].abs().max(1.0e-3),
                "Composite L idx={idx} ana={} fd={fd_l}",
                gl[idx]
            );
        }

        // Verify 4-tier monotone non-regression checker across strictly improving approximations.
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
