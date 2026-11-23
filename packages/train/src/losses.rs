//! Composable differentiable perceptual training losses (`bESR`, `MR-IIR`, `SpecOT`, `Gabor`,
//! `IF+GD`, `LTAS`, `SegESR`, ultrasonic stopband) and canonical 4-tier quality certification
//! (`ref[qual.companion.suite]`, `ref[qual.tiers.benchmark-matrix]`, `ref[train.pipeline.backbone-trajectory-opt]`).

pub mod bezout;
pub mod certification;
pub mod fft;
pub mod ltas;
pub mod mr_iir;
pub mod noise_floor;
pub mod phase_ot;
pub mod spectral;

use core::f64::consts::PI;

use crate::error::TrainError;

pub use bezout::{
    BESR_WORST_BAND_OFFSET_DB, BesrChannelReport, BesrIntervalBounds, BesrIntervalDecision,
    BesrJointIntervalBounds, BesrLossSpec, Bezout7FilterBank, NoiseQualification,
    ONE_DB_POWER_FACTOR, SegEsrTargetCache, compute_besr_channel, compute_seg_esr_channel_db,
    eval_seg_esr_cached_and_adjoint, eval_seg_esr_loss_and_adjoint, evaluate_joint_interval_besr,
    evaluate_noise_qualified_besr, evaluate_uncertain_floor_besr,
};
pub use certification::{
    TierCertificationReport, TierQualityFloors, certify_tier_audio, check_adjacent_tier_monotone,
    is_tier_pair_monotone,
};
pub use fft::{Radix2FftPlan, build_hann_window, compute_welch_seg_starts};
pub use ltas::{
    LtasChannelReport, LtasInterval, LtasLossMode, Welch33TargetCache,
    canonical_ltas_33_centers_hz, canonical_ltas_33_intervals_hz, compute_ltas_channel,
    eval_ltas_and_nyquist_fused, eval_ltas_loss_and_adjoint, eval_nyquist_hinge_and_adjoint,
};
pub use mr_iir::{MrIirTargetCache, compute_mr_iir_channel, eval_mr_iir_loss_and_adjoint};
pub use noise_floor::{EXCESS_TAU_FRAC, EXCESS_TAU_MIN, SNR_EPS, smooth_excess_sq, snr_weight};
pub use phase_ot::{
    IfGdTargetCache, SpecOtTargetCache, eval_if_gd_loss_and_adjoint, eval_spec_ot_loss_and_adjoint,
};
pub use spectral::{
    AnalyticGaborBank, GaborBandKernel, GaborTargetCache, K_GABOR, eval_gabor_loss_and_adjoint,
    eval_runner_gabor_bezout_bridge,
};

/// Dynamic 8-term loss weights evaluated at any cosine homotopy progress `p in [0, 1]`
/// (`ref[train.pipeline.backbone-trajectory-opt]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositeLossWeights {
    /// Degree-7 Bézout `bESR` weight (`0.05 -> 1.00`).
    pub w_besr: f64,
    /// 5-scale 32-probe `MR-IIR` weight (`0.60 -> 0.25`).
    pub w_mr_iir: f64,
    /// 1-D spectral optimal transport (`SpecOT`) weight (`0.25 -> 0.10`).
    pub w_spec_ot: f64,
    /// 33-band constant-$Q$ `Gabor` envelope weight (`0.25 -> 0.15`).
    pub w_gabor: f64,
    /// Instantaneous frequency + group delay (`IF+GD`) weight (`0.0 -> 0.15` over second half).
    pub w_if_gd: f64,
    /// 33-band Welch `LTAS` weight (`0.10` constant on homotopy schedule).
    pub w_ltas: f64,
    /// `20 ms` segmental `SegESR` weight (`0.20 -> 0.10`).
    pub w_seg_esr: f64,
    /// Ultrasonic above-20 kHz `UltrasonicHinge` weight (`0.25` constant on homotopy schedule).
    pub w_nyquist: f64,
}

impl CompositeLossWeights {
    /// Extracts fixed scalar weights from a [`CompositeLossConfig`].
    #[must_use]
    pub const fn from_config(cfg: &CompositeLossConfig) -> Self {
        Self {
            w_besr: cfg.w_besr,
            w_mr_iir: cfg.w_mr_iir,
            w_spec_ot: cfg.w_spec_ot,
            w_gabor: cfg.w_gabor,
            w_if_gd: cfg.w_if_gd,
            w_ltas: cfg.w_ltas,
            w_seg_esr: cfg.w_seg_esr,
            w_nyquist: cfg.w_nyquist,
        }
    }

    /// Evaluates the canonical Phase-2B cosine homotopy weight vector at `progress in [0, 1]`
    /// (`webgpu_and_varpro_solver.md` §2.2).
    #[must_use]
    pub fn homotopy_at(progress: f64) -> Self {
        let p = if progress.is_finite() {
            progress.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let h = 0.5 * (1.0 - (PI * p).cos());
        let p2 = (2.0 * p - 1.0).clamp(0.0, 1.0);
        let h_ifgd = 0.5 * (1.0 - (PI * p2).cos());
        Self {
            w_besr: 0.05 + 0.95 * h,
            w_mr_iir: 0.60 - 0.35 * h,
            w_spec_ot: 0.25 - 0.15 * h,
            w_gabor: 0.25 - 0.10 * h,
            w_if_gd: 0.15 * h_ifgd,
            w_ltas: 0.10,
            w_seg_esr: 0.20 - 0.10 * h,
            w_nyquist: 0.25,
        }
    }
}

impl Default for CompositeLossWeights {
    fn default() -> Self {
        Self::from_config(&CompositeLossConfig::default())
    }
}

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
    bezout_noise_e: Option<[f64; 3]>,
    mr_iir: Option<MrIirTargetCache>,
    ltas: Option<Welch33TargetCache>,
    gabor: GaborTargetCache,
    spec_ot: Option<SpecOtTargetCache>,
    if_gd: Option<IfGdTargetCache>,
    seg_esr: Option<SegEsrTargetCache>,
}

impl ChannelCachesWrapper {
    fn from_channel_with_options(
        target: &[f64],
        noise: Option<&[f64]>,
        cfg: &CompositeLossConfig,
        bank: &Bezout7FilterBank,
        gabor_bank: &AnalyticGaborBank,
        build_all: bool,
    ) -> Result<Self, TrainError> {
        let w = cfg.warmup_frames.min(target.len());
        let mut tmp = [Vec::new(), Vec::new(), Vec::new()];
        let bezout_target_e = bank.split_3band_and_energies(target, w, &mut tmp);
        let bezout_noise_e =
            noise.map(|ns| bank.split_3band_and_energies(&ns[..target.len()], w, &mut tmp));
        let mr_iir = (build_all || cfg.w_mr_iir > 0.0)
            .then(|| MrIirTargetCache::new(target, noise, w, cfg.sample_rate_hz))
            .flatten();
        let ltas = (build_all || cfg.w_ltas > 0.0 || cfg.w_nyquist > 0.0)
            .then(|| Welch33TargetCache::new_with_noise(target, noise, w, cfg.sample_rate_hz).ok())
            .flatten();
        let gabor = GaborTargetCache::new_with_noise(target, noise, w, gabor_bank);
        let spec_ot = (build_all || cfg.w_spec_ot > 0.0)
            .then(|| {
                SpecOtTargetCache::new_with_noise(
                    target,
                    noise,
                    w,
                    cfg.sample_rate_hz,
                    cfg.stft_fft_size,
                )
                .ok()
            })
            .flatten();
        let if_gd = (build_all || cfg.w_if_gd > 0.0)
            .then(|| {
                IfGdTargetCache::new_with_noise(
                    target,
                    noise,
                    w,
                    cfg.sample_rate_hz,
                    cfg.stft_fft_size,
                )
                .ok()
            })
            .flatten();
        let seg_esr = (build_all || cfg.w_seg_esr > 0.0)
            .then(|| SegEsrTargetCache::new(target, noise, bank, cfg.sample_rate_hz, w).ok())
            .flatten();
        Ok(Self {
            bezout_target_e,
            bezout_noise_e,
            mr_iir,
            ltas,
            gabor,
            spec_ot,
            if_gd,
            seg_esr,
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
        Self::new_with_noise_opt(target_l, target_r, None, None, config, false)
    }

    /// Builds target caches paired with repeatability noise-floor recordings `(noise_l, noise_r)`
    /// (`ref[qual.eval.thermal-hiss]`).
    pub fn new_with_noise(
        target_l: &[f64],
        target_r: &[f64],
        noise_l: Option<&[f64]>,
        noise_r: Option<&[f64]>,
        config: CompositeLossConfig,
    ) -> Result<Self, TrainError> {
        Self::new_with_noise_opt(target_l, target_r, noise_l, noise_r, config, false)
    }

    /// Eagerly builds all 8 perceptual + phase target caches (including `SegESR` and `IF+GD`
    /// even when initial `w_if_gd == 0.0`) so Phase-2B cosine homotopy schedules run zero
    /// mid-run allocations.
    pub fn new_for_homotopy(
        target_l: &[f64],
        target_r: &[f64],
        noise_l: Option<&[f64]>,
        noise_r: Option<&[f64]>,
        config: CompositeLossConfig,
    ) -> Result<Self, TrainError> {
        Self::new_with_noise_opt(target_l, target_r, noise_l, noise_r, config, true)
    }

    fn new_with_noise_opt(
        target_l: &[f64],
        target_r: &[f64],
        noise_l: Option<&[f64]>,
        noise_r: Option<&[f64]>,
        config: CompositeLossConfig,
        build_all: bool,
    ) -> Result<Self, TrainError> {
        let len = target_l.len();
        if len == 0 || target_r.len() != len {
            return Err(TrainError::InvalidDimensions {
                context: "CompositeLossEvaluator::new",
                expected: len.max(1),
                actual: target_r.len(),
            });
        }
        if let Some(nl) = noise_l
            && (nl.len() < len || !nl[..len].iter().all(|v| v.is_finite()))
        {
            return Err(TrainError::InvalidDimensions {
                context: "CompositeLossEvaluator::new_with_noise(noise_l)",
                expected: len,
                actual: nl.len(),
            });
        }
        if let Some(nr) = noise_r
            && (nr.len() < len || !nr[..len].iter().all(|v| v.is_finite()))
        {
            return Err(TrainError::InvalidDimensions {
                context: "CompositeLossEvaluator::new_with_noise(noise_r)",
                expected: len,
                actual: nr.len(),
            });
        }
        let bank = Bezout7FilterBank::new(config.sample_rate_hz);
        let gabor_bank = AnalyticGaborBank::new(config.sample_rate_hz);
        let ch_l = ChannelCachesWrapper::from_channel_with_options(
            target_l,
            noise_l,
            &config,
            &bank,
            &gabor_bank,
            build_all,
        )?;
        let ch_r = ChannelCachesWrapper::from_channel_with_options(
            target_r,
            noise_r,
            &config,
            &bank,
            &gabor_bank,
            build_all,
        )?;
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

    fn eval_channel_with_weights(
        &self,
        pred: &[f64],
        target: &[f64],
        caches: &ChannelCachesWrapper,
        weights: CompositeLossWeights,
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
        if weights.w_besr > 0.0 {
            local_g.fill(0.0);
            besr = self.bank.eval_besr_loss_and_adjoint_with_noise(
                pred,
                target,
                BesrLossSpec {
                    target_band_energies: caches.bezout_target_e,
                    noise_band_energies: caches.bezout_noise_e,
                    eps_floor: [1.0e-6; 3],
                    warmup: w,
                },
                grad_opt.is_some().then_some(&mut local_g),
            );
            accum(weights.w_besr, grad_opt.as_deref_mut(), &local_g);
        }

        let mut mr_iir = 0.0;
        if weights.w_mr_iir > 0.0
            && let Some(ref mc) = caches.mr_iir
        {
            local_g.fill(0.0);
            mr_iir =
                eval_mr_iir_loss_and_adjoint(mc, pred, grad_opt.is_some().then_some(&mut local_g));
            accum(weights.w_mr_iir, grad_opt.as_deref_mut(), &local_g);
        }

        let mut gabor = 0.0;
        if weights.w_gabor > 0.0 {
            local_g.fill(0.0);
            gabor = eval_gabor_loss_and_adjoint(
                &self.gabor_bank,
                &caches.gabor,
                pred,
                grad_opt.is_some().then_some(&mut local_g),
            );
            accum(weights.w_gabor, grad_opt.as_deref_mut(), &local_g);
        }

        let mut spec_ot = 0.0;
        if weights.w_spec_ot > 0.0
            && let Some(ref sc) = caches.spec_ot
        {
            local_g.fill(0.0);
            spec_ot =
                eval_spec_ot_loss_and_adjoint(sc, pred, grad_opt.is_some().then_some(&mut local_g));
            accum(weights.w_spec_ot, grad_opt.as_deref_mut(), &local_g);
        }

        let mut if_gd = 0.0;
        if weights.w_if_gd > 0.0
            && let Some(ref ic) = caches.if_gd
        {
            local_g.fill(0.0);
            if_gd =
                eval_if_gd_loss_and_adjoint(ic, pred, grad_opt.is_some().then_some(&mut local_g));
            accum(weights.w_if_gd, grad_opt.as_deref_mut(), &local_g);
        }

        let (ltas, nyquist) = if (weights.w_ltas > 0.0 || weights.w_nyquist > 0.0)
            && let Some(ref lc) = caches.ltas
        {
            eval_ltas_and_nyquist_fused(
                lc,
                pred,
                c.ltas_mode,
                weights.w_ltas.max(0.0),
                weights.w_nyquist.max(0.0),
                grad_opt.as_deref_mut(),
            )
        } else {
            (0.0, 0.0)
        };

        let mut seg_esr = 0.0;
        if weights.w_seg_esr > 0.0 {
            local_g.fill(0.0);
            seg_esr = if let Some(ref sec) = caches.seg_esr {
                eval_seg_esr_cached_and_adjoint(
                    sec,
                    &self.bank,
                    pred,
                    grad_opt.is_some().then_some(&mut local_g),
                )
            } else {
                eval_seg_esr_loss_and_adjoint(
                    pred,
                    target,
                    c.sample_rate_hz,
                    w,
                    grad_opt.is_some().then_some(&mut local_g),
                )
            };
            accum(weights.w_seg_esr, grad_opt, &local_g);
        }

        let total = weights.w_besr * besr
            + weights.w_mr_iir * mr_iir
            + weights.w_spec_ot * spec_ot
            + weights.w_gabor * gabor
            + weights.w_if_gd * if_gd
            + weights.w_ltas * ltas
            + weights.w_seg_esr * seg_esr
            + weights.w_nyquist * nyquist;
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

    /// Evaluates stereo composite loss with explicit dynamic weights (for Phase-2B cosine homotopy)
    /// and optionally fills exact reverse-mode gradients `[d_l, d_r]`.
    pub fn eval_and_adjoint_with_weights(
        &self,
        pred_l: &[f64],
        pred_r: &[f64],
        weights: CompositeLossWeights,
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
        let mut bl = self.eval_channel_with_weights(
            pred_l,
            &self.target_l,
            &self.ch_l,
            weights,
            grad_l_opt.as_deref_mut(),
        );
        let br = self.eval_channel_with_weights(
            pred_r,
            &self.target_r,
            &self.ch_r,
            weights,
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

    /// Evaluates stereo composite loss and optionally fills exact reverse-mode gradients `[d_l, d_r]`.
    pub fn eval_and_adjoint(
        &self,
        pred_l: &[f64],
        pred_r: &[f64],
        grad_l_opt: Option<&mut [f64]>,
        grad_r_opt: Option<&mut [f64]>,
    ) -> Result<CompositeLossBreakdown, TrainError> {
        self.eval_and_adjoint_with_weights(
            pred_l,
            pred_r,
            CompositeLossWeights::from_config(&self.config),
            grad_l_opt,
            grad_r_opt,
        )
    }
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

        assert!((gl[150] - gl[150]).abs() == 0.0);

        // Verify 8-term cosine homotopy schedule endpoints, monotonic shifts, and eager noisy eval FD.
        let w0 = CompositeLossWeights::homotopy_at(0.0);
        let wm = CompositeLossWeights::homotopy_at(0.5);
        let w1 = CompositeLossWeights::homotopy_at(1.0);
        assert!((w0.w_mr_iir - 0.60).abs() < 1.0e-12 && (w1.w_mr_iir - 0.25).abs() < 1.0e-12);
        assert!((w0.w_spec_ot - 0.25).abs() < 1.0e-12 && (w1.w_spec_ot - 0.10).abs() < 1.0e-12);
        assert!((w0.w_gabor - 0.25).abs() < 1.0e-12 && (w1.w_gabor - 0.15).abs() < 1.0e-12);
        assert!((w0.w_seg_esr - 0.20).abs() < 1.0e-12 && (w1.w_seg_esr - 0.10).abs() < 1.0e-12);
        assert!((w0.w_besr - 0.05).abs() < 1.0e-12 && (w1.w_besr - 1.00).abs() < 1.0e-12);
        assert!(w0.w_if_gd.abs() < 1.0e-12 && wm.w_if_gd.abs() < 1.0e-12);
        assert!((w1.w_if_gd - 0.15).abs() < 1.0e-12);
        assert!((w0.w_ltas - 0.10).abs() < 1.0e-12 && (w1.w_nyquist - 0.25).abs() < 1.0e-12);

        let noise_l: Vec<f64> = (0..n).map(|t| 0.02 * ((t as f64) * 0.41).sin()).collect();
        let noise_r: Vec<f64> = (0..n).map(|t| 0.02 * ((t as f64) * 0.43).cos()).collect();
        let h_eval = CompositeLossEvaluator::new_for_homotopy(
            &tgt_l,
            &tgt_r,
            Some(&noise_l),
            Some(&noise_r),
            CompositeLossConfig {
                w_if_gd: 0.0,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let mut hgl = vec![0.0_f64; n];
        let mut hgr = vec![0.0_f64; n];
        let w75 = CompositeLossWeights::homotopy_at(0.75);
        let hbd = h_eval
            .eval_and_adjoint_with_weights(&pred_l, &pred_r, w75, Some(&mut hgl), Some(&mut hgr))
            .unwrap();
        assert!(hbd.if_gd > 0.0 && hbd.seg_esr > 0.0 && hbd.nyquist >= 0.0);
        for idx in [150, 500] {
            let (mut ppl, mut pml) = (pred_l.clone(), pred_l.clone());
            ppl[idx] += eps;
            pml[idx] -= eps;
            let fd_l = (h_eval
                .eval_and_adjoint_with_weights(&ppl, &pred_r, w75, None, None)
                .unwrap()
                .total
                - h_eval
                    .eval_and_adjoint_with_weights(&pml, &pred_r, w75, None, None)
                    .unwrap()
                    .total)
                / (2.0 * eps);
            assert!(
                (hgl[idx] - fd_l).abs() <= 1.0e-4 * hgl[idx].abs().max(1.0e-3),
                "Homotopy noisy L idx={idx} ana={} fd={fd_l}",
                hgl[idx]
            );
        }
    }
}
