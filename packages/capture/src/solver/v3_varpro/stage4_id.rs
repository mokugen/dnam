#![forbid(unsafe_code)]
//! Black-box Stage IV spatial reflection & FDN identification (`stage4_id`).
//!
//! Dispatches after backbone + `f32` polish when `Ultra` residual error warrants Stage IV:
//! 1. Discrete early room reflections (`room_lti`): 2-tap OMP lag detection (`64..=960` samples) +
//!    causal `fp16` FDN deconvolution + Stage III refit.
//! 2. Incoherent modulation/delay/FDN rigs (`incoherent_rig`): prime FDN quartet + minimum-phase
//!    cepstral EQ + shared low-frequency Kautz trim + top-down 4-tier monotone cascade.

mod incoherent_rig;
mod room_lti;

use std::f64::consts::TAU;

use dnam_engine::Tier;
use dnam_engine::stages::{V3ColdTierModel, V3CoreInstance, V3MatryoshkaModel};

use crate::metrics::TargetRepeatabilityProfile;
use crate::metrics::certification::{
    PerceptualCertificationSummary, TargetChannelCertCache, verify_monotone_4tier_summaries,
};
use crate::solver::backends::TrainerBackend;
use crate::solver::canonical_besr_warmup;
use crate::solver::progress::telemetry::{
    Stage4SummaryStep, VisualProbeAnchor, emit_stage4_summary_step,
};
use crate::stimulus::fft::{Complex64, fft_radix2_in_place};

use super::model_f64::{V3TrainDataset, V3TrainStem};

const BYPASS_ULTRA_BESR_DB: f64 = -34.5;
const BYPASS_ULTRA_LTAS_DB: f64 = 0.09;
const INCOHERENT_TRIGGER_BESR_DB: f64 = -16.0;

pub(in crate::solver) fn render_tier_stereo<B: TrainerBackend>(
    cold: &V3ColdTierModel,
    in_l: &[f32],
    in_r: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = in_l.len().min(in_r.len());
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];
    if n == 0 {
        return (out_l, out_r);
    }
    let keep_in_r = !cold.mono_out || cold.feat_params.key_link > 0.0;
    let r_in = if keep_in_r { &in_r[..n] } else { &in_l[..n] };
    let mut inst = V3CoreInstance::<B>::from_cold_model(cold);
    inst.reset_to_dc_equilibrium();
    inst.process_block_stereo(&in_l[..n], r_in, &mut out_l, &mut out_r);
    (out_l, out_r)
}

pub(super) struct Stem0CertCaches {
    cache_l: TargetChannelCertCache,
    cache_r: Option<TargetChannelCertCache>,
    n: usize,
}

impl Stem0CertCaches {
    pub(super) fn new(stem: &V3TrainStem, sr_hz: f64, mono_out: bool) -> Option<Self> {
        let n = stem.usable_len();
        let warmup = stem.warmup.max(canonical_besr_warmup(n));
        if n <= warmup + 64 {
            return None;
        }
        let tl: Vec<f64> = stem.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
        let cache_l = TargetChannelCertCache::new(&tl, sr_hz, warmup);
        let cache_r = (!mono_out).then(|| {
            let tr: Vec<f64> = stem.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
            TargetChannelCertCache::new(&tr, sr_hz, warmup)
        });
        Some(Self {
            cache_l,
            cache_r,
            n,
        })
    }

    pub(super) fn certify_single_tier<B: TrainerBackend>(
        &mut self,
        cold: &V3ColdTierModel,
        stem: &V3TrainStem,
        prev_l: Option<&PerceptualCertificationSummary>,
        prev_r: Option<&PerceptualCertificationSummary>,
    ) -> (
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
    ) {
        let prof = TargetRepeatabilityProfile::deterministic();
        let (out_l, out_r) =
            render_tier_stereo::<B>(cold, &stem.in_l[..self.n], &stem.in_r[..self.n]);
        let mdl_l: Vec<f64> = out_l.iter().map(|&v| f64::from(v)).collect();
        let sum_l = self
            .cache_l
            .certify_universal_with_prescreen(cold.tier, &mdl_l, &prof, prev_l);
        let sum_r = self.cache_r.as_mut().map(|cr| {
            let mdl_r: Vec<f64> = out_r.iter().map(|&v| f64::from(v)).collect();
            cr.certify_universal_with_prescreen(cold.tier, &mdl_r, &prof, prev_r)
        });
        (sum_l, sum_r)
    }

    pub(super) fn certify_matryoshka<B: TrainerBackend>(
        &mut self,
        model: &V3MatryoshkaModel,
        stem: &V3TrainStem,
    ) -> Option<(
        [PerceptualCertificationSummary; 4],
        Option<[PerceptualCertificationSummary; 4]>,
    )> {
        let mut sums_l = Vec::with_capacity(4);
        let mut sums_r = Vec::with_capacity(4);
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let cold = model.extract_tier(tier);
            let (sl, sr) = self.certify_single_tier::<B>(&cold, stem, None, None);
            sums_l.push(sl);
            if let Some(sr_val) = sr {
                sums_r.push(sr_val);
            }
        }
        let arr_l: [PerceptualCertificationSummary; 4] = sums_l.try_into().ok()?;
        let arr_r: Option<[PerceptualCertificationSummary; 4]> = if self.cache_r.is_some() {
            Some(sums_r.try_into().ok()?)
        } else {
            None
        };
        Some((arr_l, arr_r))
    }
}

pub(super) fn summaries_monotone(
    l: &[PerceptualCertificationSummary; 4],
    r: Option<&[PerceptualCertificationSummary; 4]>,
) -> bool {
    verify_monotone_4tier_summaries(l) && r.is_none_or(verify_monotone_4tier_summaries)
}

fn static_hammerstein_residual(stem0: &V3TrainStem) -> Option<(f64, Vec<f32>)> {
    const FIR_TAPS: usize = 16;
    const COLS: usize = 1 + FIR_TAPS + 8; // bias + 16 FIR taps + 8 static nonlinear features = 25
    let n = stem0.usable_len();
    let start = 4_096_usize;
    let fit_end = n.min(69_632);
    if fit_end <= start + FIR_TAPS + 64 {
        return None;
    }
    let mut g = vec![0.0_f64; COLS * COLS];
    let mut rhs = [0.0_f64; COLS];
    let mut sig_sq = 0.0_f64;
    let mut row = [0.0_f64; COLS];
    for t in start..fit_end {
        let y = f64::from(stem0.tgt_l[t]);
        sig_sq += y * y;
        let x0 = f64::from(stem0.in_l[t]);
        row[0] = 1.0;
        for k in 0..FIR_TAPS {
            row[1 + k] = f64::from(stem0.in_l[t - k]);
        }
        let t3 = (3.0 * x0).tanh();
        let t8 = (8.0 * x0).tanh();
        let x2 = x0 * x0;
        row[17] = x2;
        row[18] = x2 * x0;
        row[19] = x0 * x0.abs();
        row[20] = t3;
        row[21] = t8;
        row[22] = t3 * t3;
        row[23] = t8 * t8;
        row[24] = x2 * x2 * x0;
        for i in 0..COLS {
            let ri = row[i];
            rhs[i] += y * ri;
            for j in i..COLS {
                g[i * COLS + j] += ri * row[j];
            }
        }
    }
    if sig_sq <= 1.0e-12 {
        return None;
    }
    let mean_diag = ((0..COLS).map(|i| g[i * COLS + i]).sum::<f64>() / (COLS as f64)).max(1.0e-12);
    let ridge = 1.0e-4 * mean_diag;
    for i in 0..COLS {
        g[i * COLS + i] += ridge;
        for j in 0..i {
            g[i * COLS + j] = g[j * COLS + i];
        }
    }
    for j in 0..COLS {
        let mut diag = g[j * COLS + j];
        for k in 0..j {
            diag -= g[j * COLS + k] * g[j * COLS + k];
        }
        if !diag.is_finite() || diag <= 1.0e-15 {
            return None;
        }
        let d = diag.sqrt();
        g[j * COLS + j] = d;
        for i in (j + 1)..COLS {
            let mut s = g[i * COLS + j];
            for k in 0..j {
                s -= g[i * COLS + k] * g[j * COLS + k];
            }
            g[i * COLS + j] = s / d;
        }
    }
    let mut yv = [0.0_f64; COLS];
    for i in 0..COLS {
        let mut s = rhs[i];
        for k in 0..i {
            s -= g[i * COLS + k] * yv[k];
        }
        yv[i] = s / g[i * COLS + i];
    }
    let mut w = [0.0_f64; COLS];
    for i in (0..COLS).rev() {
        let mut s = yv[i];
        for k in (i + 1)..COLS {
            s -= g[k * COLS + i] * w[k];
        }
        w[i] = s / g[i * COLS + i];
    }
    let mut res_l = vec![0.0_f32; fit_end];
    let mut err_sq = 0.0_f64;
    for t in start..fit_end {
        let x0 = f64::from(stem0.in_l[t]);
        let t3 = (3.0 * x0).tanh();
        let t8 = (8.0 * x0).tanh();
        let x2 = x0 * x0;
        let mut pred = w[0]
            + w[17] * x2
            + w[18] * x2 * x0
            + w[19] * x0 * x0.abs()
            + w[20] * t3
            + w[21] * t8
            + w[22] * t3 * t3
            + w[23] * t8 * t8
            + w[24] * x2 * x2 * x0;
        for k in 0..FIR_TAPS {
            pred += w[1 + k] * f64::from(stem0.in_l[t - k]);
        }
        let d = f64::from(stem0.tgt_l[t]) - pred;
        err_sq += d * d;
        res_l[t] = d as f32;
    }
    let hamm_db = 10.0 * (err_sq / sig_sq).max(1.0e-12).log10();
    Some((hamm_db, res_l))
}

fn sliding_residual_coherence(in_l: &[f32], res_l: &[f32], start: usize, end: usize) -> f64 {
    const N_DFT: usize = 2_048;
    const HOP: usize = 1_024;
    let hi = end.min(in_l.len()).min(res_l.len());
    if hi < start + N_DFT {
        return 1.0;
    }
    let denom = (N_DFT - 1) as f64;
    let win: [f64; N_DFT] = std::array::from_fn(|j| 0.5 * (1.0 - (TAU * (j as f64) / denom).cos()));
    let half = N_DFT / 2;
    let mut pxx = vec![0.0_f64; half + 1];
    let mut prr = vec![0.0_f64; half + 1];
    let mut pxr_re = vec![0.0_f64; half + 1];
    let mut pxr_im = vec![0.0_f64; half + 1];
    let mut bx = vec![Complex64::new(0.0, 0.0); N_DFT];
    let mut br = vec![Complex64::new(0.0, 0.0); N_DFT];

    let mut off = start;
    while off + N_DFT <= hi {
        for j in 0..N_DFT {
            let wj = win[j];
            bx[j] = Complex64::new(f64::from(in_l[off + j]) * wj, 0.0);
            br[j] = Complex64::new(f64::from(res_l[off + j]) * wj, 0.0);
        }
        fft_radix2_in_place(&mut bx, false);
        fft_radix2_in_place(&mut br, false);
        for k in 1..=half {
            let (xr, xi) = (bx[k].re, bx[k].im);
            let (rr, ri) = (br[k].re, br[k].im);
            pxx[k] += xr * xr + xi * xi;
            prr[k] += rr * rr + ri * ri;
            pxr_re[k] += xr * rr + xi * ri;
            pxr_im[k] += xr * ri - xi * rr;
        }
        off += HOP;
    }

    let max_pxx = pxx[1..=half].iter().copied().fold(0.0_f64, f64::max);
    if max_pxx <= 1.0e-18 {
        return 1.0;
    }
    let gate = 0.01 * max_pxx;
    let (mut coh_sum, mut count) = (0.0_f64, 0_usize);
    for k in 1..=half {
        if pxx[k] >= gate && prr[k] > 1.0e-18 {
            let num = pxr_re[k] * pxr_re[k] + pxr_im[k] * pxr_im[k];
            coh_sum += (num / (pxx[k] * prr[k])).clamp(0.0, 1.0);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        coh_sum / (count as f64)
    }
}

/// Returns `true` when `stem0` is a full contiguous (`>= 90,000`-sample) incoherent rig capture
/// whose `25`-column static Hammerstein residual exceeds `-13.0 dB` AND whose sliding
/// `2,048`-sample Hann STFT residual coherence `|P_{xr}|^2 / (P_{xx} P_{rr})` drops below `0.65`.
pub(super) fn stem_is_incoherent_candidate(stem0: &V3TrainStem) -> bool {
    if stem0.usable_len() < 90_000 {
        return false;
    }
    let Some((hamm_db, res_l)) = static_hammerstein_residual(stem0) else {
        return false;
    };
    hamm_db > -13.0 && sliding_residual_coherence(&stem0.in_l, &res_l, 4_096, res_l.len()) < 0.65
}

/// Extends incoherent rig detection to datasets whose empirical repeatability probe established
/// positive `band_inc_ratio` (e.g., stationary time-invariant modulated/diffuse rigs).
pub(super) fn dataset_is_incoherent_candidate(dataset: &V3TrainDataset) -> bool {
    let Some(stem0) = dataset.stems.first() else {
        return false;
    };
    if stem_is_incoherent_candidate(stem0) {
        return true;
    }
    stem0.usable_len() >= 90_000
        && dataset.rep_profiles.is_some_and(|profs| {
            profs
                .iter()
                .any(|p| p.band_inc_ratio.iter().any(|&r| r > 0.02))
        })
}

pub(super) fn worst_ultra_besr_db(
    l: &[PerceptualCertificationSummary; 4],
    r: Option<&[PerceptualCertificationSummary; 4]>,
) -> f64 {
    l[3].raw_besr_db
        .max(r.map_or(f64::NEG_INFINITY, |sr| sr[3].raw_besr_db))
}

pub(super) fn should_enter_incoherent_rig(ultra_besr: f64, ultra_ltas: f64, is_inc: bool) -> bool {
    ultra_besr > INCOHERENT_TRIGGER_BESR_DB || (is_inc && ultra_ltas > BYPASS_ULTRA_LTAS_DB)
}

/// Refines Stage IV spatial reflections or incoherent FDN cascade strictly when black-box
/// waveform gates indicate unexplained discrete room echoes or diffuse/modulated rig behavior.
#[inline(never)]
#[must_use]
pub fn refine_stage4_matryoshka_with_backend<B: TrainerBackend>(
    model: V3MatryoshkaModel,
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> V3MatryoshkaModel {
    let Some(stem0) = dataset.stems.first() else {
        return model;
    };
    let Some(mut caches) = Stem0CertCaches::new(stem0, dataset.sample_rate_hz, mono_out) else {
        return model;
    };
    let Some((base_l, base_r)) = caches.certify_matryoshka::<B>(&model, stem0) else {
        return model;
    };

    let anchor = VisualProbeAnchor::from_dataset(dataset);
    let ultra_besr = worst_ultra_besr_db(&base_l, base_r.as_ref());
    let ultra_ltas = base_l[3]
        .ltas_max_db
        .max(base_r.as_ref().map_or(0.0, |sr| sr[3].ltas_max_db));
    let mono_ok = summaries_monotone(&base_l, base_r.as_ref());
    if mono_ok && ultra_besr <= BYPASS_ULTRA_BESR_DB && ultra_ltas <= BYPASS_ULTRA_LTAS_DB {
        emit_stage4_summary_step::<B>(
            &dataset.progress,
            anchor.as_ref(),
            Stage4SummaryStep {
                model: &model,
                stem0,
                tier: Tier::Ultra,
                permille: 980,
                step: 1,
                total_steps: 1,
                sl: &base_l[3],
                sr: base_r.as_ref().map(|sr| &sr[3]),
            },
        );
        return model;
    }
    emit_stage4_summary_step::<B>(
        &dataset.progress,
        anchor.as_ref(),
        Stage4SummaryStep {
            model: &model,
            stem0,
            tier: Tier::Ultra,
            permille: 150,
            step: 1,
            total_steps: 5,
            sl: &base_l[3],
            sr: base_r.as_ref().map(|sr| &sr[3]),
        },
    );

    let is_inc_cand = dataset_is_incoherent_candidate(dataset);
    if !is_inc_cand
        && let Some(room_model) =
            room_lti::fit_room_lti_deconv::<B>(&model, dataset, mono_out, &mut caches, ultra_besr)
    {
        return room_model;
    }

    let stage0_active = model.stage0.op_p_sub_mix.abs() > 1.0e-6
        || model.stage0.op_p_sub2_mix.abs() > 1.0e-6
        || model.stage0.op_p_oct_mix.abs() > 1.0e-6
        || model.stage0.op_p_env_mix.abs() > 1.0e-6;
    if (!stage0_active || ultra_ltas > 0.35)
        && should_enter_incoherent_rig(ultra_besr, ultra_ltas, is_inc_cand)
        && let Some(rig_model) =
            incoherent_rig::fit_incoherent_rig_cascade::<B>(&model, dataset, mono_out, stem0)
    {
        return rig_model;
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::ScalarTrainerBackend;
    use std::f64::consts::TAU;

    #[test]
    fn test_stage4_bypass_gate_and_incoherent_ltas_trigger() {
        let n = 4_096;
        let dry: Vec<f32> = (0..n)
            .map(|i| (0.4 * (TAU * 220.0 * (i as f64) / 48_000.0).sin()) as f32)
            .collect();
        let model = V3MatryoshkaModel::dual_rate_default(true);
        let (tgt, _) = render_tier_stereo::<ScalarTrainerBackend>(
            &model.extract_tier(Tier::Ultra),
            &dry,
            &dry,
        );
        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: dry.clone(),
                in_r: dry,
                tgt_l: tgt.clone(),
                tgt_r: tgt,
                warmup: 256,
            }],
            48_000.0,
        );
        let refined =
            refine_stage4_matryoshka_with_backend::<ScalarTrainerBackend>(model.clone(), &ds, true);
        assert_eq!(refined.stage4[3].wet_mix, model.stage4[3].wet_mix);
        assert_eq!(refined.stage4[3].tap_gains, model.stage4[3].tap_gains);

        assert!(should_enter_incoherent_rig(f64::NEG_INFINITY, 0.45, true));
        assert!(!should_enter_incoherent_rig(f64::NEG_INFINITY, 0.04, true));
        assert!(!should_enter_incoherent_rig(f64::NEG_INFINITY, 0.45, false));
        assert!(should_enter_incoherent_rig(-8.0, 0.02, false));
    }
}
