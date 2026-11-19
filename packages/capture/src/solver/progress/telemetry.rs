#![forbid(unsafe_code)]
//! Live Waveform, 32-Band Spectral Error & 16-Pole Kautz Annealing Telemetry (`solver::progress::telemetry`).
//!
//! Invoked strictly at coarse training boundaries when `TrainerProgressHandle::is_active()` is `true`
//! (`0` allocations and `0` CPU cycles when no UI listener/seqlock is attached).

use std::f64::consts::TAU;

use dnam_engine::Tier;
use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::stage2_gru::MAX_KAUTZ_POLES;

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::backends::TrainerBackend;
use crate::solver::v3_varpro::ltas_adj::{
    Welch33Scratch, Welch33TargetSummary, build_welch33_target_summary,
    eval_welch33_lse_minimax_with_nyq_and_adjoint,
};
use crate::solver::v3_varpro::model_f64::{V3MatryoshkaOptF64, V3TrainDataset, V3TrainStem};
use crate::solver::v3_varpro::stage4_id::render_tier_stereo;

use super::{TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus};

const MAX_PROBE_PREFIX_SAMPLES: usize = 16_384;
const WAVE_WINDOW_SAMPLES: usize = 128;

/// Precomputed peak-transient window and target Welch-33 spectral reference for live UI scopes.
#[derive(Clone, Debug)]
pub struct VisualProbeAnchor {
    pub win_start: usize,
    pub render_len: usize,
    pub inv_peak: f64,
    pub target_wave_i8: [i8; 64],
    pub tgt_l_f64: Vec<f64>,
    pub welch_l: Welch33TargetSummary,
    pub sr_hz: f64,
}

impl VisualProbeAnchor {
    /// Extracts the anchor only when `dataset.progress.is_active()`, paying `0` cost otherwise.
    #[must_use]
    pub fn from_dataset(dataset: &V3TrainDataset) -> Option<Self> {
        if !dataset.progress.is_active() {
            return None;
        }
        dataset
            .stems
            .first()
            .and_then(|s| Self::from_stem0(s, dataset.sample_rate_hz))
    }

    /// Extracts the highest-energy `128`-sample post-warmup window from the first `<= 16,384` samples.
    #[must_use]
    pub fn from_stem0(stem0: &V3TrainStem, sr_hz: f64) -> Option<Self> {
        let usable = stem0.usable_len();
        if usable < WAVE_WINDOW_SAMPLES {
            return None;
        }
        let render_len = usable
            .min(MAX_PROBE_PREFIX_SAMPLES)
            .max(WAVE_WINDOW_SAMPLES);
        let search_lo = stem0
            .warmup
            .min(render_len.saturating_sub(WAVE_WINDOW_SAMPLES));
        let mut best_start = search_lo;
        let mut best_energy = -1.0_f64;
        for start in (search_lo..=render_len - WAVE_WINDOW_SAMPLES).step_by(32) {
            let e: f64 = stem0.tgt_l[start..start + WAVE_WINDOW_SAMPLES]
                .iter()
                .map(|&v| {
                    let x = f64::from(v);
                    x * x
                })
                .sum();
            if e > best_energy {
                best_energy = e;
                best_start = start;
            }
        }
        let peak = stem0.tgt_l[best_start..best_start + WAVE_WINDOW_SAMPLES]
            .iter()
            .fold(1.0e-6_f64, |acc, &v| acc.max(f64::from(v.abs())));
        let inv_peak = 127.0 / peak;
        let mut target_wave_i8 = [0_i8; 64];
        for (k, dst) in target_wave_i8.iter_mut().enumerate() {
            let s = f64::from(stem0.tgt_l[best_start + k * 2]);
            *dst = (s * inv_peak).round().clamp(-127.0, 127.0) as i8;
        }
        let tgt_l_f64: Vec<f64> = stem0.tgt_l[..render_len]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let w_spec = stem0.warmup.min(render_len / 2);
        let welch_l = build_welch33_target_summary(&tgt_l_f64, w_spec, sr_hz.max(8_000.0));
        Some(Self {
            win_start: best_start,
            render_len,
            inv_peak,
            target_wave_i8,
            tgt_l_f64,
            welch_l,
            sr_hz: sr_hz.max(8_000.0),
        })
    }
}

/// Encodes `16` shared `(r, theta_rad)` Kautz poles into normalized `(radius_u8, log_freq_u8)` bytes.
#[must_use]
pub fn encode_kautz_poles_q8(
    poles: &[(f32, f32); MAX_KAUTZ_POLES],
    sr_hz: f64,
) -> [(u8, u8); MAX_KAUTZ_POLES] {
    let mut out = [(0_u8, 0_u8); MAX_KAUTZ_POLES];
    let log_span = 1_000.0_f64.ln();
    for (idx, &(r_raw, theta)) in poles.iter().enumerate() {
        let r_norm = ((f64::from(r_raw) - 0.5) / 0.5).clamp(0.0, 1.0);
        let f_hz = (f64::from(theta) * sr_hz / TAU).clamp(20.0, 20_000.0);
        let f_norm = ((f_hz / 20.0).ln() / log_span).clamp(0.0, 1.0);
        out[idx] = (
            (r_norm * 255.0).round() as u8,
            (f_norm * 255.0).round() as u8,
        );
    }
    out
}

/// Computes `32`-band base-10 `1/3`-octave error in `0.05 dB` units (`[-127, 127]` = `±6.35 dB`).
#[must_use]
pub fn compute_ltas_error_q20_i8(
    welch: &Welch33TargetSummary,
    sc: &Welch33Scratch,
    out_l: &[f64],
) -> ([i8; 32], f32) {
    let mut scratch = Welch33Scratch::default();
    let _ = eval_welch33_lse_minimax_with_nyq_and_adjoint(
        welch,
        out_l,
        None,
        12.0,
        0.0,
        0.0,
        &mut scratch,
    );
    let _ = sc;
    let mut errs = [0_i8; 32];
    for (b, slot) in errs.iter_mut().enumerate() {
        if !welch.active[b] {
            continue;
        }
        let pb = scratch.band_p[b] + welch.floor;
        let pt = welch.tgt_band_p[b] + welch.floor;
        let db: f64 = 10.0 * (pb / pt).log10();
        *slot = if db.is_finite() {
            (db * 20.0).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
    }
    (errs, scratch.last_max_abs_db as f32)
}

/// Parameters describing scalar coordinates for a visual fitting milestone.
#[derive(Clone, Copy, Debug)]
pub struct VisualMilestoneMeta {
    pub phase: TrainingPhase,
    pub permille: u16,
    pub step: u32,
    pub total_steps: u32,
    pub stage_step: u32,
    pub stage_total_steps: u32,
    pub tier: Tier,
    pub best_loss: f32,
    pub mr_iir_loss: f32,
    pub tier_besr_db: [f32; 4],
    pub tier_ltas_db: [f32; 4],
}

/// Renders a short `<= 16,384`-sample prefix of `model` at `meta.tier` and publishes a full `256 B`
/// visual + metric snapshot into `handle`. Returns immediately when `!handle.is_active()`.
pub fn emit_cold_visual_milestone<B: TrainerBackend>(
    handle: &TrainerProgressHandle,
    model: &V3MatryoshkaModel,
    stem0: &V3TrainStem,
    anchor: &VisualProbeAnchor,
    meta: VisualMilestoneMeta,
) {
    if !handle.is_active() {
        return;
    }
    let cold = model.extract_tier(meta.tier);
    let (ol, _) = render_tier_stereo::<B>(
        &cold,
        &stem0.in_l[..anchor.render_len],
        &stem0.in_r[..anchor.render_len],
    );
    let mut model_wave_i8 = [0_i8; 64];
    for (k, dst) in model_wave_i8.iter_mut().enumerate() {
        let idx = anchor.win_start + k * 2;
        let v = f64::from(ol.get(idx).copied().unwrap_or(0.0));
        *dst = (v * anchor.inv_peak).round().clamp(-127.0, 127.0) as i8;
    }
    let out_f64: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
    let scratch = Welch33Scratch::default();
    let (ltas_error_q20_i8, meas_ltas) =
        compute_ltas_error_q20_i8(&anchor.welch_l, &scratch, &out_f64);
    let w_eval = anchor.welch_l.warmup.min(anchor.render_len);
    let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
    for t in w_eval..anchor.render_len {
        let y = anchor.tgt_l_f64[t];
        let d = out_f64[t] - y;
        sig_sq += y * y;
        err_sq += d * d;
    }
    let prefix_esr_db = if sig_sq > 1.0e-18 {
        (10.0 * (err_sq / sig_sq).max(1.0e-12).log10()) as f32
    } else {
        f32::NAN
    };
    let mut tier_besr_db = meta.tier_besr_db;
    let mut tier_ltas_db = meta.tier_ltas_db;
    let t_idx = meta.tier as usize;
    if !tier_besr_db[t_idx].is_finite() {
        tier_besr_db[t_idx] = prefix_esr_db;
    }
    if !tier_ltas_db[t_idx].is_finite() && meas_ltas.is_finite() {
        tier_ltas_db[t_idx] = meas_ltas;
    }
    let best_loss = if meta.best_loss.is_finite() {
        meta.best_loss
    } else if sig_sq > 1.0e-18 {
        (err_sq / sig_sq) as f32
    } else {
        0.0
    };

    handle.emit(&TrainerProgressSnapshot {
        status: TrainingStatus::Running,
        phase: meta.phase,
        active_tier: Some(meta.tier),
        clamped_controls: 0,
        has_visual_frame: true,
        progress_permille: meta.permille,
        step: meta.step,
        total_steps: meta.total_steps,
        stage_step: meta.stage_step,
        stage_total_steps: meta.stage_total_steps,
        best_loss,
        mr_iir_loss: meta.mr_iir_loss,
        tier_besr_db,
        tier_ltas_db,
        target_wave_i8: anchor.target_wave_i8,
        model_wave_i8,
        ltas_error_q20_i8,
        kautz_poles_q8: encode_kautz_poles_q8(&model.kautz_poles, anchor.sr_hz),
    });
}

/// Convenience wrapper converting an active optimization state [`V3MatryoshkaOptF64`] when `handle.is_active()`.
pub fn emit_opt_visual_milestone<B: TrainerBackend>(
    handle: &TrainerProgressHandle,
    model_opt: &V3MatryoshkaOptF64,
    stem0: &V3TrainStem,
    anchor: &VisualProbeAnchor,
    meta: VisualMilestoneMeta,
) {
    if !handle.is_active() {
        return;
    }
    let cold = model_opt.to_matryoshka_cold();
    emit_cold_visual_milestone::<B>(handle, &cold, stem0, anchor, meta);
}

/// Coarse optimization step metadata for `v3` solver milestone emission.
#[derive(Clone, Copy, Debug)]
pub struct OptStepSpec {
    pub phase: TrainingPhase,
    pub permille: u16,
    pub step: u32,
    pub total_steps: u32,
    pub tier: Tier,
    pub loss: f64,
}

/// Emits a visual milestone from an active [`V3MatryoshkaOptF64`] when `dataset.progress.is_active()`.
pub fn emit_dataset_opt_step<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    anchor: Option<&VisualProbeAnchor>,
    model_opt: &V3MatryoshkaOptF64,
    spec: OptStepSpec,
) {
    if let (Some(a), Some(stem0)) = (anchor, dataset.stems.first()) {
        emit_opt_visual_milestone::<B>(
            &dataset.progress,
            model_opt,
            stem0,
            a,
            VisualMilestoneMeta {
                phase: spec.phase,
                permille: spec.permille,
                step: spec.step,
                total_steps: spec.total_steps,
                stage_step: spec.step,
                stage_total_steps: spec.total_steps,
                tier: spec.tier,
                best_loss: if spec.loss.is_finite() {
                    spec.loss as f32
                } else {
                    f32::NAN
                },
                mr_iir_loss: f32::NAN,
                tier_besr_db: [f32::NAN; 4],
                tier_ltas_db: [f32::NAN; 4],
            },
        );
    }
}

/// Stage IV perceptual certification milestone metadata for `emit_stage4_summary_step`.
#[derive(Clone, Copy, Debug)]
pub struct Stage4SummaryStep<'a> {
    pub model: &'a V3MatryoshkaModel,
    pub stem0: &'a V3TrainStem,
    pub tier: Tier,
    pub permille: u16,
    pub step: u32,
    pub total_steps: u32,
    pub sl: &'a PerceptualCertificationSummary,
    pub sr: Option<&'a PerceptualCertificationSummary>,
}

/// Emits a Stage IV perceptual certification visual milestone when `handle.is_active()`.
pub fn emit_stage4_summary_step<B: TrainerBackend>(
    handle: &TrainerProgressHandle,
    anchor: Option<&VisualProbeAnchor>,
    step_info: Stage4SummaryStep<'_>,
) {
    if let Some(a) = anchor {
        let mut tier_besr_db = [f32::NAN; 4];
        let mut tier_ltas_db = [f32::NAN; 4];
        let besr = step_info.sl.raw_besr_db.max(
            step_info
                .sr
                .map_or(step_info.sl.raw_besr_db, |r| r.raw_besr_db),
        ) as f32;
        let ltas = step_info.sl.ltas_max_db.max(
            step_info
                .sr
                .map_or(step_info.sl.ltas_max_db, |r| r.ltas_max_db),
        ) as f32;
        let mr = step_info
            .sl
            .mr_iir
            .max(step_info.sr.map_or(step_info.sl.mr_iir, |r| r.mr_iir)) as f32;
        let idx = step_info.tier as usize;
        tier_besr_db[idx] = besr;
        tier_ltas_db[idx] = ltas;
        emit_cold_visual_milestone::<B>(
            handle,
            step_info.model,
            step_info.stem0,
            a,
            VisualMilestoneMeta {
                phase: TrainingPhase::Stage4SpatialCascade,
                permille: step_info.permille,
                step: step_info.step,
                total_steps: step_info.total_steps,
                stage_step: step_info.step,
                stage_total_steps: step_info.total_steps,
                tier: step_info.tier,
                best_loss: 10.0_f32.powf(besr / 10.0),
                mr_iir_loss: mr,
                tier_besr_db,
                tier_ltas_db,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::AtomicTrainerProgress;
    use crate::solver::backends::ScalarTrainerBackend;
    use std::f64::consts::TAU;
    use std::sync::Arc;

    #[test]
    fn test_visual_probe_anchor_and_cold_visual_milestone() {
        let n = 4_096;
        let dry: Vec<f32> = (0..n)
            .map(|i| {
                let tf = (i as f64) / 48_000.0;
                let s: f64 = [80.0, 160.0, 330.0, 680.0, 1_400.0, 2_900.0, 6_100.0]
                    .iter()
                    .enumerate()
                    .map(|(k, &f)| (TAU * f * tf + 0.4 * (k as f64)).sin())
                    .sum();
                (0.08 * s) as f32
            })
            .collect();
        let mut model = V3MatryoshkaModel::dual_rate_default(true);
        for (i, slot_p) in model.kautz_poles.iter_mut().enumerate() {
            let f = 40.0 + (i as f64) * 120.0;
            *slot_p = (0.95, (TAU * f / 48_000.0) as f32);
        }
        for ro in &mut model.readouts {
            ro.fir_l[0] = 0.5;
            ro.fir_r[0] = 0.5;
        }
        let (tgt_l, tgt_r) = render_tier_stereo::<ScalarTrainerBackend>(
            &model.extract_tier(Tier::Ultra),
            &dry,
            &dry,
        );
        let stem = V3TrainStem {
            in_l: dry.clone(),
            in_r: dry,
            tgt_l,
            tgt_r,
            warmup: 256,
        };
        let anchor = VisualProbeAnchor::from_stem0(&stem, 48_000.0).expect("anchor");
        assert!(anchor.win_start >= 256);

        let slot = Arc::new(AtomicTrainerProgress::new());
        let handle = TrainerProgressHandle::from_slot(Arc::clone(&slot));
        emit_cold_visual_milestone::<ScalarTrainerBackend>(
            &handle,
            &model,
            &stem,
            &anchor,
            VisualMilestoneMeta {
                phase: TrainingPhase::StageBBackbone,
                permille: 450,
                step: 3,
                total_steps: 7,
                stage_step: 2,
                stage_total_steps: 5,
                tier: Tier::Ultra,
                best_loss: 0.01,
                mr_iir_loss: f32::NAN,
                tier_besr_db: [f32::NAN; 4],
                tier_ltas_db: [f32::NAN; 4],
            },
        );
        let snap = slot.poll();
        assert!(snap.has_visual_frame);
        assert_eq!(snap.status, TrainingStatus::Running);
        assert_eq!(snap.progress_permille, 450);
        assert_eq!(snap.target_wave_i8, snap.model_wave_i8);
        assert!(snap.kautz_poles_q8[0].0 > 200);
        assert!(
            snap.tier_ltas_db[3].is_finite() && snap.tier_ltas_db[3] < 0.05,
            "unexpected tier_ltas_db={:?}",
            snap.tier_ltas_db
        );
    }
}
