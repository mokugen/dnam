#![forbid(unsafe_code)]
//! Live Waveform, 32-Band Spectral Error & 16-Pole Kautz Annealing Telemetry (`solver::progress::telemetry`).
//!
//! Invoked strictly at coarse training boundaries when `TrainerProgressHandle::is_active()` is `true`
//! (`0` allocations and `0` CPU cycles when no UI listener/seqlock is attached).

use dnam_train::losses::{Welch33TargetCache, compute_ltas_channel};
use dnam_train::{
    Pass5EmitConfig, SonaModelF64, encode_model_to_dnam_bin, quantize_kautz_pole_q8,
    quantize_ltas_error_q20_i8,
};
use dnam_vm::bin::DnamBinView;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use super::{TrainerProgressHandle, TrainerProgressSnapshot, TrainingPhase, TrainingStatus};

const MAX_PROBE_PREFIX_SAMPLES: usize = 16_384;
const WAVE_WINDOW_SAMPLES: usize = 128;
const _: () = assert!(WAVE_WINDOW_SAMPLES <= MAX_PROBE_PREFIX_SAMPLES);

/// Precomputed peak-transient window and target Welch-33 spectral reference for live UI scopes.
#[derive(Clone, Debug)]
pub struct VisualProbeAnchor {
    pub win_start: usize,
    pub render_len: usize,
    pub inv_peak: f64,
    pub target_wave_i8: [i8; 64],
    pub tgt_l_f64: Vec<f64>,
    pub welch_l: Welch33TargetCache,
    pub sr_hz: f64,
}

impl VisualProbeAnchor {
    /// Extracts the highest-energy `128`-sample post-warmup window from the first `<= 16,384` samples.
    #[must_use]
    pub fn from_stereo_probe(tgt_l: &[f32], warmup: usize, sr_hz: f64) -> Option<Self> {
        let usable = tgt_l.len();
        if usable < WAVE_WINDOW_SAMPLES {
            return None;
        }
        let render_len = usable.clamp(WAVE_WINDOW_SAMPLES, MAX_PROBE_PREFIX_SAMPLES);
        let search_lo = warmup.min(render_len.saturating_sub(WAVE_WINDOW_SAMPLES));
        let mut best_start = search_lo;
        let mut best_energy = -1.0_f64;
        for start in (search_lo..=render_len - WAVE_WINDOW_SAMPLES).step_by(32) {
            let e: f64 = tgt_l[start..start + WAVE_WINDOW_SAMPLES]
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
        let peak = tgt_l[best_start..best_start + WAVE_WINDOW_SAMPLES]
            .iter()
            .fold(1.0e-6_f64, |acc, &v| acc.max(f64::from(v.abs())));
        let inv_peak = 127.0 / peak;
        let mut target_wave_i8 = [0_i8; 64];
        for (k, dst) in target_wave_i8.iter_mut().enumerate() {
            let s = f64::from(tgt_l[best_start + k * 2]);
            *dst = (s * inv_peak).round().clamp(-127.0, 127.0) as i8;
        }
        let tgt_l_f64: Vec<f64> = tgt_l[..render_len].iter().map(|&v| f64::from(v)).collect();
        let w_spec = warmup.min(render_len / 2);
        let sr = sr_hz.max(8_000.0);
        let welch_l = Welch33TargetCache::new(&tgt_l_f64, w_spec, sr).ok()?;
        Some(Self {
            win_start: best_start,
            render_len,
            inv_peak,
            target_wave_i8,
            tgt_l_f64,
            welch_l,
            sr_hz: sr,
        })
    }
}

/// Encodes up to `16` leading `SonaModelF64` Kautz modal poles into normalized `(radius_u8, log_freq_u8)` bytes.
#[must_use]
pub fn encode_kautz_poles_q8(model: &SonaModelF64, sr_hz: f64) -> [(u8, u8); 16] {
    let mut out = [(0_u8, 0_u8); 16];
    let n = model.active_kautz_pairs.min(16).min(model.kautz.len());
    for (idx, dst) in out.iter_mut().take(n).enumerate() {
        let kp = model.kautz[idx];
        let r_base = kp.r_low + 0.5 * kp.dr;
        let [rq, fq] = quantize_kautz_pole_q8(r_base, kp.theta, sr_hz);
        *dst = (rq, fq);
    }
    out
}

/// Computes `32`-band base-10 `1/3`-octave error in `0.05 dB` units (`[-127, 127]`).
#[must_use]
pub fn compute_ltas_error_q20_i8(
    welch: &Welch33TargetCache,
    sr_hz: f64,
    out_l: &[f64],
) -> ([i8; 32], f32) {
    let report = compute_ltas_channel(welch, out_l, None);
    let pred_p = Welch33TargetCache::new(out_l, welch.warmup.min(out_l.len() / 2), sr_hz)
        .map_or([0.0_f64; 33], |c| c.tgt_band_p);
    let q = quantize_ltas_error_q20_i8(&pred_p, &welch.tgt_band_p, welch.floor);
    (q, report.max_adj_deviation_db as f32)
}

/// Scalar metadata describing a visual milestone snapshot.
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

/// Renders a short `<= 16,384`-sample prefix of `model` via `SonaVm` and publishes a `256 B` `DNPR` frame.
pub fn emit_sona_visual_milestone(
    handle: &TrainerProgressHandle,
    model: &SonaModelF64,
    in_l: &[f32],
    in_r: &[f32],
    anchor: &VisualProbeAnchor,
    meta: VisualMilestoneMeta,
) {
    if !handle.is_active() {
        return;
    }
    let render_n = anchor.render_len.min(in_l.len()).min(in_r.len());
    if render_n <= anchor.win_start + 127 {
        return;
    }
    let cfg = Pass5EmitConfig::default();
    let Ok(dbin) = encode_model_to_dnam_bin(model, cfg) else {
        return;
    };
    let Ok(view) = DnamBinView::parse(&dbin) else {
        return;
    };
    let Ok(mut vm) =
        SonaVm::<DefaultMathBackend>::from_view_owned(&view, meta.tier, anchor.sr_hz as f32, false)
    else {
        return;
    };
    let mut ol = vec![0.0_f32; render_n];
    let mut or = vec![0.0_f32; render_n];
    if vm
        .process_block(&in_l[..render_n], &in_r[..render_n], &mut ol, &mut or)
        .is_err()
    {
        return;
    }
    let mut model_wave_i8 = [0_i8; 64];
    for (k, dst) in model_wave_i8.iter_mut().enumerate() {
        let idx = anchor.win_start + k * 2;
        let v = f64::from(ol.get(idx).copied().unwrap_or(0.0));
        *dst = (v * anchor.inv_peak).round().clamp(-127.0, 127.0) as i8;
    }
    let out_f64: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
    let (ltas_error_q20_i8, meas_ltas) =
        compute_ltas_error_q20_i8(&anchor.welch_l, anchor.sr_hz, &out_f64);
    let w_eval = anchor.welch_l.warmup.min(render_n);
    let (mut sig_sq, mut err_sq) = (0.0_f64, 0.0_f64);
    for t in w_eval..render_n {
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
    let t_idx = meta.tier.index();
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
        kautz_poles_q8: encode_kautz_poles_q8(model, anchor.sr_hz),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::AtomicTrainerProgress;
    use std::f64::consts::TAU;
    use std::sync::Arc;

    #[test]
    fn test_visual_probe_anchor_and_sona_visual_milestone() {
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
        let mut model = SonaModelF64::new(48_000.0);
        model.active_fir_taps = 4;
        model.fir_l[0] = 0.5;
        model.fir_r[0] = 0.5;
        model.active_kautz_pairs = 4;
        for i in 0..4 {
            let f = 40.0 + (i as f64) * 120.0;
            model.kautz[i].r_low = 0.95;
            model.kautz[i].theta = TAU * f / 48_000.0;
        }
        let raw = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).expect("encode");
        let view = DnamBinView::parse(&raw).expect("parse");
        let mut vm =
            SonaVm::<DefaultMathBackend>::from_view_owned(&view, Tier::Ultra, 48_000.0, false)
                .expect("vm");
        let (mut tgt_l, mut tgt_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        vm.process_block(&dry, &dry, &mut tgt_l, &mut tgt_r)
            .expect("render");

        let anchor = VisualProbeAnchor::from_stereo_probe(&tgt_l, 256, 48_000.0).expect("anchor");
        assert!(anchor.win_start >= 256);

        let slot = Arc::new(AtomicTrainerProgress::new());
        let handle = TrainerProgressHandle::from_slot(Arc::clone(&slot));
        emit_sona_visual_milestone(
            &handle,
            &model,
            &dry,
            &dry,
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
