#![forbid(unsafe_code)]
//! Band-limited sweep HF harmonic generation & nonlinear Stage II track preservation
//! (`incoherent_rig::harmonic_cal`).

use std::f64::consts::PI;

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES};
use dnam_engine::stages::stage4_tank::v3_fdn::V3Stage4TankParams;
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};

use crate::metrics::companion::LtasTargetCache;
use crate::solver::backends::TrainerBackend;
use crate::solver::canonical_besr_warmup;

use super::super::super::model_f64::{V3TrainDataset, V3TrainStem};
use super::super::render_tier_stereo;
use super::comb_detect::{
    apply_broadband_kautz_poles, apply_lf_kautz_poles, clear_readouts, compute_broadband_rms,
};
use super::s2_modal;

const MAX_SWEEP_HF_SAMPLES: usize = 16_896;

/// Truncates `stem` cleanly to at most `max_len` samples with a matched warmup for fast screening.
pub(super) fn slice_stem_prefix(stem: &V3TrainStem, max_len: usize) -> V3TrainStem {
    let n = stem.usable_len().min(max_len);
    V3TrainStem {
        in_l: stem.in_l[..n].to_vec(),
        in_r: stem.in_r[..n].to_vec(),
        tgt_l: stem.tgt_l[..n].to_vec(),
        tgt_r: stem.tgt_r[..n].to_vec(),
        warmup: stem.warmup.min(canonical_besr_warmup(n)),
    }
}

pub(super) fn prepare_linear_base(
    baseline: &V3MatryoshkaModel,
    mono_out: bool,
    sr_hz: f64,
) -> V3MatryoshkaModel {
    let mut m = V3MatryoshkaModel::identity(false, mono_out);
    m.stage0 = baseline.stage0;
    m.tone_svf.sample_rate_hz = sr_hz as f32;
    let cal = s2_modal::build_1x_linear_resonator_cal(sr_hz);
    m.svf = cal.svf;
    m.feat_params = cal.feat_params;
    m.blocks = cal.blocks;
    m.rate_1x_cal = Some(cal);
    apply_lf_kautz_poles(&mut m, sr_hz);
    clear_readouts(&mut m);
    m
}

fn build_scaled_stage2_track<B: TrainerBackend>(
    baseline: &V3MatryoshkaModel,
    stem0: &V3TrainStem,
    drive_scale: f32,
    sr_hz: f64,
) -> V3MatryoshkaModel {
    let mut m = baseline.clone();
    m.os2x = true;
    m.tier_os2x = [false, false, true, true];
    apply_broadband_kautz_poles(&mut m, sr_hz);
    m.rate_1x_cal = Some(s2_modal::build_1x_linear_resonator_cal(sr_hz));
    for (ti, ro) in m.readouts.iter_mut().enumerate() {
        if !m.tier_os2x[ti] {
            *ro = V3TierReadout::default();
            ro.fir_l = [0.0; 64];
            ro.fir_r = [0.0; 64];
            continue;
        }
        for ch in 0..2 {
            ro.pre_cab_u_gain[ch] = (ro.pre_cab_u_gain[ch] + ro.direct_s2_u_gain[ch]) * drive_scale;
            ro.pre_cab_hp_gain[ch] =
                (ro.pre_cab_hp_gain[ch] + ro.direct_s2_hp_gain[ch]) * drive_scale;
            for (pw, &dw) in ro.pre_cab_w[ch].iter_mut().zip(&ro.direct_s2_w[ch]) {
                *pw = (*pw + dw) * drive_scale;
            }
        }
        ro.direct_s2_u_gain = [0.0; 2];
        ro.direct_s2_hp_gain = [0.0; 2];
        ro.direct_s2_w = [[0.0; MAX_TOTAL_GRU_LANES]; 2];
        ro.svf_direct_taps_l = [0.0; 8];
        ro.svf_direct_taps_r = [0.0; 8];
        ro.fir_l = [0.0; 64];
        ro.fir_r = [0.0; 64];
        ro.kautz_c_l = [(0.0, 0.0); MAX_KAUTZ_POLES];
        ro.kautz_c_r = [(0.0, 0.0); MAX_KAUTZ_POLES];
    }

    // Match cross-rate `1x`/`2x` Stage II excitation RMS before Stage III/IV across active `2x` tiers.
    let n_probe = stem0.usable_len().min(8_192);
    let w_probe = stem0.warmup.min(n_probe / 4);
    if n_probe > w_probe + 64 {
        let mut probe = m.clone();
        probe.stage0.clean_sub_mix = 0.0;
        probe.stage0.op_p_route_clean = false;
        let id_tank = V3Stage4TankParams {
            wet_mix: 0.0,
            dry_mix: 1.0,
            g_loop: 0.0,
            base_psi: 0.0,
            vca_kappa: 0.0,
            vca_min_gain: 1.0,
            ..V3Stage4TankParams::identity()
        };
        for ti in 0..4 {
            probe.stage4[ti] = id_tank;
            probe.readouts[ti].fir_l[0] = 1.0;
            probe.readouts[ti].fir_r[0] = 1.0;
        }
        let (ul_u, ur_u) = render_tier_stereo::<B>(
            &probe.extract_tier(Tier::Ultra),
            &stem0.in_l[..n_probe],
            &stem0.in_r[..n_probe],
        );
        let rms_u = compute_broadband_rms(&ul_u, &ur_u, w_probe, m.mono_out);
        if rms_u > 1.0e-7 {
            for tier in [Tier::Lite, Tier::Full, Tier::Pro] {
                let ti = tier as usize;
                if !m.tier_os2x[ti] {
                    continue;
                }
                let (ul_t, ur_t) = render_tier_stereo::<B>(
                    &probe.extract_tier(tier),
                    &stem0.in_l[..n_probe],
                    &stem0.in_r[..n_probe],
                );
                let rms_t = compute_broadband_rms(&ul_t, &ur_t, w_probe, m.mono_out);
                if rms_t > 1.0e-9 {
                    let norm = (rms_u / rms_t).clamp(0.25, 4.0) as f32;
                    if (norm - 1.0).abs() > 1.0e-3 {
                        for ch in 0..2 {
                            m.readouts[ti].pre_cab_u_gain[ch] *= norm;
                            m.readouts[ti].pre_cab_hp_gain[ch] *= norm;
                            for pw in &mut m.readouts[ti].pre_cab_w[ch] {
                                *pw *= norm;
                            }
                        }
                    }
                }
            }
        }
    }
    m
}

/// Builds candidate Stage II tracks: pure linear `1x`-modal base plus calibrated Stage II harmonic
/// tracks whenever `baseline` exhibits active nonlinear excitation.
pub(super) fn prepare_harmonic_tracks<B: TrainerBackend>(
    baseline: &V3MatryoshkaModel,
    dataset: &V3TrainDataset,
    stem0: &V3TrainStem,
    warmup: usize,
    tgt_rms: f64,
    sr_hz: f64,
) -> Vec<V3MatryoshkaModel> {
    let stage0_active = baseline.stage0.op_p_sub_mix.abs() > 1.0e-6
        || baseline.stage0.op_p_sub2_mix.abs() > 1.0e-6
        || baseline.stage0.op_p_oct_mix.abs() > 1.0e-6
        || baseline.stage0.op_p_env_mix.abs() > 1.0e-6;
    let mut tracks = Vec::with_capacity(4);
    if stage0_active {
        let mut clean_s0 = prepare_linear_base(baseline, baseline.mono_out, sr_hz);
        clean_s0.stage0 = dnam_engine::stages::router::v3_stage0::V3Stage0Params::identity();
        tracks.push(clean_s0);
    }
    tracks.push(prepare_linear_base(baseline, baseline.mono_out, sr_hz));
    let pre_cab_zero = baseline.readouts[3]
        .pre_cab_w
        .iter()
        .flatten()
        .all(|&w| w.abs() < 1.0e-7)
        && baseline.readouts[3]
            .pre_cab_hp_gain
            .iter()
            .all(|&g| g.abs() < 1.0e-6);
    let svf_feed_identity = baseline.svf.stage2_feed_taps[1..]
        .iter()
        .all(|&v| v.abs() < 1.0e-6);
    if pre_cab_zero && svf_feed_identity {
        return tracks;
    }
    let n = stem0.usable_len();
    let (ol, or) = render_tier_stereo::<B>(
        &baseline.extract_tier(Tier::Ultra),
        &stem0.in_l[..n],
        &stem0.in_r[..n],
    );
    let base_rms = compute_broadband_rms(&ol, &or, warmup, baseline.mono_out);
    if base_rms < 0.02 * tgt_rms {
        return tracks;
    }
    tracks.push(build_scaled_stage2_track::<B>(baseline, stem0, 1.0, sr_hz));
    if dataset.stems.len() >= 2 {
        tracks.push(build_scaled_stage2_track::<B>(baseline, stem0, 0.70, sr_hz));
    }
    tracks
}

/// Evaluates HF harmonic generation (`> 4 kHz`) on band-limited sweep `[512..16_896)` so models
/// driving incoherent reverb/FDN tails preserve realistic HF harmonic synthesis on Holdout stems.
pub(super) struct SweepHarmonicCheck {
    stem: V3TrainStem,
    ltas_l: LtasTargetCache,
    ltas_r: Option<LtasTargetCache>,
    mono_out: bool,
}

impl SweepHarmonicCheck {
    pub(super) fn from_dataset(dataset: &V3TrainDataset, mono_out: bool) -> Option<Self> {
        let stem1 = dataset.stems.get(1)?;
        let n = stem1.usable_len().min(MAX_SWEEP_HF_SAMPLES);
        let warmup = 512.min(n / 4);
        if n <= warmup + 4_096 {
            return None;
        }
        let stem = V3TrainStem {
            in_l: stem1.in_l[..n].to_vec(),
            in_r: stem1.in_r[..n].to_vec(),
            tgt_l: stem1.tgt_l[..n].to_vec(),
            tgt_r: stem1.tgt_r[..n].to_vec(),
            warmup,
        };
        let tl: Vec<f64> = stem.tgt_l[warmup..].iter().map(|&v| f64::from(v)).collect();
        let ltas_l = LtasTargetCache::new(&tl, dataset.sample_rate_hz, 20_000.0)?;
        let ltas_r = (!mono_out)
            .then(|| {
                let tr: Vec<f64> = stem.tgt_r[warmup..].iter().map(|&v| f64::from(v)).collect();
                LtasTargetCache::new(&tr, dataset.sample_rate_hz, 20_000.0)
            })
            .flatten();
        Some(Self {
            stem,
            ltas_l,
            ltas_r,
            mono_out,
        })
    }

    pub(super) fn score_penalty<B: TrainerBackend>(&mut self, model: &V3MatryoshkaModel) -> f64 {
        let n = self.stem.usable_len();
        let w = self.stem.warmup;
        let (ol, or) = render_tier_stereo::<B>(
            &model.extract_tier(Tier::Ultra),
            &self.stem.in_l[..n],
            &self.stem.in_r[..n],
        );
        let mut pen = 0.0_f64;
        let check_ch = |cache: &mut LtasTargetCache, mdl: &[f32]| -> f64 {
            let m64: Vec<f64> = mdl[w..].iter().map(|&v| f64::from(v)).collect();
            let Some((py, pm)) = cache.band_powers(&m64) else {
                return 0.0;
            };
            let mean_py = (py.iter().sum::<f64>() / 33.0).max(1.0e-18);
            let mut max_hf_dev = 0.0_f64;
            for b in 22..32 {
                if py[b] > 1.0e-4 * mean_py {
                    let dev = (10.0 * (py[b].max(1.0e-24) / pm[b].max(1.0e-24)).log10()).abs();
                    max_hf_dev = max_hf_dev.max(dev);
                }
            }
            (max_hf_dev - 1.5).max(0.0) * 0.12
        };
        pen += check_ch(&mut self.ltas_l, &ol);
        if !self.mono_out
            && let Some(lr) = self.ltas_r.as_mut()
        {
            pen = pen.max(check_ch(lr, &or));
        }
        let _ = PI;
        pen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::ScalarTrainerBackend;
    use std::f64::consts::TAU;

    #[test]
    fn test_sweep_harmonic_check_penalizes_missing_hf_overtones() {
        let n = 16_384;
        let dry: Vec<f32> = (0..n)
            .map(|i| (0.4 * (TAU * 550.0 * (i as f64) / 48_000.0).sin()) as f32)
            .collect();
        let nl_tgt: Vec<f32> = dry
            .iter()
            .map(|&x| (4.5 * f64::from(x)).tanh() as f32 * 0.25)
            .collect();
        let dummy = V3TrainStem {
            in_l: dry.clone(),
            in_r: dry.clone(),
            tgt_l: nl_tgt.clone(),
            tgt_r: nl_tgt.clone(),
            warmup: 512,
        };
        let ds = V3TrainDataset::new(vec![dummy.clone(), dummy], 48_000.0);
        let mut chk = SweepHarmonicCheck::from_dataset(&ds, true).expect("sweep check builds");
        let base = V3MatryoshkaModel::dual_rate_default(true);
        let lin_pen = chk.score_penalty::<ScalarTrainerBackend>(&base);
        assert!(
            lin_pen > 0.2,
            "linear identity must be penalized on saturated sweep harmonics: {lin_pen}"
        );
    }
}
