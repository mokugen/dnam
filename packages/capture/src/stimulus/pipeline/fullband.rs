#![forbid(unsafe_code)]
//! `2.0 s` (`96,000`-frame) Full-Bandwidth (`20 Hz–20 kHz`) Identification Excitation & Causal
//! Black-Box Deadtime/Polarity Probe (`stimulus::pipeline::fullband`).

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};

use dnam_engine::schema::DnamError;

use crate::align::realign_fractional_kaiser64;
use crate::metrics::{TargetRepeatabilityProfile, build_repeatability_probe_stream};
use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::vault::VaultCaptureSpec;
use crate::resampler::{resample_arbitrary_rate_to_48k, resampled_frame_len};
use crate::solver::v3_varpro;
use crate::stimulus::laws::TopologyProbeMode;
use crate::stimulus::physics_probe;
use crate::stimulus::sealing::{
    StemSealRequest, resample_48k_channels_to_sr,
    seal_coresets_to_inpc_and_vault_with_stereo_override,
};
use crate::targets::{DistillationTarget, StemStimulus};
use crate::{CORESET_FP32_FRAMES_PER_STEM, metrics};

/// Returns the mono left channel of [`v3_varpro::build_v3_causal_probe_excitation`] at `48 kHz`.
#[must_use]
pub(super) fn build_v3_causal_probe_excitation_48k() -> Vec<f32> {
    v3_varpro::build_v3_causal_probe_excitation()
        .into_iter()
        .map(|lr| lr[0])
        .collect()
}

/// Estimates strictly causal deadtime in samples (`>= 0.0`) and polarity inversion from
/// [`v3_varpro::estimate_v3_causal_deadtime_and_polarity`].
#[must_use]
pub(super) fn estimate_v3_causal_deadtime_and_polarity(dry: &[f32], wet: &[f32]) -> (f64, bool) {
    v3_varpro::estimate_v3_causal_deadtime_and_polarity(dry, wet)
}

fn ac_rms_window(sig: &[f32], lo: usize, hi: usize) -> f64 {
    let hi_c = hi.min(sig.len());
    if hi_c <= lo {
        return 0.0;
    }
    let count = (hi_c - lo) as f64;
    let mean = sig[lo..hi_c].iter().map(|&v| f64::from(v)).sum::<f64>() / count;
    let var = sig[lo..hi_c]
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / count;
    var.sqrt()
}

const TI_PERIOD_SAMPLES: usize = 1_024;
const TI_ACTIVE_PERIODS: usize = 16;
const TI_ACTIVE_SAMPLES: usize = TI_PERIOD_SAMPLES * TI_ACTIVE_PERIODS;
const TI_FADE_SAMPLES: usize = 256;
const TI_ORBIT_LEN: usize = 2 * TI_PERIOD_SAMPLES;
const TI_ORBIT_A_START: usize = 11 * TI_PERIOD_SAMPLES;
const TI_ORBIT_B_START: usize = TI_ORBIT_A_START + TI_ORBIT_LEN;
const TI_TAIL_OFFSET_SAMPLES: usize = 5_120; // 106.67 ms > max dry/cab decay
const TI_TAIL_START_SAMPLES: usize = TI_ACTIVE_SAMPLES + TI_TAIL_OFFSET_SAMPLES;
const TI_TOTAL_SAMPLES: usize = 24_576;
const TI_HARMONICS: [usize; 8] = [4, 8, 12, 20, 32, 52, 84, 136];
const _: () = assert!(
    TI_FADE_SAMPLES < TI_ORBIT_A_START
        && TI_ORBIT_B_START + TI_ORBIT_LEN <= TI_ACTIVE_SAMPLES - TI_FADE_SAMPLES
        && TI_TAIL_START_SAMPLES < TI_TOTAL_SAMPLES
);

/// Synthesizes a `24,576`-sample (`512 ms @ 48 kHz`) periodic time-invariance and sustained-tail
/// probe: `16` exact `1,024`-sample harmonic periods (with `256`-sample raised-cosine fades) followed
/// by an `8,192`-sample (`170.7 ms`) silent tail `[16_384..24_576)`.
///
/// Every harmonic is a multiple of `4`, so deterministic nonlinear, `Op-P` sub-octave (`f0/2`,
/// `f0/4`), dynamic, and LTI systems converge onto an exact `2,048`-sample periodic orbit before
/// `[11_264..15_360)`, whereas time-varying modulation (chorus/rotary/LFO) breaks periodicity and
/// long echo/FDN tails exceed `106 ms` inside `[21_504..24_576)`.
#[must_use]
pub(super) fn build_time_invariance_probe_48k() -> Vec<f32> {
    let mut one_period = [0.0_f64; TI_PERIOD_SAMPLES];
    let mut peak = 1.0e-12_f64;
    for (i, slot) in one_period.iter_mut().enumerate() {
        let mut s = 0.0_f64;
        for (idx, &k) in TI_HARMONICS.iter().enumerate() {
            let phase = (idx as f64) * std::f64::consts::GOLDEN_RATIO;
            let amp = 1.0 / ((k as f64).sqrt());
            s += amp * (TAU * (k as f64) * (i as f64) / (TI_PERIOD_SAMPLES as f64) + phase).cos();
        }
        peak = peak.max(s.abs());
        *slot = s;
    }
    let scale = 0.25 / peak;
    let mut out = vec![0.0_f32; TI_TOTAL_SAMPLES];
    for (i, slot) in out[..TI_ACTIVE_SAMPLES].iter_mut().enumerate() {
        let env = if i < TI_FADE_SAMPLES {
            0.5 * (1.0 - (PI * (i as f64) / (TI_FADE_SAMPLES as f64)).cos())
        } else if i + TI_FADE_SAMPLES >= TI_ACTIVE_SAMPLES {
            let rem = TI_ACTIVE_SAMPLES - 1 - i;
            0.5 * (1.0 - (PI * (rem as f64) / (TI_FADE_SAMPLES as f64)).cos())
        } else {
            1.0
        };
        *slot = (env * scale * one_period[i % TI_PERIOD_SAMPLES]) as f32;
    }
    out
}

fn periodic_orbit_breaks_in_channel(wet: &[f32]) -> bool {
    if wet.len() < TI_ORBIT_B_START + TI_ORBIT_LEN {
        return false;
    }
    let a = &wet[TI_ORBIT_A_START..TI_ORBIT_B_START];
    let b = &wet[TI_ORBIT_B_START..TI_ORBIT_B_START + TI_ORBIT_LEN];
    let n_f = TI_ORBIT_LEN as f64;
    let mean_a = a.iter().map(|&v| f64::from(v)).sum::<f64>() / n_f;
    let mean_b = b.iter().map(|&v| f64::from(v)).sum::<f64>() / n_f;
    let (mut e_a, mut e_b, mut dot) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&va, &vb) in a.iter().zip(b) {
        let da = f64::from(va) - mean_a;
        let db = f64::from(vb) - mean_b;
        e_a += da * da;
        e_b += db * db;
        dot += da * db;
    }
    if e_a <= 1.0e-10 || e_b <= 1.0e-10 {
        return false;
    }
    // Optimal scalar gain absorbs ultra-slow envelope drift while exposing phase/delay modulation.
    let alpha = (dot / e_a).clamp(0.5, 2.0);
    let mut res_e = 0.0_f64;
    for (&va, &vb) in a.iter().zip(b) {
        let da = f64::from(va) - mean_a;
        let db = f64::from(vb) - mean_b;
        let d = db - alpha * da;
        res_e += d * d;
    }
    (res_e / e_b) > 0.015
}

fn channel_requires_contiguous_96k(wet: &[f32]) -> bool {
    let rms_act = ac_rms_window(wet, TI_ORBIT_A_START, TI_ORBIT_B_START + TI_ORBIT_LEN);
    let rms_tail = ac_rms_window(wet, TI_TAIL_START_SAMPLES, TI_TOTAL_SAMPLES);
    if rms_act > 1.0e-5 && rms_tail > 0.020 * rms_act {
        return true;
    }
    periodic_orbit_breaks_in_channel(wet)
}

/// Detects sustained FDN/echo tails (`> 106 ms`) or time-varying modulated delay lines from the
/// periodic time-invariance probe so `StarCenter` captures a full contiguous `96,000`-sample
/// multisine without sweep reset.
#[must_use]
pub(super) fn probe_requires_contiguous_96k_fullband(
    ti_wet_l: &[f32],
    ti_wet_r: Option<&[f32]>,
) -> bool {
    channel_requires_contiguous_96k(ti_wet_l)
        || ti_wet_r.is_some_and(channel_requires_contiguous_96k)
}

/// Synthesizes a `96,000`-frame (`2.0 s @ 48 kHz`) full-bandwidth (`20 Hz–20 kHz`) identification
/// excitation covering all `33` `1/3`-octave `LTAS` bands, musical transients, dynamic levels,
/// and a `24,000`-sample logarithmic `20 Hz–20 kHz` sweep (`[0..72_000)` multi-regime + `[72_000..96_000)` sweep).
#[cfg(test)]
#[must_use]
pub(super) fn build_fullband_identification_coreset_48k(
    seed_256: &[u8; 256],
    stereo: bool,
    aux_key_mode: bool,
    clamp_dac: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    build_fullband_identification_coreset_48k_opts(seed_256, stereo, aux_key_mode, clamp_dac, false)
}

/// Option-parameterized variant of [`build_fullband_identification_coreset_48k`] supporting
/// contiguous `96,000`-sample multisine capture for incoherent modulation / echo / FDN targets.
#[must_use]
pub(super) fn build_fullband_identification_coreset_48k_opts(
    seed_256: &[u8; 256],
    stereo: bool,
    aux_key_mode: bool,
    clamp_dac: bool,
    contiguous_96k: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n = CORESET_FP32_FRAMES_PER_STEM;
    let sr = 48_000.0_f64;
    let body_len = if contiguous_96k {
        n
    } else {
        72_000usize.min(n)
    };
    let centers = metrics::canonical_ltas_33_centers_hz();
    let weights: [f64; 33] =
        std::array::from_fn(|k| 1.0 / (1.0 + (centers[k] / 2_400.0).powi(2)).sqrt());
    let norm = weights
        .iter()
        .map(|w| w * w)
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    let scale = 0.52 / norm;
    let seed_phases: [f64; 33] = std::array::from_fn(|k| {
        let base = ((k as f64 + 1.0) * std::f64::consts::GOLDEN_RATIO).fract() * TAU;
        let tweak = (f64::from(seed_256[k % seed_256.len()]) / 255.0 - 0.5) * 0.25;
        base + tweak
    });

    let mut dry_l = vec![0.0_f32; n];
    let mut dry_r = stereo.then(|| vec![0.0_f32; n]);
    let settle_samples = if contiguous_96k {
        0
    } else {
        1_024usize.min(body_len / 8)
    };
    let active_body = body_len.saturating_sub(settle_samples);

    // Part 1 [0..70_976) (or contiguous [0..96_000)): pink-weighted 33-band excitation
    for i in 0..active_body {
        let t = (i as f64) / sr;
        let fade_in = if i < 128 {
            0.5 * (1.0 - (PI * (i as f64) / 128.0).cos())
        } else {
            1.0
        };
        let fade_out = if !contiguous_96k && i + 64 >= active_body {
            let rem = active_body - 1 - i;
            0.5 * (1.0 - (PI * (rem as f64) / 64.0).cos())
        } else {
            1.0
        };
        let env = fade_in
            * fade_out
            * (0.45 + 0.55 * (-((i % 2_400) as f64) / 800.0).exp())
            * (0.72 + 0.28 * (TAU * 2.3 * t).sin());
        let mut sl = 0.0_f64;
        let mut sr_sum = 0.0_f64;
        for (k, &fc) in centers.iter().enumerate() {
            let arg = TAU * fc * t;
            let phi_l = seed_phases[k];
            sl += weights[k] * (arg + phi_l).cos();
            if stereo {
                let phi_r = phi_l + 0.29 * ((k & 3) as f64 + 0.5);
                sr_sum += weights[k] * (arg + phi_r).cos();
            }
        }
        let dl_raw = env * scale * sl;
        dry_l[i] = if clamp_dac || aux_key_mode {
            dl_raw.clamp(-0.90, 0.90) as f32
        } else {
            dl_raw as f32
        };
        if let Some(dr) = dry_r.as_mut() {
            if aux_key_mode {
                dr[i] = if i >= 57_000 {
                    let scale_mod = 0.65 + 0.35 * (0.5 + 0.5 * (TAU * 1.9 * t).sin());
                    ((scale_mod * 0.96) * (dry_l[i] as f64)).clamp(-0.90, 0.90) as f32
                } else {
                    let raw_env_r = if i < 2_600 {
                        (0.88 + 0.17 * fade_in).clamp(0.04, 1.05)
                    } else if i < 4_400 || (40_000..57_000).contains(&i) {
                        0.04
                    } else if i < 22_000 {
                        let swell_r = 0.5 - 0.5 * (TAU * 3.1 * t).cos();
                        let trans_r = (-(((i - 4_400) % 2_000) as f64) / 550.0).exp();
                        (0.16 + 0.42 * (0.55 * swell_r + 0.45 * trans_r)).clamp(0.16, 0.58)
                    } else {
                        let trans_r = (-(((i - 22_000) % 2_400) as f64) / 800.0).exp();
                        let pulse_r = 0.5 - 0.5 * (TAU * 2.3 * t + 0.4).cos();
                        (0.62 + 0.43 * (0.50 * pulse_r + 0.50 * trans_r)).clamp(0.62, 1.05)
                    };
                    let env_r = fade_in * fade_out * raw_env_r;
                    (env_r * scale * sr_sum).clamp(-0.90, 0.90) as f32
                };
            } else {
                let dr_raw = env * scale * sr_sum;
                dr[i] = if clamp_dac {
                    dr_raw.clamp(-0.90, 0.90) as f32
                } else {
                    dr_raw as f32
                };
            }
        }
    }

    // Part 2 [72_000..96_000): modulated logarithmic 20 Hz -> 20 kHz identification sweep
    let sweep_start = body_len;
    let sweep_len = n.saturating_sub(sweep_start);
    let mut phase = 0.0_f64;
    let f0 = 20.0_f64;
    let ratio = (20_000.0_f64 / f0).ln();
    for i in 0..sweep_len {
        let frac = (i as f64) / (sweep_len.max(1) as f64);
        let t = (i as f64) / sr;
        let freq = f0 * (ratio * frac).exp();
        phase += TAU * freq / sr;
        let fade_in = if i < 64 {
            0.5 * (1.0 - (PI * (i as f64) / 64.0).cos())
        } else {
            1.0
        };
        let fade_out = if i + 64 >= sweep_len {
            let rem = sweep_len - 1 - i;
            0.5 * (1.0 - (PI * (rem as f64) / 64.0).cos())
        } else {
            1.0
        };
        let pick_decay = (-((i % 1_600) as f64) / 480.0).exp();
        let swell = (0.5 - 0.5 * (TAU * 1.6 * t).cos()).powf(1.4);
        let amp = fade_in * fade_out * (0.03 + 0.39 * swell * (0.25 + 0.75 * pick_decay));
        let bass_mod_raw = 0.28 * (TAU * 98.0 * t + 0.3 * (TAU * 2.1 * t).sin()).sin();
        let mod_gate = if i < 6_144 {
            0.0
        } else if i < 6_400 {
            0.5 * (1.0 - (PI * ((i - 6_144) as f64) / 256.0).cos())
        } else {
            1.0
        };
        let bass_mod = mod_gate * bass_mod_raw;
        dry_l[sweep_start + i] = (amp * (0.72 * phase.sin() + bass_mod)).clamp(-0.90, 0.90) as f32;
        if let Some(dr) = dry_r.as_mut() {
            let amp_r = if aux_key_mode {
                let raw_r = if (7_200..13_200).contains(&i) {
                    0.04
                } else if (13_200..18_600).contains(&i) {
                    let swell_r = 0.5 - 0.5 * (TAU * 2.1 * t).cos();
                    let trans_r = (-((i % 1_800) as f64) / 450.0).exp();
                    (0.18 + 0.37 * (0.50 * swell_r + 0.50 * trans_r)).clamp(0.18, 0.55)
                } else {
                    let swell_r = 0.5 + 0.5 * (TAU * 1.45 * t + 0.6).cos();
                    let trans_r = (-(((i + 600) % 2_100) as f64) / 500.0).exp();
                    (0.55 + 0.45 * (0.55 * swell_r + 0.45 * trans_r)).clamp(0.45, 1.00)
                };
                fade_in * fade_out * raw_r
            } else {
                amp
            };
            let val_r = amp_r * (0.70 * (phase + 0.08).sin() + 0.95 * bass_mod);
            dr[sweep_start + i] = val_r.clamp(-0.90, 0.90) as f32;
        }
    }

    (dry_l, dry_r)
}

/// Parameters describing a full-bandwidth (`96,000`-frame) identification render & seal request.
pub(super) struct FullbandStemRequest<'a> {
    pub stem_id: &'a str,
    pub sample_rate_hz: u32,
    pub topology_mode: TopologyProbeMode,
    pub seed_256: &'a [u8; 256],
    pub carrier_bins: &'a [usize],
    pub valid_frames: usize,
    pub full_dry_l_48k: &'a [f32],
    pub full_dry_r_48k: Option<&'a [f32]>,
    pub ctrl_start: &'a [f32],
    pub ctrl_end: Option<&'a [f32]>,
    pub is_star_center: bool,
    pub windows: Vec<StmcTrajectoryWindow>,
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    pub shared_align: (f64, bool, bool),
    pub compact_render_frames_48k: Option<usize>,
}

/// Renders a shared `full_dry` identification stem (or compact leading slice for stationary Stage C
/// stops/switches) through `target` and seals `INPC` + `VaultCaptureSpec` under causal alignment.
pub(super) fn render_and_seal_fullband_stem(
    target: &mut dyn DistillationTarget,
    mut req: FullbandStemRequest<'_>,
) -> Result<(Vec<u8>, VaultCaptureSpec), DnamError> {
    let sr = req.sample_rate_hz.clamp(8_000, 384_000);
    let need_aux = matches!(
        req.topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let render_len_48k =
        req.compact_render_frames_48k
            .map_or(req.full_dry_l_48k.len(), |compact_n| {
                let pad = (req.shared_align.0.ceil().max(0.0) as usize) + 128;
                (compact_n + pad).clamp(1_024, req.full_dry_l_48k.len())
            });
    let dry_l_render = &req.full_dry_l_48k[..render_len_48k];
    let dry_r_render = req
        .full_dry_r_48k
        .map(|r| &r[..render_len_48k.min(r.len())]);
    let aux_48k = need_aux.then_some(dry_l_render);
    let (in_1, in_2, in_3) = resample_48k_channels_to_sr(sr, dry_l_render, dry_r_render, aux_48k);

    let trajs = req.ctrl_end.map(|end_c| {
        let len = in_1.len().max(1);
        req.ctrl_start
            .iter()
            .zip(end_c.iter())
            .map(|(&c0, &c1)| {
                if (c1 - c0).abs() <= 1.0e-7 {
                    vec![c0; len]
                } else {
                    (0..len)
                        .map(|t| {
                            let alpha = (t as f32) / ((len - 1).max(1) as f32);
                            c0 + (c1 - c0) * alpha
                        })
                        .collect()
                }
            })
            .collect::<Vec<Vec<f32>>>()
    });

    let stim = StemStimulus {
        stem_id: req.stem_id.to_string(),
        sample_rate_hz: sr,
        in_1,
        in_2,
        in_3,
        in_4: None,
        control_values: req.ctrl_start.to_vec(),
        control_trajectories: trajs,
        freeze_transition_window: None,
    };
    target.reset_state();
    let resp = target.render_stem(&stim)?;
    let resp_sr = resp.sample_rate_hz.clamp(8_000, 384_000);
    let to_48k = |ch: &[f32]| -> Vec<f32> {
        if resp_sr == 48_000 {
            ch.to_vec()
        } else {
            let dst = resampled_frame_len(ch.len(), resp_sr, 48_000);
            resample_arbitrary_rate_to_48k(ch, resp_sr, Some(dst))
        }
    };
    let wet_l_48k = to_48k(&resp.out_1);
    let wet_r_48k = resp.out_2.as_deref().map(to_48k);

    let star_phys = if req.is_star_center {
        let probe_len = 13_824.min(req.full_dry_l_48k.len()).min(wet_l_48k.len());
        let comb_len = 4_608.min(probe_len);
        let phys = physics_probe::extract_star_physics_from_audio(
            &req.full_dry_l_48k[..probe_len],
            req.full_dry_r_48k.map(|r| &r[..probe_len.min(r.len())]),
            &wet_l_48k[..probe_len],
            wet_r_48k.as_deref().map(|r| &r[..probe_len.min(r.len())]),
            &req.full_dry_l_48k[..comb_len],
            &wet_l_48k[..comb_len],
            wet_r_48k.as_deref().map(|r| &r[..comb_len.min(r.len())]),
            &wet_l_48k[wet_l_48k.len().saturating_sub(4_608)..],
            wet_r_48k
                .as_deref()
                .map(|r| &r[r.len().saturating_sub(4_608)..]),
            req.carrier_bins,
        );
        req.user_metadata.extend(phys.metadata);
        Some(phys.extracted)
    } else {
        None
    };

    let (inpc, mut spec, _, _) = seal_coresets_to_inpc_and_vault_with_stereo_override(
        StemSealRequest {
            stem_id: req.stem_id,
            is_star_center: req.is_star_center,
            seed_256: req.seed_256,
            carrier_bins: req.carrier_bins,
            valid_frames: req.valid_frames,
            dry_48k_l: req.full_dry_l_48k,
            dry_48k_r: req.full_dry_r_48k,
            wet_48k_l: &wet_l_48k,
            wet_48k_r: wet_r_48k.as_deref(),
            windows: req.windows,
            controls_meta: req.controls_meta,
            user_metadata: req.user_metadata,
        },
        Some(req.shared_align),
    )?;
    if let Some(ext) = star_phys {
        spec.extracted_physics = Some(ext);
    }
    Ok((inpc, spec))
}

/// Renders [`build_repeatability_probe_stream`] through `target` and measures empirical per-band
/// repeatability floors (`TargetRepeatabilityProfile`) strictly from `(x, y)` probes.
pub(super) fn probe_target_repeatability_profiles(
    target: &mut dyn DistillationTarget,
    sample_rate_hz: u32,
    topology_mode: TopologyProbeMode,
    control_values: &[f32],
    shared_align: (f64, bool, bool),
) -> Result<[TargetRepeatabilityProfile; 2], DnamError> {
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let (rep_dry, rep_windows, rep_late_w) = build_repeatability_probe_stream(48_000.0);
    let dry_l: Vec<f32> = rep_dry.iter().map(|lr| lr[0]).collect();
    let dry_r: Vec<f32> = rep_dry.iter().map(|lr| lr[1]).collect();
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let (in_1, in_2, in_3) = resample_48k_channels_to_sr(
        sr,
        &dry_l,
        need_stereo.then_some(dry_r.as_slice()),
        need_aux.then_some(dry_l.as_slice()),
    );

    target.reset_state();
    let resp = target.render_stem(&StemStimulus {
        stem_id: "repeatability_probe".to_string(),
        sample_rate_hz: sr,
        in_1,
        in_2,
        in_3,
        in_4: None,
        control_values: control_values.to_vec(),
        control_trajectories: None,
        freeze_transition_window: None,
    })?;
    let resp_sr = resp.sample_rate_hz.clamp(8_000, 384_000);
    let to_48k = |ch: &[f32]| -> Vec<f32> {
        if resp_sr == 48_000 {
            ch.to_vec()
        } else {
            let dst = resampled_frame_len(ch.len(), resp_sr, 48_000);
            resample_arbitrary_rate_to_48k(ch, resp_sr, Some(dst))
        }
    };
    let (causal_lag, inv_l, inv_r) = shared_align;
    let al_l = realign_fractional_kaiser64(&to_48k(&resp.out_1), causal_lag, 0.0, inv_l);
    let wl: Vec<f64> = al_l.iter().map(|&v| f64::from(v)).collect();
    let prof_l =
        TargetRepeatabilityProfile::from_probe_response(&wl, &rep_windows, rep_late_w, 48_000.0);
    let prof_r = resp.out_2.as_deref().map_or(prof_l, |o2| {
        let al_r = realign_fractional_kaiser64(&to_48k(o2), causal_lag, 0.0, inv_r);
        let wr: Vec<f64> = al_r.iter().map(|&v| f64::from(v)).collect();
        TargetRepeatabilityProfile::from_probe_response(&wr, &rep_windows, rep_late_w, 48_000.0)
    });
    Ok([prof_l, prof_r])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_requires_contiguous_96k_fullband_flags_tail_and_lfo() {
        let ti_dry = build_time_invariance_probe_48k();
        let seed = [0x42_u8; 256];

        // Deterministic static waveshaper + short 350-sample reflection -> standard [70_976 + sweep] path.
        let mut static_ti_wet = vec![0.0_f32; ti_dry.len()];
        for (i, slot) in static_ti_wet.iter_mut().enumerate() {
            let early = if i >= 350 { ti_dry[i - 350] } else { 0.0 };
            let shaped = (3.5 * f64::from(ti_dry[i])).tanh() as f32;
            *slot = 0.7 * shaped + 0.3 * early;
        }
        assert!(!probe_requires_contiguous_96k_fullband(
            &static_ti_wet,
            None
        ));

        // Sustained FDN / BBD tail past 106 ms ([21_504..24_576)) -> contiguous 96k coreset.
        let mut tail_ti_wet = static_ti_wet.clone();
        for (i, slot) in tail_ti_wet[TI_TAIL_START_SAMPLES..TI_TOTAL_SAMPLES]
            .iter_mut()
            .enumerate()
        {
            *slot = 0.05 * ((i as f32) * 0.11).sin();
        }
        assert!(probe_requires_contiguous_96k_fullband(&tail_ti_wet, None));

        // Modulated chorus lag shift breaking 2048-sample periodic orbit -> contiguous 96k coreset.
        let mut mod_ti_wet = vec![0.0_f32; ti_dry.len()];
        for (i, slot) in mod_ti_wet.iter_mut().enumerate() {
            let t = (i as f64) / 48_000.0;
            let lag = (400.0 + 35.0 * (TAU * 2.0 * t).sin()).round() as usize;
            let wet = if i >= lag { ti_dry[i - lag] } else { 0.0 };
            *slot = 0.55 * ti_dry[i] + 0.45 * wet;
        }
        assert!(probe_requires_contiguous_96k_fullband(&mod_ti_wet, None));

        let (ctg_l, _) =
            build_fullband_identification_coreset_48k_opts(&seed, false, false, false, true);
        let (std_l, _) = build_fullband_identification_coreset_48k(&seed, false, false, false);
        assert!(ctg_l[71_100..71_900].iter().any(|&v| v.abs() > 1.0e-3));
        assert!(std_l[71_100..71_900].iter().all(|&v| v.abs() <= 1.0e-6));
    }
}
