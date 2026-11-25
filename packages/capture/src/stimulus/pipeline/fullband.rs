#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Full-Bandwidth `2.0–2.5 s` (`96,000–120,000`-frame) `Schoukens–Pintelon` Stem Rendering,
//! Hybrid Alignment Sealing, Repeatability Profiling & Causal Probe Bridge (`stimulus::pipeline::fullband`).

use std::collections::BTreeMap;

use dnam_engine::schema::DnamError;

use crate::align::realign_fractional_kaiser64;
use crate::metrics::{TargetRepeatabilityProfile, build_repeatability_probe_stream};
use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::schema::ExtractedPhysics;
use crate::proj::vault::VaultCaptureSpec;
use crate::resampler::{resample_arbitrary_rate_to_48k, resampled_frame_len};
use crate::stimulus::laws::TopologyProbeMode;
use crate::stimulus::physics_probe;
use crate::stimulus::sealing::{
    SharedStemAlignment, StemSealRequest, resample_48k_channels_to_sr,
    seal_coresets_to_inpc_and_vault_with_full_align,
};
use crate::targets::{DistillationTarget, StemStimulus};

use crate::stimulus::pipeline::causal_probe::build_v3_causal_probe_excitation;
use crate::stimulus::pipeline::schoukens::{
    OddRandomMultisinePlan, Regime2DecayInput, SchoukensAnalysisConfig,
    analyze_schoukens_pintelon_physics,
};

mod causal_guard;
mod star_schoukens;

pub(super) use star_schoukens::{StarCenterCaptureRequest, capture_fullband_star_center_schoukens};

/// Returns the mono left channel of [`build_v3_causal_probe_excitation`] at `48 kHz`.
#[must_use]
pub(super) fn build_v3_causal_probe_excitation_48k() -> Vec<f32> {
    build_v3_causal_probe_excitation()
        .into_iter()
        .map(|lr| lr[0])
        .collect()
}

/// Resamples a rendered target response slice from `resp_sr` back onto the canonical `48 kHz` grid.
pub(super) fn normalize_response_to_48k(ch: &[f32], resp_sr: u32) -> Vec<f32> {
    let clamped = resp_sr.clamp(8_000, 384_000);
    if clamped == 48_000 {
        ch.to_vec()
    } else {
        let dst = resampled_frame_len(ch.len(), clamped, 48_000);
        resample_arbitrary_rate_to_48k(ch, clamped, Some(dst))
    }
}

/// Input stem buffers and control sweeps for [`render_stem_48k`].
#[derive(Debug)]
pub(super) struct StemRenderRequest<'a> {
    pub stem_id: &'a str,
    pub sample_rate_hz: u32,
    pub topology_mode: TopologyProbeMode,
    pub dry_l_48k: &'a [f32],
    pub dry_r_48k: Option<&'a [f32]>,
    pub aux_in_3_48k: Option<&'a [f32]>,
    pub ctrl_start: &'a [f32],
    pub ctrl_end: Option<&'a [f32]>,
}

/// Renders a canonical `48 kHz` multi-channel stem through `target` (resampling cleanly to/from
/// `sample_rate_hz`) and returns `(wet_l_48k, Option<wet_r_48k>)`.
pub(super) fn render_stem_48k(
    target: &mut dyn DistillationTarget,
    req: StemRenderRequest<'_>,
) -> Result<(Vec<f32>, Option<Vec<f32>>), DnamError> {
    let sr = req.sample_rate_hz.clamp(8_000, 384_000);
    let need_aux = matches!(
        req.topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let fallback_aux = need_aux.then_some(req.dry_l_48k);
    let eff_aux = req.aux_in_3_48k.or(fallback_aux);
    let (in_1, in_2, in_3) = resample_48k_channels_to_sr(sr, req.dry_l_48k, req.dry_r_48k, eff_aux);

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
    let wet_l_48k = normalize_response_to_48k(&resp.out_1, resp.sample_rate_hz);
    let wet_r_48k = resp
        .out_2
        .as_deref()
        .map(|o2| normalize_response_to_48k(o2, resp.sample_rate_hz));
    Ok((wet_l_48k, wet_r_48k))
}

/// Parameters describing a full-bandwidth (`96,000–120,000`-frame) stem seal request.
#[derive(Debug)]
pub(super) struct FullbandStemRequest<'a> {
    pub stem_id: &'a str,
    pub sample_rate_hz: u32,
    pub topology_mode: TopologyProbeMode,
    pub seed_256: &'a [u8; 256],
    pub carrier_bins: &'a [usize],
    pub valid_frames: usize,
    pub full_dry_l_48k: &'a [f32],
    pub full_dry_r_48k: Option<&'a [f32]>,
    pub aux_in_3_48k: Option<&'a [f32]>,
    pub ctrl_start: &'a [f32],
    pub ctrl_end: Option<&'a [f32]>,
    pub is_star_center: bool,
    pub windows: Vec<StmcTrajectoryWindow>,
    pub controls_meta: BTreeMap<String, serde_json::Value>,
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    pub shared_align: SharedStemAlignment,
    pub optional_spcm: Option<Vec<u8>>,
    pub pre_extracted_physics: Option<(ExtractedPhysics, BTreeMap<String, serde_json::Value>)>,
    pub pre_rendered_aligned_wet: Option<(Vec<f32>, Option<Vec<f32>>)>,
}

/// Input slices for [`compute_star_schoukens_physics`].
#[derive(Debug)]
pub(super) struct StarPhysicsInputs<'a> {
    pub plan: &'a OddRandomMultisinePlan,
    pub r1_dry_period_l: &'a [f32],
    pub r1_settled_periods_l: &'a [&'a [f32]],
    pub r1_settled_periods_r: Option<&'a [&'a [f32]]>,
    pub aligned_wet_r2_l: &'a [f32],
    pub aligned_wet_r2_r: Option<&'a [f32]>,
    pub r2_gap_ranges: &'a [(usize, usize)],
    pub coreset_dry_l: &'a [f32],
    pub coreset_dry_r: Option<&'a [f32]>,
    pub coreset_wet_l: &'a [f32],
    pub coreset_wet_r: Option<&'a [f32]>,
    pub probe_comb_dry_l: &'a [f32],
    pub probe_comb_wet_l: &'a [f32],
    pub probe_comb_wet_r: Option<&'a [f32]>,
    pub probe_carrier_bins: &'a [usize],
    pub coreset_regime_offsets: [u32; 5],
    pub calb_noise_floor_dbfs: f32,
}

/// Computes merged `ExtractedPhysics` + `user_metadata` combining comb/echo probe telemetry and
/// `Schoukens–Pintelon` (`BLA`, even/odd distortion, `Law 0.A` variance, `EDC` $T_{60}$, offsets).
pub(super) fn compute_star_schoukens_physics(
    inputs: StarPhysicsInputs<'_>,
) -> (ExtractedPhysics, BTreeMap<String, serde_json::Value>) {
    let coreset_probe_len = 13_824
        .min(inputs.coreset_dry_l.len())
        .min(inputs.coreset_wet_l.len());
    let comb_len = 4_608
        .min(inputs.probe_comb_dry_l.len())
        .min(inputs.probe_comb_wet_l.len());
    let tail_start = inputs.r2_gap_ranges.last().map_or(
        inputs.probe_comb_wet_l.len().saturating_sub(4_608),
        |&(gs, ge)| {
            if ge.saturating_sub(gs) >= 4_608 && ge <= inputs.aligned_wet_r2_l.len() {
                gs
            } else {
                inputs.probe_comb_wet_l.len().saturating_sub(4_608)
            }
        },
    );
    let (tail_l, tail_r) = if let Some(&(gs, ge)) = inputs.r2_gap_ranges.last()
        && ge.saturating_sub(gs) >= 4_608
        && ge <= inputs.aligned_wet_r2_l.len()
    {
        (
            &inputs.aligned_wet_r2_l[tail_start..tail_start + 4_608],
            inputs
                .aligned_wet_r2_r
                .filter(|r| tail_start + 4_608 <= r.len())
                .map(|r| &r[tail_start..tail_start + 4_608])
                .or_else(|| {
                    inputs
                        .probe_comb_wet_r
                        .map(|r| &r[r.len().saturating_sub(4_608)..])
                }),
        )
    } else {
        (
            &inputs.probe_comb_wet_l[inputs.probe_comb_wet_l.len().saturating_sub(4_608)..],
            inputs
                .probe_comb_wet_r
                .map(|r| &r[r.len().saturating_sub(4_608)..]),
        )
    };

    let mut base = physics_probe::extract_star_physics_from_audio(
        &inputs.coreset_dry_l[..coreset_probe_len],
        inputs
            .coreset_dry_r
            .map(|r| &r[..coreset_probe_len.min(r.len())]),
        &inputs.coreset_wet_l[..coreset_probe_len],
        inputs
            .coreset_wet_r
            .map(|r| &r[..coreset_probe_len.min(r.len())]),
        &inputs.probe_comb_dry_l[..comb_len],
        &inputs.probe_comb_wet_l[..comb_len],
        inputs.probe_comb_wet_r.map(|r| &r[..comb_len.min(r.len())]),
        tail_l,
        tail_r,
        inputs.probe_carrier_bins,
    );

    let sp = analyze_schoukens_pintelon_physics(
        inputs.plan,
        inputs.r1_dry_period_l,
        inputs.r1_settled_periods_l,
        inputs.r1_settled_periods_r,
        Regime2DecayInput {
            wet_l: inputs.aligned_wet_r2_l,
            wet_r: inputs.aligned_wet_r2_r,
            gap_ranges: inputs.r2_gap_ranges,
        },
        SchoukensAnalysisConfig {
            calb_noise_floor_dbfs: inputs.calb_noise_floor_dbfs,
            coreset_regime_offsets: inputs.coreset_regime_offsets,
        },
    );
    sp.apply_to_extracted_physics(&mut base.extracted);
    if !sp.edc_t60_s.is_finite() || sp.edc_t60_s == 0.0 {
        base.extracted.edc_t60_s = 0.0;
    } else {
        base.metadata.insert(
            "phys_rt60_s".to_string(),
            serde_json::json!(f64::from(sp.edc_t60_s)),
        );
    }
    (base.extracted, base.metadata)
}

/// Renders a shared `full_dry` coreset through `target` (or seals pre-rendered condensed `SPCM`
/// slices) and seals `INPC` + `VaultCaptureSpec` under full hybrid lag + PPM alignment.
pub(super) fn render_and_seal_fullband_stem(
    target: &mut dyn DistillationTarget,
    mut req: FullbandStemRequest<'_>,
) -> Result<(Vec<u8>, VaultCaptureSpec), DnamError> {
    let used_pre_rendered = req.pre_rendered_aligned_wet.is_some();
    let (wet_l_48k, wet_r_48k, eff_align) =
        if let Some((al_l, al_r)) = req.pre_rendered_aligned_wet.take() {
            (
                al_l,
                al_r,
                SharedStemAlignment {
                    lag_samples: 0.0,
                    clock_drift_ppm: 0.0,
                    invert_l: false,
                    invert_r: false,
                },
            )
        } else {
            let (wl, wr) = render_stem_48k(
                target,
                StemRenderRequest {
                    stem_id: req.stem_id,
                    sample_rate_hz: req.sample_rate_hz,
                    topology_mode: req.topology_mode,
                    dry_l_48k: req.full_dry_l_48k,
                    dry_r_48k: req.full_dry_r_48k,
                    aux_in_3_48k: req.aux_in_3_48k,
                    ctrl_start: req.ctrl_start,
                    ctrl_end: req.ctrl_end,
                },
            )?;
            (wl, wr, req.shared_align)
        };

    let mut extracted_opt = None;
    if let Some((ep, meta)) = req.pre_extracted_physics.take() {
        req.user_metadata.extend(meta);
        extracted_opt = Some(ep);
    }

    let (inpc, mut spec, _, _) = seal_coresets_to_inpc_and_vault_with_full_align(
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
        Some(eff_align),
    )?;
    if used_pre_rendered {
        spec.fractional_lag_samples_f32 = req.shared_align.lag_samples as f32;
        if req.shared_align.invert_l {
            spec.user_metadata
                .insert("wet_polarity_inverted".to_string(), serde_json::json!(true));
        }
        if req.shared_align.invert_r {
            spec.user_metadata.insert(
                "wet_r_polarity_inverted".to_string(),
                serde_json::json!(true),
            );
        }
        if req.shared_align.clock_drift_ppm != 0.0 {
            spec.user_metadata.insert(
                "clock_drift_ppm".to_string(),
                serde_json::json!(req.shared_align.clock_drift_ppm),
            );
        }
    }
    if let Some(ext) = extracted_opt {
        spec.extracted_physics = Some(ext);
    }
    if req.optional_spcm.is_some() {
        spec.optional_spcm = req.optional_spcm;
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
    shared_align: SharedStemAlignment,
) -> Result<[TargetRepeatabilityProfile; 2], DnamError> {
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let (rep_dry, rep_windows, rep_late_w) = build_repeatability_probe_stream(48_000.0);
    let dry_l: Vec<f32> = rep_dry.iter().map(|lr| lr[0]).collect();
    let dry_r: Vec<f32> = rep_dry.iter().map(|lr| lr[1]).collect();
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let (wet_l_48k, wet_r_48k) = render_stem_48k(
        target,
        StemRenderRequest {
            stem_id: "repeatability_probe",
            sample_rate_hz: sr,
            topology_mode,
            dry_l_48k: &dry_l,
            dry_r_48k: need_stereo.then_some(dry_r.as_slice()),
            aux_in_3_48k: None,
            ctrl_start: control_values,
            ctrl_end: None,
        },
    )?;
    let al_l = realign_fractional_kaiser64(
        &wet_l_48k,
        shared_align.lag_samples,
        shared_align.clock_drift_ppm,
        shared_align.invert_l,
    );
    let wl: Vec<f64> = al_l.iter().map(|&v| f64::from(v)).collect();
    let prof_l =
        TargetRepeatabilityProfile::from_probe_response(&wl, &rep_windows, rep_late_w, 48_000.0);
    let prof_r = wet_r_48k.as_deref().map_or(prof_l, |o2| {
        let al_r = realign_fractional_kaiser64(
            o2,
            shared_align.lag_samples,
            shared_align.clock_drift_ppm,
            shared_align.invert_r,
        );
        let wr: Vec<f64> = al_r.iter().map(|&v| f64::from(v)).collect();
        TargetRepeatabilityProfile::from_probe_response(&wr, &rep_windows, rep_late_w, 48_000.0)
    });
    Ok([prof_l, prof_r])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::proj::capsules::{decode_proj_inpc_f32, decode_proj_stmc_f32};
    use crate::proj::directory::{verify_proj_inpc_capsule, verify_proj_stmc_capsule};
    use crate::stimulus::pipeline::schoukens::{
        SCHOUKENS_CORESET_25S_OFFSETS, build_schoukens_pintelon_coreset_48k,
    };
    use crate::targets::StemResponse;

    struct PassTarget;
    impl DistillationTarget for PassTarget {
        fn reset_state(&mut self) {}
        fn render_stem(&mut self, s: &StemStimulus) -> Result<StemResponse, DnamError> {
            Ok(StemResponse {
                stem_id: s.stem_id.clone(),
                sample_rate_hz: s.sample_rate_hz,
                out_1: s.in_1.clone(),
                out_2: None,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn test_fullband_render_and_seal_120k_schoukens_with_ppm_and_aux() {
        let seed = [0x61_u8; 256];
        let exc = build_schoukens_pintelon_coreset_48k(
            &seed,
            TopologyProbeMode::Mono1To1,
            120_000,
            &BTreeSet::new(),
        );
        let mut target = PassTarget;
        let win = vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 0.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        }];
        let align = SharedStemAlignment {
            lag_samples: 0.0,
            clock_drift_ppm: 0.0,
            invert_l: false,
            invert_r: false,
        };
        let (inpc, spec) = render_and_seal_fullband_stem(
            &mut target,
            FullbandStemRequest {
                stem_id: "star_120k",
                sample_rate_hz: 48_000,
                topology_mode: TopologyProbeMode::Mono1To1,
                seed_256: &seed,
                carrier_bins: &[23, 57],
                valid_frames: 6,
                full_dry_l_48k: &exc.dry_l,
                full_dry_r_48k: None,
                aux_in_3_48k: None,
                ctrl_start: &[],
                ctrl_end: None,
                is_star_center: true,
                windows: win,
                controls_meta: BTreeMap::new(),
                user_metadata: BTreeMap::new(),
                shared_align: align,
                optional_spcm: None,
                pre_extracted_physics: None,
                pre_rendered_aligned_wet: None,
            },
        )
        .unwrap();
        let inpc_view = verify_proj_inpc_capsule(&inpc).unwrap();
        assert_eq!(inpc_view.num_frames, 120_000);
        let (dl, _) = decode_proj_inpc_f32(&inpc_view);
        assert_eq!(dl.len(), SCHOUKENS_CORESET_25S_OFFSETS[4] as usize);

        let stmc_bytes = {
            let mut f = crate::proj::STMC_FLAG_ENABLED | crate::proj::STMC_FLAG_STAR_CENTER;
            if spec.is_stereo {
                f |= crate::proj::STMC_FLAG_STEREO;
            }
            crate::proj::capsules::build_proj_stmc_capsule(
                0,
                f,
                spec.pad_gain_f32,
                spec.fractional_lag_samples_f32,
                &spec.band_psd_summary_88,
                spec.n_comb_bins,
                &spec.windows,
                &spec.spectral_summary_4096,
                &spec.wet_coreset_fp32,
            )
            .unwrap()
        };
        let stmc_view = verify_proj_stmc_capsule(&stmc_bytes).unwrap();
        let (wl, _) = decode_proj_stmc_f32(&stmc_view);
        assert_eq!(wl.len(), 120_000);
    }

    #[test]
    fn test_fullband_96k_ctrl_end_4cm_and_short_r2_r_guard() {
        let seed = [0x62_u8; 256];
        let exc = build_schoukens_pintelon_coreset_48k(
            &seed,
            TopologyProbeMode::FourCableMethod4Cm,
            96_000,
            &BTreeSet::new(),
        );
        let mut target = PassTarget;
        let win = vec![StmcTrajectoryWindow {
            control_id: 0,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 0.25,
            window_weight: 1.0,
            emr_velocity: 0.5,
        }];
        let align = SharedStemAlignment {
            lag_samples: 0.0,
            clock_drift_ppm: 12.0,
            invert_l: false,
            invert_r: false,
        };
        let (inpc, spec) = render_and_seal_fullband_stem(
            &mut target,
            FullbandStemRequest {
                stem_id: "sweep_4cm_96k",
                sample_rate_hz: 96_000,
                topology_mode: TopologyProbeMode::FourCableMethod4Cm,
                seed_256: &seed,
                carrier_bins: &[23, 57],
                valid_frames: 6,
                full_dry_l_48k: &exc.dry_l,
                full_dry_r_48k: exc.dry_r.as_deref(),
                aux_in_3_48k: exc.aux_in_3.as_deref(),
                ctrl_start: &[0.25],
                ctrl_end: Some(&[0.75]),
                is_star_center: false,
                windows: win.clone(),
                controls_meta: BTreeMap::new(),
                user_metadata: BTreeMap::new(),
                shared_align: align,
                optional_spcm: None,
                pre_extracted_physics: None,
                pre_rendered_aligned_wet: None,
            },
        )
        .unwrap();
        assert_eq!(verify_proj_inpc_capsule(&inpc).unwrap().num_frames, 96_000);
        assert_eq!(spec.stem_id, "sweep_4cm_96k");

        // Exercise `compute_star_schoukens_physics` short `aligned_wet_r2_r` tail bounds guard
        // plus `pre_extracted_physics`, `pre_rendered_aligned_wet`, and `optional_spcm` paths.
        let plan = crate::stimulus::pipeline::schoukens::build_odd_random_multisine_plan(
            &seed,
            &BTreeSet::new(),
        );
        let r1 = &exc.dry_l[..16_384];
        let r2 = &exc.dry_l[..12_000];
        let pre_phys = compute_star_schoukens_physics(StarPhysicsInputs {
            plan: &plan,
            r1_dry_period_l: r1,
            r1_settled_periods_l: &[r1],
            r1_settled_periods_r: None,
            aligned_wet_r2_l: r2,
            aligned_wet_r2_r: Some(&r2[..100]),
            r2_gap_ranges: &[(5_000, 10_000)],
            coreset_dry_l: &exc.dry_l[..13_824],
            coreset_dry_r: None,
            coreset_wet_l: &exc.dry_l[..13_824],
            coreset_wet_r: None,
            probe_comb_dry_l: &exc.dry_l[..4_608],
            probe_comb_wet_l: &exc.dry_l[..4_608],
            probe_comb_wet_r: Some(&exc.dry_l[..4_608]),
            probe_carrier_bins: &[23, 57],
            coreset_regime_offsets:
                crate::stimulus::pipeline::schoukens::SCHOUKENS_CORESET_20S_OFFSETS,
            calb_noise_floor_dbfs: -100.0,
        });
        let (_, spec_short_r) = render_and_seal_fullband_stem(
            &mut target,
            FullbandStemRequest {
                stem_id: "pre_short_r",
                sample_rate_hz: 48_000,
                topology_mode: TopologyProbeMode::MonoToStereo1To2,
                seed_256: &seed,
                carrier_bins: &[23, 57],
                valid_frames: 6,
                full_dry_l_48k: &exc.dry_l,
                full_dry_r_48k: None,
                aux_in_3_48k: None,
                ctrl_start: &[],
                ctrl_end: None,
                is_star_center: true,
                windows: win,
                controls_meta: BTreeMap::new(),
                user_metadata: BTreeMap::new(),
                shared_align: SharedStemAlignment {
                    lag_samples: 1.5,
                    clock_drift_ppm: 0.0,
                    invert_l: true,
                    invert_r: false,
                },
                optional_spcm: Some(vec![0_u8; 64]),
                pre_extracted_physics: Some(pre_phys),
                pre_rendered_aligned_wet: Some((exc.dry_l.clone(), Some(exc.dry_l.clone()))),
            },
        )
        .unwrap();
        assert!(spec_short_r.extracted_physics.is_some());
        assert_eq!(
            spec_short_r.user_metadata.get("wet_polarity_inverted"),
            Some(&serde_json::json!(true))
        );
    }
}
