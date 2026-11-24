#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! `StarCenter` Two-Tier `Schoukens–Pintelon` Capture, Causal Probe Pre-Ringing Guard,
//! Two-Pass `Law 0.A`/`PPM` Alignment & Physics Extraction (`stimulus::pipeline::fullband::star_schoukens`).

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::DnamError;

use crate::CANONICAL_TRAINER_SAMPLE_RATE_HZ;
use crate::align::{estimate_hybrid_multiregime_alignment, realign_fractional_kaiser64};
use crate::metrics::TargetRepeatabilityProfile;
use crate::proj::directory::StmcTrajectoryWindow;
use crate::proj::schema::ExtractedPhysics;
use crate::proj::vault::VaultCaptureSpec;
use crate::stimulus::laws::TopologyProbeMode;
use crate::stimulus::pipeline::causal_probe::estimate_v3_causal_deadtime_and_polarity;
use crate::stimulus::pipeline::schoukens::{
    OddRandomMultisinePlan, SCHOUKENS_R1_PERIOD_SAMPLES, SCHOUKENS_R1_SPCM_PERIODS,
    build_schoukens_pintelon_coreset_48k, build_schoukens_pintelon_spcm_48k,
    condense_spcm_to_coreset_for_topology, coreset_offsets_for_frames, encode_spcm_chunk,
};
use crate::stimulus::pipeline::stems::spectral_tilt_from_stmc_4096;
use crate::stimulus::sealing::SharedStemAlignment;
use crate::targets::DistillationTarget;

use super::causal_guard::{has_causal_probe_activity, refine_causal_probe_tau_coarse};
use super::{
    FullbandStemRequest, StarPhysicsInputs, StemRenderRequest,
    build_v3_causal_probe_excitation_48k, compute_star_schoukens_physics,
    probe_target_repeatability_profiles, render_and_seal_fullband_stem, render_stem_48k,
};

fn prepend_with_r1_prefix(src: &[f32], prepend_periods: usize) -> Vec<f32> {
    let prefix_len = SCHOUKENS_R1_PERIOD_SAMPLES.min(src.len());
    let mut out = Vec::with_capacity(prepend_periods * prefix_len + src.len());
    for _ in 0..prepend_periods {
        out.extend_from_slice(&src[..prefix_len]);
    }
    out.extend_from_slice(src);
    out
}

/// Complete `StarCenter` Two-Tier `Schoukens–Pintelon` capture outputs and shared stem buffers.
#[derive(Debug)]
pub(in crate::stimulus::pipeline) struct FullbandStarCapture {
    pub inpc_bytes: Vec<u8>,
    pub star_spec: VaultCaptureSpec,
    pub dry_l_48k: Vec<f32>,
    pub dry_r_48k: Option<Vec<f32>>,
    pub aux_in_3_48k: Option<Vec<f32>>,
    pub shared_align_stereo: SharedStemAlignment,
    pub rep_profiles: Option<[TargetRepeatabilityProfile; 2]>,
    pub star_tilt: f64,
}

type CoresetExtractor<'a> =
    &'a dyn Fn(&[f32], Option<&[f32]>) -> Result<(Vec<f32>, Option<Vec<f32>>), DnamError>;

struct FullbandAlignContext<'a> {
    dry_full_l: &'a [f32],
    dry_full_r: Option<&'a [f32]>,
    wet_full_l_raw: &'a [f32],
    wet_full_r_raw: Option<&'a [f32]>,
    plan: &'a OddRandomMultisinePlan,
    r1_period_l: &'a [f32],
    total_r1_periods: usize,
    r2_gap_ranges_full: &'a [(usize, usize)],
    tau_coarse: f64,
    inv_r_probe: bool,
    max_lag: usize,
    calb_noise_floor_dbfs: f32,
    coreset_offsets: [u32; 5],
    core_dry_l: &'a [f32],
    core_dry_r: Option<&'a [f32]>,
    probe_comb_dry_l: &'a [f32],
    probe_comb_wet_l: &'a [f32],
    probe_comb_wet_r: Option<&'a [f32]>,
    probe_carrier_bins: &'a [usize],
    extract_coreset: CoresetExtractor<'a>,
}

struct FullbandAlignOutput {
    aligned_full_l: Vec<f32>,
    aligned_full_r: Option<Vec<f32>>,
    core_wet_l: Vec<f32>,
    core_wet_r: Option<Vec<f32>>,
    shared_align: SharedStemAlignment,
    extracted_phys: ExtractedPhysics,
    phys_meta: BTreeMap<String, serde_json::Value>,
}

struct PassAlignResult {
    wl_al: Vec<f32>,
    wr_al: Option<Vec<f32>>,
    core_wet_l: Vec<f32>,
    core_wet_r: Option<Vec<f32>>,
    ep: ExtractedPhysics,
    meta: BTreeMap<String, serde_json::Value>,
}

fn estimate_right_channel_polarity(
    ctx: &FullbandAlignContext<'_>,
    law0a_ltv_flag: bool,
    inv_l_fallback: bool,
) -> bool {
    match ctx.wet_full_r_raw {
        Some(wr) => {
            let ref_r = ctx.dry_full_r.unwrap_or(ctx.dry_full_l);
            let n_r = ref_r
                .len()
                .min(wr.len())
                .min(CANONICAL_TRAINER_SAMPLE_RATE_HZ as usize);
            if n_r >= 256 {
                match estimate_hybrid_multiregime_alignment(
                    &ref_r[..n_r],
                    &wr[..n_r],
                    CANONICAL_TRAINER_SAMPLE_RATE_HZ,
                    ctx.max_lag,
                    Some(ctx.tau_coarse),
                    law0a_ltv_flag,
                ) {
                    Ok(ar) if ar.linear_coherence >= 0.12 => ar.polarity_inverted,
                    _ => ctx.inv_r_probe,
                }
            } else {
                ctx.inv_r_probe
            }
        }
        None => inv_l_fallback,
    }
}

fn align_and_analyze_fullband_render(
    ctx: FullbandAlignContext<'_>,
) -> Result<FullbandAlignOutput, DnamError> {
    let n_l = ctx.dry_full_l.len().min(ctx.wet_full_l_raw.len());
    let mut align_l = estimate_hybrid_multiregime_alignment(
        &ctx.dry_full_l[..n_l],
        &ctx.wet_full_l_raw[..n_l],
        CANONICAL_TRAINER_SAMPLE_RATE_HZ,
        ctx.max_lag,
        Some(ctx.tau_coarse),
        false,
    )
    .map_err(|e| DnamError::SchemaValidationError(format!("hybrid alignment failed: {e}")))?;

    let mut invert_r = estimate_right_channel_polarity(&ctx, false, align_l.polarity_inverted);

    let run_pass =
        |lag: f64, ppm: f64, inv_l: bool, inv_r: bool| -> Result<PassAlignResult, DnamError> {
            let wl_al = realign_fractional_kaiser64(&ctx.wet_full_l_raw[..n_l], lag, ppm, inv_l);
            let wr_al = ctx.wet_full_r_raw.map(|wr| {
                let n_r = wr
                    .len()
                    .min(ctx.dry_full_r.map_or(ctx.dry_full_l.len(), |r| r.len()));
                realign_fractional_kaiser64(&wr[..n_r], lag, ppm, inv_r)
            });
            let r1_l_refs: Vec<&[f32]> = (1..ctx.total_r1_periods)
                .filter_map(|p| {
                    let s = p * SCHOUKENS_R1_PERIOD_SAMPLES;
                    let e = s + SCHOUKENS_R1_PERIOD_SAMPLES;
                    (e <= wl_al.len()).then(|| &wl_al[s..e])
                })
                .collect();
            let r1_r_vec: Option<Vec<&[f32]>> = wr_al.as_deref().map(|wr| {
                (1..ctx.total_r1_periods)
                    .filter_map(|p| {
                        let s = p * SCHOUKENS_R1_PERIOD_SAMPLES;
                        let e = s + SCHOUKENS_R1_PERIOD_SAMPLES;
                        (e <= wr.len()).then(|| &wr[s..e])
                    })
                    .collect()
            });
            let r1_r_refs = r1_r_vec.as_deref().filter(|v| !v.is_empty());
            let (core_wet_l, core_wet_r) = (ctx.extract_coreset)(&wl_al, wr_al.as_deref())?;
            let (ep, meta) = compute_star_schoukens_physics(StarPhysicsInputs {
                plan: ctx.plan,
                r1_dry_period_l: ctx.r1_period_l,
                r1_settled_periods_l: &r1_l_refs,
                r1_settled_periods_r: r1_r_refs,
                aligned_wet_r2_l: &wl_al,
                aligned_wet_r2_r: wr_al.as_deref(),
                r2_gap_ranges: ctx.r2_gap_ranges_full,
                coreset_dry_l: ctx.core_dry_l,
                coreset_dry_r: ctx.core_dry_r,
                coreset_wet_l: &core_wet_l,
                coreset_wet_r: core_wet_r.as_deref(),
                probe_comb_dry_l: ctx.probe_comb_dry_l,
                probe_comb_wet_l: ctx.probe_comb_wet_l,
                probe_comb_wet_r: ctx.probe_comb_wet_r,
                probe_carrier_bins: ctx.probe_carrier_bins,
                coreset_regime_offsets: ctx.coreset_offsets,
                calb_noise_floor_dbfs: ctx.calb_noise_floor_dbfs,
            });
            Ok(PassAlignResult {
                wl_al,
                wr_al,
                core_wet_l,
                core_wet_r,
                ep,
                meta,
            })
        };

    let mut pass = run_pass(
        align_l.total_lag_samples,
        align_l.clock_drift_ppm,
        align_l.polarity_inverted,
        invert_r,
    )?;

    if pass.ep.law0a_ltv_flag && align_l.clock_drift_ppm.abs() > 0.0 {
        align_l = estimate_hybrid_multiregime_alignment(
            &ctx.dry_full_l[..n_l],
            &ctx.wet_full_l_raw[..n_l],
            CANONICAL_TRAINER_SAMPLE_RATE_HZ,
            ctx.max_lag,
            Some(ctx.tau_coarse),
            true,
        )
        .map_err(|e| DnamError::SchemaValidationError(format!("hybrid alignment failed: {e}")))?;
        invert_r = estimate_right_channel_polarity(&ctx, true, align_l.polarity_inverted);
        pass = run_pass(
            align_l.total_lag_samples,
            align_l.clock_drift_ppm,
            align_l.polarity_inverted,
            invert_r,
        )?;
    }

    let shared_align = SharedStemAlignment {
        lag_samples: align_l.total_lag_samples,
        clock_drift_ppm: align_l.clock_drift_ppm,
        invert_l: align_l.polarity_inverted,
        invert_r,
    };
    Ok(FullbandAlignOutput {
        aligned_full_l: pass.wl_al,
        aligned_full_r: pass.wr_al,
        core_wet_l: pass.core_wet_l,
        core_wet_r: pass.core_wet_r,
        shared_align,
        extracted_phys: pass.ep,
        phys_meta: pass.meta,
    })
}

/// Request parameters for [`capture_fullband_star_center_schoukens`].
#[derive(Debug)]
pub(in crate::stimulus::pipeline) struct StarCenterCaptureRequest<'a> {
    pub sample_rate_hz: u32,
    pub topology_mode: TopologyProbeMode,
    pub seed_256: &'a [u8; 256],
    pub carrier_bins: &'a [usize],
    pub valid_frames: usize,
    pub base_ctrls: &'a [f32],
    pub star_meta: BTreeMap<String, serde_json::Value>,
    pub probe_comb_dry_l: &'a [f32],
    pub probe_comb_wet_l: &'a [f32],
    pub probe_comb_wet_r: Option<&'a [f32]>,
    pub use_spcm_tier1: bool,
    pub coreset_frames_48k: usize,
    pub calb_noise_floor_dbfs: f32,
}

/// Captures `StarCenter` via the `v3` causal deadtime/polarity probe + Two-Tier
/// `Schoukens–Pintelon` excitation (`9.0 s` `SPCM` optional or default `2.0–2.5 s` coreset),
/// solving hybrid sub-sample/PPM alignment, physics descriptors, conditional repeatability profiles,
/// and sealed `INPC`/`VaultCaptureSpec`.
pub(in crate::stimulus::pipeline) fn capture_fullband_star_center_schoukens(
    target: &mut dyn DistillationTarget,
    req: StarCenterCaptureRequest<'_>,
) -> Result<FullbandStarCapture, DnamError> {
    let StarCenterCaptureRequest {
        sample_rate_hz,
        topology_mode,
        seed_256,
        carrier_bins,
        valid_frames,
        base_ctrls,
        star_meta,
        probe_comb_dry_l,
        probe_comb_wet_l,
        probe_comb_wet_r,
        use_spcm_tier1,
        coreset_frames_48k,
        calb_noise_floor_dbfs,
    } = req;
    let causal_probe = build_v3_causal_probe_excitation_48k();
    let probe_dry_r = matches!(topology_mode, TopologyProbeMode::TrueStereo2To2)
        .then_some(causal_probe.as_slice());
    let (pw_l, pw_r) = render_stem_48k(
        target,
        StemRenderRequest {
            stem_id: "causal_probe_v3",
            sample_rate_hz,
            topology_mode,
            dry_l_48k: &causal_probe,
            dry_r_48k: probe_dry_r,
            aux_in_3_48k: None,
            ctrl_start: base_ctrls,
            ctrl_end: None,
        },
    )?;
    let (tau_l, has_l) = if has_causal_probe_activity(&causal_probe, &pw_l) {
        let (lag_l, _) = estimate_v3_causal_deadtime_and_polarity(&causal_probe, &pw_l);
        (
            refine_causal_probe_tau_coarse(&causal_probe, &pw_l, lag_l),
            true,
        )
    } else {
        (0.0, false)
    };
    let (tau_coarse, inv_r_probe) = match pw_r.as_deref() {
        Some(r) if has_causal_probe_activity(&causal_probe, r) => {
            let (lag_r, pol_r) = estimate_v3_causal_deadtime_and_polarity(&causal_probe, r);
            let tau_r = refine_causal_probe_tau_coarse(&causal_probe, r, lag_r);
            let combined_tau = if has_l { tau_l.min(tau_r) } else { tau_r };
            (combined_tau, pol_r)
        }
        _ => (tau_l, false),
    };
    let max_lag = (tau_coarse.ceil().max(0.0) as usize + 64).clamp(128, 2_048);
    let coreset_len = (coreset_frames_48k.clamp(
        crate::CORESET_FP32_MIN_FRAMES_PER_STEM,
        crate::CORESET_FP32_MAX_FRAMES_PER_STEM,
    )) & !15;
    let empty_mask = BTreeSet::new();

    let (
        core_dry_l,
        core_dry_r,
        core_aux_3,
        core_wet_l,
        core_wet_r,
        shared_align_stereo,
        extracted_phys,
        phys_meta,
        optional_spcm,
    ) = if use_spcm_tier1 {
        let exc_spcm = build_schoukens_pintelon_spcm_48k(seed_256, topology_mode, &empty_mask);
        let (wet_spcm_l, wet_spcm_r) = render_stem_48k(
            target,
            StemRenderRequest {
                stem_id: "star_center_spcm",
                sample_rate_hz,
                topology_mode,
                dry_l_48k: &exc_spcm.dry_l,
                dry_r_48k: exc_spcm.dry_r.as_deref(),
                aux_in_3_48k: exc_spcm.aux_in_3.as_deref(),
                ctrl_start: base_ctrls,
                ctrl_end: None,
            },
        )?;
        let (core_dry_l, core_dry_r, condense_offsets) = condense_spcm_to_coreset_for_topology(
            &exc_spcm.dry_l,
            exc_spcm.dry_r.as_deref(),
            coreset_len,
            topology_mode,
        )?;
        let core_aux_3 = exc_spcm
            .aux_in_3
            .as_deref()
            .map(|a| {
                condense_spcm_to_coreset_for_topology(a, None, coreset_len, topology_mode)
                    .map(|(c, _, _)| c)
            })
            .transpose()?;
        let out = align_and_analyze_fullband_render(FullbandAlignContext {
            dry_full_l: &exc_spcm.dry_l,
            dry_full_r: exc_spcm.dry_r.as_deref(),
            wet_full_l_raw: &wet_spcm_l,
            wet_full_r_raw: wet_spcm_r.as_deref(),
            plan: &exc_spcm.r1_plan,
            r1_period_l: &exc_spcm.r1_period_l,
            total_r1_periods: SCHOUKENS_R1_SPCM_PERIODS,
            r2_gap_ranges_full: &exc_spcm.r2_gap_ranges_global,
            tau_coarse,
            inv_r_probe,
            max_lag,
            calb_noise_floor_dbfs,
            coreset_offsets: condense_offsets,
            core_dry_l: &core_dry_l,
            core_dry_r: core_dry_r.as_deref(),
            probe_comb_dry_l,
            probe_comb_wet_l,
            probe_comb_wet_r,
            probe_carrier_bins: carrier_bins,
            extract_coreset: &|wl, wr| {
                let (cl, cr, _) =
                    condense_spcm_to_coreset_for_topology(wl, wr, coreset_len, topology_mode)?;
                Ok((cl, cr))
            },
        })?;
        // Persist aligned raw fp32 stem audio into SPCM (`ref[container.proj.lifecycle-states]`).
        let spcm_raw_n = exc_spcm.dry_l.len().min(out.aligned_full_l.len());
        let optional_spcm = Some(encode_spcm_chunk(
            &out.aligned_full_l[..spcm_raw_n],
            out.aligned_full_r
                .as_deref()
                .map(|r| &r[..r.len().min(spcm_raw_n)]),
        ));
        (
            core_dry_l,
            core_dry_r,
            core_aux_3,
            out.core_wet_l,
            out.core_wet_r,
            out.shared_align,
            out.extracted_phys,
            out.phys_meta,
            optional_spcm,
        )
    } else {
        let exc_core =
            build_schoukens_pintelon_coreset_48k(seed_256, topology_mode, coreset_len, &empty_mask);
        let core_r1_periods = (exc_core.regime_offsets[1] as usize) / SCHOUKENS_R1_PERIOD_SAMPLES;
        let prepend_periods = 3_usize.saturating_sub(core_r1_periods).max(1);
        let prepend_samples = prepend_periods * SCHOUKENS_R1_PERIOD_SAMPLES;
        let full_dry_l = prepend_with_r1_prefix(&exc_core.dry_l, prepend_periods);
        let full_dry_r = exc_core
            .dry_r
            .as_deref()
            .map(|r| prepend_with_r1_prefix(r, prepend_periods));
        let full_aux_3 = exc_core
            .aux_in_3
            .as_deref()
            .map(|a| prepend_with_r1_prefix(a, prepend_periods));
        let (wet_full_l, wet_full_r) = render_stem_48k(
            target,
            StemRenderRequest {
                stem_id: "star_center_full",
                sample_rate_hz,
                topology_mode,
                dry_l_48k: &full_dry_l,
                dry_r_48k: full_dry_r.as_deref(),
                aux_in_3_48k: full_aux_3.as_deref(),
                ctrl_start: base_ctrls,
                ctrl_end: None,
            },
        )?;
        let shifted_gaps: Vec<(usize, usize)> = exc_core
            .r2_gap_ranges_global
            .iter()
            .map(|&(gs, ge)| (gs + prepend_samples, ge + prepend_samples))
            .collect();
        let coreset_offsets = coreset_offsets_for_frames(coreset_len);
        let out = align_and_analyze_fullband_render(FullbandAlignContext {
            dry_full_l: &full_dry_l,
            dry_full_r: full_dry_r.as_deref(),
            wet_full_l_raw: &wet_full_l,
            wet_full_r_raw: wet_full_r.as_deref(),
            plan: &exc_core.r1_plan,
            r1_period_l: &exc_core.r1_period_l,
            total_r1_periods: prepend_periods + core_r1_periods,
            r2_gap_ranges_full: &shifted_gaps,
            tau_coarse,
            inv_r_probe,
            max_lag,
            calb_noise_floor_dbfs,
            coreset_offsets,
            core_dry_l: &exc_core.dry_l,
            core_dry_r: exc_core.dry_r.as_deref(),
            probe_comb_dry_l,
            probe_comb_wet_l,
            probe_comb_wet_r,
            probe_carrier_bins: carrier_bins,
            extract_coreset: &|wl, wr| {
                let end = (prepend_samples + coreset_len).min(wl.len());
                let slice_l = if end > prepend_samples {
                    wl[prepend_samples..end].to_vec()
                } else {
                    vec![0.0_f32; coreset_len]
                };
                let slice_r = wr.map(|r| {
                    let r_end = (prepend_samples + coreset_len).min(r.len());
                    if r_end > prepend_samples {
                        r[prepend_samples..r_end].to_vec()
                    } else {
                        vec![0.0_f32; coreset_len]
                    }
                });
                Ok((slice_l, slice_r))
            },
        })?;
        (
            exc_core.dry_l,
            exc_core.dry_r,
            exc_core.aux_in_3,
            out.core_wet_l,
            out.core_wet_r,
            out.shared_align,
            out.extracted_phys,
            out.phys_meta,
            None,
        )
    };

    let rep_profiles = if extracted_phys.law0a_ltv_flag || extracted_phys.edc_t60_s > 0.15 {
        Some(probe_target_repeatability_profiles(
            target,
            sample_rate_hz,
            topology_mode,
            base_ctrls,
            shared_align_stereo,
        )?)
    } else {
        None
    };

    let (inpc_bytes, star_spec) = render_and_seal_fullband_stem(
        target,
        FullbandStemRequest {
            stem_id: "star_center",
            sample_rate_hz,
            topology_mode,
            seed_256,
            carrier_bins,
            valid_frames,
            full_dry_l_48k: &core_dry_l,
            full_dry_r_48k: core_dry_r.as_deref(),
            aux_in_3_48k: core_aux_3.as_deref(),
            ctrl_start: base_ctrls,
            ctrl_end: None,
            is_star_center: true,
            windows: vec![StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: base_ctrls.first().copied().unwrap_or(0.0),
                window_weight: 1.0,
                emr_velocity: 0.0,
            }],
            controls_meta: star_meta,
            user_metadata: BTreeMap::new(),
            shared_align: shared_align_stereo,
            optional_spcm,
            pre_extracted_physics: Some((extracted_phys, phys_meta)),
            pre_rendered_aligned_wet: Some((core_wet_l, core_wet_r)),
        },
    )?;
    let star_tilt = spectral_tilt_from_stmc_4096(&star_spec.spectral_summary_4096);
    Ok(FullbandStarCapture {
        inpc_bytes,
        star_spec,
        dry_l_48k: core_dry_l,
        dry_r_48k: core_dry_r,
        aux_in_3_48k: core_aux_3,
        shared_align_stereo,
        rep_profiles,
        star_tilt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stimulus::pipeline::schoukens::{
        SCHOUKENS_CORESET_20S_OFFSETS, SCHOUKENS_CORESET_25S_OFFSETS,
        SCHOUKENS_SPCM_CANONICAL_FRAMES, SpcmHeader64BWire,
    };
    use crate::targets::{StemResponse, StemStimulus};

    #[derive(Default)]
    struct SynthStarTarget {
        lag_samples: f64,
        clock_drift_ppm: f64,
        invert_l: bool,
        invert_r: bool,
        stereo_out: bool,
        silent_r: bool,
        idle_noise_r: f64,
        ltv_mod_depth: f64,
        tail_t60_s: Option<f64>,
    }

    impl DistillationTarget for SynthStarTarget {
        fn reset_state(&mut self) {}

        fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
            let sr = f64::from(stim.sample_rate_hz.max(1));
            let rv_pole = self
                .tail_t60_s
                .map_or(0.0, |t60| (-6.907_755_278_982_137 / (t60 * sr)).exp());
            let mut rv_state = 0.0_f64;
            let mut proc_l = Vec::with_capacity(stim.in_1.len());
            for (i, &x) in stim.in_1.iter().enumerate() {
                let ltv = if self.ltv_mod_depth > 0.0 {
                    1.0 + self.ltv_mod_depth * (std::f64::consts::TAU * 2.1 * (i as f64) / sr).sin()
                } else {
                    1.0
                };
                let base = 0.72 * ltv * f64::from(x) + 0.02 * f64::from(x).powi(3);
                let y = if rv_pole > 0.0 {
                    rv_state = rv_pole * rv_state + 0.03 * base;
                    0.70 * base + 0.30 * rv_state
                } else {
                    base
                };
                proc_l.push(y as f32);
            }
            let out_1 = realign_fractional_kaiser64(
                &proc_l,
                -self.lag_samples,
                -self.clock_drift_ppm,
                self.invert_l,
            );
            let out_2 = if self.silent_r {
                if self.idle_noise_r > 0.0 {
                    let mut rng = 0x51A7_E80D_BEEF_1234_u64;
                    let noise: Vec<f32> = (0..stim.in_1.len())
                        .map(|_| {
                            rng = rng
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            let u = ((rng >> 32) as f64) / f64::from(u32::MAX) - 0.5;
                            (2.0 * self.idle_noise_r * u) as f32
                        })
                        .collect();
                    Some(noise)
                } else {
                    Some(vec![0.0_f32; stim.in_1.len()])
                }
            } else if self.stereo_out {
                let src_r = stim.in_2.as_deref().unwrap_or(&stim.in_1);
                let proc_r: Vec<f32> = src_r
                    .iter()
                    .enumerate()
                    .map(|(i, &xr)| {
                        (0.70 * (0.85 * f64::from(xr) + 0.12 * f64::from(stim.in_1[i]))) as f32
                    })
                    .collect();
                Some(realign_fractional_kaiser64(
                    &proc_r,
                    -self.lag_samples,
                    -self.clock_drift_ppm,
                    self.invert_r,
                ))
            } else {
                None
            };
            Ok(StemResponse {
                stem_id: stim.stem_id.clone(),
                sample_rate_hz: stim.sample_rate_hz,
                out_1,
                out_2,
                out_3: None,
                out_4: None,
                internal_nodes: BTreeMap::new(),
            })
        }
    }

    fn comb_probe_13k() -> Vec<f32> {
        (0..13_824)
            .map(|i| 0.25 * (std::f32::consts::TAU * 220.0 * (i as f32) / 48_000.0).sin())
            .collect()
    }

    #[test]
    fn test_star_schoukens_crossfeed_spcm_ppm_ltv_t60_and_aux_topologies() {
        let seed = [0x4D_u8; 256];
        let comb = comb_probe_13k();
        let bins = [23_usize, 57];

        fn run_star_cap(
            target: &mut SynthStarTarget,
            fix: (&[u8; 256], &[usize], &[f32]),
            mode: TopologyProbeMode,
            comb_r: Option<&[f32]>,
            use_spcm_tier1: bool,
            coreset_frames_48k: usize,
        ) -> FullbandStarCapture {
            let (seed_256, carrier_bins, comb) = fix;
            capture_fullband_star_center_schoukens(
                target,
                StarCenterCaptureRequest {
                    sample_rate_hz: 48_000,
                    topology_mode: mode,
                    seed_256,
                    carrier_bins,
                    valid_frames: 6,
                    base_ctrls: &[],
                    star_meta: BTreeMap::new(),
                    probe_comb_dry_l: comb,
                    probe_comb_wet_l: comb,
                    probe_comb_wet_r: comb_r,
                    use_spcm_tier1,
                    coreset_frames_48k,
                    calb_noise_floor_dbfs: -100.0,
                },
            )
            .unwrap()
        }

        // 1. MonoToStereo1To2 with L/R cross-feed & independent R polarity flip at 96k (2.0 s).
        let mut cross = SynthStarTarget {
            lag_samples: 12.25,
            invert_l: false,
            invert_r: true,
            stereo_out: true,
            ..Default::default()
        };
        let cap_cross = run_star_cap(
            &mut cross,
            (&seed, &bins, &comb),
            TopologyProbeMode::MonoToStereo1To2,
            Some(&comb),
            false,
            96_000,
        );
        assert_eq!(cap_cross.dry_l_48k.len(), 96_000);
        assert!(!cap_cross.shared_align_stereo.invert_l);
        assert!(cap_cross.shared_align_stereo.invert_r);
        let ep_cross = cap_cross.star_spec.extracted_physics.unwrap();
        assert_eq!(
            ep_cross.coreset_regime_offsets,
            SCHOUKENS_CORESET_20S_OFFSETS
        );

        // 2. TrueStereo2To2 + Tier-1 SPCM + 37.38 lag + +45 ppm + L invert at 120k (2.5 s).
        let mut spcm_t = SynthStarTarget {
            lag_samples: 37.38,
            clock_drift_ppm: 45.0,
            invert_l: true,
            stereo_out: true,
            ..Default::default()
        };
        let cap_spcm = run_star_cap(
            &mut spcm_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::TrueStereo2To2,
            Some(&comb),
            true,
            120_000,
        );
        let raw_spcm = cap_spcm.star_spec.optional_spcm.as_ref().unwrap();
        assert_eq!(raw_spcm.len(), 64 + SCHOUKENS_SPCM_CANONICAL_FRAMES * 8);
        assert_eq!(
            &raw_spcm[0..4],
            &SpcmHeader64BWire {
                magic: *b"SPCM",
                reserved: [0; 60]
            }
            .magic
        );
        assert!(
            (cap_spcm.shared_align_stereo.lag_samples - 37.38).abs() < 0.05,
            "unexpected spcm alignment: {:?}",
            cap_spcm.shared_align_stereo
        );
        assert!(
            (cap_spcm.shared_align_stereo.clock_drift_ppm - 45.0).abs() < 4.0,
            "unexpected spcm ppm: {:?}",
            cap_spcm.shared_align_stereo
        );
        assert!(cap_spcm.shared_align_stereo.invert_l);
        let ep_spcm = cap_spcm.star_spec.extracted_physics.unwrap();
        assert!(!ep_spcm.law0a_ltv_flag);
        assert_eq!(
            ep_spcm.coreset_regime_offsets,
            SCHOUKENS_CORESET_25S_OFFSETS
        );

        // 3. LTV modulation + T60 tail -> forces ppm=0, Law 0.A flag, edc_t60_s > 0.15, and rep_profiles.
        let mut ltv_t = SynthStarTarget {
            lag_samples: 18.25,
            ltv_mod_depth: 0.30,
            tail_t60_s: Some(0.8),
            ..Default::default()
        };
        let cap_ltv = run_star_cap(
            &mut ltv_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::Mono1To1,
            None,
            false,
            120_000,
        );
        let ep_ltv = cap_ltv.star_spec.extracted_physics.unwrap();
        assert!(ep_ltv.law0a_ltv_flag);
        assert_eq!(cap_ltv.shared_align_stereo.clock_drift_ppm, 0.0);
        assert!(ep_ltv.edc_t60_s > 0.15);
        assert!(cap_ltv.rep_profiles.is_some());

        // 4. 4CM and DualIn 2->1 populate aux_in_3_48k and dry_r_48k cleanly.
        let mut aux_t = SynthStarTarget::default();
        let cap_4cm = run_star_cap(
            &mut aux_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::FourCableMethod4Cm,
            None,
            false,
            96_000,
        );
        assert_eq!(cap_4cm.aux_in_3_48k.as_ref().map(|a| a.len()), Some(96_000));
        let cap_2to1 = run_star_cap(
            &mut aux_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::DualInToMono2To1,
            None,
            false,
            96_000,
        );
        assert_eq!(
            cap_2to1.aux_in_3_48k.as_ref().map(|a| a.len()),
            Some(96_000)
        );
        assert_eq!(cap_2to1.dry_r_48k.as_ref().map(|r| r.len()), Some(96_000));

        // 5. TrueStereo2To2 with silent R triggers causal R-activity gate cleanly.
        let mut sil_r_t = SynthStarTarget {
            silent_r: true,
            invert_r: true,
            ..Default::default()
        };
        let cap_sil_r = run_star_cap(
            &mut sil_r_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::TrueStereo2To2,
            None,
            false,
            96_000,
        );
        assert!(!cap_sil_r.shared_align_stereo.invert_r);

        // 6. LTV + non-zero clock_drift_ppm + invert_r triggers Pass-2 PPM rollback + R-polarity re-eval.
        let mut ltv_ppm_t = SynthStarTarget {
            lag_samples: 12.25,
            clock_drift_ppm: 45.0,
            invert_r: true,
            stereo_out: true,
            ltv_mod_depth: 0.30,
            ..Default::default()
        };
        let cap_ltv_ppm = run_star_cap(
            &mut ltv_ppm_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::TrueStereo2To2,
            None,
            false,
            96_000,
        );
        assert!(
            cap_ltv_ppm
                .star_spec
                .extracted_physics
                .unwrap()
                .law0a_ltv_flag
        );
        assert_eq!(cap_ltv_ppm.shared_align_stereo.clock_drift_ppm, 0.0);
        assert!(cap_ltv_ppm.shared_align_stereo.invert_r);

        // 7. TrueStereo2To2 with -80 dBFS noisy idle R + lag=160.25 must ignore noisy R so tau_coarse does not collapse.
        let mut noisy_r_t = SynthStarTarget {
            lag_samples: 160.25,
            silent_r: true,
            idle_noise_r: 1.0e-4,
            invert_r: true,
            ..Default::default()
        };
        let cap_noisy_r = run_star_cap(
            &mut noisy_r_t,
            (&seed, &bins, &comb),
            TopologyProbeMode::TrueStereo2To2,
            None,
            false,
            96_000,
        );
        assert!(!cap_noisy_r.shared_align_stereo.invert_r);
        assert!(
            (cap_noisy_r.shared_align_stereo.lag_samples - 160.25).abs() < 0.05,
            "lag={}",
            cap_noisy_r.shared_align_stereo.lag_samples
        );
    }
}
