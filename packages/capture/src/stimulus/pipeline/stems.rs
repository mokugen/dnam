#![forbid(unsafe_code)]
//! Endpoint Probing, Closed-Loop Stem Rendering, and Freeze-Transition Helpers
//! (`stimulus::pipeline::stems`).

use std::collections::BTreeMap;

use dnam_engine::schema::DnamError;
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::F32;

use crate::proj::vault::VaultCaptureSpec;
use crate::resampler::{resample_arbitrary_rate_to_48k, resampled_frame_len};
use crate::targets::{DistillationTarget, StemResponse, StemStimulus};

use super::super::batch::resample_48k_channels_to_sr;
use super::super::fft::ADAPTIVE_FRAME_SAMPLES;
use super::super::laws::TopologyProbeMode;
use super::super::physics_probe;
use super::super::safeguards::CaptureWarning;
use super::super::{AdaptiveStimulusController, StimulusConfig};

pub(super) type StarProbe<'a> = (&'a [f32], &'a [f32], Option<&'a [f32]>, &'a [usize]);
pub(super) type ClosedLoopStemOut = (
    Vec<u8>,
    VaultCaptureSpec,
    Vec<CaptureWarning>,
    f64,
    f64,
    bool,
);

pub(super) fn spectral_tilt_from_stmc_4096(spec_4096: &[u8; 4_096]) -> f64 {
    let Ok(bins) = <[F32]>::ref_from_bytes(spec_4096) else {
        return 0.0;
    };
    let mut weighted_e = 0.0_f64;
    let mut total_e = 0.0_f64;
    for (i, bin) in bins.iter().enumerate().take(512).skip(1) {
        let m = f64::from(bin.get());
        let p = m * m;
        weighted_e += ((i as f64) / 32.0) * p;
        total_e += p;
    }
    weighted_e / (total_e + 1.0e-18)
}

/// Renders the deterministic endpoint probe (`1` `96 ms` comb frame + `9` `96 ms` silent tail
/// frames = `864 ms` tail) and normalizes the wet response to `48 kHz` (`r[train.capture.invariants]`).
pub(super) fn render_endpoint_probe_48k(
    target: &mut dyn DistillationTarget,
    probe_dry_l_48k: &[f32],
    sample_rate_hz: u32,
    topology_mode: TopologyProbeMode,
    control_values: &[f32],
) -> Result<(Vec<f32>, Option<Vec<f32>>), DnamError> {
    let sr = sample_rate_hz.clamp(8_000, 384_000);
    let need_stereo = matches!(
        topology_mode,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let need_aux = matches!(
        topology_mode,
        TopologyProbeMode::FourCableMethod4Cm | TopologyProbeMode::DualInToMono2To1
    );
    let dry_r_48k = need_stereo.then_some(probe_dry_l_48k);
    let aux_48k = need_aux.then_some(probe_dry_l_48k);
    let (in_1, in_2, in_3) = resample_48k_channels_to_sr(sr, probe_dry_l_48k, dry_r_48k, aux_48k);

    target.reset_state();
    let resp = target.render_stem(&StemStimulus {
        stem_id: "endpoint_probe".to_string(),
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
    let out_l = to_48k(&resp.out_1);
    let out_r = resp.out_2.as_deref().map(to_48k);
    Ok((out_l, out_r))
}

pub(super) fn run_single_closed_loop_stem(
    target: &mut dyn DistillationTarget,
    cfg: StimulusConfig,
    is_star_center: bool,
    controls_meta: BTreeMap<String, serde_json::Value>,
    shared_align: Option<(f64, bool)>,
    star_probe: Option<StarProbe<'_>>,
) -> Result<ClosedLoopStemOut, DnamError> {
    target.reset_state();
    let mut ctrl = AdaptiveStimulusController::new(cfg);
    let mut last_resp: Option<StemResponse> = None;
    while let Some(frame) = ctrl.next_frame(last_resp.as_ref()) {
        if ctrl.is_entering_coreset() {
            target.reset_state();
        }
        last_resp = Some(target.render_frame(&frame)?);
    }
    let warnings = ctrl.warnings().to_vec();
    let (inpc, mut spec, lag, pol) = ctrl.seal_inpc_and_vault_spec_with_shared_align(
        is_star_center,
        controls_meta,
        shared_align,
    )?;
    if is_star_center
        && let Some((probe_dry_l, probe_wet_l, probe_wet_r, carrier_bins)) = star_probe
    {
        let (cdl, cdr, cwl, cwr) = ctrl.coreset_buffers_48k();
        let comb_len = ADAPTIVE_FRAME_SAMPLES
            .min(probe_dry_l.len())
            .min(probe_wet_l.len());
        let comb_wet_r = probe_wet_r.map(|r| &r[..comb_len.min(r.len())]);
        let tail_wet_l = if probe_wet_l.len() > ADAPTIVE_FRAME_SAMPLES {
            &probe_wet_l[ADAPTIVE_FRAME_SAMPLES..]
        } else {
            &[]
        };
        let tail_wet_r = probe_wet_r.map(|r| {
            if r.len() > ADAPTIVE_FRAME_SAMPLES {
                &r[ADAPTIVE_FRAME_SAMPLES..]
            } else {
                &[][..]
            }
        });
        let phys = physics_probe::extract_star_physics_from_audio(
            cdl,
            cdr,
            cwl,
            cwr,
            &probe_dry_l[..comb_len],
            &probe_wet_l[..comb_len],
            comb_wet_r,
            tail_wet_l,
            tail_wet_r,
            carrier_bins,
        );
        spec.extracted_physics = Some(phys.extracted);
        spec.user_metadata.extend(phys.metadata);
    }
    let tilt = spectral_tilt_from_stmc_4096(&spec.spectral_summary_4096);
    Ok((inpc, spec, warnings, tilt, lag, pol))
}
