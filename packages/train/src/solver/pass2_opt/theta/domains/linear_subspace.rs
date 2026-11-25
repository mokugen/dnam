//! Pure-linear modal + active `Op-T` optimization subspace excluding `CoreQuad`, `InterBlockTile`,
//! `SensoryQuad`, `Crossbar`, `Op-N[Pre]`, and nonlinear `w_ro` weights for strictly LTI Pass-2
//! trajectories (`ref[train.pipeline.backbone-trajectory-opt]`, `REQ-1.9`).

use core::f64::consts::PI;

use crate::control::stability::{MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP};
use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::super::super::model::SonaModelF64;
use super::super::super::super::pass1_probe::kautz_fit::audible_kautz_theta_bounds;
use super::super::super::eval::core_adj::active_opt_quads_for_core_quads;
use super::super::{TIER_KAUTZ, TIER_QUADS};
use super::opt_subspace::active_opt_quad_spec;
use super::{ParamKind, ThetaEntry};

/// Appends strictly linear modal + active `Op-T` descriptors (omitting all `CoreQuad`,
/// `InterBlockTile`, `SensoryQuad`, `Crossbar`, `Op-N[Pre]`, and `w_ro` coordinates) and
/// returns the matching `kautz_active_mask`.
pub(in super::super) fn append_linear_modal_transport_entries(
    entries: &mut Vec<ThetaEntry>,
    model: &SonaModelF64,
    tier_idx: usize,
    include_extras: bool,
    include_direct_readouts: bool,
    max_tau: f64,
    kautz_r_max: f64,
) -> (usize, usize, [bool; 32]) {
    let ti = tier_idx.min(3);
    let q_count = TIER_QUADS[ti];
    let k_count = model.active_kautz_pairs.min(TIER_KAUTZ[ti]);
    let max_opt_quads = active_opt_quads_for_core_quads(q_count);
    let (active_opt_quads, opt_feedforward_only_mask) = active_opt_quad_spec(model, max_opt_quads);

    entries.push(ThetaEntry::f32(
        ParamKind::RoPole48k,
        ParamBoundSpec::ComplementLog {
            lo_c: 1.0 - MAX_RO_POLE_48K_F64,
            hi_c: 1.0,
        },
    ));
    let (th_min, th_max) = audible_kautz_theta_bounds(model.sample_rate_hz);
    let mut min_existing_theta = th_min;
    let mut max_existing_theta = th_max;
    let mut max_existing_r = 0.0_f64;
    for kp in model.kautz.iter().take(k_count) {
        if kp.r_low > 0.0 && kp.b_gamma > 0.0 && kp.r_low.is_finite() && kp.theta.is_finite() {
            min_existing_theta = min_existing_theta.min(kp.theta - 1.0e-4);
            max_existing_theta = max_existing_theta.max(kp.theta + 1.0e-4);
            max_existing_r = max_existing_r.max(kp.r_low);
        }
    }
    let r_hi = kautz_r_max.max(max_existing_r + 1.0e-4).clamp(0.95, 0.9995);
    let kautz_r_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - r_hi,
        hi_c: 1.0 - 0.25,
    };
    let kautz_theta_box = ParamBoundSpec::LogBox {
        lo: min_existing_theta.clamp(1.0e-4, th_min),
        hi: max_existing_theta.clamp(th_max, PI - 1.0e-4),
    };
    let bg_box = ParamBoundSpec::LogBox {
        lo: 1.0e-4,
        hi: 8.0,
    };
    for k in 0..k_count {
        entries.push(ThetaEntry::f32(ParamKind::KautzRLow(k), kautz_r_box));
        entries.push(ThetaEntry::f32(ParamKind::KautzTheta(k), kautz_theta_box));
        if include_extras {
            entries.push(ThetaEntry::f32(ParamKind::KautzBGamma(k), bg_box));
        }
    }

    let tau_max = max_tau.clamp(MIN_DELAY_CROSSBAR_SAMP + 1.0, 48_000.0);
    let delay_box = ParamBoundSpec::LogBox {
        lo: MIN_DELAY_CROSSBAR_SAMP,
        hi: tau_max,
    };
    let tap_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 2.0,
    };
    let unit_box = ParamBoundSpec::Box { lo: 0.0, hi: 1.0 };
    let angle_box = ParamBoundSpec::Box { lo: -PI, hi: PI };
    let gloop_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - f64::from(dnam_kernels::TANK_MAX_G_LIN_F32),
        hi_c: 1.0,
    };
    let rdc_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - f64::from(dnam_kernels::MAX_PARALRU_R_F32),
        hi_c: 1.0 - 0.95,
    };
    let damp_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - 0.9995,
        hi_c: 1.0,
    };
    let span_max = (tau_max - MIN_DELAY_CROSSBAR_SAMP).max(1.0);
    let span_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 32.0,
        limit: span_max,
    };
    let lfo_rate_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 1.0,
        limit: 16.0,
    };
    let lfo_depth_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 4.0,
        limit: 512.0,
    };
    let tone_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 16.0,
    };
    let ap_box = ParamBoundSpec::Box {
        lo: -0.75,
        hi: 0.75,
    };
    let sigma_box = ParamBoundSpec::LogBox {
        lo: 1.0e-3,
        hi: 0.984_375,
    };
    let freeze_box = ParamBoundSpec::Box { lo: 0.0, hi: 0.999 };

    for q in 0..active_opt_quads {
        if opt_feedforward_only_mask[q] {
            for l in 0..4 {
                entries.push(ThetaEntry::f32(ParamKind::OpTDelay(q, l), delay_box));
                entries.push(ThetaEntry::fp16(ParamKind::OpTInputGain(q, l), tap_box));
                if include_extras {
                    entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapL(q, l), tap_box));
                    entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapR(q, l), tap_box));
                }
            }
            continue;
        }
        for l in 0..4 {
            entries.push(ThetaEntry::f32(ParamKind::OpTDelay(q, l), delay_box));
            entries.push(ThetaEntry::f32(ParamKind::OpTRampSpan(q, l), span_box));
            entries.push(ThetaEntry::fp16(
                ParamKind::OpTLfoDepth(q, l),
                lfo_depth_box,
            ));
            entries.push(ThetaEntry::fp16(ParamKind::OpTInputGain(q, l), tap_box));
            entries.push(ThetaEntry::fp16(ParamKind::OpTAllpassG(q, l), ap_box));
            if include_extras {
                entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapL(q, l), tap_box));
                entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapR(q, l), tap_box));
            }
        }
        entries.push(ThetaEntry::f32(ParamKind::OpTGLoop(q), gloop_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTRDc(q), rdc_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTDampPole(q), damp_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTLfoRate(q), lfo_rate_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTRampRate(q), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTFbPsi(q), angle_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTSigmaKnee(q), sigma_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTFreezeMix(q), freeze_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTWindowDepth(q), unit_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTScatPsi(q), angle_box));
    }
    if include_extras && active_opt_quads > 0 {
        entries.push(ThetaEntry::fp16(ParamKind::TankDryMix, tap_box));
    }
    if include_direct_readouts {
        let pre_box = ParamBoundSpec::Box { lo: -4.0, hi: 4.0 };
        let ro_box = ParamBoundSpec::Box { lo: -8.0, hi: 8.0 };
        for c in 0..2 {
            entries.push(ThetaEntry::fp16(ParamKind::PreConvU(c), pre_box));
            entries.push(ThetaEntry::fp16(ParamKind::DirectS2U(c), ro_box));
            entries.push(ThetaEntry::fp16(ParamKind::DirectS2Hp(c), ro_box));
        }
        for k in 0..k_count {
            entries.push(ThetaEntry::f32(ParamKind::KautzCReL(k), ro_box));
            entries.push(ThetaEntry::f32(ParamKind::KautzCImL(k), ro_box));
            entries.push(ThetaEntry::f32(ParamKind::KautzCReR(k), ro_box));
            entries.push(ThetaEntry::f32(ParamKind::KautzCImR(k), ro_box));
        }
    }

    let mut kautz_active_mask = [false; 32];
    for (k, kp) in model.kautz.iter().enumerate().take(k_count) {
        kautz_active_mask[k] = kp.r_low > 0.0 && kp.b_gamma > 0.0;
    }
    (q_count, k_count, kautz_active_mask)
}

#[cfg(test)]
mod tests {
    use super::super::super::ThetaLayout;
    use super::super::ParamKind;
    use super::*;
    use crate::solver::model::TransportQuadF64;

    #[test]
    fn test_linear_modal_subspace_excludes_core_tiles_and_w_ro() {
        let mut model = SonaModelF64::new(48_000.0);
        model.active_kautz_pairs = 6;
        for kp in &mut model.kautz[..6] {
            kp.r_low = 0.94;
            kp.theta = 0.12;
            kp.b_gamma = 0.34;
        }
        model.transport_quads[0] = TransportQuadF64 {
            delay_samples: [67.0, 113.0, 173.0, 241.0],
            input_gains: [1.0; 4],
            wet_tap_l: [0.22, 0.0, -0.11, 0.0],
            wet_tap_r: [0.0, 0.19, 0.0, 0.14],
            allpass_delays: [0; 4],
            ..TransportQuadF64::default()
        };
        let layout =
            ThetaLayout::for_tier_from_model_linear_modal(&model, 3, true, true, 512.0, 0.9992);
        let kinds = layout.entries_kinds();
        assert!(kinds.len() < 120, "dim={}", kinds.len());
        assert!(kinds.iter().any(|k| matches!(k, ParamKind::KautzRLow(0))));
        assert!(kinds.iter().any(|k| matches!(k, ParamKind::OpTDelay(0, 0))));
        assert!(kinds.iter().any(|k| matches!(k, ParamKind::KautzCReL(0))));
        assert!(!kinds.iter().any(|k| matches!(
            k,
            ParamKind::CoreSvfG(_, _)
                | ParamKind::CoreAdaaDrive(_, _)
                | ParamKind::TileOdd(_, _, _)
                | ParamKind::SenseSvfG(_)
                | ParamKind::WReadout(_, _)
                | ParamKind::OpTGLoop(_)
        )));
    }
}
