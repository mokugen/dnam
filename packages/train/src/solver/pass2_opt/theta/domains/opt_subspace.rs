//! Active `Op-T` quad detection and passive feed-forward LTI subspace locking for
//! [`super::NonlinearSubspaceSpec`] (`ref[train.pipeline.backbone-trajectory-opt]`).

use super::super::super::super::model::{SonaModelF64, TransportQuadF64};
use super::super::super::op_t_quad::has_wet_op_t_output;

/// Returns `true` when `tq` has active wet output and is strictly passive LTI feed-forward
/// (zero feedback, zero modulation, zero scattering/allpass/freezing).
#[inline]
#[must_use]
pub(in super::super) fn is_feedforward_only_op_t_quad(tq: &TransportQuadF64) -> bool {
    has_wet_op_t_output(tq)
        && tq.g_loop.abs() <= 1.0e-12
        && tq.lfo_rate_hz.abs() <= 1.0e-12
        && tq.ramp_rate_hz.abs() <= 1.0e-12
        && tq.freeze_mix.abs() <= 1.0e-12
        && tq.damp_pole.abs() <= 1.0e-12
        && tq.scat_psi.abs() <= 1.0e-12
        && tq.fb_psi.abs() <= 1.0e-12
        && tq.allpass_g.iter().all(|&v| v.abs() <= 1.0e-12)
        && tq.allpass_delays == [0; 4]
        && tq.lfo_depths.iter().all(|&v| v.abs() <= 1.0e-12)
        && tq.ramp_spans_samples.iter().all(|&v| v.abs() <= 1.0e-12)
}

/// Computes `(active_opt_quads, opt_feedforward_only_mask)` for `max_opt_quads` active slots.
pub(in super::super) fn active_opt_quad_spec(
    model: &SonaModelF64,
    max_opt_quads: usize,
) -> (usize, [bool; 4]) {
    let active_opt_quads = (0..max_opt_quads)
        .rev()
        .find(|&q| {
            has_wet_op_t_output(&model.transport_quads[q])
                || model.transport_quads[q].g_loop.abs() > 1.0e-9
                || model.transport_quads[q]
                    .input_gains
                    .iter()
                    .any(|&v| v.abs() > 1.0e-9)
                || model.opt_to_lr[(q * 4)..((q + 1) * 4)]
                    .iter()
                    .any(|row| row[0].abs() > 1.0e-9 || row[1].abs() > 1.0e-9)
        })
        .map_or(0, |q| q + 1);
    let mut ff_mask = [false; 4];
    for (q, flag) in ff_mask.iter_mut().enumerate().take(active_opt_quads) {
        *flag = is_feedforward_only_op_t_quad(&model.transport_quads[q])
            && model.opt_to_lr[(q * 4)..((q + 1) * 4)]
                .iter()
                .all(|row| row[0].abs() <= 1.0e-12 && row[1].abs() <= 1.0e-12);
    }
    (active_opt_quads, ff_mask)
}

#[cfg(test)]
mod tests {
    use super::super::super::ThetaLayout;
    use super::super::ParamKind;
    use super::*;

    #[test]
    fn test_passive_feedforward_quad_locks_feedback_mod_and_inactive_quads() {
        let mut model = SonaModelF64::new(48_000.0);
        model.transport_quads[0] = TransportQuadF64 {
            delay_samples: [71.0, 137.0, 199.0, 311.0],
            input_gains: [1.0; 4],
            wet_tap_l: [0.28, 0.0, -0.16, 0.0],
            wet_tap_r: [0.0, 0.25, 0.0, 0.15],
            allpass_delays: [0; 4],
            ..TransportQuadF64::default()
        };
        let layout = ThetaLayout::for_tier_from_model(&model, 3, true, false, 512.0);
        let kinds = layout.entries_kinds();
        assert!(kinds.iter().any(|k| matches!(k, ParamKind::OpTDelay(0, 0))));
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, ParamKind::OpTInputGain(0, 3)))
        );
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, ParamKind::OpTWetTapL(0, 0)))
        );
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, ParamKind::OpTWetTapR(0, 3)))
        );
        assert!(kinds.iter().any(|k| matches!(k, ParamKind::TankDryMix)));
        assert!(!kinds.iter().any(|k| matches!(
            k,
            ParamKind::OpTGLoop(_)
                | ParamKind::OpTLfoRate(_)
                | ParamKind::OpTRampRate(_)
                | ParamKind::OpTLfoDepth(_, _)
                | ParamKind::OpTRampSpan(_, _)
                | ParamKind::OpTAllpassG(_, _)
                | ParamKind::OptToLr(_, _)
                | ParamKind::OpTDelay(1..=3, _)
        )));
    }
}
