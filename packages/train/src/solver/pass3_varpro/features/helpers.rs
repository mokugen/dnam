//! Stateless Pass-3 high-pass shelf, `Op-T` bank filtering, and tier prefix unpacking helpers.

use crate::control::stability::MAX_RO_POLE_48K_F64;

use super::super::super::model::cascade::sanitize_unit_f64;
use super::super::super::model::{SonaModelF64, TierReadoutPrefixF64};
use super::super::super::pass2_opt::op_t_quad::{
    eval_op_t_bank_stereo_and_adjoint, has_wet_op_t_output,
};

/// Evaluates the scalar 1-pole high-pass branch `u[t] - lp(u; pole)[t]`.
#[must_use]
pub(in super::super) fn eval_highpass_pole(sig: &[f64], pole: f64) -> Vec<f64> {
    let mut hp = vec![0.0_f64; sig.len()];
    if pole <= 0.0 || !pole.is_finite() {
        return hp;
    }
    let p = pole.clamp(0.0, MAX_RO_POLE_48K_F64);
    let one_m_p = 1.0 - p;
    let mut lp = 0.0_f64;
    for (i, &u) in sig.iter().enumerate() {
        lp = p * lp + one_m_p * u;
        hp[i] = u - lp;
    }
    hp
}

#[inline]
pub(super) fn is_op_t_active_up_to(model: &SonaModelF64, active_opt: usize) -> bool {
    model.transport_quads[..active_opt.min(model.transport_quads.len())]
        .iter()
        .any(has_wet_op_t_output)
}

#[inline]
pub(super) fn is_op_t_strictly_lti_up_to(model: &SonaModelF64, active_opt: usize) -> bool {
    model.transport_quads[..active_opt.min(model.transport_quads.len())]
        .iter()
        .all(|tq| {
            tq.g_loop.abs() <= 1.0e-12
                && tq.lfo_rate_hz.abs() <= 1.0e-12
                && tq.ramp_rate_hz.abs() <= 1.0e-12
                && tq.freeze_mix.abs() <= 1.0e-12
                && tq.lfo_depths.iter().all(|&v| v.abs() <= 1.0e-12)
                && tq.ramp_spans_samples.iter().all(|&v| v.abs() <= 1.0e-12)
        })
}

pub(in super::super) fn apply_op_t_series_channel_bank(
    model: &SonaModelF64,
    sig: &[f64],
    is_right: bool,
    active_opt: usize,
) -> Vec<f64> {
    let dry = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let q_opt = active_opt.clamp(1, model.transport_quads.len());
    if !is_op_t_active_up_to(model, q_opt) {
        return sig.iter().map(|&v| dry * v).collect();
    }
    let zeros = vec![0.0_f64; sig.len()];
    let (xl, xr) = if is_right {
        (zeros.as_slice(), sig)
    } else {
        (sig, zeros.as_slice())
    };
    let (tl, tr, _) = eval_op_t_bank_stereo_and_adjoint(
        xl,
        xr,
        &model.transport_quads,
        q_opt,
        model.sample_rate_hz,
        model.dc_blocker_pole_5hz,
        None,
    );
    let wet = if is_right { &tr } else { &tl };
    sig.iter().zip(wet).map(|(&s, &w)| dry * s + w).collect()
}

pub(super) fn apply_op_t_stereo_bank(
    model: &SonaModelF64,
    sig_l: &[f64],
    sig_r: &[f64],
    active_opt: usize,
) -> (Vec<f64>, Vec<f64>) {
    let dry = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let q_opt = active_opt.clamp(1, model.transport_quads.len());
    if !is_op_t_active_up_to(model, q_opt) {
        return (
            sig_l.iter().map(|&v| dry * v).collect(),
            sig_r.iter().map(|&v| dry * v).collect(),
        );
    }
    let (tl, tr, _) = eval_op_t_bank_stereo_and_adjoint(
        sig_l,
        sig_r,
        &model.transport_quads,
        q_opt,
        model.sample_rate_hz,
        model.dc_blocker_pole_5hz,
        None,
    );
    let out_l = sig_l.iter().zip(&tl).map(|(&s, &w)| dry * s + w).collect();
    let out_r = sig_r.iter().zip(&tr).map(|(&s, &w)| dry * s + w).collect();
    (out_l, out_r)
}

pub(in super::super) fn unpack_weights_into_prefix(
    weights_l: &[f64],
    weights_r: &[f64],
    active_taps: usize,
    pc_gain: [f64; 2],
    max_tier_idx: usize,
    base_model: &SonaModelF64,
) -> TierReadoutPrefixF64 {
    let mut pfx = TierReadoutPrefixF64 {
        pre_conv_u_gain: pc_gain,
        pre_conv_hp_gain: base_model.pre_conv_hp_gain,
        direct_s2_u_gain: [weights_l[0].clamp(-8.0, 8.0), weights_r[0].clamp(-8.0, 8.0)],
        direct_s2_hp_gain: [weights_l[1].clamp(-8.0, 8.0), weights_r[1].clamp(-8.0, 8.0)],
        ..TierReadoutPrefixF64::default()
    };
    for l in 0..40 {
        pfx.w_ro[l][0..6].copy_from_slice(&base_model.w_ro[l][0..6]);
    }
    for (k, kz) in base_model.kautz.iter().enumerate() {
        pfx.kautz_b_gamma[k] = kz.b_gamma;
        pfx.kautz_c_re_l[k] = kz.c_re_l;
        pfx.kautz_c_im_l[k] = kz.c_im_l;
        pfx.kautz_c_re_r[k] = kz.c_re_r;
        pfx.kautz_c_im_r[k] = kz.c_im_r;
    }
    let tier_defs = [
        (0..16_usize, 0..active_taps.min(32)),
        (
            16..24_usize,
            32..(32 + active_taps.saturating_sub(32).min(16)),
        ),
        (
            24..32_usize,
            48..(48 + active_taps.saturating_sub(48).min(48)),
        ),
        (
            32..40_usize,
            96..(96 + active_taps.saturating_sub(96).min(32)),
        ),
    ];
    let mut col = 2_usize;
    for (ti, (lane_range, fir_range)) in tier_defs.into_iter().enumerate() {
        if ti > max_tier_idx {
            break;
        }
        for l in lane_range {
            pfx.w_ro[l][6] = weights_l[col].clamp(-8.0, 8.0);
            pfx.w_ro[l][7] = weights_r[col].clamp(-8.0, 8.0);
            pfx.w_ro[l][8] = weights_l[col + 1].clamp(-8.0, 8.0);
            pfx.w_ro[l][9] = weights_r[col + 1].clamp(-8.0, 8.0);
            pfx.w_ro[l][10] = weights_l[col + 2].clamp(-8.0, 8.0);
            pfx.w_ro[l][11] = weights_r[col + 2].clamp(-8.0, 8.0);
            col += 3;
        }
        for k in fir_range {
            pfx.fir_l[k] = weights_l[col].clamp(-8.0, 8.0);
            pfx.fir_r[k] = weights_r[col].clamp(-8.0, 8.0);
            col += 1;
        }
    }
    pfx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass3_helpers_hp_op_t_and_unpack_prefix() {
        let sig = [0.0, 1.0, 1.0, 1.0];
        assert!(eval_highpass_pole(&sig, 0.0).iter().all(|&v| v == 0.0));
        let hp = eval_highpass_pole(&sig, 0.9);
        assert!(hp[1] > 0.8 && hp[3] < hp[2]);

        let mut model = SonaModelF64::new(48_000.0);
        assert!(!is_op_t_active_up_to(&model, 1));
        assert!(is_op_t_strictly_lti_up_to(&model, 1));
        let dry_only = apply_op_t_series_channel_bank(&model, &sig, false, 1);
        assert_eq!(dry_only, sig);
        let (sl, sr) = apply_op_t_stereo_bank(&model, &sig, &sig, 1);
        assert_eq!(sl, sig);
        assert_eq!(sr, sig);

        model.transport_quads[0].input_gains[0] = 0.5;
        model.transport_quads[0].wet_tap_l[0] = 0.25;
        model.transport_quads[0].delay_samples[0] = 5.0;
        model.transport_quads[0].lfo_rate_hz = 1.0;
        assert!(is_op_t_active_up_to(&model, 1));
        assert!(!is_op_t_strictly_lti_up_to(&model, 1));
        let active_s = apply_op_t_series_channel_bank(&model, &[0.0; 16], true, 1);
        assert_eq!(active_s.len(), 16);
        let (al, ar) = apply_op_t_stereo_bank(&model, &[1.0; 16], &[1.0; 16], 1);
        assert!(al.iter().chain(&ar).all(|v| v.is_finite()));

        let wl = vec![0.1_f64; 2 + 16 * 3 + 32];
        let wr = vec![-0.1_f64; 2 + 16 * 3 + 32];
        let pfx = unpack_weights_into_prefix(&wl, &wr, 32, [0.5, -0.5], 0, &model);
        assert_eq!(pfx.pre_conv_u_gain, [0.5, -0.5]);
        assert!((pfx.direct_s2_u_gain[0] - 0.1).abs() < 1.0e-12);
        assert!((pfx.fir_l[0] - 0.1).abs() < 1.0e-12);
    }
}
