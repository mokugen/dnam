//! Stage III Early FIR + 32-Pole Kautz ZOH Sample-Rate Adaptation & `3 kHz` Formant Modulation.

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, algebraic_unit_rotor, sqrt_f64};
use crate::operators::op_m::KautzModalBank;
use crate::stages::stage1_svf::exp2_signed_oct_f32;
use crate::stages::stage2_gru::block::{FIR_RING_MASK, MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES};

/// Host-rate Stage III modal formant and radial-$Q$ modulation parameters (`3 kHz` control boundary).
#[derive(Clone, Copy, Debug)]
pub struct KautzRetuneContext<'a> {
    pub host_poles: &'a [(f32, f32); MAX_KAUTZ_POLES],
    pub host_b_gamma: &'a [f32; MAX_KAUTZ_POLES],
    pub host_inv_1mr2: &'a [f32; MAX_KAUTZ_POLES],
    pub num_banks: usize,
    pub oct_shift: f32,
    pub damp_delta: f32,
    pub prev_oct_shift: f32,
}

/// Computes ZOH-rescaled Stage III Kautz onset delay in samples (`round(16 / rho_sr)`,
/// clamped inside the FIR mirrored ring buffer, `ref[dsp.zoh.stage3-fir-onset]`).
#[inline]
#[must_use]
pub fn compute_onset_delay_samples(zoh: &ZohRescaledParams) -> usize {
    if (zoh.rho_sr - 1.0).abs() < 1.0e-12 {
        return 16;
    }
    let d_f64 = f64::from(zoh.rescale_alignment_delay(16.0));
    (d_f64.round() as usize).clamp(14, FIR_RING_MASK)
}

/// Rescales Stage III complex Kautz modal poles `(r, theta)` under continuous-time ZOH
/// (`ref[dsp.zoh.op-m-poles]`), preserving `T_60` and physical resonance frequencies across
/// `44.1 / 48 / 88.2 / 96 / 192 kHz`.
#[must_use]
pub fn rescale_kautz_poles_with_zoh(
    base_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    zoh: &ZohRescaledParams,
) -> [(f32, f32); MAX_KAUTZ_POLES] {
    if (zoh.rho_sr - 1.0).abs() < 1.0e-12 {
        return *base_poles;
    }
    let mut out = [(0.90_f32, 0.25_f32); MAX_KAUTZ_POLES];
    for (m, &(r_raw, theta_raw)) in base_poles.iter().enumerate() {
        let r_clamped = r_raw.clamp(0.25, 0.9985);
        let theta_clamped = theta_raw.clamp(0.0, core::f32::consts::PI - 1.0e-3);
        let (r_host, _, theta_host, _) =
            zoh.rescale_op_m_channel(r_clamped, 0.0, theta_clamped, 1.0);
        out[m] = (
            r_host.clamp(0.25, 0.9995),
            theta_host.clamp(0.0, core::f32::consts::PI - 1.0e-4),
        );
    }
    out
}

/// Retunes active Stage III Kautz modal pole frequencies (log-octave formant scaling),
/// radial-$Q$ damping, orthonormal $\mathcal{H}_2$ input scaling `b_gamma`, and sweep contraction
/// at the `3 kHz` control rate (`0 FDIV`, `0 FSQRT`, `0 libm`, `ref[dsp.op-m.stage3.coupled-form]`,
/// `ref[dsp.op-m.stage3.sweep-contraction]`).
#[inline]
pub fn retune_kautz_formant_3khz<B: MathBackend>(
    kautz_l: &mut [KautzModalBank<4>; MAX_KAUTZ_BANKS],
    kautz_r: &mut [KautzModalBank<4>; MAX_KAUTZ_BANKS],
    ctx: &KautzRetuneContext<'_>,
) {
    let oct = ctx.oct_shift.clamp(-1.5, 1.5);
    let prev_oct = ctx.prev_oct_shift.clamp(-1.5, 1.5);
    let damp = ctx.damp_delta.clamp(-1.5, 1.5);
    let banks = ctx.num_banks.min(MAX_KAUTZ_BANKS);

    if oct.abs() <= 1.0e-7 && damp.abs() <= 1.0e-7 && (oct - prev_oct).abs() <= 1.0e-7 {
        for b in 0..banks {
            let mut cos_v = [1.0_f32; 4];
            let mut sin_v = [0.0_f32; 4];
            let mut r_v = [0.0_f32; 4];
            let mut bg_v = [0.0_f32; 4];
            for lane in 0..4 {
                let m = (b << 2) + lane;
                let (r_base, th_base) = ctx.host_poles[m];
                let r = r_base.clamp(0.25, 0.9995);
                let th = th_base.clamp(0.0, core::f32::consts::PI - 1.0e-4);
                let (c, s) = algebraic_unit_rotor(th);
                r_v[lane] = r;
                cos_v[lane] = c;
                sin_v[lane] = s;
                bg_v[lane] = ctx.host_b_gamma[m];
            }
            let rv = B::Vec4::load(&r_v);
            let lam_re = rv.mul(B::Vec4::load(&cos_v));
            let lam_im = rv.mul(B::Vec4::load(&sin_v));
            let bg = B::Vec4::load(&bg_v);
            lam_re.store(&mut kautz_l[b].lambda_re);
            lam_im.store(&mut kautz_l[b].lambda_im);
            bg.store(&mut kautz_l[b].b_gamma);
            lam_re.store(&mut kautz_r[b].lambda_re);
            lam_im.store(&mut kautz_r[b].lambda_im);
            bg.store(&mut kautz_r[b].b_gamma);
        }
        return;
    }

    let freq_scale = exp2_signed_oct_f32(oct);
    let prev_freq_scale = exp2_signed_oct_f32(prev_oct);
    let damp_scale = exp2_signed_oct_f32(damp);
    let one = B::Vec4::splat(1.0);
    let min_omr2 = B::Vec4::splat(1.0e-7);

    for b in 0..banks {
        let mut cos_v = [1.0_f32; 4];
        let mut sin_v = [0.0_f32; 4];
        let mut r_v = [0.0_f32; 4];
        let mut bg_v = [0.0_f32; 4];
        let mut inv_v = [1.0_f32; 4];
        for lane in 0..4 {
            let m = (b << 2) + lane;
            let (r_base_raw, th_base_raw) = ctx.host_poles[m];
            let r_base = r_base_raw.clamp(0.25, 0.9995);
            let th_base = th_base_raw.clamp(0.0, core::f32::consts::PI - 1.0e-4);
            let (th_new, dtheta) = if th_base <= 1.0e-6 {
                (0.0, 0.0)
            } else {
                let tn = (th_base * freq_scale).clamp(1.0e-4, core::f32::consts::PI - 1.0e-4);
                let tp = (th_base * prev_freq_scale).clamp(1.0e-4, core::f32::consts::PI - 1.0e-4);
                (tn, (tn - tp).abs())
            };
            let r_damped = (1.0 - (1.0 - r_base) * damp_scale).clamp(0.25, 0.9995);
            let r_eff = if dtheta > 1.0e-6 {
                r_damped.min((0.9950 - 0.15 * dtheta).max(0.25))
            } else {
                r_damped
            };
            let (c, s) = algebraic_unit_rotor(th_new);
            r_v[lane] = r_eff;
            cos_v[lane] = c;
            sin_v[lane] = s;
            bg_v[lane] = ctx.host_b_gamma[m];
            inv_v[lane] = ctx.host_inv_1mr2[m];
        }
        let rv = B::Vec4::load(&r_v);
        let lam_re = rv.mul(B::Vec4::load(&cos_v));
        let lam_im = rv.mul(B::Vec4::load(&sin_v));
        let omr2 = one.fms(rv, rv).max(min_omr2);
        let ratio_sq = omr2.mul(B::Vec4::load(&inv_v));
        let ratio = ratio_sq.mul(ratio_sq.rsqrt_pos_nr3());
        let bg_eff = B::Vec4::load(&bg_v).mul(ratio);

        lam_re.store(&mut kautz_l[b].lambda_re);
        lam_im.store(&mut kautz_l[b].lambda_im);
        bg_eff.store(&mut kautz_l[b].b_gamma);
        lam_re.store(&mut kautz_r[b].lambda_re);
        lam_im.store(&mut kautz_r[b].lambda_im);
        bg_eff.store(&mut kautz_r[b].b_gamma);
    }
}

/// Computes orthonormal `b_gamma = sqrt(1 - r^2)` in `f64` for cold initialization (`0` RT calls).
#[inline]
#[must_use]
pub(crate) fn cold_b_gamma(r: f32) -> f32 {
    sqrt_f64((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend, newton6_reciprocal_f64};

    #[test]
    fn test_kautz_zoh_scaling_octave_formant_damping_and_sweep_contraction() {
        let zoh48 = ZohRescaledParams::new(48_000, 48_000);
        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        assert_eq!(compute_onset_delay_samples(&zoh48), 16);
        assert_eq!(compute_onset_delay_samples(&zoh96), 32);

        let mut base = [(0.92_f32, 0.40_f32); MAX_KAUTZ_POLES];
        base[0] = (0.985, 0.008); // 61 Hz cabinet port mode
        base[1] = (0.92, 0.0); // real pole stays real
        assert_eq!(rescale_kautz_poles_with_zoh(&base, &zoh48), base);
        let scaled = rescale_kautz_poles_with_zoh(&base, &zoh96);
        assert!(scaled[0].0 > base[0].0);
        assert!((scaled[0].1 - 0.004).abs() < 1.0e-5);

        let bg: [f32; MAX_KAUTZ_POLES] = base.map(|(r, _)| cold_b_gamma(r));
        let inv: [f32; MAX_KAUTZ_POLES] =
            base.map(|(r, _)| newton6_reciprocal_f64(1.0 - f64::from(r) * f64::from(r)) as f32);

        let mut kl: [KautzModalBank<4>; MAX_KAUTZ_BANKS] =
            core::array::from_fn(|_| KautzModalBank::new());
        let mut kr: [KautzModalBank<4>; MAX_KAUTZ_BANKS] =
            core::array::from_fn(|_| KautzModalBank::new());
        // Verify log-octave doubling (+1 oct doubles theta for both 61 Hz and ~3 kHz modes)
        retune_kautz_formant_3khz::<DefaultMathBackend>(
            &mut kl,
            &mut kr,
            &KautzRetuneContext {
                host_poles: &base,
                host_b_gamma: &bg,
                host_inv_1mr2: &inv,
                num_banks: 2,
                oct_shift: 1.0,
                damp_delta: 0.0,
                prev_oct_shift: 1.0,
            },
        );
        let r_lo = kl[0].lambda_re[0].hypot(kl[0].lambda_im[0]);
        let th_lo = kl[0].lambda_im[0].atan2(kl[0].lambda_re[0]);
        let th_hi = kl[0].lambda_im[2].atan2(kl[0].lambda_re[2]);
        assert!((th_lo - 0.016).abs() < 1.0e-4, "61 Hz mode theta={th_lo}");
        assert!((th_hi - 0.80).abs() < 1.0e-4, "3 kHz mode theta={th_hi}");
        assert_eq!(kl[0].lambda_im[1], 0.0);
        assert!((r_lo - 0.985).abs() < 1.0e-5);
        assert!((kl[0].b_gamma[0] - bg[0]).abs() < 1.0e-5);

        // Verify radial-Q damping & dynamic sweep contraction + Scalar parity
        let mut kl_s = kl.clone();
        let mut kr_s = kr.clone();
        retune_kautz_formant_3khz::<DefaultMathBackend>(
            &mut kl,
            &mut kr,
            &KautzRetuneContext {
                host_poles: &base,
                host_b_gamma: &bg,
                host_inv_1mr2: &inv,
                num_banks: 2,
                oct_shift: 1.0,
                damp_delta: -1.0,
                prev_oct_shift: 0.0,
            },
        );
        retune_kautz_formant_3khz::<ScalarBackend>(
            &mut kl_s,
            &mut kr_s,
            &KautzRetuneContext {
                host_poles: &base,
                host_b_gamma: &bg,
                host_inv_1mr2: &inv,
                num_banks: 2,
                oct_shift: 1.0,
                damp_delta: -1.0,
                prev_oct_shift: 0.0,
            },
        );
        let r_contracted = kl[0].lambda_re[2].hypot(kl[0].lambda_im[2]);
        let dtheta = 0.40_f32;
        assert!(r_contracted <= 0.9950 - 0.15 * dtheta + 1.0e-5);
        let expected_bg = cold_b_gamma(r_contracted);
        assert!((kl[0].b_gamma[2] - expected_bg).abs() < 1.0e-4);
        assert!((kl[0].b_gamma[2] - kl_s[0].b_gamma[2]).abs() < 1.0e-6);

        // Zero modulation restores exact 0-ULP resting b_gamma
        retune_kautz_formant_3khz::<DefaultMathBackend>(
            &mut kl,
            &mut kr,
            &KautzRetuneContext {
                host_poles: &base,
                host_b_gamma: &bg,
                host_inv_1mr2: &inv,
                num_banks: 2,
                oct_shift: 0.0,
                damp_delta: 0.0,
                prev_oct_shift: 0.0,
            },
        );
        assert_eq!(kl[0].b_gamma[0].to_bits(), bg[0].to_bits());
    }
}
