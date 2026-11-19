//! Stage III Early FIR + 32-Pole Kautz ZOH Sample-Rate Adaptation, Shared-Pole Stereo SoA Bank,
//! & `3 kHz` Formant Modulation (`stage3_cab/kautz_zoh.rs`).

use dnam_kernels::{KautzQuadState, step_kautz_quad_stereo8};

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, algebraic_unit_rotor, exp2_signed_oct_f32, sqrt_f64};
use crate::stages::sona_core::model::{MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES};

/// Shared-pole `4`-pair stereo Kautz SoA bank matching `dnam_kernels::KautzQuadState` (`176 B`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KautzStereoBank4 {
    pub s_re_l: [f32; 4],
    pub s_im_l: [f32; 4],
    pub s_re_r: [f32; 4],
    pub s_im_r: [f32; 4],
    pub lambda_re: [f32; 4],
    pub lambda_im: [f32; 4],
    pub b_gamma: [f32; 4],
    pub c_re_l: [f32; 4],
    pub c_im_l: [f32; 4],
    pub c_re_r: [f32; 4],
    pub c_im_r: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<KautzStereoBank4>() == 176);

impl Default for KautzStereoBank4 {
    fn default() -> Self {
        Self::new()
    }
}

impl KautzStereoBank4 {
    /// Initializes a resting zeroed 4-pair shared-pole stereo Kautz bank.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            s_re_l: [0.0; 4],
            s_im_l: [0.0; 4],
            s_re_r: [0.0; 4],
            s_im_r: [0.0; 4],
            lambda_re: [0.0; 4],
            lambda_im: [0.0; 4],
            b_gamma: [1.0; 4],
            c_re_l: [0.0; 4],
            c_im_l: [0.0; 4],
            c_re_r: [0.0; 4],
            c_im_r: [0.0; 4],
        }
    }

    /// Zeros recursive modal state across both Left and Right channels.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.s_re_l = [0.0; 4];
        self.s_im_l = [0.0; 4];
        self.s_re_r = [0.0; 4];
        self.s_im_r = [0.0; 4];
    }

    /// Sets complex conjugate pole pair `lane` (`0..4`) with distinct L/R complex residues.
    pub fn set_pole_stereo(
        &mut self,
        lane: usize,
        r: f32,
        theta: f32,
        b_gamma: f32,
        cl: (f32, f32),
        cr: (f32, f32),
    ) {
        if lane >= 4 {
            return;
        }
        let r_c = r.clamp(0.0, 0.9995);
        let th_c = theta.clamp(0.0, core::f32::consts::PI - 1.0e-4);
        let (c, s) = algebraic_unit_rotor(th_c);
        self.lambda_re[lane] = r_c * c;
        self.lambda_im[lane] = r_c * s;
        self.b_gamma[lane] = b_gamma;
        self.c_re_l[lane] = cl.0;
        self.c_im_l[lane] = cl.1;
        self.c_re_r[lane] = cr.0;
        self.c_im_r[lane] = cr.1;
    }

    /// Loads this bank into a register-resident [`dnam_kernels::KautzQuadState`] for sub-block stepping.
    #[inline(always)]
    #[must_use]
    pub fn to_kernel_state<V: dnam_kernels::KernelVec4>(&self) -> KautzQuadState<V> {
        KautzQuadState {
            z_re_l: V::from_array(self.s_re_l),
            z_im_l: V::from_array(self.s_im_l),
            z_re_r: V::from_array(self.s_re_r),
            z_im_r: V::from_array(self.s_im_r),
            alpha_re: V::from_array(self.lambda_re),
            alpha_im: V::from_array(self.lambda_im),
            b_re: V::from_array(self.b_gamma),
            c_re_l: V::from_array(self.c_re_l),
            c_im_l: V::from_array(self.c_im_l),
            c_re_r: V::from_array(self.c_re_r),
            c_im_r: V::from_array(self.c_im_r),
        }
    }

    /// Writes evolved state `z_re_l..z_im_r` back after stepping.
    #[inline(always)]
    pub fn store_kernel_state<V: dnam_kernels::KernelVec4>(&mut self, st: &KautzQuadState<V>) {
        self.s_re_l = st.z_re_l.to_array();
        self.s_im_l = st.z_im_l.to_array();
        self.s_re_r = st.z_re_r.to_array();
        self.s_im_r = st.z_im_r.to_array();
    }

    /// Steps 4 conjugate Kautz pole pairs via shared `dnam_kernels::step_kautz_quad_stereo8`.
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(&mut self, x_l: f32, x_r: f32) -> (f32, f32) {
        let mut st = self.to_kernel_state::<B::Vec4>();
        let out = step_kautz_quad_stereo8::<B::Stereo8>(&mut st, x_l, x_r);
        self.store_kernel_state(&st);
        out
    }
}

/// Cold Stage III Kautz ZOH host poles, orthonormal scaling tables, and `3 kHz` slew memory
/// kept out of the hot L1D audio block (`SonaColdTail` / boxed cold descriptor).
#[derive(Clone, Debug, PartialEq)]
pub struct KautzColdSlewState {
    pub host_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub host_b_gamma: [f32; MAX_KAUTZ_POLES],
    pub host_inv_1mr2: [f32; MAX_KAUTZ_POLES],
    pub last_dtheta: f32,
    pub last_damp: f32,
    pub prev_oct: f32,
    pub slew_active: bool,
}

impl Default for KautzColdSlewState {
    fn default() -> Self {
        Self::new()
    }
}

impl KautzColdSlewState {
    /// Resting `48 kHz` cold Kautz slew table (`r = 0.90, theta = 0.25`).
    #[must_use]
    pub fn new() -> Self {
        let inv_omr2 = crate::math::newton6_reciprocal_f64(1.0 - 0.81) as f32;
        Self {
            host_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            host_b_gamma: [cold_b_gamma(0.90); MAX_KAUTZ_POLES],
            host_inv_1mr2: [inv_omr2; MAX_KAUTZ_POLES],
            last_dtheta: 0.0,
            last_damp: 0.0,
            prev_oct: 0.0,
            slew_active: false,
        }
    }

    /// Resets `3 kHz` slew trackers without altering cold poles/residues.
    pub fn reset_slew(&mut self) {
        self.last_dtheta = 0.0;
        self.last_damp = 0.0;
        self.prev_oct = 0.0;
        self.slew_active = false;
    }
}

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

/// Computes ZOH-rescaled Stage III Kautz onset delay in samples (`round(16 / rho_sr)`).
#[inline]
#[must_use]
pub fn compute_onset_delay_samples(zoh: &ZohRescaledParams) -> usize {
    if (zoh.rho_sr - 1.0).abs() < 1.0e-12 {
        return 16;
    }
    let d_f64 = f64::from(zoh.rescale_alignment_delay(16.0));
    (crate::math::round_f64(d_f64) as usize).clamp(14, 1_023)
}

/// Rescales Stage III complex Kautz modal poles `(r, theta)` under continuous-time ZOH.
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

/// Retunes a contiguous slice of Stage III Kautz modal banks starting at global bank `base_bank`
/// under `3 kHz` formant/damping/contraction targets (`0 FDIV`, `0 FSQRT`, `0 libm`).
#[inline]
pub fn retune_kautz_slice_3khz<B: MathBackend>(
    base_bank: usize,
    kautz: &mut [KautzStereoBank4],
    ctx: &KautzRetuneContext<'_>,
) {
    let oct = ctx.oct_shift.clamp(-1.5, 1.5);
    let prev_oct = ctx.prev_oct_shift.clamp(-1.5, 1.5);
    let damp = ctx.damp_delta.clamp(-1.5, 1.5);
    let end_bank = ctx.num_banks.min(MAX_KAUTZ_BANKS);
    if base_bank >= end_bank {
        return;
    }
    let count = kautz.len().min(end_bank - base_bank);

    if oct.abs() <= 1.0e-7 && damp.abs() <= 1.0e-7 && (oct - prev_oct).abs() <= 1.0e-7 {
        for (local_b, bank) in kautz.iter_mut().take(count).enumerate() {
            let b = base_bank + local_b;
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
            bank.lambda_re = rv.mul(B::Vec4::load(&cos_v)).to_array();
            bank.lambda_im = rv.mul(B::Vec4::load(&sin_v)).to_array();
            bank.b_gamma = bg_v;
        }
        return;
    }

    let freq_scale = exp2_signed_oct_f32(oct);
    let prev_freq_scale = exp2_signed_oct_f32(prev_oct);
    let damp_scale = exp2_signed_oct_f32(damp);
    let one = B::Vec4::splat(1.0);
    let min_omr2 = B::Vec4::splat(1.0e-7);

    for (local_b, bank) in kautz.iter_mut().take(count).enumerate() {
        let b = base_bank + local_b;
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
        let omr2 = one.fms(rv, rv).max(min_omr2);
        let ratio_sq = omr2.mul(B::Vec4::load(&inv_v));
        let ratio = ratio_sq.mul(ratio_sq.rsqrt_pos_nr3());
        let bg_eff = B::Vec4::load(&bg_v).mul(ratio);

        bank.lambda_re = rv.mul(B::Vec4::load(&cos_v)).to_array();
        bank.lambda_im = rv.mul(B::Vec4::load(&sin_v)).to_array();
        bank.b_gamma = bg_eff.to_array();
    }
}

/// Retunes active Stage III Kautz modal pole frequencies, radial-$Q$ damping, orthonormal `b_gamma`,
/// and sweep contraction at the `3 kHz` control rate (`0 FDIV`, `0 FSQRT`, `0 libm`).
#[inline]
pub fn retune_kautz_formant_3khz<B: MathBackend>(
    kautz: &mut [KautzStereoBank4; MAX_KAUTZ_BANKS],
    ctx: &KautzRetuneContext<'_>,
) {
    retune_kautz_slice_3khz::<B>(0, kautz, ctx);
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
        base[0] = (0.985, 0.008);
        base[1] = (0.92, 0.0);
        assert_eq!(rescale_kautz_poles_with_zoh(&base, &zoh48), base);
        let scaled = rescale_kautz_poles_with_zoh(&base, &zoh96);
        assert!(scaled[0].0 > base[0].0);
        assert!((scaled[0].1 - 0.004).abs() < 1.0e-5);

        let bg: [f32; MAX_KAUTZ_POLES] = base.map(|(r, _)| cold_b_gamma(r));
        let inv: [f32; MAX_KAUTZ_POLES] =
            base.map(|(r, _)| newton6_reciprocal_f64(1.0 - f64::from(r) * f64::from(r)) as f32);

        let mut kb = [KautzStereoBank4::new(); MAX_KAUTZ_BANKS];
        retune_kautz_formant_3khz::<DefaultMathBackend>(
            &mut kb,
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
        let r_lo = kb[0].lambda_re[0].hypot(kb[0].lambda_im[0]);
        let th_lo = kb[0].lambda_im[0].atan2(kb[0].lambda_re[0]);
        let th_hi = kb[0].lambda_im[2].atan2(kb[0].lambda_re[2]);
        assert!((th_lo - 0.016).abs() < 1.0e-4, "61 Hz mode theta={th_lo}");
        assert!((th_hi - 0.80).abs() < 1.0e-4, "3 kHz mode theta={th_hi}");
        assert_eq!(kb[0].lambda_im[1], 0.0);
        assert!((r_lo - 0.985).abs() < 1.0e-5);
        assert!((kb[0].b_gamma[0] - bg[0]).abs() < 1.0e-5);

        let mut kb_s = kb;
        retune_kautz_formant_3khz::<DefaultMathBackend>(
            &mut kb,
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
            &mut kb_s,
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
        let r_contracted = kb[0].lambda_re[2].hypot(kb[0].lambda_im[2]);
        assert!(r_contracted <= 0.9950 - 0.15 * 0.40 + 1.0e-5);
        assert!((kb[0].b_gamma[2] - kb_s[0].b_gamma[2]).abs() < 1.0e-6);

        retune_kautz_formant_3khz::<DefaultMathBackend>(
            &mut kb,
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
        assert_eq!(kb[0].b_gamma[0].to_bits(), bg[0].to_bits());
    }
}
