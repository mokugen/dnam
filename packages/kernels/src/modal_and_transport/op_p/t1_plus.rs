//! `Op-P` v4.0 `56 B` `T1+` 8-feature sub-harmonic, `2theta` doubler, Octavia/`OC-2`, and
//! stereo/2-band analytic tracker (`ref[dsp.op-m.tracker.t1-plus]`,
//! `ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`,
//! and `spirv-unknown-naga-wgsl`-clean.

use crate::simd::{
    DC_BLOCKER_5HZ_48K_F32, GpuVec4, KernelVec4, MAX_PARALRU_R_F32, canonicalize_zero_f32_bits,
    const_abs_f32, const_dc_blocker_pole_bits, flush_subnormal_f32_bits, scrub_and_flush_f32_bits,
};

use super::dc_and_tracker::tracker_crosses_branch_cut_f32;

/// Silence squelch squared-magnitude threshold (`r^2 < 1e-12`).
const T1_PLUS_SILENCE_R2: f32 = 1.0e-12;
/// Half-angle `(1 + |u_re|) * 0.5` singularity floor.
const T1_PLUS_HALF_ANGLE_EPS: f32 = 1.0e-12;
/// Default `50 Hz` pre-`T1` tracking LPF pole at `96 kHz` `2x` core (`ref[dsp.op-m.tracker.t1-plus]`).
pub const T1_PLUS_DEFAULT_TRK_LPF_96K_F32: f32 =
    f32::from_bits(const_dc_blocker_pole_bits(50, 96_000));
/// Default `200 Hz` 2-band sub-band tracking LPF pole at `96 kHz` `2x` core.
pub const T1_PLUS_DEFAULT_SUB_LPF_96K_F32: f32 =
    f32::from_bits(const_dc_blocker_pole_bits(200, 96_000));

const _: () =
    assert!(T1_PLUS_DEFAULT_TRK_LPF_96K_F32 > 0.99 && T1_PLUS_DEFAULT_TRK_LPF_96K_F32 < 1.0);
const _: () =
    assert!(T1_PLUS_DEFAULT_SUB_LPF_96K_F32 > 0.98 && T1_PLUS_DEFAULT_SUB_LPF_96K_F32 < 1.0);

#[inline(always)]
fn clamp_nan_to(x: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if x >= lo {
        if x <= hi { x } else { hi }
    } else if x < lo {
        lo
    } else {
        fallback
    }
}

#[inline(always)]
fn cancellation_free_half_angle_nr3(re: f32, im: f32) -> (f32, f32) {
    let h = (0.5 * (1.0 + const_abs_f32(re))).max(T1_PLUS_HALF_ANGLE_EPS);
    let rho_half = GpuVec4::splat(h).rsqrt_pos_nr3().to_array()[0];
    let v_maj = (h * rho_half).clamp(0.0, 1.0);
    let v_min = (const_abs_f32(im) * (0.5 * rho_half)).clamp(0.0, 1.0);
    let (c, abs_s) = if re >= 0.0 {
        (v_maj, v_min)
    } else {
        (v_min, v_maj)
    };
    let sigma_pos = if im >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    (c, sigma_pos * abs_s)
}

/// Evaluates the globally $C^1$ cubic-Hermite `soft_square` shaper `u * (1.5 - 0.5 * u^2)` on
/// `clamp(2 * sub1_sin, -1, 1)` (`f(±1) = ±1`, `f'(±1) = 0`, `ref[dsp.op-m.tracker.t1-plus]`).
#[inline(always)]
#[must_use]
pub fn eval_t1_plus_soft_square_f32(sub1_sin: f32) -> f32 {
    let u = clamp_nan_to(2.0 * sub1_sin, -1.0, 1.0, 0.0);
    u * (1.5 - 0.5 * u * u)
}

/// Exact `28 B` (`align(4)`) parameter bundle for [`T1PlusKernel`]
/// (`ref[dsp.op-m.tracker.t1-plus]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1PlusParams {
    /// Learned 1-pole pre-`T1` tracking LPF pole (`0.0`..`MAX_PARALRU_R_F32`).
    pub pole_trk: f32,
    /// `env_dc` / `sub_env_dc` envelope tracker pole (`0.0`..`MAX_PARALRU_R_F32`).
    pub pole_env_dc: f32,
    /// Low-band `2`-band tracker LPF pole (`0.0`..`MAX_PARALRU_R_F32`).
    pub pole_sub: f32,
    /// Octavia full-wave DC-cancelled rectifier mix (`feat[6]`).
    pub alpha_oct: f32,
    /// `OC-2` $C^1$ soft-square sub-octave mix (`feat[6]`).
    pub beta_oc2: f32,
    /// Differential L/R stereo asymmetry mix (`feat[7]`).
    pub stereo_mix: f32,
    /// Sub-band envelope tracking mix (`feat[7]`).
    pub sub_band_mix: f32,
}

const _: () = assert!(core::mem::size_of::<T1PlusParams>() == 28);
const _: () = assert!(core::mem::align_of::<T1PlusParams>() == 4);

impl Default for T1PlusParams {
    #[inline(always)]
    fn default() -> Self {
        Self::new_96k_default()
    }
}

impl T1PlusParams {
    /// Returns nominal `2x` `96 kHz` `T1+` defaults (`50 Hz` pre-`T1` LPF, `5 Hz` `env_dc`, `200 Hz` sub-band).
    #[inline(always)]
    #[must_use]
    pub const fn new_96k_default() -> Self {
        Self {
            pole_trk: T1_PLUS_DEFAULT_TRK_LPF_96K_F32,
            pole_env_dc: DC_BLOCKER_5HZ_48K_F32,
            pole_sub: T1_PLUS_DEFAULT_SUB_LPF_96K_F32,
            alpha_oct: 1.0,
            beta_oc2: 0.0,
            stereo_mix: 1.0,
            sub_band_mix: 0.0,
        }
    }
}

/// Exact `56 B` (`align(4)`) `Op-P` v4.0 `T1+` analytic 8-feature tracker state
/// (`ref[dsp.op-m.tracker.t1-plus]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1PlusKernel {
    prev_u_re: f32,
    prev_u_im: f32,
    sheet: f32,
    prev_w1_re: f32,
    prev_w1_im: f32,
    sheet2: f32,
    prev_valid: u32,
    trk_lpf_i: f32,
    trk_lpf_q: f32,
    env_dc: f32,
    sub_lpf_i: f32,
    sub_lpf_q: f32,
    stereo_diff_lpf: f32,
    sub_env_dc: f32,
}

const _: () = assert!(core::mem::size_of::<T1PlusKernel>() == 56);
const _: () = assert!(core::mem::align_of::<T1PlusKernel>() == 4);

impl Default for T1PlusKernel {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl T1PlusKernel {
    /// Creates a clean startup [`T1PlusKernel`] (`sheet = +1`, `sheet2 = +1`, `prev_u = prev_w1 = (1, 0)`,
    /// `prev_valid = false`, LPF/envelope states zeroed).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            sheet: 1.0,
            prev_w1_re: 1.0,
            prev_w1_im: 0.0,
            sheet2: 1.0,
            prev_valid: 0,
            trk_lpf_i: 0.0,
            trk_lpf_q: 0.0,
            env_dc: 0.0,
            sub_lpf_i: 0.0,
            sub_lpf_q: 0.0,
            stereo_diff_lpf: 0.0,
            sub_env_dc: 0.0,
        }
    }

    /// Resets all sheets, phase history, and tracking filters to startup defaults.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Steps the `56 B` `T1+` analytic tracker on true-stereo `H1` outputs `(i_l, q_l, i_r, q_r)`
    /// (`0 FDIV`, `0 FSQRT`, `0 libm`), emitting `[feat[0..8]]`:
    /// - `feat[0..4]`: `env_gate * [cos(theta/2), sin(theta/2), cos(theta/4), sin(theta/4)]`
    /// - `feat[4..6]`: `3`-FMA `2theta` doubler `env_gate * [u_re^2 - u_im^2, 2 * u_re * u_im]`
    /// - `feat[6]`: `alpha_oct * (|x_trk| - env_dc) + beta_oc2 * env_gate * soft_square(w_1)`
    /// - `feat[7]`: `stereo_mix * stereo_diff_lpf + sub_band_mix * (|sub_lpf_i| - sub_env_dc)`
    ///
    /// `ref[dsp.op-m.tracker.t1-plus]`
    #[inline(always)]
    pub fn step_stereo(
        &mut self,
        i_l: f32,
        q_l: f32,
        i_r: f32,
        q_r: f32,
        params: &T1PlusParams,
    ) -> [f32; 8] {
        let il = scrub_and_flush_f32_bits(i_l);
        let ql = scrub_and_flush_f32_bits(q_l);
        let ir = scrub_and_flush_f32_bits(i_r);
        let qr = scrub_and_flush_f32_bits(q_r);

        let p_trk = clamp_nan_to(params.pole_trk, 0.0, MAX_PARALRU_R_F32, 0.0);
        let p_env = clamp_nan_to(params.pole_env_dc, 0.0, MAX_PARALRU_R_F32, 0.0);
        let p_sub = clamp_nan_to(params.pole_sub, 0.0, MAX_PARALRU_R_F32, 0.0);
        let alpha_oct = clamp_nan_to(params.alpha_oct, -16.0, 16.0, 0.0);
        let beta_oc2 = clamp_nan_to(params.beta_oc2, -16.0, 16.0, 0.0);
        let stereo_mix = clamp_nan_to(params.stereo_mix, -16.0, 16.0, 0.0);
        let sub_band_mix = clamp_nan_to(params.sub_band_mix, -16.0, 16.0, 0.0);

        let i_mid = 0.5 * (il + ir);
        let q_mid = 0.5 * (ql + qr);
        let i_diff = 0.5 * (il - ir);

        let g_trk = 1.0 - p_trk;
        self.trk_lpf_i = canonicalize_zero_f32_bits(p_trk * self.trk_lpf_i + g_trk * i_mid);
        self.trk_lpf_q = canonicalize_zero_f32_bits(p_trk * self.trk_lpf_q + g_trk * q_mid);
        self.stereo_diff_lpf =
            canonicalize_zero_f32_bits(p_trk * self.stereo_diff_lpf + g_trk * i_diff);

        let g_sub = 1.0 - p_sub;
        self.sub_lpf_i = canonicalize_zero_f32_bits(p_sub * self.sub_lpf_i + g_sub * i_mid);
        self.sub_lpf_q = canonicalize_zero_f32_bits(p_sub * self.sub_lpf_q + g_sub * q_mid);

        let g_env = 1.0 - p_env;
        let abs_x_trk = const_abs_f32(self.trk_lpf_i);
        self.env_dc = canonicalize_zero_f32_bits(p_env * self.env_dc + g_env * abs_x_trk);
        let abs_x_sub = const_abs_f32(self.sub_lpf_i);
        self.sub_env_dc = canonicalize_zero_f32_bits(p_env * self.sub_env_dc + g_env * abs_x_sub);

        let octavia = alpha_oct * (abs_x_trk - self.env_dc);
        let feat7 = canonicalize_zero_f32_bits(
            stereo_mix * self.stereo_diff_lpf + sub_band_mix * (abs_x_sub - self.sub_env_dc),
        );

        let r2 = self.trk_lpf_i * self.trk_lpf_i + self.trk_lpf_q * self.trk_lpf_q;
        if !r2.is_finite() || r2 < T1_PLUS_SILENCE_R2 {
            self.prev_valid = 0;
            return [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                canonicalize_zero_f32_bits(octavia),
                feat7,
            ];
        }

        let inv_r = GpuVec4::splat(r2).rsqrt_pos_nr3().to_array()[0];
        let u_re = (self.trk_lpf_i * inv_r).clamp(-1.0, 1.0);
        let u_im = (self.trk_lpf_q * inv_r).clamp(-1.0, 1.0);
        let env_gate = flush_subnormal_f32_bits(r2 * inv_r);

        if self.prev_valid != 0
            && tracker_crosses_branch_cut_f32(self.prev_u_re, self.prev_u_im, u_re, u_im)
        {
            self.sheet = -self.sheet;
        }
        let (c1, s1) = cancellation_free_half_angle_nr3(u_re, u_im);
        let sub1_cos = self.sheet * c1;
        let sub1_sin = self.sheet * s1;

        if self.prev_valid != 0
            && tracker_crosses_branch_cut_f32(self.prev_w1_re, self.prev_w1_im, sub1_cos, sub1_sin)
        {
            self.sheet2 = -self.sheet2;
        }
        let (c2, s2) = cancellation_free_half_angle_nr3(sub1_cos, sub1_sin);
        let sub2_cos = self.sheet2 * c2;
        let sub2_sin = self.sheet2 * s2;

        self.prev_u_re = u_re;
        self.prev_u_im = u_im;
        self.prev_w1_re = sub1_cos;
        self.prev_w1_im = sub1_sin;
        self.prev_valid = 1;

        // 3-FMA 2*theta doubler: [u_re^2 - u_im^2, 2 * u_re * u_im] * env_gate
        let d_re = u_re * u_re - u_im * u_im;
        let d_im = 2.0 * u_re * u_im;
        let soft_sq = env_gate * eval_t1_plus_soft_square_f32(sub1_sin);

        [
            canonicalize_zero_f32_bits(sub1_cos * env_gate),
            canonicalize_zero_f32_bits(sub1_sin * env_gate),
            canonicalize_zero_f32_bits(sub2_cos * env_gate),
            canonicalize_zero_f32_bits(sub2_sin * env_gate),
            canonicalize_zero_f32_bits(env_gate * d_re),
            canonicalize_zero_f32_bits(env_gate * d_im),
            canonicalize_zero_f32_bits(octavia + beta_oc2 * soft_sq),
            feat7,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct T1PlusBpttInput<'a> {
        il: &'a [f64],
        ql: &'a [f64],
        ir: &'a [f64],
        qr: &'a [f64],
        pole_trk: f64,
        pole_env: f64,
        pole_sub: f64,
        alpha_oct: f64,
        beta_oc2: f64,
        stereo_mix: f64,
        sub_band_mix: f64,
        sheet: f64,
    }

    struct T1PlusBpttOut {
        loss: f64,
        adj_il: [f64; 4],
        adj_ql: [f64; 4],
        adj_ir: [f64; 4],
        adj_qr: [f64; 4],
        adj_ptrk: f64,
        adj_psub: f64,
        adj_aoct: f64,
        adj_boc2: f64,
        adj_smix: f64,
        adj_sbmix: f64,
    }

    fn t1_plus_bptt_loss_f64(inp: &T1PlusBpttInput<'_>) -> T1PlusBpttOut {
        let (il, ql, ir, qr) = (inp.il, inp.ql, inp.ir, inp.qr);
        let (pole_trk, pole_env, pole_sub) = (inp.pole_trk, inp.pole_env, inp.pole_sub);
        let (alpha_oct, beta_oc2, stereo_mix, sub_band_mix, sheet) = (
            inp.alpha_oct,
            inp.beta_oc2,
            inp.stereo_mix,
            inp.sub_band_mix,
            inp.sheet,
        );
        let n = il.len();
        let mut trk_i = 0.0_f64;
        let mut trk_q = 0.0_f64;
        let mut env_dc = 0.0_f64;
        let mut sub_i = 0.0_f64;
        let mut sdiff = 0.0_f64;
        let mut sub_env = 0.0_f64;

        let mut hist_trk_i = vec![0.0_f64; n];
        let mut hist_trk_q = vec![0.0_f64; n];
        let mut hist_env = vec![0.0_f64; n];
        let mut hist_sub_i = vec![0.0_f64; n];
        let mut hist_sdiff = vec![0.0_f64; n];
        let mut hist_sub_env = vec![0.0_f64; n];

        let mut loss = 0.0_f64;
        let mut out_feats = [[0.0_f64; 8]; 8];
        for t in 0..n {
            let i_mid = 0.5 * (il[t] + ir[t]);
            let q_mid = 0.5 * (ql[t] + qr[t]);
            let i_diff = 0.5 * (il[t] - ir[t]);

            trk_i = pole_trk * trk_i + (1.0 - pole_trk) * i_mid;
            trk_q = pole_trk * trk_q + (1.0 - pole_trk) * q_mid;
            sdiff = pole_trk * sdiff + (1.0 - pole_trk) * i_diff;
            sub_i = pole_sub * sub_i + (1.0 - pole_sub) * i_mid;
            env_dc = pole_env * env_dc + (1.0 - pole_env) * trk_i.abs();
            sub_env = pole_env * sub_env + (1.0 - pole_env) * sub_i.abs();

            hist_trk_i[t] = trk_i;
            hist_trk_q[t] = trk_q;
            hist_env[t] = env_dc;
            hist_sub_i[t] = sub_i;
            hist_sdiff[t] = sdiff;
            hist_sub_env[t] = sub_env;

            let r = (trk_i * trk_i + trk_q * trk_q).sqrt();
            let u_re = trk_i / r;
            let u_im = trk_q / r;
            let s1 = sheet * (0.5 * (1.0 - u_re)).sqrt() * (if u_im >= 0.0 { 1.0 } else { -1.0 });
            let u_sq = (2.0 * s1).clamp(-1.0, 1.0);
            let soft_sq = r * u_sq * (1.5 - 0.5 * u_sq * u_sq);
            let f4 = r * (u_re * u_re - u_im * u_im);
            let f5 = r * (2.0 * u_re * u_im);
            let f6 = alpha_oct * (trk_i.abs() - env_dc) + beta_oc2 * soft_sq;
            let f7 = stereo_mix * sdiff + sub_band_mix * (sub_i.abs() - sub_env);
            out_feats[t][4] = f4;
            out_feats[t][5] = f5;
            out_feats[t][6] = f6;
            out_feats[t][7] = f7;
            loss += f4 + 0.5 * f5 - 0.25 * f6 + 0.75 * f7;
        }

        // Reverse-time BPTT adjoint
        let mut adj_il = [0.0_f64; 4];
        let mut adj_ql = [0.0_f64; 4];
        let mut adj_ir = [0.0_f64; 4];
        let mut adj_qr = [0.0_f64; 4];
        let (mut adj_ptrk, mut adj_psub, mut adj_aoct, mut adj_boc2, mut adj_smix, mut adj_sbmix) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        let (mut l_ti, mut l_tq, mut l_env, mut l_si, mut l_sdiff, mut l_senv) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        for t in (0..n).rev() {
            let ti = hist_trk_i[t];
            let tq = hist_trk_q[t];
            let edc = hist_env[t];
            let si = hist_sub_i[t];
            let sd = hist_sdiff[t];
            let sedc = hist_sub_env[t];

            let r2 = ti * ti + tq * tq;
            let r = r2.sqrt();
            let u_re = ti / r;
            let u_im = tq / r;
            let sgn_im = if u_im >= 0.0 { 1.0 } else { -1.0 };
            let s1 = sheet * (0.5 * (1.0 - u_re)).sqrt() * sgn_im;
            let u_sq = (2.0 * s1).clamp(-1.0, 1.0);
            let soft_sq_unit = u_sq * (1.5 - 0.5 * u_sq * u_sq);

            let af4 = 1.0;
            let af5 = 0.5;
            let af6 = -0.25;
            let af7 = 0.75;

            adj_aoct += af6 * (ti.abs() - edc);
            adj_boc2 += af6 * (r * soft_sq_unit);
            adj_smix += af7 * sd;
            adj_sbmix += af7 * (si.abs() - sedc);

            l_sdiff += af7 * stereo_mix;
            let d_abs_si = af7 * sub_band_mix;
            l_senv += -af7 * sub_band_mix;
            let d_abs_ti = af6 * alpha_oct;
            l_env += -af6 * alpha_oct;

            let a_soft_sq = af6 * beta_oc2;
            let d_r = af4 * (u_re * u_re - u_im * u_im)
                + af5 * (2.0 * u_re * u_im)
                + a_soft_sq * soft_sq_unit;
            let du_re = af4 * r * (2.0 * u_re) + af5 * r * (2.0 * u_im);
            let du_im = af4 * r * (-2.0 * u_im) + af5 * r * (2.0 * u_re);
            let ds1 = if (2.0 * s1).abs() < 1.0 {
                a_soft_sq * r * (1.5 - 1.5 * u_sq * u_sq) * 2.0
            } else {
                0.0
            };
            let du_re = du_re + ds1 * sheet * sgn_im * (-0.25 / (0.5 * (1.0 - u_re)).sqrt());

            let dr_dti = u_re;
            let dr_dtq = u_im;
            let dure_dti = (1.0 - u_re * u_re) / r;
            let dure_dtq = -(u_re * u_im) / r;
            let duim_dti = -(u_re * u_im) / r;
            let duim_dtq = (1.0 - u_im * u_im) / r;

            l_ti += d_r * dr_dti + du_re * dure_dti + du_im * duim_dti;
            l_tq += d_r * dr_dtq + du_re * dure_dtq + du_im * duim_dtq;

            // sub_env & env_dc pullbacks
            let senv_prev = if t > 0 { hist_sub_env[t - 1] } else { 0.0 };
            let si_prev = if t > 0 { hist_sub_i[t - 1] } else { 0.0 };
            let env_prev = if t > 0 { hist_env[t - 1] } else { 0.0 };
            let ti_prev = if t > 0 { hist_trk_i[t - 1] } else { 0.0 };
            let tq_prev = if t > 0 { hist_trk_q[t - 1] } else { 0.0 };
            let sd_prev = if t > 0 { hist_sdiff[t - 1] } else { 0.0 };

            let _ = senv_prev;
            let _ = env_prev;
            l_si += d_abs_si * si.signum() + l_senv * (1.0 - pole_env) * si.signum();
            l_ti += d_abs_ti * ti.signum() + l_env * (1.0 - pole_env) * ti.signum();

            let i_mid = 0.5 * (il[t] + ir[t]);
            let q_mid = 0.5 * (ql[t] + qr[t]);
            let i_diff = 0.5 * (il[t] - ir[t]);

            adj_ptrk +=
                l_ti * (ti_prev - i_mid) + l_tq * (tq_prev - q_mid) + l_sdiff * (sd_prev - i_diff);
            adj_psub += l_si * (si_prev - i_mid);

            let a_imid = l_ti * (1.0 - pole_trk) + l_si * (1.0 - pole_sub);
            let a_qmid = l_tq * (1.0 - pole_trk);
            let a_idiff = l_sdiff * (1.0 - pole_trk);

            adj_il[t] = 0.5 * (a_imid + a_idiff);
            adj_ir[t] = 0.5 * (a_imid - a_idiff);
            adj_ql[t] = 0.5 * a_qmid;
            adj_qr[t] = 0.5 * a_qmid;

            l_ti *= pole_trk;
            l_tq *= pole_trk;
            l_sdiff *= pole_trk;
            l_si *= pole_sub;
            l_env *= pole_env;
            l_senv *= pole_env;
        }

        T1PlusBpttOut {
            loss,
            adj_il,
            adj_ql,
            adj_ir,
            adj_qr,
            adj_ptrk,
            adj_psub,
            adj_aoct,
            adj_boc2,
            adj_smix,
            adj_sbmix,
        }
    }

    #[test]
    fn test_t1_plus_layout_doubler_octavia_soft_square_and_mono_idle() {
        assert_eq!(core::mem::size_of::<T1PlusKernel>(), 56);
        assert_eq!(core::mem::align_of::<T1PlusKernel>(), 4);
        assert_eq!(core::mem::size_of::<T1PlusParams>(), 28);
        assert_eq!(core::mem::align_of::<T1PlusParams>(), 4);

        let params = T1PlusParams::new_96k_default();
        let mut st = T1PlusKernel::new();

        // Zero-DC & non-finite safety
        for &dirty in &[0.0_f32, -0.0, f32::from_bits(1), f32::NAN, f32::INFINITY] {
            let f = st.step_stereo(dirty, dirty, dirty, dirty, &params);
            assert_eq!(f.map(f32::to_bits), [0_u32; 8]);
        }

        // 0-ULP mono idle on feat[7] when sub_band_mix == 0.0 and L == R
        st.reset();
        let mut mono_p = params;
        mono_p.pole_trk = 0.0; // instant tracking for analytical 2*theta verification
        mono_p.beta_oc2 = 1.0;
        for k in 0..64 {
            let theta = 0.25 * (k as f64) + 0.1;
            let amp = 0.8_f64;
            let i_v = (amp * theta.cos()) as f32;
            let q_v = (amp * theta.sin()) as f32;
            let f = st.step_stereo(i_v, q_v, i_v, q_v, &mono_p);
            assert_eq!(f[7].to_bits(), 0, "mono feat[7] must be +0.0");

            // 2theta doubler accuracy on feat[4..6]
            let exp_c2 = amp * (2.0 * theta).cos();
            let exp_s2 = amp * (2.0 * theta).sin();
            assert!((f64::from(f[4]) - exp_c2).abs() < 1.0e-5);
            assert!((f64::from(f[5]) - exp_s2).abs() < 1.0e-5);
        }

        // Octavia DC cancellation: on constant |x_trk|, |x_trk| - env_dc decays toward 0
        st.reset();
        let oct_p = T1PlusParams {
            pole_trk: 0.0,
            pole_env_dc: 0.9,
            alpha_oct: 1.0,
            beta_oc2: 0.0,
            ..params
        };
        let mut last_f6 = 1.0_f32;
        for _ in 0..256 {
            last_f6 = st.step_stereo(0.5, 0.0, 0.5, 0.0, &oct_p)[6];
        }
        assert!(last_f6.abs() < 1.0e-5, "Octavia DC residual={last_f6}");

        // soft_square C1 check at boundary sub1_sin = ±0.5 (2*sub1_sin = ±1.0)
        let s_m = eval_t1_plus_soft_square_f32(0.5 - 1.0e-4);
        let s_0 = eval_t1_plus_soft_square_f32(0.5);
        let s_p = eval_t1_plus_soft_square_f32(0.5 + 1.0e-4);
        assert_eq!(s_0, 1.0);
        assert_eq!(s_p, 1.0);
        assert!((s_0 - s_m).abs() < 1.0e-6);

        // +-1e9 finite safety
        let ext = st.step_stereo(1.0e9, -1.0e9, -1.0e9, 1.0e9, &params);
        assert!(ext.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_t1_plus_bptt_gradients_match_f64_central_fd() {
        let il = [0.6_f64, 0.5, 0.4, 0.7];
        let ql = [0.3_f64, 0.4, 0.5, 0.2];
        let ir = [0.4_f64, 0.3, 0.5, 0.6];
        let qr = [0.2_f64, 0.3, 0.4, 0.3];
        let base = T1PlusBpttInput {
            il: &il,
            ql: &ql,
            ir: &ir,
            qr: &qr,
            pole_trk: 0.85,
            pole_env: 0.92,
            pole_sub: 0.75,
            alpha_oct: 1.2,
            beta_oc2: -0.8,
            stereo_mix: 0.6,
            sub_band_mix: 0.9,
            sheet: 1.0,
        };
        let out = t1_plus_bptt_loss_f64(&base);

        let eps = 1.0e-7_f64;
        for t in 0..4 {
            let mut p = il;
            let mut m = il;
            p[t] += eps;
            m[t] -= eps;
            let fd = (t1_plus_bptt_loss_f64(&T1PlusBpttInput { il: &p, ..base }).loss
                - t1_plus_bptt_loss_f64(&T1PlusBpttInput { il: &m, ..base }).loss)
                / (2.0 * eps);
            assert!((out.adj_il[t] - fd).abs() < 1.0e-6);

            let mut p = ql;
            let mut m = ql;
            p[t] += eps;
            m[t] -= eps;
            let fd = (t1_plus_bptt_loss_f64(&T1PlusBpttInput { ql: &p, ..base }).loss
                - t1_plus_bptt_loss_f64(&T1PlusBpttInput { ql: &m, ..base }).loss)
                / (2.0 * eps);
            assert!((out.adj_ql[t] - fd).abs() < 1.0e-6);

            let mut p = ir;
            let mut m = ir;
            p[t] += eps;
            m[t] -= eps;
            let fd = (t1_plus_bptt_loss_f64(&T1PlusBpttInput { ir: &p, ..base }).loss
                - t1_plus_bptt_loss_f64(&T1PlusBpttInput { ir: &m, ..base }).loss)
                / (2.0 * eps);
            assert!((out.adj_ir[t] - fd).abs() < 1.0e-6);

            let mut p = qr;
            let mut m = qr;
            p[t] += eps;
            m[t] -= eps;
            let fd = (t1_plus_bptt_loss_f64(&T1PlusBpttInput { qr: &p, ..base }).loss
                - t1_plus_bptt_loss_f64(&T1PlusBpttInput { qr: &m, ..base }).loss)
                / (2.0 * eps);
            assert!((out.adj_qr[t] - fd).abs() < 1.0e-6);
        }

        let fd_ptrk = (t1_plus_bptt_loss_f64(&T1PlusBpttInput {
            pole_trk: base.pole_trk + eps,
            ..base
        })
        .loss
            - t1_plus_bptt_loss_f64(&T1PlusBpttInput {
                pole_trk: base.pole_trk - eps,
                ..base
            })
            .loss)
            / (2.0 * eps);
        let fd_psub = (t1_plus_bptt_loss_f64(&T1PlusBpttInput {
            pole_sub: base.pole_sub + eps,
            ..base
        })
        .loss
            - t1_plus_bptt_loss_f64(&T1PlusBpttInput {
                pole_sub: base.pole_sub - eps,
                ..base
            })
            .loss)
            / (2.0 * eps);
        let fd_aoct = (t1_plus_bptt_loss_f64(&T1PlusBpttInput {
            alpha_oct: base.alpha_oct + eps,
            ..base
        })
        .loss
            - t1_plus_bptt_loss_f64(&T1PlusBpttInput {
                alpha_oct: base.alpha_oct - eps,
                ..base
            })
            .loss)
            / (2.0 * eps);
        let fd_boc2 = (t1_plus_bptt_loss_f64(&T1PlusBpttInput {
            beta_oc2: base.beta_oc2 + eps,
            ..base
        })
        .loss
            - t1_plus_bptt_loss_f64(&T1PlusBpttInput {
                beta_oc2: base.beta_oc2 - eps,
                ..base
            })
            .loss)
            / (2.0 * eps);
        let fd_smix = (t1_plus_bptt_loss_f64(&T1PlusBpttInput {
            stereo_mix: base.stereo_mix + eps,
            ..base
        })
        .loss
            - t1_plus_bptt_loss_f64(&T1PlusBpttInput {
                stereo_mix: base.stereo_mix - eps,
                ..base
            })
            .loss)
            / (2.0 * eps);
        let fd_sbmix = (t1_plus_bptt_loss_f64(&T1PlusBpttInput {
            sub_band_mix: base.sub_band_mix + eps,
            ..base
        })
        .loss
            - t1_plus_bptt_loss_f64(&T1PlusBpttInput {
                sub_band_mix: base.sub_band_mix - eps,
                ..base
            })
            .loss)
            / (2.0 * eps);
        assert!((out.adj_ptrk - fd_ptrk).abs() < 1.0e-6);
        assert!((out.adj_psub - fd_psub).abs() < 1.0e-6);
        assert!((out.adj_aoct - fd_aoct).abs() < 1.0e-6);
        assert!((out.adj_boc2 - fd_boc2).abs() < 1.0e-6);
        assert!((out.adj_smix - fd_smix).abs() < 1.0e-6);
        assert!((out.adj_sbmix - fd_sbmix).abs() < 1.0e-6);
    }
}
