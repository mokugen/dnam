#![forbid(unsafe_code)]
//! `8192`-FFT minimum-phase real-cepstrum filter design and log-frequency weighted Stage III
//! (`FIR` + `Kautz`) spectral equalization solver (`incoherent_rig/cepstral_eq.rs`).

use std::f64::consts::{LN_10, PI, TAU};

use dnam_engine::stages::stage2_gru::block::MAX_FIR_SEGMENTS;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, tier_kautz_banks};
use dnam_engine::stages::stage4_tank::v3_fdn::V3Stage4TankParams;
use dnam_engine::stages::{V3HotCore, V3MatryoshkaModel, V3TierReadout};
use dnam_engine::{FtzDazGuard, Tier};

use crate::metrics::companion::canonical_ltas_33_centers_hz;
use crate::solver::backends::ScalarTrainerBackend;
use crate::stimulus::fft::{Complex64, fft_radix2_in_place};

use super::s2_modal::{S2_MODAL_COLS, build_1x_linear_resonator_cal};

const CEPSTRUM_N_FFT: usize = 8_192;
const NUM_EQ_SAMPLES: usize = 256;
const ONSET_DELAY_48K: f64 = 16.0;

pub(super) struct EqFreqGrid {
    pub(super) bins: Vec<usize>,
    pub(super) omega: Vec<f64>,
    pub(super) weights: Vec<f64>,
}

pub(super) struct Stage3EqSolution {
    pub(super) fir: [f32; 64],
    pub(super) kautz_c: [(f32, f32); MAX_KAUTZ_POLES],
    pub(super) direct_s2: [f64; S2_MODAL_COLS],
}

pub(super) fn build_eq_freq_grid(sr_hz: f64) -> EqFreqGrid {
    let bin_20k = ((20_000.0 * (CEPSTRUM_N_FFT as f64) / sr_hz).floor() as usize)
        .clamp(32, CEPSTRUM_N_FFT / 2 - 1);
    let min_bin = 3_usize.min(bin_20k);
    let mut bins = Vec::with_capacity(NUM_EQ_SAMPLES + 25);
    bins.push(0_usize);
    for i in 0..NUM_EQ_SAMPLES {
        let frac = (i as f64) / ((NUM_EQ_SAMPLES - 1) as f64);
        let b_f = (min_bin as f64) * ((bin_20k as f64) / (min_bin as f64)).powf(frac);
        let b = (b_f.round() as usize).clamp(min_bin, bin_20k);
        if bins.last().copied() != Some(b) {
            bins.push(b);
        }
    }
    let half = CEPSTRUM_N_FFT / 2;
    if bin_20k < half {
        const NYQ_GUARD_PTS: usize = 24;
        for j in 1..=NYQ_GUARD_PTS {
            let b = bin_20k + ((half - bin_20k) * j) / NYQ_GUARD_PTS;
            if bins.last().copied() != Some(b) {
                bins.push(b);
            }
        }
    }
    let m = bins.len();
    let omega = bins
        .iter()
        .map(|&k| TAU * (k as f64) / (CEPSTRUM_N_FFT as f64))
        .collect();
    // Exact log-frequency trapezoidal weights so equal octave spans contribute equally at LF & HF.
    let mut weights = Vec::with_capacity(m);
    for p in 0..m {
        let d_ln = if bins[p] == 0 {
            (bins.get(1).copied().unwrap_or(1).max(1) as f64).ln()
        } else {
            let b_prev = if p == 0 || bins[p - 1] == 0 {
                (bins[p] as f64) * 0.85
            } else {
                bins[p - 1] as f64
            };
            let b_next = if p + 1 == m {
                (bins[m - 1] as f64) * 1.15
            } else {
                bins[p + 1] as f64
            };
            (b_next.ln() - b_prev.ln()).max(1.0e-3)
        };
        weights.push(d_ln);
    }
    let w_mean = (weights.iter().sum::<f64>() / (m.max(1) as f64)).max(1.0e-12);
    for w in &mut weights {
        *w /= w_mean;
    }
    EqFreqGrid {
        bins,
        omega,
        weights,
    }
}

fn interpolate_ltas_db(centers: &[f64; 33], target_db: &[f64; 33], freq_hz: f64) -> f64 {
    if freq_hz <= centers[0] {
        return target_db[0];
    }
    if freq_hz >= centers[32] {
        return target_db[32];
    }
    let lf = freq_hz.ln();
    for b in 0..32 {
        let f0 = centers[b];
        let f1 = centers[b + 1];
        if freq_hz <= f1 {
            let t = (lf - f0.ln()) / (f1.ln() - f0.ln()).max(1.0e-12);
            return target_db[b] + t * (target_db[b + 1] - target_db[b]);
        }
    }
    target_db[32]
}

fn design_minphase_spectrum(target_db: &[f64; 33], sr_hz: f64) -> Vec<Complex64> {
    let centers = canonical_ltas_33_centers_hz();
    let half = CEPSTRUM_N_FFT / 2;
    let db_to_ln = LN_10 / 20.0;
    let df = sr_hz / (CEPSTRUM_N_FFT as f64);
    let mut log_mag = vec![Complex64::new(0.0, 0.0); CEPSTRUM_N_FFT];
    let nyq = (0.5 * sr_hz).max(20_001.0);
    let base_20k = interpolate_ltas_db(&centers, target_db, 20_000.0);
    for k in 0..=half {
        let f = (k as f64) * df;
        let db = if k == 0 {
            target_db[0].clamp(-24.0, 24.0)
        } else if f <= 20_000.0 {
            interpolate_ltas_db(&centers, target_db, f).clamp(-24.0, 24.0)
        } else {
            let t_hf = ((f - 20_000.0) / (nyq - 20_000.0)).clamp(0.0, 1.0);
            let roll_db = -9.0 * 0.5 * (1.0 - (PI * t_hf).cos());
            (base_20k + roll_db).clamp(-24.0, 24.0)
        };
        let ln_amp = db * db_to_ln;
        log_mag[k] = Complex64::new(ln_amp, 0.0);
        if k > 0 && k < half {
            log_mag[CEPSTRUM_N_FFT - k] = Complex64::new(ln_amp, 0.0);
        }
    }
    fft_radix2_in_place(&mut log_mag, true);

    let mut min_cep = vec![Complex64::new(0.0, 0.0); CEPSTRUM_N_FFT];
    min_cep[0] = log_mag[0];
    for k in 1..half {
        min_cep[k] = Complex64::new(2.0 * log_mag[k].re, 2.0 * log_mag[k].im);
    }
    min_cep[half] = log_mag[half];
    fft_radix2_in_place(&mut min_cep, false);

    for bin in &mut min_cep {
        let mag = bin.re.clamp(-6.0, 6.0).exp();
        let (s, c) = bin.im.sin_cos();
        *bin = Complex64::new(mag * c, mag * s);
    }
    min_cep
}

fn factor_spd_cholesky(n: usize, a: &mut [f64]) -> bool {
    for j in 0..n {
        let mut sum = a[j * n + j];
        for k in 0..j {
            sum -= a[j * n + k] * a[j * n + k];
        }
        if !sum.is_finite() || sum <= 1.0e-12 {
            return false;
        }
        let diag = sum.sqrt();
        a[j * n + j] = diag;
        let inv = 1.0 / diag;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s * inv;
        }
    }
    true
}

fn solve_spd_from_factor(n: usize, chol_l: &[f64], b: &mut [f64]) -> Option<Vec<f64>> {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= chol_l[i * n + k] * b[k];
        }
        b[i] = s / chol_l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= chol_l[k * n + i] * b[k];
        }
        b[i] = s / chol_l[i * n + i];
        if !b[i].is_finite() {
            return None;
        }
    }
    Some(b[..n].to_vec())
}

fn compute_s2_modal_freq_cols(sr_hz: f64, grid: &EqFreqGrid) -> (Vec<f64>, Vec<f64>) {
    const IMP_LEN: usize = 2_048;
    let m_pts = grid.bins.len();
    let mut probe = V3MatryoshkaModel::identity(false, true);
    probe.tone_svf.sample_rate_hz = sr_hz as f32;
    let cal = build_1x_linear_resonator_cal(sr_hz);
    probe.svf = cal.svf;
    probe.feat_params = cal.feat_params;
    probe.blocks = cal.blocks;
    probe.rate_1x_cal = Some(cal);

    let mut cold = probe.extract_tier(Tier::Lite);
    cold.readout = V3TierReadout::zeros();
    cold.readout.pre_cab_u_gain = [1.0, 1.0];
    cold.readout.fir_l[0] = 1.0;
    cold.stage4 = V3Stage4TankParams {
        wet_mix: 0.0,
        dry_mix: 1.0,
        g_loop: 0.0,
        base_psi: 0.0,
        vca_kappa: 0.0,
        vca_min_gain: 1.0,
        ..V3Stage4TankParams::identity()
    };

    let mut u_fft = vec![Complex64::new(0.0, 0.0); CEPSTRUM_N_FFT];
    let mut h_fft: Vec<Vec<Complex64>> = (0..S2_MODAL_COLS)
        .map(|_| vec![Complex64::new(0.0, 0.0); CEPSTRUM_N_FFT])
        .collect();
    {
        let _ftz = FtzDazGuard::new();
        let mut core = V3HotCore::<ScalarTrainerBackend>::from_cold_model(&cold);
        core.reset_to_dc_equilibrium();
        for i in 0..IMP_LEN {
            let x = if i == 0 { 1.0e-2_f32 } else { 0.0 };
            let [out_l, _] = core.step_stereo(x, x);
            u_fft[i] = Complex64::new(f64::from(out_l), 0.0);
            for b in 0..3 {
                for l in 0..4 {
                    h_fft[b * 4 + l][i] = Complex64::new(f64::from(core.state_l.h[b][0][l]), 0.0);
                }
            }
        }
    }
    fft_radix2_in_place(&mut u_fft, false);
    for hm in &mut h_fft {
        fft_radix2_in_place(hm, false);
    }

    let mut s2_re = vec![0.0_f64; S2_MODAL_COLS * m_pts];
    let mut s2_im = vec![0.0_f64; S2_MODAL_COLS * m_pts];
    for (p, &k) in grid.bins.iter().enumerate() {
        let u_k = u_fft[k];
        let denom = u_k.norm_sqr().max(1.0e-24);
        for m in 0..S2_MODAL_COLS {
            let h_k = h_fft[m][k];
            let idx = m * m_pts + p;
            s2_re[idx] = (h_k.re * u_k.re + h_k.im * u_k.im) / denom;
            s2_im[idx] = (h_k.im * u_k.re - h_k.re * u_k.im) / denom;
        }
    }
    (s2_re, s2_im)
}

/// Pre-factorized Stage III (+ optional `s2_modal`) cepstral EQ basis and Cholesky factor.
pub(super) struct PrecomputedTierEq {
    dim: usize,
    fir_len: usize,
    k_cnt: usize,
    s2_cnt: usize,
    m_pts: usize,
    col_re: Vec<f64>,
    col_im: Vec<f64>,
    chol_l: Vec<f64>,
}

impl PrecomputedTierEq {
    pub(super) fn new(
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
        tier: Tier,
        use_s2_modal: bool,
        sr_hz: f64,
        grid: &EqFreqGrid,
    ) -> Option<Self> {
        let fir_len = ((tier as usize) + 1) * 16;
        let k_cnt = tier_kautz_banks(tier) * 4;
        let s2_cnt = if use_s2_modal { S2_MODAL_COLS } else { 0 };
        let dim = fir_len + 2 * k_cnt + s2_cnt;
        let m_pts = grid.bins.len();

        let mut col_re = vec![0.0_f64; dim * m_pts];
        let mut col_im = vec![0.0_f64; dim * m_pts];
        for (p, &w) in grid.omega.iter().enumerate() {
            let (s_on, c_on) = (ONSET_DELAY_48K * w).sin_cos();
            let e_on = Complex64::new(c_on, -s_on);
            let (sw, cw) = w.sin_cos();
            let e_mw = Complex64::new(cw, -sw);
            for d in 0..fir_len {
                let (sd, cd) = ((d as f64) * w).sin_cos();
                let idx = d * m_pts + p;
                col_re[idx] = cd;
                col_im[idx] = -sd;
            }
            for (m, &(r_raw, th_raw)) in kautz_poles.iter().enumerate().take(k_cnt) {
                let r = f64::from(r_raw.clamp(0.25, 0.9995));
                let th = f64::from(th_raw.clamp(1.0e-4, (PI as f32) - 1.0e-4));
                let bg = (1.0 - r * r).max(1.0e-7).sqrt();
                let (sth, cth) = th.sin_cos();
                let lam = Complex64::new(r * cth, r * sth);
                let lam_c = Complex64::new(r * cth, -r * sth);

                let d1 = Complex64::new(
                    1.0 - (lam.re * e_mw.re - lam.im * e_mw.im),
                    -(lam.re * e_mw.im + lam.im * e_mw.re),
                );
                let d2 = Complex64::new(
                    1.0 - (lam_c.re * e_mw.re - lam_c.im * e_mw.im),
                    -(lam_c.re * e_mw.im + lam_c.im * e_mw.re),
                );
                let inv1 = Complex64::new(d1.re / d1.norm_sqr(), -d1.im / d1.norm_sqr());
                let inv2 = Complex64::new(d2.re / d2.norm_sqr(), -d2.im / d2.norm_sqr());
                let sum_re = Complex64::new(
                    0.5 * bg * (inv1.re + inv2.re),
                    0.5 * bg * (inv1.im + inv2.im),
                );
                let diff = Complex64::new(inv1.re - inv2.re, inv1.im - inv2.im);
                let sum_im = Complex64::new(-0.5 * bg * diff.im, 0.5 * bg * diff.re);

                let phi_re = Complex64::new(
                    sum_re.re * e_on.re - sum_re.im * e_on.im,
                    sum_re.re * e_on.im + sum_re.im * e_on.re,
                );
                let phi_im = Complex64::new(
                    sum_im.re * e_on.re - sum_im.im * e_on.im,
                    sum_im.re * e_on.im + sum_im.im * e_on.re,
                );
                let c_idx_r = (fir_len + 2 * m) * m_pts + p;
                let c_idx_i = (fir_len + 2 * m + 1) * m_pts + p;
                col_re[c_idx_r] = phi_re.re;
                col_im[c_idx_r] = phi_re.im;
                col_re[c_idx_i] = phi_im.re;
                col_im[c_idx_i] = phi_im.im;
            }
        }
        if use_s2_modal {
            let (s2_re, s2_im) = compute_s2_modal_freq_cols(sr_hz, grid);
            let s2_off = (fir_len + 2 * k_cnt) * m_pts;
            col_re[s2_off..].copy_from_slice(&s2_re);
            col_im[s2_off..].copy_from_slice(&s2_im);
        }

        let mut ata = vec![0.0_f64; dim * dim];
        for p in 0..m_pts {
            let w_p = grid.weights[p];
            for r in 0..dim {
                let rr = col_re[r * m_pts + p];
                let ri = col_im[r * m_pts + p];
                for c in 0..=r {
                    let cr = col_re[c * m_pts + p];
                    let ci = col_im[c * m_pts + p];
                    ata[r * dim + c] += w_p * (rr * cr + ri * ci);
                }
            }
        }
        let scale = (m_pts as f64).max(1.0);
        let s2_start = fir_len + 2 * k_cnt;
        for r in 0..dim {
            let ridge = if r < fir_len {
                1.0e-5
            } else if r < s2_start {
                2.0e-5
            } else {
                1.0e-4
            } * scale;
            ata[r * dim + r] += ridge;
        }
        if !factor_spd_cholesky(dim, &mut ata) {
            return None;
        }
        Some(Self {
            dim,
            fir_len,
            k_cnt,
            s2_cnt,
            m_pts,
            col_re,
            col_im,
            chol_l: ata,
        })
    }

    pub(super) fn solve(
        &self,
        target_db: &[f64; 33],
        sr_hz: f64,
        grid: &EqFreqGrid,
    ) -> Option<Stage3EqSolution> {
        let spec = design_minphase_spectrum(target_db, sr_hz);
        let mut atb = vec![0.0_f64; self.dim];
        for p in 0..self.m_pts {
            let w_p = grid.weights[p];
            let tgt = spec[grid.bins[p]];
            for (r, rhs_r) in atb.iter_mut().enumerate() {
                let rr = self.col_re[r * self.m_pts + p];
                let ri = self.col_im[r * self.m_pts + p];
                *rhs_r += w_p * (rr * tgt.re + ri * tgt.im);
            }
        }
        let sol = solve_spd_from_factor(self.dim, &self.chol_l, &mut atb)?;
        let mut fir = [0.0_f32; 64];
        for d in 0..self.fir_len.min(MAX_FIR_SEGMENTS * 16) {
            fir[d] = (sol[d] as f32).clamp(-64.0, 64.0);
        }
        let mut kautz_c = [(0.0_f32, 0.0_f32); MAX_KAUTZ_POLES];
        for m in 0..self.k_cnt {
            kautz_c[m] = (
                (sol[self.fir_len + 2 * m] as f32).clamp(-16.0, 16.0),
                (sol[self.fir_len + 2 * m + 1] as f32).clamp(-16.0, 16.0),
            );
        }
        let mut direct_s2 = [0.0_f64; S2_MODAL_COLS];
        if self.s2_cnt > 0 {
            let s2_off = self.fir_len + 2 * self.k_cnt;
            direct_s2.copy_from_slice(&sol[s2_off..s2_off + S2_MODAL_COLS]);
            let sq: f64 = direct_s2.iter().map(|&w| w * w).sum();
            if sq > 40.0 {
                let scale = (40.0 / sq).sqrt();
                for w in &mut direct_s2 {
                    *w *= scale;
                }
            }
        }
        Some(Stage3EqSolution {
            fir,
            kautz_c,
            direct_s2,
        })
    }
}

#[cfg(test)]
pub(super) fn fit_stage3_freq_eq(
    kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    tier: Tier,
    use_s2_modal: bool,
    target_db: &[f64; 33],
    sr_hz: f64,
    grid: &EqFreqGrid,
) -> Option<Stage3EqSolution> {
    PrecomputedTierEq::new(kautz_poles, tier, use_s2_modal, sr_hz, grid)?
        .solve(target_db, sr_hz, grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::v3_varpro::stage4_id::incoherent_rig::comb_detect::apply_lf_kautz_poles;
    use dnam_engine::stages::V3MatryoshkaModel;

    fn eval_cepstral_max_band_err_db(
        eq: &PrecomputedTierEq,
        sol: &Stage3EqSolution,
        target_db: &[f64; 33],
        sr_hz: f64,
        grid: &EqFreqGrid,
    ) -> f64 {
        let spec = design_minphase_spectrum(target_db, sr_hz);
        let mut weights = vec![0.0_f64; eq.dim];
        for d in 0..eq.fir_len {
            weights[d] = f64::from(sol.fir[d]);
        }
        for m in 0..eq.k_cnt {
            weights[eq.fir_len + 2 * m] = f64::from(sol.kautz_c[m].0);
            weights[eq.fir_len + 2 * m + 1] = f64::from(sol.kautz_c[m].1);
        }
        if eq.s2_cnt > 0 {
            let s2_off = eq.fir_len + 2 * eq.k_cnt;
            weights[s2_off..s2_off + S2_MODAL_COLS].copy_from_slice(&sol.direct_s2);
        }
        let mut max_err = 0.0_f64;
        for (p, &bin) in grid.bins.iter().enumerate() {
            let freq = (bin as f64) * sr_hz / (CEPSTRUM_N_FFT as f64);
            if !(30.0..=18_000.0).contains(&freq) {
                continue;
            }
            let (mut hr, mut hi) = (0.0_f64, 0.0_f64);
            for (r, &wr) in weights.iter().enumerate() {
                hr += wr * eq.col_re[r * eq.m_pts + p];
                hi += wr * eq.col_im[r * eq.m_pts + p];
            }
            let mag_hat = (hr * hr + hi * hi).sqrt().max(1.0e-12);
            let mag_tgt = spec[bin].norm_sqr().sqrt().max(1.0e-12);
            let err_db = (20.0 * (mag_hat / mag_tgt).log10()).abs();
            max_err = max_err.max(err_db);
        }
        max_err
    }

    #[test]
    fn test_precomputed_cepstral_eq_matches_one_shot_across_tilts() {
        let sr = 48_000.0;
        let grid = build_eq_freq_grid(sr);
        let model = V3MatryoshkaModel::dual_rate_default(true);
        let pre = PrecomputedTierEq::new(&model.kautz_poles, Tier::Full, false, sr, &grid).unwrap();
        let flat = [0.0_f64; 33];
        let tilt = std::array::from_fn(|b| -0.15 * (b as f64));
        let sol_flat_a = pre.solve(&flat, sr, &grid).unwrap();
        let sol_flat_b =
            fit_stage3_freq_eq(&model.kautz_poles, Tier::Full, false, &flat, sr, &grid).unwrap();
        assert_eq!(sol_flat_a.fir, sol_flat_b.fir);
        assert_eq!(sol_flat_a.kautz_c, sol_flat_b.kautz_c);
        assert_eq!(sol_flat_a.direct_s2, [0.0_f64; S2_MODAL_COLS]);
        let sol_tilt = pre.solve(&tilt, sr, &grid).unwrap();
        assert_ne!(sol_tilt.fir[0..16], sol_flat_a.fir[0..16]);
    }

    #[test]
    fn test_precomputed_cepstral_eq_s2_modal_closes_midrange_hole_at_lite_and_full() {
        let sr = 48_000.0;
        let grid = build_eq_freq_grid(sr);
        let mut model = V3MatryoshkaModel::identity(false, true);
        apply_lf_kautz_poles(&mut model, sr);
        let centers = canonical_ltas_33_centers_hz();
        let resonant_db: [f64; 33] = std::array::from_fn(|b| {
            let oct = (centers[b] / 720.0).log2();
            4.5 * (-oct * oct / 0.35).exp() - 2.0 * ((centers[b] / 2_100.0).log2()).tanh()
        });

        for tier in [Tier::Lite, Tier::Full] {
            let eq_no_s2 =
                PrecomputedTierEq::new(&model.kautz_poles, tier, false, sr, &grid).unwrap();
            let sol_no_s2 = eq_no_s2.solve(&resonant_db, sr, &grid).unwrap();
            let err_no_s2 =
                eval_cepstral_max_band_err_db(&eq_no_s2, &sol_no_s2, &resonant_db, sr, &grid);

            let eq_with_s2 =
                PrecomputedTierEq::new(&model.kautz_poles, tier, true, sr, &grid).unwrap();
            let sol_with_s2 = eq_with_s2.solve(&resonant_db, sr, &grid).unwrap();
            let err_with_s2 =
                eval_cepstral_max_band_err_db(&eq_with_s2, &sol_with_s2, &resonant_db, sr, &grid);

            let s2_sq: f64 = sol_with_s2.direct_s2.iter().map(|w| w * w).sum();
            assert!(
                s2_sq > 0.01 && s2_sq <= 40.000_001,
                "tier={tier:?} s2_sq={s2_sq}"
            );
            assert!(
                err_with_s2 < err_no_s2 * 0.5 && err_with_s2 < 0.75,
                "tier={tier:?}: expected s2_modal to cut error (<0.75 dB), got {err_with_s2:.3} vs no_s2 {err_no_s2:.3}"
            );
        }
    }
}
