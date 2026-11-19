#![forbid(unsafe_code)]
//! Exact 4-Band Partition-of-Unity TPT Filter Bank (`V3PostStageConditioner`) & Per-Band RMS
//! Gauss-Newton Residual/Jacobian (`solver::stage_c_knobs::jacobian::four_band`).

use dnam_engine::stages::hot_core::V3PostStageConditioner;

use crate::solver::uplo_u_index;

use super::StageTargetJacobian;

/// Cytomic mid-band SVF damping matching `V3PostStageConditioner` (`post_cond.rs`).
const POST_MID_K_DAMP: f64 = 1.25;
/// Minimum fraction of total wet energy (`5% ≈ -13 dB`) for a band to be checked for Wiener collapse.
const ACTIVE_BAND_MIN_ENERGY_FRAC: f64 = 0.05;
/// Maximum band waveform correlation (`0.65`) indicating phase rotation / decorrelation in that band.
const BAND_PHASE_DECORR_RHO_MAX: f64 = 0.65;

const _: () = {
    let pc = V3PostStageConditioner::new_48k();
    let k_from_coeff = (1.0_f64 - (pc.a1 as f64) - (pc.a3 as f64)) / (pc.a2 as f64);
    assert!((k_from_coeff - POST_MID_K_DAMP).abs() < 1.0e-5);
    assert!(
        ACTIVE_BAND_MIN_ENERGY_FRAC > 0.0
            && ACTIVE_BAND_MIN_ENERGY_FRAC < 0.25
            && BAND_PHASE_DECORR_RHO_MAX > 0.0
            && BAND_PHASE_DECORR_RHO_MAX < 1.0
    );
};

fn split_4band_iter(samples: impl Iterator<Item = f64>, cap: usize) -> [Vec<f64>; 4] {
    let pc = V3PostStageConditioner::new_48k();
    let (a_sub, a1, a2, a3) = (
        f64::from(pc.a_sub),
        f64::from(pc.a1),
        f64::from(pc.a2),
        f64::from(pc.a3),
    );
    let (mut sub_z, mut ic1, mut ic2) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut bands = [
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    ];
    for x in samples {
        let v_sub = a_sub * (x - sub_z);
        let lp_sub = v_sub + sub_z;
        sub_z = lp_sub + v_sub;

        let v3 = x - ic2;
        let v1 = a1 * ic1 + a2 * v3;
        let v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2.0 * v1 - ic1;
        ic2 = 2.0 * v2 - ic2;

        let lm = v2 - lp_sub;
        let pres = POST_MID_K_DAMP * v1;
        let hp = x - pres - v2;
        bands[0].push(lp_sub);
        bands[1].push(lm);
        bands[2].push(pres);
        bands[3].push(hp);
    }
    bands
}

/// Splits a real `f32` signal into `V3PostStageConditioner`'s 4 partition-of-unity bands.
pub(super) fn split_4band_f32(sig: &[f32]) -> [Vec<f64>; 4] {
    split_4band_iter(sig.iter().map(|&v| f64::from(v)), sig.len())
}

/// Splits a real `f64` signal into `V3PostStageConditioner`'s 4 partition-of-unity bands.
pub(super) fn split_4band_f64(sig: &[f64]) -> [Vec<f64>; 4] {
    split_4band_iter(sig.iter().copied(), sig.len())
}

fn empty_4band() -> [Vec<f64>; 4] {
    [Vec::new(), Vec::new(), Vec::new(), Vec::new()]
}

fn split_stereo_bands_f32(
    sig_l: &[f32],
    sig_r: &[f32],
    n: usize,
    mono: bool,
) -> ([Vec<f64>; 4], [Vec<f64>; 4]) {
    let bl = split_4band_f32(&sig_l[..n]);
    let br = if mono {
        empty_4band()
    } else {
        split_4band_f32(&sig_r[..n])
    };
    (bl, br)
}

fn band_amps(
    bl: &[Vec<f64>; 4],
    br: &[Vec<f64>; 4],
    warmup: usize,
    n: usize,
    mono: bool,
) -> [f64; 4] {
    std::array::from_fn(|b| {
        let mut e = 0.0_f64;
        for t in warmup..n {
            let vl = bl[b][t];
            e += vl * vl;
            if !mono {
                let vr = br[b][t];
                e += vr * vr;
            }
        }
        e.sqrt()
    })
}

fn compute_band_jac_proj(
    best_bands: &[Vec<f64>; 4],
    jac_bands: &[Vec<f64>; 4],
    amps: &[f64; 4],
    warmup: usize,
    n: usize,
) -> [f64; 4] {
    let mut g = [0.0_f64; 4];
    for b in 0..4 {
        if amps[b] <= 1.0e-9 {
            continue;
        }
        let mut dot = 0.0_f64;
        for t in warmup..n {
            dot += best_bands[b][t] * jac_bands[b][t];
        }
        g[b] = dot / amps[b];
    }
    g
}

fn add_stereo_band_jac_proj(
    out: &mut [f64; 4],
    best_r: &[Vec<f64>; 4],
    jac_r: &[Vec<f64>; 4],
    amps: &[f64; 4],
    warmup: usize,
    n: usize,
) {
    for b in 0..4 {
        if amps[b] <= 1.0e-9 {
            continue;
        }
        let mut dot = 0.0_f64;
        for t in warmup..n {
            dot += best_r[b][t] * jac_r[b][t];
        }
        out[b] += dot / amps[b];
    }
}

/// Precomputed 4-band TPT projections of wet target and `Ultra`/`Lite` stage-target Jacobians.
pub(super) struct FourBandJacobianCtx {
    pub(super) wet_amp: [f64; 4],
    jac_u_l: [[Vec<f64>; 4]; 16],
    jac_u_r: [[Vec<f64>; 4]; 16],
    jac_l_l: [[Vec<f64>; 4]; 16],
    jac_l_r: [[Vec<f64>; 4]; 16],
    warmup: usize,
    n: usize,
    mono: bool,
}

impl FourBandJacobianCtx {
    /// Decomposes the target slice and active Jacobian columns into the 4 TPT frequency bands.
    pub(super) fn new(
        wet_slice: &[[f32; 2]],
        eff_jac: &StageTargetJacobian,
        warmup: usize,
        n: usize,
    ) -> Self {
        let mono = eff_jac.mono_out;
        let wl_in: Vec<f32> = wet_slice[..n].iter().map(|lr| lr[0]).collect();
        let wet_l = split_4band_f32(&wl_in);
        let wet_r = if mono {
            empty_4band()
        } else {
            let wr_in: Vec<f32> = wet_slice[..n].iter().map(|lr| lr[1]).collect();
            split_4band_f32(&wr_in)
        };
        let wet_amp = band_amps(&wet_l, &wet_r, warmup, n, mono);

        let split_col_bank = |cols_src: &[Vec<f64>], skip_bank: bool| -> [[Vec<f64>; 4]; 16] {
            std::array::from_fn(|k| {
                if !skip_bank && eff_jac.active_cols[k] {
                    split_4band_f64(&cols_src[k][..n])
                } else {
                    empty_4band()
                }
            })
        };

        let jac_u_l = split_col_bank(&eff_jac.jac_l, false);
        let jac_l_l = split_col_bank(&eff_jac.lite_jac_l, false);
        let jac_u_r = split_col_bank(&eff_jac.jac_r, mono);
        let jac_l_r = split_col_bank(&eff_jac.lite_jac_r, mono);

        Self {
            wet_amp,
            jac_u_l,
            jac_u_r,
            jac_l_l,
            jac_l_r,
            warmup,
            n,
            mono,
        }
    }

    /// Evaluates total per-band RMS discrepancy `sum_b (wet_amp[b] - amp_b)^2` for `(Ultra, Lite)`.
    pub(super) fn penalty(
        &self,
        ul: &[f32],
        ur: &[f32],
        ll: &[f32],
        lr: &[f32],
    ) -> (f64, f64) {
        let (bul, bur) = split_stereo_bands_f32(ul, ur, self.n, self.mono);
        let (bll, blr) = split_stereo_bands_f32(ll, lr, self.n, self.mono);
        let amp_u = band_amps(&bul, &bur, self.warmup, self.n, self.mono);
        let amp_l = band_amps(&bll, &blr, self.warmup, self.n, self.mono);
        let pen_u: f64 = (0..4).map(|b| (self.wet_amp[b] - amp_u[b]).powi(2)).sum();
        let pen_l: f64 = (0..4).map(|b| (self.wet_amp[b] - amp_l[b]).powi(2)).sum();
        (pen_u, pen_l)
    }

    /// Accumulates per-band negative gradient contributions into `grad_out` (`[Ultra, Lite]`)
    /// and rank-4 Gauss-Newton Hessian updates into `gram_step` (`uplo='U'`).
    pub(super) fn accumulate_grad_and_gram(
        &self,
        w_band: f64,
        best_u: [&[f32]; 2],
        best_l: [&[f32]; 2],
        active_cols: &[bool; 16],
        grad_out: &mut [[f64; 16]; 2],
        gram_step: &mut [f64],
    ) {
        let (bul, bur) = split_stereo_bands_f32(best_u[0], best_u[1], self.n, self.mono);
        let (bll, blr) = split_stereo_bands_f32(best_l[0], best_l[1], self.n, self.mono);
        let amp_u = band_amps(&bul, &bur, self.warmup, self.n, self.mono);
        let amp_l = band_amps(&bll, &blr, self.warmup, self.n, self.mono);
        let diff_u: [f64; 4] = std::array::from_fn(|b| self.wet_amp[b] - amp_u[b]);
        let diff_l: [f64; 4] = std::array::from_fn(|b| self.wet_amp[b] - amp_l[b]);

        let mut g_u = [[0.0_f64; 4]; 16];
        let mut g_l = [[0.0_f64; 4]; 16];
        for k in 0..16 {
            if !active_cols[k] {
                continue;
            }
            g_u[k] = compute_band_jac_proj(&bul, &self.jac_u_l[k], &amp_u, self.warmup, self.n);
            g_l[k] = compute_band_jac_proj(&bll, &self.jac_l_l[k], &amp_l, self.warmup, self.n);
            if !self.mono {
                add_stereo_band_jac_proj(
                    &mut g_u[k],
                    &bur,
                    &self.jac_u_r[k],
                    &amp_u,
                    self.warmup,
                    self.n,
                );
                add_stereo_band_jac_proj(
                    &mut g_l[k],
                    &blr,
                    &self.jac_l_r[k],
                    &amp_l,
                    self.warmup,
                    self.n,
                );
            }
            let (mut su, mut sl) = (0.0_f64, 0.0_f64);
            for b in 0..4 {
                su += diff_u[b] * g_u[k][b];
                sl += diff_l[b] * g_l[k][b];
            }
            grad_out[0][k] += w_band * su;
            grad_out[1][k] += w_band * sl;
        }

        for j in 0..16 {
            if !active_cols[j] {
                continue;
            }
            for i in 0..=j {
                if !active_cols[i] {
                    continue;
                }
                let mut h_ij = 0.0_f64;
                for b in 0..4 {
                    h_ij += g_u[i][b] * g_u[j][b] + g_l[i][b] * g_l[j][b];
                }
                gram_step[uplo_u_index(i, j)] += 0.5 * w_band * h_ij;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn band_cross_corr(
    a_l: &[Vec<f64>; 4],
    a_r: &[Vec<f64>; 4],
    b_l: &[Vec<f64>; 4],
    b_r: &[Vec<f64>; 4],
    band: usize,
    warmup: usize,
    n: usize,
    mono: bool,
    amp_a: f64,
    amp_b: f64,
) -> f64 {
    let mut dot = 0.0_f64;
    for t in warmup..n {
        dot += a_l[band][t] * b_l[band][t];
        if !mono {
            dot += a_r[band][t] * b_r[band][t];
        }
    }
    dot / (amp_a * amp_b + 1.0e-12)
}

/// Detects whether any active TPT band suffered phase-decorrelated Wiener collapse in Pass 1
/// (`rho_b < BAND_PHASE_DECORR_RHO_MAX` and per-band octave error `< oct_floor`) even when
/// higher-energy in-phase bands mask broadband `oct_err_p1`.
#[allow(clippy::too_many_arguments)]
pub(super) fn has_per_band_wiener_collapse(
    wet_slice: &[[f32; 2]],
    eff_jac: &StageTargetJacobian,
    ul: &[f32],
    ur: &[f32],
    ll: &[f32],
    lr: &[f32],
    oct_floor: f32,
) -> bool {
    let n = eff_jac
        .n_fit
        .min(wet_slice.len())
        .min(ul.len())
        .min(ll.len());
    let warmup = eff_jac.warmup.min(n);
    if n <= warmup + 8 {
        return false;
    }
    let mono = eff_jac.mono_out;
    let wl: Vec<f32> = wet_slice[..n].iter().map(|v| v[0]).collect();
    let wr: Vec<f32> = wet_slice[..n]
        .iter()
        .map(|v| if mono { v[0] } else { v[1] })
        .collect();
    let (wet_l, wet_r) = split_stereo_bands_f32(&wl, &wr, n, mono);
    let (sul, sur) =
        split_stereo_bands_f32(&eff_jac.star_out_l, &eff_jac.star_out_r, n, mono);
    let (sll, slr) =
        split_stereo_bands_f32(&eff_jac.lite_star_l, &eff_jac.lite_star_r, n, mono);
    let (mul, mur) = split_stereo_bands_f32(ul, ur, n, mono);
    let (mll, mlr) = split_stereo_bands_f32(ll, lr, n, mono);

    let wet_amp = band_amps(&wet_l, &wet_r, warmup, n, mono);
    let star_u_amp = band_amps(&sul, &sur, warmup, n, mono);
    let star_l_amp = band_amps(&sll, &slr, warmup, n, mono);
    let mdl_u_amp = band_amps(&mul, &mur, warmup, n, mono);
    let mdl_l_amp = band_amps(&mll, &mlr, warmup, n, mono);

    let total_wet_e: f64 = wet_amp.iter().map(|a| a * a).sum();
    if total_wet_e <= 1.0e-10 {
        return false;
    }
    for b in 0..4 {
        let wet_e = wet_amp[b] * wet_amp[b];
        if wet_e < ACTIVE_BAND_MIN_ENERGY_FRAC * total_wet_e {
            continue;
        }
        let rho_u = band_cross_corr(
            &wet_l,
            &wet_r,
            &sul,
            &sur,
            b,
            warmup,
            n,
            mono,
            wet_amp[b],
            star_u_amp[b],
        );
        let rho_l = band_cross_corr(
            &wet_l,
            &wet_r,
            &sll,
            &slr,
            b,
            warmup,
            n,
            mono,
            wet_amp[b],
            star_l_amp[b],
        );
        let rho_b = 0.5 * (rho_u + rho_l);
        let mdl_e = 0.5 * (mdl_u_amp[b] * mdl_u_amp[b] + mdl_l_amp[b] * mdl_l_amp[b]);
        let oct_err_b = (0.5 * ((mdl_e + 1.0e-12) / (wet_e + 1.0e-12)).log2()) as f32;
        if rho_b < BAND_PHASE_DECORR_RHO_MAX && oct_err_b < oct_floor {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use crate::solver::uplo_u_packed_len;

    use super::*;

    #[test]
    fn test_four_band_split_exact_partition_of_unity() {
        let n = 512_usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                (0.35 * (TAU * 90.0 * t).sin()
                    + 0.25 * (TAU * 700.0 * t).cos()
                    + 0.20 * (TAU * 2_400.0 * t).sin()
                    + 0.15 * (TAU * 7_500.0 * t).cos()) as f32
            })
            .collect();
        let bands = split_4band_f32(&x);
        let max_err = (0..n)
            .map(|i| {
                let sum = bands[0][i] + bands[1][i] + bands[2][i] + bands[3][i];
                (sum - f64::from(x[i])).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1.0e-12,
            "partition of unity violated: max_err={max_err}"
        );
    }

    #[test]
    fn test_four_band_penalty_gradient_and_gauss_newton_hessian_fd_parity() {
        let n = 256_usize;
        let warmup = 32_usize;
        let star_l: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                (0.22 * (TAU * 180.0 * t).sin() + 0.14 * (TAU * 2_100.0 * t).cos()) as f32
            })
            .collect();
        let wet_slice: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let v = (0.31 * (TAU * 180.0 * t).cos() + 0.24 * (TAU * 3_100.0 * t).sin()) as f32;
                [v, v]
            })
            .collect();
        let mut active_cols = [false; 16];
        active_cols[0] = true;
        active_cols[1] = true;
        let mut jac_l = vec![vec![0.0_f64; n]; 16];
        let mut lite_jac_l = vec![vec![0.0_f64; n]; 16];
        for i in warmup..n {
            let t = (i as f64) / 48_000.0;
            jac_l[0][i] = 0.28 * (TAU * 180.0 * t + 0.3).cos();
            jac_l[1][i] = 0.19 * (TAU * 2_800.0 * t).sin();
            lite_jac_l[0][i] = 0.25 * (TAU * 180.0 * t + 0.4).cos();
            lite_jac_l[1][i] = 0.17 * (TAU * 2_800.0 * t + 0.2).sin();
        }
        let eff_jac = StageTargetJacobian {
            star_out_l: star_l.clone(),
            star_out_r: star_l.clone(),
            lite_star_l: star_l.clone(),
            lite_star_r: star_l.clone(),
            jac_l: jac_l.clone(),
            jac_r: jac_l.clone(),
            lite_jac_l: lite_jac_l.clone(),
            lite_jac_r: lite_jac_l.clone(),
            active_cols,
            gram_uplo: vec![0.0_f64; uplo_u_packed_len(16)],
            n_fit: n,
            warmup,
            mono_out: true,
        };
        let ctx = FourBandJacobianCtx::new(&wet_slice, &eff_jac, warmup, n);
        let w_band = 2.5_f64;
        let mut grad_out = [[0.0_f64; 16]; 2];
        let mut gram_step = vec![0.0_f64; uplo_u_packed_len(16)];
        ctx.accumulate_grad_and_gram(
            w_band,
            [&star_l, &star_l],
            [&star_l, &star_l],
            &active_cols,
            &mut grad_out,
            &mut gram_step,
        );

        let dir = [0.6_f64, -0.8_f64];
        let eps = 2.0e-3_f64;
        let render_shifted = |step: f64, src_jac: &[Vec<f64>]| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    (f64::from(star_l[i]) + step * (dir[0] * src_jac[0][i] + dir[1] * src_jac[1][i]))
                        as f32
                })
                .collect()
        };
        let up_u = render_shifted(eps, &jac_l);
        let up_l = render_shifted(eps, &lite_jac_l);
        let dn_u = render_shifted(-eps, &jac_l);
        let dn_l = render_shifted(-eps, &lite_jac_l);
        let (pu_p, pl_p) = ctx.penalty(&up_u, &up_u, &up_l, &up_l);
        let (pu_m, pl_m) = ctx.penalty(&dn_u, &dn_u, &dn_l, &dn_l);
        let fd_dir_deriv = 0.25 * w_band * ((pu_p + pl_p) - (pu_m + pl_m)) / (2.0 * eps);
        let analytic_dir_deriv =
            -0.5 * ((grad_out[0][0] + grad_out[1][0]) * dir[0] + (grad_out[0][1] + grad_out[1][1]) * dir[1]);
        assert!(
            (fd_dir_deriv - analytic_dir_deriv).abs() < 1.0e-5,
            "FD={fd_dir_deriv} vs analytic={analytic_dir_deriv}"
        );
        assert!(gram_step[uplo_u_index(0, 0)] > 0.0);
        assert!(gram_step[uplo_u_index(1, 1)] > 0.0);
    }

    #[test]
    fn test_per_band_wiener_collapse_detects_single_phase_inverted_band() {
        let n = 512_usize;
        let warmup = 64_usize;
        let dry: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                (0.45 * (TAU * 120.0 * t).sin() + 0.30 * (TAU * 5_500.0 * t).sin()) as f32
            })
            .collect();
        // Wet has phase-inverted 5.5 kHz band at full unity level; model matches 120 Hz but attenuates 5.5 kHz by -6 dB.
        let wet_slice: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let w = (0.45 * (TAU * 120.0 * t).sin() - 0.30 * (TAU * 5_500.0 * t).sin()) as f32;
                [w, w]
            })
            .collect();
        let collapsed_mdl: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                (0.45 * (TAU * 120.0 * t).sin() + 0.08 * (TAU * 5_500.0 * t).sin()) as f32
            })
            .collect();
        let eff_jac = StageTargetJacobian {
            star_out_l: dry.clone(),
            star_out_r: dry.clone(),
            lite_star_l: dry.clone(),
            lite_star_r: dry.clone(),
            jac_l: vec![vec![0.0_f64; n]; 16],
            jac_r: vec![vec![0.0_f64; n]; 16],
            lite_jac_l: vec![vec![0.0_f64; n]; 16],
            lite_jac_r: vec![vec![0.0_f64; n]; 16],
            active_cols: [false; 16],
            gram_uplo: vec![0.0_f64; uplo_u_packed_len(16)],
            n_fit: n,
            warmup,
            mono_out: true,
        };
        assert!(has_per_band_wiener_collapse(
            &wet_slice,
            &eff_jac,
            &collapsed_mdl,
            &collapsed_mdl,
            &collapsed_mdl,
            &collapsed_mdl,
            -0.55,
        ));
        // Coherent dry == wet == mdl must not trigger collapse detection.
        let coherent_wet: Vec<[f32; 2]> = dry.iter().map(|&v| [v, v]).collect();
        assert!(!has_per_band_wiener_collapse(
            &coherent_wet,
            &eff_jac,
            &dry,
            &dry,
            &dry,
            &dry,
            -0.55,
        ));
    }
}
