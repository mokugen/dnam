#![forbid(unsafe_code)]
//! Dual-Rate Steiglitz–McBride Rational Modal Pole Identification, Deconvolved Stage III
//! Stereo FIR + Nested Kautz Residue Fitting, and Channel-Aligned Stage IV Reflection Extraction
//! (`solver::stage_a::id`).
//!
//! Canonical Reference: `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.physics-probe]`).

use std::f64::consts::TAU;

use super::super::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};

/// Finds all `P` complex roots of monic polynomial $z^P + a_1 z^{P-1} + \dots + a_P$ via
/// Durand–Kerner iterations in `f64` and reflects unstable roots inside `max_radius`.
pub(crate) fn find_and_stabilize_monic_roots(
    a: &[f64],
    max_radius: f64,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let p = a.len();
    let mut roots: Vec<(f64, f64)> = (0..p)
        .map(|k| {
            let ang = TAU * (k as f64 + 0.37) / (p as f64);
            let rad = 0.86 - 0.04 * ((k % 3) as f64);
            (rad * ang.cos(), rad * ang.sin())
        })
        .collect();

    for _ in 0..28 {
        for i in 0..p {
            let (zr, zi) = roots[i];
            let (mut pr, mut pi) = (1.0_f64, 0.0_f64);
            for &ak in a {
                let nr = pr * zr - pi * zi + ak;
                let ni = pr * zi + pi * zr;
                pr = nr;
                pi = ni;
            }
            let (mut dr, mut di) = (1.0_f64, 0.0_f64);
            for (j, &(or, oi)) in roots.iter().enumerate() {
                if i != j {
                    let (vr, vi) = (zr - or, zi - oi);
                    let nr = dr * vr - di * vi;
                    let ni = dr * vi + di * vr;
                    dr = nr;
                    di = ni;
                }
            }
            let den = (dr * dr + di * di).max(1.0e-24);
            let step_r = (pr * dr + pi * di) / den;
            let step_i = (pi * dr - pr * di) / den;
            roots[i] = (
                (zr - step_r).clamp(-2.0, 2.0),
                (zi - step_i).clamp(-2.0, 2.0),
            );
        }
    }

    for (zr, zi) in &mut roots {
        let mag = (*zr * *zr + *zi * *zi).sqrt();
        if mag > max_radius && mag > 1.0e-12 {
            let s = max_radius / mag;
            *zr *= s;
            *zi *= s;
        }
    }

    let mut poly_re = vec![0.0_f64; p + 1];
    let mut poly_im = vec![0.0_f64; p + 1];
    poly_re[0] = 1.0;
    for (deg, &(rr, ri)) in roots.iter().enumerate() {
        for k in (1..=deg + 1).rev() {
            let (prev_r, prev_i) = (poly_re[k - 1], poly_im[k - 1]);
            poly_re[k] -= prev_r * rr - prev_i * ri;
            poly_im[k] -= prev_r * ri + prev_i * rr;
        }
    }
    let a_stab: Vec<f64> = poly_re[1..=p]
        .iter()
        .map(|&v| v.clamp(-64.0, 64.0))
        .collect();
    (roots, a_stab)
}

fn run_steiglitz_mcbride_arma(
    x: &[f64],
    y: &[f64],
    p_ar: usize,
    q_ma: usize,
    max_r: f64,
) -> (Vec<(f64, f64)>, Vec<f64>) {
    let n = x.len().min(y.len());
    let dim = p_ar + q_ma;
    let mut a_cur = vec![0.0_f64; p_ar];
    let mut roots_final = Vec::new();
    let mut lin_arx_res = vec![0.0_f64; n];
    if n <= p_ar + 8 {
        return (roots_final, lin_arx_res);
    }

    for sm_iter in 0..3 {
        let mut xf = vec![0.0_f64; n];
        let mut yf = vec![0.0_f64; n];
        if sm_iter == 0 {
            xf.copy_from_slice(&x[..n]);
            yf.copy_from_slice(&y[..n]);
        } else {
            for i in 0..n {
                let (mut sx, mut sy) = (x[i], y[i]);
                for k in 1..=i.min(p_ar) {
                    sx -= a_cur[k - 1] * xf[i - k];
                    sy -= a_cur[k - 1] * yf[i - k];
                }
                xf[i] = sx.clamp(-16.0, 16.0);
                yf[i] = sy.clamp(-16.0, 16.0);
            }
        }

        let mut g = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut c = vec![[0.0_f64; 2]; dim];
        let mut phi = vec![0.0_f64; dim];
        let i_start = if n <= 128 { 0 } else { p_ar };
        for i in i_start..n {
            for k in 0..p_ar {
                phi[k] = if i > k { -yf[i - 1 - k] } else { 0.0 };
            }
            for m in 0..q_ma {
                phi[p_ar + m] = if i >= m { xf[i - m] } else { 0.0 };
            }
            let target = yf[i];
            for col in 0..dim {
                let pc = phi[col];
                let base = uplo_u_index(0, col);
                for row in 0..=col {
                    g[base + row] += phi[row] * pc;
                }
                c[col][0] += pc * target;
            }
        }

        let mut d_inv = vec![0.0_f64; dim];
        let mut g_eq = vec![0.0_f64; uplo_u_packed_len(dim)];
        let mut c_eq = vec![[0.0_f64; 2]; dim];
        for col in 0..dim {
            let diag = g[uplo_u_index(col, col)];
            d_inv[col] = if diag > 1.0e-20 {
                1.0 / diag.sqrt()
            } else {
                0.0
            };
        }
        for col in 0..dim {
            let base = uplo_u_index(0, col);
            let dc = d_inv[col];
            for row in 0..=col {
                g_eq[base + row] = g[base + row] * d_inv[row] * dc;
            }
            if dc == 0.0 {
                g_eq[base + col] = 1.0;
            }
            c_eq[col][0] = c[col][0] * dc;
        }
        let Ok(sol_eq) = solve_uplo_u_cholesky_2ch(&g_eq, &c_eq, 1.0e-5) else {
            break;
        };
        let sol: Vec<[f64; 2]> = (0..dim).map(|k| [sol_eq[k][0] * d_inv[k], 0.0]).collect();

        if sm_iter == 0 {
            for i in p_ar.max(q_ma)..n {
                let mut pred = 0.0_f64;
                for k in 0..p_ar {
                    pred -= sol[k][0] * y[i - 1 - k];
                }
                for m in 0..q_ma {
                    pred += sol[p_ar + m][0] * x[i - m];
                }
                lin_arx_res[i] = y[i] - pred;
            }
        }

        let a_raw: Vec<f64> = (0..p_ar).map(|k| sol[k][0]).collect();
        let (roots, a_stab) = find_and_stabilize_monic_roots(&a_raw, max_r);
        a_cur = a_stab;
        roots_final = roots;
    }

    (roots_final, lin_arx_res)
}

/// Runs dual-rate (`48 kHz` broadband + `6 kHz` `8x`-decimated LF) prefiltered Steiglitz–McBride
/// rational pole identification AND measures nonlinear distortion variance (`r[train.pipeline.physics-probe]`).
pub(crate) fn identify_steiglitz_mcbride_and_linearity(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    hf_growth: f32,
) -> (f32, Vec<(f32, f32)>, usize) {
    const P_AR: usize = 18;
    const Q_MA: usize = 17;
    let n = dry_lr.len().min(wet_lr.len()).min(12_000);
    if n < 96 {
        return ((hf_growth - 1.0).clamp(0.0, 1.0), Vec::new(), 0);
    }

    let mut x = vec![0.0_f64; n];
    let mut y = vec![0.0_f64; n];
    let mut wet_e = 0.0_f64;
    let nl_end = n.min(4_096);
    let mut mean_abs_x = 0.0_f64;
    for i in 0..n {
        let xi = 0.5 * (f64::from(dry_lr[i][0]) + f64::from(dry_lr[i][1]));
        let (wl, wr) = (f64::from(wet_lr[i][0]), f64::from(wet_lr[i][1]));
        // Asymmetric stereo blend (1.0*L + 0.37*R) never cancels same-sign or opposite-sign modes.
        let yi = if (wl - wr).abs() <= 1.0e-7 {
            wl
        } else {
            (wl + 0.37 * wr) * (1.0 / 1.37)
        };
        x[i] = if xi.is_finite() { xi } else { 0.0 };
        y[i] = if yi.is_finite() { yi } else { 0.0 };
        wet_e += y[i] * y[i];
        if i < nl_end {
            mean_abs_x += x[i].abs();
        }
    }
    mean_abs_x /= nl_end as f64;
    if wet_e <= 1.0e-12 {
        return (0.0, Vec::new(), 0);
    }

    // Deconvolve early 104-tap causal impulse response and gate before discrete room reflections
    // (> 24 samples post-onset) so 48 kHz SM identifies true modal poles rather than comb echoes.
    let (bb_start, bb_end) = if n > 8_744 {
        let tot_uhf: f64 = (2..n)
            .map(|i| (x[i] - 2.0 * x[i - 1] + x[i - 2]).powi(2))
            .sum();
        let tail_s = n - 3_256;
        let tail_uhf: f64 = (tail_s.max(2)..n)
            .map(|i| (x[i] - 2.0 * x[i - 1] + x[i - 2]).powi(2))
            .sum();
        if tail_uhf > 0.25 * tot_uhf {
            (8_744, n)
        } else {
            let win = 7_200.min(n);
            let mut best_s = 0usize;
            let mut best_sc = -1.0_f64;
            let mut s = 0usize;
            loop {
                let e_end = (s + win).min(n);
                let mut el = 0.0_f64;
                let mut eh = 0.0_f64;
                for i in (s + 2)..e_end {
                    el += x[i] * x[i];
                    let d2 = x[i] - 2.0 * x[i - 1] + x[i - 2];
                    eh += d2 * d2;
                }
                let sc = el.sqrt() * eh;
                if sc > best_sc {
                    best_sc = sc;
                    best_s = s;
                }
                if e_end == n {
                    break;
                }
                s = (s + 1_024).min(n - win);
            }
            (best_s, best_s + win)
        }
    } else {
        (0, n)
    };
    let h_ir = solve_causal_fir_1ch(&x[bb_start..bb_end], &y[bb_start..bb_end], 0, 160);
    let early_peak = (0..24.min(h_ir.len()))
        .max_by(|&a, &b| h_ir[a].abs().total_cmp(&h_ir[b].abs()))
        .unwrap_or(0);
    let full_peak = (0..96.min(h_ir.len()))
        .max_by(|&a, &b| h_ir[a].abs().total_cmp(&h_ir[b].abs()))
        .unwrap_or(early_peak);
    let k_peak = if h_ir.get(full_peak).map_or(0.0, |v| v.abs())
        > 2.2 * h_ir.get(early_peak).map_or(0.0, |v| v.abs())
    {
        full_peak
    } else {
        early_peak
    };
    let peak_abs = h_ir.get(k_peak).map_or(0.0, |v| v.abs()).max(1.0e-9);
    let mut gate_end = (k_peak + 78).min(h_ir.len());
    for k in (k_peak + 24)..h_ir.len().saturating_sub(2) {
        let prev_rms = (h_ir[k - 12..k - 2].iter().map(|v| v * v).sum::<f64>() * 0.1)
            .sqrt()
            .max(1.0e-9);
        if prev_rms < 0.35 * peak_abs && h_ir[k].abs() > 3.2 * prev_rms {
            gate_end = k - 2;
            break;
        }
    }
    let mut delta_hp = vec![0.0_f64; gate_end];
    let mut h_hp = vec![0.0_f64; gate_end];
    if gate_end >= 2 {
        delta_hp[0] = 1.0;
        delta_hp[1] = -0.965;
        h_hp[0] = h_ir[0];
        for k in 1..gate_end {
            h_hp[k] = h_ir[k] - 0.965 * h_ir[k - 1];
        }
    }
    let (roots_48k, _) =
        run_steiglitz_mcbride_arma(&delta_hp, &h_hp, 16, (k_peak + 10).min(48), 0.994);
    let (_, lin_arx_res) =
        run_steiglitz_mcbride_arma(&x[..nl_end], &y[..nl_end], P_AR, Q_MA, 0.992);

    const NL_LAGS: [usize; 4] = [0, 3, 7, 13];
    const DIM_AUG: usize = 16;
    let mut g_aug = vec![0.0_f64; uplo_u_packed_len(DIM_AUG)];
    let mut c_aug = vec![[0.0_f64; 2]; DIM_AUG];
    let mut psi = [0.0_f64; DIM_AUG];
    let mut wet_nl_e = 0.0_f64;
    for i in P_AR..nl_end {
        wet_nl_e += y[i] * y[i];
        for (idx_l, &lag) in NL_LAGS.iter().enumerate() {
            let xd = x[i - lag];
            psi[idx_l] = xd;
            psi[4 + idx_l * 3] = xd * xd.abs();
            psi[4 + idx_l * 3 + 1] = xd * xd * xd - 0.5 * xd;
            psi[4 + idx_l * 3 + 2] = xd.abs() - mean_abs_x;
        }
        let r = lin_arx_res[i];
        for col in 0..DIM_AUG {
            let pc = psi[col];
            let base = uplo_u_index(0, col);
            for row in 0..=col {
                g_aug[base + row] += psi[row] * pc;
            }
            c_aug[col][0] += pc * r;
        }
    }
    let mut g_lin4 = vec![0.0_f64; uplo_u_packed_len(4)];
    for col in 0..4 {
        let src = uplo_u_index(0, col);
        g_lin4[src..src + col + 1].copy_from_slice(&g_aug[src..src + col + 1]);
    }
    let e_lin4 = solve_uplo_u_cholesky_2ch(&g_lin4, &c_aug[..4], 1.0e-6).map_or(0.0, |w| {
        (0..4).map(|k| w[k][0] * c_aug[k][0]).sum::<f64>().max(0.0)
    });
    let e_aug16 = solve_uplo_u_cholesky_2ch(&g_aug, &c_aug, 1.0e-6).map_or(0.0, |w| {
        (0..DIM_AUG)
            .map(|k| w[k][0] * c_aug[k][0])
            .sum::<f64>()
            .max(0.0)
    });
    let lin_res_e: f64 = lin_arx_res[P_AR.max(Q_MA)..nl_end]
        .iter()
        .map(|r| r * r)
        .sum();
    let lin_unexplained = (lin_res_e / wet_nl_e.max(1.0e-12)).clamp(0.0, 1.0) as f32;
    let nl_gate = ((lin_unexplained - 0.004) * 35.0).clamp(0.0, 1.0);
    let nl_excess_ratio = ((e_aug16 - e_lin4).max(0.0) / wet_nl_e.max(1.0e-12)).clamp(0.0, 1.0);
    let nl_from_res = ((nl_excess_ratio - 0.0045).max(0.0).sqrt() * 3.4) as f32;
    let nl_from_hf = nl_gate * (0.55 * (hf_growth - 1.06).max(0.0)).clamp(0.0, 0.85);
    let nl_drive = (nl_from_res + nl_from_hf).clamp(0.0, 1.0);

    // Guard against Froissart pole-zero doublets on memoryless / short-FIR / tonal targets where
    // a 16-tap (0.33 ms) causal FIR already explains >= 99.9% of the transfer (< -30 dB residual).
    let h16 = solve_causal_fir_1ch(&x[bb_start..bb_end], &y[bb_start..bb_end], 0, 16);
    let (mut fir16_res_e, mut fir16_tot_e) = (0.0_f64, 0.0_f64);
    for i in (bb_start + 16)..bb_end {
        let mut p = 0.0_f64;
        for (k, &hk) in h16.iter().enumerate() {
            p += hk * x[i - k];
        }
        let d = y[i] - p;
        fir16_res_e += d * d;
        fir16_tot_e += y[i] * y[i];
    }
    if fir16_tot_e > 1.0e-12 && fir16_res_e <= 1.0e-3 * fir16_tot_e {
        return (nl_drive, Vec::new(), k_peak);
    }

    let mut candidates: Vec<(f64, f32, f32)> = Vec::new();
    // Low-frequency 8x-decimated (6 kHz) pass across all 12k samples resolves 35..900 Hz resonances
    if n >= 1_024 {
        let n_low = n / 8;
        let mut x_low = vec![0.0_f64; n_low];
        let mut y_low = vec![0.0_f64; n_low];
        for k in 0..n_low {
            let b = k * 8;
            x_low[k] = 0.125 * x[b..b + 8].iter().sum::<f64>();
            y_low[k] = 0.125 * y[b..b + 8].iter().sum::<f64>();
        }
        let mut x_lp_hp = vec![0.0_f64; n_low];
        let mut y_lp_hp = vec![0.0_f64; n_low];
        for k in 1..n_low {
            x_lp_hp[k] = x_low[k] - 0.88 * x_low[k - 1];
            y_lp_hp[k] = y_low[k] - 0.88 * y_low[k - 1];
        }
        let (roots_6k, _) = run_steiglitz_mcbride_arma(&x_lp_hp, &y_lp_hp, 12, 50, 0.985);
        for (zr, zi) in roots_6k {
            if zi <= 1.0e-3 {
                continue;
            }
            let rad_6k = (zr * zr + zi * zi).sqrt();
            let th_6k = zi.atan2(zr);
            if rad_6k >= 0.40 && (0.036..=1.25).contains(&th_6k) {
                let r_48 = rad_6k.powf(0.125).clamp(0.88, 0.994);
                let th_48 = th_6k * 0.125;
                let q_score = 1.25 * (r_48 / (1.004 - r_48.min(0.995)));
                candidates.push((q_score, r_48 as f32, th_48 as f32));
            }
        }
    }

    for (zr, zi) in roots_48k {
        if zi <= 1.5e-3 {
            continue;
        }
        let rad = (zr * zr + zi * zi).sqrt();
        let theta = zi.atan2(zr);
        if rad >= 0.65 && (0.085..=1.35).contains(&theta) {
            let r_clamped = rad.clamp(0.85, 0.994) as f32;
            candidates.push((rad / (1.005 - rad.min(0.995)), r_clamped, theta as f32));
        }
    }

    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut sm_poles: Vec<(f32, f32)> = Vec::new();
    for (_, r_m, th_m) in candidates {
        let distinct = sm_poles
            .iter()
            .all(|&(_, eth)| (eth - th_m).abs() >= 0.003 && (eth.ln() - th_m.ln()).abs() >= 0.14);
        if distinct {
            sm_poles.push((r_m, th_m));
            if sm_poles.len() >= 16 {
                break;
            }
        }
    }

    (nl_drive, sm_poles, k_peak)
}

/// Estimates the causal `64`-tap (`1.33 ms` at `48 kHz`) stereo Early FIR impulse response via
/// regularized autocorrelation/cross-correlation Cholesky (`r[train.pipeline.physics-probe]`).
pub(crate) fn solve_early_fir_64_stereo(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> ([f32; 64], [f32; 64]) {
    const TAPS: usize = 64;
    let n = dry_lr.len().min(wet_lr.len());
    if n <= TAPS {
        let mut hl = [0.0_f32; TAPS];
        let mut hr = [0.0_f32; TAPS];
        hl[0] = 0.25;
        hr[0] = 0.25;
        return (hl, hr);
    }

    let mut r_xx = [0.0_f64; TAPS];
    let mut r_xy = [[0.0_f64; 2]; TAPS];
    let step = if n > 4_000 { 2 } else { 1 };
    let mut idx = TAPS;
    while idx < n {
        let x0 = 0.5 * (f64::from(dry_lr[idx][0]) + f64::from(dry_lr[idx][1]));
        let yl = f64::from(wet_lr[idx][0]);
        let yr = f64::from(wet_lr[idx][1]);
        if x0.is_finite() && yl.is_finite() && yr.is_finite() {
            for lag in 0..TAPS {
                let x_lag =
                    0.5 * (f64::from(dry_lr[idx - lag][0]) + f64::from(dry_lr[idx - lag][1]));
                r_xx[lag] += x0 * x_lag;
                r_xy[lag][0] += x_lag * yl;
                r_xy[lag][1] += x_lag * yr;
            }
        }
        idx += step;
    }

    let mut g = vec![0.0_f64; uplo_u_packed_len(TAPS)];
    for j in 0..TAPS {
        for i in 0..=j {
            g[uplo_u_index(i, j)] = r_xx[j - i];
        }
    }
    let ridge = (0.01 * r_xx[0]).max(1.0e-5);
    let solved =
        solve_uplo_u_cholesky_2ch(&g, &r_xy, ridge).unwrap_or_else(|_| vec![[0.0; 2]; TAPS]);

    let mut hl = [0.0_f32; TAPS];
    let mut hr = [0.0_f32; TAPS];
    for k in 0..TAPS {
        hl[k] = (solved[k][0] as f32).clamp(-1.5, 1.5);
        hr[k] = (solved[k][1] as f32).clamp(-1.5, 1.5);
    }
    (hl, hr)
}

fn solve_equilibrated_1ch(g: &[f64], c: &[[f64; 2]], dim: usize, rel_ridge: f64) -> Vec<f64> {
    let mut d_inv = vec![0.0_f64; dim];
    let mut g_eq = vec![0.0_f64; uplo_u_packed_len(dim)];
    let mut c_eq = vec![[0.0_f64; 2]; dim];
    for col in 0..dim {
        let diag = g[uplo_u_index(col, col)];
        d_inv[col] = if diag > 1.0e-20 {
            diag.sqrt().recip()
        } else {
            0.0
        };
    }
    for col in 0..dim {
        let (base, dc) = (uplo_u_index(0, col), d_inv[col]);
        for row in 0..=col {
            g_eq[base + row] = g[base + row] * d_inv[row] * dc;
        }
        if dc == 0.0 {
            g_eq[base + col] = 1.0;
        }
        g_eq[base + col] += 1.5e-5 * (col as f64);
        c_eq[col][0] = c[col][0] * dc;
    }
    solve_uplo_u_cholesky_2ch(&g_eq, &c_eq, rel_ridge).map_or_else(
        |_| vec![0.0_f64; dim],
        |sol| (0..dim).map(|k| sol[k][0] * d_inv[k]).collect(),
    )
}

fn solve_causal_fir_1ch(u: &[f64], target: &[f64], lag_offset: usize, taps: usize) -> Vec<f64> {
    let n = u.len().min(target.len());
    if n <= lag_offset + taps + 1 {
        return vec![0.0_f64; taps];
    }
    let mut g = vec![0.0_f64; uplo_u_packed_len(taps)];
    let mut c = vec![[0.0_f64; 2]; taps];
    let q3 = (3 * n) / 4;
    let q23 = (2 * n) / 3;
    let tot_u: f64 = u[..n].iter().map(|v| v * v).sum();
    let q3_u: f64 = u[q3..n].iter().map(|v| v * v).sum();
    let q3_active = q3_u > 0.05 * tot_u;
    for i in (lag_offset + taps + 1)..n {
        let w = if n >= 8_000 && ((q3_active && i >= q3) || (!q3_active && i < q23)) {
            4.0
        } else {
            1.0
        };
        let ti = w * target[i];
        let ti_hp = 1.5 * w * (target[i] - 0.96 * target[i - 1]);
        for col in 0..taps {
            let idx = i - lag_offset - col;
            let uc = u[idx];
            let uc_hp = uc - 0.96 * u[idx - 1];
            let base = uplo_u_index(0, col);
            for row in 0..=col {
                let r_idx = i - lag_offset - row;
                let ur = u[r_idx];
                let ur_hp = ur - 0.96 * u[r_idx - 1];
                g[base + row] += w * (ur * uc + 1.5 * ur_hp * uc_hp);
            }
            c[col][0] += uc * ti + uc_hp * ti_hp;
        }
    }
    solve_equilibrated_1ch(&g, &c, taps, 2.0e-5)
}
