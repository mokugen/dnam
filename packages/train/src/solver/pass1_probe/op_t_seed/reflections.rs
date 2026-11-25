//! Welch cross-spectral Wiener deconvolution, fixed-lag least-squares fitting, and exact causal
//! $(I + \mathrm{Op\text{-}T}_0)^{-1}$ series inversion for discrete late feed-forward reflections
//! (`>= 32` samples beyond the `Lite` Early FIR horizon, `ref[train.pipeline.physics-probe]`).

use core::f64::consts::TAU;

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;
use crate::losses::Radix2FftPlan;

use super::super::super::model::TransportQuadF64;
use super::super::super::pass2_opt::op_t_quad::eval_op_t_quad_stereo_with_wet_dc_and_adjoint;
use super::super::super::pass2_opt::op_t_quad::forward::{OpTQuadStepState, OpTResolvedParams};

/// High-Q near-unity DC-blocking pole (`0.999995 = 0x3F7F_FFAC`, ~0.038 Hz @ 48 kHz) used for
/// strictly passive feed-forward reflection `Op-T` quads (`g_loop = 0`) to preserve exact DC zero
/// while suppressing 5 Hz transient ringing tails (`ref[dsp.op-t.ring.declaration]`).
pub(crate) const PASSIVE_FF_DC_POLE_F64: f64 = dnam_kernels::MAX_PARALRU_R_F32 as f64;
const _: () = assert!(PASSIVE_FF_DC_POLE_F64 > 0.9999 && PASSIVE_FF_DC_POLE_F64 < 1.0);

// Cross-spectral Wiener reflection detector & passive feed-forward LS thresholds (`ref[train.pipeline.physics-probe]`).
const WELCH_BLOCK_CAP: usize = 8_192;
const LS_WINDOW_CAP: usize = 32_768;
const MIN_REFLECTION_LAG: usize = 32;
const MIN_PEAK_Z_SCORE: f64 = 6.5;
const MIN_WIENER_COEFF: f64 = 0.025;
const MAX_DRY_AUTOCORR_RHO: f64 = 0.35;
const MIN_LOCAL_SHARPNESS: f64 = 1.5;
const MAX_CHANNEL_TAPS_L1: f64 = 0.85;
const MIN_SOLVED_TAP_MAG: f64 = 0.02;

const _: () = assert!(
    WELCH_BLOCK_CAP.is_power_of_two()
        && LS_WINDOW_CAP >= WELCH_BLOCK_CAP
        && MIN_REFLECTION_LAG == 32
        && MIN_PEAK_Z_SCORE > 0.0
        && MIN_WIENER_COEFF > 0.0
        && MIN_WIENER_COEFF < 1.0
        && MAX_DRY_AUTOCORR_RHO > 0.0
        && MAX_DRY_AUTOCORR_RHO < 1.0
        && MIN_LOCAL_SHARPNESS > 1.0
        && MAX_CHANNEL_TAPS_L1 > 0.0
        && MAX_CHANNEL_TAPS_L1 < 1.0
        && MIN_SOLVED_TAP_MAG > 0.0
        && MIN_SOLVED_TAP_MAG < MAX_CHANNEL_TAPS_L1
);

/// Detects up to two dominant feed-forward discrete reflection lags `>= 32` samples per channel
/// via Welch-averaged cross-spectral Wiener deconvolution $S_{R\hat{B}} / (S_{\hat{B}\hat{B}} + \lambda)$.
#[must_use]
pub(crate) fn detect_channel_wiener_reflections(
    input: &[f64],
    b_hat: &[f64],
    res: &[f64],
    start: usize,
    max_lag: usize,
) -> Vec<f64> {
    let n = input.len().min(b_hat.len()).min(res.len());
    let avail = n.saturating_sub(start);
    if avail < 2_048 {
        return Vec::new();
    }
    let b_len = (avail / 2).min(WELCH_BLOCK_CAP);
    if b_len < 1_024 {
        return Vec::new();
    }
    let lag_lo = MIN_REFLECTION_LAG;
    let lag_hi = max_lag.min(b_len / 2).min(4_096);
    if lag_hi < lag_lo {
        return Vec::new();
    }

    let l_fft = (2 * b_len).next_power_of_two();
    let Ok(plan) = Radix2FftPlan::new(l_fft) else {
        return Vec::new();
    };

    let num_blocks = ((2 * avail) / b_len).clamp(2, 8);
    let slack = avail.saturating_sub(b_len);
    let inv_bm1 = 1.0 / ((b_len - 1) as f64);
    let hann: Vec<f64> = (0..b_len)
        .map(|t| 0.5 * (1.0 - (TAU * (t as f64) * inv_bm1).cos()))
        .collect();
    let mut s_rb_re = vec![0.0_f64; l_fft];
    let mut s_rb_im = vec![0.0_f64; l_fft];
    let mut s_bb = vec![0.0_f64; l_fft];
    let mut xr = vec![0.0_f64; l_fft];
    let mut xi = vec![0.0_f64; l_fft];
    let mut yr = vec![0.0_f64; l_fft];
    let mut yi = vec![0.0_f64; l_fft];

    for b_idx in 0..num_blocks {
        let s0 = start + (b_idx * slack) / (num_blocks - 1);
        let inv_b = 1.0 / (b_len as f64);
        let b_mean: f64 = b_hat[s0..s0 + b_len].iter().sum::<f64>() * inv_b;
        let r_mean: f64 = res[s0..s0 + b_len].iter().sum::<f64>() * inv_b;
        xr.fill(0.0);
        xi.fill(0.0);
        yr.fill(0.0);
        yi.fill(0.0);
        for t in 0..b_len {
            let bv = b_hat[s0 + t];
            let rv = res[s0 + t];
            let w = hann[t];
            xr[t] = if bv.is_finite() {
                (bv - b_mean) * w
            } else {
                0.0
            };
            yr[t] = if rv.is_finite() {
                (rv - r_mean) * w
            } else {
                0.0
            };
        }
        plan.fft_in_place(&mut xr, &mut xi);
        plan.fft_in_place(&mut yr, &mut yi);
        for k in 0..l_fft {
            s_rb_re[k] += yr[k] * xr[k] + yi[k] * xi[k];
            s_rb_im[k] += yi[k] * xr[k] - yr[k] * xi[k];
            s_bb[k] += xr[k] * xr[k] + xi[k] * xi[k];
        }
    }

    let mean_s_bb = s_bb.iter().sum::<f64>() / (l_fft as f64);
    if !mean_s_bb.is_finite() || mean_s_bb <= 1.0e-18 {
        return Vec::new();
    }
    let lambda = 1.0e-3 * mean_s_bb;
    let mut w_re = vec![0.0_f64; l_fft];
    let mut w_im = vec![0.0_f64; l_fft];
    for k in 0..l_fft {
        let den = s_bb[k] + lambda;
        w_re[k] = s_rb_re[k] / den;
        w_im[k] = s_rb_im[k] / den;
    }
    plan.ifft_in_place(&mut w_re, &mut w_im);

    let mut mags: Vec<f64> = (lag_lo..=lag_hi)
        .filter_map(|lag| {
            let m = w_re[lag].abs();
            m.is_finite().then_some(m)
        })
        .collect();
    if mags.is_empty() {
        return Vec::new();
    }
    mags.sort_by(f64::total_cmp);
    let sigma_mad = (mags[mags.len() / 2] / 0.674_489_75).max(1.0e-12);

    let rho_cap = n.min(start + 16_384);
    let rho_len = rho_cap.saturating_sub(start);
    let in_mean = if rho_len > 0 {
        input[start..rho_cap].iter().sum::<f64>() / (rho_len as f64)
    } else {
        0.0
    };
    let in_e0: f64 = input[start..rho_cap]
        .iter()
        .map(|&v| {
            let d = v - in_mean;
            d * d
        })
        .sum::<f64>()
        .max(1.0e-12);

    let mut candidates: Vec<(f64, f64)> = Vec::new();
    for lag in lag_lo..=lag_hi {
        let mag = w_re[lag].abs();
        if !mag.is_finite() || mag < MIN_WIENER_COEFF || mag / sigma_mad < MIN_PEAK_Z_SCORE {
            continue;
        }
        let lo_nb = lag.saturating_sub(2).max(lag_lo);
        let hi_nb = (lag + 2).min(lag_hi);
        if (lo_nb..=hi_nb).any(|j| j != lag && w_re[j].abs() > mag) {
            continue;
        }
        let lo_11 = lag.saturating_sub(5).max(lag_lo);
        let hi_11 = (lag + 5).min(lag_hi);
        let mut local_mags: Vec<f64> = (lo_11..=hi_11).map(|j| w_re[j].abs()).collect();
        local_mags.sort_by(f64::total_cmp);
        let med_11 = local_mags[local_mags.len() / 2];
        if mag < MIN_LOCAL_SHARPNESS * med_11 {
            continue;
        }

        let dot: f64 = ((start + lag)..rho_cap)
            .map(|t| (input[t] - in_mean) * (input[t - lag] - in_mean))
            .sum();
        if dot.abs() / in_e0 >= MAX_DRY_AUTOCORR_RHO {
            continue;
        }

        let refined_lag = if lag > lag_lo && lag < lag_hi {
            let y0 = w_re[lag];
            let ym = w_re[lag - 1];
            let yp = w_re[lag + 1];
            if ym.abs() >= 0.15 * mag || yp.abs() >= 0.15 * mag {
                let denom = 2.0 * (ym - 2.0 * y0 + yp);
                if denom.abs() > 1.0e-12 {
                    let delta = ((ym - yp) / denom).clamp(-0.5, 0.5);
                    (lag as f64) + delta
                } else {
                    lag as f64
                }
            } else {
                lag as f64
            }
        } else {
            lag as f64
        };
        candidates.push((refined_lag, mag));
    }

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));
    let mut picked: Vec<f64> = Vec::with_capacity(2);
    for (lag, _) in candidates {
        if picked.iter().all(|&pl| (pl - lag).abs() >= 3.0) {
            picked.push(lag);
            if picked.len() >= 2 {
                break;
            }
        }
    }
    picked
}

/// Builds a strictly passive pure feed-forward [`TransportQuadF64`] assigning up to two Left lags
/// to lanes `[0, 2]` and up to two Right lags to lanes `[1, 3]` (`g_loop = 0`, zero diffusion).
#[must_use]
pub(crate) fn build_feedforward_quad_template(
    lags_l: &[f64],
    lags_r: &[f64],
) -> (TransportQuadF64, [bool; 4]) {
    let mut tq = TransportQuadF64 {
        g_loop: 0.0,
        r_dc: PASSIVE_FF_DC_POLE_F64,
        damp_pole: 0.0,
        allpass_delays: [0; 4],
        allpass_g: [0.0; 4],
        scat_psi: 0.0,
        fb_psi: 0.0,
        ext_f32: true,
        ..TransportQuadF64::default()
    };
    let mut active = [false; 4];
    for (idx, lane) in [0_usize, 2].into_iter().enumerate() {
        if let Some(&lag) = lags_l.get(idx)
            && lag.is_finite()
            && lag >= MIN_DELAY_CROSSBAR_SAMP
        {
            tq.delay_samples[lane] = lag.clamp(MIN_DELAY_CROSSBAR_SAMP, 48_000.0);
            tq.input_gains[lane] = 1.0;
            active[lane] = true;
        }
    }
    for (idx, lane) in [1_usize, 3].into_iter().enumerate() {
        if let Some(&lag) = lags_r.get(idx)
            && lag.is_finite()
            && lag >= MIN_DELAY_CROSSBAR_SAMP
        {
            tq.delay_samples[lane] = lag.clamp(MIN_DELAY_CROSSBAR_SAMP, 48_000.0);
            tq.input_gains[lane] = 1.0;
            active[lane] = true;
        }
    }
    (tq, active)
}

fn fit_lag1_whitener_rho(sig: &[f64], blocks: &[(usize, usize)]) -> f64 {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for &(bs, be) in blocks {
        for t in (bs + 1)..be {
            let x1 = sig[t - 1];
            num += sig[t] * x1;
            den += x1 * x1;
        }
    }
    if den > 1.0e-12 && num.is_finite() {
        // Clamp slightly inside unit circle so broadband LF energy is de-emphasized cleanly
        // without erasing true broadband reflection fringes above ~500 Hz.
        (num / den).clamp(0.0, 0.96)
    } else {
        0.0
    }
}

fn whiten_blocks_diff(sig: &[f64], rho: f64) -> Vec<f64> {
    if rho <= 0.0 {
        return sig.to_vec();
    }
    let mut out = vec![0.0_f64; sig.len()];
    if !sig.is_empty() {
        out[0] = sig[0];
        for t in 1..sig.len() {
            out[t] = sig[t] - rho * sig[t - 1];
        }
    }
    out
}

fn solve_2tap_regularized_blocks(
    col0: &[f64],
    col1_opt: Option<&[f64]>,
    res: &[f64],
    blocks: &[(usize, usize)],
    rho_w: f64,
) -> (f64, f64) {
    let c0_w = whiten_blocks_diff(col0, rho_w);
    let r_w = whiten_blocks_diff(res, rho_w);
    let (mut g00, mut r0) = (0.0_f64, 0.0_f64);
    for &(bs, be) in blocks {
        for t in (bs + 1)..be {
            let a = c0_w[t];
            g00 += a * a;
            r0 += a * r_w[t];
        }
    }
    let Some(col1) = col1_opt else {
        if g00 <= 1.0e-12 {
            return (0.0, 0.0);
        }
        return ((r0 / (g00 * (1.0 + 1.0e-8))).clamp(-1.9, 1.9), 0.0);
    };

    let c1_w = whiten_blocks_diff(col1, rho_w);
    let (mut g01, mut g11, mut r1) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &(bs, be) in blocks {
        for t in (bs + 1)..be {
            let a = c0_w[t];
            let b = c1_w[t];
            g01 += a * b;
            g11 += b * b;
            r1 += b * r_w[t];
        }
    }
    let ridge = 1.0e-8 * 0.5 * (g00 + g11).max(1.0e-12);
    let a00 = g00 + ridge;
    let a11 = g11 + ridge;
    let det = a00 * a11 - g01 * g01;
    if det.abs() <= 1.0e-18 {
        let w0 = if a00 > 1.0e-12 {
            (r0 / a00).clamp(-1.9, 1.9)
        } else {
            0.0
        };
        return (w0, 0.0);
    }
    let w0 = ((a11 * r0 - g01 * r1) / det).clamp(-1.9, 1.9);
    let w1 = ((a00 * r1 - g01 * r0) / det).clamp(-1.9, 1.9);
    (w0, w1)
}

fn eval_single_lane_unit_wet(
    b_l: &[f64],
    b_r: &[f64],
    delay: f64,
    is_right_lane: bool,
    sample_rate_hz: f64,
) -> Vec<f64> {
    let mut tq = TransportQuadF64 {
        g_loop: 0.0,
        r_dc: PASSIVE_FF_DC_POLE_F64,
        damp_pole: 0.0,
        ext_f32: true,
        ..TransportQuadF64::default()
    };
    let lane = usize::from(is_right_lane);
    tq.delay_samples[lane] = delay.clamp(MIN_DELAY_CROSSBAR_SAMP, 48_000.0);
    tq.input_gains[lane] = 1.0;
    if is_right_lane {
        tq.wet_tap_r[lane] = 1.0;
    } else {
        tq.wet_tap_l[lane] = 1.0;
    }
    let (wl, wr, _) = eval_op_t_quad_stereo_with_wet_dc_and_adjoint(
        b_l,
        b_r,
        &tq,
        sample_rate_hz,
        None,
        PASSIVE_FF_DC_POLE_F64,
    );
    if is_right_lane { wr } else { wl }
}

fn normalize_pair_taps(w0: f64, w1: f64, active1: bool) -> (f64, f64) {
    let p0 = if w0.abs() >= MIN_SOLVED_TAP_MAG {
        w0
    } else {
        0.0
    };
    let p1 = if active1 && w1.abs() >= MIN_SOLVED_TAP_MAG {
        w1
    } else {
        0.0
    };
    let l1 = p0.abs() + p1.abs();
    if l1 > MAX_CHANNEL_TAPS_L1 {
        let s = MAX_CHANNEL_TAPS_L1 / l1;
        (p0 * s, p1 * s)
    } else {
        (p0, p1)
    }
}

/// Stereo feed-forward `Quad 0` least-squares refinement inputs.
pub(crate) struct FeedforwardQuadLsInputs<'a> {
    pub b_l: &'a [f64],
    pub b_r: &'a [f64],
    pub target_l: &'a [f64],
    pub target_r: &'a [f64],
    pub start: usize,
    pub sample_rate_hz: f64,
    pub lags_l: &'a [f64],
    pub lags_r: &'a [f64],
}

/// Solves exact regularized stereo least squares for pure feed-forward `Quad 0` wet taps against
/// `target - b` on `[start..win_ls]`, enforcing per-channel non-regression and `>= 20%` total
/// residual reduction.
#[must_use]
pub(crate) fn solve_feedforward_quad_ls(
    inp: &FeedforwardQuadLsInputs<'_>,
) -> Option<(TransportQuadF64, f64)> {
    let FeedforwardQuadLsInputs {
        b_l,
        b_r,
        target_l,
        target_r,
        start,
        sample_rate_hz,
        lags_l,
        lags_r,
    } = *inp;
    let n = b_l
        .len()
        .min(b_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let blocks = super::super::kautz_varpro::select_multiregime_blocks(
        target_l,
        target_r,
        start,
        LS_WINDOW_CAP,
    );
    let win_ls = blocks.last().map_or(start, |&(_, e)| e).min(n);
    if win_ls <= start + 16 || blocks.is_empty() {
        return None;
    }
    let (mut tq, mut active) = build_feedforward_quad_template(lags_l, lags_r);
    if !active.iter().any(|&a| a) {
        return None;
    }

    let res_l: Vec<f64> = (0..win_ls).map(|t| target_l[t] - b_l[t]).collect();
    let res_r: Vec<f64> = (0..win_ls).map(|t| target_r[t] - b_r[t]).collect();
    let mut rho_l = 0.0_f64;
    let mut rho_r = 0.0_f64;

    if active[0] {
        let c0 = eval_single_lane_unit_wet(
            &b_l[..win_ls],
            &b_r[..win_ls],
            tq.delay_samples[0],
            false,
            sample_rate_hz,
        );
        let c1_opt = active[2].then(|| {
            eval_single_lane_unit_wet(
                &b_l[..win_ls],
                &b_r[..win_ls],
                tq.delay_samples[2],
                false,
                sample_rate_hz,
            )
        });
        rho_l = fit_lag1_whitener_rho(&b_l[..win_ls], &blocks);
        let (w0_raw, w1_raw) =
            solve_2tap_regularized_blocks(&c0, c1_opt.as_deref(), &res_l, &blocks, rho_l);
        let (w0, w1) = normalize_pair_taps(w0_raw, w1_raw, active[2]);
        tq.wet_tap_l[0] = w0;
        tq.wet_tap_l[2] = w1;
        if w0 == 0.0 {
            active[0] = false;
            tq.input_gains[0] = 0.0;
        }
        if w1 == 0.0 {
            active[2] = false;
            tq.input_gains[2] = 0.0;
        }
    }

    if active[1] {
        let c0 = eval_single_lane_unit_wet(
            &b_l[..win_ls],
            &b_r[..win_ls],
            tq.delay_samples[1],
            true,
            sample_rate_hz,
        );
        let c1_opt = active[3].then(|| {
            eval_single_lane_unit_wet(
                &b_l[..win_ls],
                &b_r[..win_ls],
                tq.delay_samples[3],
                true,
                sample_rate_hz,
            )
        });
        rho_r = fit_lag1_whitener_rho(&b_r[..win_ls], &blocks);
        let (w0_raw, w1_raw) =
            solve_2tap_regularized_blocks(&c0, c1_opt.as_deref(), &res_r, &blocks, rho_r);
        let (w0, w1) = normalize_pair_taps(w0_raw, w1_raw, active[3]);
        tq.wet_tap_r[1] = w0;
        tq.wet_tap_r[3] = w1;
        if w0 == 0.0 {
            active[1] = false;
            tq.input_gains[1] = 0.0;
        }
        if w1 == 0.0 {
            active[3] = false;
            tq.input_gains[3] = 0.0;
        }
    }

    if !active.iter().any(|&a| a) {
        return None;
    }

    // Score acceptance in the same lag-1 AR whitened metric minimized by the LS solve so a
    // distorted preliminary backbone that swallowed LF reflection gain does not fail pre-refit
    // residual checks (`try_accept_series_ff_reflections` enforces full-band acceptance after refit).
    let (wl, wr, _) = eval_op_t_quad_stereo_with_wet_dc_and_adjoint(
        &b_l[..win_ls],
        &b_r[..win_ls],
        &tq,
        sample_rate_hz,
        None,
        PASSIVE_FF_DC_POLE_F64,
    );
    let res_l_w = whiten_blocks_diff(&res_l, rho_l);
    let res_r_w = whiten_blocks_diff(&res_r, rho_r);
    let wl_w = whiten_blocks_diff(&wl, rho_l);
    let wr_w = whiten_blocks_diff(&wr, rho_r);
    let (mut before_l, mut before_r, mut after_l, mut after_r) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for &(bs, be) in &blocks {
        for t in (bs + 1)..be {
            let dl0 = res_l_w[t];
            let dr0 = res_r_w[t];
            let dl1 = dl0 - wl_w[t];
            let dr1 = dr0 - wr_w[t];
            before_l += dl0 * dl0;
            before_r += dr0 * dr0;
            after_l += dl1 * dl1;
            after_r += dr1 * dr1;
        }
    }
    let before_tot = before_l + before_r;
    let after_tot = after_l + after_r;
    if before_tot <= 1.0e-18
        || !after_tot.is_finite()
        || after_l > 1.001 * before_l + 1.0e-12
        || after_r > 1.001 * before_r + 1.0e-12
        || after_tot >= 0.80 * before_tot
    {
        return None;
    }
    Some((tq, after_tot))
}

/// Runs exact causal series inversion $b[t] = y[t] - \mathrm{Op\text{-}T}_0(b)[t]$ for a passive
/// feed-forward `TransportQuadF64` (`delay >= MIN_DELAY_CROSSBAR_SAMP`, strict state match with
/// [`eval_op_t_quad_stereo_and_adjoint`]).
#[must_use]
pub(crate) fn causal_invert_feedforward_op_t(
    target_l: &[f64],
    target_r: &[f64],
    tq: &TransportQuadF64,
    sample_rate_hz: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = target_l.len().min(target_r.len());
    if n == 0 {
        return None;
    }
    let tap_l1_l: f64 = tq.wet_tap_l.iter().map(|v| v.abs()).sum();
    let tap_l1_r: f64 = tq.wet_tap_r.iter().map(|v| v.abs()).sum();
    if tap_l1_l > MAX_CHANNEL_TAPS_L1 + 1.0e-6 || tap_l1_r > MAX_CHANNEL_TAPS_L1 + 1.0e-6 {
        return None;
    }
    let resolved =
        OpTResolvedParams::from_quad_with_wet_dc(tq, sample_rate_hz, n, PASSIVE_FF_DC_POLE_F64);
    let mut state = OpTQuadStepState::new(resolved, n, false);
    let (mut b_l, mut b_r) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut y_sq, mut b_sq) = (0.0_f64, 0.0_f64);

    for t in 0..n {
        let top = state.step_top_of_frame_read();
        let (yl_wet, yr_wet) = state.preview_wet_output(&top);
        let bl = target_l[t] - yl_wet;
        let br = target_r[t] - yr_wet;
        if !bl.is_finite() || !br.is_finite() {
            return None;
        }
        let _ = state.step_end_of_frame_write(&top, bl, br);
        b_l.push(bl);
        b_r.push(br);
        y_sq += target_l[t] * target_l[t] + target_r[t] * target_r[t];
        b_sq += bl * bl + br * br;
    }
    if !y_sq.is_finite() || !b_sq.is_finite() || (y_sq > 1.0e-18 && b_sq > 16.0 * y_sq) {
        return None;
    }
    Some((b_l, b_r))
}

/// Identifies feed-forward series reflections on post-modal residuals and returns the seeded
/// `Quad 0` together with the causally deconvolved base target `(target - OpT_0(deconv))`.
pub(crate) fn seed_series_feedforward_reflections(
    slices: [&[f64]; 8], // [in_l, in_r, b_l, b_r, res_l, res_r, target_l, target_r]
    start: usize,
    sample_rate_hz: f64,
) -> Option<(TransportQuadF64, Vec<f64>, Vec<f64>)> {
    let [in_l, in_r, b_l, b_r, res_l, res_r, target_l, target_r] = slices;
    let n = in_l
        .len()
        .min(in_r.len())
        .min(b_l.len())
        .min(b_r.len())
        .min(res_l.len())
        .min(res_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let max_lag = ((n.saturating_sub(start)) / 2).min(4_096);
    if max_lag < MIN_REFLECTION_LAG {
        return None;
    }
    let lags_l = detect_channel_wiener_reflections(in_l, b_l, res_l, start, max_lag);
    let lags_r = detect_channel_wiener_reflections(in_r, b_r, res_r, start, max_lag);
    if lags_l.is_empty() && lags_r.is_empty() {
        return None;
    }
    let (tq, _) = solve_feedforward_quad_ls(&FeedforwardQuadLsInputs {
        b_l,
        b_r,
        target_l,
        target_r,
        start,
        sample_rate_hz,
        lags_l: &lags_l,
        lags_r: &lags_r,
    })?;
    let (deconv_l, deconv_r) =
        causal_invert_feedforward_op_t(target_l, target_r, &tq, sample_rate_hz)?;
    Some((tq, deconv_l, deconv_r))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiener_deconv_causal_invert_and_lti_nl_rejection() {
        let sr = 48_000.0_f64;
        let n = 16_384_usize;
        let mut x = vec![0.0_f64; n];
        let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
        for v in &mut x {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = 0.5 * (((rng >> 33) as f64) / ((1_u64 << 30) as f64) - 1.0);
        }

        // Resonant low-pass base filter (`b_true`) + 4 stereo early reflections (`[71, 137, 199, 311]`).
        let mut b_true = vec![0.0_f64; n];
        let mut s1 = 0.0_f64;
        for (t, &xv) in x.iter().enumerate() {
            s1 = 0.88 * s1 + 0.25 * xv;
            b_true[t] = 0.2 * xv + 0.65 * s1;
        }
        let mut y_l = vec![0.0_f64; n];
        let mut y_r = vec![0.0_f64; n];
        for t in 0..n {
            let l1 = if t >= 71 { 0.28 * b_true[t - 71] } else { 0.0 };
            let l2 = if t >= 199 {
                -0.16 * b_true[t - 199]
            } else {
                0.0
            };
            let r1 = if t >= 137 {
                0.25 * b_true[t - 137]
            } else {
                0.0
            };
            let r2 = if t >= 311 {
                0.15 * b_true[t - 311]
            } else {
                0.0
            };
            y_l[t] = b_true[t] + l1 + l2;
            y_r[t] = b_true[t] + r1 + r2;
        }
        let res_l: Vec<f64> = y_l.iter().zip(&b_true).map(|(a, b)| a - b).collect();
        let res_r: Vec<f64> = y_r.iter().zip(&b_true).map(|(a, b)| a - b).collect();

        let (tq, dec_l, dec_r) = seed_series_feedforward_reflections(
            [&x, &x, &b_true, &b_true, &res_l, &res_r, &y_l, &y_r],
            64,
            sr,
        )
        .expect("series reflections should be recovered");
        assert_eq!(tq.g_loop, 0.0);
        assert!(tq.ext_f32);
        assert!(
            (tq.delay_samples[0] - 71.0).abs() <= 0.6,
            "d0={}",
            tq.delay_samples[0]
        );
        assert!(
            (tq.delay_samples[2] - 199.0).abs() <= 0.6,
            "d2={}",
            tq.delay_samples[2]
        );
        assert!(
            (tq.delay_samples[1] - 137.0).abs() <= 0.6,
            "d1={}",
            tq.delay_samples[1]
        );
        assert!(
            (tq.delay_samples[3] - 311.0).abs() <= 0.6,
            "d3={}",
            tq.delay_samples[3]
        );
        assert!(
            (tq.wet_tap_l[0] - 0.28).abs() < 0.04,
            "wl0={}",
            tq.wet_tap_l[0]
        );
        assert!(
            (tq.wet_tap_l[2] + 0.16).abs() < 0.04,
            "wl2={}",
            tq.wet_tap_l[2]
        );
        assert!(
            (tq.wet_tap_r[1] - 0.25).abs() < 0.04,
            "wr1={}",
            tq.wet_tap_r[1]
        );
        assert!(
            (tq.wet_tap_r[3] - 0.15).abs() < 0.04,
            "wr3={}",
            tq.wet_tap_r[3]
        );

        // Causal inversion roundtrip recovers `y_l/y_r` to `< 1e-12` RMS relative error.
        let (wet_rec_l, wet_rec_r, _) = eval_op_t_quad_stereo_with_wet_dc_and_adjoint(
            &dec_l,
            &dec_r,
            &tq,
            sr,
            None,
            PASSIVE_FF_DC_POLE_F64,
        );
        let rt_err: f64 = (0..n)
            .map(|t| {
                (y_l[t] - (dec_l[t] + wet_rec_l[t])).powi(2)
                    + (y_r[t] - (dec_r[t] + wet_rec_r[t])).powi(2)
            })
            .sum::<f64>()
            / y_l
                .iter()
                .chain(&y_r)
                .map(|v| v * v)
                .sum::<f64>()
                .max(1.0e-12);
        assert!(rt_err < 1.0e-12, "causal invert roundtrip err={rt_err}");

        // Memoryless overdrive + steep narrowband LTI bandpass must NOT trigger false reflections.
        let od_y: Vec<f64> = x.iter().map(|&v| (3.0 * v).tanh()).collect();
        let od_res: Vec<f64> = od_y.iter().zip(&x).map(|(y, &xv)| y - 2.0 * xv).collect();
        assert!(
            seed_series_feedforward_reflections(
                [&x, &x, &x, &x, &od_res, &od_res, &od_y, &od_y],
                64,
                sr
            )
            .is_none()
        );

        let r_lp = (-TAU * 900.0 / sr).exp();
        let mut lp = [0.0_f64; 6];
        let mut bp_y = vec![0.0_f64; n];
        for (i, &xv) in x.iter().enumerate() {
            let mut v = xv;
            for st in &mut lp {
                *st = (1.0 - r_lp) * v + r_lp * *st;
                v = *st;
            }
            bp_y[i] = 2.0 * v;
        }
        let bp_res: Vec<f64> = bp_y.iter().zip(&x).map(|(y, &xv)| y - 0.3 * xv).collect();
        assert!(
            seed_series_feedforward_reflections(
                [&x, &x, &x, &x, &bp_res, &bp_res, &bp_y, &bp_y],
                64,
                sr
            )
            .is_none()
        );
    }
}
