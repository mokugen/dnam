//! 1-D Spectral Optimal Transport (`SpecOT`) and Instantaneous Frequency + Group Delay
//! (`IF+GD`) unit-phasor coherence losses with exact STFT Wirtinger/overlap-add adjoints
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use crate::error::TrainError;

use super::fft::{
    PairedFftWorkspace, Radix2FftPlan, build_hann_window, paired_real_backward_adjoint,
    paired_real_forward_spectra,
};

const PHASOR_EPS: f64 = 1.0e-6;

#[derive(Clone, Debug)]
struct StftFrameGrid {
    len: usize,
    seg_len: usize,
    starts: Vec<usize>,
    win: Vec<f64>,
    plan: Radix2FftPlan,
    k_lo: usize,
    k_hi: usize,
}

impl StftFrameGrid {
    fn new(
        len: usize,
        warmup: usize,
        sample_rate_hz: f64,
        win_len: usize,
        hop_len: usize,
    ) -> Result<Self, TrainError> {
        let seg_len = win_len.clamp(32, len.saturating_sub(warmup).max(32));
        let n_fft = seg_len.next_power_of_two();
        let plan = Radix2FftPlan::new(n_fft)?;
        let (win, _) = build_hann_window(seg_len);
        let hop = hop_len.max(8);
        let mut starts = Vec::new();
        let mut st = warmup;
        while st + seg_len <= len {
            starts.push(st);
            st += hop;
        }
        if starts.is_empty() && warmup + seg_len <= len {
            starts.push(warmup);
        }
        let df = sample_rate_hz / (n_fft as f64);
        let k_lo = ((20.0 / df).floor() as usize).max(1);
        let k_hi = ((20_000.0_f64.min(0.49 * sample_rate_hz) / df).ceil() as usize).min(n_fft / 2);
        Ok(Self {
            len,
            seg_len,
            starts,
            win,
            plan,
            k_lo,
            k_hi: k_hi.max(k_lo + 1),
        })
    }

    fn forward_frames(&self, sig: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n_fft = self.plan.n_fft();
        let half_p1 = n_fft / 2 + 1;
        let m_cnt = self.starts.len();
        let mut re_frames = vec![vec![0.0_f64; half_p1]; m_cnt];
        let mut im_frames = vec![vec![0.0_f64; half_p1]; m_cnt];
        let mut ws = PairedFftWorkspace::new(n_fft);
        let (mut r0, mut i0, mut r1, mut i1) = (
            vec![0.0_f64; half_p1],
            vec![0.0_f64; half_p1],
            vec![0.0_f64; half_p1],
            vec![0.0_f64; half_p1],
        );
        let mut m = 0_usize;
        while m < m_cnt {
            let s0 = &sig[self.starts[m]..self.starts[m] + self.seg_len];
            let s1 = (m + 1 < m_cnt)
                .then(|| &sig[self.starts[m + 1]..self.starts[m + 1] + self.seg_len]);
            paired_real_forward_spectra(
                &self.plan,
                s0,
                s1,
                &self.win,
                &mut ws,
                (&mut r0, &mut i0),
                Some((&mut r1, &mut i1)),
            );
            re_frames[m].copy_from_slice(&r0);
            im_frames[m].copy_from_slice(&i0);
            if m + 1 < m_cnt {
                re_frames[m + 1].copy_from_slice(&r1);
                im_frames[m + 1].copy_from_slice(&i1);
            }
            m += 2;
        }
        (re_frames, im_frames)
    }

    fn backward_frames(&self, g_re: &[Vec<f64>], g_im: &[Vec<f64>], grad: &mut [f64]) {
        let n_fft = self.plan.n_fft();
        let m_cnt = self.starts.len();
        let mut ws = PairedFftWorkspace::new(n_fft);
        let (mut dy0, mut dy1) = (vec![0.0_f64; self.seg_len], vec![0.0_f64; self.seg_len]);
        let mut m = 0_usize;
        while m < m_cnt {
            dy0.fill(0.0);
            dy1.fill(0.0);
            let b0 = self.starts[m];
            let has1 = m + 1 < m_cnt;
            paired_real_backward_adjoint(
                &self.plan,
                &self.win,
                (&g_re[m], &g_im[m]),
                has1.then(|| (g_re[m + 1].as_slice(), g_im[m + 1].as_slice())),
                &mut ws,
                &mut dy0,
                has1.then_some(&mut dy1),
            );
            for (g, d) in grad[b0..b0 + self.seg_len].iter_mut().zip(&dy0) {
                *g += *d;
            }
            if has1 {
                let b1 = self.starts[m + 1];
                for (g, d) in grad[b1..b1 + self.seg_len].iter_mut().zip(&dy1) {
                    *g += *d;
                }
            }
            m += 2;
        }
    }
}

/// Target CDF cache for 1-D Spectral Optimal Transport (`SpecOT`, `ref[train.pipeline.backbone-trajectory-opt]`).
#[derive(Clone, Debug)]
pub struct SpecOtTargetCache {
    grid: StftFrameGrid,
    tgt_cdf: Vec<f64>,
    bin_weights: Vec<f64>,
}

impl SpecOtTargetCache {
    /// Precomputes normalized target power CDF and log-frequency transport weights.
    pub fn new(
        target: &[f64],
        warmup: usize,
        sample_rate_hz: f64,
        fft_size: usize,
    ) -> Result<Self, TrainError> {
        let grid =
            StftFrameGrid::new(target.len(), warmup, sample_rate_hz, fft_size, fft_size / 2)?;
        if grid.starts.is_empty() {
            return Err(TrainError::InvalidDimensions {
                context: "SpecOtTargetCache::new(insufficient frames)",
                expected: warmup + grid.seg_len,
                actual: target.len(),
            });
        }
        let (re, im) = grid.forward_frames(target);
        let m_cnt = re.len() as f64;
        let k_len = grid.k_hi - grid.k_lo;
        let mut p = vec![0.0_f64; k_len];
        let mut z = 0.0_f64;
        for (idx, k) in (grid.k_lo..grid.k_hi).enumerate() {
            let mean_p = re
                .iter()
                .zip(&im)
                .map(|(r, i)| r[k] * r[k] + i[k] * i[k])
                .sum::<f64>()
                / m_cnt
                + 1.0e-15;
            p[idx] = mean_p;
            z += mean_p;
        }
        let mut cdf = vec![0.0_f64; k_len];
        let mut run = 0.0_f64;
        for idx in 0..k_len {
            run += p[idx] / z.max(1.0e-24);
            cdf[idx] = run;
        }
        let bin_weights: Vec<f64> = (grid.k_lo..grid.k_hi)
            .map(|k| (((k + 1) as f64) / (k as f64)).ln())
            .collect();
        Ok(Self {
            grid,
            tgt_cdf: cdf,
            bin_weights,
        })
    }
}

/// Evaluates smoothed 1-D Wasserstein-1 `SpecOT` loss and accumulates exact STFT adjoint into `grad_opt`.
pub fn eval_spec_ot_loss_and_adjoint(
    cache: &SpecOtTargetCache,
    pred: &[f64],
    grad_opt: Option<&mut [f64]>,
) -> f64 {
    if pred.len() != cache.grid.len || cache.grid.starts.is_empty() {
        return 0.0;
    }
    let (re, im) = cache.grid.forward_frames(pred);
    let m_cnt = re.len();
    let inv_m = 1.0 / (m_cnt as f64);
    let k_lo = cache.grid.k_lo;
    let k_hi = cache.grid.k_hi;
    let k_len = k_hi - k_lo;

    let mut p_hat = vec![0.0_f64; k_len];
    let mut z = 0.0_f64;
    for (idx, k) in (k_lo..k_hi).enumerate() {
        let mp = re
            .iter()
            .zip(&im)
            .map(|(r, i)| r[k] * r[k] + i[k] * i[k])
            .sum::<f64>()
            * inv_m
            + 1.0e-15;
        p_hat[idx] = mp;
        z += mp;
    }

    let delta = 1.0e-5_f64;
    let inv_z = 1.0 / z.max(1.0e-24);
    let mut d_cdf = vec![0.0_f64; k_len];
    let mut run_cdf = 0.0_f64;
    let mut loss = 0.0_f64;
    for idx in 0..k_len {
        let pmass = p_hat[idx] * inv_z;
        run_cdf += pmass;
        let diff = run_cdf - cache.tgt_cdf[idx];
        let root = (diff * diff + delta * delta).sqrt();
        loss += cache.bin_weights[idx] * (root - delta);
        d_cdf[idx] = cache.bin_weights[idx] * (diff / root);
    }

    if let Some(grad) = grad_opt {
        let mut d_p = vec![0.0_f64; k_len];
        let mut cum_tail = 0.0_f64;
        for idx in (0..k_len).rev() {
            cum_tail += d_cdf[idx];
            d_p[idx] = cum_tail;
        }
        let dot_p_dp: f64 = p_hat
            .iter()
            .zip(&d_p)
            .map(|(&pv, &dv)| (pv * inv_z) * dv)
            .sum();

        let half_p1 = cache.grid.plan.n_fft() / 2 + 1;
        let mut g_re = vec![vec![0.0_f64; half_p1]; m_cnt];
        let mut g_im = vec![vec![0.0_f64; half_p1]; m_cnt];
        for (idx, k) in (k_lo..k_hi).enumerate() {
            let d_pow = (d_p[idx] - dot_p_dp) * inv_z;
            let c = 2.0 * inv_m * d_pow;
            for m in 0..m_cnt {
                g_re[m][k] = c * re[m][k];
                g_im[m][k] = c * im[m][k];
            }
        }
        cache.grid.backward_frames(&g_re, &g_im, grad);
    }
    loss
}

/// Target unit-phasor cache for Instantaneous Frequency + Group Delay (`IF+GD`) coherence loss.
#[derive(Clone, Debug)]
pub struct IfGdTargetCache {
    grid: StftFrameGrid,
    if_u: Vec<f64>,
    if_v: Vec<f64>,
    if_w: Vec<f64>,
    gd_u: Vec<f64>,
    gd_v: Vec<f64>,
    gd_w: Vec<f64>,
    total_w: f64,
}

fn unit_phasor_diff(r0: f64, i0: f64, r1: f64, i1: f64) -> (f64, f64) {
    let dot = r0 * r1 + i0 * i1;
    let cross = i0 * r1 - r0 * i1;
    let d0 = (r0 * r0 + i0 * i0 + PHASOR_EPS).sqrt();
    let d1 = (r1 * r1 + i1 * i1 + PHASOR_EPS).sqrt();
    let den = d0 * d1;
    (dot / den, cross / den)
}

#[derive(Clone, Copy)]
struct PhasorPairInput {
    r0: f64,
    i0: f64,
    r1: f64,
    i1: f64,
    tgt_u: f64,
    tgt_v: f64,
    weight: f64,
}

#[inline]
fn accum_phasor_pair_grad(p: PhasorPairInput) -> (f64, [f64; 4]) {
    let dot = p.r0 * p.r1 + p.i0 * p.i1;
    let cross = p.i0 * p.r1 - p.r0 * p.i1;
    let d0 = (p.r0 * p.r0 + p.i0 * p.i0 + PHASOR_EPS).sqrt();
    let d1 = (p.r1 * p.r1 + p.i1 * p.i1 + PHASOR_EPS).sqrt();
    let den = d0 * d1;
    let u = dot / den;
    let v = cross / den;
    let du = u - p.tgt_u;
    let dv = v - p.tgt_v;
    let val = 0.5 * p.weight * (du * du + dv * dv);

    let g_dot = p.weight * du / den;
    let g_cross = p.weight * dv / den;
    let g_den = -p.weight * (du * u + dv * v) / den;
    let s0 = g_den * (d1 / d0);
    let s1 = g_den * (d0 / d1);

    let dr0 = g_dot * p.r1 - g_cross * p.i1 + s0 * p.r0;
    let di0 = g_dot * p.i1 + g_cross * p.r1 + s0 * p.i0;
    let dr1 = g_dot * p.r0 + g_cross * p.i0 + s1 * p.r1;
    let di1 = g_dot * p.i0 - g_cross * p.r0 + s1 * p.i1;
    (val, [dr0, di0, dr1, di1])
}

impl IfGdTargetCache {
    /// Precomputes normalized instantaneous frequency & group delay unit phasors weighted by target energy.
    pub fn new(
        target: &[f64],
        warmup: usize,
        sample_rate_hz: f64,
        fft_size: usize,
    ) -> Result<Self, TrainError> {
        let grid =
            StftFrameGrid::new(target.len(), warmup, sample_rate_hz, fft_size, fft_size / 4)?;
        if grid.starts.len() < 2 {
            return Err(TrainError::InvalidDimensions {
                context: "IfGdTargetCache::new(need >= 2 frames)",
                expected: warmup + fft_size + fft_size / 4,
                actual: target.len(),
            });
        }
        let (re, im) = grid.forward_frames(target);
        let m_cnt = re.len();
        let k_len = grid.k_hi - grid.k_lo;
        let mut if_u = vec![0.0_f64; m_cnt * k_len];
        let mut if_v = vec![0.0_f64; m_cnt * k_len];
        let mut if_w = vec![0.0_f64; m_cnt * k_len];
        let mut gd_u = vec![0.0_f64; m_cnt * k_len];
        let mut gd_v = vec![0.0_f64; m_cnt * k_len];
        let mut gd_w = vec![0.0_f64; m_cnt * k_len];
        let mut total_w = 0.0_f64;

        for m in 0..m_cnt {
            for (ki, k) in (grid.k_lo..grid.k_hi).enumerate() {
                let idx = m * k_len + ki;
                let e0 = (re[m][k] * re[m][k] + im[m][k] * im[m][k]).sqrt();
                if m > 0 {
                    let (u, v) = unit_phasor_diff(re[m][k], im[m][k], re[m - 1][k], im[m - 1][k]);
                    let w =
                        e0.min((re[m - 1][k] * re[m - 1][k] + im[m - 1][k] * im[m - 1][k]).sqrt());
                    if_u[idx] = u;
                    if_v[idx] = v;
                    if_w[idx] = w;
                    total_w += w;
                }
                if ki > 0 {
                    let (u, v) = unit_phasor_diff(re[m][k], im[m][k], re[m][k - 1], im[m][k - 1]);
                    let w =
                        e0.min((re[m][k - 1] * re[m][k - 1] + im[m][k - 1] * im[m][k - 1]).sqrt());
                    gd_u[idx] = u;
                    gd_v[idx] = v;
                    gd_w[idx] = w;
                    total_w += w;
                }
            }
        }
        Ok(Self {
            grid,
            if_u,
            if_v,
            if_w,
            gd_u,
            gd_v,
            gd_w,
            total_w: total_w.max(1.0e-12),
        })
    }
}

/// Evaluates normalized `IF+GD` phase-coherence loss and accumulates exact Wirtinger STFT adjoint
/// into `grad_opt`.
pub fn eval_if_gd_loss_and_adjoint(
    cache: &IfGdTargetCache,
    pred: &[f64],
    grad_opt: Option<&mut [f64]>,
) -> f64 {
    if pred.len() != cache.grid.len || cache.grid.starts.len() < 2 {
        return 0.0;
    }
    let (re, im) = cache.grid.forward_frames(pred);
    let m_cnt = re.len();
    let half_p1 = cache.grid.plan.n_fft() / 2 + 1;
    let k_lo = cache.grid.k_lo;
    let k_hi = cache.grid.k_hi;
    let k_len = k_hi - k_lo;
    let need_grad = grad_opt.is_some();
    let inv_w = 1.0 / cache.total_w;

    let mut g_re = if need_grad {
        vec![vec![0.0_f64; half_p1]; m_cnt]
    } else {
        Vec::new()
    };
    let mut g_im = if need_grad {
        vec![vec![0.0_f64; half_p1]; m_cnt]
    } else {
        Vec::new()
    };
    let mut loss = 0.0_f64;

    for m in 0..m_cnt {
        for (ki, k) in (k_lo..k_hi).enumerate() {
            let idx = m * k_len + ki;
            if m > 0 {
                let (v, [dr0, di0, dr1, di1]) = accum_phasor_pair_grad(PhasorPairInput {
                    r0: re[m][k],
                    i0: im[m][k],
                    r1: re[m - 1][k],
                    i1: im[m - 1][k],
                    tgt_u: cache.if_u[idx],
                    tgt_v: cache.if_v[idx],
                    weight: cache.if_w[idx] * inv_w,
                });
                loss += v;
                if need_grad {
                    g_re[m][k] += dr0;
                    g_im[m][k] += di0;
                    g_re[m - 1][k] += dr1;
                    g_im[m - 1][k] += di1;
                }
            }
            if ki > 0 {
                let (v, [dr0, di0, dr1, di1]) = accum_phasor_pair_grad(PhasorPairInput {
                    r0: re[m][k],
                    i0: im[m][k],
                    r1: re[m][k - 1],
                    i1: im[m][k - 1],
                    tgt_u: cache.gd_u[idx],
                    tgt_v: cache.gd_v[idx],
                    weight: cache.gd_w[idx] * inv_w,
                });
                loss += v;
                if need_grad {
                    g_re[m][k] += dr0;
                    g_im[m][k] += di0;
                    g_re[m][k - 1] += dr1;
                    g_im[m][k - 1] += di1;
                }
            }
        }
    }

    if let Some(grad) = grad_opt {
        cache.grid.backward_frames(&g_re, &g_im, grad);
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_ot_and_if_gd_losses_match_central_fd() {
        let n = 1_400_usize;
        let target: Vec<f64> = (0..n)
            .map(|t| {
                0.5 * ((t as f64) * 0.045).sin()
                    + 0.3 * ((t as f64) * 0.21).cos()
                    + 0.2 * ((t as f64) * 0.92).sin()
            })
            .collect();
        let pred: Vec<f64> = target
            .iter()
            .enumerate()
            .map(|(t, &v)| 0.85 * v + 0.05 * ((t as f64) * 0.37).sin())
            .collect();

        let sot_cache = SpecOtTargetCache::new(&target, 64, 48_000.0, 256).unwrap();
        let ifgd_cache = IfGdTargetCache::new(&target, 64, 48_000.0, 256).unwrap();
        let (mut gs, mut gi) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let _ = eval_spec_ot_loss_and_adjoint(&sot_cache, &pred, Some(&mut gs));
        let _ = eval_if_gd_loss_and_adjoint(&ifgd_cache, &pred, Some(&mut gi));

        let eps = 1.0e-6_f64;
        for idx in [150, 512, 960] {
            let mut pp = pred.clone();
            let mut pm = pred.clone();
            pp[idx] += eps;
            pm[idx] -= eps;
            let fd_s = (eval_spec_ot_loss_and_adjoint(&sot_cache, &pp, None)
                - eval_spec_ot_loss_and_adjoint(&sot_cache, &pm, None))
                / (2.0 * eps);
            let fd_i = (eval_if_gd_loss_and_adjoint(&ifgd_cache, &pp, None)
                - eval_if_gd_loss_and_adjoint(&ifgd_cache, &pm, None))
                / (2.0 * eps);
            assert!(
                (gs[idx] - fd_s).abs() <= 1.0e-5 * gs[idx].abs().max(1.0e-4),
                "SpecOT idx={idx} ana={} fd={fd_s}",
                gs[idx]
            );
            assert!(
                (gi[idx] - fd_i).abs() <= 1.0e-5 * gi[idx].abs().max(1.0e-4),
                "IF+GD idx={idx} ana={} fd={fd_i}",
                gi[idx]
            );
        }
    }
}
