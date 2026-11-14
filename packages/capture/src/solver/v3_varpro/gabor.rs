#![forbid(unsafe_code)]
//! Hop-decimated Constant-$Q$ Analytic Gabor Wavelet Bank (`24 Hz .. 17.5 kHz`).
//!
//! Replaces recursive IIR filter banks in [`super::model_f64`] and [`super::basis`] with
//! zero-state, linear-phase, DC-admissible Hann-modulated complex atoms that yield exact
//! overlap-add reverse adjoints, phase-invariant envelopes, stopband LTAS gradients, and
//! closed-form `VarPro` quadrature Gram constraints with zero external FFT dependencies.

use std::f64::consts::PI;

/// Number of log-spaced analytic Gabor bands spanning `24 Hz .. 17,500 Hz`.
pub const K_GABOR: usize = 24;

#[derive(Clone, Debug)]
pub struct GaborBandSpec {
    pub center_hz: f64,
    pub win_len: usize,
    pub hop_len: usize,
    pub cos_k: Vec<f64>,
    pub sin_k: Vec<f64>,
}

/// 24-band Constant-$Q$ Analytic Gabor filterbank (`4.0` cycles nominal, `66.7%` COLA hop).
#[derive(Clone, Debug)]
pub struct AnalyticGaborBank {
    pub sample_rate_hz: f64,
    pub bands: Vec<GaborBandSpec>,
}

/// Precomputed per-stem target Gabor spectra, envelopes, and `VarPro` band weights.
#[derive(Clone, Debug, Default)]
pub struct GaborTargetSummary {
    pub warmup: usize,
    pub len: usize,
    pub p_sig_bar: f64,
    pub active_bands: usize,
    pub mean_band_pow: f64,
    pub n_frames: [usize; K_GABOR],
    pub band_offsets: [usize; K_GABOR],
    pub tgt_re: Vec<f64>,
    pub tgt_im: Vec<f64>,
    pub tgt_env: Vec<f64>,
    pub tgt_pow: [f64; K_GABOR],
    pub varpro_band_w: [f64; K_GABOR],
}

/// Reusable scratch buffers for [`eval_gabor_loss_and_adjoint`].
#[derive(Clone, Debug, Default)]
pub struct GaborAdjointScratch {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

#[inline(always)]
pub(crate) fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let chunks = n / 4;
    for c in 0..chunks {
        let i = c * 4;
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..n {
        s += a[i] * b[i];
    }
    s
}

impl GaborBandSpec {
    /// Number of complete windows of this band fitting in `x[warmup..len]`.
    #[inline(always)]
    #[must_use]
    pub fn frame_count(&self, len: usize, warmup: usize) -> usize {
        let w = warmup.min(len);
        if len >= w + self.win_len {
            (len - w - self.win_len) / self.hop_len + 1
        } else {
            0
        }
    }

    /// Projects `x` onto this band's cosine/sine atoms into `out_re` and `out_im`.
    pub fn project_frames(&self, x: &[f64], warmup: usize, out_re: &mut [f64], out_im: &mut [f64]) {
        let m_cnt = out_re.len().min(out_im.len());
        let w = warmup.min(x.len());
        for m in 0..m_cnt {
            let start = w + m * self.hop_len;
            let seg = &x[start..start + self.win_len];
            out_re[m] = dot_slice(seg, &self.cos_k);
            out_im[m] = dot_slice(seg, &self.sin_k);
        }
    }
}

impl AnalyticGaborBank {
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let sr = sample_rate_hz.max(8_000.0);
        let log_lo = 24.0_f64.ln();
        let log_hi = (17_500.0_f64.min(0.45 * sr)).max(24.5).ln();
        let mut bands = Vec::with_capacity(K_GABOR);
        for b in 0..K_GABOR {
            let frac = (b as f64) / ((K_GABOR - 1) as f64);
            let center_hz = (log_lo + (log_hi - log_lo) * frac).exp();
            let raw_len = (4.0 * sr / center_hz).round() as usize;
            let win_len = raw_len.clamp(48, 3_600);
            let hop_len = (win_len / 3).max(16);
            let denom_w = (win_len - 1).max(1) as f64;
            let mut w = Vec::with_capacity(win_len);
            let mut c_raw = Vec::with_capacity(win_len);
            let mut s_raw = Vec::with_capacity(win_len);
            let mut w_sum = 0.0_f64;
            let mut c_sum = 0.0_f64;
            let mut s_sum = 0.0_f64;
            for m in 0..win_len {
                let win = 0.5 * (1.0 - (2.0 * PI * (m as f64) / denom_w).cos());
                let phase = 2.0 * PI * center_hz * ((m as f64) - 0.5 * denom_w) / sr;
                let cv = phase.cos();
                let sv = phase.sin();
                w.push(win);
                c_raw.push(cv);
                s_raw.push(sv);
                w_sum += win;
                c_sum += win * cv;
                s_sum += win * sv;
            }
            let mu_c = c_sum / w_sum.max(1.0e-12);
            let mu_s = s_sum / w_sum.max(1.0e-12);
            let mut cos_k = Vec::with_capacity(win_len);
            let mut sin_k = Vec::with_capacity(win_len);
            let mut norm_sq = 0.0_f64;
            for m in 0..win_len {
                let ca = w[m] * (c_raw[m] - mu_c);
                let sa = w[m] * (s_raw[m] - mu_s);
                cos_k.push(ca);
                sin_k.push(sa);
                norm_sq += ca * ca + sa * sa;
            }
            let inv_norm = norm_sq.sqrt().max(1.0e-12).recip();
            for m in 0..win_len {
                cos_k[m] *= inv_norm;
                sin_k[m] *= inv_norm;
            }
            bands.push(GaborBandSpec {
                center_hz,
                win_len,
                hop_len,
                cos_k,
                sin_k,
            });
        }
        Self {
            sample_rate_hz: sr,
            bands,
        }
    }
}

/// Builds a [`GaborTargetSummary`] from `tgt[warmup..]` across all 24 analytic Gabor bands.
#[must_use]
pub fn build_gabor_target_summary(
    tgt: &[f64],
    warmup: usize,
    bank: &AnalyticGaborBank,
) -> GaborTargetSummary {
    let n = tgt.len();
    let w = warmup.min(n);
    let active_len = (n - w).max(1) as f64;
    let p_sig_bar = (tgt[w..n].iter().map(|&v| v * v).sum::<f64>() / active_len).max(1.0e-10);

    let mut n_frames = [0usize; K_GABOR];
    let mut band_offsets = [0usize; K_GABOR];
    let mut total_frames = 0usize;
    for (b, spec) in bank.bands.iter().enumerate() {
        let m_cnt = spec.frame_count(n, w);
        n_frames[b] = m_cnt;
        band_offsets[b] = total_frames;
        total_frames += m_cnt;
    }

    let mut tgt_re = vec![0.0_f64; total_frames];
    let mut tgt_im = vec![0.0_f64; total_frames];
    let mut tgt_env = vec![0.0_f64; total_frames];
    let mut tgt_pow = [0.0_f64; K_GABOR];
    let mut active_bands = 0usize;
    let mut sum_band_pow = 0.0_f64;

    for (b, spec) in bank.bands.iter().enumerate() {
        let m_cnt = n_frames[b];
        if m_cnt == 0 {
            continue;
        }
        let off = band_offsets[b];
        spec.project_frames(
            tgt,
            w,
            &mut tgt_re[off..off + m_cnt],
            &mut tgt_im[off..off + m_cnt],
        );
        let mut p_sum = 0.0_f64;
        for m in 0..m_cnt {
            let re = tgt_re[off + m];
            let im = tgt_im[off + m];
            let p = re * re + im * im;
            tgt_env[off + m] = (p + 1.0e-18).sqrt();
            p_sum += p;
        }
        let pb = p_sum / (m_cnt as f64);
        tgt_pow[b] = pb;
        active_bands += 1;
        sum_band_pow += pb;
    }

    let k_act = active_bands.max(1) as f64;
    let mean_band_pow = (sum_band_pow / k_act).max(1.0e-10);
    let vp_floor = (1.0e-5 * mean_band_pow).max(1.0e-12);
    let mut varpro_band_w = [0.0_f64; K_GABOR];
    for b in 0..K_GABOR {
        let m_cnt = n_frames[b];
        if m_cnt > 0 {
            let denom = tgt_pow[b].max(vp_floor);
            varpro_band_w[b] = 0.04 / (k_act * (m_cnt as f64) * denom);
        }
    }

    GaborTargetSummary {
        warmup: w,
        len: n,
        p_sig_bar,
        active_bands,
        mean_band_pow,
        n_frames,
        band_offsets,
        tgt_re,
        tgt_im,
        tgt_env,
        tgt_pow,
        varpro_band_w,
    }
}

/// Evaluates the 24-band Gabor envelope + LTAS loss and accumulates the exact overlap-add
/// reverse adjoint directly into `dy`.
#[allow(clippy::too_many_arguments)]
pub fn eval_gabor_loss_and_adjoint(
    bank: &AnalyticGaborBank,
    summary: &GaborTargetSummary,
    y: &[f64],
    mut grad_dy: Option<&mut [f64]>,
    stem_w: f64,
    ch_scale: f64,
    env_weight: f64,
    ltas_weight: f64,
    scratch: &mut GaborAdjointScratch,
) -> f64 {
    if (env_weight <= 0.0 && ltas_weight <= 0.0)
        || summary.active_bands == 0
        || y.len() != summary.len
        || grad_dy.as_deref().is_some_and(|dy| dy.len() != summary.len)
    {
        return 0.0;
    }
    let total_frames = summary.band_offsets[K_GABOR - 1] + summary.n_frames[K_GABOR - 1];
    if total_frames == 0 {
        return 0.0;
    }
    scratch.re.resize(total_frames, 0.0);
    scratch.im.resize(total_frames, 0.0);

    let k_act = summary.active_bands.max(1) as f64;
    let scale_env = (stem_w * ch_scale * env_weight) / (k_act * summary.mean_band_pow);
    let scale_ltas = (stem_w * ch_scale * ltas_weight) / k_act;
    let ltas_floor = (1.0e-7 * summary.mean_band_pow).max(1.0e-12);
    let need_grad = grad_dy.is_some();
    let mut loss = 0.0_f64;

    for (b, spec) in bank.bands.iter().enumerate() {
        let m_cnt = summary.n_frames[b];
        if m_cnt == 0 {
            continue;
        }
        let off = summary.band_offsets[b];
        let m_f = m_cnt as f64;
        let out_re = &mut scratch.re[off..off + m_cnt];
        let out_im = &mut scratch.im[off..off + m_cnt];
        spec.project_frames(y, summary.warmup, out_re, out_im);

        let mut pow_sum = 0.0_f64;
        let mut env_sum = 0.0_f64;
        for m in 0..m_cnt {
            let re = out_re[m];
            let im = out_im[m];
            let p = re * re + im * im;
            pow_sum += p;
            if env_weight > 0.0 {
                let e_y = (p + 1.0e-18).sqrt();
                let diff_e = e_y - summary.tgt_env[off + m];
                env_sum += diff_e * diff_e;
            }
        }
        if env_weight > 0.0 {
            loss += (scale_env / m_f) * env_sum;
        }
        let py_b = pow_sum / m_f;
        let pt_b = summary.tgt_pow[b];
        let log_diff = (py_b + ltas_floor).ln() - (pt_b + ltas_floor).ln();
        if ltas_weight > 0.0 {
            loss += scale_ltas * log_diff * log_diff;
        }

        if need_grad {
            let dy = grad_dy.as_deref_mut().expect("checked is_some");
            let d_ltas_over_m = if ltas_weight > 0.0 {
                scale_ltas * 4.0 * log_diff / ((py_b + ltas_floor) * m_f)
            } else {
                0.0
            };
            let d_env_scale = if env_weight > 0.0 {
                2.0 * scale_env / m_f
            } else {
                0.0
            };

            for m in 0..m_cnt {
                let re = out_re[m];
                let im = out_im[m];
                let e_y = (re * re + im * im + 1.0e-18).sqrt();
                let coeff = d_ltas_over_m
                    + if env_weight > 0.0 {
                        d_env_scale * (e_y - summary.tgt_env[off + m]) / e_y
                    } else {
                        0.0
                    };
                let d_re = coeff * re;
                let d_im = coeff * im;
                let start = summary.warmup + m * spec.hop_len;
                for u in 0..spec.win_len {
                    dy[start + u] += d_re * spec.cos_k[u] + d_im * spec.sin_k[u];
                }
            }
        }
    }

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gabor_atoms_zero_dc_and_exact_adjoint_vs_fd() {
        let bank = AnalyticGaborBank::new(48_000.0);
        assert_eq!(bank.bands.len(), K_GABOR);
        for (b, spec) in bank.bands.iter().enumerate() {
            let dc_c: f64 = spec.cos_k.iter().sum();
            let dc_s: f64 = spec.sin_k.iter().sum();
            let energy: f64 = spec
                .cos_k
                .iter()
                .zip(&spec.sin_k)
                .map(|(c, s)| c * c + s * s)
                .sum();
            assert!(dc_c.abs() < 1.0e-12, "band {b} cos DC leak: {dc_c}");
            assert!(dc_s.abs() < 1.0e-12, "band {b} sin DC leak: {dc_s}");
            assert!((energy - 1.0).abs() < 1.0e-12, "band {b} norm: {energy}");
        }

        let n = 320;
        let warmup = 24;
        let tgt: Vec<f64> = (0..n)
            .map(|i| 0.4 * (i as f64 * 0.13).sin() + 0.18 * (i as f64 * 0.71).cos())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.32 * (i as f64 * 0.13 + 0.4).sin() + 0.21 * (i as f64 * 0.85).cos() + 0.05)
            .collect();
        let summary = build_gabor_target_summary(&tgt, warmup, &bank);
        assert!(summary.active_bands > 0);
        let mut scratch = GaborAdjointScratch::default();
        let mut dy = vec![0.0_f64; n];
        let _ = eval_gabor_loss_and_adjoint(
            &bank,
            &summary,
            &y,
            Some(&mut dy),
            1.0,
            0.5,
            0.35,
            0.05,
            &mut scratch,
        );

        let h = 1.0e-6_f64;
        let mut max_rel = 0.0_f64;
        for idx in (0..n).step_by(7) {
            let mut yp = y.clone();
            let mut ym = y.clone();
            yp[idx] += h;
            ym[idx] -= h;
            let lp = eval_gabor_loss_and_adjoint(
                &bank,
                &summary,
                &yp,
                None,
                1.0,
                0.5,
                0.35,
                0.05,
                &mut scratch,
            );
            let lm = eval_gabor_loss_and_adjoint(
                &bank,
                &summary,
                &ym,
                None,
                1.0,
                0.5,
                0.35,
                0.05,
                &mut scratch,
            );
            let g_fd = (lp - lm) / (2.0 * h);
            let rel = (dy[idx] - g_fd).abs() / g_fd.abs().max(dy[idx].abs()).max(1.0e-10);
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1.0e-5,
                "sample {idx} ana={} fd={} rel={rel}",
                dy[idx],
                g_fd
            );
        }
        assert!(max_rel < 1.0e-5);
    }
}
