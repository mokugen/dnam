#![forbid(unsafe_code)]
//! 33-Band Analytic Gabor Envelope & Spectral Tilt Adjoint (`20 Hz .. 20 kHz`).
//!
//! Aligns its center frequencies directly with [`canonical_ltas_33_centers_hz`] so multi-scale
//! time-frequency envelope and LTAS supervision covers every 1/3-octave band without blind spots,
//! executing via [`GaborOperatorBackend`] (`dot_pair_f64` + `axpy_pair_f64`).

use std::f64::consts::PI;

use crate::metrics::companion::canonical_ltas_33_centers_hz;
use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

/// Number of canonical 1/3-octave Gabor wavelet filterbank bands (`20 Hz .. 20 kHz`).
pub const K_GABOR: usize = 33;

/// Precomputed quadrature Hann-windowed Gabor wavelet kernel for one 1/3-octave band.
#[derive(Clone, Debug)]
pub struct GaborBandSpec {
    pub center_hz: f64,
    pub win_len: usize,
    pub hop_len: usize,
    pub cos_k: Vec<f64>,
    pub sin_k: Vec<f64>,
}

impl GaborBandSpec {
    fn new(center_hz: f64, sample_rate_hz: f64) -> Self {
        let f_k = center_hz.clamp(20.0, 0.455 * sample_rate_hz);
        // Constant-Q bandwidth (~1/3 octave -> Q ~ 4.3 -> ~6.5 cycles inside Hann mainlobe).
        let raw_len = ((6.5 * sample_rate_hz / f_k).round() as usize).clamp(48, 4_096);
        let win_len = raw_len & !1;
        let hop_len = (win_len / 4).max(8);

        let mut w = vec![0.0_f64; win_len];
        let denom = (win_len - 1).max(1) as f64;
        let mut w_sum = 0.0_f64;
        for (i, slot) in w.iter_mut().enumerate() {
            let v = 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos());
            *slot = v;
            w_sum += v;
        }

        let omega = 2.0 * PI * f_k / sample_rate_hz;
        let half = 0.5 * denom;
        let mut cos_raw = Vec::with_capacity(win_len);
        let mut sin_raw = Vec::with_capacity(win_len);
        let (mut c_acc, mut s_acc) = (0.0_f64, 0.0_f64);
        for (i, &wi) in w.iter().enumerate() {
            let arg = omega * ((i as f64) - half);
            let (s, c) = arg.sin_cos();
            let wc = wi * c;
            let ws = wi * s;
            c_acc += wc;
            s_acc += ws;
            cos_raw.push(wc);
            sin_raw.push(ws);
        }
        let c_dc = c_acc / w_sum.max(1.0e-12);
        let s_dc = s_acc / w_sum.max(1.0e-12);

        let mut cos_k = Vec::with_capacity(win_len);
        let mut sin_k = Vec::with_capacity(win_len);
        let mut norm_sq = 0.0_f64;
        for i in 0..win_len {
            let cv = cos_raw[i] - w[i] * c_dc;
            let sv = sin_raw[i] - w[i] * s_dc;
            norm_sq += cv * cv + sv * sv;
            cos_k.push(cv);
            sin_k.push(sv);
        }
        let inv_norm = 1.0 / norm_sq.sqrt().max(1.0e-12);
        for i in 0..win_len {
            cos_k[i] *= inv_norm;
            sin_k[i] *= inv_norm;
        }

        Self {
            center_hz: f_k,
            win_len,
            hop_len,
            cos_k,
            sin_k,
        }
    }

    #[inline]
    fn num_frames(&self, len: usize, warmup: usize) -> usize {
        let avail = len.saturating_sub(warmup);
        if avail < self.win_len {
            0
        } else {
            1 + (avail - self.win_len) / self.hop_len
        }
    }

    #[inline]
    fn project_frames_with_backend<B: TrainerBackend>(
        &self,
        x: &[f64],
        warmup: usize,
        out_re: &mut [f64],
        out_im: &mut [f64],
    ) {
        B::project_frames(
            x,
            warmup,
            self.win_len,
            self.hop_len,
            &self.cos_k,
            &self.sin_k,
            out_re,
            out_im,
        );
    }
}

/// Complete 33-band constant-$Q$ analytic Gabor wavelet bank aligned to 1/3-octave LTAS centers.
#[derive(Clone, Debug)]
pub struct AnalyticGaborBank {
    pub bands: Vec<GaborBandSpec>,
}

impl AnalyticGaborBank {
    /// Constructs a 33-band Gabor bank covering canonical 1/3-octave centers at `sample_rate_hz`.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let centers = canonical_ltas_33_centers_hz();
        let bands = centers
            .iter()
            .map(|&fc| GaborBandSpec::new(fc, sample_rate_hz))
            .collect();
        Self { bands }
    }
}

/// Precomputed Gabor frame envelopes and band powers for one target channel.
#[derive(Clone, Debug)]
pub struct GaborTargetSummary {
    pub len: usize,
    pub warmup: usize,
    pub active_bands: usize,
    pub mean_band_pow: f64,
    pub n_frames: [usize; K_GABOR],
    pub band_offsets: [usize; K_GABOR],
    pub tgt_env: Vec<f64>,
    pub tgt_pow: [f64; K_GABOR],
}

/// Reusable scratch buffers for zero-allocation Gabor loss and adjoint accumulation.
#[derive(Clone, Debug, Default)]
pub struct GaborAdjointScratch {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
    pub d_re: Vec<f64>,
    pub d_im: Vec<f64>,
}

/// Precomputes target Gabor envelopes and mean band powers over `tgt[warmup..]`.
#[must_use]
pub fn build_gabor_target_summary(
    tgt: &[f64],
    warmup: usize,
    bank: &AnalyticGaborBank,
) -> GaborTargetSummary {
    let len = tgt.len();
    let w = warmup.min(len);
    let mut n_frames = [0usize; K_GABOR];
    let mut band_offsets = [0usize; K_GABOR];
    let mut total_frames = 0usize;
    let mut active_bands = 0usize;
    for (b, spec) in bank.bands.iter().enumerate() {
        band_offsets[b] = total_frames;
        let m_cnt = spec.num_frames(len, w);
        n_frames[b] = m_cnt;
        if m_cnt > 0 {
            active_bands += 1;
        }
        total_frames += m_cnt;
    }

    let mut tgt_env = vec![0.0_f64; total_frames];
    let mut tgt_pow = [0.0_f64; K_GABOR];
    let mut re = Vec::new();
    let mut im = Vec::new();
    let mut pow_acc = 0.0_f64;

    struct ProjectTask<'a> {
        spec: &'a GaborBandSpec,
        tgt: &'a [f64],
        w: usize,
        out_re: &'a mut [f64],
        out_im: &'a mut [f64],
    }
    impl TrainerTask<()> for ProjectTask<'_> {
        fn run<B: TrainerBackend>(self) {
            self.spec
                .project_frames_with_backend::<B>(self.tgt, self.w, self.out_re, self.out_im);
        }
    }
    let dyn_tr = DynamicTrainer::detect();

    for (b, spec) in bank.bands.iter().enumerate() {
        let m_cnt = n_frames[b];
        if m_cnt == 0 {
            continue;
        }
        re.resize(m_cnt, 0.0);
        im.resize(m_cnt, 0.0);
        dyn_tr.dispatch(ProjectTask {
            spec,
            tgt,
            w,
            out_re: &mut re[..m_cnt],
            out_im: &mut im[..m_cnt],
        });
        let off = band_offsets[b];
        let mut band_p_sum = 0.0_f64;
        for m in 0..m_cnt {
            let p = re[m] * re[m] + im[m] * im[m];
            band_p_sum += p;
            tgt_env[off + m] = (p + 1.0e-18).sqrt();
        }
        let pb = band_p_sum / (m_cnt as f64);
        tgt_pow[b] = pb;
        pow_acc += pb;
    }

    let mean_band_pow = if active_bands > 0 {
        (pow_acc / (active_bands as f64)).max(1.0e-12)
    } else {
        1.0
    };

    GaborTargetSummary {
        len,
        warmup: w,
        active_bands,
        mean_band_pow,
        n_frames,
        band_offsets,
        tgt_env,
        tgt_pow,
    }
}

/// Evaluates the 33-band Gabor envelope + LTAS loss under backend `B` and accumulates
/// the exact overlap-add reverse adjoint into `grad_dy`.
#[allow(clippy::too_many_arguments)]
pub fn eval_gabor_loss_and_adjoint_with_backend<B: TrainerBackend>(
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
        spec.project_frames_with_backend::<B>(y, summary.warmup, out_re, out_im);

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
            if scratch.d_re.len() < m_cnt {
                scratch.d_re.resize(m_cnt, 0.0);
                scratch.d_im.resize(m_cnt, 0.0);
            }
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
                scratch.d_re[m] = coeff * re;
                scratch.d_im[m] = coeff * im;
            }
            B::overlap_add_adjoint(
                dy,
                summary.warmup,
                spec.win_len,
                spec.hop_len,
                &spec.cos_k,
                &spec.sin_k,
                &scratch.d_re[..m_cnt],
                &scratch.d_im[..m_cnt],
            );
        }
    }

    loss
}

/// Evaluates the 33-band Gabor envelope + LTAS loss and accumulates the exact overlap-add
/// reverse adjoint directly into `dy`.
#[allow(clippy::too_many_arguments)]
pub fn eval_gabor_loss_and_adjoint(
    bank: &AnalyticGaborBank,
    summary: &GaborTargetSummary,
    y: &[f64],
    grad_dy: Option<&mut [f64]>,
    stem_w: f64,
    ch_scale: f64,
    env_weight: f64,
    ltas_weight: f64,
    scratch: &mut GaborAdjointScratch,
) -> f64 {
    struct EvalTask<'a> {
        bank: &'a AnalyticGaborBank,
        summary: &'a GaborTargetSummary,
        y: &'a [f64],
        grad_dy: Option<&'a mut [f64]>,
        stem_w: f64,
        ch_scale: f64,
        env_weight: f64,
        ltas_weight: f64,
        scratch: &'a mut GaborAdjointScratch,
    }
    impl TrainerTask<f64> for EvalTask<'_> {
        fn run<B: TrainerBackend>(self) -> f64 {
            eval_gabor_loss_and_adjoint_with_backend::<B>(
                self.bank,
                self.summary,
                self.y,
                self.grad_dy,
                self.stem_w,
                self.ch_scale,
                self.env_weight,
                self.ltas_weight,
                self.scratch,
            )
        }
    }
    DynamicTrainer::detect().dispatch(EvalTask {
        bank,
        summary,
        y,
        grad_dy,
        stem_w,
        ch_scale,
        env_weight,
        ltas_weight,
        scratch,
    })
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
        let centers = canonical_ltas_33_centers_hz();
        let make_sig = |phase_off: f64, gain: f64| -> Vec<f64> {
            let scale = gain / (centers.len() as f64).sqrt();
            (0..n)
                .map(|i| {
                    let t = (i as f64) / 48_000.0;
                    let mut s = 0.0_f64;
                    for (k, &fc) in centers.iter().enumerate() {
                        s += (2.0 * PI * fc * t + (k as f64) * 0.67 + phase_off).sin();
                    }
                    scale * s
                })
                .collect()
        };
        let tgt = make_sig(0.0, 0.45);
        let y = make_sig(0.35, 0.39);
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

        let dy_peak = dy.iter().map(|v| v.abs()).fold(1.0e-8, f64::max);
        let h = 1.0e-7_f64;
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
            let denom = g_fd.abs().max(dy[idx].abs()).max(dy_peak * 1.0e-2);
            let rel = (dy[idx] - g_fd).abs() / denom;
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
