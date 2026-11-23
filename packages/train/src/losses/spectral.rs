//! Multi-window 33-band constant-$Q$ `Gabor` wavelet loss, `TrainRunner::gabor_bezout_grad`
//! bridge, and exact overlap-add adjoint (`ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::PI;

use crate::error::TrainError;
use crate::gpu::abi::GaborBezoutUniforms;
use crate::math::{DynamicTrainer, TrainerMathBackend, TrainerTask};
use crate::runner::{GaborBezoutInput, GaborBezoutOutput, TrainRunner};

use super::ltas::canonical_ltas_33_centers_hz;

/// Number of canonical 1/3-octave Gabor bands.
pub const K_GABOR: usize = 33;
const _: () = assert!(K_GABOR == 33);

/// Constant-$Q$ quadrature Hann-windowed Gabor kernel for one 1/3-octave band.
#[derive(Clone, Debug)]
pub struct GaborBandKernel {
    /// Center frequency in Hz.
    pub center_hz: f64,
    /// Window length in samples (even).
    pub win_len: usize,
    /// Hop size in samples (`>= 8`).
    pub hop_len: usize,
    /// Cosine quadrature taps (`win_len`).
    pub cos_k: Vec<f64>,
    /// Sine quadrature taps (`win_len`).
    pub sin_k: Vec<f64>,
}

impl GaborBandKernel {
    fn new(center_hz: f64, sample_rate_hz: f64) -> Self {
        let f_k = center_hz.clamp(20.0, 0.455 * sample_rate_hz);
        let win_len = (((6.5 * sample_rate_hz / f_k).round() as usize).clamp(48, 4_096)) & !1;
        let hop_len = (win_len / 4).max(8);

        let mut win = vec![0.0_f64; win_len];
        let denom = (win_len - 1).max(1) as f64;
        let mut w_sum = 0.0_f64;
        for (i, w) in win.iter_mut().enumerate() {
            let v = 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos());
            *w = v;
            w_sum += v;
        }
        let omega = 2.0 * PI * f_k / sample_rate_hz;
        let half = 0.5 * denom;
        let (mut cos_raw, mut sin_raw) = (Vec::with_capacity(win_len), Vec::with_capacity(win_len));
        let (mut c_acc, mut s_acc) = (0.0_f64, 0.0_f64);
        for (i, &wi) in win.iter().enumerate() {
            let (s, c) = (omega * ((i as f64) - half)).sin_cos();
            c_acc += wi * c;
            s_acc += wi * s;
            cos_raw.push(wi * c);
            sin_raw.push(wi * s);
        }
        let c_dc = c_acc / w_sum.max(1.0e-12);
        let s_dc = s_acc / w_sum.max(1.0e-12);
        let (mut cos_k, mut sin_k) = (Vec::with_capacity(win_len), Vec::with_capacity(win_len));
        let mut norm_sq = 0.0_f64;
        for i in 0..win_len {
            let cv = cos_raw[i] - win[i] * c_dc;
            let sv = sin_raw[i] - win[i] * s_dc;
            norm_sq += cv * cv + sv * sv;
            cos_k.push(cv);
            sin_k.push(sv);
        }
        let inv = 1.0 / norm_sq.sqrt().max(1.0e-12);
        for i in 0..win_len {
            cos_k[i] *= inv;
            sin_k[i] *= inv;
        }
        Self {
            center_hz: f_k,
            win_len,
            hop_len,
            cos_k,
            sin_k,
        }
    }

    fn num_frames(&self, len: usize, warmup: usize) -> usize {
        let avail = len.saturating_sub(warmup);
        if avail < self.win_len {
            0
        } else {
            1 + (avail - self.win_len) / self.hop_len
        }
    }

    fn project_slice(&self, x: &[f64], warmup: usize, out_re: &mut [f64], out_im: &mut [f64]) {
        struct Task<'a>(
            &'a [f64],
            usize,
            usize,
            usize,
            &'a [f64],
            &'a [f64],
            &'a mut [f64],
            &'a mut [f64],
        );
        impl TrainerTask<()> for Task<'_> {
            fn run<B: TrainerMathBackend>(self) {
                for m in 0..self.6.len() {
                    let s = self.1 + m * self.2;
                    let (re, im) = B::dot_pair_f64(&self.0[s..s + self.3], self.4, self.5);
                    self.6[m] = re;
                    self.7[m] = im;
                }
            }
        }
        DynamicTrainer::detect().dispatch(Task(
            x,
            warmup,
            self.hop_len,
            self.win_len,
            &self.cos_k,
            &self.sin_k,
            out_re,
            out_im,
        ));
    }

    fn overlap_add_slice(&self, d_re: &[f64], d_im: &[f64], warmup: usize, grad: &mut [f64]) {
        struct Task<'a>(
            &'a [f64],
            &'a [f64],
            usize,
            usize,
            usize,
            &'a [f64],
            &'a [f64],
            &'a mut [f64],
        );
        impl TrainerTask<()> for Task<'_> {
            fn run<B: TrainerMathBackend>(self) {
                for m in 0..self.0.len() {
                    let s = self.2 + m * self.3;
                    B::axpy_pair_f64(
                        &mut self.7[s..s + self.4],
                        self.0[m],
                        self.5,
                        self.1[m],
                        self.6,
                    );
                }
            }
        }
        DynamicTrainer::detect().dispatch(Task(
            d_re,
            d_im,
            warmup,
            self.hop_len,
            self.win_len,
            &self.cos_k,
            &self.sin_k,
            grad,
        ));
    }
}

/// 33-band constant-$Q$ analytic Gabor wavelet bank aligned to 1/3-octave LTAS centers.
#[derive(Clone, Debug)]
pub struct AnalyticGaborBank {
    /// Per-band constant-$Q$ quadrature kernels (`33`).
    pub bands: Vec<GaborBandKernel>,
}

impl AnalyticGaborBank {
    /// Constructs a 33-band Gabor wavelet filterbank at `sample_rate_hz`.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let bands = canonical_ltas_33_centers_hz()
            .into_iter()
            .map(|fc| GaborBandKernel::new(fc, sample_rate_hz))
            .collect();
        Self { bands }
    }
}

/// Precomputed Gabor target envelopes and powers.
#[derive(Clone, Debug)]
pub struct GaborTargetCache {
    len: usize,
    warmup: usize,
    active_bands: usize,
    mean_band_pow: f64,
    n_frames: [usize; K_GABOR],
    band_snr_w: [f64; K_GABOR],
    band_noise_p: [f64; K_GABOR],
    tgt_re: Vec<Vec<f64>>,
    tgt_im: Vec<Vec<f64>>,
    tgt_env: Vec<Vec<f64>>,
}

impl GaborTargetCache {
    /// Precomputes constant-$Q$ Gabor wavelet envelopes across `tgt[warmup..]`.
    #[must_use]
    pub fn new(tgt: &[f64], warmup: usize, bank: &AnalyticGaborBank) -> Self {
        Self::new_with_noise(tgt, None, warmup, bank)
    }

    /// Precomputes constant-$Q$ Gabor wavelet envelopes alongside per-band repeatability
    /// noise-floor weights and smoothing floors (`ref[qual.eval.thermal-hiss]`).
    #[must_use]
    pub fn new_with_noise(
        tgt: &[f64],
        noise: Option<&[f64]>,
        warmup: usize,
        bank: &AnalyticGaborBank,
    ) -> Self {
        let len = tgt.len();
        let w = warmup.min(len);
        let noise_valid =
            noise.filter(|ns| ns.len() >= len && ns[..len].iter().all(|v| v.is_finite()));
        let mut n_frames = [0_usize; K_GABOR];
        let mut band_snr_w = [1.0_f64; K_GABOR];
        let mut band_noise_p = [0.0_f64; K_GABOR];
        let mut tgt_re = vec![Vec::new(); K_GABOR];
        let mut tgt_im = vec![Vec::new(); K_GABOR];
        let mut tgt_env = vec![Vec::new(); K_GABOR];
        let mut active_bands = 0_usize;
        let mut pow_acc = 0.0_f64;

        for (b, spec) in bank.bands.iter().enumerate() {
            let m_cnt = spec.num_frames(len, w);
            n_frames[b] = m_cnt;
            if m_cnt == 0 {
                continue;
            }
            active_bands += 1;
            let mut re = vec![0.0_f64; m_cnt];
            let mut im = vec![0.0_f64; m_cnt];
            spec.project_slice(tgt, w, &mut re, &mut im);
            let mut env = vec![0.0_f64; m_cnt];
            let mut sum_p = 0.0_f64;
            for m in 0..m_cnt {
                let p = re[m] * re[m] + im[m] * im[m];
                sum_p += p;
                env[m] = (p + 1.0e-18).sqrt();
            }
            let mean_tgt_p = sum_p / (m_cnt as f64);
            pow_acc += mean_tgt_p;
            if let Some(ns) = noise_valid {
                let mut nre = vec![0.0_f64; m_cnt];
                let mut nim = vec![0.0_f64; m_cnt];
                spec.project_slice(&ns[..len], w, &mut nre, &mut nim);
                let n_pow = nre
                    .iter()
                    .zip(&nim)
                    .map(|(a, c)| a * a + c * c)
                    .sum::<f64>()
                    / (m_cnt as f64);
                band_noise_p[b] = n_pow;
                band_snr_w[b] = super::noise_floor::snr_weight(mean_tgt_p, n_pow);
            }
            tgt_re[b] = re;
            tgt_im[b] = im;
            tgt_env[b] = env;
        }
        let mean_band_pow = (pow_acc / (active_bands.max(1) as f64)).max(1.0e-12);
        Self {
            len,
            warmup: w,
            active_bands,
            mean_band_pow,
            n_frames,
            band_snr_w,
            band_noise_p,
            tgt_re,
            tgt_im,
            tgt_env,
        }
    }
}

/// Evaluates 33-band Gabor wavelet envelope + spectral convergence loss and accumulates exact
/// overlap-add gradient into `grad_opt`.
pub fn eval_gabor_loss_and_adjoint(
    bank: &AnalyticGaborBank,
    cache: &GaborTargetCache,
    pred: &[f64],
    mut grad_opt: Option<&mut [f64]>,
) -> f64 {
    if pred.len() != cache.len || cache.active_bands == 0 {
        return 0.0;
    }
    let norm_band = 1.0 / ((cache.active_bands as f64) * cache.mean_band_pow);
    let mut loss = 0.0_f64;
    for (b, spec) in bank.bands.iter().enumerate() {
        let m_cnt = cache.n_frames[b];
        if m_cnt == 0 {
            continue;
        }
        let mut re = vec![0.0_f64; m_cnt];
        let mut im = vec![0.0_f64; m_cnt];
        spec.project_slice(pred, cache.warmup, &mut re, &mut im);
        let inv_m = norm_band / (m_cnt as f64);
        let w_b = cache.band_snr_w[b] * inv_m;
        let n_floor = cache.band_noise_p[b];
        let mut d_re = vec![0.0_f64; m_cnt];
        let mut d_im = vec![0.0_f64; m_cnt];
        for m in 0..m_cnt {
            let e_hat = (re[m] * re[m] + im[m] * im[m] + 1.0e-18).sqrt();
            let diff_e = e_hat - cache.tgt_env[b][m];
            let dr = re[m] - cache.tgt_re[b][m];
            let di = im[m] - cache.tgt_im[b][m];
            let res_sq = diff_e * diff_e + 0.25 * (dr * dr + di * di);
            let (e_eff, d_e) = super::noise_floor::smooth_excess_sq(res_sq, n_floor);
            loss += w_b * e_eff;
            if grad_opt.is_some() {
                let s_scale = w_b * d_e;
                let s_env = 2.0 * s_scale * diff_e / e_hat;
                d_re[m] = s_env * re[m] + 0.5 * s_scale * dr;
                d_im[m] = s_env * im[m] + 0.5 * s_scale * di;
            }
        }
        if let Some(grad) = grad_opt.as_deref_mut() {
            spec.overlap_add_slice(&d_re, &d_im, cache.warmup, grad);
        }
    }
    loss
}

/// Bridges stereo predicted/target capsules to GPU/CPU `TrainRunner::gabor_bezout_grad`.
pub fn eval_runner_gabor_bezout_bridge(
    runner: &TrainRunner,
    uniforms: GaborBezoutUniforms,
    pred_audio: &[[f32; 2]],
    target_audio: &[[f32; 2]],
    bezout_taps: &[[f32; 4]],
) -> Result<GaborBezoutOutput, TrainError> {
    runner.gabor_bezout_grad(&GaborBezoutInput {
        uniforms,
        pred_audio,
        target_audio,
        bezout_taps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gabor_33_loss_matches_central_fd_and_runner_bridge() {
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

        let bank = AnalyticGaborBank::new(48_000.0);
        let gab_cache = GaborTargetCache::new(&target, 64, &bank);
        let mut gg = vec![0.0_f64; n];
        let _ = eval_gabor_loss_and_adjoint(&bank, &gab_cache, &pred, Some(&mut gg));

        let eps = 1.0e-6_f64;
        for idx in [150, 512, 960] {
            let mut pp = pred.clone();
            let mut pm = pred.clone();
            pp[idx] += eps;
            pm[idx] -= eps;
            let fd_g = (eval_gabor_loss_and_adjoint(&bank, &gab_cache, &pp, None)
                - eval_gabor_loss_and_adjoint(&bank, &gab_cache, &pm, None))
                / (2.0 * eps);
            assert!(
                (gg[idx] - fd_g).abs() <= 1.0e-5 * gg[idx].abs().max(1.0e-4),
                "Gabor idx={idx} ana={} fd={fd_g}",
                gg[idx]
            );
        }

        let runner = TrainRunner::cpu();
        let stereo_p = vec![[0.2_f32, -0.1]; 64];
        let br = eval_runner_gabor_bezout_bridge(
            &runner,
            GaborBezoutUniforms {
                num_frames: 64,
                fft_size: 64,
                hop_size: 64,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 0.7,
                w_hf: 0.4,
                eps_noise: 1.0e-5,
            },
            &stereo_p,
            &stereo_p,
            &[[1.0; 4]; 64],
        )
        .unwrap();
        assert_eq!(br.band_losses.len(), 1);

        // Noisy Gabor cache reduces pure-noise weight and preserves exact FD parity.
        let noise: Vec<f64> = (0..n).map(|i| 0.02 * ((i as f64) * 0.43).cos()).collect();
        let noisy_gab = GaborTargetCache::new_with_noise(&target, Some(&noise), 64, &bank);
        let mut g_noisy = vec![0.0_f64; n];
        let v_noisy = eval_gabor_loss_and_adjoint(&bank, &noisy_gab, &pred, Some(&mut g_noisy));
        assert!(v_noisy.is_finite());
        let mut pp = pred.clone();
        let mut pm = pred.clone();
        pp[512] += eps;
        pm[512] -= eps;
        let fd_noisy = (eval_gabor_loss_and_adjoint(&bank, &noisy_gab, &pp, None)
            - eval_gabor_loss_and_adjoint(&bank, &noisy_gab, &pm, None))
            / (2.0 * eps);
        assert!(
            (g_noisy[512] - fd_noisy).abs() <= 1.0e-5 * g_noisy[512].abs().max(1.0e-4),
            "Gabor noisy idx=512 ana={} fd={fd_noisy}",
            g_noisy[512]
        );
    }
}
