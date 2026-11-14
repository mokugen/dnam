//! SIMD-accelerated Constant-$Q$ Analytic Gabor wavelet operators (`24 Hz .. 17.5 kHz`).

use dnam_engine::math::trainer::TrainerMathBackend;

/// Hardware-abstracted Gabor wavelet projection and overlap-add adjoint operator (`Op-Gabor`).
pub trait GaborOperatorBackend: TrainerMathBackend {
    /// Projects `x[warmup..]` onto a cosine/sine Gabor atom pair in a single pass.
    #[allow(clippy::too_many_arguments)]
    fn project_frames(
        x: &[f64],
        warmup: usize,
        win_len: usize,
        hop_len: usize,
        cos_k: &[f64],
        sin_k: &[f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
    );

    /// Accumulates weighted cosine/sine Gabor adjoint contributions directly into `dy`.
    #[allow(clippy::too_many_arguments)]
    fn overlap_add_adjoint(
        dy: &mut [f64],
        warmup: usize,
        win_len: usize,
        hop_len: usize,
        cos_k: &[f64],
        sin_k: &[f64],
        d_re: &[f64],
        d_im: &[f64],
    );
}

impl<M: TrainerMathBackend> GaborOperatorBackend for M {
    #[inline]
    fn project_frames(
        x: &[f64],
        warmup: usize,
        win_len: usize,
        hop_len: usize,
        cos_k: &[f64],
        sin_k: &[f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
    ) {
        let m_cnt = out_re.len().min(out_im.len());
        let w = warmup.min(x.len());
        let c = &cos_k[..win_len];
        let s = &sin_k[..win_len];
        for m in 0..m_cnt {
            let start = w + m * hop_len;
            let seg = &x[start..start + win_len];
            let (re, im) = Self::dot_pair_f64(seg, c, s);
            out_re[m] = re;
            out_im[m] = im;
        }
    }

    #[inline]
    fn overlap_add_adjoint(
        dy: &mut [f64],
        warmup: usize,
        win_len: usize,
        hop_len: usize,
        cos_k: &[f64],
        sin_k: &[f64],
        d_re: &[f64],
        d_im: &[f64],
    ) {
        let m_cnt = d_re.len().min(d_im.len());
        let c = &cos_k[..win_len];
        let s = &sin_k[..win_len];
        for m in 0..m_cnt {
            let start = warmup + m * hop_len;
            Self::axpy_pair_f64(&mut dy[start..start + win_len], d_re[m], c, d_im[m], s);
        }
    }
}
