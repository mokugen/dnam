//! Deterministic CPU reference runners for `prescreen_population` and `gabor_bezout_grad`
//! (Mode A causal Bézout FIR and Mode B radix-2 DIT FFT + DIF conjugate inverse pullback
//! using `dnam_kernels::algebraic_unit_rotor_f32`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::f32::consts::FRAC_1_SQRT_2;

use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_SAG_GAIN, SonaParaLruDuhemCoeffs4, TptSvfCoeffs4,
    algebraic_unit_rotor_f32, apply_stereo_link_vec4_pair, canonicalize_zero_f32_bits,
    const_f32_pow2, eval_poly_adaa1_odd_even_vec4, prewarp_tan_pi_f_over_fs_f32,
    step_paralru_duhem_vec4, step_tpt_svf_vec4,
};

use crate::error::TrainError;
use crate::gpu::abi::{PRESCREEN_SEED_PARAM_STRIDE, TILE_FRAMES};
use crate::runner::types::{GaborBezoutInput, GaborBezoutOutput, PrescreenInput};

const Q32_TO_RAD_F32: f32 = core::f32::consts::TAU * const_f32_pow2(-32);
const MIN_EPS_NOISE: f32 = 1.0e-12;
const MAX_BEZOUT_TAPS: usize = 64;
const IEEE_F32_EXP_MASK: u32 = 0x7f80_0000;

#[inline(always)]
const fn is_finite_f32(x: f32) -> bool {
    (x.to_bits() & IEEE_F32_EXP_MASK) != IEEE_F32_EXP_MASK
}

#[inline(always)]
fn clamp_f32_safe(x: f32, lo: f32, hi: f32) -> f32 {
    if is_finite_f32(x) {
        canonicalize_zero_f32_bits(x.clamp(lo, hi))
    } else if x.is_nan() {
        lo
    } else if x > 0.0 {
        hi
    } else {
        lo
    }
}

struct PrescreenQuad {
    svf: TptSvfCoeffs4<GpuVec4>,
    two_svf_g: f32,
    w_lp: f32,
    w_bp: f32,
    w_hp: f32,
    w_dir: f32,
    lru: SonaParaLruDuhemCoeffs4<GpuVec4>,
    rect_sign: f32,
    sag_depth: f32,
    adaa_drive: f32,
    lru_bias_drive: f32,
}

fn load_prescreen_quad(
    seed_params: &[f32],
    seed_idx: usize,
    quad_idx: usize,
    num_seeds: usize,
    num_quads: usize,
) -> PrescreenQuad {
    let per_quad_len = num_seeds * num_quads * PRESCREEN_SEED_PARAM_STRIDE;
    let base = if seed_params.len() >= per_quad_len {
        (seed_idx * num_quads + quad_idx) * PRESCREEN_SEED_PARAM_STRIDE
    } else {
        seed_idx * PRESCREEN_SEED_PARAM_STRIDE
    };
    let read = |off: usize| seed_params[(base + off).min(seed_params.len() - 1)];
    let fc_ratio = if is_finite_f32(read(0)) {
        read(0)
    } else {
        1.0e-3
    };
    let g = prewarp_tan_pi_f_over_fs_f32(fc_ratio);
    let q_val = if is_finite_f32(read(1)) {
        clamp_f32_safe(read(1), 0.05, 64.0)
    } else {
        FRAC_1_SQRT_2
    };
    let k = 1.0 / q_val;
    PrescreenQuad {
        svf: TptSvfCoeffs4::from_gk(GpuVec4::splat(g), GpuVec4::splat(k)),
        two_svf_g: clamp_f32_safe(g + g, 0.0, 128.0),
        w_lp: canonicalize_zero_f32_bits(read(2)),
        w_bp: canonicalize_zero_f32_bits(read(3)),
        w_hp: canonicalize_zero_f32_bits(read(4)),
        w_dir: canonicalize_zero_f32_bits(read(5)),
        lru: SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(canonicalize_zero_f32_bits(read(6))),
            GpuVec4::splat(canonicalize_zero_f32_bits(read(7))),
            GpuVec4::splat(canonicalize_zero_f32_bits(read(8))),
            GpuVec4::splat(canonicalize_zero_f32_bits(read(9))),
            GpuVec4::splat(canonicalize_zero_f32_bits(read(10))),
            GpuVec4::splat(canonicalize_zero_f32_bits(read(11))),
        ),
        rect_sign: clamp_f32_safe(read(12), -1.0, 1.0),
        sag_depth: clamp_f32_safe(read(13), -4.0, 4.0),
        adaa_drive: clamp_f32_safe(read(14), -64.0, 64.0),
        lru_bias_drive: clamp_f32_safe(read(15), -64.0, 64.0),
    }
}

/// Deterministic CPU reference implementation of `prescreen_population`.
pub fn run_prescreen_population_cpu(input: &PrescreenInput<'_>) -> Result<Vec<f32>, TrainError> {
    let (num_seeds, num_frames, num_quads, num_stems) = input.validate()?;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let eps_floor =
        if is_finite_f32(input.uniforms.eps_floor) && input.uniforms.eps_floor > MIN_EPS_NOISE {
            input.uniforms.eps_floor
        } else {
            MIN_EPS_NOISE
        };
    let mut losses = Vec::with_capacity(num_stems * num_seeds * num_tiles);

    for s in 0..num_stems {
        for seed in 0..num_seeds {
            for tile in 0..num_tiles {
                let start = tile * TILE_FRAMES;
                let end = (start + TILE_FRAMES).min(num_frames);
                let mut pred = vec![[0.0_f32; 2]; end - start];
                for q in 0..num_quads {
                    let qp = load_prescreen_quad(input.seed_params, seed, q, num_seeds, num_quads);
                    let (mut s1_l, mut s2_l, mut s1_r, mut s2_r) = (
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                    );
                    let (mut v_l, mut v_r, mut rp_l, mut rp_r) = (
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                    );
                    for (lt, frame) in (start..end).enumerate() {
                        let [in_l, in_r] = input.input_audio[s * num_frames + frame];
                        let (in_l, in_r) = (
                            canonicalize_zero_f32_bits(in_l),
                            canonicalize_zero_f32_bits(in_r),
                        );
                        let sl =
                            step_tpt_svf_vec4(GpuVec4::splat(in_l), &mut s1_l, &mut s2_l, &qp.svf);
                        let sr =
                            step_tpt_svf_vec4(GpuVec4::splat(in_r), &mut s1_r, &mut s2_r, &qp.svf);
                        let tone_l = sl.lp.to_array()[0] * qp.w_lp
                            + sl.bp.to_array()[0] * qp.w_bp
                            + sl.hp.to_array()[0] * qp.w_hp
                            + in_l * qp.w_dir;
                        let tone_r = sr.lp.to_array()[0] * qp.w_lp
                            + sr.bp.to_array()[0] * qp.w_bp
                            + sr.hp.to_array()[0] * qp.w_hp
                            + in_r * qp.w_dir;
                        let vl = step_paralru_duhem_vec4(
                            GpuVec4::splat(qp.rect_sign * tone_l),
                            GpuVec4::splat(qp.two_svf_g * sl.bp.to_array()[0]),
                            &mut v_l,
                            &qp.lru,
                        );
                        let vr = step_paralru_duhem_vec4(
                            GpuVec4::splat(qp.rect_sign * tone_r),
                            GpuVec4::splat(qp.two_svf_g * sr.bp.to_array()[0]),
                            &mut v_r,
                            &qp.lru,
                        );
                        let (vel, ver) = apply_stereo_link_vec4_pair(vl, vr, qp.lru.stereo_link);
                        let sag_l = (1.0 - qp.sag_depth * vel.to_array()[0]).max(MIN_SAG_GAIN);
                        let sag_r = (1.0 - qp.sag_depth * ver.to_array()[0]).max(MIN_SAG_GAIN);
                        let r_l = GpuVec4::splat(
                            tone_l * qp.adaa_drive * sag_l + qp.lru_bias_drive * vel.to_array()[0],
                        );
                        let r_r = GpuVec4::splat(
                            tone_r * qp.adaa_drive * sag_r + qp.lru_bias_drive * ver.to_array()[0],
                        );
                        let (ol, _, _) = eval_poly_adaa1_odd_even_vec4(rp_l, r_l);
                        let (or, _, _) = eval_poly_adaa1_odd_even_vec4(rp_r, r_r);
                        rp_l = r_l;
                        rp_r = r_r;
                        pred[lt][0] += ol.to_array()[0];
                        pred[lt][1] += or.to_array()[0];
                    }
                }
                let mut err_e = 0.0_f32;
                let mut tgt_e = 0.0_f32;
                for (lt, frame) in (start..end).enumerate() {
                    let [tl, tr] = input.target_audio[s * num_frames + frame];
                    let (dl, dr) = (pred[lt][0] - tl, pred[lt][1] - tr);
                    err_e += dl * dl + dr * dr;
                    tgt_e += tl * tl + tr * tr;
                }
                losses.push(canonicalize_zero_f32_bits(err_e / (tgt_e + eps_floor)));
            }
        }
    }
    Ok(losses)
}

#[inline]
fn bit_reverse(x: usize, log2_n: u32) -> usize {
    let mut v = x;
    let mut r = 0_usize;
    for _ in 0..log2_n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

#[inline(always)]
fn cmul_stereo(cos_w: f32, sin_w: f32, z: [f32; 4]) -> [f32; 4] {
    [
        cos_w * z[0] - sin_w * z[1],
        cos_w * z[1] + sin_w * z[0],
        cos_w * z[2] - sin_w * z[3],
        cos_w * z[3] + sin_w * z[2],
    ]
}

fn run_gabor_mode_a(input: &GaborBezoutInput<'_>) -> GaborBezoutOutput {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_stems = (u.num_stems as usize).max(1);
    let num_windows = num_frames.div_ceil(TILE_FRAMES);
    let num_taps = (u.hop_size as usize).clamp(1, MAX_BEZOUT_TAPS);
    let eps = if is_finite_f32(u.eps_noise) && u.eps_noise > MIN_EPS_NOISE {
        u.eps_noise
    } else {
        MIN_EPS_NOISE
    };

    let mut d_loss_d_pred = vec![[0.0_f32; 2]; num_stems * num_frames];
    let mut band_losses = vec![[0.0_f32; 4]; num_stems * num_windows];

    for stem in 0..num_stems {
        for win in 0..num_windows {
            let base = win * TILE_FRAMES;
            let valid = (num_frames - base).min(TILE_FRAMES);
            let mut y_lm = vec![[0.0_f32; 4]; valid];
            let mut y_h = vec![[0.0_f32; 2]; valid];
            let (mut e_lf, mut e_mf, mut e_hf, mut t_lf, mut t_mf, mut t_hf) =
                (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

            for n in 0..valid {
                let (mut yll, mut ylr, mut yml, mut ymr, mut yhl, mut yhr) =
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                let (mut tll, mut tlr, mut tml, mut tmr, mut thl, mut thr) =
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                for k in 0..num_taps.min(n + 1) {
                    let tap = input.bezout_taps[k];
                    let g_idx = stem * num_frames + base + n - k;
                    let p = input.pred_audio[g_idx];
                    let t = input.target_audio[g_idx];
                    let (el, er) = (p[0] - t[0], p[1] - t[1]);
                    yll += tap[0] * el;
                    ylr += tap[0] * er;
                    yml += tap[1] * el;
                    ymr += tap[1] * er;
                    yhl += tap[2] * el;
                    yhr += tap[2] * er;
                    tll += tap[0] * t[0];
                    tlr += tap[0] * t[1];
                    tml += tap[1] * t[0];
                    tmr += tap[1] * t[1];
                    thl += tap[2] * t[0];
                    thr += tap[2] * t[1];
                }
                y_lm[n] = [yll, ylr, yml, ymr];
                y_h[n] = [yhl, yhr];
                e_lf += yll * yll + ylr * ylr;
                e_mf += yml * yml + ymr * ymr;
                e_hf += yhl * yhl + yhr * yhr;
                t_lf += tll * tll + tlr * tlr;
                t_mf += tml * tml + tmr * tmr;
                t_hf += thl * thl + thr * thr;
            }

            let (inv_lf, inv_mf, inv_hf) =
                (1.0 / (t_lf + eps), 1.0 / (t_mf + eps), 1.0 / (t_hf + eps));
            let (l_lf, l_mf, l_hf) = (e_lf * inv_lf, e_mf * inv_mf, e_hf * inv_hf);
            let total = u.w_lf * l_lf + u.w_mf * l_mf + u.w_hf * l_hf;
            band_losses[stem * num_windows + win] = [l_lf, l_mf, l_hf, total];

            let (n_lf, n_mf, n_hf) = (
                2.0 * u.w_lf * inv_lf,
                2.0 * u.w_mf * inv_mf,
                2.0 * u.w_hf * inv_hf,
            );
            for n in 0..valid {
                let (mut gl, mut gr) = (0.0_f32, 0.0_f32);
                for k in 0..num_taps {
                    if n + k < valid {
                        let tap = input.bezout_taps[k];
                        let ym = y_lm[n + k];
                        let yh = y_h[n + k];
                        gl += tap[0] * (ym[0] * n_lf)
                            + tap[1] * (ym[2] * n_mf)
                            + tap[2] * (yh[0] * n_hf);
                        gr += tap[0] * (ym[1] * n_lf)
                            + tap[1] * (ym[3] * n_mf)
                            + tap[2] * (yh[1] * n_hf);
                    }
                }
                d_loss_d_pred[stem * num_frames + base + n] = [gl, gr];
            }
        }
    }
    GaborBezoutOutput {
        d_loss_d_pred,
        band_losses,
    }
}

fn run_gabor_mode_b(input: &GaborBezoutInput<'_>) -> Result<GaborBezoutOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_stems = (u.num_stems as usize).max(1);
    let n = u.fft_size as usize;
    let hop = u.hop_size as usize;
    if !matches!(n, 16 | 32 | 64 | 128 | 256) {
        return Err(TrainError::InvalidDimensions {
            context: "gabor_bezout_grad(fft_size)",
            expected: 64,
            actual: n,
        });
    }
    if hop < n {
        return Err(TrainError::KernelMismatch {
            context: "gabor_bezout_grad",
            detail: "Mode B requires hop_size >= fft_size for disjoint writebacks",
        });
    }
    if input.bezout_taps.len() < n {
        return Err(TrainError::InvalidDimensions {
            context: "gabor_bezout_grad(bezout_taps)",
            expected: n,
            actual: input.bezout_taps.len(),
        });
    }
    let log2_n = n.trailing_zeros();
    let num_windows = num_frames.div_ceil(hop);
    let eps = if is_finite_f32(u.eps_noise) && u.eps_noise > MIN_EPS_NOISE {
        u.eps_noise
    } else {
        MIN_EPS_NOISE
    };

    let mut d_loss_d_pred = vec![[0.0_f32; 2]; num_stems * num_frames];
    let mut band_losses = vec![[0.0_f32; 4]; num_stems * num_windows];

    for stem in 0..num_stems {
        for win in 0..num_windows {
            let win_start = win * hop;
            let mut buf_a = vec![[0.0_f32; 4]; n];
            let mut buf_b = vec![[0.0_f32; 4]; n];
            for k in 0..n {
                let frame = win_start + k;
                let rev = bit_reverse(k, log2_n);
                let win_w = input.bezout_taps[k][0];
                if frame < num_frames {
                    let g = stem * num_frames + frame;
                    let p = input.pred_audio[g];
                    let t = input.target_audio[g];
                    buf_a[rev] = [(p[0] - t[0]) * win_w, 0.0, (p[1] - t[1]) * win_w, 0.0];
                    buf_b[rev] = [t[0] * win_w, 0.0, t[1] * win_w, 0.0];
                }
            }
            for stage in 0..log2_n {
                let half = 1_usize << stage;
                let angle_shift = 31_u32 - stage;
                for m in 0..(n / 2) {
                    let group = m >> stage;
                    let j = m & (half - 1);
                    let i0 = (group << (stage + 1)) + j;
                    let i1 = i0 + half;
                    let (cos_w, sin_pos) = algebraic_unit_rotor_f32(
                        (((j as u32) << angle_shift) as f32) * Q32_TO_RAD_F32,
                    );
                    let a0 = buf_a[i0];
                    let a1 = cmul_stereo(cos_w, -sin_pos, buf_a[i1]);
                    let b0 = buf_b[i0];
                    let b1 = cmul_stereo(cos_w, -sin_pos, buf_b[i1]);
                    for d in 0..4 {
                        buf_a[i0][d] = a0[d] + a1[d];
                        buf_a[i1][d] = a0[d] - a1[d];
                        buf_b[i0][d] = b0[d] + b1[d];
                        buf_b[i1][d] = b0[d] - b1[d];
                    }
                }
            }
            let (mut el, mut tl) = ([0.0_f32; 3], [0.0_f32; 3]);
            for k in 0..n {
                let ea = buf_a[k];
                let tb = buf_b[k];
                let e2 = ea[0] * ea[0] + ea[1] * ea[1] + ea[2] * ea[2] + ea[3] * ea[3];
                let t2 = tb[0] * tb[0] + tb[1] * tb[1] + tb[2] * tb[2] + tb[3] * tb[3];
                for b in 0..3 {
                    let w = input.bezout_taps[k][b + 1];
                    el[b] += e2 * w;
                    tl[b] += t2 * w;
                }
            }
            let inv = [
                1.0 / (tl[0] + eps),
                1.0 / (tl[1] + eps),
                1.0 / (tl[2] + eps),
            ];
            let (ll, lm, lh) = (el[0] * inv[0], el[1] * inv[1], el[2] * inv[2]);
            band_losses[stem * num_windows + win] =
                [ll, lm, lh, u.w_lf * ll + u.w_mf * lm + u.w_hf * lh];
            for (k, item) in buf_a.iter_mut().enumerate() {
                let ck = (u.w_lf * inv[0]) * input.bezout_taps[k][1]
                    + (u.w_mf * inv[1]) * input.bezout_taps[k][2]
                    + (u.w_hf * inv[2]) * input.bezout_taps[k][3];
                for v in item.iter_mut() {
                    *v *= 2.0 * ck;
                }
            }
            for rev_s in 0..log2_n {
                let s = log2_n - 1 - rev_s;
                let half = 1_usize << s;
                let angle_shift = 31_u32 - s;
                for m in 0..(n / 2) {
                    let group = m >> s;
                    let j = m & (half - 1);
                    let i0 = (group << (s + 1)) + j;
                    let i1 = i0 + half;
                    let (cos_w, sin_conj) = algebraic_unit_rotor_f32(
                        (((j as u32) << angle_shift) as f32) * Q32_TO_RAD_F32,
                    );
                    let g0 = buf_a[i0];
                    let g1 = buf_a[i1];
                    let diff = [g0[0] - g1[0], g0[1] - g1[1], g0[2] - g1[2], g0[3] - g1[3]];
                    for d in 0..4 {
                        buf_a[i0][d] = g0[d] + g1[d];
                    }
                    buf_a[i1] = cmul_stereo(cos_w, sin_conj, diff);
                }
            }
            for k in 0..n {
                let frame = win_start + k;
                if frame < num_frames {
                    let gt = buf_a[bit_reverse(k, log2_n)];
                    let ww = input.bezout_taps[k][0];
                    d_loss_d_pred[stem * num_frames + frame] = [gt[0] * ww, gt[2] * ww];
                }
            }
        }
    }
    Ok(GaborBezoutOutput {
        d_loss_d_pred,
        band_losses,
    })
}

/// Deterministic CPU reference implementation of `gabor_bezout_grad` (`Mode A` and `Mode B`).
pub fn run_gabor_bezout_grad_cpu(
    input: &GaborBezoutInput<'_>,
) -> Result<GaborBezoutOutput, TrainError> {
    let num_frames = input.uniforms.num_frames as usize;
    let num_stems = input.uniforms.num_stems as usize;
    if num_frames == 0 || num_stems == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "gabor_bezout_grad",
            expected: 1,
            actual: 0,
        });
    }
    let need = num_stems * num_frames;
    if input.pred_audio.len() < need || input.target_audio.len() < need {
        return Err(TrainError::InvalidDimensions {
            context: "gabor_bezout_grad(audio)",
            expected: need,
            actual: input.pred_audio.len().min(input.target_audio.len()),
        });
    }
    if input.uniforms.fft_size == 0 {
        let need_taps = (input.uniforms.hop_size as usize).clamp(1, MAX_BEZOUT_TAPS);
        if input.bezout_taps.len() < need_taps {
            return Err(TrainError::InvalidDimensions {
                context: "gabor_bezout_grad(taps)",
                expected: need_taps,
                actual: input.bezout_taps.len(),
            });
        }
        Ok(run_gabor_mode_a(input))
    } else {
        run_gabor_mode_b(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::abi::{GaborBezoutUniforms, PrescreenUniforms};

    #[test]
    fn test_cpu_prescreen_and_gabor_mode_a_b_fd() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        let n = 300;
        // 3-band stereo signal ensuring non-degenerate target energy across LF, MF, and HF bins.
        let in_a: Vec<[f32; 2]> = (0..n)
            .map(|t| {
                let ph = t as f32;
                [
                    0.55 * (0.21 * ph).sin() + 0.30 * (1.35 * ph).cos() + 0.22 * (2.55 * ph).sin(),
                    0.45 * (0.21 * ph).cos() + 0.28 * (1.42 * ph).sin() + 0.18 * (2.63 * ph).cos(),
                ]
            })
            .collect();
        let tg_a: Vec<[f32; 2]> = (0..n)
            .map(|t| {
                let ph = t as f32;
                [
                    0.35 * (0.21 * ph).sin() + 0.24 * (1.35 * ph).cos() + 0.16 * (2.55 * ph).sin(),
                    0.25 * (0.23 * ph).cos() + 0.22 * (1.42 * ph).sin() + 0.14 * (2.63 * ph).cos(),
                ]
            })
            .collect();
        let mut seeds = vec![0.0_f32; 32];
        for i in 0..2 {
            let b = i * 16;
            seeds[b] = 0.05 + 0.04 * (i as f32);
            seeds[b + 1] = 0.8;
            seeds[b + 2] = 0.5;
            seeds[b + 6] = 0.8;
            seeds[b + 7] = 0.3;
            seeds[b + 12] = 1.0;
            seeds[b + 14] = 1.2;
        }
        let losses = run_prescreen_population_cpu(&PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 2,
                num_frames: n as u32,
                num_quads: 1,
                num_stems: 1,
                sample_rate_hz: 48_000.0,
                eps_floor: 1.0e-5,
                _pad0: 0,
                _pad1: 0,
            },
            input_audio: &in_a,
            target_audio: &tg_a,
            seed_params: &seeds,
        })
        .unwrap();
        assert_eq!(losses.len(), 4);
        assert!(losses.iter().all(|v| v.is_finite() && *v > 0.0));

        // Mode A pure-f64 causal FIR FD check
        let taps_a = vec![
            [0.25_f32, -0.35, 0.45, 1.0],
            [0.35, 0.20, -0.35, 1.0],
            [0.40, 0.15, -0.10, 1.0],
        ];
        let u_a = GaborBezoutUniforms {
            num_frames: 270,
            fft_size: 0,
            hop_size: 3,
            num_stems: 1,
            w_lf: 1.0,
            w_mf: 0.7,
            w_hf: 0.4,
            eps_noise: 1.0e-5,
        };
        let pred_a: Vec<[f32; 2]> = in_a[..270].to_vec();
        let tgt_a: Vec<[f32; 2]> = tg_a[..270].to_vec();
        let out_a = run_gabor_bezout_grad_cpu(&GaborBezoutInput {
            uniforms: u_a,
            pred_audio: &pred_a,
            target_audio: &tgt_a,
            bezout_taps: &taps_a,
        })
        .unwrap();
        assert_eq!(out_a.band_losses.len(), 2);
        let eval_mode_a_f64 = |idx_pert: usize, delta_x: f64| -> f64 {
            let mut el = [0.0_f64; 3];
            let mut tl = [0.0_f64; 3];
            for t in 0..256 {
                let mut y = [[0.0_f64; 2]; 3];
                let mut tv = [[0.0_f64; 2]; 3];
                for k in 0..taps_a.len().min(t + 1) {
                    let s_idx = t - k;
                    let px =
                        f64::from(pred_a[s_idx][0]) + if s_idx == idx_pert { delta_x } else { 0.0 };
                    let py = f64::from(pred_a[s_idx][1]);
                    let tx = f64::from(tgt_a[s_idx][0]);
                    let ty = f64::from(tgt_a[s_idx][1]);
                    for b in 0..3 {
                        let h = f64::from(taps_a[k][b]);
                        y[b][0] += h * (px - tx);
                        y[b][1] += h * (py - ty);
                        tv[b][0] += h * tx;
                        tv[b][1] += h * ty;
                    }
                }
                for b in 0..3 {
                    el[b] += y[b][0] * y[b][0] + y[b][1] * y[b][1];
                    tl[b] += tv[b][0] * tv[b][0] + tv[b][1] * tv[b][1];
                }
            }
            let eps = f64::from(u_a.eps_noise);
            el[0] / (tl[0] + eps) + 0.7 * el[1] / (tl[1] + eps) + 0.4 * el[2] / (tl[2] + eps)
        };
        let eps_fd = 1.0e-6_f64;
        let fd_a = (eval_mode_a_f64(40, eps_fd) - eval_mode_a_f64(40, -eps_fd)) / (2.0 * eps_fd);
        assert!(
            (f64::from(out_a.d_loss_d_pred[40][0]) - fd_a).abs() <= 2.0e-4 * fd_a.abs().max(1.0)
        );

        // Mode B pure-f64 DFT FD check + overlap rejection
        let fft = 64_usize;
        let taps_b: Vec<[f32; 4]> = (0..fft)
            .map(|k| {
                let w = 0.5 * (1.0 - (core::f32::consts::TAU * (k as f32) / (fft as f32)).cos());
                let lf = f32::from(k <= fft / 8 || k >= fft - fft / 8);
                let mf = f32::from(
                    (fft / 8 + 1..fft / 3).contains(&k)
                        || (fft - fft / 3..fft - fft / 8).contains(&k),
                );
                let hf = f32::from((fft / 3..fft - fft / 3 + 1).contains(&k));
                [w, lf, mf, hf]
            })
            .collect();
        let err = run_gabor_bezout_grad_cpu(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: 128,
                fft_size: 64,
                hop_size: 32,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 0.7,
                w_hf: 0.4,
                eps_noise: 1.0e-5,
            },
            pred_audio: &in_a[..128],
            target_audio: &tg_a[..128],
            bezout_taps: &taps_b,
        });
        assert!(matches!(err, Err(TrainError::KernelMismatch { .. })));

        let out_b = run_gabor_bezout_grad_cpu(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: 128,
                fft_size: 64,
                hop_size: 64,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 0.7,
                w_hf: 0.4,
                eps_noise: 1.0e-5,
            },
            pred_audio: &in_a[..128],
            target_audio: &tg_a[..128],
            bezout_taps: &taps_b,
        })
        .unwrap();
        assert_eq!(out_b.band_losses.len(), 2);
        let eval_mode_b_f64 = |idx_pert: usize, delta_x: f64| -> f64 {
            let mut el = [0.0_f64; 3];
            let mut tl = [0.0_f64; 3];
            for k in 0..fft {
                let (mut erl, mut eil, mut err, mut eir) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                let (mut trl, mut til, mut trr, mut tir) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                for n_t in 0..fft {
                    let theta = -core::f64::consts::TAU * ((k * n_t) as f64) / (fft as f64);
                    let (sin_w, cos_w) = theta.sin_cos();
                    let wv = f64::from(taps_b[n_t][0]);
                    let px = f64::from(in_a[n_t][0]) + if n_t == idx_pert { delta_x } else { 0.0 };
                    let xl = (px - f64::from(tg_a[n_t][0])) * wv;
                    let xr = (f64::from(in_a[n_t][1]) - f64::from(tg_a[n_t][1])) * wv;
                    let ytl = f64::from(tg_a[n_t][0]) * wv;
                    let ytr = f64::from(tg_a[n_t][1]) * wv;
                    erl += xl * cos_w;
                    eil += xl * sin_w;
                    err += xr * cos_w;
                    eir += xr * sin_w;
                    trl += ytl * cos_w;
                    til += ytl * sin_w;
                    trr += ytr * cos_w;
                    tir += ytr * sin_w;
                }
                let e2 = erl * erl + eil * eil + err * err + eir * eir;
                let t2 = trl * trl + til * til + trr * trr + tir * tir;
                for b in 0..3 {
                    el[b] += e2 * f64::from(taps_b[k][b + 1]);
                    tl[b] += t2 * f64::from(taps_b[k][b + 1]);
                }
            }
            let eps = 1.0e-5_f64;
            el[0] / (tl[0] + eps) + 0.7 * el[1] / (tl[1] + eps) + 0.4 * el[2] / (tl[2] + eps)
        };
        let fd_b = (eval_mode_b_f64(32, eps_fd) - eval_mode_b_f64(32, -eps_fd)) / (2.0 * eps_fd);
        assert!(
            (f64::from(out_b.d_loss_d_pred[32][0]) - fd_b).abs() <= 2.0e-4 * fd_b.abs().max(1.0)
        );

        // Cover non-finite seed_params/eps handling, partial tail FFT padding, and error branches.
        let mut nf_seeds = vec![0.0_f32; 32];
        nf_seeds[0] = f32::NAN;
        nf_seeds[1] = f32::INFINITY;
        nf_seeds[2] = -f32::INFINITY;
        let nf_losses = run_prescreen_population_cpu(&PrescreenInput {
            uniforms: PrescreenUniforms {
                num_seeds: 2,
                num_frames: 32,
                num_quads: 1,
                num_stems: 1,
                sample_rate_hz: 48_000.0,
                eps_floor: f32::NAN,
                _pad0: 0,
                _pad1: 0,
            },
            input_audio: &in_a[..32],
            target_audio: &tg_a[..32],
            seed_params: &nf_seeds,
        })
        .unwrap();
        assert_eq!(nf_losses.len(), 2);

        let partial_b = run_gabor_bezout_grad_cpu(&GaborBezoutInput {
            uniforms: GaborBezoutUniforms {
                num_frames: 40,
                fft_size: 64,
                hop_size: 64,
                num_stems: 1,
                w_lf: 1.0,
                w_mf: 0.7,
                w_hf: 0.4,
                eps_noise: 0.0,
            },
            pred_audio: &in_a[..40],
            target_audio: &tg_a[..40],
            bezout_taps: &taps_b,
        })
        .unwrap();
        assert_eq!(partial_b.d_loss_d_pred.len(), 40);

        for bad_u in [
            GaborBezoutUniforms {
                num_frames: 0,
                ..u_a
            },
            GaborBezoutUniforms {
                num_frames: 64,
                fft_size: 48,
                hop_size: 64,
                ..u_a
            },
        ] {
            let err = run_gabor_bezout_grad_cpu(&GaborBezoutInput {
                uniforms: bad_u,
                pred_audio: &in_a[..64],
                target_audio: &tg_a[..64],
                bezout_taps: &taps_b,
            });
            assert!(matches!(err, Err(TrainError::InvalidDimensions { .. })));
        }
        assert!(matches!(
            run_gabor_bezout_grad_cpu(&GaborBezoutInput {
                uniforms: u_a,
                pred_audio: &[],
                target_audio: &[],
                bezout_taps: &taps_a,
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            run_gabor_bezout_grad_cpu(&GaborBezoutInput {
                uniforms: u_a,
                pred_audio: &pred_a,
                target_audio: &tgt_a,
                bezout_taps: &[],
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            run_gabor_bezout_grad_cpu(&GaborBezoutInput {
                uniforms: GaborBezoutUniforms {
                    num_frames: 64,
                    fft_size: 64,
                    hop_size: 64,
                    ..u_a
                },
                pred_audio: &in_a[..64],
                target_audio: &tg_a[..64],
                bezout_taps: &[],
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
    }
}
