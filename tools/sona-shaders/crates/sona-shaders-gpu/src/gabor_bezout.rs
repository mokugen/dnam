#![forbid(unsafe_code)]

//! Offline 3-band perceptual Bézout `bESR` (`fft_size == 0`) and constant-Q Gabor STFT
//! (`fft_size > 0`) loss & exact adjoint gradient compute kernel (`gabor_bezout_grad`),
//! operating strictly within `11,264 B` workgroup SRAM (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{GaborBezoutUniforms, MAX_WORKGROUP_SRAM_BYTES};
use crate::reduce::{
    WG_THREADS, clamp_idx, div_floor_u32_divfree, flush_zero, is_finite_bits, rcp_pos_f32,
    reduce_sum_f32_256, sub_sat_usize, wg_barrier,
};
use core::mem::size_of;
use dnam_kernels::{algebraic_unit_rotor_f32, const_f32_pow2};
use spirv_std::glam::{UVec3, Vec2, Vec4};

/// Exact conversion factor `2 * pi * 2^-32` from `Q0.32` FFT twiddle angles to radians.
const Q32_TO_RAD_F32: f32 = core::f32::consts::TAU * const_f32_pow2(-32);

/// Workgroup `buf_a` capacity (`256 × Vec4 = 4,096 B`).
pub const GABOR_BUF_A_VEC4S: usize = WG_THREADS;

/// Workgroup `buf_b` capacity (`256 × Vec4 = 4,096 B`).
pub const GABOR_BUF_B_VEC4S: usize = WG_THREADS;

/// Workgroup `buf_c` capacity (`256 × Vec2 = 2,048 B`).
pub const GABOR_BUF_C_VEC2S: usize = WG_THREADS;

/// Workgroup `reduce_buf` capacity (`256 × f32 = 1,024 B`).
pub const GABOR_REDUCE_FLOATS: usize = WG_THREADS;

/// Total `gabor_bezout_grad` workgroup SRAM footprint (`11,264 B <= 15,872 B`).
pub const GABOR_BEZOUT_SRAM_BYTES: usize = GABOR_BUF_A_VEC4S * size_of::<[f32; 4]>()
    + GABOR_BUF_B_VEC4S * size_of::<[f32; 4]>()
    + GABOR_BUF_C_VEC2S * size_of::<[f32; 2]>()
    + GABOR_REDUCE_FLOATS * size_of::<f32>();

const _: () = assert!(GABOR_BEZOUT_SRAM_BYTES == 11_264);
const _: () = assert!(GABOR_BEZOUT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);

/// Minimum `eps_noise` floor protecting normalized band energy denominators.
const MIN_EPS_NOISE: f32 = 1.0e-12;

/// Maximum Mode A Bézout FIR filter tap length.
const MAX_BEZOUT_TAPS: usize = 64;

#[inline(always)]
fn clamp_eps_noise(eps_noise: f32) -> f32 {
    if is_finite_bits(eps_noise) && eps_noise > MIN_EPS_NOISE {
        eps_noise
    } else {
        MIN_EPS_NOISE
    }
}

#[inline(always)]
fn bit_reverse(x: usize, log2_n: u32) -> usize {
    let mut v = x;
    let mut r = 0_usize;
    let mut i = 0_u32;
    while i < log2_n {
        r = (r << 1) | (v & 1);
        v >>= 1;
        i += 1;
    }
    r
}

#[inline(always)]
fn cmul_stereo(cos_w: f32, sin_w: f32, z: Vec4) -> Vec4 {
    Vec4::new(
        cos_w * z.x - sin_w * z.y,
        cos_w * z.y + sin_w * z.x,
        cos_w * z.z - sin_w * z.w,
        cos_w * z.w + sin_w * z.z,
    )
}

/// Mode A (`fft_size == 0`): Causal 3-band Bézout FIR `bESR` over a non-overlapping `256`-sample
/// tile with exact transpose-FIR adjoint gradient writeback.
#[inline]
fn run_bezout_fir_mode(
    win_idx: usize,
    stem_idx: usize,
    lid: usize,
    sg_lid: u32,
    uniforms: &GaborBezoutUniforms,
    pred_audio: &[Vec2],
    target_audio: &[Vec2],
    bezout_taps: &[Vec4],
    d_loss_d_pred: &mut [Vec2],
    band_losses: &mut [Vec4],
    buf_a: &mut [Vec4; GABOR_BUF_A_VEC4S],
    buf_b: &mut [Vec4; GABOR_BUF_B_VEC4S],
    buf_c: &mut [Vec2; GABOR_BUF_C_VEC2S],
    reduce_buf: &mut [f32; GABOR_REDUCE_FLOATS],
) {
    let num_frames = uniforms.num_frames as usize;
    let num_stems = (uniforms.num_stems as usize).max(1);
    let num_windows = if num_frames == 0 {
        1
    } else {
        ((num_frames - 1) >> 8) + 1
    };
    let tile_base = win_idx * WG_THREADS;
    let valid_len = sub_sat_usize(num_frames, tile_base).min(WG_THREADS);
    let num_taps = (uniforms.hop_size as usize).max(1).min(MAX_BEZOUT_TAPS);
    let active = lid < valid_len && stem_idx < num_stems;

    let frame = tile_base + lid;
    let global_idx = stem_idx * num_frames + frame;
    if active {
        let p = pred_audio[clamp_idx(global_idx, pred_audio.len())];
        let t = target_audio[clamp_idx(global_idx, target_audio.len())];
        let err_l = flush_zero(p.x - t.x);
        let err_r = flush_zero(p.y - t.y);
        let tgt_l = flush_zero(t.x);
        let tgt_r = flush_zero(t.y);
        buf_a[clamp_idx(lid, WG_THREADS)] = Vec4::new(err_l, err_r, tgt_l, tgt_r);
    } else {
        buf_a[clamp_idx(lid, WG_THREADS)] = Vec4::ZERO;
    }
    wg_barrier();

    let mut y_lf_l = 0.0_f32;
    let mut y_lf_r = 0.0_f32;
    let mut y_mf_l = 0.0_f32;
    let mut y_mf_r = 0.0_f32;
    let mut y_hf_l = 0.0_f32;
    let mut y_hf_r = 0.0_f32;
    let mut t_lf_l = 0.0_f32;
    let mut t_lf_r = 0.0_f32;
    let mut t_mf_l = 0.0_f32;
    let mut t_mf_r = 0.0_f32;
    let mut t_hf_l = 0.0_f32;
    let mut t_hf_r = 0.0_f32;

    if active {
        for k in 0..num_taps {
            if k <= lid {
                let tap = bezout_taps[clamp_idx(k, bezout_taps.len())];
                let h_lf = flush_zero(tap.x);
                let h_mf = flush_zero(tap.y);
                let h_hf = flush_zero(tap.z);
                let s = buf_a[clamp_idx(lid - k, WG_THREADS)];
                y_lf_l += h_lf * s.x;
                y_lf_r += h_lf * s.y;
                y_mf_l += h_mf * s.x;
                y_mf_r += h_mf * s.y;
                y_hf_l += h_hf * s.x;
                y_hf_r += h_hf * s.y;
                t_lf_l += h_lf * s.z;
                t_lf_r += h_lf * s.w;
                t_mf_l += h_mf * s.z;
                t_mf_r += h_mf * s.w;
                t_hf_l += h_hf * s.z;
                t_hf_r += h_hf * s.w;
            }
        }
    }

    buf_b[clamp_idx(lid, WG_THREADS)] = Vec4::new(y_lf_l, y_lf_r, y_mf_l, y_mf_r);
    buf_c[clamp_idx(lid, WG_THREADS)] = Vec2::new(y_hf_l, y_hf_r);

    let e_lf = if active {
        flush_zero(y_lf_l * y_lf_l + y_lf_r * y_lf_r)
    } else {
        0.0
    };
    let t_lf = if active {
        flush_zero(t_lf_l * t_lf_l + t_lf_r * t_lf_r)
    } else {
        0.0
    };
    let e_mf = if active {
        flush_zero(y_mf_l * y_mf_l + y_mf_r * y_mf_r)
    } else {
        0.0
    };
    let t_mf = if active {
        flush_zero(t_mf_l * t_mf_l + t_mf_r * t_mf_r)
    } else {
        0.0
    };
    let e_hf = if active {
        flush_zero(y_hf_l * y_hf_l + y_hf_r * y_hf_r)
    } else {
        0.0
    };
    let t_hf = if active {
        flush_zero(t_hf_l * t_hf_l + t_hf_r * t_hf_r)
    } else {
        0.0
    };

    let err_e_lf = reduce_sum_f32_256(reduce_buf, lid, e_lf, sg_lid);
    let tgt_e_lf = reduce_sum_f32_256(reduce_buf, lid, t_lf, sg_lid);
    let err_e_mf = reduce_sum_f32_256(reduce_buf, lid, e_mf, sg_lid);
    let tgt_e_mf = reduce_sum_f32_256(reduce_buf, lid, t_mf, sg_lid);
    let err_e_hf = reduce_sum_f32_256(reduce_buf, lid, e_hf, sg_lid);
    let tgt_e_hf = reduce_sum_f32_256(reduce_buf, lid, t_hf, sg_lid);

    let eps = clamp_eps_noise(uniforms.eps_noise);
    let w_lf = flush_zero(uniforms.w_lf);
    let w_mf = flush_zero(uniforms.w_mf);
    let w_hf = flush_zero(uniforms.w_hf);

    let inv_t_lf = rcp_pos_f32(tgt_e_lf + eps);
    let inv_t_mf = rcp_pos_f32(tgt_e_mf + eps);
    let inv_t_hf = rcp_pos_f32(tgt_e_hf + eps);

    let loss_lf = flush_zero(err_e_lf * inv_t_lf);
    let loss_mf = flush_zero(err_e_mf * inv_t_mf);
    let loss_hf = flush_zero(err_e_hf * inv_t_hf);
    let total_loss = flush_zero(w_lf * loss_lf + w_mf * loss_mf + w_hf * loss_hf);

    if lid == 0 && win_idx < num_windows && stem_idx < num_stems {
        band_losses[clamp_idx(stem_idx * num_windows + win_idx, band_losses.len())] =
            Vec4::new(loss_lf, loss_mf, loss_hf, total_loss);
    }

    let two_norm_lf = 2.0 * w_lf * inv_t_lf;
    let two_norm_mf = 2.0 * w_mf * inv_t_mf;
    let two_norm_hf = 2.0 * w_hf * inv_t_hf;

    let safe_lid = clamp_idx(lid, WG_THREADS);
    let cur_lm = buf_b[safe_lid];
    let cur_h = buf_c[safe_lid];
    buf_b[safe_lid] = Vec4::new(
        cur_lm.x * two_norm_lf,
        cur_lm.y * two_norm_lf,
        cur_lm.z * two_norm_mf,
        cur_lm.w * two_norm_mf,
    );
    buf_c[safe_lid] = Vec2::new(cur_h.x * two_norm_hf, cur_h.y * two_norm_hf);
    wg_barrier();

    if active {
        let mut grad_l = 0.0_f32;
        let mut grad_r = 0.0_f32;
        for k in 0..num_taps {
            let src_idx = lid + k;
            if src_idx < valid_len {
                let tap = bezout_taps[clamp_idx(k, bezout_taps.len())];
                let wy_lm = buf_b[clamp_idx(src_idx, WG_THREADS)];
                let wy_h = buf_c[clamp_idx(src_idx, WG_THREADS)];
                grad_l += flush_zero(tap.x) * wy_lm.x
                    + flush_zero(tap.y) * wy_lm.z
                    + flush_zero(tap.z) * wy_h.x;
                grad_r += flush_zero(tap.x) * wy_lm.y
                    + flush_zero(tap.y) * wy_lm.w
                    + flush_zero(tap.z) * wy_h.y;
            }
        }
        d_loss_d_pred[clamp_idx(global_idx, d_loss_d_pred.len())] =
            Vec2::new(flush_zero(grad_l), flush_zero(grad_r));
    }
}

/// Mode B (`fft_size > 0`): Windowed radix-2 DIT FFT across `n in {16, 32, 64, 128, 256}` using
/// `algebraic_unit_rotor_f32` twiddles and exact DIF conjugate-twiddle adjoint pullback.
#[inline]
fn run_gabor_fft_mode(
    win_idx: usize,
    stem_idx: usize,
    lid: usize,
    sg_lid: u32,
    uniforms: &GaborBezoutUniforms,
    pred_audio: &[Vec2],
    target_audio: &[Vec2],
    bezout_taps: &[Vec4],
    d_loss_d_pred: &mut [Vec2],
    band_losses: &mut [Vec4],
    buf_a: &mut [Vec4; GABOR_BUF_A_VEC4S],
    buf_b: &mut [Vec4; GABOR_BUF_B_VEC4S],
    buf_c: &mut [Vec2; GABOR_BUF_C_VEC2S],
    reduce_buf: &mut [f32; GABOR_REDUCE_FLOATS],
) {
    let mut n = 16_usize;
    let mut log2_n = 4_u32;
    let req_fft = uniforms.fft_size as usize;
    while n < req_fft && n < WG_THREADS {
        n <<= 1;
        log2_n += 1;
    }

    let num_frames = uniforms.num_frames as usize;
    let num_stems = (uniforms.num_stems as usize).max(1);
    let num_windows = div_floor_u32_divfree(band_losses.len(), num_stems).max(1);
    let hop = (uniforms.hop_size as usize).max(n);
    let win_start = win_idx * hop;
    let half_n = n >> 1;

    let tap_lid = bezout_taps[clamp_idx(lid, bezout_taps.len())];
    let win_w = if lid < n { flush_zero(tap_lid.x) } else { 0.0 };
    buf_c[clamp_idx(lid, WG_THREADS)] = Vec2::new(win_w, 0.0);

    let frame = win_start + lid;
    let in_bounds = lid < n && frame < num_frames && stem_idx < num_stems;
    let global_idx = stem_idx * num_frames + frame;
    let (err_x, err_y, tgt_x, tgt_y) = if in_bounds {
        let p = pred_audio[clamp_idx(global_idx, pred_audio.len())];
        let t = target_audio[clamp_idx(global_idx, target_audio.len())];
        (
            flush_zero((p.x - t.x) * win_w),
            flush_zero((p.y - t.y) * win_w),
            flush_zero(t.x * win_w),
            flush_zero(t.y * win_w),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let dst_idx = if lid < n {
        bit_reverse(lid, log2_n)
    } else {
        lid
    };
    buf_a[clamp_idx(dst_idx, WG_THREADS)] = Vec4::new(err_x, 0.0, err_y, 0.0);
    buf_b[clamp_idx(dst_idx, WG_THREADS)] = Vec4::new(tgt_x, 0.0, tgt_y, 0.0);
    wg_barrier();

    let mut stage = 0_u32;
    while stage < log2_n {
        let half = 1_usize << stage;
        let angle_shift = 31_u32 - stage;
        if lid < half_n {
            let group = lid >> stage;
            let j = lid & (half - 1);
            let idx0 = (group << (stage + 1)) + j;
            let idx1 = idx0 + half;
            let angle_q32 = (j as u32) << angle_shift;
            let (cos_w, sin_pos) = algebraic_unit_rotor_f32((angle_q32 as f32) * Q32_TO_RAD_F32);
            let sin_w = -sin_pos;

            let safe0 = clamp_idx(idx0, WG_THREADS);
            let safe1 = clamp_idx(idx1, WG_THREADS);

            let a0 = buf_a[safe0];
            let a1 = cmul_stereo(cos_w, sin_w, buf_a[safe1]);
            buf_a[safe0] = a0 + a1;
            buf_a[safe1] = a0 - a1;

            let b0 = buf_b[safe0];
            let b1 = cmul_stereo(cos_w, sin_w, buf_b[safe1]);
            buf_b[safe0] = b0 + b1;
            buf_b[safe1] = b0 - b1;
        }
        wg_barrier();
        stage += 1;
    }

    let ea = buf_a[clamp_idx(lid, WG_THREADS)];
    let tb = buf_b[clamp_idx(lid, WG_THREADS)];
    let lf_w = if lid < n { flush_zero(tap_lid.y) } else { 0.0 };
    let mf_w = if lid < n { flush_zero(tap_lid.z) } else { 0.0 };
    let hf_w = if lid < n { flush_zero(tap_lid.w) } else { 0.0 };

    let err_mag2 = if lid < n {
        flush_zero(ea.x * ea.x + ea.y * ea.y + ea.z * ea.z + ea.w * ea.w)
    } else {
        0.0
    };
    let tgt_mag2 = if lid < n {
        flush_zero(tb.x * tb.x + tb.y * tb.y + tb.z * tb.z + tb.w * tb.w)
    } else {
        0.0
    };

    let err_e_lf = reduce_sum_f32_256(reduce_buf, lid, err_mag2 * lf_w, sg_lid);
    let tgt_e_lf = reduce_sum_f32_256(reduce_buf, lid, tgt_mag2 * lf_w, sg_lid);
    let err_e_mf = reduce_sum_f32_256(reduce_buf, lid, err_mag2 * mf_w, sg_lid);
    let tgt_e_mf = reduce_sum_f32_256(reduce_buf, lid, tgt_mag2 * mf_w, sg_lid);
    let err_e_hf = reduce_sum_f32_256(reduce_buf, lid, err_mag2 * hf_w, sg_lid);
    let tgt_e_hf = reduce_sum_f32_256(reduce_buf, lid, tgt_mag2 * hf_w, sg_lid);

    let eps = clamp_eps_noise(uniforms.eps_noise);
    let w_lf = flush_zero(uniforms.w_lf);
    let w_mf = flush_zero(uniforms.w_mf);
    let w_hf = flush_zero(uniforms.w_hf);

    let inv_t_lf = rcp_pos_f32(tgt_e_lf + eps);
    let inv_t_mf = rcp_pos_f32(tgt_e_mf + eps);
    let inv_t_hf = rcp_pos_f32(tgt_e_hf + eps);

    let loss_lf = flush_zero(err_e_lf * inv_t_lf);
    let loss_mf = flush_zero(err_e_mf * inv_t_mf);
    let loss_hf = flush_zero(err_e_hf * inv_t_hf);
    let total_loss = flush_zero(w_lf * loss_lf + w_mf * loss_mf + w_hf * loss_hf);

    if lid == 0 && win_idx < num_windows && stem_idx < num_stems {
        band_losses[clamp_idx(stem_idx * num_windows + win_idx, band_losses.len())] =
            Vec4::new(loss_lf, loss_mf, loss_hf, total_loss);
    }

    let c_k = (w_lf * inv_t_lf) * lf_w + (w_mf * inv_t_mf) * mf_w + (w_hf * inv_t_hf) * hf_w;
    buf_a[clamp_idx(lid, WG_THREADS)] = if lid < n {
        ea * (2.0 * c_k)
    } else {
        Vec4::ZERO
    };
    wg_barrier();

    let mut rev_s = 0_u32;
    while rev_s < log2_n {
        let s = log2_n - 1 - rev_s;
        let half = 1_usize << s;
        let angle_shift = 31_u32 - s;
        if lid < half_n {
            let group = lid >> s;
            let j = lid & (half - 1);
            let idx0 = (group << (s + 1)) + j;
            let idx1 = idx0 + half;
            let angle_q32 = (j as u32) << angle_shift;
            let (cos_w, sin_conj) = algebraic_unit_rotor_f32((angle_q32 as f32) * Q32_TO_RAD_F32);

            let safe0 = clamp_idx(idx0, WG_THREADS);
            let safe1 = clamp_idx(idx1, WG_THREADS);
            let g0 = buf_a[safe0];
            let g1 = buf_a[safe1];
            buf_a[safe0] = g0 + g1;
            buf_a[safe1] = cmul_stereo(cos_w, sin_conj, g0 - g1);
        }
        wg_barrier();
        rev_s += 1;
    }

    if in_bounds {
        let rev_idx = bit_reverse(lid, log2_n);
        let g_time = buf_a[clamp_idx(rev_idx, WG_THREADS)];
        let win_cached = buf_c[clamp_idx(lid, WG_THREADS)].x;
        let grad_l = flush_zero(g_time.x * win_cached);
        let grad_r = flush_zero(g_time.z * win_cached);
        d_loss_d_pred[clamp_idx(global_idx, d_loss_d_pred.len())] = Vec2::new(grad_l, grad_r);
    }
}

/// Executes one `(win_idx = wg_id.x, stem_idx = wg_id.y)` window of `gabor_bezout_grad`
/// across `256` threads (`11,264 B` SRAM, zero `FDIV`/`FSQRT`/`IDIV`).
#[inline]
pub fn gabor_bezout_grad_impl(
    wg_id: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &GaborBezoutUniforms,
    pred_audio: &[Vec2],
    target_audio: &[Vec2],
    bezout_taps: &[Vec4],
    d_loss_d_pred: &mut [Vec2],
    band_losses: &mut [Vec4],
    buf_a: &mut [Vec4; GABOR_BUF_A_VEC4S],
    buf_b: &mut [Vec4; GABOR_BUF_B_VEC4S],
    buf_c: &mut [Vec2; GABOR_BUF_C_VEC2S],
    reduce_buf: &mut [f32; GABOR_REDUCE_FLOATS],
) {
    let win_idx = wg_id.x as usize;
    let stem_idx = wg_id.y as usize;
    let lid = lid_u32 as usize;

    if uniforms.fft_size == 0 {
        run_bezout_fir_mode(
            win_idx,
            stem_idx,
            lid,
            sg_lid,
            uniforms,
            pred_audio,
            target_audio,
            bezout_taps,
            d_loss_d_pred,
            band_losses,
            buf_a,
            buf_b,
            buf_c,
            reduce_buf,
        );
    } else {
        run_gabor_fft_mode(
            win_idx,
            stem_idx,
            lid,
            sg_lid,
            uniforms,
            pred_audio,
            target_audio,
            bezout_taps,
            d_loss_d_pred,
            band_losses,
            buf_a,
            buf_b,
            buf_c,
            reduce_buf,
        );
    }
}
