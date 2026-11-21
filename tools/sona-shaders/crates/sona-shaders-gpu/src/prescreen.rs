#![forbid(unsafe_code)]

//! Population prescreening compute kernel (`prescreen_population`) evaluating candidate `CoreQuad`
//! parameter vectors across stereo audio tiles using `8,192 B` workgroup SRAM
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{MAX_WORKGROUP_SRAM_BYTES, PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms};
use crate::reduce::{
    WG_THREADS, clamp_f32_safe, clamp_idx, flush_zero, is_finite_bits, rcp_pos_f32,
    reduce_sum_f32_pair_256, sub_sat_usize, wg_barrier,
};
use core::f32::consts::FRAC_1_SQRT_2;
use core::mem::size_of;
use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_SAG_GAIN, SonaParaLruDuhemCoeffs4, TptScanElem, TptSvfCoeffs4,
    apply_stereo_link_vec4_pair, apply_tpt_init, compose_tpt_scan, eval_poly_adaa1_odd_even_vec4,
    prewarp_tan_pi_f_over_fs_f32, step_paralru_duhem_vec4, step_tpt_svf_vec4,
};
use spirv_std::glam::{UVec3, Vec2};

/// Workgroup `scan_scratch` capacity (`1,536 × f32 = 6,144 B`).
pub const PRESCREEN_SCAN_SCRATCH_FLOATS: usize = 6 * WG_THREADS;

/// Workgroup `err_scratch` capacity (`256 × f32 = 1,024 B`).
pub const PRESCREEN_ERR_SCRATCH_FLOATS: usize = WG_THREADS;

/// Workgroup `tgt_scratch` capacity (`256 × f32 = 1,024 B`).
pub const PRESCREEN_TGT_SCRATCH_FLOATS: usize = WG_THREADS;

/// Total `prescreen_population` workgroup SRAM footprint (`8,192 B <= 15,872 B`).
pub const PRESCREEN_SRAM_BYTES: usize =
    (PRESCREEN_SCAN_SCRATCH_FLOATS + PRESCREEN_ERR_SCRATCH_FLOATS + PRESCREEN_TGT_SCRATCH_FLOATS)
        * size_of::<f32>();

const _: () = assert!(PRESCREEN_SEED_PARAM_STRIDE == 16);
const _: () = assert!(PRESCREEN_SRAM_BYTES == 8_192);
const _: () = assert!(PRESCREEN_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);

/// Minimum `eps_floor` protecting denominator inversion in normalized tile MSE.
const MIN_EPS_FLOOR: f32 = 1.0e-12;

#[derive(Copy, Clone)]
struct PrescreenQuadCoeffs {
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

#[inline(always)]
fn read_seed_param(seed_params: &[f32], base: usize, offset: usize) -> f32 {
    seed_params[clamp_idx(base + offset, seed_params.len())]
}

#[inline]
fn load_prescreen_quad_coeffs(
    seed_params: &[f32],
    seed_idx: usize,
    quad_idx: usize,
    num_seeds: usize,
    num_quads: usize,
) -> PrescreenQuadCoeffs {
    let safe_seed = clamp_idx(seed_idx, num_seeds);
    let per_quad_len = num_seeds * num_quads * PRESCREEN_SEED_PARAM_STRIDE;
    let base = if seed_params.len() >= per_quad_len {
        (safe_seed * num_quads + quad_idx) * PRESCREEN_SEED_PARAM_STRIDE
    } else {
        safe_seed * PRESCREEN_SEED_PARAM_STRIDE
    };

    let p0 = read_seed_param(seed_params, base, 0);
    let p1 = read_seed_param(seed_params, base, 1);
    let fc_ratio = if is_finite_bits(p0) { p0 } else { 1.0e-3 };
    let g = prewarp_tan_pi_f_over_fs_f32(fc_ratio);
    let q_val = if is_finite_bits(p1) {
        clamp_f32_safe(p1, 0.05, 64.0)
    } else {
        FRAC_1_SQRT_2
    };
    let k = rcp_pos_f32(q_val);

    let lru = SonaParaLruDuhemCoeffs4::from_clamped(
        GpuVec4::splat(flush_zero(read_seed_param(seed_params, base, 6))),
        GpuVec4::splat(flush_zero(read_seed_param(seed_params, base, 7))),
        GpuVec4::splat(flush_zero(read_seed_param(seed_params, base, 8))),
        GpuVec4::splat(flush_zero(read_seed_param(seed_params, base, 9))),
        GpuVec4::splat(flush_zero(read_seed_param(seed_params, base, 10))),
        GpuVec4::splat(flush_zero(read_seed_param(seed_params, base, 11))),
    );

    PrescreenQuadCoeffs {
        svf: TptSvfCoeffs4::from_gk(GpuVec4::splat(g), GpuVec4::splat(k)),
        two_svf_g: clamp_f32_safe(g + g, 0.0, 128.0),
        w_lp: flush_zero(read_seed_param(seed_params, base, 2)),
        w_bp: flush_zero(read_seed_param(seed_params, base, 3)),
        w_hp: flush_zero(read_seed_param(seed_params, base, 4)),
        w_dir: flush_zero(read_seed_param(seed_params, base, 5)),
        lru,
        rect_sign: clamp_f32_safe(read_seed_param(seed_params, base, 12), -1.0, 1.0),
        sag_depth: clamp_f32_safe(read_seed_param(seed_params, base, 13), -4.0, 4.0),
        adaa_drive: clamp_f32_safe(read_seed_param(seed_params, base, 14), -64.0, 64.0),
        lru_bias_drive: clamp_f32_safe(read_seed_param(seed_params, base, 15), -64.0, 64.0),
    }
}

#[inline(always)]
fn store_tpt_elem(
    scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    idx: usize,
    e: TptScanElem,
) {
    let i = clamp_idx(idx, WG_THREADS);
    scan_scratch[i] = e.m00;
    scan_scratch[WG_THREADS + i] = e.m01;
    scan_scratch[2 * WG_THREADS + i] = e.m10;
    scan_scratch[3 * WG_THREADS + i] = e.m11;
    scan_scratch[4 * WG_THREADS + i] = e.b0;
    scan_scratch[5 * WG_THREADS + i] = e.b1;
}

#[inline(always)]
fn load_tpt_elem(scan_scratch: &[f32; PRESCREEN_SCAN_SCRATCH_FLOATS], idx: usize) -> TptScanElem {
    let i = clamp_idx(idx, WG_THREADS);
    TptScanElem {
        m00: scan_scratch[i],
        m01: scan_scratch[WG_THREADS + i],
        m10: scan_scratch[2 * WG_THREADS + i],
        m11: scan_scratch[3 * WG_THREADS + i],
        b0: scan_scratch[4 * WG_THREADS + i],
        b1: scan_scratch[5 * WG_THREADS + i],
    }
}

/// Runs a `256`-wide exclusive Blelloch associative scan over [`TptScanElem`] in SoA SRAM.
#[inline]
fn blelloch_exclusive_tpt_scan_256(
    scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    lid: usize,
    elem: TptScanElem,
) -> TptScanElem {
    store_tpt_elem(scan_scratch, lid, elem);
    wg_barrier();

    let mut stride = 1_usize;
    while stride < WG_THREADS {
        let step = stride << 1;
        let idx = (lid + 1) * step - 1;
        if idx < WG_THREADS {
            let left = load_tpt_elem(scan_scratch, idx - stride);
            let right = load_tpt_elem(scan_scratch, idx);
            store_tpt_elem(scan_scratch, idx, compose_tpt_scan(left, right));
        }
        wg_barrier();
        stride = step;
    }

    if lid == WG_THREADS - 1 {
        store_tpt_elem(scan_scratch, WG_THREADS - 1, TptScanElem::identity());
    }
    wg_barrier();

    let mut down_stride = WG_THREADS >> 1;
    while down_stride > 0 {
        let step = down_stride << 1;
        let idx = (lid + 1) * step - 1;
        if idx < WG_THREADS {
            let left_idx = idx - down_stride;
            let t_left = load_tpt_elem(scan_scratch, left_idx);
            let t_right = load_tpt_elem(scan_scratch, idx);
            store_tpt_elem(scan_scratch, left_idx, t_right);
            store_tpt_elem(scan_scratch, idx, compose_tpt_scan(t_right, t_left));
        }
        wg_barrier();
        down_stride >>= 1;
    }

    let prefix = load_tpt_elem(scan_scratch, lid);
    wg_barrier();
    prefix
}

#[inline]
fn eval_tpt_svf_channel(
    scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    lid: usize,
    active: bool,
    u_ch: f32,
    coeffs: &PrescreenQuadCoeffs,
) -> (f32, f32) {
    let a1 = coeffs.svf.a1.to_array()[0];
    let a2 = coeffs.svf.a2.to_array()[0];
    let a3 = coeffs.svf.a3.to_array()[0];
    let elem = if active {
        TptScanElem::from_step(a1, a2, a3, u_ch)
    } else {
        TptScanElem::identity()
    };
    let prefix = blelloch_exclusive_tpt_scan_256(scan_scratch, lid, elem);
    if !active {
        return (0.0, 0.0);
    }
    let [s1_init, s2_init] = apply_tpt_init(prefix, [0.0, 0.0]);
    let mut s1 = GpuVec4::splat(s1_init);
    let mut s2 = GpuVec4::splat(s2_init);
    let svf_out = step_tpt_svf_vec4(GpuVec4::splat(u_ch), &mut s1, &mut s2, &coeffs.svf);
    let lp = svf_out.lp.to_array()[0];
    let bp = svf_out.bp.to_array()[0];
    let hp = svf_out.hp.to_array()[0];
    let tone =
        flush_zero(coeffs.w_lp * lp + coeffs.w_bp * bp + coeffs.w_hp * hp + coeffs.w_dir * u_ch);
    (tone, flush_zero(bp))
}

#[inline]
fn step_paralru_tile_on_thread0(
    scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    valid_len: usize,
    coeffs: &PrescreenQuadCoeffs,
) {
    let mut v_l = GpuVec4::zero();
    let mut v_r = GpuVec4::zero();
    for t in 0..valid_len {
        let tone_l = scan_scratch[t];
        let tone_r = scan_scratch[WG_THREADS + t];
        let bp_l = scan_scratch[2 * WG_THREADS + t];
        let bp_r = scan_scratch[3 * WG_THREADS + t];

        let u_rect_l = GpuVec4::splat(coeffs.rect_sign * tone_l);
        let u_rect_r = GpuVec4::splat(coeffs.rect_sign * tone_r);
        let flux_l = GpuVec4::splat(coeffs.two_svf_g * bp_l);
        let flux_r = GpuVec4::splat(coeffs.two_svf_g * bp_r);

        let v_raw_l = step_paralru_duhem_vec4(u_rect_l, flux_l, &mut v_l, &coeffs.lru);
        let v_raw_r = step_paralru_duhem_vec4(u_rect_r, flux_r, &mut v_r, &coeffs.lru);
        let (v_eff_l, v_eff_r) =
            apply_stereo_link_vec4_pair(v_raw_l, v_raw_r, coeffs.lru.stereo_link);

        scan_scratch[4 * WG_THREADS + t] = v_eff_l.to_array()[0];
        scan_scratch[5 * WG_THREADS + t] = v_eff_r.to_array()[0];
    }
}

/// Executes one `(seed_idx = wg_id.x, tile_idx = wg_id.y, stem_idx = wg_id.z)` tile of
/// `prescreen_population` across `256` threads (`8,192 B` SRAM, zero `FDIV`/`FSQRT`/`IDIV`).
#[inline]
pub fn prescreen_population_impl(
    wg_id: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &PrescreenUniforms,
    input_audio: &[Vec2],
    target_audio: &[Vec2],
    seed_params: &[f32],
    seed_losses: &mut [f32],
    scan_scratch: &mut [f32; PRESCREEN_SCAN_SCRATCH_FLOATS],
    err_scratch: &mut [f32; PRESCREEN_ERR_SCRATCH_FLOATS],
    tgt_scratch: &mut [f32; PRESCREEN_TGT_SCRATCH_FLOATS],
) {
    let lid = lid_u32 as usize;
    let seed_idx = wg_id.x as usize;
    let tile_idx = wg_id.y as usize;
    let stem_idx = wg_id.z as usize;

    let num_seeds = (uniforms.num_seeds as usize).max(1);
    let num_frames = uniforms.num_frames as usize;
    let num_quads = (uniforms.num_quads as usize).max(1);
    let num_stems = (uniforms.num_stems as usize).max(1);
    let num_tiles = if num_frames == 0 {
        1
    } else {
        ((num_frames - 1) >> 8) + 1
    };

    let tile_base = tile_idx * WG_THREADS;
    let valid_len = sub_sat_usize(num_frames, tile_base).min(WG_THREADS);
    let frame = tile_base + lid;
    let active = frame < num_frames;
    let global_frame = stem_idx * num_frames + frame;

    let in_lr = if active {
        let v = input_audio[clamp_idx(global_frame, input_audio.len())];
        Vec2::new(flush_zero(v.x), flush_zero(v.y))
    } else {
        Vec2::ZERO
    };

    let mut pred_l = 0.0_f32;
    let mut pred_r = 0.0_f32;

    for q in 0..num_quads {
        let coeffs = load_prescreen_quad_coeffs(seed_params, seed_idx, q, num_seeds, num_quads);
        let (tone_l, bp_l) = eval_tpt_svf_channel(scan_scratch, lid, active, in_lr.x, &coeffs);
        let (tone_r, bp_r) = eval_tpt_svf_channel(scan_scratch, lid, active, in_lr.y, &coeffs);

        let safe_lid = clamp_idx(lid, WG_THREADS);
        scan_scratch[safe_lid] = tone_l;
        scan_scratch[WG_THREADS + safe_lid] = tone_r;
        scan_scratch[2 * WG_THREADS + safe_lid] = bp_l;
        scan_scratch[3 * WG_THREADS + safe_lid] = bp_r;
        scan_scratch[4 * WG_THREADS + safe_lid] = 0.0;
        scan_scratch[5 * WG_THREADS + safe_lid] = 0.0;
        wg_barrier();

        if lid == 0 {
            step_paralru_tile_on_thread0(scan_scratch, valid_len, &coeffs);
        }
        wg_barrier();

        let ve_l = if active {
            scan_scratch[4 * WG_THREADS + safe_lid]
        } else {
            0.0
        };
        let ve_r = if active {
            scan_scratch[5 * WG_THREADS + safe_lid]
        } else {
            0.0
        };
        let sag_l = (1.0 - coeffs.sag_depth * ve_l).max(MIN_SAG_GAIN);
        let sag_r = (1.0 - coeffs.sag_depth * ve_r).max(MIN_SAG_GAIN);
        let r_curr_l = if active {
            flush_zero(tone_l * coeffs.adaa_drive * sag_l + coeffs.lru_bias_drive * ve_l)
        } else {
            0.0
        };
        let r_curr_r = if active {
            flush_zero(tone_r * coeffs.adaa_drive * sag_r + coeffs.lru_bias_drive * ve_r)
        } else {
            0.0
        };

        wg_barrier();
        scan_scratch[safe_lid] = r_curr_l;
        scan_scratch[WG_THREADS + safe_lid] = r_curr_r;
        wg_barrier();

        let r_prev_l = if lid > 0 && active {
            scan_scratch[lid - 1]
        } else {
            0.0
        };
        let r_prev_r = if lid > 0 && active {
            scan_scratch[WG_THREADS + lid - 1]
        } else {
            0.0
        };

        let (odd_l, _even_l, _) =
            eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(r_prev_l), GpuVec4::splat(r_curr_l));
        let (odd_r, _even_r, _) =
            eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(r_prev_r), GpuVec4::splat(r_curr_r));

        if active {
            pred_l += odd_l.to_array()[0];
            pred_r += odd_r.to_array()[0];
        }
        wg_barrier();
    }

    let tgt_lr = if active {
        let v = target_audio[clamp_idx(global_frame, target_audio.len())];
        Vec2::new(flush_zero(v.x), flush_zero(v.y))
    } else {
        Vec2::ZERO
    };
    let diff_l = if active { pred_l - tgt_lr.x } else { 0.0 };
    let diff_r = if active { pred_r - tgt_lr.y } else { 0.0 };
    let e2 = if active {
        flush_zero(diff_l * diff_l + diff_r * diff_r)
    } else {
        0.0
    };
    let t2 = if active {
        flush_zero(tgt_lr.x * tgt_lr.x + tgt_lr.y * tgt_lr.y)
    } else {
        0.0
    };

    let (err_total, tgt_total) =
        reduce_sum_f32_pair_256(err_scratch, tgt_scratch, lid, e2, t2, sg_lid);

    if lid == 0 && seed_idx < num_seeds && tile_idx < num_tiles && stem_idx < num_stems {
        let eps = if is_finite_bits(uniforms.eps_floor) && uniforms.eps_floor > MIN_EPS_FLOOR {
            uniforms.eps_floor
        } else {
            MIN_EPS_FLOOR
        };
        let loss = flush_zero(err_total * rcp_pos_f32(tgt_total + eps));
        let out_idx = (stem_idx * num_seeds + seed_idx) * num_tiles + tile_idx;
        seed_losses[clamp_idx(out_idx, seed_losses.len())] = loss;
    }
}
