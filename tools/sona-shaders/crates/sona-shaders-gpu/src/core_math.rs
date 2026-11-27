#![forbid(unsafe_code)]

//! Shared `#![no_std]` SPIR-V / Naga helpers, `QuadParamsUnpacked` unpacking, scalar `dnam-kernels`
//! step wrappers, associative `256`-wide scans, and analytical $C^2$ wavefolder / smooth `ParaLRU`
//! derivatives (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod adaa_deriv;
pub mod scans;

pub use adaa_deriv::{
    DuhemStepTerms, eval_c2_wf_odd_with_derivs, eval_duhem_step_terms,
    eval_paralru_duhem_affine_step, is_duhem_decoupled,
};
pub use scans::{
    linear_adjoint_reverse_suffix_256_lr, linear_forward_exclusive_scan_128_pair,
    linear_forward_inclusive_scan_256_lr, reduce_batch6_in_scratch, reduce_sum_256_in_aux,
    tpt_adjoint_const_m_reduce_256_lr, tpt_adjoint_reverse_suffix_256,
    tpt_adjoint_reverse_suffix_with_after_256, tpt_const_m_scan_128_pair, tpt_const_m_scan_256_lr,
    tpt_exclusive_scan_128_pair, tpt_forward_exclusive_scan_256, wg_barrier,
};

use crate::abi::TILE_FRAMES;
pub use crate::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE,
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_LANE_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, param_slots,
};
use dnam_kernels::{
    G_PRE_E_MAX, GpuVec4, IEEE754_F32_EXP_MASK, KernelVec4, MAX_ADAA_DRIVE, MAX_ALG_R,
    MAX_G_PRE_GAIN, MAX_GLU_MIX, MAX_LRU_BIAS_DRIVE, MAX_OUT_GAIN, MAX_PARALRU_DUHEM_GAMMA_F32,
    MAX_PARALRU_R_F32, MIN_TPT_G, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4, SonaToneFolded3,
    SonaToneFoldedBlock0, TptSvfCoeffs4, apply_stereo_link_vec4_pair, canonicalize_zero_f32_bits,
    scrub_clamped_alg_endpoint_f32, step_paralru_duhem_v4_vec4, step_tpt_svf_folded3_vec4,
    step_tpt_svf_vec4,
};
use spirv_std::glam::Vec4;

const MAX_NONNEG_MOD_SCALE: f32 = 1.0e12;
const G_PRE_V_ABS_MAX: f32 = 1.0e3;

const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(CORE_INPUT_VEC4_STRIDE == 2);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == 28);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == param_slots::OUT_GAIN + 1);
const _: () = assert!(TILE_CKPT_VEC4_STRIDE == 12);
const _: () = assert!(FEATURE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_LANE_VEC4_STRIDE == 11 && TILE_ADJ_BOUNDARY_VEC4_STRIDE == 44);
const _: () = assert!(SOFT_RECT_EPS > 0.0 && MIN_TPT_G > 0.0 && MAX_ALG_R == 4096.0);
const _: () = assert!(G_PRE_V_ABS_MAX * G_PRE_V_ABS_MAX == G_PRE_E_MAX);

/// Converts a `glam::Vec4` to a `dnam_kernels::GpuVec4`.
#[inline(always)]
#[must_use]
pub fn vec4_to_gpu(v: Vec4) -> GpuVec4 {
    GpuVec4::from_array([v.x, v.y, v.z, v.w])
}

/// Converts a `dnam_kernels::GpuVec4` to a `glam::Vec4`.
#[inline(always)]
#[must_use]
pub fn gpu_to_vec4(v: GpuVec4) -> Vec4 {
    let a = v.to_array();
    Vec4::new(a[0], a[1], a[2], a[3])
}

/// Reads lane `lane & 3` of a `Vec4` without array-indexing panic checks.
#[inline(always)]
#[must_use]
pub fn get_lane(v: Vec4, lane: usize) -> f32 {
    match lane & 3 {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => v.w,
    }
}

/// Writes `val` to lane `lane & 3` of a `Vec4` without array-indexing panic checks.
#[inline(always)]
pub fn set_lane(v: &mut Vec4, lane: usize, val: f32) {
    match lane & 3 {
        0 => v.x = val,
        1 => v.y = val,
        2 => v.z = val,
        _ => v.w = val,
    }
}

/// Loads `buf[idx]` cleanly, returning `Vec4::ZERO` when out of bounds without emitting `SilentExit` breaks.
#[inline(always)]
#[must_use]
pub fn load_vec4(buf: &[Vec4], idx: usize) -> Vec4 {
    if idx < buf.len() {
        buf[idx]
    } else {
        Vec4::ZERO
    }
}

/// Stores `val` at `buf[idx]` cleanly when `idx < buf.len()` without emitting `SilentExit` breaks.
#[inline(always)]
pub fn store_vec4(buf: &mut [Vec4], idx: usize, val: Vec4) {
    if idx < buf.len() {
        buf[idx] = val;
    }
}

/// Panic-free saturating `usize` subtraction avoiding `saturating_sub` (unimplemented on `spirv`).
#[inline(always)]
#[must_use]
pub const fn sub_sat_usize(a: usize, b: usize) -> usize {
    if a > b { a - b } else { 0 }
}

/// Division-free integer ceiling division `ceil(num / den)` for `den > 0`.
#[inline(always)]
#[must_use]
pub fn div_ceil_divfree(num: usize, den: usize) -> usize {
    if num == 0 {
        return 0;
    }
    if den <= 1 {
        return num;
    }
    let n = num - 1;
    let mut sub = den;
    let mut bit = 1_usize;
    while sub <= (n >> 1) {
        sub <<= 1;
        bit <<= 1;
    }
    let mut rem = n;
    let mut q = 0_usize;
    while bit > 0 {
        if rem >= sub {
            rem -= sub;
            q |= bit;
        }
        sub >>= 1;
        bit >>= 1;
    }
    q + 1
}

/// Pure-bitmask finiteness test avoiding `f32::is_finite()` (`OpConstant Inf` fails Naga).
#[inline(always)]
#[must_use]
pub const fn is_finite_bits(x: f32) -> bool {
    (x.to_bits() & IEEE754_F32_EXP_MASK) != IEEE754_F32_EXP_MASK
}

/// Scrubs `NaN`/`±Inf` to `+0.0`, flushes subnormals, and maps `-0.0` to `+0.0`.
#[inline(always)]
#[must_use]
pub const fn flush_zero(x: f32) -> f32 {
    canonicalize_zero_f32_bits(x)
}

/// Scrubs non-finite endpoints to `+0.0`, clamps to `[-4096.0, 4096.0]`, and canonicalizes `+0.0`.
#[inline(always)]
#[must_use]
pub fn scrub_endpoint(x: f32) -> f32 {
    scrub_clamped_alg_endpoint_f32(x)
}

/// Panic-free clamp `[lo, hi]` with `+0.0` fallback on non-finite inputs matching `clamp_range_safe_vec4`.
#[inline(always)]
#[must_use]
pub fn clamp_range_safe(x: f32, lo: f32, hi: f32) -> f32 {
    if is_finite_bits(x) {
        if x <= lo {
            flush_zero(lo)
        } else if x >= hi {
            flush_zero(hi)
        } else {
            flush_zero(x)
        }
    } else {
        0.0
    }
}

/// Panic-free non-negative clamp `[0.0, hi]` matching `dnam_kernels::sona_core::paralru::clamp_nonneg_vec4_safe`.
#[inline(always)]
#[must_use]
pub fn clamp_nonneg_safe(x: f32, hi: f32) -> f32 {
    if is_finite_bits(x) {
        if x <= 0.0 {
            0.0
        } else if x >= hi {
            hi
        } else {
            canonicalize_zero_f32_bits(x)
        }
    } else if x.to_bits() == IEEE754_F32_EXP_MASK {
        hi
    } else {
        0.0
    }
}

/// Complete unpacked `28 × Vec4` SONA v4.0 parameter bundle for one `(stem, quad)` `CoreQuad`.
#[derive(Copy, Clone)]
pub struct QuadParamsUnpacked {
    pub svf_a1: Vec4,
    pub svf_a2: Vec4,
    pub svf_a3: Vec4,
    pub svf_k: Vec4,
    pub tone_c_lp: Vec4,
    pub tone_c_bp: Vec4,
    pub tone_c_kbp: Vec4,
    pub tone_c_u: Vec4,
    pub lru_alpha_rel: Vec4,
    pub lru_alpha_atk: Vec4,
    pub lru_kappa: Vec4,
    pub lru_gamma_duhem: Vec4,
    pub lru_signed_mix: Vec4,
    pub lru_stereo_link: Vec4,
    pub svf_g: Vec4,
    pub svf_k_nl: Vec4,
    pub rect_sign: Vec4,
    pub g0: Vec4,
    pub g_inf: Vec4,
    pub d_pre: Vec4,
    pub se_w_sq: Vec4,
    pub se_d_bus: Vec4,
    pub adaa_drive: Vec4,
    pub lru_bias_drive: Vec4,
    pub lin_mix: Vec4,
    pub wavefolder_mix: Vec4,
    pub glu_mix: Vec4,
    pub out_gain: Vec4,
}

impl QuadParamsUnpacked {
    /// Loads all `28` `Vec4` slots starting at `quad_params[q_base + 0..28]`.
    #[inline]
    #[must_use]
    pub fn load(quad_params: &[Vec4], q_base: usize) -> Self {
        Self {
            svf_a1: load_vec4(quad_params, q_base + param_slots::SVF_A1),
            svf_a2: load_vec4(quad_params, q_base + param_slots::SVF_A2),
            svf_a3: load_vec4(quad_params, q_base + param_slots::SVF_A3),
            svf_k: load_vec4(quad_params, q_base + param_slots::SVF_K),
            tone_c_lp: load_vec4(quad_params, q_base + param_slots::TONE_C_LP),
            tone_c_bp: load_vec4(quad_params, q_base + param_slots::TONE_C_BP),
            tone_c_kbp: load_vec4(quad_params, q_base + param_slots::TONE_C_KBP),
            tone_c_u: load_vec4(quad_params, q_base + param_slots::TONE_C_U),
            lru_alpha_rel: load_vec4(quad_params, q_base + param_slots::LRU_ALPHA_REL),
            lru_alpha_atk: load_vec4(quad_params, q_base + param_slots::LRU_ALPHA_ATK),
            lru_kappa: load_vec4(quad_params, q_base + param_slots::LRU_KAPPA),
            lru_gamma_duhem: load_vec4(quad_params, q_base + param_slots::LRU_GAMMA_DUHEM),
            lru_signed_mix: load_vec4(quad_params, q_base + param_slots::LRU_SIGNED_MIX),
            lru_stereo_link: load_vec4(quad_params, q_base + param_slots::LRU_STEREO_LINK),
            svf_g: load_vec4(quad_params, q_base + param_slots::SVF_G),
            svf_k_nl: load_vec4(quad_params, q_base + param_slots::SVF_K_NL),
            rect_sign: load_vec4(quad_params, q_base + param_slots::RECT_SIGN),
            g0: load_vec4(quad_params, q_base + param_slots::G0),
            g_inf: load_vec4(quad_params, q_base + param_slots::G_INF),
            d_pre: load_vec4(quad_params, q_base + param_slots::D_PRE),
            se_w_sq: load_vec4(quad_params, q_base + param_slots::SE_W_SQ),
            se_d_bus: load_vec4(quad_params, q_base + param_slots::SE_D_BUS),
            adaa_drive: load_vec4(quad_params, q_base + param_slots::ADAA_DRIVE),
            lru_bias_drive: load_vec4(quad_params, q_base + param_slots::LRU_BIAS_DRIVE),
            lin_mix: load_vec4(quad_params, q_base + param_slots::LIN_MIX),
            wavefolder_mix: load_vec4(quad_params, q_base + param_slots::WAVEFOLDER_MIX),
            glu_mix: load_vec4(quad_params, q_base + param_slots::GLU_MIX),
            out_gain: load_vec4(quad_params, q_base + param_slots::OUT_GAIN),
        }
    }

    /// Extracts scalar parameters and clamp-domain openness flags for lane `c in 0..4`.
    #[inline]
    #[must_use]
    pub fn lane(&self, c: usize) -> QuadLaneParams {
        let k = get_lane(self.svf_k, c);
        let svf_g = get_lane(self.svf_g, c);
        let rel_raw = get_lane(self.lru_alpha_rel, c);
        let atk_raw = get_lane(self.lru_alpha_atk, c);
        let kap_raw = get_lane(self.lru_kappa, c);
        let gam_raw = get_lane(self.lru_gamma_duhem, c);
        let eta_raw = get_lane(self.lru_signed_mix, c);
        let link_raw = get_lane(self.lru_stereo_link, c);
        let g0_raw = get_lane(self.g0, c);
        let g_inf_raw = get_lane(self.g_inf, c);
        let d_pre_raw = get_lane(self.d_pre, c);
        let adaa_raw = get_lane(self.adaa_drive, c);
        let bias_raw = get_lane(self.lru_bias_drive, c);
        let lin_raw = get_lane(self.lin_mix, c);
        let wf_raw = get_lane(self.wavefolder_mix, c);
        let glu_raw = get_lane(self.glu_mix, c);
        let out_raw = get_lane(self.out_gain, c);

        QuadLaneParams {
            a1: get_lane(self.svf_a1, c),
            a2: get_lane(self.svf_a2, c),
            a3: get_lane(self.svf_a3, c),
            k,
            two_k: k + k,
            c_lp: get_lane(self.tone_c_lp, c),
            c_bp: get_lane(self.tone_c_bp, c),
            c_kbp: get_lane(self.tone_c_kbp, c),
            c_u: get_lane(self.tone_c_u, c),
            alpha_rel: clamp_nonneg_safe(rel_raw, MAX_PARALRU_R_F32),
            alpha_atk: clamp_nonneg_safe(atk_raw, 1.0),
            kappa: clamp_nonneg_safe(kap_raw, 1.0),
            gamma_duhem: clamp_nonneg_safe(gam_raw, MAX_PARALRU_DUHEM_GAMMA_F32),
            signed_mix: clamp_nonneg_safe(eta_raw, 1.0),
            stereo_link: clamp_nonneg_safe(link_raw, 1.0),
            alpha_rel_open: is_finite_bits(rel_raw)
                && rel_raw >= 0.0
                && rel_raw <= MAX_PARALRU_R_F32,
            alpha_atk_open: is_finite_bits(atk_raw) && atk_raw >= 0.0 && atk_raw <= 1.0,
            kappa_open: is_finite_bits(kap_raw) && kap_raw >= 0.0 && kap_raw <= 1.0,
            gamma_open: is_finite_bits(gam_raw)
                && gam_raw >= 0.0
                && gam_raw <= MAX_PARALRU_DUHEM_GAMMA_F32,
            signed_mix_open: is_finite_bits(eta_raw) && eta_raw >= 0.0 && eta_raw <= 1.0,
            stereo_link_open: is_finite_bits(link_raw) && link_raw >= 0.0 && link_raw <= 1.0,
            svf_g,
            two_svf_g: svf_g + svf_g,
            svf_k_nl: get_lane(self.svf_k_nl, c),
            rect_sign: get_lane(self.rect_sign, c),
            g0: clamp_range_safe(g0_raw, 0.0, MAX_G_PRE_GAIN),
            g_inf: clamp_range_safe(g_inf_raw, 0.0, MAX_G_PRE_GAIN),
            d_pre: clamp_range_safe(d_pre_raw, 0.0, MAX_NONNEG_MOD_SCALE),
            g0_open: is_finite_bits(g0_raw) && g0_raw >= 0.0 && g0_raw <= MAX_G_PRE_GAIN,
            g_inf_open: is_finite_bits(g_inf_raw)
                && g_inf_raw >= 0.0
                && g_inf_raw <= MAX_G_PRE_GAIN,
            d_pre_open: is_finite_bits(d_pre_raw)
                && d_pre_raw >= 0.0
                && d_pre_raw <= MAX_NONNEG_MOD_SCALE,
            adaa_drive: clamp_range_safe(adaa_raw, -MAX_ADAA_DRIVE, MAX_ADAA_DRIVE),
            lru_bias_drive: clamp_range_safe(bias_raw, -MAX_LRU_BIAS_DRIVE, MAX_LRU_BIAS_DRIVE),
            adaa_drive_open: is_finite_bits(adaa_raw)
                && adaa_raw >= -MAX_ADAA_DRIVE
                && adaa_raw <= MAX_ADAA_DRIVE,
            lru_bias_drive_open: is_finite_bits(bias_raw)
                && bias_raw >= -MAX_LRU_BIAS_DRIVE
                && bias_raw <= MAX_LRU_BIAS_DRIVE,
            lin_mix: clamp_range_safe(lin_raw, 0.0, 1.0),
            wf_mix: clamp_range_safe(wf_raw, 0.0, 1.0),
            glu_mix: clamp_range_safe(glu_raw, -MAX_GLU_MIX, MAX_GLU_MIX),
            out_gain: clamp_range_safe(out_raw, -MAX_OUT_GAIN, MAX_OUT_GAIN),
            lin_open: is_finite_bits(lin_raw) && lin_raw >= 0.0 && lin_raw <= 1.0,
            wf_open: is_finite_bits(wf_raw) && wf_raw >= 0.0 && wf_raw <= 1.0,
            glu_open: is_finite_bits(glu_raw) && glu_raw >= -MAX_GLU_MIX && glu_raw <= MAX_GLU_MIX,
            out_open: is_finite_bits(out_raw)
                && out_raw >= -MAX_OUT_GAIN
                && out_raw <= MAX_OUT_GAIN,
        }
    }
}

/// Scalar lane parameters for one SONA v4.0 `CoreQuad` lane `c in 0..4`.
#[derive(Copy, Clone)]
pub struct QuadLaneParams {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k: f32,
    pub two_k: f32,
    pub c_lp: f32,
    pub c_bp: f32,
    pub c_kbp: f32,
    pub c_u: f32,
    pub alpha_rel: f32,
    pub alpha_atk: f32,
    pub kappa: f32,
    pub gamma_duhem: f32,
    pub signed_mix: f32,
    pub stereo_link: f32,
    pub alpha_rel_open: bool,
    pub alpha_atk_open: bool,
    pub kappa_open: bool,
    pub gamma_open: bool,
    pub signed_mix_open: bool,
    pub stereo_link_open: bool,
    pub svf_g: f32,
    pub two_svf_g: f32,
    pub svf_k_nl: f32,
    pub rect_sign: f32,
    pub g0: f32,
    pub g_inf: f32,
    pub d_pre: f32,
    pub g0_open: bool,
    pub g_inf_open: bool,
    pub d_pre_open: bool,
    pub adaa_drive: f32,
    pub lru_bias_drive: f32,
    pub adaa_drive_open: bool,
    pub lru_bias_drive_open: bool,
    pub lin_mix: f32,
    pub wf_mix: f32,
    pub glu_mix: f32,
    pub out_gain: f32,
    pub lin_open: bool,
    pub wf_open: bool,
    pub glu_open: bool,
    pub out_open: bool,
}

impl QuadLaneParams {
    /// Builds `dnam_kernels::TptSvfCoeffs4<GpuVec4>` for scalar lane evaluation.
    #[inline(always)]
    #[must_use]
    pub fn to_svf_coeffs(&self) -> TptSvfCoeffs4<GpuVec4> {
        TptSvfCoeffs4 {
            a1: GpuVec4::splat(self.a1),
            a2: GpuVec4::splat(self.a2),
            a3: GpuVec4::splat(self.a3),
            k: GpuVec4::splat(self.k),
            two_k: GpuVec4::splat(self.two_k),
        }
    }

    /// Builds `dnam_kernels::SonaToneFolded3<GpuVec4>` for Blocks `1..3` folded tone evaluation
    /// (`c_bp + c_kbp * k`).
    #[inline(always)]
    #[must_use]
    pub fn to_tone_folded3(&self) -> SonaToneFolded3<GpuVec4> {
        self.to_tone_block0().to_folded3(GpuVec4::splat(self.k))
    }

    /// Builds `dnam_kernels::SonaToneFoldedBlock0<GpuVec4>` for Block `0` PHS folded tone evaluation.
    #[inline(always)]
    #[must_use]
    pub fn to_tone_block0(&self) -> SonaToneFoldedBlock0<GpuVec4> {
        SonaToneFoldedBlock0 {
            c_lp: GpuVec4::splat(self.c_lp),
            c_bp: GpuVec4::splat(self.c_bp),
            c_kbp: GpuVec4::splat(self.c_kbp),
            c_u: GpuVec4::splat(self.c_u),
        }
    }

    /// Builds `dnam_kernels::SonaParaLruDuhemCoeffs4<GpuVec4>` for scalar lane evaluation.
    #[inline(always)]
    #[must_use]
    pub fn to_lru_coeffs(&self) -> SonaParaLruDuhemCoeffs4<GpuVec4> {
        SonaParaLruDuhemCoeffs4 {
            alpha_rel: GpuVec4::splat(self.alpha_rel),
            alpha_atk: GpuVec4::splat(self.alpha_atk),
            kappa: GpuVec4::splat(self.kappa),
            gamma_duhem: GpuVec4::splat(self.gamma_duhem),
            signed_mix: GpuVec4::splat(self.signed_mix),
            stereo_link: GpuVec4::splat(self.stereo_link),
        }
    }
}

/// Scalar `g_pre(v_eff^2)` terms used by forward and adjoint passes.
#[derive(Copy, Clone)]
pub struct GPreTerms {
    pub g_pre: f32,
    pub e: f32,
    pub de: f32,
    pub inv_den: f32,
    pub v_abs: f32,
    pub v_clean: f32,
}

/// Evaluates scalar $C^\infty$ pre-ADAA gain `g_pre(v_eff^2)` and intermediate terms
/// matching `dnam_kernels::eval_g_pre_vec4`.
#[inline(always)]
#[must_use]
pub fn eval_g_pre_terms_f32(v_eff: f32, lp: &QuadLaneParams) -> GPreTerms {
    let v_clean = flush_zero(v_eff);
    let v_abs = v_clean.abs().min(G_PRE_V_ABS_MAX);
    let e = (v_abs * v_abs).min(G_PRE_E_MAX);
    let de = (lp.d_pre * e).min(MAX_NONNEG_MOD_SCALE);
    let inv_den = if de <= 0.0 {
        1.0
    } else {
        GpuVec4::splat(1.0 + de).rcp_ge1_nr2().to_array()[0]
    };
    let g_pre = if de <= 0.0 {
        lp.g0
    } else {
        let lo = lp.g0.min(lp.g_inf);
        let hi = lp.g0.max(lp.g_inf);
        clamp_range_safe((lp.g0 + lp.g_inf * de) * inv_den, lo, hi)
    };
    GPreTerms {
        g_pre,
        e,
        de,
        inv_den,
        v_abs,
        v_clean,
    }
}

/// Evaluates one scalar linear `TPT-SVF` step via `dnam_kernels::step_tpt_svf_vec4`, returning `(lp, bp)`.
#[inline(always)]
#[must_use]
pub fn eval_tpt_svf_lp_bp_scalar(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    svf: &TptSvfCoeffs4<GpuVec4>,
) -> (f32, f32) {
    let mut s1 = GpuVec4::splat(s1_prev);
    let mut s2 = GpuVec4::splat(s2_prev);
    let out = step_tpt_svf_vec4(GpuVec4::splat(u), &mut s1, &mut s2, svf);
    (out.lp.to_array()[0], out.bp.to_array()[0])
}

/// Evaluates one scalar Blocks `1..3` 3-tap folded `TPT-SVF` step, returning `(tone, bp)`.
#[inline(always)]
#[must_use]
pub fn eval_tpt_svf_folded3_scalar(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    svf: &TptSvfCoeffs4<GpuVec4>,
    tone: &SonaToneFolded3<GpuVec4>,
) -> (f32, f32) {
    let mut s1 = GpuVec4::splat(s1_prev);
    let mut s2 = GpuVec4::splat(s2_prev);
    let (tone_v, bp_v) = step_tpt_svf_folded3_vec4(GpuVec4::splat(u), &mut s1, &mut s2, svf, tone);
    (tone_v.to_array()[0], bp_v.to_array()[0])
}

/// Evaluates Block `0` PHS 4-tap folded tone mix `c_lp * LP + (c_bp + c_kbp * k_eff) * BP + c_u * u`.
#[inline(always)]
#[must_use]
pub fn eval_tone_block0_scalar(lp: f32, bp: f32, k_eff: f32, u: f32, p: &QuadLaneParams) -> f32 {
    p.to_tone_block0()
        .eval_vec4(
            GpuVec4::splat(lp),
            GpuVec4::splat(bp),
            GpuVec4::splat(k_eff),
            GpuVec4::splat(u),
        )
        .to_array()[0]
}

/// Steps one joint `(L, R)` SONA v4.0 smooth Selective-Duhem `ParaLRU` + stereo-link sample via
/// `dnam_kernels::step_paralru_duhem_v4_vec4`, returning `(v_raw_l, v_raw_r, v_eff_l, v_eff_r)`.
#[inline(always)]
#[must_use]
pub fn step_duhem_stereo_scalar(
    tone_l: f32,
    tone_r: f32,
    bp_l: f32,
    bp_r: f32,
    v_l: &mut f32,
    v_r: &mut f32,
    p: &QuadLaneParams,
    lru: &SonaParaLruDuhemCoeffs4<GpuVec4>,
) -> (f32, f32, f32, f32) {
    let mut vl = GpuVec4::splat(*v_l);
    let mut vr = GpuVec4::splat(*v_r);
    let raw_l = step_paralru_duhem_v4_vec4(
        GpuVec4::splat(p.rect_sign * tone_l),
        GpuVec4::splat(p.two_svf_g * bp_l),
        &mut vl,
        lru,
    );
    let raw_r = step_paralru_duhem_v4_vec4(
        GpuVec4::splat(p.rect_sign * tone_r),
        GpuVec4::splat(p.two_svf_g * bp_r),
        &mut vr,
        lru,
    );
    let (eff_l, eff_r) = apply_stereo_link_vec4_pair(raw_l, raw_r, lru.stereo_link);
    let rl = raw_l.to_array()[0];
    let rr = raw_r.to_array()[0];
    *v_l = rl;
    *v_r = rr;
    (rl, rr, eff_l.to_array()[0], eff_r.to_array()[0])
}

/// Steps one single-channel SONA v4.0 smooth Selective-Duhem `ParaLRU` sample via
/// `dnam_kernels::step_paralru_duhem_v4_vec4`, returning `v_raw`.
#[inline(always)]
#[must_use]
pub fn step_paralru_duhem_scalar(
    tone: f32,
    bp: f32,
    v: &mut f32,
    p: &QuadLaneParams,
    lru: &SonaParaLruDuhemCoeffs4<GpuVec4>,
) -> f32 {
    let mut v_reg = GpuVec4::splat(*v);
    let raw = step_paralru_duhem_v4_vec4(
        GpuVec4::splat(p.rect_sign * tone),
        GpuVec4::splat(p.two_svf_g * bp),
        &mut v_reg,
        lru,
    )
    .to_array()[0];
    *v = raw;
    raw
}
