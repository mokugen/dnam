#![forbid(unsafe_code)]

//! Shared `#![no_std]` SPIR-V / Naga helpers, `QuadParamsUnpacked` unpacking, scalar `dnam-kernels`
//! step wrappers, associative `256`-wide scans, and analytical `PolyADAA1` / $C^2$ wavefolder
//! derivatives (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod adaa_deriv;
pub mod scans;

pub use adaa_deriv::{DuhemStepTerms, eval_adaa_and_wf_with_derivs, eval_duhem_step_terms};
pub use scans::{
    linear_adjoint_reverse_suffix_256_lr, linear_forward_exclusive_scan_128_pair,
    reduce_batch6_in_scratch, reduce_sum_256_in_aux, tpt_adjoint_reverse_suffix_256,
    tpt_exclusive_scan_128_pair, tpt_exclusive_scan_256, wg_barrier,
};

use crate::abi::TILE_FRAMES;
pub use crate::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE,
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_LANE_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, param_slots,
};
use dnam_kernels::{
    GpuVec4, IEEE754_F32_EXP_MASK, KernelVec4, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32,
    MIN_TPT_G, SOFT_RECT_EPS, SVF_PHS_E_MAX, SonaParaLruDuhemCoeffs4, TptSvfCoeffs4,
    apply_stereo_link_vec4_pair, canonicalize_zero_f32_bits, step_paralru_duhem_vec4,
    step_tpt_svf_phs_vec4, step_tpt_svf_vec4,
};
pub use dnam_kernels::{MAX_ADAA_ENDPOINT, MIN_SAG_GAIN};
use spirv_std::glam::Vec4;

const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(CORE_INPUT_VEC4_STRIDE == 2);
const _: () = assert!(QUAD_PARAMS_VEC4_STRIDE == param_slots::SVF_K_NL + 1);
const _: () = assert!(TILE_CKPT_VEC4_STRIDE == 8);
const _: () = assert!(FEATURE_VEC4_STRIDE == 6);
const _: () = assert!(TILE_ADJ_LANE_VEC4_STRIDE == 6 && TILE_ADJ_BOUNDARY_VEC4_STRIDE == 24);
const _: () = assert!(MIN_SAG_GAIN > 0.0 && SOFT_RECT_EPS > 0.0);

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

/// Scrubs non-finite endpoints to `+0.0`, clamps to `[-MAX_ADAA_ENDPOINT, MAX_ADAA_ENDPOINT]`, and canonicalizes `+0.0`.
#[inline(always)]
#[must_use]
pub fn scrub_endpoint(x: f32) -> f32 {
    let y = canonicalize_zero_f32_bits(x);
    if y < -MAX_ADAA_ENDPOINT {
        -MAX_ADAA_ENDPOINT
    } else if y > MAX_ADAA_ENDPOINT {
        MAX_ADAA_ENDPOINT
    } else {
        y
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

/// Complete unpacked `25 × Vec4` parameter bundle for one `(stem, quad)` `CoreQuad`.
#[derive(Copy, Clone)]
pub struct QuadParamsUnpacked {
    pub svf_a1: Vec4,
    pub svf_a2: Vec4,
    pub svf_a3: Vec4,
    pub svf_k: Vec4,
    pub svf_two_k: Vec4,
    pub tone_lp: Vec4,
    pub tone_bp: Vec4,
    pub tone_hp: Vec4,
    pub tone_ap: Vec4,
    pub tone_dir: Vec4,
    pub lru_alpha_rel: Vec4,
    pub lru_alpha_atk: Vec4,
    pub lru_kappa: Vec4,
    pub lru_gamma_duhem: Vec4,
    pub lru_signed_mix: Vec4,
    pub lru_stereo_link: Vec4,
    pub two_svf_g: Vec4,
    pub rect_sign: Vec4,
    pub sag_depth: Vec4,
    pub adaa_drive: Vec4,
    pub lru_bias_drive: Vec4,
    pub wavefolder_mix: Vec4,
    pub glu_mix: Vec4,
    pub svf_g: Vec4,
    pub svf_k_nl: Vec4,
}

impl QuadParamsUnpacked {
    /// Loads all `25` `Vec4` slots starting at `quad_params[q_base + 0..25]`.
    #[inline]
    #[must_use]
    pub fn load(quad_params: &[Vec4], q_base: usize) -> Self {
        Self {
            svf_a1: load_vec4(quad_params, q_base + param_slots::SVF_A1),
            svf_a2: load_vec4(quad_params, q_base + param_slots::SVF_A2),
            svf_a3: load_vec4(quad_params, q_base + param_slots::SVF_A3),
            svf_k: load_vec4(quad_params, q_base + param_slots::SVF_K),
            svf_two_k: load_vec4(quad_params, q_base + param_slots::SVF_TWO_K),
            tone_lp: load_vec4(quad_params, q_base + param_slots::TONE_LP),
            tone_bp: load_vec4(quad_params, q_base + param_slots::TONE_BP),
            tone_hp: load_vec4(quad_params, q_base + param_slots::TONE_HP),
            tone_ap: load_vec4(quad_params, q_base + param_slots::TONE_AP),
            tone_dir: load_vec4(quad_params, q_base + param_slots::TONE_DIR),
            lru_alpha_rel: load_vec4(quad_params, q_base + param_slots::LRU_ALPHA_REL),
            lru_alpha_atk: load_vec4(quad_params, q_base + param_slots::LRU_ALPHA_ATK),
            lru_kappa: load_vec4(quad_params, q_base + param_slots::LRU_KAPPA),
            lru_gamma_duhem: load_vec4(quad_params, q_base + param_slots::LRU_GAMMA_DUHEM),
            lru_signed_mix: load_vec4(quad_params, q_base + param_slots::LRU_SIGNED_MIX),
            lru_stereo_link: load_vec4(quad_params, q_base + param_slots::LRU_STEREO_LINK),
            two_svf_g: load_vec4(quad_params, q_base + param_slots::TWO_SVF_G),
            rect_sign: load_vec4(quad_params, q_base + param_slots::RECT_SIGN),
            sag_depth: load_vec4(quad_params, q_base + param_slots::SAG_DEPTH),
            adaa_drive: load_vec4(quad_params, q_base + param_slots::ADAA_DRIVE),
            lru_bias_drive: load_vec4(quad_params, q_base + param_slots::LRU_BIAS_DRIVE),
            wavefolder_mix: load_vec4(quad_params, q_base + param_slots::WAVEFOLDER_MIX),
            glu_mix: load_vec4(quad_params, q_base + param_slots::GLU_MIX),
            svf_g: load_vec4(quad_params, q_base + param_slots::SVF_G),
            svf_k_nl: load_vec4(quad_params, q_base + param_slots::SVF_K_NL),
        }
    }

    /// Extracts scalar parameters and clamp-domain openness flags for lane `c in 0..4`.
    #[inline]
    #[must_use]
    pub fn lane(&self, c: usize) -> QuadLaneParams {
        let rel_raw = get_lane(self.lru_alpha_rel, c);
        let atk_raw = get_lane(self.lru_alpha_atk, c);
        let kap_raw = get_lane(self.lru_kappa, c);
        let gam_raw = get_lane(self.lru_gamma_duhem, c);
        let eta_raw = get_lane(self.lru_signed_mix, c);
        let link_raw = get_lane(self.lru_stereo_link, c);
        let wf_raw = get_lane(self.wavefolder_mix, c);

        QuadLaneParams {
            a1: get_lane(self.svf_a1, c),
            a2: get_lane(self.svf_a2, c),
            a3: get_lane(self.svf_a3, c),
            k: get_lane(self.svf_k, c),
            two_k: get_lane(self.svf_two_k, c),
            w_lp: get_lane(self.tone_lp, c),
            w_bp: get_lane(self.tone_bp, c),
            w_hp: get_lane(self.tone_hp, c),
            w_ap: get_lane(self.tone_ap, c),
            w_dir: get_lane(self.tone_dir, c),
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
            two_svf_g: get_lane(self.two_svf_g, c),
            rect_sign: get_lane(self.rect_sign, c),
            sag_depth: get_lane(self.sag_depth, c),
            adaa_drive: get_lane(self.adaa_drive, c),
            lru_bias_drive: get_lane(self.lru_bias_drive, c),
            wf_mix: clamp_nonneg_safe(wf_raw, 1.0),
            wf_open: is_finite_bits(wf_raw) && wf_raw >= 0.0 && wf_raw <= 1.0,
            svf_g: get_lane(self.svf_g, c),
            svf_k_nl: get_lane(self.svf_k_nl, c),
        }
    }
}

/// Scalar lane parameters for one `CoreQuad` lane `c in 0..4`.
#[derive(Copy, Clone)]
pub struct QuadLaneParams {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k: f32,
    pub two_k: f32,
    pub w_lp: f32,
    pub w_bp: f32,
    pub w_hp: f32,
    pub w_ap: f32,
    pub w_dir: f32,
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
    pub two_svf_g: f32,
    pub rect_sign: f32,
    pub sag_depth: f32,
    pub adaa_drive: f32,
    pub lru_bias_drive: f32,
    pub wf_mix: f32,
    pub wf_open: bool,
    pub svf_g: f32,
    pub svf_k_nl: f32,
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

/// Evaluates one scalar `TPT-SVF` step via `dnam_kernels::step_tpt_svf_vec4`, returning
/// `(lp, bp, hp, ap)` without modifying caller-owned state unless desired.
#[inline(always)]
#[must_use]
pub fn eval_tpt_svf_scalar(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    svf: &TptSvfCoeffs4<GpuVec4>,
) -> (f32, f32, f32, f32) {
    let mut s1 = GpuVec4::splat(s1_prev);
    let mut s2 = GpuVec4::splat(s2_prev);
    let out = step_tpt_svf_vec4(GpuVec4::splat(u), &mut s1, &mut s2, svf);
    (
        out.lp.to_array()[0],
        out.bp.to_array()[0],
        out.hp.to_array()[0],
        out.ap.to_array()[0],
    )
}

/// Evaluates one scalar Block 0 Port-Hamiltonian (`PHS`) `TPT-SVF` step via
/// `dnam_kernels::step_tpt_svf_phs_vec4`, updating `s1` and `s2` in place and returning
/// `(lp, bp, hp, ap)`.
#[inline(always)]
#[must_use]
pub fn step_tpt_svf_phs_scalar(
    u: f32,
    s1: &mut f32,
    s2: &mut f32,
    p: &QuadLaneParams,
) -> (f32, f32, f32, f32) {
    let mut s1_v = GpuVec4::splat(*s1);
    let mut s2_v = GpuVec4::splat(*s2);
    let out = step_tpt_svf_phs_vec4(
        GpuVec4::splat(u),
        &mut s1_v,
        &mut s2_v,
        GpuVec4::splat(p.two_svf_g * 0.5),
        GpuVec4::splat(p.k),
        GpuVec4::splat(p.svf_k_nl),
    );
    *s1 = s1_v.to_array()[0];
    *s2 = s2_v.to_array()[0];
    (
        out.lp.to_array()[0],
        out.bp.to_array()[0],
        out.hp.to_array()[0],
        out.ap.to_array()[0],
    )
}

/// Per-frame scalar PHS coefficients and derivatives anchored at pre-step state `s1_prev`.
#[derive(Copy, Clone)]
pub struct PhsFrameTerms {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k_eff: f32,
    pub e: f32,
    pub gamma: f32,
    pub taps: (f32, f32, f32, f32),
}

impl PhsFrameTerms {
    #[inline(always)]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            k_eff: 0.0,
            e: 0.0,
            gamma: 0.0,
            taps: (0.0, 0.0, 0.0, 0.0),
        }
    }
}

/// Evaluates scalar Block 0 PHS coefficients `(a1, a2, a3, k_eff, e, gamma)` alongside exact
/// `0-ULP` forward taps `(lp, bp, hp, ap)` from `step_tpt_svf_phs_scalar`.
#[inline(always)]
#[must_use]
pub fn eval_tpt_svf_phs_frame(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    lp: &QuadLaneParams,
) -> PhsFrameTerms {
    let g_safe = (lp.two_svf_g * 0.5).max(MIN_TPT_G);
    let k_safe = lp.k.max(0.0);
    let k_nl_safe = lp.svf_k_nl.max(0.0);
    let s1_sq = s1_prev * s1_prev;
    let e = s1_sq.min(SVF_PHS_E_MAX);
    let gamma = if s1_sq < SVF_PHS_E_MAX {
        (k_nl_safe + k_nl_safe) * s1_prev
    } else {
        0.0
    };
    let k_eff = k_safe + k_nl_safe * e;
    let denom = 1.0 + g_safe * (g_safe + k_eff);
    let a1 = GpuVec4::splat(denom).rcp_ge1_nr2().to_array()[0];
    let a2 = g_safe * a1;
    let a3 = g_safe * a2;
    let mut s1_tmp = s1_prev;
    let mut s2_tmp = s2_prev;
    let taps = step_tpt_svf_phs_scalar(u, &mut s1_tmp, &mut s2_tmp, lp);
    PhsFrameTerms {
        a1,
        a2,
        a3,
        k_eff,
        e,
        gamma,
        taps,
    }
}

/// Evaluates the linear 4-tap + direct tone mix matching `dnam_kernels::step_core_quad_stereo8`.
#[inline(always)]
#[must_use]
pub fn eval_tone_scalar(lp: f32, bp: f32, hp: f32, ap: f32, u: f32, p: &QuadLaneParams) -> f32 {
    (((lp * p.w_lp + bp * p.w_bp) + hp * p.w_hp) + ap * p.w_ap) + u * p.w_dir
}

/// Steps one joint `(L, R)` Selective-Duhem `ParaLRU` + stereo-link sample via `dnam_kernels`,
/// returning `(v_raw_l, v_raw_r, v_eff_l, v_eff_r)`.
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
    let raw_l = step_paralru_duhem_vec4(
        GpuVec4::splat(p.rect_sign * tone_l),
        GpuVec4::splat(p.two_svf_g * bp_l),
        &mut vl,
        lru,
    );
    let raw_r = step_paralru_duhem_vec4(
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
