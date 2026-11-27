//! Bridge preserving the `SONA v3.0` (`[4,3,2,1]`, `35` tiles, `25 × Vec4` parameters,
//! `PolyADAA1`, `soft_rect_zero_dc`, 5-tap tone, `8 × Vec4` checkpoints) CPU wave cascade
//! & adjoint pipeline until `C3.1` migrates `dnam-vm` and `packages/capture` (`Oracle-01`)
//! onto `SONA v4.0`.

mod math;
mod wave;

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, MIN_TPT_G,
    SonaBlockStateStereo8, SonaParaLruDuhemCoeffs4, SonaToneWeights4, StereoVec8, TptSvfCoeffs4,
    canonicalize_zero_f32_bits,
};

use crate::error::TrainError;
use crate::gpu::abi::{BLOCK_IDX_DIRECT_LINEAR, INVALID_QUAD_SLOT};

pub(crate) use math::{
    FwdFrameState, LaneParams, ReverseCarry, ReverseStepContext, TileGradAccum, flush_zero,
    step_lane_forward, step_lane_reverse,
};
pub(crate) use wave::{run_v3_wave_core_adjoint, run_v3_wave_core_cascade};

/// Canonical `SONA v3.0` `25 × [f32; 4]` parameter stride.
pub(crate) const V3_QUAD_PARAMS_VEC4_STRIDE: usize = v3_quad_param_slots::TOTAL_SLOTS;

/// Canonical `SONA v3.0` `8 × [f32; 4]` tile checkpoint stride.
pub(crate) const V3_TILE_CKPT_VEC4_STRIDE: usize = 8;

/// Canonical `SONA v3.0` `16 × [f32; 4]` exit adjoint boundary stride.
pub(crate) const V3_EXIT_ADJ_BOUNDARY_VEC4_STRIDE: usize = 16;

const V3_BLOCK_QUAD_SLOTS: [[usize; 4]; 4] = [
    [0, 4, 6, 8],
    [1, 5, 9, INVALID_QUAD_SLOT],
    [2, 7, INVALID_QUAD_SLOT, INVALID_QUAD_SLOT],
    [3, INVALID_QUAD_SLOT, INVALID_QUAD_SLOT, INVALID_QUAD_SLOT],
];

const _: () = assert!(V3_QUAD_PARAMS_VEC4_STRIDE == 25);
const _: () = assert!(V3_TILE_CKPT_VEC4_STRIDE == 8);
const _: () = assert!(V3_EXIT_ADJ_BOUNDARY_VEC4_STRIDE == 16);

/// Slot indices (`0..25`) for `SONA v3.0` `CoreQuadParamsStereo4`.
pub(crate) mod v3_quad_param_slots {
    pub(crate) const SVF_A1: usize = 0;
    pub(crate) const SVF_A2: usize = 1;
    pub(crate) const SVF_A3: usize = 2;
    pub(crate) const SVF_K: usize = 3;
    pub(crate) const SVF_TWO_K: usize = 4;
    pub(crate) const TONE_LP: usize = 5;
    pub(crate) const TONE_BP: usize = 6;
    pub(crate) const TONE_HP: usize = 7;
    pub(crate) const TONE_AP: usize = 8;
    pub(crate) const TONE_DIR: usize = 9;
    pub(crate) const LRU_ALPHA_REL: usize = 10;
    pub(crate) const LRU_ALPHA_ATK: usize = 11;
    pub(crate) const LRU_KAPPA: usize = 12;
    pub(crate) const LRU_GAMMA_DUHEM: usize = 13;
    pub(crate) const LRU_SIGNED_MIX: usize = 14;
    pub(crate) const LRU_STEREO_LINK: usize = 15;
    pub(crate) const TWO_SVF_G: usize = 16;
    pub(crate) const RECT_SIGN: usize = 17;
    pub(crate) const SAG_DEPTH: usize = 18;
    pub(crate) const ADAA_DRIVE: usize = 19;
    pub(crate) const LRU_BIAS_DRIVE: usize = 20;
    pub(crate) const WAVEFOLDER_MIX: usize = 21;
    pub(crate) const GLU_MIX: usize = 22;
    pub(crate) const SVF_G: usize = 23;
    pub(crate) const SVF_K_NL: usize = 24;
    pub(crate) const TOTAL_SLOTS: usize = SVF_K_NL + 1;
}

#[inline(always)]
pub(crate) const fn v3_block_quad_count(block_idx: usize, num_quads: usize) -> usize {
    let row = V3_BLOCK_QUAD_SLOTS[block_idx & 3];
    let mut count = 0_usize;
    let mut k = 0_usize;
    while k < 4 {
        if row[k] < num_quads {
            count += 1;
        }
        k += 1;
    }
    count
}

#[inline(always)]
pub(crate) const fn v3_resolve_quad_idx(block_idx: u32, q_local: usize, num_quads: usize) -> usize {
    if block_idx == BLOCK_IDX_DIRECT_LINEAR {
        if q_local < num_quads {
            q_local
        } else {
            INVALID_QUAD_SLOT
        }
    } else if q_local < 4 {
        let slot = V3_BLOCK_QUAD_SLOTS[(block_idx as usize) & 3][q_local];
        if slot < num_quads {
            slot
        } else {
            INVALID_QUAD_SLOT
        }
    } else {
        INVALID_QUAD_SLOT
    }
}

#[inline(always)]
pub(crate) fn canon_vec4(v: GpuVec4) -> [f32; 4] {
    v.to_array().map(canonicalize_zero_f32_bits)
}

pub(crate) fn pack_v3_core_quad_params(
    params: &CoreQuadParamsStereo4<GpuVec4>,
) -> [[f32; 4]; V3_QUAD_PARAMS_VEC4_STRIDE] {
    let g = params.two_svf_g.mul(GpuVec4::splat(0.5));
    let mut out = [[0.0_f32; 4]; V3_QUAD_PARAMS_VEC4_STRIDE];
    out[v3_quad_param_slots::SVF_A1] = params.svf.a1.to_array();
    out[v3_quad_param_slots::SVF_A2] = params.svf.a2.to_array();
    out[v3_quad_param_slots::SVF_A3] = params.svf.a3.to_array();
    out[v3_quad_param_slots::SVF_K] = params.svf.k.to_array();
    out[v3_quad_param_slots::SVF_TWO_K] = params.svf.two_k.to_array();
    out[v3_quad_param_slots::TONE_LP] = params.tone.lp.to_array();
    out[v3_quad_param_slots::TONE_BP] = params.tone.bp.to_array();
    out[v3_quad_param_slots::TONE_HP] = params.tone.hp.to_array();
    out[v3_quad_param_slots::TONE_AP] = params.tone.ap.to_array();
    out[v3_quad_param_slots::TONE_DIR] = params.tone.dir.to_array();
    out[v3_quad_param_slots::LRU_ALPHA_REL] = params.lru.alpha_rel.to_array();
    out[v3_quad_param_slots::LRU_ALPHA_ATK] = params.lru.alpha_atk.to_array();
    out[v3_quad_param_slots::LRU_KAPPA] = params.lru.kappa.to_array();
    out[v3_quad_param_slots::LRU_GAMMA_DUHEM] = params.lru.gamma_duhem.to_array();
    out[v3_quad_param_slots::LRU_SIGNED_MIX] = params.lru.signed_mix.to_array();
    out[v3_quad_param_slots::LRU_STEREO_LINK] = params.lru.stereo_link.to_array();
    out[v3_quad_param_slots::TWO_SVF_G] = params.two_svf_g.to_array();
    out[v3_quad_param_slots::RECT_SIGN] = params.rect_sign.to_array();
    out[v3_quad_param_slots::SAG_DEPTH] = params.sag_depth.to_array();
    out[v3_quad_param_slots::ADAA_DRIVE] = params.adaa_drive.to_array();
    out[v3_quad_param_slots::LRU_BIAS_DRIVE] = params.lru_bias_drive.to_array();
    out[v3_quad_param_slots::WAVEFOLDER_MIX] = params.wavefolder_mix.to_array();
    out[v3_quad_param_slots::GLU_MIX] = params.glu_mix.to_array();
    out[v3_quad_param_slots::SVF_G] = g.to_array();
    out[v3_quad_param_slots::SVF_K_NL] = params.svf_k_nl.to_array();
    out
}

pub(crate) fn unpack_v3_core_quad_params(
    packed: &[[f32; 4]],
) -> Result<CoreQuadParamsStereo4<GpuVec4>, TrainError> {
    if packed.len() < V3_QUAD_PARAMS_VEC4_STRIDE {
        return Err(TrainError::InvalidDimensions {
            context: "unpack_v3_core_quad_params",
            expected: V3_QUAD_PARAMS_VEC4_STRIDE,
            actual: packed.len(),
        });
    }
    let svf = TptSvfCoeffs4 {
        a1: GpuVec4::from_array(packed[v3_quad_param_slots::SVF_A1]),
        a2: GpuVec4::from_array(packed[v3_quad_param_slots::SVF_A2]),
        a3: GpuVec4::from_array(packed[v3_quad_param_slots::SVF_A3]),
        k: GpuVec4::from_array(packed[v3_quad_param_slots::SVF_K]),
        two_k: GpuVec4::from_array(packed[v3_quad_param_slots::SVF_TWO_K]),
    };
    let tone = SonaToneWeights4 {
        lp: GpuVec4::from_array(packed[v3_quad_param_slots::TONE_LP]),
        bp: GpuVec4::from_array(packed[v3_quad_param_slots::TONE_BP]),
        hp: GpuVec4::from_array(packed[v3_quad_param_slots::TONE_HP]),
        ap: GpuVec4::from_array(packed[v3_quad_param_slots::TONE_AP]),
        dir: GpuVec4::from_array(packed[v3_quad_param_slots::TONE_DIR]),
    };
    let lru = SonaParaLruDuhemCoeffs4::from_clamped(
        GpuVec4::from_array(packed[v3_quad_param_slots::LRU_ALPHA_REL]),
        GpuVec4::from_array(packed[v3_quad_param_slots::LRU_ALPHA_ATK]),
        GpuVec4::from_array(packed[v3_quad_param_slots::LRU_KAPPA]),
        GpuVec4::from_array(packed[v3_quad_param_slots::LRU_GAMMA_DUHEM]),
        GpuVec4::from_array(packed[v3_quad_param_slots::LRU_SIGNED_MIX]),
        GpuVec4::from_array(packed[v3_quad_param_slots::LRU_STEREO_LINK]),
    );
    Ok(CoreQuadParamsStereo4 {
        svf,
        tone,
        lru,
        two_svf_g: GpuVec4::from_array(packed[v3_quad_param_slots::TWO_SVF_G]),
        svf_k_nl: GpuVec4::from_array(packed[v3_quad_param_slots::SVF_K_NL]),
        rect_sign: GpuVec4::from_array(packed[v3_quad_param_slots::RECT_SIGN]),
        sag_depth: GpuVec4::from_array(packed[v3_quad_param_slots::SAG_DEPTH]),
        adaa_drive: GpuVec4::from_array(packed[v3_quad_param_slots::ADAA_DRIVE]),
        lru_bias_drive: GpuVec4::from_array(packed[v3_quad_param_slots::LRU_BIAS_DRIVE]),
        wavefolder_mix: GpuVec4::from_array(packed[v3_quad_param_slots::WAVEFOLDER_MIX]),
        glu_mix: GpuVec4::from_array(packed[v3_quad_param_slots::GLU_MIX]),
    })
}

pub(crate) fn read_v3_checkpoint(
    ckpts: &[[f32; 4]],
    q_global: usize,
    num_tiles: usize,
    tile: usize,
) -> SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>> {
    let base = (q_global * num_tiles + tile) * V3_TILE_CKPT_VEC4_STRIDE;
    SonaBlockStateStereo8 {
        svf_s1: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base]),
            GpuVec4::from_array(ckpts[base + 1]),
        ),
        svf_s2: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 2]),
            GpuVec4::from_array(ckpts[base + 3]),
        ),
        lru_v: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 4]),
            GpuVec4::from_array(ckpts[base + 5]),
        ),
        adaa_prev: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 6]),
            GpuVec4::from_array(ckpts[base + 7]),
        ),
    }
}

pub(crate) fn write_v3_checkpoint(
    ckpts: &mut [[f32; 4]],
    q_global: usize,
    num_tiles: usize,
    tile: usize,
    state: &SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>>,
) {
    let base = (q_global * num_tiles + tile) * V3_TILE_CKPT_VEC4_STRIDE;
    let (s1_l, s1_r) = state.svf_s1.into_lr();
    let (s2_l, s2_r) = state.svf_s2.into_lr();
    let (v_l, v_r) = state.lru_v.into_lr();
    let (ap_l, ap_r) = state.adaa_prev.into_lr();
    ckpts[base] = canon_vec4(s1_l);
    ckpts[base + 1] = canon_vec4(s1_r);
    ckpts[base + 2] = canon_vec4(s2_l);
    ckpts[base + 3] = canon_vec4(s2_r);
    ckpts[base + 4] = canon_vec4(v_l);
    ckpts[base + 5] = canon_vec4(v_r);
    ckpts[base + 6] = canon_vec4(ap_l);
    ckpts[base + 7] = canon_vec4(ap_r);
}

pub(crate) fn write_v3_lane_param_grads(
    out_grads: &mut [[f32; 4]],
    pg_base: usize,
    c: usize,
    lp: &LaneParams,
    acc: &TileGradAccum,
    is_phs: bool,
) {
    for slot in 0..V3_QUAD_PARAMS_VEC4_STRIDE {
        out_grads[pg_base + slot][c] = acc.slots[slot];
    }
    let (g_svf_k, g_svf_k_nl, g_svf_g) = if is_phs {
        (
            if lp.k >= 0.0 {
                acc.slots[v3_quad_param_slots::SVF_K]
            } else {
                0.0
            },
            if lp.svf_k_nl >= 0.0 {
                acc.slots[v3_quad_param_slots::SVF_K_NL]
            } else {
                0.0
            },
            if lp.svf_g >= MIN_TPT_G {
                acc.slots[v3_quad_param_slots::SVF_G]
                    + 2.0 * acc.slots[v3_quad_param_slots::TWO_SVF_G]
            } else {
                0.0
            },
        )
    } else {
        let r0 = [acc.slots[0], acc.slots[1], acc.slots[2]];
        let g_k = acc.slots[3] + 2.0 * acc.slots[4]
            - lp.a2 * (r0[0] * lp.a1 + r0[1] * lp.a2 + r0[2] * lp.a3);
        let g_g = if lp.svf_g >= MIN_TPT_G {
            let denom_g_term = 2.0 * lp.a2 + lp.k * lp.a1;
            let da1_dg = -lp.a1 * denom_g_term;
            let da2_dg = lp.a1 - lp.a2 * denom_g_term;
            let da3_dg = (lp.a2 + lp.a2) - lp.a3 * denom_g_term;
            r0[0] * da1_dg
                + r0[1] * da2_dg
                + r0[2] * da3_dg
                + 2.0 * acc.slots[v3_quad_param_slots::TWO_SVF_G]
        } else {
            0.0
        };
        (g_k, 0.0, g_g)
    };
    out_grads[pg_base + v3_quad_param_slots::SVF_K][c] = g_svf_k;
    out_grads[pg_base + v3_quad_param_slots::SVF_G][c] = g_svf_g;
    out_grads[pg_base + v3_quad_param_slots::SVF_K_NL][c] = g_svf_k_nl;
}
