#![forbid(unsafe_code)]
//! Compact Forward/Adjoint Tapes and Clamped Engine Parameters for Stage I–II `v3` BPTT.

use std::f64::consts::PI;

use dnam_engine::stages::stage1_svf::SVF_W_MAX;
use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_KAUTZ_POLES, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
};

use super::super::model_f64::V3MatryoshkaOptF64;
use super::tanh_pade5_with_deriv;

/// Maximum active GRU lanes across all 3 Matryoshka blocks (`12 + 8 + 4 = 24`).
pub const MAX_ACTIVE_CASCADE_LANES: usize = 24;

/// Tight per-block offsets into [`GruCascadeSubTape`] (`[0, 12, 20]`).
pub const BLK_TAPE_OFF: [usize; NUM_GRU_BLOCKS] = [0, 12, 20];

/// Forward/reverse per-substep tape for Stage I 2-section Cytomic TPT SVF.
#[derive(Clone, Copy, Debug, Default)]
pub struct SvfChanTape {
    pub taps: [f64; 7],
    pub det_lp_prev: f64,
    pub det_lp: f64,
    pub v_det_hp: f64,
    pub det_raw_sign: f64,
    pub env_prev: f64,
    pub atk_taken: bool,
    pub rsq3: f64,
    pub e: f64,
    pub w1_unclamped: f64,
    pub oct_p: f64,
    pub g1: f64,
    pub g2: f64,
    pub s1_c1: f64,
    pub s1_a1: f64,
    pub s1_a2: f64,
    pub s1_a3: f64,
    pub s1_v1: f64,
    pub s1_v3: f64,
    pub s2_c1: f64,
    pub s2_a1: f64,
    pub s2_a2: f64,
    pub s2_a3: f64,
    pub s2_v1: f64,
    pub s2_v3: f64,
}

/// Forward/reverse substep tape for multi-timescale Stage II input features.
#[derive(Clone, Copy, Debug, Default)]
pub struct InputFeatSubTape {
    pub feats: [f64; NUM_IN_FEATS_B0],
    pub lp2: f64,
    pub raw_abs_u: f64,
    pub rect: f64,
    pub r_mix_above_thresh: bool,
    pub u_lp_prev: f64,
    pub env_fast_prev: f64,
    pub env_slow_prev: f64,
    pub sign_u: f64,
    pub fast_atk_taken: bool,
    pub slow_atk_taken: bool,
    pub pinch_inv: f64,
    pub rect_slow: f64,
    pub slow_rsq3: f64,
    pub up_d2: f64,
    pub up_zoh_diff: f64,
    pub up_dxp_diff: f64,
    pub up_hb_diff: [f64; 2],
}

/// Compact forward/reverse substep tape for the 3-block Stage II Matryoshka GRU cascade (`24` active lanes).
#[derive(Clone, Copy, Debug)]
pub struct GruCascadeSubTape {
    pub z: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub dz_daz: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub r: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub dr_dar: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub a_sig: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub delta_tanh: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub dtanh_an: [f64; MAX_ACTIVE_CASCADE_LANES],
    pub a_sig_mask_bits: [u16; NUM_GRU_BLOCKS],
    pub clamp_mask_bits: [u16; NUM_GRU_BLOCKS],
}

impl Default for GruCascadeSubTape {
    fn default() -> Self {
        Self {
            z: [0.0; MAX_ACTIVE_CASCADE_LANES],
            dz_daz: [0.0; MAX_ACTIVE_CASCADE_LANES],
            r: [0.0; MAX_ACTIVE_CASCADE_LANES],
            dr_dar: [0.0; MAX_ACTIVE_CASCADE_LANES],
            a_sig: [0.0; MAX_ACTIVE_CASCADE_LANES],
            delta_tanh: [0.0; MAX_ACTIVE_CASCADE_LANES],
            dtanh_an: [0.0; MAX_ACTIVE_CASCADE_LANES],
            a_sig_mask_bits: [u16::MAX; NUM_GRU_BLOCKS],
            clamp_mask_bits: [u16::MAX; NUM_GRU_BLOCKS],
        }
    }
}

#[inline(always)]
fn cl_mask(v: f64, lo: f64, hi: f64) -> (f64, f64) {
    if v <= lo {
        (lo, 0.0)
    } else if v >= hi {
        (hi, 0.0)
    } else {
        (v, 1.0)
    }
}

/// Clamped parameter values paired with straight-through boundary masks `(val, mask)`.
pub(crate) struct ClampedCoreParams {
    pub up_quad: (f64, f64),
    pub dec_c_prev1: (f64, f64),
    pub dec_c0: (f64, f64),
    pub up_pre_emph: (f64, f64),
    pub dec_aa_mix: (f64, f64),
    pub up_aa_mix: (f64, f64),
    pub up_zoh_mix: (f64, f64),
    pub base_g: (f64, f64),
    pub sec2_r: (f64, f64),
    pub env_oct: (f64, f64),
    pub env_comp: (f64, f64),
    pub atk: (f64, f64),
    pub rel: (f64, f64),
    pub det_lp: (f64, f64),
    pub det_hp_mix: (f64, f64),
    pub dk1: (f64, f64),
    pub dk2: (f64, f64),
    pub s2_bp: (f64, f64),
    pub s2_hp: (f64, f64),
    pub link: (f64, f64),
    pub g_min: (f64, f64),
    pub g_max: (f64, f64),
    pub feat_lp2_mix: (f64, f64),
    pub hp_beta: (f64, f64),
    pub fast_atk: (f64, f64),
    pub fast_rel: (f64, f64),
    pub pinch_k: (f64, f64),
    pub slow_atk: (f64, f64),
    pub slow_rel: (f64, f64),
    pub slow_det_comp: (f64, f64),
    pub key_link: (f64, f64),
    pub det_thresh: (f64, f64),
    pub b_z: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub b_r: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub b_n: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub k_sag: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub k_quad: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub tanh_bn: [[f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub dtanh_bn: [[f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub kautz_pre: [(f64, f64, f64, f64, f64, f64); MAX_KAUTZ_POLES],
}

impl ClampedCoreParams {
    pub(crate) fn from_model(m: &V3MatryoshkaOptF64) -> Self {
        let w_max = f64::from(SVF_W_MAX);
        let g_min = cl_mask(m.svf_g_min, 1.0e-4, w_max);
        let g_max = if m.svf_g_max < g_min.0 {
            (g_min.0, 0.0)
        } else {
            cl_mask(m.svf_g_max, 1.0e-4, w_max)
        };
        let mut b_z = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut b_r = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut b_n = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut k_sag = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut k_quad = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut tanh_bn = [[0.0; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut dtanh_bn = [[0.0; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        for b in 0..NUM_GRU_BLOCKS {
            for l in 0..MAX_BLOCK_LANES {
                b_z[b][l] = cl_mask(m.blocks[b].b_z[l], -8.0, 8.2);
                b_r[b][l] = cl_mask(m.blocks[b].b_r[l], -8.0, 8.2);
                b_n[b][l] = cl_mask(m.blocks[b].b_n[l], -3.8, 3.8);
                k_sag[b][l] = cl_mask(m.blocks[b].k_sag[l], 0.0, 8.0);
                k_quad[b][l] = cl_mask(m.blocks[b].k_quad[l], -0.45, 0.45);
                (tanh_bn[b][l], dtanh_bn[b][l]) = tanh_pade5_with_deriv(b_n[b][l].0);
            }
        }
        let kautz_pre = m.kautz_poles.map(|(rr, tt)| {
            let (r, r_m) = cl_mask(rr, 0.25, 0.9985);
            let (th, t_m) = cl_mask(tt, 0.0, PI - 1.0e-3);
            let bg = (1.0 - r * r).max(1.0e-7).sqrt();
            let (st, ct) = th.sin_cos();
            (r, bg, ct, st, r_m, t_m)
        });
        Self {
            up_quad: cl_mask(m.os_up_quad, -0.20, 0.05),
            dec_c_prev1: cl_mask(m.os_dec_c_prev1, 0.0, 0.28),
            dec_c0: cl_mask(m.os_dec_c0, 0.0, 0.55),
            up_pre_emph: cl_mask(m.os_up_pre_emph, -0.28, 1.0),
            dec_aa_mix: cl_mask(m.os_dec_aa_mix, 0.0, 1.0),
            up_aa_mix: cl_mask(m.os_up_aa_mix, 0.0, 1.0),
            up_zoh_mix: cl_mask(m.os_up_zoh_mix, 0.0, 1.0),
            base_g: cl_mask(m.svf_base_g, 1.0e-4, w_max),
            sec2_r: cl_mask(m.svf_sec2_g_ratio, 0.25, 4.0),
            env_oct: cl_mask(m.svf_env_oct_sens, -24.0, 24.0),
            env_comp: cl_mask(m.svf_env_comp, 0.0, 16.0),
            atk: cl_mask(m.svf_atk_alpha, 0.001, 0.9999),
            rel: cl_mask(m.svf_rel_alpha, 0.001, 0.99995),
            det_lp: cl_mask(m.svf_det_lp_pole, 0.05, 0.995),
            det_hp_mix: cl_mask(m.svf_det_hp_mix, -1.0, 2.0),
            dk1: cl_mask(m.svf_damp_k1, 0.05, 4.0),
            dk2: cl_mask(m.svf_damp_k2, 0.05, 4.0),
            s2_bp: cl_mask(m.svf_sec2_bp_feed, -2.0, 2.0),
            s2_hp: cl_mask(m.svf_sec2_hp_feed, -2.0, 2.0),
            link: cl_mask(m.svf_stereo_link, 0.0, 1.0),
            g_min,
            g_max,
            feat_lp2_mix: cl_mask(m.svf_feat_lp2_mix, 0.0, 1.0),
            hp_beta: cl_mask(m.feat_in_hp_beta, 1.0e-4, 0.50),
            fast_atk: cl_mask(m.feat_fast_atk_beta, 1.0e-4, 0.999),
            fast_rel: cl_mask(m.feat_fast_rel_beta, 1.0e-5, 0.999),
            pinch_k: cl_mask(m.feat_pinch_k, 0.0, 25_000.0),
            slow_atk: cl_mask(m.feat_slow_atk_beta, 1.0e-5, 0.50),
            slow_rel: cl_mask(m.feat_slow_rel_beta, 1.0e-6, 0.10),
            slow_det_comp: cl_mask(m.feat_slow_det_comp, 0.0, 64.0),
            key_link: cl_mask(m.feat_key_link, 0.0, 1.0),
            det_thresh: cl_mask(m.feat_det_thresh, 0.0, 0.50),
            b_z,
            b_r,
            b_n,
            k_sag,
            k_quad,
            tanh_bn,
            dtanh_bn,
            kautz_pre,
        }
    }
}
