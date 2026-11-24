//! Shared Pass-2 `CoreQuad` + `InterBlockTile` gradient structures and tile/ZOH adjoint
//! helpers shared across the fast block-DAG path (`eval`) and frame-interleaved `causal_loop`
//! (`ref[train.pipeline.backbone-trajectory-opt]`).

use dnam_kernels::{MAX_SVF_K_NL, MIN_TPT_G};
use dnam_vm::{SONA_INTERBLOCK_TILES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER};

use crate::gpu::abi::{FEATURE_VEC4_STRIDE, TILE_WEIGHT_GRAD_FLOATS, quad_param_slots};

use super::super::super::model::cascade::sanitize_unit_f64;
use super::super::super::model::{CoreQuadF64, SonaModelF64};
use super::super::crossbar::CrossbarGrads;
use super::super::op_n_pre::OpNPreGrads;
use super::super::op_t_quad::OpTQuadAdjointGrads;
use super::super::sensory::SensoryQuadGrads;
use super::super::zoh_jacobian::apply_core_quad_zoh_jacobian;

/// Pre-sanitized incoming `InterBlockTile` `(src_q -> dst_q)` weight bundle.
#[derive(Clone, Debug)]
pub(crate) struct IncomingTileStep {
    pub src_q: usize,
    pub wo: [[f64; 4]; 4],
    pub we: [[f64; 4]; 4],
    pub wl: [[f64; 4]; 4],
}

/// Returns active `SONA_INTERBLOCK_TILES` count `(3 / 13 / 23 / 35)` for `q_count` `CoreQuad`s.
#[inline]
#[must_use]
pub(crate) const fn active_tiles_for_quads(q_count: usize) -> usize {
    match q_count {
        0..=4 => 3,
        5..=6 => 13,
        7..=8 => 23,
        _ => 35,
    }
}

/// Returns active `Op-T` `TransportQuad` count `(1 / 2 / 3 / 4)` for `q_count` `CoreQuad`s.
#[inline]
#[must_use]
pub(crate) const fn active_opt_quads_for_core_quads(q_count: usize) -> usize {
    match q_count {
        0..=4 => 1,
        5..=6 => 2,
        7..=8 => 3,
        _ => 4,
    }
}

/// All `21` continuous per-lane gradients for one [`CoreQuadF64`] (discrete `rect_sign` excluded,
/// `svf_k_nl` active on Block `0` and strictly `0.0` on Blocks `1..=3`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Pass2CoreQuadGrads {
    pub svf_g: [f64; 4],
    pub svf_k: [f64; 4],
    pub svf_k_nl: [f64; 4],
    pub tone_lp: [f64; 4],
    pub tone_bp: [f64; 4],
    pub tone_hp: [f64; 4],
    pub tone_ap: [f64; 4],
    pub tone_dir: [f64; 4],
    pub lru_alpha_rel: [f64; 4],
    pub lru_alpha_atk: [f64; 4],
    pub lru_kappa: [f64; 4],
    pub lru_duhem_gamma: [f64; 4],
    pub lru_signed_mix: [f64; 4],
    pub lru_stereo_link: [f64; 4],
    pub sag_depth: [f64; 4],
    pub adaa_drive: [f64; 4],
    pub lru_bias_drive: [f64; 4],
    pub wavefolder_mix: [f64; 4],
    pub glu_mix: [f64; 4],
    pub in_mid_weight: [f64; 4],
    pub in_side_weight: [f64; 4],
}

/// All `48` continuous coupling weights for one [`super::super::model::InterBlockTileF64`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Pass2TileGrads {
    pub w_odd_4x4: [[f64; 4]; 4],
    pub w_even_4x4: [[f64; 4]; 4],
    pub w_lru_4x4: [[f64; 4]; 4],
}

/// Complete full-subspace Pass-2 continuous + readout gradients matching [`SonaModelF64`].
#[derive(Clone, Debug, Default)]
pub(crate) struct Pass2FullGrads {
    pub core_quads: Vec<Pass2CoreQuadGrads>,
    pub tiles: Vec<Pass2TileGrads>,
    pub sensory: SensoryQuadGrads,
    pub crossbar: CrossbarGrads,
    pub op_n_pre: OpNPreGrads,
    pub d_ro_pole_48k: f64,
    /// Per-pair Kautz gradients `[d_r_low, d_theta, d_b_gamma, d_c_re_l, d_c_im_l, d_c_re_r, d_c_im_r]`.
    pub d_kautz: Vec<[f64; 7]>,
    pub op_t: [OpTQuadAdjointGrads; 4],
    pub d_tank_dry_mix: f64,
    pub d_w_ro: Vec<[f64; 12]>,
    pub d_direct_s2_u: [f64; 2],
    pub d_direct_s2_hp: [f64; 2],
    pub d_pre_conv_u: [f64; 2],
    pub d_pre_conv_hp: [f64; 2],
}

impl Pass2FullGrads {
    #[must_use]
    pub(crate) fn with_dims(q_count: usize, k_count: usize) -> Self {
        Self {
            core_quads: vec![Pass2CoreQuadGrads::default(); q_count],
            tiles: vec![Pass2TileGrads::default(); active_tiles_for_quads(q_count)],
            sensory: SensoryQuadGrads::default(),
            crossbar: CrossbarGrads::default(),
            op_n_pre: OpNPreGrads::default(),
            d_ro_pole_48k: 0.0,
            d_kautz: vec![[0.0_f64; 7]; k_count],
            op_t: [OpTQuadAdjointGrads::default(); 4],
            d_tank_dry_mix: 0.0,
            d_w_ro: vec![[0.0_f64; 12]; q_count * 4],
            d_direct_s2_u: [0.0; 2],
            d_direct_s2_hp: [0.0; 2],
            d_pre_conv_u: [0.0; 2],
            d_pre_conv_hp: [0.0; 2],
        }
    }
}

/// Accumulates all non-derived `QUAD_PARAMS_VEC4_STRIDE` slots for `tile` (`pg_base = tile * 25`)
/// across all `4` lanes of quad `q`, respecting clamp boundaries and Block 0 `svf_k_nl` gating.
pub(crate) fn add_tile_param_grads_to_quad(
    quad: &CoreQuadF64,
    block_idx: u32,
    tile_grads: &[[f32; 4]],
    pg_base: usize,
    dst: &mut Pass2CoreQuadGrads,
) {
    let slot_val = |slot: usize, c: usize| -> f64 { f64::from(tile_grads[pg_base + slot][c]) };
    let open_abs = |v: f64, limit: f64| -> bool { v.is_finite() && v.abs() < limit };
    let open_range = |v: f64, lo: f64, hi: f64| -> bool { v.is_finite() && v >= lo && v < hi };

    for c in 0..4 {
        if open_range(quad.svf_g[c], f64::from(MIN_TPT_G), 64.0) {
            dst.svf_g[c] += slot_val(quad_param_slots::SVF_G, c);
        }
        if open_range(quad.svf_k[c], 0.0, 8.0) {
            dst.svf_k[c] += slot_val(quad_param_slots::SVF_K, c);
        }
        if block_idx == 0 && open_range(quad.svf_k_nl[c], 0.0, f64::from(MAX_SVF_K_NL)) {
            dst.svf_k_nl[c] += slot_val(quad_param_slots::SVF_K_NL, c);
        }
        if open_abs(quad.tone_lp[c], 16.0) {
            dst.tone_lp[c] += slot_val(quad_param_slots::TONE_LP, c);
        }
        if open_abs(quad.tone_bp[c], 16.0) {
            dst.tone_bp[c] += slot_val(quad_param_slots::TONE_BP, c);
        }
        if open_abs(quad.tone_hp[c], 16.0) {
            dst.tone_hp[c] += slot_val(quad_param_slots::TONE_HP, c);
        }
        if open_abs(quad.tone_ap[c], 16.0) {
            dst.tone_ap[c] += slot_val(quad_param_slots::TONE_AP, c);
        }
        if open_abs(quad.tone_dir[c], 16.0) {
            dst.tone_dir[c] += slot_val(quad_param_slots::TONE_DIR, c);
        }
        dst.lru_alpha_rel[c] += slot_val(quad_param_slots::LRU_ALPHA_REL, c);
        dst.lru_alpha_atk[c] += slot_val(quad_param_slots::LRU_ALPHA_ATK, c);
        dst.lru_kappa[c] += slot_val(quad_param_slots::LRU_KAPPA, c);
        dst.lru_duhem_gamma[c] += slot_val(quad_param_slots::LRU_GAMMA_DUHEM, c);
        dst.lru_signed_mix[c] += slot_val(quad_param_slots::LRU_SIGNED_MIX, c);
        dst.lru_stereo_link[c] += slot_val(quad_param_slots::LRU_STEREO_LINK, c);
        if open_abs(quad.sag_depth[c], 8.0) {
            dst.sag_depth[c] += slot_val(quad_param_slots::SAG_DEPTH, c);
        }
        if open_abs(quad.adaa_drive[c], 32.0) {
            dst.adaa_drive[c] += slot_val(quad_param_slots::ADAA_DRIVE, c);
        }
        if open_abs(quad.lru_bias_drive[c], 16.0) {
            dst.lru_bias_drive[c] += slot_val(quad_param_slots::LRU_BIAS_DRIVE, c);
        }
        dst.wavefolder_mix[c] += slot_val(quad_param_slots::WAVEFOLDER_MIX, c);
        if open_abs(quad.glu_mix[c], 4.0) {
            dst.glu_mix[c] += slot_val(quad_param_slots::GLU_MIX, c);
        }
    }
}

/// Applies ZOH chain-rule rescaling (`rho_core`) to `(svf_g, lru_alpha_rel, lru_alpha_atk,
/// lru_kappa)` of `dst` matching `CoreQuadF64::rescaled_for_rho`.
pub(crate) fn finalize_core_quad_zoh(
    quad: &CoreQuadF64,
    rho_core: f64,
    dst: &mut Pass2CoreQuadGrads,
) {
    apply_core_quad_zoh_jacobian(
        quad,
        rho_core,
        &mut dst.svf_g,
        &mut dst.lru_alpha_rel,
        &mut dst.lru_alpha_atk,
        &mut dst.lru_kappa,
    );
}

/// Collects all non-zero incoming causal `InterBlockTile`s `(src_q -> dst_q)` for `q`, verifying
/// `SONA_ULTRA_QUAD_TRAVERSAL_ORDER` topological precedence in debug builds.
pub(crate) fn collect_incoming_tiles(
    model: &SonaModelF64,
    dst_q: usize,
    q_count: usize,
    active_tiles: usize,
    quad_done: &[bool],
) -> Vec<IncomingTileStep> {
    let mut out = Vec::new();
    for (tile_idx, &(src_q, target_q)) in SONA_INTERBLOCK_TILES[..active_tiles].iter().enumerate() {
        if target_q != dst_q || src_q >= q_count {
            continue;
        }
        debug_assert!(
            quad_done[src_q],
            "topological violation: src_q={src_q} -> dst_q={dst_q}"
        );
        let tile = &model.tiles[tile_idx];
        let mut wo = [[0.0_f64; 4]; 4];
        let mut we = [[0.0_f64; 4]; 4];
        let mut wl = [[0.0_f64; 4]; 4];
        let mut any_nonzero = false;
        for s in 0..4 {
            for d in 0..4 {
                let o = sanitize_unit_f64(tile.w_odd_4x4[s][d], -8.0, 8.0, 0.0);
                let e = sanitize_unit_f64(tile.w_even_4x4[s][d], -8.0, 8.0, 0.0);
                let l = sanitize_unit_f64(tile.w_lru_4x4[s][d], -8.0, 8.0, 0.0);
                wo[s][d] = o;
                we[s][d] = e;
                wl[s][d] = l;
                if o != 0.0 || e != 0.0 || l != 0.0 {
                    any_nonzero = true;
                }
            }
        }
        if any_nonzero {
            out.push(IncomingTileStep { src_q, wo, we, wl });
        }
    }
    let _ = SONA_ULTRA_QUAD_TRAVERSAL_ORDER;
    out
}

/// Mutable `(d_cb_l, d_cb_r)` crossbar injection adjoint buffers indexed as `[q * n + t][lane]`.
pub(crate) type CrossbarGradMut<'a> = (&'a mut [[f64; 4]], &'a mut [[f64; 4]]);

/// Immutable frame context passed into [`backprop_core_input_frame`].
#[derive(Clone, Copy)]
pub(crate) struct CoreInputBackpropCtx<'a> {
    pub model: &'a SonaModelF64,
    pub q: usize,
    pub t: usize,
    pub n: usize,
    pub q_count: usize,
    pub active_tiles: usize,
    pub ux_l_t: f64,
    pub ux_r_t: f64,
    pub feature_outputs: &'a [[f32; 4]],
}

/// Mutable gradient accumulators populated by [`backprop_core_input_frame`].
pub(crate) struct CoreInputBackpropOut<'a> {
    pub d_feats: &'a mut [[f32; 4]],
    pub d_quad: &'a mut Pass2CoreQuadGrads,
    pub d_tiles: &'a mut [Pass2TileGrads],
    pub d_cb_opt: Option<CrossbarGradMut<'a>>,
}

/// Accumulates deterministic wave-reduced `InterBlockTile` weight gradients (`48` floats per tile:
/// `[w_odd_4x4, w_even_4x4, w_lru_4x4]`) into `d_tiles`, honoring the `[-8.0, 8.0]` clamp boundaries.
pub(crate) fn add_wave_tile_weight_grads(
    model: &SonaModelF64,
    q_count: usize,
    active_tiles: usize,
    flat_grads: &[f32],
    d_tiles: &mut [Pass2TileGrads],
) {
    for (tile_idx, &(src_q, dst_q)) in SONA_INTERBLOCK_TILES[..active_tiles].iter().enumerate() {
        if src_q >= q_count || dst_q >= q_count || tile_idx >= d_tiles.len() {
            continue;
        }
        let base = tile_idx * TILE_WEIGHT_GRAD_FLOATS;
        if base + TILE_WEIGHT_GRAD_FLOATS > flat_grads.len() {
            break;
        }
        let tile = &model.tiles[tile_idx];
        let dt = &mut d_tiles[tile_idx];
        for s in 0..4 {
            for d in 0..4 {
                let idx = s * 4 + d;
                if tile.w_odd_4x4[s][d].is_finite() && tile.w_odd_4x4[s][d].abs() < 8.0 {
                    dt.w_odd_4x4[s][d] += f64::from(flat_grads[base + idx]);
                }
                if tile.w_even_4x4[s][d].is_finite() && tile.w_even_4x4[s][d].abs() < 8.0 {
                    dt.w_even_4x4[s][d] += f64::from(flat_grads[base + 16 + idx]);
                }
                if tile.w_lru_4x4[s][d].is_finite() && tile.w_lru_4x4[s][d].abs() < 8.0 {
                    dt.w_lru_4x4[s][d] += f64::from(flat_grads[base + 32 + idx]);
                }
            }
        }
    }
}

/// Propagates single-frame input adjoints `(gin_l, gin_r)` of quad `q` back into M/S input weights
/// & `(gx_ux_l, gx_ux_r)` and optional `Crossbar` `d_cb_l/r`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backprop_local_core_input_frame(
    model: &SonaModelF64,
    q: usize,
    t: usize,
    n: usize,
    ux_l_t: f64,
    ux_r_t: f64,
    gin_l: [f32; 4],
    gin_r: [f32; 4],
    d_quad: &mut Pass2CoreQuadGrads,
    mut d_cb_opt: Option<CrossbarGradMut<'_>>,
) -> (f64, f64) {
    let mid = 0.5 * sanitize_unit_f64(ux_l_t + ux_r_t, -2.0e6, 2.0e6, 0.0);
    let side = 0.5 * sanitize_unit_f64(ux_l_t - ux_r_t, -2.0e6, 2.0e6, 0.0);
    let mut g_ux_l = 0.0_f64;
    let mut g_ux_r = 0.0_f64;
    let quad = &model.core_quads[q];

    for d in 0..4 {
        let gl_d = f64::from(gin_l[d]);
        let gr_d = f64::from(gin_r[d]);
        let wm = sanitize_unit_f64(quad.in_mid_weight[d], -16.0, 16.0, 0.0);
        let ws = sanitize_unit_f64(quad.in_side_weight[d], -16.0, 16.0, 0.0);
        if quad.in_mid_weight[d].is_finite() && quad.in_mid_weight[d].abs() < 16.0 {
            d_quad.in_mid_weight[d] += (gl_d + gr_d) * mid;
        }
        if quad.in_side_weight[d].is_finite() && quad.in_side_weight[d].abs() < 16.0 {
            d_quad.in_side_weight[d] += (gl_d - gr_d) * side;
        }
        let d_mid = wm * (gl_d + gr_d);
        let d_side = ws * (gl_d - gr_d);
        g_ux_l += 0.5 * (d_mid + d_side);
        g_ux_r += 0.5 * (d_mid - d_side);

        if let Some((ref mut d_cb_l, ref mut d_cb_r)) = d_cb_opt {
            let cb_idx = q * n + t;
            d_cb_l[cb_idx][d] += gl_d;
            d_cb_r[cb_idx][d] += gr_d;
        }
    }

    (g_ux_l, g_ux_r)
}

/// Propagates single-frame input adjoints `(gin_l, gin_r)` of quad `q` back into M/S input weights
/// & `(gx_ux_l, gx_ux_r)`, optional `Crossbar` `d_cb_l/r`, and upstream `InterBlockTile`s / `d_feats`.
pub(crate) fn backprop_core_input_frame(
    ctx: &CoreInputBackpropCtx<'_>,
    gin_l: [f32; 4],
    gin_r: [f32; 4],
    out: CoreInputBackpropOut<'_>,
) -> (f64, f64) {
    let CoreInputBackpropCtx {
        model,
        q,
        t,
        n,
        q_count,
        active_tiles,
        ux_l_t,
        ux_r_t,
        feature_outputs,
    } = *ctx;
    let CoreInputBackpropOut {
        d_feats,
        d_quad,
        d_tiles,
        d_cb_opt,
    } = out;
    let (g_ux_l, g_ux_r) = backprop_local_core_input_frame(
        model, q, t, n, ux_l_t, ux_r_t, gin_l, gin_r, d_quad, d_cb_opt,
    );

    for (tile_idx, &(src_q, dst_q)) in SONA_INTERBLOCK_TILES[..active_tiles].iter().enumerate() {
        if dst_q != q || src_q >= q_count {
            continue;
        }
        let tile = &model.tiles[tile_idx];
        let dt = &mut d_tiles[tile_idx];
        let sbase = (src_q * n + t) * FEATURE_VEC4_STRIDE;
        let v_eff_l = feature_outputs[sbase];
        let v_eff_r = feature_outputs[sbase + 1];
        let odd_l = feature_outputs[sbase + 2];
        let odd_r = feature_outputs[sbase + 3];
        let even_l = feature_outputs[sbase + 4];
        let even_r = feature_outputs[sbase + 5];
        for s in 0..4 {
            let ol = f64::from(odd_l[s]);
            let or_ = f64::from(odd_r[s]);
            let el = f64::from(even_l[s]);
            let er = f64::from(even_r[s]);
            let vl = f64::from(v_eff_l[s]);
            let vr = f64::from(v_eff_r[s]);
            for d in 0..4 {
                let gl_d = f64::from(gin_l[d]);
                let gr_d = f64::from(gin_r[d]);
                if tile.w_odd_4x4[s][d].is_finite() && tile.w_odd_4x4[s][d].abs() < 8.0 {
                    dt.w_odd_4x4[s][d] += gl_d * ol + gr_d * or_;
                }
                if tile.w_even_4x4[s][d].is_finite() && tile.w_even_4x4[s][d].abs() < 8.0 {
                    dt.w_even_4x4[s][d] += gl_d * el + gr_d * er;
                }
                if tile.w_lru_4x4[s][d].is_finite() && tile.w_lru_4x4[s][d].abs() < 8.0 {
                    dt.w_lru_4x4[s][d] += gl_d * vl + gr_d * vr;
                }
                let wo = sanitize_unit_f64(tile.w_odd_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                let we = sanitize_unit_f64(tile.w_even_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                let wl = sanitize_unit_f64(tile.w_lru_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                d_feats[sbase][s] += wl * (gl_d as f32);
                d_feats[sbase + 1][s] += wl * (gr_d as f32);
                d_feats[sbase + 2][s] += wo * (gl_d as f32);
                d_feats[sbase + 3][s] += wo * (gr_d as f32);
                d_feats[sbase + 4][s] += we * (gl_d as f32);
                d_feats[sbase + 5][s] += we * (gr_d as f32);
            }
        }
    }

    (g_ux_l, g_ux_r)
}
