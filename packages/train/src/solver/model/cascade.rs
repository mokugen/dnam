//! Causal `SONA_ULTRA_QUAD_TRAVERSAL_ORDER` stereo core cascade with `InterBlockTile` coupling,
//! ZOH rate rescaling, and `2×` poly-quad oversampling/decimation matching `dnam-vm`
//! (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.zoh.core-paralru]`).

use dnam_vm::{
    SONA_CORE_QUAD_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER,
};

use crate::error::TrainError;
use crate::gpu::abi::{FEATURE_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES};
use crate::runner::{TrainRunner, pack_core_quad_params};

use super::{SonaModelF64, run_planned_core_scan};

/// Output returned by [`run_causal_core_cascade_stereo`].
#[derive(Clone, Debug)]
pub(crate) struct CausalCascadeOutput {
    pub core_inputs: Vec<[f32; 4]>,
    pub tile_checkpoints: Vec<[f32; 4]>,
    pub feature_outputs: Vec<[f32; 4]>,
}

#[inline]
pub(crate) fn sanitize_unit_f64(v: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

/// Upsamples `1x` stereo `(x_l, x_r)` into `2x` length using the exact causal polynomial
/// interpolation law of `dnam_vm::graph::oversample::SonaStereoOversampler2x` (`aa_mix = 0`).
#[must_use]
pub(crate) fn upsample_stereo_2x_f64(
    x_l: &[f64],
    x_r: &[f64],
    up_quad_coeff: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = x_l.len().min(x_r.len());
    let coeff = sanitize_unit_f64(up_quad_coeff, -0.20, 0.05, -0.125);
    let mut up_l = Vec::with_capacity(n * 2);
    let mut up_r = Vec::with_capacity(n * 2);
    let (mut prev1_l, mut prev2_l) = (0.0_f64, 0.0_f64);
    let (mut prev1_r, mut prev2_r) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let xl = sanitize_unit_f64(x_l[i], -1.0e6, 1.0e6, 0.0);
        let xr = sanitize_unit_f64(x_r[i], -1.0e6, 1.0e6, 0.0);
        let d2_l = xl - 2.0 * prev1_l + prev2_l;
        let d2_r = xr - 2.0 * prev1_r + prev2_r;
        let sub0_l = 0.5 * xl + 0.5 * prev1_l + coeff * d2_l;
        let sub0_r = 0.5 * xr + 0.5 * prev1_r + coeff * d2_r;
        prev2_l = prev1_l;
        prev1_l = xl;
        prev2_r = prev1_r;
        prev1_r = xr;
        up_l.push(sub0_l);
        up_l.push(xl);
        up_r.push(sub0_r);
        up_r.push(xr);
    }
    (up_l, up_r)
}

/// Decimates one `2x` stream (`2 * n` samples) back to `1x` (`n` samples) using the causal
/// 3-tap decimator of `dnam_vm::graph::oversample::SonaStereoOversampler2x` (`dec_aa_mix = 0`).
#[must_use]
pub(crate) fn downsample_lane_2x_f64(sig_2x: &[f64], dec_c_prev1: f64, dec_c0: f64) -> Vec<f64> {
    let n = sig_2x.len() / 2;
    let c_prev1 = sanitize_unit_f64(dec_c_prev1, 0.0, 0.28, 0.15);
    let c0 = sanitize_unit_f64(dec_c0, 0.0, 0.55, 0.50);
    let c1 = 1.0 - c_prev1 - c0;
    let mut out = Vec::with_capacity(n);
    let mut prev1 = 0.0_f64;
    for i in 0..n {
        let sub0 = sig_2x[2 * i];
        let sub1 = sig_2x[2 * i + 1];
        out.push(c_prev1 * prev1 + c0 * sub0 + c1 * sub1);
        prev1 = sub1;
    }
    out
}

/// Evaluates active `CoreQuad` blocks in canonical `SONA_ULTRA_QUAD_TRAVERSAL_ORDER` with
/// M/S input coupling, causal `SONA_INTERBLOCK_TILES`, and ZOH rescaling at `rho_core`.
#[cfg(test)]
pub(crate) fn run_causal_core_cascade_stereo(
    runner: &TrainRunner,
    model: &SonaModelF64,
    ux_l: &[f64],
    ux_r: &[f64],
    active_quads: usize,
    rho_core: f64,
    write_checkpoints: bool,
) -> Result<CausalCascadeOutput, TrainError> {
    run_causal_core_cascade_with_crossbar(
        runner,
        model,
        ux_l,
        ux_r,
        CascadeCrossbarSpec {
            active_quads,
            rho_core,
            write_checkpoints,
            crossbar_add: None,
        },
    )
}

/// Borrowed per-frame `(cb_add_l, cb_add_r)` crossbar injections indexed as `[q * n + t][lane]`.
pub(crate) type CrossbarInjection<'a> = (&'a [[f64; 4]], &'a [[f64; 4]]);

/// Execution configuration passed to [`run_causal_core_cascade_with_crossbar`].
#[derive(Clone, Copy)]
pub(crate) struct CascadeCrossbarSpec<'a> {
    pub active_quads: usize,
    pub rho_core: f64,
    pub write_checkpoints: bool,
    pub crossbar_add: Option<CrossbarInjection<'a>>,
}

/// Evaluates active `CoreQuad` blocks with optional precomputed per-quad `Crossbar` injections
/// `(cb_add_l, cb_add_r)` indexed as `[q * n + t][lane]`.
pub(crate) fn run_causal_core_cascade_with_crossbar(
    runner: &TrainRunner,
    model: &SonaModelF64,
    ux_l: &[f64],
    ux_r: &[f64],
    spec: CascadeCrossbarSpec<'_>,
) -> Result<CausalCascadeOutput, TrainError> {
    let CascadeCrossbarSpec {
        active_quads,
        rho_core,
        write_checkpoints,
        crossbar_add,
    } = spec;
    let n = ux_l.len().min(ux_r.len());
    let q_count = active_quads.clamp(1, super::NUM_CORE_QUADS_ULTRA);
    let active_tiles = match q_count {
        0..=4 => 3_usize,
        5..=6 => 13_usize,
        7..=8 => 23_usize,
        _ => 35_usize,
    };
    let num_tiles = n.div_ceil(TILE_FRAMES);

    let mut core_inputs = vec![[0.0_f32; 4]; q_count * n * 2];
    let mut feature_outputs = vec![[0.0_f32; 4]; q_count * n * FEATURE_VEC4_STRIDE];
    let ckpt_per_quad = if write_checkpoints {
        num_tiles * TILE_CKPT_VEC4_STRIDE
    } else {
        0
    };
    let mut tile_checkpoints = vec![[0.0_f32; 4]; q_count * ckpt_per_quad];
    let mut quad_done = [false; super::NUM_CORE_QUADS_ULTRA];

    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER {
        if q >= q_count {
            continue;
        }
        let quad = model.core_quads[q].rescaled_for_rho(rho_core);
        let in_mid: [f64; 4] =
            core::array::from_fn(|d| sanitize_unit_f64(quad.in_mid_weight[d], -16.0, 16.0, 0.0));
        let in_side: [f64; 4] =
            core::array::from_fn(|d| sanitize_unit_f64(quad.in_side_weight[d], -16.0, 16.0, 0.0));
        type IncomingTileStep = (usize, [[f64; 4]; 4], [[f64; 4]; 4], [[f64; 4]; 4]);
        let incoming_tiles: Vec<IncomingTileStep> = SONA_INTERBLOCK_TILES[..active_tiles]
            .iter()
            .enumerate()
            .filter_map(|(tile_idx, &(src_q, dst_q))| {
                if dst_q != q || src_q >= q_count {
                    return None;
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
                any_nonzero.then_some((src_q, wo, we, wl))
            })
            .collect();

        let mut quad_in = vec![[0.0_f32; 4]; n * 2];
        for t in 0..n {
            let ul = sanitize_unit_f64(ux_l[t], -1.0e6, 1.0e6, 0.0);
            let ur = sanitize_unit_f64(ux_r[t], -1.0e6, 1.0e6, 0.0);
            let mid = 0.5 * (ul + ur);
            let side = 0.5 * (ul - ur);
            let cb_idx = q * n + t;
            let (cb_l, cb_r) = match crossbar_add {
                Some((cl, cr)) if cb_idx < cl.len() && cb_idx < cr.len() => {
                    (cl[cb_idx], cr[cb_idx])
                }
                _ => ([0.0; 4], [0.0; 4]),
            };
            let mut acc_l = [0.0_f64; 4];
            let mut acc_r = [0.0_f64; 4];
            for d in 0..4 {
                acc_l[d] = in_mid[d] * mid + in_side[d] * side + cb_l[d];
                acc_r[d] = in_mid[d] * mid - in_side[d] * side + cb_r[d];
            }
            for &(src_q, ref wo, ref we, ref wl) in &incoming_tiles {
                let fbase = (src_q * n + t) * 6;
                let v_eff_l = feature_outputs[fbase];
                let v_eff_r = feature_outputs[fbase + 1];
                let odd_l = feature_outputs[fbase + 2];
                let odd_r = feature_outputs[fbase + 3];
                let even_l = feature_outputs[fbase + 4];
                let even_r = feature_outputs[fbase + 5];
                for s in 0..4 {
                    let ol = f64::from(odd_l[s]);
                    let or_ = f64::from(odd_r[s]);
                    let el = f64::from(even_l[s]);
                    let er = f64::from(even_r[s]);
                    let vl = f64::from(v_eff_l[s]);
                    let vr = f64::from(v_eff_r[s]);
                    for d in 0..4 {
                        acc_l[d] += wo[s][d] * ol + we[s][d] * el + wl[s][d] * vl;
                        acc_r[d] += wo[s][d] * or_ + we[s][d] * er + wl[s][d] * vr;
                    }
                }
            }
            quad_in[t * 2] = acc_l.map(|v| v as f32);
            quad_in[t * 2 + 1] = acc_r.map(|v| v as f32);
        }

        let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
        let single_params = pack_core_quad_params(&quad.to_kernel_params_for_block(block_idx));
        let scan = run_planned_core_scan(
            runner,
            n,
            1,
            1,
            write_checkpoints,
            block_idx,
            &quad_in,
            &single_params,
        )?;
        core_inputs[(q * n * 2)..((q + 1) * n * 2)].copy_from_slice(&quad_in);
        feature_outputs[(q * n * 6)..((q + 1) * n * 6)]
            .copy_from_slice(&scan.feature_outputs[..(n * 6)]);
        if write_checkpoints && !scan.tile_checkpoints.is_empty() {
            tile_checkpoints[(q * ckpt_per_quad)..((q + 1) * ckpt_per_quad)]
                .copy_from_slice(&scan.tile_checkpoints[..ckpt_per_quad]);
        }
        quad_done[q] = true;
    }

    Ok(CausalCascadeOutput {
        core_inputs,
        tile_checkpoints,
        feature_outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsample_downsample_and_causal_cascade_topology() {
        let runner = TrainRunner::cpu();
        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].adaa_drive = [1.2; 4];
        model.tiles[0].w_odd_4x4[0][0] = 0.5;
        let x_l: Vec<f64> = (0..64).map(|t| 0.3 * ((t as f64) * 0.1).sin()).collect();
        let x_r: Vec<f64> = (0..64).map(|t| -0.2 * ((t as f64) * 0.17).cos()).collect();
        let out =
            run_causal_core_cascade_stereo(&runner, &model, &x_l, &x_r, 4, 1.0, false).unwrap();
        assert_eq!(out.feature_outputs.len(), 4 * 64 * 6);
        // Tile 0 connects quad 0 -> quad 1: quad 1 input must incorporate quad 0 odd output.
        assert!((out.core_inputs[64 * 2 + 20][0] - out.core_inputs[20][0]).abs() > 1.0e-5);

        let (up_l, up_r) = upsample_stereo_2x_f64(&x_l, &x_r, -0.125);
        assert_eq!(up_l.len(), 128);
        assert_eq!(up_r.len(), 128);
        let dec = downsample_lane_2x_f64(&up_l, 0.15, 0.50);
        assert_eq!(dec.len(), 64);
        assert!(dec.iter().all(|v| v.is_finite()));
    }
}
