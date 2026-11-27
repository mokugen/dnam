//! Causal `SONA_ULTRA_QUAD_TRAVERSAL_ORDER` stereo core cascade with `InterBlockTile` coupling,
//! ZOH rate rescaling, and `2×` poly-quad oversampling/decimation matching `dnam-vm`
//! (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[dsp.zoh.core-paralru]`).

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};
use dnam_vm::{SONA_CORE_QUAD_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES};

use crate::error::TrainError;
use crate::gpu::abi::{CORE_INPUT_VEC4_STRIDE, TILE_DESC_VEC4_STRIDE};
use crate::runner::TrainRunner;
use crate::runner::types::{WaveCoreCascadeInput, pack_tile_descriptor};

use super::SonaModelF64;
use super::v3_core_bridge::{
    V3_QUAD_PARAMS_VEC4_STRIDE, pack_v3_core_quad_params, run_v3_wave_core_cascade,
};

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

/// Exact linear adjoint (transpose) of `upsample_stereo_2x_f64` for a single channel: maps `2x`
/// core sensitivities (`len = 2 * n`) back to `1x` post-`Op-N[Pre]` sensitivities (`len = n`).
#[must_use]
pub(crate) fn upsample_lane_2x_adjoint_f64(d_sig_2x: &[f64], up_quad_coeff: f64) -> Vec<f64> {
    let coeff = sanitize_unit_f64(up_quad_coeff, -0.20, 0.05, -0.125);
    let n = d_sig_2x.len() / 2;
    let mut out = vec![0.0_f64; n];
    for t in 0..n {
        let d0 = d_sig_2x[2 * t];
        let d1 = d_sig_2x[2 * t + 1];
        out[t] += d1 + (0.5 + coeff) * d0;
        if t >= 1 {
            out[t - 1] += (0.5 - 2.0 * coeff) * d0;
        }
        if t >= 2 {
            out[t - 2] += coeff * d0;
        }
    }
    out
}

/// Exact linear adjoint (transpose) of [`downsample_lane_2x_f64`]: maps `1x` core readout/bus
/// sensitivities (`len = n`) back to `2x` sub-sample sensitivities (`len = 2 * n`).
#[must_use]
pub(crate) fn downsample_lane_2x_adjoint_f64(
    d_sig_1x: &[f64],
    dec_c_prev1: f64,
    dec_c0: f64,
) -> Vec<f64> {
    let c_p1 = sanitize_unit_f64(dec_c_prev1, 0.0, 0.28, 0.15);
    let c0 = sanitize_unit_f64(dec_c0, 0.0, 0.55, 0.50);
    let c1 = 1.0 - c_p1 - c0;
    let n = d_sig_1x.len();
    let mut out = vec![0.0_f64; n * 2];
    for (t, &d) in d_sig_1x.iter().enumerate() {
        out[2 * t] += c0 * d;
        out[2 * t + 1] += c1 * d;
        if t >= 1 {
            out[2 * (t - 1) + 1] += c_p1 * d;
        }
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

/// Packs ZOH-rescaled `CoreQuad` parameters (`q_count * 25`) and active causal
/// `SONA_INTERBLOCK_TILES` descriptors (`active_tiles * 13`) for wave-batched GPU/CPU cascade and
/// adjoint execution.
pub(crate) fn pack_wave_quads_and_tiles(
    model: &SonaModelF64,
    q_count: usize,
    active_tiles: usize,
    rho_core: f64,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut quad_params = Vec::with_capacity(q_count * V3_QUAD_PARAMS_VEC4_STRIDE);
    for q in 0..q_count {
        let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
        let quad = model.core_quads[q].rescaled_for_rho(rho_core);
        quad_params.extend_from_slice(&pack_v3_core_quad_params(
            &quad.to_kernel_params_for_block(block_idx),
        ));
    }

    let mut tile_descriptors = Vec::with_capacity(active_tiles * TILE_DESC_VEC4_STRIDE);
    for (tile_idx, &(src_q, dst_q)) in SONA_INTERBLOCK_TILES[..active_tiles].iter().enumerate() {
        let src_block = SONA_CORE_QUAD_BLOCK_AND_LOCAL[src_q].0 as u32;
        let dst_block = SONA_CORE_QUAD_BLOCK_AND_LOCAL[dst_q].0 as u32;
        let (wo, we, wl) = if src_q < q_count && dst_q < q_count {
            let tile = &model.tiles[tile_idx];
            let mut wo = [[0.0_f32; 4]; 4];
            let mut we = [[0.0_f32; 4]; 4];
            let mut wl = [[0.0_f32; 4]; 4];
            for s in 0..4 {
                for d in 0..4 {
                    wo[s][d] = sanitize_unit_f64(tile.w_odd_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                    we[s][d] = sanitize_unit_f64(tile.w_even_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                    wl[s][d] = sanitize_unit_f64(tile.w_lru_4x4[s][d], -8.0, 8.0, 0.0) as f32;
                }
            }
            (wo, we, wl)
        } else {
            ([[0.0; 4]; 4], [[0.0; 4]; 4], [[0.0; 4]; 4])
        };
        tile_descriptors.extend_from_slice(&pack_tile_descriptor(
            src_q as u32,
            dst_q as u32,
            src_block,
            dst_block,
            &wo,
            &we,
            &wl,
        ));
    }

    (quad_params, tile_descriptors)
}

/// Evaluates active `CoreQuad` blocks with optional precomputed per-quad `Crossbar` injections
/// `(cb_add_l, cb_add_r)` indexed as `[q * n + t][lane]` via wave-batched `Waves 0..=3` +
/// `tile_mix_fwd_adj(0)` execution.
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
    if n == 0 {
        return Ok(CausalCascadeOutput {
            core_inputs: Vec::new(),
            tile_checkpoints: Vec::new(),
            feature_outputs: Vec::new(),
        });
    }
    let q_count = active_quads.clamp(1, super::NUM_CORE_QUADS_ULTRA);
    let active_tiles = match q_count {
        0..=4 => 3_usize,
        5..=6 => 13_usize,
        7..=8 => 23_usize,
        _ => 35_usize,
    };

    let mut base_core_inputs = vec![[0.0_f32; 4]; q_count * n * CORE_INPUT_VEC4_STRIDE];
    for q in 0..q_count {
        let quad = &model.core_quads[q];
        let in_mid: [f64; 4] =
            core::array::from_fn(|d| sanitize_unit_f64(quad.in_mid_weight[d], -16.0, 16.0, 0.0));
        let in_side: [f64; 4] =
            core::array::from_fn(|d| sanitize_unit_f64(quad.in_side_weight[d], -16.0, 16.0, 0.0));
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
            let mut acc_l = [0.0_f32; 4];
            let mut acc_r = [0.0_f32; 4];
            for d in 0..4 {
                acc_l[d] = (in_mid[d] * mid + in_side[d] * side + cb_l[d]) as f32;
                acc_r[d] = (in_mid[d] * mid - in_side[d] * side + cb_r[d]) as f32;
            }
            let base = (q * n + t) * CORE_INPUT_VEC4_STRIDE;
            base_core_inputs[base] = acc_l;
            base_core_inputs[base + 1] = acc_r;
        }
    }

    let (quad_params, tile_descriptors) =
        pack_wave_quads_and_tiles(model, q_count, active_tiles, rho_core);
    let _ = runner;
    let wave = run_v3_wave_core_cascade(&WaveCoreCascadeInput {
        num_frames: n,
        num_quads: q_count,
        num_stems: 1,
        active_tiles,
        write_checkpoints,
        fixed_point_scan: false,
        soft_rect_eps: SOFT_RECT_EPS,
        tiny_crossing_eps: TINY_CROSSING_EPS,
        base_core_inputs: &base_core_inputs,
        quad_params: &quad_params,
        tile_descriptors: &tile_descriptors,
    })?;

    Ok(CausalCascadeOutput {
        core_inputs: wave.core_inputs,
        tile_checkpoints: wave.tile_checkpoints,
        feature_outputs: wave.feature_outputs,
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

        // Adjoint dot-product identities `<A x, y> == <x, A^T y>` to machine precision.
        let y_2x: Vec<f64> = (0..128)
            .map(|i| 0.4 * ((i as f64) * 0.23).cos() - 0.15 * ((i as f64) * 0.07).sin())
            .collect();
        let adj_up = upsample_lane_2x_adjoint_f64(&y_2x, -0.125);
        let lhs_up: f64 = up_l.iter().zip(&y_2x).map(|(a, b)| a * b).sum();
        let rhs_up: f64 = x_l.iter().zip(&adj_up).map(|(a, b)| a * b).sum();
        assert!((lhs_up - rhs_up).abs() < 1.0e-13, "up {lhs_up} vs {rhs_up}");

        let dec_x = downsample_lane_2x_f64(&y_2x, 0.15, 0.50);
        let adj_dec = downsample_lane_2x_adjoint_f64(&x_r, 0.15, 0.50);
        let lhs_dec: f64 = dec_x.iter().zip(&x_r).map(|(a, b)| a * b).sum();
        let rhs_dec: f64 = y_2x.iter().zip(&adj_dec).map(|(a, b)| a * b).sum();
        assert!(
            (lhs_dec - rhs_dec).abs() < 1.0e-13,
            "dec {lhs_dec} vs {rhs_dec}"
        );
    }
}
