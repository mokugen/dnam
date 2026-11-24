#![forbid(unsafe_code)]

//! Hardware parity and deterministic reduction verification for `tile_mix_fwd_adj` (`dir_mode in 0..=3`)
//! across all `35` lower-triangular Ultra DAG connections (`num_quads = 10`).

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, dispatch_kernel, read_f32_vec, read_vec4_vec,
};
use sona_shaders::abi::{
    BLOCK_WAVE_ALL, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, TILE_DESC_VEC4_STRIDE,
    TILE_FRAMES, TILE_MIX_DIR_FORWARD, TILE_MIX_DIR_PARTIAL_WEIGHTS, TILE_MIX_DIR_PULLBACK,
    TILE_MIX_DIR_REDUCE_WEIGHTS, TILE_WEIGHT_GRAD_FLOATS, TileMixUniforms, block_quad_count,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

const ULTRA_TILES_35: [(usize, usize); 35] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 5),
    (4, 1),
    (4, 5),
    (0, 2),
    (4, 2),
    (0, 3),
    (4, 3),
    (5, 2),
    (1, 3),
    (5, 3),
    (6, 1),
    (6, 5),
    (0, 7),
    (4, 7),
    (6, 2),
    (6, 7),
    (6, 3),
    (1, 7),
    (5, 7),
    (7, 3),
    (0, 9),
    (4, 9),
    (6, 9),
    (8, 1),
    (8, 5),
    (8, 9),
    (8, 2),
    (8, 7),
    (8, 3),
    (9, 2),
    (9, 7),
    (9, 3),
];

fn block_of_quad(q: usize) -> u32 {
    match q {
        0 | 4 | 6 | 8 => 0,
        1 | 5 | 9 => 1,
        2 | 7 => 2,
        _ => 3,
    }
}

pub fn verify_tile_mix_ultra_dag_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let num_frames = 300_usize;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let num_quads = 10_usize;
    let num_stems = 2_usize;
    let active_tiles = ULTRA_TILES_35.len();

    let mut base_inputs =
        vec![[0.0_f32; 4]; num_stems * num_quads * num_frames * CORE_INPUT_VEC4_STRIDE];
    for (idx, v) in base_inputs.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.031;
        *v = [
            0.2 * ph.sin(),
            -0.15 * ph.cos(),
            0.1 * (2.0 * ph).sin(),
            0.25 * (0.5 * ph).cos(),
        ];
    }
    let mut feats = vec![[0.0_f32; 4]; num_stems * num_quads * num_frames * FEATURE_VEC4_STRIDE];
    for (idx, v) in feats.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.017;
        *v = [
            0.3 * ph.sin(),
            0.2 * ph.cos(),
            -0.25 * (1.3 * ph).sin(),
            0.18 * (0.7 * ph).cos(),
        ];
    }
    let mut dl_inputs = vec![[0.0_f32; 4]; base_inputs.len()];
    for (idx, v) in dl_inputs.iter_mut().enumerate() {
        let ph = (idx as f32) * 0.023;
        *v = [
            0.15 * ph.cos(),
            -0.22 * ph.sin(),
            0.19 * (1.1 * ph).cos(),
            -0.14 * ph.sin(),
        ];
    }

    let mut descs = Vec::with_capacity(active_tiles * TILE_DESC_VEC4_STRIDE);
    for (tile_i, &(src_q, dst_q)) in ULTRA_TILES_35.iter().enumerate() {
        descs.push([
            f32::from_bits(src_q as u32),
            f32::from_bits(dst_q as u32),
            f32::from_bits(block_of_quad(src_q)),
            f32::from_bits(block_of_quad(dst_q)),
        ]);
        for mat in 0..3 {
            for s in 0..4 {
                let b = ((tile_i * 3 + mat) * 4 + s) as f32 * 0.01;
                descs.push([
                    0.12 * b.sin(),
                    -0.09 * b.cos(),
                    0.07 * (1.5 * b).sin(),
                    -0.11 * (0.8 * b).cos(),
                ]);
            }
        }
    }

    // 1. Forward mode wave-by-wave (waves 1..=3) vs exact f64 reference.
    let mut gpu_inputs = base_inputs.clone();
    let dummy_wgrads =
        vec![0.0_f32; active_tiles * (1 + num_stems * num_tiles) * TILE_WEIGHT_GRAD_FLOATS];
    let mut dummy_dl_feats = vec![[0.0_f32; 4]; feats.len()];
    for wave in 1..4_u32 {
        let u = TileMixUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_active_tiles: active_tiles as u32,
            dir_mode: TILE_MIX_DIR_FORWARD,
            num_stems: num_stems as u32,
            block_wave: wave,
            num_quads: num_quads as u32,
            _pad1: 0,
        };
        let (out, _) = dispatch_kernel(
            ctx,
            KernelId::TileMixFwdAdj,
            variant,
            (
                block_quad_count(wave as usize, num_quads) as u32,
                num_tiles as u32,
                num_stems as u32,
            ),
            u.as_bytes(),
            &[
                (true, as_byte_slice(&gpu_inputs)),
                (false, as_byte_slice(&descs)),
                (false, as_byte_slice(&feats)),
                (true, as_byte_slice(&dummy_wgrads)),
                (true, as_byte_slice(&dummy_dl_feats)),
            ],
        );
        gpu_inputs = read_vec4_vec(&out[0]);
    }

    let mut cpu_inputs = vec![[0.0_f64; 4]; base_inputs.len()];
    for (i, v) in base_inputs.iter().enumerate() {
        cpu_inputs[i] = v.map(f64::from);
    }
    for (tile_i, &(src_q, dst_q)) in ULTRA_TILES_35.iter().enumerate() {
        let db = tile_i * TILE_DESC_VEC4_STRIDE;
        for stem in 0..num_stems {
            for t in 0..num_frames {
                let ib = ((stem * num_quads + dst_q) * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                let fb = ((stem * num_quads + src_q) * num_frames + t) * FEATURE_VEC4_STRIDE;
                for s in 0..4 {
                    for d in 0..4 {
                        cpu_inputs[ib][d] += f64::from(descs[db + 1 + s][d])
                            * f64::from(feats[fb + 2][s])
                            + f64::from(descs[db + 5 + s][d]) * f64::from(feats[fb + 4][s])
                            + f64::from(descs[db + 9 + s][d]) * f64::from(feats[fb][s]);
                        cpu_inputs[ib + 1][d] += f64::from(descs[db + 1 + s][d])
                            * f64::from(feats[fb + 3][s])
                            + f64::from(descs[db + 5 + s][d]) * f64::from(feats[fb + 5][s])
                            + f64::from(descs[db + 9 + s][d]) * f64::from(feats[fb + 1][s]);
                    }
                }
            }
        }
    }
    for (i, (&g, &c)) in gpu_inputs.iter().zip(&cpu_inputs).enumerate() {
        for l in 0..4 {
            assert!(
                (f64::from(g[l]) - c[l]).abs() <= 1.0e-5,
                "tile_mix fwd ({label}) mismatch idx={i} l={l}: gpu={} cpu={}",
                g[l],
                c[l]
            );
        }
    }

    // 2. Reverse pullback (dir_mode=1) across ALL quads (`block_wave = BLOCK_WAVE_ALL`).
    let u_pb = TileMixUniforms {
        num_frames: num_frames as u32,
        num_tiles: num_tiles as u32,
        num_active_tiles: active_tiles as u32,
        dir_mode: TILE_MIX_DIR_PULLBACK,
        num_stems: num_stems as u32,
        block_wave: BLOCK_WAVE_ALL,
        num_quads: num_quads as u32,
        _pad1: 0,
    };
    let (pb_out, _) = dispatch_kernel(
        ctx,
        KernelId::TileMixFwdAdj,
        variant,
        (num_quads as u32, num_tiles as u32, num_stems as u32),
        u_pb.as_bytes(),
        &[
            (true, as_byte_slice(&dl_inputs)),
            (false, as_byte_slice(&descs)),
            (false, as_byte_slice(&feats)),
            (true, as_byte_slice(&dummy_wgrads)),
            (true, as_byte_slice(&dummy_dl_feats)),
        ],
    );
    let gpu_dl_feats = read_vec4_vec(&pb_out[2]);
    let mut cpu_dl_feats = vec![[0.0_f64; 4]; feats.len()];
    for (tile_i, &(src_q, dst_q)) in ULTRA_TILES_35.iter().enumerate() {
        let db = tile_i * TILE_DESC_VEC4_STRIDE;
        for stem in 0..num_stems {
            for t in 0..num_frames {
                let ib = ((stem * num_quads + dst_q) * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                let fb = ((stem * num_quads + src_q) * num_frames + t) * FEATURE_VEC4_STRIDE;
                for s in 0..4 {
                    for d in 0..4 {
                        cpu_dl_feats[fb + 2][s] +=
                            f64::from(descs[db + 1 + s][d]) * f64::from(dl_inputs[ib][d]);
                        cpu_dl_feats[fb + 3][s] +=
                            f64::from(descs[db + 1 + s][d]) * f64::from(dl_inputs[ib + 1][d]);
                        cpu_dl_feats[fb + 4][s] +=
                            f64::from(descs[db + 5 + s][d]) * f64::from(dl_inputs[ib][d]);
                        cpu_dl_feats[fb + 5][s] +=
                            f64::from(descs[db + 5 + s][d]) * f64::from(dl_inputs[ib + 1][d]);
                        cpu_dl_feats[fb][s] +=
                            f64::from(descs[db + 9 + s][d]) * f64::from(dl_inputs[ib][d]);
                        cpu_dl_feats[fb + 1][s] +=
                            f64::from(descs[db + 9 + s][d]) * f64::from(dl_inputs[ib + 1][d]);
                    }
                }
            }
        }
    }
    for (i, (&g, &c)) in gpu_dl_feats.iter().zip(&cpu_dl_feats).enumerate() {
        for l in 0..4 {
            assert!(
                (f64::from(g[l]) - c[l]).abs() <= 1.0e-5,
                "tile_mix pullback ({label}) mismatch idx={i} l={l}"
            );
        }
    }

    // 3. Deterministic 2-pass weight gradients (`dir_mode = 2` -> `dir_mode = 3`) + repeat bit-identity.
    let run_weights = || -> Vec<f32> {
        let u2 = TileMixUniforms {
            dir_mode: TILE_MIX_DIR_PARTIAL_WEIGHTS,
            block_wave: BLOCK_WAVE_ALL,
            ..u_pb
        };
        let (o2, _) = dispatch_kernel(
            ctx,
            KernelId::TileMixFwdAdj,
            variant,
            (active_tiles as u32, num_tiles as u32, num_stems as u32),
            u2.as_bytes(),
            &[
                (true, as_byte_slice(&dl_inputs)),
                (false, as_byte_slice(&descs)),
                (false, as_byte_slice(&feats)),
                (true, as_byte_slice(&dummy_wgrads)),
                (true, as_byte_slice(&dummy_dl_feats)),
            ],
        );
        let u3 = TileMixUniforms {
            dir_mode: TILE_MIX_DIR_REDUCE_WEIGHTS,
            ..u2
        };
        let (o3, _) = dispatch_kernel(
            ctx,
            KernelId::TileMixFwdAdj,
            variant,
            (active_tiles as u32, 1, 1),
            u3.as_bytes(),
            &[
                (true, as_byte_slice(&dl_inputs)),
                (false, as_byte_slice(&descs)),
                (false, as_byte_slice(&feats)),
                (true, &o2[1]),
                (true, as_byte_slice(&dummy_dl_feats)),
            ],
        );
        read_f32_vec(&o3[1])[..active_tiles * TILE_WEIGHT_GRAD_FLOATS].to_vec()
    };
    let wg_run1 = run_weights();
    let wg_run2 = run_weights();
    for (i, (&a, &b)) in wg_run1.iter().zip(&wg_run2).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "tile_mix ({label}) non-deterministic wgrad idx={i}"
        );
    }

    let mut cpu_wg = vec![0.0_f64; active_tiles * TILE_WEIGHT_GRAD_FLOATS];
    for (tile_i, &(src_q, dst_q)) in ULTRA_TILES_35.iter().enumerate() {
        let base = tile_i * TILE_WEIGHT_GRAD_FLOATS;
        for stem in 0..num_stems {
            for t in 0..num_frames {
                let ib = ((stem * num_quads + dst_q) * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                let fb = ((stem * num_quads + src_q) * num_frames + t) * FEATURE_VEC4_STRIDE;
                for s in 0..4 {
                    for d in 0..4 {
                        let off = s * 4 + d;
                        cpu_wg[base + off] += f64::from(feats[fb + 2][s])
                            * f64::from(dl_inputs[ib][d])
                            + f64::from(feats[fb + 3][s]) * f64::from(dl_inputs[ib + 1][d]);
                        cpu_wg[base + 16 + off] += f64::from(feats[fb + 4][s])
                            * f64::from(dl_inputs[ib][d])
                            + f64::from(feats[fb + 5][s]) * f64::from(dl_inputs[ib + 1][d]);
                        cpu_wg[base + 32 + off] += f64::from(feats[fb][s])
                            * f64::from(dl_inputs[ib][d])
                            + f64::from(feats[fb + 1][s]) * f64::from(dl_inputs[ib + 1][d]);
                    }
                }
            }
        }
    }
    for (i, (&g, &c)) in wg_run1.iter().zip(&cpu_wg).enumerate() {
        assert!(
            (f64::from(g) - c).abs() <= 1.0e-4 * c.abs().max(1.0),
            "tile_mix weight_grad ({label}) mismatch idx={i}: gpu={g} cpu={c}"
        );
    }
    let _ = &mut dummy_dl_feats;
}
