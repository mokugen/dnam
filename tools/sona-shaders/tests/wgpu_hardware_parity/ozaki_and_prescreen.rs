#![forbid(unsafe_code)]

//! Hardware parity, `< 1 ms` startup probes, `0-ULP` Ozaki Gram/RHS verification, `target_channels == 0`
//! guard verification, and `prescreen_population` verification (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, dispatch_kernel, read_f32_vec, read_i32_vec,
    read_u32_vec,
};
use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_SAG_GAIN, SonaParaLruDuhemCoeffs4, TptSvfCoeffs4,
    accum_ozaki_chunk_i32, apply_stereo_link_vec4_pair, eval_alg_adaa1_vec4, pow2_ceil_fp32,
    prewarp_tan_pi_f_over_fs_f32, reconstruct_ozaki_dot_f64, scrub_clamped_alg_endpoint_f32,
    step_paralru_duhem_v4_vec4, step_tpt_svf_vec4,
};
use sona_shaders::abi::{
    OZAKI_FLAG_ACCUMULATE_SPLIT64, OZAKI_PAIR_TILE_ROWS_DEFAULT, OZAKI_SPLIT64_WORDS_PER_PAIR,
    OZAKI_WORDS_PER_PAIR, OzakiGramUniforms, PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

fn split64_from_chunks(chunks: &[[i32; 6]]) -> [u32; 4] {
    let mut total_hi: i64 = 0;
    let mut total_lo: i64 = 0;
    for c in chunks {
        total_hi = total_hi
            .wrapping_add((i64::from(c[0]) << 16) + (i64::from(c[1]) << 8) + i64::from(c[2]));
        total_lo = total_lo.wrapping_add((i64::from(c[3]) << 8) + i64::from(c[4]));
    }
    let hi_u = total_hi as u64;
    let lo_u = total_lo as u64;
    [
        hi_u as u32,
        (hi_u >> 32) as u32,
        lo_u as u32,
        (lo_u >> 32) as u32,
    ]
}

fn ozaki_2d_wg(total_pairs: usize, rows: u32) -> (u32, u32) {
    let r = rows.max(1) as usize;
    ((total_pairs.min(r)) as u32, total_pairs.div_ceil(r) as u32)
}

pub fn verify_startup_probes_and_ozaki_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();

    // < 1 ms startup probe: extreme dynamic-range cancellation [+1e4, -1e4, +1e-3] and NaN/Inf scrubbing,
    // verified via 2-pass split-64 (reduce_pass = 0 -> reduce_pass = 1).
    let probe_frames = 128_usize;
    let mut probe_feat = vec![0.0_f32; 2 * probe_frames];
    let mut probe_tgt = vec![0.0_f32; probe_frames];
    for t in 0..probe_frames {
        probe_feat[t] = match t {
            0 => 1.0e4,
            1 => -1.0e4,
            2 => 1.0e-3,
            3 => f32::NAN,
            4 => f32::INFINITY,
            _ => ((t as f32) * 0.13).sin() * 0.5,
        };
        probe_feat[probe_frames + t] = 1.0;
        probe_tgt[t] = if t == 2 { 1.0 } else { 0.0 };
    }
    let probe_scales = [
        [pow2_ceil_fp32(1.0e4).0, pow2_ceil_fp32(1.0e4).1],
        [1.0_f32, 1.0_f32],
        [1.0_f32, 0.0_f32],
    ];
    let probe_pairs = 5_usize;
    let probe_tail_base = probe_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let probe_init = vec![0_i32; probe_tail_base + probe_pairs * OZAKI_WORDS_PER_PAIR];
    let mut probe_u = OzakiGramUniforms {
        num_frames: probe_frames as u32,
        feature_dim: 2,
        target_channels: 1,
        chunk_len: 1024,
        num_chunks: 1,
        pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
        reduce_pass: 0,
        _pad: 0,
    };
    let (probe_p0, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (probe_pairs as u32, 1, 1),
        probe_u.as_bytes(),
        &[
            (false, as_byte_slice(&probe_feat)),
            (false, as_byte_slice(&probe_tgt)),
            (false, as_byte_slice(&probe_scales)),
            (true, as_byte_slice(&probe_init)),
        ],
    );
    probe_u.reduce_pass = 1;
    let (probe_p1, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (probe_pairs as u32, 1, 1),
        probe_u.as_bytes(),
        &[
            (false, as_byte_slice(&probe_feat)),
            (false, as_byte_slice(&probe_tgt)),
            (false, as_byte_slice(&probe_scales)),
            (true, &probe_p0[0]),
        ],
    );
    let probe_got_i32 = read_i32_vec(&probe_p1[0]);
    let probe_got_u32 = read_u32_vec(&probe_p1[0]);
    let ref_cancel = accum_ozaki_chunk_i32(
        &probe_feat[..probe_frames],
        &probe_feat[probe_frames..],
        probe_scales[0][1],
        probe_scales[1][1],
    );
    assert_eq!(
        &probe_got_i32
            [probe_tail_base + OZAKI_WORDS_PER_PAIR..probe_tail_base + 2 * OZAKI_WORDS_PER_PAIR],
        &ref_cancel,
        "Ozaki probe pass-0 cancellation mismatch ({label})"
    );
    assert_eq!(
        &probe_got_u32[OZAKI_SPLIT64_WORDS_PER_PAIR..2 * OZAKI_SPLIT64_WORDS_PER_PAIR],
        &split64_from_chunks(&[ref_cancel]),
        "Ozaki probe split-64 cancellation mismatch ({label})"
    );
    let cancel_dot = reconstruct_ozaki_dot_f64(&ref_cancel, probe_scales[0][0], probe_scales[1][0]);
    assert!(
        cancel_dot.is_finite(),
        "Ozaki probe NaN/Inf scrubbing failed ({label})"
    );

    // Multi-column (>256 pairs => 2D wg_y > 1), multi-chunk Gram+RHS parity + accumulate mode at 0 ULP.
    let num_frames = 1500_usize;
    let feature_dim = 24_usize;
    let target_channels = 2_usize;
    let chunk_len = 1024_usize;
    let num_chunks = 2_usize;
    let total_pairs = feature_dim * (feature_dim + 1) / 2 + feature_dim * target_channels;
    assert!(total_pairs > 256);

    let mut features = vec![0.0_f32; feature_dim * num_frames];
    let mut targets = vec![0.0_f32; target_channels * num_frames];
    let mut col_scales = vec![[0.0_f32; 2]; feature_dim + target_channels];
    for c in 0..feature_dim {
        let mut max_abs = 0.0_f32;
        for t in 0..num_frames {
            let v = ((t as f32) * 0.037 * ((c + 1) as f32)).sin() * (1.5 + 0.1 * c as f32);
            features[c * num_frames + t] = v;
            max_abs = max_abs.max(v.abs());
        }
        let (scale, inv) = pow2_ceil_fp32(max_abs);
        col_scales[c] = if c % 2 == 0 {
            [scale, inv]
        } else {
            [max_abs, 0.0]
        };
    }
    for ch in 0..target_channels {
        let mut max_abs = 0.0_f32;
        for t in 0..num_frames {
            let v = ((t as f32) * 0.071 + (ch as f32)).cos() * 2.25;
            targets[ch * num_frames + t] = v;
            max_abs = max_abs.max(v.abs());
        }
        col_scales[feature_dim + ch] = [max_abs, 0.0];
    }

    let (wg_x, wg_y) = ozaki_2d_wg(total_pairs, OZAKI_PAIR_TILE_ROWS_DEFAULT);
    let tail_base = total_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let buckets_init = vec![0_i32; tail_base + total_pairs * num_chunks * OZAKI_WORDS_PER_PAIR];
    let mut uniforms = OzakiGramUniforms {
        num_frames: num_frames as u32,
        feature_dim: feature_dim as u32,
        target_channels: target_channels as u32,
        chunk_len: chunk_len as u32,
        num_chunks: num_chunks as u32,
        pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
        reduce_pass: 0,
        _pad: 0,
    };
    let (out_p0, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (wg_x, wg_y, num_chunks as u32),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&features)),
            (false, as_byte_slice(&targets)),
            (false, as_byte_slice(&col_scales)),
            (true, as_byte_slice(&buckets_init)),
        ],
    );
    uniforms.reduce_pass = 1;
    let (out_p1, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (wg_x, wg_y, 1),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&features)),
            (false, as_byte_slice(&targets)),
            (false, as_byte_slice(&col_scales)),
            (true, &out_p0[0]),
        ],
    );
    let got_i32 = read_i32_vec(&out_p1[0]);
    let got_u32 = read_u32_vec(&out_p1[0]);

    let mut pair = 0_usize;
    for i in 0..feature_dim {
        let (sx, isx) = pow2_ceil_fp32(col_scales[i][0]);
        for j in i..feature_dim {
            let (sy, isy) = pow2_ceil_fp32(col_scales[j][0]);
            let mut ref_chunks = Vec::with_capacity(num_chunks);
            for ch in 0..num_chunks {
                let s = ch * chunk_len;
                let e = (s + chunk_len).min(num_frames);
                let ref_acc = accum_ozaki_chunk_i32(
                    &features[i * num_frames + s..i * num_frames + e],
                    &features[j * num_frames + s..j * num_frames + e],
                    isx,
                    isy,
                );
                let base = tail_base + (pair * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                assert_eq!(
                    &got_i32[base..base + OZAKI_WORDS_PER_PAIR],
                    &ref_acc,
                    "Ozaki ({label}) Gram 0-ULP mismatch pair={pair} ch={ch}"
                );
                ref_chunks.push(ref_acc);
            }
            let s64_base = pair * OZAKI_SPLIT64_WORDS_PER_PAIR;
            let expected_s64 = split64_from_chunks(&ref_chunks);
            assert_eq!(
                &got_u32[s64_base..s64_base + OZAKI_SPLIT64_WORDS_PER_PAIR],
                &expected_s64,
                "Ozaki ({label}) split-64 Gram mismatch pair={pair}"
            );
            let hi = (((expected_s64[1] as u64) << 32) | (expected_s64[0] as u64)) as i64;
            let lo = (((expected_s64[3] as u64) << 32) | (expected_s64[2] as u64)) as i64;
            let total_i128 = ((hi as i128) << 16) + (lo as i128);
            let f64_recon =
                (total_i128 as f64) * (2.0_f64).powi(-48) * f64::from(sx) * f64::from(sy);
            let (exact, l1_sum): (f64, f64) =
                (0..num_frames).fold((0.0_f64, 0.0_f64), |(acc, l1), t| {
                    let p = (features[i * num_frames + t] as f64)
                        * (features[j * num_frames + t] as f64);
                    (acc + p, l1 + p.abs())
                });
            assert!(
                (f64_recon - exact).abs() <= 1.0e-5 * l1_sum.max(1.0),
                "Ozaki ({label}) split64 recon pair={pair}: recon={f64_recon} exact={exact}"
            );
            pair += 1;
        }
    }
    for i in 0..feature_dim {
        let isx = pow2_ceil_fp32(col_scales[i][0]).1;
        for tc in 0..target_channels {
            let isy = pow2_ceil_fp32(col_scales[feature_dim + tc][0]).1;
            let mut ref_chunks = Vec::with_capacity(num_chunks);
            for ch in 0..num_chunks {
                let s = ch * chunk_len;
                let e = (s + chunk_len).min(num_frames);
                let ref_acc = accum_ozaki_chunk_i32(
                    &features[i * num_frames + s..i * num_frames + e],
                    &targets[tc * num_frames + s..tc * num_frames + e],
                    isx,
                    isy,
                );
                let base = tail_base + (pair * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                assert_eq!(
                    &got_i32[base..base + OZAKI_WORDS_PER_PAIR],
                    &ref_acc,
                    "Ozaki ({label}) RHS 0-ULP mismatch pair={pair} ch={ch}"
                );
                ref_chunks.push(ref_acc);
            }
            let s64_base = pair * OZAKI_SPLIT64_WORDS_PER_PAIR;
            assert_eq!(
                &got_u32[s64_base..s64_base + OZAKI_SPLIT64_WORDS_PER_PAIR],
                &split64_from_chunks(&ref_chunks),
                "Ozaki ({label}) split-64 RHS mismatch pair={pair}"
            );
            pair += 1;
        }
    }

    // Verify OZAKI_FLAG_ACCUMULATE_SPLIT64 doubles every pair cleanly when re-running reduce_pass = 1.
    uniforms._pad = OZAKI_FLAG_ACCUMULATE_SPLIT64;
    let (out_acc, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (wg_x, wg_y, 1),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&features)),
            (false, as_byte_slice(&targets)),
            (false, as_byte_slice(&col_scales)),
            (true, &out_p1[0]),
        ],
    );
    let got_acc = read_u32_vec(&out_acc[0]);
    for p in 0..total_pairs {
        let b = p * OZAKI_SPLIT64_WORDS_PER_PAIR;
        let single_hi = ((got_u32[b + 1] as u64) << 32) | (got_u32[b] as u64);
        let single_lo = ((got_u32[b + 3] as u64) << 32) | (got_u32[b + 2] as u64);
        let double_hi = ((got_acc[b + 1] as u64) << 32) | (got_acc[b] as u64);
        let double_lo = ((got_acc[b + 3] as u64) << 32) | (got_acc[b + 2] as u64);
        assert_eq!(double_hi, single_hi.wrapping_mul(2));
        assert_eq!(double_lo, single_lo.wrapping_mul(2));
    }

    // Zero target channels guard: pairs >= num_gram_pairs must remain untouched sentinel.
    let zero_tc_frames = 128_usize;
    let zero_tc_dim = 3_usize;
    let zero_tc_gram_pairs = zero_tc_dim * (zero_tc_dim + 1) / 2;
    let zero_tc_dispatch = zero_tc_gram_pairs + 2;
    let zero_tc_tail = zero_tc_gram_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let sentinel = 0x5A5A_5A5A_i32;
    let mut zt_feats = vec![0.0_f32; zero_tc_dim * zero_tc_frames];
    let mut zt_scales = vec![[1.0_f32, 1.0_f32]; zero_tc_dim];
    for c in 0..zero_tc_dim {
        for t in 0..zero_tc_frames {
            zt_feats[c * zero_tc_frames + t] = ((t as f32) * 0.05 * ((c + 1) as f32)).sin();
        }
        zt_scales[c] = [1.0, 1.0];
    }
    let dummy_target = [999.0_f32; 1];
    let zt_init = vec![sentinel; zero_tc_tail + zero_tc_dispatch * OZAKI_WORDS_PER_PAIR];
    let zt_u = OzakiGramUniforms {
        num_frames: zero_tc_frames as u32,
        feature_dim: zero_tc_dim as u32,
        target_channels: 0,
        chunk_len: 1024,
        num_chunks: 1,
        pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
        reduce_pass: 0,
        _pad: 0,
    };
    let (zt_out, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (zero_tc_dispatch as u32, 1, 1),
        zt_u.as_bytes(),
        &[
            (false, as_byte_slice(&zt_feats)),
            (false, as_byte_slice(&dummy_target)),
            (false, as_byte_slice(&zt_scales)),
            (true, as_byte_slice(&zt_init)),
        ],
    );
    let zt_got = read_i32_vec(&zt_out[0]);
    for slot in zero_tc_gram_pairs..zero_tc_dispatch {
        let base = zero_tc_tail + slot * OZAKI_WORDS_PER_PAIR;
        assert_eq!(
            &zt_got[base..base + OZAKI_WORDS_PER_PAIR],
            &[sentinel; OZAKI_WORDS_PER_PAIR],
            "target_channels == 0 wrote out-of-range pair={slot} ({label})"
        );
    }
}

pub fn verify_prescreen_population_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();
    let num_seeds = 4_usize;
    let num_frames = 300_usize;
    let num_quads = 2_usize;
    let num_stems = 2_usize;
    let num_tiles = 2_usize;

    let mut input_audio = vec![[0.0_f32; 2]; num_stems * num_frames];
    let mut target_audio = vec![[0.0_f32; 2]; num_stems * num_frames];
    for s in 0..num_stems {
        for t in 0..num_frames {
            let ph = (t as f32) * 0.11 + (s as f32) * 0.7;
            input_audio[s * num_frames + t] = [0.55 * ph.sin(), 0.42 * (1.3 * ph).cos()];
            target_audio[s * num_frames + t] = [0.22 * (0.9 * ph).sin(), 0.19 * (1.1 * ph).cos()];
        }
    }
    let mut seed_params = vec![0.0_f32; num_seeds * num_quads * PRESCREEN_SEED_PARAM_STRIDE];
    for seed in 0..num_seeds {
        for q in 0..num_quads {
            let b = (seed * num_quads + q) * PRESCREEN_SEED_PARAM_STRIDE;
            seed_params[b] = 0.04 + 0.04 * (seed as f32) + 0.02 * (q as f32);
            seed_params[b + 1] = 0.65 + 0.25 * (seed as f32);
            seed_params[b + 2] = 0.35;
            seed_params[b + 3] = 0.40;
            seed_params[b + 4] = -0.15;
            seed_params[b + 5] = 0.20;
            seed_params[b + 6] = 0.85;
            seed_params[b + 7] = 0.40;
            seed_params[b + 8] = 0.30;
            seed_params[b + 9] = 0.20;
            seed_params[b + 10] = 0.50;
            seed_params[b + 11] = 0.25;
            seed_params[b + 12] = if (seed + q) % 2 == 0 { 1.0 } else { -1.0 };
            seed_params[b + 13] = 0.25;
            seed_params[b + 14] = 1.4;
            seed_params[b + 15] = 0.15;
        }
    }

    let mut expected_losses = Vec::with_capacity(num_stems * num_seeds * num_tiles);
    for s in 0..num_stems {
        for seed in 0..num_seeds {
            for tile in 0..num_tiles {
                let start = tile * 256;
                let end = (start + 256).min(num_frames);
                let mut pred = vec![[0.0_f32; 2]; end - start];
                for q in 0..num_quads {
                    let b = (seed * num_quads + q) * PRESCREEN_SEED_PARAM_STRIDE;
                    let g = prewarp_tan_pi_f_over_fs_f32(seed_params[b]);
                    let k = 1.0 / seed_params[b + 1];
                    let svf = TptSvfCoeffs4::from_gk(GpuVec4::splat(g), GpuVec4::splat(k));
                    let lru = SonaParaLruDuhemCoeffs4::from_clamped(
                        GpuVec4::splat(seed_params[b + 6]),
                        GpuVec4::splat(seed_params[b + 7]),
                        GpuVec4::splat(seed_params[b + 8]),
                        GpuVec4::splat(seed_params[b + 9]),
                        GpuVec4::splat(seed_params[b + 10]),
                        GpuVec4::splat(seed_params[b + 11]),
                    );
                    let (mut s1_l, mut s2_l, mut s1_r, mut s2_r) = (
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                    );
                    let (mut v_l, mut v_r, mut r_prev_l, mut r_prev_r) = (
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                        GpuVec4::zero(),
                    );
                    for (local_t, frame) in (start..end).enumerate() {
                        let [in_l, in_r] = input_audio[s * num_frames + frame];
                        let svf_l =
                            step_tpt_svf_vec4(GpuVec4::splat(in_l), &mut s1_l, &mut s2_l, &svf);
                        let svf_r =
                            step_tpt_svf_vec4(GpuVec4::splat(in_r), &mut s1_r, &mut s2_r, &svf);
                        let tone_l = svf_l.lp.to_array()[0] * seed_params[b + 2]
                            + svf_l.bp.to_array()[0] * seed_params[b + 3]
                            + svf_l.hp.to_array()[0] * seed_params[b + 4]
                            + in_l * seed_params[b + 5];
                        let tone_r = svf_r.lp.to_array()[0] * seed_params[b + 2]
                            + svf_r.bp.to_array()[0] * seed_params[b + 3]
                            + svf_r.hp.to_array()[0] * seed_params[b + 4]
                            + in_r * seed_params[b + 5];
                        let u_rl = GpuVec4::splat(seed_params[b + 12] * tone_l);
                        let u_rr = GpuVec4::splat(seed_params[b + 12] * tone_r);
                        let fl = GpuVec4::splat(2.0 * g * svf_l.bp.to_array()[0]);
                        let fr = GpuVec4::splat(2.0 * g * svf_r.bp.to_array()[0]);
                        let vl = step_paralru_duhem_v4_vec4(u_rl, fl, &mut v_l, &lru);
                        let vr = step_paralru_duhem_v4_vec4(u_rr, fr, &mut v_r, &lru);
                        let (vel, ver) = apply_stereo_link_vec4_pair(vl, vr, lru.stereo_link);
                        let sag_l =
                            (1.0 - seed_params[b + 13] * vel.to_array()[0]).max(MIN_SAG_GAIN);
                        let sag_r =
                            (1.0 - seed_params[b + 13] * ver.to_array()[0]).max(MIN_SAG_GAIN);
                        let r_l = GpuVec4::splat(scrub_clamped_alg_endpoint_f32(
                            tone_l * seed_params[b + 14] * sag_l
                                + seed_params[b + 15] * vel.to_array()[0],
                        ));
                        let r_r = GpuVec4::splat(scrub_clamped_alg_endpoint_f32(
                            tone_r * seed_params[b + 14] * sag_r
                                + seed_params[b + 15] * ver.to_array()[0],
                        ));
                        let (ol, _, _) = eval_alg_adaa1_vec4(r_prev_l, r_l);
                        let (or, _, _) = eval_alg_adaa1_vec4(r_prev_r, r_r);
                        r_prev_l = r_l;
                        r_prev_r = r_r;
                        pred[local_t][0] += ol.to_array()[0];
                        pred[local_t][1] += or.to_array()[0];
                    }
                }
                let mut err = 0.0_f64;
                let mut tgt_e = 0.0_f64;
                for (local_t, frame) in (start..end).enumerate() {
                    let [tl, tr] = target_audio[s * num_frames + frame];
                    let dl = (pred[local_t][0] - tl) as f64;
                    let dr = (pred[local_t][1] - tr) as f64;
                    err += dl * dl + dr * dr;
                    tgt_e += (tl as f64) * (tl as f64) + (tr as f64) * (tr as f64);
                }
                expected_losses.push((err / (tgt_e + 1.0e-5)) as f32);
            }
        }
    }

    let uniforms = PrescreenUniforms {
        num_seeds: num_seeds as u32,
        num_frames: num_frames as u32,
        num_quads: num_quads as u32,
        num_stems: num_stems as u32,
        sample_rate_hz: 48_000.0,
        eps_floor: 1.0e-5,
        _pad0: 0,
        _pad1: 0,
    };
    let losses_init = vec![0.0_f32; num_stems * num_seeds * num_tiles];
    let (out, _) = dispatch_kernel(
        ctx,
        KernelId::PrescreenPopulation,
        variant,
        (num_seeds as u32, num_tiles as u32, num_stems as u32),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&input_audio)),
            (false, as_byte_slice(&target_audio)),
            (false, as_byte_slice(&seed_params)),
            (true, as_byte_slice(&losses_init)),
        ],
    );
    let got = read_f32_vec(&out[0]);
    for i in 0..got.len() {
        let rel = (got[i] - expected_losses[i]).abs() / expected_losses[i].abs().max(1.0e-6);
        assert!(
            rel < 1.0e-4,
            "prescreen ({label}) mismatch idx={i}: got={} exp={} rel={rel}",
            got[i],
            expected_losses[i]
        );
    }
}
