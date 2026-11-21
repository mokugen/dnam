#![forbid(unsafe_code)]

//! Hardware parity, `< 1 ms` startup probes, `0-ULP` Ozaki Gram/RHS verification, `target_channels == 0`
//! guard verification, and `prescreen_population` verification (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::harness::{
    GpuContext, ShaderVariant, as_byte_slice, dispatch_kernel, read_f32_vec, read_i32_vec,
};
use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_SAG_GAIN, SonaParaLruDuhemCoeffs4, TptSvfCoeffs4,
    accum_ozaki_chunk_i32, apply_stereo_link_vec4_pair, eval_poly_adaa1_odd_even_vec4,
    pow2_ceil_fp32, prewarp_tan_pi_f_over_fs_f32, reconstruct_ozaki_dot_f64,
    step_paralru_duhem_vec4, step_tpt_svf_vec4,
};
use sona_shaders::abi::{
    OZAKI_WORDS_PER_PAIR, OzakiGramUniforms, PRESCREEN_SEED_PARAM_STRIDE, PrescreenUniforms,
};
use sona_shaders::manifest::KernelId;
use zerocopy::IntoBytes;

pub fn verify_startup_probes_and_ozaki_parity(ctx: &GpuContext, variant: ShaderVariant) {
    let label = variant.label();

    // < 1 ms startup probe: extreme dynamic-range cancellation [+1e4, -1e4, +1e-3] and NaN/Inf scrubbing.
    let probe_frames = 128_usize;
    let mut probe_feat = vec![0.0_f32; 2 * probe_frames];
    let mut probe_tgt = vec![0.0_f32; probe_frames];
    for t in 0..probe_frames {
        probe_feat[t] = if t == 0 {
            1.0e4
        } else if t == 1 {
            -1.0e4
        } else if t == 2 {
            1.0e-3
        } else if t == 3 {
            f32::NAN
        } else if t == 4 {
            f32::INFINITY
        } else {
            ((t as f32) * 0.13).sin() * 0.5
        };
        probe_feat[probe_frames + t] = 1.0;
        probe_tgt[t] = if t == 2 { 1.0 } else { 0.0 };
    }
    let probe_scales = [
        [pow2_ceil_fp32(1.0e4).0, pow2_ceil_fp32(1.0e4).1],
        [1.0_f32, 1.0_f32],
        [1.0_f32, 0.0_f32],
    ];
    let probe_u = OzakiGramUniforms {
        num_frames: probe_frames as u32,
        feature_dim: 2,
        target_channels: 1,
        chunk_len: 1024,
        num_chunks: 1,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let probe_out_init = vec![0_i32; 5 * OZAKI_WORDS_PER_PAIR];
    let (probe_out, _probe_elapsed) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (5, 1, 1),
        probe_u.as_bytes(),
        &[
            (false, as_byte_slice(&probe_feat)),
            (false, as_byte_slice(&probe_tgt)),
            (false, as_byte_slice(&probe_scales)),
            (true, as_byte_slice(&probe_out_init)),
        ],
    );
    let probe_got = read_i32_vec(&probe_out[0]);
    let ref_cancel = accum_ozaki_chunk_i32(
        &probe_feat[..probe_frames],
        &probe_feat[probe_frames..],
        probe_scales[0][1],
        probe_scales[1][1],
    );
    assert_eq!(
        &probe_got[OZAKI_WORDS_PER_PAIR..2 * OZAKI_WORDS_PER_PAIR],
        &ref_cancel,
        "Ozaki probe cancellation mismatch ({label})"
    );
    let cancel_dot = reconstruct_ozaki_dot_f64(&ref_cancel, probe_scales[0][0], probe_scales[1][0]);
    assert!(
        cancel_dot.is_finite(),
        "Ozaki probe NaN/Inf scrubbing failed ({label})"
    );

    // Multi-column, multi-chunk full Gram + RHS parity at bit-exact 0 ULP.
    let num_frames = 1500_usize;
    let feature_dim = 5_usize;
    let target_channels = 2_usize;
    let chunk_len = 1024_usize;
    let num_chunks = 2_usize;
    let total_pairs = feature_dim * (feature_dim + 1) / 2 + feature_dim * target_channels;

    let mut features = vec![0.0_f32; feature_dim * num_frames];
    let mut targets = vec![0.0_f32; target_channels * num_frames];
    let mut col_scales = vec![[0.0_f32; 2]; feature_dim + target_channels];
    for c in 0..feature_dim {
        let mut max_abs = 0.0_f32;
        for t in 0..num_frames {
            let v = ((t as f32) * 0.037 * ((c + 1) as f32)).sin() * (1.5 + c as f32);
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

    let uniforms = OzakiGramUniforms {
        num_frames: num_frames as u32,
        feature_dim: feature_dim as u32,
        target_channels: target_channels as u32,
        chunk_len: chunk_len as u32,
        num_chunks: num_chunks as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let buckets_init = vec![0_i32; total_pairs * num_chunks * OZAKI_WORDS_PER_PAIR];
    let (out, _) = dispatch_kernel(
        ctx,
        KernelId::OzakiGramRhsI32,
        variant,
        (total_pairs as u32, num_chunks as u32, 1),
        uniforms.as_bytes(),
        &[
            (false, as_byte_slice(&features)),
            (false, as_byte_slice(&targets)),
            (false, as_byte_slice(&col_scales)),
            (true, as_byte_slice(&buckets_init)),
        ],
    );
    let got = read_i32_vec(&out[0]);

    let mut pair = 0_usize;
    for i in 0..feature_dim {
        let (sx, isx) = pow2_ceil_fp32(col_scales[i][0]);
        for j in i..feature_dim {
            let (sy, isy) = pow2_ceil_fp32(col_scales[j][0]);
            let mut f64_sum = 0.0_f64;
            for ch in 0..num_chunks {
                let s = ch * chunk_len;
                let e = (s + chunk_len).min(num_frames);
                let ref_acc = accum_ozaki_chunk_i32(
                    &features[i * num_frames + s..i * num_frames + e],
                    &features[j * num_frames + s..j * num_frames + e],
                    isx,
                    isy,
                );
                let base = (pair * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                assert_eq!(
                    &got[base..base + OZAKI_WORDS_PER_PAIR],
                    &ref_acc,
                    "Ozaki ({label}) Gram 0-ULP mismatch pair={pair} ch={ch}"
                );
                f64_sum += reconstruct_ozaki_dot_f64(&ref_acc, sx, sy);
            }
            let (exact, l1_sum): (f64, f64) =
                (0..num_frames).fold((0.0_f64, 0.0_f64), |(acc, l1), t| {
                    let p = (features[i * num_frames + t] as f64)
                        * (features[j * num_frames + t] as f64);
                    (acc + p, l1 + p.abs())
                });
            assert!(
                (f64_sum - exact).abs() <= 1.0e-5 * l1_sum.max(1.0),
                "Ozaki ({label}) f64 recon pair={pair}: recon={f64_sum} exact={exact} l1={l1_sum}"
            );
            pair += 1;
        }
    }
    for i in 0..feature_dim {
        let (sx, isx) = pow2_ceil_fp32(col_scales[i][0]);
        for tc in 0..target_channels {
            let (sy, isy) = pow2_ceil_fp32(col_scales[feature_dim + tc][0]);
            for ch in 0..num_chunks {
                let s = ch * chunk_len;
                let e = (s + chunk_len).min(num_frames);
                let ref_acc = accum_ozaki_chunk_i32(
                    &features[i * num_frames + s..i * num_frames + e],
                    &targets[tc * num_frames + s..tc * num_frames + e],
                    isx,
                    isy,
                );
                let base = (pair * num_chunks + ch) * OZAKI_WORDS_PER_PAIR;
                assert_eq!(
                    &got[base..base + OZAKI_WORDS_PER_PAIR],
                    &ref_acc,
                    "Ozaki ({label}) RHS 0-ULP mismatch pair={pair} ch={ch}"
                );
                let _ = reconstruct_ozaki_dot_f64(&ref_acc, sx, sy);
            }
            pair += 1;
        }
    }

    // Zero target channels guard: pairs >= num_gram_pairs must remain untouched sentinel.
    let zero_tc_frames = 128_usize;
    let zero_tc_dim = 3_usize;
    let zero_tc_gram_pairs = zero_tc_dim * (zero_tc_dim + 1) / 2;
    let zero_tc_dispatch = zero_tc_gram_pairs + 2;
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
    let zt_init = vec![sentinel; zero_tc_dispatch * OZAKI_WORDS_PER_PAIR];
    let zt_u = OzakiGramUniforms {
        num_frames: zero_tc_frames as u32,
        feature_dim: zero_tc_dim as u32,
        target_channels: 0,
        chunk_len: 1024,
        num_chunks: 1,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
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
        let base = slot * OZAKI_WORDS_PER_PAIR;
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
                        let vl = step_paralru_duhem_vec4(u_rl, fl, &mut v_l, &lru);
                        let vr = step_paralru_duhem_vec4(u_rr, fr, &mut v_r, &lru);
                        let (vel, ver) = apply_stereo_link_vec4_pair(vl, vr, lru.stereo_link);
                        let sag_l =
                            (1.0 - seed_params[b + 13] * vel.to_array()[0]).max(MIN_SAG_GAIN);
                        let sag_r =
                            (1.0 - seed_params[b + 13] * ver.to_array()[0]).max(MIN_SAG_GAIN);
                        let r_l = GpuVec4::splat(
                            tone_l * seed_params[b + 14] * sag_l
                                + seed_params[b + 15] * vel.to_array()[0],
                        );
                        let r_r = GpuVec4::splat(
                            tone_r * seed_params[b + 14] * sag_r
                                + seed_params[b + 15] * ver.to_array()[0],
                        );
                        let (ol, _, _) = eval_poly_adaa1_odd_even_vec4(r_prev_l, r_l);
                        let (or, _, _) = eval_poly_adaa1_odd_even_vec4(r_prev_r, r_r);
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
