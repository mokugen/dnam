#![forbid(unsafe_code)]
//! `Case #8` 3-Layer Buyer Watermarking (`SONA` `0-ULP` Sign-Flip Gauge, `SO(4)` + Kautz
//! Isometric Gauge, Layer-2 Pole-Angle Phase Coding & Black-Box Phase ID, Layer-3 Tardos
//! `N = 5` Collusion Attacks), split from `chains_and_watermarks.rs` (`< 900` LOC ceiling).

use dnam_capture::proj::export_runtime_dnam;
use dnam_capture::solver::{
    BuyerWatermarkConfig, derive_buyer_phase_seed, derive_buyer_tardos_codebook,
    export_runtime_dnam_watermarked, extract_arena_pole_angles_32,
    identify_pole_angles_blackbox_from_audio, render_blackbox_probe_response,
};
use dnam_engine::container::{
    DnamContainerView, build_dnam_container_from_dnam_bin, extract_dnam_bin,
};
use dnam_engine::provenance::{
    DefaultCryptoProvider, KAUTZ_TOTAL_CHANNELS, TARDOS_CASCADE_WEIGHTS,
    apply_sona_sign_gauge_watermark, derive_sona_sign_gauge_40, extract_structural_hash_224d,
    for_each_tardos_f32_offset_up_to, for_each_tardos_fp16_offset_up_to, is_role_a_cascade_index,
    role_b_head_slot_for_index, score_phase_coding_arena, score_phase_coding_blackbox,
    score_tardos_watermark_7680, structural_cosine_similarity,
};
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::Chain04TubeAmpFmv4x12;
use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3UltraDeltaWire, DnamBinBuildInput, DnamBinView, encode_dnam_bin,
};
use dnam_vm::math::DefaultMathBackend;
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, IntoBytes};

use super::verify_chain_distillation;

fn read_f32_wire(buf: &[u8], off: usize) -> f32 {
    F32::read_from_prefix(&buf[off..]).map_or(0.0, |(v, _)| v.get())
}

fn write_f32_wire(buf: &mut [u8], off: usize, v: f32) {
    let _ = F32::new(v).write_to_prefix(&mut buf[off..]);
}

fn read_u16_wire(buf: &[u8], off: usize) -> u16 {
    U16::read_from_prefix(&buf[off..]).map_or(0, |(v, _)| v.get())
}

fn write_u16_wire(buf: &mut [u8], off: usize, v: u16) {
    let _ = U16::new(v).write_to_prefix(&mut buf[off..]);
}

#[test]
#[ignore = "deferred to M21/M25-B02 (Chain-04 full fixture capture & Matryoshka Ultra Kautz poles)"]
fn test_case_08_three_layer_buyer_watermark_collusion_and_blackbox_phase_id() {
    // verify[train.pipeline.host-parity-and-export]
    let crypto = DefaultCryptoProvider::default();
    let proj_bytes = verify_chain_distillation(Chain04TubeAmpFmv4x12::new(), false);
    let master_dnam = export_runtime_dnam(&proj_bytes, Some(Tier::Ultra)).unwrap();
    let master_view = DnamContainerView::from_bytes(&master_dnam, &crypto).unwrap();
    let master_arena = master_view.arena_storage.as_bytes().to_vec();
    let master_hash = extract_structural_hash_224d(&master_arena).unwrap();

    let mk_cfg = |buyer_idx: u8| {
        let mut buyer_id = [0u8; 16];
        buyer_id[0] = buyer_idx;
        buyer_id[15] = buyer_idx.wrapping_mul(37).wrapping_add(11);
        BuyerWatermarkConfig::new(
            [0xA5; 32],
            "chain04_tube_amp_v3",
            format!("lic_2026_buyer_{buyer_idx:02}"),
            buyer_id,
        )
    };

    // Export 5 colluding buyer copies + 1 innocent buyer codebook/seed
    let buyer_cfgs: Vec<BuyerWatermarkConfig> = (1..=5).map(mk_cfg).collect();
    let innocent_cfg = mk_cfg(99);
    let innocent_code = derive_buyer_tardos_codebook(&innocent_cfg);
    let innocent_phase_seed = derive_buyer_phase_seed(&innocent_cfg);

    let mut buyer_dnams = Vec::with_capacity(5);
    let mut buyer_arenas = Vec::with_capacity(5);
    let mut buyer_codes = Vec::with_capacity(5);

    for cfg in &buyer_cfgs {
        let code = derive_buyer_tardos_codebook(cfg);
        // Role A (64 coordinates) & unmapped tail indices must be 0.0; 5,807 Role B chips must be +-1.0
        let mut role_a_zeros = 0usize;
        let mut role_b_pm1 = 0usize;
        for idx in 0..TARDOS_CASCADE_WEIGHTS {
            if role_b_head_slot_for_index(idx).is_some() {
                assert!(!is_role_a_cascade_index(idx));
                assert!(
                    (code.chips[idx].abs() - 1.0).abs() < 1.0e-6,
                    "Role B chip {idx} must be +-1.0"
                );
                role_b_pm1 += 1;
            } else {
                assert_eq!(
                    code.chips[idx], 0.0,
                    "Role A / unmapped chip {idx} must be untouched (0.0)"
                );
                if is_role_a_cascade_index(idx) {
                    role_a_zeros += 1;
                }
            }
        }
        assert_eq!(role_a_zeros, 64);
        assert_eq!(role_b_pm1, 5_807);

        let wm_dnam = export_runtime_dnam_watermarked(&proj_bytes, cfg, Some(Tier::Ultra)).unwrap();
        let wm_view = DnamContainerView::from_bytes(&wm_dnam, &crypto).unwrap();
        let wm_arena = wm_view.arena_storage.as_bytes().to_vec();

        // 224-D gauge-invariant structural hash similarity > 0.9999 vs master
        let wm_hash = extract_structural_hash_224d(&wm_arena).unwrap();
        let cos_sim = structural_cosine_similarity(&master_hash, &wm_hash);
        assert!(
            cos_sim > 0.9999,
            "224-D structural hash cosine similarity must exceed 0.9999, got {cos_sim:.8}"
        );

        // Layer-2 white-box arena phase coding Z-score > 3.85 (innocent < 3.0)
        let phase_seed = derive_buyer_phase_seed(cfg);
        let z_phase_wb = score_phase_coding_arena(&master_arena, &wm_arena, &phase_seed).unwrap();
        let z_phase_inn =
            score_phase_coding_arena(&master_arena, &wm_arena, &innocent_phase_seed).unwrap();
        assert!(
            z_phase_wb > 3.85,
            "Layer-2 white-box Z must beat 3.85, got {z_phase_wb:.2}"
        );
        assert!(
            z_phase_inn.abs() < 3.0,
            "Innocent phase Z too high: {z_phase_inn:.2}"
        );

        buyer_dnams.push(wm_dnam);
        buyer_arenas.push(wm_arena);
        buyer_codes.push(code);
    }

    // 0-ULP SONA sign-flip gauge parity test (Layer 1)
    let master_dbin = extract_dnam_bin(&master_dnam).unwrap();
    let master_bin_view = DnamBinView::parse(master_dbin).unwrap();
    let mut signed_arena = master_arena.clone();
    let sign_seed = derive_buyer_phase_seed(&buyer_cfgs[0]);
    apply_sona_sign_gauge_watermark(&mut signed_arena, &derive_sona_sign_gauge_40(&sign_seed))
        .unwrap();
    let a0 = Arena0LiteWire::ref_from_prefix(&signed_arena).unwrap().0;
    let a1 = Arena1FullDeltaWire::ref_from_prefix(&signed_arena[ARENA_TIER_BYTES[0]..])
        .unwrap()
        .0;
    let a2 = Arena2ProDeltaWire::ref_from_prefix(&signed_arena[ARENA_TIER_BYTES[1]..])
        .unwrap()
        .0;
    let a3 = Arena3UltraDeltaWire::ref_from_prefix(&signed_arena[ARENA_TIER_BYTES[2]..])
        .unwrap()
        .0;
    let signed_dbin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: master_bin_view.header.reference_sample_rate_hz.get(),
        header_flags: master_bin_view.header.flags.get(),
        ext_ring_pow2_frames: master_bin_view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: master_bin_view.runtime_meta.ext_ring_lines,
        ext_ring_format: master_bin_view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: master_bin_view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: master_bin_view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: master_bin_view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: master_bin_view.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: master_bin_view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: master_bin_view.runtime_meta.content_hash_prefix,
        controls: master_bin_view.control_descriptors,
        cp1_overflow: master_bin_view.cp1_overflow,
        tsyn: master_bin_view.tsyn_entries,
        cext: master_bin_view.cext,
        arena0: a0,
        arena1: Some(a1),
        arena2: Some(a2),
        arena3: Some(a3),
    })
    .unwrap();
    let mut signed_hdr = master_view.header.clone();
    let signed_dnam = build_dnam_container_from_dnam_bin(&mut signed_hdr, &signed_dbin).unwrap();

    // Perceptual audio transparency between master and Buyer #1 watermarked copy (+ 0-ULP SONA sign gauge)
    let mut eng_m =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&master_dnam, Tier::Ultra, 48_000)
            .unwrap();
    let mut eng_s =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&signed_dnam, Tier::Ultra, 48_000)
            .unwrap();
    let mut eng_w =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&buyer_dnams[0], Tier::Ultra, 48_000)
            .unwrap();
    let mut max_diff = 0.0_f32;
    for n in 0..1_024 {
        let x = 0.25 * (std::f32::consts::TAU * 330.0 * (n as f32) / 48_000.0).sin();
        let ym = eng_m.step_frame_sona(x, x, false);
        let ys = eng_s.step_frame_sona(x, x, false);
        assert_eq!(
            ym[0].to_bits(),
            ys[0].to_bits(),
            "SONA sign gauge L 0-ULP at {n}"
        );
        assert_eq!(
            ym[1].to_bits(),
            ys[1].to_bits(),
            "SONA sign gauge R 0-ULP at {n}"
        );
        let yw = eng_w.step_frame_sona(x, x, false);
        max_diff = max_diff.max((ym[0] - yw[0]).abs().max((ym[1] - yw[1]).abs()));
    }
    assert!(
        max_diff < 2.0e-2,
        "Watermarked copy must remain perceptually transparent: {max_diff}"
    );

    // N = 5 Collusion Attacks: (1) Linear Average, (2) Coordinate Median, (3) 64B Block Interleave
    let arena_len = master_arena.len();
    let mut colluded_avg = master_arena.clone();
    let mut colluded_med = master_arena.clone();
    let mut colluded_int = master_arena.clone();

    for_each_tardos_f32_offset_up_to(arena_len, |_idx, off| {
        let vals: [f32; 5] = std::array::from_fn(|b| read_f32_wire(&buyer_arenas[b], off));
        if vals.iter().all(|v| v.is_finite()) {
            let mean = vals.iter().sum::<f32>() / 5.0;
            write_f32_wire(&mut colluded_avg, off, mean);
            let mut sorted = vals;
            sorted.sort_by(|a, b| a.total_cmp(b));
            write_f32_wire(&mut colluded_med, off, sorted[2]);
        }
    });
    for_each_tardos_fp16_offset_up_to(arena_len, |_idx, off| {
        let vals: [f32; 5] =
            std::array::from_fn(|b| fp16_to_f32(read_u16_wire(&buyer_arenas[b], off)));
        if vals.iter().all(|v| v.is_finite()) {
            let mean = vals.iter().sum::<f32>() / 5.0;
            write_u16_wire(&mut colluded_avg, off, f32_to_fp16(mean));
            let mut sorted = vals;
            sorted.sort_by(|a, b| a.total_cmp(b));
            write_u16_wire(&mut colluded_med, off, f32_to_fp16(sorted[2]));
        }
    });
    for blk in 0..(arena_len / 64) {
        let src = &buyer_arenas[blk % 5][blk * 64..(blk + 1) * 64];
        colluded_int[blk * 64..(blk + 1) * 64].copy_from_slice(src);
    }

    for (attack_name, suspect) in [
        ("linear_average", &colluded_avg),
        ("coordinate_median", &colluded_med),
        ("block_interleave", &colluded_int),
    ] {
        let max_colluder_z = buyer_codes
            .iter()
            .map(|c| {
                score_tardos_watermark_7680(&master_arena, suspect, &c.accusation_weights).unwrap()
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let innocent_z =
            score_tardos_watermark_7680(&master_arena, suspect, &innocent_code.accusation_weights)
                .unwrap();
        assert!(
            max_colluder_z > 8.0,
            "{attack_name}: N=5 Tardos max colluder Z must exceed 8.0, got {max_colluder_z:.2}"
        );
        assert!(
            innocent_z.abs() < 3.5,
            "{attack_name}: innocent buyer |Z| must stay < 3.5, got {innocent_z:.2}"
        );
    }

    // Black-box audio-only Layer-2 pole-angle phase identification (no arena inspection of suspect!)
    let suspect_probe_audio = render_blackbox_probe_response(&buyer_dnams[0], 4_096).unwrap();
    let master_angles_gt = extract_arena_pole_angles_32(&master_arena);
    assert_eq!(master_angles_gt.len(), KAUTZ_TOTAL_CHANNELS);
    let (master_angles, est_suspect_angles) =
        identify_pole_angles_blackbox_from_audio(&master_dnam, &suspect_probe_audio).unwrap();
    let buyer0_seed = derive_buyer_phase_seed(&buyer_cfgs[0]);
    let z_bb_guilty =
        score_phase_coding_blackbox(&master_angles, &est_suspect_angles, &buyer0_seed).unwrap();
    let z_bb_innocent =
        score_phase_coding_blackbox(&master_angles, &est_suspect_angles, &innocent_phase_seed)
            .unwrap();
    assert!(
        z_bb_guilty >= 3.85,
        "Black-box audio-only pole-angle phase Z must be >= 3.85, got {z_bb_guilty:.2}"
    );
    assert!(
        z_bb_innocent.abs() < 3.0,
        "Black-box innocent buyer |Z| must stay < 3.0, got {z_bb_innocent:.2}"
    );
}
