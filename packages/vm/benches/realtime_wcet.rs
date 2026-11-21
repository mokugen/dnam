//! Non-Criterion Real-Time WCET, Tail-Latency Jitter (`P50`/`P99`/`P99.9`/`Max`),
//! `ns/sample`, `x Real-Time @ 48 kHz`, and `% Single-Core CPU` benchmark (`benches/realtime_wcet.rs`).
//!
//! Run with:
//!   `cargo bench -p dnam-vm --bench realtime_wcet`

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Instant;

use dnam_vm::bin::{
    PIPE_CANONICAL_V3_SONA, SONA_META_CORE_MODS_PRESENT, SONA_META_OS_EXT_PRESENT,
    SONA_META_ROUTER_EXT, SONA_META_TANK_LFO, SONA_META_TANK_TAP_GAINS, TANK_FLAG_EXT_F32,
    TANK_FLAG_SCHROEDER_ENABLE,
};
use dnam_vm::math::{TANK_DC_BLOCKER_48K_F32, f32_to_fp16};
use dnam_vm::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, CextDispatchWire,
    ControlClampDescriptor, DefaultMathBackend, DnamBinBuildInput, DnamBinView, DynamicSonaVm,
    FtzDazGuard, KautzSoaBlockWire, OpNPreBaseWire, SonaCrossbarBlockWire, SonaGroupWire,
    SonaHotSlab, SonaInterBlockTilePairWire, TankQuadWire, Tier, encode_dnam_bin,
};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromZeros, IntoBytes};

fn le_f32<const N: usize>(vals: [f32; N]) -> [F32; N] {
    core::array::from_fn(|i| F32::new(vals[i]))
}

fn fp16_arr<const N: usize>(vals: [f32; N]) -> [U16; N] {
    core::array::from_fn(|i| U16::new(f32_to_fp16(vals[i])))
}

fn fp16_val(v: f32) -> U16 {
    U16::new(f32_to_fp16(v))
}

fn make_group(g: usize) -> SonaGroupWire {
    let s = 1.0 + 0.05 * (g as f32);
    let rs = if (g & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
    SonaGroupWire {
        svf_g: le_f32([0.14 * s, 0.28 * s, 0.45 * s, 0.70 * s]),
        svf_k: le_f32([0.70, 0.75, 0.80, 0.85]),
        lru_alpha_rel: le_f32([0.85, 0.87, 0.89, 0.91]),
        lru_alpha_atk: le_f32([0.20, 0.24, 0.28, 0.32]),
        lru_kappa: le_f32([0.35, 0.40, 0.45, 0.50]),
        adaa_drive: le_f32([0.30, 0.34, 0.28, 0.32]),
        lru_bias_drive: le_f32([0.04 * rs, -0.03 * rs, 0.05 * rs, -0.02 * rs]),
        in_x_weight: le_f32([0.22, -0.18, 0.24, -0.16]),
        tone_lp_fp16: fp16_arr([0.35, 0.30, 0.25, 0.20]),
        tone_bp_fp16: fp16_arr([0.20, 0.25, 0.20, 0.15]),
        tone_hp_fp16: fp16_arr([0.15, 0.15, 0.20, 0.25]),
        tone_ap_fp16: fp16_arr([0.10, 0.12, 0.15, 0.18]),
        tone_dir_fp16: fp16_arr([0.20, 0.18, 0.20, 0.22]),
        sag_depth_fp16: fp16_arr([0.05; 4]),
        wavefolder_mix_fp16: fp16_arr([0.04; 4]),
        rect_sign_fp16: fp16_arr([rs, -rs, rs, -rs]),
    }
}

fn make_tile(t: usize) -> SonaInterBlockTilePairWire {
    let mut tile = SonaInterBlockTilePairWire::new_zeroed();
    let sign = if (t & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
    for k in 0..4 {
        tile.w_odd_4x4[k][k] = F32::new(0.035 * sign);
        tile.w_even_4x4_fp16[k][k] = fp16_val(0.015 * sign);
        tile.w_lru_4x4_fp16[k][k] = fp16_val(0.012 * sign);
    }
    tile
}

fn make_crossbar() -> SonaCrossbarBlockWire {
    let mut blk = SonaCrossbarBlockWire::new_zeroed();
    for g in 0..10 {
        for l in 0..4 {
            blk.op_p_to_group_fp16[g][l] = fp16_val(0.05);
            blk.phasor_to_group_fp16[g][l] = fp16_val(-0.04);
        }
    }
    for i in 0..16 {
        blk.opt_to_lr_fp16[i] = [fp16_val(0.02), fp16_val(-0.02)];
    }
    blk.phasor_stereo_phase_fp16[0..4].copy_from_slice(&fp16_arr([0.2, 0.5, 0.8, 1.1]));
    blk
}

fn make_kautz() -> KautzSoaBlockWire {
    KautzSoaBlockWire {
        r_low: le_f32([0.68, 0.72, 0.76, 0.80]),
        dr: le_f32([0.04, 0.05, 0.04, 0.06]),
        theta: le_f32([0.14, 0.30, 0.48, 0.70]),
        b_gamma: le_f32([0.08, 0.07, 0.06, 0.05]),
        c_re_l: le_f32([0.025, -0.018, 0.015, -0.012]),
        c_im_l: le_f32([0.010, 0.014, -0.011, 0.009]),
        c_re_r: le_f32([0.022, 0.016, -0.014, 0.011]),
        c_im_r: le_f32([-0.012, 0.011, 0.013, -0.008]),
    }
}

fn make_quad(q: usize) -> TankQuadWire {
    let base = 17.0 + 24.0 * (q as f32);
    TankQuadWire {
        delay_samples: le_f32([base, base + 6.0, base + 12.0, base + 18.0]),
        ramp_spans_samples: le_f32([4.0, 5.0, 6.0, 7.0]),
        g_loop: F32::new(0.42),
        r_dc: F32::new(TANK_DC_BLOCKER_48K_F32),
        damp_pole: F32::new(0.25),
        lfo_rate_hz: F32::new(1.5 + 0.3 * (q as f32)),
        ramp_rate_hz: F32::new(0.8),
        vca_pole: F32::new(0.95),
        lfo_depths_fp16: fp16_arr([0.4; 4]),
        lfo_phase_off_q16: [
            U16::new(0),
            U16::new(0x4000),
            U16::new(0x8000),
            U16::new(0xC000),
        ],
        ramp_phase_off_q16: [
            U16::new(0x1000),
            U16::new(0x5000),
            U16::new(0x9000),
            U16::new(0xD000),
        ],
        input_gains_fp16: fp16_arr([0.35; 4]),
        wet_tap_l_fp16: fp16_arr([0.12, -0.10, 0.11, -0.09]),
        wet_tap_r_fp16: fp16_arr([-0.09, 0.11, -0.08, 0.12]),
        allpass_g_fp16: if q == 0 {
            fp16_arr([0.30, -0.25, 0.20, -0.15])
        } else {
            fp16_arr([0.0; 4])
        },
        fb_psi_fp16: fp16_val(0.22),
        sigma_knee_fp16: fp16_val(1.5),
        dry_mix_fp16: fp16_val(0.25),
        window_depth_fp16: fp16_val(0.45),
        vca_sens_fp16: fp16_val(0.15),
        allpass_delays: if q == 0 { [11, 17, 23, 31] } else { [0; 4] },
        flags: if q == 0 {
            TANK_FLAG_SCHROEDER_ENABLE
        } else {
            TANK_FLAG_EXT_F32
        },
        reserved: 0,
    }
}

fn fill_ro(words: &mut [U16], lanes: usize) {
    let wc = lanes * 12;
    for l in 0..lanes {
        let b = l * 12;
        words[b] = fp16_val(0.04);
        words[b + 1] = fp16_val(0.035);
        words[b + 4] = fp16_val(0.03);
        words[b + 5] = fp16_val(0.025);
        words[b + 6] = fp16_val(0.04);
        words[b + 7] = fp16_val(0.035);
        words[b + 10] = fp16_val(0.03);
        words[b + 11] = fp16_val(0.025);
    }
    for i in 0..8 {
        words[wc + i] = fp16_val(if i == 4 || i == 5 { 0.5 } else { 0.1 });
    }
}

fn build_bench_ultra_fixture() -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    let meta = SONA_META_ROUTER_EXT
        | SONA_META_OS_EXT_PRESENT
        | SONA_META_TANK_TAP_GAINS
        | SONA_META_TANK_LFO
        | SONA_META_CORE_MODS_PRESENT;
    a0.crossbar.pipeline_flags = U32::new(PIPE_CANONICAL_V3_SONA);
    a0.crossbar.sona_meta_flags = U32::new(meta);
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.crossbar.cell_desc.oversample_factor = U16::new(2);
    a0.crossbar.cell_desc.cell_flags = U32::new(meta);
    a0.crossbar.carrier_basis_w_fp16 = [fp16_arr([0.1; 4]); 3];
    a0.crossbar.op_n_pre_base = OpNPreBaseWire {
        gamma_bc: F32::new(0.05),
        zoh_hold_depth: F32::new(0.04),
        beta_q: F32::new(0.08),
    };
    a0.crossbar.phasor_freq_hz = le_f32([1.5, 3.0, 5.0, 7.0]);
    a0.crossbar.gate_and_crossbar_fp16 =
        fp16_arr([0.002, 0.0005, 500.0, 0.95, 0.1, 0.02, 0.15, 0.1]);
    a0.groups = core::array::from_fn(make_group);
    a0.tiles = core::array::from_fn(make_tile);
    a0.cabinet_fir_lite.fir_l[0] = F32::new(0.5);
    a0.cabinet_fir_lite.fir_r[0] = F32::new(0.5);
    a0.cabinet_kautz_lite = [make_kautz(), make_kautz()];
    a0.spatial_tank_lite = make_quad(0);
    a0.varpro_readout_lite.direct_s2_u_gain = le_f32([0.5, 0.5]);
    fill_ro(&mut a0.varpro_readout_lite.w_ro_fp16[..208], 16);
    a0.control_bus_lite.lfo_dphi_train = le_f32([0.004, 0.008]);
    a0.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(2);
    a0.control_bus_lite.cp1_dir.entries[0] = [0, 1, 0xFF];
    a0.control_bus_lite.cp1_dir.entries[1] = [1, 1, 0xFF];
    for f in 0..4 {
        for d in 0..8 {
            let sign = if (f + d) % 2 == 0 { 1.0 } else { -1.0 };
            a0.control_bus_lite.u_int_base[f][d] = F32::new(sign * (0.015 + 0.002 * d as f32));
        }
    }
    for r in 0..16 {
        for c in 0..16 {
            let sign = if (r + c) % 3 == 0 { -1.0 } else { 1.0 };
            a0.control_bus_lite.w_ctrl[r][c] = F32::new(sign * (0.004 + 0.001 * (c % 4) as f32));
        }
    }
    for i in 0..13 {
        let off = i * 4;
        a0.control_bus_lite.cp1_base[off..off + 4].copy_from_slice(F32::new(0.1).as_bytes());
        a0.control_bus_lite.cp1_base[52 + off..56 + off].copy_from_slice(F32::new(0.08).as_bytes());
    }

    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    a1.groups = [make_group(4), make_group(5)];
    a1.tiles = core::array::from_fn(|t| make_tile(3 + t));
    a1.crossbar_full = make_crossbar();
    a1.cabinet_kautz_full = make_kautz();
    a1.spatial_tank_full = make_quad(1);
    fill_ro(&mut a1.varpro_readout_full.w_ro_and_mods_fp16, 24);

    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    a2.groups = [make_group(6), make_group(7)];
    a2.tiles = core::array::from_fn(|t| make_tile(13 + t));
    a2.crossbar_pro = make_crossbar();
    a2.cabinet_kautz_pro = [make_kautz(), make_kautz(), make_kautz()];
    a2.spatial_tank_pro[0] = make_quad(2);
    fill_ro(&mut a2.varpro_readout_pro.w_ro_and_mods_fp16, 32);

    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    a3.groups = [make_group(8), make_group(9)];
    a3.tiles = core::array::from_fn(|t| make_tile(23 + t));
    a3.crossbar_ultra = make_crossbar();
    for k in 0..8 {
        for col in 0..120 {
            a3.stage_latent_core_mods_fp16[k * 128 + col] = fp16_val(0.01);
        }
    }
    a3.cabinet_kautz_ultra = [make_kautz(), make_kautz()];
    a3.spatial_tank_ultra[0] = make_quad(3);
    fill_ro(&mut a3.varpro_readout_ultra.w_ro_and_mods_fp16, 40);

    for r in 0..16 {
        for c in 0..4 {
            let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            a3.control_bus_ultra_delta.w_ctrl_ultra[r][c] = F32::new(sign * 0.005);
        }
    }
    for b in 0..3 {
        for f in 0..4 {
            for d in 0..8 {
                let sign = if (b + f + d) % 2 == 0 { 1.0 } else { -1.0 };
                a3.control_bus_ultra_delta.u_int_ultra[b][f][d] = F32::new(sign * 0.01);
            }
        }
    }
    for i in 0..12 {
        let off = i * 4;
        a3.control_bus_ultra_delta.cp1_ultra[off..off + 4]
            .copy_from_slice(F32::new(0.04).as_bytes());
        a3.control_bus_ultra_delta.cp1_ultra[48 + off..52 + off]
            .copy_from_slice(F32::new(0.03).as_bytes());
    }
    for idx in 0..160 {
        let code = if idx % 2 == 0 { 0x2C00 } else { 0x2800 };
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[idx] = U16::new(code);
    }

    let c0 = ControlClampDescriptor::range(0.0, 1.0, 0.5, 2, [0.0; 3]).to_wire();
    let c1 = ControlClampDescriptor::range(0.0, 1.0, 0.5, 2, [0.0; 3]).to_wire();
    let mut cext = CextDispatchWire::new_zeroed();
    for r in 0..16 {
        for c in 0..16 {
            let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            cext.w_ctrl_ext[r][c] = F32::new(sign * 0.003);
        }
        for c in 0..4 {
            let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            cext.w_ctrl_ultra_ext[r][c] = F32::new(sign * 0.002);
        }
    }

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: 0,
        ext_ring_pow2_frames: 10,
        ext_ring_lines: 4,
        ext_ring_format: 1,
        declared_max_delay_48k: 200,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 1_000.0,
        r_ac_default: 0.995,
        switch_xfade_samples: 64,
        content_hash_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
        controls: &[c0, c1],
        cp1_overflow: &[],
        tsyn: &[],
        cext: Some(&cext),
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
    .expect("encode bench fixture")
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn measure_bind_prevalidated_ns(ultra_bytes: &[u8]) -> f64 {
    let iters = 10_000_usize;
    let mut samples_ns = Vec::with_capacity(iters);
    for _ in 0..1_000 {
        let v = DnamBinView::bind_prevalidated(black_box(ultra_bytes)).unwrap();
        black_box(v.retained_tier);
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        let v = DnamBinView::bind_prevalidated(black_box(ultra_bytes)).unwrap();
        black_box(v.retained_tier);
        samples_ns.push(t0.elapsed().as_nanos() as f64);
    }
    samples_ns.sort_by(f64::total_cmp);
    percentile(&samples_ns, 50.0)
}

fn bench_vm_block(ultra_bytes: &[u8], tier: Tier, block_frames: usize, label: &str) -> f64 {
    let mut vm = DynamicSonaVm::open_dnam_bin(ultra_bytes, tier, 48_000.0).expect("open VM");
    vm.set_controls_batch(&[(0, 0.85), (1, 0.20)])
        .expect("set_controls_batch");
    let in_l: Vec<f32> = (0..block_frames)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.25 * (2.0 * core::f32::consts::PI * 220.0 * t).sin()
                + 0.10 * (2.0 * core::f32::consts::PI * 1_100.0 * t).cos()
        })
        .collect();
    let in_r: Vec<f32> = (0..block_frames)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            -0.22 * (2.0 * core::f32::consts::PI * 330.0 * t).sin()
                + 0.12 * (2.0 * core::f32::consts::PI * 880.0 * t).cos()
        })
        .collect();
    let mut out_l = vec![0.0_f32; block_frames];
    let mut out_r = vec![0.0_f32; block_frames];

    for _ in 0..64 {
        vm.process_block(black_box(&in_l), black_box(&in_r), &mut out_l, &mut out_r)
            .unwrap();
    }
    assert!(
        out_l.iter().chain(out_r.iter()).all(|s| s.is_finite()),
        "WCET benchmark produced non-finite output on {label}"
    );
    let energy: f64 = out_l
        .iter()
        .chain(out_r.iter())
        .map(|&s| f64::from(s) * f64::from(s))
        .sum();
    assert!(
        energy > 1.0e-6,
        "WCET benchmark produced unexpectedly silent output on {label}: energy={energy}"
    );

    let num_blocks = 2_000_usize;
    let mut latencies_ns = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let t0 = Instant::now();
        vm.process_block(black_box(&in_l), black_box(&in_r), &mut out_l, &mut out_r)
            .unwrap();
        latencies_ns.push(t0.elapsed().as_nanos() as f64);
        black_box((out_l[0], out_r[0]));
    }
    latencies_ns.sort_by(f64::total_cmp);

    let p50_ns = percentile(&latencies_ns, 50.0);
    let p99_ns = percentile(&latencies_ns, 99.0);
    let p999_ns = percentile(&latencies_ns, 99.9);
    let max_ns = *latencies_ns.last().unwrap_or(&0.0);
    let ns_per_sample = p50_ns / (block_frames as f64);
    let budget_ns = (block_frames as f64) * 1_000_000_000.0 / 48_000.0;
    let x_rt = budget_ns / p50_ns;
    let cpu_pct = 100.0 / x_rt;

    println!(
        "| {label:<18} | {block_frames:>5} | {:>7.2} | {:>7.2} | \
         {:>7.2} | {:>7.2} | {:>9.1} | {:>9.2}x | {:>7.3}% |",
        p50_ns / 1_000.0,
        p99_ns / 1_000.0,
        p999_ns / 1_000.0,
        max_ns / 1_000.0,
        ns_per_sample,
        x_rt,
        cpu_pct,
    );
    x_rt
}

fn main() {
    let _ftz = FtzDazGuard::new();
    assert_eq!(
        [
            SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false),
            SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Full, true),
            SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Pro, true),
            SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true),
        ],
        [4_544, 7_552, 11_200, 14_656]
    );

    let ultra_bytes = build_bench_ultra_fixture();
    let bind_p50_ns = measure_bind_prevalidated_ns(&ultra_bytes);
    println!("DnamBinView::bind_prevalidated median latency: {bind_p50_ns:.2} ns");
    if cfg!(not(debug_assertions)) {
        assert!(
            bind_p50_ns < 50.0,
            "bind_prevalidated median {bind_p50_ns:.2} ns exceeded 50 ns budget"
        );
    }

    println!(
        "=========================================================================================="
    );
    println!(
        " dnam-vm Real-Time WCET @ 48 kHz (Hot Bytes: Lite=4544, Full=7552, Pro=11200, Ultra=14656)"
    );
    println!(
        "=========================================================================================="
    );
    println!(
        "| {:<18} | {:>5} | {:>7} | {:>7} | {:>7} | {:>7} | {:>9} | {:>10} | {:>8} |",
        "Tier (OS Mode)",
        "Block",
        "P50(us)",
        "P99(us)",
        "P99.9",
        "Max(us)",
        "ns/sample",
        "x RealTime",
        "CPU Load"
    );
    println!(
        "|--------------------|-------|---------|---------|---------|---------|-----------|------------|----------|"
    );

    let tiers = [
        (Tier::Lite, "Lite  (1x, C=16)", 14.0_f64),
        (Tier::Full, "Full  (2x, C=24)", 8.0_f64),
        (Tier::Pro, "Pro   (2x, C=32)", 4.0_f64),
        (Tier::Ultra, "Ultra (2x, C=40)", 2.0_f64),
    ];
    for &(tier, label, min_rt_128) in &tiers {
        let _ = bench_vm_block(&ultra_bytes, tier, 16, label);
        let _ = bench_vm_block(&ultra_bytes, tier, 64, label);
        let rt_128 = bench_vm_block(&ultra_bytes, tier, 128, label);
        if cfg!(not(debug_assertions)) {
            assert!(
                rt_128 >= min_rt_128,
                "{label} 128-frame throughput {rt_128:.2}x below required {min_rt_128:.1}x real-time"
            );
        }
    }
    println!(
        "=========================================================================================="
    );
}
