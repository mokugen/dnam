#![forbid(unsafe_code)]
//! Step 2.3 Structural Isomorphism Oracles (`Oracle-01`, `Oracle-02`, `Oracle-03`) for the
//! `v3` Deterministic Analytical + `VarPro` Solver (`dnam_capture::solver`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`r[train.pipeline.*]`, `r[train.gram.*]`)
//! - `docs/specs/containers/project_container_format.md` (`r[container.proj.*]`)

#[path = "oracles/cp1_and_vault.rs"]
mod cp1_and_vault;

use std::collections::BTreeMap;

use dnam_capture::CORESET_FP32_FRAMES_PER_STEM;
use dnam_capture::metrics::{evaluate_besr_from_band_energies, split_bezout_3band_energies};
use dnam_capture::proj::{
    DnamProjManifest, PROJ_STAT_BYTES_ULTRA, StemProvenance, StmcTrajectoryWindow,
    VaultCaptureSpec, add_capture_with_options, build_dnam_proj_container_with_cp1,
    build_proj_inpc_capsule, export_runtime_dnam,
};
use dnam_capture::solver::{
    SolverOptions, StageAConfig, Wide192Accum, compute_integrated_lufs, run_stage_a_physics_probe,
    solve_stage_b_backbone,
};
use dnam_capture::stimulus::encode_coreset_fp32;
use dnam_engine::container::{
    DnamContainerView, build_dnam_container_with_cp1, strip_container_to_tier,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, apply_isometric_gauge_watermark, extract_structural_hash_224d,
    structural_cosine_similarity,
};
use dnam_engine::schema::{ControlSchemaDef, DnamJsonHeader};
use dnam_engine::stages::stage2_gru::{flat_gru_lane_index, tier_block_groups, tier_kautz_banks};
use dnam_engine::stages::weights::{
    ARENA_ULTRA_BYTES, pack_v3_matryoshka_to_arena, unpack_v3_matryoshka_from_arena,
};
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};
use dnam_engine::{DnamEngine, Tier};
use sha2::{Digest, Sha256};

pub(super) fn clear_output_readout_weights(ro: &mut V3TierReadout) {
    ro.svf_direct_taps_l = [0.0; 8];
    ro.svf_direct_taps_r = [0.0; 8];
    ro.direct_s2_u_gain = [0.0; 2];
    ro.direct_s2_hp_gain = [0.0; 2];
    ro.direct_s2_w = [[0.0; 36]; 2];
    ro.fir_l = [0.0; 64];
    ro.fir_r = [0.0; 64];
    ro.kautz_c_l = [(0.0, 0.0); 16];
    ro.kautz_c_r = [(0.0, 0.0); 16];
}

pub(super) fn write_f32(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

pub(super) fn pack_dnam(
    hdr: &DnamJsonHeader,
    arena: &[u8],
    slots: &[Cp1ContributorSlot],
) -> Vec<u8> {
    let mut h = hdr.clone();
    build_dnam_container_with_cp1(&mut h, arena, arena, false, slots).unwrap()
}

pub(super) fn ctrl_meta(gain: f64, channel: &str) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        ("gain".into(), serde_json::json!(gain)),
        ("channel".into(), serde_json::json!(channel)),
    ])
}

pub(super) fn make_range_ctrl(id: &str, lo: f32, hi: f32, def: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id, "name": id, "type": "range", "lobatto_degree": 2,
        "declared_range": [lo, hi], "safe_clamped_range": [lo, hi], "default": def
    }))
    .unwrap()
}

pub(super) fn make_switch_ctrl(id: &str, opts: &[&str], def_idx: usize) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id, "name": id, "type": "switch",
        "declared_options": opts, "active_options": opts, "default": opts[def_idx]
    }))
    .unwrap()
}

pub(super) fn stmc_win(
    control_id: u16,
    ctrl_kind: u8,
    active_opt: u8,
    u_phys: f32,
) -> StmcTrajectoryWindow {
    StmcTrajectoryWindow {
        control_id,
        ctrl_kind,
        active_opt,
        u_phys,
        window_weight: 1.0,
        emr_velocity: 0.0,
    }
}

pub(super) fn pcm_sha256_hex(samples: &[f32]) -> String {
    let mut h = Sha256::new();
    for &s in samples {
        h.update((if s == 0.0 { 0.0_f32 } else { s }).to_le_bytes());
    }
    let digest: [u8; 32] = h.finalize().into();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

pub(super) fn make_vault_spec(
    stem_id: &str,
    is_star_center: bool,
    meta: BTreeMap<String, serde_json::Value>,
    dry: &[f32],
    wet_interleaved: &[f32],
    windows: Vec<StmcTrajectoryWindow>,
    optional_spcm: Option<Vec<u8>>,
) -> VaultCaptureSpec {
    let mut wet_padded = vec![0.0_f32; CORESET_FP32_FRAMES_PER_STEM * 2];
    let copy_len = wet_interleaved.len().min(wet_padded.len());
    wet_padded[..copy_len].copy_from_slice(&wet_interleaved[..copy_len]);
    VaultCaptureSpec {
        stem_id: stem_id.to_string(),
        is_star_center,
        enabled: true,
        controls_meta: meta,
        pad_gain_f32: 1.0,
        fractional_lag_samples_f32: 0.0,
        provenance: StemProvenance {
            input_pcm_sha256: pcm_sha256_hex(dry),
            target_pcm_sha256: pcm_sha256_hex(&wet_padded),
            num_frames: CORESET_FP32_FRAMES_PER_STEM as u64,
        },
        user_metadata: BTreeMap::new(),
        extracted_physics: None,
        windows,
        band_psd_summary_88: [0u8; 88],
        n_comb_bins: 32,
        spectral_summary_4096: [0u8; 4_096],
        wet_coreset_fp32: {
            let mut out = Vec::with_capacity(wet_padded.len() * 4);
            for &s in &wet_padded {
                out.extend_from_slice(&s.to_le_bytes());
            }
            out
        },
        optional_spcm,
    }
}

pub(super) fn synthesize_rich_coreset_dry(n: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
            let t = (i as f32) / 48_000.0;
            let env = 0.25 + 0.65 * ((i as f32 * 0.0017).sin().abs());
            let tones = 0.45 * (std::f32::consts::TAU * 110.0 * t).sin()
                + 0.30 * (std::f32::consts::TAU * 440.0 * t).cos()
                + 0.20 * (std::f32::consts::TAU * 1_760.0 * t).sin();
            (env * (0.65 * tones + 0.35 * noise)).clamp(-0.85, 0.85)
        })
        .collect()
}

/// Builds a ground-truth `28,672 B` `v3` Matryoshka arena with active Stage I SVF taps, Stage II
/// GRU blocks (`0..3`), `16` shared Kautz poles, Stage IV coprime delays, and nested 4-tier
/// stereo readouts whose progressive capacity yields strictly monotone `Lite..Ultra` accuracy.
pub(super) fn build_oracle01_teacher_arena_v3() -> Vec<u8> {
    let mut model = V3MatryoshkaModel::identity(false, false);
    model.stage0.xo_hz = 135.0;

    const BLK_LANES: [usize; 3] = [12, 8, 4];
    for (b, &n_lanes) in BLK_LANES.iter().enumerate() {
        let blk = &mut model.blocks[b];
        for feat in 0..blk.w_in_n.len() {
            for lane in 0..n_lanes {
                let s = (((b * 7 + feat * 5 + lane * 3 + 1) % 11) as f32 - 5.0) * 0.025;
                blk.w_in_n[feat][lane] = s;
                blk.w_in_z[feat][lane] = 0.5 * s;
            }
        }
        for i in 0..n_lanes {
            for j in 0..=i {
                blk.w_prev_n[i][j] = 0.04 * (((i + j + b) % 5) as f32 - 2.0) / 2.0;
                blk.w_rec_n[i][j] = 0.05 * (((i * 3 + j + 1) % 7) as f32 - 3.0) / 3.0;
            }
            blk.b_z[i] = -0.3;
        }
    }

    for (m, dst) in model.kautz_poles.iter_mut().enumerate() {
        let r = 0.86_f32 + 0.006 * (m as f32);
        let theta = 0.14_f32 + 0.14 * (m as f32);
        *dst = (r, theta);
    }

    let coprime_delays: [[usize; 4]; 4] = [
        [29, 43, 59, 73],
        [101, 137, 173, 199],
        [251, 293, 331, 373],
        [419, 467, 509, 547],
    ];
    for t in 0..4 {
        model.stage4[t].delay_lengths = coprime_delays[t];
        model.stage4[t].g_loop = 0.25;
        model.stage4[t].wet_mix = 0.0;
    }

    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut prev_ro = V3TierReadout::zeros();
    prev_ro.pre_cab_u_gain = [0.75, 0.72];
    prev_ro.direct_s2_u_gain = [0.22, -0.18];
    prev_ro.direct_s2_hp_gain = [0.06, 0.06];
    for ch in 0..2 {
        for i in 0..6 {
            prev_ro.svf_direct_taps_l[i] = 0.02 * ((i + 1) as f32);
            prev_ro.svf_direct_taps_r[i] = 0.02 * ((i + 1) as f32) * (1.0 - 0.1 * (ch as f32));
        }
    }
    // Share `pre_cab_w` on group 0 across all 4 tiers so `s2` is invariant (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`).
    for b in 0..3 {
        for l in 0..4 {
            let flat = flat_gru_lane_index(b, 0, l);
            let phase = ((flat * 5 + 3) % 7) as f32 - 3.0;
            prev_ro.pre_cab_w[0][flat] = 0.045 * phase / 3.0;
            prev_ro.pre_cab_w[1][flat] = 0.039 * (-phase) / 3.0;
        }
    }

    let tier_scales = [0.36_f32, 0.16, 0.07, 0.025];
    for (t_idx, &tier) in tiers.iter().enumerate() {
        let mut ro = prev_ro.clone();
        let sc = tier_scales[t_idx];
        let fir_lo = t_idx * 16;
        let fir_hi = fir_lo + 16;
        for idx in fir_lo..fir_hi {
            let vl = sc * 0.45 * (((idx * 11 + t_idx * 5 + 3) % 13) as f32 - 6.0) / 6.0;
            let vr = sc * 0.45 * (((idx * 7 + t_idx * 9 + 2) % 13) as f32 - 6.0) / 6.0;
            ro.fir_l[idx] = vl;
            ro.fir_r[idx] = vr;
        }
        let k_lo = if t_idx == 0 {
            0
        } else {
            tier_kautz_banks(tiers[t_idx - 1]) * 4
        };
        let k_hi = tier_kautz_banks(tier) * 4;
        for m in k_lo..k_hi {
            let sign = if m % 2 == 0 { 1.0 } else { -0.8 };
            ro.kautz_c_l[m] = (sc * 0.35 * sign, -sc * 0.22 * sign);
            ro.kautz_c_r[m] = (sc * 0.30 * sign, sc * 0.24 * sign);
        }
        for b in 0..3 {
            let g_lo = if t_idx == 0 {
                0
            } else {
                tier_block_groups(tiers[t_idx - 1], b)
            };
            let g_hi = tier_block_groups(tier, b);
            for g in g_lo..g_hi {
                for l in 0..4 {
                    let flat = flat_gru_lane_index(b, g, l);
                    let phase = ((flat * 5 + t_idx * 3) % 7) as f32 - 3.0;
                    ro.direct_s2_w[0][flat] = sc * 0.16 * phase / 3.0;
                    ro.direct_s2_w[1][flat] = sc * 0.15 * phase / 3.0;
                }
            }
        }
        prev_ro = ro.clone();
        ro.mask_to_tier(tier);
        model.readouts[t_idx] = ro;
    }

    let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&model, &mut arena);
    arena
}

pub(super) fn render_stereo_dnam(
    dnam_bytes: &[u8],
    tier: Tier,
    dry: &[f32],
    ctrls: &[(usize, f32)],
) -> Vec<[f32; 2]> {
    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(dnam_bytes, tier, 48_000).unwrap();
    for &(idx, val) in ctrls {
        eng.set_control_immediate(idx, val);
    }
    dry.iter()
        .map(|&x| eng.step_frame_tora(x, x, false))
        .collect()
}

pub(super) fn esr_db_stereo(target: &[[f32; 2]], pred: &[[f32; 2]]) -> f64 {
    let skip = 128.min(target.len() / 8);
    let mut err = 0.0_f64;
    let mut sig = 0.0_f64;
    for (&[tl, tr], &[pl, pr]) in target[skip..].iter().zip(&pred[skip..]) {
        let dl = f64::from(tl) - f64::from(pl);
        let dr = f64::from(tr) - f64::from(pr);
        err += dl * dl + dr * dr;
        sig += f64::from(tl) * f64::from(tl) + f64::from(tr) * f64::from(tr);
    }
    10.0 * (err / sig.max(1.0e-15)).max(1.0e-15).log10()
}

pub(super) fn besr_3band_stereo_db(target: &[[f32; 2]], pred: &[[f32; 2]]) -> (f64, [f64; 3]) {
    let ch_rep = |ch: usize| {
        let t: Vec<f64> = target.iter().map(|v| f64::from(v[ch])).collect();
        let p: Vec<f64> = pred.iter().map(|v| f64::from(v[ch])).collect();
        let (s, e, n) = split_bezout_3band_energies(&t, &p, None, 48_000.0, 512);
        evaluate_besr_from_band_energies(s, e, n)
    };
    let (rl, rr) = (ch_rep(0), ch_rep(1));
    (
        rl.besr_db.max(rr.besr_db),
        std::array::from_fn(|b| rl.band_esr_db[b].max(rr.band_esr_db[b])),
    )
}

pub(super) fn interleave_lr(frames: &[[f32; 2]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames.len() * 2);
    for &[l, r] in frames {
        out.push(l);
        out.push(r);
    }
    out
}

pub(super) fn assert_0_ulp_stereo(a: &[[f32; 2]], b: &[[f32; 2]], msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x[0].to_bits(), y[0].to_bits(), "{msg}: L ULP mismatch");
        assert_eq!(x[1].to_bits(), y[1].to_bits(), "{msg}: R ULP mismatch");
    }
}

#[test]
fn test_oracle_01_256d_schur_recovery_prefix_strip_and_gauge_invariance() {
    // verify[train.pipeline.stage-b-harvest]
    // verify[train.pipeline.stage-c-schur]
    // verify[train.pipeline.isomorphism-parity]
    // verify[train.gram.accuracy-contract]
    // verify[train.gram.wide192]
    let teacher_arena = build_oracle01_teacher_arena_v3();
    let mut teacher_hdr = DnamJsonHeader::default_reference();
    teacher_hdr.io_config.output_mode = "stereo".to_string();
    let teacher_dnam = pack_dnam(&teacher_hdr, &teacher_arena, &[]);

    // 1. Wide192Accum cancellation immunity: [1e8, 2^-30, -1e8] repeated 4,096 times
    let tiny = 2.0_f64.powi(-30);
    let mut wide = Wide192Accum::default();
    let mut naive = 0.0_f64;
    for _ in 0..4_096 {
        wide.add_f64(1.0e8);
        wide.add_f64(tiny);
        wide.add_f64(-1.0e8);
        naive = ((naive + 1.0e8) + tiny) - 1.0e8;
    }
    let expected_sum = 4_096.0 * tiny;
    assert!((wide.to_f64() - expected_sum).abs() <= 1.0e-18);
    assert!((naive - expected_sum).abs() > 1.0e-7);

    // 2. Render 96,000-frame coreset through teacher Ultra model
    let dry = synthesize_rich_coreset_dry(CORESET_FP32_FRAMES_PER_STEM);
    let dry_fp32 = encode_coreset_fp32(&dry, None);
    let wet_ultra = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry, &[]);
    let dry_lr: Vec<[f32; 2]> = dry.iter().map(|&x| [x, x]).collect();

    // Exact f32 Stage B recovery (zero target quantization): must beat -85 dB
    let mut student_model = unpack_v3_matryoshka_from_arena(&teacher_arena, false);
    for ro in &mut student_model.readouts {
        clear_output_readout_weights(ro);
    }
    let mut student_arena = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&student_model, &mut student_arena);

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    let stage_b = solve_stage_b_backbone(
        &manifest,
        &mut student_arena,
        &dry_lr,
        &wet_ultra,
        0,
        1,
        true,
    )
    .expect("Stage B v3 backbone solve");
    assert_eq!(stage_b.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
    let exact_student_dnam = pack_dnam(&manifest.base, &student_arena, &[]);
    let pred_exact_ultra = render_stereo_dnam(&exact_student_dnam, Tier::Ultra, &dry, &[]);
    let esr_exact_db = esr_db_stereo(&wet_ultra, &pred_exact_ultra);
    let (besr_exact_db, besr_exact_bands) = besr_3band_stereo_db(&wet_ultra, &pred_exact_ultra);
    assert!(
        esr_exact_db < -85.0
            && besr_exact_db < -85.0
            && besr_exact_bands.iter().all(|&b| b < -85.0),
        "Oracle-01 exact f32 recovery must exceed -85 dB (ESR={esr_exact_db:.2}, bESR={besr_exact_db:.2}, bands={besr_exact_bands:?})"
    );

    // 3. End-to-end .dnam-proj fp32 capsule solve: Ultra bESR < -60 dB + strict 4-tier monotonicity
    let mut seed_model = unpack_v3_matryoshka_from_arena(&teacher_arena, false);
    for ro in &mut seed_model.readouts {
        clear_output_readout_weights(ro);
    }
    let mut seed_arena = vec![0u8; ARENA_ULTRA_BYTES];
    pack_v3_matryoshka_to_arena(&seed_model, &mut seed_arena);
    let seed_proj =
        build_dnam_proj_container_with_cp1(&mut manifest.clone(), &[], &seed_arena, &[]).unwrap();
    let wet_inter = interleave_lr(&wet_ultra);
    let inpc = build_proj_inpc_capsule(1, &[0x11; 256], &[0u8; 320], &dry_fp32).unwrap();
    let star_spec = make_vault_spec(
        "oracle01_star",
        true,
        BTreeMap::new(),
        &dry,
        &wet_inter,
        vec![stmc_win(0, 0, 0xFF, 0.0)],
        None,
    );
    let solved_proj = add_capture_with_options(
        &seed_proj,
        star_spec,
        Some(&inpc),
        SolverOptions::preserve_backbone(),
    )
    .unwrap();

    let dnam_ultra = export_runtime_dnam(&solved_proj, Some(Tier::Ultra)).unwrap();
    let dnam_pro = export_runtime_dnam(&solved_proj, Some(Tier::Pro)).unwrap();
    let dnam_full = export_runtime_dnam(&solved_proj, Some(Tier::Full)).unwrap();
    let dnam_lite = export_runtime_dnam(&solved_proj, Some(Tier::Lite)).unwrap();

    let crypto = DefaultCryptoProvider::default();
    let sol_view = DnamContainerView::from_bytes(&dnam_ultra, &crypto).unwrap();
    let untrimmed_arena = sol_view.arena_storage.as_bytes().to_vec();
    let untrimmed_dnam = pack_dnam(&manifest.base, &untrimmed_arena, &[]);
    let tier_esrs = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra].map(|tier| {
        let pred = render_stereo_dnam(&untrimmed_dnam, tier, &dry, &[]);
        esr_db_stereo(&wet_ultra, &pred)
    });
    assert!(
        tier_esrs[3] < -60.0 && tier_esrs.windows(2).all(|w| w[1] < w[0]),
        "Oracle-01 fp32 4-tier ESR must beat -60 dB and be monotone: {tier_esrs:?}"
    );
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let tier_dnams = [&dnam_lite, &dnam_full, &dnam_pro, &dnam_ultra];
    let mut tier_lufs = [0.0_f64; 4];
    let tier_besr_full: [(f64, [f64; 3]); 4] = std::array::from_fn(|i| {
        let pred = render_stereo_dnam(tier_dnams[i], tiers[i], &dry, &[]);
        tier_lufs[i] = compute_integrated_lufs(&pred[512..], 48_000);
        besr_3band_stereo_db(&wet_ultra, &pred)
    });
    let tier_besrs = tier_besr_full.map(|x| x.0);
    let ub = tier_besr_full[3].1;
    println!(
        "[CERT-ORACLE-01] f32={besr_exact_db:.3} {besr_exact_bands:.3?} | proj={tier_besrs:.3?} {ub:.3?} lufs={tier_lufs:.3?}"
    );
    assert!(
        tier_besrs[3] < -60.0 && tier_besrs.windows(2).all(|w| w[1] < w[0]),
        "Oracle-01 4-tier 3-band Bezout bESR must beat -60 dB and be monotone: {tier_besrs:?}"
    );

    // 4. 0.000000 ULP keyless prefix stripping invariance: Ultra -> Pro -> Full -> Lite
    let stripped_pro = strip_container_to_tier(&dnam_ultra, Tier::Pro).unwrap();
    let stripped_full = strip_container_to_tier(&stripped_pro, Tier::Full).unwrap();
    let stripped_lite = strip_container_to_tier(&stripped_full, Tier::Lite).unwrap();
    for (tier, stripped, direct) in [
        (Tier::Pro, &stripped_pro, &dnam_pro),
        (Tier::Full, &stripped_full, &dnam_full),
        (Tier::Lite, &stripped_lite, &dnam_lite),
    ] {
        let y_s = render_stereo_dnam(stripped, tier, &dry[..512], &[]);
        let y_d = render_stereo_dnam(direct, tier, &dry[..512], &[]);
        assert_0_ulp_stereo(&y_s, &y_d, &format!("prefix strip {tier:?}"));
    }

    // 5. Dyadic modal gauge (0.000000 ULP) and continuous SO(4) gauge (< 1e-5)
    let mut kappa = [1.0_f32; 16];
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = [0.5_f32, 2.0, 1.0][i % 3];
    }
    let q_ident: [[f32; 4]; 4] =
        std::array::from_fn(|r| std::array::from_fn(|c| if r == c { 1.0 } else { 0.0 }));
    let mut gauged_arena = teacher_arena.clone();
    apply_isometric_gauge_watermark(&mut gauged_arena, &kappa, &q_ident).unwrap();

    let h_orig = extract_structural_hash_224d(&teacher_arena).unwrap();
    let h_gaug = extract_structural_hash_224d(&gauged_arena).unwrap();
    assert!(1.0 - structural_cosine_similarity(&h_orig, &h_gaug) < 1.0e-12);

    let d_gaug = pack_dnam(&teacher_hdr, &gauged_arena, &[]);
    let y0 = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry[..256], &[]);
    let y1 = render_stereo_dnam(&d_gaug, Tier::Ultra, &dry[..256], &[]);
    assert_0_ulp_stereo(&y0, &y1, "dyadic modal isometric gauge");

    let (c, s) = (0.6_f32, 0.8_f32);
    let q_so4 = [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, c, -s],
        [0.0, 0.0, s, c],
    ];
    let mut so4_arena = teacher_arena.clone();
    apply_isometric_gauge_watermark(&mut so4_arena, &[1.0; 16], &q_so4).unwrap();
    let d_so4 = pack_dnam(&teacher_hdr, &so4_arena, &[]);
    let y_so4 = render_stereo_dnam(&d_so4, Tier::Ultra, &dry[..256], &[]);
    for (a, b) in y0.iter().zip(&y_so4) {
        assert!((a[0] - b[0]).abs() < 1.0e-5 && (a[1] - b[1]).abs() < 1.0e-5);
    }

    let mut fresh_arena = teacher_arena;
    let rep_a = run_stage_a_physics_probe(
        &manifest,
        &mut fresh_arena,
        &dry_lr[..2_048],
        &wet_ultra[..2_048],
        None,
        StageAConfig {
            preserve_existing_backbone: false,
            vp_gn_steps: 1,
        },
    );
    assert!(!rep_a.preserved_backbone && rep_a.kautz_poles_lifted == 16);
}
