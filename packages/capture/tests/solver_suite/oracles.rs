#![forbid(unsafe_code)]
//! Step 2.3 Structural Isomorphism Oracles (`Oracle-01`, `Oracle-02`, `Oracle-03`) for the
//! `v3` Deterministic Analytical + `VarPro` Solver (`dnam_capture::solver`).
//!
//! Canonical References:
//! - `docs/specs/training/webgpu_and_varpro_solver.md` (`r[train.pipeline.*]`, `r[train.gram.*]`)
//! - `docs/specs/containers/project_container_format.md` (`r[container.proj.*]`)

#[path = "oracles/cp1_and_vault.rs"]
mod cp1_and_vault;

use std::collections::BTreeMap;

use dnam_capture::CORESET_FP32_FRAMES_PER_STEM;
use dnam_capture::metrics::{evaluate_besr_from_band_energies, split_bezout_3band_energies};
use dnam_capture::proj::{
    DnamProjManifest, PROJ_STAT_BYTES_ULTRA, StemProvenance, StmcTrajectoryWindow,
    VaultCaptureSpec, add_capture_with_options, build_dnam_proj_container_with_dnam_bin,
    build_proj_inpc_capsule, export_runtime_dnam,
};
use dnam_capture::solver::{
    SolverOptions, StageAConfig, Wide192Accum, compute_integrated_lufs, run_stage_a_physics_probe,
    solve_stage_b_backbone, write_inline_cp1_if_fits,
};
use dnam_capture::stimulus::encode_coreset_fp32;
use dnam_engine::container::{
    build_dnam_container_from_dnam_bin, encode_cp1_shared_stream, extract_dnam_bin,
    strip_container_to_tier,
};
use dnam_engine::provenance::{
    apply_isometric_gauge_watermark, apply_sona_sign_gauge_watermark, extract_structural_hash_224d,
    structural_cosine_similarity,
};
use dnam_engine::schema::{ControlKind, ControlSchemaDef, DnamJsonHeader};
use dnam_engine::{DnamEngine, Tier};
use dnam_train::control::{
    ControlKindSpec, ControlSpec, StageStabilityConstraint, audit_phase1_controls,
};
use dnam_train::{
    Pass5EmitConfig, SonaModelF64, decode_dnam_bin_to_model, encode_model_to_dnam_bin,
};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0ControlBusWire, Arena0LiteWire, Arena1FullDeltaWire,
    Arena2ProDeltaWire, Arena3UltraDeltaWire, DnamBinBuildInput, DnamBinView, encode_dnam_bin,
};
use dnam_vm::control::Cp1ContributorSlot;
use dnam_vm::math::DefaultMathBackend;
use sha2::{Digest, Sha256};
use zerocopy::byteorder::little_endian::F32;
use zerocopy::{FromBytes, FromZeros, IntoBytes};

pub(super) fn clear_output_readout_weights(m: &mut SonaModelF64) {
    m.direct_s2_u_gain = [0.0; 2];
    m.direct_s2_hp_gain = [0.0; 2];
    for row in &mut m.w_ro {
        row[6..12].fill(0.0);
    }
    m.fir_l = [0.0; 128];
    m.fir_r = [0.0; 128];
    m.tier_overrides = [None, None, None, None];
    m.prefix_override_needed = [false; 4];
}

pub(super) fn pack_dnam_from_dbin(hdr: &DnamJsonHeader, dbin: &[u8]) -> Vec<u8> {
    let mut h = hdr.clone();
    build_dnam_container_from_dnam_bin(&mut h, dbin).unwrap()
}

fn reencode_dbin_with_arena(base_dbin: &[u8], arena_bytes: &[u8]) -> Vec<u8> {
    let view = DnamBinView::parse(base_dbin).unwrap();
    let retained_idx = view.retained_tier.index();
    let (a0, _) = Arena0LiteWire::ref_from_prefix(arena_bytes).unwrap();
    let a1 = (retained_idx >= 1).then(|| {
        Arena1FullDeltaWire::ref_from_prefix(&arena_bytes[ARENA_TIER_BYTES[0]..])
            .unwrap()
            .0
    });
    let a2 = (retained_idx >= 2).then(|| {
        Arena2ProDeltaWire::ref_from_prefix(&arena_bytes[ARENA_TIER_BYTES[1]..])
            .unwrap()
            .0
    });
    let a3 = (retained_idx >= 3).then(|| {
        Arena3UltraDeltaWire::ref_from_prefix(&arena_bytes[ARENA_TIER_BYTES[2]..])
            .unwrap()
            .0
    });
    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
        header_flags: view.header.flags.get(),
        ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: view.runtime_meta.ext_ring_lines,
        ext_ring_format: view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: view.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: view.control_descriptors,
        cp1_overflow: view.cp1_overflow,
        tsyn: view.tsyn_entries,
        cext: view.cext,
        arena0: a0,
        arena1: a1,
        arena2: a2,
        arena3: a3,
    })
    .unwrap()
}

pub(super) fn inject_controls_and_cp1_into_dbin(
    base_dbin: &[u8],
    manifest: &DnamProjManifest,
    slots: &[Cp1ContributorSlot],
    mutate_cb: impl FnOnce(&mut Arena0ControlBusWire),
) -> Vec<u8> {
    let view = DnamBinView::parse(base_dbin).unwrap();
    let mut a0 = *view.arena0;
    let a1 = view.arena1.copied();
    let a2 = view.arena2.copied();
    let a3 = view.arena3.copied();
    mutate_cb(&mut a0.control_bus_lite);

    let specs: Vec<ControlSpec> = manifest
        .base
        .controls_schema
        .iter()
        .map(|ctrl| match ctrl.kind {
            ControlKind::Range => {
                let decl = ctrl.declared_range.unwrap_or([0.0, 1.0]);
                let lo = f64::from(decl[0]);
                let hi = f64::from(decl[1].max(decl[0]));
                let def_val = ctrl.default.as_f64().unwrap_or(lo).clamp(lo, hi);
                ControlSpec {
                    kind: ControlKindSpec::Range {
                        declared_min: lo,
                        declared_max: hi,
                        default_val: def_val,
                        obs_min: lo,
                        obs_max: hi,
                        unique_values: if (hi - lo).abs() > 1.0e-6 { 3 } else { 1 },
                        lobatto_anchors: [lo as f32, def_val as f32, hi as f32],
                    },
                }
            }
            ControlKind::Switch => {
                let opts = ctrl.declared_options.as_deref().unwrap_or(&[]);
                let n_opts = opts.len().clamp(1, 16) as u8;
                let def_str = ctrl.default.as_str().unwrap_or("");
                let def_opt = opts.iter().position(|o| o == def_str).unwrap_or(0) as u8;
                let mask = if n_opts >= 16 {
                    u16::MAX
                } else {
                    (1_u16 << n_opts) - 1
                };
                ControlSpec {
                    kind: ControlKindSpec::Switch {
                        num_options: n_opts,
                        default_opt: def_opt,
                        declared_mask: mask,
                        observed_mask: mask,
                    },
                }
            }
        })
        .collect();

    let phase1 = audit_phase1_controls(&specs, &StageStabilityConstraint::default()).unwrap();

    let mut cp1_overflow = Vec::new();
    if slots.is_empty() {
        a0.control_bus_lite.cp1_dir = ZeroedHelper::zeroed_cp1_dir();
        a0.control_bus_lite.cp1_base.fill(0);
    } else {
        write_inline_cp1_if_fits(a0.as_mut_bytes(), slots);
        if usize::from(a0.control_bus_lite.cp1_dir.cp1_dir_count.get()) != slots.len() {
            a0.control_bus_lite.cp1_dir = ZeroedHelper::zeroed_cp1_dir();
            a0.control_bus_lite.cp1_base.fill(0);
            cp1_overflow = encode_cp1_shared_stream(slots).unwrap();
        }
    }

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
        header_flags: view.header.flags.get(),
        ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: view.runtime_meta.ext_ring_lines,
        ext_ring_format: view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: view.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: &phase1.wire_descriptors,
        cp1_overflow: &cp1_overflow,
        tsyn: view.tsyn_entries,
        cext: view.cext,
        arena0: &a0,
        arena1: a1.as_ref(),
        arena2: a2.as_ref(),
        arena3: a3.as_ref(),
    })
    .unwrap()
}

struct ZeroedHelper;
impl ZeroedHelper {
    #[inline]
    fn zeroed_cp1_dir() -> dnam_vm::bin::Cp1InlineDirWire {
        dnam_vm::bin::Cp1InlineDirWire::new_zeroed()
    }
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

/// Builds a ground-truth `SONA v3.0` `.dnam-bin` with active `CoreQuad` blocks (`0..10`),
/// inter-block `tiles`, `32` shared Kautz poles, `Op-T` coprime delays, and nested 4-tier stereo
/// readouts whose progressive capacity yields strictly monotone `Lite..Ultra` accuracy.
pub(super) fn build_oracle01_teacher_dbin() -> Vec<u8> {
    let mut model = SonaModelF64::new(48_000.0);
    model.oversample_upper_tiers_2x = false;
    model.active_fir_taps = 128;
    model.active_kautz_pairs = 32;

    for (q_idx, q) in model.core_quads.iter_mut().enumerate() {
        let (blk, local) = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q_idx];
        for lane in 0..4 {
            let phase = (((blk * 7 + local * 5 + lane * 3 + 1) % 11) as f64 - 5.0) / 5.0;
            q.svf_g[lane] = 0.08 + 0.04 * ((lane + blk) as f64);
            q.svf_k[lane] = 0.65 + 0.10 * (lane as f64);
            q.tone_lp[lane] = 0.35 + 0.08 * phase;
            q.tone_bp[lane] = 0.15 * phase;
            q.tone_hp[lane] = 0.10 * (-phase);
            q.tone_dir[lane] = 0.35;
            q.in_mid_weight[lane] = 0.32 + 0.05 * phase;
            q.in_side_weight[lane] = 0.14 * (-phase);
            q.adaa_drive[lane] = 0.90 + 0.15 * phase;
            q.lru_alpha_rel[lane] = 0.985 - 0.002 * (lane as f64);
            q.lru_kappa[lane] = 0.08 * phase.abs();
            q.lru_bias_drive[lane] = 0.12 * phase;
        }
    }
    for (t_idx, tile) in model.tiles.iter_mut().enumerate() {
        for s in 0..4 {
            for d in 0..4 {
                let phase = (((t_idx * 5 + s * 3 + d + 1) % 7) as f64 - 3.0) / 3.0;
                tile.w_odd_4x4[s][d] = 0.03 * phase;
                tile.w_even_4x4[s][d] = 0.02 * (-phase);
                tile.w_lru_4x4[s][d] = 0.015 * phase;
            }
        }
    }

    let kautz_scales = [0.36_f64, 0.16, 0.07, 0.025];
    for (m, kp) in model.kautz.iter_mut().enumerate() {
        let tier_idx = if m < 8 {
            0
        } else if m < 12 {
            1
        } else if m < 24 {
            2
        } else {
            3
        };
        let sc = kautz_scales[tier_idx];
        let sign = if m % 2 == 0 { 1.0 } else { -0.8 };
        kp.r_low = 0.86 + 0.003 * (m as f64);
        kp.dr = 0.001;
        kp.theta = 0.08 + 0.085 * (m as f64);
        kp.b_gamma = 1.0;
        kp.c_re_l = sc * 0.18 * sign;
        kp.c_im_l = -sc * 0.11 * sign;
        kp.c_re_r = sc * 0.15 * sign;
        kp.c_im_r = sc * 0.12 * sign;
    }

    let coprime_delays: [[f64; 4]; 4] = [
        [29.0, 43.0, 59.0, 73.0],
        [101.0, 137.0, 173.0, 199.0],
        [251.0, 293.0, 331.0, 373.0],
        [419.0, 467.0, 509.0, 547.0],
    ];
    for t in 0..4 {
        model.transport_quads[t].delay_samples = coprime_delays[t];
        model.transport_quads[t].g_loop = 0.25;
        model.transport_quads[t].input_gains = [0.0; 4];
        model.transport_quads[t].wet_tap_l = [0.0; 4];
        model.transport_quads[t].wet_tap_r = [0.0; 4];
        model.transport_quads[t].ext_f32 = false;
    }
    model.tank_dry_mix = 1.0;

    model.pre_conv_u_gain = [0.75, 0.72];
    model.direct_s2_u_gain = [0.22, -0.18];
    model.direct_s2_hp_gain = [0.06, 0.06];

    let fir_ranges = [0..32_usize, 32..48, 48..96, 96..128];
    let lane_ranges = [0..16_usize, 16..24, 24..32, 32..40];
    let tier_scales = [0.36_f64, 0.16, 0.07, 0.025];
    for (t_idx, &sc) in tier_scales.iter().enumerate() {
        for idx in fir_ranges[t_idx].clone() {
            let vl = sc * 0.45 * ((((idx * 11 + t_idx * 5 + 3) % 13) as f64) - 6.0) / 6.0;
            let vr = sc * 0.45 * ((((idx * 7 + t_idx * 9 + 2) % 13) as f64) - 6.0) / 6.0;
            model.fir_l[idx] = vl;
            model.fir_r[idx] = vr;
        }
        for l in lane_ranges[t_idx].clone() {
            for c in 0..6 {
                let phase = (((l * 5 + c * 3 + t_idx) % 7) as f64 - 3.0) / 3.0;
                if t_idx == 0 {
                    model.w_ro[l][c] = 0.040 * phase;
                }
                model.w_ro[l][6 + c] = sc * 0.15 * phase;
            }
        }
    }

    encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap()
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
        .map(|&x| eng.step_frame_sona(x, x, false))
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
    // verify[train.pipeline.backbone-trajectory-opt]
    // verify[train.pipeline.nested-varpro-schur]
    // verify[train.pipeline.isomorphism-parity]
    // verify[train.gram.accuracy-contract]
    // verify[train.gram.wide192]
    let teacher_dbin = build_oracle01_teacher_dbin();
    let mut teacher_hdr = DnamJsonHeader::default_reference();
    teacher_hdr.io_config.output_mode = "stereo".to_string();
    let teacher_dnam = pack_dnam_from_dbin(&teacher_hdr, &teacher_dbin);

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
    let mut student_model = decode_dnam_bin_to_model(&teacher_dbin).unwrap();
    clear_output_readout_weights(&mut student_model);
    let mut student_dbin =
        encode_model_to_dnam_bin(&student_model, Pass5EmitConfig::default()).unwrap();

    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.output_mode = "stereo".to_string();
    let stage_b = solve_stage_b_backbone(
        &manifest,
        &mut student_dbin,
        &dry_lr,
        &wet_ultra,
        0,
        1,
        true,
    )
    .expect("Stage B SONA v3 backbone solve");
    assert_eq!(stage_b.stat_bytes.len(), PROJ_STAT_BYTES_ULTRA);
    let exact_student_dnam = pack_dnam_from_dbin(&manifest.base, &student_dbin);
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
    let mut seed_model = decode_dnam_bin_to_model(&teacher_dbin).unwrap();
    clear_output_readout_weights(&mut seed_model);
    let seed_dbin = encode_model_to_dnam_bin(&seed_model, Pass5EmitConfig::default()).unwrap();
    let seed_proj =
        build_dnam_proj_container_with_dnam_bin(&mut manifest.clone(), &[], &seed_dbin).unwrap();
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

    let sol_dbin = extract_dnam_bin(&dnam_ultra).unwrap();
    let sol_view = DnamBinView::parse(sol_dbin).unwrap();
    let mut untrimmed_a0 = *sol_view.arena0;
    untrimmed_a0.crossbar.tier_gain_trim = [F32::new(1.0); 4];
    let untrimmed_dbin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: sol_view.header.reference_sample_rate_hz.get(),
        header_flags: sol_view.header.flags.get(),
        ext_ring_pow2_frames: sol_view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: sol_view.runtime_meta.ext_ring_lines,
        ext_ring_format: sol_view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: sol_view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: sol_view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: sol_view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: sol_view.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: sol_view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: sol_view.runtime_meta.content_hash_prefix,
        controls: sol_view.control_descriptors,
        cp1_overflow: sol_view.cp1_overflow,
        tsyn: sol_view.tsyn_entries,
        cext: sol_view.cext,
        arena0: &untrimmed_a0,
        arena1: sol_view.arena1,
        arena2: sol_view.arena2,
        arena3: sol_view.arena3,
    })
    .unwrap();
    let untrimmed_dnam = pack_dnam_from_dbin(&manifest.base, &untrimmed_dbin);
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

    // 5. Dyadic modal gauge (0.000000 ULP), SONA 40-lane sign gauge (0.000000 ULP), and SO(4) gauge
    let arena0_off = DnamBinView::parse(&teacher_dbin)
        .unwrap()
        .header
        .arena0_offset
        .get() as usize;
    let mut kappa = [1.0_f32; 32];
    for (i, k) in kappa.iter_mut().enumerate() {
        *k = [0.5_f32, 2.0, 1.0][i % 3];
    }
    let q_ident: [[f32; 4]; 4] =
        std::array::from_fn(|r| std::array::from_fn(|c| if r == c { 1.0 } else { 0.0 }));
    let mut gauged_arena = teacher_dbin[arena0_off..].to_vec();
    apply_isometric_gauge_watermark(&mut gauged_arena, &kappa, &q_ident).unwrap();

    let h_orig = extract_structural_hash_224d(&teacher_dbin[arena0_off..]).unwrap();
    let h_gaug = extract_structural_hash_224d(&gauged_arena).unwrap();
    assert!(1.0 - structural_cosine_similarity(&h_orig, &h_gaug) < 1.0e-12);

    let gauged_dbin = reencode_dbin_with_arena(&teacher_dbin, &gauged_arena);
    let d_gaug = pack_dnam_from_dbin(&teacher_hdr, &gauged_dbin);
    let y0 = render_stereo_dnam(&teacher_dnam, Tier::Ultra, &dry[..256], &[]);
    let y1 = render_stereo_dnam(&d_gaug, Tier::Ultra, &dry[..256], &[]);
    assert_0_ulp_stereo(&y0, &y1, "dyadic modal isometric gauge");

    let signs_40: [f32; 40] = std::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
    let mut sign_arena = teacher_dbin[arena0_off..].to_vec();
    apply_sona_sign_gauge_watermark(&mut sign_arena, &signs_40).unwrap();
    let sign_dbin = reencode_dbin_with_arena(&teacher_dbin, &sign_arena);
    let d_sign = pack_dnam_from_dbin(&teacher_hdr, &sign_dbin);
    let y_sign = render_stereo_dnam(&d_sign, Tier::Ultra, &dry[..256], &[]);
    assert_0_ulp_stereo(&y0, &y_sign, "SONA 40-lane sign gauge");

    let (c, s) = (0.6_f32, 0.8_f32);
    let q_so4 = [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, c, -s],
        [0.0, 0.0, s, c],
    ];
    let mut so4_arena = teacher_dbin[arena0_off..].to_vec();
    apply_isometric_gauge_watermark(&mut so4_arena, &[1.0; 32], &q_so4).unwrap();
    let so4_dbin = reencode_dbin_with_arena(&teacher_dbin, &so4_arena);
    let d_so4 = pack_dnam_from_dbin(&teacher_hdr, &so4_dbin);
    let y_so4 = render_stereo_dnam(&d_so4, Tier::Ultra, &dry[..256], &[]);
    for (a, b) in y0.iter().zip(&y_so4) {
        assert!((a[0] - b[0]).abs() < 1.0e-5 && (a[1] - b[1]).abs() < 1.0e-5);
    }

    let mut fresh_dbin = teacher_dbin;
    let rep_a = run_stage_a_physics_probe(
        &manifest,
        &mut fresh_dbin,
        &dry_lr[..2_048],
        &wet_ultra[..2_048],
        None,
        StageAConfig {
            preserve_existing_backbone: false,
            vp_gn_steps: 1,
        },
    );
    assert!(
        !rep_a.preserved_backbone && rep_a.kautz_poles_lifted > 0 && rep_a.kautz_poles_lifted <= 32
    );
}
