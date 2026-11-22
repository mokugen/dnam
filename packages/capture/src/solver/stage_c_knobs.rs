#![forbid(unsafe_code)]
//! Stage C (`Parallel Multi-Knob & Switch Conditioning`: `CP1` + `"CEXT"` + `W_ctrl` + `U_int`,
//! `solver::stage_c_knobs`).
//!
//! Canonical References:
//! - `docs/specs/training/webgpu_and_varpro_solver.md` (`ref[train.pipeline.control-projection-fit]`)
//! - `docs/specs/dsp/parametric_control_bus.md` (`ref[dsp.ctrl.accum.cp1-scaling]`, `ref[dsp.ctrl.accum.anova-fm]`)

mod collect;
mod point_gn;
mod sweep_windows;

use std::collections::{BTreeMap, BTreeSet};

use dnam_engine::schema::{ControlKind, DnamError};
use dnam_train::control::{
    ControlKindSpec, ControlSpec, Cp1FitConfig, Cp1WireBundle, FourBandTptBank,
    StageStabilityConstraint, audit_phase1_controls, enforce_arena0_stability_invariants,
    fit_and_pack_cp1_audited,
};
use dnam_vm::Tier;
use dnam_vm::bin::{DnamBinBuildInput, DnamBinView, encode_dnam_bin, strip_dnam_bin_to_tier};
use dnam_vm::control::cp1::decode_cp1_contributors;
use dnam_vm::control::{Cp1ContributorSlot, MAX_EXACT_CONTRIBUTORS};
use zerocopy::byteorder::little_endian::F32;
use zerocopy::{FromBytes, FromZeros};

use self::collect::{collect_stage_c_points, ensure_star_center_anchor};
use self::point_gn::{PointSolveInputs, solve_point_cp1_sample};
use crate::proj::capsules::StmcTrajectoryWindow;
use crate::proj::schema::DnamProjManifest;
use crate::proj::verify_dnam_bin_zero_dc;
use crate::stimulus::sealing::CORESET_A5_FRAMES_48K;

const PASS4_ACTIVE_SLOTS: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 10, 11, 13, 15, 22, 26, 30];
pub(super) const A5_STATIONARY_FIT_LEN: usize = 4_096;
pub(super) const A5_ACTIVE_BURST_END: usize =
    (3 * CORESET_A5_FRAMES_48K) / 4 + ((CORESET_A5_FRAMES_48K / 4) * 55) / 100;
pub(super) const A5_STATIONARY_OFFSET: usize = A5_ACTIVE_BURST_END - A5_STATIONARY_FIT_LEN;
const _: () = assert!(A5_STATIONARY_OFFSET >= CORESET_A5_FRAMES_48K / 2);
const _: () = assert!(A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN == A5_ACTIVE_BURST_END);

/// Output from Stage C multi-knob / switch projection (`ref[train.pipeline.control-projection-fit]`).
#[derive(Debug, Clone)]
pub struct StageCKnobsOutput {
    /// Wall-clock Stage C duration in milliseconds.
    pub elapsed_ms: f32,
    /// Ordered `CP1` contributor records written inline or in the `"CP1\0"` overflow stream.
    pub cp1_slots: Vec<Cp1ContributorSlot>,
}

#[inline]
fn now_ms_token() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}

#[inline]
fn elapsed_ms_since(token: Option<std::time::Instant>) -> f32 {
    token.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1_000.0)
}

#[inline]
fn map_stage_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

/// Returns the default `StarCenter` (`s = 0`) physical/index control vector `[0..P]` from `manifest`.
#[must_use]
pub fn star_center_control_values(manifest: &DnamProjManifest) -> Vec<f32> {
    manifest
        .base
        .controls_schema
        .iter()
        .map(|ctrl| match ctrl.kind {
            ControlKind::Range => ctrl.default.as_f64().unwrap_or(0.0) as f32,
            ControlKind::Switch => {
                let def_str = ctrl.default.as_str().unwrap_or("");
                let idx = ctrl
                    .declared_options
                    .as_ref()
                    .and_then(|opts| opts.iter().position(|o| o == def_str))
                    .unwrap_or(0);
                idx as f32
            }
        })
        .collect()
}

/// Decodes little-endian `fp32` (`<f4`) interleaved PCM bytes (`INPC` or `STMC`) into `[L, R]` frames.
#[must_use]
pub fn decode_coreset_lr(bytes_fp32: &[u8], channels: usize) -> Vec<[f32; 2]> {
    let ch = channels.max(1);
    let bytes_per_frame = ch * 4;
    let n_frames = bytes_fp32.len() / bytes_per_frame;
    let raw = &bytes_fp32[..n_frames * bytes_per_frame];
    let Ok(words) = <[F32]>::ref_from_bytes(raw) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let base = f * ch;
        let l = words[base].get();
        let r = if ch >= 2 { words[base + 1].get() } else { l };
        out.push([
            if l.is_finite() { l } else { 0.0 },
            if r.is_finite() { r } else { 0.0 },
        ]);
    }
    out
}

pub(super) fn parse_stem_control_values(
    manifest: &DnamProjManifest,
    controls_meta: &BTreeMap<String, serde_json::Value>,
    base_ctrls: &[f32],
) -> Vec<f32> {
    let mut out = base_ctrls.to_vec();
    for (p, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        let Some(val) = controls_meta.get(&ctrl.id) else {
            continue;
        };
        match ctrl.kind {
            ControlKind::Range => {
                if let Some(v) = val.as_f64()
                    && v.is_finite()
                {
                    out[p] = v as f32;
                }
            }
            ControlKind::Switch => {
                if let Some(opt_name) = val.as_str()
                    && let Some(idx) = ctrl
                        .declared_options
                        .as_ref()
                        .and_then(|opts| opts.iter().position(|o| o == opt_name))
                {
                    out[p] = idx as f32;
                } else if let Some(v) = val.as_f64()
                    && v.is_finite()
                {
                    out[p] = v.round().max(0.0) as f32;
                }
            }
        }
    }
    out
}

fn build_control_specs_from_observations(
    manifest: &DnamProjManifest,
    base_ctrls: &[f32],
    observed_points: &[Vec<f32>],
    observed_windows: &[&[StmcTrajectoryWindow]],
) -> Vec<ControlSpec> {
    let has_star = manifest
        .stems_registry
        .iter()
        .any(|s| s.enabled && s.is_star_center);
    manifest
        .base
        .controls_schema
        .iter()
        .enumerate()
        .map(|(p, ctrl)| match ctrl.kind {
            ControlKind::Range => {
                let decl = ctrl.declared_range.unwrap_or([0.0, 1.0]);
                let d_min = f64::from(decl[0]);
                let d_max = f64::from(decl[1].max(decl[0]));
                let def_val =
                    f64::from(base_ctrls.get(p).copied().unwrap_or(decl[0])).clamp(d_min, d_max);

                let mut vals = Vec::new();
                if has_star {
                    vals.push(def_val);
                }
                for entry in manifest.stems_registry.iter().filter(|s| s.enabled) {
                    if let Some(v) = entry.controls_meta.get(&ctrl.id).and_then(|j| j.as_f64())
                        && v.is_finite()
                    {
                        vals.push(v.clamp(d_min, d_max));
                    }
                }
                for pt in observed_points {
                    if let Some(&v) = pt.get(p)
                        && v.is_finite()
                    {
                        vals.push(f64::from(v).clamp(d_min, d_max));
                    }
                }
                for wins in observed_windows {
                    for w in *wins {
                        if usize::from(w.control_id) == p
                            && w.ctrl_kind == 0
                            && w.window_weight > 0.0
                            && w.u_phys.is_finite()
                        {
                            vals.push(f64::from(w.u_phys).clamp(d_min, d_max));
                        }
                    }
                }

                let (obs_min, obs_max, unique_values) = if vals.is_empty() {
                    (def_val, def_val, 1)
                } else {
                    vals.sort_by(f64::total_cmp);
                    let mut uniq = 1_usize;
                    for pair in vals.windows(2) {
                        if (pair[1] - pair[0]).abs() > 1.0e-4 {
                            uniq += 1;
                        }
                    }
                    (vals[0], *vals.last().unwrap_or(&vals[0]), uniq)
                };
                ControlSpec {
                    kind: ControlKindSpec::Range {
                        declared_min: d_min,
                        declared_max: d_max,
                        default_val: def_val,
                        obs_min,
                        obs_max,
                        unique_values,
                        lobatto_anchors: [
                            obs_min as f32,
                            def_val.clamp(obs_min, obs_max) as f32,
                            obs_max as f32,
                        ],
                    },
                }
            }
            ControlKind::Switch => {
                let n_opts = ctrl
                    .declared_options
                    .as_ref()
                    .map_or(1, Vec::len)
                    .clamp(1, 16) as u8;
                let def_opt = base_ctrls
                    .get(p)
                    .copied()
                    .unwrap_or(0.0)
                    .round()
                    .clamp(0.0, f32::from(n_opts.saturating_sub(1)))
                    as u8;
                let declared_mask = if n_opts >= 16 {
                    u16::MAX
                } else {
                    (1_u16 << n_opts) - 1
                };
                let mut obs_set = BTreeSet::new();
                if has_star {
                    obs_set.insert(def_opt);
                }
                for pt in observed_points {
                    if let Some(&v) = pt.get(p)
                        && v.is_finite()
                    {
                        let idx = (v.round().max(0.0) as u8).min(n_opts.saturating_sub(1));
                        obs_set.insert(idx);
                    }
                }
                for entry in manifest.stems_registry.iter().filter(|s| s.enabled) {
                    if let Some(opt_str) =
                        entry.controls_meta.get(&ctrl.id).and_then(|v| v.as_str())
                        && let Some(idx) = ctrl
                            .declared_options
                            .as_ref()
                            .and_then(|opts| opts.iter().position(|o| o == opt_str))
                    {
                        obs_set.insert(idx as u8);
                    }
                }
                for wins in observed_windows {
                    for w in *wins {
                        if usize::from(w.control_id) == p
                            && w.ctrl_kind == 1
                            && w.window_weight > 0.0
                            && usize::from(w.active_opt) < usize::from(n_opts)
                        {
                            obs_set.insert(w.active_opt);
                        }
                    }
                }
                let mut observed_mask = 0_u16;
                for idx in obs_set {
                    observed_mask |= 1_u16 << idx;
                }
                if observed_mask == 0 {
                    observed_mask = 1_u16 << def_opt;
                }
                ControlSpec {
                    kind: ControlKindSpec::Switch {
                        num_options: n_opts,
                        default_opt: def_opt,
                        declared_mask,
                        observed_mask,
                    },
                }
            }
        })
        .collect()
}

fn patch_dnam_bin_control_bus(
    dnam_bin: &mut Vec<u8>,
    phase1: Option<&dnam_train::control::Phase1ClampOutput>,
    bundle_opt: Option<&Cp1WireBundle>,
    clear_dynamic_w_ctrl: bool,
) -> Result<Vec<Cp1ContributorSlot>, DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let mut a0 = *view.arena0;
    let a1 = view.arena1.copied();
    let a2 = view.arena2.copied();
    let mut a3 = view.arena3.copied();
    let mut ro_pole = view.runtime_meta.ro_pole_48k.get();

    if clear_dynamic_w_ctrl {
        a0.control_bus_lite.u_int_base = [[F32::new(0.0); 8]; 4];
        a0.control_bus_lite.w_ctrl = [[F32::new(0.0); 16]; 16];
        a0.control_bus_lite.cp1_dir = ZeroedHelper::zeroed_cp1_dir();
        a0.control_bus_lite.cp1_base.fill(0);
        if let Some(ref mut a3w) = a3 {
            a3w.control_bus_ultra_delta.w_ctrl_ultra = [[F32::new(0.0); 4]; 16];
            a3w.control_bus_ultra_delta.u_int_ultra = [[[F32::new(0.0); 8]; 4]; 3];
            a3w.control_bus_ultra_delta.cp1_ultra.fill(0);
        }
    }

    if let Some(bundle) = bundle_opt {
        a0.control_bus_lite.trust_radius_mh = bundle.arena0_cb.trust_radius_mh;
        a0.control_bus_lite.cp1_dir = bundle.arena0_cb.cp1_dir;
        a0.control_bus_lite.u_int_base = bundle.arena0_cb.u_int_base;
        a0.control_bus_lite.cp1_base = bundle.arena0_cb.cp1_base;
        for r in 0..16 {
            for c in 0..8 {
                a0.control_bus_lite.w_ctrl[r][c] = bundle.arena0_cb.w_ctrl[r][c];
            }
        }
        if let Some(ref mut a3w) = a3 {
            a3w.control_bus_ultra_delta.w_ctrl_ultra = bundle.arena3_cb.w_ctrl_ultra;
            let has_non_zero_ultra_uint = bundle
                .arena3_cb
                .u_int_ultra
                .iter()
                .flatten()
                .flatten()
                .any(|v| v.get() != 0.0);
            if clear_dynamic_w_ctrl || has_non_zero_ultra_uint {
                a3w.control_bus_ultra_delta.u_int_ultra = bundle.arena3_cb.u_int_ultra;
            }
            a3w.control_bus_ultra_delta.cp1_ultra = bundle.arena3_cb.cp1_ultra;
        }
    }

    enforce_arena0_stability_invariants(&mut a0, &mut ro_pole);

    let wire_descs: &[dnam_vm::bin::ControlClampDescriptorWire] =
        phase1.map_or(view.control_descriptors, |p| p.wire_descriptors.as_slice());
    let cp1_overflow = bundle_opt.map_or(&[][..], |b| b.cp1_overflow.as_slice());
    let tsyn = bundle_opt.map_or(view.tsyn_entries, |b| b.tsyn.as_slice());
    let cext = if clear_dynamic_w_ctrl && bundle_opt.is_none() {
        None
    } else {
        bundle_opt.and_then(|b| b.cext.as_ref()).or(view.cext)
    };

    let reencoded = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
        header_flags: view.header.flags.get(),
        ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: view.runtime_meta.ext_ring_lines,
        ext_ring_format: view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: ro_pole,
        switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: wire_descs,
        cp1_overflow,
        tsyn,
        cext,
        arena0: &a0,
        arena1: a1.as_ref(),
        arena2: a2.as_ref(),
        arena3: a3.as_ref(),
    })
    .map_err(map_stage_err)?;
    *dnam_bin = reencoded;
    verify_dnam_bin_zero_dc(dnam_bin)?;

    let post = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let n_ctrl = usize::from(post.header.num_controls.get());
    let mut decoded = vec![Cp1ContributorSlot::zero(); MAX_EXACT_CONTRIBUTORS];
    let count = decode_cp1_contributors(
        &post.arena0.control_bus_lite,
        post.arena3.map(|w| &w.control_bus_ultra_delta),
        post.cp1_overflow,
        n_ctrl,
        &mut decoded,
    )
    .map_err(map_stage_err)?;
    decoded.truncate(count);
    Ok(decoded)
}

struct ZeroedHelper;
impl ZeroedHelper {
    #[inline]
    fn zeroed_cp1_dir() -> dnam_vm::bin::Cp1InlineDirWire {
        dnam_vm::bin::Cp1InlineDirWire::new_zeroed()
    }
}

/// Runs Stage C (`Parallel Multi-Knob & Switch Conditioning`), solving `CP1`, `"CEXT"`, `U_int`,
/// and `W_ctrl` while guaranteeing bit-exact `0-ULP` invariance at `StarCenter` (`REQ-4.2`).
pub fn solve_stage_c_knobs(
    manifest: &DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    dry_lr: &[[f32; 2]],
    active_frames_hint: usize,
    clear_dynamic_w_ctrl: bool,
) -> Result<StageCKnobsOutput, DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(|_| DnamError::BufferTooSmall {
        expected: 128,
        actual: dnam_bin.len(),
    })?;
    let max_tier = view.retained_tier;
    let timer = now_ms_token();

    if manifest.base.controls_schema.is_empty() {
        let slots = patch_dnam_bin_control_bus(dnam_bin, None, None, clear_dynamic_w_ctrl)?;
        return Ok(StageCKnobsOutput {
            elapsed_ms: elapsed_ms_since(timer),
            cp1_slots: slots,
        });
    }

    let base_ctrls = star_center_control_values(manifest);
    let collected = collect_stage_c_points(
        manifest,
        stmc_by_id,
        dry_lr,
        &base_ctrls,
        active_frames_hint,
    );
    let win_refs: Vec<&[StmcTrajectoryWindow]> =
        collected.parsed_windows.iter().map(Vec::as_slice).collect();
    let specs = build_control_specs_from_observations(
        manifest,
        &base_ctrls,
        &collected.observed_points,
        &win_refs,
    );
    let phase1 = audit_phase1_controls(&specs, &StageStabilityConstraint::default())
        .map_err(map_stage_err)?;

    if collected.pending.is_empty() || dry_lr.is_empty() {
        let slots =
            patch_dnam_bin_control_bus(dnam_bin, Some(&phase1), None, clear_dynamic_w_ctrl)?;
        return Ok(StageCKnobsOutput {
            elapsed_ms: elapsed_ms_since(timer),
            cp1_slots: slots,
        });
    }

    let ultra_slice = strip_dnam_bin_to_tier(dnam_bin, max_tier).map_err(map_stage_err)?;
    let lite_slice = strip_dnam_bin_to_tier(dnam_bin, Tier::Lite).map_err(map_stage_err)?;
    let bank = FourBandTptBank::new(48_000.0);
    let mono_out = manifest.base.io_config.output_mode == "mono";

    let mut samples = Vec::with_capacity(collected.pending.len() + 1);
    for pt in collected.pending {
        let dry_slice = &dry_lr[pt.dry_off..pt.dry_off + pt.len];
        let sample = solve_point_cp1_sample(PointSolveInputs {
            dnam_bin,
            ultra_slice,
            lite_slice,
            max_tier,
            mono_out,
            dry_win: dry_slice,
            wet_win: &pt.wet_lr,
            phys_controls: pt.phys_controls,
            specs: &specs,
            bank: &bank,
        })?;
        samples.push(sample);
    }
    ensure_star_center_anchor(manifest, &base_ctrls, &specs, &mut samples);

    let bundle_opt = if samples.is_empty() {
        None
    } else {
        let cfg = Cp1FitConfig {
            fit_ultra: max_tier == Tier::Ultra,
            ..Default::default()
        };
        Some(
            fit_and_pack_cp1_audited(&specs, &phase1, &samples, view.tsyn_entries, cfg)
                .map_err(map_stage_err)?,
        )
    };

    let slots = patch_dnam_bin_control_bus(
        dnam_bin,
        Some(&phase1),
        bundle_opt.as_ref(),
        clear_dynamic_w_ctrl,
    )?;

    Ok(StageCKnobsOutput {
        elapsed_ms: elapsed_ms_since(timer),
        cp1_slots: slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::capsules::build_proj_stmc_capsule;
    use crate::proj::schema::StemProvenance;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;
    use crate::proj::{STMC_FLAG_ENABLED, STMC_FLAG_STAR_CENTER};
    use crate::solver::stage_b_backbone::solve_stage_b_backbone;
    use dnam_engine::container::build_dnam_container_from_dnam_bin;
    use dnam_engine::{DnamEngine, Tier};
    use dnam_vm::math::DefaultMathBackend;
    use std::f64::consts::TAU;

    #[test]
    fn test_stage_c_knobs_zero_ulp_star_invariance_and_cp1_projection() {
        let mut manifest = DnamProjManifest::default_reference();
        manifest.base.controls_schema = vec![
            serde_json::from_value(serde_json::json!({
                "id": "gain",
                "name": "Gain",
                "type": "range",
                "lobatto_degree": 3,
                "declared_range": [-1.0, 1.0],
                "safe_clamped_range": [-1.0, 1.0],
                "default": 0.0
            }))
            .unwrap(),
            serde_json::from_value(serde_json::json!({
                "id": "mode",
                "name": "Mode",
                "type": "switch",
                "declared_options": ["clean", "boost"],
                "active_options": ["clean", "boost"],
                "default": "clean"
            }))
            .unwrap(),
        ];
        assert_eq!(star_center_control_values(&manifest), vec![0.0, 0.0]);

        let raw_fp32: Vec<u8> = [0.25_f32, -0.5_f32, f32::NAN, 0.75_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let dec_s = decode_coreset_lr(&raw_fp32, 2);
        assert_eq!(dec_s, vec![[0.25, -0.5], [0.0, 0.75]]);

        let n = 1_024usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.28 * (TAU * 220.0 * (i as f64) / 48_000.0).sin() as f32;
                [s, s]
            })
            .collect();
        let wet_star: Vec<[f32; 2]> = dry.iter().map(|&[l, r]| [0.8 * l, 0.8 * r]).collect();
        let wet_gain_hi: Vec<[f32; 2]> = dry
            .iter()
            .map(|&[l, r]| [(1.4 * l).tanh(), (1.4 * r).tanh()])
            .collect();
        let wet_boost: Vec<[f32; 2]> = dry
            .iter()
            .map(|&[l, r]| [(2.0 * l).tanh(), (2.0 * r).tanh()])
            .collect();

        let mut dnam_bin = initialize_backbone_dnam_bin_if_needed(None).expect("init dnam-bin");
        let _sb = solve_stage_b_backbone(&manifest, &mut dnam_bin, &dry, &wet_star, n, 2, false)
            .expect("stage b should succeed");

        let mut stmc_by_id = BTreeMap::new();
        let cap_n = crate::CORESET_FP32_FRAMES_PER_STEM;
        let mk_stmc = |idx: u16, wet: &[[f32; 2]]| -> Vec<u8> {
            let mut bytes = vec![0u8; cap_n * 4];
            for t in 0..wet.len().min(cap_n) {
                bytes[t * 4..t * 4 + 4].copy_from_slice(&wet[t][0].to_le_bytes());
            }
            let flags = STMC_FLAG_ENABLED | if idx == 0 { STMC_FLAG_STAR_CENTER } else { 0 };
            let wins = [StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: if idx == 1 { 1.0 } else { 0.0 },
                window_weight: 1.0,
                emr_velocity: 0.0,
            }];
            build_proj_stmc_capsule(
                idx,
                flags,
                1.0,
                0.0,
                &[0u8; 88],
                0,
                &wins,
                &[0u8; 4_096],
                &bytes,
            )
            .unwrap()
        };
        stmc_by_id.insert(0, mk_stmc(0, &wet_star));
        stmc_by_id.insert(1, mk_stmc(1, &wet_gain_hi));
        stmc_by_id.insert(2, mk_stmc(2, &wet_boost));

        let prov = StemProvenance {
            input_pcm_sha256: "0".repeat(64),
            target_pcm_sha256: "1".repeat(64),
            num_frames: n as u64,
        };
        manifest.stems_registry = vec![
            crate::proj::schema::StemRegistryEntry {
                stem_id: "star".to_string(),
                chunk_id: 0,
                enabled: true,
                is_star_center: true,
                controls_meta: BTreeMap::from([
                    ("gain".into(), serde_json::json!(0.0)),
                    ("mode".into(), serde_json::json!("clean")),
                ]),
                provenance: prov.clone(),
                user_metadata: BTreeMap::new(),
                extracted_physics: None,
            },
            crate::proj::schema::StemRegistryEntry {
                stem_id: "gain_hi".to_string(),
                chunk_id: 1,
                enabled: true,
                is_star_center: false,
                controls_meta: BTreeMap::from([
                    ("gain".into(), serde_json::json!(1.0)),
                    ("mode".into(), serde_json::json!("clean")),
                ]),
                provenance: prov.clone(),
                user_metadata: BTreeMap::new(),
                extracted_physics: None,
            },
            crate::proj::schema::StemRegistryEntry {
                stem_id: "boost_sw".to_string(),
                chunk_id: 2,
                enabled: true,
                is_star_center: false,
                controls_meta: BTreeMap::from([
                    ("gain".into(), serde_json::json!(0.0)),
                    ("mode".into(), serde_json::json!("boost")),
                ]),
                provenance: prov,
                user_metadata: BTreeMap::new(),
                extracted_physics: None,
            },
        ];

        // Capture StarCenter output BEFORE Stage C
        let mut hdr_before = manifest.base.clone();
        let dnam_before = build_dnam_container_from_dnam_bin(&mut hdr_before, &dnam_bin).unwrap();
        let mut eng_before =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_before, Tier::Ultra, 48_000)
                .unwrap();
        let mut ref_out = [0.0_f32; 64];
        for i in 0..64 {
            ref_out[i] = eng_before.step_frame_sona(dry[i][0], dry[i][1], false)[0];
        }

        let sc = solve_stage_c_knobs(&manifest, &mut dnam_bin, &stmc_by_id, &dry, n, true)
            .expect("stage c should succeed");
        assert_eq!(sc.cp1_slots.len(), 2);

        let mut hdr_after = manifest.base.clone();
        let dnam_after = build_dnam_container_from_dnam_bin(&mut hdr_after, &dnam_bin).unwrap();
        let mut eng_after =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_after, Tier::Ultra, 48_000)
                .unwrap();
        for i in 0..64 {
            let y = eng_after.step_frame_sona(dry[i][0], dry[i][1], false)[0];
            assert_eq!(
                y.to_bits(),
                ref_out[i].to_bits(),
                "0-ULP StarCenter broken at {i}"
            );
        }
    }
}
