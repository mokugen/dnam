#![forbid(unsafe_code)]
//! Stage C (`Parallel Multi-Knob & Switch Conditioning`: `CP1` + `W_ctrl` + `U_int`,
//! `solver::stage_c_knobs`).
//!
//! Canonical References:
//! - `docs/specs/training/four_stage_pipeline.md` (`ref[train.pipeline.stage-d-control]`)
//! - `docs/specs/dsp/parametric_control_bus.md` (`ref[dsp.ctrl.accum.cp1-scaling]`, `ref[dsp.ctrl.accum.anova-fm]`)

mod jacobian;
mod sweep_windows;

use std::collections::BTreeMap;

use dnam_engine::control::{ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::schema::{ControlKind, DnamError};
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, ARENA_ULTRA_BYTES, ARENA0_UINT_BASE_OFFSET, ARENA0_W_CTRL_OFFSET,
    ARENA3_UINT_ULTRA_OFFSET, ARENA3_W_CTRL_ULTRA_OFFSET,
};

use self::jacobian::{
    build_stage_target_jacobian, solve_point_stage_targets, top_target_basis_8_4,
};
use self::sweep_windows::{SweepExtractCtx, try_collect_sweep_windows};
use crate::proj::STMC_FLAG_STEREO;
use crate::proj::bundle::zero_arena_b_head_dc_bias;
use crate::proj::capsules::verify_proj_stmc_capsule;
use crate::proj::schema::DnamProjManifest;
use crate::solver::stages::{sync_clamp_descriptors_to_arena, write_inline_cp1_if_fits};
use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};
use crate::stimulus::sealing::CORESET_A5_FRAMES_48K;

/// Window length (`4,096` samples) spanning `Q3` chord + full `Q4` `25 Hz..17.5 kHz` log chirp.
const A5_STATIONARY_FIT_LEN: usize = 4_096;
/// End of active `Q4` chirp burst (`10,650`) before the silent `CORESET_A5` decay gap.
const A5_ACTIVE_BURST_END: usize =
    (3 * CORESET_A5_FRAMES_48K) / 4 + ((CORESET_A5_FRAMES_48K / 4) * 55) / 100;
/// Start of `CORESET_A5` stationary+chirp window (`6,554`) inside `Q3` (`[6,000..9,000)`) so
/// `[6,554..10,650)` retains `2,446` chord samples plus `100%` of the `Q4` `25 Hz..17.5 kHz` chirp.
const A5_STATIONARY_OFFSET: usize = A5_ACTIVE_BURST_END - A5_STATIONARY_FIT_LEN;
const _: () = assert!(A5_STATIONARY_OFFSET >= CORESET_A5_FRAMES_48K / 2);
const _: () = assert!(A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN == A5_ACTIVE_BURST_END);

/// Output from Stage C multi-knob / switch projection (`r[train.pipeline.stage-d-control]`).
#[derive(Debug, Clone)]
pub struct StageCKnobsOutput {
    /// Wall-clock Stage C duration in milliseconds.
    pub elapsed_ms: f32,
    /// Ordered `CP1` contributor records written inline (up to `10` slots) and in Region 4 `CP1` (`P >= 1`).
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
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let base = f * bytes_per_frame;
        let l = f32::from_le_bytes([
            bytes_fp32[base],
            bytes_fp32[base + 1],
            bytes_fp32[base + 2],
            bytes_fp32[base + 3],
        ]);
        let r = if ch >= 2 {
            f32::from_le_bytes([
                bytes_fp32[base + 4],
                bytes_fp32[base + 5],
                bytes_fp32[base + 6],
                bytes_fp32[base + 7],
            ])
        } else {
            l
        };
        out.push([
            if l.is_finite() { l } else { 0.0 },
            if r.is_finite() { r } else { 0.0 },
        ]);
    }
    out
}

fn parse_stem_control_values(
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

struct ContributorMeta {
    kind: u8,
    control_idx: u16,
    option_idx: u8,
    partner_idx: u16,
    lanes: u8,
}

fn enumerate_contributors(
    manifest: &DnamProjManifest,
    descriptors: &[ControlClampDescriptor],
    active_switch_opts: &[u8],
) -> (Vec<ContributorMeta>, Vec<usize>) {
    let mut contribs = Vec::new();
    let mut active_range_ctrls = Vec::new();
    for (p, ctrl) in manifest.base.controls_schema.iter().enumerate() {
        let Some(desc) = descriptors.get(p) else {
            break;
        };
        match ctrl.kind {
            ControlKind::Range => {
                let lanes = desc.lobatto_degree.min(3);
                if lanes > 0 {
                    active_range_ctrls.push(p);
                    contribs.push(ContributorMeta {
                        kind: 0,
                        control_idx: p as u16,
                        option_idx: 0xFF,
                        partner_idx: 0,
                        lanes,
                    });
                }
            }
            ControlKind::Switch => {
                let def_opt = (desc.default_val.max(0.0) + 0.5) as u8;
                let n_opts = ctrl.declared_options.as_ref().map_or(0, Vec::len).min(16);
                let _ = active_switch_opts;
                for opt_idx in 0..n_opts {
                    if (opt_idx as u8) != def_opt {
                        contribs.push(ContributorMeta {
                            kind: 1,
                            control_idx: p as u16,
                            option_idx: opt_idx as u8,
                            partner_idx: 0,
                            lanes: 1,
                        });
                    }
                }
            }
        }
    }
    (contribs, active_range_ctrls)
}

fn eval_single_range_psi(desc: &ControlClampDescriptor, raw_val: f32, sw_opts: &[u8]) -> [f32; 3] {
    let (_, u_norm, _) = desc.clamp_and_normalize(raw_val, sw_opts);
    desc.eval_lobatto_basis(u_norm)
}

fn eval_contributor_basis_row(
    contribs: &[ContributorMeta],
    descriptors: &[ControlClampDescriptor],
    phys_vals: &[f32],
) -> Vec<f64> {
    let n_ctrl = phys_vals.len();
    let mut sw_opts = vec![0u8; n_ctrl];
    for (p, &val) in phys_vals.iter().enumerate() {
        if descriptors.get(p).is_some_and(|d| d.ctrl_kind == 1) {
            sw_opts[p] = (val.round().max(0.0) as u8).min(15);
        }
    }
    let mut out = Vec::new();
    for c in contribs {
        let p = usize::from(c.control_idx);
        let raw = phys_vals.get(p).copied().unwrap_or(0.0);
        match c.kind {
            0 => {
                let psi = descriptors
                    .get(p)
                    .map_or([0.0; 3], |d| eval_single_range_psi(d, raw, &sw_opts));
                for l in 0..usize::from(c.lanes) {
                    out.push(f64::from(psi[l]));
                }
            }
            1 => {
                let is_active = sw_opts.get(p).copied().unwrap_or(0) == c.option_idx;
                out.push(if is_active { 1.0 } else { 0.0 });
            }
            2 => {
                let q = usize::from(c.partner_idx);
                let p1 = descriptors
                    .get(p)
                    .map_or(0.0, |d| eval_single_range_psi(d, raw, &sw_opts)[0]);
                let p2 = descriptors.get(q).map_or(0.0, |d| {
                    let rv2 = phys_vals.get(q).copied().unwrap_or(0.0);
                    eval_single_range_psi(d, rv2, &sw_opts)[0]
                });
                out.push(f64::from(p1 * p2));
            }
            _ => out.push(0.0),
        }
    }
    out
}

fn fit_cp1_and_write_arena(
    arena: &mut [u8],
    contribs: &[ContributorMeta],
    basis_rows: &[Vec<f64>],
    target_rows: &[[f32; 16]],
    row_weights: &[f64],
    short_capture: bool,
) -> Result<Vec<Cp1ContributorSlot>, DnamError> {
    if contribs.is_empty() || basis_rows.is_empty() {
        let empty = Vec::new();
        write_inline_cp1_if_fits(arena, &empty);
        return Ok(empty);
    }
    let (w_base, w_ultra) = top_target_basis_8_4(target_rows);
    for (r, row_w) in w_base.iter().enumerate() {
        let off = ARENA0_W_CTRL_OFFSET + r * 64;
        // Columns 0..8 map static `z_head[0..4]` + `ctrl_basis_4[0..4]`; preserve cols 8..16
        // (LFO/envelope modulation and Layer-3 Tardos watermark carriers).
        for (c, &w) in row_w[..8].iter().enumerate() {
            arena[off + c * 4..off + (c + 1) * 4].copy_from_slice(&(w as f32).to_le_bytes());
        }
        let u_off = ARENA3_W_CTRL_ULTRA_OFFSET + r * 16;
        if arena.len() >= ARENA3_W_CTRL_ULTRA_OFFSET + 256 {
            for (c, &w) in w_ultra[r].iter().enumerate() {
                arena[u_off + c * 4..u_off + (c + 1) * 4]
                    .copy_from_slice(&(w as f32).to_le_bytes());
            }
        }
    }

    let k_dim = basis_rows[0].len();
    let short_many = short_capture && contribs.len() > 8;
    let mut col_reg = Vec::with_capacity(k_dim);
    for meta in contribs {
        match meta.kind {
            0 => {
                for l in 0..usize::from(meta.lanes) {
                    col_reg.push(match l {
                        0 => 1.0e-4,
                        1 => 1.5e-3,
                        _ => 4.0e-2,
                    });
                }
            }
            1 => col_reg.push(1.0e-4),
            _ => col_reg.push(4.0e-2),
        }
    }
    let mut gram = vec![0.0_f64; uplo_u_packed_len(k_dim)];
    for (s_idx, row) in basis_rows.iter().enumerate() {
        let w = row_weights.get(s_idx).copied().unwrap_or(1.0);
        for j in 0..k_dim {
            for i in 0..=j {
                gram[uplo_u_index(i, j)] += w * row[i] * row[j];
            }
        }
    }
    let shrink = if short_many {
        0.08 * ((contribs.len() as f64) / 8.0)
    } else {
        0.0
    };
    for (i, &reg) in col_reg.iter().enumerate().take(k_dim) {
        let diag_idx = uplo_u_index(i, i);
        let iso_reg = if short_many {
            shrink * gram[diag_idx].max(0.25)
        } else {
            0.0
        };
        gram[diag_idx] += reg + iso_reg;
    }
    let mut z_cols = vec![[0.0_f32; 12]; k_dim];
    for pair in 0..6 {
        if short_many && pair >= 4 {
            continue;
        }
        let mut rhs = vec![[0.0_f64; 2]; k_dim];
        for (s_idx, row) in basis_rows.iter().enumerate() {
            let w = row_weights.get(s_idx).copied().unwrap_or(1.0);
            let mut z_l = 0.0_f64;
            let mut z_r = 0.0_f64;
            for r in 0..16 {
                let t = f64::from(target_rows[s_idx][r]);
                let wl = if pair < 4 {
                    w_base[r][pair * 2]
                } else {
                    w_ultra[r][(pair - 4) * 2]
                };
                let wr = if pair < 4 {
                    w_base[r][pair * 2 + 1]
                } else {
                    w_ultra[r][(pair - 4) * 2 + 1]
                };
                z_l += wl * t;
                z_r += wr * t;
            }
            for i in 0..k_dim {
                rhs[i][0] += w * row[i] * z_l;
                rhs[i][1] += w * row[i] * z_r;
            }
        }
        let sol = solve_uplo_u_cholesky_2ch(&gram, &rhs, 1.0e-7).map_err(|e| {
            DnamError::SchemaValidationError(format!("Stage C CP1 Cholesky failed: {e}"))
        })?;
        for (i, &[sl, sr]) in sol.iter().enumerate() {
            z_cols[i][pair * 2] = (sl as f32).clamp(-8.0, 8.0);
            z_cols[i][pair * 2 + 1] = (sr as f32).clamp(-8.0, 8.0);
        }
    }

    let mut slots = Vec::with_capacity(contribs.len());
    let mut col_cursor = 0usize;
    for meta in contribs {
        let mut slot = Cp1ContributorSlot {
            lanes: meta.lanes,
            ctrl_kind: meta.kind,
            control_idx: meta.control_idx,
            option_idx: meta.option_idx,
            partner_control_id: meta.partner_idx,
            has_ultra: true,
            ..Cp1ContributorSlot::zero()
        };
        for l in 0..usize::from(meta.lanes) {
            let z = z_cols[col_cursor];
            slot.main_base[l].copy_from_slice(&z[..8]);
            slot.main_ultra[l][4..8].copy_from_slice(&z[8..12]);
            col_cursor += 1;
        }
        slots.push(slot);
    }
    write_inline_cp1_if_fits(arena, &slots);
    Ok(slots)
}

/// Runs Stage C (`Parallel Multi-Knob & Switch Conditioning`), solving `CP1`, `U_int`, and
/// `W_ctrl` while guaranteeing bit-exact `0-ULP` invariance at `StarCenter` (`REQ-4.2`).
pub fn solve_stage_c_knobs(
    manifest: &DnamProjManifest,
    arena: &mut [u8],
    stmc_by_id: &BTreeMap<u32, Vec<u8>>,
    dry_lr: &[[f32; 2]],
    active_frames_hint: usize,
    clear_dynamic_w_ctrl: bool,
) -> Result<StageCKnobsOutput, DnamError> {
    if arena.len() < ARENA_LITE_BYTES {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena.len(),
        });
    }
    let timer = now_ms_token();
    let descriptors = sync_clamp_descriptors_to_arena(&manifest.base, arena)?;
    if clear_dynamic_w_ctrl {
        arena[ARENA0_UINT_BASE_OFFSET..ARENA0_UINT_BASE_OFFSET + 128].fill(0);
        for r in 0..16 {
            let off = ARENA0_W_CTRL_OFFSET + r * 64;
            arena[off..off + 32].fill(0);
        }
        if arena.len() >= ARENA_ULTRA_BYTES {
            arena[ARENA3_W_CTRL_ULTRA_OFFSET..ARENA3_W_CTRL_ULTRA_OFFSET + 256].fill(0);
            arena[ARENA3_UINT_ULTRA_OFFSET..ARENA3_UINT_ULTRA_OFFSET + 384].fill(0);
        }
    }

    let base_ctrls = star_center_control_values(manifest);
    let sw_opts = vec![0u8; base_ctrls.len()];
    let (mut contribs, active_ranges) = enumerate_contributors(manifest, &descriptors, &sw_opts);

    let non_star_stems: Vec<_> = manifest
        .stems_registry
        .iter()
        .filter(|s| s.enabled && !s.is_star_center)
        .collect();
    if non_star_stems.is_empty() || contribs.is_empty() || dry_lr.is_empty() {
        let slots: Vec<Cp1ContributorSlot> = contribs
            .into_iter()
            .map(|m| Cp1ContributorSlot {
                lanes: m.lanes,
                ctrl_kind: m.kind,
                control_idx: m.control_idx,
                option_idx: m.option_idx,
                partner_control_id: m.partner_idx,
                ..Cp1ContributorSlot::zero()
            })
            .collect();
        write_inline_cp1_if_fits(arena, &slots);
        zero_arena_b_head_dc_bias(arena);
        return Ok(StageCKnobsOutput {
            elapsed_ms: elapsed_ms_since(timer),
            cp1_slots: slots,
        });
    }

    let avail = dry_lr.len();
    let n_active = if active_frames_hint == 0 {
        avail
    } else {
        active_frames_hint.min(avail)
    };
    let n_fit = if n_active >= 48_000 {
        4_096
    } else {
        2_048.min(n_active)
    };
    let mono_out = manifest.base.io_config.output_mode == "mono";
    let jac = build_stage_target_jacobian(arena, mono_out, dry_lr, n_fit);
    let a5_stat_jac = (n_active >= A5_ACTIVE_BURST_END
        && n_active < 48_000
        && dry_lr.len() >= A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN)
        .then(|| {
            build_stage_target_jacobian(
                arena,
                mono_out,
                &dry_lr[A5_STATIONARY_OFFSET..A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN],
                A5_STATIONARY_FIT_LEN,
            )
        });

    let mut point_ctrls = Vec::new();
    let mut point_targets = Vec::new();
    let mut has_multiknob_pairs = Vec::new();

    for stem in &non_star_stems {
        let Some(stmc_raw) = stmc_by_id.get(&stem.chunk_id) else {
            continue;
        };
        let Ok(stmc_view) = verify_proj_stmc_capsule(stmc_raw) else {
            continue;
        };
        let ch = if (stmc_view.flags & STMC_FLAG_STEREO) != 0 {
            2
        } else {
            1
        };
        let wet_lr = decode_coreset_lr(stmc_view.wet_coreset_fp32, ch);
        let base_meta_ctrls = parse_stem_control_values(manifest, &stem.controls_meta, &base_ctrls);

        let moved_ranges: Vec<usize> = active_ranges
            .iter()
            .copied()
            .filter(|&p| (base_meta_ctrls[p] - base_ctrls[p]).abs() > 1.0e-4)
            .collect();
        if moved_ranges.len() == 2 {
            has_multiknob_pairs.push((moved_ranges[0], moved_ranges[1]));
        }

        let sweep_ctx = SweepExtractCtx {
            arena,
            dry_lr,
            wet_lr: &wet_lr,
            jac: &jac,
            n_fit,
            n_active,
        };
        if try_collect_sweep_windows(
            &sweep_ctx,
            &stmc_view.windows,
            &base_meta_ctrls,
            &mut has_multiknob_pairs,
            &mut point_targets,
            &mut point_ctrls,
        ) {
            continue;
        }
        if wet_lr.len() >= 64 {
            let solved = if let Some(ref s_jac) = a5_stat_jac
                && wet_lr.len() >= A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN
            {
                solve_point_stage_targets(
                    arena,
                    &dry_lr[A5_STATIONARY_OFFSET..A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN],
                    &wet_lr[A5_STATIONARY_OFFSET..A5_STATIONARY_OFFSET + A5_STATIONARY_FIT_LEN],
                    0,
                    s_jac,
                )
            } else {
                let len = wet_lr.len().min(n_fit);
                solve_point_stage_targets(arena, dry_lr, &wet_lr[..len], 0, &jac)
            };
            point_targets.push(solved);
            point_ctrls.push(base_meta_ctrls);
        }
    }

    for meta in &mut contribs {
        if meta.kind == 0 {
            let p = usize::from(meta.control_idx);
            let desc = &descriptors[p];
            let has_interior_1d = point_ctrls.iter().any(|pv| {
                let pure_1d = descriptors.iter().enumerate().all(|(q, dq)| {
                    if q == p {
                        return true;
                    }
                    let vq = pv.get(q).copied().unwrap_or(0.0);
                    let bq = base_ctrls.get(q).copied().unwrap_or(0.0);
                    if dq.ctrl_kind == 1 {
                        (vq.round() - bq.round()).abs() <= 1.0e-4
                    } else {
                        (vq - bq).abs() <= 1.0e-4
                    }
                });
                if !pure_1d {
                    return false;
                }
                let raw_p = pv.get(p).copied().unwrap_or(0.0);
                let (_, u_norm, _) = desc.clamp_and_normalize(raw_p, &sw_opts);
                (u_norm - desc.anchor_nodes_norm[0]).abs() > 0.05 && u_norm.abs() < 0.92
            });
            if !has_interior_1d {
                meta.lanes = meta.lanes.min(2);
            }
        }
    }

    has_multiknob_pairs.sort_unstable();
    has_multiknob_pairs.dedup();
    for (p, q) in has_multiknob_pairs {
        contribs.push(ContributorMeta {
            kind: 2,
            control_idx: p as u16,
            option_idx: 0xFF,
            partner_idx: q as u16,
            lanes: 1,
        });
    }

    let basis_rows: Vec<Vec<f64>> = point_ctrls
        .iter()
        .map(|pv| eval_contributor_basis_row(&contribs, &descriptors, pv))
        .collect();
    let row_weights: Vec<f64> = point_ctrls
        .iter()
        .map(|pv| {
            let moved_count = pv
                .iter()
                .zip(&base_ctrls)
                .filter(|&(&v, &b)| (v - b).abs() > 1.0e-4)
                .count();
            if moved_count > 2 { 0.25 } else { 1.0 }
        })
        .collect();
    let slots = fit_cp1_and_write_arena(
        arena,
        &contribs,
        &basis_rows,
        &point_targets,
        &row_weights,
        n_active < 48_000,
    )?;
    zero_arena_b_head_dc_bias(arena);

    Ok(StageCKnobsOutput {
        elapsed_ms: elapsed_ms_since(timer),
        cp1_slots: slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::capsules::build_proj_stmc_capsule;
    use crate::proj::directory::StmcTrajectoryWindow;
    use crate::proj::schema::StemProvenance;
    use crate::proj::{STMC_FLAG_ENABLED, STMC_FLAG_STAR_CENTER};
    use crate::solver::stage_b_backbone::solve_stage_b_backbone;
    use dnam_engine::container::build_dnam_container_with_cp1;
    use dnam_engine::control::bus::cp1_and_uint::decode_cp1_inline_contributors;
    use dnam_engine::math::DefaultMathBackend;
    use dnam_engine::{DnamEngine, Tier};
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

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let _sb = solve_stage_b_backbone(&manifest, &mut arena, &dry, &wet_star, n, 2, false)
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
        let dnam_before =
            build_dnam_container_with_cp1(&mut hdr_before, &arena, &arena, false, &[]).unwrap();
        let mut eng_before =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_before, Tier::Ultra, 48_000)
                .unwrap();
        let mut ref_out = [0.0_f32; 64];
        for i in 0..64 {
            ref_out[i] = eng_before.step_frame_tora(dry[i][0], dry[i][1], false)[0];
        }

        let sc = solve_stage_c_knobs(&manifest, &mut arena, &stmc_by_id, &dry, n, true)
            .expect("stage c should succeed");
        assert_eq!(sc.cp1_slots.len(), 2);
        let mut decoded_slots = [Cp1ContributorSlot::zero(); 16];
        assert_eq!(
            decode_cp1_inline_contributors(&arena, &mut decoded_slots).unwrap(),
            2
        );

        let mut hdr_after = manifest.base.clone();
        let dnam_after =
            build_dnam_container_with_cp1(&mut hdr_after, &arena, &arena, false, &sc.cp1_slots)
                .unwrap();
        let mut eng_after =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_after, Tier::Ultra, 48_000)
                .unwrap();
        for i in 0..64 {
            let y = eng_after.step_frame_tora(dry[i][0], dry[i][1], false)[0];
            assert_eq!(
                y.to_bits(),
                ref_out[i].to_bits(),
                "0-ULP StarCenter broken at {i}"
            );
        }
    }
}
