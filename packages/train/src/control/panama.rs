//! Deterministic `< 10 ms` CPU `PANAMA` (Predictive Anchored Multi-Ensemble Acquisition)
//! multi-knob corner recommender (`ref[train.audit.report.prescription]`,
//! `docs/specs/training/control_space_audit.md §2`).

mod candidates;
pub mod chacha;
mod ensemble;

use dnam_vm::bin::DnamBinView;
use dnam_vm::control::{ControlBus, STAGE_TARGET_SLOTS};
use dnam_vm::math::DefaultMathBackend;
use sha2::{Digest, Sha256};

use crate::control::clamp::dynamic::compute_dynamic_excursion_bounds_32;
use crate::control::cp1::CP1_DELAY_VARIANCE_GATE_SAMPLES2;
use crate::error::TrainError;
use crate::linalg::{checked_uplo_u_packed_len, factorize_uplo_u_cholesky};
use crate::wide_gram::uplo_u_index;

use candidates::{
    ExclusionCtx, build_varying_axes, candidate_passes_exclusions, compare_scored_candidates,
    delta_axis, format_panama_prescription, generate_candidate_grid, quantize_tuple,
    round_half_away_i64,
};
use ensemble::{
    EnsembleFitCtx, compute_main_effect_dim, eval_main_effect_psi, fit_ensemble_members,
    score_candidate_sigma2,
};

/// Number of `PANAMA` ensemble members (`M = 5`).
pub const PANAMA_ENSEMBLE_SIZE: usize = 5;
/// Baseline Pass-4 `CP1` ridge regularization (`lambda_cp1 = 1e-3`).
pub const PANAMA_LAMBDA_CP1: f64 = 1.0e-3;
/// Ordered multiplicative ridge schedule `lambda_m = mult[m] * lambda_cp1`.
pub const PANAMA_LAMBDA_MULTS: [f64; PANAMA_ENSEMBLE_SIZE] = [0.25, 0.5, 1.0, 2.0, 4.0];
/// Leverage uncertainty weight (`gamma_lev = 0.05`).
pub const PANAMA_GAMMA_LEV: f64 = 0.05;
/// Minimum normalized disagreement/leverage threshold (`sigma2 >= 1e-6`).
pub const PANAMA_MIN_SIGMA2: f64 = 1.0e-6;
/// Maximum top recommended corners emitted per audit pass (`3`).
pub const PANAMA_MAX_RECOMMENDATIONS: usize = 3;
/// Minimum Chebyshev normalized distance `d_inf > 1e-3` from existing sample rows (`(a)`).
pub const PANAMA_EXCLUSION_DINF: f64 = 1.0e-3;
/// Conditional-bound tolerance `1e-6` (`(b)`).
pub const PANAMA_COND_BOUND_TOL: f64 = 1.0e-6;
/// Normalized delta threshold `> 1e-6` identifying non-center coordinates.
pub const PANAMA_DELTA_TOL: f64 = 1.0e-6;
/// Full Cartesian candidate grid cutoff (`P_var <= 5`).
pub const PANAMA_FULL_GRID_MAX_PVAR: usize = 5;
/// Number of top normalized-stddev 3D triples included when `P_var > 5`.
pub const PANAMA_TOP_TRIPLES_COUNT: usize = 6;

const _: () = assert!(PANAMA_ENSEMBLE_SIZE == 5);
const _: () = assert!(PANAMA_LAMBDA_CP1 > 0.0);
const _: () = assert!(PANAMA_GAMMA_LEV > 0.0);
const _: () = assert!(PANAMA_MIN_SIGMA2 > 0.0);
const _: () = assert!(PANAMA_MAX_RECOMMENDATIONS == 3);
const _: () = assert!(PANAMA_FULL_GRID_MAX_PVAR == 5);
const _: () = assert!(PANAMA_TOP_TRIPLES_COUNT == 6);
const _: () = assert!(STAGE_TARGET_SLOTS == 32);

/// Continuous range vs discrete switch control metadata for `PANAMA`.
#[derive(Clone, Debug, PartialEq)]
pub enum PanamaControlKind {
    /// Continuous calibrated control bounded by `[safe_min, safe_max]` around `star_center`.
    Range {
        /// Safe minimum physical value (`Phase 1` range-clamped).
        safe_min: f64,
        /// Safe maximum physical value (`Phase 1` range-clamped).
        safe_max: f64,
        /// Reference `StarCenter` physical setting.
        star_center: f64,
    },
    /// Discrete switch selector with observed active options and option display labels.
    Switch {
        /// Observed active integer option indices.
        active_options: Vec<u8>,
        /// Declared option names indexed by option number.
        declared_option_names: Vec<String>,
        /// Reference `StarCenter` option index.
        star_center_option: u8,
    },
}

/// Canonical control descriptor input supplied to `recommend_panama_corners`.
#[derive(Clone, Debug, PartialEq)]
pub struct PanamaControlDesc {
    /// Canonical schema control identifier (`ControlDescriptorJson.id`).
    pub control_id: String,
    /// Position of this control inside `.dnam-bin` `control_descriptors`.
    pub wire_ctrl_idx: usize,
    /// Continuous range vs discrete switch schema details.
    pub kind: PanamaControlKind,
}

/// One Pass-4 solved sample row (`stationary` stem or synchronous `MODE_A_SWEEP` window).
#[derive(Clone, Debug, PartialEq)]
pub struct PanamaSampleRow {
    /// Deterministic stem identifier (for `(stem_id, window_idx)` sorting).
    pub stem_id: String,
    /// Zero-based window index within the stem.
    pub window_idx: usize,
    /// Physical control vector aligned one-to-one with `PanamaInput::controls`.
    pub control_values: Vec<f64>,
    /// Solved Pass-4 `StarCenter`-centered 32-slot targets (`slot 31 == +0.0`).
    pub stage_targets: [f64; STAGE_TARGET_SLOTS],
}

/// Full input bundle required by deterministic `PANAMA` corner recommendation.
#[derive(Clone, Debug)]
pub struct PanamaInput<'a> {
    /// Parsed compiled `.dnam-bin` view (wire descriptors, `"W_ctrl"`, optional `"CEXT"`).
    pub view: &'a DnamBinView<'a>,
    /// Schema controls matching `PanamaSampleRow::control_values` length and order.
    pub controls: &'a [PanamaControlDesc],
    /// Solved Pass-4 stationary and/or sweep window rows.
    pub sample_rows: &'a [PanamaSampleRow],
    /// Lowercase 64-hex `target_pcm_sha256` strings across all enabled stems.
    pub enabled_stem_target_sha256s: &'a [String],
}

/// Ranked `PANAMA` corner recommendation with exact formatted prescription string.
#[derive(Clone, Debug, PartialEq)]
pub struct PanamaCornerScore {
    /// Recommended setting values `(control_id, physical_val)` sorted ascending by `control_id`.
    pub control_values: Vec<(String, f64)>,
    /// Ensemble disagreement plus regularized leverage uncertainty `sigma2`.
    pub sigma2: f64,
    /// Exact single-line `[PANAMA] Corner (...) (sigma2=...)` prescription string.
    pub prescription: String,
}

/// Deterministic output returned by `recommend_panama_corners`.
#[derive(Clone, Debug, PartialEq)]
pub struct PanamaOutput {
    /// Lowercase 64-hex SHA-256 over sorted enabled-stem `target_pcm_sha256` hashes.
    pub stems_key: String,
    /// Ranked top `0..=3` recommended corners (`sigma2 >= 1e-6`).
    pub candidates: Vec<PanamaCornerScore>,
    /// Top `0..=3` exact `[PANAMA]` lines to append to `audit_report.prescriptions`.
    pub prescriptions: Vec<String>,
}

fn format_sha256_lower_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in digest {
        out.push(char::from(HEX[usize::from(b >> 4)]));
        out.push(char::from(HEX[usize::from(b & 0x0F)]));
    }
    out
}

/// Computes `stems_key`: lowercase 64-hex SHA-256 over sorted lowercase 64-hex enabled stem hashes
/// joined by ASCII LF (`0x0A`) with no trailing newline (`ref[train.audit.report.prescription]`).
#[must_use]
pub fn compute_stems_key_sha256(enabled_stem_target_sha256s: &[String]) -> String {
    let mut sorted: Vec<String> = enabled_stem_target_sha256s
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    sorted.sort();
    let mut hasher = Sha256::new();
    for (idx, h) in sorted.iter().enumerate() {
        if idx > 0 {
            hasher.update(b"\n");
        }
        hasher.update(h.as_bytes());
    }
    let digest: [u8; 32] = hasher.finalize().into();
    format_sha256_lower_hex(&digest)
}

/// Evaluates deterministic `< 10 ms` CPU `PANAMA` active corner acquisition over Pass-4 targets
/// (`ref[train.audit.report.prescription]`).
pub fn recommend_panama_corners(input: &PanamaInput<'_>) -> Result<PanamaOutput, TrainError> {
    let stems_key = compute_stems_key_sha256(input.enabled_stem_target_sha256s);
    let (axes, fixed_stars) = build_varying_axes(input.view, input.controls)?;
    if axes.len() < 2 || input.sample_rows.is_empty() {
        return Ok(PanamaOutput {
            stems_key,
            candidates: Vec::new(),
            prescriptions: Vec::new(),
        });
    }

    let n_ctrls = input.controls.len();
    for row in input.sample_rows {
        if row.control_values.len() != n_ctrls {
            return Err(TrainError::InvalidDimensions {
                context: "panama_sample_row.control_values",
                expected: n_ctrls,
                actual: row.control_values.len(),
            });
        }
        if !row.control_values.iter().all(|v| v.is_finite())
            || !row.stage_targets.iter().all(|v| v.is_finite())
        {
            return Err(TrainError::NonFiniteInput {
                context: "panama_sample_row",
            });
        }
    }

    let mut sorted_rows: Vec<&PanamaSampleRow> = input.sample_rows.iter().collect();
    sorted_rows.sort_by(|a, b| {
        a.stem_id
            .cmp(&b.stem_id)
            .then_with(|| a.window_idx.cmp(&b.window_idx))
    });

    let n_samp = sorted_rows.len() as f64;
    let mean12 = sorted_rows.iter().map(|r| r.stage_targets[12]).sum::<f64>() / n_samp;
    let var12 = sorted_rows
        .iter()
        .map(|r| {
            let d = r.stage_targets[12] - mean12;
            d * d
        })
        .sum::<f64>()
        / n_samp;
    let gate_slot12 = var12 < CP1_DELAY_VARIANCE_GATE_SAMPLES2;

    let mut targets: Vec<[f64; 31]> = Vec::with_capacity(sorted_rows.len());
    let mut min_slots = [f64::INFINITY; 31];
    let mut max_slots = [f64::NEG_INFINITY; 31];
    for row in &sorted_rows {
        let mut t = [0.0_f64; 31];
        t.copy_from_slice(&row.stage_targets[..31]);
        if gate_slot12 {
            t[12] = 0.0;
        }
        for r in 0..31 {
            min_slots[r] = min_slots[r].min(t[r]);
            max_slots[r] = max_slots[r].max(t[r]);
        }
        targets.push(t);
    }

    let mut w_sens = [0.0_f64; 31];
    for r in 0..31 {
        let span = max_slots[r] - min_slots[r];
        if span >= 1.0e-6 {
            w_sens[r] = 2.0 / span;
        }
    }

    let basis_dim = compute_main_effect_dim(&axes);
    let packed_len = checked_uplo_u_packed_len(basis_dim).ok_or(TrainError::InvalidDimensions {
        context: "panama_basis_gram",
        expected: usize::MAX,
        actual: basis_dim,
    })?;

    let mut sample_var_rows = Vec::with_capacity(sorted_rows.len());
    let mut psi_rows = Vec::with_capacity(sorted_rows.len());
    let mut g0_uplo = vec![0.0_f64; packed_len];
    let mut anchor_gram = vec![0.0_f64; packed_len];
    let mut anchor_rhs_flat = vec![0.0_f64; 31 * basis_dim];
    let mut interior_indices = Vec::new();

    for (row_idx, row) in sorted_rows.iter().enumerate() {
        let var_vec: Vec<f64> = axes
            .iter()
            .map(|ax| row.control_values[ax.orig_idx])
            .collect();
        let moved = axes
            .iter()
            .enumerate()
            .filter(|&(c, ax)| delta_axis(ax, var_vec[c], ax.star_center_val) > PANAMA_DELTA_TOL)
            .count();
        let psi = eval_main_effect_psi(&axes, &var_vec);

        for j in 0..basis_dim {
            let pj = psi[j];
            for (i, &pi) in psi.iter().enumerate().take(j + 1) {
                g0_uplo[uplo_u_index(i, j)] += pi * pj;
            }
        }
        if moved <= 1 {
            for j in 0..basis_dim {
                let pj = psi[j];
                for (i, &pi) in psi.iter().enumerate().take(j + 1) {
                    anchor_gram[uplo_u_index(i, j)] += pi * pj;
                }
            }
            for r in 0..31 {
                let tr = targets[row_idx][r];
                if tr != 0.0 {
                    let base = r * basis_dim;
                    for c in 0..basis_dim {
                        anchor_rhs_flat[base + c] += psi[c] * tr;
                    }
                }
            }
        } else {
            interior_indices.push(row_idx);
        }
        sample_var_rows.push(var_vec);
        psi_rows.push(psi);
    }

    let r0_factor = factorize_uplo_u_cholesky(basis_dim, &g0_uplo, PANAMA_LAMBDA_CP1)?;
    let members = fit_ensemble_members(&EnsembleFitCtx {
        axes_dim: basis_dim,
        anchor_gram: &anchor_gram,
        anchor_rhs: &anchor_rhs_flat,
        interior_indices: &interior_indices,
        psi_rows: &psi_rows,
        targets: &targets,
        w_sens: &w_sens,
        stems_key: &stems_key,
    })?;

    let candidates_raw = generate_candidate_grid(&axes, &sorted_rows);
    let delta_32 = compute_dynamic_excursion_bounds_32(input.view);
    let mut bus = ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(
        input.view,
        input.view.retained_tier,
        48_000.0,
    )?;

    let exclusion_ctx = ExclusionCtx {
        view: input.view,
        controls: input.controls,
        axes: &axes,
        fixed_stars: &fixed_stars,
        sample_var_rows: &sample_var_rows,
        delta_32: &delta_32,
    };
    let mut scored: Vec<(i64, Vec<i64>, Vec<f64>, f64)> = Vec::new();
    for cand_var in candidates_raw {
        if !candidate_passes_exclusions(&exclusion_ctx, &cand_var, &mut bus)? {
            continue;
        }
        let sigma2 = score_candidate_sigma2(&axes, &cand_var, &members, &w_sens, &r0_factor)?;
        if sigma2.is_finite() && sigma2 >= PANAMA_MIN_SIGMA2 {
            let score_q = round_half_away_i64(sigma2 * 1.0e6);
            let tuple_q = quantize_tuple(&axes, &cand_var);
            scored.push((score_q, tuple_q, cand_var, sigma2));
        }
    }

    scored.sort_by(compare_scored_candidates);
    scored.truncate(PANAMA_MAX_RECOMMENDATIONS);

    let mut out_candidates = Vec::with_capacity(scored.len());
    let mut out_prescriptions = Vec::with_capacity(scored.len());
    for (_, _, cand_var, sigma2) in scored {
        let mut all_sorted: Vec<(String, f64)> = input
            .controls
            .iter()
            .enumerate()
            .map(|(idx, c)| (c.control_id.clone(), fixed_stars[idx]))
            .collect();
        for (ax, &v) in axes.iter().zip(cand_var.iter()) {
            all_sorted[ax.orig_idx].1 = v;
        }
        all_sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let prescription = format_panama_prescription(&axes, &cand_var, sigma2);
        out_prescriptions.push(prescription.clone());
        out_candidates.push(PanamaCornerScore {
            control_values: all_sorted,
            sigma2,
            prescription,
        });
    }

    Ok(PanamaOutput {
        stems_key,
        candidates: out_candidates,
        prescriptions: out_prescriptions,
    })
}

#[cfg(test)]
mod tests {
    use dnam_vm::bin::{
        ConditionalClampRuleWire, ControlClampDescriptorWire, DnamBinBuildInput, DnamBinView,
        encode_dnam_bin,
    };
    use dnam_vm::control::cp1::Cp1ContributorSlot;
    use dnam_vm::control::descriptor::compute_scale_offset_norm;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    use crate::control::cp1::pack_cp1_slots_into_wire;
    use crate::solver::model::SonaModelF64;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};

    use super::*;

    fn build_range_desc(min: f32, max: f32, default_val: f32) -> ControlClampDescriptorWire {
        let (scale, offset) = compute_scale_offset_norm(min, max);
        let u_def = (default_val * scale + offset).clamp(-1.0, 1.0);
        let mut w = ControlClampDescriptorWire::new_zeroed();
        w.ctrl_kind = 0;
        w.lobatto_degree = 3;
        w.default_val = F32::new(default_val);
        w.safe_min_base = F32::new(min);
        w.safe_max_base = F32::new(max);
        w.scale_norm = F32::new(scale);
        w.offset_norm = F32::new(offset);
        w.anchor_nodes_norm = [F32::new(u_def), F32::new(-1.0), F32::new(1.0)];
        w
    }

    fn build_switch_desc(active_mask: u16, default_opt: u8) -> ControlClampDescriptorWire {
        let max_opt = (16 - active_mask.leading_zeros()).saturating_sub(1) as u8;
        let mut w = ControlClampDescriptorWire::new_zeroed();
        w.ctrl_kind = 1;
        w.lobatto_degree = max_opt;
        w.active_options_mask = U16::new(active_mask);
        w.default_val = F32::new(f32::from(default_opt));
        w.safe_min_base = F32::new(0.0);
        w.safe_max_base = F32::new(f32::from(max_opt));
        w
    }

    fn build_bin_with_descs(
        descs: Vec<ControlClampDescriptorWire>,
        delay_mod_weight: f32,
    ) -> Vec<u8> {
        let mut model = SonaModelF64::new(48_000.0);
        for tq in &mut model.transport_quads {
            tq.delay_samples = [16.0; 4];
            tq.g_loop = 0.5;
        }
        let base_bytes = encode_model_to_dnam_bin(
            &model,
            Pass5EmitConfig {
                calibrate_lufs_trims: false,
                ..Pass5EmitConfig::default()
            },
        )
        .unwrap();
        let base_view = DnamBinView::parse(&base_bytes).unwrap();
        let mut a0 = *base_view.arena0;
        let mut a3 = *base_view.arena3.unwrap();
        if delay_mod_weight != 0.0 && !descs.is_empty() {
            let mut slot0 = Cp1ContributorSlot::zero();
            slot0.control_idx = 0;
            slot0.ctrl_kind = 0;
            slot0.lanes = 1;
            slot0.main_base[0][4] = 1.0;
            pack_cp1_slots_into_wire(
                &[slot0],
                descs.len(),
                true,
                &mut a0.control_bus_lite,
                &mut a3.control_bus_ultra_delta,
            )
            .unwrap();
            a0.control_bus_lite.w_ctrl[12][4] = F32::new(delay_mod_weight);
        }
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: base_view.header.flags.get(),
            ext_ring_pow2_frames: 7,
            ext_ring_lines: 16,
            ext_ring_format: 0,
            declared_max_delay_48k: 64,
            dc_blocker_pole_5hz: 0.999,
            tail_time_ms: 80.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 64,
            content_hash_prefix: [0; 8],
            controls: &descs,
            cp1_overflow: &[],
            tsyn: &[],
            cext: base_view.cext,
            arena0: &a0,
            arena1: base_view.arena1,
            arena2: base_view.arena2,
            arena3: Some(&a3),
        };
        encode_dnam_bin(&input).unwrap()
    }

    #[test]
    fn test_panama_stems_key_neg_zero_and_anchor_only_n_int_zero() {
        let k1 =
            compute_stems_key_sha256(&["BBBB".repeat(16), "AAAA".repeat(16), "aaaa".repeat(16)]);
        let k2 =
            compute_stems_key_sha256(&["aaaa".repeat(16), "aaaa".repeat(16), "bbbb".repeat(16)]);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64);

        let bytes = build_bin_with_descs(vec![build_range_desc(-1.0, 1.0, 0.0)], 0.0);
        let view = DnamBinView::parse(&bytes).unwrap();
        let controls = vec![PanamaControlDesc {
            control_id: "gain".into(),
            wire_ctrl_idx: 0,
            kind: PanamaControlKind::Range {
                safe_min: -1.0,
                safe_max: 1.0,
                star_center: 0.0,
            },
        }];
        let out_lt2 = recommend_panama_corners(&PanamaInput {
            view: &view,
            controls: &controls,
            sample_rows: &[],
            enabled_stem_target_sha256s: &[],
        })
        .unwrap();
        assert!(out_lt2.candidates.is_empty());

        let bytes2 = build_bin_with_descs(
            vec![
                build_range_desc(-1.0e-7, 1.0, 1.0),
                build_switch_desc(0b11, 0),
            ],
            0.0,
        );
        let view2 = DnamBinView::parse(&bytes2).unwrap();
        let ctrls2 = vec![
            PanamaControlDesc {
                control_id: "boost".into(),
                wire_ctrl_idx: 1,
                kind: PanamaControlKind::Switch {
                    active_options: vec![0, 1],
                    declared_option_names: vec!["off".into(), "on".into()],
                    star_center_option: 0,
                },
            },
            PanamaControlDesc {
                control_id: "gain".into(),
                wire_ctrl_idx: 0,
                kind: PanamaControlKind::Range {
                    safe_min: -1.0e-7,
                    safe_max: 1.0,
                    star_center: 1.0,
                },
            },
        ];
        let mut t_a = [0.0_f64; STAGE_TARGET_SLOTS];
        t_a[0] = 1.0;
        let mut t_b = [0.0_f64; STAGE_TARGET_SLOTS];
        t_b[0] = -0.8;
        let rows2 = vec![
            PanamaSampleRow {
                stem_id: "s0".into(),
                window_idx: 0,
                control_values: vec![0.0, 1.0],
                stage_targets: [0.0; STAGE_TARGET_SLOTS],
            },
            PanamaSampleRow {
                stem_id: "s1".into(),
                window_idx: 0,
                control_values: vec![1.0, 1.0],
                stage_targets: t_a,
            },
            PanamaSampleRow {
                stem_id: "s2".into(),
                window_idx: 0,
                control_values: vec![0.0, 0.5],
                stage_targets: t_b,
            },
        ];
        let out2 = recommend_panama_corners(&PanamaInput {
            view: &view2,
            controls: &ctrls2,
            sample_rows: &rows2,
            enabled_stem_target_sha256s: &["01".repeat(32)],
        })
        .unwrap();
        assert!(!out2.prescriptions.is_empty() && out2.prescriptions.len() <= 3);
        assert!(out2.prescriptions.iter().all(|p| !p.contains("-0.0000")));
        assert!(out2.prescriptions.iter().any(|p| p.contains("boost='on'")));
        assert!(out2.prescriptions.iter().any(|p| p.contains("gain=0.0000")));
    }

    #[test]
    fn test_panama_n_int_repeatability_p_var_gt5_and_all_exclusions() {
        let mut d0 = build_range_desc(0.0, 10.0, 5.0);
        d0.cond_rules[0] = ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(1),
            switch_opt_idx: 1,
            rule_op: 1,
            bound_val: F32::new(7.0),
        };
        let mut descs = vec![d0, build_switch_desc(0b11, 0)];
        for _ in 2..6 {
            descs.push(build_range_desc(-1.0, 1.0, 0.0));
        }
        let bytes = build_bin_with_descs(descs, -20.0);
        let view = DnamBinView::parse(&bytes).unwrap();

        let controls = vec![
            PanamaControlDesc {
                control_id: "a_drive".into(),
                wire_ctrl_idx: 0,
                kind: PanamaControlKind::Range {
                    safe_min: 0.0,
                    safe_max: 10.0,
                    star_center: 5.0,
                },
            },
            PanamaControlDesc {
                control_id: "b_mode".into(),
                wire_ctrl_idx: 1,
                kind: PanamaControlKind::Switch {
                    active_options: vec![0, 1],
                    declared_option_names: vec!["clean".into(), "hot".into()],
                    star_center_option: 0,
                },
            },
            PanamaControlDesc {
                control_id: "c_bass".into(),
                wire_ctrl_idx: 2,
                kind: PanamaControlKind::Range {
                    safe_min: -1.0,
                    safe_max: 1.0,
                    star_center: 0.0,
                },
            },
            PanamaControlDesc {
                control_id: "d_mid".into(),
                wire_ctrl_idx: 3,
                kind: PanamaControlKind::Range {
                    safe_min: -1.0,
                    safe_max: 1.0,
                    star_center: 0.0,
                },
            },
            PanamaControlDesc {
                control_id: "e_treb".into(),
                wire_ctrl_idx: 4,
                kind: PanamaControlKind::Range {
                    safe_min: -1.0,
                    safe_max: 1.0,
                    star_center: 0.0,
                },
            },
            PanamaControlDesc {
                control_id: "f_pres".into(),
                wire_ctrl_idx: 5,
                kind: PanamaControlKind::Range {
                    safe_min: -1.0,
                    safe_max: 1.0,
                    star_center: 0.0,
                },
            },
        ];

        let mk_row = |stem: &str, vals: Vec<f64>, t0: f64| -> PanamaSampleRow {
            let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
            st[0] = t0;
            PanamaSampleRow {
                stem_id: stem.into(),
                window_idx: 0,
                control_values: vals,
                stage_targets: st,
            }
        };
        let rows = vec![
            mk_row("s0", vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
            mk_row("s1", vec![5.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.6),
            mk_row("s2", vec![0.0, 0.0, -1.0, 0.0, 0.0, 0.0], -0.4),
            mk_row("s3", vec![5.0, 0.0, 1.0, 1.0, 0.0, 0.0], 0.5),
            mk_row("s4", vec![5.0, 1.0, 0.0, -1.0, 1.0, 0.0], -0.7),
            mk_row("s5", vec![7.0, 1.0, 0.0, 0.0, -1.0, 1.0], 0.9),
        ];

        let out_a = recommend_panama_corners(&PanamaInput {
            view: &view,
            controls: &controls,
            sample_rows: &rows,
            enabled_stem_target_sha256s: &["aa".repeat(32), "bb".repeat(32)],
        })
        .unwrap();
        let out_b = recommend_panama_corners(&PanamaInput {
            view: &view,
            controls: &controls,
            sample_rows: &rows,
            enabled_stem_target_sha256s: &["bb".repeat(32), "aa".repeat(32)],
        })
        .unwrap();

        assert_eq!(out_a, out_b);
        assert_eq!(out_a.prescriptions.len(), 3);
        let (axes, _) = build_varying_axes(&view, &controls).unwrap();
        for cand in &out_a.candidates {
            let cand_vals: Vec<f64> = cand.control_values.iter().map(|&(_, v)| v).collect();
            for row in &rows {
                assert!(
                    candidates::d_inf(&axes, &cand_vals, &row.control_values)
                        > PANAMA_EXCLUSION_DINF
                );
            }
            let a_val = cand.control_values[0].1;
            let b_val = cand.control_values[1].1;
            assert!(!(b_val == 1.0 && a_val > 7.0));
            assert!(a_val < 9.5);
        }
    }
}
