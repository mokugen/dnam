//! `PANAMA` varying-axis normalization, `P_var <= 5` / `P_var > 5` candidate grid synthesis,
//! exclusions `(a)`/`(b)`/`(c)`, canonical tie-breaking sort, and prescription formatting
//! (`ref[train.audit.report.prescription]`).

use std::cmp::Ordering;
use std::collections::BTreeSet;

use dnam_vm::bin::DnamBinView;
use dnam_vm::control::{ControlBus, ControlClampDescriptor, STAGE_TARGET_SLOTS};
use dnam_vm::math::DefaultMathBackend;

use crate::control::clamp::dynamic::{
    DynamicCornerViolation, check_corner_dynamic_stability, eval_static_control_bus_targets,
};
use crate::error::TrainError;

use super::{
    PANAMA_COND_BOUND_TOL, PANAMA_DELTA_TOL, PANAMA_EXCLUSION_DINF, PANAMA_FULL_GRID_MAX_PVAR,
    PANAMA_TOP_TRIPLES_COUNT, PanamaControlDesc, PanamaControlKind, PanamaSampleRow,
};

/// Normalized varying axis descriptor (`0..P_var` sorted ascending by `control_id`).
#[derive(Clone, Debug)]
pub(super) struct VaryingAxis {
    pub orig_idx: usize,
    pub control_id: String,
    pub star_center_val: f64,
    pub kind: AxisKind,
}

#[derive(Clone, Debug)]
pub(super) enum AxisKind {
    Range {
        safe_min: f64,
        safe_max: f64,
        choices: [f64; 3],
        desc: ControlClampDescriptor,
        u_star_norm: f32,
        star_lobatto: [f32; 3],
    },
    Switch {
        sorted_active_opts: Vec<u8>,
        grid_opts: Vec<u8>,
        declared_option_names: Vec<String>,
        star_center_opt: u8,
    },
}

/// Symmetric half-away-from-zero integer rounding: `sign(x) * floor(|x| + 0.5)`.
#[inline]
pub(super) fn round_half_away_i64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5).floor() as i64
    } else {
        -(x.abs() + 0.5).floor() as i64
    }
}

/// Formats an IEEE-754 binary64 scalar to 4 decimals (`RNE`), normalizing `"-0.0000"` to `"0.0000"`.
#[inline]
pub(super) fn format_f64_4dp_no_neg_zero(v: f64) -> String {
    let s = format!("{v:.4}");
    if s == "-0.0000" {
        "0.0000".to_string()
    } else {
        s
    }
}

#[inline]
fn decode_switch_opt(v: f64) -> u8 {
    round_half_away_i64(v).clamp(0, i64::from(u8::MAX)) as u8
}

/// Normalizes `v` into `[0.0, 1.0]` along `axis` (`z_c(v)`).
pub(super) fn normalize_axis_value(axis: &VaryingAxis, v: f64) -> f64 {
    match &axis.kind {
        AxisKind::Range {
            safe_min, safe_max, ..
        } => {
            let span = safe_max - safe_min;
            if span > 0.0 {
                (v - safe_min) / span
            } else {
                0.0
            }
        }
        AxisKind::Switch {
            sorted_active_opts, ..
        } => {
            let opt = decode_switch_opt(v);
            let k = sorted_active_opts.len();
            if k <= 1 {
                return 0.0;
            }
            let rank = sorted_active_opts
                .iter()
                .position(|&o| o == opt)
                .unwrap_or(0);
            (rank as f64) / ((k - 1) as f64)
        }
    }
}

/// Computes `Delta_c(u_c, u'_c)` (`|z_c(a) - z_c(b)|` for range, `0`/`1` for switch).
#[inline]
pub(super) fn delta_axis(axis: &VaryingAxis, a: f64, b: f64) -> f64 {
    match &axis.kind {
        AxisKind::Range { .. } => {
            (normalize_axis_value(axis, a) - normalize_axis_value(axis, b)).abs()
        }
        AxisKind::Switch { .. } => {
            if decode_switch_opt(a) == decode_switch_opt(b) {
                0.0
            } else {
                1.0
            }
        }
    }
}

#[inline]
pub(super) fn quantize_axis_tuple_elem(axis: &VaryingAxis, v: f64) -> i64 {
    match &axis.kind {
        AxisKind::Range { .. } => round_half_away_i64(v * 1.0e4),
        AxisKind::Switch { .. } => i64::from(decode_switch_opt(v)) * 10_000,
    }
}

pub(super) fn quantize_tuple(axes: &[VaryingAxis], var_vals: &[f64]) -> Vec<i64> {
    axes.iter()
        .zip(var_vals.iter())
        .map(|(ax, &v)| quantize_axis_tuple_elem(ax, v))
        .collect()
}

pub(super) fn d_inf(axes: &[VaryingAxis], a: &[f64], b: &[f64]) -> f64 {
    let mut max_d = 0.0_f64;
    for (ax, (&va, &vb)) in axes.iter().zip(a.iter().zip(b.iter())) {
        max_d = max_d.max(delta_axis(ax, va, vb));
    }
    max_d
}

pub(super) fn build_varying_axes(
    view: &DnamBinView<'_>,
    controls: &[PanamaControlDesc],
) -> Result<(Vec<VaryingAxis>, Vec<f64>), TrainError> {
    let n_wire = view.control_descriptors.len();
    let mut seen_ids = BTreeSet::new();
    let mut seen_wires = BTreeSet::new();
    let mut fixed_stars = Vec::with_capacity(controls.len());
    let mut axes = Vec::new();

    for (orig_idx, ctrl) in controls.iter().enumerate() {
        if ctrl.control_id.is_empty() || !seen_ids.insert(ctrl.control_id.as_str()) {
            return Err(TrainError::InvalidControlSpec {
                context: "panama",
                detail: "PANAMA control_id must be non-empty and unique",
            });
        }
        if ctrl.wire_ctrl_idx >= n_wire || !seen_wires.insert(ctrl.wire_ctrl_idx) {
            return Err(TrainError::InvalidControlSpec {
                context: "panama",
                detail: "PANAMA wire_ctrl_idx out of range or duplicated",
            });
        }
        match &ctrl.kind {
            PanamaControlKind::Range {
                safe_min,
                safe_max,
                star_center,
            } => {
                if !safe_min.is_finite()
                    || !safe_max.is_finite()
                    || !star_center.is_finite()
                    || safe_max < safe_min
                {
                    return Err(TrainError::InvalidControlSpec {
                        context: "panama",
                        detail: "PANAMA range bounds or star_center invalid",
                    });
                }
                fixed_stars.push(*star_center);
                if safe_max - safe_min > PANAMA_DELTA_TOL {
                    let desc = ControlClampDescriptor::from_wire(
                        &view.control_descriptors[ctrl.wire_ctrl_idx],
                    );
                    let mid = 0.5 * (safe_min + safe_max);
                    let u_star_norm = ((*star_center as f32) * desc.scale_norm + desc.offset_norm)
                        .clamp(-1.0, 1.0);
                    let star_lobatto = desc.eval_lobatto_basis(u_star_norm);
                    axes.push(VaryingAxis {
                        orig_idx,
                        control_id: ctrl.control_id.clone(),
                        star_center_val: *star_center,
                        kind: AxisKind::Range {
                            safe_min: *safe_min,
                            safe_max: *safe_max,
                            choices: [*safe_min, mid, *safe_max],
                            desc,
                            u_star_norm,
                            star_lobatto,
                        },
                    });
                }
            }
            PanamaControlKind::Switch {
                active_options,
                declared_option_names,
                star_center_option,
            } => {
                let mut opts = active_options.clone();
                opts.sort_unstable();
                opts.dedup();
                if opts.is_empty() {
                    return Err(TrainError::InvalidControlSpec {
                        context: "panama",
                        detail: "PANAMA switch must have >= 1 active option",
                    });
                }
                fixed_stars.push(f64::from(*star_center_option));
                if opts.len() >= 2 {
                    let grid_opts = opts.iter().copied().take(3).collect();
                    axes.push(VaryingAxis {
                        orig_idx,
                        control_id: ctrl.control_id.clone(),
                        star_center_val: f64::from(*star_center_option),
                        kind: AxisKind::Switch {
                            sorted_active_opts: opts,
                            grid_opts,
                            declared_option_names: declared_option_names.clone(),
                            star_center_opt: *star_center_option,
                        },
                    });
                }
            }
        }
    }

    axes.sort_by(|a, b| a.control_id.cmp(&b.control_id));
    Ok((axes, fixed_stars))
}

fn axis_grid_choices(axis: &VaryingAxis) -> Vec<f64> {
    match &axis.kind {
        AxisKind::Range { choices, .. } => choices.to_vec(),
        AxisKind::Switch { grid_opts, .. } => grid_opts.iter().map(|&o| f64::from(o)).collect(),
    }
}

fn compute_norm_stddevs(axes: &[VaryingAxis], sorted_rows: &[&PanamaSampleRow]) -> Vec<f64> {
    let n = sorted_rows.len() as f64;
    axes.iter()
        .map(|ax| {
            let mut sum = 0.0_f64;
            for r in sorted_rows {
                sum += normalize_axis_value(ax, r.control_values[ax.orig_idx]);
            }
            let mean = sum / n;
            let mut var = 0.0_f64;
            for r in sorted_rows {
                let d = normalize_axis_value(ax, r.control_values[ax.orig_idx]) - mean;
                var += d * d;
            }
            (var / n).sqrt()
        })
        .collect()
}

/// Generates candidate tuples (`[u_0, ..., u_{P_var-1}]`) for `P_var <= 5` (full Cartesian grid)
/// or `P_var > 5` (deduplicated union of all 2D pairs + top-6 normalized-stddev 3D triples).
pub(super) fn generate_candidate_grid(
    axes: &[VaryingAxis],
    sorted_rows: &[&PanamaSampleRow],
) -> Vec<Vec<f64>> {
    let p_var = axes.len();
    let grids: Vec<Vec<f64>> = axes.iter().map(axis_grid_choices).collect();
    let mut seen: BTreeSet<Vec<i64>> = BTreeSet::new();
    let mut out = Vec::new();
    let mut push_unique = |cand: Vec<f64>| {
        let key = quantize_tuple(axes, &cand);
        if seen.insert(key) {
            out.push(cand);
        }
    };

    if p_var <= PANAMA_FULL_GRID_MAX_PVAR {
        let mut idx = vec![0_usize; p_var];
        loop {
            let cand: Vec<f64> = (0..p_var).map(|c| grids[c][idx[c]]).collect();
            push_unique(cand);
            let mut pos = p_var;
            let mut advanced = false;
            while pos > 0 {
                pos -= 1;
                idx[pos] += 1;
                if idx[pos] < grids[pos].len() {
                    advanced = true;
                    break;
                }
                idx[pos] = 0;
            }
            if !advanced {
                break;
            }
        }
        return out;
    }

    let star: Vec<f64> = axes.iter().map(|ax| ax.star_center_val).collect();
    for i in 0..p_var {
        for j in (i + 1)..p_var {
            for &vi in &grids[i] {
                for &vj in &grids[j] {
                    let mut cand = star.clone();
                    cand[i] = vi;
                    cand[j] = vj;
                    push_unique(cand);
                }
            }
        }
    }

    let s = compute_norm_stddevs(axes, sorted_rows);
    let mut triples: Vec<(i64, usize, usize, usize)> = Vec::new();
    for i in 0..p_var {
        for j in (i + 1)..p_var {
            for k in (j + 1)..p_var {
                let score = round_half_away_i64((s[i] + s[j] + s[k]) * 1.0e6);
                triples.push((score, i, j, k));
            }
        }
    }
    triples.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| (a.1, a.2, a.3).cmp(&(b.1, b.2, b.3)))
    });
    for &(_, i, j, k) in triples.iter().take(PANAMA_TOP_TRIPLES_COUNT) {
        for &vi in &grids[i] {
            for &vj in &grids[j] {
                for &vk in &grids[k] {
                    let mut cand = star.clone();
                    cand[i] = vi;
                    cand[j] = vj;
                    cand[k] = vk;
                    push_unique(cand);
                }
            }
        }
    }

    out
}

/// Read-only shared context passed into `PANAMA` candidate exclusion checks.
pub(super) struct ExclusionCtx<'a> {
    pub view: &'a DnamBinView<'a>,
    pub controls: &'a [PanamaControlDesc],
    pub axes: &'a [VaryingAxis],
    pub fixed_stars: &'a [f64],
    pub sample_var_rows: &'a [Vec<f64>],
    pub delta_32: &'a [f64; STAGE_TARGET_SLOTS],
}

/// Checks `PANAMA` exclusions `(a)` (`d_inf <= 1e-3`), `(b)` (`conditional_bounds`), and
/// `(c)` (static `ControlBus(u)` vs raw `StageTargetBounds` + pre-clamp `[CLAMP-DYNAMIC]`).
pub(super) fn candidate_passes_exclusions(
    ctx: &ExclusionCtx<'_>,
    cand_var: &[f64],
    bus: &mut ControlBus<'_, DefaultMathBackend>,
) -> Result<bool, TrainError> {
    if ctx
        .sample_var_rows
        .iter()
        .any(|sr| d_inf(ctx.axes, cand_var, sr) <= PANAMA_EXCLUSION_DINF)
    {
        return Ok(false);
    }

    let mut all_vals = ctx.fixed_stars.to_vec();
    for (ax, &v) in ctx.axes.iter().zip(cand_var.iter()) {
        all_vals[ax.orig_idx] = v;
    }

    let n_wire = ctx.view.control_descriptors.len();
    let mut active_switches = vec![0_u8; n_wire];
    for (idx, ctrl) in ctx.controls.iter().enumerate() {
        if matches!(ctrl.kind, PanamaControlKind::Switch { .. }) {
            active_switches[ctrl.wire_ctrl_idx] = decode_switch_opt(all_vals[idx]);
        }
    }

    for (idx, ctrl) in ctx.controls.iter().enumerate() {
        if matches!(ctrl.kind, PanamaControlKind::Range { .. }) {
            let desc = ControlClampDescriptor::from_wire(
                &ctx.view.control_descriptors[ctrl.wire_ctrl_idx],
            );
            let (eff_min, eff_max) = desc.effective_bounds(&active_switches);
            let u_c = all_vals[idx];
            let clamped = u_c.clamp(f64::from(eff_min), f64::from(eff_max));
            if (u_c - clamped).abs() > PANAMA_COND_BOUND_TOL {
                return Ok(false);
            }
        }
    }

    let updates: Vec<(usize, f32)> = ctx
        .controls
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.wire_ctrl_idx, all_vals[idx] as f32))
        .collect();
    let (static_targets, static_z_stage) = eval_static_control_bus_targets(bus, &updates)?;
    let report = check_corner_dynamic_stability(
        ctx.view,
        ctx.view.retained_tier,
        &static_targets,
        &static_z_stage,
        ctx.delta_32,
        true,
    );
    Ok(!report
        .violations
        .iter()
        .any(DynamicCornerViolation::is_panama_exclusion))
}

/// Deterministic total order on `(score_q, tuple_q)` (`score_q` desc, then `tuple_q` asc).
pub(super) fn compare_scored_candidates(
    a: &(i64, Vec<i64>, Vec<f64>, f64),
    b: &(i64, Vec<i64>, Vec<f64>, f64),
) -> Ordering {
    b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1))
}

/// Formats `"[PANAMA] Corner (<id>=<val>, ...) (sigma2=<x.xxxx>)"` over controls with
/// `Delta_c > 1e-6` from `StarCenter` in ascending `control_id` order.
pub(super) fn format_panama_prescription(
    axes: &[VaryingAxis],
    cand_var: &[f64],
    sigma2: f64,
) -> String {
    let mut parts = Vec::new();
    for (ax, &val) in axes.iter().zip(cand_var.iter()) {
        if delta_axis(ax, val, ax.star_center_val) <= PANAMA_DELTA_TOL {
            continue;
        }
        let formatted_val = match &ax.kind {
            AxisKind::Range { .. } => format_f64_4dp_no_neg_zero(val),
            AxisKind::Switch {
                declared_option_names,
                ..
            } => {
                let opt = usize::from(decode_switch_opt(val));
                let label = declared_option_names
                    .get(opt)
                    .map_or_else(|| opt.to_string(), Clone::clone);
                format!("'{label}'")
            }
        };
        parts.push(format!("{}={formatted_val}", ax.control_id));
    }
    format!(
        "[PANAMA] Corner ({}) (sigma2={})",
        parts.join(", "),
        format_f64_4dp_no_neg_zero(sigma2),
    )
}
