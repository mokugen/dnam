//! Phase-1 `[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, and `[CLAMP-STABILITY]` control-space audit
//! (`ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//! `ref[train.audit.phase1.clamp-stability]`).

use dnam_kernels::{MAX_LOOP_GAIN_F32, TANK_MAX_G_LIN_F32};
use dnam_vm::bin::{ConditionalClampRuleWire, ControlClampDescriptorWire};
use dnam_vm::control::descriptor::{ControlClampDescriptor, compute_scale_offset_norm};
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16, U32};

use crate::error::TrainError;

use super::super::stability::{MIN_DELAY_CROSSBAR_SAMP, MIN_DELAY_STANDALONE_SAMP};
use super::{ControlKindSpec, ControlSpec, Phase1ClampOutput, StageStabilityConstraint};

/// Upper exact Cartesian corner enumeration ceiling (`3^9`).
const MAX_EXACT_CORNERS: usize = 19_683;

fn build_norm_range_choices(obs_min: f64, obs_max: f64, decl_min: f64, decl_max: f64) -> [f64; 3] {
    let span = (decl_max - decl_min).max(1.0e-12);
    let to_norm = |u: f64| -> f64 { (2.0 * (u - decl_min) / span - 1.0).clamp(-1.0, 1.0) };
    let n_lo = to_norm(obs_min);
    let n_hi = to_norm(obs_max);
    [n_lo, 0.5 * (n_lo + n_hi), n_hi]
}

fn tighten_or_push_conditional_rule(
    rules: &mut Vec<ConditionalClampRuleWire>,
    fb_idx: usize,
    hazard_switches: &[(usize, u8)],
    rule_op: u8,
    phys_req: f32,
    warnings: &mut Vec<String>,
) -> Result<(), TrainError> {
    if hazard_switches.len() == 1 {
        let (sw_c, sw_opt) = hazard_switches[0];
        for idx in 0..rules.len() {
            let preceded_by_and = idx > 0 && rules[idx - 1].rule_op == 3;
            let r = &rules[idx];
            if !preceded_by_and
                && usize::from(r.switch_ctrl_idx.get()) == sw_c
                && r.switch_opt_idx == sw_opt
                && r.rule_op == rule_op
            {
                let cur = r.bound_val.get();
                let next = if rule_op == 1 {
                    cur.min(phys_req)
                } else {
                    cur.max(phys_req)
                };
                if next != cur {
                    rules[idx].bound_val = F32::new(next);
                    warnings.push(format!(
                        "[CLAMP-STABILITY] ctrl[{fb_idx}] tightened rule (sw={sw_c}:{sw_opt}, op={rule_op}, bound={next:.3})"
                    ));
                }
                return Ok(());
            }
        }
        if rules.len() >= 3 {
            return Err(TrainError::ConditionalClampOverflow {
                control_index: fb_idx,
                required_rules: rules.len() + 1,
                max_rules: 3,
            });
        }
        rules.push(ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(sw_c as u16),
            switch_opt_idx: sw_opt,
            rule_op,
            bound_val: F32::new(phys_req),
        });
        warnings.push(format!(
            "[CLAMP-STABILITY] ctrl[{fb_idx}] conditional rule (sw={sw_c}:{sw_opt}, op={rule_op}, bound={phys_req:.3})"
        ));
        return Ok(());
    }

    let need = hazard_switches.len();
    if let Some(max_start) = rules.len().checked_sub(need) {
        for start in 0..=max_start {
            if start > 0 && rules[start - 1].rule_op == 3 {
                continue;
            }
            let end = start + need;
            let matches = hazard_switches
                .iter()
                .enumerate()
                .all(|(k, &(sw_c, sw_opt))| {
                    let r = &rules[start + k];
                    let exp_op = if k + 1 < need { 3 } else { rule_op };
                    usize::from(r.switch_ctrl_idx.get()) == sw_c
                        && r.switch_opt_idx == sw_opt
                        && r.rule_op == exp_op
                });
            if matches {
                let term = &mut rules[end - 1];
                let cur = term.bound_val.get();
                term.bound_val = F32::new(if rule_op == 1 {
                    cur.min(phys_req)
                } else {
                    cur.max(phys_req)
                });
                return Ok(());
            }
        }
    }
    if rules.len() + need > 3 {
        return Err(TrainError::ConditionalClampOverflow {
            control_index: fb_idx,
            required_rules: rules.len() + need,
            max_rules: 3,
        });
    }
    for (k, &(sw_c, sw_opt)) in hazard_switches.iter().enumerate() {
        let is_last = k + 1 == need;
        rules.push(ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(sw_c as u16),
            switch_opt_idx: sw_opt,
            rule_op: if is_last { rule_op } else { 3 },
            bound_val: F32::new(if is_last { phys_req } else { 0.0 }),
        });
    }
    warnings.push(format!(
        "[CLAMP-STABILITY] ctrl[{fb_idx}] conjunction rule ({need} switches, op={rule_op}, bound={phys_req:.3})"
    ));
    Ok(())
}

struct FeedbackHazardParams {
    fb_idx: usize,
    val: f64,
    limit_abs: f64,
    slope: f64,
}

fn check_feedback_hazard(
    specs: &[ControlSpec],
    corner: &[(f64, u8)],
    hazard_switches: &[(usize, u8)],
    hazard: FeedbackHazardParams,
    safe_bounds_mut: &mut [(f64, f64)],
    rules_mut: &mut [Vec<ConditionalClampRuleWire>],
    warnings: &mut Vec<String>,
) -> Result<(), TrainError> {
    let FeedbackHazardParams {
        fb_idx,
        val,
        limit_abs,
        slope,
    } = hazard;
    if fb_idx >= specs.len() || fb_idx >= corner.len() {
        return Err(TrainError::InvalidControlSpec {
            context: "check_feedback_hazard",
            detail: "feedback control index out of bounds",
        });
    }
    if val.abs() < limit_abs || slope.abs() <= 1.0e-12 {
        return Ok(());
    }
    let ControlKindSpec::Range {
        declared_min,
        declared_max,
        ..
    } = specs[fb_idx].kind
    else {
        return Ok(());
    };
    let max_safe = limit_abs - 1.0e-4;
    let target_val = if val >= 0.0 { max_safe } else { -max_safe };
    let nv_curr = corner[fb_idx].0;
    let base_no_fb = val - slope * nv_curr;
    let nv_req = (target_val - base_no_fb) / slope;
    let phys_req = (declared_min + 0.5 * (nv_req + 1.0) * (declared_max - declared_min))
        .clamp(safe_bounds_mut[fb_idx].0, safe_bounds_mut[fb_idx].1);
    let is_upper_bound = (val >= 0.0 && slope > 0.0) || (val < 0.0 && slope < 0.0);
    let rule_op = if is_upper_bound { 1 } else { 2 };

    if hazard_switches.is_empty() {
        if is_upper_bound && phys_req < safe_bounds_mut[fb_idx].1 {
            safe_bounds_mut[fb_idx].1 = phys_req;
        } else if !is_upper_bound && phys_req > safe_bounds_mut[fb_idx].0 {
            safe_bounds_mut[fb_idx].0 = phys_req;
        }
        return Ok(());
    }
    tighten_or_push_conditional_rule(
        &mut rules_mut[fb_idx],
        fb_idx,
        hazard_switches,
        rule_op,
        phys_req as f32,
        warnings,
    )
}

fn check_corner(
    specs: &[ControlSpec],
    constraint: &StageStabilityConstraint,
    min_delay: f64,
    corner: &[(f64, u8)],
    safe_bounds_mut: &mut [(f64, f64)],
    rules_mut: &mut [Vec<ConditionalClampRuleWire>],
    warnings: &mut Vec<String>,
) -> Result<(), TrainError> {
    let mut tau = constraint.base_tau;
    let mut g_lin = constraint.base_g_lin;
    let mut g_loop = constraint.base_loop_gain;
    let mut hazard_switches: Vec<(usize, u8)> = Vec::new();

    for (i, &(nv, opt)) in corner.iter().enumerate() {
        if matches!(specs[i].kind, ControlKindSpec::Range { .. }) {
            tau += constraint.delay_slopes.get(i).copied().unwrap_or(0.0) * nv;
            g_lin += constraint.g_lin_slopes.get(i).copied().unwrap_or(0.0) * nv;
            g_loop += constraint.loop_gain_slopes.get(i).copied().unwrap_or(0.0) * nv;
        } else {
            for &(sw_c, sw_opt, extra) in &constraint.switch_hazards {
                if sw_c == i && opt == sw_opt {
                    g_loop += extra;
                    if !hazard_switches.contains(&(sw_c, sw_opt)) {
                        hazard_switches.push((sw_c, sw_opt));
                    }
                }
            }
        }
    }

    if let Some(d_idx) = constraint.delay_ctrl_idx
        && d_idx >= specs.len()
    {
        return Err(TrainError::InvalidControlSpec {
            context: "check_corner",
            detail: "delay control index out of bounds",
        });
    }
    if tau < min_delay
        && let Some(d_idx) = constraint.delay_ctrl_idx
        && let ControlKindSpec::Range {
            declared_min,
            declared_max,
            ..
        } = specs[d_idx].kind
    {
        let slope = constraint.delay_slopes.get(d_idx).copied().unwrap_or(0.0);
        if slope.abs() > 1.0e-12 {
            let nv_curr = corner[d_idx].0;
            let tau_without = tau - slope * nv_curr;
            let nv_req = (min_delay - tau_without) / slope;
            let phys_req = declared_min + 0.5 * (nv_req + 1.0) * (declared_max - declared_min);
            if slope > 0.0 && phys_req > safe_bounds_mut[d_idx].0 {
                safe_bounds_mut[d_idx].0 = phys_req.min(safe_bounds_mut[d_idx].1);
                warnings.push(format!(
                    "[CLAMP-STABILITY] ctrl[{d_idx}] tightened min_floor -> {:.3} (tau >= {min_delay:.1})",
                    safe_bounds_mut[d_idx].0
                ));
            } else if slope < 0.0 && phys_req < safe_bounds_mut[d_idx].1 {
                safe_bounds_mut[d_idx].1 = phys_req.max(safe_bounds_mut[d_idx].0);
                warnings.push(format!(
                    "[CLAMP-STABILITY] ctrl[{d_idx}] tightened max_ceil -> {:.3} (tau >= {min_delay:.1})",
                    safe_bounds_mut[d_idx].1
                ));
            }
        }
    }

    if let Some(fb_idx) = constraint.feedback_ctrl_idx {
        let loop_slope = constraint
            .loop_gain_slopes
            .get(fb_idx)
            .copied()
            .unwrap_or(0.0);
        check_feedback_hazard(
            specs,
            corner,
            &hazard_switches,
            FeedbackHazardParams {
                fb_idx,
                val: g_loop,
                limit_abs: f64::from(MAX_LOOP_GAIN_F32),
                slope: loop_slope,
            },
            safe_bounds_mut,
            rules_mut,
            warnings,
        )?;
        let glin_slope = constraint.g_lin_slopes.get(fb_idx).copied().unwrap_or(0.0);
        check_feedback_hazard(
            specs,
            corner,
            &[],
            FeedbackHazardParams {
                fb_idx,
                val: g_lin,
                limit_abs: f64::from(TANK_MAX_G_LIN_F32),
                slope: glin_slope,
            },
            safe_bounds_mut,
            rules_mut,
            warnings,
        )?;
    }
    Ok(())
}

fn enumerate_corners_and_audit(
    specs: &[ControlSpec],
    constraint: &StageStabilityConstraint,
    safe_bounds: &mut [(f64, f64)],
    cond_rules: &mut [Vec<ConditionalClampRuleWire>],
    warnings: &mut Vec<String>,
) -> Result<usize, TrainError> {
    let min_delay = if constraint.crossbar_coupled {
        MIN_DELAY_CROSSBAR_SAMP
    } else {
        MIN_DELAY_STANDALONE_SAMP
    };
    let n_ctrls = specs.len();
    if n_ctrls == 0 {
        return Ok(1);
    }

    let mut axes: Vec<Vec<(f64, u8)>> = Vec::with_capacity(n_ctrls);
    let mut est_product = 1_usize;
    for (idx, sp) in specs.iter().enumerate() {
        match sp.kind {
            ControlKindSpec::Range {
                declared_min,
                declared_max,
                ..
            } => {
                let (s_lo, s_hi) = safe_bounds[idx];
                let pts = build_norm_range_choices(s_lo, s_hi, declared_min, declared_max);
                axes.push(vec![(pts[0], 0), (pts[1], 1), (pts[2], 2)]);
                est_product = est_product.saturating_mul(3);
            }
            ControlKindSpec::Switch {
                num_options,
                declared_mask,
                observed_mask,
                ..
            } => {
                let active_mask = declared_mask & observed_mask;
                let mut opts = Vec::new();
                for o in 0..num_options.min(16) {
                    if (active_mask & (1_u16 << o)) != 0 {
                        opts.push((f64::from(o), o));
                    }
                }
                if opts.is_empty() {
                    opts.push((0.0, 0));
                }
                est_product = est_product.saturating_mul(opts.len());
                axes.push(opts);
            }
        }
    }

    let mut audited = 0_usize;
    let mut corner = vec![(0.0_f64, 0_u8); n_ctrls];
    if est_product <= MAX_EXACT_CORNERS {
        let mut idx_vec = vec![0_usize; n_ctrls];
        loop {
            for (c, &ai) in idx_vec.iter().enumerate() {
                corner[c] = axes[c][ai];
            }
            check_corner(
                specs,
                constraint,
                min_delay,
                &corner,
                safe_bounds,
                cond_rules,
                warnings,
            )?;
            audited += 1;
            let mut pos = n_ctrls;
            while pos > 0 {
                pos -= 1;
                idx_vec[pos] += 1;
                if idx_vec[pos] < axes[pos].len() {
                    break;
                }
                idx_vec[pos] = 0;
                if pos == 0 {
                    return Ok(audited);
                }
            }
        }
    } else {
        for (c, ax) in axes.iter().enumerate() {
            corner[c] = ax[ax.len() / 2];
        }
        for i in 0..n_ctrls {
            for &vi in &axes[i] {
                corner[i] = vi;
                check_corner(
                    specs,
                    constraint,
                    min_delay,
                    &corner,
                    safe_bounds,
                    cond_rules,
                    warnings,
                )?;
                audited += 1;
                if audited >= MAX_EXACT_CORNERS {
                    return Ok(audited);
                }
            }
            for j in (i + 1)..n_ctrls {
                for &vi in &[axes[i][0], axes[i][axes[i].len() - 1]] {
                    for &vj in &[axes[j][0], axes[j][axes[j].len() - 1]] {
                        corner[i] = vi;
                        corner[j] = vj;
                        check_corner(
                            specs,
                            constraint,
                            min_delay,
                            &corner,
                            safe_bounds,
                            cond_rules,
                            warnings,
                        )?;
                        audited += 1;
                        if audited >= MAX_EXACT_CORNERS {
                            return Ok(audited);
                        }
                    }
                }
                corner[j] = axes[j][axes[j].len() / 2];
            }
            corner[i] = axes[i][axes[i].len() / 2];
        }
    }
    Ok(audited)
}

fn pack_star_anchored_nodes(
    default_phys: f64,
    s_min: f64,
    s_max: f64,
    lobatto_anchors: [f32; 3],
    scale_norm: f32,
    offset_norm: f32,
) -> [f32; 3] {
    let to_norm = |v: f64| -> f32 { ((v as f32) * scale_norm + offset_norm).clamp(-1.0, 1.0) };
    let u_star = to_norm(default_phys);
    let mut cands = Vec::with_capacity(5);
    for &phys_a in &lobatto_anchors {
        let na = (phys_a * scale_norm + offset_norm).clamp(-1.0, 1.0);
        cands.push(na);
    }
    cands.push(to_norm(s_min));
    cands.push(to_norm(s_max));

    let mut nodes = [u_star, u_star, u_star];
    let mut filled = 1_usize;
    for na in cands {
        if filled >= 3 {
            break;
        }
        if !nodes[..filled].iter().any(|&x| (x - na).abs() <= 1.0e-5) {
            nodes[filled] = na;
            filled += 1;
        }
    }
    nodes
}

/// Runs the Phase-1 audit (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`) over `specs`,
/// computing `scale_norm` / `offset_norm` on declared ranges while clamping `safe_min_base` /
/// `safe_max_base` to observed bounds and stability corners.
pub fn audit_phase1_controls(
    specs: &[ControlSpec],
    constraint: &StageStabilityConstraint,
) -> Result<Phase1ClampOutput, TrainError> {
    if specs.len() > 255 {
        return Err(TrainError::InvalidControlSpec {
            context: "audit_phase1_controls",
            detail: "control count exceeds v3.0 limit of 255",
        });
    }
    let mut warnings = Vec::new();
    let mut locked_controls = Vec::with_capacity(specs.len());
    let mut safe_ranges = Vec::with_capacity(specs.len());
    let mut active_masks = Vec::with_capacity(specs.len());
    let mut defaults = Vec::with_capacity(specs.len());

    for (idx, sp) in specs.iter().enumerate() {
        match sp.kind {
            ControlKindSpec::Range {
                declared_min,
                declared_max,
                default_val,
                obs_min,
                obs_max,
                unique_values,
                lobatto_anchors,
            } => {
                if !declared_min.is_finite()
                    || !declared_max.is_finite()
                    || declared_max <= declared_min
                    || !default_val.is_finite()
                    || !obs_min.is_finite()
                    || !obs_max.is_finite()
                    || obs_max < obs_min
                    || !lobatto_anchors.iter().all(|v| v.is_finite())
                {
                    return Err(TrainError::InvalidControlSpec {
                        context: "audit_phase1_controls",
                        detail: "non-finite or inverted range control bounds or anchors",
                    });
                }
                let s_min = obs_min.clamp(declared_min, declared_max);
                let s_max = obs_max.clamp(s_min, declared_max);
                let is_locked = unique_values <= 1;
                if s_min > declared_min || s_max < declared_max {
                    warnings.push(format!(
                        "[CLAMP-RANGE] ctrl[{idx}]: clamped declared [{declared_min:.3}, {declared_max:.3}] -> safe [{s_min:.3}, {s_max:.3}]"
                    ));
                }
                if is_locked {
                    warnings.push(format!(
                        "[CLAMP-RANGE] ctrl[{idx}]: single observed value -> locked K_p = 0"
                    ));
                }
                locked_controls.push(is_locked);
                safe_ranges.push((s_min, s_max));
                active_masks.push(0);
                defaults.push(default_val.clamp(s_min, s_max));
            }
            ControlKindSpec::Switch {
                num_options,
                default_opt,
                declared_mask,
                observed_mask,
            } => {
                if !(1..=16).contains(&num_options) {
                    return Err(TrainError::InvalidControlSpec {
                        context: "audit_phase1_controls",
                        detail: "switch option count must be in 1..=16",
                    });
                }
                let full_mask = u16::try_from((1_u32 << num_options) - 1).unwrap_or(u16::MAX);
                let obs_masked = declared_mask & observed_mask & full_mask;
                if obs_masked == 0 {
                    return Err(TrainError::InvalidControlSpec {
                        context: "audit_phase1_controls",
                        detail: "switch has zero valid observed options",
                    });
                }
                if obs_masked != (declared_mask & full_mask) {
                    warnings.push(format!(
                        "[CLAMP-SWITCH] ctrl[{idx}]: restricted mask {:#06x} -> observed {obs_masked:#06x}",
                        declared_mask & full_mask
                    ));
                }
                let def_cand = default_opt.min(num_options - 1).min(15);
                let def_opt = if (obs_masked & (1_u16 << def_cand)) != 0 {
                    def_cand
                } else {
                    obs_masked.trailing_zeros() as u8
                };
                locked_controls.push(obs_masked.count_ones() <= 1);
                safe_ranges.push((0.0, f64::from(num_options - 1)));
                active_masks.push(obs_masked);
                defaults.push(f64::from(def_opt));
            }
        }
    }

    let mut cond_rules = vec![Vec::<ConditionalClampRuleWire>::new(); specs.len()];
    let audited_corners = enumerate_corners_and_audit(
        specs,
        constraint,
        &mut safe_ranges,
        &mut cond_rules,
        &mut warnings,
    )?;

    let mut wire_descriptors = Vec::with_capacity(specs.len());
    for (idx, sp) in specs.iter().enumerate() {
        let mut wire = ControlClampDescriptorWire::new_zeroed();
        match sp.kind {
            ControlKindSpec::Range {
                declared_min,
                declared_max,
                unique_values,
                lobatto_anchors,
                ..
            } => {
                let (scale_norm, offset_norm) =
                    compute_scale_offset_norm(declared_min as f32, declared_max as f32);
                let (s_min, s_max) = safe_ranges[idx];
                let def_phys = defaults[idx].clamp(s_min, s_max);
                wire.ctrl_kind = 0;
                wire.lobatto_degree = unique_values.saturating_sub(1).min(3) as u8;
                wire.active_options_mask = U16::new(0);
                wire.default_val = F32::new(def_phys as f32);
                wire.safe_min_base = F32::new(s_min as f32);
                wire.safe_max_base = F32::new(s_max as f32);
                wire.scale_norm = F32::new(scale_norm);
                wire.offset_norm = F32::new(offset_norm);
                let nodes = pack_star_anchored_nodes(
                    def_phys,
                    s_min,
                    s_max,
                    lobatto_anchors,
                    scale_norm,
                    offset_norm,
                );
                for (a_slot, norm_a) in nodes.into_iter().enumerate() {
                    wire.anchor_nodes_norm[a_slot] = F32::new(norm_a);
                }
            }
            ControlKindSpec::Switch { num_options, .. } => {
                wire.ctrl_kind = 1;
                wire.lobatto_degree = num_options - 1;
                wire.active_options_mask = U16::new(active_masks[idx]);
                wire.default_val = F32::new(defaults[idx] as f32);
                wire.safe_min_base = F32::new(0.0);
                wire.safe_max_base = F32::new(f32::from(num_options - 1));
                wire.scale_norm = F32::new(1.0);
                wire.offset_norm = F32::new(0.0);
            }
        }
        for (r_slot, rule) in cond_rules[idx].iter().enumerate().take(3) {
            wire.cond_rules[r_slot] = *rule;
        }
        wire._reserved = U32::new(0);
        let desc = ControlClampDescriptor::from_wire(&wire);
        let (eff_lo, eff_hi) = desc.effective_bounds(&[]);
        if !eff_lo.is_finite() || !eff_hi.is_finite() || eff_lo > eff_hi {
            return Err(TrainError::InvalidControlSpec {
                context: "audit_phase1_controls",
                detail: "effective control bounds invalid after wire roundtrip",
            });
        }
        wire_descriptors.push(wire);
    }

    Ok(Phase1ClampOutput {
        wire_descriptors,
        locked_controls,
        safe_ranges,
        audited_corners,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase1_range_lobatto_degree_star_anchor_and_glin_conjunctions() {
        let specs = vec![
            ControlSpec {
                kind: ControlKindSpec::Range {
                    declared_min: 0.0,
                    declared_max: 10.0,
                    default_val: 5.0,
                    obs_min: 2.0,
                    obs_max: 8.0,
                    unique_values: 5,
                    lobatto_anchors: [2.0, 5.0, 8.0],
                },
            },
            ControlSpec {
                kind: ControlKindSpec::Switch {
                    num_options: 4,
                    default_opt: 0,
                    declared_mask: 0b1111,
                    observed_mask: 0b0111,
                },
            },
            ControlSpec {
                kind: ControlKindSpec::Range {
                    declared_min: 0.0,
                    declared_max: 10.0,
                    default_val: 4.0,
                    obs_min: 4.0,
                    obs_max: 4.0,
                    unique_values: 1,
                    lobatto_anchors: [4.0, 4.0, 4.0],
                },
            },
        ];
        let mut constraint = StageStabilityConstraint {
            delay_ctrl_idx: None,
            delay_slopes: vec![0.0; 3],
            base_tau: 16.0,
            base_g_lin: 0.6,
            g_lin_slopes: vec![0.9, 0.0, 0.0],
            switch_hazards: vec![(1, 2, 0.55)],
            base_loop_gain: 0.3,
            loop_gain_slopes: vec![0.3, 0.0, 0.0],
            feedback_ctrl_idx: Some(0),
            crossbar_coupled: true,
        };
        let p1 = audit_phase1_controls(&specs, &constraint).unwrap();
        // `ctrl[2]` has `unique_values == 1` -> `lobatto_degree == 0`.
        assert_eq!(p1.wire_descriptors[2].lobatto_degree, 0);
        // `ctrl[0]` pins `anchor_nodes_norm[0]` to StarCenter -> 0-ULP zero Lobatto basis at StarCenter.
        let desc0 = ControlClampDescriptor::from_wire(&p1.wire_descriptors[0]);
        let (_, u_star, _) = desc0.clamp_and_normalize(desc0.default_val, &[]);
        assert_eq!(desc0.eval_lobatto_basis(u_star), [0.0; 3]);
        // `g_lin` tightening + switch tightening both applied cleanly.
        assert!(p1.safe_ranges[0].1 < 8.0);
        assert_eq!(p1.wire_descriptors[0].cond_rules[0].rule_op, 1);

        // Multi-switch hazard uses `rule_op = 3` conjunctions + non-finite default rejection.
        let sw2 = ControlSpec {
            kind: ControlKindSpec::Switch {
                num_options: 2,
                default_opt: 0,
                declared_mask: 0b11,
                observed_mask: 0b11,
            },
        };
        let mut conj_specs = specs.clone();
        conj_specs.push(sw2);
        constraint.switch_hazards = vec![(1, 1, 0.4), (3, 1, 0.45)];
        constraint.g_lin_slopes = vec![0.0; 4];
        constraint.loop_gain_slopes = vec![0.3, 0.0, 0.0, 0.0];
        let p_conj = audit_phase1_controls(&conj_specs, &constraint).unwrap();
        assert_eq!(p_conj.wire_descriptors[0].cond_rules[0].rule_op, 3);
        assert_eq!(p_conj.wire_descriptors[0].cond_rules[1].rule_op, 1);

        let mut bad = specs;
        if let ControlKindSpec::Range {
            ref mut default_val,
            ..
        } = bad[0].kind
        {
            *default_val = f64::NAN;
        }
        assert!(matches!(
            audit_phase1_controls(&bad, &StageStabilityConstraint::default()),
            Err(TrainError::InvalidControlSpec { .. })
        ));

        // 3-switch conjunction followed by 2-switch suffix must fail closed (`ConditionalClampOverflow`),
        // while unconditional `g_lin` hazard under 2 switch hazards must not consume inline rule slots.
        let sw_pair = ControlSpec {
            kind: ControlKindSpec::Switch {
                num_options: 2,
                default_opt: 0,
                declared_mask: 0b11,
                observed_mask: 0b11,
            },
        };
        let suffix_specs = vec![
            ControlSpec {
                kind: ControlKindSpec::Range {
                    declared_min: 0.0,
                    declared_max: 1.0,
                    default_val: 0.5,
                    obs_min: 0.0,
                    obs_max: 1.0,
                    unique_values: 5,
                    lobatto_anchors: [0.0, 0.5, 1.0],
                },
            },
            sw_pair.clone(),
            sw_pair.clone(),
            sw_pair,
        ];
        let mut c_suffix = StageStabilityConstraint {
            feedback_ctrl_idx: Some(0),
            base_tau: 20.0,
            base_loop_gain: 0.1,
            loop_gain_slopes: vec![0.1, 0.0, 0.0, 0.0],
            switch_hazards: vec![(1, 0, 0.1), (2, 1, 0.45), (3, 1, 0.45)],
            ..Default::default()
        };
        assert!(matches!(
            audit_phase1_controls(&suffix_specs, &c_suffix),
            Err(TrainError::ConditionalClampOverflow {
                required_rules: 5,
                max_rules: 3,
                ..
            })
        ));

        // Unconditional `g_lin` with safe `g_loop` and 2 switch hazards must tighten unconditionally.
        c_suffix.base_g_lin = 0.6;
        c_suffix.g_lin_slopes = vec![0.9, 0.0, 0.0, 0.0];
        c_suffix.loop_gain_slopes = vec![0.0; 4];
        c_suffix.switch_hazards = vec![(1, 1, 0.1), (2, 1, 0.1)];
        let p_glin = audit_phase1_controls(&suffix_specs, &c_suffix).unwrap();
        assert!(p_glin.safe_ranges[0].1 < 1.0);
        assert!(
            p_glin.wire_descriptors[0]
                .cond_rules
                .iter()
                .all(|r| r.rule_op == 0)
        );
    }
}
