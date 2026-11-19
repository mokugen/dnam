//! Compilation of signed `controls_schema` definitions into cache-aligned
//! [`ControlClampDescriptor`] wire entries (`schema/header/clamp_compiler.rs`).

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::control::{ConditionalClampRule, ControlClampDescriptor};
use crate::schema::controls::{ControlKind, ControlSchemaDef};
use crate::schema::error::{DnamError, reject_if};

/// Compiles `controls_schema` into binary [`ControlClampDescriptor`] entries (`0..=255` controls).
pub(super) fn compile_controls_schema_descriptors(
    controls_schema: &[ControlSchemaDef],
) -> Result<Vec<ControlClampDescriptor>, DnamError> {
    if controls_schema.len() > u8::MAX as usize {
        return Err(DnamError::SchemaValidationError(
            "controls_schema exceeds u8::MAX (255) controls in binary preamble".to_string(),
        ));
    }
    let mut table = vec![ControlClampDescriptor::empty(); controls_schema.len()];

    for (p, ctrl) in controls_schema.iter().enumerate() {
        table[p] = match ctrl.kind {
            ControlKind::Range => compile_range_descriptor(controls_schema, ctrl)?,
            ControlKind::Switch => compile_switch_descriptor(ctrl)?,
        };
    }

    Ok(table)
}

fn compile_range_descriptor(
    controls_schema: &[ControlSchemaDef],
    ctrl: &ControlSchemaDef,
) -> Result<ControlClampDescriptor, DnamError> {
    let decl = ctrl.declared_range.ok_or_else(|| {
        DnamError::SchemaValidationError("range control missing declared_range".to_string())
    })?;
    if !decl[0].is_finite() || !decl[1].is_finite() {
        return Err(DnamError::SchemaValidationError(
            "range control declared_range must be finite".to_string(),
        ));
    }
    let safe = ctrl.safe_clamped_range.unwrap_or(decl);
    if !safe[0].is_finite() || !safe[1].is_finite() || safe[0] > safe[1] {
        return Err(DnamError::SchemaValidationError(
            "range control safe_clamped_range must be finite and ordered".to_string(),
        ));
    }
    let d = f64::from(decl[1]) - f64::from(decl[0]);
    reject_if(
        !d.is_finite() || d <= 1.0e-6,
        "range control declared_range span must be > 0",
    )?;
    let scale_f64 = 2.0 / d;
    let offset_f64 = -1.0 - f64::from(decl[0]) * scale_f64;
    let scale_norm = scale_f64 as f32;
    let offset_norm = offset_f64 as f32;
    reject_if(
        !scale_norm.is_finite() || !offset_norm.is_finite(),
        "range control normalization scale/offset overflowed f32",
    )?;
    let default_f64 = ctrl.default.as_f64().ok_or_else(|| {
        DnamError::SchemaValidationError("range control default must be numeric".to_string())
    })?;
    if !default_f64.is_finite() {
        return Err(DnamError::SchemaValidationError(
            "range control default must be finite".to_string(),
        ));
    }
    let default_val = default_f64 as f32;
    let degree = ctrl.lobatto_degree.unwrap_or(2).min(3);
    let mut anchors = [0.0_f32; 3];
    if let Some(ref nodes) = ctrl.anchor_nodes_norm {
        for (i, &val) in nodes.iter().take(3).enumerate() {
            anchors[i] = val.clamp(-1.0, 1.0);
        }
    } else {
        anchors[0] = (default_val * scale_norm + offset_norm).clamp(-1.0, 1.0);
        anchors[1] = -1.0;
        anchors[2] = 1.0;
    }

    let cond_rules = compile_conditional_rules(controls_schema, ctrl)?;

    let safe_min = safe[0].max(decl[0]);
    let safe_max = safe[1].min(decl[1]);
    if safe_min > safe_max {
        return Err(DnamError::SchemaValidationError(
            "range control safe_clamped_range does not intersect declared_range".to_string(),
        ));
    }

    Ok(ControlClampDescriptor {
        ctrl_kind: 0,
        lobatto_degree: degree,
        active_options_mask: 0x0001,
        default_val: default_val.clamp(safe_min, safe_max),
        safe_min_base: safe_min,
        safe_max_base: safe_max,
        scale_norm,
        offset_norm,
        anchor_nodes_norm: anchors,
        _reserved: 0,
        cond_rules,
    })
}

fn inward_round_bounds(
    min_opt: Option<f64>,
    max_opt: Option<f64>,
) -> Result<(Option<f32>, Option<f32>), DnamError> {
    if min_opt.is_none() && max_opt.is_none() {
        return Err(DnamError::SchemaValidationError(
            "conditional_bounds entry must specify min and/or max".to_string(),
        ));
    }
    let inward_min = if let Some(min_v) = min_opt {
        reject_if(!min_v.is_finite(), "conditional_bounds min must be finite")?;
        let mut mf = min_v as f32;
        if f64::from(mf) < min_v {
            mf = mf.next_up();
        }
        reject_if(!mf.is_finite(), "conditional_bounds min overflows f32")?;
        Some(mf)
    } else {
        None
    };
    let inward_max = if let Some(max_v) = max_opt {
        reject_if(!max_v.is_finite(), "conditional_bounds max must be finite")?;
        let mut mf = max_v as f32;
        if f64::from(mf) > max_v {
            mf = mf.next_down();
        }
        reject_if(!mf.is_finite(), "conditional_bounds max overflows f32")?;
        Some(mf)
    } else {
        None
    };
    if let (Some(lo), Some(hi)) = (min_opt, max_opt) {
        reject_if(lo > hi, "conditional_bounds min cannot exceed max")?;
    }
    if let (Some(lo), Some(hi)) = (inward_min, inward_max) {
        reject_if(
            lo > hi,
            "conditional_bounds inward f32 min cannot exceed max",
        )?;
    }
    Ok((inward_min, inward_max))
}

fn resolve_ordered_when(
    controls_schema: &[ControlSchemaDef],
    when: &[(alloc::string::String, alloc::string::String)],
) -> Result<Option<Vec<(u16, u8)>>, DnamError> {
    let mut pairs = Vec::with_capacity(when.len());
    for (sw_id, opt_id) in when {
        let Some(pair) = resolve_switch_pair(controls_schema, sw_id, opt_id) else {
            return Ok(None);
        };
        pairs.push(pair);
    }
    let mut prev_sw: Option<u16> = None;
    for &(sw_idx, _) in &pairs {
        if let Some(prev) = prev_sw
            && sw_idx <= prev
        {
            return Err(DnamError::SchemaValidationError(
                "conditional_bounds when conjunction must follow controls_schema order".to_string(),
            ));
        }
        prev_sw = Some(sw_idx);
    }
    Ok(Some(pairs))
}

fn compile_conditional_rules(
    controls_schema: &[ControlSchemaDef],
    ctrl: &ControlSchemaDef,
) -> Result<[ConditionalClampRule; 3], DnamError> {
    let mut cond_rules = [ConditionalClampRule::empty(); 3];
    let mut rule_idx = 0usize;

    for bound in &ctrl.conditional_bounds {
        reject_if(
            bound.when.is_empty(),
            "conditional_bounds entry missing selector conjunction",
        )?;
        let (inward_min, inward_max) = inward_round_bounds(bound.min, bound.max)?;
        let Some(pairs) = resolve_ordered_when(controls_schema, &bound.when)? else {
            continue;
        };

        let term_slots = usize::from(inward_max.is_some()) + usize::from(inward_min.is_some());
        let slots_needed = (pairs.len() - 1) + term_slots;
        if rule_idx + slots_needed > 3 {
            return Err(DnamError::SchemaValidationError(
                "conditional_bounds rules exceed 3 wire slots in ControlClampDescriptor".into(),
            ));
        }

        for &(sw_idx, opt_idx) in &pairs[..pairs.len() - 1] {
            cond_rules[rule_idx] = ConditionalClampRule {
                switch_ctrl_idx: sw_idx,
                switch_opt_idx: opt_idx,
                rule_op: 3,
                bound_val: 0.0,
            };
            rule_idx += 1;
        }
        let (last_sw, last_opt) = pairs[pairs.len() - 1];
        let (max_op, min_op) = if pairs.len() >= 2 && inward_max.is_some() && inward_min.is_some() {
            (5_u8, 6_u8)
        } else {
            (1_u8, 2_u8)
        };
        if let Some(hi) = inward_max {
            cond_rules[rule_idx] = ConditionalClampRule {
                switch_ctrl_idx: last_sw,
                switch_opt_idx: last_opt,
                rule_op: max_op,
                bound_val: hi,
            };
            rule_idx += 1;
        }
        if let Some(lo) = inward_min {
            cond_rules[rule_idx] = ConditionalClampRule {
                switch_ctrl_idx: last_sw,
                switch_opt_idx: last_opt,
                rule_op: min_op,
                bound_val: lo,
            };
            rule_idx += 1;
        }
    }

    Ok(cond_rules)
}

fn compile_switch_descriptor(ctrl: &ControlSchemaDef) -> Result<ControlClampDescriptor, DnamError> {
    let decl_opts = ctrl.declared_options.as_ref().ok_or_else(|| {
        DnamError::SchemaValidationError("switch control missing declared_options".to_string())
    })?;
    if decl_opts.is_empty() || decl_opts.len() > 16 {
        return Err(DnamError::SchemaValidationError(
            "switch control must have 1..=16 declared_options".to_string(),
        ));
    }
    if let Some(ref repr_opts) = ctrl.represented_switch_options {
        let mut prev: Option<u8> = None;
        for &opt_idx in repr_opts {
            if (opt_idx as usize) >= decl_opts.len() {
                return Err(DnamError::SchemaValidationError(
                    "represented_switch_options index out of range".to_string(),
                ));
            }
            if let Some(p) = prev
                && opt_idx <= p
            {
                return Err(DnamError::SchemaValidationError(
                    "represented_switch_options must be strictly increasing".to_string(),
                ));
            }
            prev = Some(opt_idx);
        }
    }
    let active_opts = ctrl.active_options.as_ref().unwrap_or(decl_opts);
    let mut mask: u16 = 0;
    for (idx, opt_id) in decl_opts.iter().enumerate() {
        if active_opts.iter().any(|a| a == opt_id) {
            mask |= 1_u16 << idx;
        }
    }
    if mask == 0 {
        mask = 1;
    }
    let default_str = ctrl.default.as_str().unwrap_or(decl_opts[0].as_str());
    let default_idx = decl_opts.iter().position(|o| o == default_str).unwrap_or(0);

    Ok(ControlClampDescriptor {
        ctrl_kind: 1,
        lobatto_degree: (decl_opts.len() as u8).saturating_sub(1),
        active_options_mask: mask,
        default_val: default_idx as f32,
        safe_min_base: 0.0,
        safe_max_base: (decl_opts.len() - 1) as f32,
        scale_norm: 1.0,
        offset_norm: 0.0,
        anchor_nodes_norm: [0.0; 3],
        _reserved: 0,
        cond_rules: [ConditionalClampRule::empty(); 3],
    })
}

fn resolve_switch_pair(
    controls_schema: &[ControlSchemaDef],
    sw_id: &str,
    opt_id: &str,
) -> Option<(u16, u8)> {
    for (idx, ctrl) in controls_schema.iter().enumerate() {
        if ctrl.id == sw_id
            && ctrl.kind == ControlKind::Switch
            && let Some(ref opts) = ctrl.declared_options
            && let Some(opt_idx) = opts.iter().position(|o| o == opt_id)
        {
            return Some((idx as u16, opt_idx as u8));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::wire::StructuredConditionalBound;
    use crate::schema::controls::ControlUiHints;
    use alloc::collections::BTreeMap;
    use alloc::string::String;

    #[test]
    fn test_compile_structured_conditional_bounds_and_represented_switch_options() {
        let controls = vec![
            ControlSchemaDef {
                id: String::from("gain"),
                name: String::from("Gain"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(1),
                anchor_nodes_norm: None,
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(3.0),
                conditional_bounds: vec![
                    StructuredConditionalBound {
                        when: vec![(String::from("sw"), String::from("off"))],
                        min: None,
                        max: Some(8.0),
                    },
                    StructuredConditionalBound {
                        when: vec![(String::from("sw"), String::from("on"))],
                        min: Some(2.0),
                        max: None,
                    },
                ],
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("sw"),
                name: String::from("Sw"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("off"),
                conditional_bounds: Vec::new(),
                represented_switch_options: Some(vec![1]),
                declared_options: Some(vec![String::from("off"), String::from("on")]),
                active_options: Some(vec![String::from("off")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let compiled = compile_controls_schema_descriptors(&controls).unwrap();
        assert_eq!(compiled[0].cond_rules[0].rule_op, 1);
        assert_eq!(compiled[0].cond_rules[1].rule_op, 2);
        assert_eq!(compiled[1].active_options_mask, 1);
    }

    fn make_switch(id: &str, opts: &[&str]) -> ControlSchemaDef {
        ControlSchemaDef {
            id: String::from(id),
            name: String::from(id),
            kind: ControlKind::Switch,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: None,
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!(opts[0]),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: Some(opts.iter().map(|s| String::from(*s)).collect()),
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }
    }

    fn make_range(
        id: &str,
        decl: [f32; 2],
        bounds: Vec<StructuredConditionalBound>,
    ) -> ControlSchemaDef {
        ControlSchemaDef {
            id: String::from(id),
            name: String::from(id),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(1),
            anchor_nodes_norm: None,
            declared_range: Some(decl),
            safe_clamped_range: Some(decl),
            default: serde_json::json!(decl[0]),
            conditional_bounds: bounds,
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn test_multi_switch_conjunctions_inward_f64_and_fail_closed_overflow() {
        // 2-switch conjunction with both min & max -> rule_ops [3, 5, 6]
        let controls_2 = vec![
            make_range(
                "gain",
                [0.0, 10.0],
                vec![StructuredConditionalBound {
                    when: vec![
                        (String::from("sw1"), String::from("on")),
                        (String::from("sw2"), String::from("hi")),
                    ],
                    min: Some(2.0 + 1.0e-9),
                    max: Some(8.0 - 1.0e-9),
                }],
            ),
            make_switch("sw1", &["off", "on"]),
            make_switch("sw2", &["lo", "hi"]),
        ];
        let compiled_2 = compile_controls_schema_descriptors(&controls_2).unwrap();
        let d2 = &compiled_2[0];
        assert_eq!(
            [
                d2.cond_rules[0].rule_op,
                d2.cond_rules[1].rule_op,
                d2.cond_rules[2].rule_op,
            ],
            [3, 5, 6]
        );
        // Inward f32 rounding: min >= 2.0 + 1e-9, max <= 8.0 - 1e-9
        assert!(f64::from(d2.cond_rules[1].bound_val) <= 8.0 - 1.0e-9);
        assert!(f64::from(d2.cond_rules[2].bound_val) >= 2.0 + 1.0e-9);
        assert_eq!(d2.clamp_and_normalize(9.5, &[0, 1, 0]).0, 9.5);
        assert!(d2.clamp_and_normalize(9.5, &[0, 1, 1]).0 < 8.0);
        assert!(d2.clamp_and_normalize(0.5, &[0, 1, 1]).0 > 2.0);

        // 3-switch conjunction with max -> rule_ops [3, 3, 1]
        let controls_3 = vec![
            make_range(
                "gain",
                [0.0, 10.0],
                vec![StructuredConditionalBound {
                    when: vec![
                        (String::from("sw1"), String::from("on")),
                        (String::from("sw2"), String::from("hi")),
                        (String::from("sw3"), String::from("b")),
                    ],
                    min: None,
                    max: Some(4.5),
                }],
            ),
            make_switch("sw1", &["off", "on"]),
            make_switch("sw2", &["lo", "hi"]),
            make_switch("sw3", &["a", "b"]),
        ];
        let compiled_3 = compile_controls_schema_descriptors(&controls_3).unwrap();
        let d3 = &compiled_3[0];
        assert_eq!(
            [
                d3.cond_rules[0].rule_op,
                d3.cond_rules[1].rule_op,
                d3.cond_rules[2].rule_op,
            ],
            [3, 3, 1]
        );
        assert_eq!(d3.clamp_and_normalize(9.0, &[0, 1, 1, 0]).0, 9.0);
        assert_eq!(d3.clamp_and_normalize(9.0, &[0, 1, 1, 1]).0, 4.5);

        // Overflow (> 3 wire slots) must fail closed
        let mut overflow_controls = controls_3.clone();
        overflow_controls[0].conditional_bounds[0].min = Some(1.0);
        let err = compile_controls_schema_descriptors(&overflow_controls).unwrap_err();
        assert!(
            matches!(err, DnamError::SchemaValidationError(msg) if msg.contains("3 wire slots"))
        );

        // Empty min+max must fail closed
        let mut empty_bound = controls_2.clone();
        empty_bound[0].conditional_bounds[0].min = None;
        empty_bound[0].conditional_bounds[0].max = None;
        assert!(compile_controls_schema_descriptors(&empty_bound).is_err());

        // f64 normalization span check: huge range [-1e38, 1e38] does not overflow f32 span
        let huge = vec![make_range("wide", [-1.0e38, 1.0e38], Vec::new())];
        let huge_c = compile_controls_schema_descriptors(&huge).unwrap();
        assert!(huge_c[0].scale_norm.is_finite() && huge_c[0].offset_norm.is_finite());
    }
}
