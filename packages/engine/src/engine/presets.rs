//! Stack-allocated preset resolution and calibrated control-role side effects for [`DnamEngine`].

use crate::control::ControlClampDescriptor;
use crate::engine::{DnamEngine, EngineBackend};
use crate::schema::{ControlCalibrationRole, ControlKind, DnamError, DnamJsonHeader};

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Applies a preset by string identifier (`presets[*].id`).
    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        let batch = resolve_preset_batch_by_id(
            self.header.as_ref(),
            &self.control_bus.descriptors,
            preset_id,
        )?;
        self.apply_resolved_preset_batch(&batch);
        Ok(())
    }

    /// Applies a preset by index without any heap allocations (`ref[dsp.ctrl.op-m.spsc-ingress]`).
    pub fn apply_preset_by_index(&mut self, preset_index: usize) -> Result<(), DnamError> {
        let batch = resolve_preset_batch_by_index(
            self.header.as_ref(),
            &self.control_bus.descriptors,
            preset_index,
        )?;
        self.apply_resolved_preset_batch(&batch);
        Ok(())
    }

    /// Real-time preset application by index returning `bool` (zero `DnamError` heap allocation or drop glue).
    #[inline]
    pub(crate) fn apply_preset_by_index_rt(&mut self, preset_index: usize) -> bool {
        let Some(batch) = resolve_preset_batch_by_index_opt(
            self.header.as_ref(),
            &self.control_bus.descriptors,
            preset_index,
        ) else {
            return false;
        };
        self.apply_resolved_preset_batch(&batch);
        true
    }

    fn apply_resolved_preset_batch(&mut self, batch: &ResolvedPresetBatch) {
        for &(idx, val, is_sw) in &batch.default_restores[..batch.restore_count] {
            if is_sw {
                self.set_control_immediate(idx, val);
            } else {
                self.set_control_value(idx, val);
            }
        }
        for &(idx, val) in &batch.switch_updates[..batch.switch_count] {
            self.set_control_immediate(idx, val);
        }
        for &(idx, val) in &batch.range_updates[..batch.range_count] {
            self.set_control_value(idx, val);
        }
        let extra_len = self
            .header
            .as_ref()
            .map_or(0, |h| h.controls_schema.len().saturating_sub(256));
        if extra_len > 0 {
            let p_idx = batch.preset_index as usize;
            let total_len = (256 + extra_len).min(self.control_bus.raw_values.len());
            if self.control_bus.num_controls < total_len {
                self.control_bus.num_controls = total_len;
            }
            for idx in 256..total_len {
                let Some((is_sw, target_val)) = self.header.as_ref().and_then(|hdr| {
                    let preset = hdr.presets.get(p_idx)?;
                    let def = hdr.controls_schema.get(idx)?;
                    let is_sw = def.kind == ControlKind::Switch;
                    if let Some(val) = preset.controls.get(&def.id) {
                        if let Some(f) = val.as_f64() {
                            Some((is_sw, f as f32))
                        } else if let Some(s) = val.as_str()
                            && let Some(ref opts) = def.declared_options
                            && let Some(opt_idx) = opts.iter().position(|o| o == s)
                        {
                            Some((true, opt_idx as f32))
                        } else {
                            None
                        }
                    } else {
                        let def_val = self
                            .control_bus
                            .descriptors
                            .get(idx)
                            .map(|d| d.default_val)
                            .unwrap_or(0.0);
                        Some((is_sw, def_val))
                    }
                }) else {
                    continue;
                };
                if is_sw {
                    self.set_control_immediate(idx, target_val);
                } else {
                    self.set_control_value(idx, target_val);
                }
            }
        }
    }
}

/// Applies calibrated control role side effects (`StereoMode`, `StereoWidth`, `FreezeLatch`)
/// shared between [`DnamEngine`] and [`crate::host::AcceptedStateShadow`].
pub(crate) fn apply_calibrated_control_role(
    header: Option<&DnamJsonHeader>,
    ctrl_idx: usize,
    physical_value: f32,
    stereo_width: &mut f32,
    is_frozen: &mut bool,
    supports_freeze: bool,
) {
    let Some(hdr) = header else {
        return;
    };
    let Some(ctrl_def) = hdr.controls_schema.get(ctrl_idx) else {
        return;
    };
    let Some(ref cal) = ctrl_def.calibration else {
        return;
    };
    let decl = ctrl_def.declared_range.unwrap_or([0.0, 10.0]);
    let span = (decl[1] - decl[0]).max(1e-6);
    let inv_span = crate::math::newton6_reciprocal_f64(span as f64) as f32;
    let norm_01 = ((physical_value - decl[0]) * inv_span).clamp(0.0, 1.0);

    match cal.role {
        ControlCalibrationRole::StereoMode => {
            if let Some(ref opts) = ctrl_def.declared_options {
                let idx = (physical_value.round() as isize).max(0) as usize;
                if let Some(opt_label) = opts.get(idx)
                    && let Some(w) = cal.stereo_width_for_option(opt_label)
                {
                    *stereo_width = w;
                }
            }
        }
        ControlCalibrationRole::StereoWidth => {
            let u = (2.0 * norm_01 - 1.0).clamp(-1.0, 1.0);
            let w = cal.evaluate_lobatto_anchors(u).unwrap_or_else(|| {
                if decl[1] <= 2.0 {
                    physical_value
                } else {
                    norm_01
                }
            });
            *stereo_width = w.clamp(0.0, 2.0);
        }
        ControlCalibrationRole::FreezeLatch => {
            if ctrl_def.kind == ControlKind::Switch {
                if let Some(ref opts) = ctrl_def.declared_options {
                    let idx = (physical_value.round() as isize).max(0) as usize;
                    if let Some(opt_label) = opts.get(idx)
                        && let Some(alpha) = cal.freeze_amount_for_option(opt_label)
                    {
                        *is_frozen = (alpha > 0.5) && supports_freeze;
                    }
                }
            } else {
                *is_frozen = (norm_01 > 0.5) && supports_freeze;
            }
        }
        _ => {}
    }
}

/// Stack-allocated preset update batch shared between [`DnamEngine`] and [`crate::host::ControlProducer`].
pub(crate) struct ResolvedPresetBatch {
    pub preset_index: u16,
    pub default_restores: [(usize, f32, bool); 256],
    pub restore_count: usize,
    pub switch_updates: [(usize, f32); 256],
    pub switch_count: usize,
    pub range_updates: [(usize, f32); 256],
    pub range_count: usize,
}

/// Resolves a preset by `preset_id` into a stack-allocated [`ResolvedPresetBatch`].
pub(crate) fn resolve_preset_batch_by_id(
    header: Option<&DnamJsonHeader>,
    descriptors: &[ControlClampDescriptor],
    preset_id: &str,
) -> Result<ResolvedPresetBatch, DnamError> {
    let Some(hdr) = header else {
        return Err(DnamError::SchemaValidationError("header not loaded".into()));
    };
    let Some(preset_index) = hdr.presets.iter().position(|p| p.id == preset_id) else {
        return Err(DnamError::SchemaValidationError(alloc::format!(
            "unknown preset id: {preset_id}"
        )));
    };
    resolve_preset_batch_by_index(Some(hdr), descriptors, preset_index)
}

/// Resolves a preset by `preset_index` into a stack-allocated [`ResolvedPresetBatch`] without heap allocation.
pub(crate) fn resolve_preset_batch_by_index_opt(
    header: Option<&DnamJsonHeader>,
    descriptors: &[ControlClampDescriptor],
    preset_index: usize,
) -> Option<ResolvedPresetBatch> {
    let hdr = header?;
    let preset = hdr.presets.get(preset_index)?;
    let mut batch = ResolvedPresetBatch {
        preset_index: preset_index as u16,
        default_restores: [(0usize, 0.0f32, false); 256],
        restore_count: 0,
        switch_updates: [(0usize, 0.0f32); 256],
        switch_count: 0,
        range_updates: [(0usize, 0.0f32); 256],
        range_count: 0,
    };
    for (idx, def) in hdr.controls_schema.iter().enumerate() {
        if !preset.controls.contains_key(&def.id)
            && batch.restore_count < batch.default_restores.len()
        {
            let is_switch = def.kind == ControlKind::Switch;
            let def_val = descriptors.get(idx).map(|d| d.default_val).unwrap_or(0.0);
            batch.default_restores[batch.restore_count] = (idx, def_val, is_switch);
            batch.restore_count += 1;
        }
    }
    for (id, val) in &preset.controls {
        if let Some((idx, def)) = hdr
            .controls_schema
            .iter()
            .enumerate()
            .find(|(_, c)| &c.id == id)
        {
            let is_switch = def.kind == ControlKind::Switch;
            if let Some(f) = val.as_f64() {
                if is_switch && batch.switch_count < batch.switch_updates.len() {
                    batch.switch_updates[batch.switch_count] = (idx, f as f32);
                    batch.switch_count += 1;
                } else if !is_switch && batch.range_count < batch.range_updates.len() {
                    batch.range_updates[batch.range_count] = (idx, f as f32);
                    batch.range_count += 1;
                }
            } else if let Some(s) = val.as_str()
                && let Some(ref opts) = def.declared_options
                && let Some(opt_idx) = opts.iter().position(|o| o == s)
                && batch.switch_count < batch.switch_updates.len()
            {
                batch.switch_updates[batch.switch_count] = (idx, opt_idx as f32);
                batch.switch_count += 1;
            }
        }
    }
    Some(batch)
}

/// Resolves a preset by `preset_index` into a stack-allocated [`ResolvedPresetBatch`].
pub(crate) fn resolve_preset_batch_by_index(
    header: Option<&DnamJsonHeader>,
    descriptors: &[ControlClampDescriptor],
    preset_index: usize,
) -> Result<ResolvedPresetBatch, DnamError> {
    let Some(hdr) = header else {
        return Err(DnamError::SchemaValidationError("header not loaded".into()));
    };
    resolve_preset_batch_by_index_opt(Some(hdr), descriptors, preset_index)
        .ok_or_else(|| DnamError::SchemaValidationError("preset index out of range".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::math::ScalarBackend;
    use crate::schema::{ControlCalibration, ControlSchemaDef, ControlUiHints, PresetDef};
    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

    fn sample_range_ctrl(id: &str) -> ControlSchemaDef {
        ControlSchemaDef {
            id: String::from(id),
            name: String::from(id),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(1),
            anchor_nodes_norm: Some(vec![0.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn test_preset_resolution_error_branches_and_extra_controls_over_256() {
        assert!(resolve_preset_batch_by_id(None, &[], "p0").is_err());
        assert!(resolve_preset_batch_by_index(None, &[], 0).is_err());

        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema.clear();
        for i in 0..260 {
            let mut c = sample_range_ctrl(&format!("c{i}"));
            if i == 1 || i == 257 || i == 258 || i == 259 {
                c.kind = ControlKind::Switch;
                c.declared_range = None;
                c.safe_clamped_range = None;
                c.lobatto_degree = None;
                c.anchor_nodes_norm = None;
                c.declared_options = Some(vec![String::from("Off"), String::from("On")]);
                c.active_options = Some(vec![String::from("Off"), String::from("On")]);
                c.default = serde_json::json!("Off");
            }
            header.controls_schema.push(c);
        }

        let mut ctrls = BTreeMap::new();
        ctrls.insert(String::from("c0"), serde_json::json!(2.5));
        ctrls.insert(String::from("c1"), serde_json::json!("On"));
        ctrls.insert(String::from("c256"), serde_json::json!(3.5));
        ctrls.insert(String::from("c257"), serde_json::json!("On"));
        ctrls.insert(String::from("c259"), serde_json::json!(true)); // non-scalar ignored
        header.presets.push(PresetDef {
            id: String::from("p_extra"),
            name: String::from("Extra"),
            description: None,
            controls: ctrls,
            user_metadata: BTreeMap::new(),
        });

        assert!(resolve_preset_batch_by_id(Some(&header), &[], "missing").is_err());
        assert!(resolve_preset_batch_by_index(Some(&header), &[], 99).is_err());

        let base_hdr = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut base_hdr_mut = base_hdr;
        let container = build_dnam_container(&mut base_hdr_mut, &arena, &arena, false).unwrap();
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, crate::Tier::Lite, 48_000).unwrap();
        engine.header = Some(header);
        engine.control_bus.ensure_control_capacity(260);

        assert!(!engine.apply_preset_by_index_rt(99));
        engine.apply_preset("p_extra").unwrap();
        assert!(engine.apply_preset_by_index(0).is_ok());
        assert!(engine.apply_preset_by_index_rt(0));
    }

    #[test]
    fn test_apply_calibrated_control_role_variants() {
        let mut sw = 1.0_f32;
        let mut frozen = false;
        apply_calibrated_control_role(None, 0, 1.0, &mut sw, &mut frozen, true);

        let mut hdr = DnamJsonHeader::default_reference();
        let base = sample_range_ctrl("cal");
        hdr.controls_schema = vec![
            {
                let mut c = base.clone();
                c.kind = ControlKind::Switch;
                c.declared_options = Some(vec![String::from("Mono"), String::from("Stereo")]);
                c.calibration = Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoMode,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            {
                let mut c = base.clone();
                c.declared_range = Some([0.0, 2.0]);
                c.calibration = Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoWidth,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            {
                let mut c = base.clone();
                c.declared_range = Some([0.0, 100.0]);
                c.calibration = Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoWidth,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            {
                let mut c = base.clone();
                c.kind = ControlKind::Switch;
                c.declared_options = Some(vec![String::from("Off"), String::from("Freeze")]);
                c.calibration = Some(ControlCalibration {
                    role: ControlCalibrationRole::FreezeLatch,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            {
                let mut c = base.clone();
                c.declared_range = Some([0.0, 1.0]);
                c.calibration = Some(ControlCalibration {
                    role: ControlCalibrationRole::FreezeLatch,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: BTreeMap::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            base.clone(),
        ];

        apply_calibrated_control_role(Some(&hdr), 99, 1.0, &mut sw, &mut frozen, true);
        apply_calibrated_control_role(Some(&hdr), 5, 1.0, &mut sw, &mut frozen, true);
        apply_calibrated_control_role(Some(&hdr), 0, 0.0, &mut sw, &mut frozen, true);
        assert_eq!(sw, 0.0);
        apply_calibrated_control_role(Some(&hdr), 1, 1.5, &mut sw, &mut frozen, true);
        assert!((sw - 1.5).abs() < 1e-6);
        apply_calibrated_control_role(Some(&hdr), 2, 50.0, &mut sw, &mut frozen, true);
        assert!((sw - 0.5).abs() < 1e-6);
        apply_calibrated_control_role(Some(&hdr), 3, 1.0, &mut sw, &mut frozen, true);
        assert!(frozen);
        apply_calibrated_control_role(Some(&hdr), 4, 0.9, &mut sw, &mut frozen, false);
        assert!(!frozen);
    }
}
