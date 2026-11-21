//! Stack-allocated preset resolution and calibrated control-role side effects for [`DnamEngine`].

use dnam_vm::control::ControlClampDescriptor;
use dnam_vm::math::{rcp_nr6_f64, round_f32};

use crate::engine::{DnamEngine, EngineBackend, ReloadHeaderFlags};
use crate::schema::{BlendMode, ControlCalibrationRole, ControlKind, DnamError, DnamJsonHeader};

/// Maximum calibrated host-role controls (`StereoWidth`, `StereoMode`, `Mix`) cached across
/// allocation-free RT model hot-swaps ([`ReloadHeaderFlags`]).
pub(crate) const MAX_RELOAD_HOST_ROLE_CONTROLS: usize = 16;

/// Zero-allocation calibrated host-role descriptor retained across [`DnamEngine::reload_model_in_place`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReloadCalibratedControl {
    pub(crate) ctrl_idx: u8,
    pub(crate) role: ControlCalibrationRole,
    pub(crate) decl_min: f32,
    pub(crate) decl_max: f32,
    pub(crate) has_anchors: bool,
    pub(crate) anchors: [f32; 3],
    pub(crate) stereo_mode_opt_widths: [f32; 16],
    pub(crate) stereo_mode_opt_mask: u16,
}

impl ReloadCalibratedControl {
    /// Returns an inert placeholder slot with no mapped switch options or anchors.
    #[inline]
    pub(crate) const fn empty() -> Self {
        Self {
            ctrl_idx: 0,
            role: ControlCalibrationRole::Feedback,
            decl_min: 0.0,
            decl_max: 1.0,
            has_anchors: false,
            anchors: [0.0; 3],
            stereo_mode_opt_widths: [0.0; 16],
            stereo_mode_opt_mask: 0,
        }
    }
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Applies a preset by string identifier (`presets[*].id`).
    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        let batch =
            resolve_preset_batch_by_id(self.header(), self.control_descriptors(), preset_id)?;
        self.apply_resolved_preset_batch(&batch);
        Ok(())
    }

    /// Applies a preset by index without any heap allocations (`ref[dsp.ctrl.op-m.spsc-ingress]`).
    pub fn apply_preset_by_index(&mut self, preset_index: usize) -> Result<(), DnamError> {
        let batch =
            resolve_preset_batch_by_index(self.header(), self.control_descriptors(), preset_index)?;
        self.apply_resolved_preset_batch(&batch);
        Ok(())
    }

    /// Real-time preset application by index returning `bool` (zero `DnamError` heap allocation or drop glue).
    #[inline]
    pub(crate) fn apply_preset_by_index_rt(&mut self, preset_index: usize) -> bool {
        let Some(batch) = resolve_preset_batch_by_index_opt(
            self.header(),
            self.control_descriptors(),
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
    }
}

#[inline]
fn eval_lobatto_anchors_array(anchors: [f32; 3], u: f32) -> f32 {
    let [v_min, v_mid, v_max] = anchors;
    let u_clamped = u.clamp(-1.0, 1.0);
    let c1 = 0.5 * (v_max - v_min);
    let c2 = 0.5 * (v_max + v_min) - v_mid;
    v_mid + c1 * u_clamped + c2 * u_clamped * u_clamped
}

#[inline]
fn compute_norm_01(physical_value: f32, decl: [f32; 2]) -> f32 {
    let span = (decl[1] - decl[0]).max(1e-6);
    let inv_span = rcp_nr6_f64(span as f64) as f32;
    ((physical_value - decl[0]) * inv_span).clamp(0.0, 1.0)
}

#[inline]
fn eval_stereo_width_range(physical_value: f32, decl: [f32; 2], anchors: Option<[f32; 3]>) -> f32 {
    let norm_01 = compute_norm_01(physical_value, decl);
    let u = (2.0 * norm_01 - 1.0).clamp(-1.0, 1.0);
    let w = match anchors {
        Some(a) => eval_lobatto_anchors_array(a, u),
        None if decl[1] <= 2.0 => physical_value,
        None => norm_01,
    };
    w.clamp(0.0, 2.0)
}

#[inline]
fn eval_mix_ratio(physical_value: f32, decl: [f32; 2], anchors: Option<[f32; 3]>) -> f32 {
    let norm_01 = compute_norm_01(physical_value, decl);
    let u = (2.0 * norm_01 - 1.0).clamp(-1.0, 1.0);
    let raw_mix = match anchors {
        Some(a) => eval_lobatto_anchors_array(a, u),
        None => norm_01,
    };
    let scaled = if raw_mix > 1.0 && decl[1] > 1.0 {
        norm_01
    } else {
        raw_mix
    };
    scaled.clamp(0.0, 1.0)
}

/// Evaluates calibrated control-role side effects (`StereoMode`, `StereoWidth`) shared between
/// [`DnamEngine`] and [`crate::host::AcceptedStateShadow`].
pub(crate) fn apply_calibrated_control_role(
    header: Option<&DnamJsonHeader>,
    ctrl_idx: usize,
    physical_value: f32,
    stereo_width: &mut f32,
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

    match cal.role {
        ControlCalibrationRole::StereoMode => {
            if let Some(ref opts) = ctrl_def.declared_options {
                let idx = (round_f32(physical_value) as isize).max(0) as usize;
                if let Some(opt_label) = opts.get(idx)
                    && let Some(w) = cal.stereo_width_for_option(opt_label)
                {
                    *stereo_width = w;
                }
            }
        }
        ControlCalibrationRole::StereoWidth => {
            *stereo_width =
                eval_stereo_width_range(physical_value, decl, cal.lobatto_physical_anchors);
        }
        _ => {}
    }
}

/// Evaluates calibrated control-role side effects (`StereoMode`, `StereoWidth`) from
/// zero-allocation [`ReloadHeaderFlags`] after an RT model hot-swap.
pub(crate) fn apply_calibrated_control_role_from_reload(
    flags: &ReloadHeaderFlags,
    ctrl_idx: usize,
    physical_value: f32,
    stereo_width: &mut f32,
) {
    let count = (flags.host_role_count as usize).min(flags.host_role_slots.len());
    let Some(slot) = flags.host_role_slots[..count]
        .iter()
        .find(|s| s.ctrl_idx as usize == ctrl_idx)
    else {
        return;
    };
    match slot.role {
        ControlCalibrationRole::StereoMode => {
            let idx = (round_f32(physical_value) as isize).max(0) as usize;
            if idx < 16 && (slot.stereo_mode_opt_mask & (1_u16 << idx)) != 0 {
                *stereo_width = slot.stereo_mode_opt_widths[idx];
            }
        }
        ControlCalibrationRole::StereoWidth => {
            let anchors = slot.has_anchors.then_some(slot.anchors);
            *stereo_width =
                eval_stereo_width_range(physical_value, [slot.decl_min, slot.decl_max], anchors);
        }
        _ => {}
    }
}

/// Resolves normalized `[0.0, 1.0]` parallel mix ratio for `ControlCalibrationRole::Mix`.
pub(crate) fn resolve_calibrated_host_blend(
    header: Option<&DnamJsonHeader>,
    ctrl_idx: usize,
    physical_value: f32,
) -> Option<f32> {
    let hdr = header?;
    let ctrl_def = hdr.controls_schema.get(ctrl_idx)?;
    let cal = ctrl_def.calibration.as_ref()?;
    if cal.role != ControlCalibrationRole::Mix {
        return None;
    }
    let decl = ctrl_def.declared_range.unwrap_or([0.0, 1.0]);
    Some(eval_mix_ratio(
        physical_value,
        decl,
        cal.lobatto_physical_anchors,
    ))
}

/// Resolves normalized `[0.0, 1.0]` parallel mix ratio for `ControlCalibrationRole::Mix`
/// from zero-allocation [`ReloadHeaderFlags`].
pub(crate) fn resolve_calibrated_host_blend_from_reload(
    flags: &ReloadHeaderFlags,
    ctrl_idx: usize,
    physical_value: f32,
) -> Option<f32> {
    if flags.blend_mode != BlendMode::HostParallel {
        return None;
    }
    let count = (flags.host_role_count as usize).min(flags.host_role_slots.len());
    let slot = flags.host_role_slots[..count]
        .iter()
        .find(|s| s.ctrl_idx as usize == ctrl_idx)?;
    if slot.role != ControlCalibrationRole::Mix {
        return None;
    }
    let anchors = slot.has_anchors.then_some(slot.anchors);
    Some(eval_mix_ratio(
        physical_value,
        [slot.decl_min, slot.decl_max],
        anchors,
    ))
}

/// Stack-allocated preset update batch shared between [`DnamEngine`] and [`crate::host::ControlProducer`].
pub(crate) struct ResolvedPresetBatch {
    pub(crate) preset_index: u16,
    pub(crate) default_restores: [(usize, f32, bool); 256],
    pub(crate) restore_count: usize,
    pub(crate) switch_updates: [(usize, f32); 256],
    pub(crate) switch_count: usize,
    pub(crate) range_updates: [(usize, f32); 256],
    pub(crate) range_count: usize,
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
    use crate::schema::{ControlCalibration, ControlSchemaDef, ControlUiHints, PresetDef};
    use crate::test_support::build_test_dnam_container;
    use crate::types::Tier;
    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use dnam_vm::math::ScalarBackend;

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
            conditional_bounds: Vec::new(),
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
    fn test_preset_resolution_error_branches_and_max_255_controls() {
        assert!(resolve_preset_batch_by_id(None, &[], "p0").is_err());
        assert!(resolve_preset_batch_by_index(None, &[], 0).is_err());

        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema.clear();
        for i in 0..255 {
            let mut c = sample_range_ctrl(&format!("c{i}"));
            if i == 1 || i == 253 || i == 254 {
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
        ctrls.insert(String::from("c252"), serde_json::json!(3.5));
        ctrls.insert(String::from("c253"), serde_json::json!("On"));
        ctrls.insert(String::from("c254"), serde_json::json!(1));
        header.presets.push(PresetDef {
            id: String::from("p_full"),
            name: String::from("Full"),
            description: None,
            controls: ctrls,
            user_metadata: BTreeMap::new(),
        });

        assert!(resolve_preset_batch_by_id(Some(&header), &[], "missing").is_err());
        assert!(resolve_preset_batch_by_index(Some(&header), &[], 99).is_err());

        let mut header_clone = header.clone();
        let container = build_test_dnam_container(&mut header_clone, Tier::Lite, false);
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        assert!(!engine.apply_preset_by_index_rt(99));
        engine.apply_preset("p_full").unwrap();
        assert!((engine.clamped_control_values()[0] - 2.5).abs() < 1e-6);
        assert!((engine.clamped_control_values()[1] - 1.0).abs() < 1e-6);
        assert!((engine.clamped_control_values()[252] - 3.5).abs() < 1e-6);
        assert!((engine.clamped_control_values()[253] - 1.0).abs() < 1e-6);
        assert!(engine.apply_preset_by_index(0).is_ok());
        assert!(engine.apply_preset_by_index_rt(0));
    }

    #[test]
    fn test_apply_calibrated_control_role_variants() {
        let mut sw = 1.0_f32;
        apply_calibrated_control_role(None, 0, 1.0, &mut sw);

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
                    conditional_physical_anchors: Vec::new(),
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
                    conditional_physical_anchors: Vec::new(),
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
                    conditional_physical_anchors: Vec::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            {
                let mut c = base.clone();
                c.declared_range = Some([0.0, 1.0]);
                c.calibration = Some(ControlCalibration {
                    role: ControlCalibrationRole::Mix,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: Vec::new(),
                    tempo_sync_default_subdivision: None,
                });
                c
            },
            base,
        ];

        apply_calibrated_control_role(Some(&hdr), 99, 1.0, &mut sw);
        apply_calibrated_control_role(Some(&hdr), 4, 1.0, &mut sw);
        apply_calibrated_control_role(Some(&hdr), 0, 0.0, &mut sw);
        assert_eq!(sw, 0.0);
        apply_calibrated_control_role(Some(&hdr), 1, 1.5, &mut sw);
        assert!((sw - 1.5).abs() < 1e-6);
        apply_calibrated_control_role(Some(&hdr), 2, 50.0, &mut sw);
        assert!((sw - 0.5).abs() < 1e-6);
        assert_eq!(resolve_calibrated_host_blend(Some(&hdr), 3, 0.4), Some(0.4));
        assert_eq!(resolve_calibrated_host_blend(Some(&hdr), 4, 0.4), None);
    }

    #[test]
    fn test_conditional_clamp_tightening_on_stereo_width_and_mix_controls() {
        use crate::container::wire::StructuredConditionalBound;

        let mut hdr = DnamJsonHeader::default_reference();
        hdr.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        hdr.io_config.capabilities.default_blend = 0.5;
        let cond_tight = vec![StructuredConditionalBound {
            when: vec![(String::from("gate"), String::from("Tight"))],
            min: Some(0.25),
            max: Some(0.75),
        }];
        let mut gate = sample_range_ctrl("gate");
        gate.kind = ControlKind::Switch;
        gate.declared_range = None;
        gate.safe_clamped_range = None;
        gate.lobatto_degree = None;
        gate.anchor_nodes_norm = None;
        gate.declared_options = Some(vec![String::from("Open"), String::from("Tight")]);
        gate.active_options = Some(vec![String::from("Open"), String::from("Tight")]);
        gate.default = serde_json::json!("Open");

        let mut width_ctrl = sample_range_ctrl("width");
        width_ctrl.declared_range = Some([0.0, 2.0]);
        width_ctrl.safe_clamped_range = Some([0.0, 2.0]);
        width_ctrl.default = serde_json::json!(1.0);
        width_ctrl.conditional_bounds = cond_tight.clone();
        width_ctrl.calibration = Some(ControlCalibration {
            role: ControlCalibrationRole::StereoWidth,
            lobatto_physical_anchors: Some([0.0, 1.0, 2.0]),
            conditional_physical_anchors: Vec::new(),
            tempo_sync_default_subdivision: None,
        });

        let mut mix_ctrl = sample_range_ctrl("mix");
        mix_ctrl.declared_range = Some([0.0, 1.0]);
        mix_ctrl.safe_clamped_range = Some([0.0, 1.0]);
        mix_ctrl.default = serde_json::json!(0.5);
        mix_ctrl.conditional_bounds = cond_tight;
        mix_ctrl.calibration = Some(ControlCalibration {
            role: ControlCalibrationRole::Mix,
            lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
            conditional_physical_anchors: Vec::new(),
            tempo_sync_default_subdivision: None,
        });

        hdr.controls_schema = vec![gate, width_ctrl, mix_ctrl];
        let container = build_test_dnam_container(&mut hdr, Tier::Lite, false);
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        // Activate Tight mode (`1.0`), tightening width and mix safe range to `[0.25, 0.75]`.
        engine.set_control_immediate(0, 1.0);
        engine.set_control_value(1, 1.9);
        assert!((engine.clamped_control_values()[1] - 0.75).abs() < 1.0e-6);
        assert!((engine.stereo_width() - 0.75).abs() < 1.0e-6);

        engine.set_control_immediate(1, 0.05);
        assert!((engine.clamped_control_values()[1] - 0.25).abs() < 1.0e-6);
        assert!((engine.stereo_width() - 0.25).abs() < 1.0e-6);

        engine.set_control_value(2, 0.99);
        assert!((engine.clamped_control_values()[2] - 0.75).abs() < 1.0e-6);
        assert!((engine.mix_wet_gain - 0.75).abs() < 1.0e-5);
        assert!((engine.mix_dry_gain - 0.25).abs() < 1.0e-5);

        engine.set_control_immediate(2, 0.02);
        assert!((engine.clamped_control_values()[2] - 0.25).abs() < 1.0e-6);
        assert!((engine.mix_wet_gain - 0.25).abs() < 1.0e-5);
        assert!((engine.mix_dry_gain - 0.75).abs() < 1.0e-5);
    }
}
