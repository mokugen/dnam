#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Phase-3 `[CLAMP-DYNAMIC]` extreme-corner auditing, 256-frame `[STAB-ZERO-DC]` verification,
//! and deterministic `< 10 ms` `PANAMA` corner prescriptions (`dnam_capture::audit::dynamic_and_panama`).

mod corners;
mod panama;

pub use corners::build_extreme_valid_corners;
pub use panama::append_panama_prescriptions;

use dnam_engine::schema::DnamError;
use dnam_kernels::fp16_to_f32;
use dnam_train::solver::model::stage_latent::{
    STAGE_LATENT_CORE_MODS_COLS, STAGE_LATENT_CORE_MODS_STRIDE, STAGE_LATENT_GLOBAL_MODS_START,
    STAGE_LATENT_LITE_LRU_START, STAGE_LATENT_ULTRA_ACTIVE_END,
};
use dnam_train::{
    DynamicCornerViolation, check_corner_dynamic_stability, compute_dynamic_excursion_bounds_32,
    eval_static_control_bus_targets, verify_extreme_corners_zero_dc_256,
};
use dnam_vm::Tier;
use dnam_vm::bin::{
    Arena0LiteWire, Arena3UltraDeltaWire, CextDispatchWire, DnamBinBuildInput, DnamBinView,
    encode_dnam_bin,
};
use dnam_vm::control::ControlBus;
use dnam_vm::math::{DefaultMathBackend, f32_to_fp16};
use zerocopy::byteorder::little_endian::{F32, U16};

use crate::proj::schema::DnamProjManifest;

const DYNAMIC_CONTRACT_MAX_ITERS: usize = 12;
const ZERO_DC_CONTRACT_MAX_ITERS: usize = 8;
const STAGE_LATENT_LITE_MODS_END: usize = STAGE_LATENT_LITE_LRU_START + 16;

const _: () = assert!(STAGE_LATENT_LITE_MODS_END == 80);
const _: () = assert!(STAGE_LATENT_ULTRA_ACTIVE_END == 164);
const _: () = assert!(STAGE_LATENT_CORE_MODS_COLS < STAGE_LATENT_CORE_MODS_STRIDE);

#[inline]
fn map_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

#[inline]
fn checked_tiers(retained: Tier) -> Vec<Tier> {
    if retained == Tier::Ultra {
        vec![Tier::Lite, Tier::Ultra]
    } else {
        vec![Tier::Lite]
    }
}

#[inline]
fn push_unique_tag(tags: &mut Vec<String>, tag: &str) {
    if !tags.iter().any(|existing| existing == tag) {
        tags.push(tag.to_string());
    }
}

#[derive(Default)]
struct ViolationSlots {
    slot_mask: u32,
    scale_global_stage_latent: bool,
    sanitize_cext: bool,
    scale_all_stage_latent: bool,
}

fn collect_violation_slots(violations: &[DynamicCornerViolation]) -> ViolationSlots {
    let mut out = ViolationSlots::default();
    for v in violations {
        match *v {
            DynamicCornerViolation::RawStageTargetBound { slot, .. } => {
                if slot < 31 {
                    out.slot_mask |= 1_u32 << slot;
                } else {
                    out.sanitize_cext = true;
                }
            }
            DynamicCornerViolation::KautzPoleRadius { .. } => {
                out.slot_mask |= 1_u32 << 11;
                out.scale_global_stage_latent = true;
            }
            DynamicCornerViolation::TransportDampPole { .. } => {
                out.slot_mask |= 1_u32 << 21;
            }
            DynamicCornerViolation::TransportDelayFloor { .. } => {
                out.slot_mask |= (1_u32 << 12) | (1_u32 << 17) | (1_u32 << 20);
            }
            DynamicCornerViolation::TransportLoopGain { .. } => {
                out.slot_mask |= 1_u32 << 13;
            }
            DynamicCornerViolation::TransportAllpassGain { .. } => {
                out.slot_mask |= 1_u32 << 25;
            }
            DynamicCornerViolation::OpNPreZohHoldDepth { .. } => {
                out.slot_mask |= 1_u32 << 27;
            }
            DynamicCornerViolation::PhasorRateOctave { .. } => {
                out.slot_mask |= 1_u32 << 29;
            }
            DynamicCornerViolation::SensoryVcaGainDelta { .. } => {
                out.slot_mask |= 1_u32 << 30;
            }
            DynamicCornerViolation::StageLatentTableBound { .. } => {
                out.scale_all_stage_latent = true;
            }
            _ => {}
        }
    }
    out
}

fn scale_fp16_slice(slice: &mut [U16], scale: f32) -> bool {
    let mut changed = false;
    for w in slice {
        let v = fp16_to_f32(w.get());
        if v != 0.0 || !v.is_finite() {
            let next = if v.is_finite() { v * scale } else { 0.0 };
            *w = U16::new(f32_to_fp16(next));
            changed = true;
        }
    }
    changed
}

fn zero_reserved_u16_slice(slice: &mut [U16]) -> bool {
    let mut changed = false;
    for w in slice {
        if w.get() != 0 {
            *w = U16::new(0);
            changed = true;
        }
    }
    changed
}

fn contract_stage_latent_tables(
    a0: &mut Arena0LiteWire,
    a3: Option<&mut Arena3UltraDeltaWire>,
    scale: f32,
) -> bool {
    let mut changed = scale_fp16_slice(
        &mut a0.linear_readout_lite.stage_latent_mods_fp16
            [STAGE_LATENT_GLOBAL_MODS_START..STAGE_LATENT_LITE_MODS_END],
        scale,
    );
    if let Some(a3w) = a3 {
        changed |= scale_fp16_slice(
            &mut a3w.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16
                [..STAGE_LATENT_ULTRA_ACTIVE_END],
            scale,
        );
        changed |= zero_reserved_u16_slice(
            &mut a3w.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16
                [STAGE_LATENT_ULTRA_ACTIVE_END..],
        );
        for row in a3w
            .stage_latent_core_mods_fp16
            .as_chunks_mut::<STAGE_LATENT_CORE_MODS_STRIDE>()
            .0
        {
            changed |= scale_fp16_slice(&mut row[..STAGE_LATENT_CORE_MODS_COLS], scale);
            changed |= zero_reserved_u16_slice(&mut row[STAGE_LATENT_CORE_MODS_COLS..]);
        }
    }
    changed
}

fn scale_f32_slice(slice: &mut [F32], scale: f32) -> bool {
    let mut changed = false;
    for w in slice {
        let v = w.get();
        if v != 0.0 || !v.is_finite() {
            let next = if v.is_finite() { v * scale } else { 0.0 };
            *w = F32::new(next);
            changed = true;
        }
    }
    changed
}

fn reencode_view_with_control_edits(
    view: &DnamBinView<'_>,
    a0: &Arena0LiteWire,
    a3: Option<&Arena3UltraDeltaWire>,
    cext: Option<&CextDispatchWire>,
) -> Result<Vec<u8>, DnamError> {
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
        cext,
        arena0: a0,
        arena1: view.arena1,
        arena2: view.arena2,
        arena3: a3,
    })
    .map_err(map_err)
}

fn contract_dynamic_violation_slots(
    dnam_bin: &mut Vec<u8>,
    slots: &ViolationSlots,
    static_breach_mask: u32,
    delta_32: &[f64; 32],
    scale: f32,
) -> Result<bool, DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(map_err)?;
    let mut a0 = *view.arena0;
    let mut a3 = view.arena3.copied();
    let mut cext = view.cext.copied();
    let mut changed = false;

    if let Some(ref mut cw) = cext
        && slots.sanitize_cext
    {
        for row in &mut cw.w_ctrl_ext {
            changed |= scale_f32_slice(&mut row[0..4], 0.0);
        }
        changed |= scale_f32_slice(&mut cw.w_ctrl_ext[15], 0.0);
        changed |= scale_f32_slice(&mut cw.w_ctrl_ultra_ext[15], 0.0);
    }

    for slot in 0..31_usize {
        if (slots.slot_mask & (1_u32 << slot)) == 0 {
            continue;
        }
        let dyn_only = delta_32[slot] > 0.0 && (static_breach_mask & (1_u32 << slot)) == 0;
        let mut slot_changed = false;
        if slot < 16 {
            slot_changed |= scale_f32_slice(&mut a0.control_bus_lite.w_ctrl[slot][8..16], scale);
            if !dyn_only || !slot_changed {
                slot_changed |= scale_f32_slice(&mut a0.control_bus_lite.w_ctrl[slot][0..8], scale);
                if let Some(ref mut a3w) = a3 {
                    slot_changed |=
                        scale_f32_slice(&mut a3w.control_bus_ultra_delta.w_ctrl_ultra[slot], scale);
                }
            }
        } else if let Some(ref mut cw) = cext {
            let r = slot - 16;
            slot_changed |= scale_f32_slice(&mut cw.w_ctrl_ext[r][8..16], scale);
            if !dyn_only || !slot_changed {
                slot_changed |= scale_f32_slice(&mut cw.w_ctrl_ext[r][4..8], scale);
                slot_changed |= scale_f32_slice(&mut cw.w_ctrl_ultra_ext[r], scale);
            }
        }
        changed |= slot_changed;
    }

    if (slots.scale_global_stage_latent && !changed) || slots.scale_all_stage_latent {
        changed |= contract_stage_latent_tables(&mut a0, a3.as_mut(), scale);
    }

    if changed {
        *dnam_bin = reencode_view_with_control_edits(&view, &a0, a3.as_ref(), cext.as_ref())?;
    }
    Ok(changed)
}

fn contract_all_control_excursions(dnam_bin: &mut Vec<u8>, scale: f32) -> Result<bool, DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(map_err)?;
    let mut a0 = *view.arena0;
    let mut a3 = view.arena3.copied();
    let mut cext = view.cext.copied();
    let mut changed = false;

    for row in &mut a0.control_bus_lite.w_ctrl {
        changed |= scale_f32_slice(row, scale);
    }
    if let Some(ref mut a3w) = a3 {
        for row in &mut a3w.control_bus_ultra_delta.w_ctrl_ultra {
            changed |= scale_f32_slice(row, scale);
        }
    }
    changed |= contract_stage_latent_tables(&mut a0, a3.as_mut(), scale);
    if let Some(ref mut cw) = cext {
        for r in 0..15 {
            changed |= scale_f32_slice(&mut cw.w_ctrl_ext[r][4..16], scale);
            changed |= scale_f32_slice(&mut cw.w_ctrl_ultra_ext[r], scale);
        }
    }

    if changed {
        *dnam_bin = reencode_view_with_control_edits(&view, &a0, a3.as_ref(), cext.as_ref())?;
    }
    Ok(changed)
}

/// Runs closed-form `[CLAMP-DYNAMIC]` extreme-corner excursion verification and conservative
/// `0-ULP StarCenter` contraction, followed by 256-frame extreme-corner `[STAB-ZERO-DC]` runtime
/// certification across `Lite` and (when present) `Ultra` (`ref[train.audit.phase3.clamp-dynamic]`,
/// `ref[train.audit.phase3.numeric-envelope]`).
pub fn audit_dynamic_corners_and_zero_dc(
    manifest: &mut DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    full_backbone: bool,
) -> Result<(), DnamError> {
    let _ = full_backbone;
    for iter in 0..DYNAMIC_CONTRACT_MAX_ITERS {
        let view = DnamBinView::parse(dnam_bin).map_err(map_err)?;
        let n_ctrl = usize::from(view.header.num_controls.get());
        let corners = build_extreme_valid_corners(view.control_descriptors, n_ctrl);
        let delta_32 = compute_dynamic_excursion_bounds_32(&view);

        let mut all_violations = Vec::new();
        let mut static_breach_mask = 0_u32;
        for tier in checked_tiers(view.retained_tier) {
            let mut bus =
                ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view, tier, 48_000.0)
                    .map_err(map_err)?;
            for corner in &corners {
                let (static_t, static_z) =
                    eval_static_control_bus_targets(&mut bus, corner).map_err(map_err)?;
                let rep = check_corner_dynamic_stability(
                    &view, tier, &static_t, &static_z, &delta_32, true,
                );
                if !rep.passed {
                    let static_rep = check_corner_dynamic_stability(
                        &view, tier, &static_t, &static_z, &[0.0; 32], true,
                    );
                    let static_slots = collect_violation_slots(&static_rep.violations);
                    static_breach_mask |= static_slots.slot_mask;
                    all_violations.extend(rep.violations);
                }
            }
        }

        if all_violations.is_empty() {
            break;
        }
        let slots = collect_violation_slots(&all_violations);
        let scale = if iter + 1 < DYNAMIC_CONTRACT_MAX_ITERS {
            0.5
        } else {
            0.0
        };
        let contracted = contract_dynamic_violation_slots(
            dnam_bin,
            &slots,
            static_breach_mask,
            &delta_32,
            scale,
        )?;
        if contracted {
            let ar = manifest.audit_report.get_or_insert_with(Default::default);
            push_unique_tag(&mut ar.phase3_clamps, "CLAMP-DYNAMIC");
        } else {
            break;
        }
    }

    for attempt in 0..=ZERO_DC_CONTRACT_MAX_ITERS {
        let view = DnamBinView::parse(dnam_bin).map_err(map_err)?;
        let n_ctrl = usize::from(view.header.num_controls.get());
        let corners = build_extreme_valid_corners(view.control_descriptors, n_ctrl);

        let mut all_passed = true;
        let mut worst_abs = 0.0_f32;
        for tier in checked_tiers(view.retained_tier) {
            let rep = verify_extreme_corners_zero_dc_256(&view, tier, &corners).map_err(map_err)?;
            worst_abs = worst_abs.max(rep.max_abs_dc);
            if !rep.passed {
                all_passed = false;
                break;
            }
        }
        if all_passed {
            return Ok(());
        }
        if attempt < ZERO_DC_CONTRACT_MAX_ITERS {
            let scale = if attempt + 1 < ZERO_DC_CONTRACT_MAX_ITERS {
                0.5
            } else {
                0.0
            };
            let contracted = contract_all_control_excursions(dnam_bin, scale)?;
            if contracted {
                let ar = manifest.audit_report.get_or_insert_with(Default::default);
                push_unique_tag(&mut ar.phase1_clamps, "CLAMP-STABILITY");
                continue;
            }
        }
        return Err(DnamError::SchemaValidationError(format!(
            "[STAB-ZERO-DC] extreme-corner 256-frame zero-DC verification failed (peak={worst_abs:.6e} > 1e-5)"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;
    use dnam_vm::SonaVm;
    use dnam_vm::bin::SONA_META_CORE_MODS_PRESENT;
    use dnam_vm::control::ControlClampDescriptor;
    use zerocopy::byteorder::little_endian::U32;

    fn encode_single_knob_test_bin(
        v0: &DnamBinView<'_>,
        a0: &Arena0LiteWire,
        a3: &Arena3UltraDeltaWire,
        cext: &CextDispatchWire,
    ) -> Vec<u8> {
        let mut d = ControlClampDescriptor::empty();
        d.ctrl_kind = 0;
        d.lobatto_degree = 3;
        d.safe_min_base = 0.0;
        d.safe_max_base = 1.0;
        let ctrls = [d.to_wire()];
        encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: v0.header.reference_sample_rate_hz.get(),
            header_flags: v0.header.flags.get(),
            ext_ring_pow2_frames: v0.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: v0.runtime_meta.ext_ring_lines,
            ext_ring_format: v0.runtime_meta.ext_ring_format,
            declared_max_delay_48k: v0.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: v0.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: v0.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: v0.runtime_meta.ro_pole_48k.get(),
            switch_xfade_samples: v0.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: v0.runtime_meta.content_hash_prefix,
            controls: &ctrls,
            cp1_overflow: &[],
            tsyn: &[],
            cext: Some(cext),
            arena0: a0,
            arena1: v0.arena1,
            arena2: v0.arena2,
            arena3: Some(a3),
        })
        .unwrap()
    }

    fn render_star_impulse(bin: &[u8]) -> Vec<[f32; 2]> {
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Ultra, 48_000.0).unwrap();
        vm.reset();
        (0..64)
            .map(|i| {
                let x = if i == 0 { 0.5 } else { 0.0 };
                let (l, r) = vm.step_frame(x, x).unwrap();
                [l, r]
            })
            .collect()
    }

    #[test]
    fn test_audit_dynamic_corners_contracts_excess_excursion_with_zero_ulp_star_and_zero_dc() {
        let base_bin = initialize_backbone_dnam_bin_if_needed(None).unwrap();
        let v0 = DnamBinView::parse(&base_bin).unwrap();
        let mut a0 = *v0.arena0;
        let a3 = *v0.arena3.unwrap();
        a0.conv_kautz_lite[0].r_low = [F32::new(0.8); 4];
        a0.conv_kautz_lite[0].b_gamma = [F32::new(1.0); 4];
        a0.control_bus_lite.w_ctrl[13][0] = F32::new(2.0);
        a0.control_bus_lite.w_ctrl[13][8] = F32::new(0.8);
        let mut cext = CextDispatchWire {
            w_ctrl_ext: [[F32::new(0.0); 16]; 16],
            w_ctrl_ultra_ext: [[F32::new(0.0); 4]; 16],
        };
        cext.w_ctrl_ext[14][12] = F32::new(2.5);
        let mut dbin = encode_single_knob_test_bin(&v0, &a0, &a3, &cext);

        let star_before = render_star_impulse(&dbin);
        let mut manifest = DnamProjManifest::default_reference();
        audit_dynamic_corners_and_zero_dc(&mut manifest, &mut dbin, true).unwrap();
        let star_after = render_star_impulse(&dbin);

        for (b, a) in star_before.iter().zip(star_after.iter()) {
            assert_eq!(b[0].to_bits(), a[0].to_bits());
            assert_eq!(b[1].to_bits(), a[1].to_bits());
        }
        let ar = manifest.audit_report.as_ref().unwrap();
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-DYNAMIC"));
    }

    #[test]
    fn test_audit_dynamic_corners_contracts_slot27_zoh_and_ultra_stage_latent_tables() {
        let base_bin = initialize_backbone_dnam_bin_if_needed(None).unwrap();
        let v0 = DnamBinView::parse(&base_bin).unwrap();
        let mut a0 = *v0.arena0;
        let mut a3 = *v0.arena3.unwrap();
        a0.crossbar.op_n_pre_base.zoh_hold_depth = F32::new(0.6);
        a0.crossbar.sona_meta_flags =
            U32::new(a0.crossbar.sona_meta_flags.get() | SONA_META_CORE_MODS_PRESENT);
        a0.crossbar.cell_desc.cell_flags = a0.crossbar.sona_meta_flags;
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[160] = U16::new(f32_to_fp16(6.0));
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[164] = U16::new(f32_to_fp16(1.0));
        a3.stage_latent_core_mods_fp16[0] = U16::new(f32_to_fp16(6.0));
        let mut cext = CextDispatchWire {
            w_ctrl_ext: [[F32::new(0.0); 16]; 16],
            w_ctrl_ultra_ext: [[F32::new(0.0); 4]; 16],
        };
        // Slot 27 (`op_n_pre_zoh_hold_delta`, `r = 11`) violates `OpNPreZohHoldDepth`;
        // benign slot 28 (`op_n_pre_beta_q_delta`, `r = 12`) must remain untouched.
        cext.w_ctrl_ext[11][12] = F32::new(0.6);
        cext.w_ctrl_ext[12][12] = F32::new(0.25);
        let mut dbin = encode_single_knob_test_bin(&v0, &a0, &a3, &cext);

        let star_before = render_star_impulse(&dbin);
        let mut manifest = DnamProjManifest::default_reference();
        audit_dynamic_corners_and_zero_dc(&mut manifest, &mut dbin, true).unwrap();
        let star_after = render_star_impulse(&dbin);

        for (b, a) in star_before.iter().zip(star_after.iter()) {
            assert_eq!(b[0].to_bits(), a[0].to_bits());
            assert_eq!(b[1].to_bits(), a[1].to_bits());
        }

        let audited = DnamBinView::parse(&dbin).unwrap();
        let audited_cext = audited.cext.unwrap();
        assert!(audited_cext.w_ctrl_ext[11][12].get() < 0.4);
        assert_eq!(
            audited_cext.w_ctrl_ext[12][12].get().to_bits(),
            0.25_f32.to_bits()
        );
        let audited_a3 = audited.arena3.unwrap();
        assert!(
            fp16_to_f32(
                audited_a3
                    .control_bus_ultra_delta
                    .stage_latent_ctrl_ultra_fp16[160]
                    .get()
            )
            .abs()
                <= 4.0
        );
        assert_eq!(
            audited_a3
                .control_bus_ultra_delta
                .stage_latent_ctrl_ultra_fp16[164]
                .get(),
            0
        );
        assert!(fp16_to_f32(audited_a3.stage_latent_core_mods_fp16[0].get()).abs() <= 4.0);
        assert_eq!(audited_a3.stage_latent_core_mods_fp16[120].get(), 0);

        let n_ctrl = usize::from(audited.header.num_controls.get());
        let corners = build_extreme_valid_corners(audited.control_descriptors, n_ctrl);
        let delta_32 = compute_dynamic_excursion_bounds_32(&audited);
        for tier in [Tier::Lite, Tier::Ultra] {
            let mut bus =
                ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&audited, tier, 48_000.0)
                    .unwrap();
            for corner in &corners {
                let (static_t, static_z) =
                    eval_static_control_bus_targets(&mut bus, corner).unwrap();
                let rep = check_corner_dynamic_stability(
                    &audited, tier, &static_t, &static_z, &delta_32, true,
                );
                assert!(rep.passed && rep.violations.is_empty());
            }
        }

        let ar = manifest.audit_report.as_ref().unwrap();
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-DYNAMIC"));
    }
}
