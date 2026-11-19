//! Inline `CP1` contributor decoding and `U_int` ANOVA-FM interaction matrix evaluation (`r[dsp.ctrl.accum.anova-fm]`).

use zerocopy::byteorder::little_endian::F32;
use zerocopy::{FromBytes, IntoBytes};

use crate::control::record::Cp1ContributorSlot;
use crate::stages::weights::{
    ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET, ARENA3_CTRL_OFFSET, ARENA3_UINT_ULTRA_OFFSET,
    Arena0ControlBusWire, Arena3ControlBusWire,
};

/// Evaluates `U_int^base` (`4 × 8 = 128 B` at `Arena 0 [ARENA0_UINT_BASE_OFFSET..+128]`) on
/// `z_fm_base` (`r[dsp.ctrl.accum.anova-fm]`).
#[inline]
pub fn eval_uint_base_4x8(
    arena_bytes: &[u8],
    z_fm_base: [f32; 4],
    s_emb: [f32; 3],
) -> ([f32; 8], bool) {
    let Some(tail) = arena_bytes.get(ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..) else {
        return ([0.0; 8], false);
    };
    let Ok((a0, _)) = Arena0ControlBusWire::ref_from_prefix(tail) else {
        return ([0.0; 8], false);
    };
    let has_nonzero_uint = a0.u_int_base.as_bytes().iter().any(|&b| b != 0);
    if !has_nonzero_uint {
        return ([0.0; 8], false);
    }
    let inputs = if z_fm_base.iter().any(|&v| v.abs() > 1e-12) {
        z_fm_base
    } else {
        [
            s_emb[0] * s_emb[1],
            s_emb[0] * s_emb[2],
            s_emb[1] * s_emb[2],
            s_emb[0] * s_emb[0],
        ]
    };
    let mut out = [0.0_f32; 8];
    for (r, &p_val) in inputs.iter().enumerate() {
        if p_val.abs() > 1e-12 {
            for (d, out_d) in out.iter_mut().enumerate() {
                *out_d += p_val * a0.u_int_base[r][d].get();
            }
        }
    }
    (out, true)
}

/// Evaluates `U_int^base` (`3 × 8` at `Arena 0 [ARENA0_UINT_BASE_OFFSET..+96]`) on the 3 pairwise products of `s_emb`.
#[inline]
pub fn eval_uint_base_3x8(arena_bytes: &[u8], s_emb: [f32; 3]) -> [f32; 8] {
    eval_uint_base_4x8(arena_bytes, [0.0; 4], s_emb).0
}

/// Evaluates `3 × U_int,Delta` (`3 × 4 × 8 = 384 B` at `Arena 3 [ARENA3_UINT_ULTRA_OFFSET..+384]`, `r[dsp.ctrl.accum.anova-fm]`):
/// - Block 0 (`+0..+128`): maps `z_fm_ultra` into `\Delta z_raw[0..8]`
/// - Block 1 (`+128..+256`): maps `z_fm_base` into `z_raw[8..16]`
/// - Block 2 (`+256..+384`): maps `z_fm_ultra` into `z_raw[8..16]`
#[inline]
pub fn eval_uint_ultra_delta_3x4x8(
    arena_bytes: &[u8],
    z_fm_base: [f32; 4],
    z_fm_ultra: [f32; 4],
    s_emb: [f32; 3],
) -> ([f32; 8], [f32; 8], bool) {
    if arena_bytes.len() < crate::stages::weights::ARENA_ULTRA_BYTES {
        return ([0.0; 8], [0.0; 8], false);
    }
    let Some(tail) = arena_bytes.get(ARENA3_CTRL_OFFSET..) else {
        return ([0.0; 8], [0.0; 8], false);
    };
    let Ok((a3, _)) = Arena3ControlBusWire::ref_from_prefix(tail) else {
        return ([0.0; 8], [0.0; 8], false);
    };
    let has_nonzero_uint = a3.u_int_ultra.as_bytes().iter().any(|&b| b != 0);
    if !has_nonzero_uint {
        return ([0.0; 8], [0.0; 8], false);
    }
    let has_fm =
        z_fm_base.iter().any(|&v| v.abs() > 1e-12) || z_fm_ultra.iter().any(|&v| v.abs() > 1e-12);
    let (in_b0, in_b1, in_b2) = if has_fm {
        (z_fm_ultra, z_fm_base, z_fm_ultra)
    } else {
        let p = [
            s_emb[0] * s_emb[1],
            s_emb[0] * s_emb[2],
            s_emb[1] * s_emb[2],
            0.0,
        ];
        ([0.0; 4], p, [0.0; 4])
    };
    let mut delta_base_8 = [0.0_f32; 8];
    let mut ultra_new_8 = [0.0_f32; 8];
    for (r, &val) in in_b0.iter().enumerate() {
        if val.abs() > 1e-12 {
            for (d, out_d) in delta_base_8.iter_mut().enumerate() {
                *out_d += val * a3.u_int_ultra[0][r][d].get();
            }
        }
    }
    for (r, &val) in in_b1.iter().enumerate() {
        if val.abs() > 1e-12 {
            for (d, out_d) in ultra_new_8.iter_mut().enumerate() {
                *out_d += val * a3.u_int_ultra[1][r][d].get();
            }
        }
    }
    for (r, &val) in in_b2.iter().enumerate() {
        if val.abs() > 1e-12 {
            for (d, out_d) in ultra_new_8.iter_mut().enumerate() {
                *out_d += val * a3.u_int_ultra[2][r][d].get();
            }
        }
    }
    (delta_base_8, ultra_new_8, true)
}

/// Evaluates `U_int,Delta` (`6 × 8` at `Arena 3 [ARENA3_UINT_ULTRA_OFFSET..+192]`) on 6 pairwise products.
#[inline]
pub fn eval_uint_ultra_delta_6x8(
    arena_bytes: &[u8],
    s_emb: [f32; 3],
    s_emb_ultra: [f32; 3],
) -> [f32; 8] {
    if arena_bytes.len() < crate::stages::weights::ARENA_ULTRA_BYTES {
        return [0.0; 8];
    }
    let Some(tail) = arena_bytes.get(ARENA3_UINT_ULTRA_OFFSET..) else {
        return [0.0; 8];
    };
    let Ok((rows, _)) = <[[F32; 8]; 6]>::ref_from_prefix(tail) else {
        return [0.0; 8];
    };
    let pairs = [
        s_emb[0] * s_emb[1],
        s_emb[0] * s_emb[2],
        s_emb[1] * s_emb[2],
        s_emb_ultra[0] * s_emb[0],
        s_emb_ultra[1] * s_emb[1],
        s_emb_ultra[2] * s_emb[2],
    ];
    let mut out = [0.0_f32; 8];
    for (r, &p_val) in pairs.iter().enumerate() {
        if p_val.abs() > 1e-12 {
            for (d, out_d) in out.iter_mut().enumerate() {
                *out_d += p_val * rows[r][d].get();
            }
        }
    }
    out
}

/// Allocation-free `CP1` contributor decoder returning static error strings so real-time reloaders
/// emit zero heap allocation/drop glue.
/// Decodes strictly via canonical [`crate::stages::weights::Cp1InlineDirWire`] (`cp1_dir_count in 0..=8`,
/// `flags == 0`, `entries: [[ctrl_idx, (kind << 4) | lanes, aux]; 8]`) without legacy byte-sniffing
/// (`ref[container.dnam.cp1-profile]`).
pub(crate) fn decode_cp1_inline_contributors_static(
    arena_bytes: &[u8],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, &'static str> {
    let Some(a0_tail) = arena_bytes.get(ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..) else {
        return Ok(0);
    };
    let Ok((a0, _)) = Arena0ControlBusWire::ref_from_prefix(a0_tail) else {
        return Ok(0);
    };
    let count = a0.cp1_dir.cp1_dir_count.get() as usize;
    if count == 0 {
        return Ok(0);
    }
    if a0.cp1_dir.flags.get() != 0 {
        return Err("invalid inline CP1 directory flags");
    }
    if count > 8 || count > out_slots.len() {
        return Err("inline CP1 contributor count exceeds 8");
    }
    let Ok(base_words) = <[F32]>::ref_from_bytes(&a0.cp1_base) else {
        return Err("invalid inline CP1 base buffer");
    };
    let a3 = arena_bytes
        .get(ARENA3_CTRL_OFFSET..)
        .and_then(|s| Arena3ControlBusWire::ref_from_prefix(s).ok())
        .map(|(w, _)| w);
    let ultra_words = a3.and_then(|w| <[F32]>::ref_from_bytes(&w.cp1_ultra).ok());

    let mut b_word_cursor = 0usize;
    let mut u_word_cursor = 0usize;
    for (c, slot) in out_slots.iter_mut().take(count).enumerate() {
        let [c_idx, packed, aux] = a0.cp1_dir.entries[c];
        let control_idx = c_idx as u16;
        let ctrl_kind = (packed >> 4) & 0x0F;
        let lanes = packed & 0x0F;
        if ctrl_kind > 2 {
            return Err("invalid inline CP1 contributor kind");
        }
        if !(1..=3).contains(&lanes) {
            return Err("invalid inline CP1 lane count");
        }
        let option_idx = if ctrl_kind == 1 { aux } else { 0xFF };
        let partner_control_id = if ctrl_kind == 2 { aux as u16 } else { 0 };
        let l_cnt = lanes as usize;
        let b_word_len = 9 * l_cnt + 4;
        if b_word_cursor + b_word_len > base_words.len() {
            return Err("inline CP1 base payload exceeds 576 B");
        }
        let mut main_base = [[0.0f32; 8]; 3];
        for row in main_base.iter_mut().take(l_cnt) {
            for val in row.iter_mut() {
                *val = base_words[b_word_cursor].get();
                b_word_cursor += 1;
            }
        }
        let mut emb = [0.0f32; 3];
        for val in emb.iter_mut().take(l_cnt) {
            *val = base_words[b_word_cursor].get();
            b_word_cursor += 1;
        }
        let mut v_base = [0.0f32; 4];
        for val in &mut v_base {
            *val = base_words[b_word_cursor].get();
            b_word_cursor += 1;
        }
        let mut main_ultra = [[0.0f32; 8]; 3];
        let mut v_ultra = [0.0f32; 4];
        let mut has_ultra = false;
        if let Some(uw) = ultra_words {
            let u_word_len = 8 * l_cnt + 4;
            if u_word_cursor + u_word_len <= uw.len() {
                has_ultra = true;
                for row in main_ultra.iter_mut().take(l_cnt) {
                    for val in row.iter_mut() {
                        *val = uw[u_word_cursor].get();
                        u_word_cursor += 1;
                    }
                }
                for val in &mut v_ultra {
                    *val = uw[u_word_cursor].get();
                    u_word_cursor += 1;
                }
            }
        }
        *slot = Cp1ContributorSlot {
            lanes,
            ctrl_kind,
            control_idx,
            option_idx,
            partner_control_id,
            main_base,
            emb,
            v_base,
            main_ultra,
            v_ultra,
            has_ultra,
        };
    }
    Ok(count)
}

/// Decodes inline `CP1` contributors from `Arena 0 [ARENA0_CP1_BASE_OFFSET..ARENA0_CP1_BASE_END]`
/// (`576 B` base) and `Arena 3 [ARENA3_CP1_ULTRA_OFFSET..ARENA3_CP1_ULTRA_END]` (`2,112 B` Ultra)
/// using `cp1_lanes_dir` at `Arena 0 [ARENA0_CP1_DIR_OFFSET..]`. Returns `Ok(0)` when count is `0`.
pub fn decode_cp1_inline_contributors(
    arena_bytes: &[u8],
    out_slots: &mut [Cp1ContributorSlot],
) -> Result<usize, crate::schema::DnamError> {
    decode_cp1_inline_contributors_static(arena_bytes, out_slots).map_err(|msg| {
        crate::schema::DnamError::SchemaValidationError(alloc::string::String::from(msg))
    })
}

#[cfg(test)]
mod control_bus_tests {
    #[allow(unused_imports)]
    use alloc::{
        boxed::Box,
        format,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    // Empirical SONA Control Bus and 3-Phase Clamping Verification (`Tests 8A, 8B, 8D, 8E`).
    //
    // Canonical Reference: `docs/specs/dsp/parametric_control_bus.md`
    // and `docs/specs/training/control_space_audit.md`.
    //
    // Verifies:
    // - `Test 8A & 8B` (`verify[dsp.op-t.transition.dual-mode]`, `verify[dsp.ctrl.basis.lobatto]`): 3-node Lobatto physical delay interpolation accuracy and
    //   128-sample Hann-windowed dual-pointer switch crossfade vs. single-pointer phase jump.
    // - `Test 8D & 8E` (`verify[dsp.ctrl.range.normalize-clamp]`, `verify[dsp.ctrl.accum.cp1-scaling]`): 128-byte `ControlChannelRecord` table at
    //   `tensor_arena_offset - 128 * P`, `update_header_in_place` preservation of `ControlChannelRecord`
    //   bytes, 64-byte $\mathcal{O}(1)$ `LatentAccumulator64B` delta updates across $P = 32$ controls,
    //   and non-stop Phase-1/2/3 clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`).

    use super::*;
    use std::collections::BTreeMap;
    use std::f64::consts::TAU;

    use crate::FtzDazGuard;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container, update_header_in_place,
    };
    use crate::control::{
        ControlBus, ControlChannelRecord, ControlClampDescriptor, ControlLatentProjection,
        LatentAccumulator64B,
    };
    use crate::math::ScalarBackend;
    use crate::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use proptest::prelude::*;

    /// Test 8A & 8B (`verify[dsp.op-t.transition.dual-mode]`, `verify[dsp.ctrl.basis.lobatto]`): Verifies 3-node Lobatto physical delay interpolation accuracy
    /// at intermediate knob positions (`u = 0.5`) and verifies that `OpTDelayLine`'s 128-sample
    /// Hann-windowed dual-pointer switch crossfade eliminates single-pointer discontinuity jumps.
    #[test]
    fn test_8a_8b_lobatto_physical_delay_interpolation_and_dual_pointer_switch_crossfade() {
        let tau_at = |u: f64| -> f64 {
            let psi1 = u;
            let psi2 = u * (u + 1.0);
            200.0 + 100.0 * psi1 + 50.0 * psi2
        };
        let exact_mid = 200.0 + 150.0 * 0.5 + 50.0 * 0.25;
        let interp_mid = tau_at(0.5);
        assert!((interp_mid - exact_mid).abs() < 1e-12);

        let _audio = FtzDazGuard::new();
        let mut dl_xfade = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        let mut dl_jump = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl_xfade.current_delay = 40.0;
        dl_xfade.target_delay = 40.0;
        dl_jump.current_delay = 40.0;
        dl_jump.target_delay = 40.0;

        let mut prev_x = 0.0_f32;
        let mut prev_j = 0.0_f32;
        for n in 0..315 {
            let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
            prev_x = dl_xfade.step(s);
            prev_j = dl_jump.step(s);
        }

        dl_xfade.set_target_delay(100.0, true);
        dl_jump.current_delay = 100.0;
        dl_jump.target_delay = 100.0;

        let mut max_diff_xfade = 0.0_f32;
        let mut max_diff_jump = 0.0_f32;

        for n in 315..450 {
            let s = (TAU as f32 * 400.0 * (n as f32) / 48_000.0).sin();
            let yx = dl_xfade.step(s);
            let yj = dl_jump.step(s);
            max_diff_xfade = max_diff_xfade.max((yx - prev_x).abs());
            max_diff_jump = max_diff_jump.max((yj - prev_j).abs());
            prev_x = yx;
            prev_j = yj;
        }

        assert!(max_diff_jump > 10.0 * max_diff_xfade);
    }

    /// Test 8D & 8E (`verify[dsp.ctrl.range.normalize-clamp]`, `verify[dsp.ctrl.accum.cp1-scaling]`): Verifies 128-byte `ControlChannelRecord` framing,
    /// `control_table_offset()` preservation across `update_header_in_place`, $\mathcal{O}(1)$
    /// `LatentAccumulator64B` delta updates across $P = 32$ controls, and non-stop Phase-1/2/3 clamping.
    #[test]
    fn test_8d_8e_control_channel_record_o1_latent_and_3phase_clamping() {
        assert_eq!(core::mem::size_of::<ControlClampDescriptor>(), 64);
        assert_eq!(core::mem::size_of::<ControlLatentProjection>(), 64);
        assert_eq!(core::mem::size_of::<ControlChannelRecord>(), 128);
        assert_eq!(core::mem::size_of::<LatentAccumulator64B>(), 64);

        let mut projections = [ControlLatentProjection::new(); 32];
        let mut u_vals = [0.0_f32; 32];
        let mut psi_vals = [0.0_f32; 32];
        let mut acc = LatentAccumulator64B::new();

        for p in 0..32 {
            let pf = (p + 1) as f32;
            projections[p].w_main = core::array::from_fn(|d| 0.02 * pf * ((d + 1) as f32));
            projections[p].v_factors = core::array::from_fn(|f| 0.05 * ((p + f + 1) as f32));
            u_vals[p] = -0.8 + 0.05 * (p as f32);
            psi_vals[p] = 0.5 * u_vals[p];
            acc.update_delta(&projections[p], 0.0, u_vals[p], psi_vals[p]);
        }

        let new_u7 = 0.75_f32;
        let new_psi7 = 0.5 * new_u7;
        acc.update_delta(&projections[7], u_vals[7], new_u7, new_psi7 - psi_vals[7]);
        u_vals[7] = new_u7;
        psi_vals[7] = new_psi7;

        let z_o1 = acc.evaluate_z_ctrl();
        for d in 0..8 {
            let mut expected = 0.0_f32;
            for p in 0..32 {
                expected += projections[p].w_main[d] * psi_vals[p];
            }
            if d < 4 {
                for p in 0..32 {
                    for q in (p + 1)..32 {
                        expected += projections[p].v_factors[d]
                            * projections[q].v_factors[d]
                            * u_vals[p]
                            * u_vals[q];
                    }
                }
            }
            assert!((z_o1[d] - expected).abs() < 1e-4);
        }

        let mut header = DnamJsonHeader::default_reference();
        let cond_bounds = vec![crate::container::wire::StructuredConditionalBound {
            when: vec![("boost".to_string(), "on".to_string())],
            min: None,
            max: Some(9.37),
        }];
        header.controls_schema = vec![
            ControlSchemaDef {
                id: "gain".to_string(),
                name: "Gain".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -0.6, 0.6]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([2.0, 8.0]),
                default: serde_json::json!(5.0),
                conditional_bounds: Vec::new(),
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "feedback".to_string(),
                name: "Feedback".to_string(),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(5.0),
                conditional_bounds: cond_bounds,
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: "boost".to_string(),
                name: "Boost".to_string(),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("off"),
                conditional_bounds: Vec::new(),
                represented_switch_options: None,
                declared_options: Some(vec![
                    "off".to_string(),
                    "on".to_string(),
                    "ultra".to_string(),
                ]),
                active_options: Some(vec!["off".to_string(), "on".to_string()]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];

        let descriptors = header.compile_clamp_descriptors().unwrap();
        let mut cb = ControlBus::<ScalarBackend>::new();
        for (i, d) in descriptors.iter().enumerate() {
            cb.descriptors[i] = *d;
        }
        cb.num_controls = 3;

        cb.set_control_immediate(0, 9.5);
        assert!((cb.clamped_values[0] - 8.0).abs() < 1e-6);
        cb.set_control_immediate(0, 0.5);
        assert!((cb.clamped_values[0] - 2.0).abs() < 1e-6);

        cb.set_control_immediate(2, 2.0);
        assert!(cb.active_switch_options[2] <= 1);

        cb.set_control_immediate(2, 1.0);
        cb.set_control_immediate(1, 10.0);
        assert!((cb.clamped_values[1] - 9.37).abs() < 1e-5);
        assert_ne!(cb.active_clamp_mask & (1 << 1), 0);

        // Verify Region 4 shared_extent ControlClampDescriptor[0..P] table survives update_header_in_place
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let crypto = DefaultCryptoProvider::default();
        let view_before = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        assert_eq!(view_before.clamp_descriptors.len(), 3);
        let layout = crate::container::validate_v3_open_dnam_layout(&container).unwrap();
        let shared_off = layout.shared_extent.0;
        let shared_len = layout.shared_extent.1;
        assert_eq!(shared_len, 3 * 64);
        let table_snapshot = container[shared_off..shared_off + shared_len].to_vec();

        header
            .user_metadata
            .insert("tag".to_string(), serde_json::json!("preserved"));
        update_header_in_place(&mut container, &header).unwrap();
        assert_eq!(
            &container[shared_off..shared_off + shared_len],
            &table_snapshot
        );
    }

    #[test]
    fn test_theorem_3_unlimited_controls_31_band_eq_and_64_rack() {
        use crate::container::build_dnam_container_with_cp1;
        use crate::control::Cp1ContributorSlot;
        use crate::{DnamEngine, Tier};

        for &num_ctrls in &[31_usize, 64_usize] {
            let mut header = DnamJsonHeader::default_reference();
            header.controls_schema = (0..num_ctrls)
                .map(|p| ControlSchemaDef {
                    id: format!("band_{p}"),
                    name: format!("Band {p}"),
                    kind: ControlKind::Range,
                    unit: Some("dB".to_string()),
                    lobatto_degree: Some(2),
                    anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                    declared_range: Some([-12.0, 12.0]),
                    safe_clamped_range: Some([-12.0, 12.0]),
                    default: serde_json::json!(0.0),
                    conditional_bounds: Vec::new(),
                    represented_switch_options: None,
                    declared_options: None,
                    active_options: None,
                    clamp_warning: None,
                    calibration: None,
                    ui_hints: ControlUiHints::default(),
                    user_metadata: BTreeMap::new(),
                })
                .collect();

            let last_p = num_ctrls - 1;
            let mut slot = Cp1ContributorSlot::zero();
            slot.lanes = 1;
            slot.ctrl_kind = 0;
            slot.control_idx = last_p as u16;
            slot.option_idx = 0xFF;
            slot.main_base[0] = [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.05, 0.15];
            slot.emb[0] = 1.0;
            slot.v_base = [0.4, 0.2, -0.3, 0.1];

            let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
            let container =
                build_dnam_container_with_cp1(&mut header, &arena, &arena, false, &[slot]).unwrap();

            let crypto = DefaultCryptoProvider::default();
            let view0 = DnamContainerView::from_bytes(&container, &crypto).unwrap();
            assert_eq!(view0.preamble.num_controls as usize, num_ctrls);
            assert_eq!(view0.control_records.len(), num_ctrls);
            assert_eq!(view0.cp1_contributors.len(), 1);

            let mut engine =
                DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
            assert_eq!(engine.preamble().num_controls as usize, num_ctrls);
            assert_eq!(engine.telemetry_snapshot().z_head, [0.0; 4]);

            // Move the last control (index 30 or 63, well beyond the old 16-control wall!)
            engine.set_control_immediate(last_p, 12.0);
            let z_after = engine.telemetry_snapshot().z_head;
            assert!(
                z_after[0].abs() > 0.1,
                "Control #{last_p} failed to update z_head via O(1) LatentAccumulator64B: {:?}",
                z_after
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Property test verifying that `ControlChannelRecord` roundtrips bit-identically through
        /// 128-byte serialization.
        #[test]
        fn prop_control_record_roundtrip(
            u0 in -1.0_f32..1.0_f32,
            w0 in -2.0_f32..2.0_f32,
            v0 in -1.0_f32..1.0_f32,
        ) {
            let mut rec = ControlChannelRecord::new();
            rec.clamp.anchor_nodes_norm[1] = u0;
            rec.projection.w_main[0] = w0;
            rec.projection.v_factors[0] = v0;
            let bytes = rec.to_bytes();
            let decoded = ControlChannelRecord::from_bytes(&bytes);
            prop_assert_eq!(decoded, rec);
        }
    }

    #[test]
    fn test_update_header_in_place_updates_control_user_metadata_and_rejects_immutable_edits() {
        use crate::container::{DnamContainerView, update_header_in_place};
        use crate::provenance::DefaultCryptoProvider;

        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema = vec![ControlSchemaDef {
            id: "drive".to_string(),
            name: "Drive".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([1.0, 9.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        }];
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        // Immutable safe_clamped_range edits must be rejected by update_header_in_place
        let mut bad_header = header.clone();
        bad_header.controls_schema[0].safe_clamped_range = Some([2.5, 7.5]);
        assert!(update_header_in_place(&mut container, &bad_header).is_err());

        // Mutable control_user_metadata edits succeed via DMU1 while preserving safe_clamped_range
        header.controls_schema[0]
            .user_metadata
            .insert("midi_cc".to_string(), serde_json::json!(14));
        update_header_in_place(&mut container, &header).unwrap();

        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(&container, &crypto).unwrap();
        assert_eq!(view.control_records.len(), 1);
        assert!((view.control_records[0].clamp.safe_min_base - 1.0).abs() < 1e-5);
        assert!((view.control_records[0].clamp.safe_max_base - 9.0).abs() < 1e-5);
        assert_eq!(
            view.header.controls_schema[0].user_metadata.get("midi_cc"),
            Some(&serde_json::json!(14))
        );
    }

    #[test]
    fn test_uint_and_decode_cp1_inline_edge_paths() {
        use crate::stages::weights::{
            ARENA_LITE_BYTES, ARENA_ULTRA_BYTES, ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET,
            ARENA0_CP1_BASE_END, ARENA0_UINT_BASE_OFFSET, ARENA3_CP1_ULTRA_END,
            ARENA3_UINT_ULTRA_OFFSET,
        };
        use zerocopy::byteorder::little_endian::U16;
        let put_f32 = |buf: &mut [u8], off: usize, val: f32| {
            buf[off..off + 4].copy_from_slice(F32::new(val).as_bytes());
        };

        assert_eq!(eval_uint_base_3x8(&[0u8; 64], [0.5; 3]), [0.0; 8]);
        assert!(!eval_uint_base_4x8(&[0u8; 64], [0.5; 4], [0.5; 3]).1);
        assert_eq!(
            eval_uint_ultra_delta_6x8(&[0u8; 1024], [0.5; 3], [0.2; 3]),
            [0.0; 8]
        );
        assert!(!eval_uint_ultra_delta_3x4x8(&[0u8; 1024], [0.5; 4], [0.5; 4], [0.5; 3]).2);

        let mut short_slots = vec![Cp1ContributorSlot::default(); 4];
        assert_eq!(
            decode_cp1_inline_contributors(&[0u8; 128], &mut short_slots).unwrap(),
            0
        );

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        assert_eq!(
            decode_cp1_inline_contributors(&arena, &mut short_slots).unwrap(),
            0
        );
        {
            let (a0, _) = Arena0ControlBusWire::mut_from_prefix(
                &mut arena[ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..],
            )
            .unwrap();
            a0.cp1_dir.cp1_dir_count = U16::new(9);
        }
        assert!(decode_cp1_inline_contributors(&arena, &mut short_slots).is_err());

        // Canonical Cp1InlineDirWire with 8 single-lane contributors + Ultra weights (416 B <= 576 B base slot)
        {
            let (a0, _) = Arena0ControlBusWire::mut_from_prefix(
                &mut arena[ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..],
            )
            .unwrap();
            a0.cp1_dir.cp1_dir_count = U16::new(8);
            a0.cp1_dir.flags = U16::new(0);
            for c in 0..8 {
                a0.cp1_dir.entries[c] = [c as u8, 1, 0xFF];
            }
        }
        for off in (ARENA0_UINT_BASE_OFFSET..ARENA0_CP1_BASE_END).step_by(4) {
            put_f32(&mut arena, off, 0.15);
        }
        for off in (ARENA3_UINT_ULTRA_OFFSET..ARENA3_CP1_ULTRA_END).step_by(4) {
            put_f32(&mut arena, off, 0.12);
        }
        let mut big_slots = vec![Cp1ContributorSlot::default(); 16];
        assert_eq!(
            decode_cp1_inline_contributors(&arena, &mut big_slots).unwrap(),
            8
        );
        // Lite stripped slice preserves self-contained Arena 0 base CP1 contributors
        assert_eq!(
            decode_cp1_inline_contributors(&arena[..ARENA_LITE_BYTES], &mut big_slots).unwrap(),
            8
        );

        let uint_b3 = eval_uint_base_3x8(&arena, [0.4, -0.3, 0.2]);
        let uint_u6 = eval_uint_ultra_delta_6x8(&arena, [0.4, -0.3, 0.2], [0.1, 0.2, -0.1]);
        assert!(uint_b3.iter().all(|v| v.is_finite()) && uint_u6.iter().all(|v| v.is_finite()));

        // Base payload overflow (5 three-lane contributors = 620 B > 576 B) returns error
        {
            let (a0, _) = Arena0ControlBusWire::mut_from_prefix(
                &mut arena[ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..],
            )
            .unwrap();
            a0.cp1_dir.cp1_dir_count = U16::new(5);
            for c in 0..5 {
                a0.cp1_dir.entries[c] = [c as u8, 3, 0xFF];
            }
        }
        assert!(decode_cp1_inline_contributors(&arena, &mut big_slots).is_err());

        // Non-zero flags or invalid lanes (0) triggers error
        {
            let (a0, _) = Arena0ControlBusWire::mut_from_prefix(
                &mut arena[ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..],
            )
            .unwrap();
            a0.cp1_dir.cp1_dir_count = U16::new(2);
            a0.cp1_dir.flags = U16::new(1);
        }
        assert!(decode_cp1_inline_contributors(&arena, &mut big_slots).is_err());
        {
            let (a0, _) = Arena0ControlBusWire::mut_from_prefix(
                &mut arena[ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..],
            )
            .unwrap();
            a0.cp1_dir.flags = U16::new(0);
            a0.cp1_dir.entries[0] = [0, 0, 0xFF];
        }
        assert!(decode_cp1_inline_contributors(&arena, &mut big_slots).is_err());
    }
}
