//! Canonical `CP1` Lobatto ANOVA, exact-subtraction Rank-4/8 FM regression, and 31-slot
//! orthogonal latent dispatch factorization into `W_ctrl` / `"CEXT"` / `W_ctrl_ultra`
//! (`ref[dsp.ctrl.accum.anova-fm]`, `ref[dsp.ctrl.dispatch.matrix]`,
//! `ref[dsp.ctrl.calib.variance-gating]`).

mod ultra;

use dnam_vm::bin::{
    Arena0ControlBusWire, Arena3ControlBusWire, CextDispatchWire, ControlClampDescriptorWire,
    TempoSyncEntryWire,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::control::cp1::Cp1ContributorSlot;
use dnam_vm::control::descriptor::ControlClampDescriptor;
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::F32;

use crate::error::TrainError;

use super::super::clamp::{
    ControlKindSpec, ControlSpec, Phase1ClampOutput, StageStabilityConstraint,
    audit_phase1_controls,
};
use super::subspace::{
    ACTIVE_DISPATCH_SLOTS, RANK_BASE_FM, RANK_STATIC_LATENT, contributor_features_matrix,
    factor_gauge_invariant_dispatch_subspace, fit_fm_als_u_and_v, slot_cooccurrence_mask,
    solve_ols_ridge_uplo,
};
use super::wire::pack_cp1_slots_into_wire;
use super::{Cp1ControlSample, Cp1FitConfig, Cp1WireBundle};
use ultra::{UltraFitInputs, UltraFitOutputs, fit_ultra_dispatch_and_fm};

const DELAY_LFO_GATE_SLOTS: [usize; 1] = [12];

fn eval_descriptor_norm(desc: &ControlClampDescriptor, val: f32) -> f64 {
    f64::from(desc.clamp_and_normalize(val, &[]).1)
}

fn build_cp1_slots(
    specs: &[ControlSpec],
    locked_controls: &[bool],
    descs: &[ControlClampDescriptor],
    fit_ultra: bool,
) -> (Vec<Cp1ContributorSlot>, usize) {
    let mut slots = Vec::new();
    let mut distinct_controls = 0_usize;
    for (c, sp) in specs.iter().enumerate() {
        if locked_controls[c] {
            continue;
        }
        match sp.kind {
            ControlKindSpec::Range { .. } => {
                let lanes = descs[c].lobatto_degree.min(3);
                if lanes == 0 {
                    continue;
                }
                distinct_controls += 1;
                let mut s = Cp1ContributorSlot::zero();
                s.control_idx = c as u16;
                s.ctrl_kind = 0;
                s.lanes = lanes;
                s.aux = 0xFF;
                s.emb = [1.0, 0.0, 0.0];
                s.has_ultra = fit_ultra;
                slots.push(s);
            }
            ControlKindSpec::Switch { num_options, .. } => {
                let def_opt = descs[c].default_switch_option();
                let obs_mask = descs[c].active_options_mask;
                let mut added = false;
                for o in 0..num_options.min(16) {
                    if o != def_opt && (obs_mask & (1_u16 << o)) != 0 {
                        let mut s = Cp1ContributorSlot::zero();
                        s.control_idx = c as u16;
                        s.ctrl_kind = 1;
                        s.lanes = 1;
                        s.aux = o;
                        s.has_ultra = fit_ultra;
                        slots.push(s);
                        added = true;
                    }
                }
                if added {
                    distinct_controls += 1;
                }
            }
        }
    }
    (slots, distinct_controls)
}

fn fit_and_pack_cp1_inner(
    specs: &[ControlSpec],
    locked_controls: &[bool],
    wire_descs: &[ControlClampDescriptorWire],
    samples: &[Cp1ControlSample],
    tsyn_rows: &[TempoSyncEntryWire],
    config: Cp1FitConfig,
) -> Result<Cp1WireBundle, TrainError> {
    let n_ctrls = specs.len();
    if locked_controls.len() != n_ctrls || wire_descs.len() != n_ctrls {
        return Err(TrainError::InvalidDimensions {
            context: "fit_and_pack_cp1::descriptors",
            expected: n_ctrls,
            actual: locked_controls.len().min(wire_descs.len()),
        });
    }
    if samples.is_empty() {
        return Err(TrainError::InvalidDimensions {
            context: "fit_and_pack_cp1::samples",
            expected: 1,
            actual: 0,
        });
    }

    let descs: Vec<ControlClampDescriptor> = wire_descs
        .iter()
        .map(ControlClampDescriptor::from_wire)
        .collect();

    let mut center_idx = 0_usize;
    let mut best_dist = f64::INFINITY;
    let mut norm_table = Vec::with_capacity(samples.len());
    for (si, samp) in samples.iter().enumerate() {
        if samp.phys_controls.len() != n_ctrls {
            return Err(TrainError::InvalidDimensions {
                context: "fit_and_pack_cp1::phys_controls",
                expected: n_ctrls,
                actual: samp.phys_controls.len(),
            });
        }
        let mut nv = vec![0.0_f64; n_ctrls];
        let mut dist_sq = 0.0_f64;
        for c in 0..n_ctrls {
            nv[c] = eval_descriptor_norm(&descs[c], samp.phys_controls[c]);
            let d = match specs[c].kind {
                ControlKindSpec::Range { .. } => nv[c] - f64::from(descs[c].anchor_nodes_norm[0]),
                ControlKindSpec::Switch { .. } => {
                    let sw = samp.switch_options.get(c).copied().unwrap_or(0);
                    if sw == descs[c].default_switch_option() {
                        0.0
                    } else {
                        1.0
                    }
                }
            };
            dist_sq += d * d;
        }
        if dist_sq < best_dist {
            best_dist = dist_sq;
            center_idx = si;
        }
        norm_table.push(nv);
    }

    let center_base = samples[center_idx].stage_targets_base;
    let center_ultra = samples[center_idx]
        .stage_targets_ultra_delta
        .unwrap_or([0.0; STAGE_TARGET_SLOTS]);
    let (mut slots, distinct_controls) =
        build_cp1_slots(specs, locked_controls, &descs, config.fit_ultra);

    let mut basis_cols: Vec<(usize, usize)> = Vec::new();
    for (si, s) in slots.iter().enumerate() {
        for l in 0..usize::from(s.lanes) {
            basis_cols.push((si, l));
        }
    }

    let n_obs = samples.len();
    let mut design_main = vec![vec![0.0_f64; basis_cols.len()]; n_obs];
    let mut target_31 = vec![vec![0.0_f64; ACTIVE_DISPATCH_SLOTS]; n_obs];
    let mut target_u_delta_31 = vec![vec![0.0_f64; ACTIVE_DISPATCH_SLOTS]; n_obs];
    for (i, samp) in samples.iter().enumerate() {
        for (col, &(si, l)) in basis_cols.iter().enumerate() {
            let sl = &slots[si];
            let c = usize::from(sl.control_idx);
            design_main[i][col] = if sl.ctrl_kind == 0 {
                f64::from(descs[c].eval_lobatto_basis(norm_table[i][c] as f32)[l])
            } else if samp.switch_options.get(c).copied().unwrap_or(0) == sl.aux {
                1.0
            } else {
                0.0
            };
        }
        if !samp.stage_targets_base.iter().all(|v| v.is_finite()) {
            return Err(TrainError::NonFiniteInput {
                context: "fit_and_pack_cp1_audited(stage_targets_base)",
            });
        }
        if let Some(ud) = samp.stage_targets_ultra_delta
            && !ud.iter().all(|v| v.is_finite())
        {
            return Err(TrainError::NonFiniteInput {
                context: "fit_and_pack_cp1_audited(stage_targets_ultra_delta)",
            });
        }
        for d in 0..ACTIVE_DISPATCH_SLOTS {
            target_31[i][d] = samp.stage_targets_base[d] - center_base[d];
            if let Some(ud) = samp.stage_targets_ultra_delta {
                target_u_delta_31[i][d] = ud[d] - center_ultra[d];
            }
        }
    }

    if !config.trust_radius_mh.is_finite() || config.trust_radius_mh <= 0.0 {
        return Err(TrainError::NonFiniteInput {
            context: "fit_and_pack_cp1_audited(trust_radius_mh)",
        });
    }

    for s_idx in DELAY_LFO_GATE_SLOTS {
        let mean: f64 = target_31.iter().map(|r| r[s_idx]).sum::<f64>() / (n_obs as f64);
        let var: f64 = target_31
            .iter()
            .map(|r| (r[s_idx] - mean).powi(2))
            .sum::<f64>()
            / (n_obs as f64);
        if var < config.delay_variance_gate_samples2 {
            for r in &mut target_31 {
                r[s_idx] = 0.0;
            }
            for r in &mut target_u_delta_31 {
                r[s_idx] = 0.0;
            }
        }
    }

    let (z_base_target_arr, w_static) =
        factor_gauge_invariant_dispatch_subspace(&target_31, config.trust_radius_mh);
    let z_base_target: Vec<Vec<f64>> = z_base_target_arr.iter().map(|r| r.to_vec()).collect();

    let mut a0_cb = Arena0ControlBusWire::new_zeroed();
    let mut a3_cb = Arena3ControlBusWire::new_zeroed();
    a0_cb.rho_drift_train = F32::new(0.999);
    a0_cb.trust_radius_mh = F32::new(config.trust_radius_mh.clamp(0.01, 16.0));
    a0_cb.rho_head_and_ctrl_tail[0] = a0_cb.trust_radius_mh;

    let w_main = solve_ols_ridge_uplo(&design_main, &z_base_target, config.l2_reg);
    for (col, &(si, l)) in basis_cols.iter().enumerate() {
        if let Some(row) = w_main.get(col) {
            for d in 0..8 {
                slots[si].main_base[l][d] = row[d] as f32;
            }
        }
    }

    if distinct_controls >= 2 {
        let phi = contributor_features_matrix(&slots, &norm_table, &descs, samples);
        let cooccurs = slot_cooccurrence_mask(&phi, slots.len());
        for (si, sl) in slots.iter_mut().enumerate() {
            if !cooccurs[si] {
                sl.v_base = [0.0; RANK_BASE_FM];
                sl.v_ultra = [0.0; RANK_BASE_FM];
                continue;
            }
            for f in 0..RANK_BASE_FM {
                let phase_b = ((si + 1) * (f + 1)) as f64;
                sl.v_base[f] = (0.35 * phase_b.sin() + 0.15) as f32;
                if config.fit_ultra {
                    let phase_u = ((si + 1) * (f + 5)) as f64;
                    sl.v_ultra[f] = (0.35 * phase_u.cos() - 0.12) as f32;
                }
            }
        }
    }

    let mut z_base_hat = vec![[0.0_f64; 8]; n_obs];
    for (i, d_row) in design_main.iter().enumerate() {
        for (col, &(si, l)) in basis_cols.iter().enumerate() {
            for d in 0..8 {
                z_base_hat[i][d] += d_row[col] * f64::from(slots[si].main_base[l][d]);
            }
        }
    }

    let fm_pair_base = if distinct_controls >= 2 {
        let inter_res: Vec<Vec<f64>> = z_base_target
            .iter()
            .zip(&z_base_hat)
            .map(|(t, h)| t.iter().zip(h).map(|(tv, hv)| tv - hv).collect())
            .collect();
        let (fm, u_int_fit) = fit_fm_als_u_and_v(
            &mut slots,
            &norm_table,
            &descs,
            samples,
            &inter_res,
            config.l2_reg,
            false,
        );
        for f in 0..RANK_BASE_FM {
            if let Some(row) = u_int_fit.get(f) {
                for d in 0..8 {
                    let w = row[d] as f32;
                    a0_cb.u_int_base[f][d] = F32::new(w);
                    for (i, fm_row) in fm.iter().enumerate() {
                        z_base_hat[i][d] += fm_row[f] * f64::from(w);
                    }
                }
            }
        }
        Some(fm)
    } else {
        None
    };

    let mut cext_wire = CextDispatchWire::new_zeroed();
    let mut need_cext = false;
    for (s_idx, row) in w_static.iter().enumerate() {
        for c in 0..RANK_STATIC_LATENT {
            let w = row[c] as f32;
            if s_idx < 16 {
                a0_cb.w_ctrl[s_idx][c] = F32::new(w);
            } else if c >= 4 {
                if w.abs() > 1.0e-7 {
                    need_cext = true;
                }
                cext_wire.w_ctrl_ext[s_idx - 16][c] = F32::new(w);
            }
        }
    }
    for row in &mut cext_wire.w_ctrl_ext {
        row[..4].fill(F32::new(0.0));
    }
    cext_wire.w_ctrl_ext[15].fill(F32::new(0.0));

    if config.fit_ultra {
        let inp = UltraFitInputs {
            basis_cols: &basis_cols,
            design_main: &design_main,
            norm_table: &norm_table,
            descs: &descs,
            samples,
            target_31: &target_31,
            target_u_delta_31: &target_u_delta_31,
            z_base_target: &z_base_target,
            z_base_hat: &z_base_hat,
            fm_pair_base: fm_pair_base.as_deref(),
            config,
            a0_cb: &a0_cb,
        };
        let mut out = UltraFitOutputs {
            slots: &mut slots,
            a3_cb: &mut a3_cb,
            cext_wire: &mut cext_wire,
            need_cext: &mut need_cext,
        };
        fit_ultra_dispatch_and_fm(&inp, &mut out);
    }

    let cp1_overflow =
        pack_cp1_slots_into_wire(&slots, n_ctrls, config.fit_ultra, &mut a0_cb, &mut a3_cb)?;

    Ok(Cp1WireBundle {
        arena0_cb: a0_cb,
        arena3_cb: a3_cb,
        cext: need_cext.then_some(cext_wire),
        tsyn: tsyn_rows.to_vec(),
        cp1_overflow,
        contributor_count: slots.len(),
        distinct_controls,
    })
}

/// Fits Lobatto `CP1` main-effect & Rank-4/8 FM ANOVA tables plus 31-slot `W_ctrl` / `"CEXT"`
/// dispatch matrices reusing an existing Phase-1 audit output (`ref[dsp.ctrl.accum.cp1-scaling]`).
pub fn fit_and_pack_cp1_audited(
    specs: &[ControlSpec],
    phase1: &Phase1ClampOutput,
    samples: &[Cp1ControlSample],
    tsyn_rows: &[TempoSyncEntryWire],
    config: Cp1FitConfig,
) -> Result<Cp1WireBundle, TrainError> {
    fit_and_pack_cp1_inner(
        specs,
        &phase1.locked_controls,
        &phase1.wire_descriptors,
        samples,
        tsyn_rows,
        config,
    )
}

/// Fits Lobatto `CP1` main-effect & Rank-4/8 FM ANOVA tables plus 31-slot `W_ctrl` / `"CEXT"`
/// dispatch matrices across `samples`, gating delay slot `12` (`tank_delay_mod`) below `0.2` samples²
/// and strictly zeroing `U_int` whenever fewer than `2` physical controls vary.
pub fn fit_and_pack_cp1(
    specs: &[ControlSpec],
    locked_controls: &[bool],
    samples: &[Cp1ControlSample],
    tsyn_rows: &[TempoSyncEntryWire],
    config: Cp1FitConfig,
) -> Result<Cp1WireBundle, TrainError> {
    let phase1 = audit_phase1_controls(specs, &StageStabilityConstraint::default())?;
    fit_and_pack_cp1_inner(
        specs,
        locked_controls,
        &phase1.wire_descriptors,
        samples,
        tsyn_rows,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::super::wire::build_tsyn_entry_wire;
    use super::*;
    use dnam_vm::bin::{CP1_STREAM_MAGIC, DnamBinView, encode_dnam_bin};
    use dnam_vm::control::bus::ControlBus;
    use dnam_vm::control::cp1::{count_distinct_contributor_controls, decode_cp1_contributors};
    use dnam_vm::math::{DefaultMathBackend, TANK_DC_BLOCKER_48K_F32};
    use dnam_vm::{DnamBinBuildInput, Tier};

    fn make_range_spec(min: f64, max: f64) -> ControlSpec {
        ControlSpec {
            kind: ControlKindSpec::Range {
                declared_min: min,
                declared_max: max,
                default_val: 0.5 * (min + max),
                obs_min: min,
                obs_max: max,
                unique_values: 5,
                lobatto_anchors: [min as f32, (0.5 * (min + max)) as f32, max as f32],
            },
        }
    }

    #[test]
    fn test_single_control_starcenter_invariance_and_multicontrol_cp1_overflow() {
        let spec1 = vec![make_range_spec(0.0, 10.0)];
        let samps1: Vec<Cp1ControlSample> = [0.0_f32, 2.5, 5.0, 7.5, 10.0]
            .into_iter()
            .map(|u| {
                let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
                st[0] = 0.2 * f64::from(u);
                st[16] = 0.05 * f64::from(u);
                st[31] = 9.0;
                Cp1ControlSample {
                    phys_controls: vec![u],
                    switch_options: vec![0],
                    stage_targets_base: st,
                    stage_targets_ultra_delta: None,
                }
            })
            .collect();
        let phase1 = audit_phase1_controls(&spec1, &StageStabilityConstraint::default()).unwrap();
        let res1 = fit_and_pack_cp1_audited(&spec1, &phase1, &samps1, &[], Cp1FitConfig::default())
            .unwrap();
        assert!(res1.distinct_controls < 2);
        assert!(
            res1.arena0_cb
                .u_int_base
                .iter()
                .flatten()
                .all(|w| w.get() == 0.0)
        );
        let cext1 = res1.cext.as_ref().unwrap();
        for r in 0..16 {
            for c in 8..16 {
                assert_eq!(res1.arena0_cb.w_ctrl[r][c].get(), 0.0);
                assert_eq!(cext1.w_ctrl_ext[r][c].get(), 0.0);
            }
        }
        assert!(cext1.w_ctrl_ext[15].iter().all(|w| w.get() == 0.0));

        let mut a0_1 = DnamBinBuildInput::default_arena0_lite();
        a0_1.control_bus_lite = res1.arena0_cb;
        a0_1.control_bus_lite.tank_dry_mix_fp16 =
            zerocopy::byteorder::little_endian::U16::new(dnam_kernels::f32_to_fp16(1.0));
        let bin1 = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 40.0,
            ro_pole_48k: 0.9,
            switch_xfade_samples: 64,
            content_hash_prefix: [1; 8],
            controls: &phase1.wire_descriptors,
            cp1_overflow: &res1.cp1_overflow,
            tsyn: &res1.tsyn,
            cext: res1.cext.as_ref(),
            arena0: &a0_1,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let view1 = DnamBinView::parse(&bin1).unwrap();
        let mut bus =
            ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view1, Tier::Lite, 48_000.0)
                .unwrap();
        bus.set_control_physical(0, 5.0).unwrap();
        bus.eval_static_and_snap_if_dirty();
        assert_eq!(bus.stage_targets, [0.0_f32; STAGE_TARGET_SLOTS]);
        bus.set_control_physical(0, 7.5).unwrap();
        bus.eval_static_and_snap_if_dirty();
        assert!(
            (bus.stage_targets[0] - 0.5).abs() < 0.02,
            "slot 0={}",
            bus.stage_targets[0]
        );
        assert_eq!(bus.stage_targets[31], 0.0);

        let specs10: Vec<ControlSpec> = (0..10).map(|_| make_range_spec(-1.0, 1.0)).collect();
        let mut samps10 = Vec::new();
        for k in 0..24 {
            let mut phys = vec![0.0_f32; 10];
            for (c, slot) in phys.iter_mut().enumerate() {
                *slot = (((k * (c + 1)) % 7) as f32 - 3.0) / 3.0;
            }
            let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
            let mut sud = [0.0_f64; STAGE_TARGET_SLOTS];
            st[0] = f64::from(phys[0]) * 0.4 + f64::from(phys[1]) * 0.2 * f64::from(phys[0]);
            st[22] = -0.3 * f64::from(phys[2]);
            st[30] = 0.8 * f64::from(phys[3]);
            sud[0] = 0.1 * f64::from(phys[0]);
            samps10.push(Cp1ControlSample {
                phys_controls: phys,
                switch_options: vec![0; 10],
                stage_targets_base: st,
                stage_targets_ultra_delta: Some(sud),
            });
        }
        let tsyn = vec![
            build_tsyn_entry_wire(0, true, 1.0, 1.0, [-1.0, 0.0, 1.0], (-1.0, 1.0), true).unwrap(),
        ];
        let p1_10 = audit_phase1_controls(&specs10, &StageStabilityConstraint::default()).unwrap();
        let res10 =
            fit_and_pack_cp1_audited(&specs10, &p1_10, &samps10, &tsyn, Cp1FitConfig::default())
                .unwrap();
        assert!(res10.cp1_overflow.starts_with(&CP1_STREAM_MAGIC));
        assert!(res10.cext.is_some());
        let mut decoded10 = [Cp1ContributorSlot::zero(); 16];
        let m10 = decode_cp1_contributors(
            &res10.arena0_cb,
            Some(&res10.arena3_cb),
            &res10.cp1_overflow,
            10,
            &mut decoded10,
        )
        .unwrap();
        assert_eq!(m10, 10);
        assert_eq!(count_distinct_contributor_controls(&decoded10[..m10]), 10);
        assert_ne!(decoded10[0].v_base, decoded10[0].v_ultra);

        // NaN trust_radius_mh and non-finite stage targets must fail closed; slot 17 octaves not gated.
        assert!(matches!(
            fit_and_pack_cp1_audited(
                &specs10,
                &p1_10,
                &samps10,
                &tsyn,
                Cp1FitConfig {
                    trust_radius_mh: f32::NAN,
                    ..Cp1FitConfig::default()
                },
            ),
            Err(TrainError::NonFiniteInput { .. })
        ));
        let mut bad_samps = samps10.clone();
        bad_samps[0].stage_targets_base[0] = f64::NAN;
        assert!(matches!(
            fit_and_pack_cp1_audited(&specs10, &p1_10, &bad_samps, &tsyn, Cp1FitConfig::default()),
            Err(TrainError::NonFiniteInput { .. })
        ));
        assert_eq!(DELAY_LFO_GATE_SLOTS, [12]);
    }

    #[test]
    fn test_unobserved_switch_pairs_zero_fm_factors_and_preserve_additive_corner() {
        let specs = vec![
            make_range_spec(-1.0, 1.0),
            make_range_spec(-1.0, 1.0),
            ControlSpec {
                kind: ControlKindSpec::Switch {
                    num_options: 3,
                    default_opt: 0,
                    declared_mask: 0b111,
                    observed_mask: 0b111,
                },
            },
            ControlSpec {
                kind: ControlKindSpec::Switch {
                    num_options: 3,
                    default_opt: 1,
                    declared_mask: 0b111,
                    observed_mask: 0b111,
                },
            },
        ];
        let mut samps = Vec::new();
        let make_pt = |r0: f32, r1: f32, sw2: u8, sw3: u8| {
            let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
            let s2 = match sw2 {
                1 => 0.45,
                2 => -0.55,
                _ => 0.0,
            };
            let s3 = match sw3 {
                0 => -0.40,
                2 => 0.50,
                _ => 0.0,
            };
            st[0] = 0.5 * f64::from(r0) + 0.35 * f64::from(r0 * r1);
            st[1] = 0.4 * f64::from(r1);
            st[2] = s2 + s3;
            Cp1ControlSample {
                phys_controls: vec![r0, r1, f32::from(sw2), f32::from(sw3)],
                switch_options: vec![0, 0, sw2, sw3],
                stage_targets_base: st,
                stage_targets_ultra_delta: None,
            }
        };
        samps.push(make_pt(0.0, 0.0, 0, 1));
        for (r0, r1) in [(1.0_f32, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            samps.push(make_pt(r0, r1, 0, 1));
        }
        samps.push(make_pt(0.0, 0.0, 1, 1));
        samps.push(make_pt(0.0, 0.0, 2, 1));
        samps.push(make_pt(0.0, 0.0, 0, 0));
        samps.push(make_pt(0.0, 0.0, 0, 2));

        let phase1 = audit_phase1_controls(&specs, &StageStabilityConstraint::default()).unwrap();
        let res = fit_and_pack_cp1_audited(
            &specs,
            &phase1,
            &samps,
            &[],
            Cp1FitConfig {
                l2_reg: 1.0e-5,
                ..Cp1FitConfig::default()
            },
        )
        .unwrap();
        let mut decoded = [Cp1ContributorSlot::zero(); 8];
        let count = decode_cp1_contributors(
            &res.arena0_cb,
            Some(&res.arena3_cb),
            &res.cp1_overflow,
            specs.len(),
            &mut decoded,
        )
        .unwrap();
        for sl in &decoded[..count] {
            if sl.ctrl_kind == 1 {
                assert_eq!(sl.v_base, [0.0; 4]);
                assert_eq!(sl.v_ultra, [0.0; 4]);
            }
        }

        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.control_bus_lite = res.arena0_cb;
        a0.control_bus_lite.tank_dry_mix_fp16 =
            zerocopy::byteorder::little_endian::U16::new(dnam_kernels::f32_to_fp16(1.0));
        let bin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 40.0,
            ro_pole_48k: 0.9,
            switch_xfade_samples: 64,
            content_hash_prefix: [2; 8],
            controls: &phase1.wire_descriptors,
            cp1_overflow: &res.cp1_overflow,
            tsyn: &res.tsyn,
            cext: res.cext.as_ref(),
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let view = DnamBinView::parse(&bin).unwrap();
        let mut bus =
            ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view, Tier::Lite, 48_000.0)
                .unwrap();
        bus.set_control_physical(2, 2.0).unwrap();
        bus.set_control_physical(3, 0.0).unwrap();
        bus.eval_static_and_snap_if_dirty();
        let expected_slot2 = -0.55_f32 + (-0.40_f32);
        assert!(
            (bus.stage_targets[2] - expected_slot2).abs() < 0.01,
            "simultaneous unobserved switches slot[2]={} vs expected={expected_slot2}",
            bus.stage_targets[2]
        );
    }

    #[test]
    fn test_cext_cols04_zero_and_so4_rotation_invariance() {
        let specs: Vec<ControlSpec> = (0..3).map(|_| make_range_spec(-1.0, 1.0)).collect();
        let mut samps = Vec::new();
        for a in [-1.0_f32, 0.0, 1.0] {
            for b in [-1.0_f32, 0.0, 1.0] {
                for c in [-1.0_f32, 0.0, 1.0] {
                    let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
                    st[0] = 0.4 * f64::from(a) + 0.15 * f64::from(a * b);
                    st[1] = -0.3 * f64::from(b);
                    st[2] = 0.25 * f64::from(c);
                    st[16] = 0.5 * f64::from(a);
                    st[18] = -0.6 * f64::from(b) + 0.2 * f64::from(a * c);
                    st[22] = 0.35 * f64::from(c);
                    st[30] = 0.45 * f64::from(a) - 0.25 * f64::from(c);
                    st[31] = 12.0; // must be ignored/zeroed
                    samps.push(Cp1ControlSample {
                        phys_controls: vec![a, b, c],
                        switch_options: vec![0; 3],
                        stage_targets_base: st,
                        stage_targets_ultra_delta: None,
                    });
                }
            }
        }
        let phase1 = audit_phase1_controls(&specs, &StageStabilityConstraint::default()).unwrap();
        let res = fit_and_pack_cp1_audited(&specs, &phase1, &samps, &[], Cp1FitConfig::default())
            .unwrap();
        let cext = res.cext.expect("cext present");
        for r in 0..16 {
            for col in 0..4 {
                assert_eq!(cext.w_ctrl_ext[r][col].get(), 0.0);
            }
        }
        assert!(cext.w_ctrl_ext[15].iter().all(|v| v.get() == 0.0));
        assert!(cext.w_ctrl_ultra_ext[15].iter().all(|v| v.get() == 0.0));

        // Rotate z_head[0..4] in Arena0 without touching `"CEXT"` -> stage_targets[0..32] invariant!
        let make_bin = |cb: &Arena0ControlBusWire, a3_cb: &Arena3ControlBusWire| {
            let mut a0 = DnamBinBuildInput::default_arena0_lite();
            a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0x000F);
            a0.control_bus_lite = *cb;
            a0.control_bus_lite.tank_dry_mix_fp16 =
                zerocopy::byteorder::little_endian::U16::new(dnam_kernels::f32_to_fp16(1.0));
            let a1 = dnam_vm::bin::Arena1FullDeltaWire::new_zeroed();
            let a2 = dnam_vm::bin::Arena2ProDeltaWire::new_zeroed();
            let mut a3 = dnam_vm::bin::Arena3UltraDeltaWire::new_zeroed();
            a3.control_bus_ultra_delta = *a3_cb;
            encode_dnam_bin(&DnamBinBuildInput {
                reference_sample_rate_hz: 48_000,
                header_flags: 0,
                ext_ring_pow2_frames: 0,
                ext_ring_lines: 0,
                ext_ring_format: 0,
                declared_max_delay_48k: 4_096,
                dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
                tail_time_ms: 40.0,
                ro_pole_48k: 0.9,
                switch_xfade_samples: 64,
                content_hash_prefix: [9; 8],
                controls: &phase1.wire_descriptors,
                cp1_overflow: &res.cp1_overflow,
                tsyn: &res.tsyn,
                cext: Some(&cext),
                arena0: &a0,
                arena1: Some(&a1),
                arena2: Some(&a2),
                arena3: Some(&a3),
            })
            .unwrap()
        };

        let bin_unrot = make_bin(&res.arena0_cb, &res.arena3_cb);
        let view_unrot = DnamBinView::parse(&bin_unrot).unwrap();
        let mut bus_unrot = ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(
            &view_unrot,
            Tier::Lite,
            48_000.0,
        )
        .unwrap();

        // Construct orthogonal SO(4) rotation Q (product of two Givens rotations).
        let (s1, c1) = 0.6_f32.sin_cos();
        let (s2, c2) = (-0.8_f32).sin_cos();
        let q = [
            [c1, -s1, 0.0, 0.0],
            [s1, c1, 0.0, 0.0],
            [0.0, 0.0, c2, -s2],
            [0.0, 0.0, s2, c2],
        ];
        let mut decoded_slots = [Cp1ContributorSlot::zero(); 8];
        let m = decode_cp1_contributors(
            &res.arena0_cb,
            Some(&res.arena3_cb),
            &res.cp1_overflow,
            specs.len(),
            &mut decoded_slots,
        )
        .unwrap();
        for sl in &mut decoded_slots[..m] {
            for lane in 0..usize::from(sl.lanes) {
                let x = [
                    sl.main_base[lane][0],
                    sl.main_base[lane][1],
                    sl.main_base[lane][2],
                    sl.main_base[lane][3],
                ];
                for r in 0..4 {
                    sl.main_base[lane][r] = (0..4).map(|k| q[r][k] * x[k]).sum();
                }
            }
        }
        let mut rot_a0 = res.arena0_cb;
        let mut rot_a3 = res.arena3_cb;
        let _ = pack_cp1_slots_into_wire(
            &decoded_slots[..m],
            specs.len(),
            true,
            &mut rot_a0,
            &mut rot_a3,
        )
        .unwrap();
        for f in 0..4 {
            let x = [
                res.arena0_cb.u_int_base[f][0].get(),
                res.arena0_cb.u_int_base[f][1].get(),
                res.arena0_cb.u_int_base[f][2].get(),
                res.arena0_cb.u_int_base[f][3].get(),
            ];
            for r in 0..4 {
                rot_a0.u_int_base[f][r] = F32::new((0..4).map(|k| q[r][k] * x[k]).sum());
            }
        }
        for row in 0..16 {
            let x = [
                res.arena0_cb.w_ctrl[row][0].get(),
                res.arena0_cb.w_ctrl[row][1].get(),
                res.arena0_cb.w_ctrl[row][2].get(),
                res.arena0_cb.w_ctrl[row][3].get(),
            ];
            for r in 0..4 {
                rot_a0.w_ctrl[row][r] = F32::new((0..4).map(|k| q[r][k] * x[k]).sum());
            }
        }
        let bin_rot = make_bin(&rot_a0, &rot_a3);
        let view_rot = DnamBinView::parse(&bin_rot).unwrap();
        let mut bus_rot =
            ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view_rot, Tier::Lite, 48_000.0)
                .unwrap();

        for (c, &v) in [0.8_f32, -0.7, 0.65].iter().enumerate() {
            bus_unrot.set_control_physical(c, v).unwrap();
            bus_rot.set_control_physical(c, v).unwrap();
        }
        bus_unrot.eval_static_and_snap_if_dirty();
        bus_rot.eval_static_and_snap_if_dirty();
        for s in 0..STAGE_TARGET_SLOTS {
            assert!(
                (bus_unrot.stage_targets[s] - bus_rot.stage_targets[s]).abs() < 1.0e-5,
                "slot {s}: {} vs {}",
                bus_unrot.stage_targets[s],
                bus_rot.stage_targets[s]
            );
        }
    }
}
