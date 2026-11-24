//! Control-space safety auditing (`Phase 1..3`), 31-slot batched `ControlBus` adjoints,
//! deterministic `PANAMA` corner recommender, Lobatto `CP1` ANOVA + Rank-4/8 FM regression,
//! inline / `"CP1\0"` serialization, and Two-Pass 4-Band Wiener recovery.

pub mod adjoint;
pub mod clamp;
pub mod cp1;
pub mod panama;
pub mod stability;
pub mod wiener;

pub use adjoint::{
    HoistedStageTargets, StageTargetAndLatentGrads, StageTargetEvalContext, StageTargetEvalResult,
    StageTargetSolveConfig, StageTargetSolveInput, StageTargetSolveReport,
    eval_control_trajectory_3khz_vm, eval_stage_targets_loss_and_grad,
    hoist_stage_targets_onto_model, solve_point_stage_targets_adjoint,
};
pub use clamp::{
    ControlKindSpec, ControlSpec, DYNAMIC_ALLPASS_G_MAX, DYNAMIC_DAMP_POLE_MAX, DYNAMIC_G_LOOP_MAX,
    DYNAMIC_KAUTZ_R_MAX, DynamicCornerAuditReport, DynamicCornerViolation, ExtremeCornerDcReport,
    MAX_GAIN_6DB_LIN, MAX_GAIN_IDEMPOTENT, Phase1ClampOutput, Phase3AuditInput, Phase3AuditReport,
    StageStabilityConstraint, ZERO_DC_CORNER_FRAMES, ZERO_DC_CORNER_MAX_ABS, audit_phase1_controls,
    audit_phase3_dynamic_and_head_gain, check_corner_dynamic_stability,
    compute_dynamic_excursion_bounds_32, eval_static_control_bus_targets,
    evaluate_readout_spectral_magnitudes, verify_extreme_corners_zero_dc_256,
};
pub use cp1::{
    CP1_DELAY_VARIANCE_GATE_SAMPLES2, Cp1ControlSample, Cp1FitConfig, Cp1WireBundle,
    TSYN_SUBDIV_QUARTER, build_tsyn_entry_wire, build_tsyn_entry_wire_with_cond, fit_and_pack_cp1,
    fit_and_pack_cp1_audited, pack_cp1_slots_into_wire,
};
pub use panama::{
    PANAMA_COND_BOUND_TOL, PANAMA_DELTA_TOL, PANAMA_ENSEMBLE_SIZE, PANAMA_EXCLUSION_DINF,
    PANAMA_FULL_GRID_MAX_PVAR, PANAMA_GAMMA_LEV, PANAMA_LAMBDA_CP1, PANAMA_LAMBDA_MULTS,
    PANAMA_MAX_RECOMMENDATIONS, PANAMA_MIN_SIGMA2, PANAMA_TOP_TRIPLES_COUNT, PanamaControlDesc,
    PanamaControlKind, PanamaCornerScore, PanamaInput, PanamaOutput, PanamaSampleRow,
    compute_stems_key_sha256, recommend_panama_corners,
};
pub use stability::{
    CORNER_BARRIER_FLOOR, MAX_RO_POLE_48K_F32, MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP,
    MIN_DELAY_STANDALONE_SAMP, RAIL_BURST_FRAMES, RailBurstReport, SAT_OCCUPANCY_LIMIT,
    SonaCoreStabilityParams, StageTargetBounds, check_preact_occupancy,
    enforce_arena_block0_k_nl_and_clamp, enforce_arena0_stability_invariants,
    eval_corner_barrier_loss_and_grad, eval_stop_grad_corner_and_phys_barrier,
    run_phase2_rail_burst, run_phase2_rail_burst_reuse,
};
pub use wiener::{
    ACTIVE_BAND_MIN_ENERGY_FRAC, BAND_PHASE_DECORR_RHO_MAX, FOUR_BAND_HI_HZ, FOUR_BAND_LO_HZ,
    FOUR_BAND_RMS_MATCH_WEIGHT, FourBandTptBank, FourBandWienerResult, SweepWindowSegment,
    extract_sweep_window_midpoints, render_vm_mono_probe, render_vm_mono_probe_reuse,
    solve_control_jacobian_32slot, solve_two_pass_four_band_wiener,
};

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::Tier;
    use dnam_vm::bin::{DnamBinView, encode_dnam_bin};
    use dnam_vm::control::STAGE_TARGET_SLOTS;
    use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;

    #[test]
    fn test_control_phase1_cp1_wiener_rail_and_phase3_pipeline() {
        let specs = vec![
            ControlSpec {
                kind: ControlKindSpec::Range {
                    declared_min: 0.0,
                    declared_max: 1.0,
                    default_val: 0.5,
                    obs_min: 0.2,
                    obs_max: 0.8,
                    unique_values: 5,
                    lobatto_anchors: [0.2, 0.5, 0.8],
                },
            },
            ControlSpec {
                kind: ControlKindSpec::Switch {
                    num_options: 2,
                    default_opt: 0,
                    declared_mask: 0b11,
                    observed_mask: 0b11,
                },
            },
        ];
        let p1 = audit_phase1_controls(&specs, &StageStabilityConstraint::default()).unwrap();
        let samples = vec![
            Cp1ControlSample {
                phys_controls: vec![0.2, 0.0],
                switch_options: vec![0, 0],
                stage_targets_base: [0.0; STAGE_TARGET_SLOTS],
                stage_targets_ultra_delta: None,
            },
            Cp1ControlSample {
                phys_controls: vec![0.5, 0.0],
                switch_options: vec![0, 0],
                stage_targets_base: [0.0; STAGE_TARGET_SLOTS],
                stage_targets_ultra_delta: None,
            },
            Cp1ControlSample {
                phys_controls: vec![0.8, 1.0],
                switch_options: vec![0, 1],
                stage_targets_base: [0.1; STAGE_TARGET_SLOTS],
                stage_targets_ultra_delta: None,
            },
        ];
        let cp1 =
            fit_and_pack_cp1_audited(&specs, &p1, &samples, &[], Cp1FitConfig::default()).unwrap();

        let mut a0 = dnam_vm::DnamBinBuildInput::default_arena0_lite();
        a0.control_bus_lite = cp1.arena0_cb;
        let mut ro_p = 0.85_f32;
        enforce_arena0_stability_invariants(&mut a0, &mut ro_p);
        let bin = encode_dnam_bin(&dnam_vm::DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 40.0,
            ro_pole_48k: ro_p,
            switch_xfade_samples: 64,
            content_hash_prefix: [3; 8],
            controls: &p1.wire_descriptors,
            cp1_overflow: &cp1.cp1_overflow,
            tsyn: &cp1.tsyn,
            cext: cp1.cext.as_ref(),
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let view = DnamBinView::parse(&bin).unwrap();
        assert_eq!(view.control_descriptors.len(), 2);

        let rail = run_phase2_rail_burst(&bin, Tier::Lite, 48_000.0, 1.0).unwrap();
        assert!(!rail.flagged);

        let gains = vec![2.3];
        let mags = evaluate_readout_spectral_magnitudes(&gains, 0.0, 48_000.0, 32);
        let mut p3 = Phase3AuditInput {
            min_delay_samples: 6.0,
            max_delay_excursion: 0.5,
            max_paralru_r_excursion: 0.95,
            max_tank_g_excursion: 0.8,
            crossbar_coupled: true,
            ro_pole_48k: ro_p,
            readout_gains_l: gains,
            readout_gains_r: vec![1.0],
            freq_mag_l: mags,
            freq_mag_r: vec![1.0; 32],
            burst_peak_abs: 2.5,
        };
        let r3 = audit_phase3_dynamic_and_head_gain(&mut p3).unwrap();
        assert!(r3.peak_gain_after <= MAX_GAIN_6DB_LIN);
    }
}
