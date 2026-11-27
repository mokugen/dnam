//! Differentiable `31`-slot `stage_targets` + `stage_latent_*` hoisting, `Pass2` +
//! `CompositeLossEvaluator` reverse-mode pullback, bounded `SsBroydenState` + zero-alloc
//! persistent-`SonaVm` point solver, and streaming `3 kHz` VM trajectory evaluation
//! (`ref[train.pipeline.control-projection-fit]`, `ref[dsp.ctrl.dispatch.modulation-laws]`).

pub mod hoist;
pub mod solve;

pub use hoist::{HoistedStageTargets, StageTargetAndLatentGrads, hoist_stage_targets_onto_model};
pub use solve::{
    StageTargetEvalContext, StageTargetEvalResult, StageTargetSolveConfig, StageTargetSolveInput,
    StageTargetSolveReport, eval_control_trajectory_3khz_vm, eval_stage_targets_loss_and_grad,
    solve_point_stage_targets_adjoint,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::stability::StageTargetBounds;
    use crate::losses::{CompositeLossConfig, CompositeLossEvaluator, CompositeLossWeights};
    use crate::runner::TrainRunner;
    use crate::solver::model::{KautzPairF64, SonaModelF64};
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};
    use dnam_vm::control::STAGE_TARGET_SLOTS;
    use dnam_vm::math::DefaultMathBackend;
    use dnam_vm::{SonaVm, Tier};

    fn rich_test_model(sr: f64, with_core_mods: bool, coupled: bool) -> SonaModelF64 {
        let mut m = SonaModelF64::new(sr);
        m.oversample_upper_tiers_2x = false;
        m.sense_in_l = [0.55, -0.25, 0.30, 0.15];
        m.sense_in_r = [-0.20, 0.50, 0.25, -0.35];
        m.sense_vca_w = [0.65, -0.45, 0.35, 0.25];
        m.sense_vca_bias = 0.08;
        m.phasor_freq_hz = [-180.0, 320.0, -470.0, 760.0];
        m.phasor_init_phase_q32 = [0x1000_0000, 0x3500_0000, 0x7000_0000, 0xB000_0000];
        m.crossbar_op_gains = [0.55, if coupled { 0.35 } else { 0.0 }];
        m.carrier_basis_w = [
            [0.35, 0.22, -0.18, 0.20],
            [-0.20, 0.30, 0.25, -0.15],
            [0.18, -0.14, 0.28, 0.22],
        ];
        for q in 0..10 {
            m.op_p_to_core[q] = [0.15, -0.12, 0.09, -0.07];
            m.phasor_to_core[q] = [0.12, 0.10, -0.08, 0.06];
            m.core_quads[q].svf_g = [0.16, 0.28, 0.44, 0.72];
            m.core_quads[q].svf_k = [0.85, 1.05, 0.70, 1.25];
            m.core_quads[q].svf_k_nl = if q == 0 || q == 4 || q == 6 || q == 8 {
                [0.2, 0.15, 0.25, 0.1]
            } else {
                [0.0; 4]
            };
            m.core_quads[q].tone_lp = [0.4, 0.3, 0.25, 0.2];
            m.core_quads[q].tone_bp = [0.25, -0.2, 0.3, -0.15];
            m.core_quads[q].tone_hp = [0.15, 0.2, -0.15, 0.25];
            m.core_quads[q].tone_dir = [0.25; 4];
            m.core_quads[q].in_mid_weight = [0.45, -0.35, 0.30, 0.25];
            m.core_quads[q].in_side_weight = [0.2, 0.25, -0.2, 0.15];
            m.core_quads[q].adaa_drive = [0.9, -0.8, 0.7, 0.6];
            m.core_quads[q].lru_alpha_rel = [0.92, 0.88, 0.84, 0.78];
            m.core_quads[q].lru_alpha_atk = [0.22, 0.30, 0.38, 0.45];
            m.core_quads[q].lru_kappa = [0.15, 0.22, 0.18, 0.25];
            m.core_quads[q].lru_bias_drive = [0.25, -0.18, 0.12, -0.15];
            m.core_quads[q].wavefolder_mix = [0.35; 4];
            m.core_quads[q].rect_sign = [1.0, -1.0, 1.0, -1.0];
            for k in 0..4 {
                let sgn = if (((q >> 1) + k) & 1) == 0 { 1.0 } else { -1.0 };
                m.w_ro[q * 4 + k][0] = sgn * 0.032;
                m.w_ro[q * 4 + k][1] = -sgn * 0.028;
                m.w_ro[q * 4 + k][4] = sgn * 0.024;
                m.w_ro[q * 4 + k][5] = -sgn * 0.020;
                m.w_ro[q * 4 + k][6] = sgn * 0.036;
                m.w_ro[q * 4 + k][7] = -sgn * 0.032;
                m.w_ro[q * 4 + k][10] = sgn * 0.028;
                m.w_ro[q * 4 + k][11] = -sgn * 0.024;
            }
        }
        m.op_n_pre_gamma_bc = 0.42;
        m.op_n_pre_zoh_hold_depth = 0.28;
        m.op_n_pre_beta_q = 2.2;
        m.ro_pole_48k = 0.82;
        m.pre_conv_u_gain = [0.35, 0.30];
        m.pre_conv_hp_gain = [0.18, -0.15];
        m.direct_s2_u_gain = [0.40, 0.36];
        m.direct_s2_hp_gain = [-0.12, 0.14];
        m.fir_l[0] = 0.22;
        m.fir_r[1] = 0.18;
        m.active_fir_taps = 4;
        for k in 0..4 {
            m.kautz[k] = KautzPairF64 {
                r_low: 0.78 + 0.03 * (k as f64),
                dr: 0.0,
                theta: 0.28 + 0.25 * (k as f64),
                b_gamma: 0.35,
                c_re_l: 0.24,
                c_im_l: 0.09,
                c_re_r: 0.20,
                c_im_r: -0.11,
            };
        }
        m.active_kautz_pairs = 4;
        m.tank_dry_mix = 0.65;
        m.op_t_wet_scale = 1.0;
        for q in 0..4 {
            let tq = &mut m.transport_quads[q];
            tq.delay_samples = [6.4, 8.8, 11.3, 14.7];
            tq.ramp_spans_samples = [0.36, 0.34, 0.32, 0.38];
            tq.g_loop = 0.52;
            tq.r_dc = 0.992;
            tq.damp_pole = 0.35;
            tq.lfo_rate_hz = 13.0;
            tq.ramp_rate_hz = 12.0;
            tq.lfo_depths = [2.8, 2.2, 3.1, 1.9];
            tq.lfo_phase_off_q16 = [0, 1_024, 2_048, 3_072];
            tq.ramp_phase_off_q16 = [4_096, 8_192, 12_288, 16_384];
            tq.input_gains = [0.38, -0.26, 0.31, -0.22];
            tq.wet_tap_l = [0.32, 0.18, -0.24, 0.14];
            tq.wet_tap_r = [-0.16, 0.34, 0.12, -0.28];
            tq.allpass_delays = [3, 5, 7, 11];
            tq.allpass_g = [0.22, -0.18, 0.26, -0.14];
            tq.fb_psi = 0.32;
            tq.sigma_knee = 0.85;
            tq.freeze_mix = 0.12;
            tq.window_depth = 0.50;
            tq.scat_psi = 0.22;
        }
        if coupled {
            m.opt_to_lr[0] = [0.55, -0.35];
            m.opt_to_lr[1] = [-0.30, 0.50];
        }
        if with_core_mods {
            let mut cm = Box::new([[0.0_f32; 120]; 8]);
            cm[0][0] = 0.12;
            cm[0][40] = -0.08;
            cm[0][80] = 0.10;
            m.stage_latent_mods.core_mods_ultra = Some(cm);
        }
        m
    }

    #[test]
    fn test_control_adjoint_central_fd_all_31_slots_and_latent_tables() {
        let runner = TrainRunner::cpu();
        let n = 192_usize;
        let sr = 48_000.0_f64;
        let centers = crate::losses::canonical_ltas_33_centers_hz();
        let synth_in = |phase: f64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let t = (i as f64) / sr;
                    let mut s = 0.02 * (core::f64::consts::TAU * 22_200.0 * t + phase).sin();
                    for (b, &fc) in centers.iter().enumerate() {
                        s += 0.022
                            * (core::f64::consts::TAU * fc * t + phase + 0.37 * (b as f64)).sin();
                    }
                    s
                })
                .collect()
        };
        let in_l = synth_in(0.0);
        let in_r = synth_in(0.7);
        let target_l: Vec<f64> = in_l
            .iter()
            .enumerate()
            .map(|(i, &v)| 0.72 * v + 0.03 * ((i as f64) * 0.19).cos())
            .collect();
        let target_r: Vec<f64> = in_r
            .iter()
            .enumerate()
            .map(|(i, &v)| 0.68 * v - 0.03 * ((i as f64) * 0.23).sin())
            .collect();
        let loss_eval = CompositeLossEvaluator::new(
            &target_l,
            &target_r,
            CompositeLossConfig {
                sample_rate_hz: sr,
                warmup_frames: 24,
                w_besr: 1.0,
                w_mr_iir: 0.35,
                w_spec_ot: 0.0,
                w_gabor: 0.15,
                w_if_gd: 0.0,
                w_ltas: 0.10,
                w_seg_esr: 0.15,
                w_nyquist: 0.10,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let weights = CompositeLossWeights::from_config(&CompositeLossConfig {
            w_besr: 1.0,
            w_mr_iir: 0.35,
            w_spec_ot: 0.0,
            w_gabor: 0.15,
            w_if_gd: 0.0,
            w_ltas: 0.10,
            w_seg_esr: 0.15,
            w_nyquist: 0.10,
            ..CompositeLossConfig::default()
        });

        // Isolated non-cancelling Block 0 PHS & Block 1 Linear `slot = 1` (`svf_k`) accuracy check.
        for (label, target_q, k_nl) in [
            ("block0_phs", 0_usize, [0.2, 0.15, 0.25, 0.1]),
            ("block1_lin", 1_usize, [0.0; 4]),
        ] {
            let mut iso = rich_test_model(sr, false, false);
            iso.w_ro = [[0.0; 12]; 40];
            for k in 0..4 {
                iso.w_ro[target_q * 4 + k][0] = 0.25;
                iso.w_ro[target_q * 4 + k][6] = 0.25;
            }
            iso.core_quads[target_q].adaa_drive = [0.4; 4];
            iso.core_quads[target_q].tone_bp = [0.35; 4];
            iso.core_quads[target_q].tone_hp = [0.25; 4];
            iso.core_quads[target_q].svf_k_nl = k_nl;
            let mut st_iso = [0.0_f64; STAGE_TARGET_SLOTS];
            st_iso[1] = -0.05;
            let res_iso = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &iso,
                    tier: Tier::Lite,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &st_iso,
                &[0.0; 8],
                true,
            )
            .unwrap();
            let ana_k = res_iso.grads.as_ref().unwrap().d_stage_targets[1];
            let eps_k = 5.0e-4_f64;
            let (mut stp, mut stm) = (st_iso, st_iso);
            stp[1] += eps_k;
            stm[1] -= eps_k;
            let lp = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &iso,
                    tier: Tier::Lite,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &stp,
                &[0.0; 8],
                false,
            )
            .unwrap()
            .breakdown
            .total;
            let lm = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &iso,
                    tier: Tier::Lite,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &stm,
                &[0.0; 8],
                false,
            )
            .unwrap()
            .breakdown
            .total;
            let fd_k = (lp - lm) / (2.0 * eps_k);
            let scale_k = ana_k.abs().max(fd_k.abs()).max(1.0e-3);
            assert!(
                (ana_k - fd_k).abs() < 5.0e-3 * scale_k,
                "isolated {label} svf_k mismatch: ana={ana_k} fd={fd_k}"
            );
        }

        for &coupled in &[false, true] {
            let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
            st[0] = 0.08;
            st[1] = -0.05;
            st[2] = 0.12;
            st[7] = -0.08;
            st[10] = 0.06;
            st[11] = -0.10;
            st[12] = 1.5;
            st[13] = 0.05;
            st[14] = 0.12;
            st[15] = 0.10;
            st[16] = 0.04;
            st[17] = -0.07;
            st[18] = 0.35;
            st[19] = 0.40;
            st[20] = -0.10;
            st[21] = 0.08;
            st[22] = -0.05;
            st[23] = 0.12;
            st[24] = -0.06;
            st[25] = 0.08;
            st[26] = 0.06;
            st[27] = 0.05;
            st[28] = -0.12;
            st[29] = 0.15;
            st[30] = -0.07;
            let z_stage = [0.42, -0.31, 0.25, -0.18, 0.22, -0.14, 0.19, -0.11];

            let eval_slot_fd = |m: &SonaModelF64, slot: usize, eps: f64| -> f64 {
                let (mut stp, mut stm) = (st, st);
                stp[slot] += eps;
                stm[slot] -= eps;
                let lp = eval_stage_targets_loss_and_grad(
                    &StageTargetEvalContext {
                        runner: &runner,
                        base: m,
                        tier: Tier::Ultra,
                        in_l: &in_l,
                        in_r: &in_r,
                        loss_eval: &loss_eval,
                        weights,
                    },
                    &stp,
                    &z_stage,
                    false,
                )
                .unwrap()
                .breakdown
                .total;
                let lm = eval_stage_targets_loss_and_grad(
                    &StageTargetEvalContext {
                        runner: &runner,
                        base: m,
                        tier: Tier::Ultra,
                        in_l: &in_l,
                        in_r: &in_r,
                        loss_eval: &loss_eval,
                        weights,
                    },
                    &stm,
                    &z_stage,
                    false,
                )
                .unwrap()
                .breakdown
                .total;
                (lp - lm) / (2.0 * eps)
            };

            // 1. Pure-f64 exact verification (`w_ro = 0`, zero f32 CoreQuad wave rounding).
            let mut pure_f64 = rich_test_model(sr, false, coupled);
            pure_f64.w_ro = [[0.0; 12]; 40];
            let res_f64 = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &pure_f64,
                    tier: Tier::Ultra,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &st,
                &z_stage,
                true,
            )
            .unwrap();
            let g_f64 = res_f64.grads.as_ref().unwrap();
            assert_eq!(g_f64.d_stage_targets[31], 0.0);
            assert!(
                g_f64.d_stage_targets[24].abs() > 1.0e-5,
                "expected active non-zero window_depth gradient on pure_f64 coupled={coupled}"
            );
            let mut bad_f64 = Vec::new();
            for slot in (10..=28).chain(core::iter::once(30)) {
                let fd = eval_slot_fd(&pure_f64, slot, 1.0e-5_f64);
                let ana = g_f64.d_stage_targets[slot];
                let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                if (ana - fd).abs() >= 1.0e-4 * scale {
                    bad_f64.push(format!("slot={slot}: ana={ana} fd={fd}"));
                }
            }
            assert!(
                bad_f64.is_empty(),
                "pure-f64 coupled={coupled} slot mismatches: {bad_f64:?}"
            );

            // 2. Full mixed f32/f64 verification across all 31 active slots and latent tables.
            let base = rich_test_model(sr, true, coupled);
            let res = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &base,
                    tier: Tier::Ultra,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &st,
                &z_stage,
                true,
            )
            .unwrap();
            let g = res.grads.as_ref().unwrap();
            assert_eq!(g.d_stage_targets[31], 0.0);

            let mut bad_slots = Vec::new();
            for slot in 0..30 {
                let (eps, tol) = match slot {
                    27 => (1.0e-5_f64, 2.0e-2_f64),
                    28 => (5.0e-5_f64, 2.0e-2_f64),
                    0..=9 | 15 | 26 | 29 => (5.0e-4_f64, 2.0e-2_f64),
                    18 | 19 if coupled => (1.0e-2_f64, 2.0e-2_f64),
                    10..=21 | 23..=25 if coupled => (5.0e-4_f64, 2.0e-2_f64),
                    _ => (1.0e-5_f64, 1.0e-4_f64),
                };
                let fd = eval_slot_fd(&base, slot, eps);
                let ana = g.d_stage_targets[slot];
                let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                if (ana - fd).abs() >= tol * scale {
                    bad_slots.push(format!("slot={slot}: ana={ana} fd={fd}"));
                }
            }
            // Slot 30 (`sense_vca_bias`) in full mixed f32/f64 with Op-P zero-crossing gate disabled.
            let mut base_s30 = base.clone();
            base_s30.crossbar_op_gains[0] = 0.0;
            let res_s30 = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &base_s30,
                    tier: Tier::Ultra,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &st,
                &z_stage,
                true,
            )
            .unwrap();
            let ana_s30 = res_s30.grads.as_ref().unwrap().d_stage_targets[30];
            let fd_s30 = eval_slot_fd(&base_s30, 30, 5.0e-4_f64);
            let scale_s30 = ana_s30.abs().max(fd_s30.abs()).max(1.0e-3);
            if (ana_s30 - fd_s30).abs() >= 2.0e-2 * scale_s30 {
                bad_slots.push(format!("slot=30: ana={ana_s30} fd={fd_s30}"));
            }
            assert!(
                bad_slots.is_empty(),
                "coupled={coupled} slot mismatches: {bad_slots:?}"
            );

            // Verify latent tables (`global_stage_mods`, `group_*_ultra`, and `core_mods_ultra`).
            let check_latent =
                |label: &str,
                 ana: f32,
                 eps_l: f32,
                 tol: f64,
                 mutate: &dyn Fn(&mut SonaModelF64, f32)| {
                    let (mut bp, mut bm) = (base.clone(), base.clone());
                    mutate(&mut bp, eps_l);
                    mutate(&mut bm, -eps_l);
                    let lp = eval_stage_targets_loss_and_grad(
                        &StageTargetEvalContext {
                            runner: &runner,
                            base: &bp,
                            tier: Tier::Ultra,
                            in_l: &in_l,
                            in_r: &in_r,
                            loss_eval: &loss_eval,
                            weights,
                        },
                        &st,
                        &z_stage,
                        false,
                    )
                    .unwrap()
                    .breakdown
                    .total;
                    let lm = eval_stage_targets_loss_and_grad(
                        &StageTargetEvalContext {
                            runner: &runner,
                            base: &bm,
                            tier: Tier::Ultra,
                            in_l: &in_l,
                            in_r: &in_r,
                            loss_eval: &loss_eval,
                            weights,
                        },
                        &st,
                        &z_stage,
                        false,
                    )
                    .unwrap()
                    .breakdown
                    .total;
                    let fd = (lp - lm) / (2.0 * f64::from(eps_l));
                    let a64 = f64::from(ana);
                    let scale = a64.abs().max(fd.abs()).max(1.0e-3);
                    assert!(
                        (a64 - fd).abs() < tol * scale,
                        "coupled={coupled} {label}: ana={a64} fd={fd}"
                    );
                };
            check_latent(
                "global[0][0]",
                g.d_latent_mods.global_stage_mods[0][0],
                1.0e-3,
                2.0e-2,
                &|m, d| {
                    m.stage_latent_mods.global_stage_mods[0][0] += d;
                },
            );
            let (latent_eps, latent_tol) = if coupled {
                (1.0e-3, 2.0e-2)
            } else {
                (1.0e-4, 1.0e-4)
            };
            check_latent(
                "global[3][4]",
                g.d_latent_mods.global_stage_mods[3][4],
                latent_eps,
                latent_tol,
                &|m, d| {
                    m.stage_latent_mods.global_stage_mods[3][4] += d;
                },
            );
            check_latent(
                "global[4][2]",
                g.d_latent_mods.global_stage_mods[4][2],
                latent_eps,
                latent_tol,
                &|m, d| {
                    m.stage_latent_mods.global_stage_mods[4][2] += d;
                },
            );
            check_latent(
                "group_adaa[1][0]",
                g.d_latent_mods.group_adaa_scale_ultra[1][0],
                1.0e-3,
                2.0e-2,
                &|m, d| {
                    m.stage_latent_mods.group_adaa_scale_ultra[1][0] += d;
                },
            );
            check_latent(
                "group_lru[2][5]",
                g.d_latent_mods.group_lru_bias_ultra[2][5],
                1.0e-3,
                2.0e-2,
                &|m, d| {
                    m.stage_latent_mods.group_lru_bias_ultra[2][5] += d;
                },
            );
            let cm_g = g.d_latent_mods.core_mods_ultra.as_ref().unwrap();
            check_latent("core_mods[0][0]", cm_g[0][0], 1.0e-3, 2.0e-2, &|m, d| {
                m.stage_latent_mods.core_mods_ultra.as_mut().unwrap()[0][0] += d;
            });
            check_latent("core_mods[0][40]", cm_g[0][40], 1.0e-3, 2.0e-2, &|m, d| {
                m.stage_latent_mods.core_mods_ultra.as_mut().unwrap()[0][40] += d;
            });
            check_latent("core_mods[0][80]", cm_g[0][80], 1.0e-3, 2.0e-2, &|m, d| {
                m.stage_latent_mods.core_mods_ultra.as_mut().unwrap()[0][80] += d;
            });
        }

        // Verify Lite block tables (`block_adaa_scale_lite` & `block_lru_bias_lite`).
        let base_lite = rich_test_model(sr, false, false);
        let z_lite = [0.35, -0.25, 0.18, -0.12, 0.0, 0.0, 0.0, 0.0];
        let st_lite = [0.05_f64; STAGE_TARGET_SLOTS];
        let g_lite = eval_stage_targets_loss_and_grad(
            &StageTargetEvalContext {
                runner: &runner,
                base: &base_lite,
                tier: Tier::Lite,
                in_l: &in_l,
                in_r: &in_r,
                loss_eval: &loss_eval,
                weights,
            },
            &st_lite,
            &z_lite,
            true,
        )
        .unwrap()
        .grads
        .unwrap();
        let eps_l = 1.0e-3_f32;
        for (ana, mutate) in [
            (
                g_lite.d_latent_mods.block_adaa_scale_lite[1][0],
                (&|m: &mut SonaModelF64, d: f32| {
                    m.stage_latent_mods.block_adaa_scale_lite[1][0] += d;
                }) as &dyn Fn(&mut SonaModelF64, f32),
            ),
            (
                g_lite.d_latent_mods.block_lru_bias_lite[2][3],
                (&|m: &mut SonaModelF64, d: f32| {
                    m.stage_latent_mods.block_lru_bias_lite[2][3] += d;
                }) as &dyn Fn(&mut SonaModelF64, f32),
            ),
        ] {
            let (mut bp, mut bm) = (base_lite.clone(), base_lite.clone());
            mutate(&mut bp, eps_l);
            mutate(&mut bm, -eps_l);
            let lp = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &bp,
                    tier: Tier::Lite,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &st_lite,
                &z_lite,
                false,
            )
            .unwrap()
            .breakdown
            .total;
            let lm = eval_stage_targets_loss_and_grad(
                &StageTargetEvalContext {
                    runner: &runner,
                    base: &bm,
                    tier: Tier::Lite,
                    in_l: &in_l,
                    in_r: &in_r,
                    loss_eval: &loss_eval,
                    weights,
                },
                &st_lite,
                &z_lite,
                false,
            )
            .unwrap()
            .breakdown
            .total;
            let fd = (lp - lm) / (2.0 * f64::from(eps_l));
            let scale = f64::from(ana).abs().max(fd.abs()).max(1.0e-3);
            assert!(
                (f64::from(ana) - fd).abs() < 2.0e-2 * scale,
                "lite block latent: ana={ana} fd={fd}"
            );
        }
    }

    #[test]
    fn test_solve_point_stage_targets_adjoint_recovery_and_trajectory_vm() {
        let runner = TrainRunner::cpu();
        let n = 256_usize;
        let sr = 48_000.0_f64;
        let base = rich_test_model(sr, false, false);
        let in_l: Vec<f64> = (0..n)
            .map(|t| 0.32 * ((t as f64) * 0.065).sin() + 0.14 * ((t as f64) * 0.24).cos())
            .collect();
        let in_r: Vec<f64> = (0..n)
            .map(|t| 0.28 * ((t as f64) * 0.095).cos() - 0.12 * ((t as f64) * 0.21).sin())
            .collect();

        let mut true_st = [0.0_f64; STAGE_TARGET_SLOTS];
        true_st[0] = 0.45;
        true_st[2] = -0.35;
        true_st[13] = 0.22;
        true_st[15] = 0.40;
        true_st[22] = 0.30;
        true_st[30] = -0.25;
        let hoisted_true = hoist_stage_targets_onto_model(&base, &true_st, &[0.0; 8], Tier::Lite);
        let target_buf =
            eval_model_pass2_audio_for_test(&runner, &hoisted_true.model, &in_l, &in_r, 4);

        let bounds = StageTargetBounds::canonical(true, 16.0);
        let mut cfg = StageTargetSolveConfig {
            active_slots: vec![0, 2, 13, 15, 22, 30],
            tier: Tier::Lite,
            warmup_frames: 24,
            l2_reg: 1.0e-8,
            barrier_weight: 1.0e-6,
            vm_micro_polish_iters: 1,
            vm_micro_polish_eps: 5.0e-3,
            ..StageTargetSolveConfig::default()
        };
        cfg.opt.max_iters = 24;

        let dnam_bin = encode_model_to_dnam_bin(&base, Pass5EmitConfig::default()).unwrap();
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&dnam_bin, Tier::Lite, sr as f32)
                .unwrap();

        let rep = solve_point_stage_targets_adjoint(
            &StageTargetSolveInput {
                runner: &runner,
                base: &base,
                in_l: &in_l,
                in_r: &in_r,
                target_l: &target_buf.0,
                target_r: &target_buf.1,
                init_targets: &[0.0; STAGE_TARGET_SLOTS],
                z_stage: &[0.0; 8],
                bounds: &bounds,
            },
            &cfg,
            Some(&mut vm),
        )
        .unwrap();
        assert_eq!(rep.stage_targets[31], 0.0);
        assert!(
            rep.final_loss < 0.05 * rep.initial_loss,
            "final={} init={}",
            rep.final_loss,
            rep.initial_loss
        );

        let in_l32: Vec<f32> = in_l.iter().map(|&v| v as f32).collect();
        let in_r32: Vec<f32> = in_r.iter().map(|&v| v as f32).collect();
        let n_ticks = n.div_ceil(16);
        let traj: Vec<[f32; STAGE_TARGET_SLOTS]> = (0..n_ticks)
            .map(|k| {
                let mut s = [0.0_f32; STAGE_TARGET_SLOTS];
                let f = (k as f32) / (n_ticks.max(1) as f32);
                s[0] = 0.25 * f;
                s[22] = -0.15 * f;
                s[31] = 7.0;
                s
            })
            .collect();
        let zs = vec![[0.0_f32; 8]; n_ticks];
        let (tl1, tr1) =
            eval_control_trajectory_3khz_vm(&mut vm, &in_l32, &in_r32, &traj, &zs).unwrap();
        let (tl2, tr2) =
            eval_control_trajectory_3khz_vm(&mut vm, &in_l32, &in_r32, &traj, &zs).unwrap();
        assert_eq!(tl1.len(), n);
        assert_eq!(tl1, tl2);
        assert_eq!(tr1, tr2);
    }

    fn eval_model_pass2_audio_for_test(
        runner: &TrainRunner,
        model: &SonaModelF64,
        in_l: &[f64],
        in_r: &[f64],
        q_count: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let buf = crate::solver::pass2_opt::eval::eval_model_pass2_audio(
            runner, model, in_l, in_r, q_count,
        )
        .unwrap();
        (buf.pred_l, buf.pred_r)
    }
}
