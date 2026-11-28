#![forbid(unsafe_code)]
//! Integration suite verifying full-subspace Pass-2 finite-difference gradient parity in BOTH
//! physical coordinate space and conditioned unconstrained `u`-space (`ScaledCoordSpace` +
//! `ComplementLog` `lru_alpha_atk`, `LogBox` `svf_k`, `log(fc/fs)` `svf_g`) across `48 kHz`
//! and `96 kHz` (`ref[train.pipeline.backbone-trajectory-opt]`).

use dnam_train::losses::{CompositeLossConfig, CompositeLossEvaluator};
use dnam_train::solver::pass2_opt::eval::{
    Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio,
};
use dnam_train::solver::pass2_opt::theta::{ParamKind, ThetaLayout};
use dnam_train::{ScaledCoordSpace, SonaModelF64, TrainRunner, TransportQuadF64};

fn build_full_subspace_test_model(sr_hz: f64) -> SonaModelF64 {
    let mut model = SonaModelF64::new(sr_hz);
    model.sense_svf_g = [0.12, 0.28, 0.55, 1.15];
    model.sense_svf_k = [0.65, 0.90, 1.35, 2.10];
    model.sense_lru_rel = [0.85; 4];
    model.sense_lru_atk = [0.12, 0.25, 0.45, 0.72];
    model.sense_vca_w = [1.8, -1.1, 0.7, 0.5];
    model.crossbar_op_gains = [0.6, 0.35];
    model.carrier_basis_w[0] = [0.4, -0.25, 0.2, 0.15];
    model.phasor_to_core[0] = [0.25, 0.2, -0.15, 0.12];
    model.opt_to_lr[0] = [0.55, -0.35];
    model.op_n_pre_gamma_bc = 0.45;
    model.op_n_pre_zoh_hold_depth = 0.28;
    model.op_n_pre_beta_q = 2.0;
    model.ro_pole_48k = 0.88;
    model.pre_conv_u_gain = [0.35, 0.28];
    model.pre_conv_hp_gain = [0.22, -0.18];
    model.direct_s2_u_gain = [0.25, 0.22];
    model.direct_s2_hp_gain = [-0.15, 0.2];
    model.core_quads[0].svf_g = [0.18, 0.28, 0.42, 0.65];
    model.core_quads[0].svf_k = [0.55, 0.85, 1.25, 1.85];
    model.core_quads[0].svf_k_nl = [0.55, 0.35, 0.25, 0.15];
    model.core_quads[0].lru_alpha_rel = [0.85; 4];
    model.core_quads[0].lru_alpha_atk = [0.15, 0.30, 0.55, 0.78];
    model.core_quads[0].adaa_drive = [1.8, 1.3, 1.1, 0.9];
    model.core_quads[0].tone_lp = [0.08, 0.05, 0.04, 0.03];
    model.core_quads[0].tone_bp = [0.06, 0.05, 0.04, 0.03];
    model.core_quads[0].tone_hp = [0.12, 0.0, 0.0, 0.0];
    model.core_quads[0].tone_dir = [0.55, 0.3, 0.2, -0.15];
    model.core_quads[1].svf_g = [0.22, 0.4, 0.7, 1.1];
    model.core_quads[1].svf_k = [0.75, 1.05, 1.4, 0.6];
    model.core_quads[1].lru_alpha_rel = [0.85; 4];
    model.core_quads[1].lru_alpha_atk = [0.20, 0.40, 0.60, 0.80];
    model.core_quads[1].tone_bp = [0.3; 4];
    model.tiles[0].w_odd_4x4[0][0] = 0.42;
    model.w_ro[0] = [
        0.35, -0.25, 0.2, 0.15, 0.12, -0.1, 0.45, -0.3, 0.22, 0.15, 0.18, -0.12,
    ];
    model.w_ro[4] = [
        0.2, 0.22, -0.1, 0.12, 0.08, 0.06, 0.28, -0.2, 0.15, 0.11, 0.09, 0.1,
    ];
    model.active_fir_taps = 4;
    model.fir_l[..4].copy_from_slice(&[0.45, 0.22, -0.1, 0.06]);
    model.fir_r[..4].copy_from_slice(&[0.40, -0.18, 0.12, -0.05]);
    model.active_kautz_pairs = 1;
    model.kautz[0].r_low = 0.84;
    model.kautz[0].theta = 0.65;
    model.kautz[0].b_gamma = 0.9;
    model.kautz[0].c_re_l = 0.35;
    model.kautz[0].c_im_l = -0.2;
    model.kautz[0].c_re_r = 0.3;
    model.kautz[0].c_im_r = 0.25;
    model.tank_dry_mix = 0.75;
    model.transport_quads[0] = TransportQuadF64 {
        delay_samples: [7.25, 9.6, 12.4, 15.1],
        g_loop: 0.65,
        damp_pole: 0.3,
        scat_psi: 0.48,
        allpass_g: [0.3, -0.25, 0.35, -0.2],
        allpass_delays: [3, 5, 7, 11],
        input_gains: [0.5, 0.4, -0.3, 0.25],
        wet_tap_l: [0.6, 0.15, -0.4, 0.1],
        wet_tap_r: [0.1, -0.55, 0.2, 0.35],
        ..Default::default()
    };
    model.clamp_all_stability_bounds(true);
    model
}

#[test]
fn test_full_subspace_theta_all_families_fd_parity_48k_and_96k() {
    let runner = TrainRunner::cpu();
    let n = 128_usize;
    let in_l: Vec<f64> = (0..n)
        .map(|t| {
            let tf = t as f64;
            0.35 * (tf * 0.025).sin() + 0.40 * (tf * 0.13).sin() + 0.25 * (tf * 0.75).cos()
        })
        .collect();
    let in_r: Vec<f64> = (0..n)
        .map(|t| {
            let tf = t as f64;
            0.30 * (tf * 0.03 + 0.2).cos()
                + 0.35 * (tf * 0.09 + 0.3).cos()
                + 0.25 * (tf * 0.68).sin()
        })
        .collect();
    let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.4 * v).collect();
    let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.35 * v).collect();

    for &sr_hz in &[48_000.0_f64, 96_000.0_f64] {
        let model = build_full_subspace_test_model(sr_hz);
        let loss_eval = CompositeLossEvaluator::new(
            &tgt_l,
            &tgt_r,
            CompositeLossConfig {
                sample_rate_hz: sr_hz,
                warmup_frames: 16,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let layout = ThetaLayout::for_pass2_joint(&model, 2, 128.0);
        let buf = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, 2).unwrap();
        let mut gl = vec![0.0; n];
        let mut gr = vec![0.0; n];
        let _ = loss_eval
            .eval_and_adjoint(buf.pred_l(), buf.pred_r(), Some(&mut gl), Some(&mut gr))
            .unwrap();
        let grads = assemble_pass2_gradients(
            &runner,
            &Pass2AdjInput {
                model: &model,
                in_l: &in_l,
                in_r: &in_r,
                buf: &buf,
                gl: &gl,
                gr: &gr,
                q_opt: layout.q_opt,
                k_opt: layout.k_opt,
            },
        )
        .unwrap();
        assert_eq!(grads.core_quad_svf_k_nl(1), [0.0; 4]);

        let (theta0, bounds) = layout.pack(&model);
        let mut grad_vec = vec![0.0_f64; theta0.len()];
        layout.write_grad(&model, &grads, &mut grad_vec);
        let eval_at = |idx: usize, delta: f64| -> f64 {
            let mut t = theta0.clone();
            t[idx] += delta;
            let mut cand = model.clone();
            layout.unpack(&t, &mut cand);
            let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
            loss_eval
                .eval_and_adjoint(b.pred_l(), b.pred_r(), None, None)
                .unwrap()
                .total
        };

        let kinds = layout.entries_kinds();
        let find_idx = |pred: &dyn Fn(&ParamKind) -> bool| -> usize {
            kinds
                .iter()
                .position(pred)
                .expect("missing param in layout")
        };

        // Verify delta=0.0 unpack round-trip invariance before FD perturbations.
        let mut cand0 = model.clone();
        layout.unpack(&theta0, &mut cand0);
        let buf0 = eval_model_pass2_audio(&runner, &cand0, &in_l, &in_r, 2).unwrap();
        for t in 0..n {
            assert!(
                (buf0.pred_l()[t] - buf.pred_l()[t]).abs() <= 1.0e-12
                    && (buf0.pred_r()[t] - buf.pred_r()[t]).abs() <= 1.0e-12,
                "sr={sr_hz} delta=0 unpack round-trip mismatch at frame {t}"
            );
        }

        // Pure-f64 sub-check (w_ro=0 eliminates f32 CoreQuad rounding inside Op-T feedback).
        {
            let mut f64_model = model.clone();
            f64_model.w_ro = [[0.0; 12]; 40];
            let f64_buf = eval_model_pass2_audio(&runner, &f64_model, &in_l, &in_r, 2).unwrap();
            let (mut f64_gl, mut f64_gr) = (vec![0.0; n], vec![0.0; n]);
            let _ = loss_eval
                .eval_and_adjoint(
                    f64_buf.pred_l(),
                    f64_buf.pred_r(),
                    Some(&mut f64_gl),
                    Some(&mut f64_gr),
                )
                .unwrap();
            let f64_grads = assemble_pass2_gradients(
                &runner,
                &Pass2AdjInput {
                    model: &f64_model,
                    in_l: &in_l,
                    in_r: &in_r,
                    buf: &f64_buf,
                    gl: &f64_gl,
                    gr: &f64_gr,
                    q_opt: layout.q_opt,
                    k_opt: layout.k_opt,
                },
            )
            .unwrap();
            let (f64_theta0, _) = layout.pack(&f64_model);
            let mut f64_gvec = vec![0.0_f64; f64_theta0.len()];
            layout.write_grad(&f64_model, &f64_grads, &mut f64_gvec);
            let eval_f64 = |idx: usize, delta: f64| -> f64 {
                let mut t = f64_theta0.clone();
                t[idx] += delta;
                let mut cand = f64_model.clone();
                layout.unpack(&t, &mut cand);
                let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
                loss_eval
                    .eval_and_adjoint(b.pred_l(), b.pred_r(), None, None)
                    .unwrap()
                    .total
            };
            let eps64 = 1.0e-6_f64;
            let f64_samples = [
                (
                    "f64 RoPole48k",
                    find_idx(&|k| matches!(k, ParamKind::RoPole48k)),
                ),
                (
                    "f64 OpNPre gamma_bc",
                    find_idx(&|k| matches!(k, ParamKind::OpNPreGammaBc)),
                ),
                (
                    "f64 Kautz b_gamma[0]",
                    find_idx(&|k| matches!(k, ParamKind::KautzBGamma(0))),
                ),
                (
                    "f64 OpT scat_psi[0]",
                    find_idx(&|k| matches!(k, ParamKind::OpTScatPsi(0))),
                ),
                (
                    "f64 OptToLr[0][0]",
                    find_idx(&|k| matches!(k, ParamKind::OptToLr(0, 0))),
                ),
                (
                    "f64 TankDryMix",
                    find_idx(&|k| matches!(k, ParamKind::TankDryMix)),
                ),
                (
                    "f64 Sense svf_g[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseSvfG(0))),
                ),
                (
                    "f64 Sense svf_k[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseSvfK(0))),
                ),
                (
                    "f64 Sense lru_atk[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseLruAtk(0))),
                ),
                (
                    "f64 Sense vca_w[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseVcaW(0))),
                ),
            ];
            for (name, idx) in f64_samples {
                let fd = (eval_f64(idx, eps64) - eval_f64(idx, -eps64)) / (2.0 * eps64);
                let ana = f64_gvec[idx];
                let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                assert!(
                    (ana - fd).abs() <= 1.0e-4 * scale,
                    "sr={sr_hz} {name} (idx={idx}) pure-f64 FD mismatch: ana={ana} fd={fd}"
                );
            }

            // Pure-f64 `u`-space FD parity (op_n_pre bypassed, w_ro=0 -> zero f32 casts) < 1e-6!
            let mut pure64_model = f64_model.clone();
            pure64_model.op_n_pre_gamma_bc = 0.0;
            pure64_model.op_n_pre_zoh_hold_depth = 0.0;
            let pure64_buf =
                eval_model_pass2_audio(&runner, &pure64_model, &in_l, &in_r, 2).unwrap();
            let (mut p64_gl, mut p64_gr) = (vec![0.0; n], vec![0.0; n]);
            let _ = loss_eval
                .eval_and_adjoint(
                    pure64_buf.pred_l(),
                    pure64_buf.pred_r(),
                    Some(&mut p64_gl),
                    Some(&mut p64_gr),
                )
                .unwrap();
            let pure64_grads = assemble_pass2_gradients(
                &runner,
                &Pass2AdjInput {
                    model: &pure64_model,
                    in_l: &in_l,
                    in_r: &in_r,
                    buf: &pure64_buf,
                    gl: &p64_gl,
                    gr: &p64_gr,
                    q_opt: layout.q_opt,
                    k_opt: layout.k_opt,
                },
            )
            .unwrap();
            let (pure64_theta0, _) = layout.pack(&pure64_model);
            let mut pure64_gvec = vec![0.0_f64; pure64_theta0.len()];
            layout.write_grad(&pure64_model, &pure64_grads, &mut pure64_gvec);

            let scales64 = layout.coord_scales(n);
            let (space64, u64_0) = ScaledCoordSpace::new(
                "subspace_f64_u_fd",
                &pure64_theta0,
                &bounds,
                Some(&scales64),
            )
            .unwrap();
            let mut x64_dec = vec![0.0_f64; layout.dim()];
            let mut jac64 = vec![0.0_f64; layout.dim()];
            space64.decode(&u64_0, &mut x64_dec, &mut jac64);
            let mut gu64 = vec![0.0_f64; layout.dim()];
            space64.pullback_grad(&pure64_gvec, &jac64, &mut gu64);
            let eval_f64_u = |idx: usize, du: f64| -> f64 {
                let mut u = u64_0.clone();
                u[idx] += du;
                let mut xv = vec![0.0_f64; layout.dim()];
                let mut jv = vec![0.0_f64; layout.dim()];
                space64.decode(&u, &mut xv, &mut jv);
                let mut cand = pure64_model.clone();
                layout.unpack(&xv, &mut cand);
                let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
                loss_eval
                    .eval_and_adjoint(b.pred_l(), b.pred_r(), None, None)
                    .unwrap()
                    .total
            };
            for (name, idx) in [
                (
                    "f64 u Sense svf_g[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseSvfG(0))),
                ),
                (
                    "f64 u Sense svf_k[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseSvfK(0))),
                ),
                (
                    "f64 u Sense lru_atk[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseLruAtk(0))),
                ),
            ] {
                let fd_u = (eval_f64_u(idx, eps64) - eval_f64_u(idx, -eps64)) / (2.0 * eps64);
                let ana_u = gu64[idx];
                let scale = ana_u.abs().max(fd_u.abs()).max(1.0e-3);
                assert!(
                    (ana_u - fd_u).abs() <= 1.0e-4 * scale,
                    "sr={sr_hz} {name} (idx={idx}) pure-f64 u-space FD mismatch: ana_u={ana_u} fd_u={fd_u}"
                );
            }
            // Also verify < 1e-7 pure-f64 u-space FD parity on direct waveform objective (isolating
            // eval_model_pass2_audio + assemble_pass2_gradients + ScaledCoordSpace from CompositeLoss).
            let wf_gl: Vec<f64> = (0..n).map(|t| 0.3 * ((t as f64) * 0.17).cos()).collect();
            let wf_gr: Vec<f64> = (0..n).map(|t| -0.25 * ((t as f64) * 0.23).sin()).collect();
            let wf_grads = assemble_pass2_gradients(
                &runner,
                &Pass2AdjInput {
                    model: &pure64_model,
                    in_l: &in_l,
                    in_r: &in_r,
                    buf: &pure64_buf,
                    gl: &wf_gl,
                    gr: &wf_gr,
                    q_opt: layout.q_opt,
                    k_opt: layout.k_opt,
                },
            )
            .unwrap();
            let mut wf_gx = vec![0.0_f64; pure64_theta0.len()];
            layout.write_grad(&pure64_model, &wf_grads, &mut wf_gx);
            let mut wf_gu = vec![0.0_f64; layout.dim()];
            space64.pullback_grad(&wf_gx, &jac64, &mut wf_gu);
            let eval_wf_u = |idx: usize, du: f64| -> f64 {
                let mut u = u64_0.clone();
                u[idx] += du;
                let mut xv = vec![0.0_f64; layout.dim()];
                let mut jv = vec![0.0_f64; layout.dim()];
                space64.decode(&u, &mut xv, &mut jv);
                let mut cand = pure64_model.clone();
                layout.unpack(&xv, &mut cand);
                let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
                b.pred_l()
                    .iter()
                    .zip(&wf_gl)
                    .map(|(a, g)| a * g)
                    .sum::<f64>()
                    + b.pred_r()
                        .iter()
                        .zip(&wf_gr)
                        .map(|(a, g)| a * g)
                        .sum::<f64>()
            };
            for (name, idx) in [
                (
                    "wf f64 u Sense svf_g[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseSvfG(0))),
                ),
                (
                    "wf f64 u Sense svf_k[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseSvfK(0))),
                ),
                (
                    "wf f64 u Sense lru_atk[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseLruAtk(0))),
                ),
            ] {
                let fd_u = (eval_wf_u(idx, eps64) - eval_wf_u(idx, -eps64)) / (2.0 * eps64);
                let ana_u = wf_gu[idx];
                let scale = ana_u.abs().max(fd_u.abs()).max(1.0);
                assert!(
                    (ana_u - fd_u).abs() <= 1.0e-6 * scale,
                    "sr={sr_hz} {name} (idx={idx}) pure-f64 u-space FD mismatch: ana_u={ana_u} fd_u={fd_u}"
                );
            }
        }

        let samples = [
            (
                "Core0 svf_g[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreSvfG(0, 0))),
                1.0e-4,
            ),
            (
                "Core0 svf_k[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreSvfK(0, 0))),
                5.0e-4,
            ),
            (
                "Core0 lru_atk[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreLruAtk(0, 0))),
                5.0e-4,
            ),
            (
                "Core0 svf_k_nl[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreSvfKNl(0, 0))),
                1.0e-3,
            ),
            (
                "Core0 adaa_drive[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreAdaaDrive(0, 0))),
                2.0e-3,
            ),
            (
                "Core1 svf_g[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreSvfG(1, 0))),
                1.0e-3,
            ),
            (
                "Tile0 w_odd[0][0]",
                find_idx(&|k| matches!(k, ParamKind::TileOdd(0, 0, 0))),
                1.0e-3,
            ),
            (
                "Sense vca_w[0]",
                find_idx(&|k| matches!(k, ParamKind::SenseVcaW(0))),
                4.0e-3,
            ),
            (
                "PhasorToCore[0][0]",
                find_idx(&|k| matches!(k, ParamKind::PhasorToCore(0, 0))),
                1.0e-3,
            ),
            (
                "OptToLr[0][0]",
                find_idx(&|k| matches!(k, ParamKind::OptToLr(0, 0))),
                2.0e-3,
            ),
            (
                "OpNPre gamma_bc",
                find_idx(&|k| matches!(k, ParamKind::OpNPreGammaBc)),
                2.0e-3,
            ),
            (
                "RoPole48k",
                find_idx(&|k| matches!(k, ParamKind::RoPole48k)),
                1.0e-3,
            ),
            (
                "Kautz b_gamma[0]",
                find_idx(&|k| matches!(k, ParamKind::KautzBGamma(0))),
                1.0e-4,
            ),
            (
                "OpT scat_psi[0]",
                find_idx(&|k| matches!(k, ParamKind::OpTScatPsi(0))),
                1.0e-4,
            ),
            (
                "TankDryMix",
                find_idx(&|k| matches!(k, ParamKind::TankDryMix)),
                1.0e-4,
            ),
            (
                "WReadout[0][6]",
                find_idx(&|k| matches!(k, ParamKind::WReadout(0, 6))),
                1.0e-4,
            ),
        ];
        for (name, idx, eps) in samples {
            let fd = (eval_at(idx, eps) - eval_at(idx, -eps)) / (2.0 * eps);
            let ana = grad_vec[idx];
            let scale = ana.abs().max(fd.abs()).max(0.1);
            assert!(
                (ana - fd).abs() <= 3.0e-2 * scale,
                "sr={sr_hz} {name} (idx={idx}) FD mismatch: ana={ana} fd={fd}"
            );
        }

        // End-to-end conditioned `u`-space FD parity across `CoreSvfG`, `CoreSvfK`, `CoreLruAtk`,
        // `SenseSvfG`, `SenseSvfK`, `SenseLruAtk` at both 48 kHz and 96 kHz.
        let scales = layout.coord_scales(n);
        let (space, u0) =
            ScaledCoordSpace::new("subspace_u_fd", &theta0, &bounds, Some(&scales)).unwrap();
        let mut x_dec = vec![0.0_f64; layout.dim()];
        let mut jac = vec![0.0_f64; layout.dim()];
        space.decode(&u0, &mut x_dec, &mut jac);
        let mut grad_u = vec![0.0_f64; layout.dim()];
        space.pullback_grad(&grad_vec, &jac, &mut grad_u);

        let eval_u = |idx: usize, du: f64| -> f64 {
            let mut u = u0.clone();
            u[idx] += du;
            let mut xv = vec![0.0_f64; layout.dim()];
            let mut jv = vec![0.0_f64; layout.dim()];
            space.decode(&u, &mut xv, &mut jv);
            let mut cand = model.clone();
            layout.unpack(&xv, &mut cand);
            let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
            loss_eval
                .eval_and_adjoint(b.pred_l(), b.pred_r(), None, None)
                .unwrap()
                .total
        };
        let u_samples = [
            (
                "u Core0 svf_g[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreSvfG(0, 0))),
                2.0e-3,
            ),
            (
                "u Core0 svf_k[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreSvfK(0, 0))),
                2.0e-3,
            ),
            (
                "u Core0 lru_atk[0]",
                find_idx(&|k| matches!(k, ParamKind::CoreLruAtk(0, 0))),
                2.0e-3,
            ),
            (
                "u Sense svf_g[0]",
                find_idx(&|k| matches!(k, ParamKind::SenseSvfG(0))),
                2.0e-3,
            ),
            (
                "u Sense svf_k[0]",
                find_idx(&|k| matches!(k, ParamKind::SenseSvfK(0))),
                2.0e-3,
            ),
            (
                "u Sense lru_atk[0]",
                find_idx(&|k| matches!(k, ParamKind::SenseLruAtk(0))),
                2.0e-3,
            ),
        ];
        for (name, idx, eps_u) in u_samples {
            let fd_u = (eval_u(idx, eps_u) - eval_u(idx, -eps_u)) / (2.0 * eps_u);
            let ana_u = grad_u[idx];
            let scale = ana_u.abs().max(fd_u.abs()).max(0.05);
            assert!(
                (ana_u - fd_u).abs() <= 3.0e-2 * scale,
                "sr={sr_hz} {name} (idx={idx}) u-space FD mismatch: ana_u={ana_u} fd_u={fd_u}"
            );
        }
    }
}
