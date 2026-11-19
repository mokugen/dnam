#![forbid(unsafe_code)]
//! Dedicated `1x` linear modal resonator basis configured into `Stage II Group 0` (`3` blocks x `4`
//! lanes = `6` conjugate pole pairs) so `Lite` and `Full` exceed `33` Welch-33 degrees of freedom
//! without modifying runtime engine topology or `.dnam` container layout.

use std::f64::consts::{PI, TAU};

use dnam_engine::stages::stage1_svf::V3Stage1SvfParams;
use dnam_engine::stages::stage2_gru::{NUM_GRU_BLOCKS, V3GruBlockCold, V3InputFeaturesParams};
use dnam_engine::stages::stage3_cab::V3Rate1xCalibration;
use dnam_engine::stages::stage4_tank::v3_fdn::V3Stage4TankHot;
use dnam_engine::stages::{V3HotCore, V3MatryoshkaModel, V3TierReadout};
use dnam_engine::{FtzDazGuard, Tier};

use crate::solver::backends::TrainerBackend;
use crate::solver::v3_varpro::primitives::par_map_indexed;

use super::super::super::model_f64::{V3TrainDataset, V3TrainStem};

pub(super) const S2_MODAL_COLS: usize = 12;
const S2_MODAL_MAX_L2_SQ: f64 = 248.0;

const S2_MODAL_MODES: [(f64, f64); 6] = [
    (380.0, 0.32),
    (620.0, 0.32),
    (1_020.0, 0.35),
    (1_620.0, 0.35),
    (2_550.0, 0.38),
    (4_000.0, 0.40),
];

pub(super) struct DecoupledS2Cols {
    pub(super) l: [Vec<f32>; S2_MODAL_COLS],
    pub(super) r: [Vec<f32>; S2_MODAL_COLS],
}

pub(super) struct CoupledS2Cols {
    pub(super) ll: [Vec<f32>; S2_MODAL_COLS],
    pub(super) rl: [Vec<f32>; S2_MODAL_COLS],
    pub(super) lr: [Vec<f32>; S2_MODAL_COLS],
    pub(super) rr: [Vec<f32>; S2_MODAL_COLS],
}

pub(in crate::solver::v3_varpro::stage4_id) fn build_1x_linear_resonator_cal(
    sr_hz: f64,
) -> V3Rate1xCalibration {
    let sr = sr_hz.max(44_100.0);
    let mut blocks = [V3GruBlockCold::default(); NUM_GRU_BLOCKS];
    for (b_idx, blk) in blocks.iter_mut().enumerate() {
        blk.b_z[..4].fill(8.0);
        blk.b_r[..4].fill(8.0);
        blk.b_n[..4].fill(0.0);
        for pair in 0..2 {
            let (f_hz, bw) = S2_MODAL_MODES[b_idx * 2 + pair];
            let l0 = pair * 2;
            let l1 = l0 + 1;
            let rho = (-PI * bw * f_hz / sr).exp().clamp(0.45, 0.992);
            let th = (TAU * f_hz / sr).clamp(1.0e-3, PI - 1.0e-3);
            let (s, c) = th.sin_cos();
            let rho_f = rho as f32;
            blk.w_rec_n[l0][l0] = rho_f * (c as f32);
            blk.w_rec_n[l0][l1] = -rho_f * (s as f32);
            blk.w_rec_n[l1][l0] = rho_f * (s as f32);
            blk.w_rec_n[l1][l1] = rho_f * (c as f32);
            blk.w_in_n[0][l0] = (0.30 * (1.0 - rho * rho).max(1.0e-6).sqrt()) as f32;
        }
        blk.clamp_in_place();
    }
    V3Rate1xCalibration {
        svf: V3Stage1SvfParams::identity_bypass(),
        feat_params: V3InputFeaturesParams::default(),
        blocks,
    }
}

pub(super) fn has_1x_modal_basis(model: &V3MatryoshkaModel, tier: Tier) -> bool {
    !model.tier_os2x[tier as usize]
        && model
            .rate_1x_cal
            .as_ref()
            .is_some_and(|c| (c.blocks[0].b_z[0] - 8.0).abs() < 1.0e-3)
}

pub(super) fn max_direct_s2_l2_sq(ro: &V3TierReadout, ch: usize) -> f64 {
    let pre_cab_sq: f64 = ro.pre_cab_w[ch]
        .iter()
        .map(|&w| f64::from(w) * f64::from(w))
        .sum();
    (S2_MODAL_MAX_L2_SQ - pre_cab_sq).max(0.0)
}

fn extract_raw_group0_waves<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
) -> ([Vec<f32>; S2_MODAL_COLS], [Vec<f32>; S2_MODAL_COLS]) {
    let n = stem.usable_len();
    let mut probe = model.extract_tier(tier);
    probe.readout = V3TierReadout::zeros();
    let mut core = V3HotCore::<B>::from_cold_model(&probe);
    core.reset_to_dc_equilibrium();

    let mut hl = std::array::from_fn(|_| vec![0.0_f32; n]);
    let mut hr = std::array::from_fn(|_| vec![0.0_f32; n]);
    let _ftz = FtzDazGuard::new();
    for i in 0..n {
        let _ = core.step_stereo(stem.in_l[i], stem.in_r[i]);
        for b in 0..3 {
            for l in 0..4 {
                let m = b * 4 + l;
                hl[m][i] = core.state_l.h[b][0][l];
                hr[m][i] = core.state_r.h[b][0][l];
            }
        }
    }
    (hl, hr)
}

fn sim_tank_column<B: TrainerBackend>(
    stage4: &dnam_engine::stages::stage4_tank::v3_fdn::V3Stage4TankParams,
    in_l: &[f32],
    in_r: &[f32],
    mono: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n = in_l.len().min(in_r.len());
    let mut cl = vec![0.0_f32; n];
    let mut cr = vec![0.0_f32; n];
    let _ftz = FtzDazGuard::new();
    let mut tank = V3Stage4TankHot::from_cold(stage4);
    for i in 0..n {
        let [a, b] = tank.step_stereo::<B>(in_l[i], if mono { in_l[i] } else { in_r[i] });
        cl[i] = a;
        cr[i] = b;
    }
    (cl, cr)
}

pub(super) fn extract_decoupled_s2_cols<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
) -> DecoupledS2Cols {
    let (hl, hr) = extract_raw_group0_waves::<B>(model, tier, stem);
    let n = stem.usable_len();
    let stage4 = model.stage4[tier as usize];
    let mono = model.mono_out;
    let mut pairs: Vec<(Vec<f32>, Vec<f32>)> = if n >= 8_192 {
        par_map_indexed(S2_MODAL_COLS, |m| {
            sim_tank_column::<B>(&stage4, &hl[m], &hr[m], mono)
        })
    } else {
        (0..S2_MODAL_COLS)
            .map(|m| sim_tank_column::<B>(&stage4, &hl[m], &hr[m], mono))
            .collect()
    };
    let mut it = pairs.drain(..);
    let mut col_l: [Vec<f32>; S2_MODAL_COLS] = std::array::from_fn(|_| Vec::new());
    let mut col_r: [Vec<f32>; S2_MODAL_COLS] = std::array::from_fn(|_| Vec::new());
    for (m, (a, b)) in it.by_ref().enumerate() {
        col_l[m] = a;
        col_r[m] = b;
    }
    DecoupledS2Cols { l: col_l, r: col_r }
}

pub(super) fn extract_coupled_s2_cols<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
) -> CoupledS2Cols {
    let (hl, hr) = extract_raw_group0_waves::<B>(model, tier, stem);
    let n = stem.usable_len();
    let stage4 = model.stage4[tier as usize];
    let zeros = vec![0.0_f32; n];
    let mut jobs: Vec<(Vec<f32>, Vec<f32>)> = if n >= 8_192 {
        par_map_indexed(2 * S2_MODAL_COLS, |job| {
            let m = job % S2_MODAL_COLS;
            if job < S2_MODAL_COLS {
                sim_tank_column::<B>(&stage4, &hl[m], &zeros, false)
            } else {
                sim_tank_column::<B>(&stage4, &zeros, &hr[m], false)
            }
        })
    } else {
        (0..2 * S2_MODAL_COLS)
            .map(|job| {
                let m = job % S2_MODAL_COLS;
                if job < S2_MODAL_COLS {
                    sim_tank_column::<B>(&stage4, &hl[m], &zeros, false)
                } else {
                    sim_tank_column::<B>(&stage4, &zeros, &hr[m], false)
                }
            })
            .collect()
    };
    let mut ll: [Vec<f32>; S2_MODAL_COLS] = std::array::from_fn(|_| Vec::new());
    let mut rl: [Vec<f32>; S2_MODAL_COLS] = std::array::from_fn(|_| Vec::new());
    let mut lr: [Vec<f32>; S2_MODAL_COLS] = std::array::from_fn(|_| Vec::new());
    let mut rr: [Vec<f32>; S2_MODAL_COLS] = std::array::from_fn(|_| Vec::new());
    for (idx, (a, b)) in jobs.drain(..).enumerate() {
        let m = idx % S2_MODAL_COLS;
        if idx < S2_MODAL_COLS {
            ll[m] = a;
            rl[m] = b;
        } else {
            lr[m] = a;
            rr[m] = b;
        }
    }
    CoupledS2Cols { ll, rl, lr, rr }
}

pub(super) fn read_direct_s2_weights(ro: &V3TierReadout, ch: usize, dst: &mut [f64]) {
    for b in 0..3 {
        for l in 0..4 {
            dst[b * 4 + l] = f64::from(ro.direct_s2_w[ch][b * 12 + l]);
        }
    }
}

pub(super) fn write_direct_s2_weights(src: &[f64], ch: usize, ro: &mut V3TierReadout) {
    // Keep combined `pre_cab_w + direct_s2_w` L2 norm safely below Phase-3 +6.0 dB head gain rail.
    let max_sq = max_direct_s2_l2_sq(ro, ch);
    let sq: f64 = src[..S2_MODAL_COLS].iter().map(|&w| w * w).sum();
    let scale = if max_sq <= 0.0 {
        0.0
    } else if sq > max_sq {
        (max_sq / sq).sqrt()
    } else {
        1.0
    };
    for b in 0..3 {
        for l in 0..4 {
            ro.direct_s2_w[ch][b * 12 + l] = (src[b * 4 + l] * scale) as f32;
        }
    }
}

fn fit_1x_modal_stems<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    stems: &[V3TrainStem],
    weights: &[f64],
) {
    use crate::solver::v3_varpro::basis::linear_basis_dim;
    use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};
    use dnam_engine::operators::op_m::algebraic_unit_rotor_op_m;
    use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, tier_kautz_banks};

    let t = tier as usize;
    if model.tier_os2x[t] || stems.is_empty() {
        return;
    }
    let has_valid = stems.iter().any(|st| {
        let n = st.usable_len();
        n > st.warmup.min(n.saturating_sub(1)) + 64
    });
    if !has_valid {
        return;
    }
    let sr = 48_000.0_f64;
    model.rate_1x_cal = Some(build_1x_linear_resonator_cal(sr));
    model.readouts[t].pre_cab_u_gain = [1.0, 1.0];
    model.readouts[t].pre_cab_hp_gain = [0.0, 0.0];
    model.readouts[t].pre_cab_w = [[0.0; MAX_TOTAL_GRU_LANES]; 2];
    model.readouts[t].svf_direct_taps_l = [0.0; 8];
    model.readouts[t].svf_direct_taps_r = [0.0; 8];
    model.readouts[t].direct_s2_u_gain = [0.0, 0.0];
    model.readouts[t].direct_s2_hp_gain = [0.0, 0.0];

    let fir_len = (t + 1) * 16;
    let k_cnt = tier_kautz_banks(tier) * 4;
    let dim_l = linear_basis_dim(tier) + S2_MODAL_COLS;
    let k_prep: Vec<(f32, f32, f32)> = model.kautz_poles[..k_cnt]
        .iter()
        .map(|&(r_raw, th_raw)| {
            let r = r_raw.clamp(0.25, 0.9995);
            let th = th_raw.clamp(1.0e-4, core::f32::consts::PI - 1.0e-4);
            let (ct, st) = algebraic_unit_rotor_op_m(th);
            let bg = ((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)).sqrt() as f32;
            (r * ct, r * st, bg)
        })
        .collect();

    let n_ch = if model.mono_out { 1 } else { 2 };
    let mut grams = vec![vec![0.0_f64; uplo_u_packed_len(dim_l)]; n_ch];
    let mut rhss = vec![vec![[0.0_f64; 2]; dim_l]; n_ch];

    for (si, stem) in stems.iter().enumerate() {
        let sw = weights.get(si).copied().unwrap_or(1.0).max(0.0);
        let n = stem.usable_len();
        let warmup = stem.warmup.min(n.saturating_sub(1));
        if sw <= 0.0 || n <= warmup + 64 {
            continue;
        }
        let s2_cols = extract_decoupled_s2_cols::<B>(model, tier, stem);
        let mut tank = V3Stage4TankHot::from_cold(&model.stage4[t]);
        let mut s3_in_l = vec![0.0_f32; n];
        let mut s3_in_r = vec![0.0_f32; n];
        for i in 0..n {
            let ir = if model.mono_out {
                stem.in_l[i]
            } else {
                stem.in_r[i]
            };
            let [yl, yr] = tank.step_stereo::<B>(stem.in_l[i], ir);
            s3_in_l[i] = yl;
            s3_in_r[i] = yr;
        }
        for ch in 0..n_ch {
            let x = if ch == 0 { &s3_in_l } else { &s3_in_r };
            let y = if ch == 0 { &stem.tgt_l } else { &stem.tgt_r };
            let s2 = if ch == 0 { &s2_cols.l } else { &s2_cols.r };
            let mut ks = [(0.0_f32, 0.0_f32); MAX_KAUTZ_POLES];
            let mut row = vec![0.0_f64; dim_l];
            for i in 0..n {
                for d in 0..fir_len {
                    row[d] = if i >= d { f64::from(x[i - d]) } else { 0.0 };
                }
                let x_del = if i >= 16 { x[i - 16] } else { 0.0 };
                for (m, &(lr, li, bg)) in k_prep.iter().enumerate() {
                    let (sr0, si0) = ks[m];
                    let nr = lr * sr0 - li * si0 + bg * x_del;
                    let ni = li * sr0 + lr * si0;
                    ks[m] = (nr, ni);
                    row[fir_len + 2 * m] = f64::from(nr);
                    row[fir_len + 2 * m + 1] = -f64::from(ni);
                }
                let s2_off = fir_len + 2 * k_cnt;
                for m in 0..S2_MODAL_COLS {
                    row[s2_off + m] = f64::from(s2[m][i]);
                }
                if i >= warmup {
                    let tgt = f64::from(y[i]);
                    for c in 0..dim_l {
                        let rc = sw * row[c];
                        rhss[ch][c][0] += rc * tgt;
                        for r in 0..=c {
                            grams[ch][uplo_u_index(r, c)] += row[r] * rc;
                        }
                    }
                }
            }
        }
    }

    let s2_off = fir_len + 2 * k_cnt;
    for ch in 0..n_ch {
        for m in 0..S2_MODAL_COLS {
            let idx = s2_off + m;
            grams[ch][uplo_u_index(idx, idx)] += 1.0e-3;
        }
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(&grams[ch], &rhss[ch], 1.0e-7) {
            let ro = &mut model.readouts[t];
            if ch == 0 {
                ro.fir_l = [0.0; 64];
                ro.kautz_c_l = [(0.0, 0.0); MAX_KAUTZ_POLES];
                for d in 0..fir_len {
                    ro.fir_l[d] = sol[d][0] as f32;
                }
                for m in 0..k_cnt {
                    ro.kautz_c_l[m] = (
                        sol[fir_len + 2 * m][0] as f32,
                        sol[fir_len + 2 * m + 1][0] as f32,
                    );
                }
            } else {
                ro.fir_r = [0.0; 64];
                ro.kautz_c_r = [(0.0, 0.0); MAX_KAUTZ_POLES];
                for d in 0..fir_len {
                    ro.fir_r[d] = sol[d][0] as f32;
                }
                for m in 0..k_cnt {
                    ro.kautz_c_r[m] = (
                        sol[fir_len + 2 * m][0] as f32,
                        sol[fir_len + 2 * m + 1][0] as f32,
                    );
                }
            }
            let s2_vals: Vec<f64> = (0..S2_MODAL_COLS).map(|m| sol[s2_off + m][0]).collect();
            write_direct_s2_weights(&s2_vals, ch, ro);
        }
    }
    if model.mono_out {
        model.readouts[t].fir_r = model.readouts[t].fir_l;
        model.readouts[t].kautz_c_r = model.readouts[t].kautz_c_l;
        model.readouts[t].direct_s2_w[1] = model.readouts[t].direct_s2_w[0];
    }
    model.readouts[t].mask_to_tier(tier);
}

/// Fits a `1x` linear readout (`FIR` + `Kautz` + `12` Stage-2 Group-0 resonators) on a single stem
/// via exact `uplo='U'` Cholesky regression for deterministic `1x` LTI tiers (`Lite` / `Full`).
pub(in crate::solver::v3_varpro::stage4_id) fn fit_1x_modal_tier_linear_cholesky<
    B: TrainerBackend,
>(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
) {
    fit_1x_modal_stems::<B>(model, tier, std::slice::from_ref(stem), &[1.0]);
}

/// Fits a `1x` linear modal readout accumulated across all stems in `dataset` weighted by `stem_weights`.
pub(in crate::solver::v3_varpro::stage4_id) fn fit_1x_modal_tier_linear_cholesky_dataset<
    B: TrainerBackend,
>(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    dataset: &V3TrainDataset,
) {
    fit_1x_modal_stems::<B>(model, tier, &dataset.stems, &dataset.stem_weights);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::ScalarTrainerBackend;

    #[test]
    fn test_s2_modal_1x_basis_matches_hot_core_superposition() {
        let sr = 48_000.0;
        let n = 1_024;
        let mut model = V3MatryoshkaModel::dual_rate_default(false);
        model.rate_1x_cal = Some(build_1x_linear_resonator_cal(sr));
        assert!(has_1x_modal_basis(&model, Tier::Lite));
        assert!(has_1x_modal_basis(&model, Tier::Full));
        assert!(!has_1x_modal_basis(&model, Tier::Pro));
        assert!(!has_1x_modal_basis(&model, Tier::Ultra));

        let in_l: Vec<f32> = (0..n).map(|i| 0.12 * ((i as f32) * 0.17).sin()).collect();
        let in_r: Vec<f32> = (0..n).map(|i| 0.12 * ((i as f32) * 0.23).cos()).collect();
        let stem = V3TrainStem {
            in_l: in_l.clone(),
            in_r: in_r.clone(),
            tgt_l: vec![0.0; n],
            tgt_r: vec![0.0; n],
            warmup: 64,
        };
        let cols = extract_decoupled_s2_cols::<ScalarTrainerBackend>(&model, Tier::Lite, &stem);
        let w = [
            0.7, -0.4, 0.5, 0.2, -0.6, 0.3, -0.2, 0.8, 0.4, -0.5, 0.25, -0.35,
        ];
        write_direct_s2_weights(&w, 0, &mut model.readouts[0]);
        write_direct_s2_weights(&w, 1, &mut model.readouts[0]);
        model.readouts[0].pre_cab_u_gain = [0.0, 0.0];
        model.readouts[0].fir_l = [0.0; 64];
        model.readouts[0].fir_r = [0.0; 64];

        let cold = model.extract_tier(Tier::Lite);
        let mut core = V3HotCore::<ScalarTrainerBackend>::from_cold_model(&cold);
        core.reset_to_dc_equilibrium();
        let (mut ol, mut or) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        core.process_block_stereo(&in_l, &in_r, &mut ol, &mut or);
        for i in 0..n {
            let mut pl = 0.0_f64;
            let mut pr = 0.0_f64;
            for m in 0..S2_MODAL_COLS {
                pl += w[m] * f64::from(cols.l[m][i]);
                pr += w[m] * f64::from(cols.r[m][i]);
            }
            assert!((f64::from(ol[i]) - pl).abs() < 1.0e-5, "L mismatch at {i}");
            assert!((f64::from(or[i]) - pr).abs() < 1.0e-5, "R mismatch at {i}");
        }
    }
}
