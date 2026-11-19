#![forbid(unsafe_code)]
//! Exact `f32` White-Box `preserve_existing_backbone` Readout Solver (`Oracle-01` / Path 1).

use dnam_engine::math::{DefaultMathBackend, SimdVec4, flush_subnormal_f32_bits};
use dnam_engine::operators::op_m::KautzModalBank;
use dnam_engine::stages::router::V3Stage0Hot;
use dnam_engine::stages::stage1_svf::V3Stage1SvfState;
use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS,
    NUM_IN_FEATS_B0, V3GruBlockHot, V3GruChannelState, V3InputFeaturesState,
    V3InterBlockToneSvfState, ZeroLagOversampler2x, flat_gru_lane_index, tier_block_groups_array,
    tier_block_lanes_array, tier_fir_segments, tier_kautz_banks, tier_total_gru_lanes,
};
use dnam_engine::stages::{V3ColdTierModel, V3MatryoshkaModel, V3TierReadout};
use dnam_engine::{FtzDazGuard, Tier};

use super::WeightedStemRef;
use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};
use crate::solver::v3_varpro::basis::{
    ALL_TIERS, linear_basis_dim, solve_matryoshka_varpro_cholesky,
};
use crate::solver::v3_varpro::{V3MatryoshkaOptF64, V3TrainDataset, V3TrainStem};
use crate::solver::{
    canonical_besr_warmup, solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len,
};

const KAUTZ_ONSET_SAMPLES: usize = 16;

struct TierEngineBasis {
    clean_sub: [Vec<f32>; 2],
    s2: [Vec<f32>; 2],
    tap: [Vec<[f64; 7]>; 2],
    u: [Vec<f64>; 2],
    hp: [Vec<f64>; 2],
    h: [Vec<[f64; MAX_TOTAL_GRU_LANES]>; 2],
}

fn has_zeroed_pre_cab(ro: &V3TierReadout) -> bool {
    ro.pre_cab_u_gain.iter().all(|&v| v == 0.0)
        && ro.pre_cab_hp_gain.iter().all(|&v| v == 0.0)
        && ro.pre_cab_w.iter().flatten().all(|&v| v == 0.0)
}

fn populate_zeroed_pre_cab(
    model: &mut V3MatryoshkaModel,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) {
    let mut zero_mask = [false; 4];
    for (tau, ro) in model.readouts.iter_mut().enumerate() {
        if has_zeroed_pre_cab(ro) {
            ro.pre_cab_u_gain = [1.0, 1.0];
            zero_mask[tau] = true;
        }
    }
    if !zero_mask.iter().any(|&z| z) {
        return;
    }
    let n = dry_lr.len().min(wet_lr.len());
    let mono = model.mono_out;
    let keep_in_r = !mono || model.feat_params.key_link > 0.0;
    let stem = V3TrainStem {
        in_l: dry_lr[..n].iter().map(|lr| lr[0]).collect(),
        in_r: dry_lr[..n]
            .iter()
            .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
            .collect(),
        tgt_l: wet_lr[..n].iter().map(|lr| lr[0]).collect(),
        tgt_r: wet_lr[..n]
            .iter()
            .map(|lr| if mono { lr[0] } else { lr[1] })
            .collect(),
        warmup: canonical_besr_warmup(n),
    };
    let ds = V3TrainDataset::new_weighted_opts(vec![(stem, 1.0)], 48_000.0, false, false);
    let mut opt = Box::new(V3MatryoshkaOptF64::from_matryoshka_cold(model));
    opt.active_tier_weights = [0.20, 0.22, 0.26, 0.32];
    solve_matryoshka_varpro_cholesky(&mut opt, &ds, 2);
    let fitted = opt.to_matryoshka_cold();
    for tau in 0..4 {
        if zero_mask[tau] {
            model.readouts[tau].pre_cab_u_gain = fitted.readouts[tau].pre_cab_u_gain;
            model.readouts[tau].pre_cab_hp_gain = fitted.readouts[tau].pre_cab_hp_gain;
            model.readouts[tau].pre_cab_w = fitted.readouts[tau].pre_cab_w;
        }
    }
}

fn eval_s2_sample(
    ro: &V3TierReadout,
    ch: usize,
    feats: &[f32; NUM_IN_FEATS_B0],
    h_ch: &[[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    active_groups: &[usize; NUM_GRU_BLOCKS],
    cab_inj_term: f32,
) -> f32 {
    let mut s2 =
        ro.pre_cab_u_gain[ch] * feats[0] + ro.pre_cab_hp_gain[ch] * feats[1] + cab_inj_term;
    for (b, &g_b) in active_groups.iter().enumerate() {
        for g in 0..g_b {
            for l in 0..4 {
                s2 += ro.pre_cab_w[ch][flat_gru_lane_index(b, g, l)] * h_ch[b][g][l];
            }
        }
    }
    s2
}

#[allow(clippy::too_many_arguments)]
fn step_substep_blocks(
    blocks_hot: &[V3GruBlockHot; NUM_GRU_BLOCKS],
    active_groups: &[usize; NUM_GRU_BLOCKS],
    active_ch: usize,
    feats: &[[f32; NUM_IN_FEATS_B0]; 2],
    u_s2: [f32; 2],
    ch_states: &mut [V3GruChannelState; 2],
    tone_svf: &mut V3InterBlockToneSvfState,
    tone_out: &mut [f32; 2],
) {
    let zero_w = [[0.0_f32; 4]; MAX_GROUPS];
    let mut blk_feats = *feats;
    let mut prev_h = [[[0.0_f32; 4]; MAX_GROUPS]; 2];
    let mut tone_in = [0.0_f32; 2];

    for b in 0..NUM_GRU_BLOCKS {
        let g_b = active_groups[b];
        for (ch, cst) in ch_states.iter_mut().enumerate().take(active_ch) {
            let (mut ca, mut da) = (
                <DefaultMathBackend as dnam_engine::math::MathBackend>::Vec4::zero(),
                <DefaultMathBackend as dnam_engine::math::MathBackend>::Vec4::zero(),
            );
            blocks_hot[b].step_block::<DefaultMathBackend>(
                b,
                g_b,
                &blk_feats[ch],
                &prev_h[ch],
                &mut cst.h[b],
                &mut ca,
                &mut da,
                &zero_w,
                &zero_w,
            );
            prev_h[ch] = cst.h[b];
            if b == 0 {
                let h0w = &tone_svf.params.h0_proj_w;
                let h0 = &prev_h[ch][0];
                tone_in[ch] = tone_svf.params.s1_proj_w * u_s2[ch]
                    + h0w[0] * h0[0]
                    + h0w[1] * h0[1]
                    + h0w[2] * h0[2]
                    + h0w[3] * h0[3];
            }
        }
        if b == 0 {
            let r_in = if active_ch == 1 {
                tone_in[0]
            } else {
                tone_in[1]
            };
            *tone_out = tone_svf.step_stereo(tone_in[0], r_in);
            for ch in 0..active_ch {
                blk_feats[ch][0] = feats[ch][0] + tone_svf.params.b1_inject_w * tone_out[ch];
            }
        } else {
            for ch in 0..active_ch {
                blk_feats[ch][0] = feats[ch][0];
            }
        }
    }
}

fn alloc_empty_basis(n: usize, active_ch: usize) -> TierEngineBasis {
    let n_r = if active_ch > 1 { n } else { 0 };
    TierEngineBasis {
        clean_sub: [vec![0.0; n], vec![0.0; n_r]],
        s2: [vec![0.0; n], vec![0.0; n_r]],
        tap: [vec![[0.0; 7]; n], vec![[0.0; 7]; n_r]],
        u: [vec![0.0; n], vec![0.0; n_r]],
        hp: [vec![0.0; n], vec![0.0; n_r]],
        h: [
            vec![[0.0; MAX_TOTAL_GRU_LANES]; n],
            vec![[0.0; MAX_TOTAL_GRU_LANES]; n_r],
        ],
    }
}

fn extract_tier_engine_basis(cold: &V3ColdTierModel, dry_lr: &[[f32; 2]]) -> TierEngineBasis {
    let _ftz = FtzDazGuard::new();
    let n = dry_lr.len();
    let active_ch = if cold.mono_out { 1 } else { 2 };
    let mut basis = alloc_empty_basis(n, active_ch);

    let mut stage0 = V3Stage0Hot::from_cold(&cold.stage0);
    let mut svf_p = cold.svf;
    svf_p.clamp_in_place();
    let mut feat_p = cold.feat_params;
    feat_p.clamp_in_place();
    let mut os_p = cold.os_params;
    os_p.clamp_in_place();
    let mut tone_svf = V3InterBlockToneSvfState::from_cold(&cold.tone_svf);
    let active_groups = tier_block_groups_array(cold.tier);
    let blocks_hot: [V3GruBlockHot; NUM_GRU_BLOCKS] = std::array::from_fn(|b| {
        V3GruBlockHot::from_cold::<DefaultMathBackend>(&cold.blocks[b], active_groups[b])
    });

    let mut svf_state = V3Stage1SvfState::default();
    let (mut os_l, mut os_r) = (
        ZeroLagOversampler2x::default(),
        ZeroLagOversampler2x::default(),
    );
    let (mut dec_tap_p, mut dec_u_p, mut dec_hp_p, mut dec_s2_p) =
        ([[0.0_f32; 7]; 2], [0.0_f32; 2], [0.0_f32; 2], [0.0_f32; 2]);
    let mut dec_h_p = [[0.0_f32; MAX_TOTAL_GRU_LANES]; 2];
    let (mut dec_tap_aa, mut dec_u_aa, mut dec_hp_aa, mut dec_s2_aa) = (
        [[[0.0_f32; 4]; 7]; 2],
        [[0.0_f32; 4]; 2],
        [[0.0_f32; 4]; 2],
        [[0.0_f32; 4]; 2],
    );
    let mut dec_h_aa = [[[0.0_f32; 4]; MAX_TOTAL_GRU_LANES]; 2];
    let mut ch_states = [V3GruChannelState::default(), V3GruChannelState::default()];
    let sub_steps = if cold.os2x { 2 } else { 1 };

    let keep_in_r = !cold.mono_out || cold.feat_params.key_link > 0.0;
    for (t, &[in_l, in_r]) in dry_lr.iter().enumerate() {
        let xl = if in_l.is_finite() {
            in_l.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        let xr = if !keep_in_r {
            xl
        } else if in_r.is_finite() {
            in_r.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        let [s0_l, s0_r, csub_l, csub_r] = stage0.step_stereo(xl, xr);
        basis.clean_sub[0][t] = csub_l;
        if active_ch > 1 {
            basis.clean_sub[1][t] = csub_r;
        }
        let xs_sub = if cold.os2x {
            let lp = os_l.upsample(s0_l, &os_p);
            let rp = if !keep_in_r {
                lp
            } else {
                os_r.upsample(s0_r, &os_p)
            };
            [[lp[0], rp[0]], [lp[1], rp[1]]]
        } else {
            [[s0_l, s0_r], [0.0, 0.0]]
        };
        let (mut tap_sub, mut u_sub, mut hp_sub, mut s2_sub) = (
            [[[0.0_f32; 7]; 2]; 2],
            [[0.0_f32; 2]; 2],
            [[0.0_f32; 2]; 2],
            [[0.0_f32; 2]; 2],
        );
        let mut h_sub = [[[0.0_f32; MAX_TOTAL_GRU_LANES]; 2]; 2];

        for sub in 0..sub_steps {
            let (u_s2, _, tap_ch) =
                svf_state.step_stereo_with_taps(&svf_p, xs_sub[sub][0], xs_sub[sub][1]);
            tap_sub[sub] = tap_ch;
            let (left_s, right_s) = ch_states.split_at_mut(1);
            let (feat_l, feat_r) = V3InputFeaturesState::step_stereo_with_lp2::<DefaultMathBackend>(
                &mut left_s[0].feat,
                &mut right_s[0].feat,
                &feat_p,
                u_s2[0],
                u_s2[1],
                [tap_ch[0][4], tap_ch[1][4]],
                svf_p.feat_lp2_mix,
            );
            let feats = [feat_l, feat_r];
            for ch in 0..active_ch {
                u_sub[sub][ch] = feats[ch][0];
                hp_sub[sub][ch] = feats[ch][1];
            }
            let mut tone_out = [0.0_f32; 2];
            step_substep_blocks(
                &blocks_hot,
                &active_groups,
                active_ch,
                &feats,
                u_s2,
                &mut ch_states,
                &mut tone_svf,
                &mut tone_out,
            );
            for ch in 0..active_ch {
                s2_sub[sub][ch] = eval_s2_sample(
                    &cold.readout,
                    ch,
                    &feats[ch],
                    &ch_states[ch].h,
                    &active_groups,
                    tone_svf.params.cab_inject_w * tone_out[ch],
                );
                for (b, &g_b) in active_groups.iter().enumerate() {
                    for g in 0..g_b {
                        for l in 0..4 {
                            h_sub[sub][ch][flat_gru_lane_index(b, g, l)] = ch_states[ch].h[b][g][l];
                        }
                    }
                }
            }
        }

        for ch in 0..active_ch {
            if !cold.os2x {
                basis.s2[ch][t] = s2_sub[0][ch];
                basis.tap[ch][t] = tap_sub[0][ch].map(f64::from);
                basis.u[ch][t] = f64::from(u_sub[0][ch]);
                basis.hp[ch][t] = f64::from(hp_sub[0][ch]);
                basis.h[ch][t] = h_sub[0][ch].map(f64::from);
            } else {
                basis.s2[ch][t] = ZeroLagOversampler2x::downsample(
                    s2_sub[0][ch],
                    s2_sub[1][ch],
                    &mut dec_s2_p[ch],
                    &mut dec_s2_aa[ch],
                    &os_p,
                );
                for i in 0..7 {
                    basis.tap[ch][t][i] = f64::from(ZeroLagOversampler2x::downsample(
                        tap_sub[0][ch][i],
                        tap_sub[1][ch][i],
                        &mut dec_tap_p[ch][i],
                        &mut dec_tap_aa[ch][i],
                        &os_p,
                    ));
                }
                basis.u[ch][t] = f64::from(ZeroLagOversampler2x::downsample(
                    u_sub[0][ch],
                    u_sub[1][ch],
                    &mut dec_u_p[ch],
                    &mut dec_u_aa[ch],
                    &os_p,
                ));
                basis.hp[ch][t] = f64::from(ZeroLagOversampler2x::downsample(
                    hp_sub[0][ch],
                    hp_sub[1][ch],
                    &mut dec_hp_p[ch],
                    &mut dec_hp_aa[ch],
                    &os_p,
                ));
                for (b, &g_b) in active_groups.iter().enumerate() {
                    let base = b * MAX_BLOCK_LANES;
                    for idx in base..base + g_b * 4 {
                        basis.h[ch][t][idx] = f64::from(ZeroLagOversampler2x::downsample(
                            h_sub[0][ch][idx],
                            h_sub[1][ch][idx],
                            &mut dec_h_p[ch][idx],
                            &mut dec_h_aa[ch][idx],
                            &os_p,
                        ));
                    }
                }
            }
        }
    }
    basis
}

fn build_tier_f32_columns(
    cold: &V3ColdTierModel,
    basis: &TierEngineBasis,
    ch: usize,
) -> Vec<Vec<f64>> {
    let n = basis.s2[ch].len();
    let c_t = tier_block_lanes_array(cold.tier);
    let total_h = tier_total_gru_lanes(cold.tier);
    let f_t = tier_fir_segments(cold.tier) * 16;
    let k_banks = tier_kautz_banks(cold.tier);
    let p_dim = linear_basis_dim(cold.tier);
    let mut cols = vec![vec![0.0_f64; n]; p_dim];

    for t in 0..n {
        for i in 0..7 {
            cols[i][t] = basis.tap[ch][t][i];
        }
        cols[7][t] = basis.u[ch][t];
        cols[8][t] = basis.hp[ch][t];
    }
    let mut idx = 9;
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for l in 0..c_t[b] {
            let lane = base + l;
            for t in 0..n {
                cols[idx][t] = basis.h[ch][t][lane];
            }
            idx += 1;
        }
    }
    let s2 = &basis.s2[ch];
    let s2_f64: Vec<f64> = s2.iter().map(|&v| f64::from(v)).collect();
    for k in 0..f_t {
        if k < n {
            cols[idx + k][k..n].copy_from_slice(&s2_f64[..n - k]);
        }
    }

    let k_base = 9 + total_h + f_t;
    let mut banks: [KautzModalBank<4>; 4] = std::array::from_fn(|_| KautzModalBank::<4>::new());
    for (b, bank) in banks.iter_mut().enumerate().take(k_banks) {
        for lane in 0..4 {
            let m = b * 4 + lane;
            if m < MAX_KAUTZ_POLES {
                let (r_raw, th_raw) = cold.kautz_poles[m];
                let r = r_raw.clamp(0.25, 0.9995);
                let th = th_raw.clamp(0.0, core::f32::consts::PI - 1.0e-4);
                let bg = if cold.readout.kautz_b_gamma[m] > 0.0 {
                    cold.readout.kautz_b_gamma[m]
                } else {
                    (1.0 - f64::from(r) * f64::from(r)).max(1.0e-7).sqrt() as f32
                };
                bank.set_pole_with_b_gamma(lane, r, th, 0.0, bg, 1.0, 0.0);
            }
        }
    }
    for (t, _) in s2.iter().enumerate() {
        let x_del = if t >= KAUTZ_ONSET_SAMPLES {
            s2[t - KAUTZ_ONSET_SAMPLES]
        } else {
            0.0
        };
        for (b, bank) in banks.iter_mut().enumerate().take(k_banks) {
            for lane in 0..4 {
                let m = b * 4 + lane;
                let nr = flush_subnormal_f32_bits(
                    bank.lambda_re[lane] * bank.s_re[lane] - bank.lambda_im[lane] * bank.s_im[lane]
                        + bank.b_gamma[lane] * x_del,
                );
                let ni = flush_subnormal_f32_bits(
                    bank.lambda_im[lane] * bank.s_re[lane] + bank.lambda_re[lane] * bank.s_im[lane],
                );
                bank.s_re[lane] = nr;
                bank.s_im[lane] = ni;
                cols[k_base + 2 * m][t] = f64::from(nr);
                cols[k_base + 2 * m + 1][t] = -f64::from(ni);
            }
        }
    }
    cols
}

fn unpack_theta_to_readout(tier: Tier, ch: usize, theta: &[f64], r_out: &mut V3TierReadout) {
    let c_t = tier_block_lanes_array(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = tier_kautz_banks(tier) * 4;
    if ch == 0 {
        r_out.svf_direct_taps_l = [0.0; 8];
        for i in 0..7 {
            r_out.svf_direct_taps_l[i] = theta[i] as f32;
        }
    } else {
        r_out.svf_direct_taps_r = [0.0; 8];
        for i in 0..7 {
            r_out.svf_direct_taps_r[i] = theta[i] as f32;
        }
    }
    r_out.direct_s2_u_gain[ch] = theta[7] as f32;
    r_out.direct_s2_hp_gain[ch] = theta[8] as f32;
    r_out.direct_s2_w[ch].fill(0.0);
    let mut idx = 9;
    for (b, &cb) in c_t.iter().enumerate() {
        let base = b * MAX_BLOCK_LANES;
        for l in 0..cb {
            r_out.direct_s2_w[ch][base + l] = theta[idx] as f32;
            idx += 1;
        }
    }
    let fir = if ch == 0 {
        &mut r_out.fir_l
    } else {
        &mut r_out.fir_r
    };
    fir.fill(0.0);
    for k in 0..f_t {
        fir[k] = theta[idx + k] as f32;
    }
    idx += f_t;
    let kc = if ch == 0 {
        &mut r_out.kautz_c_l
    } else {
        &mut r_out.kautz_c_r
    };
    kc.fill((0.0, 0.0));
    for m in 0..k_t {
        kc[m] = (theta[idx + 2 * m] as f32, theta[idx + 2 * m + 1] as f32);
    }
}

fn pack_cold_readout_to_theta(tier: Tier, ch: usize, r_out: &V3TierReadout) -> Vec<f64> {
    let c_t = tier_block_lanes_array(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = tier_kautz_banks(tier) * 4;
    let mut theta = Vec::with_capacity(linear_basis_dim(tier));
    let taps = if ch == 0 {
        &r_out.svf_direct_taps_l
    } else {
        &r_out.svf_direct_taps_r
    };
    for &v in &taps[..7] {
        theta.push(f64::from(v));
    }
    theta.push(f64::from(r_out.direct_s2_u_gain[ch]));
    theta.push(f64::from(r_out.direct_s2_hp_gain[ch]));
    for (b, &cb) in c_t.iter().enumerate() {
        let base = b * MAX_BLOCK_LANES;
        for l in 0..cb {
            theta.push(f64::from(r_out.direct_s2_w[ch][base + l]));
        }
    }
    let fir = if ch == 0 { &r_out.fir_l } else { &r_out.fir_r };
    for &v in &fir[..f_t] {
        theta.push(f64::from(v));
    }
    let kc = if ch == 0 {
        &r_out.kautz_c_l
    } else {
        &r_out.kautz_c_r
    };
    for &(re, im) in &kc[..k_t] {
        theta.push(f64::from(re));
        theta.push(f64::from(im));
    }
    theta
}

fn solve_gram_and_unpack(
    tier: Tier,
    ch: usize,
    gram: &[f64],
    rhs: &[[f64; 2]],
    trace: f64,
    anchor_alpha: Option<f64>,
    r_out: &mut V3TierReadout,
) {
    let p_dim = rhs.len();
    let mean_diag = (trace / (p_dim.max(1) as f64)).max(1.0e-12);
    let (g_work, r_work, mults): (Vec<f64>, Vec<[f64; 2]>, &[f64]) =
        if let Some(alpha) = anchor_alpha {
            let theta0 = pack_cold_readout_to_theta(tier, ch, r_out);
            let mut gw = gram.to_vec();
            let mut rw = rhs.to_vec();
            let lam = alpha * mean_diag;
            for (j, &t0) in theta0.iter().enumerate().take(p_dim) {
                gw[uplo_u_index(j, j)] += lam;
                rw[j][0] += lam * t0;
            }
            (gw, rw, &[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
        } else {
            (
                gram.to_vec(),
                rhs.to_vec(),
                &[1.0e-13, 1.0e-11, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3],
            )
        };
    let mut theta = vec![0.0_f64; p_dim];
    for &mult in mults {
        let ridge = mult * mean_diag + 1.0e-15;
        if let Ok(sol) = solve_uplo_u_cholesky_2ch(&g_work, &r_work, ridge) {
            theta = sol.into_iter().map(|v| v[0]).collect();
            break;
        }
    }
    unpack_theta_to_readout(tier, ch, &theta, r_out);
}

fn mirror_readout_l_to_r(ro: &mut V3TierReadout) {
    ro.svf_direct_taps_r = ro.svf_direct_taps_l;
    ro.pre_cab_u_gain[1] = ro.pre_cab_u_gain[0];
    ro.pre_cab_hp_gain[1] = ro.pre_cab_hp_gain[0];
    ro.pre_cab_w[1] = ro.pre_cab_w[0];
    ro.direct_s2_u_gain[1] = ro.direct_s2_u_gain[0];
    ro.direct_s2_hp_gain[1] = ro.direct_s2_hp_gain[0];
    ro.direct_s2_w[1] = ro.direct_s2_w[0];
    ro.fir_r = ro.fir_l;
    ro.kautz_c_r = ro.kautz_c_l;
}

struct WeightedGramAccum {
    active_ch: usize,
    grams: Vec<Vec<f64>>,
    rhss: Vec<Vec<[f64; 2]>>,
    traces: Vec<f64>,
}

impl WeightedGramAccum {
    fn new(p_dim: usize, active_ch: usize) -> Self {
        Self {
            active_ch,
            grams: vec![vec![0.0_f64; uplo_u_packed_len(p_dim)]; active_ch],
            rhss: vec![vec![[0.0_f64; 2]; p_dim]; active_ch],
            traces: vec![0.0_f64; active_ch],
        }
    }

    fn accumulate_stem<B: TrainerBackend>(
        &mut self,
        cold_t: &V3ColdTierModel,
        dry_lr: &[[f32; 2]],
        wet_lr: &[[f32; 2]],
        weight: f64,
    ) {
        let n = dry_lr.len().min(wet_lr.len());
        if n == 0 || weight <= 0.0 || !weight.is_finite() {
            return;
        }
        let warmup = canonical_besr_warmup(n).min(n.saturating_sub(1));
        let basis = extract_tier_engine_basis(cold_t, &dry_lr[..n]);
        for ch in 0..self.active_ch {
            let cols = build_tier_f32_columns(cold_t, &basis, ch);
            let p_dim = cols.len();
            let target: Vec<f64> = (0..n)
                .map(|t| f64::from(wet_lr[t][ch]) - f64::from(basis.clean_sub[ch][t]))
                .collect();
            for j in 0..p_dim {
                let cj = &cols[j][warmup..n];
                self.rhss[ch][j][0] += weight * B::dot_f64(cj, &target[warmup..n]);
                for i in 0..=j {
                    let dot = weight * B::dot_f64(&cols[i][warmup..n], cj);
                    self.grams[ch][uplo_u_index(i, j)] += dot;
                    if i == j {
                        self.traces[ch] += dot;
                    }
                }
            }
        }
    }
}

fn solve_weighted_whitebox_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    stems: &[WeightedStemRef<'_>],
    anchor_alpha: Option<f64>,
) {
    if stems.is_empty() {
        return;
    }
    let active_ch = if model.mono_out { 1 } else { 2 };
    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        let cold_t = model.extract_tier(tier);
        let p_dim = linear_basis_dim(tier);
        let mut accum = WeightedGramAccum::new(p_dim, active_ch);
        for stem in stems {
            accum.accumulate_stem::<B>(&cold_t, stem.dry_lr, stem.wet_lr, stem.weight);
        }
        for ch in 0..active_ch {
            solve_gram_and_unpack(
                tier,
                ch,
                &accum.grams[ch],
                &accum.rhss[ch],
                accum.traces[ch],
                anchor_alpha,
                &mut model.readouts[tau],
            );
        }
        if model.mono_out {
            mirror_readout_l_to_r(&mut model.readouts[tau]);
        }
    }
}

fn dispatch_weighted_whitebox(
    model: &mut V3MatryoshkaModel,
    stems: &[WeightedStemRef<'_>],
    anchor_alpha: Option<f64>,
) {
    let Some(first) = stems.first() else {
        return;
    };
    let n0 = first.dry_lr.len().min(first.wet_lr.len());
    if n0 == 0 {
        return;
    }
    populate_zeroed_pre_cab(model, &first.dry_lr[..n0], &first.wet_lr[..n0]);
    struct WeightedWhiteboxTask<'a>(
        &'a mut V3MatryoshkaModel,
        &'a [WeightedStemRef<'a>],
        Option<f64>,
    );
    impl TrainerTask<()> for WeightedWhiteboxTask<'_> {
        fn run<B: TrainerBackend>(self) {
            solve_weighted_whitebox_with_backend::<B>(self.0, self.1, self.2);
        }
    }
    DynamicTrainer::detect().dispatch(WeightedWhiteboxTask(model, stems, anchor_alpha));
}

/// Exact `f32` engine trajectory readout solver for `preserve_existing_backbone` white-box recovery.
pub(super) fn solve_preserved_backbone_f32_readouts(
    model: &mut V3MatryoshkaModel,
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) {
    let n = dry_lr.len().min(wet_lr.len());
    if n == 0 {
        return;
    }
    dispatch_weighted_whitebox(
        model,
        &[WeightedStemRef {
            dry_lr: &dry_lr[..n],
            wet_lr: &wet_lr[..n],
            weight: 1.0,
        }],
        None,
    );
}

/// Fast anchored multi-stem weighted exact-`f32` `uplo='U'` Cholesky readout patch across all 4 tiers (`< 50 ms`).
pub(super) fn patch_v3_readouts_weighted_cholesky(
    model: &mut V3MatryoshkaModel,
    stems: &[WeightedStemRef<'_>],
) {
    dispatch_weighted_whitebox(model, stems, Some(1.0e-4));
}
