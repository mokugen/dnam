#![forbid(unsafe_code)]
//! Matryoshka group unlocking (`lanes 0..16`) after initial `Lite` prescreen.

use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_GROUPS, MAX_GROUPS_PER_BLOCK, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
};

use super::super::model_f64::{V3GruBlockF64, V3MatryoshkaOptF64};

const GRID_LEAK_SLOPES: [[f64; 4]; 3] = [
    [0.48, 0.78, 1.28, 1.75],
    [0.62, 0.92, 1.45, 2.05],
    [0.38, 1.12, 1.62, 2.35],
];

struct BlockUnlockStats {
    k_sag_g0: [f64; 4],
    k_quad_g0: [f64; 4],
    f_scales_n: [f64; 5],
    f_scales_r: [f64; 5],
    f_scales_z: [f64; 5],
    lp4_diff_pair: bool,
    r_max: f64,
    prev_scale: f64,
    bn_max: f64,
    fast_core: bool,
    sym_fast: bool,
    optical_core: bool,
}

impl BlockUnlockStats {
    fn from_block(
        blk: &V3GruBlockF64,
        b: usize,
        fast_core: bool,
        sweeping_svf: bool,
        optical_core: bool,
        k_sag_g0: [f64; 4],
    ) -> Self {
        let k_quad_g0 = std::array::from_fn(|l| blk.k_quad[l].clamp(-0.4, 0.4));
        let f_scales_n = std::array::from_fn(|f| {
            (0..4)
                .map(|l| blk.w_in_n[f][l].abs())
                .fold(0.0_f64, f64::max)
                .clamp(0.0, 36.0)
        });
        let f_scales_r = std::array::from_fn(|f| {
            (0..4)
                .map(|l| blk.w_in_r[f][l].abs())
                .fold(0.0_f64, f64::max)
                .clamp(0.0, 36.0)
        });
        let f_scales_z = std::array::from_fn(|f| {
            (0..4)
                .map(|l| blk.w_in_z[f][l].abs())
                .fold(0.0_f64, f64::max)
                .clamp(0.0, 2.0)
        });
        let r_max = f_scales_r.iter().copied().fold(0.0_f64, f64::max);
        let lp4_diff_pair = blk.w_in_n[0][0].abs() > 0.1
            && (blk.w_in_n[0][0] + blk.w_in_n[1][0]).abs() < 0.15 * blk.w_in_n[0][0].abs();
        let mut prev_scale = 0.0_f64;
        if b > 0 {
            for d in 0..4 {
                for s in 0..4 {
                    prev_scale = prev_scale.max(blk.w_prev_n[d][s].abs());
                }
            }
            prev_scale = prev_scale.clamp(0.0, 4.5);
        }
        let bn_max = (0..4)
            .map(|l| blk.b_n[l].abs())
            .fold(0.0_f64, f64::max)
            .clamp(0.0, 1.5);
        let sym_fast = fast_core && r_max < 0.25 && (bn_max < 0.12 || sweeping_svf);
        Self {
            k_sag_g0,
            k_quad_g0,
            f_scales_n,
            f_scales_r,
            f_scales_z,
            lp4_diff_pair,
            r_max,
            prev_scale,
            bn_max,
            fast_core,
            sym_fast,
            optical_core,
        }
    }
}

fn zero_dst_cols(blk: &mut V3GruBlockF64, dst: usize) {
    for f in 0..NUM_IN_FEATS_B0 {
        blk.w_in_z[f][dst] = 0.0;
        blk.w_in_r[f][dst] = 0.0;
        blk.w_in_n[f][dst] = 0.0;
    }
    for src in 0..MAX_BLOCK_LANES {
        blk.w_prev_z[dst][src] = 0.0;
        blk.w_prev_r[dst][src] = 0.0;
        blk.w_prev_n[dst][src] = 0.0;
        blk.w_rec_z[dst][src] = 0.0;
        blk.w_rec_r[dst][src] = 0.0;
        blk.w_rec_n[dst][src] = 0.0;
    }
}

fn unlock_grid_leak_block(blk: &mut V3GruBlockF64, b: usize, k_sag_g0: &[f64; 4]) {
    const PAIR_SLOPES: [f64; 3] = [0.68, 1.38, 2.05];
    let drv_0 = blk.w_in_n[0][2];
    let w_rc0 = blk.w_rec_n[2][0];
    let w_rc1 = blk.w_rec_n[2][1];
    let kq_base = blk.k_quad[2];
    let bz_lp = blk.b_z[2];
    let bz_hp = blk.b_z[3];
    let b1_fmv_hi = blk.b_z[3];
    let gv1b_pos = blk.w_prev_n[0][2];
    let gv1b_neg = blk.w_prev_n[0][3];
    let bn1 = blk.b_n[0];
    let g0_prev: [[f64; 4]; 4] =
        std::array::from_fn(|l| std::array::from_fn(|s| blk.w_prev_n[l][s]));

    for dst in 4..(MAX_GROUPS_PER_BLOCK[b] * 4) {
        let g_idx = dst >> 2;
        let lane = dst & 3;
        let g_end = (g_idx + 1) << 2;
        let slope = GRID_LEAK_SLOPES[g_idx - 1][lane];
        let base_sign = if ((dst + b) & 1) == 0 { 1.0 } else { -1.0 };
        let alpha = slope * base_sign;
        zero_dst_cols(blk, dst);

        if b == 0 {
            if lane >= 2 {
                let sp = PAIR_SLOPES[g_idx - 1];
                blk.w_in_n[0][dst] = (sp * drv_0).clamp(-36.0, 36.0);
                blk.w_rec_n[dst][0] = sp * w_rc0;
                blk.w_rec_n[dst][1] = sp * w_rc1;
                blk.k_quad[dst] = (kq_base / sp).clamp(-0.42, 0.42);
                blk.b_n[dst] = 0.0;
                blk.b_z[dst] = if lane == 2 { bz_lp } else { bz_hp };
                blk.b_r[dst] = blk.b_z[dst];
            } else {
                blk.w_in_n[0][dst] = (alpha * drv_0).clamp(-36.0, 36.0);
                blk.w_rec_n[dst][0] = alpha * w_rc0;
                blk.w_rec_n[dst][1] = alpha * w_rc1;
                blk.k_quad[dst] = (kq_base / alpha).clamp(-0.42, 0.42);
                blk.b_n[dst] = 0.0;
                blk.b_z[dst] = [7.8, -0.85][lane];
                blk.b_r[dst] = blk.b_z[dst];
            }
            blk.k_sag[dst] = 0.0;
        } else if b == 1 {
            if lane >= 2 {
                // FMV interstage tone-stack filter lanes driven from Block 1 `V1B` lanes (`0, 1`).
                blk.w_rec_n[dst][0] = 0.85 * alpha;
                blk.w_rec_n[dst][1] = 0.55 * alpha;
                blk.b_n[dst] = 0.0;
                blk.b_z[dst] = if b1_fmv_hi > -1.8 {
                    if lane == 2 { -2.68 } else { b1_fmv_hi }
                } else if lane == 2 {
                    -4.40
                } else {
                    -2.68
                };
                blk.b_r[dst] = blk.b_z[dst];
                blk.k_sag[dst] = 0.0;
                blk.k_quad[dst] = 0.0;
            } else {
                blk.w_prev_n[dst][2] = alpha * gv1b_pos;
                blk.w_prev_n[dst][3] = alpha * gv1b_neg;
                for k in 1..=g_idx {
                    blk.w_prev_n[dst][4 * k + 2] = 0.08 * alpha * gv1b_pos;
                    blk.w_prev_n[dst][4 * k + 3] = 0.08 * alpha * gv1b_neg;
                }
                blk.b_n[dst] = (alpha * bn1).clamp(-1.2, 1.2);
                blk.b_z[dst] = [7.8, -4.39, -2.70, -1.28][lane];
                blk.b_r[dst] = blk.b_z[dst];
                blk.k_sag[dst] = 0.0;
                blk.k_quad[dst] = 0.0;
            }
        } else {
            for src in 0..4 {
                blk.w_prev_n[dst][src] = alpha * g0_prev[lane][src];
            }
            for src in 4..g_end {
                let sk = src >> 2;
                let sl = src & 3;
                let sb_b1 = if ((src + 1) & 1) == 0 { 1.0 } else { -1.0 };
                let ss_b1 = GRID_LEAK_SLOPES[sk - 1][sl];
                blk.w_prev_n[dst][src] = 0.08 * alpha * g0_prev[lane][sl] / (sb_b1 * ss_b1);
            }
            blk.b_z[dst] = [7.8, 7.6, 5.2, 1.8][lane];
            blk.b_r[dst] = blk.b_z[dst];
            blk.b_n[dst] = 0.0;
            blk.k_quad[dst] = 0.0;
            blk.k_sag[dst] = (k_sag_g0[lane] * (0.80 + 0.15 * (g_idx as f64))).clamp(0.0, 6.5);
        }
    }
}

fn unlock_standard_lane(blk: &mut V3GruBlockF64, b: usize, dst: usize, st: &BlockUnlockStats) {
    const FAST_SLOPES: [[f64; 4]; 3] = [
        [0.32, 0.72, 1.35, 2.10],
        [0.42, 0.90, 1.65, 2.55],
        [0.55, 1.15, 2.00, 3.10],
    ];
    const GEN_SLOPES: [[f64; 4]; 3] = [
        [0.28, 1.25, 2.55, 0.85],
        [0.18, 0.78, 3.45, 1.15],
        [0.38, 1.68, 4.65, 1.45],
    ];
    const GEN_BZ: [[f64; 4]; 3] = [
        [6.5, 1.2, -2.0, -4.8],
        [6.4, -0.2, -3.0, -5.4],
        [6.6, 1.8, -1.4, -4.4],
    ];
    const FAST_BZ: [f64; 4] = [6.8, 6.6, 6.4, 2.8];
    const BN_WIDE: [f64; 4] = [-1.15, -0.45, 0.45, 1.15];
    const BN_STAGGER: [f64; 4] = [-0.95, 0.35, -0.35, 0.95];

    let zero_dc_reset = st.optical_core && st.fast_core && st.r_max >= 0.25 && st.bn_max < 0.02;
    let phase = (dst as f64) * 1.17 + (b as f64) * 0.73;
    let g_idx = dst >> 2;
    let lane = dst & 3;
    let slope = if st.fast_core {
        FAST_SLOPES[g_idx - 1][lane]
    } else {
        GEN_SLOPES[g_idx - 1][lane]
    };
    let base_sign = if ((dst + b) & 1) == 0 { 1.0 } else { -1.0 };
    let z_gain = if lane == 3 { 0.25 } else { 0.15 };
    for f in 0..NUM_IN_FEATS_B0 {
        let pf = phase + (f as f64) * 0.91;
        let ns = st.f_scales_n[f];
        if st.sym_fast {
            let sgn = if f == 1 && st.lp4_diff_pair {
                -base_sign
            } else {
                base_sign
            };
            blk.w_in_n[f][dst] = (0.55 * ns * slope * sgn).clamp(-40.0, 40.0);
            blk.w_in_z[f][dst] = 0.0;
            blk.w_in_r[f][dst] = 0.0;
        } else {
            let zs = st.f_scales_z[f].max(0.15);
            let sgn = if f == 0 {
                base_sign
            } else if f == 1 && st.lp4_diff_pair {
                -base_sign
            } else if pf.sin() >= 0.0 {
                1.0
            } else {
                -1.0
            };
            blk.w_in_n[f][dst] = if zero_dc_reset && f >= 3 {
                0.0
            } else if zero_dc_reset && b == 0 && f == 0 {
                [1.15, -1.15, 1.95, -1.95][lane] * (0.75 + 0.25 * (g_idx as f64))
            } else if st.fast_core {
                (0.55 * ns * slope * sgn).clamp(-40.0, 40.0)
            } else {
                (0.55 * ns * slope * sgn).clamp(-36.0, 36.0)
            };
            let base_z = blk.w_in_z[f][lane];
            blk.w_in_z[f][dst] = if base_z.abs() > 1.0e-3 {
                (0.75 * slope * base_z).clamp(-2.0, 2.0)
            } else if zero_dc_reset {
                0.0
            } else {
                z_gain * zs * pf.cos()
            };
            let base_r = blk.w_in_r[f][lane];
            blk.w_in_r[f][dst] = if zero_dc_reset && f < 3 {
                0.0
            } else if st.r_max >= 0.25 && base_r.abs() > 1.0e-3 {
                (slope * base_r).clamp(-36.0, 36.0)
            } else if f >= 3 && st.f_scales_r[f] > 1.0e-3 {
                let r_sgn = if blk.w_in_r[f][lane] < 0.0 { -1.0 } else { 1.0 };
                (0.65 * slope * st.f_scales_r[f] * r_sgn).clamp(-36.0, 36.0)
            } else if zero_dc_reset {
                0.0
            } else {
                0.72 * z_gain * zs * pf.sin()
            };
        }
    }
    let g_end = (g_idx + 1) << 2;
    let ps_scale = st.prev_scale.max(0.35);
    for src in 0..g_end {
        let ps = phase + (src as f64) * 0.67;
        blk.w_rec_n[dst][src] = if st.fast_core || lane < 2 {
            0.0
        } else if src == dst {
            0.08
        } else if lane == 3 {
            0.06 * ps.cos()
        } else {
            0.0
        };
        if b > 0 {
            blk.w_prev_n[dst][src] = if zero_dc_reset && src < 4 {
                0.65 * slope * blk.w_prev_n[lane][src]
            } else if src < 4 {
                0.45 * ps_scale * slope * ps.sin()
            } else if src == dst {
                0.55
            } else {
                0.08 * ps.cos()
            };
        }
    }
    if st.sym_fast {
        blk.b_z[dst] = if blk.b_z[lane] < 0.0 {
            (blk.b_z[lane] + [-0.35, 0.0, 0.35][g_idx - 1]).clamp(-5.5, -0.15)
        } else {
            [7.4, 7.2, 7.0, 6.8][lane]
        };
        blk.b_r[dst] = blk.b_z[dst];
        blk.b_n[dst] = if st.bn_max < 0.02 {
            0.0
        } else {
            (0.5 * blk.b_n[lane] * base_sign).clamp(-0.15, 0.15)
        };
    } else if st.fast_core {
        blk.b_z[dst] = FAST_BZ[lane];
        blk.b_r[dst] = if st.r_max >= 0.25 {
            (blk.b_r[lane] + 0.15 * base_sign).clamp(-4.0, 4.0)
        } else {
            FAST_BZ[lane]
        };
        blk.b_n[dst] = if zero_dc_reset {
            0.0
        } else {
            (0.75 * st.bn_max.max(0.35) * BN_WIDE[lane]).clamp(-1.6, 1.6)
        };
    } else {
        if b == 0 && g_idx == 1 && lane == 0 {
            blk.b_z[dst] = -6.2;
            blk.b_r[dst] = -2.8;
        } else {
            blk.b_z[dst] = GEN_BZ[g_idx - 1][lane];
            blk.b_r[dst] = blk.b_z[dst];
        }
        blk.b_n[dst] = if st.bn_max >= 0.02 {
            (0.65 * st.bn_max * BN_STAGGER[lane]).clamp(-1.2, 1.2)
        } else {
            0.0
        };
    }
    blk.k_sag[dst] = if zero_dc_reset {
        0.0
    } else {
        (st.k_sag_g0[lane] * (0.75 + 0.15 * (g_idx as f64))).clamp(0.0, 6.5)
    };
    blk.k_quad[dst] = if st.sym_fast {
        0.0
    } else {
        (0.55 * st.k_quad_g0[lane] * base_sign).clamp(-0.35, 0.35)
    };
}

fn has_prev_reset_vca(m: &V3MatryoshkaOptF64) -> bool {
    (1..NUM_GRU_BLOCKS).any(|b| {
        (0..4).any(|dst| {
            (0..4).any(|src| m.blocks[b].w_prev_r[dst][src].abs() > 0.5)
        })
    })
}

fn unlock_split_or_vca_lane(blk: &mut V3GruBlockF64, b: usize, dst: usize) {
    const SPLIT_SLOPES: [f64; 3] = [0.68, 1.28, 1.85];
    const LPF_SLOPES: [f64; 3] = [0.75, 1.15, 1.55];
    const DETUNE_OFF: [f64; 3] = [-0.45, 0.0, 0.45];

    let g_idx = dst >> 2;
    let src_l = dst & 3;
    zero_dst_cols(blk, dst);

    let is_ballistic = blk.b_n[src_l].abs() > 2.0 && blk.b_z[src_l] < -3.5;
    let is_state_lpf =
        !is_ballistic && blk.b_z[src_l] < 2.0 && (blk.b_z[src_l] - blk.b_r[src_l]).abs() < 0.35;

    if is_ballistic {
        let pol = if (g_idx & 1) == 1 { -1.0 } else { 1.0 };
        blk.b_z[dst] = (blk.b_z[src_l] + DETUNE_OFF[g_idx - 1]).clamp(-7.5, -2.5);
        blk.b_r[dst] = blk.b_r[src_l];
        blk.b_n[dst] = blk.b_n[src_l];
        for f in 0..NUM_IN_FEATS_B0 {
            blk.w_in_r[f][dst] = (pol * blk.w_in_r[f][src_l]).clamp(-60.0, 60.0);
            blk.w_in_n[f][dst] = (pol * blk.w_in_n[f][src_l]).clamp(-36.0, 36.0);
        }
        for s in 0..4 {
            blk.w_rec_r[dst][s] = (pol * blk.w_rec_r[src_l][s]).clamp(-60.0, 60.0);
            blk.w_rec_n[dst][s] = (pol * blk.w_rec_n[src_l][s]).clamp(-36.0, 36.0);
            if b > 0 {
                blk.w_prev_r[dst][s] = (pol * blk.w_prev_r[src_l][s]).clamp(-60.0, 60.0);
                blk.w_prev_n[dst][s] = (pol * blk.w_prev_n[src_l][s]).clamp(-36.0, 36.0);
            }
        }
        blk.k_sag[dst] = 0.0;
        blk.k_quad[dst] = 0.0;
    } else if is_state_lpf {
        let scale = LPF_SLOPES[g_idx - 1];
        let alt = if ((dst + b) & 1) == 0 { 1.0 } else { -1.0 };
        blk.b_z[dst] = (blk.b_z[src_l] + 0.78 * DETUNE_OFF[g_idx - 1]).clamp(-7.5, 2.0);
        blk.b_r[dst] = blk.b_z[dst];
        blk.b_n[dst] = (0.5 * blk.b_n[src_l] * alt).clamp(-1.5, 1.5);
        for f in 0..NUM_IN_FEATS_B0 {
            blk.w_in_n[f][dst] = (scale * alt * blk.w_in_n[f][src_l]).clamp(-36.0, 36.0);
        }
        for s in 0..4 {
            blk.w_rec_n[dst][s] = (scale * alt * blk.w_rec_n[src_l][s]).clamp(-36.0, 36.0);
            if b > 0 {
                blk.w_prev_n[dst][s] = (scale * alt * blk.w_prev_n[src_l][s]).clamp(-36.0, 36.0);
            }
        }
        blk.k_sag[dst] = 0.0;
        blk.k_quad[dst] = 0.0;
    } else {
        let has_vca_reset = (0..4)
            .any(|s| blk.w_prev_r[src_l][s].abs() > 0.5 || blk.w_rec_r[src_l][s].abs() > 0.5);
        let scale = SPLIT_SLOPES[g_idx - 1];
        let alt = if has_vca_reset || ((dst + b) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        for f in 0..NUM_IN_FEATS_B0 {
            blk.w_in_n[f][dst] = (scale * alt * blk.w_in_n[f][src_l]).clamp(-36.0, 36.0);
        }
        for s in 0..4 {
            blk.w_rec_n[dst][s] = (scale * alt * blk.w_rec_n[src_l][s]).clamp(-36.0, 36.0);
            blk.w_rec_r[dst][s] = (0.85 * blk.w_rec_r[src_l][s]).clamp(-5.5, 5.5);
            if b > 0 {
                blk.w_prev_n[dst][s] = (scale * alt * blk.w_prev_n[src_l][s]).clamp(-36.0, 36.0);
                blk.w_prev_r[dst][s] = (0.85 * blk.w_prev_r[src_l][s]).clamp(-5.5, 5.5);
            }
        }
        blk.b_z[dst] = blk.b_z[src_l];
        blk.b_r[dst] = blk.b_r[src_l];
        blk.b_n[dst] = (0.75 * blk.b_n[src_l] * alt).clamp(-1.5, 1.5);
        blk.k_sag[dst] = blk.k_sag[src_l].clamp(0.0, 6.5);
        blk.k_quad[dst] = (0.75 * blk.k_quad[src_l]).clamp(-0.35, 0.35);
    }
}

/// Unlocks all `4` Matryoshka groups (`lanes 0..16`) with timescale regime detection while preserving `Lite`.
pub fn unlock_matryoshka_groups_after_prescreen(m: &mut V3MatryoshkaOptF64) {
    let fast_core = (0..3).all(|l| m.blocks[0].b_z[l] > 4.5);
    let grid_leak_core = m.blocks[0].b_z[0] < -4.5 && m.blocks[0].b_z[1] < -4.5;
    let split_or_vca_core = (m.svf_feat_lp2_mix > 0.5 || has_prev_reset_vca(m)) && !grid_leak_core;
    let sweeping_svf = m.svf_env_oct_sens.abs() > 0.5;
    let optical_core = (m.mono_out && m.feat_key_link > 0.0) || m.feat_det_thresh > 0.0;

    m.max_active_groups = MAX_GROUPS;
    m.active_tier_weights = [0.22, 0.23, 0.26, 0.29];
    let lite_r = m.readouts[0].clone();
    for tau in 1..4 {
        m.readouts[tau] = lite_r.clone();
    }
    for b in 0..NUM_GRU_BLOCKS {
        let active_lanes = MAX_GROUPS_PER_BLOCK[b] * 4;
        let blk = &mut m.blocks[b];
        let k_sag_g0: [f64; 4] = std::array::from_fn(|l| blk.k_sag[l].clamp(0.0, 5.0));
        if grid_leak_core {
            unlock_grid_leak_block(blk, b, &k_sag_g0);
        } else if split_or_vca_core {
            for dst in 4..active_lanes {
                unlock_split_or_vca_lane(blk, b, dst);
            }
        } else {
            let st = BlockUnlockStats::from_block(
                blk,
                b,
                fast_core,
                sweeping_svf,
                optical_core,
                k_sag_g0,
            );
            for dst in 4..active_lanes {
                unlock_standard_lane(blk, b, dst, &st);
            }
        }
        for dst in active_lanes..MAX_BLOCK_LANES {
            zero_dst_cols(blk, dst);
            blk.b_z[dst] = 0.0;
            blk.b_r[dst] = 0.0;
            blk.b_n[dst] = 0.0;
            blk.k_sag[dst] = 0.0;
            blk.k_quad[dst] = 0.0;
        }
        blk.enforce_lower_triangular();
    }
    m.mirror_mono();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::v3_varpro::seed_init::{SeedArchetype, build_matryoshka_seed_bank_tagged};

    #[test]
    fn test_unlock_covers_grid_leak_fast_and_general_regimes() {
        let tagged = build_matryoshka_seed_bank_tagged(false);
        let mut seen_regimes = [false; 3];
        for (arch, mut model) in tagged {
            unlock_matryoshka_groups_after_prescreen(&mut model);
            assert_eq!(model.max_active_groups, MAX_GROUPS);
            assert!(model.active_tier_weights.iter().all(|&w| w > 0.0));
            match arch {
                SeedArchetype::SeriesTubeSag => seen_regimes[0] = true,
                SeedArchetype::EnvResonantSvf => seen_regimes[1] = true,
                SeedArchetype::GeneralDrive => seen_regimes[2] = true,
                SeedArchetype::PinchBiasClipper => {}
            }
            for b in 0..NUM_GRU_BLOCKS {
                for dst in 4..(MAX_GROUPS_PER_BLOCK[b] * 4) {
                    assert!(model.blocks[b].b_z[dst].is_finite());
                    assert!(model.blocks[b].k_sag[dst].is_finite());
                }
            }
        }
        assert!(seen_regimes.into_iter().all(|v| v));

        let mut split_model = V3MatryoshkaOptF64::from_matryoshka_cold(
            &dnam_engine::stages::V3MatryoshkaModel::dual_rate_default(true),
        );
        split_model.svf_feat_lp2_mix = 0.9995;
        split_model.blocks[0].b_z[..4].copy_from_slice(&[-1.8, -6.2, -5.8, 7.4]);
        split_model.blocks[0].b_r[..4].copy_from_slice(&[-1.8, -3.0, -3.0, 7.4]);
        split_model.blocks[0].b_n[..4].copy_from_slice(&[0.0, 3.0, 3.0, 0.1]);
        split_model.blocks[0].w_in_r[0][1] = 48.0;
        split_model.blocks[2].w_prev_r[0][0] = -4.2;
        unlock_matryoshka_groups_after_prescreen(&mut split_model);
        assert_eq!(split_model.max_active_groups, MAX_GROUPS);
        assert!(split_model.blocks[0].b_n[5] > 2.0);
        assert!(split_model.blocks[0].w_in_r[0][5] < -40.0);
    }
}
