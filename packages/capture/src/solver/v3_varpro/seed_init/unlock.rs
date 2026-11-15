#![forbid(unsafe_code)]
//! Matryoshka group unlocking (`lanes 0..16`) after initial `Lite` prescreen.

use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
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
    f_scales_z: [f64; 5],
    prev_scale: f64,
    bn_max: f64,
    fast_core: bool,
    sym_fast: bool,
}

impl BlockUnlockStats {
    fn from_block(
        blk: &V3GruBlockF64,
        b: usize,
        fast_core: bool,
        sweeping_svf: bool,
        k_sag_g0: [f64; 4],
    ) -> Self {
        let k_quad_g0 = std::array::from_fn(|l| blk.k_quad[l].clamp(-0.4, 0.4));
        let f_scales_n = std::array::from_fn(|f| {
            (0..4)
                .map(|l| blk.w_in_n[f][l].abs())
                .fold(0.0_f64, f64::max)
                .clamp(0.0, 36.0)
        });
        let f_scales_z = std::array::from_fn(|f| {
            (0..4)
                .map(|l| blk.w_in_z[f][l].abs())
                .fold(0.0_f64, f64::max)
                .clamp(0.0, 2.0)
        });
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
        let sym_fast = fast_core && (bn_max < 0.12 || sweeping_svf);
        Self {
            k_sag_g0,
            k_quad_g0,
            f_scales_n,
            f_scales_z,
            prev_scale,
            bn_max,
            fast_core,
            sym_fast,
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
    let gv1b_pos = blk.w_prev_n[0][2];
    let gv1b_neg = blk.w_prev_n[0][3];
    let bn1 = blk.b_n[0];
    let g0_prev: [[f64; 4]; 4] =
        std::array::from_fn(|l| std::array::from_fn(|s| blk.w_prev_n[l][s]));

    for dst in 4..MAX_BLOCK_LANES {
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
                blk.b_z[dst] = if lane == 2 { -1.044 } else { -4.772 };
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
            blk.w_in_n[f][dst] = (0.55 * ns * slope * base_sign).clamp(-40.0, 40.0);
            blk.w_in_z[f][dst] = 0.0;
            blk.w_in_r[f][dst] = 0.0;
        } else {
            let zs = st.f_scales_z[f].max(0.15);
            let sgn = if f == 0 {
                base_sign
            } else if pf.sin() >= 0.0 {
                1.0
            } else {
                -1.0
            };
            blk.w_in_n[f][dst] = if st.fast_core {
                (0.55 * ns * slope * sgn).clamp(-40.0, 40.0)
            } else {
                (0.55 * ns * slope * sgn).clamp(-36.0, 36.0)
            };
            blk.w_in_z[f][dst] = z_gain * zs * pf.cos();
            blk.w_in_r[f][dst] = 0.72 * z_gain * zs * pf.sin();
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
            blk.w_prev_n[dst][src] = if src < 4 {
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
        blk.b_r[dst] = FAST_BZ[lane];
        blk.b_n[dst] = (0.75 * st.bn_max.max(0.35) * BN_WIDE[lane]).clamp(-1.6, 1.6);
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
    blk.k_sag[dst] = (st.k_sag_g0[lane] * (0.75 + 0.15 * (g_idx as f64))).clamp(0.0, 6.5);
    blk.k_quad[dst] = if st.sym_fast {
        0.0
    } else {
        (0.55 * st.k_quad_g0[lane] * base_sign).clamp(-0.35, 0.35)
    };
}

/// Unlocks all `4` Matryoshka groups (`lanes 0..16`) with timescale regime detection while preserving `Lite`.
pub fn unlock_matryoshka_groups_after_prescreen(m: &mut V3MatryoshkaOptF64) {
    let fast_core = (0..3).all(|l| m.blocks[0].b_z[l] > 4.5);
    let grid_leak_core = m.blocks[0].b_z[0] < -4.5 && m.blocks[0].b_z[1] < -4.5;
    let sweeping_svf = m.svf_env_oct_sens.abs() > 0.5;

    m.max_active_groups = MAX_GROUPS;
    m.active_tier_weights = [0.22, 0.23, 0.26, 0.29];
    let lite_r = m.readouts[0].clone();
    for tau in 1..4 {
        m.readouts[tau] = lite_r.clone();
    }
    for b in 0..NUM_GRU_BLOCKS {
        let blk = &mut m.blocks[b];
        let k_sag_g0: [f64; 4] = std::array::from_fn(|l| blk.k_sag[l].clamp(0.0, 5.0));
        if grid_leak_core {
            unlock_grid_leak_block(blk, b, &k_sag_g0);
        } else {
            let st = BlockUnlockStats::from_block(blk, b, fast_core, sweeping_svf, k_sag_g0);
            for dst in 4..MAX_BLOCK_LANES {
                unlock_standard_lane(blk, b, dst, &st);
            }
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
                for dst in 4..MAX_BLOCK_LANES {
                    assert!(model.blocks[b].b_z[dst].is_finite());
                    assert!(model.blocks[b].k_sag[dst].is_finite());
                }
            }
        }
        assert!(seen_regimes.into_iter().all(|v| v));
    }
}
