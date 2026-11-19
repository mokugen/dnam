//! Fused Stage II Block 2 recurrence + linear readout sub-block kernel (`G2 = 1`).

use super::prepass::{PrePassIn, fill_pre_with_opt_tile};
use super::{V3KernelScratch, eval_diag_rec_step};
use crate::math::{MathBackend, SimdVec4};
use crate::stages::hot_core::V3HotCore;
use crate::stages::stage2_gru::block::MAX_GROUPS;

/// Evaluates Block 2 (`G2 = 1`) fused directly with Stage II linear readout into `sc.s2_sub`.
#[inline(never)]
pub(in super::super) fn run_block2_and_readout<
    B: MathBackend,
    const G0: usize,
    const G1: usize,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    len: usize,
) {
    const { assert!(G0 >= 1 && G0 <= CAP_G0 && CAP_G0 <= MAX_GROUPS) };
    const { assert!(G1 >= 1 && G1 <= CAP_G1 && CAP_G1 <= 2) };
    let n = len.min(SUB_CAP);
    if n == 0 {
        return;
    }

    let blk = &core.blocks[2];
    let film2 = &core.ctrl_bus.block_film_gain[2][0];
    let bn2 = &core.ctrl_bus.block_bn_eff[2][0];
    let tanh_bn2 = &core.ctrl_bus.block_tanh_bn_eff[2][0];
    let cab_inj = core.tone_svf.params.cab_inject_w;

    let inp2 = PrePassIn {
        blk,
        b1_inj: 0.0,
        f03: &sc.f03,
        f4: &sc.f4,
        tone_out: &sc.tone_out,
        h_src: &sc.h_hist1[0][1..=n],
        n,
    };
    fill_pre_with_opt_tile::<B, 0, false, true, SUB_CAP>(
        &inp2,
        &blk.w_prev_z,
        &blk.w_prev_r,
        &blk.w_prev_n,
        0,
        &mut sc.pre_zrn,
    );

    let pw0_l: [B::Vec4; G0] = core::array::from_fn(|g| B::Vec4::load(&core.pre_cab_w_l[0][g]));
    let pw0_r: [B::Vec4; G0] = core::array::from_fn(|g| B::Vec4::load(&core.pre_cab_w_r[0][g]));
    let dw0_l: [B::Vec4; G0] = core::array::from_fn(|g| B::Vec4::load(&core.direct_s2_w_l[0][g]));
    let dw0_r: [B::Vec4; G0] = core::array::from_fn(|g| B::Vec4::load(&core.direct_s2_w_r[0][g]));

    let pw1_l: [B::Vec4; G1] = core::array::from_fn(|g| B::Vec4::load(&core.pre_cab_w_l[1][g]));
    let pw1_r: [B::Vec4; G1] = core::array::from_fn(|g| B::Vec4::load(&core.pre_cab_w_r[1][g]));
    let dw1_l: [B::Vec4; G1] = core::array::from_fn(|g| B::Vec4::load(&core.direct_s2_w_l[1][g]));
    let dw1_r: [B::Vec4; G1] = core::array::from_fn(|g| B::Vec4::load(&core.direct_s2_w_r[1][g]));

    let pw2_l = B::Vec4::load(&core.pre_cab_w_l[2][0]);
    let pw2_r = B::Vec4::load(&core.pre_cab_w_r[2][0]);
    let dw2_l = B::Vec4::load(&core.direct_s2_w_l[2][0]);
    let dw2_r = B::Vec4::load(&core.direct_s2_w_r[2][0]);

    let mut hl2 = B::Vec4::load(&core.state_l.h[2][0]);
    let mut hr2 = B::Vec4::load(&core.state_r.h[2][0]);

    for i in 0..n {
        eval_diag_rec_step::<B, 0>(
            blk,
            film2,
            bn2,
            tanh_bn2,
            &sc.pre_zrn[i],
            &sc.f4[i],
            &mut hl2,
            &mut hr2,
        );

        let mut pl = pw2_l.mul(hl2);
        let mut pr = pw2_r.mul(hr2);
        let mut dl = dw2_l.mul(hl2);
        let mut dr = dw2_r.mul(hr2);

        for g in 0..G0 {
            let h0l = B::Vec4::load(&sc.h_hist0[g][i + 1][0]);
            let h0r = B::Vec4::load(&sc.h_hist0[g][i + 1][1]);
            pl = pl.fma(pw0_l[g], h0l);
            pr = pr.fma(pw0_r[g], h0r);
            dl = dl.fma(dw0_l[g], h0l);
            dr = dr.fma(dw0_r[g], h0r);
        }
        for g in 0..G1 {
            let h1l = B::Vec4::load(&sc.h_hist1[g][i + 1][0]);
            let h1r = B::Vec4::load(&sc.h_hist1[g][i + 1][1]);
            pl = pl.fma(pw1_l[g], h1l);
            pr = pr.fma(pw1_r[g], h1r);
            dl = dl.fma(dw1_l[g], h1l);
            dr = dr.fma(dw1_r[g], h1r);
        }

        let (ps_l, ps_r) = pl.horizontal_sum_pair(pr);
        let (ds_l, ds_r) = dl.horizontal_sum_pair(dr);
        let [ul, ur] = sc.u_s2[i];
        let [hpl, hpr] = sc.u_hp[i];
        let [tl, tr] = sc.tone_out[i];
        let [svf_dl, svf_dr] = sc.svf_dir[i];
        sc.s2_sub[i] = [
            core.pre_cab_u_gain[0] * ul + core.pre_cab_hp_gain[0] * hpl + cab_inj * tl + ps_l,
            core.pre_cab_u_gain[1] * ur + core.pre_cab_hp_gain[1] * hpr + cab_inj * tr + ps_r,
            core.direct_s2_u_gain[0] * ul + core.direct_s2_hp_gain[0] * hpl + svf_dl + ds_l,
            core.direct_s2_u_gain[1] * ur + core.direct_s2_hp_gain[1] * hpr + svf_dr + ds_r,
        ];
    }

    hl2.store(&mut core.state_l.h[2][0]);
    hr2.store(&mut core.state_r.h[2][0]);
}

#[cfg(test)]
mod tests {
    use super::super::{
        run_block0_subblock, run_block1_subblock, run_input_features, run_interblock_tone,
    };
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::stages::stage2_gru::block::{
        MAX_BLOCK_LANES, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, tier_block_groups,
    };
    use crate::stages::stage3_cab::V3MatryoshkaModel;
    use crate::types::Tier;

    fn run_group_outer_parity_for_tier<const G0: usize, const G1: usize>(tier: Tier) {
        let mut mat = V3MatryoshkaModel::dual_rate_default(false);
        for b in 0..NUM_GRU_BLOCKS {
            for r in 0..5 {
                for c in 0..MAX_BLOCK_LANES {
                    mat.blocks[b].w_in_n[r][c] = 0.08 * (((r + c) % 5) as f32 - 2.0);
                    mat.blocks[b].w_in_z[r][c] = 0.05 * (((r * 2 + c) % 3) as f32 - 1.0);
                    mat.blocks[b].w_in_r[r][c] = -0.04 * (((r + c * 3) % 3) as f32 - 1.0);
                }
            }
            for r in 0..MAX_BLOCK_LANES {
                for c in 0..=r {
                    mat.blocks[b].w_rec_n[r][c] = 0.06 * (((r + c) % 4) as f32 - 1.5);
                    mat.blocks[b].w_rec_z[r][c] = 0.03 * (((r + 2 * c) % 3) as f32 - 1.0);
                    mat.blocks[b].w_rec_r[r][c] = 0.03 * (((2 * r + c) % 3) as f32 - 1.0);
                    mat.blocks[b].w_prev_n[r][c] = 0.05 * (((r + c) % 3) as f32 - 1.0);
                    mat.blocks[b].w_prev_z[r][c] = 0.04;
                    mat.blocks[b].w_prev_r[r][c] = -0.03;
                }
                mat.blocks[b].b_n[r] = 0.12 - 0.02 * (r as f32);
                mat.blocks[b].b_z[r] = -0.15;
                mat.blocks[b].b_r[r] = 0.1;
                mat.blocks[b].k_sag[r] = 0.2;
            }
        }
        mat.tone_svf.h0_proj_w = [0.2, -0.1, 0.05, 0.05];
        mat.tone_svf.b1_inject_w = 0.25;
        mat.tone_svf.cab_inject_w = 0.15;
        let t_idx = tier as usize;
        for l in 0..MAX_TOTAL_GRU_LANES {
            mat.readouts[t_idx].pre_cab_w[0][l] = 0.04 * (((l % 5) as f32) - 2.0);
            mat.readouts[t_idx].pre_cab_w[1][l] = -0.03 * (((l % 3) as f32) - 1.0);
            mat.readouts[t_idx].direct_s2_w[0][l] = 0.02;
            mat.readouts[t_idx].direct_s2_w[1][l] = -0.02;
        }
        let mut cold = mat.extract_tier(tier);
        cold.os2x = false;
        let mut core_sub = V3HotCore::<DefaultMathBackend>::from_cold_model(&cold);
        let mut core_ref = core_sub.clone();

        let mut sc_sub = V3KernelScratch::<3, 2, 16, 17>::zeroed();
        let mut sc_ref = V3KernelScratch::<3, 2, 16, 17>::zeroed();
        for i in 0..16 {
            let t = i as f32;
            sc_sub.u_s2[i] = [0.35 * (t * 0.23).sin(), -0.28 * (t * 0.19).cos()];
            sc_sub.svf_dir[i] = [0.02 * t.sin(), -0.015 * t.cos()];
        }

        run_input_features::<DefaultMathBackend, 3, 2, 16, 17>(&mut core_sub, &mut sc_sub, 16);
        run_block0_subblock::<DefaultMathBackend, G0, 3, 2, 16, 17>(&mut core_sub, &mut sc_sub, 16);
        run_interblock_tone::<DefaultMathBackend, 3, 2, 16, 17>(&mut core_sub, &mut sc_sub, 16);
        run_block1_subblock::<DefaultMathBackend, G1, 3, 2, 16, 17>(&mut core_sub, &mut sc_sub, 16);
        run_block2_and_readout::<DefaultMathBackend, G0, G1, 3, 2, 16, 17>(
            &mut core_sub,
            &mut sc_sub,
            16,
        );

        for i in 0..16 {
            sc_ref.u_s2[0] = sc_sub.u_s2[i];
            sc_ref.svf_dir[0] = sc_sub.svf_dir[i];
            run_input_features::<DefaultMathBackend, 3, 2, 16, 17>(&mut core_ref, &mut sc_ref, 1);
            run_block0_subblock::<DefaultMathBackend, G0, 3, 2, 16, 17>(
                &mut core_ref,
                &mut sc_ref,
                1,
            );
            run_interblock_tone::<DefaultMathBackend, 3, 2, 16, 17>(&mut core_ref, &mut sc_ref, 1);
            run_block1_subblock::<DefaultMathBackend, G1, 3, 2, 16, 17>(
                &mut core_ref,
                &mut sc_ref,
                1,
            );
            run_block2_and_readout::<DefaultMathBackend, G0, G1, 3, 2, 16, 17>(
                &mut core_ref,
                &mut sc_ref,
                1,
            );
            for c in 0..4 {
                assert!((sc_sub.s2_sub[i][c] - sc_ref.s2_sub[0][c]).abs() < 1.0e-6);
            }
        }
        for b in 0..NUM_GRU_BLOCKS {
            for g in 0..tier_block_groups(tier, b) {
                for l in 0..4 {
                    assert!(
                        (core_sub.state_l.h[b][g][l] - core_ref.state_l.h[b][g][l]).abs() < 1.0e-6
                    );
                    assert!(
                        (core_sub.state_r.h[b][g][l] - core_ref.state_r.h[b][g][l]).abs() < 1.0e-6
                    );
                }
            }
        }
        assert!(sc_sub.s2_sub[..16].iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn test_stage2_group_outer_parity_all_groups() {
        run_group_outer_parity_for_tier::<1, 1>(Tier::Lite);
        run_group_outer_parity_for_tier::<2, 1>(Tier::Full);
        run_group_outer_parity_for_tier::<2, 2>(Tier::Pro);
        run_group_outer_parity_for_tier::<3, 2>(Tier::Ultra);
    }
}
