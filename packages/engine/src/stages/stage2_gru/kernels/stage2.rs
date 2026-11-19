//! Register-resident tapered 3-block Stage II sub-block execution kernels (`SUB_CAP = 16` or `32`).

mod prepass;
mod readout;

pub(super) use readout::run_block2_and_readout;

use crate::math::{MathBackend, SimdVec4, flush_recursive_state_vec4};
use crate::stages::hot_core::V3HotCore;
use crate::stages::stage2_gru::block::{MAX_GROUPS, V3GruBlockHot, tri_tile_index};
use crate::stages::stage2_gru::features::V3InputFeaturesState;
use crate::stages::stage2_gru::math_pade::tanh_pade5_6_with_rcp2_vec4;
use crate::stages::stage3_cab::Stage3SubblockScratch;
use prepass::{
    PrePassIn, accum_prev_and_rec_pair_to_pre, accum_tile_group_to_pre, fill_pre_with_opt_tile,
    fma4_stereo_tree, load_tile_cols,
};

/// Stack-allocated 64-byte aligned sub-block scratch buffers sized to the active tier's group
/// counts `(CAP_G0, CAP_G1)` and oversampling cadence `(SUB_CAP, HIST_CAP)`.
///
/// Stores only Block 0 and Block 1 trajectories (`h_hist0`, `h_hist1`); Block 2 (`G2 = 1`) fuses its
/// recurrence directly with Stage II readout in registers to avoid a third history buffer.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub(crate) struct V3KernelScratch<
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
> {
    pub stage1_in: [[f32; 2]; SUB_CAP],
    pub svf_env: [[f32; 2]; SUB_CAP],
    pub svf_a: [[[f32; 4]; 3]; SUB_CAP],
    pub u_s2: [[f32; 2]; SUB_CAP],
    pub svf_dir: [[f32; 2]; SUB_CAP],
    pub svf_lp2: [[f32; 2]; SUB_CAP],
    pub clean_sub: [[f32; 2]; 16],
    pub s2_dir: [[f32; 4]; 16],
    pub f03: [[[f32; 4]; 2]; SUB_CAP],
    pub f4: [[[f32; 4]; 2]; SUB_CAP],
    pub u_hp: [[f32; 2]; SUB_CAP],
    pub tone_out: [[f32; 2]; SUB_CAP],
    /// Block 0 group trajectories (`t = 0` holds incoming `h[-1]`, `1..=n` hold `h[0..n)`).
    pub h_hist0: [[[[f32; 4]; 2]; HIST_CAP]; CAP_G0],
    /// Block 1 group trajectories (`G1 <= 2`, `t = 0` holds incoming `h[-1]`).
    pub h_hist1: [[[[f32; 4]; 2]; HIST_CAP]; CAP_G1],
    /// Single-group pre-pass buffer reused across blocks/groups (`1.5 KiB` 1x / `3 KiB` 2x).
    pub pre_zrn: [[[[f32; 4]; 2]; 3]; SUB_CAP],
    pub s2_sub: [[f32; 4]; SUB_CAP],
    pub stage3: Stage3SubblockScratch<16>,
}

impl<const CAP_G0: usize, const CAP_G1: usize, const SUB_CAP: usize, const HIST_CAP: usize>
    V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>
{
    /// Initializes callback-local sub-block scratch (`HIST_CAP` must equal `SUB_CAP + 1`).
    pub(crate) const fn zeroed() -> Self {
        const { assert!(CAP_G0 >= 1 && CAP_G0 <= MAX_GROUPS) };
        const { assert!(CAP_G1 >= 1 && CAP_G1 <= 2) };
        const { assert!(HIST_CAP == SUB_CAP + 1) };
        Self {
            stage1_in: [[0.0; 2]; SUB_CAP],
            svf_env: [[0.0; 2]; SUB_CAP],
            svf_a: [[[0.0; 4]; 3]; SUB_CAP],
            u_s2: [[0.0; 2]; SUB_CAP],
            svf_dir: [[0.0; 2]; SUB_CAP],
            svf_lp2: [[0.0; 2]; SUB_CAP],
            clean_sub: [[0.0; 2]; 16],
            s2_dir: [[0.0; 4]; 16],
            f03: [[[0.0; 4]; 2]; SUB_CAP],
            f4: [[[0.0; 4]; 2]; SUB_CAP],
            u_hp: [[0.0; 2]; SUB_CAP],
            tone_out: [[0.0; 2]; SUB_CAP],
            h_hist0: [[[[0.0; 4]; 2]; HIST_CAP]; CAP_G0],
            h_hist1: [[[[0.0; 4]; 2]; HIST_CAP]; CAP_G1],
            pre_zrn: [[[[0.0; 4]; 2]; 3]; SUB_CAP],
            s2_sub: [[0.0; 4]; SUB_CAP],
            stage3: Stage3SubblockScratch::zeroed(),
        }
    }
}

/// Evaluates one diagonal-group GRU step inline with tiles loaded at point of use (zero stack spills).
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Inline diagonal step keeps hl/hr and L1D tile accesses spill-free"
)]
pub(super) fn eval_diag_rec_step<B: MathBackend, const DST_G: usize>(
    blk: &V3GruBlockHot,
    film_g: &[f32; 4],
    bn_eff: &[f32; 4],
    tanh_bn: &[f32; 4],
    pre_i: &[[[f32; 4]; 2]; 3],
    f4_i: &[[f32; 4]; 2],
    hl: &mut B::Vec4,
    hr: &mut B::Vec4,
) {
    const { assert!(DST_G < MAX_GROUPS) };
    let diag_tile = const { tri_tile_index(DST_G, DST_G) };
    let one = B::Vec4::splat(1.0);
    let half = B::Vec4::splat(0.5);
    let lo12 = B::Vec4::splat(-12.0);
    let hi12 = B::Vec4::splat(12.0);
    let lo25 = B::Vec4::splat(-2.5);
    let hi25 = B::Vec4::splat(2.5);

    let (an_l, an_r) = fma4_stereo_tree(
        B::Vec4::load(&pre_i[2][0]),
        B::Vec4::load(&pre_i[2][1]),
        load_tile_cols::<B::Vec4>(&blk.w_rec_n[diag_tile]),
        *hl,
        *hr,
    );
    let f4_l = B::Vec4::load(&f4_i[0]);
    let f4_r = B::Vec4::load(&f4_i[1]);
    let ks = B::Vec4::load(&blk.groups[DST_G].k_sag);
    let g_sag_l = one.fma(ks, f4_l);
    let g_sag_r = one.fma(ks, f4_r);
    let a_sig_l = an_l.clamp(lo12, hi12);
    let a_sig_r = an_r.clamp(lo12, hi12);
    let kq = B::Vec4::load(&blk.k_quad[DST_G]);
    let film = B::Vec4::load(film_g);
    let a_eff_l = a_sig_l.fma(a_sig_l.mul(a_sig_l), kq).mul(film);
    let a_eff_r = a_sig_r.fma(a_sig_r.mul(a_sig_r), kq).mul(film);
    let bn_v = B::Vec4::load(bn_eff);
    let th_arg_l = bn_v.fma(g_sag_l, a_eff_l);
    let th_arg_r = bn_v.fma(g_sag_r, a_eff_r);

    let (az_l, az_r) = fma4_stereo_tree(
        B::Vec4::load(&pre_i[0][0]),
        B::Vec4::load(&pre_i[0][1]),
        load_tile_cols::<B::Vec4>(&blk.w_rec_z[diag_tile]),
        *hl,
        *hr,
    );
    let (ar_l, ar_r) = fma4_stereo_tree(
        B::Vec4::load(&pre_i[1][0]),
        B::Vec4::load(&pre_i[1][1]),
        load_tile_cols::<B::Vec4>(&blk.w_rec_r[diag_tile]),
        *hl,
        *hr,
    );

    let ([tz_l, tz_r, tr_l, tr_r, th_l, th_r], inv_sag_l, inv_sag_r) =
        tanh_pade5_6_with_rcp2_vec4::<B>(
            [
                az_l.mul(half),
                az_r.mul(half),
                ar_l.mul(half),
                ar_r.mul(half),
                th_arg_l,
                th_arg_r,
            ],
            g_sag_l,
            g_sag_r,
        );

    let z_l = half.fma(tz_l, half);
    let z_r = half.fma(tz_r, half);
    let r_l = half.fma(tr_l, half);
    let r_r = half.fma(tr_r, half);
    let tanh_bn_v = B::Vec4::load(tanh_bn);
    let nv_l = inv_sag_l.mul(th_l.sub(tanh_bn_v));
    let nv_r = inv_sag_r.mul(th_r.sub(tanh_bn_v));
    *hl = flush_recursive_state_vec4::<B::Vec4>(one.sub(z_l).mul(*hl).fma(r_l, nv_l))
        .clamp(lo25, hi25);
    *hr = flush_recursive_state_vec4::<B::Vec4>(one.sub(z_r).mul(*hr).fma(r_r, nv_r))
        .clamp(lo25, hi25);
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Explicit sub-block group slices eliminate alias checks and register spills"
)]
fn run_diag_group_and_store<
    B: MathBackend,
    const DST_G: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    blk: &V3GruBlockHot,
    film_g: &[f32; 4],
    bn_eff: &[f32; 4],
    tanh_bn: &[f32; 4],
    pre_zrn: &[[[[f32; 4]; 2]; 3]; SUB_CAP],
    f4: &[[[f32; 4]; 2]; SUB_CAP],
    h_hist_grp: &mut [[[f32; 4]; 2]; HIST_CAP],
    state_l: &mut [f32; 4],
    state_r: &mut [f32; 4],
    n: usize,
) {
    const { assert!(HIST_CAP == SUB_CAP + 1) };
    h_hist_grp[0] = [*state_l, *state_r];
    let mut hl = B::Vec4::load(state_l);
    let mut hr = B::Vec4::load(state_r);
    for i in 0..n {
        eval_diag_rec_step::<B, DST_G>(
            blk,
            film_g,
            bn_eff,
            tanh_bn,
            &pre_zrn[i],
            &f4[i],
            &mut hl,
            &mut hr,
        );
        hl.store(&mut h_hist_grp[i + 1][0]);
        hr.store(&mut h_hist_grp[i + 1][1]);
    }
    hl.store(state_l);
    hr.store(state_r);
}

/// Computes `5`-channel physical envelope & pinch features across `[0..len]` (`len <= SUB_CAP`).
#[inline(never)]
pub(super) fn run_input_features<
    B: MathBackend,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    len: usize,
) {
    let n = len.min(SUB_CAP);
    let feat_l = &mut core.state_l.feat;
    let feat_r = &mut core.state_r.feat;
    let params = &core.feat_params;
    let lp2_mix = core.svf_params.feat_lp2_mix;
    let mut i = 0_usize;
    while i + 1 < n {
        let (f03_0, f03_1, f4_0, f4_1, hps) =
            V3InputFeaturesState::step_stereo_pair_with_lp2_vec4::<B>(
                feat_l,
                feat_r,
                params,
                sc.u_s2[i],
                sc.u_s2[i + 1],
                sc.svf_lp2[i],
                sc.svf_lp2[i + 1],
                lp2_mix,
            );
        f03_0[0].store(&mut sc.f03[i][0]);
        f03_0[1].store(&mut sc.f03[i][1]);
        f03_1[0].store(&mut sc.f03[i + 1][0]);
        f03_1[1].store(&mut sc.f03[i + 1][1]);
        f4_0[0].store(&mut sc.f4[i][0]);
        f4_0[1].store(&mut sc.f4[i][1]);
        f4_1[0].store(&mut sc.f4[i + 1][0]);
        f4_1[1].store(&mut sc.f4[i + 1][1]);
        sc.u_hp[i] = hps[0];
        sc.u_hp[i + 1] = hps[1];
        i += 2;
    }
    if i < n {
        let (fl, fr, es_l, es_r, hp_l, hp_r) = V3InputFeaturesState::step_stereo_with_lp2_vec4::<B>(
            feat_l,
            feat_r,
            params,
            sc.u_s2[i][0],
            sc.u_s2[i][1],
            sc.svf_lp2[i],
            lp2_mix,
        );
        fl.store(&mut sc.f03[i][0]);
        fr.store(&mut sc.f03[i][1]);
        es_l.store(&mut sc.f4[i][0]);
        es_r.store(&mut sc.f4[i][1]);
        sc.u_hp[i] = [hp_l, hp_r];
    }
}

/// Evaluates Block 0 (`G0` groups in `1..=3`) writing `h_hist0[0..G0]`.
#[inline(never)]
pub(super) fn run_block0_subblock<
    B: MathBackend,
    const G0: usize,
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
    let n = len.min(SUB_CAP);
    if n == 0 {
        return;
    }
    let blk = &core.blocks[0];
    let film = &core.ctrl_bus.block_film_gain[0];
    let bn = &core.ctrl_bus.block_bn_eff[0];
    let tanh_bn = &core.ctrl_bus.block_tanh_bn_eff[0];

    let inp0 = PrePassIn {
        blk,
        b1_inj: 0.0,
        f03: &sc.f03,
        f4: &sc.f4,
        tone_out: &sc.tone_out,
        h_src: &[],
        n,
    };
    fill_pre_with_opt_tile::<B, 0, false, false, SUB_CAP>(
        &inp0,
        &blk.w_rec_z,
        &blk.w_rec_r,
        &blk.w_rec_n,
        0,
        &mut sc.pre_zrn,
    );
    if let [h0_g0, ..] = &mut sc.h_hist0[..] {
        run_diag_group_and_store::<B, 0, SUB_CAP, HIST_CAP>(
            blk,
            &film[0],
            &bn[0],
            &tanh_bn[0],
            &sc.pre_zrn,
            &sc.f4,
            h0_g0,
            &mut core.state_l.h[0][0],
            &mut core.state_r.h[0][0],
            n,
        );
    }

    if G0 >= 2 {
        if let [h0_g0, ..] = &sc.h_hist0[..] {
            let inp1 = PrePassIn {
                blk,
                b1_inj: 0.0,
                f03: &sc.f03,
                f4: &sc.f4,
                tone_out: &sc.tone_out,
                h_src: &h0_g0[..n],
                n,
            };
            fill_pre_with_opt_tile::<B, 1, false, true, SUB_CAP>(
                &inp1,
                &blk.w_rec_z,
                &blk.w_rec_r,
                &blk.w_rec_n,
                const { tri_tile_index(1, 0) },
                &mut sc.pre_zrn,
            );
        }
        if let [_, h0_g1, ..] = &mut sc.h_hist0[..] {
            run_diag_group_and_store::<B, 1, SUB_CAP, HIST_CAP>(
                blk,
                &film[1],
                &bn[1],
                &tanh_bn[1],
                &sc.pre_zrn,
                &sc.f4,
                h0_g1,
                &mut core.state_l.h[0][1],
                &mut core.state_r.h[0][1],
                n,
            );
        }
    }

    if G0 >= 3 {
        if let [h0_g0, h0_g1, ..] = &sc.h_hist0[..] {
            let inp2 = PrePassIn {
                blk,
                b1_inj: 0.0,
                f03: &sc.f03,
                f4: &sc.f4,
                tone_out: &sc.tone_out,
                h_src: &h0_g0[..n],
                n,
            };
            fill_pre_with_opt_tile::<B, 2, false, true, SUB_CAP>(
                &inp2,
                &blk.w_rec_z,
                &blk.w_rec_r,
                &blk.w_rec_n,
                const { tri_tile_index(2, 0) },
                &mut sc.pre_zrn,
            );
            accum_tile_group_to_pre::<B, SUB_CAP>(
                &blk.w_rec_z,
                &blk.w_rec_r,
                &blk.w_rec_n,
                const { tri_tile_index(2, 1) },
                &h0_g1[..n],
                &mut sc.pre_zrn,
                n,
            );
        }
        if let [_, _, h0_g2, ..] = &mut sc.h_hist0[..] {
            run_diag_group_and_store::<B, 2, SUB_CAP, HIST_CAP>(
                blk,
                &film[2],
                &bn[2],
                &tanh_bn[2],
                &sc.pre_zrn,
                &sc.f4,
                h0_g2,
                &mut core.state_l.h[0][2],
                &mut core.state_r.h[0][2],
                n,
            );
        }
    }
}

/// Evaluates the inter-block Tone SVF between Block 0 and Block 1 into `sc.tone_out`.
#[inline(never)]
pub(super) fn run_interblock_tone<
    B: MathBackend,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    len: usize,
) {
    let n = len.min(SUB_CAP);
    if n == 0 {
        return;
    }
    let h0w = B::Vec4::load(&core.tone_svf.params.h0_proj_w);
    let s1_proj = core.tone_svf.params.s1_proj_w;

    for i in 0..n {
        let [ul, ur] = sc.u_s2[i];
        let (h0_l, h0_r) = B::Vec4::load(&sc.h_hist0[0][i + 1][0])
            .mul(h0w)
            .horizontal_sum_pair(B::Vec4::load(&sc.h_hist0[0][i + 1][1]).mul(h0w));
        sc.tone_out[i] = core
            .tone_svf
            .step_stereo(h0_l + s1_proj * ul, h0_r + s1_proj * ur);
    }
}

/// Evaluates Block 1 (`G1` groups in `1..=2`) writing `h_hist1[0..G1]`.
#[inline(never)]
pub(super) fn run_block1_subblock<
    B: MathBackend,
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
    const { assert!(G1 >= 1 && G1 <= CAP_G1 && CAP_G1 <= 2) };
    let n = len.min(SUB_CAP);
    if n == 0 {
        return;
    }
    let blk = &core.blocks[1];
    let film = &core.ctrl_bus.block_film_gain[1];
    let bn = &core.ctrl_bus.block_bn_eff[1];
    let tanh_bn = &core.ctrl_bus.block_tanh_bn_eff[1];
    let b1_inj = core.tone_svf.params.b1_inject_w;

    let inp0 = PrePassIn {
        blk,
        b1_inj,
        f03: &sc.f03,
        f4: &sc.f4,
        tone_out: &sc.tone_out,
        h_src: &sc.h_hist0[0][1..=n],
        n,
    };
    fill_pre_with_opt_tile::<B, 0, true, true, SUB_CAP>(
        &inp0,
        &blk.w_prev_z,
        &blk.w_prev_r,
        &blk.w_prev_n,
        0,
        &mut sc.pre_zrn,
    );
    if let [h1_g0, ..] = &mut sc.h_hist1[..] {
        run_diag_group_and_store::<B, 0, SUB_CAP, HIST_CAP>(
            blk,
            &film[0],
            &bn[0],
            &tanh_bn[0],
            &sc.pre_zrn,
            &sc.f4,
            h1_g0,
            &mut core.state_l.h[1][0],
            &mut core.state_r.h[1][0],
            n,
        );
    }

    if G1 >= 2 {
        if let [h0_g0, h0_g1, ..] = &sc.h_hist0[..]
            && let [h1_g0, ..] = &sc.h_hist1[..]
        {
            let inp1 = PrePassIn {
                blk,
                b1_inj,
                f03: &sc.f03,
                f4: &sc.f4,
                tone_out: &sc.tone_out,
                h_src: &h0_g0[1..=n],
                n,
            };
            fill_pre_with_opt_tile::<B, 1, true, true, SUB_CAP>(
                &inp1,
                &blk.w_prev_z,
                &blk.w_prev_r,
                &blk.w_prev_n,
                const { tri_tile_index(1, 0) },
                &mut sc.pre_zrn,
            );
            accum_prev_and_rec_pair_to_pre::<B, SUB_CAP>(
                blk,
                const { tri_tile_index(1, 1) },
                const { tri_tile_index(1, 0) },
                &h0_g1[1..=n],
                &h1_g0[..n],
                &mut sc.pre_zrn,
                n,
            );
        }
        if let [_, h1_g1, ..] = &mut sc.h_hist1[..] {
            run_diag_group_and_store::<B, 1, SUB_CAP, HIST_CAP>(
                blk,
                &film[1],
                &bn[1],
                &tanh_bn[1],
                &sc.pre_zrn,
                &sc.f4,
                h1_g1,
                &mut core.state_l.h[1][1],
                &mut core.state_r.h[1][1],
                n,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::stages::stage3_cab::V3ColdTierModel;
    use crate::types::Tier;

    #[test]
    fn test_run_input_features_odd_tail_matches_scalar_sequence() {
        let model = V3ColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut core_pair = V3HotCore::<DefaultMathBackend>::from_cold_model(&model);
        let mut core_single = core_pair.clone();
        let mut sc_pair = V3KernelScratch::<1, 1, 16, 17>::zeroed();
        let mut sc_single = V3KernelScratch::<1, 1, 16, 17>::zeroed();

        for i in 0..3 {
            let sig = [0.25 * (i as f32 + 1.0), -0.18 * (i as f32 + 1.0)];
            sc_pair.u_s2[i] = sig;
            sc_single.u_s2[0] = sig;
            run_input_features::<DefaultMathBackend, 1, 1, 16, 17>(
                &mut core_single,
                &mut sc_single,
                1,
            );
            if i == 2 {
                run_input_features::<DefaultMathBackend, 1, 1, 16, 17>(
                    &mut core_pair,
                    &mut sc_pair,
                    3,
                );
            }
            if i == 2 {
                assert!((sc_pair.f03[2][0][0] - sc_single.f03[0][0][0]).abs() < 1.0e-6);
                assert!((sc_pair.f4[2][1][0] - sc_single.f4[0][1][0]).abs() < 1.0e-6);
                assert!((sc_pair.u_hp[2][0] - sc_single.u_hp[0][0]).abs() < 1.0e-6);
            }
        }
    }
}
