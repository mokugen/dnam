//! Sub-block-first (`3 kHz` cadence) real-time block kernels for `V3HotCore`.

mod stage2;

use stage2::{
    V3KernelScratch, run_block0_subblock, run_block1_subblock, run_block2_and_readout,
    run_input_features, run_interblock_tone,
};

use crate::math::{MathBackend, SimdVec4};
use crate::stages::hot_core::V3HotCore;
use crate::stages::stage1_svf::{SvfHoistedConsts, SvfSubblockOut};
use crate::stages::stage2_gru::MAX_GROUPS;
use crate::stages::stage2_gru::frontend::UpsampleConsts;

#[inline(always)]
fn sanitize_stereo_sample(raw_l: f32, raw_r: f32, mono_out: bool) -> [f32; 2] {
    let l = if raw_l.is_finite() {
        raw_l.clamp(-1.0e9, 1.0e9)
    } else {
        0.0
    };
    let r = if mono_out {
        l
    } else if raw_r.is_finite() {
        raw_r.clamp(-1.0e9, 1.0e9)
    } else {
        0.0
    };
    [l, r]
}

#[derive(Clone, Debug)]
pub(crate) enum V3HotScratch {
    Lite1x(V3KernelScratch<1, 1, 16, 17>),
    Full1x(V3KernelScratch<2, 1, 16, 17>),
    Pro2x(V3KernelScratch<2, 2, 32, 33>),
    Ultra2x(V3KernelScratch<3, 2, 32, 33>),
    Other2x(V3KernelScratch<3, 2, 32, 33>),
    Other1x(V3KernelScratch<3, 2, 16, 17>),
}

impl V3HotScratch {
    /// Allocates stack scratch matching the active tier's group counts and oversampling cadence.
    #[inline(always)]
    pub(crate) fn for_core<B: MathBackend>(core: &V3HotCore<B>) -> Self {
        match (
            core.num_blocks,
            core.block_groups,
            core.num_fir_segments,
            core.num_kautz_banks,
            core.os2x,
        ) {
            (3, [1, 1, 1], 1, 1, false) => Self::Lite1x(V3KernelScratch::zeroed()),
            (3, [2, 1, 1], 2, 2, false) => Self::Full1x(V3KernelScratch::zeroed()),
            (3, [2, 2, 1], 3, 3, true) => Self::Pro2x(V3KernelScratch::zeroed()),
            (3, [3, 2, 1], 4, 4, true) => Self::Ultra2x(V3KernelScratch::zeroed()),
            (_, _, _, _, true) => Self::Other2x(V3KernelScratch::zeroed()),
            (_, _, _, _, false) => Self::Other1x(V3KernelScratch::zeroed()),
        }
    }

    #[inline(always)]
    fn matches_core<B: MathBackend>(&self, core: &V3HotCore<B>) -> bool {
        match (
            self,
            core.num_blocks,
            core.block_groups,
            core.num_fir_segments,
            core.num_kautz_banks,
            core.os2x,
        ) {
            (Self::Lite1x(_), 3, [1, 1, 1], 1, 1, false)
            | (Self::Full1x(_), 3, [2, 1, 1], 2, 2, false)
            | (Self::Pro2x(_), 3, [2, 2, 1], 3, 3, true)
            | (Self::Ultra2x(_), 3, [3, 2, 1], 4, 4, true) => true,
            (Self::Other2x(_), n, bg, f, k, true)
                if !(n == 3
                    && (bg == [2, 2, 1] && f == 3 && k == 3
                        || bg == [3, 2, 1] && f == 4 && k == 4)) =>
            {
                true
            }
            (Self::Other1x(_), n, bg, f, k, false)
                if !(n == 3
                    && (bg == [1, 1, 1] && f == 1 && k == 1
                        || bg == [2, 1, 1] && f == 2 && k == 2)) =>
            {
                true
            }
            _ => false,
        }
    }

    /// Dispatches one stereo sub-block on `core` using this caller-owned stack scratch.
    #[inline(always)]
    pub(crate) fn dispatch<B: MathBackend>(
        &mut self,
        core: &mut V3HotCore<B>,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        if !self.matches_core(core) {
            *self = Self::for_core(core);
        }
        match (
            self,
            core.num_blocks,
            core.block_groups,
            core.num_fir_segments,
            core.num_kautz_banks,
            core.os2x,
        ) {
            (Self::Lite1x(s), 3, [1, 1, 1], 1, 1, false) => {
                process_subblock_with_scratch::<B, 1, 1, 1, 1, false, 1, 1, 16, 17>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Full1x(s), 3, [2, 1, 1], 2, 2, false) => {
                process_subblock_with_scratch::<B, 2, 1, 2, 2, false, 2, 1, 16, 17>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Pro2x(s), 3, [2, 2, 1], 3, 3, true) => {
                process_subblock_with_scratch::<B, 2, 2, 3, 3, true, 2, 2, 32, 33>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Ultra2x(s), 3, [3, 2, 1], 4, 4, true) => {
                process_subblock_with_scratch::<B, 3, 2, 4, 4, true, 3, 2, 32, 33>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Other1x(s), 3, [2, 2, 1], 3, 3, false) => {
                process_subblock_with_scratch::<B, 2, 2, 3, 3, false, 3, 2, 16, 17>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Other1x(s), 3, [3, 2, 1], 4, 4, false) => {
                process_subblock_with_scratch::<B, 3, 2, 4, 4, false, 3, 2, 16, 17>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Other2x(s), 3, [1, 1, 1], 1, 1, true) => {
                process_subblock_with_scratch::<B, 1, 1, 1, 1, true, 3, 2, 32, 33>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            (Self::Other2x(s), 3, [2, 1, 1], 2, 2, true) => {
                process_subblock_with_scratch::<B, 2, 1, 2, 2, true, 3, 2, 32, 33>(
                    core, s, in_l, in_r, out_l, out_r,
                );
            }
            _ if core.os2x => core.process_block_2x(in_l, in_r, out_l, out_r),
            _ => core.process_block_1x(in_l, in_r, out_l, out_r),
        }
    }
}

#[inline(never)]
fn run_stage0_up<
    B: MathBackend,
    const OS2X: bool,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    in_l: &[f32],
    in_r: &[f32],
    len: usize,
) -> usize {
    let n = len.min(16).min(in_l.len()).min(in_r.len());
    let dup_l_to_r = core.mono_out && core.feat_params.key_link <= 0.0;
    let mut last = core.last_in;
    if OS2X {
        let up_c = UpsampleConsts::<B::Vec4>::from_params(&core.os_params);
        let mut s0 = B::Vec4::load(&core.os_stereo.up_hb_s0);
        let mut s1 = B::Vec4::load(&core.os_stereo.up_hb_s1);
        for i in 0..n {
            let [sl, sr] = sanitize_stereo_sample(in_l[i], in_r[i], dup_l_to_r);
            let out0 = core.stage0.step_stereo(sl, sr);
            last = [sl, sr];
            sc.clean_sub[i] = [out0[2], out0[3]];
            let (u0, u1) = core
                .os_stereo
                .upsample_stereo_reg::<B>(&mut s0, &mut s1, &up_c, out0[0], out0[1]);
            sc.stage1_in[2 * i] = u0;
            sc.stage1_in[2 * i + 1] = u1;
        }
        core.last_in = last;
        s0.store(&mut core.os_stereo.up_hb_s0);
        s1.store(&mut core.os_stereo.up_hb_s1);
        n * 2
    } else {
        for i in 0..n {
            let [sl, sr] = sanitize_stereo_sample(in_l[i], in_r[i], dup_l_to_r);
            let out0 = core.stage0.step_stereo(sl, sr);
            last = [sl, sr];
            sc.clean_sub[i] = [out0[2], out0[3]];
            sc.stage1_in[i] = [out0[0], out0[1]];
        }
        core.last_in = last;
        n
    }
}

#[inline(never)]
fn run_stage1_svf<
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
    let svf_c = SvfHoistedConsts::<B::Vec4>::from_params(
        &core.svf_params,
        core.ctrl_bus.env_fc_scale,
        core.ctrl_bus.env_k_delta,
    );
    let mut sb_out = SvfSubblockOut {
        env: &mut sc.svf_env,
        a: &mut sc.svf_a,
        u_s2: &mut sc.u_s2,
        svf_dir: &mut sc.svf_dir,
        lp2: &mut sc.svf_lp2,
    };
    core.svf_state
        .step_subblock_fast::<B, SUB_CAP>(&svf_c, &sc.stage1_in, &mut sb_out, len);
}

#[inline(never)]
fn run_back_stereo<
    B: MathBackend,
    const FIR_SEGS: usize,
    const KAUTZ_BANKS: usize,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    out_l: &mut [f32],
    out_r: &mut [f32],
    len: usize,
) {
    let n = len.min(16).min(out_l.len()).min(out_r.len());
    core.stage3
        .step_subblock_const::<B, FIR_SEGS, KAUTZ_BANKS, false, 16>(&sc.s2_dir, &mut sc.stage3, n);
    core.stage4.step_subblock::<B, false>(
        &sc.s2_dir,
        &sc.stage3.cab_out,
        &sc.clean_sub,
        &mut out_l[..n],
        &mut out_r[..n],
        n,
    );
    if core.post_cond_active {
        core.post_cond.process_slice::<false>(out_l, out_r, n);
    }
}

#[inline(never)]
fn run_back_mono<
    B: MathBackend,
    const FIR_SEGS: usize,
    const KAUTZ_BANKS: usize,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    out_l: &mut [f32],
    out_r: &mut [f32],
    len: usize,
) {
    let n = len.min(16).min(out_l.len()).min(out_r.len());
    core.stage3
        .step_subblock_const::<B, FIR_SEGS, KAUTZ_BANKS, true, 16>(&sc.s2_dir, &mut sc.stage3, n);
    core.stage4.step_subblock::<B, true>(
        &sc.s2_dir,
        &sc.stage3.cab_out,
        &sc.clean_sub,
        &mut out_l[..n],
        &mut out_r[..n],
        n,
    );
    if core.post_cond_active {
        core.post_cond.process_slice::<true>(out_l, out_r, n);
    }
}

/// Executes a contiguous stereo block reusing caller-supplied sub-block scratch across control chunks.
#[inline(never)]
pub(crate) fn process_subblock_with_scratch<
    B: MathBackend,
    const G0: usize,
    const G1: usize,
    const FIR_SEGS: usize,
    const KAUTZ_BANKS: usize,
    const OS2X: bool,
    const CAP_G0: usize,
    const CAP_G1: usize,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    sc: &mut V3KernelScratch<CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    const { assert!(G0 >= 1 && G0 <= CAP_G0 && G1 >= 1 && G1 <= CAP_G1) };
    const { assert!((!OS2X && SUB_CAP == 16) || (OS2X && SUB_CAP == 32)) };
    const { assert!(HIST_CAP == SUB_CAP + 1) };
    let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());

    let mut pos = 0_usize;
    while pos < n {
        let rem_in_tick = 16 - ((core.sample_tick & 15) as usize);
        let sub_len = (n - pos).min(rem_in_tick);
        if (core.sample_tick & 15) == 0 && core.ctrl_bus.dirty {
            core.refresh_3khz_tick();
        }

        let s2_len = run_stage0_up::<B, OS2X, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(
            core,
            sc,
            &in_l[pos..pos + sub_len],
            &in_r[pos..pos + sub_len],
            sub_len,
        );
        run_stage1_svf::<B, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(core, sc, s2_len);
        run_input_features::<B, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(core, sc, s2_len);
        run_block0_subblock::<B, G0, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(core, sc, s2_len);
        run_interblock_tone::<B, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(core, sc, s2_len);
        run_block1_subblock::<B, G1, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(core, sc, s2_len);
        run_block2_and_readout::<B, G0, G1, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(core, sc, s2_len);

        if OS2X {
            core.os_stereo.downsample_subblock_2x::<B>(
                &core.os_params,
                &sc.s2_sub,
                &mut sc.s2_dir,
                sub_len,
            );
        } else {
            sc.s2_dir[..sub_len].copy_from_slice(&sc.s2_sub[..sub_len]);
        }
        core.last_s2_dir = sc.s2_dir[sub_len - 1];

        if core.mono_out {
            run_back_mono::<B, FIR_SEGS, KAUTZ_BANKS, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(
                core,
                sc,
                &mut out_l[pos..pos + sub_len],
                &mut out_r[pos..pos + sub_len],
                sub_len,
            );
        } else {
            run_back_stereo::<B, FIR_SEGS, KAUTZ_BANKS, CAP_G0, CAP_G1, SUB_CAP, HIST_CAP>(
                core,
                sc,
                &mut out_l[pos..pos + sub_len],
                &mut out_r[pos..pos + sub_len],
                sub_len,
            );
        }

        core.sample_tick = core.sample_tick.wrapping_add(sub_len as u32);
        pos += sub_len;
    }
}

/// Executes a contiguous stereo block in `<= 16`-sample control sub-blocks matching `3 kHz` ticks.
#[inline(never)]
pub(crate) fn process_block_const<
    B: MathBackend,
    const G0: usize,
    const G1: usize,
    const FIR_SEGS: usize,
    const KAUTZ_BANKS: usize,
    const OS2X: bool,
    const SUB_CAP: usize,
    const HIST_CAP: usize,
>(
    core: &mut V3HotCore<B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    const { assert!(G0 >= 1 && G0 <= MAX_GROUPS && G1 >= 1 && G1 <= 2) };
    const { assert!((!OS2X && SUB_CAP == 16) || (OS2X && SUB_CAP == 32)) };
    const { assert!(HIST_CAP == SUB_CAP + 1) };
    let mut sc = V3KernelScratch::<G0, G1, SUB_CAP, HIST_CAP>::zeroed();
    process_subblock_with_scratch::<
        B,
        G0,
        G1,
        FIR_SEGS,
        KAUTZ_BANKS,
        OS2X,
        G0,
        G1,
        SUB_CAP,
        HIST_CAP,
    >(core, &mut sc, in_l, in_r, out_l, out_r);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::stages::stage3_cab::V3MatryoshkaModel;
    use crate::types::Tier;

    fn make_active_core(tier: Tier, mono_out: bool) -> V3HotCore<DefaultMathBackend> {
        let mut model = V3MatryoshkaModel::dual_rate_default(mono_out);
        model.blocks[0].w_in_n[0][0] = 0.45;
        model.blocks[1].w_prev_n[0][0] = 0.25;
        model.readouts[tier as usize].pre_cab_u_gain = [0.6, 0.55];
        model.readouts[tier as usize].fir_l[0] = 0.85;
        model.readouts[tier as usize].fir_r[0] = 0.80;
        V3HotCore::from_cold_model(&model.extract_tier(tier))
    }

    #[test]
    fn test_process_block_const_tiers_partial_tail_and_mono() {
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let mut blk_core = make_active_core(tier, false);
            let mut samp_core = blk_core.clone();
            let in_l = [0.15_f32; 23];
            let mut in_r = [-0.12_f32; 23];
            in_r[21] = f32::NAN;
            let (mut bl, mut br) = ([0.0_f32; 23], [0.0_f32; 23]);
            let (mut sl, mut sr) = ([0.0_f32; 23], [0.0_f32; 23]);
            blk_core.process_block_stereo(&in_l, &in_r, &mut bl, &mut br);
            for i in 0..23 {
                let [yl, yr] = samp_core.step_stereo(in_l[i], in_r[i]);
                sl[i] = yl;
                sr[i] = yr;
            }
            for i in 0..23 {
                assert!(
                    (bl[i] - sl[i]).abs() < 2.0e-5 && (br[i] - sr[i]).abs() < 2.0e-5,
                    "tier={tier:?} i={i}: blk=[{}, {}] vs samp=[{}, {}]",
                    bl[i],
                    br[i],
                    sl[i],
                    sr[i]
                );
            }
        }

        let mut mono_core = make_active_core(Tier::Full, true);
        let in_l = [0.1_f32; 11];
        let in_r = [-0.9_f32; 11];
        let (mut ol, mut or) = ([0.0_f32; 11], [0.0_f32; 11]);
        mono_core.process_block_stereo(&in_l, &in_r, &mut ol, &mut or);
        assert_eq!(ol, or);
        assert!(ol.iter().all(|v| v.is_finite()));

        // Exercise V3HotScratch across every tier/OS combination, mismatch re-init, and fallback.
        let mut sc = V3HotScratch::for_core(&make_active_core(Tier::Lite, false));
        let in16 = [0.12_f32; 16];
        let (mut out_a, mut out_b) = ([0.0_f32; 16], [0.0_f32; 16]);
        for os2x in [false, true] {
            for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
                let mut m = V3MatryoshkaModel::identity(os2x, false);
                m.readouts[tier as usize].fir_l[0] = 0.9;
                let mut core =
                    V3HotCore::<DefaultMathBackend>::from_cold_model(&m.extract_tier(tier));
                sc.dispatch(&mut core, &in16, &in16, &mut out_a, &mut out_b);
                assert!(out_a.iter().all(|v| v.is_finite()));
            }
            let mut core_custom = make_active_core(Tier::Lite, false);
            core_custom.os2x = os2x;
            core_custom.num_blocks = 2;
            sc.dispatch(&mut core_custom, &in16, &in16, &mut out_a, &mut out_b);
            assert!(out_a.iter().all(|v| v.is_finite()));
        }
    }
}
