//! Const-Generic `16`-Frame Register-Hoisted `SONA` DSP Execution (`graph/sona_core/tick16.rs`).
//!
//! Strictly `#![forbid(unsafe_code)]`, `0` `FDIV`, `0` `FSQRT`, `0` `IDIV`/`UDIV`/`SDIV`, `0` `libm`,
//! `0` heap allocations, and `0` stage-bypass flags across `Lite..Ultra` (`REQ-6.2`, `REQ-6.3`).

#![forbid(unsafe_code)]

pub mod crossbar_step;

use dnam_kernels::{
    FULL_SPEC, LITE_SPEC, PRO_SPEC, SonaGroupParamsStereo4, StereoVec8, ULTRA_SPEC,
};

use self::crossbar_step::{
    Stage0HostCoeffs, prepare_crossbar_frame, stage3_step_subblock, stage4_step_stereo,
    stage4_step_subblock, step_sona_core_sample, step_stage0_hot,
};
use crate::graph::GraphBackend;
use crate::graph::cabinet::CabinetSubblockScratch;
use crate::graph::sona_core::frontend::UpsampleConsts;
use crate::graph::sona_core::slab::ctrl_apply::{
    hoist_groups_and_readouts, refresh_3khz_tick_impl,
};
use crate::graph::sona_core::slab::{CrossbarFrame, SonaGroupReadoutRegs, SonaHotSlab};
use crate::math::SimdVec4;
use crate::math::fp16::fp16_to_f32;
use crate::types::{FtzDazGuard, Tier};

impl<B: GraphBackend> SonaHotSlab<B> {
    #[inline(always)]
    fn max_op_n_drive_vec4<const G_CAP: usize>(&mut self, mut acc: B::Vec4) -> B::Vec4 {
        for g in 0..G_CAP {
            let (l, r) = self.state_mut(g).adaa_prev.into_lr();
            acc = acc.max(l.abs().max(r.abs()));
        }
        acc
    }

    #[inline(always)]
    fn commit_op_n_drive_peak(&mut self, acc: B::Vec4) {
        let arr = acc.to_array();
        let lane_max = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
        let bounded = if lane_max.is_finite() {
            lane_max.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.cold.max_op_n_drive = self.cold.max_op_n_drive.max(bounded);
    }
    #[inline]
    fn dispatch_core_sample<
        const G_CAP: usize,
        const T_CAP: usize,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
        const OS2X: bool,
    >(
        &mut self,
        g_params: &[SonaGroupParamsStereo4<B::Vec4>; G_CAP],
        ro_regs: &[SonaGroupReadoutRegs<B::Vec4, B::Stereo8>; G_CAP],
        u_l: f32,
        u_r: f32,
        cb: &CrossbarFrame<B::Vec4>,
    ) -> [f32; 4] {
        if OS2X {
            let osp = self.os2x_hot.os_params;
            let (up0, up1) = self.os2x_hot.os_stereo.upsample_stereo::<B>(u_l, u_r, &osp);
            let sub0 = step_sona_core_sample::<B, G_CAP, T_CAP, B0, B1, B2, B3>(
                self, g_params, ro_regs, up0[0], up0[1], cb,
            );
            let mut drive_acc = self.max_op_n_drive_vec4::<G_CAP>(B::Vec4::zero());
            let sub1 = step_sona_core_sample::<B, G_CAP, T_CAP, B0, B1, B2, B3>(
                self, g_params, ro_regs, up1[0], up1[1], cb,
            );
            drive_acc = self.max_op_n_drive_vec4::<G_CAP>(drive_acc);
            self.commit_op_n_drive_peak(drive_acc);
            self.os2x_hot
                .os_stereo
                .downsample_4ch::<B>(sub0, sub1, &osp)
        } else {
            let s2 = step_sona_core_sample::<B, G_CAP, T_CAP, B0, B1, B2, B3>(
                self, g_params, ro_regs, u_l, u_r, cb,
            );
            let drive_acc = self.max_op_n_drive_vec4::<G_CAP>(B::Vec4::zero());
            self.commit_op_n_drive_peak(drive_acc);
            s2
        }
    }

    #[inline]
    fn step_post_core<const FIR_SEGS: usize, const KAUTZ_BANKS: usize, const MONO: bool>(
        &mut self,
        s2: [f32; 4],
        clean_l: f32,
        clean_r: f32,
    ) -> [f32; 2] {
        self.cold.last_s2_dir = s2;
        let mut sc = CabinetSubblockScratch::<1>::zeroed();
        let pre = [[s2[0], s2[1]]];
        stage3_step_subblock::<B, FIR_SEGS, KAUTZ_BANKS, MONO, 1>(self, &pre, &mut sc, 1);
        let core_l = s2[2] + clean_l + sc.cab_out[0][0];
        let core_r = s2[3] + clean_r + sc.cab_out[0][1];
        stage4_step_stereo(self, core_l, core_r)
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn step_stereo_tier<
        const G_CAP: usize,
        const T_CAP: usize,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
        const FIR_SEGS: usize,
        const KAUTZ_BANKS: usize,
        const OS2X: bool,
        const MONO: bool,
    >(
        &mut self,
        in_l: f32,
        in_r: f32,
    ) -> [f32; 2] {
        let xl = if in_l.is_finite() { in_l } else { 0.0 };
        let xr = if in_r.is_finite() { in_r } else { 0.0 };
        self.cold.last_in = [xl, xr];
        if (self.lite.sample_tick & 15) == 0 {
            refresh_3khz_tick_impl(self);
        }
        self.lite.sample_tick = self.lite.sample_tick.wrapping_add(1);

        let s0c = Stage0HostCoeffs::from_slab(self);
        let [s0_l, s0_r, clean_l, clean_r] = step_stage0_hot(self, &s0c, xl, xr);
        let (x_l, x_r, cb) = prepare_crossbar_frame(self, s0_l, s0_r);
        let (g_params, ro_regs) = hoist_groups_and_readouts::<B, G_CAP>(self);
        let s2 = self.dispatch_core_sample::<G_CAP, T_CAP, B0, B1, B2, B3, OS2X>(
            &g_params, &ro_regs, x_l, x_r, &cb,
        );
        self.step_post_core::<FIR_SEGS, KAUTZ_BANKS, MONO>(s2, clean_l, clean_r)
    }

    /// Processes one stereo frame at sample resolution (caller holds [`FtzDazGuard`]).
    #[inline(never)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        match (self.tier(), self.os2x(), self.mono_out()) {
            (Tier::Lite, false, false) => {
                self.step_stereo_tier::<4, 3, 1, 1, 1, 1, 2, 2, false, false>(in_l, in_r)
            }
            (Tier::Lite, false, true) => {
                self.step_stereo_tier::<4, 3, 1, 1, 1, 1, 2, 2, false, true>(in_l, in_r)
            }
            (Tier::Lite, true, false) => {
                self.step_stereo_tier::<4, 3, 1, 1, 1, 1, 2, 2, true, false>(in_l, in_r)
            }
            (Tier::Lite, true, true) => {
                self.step_stereo_tier::<4, 3, 1, 1, 1, 1, 2, 2, true, true>(in_l, in_r)
            }
            (Tier::Full, false, false) => {
                self.step_stereo_tier::<6, 13, 2, 2, 1, 1, 3, 3, false, false>(in_l, in_r)
            }
            (Tier::Full, false, true) => {
                self.step_stereo_tier::<6, 13, 2, 2, 1, 1, 3, 3, false, true>(in_l, in_r)
            }
            (Tier::Full, true, false) => {
                self.step_stereo_tier::<6, 13, 2, 2, 1, 1, 3, 3, true, false>(in_l, in_r)
            }
            (Tier::Full, true, true) => {
                self.step_stereo_tier::<6, 13, 2, 2, 1, 1, 3, 3, true, true>(in_l, in_r)
            }
            (Tier::Pro, false, false) => {
                self.step_stereo_tier::<8, 23, 3, 2, 2, 1, 6, 6, false, false>(in_l, in_r)
            }
            (Tier::Pro, false, true) => {
                self.step_stereo_tier::<8, 23, 3, 2, 2, 1, 6, 6, false, true>(in_l, in_r)
            }
            (Tier::Pro, true, false) => {
                self.step_stereo_tier::<8, 23, 3, 2, 2, 1, 6, 6, true, false>(in_l, in_r)
            }
            (Tier::Pro, true, true) => {
                self.step_stereo_tier::<8, 23, 3, 2, 2, 1, 6, 6, true, true>(in_l, in_r)
            }
            (Tier::Ultra, false, false) => {
                self.step_stereo_tier::<10, 35, 4, 3, 2, 1, 8, 8, false, false>(in_l, in_r)
            }
            (Tier::Ultra, false, true) => {
                self.step_stereo_tier::<10, 35, 4, 3, 2, 1, 8, 8, false, true>(in_l, in_r)
            }
            (Tier::Ultra, true, false) => {
                self.step_stereo_tier::<10, 35, 4, 3, 2, 1, 8, 8, true, false>(in_l, in_r)
            }
            (Tier::Ultra, true, true) => {
                self.step_stereo_tier::<10, 35, 4, 3, 2, 1, 8, 8, true, true>(in_l, in_r)
            }
        }
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn step_4port_tier<
        const G_CAP: usize,
        const T_CAP: usize,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
        const FIR_SEGS: usize,
        const KAUTZ_BANKS: usize,
        const OS2X: bool,
    >(
        &mut self,
        x_l: f32,
        x_r: f32,
        clean_l: f32,
        clean_r: f32,
        cb: &CrossbarFrame<B::Vec4>,
    ) -> ([f32; 4], [f32; 2]) {
        let (gp, ro) = hoist_groups_and_readouts::<B, G_CAP>(self);
        let s2 =
            self.dispatch_core_sample::<G_CAP, T_CAP, B0, B1, B2, B3, OS2X>(&gp, &ro, x_l, x_r, cb);
        let out = if self.mono_out() {
            self.step_post_core::<FIR_SEGS, KAUTZ_BANKS, true>(s2, clean_l, clean_r)
        } else {
            self.step_post_core::<FIR_SEGS, KAUTZ_BANKS, false>(s2, clean_l, clean_r)
        };
        (s2, out)
    }

    /// Steps one 4-port (`4CM` / `WDW` / `Sidechain`) stereo frame preserving strict `S_{ret->send} == 0`.
    #[inline(never)]
    pub fn step_4port_frame(
        &mut self,
        router: &mut crate::graph::IoRouter,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        has_return: bool,
    ) -> ([f32; 2], [f32; 2], [f32; 2]) {
        let xl = if in_l.is_finite() { in_l } else { 0.0 };
        let xr = if in_r.is_finite() { in_r } else { 0.0 };
        self.cold.last_in = [xl, xr];
        if (self.lite.sample_tick & 15) == 0 {
            refresh_3khz_tick_impl(self);
        }
        self.lite.sample_tick = self.lite.sample_tick.wrapping_add(1);

        let s0c = Stage0HostCoeffs::from_slab(self);
        let [s0_l, s0_r, clean_l, clean_r] = step_stage0_hot(self, &s0c, xl, xr);
        let _ = router.route_frame_no_return(xl, xr);
        let send_l = router.s_matrix[0][0] * s0_l + router.s_matrix[0][1] * s0_r;
        let send_r = router.s_matrix[1][0] * s0_l + router.s_matrix[1][1] * s0_r;
        let (x_l, x_r) = if has_return {
            let ret_l_a = router.return_aligner_l.step(ret_l);
            let ret_r_a = router.return_aligner_r.step(ret_r);
            let s = &router.s_matrix;
            (
                s0_l + s[2][2] * ret_l_a + s[2][3] * ret_r_a,
                s0_r + s[3][2] * ret_l_a + s[3][3] * ret_r_a,
            )
        } else {
            (s0_l, s0_r)
        };
        let (x_l, x_r, cb) = prepare_crossbar_frame(self, x_l, x_r);

        let (s2, out) = match (self.tier(), self.os2x()) {
            (Tier::Lite, false) => self
                .step_4port_tier::<4, 3, 1, 1, 1, 1, 2, 2, false>(x_l, x_r, clean_l, clean_r, &cb),
            (Tier::Lite, true) => self
                .step_4port_tier::<4, 3, 1, 1, 1, 1, 2, 2, true>(x_l, x_r, clean_l, clean_r, &cb),
            (Tier::Full, false) => self
                .step_4port_tier::<6, 13, 2, 2, 1, 1, 3, 3, false>(x_l, x_r, clean_l, clean_r, &cb),
            (Tier::Full, true) => self
                .step_4port_tier::<6, 13, 2, 2, 1, 1, 3, 3, true>(x_l, x_r, clean_l, clean_r, &cb),
            (Tier::Pro, false) => self
                .step_4port_tier::<8, 23, 3, 2, 2, 1, 6, 6, false>(x_l, x_r, clean_l, clean_r, &cb),
            (Tier::Pro, true) => self
                .step_4port_tier::<8, 23, 3, 2, 2, 1, 6, 6, true>(x_l, x_r, clean_l, clean_r, &cb),
            (Tier::Ultra, false) => self.step_4port_tier::<10, 35, 4, 3, 2, 1, 8, 8, false>(
                x_l, x_r, clean_l, clean_r, &cb,
            ),
            (Tier::Ultra, true) => self
                .step_4port_tier::<10, 35, 4, 3, 2, 1, 8, 8, true>(x_l, x_r, clean_l, clean_r, &cb),
        };
        let raw_core_l = s2[2] + clean_l;
        let raw_core_r = s2[3] + clean_r;
        let (core_l, core_r) = if self.mono_out() {
            let cm = 0.5 * (raw_core_l + raw_core_r);
            (cm, cm)
        } else {
            (raw_core_l, raw_core_r)
        };
        (out, [core_l, core_r], [send_l, send_r])
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn step_tick_subblock<
        const G_CAP: usize,
        const T_CAP: usize,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
        const FIR_SEGS: usize,
        const KAUTZ_BANKS: usize,
        const TANK_QUADS: usize,
        const OS2X: bool,
        const MONO: bool,
    >(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
        s3_sc: &mut CabinetSubblockScratch<16>,
    ) {
        let n = in_l.len();
        if (self.lite.sample_tick & 15) == 0 {
            refresh_3khz_tick_impl(self);
        }
        let s0c = Stage0HostCoeffs::from_slab(self);
        let (g_params, ro_regs) = hoist_groups_and_readouts::<B, G_CAP>(self);

        let mut clean_arr = [[0.0_f32; 2]; 16];
        let mut s2_dir = [[0.0_f32; 4]; 16];

        if OS2X {
            let osp = self.os2x_hot.os_params;
            let uc = UpsampleConsts::<B::Vec4>::from_params(&osp);
            let mut up_s0 = B::Vec4::load(&self.os2x_hot.os_stereo.up_hb_s0);
            let mut up_s1 = B::Vec4::load(&self.os2x_hot.os_stereo.up_hb_s1);
            let mut up_buf = [[0.0_f32; 2]; 32];
            let mut cb_buf = [CrossbarFrame {
                active: false,
                op_p_vec: B::Vec4::zero(),
                ssb_l: B::Vec4::zero(),
                ssb_r: B::Vec4::zero(),
            }; 16];

            for i in 0..n {
                let xl = if in_l[i].is_finite() { in_l[i] } else { 0.0 };
                let xr = if in_r[i].is_finite() { in_r[i] } else { 0.0 };
                self.cold.last_in = [xl, xr];
                self.lite.sample_tick = self.lite.sample_tick.wrapping_add(1);
                let [s0_l, s0_r, cl, cr] = step_stage0_hot(self, &s0c, xl, xr);
                clean_arr[i] = [cl, cr];
                let (ux_l, ux_r, cb) = prepare_crossbar_frame(self, s0_l, s0_r);
                cb_buf[i] = cb;
                let (u0, u1) = self
                    .os2x_hot
                    .os_stereo
                    .upsample_stereo_reg::<B>(&mut up_s0, &mut up_s1, &uc, ux_l, ux_r);
                up_buf[i << 1] = u0;
                up_buf[(i << 1) + 1] = u1;
            }
            up_s0.store(&mut self.os2x_hot.os_stereo.up_hb_s0);
            up_s1.store(&mut self.os2x_hot.os_stereo.up_hb_s1);

            let mut s2_sub = [[0.0_f32; 4]; 32];
            let mut drive_acc = B::Vec4::zero();
            for j in 0..(n << 1) {
                s2_sub[j] = step_sona_core_sample::<B, G_CAP, T_CAP, B0, B1, B2, B3>(
                    self,
                    &g_params,
                    &ro_regs,
                    up_buf[j][0],
                    up_buf[j][1],
                    &cb_buf[j >> 1],
                );
                drive_acc = self.max_op_n_drive_vec4::<G_CAP>(drive_acc);
            }
            if n > 0 {
                self.commit_op_n_drive_peak(drive_acc);
            }
            self.os2x_hot.os_stereo.downsample_subblock_2x::<B>(
                &osp,
                &s2_sub[..(n << 1)],
                &mut s2_dir,
                n,
            );
        } else {
            let mut drive_acc = B::Vec4::zero();
            for i in 0..n {
                let xl = if in_l[i].is_finite() { in_l[i] } else { 0.0 };
                let xr = if in_r[i].is_finite() { in_r[i] } else { 0.0 };
                self.cold.last_in = [xl, xr];
                self.lite.sample_tick = self.lite.sample_tick.wrapping_add(1);
                let [s0_l, s0_r, cl, cr] = step_stage0_hot(self, &s0c, xl, xr);
                clean_arr[i] = [cl, cr];
                let (ux_l, ux_r, cb) = prepare_crossbar_frame(self, s0_l, s0_r);
                s2_dir[i] = step_sona_core_sample::<B, G_CAP, T_CAP, B0, B1, B2, B3>(
                    self, &g_params, &ro_regs, ux_l, ux_r, &cb,
                );
                drive_acc = self.max_op_n_drive_vec4::<G_CAP>(drive_acc);
            }
            if n > 0 {
                self.commit_op_n_drive_peak(drive_acc);
            }
        }
        if n > 0 {
            self.cold.last_s2_dir = s2_dir[n - 1];
        }

        let mut pre_cab = [[0.0_f32; 2]; 16];
        for i in 0..n {
            pre_cab[i] = [s2_dir[i][0], s2_dir[i][1]];
        }
        stage3_step_subblock::<B, FIR_SEGS, KAUTZ_BANKS, MONO, 16>(self, &pre_cab, s3_sc, n);

        let mut s4_l = [0.0_f32; 16];
        let mut s4_r = [0.0_f32; 16];
        for i in 0..n {
            s4_l[i] = s2_dir[i][2] + clean_arr[i][0] + s3_sc.cab_out[i][0];
            s4_r[i] = s2_dir[i][3] + clean_arr[i][1] + s3_sc.cab_out[i][1];
        }
        stage4_step_subblock::<B, TANK_QUADS, MONO>(self, &s4_l, &s4_r, out_l, out_r, n);
    }

    /// Register-hoisted `16`-frame const-generic sub-block pipeline (`0` alloc, unaligned head peel).
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn process_tick16_const<
        const G_CAP: usize,
        const T_CAP: usize,
        const B0: usize,
        const B1: usize,
        const B2: usize,
        const B3: usize,
        const FIR_SEGS: usize,
        const KAUTZ_BANKS: usize,
        const TANK_QUADS: usize,
        const OS2X: bool,
        const MONO: bool,
    >(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let total = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        if self.crossbar_active() && fp16_to_f32(self.lite.op_gains_fp16[1]).abs() > 1.0e-7 {
            for i in 0..total {
                let y = self.step_stereo_tier::<
                    G_CAP, T_CAP, B0, B1, B2, B3, FIR_SEGS, KAUTZ_BANKS, OS2X, MONO,
                >(in_l[i], in_r[i]);
                out_l[i] = y[0];
                out_r[i] = if MONO { y[0] } else { y[1] };
            }
            return;
        }

        let mut off = 0usize;
        let mut s3_sc = CabinetSubblockScratch::<16>::zeroed();
        let head = ((16_u32.wrapping_sub(self.lite.sample_tick & 15)) as usize & 15).min(total);
        if head > 0 {
            self.step_tick_subblock::<
                G_CAP, T_CAP, B0, B1, B2, B3, FIR_SEGS, KAUTZ_BANKS, TANK_QUADS, OS2X, MONO,
            >(&in_l[..head], &in_r[..head], &mut out_l[..head], &mut out_r[..head], &mut s3_sc);
            off = head;
        }
        while off < total {
            let n = (total - off).min(16);
            self.step_tick_subblock::<
                G_CAP, T_CAP, B0, B1, B2, B3, FIR_SEGS, KAUTZ_BANKS, TANK_QUADS, OS2X, MONO,
            >(
                &in_l[off..off + n],
                &in_r[off..off + n],
                &mut out_l[off..off + n],
                &mut out_r[off..off + n],
                &mut s3_sc,
            );
            off += n;
        }
    }

    #[inline]
    fn dispatch_block_no_ftz(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        const L_F: usize = LITE_SPEC.fir_taps / 16;
        const L_K: usize = LITE_SPEC.kautz_pairs / 4;
        const L_Q: usize = LITE_SPEC.opt_lines / 4;
        const F_F: usize = FULL_SPEC.fir_taps / 16;
        const F_K: usize = FULL_SPEC.kautz_pairs / 4;
        const F_Q: usize = FULL_SPEC.opt_lines / 4;
        const P_F: usize = PRO_SPEC.fir_taps / 16;
        const P_K: usize = PRO_SPEC.kautz_pairs / 4;
        const P_Q: usize = PRO_SPEC.opt_lines / 4;
        const U_F: usize = ULTRA_SPEC.fir_taps / 16;
        const U_K: usize = ULTRA_SPEC.kautz_pairs / 4;
        const U_Q: usize = ULTRA_SPEC.opt_lines / 4;

        match (self.tier(), self.os2x(), self.mono_out()) {
            (Tier::Lite, false, false) => self
                .process_tick16_const::<4, 3, 1, 1, 1, 1, L_F, L_K, L_Q, false, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Lite, false, true) => self
                .process_tick16_const::<4, 3, 1, 1, 1, 1, L_F, L_K, L_Q, false, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Lite, true, false) => self
                .process_tick16_const::<4, 3, 1, 1, 1, 1, L_F, L_K, L_Q, true, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Lite, true, true) => self
                .process_tick16_const::<4, 3, 1, 1, 1, 1, L_F, L_K, L_Q, true, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Full, false, false) => self
                .process_tick16_const::<6, 13, 2, 2, 1, 1, F_F, F_K, F_Q, false, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Full, false, true) => self
                .process_tick16_const::<6, 13, 2, 2, 1, 1, F_F, F_K, F_Q, false, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Full, true, false) => self
                .process_tick16_const::<6, 13, 2, 2, 1, 1, F_F, F_K, F_Q, true, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Full, true, true) => self
                .process_tick16_const::<6, 13, 2, 2, 1, 1, F_F, F_K, F_Q, true, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Pro, false, false) => self
                .process_tick16_const::<8, 23, 3, 2, 2, 1, P_F, P_K, P_Q, false, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Pro, false, true) => self
                .process_tick16_const::<8, 23, 3, 2, 2, 1, P_F, P_K, P_Q, false, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Pro, true, false) => self
                .process_tick16_const::<8, 23, 3, 2, 2, 1, P_F, P_K, P_Q, true, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Pro, true, true) => self
                .process_tick16_const::<8, 23, 3, 2, 2, 1, P_F, P_K, P_Q, true, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Ultra, false, false) => self
                .process_tick16_const::<10, 35, 4, 3, 2, 1, U_F, U_K, U_Q, false, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Ultra, false, true) => self
                .process_tick16_const::<10, 35, 4, 3, 2, 1, U_F, U_K, U_Q, false, true>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Ultra, true, false) => self
                .process_tick16_const::<10, 35, 4, 3, 2, 1, U_F, U_K, U_Q, true, false>(
                    in_l, in_r, out_l, out_r,
                ),
            (Tier::Ultra, true, true) => self
                .process_tick16_const::<10, 35, 4, 3, 2, 1, U_F, U_K, U_Q, true, true>(
                    in_l, in_r, out_l, out_r,
                ),
        }
    }

    /// Processes a contiguous sub-block with optional `stage_targets` & `z_stage` (`0` nested FTZ guard).
    #[inline(always)]
    pub(crate) fn process_subblock_with_z_stage_no_ftz(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
        ctrl: Option<(&[f32; 16], &[f32; 8])>,
    ) {
        if let Some((stage_targets, z_stage)) = ctrl {
            self.apply_control_bus_with_z_stage(stage_targets, z_stage);
        }
        self.dispatch_block_no_ftz(in_l, in_r, out_l, out_r);
    }

    /// Processes a full stereo block (`16`-frame register-hoisted chunks + [`FtzDazGuard`]).
    #[inline(never)]
    pub fn process_block_stereo(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let _guard = FtzDazGuard::new();
        self.cold.max_op_n_drive = 0.0;
        self.dispatch_block_no_ftz(in_l, in_r, out_l, out_r);
    }

    /// Infallible stereo block entrypoint returning `Result<(), !>` (`REQ-6.2`).
    #[inline(never)]
    pub fn process_block_infallible(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), !> {
        self.process_block_stereo(in_l, in_r, out_l, out_r);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::sona_core::model::{SonaColdTierModel, SonaMatryoshkaModel};
    use crate::math::DefaultMathBackend;
    use crate::types::FtzDazGuard;

    /// verify[dsp.crossbar.top-of-frame]
    #[test]
    fn test_op_p_skipped_when_crossbar_inactive_and_head_peel_parity() {
        let _g = FtzDazGuard::new();
        let cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut s_step = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        let mut s_blk = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);

        // Advance both by 5 samples so sample_tick is unaligned (5 & 15 == 5)
        for i in 0..5 {
            let x = 0.3 * ((i as f32) * 0.17).sin();
            assert_eq!(s_step.step_stereo(x, -x), s_blk.step_stereo(x, -x));
        }
        // Inactive crossbar -> t1_dc state stayed 0.0
        assert_eq!(
            s_step.lite.t1_dc.params().0,
            dnam_kernels::DC_BLOCKER_5HZ_48K_F32
        );
        let in_l = [0.2_f32; 25];
        let in_r = [-0.2_f32; 25];
        let (mut bl, mut br) = ([0.0_f32; 25], [0.0_f32; 25]);
        s_blk.process_block_stereo(&in_l, &in_r, &mut bl, &mut br);
        for i in 0..25 {
            let st = s_step.step_stereo(in_l[i], in_r[i]);
            assert!((st[0] - bl[i]).abs() < 1.0e-6);
            assert!((st[1] - br[i]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_control_bus_superseding_core_mods_and_switch_tier_xfade() {
        let _g = FtzDazGuard::new();
        let mut mat = SonaMatryoshkaModel::identity(false, false);
        mat.ctrl.w_svf_g_oct[0] = 0.5;
        mat.ctrl.w_adaa_drive_scale[7][9] = 0.75;
        mat.ctrl.core_mods_present = true;
        mat.ctrl.set_core_mod(7, 36, 0.5);
        mat.ctrl.set_core_mod(7, 40 + 36, 0.25);
        mat.ctrl.set_core_mod(7, 80 + 36, 0.5);

        let full = mat.extract_tier(Tier::Full);
        let ultra = mat.extract_tier(Tier::Ultra);
        let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&full);
        let z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4];
        slab.snap_control_bus_with_z_stage(&[0.0; 16], &z);
        assert!(!slab.cold.ctrl_state.core_mods_active);

        slab.switch_tier(&ultra);
        slab.snap_control_bus_with_z_stage(&[0.0; 16], &z);
        assert!(slab.cold.ctrl_state.core_mods_active);
        assert!((slab.cold.ctrl_state.lane_wf_delta[36] - 0.2).abs() < 1.0e-3);
        for _ in 0..64 {
            let y = slab.step_stereo(0.25, -0.25);
            assert!(y[0].is_finite() && y[1].is_finite());
        }
        assert!((slab.lite.flags & crate::graph::sona_core::slab::FLAG_XFADE_ACTIVE) == 0);
    }

    #[test]
    fn test_tick16_mono_os2x_4port_and_empty_subblock() {
        let _g = FtzDazGuard::new();
        for (tier, os2x) in [
            (Tier::Lite, true),
            (Tier::Full, false),
            (Tier::Full, true),
            (Tier::Pro, false),
            (Tier::Pro, true),
        ] {
            let cold = SonaColdTierModel::identity_for_tier(tier, os2x, true);
            let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
            let y = slab.step_stereo(0.25, -0.15);
            assert!(y[0].is_finite() && y[0].to_bits() == y[1].to_bits());
        }

        let mut router = crate::graph::IoRouter::new();
        for (tier, os2x) in [(Tier::Lite, true), (Tier::Full, false), (Tier::Pro, false)] {
            let cold = SonaColdTierModel::identity_for_tier(tier, os2x, false);
            let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
            let (out0, core0, send0) =
                slab.step_4port_frame(&mut router, 0.3, -0.2, 0.1, -0.05, false);
            let (out1, core1, send1) =
                slab.step_4port_frame(&mut router, 0.3, -0.2, 0.1, -0.05, true);
            assert!(
                out0.iter()
                    .chain(core0.iter())
                    .chain(send0.iter())
                    .all(|v| v.is_finite())
            );
            assert!(
                out1.iter()
                    .chain(core1.iter())
                    .chain(send1.iter())
                    .all(|v| v.is_finite())
            );
        }

        let cold_lite = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut slab_empty = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold_lite);
        let mut sc = CabinetSubblockScratch::<16>::zeroed();
        slab_empty.step_tick_subblock::<4, 3, 1, 1, 1, 1, 2, 2, 1, false, false>(
            &[],
            &[],
            &mut [],
            &mut [],
            &mut sc,
        );
        assert_eq!(slab_empty.sample_tick(), 0);
    }
}
