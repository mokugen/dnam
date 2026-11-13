//! Stage II V2 Wiener-Hammerstein Post-Filtered & Pure-LTI Orthonormal Laguerre Basis (`REQ-5.6`, `[STAB-ZERO-DC]`).
//!
//! Strictly `0 FDIV`, `0 FSQRT`, `0 libm`, `0 alloc`, and `0`-ULP tier causality across `Lite..Ultra`.

use crate::container::ZohRescaledParams;
use crate::math::flush_recursive_state_f32;
use crate::operators::op_m::fast_rsqrt_nr2;

/// Number of stereo channel-paired 2-pole post-shaper LPF blocks (`2 * 16` = 32 blocks L+R).
const NUM_POST_LPF_PER_CH: usize = 20;
/// Number of stereo DC-blocked polynomial even waveshaper blocks per channel.
const NUM_EVEN_PER_CH: usize = 4;

/// Extended V2 Stage II feature-bus state (`Pure-LTI` 3-scale orthonormal Laguerre bank +
/// `Wiener-Hammerstein` 2-pole post-filtered multi-drive/asymmetric waveshaper + dispersion ladders).
#[derive(Clone, Debug)]
pub struct Stage2V2BasisState {
    // --- Pure-LTI Orthonormal Laguerre Bank (p0=0.94, p1=0.72, p2=0.30 at 48 kHz) ---
    pub base_lag_p: [f32; 3],
    pub lag_p: [f32; 3],
    pub lag_b: [f32; 3],
    pub lag0_l: [f32; 16],
    pub lag0_r: [f32; 16],
    pub lag1_l: [f32; 16],
    pub lag1_r: [f32; 16],
    pub lag2_l: [f32; 25],
    pub lag2_r: [f32; 25],

    // --- Wiener-Hammerstein Nonlinear Post-Filter Bank ---
    // Cutoffs ~[240 Hz, 380 Hz, 1.4 kHz, 4.2 kHz, 7.5 kHz] at 48 kHz
    pub base_wh_lpf: [f32; 5],
    pub wh_lpf: [f32; 5],
    pub wh_dc_r: f32,
    pub wh_disp_p: [f32; 2],
    pub wh_disp_g: [f32; 2],
    pub post_lpf_l: [[f32; 2]; NUM_POST_LPF_PER_CH],
    pub post_lpf_r: [[f32; 2]; NUM_POST_LPF_PER_CH],
    pub even_dc_l: [[f32; 2]; NUM_EVEN_PER_CH],
    pub even_dc_r: [[f32; 2]; NUM_EVEN_PER_CH],
    pub disp_fast_l: [f32; 8],
    pub disp_fast_r: [f32; 8],
    pub disp_slow_l: [f32; 8],
    pub disp_slow_r: [f32; 8],
}

impl Default for Stage2V2BasisState {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage2V2BasisState {
    pub const DEFAULT_LAG_P: [f32; 3] = [0.94, 0.64, 0.25];
    pub const DEFAULT_WH_LPF: [f32; 5] = [0.96907, 0.95146, 0.83256, 0.57710, 0.37466];
    pub const DEFAULT_DISP_P: [f32; 2] = [0.65, 0.91];

    pub const fn new() -> Self {
        Self {
            base_lag_p: Self::DEFAULT_LAG_P,
            lag_p: Self::DEFAULT_LAG_P,
            lag_b: [0.341_174_45, 0.768_374_9, 0.968_245_84],
            lag0_l: [0.0; 16],
            lag0_r: [0.0; 16],
            lag1_l: [0.0; 16],
            lag1_r: [0.0; 16],
            lag2_l: [0.0; 25],
            lag2_r: [0.0; 25],
            base_wh_lpf: Self::DEFAULT_WH_LPF,
            wh_lpf: Self::DEFAULT_WH_LPF,
            wh_dc_r: 0.9993,
            wh_disp_p: Self::DEFAULT_DISP_P,
            wh_disp_g: [0.759_934_2, 0.414_608_24],
            post_lpf_l: [[0.0; 2]; NUM_POST_LPF_PER_CH],
            post_lpf_r: [[0.0; 2]; NUM_POST_LPF_PER_CH],
            even_dc_l: [[0.0; 2]; NUM_EVEN_PER_CH],
            even_dc_r: [[0.0; 2]; NUM_EVEN_PER_CH],
            disp_fast_l: [0.0; 8],
            disp_fast_r: [0.0; 8],
            disp_slow_l: [0.0; 8],
            disp_slow_r: [0.0; 8],
        }
    }

    pub fn reset(&mut self) {
        self.lag0_l = [0.0; 16];
        self.lag0_r = [0.0; 16];
        self.lag1_l = [0.0; 16];
        self.lag1_r = [0.0; 16];
        self.lag2_l = [0.0; 25];
        self.lag2_r = [0.0; 25];
        self.post_lpf_l = [[0.0; 2]; NUM_POST_LPF_PER_CH];
        self.post_lpf_r = [[0.0; 2]; NUM_POST_LPF_PER_CH];
        self.even_dc_l = [[0.0; 2]; NUM_EVEN_PER_CH];
        self.even_dc_r = [[0.0; 2]; NUM_EVEN_PER_CH];
        self.disp_fast_l = [0.0; 8];
        self.disp_fast_r = [0.0; 8];
        self.disp_slow_l = [0.0; 8];
        self.disp_slow_r = [0.0; 8];
    }

    pub fn reconfigure_sample_rate(&mut self, zoh: &ZohRescaledParams) {
        for s in 0..3 {
            let p = zoh.rescale_real_pole(self.base_lag_p[s]).clamp(0.05, 0.995);
            self.lag_p[s] = p;
            let om = (1.0 - p * p).max(1.0e-6);
            self.lag_b[s] = om * fast_rsqrt_nr2(om);
        }
        for i in 0..5 {
            self.wh_lpf[i] = zoh
                .rescale_real_pole(self.base_wh_lpf[i])
                .clamp(0.02, 0.995);
        }
        self.wh_dc_r = zoh.r_ac_host;
        for i in 0..2 {
            let p = zoh
                .rescale_real_pole(Self::DEFAULT_DISP_P[i])
                .clamp(0.05, 0.995);
            self.wh_disp_p[i] = p;
            let om = (1.0 - p * p).max(1.0e-6);
            self.wh_disp_g[i] = om * fast_rsqrt_nr2(om);
        }
    }

    #[inline(always)]
    fn step_lag_head(state: &mut [f32], len: usize, p: f32, b: f32, x: f32) -> (f32, f32) {
        let mut prev_old = state[0];
        let next0 = flush_recursive_state_f32(p * prev_old + b * x);
        state[0] = next0;
        let mut prev_new = next0;
        for item in state.iter_mut().take(len).skip(1) {
            let cur_old = *item;
            let nxt = flush_recursive_state_f32(p * cur_old + prev_old - p * prev_new);
            *item = nxt;
            prev_old = cur_old;
            prev_new = nxt;
        }
        (prev_old, prev_new)
    }

    #[inline(always)]
    fn step_lag_cont(
        state: &mut [f32],
        start: usize,
        end: usize,
        p: f32,
        mut prev_old: f32,
        mut prev_new: f32,
    ) -> (f32, f32) {
        for item in &mut state[start..end] {
            let cur_old = *item;
            let nxt = flush_recursive_state_f32(p * cur_old + prev_old - p * prev_new);
            *item = nxt;
            prev_old = cur_old;
            prev_new = nxt;
        }
        (prev_old, prev_new)
    }

    #[inline(always)]
    fn step_2pole_lpf(state: &mut [f32; 2], a: f32, x: f32) -> f32 {
        let om = 1.0 - a;
        let y1 = flush_recursive_state_f32(a * state[0] + om * x);
        let y2 = flush_recursive_state_f32(a * state[1] + om * y1);
        state[0] = y1;
        state[1] = y2;
        y2
    }

    #[inline(always)]
    fn soft_odd_c2(x: f32) -> f32 {
        let c = x.clamp(-1.0, 1.0);
        let c2 = c * c;
        c * (1.5 - 0.5 * c2)
    }

    #[inline(always)]
    fn step_dc_even(dc: &mut [f32; 2], r_dc: f32, x: f32) -> f32 {
        let c = x.clamp(-1.5, 1.5);
        let c2 = c * c;
        let raw = c2 * (1.0 - 0.222_222_22 * c2);
        let alpha = 0.5 * (1.0 + r_dc);
        let y = flush_recursive_state_f32(alpha * (raw - dc[0]) + r_dc * dc[1]);
        dc[0] = flush_recursive_state_f32(raw);
        dc[1] = y;
        y
    }

    /// Steps the Pure-LTI 3-scale orthonormal stereo Laguerre bank (`59` states/ch = `118` slots).
    /// Zero nonlinear distortion, exact `0.0` at DC equilibrium, and `0`-ULP `REQ-5.6` isolation.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn step_pure_lti_into_fb(
        &mut self,
        x_l: f32,
        x_r: f32,
        l1_mid_re_l: f32,
        l1_mid_re_r: f32,
        l2_mid_re_l: f32,
        l2_mid_re_r: f32,
        role_a_scales: &[[f32; 16]; 4],
        branch_tiers: usize,
        fb: &mut [f32; 256],
    ) {
        let [p0, p1, p2] = self.lag_p;
        let [b0, b1, b2] = self.lag_b;

        // Tier 0 — Lite (fb[20..52], 32 floats: 16 L + 16 R)
        let (o0_l, n0_l) = Self::step_lag_head(&mut self.lag0_l, 6, p0, b0, x_l);
        let (o0_r, n0_r) = Self::step_lag_head(&mut self.lag0_r, 6, p0, b0, x_r);
        let (o1_l, n1_l) = Self::step_lag_head(&mut self.lag1_l, 5, p1, b1, x_l);
        let (o1_r, n1_r) = Self::step_lag_head(&mut self.lag1_r, 5, p1, b1, x_r);
        let (o2_l, n2_l) = Self::step_lag_head(&mut self.lag2_l, 5, p2, b2, x_l);
        let (o2_r, n2_r) = Self::step_lag_head(&mut self.lag2_r, 5, p2, b2, x_r);

        let ra0 = &role_a_scales[0];
        let l_lite = [
            x_l,
            self.lag0_l[0],
            self.lag0_l[1],
            self.lag0_l[2],
            self.lag1_l[0],
            self.lag1_l[1],
            self.lag1_l[2],
            self.lag2_l[0],
            self.lag0_l[3],
            self.lag0_l[4],
            self.lag0_l[5],
            self.lag1_l[3],
            self.lag1_l[4],
            self.lag2_l[1],
            self.lag2_l[2],
            self.lag2_l[3],
        ];
        let r_lite = [
            x_r,
            self.lag0_r[0],
            self.lag0_r[1],
            self.lag0_r[2],
            self.lag1_r[0],
            self.lag1_r[1],
            self.lag1_r[2],
            self.lag2_r[0],
            self.lag0_r[3],
            self.lag0_r[4],
            self.lag0_r[5],
            self.lag1_r[3],
            self.lag1_r[4],
            self.lag2_r[1],
            self.lag2_r[2],
            self.lag2_r[3],
        ];
        for i in 0..8 {
            fb[20 + i] = l_lite[i] * ra0[i];
            fb[28 + i] = r_lite[i] * ra0[8 + i];
        }
        fb[36..44].copy_from_slice(&l_lite[8..16]);
        fb[44..52].copy_from_slice(&r_lite[8..16]);

        if branch_tiers >= 2 {
            // Tier 1 — Full (fb[84..116], 32 floats)
            let (o0f_l, n0f_l) = Self::step_lag_cont(&mut self.lag0_l, 6, 11, p0, o0_l, n0_l);
            let (o0f_r, n0f_r) = Self::step_lag_cont(&mut self.lag0_r, 6, 11, p0, o0_r, n0_r);
            let (o1f_l, n1f_l) = Self::step_lag_cont(&mut self.lag1_l, 5, 10, p1, o1_l, n1_l);
            let (o1f_r, n1f_r) = Self::step_lag_cont(&mut self.lag1_r, 5, 10, p1, o1_r, n1_r);
            let (o2f_l, n2f_l) = Self::step_lag_cont(&mut self.lag2_l, 5, 9, p2, o2_l, n2_l);
            let (o2f_r, n2f_r) = Self::step_lag_cont(&mut self.lag2_r, 5, 9, p2, o2_r, n2_r);

            let ra1 = &role_a_scales[1];
            let full_head = [
                self.lag0_l[6] * ra1[0],
                self.lag0_l[7] * ra1[1],
                self.lag0_l[8] * ra1[2],
                self.lag1_l[5] * ra1[3],
                self.lag1_l[6] * ra1[4],
                l1_mid_re_l * ra1[5],
                self.lag2_l[5] * ra1[6],
                self.lag2_l[6] * ra1[7],
                self.lag0_r[6] * ra1[8],
                self.lag0_r[7] * ra1[9],
                self.lag0_r[8] * ra1[10],
                self.lag1_r[5] * ra1[11],
                self.lag1_r[6] * ra1[12],
                l1_mid_re_r * ra1[13],
                self.lag2_r[5] * ra1[14],
                self.lag2_r[6] * ra1[15],
            ];
            fb[84..100].copy_from_slice(&full_head);
            fb[100..116].copy_from_slice(&[
                self.lag0_l[9],
                self.lag0_l[10],
                self.lag1_l[7],
                self.lag1_l[8],
                self.lag1_l[9],
                l2_mid_re_l,
                self.lag2_l[7],
                self.lag2_l[8],
                self.lag0_r[9],
                self.lag0_r[10],
                self.lag1_r[7],
                self.lag1_r[8],
                self.lag1_r[9],
                l2_mid_re_r,
                self.lag2_r[7],
                self.lag2_r[8],
            ]);

            if branch_tiers >= 3 {
                // Tier 2 — Pro (fb[152..184], 32 floats)
                let _ = Self::step_lag_cont(&mut self.lag0_l, 11, 16, p0, o0f_l, n0f_l);
                let _ = Self::step_lag_cont(&mut self.lag0_r, 11, 16, p0, o0f_r, n0f_r);
                let _ = Self::step_lag_cont(&mut self.lag1_l, 10, 16, p1, o1f_l, n1f_l);
                let _ = Self::step_lag_cont(&mut self.lag1_r, 10, 16, p1, o1f_r, n1f_r);
                let (o2p_l, n2p_l) = Self::step_lag_cont(&mut self.lag2_l, 9, 14, p2, o2f_l, n2f_l);
                let (o2p_r, n2p_r) = Self::step_lag_cont(&mut self.lag2_r, 9, 14, p2, o2f_r, n2f_r);

                let ra2 = &role_a_scales[2];
                fb[152..168].copy_from_slice(&[
                    self.lag0_l[11] * ra2[0],
                    self.lag0_l[12] * ra2[1],
                    self.lag0_l[13] * ra2[2],
                    self.lag1_l[10] * ra2[3],
                    self.lag1_l[11] * ra2[4],
                    self.lag1_l[12] * ra2[5],
                    self.lag2_l[9] * ra2[6],
                    self.lag2_l[10] * ra2[7],
                    self.lag0_r[11] * ra2[8],
                    self.lag0_r[12] * ra2[9],
                    self.lag0_r[13] * ra2[10],
                    self.lag1_r[10] * ra2[11],
                    self.lag1_r[11] * ra2[12],
                    self.lag1_r[12] * ra2[13],
                    self.lag2_r[9] * ra2[14],
                    self.lag2_r[10] * ra2[15],
                ]);
                fb[168..184].copy_from_slice(&[
                    self.lag0_l[14],
                    self.lag0_l[15],
                    self.lag1_l[13],
                    self.lag1_l[14],
                    self.lag1_l[15],
                    self.lag2_l[11],
                    self.lag2_l[12],
                    self.lag2_l[13],
                    self.lag0_r[14],
                    self.lag0_r[15],
                    self.lag1_r[13],
                    self.lag1_r[14],
                    self.lag1_r[15],
                    self.lag2_r[11],
                    self.lag2_r[12],
                    self.lag2_r[13],
                ]);

                if branch_tiers >= 4 {
                    // Tier 3 — Ultra (fb[224..246], 22 floats)
                    let _ = Self::step_lag_cont(&mut self.lag2_l, 14, 25, p2, o2p_l, n2p_l);
                    let _ = Self::step_lag_cont(&mut self.lag2_r, 14, 25, p2, o2p_r, n2p_r);
                    let ra3 = &role_a_scales[3];
                    fb[224..240].copy_from_slice(&[
                        self.lag2_l[14] * ra3[0],
                        self.lag2_l[15] * ra3[1],
                        self.lag2_l[16] * ra3[2],
                        self.lag2_l[17] * ra3[3],
                        self.lag2_l[18] * ra3[4],
                        self.lag2_l[19] * ra3[5],
                        self.lag2_l[20] * ra3[6],
                        self.lag2_l[21] * ra3[7],
                        self.lag2_r[14] * ra3[8],
                        self.lag2_r[15] * ra3[9],
                        self.lag2_r[16] * ra3[10],
                        self.lag2_r[17] * ra3[11],
                        self.lag2_r[18] * ra3[12],
                        self.lag2_r[19] * ra3[13],
                        self.lag2_r[20] * ra3[14],
                        self.lag2_r[21] * ra3[15],
                    ]);
                    fb[240..246].copy_from_slice(&[
                        self.lag2_l[22],
                        self.lag2_l[23],
                        self.lag2_l[24],
                        self.lag2_r[22],
                        self.lag2_r[23],
                        self.lag2_r[24],
                    ]);
                }
            }
        }
    }
}

type Stage2BranchRes = [((f32, f32, f32), (f32, f32, f32), f32, f32); 3];

impl Stage2V2BasisState {
    /// Steps the Wiener-Hammerstein post-filtered nonlinear feature basis into `fb` (`REQ-5.6`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn step_wiener_hammerstein_into_fb(
        &mut self,
        x_pre_l: f32,
        x_pre_r: f32,
        tone_l: [f32; 4],
        tone_r: [f32; 4],
        v1a_out_l: [f32; 4],
        v1a_out_r: [f32; 4],
        last_v1b_hyst: [f32; 8],
        brc1_l: f32,
        brc1_r: f32,
        l1_state_l: [f32; 6],
        l1_state_r: [f32; 6],
        l2_state_l: [f32; 6],
        l2_state_r: [f32; 6],
        branch_res: Stage2BranchRes,
        wf_diff_l: f32,
        wf_diff_r: f32,
        next_sag: f32,
        role_a_scales: &[[f32; 16]; 4],
        branch_tiers: usize,
        fb: &mut [f32; 256],
    ) {
        let [a_240, _a_380, a_1400, a_4200, a_7500] = self.wh_lpf;
        let r_dc = self.wh_dc_r;

        // Tier 0 — Lite (fb[20..52]): wideband raw + post-LPF tone, V1A lanes, V1B/hyst, multi-drive odd + DC-even
        fb[20] = tone_l[0];
        fb[21] = tone_l[1];
        fb[22] = tone_l[2];
        fb[23] = tone_l[3];
        fb[24] = v1a_out_l[0];
        fb[25] = Self::step_2pole_lpf(&mut self.post_lpf_l[0], a_7500, v1a_out_l[1]);
        fb[26] = Self::step_2pole_lpf(&mut self.post_lpf_l[1], a_4200, v1a_out_l[2]);
        fb[27] = Self::step_2pole_lpf(&mut self.post_lpf_l[2], a_1400, v1a_out_l[0]);

        fb[28] = tone_r[0];
        fb[29] = tone_r[1];
        fb[30] = tone_r[2];
        fb[31] = tone_r[3];
        fb[32] = v1a_out_r[0];
        fb[33] = Self::step_2pole_lpf(&mut self.post_lpf_r[0], a_7500, v1a_out_r[1]);
        fb[34] = Self::step_2pole_lpf(&mut self.post_lpf_r[1], a_4200, v1a_out_r[2]);
        fb[35] = Self::step_2pole_lpf(&mut self.post_lpf_r[2], a_1400, v1a_out_r[0]);

        let [v1b_l1, v1b_l2, h_l, _wf_l, v1b_r1, v1b_r2, h_r, _wf_r] = last_v1b_hyst;
        let ev1_l = Self::step_dc_even(&mut self.even_dc_l[0], r_dc, 1.2 * x_pre_l);
        let ev1_r = Self::step_dc_even(&mut self.even_dc_r[0], r_dc, 1.2 * x_pre_r);
        let ev2_l = Self::step_dc_even(&mut self.even_dc_l[1], r_dc, 2.5 * tone_l[2]);
        let ev2_r = Self::step_dc_even(&mut self.even_dc_r[1], r_dc, 2.5 * tone_r[2]);

        fb[36..44].copy_from_slice(&[
            v1b_l1,
            Self::step_2pole_lpf(&mut self.post_lpf_l[6], a_4200, v1b_l1),
            Self::step_2pole_lpf(&mut self.post_lpf_l[7], a_1400, v1b_l2),
            Self::step_2pole_lpf(
                &mut self.post_lpf_l[8],
                a_7500,
                Self::soft_odd_c2(0.85 * x_pre_l),
            ),
            v1b_r1,
            Self::step_2pole_lpf(&mut self.post_lpf_r[6], a_4200, v1b_r1),
            Self::step_2pole_lpf(&mut self.post_lpf_r[7], a_1400, v1b_r2),
            Self::step_2pole_lpf(
                &mut self.post_lpf_r[8],
                a_7500,
                Self::soft_odd_c2(0.85 * x_pre_r),
            ),
        ]);
        fb[44..52].copy_from_slice(&[
            Self::step_2pole_lpf(
                &mut self.post_lpf_l[9],
                a_4200,
                Self::soft_odd_c2(2.2 * x_pre_l),
            ),
            Self::step_2pole_lpf(&mut self.post_lpf_l[10], a_4200, ev1_l),
            Self::step_2pole_lpf(&mut self.post_lpf_l[11], a_1400, ev2_l),
            brc1_l,
            Self::step_2pole_lpf(
                &mut self.post_lpf_r[9],
                a_4200,
                Self::soft_odd_c2(2.2 * x_pre_r),
            ),
            Self::step_2pole_lpf(&mut self.post_lpf_r[10], a_4200, ev1_r),
            Self::step_2pole_lpf(&mut self.post_lpf_r[11], a_1400, ev2_r),
            brc1_r,
        ]);

        if branch_tiers >= 2 {
            // Tier 1 — Full (fb[84..116])
            let [_, hpf1_l, lpf1_l, mid1_re_l, mid1_im_l, _] = l1_state_l;
            let [_, hpf1_r, lpf1_r, mid1_re_r, mid1_im_r, _] = l1_state_r;
            let [x1_l, x1_r] = [l1_state_l[0], l1_state_r[0]];
            let [x2_l, hpf2_l, lpf2_l, mid2_re_l, mid2_im_l, brc2_l] = l2_state_l;
            let [x2_r, hpf2_r, lpf2_r, mid2_re_r, mid2_im_r, brc2_r] = l2_state_r;
            let sh1_l = x1_l - lpf1_l;
            let sh1_r = x1_r - lpf1_r;
            let sh2_l = x2_l - lpf2_l;
            let sh2_r = x2_r - lpf2_r;
            let ((v2a_nl_l, v2a_lpf_l, _v2a_hpf_l), (v2a_nl_r, v2a_lpf_r, _v2a_hpf_r), _, _) =
                branch_res[0];

            let v2a_f_l = Self::step_2pole_lpf(&mut self.post_lpf_l[13], a_4200, v2a_nl_l);
            let v2a_f_r = Self::step_2pole_lpf(&mut self.post_lpf_r[13], a_4200, v2a_nl_r);
            let pwr_bp_l = Self::step_2pole_lpf(&mut self.post_lpf_l[15], a_4200, tone_l[0]);
            let pwr_bp_r = Self::step_2pole_lpf(&mut self.post_lpf_r[15], a_4200, tone_r[0]);

            let _ = Self::step_lag_head(
                &mut self.disp_fast_l,
                8,
                self.wh_disp_p[0],
                self.wh_disp_g[0],
                pwr_bp_l,
            );
            let _ = Self::step_lag_head(
                &mut self.disp_fast_r,
                8,
                self.wh_disp_p[0],
                self.wh_disp_g[0],
                pwr_bp_r,
            );

            let ra1 = &role_a_scales[1];
            fb[84..116].copy_from_slice(&[
                v2a_nl_l * ra1[0],
                v2a_f_l * ra1[1],
                v2a_lpf_l * ra1[2],
                hpf1_l * ra1[3],
                sh1_l * ra1[4],
                mid1_re_l * ra1[5],
                mid1_im_l * ra1[6],
                hpf2_l * ra1[7],
                v2a_nl_r * ra1[8],
                v2a_f_r * ra1[9],
                v2a_lpf_r * ra1[10],
                hpf1_r * ra1[11],
                sh1_r * ra1[12],
                mid1_re_r * ra1[13],
                mid1_im_r * ra1[14],
                hpf2_r * ra1[15],
                brc2_l,
                mid2_re_l,
                mid2_im_l,
                sh2_l,
                self.disp_fast_l[0],
                self.disp_fast_l[1],
                self.disp_fast_l[2],
                self.disp_fast_l[3],
                brc2_r,
                mid2_re_r,
                mid2_im_r,
                sh2_r,
                self.disp_fast_r[0],
                self.disp_fast_r[1],
                self.disp_fast_r[2],
                self.disp_fast_r[3],
            ]);

            if branch_tiers >= 3 {
                // Tier 2 — Pro (fb[152..184])
                let (
                    (v2b_nl_l, _v2b_lpf_l, v2b_hpf_l),
                    (v2b_nl_r, _v2b_lpf_r, v2b_hpf_r),
                    wf2_l,
                    wf2_r,
                ) = branch_res[1];
                let v2b_f_l = Self::step_2pole_lpf(&mut self.post_lpf_l[16], a_4200, v2b_nl_l);
                let v2b_f_r = Self::step_2pole_lpf(&mut self.post_lpf_r[16], a_4200, v2b_nl_r);
                let wf2_f_l = Self::step_2pole_lpf(&mut self.post_lpf_l[17], a_4200, wf2_l);
                let wf2_f_r = Self::step_2pole_lpf(&mut self.post_lpf_r[17], a_4200, wf2_r);
                let bass_odd_l = Self::step_2pole_lpf(
                    &mut self.post_lpf_l[18],
                    a_240,
                    Self::soft_odd_c2(2.5 * tone_l[1]),
                );
                let bass_odd_r = Self::step_2pole_lpf(
                    &mut self.post_lpf_r[18],
                    a_240,
                    Self::soft_odd_c2(2.5 * tone_r[1]),
                );
                let _ = Self::step_lag_head(
                    &mut self.disp_slow_l,
                    8,
                    self.wh_disp_p[1],
                    self.wh_disp_g[1],
                    tone_l[2],
                );
                let _ = Self::step_lag_head(
                    &mut self.disp_slow_r,
                    8,
                    self.wh_disp_p[1],
                    self.wh_disp_g[1],
                    tone_r[2],
                );
                let ev3_l = Self::step_dc_even(&mut self.even_dc_l[2], r_dc, 3.2 * pwr_bp_l);
                let ev3_r = Self::step_dc_even(&mut self.even_dc_r[2], r_dc, 3.2 * pwr_bp_r);

                let ra2 = &role_a_scales[2];
                fb[152..184].copy_from_slice(&[
                    v2b_nl_l * ra2[0],
                    v2b_f_l * ra2[1],
                    v2b_hpf_l * ra2[2],
                    wf2_f_l * ra2[3],
                    bass_odd_l * ra2[4],
                    ev3_l * ra2[5],
                    self.disp_fast_l[4] * ra2[6],
                    self.disp_fast_l[5] * ra2[7],
                    v2b_nl_r * ra2[8],
                    v2b_f_r * ra2[9],
                    v2b_hpf_r * ra2[10],
                    wf2_f_r * ra2[11],
                    bass_odd_r * ra2[12],
                    ev3_r * ra2[13],
                    self.disp_fast_r[4] * ra2[14],
                    self.disp_fast_r[5] * ra2[15],
                    self.disp_fast_l[6],
                    self.disp_fast_l[7],
                    self.disp_slow_l[0],
                    self.disp_slow_l[1],
                    self.disp_slow_l[2],
                    self.disp_slow_l[3],
                    next_sag * self.disp_fast_l[0],
                    wf_diff_l,
                    self.disp_fast_r[6],
                    self.disp_fast_r[7],
                    self.disp_slow_r[0],
                    self.disp_slow_r[1],
                    self.disp_slow_r[2],
                    self.disp_slow_r[3],
                    next_sag * self.disp_fast_r[0],
                    wf_diff_r,
                ]);

                if branch_tiers >= 4 {
                    // Tier 3 — Ultra (fb[224..246])
                    let (
                        (v3_nl_l, _v3_lpf_l, v3_hpf_l),
                        (v3_nl_r, _v3_lpf_r, v3_hpf_r),
                        wf3_l,
                        wf3_r,
                    ) = branch_res[2];
                    let v3_f_l = Self::step_2pole_lpf(&mut self.post_lpf_l[19], a_4200, v3_nl_l);
                    let v3_f_r = Self::step_2pole_lpf(&mut self.post_lpf_r[19], a_4200, v3_nl_r);
                    let ev4_l = Self::step_dc_even(&mut self.even_dc_l[3], r_dc, 1.8 * v2a_f_l);
                    let ev4_r = Self::step_dc_even(&mut self.even_dc_r[3], r_dc, 1.8 * v2a_f_r);
                    let ra3 = &role_a_scales[3];
                    fb[224..246].copy_from_slice(&[
                        v3_nl_l * ra3[0],
                        v3_f_l * ra3[1],
                        v3_hpf_l * ra3[2],
                        wf3_l * a_7500 * ra3[3],
                        ev4_l * ra3[4],
                        self.disp_slow_l[4] * ra3[5],
                        v3_nl_r * ra3[6],
                        v3_f_r * ra3[7],
                        v3_hpf_r * ra3[8],
                        wf3_r * a_7500 * ra3[9],
                        ev4_r * ra3[10],
                        self.disp_slow_r[4] * ra3[11],
                        self.disp_slow_l[5] * ra3[12],
                        v2a_f_l * h_l * ra3[13],
                        self.disp_slow_r[5] * ra3[14],
                        v2a_f_r * h_r * ra3[15],
                        self.disp_slow_l[6],
                        self.disp_slow_l[7],
                        v3_f_l * next_sag,
                        self.disp_slow_r[6],
                        self.disp_slow_r[7],
                        v3_f_r * next_sag,
                    ]);
                }
            }
        }
    }
}
