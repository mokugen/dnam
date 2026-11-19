//! Stage III Early FIR + 32-Pole Kautz Modal Bank and Nested Matryoshka Cold Models.

pub mod cold;
pub mod kautz_zoh;

pub use cold::*;
pub use kautz_zoh::{
    KautzRetuneContext, compute_onset_delay_samples, rescale_kautz_poles_with_zoh,
    retune_kautz_formant_3khz,
};

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_vec4, newton6_reciprocal_f64};
use crate::operators::op_m::KautzModalBank;
use crate::stages::router::V3Stage0Hot;
use crate::stages::stage1_svf::{V3Stage1SvfParams, V3Stage1SvfState};
use crate::stages::stage2_gru::{
    FIR_RING_CAP, FIR_RING_MASK, LOWER_TRI_TILES, MAX_BLOCK_LANES, MAX_FIR_SEGMENTS,
    MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES, V3GruGroupHot, V3InputFeaturesParams, V3InputFeaturesState,
    V3InterBlockToneSvfState, V3OversamplerParams, V3StereoOversampler2x, tier_block_groups_array,
    tier_fir_segments, tier_groups, tier_gru_blocks, tier_kautz_banks,
};
use crate::stages::stage4_tank::v3_fdn::{
    V3_TANK_LINES, V3_TANK_RING_STRIDE, V3Stage4TankHot, V3Stage4TankParams,
};
use crate::types::Tier;
use kautz_zoh::cold_b_gamma;

/// Contiguous Stage III Early FIR + 32-pole Kautz stereo cabinet parameters and state.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct V3Stage3Hot {
    pub fir_taps_l: [[[f32; 4]; 4]; MAX_FIR_SEGMENTS],
    pub fir_taps_r: [[[f32; 4]; 4]; MAX_FIR_SEGMENTS],
    pub kautz_l: [KautzModalBank<4>; MAX_KAUTZ_BANKS],
    pub kautz_r: [KautzModalBank<4>; MAX_KAUTZ_BANKS],
    pub fir_hist_l: [f32; FIR_RING_CAP * 2],
    pub fir_hist_r: [f32; FIR_RING_CAP * 2],
    pub write_pos: usize,
    pub onset_delay_samples: usize,
    pub host_kautz_poles: [(f32, f32); MAX_KAUTZ_POLES],
    pub host_kautz_b_gamma: [f32; MAX_KAUTZ_POLES],
    pub host_kautz_inv_1mr2: [f32; MAX_KAUTZ_POLES],
    pub last_dtheta_kautz: f32,
    pub last_damp_kautz: f32,
    pub prev_oct_kautz: f32,
    pub kautz_slew_active: bool,
}

impl Default for V3Stage3Hot {
    fn default() -> Self {
        Self {
            fir_taps_l: [[[0.0; 4]; 4]; MAX_FIR_SEGMENTS],
            fir_taps_r: [[[0.0; 4]; 4]; MAX_FIR_SEGMENTS],
            kautz_l: core::array::from_fn(|_| KautzModalBank::new()),
            kautz_r: core::array::from_fn(|_| KautzModalBank::new()),
            fir_hist_l: [0.0; FIR_RING_CAP * 2],
            fir_hist_r: [0.0; FIR_RING_CAP * 2],
            write_pos: 0,
            onset_delay_samples: 16,
            host_kautz_poles: [(0.90, 0.25); MAX_KAUTZ_POLES],
            host_kautz_b_gamma: [cold_b_gamma(0.90); MAX_KAUTZ_POLES],
            host_kautz_inv_1mr2: [1.0 / (1.0 - 0.81); MAX_KAUTZ_POLES],
            last_dtheta_kautz: 0.0,
            last_damp_kautz: 0.0,
            prev_oct_kautz: 0.0,
            kautz_slew_active: false,
        }
    }
}

impl V3Stage3Hot {
    /// Populates Stage III FIR taps and Kautz poles/residues from cold readout and poles.
    #[must_use]
    pub fn from_cold(model: &V3ColdTierModel) -> Self {
        let mut out = Self::default();
        out.update_readout(&model.readout, &model.kautz_poles);
        out
    }

    fn apply_fir_taps(&mut self, fir_l: &[f32; 64], fir_r: &[f32; 64]) {
        for seg in 0..MAX_FIR_SEGMENTS {
            for k4 in 0..4 {
                for i in 0..4 {
                    let idx = (seg << 4) + (k4 << 2) + i;
                    self.fir_taps_l[seg][k4][i] = fir_l[idx];
                    self.fir_taps_r[seg][k4][i] = fir_r[idx];
                }
            }
        }
    }

    fn apply_kautz_poles_and_residues(
        &mut self,
        readout: &V3TierReadout,
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    ) {
        self.apply_kautz_poles_and_residues_scaled(readout, kautz_poles, None);
    }

    fn apply_kautz_poles_and_residues_scaled(
        &mut self,
        readout: &V3TierReadout,
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
        zoh_base_poles: Option<&[(f32, f32); MAX_KAUTZ_POLES]>,
    ) {
        self.host_kautz_poles = *kautz_poles;
        self.last_dtheta_kautz = 0.0;
        self.last_damp_kautz = 0.0;
        self.prev_oct_kautz = 0.0;
        self.kautz_slew_active = false;
        for b in 0..MAX_KAUTZ_BANKS {
            for lane in 0..4 {
                let m = (b << 2) + lane;
                let (r_raw, theta_raw) = kautz_poles[m];
                let r = r_raw.clamp(0.25, 0.9995);
                let theta = theta_raw.clamp(0.0, core::f32::consts::PI - 1.0e-4);
                let bg_stored = readout.kautz_b_gamma[m];
                let b_gamma = if bg_stored > 0.0 {
                    if let Some(base_poles) = zoh_base_poles {
                        let base_r = base_poles[m].0.clamp(0.25, 0.9995);
                        let base_bg = f64::from(cold_b_gamma(base_r)).max(1.0e-12);
                        let scale = f64::from(bg_stored) / base_bg;
                        (f64::from(cold_b_gamma(r)) * scale) as f32
                    } else {
                        bg_stored
                    }
                } else {
                    cold_b_gamma(r)
                };
                self.host_kautz_b_gamma[m] = b_gamma;
                self.host_kautz_inv_1mr2[m] =
                    newton6_reciprocal_f64((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)) as f32;
                let (cl_re, cl_im) = readout.kautz_c_l[m];
                let (cr_re, cr_im) = readout.kautz_c_r[m];
                self.kautz_l[b].set_pole_with_b_gamma(lane, r, theta, 0.0, b_gamma, cl_re, cl_im);
                self.kautz_r[b].set_pole_with_b_gamma(lane, r, theta, 0.0, b_gamma, cr_re, cr_im);
            }
        }
    }

    /// Updates Stage III FIR taps and Kautz poles/residues in-place without clearing ring/bank state.
    pub fn update_readout(
        &mut self,
        readout: &V3TierReadout,
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    ) {
        self.apply_fir_taps(&readout.fir_l, &readout.fir_r);
        self.apply_kautz_poles_and_residues(readout, kautz_poles);
    }

    /// Rescales Stage III Early FIR taps, Kautz modal poles, and FIR-to-Kautz onset delay
    /// under continuous-time ZOH (`ref[dsp.zoh.op-m-poles]`, `ref[dsp.zoh.stage3-fir-onset]`).
    pub fn reconfigure_with_zoh(
        &mut self,
        readout: &V3TierReadout,
        base_poles: &[(f32, f32); MAX_KAUTZ_POLES],
        num_fir_segments: usize,
        zoh: &ZohRescaledParams,
    ) {
        self.onset_delay_samples = compute_onset_delay_samples(zoh);
        if (zoh.rho_sr - 1.0).abs() < 1.0e-12 {
            self.apply_fir_taps(&readout.fir_l, &readout.fir_r);
            self.apply_kautz_poles_and_residues(readout, base_poles);
            return;
        }
        let active_taps = (num_fir_segments * 16).min(64);
        let mut rescaled_l = [0.0_f32; 64];
        let mut rescaled_r = [0.0_f32; 64];
        if active_taps > 0 {
            zoh.rescale_fir_taps(
                &readout.fir_l[..active_taps],
                &mut rescaled_l[..active_taps],
            );
            zoh.rescale_fir_taps(
                &readout.fir_r[..active_taps],
                &mut rescaled_r[..active_taps],
            );
        }
        self.apply_fir_taps(&rescaled_l, &rescaled_r);
        let host_poles = rescale_kautz_poles_with_zoh(base_poles, zoh);
        self.apply_kautz_poles_and_residues_scaled(readout, &host_poles, Some(base_poles));
    }

    /// Snaps active Kautz poles directly to `(oct_shift, damp_delta)` without slew contraction.
    #[inline]
    pub fn snap_kautz_immediate<B: MathBackend>(
        &mut self,
        num_kautz_banks: usize,
        oct_shift: f32,
        damp_delta: f32,
    ) {
        self.prev_oct_kautz = oct_shift;
        self.last_dtheta_kautz = oct_shift;
        self.last_damp_kautz = damp_delta;
        self.kautz_slew_active = false;
        retune_kautz_formant_3khz::<B>(
            &mut self.kautz_l,
            &mut self.kautz_r,
            &KautzRetuneContext {
                host_poles: &self.host_kautz_poles,
                host_b_gamma: &self.host_kautz_b_gamma,
                host_inv_1mr2: &self.host_kautz_inv_1mr2,
                num_banks: num_kautz_banks,
                oct_shift,
                damp_delta,
                prev_oct_shift: oct_shift,
            },
        );
    }

    /// Updates active Kautz pole frequencies (`oct_shift`), radial-$Q$ damping (`damp_delta`),
    /// and sweep contraction at `3 kHz` cadence without audio-rate divisions.
    #[inline]
    pub fn retune_kautz_3khz<B: MathBackend>(
        &mut self,
        num_kautz_banks: usize,
        oct_shift: f32,
        damp_delta: f32,
    ) {
        if (oct_shift - self.last_dtheta_kautz).abs() <= 1.0e-9
            && (damp_delta - self.last_damp_kautz).abs() <= 1.0e-9
            && !self.kautz_slew_active
        {
            return;
        }
        let prev_oct_shift = self.prev_oct_kautz;
        self.kautz_slew_active = (oct_shift - prev_oct_shift).abs() > 1.0e-6;
        self.prev_oct_kautz = oct_shift;
        self.last_dtheta_kautz = oct_shift;
        self.last_damp_kautz = damp_delta;
        retune_kautz_formant_3khz::<B>(
            &mut self.kautz_l,
            &mut self.kautz_r,
            &KautzRetuneContext {
                host_poles: &self.host_kautz_poles,
                host_b_gamma: &self.host_kautz_b_gamma,
                host_inv_1mr2: &self.host_kautz_inv_1mr2,
                num_banks: num_kautz_banks,
                oct_shift,
                damp_delta,
                prev_oct_shift,
            },
        );
    }

    /// Resets the dual-mirrored FIR ring buffers and all Kautz banks to resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        self.fir_hist_l.fill(0.0);
        self.fir_hist_r.fill(0.0);
        self.write_pos = 0;
        self.last_dtheta_kautz = 0.0;
        self.last_damp_kautz = 0.0;
        self.prev_oct_kautz = 0.0;
        self.kautz_slew_active = false;
        for b in 0..MAX_KAUTZ_BANKS {
            self.kautz_l[b].reset();
            self.kautz_r[b].reset();
        }
    }

    #[inline(always)]
    fn push_fir(&mut self, s2_l: f32, s2_r: f32) -> usize {
        let pos = self.write_pos.wrapping_sub(1) & FIR_RING_MASK;
        self.write_pos = pos;
        self.fir_hist_l[pos] = s2_l;
        self.fir_hist_l[pos + FIR_RING_CAP] = s2_l;
        self.fir_hist_r[pos] = s2_r;
        self.fir_hist_r[pos + FIR_RING_CAP] = s2_r;
        pos
    }

    #[inline(always)]
    fn eval_fir_l<B: MathBackend>(&self, pos: usize, num_fir_segments: usize) -> f32 {
        let mut acc = B::Vec4::zero();
        for seg in 0..num_fir_segments.min(MAX_FIR_SEGMENTS) {
            let s = (pos & FIR_RING_MASK) + ((seg & 3) << 4);
            let (chunks, _) = self.fir_hist_l[s..s + 16].as_chunks::<4>();
            let t = &self.fir_taps_l[seg];
            for k4 in 0..4 {
                acc = acc.fma(B::Vec4::load(&chunks[k4]), B::Vec4::load(&t[k4]));
            }
        }
        acc.horizontal_sum()
    }

    #[inline(always)]
    fn eval_fir_stereo<B: MathBackend>(&self, pos: usize, num_fir_segments: usize) -> (f32, f32) {
        let mut acc_l = B::Vec4::zero();
        let mut acc_r = B::Vec4::zero();
        for seg in 0..num_fir_segments.min(MAX_FIR_SEGMENTS) {
            let s = (pos & FIR_RING_MASK) + ((seg & 3) << 4);
            let (hl_chunks, _) = self.fir_hist_l[s..s + 16].as_chunks::<4>();
            let (hr_chunks, _) = self.fir_hist_r[s..s + 16].as_chunks::<4>();
            let tl = &self.fir_taps_l[seg];
            let tr = &self.fir_taps_r[seg];
            for k4 in 0..4 {
                acc_l = acc_l.fma(B::Vec4::load(&hl_chunks[k4]), B::Vec4::load(&tl[k4]));
                acc_r = acc_r.fma(B::Vec4::load(&hr_chunks[k4]), B::Vec4::load(&tr[k4]));
            }
        }
        acc_l.horizontal_sum_pair(acc_r)
    }

    #[inline(always)]
    fn step_kautz_mono_unscaled<B: MathBackend>(
        bank: &mut KautzModalBank<4>,
        x_vec: B::Vec4,
    ) -> B::Vec4 {
        let l_re = B::Vec4::load(&bank.lambda_re);
        let l_im = B::Vec4::load(&bank.lambda_im);
        let b_g = B::Vec4::load(&bank.b_gamma);
        let c_r = B::Vec4::load(&bank.c_re);
        let c_i = B::Vec4::load(&bank.c_im);
        let prev_re = B::Vec4::load(&bank.s_re);
        let prev_im = B::Vec4::load(&bank.s_im);
        let next_re = l_re.mul(prev_re).sub(l_im.mul(prev_im)).fma(b_g, x_vec);
        let next_im = l_im.mul(prev_re).add(l_re.mul(prev_im));
        flush_recursive_state_vec4::<B::Vec4>(next_re).store(&mut bank.s_re);
        flush_recursive_state_vec4::<B::Vec4>(next_im).store(&mut bank.s_im);
        c_r.mul(next_re).sub(c_i.mul(next_im))
    }

    /// Steps Stage III FIR + Kautz modal banks for stereo inputs `(s2_l, s2_r)`.
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(
        &mut self,
        s2_l: f32,
        s2_r: f32,
        num_fir_segments: usize,
        num_kautz_banks: usize,
        mono_out: bool,
    ) -> [f32; 2] {
        let pos = self.push_fir(s2_l, s2_r);
        let onset = self.onset_delay_samples.clamp(1, FIR_RING_MASK);
        let delayed_l = self.fir_hist_l[(pos & FIR_RING_MASK) + onset];
        let delayed_r = self.fir_hist_r[(pos & FIR_RING_MASK) + onset];
        let x_l_vec = B::Vec4::splat(delayed_l);
        let x_r_vec = B::Vec4::splat(delayed_r);

        if mono_out {
            let fir_l = self.eval_fir_l::<B>(pos, num_fir_segments);
            let mut k_acc = B::Vec4::zero();
            for b in 0..num_kautz_banks.min(MAX_KAUTZ_BANKS) {
                k_acc = k_acc.add(Self::step_kautz_mono_unscaled::<B>(
                    &mut self.kautz_l[b],
                    x_l_vec,
                ));
            }
            let out_l = fir_l + k_acc.horizontal_sum();
            [out_l, out_l]
        } else {
            let (fir_l, fir_r) = self.eval_fir_stereo::<B>(pos, num_fir_segments);
            let mut k_acc_l = B::Vec4::zero();
            let mut k_acc_r = B::Vec4::zero();
            for b in 0..num_kautz_banks.min(MAX_KAUTZ_BANKS) {
                let (vl, vr) = self.kautz_l[b].step_4lane_stereo_simd_unscaled::<B>(
                    &mut self.kautz_r[b],
                    x_l_vec,
                    x_r_vec,
                );
                k_acc_l = k_acc_l.add(vl);
                k_acc_r = k_acc_r.add(vr);
            }
            let (kl, kr) = k_acc_l.horizontal_sum_pair(k_acc_r);
            [fir_l + kl, fir_r + kr]
        }
    }

    /// Sub-block FIR + bank-outer Kautz evaluation keeping each bank's coefficients and state in registers across `0..len`.
    #[inline(always)]
    pub(crate) fn step_subblock_const<
        B: MathBackend,
        const FIR_SEGS: usize,
        const KAUTZ_BANKS: usize,
        const MONO: bool,
        const CAP: usize,
    >(
        &mut self,
        s2_dir: &[[f32; 4]; CAP],
        sc: &mut Stage3SubblockScratch<CAP>,
        len: usize,
    ) {
        const { assert!(FIR_SEGS <= MAX_FIR_SEGMENTS && KAUTZ_BANKS <= MAX_KAUTZ_BANKS) };
        let n = len.min(CAP);
        if FIR_SEGS == 0 && KAUTZ_BANKS == 0 {
            sc.cab_out[..n].fill([0.0, 0.0]);
            return;
        }
        let onset = self.onset_delay_samples.clamp(1, FIR_RING_MASK);
        let pos_buf = &mut sc.pos_buf;
        let x_del_l = &mut sc.x_del_l;
        let x_del_r = &mut sc.x_del_r;
        for i in 0..n {
            let p = self.push_fir(s2_dir[i][0], s2_dir[i][1]) & FIR_RING_MASK;
            pos_buf[i] = p;
            x_del_l[i] = self.fir_hist_l[p + onset];
            if !MONO {
                x_del_r[i] = self.fir_hist_r[p + onset];
            }
        }

        let acc_l = &mut sc.acc_l;
        let acc_r = &mut sc.acc_r;
        let cab_out = &mut sc.cab_out;
        for seg in 0..FIR_SEGS {
            let off = (seg & 3) << 4;
            let tl = &self.fir_taps_l[seg];
            let (tl0, tl1, tl2, tl3) = (
                B::Vec4::load(&tl[0]),
                B::Vec4::load(&tl[1]),
                B::Vec4::load(&tl[2]),
                B::Vec4::load(&tl[3]),
            );
            if MONO {
                for i in 0..n {
                    let s = (pos_buf[i] & FIR_RING_MASK) + off;
                    let (hl, _) = self.fir_hist_l[s..s + 16].as_chunks::<4>();
                    let f0 = B::Vec4::load(&hl[0]).mul(tl0);
                    let f1 = B::Vec4::load(&hl[1]).mul(tl1);
                    let seg_l = f0
                        .fma(B::Vec4::load(&hl[2]), tl2)
                        .add(f1.fma(B::Vec4::load(&hl[3]), tl3));
                    let cur_l = if seg == 0 {
                        seg_l
                    } else {
                        B::Vec4::load(&acc_l[i]).add(seg_l)
                    };
                    cur_l.store(&mut acc_l[i]);
                }
            } else {
                let tr = &self.fir_taps_r[seg];
                let (tr0, tr1, tr2, tr3) = (
                    B::Vec4::load(&tr[0]),
                    B::Vec4::load(&tr[1]),
                    B::Vec4::load(&tr[2]),
                    B::Vec4::load(&tr[3]),
                );
                for i in 0..n {
                    let s = (pos_buf[i] & FIR_RING_MASK) + off;
                    let (hl, _) = self.fir_hist_l[s..s + 16].as_chunks::<4>();
                    let (hr, _) = self.fir_hist_r[s..s + 16].as_chunks::<4>();
                    let fl0 = B::Vec4::load(&hl[0]).mul(tl0);
                    let fl1 = B::Vec4::load(&hl[1]).mul(tl1);
                    let seg_l = fl0
                        .fma(B::Vec4::load(&hl[2]), tl2)
                        .add(fl1.fma(B::Vec4::load(&hl[3]), tl3));
                    let cur_l = if seg == 0 {
                        seg_l
                    } else {
                        B::Vec4::load(&acc_l[i]).add(seg_l)
                    };
                    cur_l.store(&mut acc_l[i]);

                    let fr0 = B::Vec4::load(&hr[0]).mul(tr0);
                    let fr1 = B::Vec4::load(&hr[1]).mul(tr1);
                    let seg_r = fr0
                        .fma(B::Vec4::load(&hr[2]), tr2)
                        .add(fr1.fma(B::Vec4::load(&hr[3]), tr3));
                    let cur_r = if seg == 0 {
                        seg_r
                    } else {
                        B::Vec4::load(&acc_r[i]).add(seg_r)
                    };
                    cur_r.store(&mut acc_r[i]);
                }
            }
        }

        for b in 0..KAUTZ_BANKS {
            let first_write = FIR_SEGS == 0 && b == 0;
            let bank_l = &mut self.kautz_l[b];
            let l_re = B::Vec4::load(&bank_l.lambda_re);
            let l_im = B::Vec4::load(&bank_l.lambda_im);
            let b_g = B::Vec4::load(&bank_l.b_gamma);
            let c_r_l = B::Vec4::load(&bank_l.c_re);
            let c_i_l = B::Vec4::load(&bank_l.c_im);
            let mut s_re_l = B::Vec4::load(&bank_l.s_re);
            let mut s_im_l = B::Vec4::load(&bank_l.s_im);

            if MONO {
                for i in 0..n {
                    let x_v = B::Vec4::splat(x_del_l[i]);
                    let next_re = l_re.mul(s_re_l).sub(l_im.mul(s_im_l)).fma(b_g, x_v);
                    let next_im = l_im.mul(s_re_l).add(l_re.mul(s_im_l));
                    s_re_l = flush_recursive_state_vec4::<B::Vec4>(next_re);
                    s_im_l = flush_recursive_state_vec4::<B::Vec4>(next_im);
                    let y_l = c_r_l.mul(next_re).sub(c_i_l.mul(next_im));
                    let cur_l = if first_write {
                        y_l
                    } else {
                        B::Vec4::load(&acc_l[i]).add(y_l)
                    };
                    cur_l.store(&mut acc_l[i]);
                }
                s_re_l.store(&mut bank_l.s_re);
                s_im_l.store(&mut bank_l.s_im);
            } else {
                let bank_r = &mut self.kautz_r[b];
                let c_r_r = B::Vec4::load(&bank_r.c_re);
                let c_i_r = B::Vec4::load(&bank_r.c_im);
                let mut s_re_r = B::Vec4::load(&bank_r.s_re);
                let mut s_im_r = B::Vec4::load(&bank_r.s_im);

                for i in 0..n {
                    let xl_v = B::Vec4::splat(x_del_l[i]);
                    let xr_v = B::Vec4::splat(x_del_r[i]);

                    let next_re_l = l_re.mul(s_re_l).sub(l_im.mul(s_im_l)).fma(b_g, xl_v);
                    let next_im_l = l_im.mul(s_re_l).add(l_re.mul(s_im_l));
                    let next_re_r = l_re.mul(s_re_r).sub(l_im.mul(s_im_r)).fma(b_g, xr_v);
                    let next_im_r = l_im.mul(s_re_r).add(l_re.mul(s_im_r));

                    s_re_l = flush_recursive_state_vec4::<B::Vec4>(next_re_l);
                    s_im_l = flush_recursive_state_vec4::<B::Vec4>(next_im_l);
                    s_re_r = flush_recursive_state_vec4::<B::Vec4>(next_re_r);
                    s_im_r = flush_recursive_state_vec4::<B::Vec4>(next_im_r);

                    let y_l = c_r_l.mul(next_re_l).sub(c_i_l.mul(next_im_l));
                    let y_r = c_r_r.mul(next_re_r).sub(c_i_r.mul(next_im_r));
                    if first_write {
                        y_l.store(&mut acc_l[i]);
                        y_r.store(&mut acc_r[i]);
                    } else {
                        B::Vec4::load(&acc_l[i]).add(y_l).store(&mut acc_l[i]);
                        B::Vec4::load(&acc_r[i]).add(y_r).store(&mut acc_r[i]);
                    }
                }
                s_re_l.store(&mut bank_l.s_re);
                s_im_l.store(&mut bank_l.s_im);
                s_re_r.store(&mut bank_r.s_re);
                s_im_r.store(&mut bank_r.s_im);
            }
        }

        for i in 0..n {
            if MONO {
                let y = B::Vec4::load(&acc_l[i]).horizontal_sum();
                cab_out[i] = [y, y];
            } else {
                let (kl, kr) =
                    B::Vec4::load(&acc_l[i]).horizontal_sum_pair(B::Vec4::load(&acc_r[i]));
                cab_out[i] = [kl, kr];
            }
        }
    }
}

/// Hoisted Stage III sub-block scratch keeping zeroed buffers out of inner sub-block helper frames.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub(crate) struct Stage3SubblockScratch<const CAP: usize> {
    pub pos_buf: [usize; CAP],
    pub x_del_l: [f32; CAP],
    pub x_del_r: [f32; CAP],
    pub acc_l: [[f32; 4]; CAP],
    pub acc_r: [[f32; 4]; CAP],
    pub cab_out: [[f32; 2]; CAP],
}

impl<const CAP: usize> Stage3SubblockScratch<CAP> {
    /// Initializes Stage III scratch once without `MathBackend::Vec4` bounds so `V3HotCore` derives `Clone + Debug`.
    #[inline(always)]
    pub(crate) const fn zeroed() -> Self {
        Self {
            pos_buf: [0; CAP],
            x_del_l: [0.0; CAP],
            x_del_r: [0.0; CAP],
            acc_l: [[0.0; 4]; CAP],
            acc_r: [[0.0; 4]; CAP],
            cab_out: [[0.0; 2]; CAP],
        }
    }
}

/// Computes active L1D working-set bytes read/written per host stereo sample for `tier` and `os2x`.
#[must_use]
pub const fn active_hot_bytes_for_tier(tier: Tier, os2x: bool) -> usize {
    let [g0, g1, g2] = tier_block_groups_array(tier);
    let _ = (tier_groups(tier), tier_gru_blocks(tier));
    let g_sum = g0 + g1 + g2;
    let f = tier_fir_segments(tier);
    let k = tier_kautz_banks(tier);
    let tri0 = (g0 * (g0 + 1)) >> 1;
    let tri1 = (g1 * (g1 + 1)) >> 1;
    let tri2 = (g2 * (g2 + 1)) >> 1;
    const STAGE0_COLD_ONLY_BYTES: usize = 4 * core::mem::size_of::<f32>();
    const STAGE4_COLD_ONLY_BYTES: usize =
        core::mem::size_of::<Option<core::ptr::NonNull<u8>>>() + 5 * core::mem::size_of::<f32>();
    let stage0_bytes = core::mem::size_of::<V3Stage0Hot>() - STAGE0_COLD_ONLY_BYTES;
    let svf_bytes =
        core::mem::size_of::<V3Stage1SvfParams>() + core::mem::size_of::<V3Stage1SvfState>();
    let os_bytes = if os2x {
        core::mem::size_of::<V3OversamplerParams>() + core::mem::size_of::<V3StereoOversampler2x>()
    } else {
        0
    };
    let feat_bytes = core::mem::size_of::<V3InputFeaturesParams>()
        + 2 * core::mem::size_of::<V3InputFeaturesState>();
    let gru_group_bytes = g_sum * core::mem::size_of::<V3GruGroupHot>();
    let gru_k_quad_bytes = g_sum * 16;
    // Block 0: 3 w_rec tiles; Blocks 1 & 2: 3 w_prev + 3 w_rec tiles each.
    let gru_tile_bytes = (3 * tri0 + 6 * tri1 + 6 * tri2) * 64;
    let _ = LOWER_TRI_TILES;
    let _ = MAX_BLOCK_LANES;
    let gru_state_bytes = 2 * g_sum * 16;
    let bridge_bytes = 2 * (16 + g_sum * 32);
    let tone_svf_bytes = core::mem::size_of::<V3InterBlockToneSvfState>();
    let ctrl_bytes = g_sum * 48 + 128;
    let fir_bytes = 2 * f * 64 + 2 * f * 64;
    let kautz_bytes = 2 * k * core::mem::size_of::<KautzModalBank<4>>();
    let stage4_header_bytes = core::mem::size_of::<V3Stage4TankHot>()
        - (V3_TANK_RING_STRIDE * V3_TANK_LINES * 2)
        - STAGE4_COLD_ONLY_BYTES;
    let stage4_bytes = stage4_header_bytes + V3Stage4TankParams::active_ring_bytes(tier);
    stage0_bytes
        + svf_bytes
        + os_bytes
        + feat_bytes
        + gru_group_bytes
        + gru_k_quad_bytes
        + gru_tile_bytes
        + gru_state_bytes
        + bridge_bytes
        + tone_svf_bytes
        + ctrl_bytes
        + fir_bytes
        + kautz_bytes
        + stage4_bytes
}

const _: () = assert!(active_hot_bytes_for_tier(Tier::Lite, false) <= 4_608);
const _: () = assert!(active_hot_bytes_for_tier(Tier::Ultra, true) <= 28_672);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3_readout_mask_and_cold_tier_hot_bytes() {
        let mut ro = V3TierReadout::default();
        ro.fir_l.fill(0.5);
        ro.fir_r.fill(0.5);
        ro.mask_to_tier(Tier::Lite);
        assert_eq!(ro.fir_l[20], 0.0);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Full);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Pro);
        ro.mask_to_tier(Tier::Ultra);

        let mat = V3MatryoshkaModel::identity(true, false);
        let four = mat.to_four_tiers();
        assert_eq!(four[0].tier, Tier::Lite);
        assert_eq!(four[3].tier, Tier::Ultra);
        assert_eq!(mat.extract_tier(Tier::Full).tier, Tier::Full);

        let cold_ultra = V3ColdTierModel::identity_for_tier(Tier::Ultra, false, false);
        assert_eq!(cold_ultra.tier, Tier::Ultra);
        assert!(
            active_hot_bytes_for_tier(Tier::Lite, true)
                < active_hot_bytes_for_tier(Tier::Ultra, true)
        );

        let mut s3 = V3Stage3Hot::default();
        let mut sc = Stage3SubblockScratch::<16>::zeroed();
        s3.step_subblock_const::<crate::math::DefaultMathBackend, 0, 0, true, 16>(
            &[[1.0, 1.0, 0.0, 0.0]; 16],
            &mut sc,
            4,
        );
        assert_eq!(sc.cab_out[0], [0.0, 0.0]);

        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        s3.update_readout(&ro, &mat.kautz_poles);
        s3.reconfigure_with_zoh(&ro, &mat.kautz_poles, 1, &zoh96);
        assert_eq!(s3.onset_delay_samples, 32);
        s3.retune_kautz_3khz::<crate::math::DefaultMathBackend>(1, 0.12, 0.0);
        s3.retune_kautz_3khz::<crate::math::DefaultMathBackend>(1, 0.12, 0.0);
        s3.retune_kautz_3khz::<crate::math::DefaultMathBackend>(4, -0.2, 0.15);
        s3.retune_kautz_3khz::<crate::math::DefaultMathBackend>(4, 0.0, 0.0);

        let mono_step = s3.step_stereo::<crate::math::DefaultMathBackend>(0.5, -0.4, 4, 4, true);
        let stereo_step = s3.step_stereo::<crate::math::DefaultMathBackend>(0.5, -0.4, 4, 4, false);
        assert!(mono_step[0].is_finite() && stereo_step[1].is_finite());

        // Verify stationary tick un-contracts slew damping and snap_kautz_immediate restores 0-ULP rest poles.
        let mut s_ref = V3Stage3Hot::default();
        s_ref.update_readout(&ro, &mat.kautz_poles);
        s_ref.snap_kautz_immediate::<crate::math::DefaultMathBackend>(2, 0.0, 0.0);
        let mut s_sw = s_ref.clone();
        s_sw.retune_kautz_3khz::<crate::math::DefaultMathBackend>(2, 0.65, 0.0);
        assert!(s_sw.kautz_slew_active);
        s_sw.retune_kautz_3khz::<crate::math::DefaultMathBackend>(2, 0.0, 0.0);
        assert!(s_sw.kautz_slew_active);
        s_sw.retune_kautz_3khz::<crate::math::DefaultMathBackend>(2, 0.0, 0.0);
        assert!(!s_sw.kautz_slew_active);
        assert_eq!(s_sw.kautz_l[0].lambda_re, s_ref.kautz_l[0].lambda_re);
        assert_eq!(s_sw.kautz_l[0].b_gamma, s_ref.kautz_l[0].b_gamma);
        s_sw.retune_kautz_3khz::<crate::math::DefaultMathBackend>(2, -0.5, 0.2);
        s_sw.snap_kautz_immediate::<crate::math::DefaultMathBackend>(2, 0.0, 0.0);
        assert_eq!(s_sw.kautz_l[0].lambda_re, s_ref.kautz_l[0].lambda_re);
        assert_eq!(s_sw.kautz_l[0].b_gamma, s_ref.kautz_l[0].b_gamma);
    }
}
