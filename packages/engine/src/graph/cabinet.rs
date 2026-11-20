//! Stage III Early FIR (`128`-tap Ultra) + `32`-Pair Shared-Pole Kautz Modal Bank.

pub mod backend;
pub mod cold;
pub mod high_sr;
pub mod kautz_zoh;

use alloc::boxed::Box;
pub use backend::{CabinetBackend, dot16_stereo_vec4};
pub use cold::*;
pub use high_sr::{CabinetHighSrFirExt, STAGE3_EXT_FIR_CAP, STAGE3_EXT_FIR_MASK};
pub use kautz_zoh::{
    KautzColdSlewState, KautzRetuneContext, KautzStereoBank4, compute_onset_delay_samples,
    rescale_kautz_poles_with_zoh, retune_kautz_formant_3khz, retune_kautz_slice_3khz,
};

#[cfg(test)]
use dnam_kernels::step_kautz_quad_stereo8;

use crate::container::ZohRescaledParams;
use crate::graph::sona_core::model::{
    FIR_RING_CAP, FIR_RING_MASK, MAX_FIR_SEGMENTS, MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES,
    SONA_FIR_MAX_TAPS, tier_fir_segments,
};
use crate::math::{MathBackend, SimdVec4, newton6_reciprocal_f64};
use crate::types::Tier;
use kautz_zoh::cold_b_gamma;

/// Returns `(ring_cap, ring_mask)` for tier-masked Stage III double-mirrored FIR rings
/// (`Lite: 32/64`, `Full: 64/128`, `Pro|Ultra: 128/256`).
#[inline]
#[must_use]
pub const fn tier_fir_ring_cap_mask(tier: Tier) -> (usize, usize) {
    match tier {
        Tier::Lite => (32, 31),
        Tier::Full => (64, 63),
        Tier::Pro | Tier::Ultra => (128, 127),
    }
}

/// Contiguous Stage III Early FIR (`128` taps/ch max) + `32`-pole shared Kautz stereo cabinet,
/// with cold slew tables and high-SR/bounce extended FIR moved out of the hot L1D block.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct CabinetHot {
    pub fir_taps_l: [[[f32; 4]; 4]; MAX_FIR_SEGMENTS],
    pub fir_taps_r: [[[f32; 4]; 4]; MAX_FIR_SEGMENTS],
    pub kautz: [KautzStereoBank4; MAX_KAUTZ_BANKS],
    pub fir_hist_l: [f32; FIR_RING_CAP * 2],
    pub fir_hist_r: [f32; FIR_RING_CAP * 2],
    pub write_pos: u32,
    pub onset_delay_samples: u32,
    pub fir_ring_cap: u32,
    pub fir_ring_mask: u32,
    pub cold_slew: Box<KautzColdSlewState>,
    pub cold_fir_ext: Box<CabinetHighSrFirExt>,
    pub cold_fir_ext_active: bool,
}

impl Default for CabinetHot {
    fn default() -> Self {
        Self {
            fir_taps_l: [[[0.0; 4]; 4]; MAX_FIR_SEGMENTS],
            fir_taps_r: [[[0.0; 4]; 4]; MAX_FIR_SEGMENTS],
            kautz: [KautzStereoBank4::new(); MAX_KAUTZ_BANKS],
            fir_hist_l: [0.0; FIR_RING_CAP * 2],
            fir_hist_r: [0.0; FIR_RING_CAP * 2],
            write_pos: 0,
            onset_delay_samples: 16,
            fir_ring_cap: FIR_RING_CAP as u32,
            fir_ring_mask: FIR_RING_MASK as u32,
            cold_slew: Box::new(KautzColdSlewState::new()),
            cold_fir_ext: Box::new(CabinetHighSrFirExt::new_preallocated()),
            cold_fir_ext_active: false,
        }
    }
}

impl CabinetHot {
    /// Populates Stage III FIR taps and Kautz poles/residues from cold readout and poles.
    #[must_use]
    pub fn from_cold(model: &SonaColdTierModel) -> Self {
        let mut out = Self::default();
        let (cap, mask) = tier_fir_ring_cap_mask(model.tier);
        out.fir_ring_cap = cap as u32;
        out.fir_ring_mask = mask as u32;
        out.update_readout(&model.readout, &model.kautz_poles);
        out
    }

    /// Returns whether Stage III Kautz `3 kHz` slew contraction is currently active.
    #[inline]
    #[must_use]
    pub fn is_kautz_slew_active(&self) -> bool {
        self.cold_slew.slew_active
    }

    fn apply_fir_taps(
        &mut self,
        fir_l: &[f32; SONA_FIR_MAX_TAPS],
        fir_r: &[f32; SONA_FIR_MAX_TAPS],
    ) {
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

    fn apply_kautz_poles_and_residues_scaled(
        &mut self,
        readout: &SonaTierReadout,
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
        zoh_base_poles: Option<&[(f32, f32); MAX_KAUTZ_POLES]>,
    ) {
        self.cold_slew.host_poles = *kautz_poles;
        self.cold_slew.reset_slew();
        for b in 0..MAX_KAUTZ_BANKS {
            for lane in 0..4 {
                let m = (b << 2) + lane;
                let (r_raw, theta_raw) = kautz_poles[m];
                let r = r_raw.clamp(0.25, 0.9995);
                let theta = theta_raw.clamp(0.0, core::f32::consts::PI - 1.0e-4);
                // Preserve stored `kautz_b_gamma[m]` >= 0.0 verbatim (no 0.0 -> cold_b_gamma coercion).
                let bg_stored = if readout.kautz_b_gamma[m].is_finite() {
                    readout.kautz_b_gamma[m].max(0.0)
                } else {
                    0.0
                };
                let b_gamma = if bg_stored > 0.0 {
                    if let Some(base_poles) = zoh_base_poles {
                        let r_base = base_poles[m].0.clamp(0.25, 0.9995);
                        let ratio = cold_b_gamma(r)
                            * (newton6_reciprocal_f64(f64::from(cold_b_gamma(r_base))) as f32);
                        bg_stored * ratio
                    } else {
                        bg_stored
                    }
                } else {
                    0.0
                };
                self.cold_slew.host_b_gamma[m] = b_gamma;
                self.cold_slew.host_inv_1mr2[m] =
                    newton6_reciprocal_f64((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)) as f32;
                self.kautz[b].set_pole_stereo(
                    lane,
                    r,
                    theta,
                    b_gamma,
                    readout.kautz_c_l[m],
                    readout.kautz_c_r[m],
                );
            }
        }
    }

    /// Updates Stage III FIR taps and Kautz poles/residues without clearing ring/filter history.
    pub fn update_readout(
        &mut self,
        readout: &SonaTierReadout,
        kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    ) {
        self.onset_delay_samples = 16;
        self.cold_fir_ext_active = false;
        self.apply_fir_taps(&readout.fir_l, &readout.fir_r);
        self.apply_kautz_poles_and_residues_scaled(readout, kautz_poles, None);
    }

    /// Reconfigures Stage III FIR (hot `48 kHz` inline or cold `Cases (a)-(d)` extended ring) and
    /// Kautz poles/residues under ZOH (`0` RT allocs / drops).
    pub fn reconfigure_with_zoh_mode(
        &mut self,
        readout: &SonaTierReadout,
        base_kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
        num_fir_segments: usize,
        os_factor: usize,
        is_bounce_4x: bool,
        zoh: &ZohRescaledParams,
    ) {
        if (zoh.rho_sr - 1.0).abs() < 1.0e-12 {
            self.onset_delay_samples = 16;
            self.cold_fir_ext_active = false;
            self.apply_fir_taps(&readout.fir_l, &readout.fir_r);
            self.apply_kautz_poles_and_residues_scaled(readout, base_kautz_poles, None);
            return;
        }
        let active_taps = (num_fir_segments * 16).clamp(16, SONA_FIR_MAX_TAPS);
        self.cold_fir_ext.reconfigure(
            &readout.fir_l,
            &readout.fir_r,
            active_taps,
            os_factor,
            is_bounce_4x,
            zoh,
        );
        self.onset_delay_samples = self.cold_fir_ext.onset_delay_samples as u32;
        self.cold_fir_ext_active = true;
        let scaled_poles = rescale_kautz_poles_with_zoh(base_kautz_poles, zoh);
        self.apply_kautz_poles_and_residues_scaled(readout, &scaled_poles, Some(base_kautz_poles));
    }

    /// Reconfigures Stage III FIR taps and Kautz poles/residues under `1x` live ZOH (`0` truncation).
    pub fn reconfigure_with_zoh(
        &mut self,
        readout: &SonaTierReadout,
        base_kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
        num_fir_segments: usize,
        zoh: &ZohRescaledParams,
    ) {
        self.reconfigure_with_zoh_mode(readout, base_kautz_poles, num_fir_segments, 1, false, zoh);
    }

    /// Snaps Stage III Kautz poles directly to target `oct_shift`/`damp_delta` with `0` slew damping.
    #[inline]
    pub fn snap_kautz_immediate<B: MathBackend>(
        &mut self,
        num_kautz_banks: usize,
        oct_shift: f32,
        damp_delta: f32,
    ) {
        self.cold_slew.last_dtheta = oct_shift;
        self.cold_slew.last_damp = damp_delta;
        self.cold_slew.prev_oct = oct_shift;
        self.cold_slew.slew_active = false;
        retune_kautz_formant_3khz::<B>(
            &mut self.kautz,
            &KautzRetuneContext {
                host_poles: &self.cold_slew.host_poles,
                host_b_gamma: &self.cold_slew.host_b_gamma,
                host_inv_1mr2: &self.cold_slew.host_inv_1mr2,
                num_banks: num_kautz_banks,
                oct_shift,
                damp_delta,
                prev_oct_shift: oct_shift,
            },
        );
    }

    /// Retunes Stage III Kautz modal frequencies and radial damping at the `3 kHz` control boundary.
    #[inline]
    pub fn retune_kautz_3khz<B: MathBackend>(
        &mut self,
        num_kautz_banks: usize,
        oct_shift: f32,
        damp_delta: f32,
    ) {
        let prev_oct = self.cold_slew.prev_oct;
        let oct_changed = (oct_shift - self.cold_slew.last_dtheta).abs() > 1.0e-7;
        let damp_changed = (damp_delta - self.cold_slew.last_damp).abs() > 1.0e-7;
        let settling_slew = self.cold_slew.slew_active && (oct_shift - prev_oct).abs() > 1.0e-7;
        let needs_uncontract = self.cold_slew.slew_active;
        if !oct_changed && !damp_changed && !settling_slew && !needs_uncontract {
            return;
        }
        self.cold_slew.slew_active = oct_changed || (oct_shift - prev_oct).abs() > 1.0e-7;
        self.cold_slew.last_dtheta = oct_shift;
        self.cold_slew.last_damp = damp_delta;
        self.cold_slew.prev_oct = oct_shift;
        retune_kautz_formant_3khz::<B>(
            &mut self.kautz,
            &KautzRetuneContext {
                host_poles: &self.cold_slew.host_poles,
                host_b_gamma: &self.cold_slew.host_b_gamma,
                host_inv_1mr2: &self.cold_slew.host_inv_1mr2,
                num_banks: num_kautz_banks,
                oct_shift,
                damp_delta,
                prev_oct_shift: prev_oct,
            },
        );
    }

    /// Zeros Early FIR delay rings (including optional cold high-SR ring) and Kautz modal states cleanly.
    pub fn reset(&mut self) {
        self.fir_hist_l.fill(0.0);
        self.fir_hist_r.fill(0.0);
        self.write_pos = 0;
        self.cold_slew.reset_slew();
        self.cold_fir_ext.reset();
        for b in 0..MAX_KAUTZ_BANKS {
            self.kautz[b].reset();
        }
    }

    #[inline(always)]
    fn push_fir(&mut self, s2_l: f32, s2_r: f32) -> usize {
        let cap = (self.fir_ring_cap as usize).clamp(32, FIR_RING_CAP);
        let mask = (self.fir_ring_mask as usize).min(FIR_RING_MASK);
        let pos = (self.write_pos as usize).wrapping_sub(1) & mask;
        self.write_pos = pos as u32;
        self.fir_hist_l[pos] = s2_l;
        self.fir_hist_l[pos + cap] = s2_l;
        self.fir_hist_r[pos] = s2_r;
        self.fir_hist_r[pos + cap] = s2_r;
        pos
    }

    #[inline(always)]
    fn eval_fir_stereo<B: MathBackend>(&self, pos: usize, num_fir_segments: usize) -> (f32, f32) {
        let mut acc_l = B::Vec4::zero();
        let mut acc_r = B::Vec4::zero();
        for seg in 0..num_fir_segments.min(MAX_FIR_SEGMENTS) {
            let s = (pos & (self.fir_ring_mask as usize)) + (seg << 4);
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

    /// Steps Stage III FIR + Kautz modal banks for stereo inputs `(s2_l, s2_r)`, always
    /// advancing both L/R states and folding to `[mid, mid]` when `mono_out` is enabled.
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(
        &mut self,
        s2_l: f32,
        s2_r: f32,
        num_fir_segments: usize,
        num_kautz_banks: usize,
        mono_out: bool,
    ) -> [f32; 2] {
        let active_banks = num_kautz_banks.min(MAX_KAUTZ_BANKS);
        let (fir_l, fir_r, delayed_l, delayed_r) = if self.cold_fir_ext_active {
            let (fir, dl, dr) = self.cold_fir_ext.step_frame::<B, false>(s2_l, s2_r);
            (fir[0], fir[1], dl, dr)
        } else {
            let pos = self.push_fir(s2_l, s2_r);
            let mask = self.fir_ring_mask as usize;
            let onset = (self.onset_delay_samples as usize).clamp(1, mask.max(1));
            let dl = self.fir_hist_l[(pos & mask) + onset];
            let dr = self.fir_hist_r[(pos & mask) + onset];
            let (fl, fr) = self.eval_fir_stereo::<B>(pos, num_fir_segments);
            (fl, fr, dl, dr)
        };

        let mut kl_sum = 0.0_f32;
        let mut kr_sum = 0.0_f32;
        for b in 0..active_banks {
            let (kl, kr) = self.kautz[b].step_stereo::<B>(delayed_l, delayed_r);
            kl_sum += kl;
            kr_sum += kr;
        }
        let out_l = fir_l + kl_sum;
        let out_r = fir_r + kr_sum;
        if mono_out {
            let mid = 0.5 * (out_l + out_r);
            [mid, mid]
        } else {
            [out_l, out_r]
        }
    }

    /// Const-generic sub-block FIR (`i` outer in SIMD registers, `seg` inner) + register-hoisted
    /// bank-outer Kautz evaluation across `0..len` (`128` base taps or `<= 1,024` ZOH cold extension).
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn step_subblock_const<
        B: MathBackend,
        const FIR_SEGS: usize,
        const KAUTZ_BANKS: usize,
        const MONO: bool,
        const CAP: usize,
    >(
        &mut self,
        pre_cab: &[[f32; 2]; CAP],
        sc: &mut CabinetSubblockScratch<CAP>,
        len: usize,
    ) {
        const { assert!(FIR_SEGS <= MAX_FIR_SEGMENTS && KAUTZ_BANKS <= MAX_KAUTZ_BANKS) };
        let n = len.min(CAP);
        if FIR_SEGS == 0 && KAUTZ_BANKS == 0 {
            sc.cab_out[..n].fill([0.0, 0.0]);
            return;
        }
        let x_del_l = &mut sc.x_del_l;
        let x_del_r = &mut sc.x_del_r;
        let cab_out = &mut sc.cab_out;

        if self.cold_fir_ext_active {
            for i in 0..n {
                let (fir, dl, dr) = self
                    .cold_fir_ext
                    .step_frame::<B, false>(pre_cab[i][0], pre_cab[i][1]);
                cab_out[i] = fir;
                x_del_l[i] = dl;
                x_del_r[i] = dr;
            }
        } else {
            let mask = self.fir_ring_mask as usize;
            let onset = (self.onset_delay_samples as usize).clamp(1, mask.max(1));
            for i in 0..n {
                let p = self.push_fir(pre_cab[i][0], pre_cab[i][1]) & mask;
                x_del_l[i] = self.fir_hist_l[p + onset];
                x_del_r[i] = self.fir_hist_r[p + onset];
                let mut acc_l = B::Vec4::zero();
                let mut acc_r = B::Vec4::zero();
                for seg in 0..FIR_SEGS {
                    let s = p + (seg << 4);
                    let (hl, _) = self.fir_hist_l[s..s + 16].as_chunks::<4>();
                    let tl = &self.fir_taps_l[seg];
                    let f0 = B::Vec4::load(&hl[0]).mul(B::Vec4::load(&tl[0]));
                    let f1 = B::Vec4::load(&hl[1]).mul(B::Vec4::load(&tl[1]));
                    acc_l = acc_l
                        .add(f0.fma(B::Vec4::load(&hl[2]), B::Vec4::load(&tl[2])))
                        .add(f1.fma(B::Vec4::load(&hl[3]), B::Vec4::load(&tl[3])));
                    let (hr, _) = self.fir_hist_r[s..s + 16].as_chunks::<4>();
                    let tr = &self.fir_taps_r[seg];
                    let g0 = B::Vec4::load(&hr[0]).mul(B::Vec4::load(&tr[0]));
                    let g1 = B::Vec4::load(&hr[1]).mul(B::Vec4::load(&tr[1]));
                    acc_r = acc_r
                        .add(g0.fma(B::Vec4::load(&hr[2]), B::Vec4::load(&tr[2])))
                        .add(g1.fma(B::Vec4::load(&hr[3]), B::Vec4::load(&tr[3])));
                }
                let (fl, fr) = acc_l.horizontal_sum_pair(acc_r);
                cab_out[i] = [fl, fr];
            }
        }

        for b in 0..KAUTZ_BANKS {
            let mut st = self.kautz[b].to_kernel_state::<B::Vec4>();
            for i in 0..n {
                let (kl, kr) =
                    step_kautz_quad_stereo8::<B::Stereo8>(&mut st, x_del_l[i], x_del_r[i]);
                cab_out[i][0] += kl;
                cab_out[i][1] += kr;
            }
            self.kautz[b].store_kernel_state(&st);
        }
        if MONO {
            for frame in &mut cab_out[..n] {
                let mid = 0.5 * (frame[0] + frame[1]);
                *frame = [mid, mid];
            }
        }
    }
}

/// Hoisted Stage III sub-block scratch keeping delay/output buffers out of stack loop frames.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub(crate) struct CabinetSubblockScratch<const CAP: usize> {
    pub(crate) x_del_l: [f32; CAP],
    pub(crate) x_del_r: [f32; CAP],
    pub(crate) cab_out: [[f32; 2]; CAP],
}

impl<const CAP: usize> CabinetSubblockScratch<CAP> {
    #[inline(always)]
    pub(crate) const fn zeroed() -> Self {
        Self {
            x_del_l: [0.0; CAP],
            x_del_r: [0.0; CAP],
            cab_out: [[0.0; 2]; CAP],
        }
    }
}

/// Computes active `SONA` L1D working-set bytes for `(tier, os2x)` matching real `SonaHotSlab`
/// prefix footprints (`<= 4,608 / 7,936 / 15,872 / 30,720 B`).
#[must_use]
pub const fn active_hot_bytes_for_tier(tier: Tier, os2x: bool) -> usize {
    let f = tier_fir_segments(tier);
    let _ = f;
    crate::graph::sona_core::slab::sona_slab_active_hot_bytes(tier, os2x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_sona_readout_mask_stage3_128_taps_and_kautz_retune() {
        let mut ro = SonaTierReadout::default();
        ro.fir_l.fill(0.5);
        ro.fir_r.fill(0.5);
        ro.mask_to_tier(Tier::Lite);
        assert_eq!(ro.fir_l[31], 0.5);
        assert_eq!(ro.fir_l[32], 0.0);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Full);
        assert_eq!(ro.fir_l[47], 0.5);
        assert_eq!(ro.fir_l[48], 0.0);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Pro);
        assert_eq!(ro.fir_l[95], 0.5);
        assert_eq!(ro.fir_l[96], 0.0);
        ro.fir_l.fill(0.5);
        ro.mask_to_tier(Tier::Ultra);
        assert_eq!(ro.fir_l[127], 0.5);

        let mat = SonaMatryoshkaModel::identity(true, false);
        let four = mat.to_four_tiers();
        assert_eq!(four[0].tier, Tier::Lite);
        assert_eq!(four[3].tier, Tier::Ultra);

        let mut s3 = CabinetHot::default();
        let mut sc = CabinetSubblockScratch::<16>::zeroed();
        s3.step_subblock_const::<DefaultMathBackend, 0, 0, true, 16>(&[[1.0, 1.0]; 16], &mut sc, 4);
        assert_eq!(sc.cab_out[0], [0.0, 0.0]);

        // Verify all 8 FIR segments (128 taps, seg 7 at offset 112..128) evaluate distinctly without (seg & 3).
        ro.fir_l.fill(0.0);
        ro.fir_r.fill(0.0);
        ro.fir_l[120] = 0.75;
        ro.fir_r[127] = -0.5;
        // Explicitly preserve b_gamma = 0.0 verbatim (no cold_b_gamma coercion).
        ro.kautz_b_gamma[0] = 0.0;
        s3.update_readout(&ro, &mat.kautz_poles);
        assert_eq!(s3.kautz[0].b_gamma[0].to_bits(), 0);
        for b in 0..MAX_KAUTZ_BANKS {
            s3.kautz[b].c_re_l = [0.0; 4];
            s3.kautz[b].c_im_l = [0.0; 4];
            s3.kautz[b].c_re_r = [0.0; 4];
            s3.kautz[b].c_im_r = [0.0; 4];
        }
        let mut pre = [[0.0_f32; 2]; 16];
        pre[0] = [1.0, 1.0];
        let mut imp_l = alloc::vec::Vec::new();
        let mut imp_r = alloc::vec::Vec::new();
        for blk in 0..9 {
            let input = if blk == 0 { pre } else { [[0.0; 2]; 16] };
            s3.step_subblock_const::<DefaultMathBackend, 8, 8, false, 16>(&input, &mut sc, 16);
            for i in 0..16 {
                imp_l.push(sc.cab_out[i][0]);
                imp_r.push(sc.cab_out[i][1]);
            }
        }
        assert!((imp_l[120] - 0.75).abs() < 1e-6);
        assert!((imp_r[127] + 0.5).abs() < 1e-6);

        let mut ro_nz = ro.clone();
        ro_nz.kautz_b_gamma.fill(0.25);
        let mut s_ref = CabinetHot::default();
        s_ref.update_readout(&ro_nz, &mat.kautz_poles);
        s_ref.snap_kautz_immediate::<DefaultMathBackend>(2, 0.0, 0.0);
        let mut s_sw = s_ref.clone();
        s_sw.retune_kautz_3khz::<DefaultMathBackend>(2, 0.65, 0.0);
        assert!(s_sw.is_kautz_slew_active());
        s_sw.retune_kautz_3khz::<DefaultMathBackend>(2, 0.0, 0.0);
        s_sw.retune_kautz_3khz::<DefaultMathBackend>(2, 0.0, 0.0);
        assert!(!s_sw.is_kautz_slew_active());
        assert_eq!(s_sw.kautz[0].lambda_re, s_ref.kautz[0].lambda_re);
        assert_eq!(s_sw.kautz[0].b_gamma, s_ref.kautz[0].b_gamma);
    }

    #[test]
    fn test_stage3_zoh_mono_reset_and_tier_hot_bytes() {
        let mut prev_bytes = 0usize;
        for (tier, expected_cap_mask) in [
            (Tier::Lite, (32, 31)),
            (Tier::Full, (64, 63)),
            (Tier::Pro, (128, 127)),
            (Tier::Ultra, (128, 127)),
        ] {
            assert_eq!(tier_fir_ring_cap_mask(tier), expected_cap_mask);
            let b1 = active_hot_bytes_for_tier(tier, false);
            let b2 = active_hot_bytes_for_tier(tier, true);
            assert!(b1 > prev_bytes && b2 > b1 && b2 <= 30_720);
            prev_bytes = b1;
        }

        let cold_pro = SonaColdTierModel::identity_for_tier(Tier::Pro, false, false);
        let mut s3 = CabinetHot::from_cold(&cold_pro);
        assert_eq!(s3.fir_ring_cap, 128);
        assert_eq!(s3.fir_ring_mask, 127);

        let mut ro = cold_pro.readout.clone();
        ro.kautz_b_gamma[0] = f32::NAN;
        ro.kautz_b_gamma[1] = 0.25;
        ro.kautz_c_l[1] = (0.3, 0.1);
        ro.kautz_c_r[1] = (-0.2, 0.15);
        s3.update_readout(&ro, &cold_pro.kautz_poles);
        assert_eq!(s3.kautz[0].b_gamma[0], 0.0);
        assert_eq!(s3.kautz[0].b_gamma[1], 0.25);

        let zoh48 = ZohRescaledParams::new(48_000, 48_000);
        s3.reconfigure_with_zoh(&ro, &cold_pro.kautz_poles, 3, &zoh48);
        assert!(!s3.cold_fir_ext_active);
        assert_eq!(s3.onset_delay_samples, 16);

        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        s3.reconfigure_with_zoh(&ro, &cold_pro.kautz_poles, 3, &zoh96);
        assert!(s3.cold_fir_ext_active);
        assert_eq!(s3.onset_delay_samples, 32);

        let st_stereo = s3.step_stereo::<DefaultMathBackend>(0.5, -0.25, 3, 3, false);
        let st_mono = s3.step_stereo::<DefaultMathBackend>(0.5, -0.25, 3, 3, true);
        assert!(st_stereo[0].is_finite() && st_stereo[1].is_finite());
        assert_eq!(st_mono[0].to_bits(), st_mono[1].to_bits());

        let mut sc = CabinetSubblockScratch::<16>::zeroed();
        let pre = [[0.4_f32, -0.2]; 16];
        s3.step_subblock_const::<DefaultMathBackend, 3, 3, true, 16>(&pre, &mut sc, 4);
        for i in 0..4 {
            assert_eq!(sc.cab_out[i][0].to_bits(), sc.cab_out[i][1].to_bits());
            assert!(sc.cab_out[i][0].is_finite());
        }

        s3.retune_kautz_3khz::<DefaultMathBackend>(2, 0.0, 0.0);
        s3.retune_kautz_3khz::<DefaultMathBackend>(2, 0.0, 0.0);
        s3.reset();
        assert_eq!(s3.write_pos, 0);
        assert!(!s3.is_kautz_slew_active());

        let zoh192 = ZohRescaledParams::new(48_000, 192_000);
        s3.reconfigure_with_zoh_mode(&ro, &cold_pro.kautz_poles, 4, 1, true, &zoh192);
        assert!(s3.cold_fir_ext_active);
    }
}
