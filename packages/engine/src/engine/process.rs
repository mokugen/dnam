//! Real-time stereo and mono audio processing loops for [`DnamEngine`] (`0 FDIV`, `0 FSQRT`,
//! `0 IDIV`, `0 libm`, `0` heap allocations).
//!
//! Submodules:
//! - [`multiport`]: 4-port (`4CM`, `WDW`, `Sidechain`) stereo processing loops.

pub mod multiport;

use dnam_kernels::scrub_and_flush_f32_bits;
use dnam_vm::SonaVm;
use dnam_vm::math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, SAFETY_CEILING_F32, sanitize_input_sample,
    sanitize_output_sample,
};

use super::{DnamEngine, EngineBackend, HostShellPatchbay};
use crate::provenance::PreviewGuardState;
use crate::types::FtzDazGuard;

/// Unified multi-port audio I/O descriptor (`Stage 0` 8-port router contract: `in_1..4`, `out_1..4`).
#[derive(Debug)]
pub struct DnamAudioIo<'b> {
    pub in_1: &'b [f32],
    pub in_2: Option<&'b [f32]>,
    pub in_3: Option<&'b [f32]>,
    pub in_4: Option<&'b [f32]>,
    pub out_1: &'b mut [f32],
    pub out_2: Option<&'b mut [f32]>,
    pub out_3: Option<&'b mut [f32]>,
    pub out_4: Option<&'b mut [f32]>,
}

/// Shared host shell stereo post-processing state passed cleanly across live and bounce VMs.
pub(crate) struct StereoShellParams<'b> {
    pub(crate) bypassed: bool,
    pub(crate) trails_enabled: bool,
    pub(crate) single_input: bool,
    pub(crate) stereo_width: f32,
    pub(crate) mix_dry_gain: f32,
    pub(crate) mix_wet_gain: f32,
    pub(crate) wet_polarity_inverted: bool,
    pub(crate) preview_mode: bool,
    pub(crate) preview_guard: Option<&'b mut PreviewGuardState>,
    pub(crate) saturation_flags: &'b mut u32,
    pub(crate) last_block_max_drive: &'b mut f32,
}

#[inline(always)]
fn all_within_envelope_bits(slice: &[f32], limit_bits: u32) -> bool {
    let mut i = 0;
    while i < slice.len() {
        if (slice[i].to_bits() & 0x7fff_ffff) > limit_bits {
            return false;
        }
        i += 1;
    }
    true
}

#[inline(always)]
pub(crate) fn apply_host_post_mix(
    vm_wet_pol_inv: bool,
    mono_out: bool,
    params: &mut StereoShellParams<'_>,
    dry_l: &[f32],
    dry_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let eff_dry_g = if params.bypassed {
        1.0_f32
    } else {
        params.mix_dry_gain
    };
    let host_pol_sign = if !vm_wet_pol_inv && params.wet_polarity_inverted {
        -1.0_f32
    } else {
        1.0_f32
    };
    let eff_wet_g = host_pol_sign * params.mix_wet_gain;
    let width = HostShellPatchbay::clamp_ms_width(params.stereo_width);
    let width_active = !mono_out && (width - 1.0).abs() > 1.0e-6;
    let mix_active = eff_dry_g != 0.0 || (eff_wet_g - 1.0).abs() > 1.0e-6;
    let ceil_bits = SAFETY_CEILING_F32.to_bits();

    if !width_active && !mix_active {
        if all_within_envelope_bits(out_l, ceil_bits) && all_within_envelope_bits(out_r, ceil_bits)
        {
            return;
        }
        for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
            *l = sanitize_output_sample(*l, params.saturation_flags);
            *r = sanitize_output_sample(*r, params.saturation_flags);
        }
        return;
    }

    for i in 0..out_l.len() {
        let mut wl = out_l[i];
        let mut wr = out_r[i];
        if width_active {
            (wl, wr) = HostShellPatchbay::apply_ms_width(wl, wr, width);
        }
        let yl = eff_dry_g * dry_l[i] + eff_wet_g * wl;
        let yr = eff_dry_g * dry_r[i] + eff_wet_g * wr;
        out_l[i] = sanitize_output_sample(yl, params.saturation_flags);
        out_r[i] = sanitize_output_sample(yr, params.saturation_flags);
    }
}

/// Shared zero-allocation stereo block executor over an arbitrary [`SonaVm`].
#[inline(never)]
pub(crate) fn process_stereo_vm_block<B: EngineBackend>(
    vm: &mut SonaVm<'static, B>,
    left_in: &[f32],
    right_in: &[f32],
    left_out: &mut [f32],
    right_out: &mut [f32],
    params: &mut StereoShellParams<'_>,
) {
    let num_frames = left_in
        .len()
        .min(right_in.len())
        .min(left_out.len())
        .min(right_out.len());
    if num_frames == 0 {
        return;
    }

    let li = &left_in[..num_frames];
    let ri = if params.single_input {
        li
    } else {
        &right_in[..num_frames]
    };
    let lo = &mut left_out[..num_frames];
    let ro = &mut right_out[..num_frames];

    if params.bypassed && !params.trails_enabled {
        for i in 0..num_frames {
            lo[i] = sanitize_input_sample(
                li[i],
                CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                params.saturation_flags,
            );
            ro[i] = sanitize_input_sample(
                ri[i],
                CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                params.saturation_flags,
            );
        }
        return;
    }

    let vm_wet_pol_inv = vm.slab().wet_polarity_inv();
    let mono_out = vm.slab().mono_out();
    let mut san_l = [0.0_f32; 64];
    let mut san_r = [0.0_f32; 64];
    let zeros = [0.0_f32; 64];
    let mut off = 0_usize;

    while off < num_frames {
        let n = (num_frames - off).min(64);
        let mut peak = *params.last_block_max_drive;
        for i in 0..n {
            let sl = sanitize_input_sample(
                li[off + i],
                CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                params.saturation_flags,
            );
            let sr = if params.single_input {
                sl
            } else {
                sanitize_input_sample(
                    ri[off + i],
                    CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                    params.saturation_flags,
                )
            };
            san_l[i] = scrub_and_flush_f32_bits(sl);
            san_r[i] = scrub_and_flush_f32_bits(sr);
            if !params.bypassed {
                let mag = san_l[i].abs().max(san_r[i].abs()).min(1.0);
                if mag > peak {
                    peak = mag;
                }
            }
        }
        *params.last_block_max_drive = peak;

        let (vm_in_l, vm_in_r) = if params.bypassed {
            (&zeros[..n], &zeros[..n])
        } else {
            (&san_l[..n], &san_r[..n])
        };
        let _ = vm.process_block(
            vm_in_l,
            vm_in_r,
            &mut lo[off..off + n],
            &mut ro[off..off + n],
        );
        apply_host_post_mix(
            vm_wet_pol_inv,
            mono_out,
            params,
            &san_l[..n],
            &san_r[..n],
            &mut lo[off..off + n],
            &mut ro[off..off + n],
        );
        off += n;
    }

    if params.preview_mode
        && let Some(guard) = params.preview_guard.as_deref_mut()
    {
        guard.process_block_in_place(lo, ro);
    }
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Low-level single-frame `SONA` stepping (`0 FDIV`, `0 FSQRT`).
    ///
    /// Direct callers must hold a [`FtzDazGuard`] for the enclosing batch.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(never)]
    pub fn step_frame_sona(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
        self.sync_hot_core_tier();
        let raw_l = sanitize_input_sample(
            in_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let raw_r = if self.single_input {
            raw_l
        } else {
            sanitize_input_sample(
                in_r,
                CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                &mut self.saturation_flags,
            )
        };
        let sl = scrub_and_flush_f32_bits(raw_l);
        let sr = scrub_and_flush_f32_bits(raw_r);

        if self.bypassed && !self.trails_enabled {
            return if mono_only {
                let m = 0.5 * (sl + sr);
                [m, m]
            } else {
                [sl, sr]
            };
        }

        let (eff_l, eff_r) = if self.bypassed {
            (0.0_f32, 0.0_f32)
        } else {
            let peak = sl.abs().max(sr.abs()).min(1.0);
            if peak > self.last_block_max_drive {
                self.last_block_max_drive = peak;
            }
            (sl, sr)
        };

        let (mut yl, mut yr) = self.vm.step_frame(eff_l, eff_r).unwrap_or((0.0, 0.0));
        let vm_wet_pol_inv = self.vm.slab().wet_polarity_inv();
        let host_pol_sign = if !vm_wet_pol_inv && self.json_wet_polarity_inverted() {
            -1.0_f32
        } else {
            1.0_f32
        };
        let mono_vm = self.vm.slab().mono_out();
        let dry_g = if self.bypassed {
            1.0
        } else {
            self.mix_dry_gain
        };
        let wet_g = host_pol_sign * self.mix_wet_gain;
        if !mono_vm {
            (yl, yr) = HostShellPatchbay::apply_ms_width(yl, yr, self.patchbay.ms_width);
        }
        yl = sanitize_output_sample(dry_g * sl + wet_g * yl, &mut self.saturation_flags);
        yr = sanitize_output_sample(dry_g * sr + wet_g * yr, &mut self.saturation_flags);
        if self.preview_mode {
            (yl, yr) = self.preview_guard.process_sample_lr(yl, yr);
        }
        if mono_only {
            let m = 0.5 * (yl + yr);
            [m, m]
        } else {
            [yl, yr]
        }
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`).
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn process_block(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let _ftz = FtzDazGuard::new();
        self.last_block_max_drive = 0.0;
        let num_frames = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        self.tempo_state.advance_frames(num_frames);
        self.sync_hot_core_tier();

        let mut params = StereoShellParams {
            bypassed: self.bypassed,
            trails_enabled: self.trails_enabled,
            single_input: self.single_input,
            stereo_width: self.patchbay.ms_width,
            mix_dry_gain: self.mix_dry_gain,
            mix_wet_gain: self.mix_wet_gain,
            wet_polarity_inverted: self.json_wet_polarity_inverted(),
            preview_mode: self.preview_mode,
            preview_guard: if self.preview_mode {
                Some(&mut self.preview_guard)
            } else {
                None
            },
            saturation_flags: &mut self.saturation_flags,
            last_block_max_drive: &mut self.last_block_max_drive,
        };
        process_stereo_vm_block(
            &mut self.vm,
            &left_in[..num_frames],
            &right_in[..num_frames],
            &mut left_out[..num_frames],
            &mut right_out[..num_frames],
            &mut params,
        );
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`), matching `process_block`.
    #[inline]
    pub fn process_stereo(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        self.process_block(left_in, right_in, left_out, right_out);
    }

    /// Processes a stereo audio block into mono (`Stereo -> Mono`), preserving Mid output.
    pub fn process_stereo_to_mono(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        mono_out: &mut [f32],
    ) {
        let _ftz = FtzDazGuard::new();
        self.last_block_max_drive = 0.0;
        let num_frames = left_in.len().min(right_in.len()).min(mono_out.len());
        self.tempo_state.advance_frames(num_frames);
        self.sync_hot_core_tier();

        let mut scratch_l = [0.0_f32; 64];
        let mut scratch_r = [0.0_f32; 64];
        let mut params = StereoShellParams {
            bypassed: self.bypassed,
            trails_enabled: self.trails_enabled,
            single_input: self.single_input,
            stereo_width: self.patchbay.ms_width,
            mix_dry_gain: self.mix_dry_gain,
            mix_wet_gain: self.mix_wet_gain,
            wet_polarity_inverted: self.json_wet_polarity_inverted(),
            preview_mode: self.preview_mode,
            preview_guard: if self.preview_mode {
                Some(&mut self.preview_guard)
            } else {
                None
            },
            saturation_flags: &mut self.saturation_flags,
            last_block_max_drive: &mut self.last_block_max_drive,
        };

        let mut off = 0_usize;
        while off < num_frames {
            let n = (num_frames - off).min(64);
            process_stereo_vm_block(
                &mut self.vm,
                &left_in[off..off + n],
                &right_in[off..off + n],
                &mut scratch_l[..n],
                &mut scratch_r[..n],
                &mut params,
            );
            for i in 0..n {
                mono_out[off + i] = 0.5 * (scratch_l[i] + scratch_r[i]);
            }
            off += n;
        }
    }

    /// Processes a mono audio block into true stereo (`Mono -> Stereo`).
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let num_frames = mono_in.len().min(left_out.len()).min(right_out.len());
        self.process_block(
            &mono_in[..num_frames],
            &mono_in[..num_frames],
            &mut left_out[..num_frames],
            &mut right_out[..num_frames],
        );
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        let num_frames = mono_in.len().min(mono_out.len());
        self.process_stereo_to_mono(&mono_in[..num_frames], &mono_in[..num_frames], mono_out);
    }

    /// Processes a stereo audio block strictly in-place with zero heap allocations.
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        self.last_block_max_drive = 0.0;
        let num_frames = left_io.len().min(right_io.len());
        self.tempo_state.advance_frames(num_frames);
        self.sync_hot_core_tier();

        let mut in_l = [0.0_f32; 64];
        let mut in_r = [0.0_f32; 64];
        let mut params = StereoShellParams {
            bypassed: self.bypassed,
            trails_enabled: self.trails_enabled,
            single_input: self.single_input,
            stereo_width: self.patchbay.ms_width,
            mix_dry_gain: self.mix_dry_gain,
            mix_wet_gain: self.mix_wet_gain,
            wet_polarity_inverted: self.json_wet_polarity_inverted(),
            preview_mode: self.preview_mode,
            preview_guard: if self.preview_mode {
                Some(&mut self.preview_guard)
            } else {
                None
            },
            saturation_flags: &mut self.saturation_flags,
            last_block_max_drive: &mut self.last_block_max_drive,
        };

        let mut off = 0_usize;
        while off < num_frames {
            let n = (num_frames - off).min(64);
            in_l[..n].copy_from_slice(&left_io[off..off + n]);
            in_r[..n].copy_from_slice(&right_io[off..off + n]);
            process_stereo_vm_block(
                &mut self.vm,
                &in_l[..n],
                &in_r[..n],
                &mut left_io[off..off + n],
                &mut right_io[off..off + n],
                &mut params,
            );
            off += n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::DnamJsonHeader;
    use crate::test_support::{build_test_dnam_bin, build_test_dnam_container};
    use crate::types::Tier;
    use dnam_vm::bin::DNBN_FLAG_STEREO_OUT;
    use dnam_vm::math::ScalarBackend;

    #[test]
    fn test_process_modes_and_in_place() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Lite, true);

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        let mut l = [0.1f32; 16];
        let mut r = [-0.1f32; 16];
        engine.process_block_in_place(&mut l, &mut r);
        assert!(l[0].is_finite() && r[0].is_finite());

        engine.set_offline_bounce(true);
        let mut l_out = [0.0f32; 16];
        let mut r_out = [0.0f32; 16];
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        engine.process_mono_to_stereo(&l, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        engine.process_mono(&l, &mut l_out);
        assert!(l_out[0].is_finite());

        engine.process_block_in_place(&mut l, &mut r);
        assert!(l[0].is_finite());

        engine.set_offline_bounce(false);
        engine.set_bypass(true, true);
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        engine.set_bypass(true, false);
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert_eq!(l_out, l);
        engine.process_mono(&l, &mut l_out);
        assert_eq!(l_out, l);
    }

    #[test]
    fn single_input_container_duplicates_left_and_ignores_right_channel() {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.inputs = 1;
        let container = build_test_dnam_container(&mut header, Tier::Lite, true);

        let mut eng_diff =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_same =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        for _ in 0..8 {
            let y_diff = eng_diff.step_frame_sona(0.25, -0.73, false);
            let y_same = eng_same.step_frame_sona(0.25, 0.25, false);
            assert_eq!(y_diff[0].to_bits(), y_same[0].to_bits());
            assert_eq!(y_diff[1].to_bits(), y_same[1].to_bits());
        }

        eng_diff.reset_to_dc_equilibrium();
        eng_same.reset_to_dc_equilibrium();
        let in_l = [0.25_f32; 8];
        let in_r_diff = [-0.73_f32; 8];
        let (mut out_dl, mut out_dr) = ([0.0_f32; 8], [0.0_f32; 8]);
        let (mut out_sl, mut out_sr) = ([0.0_f32; 8], [0.0_f32; 8]);
        eng_diff.process_block(&in_l, &in_r_diff, &mut out_dl, &mut out_dr);
        eng_same.process_block(&in_l, &in_l, &mut out_sl, &mut out_sr);
        for i in 0..8 {
            assert_eq!(out_dl[i].to_bits(), out_sl[i].to_bits());
            assert_eq!(out_dr[i].to_bits(), out_sr[i].to_bits());
        }
    }

    #[test]
    fn process_mono_outputs_mid_and_advances_stereo_state() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Lite, true);

        let mut eng_mono =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_st =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        let mono_in = [0.4_f32; 32];
        let mut mono_out = [0.0_f32; 32];
        let (mut st_l, mut st_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        eng_mono.process_mono(&mono_in, &mut mono_out);
        eng_st.process_block(&mono_in, &mono_in, &mut st_l, &mut st_r);

        for i in 0..32 {
            let expected_mid = 0.5 * (st_l[i] + st_r[i]);
            assert!((mono_out[i] - expected_mid).abs() < 1.0e-6, "sample {i}");
        }
        assert_eq!(
            eng_mono.vm.slab().lite.sample_tick,
            eng_st.vm.slab().lite.sample_tick
        );
    }

    /// verify[dsp.router.topology-dispatch]
    #[test]
    fn test_four_topology_entrypoints_preserve_dual_mic_and_mid_downmix() {
        let dbin = build_test_dnam_bin(Tier::Lite, 48_000, DNBN_FLAG_STEREO_OUT, &[], &[], true);
        let mut header = DnamJsonHeader::default_reference();
        let container =
            crate::container::build_dnam_container_from_dnam_bin(&mut header, &dbin).unwrap();

        let in_l = [0.35_f32; 24];
        let in_r = [-0.20_f32; 24];
        let mut s_l = [0.0_f32; 24];
        let mut s_r = [0.0_f32; 24];
        let mut s2m_out = [0.0_f32; 24];
        let mut m2s_l = [0.0_f32; 24];
        let mut m2s_r = [0.0_f32; 24];
        let mut m_out = [0.0_f32; 24];

        let mut e_st =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut e_s2m =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut e_m2s =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut e_mono =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        e_st.process_stereo(&in_l, &in_r, &mut s_l, &mut s_r);
        e_s2m.process_stereo_to_mono(&in_l, &in_r, &mut s2m_out);
        e_m2s.process_mono_to_stereo(&in_l, &mut m2s_l, &mut m2s_r);
        e_mono.process_mono(&in_l, &mut m_out);

        for i in 0..24 {
            assert!((s2m_out[i] - 0.5 * (s_l[i] + s_r[i])).abs() < 1.0e-6);
            assert!((m_out[i] - 0.5 * (m2s_l[i] + m2s_r[i])).abs() < 1.0e-6);
        }
    }

    #[test]
    fn trails_bypass_preserves_calibrated_polarity_and_trim_continuity() {
        let mut ref_hdr = DnamJsonHeader::default_reference();
        let ref_c =
            crate::test_support::build_spatial_test_dnam_container(&mut ref_hdr, Tier::Lite);
        let mut pol_hdr = DnamJsonHeader::default_reference();
        pol_hdr.io_config.capabilities.wet_polarity_inverted = true;
        pol_hdr.io_config.leveling.auto_unity_enabled = true;
        pol_hdr.io_config.leveling.calibrated_trim_db = -6.0206;
        let pol_c =
            crate::test_support::build_spatial_test_dnam_container(&mut pol_hdr, Tier::Lite);

        let excite = [0.5_f32; 32];
        let zeros = [0.0_f32; 16];
        let (mut rl, mut rr) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut pl, mut pr) = ([0.0_f32; 32], [0.0_f32; 32]);

        let mut eng_ref =
            DnamEngine::<ScalarBackend>::from_bytes(&ref_c, Tier::Lite, 48_000).unwrap();
        let mut eng_pol =
            DnamEngine::<ScalarBackend>::from_bytes(&pol_c, Tier::Lite, 48_000).unwrap();
        eng_ref.process_block(&excite, &excite, &mut rl, &mut rr);
        eng_pol.process_block(&excite, &excite, &mut pl, &mut pr);

        eng_ref.set_bypass(true, true);
        eng_pol.set_bypass(true, true);
        let (mut brl, mut brr) = ([0.0_f32; 16], [0.0_f32; 16]);
        let (mut bpl, mut bpr) = ([0.0_f32; 16], [0.0_f32; 16]);
        eng_ref.process_block(&zeros, &zeros, &mut brl, &mut brr);
        eng_pol.process_block(&zeros, &zeros, &mut bpl, &mut bpr);
        assert!(brl[0].abs() > 1.0e-5);
        assert!((bpl[0] + 0.5 * brl[0]).abs() < 1.0e-3);

        let step_ref = eng_ref.step_frame_sona(0.0, 0.0, false);
        let step_pol = eng_pol.step_frame_sona(0.0, 0.0, false);
        assert!(step_ref[0].abs() > 1.0e-5);
        assert!((step_pol[0] + 0.5 * step_ref[0]).abs() < 1.0e-3);
    }

    #[test]
    fn test_stereo_all_four_polarity_truth_table_single_source() {
        use crate::schema::BlendMode;
        use crate::test_support::build_spatial_test_dnam_container_with_dnbn_polarity;

        let build_case = |dnbn_inv: bool, json_inv: bool| {
            let mut hdr = DnamJsonHeader::default_reference();
            hdr.io_config.capabilities.blend_mode = BlendMode::HostParallel;
            hdr.io_config.capabilities.default_blend = 0.4;
            hdr.io_config.capabilities.wet_polarity_inverted = json_inv;
            let bytes = build_spatial_test_dnam_container_with_dnbn_polarity(
                &mut hdr,
                Tier::Lite,
                dnbn_inv,
            );
            DnamEngine::<ScalarBackend>::from_bytes_owned(&bytes, Tier::Lite, 48_000).unwrap()
        };

        let in_l = [0.45_f32; 24];
        let in_r = [-0.25_f32; 24];
        let dry_l = 0.6 * in_l[0];
        let dry_r = 0.6 * in_r[0];

        let mut e_ff = build_case(false, false);
        let mut e_tf = build_case(true, false);
        let mut e_ft = build_case(false, true);
        let mut e_tt = build_case(true, true);

        let (mut l_ff, mut r_ff) = ([0.0_f32; 24], [0.0_f32; 24]);
        let (mut l_tf, mut r_tf) = ([0.0_f32; 24], [0.0_f32; 24]);
        let (mut l_ft, mut r_ft) = ([0.0_f32; 24], [0.0_f32; 24]);
        let (mut l_tt, mut r_tt) = ([0.0_f32; 24], [0.0_f32; 24]);
        e_ff.process_block(&in_l, &in_r, &mut l_ff, &mut r_ff);
        e_tf.process_block(&in_l, &in_r, &mut l_tf, &mut r_tf);
        e_ft.process_block(&in_l, &in_r, &mut l_ft, &mut r_ft);
        e_tt.process_block(&in_l, &in_r, &mut l_tt, &mut r_tt);

        for i in 0..24 {
            let wet_ff_l = l_ff[i] - dry_l;
            let wet_ff_r = r_ff[i] - dry_r;
            assert!(wet_ff_l.abs() > 1.0e-4);
            for (l_inv, r_inv) in [(&l_tf, &r_tf), (&l_ft, &r_ft), (&l_tt, &r_tt)] {
                assert!((l_inv[i] - (dry_l - wet_ff_l)).abs() < 1.0e-5);
                assert!((r_inv[i] - (dry_r - wet_ff_r)).abs() < 1.0e-5);
            }
        }

        // step_frame_sona and trails bypass agree identically across all 3 inverted cases.
        for eng in [&mut e_ff, &mut e_tf, &mut e_ft, &mut e_tt] {
            eng.set_bypass(true, true);
        }
        let s_ff = e_ff.step_frame_sona(0.3, -0.1, false);
        let s_tf = e_tf.step_frame_sona(0.3, -0.1, false);
        let s_ft = e_ft.step_frame_sona(0.3, -0.1, false);
        let s_tt = e_tt.step_frame_sona(0.3, -0.1, false);
        let wet_s_l = s_ff[0] - 0.3;
        let wet_s_r = s_ff[1] + 0.1;
        assert!(wet_s_l.abs() > 1.0e-4);
        for s_inv in [s_tf, s_ft, s_tt] {
            assert!((s_inv[0] - (0.3 - wet_s_l)).abs() < 1.0e-5);
            assert!((s_inv[1] - (-0.1 - wet_s_r)).abs() < 1.0e-5);
        }
        assert!(e_tf.telemetry_snapshot().mix_wet_gain < 0.0);
        assert!(e_ft.telemetry_snapshot().mix_wet_gain < 0.0);
        assert!(e_tt.telemetry_snapshot().mix_wet_gain < 0.0);
        assert!(e_ff.telemetry_snapshot().mix_wet_gain > 0.0);
    }
}
