//! Real-time audio processing loops for [`DnamEngine`].
//!
//! Submodules:
//! - [`multiport`]: 4-port (`4CM`, `WDW`, `Sidechain`) stereo processing loops.

pub mod multiport;

use super::{DnamEngine, EngineBackend};
use crate::math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, SAFETY_CEILING_F32, flush_recursive_state_f32,
    sanitize_input_sample, sanitize_output_sample,
};
use crate::stages::stage2_gru::kernels::V3HotScratch;
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

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Synchronizes static control projections, `current_stage_params`, and [`crate::stages::V3HotCore`]
    /// conditioning immediately without waiting for the next `3 kHz` tick.
    pub(crate) fn sync_immediate_control_state_inner(&mut self) {
        let arena = self.arena_storage.as_bytes();
        self.control_bus.refresh_static_state(arena);
        self.current_stage_params = self.control_bus.stage_targets;
        self.stage_ramp_deltas = [0.0; 16];
        self.hot_core
            .snap_control_bus_stage_targets(&self.current_stage_params);
    }

    /// Executes the decimated `3 kHz` Control Bus tick and updates ramp slopes.
    #[inline(never)]
    pub(crate) fn step_control_tick_3khz(&mut self) {
        let avg_energy = self.ctrl_energy_accum * self.zoh.alpha_ramp_host;
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        let det_env = self.stage0_router.detector_envelopes;
        let arena = self.arena_storage.as_bytes();
        let _changed = self
            .control_bus
            .tick_3khz_with_detectors(avg_energy, Some(det_env), arena);
        self.stage_ramp_deltas = if self.control_bus.unified_w_ctrl {
            crate::control::ControlBus::<B>::compute_micro_ramp_deltas(
                &self.current_stage_params,
                &self.control_bus.stage_targets,
                self.zoh.alpha_ramp_host,
            )
        } else {
            crate::control::ControlBus::<B>::compute_micro_ramp_deltas_12(
                &self.current_stage_params,
                &self.control_bus.stage_targets,
                self.zoh.alpha_ramp_host,
            )
        };
    }

    #[inline(always)]
    fn advance_control_subblock(&mut self, in_l: &[f32], in_r: &[f32]) {
        let n = in_l.len();
        let nf = n as f32;
        for r in 0..16 {
            self.current_stage_params[r] += nf * self.stage_ramp_deltas[r];
        }
        self.hot_core
            .apply_control_bus_stage_targets(&self.current_stage_params);
        let mut e = self.ctrl_energy_accum;
        for i in 0..n {
            e += 0.5 * (in_l[i].abs() + in_r[i].abs());
        }
        self.ctrl_energy_accum = flush_recursive_state_f32(e);
        self.ctrl_sub_counter += n;
        if self.ctrl_sub_counter >= self.zoh.control_decimation_host.max(1) {
            self.step_control_tick_3khz();
        }
    }

    #[inline(always)]
    fn finalize_stereo_output_slice(
        &mut self,
        dry_l: &[f32],
        dry_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let tier_trim = self.cached_tier_gain_trim[self.effective_tier() as usize];
        let eff_dry_g = if self.bypassed {
            1.0
        } else {
            self.mix_dry_gain * tier_trim
        };
        let eff_wet_g = if self.bypassed {
            self.mix_wet_gain
        } else {
            self.mix_wet_gain * tier_trim
        };
        let width = self.stereo_width;
        let width_active = !self.hot_core.mono_out && (width - 1.0).abs() > 1.0e-6;
        let mix_active = eff_dry_g != 0.0 || (eff_wet_g - 1.0).abs() > 1.0e-6;
        let ceil_bits = SAFETY_CEILING_F32.to_bits();

        if !width_active && !mix_active {
            if all_within_envelope_bits(out_l, ceil_bits)
                && all_within_envelope_bits(out_r, ceil_bits)
            {
                return;
            }
            for i in 0..out_l.len() {
                out_l[i] = sanitize_output_sample(out_l[i], &mut self.saturation_flags);
                out_r[i] = sanitize_output_sample(out_r[i], &mut self.saturation_flags);
            }
            return;
        }

        for i in 0..out_l.len() {
            let mut wl = out_l[i];
            let mut wr = out_r[i];
            if width_active {
                let mid = 0.5 * (wl + wr);
                let side = 0.5 * (wl - wr);
                wl = mid + width * side;
                wr = mid - width * side;
            }
            let yl = eff_dry_g * dry_l[i] + eff_wet_g * wl;
            let yr = eff_dry_g * dry_r[i] + eff_wet_g * wr;
            out_l[i] = sanitize_output_sample(yl, &mut self.saturation_flags);
            out_r[i] = sanitize_output_sample(yr, &mut self.saturation_flags);
        }
    }

    #[inline(never)]
    fn process_block_clean_unity_or_ctrl(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let num_frames = left_in.len();
        let mut sc = V3HotScratch::for_core(&self.hot_core);
        let ctrl_active = self.control_bus.num_controls > 0
            || self.control_bus.is_dirty
            || self.control_bus.live_analog_drift_enabled;
        if !ctrl_active {
            self.hot_core
                .process_subblock_no_ftz(&mut sc, left_in, right_in, left_out, right_out);
            self.finalize_stereo_output_slice(left_in, right_in, left_out, right_out);
            return;
        }

        let k_ctrl = self.zoh.control_decimation_host.max(1);
        let mut off = 0;
        while off < num_frames {
            let rem_tick = k_ctrl.saturating_sub(self.ctrl_sub_counter).max(1);
            let chunk = rem_tick.min(num_frames - off);
            let sub_in_l = &left_in[off..off + chunk];
            let sub_in_r = &right_in[off..off + chunk];
            let sub_out_l = &mut left_out[off..off + chunk];
            let sub_out_r = &mut right_out[off..off + chunk];
            let nf = chunk as f32;
            for r in 0..16 {
                self.current_stage_params[r] += nf * self.stage_ramp_deltas[r];
            }
            self.hot_core
                .apply_control_bus_stage_targets(&self.current_stage_params);
            self.hot_core
                .process_subblock_no_ftz(&mut sc, sub_in_l, sub_in_r, sub_out_l, sub_out_r);
            let mut e = self.ctrl_energy_accum;
            for i in 0..chunk {
                e += 0.5 * (sub_in_l[i].abs() + sub_in_r[i].abs());
            }
            self.ctrl_energy_accum = flush_recursive_state_f32(e);
            self.ctrl_sub_counter += chunk;
            if self.ctrl_sub_counter >= k_ctrl {
                self.step_control_tick_3khz();
            }
            self.finalize_stereo_output_slice(sub_in_l, sub_in_r, sub_out_l, sub_out_r);
            off += chunk;
        }
    }

    #[inline(never)]
    fn process_block_chunked_guarded(
        &mut self,
        sc: &mut V3HotScratch,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let num_frames = left_in.len();
        let k_ctrl = self.zoh.control_decimation_host.max(1);
        let ctrl_active = self.control_bus.num_controls > 0
            || self.control_bus.is_dirty
            || self.control_bus.live_analog_drift_enabled;
        let is_frozen = self.hot_core.stage4.freeze;
        let live_scale = if self.bypassed || (self.freeze_mutes_dry && is_frozen) {
            0.0_f32
        } else {
            1.0_f32
        };

        let mut san_l = [0.0_f32; 64];
        let mut san_r = [0.0_f32; 64];
        let mut eff_l = [0.0_f32; 64];
        let mut eff_r = [0.0_f32; 64];

        let mut off = 0;
        while off < num_frames {
            let max_chunk = if ctrl_active {
                k_ctrl.saturating_sub(self.ctrl_sub_counter).clamp(1, 64)
            } else {
                64
            };
            let n = (num_frames - off).min(max_chunk);
            for i in 0..n {
                let raw_l = left_in[off + i];
                let raw_r = if self.single_input {
                    raw_l
                } else {
                    right_in[off + i]
                };
                let sl = sanitize_input_sample(
                    raw_l,
                    CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                    &mut self.saturation_flags,
                );
                let sr = sanitize_input_sample(
                    raw_r,
                    CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                    &mut self.saturation_flags,
                );
                san_l[i] = sl;
                san_r[i] = sr;
                eff_l[i] = sl * live_scale;
                eff_r[i] = sr * live_scale;
            }
            if ctrl_active {
                self.advance_control_subblock(&eff_l[..n], &eff_r[..n]);
            }
            self.hot_core.process_subblock_no_ftz(
                sc,
                &eff_l[..n],
                &eff_r[..n],
                &mut left_out[off..off + n],
                &mut right_out[off..off + n],
            );
            self.finalize_stereo_output_slice(
                &san_l[..n],
                &san_r[..n],
                &mut left_out[off..off + n],
                &mut right_out[off..off + n],
            );
            off += n;
        }
    }

    /// Low-level single-frame TORA stepping (`0 FDIV`, `0 FSQRT`).
    ///
    /// Direct callers must hold a [`FtzDazGuard`] for the enclosing batch. This
    /// method does not perform block-level preview accounting, tempo advancement,
    /// or offline oversampling; use the block APIs when those behaviors are needed.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    #[inline(never)]
    pub fn step_frame_tora(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
        self.sync_hot_core_tier();
        let in_r = if self.single_input { in_l } else { in_r };
        let in_l = sanitize_input_sample(
            in_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let in_r = sanitize_input_sample(
            in_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        if self.bypassed && !self.trails_enabled {
            return if mono_only {
                [in_l, in_l]
            } else {
                [in_l, in_r]
            };
        }

        let is_frozen = self.hot_core.stage4.freeze;
        let live_scale = if self.bypassed || (self.freeze_mutes_dry && is_frozen) {
            0.0
        } else {
            1.0
        };
        let eff_in_l = in_l * live_scale;
        let eff_in_r = in_r * live_scale;

        let k_ctrl = self.zoh.control_decimation_host.max(1);
        self.ctrl_energy_accum = flush_recursive_state_f32(
            self.ctrl_energy_accum + 0.5 * (eff_in_l.abs() + eff_in_r.abs()),
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            self.step_control_tick_3khz();
        }
        if self.control_bus.unified_w_ctrl {
            crate::control::ControlBus::<B>::step_micro_ramp_16(
                &mut self.current_stage_params,
                &self.stage_ramp_deltas,
            );
        } else {
            crate::control::ControlBus::<B>::step_micro_ramp_12(
                &mut self.current_stage_params,
                &self.stage_ramp_deltas,
            );
        }
        self.hot_core
            .apply_control_bus_stage_targets(&self.current_stage_params);

        let [mut wet_l, mut wet_r] = self.hot_core.step_stereo(eff_in_l, eff_in_r);

        if !mono_only && !self.hot_core.mono_out && (self.stereo_width - 1.0).abs() > 1e-6 {
            let mid = 0.5 * (wet_l + wet_r);
            let side = 0.5 * (wet_l - wet_r);
            wet_l = mid + self.stereo_width * side;
            wet_r = mid - self.stereo_width * side;
        }

        let tier_trim = self.cached_tier_gain_trim[self.effective_tier() as usize];
        let eff_dry_g = if self.bypassed {
            1.0
        } else {
            self.mix_dry_gain * tier_trim
        };
        let eff_wet_g = if self.bypassed {
            self.mix_wet_gain
        } else {
            self.mix_wet_gain * tier_trim
        };

        let out_l = sanitize_output_sample(
            eff_dry_g * in_l + eff_wet_g * wet_l,
            &mut self.saturation_flags,
        );
        let out_r = if mono_only {
            out_l
        } else {
            sanitize_output_sample(
                eff_dry_g * in_r + eff_wet_g * wet_r,
                &mut self.saturation_flags,
            )
        };

        [out_l, out_r]
    }

    /// Internal 4x oversampled stepping.
    #[inline(never)]
    pub(crate) fn step_frame_4x_tora(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
        let in_r = if self.single_input { in_l } else { in_r };
        let in_l = sanitize_input_sample(
            in_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let in_r = sanitize_input_sample(
            in_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        self.sync_hot_core_tier();
        let up_frames = self.oversampler.upsample_4x(in_l, in_r);
        let mut proc_frames = [[0.0_f32; 2]; 4];
        for (m, frame) in up_frames.iter().enumerate() {
            proc_frames[m] = self.step_frame_tora(frame[0], frame[1], mono_only);
        }
        let [dl, dr] = self.oversampler.downsample_4x(proc_frames);
        [
            sanitize_output_sample(dl, &mut self.saturation_flags),
            sanitize_output_sample(dr, &mut self.saturation_flags),
        ]
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
        let num_frames = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        self.tempo_state.advance_frames(num_frames);
        self.sync_hot_core_tier();

        let li = &left_in[..num_frames];
        let ri = if self.single_input {
            li
        } else {
            &right_in[..num_frames]
        };
        let lo = &mut left_out[..num_frames];
        let ro = &mut right_out[..num_frames];

        if self.bypassed && !self.trails_enabled {
            for i in 0..num_frames {
                lo[i] = sanitize_input_sample(
                    li[i],
                    CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                    &mut self.saturation_flags,
                );
                ro[i] = sanitize_input_sample(
                    ri[i],
                    CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                    &mut self.saturation_flags,
                );
            }
            return;
        }

        if self.offline_bounce {
            for i in 0..num_frames {
                let [l, r] = self.step_frame_4x_tora(li[i], ri[i], false);
                lo[i] = l;
                ro[i] = r;
            }
            return;
        }

        let env_bits = CERTIFIED_INPUT_ENVELOPE_MIN_F32.to_bits();
        let clean_inputs = !self.bypassed
            && !(self.freeze_mutes_dry && self.hot_core.stage4.freeze)
            && all_within_envelope_bits(li, env_bits)
            && (self.single_input || all_within_envelope_bits(ri, env_bits));
        if clean_inputs {
            self.process_block_clean_unity_or_ctrl(li, ri, lo, ro);
        } else {
            let mut sc = V3HotScratch::for_core(&self.hot_core);
            self.process_block_chunked_guarded(&mut sc, li, ri, lo, ro);
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
        let _ftz = FtzDazGuard::new();
        let num_frames = mono_in.len().min(mono_out.len());
        self.tempo_state.advance_frames(num_frames);
        self.sync_hot_core_tier();
        let mi = &mono_in[..num_frames];
        let mo = &mut mono_out[..num_frames];

        if self.bypassed && !self.trails_enabled {
            for i in 0..num_frames {
                mo[i] = sanitize_input_sample(
                    mi[i],
                    CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                    &mut self.saturation_flags,
                );
            }
            return;
        }
        if self.offline_bounce {
            for i in 0..num_frames {
                let [l, _] = self.step_frame_4x_tora(mi[i], mi[i], true);
                mo[i] = l;
            }
            return;
        }
        let mut sc = V3HotScratch::for_core(&self.hot_core);
        let mut scratch_r = [0.0_f32; 64];
        let mut off = 0;
        while off < num_frames {
            let n = (num_frames - off).min(64);
            self.process_block_chunked_guarded(
                &mut sc,
                &mi[off..off + n],
                &mi[off..off + n],
                &mut mo[off..off + n],
                &mut scratch_r[..n],
            );
            off += n;
        }
    }

    /// Processes in-place stereo audio blocks.
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        let _ftz = FtzDazGuard::new();
        let num_frames = left_io.len().min(right_io.len());
        self.tempo_state.advance_frames(num_frames);
        self.sync_hot_core_tier();

        if self.offline_bounce || (self.bypassed && !self.trails_enabled) {
            for i in 0..num_frames {
                let [l, r] = if self.offline_bounce {
                    self.step_frame_4x_tora(left_io[i], right_io[i], false)
                } else {
                    self.step_frame_tora(left_io[i], right_io[i], false)
                };
                left_io[i] = l;
                right_io[i] = r;
            }
            return;
        }

        let mut sc = V3HotScratch::for_core(&self.hot_core);
        let mut in_l_buf = [0.0_f32; 64];
        let mut in_r_buf = [0.0_f32; 64];
        let mut off = 0;
        while off < num_frames {
            let n = (num_frames - off).min(64);
            in_l_buf[..n].copy_from_slice(&left_io[off..off + n]);
            in_r_buf[..n].copy_from_slice(&right_io[off..off + n]);
            self.process_block_chunked_guarded(
                &mut sc,
                &in_l_buf[..n],
                &in_r_buf[..n],
                &mut left_io[off..off + n],
                &mut right_io[off..off + n],
            );
            off += n;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::engine::DnamEngine;
    use crate::math::ScalarBackend;
    use crate::schema::DnamJsonHeader;
    use crate::stages::weights::ARENA0_STAGE3_EARLY_FIR_OFFSET;
    use crate::types::Tier;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_process_modes_and_in_place() {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.capabilities.supports_freeze = true;
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48000).unwrap();

        let mut l = [0.1f32; 16];
        let mut r = [-0.1f32; 16];
        engine.process_block_in_place(&mut l, &mut r);
        assert!(l[0].is_finite());
        assert!(r[0].is_finite());

        // Offline bounce mode in all process functions
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

        // Bypassed with trails
        engine.set_offline_bounce(false);
        engine.set_bypass(true, true);
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());

        // Bypassed without trails (stereo & mono)
        engine.set_bypass(true, false);
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert_eq!(l_out, l);
        engine.process_mono(&l, &mut l_out);
        assert_eq!(l_out, l);

        // Freeze mutes dry
        engine.set_bypass(false, false);
        engine.set_freeze(true);
        engine.freeze_mutes_dry = true;
        engine.process_block(&l, &r, &mut l_out, &mut r_out);
        assert!(l_out[0].is_finite());
    }

    #[test]
    fn single_input_container_duplicates_left_and_ignores_right_channel() {
        let mut header = DnamJsonHeader::default_reference();
        header.io_config.inputs = 1;
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        // Populate Stage 0 router (pass-through) + Stage III Early FIR (L/R tap 0)
        arena[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
        arena[20..24].copy_from_slice(&1.0_f32.to_le_bytes());
        arena[ARENA0_STAGE3_EARLY_FIR_OFFSET..ARENA0_STAGE3_EARLY_FIR_OFFSET + 4]
            .copy_from_slice(&1.0_f32.to_le_bytes());
        arena[ARENA0_STAGE3_EARLY_FIR_OFFSET + 64..ARENA0_STAGE3_EARLY_FIR_OFFSET + 68]
            .copy_from_slice(&1.0_f32.to_le_bytes());
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut eng_diff =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_same =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        for _ in 0..8 {
            let y_diff = eng_diff.step_frame_tora(0.25, -0.73, false);
            let y_same = eng_same.step_frame_tora(0.25, 0.25, false);
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
}
