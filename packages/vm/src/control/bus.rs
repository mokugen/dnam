//! Dual-mode `#![no_std]` + `!alloc`-capable `3 kHz` [`ControlBus`] and [`ControlBusBacking`].

pub mod dispatch;
pub mod laws;
mod load;
mod tick;

#[cfg(feature = "alloc")]
use alloc::vec;
use core::marker::PhantomData;

use dispatch::{
    DEFAULT_DRIFT_PRNG_SEED, DispatchInputs, DispatchMatrices, STAGE_TARGET_SLOTS,
    advance_micro_ramp_frame, control_stride_for_rate, control_tick_ratio, evaluate_w_ctrl_32,
    prepare_micro_ramp_deltas, step_ou_drift,
};
use tick::{
    StaticEvalInputs, StaticEvalOutputs, evaluate_static_cp1_latents, step_lfos_and_harmonics,
    update_env_states,
};

use crate::bin::TempoSyncEntryWire;
use crate::control::exact_int::ExactIntFactorAccumulator;
use crate::control::tempo::apply_tsyn_table;
use crate::control::{ControlClampDescriptor, Cp1ContributorSlot};
use crate::math::{
    DefaultMathBackend, MAX_ROTOR_ANGLE_F32_BITS, MathBackend, Q0_192Phase, powf_f64, rcp_nr6_f64,
    round_f64,
};
use crate::types::{ColdBuf, ColdSlice, Tier, VmError};

/// Mutable backing buffers for [`ControlBus`], supporting both `--no-default-features`
/// (`Borrowed`) and optional `alloc` (`Owned`) initialization.
#[derive(Debug)]
pub struct ControlBusBacking<'a> {
    /// Conditional clamp descriptors (`len >= num_controls`).
    pub descriptors: ColdBuf<'a, ControlClampDescriptor>,
    /// Raw physical values (`len >= num_controls`).
    pub raw_values: ColdBuf<'a, f32>,
    /// Clamped physical values (`len >= num_controls`).
    pub clamped_values: ColdBuf<'a, f32>,
    /// Normalized coordinates in `[-1, +1]` (`len >= num_controls`).
    pub normalized_values: ColdBuf<'a, f32>,
    /// Active switch option per control (`len >= num_controls`).
    pub active_switch_options: ColdBuf<'a, u8>,
    /// Decoded `CP1` contributor slots (`len >= num_contributors`).
    pub contributors: ColdBuf<'a, Cp1ContributorSlot>,
    /// Pure-binary `"TSYN"` tempo rows.
    pub tsyn_entries: ColdSlice<'a, TempoSyncEntryWire>,
}

impl<'a> ControlBusBacking<'a> {
    /// Wraps caller-supplied mutable slices with zero heap allocation (`--no-default-features`).
    pub fn from_slices(
        descriptors: &'a mut [ControlClampDescriptor],
        raw_values: &'a mut [f32],
        clamped_values: &'a mut [f32],
        normalized_values: &'a mut [f32],
        active_switch_options: &'a mut [u8],
        contributors: &'a mut [Cp1ContributorSlot],
        tsyn_entries: &'a [TempoSyncEntryWire],
    ) -> Result<Self, VmError> {
        Ok(Self {
            descriptors: ColdBuf::borrowed(descriptors),
            raw_values: ColdBuf::borrowed(raw_values),
            clamped_values: ColdBuf::borrowed(clamped_values),
            normalized_values: ColdBuf::borrowed(normalized_values),
            active_switch_options: ColdBuf::borrowed(active_switch_options),
            contributors: ColdBuf::borrowed(contributors),
            tsyn_entries: ColdSlice::borrowed(tsyn_entries),
        })
    }

    /// Allocates owned backing storage sized for `num_controls` and `max_contributors`.
    #[cfg(feature = "alloc")]
    pub fn new_owned(
        num_controls: usize,
        max_contributors: usize,
        tsyn: &[TempoSyncEntryWire],
    ) -> Self {
        let cap = num_controls.max(1);
        let c_cap = max_contributors.max(8);
        Self {
            descriptors: ColdBuf::Owned(vec![ControlClampDescriptor::empty(); cap]),
            raw_values: ColdBuf::Owned(vec![0.0; cap]),
            clamped_values: ColdBuf::Owned(vec![0.0; cap]),
            normalized_values: ColdBuf::Owned(vec![0.0; cap]),
            active_switch_options: ColdBuf::Owned(vec![0; cap]),
            contributors: ColdBuf::Owned(vec![Cp1ContributorSlot::zero(); c_cap]),
            tsyn_entries: ColdSlice::from_slice_owned(tsyn),
        }
    }
}

/// Parametric `3 kHz` `ControlBus` supporting `Lite..Ultra`, `"CEXT"`, `"TSYN"`, `"CP1\0"`,
/// `9/18-word` exact-integer sufficient statistics, and division-free micro-ramping.
pub struct ControlBus<'a, B: MathBackend = DefaultMathBackend> {
    /// Unsmoothed `3 kHz` dispatch targets `[0..16]` base + `[16..32]` `"CEXT"`.
    pub stage_targets: [f32; STAGE_TARGET_SLOTS],
    /// Previous `K_ctrl` boundary target snapshot.
    pub prev_stage_targets: [f32; STAGE_TARGET_SLOTS],
    /// Per-VM-frame ramped stage parameters (`[16]` snaps immediately; `[31] == 0.0`).
    pub current_stage_params: [f32; STAGE_TARGET_SLOTS],
    /// Per-VM-frame micro-ramp step delta (`[16] == 0.0`, `[31] == 0.0`).
    pub stage_ramp_deltas: [f32; STAGE_TARGET_SLOTS],
    /// Base `SO(4)` gauge-covariant head coordinates `[0..4]`.
    pub z_head: [f32; 4],
    /// Base unrotated stage sensitivity coordinates `[0..4]`.
    pub ctrl_basis_4: [f32; 4],
    /// Ultra gauge-covariant head coordinates `[0..4]`.
    pub z_head_ultra: [f32; 4],
    /// Ultra unrotated stage sensitivity coordinates `[4..8]`.
    pub z_stage_ultra: [f32; 4],
    /// Two `Q0.192` fixed-point LFO phase accumulators.
    pub lfo_phases_q192: [Q0_192Phase; 2],
    /// Host-rate-scaled `f32` LFO increments as raw IEEE-754 bits.
    pub lfo_dphi_host_bits: [u32; 2],
    /// Wire reference `3 kHz` LFO increments `[f32; 2]`.
    pub lfo_dphi_train: [f32; 2],
    /// Unit phasor state `[[cos, sin]; 2]`.
    pub lfo_phasors: [[f32; 2]; 2],
    /// Chebyshev harmonics `[T1..T4]` for LFO 0 and LFO 1.
    pub lfo_harmonics: [f32; 8],
    /// Host-rate ZOH-scaled standalone envelope poles `[0..3]`.
    pub env_poles_host: [f32; 3],
    /// Reference `3 kHz` standalone envelope poles `[0..3]`.
    pub env_poles_train: [f32; 3],
    /// Active 4-band envelope vector `[e_0, e_1, e_2, e_3]`.
    pub env_states: [f32; 4],
    /// 4-channel OU analog drift state.
    pub ou_drift_state: [f32; 4],
    /// `xorshift64*` analog drift PRNG seed/state.
    pub drift_prng_state: u64,
    /// Drift amplitude `[0.0, 2.0]`.
    pub sigma_drift: f32,
    /// Reference `3 kHz` drift autocorrelation pole `(0.0, 0.9999]`.
    pub rho_drift_train: f32,
    /// Host ZOH-scaled drift autocorrelation pole.
    pub rho_drift_host: f32,
    /// 4-D trust-region radius (`[0.01, 16.0]`, default `2.5`).
    pub trust_radius_mh: f32,
    /// Wire `48 kHz` anti-zipper pole scalar (`(0.0, 1.0]`).
    pub anti_zipper_scale: f32,
    /// Host micro-ramp alpha (`clamp(s_ramp * 16.0 / K_ctrl, 0.0, 1.0)`).
    pub alpha_ramp: f32,
    /// Active tier.
    pub tier: Tier,
    /// Reference model sample rate in Hz (`44_100..=96_000`).
    pub ref_sample_rate_hz: f32,
    /// Current host VM sample rate in Hz.
    pub host_sample_rate_hz: f32,
    /// Decimation stride $K_{\text{ctrl}}(f_{s,\text{vm}}) \ge 8$.
    pub ctrl_stride: u32,
    /// Precomputed reciprocal stride `1.0 / (ctrl_stride as f32)` for RT `step_frame`.
    pub inv_ctrl_stride: f32,
    /// Sub-window frame counter `0..ctrl_stride`.
    pub ctrl_sub_counter: u32,
    /// Running sum of `(0.5 * (|L| + |R|))` for standalone envelope mode.
    window_abs_acc: f32,
    /// Active physical control count `P`.
    pub num_controls: usize,
    /// Active `CP1` contributor count.
    pub num_contributors: usize,
    /// Distinct physical controls referenced by `CP1`.
    pub distinct_contributor_controls: usize,
    /// Active `TSYN` row count.
    pub tsyn_count: usize,
    /// Whether `lfo` or `W_ctrl[:, 8..16]` modulators are non-zero.
    pub has_internal_modulators: bool,
    /// Whether static control state changed since last boundary.
    pub is_dirty: bool,
    /// Live OU analog drift enable flag (automatically gated by `!offline_bounce`).
    pub live_analog_drift_enabled: bool,
    /// Offline deterministic bounce mode (disables stochastic drift).
    pub offline_bounce: bool,
    /// Base `U_int[4][8]` interaction matrix.
    pub u_int_base: [[f32; 8]; 4],
    /// Ultra delta `3 × U_int[4][8]`.
    pub u_int_ultra: [[[f32; 8]; 4]; 3],
    /// Base dispatch matrix `W_ctrl[16][16]`.
    pub w_ctrl: [[f32; 16]; 16],
    /// Ultra dispatch matrix `W_ctrl_ultra[16][4]`.
    pub w_ctrl_ultra: [[f32; 4]; 16],
    /// Whether `1,280 B` `"CEXT"` matrices are present.
    pub cext_present: bool,
    /// Extended dispatch matrix `w_ctrl_ext[16][16]`.
    pub w_ctrl_ext: [[f32; 16]; 16],
    /// Extended Ultra delta `w_ctrl_ultra_ext[16][4]`.
    pub w_ctrl_ultra_ext: [[f32; 4]; 16],
    /// Base rank-4 FM exact-integer accumulators.
    exact_fm_base: [ExactIntFactorAccumulator; 4],
    /// Ultra rank-4 FM exact-integer accumulators.
    exact_fm_ultra: [ExactIntFactorAccumulator; 4],
    /// Mutable/borrowed storage backing.
    pub backing: ControlBusBacking<'a>,
    _backend: PhantomData<B>,
}

impl<B: MathBackend> core::fmt::Debug for ControlBus<'_, B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ControlBus")
            .field("backend", &B::NAME)
            .field("tier", &self.tier)
            .field("num_controls", &self.num_controls)
            .field("num_contributors", &self.num_contributors)
            .field("ctrl_stride", &self.ctrl_stride)
            .field("cext_present", &self.cext_present)
            .finish()
    }
}

impl<'a, B: MathBackend> ControlBus<'a, B> {
    pub(crate) fn configure_sample_rate(&mut self, sample_rate_hz: f32) {
        self.host_sample_rate_hz = sample_rate_hz;
        let stride = control_stride_for_rate(sample_rate_hz);
        self.ctrl_stride = stride;
        let inv_stride = rcp_nr6_f64(f64::from(stride));
        self.inv_ctrl_stride = inv_stride as f32;
        let rho_ctrl = control_tick_ratio(sample_rate_hz, self.ref_sample_rate_hz);
        for i in 0..2 {
            let train = self.lfo_dphi_train[i];
            if (train.to_bits() & 0x7FFF_FFFF) == 0 {
                self.lfo_dphi_host_bits[i] = 0;
            } else {
                let scaled = (f64::from(train) * rho_ctrl) as f32;
                let max_ang = f32::from_bits(MAX_ROTOR_ANGLE_F32_BITS);
                self.lfo_dphi_host_bits[i] = scaled.clamp(-max_ang, max_ang).to_bits();
            }
        }
        for b in 0..3 {
            let p = self.env_poles_train[b];
            self.env_poles_host[b] = (powf_f64(f64::from(p), rho_ctrl) as f32).clamp(0.0, 0.99995);
        }
        self.rho_drift_host =
            (powf_f64(f64::from(self.rho_drift_train), rho_ctrl) as f32).clamp(0.0, 0.9999);
        let alpha = (f64::from(self.anti_zipper_scale) * 16.0 * inv_stride) as f32;
        self.alpha_ramp = alpha.clamp(0.0, 1.0);
    }

    /// Re-times an in-flight control window cleanly upon an admissible host sample-rate change
    /// (`r[dsp.ctrl.op-m.rate-retime]`).
    pub fn retime_sample_rate(&mut self, new_sample_rate_hz: f32) {
        if !new_sample_rate_hz.is_finite() || new_sample_rate_hz <= 0.0 {
            return;
        }
        let k_old = self.ctrl_stride;
        let consumed = self.ctrl_sub_counter.min(k_old.saturating_sub(1));
        self.configure_sample_rate(new_sample_rate_hz);
        let k_new = self.ctrl_stride;
        let rem_new = round_f64(
            f64::from(k_old - consumed) * f64::from(k_new) * rcp_nr6_f64(f64::from(k_old)),
        ) as u32;
        let clamped_rem = rem_new.min(k_new);
        self.ctrl_sub_counter = k_new.saturating_sub(clamped_rem);
    }

    /// Resets dynamic modulators, OU drift, exact accumulators, and snaps controls to defaults
    /// at resting zero equilibrium (`ref[dsp.ctrl.modulators.dynamic]`).
    pub fn reset(&mut self) {
        let p = self.num_controls;
        for i in 0..p {
            self.backing.raw_values[i] = self.backing.descriptors[i].default_val;
        }
        for acc in &mut self.exact_fm_base {
            acc.reset();
        }
        for acc in &mut self.exact_fm_ultra {
            acc.reset();
        }
        self.lfo_phases_q192 = [Q0_192Phase::ZERO; 2];
        self.lfo_phasors = [[1.0, 0.0]; 2];
        self.lfo_harmonics = [0.0; 8];
        self.env_states = [0.0; 4];
        self.ou_drift_state = [0.0; 4];
        self.drift_prng_state = DEFAULT_DRIFT_PRNG_SEED;
        self.ctrl_sub_counter = 0;
        self.window_abs_acc = 0.0;
        self.is_dirty = true;

        self.clamp_and_eval_static_if_dirty();
        let q2_init = [0.0_f32, 0.0, 1.0, 0.0];
        let q3_zero = [0.0_f32; 4];
        evaluate_w_ctrl_32::<B>(
            DispatchInputs {
                q16_head: &self.z_head,
                q16_basis: &self.ctrl_basis_4,
                q16_lfo: &q2_init,
                q16_env: &q3_zero,
                qu: &self.z_stage_ultra,
                is_ultra: self.tier == Tier::Ultra,
                cext_present: self.cext_present,
            },
            DispatchMatrices {
                w_ctrl: &self.w_ctrl,
                w_ctrl_ultra: &self.w_ctrl_ultra,
                w_ctrl_ext: &self.w_ctrl_ext,
                w_ctrl_ultra_ext: &self.w_ctrl_ultra_ext,
            },
            &mut self.stage_targets,
        );
        self.current_stage_params = self.stage_targets;
        self.prev_stage_targets = self.stage_targets;
        self.stage_ramp_deltas = [0.0; STAGE_TARGET_SLOTS];
    }

    /// Updates a single physical control value and marks static control state dirty.
    pub fn set_control_physical(&mut self, ctrl_idx: usize, val: f32) -> Result<(), VmError> {
        if ctrl_idx >= self.num_controls {
            return Err(VmError::InvalidControlDescriptor);
        }
        self.backing.raw_values[ctrl_idx] = val;
        self.is_dirty = true;
        Ok(())
    }

    /// Applies an atomic batch of `(ctrl_idx, physical_value)` updates (`r[dsp.ctrl.switch.conditional-batch]`).
    pub fn set_controls_batch(&mut self, updates: &[(usize, f32)]) -> Result<(), VmError> {
        for &(idx, _) in updates {
            if idx >= self.num_controls {
                return Err(VmError::InvalidControlDescriptor);
            }
        }
        for &(idx, val) in updates {
            self.backing.raw_values[idx] = val;
        }
        self.is_dirty = true;
        Ok(())
    }

    /// Applies active `"TSYN"` rows at `bpm` over the current active switch state and marks dirty.
    pub fn apply_tempo_bpm(&mut self, bpm: f32) -> usize {
        if self.tsyn_count == 0 {
            return 0;
        }
        let n = self.num_controls;
        ControlClampDescriptor::clamp_and_normalize_all(
            &self.backing.descriptors[..n],
            n,
            &self.backing.raw_values[..n],
            &mut self.backing.clamped_values[..n],
            &mut self.backing.normalized_values[..n],
            &mut self.backing.active_switch_options[..n],
        );
        let updated = apply_tsyn_table(
            self.backing.tsyn_entries.as_slice(),
            &self.backing.descriptors[..n],
            &self.backing.active_switch_options[..n],
            bpm,
            &mut self.backing.raw_values[..n],
            &mut self.backing.clamped_values[..n],
            &mut self.backing.normalized_values[..n],
        );
        if updated > 0 {
            self.is_dirty = true;
        }
        updated
    }

    /// Returns the concatenated unrotated 8-D stage sensitivity vector `[ctrl_basis_4[0..4], z_stage_ultra[0..4]]`.
    #[inline]
    pub const fn z_stage(&self) -> [f32; 8] {
        [
            self.ctrl_basis_4[0],
            self.ctrl_basis_4[1],
            self.ctrl_basis_4[2],
            self.ctrl_basis_4[3],
            self.z_stage_ultra[0],
            self.z_stage_ultra[1],
            self.z_stage_ultra[2],
            self.z_stage_ultra[3],
        ]
    }

    /// Returns `true` when any control, tempo row, internal modulator, or live drift requires 3 kHz ticking.
    #[inline]
    pub const fn should_tick(&self) -> bool {
        self.num_controls > 0
            || self.tsyn_count > 0
            || self.has_internal_modulators
            || self.is_dirty
            || (self.live_analog_drift_enabled && !self.offline_bounce && self.sigma_drift > 0.0)
    }

    /// Re-evaluates static `CP1` + exact-integer FM + 4-D trust clips if `is_dirty`, clearing `is_dirty`.
    pub fn clamp_and_eval_static_if_dirty(&mut self) {
        if !self.is_dirty {
            return;
        }
        let n = self.num_controls;
        if n > 0 {
            ControlClampDescriptor::clamp_and_normalize_all(
                &self.backing.descriptors[..n],
                n,
                &self.backing.raw_values[..n],
                &mut self.backing.clamped_values[..n],
                &mut self.backing.normalized_values[..n],
                &mut self.backing.active_switch_options[..n],
            );
            evaluate_static_cp1_latents(
                StaticEvalInputs {
                    descriptors: &self.backing.descriptors[..n],
                    norm_vals: &self.backing.normalized_values[..n],
                    active_switches: &self.backing.active_switch_options[..n],
                    contributors: &self.backing.contributors[..self.num_contributors],
                    distinct_controls: self.distinct_contributor_controls,
                    is_ultra: self.tier == Tier::Ultra,
                    trust_radius_mh: self.trust_radius_mh,
                    u_int_base: &self.u_int_base,
                    u_int_ultra: &self.u_int_ultra,
                },
                StaticEvalOutputs {
                    exact_fm_base: &mut self.exact_fm_base,
                    exact_fm_ultra: &mut self.exact_fm_ultra,
                    z_head: &mut self.z_head,
                    ctrl_basis_4: &mut self.ctrl_basis_4,
                    z_head_ultra: &mut self.z_head_ultra,
                    z_stage_ultra: &mut self.z_stage_ultra,
                },
            );
        } else {
            self.z_head = [0.0; 4];
            self.ctrl_basis_4 = [0.0; 4];
            self.z_head_ultra = [0.0; 4];
            self.z_stage_ultra = [0.0; 4];
        }
        self.is_dirty = false;
    }

    /// Executes one `3 kHz` control-boundary update: optional static basis eval (only if `is_dirty`),
    /// `Q0.192` LFO step, envelope update, `W_ctrl` + `"CEXT"` dispatch, OU drift, and ramp setup
    /// (with immediate snap on slot `16` `op_t_we_drive`).
    pub fn tick_control_boundary(
        &mut self,
        router_envelopes: Option<[f32; 4]>,
        audio_abs_energy: f32,
    ) {
        if !self.should_tick() {
            return;
        }
        self.clamp_and_eval_static_if_dirty();
        let q2 = step_lfos_and_harmonics(
            &mut self.lfo_phases_q192,
            &self.lfo_dphi_host_bits,
            &mut self.lfo_phasors,
            &mut self.lfo_harmonics,
        );
        update_env_states(
            &mut self.env_states,
            &self.env_poles_host,
            router_envelopes,
            audio_abs_energy,
        );
        evaluate_w_ctrl_32::<B>(
            DispatchInputs {
                q16_head: &self.z_head,
                q16_basis: &self.ctrl_basis_4,
                q16_lfo: &q2,
                q16_env: &self.env_states,
                qu: &self.z_stage_ultra,
                is_ultra: self.tier == Tier::Ultra,
                cext_present: self.cext_present,
            },
            DispatchMatrices {
                w_ctrl: &self.w_ctrl,
                w_ctrl_ultra: &self.w_ctrl_ultra,
                w_ctrl_ext: &self.w_ctrl_ext,
                w_ctrl_ultra_ext: &self.w_ctrl_ultra_ext,
            },
            &mut self.stage_targets,
        );
        step_ou_drift(
            self.live_analog_drift_enabled,
            self.offline_bounce,
            self.sigma_drift,
            self.rho_drift_host,
            &mut self.drift_prng_state,
            &mut self.ou_drift_state,
            &mut self.stage_targets,
        );
        prepare_micro_ramp_deltas(
            &self.stage_targets,
            &mut self.prev_stage_targets,
            &mut self.current_stage_params,
            &mut self.stage_ramp_deltas,
            self.alpha_ramp,
        );
    }

    /// Advances the control bus by one VM audio frame with deterministic chunk-size invariance.
    #[inline]
    pub fn step_frame(&mut self, router_envelopes: Option<[f32; 4]>, frame_abs_mean: f32) {
        if !self.should_tick() {
            return;
        }
        if self.ctrl_sub_counter == 0 {
            let energy = self.window_abs_acc * self.inv_ctrl_stride;
            self.window_abs_acc = 0.0;
            self.tick_control_boundary(router_envelopes, energy);
        }
        self.window_abs_acc += frame_abs_mean;
        advance_micro_ramp_frame(
            &self.stage_targets,
            &self.prev_stage_targets,
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
            self.ctrl_sub_counter,
            self.ctrl_stride,
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= self.ctrl_stride {
            self.ctrl_sub_counter = 0;
        }
    }
}

/// Concrete [`ControlBus`] bound to the host-optimal [`DefaultMathBackend`].
pub type DefaultControlBus<'a> = ControlBus<'a, DefaultMathBackend>;

/// Non-inlined real-time frame-step entrypoint for disassembly audit (`ref[sdk.rust.realtime-disassembly-audit]`).
#[inline(never)]
pub fn step_control_frame_default(
    bus: &mut DefaultControlBus<'_>,
    router_envelopes: Option<[f32; 4]>,
    frame_abs_mean: f32,
) {
    bus.step_frame(router_envelopes, frame_abs_mean);
}

/// Non-inlined `3 kHz` boundary entrypoint for disassembly audit (`ref[sdk.rust.realtime-disassembly-audit]`).
#[inline(never)]
pub fn step_control_tick_3khz_default(
    bus: &mut DefaultControlBus<'_>,
    router_envelopes: Option<[f32; 4]>,
    audio_abs_energy: f32,
) {
    bus.tick_control_boundary(router_envelopes, audio_abs_energy);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::{DnamBinBuildInput, DnamBinView, encode_dnam_bin_into};

    #[test]
    fn test_control_bus_zero_control_idle_identity_no_alloc() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            r_ac_default: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0u8; 128 + 4_608];
        encode_dnam_bin_into(&input, &mut buf).unwrap();
        let view = DnamBinView::parse(&buf).unwrap();

        let mut descs = [ControlClampDescriptor::empty(); 1];
        let mut raw = [0.0; 1];
        let mut clamped = [0.0; 1];
        let mut norm = [0.0; 1];
        let mut sw = [0u8; 1];
        let mut slots = [Cp1ContributorSlot::zero(); 8];
        let backing = ControlBusBacking::from_slices(
            &mut descs,
            &mut raw,
            &mut clamped,
            &mut norm,
            &mut sw,
            &mut slots,
            &[],
        )
        .unwrap();

        let mut bus: ControlBus<'_> =
            ControlBus::from_dnam_bin(&view, Tier::Lite, 48_000.0, backing).unwrap();
        assert!(!bus.should_tick());
        for _ in 0..32 {
            bus.step_frame(None, 0.25);
        }
        assert_eq!(bus.current_stage_params, [0.0; 32]);

        struct NullSink;
        impl core::fmt::Write for NullSink {
            fn write_str(&mut self, _s: &str) -> core::fmt::Result {
                Ok(())
            }
        }
        use core::fmt::Write;
        write!(NullSink, "{bus:?}").unwrap();

        assert_eq!(
            bus.set_control_physical(0, 1.0).unwrap_err(),
            VmError::InvalidControlDescriptor
        );
        assert_eq!(
            bus.set_controls_batch(&[(0, 1.0)]).unwrap_err(),
            VmError::InvalidControlDescriptor
        );
        bus.retime_sample_rate(0.0);
        assert_eq!(bus.host_sample_rate_hz, 48_000.0);
        bus.retime_sample_rate(96_000.0);
        assert_eq!(bus.ctrl_stride, 32);

        bus.live_analog_drift_enabled = true;
        bus.offline_bounce = false;
        bus.sigma_drift = 0.05;
        step_control_tick_3khz_default(&mut bus, Some([0.1, 0.2, 0.3, 0.4]), 0.1);
        step_control_frame_default(&mut bus, None, 0.05);
        assert!(bus.should_tick());
        bus.reset();

        assert_eq!(bus.apply_tempo_bpm(120.0), 0);
        assert!(
            ControlBusBacking::from_slices(
                &mut descs,
                &mut raw,
                &mut clamped,
                &mut norm,
                &mut sw,
                &mut [],
                &[],
            )
            .is_ok()
        );
    }
}
