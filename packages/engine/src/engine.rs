//! Primary generic real-time DNAM inference engine (`DnamEngine<'a, B>`).

pub mod oversampler;
pub mod presets;
pub mod process;
pub mod runtime;
pub mod tempo;

use crate::container::{
    ARENA_LITE_END_CL, ARENA_OFFSET_LITE_END, DnamContainerView, DnamPreamble, ZohRescaledParams,
};
use crate::control::ControlBus;
use crate::math::DefaultMathBackend;
use crate::memory::{AlignedBuffer, AlignedFp16Block, ArenaStorage, CacheLine64};
use crate::provenance::{CryptoProvider, DefaultCryptoProvider, VerificationStatus};
use crate::schema::{
    ControlCalibrationRole, DnamError, DnamJsonHeader, DnamProtectionProfile, TempoSubdivision,
    cold_pow_f64,
};
use crate::stages::{StageBackend, StagePipeline};
use crate::types::{DnamTelemetrySnapshot, Tier};

pub use oversampler::{OFFLINE_BOUNCE_PDC_SAMPLES, Oversampler4xState};
pub(crate) use presets::{apply_calibrated_control_role, resolve_preset_batch_by_id};
pub use process::{
    DnamAudioIo, FeatureBusBackend, tile_collapse_svd_slice_vec4, tile_dot64_stereo_vec4,
    tile_dot64_vec4,
};
pub use tempo::TempoSyncState;

/// Composite SIMD backend trait required by [`DnamEngine`] across all operators, stages, and the 256-D Feature Bus.
pub trait EngineBackend: StageBackend + FeatureBusBackend {}

impl<B: StageBackend + FeatureBusBackend> EngineBackend for B {}

/// Zero-allocation real-time DNAM inference engine parameterized over an
/// [`EngineBackend`] (`DefaultMathBackend` by default).
#[derive(Debug)]
pub struct DnamEngine<'a, B: EngineBackend = DefaultMathBackend> {
    pub(crate) preamble: DnamPreamble,
    pub(crate) header: Option<DnamJsonHeader>,
    pub(crate) verification_status: VerificationStatus,
    pub(crate) protection_profile: DnamProtectionProfile,
    pub(crate) arena_storage: ArenaStorage<'a>,
    pub(crate) tier: Tier,
    pub(crate) max_available_tier: Tier,
    pub(crate) max_supported_tier: Tier,
    pub(crate) offline_bounce: bool,
    pub(crate) maximize_tier_on_bounce: bool,
    pub(crate) oversampler: Oversampler4xState,
    pub(crate) oversampler_aux: Oversampler4xState,
    pub(crate) tempo_state: TempoSyncState,
    pub(crate) zoh: ZohRescaledParams,
    pub(crate) ctrl_sub_counter: usize,
    pub(crate) ctrl_energy_accum: f32,
    pub(crate) bypassed: bool,
    pub(crate) trails_enabled: bool,
    pub(crate) freeze_mutes_dry: bool,
    pub(crate) single_input: bool,
    pub(crate) stereo_width: f32,
    pub(crate) current_stage_params: [f32; 16],
    pub(crate) stage_ramp_deltas: [f32; 16],
    pub(crate) collapsed_head_wl: [[B::Vec4; 16]; 4],
    pub(crate) collapsed_head_wr: [[B::Vec4; 16]; 4],
    pub(crate) cached_role_b_slices: alloc::boxed::Box<[[B::Vec4; 32]; 4]>,
    pub(crate) feature_bus_scratch: alloc::boxed::Box<[f32; 256]>,
    pub(crate) collapsed_b_head: [f32; 2],
    pub(crate) cached_z_head: [f32; 4],
    pub(crate) cached_z_head_ultra: [f32; 4],
    pub(crate) cached_head_tiers: usize,
    pub(crate) cached_tier_gain_trim: [f32; 4],
    pub saturation_flags: u32,
    pub control_bus: ControlBus<B>,
    pub stage_pipeline: StagePipeline,
    pub mix_dry_gain: f32,
    pub mix_wet_gain: f32,
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Creates an unconfigured zero-allocation [`DnamEngine`] shell.
    pub(crate) fn new_empty(max_supported_tier: Tier) -> Self {
        let zoh = ZohRescaledParams {
            sample_rate_host_hz: 48_000,
            rho_sr: 1.0,
            control_decimation_host: 16,
            rho_ctrl: 1.0,
            alpha_ramp_host: 0.0625,
            r_ac_host: 0.9993,
            r_dc_host: 0.995,
            tau_ref_samples_host: 64.0,
            switch_xfade_samples_host: 128,
        };
        let preamble = DnamPreamble {
            magic: crate::container::DNAM_MAGIC,
            version_major: crate::container::CONTAINER_VERSION_MAJOR,
            version_minor: crate::container::CONTAINER_VERSION_MINOR,
            flags: crate::container::FLAG_HAS_LITE | crate::container::FLAG_STEREO_OUT,
            num_controls: 0,
            sample_rate_train_hz: 48_000,
            json_header_byte_len: 0,
            tensor_arena_offset: 4096,
            tensor_arena_byte_len: ARENA_OFFSET_LITE_END,
            arena_lite_end_cl: ARENA_LITE_END_CL,
            arena_full_end_cl: ARENA_LITE_END_CL,
            arena_pro_end_cl: ARENA_LITE_END_CL,
            arena_ultra_end_cl: ARENA_LITE_END_CL,
            arena_crc32c_lite: 0,
            arena_crc32c_full: 0,
            arena_crc32c_pro: 0,
            arena_crc32c_ultra: 0,
            _reserved_stems: 0,
            _reserved: 0,
            tail_time_ms: 0.0,
            header_crc32c: 0,
        };
        Self {
            preamble,
            header: None,
            verification_status: VerificationStatus::Unsigned,
            protection_profile: DnamProtectionProfile::Open,
            arena_storage: ArenaStorage::from_slice_or_copy(&[]),
            tier: Tier::Lite,
            max_available_tier: Tier::Lite,
            max_supported_tier,
            offline_bounce: false,
            maximize_tier_on_bounce: true,
            oversampler: Oversampler4xState::new(),
            oversampler_aux: Oversampler4xState::new(),
            tempo_state: TempoSyncState::new(),
            zoh,
            ctrl_sub_counter: 0,
            ctrl_energy_accum: 0.0,
            bypassed: false,
            trails_enabled: false,
            freeze_mutes_dry: false,
            single_input: false,
            stereo_width: 1.0,
            current_stage_params: [0.0; 16],
            stage_ramp_deltas: [0.0; 16],
            collapsed_head_wl: [[<B::Vec4 as crate::math::SimdVec4>::zero(); 16]; 4],
            collapsed_head_wr: [[<B::Vec4 as crate::math::SimdVec4>::zero(); 16]; 4],
            cached_role_b_slices: alloc::boxed::Box::new(
                [[<B::Vec4 as crate::math::SimdVec4>::splat(1.0); 32]; 4],
            ),
            feature_bus_scratch: alloc::boxed::Box::new([0.0; 256]),
            collapsed_b_head: [0.0; 2],
            cached_z_head: [f32::NAN; 4],
            cached_z_head_ultra: [f32::NAN; 4],
            cached_head_tiers: 0,
            cached_tier_gain_trim: [1.0; 4],
            saturation_flags: 0,
            control_bus: ControlBus::new(),
            stage_pipeline: StagePipeline::new(),
            mix_dry_gain: 0.0,
            mix_wet_gain: 1.0,
        }
    }

    /// Loads a `.dnam` container from `bytes` using the default [`CryptoProvider`].
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Loads a `.dnam` container capped at `max_supported_tier`.
    pub fn from_bytes_with_max_tier(
        bytes: &'a [u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes_with_max_tier(bytes, &crypto, max_supported_tier)?;
        let mut engine = Self::new_empty(max_supported_tier);
        engine.init_from_view(view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
    }

    /// Creates an owned `'static` [`DnamEngine`].
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Creates an owned `'static` [`DnamEngine`] capped at `max_supported_tier`.
    pub fn from_bytes_owned_with_max_tier(
        bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes_with_max_tier(bytes, &crypto, max_supported_tier)?;
        let arena_bytes = view.arena_storage.as_bytes();
        let mut owned_arena = AlignedBuffer::<CacheLine64>::new_zeroed(arena_bytes.len() / 64);
        owned_arena.as_bytes_mut().copy_from_slice(arena_bytes);
        let owned_view = DnamContainerView {
            preamble: view.preamble,
            header: view.header,
            verification_status: view.verification_status,
            protection_profile: view.protection_profile,
            control_records: view.control_records,
            clamp_descriptors: view.clamp_descriptors,
            cp1_contributors: view.cp1_contributors,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
        };
        let mut engine = DnamEngine::<'static, B>::new_empty(max_supported_tier);
        engine.init_from_view(owned_view, tier, max_supported_tier, sample_rate_hz)?;
        Ok(engine)
    }

    /// Loads a `.dnam` container with a custom [`CryptoProvider`].
    pub fn from_bytes_owned_with_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        let view = DnamContainerView::from_bytes(bytes, crypto)?;
        let arena_bytes = view.arena_storage.as_bytes();
        let mut owned_arena = AlignedBuffer::<CacheLine64>::new_zeroed(arena_bytes.len() / 64);
        owned_arena.as_bytes_mut().copy_from_slice(arena_bytes);
        let owned_view = DnamContainerView {
            preamble: view.preamble,
            header: view.header,
            verification_status: view.verification_status,
            protection_profile: view.protection_profile,
            control_records: view.control_records,
            clamp_descriptors: view.clamp_descriptors,
            cp1_contributors: view.cp1_contributors,
            arena_storage: ArenaStorage::from_owned_buffer(owned_arena),
        };
        let mut engine = DnamEngine::<'static, B>::new_empty(Tier::Ultra);
        engine.init_from_view(owned_view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
    }

    /// Loads a borrowed `.dnam` container with a custom [`CryptoProvider`].
    pub fn from_bytes_with_crypto<C: CryptoProvider>(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<Self, DnamError> {
        let view = DnamContainerView::from_bytes(bytes, crypto)?;
        let mut engine = Self::new_empty(Tier::Ultra);
        engine.init_from_view(view, tier, Tier::Ultra, sample_rate_hz)?;
        Ok(engine)
    }

    /// Zero-Copy Scratch Loader: borrows caller-supplied `scratch` and arena.
    pub fn from_bytes_with_scratch(
        bytes: &'a [u8],
        _scratch: &'a mut [AlignedFp16Block],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        Self::from_bytes(bytes, tier, sample_rate_hz)
    }

    /// Pre-allocates a [`DnamEngine`] with maximum `Arena 0..3` (`64,000 B`).
    pub fn with_max_capacity() -> Self {
        Self::with_max_tier(Tier::Ultra)
    }

    /// Pre-allocates a [`DnamEngine`] capped at `max_supported_tier`.
    pub fn with_max_tier(max_supported_tier: Tier) -> Self {
        let mut engine = Self::new_empty(max_supported_tier);
        engine.arena_storage = ArenaStorage::with_max_tier(max_supported_tier);
        engine.control_bus.ensure_control_capacity(256);
        engine.stage_pipeline.reserve_extended_capacity();
        engine
    }

    #[inline]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        self.max_available_tier
    }

    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        self.max_supported_tier
    }

    #[inline]
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        self.max_supported_tier = max_tier;
        if self.tier > max_tier {
            self.tier = max_tier;
        }
    }

    #[inline]
    pub fn effective_tier(&self) -> Tier {
        if self.offline_bounce && self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
        }
    }

    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        self.offline_bounce
    }

    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        self.control_bus.offline_bounce = enabled;
        if self.offline_bounce != enabled {
            self.offline_bounce = enabled;
            let _ = self.reconfigure_for_bounce_toggle();
        }
    }

    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        self.maximize_tier_on_bounce
    }

    #[inline]
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        self.maximize_tier_on_bounce = enabled;
    }

    #[inline]
    pub const fn latency_samples(&self) -> usize {
        if self.offline_bounce {
            OFFLINE_BOUNCE_PDC_SAMPLES
        } else {
            0
        }
    }

    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        let max_tier = self.max_available_tier.min(self.max_supported_tier);
        self.tier = tier.min(max_tier);
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate(sample_rate_hz)
    }

    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.zoh.sample_rate_host_hz
    }

    #[inline]
    pub const fn zoh(&self) -> &ZohRescaledParams {
        &self.zoh
    }

    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        &self.verification_status
    }

    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        self.protection_profile
    }

    #[inline]
    pub const fn preamble(&self) -> &DnamPreamble {
        &self.preamble
    }

    #[inline]
    pub const fn header(&self) -> Option<&DnamJsonHeader> {
        self.header.as_ref()
    }

    /// Returns the active host tempo in BPM.
    #[inline]
    pub const fn current_bpm(&self) -> f32 {
        self.tempo_state.current_bpm
    }

    /// Returns the number of user controls $P \in [0, 255]$.
    #[inline]
    pub const fn num_controls(&self) -> usize {
        self.control_bus.num_controls
    }

    /// Returns the slice of raw user-requested physical control values (`len == num_controls`).
    #[inline]
    pub fn raw_control_values(&self) -> &[f32] {
        let n = self
            .control_bus
            .num_controls
            .min(self.control_bus.raw_values.len());
        &self.control_bus.raw_values[..n]
    }

    /// Returns the slice of clamped physical control values (`len == num_controls`).
    #[inline]
    pub fn clamped_control_values(&self) -> &[f32] {
        let n = self
            .control_bus
            .num_controls
            .min(self.control_bus.clamped_values.len());
        &self.control_bus.clamped_values[..n]
    }

    /// Returns the compiled 64-byte [`crate::control::ControlClampDescriptor`] slice (`len == num_controls`).
    #[inline]
    pub fn control_descriptors(&self) -> &[crate::control::ControlClampDescriptor] {
        let n = self
            .control_bus
            .num_controls
            .min(self.control_bus.descriptors.len());
        &self.control_bus.descriptors[..n]
    }

    /// Snaps all controls immediately back to their schema defaults and refreshes static head/tank state.
    pub fn reset_controls_to_defaults(&mut self) {
        self.control_bus.reset_to_defaults();
        let n = self.control_bus.num_controls;
        for idx in 0..n {
            if let Some(&val) = self.control_bus.clamped_values.get(idx) {
                self.refresh_calibrated_control_effects(idx, val);
            }
        }
        self.sync_immediate_control_state();
    }

    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus.set_control_value(ctrl_idx, physical_value);
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
    }

    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        self.control_bus
            .set_control_immediate(ctrl_idx, physical_value);
        self.refresh_calibrated_control_effects(ctrl_idx, physical_value);
        self.sync_immediate_control_state();
    }

    fn sync_immediate_control_state(&mut self) {
        self.sync_immediate_control_state_inner();
    }

    fn refresh_calibrated_control_effects(&mut self, ctrl_idx: usize, physical_value: f32) {
        let supports_freeze = self.supports_freeze();
        apply_calibrated_control_role(
            self.header.as_ref(),
            ctrl_idx,
            physical_value,
            &mut self.stereo_width,
            &mut self.stage_pipeline.stage4.freeze,
            supports_freeze,
        );
    }

    /// Returns whether this model supports captured hardware freeze (`supports_freeze == true` or
    /// a `FreezeLatch`-calibrated control in `controls_schema`, `ref[dsp.stage4.captured-freeze]`).
    #[inline]
    pub fn supports_freeze(&self) -> bool {
        match self.header {
            Some(ref hdr) => {
                hdr.io_config.capabilities.supports_freeze
                    || hdr.controls_schema.iter().any(|c| {
                        c.calibration
                            .as_ref()
                            .is_some_and(|cal| cal.role == ControlCalibrationRole::FreezeLatch)
                    })
            }
            None => true,
        }
    }

    #[inline]
    pub fn set_freeze(&mut self, freeze: bool) {
        self.stage_pipeline.stage4.freeze = freeze && self.supports_freeze();
    }

    #[inline]
    pub fn set_freeze_amount(&mut self, alpha: f32) {
        self.stage_pipeline.stage4.freeze = (alpha > 0.5) && self.supports_freeze();
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.stage_pipeline.stage4.freeze
    }

    #[inline]
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        self.bypassed = bypassed;
        self.trails_enabled = trails;
    }

    #[inline]
    pub const fn bypass_state(&self) -> (bool, bool) {
        (self.bypassed, self.trails_enabled)
    }

    #[inline]
    pub fn set_stereo_width(&mut self, width: f32) {
        self.stereo_width = width.clamp(0.0, 2.0);
    }

    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        self.stereo_width
    }

    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        let b = blend_wet.clamp(0.0, 1.0);
        let (pol_sign, cal_trim_db) = if let Some(ref hdr) = self.header {
            let sign = if hdr.io_config.capabilities.wet_polarity_inverted {
                -1.0_f32
            } else {
                1.0_f32
            };
            let trim = if apply_unity_leveling && hdr.io_config.leveling.auto_unity_enabled {
                hdr.io_config.leveling.calibrated_trim_db
            } else {
                0.0_f32
            };
            (sign, trim)
        } else {
            (1.0_f32, 0.0_f32)
        };

        let out_scale = cold_pow_f64(10.0, (output_gain_db as f64) * 0.05) as f32;
        let wet_trim_scale =
            cold_pow_f64(10.0, ((output_gain_db + cal_trim_db) as f64) * 0.05) as f32;

        self.mix_dry_gain = (1.0 - b) * out_scale;
        self.mix_wet_gain = pol_sign * b * wet_trim_scale;
    }

    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        self.tempo_state.set_bpm(bpm);
        let Some(ref hdr) = self.header else {
            return 0;
        };
        let mut updates = [(0usize, 0.0f32); 512];
        let mut tempo_clamp_mask = 0u16;
        let count = TempoSyncState::resolve_sync_updates_with_clamp_mask_into(
            hdr,
            self.tempo_state.current_bpm,
            subdivision_override,
            fold_octave,
            &mut updates,
            &mut tempo_clamp_mask,
        );
        for &(idx, val) in &updates[..count] {
            self.set_control_immediate(idx, val);
        }
        self.control_bus.active_clamp_mask |= tempo_clamp_mask;
        count
    }

    /// Registers a tap-tempo event at the current audio frame counter, updates the host BPM,
    /// and synchronizes any `DelayTimeMs` or `LfoRateHz` controls.
    pub fn register_tap_tempo(
        &mut self,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> (f32, usize) {
        let bpm = self.tempo_state.register_tap();
        let count = self.sync_tempo(bpm, subdivision_override, fold_octave);
        (bpm, count)
    }

    /// Sets the active host tempo in BPM without modifying control values.
    #[inline]
    pub fn set_bpm(&mut self, bpm: f32) {
        self.tempo_state.set_bpm(bpm);
    }

    /// Accumulates additional clamping bits into `active_clamp_mask`.
    #[inline]
    pub(crate) fn or_active_clamp_mask(&mut self, mask: u16) {
        self.control_bus.active_clamp_mask |= mask;
    }

    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        DnamTelemetrySnapshot {
            active_tier: self.effective_tier() as u8,
            stereo_out: u8::from(self.preamble.is_stereo_out()),
            bypassed: u8::from(self.bypassed),
            is_frozen: u8::from(self.is_frozen()),
            active_clamp_mask: self.control_bus.active_clamp_mask,
            control_decimation_host: self.zoh.control_decimation_host as u16,
            max_op_n_drive: 1.0,
            max_pole_radius: 0.999,
            max_effective_loop_gain: self.stage_pipeline.stage4.g_loop,
            tail_time_ms: self.preamble.tail_time_ms,
            z_head: self.control_bus.z_head,
            env_states: self.control_bus.env_states,
            lfo_harmonics: self.control_bus.lfo_harmonics,
            clamped_controls: self.control_bus.telemetry_clamped_controls(),
            overflow_or_saturation_flags: self.saturation_flags,
        }
    }

    /// Enables or disables optional live Ornstein–Uhlenbeck analog drift (`false` by default).
    #[inline]
    pub fn set_live_analog_drift(&mut self, enabled: bool) {
        self.control_bus.live_analog_drift_enabled = enabled;
    }

    /// Returns whether live Ornstein–Uhlenbeck analog drift is enabled.
    #[inline]
    pub fn live_analog_drift_enabled(&self) -> bool {
        self.control_bus.live_analog_drift_enabled
    }
}
