//! Primary generic real-time DNAM inference engine (`DnamEngine<'a, B>`).
//!
//! Wraps two isolated [`dnam_vm::SonaVm`] instances (`vm` for live audio and `bounce_vm` for
//! zero-mutation offline bounce rendering), a Layer 1 [`HostShellPatchbay`], and a storefront
//! [`PreviewGuardState`] behind `#![forbid(unsafe_code)]`.

pub mod patchbay;
pub mod presets;
pub mod process;
pub mod runtime;
pub mod shell;
pub mod tempo;

use core::marker::PhantomData;

use dnam_vm::control::ControlClampDescriptor;
use dnam_vm::{SonaVm, ZohRateContext};

use crate::container::DnamPreamble;
use crate::memory::AlignedFp16Block;
use crate::provenance::{
    CryptoProvider, DefaultCryptoProvider, PreviewGuardState, VerificationStatus,
};
use crate::schema::{
    BlendMode, DnamError, DnamJsonHeader, DnamProtectionProfile, TempoSubdivision,
};
use crate::types::Tier;

pub use patchbay::{HostShellPatchbay, IoRoutingMode, ParallelAlignmentMode, WdwDryTapMode};
pub(crate) use presets::{
    MAX_RELOAD_HOST_ROLE_CONTROLS, ReloadCalibratedControl, apply_calibrated_control_role,
    apply_calibrated_control_role_from_reload, resolve_calibrated_host_blend,
    resolve_calibrated_host_blend_from_reload, resolve_preset_batch_by_id,
};
pub use process::DnamAudioIo;
pub use tempo::TempoSyncState;

/// Fixed offline `4x` bounce linear-phase FIR latency in host samples (`12`, `dnam-vm`).
pub const OFFLINE_BOUNCE_PDC_SAMPLES: usize = dnam_vm::OFFLINE_BOUNCE_PDC_SAMPLES;

/// Composite SIMD backend trait required by [`DnamEngine`] across all VM and DSP stages.
pub trait EngineBackend: dnam_vm::GraphBackend {}

impl<B: dnam_vm::GraphBackend> EngineBackend for B {}

/// Compact, zero-allocation header flags retained when [`DnamEngine::reload_model_in_place`]
/// discards the active view of [`DnamJsonHeader`] without dropping heap allocations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReloadHeaderFlags {
    pub(crate) single_input: bool,
    pub(crate) stereo_out: bool,
    pub(crate) blend_mode: BlendMode,
    pub(crate) default_blend: f32,
    pub(crate) wet_polarity_inverted: bool,
    pub(crate) auto_unity_enabled: bool,
    pub(crate) calibrated_trim_db: f32,
    pub(crate) dc_blocker_pole: f32,
    pub(crate) tzf_reference_tap_samples: f32,
    pub(crate) switch_xfade_samples: u16,
    pub(crate) host_role_slots: [ReloadCalibratedControl; MAX_RELOAD_HOST_ROLE_CONTROLS],
    pub(crate) host_role_count: u8,
}

impl Default for ReloadHeaderFlags {
    #[inline]
    fn default() -> Self {
        Self {
            single_input: false,
            stereo_out: true,
            blend_mode: BlendMode::None,
            default_blend: 1.0,
            wet_polarity_inverted: false,
            auto_unity_enabled: true,
            calibrated_trim_db: 0.0,
            dc_blocker_pole: 0.995,
            tzf_reference_tap_samples: 64.0,
            switch_xfade_samples: 128,
            host_role_slots: [ReloadCalibratedControl::empty(); MAX_RELOAD_HOST_ROLE_CONTROLS],
            host_role_count: 0,
        }
    }
}

/// Zero-allocation Layer 1 `.dnam` shell engine wrapping [`dnam_vm::SonaVm`] over an
/// [`EngineBackend`] (`dnam_vm::DefaultMathBackend` by default).
#[derive(Debug)]
pub struct DnamEngine<'a, B: EngineBackend = dnam_vm::DefaultMathBackend> {
    pub(crate) _marker: PhantomData<&'a ()>,
    pub(crate) preamble: DnamPreamble,
    pub(crate) header: Option<DnamJsonHeader>,
    pub(crate) header_active: bool,
    pub(crate) reloaded_header_flags: Option<ReloadHeaderFlags>,
    pub(crate) verification_status: VerificationStatus,
    pub(crate) verification_status_active: bool,
    pub(crate) protection_profile: DnamProtectionProfile,
    pub(crate) tier: Tier,
    pub(crate) max_available_tier: Tier,
    pub(crate) max_supported_tier: Tier,
    pub(crate) preallocated_max_tier: Tier,
    pub(crate) offline_bounce: bool,
    pub(crate) maximize_tier_on_bounce: bool,
    pub(crate) tempo_state: TempoSyncState,
    pub(crate) bypassed: bool,
    pub(crate) trails_enabled: bool,
    pub(crate) single_input: bool,
    pub(crate) saturation_flags: u32,
    pub(crate) last_block_max_drive: f32,
    pub(crate) tail_time_ms: f32,
    pub(crate) extra_clamp_mask: u16,
    pub(crate) preview_mode: bool,
    pub(crate) preview_guard: PreviewGuardState,
    pub(crate) vm: alloc::boxed::Box<SonaVm<'static, B>>,
    pub(crate) bounce_vm: alloc::boxed::Box<SonaVm<'static, B>>,
    pub(crate) patchbay: HostShellPatchbay,
    pub(crate) bounce_patchbay: HostShellPatchbay,
    pub mix_dry_gain: f32,
    pub mix_wet_gain: f32,
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Loads a `.dnam` or `.dnam-seal` container from `bytes` using the default [`CryptoProvider`].
    /// Rejects `"container_kind": "preview"` fail-closed (`r[sec.preview.containment]`).
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Loads a `.dnam` or `.dnam-seal` container capped at `max_supported_tier`.
    /// Rejects `"container_kind": "preview"` fail-closed (`r[sec.preview.containment]`).
    pub fn from_bytes_with_max_tier(
        bytes: &'a [u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        runtime::load_engine_with_options(
            bytes,
            tier,
            max_supported_tier,
            sample_rate_hz,
            &crypto,
            false,
            None,
        )
    }

    /// Creates an owned `'static` [`DnamEngine`] from `bytes` (rejects `"container_kind": "preview"`).
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
        runtime::load_engine_with_options(
            bytes,
            tier,
            max_supported_tier,
            sample_rate_hz,
            &crypto,
            false,
            None,
        )
    }

    /// Loads an owned `'static` [`DnamEngine`] with a custom [`CryptoProvider`].
    pub fn from_bytes_owned_with_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        runtime::load_engine_with_options(
            bytes,
            tier,
            Tier::Ultra,
            sample_rate_hz,
            crypto,
            false,
            None,
        )
    }

    /// Loads a `.dnam` or `.dnam-seal` container with a custom [`CryptoProvider`].
    pub fn from_bytes_with_crypto<C: CryptoProvider>(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<Self, DnamError> {
        runtime::load_engine_with_options(
            bytes,
            tier,
            Tier::Ultra,
            sample_rate_hz,
            crypto,
            false,
            None,
        )
    }

    /// Loads a container using caller-supplied scratch metadata for compatibility with existing
    /// zero-copy call sites (`SonaVm` owns its L1D hot execution slab directly).
    pub fn from_bytes_with_scratch(
        bytes: &'a [u8],
        _scratch: &'a mut [AlignedFp16Block],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        Self::from_bytes(bytes, tier, sample_rate_hz)
    }

    /// Loads a storefront `.dnam-prev` (`"container_kind": "preview"`) container explicitly
    /// permitted for sandboxed Web/AudioWorklet preview (`r[sec.preview.containment]`).
    ///
    /// Preview containers enforce a strict [`Tier::Lite`] ceiling and engage Layers 5–6
    /// (`PreviewGuardState` probe phase flutter + sparse `-42 dB` micro-echo watermarking).
    pub fn from_preview_bytes(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let seed = crate::provenance::sha256_digest(bytes);
        Self::from_preview_bytes_with_seed(bytes, tier, sample_rate_hz, &seed)
    }

    /// Loads a storefront `.dnam-prev` container keyed by explicit 32-byte `seed`.
    pub fn from_preview_bytes_with_seed(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        seed: &[u8; 32],
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        runtime::load_engine_with_options(
            bytes,
            tier,
            Tier::Lite,
            sample_rate_hz,
            &crypto,
            true,
            Some(*seed),
        )
    }

    /// Loads an owned `'static` storefront `.dnam-prev` container with custom `CryptoProvider` and `seed`.
    pub fn from_preview_bytes_owned_with_seed<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        seed: &[u8; 32],
        crypto: &C,
    ) -> Result<DnamEngine<'static, B>, DnamError> {
        runtime::load_engine_with_options(
            bytes,
            tier,
            Tier::Lite,
            sample_rate_hz,
            crypto,
            true,
            Some(*seed),
        )
    }

    /// Pre-allocates a [`DnamEngine`] with maximum `Ultra` + control + transport capacity for
    /// zero-allocation `reload_model_in_place`.
    pub fn with_max_capacity() -> Self {
        Self::with_max_tier(Tier::Ultra)
    }

    /// Pre-allocates a [`DnamEngine`] capped at `max_supported_tier` with full live reload
    /// headroom (`255` controls, `64` `CP1` contributors, `32` `"TSYN"`, transport + Schroeder rings).
    pub fn with_max_tier(max_supported_tier: Tier) -> Self {
        let headroom = dnam_vm::graph::VmReloadHeadroom::for_live_engine()
            .with_transport_headroom(16_384, 16_384, 4_096);
        Self::new_empty_with_headroom(max_supported_tier, max_supported_tier, headroom)
    }

    #[inline]
    pub(crate) fn sync_hot_core_tier(&mut self) {
        let eff = self.effective_tier();
        if self.vm.slab().lite.tier != eff {
            let _ = self.vm.switch_tier(eff);
            let zoh = self.vm.slab().cold().zoh;
            self.patchbay
                .reconfigure_sample_rate(zoh.f_s_vm, zoh.is_bounce_4x);
        }
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
        let ceiling = if self.preview_mode {
            Tier::Lite
        } else {
            max_tier
        };
        self.max_supported_tier = ceiling;
        if self.tier > ceiling {
            self.tier = ceiling;
        }
        self.sync_hot_core_tier();
    }

    #[inline]
    pub fn effective_tier(&self) -> Tier {
        if self.preview_mode {
            return Tier::Lite;
        }
        if self.offline_bounce && self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
                .min(self.max_available_tier)
                .min(self.max_supported_tier)
        }
    }

    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        self.offline_bounce
    }

    #[inline]
    pub fn set_offline_bounce(&mut self, enabled: bool) {
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
        self.sync_hot_core_tier();
    }

    #[inline]
    pub const fn latency_samples(&self) -> usize {
        self.patchbay.pdc_host_samples()
    }

    #[inline]
    pub fn set_tier(&mut self, tier: Tier) {
        let max_tier = self.max_available_tier.min(self.max_supported_tier);
        self.tier = if self.preview_mode {
            Tier::Lite
        } else {
            tier.min(max_tier)
        };
        self.sync_hot_core_tier();
    }

    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate(sample_rate_hz)
    }

    #[inline]
    pub fn sample_rate_hz(&self) -> u32 {
        self.tempo_state.sample_rate_hz
    }

    #[inline]
    pub fn zoh(&self) -> &ZohRateContext {
        &self.vm.slab().cold().zoh
    }

    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        static UNSIGNED: VerificationStatus = VerificationStatus::Unsigned;
        if self.verification_status_active {
            &self.verification_status
        } else {
            &UNSIGNED
        }
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
        if self.header_active {
            self.header.as_ref()
        } else {
            None
        }
    }

    /// Returns the active host tempo in BPM.
    #[inline]
    pub const fn current_bpm(&self) -> f32 {
        self.tempo_state.current_bpm
    }

    /// Returns the number of user controls $P \in [0, 255]$.
    #[inline]
    pub fn num_controls(&self) -> usize {
        self.vm
            .slab()
            .cold()
            .ctrl_bus
            .as_ref()
            .map_or(0, |b| b.num_controls)
    }

    /// Returns the slice of raw user-requested physical control values (`len == num_controls`).
    #[inline]
    pub fn raw_control_values(&self) -> &[f32] {
        self.vm.slab().cold().ctrl_bus.as_ref().map_or(&[], |b| {
            let n = b.num_controls.min(b.backing.raw_values.len());
            &b.backing.raw_values[..n]
        })
    }

    /// Returns the slice of clamped physical control values (`len == num_controls`).
    #[inline]
    pub fn clamped_control_values(&self) -> &[f32] {
        self.vm.slab().cold().ctrl_bus.as_ref().map_or(&[], |b| {
            let n = b.num_controls.min(b.backing.clamped_values.len());
            &b.backing.clamped_values[..n]
        })
    }

    /// Returns the compiled 64-byte [`ControlClampDescriptor`] slice (`len == num_controls`).
    #[inline]
    pub fn control_descriptors(&self) -> &[ControlClampDescriptor] {
        self.vm.slab().cold().ctrl_bus.as_ref().map_or(&[], |b| {
            let n = b.num_controls.min(b.backing.descriptors.len());
            &b.backing.descriptors[..n]
        })
    }

    /// Resets all controls back to their schema defaults and latches `SonaVm` immediately.
    pub fn reset_controls_to_defaults(&mut self) {
        if let Some(bus) = self.vm.slab_mut().cold_mut().ctrl_bus.as_mut() {
            let n = bus.num_controls.min(bus.backing.descriptors.len());
            for i in 0..n {
                bus.backing.raw_values[i] = bus.backing.descriptors[i].default_val;
            }
            bus.is_dirty = true;
            bus.clamp_and_eval_static_if_dirty();
        }
        self.extra_clamp_mask = 0;
        let n = self.num_controls();
        for idx in 0..n {
            if let Some(&val) = self.clamped_control_values().get(idx) {
                self.refresh_calibrated_control_effects(idx, val);
            }
        }
        self.sync_immediate_control_state();
    }

    /// Schedules a smoothed physical control change via `SonaVm`.
    #[inline]
    pub fn set_control_value(&mut self, ctrl_idx: usize, physical_value: f32) {
        let _ = self.vm.set_control_physical(ctrl_idx, physical_value);
        if let Some(bus) = self.vm.slab_mut().cold_mut().ctrl_bus.as_mut() {
            bus.clamp_and_eval_static_if_dirty();
        }
        self.refresh_calibrated_control_and_dependents(ctrl_idx, physical_value);
    }

    /// Sets a physical control value and latches `SonaVm` and `HostShellPatchbay` immediately.
    #[inline]
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, physical_value: f32) {
        let _ = self.vm.set_control_physical(ctrl_idx, physical_value);
        self.sync_immediate_control_state();
        self.refresh_calibrated_control_and_dependents(ctrl_idx, physical_value);
    }

    #[inline]
    fn refresh_calibrated_control_and_dependents(&mut self, ctrl_idx: usize, fallback_val: f32) {
        let is_switch = self
            .control_descriptors()
            .get(ctrl_idx)
            .is_some_and(|d| d.ctrl_kind == 1);
        if is_switch {
            let n = self.num_controls();
            for idx in 0..n {
                if idx != ctrl_idx
                    && let Some(&val) = self.clamped_control_values().get(idx)
                {
                    self.refresh_calibrated_control_effects(idx, val);
                }
            }
        }
        let clamped_val = self
            .clamped_control_values()
            .get(ctrl_idx)
            .copied()
            .unwrap_or(fallback_val);
        self.refresh_calibrated_control_effects(ctrl_idx, clamped_val);
    }

    #[inline]
    pub(crate) fn sync_immediate_control_state(&mut self) {
        dnam_vm::graph::slab::ctrl_latch::latch_immediate_at_init_or_reset(self.vm.slab_mut());
    }

    #[inline(always)]
    pub(crate) fn json_wet_polarity_inverted(&self) -> bool {
        if let Some(hdr) = self.header() {
            hdr.io_config.capabilities.wet_polarity_inverted
        } else if let Some(flags) = self.reloaded_header_flags {
            flags.wet_polarity_inverted
        } else {
            false
        }
    }

    fn refresh_calibrated_control_effects(&mut self, ctrl_idx: usize, physical_value: f32) {
        let mut width = self.patchbay.ms_width;
        let next_blend = if self.header_active {
            let active_header = self.header.as_ref();
            apply_calibrated_control_role(active_header, ctrl_idx, physical_value, &mut width);
            if let Some(hdr) = active_header
                && hdr.io_config.capabilities.blend_mode == BlendMode::HostParallel
            {
                resolve_calibrated_host_blend(Some(hdr), ctrl_idx, physical_value)
            } else {
                None
            }
        } else if let Some(ref flags) = self.reloaded_header_flags {
            apply_calibrated_control_role_from_reload(flags, ctrl_idx, physical_value, &mut width);
            resolve_calibrated_host_blend_from_reload(flags, ctrl_idx, physical_value)
        } else {
            None
        };
        self.set_stereo_width(width);
        if let Some(blend) = next_blend {
            self.configure_host_mix_and_leveling(blend, 0.0, true);
        }
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

    /// Sets post-VM M/S stereo width clamped to `[0.0, 2.0]`.
    #[inline]
    pub fn set_stereo_width(&mut self, width: f32) {
        self.patchbay.set_ms_width(width);
        self.bounce_patchbay.set_ms_width(width);
    }

    /// Alias matching [`HostShellPatchbay::set_ms_width`].
    #[inline]
    pub fn set_ms_width(&mut self, width: f32) {
        self.set_stereo_width(width);
    }

    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        self.patchbay.ms_width
    }

    /// Configures host shell parallel wet/dry blend, output gain in dB, and optional auto-unity leveling.
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        let b = blend_wet.clamp(0.0, 1.0);
        let cal_trim_db = if let Some(hdr) = self.header() {
            if apply_unity_leveling && hdr.io_config.leveling.auto_unity_enabled {
                hdr.io_config.leveling.calibrated_trim_db
            } else {
                0.0_f32
            }
        } else if let Some(flags) = self.reloaded_header_flags {
            if apply_unity_leveling && flags.auto_unity_enabled {
                flags.calibrated_trim_db
            } else {
                0.0_f32
            }
        } else {
            0.0_f32
        };

        let out_scale = dnam_vm::math::powf_f64(10.0, (output_gain_db as f64) * 0.05) as f32;
        let wet_trim_scale =
            dnam_vm::math::powf_f64(10.0, ((output_gain_db + cal_trim_db) as f64) * 0.05) as f32;

        self.mix_dry_gain = (1.0 - b) * out_scale;
        self.mix_wet_gain = b * wet_trim_scale;
    }

    /// Synchronizes all tempo-synced controls (`"TSYN"` table or header subdivision rules).
    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        self.tempo_state.set_bpm(bpm);
        let Some(hdr) = self.header() else {
            return if subdivision_override.is_none() {
                let updated = self.vm.apply_tempo_bpm(self.tempo_state.current_bpm);
                if updated > 0 {
                    self.sync_immediate_control_state();
                }
                updated
            } else {
                0
            };
        };
        let mut updates = [(0usize, 0.0f32); 256];
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
        self.extra_clamp_mask |= tempo_clamp_mask;
        count
    }

    /// Registers a tap-tempo event at the current audio frame counter, updates the host BPM,
    /// and synchronizes any tempo-bound controls.
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

    #[inline]
    pub(crate) fn or_active_clamp_mask(&mut self, mask: u16) {
        self.extra_clamp_mask |= mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::wire::StructuredConditionalBound;
    use crate::schema::{
        ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
    };
    use crate::test_support::build_test_dnam_container;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use dnam_vm::math::ScalarBackend;

    #[test]
    fn test_switch_tightened_bounds_immediately_clamp_calibrated_stereo_width_and_mix() {
        let mut hdr = DnamJsonHeader::default_reference();
        hdr.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        hdr.io_config.capabilities.default_blend = 0.5;
        let tight = vec![StructuredConditionalBound {
            when: vec![(String::from("gate"), String::from("Tight"))],
            min: Some(0.25),
            max: Some(0.75),
        }];
        hdr.controls_schema = vec![
            ControlSchemaDef {
                id: String::from("gate"),
                name: String::from("Gate"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Open"),
                conditional_bounds: Vec::new(),
                represented_switch_options: None,
                declared_options: Some(vec![String::from("Open"), String::from("Tight")]),
                active_options: Some(vec![String::from("Open"), String::from("Tight")]),
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("width"),
                name: String::from("Width"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 2.0]),
                safe_clamped_range: Some([0.0, 2.0]),
                default: serde_json::json!(1.8),
                conditional_bounds: tight.clone(),
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoWidth,
                    lobatto_physical_anchors: Some([0.0, 1.0, 2.0]),
                    conditional_physical_anchors: Vec::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("mix"),
                name: String::from("Mix"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 1.0]),
                safe_clamped_range: Some([0.0, 1.0]),
                default: serde_json::json!(0.95),
                conditional_bounds: tight,
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::Mix,
                    lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
                    conditional_physical_anchors: Vec::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let bytes = build_test_dnam_container(&mut hdr, Tier::Lite, false);
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&bytes, Tier::Lite, 48_000).unwrap();
        assert!((engine.stereo_width() - 1.8).abs() < 1.0e-5);
        assert!((engine.mix_wet_gain - 0.95).abs() < 1.0e-5);

        engine.set_control_value(0, 1.0);
        assert!((engine.stereo_width() - 0.75).abs() < 1.0e-5);
        assert!((engine.mix_wet_gain - 0.75).abs() < 1.0e-5);
        assert!((engine.mix_dry_gain - 0.25).abs() < 1.0e-5);

        engine.set_control_immediate(0, 0.0);
        assert!((engine.stereo_width() - 1.8).abs() < 1.0e-5);
        assert!((engine.mix_wet_gain - 0.95).abs() < 1.0e-5);
        engine.set_control_immediate(0, 1.0);
        assert!((engine.stereo_width() - 0.75).abs() < 1.0e-5);
        assert!((engine.mix_wet_gain - 0.75).abs() < 1.0e-5);
    }
}
