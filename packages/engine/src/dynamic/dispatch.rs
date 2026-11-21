//! Method dispatch and tests for [`DynamicDnamEngine`].

use dnam_vm::ZohRateContext;
use dnam_vm::control::ControlClampDescriptor;

use super::DynamicDnamEngine;
use crate::container::DnamPreamble;
use crate::engine::{HostShellPatchbay, IoRoutingMode, ParallelAlignmentMode, WdwDryTapMode};
use crate::provenance::{PreviewGuardState, VerificationStatus};
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile, TempoSubdivision};
use crate::types::{DnamTelemetrySnapshot, Tier};

macro_rules! dispatch_engine {
    ($self:expr, $e:ident => $body:expr) => {
        match $self {
            DynamicDnamEngine::Scalar($e) => $body,
            #[cfg(target_arch = "aarch64")]
            DynamicDnamEngine::Neon($e) => $body,
            #[cfg(target_arch = "aarch64")]
            DynamicDnamEngine::ArmSve2($e) => $body,
            #[cfg(target_arch = "aarch64")]
            DynamicDnamEngine::ArmSme($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Sse2($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Sse42($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Avx2Fma($e) => $body,
            #[cfg(target_arch = "x86_64")]
            DynamicDnamEngine::X86Simd($e) => $body,
            #[cfg(target_arch = "wasm32")]
            DynamicDnamEngine::WasmSimd128($e) => $body,
        }
    };
}

impl<'a> DynamicDnamEngine<'a> {
    /// Returns the container's resolved [`DnamProtectionProfile`].
    #[inline]
    pub const fn protection_profile(&self) -> DnamProtectionProfile {
        dispatch_engine!(self, e => e.protection_profile())
    }

    /// Returns the container's cryptographic [`VerificationStatus`].
    #[inline]
    pub const fn verification_status(&self) -> &VerificationStatus {
        dispatch_engine!(self, e => e.verification_status())
    }

    /// Returns the loaded container [`DnamPreamble`].
    #[inline]
    pub const fn preamble(&self) -> &DnamPreamble {
        dispatch_engine!(self, e => e.preamble())
    }

    /// Returns the parsed [`DnamJsonHeader`] if loaded.
    #[inline]
    pub const fn header(&self) -> Option<&DnamJsonHeader> {
        dispatch_engine!(self, e => e.header())
    }

    /// Returns the active host sample rate in Hz.
    #[inline]
    pub fn sample_rate_hz(&self) -> u32 {
        dispatch_engine!(self, e => e.sample_rate_hz())
    }

    /// Returns the current [`ZohRateContext`] for the active host sample rate.
    #[inline]
    pub fn zoh(&self) -> &ZohRateContext {
        dispatch_engine!(self, e => e.zoh())
    }

    /// Switches the active [`Tier`].
    pub fn set_tier(&mut self, tier: Tier) {
        dispatch_engine!(self, e => e.set_tier(tier))
    }

    /// Returns the requested [`Tier`].
    #[inline]
    pub const fn tier(&self) -> Tier {
        dispatch_engine!(self, e => e.tier())
    }

    /// Returns the highest tier present in the container.
    #[inline]
    pub const fn max_available_tier(&self) -> Tier {
        dispatch_engine!(self, e => e.max_available_tier())
    }

    /// Returns the maximum tier permitted by the host.
    #[inline]
    pub const fn max_supported_tier(&self) -> Tier {
        dispatch_engine!(self, e => e.max_supported_tier())
    }

    /// Sets the host ceiling [`Tier`].
    pub fn set_max_supported_tier(&mut self, max_tier: Tier) {
        dispatch_engine!(self, e => e.set_max_supported_tier(max_tier))
    }

    /// Returns the currently active execution [`Tier`].
    #[inline]
    pub fn effective_tier(&self) -> Tier {
        dispatch_engine!(self, e => e.effective_tier())
    }

    /// Returns `true` when `4x` oversampled Offline Bounce mode is active.
    #[inline]
    pub const fn offline_bounce(&self) -> bool {
        dispatch_engine!(self, e => e.offline_bounce())
    }

    /// Enables or disables Offline Bounce mode (`4x` polyphase oversampling, `12`-sample PDC).
    pub fn set_offline_bounce(&mut self, enabled: bool) {
        dispatch_engine!(self, e => e.set_offline_bounce(enabled))
    }

    /// Returns whether Offline Bounce promotes execution to `min(max_available, max_supported)`.
    #[inline]
    pub const fn maximize_tier_on_bounce(&self) -> bool {
        dispatch_engine!(self, e => e.maximize_tier_on_bounce())
    }

    /// Enables or disables automatic tier maximization during Offline Bounce.
    pub fn set_maximize_tier_on_bounce(&mut self, enabled: bool) {
        dispatch_engine!(self, e => e.set_maximize_tier_on_bounce(enabled))
    }

    /// Returns active PDC latency in host samples.
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        dispatch_engine!(self, e => e.latency_samples())
    }

    /// Renders a complete stereo buffer in isolated Offline Bounce mode, stripping the 12-sample
    /// PDC lead-in while preserving live recursive state on `self`.
    pub fn render_offline_bounce(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.render_offline_bounce(left_in, right_in, left_out, right_out))
    }

    /// Reconfigures all ZOH parameters for a new host sample rate.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate_hz: u32) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.set_sample_rate(sample_rate_hz))
    }

    /// Real-time zero-allocation sample-rate retiming returning `bool` (zero `DnamError` drop glue).
    #[inline(never)]
    pub(crate) fn set_sample_rate_rt(&mut self, sample_rate_hz: u32) -> bool {
        dispatch_engine!(self, e => e.reconfigure_for_sample_rate_rt(sample_rate_hz).is_ok())
    }

    /// Schedules a smoothed physical control value change.
    pub fn set_control_value(&mut self, ctrl_idx: usize, value: f32) {
        dispatch_engine!(self, e => e.set_control_value(ctrl_idx, value))
    }

    /// Sets a control value and updates static DSP parameters immediately.
    pub fn set_control_immediate(&mut self, ctrl_idx: usize, value: f32) {
        dispatch_engine!(self, e => e.set_control_immediate(ctrl_idx, value))
    }

    /// Returns the number of controls in the loaded container.
    #[inline]
    pub fn num_controls(&self) -> usize {
        dispatch_engine!(self, e => e.num_controls())
    }

    /// Returns raw user-requested physical control values.
    #[inline]
    pub fn raw_control_values(&self) -> &[f32] {
        dispatch_engine!(self, e => e.raw_control_values())
    }

    /// Returns clamped physical control values.
    #[inline]
    pub fn clamped_control_values(&self) -> &[f32] {
        dispatch_engine!(self, e => e.clamped_control_values())
    }

    /// Returns compiled control clamp descriptors.
    #[inline]
    pub fn control_descriptors(&self) -> &[ControlClampDescriptor] {
        dispatch_engine!(self, e => e.control_descriptors())
    }

    /// Snaps all controls back to schema defaults.
    #[inline]
    pub fn reset_controls_to_defaults(&mut self) {
        dispatch_engine!(self, e => e.reset_controls_to_defaults())
    }

    /// Resets recursive DSP state to exact DC equilibrium.
    pub fn reset_to_dc_equilibrium(&mut self) {
        dispatch_engine!(self, e => e.reset_to_dc_equilibrium())
    }

    /// Returns whether storefront preview containment mode is active.
    #[inline]
    pub const fn is_preview_mode(&self) -> bool {
        dispatch_engine!(self, e => e.is_preview_mode())
    }

    /// Returns an immutable view of [`PreviewGuardState`].
    #[inline]
    pub const fn preview_guard(&self) -> &PreviewGuardState {
        dispatch_engine!(self, e => e.preview_guard())
    }

    /// Returns an immutable view of [`HostShellPatchbay`].
    #[inline]
    pub const fn patchbay(&self) -> &HostShellPatchbay {
        dispatch_engine!(self, e => e.patchbay())
    }

    /// Returns a mutable view of [`HostShellPatchbay`].
    #[inline]
    pub fn patchbay_mut(&mut self) -> &mut HostShellPatchbay {
        dispatch_engine!(self, e => e.patchbay_mut())
    }

    /// Selects the host shell multi-port routing mode.
    pub fn set_routing_mode(&mut self, mode: IoRoutingMode) {
        dispatch_engine!(self, e => e.set_routing_mode(mode))
    }

    /// Selects the WDW dry center tap mode.
    #[inline(never)]
    pub fn set_wdw_mode(&mut self, mode: WdwDryTapMode) {
        dispatch_engine!(self, e => e.set_wdw_mode(mode))
    }

    /// Enables or disables WDW routing.
    #[inline(never)]
    pub fn set_wdw_enabled(&mut self, enabled: bool) {
        dispatch_engine!(self, e => e.set_wdw_enabled(enabled))
    }

    /// Configures external FX return alignment delays in VM samples.
    pub fn configure_return_alignment(&mut self, delay_l_samples: f32, delay_r_samples: f32) {
        dispatch_engine!(self, e => e.configure_return_alignment(delay_l_samples, delay_r_samples))
    }

    /// Configures parallel-path Kaiser/Catmull-Rom alignment.
    pub fn configure_parallel_alignment(
        &mut self,
        mode: ParallelAlignmentMode,
        enabled: bool,
        delta_l_48k: f32,
        delta_r_48k: f32,
    ) {
        dispatch_engine!(self, e => e.configure_parallel_alignment(mode, enabled, delta_l_48k, delta_r_48k))
    }

    /// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across all active `TransportQuad`s.
    pub fn set_quad_write_enable(&mut self, locked: bool) {
        dispatch_engine!(self, e => e.set_quad_write_enable(locked))
    }

    /// Sets or clears host-direct `Op-T` parameter lock.
    pub fn set_op_t_write_enable(&mut self, locked: bool) {
        dispatch_engine!(self, e => e.set_op_t_write_enable(locked))
    }

    /// Enables or disables live Ornstein–Uhlenbeck analog drift.
    #[inline]
    pub fn set_live_analog_drift(&mut self, enabled: bool) {
        dispatch_engine!(self, e => e.set_live_analog_drift(enabled))
    }

    /// Returns whether live Ornstein–Uhlenbeck analog drift is enabled.
    #[inline]
    pub fn live_analog_drift_enabled(&self) -> bool {
        dispatch_engine!(self, e => e.live_analog_drift_enabled())
    }

    /// Sets bypass state and spillover trails behavior.
    #[inline(never)]
    pub fn set_bypass(&mut self, bypassed: bool, trails: bool) {
        dispatch_engine!(self, e => e.set_bypass(bypassed, trails))
    }

    /// Returns `(bypassed, trails_enabled)`.
    #[inline]
    pub const fn bypass_state(&self) -> (bool, bool) {
        dispatch_engine!(self, e => e.bypass_state())
    }

    /// Sets post-VM M/S stereo width clamped to `[0.0, 2.0]`.
    pub fn set_stereo_width(&mut self, width: f32) {
        dispatch_engine!(self, e => e.set_stereo_width(width))
    }

    /// Alias matching [`HostShellPatchbay::set_ms_width`].
    #[inline]
    pub fn set_ms_width(&mut self, width: f32) {
        dispatch_engine!(self, e => e.set_ms_width(width))
    }

    /// Returns the current M/S stereo width.
    #[inline]
    pub const fn stereo_width(&self) -> f32 {
        dispatch_engine!(self, e => e.stereo_width())
    }

    /// Configures host parallel blend, output gain, and auto-unity leveling.
    #[inline]
    pub fn configure_host_mix_and_leveling(
        &mut self,
        blend_wet: f32,
        output_gain_db: f32,
        apply_unity_leveling: bool,
    ) {
        dispatch_engine!(self, e => e.configure_host_mix_and_leveling(blend_wet, output_gain_db, apply_unity_leveling))
    }

    /// Returns the active host tempo in BPM.
    #[inline]
    pub const fn current_bpm(&self) -> f32 {
        dispatch_engine!(self, e => e.current_bpm())
    }

    /// Sets host BPM without modifying control values.
    #[inline]
    pub fn set_bpm(&mut self, bpm: f32) {
        dispatch_engine!(self, e => e.set_bpm(bpm))
    }

    /// Synchronizes all tempo-bound controls to `(bpm, subdivision_override, fold_octave)`.
    pub fn sync_tempo(
        &mut self,
        bpm: f32,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> usize {
        dispatch_engine!(self, e => e.sync_tempo(bpm, subdivision_override, fold_octave))
    }

    /// Registers a tap-tempo event and updates synchronized controls.
    pub fn register_tap_tempo(
        &mut self,
        subdivision_override: Option<TempoSubdivision>,
        fold_octave: bool,
    ) -> (f32, usize) {
        dispatch_engine!(self, e => e.register_tap_tempo(subdivision_override, fold_octave))
    }

    /// Applies a preset by string identifier.
    #[inline]
    pub fn apply_preset(&mut self, preset_id: &str) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.apply_preset(preset_id))
    }

    /// Applies a preset by index without heap allocations.
    #[inline]
    pub fn apply_preset_by_index(&mut self, preset_index: usize) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.apply_preset_by_index(preset_index))
    }

    #[inline]
    pub(crate) fn apply_preset_by_index_rt(&mut self, preset_index: usize) -> bool {
        dispatch_engine!(self, e => e.apply_preset_by_index_rt(preset_index))
    }

    /// Processes stereo audio in-place (`0` heap allocations).
    pub fn process_block_in_place(&mut self, left_io: &mut [f32], right_io: &mut [f32]) {
        dispatch_engine!(self, e => e.process_block_in_place(left_io, right_io))
    }

    /// Processes a stereo audio block (`0` heap allocations).
    pub fn process_block(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_block(left_in, right_in, left_out, right_out))
    }

    /// Processes a stereo audio block (`Stereo -> Stereo`), matching `process_block`.
    pub fn process_stereo(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_stereo(left_in, right_in, left_out, right_out))
    }

    /// Processes a stereo audio block into mono (`Stereo -> Mono`).
    pub fn process_stereo_to_mono(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        mono_out: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_stereo_to_mono(left_in, right_in, mono_out))
    }

    /// Processes a mono audio block into true stereo (`Mono -> Stereo`).
    pub fn process_mono_to_stereo(
        &mut self,
        mono_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_mono_to_stereo(mono_in, left_out, right_out))
    }

    /// Processes a mono audio block (`Mono -> Mono`).
    pub fn process_mono(&mut self, mono_in: &[f32], mono_out: &mut [f32]) {
        dispatch_engine!(self, e => e.process_mono(mono_in, mono_out))
    }

    /// Processes a multi-port audio block (`Stage 0` 8-port router contract).
    pub fn process_block_io(&mut self, io: crate::engine::DnamAudioIo<'_>) {
        dispatch_engine!(self, e => e.process_block_io(io))
    }

    /// Processes a 4-Cable Method (`4CM`) block.
    #[expect(
        clippy::too_many_arguments,
        reason = "4-Cable Method exposes explicit 4 stereo I/O port buffers"
    )]
    pub fn process_block_4cm(
        &mut self,
        inst_l: &[f32],
        inst_r: &[f32],
        ret_l: &[f32],
        ret_r: &[f32],
        main_l: &mut [f32],
        main_r: &mut [f32],
        send_l: &mut [f32],
        send_r: &mut [f32],
    ) {
        dispatch_engine!(
            self,
            e => e.process_block_4cm(inst_l, inst_r, ret_l, ret_r, main_l, main_r, send_l, send_r)
        )
    }

    /// Processes a Wet/Dry/Wet (`WDW`) 3-output block.
    pub fn process_block_wdw(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        wet_l: &mut [f32],
        wet_r: &mut [f32],
        dry_center: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_block_wdw(in_l, in_r, wet_l, wet_r, dry_center))
    }

    /// Processes a sidechain / auxiliary return stereo block.
    pub fn process_block_sidechain(
        &mut self,
        main_l: &[f32],
        main_r: &[f32],
        aux_l: &[f32],
        aux_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        dispatch_engine!(self, e => e.process_block_sidechain(main_l, main_r, aux_l, aux_r, out_l, out_r))
    }

    /// Low-level single-frame `SONA` stepping (`0 FDIV`, `0 FSQRT`).
    pub fn step_frame_sona(&mut self, in_l: f32, in_r: f32, mono_only: bool) -> [f32; 2] {
        dispatch_engine!(self, e => e.step_frame_sona(in_l, in_r, mono_only))
    }

    /// Hot-swaps a new canonical `v3.0` `"sona"` `.dnam` model in-place without heap allocations
    /// (`bytes` must live for `'a`).
    pub fn reload_model_in_place(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.reload_model_in_place(bytes, tier, sample_rate_hz))
    }

    /// Hot-swaps a new `.dnam` model in-place into an owned engine without requiring `bytes`
    /// to outlive `'a`.
    #[inline]
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        dispatch_engine!(self, e => e.reload_model_in_place_owned(bytes, tier, sample_rate_hz))
    }

    /// Returns a lock-free [`DnamTelemetrySnapshot`].
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        dispatch_engine!(self, e => e.telemetry_snapshot())
    }

    #[inline]
    pub(crate) fn tempo_total_frames(&self) -> u64 {
        dispatch_engine!(self, e => e.tempo_total_frames())
    }

    #[inline]
    pub(crate) fn tier_os2x(&self) -> [bool; 4] {
        dispatch_engine!(self, e => e.tier_os2x())
    }

    #[inline]
    pub(crate) fn ref_sample_rate_hz(&self) -> f32 {
        dispatch_engine!(self, e => e.ref_sample_rate_hz())
    }

    #[cfg(any(test, feature = "std"))]
    #[inline]
    pub(crate) fn compute_zoh_for_mode(
        &self,
        sample_rate_hz: u32,
        offline_bounce: bool,
    ) -> ZohRateContext {
        dispatch_engine!(self, e => e.compute_zoh_for_mode(sample_rate_hz, offline_bounce))
    }

    #[inline]
    pub(crate) fn or_active_clamp_mask(&mut self, mask: u16) {
        dispatch_engine!(self, e => e.or_active_clamp_mask(mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::DnamJsonHeader;
    use crate::test_support::build_test_dnam_container;
    use dnam_vm::math::BackendKind;

    #[test]
    fn test_dynamic_dnam_engine_all_variants() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Ultra, true);

        let kinds = [
            BackendKind::Scalar,
            BackendKind::Neon,
            BackendKind::ArmSve2,
            BackendKind::ArmSme,
            BackendKind::X86Sse2,
            BackendKind::X86Sse42,
            BackendKind::X86Avx2Fma,
            BackendKind::X86Simd,
            BackendKind::WasmSimd128,
        ];
        for kind in kinds {
            let mut dyn_eng =
                DynamicDnamEngine::from_bytes_with_backend(&container, Tier::Ultra, 48_000, kind)
                    .unwrap();
            let _ = dyn_eng.backend_kind();
            dyn_eng.set_tier(Tier::Lite);
            dyn_eng.set_sample_rate(44_100).unwrap();
            dyn_eng.set_control_value(0, 5.0);
            dyn_eng.set_control_immediate(0, 5.0);
            let _ = dyn_eng.protection_profile();
            let _ = dyn_eng.tier();
            let _ = dyn_eng.max_available_tier();
            let _ = dyn_eng.max_supported_tier();
            dyn_eng.set_max_supported_tier(Tier::Full);
            dyn_eng.set_offline_bounce(true);
            assert!(dyn_eng.offline_bounce());
            assert_eq!(dyn_eng.latency_samples(), 12);
            dyn_eng.set_offline_bounce(false);
            assert_eq!(dyn_eng.latency_samples(), 0);
            dyn_eng.set_maximize_tier_on_bounce(true);
            assert!(dyn_eng.maximize_tier_on_bounce());
            assert_eq!(dyn_eng.effective_tier(), Tier::Lite);
            assert_eq!(dyn_eng.sample_rate_hz(), 44_100);
            assert_eq!(dyn_eng.zoh().host_rate_hz, 44_100.0);
            let _ = dyn_eng.verification_status();
            let _ = dyn_eng.preamble();
            assert!(dyn_eng.header().is_some());
            assert_eq!(dyn_eng.num_controls(), 0);
            assert_eq!(dyn_eng.raw_control_values().len(), dyn_eng.num_controls());
            assert_eq!(
                dyn_eng.clamped_control_values().len(),
                dyn_eng.num_controls()
            );
            assert_eq!(dyn_eng.control_descriptors().len(), dyn_eng.num_controls());
            dyn_eng.reset_controls_to_defaults();
            dyn_eng.set_bypass(true, true);
            assert_eq!(dyn_eng.bypass_state(), (true, true));
            dyn_eng.set_bypass(false, false);
            assert_eq!(dyn_eng.bypass_state(), (false, false));
            dyn_eng.set_stereo_width(1.5);
            assert!((dyn_eng.stereo_width() - 1.5).abs() < 1e-6);
            dyn_eng.configure_host_mix_and_leveling(0.8, -1.5, true);
            let _ = dyn_eng.sync_tempo(128.0, Some(TempoSubdivision::Eighth), false);
            assert!((dyn_eng.current_bpm() - 128.0).abs() < 1e-6);
            let (tap_bpm, _) = dyn_eng.register_tap_tempo(None, false);
            assert!(tap_bpm >= 20.0);
            assert!(dyn_eng.apply_preset("nonexistent").is_err());

            let l_in = [0.1f32; 32];
            let r_in = [-0.1f32; 32];
            let mut l_out = [0.0f32; 32];
            let mut r_out = [0.0f32; 32];
            assert!(dyn_eng.set_sample_rate_rt(96_000));
            assert!(!dyn_eng.set_sample_rate_rt(12_000));
            assert!(dyn_eng.set_sample_rate_rt(48_000));
            dyn_eng.set_routing_mode(IoRoutingMode::Stereo);
            dyn_eng.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
            dyn_eng.set_wdw_enabled(false);
            dyn_eng.configure_return_alignment(0.0, 0.0);
            dyn_eng.configure_parallel_alignment(
                ParallelAlignmentMode::FullBand64Tap,
                false,
                0.0,
                0.0,
            );
            dyn_eng.set_quad_write_enable(false);
            dyn_eng.set_op_t_write_enable(false);
            let frame_y = dyn_eng.step_frame_sona(0.1, -0.1, false);
            assert!(frame_y[0].is_finite() && frame_y[1].is_finite());
            dyn_eng.render_offline_bounce(&l_in, &r_in, &mut l_out, &mut r_out);
            dyn_eng.process_block(&l_in, &r_in, &mut l_out, &mut r_out);
            dyn_eng.process_stereo(&l_in, &r_in, &mut l_out, &mut r_out);
            dyn_eng.process_stereo_to_mono(&l_in, &r_in, &mut l_out);
            let mut l_ip = l_in;
            let mut r_ip = r_in;
            dyn_eng.process_block_in_place(&mut l_ip, &mut r_ip);
            dyn_eng.process_mono_to_stereo(&l_in, &mut l_out, &mut r_out);
            dyn_eng.process_mono(&l_in, &mut l_out);
            dyn_eng.process_block_io(crate::engine::DnamAudioIo {
                in_1: &l_in,
                in_2: Some(&r_in),
                in_3: None,
                in_4: None,
                out_1: &mut l_out,
                out_2: Some(&mut r_out),
                out_3: None,
                out_4: None,
            });
            let mut send_l = [0.0f32; 32];
            let mut send_r = [0.0f32; 32];
            dyn_eng.process_block_4cm(
                &l_in,
                &r_in,
                &l_in,
                &r_in,
                &mut l_out,
                &mut r_out,
                &mut send_l,
                &mut send_r,
            );
            let mut dry_center = [0.0f32; 32];
            dyn_eng.process_block_wdw(&l_in, &r_in, &mut l_out, &mut r_out, &mut dry_center);
            dyn_eng.process_block_sidechain(&l_in, &r_in, &l_in, &r_in, &mut l_out, &mut r_out);
            dyn_eng.reset_to_dc_equilibrium();
            dyn_eng
                .reload_model_in_place(&container, Tier::Lite, 48_000)
                .unwrap();
            dyn_eng
                .reload_model_in_place_owned(&container, Tier::Lite, 48_000)
                .unwrap();
            let snap = dyn_eng.telemetry_snapshot();
            assert_eq!(snap.active_tier, Tier::Lite as u8);

            let _owned_kind = DynamicDnamEngine::from_bytes_owned_with_backend(
                &container,
                Tier::Lite,
                48_000,
                kind,
            )
            .unwrap();
        }

        let _owned = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        let _auto = DynamicDnamEngine::from_bytes(&container, Tier::Lite, 48_000).unwrap();
    }

    proptest::proptest! {
        #[test]
        fn prop_dynamic_engine_in_place_and_forwarding_parity(
            width in 0.0f32..=2.0f32,
            bpm in 20.0f32..=400.0f32,
            bypassed in proptest::bool::ANY,
            trails in proptest::bool::ANY,
            blend_wet in 0.0f32..=1.0f32,
            gain_db in -12.0f32..=12.0f32,
            in_sample in -1.0f32..=1.0f32,
        ) {
            let mut header = DnamJsonHeader::default_reference();
            let container = build_test_dnam_container(&mut header, Tier::Ultra, true);

            let mut eng_oop = DynamicDnamEngine::from_bytes_with_backend(
                &container,
                Tier::Lite,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();
            let mut eng_ip = DynamicDnamEngine::from_bytes_with_backend(
                &container,
                Tier::Lite,
                48_000,
                BackendKind::Scalar,
            )
            .unwrap();

            for eng in [&mut eng_oop, &mut eng_ip] {
                eng.set_stereo_width(width);
                eng.set_bypass(bypassed, trails);
                eng.configure_host_mix_and_leveling(blend_wet, gain_db, true);
                eng.sync_tempo(bpm, None, false);
            }

            let l_in = [in_sample; 16];
            let r_in = [-in_sample * 0.5; 16];
            let mut l_out = [0.0f32; 16];
            let mut r_out = [0.0f32; 16];
            eng_oop.process_block(&l_in, &r_in, &mut l_out, &mut r_out);

            let mut l_ip = l_in;
            let mut r_ip = r_in;
            eng_ip.process_block_in_place(&mut l_ip, &mut r_ip);

            for i in 0..16 {
                proptest::prop_assert_eq!(l_out[i].to_bits(), l_ip[i].to_bits());
                proptest::prop_assert_eq!(r_out[i].to_bits(), r_ip[i].to_bits());
            }
        }
    }
}
