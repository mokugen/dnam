//! Runtime control, `3 kHz` cadence, lifecycle, router, and telemetry methods on [`DynamicSonaVm`]
//! (`#![forbid(unsafe_code)]`, `#[inline(never)]` RT audit roots).

#![forbid(unsafe_code)]

use crate::graph::{IoAlignmentMode, WdwDryTapMode};
use crate::types::{Tier, VmError};

use super::super::SonaTelemetrySnapshot;
use super::DynamicSonaVm;

impl DynamicSonaVm {
    /// Executes one `3 kHz` boundary control tick (`Kautz`, `Tank`, `WE` Schmitt, and hoisting).
    #[inline(never)]
    pub fn step_control_tick_3khz(&mut self) {
        match self {
            Self::Scalar(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.step_control_tick_3khz(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.step_control_tick_3khz(),
        }
    }

    /// Updates a single physical control value.
    #[inline(never)]
    pub fn set_control_physical(&mut self, ctrl_idx: usize, val: f32) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_control_physical(ctrl_idx, val),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_control_physical(ctrl_idx, val),
        }
    }

    /// Atomically updates a batch of `(ctrl_idx, physical_val)` controls.
    #[inline(never)]
    pub fn set_controls_batch(&mut self, updates: &[(usize, f32)]) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_controls_batch(updates),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_controls_batch(updates),
        }
    }

    /// Applies `"TSYN"` tempo sync rows at `bpm`.
    #[inline(never)]
    pub fn apply_tempo_bpm(&mut self, bpm: f32) -> usize {
        match self {
            Self::Scalar(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.apply_tempo_bpm(bpm),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.apply_tempo_bpm(bpm),
        }
    }

    /// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across retained tank quads.
    #[inline(never)]
    pub fn set_quad_write_enable(&mut self, locked: bool) {
        self.set_op_t_write_enable(locked);
    }

    /// Alias matching [`super::super::SonaVm::set_op_t_write_enable`].
    #[inline(never)]
    pub fn set_op_t_write_enable(&mut self, locked: bool) {
        match self {
            Self::Scalar(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_op_t_write_enable(locked),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_op_t_write_enable(locked),
        }
    }

    /// Switches active tier (`0` RT allocs, `48`-frame output crossfade).
    #[inline(never)]
    pub fn switch_tier(&mut self, to_tier: Tier) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.switch_tier(to_tier),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.switch_tier(to_tier),
        }
    }

    /// Re-times all internal ZOH modules to a new host sample rate.
    #[inline(never)]
    pub fn retime_sample_rate(&mut self, new_host_rate_hz: f32) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.retime_sample_rate(new_host_rate_hz),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.retime_sample_rate(new_host_rate_hz),
        }
    }

    /// Toggles `4x` offline bounce oversampling.
    #[inline(never)]
    pub fn set_offline_bounce(&mut self, offline: bool) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_offline_bounce(offline),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_offline_bounce(offline),
        }
    }

    /// Updates post-VM Mid/Side stereo width in `[0.0, 2.0]` (non-finite inputs record telemetry).
    #[inline(never)]
    pub fn set_ms_width(&mut self, width: f32) {
        match self {
            Self::Scalar(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_ms_width(width),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_ms_width(width),
        }
    }

    /// Configures 4-port FX return alignment filter (`IoAlignmentMode`, L/R delays).
    #[inline(never)]
    pub fn configure_return_alignment(
        &mut self,
        mode: IoAlignmentMode,
        delay_l: f64,
        delay_r: f64,
    ) {
        match self {
            Self::Scalar(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.configure_return_alignment(mode, delay_l, delay_r),
        }
    }

    /// Selects the `WDW` center dry tap point and enables `WDW` 4-port output routing.
    #[inline(never)]
    pub fn set_wdw_mode(&mut self, mode: WdwDryTapMode) {
        match self {
            Self::Scalar(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_wdw_mode(mode),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_wdw_mode(mode),
        }
    }

    /// Enables or disables `WDW` 4-port center-dry / wet-only routing (`Port Out 1..3`).
    #[inline(never)]
    pub fn set_wdw_enabled(&mut self, enabled: bool) {
        match self {
            Self::Scalar(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_wdw_enabled(enabled),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_wdw_enabled(enabled),
        }
    }

    /// Restores deterministic resting zero equilibrium while preserving `HOST_WRITE_LOCK`.
    #[inline(never)]
    pub fn reset(&mut self) {
        match self {
            Self::Scalar(vm) => vm.reset(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.reset(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.reset(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.reset(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.reset(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.reset(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.reset(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.reset(),
        }
    }

    /// Enables or disables optional live OU analog drift (`false` automatically on `offline_bounce`).
    #[inline(never)]
    pub fn set_live_analog_drift_enabled(&mut self, enabled: bool) {
        match self {
            Self::Scalar(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.set_live_analog_drift_enabled(enabled),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.set_live_analog_drift_enabled(enabled),
        }
    }

    /// Returns processing latency in host samples (`0` live, `12` in `4x` offline bounce).
    #[inline(never)]
    #[must_use]
    pub fn get_latency_samples(&self) -> usize {
        match self {
            Self::Scalar(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.get_latency_samples(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.get_latency_samples(),
        }
    }

    /// Returns accumulated saturation/non-finite telemetry flags (`crate::math::saturation_flags`).
    #[inline(never)]
    #[must_use]
    pub fn saturation_flags(&self) -> u32 {
        match self {
            Self::Scalar(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.saturation_flags(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.saturation_flags(),
        }
    }

    /// Clears accumulated saturation/non-finite telemetry flags.
    #[inline(never)]
    pub fn clear_saturation_flags(&mut self) {
        match self {
            Self::Scalar(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.clear_saturation_flags(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.clear_saturation_flags(),
        }
    }

    /// Returns a snapshot of active tier, OS mode, lock bits, latency, and telemetry flags.
    #[inline(never)]
    #[must_use]
    pub fn telemetry(&self) -> SonaTelemetrySnapshot {
        match self {
            Self::Scalar(vm) => vm.telemetry(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.telemetry(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.telemetry(),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.telemetry(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.telemetry(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.telemetry(),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.telemetry(),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.telemetry(),
        }
    }
}
