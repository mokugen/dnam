//! Deterministic `#![no_std]` + `!alloc`-capable `SONA v3.0` Virtual Machine (`SonaVm`),
//! `!alloc` [`SonaVmBacking`], optional `alloc` [`SonaInstance`] / [`dynamic::DynamicSonaVm`],
//! telemetry, and concrete RT disassembly audit roots (`#![forbid(unsafe_code)]`).

#[cfg(feature = "alloc")]
pub mod dynamic;

use dnam_kernels::{QUAD_FLAG_HOST_WRITE_LOCK, QUAD_FLAG_WE_HOLD_ACTIVE};

use crate::bin::DnamBinView;
#[cfg(feature = "alloc")]
use crate::graph::slab::hydrate_slab_owned;
use crate::graph::slab::{
    SonaHotSlab, SonaSlabBacking, hydrate_slab_from_dnam_bin, process_block_sona,
    reset_to_dc_equilibrium, retime_sample_rate, set_offline_bounce, set_op_t_write_enable,
    step_control_tick_3khz, step_frame_4port_sona, step_frame_4x_sona, step_frame_sona,
    switch_tier,
};
use crate::graph::{
    Bounce4xOversampler, GraphBackend, IoAlignmentMode, WdwDryTapMode, voltage_ms_encode,
};
use crate::math::{DefaultMathBackend, const_abs_f32, sanitize_output_sample, saturation_flags};
use crate::types::{FtzDazGuard, Tier, VmError};

#[cfg(feature = "alloc")]
pub use dynamic::DynamicSonaVm;

/// `!alloc` caller-borrowed cold backing required by [`SonaVm::open_dnam_bin`].
pub type SonaVmBacking<'a, B = DefaultMathBackend> = SonaSlabBacking<'a, B>;

/// Optional owned instance alias matching [`crate::vm::SonaVm`] (`feature = "alloc"`).
#[cfg(feature = "alloc")]
pub type SonaInstance<'a, B = DefaultMathBackend> = SonaVm<'a, B>;

/// Snapshot of deterministic runtime state, latency, lock flags, and saturation telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SonaTelemetrySnapshot {
    pub saturation_flags: u32,
    pub active_tier: Tier,
    pub max_present_tier: Tier,
    pub retained_tier: Tier,
    pub os2x_active: bool,
    pub offline_bounce: bool,
    pub we_hold_active: bool,
    pub host_write_lock: bool,
    pub latency_samples: usize,
    pub sample_tick: u32,
}

#[inline(always)]
fn apply_ms_width_and_sanitize(ms_width: f32, l: f32, r: f32, sat_flags: &mut u32) -> (f32, f32) {
    let (yl, yr) = if const_abs_f32(ms_width - 1.0) <= 1.0e-7 {
        (l, r)
    } else {
        let (mid, side) = voltage_ms_encode(l, r);
        let scaled = ms_width * side;
        (mid + scaled, mid - scaled)
    };
    (
        sanitize_output_sample(yl, sat_flags),
        sanitize_output_sample(yr, sat_flags),
    )
}

/// Deterministic `SONA v3.0` Virtual Machine wrapping [`SonaHotSlab`] (`#![no_std]`, `#![forbid(unsafe_code)]`).
#[derive(Debug)]
pub struct SonaVm<'a, B: GraphBackend = DefaultMathBackend> {
    slab: SonaHotSlab<'a, B>,
    ms_width: f32,
}

impl<'a, B: GraphBackend> SonaVm<'a, B> {
    /// Parses, CRC-verifies, and hydrates a `.dnam-bin` (`"DNBN"`) payload over `!alloc` [`SonaVmBacking`].
    pub fn open_dnam_bin(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
        backing: SonaVmBacking<'a, B>,
    ) -> Result<Self, VmError> {
        let view = DnamBinView::parse(bytes)?;
        Self::from_view(&view, active_tier, host_rate_hz, false, backing)
    }

    /// Binds a pre-authenticated `.dnam-bin` buffer in `< 50 ns` and hydrates over `!alloc` backing.
    pub fn open_dnam_bin_prevalidated(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
        backing: SonaVmBacking<'a, B>,
    ) -> Result<Self, VmError> {
        let view = DnamBinView::bind_prevalidated(bytes)?;
        Self::from_view(&view, active_tier, host_rate_hz, false, backing)
    }

    /// Hydrates a [`SonaVm`] from an already validated [`DnamBinView`].
    pub fn from_view(
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
        backing: SonaVmBacking<'a, B>,
    ) -> Result<Self, VmError> {
        let slab =
            hydrate_slab_from_dnam_bin(view, active_tier, host_rate_hz, offline_bounce, backing)?;
        Ok(Self {
            slab,
            ms_width: 1.0,
        })
    }

    /// Parses, CRC-verifies, and heap-hydrates a `.dnam-bin` payload (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn open_dnam_bin_owned(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
    ) -> Result<SonaVm<'static, B>, VmError> {
        let view = DnamBinView::parse(bytes)?;
        SonaVm::<B>::from_view_owned(&view, active_tier, host_rate_hz, false)
    }

    /// Binds a pre-authenticated `.dnam-bin` buffer in `< 50 ns` and heap-hydrates (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn open_dnam_bin_prevalidated_owned(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
    ) -> Result<SonaVm<'static, B>, VmError> {
        let view = DnamBinView::bind_prevalidated(bytes)?;
        SonaVm::<B>::from_view_owned(&view, active_tier, host_rate_hz, false)
    }

    /// Heap-hydrates a [`SonaVm`] from a validated [`DnamBinView`] (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn from_view_owned(
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
    ) -> Result<SonaVm<'static, B>, VmError> {
        let slab = hydrate_slab_owned(view, active_tier, host_rate_hz, offline_bounce)?;
        Ok(SonaVm {
            slab,
            ms_width: 1.0,
        })
    }

    /// Processes a stereo audio block under hardware `FTZ`/`DAZ` + `RNE` protection (`#[inline(never)]`).
    #[inline(never)]
    pub fn process_block(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), VmError> {
        let _ftz = FtzDazGuard::new();
        process_block_sona(&mut self.slab, in_l, in_r, out_l, out_r)?;
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        let width = self.ms_width;
        let flags = &mut self.slab.cold.as_mut_slice()[0].saturation_flags;
        for i in 0..n {
            let (yl, yr) = apply_ms_width_and_sanitize(width, out_l[i], out_r[i], flags);
            out_l[i] = yl;
            out_r[i] = yr;
        }
        Ok(())
    }

    /// Processes a fixed `16`-frame slice (`3 kHz` tick at `48 kHz`) under `FTZ`/`DAZ` guard.
    #[inline(never)]
    pub fn process_tick16_const(
        &mut self,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32; 16],
        out_r: &mut [f32; 16],
    ) -> Result<(), VmError> {
        let _ftz = FtzDazGuard::new();
        self.slab.process_tick16_const(in_l, in_r, out_l, out_r)?;
        let width = self.ms_width;
        let flags = &mut self.slab.cold.as_mut_slice()[0].saturation_flags;
        for i in 0..16 {
            let (yl, yr) = apply_ms_width_and_sanitize(width, out_l[i], out_r[i], flags);
            out_l[i] = yl;
            out_r[i] = yr;
        }
        Ok(())
    }

    /// Steps one stereo host sample (`1x` live or `4x` offline bounce) under `FTZ`/`DAZ` guard.
    #[inline(never)]
    pub fn step_frame(&mut self, in_l: f32, in_r: f32) -> Result<(f32, f32), VmError> {
        let _ftz = FtzDazGuard::new();
        let (l, r) = if self.slab.cold().offline_bounce {
            step_frame_4x_sona(&mut self.slab, in_l, in_r)?
        } else {
            step_frame_sona(&mut self.slab, in_l, in_r)?
        };
        let width = self.ms_width;
        let flags = &mut self.slab.cold.as_mut_slice()[0].saturation_flags;
        Ok(apply_ms_width_and_sanitize(width, l, r, flags))
    }

    /// Steps one `4`-port (`4CM`/`WDW`) stereo frame (`(main_lr, send_lr)`) under `FTZ`/`DAZ` guard.
    #[inline(never)]
    pub fn step_frame_4port(
        &mut self,
        inst_lr: [f32; 2],
        ret_lr: [f32; 2],
        has_return: bool,
    ) -> Result<([f32; 2], [f32; 2]), VmError> {
        let _ftz = FtzDazGuard::new();
        let (main_lr, send_lr) =
            step_frame_4port_sona(&mut self.slab, inst_lr, ret_lr, has_return)?;
        let width = self.ms_width;
        let flags = &mut self.slab.cold.as_mut_slice()[0].saturation_flags;
        let (ml, mr) = apply_ms_width_and_sanitize(width, main_lr[0], main_lr[1], flags);
        Ok(([ml, mr], send_lr))
    }

    /// Executes one `3 kHz` boundary control tick (`Kautz`, `Tank`, `WE` Schmitt, and core hoisting).
    #[inline(never)]
    pub fn step_control_tick_3khz(&mut self) {
        step_control_tick_3khz(&mut self.slab);
    }

    /// Updates a single physical control value and flags the control plane for refresh.
    pub fn set_control_physical(&mut self, ctrl_idx: usize, val: f32) -> Result<(), VmError> {
        if !val.is_finite() {
            self.slab.cold_mut().saturation_flags |= saturation_flags::INVALID_NONFINITE;
        }
        let cold = self.slab.cold_mut();
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.set_control_physical(ctrl_idx, val)
        } else {
            Err(VmError::InvalidControlDescriptor)
        }
    }

    /// Atomically updates a batch of `(ctrl_idx, physical_val)` controls.
    pub fn set_controls_batch(&mut self, updates: &[(usize, f32)]) -> Result<(), VmError> {
        for &(_, v) in updates {
            if !v.is_finite() {
                self.slab.cold_mut().saturation_flags |= saturation_flags::INVALID_NONFINITE;
            }
        }
        let cold = self.slab.cold_mut();
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.set_controls_batch(updates)
        } else {
            Err(VmError::InvalidControlDescriptor)
        }
    }

    /// Evaluates active `"TSYN"` tempo-sync mappings at `bpm` (`0` when no rows apply).
    pub fn apply_tempo_bpm(&mut self, bpm: f32) -> usize {
        if !bpm.is_finite() {
            self.slab.cold_mut().saturation_flags |= saturation_flags::INVALID_NONFINITE;
            return 0;
        }
        let cold = self.slab.cold_mut();
        if let Some(bus) = cold.ctrl_bus.as_mut() {
            bus.apply_tempo_bpm(bpm)
        } else {
            0
        }
    }

    /// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across all active tank quads.
    #[inline(never)]
    pub fn set_quad_write_enable(&mut self, locked: bool) {
        self.set_op_t_write_enable(locked);
    }

    /// Alias matching [`SonaHotSlab`] naming (`set_op_t_write_enable`).
    #[inline(never)]
    pub fn set_op_t_write_enable(&mut self, locked: bool) {
        set_op_t_write_enable(&mut self.slab, locked);
    }

    /// Updates post-VM Mid/Side stereo width in `[0.0, 2.0]` (non-finite inputs record telemetry).
    pub fn set_ms_width(&mut self, width: f32) {
        if width.is_finite() {
            self.ms_width = width.clamp(0.0, 2.0);
        } else {
            self.slab.cold_mut().saturation_flags |= saturation_flags::INVALID_NONFINITE;
        }
    }

    /// Switches active tier (`0` RT allocs, `0` core reset, `48`-frame output crossfade).
    pub fn switch_tier(&mut self, to_tier: Tier) -> Result<(), VmError> {
        switch_tier(&mut self.slab, to_tier)
    }

    /// Re-times all internal ZOH filters, Hilbert H1, Cabinet FIR, Kautz, Tank, and `ControlBus`.
    pub fn retime_sample_rate(&mut self, new_host_rate_hz: f32) -> Result<(), VmError> {
        retime_sample_rate(&mut self.slab, new_host_rate_hz)
    }

    /// Toggles `4x` offline bounce linear-phase oversampling.
    pub fn set_offline_bounce(&mut self, offline: bool) -> Result<(), VmError> {
        set_offline_bounce(&mut self.slab, offline)
    }

    /// Restores deterministic resting zero equilibrium while preserving `HOST_WRITE_LOCK` (bit 2).
    pub fn reset(&mut self) {
        reset_to_dc_equilibrium(&mut self.slab);
    }

    /// Configures 4-port FX return alignment filter (`IoAlignmentMode`, L/R delays).
    pub fn configure_return_alignment(
        &mut self,
        mode: IoAlignmentMode,
        delay_l: f64,
        delay_r: f64,
    ) {
        let m_os = if self.slab.cold().offline_bounce {
            4
        } else {
            1
        };
        self.slab
            .cold_mut()
            .router
            .configure_return_alignment(mode, delay_l, delay_r, m_os);
    }

    /// Selects the `WDW` center dry tap point and enables `WDW` 4-port output routing.
    #[inline(never)]
    pub fn set_wdw_mode(&mut self, mode: WdwDryTapMode) {
        let router = &mut self.slab.cold_mut().router;
        router.wdw_mode = mode;
        router.wdw_active = true;
    }

    /// Enables or disables `WDW` 4-port center-dry / wet-only routing (`Port Out 1..3`).
    #[inline(never)]
    pub fn set_wdw_enabled(&mut self, enabled: bool) {
        self.slab.cold_mut().router.wdw_active = enabled;
    }

    /// Enables or disables optional live OU analog drift (automatically bypassed on `offline_bounce`).
    #[inline(never)]
    pub fn set_live_analog_drift_enabled(&mut self, enabled: bool) {
        let scrubbed = if let Some(bus) = self.slab.cold_mut().ctrl_bus.as_mut() {
            bus.set_live_analog_drift_enabled(enabled)
        } else {
            false
        };
        if scrubbed {
            crate::graph::latch_immediate_at_init_or_reset(&mut self.slab);
        }
    }

    /// Returns processing latency in host samples (`0` live, `12` in `4x` offline bounce).
    #[inline]
    #[must_use]
    pub fn get_latency_samples(&self) -> usize {
        if self.slab.cold().offline_bounce {
            Bounce4xOversampler::pdc_host_samples()
        } else {
            0
        }
    }

    /// Returns accumulated saturation/non-finite telemetry flags (`crate::math::saturation_flags`).
    #[inline]
    #[must_use]
    pub fn saturation_flags(&self) -> u32 {
        self.slab.cold().saturation_flags
    }

    /// Clears accumulated saturation/non-finite telemetry flags.
    #[inline(never)]
    pub fn clear_saturation_flags(&mut self) {
        self.slab.cold_mut().saturation_flags = 0;
    }

    /// Returns a lock-free snapshot of active tier, OS mode, lock bits, latency, and telemetry flags.
    #[must_use]
    pub fn telemetry(&self) -> SonaTelemetrySnapshot {
        let cold = self.slab.cold();
        let mut we_hold = false;
        let mut host_lock = false;
        for q in 0..cold.active_tank_quads {
            let f = self.slab.quad_hot(q).flags;
            we_hold |= (f & QUAD_FLAG_WE_HOLD_ACTIVE) != 0;
            host_lock |= (f & QUAD_FLAG_HOST_WRITE_LOCK) != 0;
        }
        SonaTelemetrySnapshot {
            saturation_flags: cold.saturation_flags,
            active_tier: self.slab.lite.tier,
            max_present_tier: cold.max_present_tier,
            retained_tier: cold.retained_tier,
            os2x_active: self.slab.os2x(),
            offline_bounce: cold.offline_bounce,
            we_hold_active: we_hold,
            host_write_lock: host_lock,
            latency_samples: self.get_latency_samples(),
            sample_tick: self.slab.lite.sample_tick,
        }
    }

    /// Immutable access to the underlying L1D [`SonaHotSlab`].
    #[inline]
    #[must_use]
    pub fn slab(&self) -> &SonaHotSlab<'a, B> {
        &self.slab
    }

    /// Mutable access to the underlying L1D [`SonaHotSlab`].
    #[inline]
    pub fn slab_mut(&mut self) -> &mut SonaHotSlab<'a, B> {
        &mut self.slab
    }
}

/// Concrete RT disassembly audit root: `DefaultMathBackend` block execution.
#[inline(never)]
pub fn sona_vm_process_block_default(
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    vm.process_block(in_l, in_r, out_l, out_r)
}

/// Concrete RT disassembly audit root: `DefaultMathBackend` `16`-frame tick execution.
#[inline(never)]
pub fn sona_vm_process_tick16_default(
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    in_l: &[f32; 16],
    in_r: &[f32; 16],
    out_l: &mut [f32; 16],
    out_r: &mut [f32; 16],
) -> Result<(), VmError> {
    vm.process_tick16_const(in_l, in_r, out_l, out_r)
}

/// Concrete RT disassembly audit root: `DefaultMathBackend` single-frame step.
#[inline(never)]
pub fn sona_vm_step_frame_default(
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    vm.step_frame(in_l, in_r)
}

/// Concrete RT disassembly audit root: `DefaultMathBackend` `3 kHz` control boundary step.
#[inline(never)]
pub fn sona_vm_step_control_tick_default(vm: &mut SonaVm<'_, DefaultMathBackend>) {
    vm.step_control_tick_3khz();
}

/// Concrete RT disassembly audit root: `DefaultMathBackend` live analog drift toggle & scrub.
#[inline(never)]
pub fn sona_vm_set_live_analog_drift_enabled_default(
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    enabled: bool,
) {
    vm.set_live_analog_drift_enabled(enabled);
}

/// Concrete RT disassembly audit root: `DefaultMathBackend` telemetry saturation flag clear.
#[inline(never)]
pub fn sona_vm_clear_saturation_flags_default(vm: &mut SonaVm<'_, DefaultMathBackend>) {
    vm.clear_saturation_flags();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::{DnamBinBuildInput, encode_dnam_bin_into};
    use crate::control::{ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot};
    use crate::graph::SonaColdTail;

    #[test]
    fn test_sona_vm_no_alloc_open_ms_width_telemetry_and_safety_ceiling() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
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
        let mut buf = [0_u8; 128 + 4_608];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();

        let mut cold = [SonaColdTail::<DefaultMathBackend>::empty()];
        let mut descs = [ControlClampDescriptor::empty(); 1];
        let (mut raw, mut cl, mut norm, mut sw) = ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let mut con = [Cp1ContributorSlot::zero(); 8];
        let ctrl = ControlBusBacking::from_slices(
            &mut descs,
            &mut raw,
            &mut cl,
            &mut norm,
            &mut sw,
            &mut con,
            &[],
        )
        .unwrap();
        let backing = SonaVmBacking {
            control_bus: ctrl,
            cabinet_fir_ext: &mut [],
            tank_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
            tank_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            tank_schroeder: [&mut [], &mut [], &mut [], &mut []],
            cold_tail: &mut cold,
        };
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin(&buf[..n], Tier::Lite, 48_000.0, backing)
                .unwrap();
        assert_eq!(vm.get_latency_samples(), 0);
        vm.set_ms_width(f32::NAN);
        assert_ne!(
            vm.saturation_flags() & saturation_flags::INVALID_NONFINITE,
            0
        );
        sona_vm_clear_saturation_flags_default(&mut vm);
        assert_eq!(vm.saturation_flags(), 0);
        sona_vm_set_live_analog_drift_enabled_default(&mut vm, true);
        sona_vm_set_live_analog_drift_enabled_default(&mut vm, false);
        let (yl, yr) = sona_vm_step_frame_default(&mut vm, f32::INFINITY, 200.0).unwrap();
        assert_eq!((yl, yr), (0.0, 0.0));
        assert_ne!(
            vm.saturation_flags() & saturation_flags::INVALID_NONFINITE,
            0
        );
        assert_ne!(
            vm.saturation_flags() & saturation_flags::OUTSIDE_CERTIFIED_INPUT,
            0
        );

        let mut cold_pre = [SonaColdTail::<DefaultMathBackend>::empty()];
        let mut descs_pre = [ControlClampDescriptor::empty(); 1];
        let (mut raw_pre, mut cl_pre, mut norm_pre, mut sw_pre) =
            ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let mut con_pre = [Cp1ContributorSlot::zero(); 8];
        let ctrl_pre = ControlBusBacking::from_slices(
            &mut descs_pre,
            &mut raw_pre,
            &mut cl_pre,
            &mut norm_pre,
            &mut sw_pre,
            &mut con_pre,
            &[],
        )
        .unwrap();
        let backing_pre = SonaVmBacking {
            control_bus: ctrl_pre,
            cabinet_fir_ext: &mut [],
            tank_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
            tank_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            tank_schroeder: [&mut [], &mut [], &mut [], &mut []],
            cold_tail: &mut cold_pre,
        };
        let mut vm_pre = SonaVm::<DefaultMathBackend>::open_dnam_bin_prevalidated(
            &buf[..n],
            Tier::Lite,
            48_000.0,
            backing_pre,
        )
        .unwrap();
        let pre_out = vm_pre.step_frame(0.1, -0.1).unwrap();
        assert!(pre_out.0.is_finite() && pre_out.1.is_finite());
        #[cfg(feature = "alloc")]
        {
            let mut vm_owned = SonaVm::<DefaultMathBackend>::open_dnam_bin_prevalidated_owned(
                &buf[..n],
                Tier::Lite,
                48_000.0,
            )
            .unwrap();
            assert_eq!(pre_out, vm_owned.step_frame(0.1, -0.1).unwrap());
        }
    }
}
