//! Deterministic `#![no_std]` + `!alloc`-capable `SONA v3.0` Virtual Machine (`SonaVm`),
//! `!alloc` [`SonaVmBacking`], optional `alloc` [`SonaInstance`] / [`dynamic::DynamicSonaVm`],
//! telemetry, and concrete RT disassembly audit roots (`#![forbid(unsafe_code)]`).

#[cfg(feature = "alloc")]
pub mod dynamic;
mod stage_latch;

use dnam_kernels::QUAD_FLAG_HOST_WRITE_LOCK;

use crate::bin::DnamBinView;
use crate::graph::slab::{
    SonaHotSlab, SonaSlabBacking, check_reload_slab_capacity, hydrate_slab_from_dnam_bin,
    reload_slab_from_dnam_bin_in_place, reset_to_dc_equilibrium, retime_sample_rate,
    set_offline_bounce, set_op_t_write_enable, set_quad_write_enable, switch_tier,
};
#[cfg(feature = "alloc")]
use crate::graph::slab::{VmReloadHeadroom, hydrate_slab_owned_with_headroom};
use crate::graph::{Bounce4xOversampler, GraphBackend};
use crate::math::{DefaultMathBackend, saturation_flags};
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
    pub host_write_lock: bool,
    pub latency_samples: usize,
    pub sample_tick: u32,
}

/// Deterministic pure `2x2` `SONA v3.0` Virtual Machine wrapping [`SonaHotSlab`]
/// (`#![no_std]`, `#![forbid(unsafe_code)]`).
#[derive(Debug)]
pub struct SonaVm<'a, B: GraphBackend = DefaultMathBackend> {
    slab: SonaHotSlab<'a, B>,
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
        Ok(Self { slab })
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
        Self::from_view_owned_with_headroom(
            view,
            active_tier,
            host_rate_hz,
            offline_bounce,
            VmReloadHeadroom::zero(),
        )
    }

    /// Parses, CRC-verifies, and heap-hydrates a `.dnam-bin` payload with pre-allocated reload
    /// capacity headroom (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn open_dnam_bin_with_headroom_owned(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
        headroom: VmReloadHeadroom,
    ) -> Result<SonaVm<'static, B>, VmError> {
        let view = DnamBinView::parse(bytes)?;
        SonaVm::<B>::from_view_owned_with_headroom(
            &view,
            active_tier,
            host_rate_hz,
            false,
            headroom,
        )
    }

    /// Heap-hydrates a [`SonaVm`] from `view` with explicit [`VmReloadHeadroom`] (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn from_view_owned_with_headroom(
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
        headroom: VmReloadHeadroom,
    ) -> Result<SonaVm<'static, B>, VmError> {
        let slab = hydrate_slab_owned_with_headroom(
            view,
            active_tier,
            host_rate_hz,
            offline_bounce,
            headroom,
        )?;
        Ok(SonaVm { slab })
    }

    /// Validates backing capacities and wire parameters required for an in-place reload without
    /// mutating any VM or slab state.
    #[inline]
    pub fn check_reload_capacity(
        &self,
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
    ) -> Result<(), VmError> {
        check_reload_slab_capacity(&self.slab, view, active_tier, host_rate_hz, offline_bounce)
    }

    /// Returns the latest frame's pre-spatial core `[L, R]` vs transport wet `[L, R]` partition
    /// (`core_lr[ch] + wet_lr[ch] ~= out[ch]`).
    #[inline]
    #[must_use]
    pub fn last_wdw_partition(&self) -> ([f32; 2], [f32; 2]) {
        let cold = self.slab.cold();
        (cold.last_core_lr, cold.last_transport_wet_lr)
    }

    /// Reloads a new validated `.dnam-bin` model in-place with zero heap allocations and zero
    /// deallocations (works on both `!alloc` borrowed slabs and owned headroom slabs).
    #[inline(never)]
    pub fn reload_from_view_in_place(
        &mut self,
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
    ) -> Result<(), VmError> {
        reload_slab_from_dnam_bin_in_place(
            &mut self.slab,
            view,
            active_tier,
            host_rate_hz,
            offline_bounce,
        )
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
        B::process_block_slab(&mut self.slab, in_l, in_r, out_l, out_r)
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
        B::process_tick16_slab(&mut self.slab, in_l, in_r, out_l, out_r)
    }

    /// Steps one stereo host sample (`1x` live or `4x` offline bounce) under `FTZ`/`DAZ` guard.
    #[inline(never)]
    pub fn step_frame(&mut self, in_l: f32, in_r: f32) -> Result<(f32, f32), VmError> {
        let _ftz = FtzDazGuard::new();
        B::step_frame_slab(&mut self.slab, in_l, in_r)
    }

    /// Executes one `3 kHz` boundary control tick (`Kautz`, `Transport`, and core hoisting).
    #[inline(never)]
    pub fn step_control_tick_3khz(&mut self) {
        B::step_control_tick_slab(&mut self.slab);
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

    /// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across all active `TransportQuad`s.
    #[inline(never)]
    pub fn set_quad_write_enable(&mut self, locked: bool) {
        set_quad_write_enable(&mut self.slab, locked);
    }

    /// Alias matching [`SonaHotSlab`] naming (`set_op_t_write_enable`).
    #[inline(never)]
    pub fn set_op_t_write_enable(&mut self, locked: bool) {
        set_op_t_write_enable(&mut self.slab, locked);
    }

    /// Switches active tier (`0` RT allocs, `0` core reset, `48`-frame output crossfade).
    pub fn switch_tier(&mut self, to_tier: Tier) -> Result<(), VmError> {
        switch_tier(&mut self.slab, to_tier)
    }

    /// Re-times all internal ZOH filters, `SensoryQuad`, Hilbert H1, `ModalConvolver` FIR,
    /// Kautz, `Transport`, and `ControlBus`.
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
        let mut host_lock = false;
        for q in 0..cold.active_transport_quads {
            if (self.slab.transport_quad_hot(q).flags & QUAD_FLAG_HOST_WRITE_LOCK) != 0 {
                host_lock = true;
            }
        }
        SonaTelemetrySnapshot {
            saturation_flags: cold.saturation_flags,
            active_tier: self.slab.lite.tier,
            max_present_tier: cold.max_present_tier,
            retained_tier: cold.retained_tier,
            os2x_active: self.slab.os2x(),
            offline_bounce: cold.offline_bounce,
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
    use crate::bin::{DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, encode_dnam_bin_into};
    use crate::control::{ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot};
    use crate::graph::SonaColdTail;

    #[test]
    fn test_sona_vm_no_alloc_open_telemetry_and_safety_ceiling() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
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
            convolver_fir_ext: &mut [],
            transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
            transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
            cold_tail: &mut cold,
        };
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin(&buf[..n], Tier::Lite, 48_000.0, backing)
                .unwrap();
        assert_eq!(vm.get_latency_samples(), 0);
        assert_eq!(vm.apply_tempo_bpm(f32::NAN), 0);
        assert_ne!(
            vm.saturation_flags() & saturation_flags::INVALID_NONFINITE,
            0
        );
        sona_vm_clear_saturation_flags_default(&mut vm);
        assert_eq!(vm.saturation_flags(), 0);
        vm.set_quad_write_enable(true);
        assert!(vm.telemetry().host_write_lock);
        vm.reset();
        assert!(vm.telemetry().host_write_lock);
        vm.set_op_t_write_enable(false);
        assert!(!vm.telemetry().host_write_lock);
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
            convolver_fir_ext: &mut [],
            transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
            transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
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

    fn build_wdw_test_lite_bin(
        header_flags: u16,
        controls: &[crate::bin::ControlClampDescriptorWire],
    ) -> ([u8; 128 + 4_608 + 512], usize) {
        use crate::math::f32_to_fp16;
        use zerocopy::byteorder::little_endian::{F32, U16};

        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.core_quads[0].in_mid_weight_fp16 =
            [0.6, 0.3, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        a0.core_quads[0].in_side_weight_fp16 =
            [0.25, -0.15, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        a0.core_quads[0].svf_g = [0.28; 4].map(F32::new);
        a0.core_quads[0].svf_k_fp16 = [U16::new(f32_to_fp16(0.707)); 4];
        a0.core_quads[0].adaa_drive_fp16 = [U16::new(f32_to_fp16(1.2)); 4];
        a0.core_quads[0].tone_lp_fp16 = [U16::new(f32_to_fp16(0.4)); 4];
        a0.core_quads[0].tone_dir_fp16 = [U16::new(f32_to_fp16(0.8)); 4];
        a0.core_quads[0].lru_alpha_rel = [0.2; 4].map(F32::new);
        a0.core_quads[0].lru_alpha_atk = [0.9; 4].map(F32::new);
        a0.core_quads[0].lru_kappa_fp16 = [U16::new(f32_to_fp16(0.25)); 4];
        a0.conv_fir_lite.fir_l[0] = F32::new(0.75);
        a0.conv_fir_lite.fir_r[0] = F32::new(0.65);
        a0.conv_fir_lite.fir_l[1] = F32::new(0.15);
        a0.conv_fir_lite.fir_r[1] = F32::new(-0.1);
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.8), F32::new(0.7)];
        a0.linear_readout_lite.w_ro_fp16[0] = U16::new(f32_to_fp16(0.4));
        a0.linear_readout_lite.w_ro_fp16[1] = U16::new(f32_to_fp16(0.3));
        a0.linear_readout_lite.w_ro_fp16[6] = U16::new(f32_to_fp16(0.5));
        a0.linear_readout_lite.w_ro_fp16[7] = U16::new(f32_to_fp16(0.45));
        a0.transport_lite.delay_samples = [8.0, 11.0, 14.0, 19.0].map(F32::new);
        a0.transport_lite.g_loop = F32::new(0.45);
        a0.transport_lite.input_gains_fp16 =
            [0.5, 0.35, -0.25, 0.3].map(|v| U16::new(f32_to_fp16(v)));
        a0.transport_lite.wet_tap_l_fp16 =
            [0.35, -0.2, 0.25, 0.15].map(|v| U16::new(f32_to_fp16(v)));
        a0.transport_lite.wet_tap_r_fp16 =
            [-0.2, 0.4, 0.15, 0.25].map(|v| U16::new(f32_to_fp16(v)));
        a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.75));
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [7; 8],
            controls,
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0_u8; 128 + 4_608 + 512];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();
        (buf, n)
    }

    #[test]
    fn test_sona_vm_last_wdw_partition_1x_mono_and_4x_bounce() {
        let (buf_st, n_st) = build_wdw_test_lite_bin(DNBN_FLAG_STEREO_OUT, &[]);
        let (buf_mo, n_mo) = build_wdw_test_lite_bin(0, &[]);

        // 1x stereo + process_block parity + reset.
        let mut cold = [SonaColdTail::<DefaultMathBackend>::empty()];
        let mut descs = [ControlClampDescriptor::empty(); 1];
        let (mut raw, mut cl, mut norm, mut sw) = ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let mut con = [Cp1ContributorSlot::zero(); 8];
        let mut fir_ext = [0.0_f32; crate::graph::EXT_FIR_STORAGE_F32_LEN];
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
            convolver_fir_ext: &mut fir_ext,
            transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
            transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
            cold_tail: &mut cold,
        };
        let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin(
            &buf_st[..n_st],
            Tier::Lite,
            48_000.0,
            backing,
        )
        .unwrap();
        assert_eq!(vm.last_wdw_partition(), ([0.0; 2], [0.0; 2]));

        let (mut max_core, mut max_wet) = (0.0_f32, 0.0_f32);
        for i in 0..40 {
            let xl = 0.4 * ((i as f32) * 0.21).sin();
            let xr = -0.3 * ((i as f32) * 0.17).cos();
            let (ol, or) = vm.step_frame(xl, xr).unwrap();
            let (core_lr, wet_lr) = vm.last_wdw_partition();
            max_core = max_core.max(core_lr[0].abs()).max(core_lr[1].abs());
            max_wet = max_wet.max(wet_lr[0].abs()).max(wet_lr[1].abs());
            assert!(((core_lr[0] + wet_lr[0]) - ol).abs() < 1.0e-5);
            assert!(((core_lr[1] + wet_lr[1]) - or).abs() < 1.0e-5);
        }
        assert!(max_core > 1.0e-2 && max_wet > 1.0e-3);

        let in_l = [0.25_f32; 16];
        let in_r = [-0.2_f32; 16];
        let (mut out_l, mut out_r) = ([0.0_f32; 16], [0.0_f32; 16]);
        vm.process_block(&in_l, &in_r, &mut out_l, &mut out_r)
            .unwrap();
        let (blk_core, blk_wet) = vm.last_wdw_partition();
        assert!(((blk_core[0] + blk_wet[0]) - out_l[15]).abs() < 1.0e-5);
        assert!(((blk_core[1] + blk_wet[1]) - out_r[15]).abs() < 1.0e-5);

        vm.reset();
        assert_eq!(vm.last_wdw_partition(), ([0.0; 2], [0.0; 2]));

        // 4x offline bounce steady-state partition consistency.
        vm.set_offline_bounce(true).unwrap();
        let (mut last_out_l, mut last_out_r) = (0.0_f32, 0.0_f32);
        for _ in 0..96 {
            (last_out_l, last_out_r) = vm.step_frame(0.4, -0.25).unwrap();
        }
        let (bnc_core, bnc_wet) = vm.last_wdw_partition();
        assert!(bnc_core[0].abs() > 1.0e-2 && bnc_wet[0].abs() > 1.0e-3);
        assert!(((bnc_core[0] + bnc_wet[0]) - last_out_l).abs() < 1.0e-5);
        assert!(((bnc_core[1] + bnc_wet[1]) - last_out_r).abs() < 1.0e-5);

        // mono_out = true folding parity.
        let view_mo = DnamBinView::parse(&buf_mo[..n_mo]).unwrap();
        vm.reload_from_view_in_place(&view_mo, Tier::Lite, 48_000.0, false)
            .unwrap();
        let (mut mo_max_core, mut mo_max_wet) = (0.0_f32, 0.0_f32);
        for i in 0..36 {
            let (ol, or) = vm
                .step_frame(0.45, -0.15 * ((i as f32) * 0.19).cos())
                .unwrap();
            let (core_lr, wet_lr) = vm.last_wdw_partition();
            assert_eq!(ol.to_bits(), or.to_bits());
            assert_eq!(core_lr[0].to_bits(), core_lr[1].to_bits());
            assert_eq!(wet_lr[0].to_bits(), wet_lr[1].to_bits());
            mo_max_core = mo_max_core.max(core_lr[0].abs());
            mo_max_wet = mo_max_wet.max(wet_lr[0].abs());
            assert!(((core_lr[0] + wet_lr[0]) - ol).abs() < 1.0e-5);
        }
        assert!(mo_max_core > 1.0e-2 && mo_max_wet > 1.0e-3);
    }

    #[test]
    fn test_sona_vm_check_reload_capacity_fail_closed_and_non_mutating() {
        let (buf_1, n_1) = build_wdw_test_lite_bin(DNBN_FLAG_STEREO_OUT, &[]);
        let c2 = [
            ControlClampDescriptor::range(0.0, 1.0, 0.5, 1, [0.0; 3]).to_wire(),
            ControlClampDescriptor::range(0.0, 1.0, 0.25, 1, [0.0; 3]).to_wire(),
        ];
        let (buf_2, n_2) = build_wdw_test_lite_bin(DNBN_FLAG_STEREO_OUT, &c2);
        let view_1 = DnamBinView::parse(&buf_1[..n_1]).unwrap();
        let view_2 = DnamBinView::parse(&buf_2[..n_2]).unwrap();

        let (mut cold_a, mut cold_b) = (
            [SonaColdTail::<DefaultMathBackend>::empty()],
            [SonaColdTail::<DefaultMathBackend>::empty()],
        );
        let (mut d_a, mut d_b) = (
            [ControlClampDescriptor::empty(); 1],
            [ControlClampDescriptor::empty(); 1],
        );
        let (mut r_a, mut c_a, mut n_a, mut s_a) = ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let (mut r_b, mut c_b, mut n_b, mut s_b) = ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let (mut con_a, mut con_b) = (
            [Cp1ContributorSlot::zero(); 8],
            [Cp1ContributorSlot::zero(); 8],
        );
        let ctrl_a = ControlBusBacking::from_slices(
            &mut d_a,
            &mut r_a,
            &mut c_a,
            &mut n_a,
            &mut s_a,
            &mut con_a,
            &[],
        )
        .unwrap();
        let ctrl_b = ControlBusBacking::from_slices(
            &mut d_b,
            &mut r_b,
            &mut c_b,
            &mut n_b,
            &mut s_b,
            &mut con_b,
            &[],
        )
        .unwrap();
        let mut vm_a = SonaVm::<DefaultMathBackend>::open_dnam_bin(
            &buf_1[..n_1],
            Tier::Lite,
            48_000.0,
            SonaVmBacking {
                control_bus: ctrl_a,
                convolver_fir_ext: &mut [],
                transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
                transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
                transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
                cold_tail: &mut cold_a,
            },
        )
        .unwrap();
        let mut vm_b = SonaVm::<DefaultMathBackend>::open_dnam_bin(
            &buf_1[..n_1],
            Tier::Lite,
            48_000.0,
            SonaVmBacking {
                control_bus: ctrl_b,
                convolver_fir_ext: &mut [],
                transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
                transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
                transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
                cold_tail: &mut cold_b,
            },
        )
        .unwrap();

        for i in 0..8 {
            let x = 0.2 * ((i as f32) + 1.0) * 0.1;
            let _ = vm_a.step_frame(x, -x).unwrap();
            let _ = vm_b.step_frame(x, -x).unwrap();
        }
        let snap_before = vm_a.telemetry();
        let part_before = vm_a.last_wdw_partition();

        assert!(
            vm_a.check_reload_capacity(&view_1, Tier::Lite, 48_000.0, false)
                .is_ok()
        );
        assert_eq!(
            vm_a.check_reload_capacity(&view_2, Tier::Lite, 48_000.0, false),
            Err(VmError::InsufficientBackingCapacity)
        );
        assert_eq!(
            vm_a.check_reload_capacity(&view_1, Tier::Lite, 96_000.0, false),
            Err(VmError::InsufficientBackingCapacity)
        );
        assert_eq!(
            vm_a.check_reload_capacity(&view_1, Tier::Ultra, 48_000.0, false),
            Err(VmError::InvalidTier)
        );
        assert_eq!(
            vm_a.check_reload_capacity(&view_1, Tier::Lite, 192_000.0, true),
            Err(VmError::InvalidSampleRate)
        );
        assert_eq!(vm_a.telemetry(), snap_before);
        assert_eq!(vm_a.last_wdw_partition(), part_before);

        for i in 0..8 {
            let x = 0.15 * ((i as f32) + 1.0) * 0.1;
            let (al, ar) = vm_a.step_frame(x, -0.5 * x).unwrap();
            let (bl, br) = vm_b.step_frame(x, -0.5 * x).unwrap();
            assert_eq!((al.to_bits(), ar.to_bits()), (bl.to_bits(), br.to_bits()));
        }
    }
}
