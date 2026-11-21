//! Heap-allocated runtime-dispatched [`DynamicSonaVm`] over all 8 SIMD ISA targets
//! (`feature = "alloc"`, `#![no_std]`, `#![forbid(unsafe_code)]`, zero `macro_rules!`).

#![forbid(unsafe_code)]

mod controls;

use alloc::boxed::Box;
use core::fmt;

use crate::bin::DnamBinView;
#[cfg(target_arch = "aarch64")]
use crate::math::aarch64::{ArmSmeBackend, ArmSve2Backend, NeonBackend};
#[cfg(target_arch = "wasm32")]
use crate::math::wasm::WasmSimd128Backend;
#[cfg(target_arch = "x86_64")]
use crate::math::x86::{X86Avx2FmaBackend, X86Sse2Backend, X86Sse42Backend};
use crate::math::{BackendKind, ScalarBackend, detect_best_backend, warm_simd_feature_caches};
use crate::types::{Tier, VmError};

use super::SonaVm;

/// Runtime-dispatched owned `SONA v3.0` VM selecting the host's highest safe SIMD backend.
pub enum DynamicSonaVm {
    Scalar(Box<SonaVm<'static, ScalarBackend>>),
    #[cfg(target_arch = "aarch64")]
    ArmNeon(Box<SonaVm<'static, NeonBackend>>),
    #[cfg(target_arch = "aarch64")]
    ArmSve2(Box<SonaVm<'static, ArmSve2Backend>>),
    #[cfg(target_arch = "aarch64")]
    ArmSme(Box<SonaVm<'static, ArmSmeBackend>>),
    #[cfg(target_arch = "x86_64")]
    X86Sse2(Box<SonaVm<'static, X86Sse2Backend>>),
    #[cfg(target_arch = "x86_64")]
    X86Sse42(Box<SonaVm<'static, X86Sse42Backend>>),
    #[cfg(target_arch = "x86_64")]
    X86Avx2Fma(Box<SonaVm<'static, X86Avx2FmaBackend>>),
    #[cfg(target_arch = "wasm32")]
    WasmSimd128(Box<SonaVm<'static, WasmSimd128Backend>>),
}

impl fmt::Debug for DynamicSonaVm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicSonaVm")
            .field("backend", &self.backend_kind())
            .finish()
    }
}

impl DynamicSonaVm {
    /// Parses, verifies, and opens a `.dnam-bin` payload on the optimal detected SIMD backend.
    pub fn open_dnam_bin(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
    ) -> Result<Self, VmError> {
        Self::open_with_backend(bytes, active_tier, host_rate_hz, detect_best_backend())
    }

    /// Binds a pre-authenticated `.dnam-bin` buffer in `< 50 ns` and opens on optimal detected SIMD.
    pub fn open_dnam_bin_prevalidated(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
    ) -> Result<Self, VmError> {
        Self::open_with_backend_prevalidated(
            bytes,
            active_tier,
            host_rate_hz,
            detect_best_backend(),
        )
    }

    /// Parses, verifies, and opens a `.dnam-bin` payload targeting a requested [`BackendKind`].
    pub fn open_with_backend(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
        backend: BackendKind,
    ) -> Result<Self, VmError> {
        warm_simd_feature_caches();
        let view = DnamBinView::parse(bytes)?;
        Self::from_view_with_backend(&view, active_tier, host_rate_hz, false, backend)
    }

    /// Binds a pre-authenticated `.dnam-bin` buffer targeting a requested [`BackendKind`].
    pub fn open_with_backend_prevalidated(
        bytes: &[u8],
        active_tier: Tier,
        host_rate_hz: f32,
        backend: BackendKind,
    ) -> Result<Self, VmError> {
        warm_simd_feature_caches();
        let view = DnamBinView::bind_prevalidated(bytes)?;
        Self::from_view_with_backend(&view, active_tier, host_rate_hz, false, backend)
    }

    /// Opens a [`DynamicSonaVm`] from a validated [`DnamBinView`] on the optimal SIMD backend.
    pub fn from_view(
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
    ) -> Result<Self, VmError> {
        Self::from_view_with_backend(
            view,
            active_tier,
            host_rate_hz,
            offline_bounce,
            detect_best_backend(),
        )
    }

    /// Opens a [`DynamicSonaVm`] from a validated [`DnamBinView`] for a requested [`BackendKind`].
    pub fn from_view_with_backend(
        view: &DnamBinView<'_>,
        active_tier: Tier,
        host_rate_hz: f32,
        offline_bounce: bool,
        backend: BackendKind,
    ) -> Result<Self, VmError> {
        warm_simd_feature_caches();
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmNeon | BackendKind::Neon => Ok(Self::ArmNeon(Box::new(
                SonaVm::<NeonBackend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSve2 => Ok(Self::ArmSve2(Box::new(
                SonaVm::<ArmSve2Backend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Ok(Self::ArmSme(Box::new(
                SonaVm::<ArmSmeBackend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse2 => Ok(Self::X86Sse2(Box::new(
                SonaVm::<X86Sse2Backend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse42 => Ok(Self::X86Sse42(Box::new(
                SonaVm::<X86Sse42Backend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Avx2Fma | BackendKind::X86Simd => Ok(Self::X86Avx2Fma(Box::new(
                SonaVm::<X86Avx2FmaBackend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => Ok(Self::WasmSimd128(Box::new(SonaVm::<
                WasmSimd128Backend,
            >::from_view_owned(
                view,
                active_tier,
                host_rate_hz,
                offline_bounce,
            )?))),
            _ => Ok(Self::Scalar(Box::new(
                SonaVm::<ScalarBackend>::from_view_owned(
                    view,
                    active_tier,
                    host_rate_hz,
                    offline_bounce,
                )?,
            ))),
        }
    }

    /// Returns the active [`BackendKind`] bound inside this instance.
    #[inline]
    #[must_use]
    pub const fn backend_kind(&self) -> BackendKind {
        match self {
            Self::Scalar(_) => BackendKind::Scalar,
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(_) => BackendKind::ArmNeon,
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(_) => BackendKind::ArmSve2,
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(_) => BackendKind::ArmSme,
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(_) => BackendKind::X86Sse2,
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(_) => BackendKind::X86Sse42,
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(_) => BackendKind::X86Avx2Fma,
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(_) => BackendKind::WasmSimd128,
        }
    }

    /// Processes a stereo audio block through the selected SIMD backend (`0` allocations).
    #[inline(never)]
    pub fn process_block(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.process_block(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.process_block(in_l, in_r, out_l, out_r),
        }
    }

    /// Processes a fixed `16`-frame slice through the selected SIMD backend.
    #[inline(never)]
    pub fn process_tick16_const(
        &mut self,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32; 16],
        out_r: &mut [f32; 16],
    ) -> Result<(), VmError> {
        match self {
            Self::Scalar(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.process_tick16_const(in_l, in_r, out_l, out_r),
        }
    }

    /// Steps one stereo host frame.
    #[inline(never)]
    pub fn step_frame(&mut self, in_l: f32, in_r: f32) -> Result<(f32, f32), VmError> {
        match self {
            Self::Scalar(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmNeon(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(vm) => vm.step_frame(in_l, in_r),
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(vm) => vm.step_frame(in_l, in_r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::{DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, encode_dnam_bin_into};

    #[test]
    fn test_dynamic_sona_vm_scalar_vs_best_backend_zero_ulp_parity() {
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
        let mut vm_best = DynamicSonaVm::open_dnam_bin(&buf[..n], Tier::Lite, 48_000.0).unwrap();
        let mut vm_pre =
            DynamicSonaVm::open_dnam_bin_prevalidated(&buf[..n], Tier::Lite, 48_000.0).unwrap();
        let view = DnamBinView::bind_prevalidated(&buf[..n]).unwrap();
        let mut vm_view = DynamicSonaVm::from_view(&view, Tier::Lite, 48_000.0, false).unwrap();
        let mut vm_scalar =
            DynamicSonaVm::open_with_backend(&buf[..n], Tier::Lite, 48_000.0, BackendKind::Scalar)
                .unwrap();
        let (in_l, in_r) = ([0.25_f32; 16], [-0.25_f32; 16]);
        let (mut bl, mut br) = ([0.0_f32; 16], [0.0_f32; 16]);
        let (mut pl, mut pr) = ([0.0_f32; 16], [0.0_f32; 16]);
        let (mut vl, mut vr) = ([0.0_f32; 16], [0.0_f32; 16]);
        let (mut sl, mut sr) = ([0.0_f32; 16], [0.0_f32; 16]);
        vm_best
            .process_tick16_const(&in_l, &in_r, &mut bl, &mut br)
            .unwrap();
        vm_pre
            .process_tick16_const(&in_l, &in_r, &mut pl, &mut pr)
            .unwrap();
        vm_view
            .process_tick16_const(&in_l, &in_r, &mut vl, &mut vr)
            .unwrap();
        vm_scalar
            .process_tick16_const(&in_l, &in_r, &mut sl, &mut sr)
            .unwrap();
        assert_eq!(vm_scalar.backend_kind(), BackendKind::Scalar);
        for i in 0..16 {
            assert_eq!(bl[i].to_bits(), sl[i].to_bits());
            assert_eq!(br[i].to_bits(), sr[i].to_bits());
            assert_eq!(pl[i].to_bits(), bl[i].to_bits());
            assert_eq!(pr[i].to_bits(), br[i].to_bits());
            assert_eq!(vl[i].to_bits(), bl[i].to_bits());
            assert_eq!(vr[i].to_bits(), br[i].to_bits());
        }
    }

    #[test]
    fn test_dynamic_sona_vm_runtime_controls_and_backends() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let ctrl =
            [crate::control::ControlClampDescriptor::range(0.0, 1.0, 0.5, 2, [0.0; 3]).to_wire()];
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
            controls: &ctrl,
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

        let kinds: &[BackendKind] = &[
            BackendKind::Scalar,
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmNeon,
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSve2,
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme,
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse2,
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse42,
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Avx2Fma,
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128,
        ];
        for &kind in kinds {
            let mut vm = DynamicSonaVm::open_with_backend_prevalidated(
                &buf[..n],
                Tier::Lite,
                48_000.0,
                kind,
            )
            .unwrap();
            vm.set_live_analog_drift_enabled(true);
            assert_eq!(vm.get_latency_samples(), 0);
            assert_eq!(vm.saturation_flags(), 0);
            assert_eq!(vm.apply_tempo_bpm(f32::NAN), 0);
            assert_ne!(vm.saturation_flags(), 0);
            vm.clear_saturation_flags();
            assert_eq!(vm.saturation_flags(), 0);
            vm.set_quad_write_enable(true);
            assert!(vm.telemetry().host_write_lock);
            vm.set_op_t_write_enable(false);
            assert!(!vm.telemetry().host_write_lock);
            vm.set_control_physical(0, 0.8).unwrap();
            vm.set_controls_batch(&[(0, 0.4)]).unwrap();
            vm.step_control_tick_3khz();
            assert_eq!(vm.apply_tempo_bpm(120.0), 0);
            let _ = vm.step_frame(0.1, -0.1).unwrap();
            vm.set_offline_bounce(true).unwrap();
            assert_eq!(vm.get_latency_samples(), 12);
            vm.set_offline_bounce(false).unwrap();
            vm.retime_sample_rate(96_000.0).unwrap();
            let snap = vm.telemetry();
            assert_eq!(snap.active_tier, Tier::Lite);
            vm.set_live_analog_drift_enabled(false);
            vm.reset();
        }
    }
}
