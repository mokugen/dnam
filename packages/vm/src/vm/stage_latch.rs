//! Zero-allocation in-place latched control target and reset helpers on [`SonaVm`]
//! for deterministic host/solver sweeping (`#![no_std]`, `#![forbid(unsafe_code)]`).

use crate::control::STAGE_TARGET_SLOTS;
use crate::graph::{GraphBackend, latch_immediate_at_init_or_reset, step_control_tick_3khz};
use crate::types::VmError;

use super::SonaVm;

impl<'a, B: GraphBackend> SonaVm<'a, B> {
    /// Resets DSP/slew state to resting equilibrium, disables `ControlBus`, latches `stage_targets`
    /// and `z_stage` (`slot 31` forced to `0.0`), and snaps hoisted registers with `0` allocations.
    #[inline(never)]
    pub fn reset_with_latched_stage_targets(
        &mut self,
        stage_targets: &[f32; STAGE_TARGET_SLOTS],
        z_stage: &[f32; 8],
    ) {
        self.slab.cold_mut().ctrl_bus = None;
        self.reset();
        {
            let cold = self.slab.cold_mut();
            cold.latched_stage_params = *stage_targets;
            cold.latched_stage_params[STAGE_TARGET_SLOTS - 1] = 0.0;
            cold.latched_z_stage = *z_stage;
        }
        latch_immediate_at_init_or_reset(&mut self.slab);
    }

    /// Updates `latched_stage_params` (`slot 31` forced `0.0`) and `latched_z_stage` in place and
    /// steps one `3 kHz` control slew/hoist tick without resetting delay or filter states.
    #[inline(never)]
    pub fn set_latched_stage_targets_tick(
        &mut self,
        stage_targets: &[f32; STAGE_TARGET_SLOTS],
        z_stage: &[f32; 8],
    ) {
        let cold = self.slab.cold_mut();
        cold.ctrl_bus = None;
        cold.latched_stage_params = *stage_targets;
        cold.latched_stage_params[STAGE_TARGET_SLOTS - 1] = 0.0;
        cold.latched_z_stage = *z_stage;
        step_control_tick_3khz(&mut self.slab);
    }

    /// Resets DSP/slew and `ControlBus` state, applies `ctrl_values` (`(ctrl_idx, phys_val)`), and
    /// immediately snaps all `ControlBus` / slew / hoisted registers with `0` heap allocations.
    pub fn reset_and_set_controls_immediate(
        &mut self,
        ctrl_values: &[(usize, f32)],
    ) -> Result<(), VmError> {
        self.reset();
        self.set_controls_batch(ctrl_values)?;
        latch_immediate_at_init_or_reset(&mut self.slab);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{f32_to_fp16, pack_wf_mix_and_rect_sign_fp16};
    use zerocopy::IntoBytes;
    use zerocopy::byteorder::little_endian::{F32, U16};

    use super::*;
    use crate::bin::{DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, encode_dnam_bin_into};
    use crate::control::{ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot};
    use crate::graph::SonaColdTail;
    use crate::math::DefaultMathBackend;
    use crate::types::Tier;
    use crate::vm::SonaVmBacking;

    fn build_stage_latch_test_bin() -> ([u8; 128 + 4_608 + 64], usize) {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.core_quads[0].svf_k_fp16[0] = U16::new(f32_to_fp16(0.85));
        a0.core_quads[0].adaa_drive_fp16[0] = U16::new(f32_to_fp16(1.25));
        a0.core_quads[0].wavefolder_mix_fp16[0] =
            U16::new(pack_wf_mix_and_rect_sign_fp16(0.35, 1.0));
        a0.control_bus_lite.w_ctrl[0][0] = F32::new(0.45);
        a0.control_bus_lite.w_ctrl[1][0] = F32::new(-0.15);
        a0.control_bus_lite.w_ctrl[2][0] = F32::new(0.60);

        a0.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(1);
        a0.control_bus_lite.cp1_dir.entries[0] = [0, 0x01, 0xFF];
        a0.control_bus_lite.cp1_base[..4].copy_from_slice(F32::new(1.0).as_bytes());
        a0.control_bus_lite.cp1_base[32..36].copy_from_slice(F32::new(1.0).as_bytes());
        a0.control_bus_lite.cp1_base[36..40].copy_from_slice(F32::new(1.0).as_bytes());
        let raw_desc = ControlClampDescriptor::range(-1.0, 1.0, 0.0, 1, [0.0; 3]).to_wire();

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
            controls: &[raw_desc],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0_u8; 128 + 4_608 + 64];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();
        (buf, n)
    }

    #[test]
    fn test_stage_latch_zero_alloc_equivalence_and_tick_stepping() {
        let (buf, n) = build_stage_latch_test_bin();

        let mut cold_ref = [SonaColdTail::<DefaultMathBackend>::empty()];
        let mut descs_ref = [ControlClampDescriptor::empty(); 1];
        let (mut raw_ref, mut cl_ref, mut norm_ref, mut sw_ref) =
            ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let mut con_ref = [Cp1ContributorSlot::zero(); 8];
        let ctrl_ref = ControlBusBacking::from_slices(
            &mut descs_ref,
            &mut raw_ref,
            &mut cl_ref,
            &mut norm_ref,
            &mut sw_ref,
            &mut con_ref,
            &[],
        )
        .unwrap();
        let mut vm_ref = SonaVm::<DefaultMathBackend>::open_dnam_bin(
            &buf[..n],
            Tier::Lite,
            48_000.0,
            SonaVmBacking {
                control_bus: ctrl_ref,
                convolver_fir_ext: &mut [],
                transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
                transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
                transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
                cold_tail: &mut cold_ref,
            },
        )
        .unwrap();

        let mut cold_reuse = [SonaColdTail::<DefaultMathBackend>::empty()];
        let mut descs_reuse = [ControlClampDescriptor::empty(); 1];
        let (mut raw_reuse, mut cl_reuse, mut norm_reuse, mut sw_reuse) =
            ([0.0; 1], [0.0; 1], [0.0; 1], [0; 1]);
        let mut con_reuse = [Cp1ContributorSlot::zero(); 8];
        let ctrl_reuse = ControlBusBacking::from_slices(
            &mut descs_reuse,
            &mut raw_reuse,
            &mut cl_reuse,
            &mut norm_reuse,
            &mut sw_reuse,
            &mut con_reuse,
            &[],
        )
        .unwrap();
        let mut vm_reuse = SonaVm::<DefaultMathBackend>::open_dnam_bin(
            &buf[..n],
            Tier::Lite,
            48_000.0,
            SonaVmBacking {
                control_bus: ctrl_reuse,
                convolver_fir_ext: &mut [],
                transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
                transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
                transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
                cold_tail: &mut cold_reuse,
            },
        )
        .unwrap();

        // 1. `reset_and_set_controls_immediate` matches fresh `open_dnam_bin` 0-ULP
        assert!(
            vm_reuse
                .reset_and_set_controls_immediate(&[(99, 0.5)])
                .is_err()
        );
        let _ = vm_reuse.step_frame(0.3, -0.2).unwrap();
        vm_reuse
            .reset_and_set_controls_immediate(&[(0, 0.75)])
            .unwrap();
        vm_ref.set_control_physical(0, 0.75).unwrap();
        latch_immediate_at_init_or_reset(&mut vm_ref.slab);
        for t in 0..64 {
            let x = 0.25 * ((t as f32) * 0.17).sin();
            let y_ref = vm_ref.step_frame(x, -x).unwrap();
            let y_reuse = vm_reuse.step_frame(x, -x).unwrap();
            assert_eq!(y_ref.0.to_bits(), y_reuse.0.to_bits());
            assert_eq!(y_ref.1.to_bits(), y_reuse.1.to_bits());
        }

        // 2. `reset_with_latched_stage_targets` & continuous `set_latched_stage_targets_tick`
        let mut targets = [0.0_f32; STAGE_TARGET_SLOTS];
        targets[0] = 0.35;
        targets[2] = 0.55;
        targets[STAGE_TARGET_SLOTS - 1] = 123.0; // must be forced to 0.0
        let z_stage = [0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        vm_ref.slab.cold_mut().ctrl_bus = None;
        vm_ref.reset();
        vm_ref.slab.cold_mut().latched_stage_params = targets;
        vm_ref.slab.cold_mut().latched_stage_params[STAGE_TARGET_SLOTS - 1] = 0.0;
        vm_ref.slab.cold_mut().latched_z_stage = z_stage;
        latch_immediate_at_init_or_reset(&mut vm_ref.slab);

        vm_reuse.reset_with_latched_stage_targets(&targets, &z_stage);
        assert_eq!(
            vm_reuse.slab.cold().latched_stage_params[STAGE_TARGET_SLOTS - 1],
            0.0
        );
        for t in 0..32 {
            let x = 0.20 * ((t as f32) * 0.11).cos();
            let y_ref = vm_ref.step_frame(x, x).unwrap();
            let y_reuse = vm_reuse.step_frame(x, x).unwrap();
            assert_eq!(y_ref.0.to_bits(), y_reuse.0.to_bits());
            assert_eq!(y_ref.1.to_bits(), y_reuse.1.to_bits());
        }

        targets[2] = -0.25;
        vm_reuse.set_latched_stage_targets_tick(&targets, &z_stage);
        let y_tick = vm_reuse.step_frame(0.15, -0.15).unwrap();
        assert!(y_tick.0.is_finite() && y_tick.1.is_finite());
        assert_eq!(
            vm_reuse.slab.cold().latched_stage_params[STAGE_TARGET_SLOTS - 1],
            0.0
        );
    }
}
