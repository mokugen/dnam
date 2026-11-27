//! Prefix-nested L1D hot execution slab (`4,544..14,592 B`), `!alloc` hydration, `3 kHz` control
//! latching, and `0-ULP` chunk-split invariant micro-chunking (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub mod ctrl_latch;
pub mod hydrate;
pub mod layout;
pub mod microchunk;

#[allow(unused_imports)]
pub(crate) use ctrl_latch::step_control_tick_3khz_inline;
pub use ctrl_latch::{
    hoist_core_quads_and_readouts, latch_immediate_at_init_or_reset, step_control_tick_3khz,
};
#[cfg(feature = "alloc")]
pub use hydrate::{SonaSlabOwnedBacking, hydrate_slab_owned, hydrate_slab_owned_with_headroom};
pub use hydrate::{
    VmReloadHeadroom, check_reload_slab_capacity, hydrate_slab_from_dnam_bin,
    reload_slab_from_dnam_bin_in_place, reset_to_dc_equilibrium, retime_sample_rate,
    set_offline_bounce, set_op_t_write_enable, set_quad_write_enable, switch_tier,
};
pub use layout::{
    CoreQuadExtCold, CoreQuadHot, CoreQuadModulation, CoreQuadReadoutHot, CoreQuadReadoutRegs,
    CrossbarFrame, FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_OS2X,
    FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE, MAX_CORE_QUADS, MAX_SONA_TILES,
    MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, SONA_BLOCK_QUAD_SLOTS, SONA_CORE_QUAD_BLOCK_AND_LOCAL,
    SONA_FULL_HOT_DELTA_BYTES, SONA_INTERBLOCK_TILES, SONA_LITE_HOT_BYTES, SONA_OS2X_HOT_BYTES,
    SONA_PRO_HOT_DELTA_BYTES, SONA_ULTRA_HOT_DELTA_BYTES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER,
    SonaColdTail, SonaFullHotDelta, SonaHotSlab, SonaInterBlockTileHot, SonaLiteHotPrefix,
    SonaProHotDelta, SonaSlabBacking, SonaUltraHotDelta, TIER_SWITCH_XFADE_SAMPLES,
    TierCrossbarSnapshot,
};
pub use microchunk::{
    evaluate_matryoshka_core_quads, process_block_sona, process_tick16_const, step_frame_4x_sona,
    step_frame_sona,
};
#[cfg(target_arch = "x86_64")]
pub(crate) use microchunk::{
    process_block_sona_inline, process_tick16_const_inline, step_frame_4x_sona_inline,
    step_frame_sona_inline,
};

use crate::graph::GraphBackend;
use crate::types::VmError;

impl<B: GraphBackend> SonaHotSlab<'_, B> {
    /// Processes a fixed `16`-frame (`3 kHz` tick at `48 kHz`) slice (`#[inline(never)]` RT audit root).
    #[inline(never)]
    pub fn process_tick16_const(
        &mut self,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32; 16],
        out_r: &mut [f32; 16],
    ) -> Result<(), VmError> {
        process_tick16_const(self, in_l, in_r, out_l, out_r)
    }

    /// Processes an arbitrary-length stereo block with strict `4`-frame micro-chunk gating.
    #[inline]
    pub fn process_block(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), VmError> {
        process_block_sona(self, in_l, in_r, out_l, out_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "alloc")]
    use crate::bin::{Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire};
    use crate::bin::{DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, DnamBinView, encode_dnam_bin_into};
    use crate::control::{ControlBusBacking, ControlClampDescriptor, Cp1ContributorSlot};
    #[cfg(feature = "alloc")]
    use crate::math::ScalarBackend;
    use crate::math::{DefaultMathBackend, f32_to_fp16};
    use crate::types::Tier;
    use dnam_kernels::QUAD_FLAG_HOST_WRITE_LOCK;
    #[cfg(feature = "alloc")]
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn make_lite_bin_bytes() -> ([u8; 128 + 4_608], usize) {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.transport_lite.delay_samples = [12.0, 17.0, 23.0, 31.0].map(F32::new);
        a0.transport_lite.g_loop = F32::new(0.4);
        a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
        a0.transport_lite.input_gains_fp16 = [0.25; 4].map(|v| U16::new(f32_to_fp16(v)));
        a0.transport_lite.wet_tap_l_fp16 = [0.25; 4].map(|v| U16::new(f32_to_fp16(v)));
        a0.transport_lite.wet_tap_r_fp16 = [0.25; 4].map(|v| U16::new(f32_to_fp16(v)));
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
        a0.core_quads[0].in_mid_weight_fp16 =
            [0.5, 0.25, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        a0.core_quads[0].svf_g = [0.2; 4].map(F32::new);
        a0.core_quads[0].svf_k_fp16 = [U16::new(f32_to_fp16(0.7)); 4];
        a0.core_quads[0].tone_dir_fp16 = [U16::new(f32_to_fp16(1.0)); 4];
        a0.linear_readout_lite.w_ro_fp16[6] = U16::new(f32_to_fp16(0.5));
        a0.linear_readout_lite.w_ro_fp16[7] = U16::new(f32_to_fp16(0.5));
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
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
        (buf, n)
    }

    struct LiteTestBufs<'a, B: GraphBackend> {
        cold: [SonaColdTail<'a, B>; 1],
        descs: [ControlClampDescriptor; 1],
        raw: [f32; 1],
        clamped: [f32; 1],
        norm: [f32; 1],
        sw: [u8; 1],
        contrib: [Cp1ContributorSlot; 8],
    }

    impl<B: GraphBackend> LiteTestBufs<'_, B> {
        fn new() -> Self {
            Self {
                cold: [SonaColdTail::empty()],
                descs: [ControlClampDescriptor::empty(); 1],
                raw: [0.0; 1],
                clamped: [0.0; 1],
                norm: [0.0; 1],
                sw: [0; 1],
                contrib: [Cp1ContributorSlot::zero(); 8],
            }
        }
    }

    fn hydrate_lite_no_alloc<'a, B: GraphBackend>(
        bytes: &'a [u8],
        host_hz: f32,
        bufs: &'a mut LiteTestBufs<'a, B>,
    ) -> SonaHotSlab<'a, B> {
        let view = DnamBinView::parse(bytes).unwrap();
        let ctrl = ControlBusBacking::from_slices(
            &mut bufs.descs,
            &mut bufs.raw,
            &mut bufs.clamped,
            &mut bufs.norm,
            &mut bufs.sw,
            &mut bufs.contrib,
            &[],
        )
        .unwrap();
        let backing = SonaSlabBacking {
            control_bus: ctrl,
            convolver_fir_ext: &mut [],
            transport_ext_fp16: [&mut [], &mut [], &mut [], &mut []],
            transport_ext_f32: [&mut [], &mut [], &mut [], &mut []],
            transport_schroeder: [&mut [], &mut [], &mut [], &mut []],
            cold_tail: &mut bufs.cold,
        };
        hydrate_slab_from_dnam_bin(&view, Tier::Lite, host_hz, false, backing).unwrap()
    }

    /// verify[dsp.head.direct-register-readout]
    /// verify[dsp.head.tier-equivalence]
    #[test]
    fn test_zero_ulp_chunk_invariance_starcenter_and_dual_latch_no_alloc() {
        assert_eq!(
            [
                SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false),
                SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, true),
                SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Full, true),
                SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Pro, true),
                SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true),
            ],
            [4_544, 4_800, 7_552, 11_136, 14_592]
        );
        let (buf, n) = make_lite_bin_bytes();
        let bytes = &buf[..n];
        let mut in_l = [0.0_f32; 64];
        let mut in_r = [0.0_f32; 64];
        for i in 0..64 {
            in_l[i] = ((i as f32) * 0.17) - 0.5;
            in_r[i] = ((i as f32) * -0.11) + 0.25;
        }

        let (mut out64_l, mut out64_r) = ([0.0_f32; 64], [0.0_f32; 64]);
        {
            let mut bufs = LiteTestBufs::<DefaultMathBackend>::new();
            let mut slab = hydrate_lite_no_alloc(bytes, 48_000.0, &mut bufs);
            slab.process_block(&in_l, &in_r, &mut out64_l, &mut out64_r)
                .unwrap();
            let zero = step_frame_sona(&mut slab, 0.0, 0.0);
            assert!(zero.is_ok());
            reset_to_dc_equilibrium(&mut slab);
            for _ in 0..8 {
                let (zl, zr) = step_frame_sona(&mut slab, 0.0, 0.0).unwrap();
                assert_eq!((zl.to_bits(), zr.to_bits()), (0, 0));
            }
            set_quad_write_enable(&mut slab, true);
            assert_ne!(
                slab.transport_quad_hot(0).flags & QUAD_FLAG_HOST_WRITE_LOCK,
                0
            );
            reset_to_dc_equilibrium(&mut slab);
            assert_ne!(
                slab.transport_quad_hot(0).flags & QUAD_FLAG_HOST_WRITE_LOCK,
                0
            );
            set_op_t_write_enable(&mut slab, false);
            assert_eq!(
                slab.transport_quad_hot(0).flags & QUAD_FLAG_HOST_WRITE_LOCK,
                0
            );
        }

        let (mut out_split_l, mut out_split_r) = ([0.0_f32; 64], [0.0_f32; 64]);
        {
            let mut bufs = LiteTestBufs::<DefaultMathBackend>::new();
            let mut slab = hydrate_lite_no_alloc(bytes, 48_000.0, &mut bufs);
            let mut pos = 0_usize;
            for &len in &[1_usize, 3, 7, 17, 36] {
                slab.process_block(
                    &in_l[pos..pos + len],
                    &in_r[pos..pos + len],
                    &mut out_split_l[pos..pos + len],
                    &mut out_split_r[pos..pos + len],
                )
                .unwrap();
                pos += len;
            }
        }

        for i in 0..64 {
            assert_eq!(out64_l[i].to_bits(), out_split_l[i].to_bits());
            assert_eq!(out64_r[i].to_bits(), out_split_r[i].to_bits());
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_owned_ultra_hydrate_tier_switch_and_44k1_invariance() {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
        a0.transport_lite.delay_samples = [24.0, 31.0, 37.0, 43.0].map(F32::new);
        a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
        let (a1, a2, a3) = (
            Arena1FullDeltaWire::new_zeroed(),
            Arena2ProDeltaWire::new_zeroed(),
            Arena3UltraDeltaWire::new_zeroed(),
        );
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 8,
            ext_ring_lines: 4,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 48,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        };
        let mut buf = alloc::vec![0_u8; 128 + 30_720];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();
        let view = DnamBinView::parse(&buf[..n]).unwrap();
        let mut slab_a =
            hydrate_slab_owned::<ScalarBackend>(&view, Tier::Ultra, 44_100.0, false).unwrap();
        let mut slab_b =
            hydrate_slab_owned::<ScalarBackend>(&view, Tier::Ultra, 44_100.0, false).unwrap();

        let (in_l, in_r) = ([0.3_f32; 48], [-0.2_f32; 48]);
        let (mut al, mut ar) = ([0.0_f32; 48], [0.0_f32; 48]);
        let (mut bl, mut br) = ([0.0_f32; 48], [0.0_f32; 48]);
        slab_a
            .process_block(&in_l, &in_r, &mut al, &mut ar)
            .unwrap();
        for i in 0..48 {
            let (ol, or) = step_frame_sona(&mut slab_b, in_l[i], in_r[i]).unwrap();
            (bl[i], br[i]) = (ol, or);
        }
        for i in 0..48 {
            assert_eq!(al[i].to_bits(), bl[i].to_bits());
            assert_eq!(ar[i].to_bits(), br[i].to_bits());
        }
        switch_tier(&mut slab_a, Tier::Lite).unwrap();
        assert!(slab_a.xfade_active());
    }
}
