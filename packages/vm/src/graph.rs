//! Real-time Layer 0 `SONA v3.0` DSP graph, `CabinetBackend` / `GraphBackend` traits, and
//! cache-locked L1D `SonaHotSlab` (`#![no_std]`, `#![forbid(unsafe_code)]`, `0 FDIV`, `0 FSQRT`).
//!
//! Organizes dual-rate [`zoh`], 8-port [`router`], zero-lag `2x`/`4x` [`oversample`], lag-ordered
//! [`cabinet`] (`FIR` + `Kautz`), multi-quad orthogonal [`tank`], and prefix-nested [`slab`]
//! (`ref[arch.pillars.universal-black-box]`, `ref[dsp.router.signal-flow]`).

pub mod cabinet;
pub mod oversample;
pub mod router;
pub mod slab;
pub mod tank;
pub mod zoh;

pub use cabinet::high_sr::{CabinetHighSrFirExt, CabinetHighSrMode, EXT_FIR_STORAGE_F32_LEN};
pub use cabinet::kautz::{
    KautzColdSlewState, KautzPrefixOverride, KautzStereoBank4, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS,
    hydrate_kautz_slice_and_zoh, retune_kautz_banks_3khz,
};
pub use cabinet::{
    CABINET_FULL_DELTA_FIR_TAPS, CABINET_KAUTZ_ONSET_DELAY_48K, CABINET_LITE_FIR_TAPS,
    CABINET_MAX_CHUNK_FRAMES, CABINET_MAX_FIR_TAPS, CABINET_PRO_DELTA_FIR_TAPS, CABINET_SEG_TAPS,
    CABINET_ULTRA_DELTA_FIR_TAPS, CABINET_WORK_LEN, CabinetFirHistorySlices,
    CabinetFirWeightSlices, dot16_stereo_simd, process_cabinet_48k_chunk,
};
pub use oversample::{
    Bounce4xOversampler, OFFLINE_BOUNCE_PDC_SAMPLES, SonaOs2xHot, SonaOversamplerParams,
    SonaStereoOversampler2x,
};
pub use router::{
    DEFAULT_SIDECHAIN_LR4_HZ, IoAlignmentMode, IoRouter, IoSidechainDetectorBank,
    ParallelAlignLine, RouterGateConfig, WdwDryTapMode, eval_stereo_linked_aux_detector,
    eval_wdw_routing_outputs, voltage_ms_decode, voltage_ms_encode,
};
pub use slab::{
    CrossbarFrame, FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_OS2X,
    FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE, HoistedCoreChunk,
    SONA_BLOCK_GROUP_SLOTS, SONA_GROUP_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES,
    SONA_ULTRA_GROUP_TRAVERSAL_ORDER, SonaColdTail, SonaFullHotDelta, SonaGroupHot,
    SonaGroupReadoutHot, SonaGroupReadoutRegs, SonaHotSlab, SonaInterBlockTileHot,
    SonaLiteHotPrefix, SonaProHotDelta, SonaSlabBacking, SonaUltraHotDelta,
    TIER_SWITCH_XFADE_SAMPLES, TierCrossbarSnapshot, evaluate_matryoshka_groups,
    hoist_groups_and_readouts, hydrate_slab_from_dnam_bin, latch_immediate_at_init_or_reset,
    process_block_sona, process_tick16_const, reset_to_dc_equilibrium, retime_sample_rate,
    set_offline_bounce, set_op_t_write_enable, step_control_tick_3khz, step_frame_4port_sona,
    step_frame_4x_sona, step_frame_sona, switch_tier,
};
#[cfg(feature = "alloc")]
pub use slab::{SonaSlabOwnedBacking, hydrate_slab_owned};
pub use tank::{
    PartitionedSchroederBank, QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig, QuadModSlots,
    QuadRingMut, QuadStepResult, TANK_INLINE_QUAD0_CAP, TANK_MAX_EXT_CAP, TANK_MAX_QUADS,
    TANK_MIN_EXT_CAP, TankColdStorageSummary, TankExtRing, TankExtStorage, TankQuadHotCompact,
    TankTopOfFrameResult, compute_schroeder_requirements, compute_tank_storage_requirements,
    latch_quad_nonslew_targets, required_ext_elements, reset_tank, step_tank_end_of_frame,
    step_tank_top_of_frame, tick_quad_3khz, tick_tank_3khz,
};
pub use zoh::{ZohRateContext, ballistic_alpha_f32, rescale_paralru_lane, round_sample_count_zoh};

use crate::math::{MathBackend, ScalarBackend};

/// ISA-specialized `16`-tap stereo Cabinet Early FIR inner product backend (`0 FDIV`, `0 alloc`).
pub trait CabinetBackend: MathBackend {
    /// Evaluates one `16`-tap stereo FIR segment across four `4`-lane tiles (`w_l/r`, `x_l/r`).
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32);
}

/// Complete real-time `SONA v3.0` DSP graph SIMD backend capability trait across all 8 ISA targets.
pub trait GraphBackend: MathBackend + CabinetBackend {}

impl CabinetBackend for ScalarBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

impl GraphBackend for ScalarBackend {}

#[cfg(target_arch = "aarch64")]
impl CabinetBackend for crate::math::aarch64::NeonBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl GraphBackend for crate::math::aarch64::NeonBackend {}

#[cfg(target_arch = "aarch64")]
impl CabinetBackend for crate::math::aarch64::ArmSve2Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl GraphBackend for crate::math::aarch64::ArmSve2Backend {}

#[cfg(target_arch = "aarch64")]
impl CabinetBackend for crate::math::aarch64::ArmSmeBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl GraphBackend for crate::math::aarch64::ArmSmeBackend {}

#[cfg(target_arch = "x86_64")]
impl CabinetBackend for crate::math::x86::X86Sse2Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl GraphBackend for crate::math::x86::X86Sse2Backend {}

#[cfg(target_arch = "x86_64")]
impl CabinetBackend for crate::math::x86::X86Sse42Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl GraphBackend for crate::math::x86::X86Sse42Backend {}

#[cfg(target_arch = "x86_64")]
impl CabinetBackend for crate::math::x86::X86Avx2FmaBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl GraphBackend for crate::math::x86::X86Avx2FmaBackend {}

#[cfg(target_arch = "wasm32")]
impl CabinetBackend for crate::math::wasm::WasmSimd128Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        cabinet::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "wasm32")]
impl GraphBackend for crate::math::wasm::WasmSimd128Backend {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    fn verify_dot16<B: GraphBackend>() {
        let w_l = [[0.25_f32; 4], [-0.5; 4], [0.125; 4], [0.5; 4]];
        let w_r = [[-0.25_f32; 4], [0.75; 4], [-0.125; 4], [0.25; 4]];
        let x_l = [[1.0_f32, 2.0, -1.0, 0.5]; 4];
        let x_r = [[-1.0_f32, 0.5, 2.0, -0.25]; 4];

        let (dl, dr) = B::dot16_stereo(&w_l, &w_r, &x_l, &x_r);
        let mut ref_l = 0.0_f64;
        let mut ref_r = 0.0_f64;
        for seg in 0..4 {
            for lane in 0..4 {
                ref_l += f64::from(w_l[seg][lane]) * f64::from(x_l[seg][lane]);
                ref_r += f64::from(w_r[seg][lane]) * f64::from(x_r[seg][lane]);
            }
        }
        assert!((f64::from(dl) - ref_l).abs() < 1.0e-5);
        assert!((f64::from(dr) - ref_r).abs() < 1.0e-5);
    }

    #[test]
    fn test_cabinet_and_graph_backend_dispatch() {
        verify_dot16::<ScalarBackend>();
        verify_dot16::<DefaultMathBackend>();
        #[cfg(target_arch = "aarch64")]
        {
            verify_dot16::<crate::math::aarch64::NeonBackend>();
            verify_dot16::<crate::math::aarch64::ArmSve2Backend>();
            verify_dot16::<crate::math::aarch64::ArmSmeBackend>();
        }
        #[cfg(target_arch = "x86_64")]
        {
            verify_dot16::<crate::math::x86::X86Sse2Backend>();
            verify_dot16::<crate::math::x86::X86Sse42Backend>();
            verify_dot16::<crate::math::x86::X86Avx2FmaBackend>();
        }
        #[cfg(target_arch = "wasm32")]
        {
            verify_dot16::<crate::math::wasm::WasmSimd128Backend>();
        }
    }
}
