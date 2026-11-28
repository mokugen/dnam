//! Real-time Layer 0 `SONA v4.0` DSP graph, `ConvolverBackend` / `GraphBackend` traits, and
//! cache-locked L1D `SonaHotSlab` (`#![no_std]`, `#![forbid(unsafe_code)]`, `0 FDIV`, `0 FSQRT`).
//!
//! Organizes dual-rate [`zoh`], continuous `4`-lane [`sensory`] bank, zero-lag `2x`/`4x`
//! [`oversample`], lag-ordered [`convolver`] (`FIR` + `Kautz`), multi-quad orthogonal
//! [`transport`], and prefix-nested [`slab`] (`ref[arch.pillars.universal-black-box]`,
//! `ref[dsp.router.signal-flow]`).

pub mod convolver;
pub mod oversample;
pub mod sensory;
pub mod slab;
pub mod transport;
pub mod zoh;

pub use convolver::high_sr::{ConvolverHighSrFirExt, ConvolverHighSrMode, EXT_FIR_STORAGE_F32_LEN};
pub use convolver::kautz::{
    KautzColdSlewState, KautzPrefixOverride, KautzStereoBank4, MAX_KAUTZ_BANKS, MAX_KAUTZ_PAIRS,
    hydrate_kautz_slice_and_zoh, init_log_octave_poles_h2_into, retune_kautz_banks_3khz,
};
pub use convolver::{
    CONVOLVER_FULL_DELTA_FIR_TAPS, CONVOLVER_KAUTZ_ONSET_DELAY_48K, CONVOLVER_LITE_FIR_TAPS,
    CONVOLVER_MAX_CHUNK_FRAMES, CONVOLVER_MAX_FIR_TAPS, CONVOLVER_PRO_DELTA_FIR_TAPS,
    CONVOLVER_SEG_TAPS, CONVOLVER_ULTRA_DELTA_FIR_TAPS, CONVOLVER_WORK_LEN,
    ConvolverFirHistorySlices, ConvolverFirWeightSlices, dot16_stereo_simd,
    process_convolver_48k_chunk,
};
pub use oversample::{
    CoreOversampler, CoreOversamplerLiveHot, CoreOversamplerMasterCold,
    MASTER_CTRL_DELAY_MICROCHUNKS, MasterCtrlDelayRingCold, MasterCtrlSnapshotCold, RenderMode,
};
pub use sensory::{SensoryChunkRegs, SensoryParamsHot, SensoryQuadCold, SensoryQuadHot};
pub use slab::{
    CoreQuadExtCold, CoreQuadHot, CoreQuadModulation, CoreQuadReadoutHot, CoreQuadReadoutRegs,
    CrossbarFrame, FLAG_COLD_FIR_EXT, FLAG_CROSSBAR_ACTIVE, FLAG_MONO_OUT, FLAG_OS2X,
    FLAG_SINGLE_INPUT, FLAG_WET_POLARITY_INV, FLAG_XFADE_ACTIVE, MAX_CORE_QUADS, MAX_SONA_TILES,
    MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, SONA_BLOCK_QUAD_SLOTS, SONA_CORE_QUAD_BLOCK_AND_LOCAL,
    SONA_FULL_HOT_DELTA_BYTES, SONA_INTERBLOCK_TILES, SONA_LITE_HOT_BYTES,
    SONA_PRO_HOT_DELTA_BYTES, SONA_ULTRA_HOT_DELTA_BYTES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER,
    SonaColdTail, SonaFullHotDelta, SonaHotSlab, SonaInterBlockTileHot, SonaLiteHotPrefix,
    SonaProHotDelta, SonaSlabBacking, SonaUltraHotDelta, TIER_SWITCH_XFADE_SAMPLES,
    TierCrossbarSnapshot, evaluate_matryoshka_core_quads, hoist_core_quads_and_readouts,
    hydrate_slab_from_dnam_bin, latch_immediate_at_init_or_reset, process_block_sona,
    process_tick16_const, reset_to_dc_equilibrium, retime_sample_rate, set_offline_bounce,
    set_op_t_write_enable, set_quad_write_enable, set_render_mode, step_control_tick_3khz,
    step_frame_sona, switch_tier,
};
#[cfg(feature = "alloc")]
pub use slab::{SonaSlabOwnedBacking, hydrate_slab_owned, hydrate_slab_owned_with_headroom};
pub use slab::{VmReloadHeadroom, check_reload_slab_capacity, reload_slab_from_dnam_bin_in_place};
#[cfg(feature = "alloc")]
pub use transport::ExtRingAllocSpec;
pub use transport::{
    PartitionedSchroederBank, QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig, QuadRingMut,
    QuadStepResult, QuadTopRead, TRANSPORT_ALLPASS_G_MAX, TRANSPORT_INLINE_QUAD0_CAP,
    TRANSPORT_MAX_EXT_CAP, TRANSPORT_MAX_QUADS, TRANSPORT_MIN_EXT_CAP,
    TRANSPORT_SCHROEDER_MAX_DELAY, TransportColdStorageSummary, TransportExtRing,
    TransportExtStorage, TransportModSlots, TransportQuadHotCompact, TransportStageState,
    TransportTopOfFrameResult, apply_transport_mod_slots, compute_quad_ext_requirements,
    compute_schroeder_requirements, compute_transport_storage_requirements,
    f32_to_fp16_rtz_with_sat, latch_quad_nonslew_targets, required_ext_elements, reset_transport,
    step_transport_end_of_frame, step_transport_end_of_frame_with_output,
    step_transport_top_of_frame, step_transport_top_of_frame_reads, tick_quad_3khz,
    tick_transport_3khz,
};
pub use zoh::{ZohRateContext, ballistic_alpha_f32, rescale_paralru_lane, round_sample_count_zoh};

use crate::math::{DefaultMathBackend, MathBackend, ScalarBackend};
use crate::types::VmError;

/// ISA-specialized `16`-tap stereo `ModalConvolver` Early FIR inner product backend (`0 FDIV`, `0 alloc`).
pub trait ConvolverBackend: MathBackend {
    /// Evaluates one `16`-tap stereo FIR segment across four `4`-lane tiles (`w_l/r`, `x_l/r`).
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32);
}

/// Complete real-time `SONA v4.0` DSP graph SIMD backend capability trait across all 8 ISA targets.
pub trait GraphBackend: MathBackend + ConvolverBackend {
    /// Processes a stereo audio block on `slab` (`#[target_feature]`-specialized on `X86Avx2Fma`).
    #[inline(always)]
    fn process_block_slab(
        slab: &mut SonaHotSlab<'_, Self>,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), VmError> {
        process_block_sona(slab, in_l, in_r, out_l, out_r)
    }

    /// Processes a fixed `16`-frame slice on `slab`.
    #[inline(always)]
    fn process_tick16_slab(
        slab: &mut SonaHotSlab<'_, Self>,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32; 16],
        out_r: &mut [f32; 16],
    ) -> Result<(), VmError> {
        slab.process_tick16_const(in_l, in_r, out_l, out_r)
    }

    /// Steps one stereo host sample (`RenderMode::Live` or `RenderMode::Master`) on `slab`.
    #[inline(always)]
    fn step_frame_slab(
        slab: &mut SonaHotSlab<'_, Self>,
        in_l: f32,
        in_r: f32,
    ) -> Result<(f32, f32), VmError> {
        step_frame_sona(slab, in_l, in_r)
    }

    /// Executes one `3 kHz` control boundary tick on `slab`.
    #[inline(always)]
    fn step_control_tick_slab(slab: &mut SonaHotSlab<'_, Self>) {
        step_control_tick_3khz(slab);
    }
}

impl ConvolverBackend for ScalarBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

impl GraphBackend for ScalarBackend {}

#[cfg(target_arch = "aarch64")]
impl ConvolverBackend for crate::math::aarch64::NeonBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl GraphBackend for crate::math::aarch64::NeonBackend {}

#[cfg(target_arch = "aarch64")]
impl ConvolverBackend for crate::math::aarch64::ArmSve2Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl GraphBackend for crate::math::aarch64::ArmSve2Backend {}

#[cfg(target_arch = "aarch64")]
impl ConvolverBackend for crate::math::aarch64::ArmSmeBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl GraphBackend for crate::math::aarch64::ArmSmeBackend {}

#[cfg(target_arch = "x86_64")]
impl ConvolverBackend for crate::math::x86::X86Sse2Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl GraphBackend for crate::math::x86::X86Sse2Backend {}

#[cfg(target_arch = "x86_64")]
impl ConvolverBackend for crate::math::x86::X86Sse42Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl GraphBackend for crate::math::x86::X86Sse42Backend {}

#[cfg(target_arch = "x86_64")]
impl ConvolverBackend for crate::math::x86::X86Avx2FmaBackend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl GraphBackend for crate::math::x86::X86Avx2FmaBackend {
    #[inline(always)]
    fn process_block_slab(
        slab: &mut SonaHotSlab<'_, Self>,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), VmError> {
        crate::math::x86::avx2::dispatch_avx2_process_block(slab, in_l, in_r, out_l, out_r)
    }

    #[inline(always)]
    fn process_tick16_slab(
        slab: &mut SonaHotSlab<'_, Self>,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32; 16],
        out_r: &mut [f32; 16],
    ) -> Result<(), VmError> {
        crate::math::x86::avx2::dispatch_avx2_process_tick16(slab, in_l, in_r, out_l, out_r)
    }

    #[inline(always)]
    fn step_frame_slab(
        slab: &mut SonaHotSlab<'_, Self>,
        in_l: f32,
        in_r: f32,
    ) -> Result<(f32, f32), VmError> {
        crate::math::x86::avx2::dispatch_avx2_step_frame(slab, in_l, in_r)
    }

    #[inline(always)]
    fn step_control_tick_slab(slab: &mut SonaHotSlab<'_, Self>) {
        crate::math::x86::avx2::dispatch_avx2_step_control_tick(slab);
    }
}

#[cfg(target_arch = "wasm32")]
impl ConvolverBackend for crate::math::wasm::WasmSimd128Backend {
    #[inline(always)]
    fn dot16_stereo(
        w_l: &[[f32; 4]; 4],
        w_r: &[[f32; 4]; 4],
        x_l: &[[f32; 4]; 4],
        x_r: &[[f32; 4]; 4],
    ) -> (f32, f32) {
        convolver::dot16_stereo_simd::<Self>(w_l, w_r, x_l, x_r)
    }
}

#[cfg(target_arch = "wasm32")]
impl GraphBackend for crate::math::wasm::WasmSimd128Backend {}

/// Concrete `DefaultMathBackend` audit entry point evaluating active Matryoshka core quads.
#[inline(never)]
pub fn sona_graph_eval_core_quads_default(
    slab: &mut SonaHotSlab<'_, DefaultMathBackend>,
    crossbar: &CrossbarFrame<<DefaultMathBackend as MathBackend>::Vec4>,
    u_lr: (f32, f32),
) -> ((f32, f32), (f32, f32)) {
    evaluate_matryoshka_core_quads::<DefaultMathBackend>(slab, crossbar, u_lr)
}

/// Concrete `DefaultMathBackend` audit entry point evaluating a `4`-frame convolver FIR chunk.
#[inline(never)]
pub fn sona_graph_convolver_chunk_default(
    hist: &mut ConvolverFirHistorySlices<'_>,
    weights: &ConvolverFirWeightSlices<'_>,
    inputs_lr: &[[f32; 2]; 4],
    early_out: &mut [[f32; 2]; 4],
    onset_out: &mut [[f32; 2]; 4],
) {
    process_convolver_48k_chunk::<4, _>(
        hist,
        weights,
        inputs_lr,
        early_out,
        onset_out,
        dot16_stereo_simd::<DefaultMathBackend>,
    );
}

/// Concrete `DefaultMathBackend` audit entry point executing Top-of-Frame transport reads.
#[inline(never)]
pub fn sona_graph_transport_top_default(
    state: &mut TransportStageState<'_, '_>,
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &[QuadEffectiveTargets],
    x_tank_lr: (f32, f32),
    stage_dry_mix_eff: f32,
) -> Result<TransportTopOfFrameResult, crate::types::VmError> {
    step_transport_top_of_frame::<DefaultMathBackend>(
        state,
        schroeders,
        effective,
        x_tank_lr,
        stage_dry_mix_eff,
    )
}

/// Concrete `DefaultMathBackend` audit entry point executing End-of-Frame transport writes.
#[inline(never)]
pub fn sona_graph_transport_end_default(
    state: &mut TransportStageState<'_, '_>,
    effective: &[QuadEffectiveTargets],
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    x_tank_lr: (f32, f32),
    stage_dry_mix_eff: f32,
) -> Result<((f32, f32), bool), crate::types::VmError> {
    step_transport_end_of_frame_with_output::<DefaultMathBackend>(
        state,
        effective,
        reads,
        x_tank_lr,
        stage_dry_mix_eff,
    )
}

/// Concrete `DefaultMathBackend` audit entry point stepping the `4`-lane [`SensoryQuadCold`] bank.
#[inline(never)]
pub fn sona_graph_sensory_step_default(
    cold: &SensoryQuadCold,
    hot: &mut SensoryQuadHot,
    x_l: f32,
    x_r: f32,
    vca_bias_slot30: f32,
) -> (f32, f32) {
    cold.step_frame::<DefaultMathBackend>(hot, x_l, x_r, vca_bias_slot30)
}

/// Concrete `DefaultMathBackend` audit entry point stepping `Op-N[Pre]` stereo hysteresis.
#[inline(never)]
pub fn sona_graph_op_n_pre_step_default(
    state: &mut dnam_kernels::OpNPreHotState,
    in_l: f32,
    in_r: f32,
    d_scale: f32,
    delta_step: f32,
) -> (f32, f32) {
    state.step_stereo(in_l, in_r, d_scale, delta_step)
}

/// Concrete `DefaultMathBackend` audit entry point dispatching the `3 kHz` `Op-T` transport tick.
#[inline(never)]
pub fn sona_graph_dispatch_transport_3khz_default(
    quads: &mut [TransportQuadHotCompact],
    metas: &[QuadColdMeta],
    p_q: usize,
    slots: &TransportModSlots,
    ext_rings: &mut [TransportExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &mut [QuadEffectiveTargets],
) {
    tick_transport_3khz(quads, metas, p_q, slots, ext_rings, schroeders, effective);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, SimdVec4};
    use crate::types::ColdBuf;

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
    fn test_convolver_and_graph_backend_dispatch() {
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
            if crate::math::x86::X86Avx2FmaBackend::is_supported_cached() {
                verify_dot16::<crate::math::x86::X86Avx2FmaBackend>();
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            verify_dot16::<crate::math::wasm::WasmSimd128Backend>();
        }
    }

    #[test]
    fn test_sona_graph_default_audit_entrypoints() {
        let mut cold_tail = SonaColdTail::<DefaultMathBackend>::empty();
        let mut slab =
            SonaHotSlab::empty_with_cold(ColdBuf::Borrowed(core::slice::from_mut(&mut cold_tail)));
        let zero_v = <DefaultMathBackend as MathBackend>::Vec4::zero();
        let crossbar = CrossbarFrame {
            active: false,
            op_p_vec: zero_v,
            car_l: zero_v,
            car_r: zero_v,
        };
        let (core_u, core_hp) =
            sona_graph_eval_core_quads_default(&mut slab, &crossbar, (0.25, -0.25));
        assert!(core_u.0.is_finite() && core_hp.0.is_finite());

        let (mut hist_l, mut hist_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut w_l, mut w_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        w_l[0] = 0.5;
        w_r[0] = 0.5;
        let mut hist = ConvolverFirHistorySlices::lite(&mut hist_l, &mut hist_r);
        let weights = ConvolverFirWeightSlices::lite(&w_l, &w_r);
        let inputs = [[0.2, -0.1]; 4];
        let (mut early_out, mut onset_out) = ([[0.0_f32; 2]; 4], [[0.0_f32; 2]; 4]);
        sona_graph_convolver_chunk_default(
            &mut hist,
            &weights,
            &inputs,
            &mut early_out,
            &mut onset_out,
        );
        assert!(early_out[0][0].is_finite() && onset_out[0][0].is_finite());

        let mut quads = [TransportQuadHotCompact::default()];
        let metas = [QuadColdMeta::default()];
        let mut inline_q0_ring = [[0_u16; 4]; TRANSPORT_INLINE_QUAD0_CAP];
        let mut effective = [QuadEffectiveTargets::default()];
        let slots = TransportModSlots::default();
        sona_graph_dispatch_transport_3khz_default(
            &mut quads,
            &metas,
            1,
            &slots,
            &mut [],
            &mut [],
            &mut effective,
        );
        let mut state = TransportStageState {
            quads: &mut quads,
            p_q: 1,
            inline_q0_ring: &mut inline_q0_ring,
            ext_rings: &mut [],
        };
        let top =
            sona_graph_transport_top_default(&mut state, &mut [], &effective, (0.1, -0.1), 1.0)
                .unwrap();
        let (end_lr, _sat) =
            sona_graph_transport_end_default(&mut state, &effective, &top.reads, (0.1, -0.1), 1.0)
                .unwrap();
        assert!(end_lr.0.is_finite() && end_lr.1.is_finite());

        let sensory_cold = SensoryQuadCold::default();
        let mut sensory_hot = SensoryQuadHot::default();
        let s_out =
            sona_graph_sensory_step_default(&sensory_cold, &mut sensory_hot, 0.3, -0.2, 0.0);
        assert!(s_out.0.is_finite() && s_out.1.is_finite());

        let mut op_n_pre = dnam_kernels::OpNPreHotState::zero();
        let n_out = sona_graph_op_n_pre_step_default(&mut op_n_pre, 0.3, -0.2, 1.0, 0.1);
        assert!(n_out.0.is_finite() && n_out.1.is_finite());
    }
}
