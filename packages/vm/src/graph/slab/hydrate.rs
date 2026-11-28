//! Cold `.dnam-bin` hydration (`!alloc` [`hydrate_slab_from_dnam_bin`] & owned
//! [`hydrate_slab_owned`]) and runtime lifecycle facade (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub mod apply_tier;
pub mod lifecycle;
pub mod wire_decode;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use core::marker::PhantomData;

#[cfg(feature = "alloc")]
use dnam_kernels::QUAD_FLAG_HAS_EXT;
use dnam_kernels::{
    HilbertH1StereoKernel, MAX_PARALRU_R_F32, compute_t1_dc_blocker_coeffs_clamped,
    interpolate_hilbert_h1_alphas_f64,
};
use zerocopy::FromZeros;

#[cfg(feature = "alloc")]
use super::super::transport::ring::ExtRingAllocSpec;
use super::layout::{
    FLAG_COLD_FIR_EXT, FLAG_OS2X, SonaColdTail, SonaFullHotDelta, SonaHotSlab, SonaLiteHotPrefix,
    SonaProHotDelta, SonaSlabBacking, SonaUltraHotDelta,
};
use crate::bin::{DnamBinView, TRANSPORT_FLAG_EXT_F32, TransportQuadWire};
use crate::control::ControlBus;
#[cfg(feature = "alloc")]
use crate::control::ControlBusBacking;
use crate::control::bus::required_cp1_contributors;
use crate::graph::GraphBackend;
use crate::graph::convolver::high_sr::ConvolverHighSrFirExt;
#[cfg(feature = "alloc")]
use crate::graph::convolver::high_sr::EXT_FIR_STORAGE_F32_LEN;
use crate::graph::oversample::{CoreOversamplerMasterCold, RenderMode};
use crate::graph::sensory::{SensoryParamsHot, SensoryQuadCold};
use crate::graph::transport::{
    PartitionedSchroederBank, QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig,
    TRANSPORT_INLINE_QUAD0_CAP, TRANSPORT_MAX_QUADS, TransportExtRing, TransportQuadHotCompact,
    compute_quad_ext_requirements, compute_schroeder_requirements,
};
use crate::graph::zoh::ZohRateContext;
use crate::math::{clamp_finite_f32, f32_to_fp16};
use crate::types::{ColdBuf, Tier, VmError};

use super::ctrl_latch::latch_immediate_at_init_or_reset;
use apply_tier::{apply_tier_weights_and_rescale, populate_crossbar_and_core_quads};
use lifecycle::setup_phasor_increments;
use wire_decode::decode_readout_and_overrides;

pub use lifecycle::{
    reset_to_dc_equilibrium, retime_sample_rate, set_offline_bounce, set_op_t_write_enable,
    set_quad_write_enable, set_render_mode, switch_tier,
};

/// Pre-allocated capacity headroom for zero-allocation in-place model reload
/// (`hydrate_slab_owned_with_headroom`, `SonaVm::open_dnam_bin_with_headroom_owned`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VmReloadHeadroom {
    /// Minimum control descriptor/value/switch array capacity (e.g. `255`).
    pub min_controls: usize,
    /// Minimum `CP1` contributor slot capacity (e.g. `64`).
    pub min_cp1_contributors: usize,
    /// Minimum `"TSYN"` tempo-sync row capacity (e.g. `32`).
    pub min_tsyn_entries: usize,
    /// Minimum `fp16[4]` external transport delay frames pre-allocated per quad (`0` = none extra).
    pub ext_ring_fp16_frames_per_quad: usize,
    /// Minimum `f32[4]` external transport delay frames pre-allocated per quad (`0` = none extra).
    pub ext_ring_f32_frames_per_quad: usize,
    /// Minimum partitioned Schroeder diffuser `f32` elements pre-allocated per quad (`0` = none extra).
    pub schroeder_f32_per_quad: usize,
}

impl VmReloadHeadroom {
    /// Zero extra reload headroom (exact model capacity only).
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            min_controls: 0,
            min_cp1_contributors: 0,
            min_tsyn_entries: 0,
            ext_ring_fp16_frames_per_quad: 0,
            ext_ring_f32_frames_per_quad: 0,
            schroeder_f32_per_quad: 0,
        }
    }

    /// Standard live DAW/studio control-plane reload headroom (`255` controls, `64` `CP1`, `32` `TSYN`).
    #[inline]
    #[must_use]
    pub const fn for_live_engine() -> Self {
        Self {
            min_controls: 255,
            min_cp1_contributors: 64,
            min_tsyn_entries: 32,
            ext_ring_fp16_frames_per_quad: 0,
            ext_ring_f32_frames_per_quad: 0,
            schroeder_f32_per_quad: 0,
        }
    }

    /// Attaches optional per-quad transport external ring and Schroeder reload headroom.
    #[inline]
    #[must_use]
    pub const fn with_transport_headroom(
        mut self,
        ext_ring_fp16_frames_per_quad: usize,
        ext_ring_f32_frames_per_quad: usize,
        schroeder_f32_per_quad: usize,
    ) -> Self {
        self.ext_ring_fp16_frames_per_quad = ext_ring_fp16_frames_per_quad;
        self.ext_ring_f32_frames_per_quad = ext_ring_f32_frames_per_quad;
        self.schroeder_f32_per_quad = schroeder_f32_per_quad;
        self
    }
}

fn wire_transport_quads_from_view(
    view: &DnamBinView<'_>,
) -> [TransportQuadWire; TRANSPORT_MAX_QUADS] {
    let mut quads = [view.arena0.transport_lite; TRANSPORT_MAX_QUADS];
    if let Some(a1) = view.arena1 {
        quads[1] = a1.transport_full;
    }
    if let Some(a2) = view.arena2 {
        quads[2] = a2.transport_pro[0];
    }
    if let Some(a3) = view.arena3 {
        quads[3] = a3.transport_ultra[0];
    }
    quads
}

fn setup_transport_wire_slots<B: GraphBackend>(
    view: &DnamBinView<'_>,
    slab: &mut SonaHotSlab<'_, B>,
) {
    let quads = wire_transport_quads_from_view(view);
    let cold = slab.cold_mut();
    cold.wire_transport_quads = quads;
    cold.declared_max_delay_48k = view.runtime_meta.declared_max_delay_48k.get();
    cold.ext_ring_pow2_frames = view.runtime_meta.ext_ring_pow2_frames;
    cold.dc_blocker_pole_5hz = view.runtime_meta.dc_blocker_pole_5hz.get();
}

fn prepare_common_cold_in_place<B: GraphBackend>(
    view: &DnamBinView<'_>,
    slab: &mut SonaHotSlab<'_, B>,
    active_tier: Tier,
    zoh: &ZohRateContext,
) -> Result<(), VmError> {
    let is_master = matches!(zoh.render_mode, RenderMode::Master);
    let active_taps = active_tier.early_fir_taps();
    let (ai, aq) = interpolate_hilbert_h1_alphas_f64(zoh.f_s_vm);
    let (dc_pole, _) = compute_t1_dc_blocker_coeffs_clamped(zoh.f_s_vm);
    let ro_pole_48k = clamp_finite_f32(
        view.runtime_meta.ro_pole_48k.get(),
        0.0,
        MAX_PARALRU_R_F32,
        0.0,
    );
    let base_tank_dry_mix = {
        let cold = slab.cold_mut();
        let bus = cold
            .ctrl_bus
            .as_mut()
            .ok_or(VmError::InsufficientBackingCapacity)?;
        bus.offline_bounce = is_master;
        cold.convolver_fir_ext.prepare(
            &cold.tier_fir_l[active_tier.index()][..active_taps],
            &cold.tier_fir_r[active_tier.index()][..active_taps],
            zoh.f_s_vm,
            false,
        )?;
        bus.base_tank_dry_mix()
    };

    let use_cold_fir = ConvolverHighSrFirExt::required_f32_len(zoh.f_s_vm, false) > 0;
    slab.set_flag(FLAG_COLD_FIR_EXT, use_cold_fir);
    slab.set_flag(FLAG_OS2X, zoh.m_core >= 2);
    slab.lite.tier = active_tier;
    slab.lite.hilbert = HilbertH1StereoKernel::with_coeffs(ai, aq);
    slab.lite.t1_dc.set_pole(dc_pole);
    slab.lite.sensory_quad.reset();
    slab.lite.tank_dry_mix_eff_fp16 = f32_to_fp16(base_tank_dry_mix);

    let cold = slab.cold_mut();
    let cb = cold.wire_crossbar;
    cold.max_present_tier = Tier::from_u8(view.header.max_present_tier);
    cold.retained_tier = view.retained_tier;
    cold.zoh = *zoh;
    cold.render_mode = zoh.render_mode;
    cold.base_tank_dry_mix = base_tank_dry_mix;
    cold.ro_pole_48k = ro_pole_48k;
    let sensory_cold = SensoryQuadCold::from_crossbar_wire(&cb, zoh.rho_vm);
    cold.sensory_cold = sensory_cold;
    cold.master_os = CoreOversamplerMasterCold::new();
    slab.sensory_params = SensoryParamsHot::from_cold(&sensory_cold);
    setup_phasor_increments(slab, zoh);
    Ok(())
}

fn hydrate_transport_borrowed<'a, B: GraphBackend>(
    slab: &mut SonaHotSlab<'a, B>,
    zoh: &ZohRateContext,
    ext_fp16: [&'a mut [u16]; TRANSPORT_MAX_QUADS],
    ext_f32: [&'a mut [f32]; TRANSPORT_MAX_QUADS],
    schroeder_bufs: [&'a mut [f32]; TRANSPORT_MAX_QUADS],
) -> Result<(), VmError> {
    let [fp0, fp1, fp2, fp3] = ext_fp16;
    let [f32_0, f32_1, f32_2, f32_3] = ext_f32;
    let [sch0, sch1, sch2, sch3] = schroeder_bufs;
    let fp_slots = [fp0, fp1, fp2, fp3];
    let f32_slots = [f32_0, f32_1, f32_2, f32_3];
    let sch_slots = [sch0, sch1, sch2, sch3];

    let wires = slab.cold().wire_transport_quads;
    let (decl_max, pow2, dc5) = (
        slab.cold().declared_max_delay_48k,
        slab.cold().ext_ring_pow2_frames,
        slab.cold().dc_blocker_pole_5hz,
    );
    let active_q = slab.cold().retained_tier.index() + 1;
    let disabled_wire = TransportQuadWire::new_zeroed();

    for (q, ((fp_s, f32_s), sch_s)) in fp_slots
        .into_iter()
        .zip(f32_slots)
        .zip(sch_slots)
        .enumerate()
    {
        if q >= active_q {
            slab.cold_mut().transport_ext[q] = TransportExtRing::from_borrowed_parts(
                fp_s,
                f32_s,
                false,
                false,
                TRANSPORT_INLINE_QUAD0_CAP,
            )?;
            slab.cold_mut().transport_schroeder[q] = if sch_s.is_empty() {
                PartitionedSchroederBank::disabled()
            } else {
                PartitionedSchroederBank::from_cold_buf(
                    ColdBuf::borrowed(sch_s),
                    &disabled_wire,
                    1.0,
                )?
            };
            continue;
        }
        let cfg = QuadHydrateConfig {
            quad_idx: q,
            fs_vm: zoh.f_s_vm,
            rho_vm: zoh.rho_vm,
            declared_max_delay_48k: decl_max,
            ext_ring_pow2_frames: pow2,
            dc_blocker_pole_5hz: dc5,
        };
        let (hot_q, meta_q) = TransportQuadHotCompact::from_wire(&wires[q], &cfg)?;
        *slab.transport_quad_hot_mut(q) = hot_q;
        slab.cold_mut().transport_quad_meta[q] = meta_q;
        slab.cold_mut().transport_quad_eff[q] =
            QuadEffectiveTargets::from_quad_and_meta(&hot_q, &meta_q);

        let (has_ext, cap) =
            compute_quad_ext_requirements(&wires[q], q, zoh.rho_vm, decl_max, pow2);
        let want_f32 = (wires[q].flags & TRANSPORT_FLAG_EXT_F32) != 0;
        slab.cold_mut().transport_ext[q] =
            TransportExtRing::from_borrowed_parts(fp_s, f32_s, has_ext, want_f32, cap)?;
        let (_, _, _, sch_req) = compute_schroeder_requirements(&wires[q], zoh.rho_vm);
        slab.cold_mut().transport_schroeder[q] = if sch_req == 0 {
            if sch_s.is_empty() {
                PartitionedSchroederBank::disabled()
            } else {
                PartitionedSchroederBank::from_cold_buf(
                    ColdBuf::borrowed(sch_s),
                    &disabled_wire,
                    1.0,
                )?
            }
        } else {
            PartitionedSchroederBank::from_slice(sch_s, &wires[q], zoh.rho_vm)?
        };
    }
    Ok(())
}

/// Hydrates a [`SonaHotSlab`] strictly from caller-supplied `!alloc` [`SonaSlabBacking`].
pub fn hydrate_slab_from_dnam_bin<'a, B: GraphBackend>(
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
    backing: SonaSlabBacking<'a, B>,
) -> Result<SonaHotSlab<'a, B>, VmError> {
    crate::math::warm_simd_feature_caches();
    if active_tier.index() > view.retained_tier.index() {
        return Err(VmError::InvalidTier);
    }
    if backing.cold_tail.is_empty() {
        return Err(VmError::InsufficientBackingCapacity);
    }
    backing.cold_tail[0] = SonaColdTail::empty();
    let mut slab = SonaHotSlab::empty_with_cold(ColdBuf::borrowed(backing.cold_tail));

    populate_crossbar_and_core_quads(view, &mut slab);
    decode_readout_and_overrides(view, &mut slab)?;
    setup_transport_wire_slots(view, &mut slab);

    let mode = if offline_bounce {
        RenderMode::Master
    } else {
        RenderMode::Live
    };
    let ref_hz = view.header.reference_sample_rate_hz.get() as f32;
    let zoh = ZohRateContext::compute(ref_hz, host_rate_hz, mode)?;
    let bus = ControlBus::from_dnam_bin(view, active_tier, zoh.f_s_vm as f32, backing.control_bus)?;
    let fir_ext = if backing.convolver_fir_ext.is_empty() {
        ConvolverHighSrFirExt::from_slice(&mut [], 48_000.0, false)?
    } else {
        ConvolverHighSrFirExt::from_slice(backing.convolver_fir_ext, zoh.f_s_vm, false)?
    };
    slab.cold_mut().ctrl_bus = Some(bus);
    slab.cold_mut().convolver_fir_ext = fir_ext;
    prepare_common_cold_in_place(view, &mut slab, active_tier, &zoh)?;
    hydrate_transport_borrowed(
        &mut slab,
        &zoh,
        backing.transport_ext_fp16,
        backing.transport_ext_f32,
        backing.transport_schroeder,
    )?;
    apply_tier_weights_and_rescale(&mut slab, active_tier, &zoh)?;
    latch_immediate_at_init_or_reset(&mut slab);
    Ok(slab)
}

fn validate_reload_slab_capacity<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
) -> Result<(ZohRateContext, [TransportQuadWire; TRANSPORT_MAX_QUADS]), VmError> {
    if active_tier.index() > view.retained_tier.index() {
        return Err(VmError::InvalidTier);
    }
    let mode = if offline_bounce {
        RenderMode::Master
    } else {
        RenderMode::Live
    };
    let ref_hz = view.header.reference_sample_rate_hz.get() as f32;
    let zoh = ZohRateContext::compute(ref_hz, host_rate_hz, mode)?;

    let req_fir_len = ConvolverHighSrFirExt::required_f32_len(zoh.f_s_vm, false);
    if req_fir_len > slab.cold().convolver_fir_ext.backing_f32_capacity() {
        return Err(VmError::InsufficientBackingCapacity);
    }

    {
        let bus = slab
            .cold()
            .ctrl_bus
            .as_ref()
            .ok_or(VmError::InsufficientBackingCapacity)?;
        let (ctrl_cap, cp1_cap, tsyn_cap) = bus.backing.capacity_limits();
        let num_controls = view.header.num_controls.get() as usize;
        if num_controls > ctrl_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        if !view.tsyn_entries.is_empty() && view.tsyn_entries.len() > tsyn_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        let req_cp1 = required_cp1_contributors(view)?;
        if req_cp1 > cp1_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
    }

    let wires = wire_transport_quads_from_view(view);
    let (decl_max, pow2, dc5) = (
        view.runtime_meta.declared_max_delay_48k.get(),
        view.runtime_meta.ext_ring_pow2_frames,
        view.runtime_meta.dc_blocker_pole_5hz.get(),
    );
    let retained_q = (view.retained_tier.index() + 1).min(TRANSPORT_MAX_QUADS);
    let cold = slab.cold();
    for (q, wire) in wires.iter().enumerate().take(retained_q) {
        let cfg = QuadHydrateConfig {
            quad_idx: q,
            fs_vm: zoh.f_s_vm,
            rho_vm: zoh.rho_vm,
            declared_max_delay_48k: decl_max,
            ext_ring_pow2_frames: pow2,
            dc_blocker_pole_5hz: dc5,
        };
        let _ = TransportQuadHotCompact::from_wire(wire, &cfg)?;
        let (has_ext, cap) = compute_quad_ext_requirements(wire, q, zoh.rho_vm, decl_max, pow2);
        if has_ext {
            let want_f32 = (wire.flags & TRANSPORT_FLAG_EXT_F32) != 0;
            if cold.transport_ext[q].backing_frame_capacity_for(want_f32) < cap {
                return Err(VmError::InsufficientBackingCapacity);
            }
        }
        let (_, _, _, sch_req) = compute_schroeder_requirements(wire, zoh.rho_vm);
        if sch_req > 0 && cold.transport_schroeder[q].backing_f32_capacity() < sch_req {
            return Err(VmError::InsufficientBackingCapacity);
        }
    }
    Ok((zoh, wires))
}

/// Validates all backing capacities and wire parameters required to reload `slab` in-place
/// without mutating any state (`0` heap allocations, `0` drops, fail-closed capacity).
pub fn check_reload_slab_capacity<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
) -> Result<(), VmError> {
    validate_reload_slab_capacity(slab, view, active_tier, host_rate_hz, offline_bounce).map(|_| ())
}

/// Validates all backing capacities & wire parameters and reloads a hydrated [`SonaHotSlab`]
/// in-place (`0` heap allocations, `0` drops, fail-closed [`VmError::InsufficientBackingCapacity`]).
pub fn reload_slab_from_dnam_bin_in_place<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
) -> Result<(), VmError> {
    let (zoh, wires) =
        validate_reload_slab_capacity(slab, view, active_tier, host_rate_hz, offline_bounce)?;
    let (decl_max, pow2, dc5) = (
        view.runtime_meta.declared_max_delay_48k.get(),
        view.runtime_meta.ext_ring_pow2_frames,
        view.runtime_meta.dc_blocker_pole_5hz.get(),
    );
    let retained_q = (view.retained_tier.index() + 1).min(TRANSPORT_MAX_QUADS);

    // All fail-closed capacity checks passed; reload control bus & reset hot/cold metadata in place.
    slab.lite = SonaLiteHotPrefix::empty();
    slab.full_delta = SonaFullHotDelta::empty();
    slab.pro_delta = SonaProHotDelta::empty();
    slab.ultra_delta = SonaUltraHotDelta::empty();
    slab.sensory_params = SensoryParamsHot::zero();
    slab.cold_mut().reset_metadata_preserving_backing();

    populate_crossbar_and_core_quads(view, slab);
    decode_readout_and_overrides(view, slab)?;
    setup_transport_wire_slots(view, slab);

    {
        let cold = slab.cold_mut();
        cold.ctrl_bus
            .as_mut()
            .ok_or(VmError::InsufficientBackingCapacity)?
            .reload_from_dnam_bin_in_place(view, active_tier, zoh.f_s_vm as f32)?;
        cold.convolver_fir_ext.reset_for_reload();
    }
    prepare_common_cold_in_place(view, slab, active_tier, &zoh)?;

    for (q, wire) in wires.iter().enumerate() {
        if q >= retained_q {
            *slab.transport_quad_hot_mut(q) = TransportQuadHotCompact::default();
            let cold = slab.cold_mut();
            cold.transport_quad_meta[q] = QuadColdMeta::default();
            cold.transport_quad_eff[q] = QuadEffectiveTargets::default();
            cold.transport_ext[q].deactivate_preserving_storage();
            cold.transport_schroeder[q].deactivate_preserving_storage();
            continue;
        }
        let cfg = QuadHydrateConfig {
            quad_idx: q,
            fs_vm: zoh.f_s_vm,
            rho_vm: zoh.rho_vm,
            declared_max_delay_48k: decl_max,
            ext_ring_pow2_frames: pow2,
            dc_blocker_pole_5hz: dc5,
        };
        let (hot_q, meta_q) = TransportQuadHotCompact::from_wire(wire, &cfg)?;
        *slab.transport_quad_hot_mut(q) = hot_q;
        let (has_ext, cap) = compute_quad_ext_requirements(wire, q, zoh.rho_vm, decl_max, pow2);
        let want_f32 = (wire.flags & TRANSPORT_FLAG_EXT_F32) != 0;
        let cold = slab.cold_mut();
        cold.transport_quad_meta[q] = meta_q;
        cold.transport_quad_eff[q] = QuadEffectiveTargets::from_quad_and_meta(&hot_q, &meta_q);
        cold.transport_ext[q].reconfigure_for_reload(has_ext, want_f32, cap)?;
        cold.transport_schroeder[q].reconfigure_sample_rate(wire, zoh.rho_vm)?;
    }

    apply_tier_weights_and_rescale(slab, active_tier, &zoh)?;
    latch_immediate_at_init_or_reset(slab);
    Ok(())
}

/// Owned backing tag for optional `feature = "alloc"` hydration via [`hydrate_slab_owned`].
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SonaSlabOwnedBacking<B: GraphBackend>(PhantomData<B>);

/// Owned hydration with pre-allocated reload headroom (`feature = "alloc"`), enabling zero-alloc
/// [`reload_slab_from_dnam_bin_in_place`] across live model switches (`0` allocs, `0` drops).
#[cfg(feature = "alloc")]
pub fn hydrate_slab_owned_with_headroom<B: GraphBackend>(
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
    headroom: VmReloadHeadroom,
) -> Result<SonaHotSlab<'static, B>, VmError> {
    hydrate_slab_owned_inner(
        view,
        active_tier,
        host_rate_hz,
        offline_bounce,
        headroom,
        false,
    )
}

#[cfg(feature = "alloc")]
fn hydrate_slab_owned_inner<B: GraphBackend>(
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
    headroom: VmReloadHeadroom,
    prealloc_384k: bool,
) -> Result<SonaHotSlab<'static, B>, VmError> {
    crate::math::warm_simd_feature_caches();
    if active_tier.index() > view.retained_tier.index() {
        return Err(VmError::InvalidTier);
    }
    let cold_vec = vec![SonaColdTail::<B>::empty()];
    let mut slab = SonaHotSlab::empty_with_cold(ColdBuf::Owned(cold_vec));

    populate_crossbar_and_core_quads(view, &mut slab);
    decode_readout_and_overrides(view, &mut slab)?;
    setup_transport_wire_slots(view, &mut slab);

    let mode = if offline_bounce {
        RenderMode::Master
    } else {
        RenderMode::Live
    };
    let ref_hz = view.header.reference_sample_rate_hz.get() as f32;
    let zoh = ZohRateContext::compute(ref_hz, host_rate_hz, mode)?;

    let req_contrib = required_cp1_contributors(view)?;
    let ctrl_backing = ControlBusBacking::new_owned_with_capacity(
        view.control_descriptors.len().max(headroom.min_controls),
        req_contrib.max(headroom.min_cp1_contributors).max(8),
        view.tsyn_entries.len().max(headroom.min_tsyn_entries),
        view.tsyn_entries,
    );
    let bus = ControlBus::from_dnam_bin(view, active_tier, zoh.f_s_vm as f32, ctrl_backing)?;
    let fir_ext = ConvolverHighSrFirExt::from_cold_buf(
        ColdBuf::Owned(vec![0.0_f32; EXT_FIR_STORAGE_F32_LEN]),
        zoh.f_s_vm,
        false,
    )?;
    slab.cold_mut().ctrl_bus = Some(bus);
    slab.cold_mut().convolver_fir_ext = fir_ext;
    prepare_common_cold_in_place(view, &mut slab, active_tier, &zoh)?;

    let wires: [TransportQuadWire; TRANSPORT_MAX_QUADS] = slab.cold().wire_transport_quads;
    let (decl_max, pow2, dc5) = (
        slab.cold().declared_max_delay_48k,
        slab.cold().ext_ring_pow2_frames,
        slab.cold().dc_blocker_pole_5hz,
    );
    let alloc_rho = if prealloc_384k {
        let min_rho_384k = f64::from(ref_hz) * crate::math::rcp_nr6_f64(384_000.0);
        zoh.rho_vm.min(min_rho_384k)
    } else {
        zoh.rho_vm
    };
    let disabled_wire = TransportQuadWire::new_zeroed();
    for (q, wire) in wires.iter().enumerate() {
        if q > view.retained_tier.index() {
            slab.cold_mut().transport_ext[q] = TransportExtRing::new_owned_with_headroom(
                &disabled_wire,
                &ExtRingAllocSpec {
                    quad_idx: q,
                    rho_vm: alloc_rho,
                    declared_max_delay_48k: 0,
                    ext_ring_pow2_frames: 0,
                    active_cap: None,
                    min_fp16_frames: headroom.ext_ring_fp16_frames_per_quad,
                    min_f32_frames: headroom.ext_ring_f32_frames_per_quad,
                },
            )?;
            slab.cold_mut().transport_schroeder[q] =
                PartitionedSchroederBank::new_owned_with_capacity(
                    headroom.schroeder_f32_per_quad,
                    &disabled_wire,
                    1.0,
                )?;
            continue;
        }
        let cfg = QuadHydrateConfig {
            quad_idx: q,
            fs_vm: zoh.f_s_vm,
            rho_vm: zoh.rho_vm,
            declared_max_delay_48k: decl_max,
            ext_ring_pow2_frames: pow2,
            dc_blocker_pole_5hz: dc5,
        };
        let (hot_q, meta_q) = TransportQuadHotCompact::from_wire(wire, &cfg)?;
        *slab.transport_quad_hot_mut(q) = hot_q;
        slab.cold_mut().transport_quad_meta[q] = meta_q;
        slab.cold_mut().transport_quad_eff[q] =
            QuadEffectiveTargets::from_quad_and_meta(&hot_q, &meta_q);
        let active_cap =
            ((hot_q.flags & QUAD_FLAG_HAS_EXT) != 0 && meta_q.cap > 0).then_some(meta_q.cap);
        let ext_ring = TransportExtRing::new_owned_with_headroom(
            wire,
            &ExtRingAllocSpec {
                quad_idx: q,
                rho_vm: alloc_rho,
                declared_max_delay_48k: decl_max,
                ext_ring_pow2_frames: pow2,
                active_cap,
                min_fp16_frames: headroom.ext_ring_fp16_frames_per_quad,
                min_f32_frames: headroom.ext_ring_f32_frames_per_quad,
            },
        )?;
        slab.cold_mut().transport_ext[q] = ext_ring;
        let sch_req_alloc = compute_schroeder_requirements(wire, alloc_rho).3;
        let mut sch = PartitionedSchroederBank::new_owned_with_capacity(
            sch_req_alloc.max(headroom.schroeder_f32_per_quad),
            wire,
            alloc_rho,
        )?;
        sch.reconfigure_sample_rate(wire, zoh.rho_vm)?;
        slab.cold_mut().transport_schroeder[q] = sch;
    }

    apply_tier_weights_and_rescale(&mut slab, active_tier, &zoh)?;
    latch_immediate_at_init_or_reset(&mut slab);
    Ok(slab)
}

/// Convenience heap-allocated hydration (`feature = "alloc"`), pre-sizing cold buffers for live
/// `retime_sample_rate` and `set_offline_bounce` transitions without future allocation.
#[cfg(feature = "alloc")]
pub fn hydrate_slab_owned<B: GraphBackend>(
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
) -> Result<SonaHotSlab<'static, B>, VmError> {
    hydrate_slab_owned_inner(
        view,
        active_tier,
        host_rate_hz,
        offline_bounce,
        VmReloadHeadroom::zero(),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydrate_facade_tier_counts() {
        assert_eq!(Tier::Ultra.simd_groups(), 10);
        assert_eq!(Tier::Lite.early_fir_taps(), 32);
    }
}
