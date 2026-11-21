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

use dnam_kernels::{
    HilbertH1StereoKernel, compute_t1_dc_blocker_coeffs_clamped, interpolate_hilbert_h1_alphas_f64,
};
#[cfg(feature = "alloc")]
use zerocopy::FromBytes;

use crate::bin::DnamBinView;
use crate::bin::arena_wire::TANK_FLAG_SCHROEDER_ENABLE;
#[cfg(feature = "alloc")]
use crate::bin::{Cp1StreamHeaderWire, TankQuadWire};
use crate::control::ControlBus;
#[cfg(feature = "alloc")]
use crate::control::ControlBusBacking;
use crate::graph::GraphBackend;
use crate::graph::cabinet::high_sr::CabinetHighSrFirExt;
#[cfg(feature = "alloc")]
use crate::graph::cabinet::high_sr::EXT_FIR_STORAGE_F32_LEN;
use crate::graph::oversample::Bounce4xOversampler;
use crate::graph::router::IoRouter;
use crate::graph::tank::{
    PartitionedSchroederBank, QuadEffectiveTargets, QuadHydrateConfig, TANK_MAX_QUADS, TankExtRing,
    TankQuadHotCompact, compute_quad_ext_requirements, compute_schroeder_requirements,
};
use crate::graph::zoh::ZohRateContext;
use crate::types::{ColdBuf, Tier, VmError};

use super::ctrl_latch::latch_immediate_at_init_or_reset;
use super::layout::{FLAG_COLD_FIR_EXT, FLAG_OS2X, SonaColdTail, SonaHotSlab, SonaSlabBacking};
use apply_tier::{apply_tier_weights_and_rescale, populate_crossbar_and_groups};
use lifecycle::setup_phasor_increments;
use wire_decode::{TANK_FLAG_EXT_F32, decode_readout_and_overrides};

pub use lifecycle::{
    reset_to_dc_equilibrium, retime_sample_rate, set_offline_bounce, set_op_t_write_enable,
    switch_tier,
};

fn setup_tank_wire_slots<B: GraphBackend>(view: &DnamBinView<'_>, slab: &mut SonaHotSlab<'_, B>) {
    let mut quads = [view.arena0.spatial_tank_lite; TANK_MAX_QUADS];
    if let Some(a1) = view.arena1 {
        quads[1] = a1.spatial_tank_full;
    }
    if let Some(a2) = view.arena2 {
        quads[2] = a2.spatial_tank_pro[0];
    }
    if let Some(a3) = view.arena3 {
        quads[3] = a3.spatial_tank_ultra[0];
    }
    slab.cold_mut().wire_tank_quads = quads;
    slab.cold_mut().declared_max_delay_48k = view.runtime_meta.declared_max_delay_48k.get();
    slab.cold_mut().ext_ring_pow2_frames = view.runtime_meta.ext_ring_pow2_frames;
    slab.cold_mut().dc_blocker_pole_5hz = view.runtime_meta.dc_blocker_pole_5hz.get();
}

fn prepare_common_cold<'a, B: GraphBackend>(
    view: &DnamBinView<'_>,
    slab: &mut SonaHotSlab<'a, B>,
    active_tier: Tier,
    zoh: &ZohRateContext,
    mut bus: ControlBus<'a, B>,
    mut fir_ext: CabinetHighSrFirExt<'a>,
) -> Result<(), VmError> {
    let offline = zoh.is_bounce_4x;
    bus.offline_bounce = offline;
    let active_taps = active_tier.early_fir_taps();
    fir_ext.prepare(
        &slab.cold().tier_fir_l[active_tier.index()][..active_taps],
        &slab.cold().tier_fir_r[active_tier.index()][..active_taps],
        zoh.f_s_vm,
        offline,
    )?;

    let use_cold_fir = CabinetHighSrFirExt::required_f32_len(zoh.f_s_vm, offline) > 0;
    slab.set_flag(FLAG_COLD_FIR_EXT, use_cold_fir);
    slab.set_flag(FLAG_OS2X, zoh.os2x_active);
    slab.lite.tier = active_tier;

    let (ai, aq) = interpolate_hilbert_h1_alphas_f64(zoh.f_s_vm);
    slab.lite.hilbert = HilbertH1StereoKernel::with_coeffs(ai, aq);
    let (dc_pole, _) = compute_t1_dc_blocker_coeffs_clamped(zoh.f_s_vm);
    slab.lite.t1_dc.set_pole(dc_pole);

    let cb = slab.cold().wire_crossbar;
    let cold = slab.cold_mut();
    cold.max_present_tier = Tier::from_u8(view.header.max_present_tier);
    cold.retained_tier = view.retained_tier;
    cold.zoh = *zoh;
    cold.offline_bounce = offline;
    cold.ctrl_bus = Some(bus);
    cold.cabinet_fir_ext = fir_ext;
    cold.router = IoRouter::from_crossbar_wire(&cb, zoh);
    cold.bounce4x = Bounce4xOversampler::new();
    setup_phasor_increments(slab, zoh);
    Ok(())
}

fn hydrate_tank_borrowed<'a, B: GraphBackend>(
    slab: &mut SonaHotSlab<'a, B>,
    zoh: &ZohRateContext,
    ext_fp16: [&'a mut [u16]; TANK_MAX_QUADS],
    ext_f32: [&'a mut [f32]; TANK_MAX_QUADS],
    schroeder_bufs: [&'a mut [f32]; TANK_MAX_QUADS],
) -> Result<(), VmError> {
    let [fp0, fp1, fp2, fp3] = ext_fp16;
    let [f32_0, f32_1, f32_2, f32_3] = ext_f32;
    let [sch0, sch1, sch2, sch3] = schroeder_bufs;
    let fp_slots = [fp0, fp1, fp2, fp3];
    let f32_slots = [f32_0, f32_1, f32_2, f32_3];
    let sch_slots = [sch0, sch1, sch2, sch3];

    let wires = slab.cold().wire_tank_quads;
    let (decl_max, pow2, dc5) = (
        slab.cold().declared_max_delay_48k,
        slab.cold().ext_ring_pow2_frames,
        slab.cold().dc_blocker_pole_5hz,
    );
    let active_q = slab.cold().retained_tier.index() + 1;

    for (q, ((fp_s, f32_s), sch_s)) in fp_slots
        .into_iter()
        .zip(f32_slots)
        .zip(sch_slots)
        .enumerate()
    {
        if q >= active_q {
            slab.cold_mut().tank_ext[q] = TankExtRing::none();
            slab.cold_mut().tank_schroeder[q] = PartitionedSchroederBank::disabled();
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
        let (hot_q, meta_q) = TankQuadHotCompact::from_wire(&wires[q], &cfg)?;
        *slab.quad_hot_mut(q) = hot_q;
        slab.cold_mut().tank_quad_meta[q] = meta_q;
        slab.cold_mut().tank_quad_eff[q] =
            QuadEffectiveTargets::from_quad_and_meta(&hot_q, &meta_q);

        let (has_ext, cap) =
            compute_quad_ext_requirements(&wires[q], q, zoh.rho_vm, decl_max, pow2);
        slab.cold_mut().tank_ext[q] = if !has_ext {
            TankExtRing::none()
        } else if (wires[q].flags & TANK_FLAG_EXT_F32) != 0 {
            TankExtRing::from_f32_buf(ColdBuf::borrowed(f32_s), cap)?
        } else {
            TankExtRing::from_fp16_buf(ColdBuf::borrowed(fp_s), cap)?
        };
        let (_, _, _, sch_req) = compute_schroeder_requirements(&wires[q], zoh.rho_vm);
        slab.cold_mut().tank_schroeder[q] =
            if (wires[q].flags & TANK_FLAG_SCHROEDER_ENABLE) == 0 || sch_req == 0 {
                PartitionedSchroederBank::disabled()
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

    populate_crossbar_and_groups(view, &mut slab);
    decode_readout_and_overrides(view, &mut slab)?;
    setup_tank_wire_slots(view, &mut slab);

    let tier_os2x = slab.cold().tier_os2x[active_tier.index()];
    let ref_hz = view.header.reference_sample_rate_hz.get() as f32;
    let zoh =
        ZohRateContext::compute(ref_hz, host_rate_hz, active_tier, tier_os2x, offline_bounce)?;
    let bus = ControlBus::from_dnam_bin(view, active_tier, zoh.f_s_vm as f32, backing.control_bus)?;
    let fir_ext = if backing.cabinet_fir_ext.is_empty() {
        CabinetHighSrFirExt::from_slice(&mut [], 48_000.0, false)?
    } else {
        CabinetHighSrFirExt::from_slice(backing.cabinet_fir_ext, zoh.f_s_vm, offline_bounce)?
    };
    prepare_common_cold(view, &mut slab, active_tier, &zoh, bus, fir_ext)?;
    hydrate_tank_borrowed(
        &mut slab,
        &zoh,
        backing.tank_ext_fp16,
        backing.tank_ext_f32,
        backing.tank_schroeder,
    )?;
    apply_tier_weights_and_rescale(&mut slab, active_tier, &zoh)?;
    latch_immediate_at_init_or_reset(&mut slab);
    Ok(slab)
}

/// Owned backing tag for optional `feature = "alloc"` hydration via [`hydrate_slab_owned`].
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SonaSlabOwnedBacking<B: GraphBackend>(PhantomData<B>);

/// Convenience heap-allocated hydration (`feature = "alloc"`), pre-sizing cold buffers for live
/// `retime_sample_rate` and `set_offline_bounce` transitions without future allocation.
#[cfg(feature = "alloc")]
pub fn hydrate_slab_owned<B: GraphBackend>(
    view: &DnamBinView<'_>,
    active_tier: Tier,
    host_rate_hz: f32,
    offline_bounce: bool,
) -> Result<SonaHotSlab<'static, B>, VmError> {
    crate::math::warm_simd_feature_caches();
    if active_tier.index() > view.retained_tier.index() {
        return Err(VmError::InvalidTier);
    }
    let cold_vec = vec![SonaColdTail::<B>::empty()];
    let mut slab = SonaHotSlab::empty_with_cold(ColdBuf::Owned(cold_vec));

    populate_crossbar_and_groups(view, &mut slab);
    decode_readout_and_overrides(view, &mut slab)?;
    setup_tank_wire_slots(view, &mut slab);

    let tier_os2x = slab.cold().tier_os2x[active_tier.index()];
    let ref_hz = view.header.reference_sample_rate_hz.get() as f32;
    let zoh =
        ZohRateContext::compute(ref_hz, host_rate_hz, active_tier, tier_os2x, offline_bounce)?;

    let ovf_count = if view.cp1_overflow.len() >= 8 {
        Cp1StreamHeaderWire::read_from_prefix(view.cp1_overflow)
            .map(|(h, _)| usize::from(h.count.get()))
            .unwrap_or(0)
    } else {
        0
    };
    let max_contrib = usize::from(view.arena0.control_bus_lite.cp1_dir.cp1_dir_count.get())
        .max(ovf_count)
        .max(8);
    let ctrl_backing = ControlBusBacking::new_owned(
        view.control_descriptors.len(),
        max_contrib,
        view.tsyn_entries,
    );
    let bus = ControlBus::from_dnam_bin(view, active_tier, zoh.f_s_vm as f32, ctrl_backing)?;
    let fir_ext = CabinetHighSrFirExt::from_cold_buf(
        ColdBuf::Owned(vec![0.0_f32; EXT_FIR_STORAGE_F32_LEN]),
        zoh.f_s_vm,
        offline_bounce,
    )?;
    prepare_common_cold(view, &mut slab, active_tier, &zoh, bus, fir_ext)?;

    let wires: [TankQuadWire; TANK_MAX_QUADS] = slab.cold().wire_tank_quads;
    let (decl_max, pow2, dc5) = (
        slab.cold().declared_max_delay_48k,
        slab.cold().ext_ring_pow2_frames,
        slab.cold().dc_blocker_pole_5hz,
    );
    let min_rho_384k =
        f64::from(view.header.reference_sample_rate_hz.get()) * crate::math::rcp_nr6_f64(384_000.0);
    let alloc_rho = zoh.rho_vm.min(min_rho_384k).min(0.125);
    for (q, wire) in wires.iter().enumerate() {
        if q > view.retained_tier.index() {
            slab.cold_mut().tank_ext[q] = TankExtRing::none();
            slab.cold_mut().tank_schroeder[q] = PartitionedSchroederBank::disabled();
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
        let (hot_q, meta_q) = TankQuadHotCompact::from_wire(wire, &cfg)?;
        *slab.quad_hot_mut(q) = hot_q;
        slab.cold_mut().tank_quad_meta[q] = meta_q;
        slab.cold_mut().tank_quad_eff[q] =
            QuadEffectiveTargets::from_quad_and_meta(&hot_q, &meta_q);
        let mut ext_ring = TankExtRing::new_owned_for_quad(wire, q, alloc_rho, decl_max, pow2)?;
        if (hot_q.flags & dnam_kernels::QUAD_FLAG_HAS_EXT) != 0 {
            ext_ring.reconfigure_cap(meta_q.cap)?;
        }
        slab.cold_mut().tank_ext[q] = ext_ring;
        let (_, _, _, sch_req) = compute_schroeder_requirements(wire, alloc_rho);
        slab.cold_mut().tank_schroeder[q] =
            if (wire.flags & TANK_FLAG_SCHROEDER_ENABLE) == 0 || sch_req == 0 {
                PartitionedSchroederBank::disabled()
            } else {
                PartitionedSchroederBank::from_cold_buf(
                    ColdBuf::Owned(vec![0.0_f32; sch_req]),
                    wire,
                    zoh.rho_vm,
                )?
            };
    }

    apply_tier_weights_and_rescale(&mut slab, active_tier, &zoh)?;
    latch_immediate_at_init_or_reset(&mut slab);
    Ok(slab)
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
