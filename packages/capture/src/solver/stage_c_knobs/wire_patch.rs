#![forbid(unsafe_code)]
//! Wire patching, zero-ULP `"CEXT"` gauge enforcement, and dynamic state reset on `.dnam-bin`
//! for Stage C (`solver::stage_c_knobs::wire_patch`).

use dnam_engine::schema::DnamError;
use dnam_train::MIN_DELAY_CROSSBAR_SAMP;
use dnam_train::control::{Cp1WireBundle, Phase1ClampOutput, enforce_arena0_stability_invariants};
use dnam_vm::Tier;
use dnam_vm::bin::{
    CextDispatchWire, Cp1InlineDirWire, DnamBinBuildInput, DnamBinView,
    SONA_META_CORE_MODS_PRESENT, encode_dnam_bin,
};
use dnam_vm::control::Cp1ContributorSlot;
use dnam_vm::control::MAX_EXACT_CONTRIBUTORS;
use dnam_vm::control::cp1::decode_cp1_contributors;
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16, U32};

use super::map_stage_err;
use crate::proj::verify_dnam_bin_zero_dc;

fn zero_control_bus_fields(
    a0: &mut dnam_vm::bin::Arena0LiteWire,
    a3: Option<&mut dnam_vm::bin::Arena3UltraDeltaWire>,
) {
    a0.control_bus_lite.u_int_base = [[F32::new(0.0); 8]; 4];
    a0.control_bus_lite.w_ctrl = [[F32::new(0.0); 16]; 16];
    a0.control_bus_lite.cp1_dir = Cp1InlineDirWire::new_zeroed();
    a0.control_bus_lite.cp1_base.fill(0);
    a0.control_bus_lite.lfo_dphi_train = [F32::new(0.0); 2];
    a0.control_bus_lite.lfo_skew_fp16 = [U16::new(0); 2];
    a0.control_bus_lite.ou_to_z_stage_fp16 = [U16::new(0); 4];
    a0.control_bus_lite.sigma_drift = F32::new(0.0);
    a0.linear_readout_lite.stage_latent_mods_fp16[8..80].fill(U16::new(0));
    let cleared_flags = a0.crossbar.sona_meta_flags.get() & !SONA_META_CORE_MODS_PRESENT;
    a0.crossbar.sona_meta_flags = U32::new(cleared_flags);
    a0.crossbar.cell_desc.cell_flags = U32::new(cleared_flags);

    if let Some(a3w) = a3 {
        a3w.control_bus_ultra_delta.w_ctrl_ultra = [[F32::new(0.0); 4]; 16];
        a3w.control_bus_ultra_delta.u_int_ultra = [[[F32::new(0.0); 8]; 4]; 3];
        a3w.control_bus_ultra_delta.cp1_ultra.fill(0);
        a3w.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[..164].fill(U16::new(0));
        a3w.stage_latent_core_mods_fp16.fill(U16::new(0));
    }
}

/// Clears all dynamic `ControlBus`, `CP1`, `U_int`, `"CEXT"`, and `stage_latent_*` tables on
/// `dnam_bin` ahead of Stage A/B re-pooling when `clear_dynamic_w_ctrl` is enabled.
pub(crate) fn clear_dnam_bin_control_bus_state(dnam_bin: &mut Vec<u8>) -> Result<(), DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let mut a0 = *view.arena0;
    let a1 = view.arena1.copied();
    let a2 = view.arena2.copied();
    let mut a3 = view.arena3.copied();
    let mut ro_pole = view.runtime_meta.ro_pole_48k.get();

    zero_control_bus_fields(&mut a0, a3.as_mut());
    enforce_arena0_stability_invariants(&mut a0, &mut ro_pole);

    *dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
        header_flags: view.header.flags.get(),
        ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: view.runtime_meta.ext_ring_lines,
        ext_ring_format: view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: ro_pole,
        switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: view.control_descriptors,
        cp1_overflow: &[],
        tsyn: view.tsyn_entries,
        cext: None,
        arena0: &a0,
        arena1: a1.as_ref(),
        arena2: a2.as_ref(),
        arena3: a3.as_ref(),
    })
    .map_err(map_stage_err)?;
    verify_dnam_bin_zero_dc(dnam_bin)?;
    Ok(())
}

pub(super) fn patch_dnam_bin_control_bus(
    dnam_bin: &mut Vec<u8>,
    phase1: Option<&Phase1ClampOutput>,
    bundle_opt: Option<&Cp1WireBundle>,
    clear_dynamic_w_ctrl: bool,
) -> Result<Vec<Cp1ContributorSlot>, DnamError> {
    let view = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let mut a0 = *view.arena0;
    let a1 = view.arena1.copied();
    let a2 = view.arena2.copied();
    let mut a3 = view.arena3.copied();
    let mut ro_pole = view.runtime_meta.ro_pole_48k.get();

    if clear_dynamic_w_ctrl {
        zero_control_bus_fields(&mut a0, a3.as_mut());
    }

    if let Some(bundle) = bundle_opt {
        a0.control_bus_lite.trust_radius_mh = bundle.arena0_cb.trust_radius_mh;
        a0.control_bus_lite.cp1_dir = bundle.arena0_cb.cp1_dir;
        a0.control_bus_lite.u_int_base = bundle.arena0_cb.u_int_base;
        a0.control_bus_lite.cp1_base = bundle.arena0_cb.cp1_base;
        for r in 0..16 {
            for c in 0..8 {
                a0.control_bus_lite.w_ctrl[r][c] = bundle.arena0_cb.w_ctrl[r][c];
            }
        }
        if let Some(ref mut a3w) = a3 {
            a3w.control_bus_ultra_delta.w_ctrl_ultra = bundle.arena3_cb.w_ctrl_ultra;
            let has_non_zero_ultra_uint = bundle
                .arena3_cb
                .u_int_ultra
                .iter()
                .flatten()
                .flatten()
                .any(|v| v.get() != 0.0);
            if clear_dynamic_w_ctrl || has_non_zero_ultra_uint {
                a3w.control_bus_ultra_delta.u_int_ultra = bundle.arena3_cb.u_int_ultra;
            }
            a3w.control_bus_ultra_delta.cp1_ultra = bundle.arena3_cb.cp1_ultra;
        }
    }

    enforce_arena0_stability_invariants(&mut a0, &mut ro_pole);

    let wire_descs: &[dnam_vm::bin::ControlClampDescriptorWire] =
        phase1.map_or(view.control_descriptors, |p| p.wire_descriptors.as_slice());
    let cp1_overflow = bundle_opt.map_or(&[][..], |b| b.cp1_overflow.as_slice());
    let tsyn = bundle_opt
        .filter(|b| !b.tsyn.is_empty())
        .map_or(view.tsyn_entries, |b| b.tsyn.as_slice());
    let mut merged_cext: Option<CextDispatchWire> = if clear_dynamic_w_ctrl && bundle_opt.is_none()
    {
        None
    } else if let Some(b_cext) = bundle_opt.and_then(|b| b.cext.as_ref()) {
        let mut cext = *b_cext;
        if !clear_dynamic_w_ctrl && let Some(prev) = view.cext {
            for r in 0..15 {
                for col in 8..16 {
                    cext.w_ctrl_ext[r][col] = prev.w_ctrl_ext[r][col];
                }
            }
        }
        Some(cext)
    } else {
        view.cext.copied()
    };
    if let Some(ref mut cext) = merged_cext {
        for r in 0..16 {
            for col in 0..4 {
                cext.w_ctrl_ext[r][col] = F32::new(0.0);
            }
        }
        cext.w_ctrl_ext[15] = [F32::new(0.0); 16];
        cext.w_ctrl_ultra_ext[15] = [F32::new(0.0); 4];
    }

    let reencoded = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
        header_flags: view.header.flags.get(),
        ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: view.runtime_meta.ext_ring_lines,
        ext_ring_format: view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: ro_pole,
        switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: wire_descs,
        cp1_overflow,
        tsyn,
        cext: merged_cext.as_ref(),
        arena0: &a0,
        arena1: a1.as_ref(),
        arena2: a2.as_ref(),
        arena3: a3.as_ref(),
    })
    .map_err(map_stage_err)?;
    *dnam_bin = reencoded;
    verify_dnam_bin_zero_dc(dnam_bin)?;

    let post = DnamBinView::parse(dnam_bin).map_err(map_stage_err)?;
    let n_ctrl = usize::from(post.header.num_controls.get());
    let mut decoded = vec![Cp1ContributorSlot::zero(); MAX_EXACT_CONTRIBUTORS];
    let count = decode_cp1_contributors(
        &post.arena0.control_bus_lite,
        post.arena3.map(|w| &w.control_bus_ultra_delta),
        post.cp1_overflow,
        n_ctrl,
        &mut decoded,
    )
    .map_err(map_stage_err)?;
    decoded.truncate(count);
    Ok(decoded)
}

pub(super) fn extract_view_delay_and_g_loop(view: &DnamBinView<'_>, tier: Tier) -> (f64, f64) {
    let mut quads = vec![&view.arena0.transport_lite];
    if tier >= Tier::Full
        && let Some(a1) = view.arena1
    {
        quads.push(&a1.transport_full);
    }
    if tier >= Tier::Pro
        && let Some(a2) = view.arena2
    {
        quads.push(&a2.transport_pro[0]);
    }
    if tier >= Tier::Ultra
        && let Some(a3) = view.arena3
    {
        quads.push(&a3.transport_ultra[0]);
    }
    let min_tau = quads
        .iter()
        .flat_map(|q| q.delay_samples.iter().map(|d| f64::from(d.get())))
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let max_g = quads
        .iter()
        .map(|q| f64::from(q.g_loop.get()))
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
        .clamp(0.0, 0.9990);
    (
        if min_tau.is_finite() {
            min_tau.clamp(MIN_DELAY_CROSSBAR_SAMP, 48_000.0)
        } else {
            16.0
        },
        max_g,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;

    #[test]
    fn test_clear_dnam_bin_control_bus_state_zeroes_dynamics_and_cext() {
        let mut dbin = initialize_backbone_dnam_bin_if_needed(None).unwrap();
        let v = DnamBinView::parse(&dbin).unwrap();
        let mut a0 = *v.arena0;
        let mut a3 = *v.arena3.unwrap();
        a0.control_bus_lite.w_ctrl[0][0] = F32::new(1.5);
        a0.control_bus_lite.lfo_dphi_train[0] = F32::new(0.02);
        a0.linear_readout_lite.stage_latent_mods_fp16[0] = U16::new(0x3C00); // default_knobs kept
        a0.linear_readout_lite.stage_latent_mods_fp16[12] = U16::new(0x3C00); // cleared
        a3.control_bus_ultra_delta.w_ctrl_ultra[2][1] = F32::new(-0.5);
        let mut cext = CextDispatchWire::new_zeroed();
        cext.w_ctrl_ext[0][4] = F32::new(0.75);

        dbin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: v.header.reference_sample_rate_hz.get(),
            header_flags: v.header.flags.get(),
            ext_ring_pow2_frames: v.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: v.runtime_meta.ext_ring_lines,
            ext_ring_format: v.runtime_meta.ext_ring_format,
            declared_max_delay_48k: v.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: v.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: v.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: v.runtime_meta.ro_pole_48k.get(),
            switch_xfade_samples: v.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: v.runtime_meta.content_hash_prefix,
            controls: v.control_descriptors,
            cp1_overflow: &[],
            tsyn: v.tsyn_entries,
            cext: Some(&cext),
            arena0: &a0,
            arena1: v.arena1,
            arena2: v.arena2,
            arena3: Some(&a3),
        })
        .unwrap();

        clear_dnam_bin_control_bus_state(&mut dbin).expect("clear state succeeds");
        let after = DnamBinView::parse(&dbin).unwrap();
        assert_eq!(after.arena0.control_bus_lite.w_ctrl[0][0].get(), 0.0);
        assert_eq!(after.arena0.control_bus_lite.lfo_dphi_train[0].get(), 0.0);
        assert_eq!(
            after.arena0.linear_readout_lite.stage_latent_mods_fp16[0].get(),
            0x3C00
        );
        assert_eq!(
            after.arena0.linear_readout_lite.stage_latent_mods_fp16[12].get(),
            0
        );
        assert_eq!(
            after.arena3.unwrap().control_bus_ultra_delta.w_ctrl_ultra[2][1].get(),
            0.0
        );
        assert!(after.cext.is_none());
    }
}
