//! Lifecycle initialization, sample-rate retiming, preview containment, and DC reset for
//! [`DnamEngine`].

pub mod bounce;
pub mod loader;

use dnam_vm::SonaVm;
use dnam_vm::bin::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_AUTO_UNITY,
    DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, DnamBinView, encode_dnam_bin,
};
use dnam_vm::graph::VmReloadHeadroom;
use dnam_vm::graph::slab::ctrl_latch::latch_immediate_at_init_or_reset;
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::U16;

use super::{DnamEngine, EngineBackend, HostShellPatchbay, TempoSyncState};
use crate::container::{
    ARENA_LITE_END_CL, ARENA_OFFSET_LITE_END, DnamContainerView, DnamPreamble, map_vm_error,
};
use crate::provenance::{CryptoProvider, PreviewGuardState, VerificationStatus};
use crate::schema::{DnamError, DnamProtectionProfile};
use crate::types::Tier;

/// Constructs a valid default identity `.dnam-bin` (`Tier::Lite`..=`max_tier`) for empty shells.
pub(crate) fn build_default_identity_dnam_bin(max_tier: Tier) -> alloc::vec::Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    let tier_mask: u16 = match max_tier {
        Tier::Lite => 0x0001,
        Tier::Full => 0x0003,
        Tier::Pro => 0x0007,
        Tier::Ultra => 0x000F,
    };
    a0.crossbar.cell_desc.tier_flags = U16::new(tier_mask);
    let a1 = Arena1FullDeltaWire::new_zeroed();
    let a2 = Arena2ProDeltaWire::new_zeroed();
    let a3 = Arena3UltraDeltaWire::new_zeroed();
    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: 48_000,
        header_flags: DNBN_FLAG_STEREO_OUT | DNBN_FLAG_AUTO_UNITY,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: dnam_vm::math::TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: 0.0,
        ro_pole_48k: 0.0,
        switch_xfade_samples: 128,
        content_hash_prefix: [0; 8],
        controls: &[],
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: (max_tier >= Tier::Full).then_some(&a1),
        arena2: (max_tier >= Tier::Pro).then_some(&a2),
        arena3: (max_tier >= Tier::Ultra).then_some(&a3),
    })
    .expect("default identity .dnam-bin encodes validly")
}

/// Cold container loader shared across all [`DnamEngine`] `.dnam`, `.dnam-seal`, and `.dnam-prev`
/// constructors.
pub(crate) fn load_engine_with_options<B: EngineBackend, C: CryptoProvider>(
    bytes: &[u8],
    tier: Tier,
    max_supported_tier: Tier,
    sample_rate_hz: u32,
    crypto: &C,
    allow_preview: bool,
    preview_seed: Option<[u8; 32]>,
) -> Result<DnamEngine<'static, B>, DnamError> {
    if !(44_100..=384_000).contains(&sample_rate_hz) {
        return Err(DnamError::InvalidSampleRate(sample_rate_hz));
    }
    let (view, dbin_bytes) = DnamContainerView::from_bytes_with_dbin_options(
        bytes,
        crypto,
        allow_preview,
        max_supported_tier,
    )?;
    if (view.preamble.version_major, view.preamble.version_minor)
        != (
            crate::container::CONTAINER_VERSION_MAJOR,
            crate::container::CONTAINER_VERSION_MINOR,
        )
        || view.header.architecture != "sona"
    {
        return Err(DnamError::UnsupportedVersion {
            major: view.preamble.version_major,
            minor: view.preamble.version_minor,
        });
    }

    let bin_view = DnamBinView::parse(&dbin_bytes).map_err(map_vm_error)?;
    let is_preview = allow_preview || view.header.container_kind == "preview";
    let effective_ceiling = if is_preview {
        Tier::Lite
    } else {
        max_supported_tier
    };
    let max_available_tier = bin_view.retained_tier.min(effective_ceiling);
    let active_tier = tier.min(max_available_tier);
    let bounce_tier = max_available_tier;
    let bounce_sr_hz = sample_rate_hz.min(96_000);

    // Exact capacity on `from_bytes*` keeps small-tier instances tight unless `with_max_capacity()`
    // was explicitly requested.
    let headroom = VmReloadHeadroom::zero();
    let mut vm = alloc::boxed::Box::new(
        SonaVm::<B>::from_view_owned_with_headroom(
            &bin_view,
            active_tier,
            sample_rate_hz as f32,
            false,
            headroom,
        )
        .map_err(map_vm_error)?,
    );
    let mut bounce_vm = alloc::boxed::Box::new(
        SonaVm::<B>::from_view_owned_with_headroom(
            &bin_view,
            bounce_tier,
            bounce_sr_hz as f32,
            true,
            headroom,
        )
        .map_err(map_vm_error)?,
    );
    latch_immediate_at_init_or_reset(vm.slab_mut());
    latch_immediate_at_init_or_reset(bounce_vm.slab_mut());

    let mut patchbay = HostShellPatchbay::new();
    let live_zoh = vm.slab().cold().zoh;
    patchbay.reconfigure_sample_rate(live_zoh.f_s_vm, live_zoh.is_bounce_4x);
    let mut bounce_patchbay = HostShellPatchbay::new();
    let bounce_zoh = bounce_vm.slab().cold().zoh;
    bounce_patchbay.reconfigure_sample_rate(bounce_zoh.f_s_vm, bounce_zoh.is_bounce_4x);

    let single_input = view.header.io_config.inputs == 1;
    let default_blend = view.header.io_config.capabilities.default_blend;
    let tail_time_ms = view.header.io_config.tail_time_ms;
    let seed = preview_seed.unwrap_or_else(|| crate::provenance::sha256_digest(bytes));
    let mut preview_guard = PreviewGuardState::new_from_seed(&seed);
    preview_guard.prepare(sample_rate_hz as f32);

    let mut tempo_state = super::TempoSyncState::new();
    tempo_state.sample_rate_hz = sample_rate_hz;
    let mut engine = DnamEngine {
        _marker: core::marker::PhantomData,
        preamble: view.preamble,
        header: Some(view.header),
        header_active: true,
        reloaded_header_flags: None,
        verification_status: view.verification_status,
        verification_status_active: true,
        protection_profile: view.protection_profile,
        tier: active_tier,
        max_available_tier,
        max_supported_tier: effective_ceiling,
        preallocated_max_tier: max_available_tier,
        offline_bounce: false,
        maximize_tier_on_bounce: true,
        tempo_state,
        bypassed: false,
        trails_enabled: false,
        single_input,
        saturation_flags: 0,
        last_block_max_drive: 0.0,
        tail_time_ms,
        extra_clamp_mask: 0,
        preview_mode: is_preview,
        preview_guard,
        vm,
        bounce_vm,
        patchbay,
        bounce_patchbay,
        mix_dry_gain: 0.0,
        mix_wet_gain: 1.0,
    };
    engine.configure_host_mix_and_leveling(default_blend, 0.0, true);
    let num_ctrls = engine.num_controls();
    for idx in 0..num_ctrls {
        if let Some(&val) = engine.clamped_control_values().get(idx) {
            engine.refresh_calibrated_control_effects(idx, val);
        }
    }
    Ok(engine)
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Creates an unconfigured [`DnamEngine`] shell hydrated from an identity `.dnam-bin`
    /// with live control-plane reload headroom.
    pub fn new_empty(max_supported_tier: Tier) -> Self {
        let headroom =
            VmReloadHeadroom::for_live_engine().with_transport_headroom(1_024, 1_024, 256);
        Self::new_empty_with_headroom(max_supported_tier, max_supported_tier, headroom)
    }

    pub(crate) fn new_empty_with_headroom(
        max_supported_tier: Tier,
        preallocated_max_tier: Tier,
        headroom: VmReloadHeadroom,
    ) -> Self {
        dnam_vm::math::warm_simd_feature_caches();
        let dbin = build_default_identity_dnam_bin(preallocated_max_tier);
        let bin_view = DnamBinView::parse(&dbin).expect("default identity dnam-bin is valid");
        let vm = alloc::boxed::Box::new(
            SonaVm::<B>::from_view_owned_with_headroom(
                &bin_view,
                Tier::Lite,
                48_000.0,
                false,
                headroom,
            )
            .expect("default identity live VM hydrates"),
        );
        let bounce_vm = alloc::boxed::Box::new(
            SonaVm::<B>::from_view_owned_with_headroom(
                &bin_view,
                Tier::Lite,
                48_000.0,
                true,
                headroom,
            )
            .expect("default identity bounce VM hydrates"),
        );
        let preamble = DnamPreamble {
            magic: crate::container::DNAM_MAGIC,
            version_major: crate::container::CONTAINER_VERSION_MAJOR,
            version_minor: crate::container::CONTAINER_VERSION_MINOR,
            flags: crate::container::FLAG_HAS_LITE | crate::container::FLAG_STEREO_OUT,
            num_controls: 0,
            sample_rate_train_hz: 48_000,
            json_header_byte_len: 0,
            tensor_arena_offset: 4096,
            tensor_arena_byte_len: ARENA_OFFSET_LITE_END,
            arena_lite_end_cl: ARENA_LITE_END_CL,
            arena_full_end_cl: ARENA_LITE_END_CL,
            arena_pro_end_cl: ARENA_LITE_END_CL,
            arena_ultra_end_cl: ARENA_LITE_END_CL,
            arena_crc32c_lite: 0,
            arena_crc32c_full: 0,
            arena_crc32c_pro: 0,
            arena_crc32c_ultra: 0,
            _reserved_stems: 0,
            _reserved: 0,
            tail_time_ms: 0.0,
            header_crc32c: 0,
            crc_dbin_prefix: 0,
            dbin_offset: 0,
            dbin_prefix_bytes: 0,
        };
        let mut patchbay = HostShellPatchbay::new();
        patchbay.reconfigure_sample_rate(48_000.0, false);
        let mut bounce_patchbay = HostShellPatchbay::new();
        bounce_patchbay.reconfigure_sample_rate(192_000.0, true);
        Self {
            _marker: core::marker::PhantomData,
            preamble,
            header: None,
            header_active: false,
            reloaded_header_flags: None,
            verification_status: VerificationStatus::Unsigned,
            verification_status_active: true,
            protection_profile: DnamProtectionProfile::Open,
            tier: Tier::Lite,
            max_available_tier: Tier::Lite,
            max_supported_tier,
            preallocated_max_tier,
            offline_bounce: false,
            maximize_tier_on_bounce: true,
            tempo_state: TempoSyncState::new(),
            bypassed: false,
            trails_enabled: false,
            single_input: false,
            saturation_flags: 0,
            last_block_max_drive: 0.0,
            tail_time_ms: 0.0,
            extra_clamp_mask: 0,
            preview_mode: false,
            preview_guard: PreviewGuardState::new_from_seed(&[0u8; 32]),
            vm,
            bounce_vm,
            patchbay,
            bounce_patchbay,
            mix_dry_gain: 0.0,
            mix_wet_gain: 1.0,
        }
    }

    pub(crate) fn reconfigure_for_sample_rate(
        &mut self,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate_rt(sample_rate_hz)
            .map_err(loader::wire_err_to_dnam_rt)
    }

    #[inline]
    pub(crate) fn reconfigure_for_sample_rate_rt(
        &mut self,
        sample_rate_hz: u32,
    ) -> Result<(), crate::container::ContainerWireError> {
        if !(44_100..=384_000).contains(&sample_rate_hz) {
            return Err(crate::container::ContainerWireError::InvalidSampleRate(
                sample_rate_hz,
            ));
        }
        let sr = sample_rate_hz as f32;
        self.vm
            .retime_sample_rate(sr)
            .map_err(crate::container::map_vm_error_rt)?;
        let bounce_sr = sample_rate_hz.min(96_000) as f32;
        self.bounce_vm
            .retime_sample_rate(bounce_sr)
            .map_err(crate::container::map_vm_error_rt)?;
        let live_zoh = self.vm.slab().cold().zoh;
        self.patchbay
            .reconfigure_sample_rate(live_zoh.f_s_vm, live_zoh.is_bounce_4x);
        let bounce_zoh = self.bounce_vm.slab().cold().zoh;
        self.bounce_patchbay
            .reconfigure_sample_rate(bounce_zoh.f_s_vm, bounce_zoh.is_bounce_4x);
        self.preview_guard.prepare(sr);
        self.tempo_state.sample_rate_hz = sample_rate_hz;
        Ok(())
    }

    /// Resets all states to analytical DC resting equilibrium while preserving user knob settings.
    pub fn reset_to_dc_equilibrium(&mut self) {
        let mut saved_raw = [0.0_f32; dnam_vm::bin::MAX_NUM_CONTROLS];
        let mut saved_n = 0_usize;
        if let Some(bus) = self.vm.slab().cold().ctrl_bus.as_ref() {
            saved_n = bus
                .num_controls
                .min(bus.backing.raw_values.len())
                .min(saved_raw.len());
            saved_raw[..saved_n].copy_from_slice(&bus.backing.raw_values[..saved_n]);
        }
        self.vm.reset();
        if saved_n > 0
            && let Some(bus) = self.vm.slab_mut().cold_mut().ctrl_bus.as_mut()
        {
            let copy_n = saved_n.min(bus.backing.raw_values.len());
            bus.backing.raw_values[..copy_n].copy_from_slice(&saved_raw[..copy_n]);
            bus.is_dirty = true;
        }
        self.patchbay.reset();
        self.preview_guard.reset();
        self.tempo_state.reset();
        self.last_block_max_drive = 0.0;
        self.extra_clamp_mask = 0;
        self.sync_immediate_control_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use crate::test_support::build_test_dnam_container;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use dnam_vm::math::ScalarBackend;

    #[test]
    fn test_preview_containment_and_rate_reset_lifecycle() {
        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema.push(ControlSchemaDef {
            id: String::from("drive"),
            name: String::from("Drive"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(1),
            anchor_nodes_norm: Some(alloc::vec![0.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: alloc::vec::Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        });

        let normal_c = build_test_dnam_container(&mut header, Tier::Ultra, true);
        let mut engine = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
            &normal_c,
            Tier::Pro,
            Tier::Pro,
            48_000,
        )
        .unwrap();
        assert_eq!(engine.tier(), Tier::Pro);
        assert!(!engine.is_preview_mode());
        engine.set_sample_rate(96_000).unwrap();
        assert_eq!(engine.sample_rate_hz(), 96_000);
        assert!(engine.set_sample_rate(22_050).is_err());
        engine.reset_to_dc_equilibrium();

        let mut prev_hdr = header.clone();
        prev_hdr.container_kind = String::from("preview");
        prev_hdr
            .stem_user_metadata
            .insert(String::from("stem0"), serde_json::json!({"role": "dry"}));
        let prev_c = build_test_dnam_container(&mut prev_hdr, Tier::Ultra, true);

        // Standard DAW constructors fail-closed on preview containers
        assert!(matches!(
            DnamEngine::<ScalarBackend>::from_bytes(&prev_c, Tier::Lite, 48_000),
            Err(DnamError::PreviewContainerRejected)
        ));

        // Explicit preview constructor caps at Tier::Lite and engages PreviewGuardState
        let mut prev_eng =
            DnamEngine::<ScalarBackend>::from_preview_bytes(&prev_c, Tier::Ultra, 48_000).unwrap();
        assert!(prev_eng.is_preview_mode());
        assert_eq!(prev_eng.tier(), Tier::Lite);
        assert_eq!(prev_eng.effective_tier(), Tier::Lite);
        assert_eq!(prev_eng.max_supported_tier(), Tier::Lite);
        prev_eng.set_tier(Tier::Ultra);
        assert_eq!(prev_eng.tier(), Tier::Lite);
        prev_eng.set_max_supported_tier(Tier::Ultra);
        assert_eq!(prev_eng.max_supported_tier(), Tier::Lite);

        // Impulse probe triggers PreviewGuardState Layer 5 poison engagement in preview mode
        let mut imp = [0.0_f32; 64];
        imp[0] = 0.8;
        let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);
        prev_eng.process_block(&imp, &imp, &mut ol, &mut or);
        assert!(prev_eng.preview_guard().poison_level() > 0.0);
        prev_eng.reset_to_dc_equilibrium();
        assert_eq!(prev_eng.preview_guard().poison_level(), 0.0);
    }

    #[test]
    fn sample_rate_reconfigure_above_96khz_updates_bounce_vm_and_patchbay() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Lite, true);
        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        engine.patchbay.configure_return_alignment_48k(4.0, 4.0);
        engine
            .bounce_patchbay
            .configure_return_alignment_48k(4.0, 4.0);
        assert!((engine.patchbay.return_align_l.delay_samples() - 4.0).abs() < 1.0e-5);
        assert!((engine.bounce_patchbay.return_align_l.delay_samples() - 4.0).abs() < 1.0e-5);

        engine.set_sample_rate(192_000).unwrap();
        let live_zoh = engine.vm.slab().cold().zoh;
        let bounce_zoh = engine.bounce_vm.slab().cold().zoh;
        assert_eq!(live_zoh.host_rate_hz, 192_000.0);
        assert_eq!(live_zoh.f_s_vm, 192_000.0);
        assert_eq!(bounce_zoh.host_rate_hz, 96_000.0);
        assert_eq!(bounce_zoh.f_s_vm, 384_000.0);
        assert!(bounce_zoh.is_bounce_4x);
        assert!((engine.patchbay.return_align_l.delay_samples() - 16.0).abs() < 1.0e-4);
        assert!((engine.bounce_patchbay.return_align_l.delay_samples() - 8.0).abs() < 1.0e-4);

        let in_l = [0.2_f32; 32];
        let in_r = [-0.2_f32; 32];
        let (mut out_l, mut out_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        engine.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|v| v.is_finite()) && out_r.iter().all(|v| v.is_finite()));
    }
}
