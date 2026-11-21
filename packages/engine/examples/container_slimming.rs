//! Example 4: Container Tier Slimming, In-Place `DMU1` Mutable Suffix Editing & `DMU1` Framing
//! (`examples/container_slimming.rs`).
//!
//! Demonstrates:
//! 1. Stripping an `Ultra` (`30,720 B` arena) hybrid `.dnam` container down to `Pro` (`15,872 B`),
//!    `Full` (`7,936 B`), and `Lite` (`4,608 B`) via [`strip_container_to_tier`] without
//!    modifying the immutable JSON header or embedded `dbin_prefix`, while preserving zero-copy
//!    [`extract_dnam_bin`] extraction at every stripped tier (`REQ-5.6`).
//! 2. Editing mutable `presets` and `user_metadata` in place within the `DMU1` mutable suffix
//!    trailer via [`update_header_in_place`].
//! 3. Round-tripping the `v3.0` `DMU1` mutable suffix trailer (`build_dmu1_mutable_frame` /
//!    `parse_and_validate_dmu1_frame`, `ref[container.dnam.mutable-frame]`).
//!
//! Run with:
//!   `cargo run -p dnam-engine --example container_slimming`

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{
    DnamContainerView, build_dmu1_mutable_frame, build_dnam_container_from_dnam_bin,
    extract_dnam_bin, parse_and_validate_dmu1_frame, strip_container_to_tier,
    update_header_in_place,
};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{DnamJsonHeader, PresetDef};
use dnam_vm::bin::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_AUTO_UNITY,
    DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, DnamBinView, encode_dnam_bin,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16};

fn build_ultra_hybrid_container(header: &mut DnamJsonHeader) -> Vec<u8> {
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
    let a1 = Arena1FullDeltaWire::new_zeroed();
    let a2 = Arena2ProDeltaWire::new_zeroed();
    let a3 = Arena3UltraDeltaWire::new_zeroed();

    let dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: header.sample_rate_train_hz,
        header_flags: DNBN_FLAG_STEREO_OUT | DNBN_FLAG_AUTO_UNITY,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: header.io_config.tail_time_ms,
        ro_pole_48k: header.linear_readout.ro_pole_48k,
        switch_xfade_samples: header.transport.switch_xfade_samples,
        content_hash_prefix: [0; 8],
        controls: &[],
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
    .expect("valid Layer 0 .dnam-bin");

    build_dnam_container_from_dnam_bin(header, &dnam_bin).expect("valid hybrid .dnam container")
}

fn main() {
    let mut header = DnamJsonHeader::default_reference();

    // 1. Build an Ultra (30,720 B arena) hybrid .dnam container and strip to Pro, Full, and Lite
    let ultra_container = build_ultra_hybrid_container(&mut header);
    let crypto = DefaultCryptoProvider::default();

    for tier in [Tier::Ultra, Tier::Pro, Tier::Full, Tier::Lite] {
        let stripped = strip_container_to_tier(&ultra_container, tier).unwrap();
        let view = DnamContainerView::from_bytes(&stripped, &crypto).unwrap();
        let extracted_dbin = extract_dnam_bin(&stripped).expect("zero-copy .dnam-bin extraction");
        let dbin_view = DnamBinView::parse(extracted_dbin).expect("valid extracted .dnam-bin");
        assert_eq!(dbin_view.retained_tier, tier);
        println!(
            "Stripped Tier {:<5} -> Total File: {:>6} B | Arena Prefix: {:>6} B | Extracted .dnam-bin: {:>6} B | Highest Tier: {:?}",
            tier.as_str(),
            stripped.len(),
            view.arena_storage.as_bytes().len(),
            extracted_dbin.len(),
            view.preamble.highest_tier(),
        );
    }

    // 2. Edit user_metadata and presets in place within the DMU1 mutable trailer
    let mut editable_container = ultra_container;
    header
        .user_metadata
        .insert("favorite".to_string(), serde_json::json!(true));
    header.presets.push(PresetDef {
        id: "studio_clean".to_string(),
        name: "Studio Clean".to_string(),
        description: Some("Added in-place without touching tensor arena".to_string()),
        controls: BTreeMap::new(),
        user_metadata: BTreeMap::new(),
    });
    update_header_in_place(&mut editable_container, &header)
        .expect("in-place DMU1 mutable trailer update succeeded");

    let updated_view = DnamContainerView::from_bytes(&editable_container, &crypto).unwrap();
    println!(
        "In-place header edit verified | Presets count: {} | user_metadata['favorite'] = {}",
        updated_view.header.presets.len(),
        updated_view.header.user_metadata.get("favorite").unwrap(),
    );

    // 3. Round-trip a 64B-aligned DMU1 mutable trailer frame
    let dmu1_json = br#"{"presets":[],"user_metadata":{"tag":"warm"}}"#;
    let dmu1_frame = build_dmu1_mutable_frame(dmu1_json, 64).expect("valid DMU1 frame");
    let parsed_dmu1 = parse_and_validate_dmu1_frame(&dmu1_frame).expect("verified DMU1 CRC32C");
    println!(
        "DMU1 frame length: {} B (64B aligned) | Parsed tag: {}",
        dmu1_frame.len(),
        parsed_dmu1["user_metadata"]["tag"],
    );
}
