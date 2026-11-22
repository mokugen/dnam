#![forbid(unsafe_code)]
//! Shared valid `SONA v3.0` neutral `.dnam` / `Arena 0..3` container helpers for `proj_suite`.

use dnam_engine::container::build_dnam_container_from_dnam_bin;
use dnam_engine::schema::DnamJsonHeader;
use dnam_train::{Pass5EmitConfig, SonaModelF64, encode_model_to_dnam_bin};
use dnam_vm::bin::{
    ARENA_TIER_BYTES, DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, DnamBinView,
    encode_dnam_bin,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::U16;

pub(super) fn neutral_arena_bytes(full_ultra: bool) -> Vec<u8> {
    if !full_ultra {
        return DnamBinBuildInput::default_arena0_lite().as_bytes().to_vec();
    }
    let dbin = encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default())
        .expect("neutral dnam-bin");
    let view = DnamBinView::parse(&dbin).expect("parse neutral dnam-bin");
    let mut out = Vec::with_capacity(ARENA_TIER_BYTES[3]);
    out.extend_from_slice(view.arena0.as_bytes());
    out.extend_from_slice(view.arena1.expect("arena1").as_bytes());
    out.extend_from_slice(view.arena2.expect("arena2").as_bytes());
    out.extend_from_slice(view.arena3.expect("arena3").as_bytes());
    out
}

fn header_dnam_bin_flags(hdr: &DnamJsonHeader) -> u16 {
    let mut flags = 0_u16;
    if hdr.io_config.output_mode != "mono" {
        flags |= DNBN_FLAG_STEREO_OUT;
    }
    if hdr.io_config.inputs == 1 {
        flags |= DNBN_FLAG_SINGLE_INPUT;
    }
    flags
}

pub(super) fn build_test_dnam_from_header(
    hdr: &mut DnamJsonHeader,
    driven_weights: bool,
) -> Vec<u8> {
    let base_dbin =
        encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default())
            .expect("neutral dnam-bin");
    let view = DnamBinView::parse(&base_dbin).expect("parse neutral dnam-bin");
    let mut a0 = *view.arena0;
    let a1 = *view.arena1.expect("arena1");
    let a2 = *view.arena2.expect("arena2");
    let mut a3 = *view.arena3.expect("arena3");
    if driven_weights {
        let bias = U16::new(dnam_kernels::f32_to_fp16(0.5));
        let one = U16::new(dnam_kernels::f32_to_fp16(1.0));
        let half = U16::new(dnam_kernels::f32_to_fp16(0.5));
        a0.crossbar.carrier_basis_w_fp16[0] = [one; 4];
        a0.core_quads[0].adaa_drive_fp16 = [one; 4];
        a0.core_quads[0].lru_bias_drive_fp16 = [bias; 4];
        a0.core_quads[0].tone_dir_fp16 = [one; 4];
        a0.linear_readout_lite.w_ro_fp16[..192].fill(half);
        a0.linear_readout_lite.w_ro_fp16[192..208].fill(one);
        a3.crossbar_ultra.phasor_to_core_fp16.fill([one; 4]);
        a3.core_quads[0].adaa_drive_fp16 = [one; 4];
        a3.core_quads[0].lru_bias_drive_fp16 = [bias; 4];
        a3.core_quads[0].tone_dir_fp16 = [one; 4];
        a3.linear_readout_ultra.w_ro_and_mods_fp16[..480].fill(half);
    }
    let compiled = hdr.compile_clamp_descriptors().expect("compile clamps");
    let wire_descs: Vec<_> = compiled.iter().map(|d| d.to_wire()).collect();
    let dbin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: hdr.sample_rate_train_hz,
        header_flags: header_dnam_bin_flags(hdr),
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 16,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: hdr.io_config.tail_time_ms,
        ro_pole_48k: hdr.linear_readout.ro_pole_48k,
        switch_xfade_samples: 64,
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: &wire_descs,
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
    .expect("encode test dnam-bin");
    build_dnam_container_from_dnam_bin(hdr, &dbin).expect("build test .dnam")
}

pub(super) fn try_encode_invalid_stability_dnam_bin(
    hdr: &DnamJsonHeader,
    corrupt_lite: bool,
) -> Result<Vec<u8>, dnam_vm::VmError> {
    let base_dbin =
        encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default())
            .expect("neutral dnam-bin");
    let view = DnamBinView::parse(&base_dbin).expect("parse neutral dnam-bin");
    let mut a0 = *view.arena0;
    let a1 = *view.arena1.expect("arena1");
    let a2 = *view.arena2.expect("arena2");
    let mut a3 = *view.arena3.expect("arena3");
    let bad_k_nl = [U16::new(dnam_kernels::f32_to_fp16(1.0)); 4];
    if corrupt_lite {
        a0.core_quads[1].svf_k_nl_fp16 = bad_k_nl;
    } else {
        a3.core_quads[1].svf_k_nl_fp16 = bad_k_nl;
    }
    let compiled = hdr.compile_clamp_descriptors().expect("compile clamps");
    let wire_descs: Vec<_> = compiled.iter().map(|d| d.to_wire()).collect();
    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: hdr.sample_rate_train_hz,
        header_flags: header_dnam_bin_flags(hdr),
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 16,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: hdr.io_config.tail_time_ms,
        ro_pole_48k: hdr.linear_readout.ro_pole_48k,
        switch_xfade_samples: 64,
        content_hash_prefix: view.runtime_meta.content_hash_prefix,
        controls: &wire_descs,
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
}
