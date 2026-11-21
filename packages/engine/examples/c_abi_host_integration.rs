//! Example 6: C-ABI Host Plugin & Embedded Firmware FFI Lifecycle (`examples/c_abi_host_integration.rs`).
//!
//! Demonstrates driving [`dnam_engine`] exclusively through its zero-allocation `extern "C"`
//! FFI surface (`dnam_engine_*`), as used by C/C++ VST3/AU/CLAP wrappers, iOS/Android native
//! audio threads, and WebAssembly `AudioWorkletProcessor` hosts.
//!
//! Run with:
//!   `cargo run -p dnam-engine --example c_abi_host_integration`

use dnam_engine::DnamTelemetrySnapshot;
use dnam_engine::container::build_dnam_container_from_dnam_bin;
use dnam_engine::ffi::{
    dnam_engine_create_with_max_tier, dnam_engine_destroy, dnam_engine_get_latency_samples,
    dnam_engine_get_telemetry, dnam_engine_process_block, dnam_engine_process_mono,
    dnam_engine_reset, dnam_engine_set_control_value, dnam_engine_set_offline_bounce,
    dnam_engine_set_sample_rate, dnam_engine_set_tier,
};
use dnam_engine::schema::DnamJsonHeader;
use dnam_kernels::f32_to_fp16;
use dnam_vm::bin::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_AUTO_UNITY,
    DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, encode_dnam_bin,
};
use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16};

fn build_demo_hybrid_container() -> Vec<u8> {
    let mut header = DnamJsonHeader::default_reference();
    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    let one_fp16 = U16::new(f32_to_fp16(1.0));
    a1.linear_readout_full.w_ro_and_mods_fp16[24 * 12 + 4..24 * 12 + 6].fill(one_fp16);
    a2.linear_readout_pro.w_ro_and_mods_fp16[32 * 12 + 4..32 * 12 + 6].fill(one_fp16);
    a3.linear_readout_ultra.w_ro_and_mods_fp16[40 * 12 + 4..40 * 12 + 6].fill(one_fp16);

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
    .expect("valid Layer 0 .dnam-bin payload");

    build_dnam_container_from_dnam_bin(&mut header, &dnam_bin)
        .expect("valid Layer 1 hybrid .dnam container")
}

fn main() {
    let container_bytes = build_demo_hybrid_container();

    // SAFETY: All pointers passed to the C-ABI functions below are valid, non-null stack/vec
    // slices matching `num_frames`, and `handle` is destroyed exactly once at the end of scope.
    unsafe {
        // 1. Create an engine instance capped at Pro tier (`max_supported_tier = 2`) at 48 kHz
        let handle = dnam_engine_create_with_max_tier(
            container_bytes.as_ptr(),
            container_bytes.len(),
            1, // Initial tier = Full (1)
            2, // Max supported tier = Pro (2)
            48_000,
        );
        assert!(!handle.is_null(), "C-ABI engine creation succeeded");

        // 2. Automate control 0 and process a 128-frame stereo callback + 128-frame mono callback
        dnam_engine_set_control_value(handle, 0, 0.75);

        let left_in = [0.20_f32; 128];
        let right_in = [-0.20_f32; 128];
        let mut left_out = [0.0_f32; 128];
        let mut right_out = [0.0_f32; 128];
        dnam_engine_process_block(
            handle,
            left_in.as_ptr(),
            right_in.as_ptr(),
            left_out.as_mut_ptr(),
            right_out.as_mut_ptr(),
            128,
        );

        let mut mono_out = [0.0_f32; 128];
        dnam_engine_process_mono(handle, left_in.as_ptr(), mono_out.as_mut_ptr(), 128);

        // 3. Switch host sample rate to 96 kHz and enable 4x Offline Bounce (12-sample PDC latency)
        assert_eq!(dnam_engine_set_sample_rate(handle, 96_000), 0);
        dnam_engine_set_tier(handle, 3); // Request Ultra (3); capped to Pro (2) by max_supported_tier
        dnam_engine_set_offline_bounce(handle, true);
        let pdc_samples = dnam_engine_get_latency_samples(handle);

        // 4. Read lock-free 60 fps telemetry struct over C-ABI
        let mut telemetry = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::zeroed().assume_init();
        dnam_engine_get_telemetry(handle, &mut telemetry);
        println!(
            "C-ABI Host Telemetry | Effective Tier: {} (capped at Pro=2) | PDC Latency: {} samples | Out L[127]: {:.6}",
            telemetry.active_tier, pdc_samples, left_out[127],
        );

        // 5. Reset to DC equilibrium and destroy handle
        dnam_engine_reset(handle);
        dnam_engine_destroy(handle);
    }
}
