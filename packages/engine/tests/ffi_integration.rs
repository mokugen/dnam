//! C-ABI FFI Integration Suite (`tests/ffi_integration.rs`).
//!
//! Exercises all 15 `extern "C"` C-ABI functions exported by `dnam_engine::ffi` over `.dnam`
//! containers wrapping embedded `.dnam-bin` (`verify[sdk.rust.c-abi-and-host-handle]`).

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use arena_builder::{build_sona_container_from_header_options, build_sona_header};
use dnam_engine::container::extract_dnam_bin;
use dnam_engine::ffi::{
    dnam_engine_create_from_bytes, dnam_engine_create_with_max_tier, dnam_engine_destroy,
    dnam_engine_get_latency_samples, dnam_engine_get_telemetry, dnam_engine_process_block,
    dnam_engine_process_mono, dnam_engine_reset, dnam_engine_set_control_immediate,
    dnam_engine_set_control_value, dnam_engine_set_max_supported_tier,
    dnam_engine_set_maximize_tier_on_bounce, dnam_engine_set_offline_bounce,
    dnam_engine_set_sample_rate, dnam_engine_set_tier,
};
use dnam_engine::{DnamTelemetrySnapshot, OFFLINE_BOUNCE_PDC_SAMPLES, Tier};

/// verify[sdk.rust.c-abi-and-host-handle]
#[test]
fn test_c_abi_ffi_lifecycle_and_null_pointer_safety() {
    assert_eq!(core::mem::size_of::<DnamTelemetrySnapshot>(), 184);

    let mut header = build_sona_header();
    let (_, _, container) =
        build_sona_container_from_header_options(&mut header, Tier::Ultra, false, false, true, &[]);
    assert!(extract_dnam_bin(&container).is_ok());

    let mut prev_header = build_sona_header();
    prev_header.container_kind = "preview".to_string();
    let (_, _, preview_container) = build_sona_container_from_header_options(
        &mut prev_header,
        Tier::Ultra,
        false,
        false,
        true,
        &[],
    );

    // SAFETY: Exercises all 15 C-ABI FFI entry points with explicit null/invalid pointers,
    // preview container rejection, valid handles, and disjoint/aliased buffers matching `num_frames`.
    unsafe {
        assert!(dnam_engine_create_from_bytes(core::ptr::null(), 100, 0, 48_000).is_null());
        assert!(dnam_engine_create_from_bytes(container.as_ptr(), 32, 0, 48_000).is_null());
        assert!(dnam_engine_create_from_bytes([0u8; 128].as_ptr(), 128, 0, 48_000).is_null());
        assert!(
            dnam_engine_create_from_bytes(
                preview_container.as_ptr(),
                preview_container.len(),
                0,
                48_000
            )
            .is_null()
        );
        assert!(dnam_engine_create_with_max_tier(core::ptr::null(), 100, 0, 1, 48_000).is_null());
        assert!(dnam_engine_create_with_max_tier(container.as_ptr(), 32, 0, 1, 48_000).is_null());
        assert!(dnam_engine_create_with_max_tier([0u8; 128].as_ptr(), 128, 0, 1, 48_000).is_null());
        assert!(
            dnam_engine_create_with_max_tier(
                preview_container.as_ptr(),
                preview_container.len(),
                0,
                3,
                48_000
            )
            .is_null()
        );

        dnam_engine_destroy(core::ptr::null_mut());
        dnam_engine_set_tier(core::ptr::null_mut(), 1);
        dnam_engine_set_max_supported_tier(core::ptr::null_mut(), 1);
        dnam_engine_set_offline_bounce(core::ptr::null_mut(), true);
        dnam_engine_set_maximize_tier_on_bounce(core::ptr::null_mut(), true);
        assert_eq!(dnam_engine_get_latency_samples(core::ptr::null()), 0);
        assert_eq!(
            dnam_engine_set_sample_rate(core::ptr::null_mut(), 48_000),
            -1
        );
        dnam_engine_set_control_value(core::ptr::null_mut(), 0, 0.5);
        dnam_engine_set_control_immediate(core::ptr::null_mut(), 0, 0.5);
        dnam_engine_reset(core::ptr::null_mut());
        dnam_engine_process_block(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            64,
        );
        dnam_engine_process_mono(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null_mut(),
            64,
        );
        dnam_engine_get_telemetry(core::ptr::null(), core::ptr::null_mut());

        let capped_handle =
            dnam_engine_create_with_max_tier(container.as_ptr(), container.len(), 3, 1, 48_000);
        assert!(!capped_handle.is_null());
        dnam_engine_set_max_supported_tier(capped_handle, 2);
        dnam_engine_set_maximize_tier_on_bounce(capped_handle, true);
        dnam_engine_set_offline_bounce(capped_handle, true);
        assert_eq!(
            dnam_engine_get_latency_samples(capped_handle),
            OFFLINE_BOUNCE_PDC_SAMPLES
        );
        dnam_engine_set_offline_bounce(capped_handle, false);
        assert_eq!(dnam_engine_get_latency_samples(capped_handle), 0);
        dnam_engine_destroy(capped_handle);

        let handle = dnam_engine_create_from_bytes(container.as_ptr(), container.len(), 2, 48_000);
        assert!(!handle.is_null());

        dnam_engine_set_tier(handle, 0);
        assert_eq!(dnam_engine_set_sample_rate(handle, 48_000), 0);
        assert_eq!(dnam_engine_set_sample_rate(handle, 8_000), -1);
        dnam_engine_set_control_value(handle, 0, 6.0);
        dnam_engine_set_control_immediate(handle, 0, 6.0);
        dnam_engine_reset(handle);

        let l_in = [0.25_f32; 32];
        let r_in = [-0.25_f32; 32];
        let mut l_out = [0.0_f32; 32];
        let mut r_out = [0.0_f32; 32];
        dnam_engine_process_block(
            handle,
            l_in.as_ptr(),
            r_in.as_ptr(),
            l_out.as_mut_ptr(),
            r_out.as_mut_ptr(),
            32,
        );
        assert!(l_out.iter().all(|v| v.is_finite()) && r_out.iter().all(|v| v.is_finite()));
        assert!(l_out.iter().any(|&v| v.abs() > 1e-6));

        dnam_engine_process_block(
            handle,
            l_out.as_ptr(),
            r_out.as_ptr(),
            l_out.as_mut_ptr(),
            r_out.as_mut_ptr(),
            32,
        );
        dnam_engine_process_mono(handle, l_in.as_ptr(), l_out.as_mut_ptr(), 32);
        assert!(l_out.iter().all(|v| v.is_finite()));

        let mut snap = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::uninit();
        dnam_engine_get_telemetry(handle, snap.as_mut_ptr());
        let snap = snap.assume_init();
        assert_eq!(snap.active_tier, 0);
        assert_eq!(snap.max_available_tier, 3);
        assert_eq!(snap.sample_rate_hz, 48_000);
        assert_eq!(snap.host_write_lock, 0);
        assert_eq!(snap.bypassed, 0);

        dnam_engine_destroy(handle);
    }
}
