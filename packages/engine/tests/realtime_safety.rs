//! Real-Time Zero-Allocation & Embedded Scratch Verification Suite (`tests/realtime_safety.rs`).
//!
//! Uses a custom `#[global_allocator]` (`CountingAllocator`) to prove empirically that all
//! real-time audio and control methods (`process_block`, `process_block_in_place`,
//! `process_mono_to_stereo`, `process_mono`, `set_tier`, `set_control_value`,
//! `reload_model_in_place` over `.dnam` carrying embedded `dbin_prefix`, `PreviewGuardState`,
//! `DnamHostHandle`, and C-ABI FFI) execute with **0 heap allocations** and **0 heap
//! deallocations**.

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use arena_builder::build_sona_container_from_header_options;
use dnam_engine::container::{
    ARENA_OFFSET_LITE_END, V3ContainerBuildSpec, build_v3_dnam_container_with_dbin,
};
use dnam_engine::ffi::{
    dnam_engine_create_from_bytes, dnam_engine_destroy, dnam_engine_get_latency_samples,
    dnam_engine_get_telemetry, dnam_engine_process_block, dnam_engine_process_mono,
    dnam_engine_reset, dnam_engine_set_control_immediate, dnam_engine_set_control_value,
    dnam_engine_set_max_supported_tier, dnam_engine_set_maximize_tier_on_bounce,
    dnam_engine_set_offline_bounce, dnam_engine_set_sample_rate, dnam_engine_set_tier,
};
use dnam_engine::host::DnamHostHandle;
use dnam_engine::memory::{AlignedBuffer, AlignedFp16Block, CacheLine64};
use dnam_engine::schema::{
    ControlKind, ControlSchemaDef, ControlUiHints, DnamError, DnamJsonHeader, PresetDef,
};
use dnam_engine::types::DnamTelemetrySnapshot;
use dnam_engine::{DnamAudioIo, DnamEngine, DynamicDnamEngine, Tier, extract_dnam_bin};
use dnam_vm::bin::DnamBinHeaderWire;
use dnam_vm::math::DefaultMathBackend;
use zerocopy::FromBytes;

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegates directly to `std::alloc::System` while incrementing atomic counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: Forwards valid layout to System allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: Forwards valid pointer and layout to System allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_alloc_counters() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    DEALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn current_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
}

fn current_dealloc_count() -> usize {
    DEALLOC_COUNT.load(Ordering::SeqCst)
}

fn make_test_container_with_kind(container_kind: &str) -> (AlignedBuffer<CacheLine64>, Vec<u8>) {
    let mut header = DnamJsonHeader::default_reference();
    header.container_kind = container_kind.to_string();
    header.metadata.name = "RT Safety Model".to_string();
    header.controls_schema = (0..20)
        .map(|idx| ControlSchemaDef {
            id: format!("ctrl_{idx}"),
            name: format!("Control {idx}"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        })
        .collect();
    header.presets = vec![PresetDef {
        id: "rt_preset".to_string(),
        name: "RT Preset".to_string(),
        description: None,
        controls: BTreeMap::from([("ctrl_0".to_string(), serde_json::json!(7.5))]),
        user_metadata: BTreeMap::new(),
    }];

    let (_arena, dbin, vec_bytes) =
        build_sona_container_from_header_options(&mut header, Tier::Ultra, false, false, true, &[]);
    assert!(extract_dnam_bin(&vec_bytes).is_ok());
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(vec_bytes.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&vec_bytes);
    (aligned, dbin)
}

/// verify[dsp.ctrl.op-m.spsc-ingress]
/// verify[arch.telemetry.seqlock-publication]
/// verify[sdk.rust.c-abi-and-host-handle]
#[test]
fn test_zero_heap_allocations_in_all_realtime_paths_and_embedded_scratch_loader() {
    let (container_active, template_dbin) = make_test_container_with_kind("dnam");
    let (container_small, _) = make_test_container_with_kind("dnam");
    let (container_preview, _) = make_test_container_with_kind("preview");

    // 1. Verify telemetry snapshot reports bypassed and host_write_lock status flags
    let eng_check = DnamEngine::<DefaultMathBackend>::from_bytes(
        container_small.as_bytes(),
        Tier::Full,
        48_000,
    )
    .unwrap();
    let check_snap = eng_check.telemetry_snapshot();
    assert_eq!(check_snap.bypassed, 0);
    assert_eq!(check_snap.host_write_lock, 0);

    // 2. Verify from_bytes_with_scratch and embedded .dnam-bin extraction
    let mut external_scratch = vec![AlignedFp16Block([0u16; 32]); 256];
    let mut eng_embedded = DnamEngine::<DefaultMathBackend>::from_bytes_with_scratch(
        container_active.as_bytes(),
        &mut external_scratch,
        Tier::Ultra,
        48_000,
    )
    .unwrap();
    assert!(eng_embedded.header().is_some());

    // 3. Verify all audio I/O topologies, multiport/bounce/router controls, tier switching,
    //    control updates, write locks, tempo sync, preset load, and telemetry allocate 0 bytes!
    let in_l = [0.25_f32; 256];
    let in_r = [-0.25_f32; 256];
    let ret_l = [0.1_f32; 256];
    let ret_r = [-0.1_f32; 256];
    let mut out_l = [0.0_f32; 256];
    let mut out_r = [0.0_f32; 256];
    let mut send_l = [0.0_f32; 256];
    let mut send_r = [0.0_f32; 256];
    let mut dry_c = [0.0_f32; 256];

    reset_alloc_counters();
    let _ = extract_dnam_bin(container_active.as_bytes()).unwrap();
    eng_embedded.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    eng_embedded.process_stereo(&in_l, &in_r, &mut out_l, &mut out_r);
    eng_embedded.process_block_in_place(&mut out_l, &mut out_r);
    eng_embedded.process_stereo_to_mono(&in_l, &in_r, &mut out_l);
    eng_embedded.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
    eng_embedded.process_mono(&in_l, &mut out_l);
    eng_embedded.process_block_io(DnamAudioIo {
        in_1: &in_l,
        in_2: Some(&in_r),
        in_3: None,
        in_4: None,
        out_1: &mut out_l,
        out_2: Some(&mut out_r),
        out_3: None,
        out_4: None,
    });
    eng_embedded.process_block_4cm(
        &in_l,
        &in_r,
        &ret_l,
        &ret_r,
        &mut out_l,
        &mut out_r,
        &mut send_l,
        &mut send_r,
    );
    eng_embedded.process_block_wdw(&in_l, &in_r, &mut out_l, &mut out_r, &mut dry_c);
    eng_embedded.process_block_sidechain(&in_l, &in_r, &ret_l, &ret_r, &mut out_l, &mut out_r);
    eng_embedded.set_offline_bounce(true);
    eng_embedded.set_offline_bounce(false);
    eng_embedded.set_bypass(true, true);
    eng_embedded.set_bypass(false, false);
    eng_embedded.set_quad_write_enable(true);
    eng_embedded.set_quad_write_enable(false);
    eng_embedded.set_op_t_write_enable(true);
    eng_embedded.set_op_t_write_enable(false);
    eng_embedded.set_stereo_width(1.25);
    eng_embedded.set_stereo_width(1.0);
    eng_embedded.set_tier(Tier::Lite);
    eng_embedded.set_tier(Tier::Full);
    eng_embedded.set_tier(Tier::Pro);
    eng_embedded.set_tier(Tier::Ultra);
    eng_embedded.set_control_value(0, 5.0);
    eng_embedded.set_control_immediate(0, 5.0);
    eng_embedded.sync_tempo(120.0, None, false);
    eng_embedded.register_tap_tempo(None, false);
    eng_embedded.apply_preset("rt_preset").unwrap();
    eng_embedded.reset_to_dc_equilibrium();
    eng_embedded.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
    let _snap = eng_embedded.telemetry_snapshot();
    assert_eq!(current_alloc_count(), 0);
    assert_eq!(current_dealloc_count(), 0);

    // 3b. Verify PreviewGuardState + preview engine RT audio paths allocate/deallocate 0 bytes!
    let mut prev_eng = DnamEngine::<DefaultMathBackend>::from_preview_bytes(
        container_preview.as_bytes(),
        Tier::Ultra,
        48_000,
    )
    .unwrap();
    assert!(prev_eng.is_preview_mode());
    reset_alloc_counters();
    prev_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
    prev_eng.process_block_in_place(&mut out_l, &mut out_r);
    prev_eng.reset_to_dc_equilibrium();
    assert_eq!(current_alloc_count(), 0);
    assert_eq!(current_dealloc_count(), 0);

    // 4. Verify reload_model_in_place (both valid hot-swap and error paths: bad magic, bad CRC,
    //    bad DMUE/schema, invalid JSON, PreviewContainerRejected) perform 0 allocs & 0 deallocs!
    let mut bad_crc_bytes = container_small.as_bytes().to_vec();
    bad_crc_bytes[192] ^= 0x01;
    let mut bad_schema_bytes = container_small.as_bytes().to_vec();
    let schema_last = bad_schema_bytes.len() - 1;
    bad_schema_bytes[schema_last] ^= 0x01;

    let (hdr_wire, _) = DnamBinHeaderWire::read_from_prefix(&template_dbin).unwrap();
    let arena0_off = hdr_wire.arena0_offset.get() as usize;
    let empty_lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    let bad_json_container = build_v3_dnam_container_with_dbin(&V3ContainerBuildSpec {
        protection_profile: 0,
        header_bytes: br#"{"broken": "#,
        shared_bytes: &[],
        auth_bytes: &[0u8; 64],
        dbin_prefix_bytes: &template_dbin[..arena0_off],
        tier_slices: &[empty_lite_arena.as_slice()],
        mutable_frame_bytes: None,
        auth_kind: 0,
    })
    .unwrap();

    let mut eng_pool = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    let mut dyn_pool =
        DynamicDnamEngine::from_bytes(container_active.as_bytes(), Tier::Ultra, 48_000).unwrap();

    reset_alloc_counters();
    eng_pool
        .reload_model_in_place(container_active.as_bytes(), Tier::Full, 48_000)
        .unwrap();
    eng_pool.set_offline_bounce(true);
    eng_pool.process_block_io(DnamAudioIo {
        in_1: &in_l,
        in_2: Some(&in_r),
        in_3: None,
        in_4: None,
        out_1: &mut out_l,
        out_2: Some(&mut out_r),
        out_3: None,
        out_4: None,
    });
    eng_pool.set_offline_bounce(false);
    eng_pool.process_block_4cm(
        &in_l,
        &in_r,
        &ret_l,
        &ret_r,
        &mut out_l,
        &mut out_r,
        &mut send_l,
        &mut send_r,
    );
    eng_pool.process_block_wdw(&in_l, &in_r, &mut out_l, &mut out_r, &mut dry_c);
    eng_pool.process_block_sidechain(&in_l, &in_r, &ret_l, &ret_r, &mut out_l, &mut out_r);
    eng_pool.set_bypass(true, true);
    eng_pool.set_bypass(false, false);
    eng_pool.set_quad_write_enable(true);
    eng_pool.set_quad_write_enable(false);
    eng_pool.set_op_t_write_enable(true);
    eng_pool.set_op_t_write_enable(false);
    eng_pool.set_stereo_width(1.5);
    eng_pool.set_stereo_width(1.0);
    eng_pool.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
    eng_pool
        .reload_model_in_place(container_small.as_bytes(), Tier::Lite, 48_000)
        .unwrap();
    assert!(matches!(
        eng_pool.reload_model_in_place(&[0u8; 256], Tier::Lite, 48_000),
        Err(DnamError::InvalidMagic(_))
    ));
    assert!(matches!(
        eng_pool.reload_model_in_place(&bad_crc_bytes, Tier::Lite, 48_000),
        Err(DnamError::Crc32cMismatch { .. })
    ));
    assert!(matches!(
        eng_pool.reload_model_in_place(&bad_schema_bytes, Tier::Lite, 48_000),
        Err(DnamError::SchemaValidationError(_))
    ));
    assert!(matches!(
        eng_pool.reload_model_in_place(&bad_json_container, Tier::Lite, 48_000),
        Err(DnamError::InvalidJsonHeader(_))
    ));
    assert!(matches!(
        eng_pool.reload_model_in_place(container_preview.as_bytes(), Tier::Lite, 48_000),
        Err(DnamError::PreviewContainerRejected)
    ));

    dyn_pool
        .reload_model_in_place(container_small.as_bytes(), Tier::Lite, 48_000)
        .unwrap();
    assert!(matches!(
        dyn_pool.reload_model_in_place(&[0u8; 256], Tier::Lite, 48_000),
        Err(DnamError::InvalidMagic(_))
    ));
    assert!(matches!(
        dyn_pool.reload_model_in_place(&bad_crc_bytes, Tier::Lite, 48_000),
        Err(DnamError::Crc32cMismatch { .. })
    ));
    assert!(matches!(
        dyn_pool.reload_model_in_place(&bad_schema_bytes, Tier::Lite, 48_000),
        Err(DnamError::SchemaValidationError(_))
    ));
    assert!(matches!(
        dyn_pool.reload_model_in_place(&bad_json_container, Tier::Lite, 48_000),
        Err(DnamError::InvalidJsonHeader(_))
    ));
    assert!(matches!(
        dyn_pool.reload_model_in_place(container_preview.as_bytes(), Tier::Lite, 48_000),
        Err(DnamError::PreviewContainerRejected)
    ));
    assert_eq!(current_alloc_count(), 0);
    assert_eq!(current_dealloc_count(), 0);

    // 5. Verify SpscControlRing, TelemetryCell seqlock, and RealtimeEngineSwap execute with 0 RT allocs
    {
        use dnam_engine::host::{
            RawTelemetryRecord, realtime_engine_swap, spsc_control_channel, telemetry_channel,
        };

        let initial_engine =
            DynamicDnamEngine::from_bytes_owned(container_active.as_bytes(), Tier::Full, 48_000)
                .unwrap();
        let staged_engine =
            DynamicDnamEngine::from_bytes_owned(container_small.as_bytes(), Tier::Lite, 48_000)
                .unwrap();

        let (mut producer, mut consumer) = spsc_control_channel(&initial_engine);
        let initial_rec = RawTelemetryRecord::from_engine(&initial_engine, 0);
        let (mut telem_writer, telem_reader) = telemetry_channel(initial_rec);
        let (mut swap_ctrl, mut swap_audio) = realtime_engine_swap(initial_engine);

        swap_ctrl.publish_engine_at_epoch(staged_engine, 1);

        reset_alloc_counters();
        producer.set_control_value(0, 7.5);
        producer.set_control_immediate(1, 1.0);
        producer.apply_preset("rt_preset").unwrap();
        producer.sync_tempo(132.0, None, true);
        for i in 0..80 {
            producer.set_bypass(i % 2 == 0, true);
        }

        assert_eq!(swap_audio.poll_swap_epoch(), Some(1));
        let applied = consumer.drain_into(swap_audio.engine_mut());
        assert!(applied > 0);
        telem_writer.publish_from_engine(swap_audio.engine(), consumer.drained_causal_epoch());
        let snap = telem_reader.read();
        assert_eq!(snap.causal_epoch, producer.causal_epoch());
        assert_eq!(current_alloc_count(), 0);
        assert_eq!(current_dealloc_count(), 0);

        assert_eq!(swap_ctrl.reclaim_retired(), 1);
    }

    // 5b. Verify DnamHostHandle RT control, audio, and telemetry execute with 0 allocs & 0 deallocs
    {
        let host_handle =
            DnamHostHandle::from_bytes_owned(container_active.as_bytes(), Tier::Full, 48_000)
                .unwrap();

        reset_alloc_counters();
        host_handle.set_control_value(0, 6.5);
        host_handle.set_tier(Tier::Pro);
        host_handle.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        host_handle.process_block_in_place(&mut out_l, &mut out_r);
        host_handle.process_mono(&in_l, &mut out_l);
        let host_snap = host_handle.telemetry_snapshot();
        let host_rec = host_handle.telemetry_record();
        assert_eq!(host_snap.active_tier, host_rec.snapshot.active_tier);
        assert_eq!(current_alloc_count(), 0);
        assert_eq!(current_dealloc_count(), 0);
    }

    // 6. Verify C-ABI setters, reset, process_block, process_mono, latency & telemetry = 0 allocs
    // SAFETY: Creates a valid owned handle before entering the zero-allocation window.
    let ffi_handle = unsafe {
        dnam_engine_create_from_bytes(
            container_active.as_bytes().as_ptr(),
            container_active.as_bytes().len(),
            Tier::Full as u8,
            48_000,
        )
    };
    assert!(!ffi_handle.is_null());
    let mut ffi_snap: DnamTelemetrySnapshot = eng_embedded.telemetry_snapshot();

    reset_alloc_counters();
    // SAFETY: `ffi_handle` is valid and slice pointers cover `in_l.len()` frames.
    unsafe {
        dnam_engine_set_tier(ffi_handle, Tier::Pro as u8);
        dnam_engine_set_max_supported_tier(ffi_handle, Tier::Ultra as u8);
        dnam_engine_set_offline_bounce(ffi_handle, true);
        dnam_engine_set_offline_bounce(ffi_handle, false);
        dnam_engine_set_maximize_tier_on_bounce(ffi_handle, true);
        dnam_engine_set_control_value(ffi_handle, 0, 5.0);
        dnam_engine_set_control_immediate(ffi_handle, 0, 5.0);
        assert_eq!(dnam_engine_set_sample_rate(ffi_handle, 96_000), 0);
        assert_eq!(dnam_engine_set_sample_rate(ffi_handle, 192_000), 0);
        assert_eq!(dnam_engine_set_sample_rate(ffi_handle, 22_050), -1);
        assert_eq!(dnam_engine_set_sample_rate(ffi_handle, 48_000), 0);
        dnam_engine_reset(ffi_handle);
        dnam_engine_process_block(
            ffi_handle,
            in_l.as_ptr(),
            in_r.as_ptr(),
            out_l.as_mut_ptr(),
            out_r.as_mut_ptr(),
            in_l.len(),
        );
        let l_ptr = out_l.as_mut_ptr();
        let r_ptr = out_r.as_mut_ptr();
        dnam_engine_process_block(
            ffi_handle,
            l_ptr as *const f32,
            r_ptr as *const f32,
            l_ptr,
            r_ptr,
            out_l.len(),
        );
        dnam_engine_process_block(
            ffi_handle,
            r_ptr as *const f32,
            in_r.as_ptr(),
            out_l.as_mut_ptr(),
            r_ptr,
            out_r.len(),
        );
        dnam_engine_process_mono(ffi_handle, in_l.as_ptr(), out_l.as_mut_ptr(), in_l.len());
        let mono_ptr = out_l.as_mut_ptr();
        dnam_engine_process_mono(ffi_handle, mono_ptr as *const f32, mono_ptr, out_l.len());
        let _lat = dnam_engine_get_latency_samples(ffi_handle);
        dnam_engine_get_telemetry(ffi_handle, &mut ffi_snap);
    }
    assert_eq!(current_alloc_count(), 0);
    assert_eq!(current_dealloc_count(), 0);
    // SAFETY: Frees handle allocated by `dnam_engine_create_from_bytes`.
    unsafe {
        dnam_engine_destroy(ffi_handle);
    }
}
