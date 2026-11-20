//! Real-Time Zero-Allocation & Embedded Scratch Verification Suite (`tests/realtime_safety.rs`).
//!
//! Uses a custom `#[global_allocator]` (`CountingAllocator`) to prove empirically that all
//! real-time audio and control methods (`process_block`, `process_block_in_place`,
//! `process_mono_to_stereo`, `process_mono`, `set_tier`, `set_control_value`,
//! `reload_model_in_place`, `DnamHostHandle`, and C-ABI FFI) execute with **0 heap allocations**
//! and **0 heap deallocations** (`verify[sdk.rust.c-abi-and-host-handle]`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use dnam_engine::container::{
    ARENA_OFFSET_LITE_END, ARENA_OFFSET_ULTRA_END, build_dnam_container,
    build_v3_open_dnam_container,
};
use dnam_engine::ffi::{
    dnam_engine_create_from_bytes, dnam_engine_destroy, dnam_engine_get_latency_samples,
    dnam_engine_get_telemetry, dnam_engine_process_block, dnam_engine_process_mono,
    dnam_engine_reset, dnam_engine_set_control_immediate, dnam_engine_set_control_value,
    dnam_engine_set_max_supported_tier, dnam_engine_set_maximize_tier_on_bounce,
    dnam_engine_set_offline_bounce, dnam_engine_set_tier,
};
use dnam_engine::graph::sona_core::SonaMatryoshkaModel;
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::host::DnamHostHandle;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::memory::{AlignedBuffer, AlignedFp16Block, CacheLine64};
use dnam_engine::schema::{
    ControlKind, ControlSchemaDef, ControlUiHints, DnamError, DnamJsonHeader, PresetDef,
};
use dnam_engine::types::DnamTelemetrySnapshot;
use dnam_engine::{DnamAudioIo, DnamEngine, Tier};

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

fn make_test_container_with_kind(container_kind: &str) -> AlignedBuffer<CacheLine64> {
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

    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_low_gain = 1.0;
    model.stage0.xo_high_gain = 1.0;
    model.stage4.iter_mut().for_each(|s| {
        s.g_loop = 0.72;
        s.wet_mix = 0.25;
    });
    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    pack_sona_matryoshka_to_arena(&model, &mut arena);
    let vec_bytes = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(vec_bytes.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&vec_bytes);
    aligned
}

#[test]
fn test_zero_heap_allocations_in_all_realtime_paths_and_embedded_scratch_loader() {
    let container_active = make_test_container_with_kind("dnam");
    let container_small = make_test_container_with_kind("dnam");
    let container_preview = make_test_container_with_kind("preview");

    // 1. Verify telemetry snapshot reports bypassed and is_frozen status flags
    let eng_check = DnamEngine::<DefaultMathBackend>::from_bytes(
        container_small.as_bytes(),
        Tier::Full,
        48_000,
    )
    .unwrap();
    assert_eq!(eng_check.telemetry_snapshot().bypassed, 0);
    assert_eq!(eng_check.telemetry_snapshot().is_frozen, 0);

    // 2. Verify from_bytes_with_scratch borrows 64B-aligned Arena 0..3 when aligned (zero audio allocs).
    let mut external_scratch = vec![AlignedFp16Block([0u16; 32]); 256]; // 1,024 frames * 8 lines / 32
    let mut eng_embedded = DnamEngine::<DefaultMathBackend>::from_bytes_with_scratch(
        container_active.as_bytes(),
        &mut external_scratch,
        Tier::Ultra,
        48_000,
    )
    .unwrap();
    assert!(eng_embedded.header().is_some());

    // 3. Verify all audio I/O topologies, multiport/bounce/router controls, tier switching,
    //    control updates, tempo sync, preset load, and telemetry allocate 0 bytes!
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
    eng_embedded.set_freeze(true);
    eng_embedded.set_freeze(false);
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
    assert_eq!(
        current_alloc_count(),
        0,
        "Real-time audio/control path allocated heap memory!"
    );
    assert_eq!(
        current_dealloc_count(),
        0,
        "Real-time audio/control path deallocated heap memory!"
    );

    // 4. Verify reload_model_in_place (both valid hot-swap and error paths: bad magic, bad CRC,
    //    bad DMUE/schema, invalid JSON, preview rejection) on both DnamEngine and DynamicDnamEngine
    //    perform ZERO heap allocations and ZERO deallocations!
    let mut bad_crc_bytes = container_small.as_bytes().to_vec();
    bad_crc_bytes[192] ^= 0x01;
    let mut bad_schema_bytes = container_small.as_bytes().to_vec();
    let schema_last = bad_schema_bytes.len() - 1;
    bad_schema_bytes[schema_last] ^= 0x01;
    let empty_lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
    let bad_json_container = build_v3_open_dnam_container(
        br#"{"broken": "#,
        &[0u8; 64],
        &[0u8; 64],
        &[empty_lite_arena.as_slice()],
        None,
        0,
    )
    .unwrap();
    let mut eng_pool = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    let mut dyn_pool = dnam_engine::DynamicDnamEngine::from_bytes(
        container_active.as_bytes(),
        Tier::Ultra,
        48_000,
    )
    .unwrap();
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
    eng_pool.set_freeze(true);
    eng_pool.set_freeze(false);
    eng_pool.set_stereo_width(1.5);
    eng_pool.set_stereo_width(1.0);
    eng_pool.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
    eng_pool
        .reload_model_in_place(container_small.as_bytes(), Tier::Lite, 48_000)
        .unwrap();
    let bad_magic_err = eng_pool
        .reload_model_in_place(&[0u8; 256], Tier::Lite, 48_000)
        .unwrap_err();
    assert!(matches!(bad_magic_err, DnamError::InvalidMagic(_)));
    let bad_crc_err = eng_pool
        .reload_model_in_place(&bad_crc_bytes, Tier::Lite, 48_000)
        .unwrap_err();
    assert!(matches!(bad_crc_err, DnamError::Crc32cMismatch { .. }));
    let bad_schema_err = eng_pool
        .reload_model_in_place(&bad_schema_bytes, Tier::Lite, 48_000)
        .unwrap_err();
    assert!(matches!(
        bad_schema_err,
        DnamError::SchemaValidationError(_)
    ));
    let bad_json_err = eng_pool
        .reload_model_in_place(&bad_json_container, Tier::Lite, 48_000)
        .unwrap_err();
    assert!(matches!(bad_json_err, DnamError::InvalidJsonHeader(_)));
    let preview_err = eng_pool
        .reload_model_in_place(container_preview.as_bytes(), Tier::Lite, 48_000)
        .unwrap_err();
    assert!(matches!(preview_err, DnamError::PreviewContainerRejected));

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
    assert_eq!(
        current_alloc_count(),
        0,
        "reload_model_in_place or pool real-time paths allocated heap memory!"
    );
    assert_eq!(
        current_dealloc_count(),
        0,
        "reload_model_in_place or pool real-time paths deallocated heap memory!"
    );

    // 5. Verify SpscControlRing, TelemetryCell seqlock, and RealtimeEngineSwap perform ZERO heap allocations on the RT path!
    // verify[dsp.ctrl.op-m.spsc-ingress]
    // verify[arch.telemetry.seqlock-publication]
    {
        use dnam_engine::DynamicDnamEngine;
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

        // Stage a replacement engine on the non-RT controller side before resetting allocation counters
        swap_ctrl.publish_engine_at_epoch(staged_engine, 1);

        reset_alloc_counters();

        // Producer-side control updates (Lane 1 coalescing + Lane 2 preset & tempo + Lane 3 overflow latch)
        producer.set_control_value(0, 7.5);
        producer.set_control_immediate(1, 1.0);
        producer.apply_preset("rt_preset").unwrap();
        producer.sync_tempo(132.0, None, true);
        for i in 0..80 {
            producer.set_bypass(i % 2 == 0, true);
        }

        // Real-time audio-side poll_swap, SPSC drain_into, and Seqlock telemetry publication/read
        assert_eq!(swap_audio.poll_swap_epoch(), Some(1));
        let applied = consumer.drain_into(swap_audio.engine_mut());
        assert!(applied > 0);
        telem_writer.publish_from_engine(swap_audio.engine(), consumer.drained_causal_epoch());
        let snap = telem_reader.read();
        assert_eq!(snap.causal_epoch, producer.causal_epoch());

        assert_eq!(
            current_alloc_count(),
            0,
            "SpscControlRing / TelemetryCell / RealtimeEngineSwap allocated heap memory on real-time path!"
        );
        assert_eq!(
            current_dealloc_count(),
            0,
            "SpscControlRing / TelemetryCell / RealtimeEngineSwap deallocated heap memory on real-time path!"
        );

        // Reclaim retired engine on the non-RT controller side
        assert_eq!(swap_ctrl.reclaim_retired(), 1);
    }

    // 5b. Verify DnamHostHandle real-time control, audio processing, and wait-free telemetry
    //     execute with ZERO heap allocations and ZERO deallocations (pre-warmed HostMutex).
    // verify[sdk.rust.c-abi-and-host-handle]
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
        assert_eq!(
            current_alloc_count(),
            0,
            "DnamHostHandle real-time path allocated heap memory!"
        );
        assert_eq!(
            current_dealloc_count(),
            0,
            "DnamHostHandle real-time path deallocated heap memory!"
        );
    }

    // 6. Verify C-ABI setters, reset, process_block (disjoint, in-place, crossed-channel),
    //    process_mono (disjoint, in-place), get_latency_samples, and get_telemetry execute with
    //    0 heap allocations and 0 deallocations.
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
    // SAFETY: `ffi_handle` is non-null from `dnam_engine_create_from_bytes` and buffers hold `in_l.len()` elements.
    unsafe {
        dnam_engine_set_tier(ffi_handle, Tier::Pro as u8);
        dnam_engine_set_max_supported_tier(ffi_handle, Tier::Ultra as u8);
        dnam_engine_set_offline_bounce(ffi_handle, true);
        dnam_engine_set_offline_bounce(ffi_handle, false);
        dnam_engine_set_maximize_tier_on_bounce(ffi_handle, true);
        dnam_engine_set_control_value(ffi_handle, 0, 5.0);
        dnam_engine_set_control_immediate(ffi_handle, 0, 5.0);
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
    assert_eq!(
        current_alloc_count(),
        0,
        "C-ABI real-time setters/reset/process/telemetry allocated heap memory!"
    );
    assert_eq!(
        current_dealloc_count(),
        0,
        "C-ABI real-time setters/reset/process/telemetry deallocated heap memory!"
    );
    // SAFETY: Releases the handle originally allocated by `dnam_engine_create_from_bytes`.
    unsafe {
        dnam_engine_destroy(ffi_handle);
    }
}
