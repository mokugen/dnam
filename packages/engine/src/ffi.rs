//! `extern "C"` C-ABI & WASM Linear-Memory FFI Exports (`ffi.rs`).
//!
//! Exposes a zero-allocation C-ABI surface over [`DynamicDnamEngine`](crate::DynamicDnamEngine)
//! and [`DnamTelemetrySnapshot`](crate::DnamTelemetrySnapshot) for C/C++ audio hosts,
//! `packages/js` WASM `AudioWorkletProcessor`, and embedded firmware.

use alloc::boxed::Box;
use core::slice;

use crate::{DnamTelemetrySnapshot, DynamicDnamEngine, Tier};

/// Opaque C-ABI handle owning a `'static` [`DynamicDnamEngine`].
#[derive(Debug)]
pub struct DnamEngineHandle {
    engine: DynamicDnamEngine<'static>,
}

/// Creates a new [`DnamEngineHandle`] from a `.dnam` byte buffer, copying the
/// 64-byte aligned tensor arena into owned memory so the caller can free `data_ptr` immediately.
///
/// Returns `core::ptr::null_mut()` if `data_ptr` is null or container validation fails.
///
/// # Safety
/// `data_ptr` must point to a valid readable byte buffer of at least `data_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_create_from_bytes(
    data_ptr: *const u8,
    data_len: usize,
    tier: u8,
    sample_rate_hz: u32,
) -> *mut DnamEngineHandle {
    if data_ptr.is_null() || data_len < 64 {
        return core::ptr::null_mut();
    }
    // SAFETY: Caller guarantees `data_ptr` is valid for `data_len` bytes.
    let bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let tier_enum = Tier::from_u8(tier);
    match DynamicDnamEngine::from_bytes_owned(bytes, tier_enum, sample_rate_hz) {
        Ok(engine) => Box::into_raw(Box::new(DnamEngineHandle { engine })),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Creates a new [`DnamEngineHandle`] capped at `max_supported_tier` (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
///
/// # Safety
/// `data_ptr` must point to a valid readable byte buffer of at least `data_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_create_with_max_tier(
    data_ptr: *const u8,
    data_len: usize,
    tier: u8,
    max_supported_tier: u8,
    sample_rate_hz: u32,
) -> *mut DnamEngineHandle {
    if data_ptr.is_null() || data_len < 64 {
        return core::ptr::null_mut();
    }
    // SAFETY: Caller guarantees `data_ptr` is valid for `data_len` bytes.
    let bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let tier_enum = Tier::from_u8(tier);
    let max_tier_enum = Tier::from_u8(max_supported_tier);
    match DynamicDnamEngine::from_bytes_owned_with_max_tier(
        bytes,
        tier_enum,
        max_tier_enum,
        sample_rate_hz,
    ) {
        Ok(engine) => Box::into_raw(Box::new(DnamEngineHandle { engine })),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Destroys a [`DnamEngineHandle`] previously created by [`dnam_engine_create_from_bytes`].
///
/// # Safety
/// `handle` must be either null or a valid pointer returned by [`dnam_engine_create_from_bytes`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_destroy(handle: *mut DnamEngineHandle) {
    if !handle.is_null() {
        // SAFETY: `handle` was allocated via `Box::into_raw` in `dnam_engine_create_from_bytes`.
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Switches the active slimmable tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`) with `0.00 dB` LUFS continuity.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_tier(handle: *mut DnamEngineHandle, tier: u8) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_tier(Tier::from_u8(tier));
    }
}

/// Sets the plugin/host ceiling tier (`0 = Lite`, `1 = Full`, `2 = Pro`, `3 = Ultra`).
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_max_supported_tier(
    handle: *mut DnamEngineHandle,
    max_tier: u8,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_max_supported_tier(Tier::from_u8(max_tier));
    }
}

/// Enables or disables Offline Bounce mode ($4\times$ linear-phase polyphase oversampling, `12`-sample PDC).
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_offline_bounce(
    handle: *mut DnamEngineHandle,
    enabled: bool,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_offline_bounce(enabled);
    }
}

/// Enables or disables automatic tier promotion (`min(max_available_tier, max_supported_tier)`) during Offline Bounce.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_maximize_tier_on_bounce(
    handle: *mut DnamEngineHandle,
    enabled: bool,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_maximize_tier_on_bounce(enabled);
    }
}

/// Returns the current Plugin Delay Compensation (`PDC`) latency in host samples (`0` or `12`).
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_get_latency_samples(handle: *const DnamEngineHandle) -> usize {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_ref() } {
        h.engine.latency_samples()
    } else {
        0
    }
}

/// Rescales discrete-time parameters for a new host sample rate $f_{s,\text{host}} \in [44100, 384000]\text{ Hz}$.
/// Returns `0` on success or `-1` on invalid sample rate.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_sample_rate(
    handle: *mut DnamEngineHandle,
    sample_rate_hz: u32,
) -> i32 {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() }
        && h.engine.set_sample_rate_rt(sample_rate_hz)
    {
        return 0;
    }
    -1
}

/// Updates a user control value with smooth $3\text{ kHz}$ anti-zipper interpolation.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_control_value(
    handle: *mut DnamEngineHandle,
    ctrl_idx: usize,
    value: f32,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_control_value(ctrl_idx, value);
    }
}

/// Updates a user control value and snaps immediately without anti-zipper smoothing.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_set_control_immediate(
    handle: *mut DnamEngineHandle,
    ctrl_idx: usize,
    value: f32,
) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.set_control_immediate(ctrl_idx, value);
    }
}

/// Resets all SONA stage and operator states to their exact analytical DC resting equilibrium.
///
/// # Safety
/// `handle` must be a valid non-null pointer to a [`DnamEngineHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_reset(handle: *mut DnamEngineHandle) {
    // SAFETY: Caller guarantees `handle` is either null or a valid pointer to `DnamEngineHandle`.
    if let Some(h) = unsafe { handle.as_mut() } {
        h.engine.reset_to_dc_equilibrium();
    }
}

/// Processes a stereo audio block of `num_frames` samples with zero heap allocations.
///
/// # Safety
/// - `handle` must be a valid pointer to [`DnamEngineHandle`].
/// - `left_in`, `right_in`, `left_out`, and `right_out` must each point to valid `f32` buffers of
///   at least `num_frames` elements.
/// - `left_out` and `right_out` must not alias or overlap each other.
/// - `left_in` and `right_in` must either not overlap `left_out` or `right_out` (`src` and `dst`
///   disjoint), or use identical base pointers (`left_in == left_out`, `right_in == right_out`);
///   partial shifted overlap is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_process_block(
    handle: *mut DnamEngineHandle,
    left_in: *const f32,
    right_in: *const f32,
    left_out: *mut f32,
    right_out: *mut f32,
    num_frames: usize,
) {
    if handle.is_null()
        || left_in.is_null()
        || right_in.is_null()
        || left_out.is_null()
        || right_out.is_null()
        || num_frames == 0
    {
        return;
    }
    // SAFETY: Caller guarantees valid pointers for `num_frames` elements and disjoint outputs.
    unsafe {
        let h = &mut *handle;
        let l_out_ro = left_out as *const f32;
        let r_out_ro = right_out as *const f32;
        if core::ptr::eq(left_in, l_out_ro) && core::ptr::eq(right_in, r_out_ro) {
            let l_out = slice::from_raw_parts_mut(left_out, num_frames);
            let r_out = slice::from_raw_parts_mut(right_out, num_frames);
            h.engine.process_block_in_place(l_out, r_out);
        } else if core::ptr::eq(left_in, l_out_ro)
            || core::ptr::eq(left_in, r_out_ro)
            || core::ptr::eq(right_in, l_out_ro)
            || core::ptr::eq(right_in, r_out_ro)
        {
            const CHUNK: usize = 64;
            let mut in_l_buf = [0.0_f32; CHUNK];
            let mut in_r_buf = [0.0_f32; CHUNK];
            let mut out_l_buf = [0.0_f32; CHUNK];
            let mut out_r_buf = [0.0_f32; CHUNK];
            let mut off = 0usize;
            while off < num_frames {
                let n = (num_frames - off).min(CHUNK);
                core::ptr::copy_nonoverlapping(left_in.add(off), in_l_buf.as_mut_ptr(), n);
                core::ptr::copy_nonoverlapping(right_in.add(off), in_r_buf.as_mut_ptr(), n);
                h.engine.process_block(
                    &in_l_buf[..n],
                    &in_r_buf[..n],
                    &mut out_l_buf[..n],
                    &mut out_r_buf[..n],
                );
                core::ptr::copy_nonoverlapping(out_l_buf.as_ptr(), left_out.add(off), n);
                core::ptr::copy_nonoverlapping(out_r_buf.as_ptr(), right_out.add(off), n);
                off += n;
            }
        } else {
            let l_in = slice::from_raw_parts(left_in, num_frames);
            let r_in = slice::from_raw_parts(right_in, num_frames);
            let l_out = slice::from_raw_parts_mut(left_out, num_frames);
            let r_out = slice::from_raw_parts_mut(right_out, num_frames);
            h.engine.process_block(l_in, r_in, l_out, r_out);
        }
    }
}

/// Processes a mono audio block (`Mono -> Mono`, saving 50% of output head MACs) with zero heap allocations.
///
/// # Safety
/// - `handle` must be a valid pointer to [`DnamEngineHandle`].
/// - `mono_in` and `mono_out` must each point to valid `f32` buffers of at least `num_frames`
///   elements.
/// - `mono_in` and `mono_out` must either be disjoint (`src` and `dst` must not overlap) or have
///   identical base pointers (`mono_in == mono_out`); partial shifted overlap is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_process_mono(
    handle: *mut DnamEngineHandle,
    mono_in: *const f32,
    mono_out: *mut f32,
    num_frames: usize,
) {
    if handle.is_null() || mono_in.is_null() || mono_out.is_null() || num_frames == 0 {
        return;
    }
    // SAFETY: Caller guarantees valid pointers for `num_frames` elements without partial overlap.
    unsafe {
        let h = &mut *handle;
        if core::ptr::eq(mono_in, mono_out as *const f32) {
            const CHUNK: usize = 64;
            let mut in_buf = [0.0_f32; CHUNK];
            let mut out_buf = [0.0_f32; CHUNK];
            let mut off = 0usize;
            while off < num_frames {
                let n = (num_frames - off).min(CHUNK);
                core::ptr::copy_nonoverlapping(mono_in.add(off), in_buf.as_mut_ptr(), n);
                h.engine.process_mono(&in_buf[..n], &mut out_buf[..n]);
                core::ptr::copy_nonoverlapping(out_buf.as_ptr(), mono_out.add(off), n);
                off += n;
            }
        } else {
            let m_in = slice::from_raw_parts(mono_in, num_frames);
            let m_out = slice::from_raw_parts_mut(mono_out, num_frames);
            h.engine.process_mono(m_in, m_out);
        }
    }
}

/// Copies the latest 60 fps [`DnamTelemetrySnapshot`] into `out_snapshot`.
///
/// # Safety
/// `handle` and `out_snapshot` must be valid non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnam_engine_get_telemetry(
    handle: *const DnamEngineHandle,
    out_snapshot: *mut DnamTelemetrySnapshot,
) {
    // SAFETY: Caller guarantees `handle` and `out_snapshot` are either null or valid pointers.
    if let (Some(h), Some(out)) = unsafe { (handle.as_ref(), out_snapshot.as_mut()) } {
        *out = h.engine.telemetry_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::DnamJsonHeader;
    use crate::test_support::build_test_dnam_container;

    #[test]
    fn test_all_ffi_functions_and_null_safety() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Ultra, true);

        // SAFETY: Unit test exercises FFI C-ABI interface with both null and valid test pointers.
        unsafe {
            // Null pointer safety
            assert!(dnam_engine_create_from_bytes(core::ptr::null(), 0, 0, 48000).is_null());
            assert!(dnam_engine_create_with_max_tier(core::ptr::null(), 0, 0, 0, 48000).is_null());
            dnam_engine_destroy(core::ptr::null_mut());
            dnam_engine_set_tier(core::ptr::null_mut(), 0);
            dnam_engine_set_max_supported_tier(core::ptr::null_mut(), 0);
            dnam_engine_set_offline_bounce(core::ptr::null_mut(), true);
            dnam_engine_set_maximize_tier_on_bounce(core::ptr::null_mut(), true);
            assert_eq!(dnam_engine_get_latency_samples(core::ptr::null()), 0);
            assert_eq!(
                dnam_engine_set_sample_rate(core::ptr::null_mut(), 48000),
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
                0,
            );
            dnam_engine_process_mono(
                core::ptr::null_mut(),
                core::ptr::null(),
                core::ptr::null_mut(),
                0,
            );
            dnam_engine_get_telemetry(core::ptr::null(), core::ptr::null_mut());

            // Valid handle with create_with_max_tier
            let handle =
                dnam_engine_create_with_max_tier(container.as_ptr(), container.len(), 1, 2, 48000);
            assert!(!handle.is_null());

            dnam_engine_set_max_supported_tier(handle, 1);
            dnam_engine_set_offline_bounce(handle, true);
            assert_eq!(dnam_engine_get_latency_samples(handle), 12);
            dnam_engine_set_offline_bounce(handle, false);
            assert_eq!(dnam_engine_get_latency_samples(handle), 0);
            dnam_engine_set_maximize_tier_on_bounce(handle, true);

            let l_in = [0.1f32; 16];
            let r_in = [-0.1f32; 16];
            let mut l_out = [0.0f32; 16];
            let mut r_out = [0.0f32; 16];
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_mono(handle, l_in.as_ptr(), l_out.as_mut_ptr(), 16);

            let mut snap = core::mem::MaybeUninit::<DnamTelemetrySnapshot>::uninit();
            dnam_engine_get_telemetry(handle, snap.as_mut_ptr());

            // Edge cases: null pointers in process_block and process_mono
            dnam_engine_process_block(
                handle,
                core::ptr::null(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                core::ptr::null(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                core::ptr::null_mut(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                core::ptr::null_mut(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_in.as_ptr(),
                r_in.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                0,
            );

            dnam_engine_process_mono(handle, core::ptr::null(), l_out.as_mut_ptr(), 16);
            dnam_engine_process_mono(handle, l_in.as_ptr(), core::ptr::null_mut(), 16);
            dnam_engine_process_mono(handle, l_in.as_ptr(), l_out.as_mut_ptr(), 0);

            // In-place stereo, crossed-channel alias, and in-place mono paths avoid aliasing UB
            dnam_engine_process_block(
                handle,
                l_out.as_ptr(),
                r_out.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            dnam_engine_process_block(
                handle,
                l_out.as_ptr(),
                l_out.as_ptr(),
                l_out.as_mut_ptr(),
                r_out.as_mut_ptr(),
                16,
            );
            let mut alias_l = [0.25_f32; 80];
            let mut alias_r = [-0.15_f32; 80];
            dnam_engine_process_block(
                handle,
                alias_r.as_ptr(),
                alias_l.as_ptr(),
                alias_l.as_mut_ptr(),
                alias_r.as_mut_ptr(),
                80,
            );
            assert!(alias_l.iter().all(|v| v.is_finite()));
            assert!(alias_r.iter().all(|v| v.is_finite()));
            let mut mono_buf = [0.25_f32; 80];
            dnam_engine_process_mono(handle, mono_buf.as_ptr(), mono_buf.as_mut_ptr(), 80);
            assert!(mono_buf.iter().all(|v| v.is_finite()));

            // Valid setters on active handle
            dnam_engine_set_tier(handle, 2);
            dnam_engine_set_control_value(handle, 0, 0.7);
            dnam_engine_set_control_immediate(handle, 0, 0.7);
            dnam_engine_reset(handle);
            assert_eq!(dnam_engine_set_sample_rate(handle, 48000), 0);
            assert_eq!(dnam_engine_set_sample_rate(handle, 192_000), 0);
            assert_eq!(dnam_engine_set_sample_rate(handle, 384_000), 0);
            assert_eq!(dnam_engine_set_sample_rate(handle, 0), -1);

            // Valid container via dnam_engine_create_from_bytes + Debug formatting
            let handle2 =
                dnam_engine_create_from_bytes(container.as_ptr(), container.len(), 0, 48000);
            assert!(!handle2.is_null());
            let dbg_str = alloc::format!("{:?}", *handle2);
            assert!(dbg_str.contains("DnamEngineHandle"));
            dnam_engine_destroy(handle2);

            // Invalid container parsing returns null
            let invalid_bytes = [0u8; 128];
            assert!(
                dnam_engine_create_from_bytes(
                    invalid_bytes.as_ptr(),
                    invalid_bytes.len(),
                    0,
                    48000
                )
                .is_null()
            );
            assert!(
                dnam_engine_create_with_max_tier(
                    invalid_bytes.as_ptr(),
                    invalid_bytes.len(),
                    0,
                    0,
                    48000
                )
                .is_null()
            );

            dnam_engine_destroy(handle);
        }
    }
}
