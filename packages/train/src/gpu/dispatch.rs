//! `wgpu 30.0.2` storage/uniform buffer upload, command submission, `MAP_READ` staging readback,
//! and `zerocopy` byte decoding (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::error::TrainError;

/// Minimum WebGPU storage buffer byte size (`16 B` for dummy empty-slice bindings).
const MIN_STORAGE_BUFFER_BYTES: usize = 16;

/// Casts any `IntoBytes + Immutable` slice safely into a byte slice (`0` unsafe).
#[inline]
#[must_use]
pub(crate) fn as_bytes<T: IntoBytes + Immutable>(slice: &[T]) -> &[u8] {
    slice.as_bytes()
}

/// Decodes a `zerocopy` `[f32; 4]` vector from mapped GPU bytes (`0` manual `from_le_bytes`).
pub(crate) fn read_vec4_vec(bytes: &[u8]) -> Result<Vec<[f32; 4]>, TrainError> {
    let (chunks, rem) = bytes.as_chunks::<16>();
    if !rem.is_empty() {
        return Err(TrainError::GpuBufferMapFailed(
            "unaligned [f32; 4] readback",
        ));
    }
    Ok(chunks
        .iter()
        .filter_map(|c| <[f32; 4]>::read_from_bytes(c).ok())
        .collect())
}

/// Decodes a `zerocopy` `[f32; 2]` vector from mapped GPU bytes.
pub(crate) fn read_vec2_vec(bytes: &[u8]) -> Result<Vec<[f32; 2]>, TrainError> {
    let (chunks, rem) = bytes.as_chunks::<8>();
    if !rem.is_empty() {
        return Err(TrainError::GpuBufferMapFailed(
            "unaligned [f32; 2] readback",
        ));
    }
    Ok(chunks
        .iter()
        .filter_map(|c| <[f32; 2]>::read_from_bytes(c).ok())
        .collect())
}

/// Decodes a `zerocopy` `f32` vector from mapped GPU bytes.
pub(crate) fn read_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, TrainError> {
    let (chunks, rem) = bytes.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(TrainError::GpuBufferMapFailed("unaligned f32 readback"));
    }
    Ok(chunks
        .iter()
        .filter_map(|c| f32::read_from_bytes(c).ok())
        .collect())
}

/// Decodes a `zerocopy` `i32` vector from mapped GPU bytes.
pub(crate) fn read_i32_vec(bytes: &[u8]) -> Result<Vec<i32>, TrainError> {
    let (chunks, rem) = bytes.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(TrainError::GpuBufferMapFailed("unaligned i32 readback"));
    }
    Ok(chunks
        .iter()
        .filter_map(|c| i32::read_from_bytes(c).ok())
        .collect())
}

/// Decodes a `zerocopy` `u32` vector from mapped GPU bytes.
pub(crate) fn read_u32_vec(bytes: &[u8]) -> Result<Vec<u32>, TrainError> {
    let (chunks, rem) = bytes.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(TrainError::GpuBufferMapFailed("unaligned u32 readback"));
    }
    Ok(chunks
        .iter()
        .filter_map(|c| u32::read_from_bytes(c).ok())
        .collect())
}

/// Allocates a 4-byte-aligned GPU storage buffer (`>= 16 B` so empty slices bind safely).
pub(crate) fn create_storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
    writable: bool,
    max_binding_bytes: u64,
) -> Result<wgpu::Buffer, TrainError> {
    let padded_len = (data.len().max(MIN_STORAGE_BUFFER_BYTES) + 3) & !3;
    let size = padded_len as u64;
    if size > max_binding_bytes {
        return Err(TrainError::BindingExceedsCap {
            buffer: "storage_buffer",
            bytes: size,
            cap_bytes: max_binding_bytes,
        });
    }
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if writable {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buffer, 0, data);
    }
    Ok(buffer)
}

/// Creates a uniform buffer initialized with `uniform_bytes` (`>= 32 B`).
pub(crate) fn create_uniform_buffer(
    device: &wgpu::Device,
    label: &str,
    uniform_bytes: &[u8],
) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: uniform_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Submits `encoder`, maps every staging buffer in `staging`, and returns raw byte vectors
/// trimmed to their exact element sizes (`expected_lens`).
pub(crate) fn submit_and_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    staging: Vec<(wgpu::Buffer, usize)>,
) -> Result<Vec<Vec<u8>>, TrainError> {
    static GPU_SUBMIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _gpu_guard = GPU_SUBMIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    queue.submit([encoder.finish()]);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let mut out = Vec::with_capacity(staging.len());
    for (rb, keep_len) in staging {
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| TrainError::GpuBufferMapFailed("staging channel dropped"))?
            .map_err(|_| TrainError::GpuBufferMapFailed("wgpu BufferAsyncError"))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|_| TrainError::GpuBufferMapFailed("get_mapped_range failed"))?;
        let trimmed = mapped[..keep_len.min(mapped.len())].to_vec();
        drop(mapped);
        rb.unmap();
        out.push(trimmed);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::context::WgpuContext;

    #[test]
    fn test_dispatch_readback_alignment_and_binding_cap_errors() {
        assert!(matches!(
            read_vec4_vec(&[0_u8; 3]),
            Err(TrainError::GpuBufferMapFailed(_))
        ));
        assert!(matches!(
            read_vec2_vec(&[0_u8; 3]),
            Err(TrainError::GpuBufferMapFailed(_))
        ));
        assert!(matches!(
            read_f32_vec(&[0_u8; 3]),
            Err(TrainError::GpuBufferMapFailed(_))
        ));
        assert!(matches!(
            read_i32_vec(&[0_u8; 3]),
            Err(TrainError::GpuBufferMapFailed(_))
        ));
        if let Ok(ctx) = WgpuContext::try_new() {
            let too_big = vec![0_u8; 64];
            assert!(matches!(
                create_storage_buffer(ctx.device(), ctx.queue(), "cap", &too_big, false, 32),
                Err(TrainError::BindingExceedsCap { .. })
            ));
            assert!(
                create_storage_buffer(ctx.device(), ctx.queue(), "empty", &[], false, 64).is_ok()
            );
        }
    }
}
