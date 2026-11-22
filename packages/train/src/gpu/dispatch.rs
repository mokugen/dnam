//! `wgpu 30.0.2` storage/uniform buffer upload, command submission, `MAP_READ` staging readback,
//! and `zerocopy` byte decoding (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::mem::size_of;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::error::TrainError;

/// Minimum WebGPU storage buffer byte size (`16 B` for dummy empty-slice bindings).
const MIN_STORAGE_BUFFER_BYTES: usize = 16;

/// Storage buffer descriptor paired with host upload bytes and readback selector.
pub(crate) struct StorageBinding<'a> {
    pub writable: bool,
    pub bytes: &'a [u8],
}

/// Bundled request parameters for [`dispatch_single_pass`].
pub(crate) struct SinglePassDispatch<'a> {
    pub label: &'a str,
    pub workgroups: (u32, u32, u32),
    pub uniform_bytes: &'a [u8],
    pub bindings: &'a [StorageBinding<'a>],
    pub max_binding_bytes: u64,
}

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

/// Single-dispatch helper creating storage/uniform bindings, executing `pipeline`, and reading
/// back all `writable` storage buffers in order.
pub(crate) fn dispatch_single_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    req: &SinglePassDispatch<'_>,
) -> Result<Vec<Vec<u8>>, TrainError> {
    let uniform_buf = create_uniform_buffer(device, req.label, req.uniform_bytes);
    let mut gpu_bufs = Vec::with_capacity(req.bindings.len());
    let mut rw_meta = Vec::new();
    for (idx, b) in req.bindings.iter().enumerate() {
        let buf = create_storage_buffer(
            device,
            queue,
            req.label,
            b.bytes,
            b.writable,
            req.max_binding_bytes,
        )?;
        if b.writable {
            rw_meta.push((idx, buf.size(), b.bytes.len()));
        }
        gpu_bufs.push(buf);
    }

    let mut entries = Vec::with_capacity(1 + gpu_bufs.len());
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buf.as_entire_binding(),
    });
    for (idx, buf) in gpu_bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (idx + 1) as u32,
            resource: buf.as_entire_binding(),
        });
    }
    let bgl = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(req.label),
        layout: &bgl,
        entries: &entries,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(req.label),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(req.label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(req.workgroups.0, req.workgroups.1, req.workgroups.2);
    }
    let mut staging = Vec::with_capacity(rw_meta.len());
    for (idx, alloc_size, keep_len) in rw_meta {
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: alloc_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&gpu_bufs[idx], 0, &rb, 0, alloc_size);
        staging.push((rb, keep_len));
    }

    let _ = size_of::<[u32; 8]>();
    submit_and_readback(device, queue, encoder, staging)
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
