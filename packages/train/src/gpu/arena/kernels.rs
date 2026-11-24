//! Persistent [`GpuTrainArena`] dispatches for all 7 SONA compute kernels plus batched 2-pass
//! split-64 `ozaki_gram_rhs_split64` (`<= 1.2 MiB` readback at Ultra)
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod ozaki_and_opt;
mod prescreen_and_gabor;
mod scan_and_mix;

pub(super) use ozaki_and_opt::{inner_opt_step, ozaki_gram_rhs_i32, ozaki_gram_rhs_split64};
pub(super) use prescreen_and_gabor::{gabor_bezout_grad, prescreen_population};
pub(super) use scan_and_mix::{adjoint_scan_256, fused_core_scan_256, tile_mix_fwd_adj};

use core::mem::size_of_val;

use zerocopy::{Immutable, IntoBytes};

use super::{ArenaLimits, ArenaSlot, aligned_upload};
use crate::error::TrainError;

fn upload_or_dummy<T: IntoBytes + Immutable>(
    dev: &wgpu::Device,
    queue: &wgpu::Queue,
    slot: &mut ArenaSlot,
    dummy: &mut ArenaSlot,
    slice: &[T],
    limits: ArenaLimits,
    alloc: &mut u64,
) -> Result<wgpu::Buffer, TrainError> {
    if slice.is_empty() {
        return dummy.ensure_storage(dev, 16, limits.cap_bytes, limits.budget_bytes, alloc);
    }
    let buf = slot.ensure_storage(
        dev,
        size_of_val(slice) as u64,
        limits.cap_bytes,
        limits.budget_bytes,
        alloc,
    )?;
    aligned_upload(queue, &buf, slice);
    Ok(buf)
}
