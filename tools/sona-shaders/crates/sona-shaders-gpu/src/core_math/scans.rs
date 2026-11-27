#![forbid(unsafe_code)]

//! Workgroup-uniform `256`-wide associative `TPT-SVF` forward/adjoint scans, `ParaLRU` linear
//! forward/reverse-suffix scans, and deterministic `256 -> scalar` reductions in `scan_scratch` /
//! `aux` (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod linear_scans;
mod reduce_scans;
mod tpt_scans;

pub use linear_scans::{
    linear_adjoint_reverse_suffix_256_lr, linear_forward_exclusive_scan_128_pair,
    linear_forward_inclusive_scan_256_lr,
};
pub use reduce_scans::{
    reduce_batch6_in_scratch, reduce_sum_256_in_aux, tpt_adjoint_const_m_reduce_256_lr,
    tpt_adjoint_dual_reduce_256,
};
pub use tpt_scans::{
    tpt_adjoint_reverse_suffix_256, tpt_const_m_scan_128_pair, tpt_const_m_scan_256_lr,
    tpt_exclusive_scan_128_pair,
};

use crate::abi::{
    ADJOINT_AUX_FLOATS, ADJOINT_SCAN_SCRATCH_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS, TILE_FRAMES,
};
#[cfg(target_arch = "spirv")]
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;

const _: () = assert!(TILE_FRAMES == 256);
const _: () = assert!(FORWARD_SCAN_SCRATCH_FLOATS == 6 * TILE_FRAMES);
const _: () = assert!(ADJOINT_SCAN_SCRATCH_FLOATS == 6 * TILE_FRAMES);
const _: () = assert!(ADJOINT_AUX_FLOATS == TILE_FRAMES / 2);

/// Issues a workgroup memory + execution barrier on `spirv` targets (`no-op` on host CPU).
#[inline(always)]
pub fn wg_barrier() {
    #[cfg(target_arch = "spirv")]
    workgroup_memory_barrier_with_group_sync();
}
