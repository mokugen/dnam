#![forbid(unsafe_code)]

//! Workgroup reductions (`256`-wide portable tree & W3C WebGPU subgroups) and panic-free,
//! division-free `SPIR-V` / `Naga` arithmetic helpers (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{MAX_WORKGROUP_SRAM_BYTES, TILE_FRAMES};
use dnam_kernels::{
    GpuVec4, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_MASK, IEEE754_F32_SIGN_MASK, KernelVec4,
    canonicalize_zero_f32_bits, const_f32_pow2,
};
#[cfg(target_arch = "spirv")]
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
#[cfg(all(target_arch = "spirv", feature = "subgroups"))]
use spirv_std::arch::{subgroup_f_add, subgroup_i_add};

/// Workgroup invocation count across all SONA compute kernels (`256`).
pub const WG_THREADS: usize = TILE_FRAMES;

/// Exact `2^64` factor (`const_f32_pow2(64)`) used to rescale small positive denominators above `1e-5`
/// before calling `GpuVec4::rcp_pos_nr3`, avoiding its internal `1e-6` floor on `1e-24..1e-5` values.
const TWO_POW_64_F32: f32 = const_f32_pow2(64);

const _: () = assert!(WG_THREADS == 256);
const _: () = assert!(MAX_WORKGROUP_SRAM_BYTES == 15_872);

/// Issues a workgroup memory + execution barrier on `spirv` targets (`no-op` on host CPU).
#[inline(always)]
pub fn wg_barrier() {
    #[cfg(target_arch = "spirv")]
    workgroup_memory_barrier_with_group_sync();
}

/// Clamps `idx` to `[0, len - 1]` (or `0` when `len == 0`) BEFORE indexing so `ShaderPanicStrategy::SilentExit`
/// can never prematurely exit an invocation before a workgroup barrier or subgroup operation.
#[inline(always)]
#[must_use]
pub const fn clamp_idx(idx: usize, len: usize) -> usize {
    if len == 0 {
        0
    } else {
        let max_idx = len - 1;
        if idx < max_idx { idx } else { max_idx }
    }
}

/// Panic-free saturating `usize` subtraction.
#[inline(always)]
#[must_use]
pub const fn sub_sat_usize(a: usize, b: usize) -> usize {
    if a > b { a - b } else { 0 }
}

/// Pure-bitmask finiteness check avoiding `f32::is_finite()` (`OpConstant Inf` fails Naga validation).
#[inline(always)]
#[must_use]
pub const fn is_finite_bits(x: f32) -> bool {
    (x.to_bits() & IEEE754_F32_EXP_MASK) != IEEE754_F32_EXP_MASK
}

/// Scrubs `NaN`/`±Inf` to `+0.0`, flushes subnormals, and canonicalizes `-0.0` to `+0.0`.
#[inline(always)]
#[must_use]
pub const fn flush_zero(x: f32) -> f32 {
    canonicalize_zero_f32_bits(x)
}

/// Panic-free, `OpConstant Inf`-free float clamp (`f32::clamp` panics if `lo > hi` or `NaN` -> `SilentExit`).
#[inline(always)]
#[must_use]
pub const fn clamp_f32_safe(x: f32, lo: f32, hi: f32) -> f32 {
    let lo_s = if lo <= hi { lo } else { hi };
    let hi_s = if lo <= hi { hi } else { lo };
    let bits = x.to_bits();
    if (bits & IEEE754_F32_EXP_MASK) == IEEE754_F32_EXP_MASK {
        let is_nan = (bits & IEEE754_F32_ABS_MASK) > IEEE754_F32_EXP_MASK;
        if is_nan || (bits & IEEE754_F32_SIGN_MASK) != 0 {
            flush_zero(lo_s)
        } else {
            flush_zero(hi_s)
        }
    } else if x < lo_s {
        flush_zero(lo_s)
    } else if x > hi_s {
        flush_zero(hi_s)
    } else {
        flush_zero(x)
    }
}

/// Division-free positive reciprocal `1.0 / d` via `GpuVec4::rcp_pos_nr3` (`< 1 ULP`), preserving
/// exact scaling down to `d >= 1e-24` (returns `0.0` if `d <= 0.0` or non-finite).
#[inline(always)]
#[must_use]
pub fn rcp_pos_f32(d: f32) -> f32 {
    if !is_finite_bits(d) || d <= 0.0 {
        return 0.0;
    }
    if d < 1.0e-5 {
        let d_safe = if d > 1.0e-24 { d } else { 1.0e-24 };
        let inv_scaled = GpuVec4::splat(d_safe * TWO_POW_64_F32)
            .rcp_pos_nr3()
            .to_array()[0];
        flush_zero(inv_scaled * TWO_POW_64_F32)
    } else {
        flush_zero(GpuVec4::splat(d).rcp_pos_nr3().to_array()[0])
    }
}

/// Division-free integer floor division `floor(num / den)` via binary search (`den > 0`),
/// guaranteed overflow-free for any `usize` values fitting within 32-bit storage indices.
#[inline(always)]
#[must_use]
pub fn div_floor_u32_divfree(num: usize, den: usize) -> usize {
    if den <= 1 {
        return if den == 0 { 0 } else { num };
    }
    if num < den {
        return 0;
    }
    let mut sub = den;
    let mut bit = 1_usize;
    while sub <= (num >> 1) {
        sub <<= 1;
        bit <<= 1;
    }
    let mut rem = num;
    let mut q = 0_usize;
    while bit > 0 {
        if rem >= sub {
            rem -= sub;
            q |= bit;
        }
        sub >>= 1;
        bit >>= 1;
    }
    q
}

/// Reduces one `f32` value per invocation across all `256` threads of the workgroup and broadcasts
/// the workgroup sum to every invocation in uniform control flow.
#[inline]
pub fn reduce_sum_f32_256(
    scratch: &mut [f32; WG_THREADS],
    lid: usize,
    val: f32,
    sg_lid: u32,
) -> f32 {
    let contrib = {
        let v = flush_zero(val);
        #[cfg(all(target_arch = "spirv", feature = "subgroups"))]
        {
            let sg_sum = subgroup_f_add(v);
            if sg_lid == 0 { sg_sum } else { 0.0 }
        }
        #[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
        {
            let _ = sg_lid;
            v
        }
    };

    scratch[clamp_idx(lid, WG_THREADS)] = contrib;
    wg_barrier();
    let mut stride = WG_THREADS >> 1;
    while stride > 0 {
        if lid < stride {
            let a = scratch[lid];
            let b = scratch[lid + stride];
            scratch[lid] = a + b;
        }
        wg_barrier();
        stride >>= 1;
    }
    let total = flush_zero(scratch[0]);
    wg_barrier();
    total
}

/// Simultaneously reduces two `f32` values `(va, vb)` across all `256` workgroup threads, sharing
/// barrier synchronization and broadcasting `(total_a, total_b)` to every invocation.
#[inline]
pub fn reduce_sum_f32_pair_256(
    scratch_a: &mut [f32; WG_THREADS],
    scratch_b: &mut [f32; WG_THREADS],
    lid: usize,
    va: f32,
    vb: f32,
    sg_lid: u32,
) -> (f32, f32) {
    let (contrib_a, contrib_b) = {
        let a = flush_zero(va);
        let b = flush_zero(vb);
        #[cfg(all(target_arch = "spirv", feature = "subgroups"))]
        {
            let sg_a = subgroup_f_add(a);
            let sg_b = subgroup_f_add(b);
            if sg_lid == 0 {
                (sg_a, sg_b)
            } else {
                (0.0, 0.0)
            }
        }
        #[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
        {
            let _ = sg_lid;
            (a, b)
        }
    };

    let safe_lid = clamp_idx(lid, WG_THREADS);
    scratch_a[safe_lid] = contrib_a;
    scratch_b[safe_lid] = contrib_b;
    wg_barrier();
    let mut stride = WG_THREADS >> 1;
    while stride > 0 {
        if lid < stride {
            let partner = lid + stride;
            scratch_a[lid] += scratch_a[partner];
            scratch_b[lid] += scratch_b[partner];
        }
        wg_barrier();
        stride >>= 1;
    }
    let total = (flush_zero(scratch_a[0]), flush_zero(scratch_b[0]));
    wg_barrier();
    total
}

/// Bit-exact `0-ULP` integer workgroup reduction of `5` Ozaki diagonal accumulators across all
/// `256` invocations using SoA `[5 * 256]` workgroup SRAM (`5,120 B`).
#[inline]
pub fn reduce_sum_i32_5_256(
    scratch: &mut [i32; 5 * WG_THREADS],
    lid: usize,
    acc: [i32; 5],
    sg_lid: u32,
) -> [i32; 5] {
    let contribs = {
        #[cfg(all(target_arch = "spirv", feature = "subgroups"))]
        {
            let s0 = subgroup_i_add(acc[0]);
            let s1 = subgroup_i_add(acc[1]);
            let s2 = subgroup_i_add(acc[2]);
            let s3 = subgroup_i_add(acc[3]);
            let s4 = subgroup_i_add(acc[4]);
            if sg_lid == 0 {
                [s0, s1, s2, s3, s4]
            } else {
                [0; 5]
            }
        }
        #[cfg(not(all(target_arch = "spirv", feature = "subgroups")))]
        {
            let _ = sg_lid;
            acc
        }
    };

    for k in 0..5 {
        scratch[clamp_idx(k * WG_THREADS + lid, 5 * WG_THREADS)] = contribs[k];
    }
    wg_barrier();
    let mut stride = WG_THREADS >> 1;
    while stride > 0 {
        if lid < stride {
            for k in 0..5 {
                let base = k * WG_THREADS;
                let a = scratch[base + lid];
                let b = scratch[base + lid + stride];
                scratch[base + lid] = a.wrapping_add(b);
            }
        }
        wg_barrier();
        stride >>= 1;
    }
    let out = [
        scratch[0],
        scratch[WG_THREADS],
        scratch[2 * WG_THREADS],
        scratch[3 * WG_THREADS],
        scratch[4 * WG_THREADS],
    ];
    wg_barrier();
    out
}
