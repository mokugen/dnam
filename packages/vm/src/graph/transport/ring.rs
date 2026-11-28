//! Cold external delay storage (`TransportExtStorage`, `TransportExtRing`) and partitioned
//! in-buffer Schroeder all-pass diffuser banks (`PartitionedSchroederBank`) for `Op-T` transport
//! quads (`ref[dsp.op-t.ring.q0-32-pointer]`, `ref[dsp.op-t.diffuser.partition]`).

mod schroeder;

#[cfg(feature = "alloc")]
use alloc::vec;

use dnam_kernels::scrub_and_flush_f32_bits;

use crate::bin::arena_wire::{TRANSPORT_FLAG_EXT_F32, TransportQuadWire};
use crate::math::{
    FP16_MAX_FINITE_F32, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK,
    IEEE754_F32_TO_FP16_MANTISSA_SHIFT, IEEE754_FP16_EXP_BIAS, IEEE754_FP16_MANTISSA_BITS,
    IEEE754_FP16_MAX_FINITE_BITS, IEEE754_FP16_SIGN_MASK, MathBackend, ceil_f64, clamp_finite_f32,
    const_abs_f64, fp16_to_f32, rcp_nr6_f64,
};
use crate::types::{ColdBuf, VmError};

pub use schroeder::{
    PartitionedSchroederBank, TRANSPORT_ALLPASS_G_MAX, TRANSPORT_SCHROEDER_MAX_DELAY,
    compute_schroeder_requirements,
};

/// Inline `Quad 0` frame-major `fp16` delay ring capacity (`64` frames = `512 B`).
pub const TRANSPORT_INLINE_QUAD0_CAP: usize = 64;

/// Minimum external quad delay ring capacity (`128` frames).
pub const TRANSPORT_MIN_EXT_CAP: usize = 128;

/// Maximum external quad delay ring capacity covering `8×` `48,000` sample lines (`524,288`).
pub const TRANSPORT_MAX_EXT_CAP: usize = 524_288;

const MAX_EXT_POW2_SHIFT: u8 = 19;
const MAX_WIRE_DELAY_48K: f64 = 48_000.0;
const MIN_WIRE_DELAY_48K: f64 = 5.0;
const MAX_LFO_DEPTH_48K: f64 = 512.0;
const FP16_MAX_FINITE_BITS: u32 = FP16_MAX_FINITE_F32.to_bits();

const _: () = assert!(TRANSPORT_INLINE_QUAD0_CAP.is_power_of_two());
const _: () = assert!(TRANSPORT_MIN_EXT_CAP.is_power_of_two());
const _: () = assert!(TRANSPORT_MAX_EXT_CAP.is_power_of_two());
const _: () = assert!((1_usize << MAX_EXT_POW2_SHIFT) == TRANSPORT_MAX_EXT_CAP);

/// Converts finite `f32` to IEEE-754 `fp16` (`u16`) rounding strictly toward zero (`RTZ`) so
/// passive `g_loop <= 0.9990` decays `fp16` ring subnormals cleanly to `0x0000` without limit
/// cycles (`ref[dsp.op-t.guard.fp16-decay]`). Returns `(bits, fp16_saturated)`.
#[inline(always)]
pub const fn f32_to_fp16_rtz_with_sat(val: f32) -> (u16, bool) {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & IEEE754_FP16_SIGN_MASK) as u16;
    let abs_bits = bits & IEEE754_F32_ABS_MASK;
    if abs_bits == 0 {
        return (0, false);
    }
    let exp = ((bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES) as i32;
    if exp == (IEEE754_F32_EXP_ALL_ONES as i32) {
        return (0, false);
    }
    if abs_bits > FP16_MAX_FINITE_BITS {
        return (sign | (IEEE754_FP16_MAX_FINITE_BITS as u16), true);
    }
    let mant = bits & IEEE754_F32_MANTISSA_MASK;
    let new_exp = exp - (IEEE754_F32_EXP_BIAS as i32) + (IEEE754_FP16_EXP_BIAS as i32);
    if new_exp <= 0 {
        if new_exp <= -(IEEE754_FP16_MANTISSA_BITS as i32) {
            return (0, false);
        }
        let m = mant | IEEE754_F32_IMPLICIT_BIT;
        let shift = (((IEEE754_FP16_EXP_BIAS as i32) - 1) - new_exp) as u32;
        let sub = (m >> shift) as u16;
        return (if sub == 0 { 0 } else { sign | sub }, false);
    }
    let half_mant = (mant >> IEEE754_F32_TO_FP16_MANTISSA_SHIFT) as u16;
    (
        sign | ((new_exp as u16) << IEEE754_FP16_MANTISSA_BITS) | half_mant,
        false,
    )
}

/// Computes exact cold sizing `(has_ext, cap)` in `128..=524_288` frames for transport quad
/// `quad_idx` (`op_t_transport.md §1.1.1`).
pub fn compute_quad_ext_requirements(
    quad: &TransportQuadWire,
    quad_idx: usize,
    rho_vm: f64,
    declared_max_delay_48k: u32,
    ext_ring_pow2_frames: u8,
) -> (bool, usize) {
    let scale = if rho_vm.is_finite() && rho_vm > 0.0 {
        rcp_nr6_f64(rho_vm)
    } else {
        1.0
    };
    let mut max_wire_delay_scaled = 0.0_f64;
    let mut max_total_scaled = 0.0_f64;

    for k in 0..4 {
        let raw_d = quad.delay_samples[k].get();
        let d = if raw_d.is_finite() {
            f64::from(raw_d).clamp(MIN_WIRE_DELAY_48K, MAX_WIRE_DELAY_48K)
        } else {
            MIN_WIRE_DELAY_48K
        };
        let raw_s = quad.ramp_spans_samples[k].get();
        let span = if raw_s.is_finite() {
            const_abs_f64(f64::from(raw_s)).min(MAX_WIRE_DELAY_48K)
        } else {
            0.0
        };
        let lfo = f64::from(clamp_finite_f32(
            fp16_to_f32(quad.lfo_depths_fp16[k].get()),
            0.0,
            MAX_LFO_DEPTH_48K as f32,
            0.0,
        ));
        let d_scaled = d * scale;
        if d_scaled > max_wire_delay_scaled {
            max_wire_delay_scaled = d_scaled;
        }
        let total_scaled = (d + 0.5 * span + lfo) * scale;
        if total_scaled > max_total_scaled {
            max_total_scaled = total_scaled;
        }
    }

    let max_cap_f = TRANSPORT_MAX_EXT_CAP as f64;
    let req_frames = (ceil_f64(max_total_scaled.min(max_cap_f)) as usize) + 6;
    let dbrm_pow2_cap = if ext_ring_pow2_frames >= 7 {
        1_usize << u32::from(ext_ring_pow2_frames.min(MAX_EXT_POW2_SHIFT))
    } else {
        0
    };
    let decl_scaled =
        (ceil_f64((f64::from(declared_max_delay_48k) * scale).min(max_cap_f)) as usize) + 8;

    let has_ext = (quad_idx > 0)
        || ((quad.flags & TRANSPORT_FLAG_EXT_F32) != 0)
        || (max_wire_delay_scaled > (TRANSPORT_INLINE_QUAD0_CAP.saturating_sub(8)) as f64)
        || (req_frames > TRANSPORT_INLINE_QUAD0_CAP)
        || (ext_ring_pow2_frames >= 8);

    let cap = if !has_ext {
        TRANSPORT_INLINE_QUAD0_CAP
    } else {
        req_frames
            .max(decl_scaled)
            .max(dbrm_pow2_cap)
            .clamp(TRANSPORT_MIN_EXT_CAP, TRANSPORT_MAX_EXT_CAP)
            .next_power_of_two()
            .clamp(TRANSPORT_MIN_EXT_CAP, TRANSPORT_MAX_EXT_CAP)
    };
    (has_ext, cap)
}

/// Returns required frame-major cold buffer element counts `(fp16_u16_len, f32_len)` for a quad.
#[inline]
pub fn required_ext_elements(
    quad: &TransportQuadWire,
    quad_idx: usize,
    rho_vm: f64,
    declared_max_delay_48k: u32,
    ext_ring_pow2_frames: u8,
) -> (usize, usize) {
    let (has_ext, cap) = compute_quad_ext_requirements(
        quad,
        quad_idx,
        rho_vm,
        declared_max_delay_48k,
        ext_ring_pow2_frames,
    );
    if !has_ext {
        (0, 0)
    } else if (quad.flags & TRANSPORT_FLAG_EXT_F32) != 0 {
        (0, cap * 4)
    } else {
        (cap * 4, 0)
    }
}

/// External cold delay-ring storage backing for [`TransportExtRing`].
#[derive(Debug)]
pub enum TransportExtStorage<'a> {
    /// Inline `Quad 0` ring only (`0 B` external allocation).
    None,
    /// Frame-major `[f0[4], f1[4], ...]` half-precision external storage (`cap * 4` `u16`).
    Fp16(ColdBuf<'a, u16>),
    /// Frame-major `[f0[4], f1[4], ...]` full-precision external storage (`cap * 4` `f32`).
    F32(ColdBuf<'a, f32>),
}

/// Parameters for owned quad external delay ring headroom allocation (`feature = "alloc"`).
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug)]
pub struct ExtRingAllocSpec {
    /// Transport quad index (`0..=3`).
    pub quad_idx: usize,
    /// Minimum inverse oversample / sample-rate ratio used to size frame capacity.
    pub rho_vm: f64,
    /// Header-declared maximum `48 kHz` delay in samples.
    pub declared_max_delay_48k: u32,
    /// Minimum header external ring `pow2` frame count (`<= 19`).
    pub ext_ring_pow2_frames: u8,
    /// Active runtime frame capacity (`None` uses computed minimum).
    pub active_cap: Option<usize>,
    /// Pre-allocated reload headroom frames (`fp16[4]`).
    pub min_fp16_frames: usize,
    /// Pre-allocated reload headroom frames (`f32[4]`).
    pub min_f32_frames: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtStorageMode {
    None,
    Fp16,
    F32,
}

/// External cold delay ring + 32-bit hold counters for a transport quad (`HAS_EXT != 0`).
#[derive(Debug)]
pub struct TransportExtRing<'a> {
    mode: ExtStorageMode,
    fp16_buf: ColdBuf<'a, u16>,
    f32_buf: ColdBuf<'a, f32>,
    /// Power-of-two frame capacity (`128..=524_288`).
    pub cap: usize,
    /// Bitmask `cap - 1` for division-free power-of-two wrap.
    pub mask: usize,
    /// Per-lane 32-bit freeze-hold lag offset counters used when `HAS_EXT` is set.
    pub ext_we_hold_off: [u32; 4],
}

impl<'a> TransportExtRing<'a> {
    /// Constructs an inactive external ring for inline `Quad 0` (`cap = 128`, `0` cold storage).
    #[inline]
    pub const fn none() -> Self {
        Self {
            mode: ExtStorageMode::None,
            fp16_buf: ColdBuf::Borrowed(&mut []),
            f32_buf: ColdBuf::Borrowed(&mut []),
            cap: TRANSPORT_INLINE_QUAD0_CAP,
            mask: TRANSPORT_INLINE_QUAD0_CAP - 1,
            ext_we_hold_off: [0; 4],
        }
    }

    /// Wraps a `ColdBuf<'a, u16>` for `cap` frame-major `fp16[4]` frames, failing closed if
    /// `buf.len() < cap * 4` or `cap` is invalid.
    pub fn from_fp16_buf(mut buf: ColdBuf<'a, u16>, cap: usize) -> Result<Self, VmError> {
        let req = validated_ext_len(cap)?;
        if buf.len() < req {
            return Err(VmError::InsufficientBackingCapacity);
        }
        buf.as_mut_slice()[..req].fill(0);
        Ok(Self {
            mode: ExtStorageMode::Fp16,
            fp16_buf: buf,
            f32_buf: ColdBuf::Borrowed(&mut []),
            cap,
            mask: cap - 1,
            ext_we_hold_off: [0; 4],
        })
    }

    /// Wraps a `ColdBuf<'a, f32>` for `cap` frame-major `f32[4]` frames, failing closed if
    /// `buf.len() < cap * 4` or `cap` is invalid.
    pub fn from_f32_buf(mut buf: ColdBuf<'a, f32>, cap: usize) -> Result<Self, VmError> {
        let req = validated_ext_len(cap)?;
        if buf.len() < req {
            return Err(VmError::InsufficientBackingCapacity);
        }
        buf.as_mut_slice()[..req].fill(0.0);
        Ok(Self {
            mode: ExtStorageMode::F32,
            fp16_buf: ColdBuf::Borrowed(&mut []),
            f32_buf: buf,
            cap,
            mask: cap - 1,
            ext_we_hold_off: [0; 4],
        })
    }

    /// Binds `!alloc` borrowed `fp16` / `f32` slices, preserving both buffers in place for
    /// zero-allocation reloads.
    pub fn from_borrowed_parts(
        fp16_slice: &'a mut [u16],
        f32_slice: &'a mut [f32],
        has_ext: bool,
        want_f32: bool,
        cap: usize,
    ) -> Result<Self, VmError> {
        if !has_ext {
            return Ok(Self {
                mode: ExtStorageMode::None,
                fp16_buf: ColdBuf::borrowed(fp16_slice),
                f32_buf: ColdBuf::borrowed(f32_slice),
                cap: TRANSPORT_INLINE_QUAD0_CAP,
                mask: TRANSPORT_INLINE_QUAD0_CAP - 1,
                ext_we_hold_off: [0; 4],
            });
        }
        let req = validated_ext_len(cap)?;
        if want_f32 {
            if f32_slice.len() < req {
                return Err(VmError::InsufficientBackingCapacity);
            }
            f32_slice[..req].fill(0.0);
            Ok(Self {
                mode: ExtStorageMode::F32,
                fp16_buf: ColdBuf::borrowed(fp16_slice),
                f32_buf: ColdBuf::borrowed(f32_slice),
                cap,
                mask: cap - 1,
                ext_we_hold_off: [0; 4],
            })
        } else {
            if fp16_slice.len() < req {
                return Err(VmError::InsufficientBackingCapacity);
            }
            fp16_slice[..req].fill(0);
            Ok(Self {
                mode: ExtStorageMode::Fp16,
                fp16_buf: ColdBuf::borrowed(fp16_slice),
                f32_buf: ColdBuf::borrowed(f32_slice),
                cap,
                mask: cap - 1,
                ext_we_hold_off: [0; 4],
            })
        }
    }

    /// Allocates owned cold storage sized via [`compute_quad_ext_requirements`] (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_for_quad(
        quad: &TransportQuadWire,
        quad_idx: usize,
        rho_vm: f64,
        declared_max_delay_48k: u32,
        ext_ring_pow2_frames: u8,
    ) -> Result<Self, VmError> {
        Self::new_owned_with_headroom(
            quad,
            &ExtRingAllocSpec {
                quad_idx,
                rho_vm,
                declared_max_delay_48k,
                ext_ring_pow2_frames,
                active_cap: None,
                min_fp16_frames: 0,
                min_f32_frames: 0,
            },
        )
    }

    /// Allocates owned cold storage for a quad with optional reload headroom frames (`fp16` / `f32`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_with_headroom(
        quad: &TransportQuadWire,
        spec: &ExtRingAllocSpec,
    ) -> Result<Self, VmError> {
        let (has_ext, cap_req) = compute_quad_ext_requirements(
            quad,
            spec.quad_idx,
            spec.rho_vm,
            spec.declared_max_delay_48k,
            spec.ext_ring_pow2_frames,
        );
        let want_f32 = (quad.flags & TRANSPORT_FLAG_EXT_F32) != 0;
        let run_cap = spec.active_cap.unwrap_or(cap_req);
        let f32_frames = if has_ext && want_f32 {
            cap_req.max(run_cap).max(spec.min_f32_frames)
        } else {
            spec.min_f32_frames
        };
        let fp16_frames = if has_ext && !want_f32 {
            cap_req.max(run_cap).max(spec.min_fp16_frames)
        } else {
            spec.min_fp16_frames
        };
        if !has_ext && f32_frames == 0 && fp16_frames == 0 {
            return Ok(Self::none());
        }
        let fp16_buf = if fp16_frames > 0 {
            ColdBuf::Owned(vec![0_u16; fp16_frames * 4])
        } else {
            ColdBuf::Borrowed(&mut [])
        };
        let f32_buf = if f32_frames > 0 {
            ColdBuf::Owned(vec![0.0_f32; f32_frames * 4])
        } else {
            ColdBuf::Borrowed(&mut [])
        };
        if !has_ext {
            return Ok(Self {
                mode: ExtStorageMode::None,
                fp16_buf,
                f32_buf,
                cap: TRANSPORT_INLINE_QUAD0_CAP,
                mask: TRANSPORT_INLINE_QUAD0_CAP - 1,
                ext_we_hold_off: [0; 4],
            });
        }
        let _ = validated_ext_len(run_cap)?;
        let mode = if want_f32 {
            ExtStorageMode::F32
        } else {
            ExtStorageMode::Fp16
        };
        Ok(Self {
            mode,
            fp16_buf,
            f32_buf,
            cap: run_cap,
            mask: run_cap - 1,
            ext_we_hold_off: [0; 4],
        })
    }

    /// Returns `true` when external storage is active (`Fp16` or `F32`).
    #[inline]
    pub const fn is_external(&self) -> bool {
        !matches!(self.mode, ExtStorageMode::None)
    }

    /// Returns `true` when external storage uses `32-bit` `f32` frames rather than `fp16` / none.
    #[inline(always)]
    #[must_use]
    pub const fn is_f32_storage(&self) -> bool {
        matches!(self.mode, ExtStorageMode::F32)
    }

    /// Returns `(fp16_ptr, f32_ptr)` base addresses across active/dormant buffers (`0` if unbacked).
    #[inline]
    pub fn backing_storage_addrs(&self) -> (usize, usize) {
        let fp16 = if self.fp16_buf.is_empty() {
            0
        } else {
            self.fp16_buf.as_slice().as_ptr() as usize
        };
        let f32_addr = if self.f32_buf.is_empty() {
            0
        } else {
            self.f32_buf.as_slice().as_ptr() as usize
        };
        (fp16, f32_addr)
    }

    /// Returns active backing capacity in frames (`0` when [`ExtStorageMode::None`]).
    #[inline]
    pub fn backing_frame_capacity(&self) -> usize {
        match self.mode {
            ExtStorageMode::None => 0,
            ExtStorageMode::Fp16 => self.fp16_buf.len() >> 2,
            ExtStorageMode::F32 => self.f32_buf.len() >> 2,
        }
    }

    /// Returns available frame capacity for `want_f32` (`0` if unbacked for that format).
    #[inline]
    pub fn backing_frame_capacity_for(&self, want_f32: bool) -> usize {
        if want_f32 {
            self.f32_buf.len() >> 2
        } else {
            self.fp16_buf.len() >> 2
        }
    }

    /// Deactivates active external mode while preserving backing buffers in place (`0` drops/allocs).
    #[inline]
    pub fn deactivate_preserving_storage(&mut self) {
        self.mode = ExtStorageMode::None;
        self.cap = TRANSPORT_INLINE_QUAD0_CAP;
        self.mask = TRANSPORT_INLINE_QUAD0_CAP - 1;
        self.ext_we_hold_off = [0; 4];
    }

    /// Reconfigures active external ring format and frame capacity in-place (`0` alloc, `0` drop).
    pub fn reconfigure_for_reload(
        &mut self,
        has_ext: bool,
        want_f32: bool,
        new_cap: usize,
    ) -> Result<(), VmError> {
        if !has_ext {
            self.deactivate_preserving_storage();
            return Ok(());
        }
        let _ = validated_ext_len(new_cap)?;
        if self.backing_frame_capacity_for(want_f32) < new_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        self.mode = if want_f32 {
            ExtStorageMode::F32
        } else {
            ExtStorageMode::Fp16
        };
        self.cap = new_cap;
        self.mask = new_cap - 1;
        self.reset();
        Ok(())
    }

    /// Validates backing capacity for `new_cap` frames, updates `cap`/`mask`, and zeroes state.
    pub fn reconfigure_cap(&mut self, new_cap: usize) -> Result<(), VmError> {
        let req = validated_ext_len(new_cap)?;
        let has_cap = match self.mode {
            ExtStorageMode::None => false,
            ExtStorageMode::Fp16 => self.fp16_buf.len() >= req,
            ExtStorageMode::F32 => self.f32_buf.len() >= req,
        };
        if !has_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        self.cap = new_cap;
        self.mask = new_cap - 1;
        self.reset();
        Ok(())
    }

    /// Clears active ring contents and `ext_we_hold_off` while preserving capacity/format.
    #[inline]
    pub fn reset(&mut self) {
        let req = self.cap * 4;
        match self.mode {
            ExtStorageMode::None => {}
            ExtStorageMode::Fp16 => {
                let s = self.fp16_buf.as_mut_slice();
                let len = s.len().min(req);
                s[..len].fill(0);
            }
            ExtStorageMode::F32 => {
                let s = self.f32_buf.as_mut_slice();
                let len = s.len().min(req);
                s[..len].fill(0.0);
            }
        }
        self.ext_we_hold_off = [0; 4];
    }

    /// Reads one sample at frame `frame_idx` (wrapped via `& self.mask`) and lane `lane < 4`.
    #[inline(always)]
    pub fn read_lane_tap(&self, frame_idx: usize, lane: usize) -> f32 {
        let slot = ((frame_idx & self.mask) * 4) + (lane & 3);
        match self.mode {
            ExtStorageMode::None => 0.0,
            ExtStorageMode::Fp16 => self
                .fp16_buf
                .as_slice()
                .get(slot)
                .copied()
                .map(fp16_to_f32)
                .unwrap_or(0.0),
            ExtStorageMode::F32 => self.f32_buf.as_slice().get(slot).copied().unwrap_or(0.0),
        }
    }

    /// Reads 4 backward-wrap Catmull-Rom taps `s[-1, 0, +1, +2]` at `(wr + d_int + o) & mask`
    /// using vectorized `B::fp16x4_to_f32x4` decoding for `fp16` backing rings.
    #[inline(always)]
    pub fn read_catmull_rom_4_simd<B: MathBackend>(
        &self,
        wr: u32,
        d_int: usize,
        lane: usize,
    ) -> [f32; 4] {
        let base = (wr as usize).wrapping_add(d_int);
        let l = lane & 3;
        let i0 = ((base.wrapping_sub(1) & self.mask) * 4) + l;
        let i1 = ((base & self.mask) * 4) + l;
        let i2 = ((base.wrapping_add(1) & self.mask) * 4) + l;
        let i3 = ((base.wrapping_add(2) & self.mask) * 4) + l;
        match self.mode {
            ExtStorageMode::None => [0.0; 4],
            ExtStorageMode::Fp16 => {
                let s = self.fp16_buf.as_slice();
                let words = [
                    s.get(i0).copied().unwrap_or(0),
                    s.get(i1).copied().unwrap_or(0),
                    s.get(i2).copied().unwrap_or(0),
                    s.get(i3).copied().unwrap_or(0),
                ];
                B::fp16x4_to_f32x4(&words)
            }
            ExtStorageMode::F32 => {
                let s = self.f32_buf.as_slice();
                [
                    s.get(i0).copied().unwrap_or(0.0),
                    s.get(i1).copied().unwrap_or(0.0),
                    s.get(i2).copied().unwrap_or(0.0),
                    s.get(i3).copied().unwrap_or(0.0),
                ]
            }
        }
    }

    /// Reads 4 backward-wrap Catmull-Rom taps `s[-1, 0, +1, +2]` at `(wr + d_int + o) & mask`.
    #[inline(always)]
    pub fn read_catmull_rom_4(&self, wr: u32, d_int: usize, lane: usize) -> [f32; 4] {
        let base = (wr as usize).wrapping_add(d_int);
        [
            self.read_lane_tap(base.wrapping_sub(1), lane),
            self.read_lane_tap(base, lane),
            self.read_lane_tap(base.wrapping_add(1), lane),
            self.read_lane_tap(base.wrapping_add(2), lane),
        ]
    }

    /// Writes 4 lane feedback values into frame `wr & mask`, returning `true` if `fp16` saturated.
    #[inline(always)]
    pub fn write_frame(&mut self, wr: u32, vals: [f32; 4]) -> bool {
        let base = ((wr as usize) & self.mask) * 4;
        match self.mode {
            ExtStorageMode::None => false,
            ExtStorageMode::Fp16 => {
                let s = self.fp16_buf.as_mut_slice();
                if base + 4 > s.len() {
                    return false;
                }
                let mut sat = false;
                for k in 0..4 {
                    let (h, did_sat) = f32_to_fp16_rtz_with_sat(scrub_and_flush_f32_bits(vals[k]));
                    s[base + k] = h;
                    sat |= did_sat;
                }
                sat
            }
            ExtStorageMode::F32 => {
                let s = self.f32_buf.as_mut_slice();
                if base + 4 > s.len() {
                    return false;
                }
                for k in 0..4 {
                    s[base + k] = scrub_and_flush_f32_bits(vals[k])
                        .clamp(-FP16_MAX_FINITE_F32, FP16_MAX_FINITE_F32);
                }
                false
            }
        }
    }
}

#[inline]
fn validated_ext_len(cap: usize) -> Result<usize, VmError> {
    if !cap.is_power_of_two() || !(TRANSPORT_MIN_EXT_CAP..=TRANSPORT_MAX_EXT_CAP).contains(&cap) {
        return Err(VmError::InsufficientBackingCapacity);
    }
    Ok(cap * 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::f32_to_fp16;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn base_wire(delays: [f32; 4], ap_delays: [u8; 4], flags: u8) -> TransportQuadWire {
        let mut wire = TransportQuadWire::new_zeroed();
        wire.delay_samples = delays.map(F32::new);
        wire.allpass_delays = ap_delays;
        wire.allpass_g_fp16 = [0.6, -0.5, 0.7, -0.65].map(|g| U16::new(f32_to_fp16(g)));
        wire.flags = flags;
        wire
    }

    /// verify[dsp.op-t.ring.branchless-wrap]
    /// verify[dsp.op-t.ring.capacity]
    #[test]
    fn test_quad_ext_sizing_boundaries_and_fail_closed() {
        let q0_inline = base_wire([23.0, 37.0, 49.0, 56.0], [0; 4], 0);
        assert_eq!(
            compute_quad_ext_requirements(&q0_inline, 0, 1.0, 56, 6),
            (false, 64)
        );
        assert_eq!(required_ext_elements(&q0_inline, 0, 1.0, 56, 6), (0, 0));

        // Delay > 56 on Quad 0 forces external fp16 allocation.
        let q0_57 = base_wire([23.0, 37.0, 49.0, 57.0], [0; 4], 0);
        assert_eq!(
            compute_quad_ext_requirements(&q0_57, 0, 1.0, 57, 6),
            (true, 128)
        );
        assert_eq!(required_ext_elements(&q0_57, 0, 1.0, 57, 6), (512, 0));

        // 48,000 samples @ 384 kHz (rho_vm = 0.125) caps at 524,288 frames.
        let q1_ultra = base_wire([48_000.0; 4], [0; 4], TRANSPORT_FLAG_EXT_F32);
        assert_eq!(
            compute_quad_ext_requirements(&q1_ultra, 1, 0.125, 48_000, 19),
            (true, 524_288)
        );
        assert_eq!(
            required_ext_elements(&q1_ultra, 1, 0.125, 48_000, 19),
            (0, 524_288 * 4)
        );

        let mut short_u16 = [0_u16; 511];
        assert_eq!(
            TransportExtRing::from_fp16_buf(ColdBuf::borrowed(&mut short_u16), 128).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
        let mut short_f32 = [0.0_f32; 511];
        assert_eq!(
            TransportExtRing::from_f32_buf(ColdBuf::borrowed(&mut short_f32), 128).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
    }

    /// verify[dsp.op-t.guard.fp16-decay]
    /// verify[dsp.op-t.ring.fp16-saturation]
    #[test]
    fn test_fp16_rtz_saturation_and_subnormal_decay_to_zero() {
        assert_eq!(f32_to_fp16_rtz_with_sat(65_504.0), (0x7BFF, false));
        assert_eq!(f32_to_fp16_rtz_with_sat(65_505.0), (0x7BFF, true));
        assert_eq!(f32_to_fp16_rtz_with_sat(-65_505.0), (0xFBFF, true));

        let mut h = 0x0400_u16; // Smallest positive normal fp16 (`2^-14`).
        for _ in 0..2048 {
            if h == 0 {
                break;
            }
            let next_val = 0.9990_f32 * fp16_to_f32(h);
            h = f32_to_fp16_rtz_with_sat(next_val).0;
        }
        assert_eq!(h, 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_owned_alloc_helpers_construct_exact_capacities() {
        let q_f32 = base_wire([200.0; 4], [19, 31, 0, 47], TRANSPORT_FLAG_EXT_F32);
        let ring = TransportExtRing::new_owned_for_quad(&q_f32, 0, 1.0, 200, 7).unwrap();
        assert!(ring.is_external());
        assert!(ring.is_f32_storage());
        assert_eq!(ring.cap, 256);
        let q_fp16 = base_wire([200.0; 4], [19, 31, 0, 47], 0);
        let fp16_ring = TransportExtRing::new_owned_for_quad(&q_fp16, 0, 1.0, 200, 7).unwrap();
        assert!(fp16_ring.is_external());
        assert!(!fp16_ring.is_f32_storage());
        assert!(!TransportExtRing::none().is_f32_storage());
    }
}
