//! Cold external delay storage (`TankExtStorage`, `TankExtRing`) and partitioned in-buffer
//! Schroeder all-pass diffuser banks (`PartitionedSchroederBank`) for `Op-T` spatial tank quads
//! (`ref[dsp.op-t.ring.q0-32-pointer]`, `ref[dsp.op-t.diffuser.partition]`).

#[cfg(feature = "alloc")]
use alloc::vec;

use dnam_kernels::{scrub_and_flush_f32_bits, step_schroeder_quad_vec4};

use crate::bin::arena_wire::{TANK_FLAG_EXT_F32, TANK_FLAG_SCHROEDER_ENABLE, TankQuadWire};
use crate::math::{
    FP16_MAX_FINITE_F32, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS,
    IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_BITS, IEEE754_F32_MANTISSA_MASK,
    IEEE754_F32_TO_FP16_MANTISSA_SHIFT, IEEE754_FP16_EXP_BIAS, IEEE754_FP16_MANTISSA_BITS,
    IEEE754_FP16_MAX_FINITE_BITS, IEEE754_FP16_SIGN_MASK, MathBackend, SimdVec4, ceil_f64,
    clamp_finite_f32, const_abs_f64, fp16_to_f32, rcp_nr6_f64, round_half_away_f64,
};
use crate::types::{ColdBuf, VmError};

/// Inline `Quad 0` frame-major `fp16` delay ring capacity (`128` frames = `1,024 B`).
pub const TANK_INLINE_QUAD0_CAP: usize = 128;

/// Minimum external quad delay ring capacity (`128` frames).
pub const TANK_MIN_EXT_CAP: usize = 128;

/// Maximum external quad delay ring capacity covering `8×` `48,000` sample lines (`524,288`).
pub const TANK_MAX_EXT_CAP: usize = 524_288;

/// Maximum reference `48 kHz` Schroeder delay length (`255`) scaled up to `384 kHz` (`16,384`).
pub const TANK_SCHROEDER_MAX_DELAY: usize = 16_384;

/// Maximum absolute Schroeder all-pass coefficient magnitude (`0.75`).
pub const TANK_ALLPASS_G_MAX: f32 = 0.75;

const MAX_EXT_POW2_SHIFT: u8 = 19;
const MAX_WIRE_DELAY_48K: f64 = 48_000.0;
const MIN_WIRE_DELAY_48K: f64 = 5.0;
const MAX_LFO_DEPTH_48K: f64 = 512.0;
const FP16_MAX_FINITE_BITS: u32 = FP16_MAX_FINITE_F32.to_bits();

const _: () = assert!(TANK_INLINE_QUAD0_CAP.is_power_of_two());
const _: () = assert!(TANK_MIN_EXT_CAP.is_power_of_two());
const _: () = assert!(TANK_MAX_EXT_CAP.is_power_of_two());
const _: () = assert!((1_usize << MAX_EXT_POW2_SHIFT) == TANK_MAX_EXT_CAP);

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

/// Computes exact cold sizing `(has_ext, cap)` in `128..=524_288` frames for spatial tank quad
/// `quad_idx` (`op_t_transport.md §1.1.1`).
pub fn compute_quad_ext_requirements(
    quad: &TankQuadWire,
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

    let max_cap_f = TANK_MAX_EXT_CAP as f64;
    let req_frames = (ceil_f64(max_total_scaled.min(max_cap_f)) as usize) + 6;
    let dbrm_pow2_cap = if ext_ring_pow2_frames >= 7 {
        1_usize << u32::from(ext_ring_pow2_frames.min(MAX_EXT_POW2_SHIFT))
    } else {
        0
    };
    let decl_scaled =
        (ceil_f64((f64::from(declared_max_delay_48k) * scale).min(max_cap_f)) as usize) + 8;

    let has_ext = (quad_idx > 0)
        || ((quad.flags & TANK_FLAG_EXT_F32) != 0)
        || (max_wire_delay_scaled > 120.0)
        || (req_frames > TANK_INLINE_QUAD0_CAP)
        || (ext_ring_pow2_frames >= 8);

    let cap = if !has_ext {
        TANK_INLINE_QUAD0_CAP
    } else {
        req_frames
            .max(decl_scaled)
            .max(dbrm_pow2_cap)
            .clamp(TANK_MIN_EXT_CAP, TANK_MAX_EXT_CAP)
            .next_power_of_two()
            .clamp(TANK_MIN_EXT_CAP, TANK_MAX_EXT_CAP)
    };
    (has_ext, cap)
}

/// Returns required frame-major cold buffer element counts `(fp16_u16_len, f32_len)` for a quad.
#[inline]
pub fn required_ext_elements(
    quad: &TankQuadWire,
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
    } else if (quad.flags & TANK_FLAG_EXT_F32) != 0 {
        (0, cap * 4)
    } else {
        (cap * 4, 0)
    }
}

/// Computes rate-scaled Schroeder delays, clamped gains, per-lane power-of-two capacity, and
/// total `f32` element count (`4 * cap_per_lane`) for [`PartitionedSchroederBank`].
#[inline]
pub fn compute_schroeder_requirements(
    quad: &TankQuadWire,
    rho_vm: f64,
) -> ([usize; 4], [f32; 4], usize, usize) {
    let scale = if rho_vm.is_finite() && rho_vm > 0.0 {
        rcp_nr6_f64(rho_vm)
    } else {
        1.0
    };
    let mut delays = [0_usize; 4];
    let mut allpass_g = [0.0_f32; 4];
    let mut max_delay = 0_usize;

    for k in 0..4 {
        let d48 = quad.allpass_delays[k];
        if d48 > 0 {
            let scaled = (round_half_away_f64(f64::from(d48) * scale) as usize)
                .clamp(1, TANK_SCHROEDER_MAX_DELAY);
            delays[k] = scaled;
            if scaled > max_delay {
                max_delay = scaled;
            }
        }
        let g_raw = fp16_to_f32(quad.allpass_g_fp16[k].get());
        allpass_g[k] = clamp_finite_f32(g_raw, -TANK_ALLPASS_G_MAX, TANK_ALLPASS_G_MAX, 0.0);
    }

    let cap_per_lane = (max_delay + 1).max(1).next_power_of_two();
    let total_elements = cap_per_lane * 4;
    (delays, allpass_g, cap_per_lane, total_elements)
}

/// External cold delay-ring storage backing for [`TankExtRing`].
#[derive(Debug)]
pub enum TankExtStorage<'a> {
    /// Inline `Quad 0` ring only (`0 B` external allocation).
    None,
    /// Frame-major `[f0[4], f1[4], ...]` half-precision external storage (`cap * 4` `u16`).
    Fp16(ColdBuf<'a, u16>),
    /// Frame-major `[f0[4], f1[4], ...]` full-precision external storage (`cap * 4` `f32`).
    F32(ColdBuf<'a, f32>),
}

/// External cold delay ring + 32-bit `WE` hold counters for a spatial tank quad (`HAS_EXT != 0`).
#[derive(Debug)]
pub struct TankExtRing<'a> {
    /// Backing ring storage (`None`, `Fp16`, or `F32`).
    pub storage: TankExtStorage<'a>,
    /// Power-of-two frame capacity (`128..=524_288`).
    pub cap: usize,
    /// Bitmask `cap - 1` for division-free power-of-two wrap.
    pub mask: usize,
    /// Per-lane 32-bit `WE` hold lag offset counters used when `HAS_EXT` is set.
    pub ext_we_hold_off: [u32; 4],
}

impl<'a> TankExtRing<'a> {
    /// Constructs an inactive external ring for inline `Quad 0` (`cap = 128`, `0` cold storage).
    #[inline]
    pub const fn none() -> Self {
        Self {
            storage: TankExtStorage::None,
            cap: TANK_INLINE_QUAD0_CAP,
            mask: TANK_INLINE_QUAD0_CAP - 1,
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
            storage: TankExtStorage::Fp16(buf),
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
            storage: TankExtStorage::F32(buf),
            cap,
            mask: cap - 1,
            ext_we_hold_off: [0; 4],
        })
    }

    /// Allocates owned cold storage sized via [`compute_quad_ext_requirements`] (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_for_quad(
        quad: &TankQuadWire,
        quad_idx: usize,
        rho_vm: f64,
        declared_max_delay_48k: u32,
        ext_ring_pow2_frames: u8,
    ) -> Result<Self, VmError> {
        let (has_ext, cap) = compute_quad_ext_requirements(
            quad,
            quad_idx,
            rho_vm,
            declared_max_delay_48k,
            ext_ring_pow2_frames,
        );
        if !has_ext {
            Ok(Self::none())
        } else if (quad.flags & TANK_FLAG_EXT_F32) != 0 {
            Self::from_f32_buf(ColdBuf::Owned(vec![0.0_f32; cap * 4]), cap)
        } else {
            Self::from_fp16_buf(ColdBuf::Owned(vec![0_u16; cap * 4]), cap)
        }
    }

    /// Returns `true` when external storage is allocated (`Fp16` or `F32`).
    #[inline]
    pub const fn is_external(&self) -> bool {
        !matches!(self.storage, TankExtStorage::None)
    }

    /// Returns `true` when external storage uses `32-bit` `f32` frames rather than `fp16` / none.
    #[inline(always)]
    #[must_use]
    pub const fn is_f32_storage(&self) -> bool {
        matches!(self.storage, TankExtStorage::F32(_))
    }

    /// Returns allocated backing capacity in frames (`0` when [`TankExtStorage::None`]).
    #[inline]
    pub fn backing_frame_capacity(&self) -> usize {
        match &self.storage {
            TankExtStorage::None => 0,
            TankExtStorage::Fp16(buf) => buf.len() >> 2,
            TankExtStorage::F32(buf) => buf.len() >> 2,
        }
    }

    /// Validates backing capacity for `new_cap` frames, updates `cap`/`mask`, and zeroes state.
    pub fn reconfigure_cap(&mut self, new_cap: usize) -> Result<(), VmError> {
        let req = validated_ext_len(new_cap)?;
        let has_cap = match &self.storage {
            TankExtStorage::None => false,
            TankExtStorage::Fp16(buf) => buf.len() >= req,
            TankExtStorage::F32(buf) => buf.len() >= req,
        };
        if !has_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        self.cap = new_cap;
        self.mask = new_cap - 1;
        self.reset();
        Ok(())
    }

    /// Clears ring contents and `ext_we_hold_off` while preserving capacity/format.
    #[inline]
    pub fn reset(&mut self) {
        let req = self.cap * 4;
        match &mut self.storage {
            TankExtStorage::None => {}
            TankExtStorage::Fp16(buf) => {
                let len = buf.len().min(req);
                buf.as_mut_slice()[..len].fill(0);
            }
            TankExtStorage::F32(buf) => {
                let len = buf.len().min(req);
                buf.as_mut_slice()[..len].fill(0.0);
            }
        }
        self.ext_we_hold_off = [0; 4];
    }

    /// Reads one sample at frame `frame_idx` (wrapped via `& self.mask`) and lane `lane < 4`.
    #[inline(always)]
    pub fn read_lane_tap(&self, frame_idx: usize, lane: usize) -> f32 {
        let slot = ((frame_idx & self.mask) * 4) + (lane & 3);
        match &self.storage {
            TankExtStorage::None => 0.0,
            TankExtStorage::Fp16(buf) => buf
                .as_slice()
                .get(slot)
                .copied()
                .map(fp16_to_f32)
                .unwrap_or(0.0),
            TankExtStorage::F32(buf) => buf.as_slice().get(slot).copied().unwrap_or(0.0),
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
        match &self.storage {
            TankExtStorage::None => [0.0; 4],
            TankExtStorage::Fp16(buf) => {
                let s = buf.as_slice();
                let words = [
                    s.get(i0).copied().unwrap_or(0),
                    s.get(i1).copied().unwrap_or(0),
                    s.get(i2).copied().unwrap_or(0),
                    s.get(i3).copied().unwrap_or(0),
                ];
                B::fp16x4_to_f32x4(&words)
            }
            TankExtStorage::F32(buf) => {
                let s = buf.as_slice();
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
        match &mut self.storage {
            TankExtStorage::None => false,
            TankExtStorage::Fp16(buf) => {
                let s = buf.as_mut_slice();
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
            TankExtStorage::F32(buf) => {
                let s = buf.as_mut_slice();
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
    if !cap.is_power_of_two() || !(TANK_MIN_EXT_CAP..=TANK_MAX_EXT_CAP).contains(&cap) {
        return Err(VmError::InsufficientBackingCapacity);
    }
    Ok(cap * 4)
}

/// Partitioned 4-lane frame-major in-buffer Schroeder all-pass diffuser bank
/// (`ref[dsp.op-t.diffuser.partition]`). Lanes with `delays[k] == 0` bypass cleanly (`v_k = x_k`).
#[derive(Debug)]
pub struct PartitionedSchroederBank<'a> {
    buf: ColdBuf<'a, f32>,
    cap_per_lane: usize,
    mask: usize,
    write_pos: usize,
    delays: [usize; 4],
    allpass_g: [f32; 4],
}

impl<'a> PartitionedSchroederBank<'a> {
    /// Constructs a bypassed/disabled Schroeder diffuser bank (`0` storage floats).
    #[inline]
    pub fn disabled() -> Self {
        Self {
            buf: ColdBuf::borrowed(&mut []),
            cap_per_lane: 0,
            mask: 0,
            write_pos: 0,
            delays: [0; 4],
            allpass_g: [0.0; 4],
        }
    }

    /// Hydrates a [`PartitionedSchroederBank<'a>`] from [`ColdBuf<'a, f32>`], failing closed with
    /// [`VmError::InsufficientBackingCapacity`] if `buf.len() < 4 * cap_per_lane`.
    pub fn from_cold_buf(
        mut buf: ColdBuf<'a, f32>,
        quad: &TankQuadWire,
        rho_vm: f64,
    ) -> Result<Self, VmError> {
        let (delays, allpass_g, cap_per_lane, req_len) =
            compute_schroeder_requirements(quad, rho_vm);
        if buf.len() < req_len {
            return Err(VmError::InsufficientBackingCapacity);
        }
        buf.as_mut_slice()[..req_len].fill(0.0);
        Ok(Self {
            buf,
            cap_per_lane,
            mask: cap_per_lane - 1,
            write_pos: 0,
            delays,
            allpass_g,
        })
    }

    /// Convenience constructor wrapping `&'a mut [f32]` without `alloc`.
    #[inline]
    pub fn from_slice(
        buf: &'a mut [f32],
        quad: &TankQuadWire,
        rho_vm: f64,
    ) -> Result<Self, VmError> {
        Self::from_cold_buf(ColdBuf::borrowed(buf), quad, rho_vm)
    }

    /// Allocates owned Schroeder storage sized for `(quad, rho_vm)` (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_for_quad(quad: &TankQuadWire, rho_vm: f64) -> Result<Self, VmError> {
        let (_, _, _, req_len) = compute_schroeder_requirements(quad, rho_vm);
        Self::from_cold_buf(ColdBuf::Owned(vec![0.0_f32; req_len]), quad, rho_vm)
    }

    /// Returns per-lane power-of-two ring capacity (`0` when disabled).
    #[inline]
    pub const fn cap_per_lane(&self) -> usize {
        self.cap_per_lane
    }

    /// Returns allocated cold buffer capacity in `f32` elements (`0` when disabled).
    #[inline]
    pub fn backing_f32_capacity(&self) -> usize {
        self.buf.len()
    }

    /// Returns converted integer delays per lane (`0` means clean bypass on that lane).
    #[inline]
    pub const fn delays(&self) -> [usize; 4] {
        self.delays
    }

    /// Returns clamped Schroeder gains per lane in `[-0.75, 0.75]`.
    #[inline]
    pub const fn allpass_g(&self) -> [f32; 4] {
        self.allpass_g
    }

    /// Updates active Schroeder all-pass coefficients (`[-0.75, 0.75]`), e.g. from `slot 25`.
    #[inline]
    pub fn set_allpass_g(&mut self, gains: [f32; 4]) {
        for (dst, &g) in self.allpass_g.iter_mut().zip(gains.iter()) {
            *dst = clamp_finite_f32(g, -TANK_ALLPASS_G_MAX, TANK_ALLPASS_G_MAX, 0.0);
        }
    }

    /// Recomputes ZOH-scaled Schroeder delays/gains and validates capacity for `rho_vm`.
    pub fn reconfigure_sample_rate(
        &mut self,
        quad: &TankQuadWire,
        rho_vm: f64,
    ) -> Result<(), VmError> {
        if (quad.flags & TANK_FLAG_SCHROEDER_ENABLE) == 0 {
            self.cap_per_lane = 0;
            self.mask = 0;
            self.write_pos = 0;
            self.delays = [0; 4];
            self.allpass_g = [0.0; 4];
            return Ok(());
        }
        let (delays, allpass_g, cap_per_lane, req_len) =
            compute_schroeder_requirements(quad, rho_vm);
        if self.buf.len() < req_len {
            return Err(VmError::InsufficientBackingCapacity);
        }
        self.buf.as_mut_slice()[..req_len].fill(0.0);
        self.cap_per_lane = cap_per_lane;
        self.mask = cap_per_lane - 1;
        self.write_pos = 0;
        self.delays = delays;
        self.allpass_g = allpass_g;
        Ok(())
    }

    /// Zeroes Schroeder all-pass ring state and resets `write_pos = 0`.
    #[inline]
    pub fn reset(&mut self) {
        let req = self.cap_per_lane * 4;
        let len = self.buf.len().min(req);
        self.buf.as_mut_slice()[..len].fill(0.0);
        self.write_pos = 0;
    }

    /// Steps 4 parallel Schroeder diffuser lanes simultaneously, preserving clean bypass (`y_k = x_k`)
    /// for any lane with `delays[k] == 0`.
    #[inline(always)]
    pub fn step_vec4<B: MathBackend>(&mut self, x: B::Vec4) -> B::Vec4 {
        let x_clean = x.scrub_recursive_state();
        if self.cap_per_lane == 0 || self.buf.len() < self.cap_per_lane * 4 || self.delays == [0; 4]
        {
            return x_clean;
        }
        let s = self.buf.as_slice();
        let mut delayed = [0.0_f32; 4];
        for k in 0..4 {
            let dk = self.delays[k];
            if dk > 0 {
                let r_idx = self.write_pos.wrapping_sub(dk) & self.mask;
                delayed[k] = s[r_idx * 4 + k];
            }
        }
        let g_vec = B::Vec4::load(&self.allpass_g);
        let (y_vec, write_vec) = step_schroeder_quad_vec4(x_clean, B::Vec4::load(&delayed), g_vec);
        let write_arr = write_vec.to_array();
        let base = self.write_pos * 4;
        self.buf.as_mut_slice()[base..base + 4].copy_from_slice(&write_arr);
        self.write_pos = (self.write_pos + 1) & self.mask;

        if self.delays[0] == 0 || self.delays[1] == 0 || self.delays[2] == 0 || self.delays[3] == 0
        {
            let x_arr = x_clean.to_array();
            let mut y_arr = y_vec.to_array();
            for k in 0..4 {
                if self.delays[k] == 0 {
                    y_arr[k] = x_arr[k];
                }
            }
            return B::Vec4::load(&y_arr);
        }
        y_vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend, f32_to_fp16};
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn base_wire(delays: [f32; 4], ap_delays: [u8; 4], flags: u8) -> TankQuadWire {
        let mut wire = TankQuadWire::new_zeroed();
        wire.delay_samples = delays.map(F32::new);
        wire.allpass_delays = ap_delays;
        wire.allpass_g_fp16 = [0.6, -0.5, 0.7, -0.65].map(|g| U16::new(f32_to_fp16(g)));
        wire.flags = flags;
        wire
    }

    #[test]
    fn test_quad_ext_sizing_boundaries_and_fail_closed() {
        let q0_inline = base_wire([53.0, 67.0, 79.0, 120.0], [0; 4], 0);
        assert_eq!(
            compute_quad_ext_requirements(&q0_inline, 0, 1.0, 120, 7),
            (false, 128)
        );
        assert_eq!(required_ext_elements(&q0_inline, 0, 1.0, 120, 7), (0, 0));

        // Delay > 120 on Quad 0 forces external fp16 allocation.
        let q0_121 = base_wire([53.0, 67.0, 79.0, 121.0], [0; 4], 0);
        assert_eq!(
            compute_quad_ext_requirements(&q0_121, 0, 1.0, 121, 7),
            (true, 256)
        );
        assert_eq!(required_ext_elements(&q0_121, 0, 1.0, 121, 7), (1_024, 0));

        // 48,000 samples @ 384 kHz (rho_vm = 0.125) caps at 524,288 frames.
        let q1_ultra = base_wire([48_000.0; 4], [0; 4], TANK_FLAG_EXT_F32);
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
            TankExtRing::from_fp16_buf(ColdBuf::borrowed(&mut short_u16), 128).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
        let mut short_f32 = [0.0_f32; 511];
        assert_eq!(
            TankExtRing::from_f32_buf(ColdBuf::borrowed(&mut short_f32), 128).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
    }

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

    #[test]
    fn test_schroeder_bypass_energy_preservation_and_fail_closed() {
        let q_mix = base_wire([53.0; 4], [0, 17, 0, 31], 1);
        let (delays, _, cap_per_lane, req_len) = compute_schroeder_requirements(&q_mix, 0.5);
        assert_eq!(delays, [0, 34, 0, 62]);
        assert_eq!(cap_per_lane, 64);
        assert_eq!(req_len, 256);

        let mut short = [0.0_f32; 255];
        assert_eq!(
            PartitionedSchroederBank::from_slice(&mut short, &q_mix, 0.5).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );

        let mut buf = [0.0_f32; 256];
        let mut ap = PartitionedSchroederBank::from_slice(&mut buf, &q_mix, 0.5).unwrap();
        let (mut e_in, mut e_out) = ([0.0_f64; 4], [0.0_f64; 4]);
        for n in 0..512 {
            let x = if n == 0 {
                [1.0, 0.5, -0.75, 0.25]
            } else {
                [0.0; 4]
            };
            let y = ap.step_vec4::<DefaultMathBackend>(
                <DefaultMathBackend as MathBackend>::Vec4::load(&x),
            );
            let ya = <DefaultMathBackend as MathBackend>::Vec4::to_array(y);
            // Zero-delay lanes 0 and 2 bypass verbatim.
            assert_eq!(ya[0].to_bits(), x[0].to_bits());
            assert_eq!(ya[2].to_bits(), x[2].to_bits());
            for k in 0..4 {
                e_in[k] += f64::from(x[k]) * f64::from(x[k]);
                e_out[k] += f64::from(ya[k]) * f64::from(ya[k]);
            }
        }
        for k in 0..4 {
            assert!(
                const_abs_f64(e_out[k] - e_in[k]) < 1.0e-4,
                "lane {k}: in={}, out={}",
                e_in[k],
                e_out[k]
            );
        }
        ap.reset();
        let y0 = ap.step_vec4::<ScalarBackend>(<ScalarBackend as MathBackend>::Vec4::splat(0.0));
        assert_eq!(<ScalarBackend as MathBackend>::Vec4::to_array(y0), [0.0; 4]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_owned_alloc_helpers_construct_exact_capacities() {
        let q_f32 = base_wire([200.0; 4], [19, 31, 0, 47], TANK_FLAG_EXT_F32);
        let ring = TankExtRing::new_owned_for_quad(&q_f32, 0, 1.0, 200, 7).unwrap();
        assert!(ring.is_external());
        assert!(ring.is_f32_storage());
        assert_eq!(ring.cap, 256);
        let q_fp16 = base_wire([200.0; 4], [19, 31, 0, 47], 0);
        let fp16_ring = TankExtRing::new_owned_for_quad(&q_fp16, 0, 1.0, 200, 7).unwrap();
        assert!(fp16_ring.is_external());
        assert!(!fp16_ring.is_f32_storage());
        assert!(!TankExtRing::none().is_f32_storage());
        let sch = PartitionedSchroederBank::new_owned_for_quad(&q_f32, 1.0).unwrap();
        assert_eq!(sch.cap_per_lane(), 64);
    }
}
