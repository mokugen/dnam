//! Partitioned 4-lane frame-major in-buffer Schroeder all-pass diffuser bank
//! ([`PartitionedSchroederBank`]) and rate-scaled capacity sizing (`ref[dsp.op-t.diffuser.partition]`).

#[cfg(feature = "alloc")]
use alloc::vec;

use dnam_kernels::step_schroeder_quad_vec4;

use crate::bin::arena_wire::{TRANSPORT_FLAG_SCHROEDER_ENABLE, TransportQuadWire};
use crate::math::{
    MathBackend, SimdVec4, clamp_finite_f32, fp16_to_f32, rcp_nr6_f64, round_half_away_f64,
};
use crate::types::{ColdBuf, VmError};

/// Maximum reference `48 kHz` Schroeder delay length (`255`) scaled up to `384 kHz` (`16,384`).
pub const TRANSPORT_SCHROEDER_MAX_DELAY: usize = 16_384;

/// Maximum absolute Schroeder all-pass coefficient magnitude (`0.75`).
pub const TRANSPORT_ALLPASS_G_MAX: f32 = 0.75;

/// Computes rate-scaled Schroeder delays, clamped gains, per-lane power-of-two capacity, and
/// total `f32` element count (`4 * cap_per_lane`) for [`PartitionedSchroederBank`].
#[inline]
pub fn compute_schroeder_requirements(
    quad: &TransportQuadWire,
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
                .clamp(1, TRANSPORT_SCHROEDER_MAX_DELAY);
            delays[k] = scaled;
            if scaled > max_delay {
                max_delay = scaled;
            }
        }
        let g_raw = fp16_to_f32(quad.allpass_g_fp16[k].get());
        allpass_g[k] = clamp_finite_f32(
            g_raw,
            -TRANSPORT_ALLPASS_G_MAX,
            TRANSPORT_ALLPASS_G_MAX,
            0.0,
        );
    }

    let cap_per_lane = (max_delay + 1).max(1).next_power_of_two();
    let total_elements = cap_per_lane * 4;
    (delays, allpass_g, cap_per_lane, total_elements)
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
        quad: &TransportQuadWire,
        rho_vm: f64,
    ) -> Result<Self, VmError> {
        if (quad.flags & TRANSPORT_FLAG_SCHROEDER_ENABLE) == 0 {
            return Ok(Self {
                buf,
                cap_per_lane: 0,
                mask: 0,
                write_pos: 0,
                delays: [0; 4],
                allpass_g: [0.0; 4],
            });
        }
        let (delays, allpass_g, cap_per_lane, req_len) =
            compute_schroeder_requirements(quad, rho_vm);
        if req_len == 0 || cap_per_lane == 0 {
            return Ok(Self {
                buf,
                cap_per_lane: 0,
                mask: 0,
                write_pos: 0,
                delays: [0; 4],
                allpass_g: [0.0; 4],
            });
        }
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
        quad: &TransportQuadWire,
        rho_vm: f64,
    ) -> Result<Self, VmError> {
        Self::from_cold_buf(ColdBuf::borrowed(buf), quad, rho_vm)
    }

    /// Allocates owned Schroeder storage sized for `(quad, rho_vm)` (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_for_quad(quad: &TransportQuadWire, rho_vm: f64) -> Result<Self, VmError> {
        Self::new_owned_with_capacity(0, quad, rho_vm)
    }

    /// Allocates owned Schroeder storage with at least `min_f32_len` headroom (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_with_capacity(
        min_f32_len: usize,
        quad: &TransportQuadWire,
        rho_vm: f64,
    ) -> Result<Self, VmError> {
        let req_len = if (quad.flags & TRANSPORT_FLAG_SCHROEDER_ENABLE) == 0 {
            0
        } else {
            compute_schroeder_requirements(quad, rho_vm).3
        };
        let cap = req_len.max(min_f32_len);
        if cap == 0 {
            Ok(Self::disabled())
        } else {
            Self::from_cold_buf(ColdBuf::Owned(vec![0.0_f32; cap]), quad, rho_vm)
        }
    }

    /// Deactivates all-pass diffusion while preserving backing capacity for future reload (`0` drops).
    #[inline]
    pub fn deactivate_preserving_storage(&mut self) {
        self.cap_per_lane = 0;
        self.mask = 0;
        self.write_pos = 0;
        self.delays = [0; 4];
        self.allpass_g = [0.0; 4];
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

    /// Returns the cold buffer base address (`0` if empty) for reload pointer stability checks.
    #[inline]
    pub fn backing_storage_addr(&self) -> usize {
        if self.buf.is_empty() {
            0
        } else {
            self.buf.as_slice().as_ptr() as usize
        }
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

    /// Updates active Schroeder all-pass coefficients (`[-0.75, 0.75]`), e.g. from `slot 25` and
    /// `(1 - mu_freeze)` taper.
    #[inline]
    pub fn set_allpass_g(&mut self, gains: [f32; 4]) {
        for (dst, &g) in self.allpass_g.iter_mut().zip(gains.iter()) {
            *dst = clamp_finite_f32(g, -TRANSPORT_ALLPASS_G_MAX, TRANSPORT_ALLPASS_G_MAX, 0.0);
        }
    }

    /// Recomputes ZOH-scaled Schroeder delays/gains and validates capacity for `rho_vm`.
    pub fn reconfigure_sample_rate(
        &mut self,
        quad: &TransportQuadWire,
        rho_vm: f64,
    ) -> Result<(), VmError> {
        if (quad.flags & TRANSPORT_FLAG_SCHROEDER_ENABLE) == 0 {
            self.cap_per_lane = 0;
            self.mask = 0;
            self.write_pos = 0;
            self.delays = [0; 4];
            self.allpass_g = [0.0; 4];
            return Ok(());
        }
        let (delays, allpass_g, cap_per_lane, req_len) =
            compute_schroeder_requirements(quad, rho_vm);
        if req_len == 0 || cap_per_lane == 0 {
            self.deactivate_preserving_storage();
            return Ok(());
        }
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
    use crate::math::{DefaultMathBackend, ScalarBackend, const_abs_f64, f32_to_fp16};
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

    /// verify[dsp.op-t.diffuser.partition]
    /// verify[dsp.op-t.diffuser.recurrence]
    #[test]
    fn test_schroeder_bypass_energy_preservation_and_fail_closed() {
        let q_mix = base_wire([53.0; 4], [0, 17, 0, 31], TRANSPORT_FLAG_SCHROEDER_ENABLE);
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
    fn test_owned_schroeder_allocation() {
        let q = base_wire([200.0; 4], [19, 31, 0, 47], TRANSPORT_FLAG_SCHROEDER_ENABLE);
        let sch = PartitionedSchroederBank::new_owned_for_quad(&q, 1.0).unwrap();
        assert_eq!(sch.cap_per_lane(), 64);
        assert_eq!(sch.backing_f32_capacity(), 256);
    }
}
