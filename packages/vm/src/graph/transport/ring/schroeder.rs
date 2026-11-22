//! Partitioned 4-lane SoA in-buffer Schroeder all-pass diffuser bank
//! ([`PartitionedSchroederBank`]) with fused `SO(4)` paraunitary scattering and rate-scaled
//! capacity sizing (`ref[dsp.op-t.diffuser.partition]`, `ref[dsp.op-t.diffuser.recurrence]`).

#[cfg(feature = "alloc")]
use alloc::vec;

use dnam_kernels::{
    apply_so4_even_chunk4, apply_so4_even_vec4, apply_so4_odd_chunk4, apply_so4_odd_vec4,
    step_schroeder_quad_vec4,
};

use crate::bin::arena_wire::TransportQuadWire;
use crate::math::{
    MathBackend, SimdVec4, clamp_finite_f32, fp16_to_f32, rcp_nr6_f64, round_half_away_f64,
};
use crate::types::{ColdBuf, VmError};

/// Maximum reference `48 kHz` Schroeder delay length (`255`) scaled up to `384 kHz` (`16,384`).
pub const TRANSPORT_SCHROEDER_MAX_DELAY: usize = 16_384;

/// Maximum absolute Schroeder all-pass coefficient magnitude (`0.75`).
pub const TRANSPORT_ALLPASS_G_MAX: f32 = 0.75;

/// Computes rate-scaled Schroeder delays, clamped gains, per-lane power-of-two capacity, and
/// total `f32` element count (`4 * cap_per_lane`, `0` when `allpass_delays == [0; 4]`) for
/// [`PartitionedSchroederBank`].
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

    let cap_per_lane = if max_delay == 0 {
        0
    } else {
        (max_delay + 1).next_power_of_two()
    };
    let total_elements = cap_per_lane * 4;
    (delays, allpass_g, cap_per_lane, total_elements)
}

/// Partitioned 4-lane lane-major SoA (`buf[lane * cap_per_lane + idx]`) in-buffer Schroeder
/// all-pass diffuser bank with fused `SO(4)` Givens butterfly scattering
/// (`ref[dsp.op-t.diffuser.partition]`, `ref[dsp.op-t.diffuser.recurrence]`). Lanes with
/// `delays[k] == 0` act as identity inside `SO(4)` (`y1_k = x1_k`) and allocate `0 B` when all
/// four delays are zero.
#[derive(Debug)]
pub struct PartitionedSchroederBank<'a> {
    buf: ColdBuf<'a, f32>,
    cap_per_lane: usize,
    mask: usize,
    write_pos: usize,
    delays: [usize; 4],
    allpass_g: [f32; 4],
    can_chunk4: bool,
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
            can_chunk4: false,
        }
    }

    /// Hydrates a [`PartitionedSchroederBank<'a>`] from [`ColdBuf<'a, f32>`], failing closed with
    /// [`VmError::InsufficientBackingCapacity`] if `buf.len() < 4 * cap_per_lane`.
    pub fn from_cold_buf(
        mut buf: ColdBuf<'a, f32>,
        quad: &TransportQuadWire,
        rho_vm: f64,
    ) -> Result<Self, VmError> {
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
                can_chunk4: false,
            });
        }
        if buf.len() < req_len {
            return Err(VmError::InsufficientBackingCapacity);
        }
        buf.as_mut_slice()[..req_len].fill(0.0);
        let can_chunk4 = cap_per_lane >= 8 && delays.iter().all(|&d| d == 0 || d >= 4);
        Ok(Self {
            buf,
            cap_per_lane,
            mask: cap_per_lane - 1,
            write_pos: 0,
            delays,
            allpass_g,
            can_chunk4,
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
        let req_len = compute_schroeder_requirements(quad, rho_vm).3;
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
        self.can_chunk4 = false;
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

    /// Returns converted integer delays per lane (`0` means identity bypass inside `SO(4)`).
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
        self.can_chunk4 = cap_per_lane >= 8 && delays.iter().all(|&d| d == 0 || d >= 4);
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

    /// Steps 4 parallel Schroeder diffuser lanes fused inside `SO(4)` paraunitary scattering
    /// (`R_odd(theta) * diag(A_k) * R_even(theta)`), preserving clean identity (`y1_k = x1_k`)
    /// inside `SO(4)` for any lane with `delays[k] == 0`.
    #[inline(always)]
    pub fn step_vec4<B: MathBackend>(
        &mut self,
        x: B::Vec4,
        cos_scat: f32,
        sin_scat: f32,
    ) -> B::Vec4 {
        let x_clean = x.scrub_recursive_state();
        let x1 = apply_so4_even_vec4(x_clean, cos_scat, sin_scat);
        if self.cap_per_lane == 0 || self.buf.len() < self.cap_per_lane * 4 || self.delays == [0; 4]
        {
            return apply_so4_odd_vec4(x1, cos_scat, sin_scat);
        }
        let cap = self.cap_per_lane;
        let mask = self.mask;
        let wr = self.write_pos;
        let mut delayed = [0.0_f32; 4];
        {
            let s = self.buf.as_slice();
            for k in 0..4 {
                let dk = self.delays[k];
                if dk > 0 {
                    let r_idx = wr.wrapping_sub(dk) & mask;
                    delayed[k] = s[k * cap + r_idx];
                }
            }
        }
        let g_vec = B::Vec4::load(&self.allpass_g);
        let (y_vec, write_vec) = step_schroeder_quad_vec4(x1, B::Vec4::load(&delayed), g_vec);
        let write_arr = write_vec.to_array();
        {
            let dst = self.buf.as_mut_slice();
            for k in 0..4 {
                if self.delays[k] > 0 {
                    dst[k * cap + wr] = write_arr[k];
                }
            }
        }
        self.write_pos = (wr + 1) & mask;

        let y1 = if self.delays[0] == 0
            || self.delays[1] == 0
            || self.delays[2] == 0
            || self.delays[3] == 0
        {
            let x1_arr = x1.to_array();
            let mut y_arr = y_vec.to_array();
            for k in 0..4 {
                if self.delays[k] == 0 {
                    y_arr[k] = x1_arr[k];
                }
            }
            B::Vec4::load(&y_arr)
        } else {
            y_vec
        };
        apply_so4_odd_vec4(y1, cos_scat, sin_scat)
    }

    /// Steps 4 consecutive frames `[f = 0..4]` of 4-lane fused `SO(4)` + Schroeder diffusion
    /// using lane-major SIMD `4x4` transposition when all active delays satisfy `d_k >= 4`, with
    /// exact `0-ULP` parity against four sequential [`Self::step_vec4`] calls.
    #[inline(always)]
    pub fn step_chunk4<B: MathBackend>(
        &mut self,
        x: [B::Vec4; 4],
        cos_scat: f32,
        sin_scat: f32,
    ) -> [B::Vec4; 4] {
        let x_clean = x.map(SimdVec4::scrub_recursive_state);
        if self.cap_per_lane == 0 || self.buf.len() < self.cap_per_lane * 4 || self.delays == [0; 4]
        {
            let x1 = apply_so4_even_chunk4(x_clean, cos_scat, sin_scat);
            return apply_so4_odd_chunk4(x1, cos_scat, sin_scat);
        }
        if !self.can_chunk4 {
            return [
                self.step_vec4::<B>(x_clean[0], cos_scat, sin_scat),
                self.step_vec4::<B>(x_clean[1], cos_scat, sin_scat),
                self.step_vec4::<B>(x_clean[2], cos_scat, sin_scat),
                self.step_vec4::<B>(x_clean[3], cos_scat, sin_scat),
            ];
        }

        let x1 = apply_so4_even_chunk4(x_clean, cos_scat, sin_scat);
        let x1_lanes = B::transpose_4x4(x1[0], x1[1], x1[2], x1[3]);
        let cap = self.cap_per_lane;
        let mask = self.mask;
        let wr = self.write_pos;
        let mut y_lanes = [B::Vec4::zero(); 4];
        let mut w_arrs = [[0.0_f32; 4]; 4];

        {
            let s = self.buf.as_slice();
            for k in 0..4 {
                let dk = self.delays[k];
                if dk == 0 {
                    y_lanes[k] = x1_lanes[k];
                    continue;
                }
                let lane_off = k * cap;
                let r0 = wr.wrapping_sub(dk) & mask;
                let d_arr = if r0 + 4 <= cap {
                    let src = &s[lane_off + r0..lane_off + r0 + 4];
                    [src[0], src[1], src[2], src[3]]
                } else {
                    [
                        s[lane_off + r0],
                        s[lane_off + ((r0 + 1) & mask)],
                        s[lane_off + ((r0 + 2) & mask)],
                        s[lane_off + ((r0 + 3) & mask)],
                    ]
                };
                let (y_lane, w_lane) = step_schroeder_quad_vec4(
                    x1_lanes[k],
                    B::Vec4::load(&d_arr),
                    B::Vec4::splat(self.allpass_g[k]),
                );
                y_lanes[k] = y_lane;
                w_arrs[k] = w_lane.to_array();
            }
        }

        {
            let dst = self.buf.as_mut_slice();
            for k in 0..4 {
                if self.delays[k] == 0 {
                    continue;
                }
                let lane_off = k * cap;
                let w_arr = &w_arrs[k];
                if wr + 4 <= cap {
                    dst[lane_off + wr..lane_off + wr + 4].copy_from_slice(w_arr);
                } else {
                    dst[lane_off + wr] = w_arr[0];
                    dst[lane_off + ((wr + 1) & mask)] = w_arr[1];
                    dst[lane_off + ((wr + 2) & mask)] = w_arr[2];
                    dst[lane_off + ((wr + 3) & mask)] = w_arr[3];
                }
            }
        }
        self.write_pos = (wr + 4) & mask;

        let y1 = B::transpose_4x4(y_lanes[0], y_lanes[1], y_lanes[2], y_lanes[3]);
        apply_so4_odd_chunk4(y1, cos_scat, sin_scat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend, const_abs_f64, f32_to_fp16};
    use dnam_kernels::algebraic_unit_rotor_f32;
    use proptest::prelude::*;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn base_wire(delays: [f32; 4], ap_delays: [u8; 4]) -> TransportQuadWire {
        let mut wire = TransportQuadWire::new_zeroed();
        wire.delay_samples = delays.map(F32::new);
        wire.allpass_delays = ap_delays;
        wire.allpass_g_fp16 = [0.6, -0.5, 0.7, -0.65].map(|g| U16::new(f32_to_fp16(g)));
        wire
    }

    /// verify[dsp.op-t.diffuser.partition]
    /// verify[dsp.op-t.diffuser.recurrence]
    #[test]
    fn test_schroeder_bypass_energy_preservation_and_fail_closed() {
        let q_zero = base_wire([53.0; 4], [0; 4]);
        let (delays0, _, cap0, req0) = compute_schroeder_requirements(&q_zero, 0.5);
        assert_eq!((delays0, cap0, req0), ([0; 4], 0, 0));
        let mut empty_ap = PartitionedSchroederBank::from_slice(&mut [], &q_zero, 0.5).unwrap();
        assert_eq!(empty_ap.cap_per_lane(), 0);
        let (c_s, s_s) = algebraic_unit_rotor_f32(0.55);
        let x0 = [1.0, -0.5, 0.25, -0.75];
        let y0 = empty_ap
            .step_vec4::<DefaultMathBackend>(
                <DefaultMathBackend as MathBackend>::Vec4::load(&x0),
                c_s,
                s_s,
            )
            .to_array();
        let e_in0: f64 = x0.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let e_out0: f64 = y0.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(const_abs_f64(e_out0 - e_in0) < 1.0e-5);

        let q_mix = base_wire([53.0; 4], [0, 17, 0, 31]);
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
                1.0,
                0.0,
            );
            let ya = <DefaultMathBackend as MathBackend>::Vec4::to_array(y);
            // At identity SO(4) (1, 0), zero-delay lanes 0 and 2 bypass verbatim.
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

        // Non-zero SO(4) scattering angle preserves total 4-lane paraunitary energy.
        ap.reset();
        let (mut tot_in, mut tot_out) = (0.0_f64, 0.0_f64);
        for n in 0..512 {
            let x = if n == 0 {
                [1.0, 0.5, -0.75, 0.25]
            } else {
                [0.0; 4]
            };
            let ya = ap
                .step_vec4::<DefaultMathBackend>(
                    <DefaultMathBackend as MathBackend>::Vec4::load(&x),
                    c_s,
                    s_s,
                )
                .to_array();
            for k in 0..4 {
                tot_in += f64::from(x[k]) * f64::from(x[k]);
                tot_out += f64::from(ya[k]) * f64::from(ya[k]);
            }
        }
        assert!(const_abs_f64(tot_out - tot_in) < 1.0e-4);

        ap.reset();
        let y_zero = ap.step_vec4::<ScalarBackend>(
            <ScalarBackend as MathBackend>::Vec4::splat(0.0),
            c_s,
            s_s,
        );
        assert_eq!(
            <ScalarBackend as MathBackend>::Vec4::to_array(y_zero),
            [0.0; 4]
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_owned_schroeder_allocation() {
        let q_zero = base_wire([200.0; 4], [0; 4]);
        let sch_zero = PartitionedSchroederBank::new_owned_for_quad(&q_zero, 1.0).unwrap();
        assert_eq!(
            (sch_zero.cap_per_lane(), sch_zero.backing_f32_capacity()),
            (0, 0)
        );

        let q = base_wire([200.0; 4], [19, 31, 0, 47]);
        let sch = PartitionedSchroederBank::new_owned_for_quad(&q, 1.0).unwrap();
        assert_eq!(sch.cap_per_lane(), 64);
        assert_eq!(sch.backing_f32_capacity(), 256);
    }

    fn verify_chunk4_parity_for_wire(
        ap_delays: [u8; 4],
        gains: [f32; 4],
        cos_s: f32,
        sin_s: f32,
        peels: usize,
        frames: &[[f32; 4]; 12],
    ) {
        let mut wire = TransportQuadWire::new_zeroed();
        wire.allpass_delays = ap_delays;
        wire.allpass_g_fp16 = gains.map(|g| U16::new(f32_to_fp16(g)));
        let (_, _, _, req_len) = compute_schroeder_requirements(&wire, 1.0);
        let buf_len = req_len.max(4);
        let mut buf_s = [0.0_f32; 256];
        let mut buf_c = [0.0_f32; 256];
        let mut ap_s =
            PartitionedSchroederBank::from_slice(&mut buf_s[..buf_len], &wire, 1.0).unwrap();
        let mut ap_c =
            PartitionedSchroederBank::from_slice(&mut buf_c[..buf_len], &wire, 1.0).unwrap();

        for x in frames.iter().take(peels) {
            let ys = ap_s
                .step_vec4::<DefaultMathBackend>(
                    <DefaultMathBackend as MathBackend>::Vec4::load(x),
                    cos_s,
                    sin_s,
                )
                .to_array();
            let yc = ap_c
                .step_vec4::<DefaultMathBackend>(
                    <DefaultMathBackend as MathBackend>::Vec4::load(x),
                    cos_s,
                    sin_s,
                )
                .to_array();
            for k in 0..4 {
                assert_eq!(ys[k].to_bits(), yc[k].to_bits());
            }
        }
        let rem = &frames[peels..];
        for chunk in rem.as_chunks::<4>().0 {
            let mut seq_out = [[0.0_f32; 4]; 4];
            for (f, x) in chunk.iter().enumerate() {
                seq_out[f] = ap_s
                    .step_vec4::<DefaultMathBackend>(
                        <DefaultMathBackend as MathBackend>::Vec4::load(x),
                        cos_s,
                        sin_s,
                    )
                    .to_array();
            }
            let x4 = [
                <DefaultMathBackend as MathBackend>::Vec4::load(&chunk[0]),
                <DefaultMathBackend as MathBackend>::Vec4::load(&chunk[1]),
                <DefaultMathBackend as MathBackend>::Vec4::load(&chunk[2]),
                <DefaultMathBackend as MathBackend>::Vec4::load(&chunk[3]),
            ];
            let chunk_out = ap_c.step_chunk4::<DefaultMathBackend>(x4, cos_s, sin_s);
            for f in 0..4 {
                let ca = chunk_out[f].to_array();
                for k in 0..4 {
                    assert_eq!(ca[k].to_bits(), seq_out[f][k].to_bits());
                }
            }
        }
    }

    #[test]
    fn test_schroeder_chunk4_edge_delay_cap_minus_one_parity() {
        let (c_s, s_s) = algebraic_unit_rotor_f32(0.7);
        let frames: [[f32; 4]; 12] = core::array::from_fn(|i| {
            let base = (i as f32) * 0.13 - 0.5;
            [base, -0.5 * base, 0.3 * base + 0.1, -0.25 * base]
        });
        for peels in 0..4 {
            verify_chunk4_parity_for_wire(
                [0, 7, 4, 7],
                [0.6, -0.5, 0.7, -0.65],
                c_s,
                s_s,
                peels,
                &frames,
            );
            verify_chunk4_parity_for_wire(
                [1, 2, 3, 0],
                [0.4, -0.3, 0.5, 0.0],
                c_s,
                s_s,
                peels,
                &frames,
            );
            verify_chunk4_parity_for_wire(
                [0, 0, 0, 0],
                [0.4, -0.3, 0.5, 0.2],
                c_s,
                s_s,
                peels,
                &frames,
            );
        }
    }

    proptest! {
        #[test]
        fn prop_schroeder_chunk4_matches_seq_0ulp(
            d0 in prop_oneof![Just(0u8), 1u8..=3u8, 4u8..=31u8],
            d1 in prop_oneof![Just(0u8), 1u8..=3u8, 4u8..=31u8],
            d2 in prop_oneof![Just(0u8), 1u8..=3u8, 4u8..=31u8],
            d3 in prop_oneof![Just(0u8), 1u8..=3u8, 4u8..=31u8],
            g in prop::array::uniform4(-0.75f32..0.75f32),
            psi in prop_oneof![Just(0.0f32), -core::f32::consts::PI..core::f32::consts::PI],
            peels in 0usize..4,
            s in prop::array::uniform12(prop::array::uniform4(-2.0f32..2.0f32)),
        ) {
            let (cos_s, sin_s) = algebraic_unit_rotor_f32(psi);
            verify_chunk4_parity_for_wire([d0, d1, d2, d3], g, cos_s, sin_s, peels, &s);
        }
    }
}
