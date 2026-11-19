//! Op-T Catmull-Rom Delay Line, Dual-Mode Transitions & Partitioned Schroeder Diffuser.
//!
//! Canonical Reference: `docs/specs/dsp/op_t_transport.md` (`ref[dsp.op-t.reader.catmull-rom]`,
//! `ref[dsp.op-t.transition.dual-mode]`, `ref[dsp.op-t.diffuser.recurrence]`,
//! `ref[dsp.op-t.ring.fp16-saturation]`).

pub mod window_and_vca;

pub use window_and_vca::{
    BallisticDuckingVca, OP_T_FULL_RING_CAPACITY, OP_T_LITE_RING_CAPACITY,
    OP_T_ULTRA_RING_CAPACITY, OpTBackend, SCHROEDER_COPRIME_DELAYS, catmull_rom_hermite_cubic,
    cubic_lagrange_weights_4ch_vec4, hermite_hann_partition_of_unity,
    raised_cosine_barberpole_3phase,
};

use crate::math::{
    FP16_MAX_FINITE_F32, f32_to_fp16_with_telemetry, fp16::fp16_to_f32, newton6_reciprocal_f64,
};

const FP16_MAX_FINITE_ABS_BITS: u32 = FP16_MAX_FINITE_F32.to_bits();
const _: () = assert!(FP16_MAX_FINITE_ABS_BITS == 0x477F_E000);

/// Single or partitioned-stereo delay line with power-of-2 `fp16` ring buffer, 4-tap Catmull-Rom
/// reader, partitioned Schroeder diffuser, `fp16` saturation telemetry, and dual-mode
/// slew/crossfade transitions with single pending-target coalescing
/// (`ref[dsp.op-t.transition.dual-mode]`, `ref[dsp.op-t.reader.catmull-rom]`).
#[derive(Clone, Debug)]
pub struct OpTDelayLine<const CAPACITY: usize> {
    pub ring: [u16; CAPACITY],
    pub write_ptr: usize,
    pub current_delay: f32,
    pub target_delay: f32,
    pub xfade_active: bool,
    pub xfade_idx: usize,
    pub xfade_len: usize,
    pub cached_xfade_len: usize,
    pub inv_xfade_denom: f32,
    pub old_delay: f32,
    pub pending_target: Option<(f32, bool)>,
    pub saturation_flags: u32,
    pub diff_state: [f32; 256],
    pub diff_idx: usize,
}

impl<const CAPACITY: usize> Default for OpTDelayLine<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> OpTDelayLine<CAPACITY> {
    pub const MASK: usize = CAPACITY - 1;
    pub const HALF_CAPACITY: usize = CAPACITY / 2;
    pub const HALF_MASK: usize = (CAPACITY / 2) - 1;
    pub const DEFAULT_INV_XFADE_DENOM: f32 = (1.0_f64 / 127.0_f64) as f32;

    pub const fn new() -> Self {
        Self {
            ring: [0; CAPACITY],
            write_ptr: 0,
            current_delay: 64.0,
            target_delay: 64.0,
            xfade_active: false,
            xfade_idx: 0,
            xfade_len: 128,
            cached_xfade_len: 128,
            inv_xfade_denom: Self::DEFAULT_INV_XFADE_DENOM,
            old_delay: 64.0,
            pending_target: None,
            saturation_flags: 0,
            diff_state: [0.0; 256],
            diff_idx: 0,
        }
    }

    /// Updates `xfade_len` (clamped to `>= 16`) and precomputes its reciprocal `1 / (len - 1)` (`0 FDIV`).
    #[inline]
    pub fn set_xfade_len(&mut self, len: usize) {
        let clamped = len.max(16);
        self.xfade_len = clamped;
        self.cached_xfade_len = clamped;
        self.inv_xfade_denom = newton6_reciprocal_f64((clamped - 1) as f64) as f32;
    }

    #[inline(always)]
    fn xfade_inv_denom(&mut self, len: usize) -> f32 {
        if len == self.cached_xfade_len {
            self.inv_xfade_denom
        } else {
            let inv = newton6_reciprocal_f64((len - 1) as f64) as f32;
            self.cached_xfade_len = len;
            self.inv_xfade_denom = inv;
            inv
        }
    }

    pub fn reset(&mut self) {
        self.ring.fill(0);
        self.write_ptr = 0;
        self.current_delay = self.target_delay;
        self.xfade_active = false;
        self.xfade_idx = 0;
        self.old_delay = self.target_delay;
        self.pending_target = None;
        self.saturation_flags = 0;
        self.diff_state.fill(0.0);
        self.diff_idx = 0;
    }

    #[inline(always)]
    fn apply_clamped_target(&mut self, clamped: f32, is_switch_jump: bool) {
        if self.xfade_active {
            self.pending_target = Some((clamped, is_switch_jump));
        } else if is_switch_jump && (clamped - self.current_delay).abs() > 16.0 {
            self.old_delay = self.current_delay;
            self.target_delay = clamped;
            self.xfade_active = true;
            self.xfade_idx = 0;
            self.pending_target = None;
        } else {
            self.target_delay = clamped;
            self.pending_target = None;
        }
    }

    /// Sets mono delay target in `[4.0, CAPACITY - 4]` with dual-mode transition and active-fade
    /// pending-target coalescing (`ref[dsp.op-t.transition.dual-mode]`).
    pub fn set_target_delay(&mut self, target: f32, is_switch_jump: bool) {
        let max_delay = CAPACITY.saturating_sub(4).max(4) as f32;
        self.apply_clamped_target(target.clamp(4.0, max_delay), is_switch_jump);
    }

    /// Sets partitioned-stereo delay target in `[4.0, CAPACITY / 2 - 4]` with dual-mode transition
    /// and active-fade pending-target coalescing (`ref[dsp.op-t.transition.dual-mode]`).
    pub fn set_target_delay_stereo(&mut self, target: f32, is_switch_jump: bool) {
        let max_delay = Self::HALF_CAPACITY.saturating_sub(4).max(4) as f32;
        self.apply_clamped_target(target.clamp(4.0, max_delay), is_switch_jump);
    }

    #[inline(always)]
    fn advance_delay_transition(&mut self, max_delay: f32) {
        if !self.xfade_active
            && let Some((pending, jump)) = self.pending_target.take()
        {
            self.apply_clamped_target(pending.clamp(4.0, max_delay), jump);
        }
        if !self.xfade_active {
            let diff = self.target_delay - self.current_delay;
            if diff.abs() > 0.0001 {
                self.current_delay += diff.clamp(-0.25, 0.25);
            } else {
                self.current_delay = self.target_delay;
            }
        }
    }

    /// Reads fractional mono delay in `[4.0, CAPACITY - 2]` via 4-tap Catmull-Rom interpolation
    /// (`ref[dsp.op-t.reader.catmull-rom]`). Direct callers must hold a [`crate::FtzDazGuard`].
    #[inline(always)]
    pub fn read_delay(&self, delay: f32) -> f32 {
        let max_delay = CAPACITY.saturating_sub(2).max(4) as f32;
        let d_clamped = delay.clamp(4.0, max_delay);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);

        let p_0 = (self.write_ptr + CAPACITY - d_int) & Self::MASK;
        let s_0 = fp16_to_f32(self.ring[p_0]);
        if mu == 0.0 {
            return s_0;
        }
        let p_m1 = (p_0 + 1) & Self::MASK;
        let p_1 = (p_0 + CAPACITY - 1) & Self::MASK;
        let p_2 = (p_0 + CAPACITY - 2) & Self::MASK;

        let s_m1 = fp16_to_f32(self.ring[p_m1]);
        let s_1 = fp16_to_f32(self.ring[p_1]);
        let s_2 = fp16_to_f32(self.ring[p_2]);

        catmull_rom_hermite_cubic(s_m1, s_0, s_1, s_2, mu)
    }

    #[inline(always)]
    fn xfade_alpha(&mut self, len: usize) -> f32 {
        let inv_denom = self.xfade_inv_denom(len);
        let w = (self.xfade_idx as f32) * inv_denom;
        let (c, _) = dnam_kernels::algebraic_unit_rotor_f32(w * core::f32::consts::PI);
        0.5 * (1.0 - c)
    }

    /// Advances the mono delay line by 1 audio sample: writes `input` with `fp16` saturation
    /// telemetry (`FP16_SATURATED`) and returns delayed output. Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, input: f32) -> f32 {
        let max_target = CAPACITY.saturating_sub(4).max(4) as f32;
        self.advance_delay_transition(max_target);

        let out = if self.xfade_active {
            let len = self.xfade_len.max(16);
            let d_old = self.read_delay(self.old_delay);
            let d_new = self.read_delay(self.target_delay);
            let alpha_xf = self.xfade_alpha(len);
            let mixed = (1.0 - alpha_xf) * d_old + alpha_xf * d_new;
            self.xfade_idx += 1;
            if self.xfade_idx >= len {
                self.xfade_active = false;
                self.current_delay = self.target_delay;
            }
            mixed
        } else {
            self.read_delay(self.current_delay)
        };

        self.ring[self.write_ptr] = f32_to_fp16_with_telemetry(input, &mut self.saturation_flags);
        self.write_ptr = (self.write_ptr + 1) & Self::MASK;
        out
    }

    /// Reads fractional delay in `[4.0, HALF_CAPACITY - 2]` for a partitioned stereo pair
    /// (`ring[0..CAPACITY/2]` Left, `ring[CAPACITY/2..CAPACITY]` Right) via 4-tap Catmull-Rom
    /// interpolation (`ref[dsp.op-t.reader.catmull-rom]`). Direct callers must hold a
    /// [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn read_delay_stereo(&self, delay: f32) -> (f32, f32) {
        let max_delay = Self::HALF_CAPACITY.saturating_sub(2).max(4) as f32;
        let d_clamped = delay.clamp(4.0, max_delay);
        let d_int = d_clamped as usize;
        let mu = d_clamped - (d_int as f32);
        let w_ptr = self.write_ptr & Self::HALF_MASK;

        let p_0 = (w_ptr + Self::HALF_CAPACITY - d_int) & Self::HALF_MASK;
        let r_off = Self::HALF_CAPACITY;
        if mu == 0.0 {
            return (
                fp16_to_f32(self.ring[p_0]),
                fp16_to_f32(self.ring[r_off + p_0]),
            );
        }

        let (l_2, l_1, l_0, l_m1, r_2, r_1, r_0, r_m1) = if p_0 >= 2
            && p_0 + 1 < Self::HALF_CAPACITY
        {
            let base_l = p_0 - 2;
            let base_r = r_off + base_l;
            let [l2, l1, l0, lm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(
                    self.ring[base_l..base_l + 4].try_into().unwrap(),
                );
            let [r2, r1, r0, rm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(
                    self.ring[base_r..base_r + 4].try_into().unwrap(),
                );
            (l2, l1, l0, lm1, r2, r1, r0, rm1)
        } else {
            let p_m1 = (p_0 + 1) & Self::HALF_MASK;
            let p_1 = (p_0 + Self::HALF_CAPACITY - 1) & Self::HALF_MASK;
            let p_2 = (p_0 + Self::HALF_CAPACITY - 2) & Self::HALF_MASK;
            let [l2, l1, l0, lm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(&[
                    self.ring[p_2],
                    self.ring[p_1],
                    self.ring[p_0],
                    self.ring[p_m1],
                ]);
            let [r2, r1, r0, rm1] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::fp16x4_to_f32x4(&[
                    self.ring[r_off + p_2],
                    self.ring[r_off + p_1],
                    self.ring[r_off + p_0],
                    self.ring[r_off + p_m1],
                ]);
            (l2, l1, l0, lm1, r2, r1, r0, rm1)
        };

        (
            catmull_rom_hermite_cubic(l_m1, l_0, l_1, l_2, mu),
            catmull_rom_hermite_cubic(r_m1, r_0, r_1, r_2, mu),
        )
    }

    /// Advances the partitioned stereo delay line by 1 stereo audio frame (`in_l`, `in_r`) with
    /// zero L/R crosstalk, `fp16` saturation telemetry, and single-frame slew/crossfade progression
    /// (`ref[dsp.op-t.transition.dual-mode]`, `ref[dsp.op-t.ring.fp16-saturation]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let max_target = Self::HALF_CAPACITY.saturating_sub(4).max(4) as f32;
        self.advance_delay_transition(max_target);

        let (out_l, out_r) = if self.xfade_active {
            let len = self.xfade_len.max(16);
            let (l_old, r_old) = self.read_delay_stereo(self.old_delay);
            let (l_new, r_new) = self.read_delay_stereo(self.target_delay);
            let alpha_xf = self.xfade_alpha(len);
            let one_minus_a = 1.0 - alpha_xf;
            let mix_l = one_minus_a * l_old + alpha_xf * l_new;
            let mix_r = one_minus_a * r_old + alpha_xf * r_new;
            self.xfade_idx += 1;
            if self.xfade_idx >= len {
                self.xfade_active = false;
                self.current_delay = self.target_delay;
            }
            (mix_l, mix_r)
        } else {
            self.read_delay_stereo(self.current_delay)
        };

        let w_ptr = self.write_ptr & Self::HALF_MASK;
        let abs_l = in_l.to_bits() & 0x7FFF_FFFF;
        let abs_r = in_r.to_bits() & 0x7FFF_FFFF;
        let (h_l, h_r) = if abs_l > FP16_MAX_FINITE_ABS_BITS || abs_r > FP16_MAX_FINITE_ABS_BITS {
            (
                f32_to_fp16_with_telemetry(in_l, &mut self.saturation_flags),
                f32_to_fp16_with_telemetry(in_r, &mut self.saturation_flags),
            )
        } else {
            let [hl, hr, _, _] =
                <crate::math::DefaultMathBackend as crate::math::MathBackend>::f32x4_to_fp16x4(&[
                    in_l, in_r, 0.0, 0.0,
                ]);
            (hl, hr)
        };
        self.ring[w_ptr] = h_l;
        self.ring[Self::HALF_CAPACITY + w_ptr] = h_r;
        self.write_ptr = (w_ptr + 1) & Self::HALF_MASK;
        (out_l, out_r)
    }

    /// Steps the partitioned in-buffer Schroeder all-pass diffuser via
    /// [`dnam_kernels::step_schroeder_allpass`] (`ref[dsp.op-t.diffuser.recurrence]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_diffuser(&mut self, input: f32, delay_samples: usize, a_diff: f32) -> f32 {
        let d = delay_samples.clamp(1, 255);
        let y = dnam_kernels::step_schroeder_allpass(
            input,
            &mut self.diff_state,
            0xFF,
            self.diff_idx,
            d,
            a_diff,
        );
        self.diff_idx = (self.diff_idx + 1) & 0xFF;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::math::saturation_flags;
    use proptest::prelude::*;

    #[test]
    fn test_op_t_delay_line_coalescing_bounds_and_saturation() {
        let _audio = FtzDazGuard::new();
        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        dl.reset();

        for i in 0..100 {
            let _ = dl.step(i as f32);
        }

        // Mono target allows CAPACITY - 4; stereo target clamps to HALF_CAPACITY - 4
        dl.set_target_delay(3000.0, false);
        assert_eq!(dl.target_delay, (OP_T_LITE_RING_CAPACITY - 4) as f32);
        dl.set_target_delay_stereo(3000.0, false);
        assert_eq!(dl.target_delay, (OP_T_LITE_RING_CAPACITY / 2 - 4) as f32);

        // read_delay / read_delay_stereo honor [4.0, len - 2] without touching write_ptr
        assert!(
            dl.read_delay((OP_T_LITE_RING_CAPACITY - 1) as f32)
                .is_finite()
        );
        let (sl, sr) = dl.read_delay_stereo((OP_T_LITE_RING_CAPACITY / 2 - 1) as f32);
        assert!(sl.is_finite() && sr.is_finite());

        // Pending target coalescing during active crossfade
        dl.current_delay = 64.0;
        dl.target_delay = 64.0;
        dl.set_xfade_len(32);
        dl.set_target_delay(150.0, true);
        assert!(dl.xfade_active);
        dl.set_target_delay(200.0, true);
        dl.set_target_delay(250.0, true);
        assert_eq!(dl.target_delay, 150.0);
        assert_eq!(dl.pending_target, Some((250.0, true)));

        for _ in 0..32 {
            let _ = dl.step(0.5);
        }
        assert!(!dl.xfade_active);
        assert_eq!(dl.current_delay, 150.0);

        // Next sample immediately begins transitioning to coalesced pending target 250.0
        let _ = dl.step(0.5);
        assert!(dl.xfade_active);
        assert_eq!(dl.target_delay, 250.0);
        assert!(dl.pending_target.is_none());

        for _ in 0..31 {
            let _ = dl.step(0.5);
        }
        assert!(!dl.xfade_active);
        assert_eq!(dl.current_delay, 250.0);

        // FP16 saturation telemetry on mono and stereo overflow
        assert_eq!(dl.saturation_flags, 0);
        let _ = dl.step(100_000.0);
        assert_ne!(dl.saturation_flags & saturation_flags::FP16_SATURATED, 0);
        dl.reset();
        let (ol, or) = dl.step_stereo(0.25, -100_000.0);
        assert!(ol.is_finite() && or.is_finite());
        assert_ne!(dl.saturation_flags & saturation_flags::FP16_SATURATED, 0);

        let d_out = dl.step_diffuser(0.8, 31, 0.5);
        assert!(d_out.is_finite());
    }

    #[test]
    fn test_diffuser_energy_conservation() {
        let _audio = FtzDazGuard::new();
        let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
        let mut energy_in = 0.0_f64;
        let mut energy_out = 0.0_f64;
        for n in 0..1000 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let y = dl.step_diffuser(x, 19, 0.65);
            energy_in += (x as f64) * (x as f64);
            energy_out += (y as f64) * (y as f64);
        }
        assert!((energy_in - energy_out).abs() < 1e-4);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_schroeder_diffuser_parseval_energy(
            a_diff in -0.7f32..0.7f32,
            d in 4usize..32usize,
        ) {
            let _audio = FtzDazGuard::new();
            let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
            let mut e_in = 0.0_f64;
            let mut e_out = 0.0_f64;
            for n in 0..512 {
                let x = if n < 8 { ((n as f32) * 0.7 + 0.2).sin() } else { 0.0 };
                let y = dl.step_diffuser(x, d, a_diff);
                e_in += (x as f64) * (x as f64);
                e_out += (y as f64) * (y as f64);
            }
            prop_assert!((e_in - e_out).abs() < 5e-3);
        }
    }
}
