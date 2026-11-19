//! `3 kHz` Per-Block `FiLM`, Zero-DC `b_n` Shift & Unified Control Dispatch (`stage2_gru/film.rs`).

use super::block::{MAX_BLOCK_LANES, MAX_GROUPS, NUM_GRU_BLOCKS, V3GruBlockHot};
use super::math_pade::tanh_pade5_vec4;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};
use crate::stages::stage1_svf::exp2_signed_oct_f32;

/// Number of `3 kHz` macro control knobs supported by [`V3ControlBusParams`].
pub const NUM_V3_MACRO_KNOBS: usize = 8;
/// Maximum per-block Group lanes modulated by runtime FiLM/zero-DC conditioning.
pub const MAX_CTRL_BLOCK_LANES: usize = MAX_BLOCK_LANES;
/// Number of Stage II GRU blocks conditioned by the control bus.
pub const NUM_CTRL_BLOCKS: usize = NUM_GRU_BLOCKS;

const BASE_CTRL_ALPHA_3KHZ: f32 = 0.08;

/// `3 kHz` Control Bus conditioning weights driving Stage I SVF, Tone SVF, Stage II FiLM/Zero-DC, and Stage IV.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3ControlBusParams {
    pub default_knobs: [f32; NUM_V3_MACRO_KNOBS],
    pub w_env_fc_oct: [f32; NUM_V3_MACRO_KNOBS],
    pub w_env_k: [f32; NUM_V3_MACRO_KNOBS],
    pub w_tone_fc_oct: [f32; NUM_V3_MACRO_KNOBS],
    pub w_tone_k: [f32; NUM_V3_MACRO_KNOBS],
    pub w_tone_eq: [[f32; 3]; NUM_V3_MACRO_KNOBS],
    pub w_tank_psi: [f32; NUM_V3_MACRO_KNOBS],
    pub w_film_gain: [[[f32; MAX_CTRL_BLOCK_LANES]; NUM_CTRL_BLOCKS]; NUM_V3_MACRO_KNOBS],
    pub w_film_bn: [[[f32; MAX_CTRL_BLOCK_LANES]; NUM_CTRL_BLOCKS]; NUM_V3_MACRO_KNOBS],
}

impl Default for V3ControlBusParams {
    fn default() -> Self {
        Self::identity()
    }
}

impl V3ControlBusParams {
    /// Resting identity control bus (neutral `0.5` knobs, zero conditioning offsets).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            default_knobs: [0.5; NUM_V3_MACRO_KNOBS],
            w_env_fc_oct: [0.0; NUM_V3_MACRO_KNOBS],
            w_env_k: [0.0; NUM_V3_MACRO_KNOBS],
            w_tone_fc_oct: [0.0; NUM_V3_MACRO_KNOBS],
            w_tone_k: [0.0; NUM_V3_MACRO_KNOBS],
            w_tone_eq: [[0.0; 3]; NUM_V3_MACRO_KNOBS],
            w_tank_psi: [0.0; NUM_V3_MACRO_KNOBS],
            w_film_gain: [[[0.0; MAX_CTRL_BLOCK_LANES]; NUM_CTRL_BLOCKS]; NUM_V3_MACRO_KNOBS],
            w_film_bn: [[[0.0; MAX_CTRL_BLOCK_LANES]; NUM_CTRL_BLOCKS]; NUM_V3_MACRO_KNOBS],
        }
    }

    /// Clamps knob defaults and FiLM weights into Lyapunov/BIBO bounds.
    pub fn clamp_in_place(&mut self) {
        for k in 0..NUM_V3_MACRO_KNOBS {
            self.default_knobs[k] = self.default_knobs[k].clamp(0.0, 1.0);
            self.w_env_fc_oct[k] = self.w_env_fc_oct[k].clamp(-3.0, 3.0);
            self.w_env_k[k] = self.w_env_k[k].clamp(-1.0, 1.0);
            self.w_tone_fc_oct[k] = self.w_tone_fc_oct[k].clamp(-3.0, 3.0);
            self.w_tone_k[k] = self.w_tone_k[k].clamp(-1.5, 1.5);
            for eq in &mut self.w_tone_eq[k] {
                *eq = eq.clamp(-1.5, 1.5);
            }
            self.w_tank_psi[k] = self.w_tank_psi[k].clamp(-1.5, 1.5);
            for b in 0..NUM_CTRL_BLOCKS {
                for c in 0..MAX_CTRL_BLOCK_LANES {
                    self.w_film_gain[k][b][c] = self.w_film_gain[k][b][c].clamp(-1.0, 1.0);
                    self.w_film_bn[k][b][c] = self.w_film_bn[k][b][c].clamp(-1.5, 1.5);
                }
            }
        }
    }
}

/// Hot `3 kHz` Control Bus state emitting FiLM gains, Zero-DC effective biases, and SVF deltas.
///
/// Cold `params` sit at the trailing tail so steady-state `dirty == false` blocks keep hot
/// L1D working lines minimal.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct V3ControlBusState {
    pub target_knobs: [f32; NUM_V3_MACRO_KNOBS],
    pub smoothed_knobs: [f32; NUM_V3_MACRO_KNOBS],
    pub dirty: bool,
    pub env_fc_oct_delta: f32,
    pub env_fc_scale: f32,
    pub env_k_delta: f32,
    pub tone_fc_oct_delta: f32,
    pub tone_k_delta: f32,
    pub tone_eq_delta: [f32; 3],
    pub tank_psi_delta: f32,
    pub block_film_gain: [[[f32; 4]; MAX_GROUPS]; NUM_CTRL_BLOCKS],
    pub block_bn_eff: [[[f32; 4]; MAX_GROUPS]; NUM_CTRL_BLOCKS],
    pub block_tanh_bn_eff: [[[f32; 4]; MAX_GROUPS]; NUM_CTRL_BLOCKS],
    pub params: V3ControlBusParams,
}

impl V3ControlBusState {
    /// Initializes hot control bus from cold conditioning parameters and packed GRU blocks.
    #[must_use]
    pub fn from_cold<B: MathBackend>(
        params: &V3ControlBusParams,
        blocks: &[V3GruBlockHot; NUM_GRU_BLOCKS],
        num_blocks: usize,
        block_groups: &[usize; NUM_GRU_BLOCKS],
    ) -> Self {
        let mut p = *params;
        p.clamp_in_place();
        let mut st = Self {
            target_knobs: p.default_knobs,
            smoothed_knobs: p.default_knobs,
            dirty: true,
            env_fc_oct_delta: 0.0,
            env_fc_scale: 1.0,
            env_k_delta: 0.0,
            tone_fc_oct_delta: 0.0,
            tone_k_delta: 0.0,
            tone_eq_delta: [0.0; 3],
            tank_psi_delta: 0.0,
            block_film_gain: [[[1.0; 4]; MAX_GROUPS]; NUM_CTRL_BLOCKS],
            block_bn_eff: [[[0.0; 4]; MAX_GROUPS]; NUM_CTRL_BLOCKS],
            block_tanh_bn_eff: [[[0.0; 4]; MAX_GROUPS]; NUM_CTRL_BLOCKS],
            params: p,
        };
        st.step_3khz::<B>(num_blocks, block_groups, blocks);
        st
    }

    /// Sets a macro knob `[0..7]` normalized target in `[0.0, 1.0]`.
    pub fn set_knob(&mut self, index: usize, value: f32) {
        if let Some(slot) = self.target_knobs.get_mut(index) {
            let clamped = value.clamp(0.0, 1.0);
            if (*slot - clamped).abs() > 0.0 || (self.smoothed_knobs[index] - clamped).abs() > 0.0 {
                *slot = clamped;
                self.dirty = true;
            }
        }
    }

    /// Updates control bus parameters in place and refreshes effective biases.
    pub fn update_params<B: MathBackend>(
        &mut self,
        params: &V3ControlBusParams,
        blocks: &[V3GruBlockHot; NUM_GRU_BLOCKS],
        num_blocks: usize,
        block_groups: &[usize; NUM_GRU_BLOCKS],
    ) {
        let mut p = *params;
        p.clamp_in_place();
        self.params = p;
        self.dirty = true;
        self.step_3khz::<B>(num_blocks, block_groups, blocks);
    }

    /// Refreshes Stage II `FiLM` gain, Zero-DC `b_n`, and `tanh_b_n` from explicit per-block deltas.
    ///
    /// Used by the unified `DnamEngine` `3 kHz` ControlBus dispatch (`stage_targets[0..16]`) to
    /// modulate `FiLM` and Zero-DC bias without recomputing `tanh` at audio rate.
    pub fn apply_direct_block_film<B: MathBackend>(
        &mut self,
        num_blocks: usize,
        block_groups: &[usize; NUM_GRU_BLOCKS],
        blocks: &[V3GruBlockHot; NUM_GRU_BLOCKS],
        gain_deltas: [f32; NUM_CTRL_BLOCKS],
        bn_deltas: [f32; NUM_CTRL_BLOCKS],
    ) {
        let (min_g, max_g) = (B::Vec4::splat(0.25), B::Vec4::splat(4.0));
        let (min_b, max_b) = (B::Vec4::splat(-4.5), B::Vec4::splat(4.5));
        for b in 0..num_blocks.min(NUM_CTRL_BLOCKS) {
            let g_v = B::Vec4::splat(1.0 + gain_deltas[b]).max(min_g).min(max_g);
            let bn_shift = B::Vec4::splat(bn_deltas[b]);
            for g in 0..block_groups[b].min(MAX_GROUPS) {
                g_v.store(&mut self.block_film_gain[b][g]);
                let bn_eff = B::Vec4::load(&blocks[b].groups[g].b_n)
                    .add(bn_shift)
                    .max(min_b)
                    .min(max_b);
                bn_eff.store(&mut self.block_bn_eff[b][g]);
                tanh_pade5_vec4::<B>(bn_eff).store(&mut self.block_tanh_bn_eff[b][g]);
            }
        }
    }

    /// Advances `3 kHz` knob smoothing and computes active `FiLM`, Tone `SVF`, and `tanh(b_n + shift)`.
    #[inline(always)]
    pub fn step_3khz<B: MathBackend>(
        &mut self,
        num_blocks: usize,
        block_groups: &[usize; NUM_GRU_BLOCKS],
        blocks: &[V3GruBlockHot; NUM_GRU_BLOCKS],
    ) {
        if !self.dirty {
            return;
        }
        let mut delta = [0.0_f32; NUM_V3_MACRO_KNOBS];
        let mut max_diff = 0.0_f32;
        for k in 0..NUM_V3_MACRO_KNOBS {
            let sm = self.smoothed_knobs[k]
                + BASE_CTRL_ALPHA_3KHZ * (self.target_knobs[k] - self.smoothed_knobs[k]);
            let clean = flush_recursive_state_f32(sm);
            max_diff = max_diff.max((self.target_knobs[k] - clean).abs());
            self.smoothed_knobs[k] = clean;
            delta[k] = clean - 0.5;
        }
        if max_diff <= 1.0e-7 {
            self.smoothed_knobs = self.target_knobs;
            self.dirty = false;
            for k in 0..NUM_V3_MACRO_KNOBS {
                delta[k] = self.smoothed_knobs[k] - 0.5;
            }
        }

        let (mut env_fc, mut env_k, mut tone_fc, mut tone_k, mut tank_psi) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut tone_eq = [0.0_f32; 3];
        for (k, &d) in delta.iter().enumerate() {
            env_fc += d * self.params.w_env_fc_oct[k];
            env_k += d * self.params.w_env_k[k];
            tone_fc += d * self.params.w_tone_fc_oct[k];
            tone_k += d * self.params.w_tone_k[k];
            tone_eq[0] += d * self.params.w_tone_eq[k][0];
            tone_eq[1] += d * self.params.w_tone_eq[k][1];
            tone_eq[2] += d * self.params.w_tone_eq[k][2];
            tank_psi += d * self.params.w_tank_psi[k];
        }
        self.env_fc_oct_delta = env_fc;
        self.env_fc_scale = exp2_signed_oct_f32(env_fc);
        self.env_k_delta = env_k;
        self.tone_fc_oct_delta = tone_fc;
        self.tone_k_delta = tone_k;
        self.tone_eq_delta = tone_eq;
        self.tank_psi_delta = tank_psi;

        let (one_v, min_g, max_g) = (
            B::Vec4::splat(1.0),
            B::Vec4::splat(0.25),
            B::Vec4::splat(2.5),
        );
        let (min_b, max_b) = (B::Vec4::splat(-4.5), B::Vec4::splat(4.5));
        for b in 0..num_blocks.min(NUM_CTRL_BLOCKS) {
            for g in 0..block_groups[b].min(MAX_GROUPS) {
                let (mut g_acc, mut bn_shift) = (one_v, B::Vec4::zero());
                for (k, &dk) in delta.iter().enumerate() {
                    let dk_v = B::Vec4::splat(dk);
                    let (wg_chunks, _) = self.params.w_film_gain[k][b].as_chunks::<4>();
                    let (wb_chunks, _) = self.params.w_film_bn[k][b].as_chunks::<4>();
                    g_acc = g_acc.fma(dk_v, B::Vec4::load(&wg_chunks[g]));
                    bn_shift = bn_shift.fma(dk_v, B::Vec4::load(&wb_chunks[g]));
                }
                g_acc
                    .max(min_g)
                    .min(max_g)
                    .store(&mut self.block_film_gain[b][g]);
                let bn_eff = B::Vec4::load(&blocks[b].groups[g].b_n)
                    .add(bn_shift)
                    .max(min_b)
                    .min(max_b);
                bn_eff.store(&mut self.block_bn_eff[b][g]);
                tanh_pade5_vec4::<B>(bn_eff).store(&mut self.block_tanh_bn_eff[b][g]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_control_bus_3khz_film_and_direct_deltas() {
        let mut p_ctrl = V3ControlBusParams::identity();
        p_ctrl.w_tone_fc_oct[0] = 1.5;
        p_ctrl.w_tone_eq[0] = [0.4, -0.2, 0.3];
        p_ctrl.w_film_gain[0][0][0] = 0.5;
        p_ctrl.w_film_bn[0][0][0] = 0.4;
        let blocks = [V3GruBlockHot::default(); NUM_GRU_BLOCKS];
        let mut ctrl =
            V3ControlBusState::from_cold::<DefaultMathBackend>(&p_ctrl, &blocks, 3, &[2, 1, 1]);
        ctrl.set_knob(0, 1.0);
        for _ in 0..32 {
            ctrl.step_3khz::<DefaultMathBackend>(3, &[2, 1, 1], &blocks);
        }
        assert!(ctrl.block_film_gain[0][0][0] > 1.05 && ctrl.block_film_gain[0][0][0] <= 2.5);

        ctrl.apply_direct_block_film::<DefaultMathBackend>(
            3,
            &[2, 1, 1],
            &blocks,
            [0.2, -0.1, 0.0],
            [0.1, 0.0, -0.2],
        );
        assert!((ctrl.block_film_gain[0][0][0] - 1.2).abs() < 1.0e-6);
        assert!((ctrl.block_bn_eff[2][0][0] + 0.2).abs() < 1.0e-6);
    }
}
