//! Master-mode `4x` / `2x` (`HB63` + `z^-1@2x` + `HB23` + `Box2` + `4`-slot cold control delay ring)
//! core-only oversampler state (`SonaColdTail`, `ref[arch.bounce.oversampling]`, `ADR-0026`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`.

#![forbid(unsafe_code)]

use dnam_kernels::{
    HbFir23Downsampler4Ch, HbFir23UpsamplerStereo, HbFir63Downsampler4Ch, HbFir63UpsamplerStereo,
    StereoVec8, box2_quadrature_stereo8,
};

use crate::graph::GraphBackend;
use crate::math::{SimdVec4, flush_recursive_state_f32};

/// Exact microchunk delay depth (`delta_D_up = 16` host samples = `4` cold `4`-frame microchunks)
/// separating Master-mode `HB63 + z^-1@2x + HB23` upsampling audio arrival from `1x` control ticks.
pub const MASTER_CTRL_DELAY_MICROCHUNKS: usize = 4;

/// Maximum `CoreQuad` count in `SONA v4.0` (`[3, 3, 2, 2]` Ultra = `10` quads).
const MAX_CORE_QUADS_V4: usize = dnam_kernels::MAX_CORE_QUADS_V4;
/// Maximum `Core` lane count in `SONA v4.0` (`40` lanes).
const MAX_CORE_LANES_V4: usize = dnam_kernels::MAX_TOTAL_SONA_LANES_V4;

/// Core-bound control parameters captured at a `4`-frame microchunk boundary (`delta_D_up = 16`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MasterCtrlSnapshotCold {
    pub latched_stage_params: [f32; 32],
    pub latched_z_stage: [f32; 8],
    pub lane_g_scale: [f32; MAX_CORE_LANES_V4],
    pub lane_k_delta: [f32; MAX_CORE_LANES_V4],
    pub lane_wf_delta: [f32; MAX_CORE_LANES_V4],
    pub quad_adaa_scale: [f32; MAX_CORE_QUADS_V4],
    pub quad_lru_bias: [f32; MAX_CORE_QUADS_V4],
}

impl Default for MasterCtrlSnapshotCold {
    fn default() -> Self {
        Self {
            latched_stage_params: [0.0; 32],
            latched_z_stage: [0.0; 8],
            lane_g_scale: [1.0; MAX_CORE_LANES_V4],
            lane_k_delta: [0.0; MAX_CORE_LANES_V4],
            lane_wf_delta: [0.0; MAX_CORE_LANES_V4],
            quad_adaa_scale: [1.0; MAX_CORE_QUADS_V4],
            quad_lru_bias: [0.0; MAX_CORE_QUADS_V4],
        }
    }
}

/// Zero-allocation `4`-slot circular microchunk delay ring (`delta_D_up = 16` samples) in `SonaColdTail`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MasterCtrlDelayRingCold {
    pub slots: [MasterCtrlSnapshotCold; MASTER_CTRL_DELAY_MICROCHUNKS],
    pub cursor: usize,
}

impl Default for MasterCtrlDelayRingCold {
    fn default() -> Self {
        Self::new()
    }
}

impl MasterCtrlDelayRingCold {
    /// Creates an identity-initialized `4`-slot cold control delay ring.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: [MasterCtrlSnapshotCold::default(); MASTER_CTRL_DELAY_MICROCHUNKS],
            cursor: 0,
        }
    }

    /// Resets all `4` slots to resting defaults and rewinds the write cursor.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Seeds all `4` delay slots with `initial` so `t = 0` starts with zero transient step.
    pub fn seed(&mut self, initial: MasterCtrlSnapshotCold) {
        self.slots = [initial; MASTER_CTRL_DELAY_MICROCHUNKS];
        self.cursor = 0;
    }

    /// Pushes `newest` microchunk snapshot and pops the snapshot delayed by `4` microchunks (`16` samples).
    #[inline(always)]
    pub fn push_and_pop(&mut self, newest: MasterCtrlSnapshotCold) -> MasterCtrlSnapshotCold {
        let idx = self.cursor & (MASTER_CTRL_DELAY_MICROCHUNKS - 1);
        let delayed = self.slots[idx];
        self.slots[idx] = newest;
        self.cursor = (idx + 1) & (MASTER_CTRL_DELAY_MICROCHUNKS - 1);
        delayed
    }
}

/// Cold `Master`-mode `4x` (`HB63` + `z^-1@2x` + `HB23`) and `2x` (`HB63`) oversampler +
/// per-quad `Box2` composite quadrature state housed in `SonaColdTail`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoreOversamplerMasterCold {
    /// Stage-1 `1x -> 2x` `63`-tap Kaiser half-band FIR upsampler (`248 B`).
    pub up63: HbFir63UpsamplerStereo,
    /// Stage-1 `2x -> 1x` `63`-tap Kaiser half-band FIR 4-channel downsampler (`752 B`).
    pub dn63: HbFir63Downsampler4Ch,
    /// Stage-2 `2x -> 4x` `23`-tap Remez half-band FIR upsampler (`88 B`).
    pub up23: HbFir23UpsamplerStereo,
    /// Stage-2 `4x -> 2x` `23`-tap Remez half-band FIR 4-channel downsampler (`272 B`).
    pub dn23: HbFir23Downsampler4Ch,
    /// Alignment register `z^-1@2x` (`+0.5` `1x`-sample = exact integer `PDC = 37` at `4x`, `8 B`).
    pub z_m1_2x: [f32; 2],
    /// Previous `4x` core outputs `[odd_l, odd_r, even_l, even_r]` per `CoreQuad` for `Box2`.
    pub box2_prev_quad: [[[f32; 4]; 4]; MAX_CORE_QUADS_V4],
    /// `4`-slot (`delta_D_up = 16`) cold control snapshot ring.
    pub ctrl_delay_ring: MasterCtrlDelayRingCold,
    /// Latched Core control snapshot popped at the active `4`-frame microchunk boundary (`delta_D_up = 16`).
    pub active_master_core_ctrl: MasterCtrlSnapshotCold,
}

impl Default for CoreOversamplerMasterCold {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreOversamplerMasterCold {
    /// Creates a zero-initialized [`CoreOversamplerMasterCold`] at `+0.0` equilibrium.
    #[must_use]
    pub fn new() -> Self {
        Self {
            up63: HbFir63UpsamplerStereo::new(),
            dn63: HbFir63Downsampler4Ch::new(),
            up23: HbFir23UpsamplerStereo::new(),
            dn23: HbFir23Downsampler4Ch::new(),
            z_m1_2x: [0.0; 2],
            box2_prev_quad: [[[0.0; 4]; 4]; MAX_CORE_QUADS_V4],
            ctrl_delay_ring: MasterCtrlDelayRingCold::new(),
            active_master_core_ctrl: MasterCtrlSnapshotCold::default(),
        }
    }

    /// Resets all FIR histories, `z^-1@2x` alignment register, `Box2` states, and control ring.
    pub fn reset(&mut self) {
        self.up63.reset();
        self.dn63.reset();
        self.up23.reset();
        self.dn23.reset();
        self.z_m1_2x = [0.0; 2];
        self.box2_prev_quad = [[[0.0; 4]; 4]; MAX_CORE_QUADS_V4];
        self.ctrl_delay_ring.reset();
        self.active_master_core_ctrl = MasterCtrlSnapshotCold::default();
    }

    /// Upsamples one `1x` stereo sample `[in_l, in_r]` to four chronological `4x` samples
    /// via `HB63 -> z^-1@2x -> HB23`.
    #[inline(always)]
    pub fn upsample_1x_to_4x(&mut self, in_l: f32, in_r: f32) -> [[f32; 2]; 4] {
        let s2 = self.up63.step([in_l, in_r]);
        let d2_a = self.z_m1_2x;
        let d2_b = s2[0];
        self.z_m1_2x = [
            flush_recursive_state_f32(s2[1][0]),
            flush_recursive_state_f32(s2[1][1]),
        ];
        let s4_a = self.up23.step(d2_a);
        let s4_b = self.up23.step(d2_b);
        [s4_a[0], s4_a[1], s4_b[0], s4_b[1]]
    }

    /// Downsamples four chronological `4x` 4-channel frames `[pre_L, pre_R, dir_L, dir_R]`
    /// back to one `1x` frame via `HB23 -> HB63` (`PDC = 37`).
    #[inline(always)]
    pub fn downsample_4x_to_1x<B: GraphBackend>(&mut self, sub4: [[f32; 4]; 4]) -> [f32; 4] {
        let y2_a = self
            .dn23
            .step_vec4(B::Vec4::load(&sub4[0]), B::Vec4::load(&sub4[1]))
            .to_array();
        let y2_b = self
            .dn23
            .step_vec4(B::Vec4::load(&sub4[2]), B::Vec4::load(&sub4[3]))
            .to_array();
        self.dn63
            .step_vec4(B::Vec4::load(&y2_a), B::Vec4::load(&y2_b))
            .to_array()
    }

    /// Upsamples one `1x` stereo sample `[in_l, in_r]` to two `2x` samples via `HB63` (`176.4/192 kHz`).
    #[inline(always)]
    pub fn upsample_1x_to_2x(&mut self, in_l: f32, in_r: f32) -> [[f32; 2]; 2] {
        self.up63.step([in_l, in_r])
    }

    /// Downsamples two `2x` 4-channel frames back to one `1x` frame via `HB63` (`PDC = 31`).
    #[inline(always)]
    pub fn downsample_2x_to_1x<B: GraphBackend>(&mut self, sub2: [[f32; 4]; 2]) -> [f32; 4] {
        self.dn63
            .step_vec4(B::Vec4::load(&sub2[0]), B::Vec4::load(&sub2[1]))
            .to_array()
    }

    /// Evaluates `Box2` composite quadrature on one `4x` `CoreQuad`'s `(adaa_odd, adaa_even)` pair,
    /// automatically bypassing `Box2` per lane when `lin_mix == 1.0`.
    #[inline(always)]
    pub fn apply_box2_quad<B: GraphBackend>(
        &mut self,
        quad_idx: usize,
        adaa_odd: B::Stereo8,
        adaa_even: B::Stereo8,
        lin_mix: B::Stereo8,
    ) -> (B::Stereo8, B::Stereo8) {
        let slot = &mut self.box2_prev_quad[quad_idx];
        let prev_odd = B::Stereo8::from_lr(B::Vec4::load(&slot[0]), B::Vec4::load(&slot[1]));
        let prev_even = B::Stereo8::from_lr(B::Vec4::load(&slot[2]), B::Vec4::load(&slot[3]));

        let (odd_l, odd_r) = adaa_odd.into_lr();
        let (even_l, even_r) = adaa_even.into_lr();
        odd_l.store(&mut slot[0]);
        odd_r.store(&mut slot[1]);
        even_l.store(&mut slot[2]);
        even_r.store(&mut slot[3]);

        (
            box2_quadrature_stereo8(prev_odd, adaa_odd, lin_mix),
            box2_quadrature_stereo8(prev_even, adaa_even, lin_mix),
        )
    }
}
