//! Layer 1 Host-Shell 8-Port Patchbay (`HostShellPatchbay`), 4CM/WDW/Dual-Amp routing,
//! structural `S_{ret->send} == 0_{2x2}` isolation, and post-VM voltage-normalized M/S stereo width.
//!
//! Wraps pure `2x2` `dnam-vm` instances cleanly outside the VM boundary with zero runtime
//! crossovers (`0` `LR4`, `0` `200 Hz` splitters), zero `FDIV`, zero `FSQRT`, zero `IDIV`,
//! and zero `libm` calls.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_crossbar_topology.md` §2.1–§2.3 & §3.1–§3.2
//! (`ref[dsp.router.ports]`, `ref[dsp.router.return-to-send-isolation]`,
//! `ref[dsp.router.wdw-contract]`, `ref[dsp.router.parallel-alignment]`).

#![forbid(unsafe_code)]

pub mod align;

pub use align::{
    IoAlignmentMode, PARALLEL_ALIGN_CAPACITY, ParallelAlignLine, ParallelAlignmentMode,
    RETURN_ALIGN_CAPACITY, ReturnAligner, eval_min_latency_4tap_alignment,
    prepare_kaiser_sinc_alignment_weights_into,
};

use dnam_kernels::scrub_and_flush_f32_bits;
use dnam_vm::OFFLINE_BOUNCE_PDC_SAMPLES;

const MAX_SCATTER_GAIN_F32: f32 = 4.0;
const MAX_MS_WIDTH_F32: f32 = 2.0;
const DRY_PDC_RING_CAPACITY: usize = 16;
const DRY_PDC_RING_MASK: usize = DRY_PDC_RING_CAPACITY - 1;

const _: () = assert!(
    DRY_PDC_RING_CAPACITY.is_power_of_two() && OFFLINE_BOUNCE_PDC_SAMPLES < DRY_PDC_RING_CAPACITY
);

/// Wet-Dry-Wet (`WDW`) center dry tap mode (`ref[dsp.router.wdw-contract]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WdwDryTapMode {
    /// Center dry on `Port Out 1` carries the Mid contribution `m_c` of the pre-spatial core
    /// (`Crossbar + CoreQuad + ModalConvolver`); `Ports Out 2/3` carry Spatial `Transport`
    /// output `m_w ± alpha * s_w`.
    #[default]
    PreSpatialCore,
    /// Center dry on `Port Out 1` carries voltage-normalized Mid `0.5 * (x_L + x_R)` of the
    /// direct input; `Ports Out 2/3` carry the wet effect response.
    DirectDry,
}

/// Outer host-shell multi-port I/O routing topology (`ref[dsp.router.ports]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IoRoutingMode {
    /// Standard `2x2` stereo routing (`Inst L/R -> VM L/R -> Main L/R`, `Send L/R` mirrors inst).
    #[default]
    Stereo,
    /// Mono routing (`Inst L` normaled to `R`, `Main L/R` folded to voltage-normalized Mid).
    Mono,
    /// 4-Cable Method (`Inst L -> Send L` to amp preamp; `FX Return L/R -> VM -> Main L/R`).
    FourCableMethod,
    /// Dual-amplifier send/return routing (`Inst L/R -> Send L/R`; `FX Return L/R -> VM`).
    DualAmp,
    /// Wet-Dry-Wet 3-output routing (`Port Out 1 = WDW Dry Center`, `Ports Out 2/3 = Wet L/R`).
    WetDryWet,
}

/// Evaluates explicit WDW output ports `[port_out1_dry, port_out2_wet_l, port_out3_wet_r]`
/// from `[wet_mid, wet_side]` (`ref[dsp.router.wdw-contract]`).
#[inline(always)]
#[must_use]
pub const fn eval_wdw_routing_outputs(
    mode: WdwDryTapMode,
    core_mid: f32,
    wet_ms: [f32; 2],
    width_alpha: f32,
    direct_lr: [f32; 2],
) -> [f32; 3] {
    let alpha = HostShellPatchbay::clamp_ms_width(width_alpha);
    let wet_l = wet_ms[0] + alpha * wet_ms[1];
    let wet_r = wet_ms[0] - alpha * wet_ms[1];
    let dry = match mode {
        WdwDryTapMode::PreSpatialCore => core_mid,
        WdwDryTapMode::DirectDry => 0.5 * (direct_lr[0] + direct_lr[1]),
    };
    [dry, wet_l, wet_r]
}

#[inline(always)]
fn sanitize_2x2_matrix(mat: [[f32; 2]; 2], identity_fallback: bool) -> [[f32; 2]; 2] {
    let mut out = [[0.0_f32; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            let v = mat[r][c];
            out[r][c] = if v.is_finite() {
                v.clamp(-MAX_SCATTER_GAIN_F32, MAX_SCATTER_GAIN_F32)
            } else if identity_fallback && r == c {
                1.0
            } else {
                0.0
            };
        }
    }
    out
}

/// Layer 1 Host-Shell 8-Port Patchbay wrapping a pure `2x2` `dnam-vm` instance.
///
/// Structurally omits `S_{ret->send}` so external feedback through hardware effects loops is
/// impossible by construction (`S_{ret->send} == 0_{2x2}`,
/// `ref[dsp.router.return-to-send-isolation]`).
#[derive(Clone, Debug)]
pub struct HostShellPatchbay {
    /// Instrument-to-Send `2x2` matrix (`S_{inst->send}`).
    pub s_inst_to_send: [[f32; 2]; 2],
    /// Instrument-to-Downstream-VM `2x2` matrix (`S_{inst->down}`).
    pub s_inst_to_down: [[f32; 2]; 2],
    /// Aligned-Return-to-Downstream-VM `2x2` matrix (`S_{ret->down}`).
    pub s_ret_to_down: [[f32; 2]; 2],
    /// Left external FX return fractional Catmull-Rom latency aligner.
    pub return_align_l: ReturnAligner,
    /// Right external FX return fractional Catmull-Rom latency aligner.
    pub return_align_r: ReturnAligner,
    /// Left parallel-path sub-sample alignment line (`FullBand64Tap` / `16Tap` / `4Tap`).
    pub parallel_align_l: ParallelAlignLine,
    /// Right parallel-path sub-sample alignment line (`FullBand64Tap` / `16Tap` / `4Tap`).
    pub parallel_align_r: ParallelAlignLine,
    /// WDW center dry selector (`PreSpatialCore` vs `DirectDry`).
    pub wdw_mode: WdwDryTapMode,
    /// Enables WDW routing onto `Port Out 1` (`dry_tap`) and `Ports Out 2/3` (`wet_l/r`).
    pub wdw_enabled: bool,
    /// Post-VM host-shell M/S stereo width clamped to `[0.0, 2.0]` (default `1.0`).
    pub ms_width: f32,
    /// Active multi-port routing mode.
    pub routing_mode: IoRoutingMode,
    /// Enables parallel-path Kaiser/Catmull-Rom alignment on the direct dry tap.
    pub parallel_align_enabled: bool,
    dry_pdc_ring: [f32; DRY_PDC_RING_CAPACITY],
    dry_pdc_ptr: usize,
    fs_vm_hz: f64,
    offline_bounce: bool,
}

impl Default for HostShellPatchbay {
    fn default() -> Self {
        Self::new()
    }
}

impl HostShellPatchbay {
    /// `2x2` identity scattering block.
    pub const IDENTITY_2X2: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
    /// `2x2` zero scattering block.
    pub const ZERO_2X2: [[f32; 2]; 2] = [[0.0, 0.0], [0.0, 0.0]];

    /// Creates a resting `48 kHz` stereo [`HostShellPatchbay`] with `ms_width = 1.0` and zero PDC.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            s_inst_to_send: Self::IDENTITY_2X2,
            s_inst_to_down: Self::IDENTITY_2X2,
            s_ret_to_down: Self::IDENTITY_2X2,
            return_align_l: ReturnAligner::new(),
            return_align_r: ReturnAligner::new(),
            parallel_align_l: ParallelAlignLine::new(ParallelAlignmentMode::FullBand64Tap),
            parallel_align_r: ParallelAlignLine::new(ParallelAlignmentMode::FullBand64Tap),
            wdw_mode: WdwDryTapMode::PreSpatialCore,
            wdw_enabled: false,
            ms_width: 1.0,
            routing_mode: IoRoutingMode::Stereo,
            parallel_align_enabled: false,
            dry_pdc_ring: [0.0; DRY_PDC_RING_CAPACITY],
            dry_pdc_ptr: 0,
            fs_vm_hz: 48_000.0,
            offline_bounce: false,
        }
    }

    #[inline(always)]
    const fn host_rate_hz(&self) -> f64 {
        if self.offline_bounce {
            0.25 * self.fs_vm_hz
        } else {
            self.fs_vm_hz
        }
    }

    /// Clamps `ms_width` to `[0.0, 2.0]`, returning `1.0` for non-finite inputs.
    #[inline(always)]
    #[must_use]
    pub const fn clamp_ms_width(width: f32) -> f32 {
        if !width.is_finite() {
            1.0
        } else if width < 0.0 {
            0.0
        } else if width > MAX_MS_WIDTH_F32 {
            MAX_MS_WIDTH_F32
        } else {
            width
        }
    }

    /// Voltage-normalized Mid/Side decomposition `(0.5 * (l + r), 0.5 * (l - r))`.
    #[inline(always)]
    #[must_use]
    pub const fn voltage_normalize_ms(l: f32, r: f32) -> (f32, f32) {
        (0.5 * (l + r), 0.5 * (l - r))
    }

    /// Voltage-normalized Mid/Side inverse reconstruction `(mid + side, mid - side)`.
    #[inline(always)]
    #[must_use]
    pub const fn voltage_reconstruct_ms(mid: f32, side: f32) -> (f32, f32) {
        (mid + side, mid - side)
    }

    /// Applies post-VM voltage-normalized M/S width `0.5 * (l + r) ± width * 0.5 * (l - r)`
    /// (`0` ULP pass-through at `width == 1.0`).
    #[inline(always)]
    #[must_use]
    pub const fn apply_ms_width(l: f32, r: f32, width: f32) -> (f32, f32) {
        let w = Self::clamp_ms_width(width);
        if w == 1.0 {
            return (l, r);
        }
        let mid = 0.5 * (l + r);
        let side = 0.5 * (l - r);
        (mid + w * side, mid - w * side)
    }

    /// Sets explicit `2x2` patchbay scattering blocks (`S_{ret->send}` remains unrepresented `0`).
    #[inline]
    pub fn set_scattering_matrices(
        &mut self,
        inst_to_send: [[f32; 2]; 2],
        inst_to_down: [[f32; 2]; 2],
        ret_to_down: [[f32; 2]; 2],
    ) {
        self.s_inst_to_send = sanitize_2x2_matrix(inst_to_send, true);
        self.s_inst_to_down = sanitize_2x2_matrix(inst_to_down, false);
        self.s_ret_to_down = sanitize_2x2_matrix(ret_to_down, true);
    }

    /// Extracts causal `2x2` scattering blocks from a `4x4` matrix, structurally discarding
    /// `S_{ret->send}` (`s[0..2][2..4]`, `ref[dsp.router.return-to-send-isolation]`).
    #[inline]
    pub fn set_scattering_matrix_4x4(&mut self, s: [[f32; 4]; 4]) {
        let inst_to_send = [[s[0][0], s[0][1]], [s[1][0], s[1][1]]];
        let inst_to_down = [[s[2][0], s[2][1]], [s[3][0], s[3][1]]];
        let ret_to_down = [[s[2][2], s[2][3]], [s[3][2], s[3][3]]];
        self.set_scattering_matrices(inst_to_send, inst_to_down, ret_to_down);
    }

    /// Selects a standard [`IoRoutingMode`] preset and updates scattering blocks and WDW state.
    #[inline(never)]
    pub fn set_routing_mode(&mut self, mode: IoRoutingMode) {
        self.routing_mode = mode;
        match mode {
            IoRoutingMode::Stereo | IoRoutingMode::Mono => {
                self.s_inst_to_send = Self::IDENTITY_2X2;
                self.s_inst_to_down = Self::IDENTITY_2X2;
                self.s_ret_to_down = Self::IDENTITY_2X2;
                self.wdw_enabled = false;
            }
            IoRoutingMode::FourCableMethod => {
                self.s_inst_to_send = [[1.0, 0.0], [0.0, 0.0]];
                self.s_inst_to_down = Self::ZERO_2X2;
                self.s_ret_to_down = Self::IDENTITY_2X2;
                self.wdw_enabled = false;
            }
            IoRoutingMode::DualAmp => {
                self.s_inst_to_send = Self::IDENTITY_2X2;
                self.s_inst_to_down = Self::ZERO_2X2;
                self.s_ret_to_down = Self::IDENTITY_2X2;
                self.wdw_enabled = false;
            }
            IoRoutingMode::WetDryWet => {
                self.s_inst_to_send = [[1.0, 0.0], [0.0, 0.0]];
                self.s_inst_to_down = Self::IDENTITY_2X2;
                self.s_ret_to_down = Self::IDENTITY_2X2;
                self.wdw_enabled = true;
            }
        }
    }

    /// Sets the WDW center dry tap selector (`PreSpatialCore` vs `DirectDry`).
    #[inline(never)]
    pub fn set_wdw_mode(&mut self, mode: WdwDryTapMode) {
        self.wdw_mode = mode;
    }

    /// Enables or disables WDW output routing onto `Port Out 1` and `Ports Out 2/3`.
    #[inline(never)]
    pub fn set_wdw_enabled(&mut self, enabled: bool) {
        self.wdw_enabled = enabled;
    }

    /// Sets post-VM M/S stereo width clamped to `[0.0, 2.0]`.
    #[inline]
    pub fn set_ms_width(&mut self, width: f32) {
        self.ms_width = Self::clamp_ms_width(width);
    }

    /// Sets the parallel-path sub-sample alignment mode on both channels (`0 FDIV`).
    #[inline]
    pub fn set_alignment_mode(&mut self, mode: ParallelAlignmentMode) {
        let fs_host = self.host_rate_hz();
        self.parallel_align_l.set_mode(mode);
        self.parallel_align_r.set_mode(mode);
        self.parallel_align_l
            .reconfigure_host_rate(fs_host, self.offline_bounce);
        self.parallel_align_r
            .reconfigure_host_rate(fs_host, self.offline_bounce);
    }

    /// Enables or disables parallel-path Kaiser/Catmull-Rom alignment on the direct dry tap.
    #[inline]
    pub fn set_parallel_alignment_enabled(&mut self, enabled: bool) {
        self.parallel_align_enabled = enabled;
    }

    /// Configures external FX return fractional latency delays in active VM samples.
    pub fn configure_return_alignment(&mut self, delay_l_samples: f32, delay_r_samples: f32) {
        self.return_align_l.set_delay(delay_l_samples);
        self.return_align_r.set_delay(delay_r_samples);
    }

    /// Configures `48 kHz`-anchored external FX return fractional latency delays.
    #[inline]
    pub fn configure_return_alignment_48k(&mut self, delay_l_48k: f32, delay_r_48k: f32) {
        self.return_align_l.set_delay_48k(delay_l_48k);
        self.return_align_r.set_delay_48k(delay_r_48k);
    }

    /// Configures parallel alignment mode, engagement flag, and `48 kHz`-anchored channel skews.
    pub fn configure_parallel_alignment(
        &mut self,
        mode: ParallelAlignmentMode,
        enabled: bool,
        delta_l_48k: f32,
        delta_r_48k: f32,
    ) {
        let fs_host = self.host_rate_hz();
        self.parallel_align_l.set_mode(mode);
        self.parallel_align_r.set_mode(mode);
        self.parallel_align_l
            .set_delta_48k_host(delta_l_48k, fs_host, self.offline_bounce);
        self.parallel_align_r
            .set_delta_48k_host(delta_r_48k, fs_host, self.offline_bounce);
        self.parallel_align_enabled = enabled;
    }

    /// Clears all delay-line histories while preserving routing matrices and sample-rate setup.
    pub fn reset(&mut self) {
        self.return_align_l.reset();
        self.return_align_r.reset();
        self.parallel_align_l.reset();
        self.parallel_align_r.reset();
        self.dry_pdc_ring.fill(0.0);
        self.dry_pdc_ptr = 0;
    }

    /// Reconfigures return and parallel alignment lines for `(fs_vm_hz, offline_bounce)`
    /// (`0` heap allocs, `0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`).
    pub fn reconfigure_sample_rate(&mut self, fs_vm_hz: f64, offline_bounce: bool) {
        if !fs_vm_hz.is_finite() || fs_vm_hz <= 0.0 {
            return;
        }
        self.fs_vm_hz = fs_vm_hz;
        self.offline_bounce = offline_bounce;
        let fs_host = if offline_bounce {
            0.25 * fs_vm_hz
        } else {
            fs_vm_hz
        };
        self.return_align_l.reconfigure_sample_rate(fs_host);
        self.return_align_r.reconfigure_sample_rate(fs_host);
        self.parallel_align_l
            .reconfigure_host_rate(fs_host, offline_bounce);
        self.parallel_align_r
            .reconfigure_host_rate(fs_host, offline_bounce);
    }

    /// Returns total host-sample PDC for the current parallel alignment and bounce settings.
    #[inline]
    #[must_use]
    pub const fn pdc_host_samples(&self) -> usize {
        if self.parallel_align_enabled {
            self.parallel_align_l.pdc_host_samples(self.offline_bounce)
        } else if self.offline_bounce {
            OFFLINE_BOUNCE_PDC_SAMPLES
        } else {
            0
        }
    }

    /// Routes 4 physical input ports `[inst_l, inst_r, ret_l, ret_r]` into
    /// `(vm_in, send_out, dry_tap)` while structurally isolating `send_out` from `ret`
    /// (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`).
    #[inline(always)]
    pub fn route_pre_vm(&mut self, in_ports: [f32; 4]) -> ([f32; 2], [f32; 2], f32) {
        let inst_l = scrub_and_flush_f32_bits(in_ports[0]);
        let inst_r_raw = scrub_and_flush_f32_bits(in_ports[1]);
        let inst_r = if self.routing_mode == IoRoutingMode::Mono {
            inst_l
        } else {
            inst_r_raw
        };
        let ret_l = self.return_align_l.step(in_ports[2]);
        let ret_r = self.return_align_r.step(in_ports[3]);

        let send_out = [
            self.s_inst_to_send[0][0] * inst_l + self.s_inst_to_send[0][1] * inst_r,
            self.s_inst_to_send[1][0] * inst_l + self.s_inst_to_send[1][1] * inst_r,
        ];

        let vm_in = [
            self.s_inst_to_down[0][0] * inst_l
                + self.s_inst_to_down[0][1] * inst_r
                + self.s_ret_to_down[0][0] * ret_l
                + self.s_ret_to_down[0][1] * ret_r,
            self.s_inst_to_down[1][0] * inst_l
                + self.s_inst_to_down[1][1] * inst_r
                + self.s_ret_to_down[1][0] * ret_l
                + self.s_ret_to_down[1][1] * ret_r,
        ];

        let raw_dry_tap = 0.5 * (inst_l + inst_r);
        self.dry_pdc_ring[self.dry_pdc_ptr] = raw_dry_tap;
        let delayed_dry_tap = self.dry_pdc_ring[(self.dry_pdc_ptr + DRY_PDC_RING_CAPACITY
            - OFFLINE_BOUNCE_PDC_SAMPLES)
            & DRY_PDC_RING_MASK];
        self.dry_pdc_ptr = (self.dry_pdc_ptr + 1) & DRY_PDC_RING_MASK;

        let dry_tap = if self.parallel_align_enabled {
            let dl = self.parallel_align_l.step(inst_l);
            let dr = self.parallel_align_r.step(inst_r);
            0.5 * (dl + dr)
        } else if self.offline_bounce {
            delayed_dry_tap
        } else {
            raw_dry_tap
        };

        (vm_in, send_out, dry_tap)
    }

    /// Maps `(vm_out, dry_tap, send_out)` into canonical physical output ports
    /// `[Port Out 0 (Send L), Port Out 1 (Send R / WDW Dry), Port Out 2 (Main L), Port Out 3 (Main R)]`
    /// with post-VM voltage-normalized M/S stereo width (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`).
    #[inline(always)]
    #[must_use]
    pub fn route_post_vm(&self, vm_out: [f32; 2], dry_tap: f32, send_out: [f32; 2]) -> [f32; 4] {
        let out_l_in = scrub_and_flush_f32_bits(vm_out[0]);
        let out_r_in = scrub_and_flush_f32_bits(vm_out[1]);
        let (main_l, main_r) = if self.routing_mode == IoRoutingMode::Mono {
            let mid = 0.5 * (out_l_in + out_r_in);
            (mid, mid)
        } else {
            Self::apply_ms_width(out_l_in, out_r_in, self.ms_width)
        };
        let port_out1 = if self.wdw_enabled || self.routing_mode == IoRoutingMode::WetDryWet {
            scrub_and_flush_f32_bits(dry_tap)
        } else {
            send_out[1]
        };
        [send_out[0], port_out1, main_l, main_r]
    }

    /// Maps `(vm_out, vm_core_lr, vm_wet_lr, direct_dry_tap, send_out)` into canonical physical
    /// output ports `[Port Out 0 (Send L), Port Out 1 (Send R / WDW Dry), Port Out 2 (Main L), Port Out 3 (Main R)]`,
    /// splitting pre-spatial core Mid `m_c` onto `Port Out 1` and `m_w ± alpha * s_w` onto
    /// `Ports Out 2/3` when [`WdwDryTapMode::PreSpatialCore`] is active (`ref[dsp.router.wdw-contract]`).
    #[inline(always)]
    #[must_use]
    pub fn route_post_vm_wdw(
        &self,
        vm_out: [f32; 2],
        vm_core_lr: [f32; 2],
        vm_wet_lr: [f32; 2],
        direct_dry_tap: f32,
        send_out: [f32; 2],
    ) -> [f32; 4] {
        if (self.wdw_enabled || self.routing_mode == IoRoutingMode::WetDryWet)
            && self.wdw_mode == WdwDryTapMode::PreSpatialCore
        {
            let core_mid = 0.5
                * (scrub_and_flush_f32_bits(vm_core_lr[0])
                    + scrub_and_flush_f32_bits(vm_core_lr[1]));
            let wet_l_clean = scrub_and_flush_f32_bits(vm_wet_lr[0]);
            let wet_r_clean = scrub_and_flush_f32_bits(vm_wet_lr[1]);
            let wet_ms = [
                0.5 * (wet_l_clean + wet_r_clean),
                0.5 * (wet_l_clean - wet_r_clean),
            ];
            let [dry_center, wet_l, wet_r] = eval_wdw_routing_outputs(
                WdwDryTapMode::PreSpatialCore,
                core_mid,
                wet_ms,
                self.ms_width,
                [0.0; 2],
            );
            return [send_out[0], dry_center, wet_l, wet_r];
        }
        self.route_post_vm(vm_out, direct_dry_tap, send_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// verify[dsp.router.ports]
    /// verify[dsp.router.return-to-send-isolation]
    #[test]
    fn test_patchbay_8port_modes_ms_width_and_structural_isolation() {
        let mut patchbay = HostShellPatchbay::new();
        assert_eq!(patchbay.ms_width, 1.0);
        assert_eq!(patchbay.pdc_host_samples(), 0);
        assert_eq!(
            HostShellPatchbay::voltage_normalize_ms(0.75, 0.75),
            (0.75, 0.0)
        );
        assert_eq!(
            HostShellPatchbay::voltage_reconstruct_ms(0.75, 0.0),
            (0.75, 0.75)
        );

        // 4x4 setter structurally discards S_{ret->send}.
        patchbay.set_scattering_matrix_4x4([[1.0; 4]; 4]);
        let (vm1, send1, _) = patchbay.route_pre_vm([0.5, -0.25, 0.0, 0.0]);
        let (vm2, send2, _) = patchbay.route_pre_vm([0.5, -0.25, 10.0, 2.0]);
        assert_eq!(send1, send2);
        assert_ne!(vm1, vm2);

        // 4CM isolates inst->down and routes ret->down while driving Send L only.
        patchbay.set_routing_mode(IoRoutingMode::FourCableMethod);
        let (vm_4cm, send_4cm, _) = patchbay.route_pre_vm([0.6, -0.4, 0.2, -0.3]);
        assert_eq!(send_4cm, [0.6, 0.0]);
        assert_eq!(vm_4cm, [0.2, -0.3]);

        // DualAmp drives stereo Send L/R and receives FX Return L/R into VM.
        patchbay.set_routing_mode(IoRoutingMode::DualAmp);
        let (vm_dual, send_dual, _) = patchbay.route_pre_vm([0.6, -0.4, 0.2, -0.3]);
        assert_eq!(send_dual, [0.6, -0.4]);
        assert_eq!(vm_dual, [0.2, -0.3]);

        // Mono normaling duplicates Port In 0 to R and folds Port Out 2/3 to Mid.
        patchbay.set_routing_mode(IoRoutingMode::Mono);
        let (vm_mono, send_mono, dry_mono) = patchbay.route_pre_vm([0.5, -0.9, 0.0, 0.0]);
        assert_eq!(vm_mono, [0.5, 0.5]);
        assert_eq!(send_mono, [0.5, 0.5]);
        assert_eq!(dry_mono, 0.5);
        let out_mono = patchbay.route_post_vm([0.8, 0.2], dry_mono, send_mono);
        assert_eq!(out_mono[2], 0.5);
        assert_eq!(out_mono[3], 0.5);

        // M/S width clamp and 0-ULP unity pass-through.
        patchbay.set_routing_mode(IoRoutingMode::Stereo);
        patchbay.set_ms_width(5.0);
        assert_eq!(patchbay.ms_width, 2.0);
        patchbay.set_ms_width(f32::NAN);
        assert_eq!(patchbay.ms_width, 1.0);
        assert_eq!(
            HostShellPatchbay::apply_ms_width(0.6, -0.2, 1.0),
            (0.6, -0.2)
        );
        let (wl, wr) = HostShellPatchbay::apply_ms_width(0.75, -0.25, 0.0);
        assert_eq!((wl, wr), (0.25, 0.25));

        // Non-finite scattering fallback sanitizes cleanly.
        patchbay.set_scattering_matrices(
            [[f32::NAN, 0.0], [0.0, f32::INFINITY]],
            [[f32::NAN, 0.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, f32::NAN]],
        );
        assert_eq!(patchbay.s_inst_to_send, HostShellPatchbay::IDENTITY_2X2);
        assert_eq!(patchbay.s_inst_to_down, HostShellPatchbay::ZERO_2X2);
        assert_eq!(patchbay.s_ret_to_down, [[0.5, 0.0], [0.0, 1.0]]);
    }

    /// verify[dsp.router.wdw-contract]
    /// verify[dsp.router.parallel-alignment]
    #[test]
    fn test_patchbay_wdw_contract_and_parallel_alignment_zoh() {
        let [dry_core, wet_l, wet_r] = eval_wdw_routing_outputs(
            WdwDryTapMode::PreSpatialCore,
            0.4,
            [0.2, 0.1],
            1.5,
            [0.8, 0.2],
        );
        assert!((dry_core - 0.4).abs() < 1.0e-6);
        assert!((wet_l - 0.35).abs() < 1.0e-6);
        assert!((wet_r - 0.05).abs() < 1.0e-6);

        let [dry_dir, _, _] =
            eval_wdw_routing_outputs(WdwDryTapMode::DirectDry, 0.4, [0.2, 0.1], 1.5, [0.8, 0.2]);
        assert!((dry_dir - 0.5).abs() < 1.0e-6);

        // Full WDW Port Out [Send L, Dry Center, Wet L, Wet R] contract + width independence on dry tap.
        let mut patchbay = HostShellPatchbay::new();
        patchbay.set_routing_mode(IoRoutingMode::WetDryWet);
        patchbay.set_wdw_mode(WdwDryTapMode::DirectDry);
        patchbay.set_wdw_enabled(true);
        patchbay.set_ms_width(1.75);
        patchbay.configure_return_alignment(0.0, 0.0);

        let (_vm_in, send_out, dry_tap) = patchbay.route_pre_vm([0.8, 0.2, 0.0, 0.0]);
        assert!((dry_tap - 0.5).abs() < 1.0e-6);
        let ports = patchbay.route_post_vm([0.4, 0.0], dry_tap, send_out);
        // Port Out 0 = Send L, Port Out 1 = WDW Dry Center, Port Out 2/3 = m_w +- 1.75 * s_w.
        assert_eq!(ports[0], 0.8);
        assert!((ports[1] - 0.5).abs() < 1.0e-6);
        assert!((ports[2] - 0.55).abs() < 1.0e-6);
        assert!((ports[3] - (-0.15)).abs() < 1.0e-6);

        // PreSpatialCore WDW contract via route_post_vm_wdw:
        // y_dry == m_c, y_wet_L == m_w + alpha * s_w, y_wet_R == m_w - alpha * s_w.
        patchbay.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
        let vm_core_lr = [0.6, 0.2]; // m_c = 0.4, s_c = 0.2
        let vm_wet_lr = [0.3, 0.1]; // m_w = 0.2, s_w = 0.1
        let vm_total = [vm_core_lr[0] + vm_wet_lr[0], vm_core_lr[1] + vm_wet_lr[1]];
        let core_ports =
            patchbay.route_post_vm_wdw(vm_total, vm_core_lr, vm_wet_lr, dry_tap, send_out);
        assert_eq!(core_ports[0], 0.8);
        assert!((core_ports[1] - 0.4).abs() < 1.0e-6);
        assert!((core_ports[2] - (0.2 + 1.75 * 0.1)).abs() < 1.0e-6);
        assert!((core_ports[3] - (0.2 - 1.75 * 0.1)).abs() < 1.0e-6);

        // Switching to DirectDry falls back cleanly to direct_dry_tap + total vm_out.
        patchbay.set_wdw_mode(WdwDryTapMode::DirectDry);
        let dir_ports =
            patchbay.route_post_vm_wdw(vm_total, vm_core_lr, vm_wet_lr, dry_tap, send_out);
        assert!((dir_ports[1] - 0.5).abs() < 1.0e-6);
        let (exp_l, exp_r) = HostShellPatchbay::apply_ms_width(vm_total[0], vm_total[1], 1.75);
        assert!((dir_ports[2] - exp_l).abs() < 1.0e-6);
        assert!((dir_ports[3] - exp_r).abs() < 1.0e-6);

        // Parallel alignment 4x offline bounce (43 host PDC) + 48k-anchored 1x host return alignment ZOH.
        patchbay.set_alignment_mode(ParallelAlignmentMode::FullBand64Tap);
        patchbay.configure_parallel_alignment(ParallelAlignmentMode::FullBand64Tap, true, 0.0, 0.0);
        patchbay.configure_return_alignment_48k(8.0, 8.0);
        patchbay.reconfigure_sample_rate(192_000.0, true);
        assert_eq!(patchbay.pdc_host_samples(), 43);
        assert!((patchbay.return_align_l.delay_samples() - 8.0).abs() < 1.0e-5);

        patchbay.reconfigure_sample_rate(88_200.0, false);
        assert_eq!(patchbay.pdc_host_samples(), 31);
        patchbay.configure_parallel_alignment(
            ParallelAlignmentMode::LowLatency16Tap,
            true,
            1.5,
            1.5,
        );
        assert!(patchbay.parallel_align_l.kaiser_active());
        assert_eq!(patchbay.pdc_host_samples(), 7);
        patchbay.reset();

        // Non-finite sample rate leaves existing alignment setup untouched.
        patchbay.reconfigure_sample_rate(0.0, false);
        assert_eq!(patchbay.pdc_host_samples(), 7);
        patchbay.set_parallel_alignment_enabled(false);
        assert_eq!(patchbay.pdc_host_samples(), 0);
    }

    #[test]
    fn test_patchbay_direct_dry_and_parallel_align_4x_bounce_dirac_pdc() {
        let mut pb = HostShellPatchbay::new();
        pb.set_routing_mode(IoRoutingMode::WetDryWet);
        pb.set_wdw_mode(WdwDryTapMode::DirectDry);
        pb.reconfigure_sample_rate(192_000.0, true);
        assert_eq!(pb.pdc_host_samples(), OFFLINE_BOUNCE_PDC_SAMPLES);

        let mut unaligned = [0.0_f32; 32];
        for (i, slot) in unaligned.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let (_, _, dry_tap) = pb.route_pre_vm([x, x, 0.0, 0.0]);
            *slot = dry_tap;
        }
        assert_eq!(unaligned[12], 1.0);
        assert!(
            unaligned
                .iter()
                .enumerate()
                .all(|(i, &v)| (i == 12) || v == 0.0)
        );

        pb.reset();
        pb.configure_parallel_alignment(ParallelAlignmentMode::FullBand64Tap, true, 0.0, 0.0);
        assert_eq!(pb.pdc_host_samples(), 43);
        let mut aligned = [0.0_f32; 64];
        for (i, slot) in aligned.iter_mut().enumerate() {
            let x = if i == 0 { 1.0 } else { 0.0 };
            let (_, _, dry_tap) = pb.route_pre_vm([x, x, 0.0, 0.0]);
            *slot = dry_tap;
        }
        assert_eq!(aligned[43], 1.0);
        assert!(
            aligned
                .iter()
                .enumerate()
                .all(|(i, &v)| (i == 43) || v == 0.0)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// verify[dsp.router.ports]
        /// verify[dsp.router.return-to-send-isolation]
        /// verify[dsp.router.wdw-contract]
        #[test]
        fn prop_patchbay_return_isolation_ms_roundtrip_and_wdw(
            in_l in -10.0_f32..10.0_f32,
            in_r in -10.0_f32..10.0_f32,
            ret_l in -10.0_f32..10.0_f32,
            ret_r in -10.0_f32..10.0_f32,
            s_leak in -4.0_f32..4.0_f32,
            width in -1.0_f32..3.0_f32,
            delay_ret in 0.0_f32..40.0_f32,
        ) {
            let (mid, side) = HostShellPatchbay::voltage_normalize_ms(in_l, in_r);
            let (rl, rr) = HostShellPatchbay::voltage_reconstruct_ms(mid, side);
            prop_assert!((rl - in_l).abs() < 1.0e-5);
            prop_assert!((rr - in_r).abs() < 1.0e-5);

            // Structural S_{ret->send} == 0 isolation across arbitrary 4x4 matrices and delays.
            let mut pb_zero = HostShellPatchbay::new();
            let mut pb_ret = HostShellPatchbay::new();
            let mat = [
                [0.85, 0.15, s_leak, -s_leak],
                [-0.25, 0.75, s_leak, s_leak],
                [0.4, 0.6, 0.5, 0.5],
                [0.2, 0.3, 0.7, -0.3],
            ];
            pb_zero.set_scattering_matrix_4x4(mat);
            pb_ret.set_scattering_matrix_4x4(mat);
            pb_zero.configure_return_alignment(delay_ret, delay_ret);
            pb_ret.configure_return_alignment(delay_ret, delay_ret);

            let (_, send_a, dry_a) = pb_zero.route_pre_vm([in_l, in_r, 0.0, 0.0]);
            let (_, send_b, dry_b) = pb_ret.route_pre_vm([in_l, in_r, ret_l, ret_r]);
            prop_assert_eq!(send_a[0].to_bits(), send_b[0].to_bits());
            prop_assert_eq!(send_a[1].to_bits(), send_b[1].to_bits());
            prop_assert_eq!(dry_a.to_bits(), dry_b.to_bits());

            // Post-VM M/S width leaves Mid invariant and scales Side strictly by clamped width.
            pb_ret.set_wdw_enabled(true);
            pb_ret.set_ms_width(width);
            let out_w = pb_ret.route_post_vm([in_l, in_r], dry_b, send_b);
            let clamped_w = width.clamp(0.0, 2.0);
            let (out_mid, out_side) = HostShellPatchbay::voltage_normalize_ms(out_w[2], out_w[3]);
            prop_assert_eq!(out_w[1].to_bits(), dry_b.to_bits());
            prop_assert!((out_mid - mid).abs() < 1.0e-5);
            prop_assert!((out_side - clamped_w * side).abs() < 1.0e-4);
        }
    }
}
