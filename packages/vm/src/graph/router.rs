//! 8-Port I/O Matrix Router (`IoRouter`), hysteretic noise gate, voltage-normalized Mid/Side
//! transforms, return alignment, and `200 Hz` sidechain `LR4` envelope detector bank
//! (`0 FDIV`, `0 FSQRT`, `0 libm`, `!alloc`, `ref[dsp.router.return-to-send-isolation]`,
//! `ref[dsp.router.mid-side-input]`, `ref[dsp.router.lr4-crossover]`).

#![forbid(unsafe_code)]

pub mod align;
pub mod detector;

pub use align::{IoAlignmentMode, ParallelAlignLine};
pub use detector::{
    DEFAULT_SIDECHAIN_LR4_HZ, IoSidechainDetectorBank, eval_stereo_linked_aux_detector,
};

use zerocopy::FromZeros;

use crate::bin::{CrossbarWire, SONA_META_ROUTER_EXT};
use crate::control::bus::laws::eval_router_gate_thresh;
use crate::graph::zoh::ZohRateContext;
use crate::math::{
    clamp_finite_f32, const_abs_f32, const_abs_f64, flush_recursive_state_f32, fp16_to_f32,
    powf_f64, sanitize_finite_f32,
};

const GATE_DEFAULT_ATK_BETA: f32 = 0.05;
const GATE_DEFAULT_REL_BETA: f32 = 0.002;
const GATE_MIN_BETA: f32 = 1.0e-5;
const GATE_MAX_ATK_BETA: f32 = 0.5;
const GATE_MAX_REL_BETA: f32 = 0.2;
const GATE_MAX_THRESH: f32 = 0.5;
const GATE_MAX_HYST: f32 = 0.25;
const GATE_MAX_INV_KNEE: f32 = 2_000.0;

/// Wet-Dry-Wet (`WDW`) center dry tap selector on `Port Out 1` (`ref[dsp.router.wdw-contract]`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WdwDryTapMode {
    /// Center dry on `Port Out 1` is the non-recirculating core + cabinet Mid contribution.
    PreSpatialCore = 0,
    /// Center dry on `Port Out 1` is the voltage-normalized direct input/return Mid contribution.
    DirectDry = 1,
}

#[inline]
fn rescale_gate_beta_zoh(base_beta: f32, rho_vm: f64, max_beta: f32) -> f32 {
    let clamped = base_beta.clamp(GATE_MIN_BETA, max_beta);
    if !rho_vm.is_finite() || rho_vm <= 0.0 || const_abs_f64(rho_vm - 1.0) <= 1.0e-12 {
        return clamped;
    }
    let one_minus = 1.0 - f64::from(clamped);
    ((1.0 - powf_f64(one_minus, rho_vm)) as f32).clamp(GATE_MIN_BETA, max_beta)
}

/// ZOH-adapted base parameters for the hysteretic router noise gate
/// (`ref[dsp.router.lr4-crossover]`, `ref[container.dnam.load-validation]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouterGateConfig {
    /// Base gate open threshold in `[0.0, 0.5]`.
    pub gate_thresh: f32,
    /// Hysteresis release width in `[0.0, 0.25]`.
    pub gate_hyst: f32,
    /// Inverse knee slope in `[0.0, 2000.0]`.
    pub gate_inv_knee: f32,
    /// Minimum linear attenuation floor in `[0.0, 1.0]`.
    pub gate_floor: f32,
    /// Rate-adapted gate attack step beta in `[1e-5, 0.5]`.
    pub gate_atk_beta: f32,
    /// Rate-adapted gate release step beta in `[1e-5, 0.2]`.
    pub gate_rel_beta: f32,
}

impl Default for RouterGateConfig {
    fn default() -> Self {
        Self::pass_through(1.0)
    }
}

impl RouterGateConfig {
    /// Returns the pass-through default gate configuration (`SONA_META_ROUTER_EXT == 0`) with
    /// ZOH-adapted attack/release betas for `rho_vm`.
    pub fn pass_through(rho_vm: f64) -> Self {
        Self {
            gate_thresh: 0.0,
            gate_hyst: 0.0,
            gate_inv_knee: 0.0,
            gate_floor: 1.0,
            gate_atk_beta: rescale_gate_beta_zoh(GATE_DEFAULT_ATK_BETA, rho_vm, GATE_MAX_ATK_BETA),
            gate_rel_beta: rescale_gate_beta_zoh(GATE_DEFAULT_REL_BETA, rho_vm, GATE_MAX_REL_BETA),
        }
    }

    /// Hydrates gate parameters from [`CrossbarWire`] honoring `SONA_META_ROUTER_EXT` (`bit 17`).
    pub fn from_crossbar_wire(cb: &CrossbarWire, rho_vm: f64) -> Self {
        if (cb.sona_meta_flags.get() & SONA_META_ROUTER_EXT) == 0 {
            return Self::pass_through(rho_vm);
        }
        let g0 = fp16_to_f32(cb.gate_and_crossbar_fp16[0].get());
        let g1 = fp16_to_f32(cb.gate_and_crossbar_fp16[1].get());
        let g2 = fp16_to_f32(cb.gate_and_crossbar_fp16[2].get());
        let g3 = fp16_to_f32(cb.gate_and_crossbar_fp16[3].get());
        let g4 = fp16_to_f32(cb.gate_and_crossbar_fp16[4].get());
        let g5 = fp16_to_f32(cb.gate_and_crossbar_fp16[5].get());

        let base_atk =
            clamp_finite_f32(g4, GATE_MIN_BETA, GATE_MAX_ATK_BETA, GATE_DEFAULT_ATK_BETA);
        let base_rel =
            clamp_finite_f32(g5, GATE_MIN_BETA, GATE_MAX_REL_BETA, GATE_DEFAULT_REL_BETA);
        Self {
            gate_thresh: clamp_finite_f32(g0, 0.0, GATE_MAX_THRESH, 0.0),
            gate_hyst: clamp_finite_f32(g1, 0.0, GATE_MAX_HYST, 0.0),
            gate_inv_knee: clamp_finite_f32(g2, 0.0, GATE_MAX_INV_KNEE, 0.0),
            gate_floor: clamp_finite_f32(g3, 0.0, 1.0, 0.0),
            gate_atk_beta: rescale_gate_beta_zoh(base_atk, rho_vm, GATE_MAX_ATK_BETA),
            gate_rel_beta: rescale_gate_beta_zoh(base_rel, rho_vm, GATE_MAX_REL_BETA),
        }
    }
}

/// Evaluates voltage-normalized Mid/Side encoding `(0.5 * (l + r), 0.5 * (l - r))`
/// (`ref[dsp.router.mid-side-input]`).
#[inline(always)]
pub fn voltage_ms_encode(l: f32, r: f32) -> (f32, f32) {
    (0.5 * (l + r), 0.5 * (l - r))
}

/// Evaluates voltage-normalized Mid/Side width decoding `(mid + width * side, mid - width * side)`
/// (`ref[dsp.head.mid-side-factorization]`).
#[inline(always)]
pub fn voltage_ms_decode(mid: f32, side: f32, width: f32) -> (f32, f32) {
    let s = width * side;
    (mid + s, mid - s)
}

fn decode_router_gain_and_mix(cb: &CrossbarWire) -> ([f32; 2], [[f32; 2]; 2]) {
    if (cb.sona_meta_flags.get() & SONA_META_ROUTER_EXT) == 0 {
        return ([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]]);
    }
    let gain = [
        clamp_finite_f32(cb.router_gain[0].get(), 0.0, 4.0, 1.0),
        clamp_finite_f32(cb.router_gain[1].get(), 0.0, 4.0, 1.0),
    ];
    let mut mix = [[0.0_f32; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            let fallback = if r == c { 1.0 } else { 0.0 };
            mix[r][c] = clamp_finite_f32(cb.router_mix_matrix[r][c].get(), -2.0, 2.0, fallback);
        }
    }
    (gain, mix)
}

fn decode_s_router(cb: &CrossbarWire) -> [[f32; 4]; 4] {
    let mut s = [[0.0_f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let fallback = if r == c { 1.0 } else { 0.0 };
            s[r][c] = clamp_finite_f32(cb.s_router[r][c].get(), -2.0, 2.0, fallback);
        }
    }
    enforce_return_to_send_isolation(&mut s);
    s
}

#[inline(always)]
fn enforce_return_to_send_isolation(s_router: &mut [[f32; 4]; 4]) {
    s_router[0][2] = 0.0;
    s_router[0][3] = 0.0;
    s_router[1][2] = 0.0;
    s_router[1][3] = 0.0;
}

/// 8-Port I/O Matrix Router (`2`-port stereo + `4`-port `4CM`/`WDW`/sidechain execution),
/// enforcing `S_{ret->send} == 0` and storing hot gate state externally in L1D `SonaLiteHotPrefix`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IoRouter {
    /// Causal `4x4` scattering matrix (`S_{ret->send} == 0` enforced unconditionally).
    pub s_router: [[f32; 4]; 4],
    /// Input trims `[L, R]` (`[0.0, 4.0]`).
    pub router_gain: [f32; 2],
    /// `2x2` channel/Mid-Side input mix matrix (`[-2.0, 2.0]`).
    pub router_mix_matrix: [[f32; 2]; 2],
    /// ZOH-adapted hysteretic noise gate configuration.
    pub gate_base: RouterGateConfig,
    /// `200 Hz` `TPT-LR4` sidechain crossover and 4-band ballistic envelope detector bank.
    pub detector: IoSidechainDetectorBank,
    /// Sub-sample return alignment FIR line for `FX Return L`.
    pub return_aligner_l: ParallelAlignLine,
    /// Sub-sample return alignment FIR line for `FX Return R`.
    pub return_aligner_r: ParallelAlignLine,
    /// `WDW` center dry tap selector (`PreSpatialCore` or `DirectDry`).
    pub wdw_mode: WdwDryTapMode,
    /// Whether `WDW` 4-port center-dry / wet-only routing is enabled on `Port Out 1..3`.
    pub wdw_active: bool,
    return_delay_samples: [f64; 2],
}

impl Default for IoRouter {
    #[inline]
    fn default() -> Self {
        Self::from_crossbar_wire(&CrossbarWire::new_zeroed(), &ZohRateContext::default())
    }
}

impl IoRouter {
    /// Hydrates an [`IoRouter`] from [`CrossbarWire`] and [`ZohRateContext`].
    pub fn from_crossbar_wire(cb: &CrossbarWire, zoh: &ZohRateContext) -> Self {
        let s_router = decode_s_router(cb);
        let (router_gain, router_mix_matrix) = decode_router_gain_and_mix(cb);
        let gate_base = RouterGateConfig::from_crossbar_wire(cb, zoh.rho_vm);
        let detector = IoSidechainDetectorBank::new(zoh.f_s_vm, DEFAULT_SIDECHAIN_LR4_HZ);
        let m_os = if zoh.is_bounce_4x { 4 } else { 1 };
        let return_aligner_l = ParallelAlignLine::new(IoAlignmentMode::FullBand64Tap, 0.0, m_os);
        let return_aligner_r = ParallelAlignLine::new(IoAlignmentMode::FullBand64Tap, 0.0, m_os);
        Self {
            s_router,
            router_gain,
            router_mix_matrix,
            gate_base,
            detector,
            return_aligner_l,
            return_aligner_r,
            wdw_mode: WdwDryTapMode::PreSpatialCore,
            wdw_active: false,
            return_delay_samples: [0.0, 0.0],
        }
    }

    /// Recomputes ZOH-adapted gate betas, sidechain `LR4`/ballistics, and return aligners for `zoh`.
    pub fn reconfigure_sample_rate(&mut self, cb: &CrossbarWire, zoh: &ZohRateContext) {
        self.gate_base = RouterGateConfig::from_crossbar_wire(cb, zoh.rho_vm);
        self.detector
            .reconfigure(zoh.f_s_vm, DEFAULT_SIDECHAIN_LR4_HZ);
        let m_os = if zoh.is_bounce_4x { 4 } else { 1 };
        let mode_l = self.return_aligner_l.mode();
        let mode_r = self.return_aligner_r.mode();
        self.return_aligner_l
            .reconfigure(mode_l, self.return_delay_samples[0], m_os);
        self.return_aligner_r
            .reconfigure(mode_r, self.return_delay_samples[1], m_os);
    }

    /// Configures return alignment mode and fractional sample offsets for L and R return paths.
    pub fn configure_return_alignment(
        &mut self,
        mode: IoAlignmentMode,
        delay_l: f64,
        delay_r: f64,
        m_os: usize,
    ) {
        let dl = if delay_l.is_finite() {
            delay_l.max(0.0)
        } else {
            0.0
        };
        let dr = if delay_r.is_finite() {
            delay_r.max(0.0)
        } else {
            0.0
        };
        self.return_delay_samples = [dl, dr];
        self.return_aligner_l.reconfigure(mode, dl, m_os);
        self.return_aligner_r.reconfigure(mode, dr, m_os);
    }

    /// Resets detector, return aligners, and external hot gate state `(gate_env, gate_gain) = (0.0, 1.0)`.
    #[inline]
    pub fn reset(&mut self, gate_env: &mut f32, gate_gain: &mut f32) {
        self.detector.reset();
        self.return_aligner_l.reset();
        self.return_aligner_r.reset();
        *gate_env = 0.0;
        *gate_gain = 1.0;
    }

    /// Updates `s_router` while unconditionally enforcing `S_{ret->send} == 0`
    /// (`ref[dsp.router.return-to-send-isolation]`).
    #[inline]
    pub fn set_s_router(&mut self, mut s_router: [[f32; 4]; 4]) {
        for row in &mut s_router {
            for cell in row {
                *cell = sanitize_finite_f32(*cell);
            }
        }
        enforce_return_to_send_isolation(&mut s_router);
        self.s_router = s_router;
    }

    #[inline(always)]
    fn apply_preprocess_and_gate(
        &self,
        inst_l: f32,
        inst_r: f32,
        slot30_gate_thresh_delta: f32,
        gate_env: &mut f32,
        gate_gain: &mut f32,
    ) -> (f32, f32) {
        let xl = self.router_gain[0] * inst_l;
        let xr = self.router_gain[1] * inst_r;
        let ml = self.router_mix_matrix[0][0] * xl + self.router_mix_matrix[0][1] * xr;
        let mr = self.router_mix_matrix[1][0] * xl + self.router_mix_matrix[1][1] * xr;

        let peak = 0.5 * (const_abs_f32(xl) + const_abs_f32(xr));
        let beta = if peak > *gate_env {
            self.gate_base.gate_atk_beta
        } else {
            self.gate_base.gate_rel_beta
        };
        *gate_env = flush_recursive_state_f32(*gate_env + beta * (peak - *gate_env));
        let thr = eval_router_gate_thresh(
            self.gate_base.gate_thresh,
            slot30_gate_thresh_delta,
            GATE_MAX_THRESH,
        ) - self.gate_base.gate_hyst * (*gate_gain);
        let g_target = ((*gate_env - thr) * self.gate_base.gate_inv_knee
            + self.gate_base.gate_floor)
            .clamp(0.0, 1.0);
        *gate_gain = flush_recursive_state_f32(*gate_gain + beta * (g_target - *gate_gain));
        (*gate_gain * ml, *gate_gain * mr)
    }

    /// Executes 2-port stereo routing (`detector -> router_gain -> mix_matrix -> hysteretic gate`),
    /// returning `(gated_l, gated_r)` (`ref[dsp.router.lr4-crossover]`).
    #[inline]
    pub fn step_2port(
        &mut self,
        inst_l: f32,
        inst_r: f32,
        slot30_gate_thresh_delta: f32,
        gate_env: &mut f32,
        gate_gain: &mut f32,
    ) -> (f32, f32) {
        self.detector.step(inst_l, inst_r, 0.0, 0.0);
        self.apply_preprocess_and_gate(
            inst_l,
            inst_r,
            slot30_gate_thresh_delta,
            gate_env,
            gate_gain,
        )
    }

    /// Executes 4-port (`4CM`/`WDW`/sidechain) routing with aligned returns, returning
    /// `(down_lr, send_lr)` where `down_lr` applies `s_router[2..4][0..4]` over gated inputs and
    /// aligned returns, and `send_lr` strictly obeys `S_{ret->send} == 0` (`ref[dsp.router.return-to-send-isolation]`).
    #[inline]
    pub fn step_4port(
        &mut self,
        inst_lr: [f32; 2],
        ret_lr: [f32; 2],
        has_return: bool,
        slot30_gate_thresh_delta: f32,
        gate_env: &mut f32,
        gate_gain: &mut f32,
    ) -> ([f32; 2], [f32; 2]) {
        let (down_lr, send_lr, _) = self.step_4port_internal(
            inst_lr,
            ret_lr,
            has_return,
            slot30_gate_thresh_delta,
            gate_env,
            gate_gain,
        );
        (down_lr, send_lr)
    }

    /// Evaluates voltage-normalized Mid/Side encoding `(0.5 * (l + r), 0.5 * (l - r))`.
    #[inline(always)]
    pub fn voltage_ms_encode(l: f32, r: f32) -> (f32, f32) {
        voltage_ms_encode(l, r)
    }

    /// Evaluates voltage-normalized Mid/Side width decoding `(mid + width * side, mid - width * side)`.
    #[inline(always)]
    pub fn voltage_ms_decode(mid: f32, side: f32, width: f32) -> (f32, f32) {
        voltage_ms_decode(mid, side, width)
    }

    #[inline]
    pub(crate) fn step_4port_internal(
        &mut self,
        inst_lr: [f32; 2],
        ret_lr: [f32; 2],
        has_return: bool,
        slot30_gate_thresh_delta: f32,
        gate_env: &mut f32,
        gate_gain: &mut f32,
    ) -> ([f32; 2], [f32; 2], [f32; 2]) {
        let [inst_l, inst_r] = inst_lr;
        let (ret_l_aligned, ret_r_aligned) = if has_return {
            (
                self.return_aligner_l.step(ret_lr[0]),
                self.return_aligner_r.step(ret_lr[1]),
            )
        } else {
            (0.0, 0.0)
        };
        self.detector
            .step(inst_l, inst_r, ret_l_aligned, ret_r_aligned);
        let (gl, gr) = self.apply_preprocess_and_gate(
            inst_l,
            inst_r,
            slot30_gate_thresh_delta,
            gate_env,
            gate_gain,
        );
        let send_lr = [
            self.s_router[0][0] * gl + self.s_router[0][1] * gr,
            self.s_router[1][0] * gl + self.s_router[1][1] * gr,
        ];
        let down_lr = [
            self.s_router[2][0] * gl
                + self.s_router[2][1] * gr
                + self.s_router[2][2] * ret_l_aligned
                + self.s_router[2][3] * ret_r_aligned,
            self.s_router[3][0] * gl
                + self.s_router[3][1] * gr
                + self.s_router[3][2] * ret_l_aligned
                + self.s_router[3][3] * ret_r_aligned,
        ];
        (down_lr, send_lr, [ret_l_aligned, ret_r_aligned])
    }
}

/// Evaluates `WDW` output ports `(port_out1_dry, port_out2_wet_l, port_out3_wet_r)`
/// (`ref[dsp.router.wdw-contract]`).
#[inline]
pub fn eval_wdw_routing_outputs(
    mode: WdwDryTapMode,
    core_mid: f32,
    wet_mid: f32,
    wet_side: f32,
    width_alpha: f32,
    direct_l: f32,
    direct_r: f32,
) -> (f32, f32, f32) {
    let wet_l = wet_mid + width_alpha * wet_side;
    let wet_r = wet_mid - width_alpha * wet_side;
    let dry = match mode {
        WdwDryTapMode::PreSpatialCore => core_mid,
        WdwDryTapMode::DirectDry => 0.5 * (direct_l + direct_r),
    };
    (dry, wet_l, wet_r)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    use super::*;
    use crate::math::f32_to_fp16;
    use crate::types::Tier;

    #[test]
    fn test_router_bit17_passthrough_and_4port_isolation() {
        let mut cb = CrossbarWire::new_zeroed();
        for i in 0..4 {
            cb.s_router[i][i] = F32::new(1.0);
        }
        cb.router_gain = [F32::new(2.0), F32::new(0.5)];
        cb.gate_and_crossbar_fp16[3] = U16::new(f32_to_fp16(0.25));
        let zoh = ZohRateContext::compute(48_000.0, 48_000.0, Tier::Lite, false, false).unwrap();

        let mut router = IoRouter::from_crossbar_wire(&cb, &zoh);
        assert_eq!(router.router_gain.map(f32::to_bits), [1.0_f32.to_bits(); 2]);
        assert_eq!(
            router.router_mix_matrix.map(|row| row.map(f32::to_bits)),
            [[1.0_f32.to_bits(), 0], [0, 1.0_f32.to_bits()]]
        );
        assert_eq!(router.gate_base.gate_floor.to_bits(), 1.0_f32.to_bits());

        let (mut gate_env, mut gate_gain) = (0.0_f32, 1.0_f32);
        router.reset(&mut gate_env, &mut gate_gain);
        let (y_l, y_r) = router.step_2port(0.625, -0.375, 0.0, &mut gate_env, &mut gate_gain);
        assert_eq!(y_l.to_bits(), 0.625_f32.to_bits());
        assert_eq!(y_r.to_bits(), (-0.375_f32).to_bits());

        cb.sona_meta_flags = U32::new(SONA_META_ROUTER_EXT);
        cb.gate_and_crossbar_fp16[0] = U16::new(f32_to_fp16(0.4));
        cb.gate_and_crossbar_fp16[1] = U16::new(f32_to_fp16(0.1));
        cb.gate_and_crossbar_fp16[2] = U16::new(f32_to_fp16(20.0));
        cb.gate_and_crossbar_fp16[3] = U16::new(f32_to_fp16(0.25));
        cb.gate_and_crossbar_fp16[4] = U16::new(f32_to_fp16(0.1));
        cb.gate_and_crossbar_fp16[5] = U16::new(f32_to_fp16(0.01));
        let zoh_96 = ZohRateContext::compute(48_000.0, 96_000.0, Tier::Full, true, false).unwrap();
        let mut ext_router = IoRouter::from_crossbar_wire(&cb, &zoh_96);
        assert_eq!(
            ext_router.router_gain.map(f32::to_bits),
            [2.0_f32.to_bits(), 0.5_f32.to_bits()]
        );
        assert_eq!(
            ext_router.gate_base.gate_floor.to_bits(),
            0.25_f32.to_bits()
        );

        ext_router.set_s_router([
            [0.8, 0.1, 0.9, 0.9],
            [0.1, 0.8, 0.9, 0.9],
            [0.5, 0.25, 0.5, 0.25],
            [0.25, 0.5, 0.25, 0.5],
        ]);
        assert_eq!(ext_router.s_router[0][2].to_bits(), 0);
        assert_eq!(ext_router.s_router[0][3].to_bits(), 0);
        assert_eq!(ext_router.s_router[1][2].to_bits(), 0);
        assert_eq!(ext_router.s_router[1][3].to_bits(), 0);

        ext_router.configure_return_alignment(IoAlignmentMode::MinLatency4Tap, 0.0, 0.0, 1);
        ext_router.reconfigure_sample_rate(&cb, &zoh);
        let (down_lr, send_lr) = ext_router.step_4port(
            [0.5, -0.25],
            [0.4, 0.2],
            true,
            0.0,
            &mut gate_env,
            &mut gate_gain,
        );
        assert!(down_lr[0].is_finite() && down_lr[1].is_finite());
        assert!(send_lr[0].is_finite() && send_lr[1].is_finite());

        let (m, s) = IoRouter::voltage_ms_encode(0.75, 0.75);
        assert_eq!(m.to_bits(), 0.75_f32.to_bits());
        assert_eq!(s.to_bits(), 0);
        let (dl, dr) = IoRouter::voltage_ms_decode(m, s, 1.5);
        assert_eq!(dl.to_bits(), 0.75_f32.to_bits());
        assert_eq!(dr.to_bits(), 0.75_f32.to_bits());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_4port_send_strictly_independent_of_returns_and_ms_parity(
            inst_l in -4.0_f32..=4.0_f32,
            inst_r in -4.0_f32..=4.0_f32,
            ret_l in -4.0_f32..=4.0_f32,
            ret_r in -4.0_f32..=4.0_f32,
            leak in -2.0_f32..=2.0_f32,
        ) {
            let mut cb = CrossbarWire::new_zeroed();
            cb.s_router[0] = [F32::new(0.8), F32::new(0.2), F32::new(leak), F32::new(leak)];
            cb.s_router[1] = [F32::new(-0.1), F32::new(0.9), F32::new(leak), F32::new(leak)];
            cb.s_router[2] = [F32::new(0.5), F32::new(0.0), F32::new(0.5), F32::new(0.0)];
            cb.s_router[3] = [F32::new(0.0), F32::new(0.5), F32::new(0.0), F32::new(0.5)];
            let zoh = ZohRateContext::compute(48_000.0, 48_000.0, Tier::Lite, false, false).unwrap();

            let mut r_a = IoRouter::from_crossbar_wire(&cb, &zoh);
            let mut r_b = IoRouter::from_crossbar_wire(&cb, &zoh);
            let (mut env_a, mut gain_a) = (0.0_f32, 1.0_f32);
            let (mut env_b, mut gain_b) = (0.0_f32, 1.0_f32);
            let (_, send_a) = r_a.step_4port(
                [inst_l, inst_r],
                [0.0, 0.0],
                true,
                0.0,
                &mut env_a,
                &mut gain_a,
            );
            let (_, send_b) = r_b.step_4port(
                [inst_l, inst_r],
                [ret_l, ret_r],
                true,
                0.0,
                &mut env_b,
                &mut gain_b,
            );
            prop_assert_eq!(send_a.map(f32::to_bits), send_b.map(f32::to_bits));

            let (mid, side) = IoRouter::voltage_ms_encode(inst_l, inst_r);
            let (dec_l, dec_r) = IoRouter::voltage_ms_decode(mid, side, 1.0);
            prop_assert!((dec_l - inst_l).abs() < 1.0e-5);
            prop_assert!((dec_r - inst_r).abs() < 1.0e-5);
        }
    }
}
