//! Cabinet shared-pole stereo Kautz modal bank (`176 B` [`KautzStereoBank4`]), `44.1–384 kHz`
//! continuous-time ZOH hydration with exact prefix-override replacement, and `3 kHz` contractive
//! slew retuning (`ref[dsp.op-m.kautz.tier-allocation]`, `ref[dsp.op-m.kautz.dual-state]`,
//! `ref[dsp.op-m.kautz.radial-gate]`, `ref[dsp.op-m.kautz.rate-convert]`).

use core::mem::{align_of, size_of};

use dnam_kernels::{KautzQuadState, algebraic_unit_rotor_f32, step_kautz_quad_stereo8};

use crate::bin::arena_wire::KautzSoaBlockWire;
use crate::math::{
    MathBackend, SimdVec4, clamp_finite_f32, const_abs_f32, const_abs_f64, exp2_signed_oct_f32,
    powf_f64, rcp_nr6_f64, rsqrt_nr3_vec4, sanitize_finite_f32, sqrt_f64,
};
use crate::types::VmError;

/// Maximum number of `4`-pair stereo Kautz banks (`Ultra`: `8` banks = `32` pole pairs).
pub const MAX_KAUTZ_BANKS: usize = 8;

/// Maximum number of shared conjugate Kautz pole pairs (`Ultra`: `32`).
pub const MAX_KAUTZ_PAIRS: usize = MAX_KAUTZ_BANKS * 4;

const KAUTZ_R_MIN: f32 = 0.25;
const KAUTZ_R_MAX: f32 = 0.9995;
const KAUTZ_R_ZOH_PRE_MAX: f32 = 0.9985;
const KAUTZ_THETA_ZOH_MIN: f32 = 1.0e-4;
const KAUTZ_THETA_MAX: f32 = core::f32::consts::PI - 1.0e-4;
const KAUTZ_THETA_ZOH_PRE_MAX: f32 = (core::f64::consts::PI - 1.0e-3) as f32;
const KAUTZ_OCT_DAMP_LIMIT: f32 = 1.5;
const KAUTZ_SLEW_EPS: f32 = 1.0e-7;
const KAUTZ_THETA_ZERO_EPS: f32 = 1.0e-6;
const KAUTZ_SWEEP_CEIL: f32 = 0.9950;
const KAUTZ_SWEEP_SLOPE: f32 = 0.15;
const KAUTZ_OMR2_MIN_F32: f32 = 1.0e-7;
const KAUTZ_OMR2_MIN_F64: f64 = 1.0e-7;
const RHO_UNITY_EPS_F64: f64 = 1.0e-12;

const _: () = assert!(KAUTZ_R_MIN > 0.0 && KAUTZ_R_MIN < KAUTZ_R_ZOH_PRE_MAX);
const _: () = assert!(KAUTZ_R_ZOH_PRE_MAX < KAUTZ_R_MAX);
const _: () = assert!(KAUTZ_THETA_ZOH_MIN > 0.0 && KAUTZ_THETA_MAX < core::f32::consts::PI);
const _: () = assert!(KAUTZ_THETA_ZOH_PRE_MAX < KAUTZ_THETA_MAX);
const _: () = assert!(KAUTZ_SWEEP_CEIL <= KAUTZ_R_MAX);

/// Runtime `16`-byte-aligned `4`-pair stereo shared-pole Kautz bank (`176 B`, `11 × [f32; 4]`,
/// `ref[dsp.op-m.kautz.tier-allocation]`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KautzStereoBank4 {
    /// Left channel modal real state across `4` conjugate pairs.
    pub s_re_l: [f32; 4],
    /// Left channel modal imaginary state across `4` conjugate pairs.
    pub s_im_l: [f32; 4],
    /// Right channel modal real state across `4` conjugate pairs.
    pub s_re_r: [f32; 4],
    /// Right channel modal imaginary state across `4` conjugate pairs.
    pub s_im_r: [f32; 4],
    /// Shared complex pole real part `r_eff * cos(theta_new)`.
    pub lambda_re: [f32; 4],
    /// Shared complex pole imaginary part `r_eff * sin(theta_new)`.
    pub lambda_im: [f32; 4],
    /// Effective normalized modal excitation gains `b_gamma_eff`.
    pub b_gamma: [f32; 4],
    /// Left channel complex modal readout real component.
    pub c_re_l: [f32; 4],
    /// Left channel complex modal readout imaginary component.
    pub c_im_l: [f32; 4],
    /// Right channel complex modal readout real component.
    pub c_re_r: [f32; 4],
    /// Right channel complex modal readout imaginary component.
    pub c_im_r: [f32; 4],
}

const _: () = assert!(size_of::<KautzStereoBank4>() == 176);
const _: () = assert!(align_of::<KautzStereoBank4>() == 16);

impl KautzStereoBank4 {
    /// Zeroes both Left and Right stereo modal states while preserving coefficients.
    #[inline]
    pub fn reset(&mut self) {
        self.s_re_l = [0.0; 4];
        self.s_im_l = [0.0; 4];
        self.s_re_r = [0.0; 4];
        self.s_im_r = [0.0; 4];
    }

    /// Loads this `176 B` bank into register-resident [`KautzQuadState`] vectors.
    #[inline(always)]
    pub fn to_kernel_state<B: MathBackend>(&self) -> KautzQuadState<B::Vec4> {
        KautzQuadState {
            z_re_l: B::Vec4::load(&self.s_re_l),
            z_im_l: B::Vec4::load(&self.s_im_l),
            z_re_r: B::Vec4::load(&self.s_re_r),
            z_im_r: B::Vec4::load(&self.s_im_r),
            alpha_re: B::Vec4::load(&self.lambda_re),
            alpha_im: B::Vec4::load(&self.lambda_im),
            b_re: B::Vec4::load(&self.b_gamma),
            c_re_l: B::Vec4::load(&self.c_re_l),
            c_im_l: B::Vec4::load(&self.c_im_l),
            c_re_r: B::Vec4::load(&self.c_re_r),
            c_im_r: B::Vec4::load(&self.c_im_r),
        }
    }

    /// Stores updated stereo modal state vectors from [`KautzQuadState`] back to memory.
    #[inline(always)]
    pub fn store_kernel_state<B: MathBackend>(&mut self, state: &KautzQuadState<B::Vec4>) {
        state.z_re_l.store(&mut self.s_re_l);
        state.z_im_l.store(&mut self.s_im_l);
        state.z_re_r.store(&mut self.s_re_r);
        state.z_im_r.store(&mut self.s_im_r);
    }

    /// Steps one stereo frame through this `4`-pair Kautz bank via `8`-lane stereo SIMD.
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(&mut self, x_l: f32, x_r: f32) -> (f32, f32) {
        let mut ks = self.to_kernel_state::<B>();
        let out = step_kautz_quad_stereo8::<B::Stereo8>(&mut ks, x_l, x_r);
        self.store_kernel_state::<B>(&ks);
        out
    }

    /// Steps one stereo frame across a slice of active Kautz banks with `0-ULP` parity against
    /// [`Self::step_chunk`].
    #[inline]
    pub fn step_banks_stereo<B: MathBackend>(banks: &mut [Self], x_l: f32, x_r: f32) -> (f32, f32) {
        let mut sum_l = 0.0_f32;
        let mut sum_r = 0.0_f32;
        for bank in banks.iter_mut() {
            let (yl, yr) = bank.step_stereo::<B>(x_l, x_r);
            sum_l += yl;
            sum_r += yr;
        }
        (sum_l, sum_r)
    }

    /// Hoists each bank's [`KautzQuadState`] into SIMD registers across all `C` samples of a
    /// sub-block, producing exact `0-ULP` equivalence with repeated [`Self::step_banks_stereo`]
    /// (`ref[dsp.op-m.recurrence.stereo-simd]`).
    #[inline]
    pub fn step_chunk<B: MathBackend, const C: usize>(
        banks: &mut [Self],
        inputs_lr: &[[f32; 2]; C],
        out_lr: &mut [[f32; 2]; C],
    ) {
        out_lr.fill([0.0, 0.0]);
        for bank in banks.iter_mut() {
            let mut ks = bank.to_kernel_state::<B>();
            for (n, inp) in inputs_lr.iter().enumerate() {
                let (yl, yr) = step_kautz_quad_stereo8::<B::Stereo8>(&mut ks, inp[0], inp[1]);
                out_lr[n][0] += yl;
                out_lr[n][1] += yr;
            }
            bank.store_kernel_state::<B>(&ks);
        }
    }
}

/// Cold-cached host-rate Kautz poles, `H2` gains, and deterministic `3 kHz` slew state
/// (`ref[dsp.op-m.kautz.radial-gate]`, `ref[dsp.op-m.kautz.rate-convert]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KautzColdSlewState {
    /// Rate-adapted host poles `(r_host, th_host)` in `[0.25, 0.9995] × [0, pi - 1e-4]`.
    pub host_poles: [(f32, f32); MAX_KAUTZ_PAIRS],
    /// Cold-normalized excitation gains (`0.0` verbatim for inactive poles).
    pub host_b_gamma: [f32; MAX_KAUTZ_PAIRS],
    /// Precomputed `1.0 / max(1.0 - r_host^2, 1e-7)` for division-free `3 kHz` orthonormal rescale.
    pub host_inv_1mr2: [f32; MAX_KAUTZ_PAIRS],
    /// Previous `3 kHz` `oct` control slot value.
    pub last_oct: f32,
    /// Previous `3 kHz` `damp` control slot value.
    pub last_damp: f32,
    /// Previous `oct` used for single-tick sweep contraction detection.
    pub prev_oct: f32,
    /// Active single-tick sweep contraction state flag.
    pub slew_active: bool,
}

impl Default for KautzColdSlewState {
    #[inline]
    fn default() -> Self {
        Self {
            host_poles: [(KAUTZ_R_MIN, 0.0); MAX_KAUTZ_PAIRS],
            host_b_gamma: [0.0; MAX_KAUTZ_PAIRS],
            host_inv_1mr2: [1.0; MAX_KAUTZ_PAIRS],
            last_oct: 0.0,
            last_damp: 0.0,
            prev_oct: 0.0,
            slew_active: false,
        }
    }
}

impl KautzColdSlewState {
    /// Clears `3 kHz` slew tracking history while preserving cached `host_poles` and `host_b_gamma`.
    #[inline]
    pub fn reset_slew(&mut self) {
        self.last_oct = 0.0;
        self.last_damp = 0.0;
        self.prev_oct = 0.0;
        self.slew_active = false;
    }
}

/// Optional exact-replacement Kautz prefix override slices unpacked from tier `VarProReadout`
/// blocks (`ref[dsp.op-m.kautz.dual-state]`).
#[derive(Clone, Copy, Debug, Default)]
pub struct KautzPrefixOverride<'a> {
    /// Exact replacement Left real residues `c_re_l[0..len]`.
    pub c_re_l: &'a [f32],
    /// Exact replacement Left imaginary residues `c_im_l[0..len]`.
    pub c_im_l: &'a [f32],
    /// Exact replacement Right real residues `c_re_r[0..len]`.
    pub c_re_r: &'a [f32],
    /// Exact replacement Right imaginary residues `c_im_r[0..len]`.
    pub c_im_r: &'a [f32],
    /// Exact replacement modal excitation gains `b_gamma[0..len]`.
    pub b_gamma: &'a [f32],
}

#[inline]
fn cold_b_gamma_norm(r: f32) -> f32 {
    let r64 = f64::from(r);
    let omr2 = (1.0 - r64 * r64).max(KAUTZ_OMR2_MIN_F64);
    sqrt_f64(omr2) as f32
}

#[inline]
fn hydrate_single_pole(
    r_base_48k: f32,
    th_base_48k: f32,
    b_gamma_stored: f32,
    rho_vm: f64,
    is_48k: bool,
) -> (f32, f32, f32, f32) {
    let inactive = r_base_48k <= 0.0 || !b_gamma_stored.is_finite() || b_gamma_stored <= 0.0;
    let (r_host, th_host) = if is_48k {
        (
            r_base_48k.clamp(KAUTZ_R_MIN, KAUTZ_R_MAX),
            th_base_48k.clamp(0.0, KAUTZ_THETA_MAX),
        )
    } else {
        let r_pre = r_base_48k.clamp(KAUTZ_R_MIN, KAUTZ_R_ZOH_PRE_MAX);
        let th_pre = th_base_48k.clamp(0.0, KAUTZ_THETA_ZOH_PRE_MAX);
        let r_zoh = (powf_f64(f64::from(r_pre), rho_vm) as f32).clamp(KAUTZ_R_MIN, KAUTZ_R_MAX);
        let th_zoh =
            ((rho_vm * f64::from(th_pre)) as f32).clamp(KAUTZ_THETA_ZOH_MIN, KAUTZ_THETA_MAX);
        (r_zoh, th_zoh)
    };

    let ratio = cold_b_gamma_norm(r_host)
        * (rcp_nr6_f64(f64::from(cold_b_gamma_norm(
            r_base_48k.clamp(KAUTZ_R_MIN, KAUTZ_R_MAX),
        ))) as f32);
    let host_b = if inactive {
        0.0
    } else if is_48k {
        b_gamma_stored
    } else {
        b_gamma_stored * ratio
    };
    let r64 = f64::from(r_host);
    let inv_1mr2 = rcp_nr6_f64((1.0 - r64 * r64).max(KAUTZ_OMR2_MIN_F64)) as f32;
    (r_host, th_host, host_b, inv_1mr2)
}

#[inline]
fn resolve_pole_residues(
    blk: &KautzSoaBlockWire,
    lane: usize,
    global_m: usize,
    prefix_override: Option<&KautzPrefixOverride<'_>>,
) -> (f32, [f32; 4]) {
    let ov_bg = prefix_override.and_then(|ov| ov.b_gamma.get(global_m).copied());
    let b_raw = ov_bg.unwrap_or_else(|| blk.b_gamma[lane].get());
    let b_stored = if b_raw.is_finite() && b_raw > 0.0 {
        b_raw
    } else {
        0.0
    };
    let ov_c = prefix_override.and_then(|ov| {
        Some([
            *ov.c_re_l.get(global_m)?,
            *ov.c_im_l.get(global_m)?,
            *ov.c_re_r.get(global_m)?,
            *ov.c_im_r.get(global_m)?,
        ])
    });
    let c_raw = ov_c.unwrap_or_else(|| {
        [
            blk.c_re_l[lane].get(),
            blk.c_im_l[lane].get(),
            blk.c_re_r[lane].get(),
            blk.c_im_r[lane].get(),
        ]
    });
    (b_stored, c_raw.map(sanitize_finite_f32))
}

/// Hydrates serialized [`KautzSoaBlockWire`] blocks into runtime [`KautzStereoBank4`] banks
/// and [`KautzColdSlewState`] with continuous-time ZOH rescaling and optional exact-replacement
/// prefix overrides (`ref[dsp.op-m.kautz.rate-convert]`, `ref[dsp.op-m.kautz.dual-state]`).
pub fn hydrate_kautz_slice_and_zoh(
    blocks: &[KautzSoaBlockWire],
    banks: &mut [KautzStereoBank4],
    slew: &mut KautzColdSlewState,
    rho_vm: f64,
    prefix_override: Option<&KautzPrefixOverride<'_>>,
) -> Result<usize, VmError> {
    if blocks.len() > MAX_KAUTZ_BANKS || banks.len() < blocks.len() {
        return Err(VmError::InsufficientBackingCapacity);
    }
    if !rho_vm.is_finite() || rho_vm <= 0.0 {
        return Err(VmError::InvalidSampleRate);
    }
    let is_48k = const_abs_f64(rho_vm - 1.0) < RHO_UNITY_EPS_F64;

    for (blk_idx, blk) in blocks.iter().enumerate() {
        let bank = &mut banks[blk_idx];
        for lane in 0..4 {
            let m = blk_idx * 4 + lane;
            let r_raw = blk.r_low[lane].get() + 0.5 * blk.dr[lane].get();
            let r_base_48k = clamp_finite_f32(r_raw, 0.0, KAUTZ_R_MAX, 0.0);
            let th_base_48k = sanitize_finite_f32(blk.theta[lane].get());
            let (b_stored, [c_re_l, c_im_l, c_re_r, c_im_r]) =
                resolve_pole_residues(blk, lane, m, prefix_override);

            let (r_host, th_host, host_b, inv_1mr2) =
                hydrate_single_pole(r_base_48k, th_base_48k, b_stored, rho_vm, is_48k);
            slew.host_poles[m] = (r_host, th_host);
            slew.host_b_gamma[m] = host_b;
            slew.host_inv_1mr2[m] = inv_1mr2;

            let (c_th, s_th) = algebraic_unit_rotor_f32(th_host);
            bank.lambda_re[lane] = r_host * c_th;
            bank.lambda_im[lane] = r_host * s_th;
            bank.b_gamma[lane] = host_b;
            bank.c_re_l[lane] = c_re_l;
            bank.c_im_l[lane] = c_im_l;
            bank.c_re_r[lane] = c_re_r;
            bank.c_im_r[lane] = c_im_r;
        }
    }

    let used_pairs = blocks.len() * 4;
    for m in used_pairs..MAX_KAUTZ_PAIRS {
        slew.host_poles[m] = (KAUTZ_R_MIN, 0.0);
        slew.host_b_gamma[m] = 0.0;
        slew.host_inv_1mr2[m] = 1.0;
    }
    slew.reset_slew();
    Ok(blocks.len())
}

#[inline]
fn restore_cached_kautz_poles(
    banks: &mut [KautzStereoBank4],
    active_banks: usize,
    slew: &KautzColdSlewState,
) {
    for (blk_idx, bank) in banks.iter_mut().take(active_banks).enumerate() {
        for lane in 0..4 {
            let m = blk_idx * 4 + lane;
            let (r_host, th_host) = slew.host_poles[m];
            let (c_th, s_th) = algebraic_unit_rotor_f32(th_host);
            bank.lambda_re[lane] = r_host * c_th;
            bank.lambda_im[lane] = r_host * s_th;
            bank.b_gamma[lane] = slew.host_b_gamma[m];
        }
    }
}

#[inline]
fn update_retuned_kautz_poles<B: MathBackend>(
    banks: &mut [KautzStereoBank4],
    active_banks: usize,
    slew: &KautzColdSlewState,
    oct: f32,
    damp: f32,
    prev_oct_used: f32,
) {
    let freq_scale = exp2_signed_oct_f32(oct);
    let prev_scale = exp2_signed_oct_f32(prev_oct_used);
    let damp_scale = exp2_signed_oct_f32(damp);

    for (blk_idx, bank) in banks.iter_mut().take(active_banks).enumerate() {
        let mut r_eff_arr = [0.0_f32; 4];
        let mut inv_1mr2_arr = [0.0_f32; 4];
        let mut host_bg_arr = [0.0_f32; 4];

        for lane in 0..4 {
            let m = blk_idx * 4 + lane;
            let (r_host, th_host) = slew.host_poles[m];
            let (th_new, dtheta) = if th_host <= KAUTZ_THETA_ZERO_EPS {
                (0.0, 0.0)
            } else {
                let tn = (th_host * freq_scale).clamp(KAUTZ_THETA_ZOH_MIN, KAUTZ_THETA_MAX);
                let tp = (th_host * prev_scale).clamp(KAUTZ_THETA_ZOH_MIN, KAUTZ_THETA_MAX);
                (tn, const_abs_f32(tn - tp))
            };

            let r_damped = (1.0 - (1.0 - r_host) * damp_scale).clamp(KAUTZ_R_MIN, KAUTZ_R_MAX);
            let r_eff = if dtheta > KAUTZ_THETA_ZERO_EPS {
                r_damped.min((KAUTZ_SWEEP_CEIL - KAUTZ_SWEEP_SLOPE * dtheta).max(KAUTZ_R_MIN))
            } else {
                r_damped
            };

            let (c_th, s_th) = algebraic_unit_rotor_f32(th_new);
            bank.lambda_re[lane] = r_eff * c_th;
            bank.lambda_im[lane] = r_eff * s_th;
            r_eff_arr[lane] = r_eff;
            inv_1mr2_arr[lane] = slew.host_inv_1mr2[m];
            host_bg_arr[lane] = slew.host_b_gamma[m];
        }

        let v_r = B::Vec4::load(&r_eff_arr);
        let v_omr2 = B::Vec4::splat(1.0)
            .fms(v_r, v_r)
            .max(B::Vec4::splat(KAUTZ_OMR2_MIN_F32));
        let v_ratio_sq = v_omr2.mul(B::Vec4::load(&inv_1mr2_arr));
        let v_ratio = v_ratio_sq.mul(rsqrt_nr3_vec4::<B>(v_ratio_sq));
        B::Vec4::load(&host_bg_arr)
            .mul(v_ratio)
            .store(&mut bank.b_gamma);
    }
}

/// Retunes active shared-pole Kautz banks at `3 kHz` boundaries (`slots 10..=11`) with exact
/// slew state tracking, `0.9950` rapid-sweep contraction, and `3`-step `rsqrt` orthonormal gain
/// rescaling (`ref[dsp.op-m.kautz.radial-gate]`, `ref[dsp.op-m.rotor.angle-wrap-and-retune]`).
pub fn retune_kautz_banks_3khz<B: MathBackend>(
    banks: &mut [KautzStereoBank4],
    num_banks: usize,
    slew: &mut KautzColdSlewState,
    oct: f32,
    damp: f32,
    immediate_snap: bool,
) {
    let active_banks = num_banks.min(banks.len()).min(MAX_KAUTZ_BANKS);
    let oct_c = clamp_finite_f32(oct, -KAUTZ_OCT_DAMP_LIMIT, KAUTZ_OCT_DAMP_LIMIT, 0.0);
    let damp_c = clamp_finite_f32(damp, -KAUTZ_OCT_DAMP_LIMIT, KAUTZ_OCT_DAMP_LIMIT, 0.0);

    let prev_oct_used = if immediate_snap {
        slew.slew_active = false;
        slew.last_oct = oct_c;
        slew.last_damp = damp_c;
        slew.prev_oct = oct_c;
        oct_c
    } else {
        let oct_changed = const_abs_f32(oct_c - slew.last_oct) > KAUTZ_SLEW_EPS;
        let damp_changed = const_abs_f32(damp_c - slew.last_damp) > KAUTZ_SLEW_EPS;
        let settling = slew.slew_active && const_abs_f32(oct_c - slew.prev_oct) > KAUTZ_SLEW_EPS;
        if !oct_changed && !damp_changed && !settling && !slew.slew_active {
            return;
        }
        let prev_used = slew.prev_oct;
        slew.slew_active = oct_changed || const_abs_f32(oct_c - slew.prev_oct) > KAUTZ_SLEW_EPS;
        slew.last_oct = oct_c;
        slew.last_damp = damp_c;
        slew.prev_oct = oct_c;
        prev_used
    };

    if const_abs_f32(oct_c) <= KAUTZ_SLEW_EPS
        && const_abs_f32(damp_c) <= KAUTZ_SLEW_EPS
        && const_abs_f32(oct_c - prev_oct_used) <= KAUTZ_SLEW_EPS
    {
        restore_cached_kautz_poles(banks, active_banks, slew);
    } else {
        update_retuned_kautz_poles::<B>(banks, active_banks, slew, oct_c, damp_c, prev_oct_used);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};
    use zerocopy::byteorder::little_endian::F32;

    fn make_wire_block(
        r_low: [f32; 4],
        dr: [f32; 4],
        theta: [f32; 4],
        bg: [f32; 4],
    ) -> KautzSoaBlockWire {
        KautzSoaBlockWire {
            r_low: r_low.map(F32::new),
            dr: dr.map(F32::new),
            theta: theta.map(F32::new),
            b_gamma: bg.map(F32::new),
            c_re_l: [0.5, -0.3, 0.2, 0.1].map(F32::new),
            c_im_l: [0.1, 0.05, -0.05, 0.02].map(F32::new),
            c_re_r: [0.4, 0.25, -0.1, 0.3].map(F32::new),
            c_im_r: [-0.2, 0.1, 0.05, -0.15].map(F32::new),
        }
    }

    #[test]
    fn test_kautz_hydration_inactive_sentinel_prefix_override_and_chunk_parity() {
        let blocks = [
            make_wire_block(
                [0.90, 0.0, 0.95, 0.9995],
                [0.04, 0.0, 0.02, 0.0],
                [0.12, 0.40, 0.80, 1.50],
                [0.30, 0.50, -0.10, f32::NAN],
            ),
            make_wire_block(
                [0.82, 0.88, 0.92, 0.94],
                [0.02, 0.04, 0.02, 0.02],
                [0.20, 0.45, 0.70, 1.10],
                [0.25, 0.20, 0.15, 0.10],
            ),
        ];
        let mut banks = [KautzStereoBank4::default(); 2];
        let mut slew = KautzColdSlewState::default();
        let count = hydrate_kautz_slice_and_zoh(&blocks, &mut banks, &mut slew, 0.5, None).unwrap();
        assert_eq!(count, 2);

        // Inactive slots (lane 1 r_base=0, lane 2 b_gamma<=0, lane 3 NaN) keep host_b_gamma==0.0 verbatim.
        assert!(slew.host_b_gamma[0] > 0.0);
        assert_eq!(slew.host_b_gamma[1], 0.0);
        assert_eq!(slew.host_b_gamma[2], 0.0);
        assert_eq!(slew.host_b_gamma[3], 0.0);
        assert!(slew.host_poles[1].0 >= KAUTZ_R_MIN && slew.host_poles[1].0 <= KAUTZ_R_MAX);
        // ZOH 96 kHz halves carrier angle and increases radius via powf(r_pre, 0.5).
        assert!(const_abs_f32(slew.host_poles[0].1 - 0.06) < 1.0e-6);

        // Prefix override replaces [0..2] residues and b_gamma exactly.
        let ov = KautzPrefixOverride {
            c_re_l: &[0.75, 0.1],
            c_im_l: &[0.0, 0.0],
            c_re_r: &[-0.75, 0.1],
            c_im_r: &[0.0, 0.0],
            b_gamma: &[0.45, 0.0],
        };
        let mut banks_ov = [KautzStereoBank4::default(); 2];
        let mut slew_ov = KautzColdSlewState::default();
        hydrate_kautz_slice_and_zoh(&blocks, &mut banks_ov, &mut slew_ov, 1.0, Some(&ov)).unwrap();
        assert_eq!(banks_ov[0].c_re_l[0], 0.75);
        assert_eq!(banks_ov[0].c_re_r[0], -0.75);
        assert_eq!(banks_ov[0].b_gamma[0], 0.45);
        assert_eq!(slew_ov.host_b_gamma[0], 0.45);

        // Zero-ULP parity between step_banks_stereo (C=1 x 16) and step_chunk::<_, 16>.
        let inputs: [[f32; 2]; 16] =
            core::array::from_fn(|i| [((i as f32) * 0.17) - 0.5, 0.4 - ((i as f32) * 0.11)]);
        let mut b_seq = banks_ov;
        let mut b_chk = banks_ov;
        let mut out_seq = [[0.0_f32; 2]; 16];
        for i in 0..16 {
            let (l, r) = KautzStereoBank4::step_banks_stereo::<ScalarBackend>(
                &mut b_seq,
                inputs[i][0],
                inputs[i][1],
            );
            out_seq[i] = [l, r];
        }
        let mut out_chk = [[0.0_f32; 2]; 16];
        KautzStereoBank4::step_chunk::<ScalarBackend, 16>(&mut b_chk, &inputs, &mut out_chk);
        for i in 0..16 {
            assert_eq!(out_seq[i][0].to_bits(), out_chk[i][0].to_bits());
            assert_eq!(out_seq[i][1].to_bits(), out_chk[i][1].to_bits());
        }
        assert_eq!(b_seq, b_chk);
        b_seq[0].reset();
        assert_eq!(b_seq[0].s_re_l, [0.0; 4]);
    }

    #[test]
    fn test_kautz_3khz_slew_contraction_settle_restore_and_snap() {
        let blocks = [make_wire_block(
            [0.999, 0.0, 0.95, 0.98],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.25, 0.0],
            [0.2, 0.3, 0.4, 0.5],
        )];
        let mut banks = [KautzStereoBank4::default()];
        let mut slew = KautzColdSlewState::default();
        hydrate_kautz_slice_and_zoh(&blocks, &mut banks, &mut slew, 1.0, None).unwrap();
        let baseline_bank = banks[0];

        // Steady (0, 0) tick does zero work.
        retune_kautz_banks_3khz::<DefaultMathBackend>(&mut banks, 1, &mut slew, 0.0, 0.0, false);
        assert_eq!(banks[0], baseline_bank);

        // Step jump in oct triggers 1-tick sweep contraction <= 0.9950 on lane 0 while keeping
        // inactive lane 1 at 0.0 and zero-frequency lane 3 uncontracted.
        retune_kautz_banks_3khz::<DefaultMathBackend>(&mut banks, 1, &mut slew, 0.5, 0.0, false);
        assert!(slew.slew_active);
        let r_contracted = sqrt_f64(
            f64::from(banks[0].lambda_re[0]) * f64::from(banks[0].lambda_re[0])
                + f64::from(banks[0].lambda_im[0]) * f64::from(banks[0].lambda_im[0]),
        );
        assert!(
            r_contracted <= 0.9950 + 1.0e-5,
            "r_contracted={r_contracted}"
        );
        assert_eq!(banks[0].b_gamma[1], 0.0);
        assert_eq!(banks[0].lambda_im[3], 0.0);

        // Holding oct=0.5 settles slew_active=false and restores full uncontracted radius.
        retune_kautz_banks_3khz::<DefaultMathBackend>(&mut banks, 1, &mut slew, 0.5, 0.0, false);
        assert!(!slew.slew_active);
        let r_settled = sqrt_f64(
            f64::from(banks[0].lambda_re[0]) * f64::from(banks[0].lambda_re[0])
                + f64::from(banks[0].lambda_im[0]) * f64::from(banks[0].lambda_im[0]),
        );
        assert!(r_settled > r_contracted);

        // Immediate snap to (0, 0) restores exact cached host_poles and host_b_gamma without contraction.
        retune_kautz_banks_3khz::<DefaultMathBackend>(&mut banks, 1, &mut slew, 0.0, 0.0, true);
        assert!(!slew.slew_active);
        assert_eq!(
            banks[0].lambda_re.map(f32::to_bits),
            baseline_bank.lambda_re.map(f32::to_bits)
        );
        assert_eq!(
            banks[0].lambda_im.map(f32::to_bits),
            baseline_bank.lambda_im.map(f32::to_bits)
        );
        assert_eq!(
            banks[0].b_gamma.map(f32::to_bits),
            baseline_bank.b_gamma.map(f32::to_bits)
        );
    }
}
