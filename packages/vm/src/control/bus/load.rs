//! Cold-path `.dnam-bin` loader constructors for [`super::ControlBus`].

use core::marker::PhantomData;

use super::dispatch::{
    ANTI_ZIPPER_SNAP_EPS, DEFAULT_ANTI_ZIPPER_POLE, DEFAULT_DRIFT_PRNG_SEED, STAGE_TARGET_SLOTS,
};
use super::tick::compute_has_internal_modulators;
use super::{ControlBus, ControlBusBacking};
use crate::bin::{
    Arena0ControlBusWire, Arena3ControlBusWire, CP1_STREAM_FLAG_HAS_ULTRA, CP1_STREAM_MAGIC,
    Cp1StreamHeaderWire, DnamBinView, TempoSyncEntryWire,
};
use crate::control::cp1::{
    Cp1ContributorSlot, count_distinct_contributor_controls, decode_cp1_contributors,
};
use crate::control::exact_int::{ExactIntFactorAccumulator, MAX_EXACT_CONTRIBUTORS};
use crate::control::{ControlClampDescriptor, ControlTsynBacking};
use crate::math::{MathBackend, Q0_192Phase, clamp_finite_f32, fp16_to_f32, sanitize_finite_f32};
use crate::types::{Tier, VmError};
use zerocopy::{FromBytes, FromZeros};

/// Base index of `ou_to_z_stage_ultra_fp16[4]` inside `stage_latent_ctrl_ultra_fp16[256]`.
const OU_TO_Z_STAGE_ULTRA_OFFSET: usize = 160;

#[inline]
fn canonical_zero(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { v }
}

/// Decodes `lfo_skew[0..2]` from `arena0.control_bus_lite.lfo_skew_fp16`, clamping finite values
/// to `[-1.0, 1.0]` with `0.0` fallback (`ref[container.dnam.load-validation]`).
#[inline]
pub(crate) fn decode_lfo_skew(a0_cb: &Arena0ControlBusWire) -> [f32; 2] {
    [0, 1].map(|j| {
        canonical_zero(clamp_finite_f32(
            fp16_to_f32(a0_cb.lfo_skew_fp16[j].get()),
            -1.0,
            1.0,
            0.0,
        ))
    })
}

/// Decodes `base_tank_dry_mix` from `tank_dry_mix_fp16`: `0x0000 -> 0.0`, finite -> `[-2.0, 2.0]`,
/// non-finite -> `1.0` (`ref[container.dnam.load-validation]`).
#[inline]
pub(crate) fn decode_tank_dry_mix(raw_fp16: u16) -> f32 {
    if raw_fp16 == 0 {
        0.0
    } else {
        canonical_zero(clamp_finite_f32(fp16_to_f32(raw_fp16), -2.0, 2.0, 1.0))
    }
}

/// Decodes `ou_to_z_stage[0..8]` across Arena 0 `[0..4]` and optional Arena 3 `[4..8]`
/// (`stage_latent_ctrl_ultra_fp16[160..164]`), clamping finite entries into `[-4.0, 4.0]`.
#[inline]
pub(crate) fn decode_ou_to_z_stage(
    a0_cb: &Arena0ControlBusWire,
    a3_cb: Option<&Arena3ControlBusWire>,
) -> [f32; 8] {
    let mut out = [0.0_f32; 8];
    for k in 0..4 {
        out[k] = canonical_zero(clamp_finite_f32(
            fp16_to_f32(a0_cb.ou_to_z_stage_fp16[k].get()),
            -4.0,
            4.0,
            0.0,
        ));
    }
    if let Some(a3) = a3_cb {
        for k in 0..4 {
            out[4 + k] = canonical_zero(clamp_finite_f32(
                fp16_to_f32(a3.stage_latent_ctrl_ultra_fp16[OU_TO_Z_STAGE_ULTRA_OFFSET + k].get()),
                -4.0,
                4.0,
                0.0,
            ));
        }
    }
    out
}

pub(crate) fn required_cp1_contributors(view: &DnamBinView<'_>) -> Result<usize, VmError> {
    if !view.cp1_overflow.is_empty() {
        let (hdr, _) = Cp1StreamHeaderWire::read_from_prefix(view.cp1_overflow)
            .map_err(|_| VmError::InvalidCp1Stream)?;
        if hdr.magic != CP1_STREAM_MAGIC
            || (hdr.flags & !CP1_STREAM_FLAG_HAS_ULTRA) != 0
            || hdr.reserved != 0
        {
            return Err(VmError::InvalidCp1Stream);
        }
        let count = usize::from(hdr.count.get());
        if count > MAX_EXACT_CONTRIBUTORS {
            return Err(VmError::InvalidCp1Stream);
        }
        Ok(count)
    } else {
        let count = usize::from(view.arena0.control_bus_lite.cp1_dir.cp1_dir_count.get());
        if count > 8 || view.arena0.control_bus_lite.cp1_dir.flags.get() != 0 {
            return Err(VmError::InvalidCp1Stream);
        }
        Ok(count)
    }
}

struct DecodedControlWire {
    u_int_base: [[f32; 8]; 4],
    u_int_ultra: [[[f32; 8]; 4]; 3],
    w_ctrl: [[f32; 16]; 16],
    w_ctrl_ultra: [[f32; 4]; 16],
    cext_present: bool,
    w_ctrl_ext: [[f32; 16]; 16],
    w_ctrl_ultra_ext: [[f32; 4]; 16],
    lfo_dphi_train: [f32; 2],
    sigma_drift: f32,
    rho_drift_train: f32,
    lfo_skew: [f32; 2],
    base_tank_dry_mix: f32,
    ou_to_z_stage: [f32; 8],
    trust_radius_mh: f32,
    anti_zipper_scale: f32,
    ref_sample_rate_hz: f32,
    has_internal_modulators: bool,
}

fn decode_control_wire_params(view: &DnamBinView<'_>, tier: Tier) -> DecodedControlWire {
    let a0_cb = &view.arena0.control_bus_lite;
    let a3_cb = if tier == Tier::Ultra {
        view.arena3.map(|a3| &a3.control_bus_ultra_delta)
    } else {
        None
    };

    let mut u_int_base = [[0.0_f32; 8]; 4];
    for f in 0..4 {
        for d in 0..8 {
            u_int_base[f][d] = sanitize_finite_f32(a0_cb.u_int_base[f][d].get());
        }
    }
    let mut u_int_ultra = [[[0.0_f32; 8]; 4]; 3];
    if let Some(a3) = a3_cb {
        for b in 0..3 {
            for f in 0..4 {
                for d in 0..8 {
                    u_int_ultra[b][f][d] = sanitize_finite_f32(a3.u_int_ultra[b][f][d].get());
                }
            }
        }
    }
    let mut w_ctrl = [[0.0_f32; 16]; 16];
    for r in 0..16 {
        for c in 0..16 {
            w_ctrl[r][c] = sanitize_finite_f32(a0_cb.w_ctrl[r][c].get());
        }
    }
    let mut w_ctrl_ultra = [[0.0_f32; 4]; 16];
    if let Some(a3) = a3_cb {
        for r in 0..16 {
            for c in 0..4 {
                w_ctrl_ultra[r][c] = sanitize_finite_f32(a3.w_ctrl_ultra[r][c].get());
            }
        }
    }
    let mut w_ctrl_ext = [[0.0_f32; 16]; 16];
    let mut w_ctrl_ultra_ext = [[0.0_f32; 4]; 16];
    let cext_present = view.cext.is_some();
    if let Some(cext) = view.cext {
        for r in 0..16 {
            for c in 0..16 {
                w_ctrl_ext[r][c] = sanitize_finite_f32(cext.w_ctrl_ext[r][c].get());
            }
            if tier == Tier::Ultra {
                for c in 0..4 {
                    w_ctrl_ultra_ext[r][c] = sanitize_finite_f32(cext.w_ctrl_ultra_ext[r][c].get());
                }
            }
        }
    }

    let lfo_dphi_train = [
        clamp_finite_f32(
            a0_cb.lfo_dphi_train[0].get(),
            -core::f32::consts::PI,
            core::f32::consts::PI,
            0.0,
        ),
        clamp_finite_f32(
            a0_cb.lfo_dphi_train[1].get(),
            -core::f32::consts::PI,
            core::f32::consts::PI,
            0.0,
        ),
    ];
    let sigma_drift = clamp_finite_f32(a0_cb.sigma_drift.get(), 0.0, 2.0, 0.0);
    let rho_drift_train = {
        let r = a0_cb.rho_drift_train.get();
        if !r.is_finite() || r <= 0.0 {
            0.999
        } else {
            r.min(0.9999)
        }
    };
    let lfo_skew = decode_lfo_skew(a0_cb);
    let base_tank_dry_mix = decode_tank_dry_mix(a0_cb.tank_dry_mix_fp16.get());
    let ou_to_z_stage = decode_ou_to_z_stage(a0_cb, a3_cb);
    let trust_radius_mh = {
        let tr = a0_cb.trust_radius_mh.get();
        if !tr.is_finite() || tr <= 0.0 {
            1.0
        } else {
            tr.clamp(0.01, 16.0)
        }
    };
    let anti_zipper_scale = {
        let s = view.arena0.crossbar.anti_zipper_ramp_scale.get();
        if !s.is_finite() || s <= ANTI_ZIPPER_SNAP_EPS {
            DEFAULT_ANTI_ZIPPER_POLE
        } else {
            s.clamp(ANTI_ZIPPER_SNAP_EPS, 1.0)
        }
    };
    let ref_sample_rate_hz = view.header.reference_sample_rate_hz.get() as f32;
    let has_internal_modulators = compute_has_internal_modulators(
        &[lfo_dphi_train[0].to_bits(), lfo_dphi_train[1].to_bits()],
        &w_ctrl,
        &w_ctrl_ext,
        cext_present,
    );

    DecodedControlWire {
        u_int_base,
        u_int_ultra,
        w_ctrl,
        w_ctrl_ultra,
        cext_present,
        w_ctrl_ext,
        w_ctrl_ultra_ext,
        lfo_dphi_train,
        sigma_drift,
        rho_drift_train,
        lfo_skew,
        base_tank_dry_mix,
        ou_to_z_stage,
        trust_radius_mh,
        anti_zipper_scale,
        ref_sample_rate_hz,
        has_internal_modulators,
    }
}

fn populate_control_backing_from_view(
    view: &DnamBinView<'_>,
    tier: Tier,
    num_controls: usize,
    backing: &mut ControlBusBacking<'_>,
    overwrite_tsyn: bool,
) -> Result<(usize, usize), VmError> {
    let a0_cb = &view.arena0.control_bus_lite;
    let a3_cb = if tier == Tier::Ultra {
        view.arena3.map(|a3| &a3.control_bus_ultra_delta)
    } else {
        None
    };
    let num_contributors = decode_cp1_contributors(
        a0_cb,
        a3_cb,
        view.cp1_overflow,
        num_controls,
        &mut backing.contributors,
    )?;
    if num_contributors < backing.contributors.len() {
        backing.contributors[num_contributors..].fill(Cp1ContributorSlot::zero());
    }
    let distinct_contributor_controls =
        count_distinct_contributor_controls(&backing.contributors[..num_contributors]);

    for (i, wire_desc) in view.control_descriptors.iter().enumerate() {
        let desc = ControlClampDescriptor::from_wire(wire_desc);
        backing.descriptors[i] = desc;
        backing.raw_values[i] = desc.default_val;
    }
    if num_controls < backing.descriptors.len() {
        backing.descriptors[num_controls..].fill(ControlClampDescriptor::empty());
    }
    if num_controls < backing.raw_values.len() {
        backing.raw_values[num_controls..].fill(0.0);
    }
    if num_controls < backing.clamped_values.len() {
        backing.clamped_values[num_controls..].fill(0.0);
    }
    if num_controls < backing.normalized_values.len() {
        backing.normalized_values[num_controls..].fill(0.0);
    }
    if num_controls < backing.active_switch_options.len() {
        backing.active_switch_options[num_controls..].fill(0);
    }

    if overwrite_tsyn {
        match &mut backing.tsyn_entries {
            ControlTsynBacking::Mutable(buf) => {
                let s = buf.as_mut_slice();
                let n_tsyn = view.tsyn_entries.len();
                s[..n_tsyn].copy_from_slice(view.tsyn_entries);
                if n_tsyn < s.len() {
                    s[n_tsyn..].fill(<TempoSyncEntryWire as FromZeros>::new_zeroed());
                }
            }
            ControlTsynBacking::ReadOnly(_) => {}
        }
    }
    Ok((num_contributors, distinct_contributor_controls))
}

#[derive(Clone, Copy)]
struct ControlLoadCounts {
    controls: usize,
    contributors: usize,
    distinct_controls: usize,
    tsyn: usize,
}

impl<'a, B: MathBackend> ControlBus<'a, B> {
    fn apply_decoded_state(
        &mut self,
        decoded: DecodedControlWire,
        tier: Tier,
        host_sample_rate_hz: f32,
        counts: ControlLoadCounts,
    ) {
        self.stage_targets = [0.0; STAGE_TARGET_SLOTS];
        self.prev_stage_targets = [0.0; STAGE_TARGET_SLOTS];
        self.current_stage_params = [0.0; STAGE_TARGET_SLOTS];
        self.stage_ramp_deltas = [0.0; STAGE_TARGET_SLOTS];
        self.z_head = [0.0; 4];
        self.ctrl_basis_4 = [0.0; 4];
        self.z_head_ultra = [0.0; 4];
        self.z_stage_ultra = [0.0; 4];
        self.lfo_phases_q192 = [Q0_192Phase::ZERO; 2];
        self.lfo_dphi_host_bits = [0; 2];
        self.lfo_dphi_train = decoded.lfo_dphi_train;
        self.lfo_phasors = [[1.0, 0.0]; 2];
        self.lfo_skew = decoded.lfo_skew;
        self.env_states = [0.0; 4];
        self.base_tank_dry_mix = decoded.base_tank_dry_mix;
        self.ou_to_z_stage = decoded.ou_to_z_stage;
        self.ou_drift_state = [0.0; 4];
        self.drift_prng_state = DEFAULT_DRIFT_PRNG_SEED;
        self.sigma_drift = decoded.sigma_drift;
        self.rho_drift_train = decoded.rho_drift_train;
        self.rho_drift_host = decoded.rho_drift_train;
        self.trust_radius_mh = decoded.trust_radius_mh;
        self.anti_zipper_scale = decoded.anti_zipper_scale;
        self.alpha_ramp = decoded.anti_zipper_scale;
        self.tier = tier;
        self.ref_sample_rate_hz = decoded.ref_sample_rate_hz;
        self.host_sample_rate_hz = if host_sample_rate_hz.is_finite() && host_sample_rate_hz > 0.0 {
            host_sample_rate_hz
        } else {
            48_000.0
        };
        self.ctrl_stride = 16;
        self.ctrl_sub_counter = 0;
        self.num_controls = counts.controls;
        self.num_contributors = counts.contributors;
        self.distinct_contributor_controls = counts.distinct_controls;
        self.tsyn_count = counts.tsyn;
        self.has_internal_modulators = decoded.has_internal_modulators;
        self.is_dirty = true;
        self.live_analog_drift_enabled = false;
        self.offline_bounce = false;
        self.u_int_base = decoded.u_int_base;
        self.u_int_ultra = decoded.u_int_ultra;
        self.w_ctrl = decoded.w_ctrl;
        self.w_ctrl_ultra = decoded.w_ctrl_ultra;
        self.cext_present = decoded.cext_present;
        self.w_ctrl_ext = decoded.w_ctrl_ext;
        self.w_ctrl_ultra_ext = decoded.w_ctrl_ultra_ext;
        self.exact_fm_base = [ExactIntFactorAccumulator::new(counts.contributors); 4];
        self.exact_fm_ultra = [ExactIntFactorAccumulator::new(counts.contributors); 4];
        self.configure_sample_rate(self.host_sample_rate_hz);
        self.reset();
    }

    /// Loads and initializes [`ControlBus`] from a validated [`DnamBinView`] without heap allocation
    /// (provided `backing` capacity meets `num_controls` and decoded `CP1` count).
    pub fn from_dnam_bin(
        view: &DnamBinView<'_>,
        tier: Tier,
        host_sample_rate_hz: f32,
        mut backing: ControlBusBacking<'a>,
    ) -> Result<Self, VmError> {
        crate::math::warm_simd_feature_caches();
        if tier > view.retained_tier {
            return Err(VmError::InvalidTier);
        }
        let num_controls = view.header.num_controls.get() as usize;
        let (ctrl_cap, cp1_cap, tsyn_cap) = backing.capacity_limits();
        let req_cp1 = required_cp1_contributors(view)?;
        let tsyn_needs_copy = matches!(backing.tsyn_entries, ControlTsynBacking::Mutable(_));
        if num_controls > ctrl_cap
            || req_cp1 > cp1_cap
            || (tsyn_needs_copy && view.tsyn_entries.len() > tsyn_cap)
        {
            return Err(VmError::InsufficientBackingCapacity);
        }
        let (num_contributors, distinct_contributor_controls) = populate_control_backing_from_view(
            view,
            tier,
            num_controls,
            &mut backing,
            tsyn_needs_copy,
        )?;
        let decoded = decode_control_wire_params(view, tier);
        let mut bus = Self {
            stage_targets: [0.0; STAGE_TARGET_SLOTS],
            prev_stage_targets: [0.0; STAGE_TARGET_SLOTS],
            current_stage_params: [0.0; STAGE_TARGET_SLOTS],
            stage_ramp_deltas: [0.0; STAGE_TARGET_SLOTS],
            z_head: [0.0; 4],
            ctrl_basis_4: [0.0; 4],
            z_head_ultra: [0.0; 4],
            z_stage_ultra: [0.0; 4],
            lfo_phases_q192: [Q0_192Phase::ZERO; 2],
            lfo_dphi_host_bits: [0; 2],
            lfo_dphi_train: [0.0; 2],
            lfo_phasors: [[1.0, 0.0]; 2],
            lfo_skew: [0.0; 2],
            env_states: [0.0; 4],
            base_tank_dry_mix: 0.0,
            ou_to_z_stage: [0.0; 8],
            ou_drift_state: [0.0; 4],
            drift_prng_state: DEFAULT_DRIFT_PRNG_SEED,
            sigma_drift: 0.0,
            rho_drift_train: 0.999,
            rho_drift_host: 0.999,
            trust_radius_mh: 1.0,
            anti_zipper_scale: DEFAULT_ANTI_ZIPPER_POLE,
            alpha_ramp: DEFAULT_ANTI_ZIPPER_POLE,
            tier,
            ref_sample_rate_hz: 48_000.0,
            host_sample_rate_hz: 48_000.0,
            ctrl_stride: 16,
            ctrl_sub_counter: 0,
            num_controls: 0,
            num_contributors: 0,
            distinct_contributor_controls: 0,
            tsyn_count: 0,
            has_internal_modulators: false,
            is_dirty: true,
            live_analog_drift_enabled: false,
            offline_bounce: false,
            u_int_base: [[0.0; 8]; 4],
            u_int_ultra: [[[0.0; 8]; 4]; 3],
            w_ctrl: [[0.0; 16]; 16],
            w_ctrl_ultra: [[0.0; 4]; 16],
            cext_present: false,
            w_ctrl_ext: [[0.0; 16]; 16],
            w_ctrl_ultra_ext: [[0.0; 4]; 16],
            exact_fm_base: [ExactIntFactorAccumulator::new(0); 4],
            exact_fm_ultra: [ExactIntFactorAccumulator::new(0); 4],
            backing,
            _backend: PhantomData,
        };
        bus.apply_decoded_state(
            decoded,
            tier,
            host_sample_rate_hz,
            ControlLoadCounts {
                controls: num_controls,
                contributors: num_contributors,
                distinct_controls: distinct_contributor_controls,
                tsyn: view.tsyn_entries.len(),
            },
        );
        Ok(bus)
    }

    /// Reloads all descriptors, `CP1` contributors, `"TSYN"` rows, dispatch matrices, and LFO/drift
    /// parameters in-place from `view` (`0` heap allocations, `0` drops, fail-closed capacity).
    pub fn reload_from_dnam_bin_in_place(
        &mut self,
        view: &DnamBinView<'_>,
        tier: Tier,
        host_sample_rate_hz: f32,
    ) -> Result<(), VmError> {
        if tier > view.retained_tier {
            return Err(VmError::InvalidTier);
        }
        let num_controls = view.header.num_controls.get() as usize;
        let (ctrl_cap, cp1_cap, tsyn_cap) = self.backing.capacity_limits();
        if num_controls > ctrl_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        if !view.tsyn_entries.is_empty() && view.tsyn_entries.len() > tsyn_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        let req_cp1 = required_cp1_contributors(view)?;
        if req_cp1 > cp1_cap {
            return Err(VmError::InsufficientBackingCapacity);
        }
        let (num_contributors, distinct_contributor_controls) =
            populate_control_backing_from_view(view, tier, num_controls, &mut self.backing, true)?;
        let decoded = decode_control_wire_params(view, tier);
        self.apply_decoded_state(
            decoded,
            tier,
            host_sample_rate_hz,
            ControlLoadCounts {
                controls: num_controls,
                contributors: num_contributors,
                distinct_controls: distinct_contributor_controls,
                tsyn: view.tsyn_entries.len(),
            },
        );
        Ok(())
    }

    /// Allocates owned backing storage with extra reload headroom and initializes [`ControlBus`].
    #[cfg(feature = "alloc")]
    pub fn from_dnam_bin_owned_with_headroom(
        view: &DnamBinView<'a>,
        tier: Tier,
        host_sample_rate_hz: f32,
        min_controls: usize,
        min_contributors: usize,
        min_tsyn_entries: usize,
    ) -> Result<Self, VmError> {
        let num_controls = view.header.num_controls.get() as usize;
        let req_contrib = required_cp1_contributors(view).unwrap_or(8);
        let ctrl_cap = num_controls.max(min_controls);
        let cp1_cap = req_contrib.max(min_contributors);
        let tsyn_cap = view.tsyn_entries.len().max(min_tsyn_entries);
        let backing = ControlBusBacking::new_owned_with_capacity(
            ctrl_cap,
            cp1_cap,
            tsyn_cap,
            view.tsyn_entries,
        );
        Self::from_dnam_bin(view, tier, host_sample_rate_hz, backing)
    }

    /// Allocates owned backing storage and initializes [`ControlBus`] from `view`.
    #[cfg(feature = "alloc")]
    pub fn from_dnam_bin_owned(
        view: &DnamBinView<'a>,
        tier: Tier,
        host_sample_rate_hz: f32,
    ) -> Result<Self, VmError> {
        Self::from_dnam_bin_owned_with_headroom(view, tier, host_sample_rate_hz, 0, 8, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::{DnamBinBuildInput, encode_dnam_bin_into};

    #[test]
    fn test_from_dnam_bin_rejects_missing_tier_and_tiny_backing() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let desc = ControlClampDescriptor::range(0.0, 1.0, 0.5, 1, [0.0; 3]).to_wire();
        let controls = [desc];
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &controls,
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0u8; 128 + 64 + 4_608];
        encode_dnam_bin_into(&input, &mut buf).unwrap();
        let view = DnamBinView::parse(&buf).unwrap();

        let empty_backing = ControlBusBacking::from_slices(
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &mut [],
            &[],
        )
        .unwrap();
        assert_eq!(
            ControlBus::<crate::math::DefaultMathBackend>::from_dnam_bin(
                &view,
                Tier::Lite,
                48_000.0,
                empty_backing
            )
            .unwrap_err(),
            VmError::InsufficientBackingCapacity
        );

        let mut d2 = [ControlClampDescriptor::empty(); 1];
        let mut r2 = [0.0; 1];
        let mut c2 = [0.0; 1];
        let mut n2 = [0.0; 1];
        let mut s2 = [0u8; 1];
        let mut slots2 = [crate::control::Cp1ContributorSlot::zero(); 8];
        let backing2 = ControlBusBacking::from_slices(
            &mut d2,
            &mut r2,
            &mut c2,
            &mut n2,
            &mut s2,
            &mut slots2,
            &[],
        )
        .unwrap();
        assert_eq!(
            ControlBus::<crate::math::DefaultMathBackend>::from_dnam_bin(
                &view,
                Tier::Pro,
                48_000.0,
                backing2
            )
            .unwrap_err(),
            VmError::InvalidTier
        );
        #[cfg(feature = "alloc")]
        {
            let bus_owned = ControlBus::<crate::math::DefaultMathBackend>::from_dnam_bin_owned(
                &view,
                Tier::Lite,
                48_000.0,
            )
            .unwrap();
            assert_eq!(bus_owned.tier, Tier::Lite);
        }
    }

    #[test]
    fn test_decode_m10b_lfo_skew_tank_dry_mix_and_ou_to_z_stage() {
        use zerocopy::FromZeros;
        use zerocopy::byteorder::little_endian::U16;

        use crate::math::f32_to_fp16;

        let mut a0_cb = Arena0ControlBusWire::new_zeroed();
        a0_cb.lfo_skew_fp16 = [U16::new(f32_to_fp16(2.5)), U16::new(0x7E00)];
        assert_eq!(decode_lfo_skew(&a0_cb), [1.0, 0.0]);
        a0_cb.lfo_skew_fp16 = [U16::new(f32_to_fp16(-2.5)), U16::new(0x8000)];
        assert_eq!(
            decode_lfo_skew(&a0_cb).map(f32::to_bits),
            [(-1.0_f32).to_bits(), 0]
        );

        assert_eq!(decode_tank_dry_mix(0x0000).to_bits(), 0);
        assert_eq!(decode_tank_dry_mix(0x8000).to_bits(), 0);
        assert_eq!(decode_tank_dry_mix(0x7E00), 1.0);
        assert_eq!(decode_tank_dry_mix(0x7C00), 1.0);
        assert_eq!(decode_tank_dry_mix(0xFC00), 1.0);
        assert_eq!(decode_tank_dry_mix(f32_to_fp16(3.5)), 2.0);
        assert_eq!(decode_tank_dry_mix(f32_to_fp16(-3.5)), -2.0);
        assert_eq!(decode_tank_dry_mix(f32_to_fp16(0.75)), 0.75);

        a0_cb.ou_to_z_stage_fp16 = [
            U16::new(f32_to_fp16(6.0)),
            U16::new(f32_to_fp16(-6.0)),
            U16::new(f32_to_fp16(0.5)),
            U16::new(0x7E00),
        ];
        assert_eq!(
            decode_ou_to_z_stage(&a0_cb, None),
            [4.0, -4.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let mut a3_cb = Arena3ControlBusWire::new_zeroed();
        a3_cb.stage_latent_ctrl_ultra_fp16[160] = U16::new(f32_to_fp16(1.5));
        a3_cb.stage_latent_ctrl_ultra_fp16[161] = U16::new(f32_to_fp16(-5.0));
        a3_cb.stage_latent_ctrl_ultra_fp16[162] = U16::new(0x7E00);
        a3_cb.stage_latent_ctrl_ultra_fp16[163] = U16::new(f32_to_fp16(0.25));
        assert_eq!(
            decode_ou_to_z_stage(&a0_cb, Some(&a3_cb)),
            [4.0, -4.0, 0.5, 0.0, 1.5, -4.0, 0.0, 0.25]
        );

        let mut a0_custom = DnamBinBuildInput::default_arena0_lite();
        a0_custom.control_bus_lite.lfo_skew_fp16 =
            [U16::new(f32_to_fp16(0.5)), U16::new(f32_to_fp16(-2.5))];
        a0_custom.control_bus_lite.tank_dry_mix_fp16 = U16::new(0);
        a0_custom.control_bus_lite.ou_to_z_stage_fp16 = [
            U16::new(f32_to_fp16(1.25)),
            U16::new(f32_to_fp16(-0.5)),
            U16::new(f32_to_fp16(5.0)),
            U16::new(0x7E00),
        ];
        let desc = ControlClampDescriptor::range(0.0, 1.0, 0.5, 1, [0.0; 3]).to_wire();
        let controls = [desc];
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &controls,
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0_custom,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0u8; 128 + 64 + 4_608];
        encode_dnam_bin_into(&input, &mut buf).unwrap();
        let view = DnamBinView::parse(&buf).unwrap();
        let mut d = [ControlClampDescriptor::empty(); 1];
        let mut r = [0.0; 1];
        let mut c = [0.0; 1];
        let mut n = [0.0; 1];
        let mut s = [0u8; 1];
        let mut slots = [crate::control::Cp1ContributorSlot::zero(); 8];
        let backing =
            ControlBusBacking::from_slices(&mut d, &mut r, &mut c, &mut n, &mut s, &mut slots, &[])
                .unwrap();
        let bus = ControlBus::<crate::math::DefaultMathBackend>::from_dnam_bin(
            &view,
            Tier::Lite,
            48_000.0,
            backing,
        )
        .unwrap();
        assert_eq!(bus.lfo_skew, [0.5, -1.0]);
        assert_eq!(bus.base_tank_dry_mix(), 0.0);
        assert_eq!(
            bus.ou_to_z_stage,
            [1.25, -0.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_control_bus_reload_in_place_headroom_and_fail_closed() {
        use crate::math::f32_to_fp16;
        use zerocopy::IntoBytes;
        use zerocopy::byteorder::little_endian::{F32, U16};

        let mut a0_a = DnamBinBuildInput::default_arena0_lite();
        a0_a.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.25));
        a0_a.control_bus_lite.cp1_dir.cp1_dir_count = U16::new(1);
        a0_a.control_bus_lite.cp1_dir.entries[0] = [0, 0x01, 0];
        a0_a.control_bus_lite.cp1_base[..4].copy_from_slice(F32::new(0.5).as_bytes());
        let desc_a = [ControlClampDescriptor::range(0.0, 1.0, 0.2, 1, [0.0; 3]).to_wire()];
        let input_a = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &desc_a,
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0_a,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut a0_b = DnamBinBuildInput::default_arena0_lite();
        a0_b.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.75));
        a0_b.control_bus_lite.lfo_skew_fp16 = [U16::new(f32_to_fp16(0.5)), U16::new(0)];
        let desc_b = [
            ControlClampDescriptor::range(0.0, 10.0, 2.0, 1, [0.0; 3]).to_wire(),
            ControlClampDescriptor::range(-5.0, 5.0, -1.0, 1, [0.0; 3]).to_wire(),
        ];
        let mut tsyn_row = <TempoSyncEntryWire as FromZeros>::new_zeroed();
        tsyn_row.ctrl_idx = U16::new(0);
        tsyn_row.div_ratio = F32::new(1.0);
        tsyn_row.bpm_factor = F32::new(60_000.0);
        tsyn_row.lobatto_anchors = [0.0, 5.0, 10.0].map(F32::new);
        tsyn_row.declared_min = F32::new(0.0);
        tsyn_row.declared_max = F32::new(10.0);
        let tsyn_b = [tsyn_row];
        let input_b = DnamBinBuildInput {
            controls: &desc_b,
            tsyn: &tsyn_b,
            arena0: &a0_b,
            ..input_a
        };
        let desc_c = [desc_a[0], desc_a[0], desc_a[0]];
        let input_c = DnamBinBuildInput {
            controls: &desc_c,
            ..input_a
        };

        let mut buf_a = [0u8; 8_192];
        let mut buf_b = [0u8; 8_192];
        let mut buf_c = [0u8; 8_192];
        let len_a = encode_dnam_bin_into(&input_a, &mut buf_a).unwrap();
        let len_b = encode_dnam_bin_into(&input_b, &mut buf_b).unwrap();
        let len_c = encode_dnam_bin_into(&input_c, &mut buf_c).unwrap();
        let view_a = DnamBinView::parse(&buf_a[..len_a]).unwrap();
        let view_b = DnamBinView::parse(&buf_b[..len_b]).unwrap();
        let view_c = DnamBinView::parse(&buf_c[..len_c]).unwrap();

        let mut d = [ControlClampDescriptor::empty(); 2];
        let mut r = [0.0; 2];
        let mut c = [0.0; 2];
        let mut n = [0.0; 2];
        let mut s = [0u8; 2];
        let mut slots = [crate::control::Cp1ContributorSlot::zero(); 8];
        let mut tsyn_mut = [<TempoSyncEntryWire as FromZeros>::new_zeroed(); 2];
        let backing = ControlBusBacking::from_mut_slices(
            &mut d,
            &mut r,
            &mut c,
            &mut n,
            &mut s,
            &mut slots,
            &mut tsyn_mut,
        )
        .unwrap();
        let mut bus = ControlBus::<crate::math::DefaultMathBackend>::from_dnam_bin(
            &view_a,
            Tier::Lite,
            48_000.0,
            backing,
        )
        .unwrap();

        assert_eq!(
            bus.reload_from_dnam_bin_in_place(&view_c, Tier::Lite, 48_000.0)
                .unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
        assert_eq!(bus.num_controls, 1);
        assert_eq!(bus.base_tank_dry_mix(), 0.25);

        bus.reload_from_dnam_bin_in_place(&view_b, Tier::Lite, 96_000.0)
            .unwrap();
        assert_eq!(bus.num_controls, 2);
        assert_eq!(bus.tsyn_count, 1);
        assert_eq!(bus.base_tank_dry_mix(), 0.75);
        assert_eq!(bus.lfo_skew, [0.5, 0.0]);
        assert_eq!(bus.apply_tempo_bpm(120.0), 1);

        bus.reload_from_dnam_bin_in_place(&view_a, Tier::Lite, 48_000.0)
            .unwrap();
        assert_eq!(bus.num_controls, 1);
        assert_eq!(bus.tsyn_count, 0);
        assert_eq!(bus.apply_tempo_bpm(120.0), 0);
        assert_eq!(bus.backing.raw_values[1], 0.0);

        let backing_ro =
            ControlBusBacking::from_slices(&mut d, &mut r, &mut c, &mut n, &mut s, &mut slots, &[])
                .unwrap();
        let mut bus_ro = ControlBus::<crate::math::DefaultMathBackend>::from_dnam_bin(
            &view_a,
            Tier::Lite,
            48_000.0,
            backing_ro,
        )
        .unwrap();
        assert_eq!(
            bus_ro
                .reload_from_dnam_bin_in_place(&view_b, Tier::Lite, 48_000.0)
                .unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
        bus_ro
            .reload_from_dnam_bin_in_place(&view_a, Tier::Lite, 48_000.0)
            .unwrap();
    }
}
