//! Cold-path `.dnam-bin` loader constructors for [`super::ControlBus`].

use core::marker::PhantomData;

use super::dispatch::{
    ANTI_ZIPPER_SNAP_EPS, DEFAULT_ANTI_ZIPPER_POLE, DEFAULT_DRIFT_PRNG_SEED, STAGE_TARGET_SLOTS,
};
use super::{ControlBus, ControlBusBacking};
use crate::bin::DnamBinView;
use crate::control::ControlClampDescriptor;
use crate::control::cp1::{count_distinct_contributor_controls, decode_cp1_contributors};
use crate::control::exact_int::ExactIntFactorAccumulator;
use crate::math::{MathBackend, Q0_192Phase, clamp_finite_f32, const_abs_f32, sanitize_finite_f32};
use crate::types::{Tier, VmError};

#[inline]
fn non_zero_f32_cols_8_16(m: &[[f32; 16]; 16]) -> bool {
    m.iter()
        .any(|row| row[8..16].iter().any(|&v| const_abs_f32(v) != 0.0))
}

impl<'a, B: MathBackend> ControlBus<'a, B> {
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
        if backing.descriptors.len() < num_controls
            || backing.raw_values.len() < num_controls
            || backing.clamped_values.len() < num_controls
            || backing.normalized_values.len() < num_controls
            || backing.active_switch_options.len() < num_controls
        {
            return Err(VmError::InsufficientBackingCapacity);
        }
        for (i, wire_desc) in view.control_descriptors.iter().enumerate() {
            let desc = ControlClampDescriptor::from_wire(wire_desc);
            backing.descriptors[i] = desc;
            backing.raw_values[i] = desc.default_val;
        }

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
        let distinct_contributor_controls =
            count_distinct_contributor_controls(&backing.contributors[..num_contributors]);

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
                        w_ctrl_ultra_ext[r][c] =
                            sanitize_finite_f32(cext.w_ctrl_ultra_ext[r][c].get());
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
        let env_poles_train = [0, 1, 2].map(|i| {
            clamp_finite_f32(
                a0_cb.env_poles[i].get(),
                0.0,
                0.99995,
                [0.95, 0.99, 0.998][i],
            )
        });
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
        let ref_sr = view.header.reference_sample_rate_hz.get() as f32;
        let has_internal_modulators = (const_abs_f32(lfo_dphi_train[0]) != 0.0)
            || (const_abs_f32(lfo_dphi_train[1]) != 0.0)
            || non_zero_f32_cols_8_16(&w_ctrl)
            || (cext_present && non_zero_f32_cols_8_16(&w_ctrl_ext));

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
            lfo_dphi_train,
            lfo_phasors: [[1.0, 0.0]; 2],
            lfo_harmonics: [0.0; 8],
            env_poles_host: env_poles_train,
            env_poles_train,
            env_states: [0.0; 4],
            ou_drift_state: [0.0; 4],
            drift_prng_state: DEFAULT_DRIFT_PRNG_SEED,
            sigma_drift,
            rho_drift_train,
            rho_drift_host: rho_drift_train,
            trust_radius_mh,
            anti_zipper_scale,
            alpha_ramp: anti_zipper_scale,
            tier,
            ref_sample_rate_hz: ref_sr,
            host_sample_rate_hz: if host_sample_rate_hz.is_finite() && host_sample_rate_hz > 0.0 {
                host_sample_rate_hz
            } else {
                48_000.0
            },
            ctrl_stride: 16,
            inv_ctrl_stride: 1.0 / 16.0,
            ctrl_sub_counter: 0,
            window_abs_acc: 0.0,
            num_controls,
            num_contributors,
            distinct_contributor_controls,
            tsyn_count: view.tsyn_entries.len(),
            has_internal_modulators,
            is_dirty: true,
            live_analog_drift_enabled: false,
            offline_bounce: false,
            u_int_base,
            u_int_ultra,
            w_ctrl,
            w_ctrl_ultra,
            cext_present,
            w_ctrl_ext,
            w_ctrl_ultra_ext,
            exact_fm_base: [ExactIntFactorAccumulator::new(num_contributors); 4],
            exact_fm_ultra: [ExactIntFactorAccumulator::new(num_contributors); 4],
            backing,
            _backend: PhantomData,
        };
        bus.configure_sample_rate(bus.host_sample_rate_hz);
        bus.reset();
        Ok(bus)
    }

    /// Allocates owned backing storage and initializes [`ControlBus`] from `view`.
    #[cfg(feature = "alloc")]
    pub fn from_dnam_bin_owned(
        view: &DnamBinView<'a>,
        tier: Tier,
        host_sample_rate_hz: f32,
    ) -> Result<Self, VmError> {
        let num_controls = view.header.num_controls.get() as usize;
        let max_contrib = if !view.cp1_overflow.is_empty() && view.cp1_overflow.len() >= 6 {
            zerocopy::FromBytes::read_from_prefix(view.cp1_overflow)
                .map(|(h, _): (crate::bin::Cp1StreamHeaderWire, _)| h.count.get() as usize)
                .unwrap_or(8)
        } else {
            usize::from(
                view.arena0
                    .control_bus_lite
                    .cp1_dir
                    .cp1_dir_count
                    .get()
                    .min(8),
            )
        };
        let backing = ControlBusBacking::new_owned(num_controls, max_contrib, view.tsyn_entries);
        Self::from_dnam_bin(view, tier, host_sample_rate_hz, backing)
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
            r_ac_default: 0.99,
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
}
