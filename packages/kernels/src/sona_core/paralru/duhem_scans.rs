//! Exact `0-ULP` joint `(L, R)` Selective-Duhem stereo-linked `ParaLRU` forward tile execution
//! (`ref[dsp.op-n.core.matryoshka-blocks]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::simd::{
    GpuVec4, InterleavedStereo8, KernelVec4, MAX_PARALRU_R_F32, StereoVec8,
    scrub_and_flush_f32_bits,
};

use super::{
    MAX_PARALRU_DUHEM_GAMMA_F32, SonaParaLruDuhemCoeffs4, apply_stereo_link_stereo8,
    clamp_f32_safe, step_paralru_duhem_stereo8,
};

/// Maximum clamped `two_svf_g = 2 * svf_g` (`g <= 64.0`, `ref[dsp.op-n.core.tpt-tone-basis]`).
pub const MAX_PARALRU_TWO_SVF_G_F32: f32 = 128.0;

const _: () = assert!(MAX_PARALRU_TWO_SVF_G_F32 == 128.0);

/// Clamped scalar Selective-Duhem stereo-linked `ParaLRU` parameters
/// (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParaLruDuhemParams {
    pub alpha_rel: f32,
    pub alpha_atk: f32,
    pub kappa: f32,
    pub gamma_duhem: f32,
    pub signed_mix: f32,
    pub stereo_link: f32,
    pub two_svf_g: f32,
}

impl ParaLruDuhemParams {
    /// Clamps Selective-Duhem `ParaLRU` ballistics into the unconditionally contractive domain.
    #[inline]
    #[must_use]
    pub fn clamped(
        alpha_rel: f32,
        alpha_atk: f32,
        kappa: f32,
        gamma_duhem: f32,
        signed_mix: f32,
        stereo_link: f32,
        two_svf_g: f32,
    ) -> Self {
        Self {
            alpha_rel: clamp_f32_safe(alpha_rel, 0.0, MAX_PARALRU_R_F32) + 0.0,
            alpha_atk: clamp_f32_safe(alpha_atk, 0.0, 1.0) + 0.0,
            kappa: clamp_f32_safe(kappa, 0.0, 1.0) + 0.0,
            gamma_duhem: clamp_f32_safe(gamma_duhem, 0.0, MAX_PARALRU_DUHEM_GAMMA_F32) + 0.0,
            signed_mix: clamp_f32_safe(signed_mix, 0.0, 1.0) + 0.0,
            stereo_link: clamp_f32_safe(stereo_link, 0.0, 1.0) + 0.0,
            two_svf_g: clamp_f32_safe(two_svf_g, 0.0, MAX_PARALRU_TWO_SVF_G_F32) + 0.0,
        }
    }

    /// Broadcasts scalar Selective-Duhem parameters into a 4-lane [`SonaParaLruDuhemCoeffs4`].
    #[inline]
    #[must_use]
    pub fn to_coeffs4<V: KernelVec4>(self) -> SonaParaLruDuhemCoeffs4<V> {
        SonaParaLruDuhemCoeffs4::from_clamped(
            V::splat(self.alpha_rel),
            V::splat(self.alpha_atk),
            V::splat(self.kappa),
            V::splat(self.gamma_duhem),
            V::splat(self.signed_mix),
            V::splat(self.stereo_link),
        )
    }
}

/// Output trajectories and end states from a joint `(L, R)` Selective-Duhem `ParaLRU` forward tile.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParaLruDuhemForwardOut<const N: usize> {
    pub v_traj_l: [f32; N],
    pub v_traj_r: [f32; N],
    pub v_eff_traj_l: [f32; N],
    pub v_eff_traj_r: [f32; N],
    pub v_end_l: f32,
    pub v_end_r: f32,
}

/// Steps `valid_len.min(N)` stereo frames of `(u[0], u[1])` and `(bp[0], bp[1])` through joint
/// `(L, R)` Selective-Duhem `ParaLRU` + stereo linking via [`step_paralru_duhem_stereo8`] and
/// [`apply_stereo_link_stereo8`] (`0-ULP` parity), zeroing unused tail entries `i >= valid_len`.
#[inline]
#[must_use]
pub fn run_paralru_duhem_forward_partial_tile<const N: usize>(
    u: [&[f32; N]; 2],
    bp: [&[f32; N]; 2],
    valid_len: usize,
    v_init: [f32; 2],
    params: ParaLruDuhemParams,
) -> ParaLruDuhemForwardOut<N> {
    let mut v_traj_l = [0.0_f32; N];
    let mut v_traj_r = [0.0_f32; N];
    let mut v_eff_traj_l = [0.0_f32; N];
    let mut v_eff_traj_r = [0.0_f32; N];
    let v0_l = scrub_and_flush_f32_bits(v_init[0]) + 0.0;
    let v0_r = scrub_and_flush_f32_bits(v_init[1]) + 0.0;
    let active = valid_len.min(N);
    if active == 0 {
        return ParaLruDuhemForwardOut {
            v_traj_l,
            v_traj_r,
            v_eff_traj_l,
            v_eff_traj_r,
            v_end_l: v0_l,
            v_end_r: v0_r,
        };
    }

    let coeffs = params.to_coeffs4::<GpuVec4>();
    let two_g = InterleavedStereo8::<GpuVec4>::splat(params.two_svf_g);
    let mut state = InterleavedStereo8::<GpuVec4>::splat_lr(v0_l, v0_r);

    for i in 0..active {
        let ul_c = scrub_and_flush_f32_bits(u[0][i]) + 0.0;
        let ur_c = scrub_and_flush_f32_bits(u[1][i]) + 0.0;
        let bpl_c = scrub_and_flush_f32_bits(bp[0][i]) + 0.0;
        let bpr_c = scrub_and_flush_f32_bits(bp[1][i]) + 0.0;

        let u_s = InterleavedStereo8::splat_lr(ul_c, ur_c);
        let flux_inc = two_g.mul(InterleavedStereo8::splat_lr(bpl_c, bpr_c));
        let v_raw = step_paralru_duhem_stereo8(u_s, flux_inc, &mut state, &coeffs);
        let v_eff = apply_stereo_link_stereo8(v_raw, coeffs.stereo_link);

        let (rl, rr) = v_raw.into_lr();
        let (el, er) = v_eff.into_lr();
        v_traj_l[i] = rl.to_array()[0];
        v_traj_r[i] = rr.to_array()[0];
        v_eff_traj_l[i] = el.to_array()[0];
        v_eff_traj_r[i] = er.to_array()[0];
    }

    let (end_l, end_r) = state.into_lr();
    ParaLruDuhemForwardOut {
        v_traj_l,
        v_traj_r,
        v_eff_traj_l,
        v_eff_traj_r,
        v_end_l: end_l.to_array()[0],
        v_end_r: end_r.to_array()[0],
    }
}

/// Steps all `N` stereo frames through joint `(L, R)` Selective-Duhem `ParaLRU` + stereo linking
/// with exact `0-ULP` parity against [`step_paralru_duhem_stereo8`] (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline]
#[must_use]
pub fn run_paralru_duhem_forward_tile<const N: usize>(
    u: [&[f32; N]; 2],
    bp: [&[f32; N]; 2],
    v_init: [f32; 2],
    params: ParaLruDuhemParams,
) -> ParaLruDuhemForwardOut<N> {
    run_paralru_duhem_forward_partial_tile(u, bp, N, v_init, params)
}

#[cfg(test)]
mod tests {
    use super::super::scans::run_paralru_forward_tile;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_paralru_duhem_forward_tile_edge_cases_and_zero_dc() {
        let p = ParaLruDuhemParams::clamped(0.95, 0.4, 0.7, 2.5, 0.6, 0.35, 0.8);
        let empty = run_paralru_duhem_forward_tile::<0>([&[], &[]], [&[], &[]], [-1.25, 0.75], p);
        assert_eq!(empty.v_end_l, -1.25);
        assert_eq!(empty.v_end_r, 0.75);

        let partial0 = run_paralru_duhem_forward_partial_tile(
            [&[1.0; 4], &[2.0; 4]],
            [&[0.5; 4], &[-0.5; 4]],
            0,
            [-0.5, 1.5],
            p,
        );
        assert_eq!(partial0.v_traj_l, [0.0; 4]);
        assert_eq!(partial0.v_eff_traj_r, [0.0; 4]);
        assert_eq!(partial0.v_end_l, -0.5);
        assert_eq!(partial0.v_end_r, 1.5);

        let z_neg = [-0.0_f32; 8];
        let z_pos = [0.0_f32; 8];
        let zero_dc =
            run_paralru_duhem_forward_tile([&z_neg, &z_pos], [&z_pos, &z_neg], [-0.0, -0.0], p);
        for i in 0..8 {
            assert_eq!(zero_dc.v_traj_l[i].to_bits(), 0);
            assert_eq!(zero_dc.v_traj_r[i].to_bits(), 0);
            assert_eq!(zero_dc.v_eff_traj_l[i].to_bits(), 0);
            assert_eq!(zero_dc.v_eff_traj_r[i].to_bits(), 0);
        }
        assert_eq!(zero_dc.v_end_l.to_bits(), 0);
        assert_eq!(zero_dc.v_end_r.to_bits(), 0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_paralru_duhem_forward_tile_zero_ulp_parity(
            u_l_vec in proptest::collection::vec(-8.0_f32..8.0_f32, 32),
            u_r_vec in proptest::collection::vec(-8.0_f32..8.0_f32, 32),
            bp_l_vec in proptest::collection::vec(-4.0_f32..4.0_f32, 32),
            bp_r_vec in proptest::collection::vec(-4.0_f32..4.0_f32, 32),
            v_init_l in -3.0_f32..3.0_f32,
            v_init_r in -3.0_f32..3.0_f32,
            alpha_rel in 0.05_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.05_f32..0.95_f32,
            kappa in 0.0_f32..1.0_f32,
            gamma_duhem in 0.0_f32..16.0_f32,
            signed_mix in 0.0_f32..1.0_f32,
            stereo_link in 0.0_f32..1.0_f32,
            two_svf_g in 0.01_f32..8.0_f32,
            valid_len in 1_usize..=32_usize,
        ) {
            let mut u_l = [0.0_f32; 32];
            let mut u_r = [0.0_f32; 32];
            let mut bp_l = [0.0_f32; 32];
            let mut bp_r = [0.0_f32; 32];
            u_l.copy_from_slice(&u_l_vec);
            u_r.copy_from_slice(&u_r_vec);
            bp_l.copy_from_slice(&bp_l_vec);
            bp_r.copy_from_slice(&bp_r_vec);

            let p = ParaLruDuhemParams::clamped(
                alpha_rel,
                alpha_atk,
                kappa,
                gamma_duhem,
                signed_mix,
                stereo_link,
                two_svf_g,
            );
            let full = run_paralru_duhem_forward_tile(
                [&u_l, &u_r],
                [&bp_l, &bp_r],
                [v_init_l, v_init_r],
                p,
            );
            let part = run_paralru_duhem_forward_partial_tile(
                [&u_l, &u_r],
                [&bp_l, &bp_r],
                valid_len,
                [v_init_l, v_init_r],
                p,
            );

            let coeffs = p.to_coeffs4::<GpuVec4>();
            let two_g = InterleavedStereo8::<GpuVec4>::splat(p.two_svf_g);
            let mut st = InterleavedStereo8::<GpuVec4>::splat_lr(v_init_l, v_init_r);
            for i in 0..32 {
                let u_s = InterleavedStereo8::splat_lr(u_l[i], u_r[i]);
                let flux_s = two_g.mul(InterleavedStereo8::splat_lr(bp_l[i], bp_r[i]));
                let v_raw = step_paralru_duhem_stereo8(u_s, flux_s, &mut st, &coeffs);
                let v_eff = apply_stereo_link_stereo8(v_raw, coeffs.stereo_link);
                let (rl, rr) = v_raw.into_lr();
                let (el, er) = v_eff.into_lr();
                prop_assert_eq!(full.v_traj_l[i].to_bits(), rl.to_array()[0].to_bits());
                prop_assert_eq!(full.v_traj_r[i].to_bits(), rr.to_array()[0].to_bits());
                prop_assert_eq!(full.v_eff_traj_l[i].to_bits(), el.to_array()[0].to_bits());
                prop_assert_eq!(full.v_eff_traj_r[i].to_bits(), er.to_array()[0].to_bits());
                if i < valid_len {
                    prop_assert_eq!(part.v_traj_l[i].to_bits(), full.v_traj_l[i].to_bits());
                    prop_assert_eq!(part.v_eff_traj_r[i].to_bits(), full.v_eff_traj_r[i].to_bits());
                } else {
                    prop_assert_eq!(part.v_traj_l[i].to_bits(), 0);
                    prop_assert_eq!(part.v_eff_traj_r[i].to_bits(), 0);
                }
            }
            prop_assert_eq!(full.v_end_l.to_bits(), full.v_traj_l[31].to_bits());
            prop_assert_eq!(part.v_end_l.to_bits(), full.v_traj_l[valid_len - 1].to_bits());

            // Neutral Duhem mode when gamma_duhem == 0, signed_mix == 0, stereo_link == 0.
            let p_neutral = ParaLruDuhemParams::clamped(alpha_rel, alpha_atk, kappa, 0.0, 0.0, 0.0, two_svf_g);
            let duhem_neutral = run_paralru_duhem_forward_tile(
                [&u_l, &u_r],
                [&bp_l, &bp_r],
                [v_init_l, v_init_r],
                p_neutral,
            );
            let (ref_l, ref_end_l) = run_paralru_forward_tile(&u_l, v_init_l, p.alpha_rel, p.alpha_atk, p.kappa);
            let (ref_r, ref_end_r) = run_paralru_forward_tile(&u_r, v_init_r, p.alpha_rel, p.alpha_atk, p.kappa);
            for i in 0..32 {
                prop_assert_eq!(duhem_neutral.v_traj_l[i].to_bits(), ref_l[i].to_bits());
                prop_assert_eq!(duhem_neutral.v_eff_traj_l[i].to_bits(), ref_l[i].to_bits());
                prop_assert_eq!(duhem_neutral.v_traj_r[i].to_bits(), ref_r[i].to_bits());
                prop_assert_eq!(duhem_neutral.v_eff_traj_r[i].to_bits(), ref_r[i].to_bits());
            }
            prop_assert_eq!(duhem_neutral.v_end_l.to_bits(), ref_end_l.to_bits());
            prop_assert_eq!(duhem_neutral.v_end_r.to_bits(), ref_end_r.to_bits());
        }
    }
}
