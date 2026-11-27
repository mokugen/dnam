//! Smooth non-saturating SONA v4.0 Selective-Duhem `ParaLRU` ODE (`C^2` quartic `soft_rect`,
//! smooth `1 - alpha_atk * rcp_ge1_nr2(1 + gamma * |2*g*BP|)` gating with zero gradient collapse)
//! and `O(log2 N)` reverse-time adjoint scan (`ref[dsp.op-n.core.matryoshka-blocks]`).

pub mod adjoint;

pub use adjoint::{
    paralru_duhem_v4_adjoint_step_elem, run_paralru_duhem_v4_adjoint_scan_partial_tile,
    run_paralru_duhem_v4_adjoint_scan_tile,
};

use crate::simd::{
    GpuVec4, InterleavedStereo8, KernelVec4, MAX_DUHEM_FLUX_ABS, StereoVec8,
    scrub_and_flush_f32_bits,
};

use super::duhem_scans::{ParaLruDuhemForwardOut, ParaLruDuhemParams};
use super::soft_rect::{soft_rect_quartic_stereo8, soft_rect_quartic_vec4};
use super::{ONE_THIRD_F32, SonaParaLruDuhemCoeffs4, apply_stereo_link_stereo8};

/// Steps 4 decoupled `1D` scalar `ParaLRU` v4 lanes using the $C^2$ quartic spline rectifier
/// (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline(always)]
pub fn step_paralru_v4_vec4<V: KernelVec4>(
    u: V,
    v_state: &mut V,
    alpha_rel: V,
    alpha_atk: V,
    kappa: V,
) -> V {
    let decayed = alpha_rel.mul(*v_state);
    let arg = u.fms(kappa, decayed);
    let (phi, _) = soft_rect_quartic_vec4(arg);
    let one_minus_atk = V::splat(1.0).sub(alpha_atk);
    let v_new = decayed.fma(one_minus_atk, phi).flush_recursive_state();
    *v_state = v_new;
    v_new
}

/// Steps stereo `L+R` `4`-lane `ParaLRU` v4 states in `StereoVec8` (`0-ULP` parity with
/// [`step_paralru_v4_vec4`]).
#[inline(always)]
pub fn step_paralru_v4_stereo8<S: StereoVec8>(
    u: S,
    v_state: &mut S,
    alpha_rel: S::Half,
    alpha_atk: S::Half,
    kappa: S::Half,
) -> S {
    let a_rel = S::broadcast_half(alpha_rel);
    let a_atk = S::broadcast_half(alpha_atk);
    let kap = S::broadcast_half(kappa);
    let decayed = a_rel.mul(*v_state);
    let arg = u.fms(kap, decayed);
    let (phi, _) = soft_rect_quartic_stereo8(arg);
    let one_minus_atk = S::splat(1.0).sub(a_atk);
    let v_new = decayed.fma(one_minus_atk, phi).flush_recursive_state();
    *v_state = v_new;
    v_new
}

/// Steps 4 SONA v4.0 smooth non-saturating Selective-Duhem `ParaLRU` lanes (`C^2` quartic
/// `soft_rect` and smooth `1 - alpha_atk * rcp_ge1_nr2(1 + gamma_duhem * |flux|)` gate,
/// `ref[dsp.op-n.core.matryoshka-blocks]`).
///
/// Guards `duhem_mod <= 0.0 -> inv_denom = 1.0` for `0-ULP` neutral parity and exact coercive
/// remanence, and clamps `flux_abs <= MAX_DUHEM_FLUX_ABS` (`1e18`) to prevent `f32` overflow.
#[inline(always)]
pub fn step_paralru_duhem_v4_vec4<V: KernelVec4>(
    u_rect: V,
    flux_inc: V,
    v_state: &mut V,
    coeffs: &SonaParaLruDuhemCoeffs4<V>,
) -> V {
    let zero = V::zero();
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);

    let flux_abs = flux_inc
        .flush_recursive_state()
        .abs()
        .min(V::splat(MAX_DUHEM_FLUX_ABS));
    let duhem_mod = V::select_le(
        coeffs.gamma_duhem,
        zero,
        zero,
        coeffs.gamma_duhem.mul(flux_abs),
    );
    let denom = one.add(duhem_mod);
    let inv_denom = V::select_le(duhem_mod, zero, one, denom.rcp_ge1_nr2());
    let rho = one.fms(coeffs.alpha_atk, inv_denom).clamp(zero, one);

    let d = coeffs.alpha_rel.mul(*v_state);
    let one_minus_eta = one.sub(coeffs.signed_mix);
    let z = u_rect.fms(coeffs.kappa.mul(one_minus_eta), d);
    let z_bar = z.clamp(neg_one, one);
    let z_bar2 = z_bar.mul(z_bar);
    let psi_1 = z_bar.mul(one.fms(V::splat(ONE_THIRD_F32), z_bar2));
    let (phi_rect, _) = soft_rect_quartic_vec4(z);
    let phi_eta = V::select_le(
        coeffs.signed_mix,
        zero,
        phi_rect,
        one_minus_eta.mul(phi_rect).fma(coeffs.signed_mix, psi_1),
    );
    let keep = one.fms(coeffs.signed_mix, rho);
    let v_new = keep
        .mul(d)
        .fma(rho, phi_eta)
        .flush_recursive_state()
        .add(zero);
    *v_state = v_new;
    v_new
}

/// Steps stereo `L+R` `4`-lane SONA v4.0 smooth Selective-Duhem `ParaLRU` states in `StereoVec8`
/// (`0-ULP` parity with [`step_paralru_duhem_v4_vec4`]).
#[inline(always)]
pub fn step_paralru_duhem_v4_stereo8<S: StereoVec8>(
    u_rect: S,
    flux_inc: S,
    v_state: &mut S,
    coeffs: &SonaParaLruDuhemCoeffs4<S::Half>,
) -> S {
    let zero = S::zero();
    let one = S::splat(1.0);
    let neg_one = S::splat(-1.0);
    let a_rel = S::broadcast_half(coeffs.alpha_rel);
    let a_atk = S::broadcast_half(coeffs.alpha_atk);
    let kap = S::broadcast_half(coeffs.kappa);
    let gam = S::broadcast_half(coeffs.gamma_duhem);
    let eta = S::broadcast_half(coeffs.signed_mix);

    let flux_abs = flux_inc
        .flush_recursive_state()
        .abs()
        .min(S::splat(MAX_DUHEM_FLUX_ABS));
    let duhem_mod = S::select_le(gam, zero, zero, gam.mul(flux_abs));
    let denom = one.add(duhem_mod);
    let inv_denom = S::select_le(duhem_mod, zero, one, denom.rcp_ge1_nr2());
    let rho = one.fms(a_atk, inv_denom).clamp(zero, one);

    let d = a_rel.mul(*v_state);
    let one_minus_eta = one.sub(eta);
    let z = u_rect.fms(kap.mul(one_minus_eta), d);
    let z_bar = z.clamp(neg_one, one);
    let z_bar2 = z_bar.mul(z_bar);
    let psi_1 = z_bar.mul(one.fms(S::splat(ONE_THIRD_F32), z_bar2));
    let (phi_rect, _) = soft_rect_quartic_stereo8(z);
    let phi_eta = S::select_le(
        eta,
        zero,
        phi_rect,
        one_minus_eta.mul(phi_rect).fma(eta, psi_1),
    );
    let keep = one.fms(eta, rho);
    let v_new = keep
        .mul(d)
        .fma(rho, phi_eta)
        .flush_recursive_state()
        .add(zero);
    *v_state = v_new;
    v_new
}

/// Steps `valid_len.min(N)` stereo frames through smooth SONA v4.0 Selective-Duhem `ParaLRU` +
/// stereo linking (`0-ULP` parity with [`step_paralru_duhem_v4_stereo8`]).
#[inline]
#[must_use]
pub fn run_paralru_duhem_v4_forward_partial_tile<const N: usize>(
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
        let v_raw = step_paralru_duhem_v4_stereo8(u_s, flux_inc, &mut state, &coeffs);
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

/// Steps all `N` stereo frames through smooth SONA v4.0 Selective-Duhem `ParaLRU` + stereo linking.
#[inline]
#[must_use]
pub fn run_paralru_duhem_v4_forward_tile<const N: usize>(
    u: [&[f32; N]; 2],
    bp: [&[f32; N]; 2],
    v_init: [f32; 2],
    params: ParaLruDuhemParams,
) -> ParaLruDuhemForwardOut<N> {
    run_paralru_duhem_v4_forward_partial_tile(u, bp, N, v_init, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// verify[dsp.op-n.commuted.dc-equilibrium]
    /// verify[dsp.op-n.core.matryoshka-blocks]
    /// verify[arch.fp.range-saturation]
    #[test]
    fn test_duhem_v4_neutral_remanence_and_extremes() {
        // 1. 0-ULP neutral mode & coercive remanence (eta=1, alpha_atk=1, flux_inc=0).
        let c_rem = SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(0.95),
            GpuVec4::splat(1.0),
            GpuVec4::splat(0.7),
            GpuVec4::splat(8.0),
            GpuVec4::splat(1.0),
            GpuVec4::splat(0.5),
        );
        let mut st_rem = InterleavedStereo8::<GpuVec4>::splat_lr(0.75, -0.5);
        let out_rem = step_paralru_duhem_v4_stereo8(
            InterleavedStereo8::splat(0.25),
            InterleavedStereo8::zero(),
            &mut st_rem,
            &c_rem,
        );
        assert_eq!(out_rem.into_lr().0.to_array(), [0.95 * 0.75; 4]);
        assert_eq!(out_rem.into_lr().1.to_array(), [0.95 * -0.5; 4]);

        // Neutral mode (gamma=0, eta=0) matches step_paralru_v4_vec4 to 0 ULP.
        let c_neutral = SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(0.92),
            GpuVec4::splat(0.35),
            GpuVec4::splat(0.6),
            GpuVec4::zero(),
            GpuVec4::zero(),
            GpuVec4::zero(),
        );
        let mut v_n1 = GpuVec4::from_array([0.4, -0.3, 0.0, 0.8]);
        let mut v_n2 = v_n1;
        let u_in = GpuVec4::from_array([0.7, -0.5, 0.2, 1.4]);
        let y_n1 = step_paralru_v4_vec4(
            u_in,
            &mut v_n1,
            c_neutral.alpha_rel,
            c_neutral.alpha_atk,
            c_neutral.kappa,
        );
        let y_n2 = step_paralru_duhem_v4_vec4(u_in, GpuVec4::splat(2.5), &mut v_n2, &c_neutral);
        assert_eq!(
            y_n1.to_array().map(f32::to_bits),
            y_n2.to_array().map(f32::to_bits)
        );

        // 2. ±1e9 / ±1e38 / NaN / ±Inf safety.
        let c_ext = SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(0.99),
            GpuVec4::splat(0.6),
            GpuVec4::splat(0.8),
            GpuVec4::splat(64.0),
            GpuVec4::splat(0.5),
            GpuVec4::splat(1.0),
        );
        let mut st_ext = InterleavedStereo8::<GpuVec4>::splat_lr(-1.0e9, 1.0e9);
        let out_ext = step_paralru_duhem_v4_stereo8(
            InterleavedStereo8::splat_lr(1.0e9, -1.0e9),
            InterleavedStereo8::splat_lr(1.0e38, f32::NAN),
            &mut st_ext,
            &c_ext,
        );
        let (el, er) = out_ext.into_lr();
        assert!(el.to_array().iter().all(|v| v.is_finite()));
        assert!(er.to_array().iter().all(|v| v.is_finite()));

        // 3. step_paralru_v4_stereo8 0-ULP parity and partial_tile valid_len == 0 edge case.
        let mut s4_l = GpuVec4::from_array([0.1, -0.2, 0.3, 0.0]);
        let mut s4_r = GpuVec4::from_array([-0.1, 0.4, 0.0, -0.5]);
        let mut s8 = InterleavedStereo8::from_lr(s4_l, s4_r);
        let u_s8 = InterleavedStereo8::load_lr(&[0.5, -0.3, 0.8, 0.0], &[-0.2, 0.6, 0.1, -0.4]);
        let y4_l = step_paralru_v4_vec4(
            u_s8.left,
            &mut s4_l,
            c_neutral.alpha_rel,
            c_neutral.alpha_atk,
            c_neutral.kappa,
        );
        let y4_r = step_paralru_v4_vec4(
            u_s8.right,
            &mut s4_r,
            c_neutral.alpha_rel,
            c_neutral.alpha_atk,
            c_neutral.kappa,
        );
        let y8 = step_paralru_v4_stereo8(
            u_s8,
            &mut s8,
            c_neutral.alpha_rel,
            c_neutral.alpha_atk,
            c_neutral.kappa,
        );
        assert_eq!(
            y8.into_lr().0.to_array().map(f32::to_bits),
            y4_l.to_array().map(f32::to_bits)
        );
        assert_eq!(
            y8.into_lr().1.to_array().map(f32::to_bits),
            y4_r.to_array().map(f32::to_bits)
        );

        let p_sat = ParaLruDuhemParams::clamped(0.9, 0.5, 0.4, 4.0, 0.3, 0.25, 2.0);
        let u_l = [0.8_f32; 4];
        let u_r = [-0.6_f32; 4];
        let bp_l = [0.5_f32; 4];
        let bp_r = [-0.5_f32; 4];
        let empty_fwd = run_paralru_duhem_v4_forward_partial_tile(
            [&u_l, &u_r],
            [&bp_l, &bp_r],
            0,
            [0.3, -0.4],
            p_sat,
        );
        assert_eq!(empty_fwd.v_end_l, 0.3);
        assert_eq!(empty_fwd.v_end_r, -0.4);
    }
}
