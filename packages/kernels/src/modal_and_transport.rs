//! `Op-M` / `Op-P` / `Op-T` modal, quadrature phase-tracker, and causal delay-transport kernels
//! (`ref[sdk.rust.shared-nostd-kernels]`, `ref[dsp.op-m.hilbert.lattice]`,
//! `ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-t.reader.catmull-rom]`).
//!
//! All modules are `#![no_std]`, `#![forbid(unsafe_code)]`, strictly 32-bit, zero-`libm`, and
//! `0-FDIV`/`0-FSQRT`/`0-UDIV` on runtime DSP paths.

pub mod kautz;
pub mod op_p;
pub mod op_t;

pub use kautz::{
    ComplexModalScanElem, KautzQuadState, PhasorQ32Bank4, algebraic_unit_rotor_f32,
    compose_complex_modal_scan, eval_kautz_sweep_contraction_f32, eval_q32_quadrature_f32,
    step_kautz_quad_stereo8, step_kautz_quad_vec4,
};
pub use op_p::{
    CarrierBasisWeightsVec4, HILBERT_ALPHA_I, HILBERT_ALPHA_I_BITS, HILBERT_ALPHA_Q,
    HILBERT_ALPHA_Q_BITS, HILBERT_H1_ANCHOR_BITS, HILBERT_H1_ANCHOR_RATES_HZ, HILBERT_H1_STAGES,
    HilbertH1Kernel, HilbertH1RemapHistory, HilbertH1StereoKernel, StereoAnalyticQuadrature,
    StereoCarrierRotorVec4, T1_PLUS_DEFAULT_SUB_LPF_96K_F32, T1_PLUS_DEFAULT_TRK_LPF_96K_F32,
    T1DcBlockerKernel, T1DcBlockerStereo, T1PlusKernel, T1PlusParams, T1SubHarmonicKernel,
    eval_carrier_3basis_stereo_vec4, eval_t1_plus_soft_square_f32, halve_signed_q32_increments,
    remap_allpass_chain_states_from_outputs, rotate_carrier_stereo_phase_vec4,
    tracker_crosses_branch_cut_f32,
};
#[cfg(not(target_arch = "spirv"))]
pub use op_p::{
    compute_t1_dc_blocker_coeffs_bits, compute_t1_dc_blocker_coeffs_clamped,
    interpolate_hilbert_h1_alphas, interpolate_hilbert_h1_alphas_f64,
};
pub use op_t::{
    FP16_FINITE_MAX_F32, OpTFeedbackParams, OpTFeedbackState, QUAD_FLAG_HAS_EXT,
    QUAD_FLAG_HOST_WRITE_LOCK, apply_so4_even_chunk4, apply_so4_even_transpose_vec4,
    apply_so4_even_vec4, apply_so4_odd_chunk4, apply_so4_odd_transpose_vec4, apply_so4_odd_vec4,
    blend_catmull_rom_modulated_f32, bspline_barberpole_3phase_f32, catmull_rom_hermite_f32,
    clamp_transport_delay_f32, coercive_loop_limit_vec4, compute_half_alpha_cubic_f32,
    compute_op_t_mu_win_f32, compute_op_t_pitch_down_g_loop_eff, cubic_lagrange_weights_f32,
    eval_op_t_diagonal_window_f32, eval_op_t_diagonal_window_vec4,
    hermite_hann_partition_of_unity_f32, is_so4_identity, q32_smooth_sin_f32,
    read_barberpole_m3_f32, read_catmull_rom_f32, read_catmull_rom_fp16_u16,
    read_pitch_grain_m2_f32, so4_theta_grad_vec4, step_op_t_feedback_vec4,
    step_op_t_wet_dc_stereo_f32, step_schroeder_allpass, step_schroeder_quad_vec4,
    step_we_hold_lag, step_we_hold_lag_u16, wgsl_triangle_fold_q32, write_ring_with_freeze_f32,
    write_ring_with_we_f32,
};

/// Top-of-frame analytic `Op-P` quadrature, `4`-channel `T1` sub-harmonic feature vector, and
/// `4`-lane `Q0.32` crossbar carrier outputs produced by [`compute_op_p_and_phasor_pair`].
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct OpPAndPhasorOutput {
    /// Upstream `5 Hz` DC-blocked Hilbert `I`-branch quadrature sample.
    pub hilbert_i: f32,
    /// Upstream `5 Hz` DC-blocked Hilbert `Q`-branch quadrature sample (`~90 deg` delayed relative
    /// to `hilbert_i`).
    pub hilbert_q: f32,
    /// Canonical `4`-channel `T1` sub-harmonic feature vector
    /// `[sub1_cos * env_gate, sub1_sin * env_gate, sub1_sq * env_gate, sub2_sq * env_gate]`
    /// (`ref[dsp.op-m.tracker.toggle-sheet]`).
    pub features: [f32; 4],
    /// 4-lane `Q0.32` quadrature cosine carrier values at this sample tick.
    pub phasor_cos: [f32; 4],
    /// 4-lane `Q0.32` quadrature sine carrier values at this sample tick.
    pub phasor_sin: [f32; 4],
}

/// Evaluates one top-of-frame `Op-P` (`T1DcBlockerKernel` -> `HilbertH1Kernel` ->
/// `T1SubHarmonicKernel`) + `4`-lane `Q0.32` crossbar phasor tick cleanly in 32-bit arithmetic
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.hilbert.lattice]`,
/// `ref[dsp.op-m.rotor.q0-192-lfo]`).
#[inline(always)]
pub fn compute_op_p_and_phasor_pair(
    dc_blocker: &mut T1DcBlockerKernel,
    hilbert: &mut HilbertH1Kernel,
    tracker: &mut T1SubHarmonicKernel,
    phasors: &mut PhasorQ32Bank4,
    x_in: f32,
) -> OpPAndPhasorOutput {
    let x_dc = dc_blocker.step(x_in);
    let (hilbert_i, hilbert_q) = hilbert.step(x_dc);
    let features = tracker.step(hilbert_i, hilbert_q);
    let (phasor_cos, phasor_sin) = phasors.step_quadrature();
    OpPAndPhasorOutput {
        hilbert_i,
        hilbert_q,
        features,
        phasor_cos,
        phasor_sin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8, KernelVec4};
    use proptest::prelude::*;

    /// verify[dsp.crossbar.top-of-frame]
    /// verify[dsp.op-t.diffuser.recurrence]
    /// verify[dsp.op-t.window.m2-pitch]
    /// verify[dsp.op-t.window.m3-barberpole]
    #[test]
    fn test_compute_op_p_and_phasor_pair_with_op_t_and_kautz_pipeline() {
        let mut dc_blocker = T1DcBlockerKernel::new();
        let mut hilbert = HilbertH1Kernel::new();
        let mut tracker = T1SubHarmonicKernel::new();
        let mut phasors = PhasorQ32Bank4::from_freqs_hz([0.5, 5.0, 220.0, 1_500.0], 1.0 / 48_000.0);

        let mut ring_l = [0.0_f32; 64];
        let mut ring_ap = [0.0_f32; 32];
        let mut kautz = KautzQuadState::new(
            GpuVec4::splat(0.85),
            GpuVec4::splat(0.20),
            GpuVec4::splat(0.15),
            GpuVec4::splat(0.40),
            GpuVec4::splat(0.05),
            GpuVec4::splat(0.40),
            GpuVec4::splat(-0.05),
        );

        let mut freeze_snapshot = [0.0_f32; 64];
        for n in 0..256 {
            let phase = (n as f64) * core::f64::consts::TAU * 440.0 / 48_000.0;
            let x = 0.5 * (phase.sin() as f32) + 0.125;
            let op_p = compute_op_p_and_phasor_pair(
                &mut dc_blocker,
                &mut hilbert,
                &mut tracker,
                &mut phasors,
                x,
            );
            assert!(op_p.hilbert_i.is_finite());
            assert!(op_p.hilbert_q.is_finite());
            for feat in op_p.features {
                assert!(feat.is_finite() && feat.abs() <= 1.25);
            }

            for lane in 0..4 {
                let c = op_p.phasor_cos[lane];
                let s = op_p.phasor_sin[lane];
                assert!(((c * c + s * s) - 1.0).abs() < 1.0e-3);
            }

            let delay_tau =
                clamp_transport_delay_f32(16.0 + 4.0 * op_p.phasor_sin[0], ring_l.len(), true);
            let delayed = read_barberpole_m3_f32(
                &ring_l,
                63,
                n,
                delay_tau,
                8.0,
                0.5 + 0.4 * op_p.phasor_sin[0],
            );
            let diffused = step_schroeder_allpass(delayed, &mut ring_ap, 31, n, 7, 0.5);
            let limited = coercive_loop_limit_vec4(
                GpuVec4::splat(x + 0.5 * diffused),
                GpuVec4::splat(0.8),
                GpuVec4::splat(0.95),
            )
            .to_array()[0];

            let freeze = n >= 200;
            if n == 200 {
                freeze_snapshot = ring_l;
            }
            write_ring_with_freeze_f32(&mut ring_l, 63, n, limited, freeze);
            if freeze {
                assert_eq!(ring_l.map(f32::to_bits), freeze_snapshot.map(f32::to_bits));
            }

            let (yl, yr) = step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(
                &mut kautz,
                op_p.features[0],
                diffused,
            );
            assert!(yl.is_finite() && yr.is_finite());
        }
    }

    proptest! {
        #[test]
        fn prop_op_p_and_phasor_crossbar_remains_bounded_and_finite(
            freq_hz in 10.0_f32..5_000.0_f32,
            drive in 0.0_f32..1.5_f32,
        ) {
            let mut dc_blocker = T1DcBlockerKernel::default();
            let mut hilbert = HilbertH1Kernel::default();
            let mut tracker = T1SubHarmonicKernel::default();
            let mut phasors =
                PhasorQ32Bank4::from_freqs_hz([0.05, 12.0, freq_hz, 12_000.0], 1.0 / 48_000.0);

            for n in 0..32 {
                let rad = (n as f64) * core::f64::consts::TAU * f64::from(freq_hz) / 48_000.0;
                let x = drive * (rad.cos() as f32);
                let out = compute_op_p_and_phasor_pair(
                    &mut dc_blocker,
                    &mut hilbert,
                    &mut tracker,
                    &mut phasors,
                    x,
                );
                prop_assert!(out.hilbert_i.is_finite());
                prop_assert!(out.hilbert_q.is_finite());
                for feat in out.features {
                    prop_assert!(feat.is_finite());
                }
                for lane in 0..4 {
                    let norm2 = out.phasor_cos[lane] * out.phasor_cos[lane]
                        + out.phasor_sin[lane] * out.phasor_sin[lane];
                    prop_assert!((norm2 - 1.0).abs() < 1.0e-3);
                }
            }
        }
    }
}
