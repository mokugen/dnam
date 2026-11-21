//! Example 5: Host-Shell Patchbay & Standalone `SONA` Zero-`FDIV` Operator Chain
//! (`examples/custom_operator_chain.rs`).
//!
//! Demonstrates composing Layer 1 [`HostShellPatchbay`] with shared `dnam-kernels` and `dnam-vm`
//! 32-bit operators inside a caller-held [`FtzDazGuard`]:
//! - `HostShellPatchbay` (`IoRoutingMode::FourCableMethod`, zero runtime crossovers, structural
//!   `S_{ret->send} == 0_{2x2}` isolation, and voltage-normalized M/S width)
//! - `Op-P` (`T1DcBlockerKernel` + `HilbertH1Kernel` + `T1SubHarmonicKernel` + `PhasorQ32Bank4`)
//! - `Q0_192Phase` + `contractive_rotor_f32` (`c^2 + s^2 <= 1.0`)
//! - `Op-N` stereo `CoreQuad` (`step_core_quad_stereo8` on `InterleavedStereo8<GpuVec4>`) +
//!   `eval_c2_wavefolder_adaa_vec4`
//! - `Op-M` 4-pole stereo Kautz modal cabinet resonator (`step_kautz_quad_vec4`)
//!
//! Run with:
//!   `cargo run -p dnam-engine --example custom_operator_chain`

use dnam_engine::engine::patchbay::IoRoutingMode;
use dnam_engine::{FtzDazGuard, HostShellPatchbay};
use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, HilbertH1Kernel, InterleavedStereo8, KautzQuadState,
    KernelVec4, PhasorQ32Bank4, SonaBlockStateStereo8, SonaParaLruDuhemCoeffs4, SonaToneWeights4,
    T1DcBlockerKernel, T1SubHarmonicKernel, TptSvfCoeffs4, compute_op_p_and_phasor_pair,
    eval_c2_wavefolder_adaa_vec4, step_core_quad_stereo8, step_kautz_quad_vec4,
};
use dnam_vm::math::{Q0_192Phase, contractive_rotor_f32};

fn main() {
    let _audio_guard = FtzDazGuard::new();

    // 1. Layer 1 HostShellPatchbay in 4-Cable Method with structural S_{ret->send} == 0 isolation
    let mut patchbay = HostShellPatchbay::new();
    patchbay.set_routing_mode(IoRoutingMode::FourCableMethod);
    patchbay.set_ms_width(1.15);

    // 2. Op-P top-of-frame analytic phase tracker & Q0.32 phasor bank
    let mut dc_blocker = T1DcBlockerKernel::new();
    let mut hilbert = HilbertH1Kernel::new();
    let mut sub_tracker = T1SubHarmonicKernel::new();
    let mut phasors = PhasorQ32Bank4::from_freqs_hz([1.5, 3.0, 4.5, 6.0], 1.0 / 48_000.0);
    let mut lfo_phase = Q0_192Phase::default();

    // 3. Op-N 4-lane stereo CoreQuad (TPT-SVF -> Selective-Duhem ParaLRU -> PolyADAA1 + GLU)
    let svf_g = GpuVec4([0.08, 0.16, 0.32, 0.55]);
    let svf_k = GpuVec4([0.707, 0.707, 0.707, 0.707]);
    let core_params = CoreQuadParamsStereo4 {
        svf: TptSvfCoeffs4::from_gk(svf_g, svf_k),
        tone: SonaToneWeights4::lp_only(),
        lru: SonaParaLruDuhemCoeffs4::from_clamped(
            GpuVec4::splat(0.95),
            GpuVec4::splat(0.25),
            GpuVec4::splat(0.35),
            GpuVec4::splat(1.2),
            GpuVec4::splat(0.5),
            GpuVec4::splat(0.25),
        ),
        two_svf_g: svf_g.add(svf_g),
        rect_sign: GpuVec4::splat(1.0),
        sag_depth: GpuVec4::splat(0.1),
        adaa_drive: GpuVec4::splat(1.25),
        lru_bias_drive: GpuVec4::splat(0.05),
        wavefolder_mix: GpuVec4::splat(0.2),
        glu_mix: GpuVec4::splat(0.15),
    };
    let mut core_state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    let mut prev_wf_in = GpuVec4::zero();

    // 4. Op-M 4-pole stereo Kautz modal bank
    let mut kautz = KautzQuadState::new(
        GpuVec4([0.96, 0.94, 0.91, 0.88]),
        GpuVec4([0.12, 0.24, 0.36, 0.48]),
        GpuVec4::splat(0.25),
        GpuVec4::splat(0.22),
        GpuVec4::splat(-0.08),
        GpuVec4::splat(0.20),
        GpuVec4::splat(0.07),
    );

    let mut peak_out = 0.0_f32;
    let mut max_rotor_norm = 0.0_f32;

    for n in 0..256 {
        let inst_l = 0.35 * ((n as f32) * 0.09).sin();
        let inst_r = 0.35 * ((n as f32) * 0.09).cos();
        let ret_l = 0.85 * inst_l;
        let ret_r = 0.85 * inst_r;

        let (vm_in, send_out, dry_tap) = patchbay.route_pre_vm([inst_l, inst_r, ret_l, ret_r]);

        lfo_phase.step_angle_bits(0.015_f32.to_bits()).unwrap();
        let (lfo_cos, lfo_sin) = contractive_rotor_f32((0.015 * (n as f32)) % 3.0).unwrap();
        max_rotor_norm = max_rotor_norm.max(lfo_cos * lfo_cos + lfo_sin * lfo_sin);

        let op_p = compute_op_p_and_phasor_pair(
            &mut dc_blocker,
            &mut hilbert,
            &mut sub_tracker,
            &mut phasors,
            vm_in[0] + 0.02 * lfo_sin,
        );

        let core_in_l = GpuVec4::splat(vm_in[0] + 0.10 * op_p.features[0]);
        let core_in_r = GpuVec4::splat(vm_in[1] + 0.10 * op_p.features[1]);
        let core_out = step_core_quad_stereo8(
            InterleavedStereo8 {
                left: core_in_l,
                right: core_in_r,
            },
            &mut core_state,
            &core_params,
        );
        let (wf_l, wf_r) = eval_c2_wavefolder_adaa_vec4(prev_wf_in, core_out.adaa_odd.left);
        prev_wf_in = core_out.adaa_odd.left;

        let drive_l = 0.25 * (core_out.adaa_odd.left.add(wf_l)).to_array()[0];
        let drive_r = 0.25 * (core_out.adaa_odd.right.add(wf_r)).to_array()[0];
        let vm_out = step_kautz_quad_vec4(&mut kautz, drive_l, drive_r);

        let ports_out = patchbay.route_post_vm([vm_out.0, vm_out.1], dry_tap, send_out);
        peak_out = peak_out.max(ports_out[2].abs()).max(ports_out[3].abs());
    }

    println!(
        "Patchbay + SONA kernel chain processed 256 frames | Peak Main Out: {:.6} | Max Rotor Norm (<= 1.0): {:.8}",
        peak_out, max_rotor_norm,
    );
    assert!(max_rotor_norm <= 1.0);
    assert!(peak_out.is_finite() && peak_out > 0.0);
}
