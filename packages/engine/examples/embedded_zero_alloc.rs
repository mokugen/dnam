//! Example 2: Embedded Zero-Copy Layer 0 Extraction, Scratch Load & Zero-Alloc Model Hot-Swap
//! (`examples/embedded_zero_alloc.rs`).
//!
//! Demonstrates real-time patterns for audio plugins and embedded DSP hardware:
//! 1. [`DnamEngine::from_bytes_with_scratch`]: Borrows a 64B-aligned `.dnam` container and
//!    caller-supplied scratch metadata with zero dynamic allocation in the audio loop.
//! 2. [`DnamEngine::with_max_capacity`] + [`DnamEngine::reload_model_in_place`]: Pre-allocates
//!    all headroom once at plugin startup and hot-swaps hybrid `.dnam` models with `0` heap
//!    allocations.
//! 3. `4x` Offline Bounce oversampling with zero allocations.
//! 4. Zero-copy [`extract_dnam_bin`] extraction driving [`dnam_vm::SonaVm`] directly at Layer 0.
//!
//! Run with:
//!   `cargo run -p dnam-engine --example embedded_zero_alloc`

use dnam_engine::container::{build_dnam_container_from_dnam_bin, extract_dnam_bin};
use dnam_engine::memory::{AlignedBuffer, AlignedFp16Block, CacheLine64};
use dnam_engine::schema::DnamJsonHeader;
use dnam_engine::{DnamEngine, Tier};
use dnam_kernels::f32_to_fp16;
use dnam_vm::SonaVm;
use dnam_vm::bin::{
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_AUTO_UNITY,
    DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, encode_dnam_bin,
};
use dnam_vm::math::{DefaultMathBackend, TANK_DC_BLOCKER_48K_F32};
use zerocopy::FromZeros;
use zerocopy::byteorder::little_endian::{F32, U16};

fn build_demo_model(name: &str, seed: u8) -> AlignedBuffer<CacheLine64> {
    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = name.to_string();

    let mut a0 = DnamBinBuildInput::default_arena0_lite();
    a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
    let gain = 1.0 + 0.01 * f32::from(seed);
    a0.linear_readout_lite.direct_s2_u_gain = [F32::new(gain), F32::new(gain)];
    let mut a1 = Arena1FullDeltaWire::new_zeroed();
    let mut a2 = Arena2ProDeltaWire::new_zeroed();
    let mut a3 = Arena3UltraDeltaWire::new_zeroed();
    let gain_fp16 = U16::new(f32_to_fp16(gain));
    a1.linear_readout_full.w_ro_and_mods_fp16[24 * 12 + 4..24 * 12 + 6].fill(gain_fp16);
    a2.linear_readout_pro.w_ro_and_mods_fp16[32 * 12 + 4..32 * 12 + 6].fill(gain_fp16);
    a3.linear_readout_ultra.w_ro_and_mods_fp16[40 * 12 + 4..40 * 12 + 6].fill(gain_fp16);

    let dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: header.sample_rate_train_hz,
        header_flags: DNBN_FLAG_STEREO_OUT | DNBN_FLAG_AUTO_UNITY,
        ext_ring_pow2_frames: 0,
        ext_ring_lines: 0,
        ext_ring_format: 0,
        declared_max_delay_48k: 128,
        dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
        tail_time_ms: header.io_config.tail_time_ms,
        ro_pole_48k: header.linear_readout.ro_pole_48k,
        switch_xfade_samples: header.transport.switch_xfade_samples,
        content_hash_prefix: [0; 8],
        controls: &[],
        cp1_overflow: &[],
        tsyn: &[],
        cext: None,
        arena0: &a0,
        arena1: Some(&a1),
        arena2: Some(&a2),
        arena3: Some(&a3),
    })
    .expect("valid Layer 0 .dnam-bin");

    let raw = build_dnam_container_from_dnam_bin(&mut header, &dnam_bin)
        .expect("valid Layer 1 hybrid .dnam container");
    let mut aligned = AlignedBuffer::<CacheLine64>::new_zeroed(raw.len() / 64);
    aligned.as_bytes_mut().copy_from_slice(&raw);
    aligned
}

fn main() {
    let model_a = build_demo_model("Vintage Clean Twin", 3);
    let model_b = build_demo_model("High-Gain Lead Stack", 19);

    // Pattern 1: Embedded zero-copy scratch signature compatibility
    let mut sram_delay_scratch = vec![AlignedFp16Block([0u16; 32]); 256];
    {
        let mut embedded_engine = DnamEngine::<DefaultMathBackend>::from_bytes_with_scratch(
            model_a.as_bytes(),
            &mut sram_delay_scratch,
            Tier::Lite,
            48_000,
        )
        .expect("borrowed arena + scratch initialized cleanly");

        let mono_in = [0.15_f32; 64];
        let mut mono_out = [0.0_f32; 64];
        embedded_engine.process_mono(&mono_in, &mut mono_out);
        println!(
            "[Embedded Scratch] Processed 64 mono samples | Sample[63] = {:.6}",
            mono_out[63]
        );
    }

    // Pattern 2: Pre-allocated max-capacity engine with zero-allocation in-place model hot-swap
    let mut hot_swap_engine = DnamEngine::<DefaultMathBackend>::with_max_capacity();
    hot_swap_engine
        .reload_model_in_place(model_a.as_bytes(), Tier::Full, 48_000)
        .expect("hot-swap Model A");

    let mut left_io = [0.2_f32; 64];
    let mut right_io = [-0.2_f32; 64];
    hot_swap_engine.process_block_in_place(&mut left_io, &mut right_io);
    println!(
        "[Hot-Swap Model A] Processed 64 in-place stereo samples | Out L[63] = {:.6}",
        left_io[63]
    );

    // Hot-swap to Model B at 96 kHz (Continuous-Time ZOH rescaling, 0 heap allocations)
    hot_swap_engine
        .reload_model_in_place(model_b.as_bytes(), Tier::Full, 96_000)
        .expect("hot-swap Model B @ 96 kHz");
    hot_swap_engine.process_block_in_place(&mut left_io, &mut right_io);
    println!(
        "[Hot-Swap Model B @ 96 kHz] Processed 64 in-place stereo samples | Out L[63] = {:.6}",
        left_io[63]
    );

    // Pattern 3: Enable 4x Offline Bounce oversampling with zero allocations
    hot_swap_engine.set_offline_bounce(true);
    hot_swap_engine.process_block_in_place(&mut left_io, &mut right_io);
    println!(
        "[4x Offline Bounce @ 96 kHz] Processed 64 in-place stereo samples | Out L[63] = {:.6}",
        left_io[63]
    );

    // Pattern 4: Direct Layer 0 zero-copy .dnam-bin extraction into SonaVm
    let dbin = extract_dnam_bin(model_a.as_bytes()).expect("zero-copy .dnam-bin extraction");
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(dbin, Tier::Lite, 48_000.0)
        .expect("SonaVm hydrated directly from extracted .dnam-bin");
    let (vm_l, vm_r) = vm.step_frame(0.25, -0.25).expect("step VM frame");
    println!(
        "[Layer 0 SonaVm] Extracted {} B .dnam-bin | Frame out L={:.6}, R={:.6}",
        dbin.len(),
        vm_l,
        vm_r,
    );
}
