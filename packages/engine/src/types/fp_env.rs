//! Hardware floating-point environment guard (`FTZ`/`DAZ`, `RNE`, and non-trapping exceptions).

#[cfg(any(test, target_arch = "aarch64"))]
#[inline(always)]
const fn audio_fpcr(previous: u64) -> u64 {
    let fiz_if_ah = (previous >> 1) & 1;
    (previous & !((0b11_u64 << 22) | 0x9f00)) | (1_u64 << 24) | fiz_if_ah
}

#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
const fn audio_mxcsr(previous: u32) -> u32 {
    (previous & !(0x6000 | 0x003f)) | 0x8040 | 0x1f80
}

/// Thread-bound RAII guard that selects round-to-nearest, ties-to-even, non-trapping
/// floating-point exceptions, and hardware denormal protection for an audio block
/// on `aarch64` and `x86_64`. On AArch64, it preserves `FPCR.AH` and sets
/// `FPCR.FIZ` when `AH` is set, enabling input flushing for most floating-point
/// instructions on CPUs with `FEAT_AFP`.
/// It restores the caller's control register on drop. On `x86_64`, entry clears
/// the `MXCSR` sticky exception flags, and drop restores their saved values.
/// On `aarch64`, exception status is held separately in `FPSR` and is not restored.
///
/// Engine block methods create this guard automatically. When calling per-sample
/// engine, operator, or stage methods, or control ticks directly, hold one guard
/// on the processing thread for the entire batch. Drop it before yielding or
/// calling unrelated host code. On other architectures this guard is a no-op;
/// the target must supply the required rounding behavior.
///
/// ```
/// use dnam_engine::{FtzDazGuard, operators::OpTDelayLine};
///
/// let mut delay = OpTDelayLine::<2048>::new();
/// let input = [0.0_f32; 16];
/// let mut output = [0.0_f32; 16];
/// {
///     let _audio = FtzDazGuard::new();
///     for (out, sample) in output.iter_mut().zip(input) {
///         *out = delay.step(sample);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct FtzDazGuard {
    #[cfg(target_arch = "aarch64")]
    prev_fpcr: u64,
    #[cfg(target_arch = "x86_64")]
    prev_mxcsr: u32,
    _thread_bound: core::marker::PhantomData<*mut ()>,
}

impl FtzDazGuard {
    /// Selects `RNE`, non-trapping exceptions and hardware `FTZ`/`DAZ` where available.
    #[inline(always)]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let prev_fpcr: u64;
            // SAFETY: The ARM64 FPCR is saved; only rounding, trap-enable, FZ and conditional FIZ bits change.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) prev_fpcr, options(nomem, nostack, preserves_flags));
                let new_fpcr = audio_fpcr(prev_fpcr);
                if new_fpcr != prev_fpcr {
                    core::arch::asm!("msr fpcr, {}", in(reg) new_fpcr, options(nomem, nostack, preserves_flags));
                }
            }
            Self {
                prev_fpcr,
                _thread_bound: core::marker::PhantomData,
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut prev_mxcsr: u32 = 0;
            // SAFETY: `stmxcsr` and `ldmxcsr` read/write the user-space SSE MXCSR control word.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut prev_mxcsr, options(nostack, preserves_flags));
                let new_mxcsr = audio_mxcsr(prev_mxcsr);
                if new_mxcsr != prev_mxcsr {
                    core::arch::asm!("ldmxcsr [{}]", in(reg) &new_mxcsr, options(nostack, preserves_flags));
                }
            }
            Self {
                prev_mxcsr,
                _thread_bound: core::marker::PhantomData,
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self {
                _thread_bound: core::marker::PhantomData,
            }
        }
    }
}

impl Default for FtzDazGuard {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FtzDazGuard {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            let cur_fpcr: u64;
            // SAFETY: Reads user-space FPCR and writes only if changed to avoid pipeline-flushing MSR when nested or already in audio mode.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) cur_fpcr, options(nomem, nostack, preserves_flags));
                if cur_fpcr != self.prev_fpcr {
                    core::arch::asm!("msr fpcr, {}", in(reg) self.prev_fpcr, options(nomem, nostack, preserves_flags));
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: Restores the saved MXCSR, including flags raised after entry.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &self.prev_mxcsr, options(nostack, preserves_flags));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tier;

    #[test]
    fn test_audio_fp_control_bit_transforms() {
        let fpcr_changed = (0b11_u64 << 22) | 0x9f00 | (1_u64 << 24) | 1;
        let fpcr_varied = fpcr_changed | 2;
        for preserved in [0, !fpcr_varied] {
            for traps in 0_u64..64 {
                for rounding in 0_u64..4 {
                    for denormal in 0_u64..2 {
                        for alternate in 0_u64..4 {
                            let host = preserved
                                | ((traps & 0x1f) << 8)
                                | ((traps & 0x20) << 10)
                                | (rounding << 22)
                                | (denormal << 24)
                                | alternate;
                            let active = audio_fpcr(host);
                            assert_eq!(active & 0x9f00, 0);
                            assert_eq!(active & (0b11 << 22), 0);
                            assert_ne!(active & (1 << 24), 0);
                            assert_eq!(
                                active & 1,
                                u64::from(alternate != 0),
                                "incoming AH/FIZ: {alternate:#04b}"
                            );
                            assert_eq!(active & !fpcr_changed, host & !fpcr_changed);
                            assert_eq!(audio_fpcr(active), active);
                        }
                    }
                }
            }
        }

        for masks in 0_u32..64 {
            for sticky in 0_u32..64 {
                for rounding in 0_u32..4 {
                    for denormal in 0_u32..4 {
                        let host = 0xa5a5_0000
                            | (masks << 7)
                            | sticky
                            | (rounding << 13)
                            | ((denormal & 1) << 6)
                            | ((denormal & 2) << 14);
                        let active = audio_mxcsr(host);
                        assert_eq!(active & 0x003f, 0);
                        assert_eq!(active & 0x1f80, 0x1f80);
                        assert_eq!(active & 0x6000, 0);
                        assert_eq!(active & 0x8040, 0x8040);
                        assert_eq!(active & 0xffff_0000, host & 0xffff_0000);
                    }
                }
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn read_fp_control() -> u64 {
        #[cfg(target_arch = "aarch64")]
        {
            let value: u64;
            // SAFETY: Reading the current thread's user-space FPCR does not change it.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack, preserves_flags));
            }
            value
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut value = 0_u32;
            // SAFETY: The pointer refers to a writable MXCSR-sized local value.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack, preserves_flags));
            }
            u64::from(value)
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn write_fp_control(value: u64) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: The value is derived from the current FPCR with defined bits changed.
            unsafe {
                core::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let value = value as u32;
            // SAFETY: The value is derived from the current MXCSR with defined bits changed.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack, preserves_flags));
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    struct RestoreFpControl(u64);

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    impl Drop for RestoreFpControl {
        fn drop(&mut self) {
            write_fp_control(self.0);
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn expected_audio_fp_control(host: u64) -> u64 {
        #[cfg(target_arch = "aarch64")]
        {
            let mut expected = (host & !((0b11_u64 << 22) | 0x9f00)) | (1_u64 << 24);
            if host & 0b10 != 0 {
                expected |= 0b01;
            }
            expected
        }
        #[cfg(target_arch = "x86_64")]
        {
            (host & !(0x6000 | 0x003f)) | 0x8040 | 0x1f80
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_fp_guard_selects_rne_and_restores_all_control_bits() {
        #[cfg(target_arch = "aarch64")]
        let (round_shift, denormal_bits, denormal_options, safe_original) = (
            22,
            1_u64 << 24,
            [0, 0, 1_u64 << 24, 1_u64 << 24],
            read_fp_control() & !0x9f00,
        );
        #[cfg(target_arch = "x86_64")]
        let (round_shift, denormal_bits, denormal_options, safe_original) = (
            13,
            0x8040_u64,
            [0, 0x0040, 0x8000, 0x8040],
            read_fp_control() | 0x1f80,
        );
        let _restore = RestoreFpControl(read_fp_control());
        let round_mask = 0b11_u64 << round_shift;

        for mode in 0..4 {
            for denormal in denormal_options {
                let requested = (safe_original & !(round_mask | denormal_bits))
                    | (mode << round_shift)
                    | denormal;
                write_fp_control(requested);
                let host = read_fp_control();
                assert_eq!(
                    host & (round_mask | denormal_bits),
                    requested & (round_mask | denormal_bits)
                );
                {
                    let _guard = FtzDazGuard::new();
                    let expected = expected_audio_fp_control(host);
                    assert_eq!(read_fp_control(), expected);
                    #[cfg(target_arch = "x86_64")]
                    write_fp_control(expected | 0x003f);
                    let outer = read_fp_control();
                    {
                        let _nested = FtzDazGuard::new();
                        assert_eq!(read_fp_control(), expected_audio_fp_control(outer));
                    }
                    assert_eq!(read_fp_control(), outer);

                    let one = std::hint::black_box(1.0_f32);
                    let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                    assert_eq!(
                        std::hint::black_box(one + half_ulp).to_bits(),
                        1.0_f32.to_bits()
                    );
                    let min_normal = std::hint::black_box(f32::MIN_POSITIVE);
                    let half = std::hint::black_box(0.5_f32);
                    assert_eq!(std::hint::black_box(min_normal * half).to_bits(), 0);
                    let subnormal = std::hint::black_box(f32::from_bits(0x007f_ffff));
                    let two = std::hint::black_box(2.0_f32);
                    assert_eq!(std::hint::black_box(subnormal * two).to_bits(), 0);
                }
                assert_eq!(read_fp_control(), host);
            }
        }

        write_fp_control(safe_original);
        let host = read_fp_control();
        let unwound = std::panic::catch_unwind(|| {
            let _guard = FtzDazGuard::new();
            panic!("exercise floating-point guard unwinding");
        });
        assert!(unwound.is_err());
        assert_eq!(read_fp_control(), host);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_fp_guard_restores_changes_after_unchanged_entry() {
        let original = read_fp_control();
        let _restore = RestoreFpControl(original);
        write_fp_control(expected_audio_fp_control(original));
        let host = read_fp_control();
        assert_eq!(expected_audio_fp_control(host), host);

        {
            let _guard = FtzDazGuard::new();
            assert_eq!(read_fp_control(), host);
            #[cfg(target_arch = "x86_64")]
            {
                let one = std::hint::black_box(1.0_f32);
                let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                assert_eq!(
                    std::hint::black_box(one + half_ulp).to_bits(),
                    1.0_f32.to_bits()
                );
                assert_ne!(read_fp_control() & 0x0020, 0);
            }
            #[cfg(target_arch = "aarch64")]
            {
                write_fp_control(host ^ (1 << 24));
                assert_ne!(read_fp_control(), host);
            }
        }
        assert_eq!(read_fp_control(), host);

        {
            let _guard = FtzDazGuard::new();
            #[cfg(target_arch = "aarch64")]
            let changed = host ^ (1 << 22);
            #[cfg(target_arch = "x86_64")]
            let changed = host ^ (1 << 13);
            write_fp_control(changed);
            assert_eq!(read_fp_control(), changed);
        }
        assert_eq!(read_fp_control(), host);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_standalone_sample_batch_uses_caller_fp_guard() {
        use crate::engine::DnamEngine;
        use crate::math::ScalarBackend;
        use crate::operators::SymplecticModal4Lane;
        use dnam_kernels::T1DcBlockerKernel;

        let original = read_fp_control();
        let _restore = RestoreFpControl(original);
        let mut engine = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        let mut modal = std::hint::black_box(SymplecticModal4Lane::new());
        let mut dc = std::hint::black_box(T1DcBlockerKernel::new());
        #[cfg(target_arch = "aarch64")]
        let requested = (original & !(0x9f00 | (0b11 << 22) | (1 << 24))) | (0b01 << 22);
        #[cfg(target_arch = "x86_64")]
        let requested = ((original | 0x1f80) & !(0x8040 | 0x003f | (0b11 << 13))) | (0b10 << 13);
        write_fp_control(requested);
        let host = read_fp_control();

        {
            let _guard = FtzDazGuard::new();
            for _ in 0..4 {
                engine.ctrl_sub_counter = 0;
                engine.ctrl_energy_accum = std::hint::black_box(1.0);
                let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                let _ = engine.step_frame_sona(half_ulp, half_ulp, false);
                assert_eq!(engine.ctrl_energy_accum.to_bits(), 1.0_f32.to_bits());

                engine.control_bus.env_poles[0] = std::hint::black_box(0.5);
                engine.control_bus.env_states[0] = std::hint::black_box(2.0);
                let _ = engine
                    .control_bus
                    .tick_3khz(std::hint::black_box(f32::EPSILON), &[]);
                assert_eq!(
                    engine.control_bus.env_states[0].to_bits(),
                    1.0_f32.to_bits()
                );

                let tiny = std::hint::black_box(f32::from_bits(1));
                assert_eq!(modal.step_lane(0, tiny).0.to_bits(), 0);
                assert_eq!(dc.step(std::hint::black_box(0.0)).to_bits(), 0);
            }
            let active = read_fp_control();
            #[cfg(target_arch = "aarch64")]
            assert_eq!(active, expected_audio_fp_control(host));
            #[cfg(target_arch = "x86_64")]
            assert_eq!(active & !0x003f, expected_audio_fp_control(host));
        }
        assert_eq!(read_fp_control(), host);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_fp_guard_masks_exceptions_and_restores_host_status() {
        const CHILD_ENV: &str = "DNAM_FP_GUARD_EXCEPTION_PROBE";
        if std::env::var_os(CHILD_ENV).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "types::fp_env::tests::test_fp_guard_masks_exceptions_and_restores_host_status",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "floating-point exception probe failed: {}\n{}\n{}",
                output.status,
                std::string::String::from_utf8_lossy(&output.stdout),
                std::string::String::from_utf8_lossy(&output.stderr)
            );
            std::eprint!("{}", std::string::String::from_utf8_lossy(&output.stderr));
            return;
        }

        let original = read_fp_control();
        let _restore = RestoreFpControl(original);
        #[cfg(target_arch = "aarch64")]
        let (safe, mask_bits) = (
            original & !0x9f00,
            [0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x8000],
        );
        #[cfg(target_arch = "x86_64")]
        let (safe, mask_bits) = (
            original | 0x1f80,
            [0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000],
        );

        let mut writable_traps = 0;
        for pattern in 0_u64..64 {
            let mask = mask_bits
                .iter()
                .enumerate()
                .fold(0, |acc, (i, bit)| acc | (bit * ((pattern >> i) & 1)));
            #[cfg(target_arch = "aarch64")]
            let requested = safe | mask;
            #[cfg(target_arch = "x86_64")]
            let requested = ((safe & !0x003f) | pattern) & !mask;
            write_fp_control(requested);
            let host = read_fp_control();
            let (active, after_arithmetic, nested_active, nested_restored, results) = {
                let _guard = FtzDazGuard::new();
                let active = read_fp_control();
                let zero = std::hint::black_box(0.0_f32);
                let one = std::hint::black_box(1.0_f32);
                let two = std::hint::black_box(2.0_f32);
                let min = std::hint::black_box(f32::MIN_POSITIVE);
                let tiny = std::hint::black_box(f32::from_bits(1));
                let tenth = std::hint::black_box(0.1_f32);
                let half_ulp = std::hint::black_box(f32::from_bits(0x3380_0000));
                let maximum = std::hint::black_box(f32::MAX);
                let results = [
                    std::hint::black_box(zero / zero).to_bits(),
                    std::hint::black_box(one / zero).to_bits(),
                    std::hint::black_box(maximum * two).to_bits(),
                    std::hint::black_box(min * tenth).to_bits(),
                    std::hint::black_box(one + half_ulp).to_bits(),
                    std::hint::black_box(tiny * two).to_bits(),
                ];
                let after_arithmetic = read_fp_control();
                let nested_active = {
                    let _nested = FtzDazGuard::new();
                    read_fp_control()
                };
                (
                    active,
                    after_arithmetic,
                    nested_active,
                    read_fp_control(),
                    results,
                )
            };
            let restored = read_fp_control();
            write_fp_control(safe);

            assert_eq!(active, expected_audio_fp_control(host));
            assert_eq!(nested_active, expected_audio_fp_control(after_arithmetic));
            assert_eq!(nested_restored, after_arithmetic);
            assert_eq!(restored, host);
            #[cfg(target_arch = "x86_64")]
            {
                writable_traps |= !host & mask;
                assert_eq!(host & 0x003f, pattern);
                assert_eq!(active & 0x003f, 0);
                assert_ne!(after_arithmetic & 0x003f, 0);
            }
            #[cfg(target_arch = "aarch64")]
            {
                writable_traps |= host & 0x9f00;
            }
            assert_eq!(results[0] & 0x7f80_0000, 0x7f80_0000);
            assert_ne!(results[0] & 0x007f_ffff, 0);
            assert_eq!(results[1], f32::INFINITY.to_bits());
            assert_eq!(results[2], f32::INFINITY.to_bits());
            assert_eq!(results[3], 0);
            assert_eq!(results[4], 1.0_f32.to_bits());
            assert_eq!(results[5], 0);
        }
        #[cfg(target_arch = "aarch64")]
        std::eprintln!("writable AArch64 FPCR trap-enable bits: {writable_traps:#06x}/0x9f00");
        #[cfg(target_arch = "x86_64")]
        std::eprintln!("writable x86-64 MXCSR trap-enable bits: {writable_traps:#06x}/0x1f80");

        let hook = std::panic::take_hook();
        std::panic::set_hook(std::boxed::Box::new(|_| {}));
        #[cfg(target_arch = "aarch64")]
        let requested = safe | 0x9f00;
        #[cfg(target_arch = "x86_64")]
        let requested = (safe | 0x003f) & !0x1f80;
        write_fp_control(requested);
        let host = read_fp_control();
        let unwound = std::panic::catch_unwind(|| {
            let _guard = FtzDazGuard::new();
            panic!("exercise floating-point guard unwinding with host traps enabled");
        });
        let restored = read_fp_control();
        write_fp_control(safe);
        std::panic::set_hook(hook);
        assert!(unwound.is_err());
        assert_eq!(restored, host);

        fn check_callbacks<B: crate::engine::EngineBackend>(safe: u64) {
            use crate::engine::DnamEngine;

            let mut engine = DnamEngine::<B>::new_empty(Tier::Lite);
            for bypassed in [false, true] {
                for entry in 0..4 {
                    engine.set_bypass(bypassed, false);
                    engine.ctrl_sub_counter = 0;
                    engine.ctrl_energy_accum = 0.0;
                    let input = [std::hint::black_box(f32::MAX), std::hint::black_box(0.1)];
                    let mut left = input;
                    let mut right = input;
                    #[cfg(target_arch = "aarch64")]
                    let requested = safe | 0x9f00;
                    #[cfg(target_arch = "x86_64")]
                    let requested = (safe | 0x003f) & !0x1f80;
                    write_fp_control(requested);
                    let host = read_fp_control();
                    match entry {
                        0 => engine.process_block(&input, &input, &mut left, &mut right),
                        1 => engine.process_mono_to_stereo(&input, &mut left, &mut right),
                        2 => engine.process_mono(&input, &mut left),
                        _ => engine.process_block_in_place(&mut left, &mut right),
                    }
                    let restored = read_fp_control();
                    write_fp_control(safe);
                    assert_eq!(restored, host);
                }
            }
        }

        check_callbacks::<crate::math::ScalarBackend>(safe);
        check_callbacks::<crate::math::DefaultMathBackend>(safe);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_block_entry_points_restore_fp_control_on_normal_and_early_exit() {
        use crate::engine::DnamEngine;
        use crate::math::ScalarBackend;

        #[cfg(target_arch = "aarch64")]
        let requested = (read_fp_control() & !(0x9f00 | (0b11 << 22) | (1 << 24))) | (0b01 << 22);
        #[cfg(target_arch = "x86_64")]
        let requested = ((read_fp_control() | 0x1f80) & !(0x8040 | (0b11 << 13))) | (0b01 << 13);
        let _restore = RestoreFpControl(read_fp_control());
        let mut engine = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        let input = [0.25_f32, -0.5];
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];

        for bypassed in [false, true] {
            engine.set_bypass(bypassed, false);
            write_fp_control(requested);
            let host = read_fp_control();
            engine.process_block(&input, &input, &mut left, &mut right);
            assert_eq!(read_fp_control(), host);
            engine.process_mono_to_stereo(&input, &mut left, &mut right);
            assert_eq!(read_fp_control(), host);
            engine.process_mono(&input, &mut left);
            assert_eq!(read_fp_control(), host);
            engine.process_block_in_place(&mut left, &mut right);
            assert_eq!(read_fp_control(), host);
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn test_active_block_entry_points_select_rne_and_denormal_mode() {
        use crate::engine::{DnamEngine, EngineBackend};
        use crate::math::{DefaultMathBackend, ScalarBackend};

        fn check_backend<B: EngineBackend>(requested: u64) {
            let mut engine = DnamEngine::<B>::new_empty(Tier::Lite);
            for entry in 0..4 {
                for (sample, initial, expected_bits) in [
                    (f32::from_bits(0x3380_0000), 1.0_f32, 1.0_f32.to_bits()),
                    (f32::from_bits(0x0040_0000), 0.0_f32, 0_u32),
                ] {
                    engine.ctrl_sub_counter = 0;
                    engine.ctrl_energy_accum = std::hint::black_box(initial);
                    let input = [std::hint::black_box(sample)];
                    let mut left = input;
                    let mut right = input;
                    write_fp_control(requested);
                    let host = read_fp_control();
                    match entry {
                        0 => engine.process_block(&input, &input, &mut left, &mut right),
                        1 => engine.process_mono_to_stereo(&input, &mut left, &mut right),
                        2 => engine.process_mono(&input, &mut left),
                        _ => engine.process_block_in_place(&mut left, &mut right),
                    }
                    assert_eq!(read_fp_control(), host);
                    assert_eq!(engine.ctrl_energy_accum.to_bits(), expected_bits);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        let requested = (read_fp_control() & !(0x9f00 | (0b11 << 22) | (1 << 24))) | (0b01 << 22);
        #[cfg(target_arch = "x86_64")]
        let requested = ((read_fp_control() | 0x1f80) & !(0x8040 | (0b11 << 13))) | (0b10 << 13);
        let _restore = RestoreFpControl(read_fp_control());
        check_backend::<ScalarBackend>(requested);
        check_backend::<DefaultMathBackend>(requested);
    }
}
