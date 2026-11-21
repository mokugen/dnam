//! Hardware floating-point environment guard (`FTZ`/`DAZ`, `RNE`, and non-trapping exceptions).

#[cfg(any(test, target_arch = "aarch64"))]
#[inline(always)]
const fn audio_fpcr(previous: u64) -> u64 {
    let fiz_if_ah = (previous >> 1) & 1;
    (previous & !((0b11_u64 << 22) | (1_u64 << 19) | 0x9f00)) | (1_u64 << 24) | fiz_if_ah
}

#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
const fn audio_mxcsr(previous: u32) -> u32 {
    (previous & !(0x6000 | 0x003f)) | 0x8040 | 0x1f80
}

/// Thread-bound RAII guard selecting round-to-nearest ties-to-even (`RNE`), non-trapping
/// exceptions, and hardware `FTZ`/`DAZ` flush-to-zero on `aarch64` and `x86_64`.
#[derive(Debug)]
pub struct FtzDazGuard {
    #[cfg(target_arch = "aarch64")]
    prev_fpcr: u64,
    #[cfg(target_arch = "x86_64")]
    prev_mxcsr: u32,
    _thread_bound: core::marker::PhantomData<*mut ()>,
}

impl FtzDazGuard {
    /// Enters `RNE` + non-trapping + `FTZ`/`DAZ` floating-point mode and saves caller CSR bits.
    #[inline(always)]
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            let prev_fpcr: u64;
            // SAFETY: Saves caller AArch64 FPCR and sets RNE, non-trapping, FZ, and conditional FIZ.
            unsafe {
                core::arch::asm!(
                    "mrs {}, fpcr",
                    out(reg) prev_fpcr,
                    options(nomem, nostack, preserves_flags),
                );
                let new_fpcr = audio_fpcr(prev_fpcr);
                if new_fpcr != prev_fpcr {
                    core::arch::asm!(
                        "msr fpcr, {}",
                        in(reg) new_fpcr,
                        options(nomem, nostack, preserves_flags),
                    );
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
            // SAFETY: `&raw mut prev_mxcsr` and `&raw const new_mxcsr` point to valid initialized stack `u32`s.
            unsafe {
                core::arch::asm!(
                    "stmxcsr [{}]",
                    in(reg) &raw mut prev_mxcsr,
                    options(nostack, preserves_flags),
                );
                let new_mxcsr = audio_mxcsr(prev_mxcsr);
                if new_mxcsr != prev_mxcsr {
                    core::arch::asm!(
                        "ldmxcsr [{}]",
                        in(reg) &raw const new_mxcsr,
                        options(nostack, preserves_flags),
                    );
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
            // SAFETY: Restores caller FPCR only when modified to skip redundant pipeline flushes.
            unsafe {
                core::arch::asm!(
                    "mrs {}, fpcr",
                    out(reg) cur_fpcr,
                    options(nomem, nostack, preserves_flags),
                );
                if cur_fpcr != self.prev_fpcr {
                    core::arch::asm!(
                        "msr fpcr, {}",
                        in(reg) self.prev_fpcr,
                        options(nomem, nostack, preserves_flags),
                    );
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: Restores saved MXCSR word including caller flags.
            unsafe {
                core::arch::asm!(
                    "ldmxcsr [{}]",
                    in(reg) &raw const self.prev_mxcsr,
                    options(nostack, preserves_flags),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_fp_control_bit_transforms() {
        let fpcr_changed = (0b11_u64 << 22) | (1_u64 << 19) | 0x9f00 | (1_u64 << 24) | 1;
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
                            assert_eq!(active & (1 << 19), 0);
                            assert_ne!(active & (1 << 24), 0);
                            assert_eq!(active & 1, u64::from(alternate != 0));
                            assert_eq!(active & !fpcr_changed, host & !fpcr_changed);
                            assert_eq!(audio_fpcr(active), active);
                        }
                    }
                }
            }
        }
        assert_eq!(audio_fpcr(1_u64 << 19) & (1_u64 << 19), 0);

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
            // SAFETY: User-space read of FPCR has no side effects.
            unsafe {
                core::arch::asm!("mrs {}, fpcr", out(reg) value, options(nomem, nostack, preserves_flags));
            }
            value
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut value = 0_u32;
            // SAFETY: `&raw mut value` points to an initialized local stack `u32`.
            unsafe {
                core::arch::asm!("stmxcsr [{}]", in(reg) &raw mut value, options(nostack, preserves_flags));
            }
            u64::from(value)
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn write_fp_control(value: u64) {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Restores valid FPCR test state.
            unsafe {
                core::arch::asm!("msr fpcr, {}", in(reg) value, options(nomem, nostack, preserves_flags));
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            let value = value as u32;
            // SAFETY: `&raw const value` points to an initialized local `u32` MXCSR word.
            unsafe {
                core::arch::asm!("ldmxcsr [{}]", in(reg) &raw const value, options(nostack, preserves_flags));
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
            let mut expected =
                (host & !((0b11_u64 << 22) | (1_u64 << 19) | 0x9f00)) | (1_u64 << 24);
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
    fn test_fp_guard_selects_rne_ftz_and_restores_on_drop_and_unwind() {
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
                {
                    let _guard = FtzDazGuard::new();
                    assert_eq!(read_fp_control(), expected_audio_fp_control(host));
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
                }
                assert_eq!(read_fp_control(), host);
            }
        }

        write_fp_control(safe_original);
        let host = read_fp_control();
        let unwound = std::panic::catch_unwind(|| {
            let _guard = FtzDazGuard::new();
            panic!("verify unwind restore");
        });
        assert!(unwound.is_err());
        assert_eq!(read_fp_control(), host);
    }
}
