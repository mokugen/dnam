//! Lock-free `256 B` (`64 x u32le`) `DNPR` seqlock live telemetry wire format
//! (`ref[container.proj.preamble-and-directory]`, `ref[arch.telemetry.seqlock-publication]`,
//! `ref[sdk.rust.zerocopy-wire-structs]`).

use core::f64::consts::TAU;
use core::mem::{align_of, offset_of, size_of};
use core::sync::atomic::{AtomicU32, Ordering, fence};
use std::sync::Mutex;

use dnam_vm::Tier;
use zerocopy::byteorder::little_endian::{F32, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Canonical magic word stored in word `15` (byte offset `60..64`) of [`DnprProgressWire`].
pub const PROGRESS_MAGIC_DNPR: u32 = u32::from_be_bytes(*b"DNPR");
/// Alias for [`PROGRESS_MAGIC_DNPR`].
pub const DNPR_MAGIC: u32 = PROGRESS_MAGIC_DNPR;
/// Number of `u32` little-endian words in a `DNPR` packet (`64`).
pub const PROGRESS_WIRE_WORDS: usize = 64;
/// Total `DNPR` packet byte size (`256 B`).
pub const PROGRESS_WIRE_BYTES: usize = PROGRESS_WIRE_WORDS * 4;
/// Word index of `magic` (`PROGRESS_MAGIC_DNPR`) inside the `64`-word `DNPR` packet.
pub const PROGRESS_WORD_MAGIC: usize = 15;

/// Coarse lifecycle state stored in byte 0 of `packed_state`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrainingStatus {
    /// Solver is idle / unstarted.
    #[default]
    Idle = 0,
    /// Solver is actively executing.
    Running = 1,
    /// Solver finished cleanly.
    Completed = 2,
    /// Solver halted due to an error.
    Failed = 3,
}

impl TrainingStatus {
    /// Decodes a status code byte safely.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Running,
            2 => Self::Completed,
            3 => Self::Failed,
            _ => Self::Idle,
        }
    }
}

/// Active macro training phase stored in byte 1 of `packed_state`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrainingPhase {
    /// Idle phase before any pass starts.
    #[default]
    Idle = 0,
    /// Capture acquisition phase.
    Capture = 1,
    /// Pass-1 CPU physics probe & pole initialization.
    Pass1Probe = 2,
    /// Pass-2 backbone adjoint trajectory optimization.
    Pass2Backbone = 3,
    /// Pass-3 nested VarPro readout solver.
    Pass3VarPro = 4,
    /// Pass-4 parametric CP1 & Wiener control extraction.
    Pass4Controls = 5,
    /// Pass-5 VM emission, LUFS calibration & certification.
    Pass5Audit = 6,
    /// Live null-test verification phase.
    LiveNullTest = 7,
    /// Training completed successfully.
    Completed = 8,
    /// Training failed.
    Failed = 9,
}

impl TrainingPhase {
    /// Decodes a phase code byte safely.
    #[inline]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Capture,
            2 => Self::Pass1Probe,
            3 => Self::Pass2Backbone,
            4 => Self::Pass3VarPro,
            5 => Self::Pass4Controls,
            6 => Self::Pass5Audit,
            7 => Self::LiveNullTest,
            8 => Self::Completed,
            9 => Self::Failed,
            _ => Self::Idle,
        }
    }
}

#[inline]
const fn tier_to_code(tier: Option<Tier>) -> u8 {
    match tier {
        None => 0,
        Some(Tier::Lite) => 1,
        Some(Tier::Full) => 2,
        Some(Tier::Pro) => 3,
        Some(Tier::Ultra) => 4,
    }
}

#[inline]
const fn code_to_tier(code: u8) -> Option<Tier> {
    match code {
        1 => Some(Tier::Lite),
        2 => Some(Tier::Full),
        3 => Some(Tier::Pro),
        4 => Some(Tier::Ultra),
        _ => None,
    }
}

/// Canonical `256 B` (`64 x u32le`) `DNPR` live training progress wire frame
/// (`ref[container.proj.preamble-and-directory]`, `ref[arch.telemetry.seqlock-publication]`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DnprProgressWire {
    /// Seqlock sequence counter (`odd` while writing, `even` when consistent).
    pub seq: U32,
    /// `status (u8) | (phase << 8) | (tier_code << 16) | (permille << 20)`.
    pub packed_state: U32,
    /// `step.min(0xFFFF) | (total_steps.min(0xFFFF) << 16)`.
    pub step_counters: U32,
    /// `stage_step.min(0xFFFF) | (stage_total_steps.min(0xFFFF) << 16)`.
    pub stage_counters: U32,
    /// Current best composite training loss.
    pub best_loss: F32,
    /// Current MR-IIR time-frequency loss.
    pub mr_iir: F32,
    /// `clamped_controls (u16) | (has_visual_frame << 16)`.
    pub clamped_and_vis: U32,
    /// Per-tier Bezout bESR metrics in dB (`[Lite, Full, Pro, Ultra]`; `NaN` until computed).
    pub tier_besr_db: [F32; 4],
    /// Per-tier 33-band LTAS max deviations in dB (`[Lite, Full, Pro, Ultra]`; `NaN` until computed).
    pub tier_ltas_db: [F32; 4],
    /// Canonical `PROGRESS_MAGIC_DNPR` (`0x444E_5052`) tag at word index `15` (byte `60..64`).
    pub magic: U32,
    /// Peak-transient 64-sample target waveform scope quantized as signed `i8`.
    pub target_wave_i8: [i8; 64],
    /// Peak-transient 64-sample model waveform scope quantized as signed `i8`.
    pub model_wave_i8: [i8; 64],
    /// 32-band `1/3`-octave spectral error quantized at `0.05 dB` (`Q20`) steps into `[-127, 127]`.
    pub ltas_error_q20_i8: [i8; 32],
    /// Up to 16 shared Kautz poles `[r_u8, f_u8]` across `0.5..=1.0` radius and log `20..20k Hz`.
    pub kautz_poles_q8: [[u8; 2]; 16],
}

const _: () = assert!(size_of::<DnprProgressWire>() == PROGRESS_WIRE_BYTES);
const _: () = assert!(align_of::<DnprProgressWire>() == 1);
const _: () = assert!(offset_of!(DnprProgressWire, seq) == 0);
const _: () = assert!(offset_of!(DnprProgressWire, packed_state) == 4);
const _: () = assert!(offset_of!(DnprProgressWire, step_counters) == 8);
const _: () = assert!(offset_of!(DnprProgressWire, stage_counters) == 12);
const _: () = assert!(offset_of!(DnprProgressWire, best_loss) == 16);
const _: () = assert!(offset_of!(DnprProgressWire, mr_iir) == 20);
const _: () = assert!(offset_of!(DnprProgressWire, clamped_and_vis) == 24);
const _: () = assert!(offset_of!(DnprProgressWire, tier_besr_db) == 28);
const _: () = assert!(offset_of!(DnprProgressWire, tier_ltas_db) == 44);
const _: () = assert!(offset_of!(DnprProgressWire, magic) == PROGRESS_WORD_MAGIC * 4);
const _: () = assert!(offset_of!(DnprProgressWire, target_wave_i8) == 64);
const _: () = assert!(offset_of!(DnprProgressWire, model_wave_i8) == 128);
const _: () = assert!(offset_of!(DnprProgressWire, ltas_error_q20_i8) == 192);
const _: () = assert!(offset_of!(DnprProgressWire, kautz_poles_q8) == 224);

impl DnprProgressWire {
    /// Returns a zeroed idle `DNPR` frame with valid `PROGRESS_MAGIC_DNPR` and `NaN` uncomputed metrics.
    #[must_use]
    pub fn idle_initialized() -> Self {
        Self {
            seq: U32::new(0),
            packed_state: U32::new(0),
            step_counters: U32::new(0),
            stage_counters: U32::new(0),
            best_loss: F32::new(f32::NAN),
            mr_iir: F32::new(f32::NAN),
            clamped_and_vis: U32::new(0),
            tier_besr_db: [F32::new(f32::NAN); 4],
            tier_ltas_db: [F32::new(f32::NAN); 4],
            magic: U32::new(PROGRESS_MAGIC_DNPR),
            target_wave_i8: [0_i8; 64],
            model_wave_i8: [0_i8; 64],
            ltas_error_q20_i8: [0_i8; 32],
            kautz_poles_q8: [[0_u8; 2]; 16],
        }
    }

    /// Decodes the lifecycle status from `packed_state`.
    #[inline]
    #[must_use]
    pub fn status(&self) -> TrainingStatus {
        TrainingStatus::from_code((self.packed_state.get() & 0xFF) as u8)
    }

    /// Decodes the active macro training phase from `packed_state`.
    #[inline]
    #[must_use]
    pub fn phase(&self) -> TrainingPhase {
        TrainingPhase::from_code(((self.packed_state.get() >> 8) & 0xFF) as u8)
    }

    /// Decodes the active tier (`None` until assigned).
    #[inline]
    #[must_use]
    pub fn active_tier(&self) -> Option<Tier> {
        code_to_tier(((self.packed_state.get() >> 16) & 0x0F) as u8)
    }

    /// Decodes normalized progress permille (`0..=1000`).
    #[inline]
    #[must_use]
    pub fn progress_permille(&self) -> u16 {
        (((self.packed_state.get() >> 20) & 0x0FFF) as u16).min(1_000)
    }

    /// Decodes `(step, total_steps)` from `step_counters`.
    #[inline]
    #[must_use]
    pub fn steps(&self) -> (u16, u16) {
        let w = self.step_counters.get();
        ((w & 0xFFFF) as u16, ((w >> 16) & 0xFFFF) as u16)
    }

    /// Decodes `(stage_step, stage_total_steps)` from `stage_counters`.
    #[inline]
    #[must_use]
    pub fn stage_steps(&self) -> (u16, u16) {
        let w = self.stage_counters.get();
        ((w & 0xFFFF) as u16, ((w >> 16) & 0xFFFF) as u16)
    }

    /// Decodes `(clamped_controls, has_visual_frame)` from `clamped_and_vis`.
    #[inline]
    #[must_use]
    pub fn visual_meta(&self) -> (u16, bool) {
        let w = self.clamped_and_vis.get();
        ((w & 0xFFFF) as u16, ((w >> 16) & 1) != 0)
    }
}

/// Snapshot metrics bundle passed to [`AtomicDnprProgress::publish_metrics`].
#[derive(Clone, Copy, Debug)]
pub struct ProgressMetrics {
    /// Coarse lifecycle status (`Idle`, `Running`, `Completed`, `Failed`).
    pub status: TrainingStatus,
    /// Active macro phase (`Pass1Probe`..`Completed`).
    pub phase: TrainingPhase,
    /// Active tier under evaluation (`None` before assignment).
    pub active_tier: Option<Tier>,
    /// Normalized overall progress in permille (`0..=1000`).
    pub progress_permille: u16,
    /// Overall completed pass/step counter.
    pub step: u32,
    /// Overall total step budget.
    pub total_steps: u32,
    /// Completed steps within the current stage.
    pub stage_step: u32,
    /// Total step budget for the current stage.
    pub stage_total_steps: u32,
    /// Current best composite scalar training loss.
    pub best_loss: f64,
    /// Current MR-IIR multi-resolution loss.
    pub mr_iir: f64,
    /// Count of clamped/locked physical controls reported by Phase-1 audit.
    pub clamped_controls: u16,
    /// Whether visual waveform/spectrum/pole telemetry scopes are populated.
    pub has_visual_frame: bool,
    /// Certified per-tier bESR in dB (`[Lite, Full, Pro, Ultra]`).
    pub tier_besr_db: [f32; 4],
    /// Certified per-tier max LTAS error in dB (`[Lite, Full, Pro, Ultra]`).
    pub tier_ltas_db: [f32; 4],
    /// Signed 64-sample peak-normalized target waveform scope (`[-127, 127]`).
    pub target_wave_i8: [i8; 64],
    /// Signed 64-sample peak-normalized model waveform scope (`[-127, 127]`).
    pub model_wave_i8: [i8; 64],
    /// Signed 32-band `1/3`-octave spectral error quantized in `0.05 dB` steps (`[-127, 127]`).
    pub ltas_error_q20_i8: [i8; 32],
    /// Up to 16 shared Kautz poles encoded as `[r_u8, f_u8]`.
    pub kautz_poles_q8: [[u8; 2]; 16],
}

impl Default for ProgressMetrics {
    fn default() -> Self {
        Self {
            status: TrainingStatus::Idle,
            phase: TrainingPhase::Idle,
            active_tier: None,
            progress_permille: 0,
            step: 0,
            total_steps: 0,
            stage_step: 0,
            stage_total_steps: 0,
            best_loss: f64::NAN,
            mr_iir: f64::NAN,
            clamped_controls: 0,
            has_visual_frame: false,
            tier_besr_db: [f32::NAN; 4],
            tier_ltas_db: [f32::NAN; 4],
            target_wave_i8: [0_i8; 64],
            model_wave_i8: [0_i8; 64],
            ltas_error_q20_i8: [0_i8; 32],
            kautz_poles_q8: [[0_u8; 2]; 16],
        }
    }
}

impl ProgressMetrics {
    /// Encodes metrics into a `DNPR` wire frame with zeroed `seq` (`AtomicDnprProgress` sets `seq`).
    #[must_use]
    pub fn to_wire(&self) -> DnprProgressWire {
        let perm = u32::from(self.progress_permille.min(1_000));
        let packed_state = u32::from(self.status as u8)
            | (u32::from(self.phase as u8) << 8)
            | (u32::from(tier_to_code(self.active_tier)) << 16)
            | (perm << 20);
        let step_counters = self.step.min(0xFFFF) | (self.total_steps.min(0xFFFF) << 16);
        let stage_counters =
            self.stage_step.min(0xFFFF) | (self.stage_total_steps.min(0xFFFF) << 16);
        let clamped_and_vis =
            u32::from(self.clamped_controls) | (u32::from(u8::from(self.has_visual_frame)) << 16);
        DnprProgressWire {
            seq: U32::new(0),
            packed_state: U32::new(packed_state),
            step_counters: U32::new(step_counters),
            stage_counters: U32::new(stage_counters),
            best_loss: F32::new(self.best_loss as f32),
            mr_iir: F32::new(self.mr_iir as f32),
            clamped_and_vis: U32::new(clamped_and_vis),
            tier_besr_db: self.tier_besr_db.map(F32::new),
            tier_ltas_db: self.tier_ltas_db.map(F32::new),
            magic: U32::new(PROGRESS_MAGIC_DNPR),
            target_wave_i8: self.target_wave_i8,
            model_wave_i8: self.model_wave_i8,
            ltas_error_q20_i8: self.ltas_error_q20_i8,
            kautz_poles_q8: self.kautz_poles_q8,
        }
    }
}

/// Safe lock-free `256 B` (`64 x AtomicU32`) seqlock container sharing live [`DnprProgressWire`]
/// across solver workers and UI observers with explicit ARM64 acquire/release memory barriers.
#[repr(align(64))]
#[derive(Debug)]
pub struct AtomicDnprProgress {
    words: [AtomicU32; PROGRESS_WIRE_WORDS],
    write_guard: Mutex<()>,
}

impl Default for AtomicDnprProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicDnprProgress {
    /// Allocates an idle, valid `DNPR` seqlock slot (`seq = 0`, `magic = PROGRESS_MAGIC_DNPR`).
    #[must_use]
    pub fn new() -> Self {
        let init = DnprProgressWire::idle_initialized();
        let word_slice = <[U32; PROGRESS_WIRE_WORDS]>::read_from_bytes(init.as_bytes())
            .expect("256 B wire must reinterpret as 64 U32 words");
        Self {
            words: core::array::from_fn(|i| AtomicU32::new(word_slice[i].get())),
            write_guard: Mutex::new(()),
        }
    }

    /// Publishes `wire` atomically into the seqlock buffer using odd/even sequence fencing.
    pub fn publish(&self, wire: &DnprProgressWire) {
        let Ok(_guard) = self.write_guard.lock() else {
            return;
        };
        let mut patched = *wire;
        patched.magic = U32::new(PROGRESS_MAGIC_DNPR);
        let word_slice = <[U32; PROGRESS_WIRE_WORDS]>::read_from_bytes(patched.as_bytes())
            .expect("256 B wire must reinterpret as 64 U32 words");

        let cur = self.words[0].load(Ordering::Relaxed);
        let base_even = cur & !1;
        let odd_seq = base_even.wrapping_add(1);
        let even_seq = base_even.wrapping_add(2);

        self.words[0].store(odd_seq, Ordering::Release);
        fence(Ordering::SeqCst);
        for (idx, word) in word_slice.iter().enumerate().skip(1) {
            self.words[idx].store(word.get(), Ordering::Relaxed);
        }
        fence(Ordering::Release);
        self.words[0].store(even_seq, Ordering::Release);
    }

    /// Encodes and publishes [`ProgressMetrics`] into the seqlock buffer.
    pub fn publish_metrics(&self, metrics: &ProgressMetrics) {
        self.publish(&metrics.to_wire());
    }

    /// Reads a consistent tear-free [`DnprProgressWire`] snapshot using ARM64 acquire barriers.
    #[must_use]
    pub fn snapshot(&self) -> DnprProgressWire {
        let mut raw_words = [U32::new(0); PROGRESS_WIRE_WORDS];
        for _ in 0..64 {
            let s1 = self.words[0].load(Ordering::Acquire);
            if (s1 & 1) != 0 {
                core::hint::spin_loop();
                continue;
            }
            raw_words[0] = U32::new(s1);
            for (idx, slot) in self.words.iter().enumerate().skip(1) {
                raw_words[idx] = U32::new(slot.load(Ordering::Relaxed));
            }
            fence(Ordering::Acquire);
            let s2 = self.words[0].load(Ordering::Acquire);
            if s1 == s2
                && raw_words[PROGRESS_WORD_MAGIC].get() == PROGRESS_MAGIC_DNPR
                && let Ok(parsed) = DnprProgressWire::read_from_bytes(raw_words.as_bytes())
            {
                return parsed;
            }
        }
        DnprProgressWire::idle_initialized()
    }
}

/// Quantizes up to `32` `1/3`-octave spectral error bands (`Q20`: `0.05 dB` steps into `[-127, 127]`)
/// per `ref[container.proj.preamble-and-directory]`.
#[must_use]
pub fn quantize_ltas_error_q20_i8(pred_pow: &[f64], tgt_pow: &[f64], floor_pow: f64) -> [i8; 32] {
    let mut out = [0_i8; 32];
    let fl = floor_pow.max(1.0e-24);
    for (idx, slot) in out.iter_mut().enumerate() {
        let pm = pred_pow.get(idx).copied().unwrap_or(0.0);
        let pt = tgt_pow.get(idx).copied().unwrap_or(0.0);
        if pm.is_finite() && pt.is_finite() {
            let db = 10.0 * ((pm.max(0.0) + fl) / (pt.max(0.0) + fl)).log10();
            *slot = (db * 20.0).round().clamp(-127.0, 127.0) as i8;
        }
    }
    out
}

/// Quantizes a Kautz complex pole `(radius, theta_rad)` into `[r_u8, f_u8]` per
/// `ref[container.proj.preamble-and-directory]`.
#[must_use]
pub fn quantize_kautz_pole_q8(r: f64, theta_rad: f64, sample_rate_hz: f64) -> [u8; 2] {
    if !r.is_finite() || !theta_rad.is_finite() || !sample_rate_hz.is_finite() {
        return [0, 0];
    }
    let r_u8 = (((r - 0.5) / 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
    let sr = sample_rate_hz.max(8_000.0);
    let f_hz = (theta_rad.abs() * sr / TAU).clamp(20.0, 20_000.0);
    let f_u8 = (((f_hz / 20.0).ln() / 1000.0_f64.ln()).clamp(0.0, 1.0) * 255.0).round() as u8;
    [r_u8, f_u8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dnpr_seqlock_canonical_layout_and_quantizers() {
        let prog = AtomicDnprProgress::new();
        let init = prog.snapshot();
        assert_eq!(init.magic.get(), PROGRESS_MAGIC_DNPR);
        assert_eq!(init.seq.get() % 2, 0);
        assert!(init.best_loss.get().is_nan());
        assert_eq!(init.status(), TrainingStatus::Idle);

        let pred = [2.0_f64; 32];
        let tgt = [1.0_f64; 32];
        let q20 = quantize_ltas_error_q20_i8(&pred, &tgt, 1.0e-12);
        // 10*log10(2) ~= 3.0103 dB -> *20 ~= 60 Q20 steps.
        assert_eq!(q20[0], 60);

        let pole = quantize_kautz_pole_q8(0.75, TAU * 200.0 / 48_000.0, 48_000.0);
        // r=0.75 -> 0.5 -> 128; f=200 Hz -> ln(10)/ln(1000) = 1/3 -> 85.
        assert!((i16::from(pole[0]) - 128).abs() <= 1);
        assert!((i16::from(pole[1]) - 85).abs() <= 1);

        let mut poles = [[0_u8; 2]; 16];
        poles[0] = pole;
        prog.publish_metrics(&ProgressMetrics {
            status: TrainingStatus::Running,
            phase: TrainingPhase::Pass2Backbone,
            active_tier: Some(Tier::Ultra),
            progress_permille: 750,
            step: 3,
            total_steps: 4,
            stage_step: 2,
            stage_total_steps: 4,
            best_loss: 0.015,
            mr_iir: 0.012,
            clamped_controls: 1,
            has_visual_frame: true,
            tier_besr_db: [-21.0, -25.0, -29.0, -34.0],
            tier_ltas_db: [0.6, 0.3, 0.14, 0.08],
            target_wave_i8: [42_i8; 64],
            model_wave_i8: [40_i8; 64],
            ltas_error_q20_i8: q20,
            kautz_poles_q8: poles,
        });
        let snap = prog.snapshot();
        assert_eq!(snap.magic.get(), PROGRESS_MAGIC_DNPR);
        assert_eq!(snap.seq.get(), 2);
        assert_eq!(snap.status(), TrainingStatus::Running);
        assert_eq!(snap.phase(), TrainingPhase::Pass2Backbone);
        assert_eq!(snap.active_tier(), Some(Tier::Ultra));
        assert_eq!(snap.progress_permille(), 750);
        assert_eq!(snap.steps(), (3, 4));
        assert_eq!(snap.stage_steps(), (2, 4));
        assert_eq!(snap.visual_meta(), (1, true));
        assert_eq!(snap.target_wave_i8[0], 42);
        assert_eq!(snap.model_wave_i8[63], 40);
        assert_eq!(snap.ltas_error_q20_i8[0], 60);
        assert_eq!(snap.kautz_poles_q8[0], pole);
        // Verify raw wire offset 60..64 equals LE bytes of PROGRESS_MAGIC_DNPR (0x52, 0x50, 0x4E, 0x44).
        let raw = snap.as_bytes();
        assert_eq!(&raw[60..64], &PROGRESS_MAGIC_DNPR.to_le_bytes());
    }

    #[test]
    fn test_progress_code_roundtrips_and_default() {
        for code in 0..=4_u8 {
            let _ = TrainingStatus::from_code(code);
        }
        for code in 0..=10_u8 {
            let _ = TrainingPhase::from_code(code);
        }
        for tier in [
            None,
            Some(Tier::Lite),
            Some(Tier::Full),
            Some(Tier::Pro),
            Some(Tier::Ultra),
        ] {
            let c = tier_to_code(tier);
            assert_eq!(code_to_tier(c), tier);
        }
        assert_eq!(code_to_tier(99), None);
        let def = AtomicDnprProgress::default();
        assert_eq!(def.snapshot().magic.get(), PROGRESS_MAGIC_DNPR);
        assert_eq!(quantize_kautz_pole_q8(f64::NAN, 0.1, 48_000.0), [0, 0]);
    }
}
