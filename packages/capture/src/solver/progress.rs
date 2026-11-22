#![forbid(unsafe_code)]
//! Hybrid Lock-Free `256 B` Seqlock Polling & Coarse-Grained Push Telemetry (`solver::progress`).
//!
//! Emits trainer milestones and compact realtime fitting snapshots (`64`-sample peak-transient
//! waveform scope, `32`-band `1/3`-octave spectral error curve, `16` shared Kautz poles, and
//! 4-tier certification metrics) at coarse solver boundaries (`0` inner-loop overhead).

pub mod telemetry;

use core::sync::atomic::{AtomicU32, Ordering, fence};
use std::fmt;
use std::sync::{Arc, Mutex};

use dnam_train::DnprProgressWire;
use dnam_vm::Tier;
use zerocopy::byteorder::little_endian::{F32, U32};
use zerocopy::{FromBytes, IntoBytes};

/// Magic word stored in `word[WORD_MAGIC]` (`15`) of every `256 B` (`DNPR`) progress packet.
pub const PROGRESS_MAGIC_DNPR: u32 = dnam_train::PROGRESS_MAGIC_DNPR;

/// Seqlock sequence counter word index (`0`) inside [`DnprProgressWire`].
const WORD_SEQ: usize = 0;
/// Magic word index (`15`) inside [`DnprProgressWire`].
const WORD_MAGIC: usize = 15;

/// Number of little-endian `u32` words in the `DNPR` wire frame (`64 x 4 B = 256 B`).
pub const PROGRESS_WIRE_WORDS: usize = 64;

/// Total byte size of a serialized `DNPR` progress frame (`256 B`).
pub const PROGRESS_WIRE_BYTES: usize = PROGRESS_WIRE_WORDS * 4;

const _: () = assert!(core::mem::size_of::<DnprProgressWire>() == PROGRESS_WIRE_BYTES);
const _: () = assert!(PROGRESS_WIRE_WORDS == 64);
const _: () = assert!(PROGRESS_WIRE_BYTES == 256);

/// Coarse lifecycle state reported by [`TrainerProgressSnapshot`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrainingStatus {
    #[default]
    Idle = 0,
    Running = 1,
    Completed = 2,
    Failed = 3,
}

impl TrainingStatus {
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

/// Active macro stage reported in [`TrainerProgressSnapshot`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrainingPhase {
    #[default]
    Idle = 0,
    Capture = 1,
    StageAPhysicsProbe = 2,
    StageBBackbone = 3,
    Stage4SpatialCascade = 4,
    StageCKnobs = 5,
    StageDAudit = 6,
    LiveNullTest = 7,
    Completed = 8,
    Failed = 9,
}

impl TrainingPhase {
    #[inline]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Capture,
            2 => Self::StageAPhysicsProbe,
            3 => Self::StageBBackbone,
            4 => Self::Stage4SpatialCascade,
            5 => Self::StageCKnobs,
            6 => Self::StageDAudit,
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

/// Complete immutable trainer progress & live fitting visualization snapshot (`256 B` wire compatible).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainerProgressSnapshot {
    pub status: TrainingStatus,
    pub phase: TrainingPhase,
    pub active_tier: Option<Tier>,
    pub clamped_controls: u16,
    pub has_visual_frame: bool,
    /// Monotonic normalized progress in `[0..=1_000]` (`0.1%` resolution).
    pub progress_permille: u16,
    pub step: u32,
    pub total_steps: u32,
    pub stage_step: u32,
    pub stage_total_steps: u32,
    pub best_loss: f32,
    pub mr_iir_loss: f32,
    /// Per-tier `[Lite, Full, Pro, Ultra]` `bESR` in dB (`NaN` until measured).
    pub tier_besr_db: [f32; 4],
    /// Per-tier `[Lite, Full, Pro, Ultra]` `LTAS` max deviation in dB (`NaN` until measured).
    pub tier_ltas_db: [f32; 4],
    /// Normalized `2x`-decimated target transient waveform over peak `128`-sample window (`[-127, 127]`).
    pub target_wave_i8: [i8; 64],
    /// Normalized `2x`-decimated model output waveform over the exact same window & peak scale.
    pub model_wave_i8: [i8; 64],
    /// First `32` base-10 `1/3`-octave `LTAS` spectral error bands in `0.05 dB` units (`[-127, 127]` = `±6.35 dB`).
    pub ltas_error_q20_i8: [i8; 32],
    /// Normalized `(radius_u8, log_freq_u8)` pairs for the `16` shared Kautz poles.
    pub kautz_poles_q8: [(u8, u8); 16],
}

impl Default for TrainerProgressSnapshot {
    fn default() -> Self {
        Self {
            status: TrainingStatus::Idle,
            phase: TrainingPhase::Idle,
            active_tier: None,
            clamped_controls: 0,
            has_visual_frame: false,
            progress_permille: 0,
            step: 0,
            total_steps: 0,
            stage_step: 0,
            stage_total_steps: 0,
            best_loss: f32::NAN,
            mr_iir_loss: f32::NAN,
            tier_besr_db: [f32::NAN; 4],
            tier_ltas_db: [f32::NAN; 4],
            target_wave_i8: [0; 64],
            model_wave_i8: [0; 64],
            ltas_error_q20_i8: [0; 32],
            kautz_poles_q8: [(0, 0); 16],
        }
    }
}

impl TrainerProgressSnapshot {
    /// Returns normalized progress in `[0.0, 1.0]`.
    #[inline]
    #[must_use]
    pub fn progress_fraction(&self) -> f32 {
        f32::from(self.progress_permille.min(1_000)) * 0.001
    }

    /// Returns human-readable amplitude match percentage in `[0.0, 100.0]` derived from
    /// active/best tier `bESR` dB or `best_loss` (`None` before any measurement exists).
    #[must_use]
    pub fn tone_match_percent(&self) -> Option<f32> {
        let active_esr = self
            .active_tier
            .and_then(|t| {
                let db = self.tier_besr_db[t as usize];
                db.is_finite().then_some(db)
            })
            .or_else(|| {
                self.tier_besr_db
                    .iter()
                    .rev()
                    .copied()
                    .find(|db| db.is_finite())
            })
            .or_else(|| {
                (self.best_loss.is_finite() && self.best_loss >= 0.0)
                    .then(|| 10.0 * self.best_loss.max(1.0e-12).log10())
            })?;
        Some((1.0 - 10.0_f32.powf(active_esr / 20.0)).clamp(0.0, 1.0) * 100.0)
    }

    /// Encodes this snapshot into a canonical `256 B` [`DnprProgressWire`] packet with `seq`.
    #[must_use]
    pub fn to_wire(&self, seq: u32) -> DnprProgressWire {
        let tier_code = u32::from(tier_to_code(self.active_tier));
        let perm = u32::from(self.progress_permille.min(1_000));
        let packed_state = u32::from(self.status as u8)
            | (u32::from(self.phase as u8) << 8)
            | (tier_code << 16)
            | (perm << 20);
        let mut kautz_poles_q8 = [[0_u8; 2]; 16];
        for (dst, &(r_q, f_q)) in kautz_poles_q8.iter_mut().zip(self.kautz_poles_q8.iter()) {
            *dst = [r_q, f_q];
        }
        DnprProgressWire {
            seq: U32::new(seq),
            packed_state: U32::new(packed_state),
            step_counters: U32::new(self.step.min(0xFFFF) | (self.total_steps.min(0xFFFF) << 16)),
            stage_counters: U32::new(
                self.stage_step.min(0xFFFF) | (self.stage_total_steps.min(0xFFFF) << 16),
            ),
            best_loss: F32::new(self.best_loss),
            mr_iir: F32::new(self.mr_iir_loss),
            clamped_and_vis: U32::new(
                u32::from(self.clamped_controls)
                    | (u32::from(u8::from(self.has_visual_frame)) << 16),
            ),
            tier_besr_db: self.tier_besr_db.map(F32::new),
            tier_ltas_db: self.tier_ltas_db.map(F32::new),
            magic: U32::new(PROGRESS_MAGIC_DNPR),
            target_wave_i8: self.target_wave_i8,
            model_wave_i8: self.model_wave_i8,
            ltas_error_q20_i8: self.ltas_error_q20_i8,
            kautz_poles_q8,
        }
    }

    /// Decodes a validated (`seq & 1 == 0`) `256 B` [`DnprProgressWire`] packet.
    #[must_use]
    pub fn from_wire(wire: &DnprProgressWire) -> Option<(u32, Self)> {
        let seq = wire.seq.get();
        if (seq & 1) != 0 || wire.magic.get() != PROGRESS_MAGIC_DNPR {
            return None;
        }
        let w1 = wire.packed_state.get();
        let w2 = wire.step_counters.get();
        let w3 = wire.stage_counters.get();
        let w6 = wire.clamped_and_vis.get();
        let mut kautz_poles_q8 = [(0_u8, 0_u8); 16];
        for (dst, &[r_q, f_q]) in kautz_poles_q8.iter_mut().zip(wire.kautz_poles_q8.iter()) {
            *dst = (r_q, f_q);
        }
        Some((
            seq,
            Self {
                status: TrainingStatus::from_code((w1 & 0xFF) as u8),
                phase: TrainingPhase::from_code(((w1 >> 8) & 0xFF) as u8),
                active_tier: code_to_tier(((w1 >> 16) & 0x0F) as u8),
                clamped_controls: (w6 & 0xFFFF) as u16,
                has_visual_frame: ((w6 >> 16) & 1) != 0,
                progress_permille: (((w1 >> 20) & 0x0FFF) as u16).min(1_000),
                step: w2 & 0xFFFF,
                total_steps: (w2 >> 16) & 0xFFFF,
                stage_step: w3 & 0xFFFF,
                stage_total_steps: (w3 >> 16) & 0xFFFF,
                best_loss: wire.best_loss.get(),
                mr_iir_loss: wire.mr_iir.get(),
                tier_besr_db: wire.tier_besr_db.map(F32::get),
                tier_ltas_db: wire.tier_ltas_db.map(F32::get),
                target_wave_i8: wire.target_wave_i8,
                model_wave_i8: wire.model_wave_i8,
                ltas_error_q20_i8: wire.ltas_error_q20_i8,
                kautz_poles_q8,
            },
        ))
    }

    /// Serializes this snapshot into the canonical `64`-word (`256 B`) `DNPR` layout with sequence `seq`.
    #[must_use]
    pub fn to_words(&self, seq: u32) -> [u32; PROGRESS_WIRE_WORDS] {
        let wire = self.to_wire(seq);
        let words_le = <[U32; PROGRESS_WIRE_WORDS]>::ref_from_bytes(wire.as_bytes())
            .expect("256B DnprProgressWire converts cleanly to 64 u32le words");
        words_le.map(U32::get)
    }

    /// Parses a consistent (`seq & 1 == 0`) `64`-word `DNPR` frame, returning `(seq, snapshot)`.
    #[must_use]
    pub fn from_words(words: &[u32; PROGRESS_WIRE_WORDS]) -> Option<(u32, Self)> {
        let words_le = words.map(U32::new);
        let wire = DnprProgressWire::ref_from_bytes(words_le.as_bytes()).ok()?;
        Self::from_wire(wire)
    }

    /// Serializes this snapshot into a little-endian `256 B` byte packet.
    #[must_use]
    pub fn to_bytes_256b(&self, seq: u32) -> [u8; PROGRESS_WIRE_BYTES] {
        let wire = self.to_wire(seq);
        let mut out = [0_u8; PROGRESS_WIRE_BYTES];
        out.copy_from_slice(wire.as_bytes());
        out
    }

    /// Parses a little-endian `256 B` `DNPR` packet if `seq` is even and magic matches.
    #[must_use]
    pub fn from_bytes_256b(bytes: &[u8; PROGRESS_WIRE_BYTES]) -> Option<(u32, Self)> {
        let wire = DnprProgressWire::ref_from_bytes(bytes).ok()?;
        Self::from_wire(wire)
    }
}

/// Cache-line aligned lock-free `256 B` seqlock slot for zero-overhead UI polling (`Atomics.load` / `poll()`).
#[repr(align(64))]
#[derive(Debug)]
pub struct AtomicTrainerProgress {
    words: [AtomicU32; PROGRESS_WIRE_WORDS],
    max_permille: AtomicU32,
    write_guard: Mutex<()>,
}

impl Default for AtomicTrainerProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicTrainerProgress {
    /// Allocates an idle, valid `DNPR` seqlock progress slot (`seq = 0`).
    #[must_use]
    pub fn new() -> Self {
        let init = TrainerProgressSnapshot::default().to_words(0);
        Self {
            words: init.map(AtomicU32::new),
            max_permille: AtomicU32::new(0),
            write_guard: Mutex::new(()),
        }
    }

    fn write_words_seqlock(&self, words: &[u32; PROGRESS_WIRE_WORDS]) {
        let cur = self.words[0].load(Ordering::Relaxed);
        let base_even = cur & !1;
        let odd_seq = base_even.wrapping_add(1);
        let even_seq = base_even.wrapping_add(2);
        self.words[0].store(odd_seq, Ordering::Release);
        fence(Ordering::SeqCst);
        for (idx, &val) in words.iter().enumerate().skip(1) {
            self.words[idx].store(val, Ordering::Relaxed);
        }
        fence(Ordering::Release);
        self.words[0].store(even_seq, Ordering::Release);
    }

    /// Resets the slot to `Idle` (`0‰`).
    pub fn reset(&self) {
        let _guard = self.write_guard.lock();
        self.max_permille.store(0, Ordering::Release);
        let w = TrainerProgressSnapshot::default().to_words(0);
        self.write_words_seqlock(&w);
    }

    /// Publishes `snap` into the seqlock, enforcing monotonic `progress_permille` while `Running`.
    pub fn publish(&self, snap: &TrainerProgressSnapshot) {
        let _guard = self.write_guard.lock();
        let raw_perm = u32::from(snap.progress_permille.min(1_000));
        let eff_perm = match snap.status {
            TrainingStatus::Running => {
                let prev = self.max_permille.load(Ordering::Relaxed);
                let next = prev.max(raw_perm);
                self.max_permille.store(next, Ordering::Release);
                next as u16
            }
            TrainingStatus::Completed => {
                self.max_permille.store(1_000, Ordering::Release);
                1_000
            }
            TrainingStatus::Idle | TrainingStatus::Failed => raw_perm as u16,
        };
        let mut norm = snap.clone();
        norm.progress_permille = eff_perm;
        let words = norm.to_words(0);
        self.write_words_seqlock(&words);
    }

    /// Wait-free consistent seqlock read returning `(even_seq, raw_words)`.
    fn read_consistent_words(&self) -> (u32, [u32; PROGRESS_WIRE_WORDS]) {
        let mut buf = [0_u32; PROGRESS_WIRE_WORDS];
        for _ in 0..64 {
            let s1 = self.words[WORD_SEQ].load(Ordering::Acquire);
            if (s1 & 1) != 0 {
                core::hint::spin_loop();
                continue;
            }
            buf[WORD_SEQ] = s1;
            for (idx, slot) in self.words.iter().enumerate().skip(WORD_SEQ + 1) {
                buf[idx] = slot.load(Ordering::Relaxed);
            }
            fence(Ordering::Acquire);
            let s2 = self.words[WORD_SEQ].load(Ordering::Acquire);
            if s1 == s2 && buf[WORD_MAGIC] == PROGRESS_MAGIC_DNPR {
                return (s1, buf);
            }
        }
        (0, TrainerProgressSnapshot::default().to_words(0))
    }

    /// Lock-free wait-free poll returning a consistent [`TrainerProgressSnapshot`].
    #[must_use]
    pub fn poll(&self) -> TrainerProgressSnapshot {
        let (_, buf) = self.read_consistent_words();
        TrainerProgressSnapshot::from_words(&buf)
            .map(|(_, s)| s)
            .unwrap_or_default()
    }

    /// Lock-free consistent export of the current `256 B` little-endian `DNPR` packet.
    #[must_use]
    pub fn export_bytes_256b(&self) -> [u8; PROGRESS_WIRE_BYTES] {
        let (seq, buf) = self.read_consistent_words();
        let snap = TrainerProgressSnapshot::from_words(&buf)
            .map(|(_, s)| s)
            .unwrap_or_default();
        snap.to_bytes_256b(seq)
    }
}

#[derive(Clone, Debug)]
struct CachedProgressState {
    max_permille: u16,
    has_visual_frame: bool,
    best_loss: f32,
    mr_iir_loss: f32,
    tier_besr_db: [f32; 4],
    tier_ltas_db: [f32; 4],
    target_wave_i8: [i8; 64],
    model_wave_i8: [i8; 64],
    ltas_error_q20_i8: [i8; 32],
    kautz_poles_q8: [(u8, u8); 16],
}

impl Default for CachedProgressState {
    fn default() -> Self {
        Self {
            max_permille: 0,
            has_visual_frame: false,
            best_loss: f32::NAN,
            mr_iir_loss: f32::NAN,
            tier_besr_db: [f32::NAN; 4],
            tier_ltas_db: [f32::NAN; 4],
            target_wave_i8: [0; 64],
            model_wave_i8: [0; 64],
            ltas_error_q20_i8: [0; 32],
            kautz_poles_q8: [(0, 0); 16],
        }
    }
}

pub type ProgressCallback = Arc<dyn Fn(&TrainerProgressSnapshot) + Send + Sync>;

/// Optional hybrid progress sink (`AtomicTrainerProgress` seqlock + coarse-grained callback).
///
/// When created with `TrainerProgressHandle::default()`, [`TrainerProgressHandle::is_active`]
/// returns `false` with zero heap allocations and zero solver overhead.
#[derive(Clone)]
pub struct TrainerProgressHandle {
    slot: Option<Arc<AtomicTrainerProgress>>,
    callback: Option<ProgressCallback>,
    subrange_permille: (u16, u16),
    cache: Option<Arc<Mutex<CachedProgressState>>>,
}

impl Default for TrainerProgressHandle {
    fn default() -> Self {
        Self {
            slot: None,
            callback: None,
            subrange_permille: (0, 1_000),
            cache: None,
        }
    }
}

impl fmt::Debug for TrainerProgressHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainerProgressHandle")
            .field("has_slot", &self.slot.is_some())
            .field("has_callback", &self.callback.is_some())
            .field("subrange_permille", &self.subrange_permille)
            .finish()
    }
}

impl TrainerProgressHandle {
    /// Creates a handle writing exclusively to a lock-free `256 B` [`AtomicTrainerProgress`] slot.
    #[must_use]
    pub fn from_slot(slot: Arc<AtomicTrainerProgress>) -> Self {
        Self {
            slot: Some(slot),
            callback: None,
            subrange_permille: (0, 1_000),
            cache: Some(Arc::new(Mutex::new(CachedProgressState::default()))),
        }
    }

    /// Creates a handle pushing coarse snapshots exclusively to `callback`.
    #[must_use]
    pub fn from_callback<F>(callback: F) -> Self
    where
        F: Fn(&TrainerProgressSnapshot) + Send + Sync + 'static,
    {
        Self {
            slot: None,
            callback: Some(Arc::new(callback)),
            subrange_permille: (0, 1_000),
            cache: Some(Arc::new(Mutex::new(CachedProgressState::default()))),
        }
    }

    /// Creates a hybrid handle driving both a lock-free seqlock slot and an event callback.
    #[must_use]
    pub fn with_slot_and_callback<F>(slot: Arc<AtomicTrainerProgress>, callback: F) -> Self
    where
        F: Fn(&TrainerProgressSnapshot) + Send + Sync + 'static,
    {
        Self {
            slot: Some(slot),
            callback: Some(Arc::new(callback)),
            subrange_permille: (0, 1_000),
            cache: Some(Arc::new(Mutex::new(CachedProgressState::default()))),
        }
    }

    /// Returns `true` when at least one progress sink (slot or callback) is attached.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.slot.is_some() || self.callback.is_some()
    }

    /// Emits a lightweight lifecycle/phase milestone without replacing any cached visual frame.
    pub fn emit_phase(&self, status: TrainingStatus, phase: TrainingPhase, progress_permille: u16) {
        if !self.is_active() {
            return;
        }
        self.emit(&TrainerProgressSnapshot {
            status,
            phase,
            progress_permille,
            ..Default::default()
        });
    }

    /// Returns a child handle mapping `[0..=1_000]` permille cleanly into `[lo_permille..=hi_permille]`.
    #[must_use]
    pub fn subrange(&self, lo_permille: u16, hi_permille: u16) -> Self {
        if !self.is_active() {
            return Self::default();
        }
        let (p_lo, p_hi) = self.subrange_permille;
        let span = u32::from(p_hi.saturating_sub(p_lo));
        let map =
            |x: u16| -> u16 { p_lo + ((u32::from(x.min(1_000)) * span + 500) / 1_000) as u16 };
        let lo = map(lo_permille.min(hi_permille));
        let hi = map(hi_permille.max(lo_permille));
        Self {
            slot: self.slot.clone(),
            callback: self.callback.clone(),
            subrange_permille: (lo, hi),
            cache: self.cache.clone(),
        }
    }

    /// Normalizes subrange permille, carries forward prior visual arrays/metrics if omitted,
    /// publishes to any attached seqlock, and invokes any attached callback.
    pub fn emit(&self, snap: &TrainerProgressSnapshot) {
        if !self.is_active() {
            return;
        }
        let mut out = snap.clone();
        let (lo, hi) = self.subrange_permille;
        let is_root = lo == 0 && hi == 1_000;
        let mapped = lo
            + ((u32::from(out.progress_permille.min(1_000)) * u32::from(hi.saturating_sub(lo))
                + 500)
                / 1_000) as u16;
        out.progress_permille = mapped.min(1_000);
        if !is_root && out.status == TrainingStatus::Completed {
            out.status = TrainingStatus::Running;
            if out.phase == TrainingPhase::Completed {
                out.phase = TrainingPhase::StageBBackbone;
            }
        }
        if let Some(ref cache_mtx) = self.cache
            && let Ok(mut c) = cache_mtx.lock()
        {
            if out.status == TrainingStatus::Running {
                c.max_permille = c.max_permille.max(out.progress_permille);
                out.progress_permille = c.max_permille;
            } else if out.status == TrainingStatus::Completed {
                out.progress_permille = 1_000;
                c.max_permille = 1_000;
            } else if out.status == TrainingStatus::Idle {
                c.max_permille = 0;
            }
            if out.has_visual_frame {
                c.has_visual_frame = true;
                c.target_wave_i8 = out.target_wave_i8;
                c.model_wave_i8 = out.model_wave_i8;
                c.ltas_error_q20_i8 = out.ltas_error_q20_i8;
                c.kautz_poles_q8 = out.kautz_poles_q8;
            } else if c.has_visual_frame {
                out.has_visual_frame = true;
                out.target_wave_i8 = c.target_wave_i8;
                out.model_wave_i8 = c.model_wave_i8;
                out.ltas_error_q20_i8 = c.ltas_error_q20_i8;
                out.kautz_poles_q8 = c.kautz_poles_q8;
            }
            if out.best_loss.is_finite() {
                c.best_loss = out.best_loss;
            } else if c.best_loss.is_finite() {
                out.best_loss = c.best_loss;
            }
            if out.mr_iir_loss.is_finite() {
                c.mr_iir_loss = out.mr_iir_loss;
            } else if c.mr_iir_loss.is_finite() {
                out.mr_iir_loss = c.mr_iir_loss;
            }
            for t in 0..4 {
                if out.tier_besr_db[t].is_finite() {
                    c.tier_besr_db[t] = out.tier_besr_db[t];
                } else if c.tier_besr_db[t].is_finite() {
                    out.tier_besr_db[t] = c.tier_besr_db[t];
                }
                if out.tier_ltas_db[t].is_finite() {
                    c.tier_ltas_db[t] = out.tier_ltas_db[t];
                } else if c.tier_ltas_db[t].is_finite() {
                    out.tier_ltas_db[t] = c.tier_ltas_db[t];
                }
            }
        }
        if let Some(ref slot) = self.slot {
            slot.publish(&out);
        }
        if let Some(ref cb) = self.callback {
            cb(&out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn test_progress_256b_wire_codec_seqlock_and_subrange_monotonicity() {
        let mut snap = TrainerProgressSnapshot {
            status: TrainingStatus::Running,
            phase: TrainingPhase::Stage4SpatialCascade,
            active_tier: Some(Tier::Pro),
            clamped_controls: 5,
            has_visual_frame: true,
            progress_permille: 642,
            step: 14,
            total_steps: 20,
            stage_step: 3,
            stage_total_steps: 6,
            best_loss: 0.0125,
            mr_iir_loss: 0.285,
            tier_besr_db: [-21.5, -26.0, -32.25, -36.5],
            tier_ltas_db: [0.42, 0.21, 0.11, 0.06],
            target_wave_i8: [0; 64],
            model_wave_i8: [0; 64],
            ltas_error_q20_i8: [0; 32],
            kautz_poles_q8: [(0, 0); 16],
        };
        for i in 0..64 {
            snap.target_wave_i8[i] = ((i as i32) * 2 - 64) as i8;
            snap.model_wave_i8[i] = ((i as i32) * 2 - 62) as i8;
        }
        for b in 0..32 {
            snap.ltas_error_q20_i8[b] = ((b as i32) - 16) as i8;
        }
        for p in 0..16 {
            snap.kautz_poles_q8[p] = ((120 + p * 8) as u8, (15 + p * 14) as u8);
        }

        let mut odd_words = snap.to_words(7);
        assert!(TrainerProgressSnapshot::from_words(&odd_words).is_none());
        odd_words[WORD_SEQ] = 8;
        odd_words[WORD_MAGIC] ^= 1;
        assert!(TrainerProgressSnapshot::from_words(&odd_words).is_none());

        let bytes = snap.to_bytes_256b(12);
        let (seq, decoded) =
            TrainerProgressSnapshot::from_bytes_256b(&bytes).expect("valid 256B DNPR");
        assert_eq!(seq, 12);
        assert_eq!(decoded, snap);
        assert!((decoded.progress_fraction() - 0.642).abs() < 1.0e-5);
        assert!(
            TrainerProgressSnapshot::default()
                .tone_match_percent()
                .is_none()
        );
        let expected_pro_match = (1.0 - 10.0_f32.powf(-32.25 / 20.0)) * 100.0;
        assert!((decoded.tone_match_percent().unwrap() - expected_pro_match).abs() < 1.0e-3);
        let loss_only = TrainerProgressSnapshot {
            best_loss: 0.01,
            ..Default::default()
        };
        assert!((loss_only.tone_match_percent().unwrap() - 90.0).abs() < 1.0e-3);

        let slot = Arc::new(AtomicTrainerProgress::new());
        let events: Arc<StdMutex<Vec<TrainerProgressSnapshot>>> =
            Arc::new(StdMutex::new(Vec::new()));
        let ev_clone = Arc::clone(&events);
        let root = TrainerProgressHandle::with_slot_and_callback(Arc::clone(&slot), move |s| {
            ev_clone.lock().unwrap().push(s.clone());
        });
        assert!(!TrainerProgressHandle::default().is_active());
        assert!(root.is_active());

        let sub = root.subrange(200, 800);
        sub.emit(&snap);
        let first = slot.poll();
        assert_eq!(first.status, TrainingStatus::Running);
        assert_eq!(first.progress_permille, 585);
        assert!(first.has_visual_frame);

        // Regressing permille + omitting visual arrays must preserve monotonicity and visual frame.
        let scalar_only = TrainerProgressSnapshot {
            status: TrainingStatus::Running,
            phase: TrainingPhase::StageCKnobs,
            progress_permille: 100,
            ..Default::default()
        };
        sub.emit(&scalar_only);
        let second = slot.poll();
        assert_eq!(second.progress_permille, 585);
        assert!(second.has_visual_frame);
        assert_eq!(second.target_wave_i8, snap.target_wave_i8);
        assert_eq!(second.tier_ltas_db[2], snap.tier_ltas_db[2]);

        // Subrange Completed must stay Running at 800‰; root Completed hits 1_000‰.
        sub.emit(&TrainerProgressSnapshot {
            status: TrainingStatus::Completed,
            phase: TrainingPhase::Completed,
            progress_permille: 1_000,
            ..Default::default()
        });
        assert_eq!(slot.poll().status, TrainingStatus::Running);
        assert_eq!(slot.poll().progress_permille, 800);

        root.emit(&TrainerProgressSnapshot {
            status: TrainingStatus::Completed,
            phase: TrainingPhase::Completed,
            progress_permille: 1_000,
            ..Default::default()
        });
        let done = slot.poll();
        assert_eq!(done.status, TrainingStatus::Completed);
        assert_eq!(done.progress_permille, 1_000);
        assert_eq!(events.lock().unwrap().len(), 4);

        let exported = slot.export_bytes_256b();
        let (_, round) = TrainerProgressSnapshot::from_bytes_256b(&exported).unwrap();
        assert_eq!(round, done);
        slot.reset();
        assert_eq!(slot.poll().status, TrainingStatus::Idle);
    }
}
