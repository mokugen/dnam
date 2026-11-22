#![forbid(unsafe_code)]
//! Binary Builders & Verifiers for `.dnam-proj` (`DNPJ`, `v3.0`) Payload Capsules (`proj::capsules`):
//! `CALB` (`64 B`), `INPC` (`2.0–2.5 s` `fp32`), `STMC` (`4,800 B` header + `2.0–2.5 s` `fp32`),
//! and `STAT` (`uplo='U'` `StarCenter` sufficient statistics).
//!
//! Canonical Reference: `docs/specs/containers/project_container_format.md`
//! (`ref[container.proj.inpc-capsule]`, `ref[container.proj.stmc-capsule]`, `ref[container.proj.stat-uplo-u]`).

use core::mem::{align_of, offset_of, size_of};

use dnam_engine::container::crc32c_compute;
use dnam_engine::schema::DnamError;
use dnam_train::uplo_u_index;
use zerocopy::byteorder::little_endian::{F32, F64, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{CORESET_FP32_MAX_FRAMES_PER_STEM, CORESET_FP32_MIN_FRAMES_PER_STEM};

use super::{
    INPC_MAGIC, PROJ_STAT_BYTES_FULL, PROJ_STAT_BYTES_LITE, PROJ_STAT_BYTES_PRO,
    PROJ_STAT_BYTES_ULTRA, PROJ_VERSION_MAJOR, STMC_FLAG_ENABLED, STMC_FLAG_KNOWN_MASK,
    STMC_FLAG_STAR_CENTER, STMC_FLAG_STEREO, STMC_MAGIC, STMC_MAX_TRAJECTORY_WINDOWS,
};

/// Declarative `64 B` wire struct for a `CALB` loopback calibration chunk.
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CalbChunk64BWire {
    /// Magic `"CALB"`.
    pub magic: [u8; 4],
    /// Wire format version (`3`).
    pub version: U16,
    /// Bit-depth code (`0..=4`).
    pub bit_depth_code: u8,
    /// Polarity inversion flag (`0` or `1`).
    pub polarity_inverted: u8,
    /// Device sample rate in Hz (`8000..=384000`).
    pub sample_rate_device_hz: U32,
    /// Reserved pad (`0`).
    pub reserved_pad: [u8; 4],
    /// Sub-sample loopback lag at `48 kHz`.
    pub loopback_lag_samples: F64,
    /// Signed linear loopback gain.
    pub signed_gain: F64,
    /// Reference input calibration level in dBFS.
    pub input_reference_dbfs: F32,
    /// Measured quiescent hardware noise floor in dBFS.
    pub noise_floor_dbfs: F32,
    /// Reserved tail (`0`).
    pub reserved_tail: [u8; 20],
    /// Castagnoli `CRC32C` over `bytes[0..60]`.
    pub crc32c: U32,
}

const _: () = assert!(size_of::<CalbChunk64BWire>() == 64);
const _: () = assert!(align_of::<CalbChunk64BWire>() == 1);
const _: () = assert!(offset_of!(CalbChunk64BWire, sample_rate_device_hz) == 8);
const _: () = assert!(offset_of!(CalbChunk64BWire, loopback_lag_samples) == 16);
const _: () = assert!(offset_of!(CalbChunk64BWire, input_reference_dbfs) == 32);
const _: () = assert!(offset_of!(CalbChunk64BWire, reserved_tail) == 40);
const _: () = assert!(offset_of!(CalbChunk64BWire, crc32c) == 60);

/// Fixed `64 B` `CALB` loopback calibration chunk (`r[container.proj.calb]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalbChunk64B {
    /// Device sample rate in Hz (`8000..=384000`).
    pub sample_rate_device_hz: u32,
    /// Bit-depth code (`0 = Int12`, `1 = Int16`, `2 = Int24`, `3 = Int32`, `4 = Float32`).
    pub bit_depth_code: u8,
    /// Polarity inversion flag (`0` or `1`).
    pub polarity_inverted: bool,
    /// Sub-sample loopback latency in samples at `48 kHz`.
    pub loopback_lag_samples: f64,
    /// Signed linear loopback gain.
    pub signed_gain: f64,
    /// Calibrated input reference level in dBFS.
    pub input_reference_dbfs: f32,
    /// Measured quiescent hardware noise floor in dBFS.
    pub noise_floor_dbfs: f32,
}

impl CalbChunk64B {
    /// Serializes this `CALB` record into an exact `64 B` little-endian array.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut wire = CalbChunk64BWire {
            magic: *b"CALB",
            version: U16::new(PROJ_VERSION_MAJOR),
            bit_depth_code: self.bit_depth_code,
            polarity_inverted: u8::from(self.polarity_inverted),
            sample_rate_device_hz: U32::new(self.sample_rate_device_hz),
            reserved_pad: [0; 4],
            loopback_lag_samples: F64::new(self.loopback_lag_samples),
            signed_gain: F64::new(self.signed_gain),
            input_reference_dbfs: F32::new(self.input_reference_dbfs),
            noise_floor_dbfs: F32::new(self.noise_floor_dbfs),
            reserved_tail: [0; 20],
            crc32c: U32::new(0),
        };
        let crc = crc32c_compute(&wire.as_bytes()[0..60]);
        wire.crc32c = U32::new(crc);
        let mut out = [0u8; 64];
        out.copy_from_slice(wire.as_bytes());
        out
    }

    /// Parses and validates a `64 B` `CALB` chunk.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        let Ok(wire) = CalbChunk64BWire::ref_from_bytes(bytes) else {
            return Err(DnamError::BufferTooSmall {
                expected: 64,
                actual: bytes.len(),
            });
        };
        if &wire.magic != b"CALB" {
            return Err(DnamError::InvalidMagic(wire.magic));
        }
        let ver = wire.version.get();
        if ver != PROJ_VERSION_MAJOR {
            return Err(DnamError::UnsupportedVersion {
                major: ver,
                minor: 0,
            });
        }
        let bit_depth_code = wire.bit_depth_code;
        let pol_byte = wire.polarity_inverted;
        if bit_depth_code > 4
            || pol_byte > 1
            || wire.reserved_pad != [0; 4]
            || wire.reserved_tail != [0; 20]
        {
            return Err(DnamError::SchemaValidationError(
                "invalid CALB flags or non-zero reserved bytes".to_string(),
            ));
        }
        let expected_crc = wire.crc32c.get();
        let calc_crc = crc32c_compute(&bytes[0..60]);
        if calc_crc != expected_crc {
            return Err(DnamError::Crc32cMismatch {
                region: "CALB",
                expected: expected_crc,
                computed: calc_crc,
            });
        }
        let sample_rate_device_hz = wire.sample_rate_device_hz.get();
        if !(8_000..=384_000).contains(&sample_rate_device_hz) {
            return Err(DnamError::InvalidSampleRate(sample_rate_device_hz));
        }
        let loopback_lag_samples = wire.loopback_lag_samples.get();
        let signed_gain = wire.signed_gain.get();
        let input_reference_dbfs = wire.input_reference_dbfs.get();
        let noise_floor_dbfs = wire.noise_floor_dbfs.get();
        if !loopback_lag_samples.is_finite()
            || !signed_gain.is_finite()
            || !input_reference_dbfs.is_finite()
            || !noise_floor_dbfs.is_finite()
        {
            return Err(DnamError::SchemaValidationError(
                "non-finite float in CALB chunk".to_string(),
            ));
        }
        Ok(Self {
            sample_rate_device_hz,
            bit_depth_code,
            polarity_inverted: pol_byte != 0,
            loopback_lag_samples,
            signed_gain,
            input_reference_dbfs,
            noise_floor_dbfs,
        })
    }
}

/// Declarative `64 B` wire header of an `INPC` dry stimulus capsule (`0..64`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InpcHeader64BWire {
    /// Magic `"INPC"`.
    pub magic: [u8; 4],
    /// Wire format version (`3`).
    pub version: U16,
    /// Channel count (`1` or `2`).
    pub channels: U16,
    /// Canonical sample rate (`48,000`).
    pub sample_rate_hz: U32,
    /// Coreset frame count (`96,000..=120,000`, divisible by `16`).
    pub num_frames: U32,
    /// Reserved zero tail (`40 B`).
    pub reserved: [u8; 40],
    /// Total capsule byte length (`640 + channels * num_frames * 4`).
    pub rec_len: U32,
    /// Castagnoli `CRC32C` over `bytes[64..rec_len]`.
    pub crc32c: U32,
}

const _: () = assert!(size_of::<InpcHeader64BWire>() == 64);
const _: () = assert!(align_of::<InpcHeader64BWire>() == 1);
const _: () = assert!(offset_of!(InpcHeader64BWire, num_frames) == 12);
const _: () = assert!(offset_of!(InpcHeader64BWire, rec_len) == 56);
const _: () = assert!(offset_of!(InpcHeader64BWire, crc32c) == 60);

fn verify_finite_fp32_stream(fp32_bytes: &[u8], label: &'static str) -> Result<(), DnamError> {
    let Ok(words) = <[F32]>::ref_from_bytes(fp32_bytes) else {
        return Err(DnamError::SchemaValidationError(format!(
            "{label} fp32 payload length {} is not a multiple of 4 B",
            fp32_bytes.len()
        )));
    };
    for word in words {
        if !word.get().is_finite() {
            return Err(DnamError::SchemaValidationError(format!(
                "non-finite IEEE-754 binary32 sample in {label}"
            )));
        }
    }
    Ok(())
}

fn validate_coreset_fp32_payload(
    payload: &[u8],
    channels: u16,
    label: &'static str,
) -> Result<usize, DnamError> {
    let frame_bytes = usize::from(channels) * 4;
    if frame_bytes == 0 || !payload.len().is_multiple_of(frame_bytes) {
        return Err(DnamError::SchemaValidationError(format!(
            "{label} coreset length {} is not divisible by {frame_bytes}",
            payload.len()
        )));
    }
    let num_frames = payload.len() / frame_bytes;
    if !(CORESET_FP32_MIN_FRAMES_PER_STEM..=CORESET_FP32_MAX_FRAMES_PER_STEM).contains(&num_frames)
        || !num_frames.is_multiple_of(16)
    {
        return Err(DnamError::SchemaValidationError(format!(
            "{label} frame count {num_frames} must be a multiple of 16 in {CORESET_FP32_MIN_FRAMES_PER_STEM}..={CORESET_FP32_MAX_FRAMES_PER_STEM}"
        )));
    }
    verify_finite_fp32_stream(payload, label)?;
    Ok(num_frames)
}

fn decode_fp32_interleaved(bytes: &[u8], is_stereo: bool) -> (Vec<f32>, Option<Vec<f32>>) {
    let aligned_len = bytes.len() & !3;
    let f32_vals: Vec<f32> = <[F32]>::ref_from_bytes(&bytes[..aligned_len])
        .map(|s| s.iter().map(|w| w.get()).collect())
        .unwrap_or_default();
    if !is_stereo {
        return (f32_vals, None);
    }
    let half = f32_vals.len() / 2;
    let mut l = Vec::with_capacity(half);
    let mut r = Vec::with_capacity(half);
    for pair in f32_vals.as_chunks::<2>().0 {
        l.push(pair[0]);
        r.push(pair[1]);
    }
    (l, Some(r))
}

/// Validated zero-copy view over a `2.0–2.5 s` (`96k–120k`) `fp32` `INPC` dry stimulus capsule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InpcCapsuleView<'a> {
    /// Channel count (`1` = mono, `2` = stereo).
    pub channels: u16,
    /// Canonical sample rate (`48,000 Hz`).
    pub sample_rate_hz: u32,
    /// Coreset frame count (`96,000..=120,000`, `16`-frame multiple).
    pub num_frames: u32,
    /// `256 B` deterministic stimulus seed $\boldsymbol{\Xi}$.
    pub seed_256: &'a [u8],
    /// `320 B` `AdaptiveStimulusController` frame schedule / Stage-II basis summary.
    pub schedule_320: &'a [u8],
    /// `64 B`-aligned IEEE-754 `binary32` (`fp32`) dry coreset payload.
    pub dry_coreset_fp32: &'a [u8],
}

/// Decodes an [`InpcCapsuleView`] `fp32` payload into de-interleaved `(dry_l, Option<dry_r>)`.
#[must_use]
pub fn decode_proj_inpc_f32(view: &InpcCapsuleView<'_>) -> (Vec<f32>, Option<Vec<f32>>) {
    decode_fp32_interleaved(view.dry_coreset_fp32, view.channels == 2)
}

/// Builds a canonical `2.0–2.5 s` `fp32` `INPC` dry stimulus capsule (`r[container.proj.inpc-capsule]`).
pub fn build_proj_inpc_capsule(
    channels: u16,
    seed_256: &[u8; 256],
    schedule_320: &[u8; 320],
    dry_coreset_fp32: &[u8],
) -> Result<Vec<u8>, DnamError> {
    if !matches!(channels, 1 | 2) {
        return Err(DnamError::SchemaValidationError(format!(
            "INPC channels must be 1 or 2, got {channels}"
        )));
    }
    let num_frames = validate_coreset_fp32_payload(dry_coreset_fp32, channels, "INPC")?;

    let rec_len = 640 + dry_coreset_fp32.len();
    let mut out = vec![0u8; rec_len];
    out[64..320].copy_from_slice(seed_256);
    out[320..640].copy_from_slice(schedule_320);
    out[640..rec_len].copy_from_slice(dry_coreset_fp32);
    let crc = crc32c_compute(&out[64..rec_len]);
    let hdr = InpcHeader64BWire {
        magic: INPC_MAGIC,
        version: U16::new(PROJ_VERSION_MAJOR),
        channels: U16::new(channels),
        sample_rate_hz: U32::new(48_000),
        num_frames: U32::new(num_frames as u32),
        reserved: [0; 40],
        rec_len: U32::new(rec_len as u32),
        crc32c: U32::new(crc),
    };
    out[0..64].copy_from_slice(hdr.as_bytes());
    Ok(out)
}

/// Validates a `2.0–2.5 s` `fp32` `INPC` dry stimulus capsule (`r[container.proj.inpc-capsule]`).
pub fn verify_proj_inpc_capsule(bytes: &[u8]) -> Result<InpcCapsuleView<'_>, DnamError> {
    let Ok((hdr, _)) = InpcHeader64BWire::ref_from_prefix(bytes) else {
        return Err(DnamError::BufferTooSmall {
            expected: 64,
            actual: bytes.len(),
        });
    };
    if hdr.magic != INPC_MAGIC {
        return Err(DnamError::InvalidMagic(hdr.magic));
    }
    let ver = hdr.version.get();
    if ver != PROJ_VERSION_MAJOR {
        return Err(DnamError::UnsupportedVersion {
            major: ver,
            minor: 0,
        });
    }
    let channels = hdr.channels.get();
    if !matches!(channels, 1 | 2) {
        return Err(DnamError::SchemaValidationError(format!(
            "invalid INPC channels {channels}"
        )));
    }
    let sample_rate_hz = hdr.sample_rate_hz.get();
    let num_frames = hdr.num_frames.get();
    if hdr.reserved != [0; 40] {
        return Err(DnamError::SchemaValidationError(
            "non-zero reserved bytes in INPC[16..56]".to_string(),
        ));
    }
    let nf = num_frames as usize;
    if sample_rate_hz != 48_000
        || !(CORESET_FP32_MIN_FRAMES_PER_STEM..=CORESET_FP32_MAX_FRAMES_PER_STEM).contains(&nf)
        || !nf.is_multiple_of(16)
    {
        return Err(DnamError::SchemaValidationError(
            "invalid INPC sample_rate_hz or num_frames".to_string(),
        ));
    }
    let expected_len = 640 + nf * usize::from(channels) * 4;
    let rec_len = hdr.rec_len.get() as usize;
    if rec_len != expected_len {
        return Err(DnamError::SchemaValidationError(
            "invalid INPC rec_len".to_string(),
        ));
    }
    if bytes.len() != expected_len {
        return Err(DnamError::BufferTooSmall {
            expected: expected_len,
            actual: bytes.len(),
        });
    }
    let expected_crc = hdr.crc32c.get();
    let calc_crc = crc32c_compute(&bytes[64..rec_len]);
    if calc_crc != expected_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "INPC",
            expected: expected_crc,
            computed: calc_crc,
        });
    }
    let dry_coreset_fp32 = &bytes[640..rec_len];
    verify_finite_fp32_stream(dry_coreset_fp32, "INPC")?;
    Ok(InpcCapsuleView {
        channels,
        sample_rate_hz,
        num_frames,
        seed_256: &bytes[64..320],
        schedule_320: &bytes[320..640],
        dry_coreset_fp32,
    })
}

/// Declarative `16 B` wire record for an `STMC` trajectory window.
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct StmcTrajectoryWindowWire {
    /// Control index `p`.
    pub control_id: U16,
    /// Control kind (`0` = Range, `1` = Switch).
    pub ctrl_kind: u8,
    /// Active discrete switch option index.
    pub active_opt: u8,
    /// Physical control value in this window.
    pub u_phys: F32,
    /// Leverage / SNR quadrature weight `>= 0`.
    pub window_weight: F32,
    /// Measured EMR velocity metric.
    pub emr_velocity: F32,
}

const _: () = assert!(size_of::<StmcTrajectoryWindowWire>() == 16);
const _: () = assert!(align_of::<StmcTrajectoryWindowWire>() == 1);
const _: () = assert!(offset_of!(StmcTrajectoryWindowWire, u_phys) == 4);
const _: () = assert!(offset_of!(StmcTrajectoryWindowWire, window_weight) == 8);
const _: () = assert!(offset_of!(StmcTrajectoryWindowWire, emr_velocity) == 12);

/// Single `16 B` entry in the `576 B` (`R <= 36`) Windowed Control Trajectory Table of `STMC`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StmcTrajectoryWindow {
    /// Control index `p` in `manifest.base.controls_schema`.
    pub control_id: u16,
    /// Control kind (`0` = Range, `1` = Switch).
    pub ctrl_kind: u8,
    /// Active discrete switch option index (`0` for Range controls).
    pub active_opt: u8,
    /// Physical control value $u_{p,r}$ in this `96 ms` window.
    pub u_phys: f32,
    /// Leverage / SNR quadrature weight $w_r \ge 0$ for this window.
    pub window_weight: f32,
    /// Measured EMR velocity metric in this window.
    pub emr_velocity: f32,
}

impl StmcTrajectoryWindow {
    /// Encodes this window into `16 B` little-endian.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 16] {
        let wire = StmcTrajectoryWindowWire {
            control_id: U16::new(self.control_id),
            ctrl_kind: self.ctrl_kind,
            active_opt: self.active_opt,
            u_phys: F32::new(self.u_phys),
            window_weight: F32::new(self.window_weight),
            emr_velocity: F32::new(self.emr_velocity),
        };
        let mut out = [0u8; 16];
        out.copy_from_slice(wire.as_bytes());
        out
    }

    /// Decodes and validates a `16 B` trajectory window.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        let Ok(wire) = StmcTrajectoryWindowWire::ref_from_bytes(bytes) else {
            return Err(DnamError::BufferTooSmall {
                expected: 16,
                actual: bytes.len(),
            });
        };
        let control_id = wire.control_id.get();
        let ctrl_kind = wire.ctrl_kind;
        let active_opt = wire.active_opt;
        let u_phys = wire.u_phys.get();
        let window_weight = wire.window_weight.get();
        let emr_velocity = wire.emr_velocity.get();
        if ctrl_kind > 1
            || !u_phys.is_finite()
            || !window_weight.is_finite()
            || window_weight < 0.0
            || !emr_velocity.is_finite()
        {
            return Err(DnamError::SchemaValidationError(
                "invalid StmcTrajectoryWindow ctrl_kind or non-finite float".to_string(),
            ));
        }
        Ok(Self {
            control_id,
            ctrl_kind,
            active_opt,
            u_phys,
            window_weight,
            emr_velocity,
        })
    }
}

/// Declarative `128 B` wire header of an `STMC` stem capture capsule (`0..128`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct StmcHeader128BWire {
    /// Magic `"STMC"`.
    pub magic: [u8; 4],
    /// Wire format version (`3`).
    pub version: U16,
    /// Numeric stem index.
    pub stem_index: U16,
    /// `STMC_FLAG_*` bitmask.
    pub flags: U16,
    /// Trajectory window count (`1..=36`).
    pub n_windows: U16,
    /// Input pad gain.
    pub pad_gain_f32: F32,
    /// Sub-sample alignment lag.
    pub fractional_lag_samples_f32: F32,
    /// `88 B` band-energy PSD summary (`10 × f64le` upper triangle + `total_power: f64le`).
    pub band_psd_summary_88: [u8; 88],
    /// Coreset frame count.
    pub num_frames: U32,
    /// Measured comb bin count.
    pub n_comb_bins: U32,
    /// Total capsule byte length.
    pub rec_len: U32,
    /// Castagnoli `CRC32C` over `bytes[128..rec_len]`.
    pub body_crc32c: U32,
    /// Reserved zero bytes (`0`).
    pub reserved: [u8; 4],
}

const _: () = assert!(size_of::<StmcHeader128BWire>() == 128);
const _: () = assert!(align_of::<StmcHeader128BWire>() == 1);
const _: () = assert!(offset_of!(StmcHeader128BWire, flags) == 8);
const _: () = assert!(offset_of!(StmcHeader128BWire, band_psd_summary_88) == 20);
const _: () = assert!(offset_of!(StmcHeader128BWire, num_frames) == 108);
const _: () = assert!(offset_of!(StmcHeader128BWire, body_crc32c) == 120);
const _: () = assert!(offset_of!(StmcHeader128BWire, reserved) == 124);

/// Validated view over a `STMC` `StemCaptureRecord` capsule.
#[derive(Debug, Clone, PartialEq)]
pub struct StmcCapsuleView<'a> {
    /// Numeric stem index (`0..=65535`).
    pub stem_index: u16,
    /// `STMC_FLAG_*` bitmask (Bit 0 = `enabled`).
    pub flags: u16,
    /// Analog/digital input pad gain applied during capture.
    pub pad_gain_f32: f32,
    /// Sub-sample alignment lag applied to this stem.
    pub fractional_lag_samples_f32: f32,
    /// Decoded trajectory windows (`1..=36`).
    pub windows: Vec<StmcTrajectoryWindow>,
    /// `4,096 B` spectral, sideband & ballistic summary slice (`704..4_800`).
    pub spectral_summary_4096: &'a [u8],
    /// `64 B`-aligned IEEE-754 `binary32` (`fp32`) wet coreset slice (`4_800..rec_len`).
    pub wet_coreset_fp32: &'a [u8],
}

/// Decodes an [`StmcCapsuleView`] `fp32` payload into de-interleaved `(wet_l, Option<wet_r>)`.
#[must_use]
pub fn decode_proj_stmc_f32(view: &StmcCapsuleView<'_>) -> (Vec<f32>, Option<Vec<f32>>) {
    decode_fp32_interleaved(view.wet_coreset_fp32, (view.flags & STMC_FLAG_STEREO) != 0)
}

impl StmcCapsuleView<'_> {
    /// Returns `true` if `STMC_FLAG_ENABLED` (`Bit 0`) is set.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        (self.flags & STMC_FLAG_ENABLED) != 0
    }

    /// Returns `true` if `STMC_FLAG_STAR_CENTER` (`Bit 1`) is set.
    #[must_use]
    pub const fn is_star_center(&self) -> bool {
        (self.flags & STMC_FLAG_STAR_CENTER) != 0
    }
}

/// Builds a canonical `2.0–2.5 s` `fp32` `STMC` `StemCaptureRecord` capsule (`r[container.proj.stmc-capsule]`).
#[allow(clippy::too_many_arguments)]
pub fn build_proj_stmc_capsule(
    stem_index: u16,
    flags: u16,
    pad_gain_f32: f32,
    fractional_lag_samples_f32: f32,
    band_psd_summary_88: &[u8; 88],
    n_comb_bins: u32,
    windows: &[StmcTrajectoryWindow],
    spectral_summary_4096: &[u8; 4_096],
    wet_coreset_fp32: &[u8],
) -> Result<Vec<u8>, DnamError> {
    if (flags & !STMC_FLAG_KNOWN_MASK) != 0 {
        return Err(DnamError::SchemaValidationError(format!(
            "unknown STMC flags {flags:#06x}"
        )));
    }
    if windows.is_empty() || windows.len() > STMC_MAX_TRAJECTORY_WINDOWS {
        return Err(DnamError::SchemaValidationError(format!(
            "STMC windows count {} must be in 1..={STMC_MAX_TRAJECTORY_WINDOWS}",
            windows.len()
        )));
    }
    if !pad_gain_f32.is_finite() || !fractional_lag_samples_f32.is_finite() {
        return Err(DnamError::SchemaValidationError(
            "non-finite pad_gain_f32 or fractional_lag_samples_f32 in STMC".to_string(),
        ));
    }
    let channels = if (flags & STMC_FLAG_STEREO) != 0 {
        2_u16
    } else {
        1_u16
    };
    let num_frames = validate_coreset_fp32_payload(wet_coreset_fp32, channels, "STMC")?;

    let rec_len = 4_800 + wet_coreset_fp32.len();
    let mut out = vec![0u8; rec_len];
    for (idx, w) in windows.iter().enumerate() {
        let _ = StmcTrajectoryWindow::from_bytes(&w.to_bytes())?;
        let off = 128 + idx * 16;
        out[off..off + 16].copy_from_slice(&w.to_bytes());
    }
    out[704..4_800].copy_from_slice(spectral_summary_4096);
    out[4_800..rec_len].copy_from_slice(wet_coreset_fp32);
    let body_crc = crc32c_compute(&out[128..rec_len]);
    let hdr = StmcHeader128BWire {
        magic: STMC_MAGIC,
        version: U16::new(PROJ_VERSION_MAJOR),
        stem_index: U16::new(stem_index),
        flags: U16::new(flags),
        n_windows: U16::new(windows.len() as u16),
        pad_gain_f32: F32::new(pad_gain_f32),
        fractional_lag_samples_f32: F32::new(fractional_lag_samples_f32),
        band_psd_summary_88: *band_psd_summary_88,
        num_frames: U32::new(num_frames as u32),
        n_comb_bins: U32::new(n_comb_bins),
        rec_len: U32::new(rec_len as u32),
        body_crc32c: U32::new(body_crc),
        reserved: [0; 4],
    };
    out[0..128].copy_from_slice(hdr.as_bytes());
    Ok(out)
}

/// Toggles `STMC_FLAG_ENABLED` (`Bit 0`) in-place inside a validated `STMC` capsule byte buffer.
pub fn set_stmc_capsule_enabled_flag(
    stmc_bytes: &mut [u8],
    enabled: bool,
) -> Result<(), DnamError> {
    let view = verify_proj_stmc_capsule(stmc_bytes)?;
    let new_flags = if enabled {
        view.flags | STMC_FLAG_ENABLED
    } else {
        view.flags & !STMC_FLAG_ENABLED
    };
    let stmc_bytes_len = stmc_bytes.len();
    let (hdr, _) =
        StmcHeader128BWire::mut_from_prefix(stmc_bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: 128,
            actual: stmc_bytes_len,
        })?;
    hdr.flags = U16::new(new_flags);
    Ok(())
}

/// Toggles `STMC_FLAG_STAR_CENTER` (`Bit 1`) in-place inside a validated `STMC` capsule byte buffer.
pub fn set_stmc_capsule_star_center_flag(
    stmc_bytes: &mut [u8],
    is_star_center: bool,
) -> Result<(), DnamError> {
    let view = verify_proj_stmc_capsule(stmc_bytes)?;
    let new_flags = if is_star_center {
        view.flags | STMC_FLAG_STAR_CENTER
    } else {
        view.flags & !STMC_FLAG_STAR_CENTER
    };
    let stmc_bytes_len = stmc_bytes.len();
    let (hdr, _) =
        StmcHeader128BWire::mut_from_prefix(stmc_bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: 128,
            actual: stmc_bytes_len,
        })?;
    hdr.flags = U16::new(new_flags);
    Ok(())
}

/// Validates a `2.0–2.5 s` `fp32` `STMC` `StemCaptureRecord` capsule (`r[container.proj.stmc-capsule]`).
pub fn verify_proj_stmc_capsule(bytes: &[u8]) -> Result<StmcCapsuleView<'_>, DnamError> {
    let Ok((hdr, _)) = StmcHeader128BWire::ref_from_prefix(bytes) else {
        return Err(DnamError::BufferTooSmall {
            expected: 128,
            actual: bytes.len(),
        });
    };
    if hdr.magic != STMC_MAGIC {
        return Err(DnamError::InvalidMagic(hdr.magic));
    }
    let ver = hdr.version.get();
    if ver != PROJ_VERSION_MAJOR {
        return Err(DnamError::UnsupportedVersion {
            major: ver,
            minor: 0,
        });
    }
    let stem_index = hdr.stem_index.get();
    let flags = hdr.flags.get();
    if (flags & !STMC_FLAG_KNOWN_MASK) != 0 {
        return Err(DnamError::SchemaValidationError(format!(
            "unknown STMC flags {flags:#06x}"
        )));
    }
    let n_windows = hdr.n_windows.get() as usize;
    if n_windows == 0 || n_windows > STMC_MAX_TRAJECTORY_WINDOWS {
        return Err(DnamError::SchemaValidationError(format!(
            "invalid STMC n_trajectory_windows {n_windows}"
        )));
    }
    let pad_gain_f32 = hdr.pad_gain_f32.get();
    let fractional_lag_samples_f32 = hdr.fractional_lag_samples_f32.get();
    if !pad_gain_f32.is_finite() || !fractional_lag_samples_f32.is_finite() {
        return Err(DnamError::SchemaValidationError(
            "non-finite pad_gain_f32 or fractional_lag_samples_f32 in STMC".to_string(),
        ));
    }
    let n_coreset_frames = hdr.num_frames.get();
    let rec_len = hdr.rec_len.get() as usize;
    let expected_crc = hdr.body_crc32c.get();
    if hdr.reserved != [0; 4] {
        return Err(DnamError::SchemaValidationError(
            "non-zero reserved bytes in STMC[124..128]".to_string(),
        ));
    }
    let channels = if (flags & STMC_FLAG_STEREO) != 0 {
        2usize
    } else {
        1usize
    };
    let nf = n_coreset_frames as usize;
    if !(CORESET_FP32_MIN_FRAMES_PER_STEM..=CORESET_FP32_MAX_FRAMES_PER_STEM).contains(&nf)
        || !nf.is_multiple_of(16)
    {
        return Err(DnamError::SchemaValidationError(
            "invalid STMC n_coreset_frames".to_string(),
        ));
    }
    let expected_len = 4_800 + nf * channels * 4;
    if rec_len != expected_len {
        return Err(DnamError::SchemaValidationError(
            "invalid STMC rec_len".to_string(),
        ));
    }
    if bytes.len() != expected_len {
        return Err(DnamError::BufferTooSmall {
            expected: expected_len,
            actual: bytes.len(),
        });
    }
    let calc_crc = crc32c_compute(&bytes[128..rec_len]);
    if calc_crc != expected_crc {
        return Err(DnamError::Crc32cMismatch {
            region: "STMC",
            expected: expected_crc,
            computed: calc_crc,
        });
    }
    let mut windows = Vec::with_capacity(n_windows);
    for idx in 0..n_windows {
        let off = 128 + idx * 16;
        windows.push(StmcTrajectoryWindow::from_bytes(&bytes[off..off + 16])?);
    }
    let tail_start = 128 + n_windows * 16;
    if bytes[tail_start..704].iter().any(|&b| b != 0) {
        return Err(DnamError::SchemaValidationError(
            "non-zero unused trajectory window slot in STMC[..704]".to_string(),
        ));
    }
    let wet_coreset_fp32 = &bytes[4_800..rec_len];
    verify_finite_fp32_stream(wet_coreset_fp32, "STMC")?;
    Ok(StmcCapsuleView {
        stem_index,
        flags,
        pad_gain_f32,
        fractional_lag_samples_f32,
        windows,
        spectral_summary_4096: &bytes[704..4_800],
        wet_coreset_fp32,
    })
}

/// Validates a `STAT` `StarCenter` `uplo='U'` sufficient-statistics chunk (`r[container.proj.stat-uplo-u]`),
/// returning the retained tier dimension (`64`, `128`, `192`, or `256`).
pub fn verify_proj_stat_uplo_u(bytes: &[u8]) -> Result<usize, DnamError> {
    let dim = match bytes.len() {
        PROJ_STAT_BYTES_LITE => 64,
        PROJ_STAT_BYTES_FULL => 128,
        PROJ_STAT_BYTES_PRO => 192,
        PROJ_STAT_BYTES_ULTRA => 256,
        other => {
            return Err(DnamError::SchemaValidationError(format!(
                "invalid STAT byte length {other}: expected {PROJ_STAT_BYTES_LITE}, {PROJ_STAT_BYTES_FULL}, {PROJ_STAT_BYTES_PRO}, or {PROJ_STAT_BYTES_ULTRA}"
            )));
        }
    };
    let Ok(words) = <[F64]>::ref_from_bytes(bytes) else {
        return Err(DnamError::SchemaValidationError(
            "STAT byte length is not a multiple of 8 B".to_string(),
        ));
    };
    for elem in words {
        if !elem.get().is_finite() {
            return Err(DnamError::SchemaValidationError(
                "non-finite f64 element in STAT uplo='U' chunk".to_string(),
            ));
        }
    }
    for j in 0..dim {
        let diag_idx = uplo_u_index(j, j);
        let g_jj = words[diag_idx].get();
        if g_jj < 0.0 {
            return Err(DnamError::SchemaValidationError(format!(
                "negative Gram diagonal G[{j},{j}] = {g_jj} in STAT uplo='U' chunk"
            )));
        }
    }
    Ok(dim)
}
