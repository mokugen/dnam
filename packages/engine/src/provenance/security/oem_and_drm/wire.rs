//! Declarative `#[repr(C)]` `zerocopy` wire headers and directories for `.dnam-seal` (`DNSL`),
//! protected inner framing (`DNPL`), `DME1` encrypted mutable frames, and HPKE record framing
//! (`ref[repo.code.zerocopy-wire-structs]`, `ref[sec.drm.dnsl-public-framing]`,
//! `ref[sec.drm.protected-inner-framing]`, `ref[sec.drm.dme1-mutable-suffix]`).

use core::mem::{align_of, offset_of, size_of};

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::schema::{DnamError, reject_if};

/// Canonical magic bytes for `.dnam-seal` (`DNSL`) public framing.
pub const DNSL_WIRE_MAGIC: [u8; 4] = *b"DNSL";
/// Canonical magic bytes for protected inner `DNPL` logical metadata prefixes.
pub const DNPL_WIRE_MAGIC: [u8; 4] = *b"DNPL";
/// Canonical magic bytes for `DME1` encrypted mutable frames.
pub const DME1_WIRE_MAGIC: [u8; 4] = *b"DME1";
/// Canonical `v3.0` container wire major version.
pub const DRM_WIRE_VERSION_MAJOR: u16 = 3;
/// Canonical `v3.0` container wire minor version.
pub const DRM_WIRE_VERSION_MINOR: u16 = 0;

/// Fixed 64-byte `DNSL` public preamble (`#[repr(C, align(64))]`, `ref[sec.drm.dnsl-public-framing]`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct DnslPreambleWire {
    /// Magic bytes (`b"DNSL"`).
    pub magic: [u8; 4],
    /// Wire major version (`3`).
    pub version_major: U16,
    /// Wire minor version (`0`).
    pub version_minor: U16,
    /// Container protection profile (`2 = EnclaveSealed`).
    pub protection_profile: u8,
    /// Issued maximum tier index (`0..=3`).
    pub maximum_tier: u8,
    /// Zero-filled reserved bytes (`10..64`).
    pub reserved: [u8; 54],
}

const _: () = assert!(size_of::<DnslPreambleWire>() == 64);
const _: () = assert!(align_of::<DnslPreambleWire>() == 64);
const _: () = assert!(offset_of!(DnslPreambleWire, version_major) == 4);
const _: () = assert!(offset_of!(DnslPreambleWire, protection_profile) == 8);
const _: () = assert!(offset_of!(DnslPreambleWire, reserved) == 10);

impl DnslPreambleWire {
    /// Constructs a canonical `v3.0` `DnslPreambleWire` for `EnclaveSealed` (`profile = 2`).
    pub fn new(maximum_tier: u8) -> Result<Self, DnamError> {
        reject_if(maximum_tier > 3, "unsupported DNSL maximum_tier")?;
        Ok(Self {
            magic: DNSL_WIRE_MAGIC,
            version_major: U16::new(DRM_WIRE_VERSION_MAJOR),
            version_minor: U16::new(DRM_WIRE_VERSION_MINOR),
            protection_profile: 2,
            maximum_tier,
            reserved: [0u8; 54],
        })
    }

    /// Parses and validates a 64-byte `DnslPreambleWire` via safe `zerocopy`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        let (wire, _) = Self::read_from_prefix(bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: size_of::<Self>(),
            actual: bytes.len(),
        })?;
        if wire.magic != DNSL_WIRE_MAGIC {
            return Err(DnamError::InvalidMagic(wire.magic));
        }
        let major = wire.version_major.get();
        let minor = wire.version_minor.get();
        if major != DRM_WIRE_VERSION_MAJOR || minor != DRM_WIRE_VERSION_MINOR {
            return Err(DnamError::UnsupportedVersion { major, minor });
        }
        reject_if(
            wire.protection_profile != 2
                || wire.maximum_tier > 3
                || wire.reserved.iter().any(|&b| b != 0),
            "invalid DNSL preamble fields",
        )?;
        Ok(wire)
    }

    /// Serializes this preamble into a 64-byte little-endian wire block.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        let _ = self.write_to_prefix(&mut out);
        out
    }
}

/// Fixed 128-byte `DNSL` public directory (`#[repr(C)]`, `ref[sec.drm.dnsl-public-framing]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DnslDirectoryWire {
    pub plan_offset: U64,
    pub plan_length: U64,
    pub slots_offset: U64,
    pub slots_length: U64,
    pub manifest_offset: U64,
    pub manifest_length: U64,
    pub shared_record_offset: U64,
    pub shared_ciphertext_length: U64,
    pub original_tier_ends: [U64; 4],
    pub reserved: [u8; 32],
}

const _: () = assert!(size_of::<DnslDirectoryWire>() == 128);
const _: () = assert!(align_of::<DnslDirectoryWire>() == 1);
const _: () = assert!(offset_of!(DnslDirectoryWire, shared_record_offset) == 48);
const _: () = assert!(offset_of!(DnslDirectoryWire, original_tier_ends) == 64);
const _: () = assert!(offset_of!(DnslDirectoryWire, reserved) == 96);

impl DnslDirectoryWire {
    /// Parses and validates a 128-byte `DnslDirectoryWire` via safe `zerocopy`.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, DnamError> {
        let (wire, _) = Self::ref_from_prefix(bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: size_of::<Self>(),
            actual: bytes.len(),
        })?;
        reject_if(
            wire.reserved.iter().any(|&b| b != 0) || wire.plan_offset.get() != 192,
            "invalid DNSL directory fields",
        )?;
        Ok(wire)
    }
}

/// Fixed 128-byte protected inner directory (`#[repr(C)]`, `ref[sec.drm.protected-inner-framing]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InnerDirectoryWire {
    pub plan_offset: U64,
    pub plan_length: U64,
    pub slots_offset: U64,
    pub slots_length: U64,
    pub authentication_offset: U64,
    pub authentication_length: U64,
    pub shared_record_offset: U64,
    pub shared_ciphertext_length: U64,
    pub original_tier_ends: [U64; 4],
    pub logical_metadata_sha256: [u8; 32],
}

const _: () = assert!(size_of::<InnerDirectoryWire>() == 128);
const _: () = assert!(align_of::<InnerDirectoryWire>() == 1);
const _: () = assert!(offset_of!(InnerDirectoryWire, authentication_offset) == 32);
const _: () = assert!(offset_of!(InnerDirectoryWire, original_tier_ends) == 64);
const _: () = assert!(offset_of!(InnerDirectoryWire, logical_metadata_sha256) == 96);

impl InnerDirectoryWire {
    /// Parses an `InnerDirectoryWire` prefix via safe `zerocopy`.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, DnamError> {
        let (wire, _) = Self::ref_from_prefix(bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: size_of::<Self>(),
            actual: bytes.len(),
        })?;
        reject_if(
            wire.plan_offset.get() < 192 || wire.authentication_length.get() == 0,
            "invalid inner directory fields",
        )?;
        Ok(wire)
    }
}

/// Fixed 64-byte `DNPL` protected logical-metadata preamble (`#[repr(C, align(64))]`,
/// `ref[sec.drm.protected-inner-framing]`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct DnplPreambleWire {
    /// Magic bytes (`b"DNPL"`).
    pub magic: [u8; 4],
    /// Wire major version (`3`).
    pub version_major: U16,
    /// Wire minor version (`0`).
    pub version_minor: U16,
    /// Protection profile (`1 = DeviceAllowlisted` or `2 = EnclaveSealed`).
    pub protection_profile: u8,
    /// Authentication kind (`0` or `1`).
    pub authentication_kind: u8,
    /// Buyer maximum tier (`0..=3`).
    pub buyer_max_tier: u8,
    /// Alignment pad (`0`).
    pub reserved_pad: u8,
    /// Reference training sample rate in Hz (`48_000`).
    pub sample_rate_train_hz: U32,
    /// Zero-filled reserved bytes (`16..64`).
    pub reserved: [u8; 48],
}

const _: () = assert!(size_of::<DnplPreambleWire>() == 64);
const _: () = assert!(align_of::<DnplPreambleWire>() == 64);
const _: () = assert!(offset_of!(DnplPreambleWire, protection_profile) == 8);
const _: () = assert!(offset_of!(DnplPreambleWire, buyer_max_tier) == 10);
const _: () = assert!(offset_of!(DnplPreambleWire, sample_rate_train_hz) == 12);
const _: () = assert!(offset_of!(DnplPreambleWire, reserved) == 16);

impl DnplPreambleWire {
    /// Constructs a canonical `v3.0` `DnplPreambleWire` (`SR = 48_000 Hz`).
    pub fn new(
        protection_profile: u8,
        authentication_kind: u8,
        buyer_max_tier: u8,
    ) -> Result<Self, DnamError> {
        reject_if(
            !matches!(protection_profile, 1 | 2) || authentication_kind > 1 || buyer_max_tier > 3,
            "invalid DNPL preamble arguments",
        )?;
        Ok(Self {
            magic: DNPL_WIRE_MAGIC,
            version_major: U16::new(DRM_WIRE_VERSION_MAJOR),
            version_minor: U16::new(DRM_WIRE_VERSION_MINOR),
            protection_profile,
            authentication_kind,
            buyer_max_tier,
            reserved_pad: 0,
            sample_rate_train_hz: U32::new(48_000),
            reserved: [0u8; 48],
        })
    }

    /// Parses and validates a 64-byte `DnplPreambleWire` via safe `zerocopy`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        let (wire, _) = Self::read_from_prefix(bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: size_of::<Self>(),
            actual: bytes.len(),
        })?;
        if wire.magic != DNPL_WIRE_MAGIC {
            return Err(DnamError::InvalidMagic(wire.magic));
        }
        let major = wire.version_major.get();
        let minor = wire.version_minor.get();
        if major != DRM_WIRE_VERSION_MAJOR || minor != DRM_WIRE_VERSION_MINOR {
            return Err(DnamError::UnsupportedVersion { major, minor });
        }
        reject_if(
            !matches!(wire.protection_profile, 1 | 2)
                || wire.authentication_kind > 1
                || wire.buyer_max_tier > 3
                || wire.reserved_pad != 0
                || wire.sample_rate_train_hz.get() != 48_000
                || wire.reserved.iter().any(|&b| b != 0),
            "invalid DNPL preamble fields",
        )?;
        Ok(wire)
    }

    /// Serializes this `DNPL` preamble into a 64-byte little-endian wire block.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        let _ = self.write_to_prefix(&mut out);
        out
    }
}

/// Fixed 20-byte `DME1` encrypted mutable frame header (`#[repr(C)]`, `ref[sec.drm.dme1-mutable-suffix]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Dme1HeaderWire {
    /// Magic bytes (`b"DME1"`).
    pub magic: [u8; 4],
    /// Protection profile (`1` or `2`).
    pub protection_profile: u8,
    /// Zero-filled reserved bytes (`5..8`).
    pub reserved: [u8; 3],
    /// Recipient HPKE slot count `N` (`1..=16`).
    pub recipient_count: U32,
    /// `ChaCha20-Poly1305` ciphertext + tag byte length `C`.
    pub ciphertext_length: U64,
}

const _: () = assert!(size_of::<Dme1HeaderWire>() == 20);
const _: () = assert!(align_of::<Dme1HeaderWire>() == 1);
const _: () = assert!(offset_of!(Dme1HeaderWire, protection_profile) == 4);
const _: () = assert!(offset_of!(Dme1HeaderWire, reserved) == 5);
const _: () = assert!(offset_of!(Dme1HeaderWire, recipient_count) == 8);
const _: () = assert!(offset_of!(Dme1HeaderWire, ciphertext_length) == 12);

impl Dme1HeaderWire {
    /// Constructs and validates a `Dme1HeaderWire`.
    pub fn new(
        protection_profile: u8,
        recipient_count: u32,
        ciphertext_length: u64,
    ) -> Result<Self, DnamError> {
        let min_ct = 4u64 + 32 * (recipient_count as u64) + 18 + 16;
        reject_if(
            !matches!(protection_profile, 1 | 2)
                || !(1..=16).contains(&recipient_count)
                || !(min_ct..=4_096).contains(&ciphertext_length),
            "invalid encrypted mutable header",
        )?;
        Ok(Self {
            magic: DME1_WIRE_MAGIC,
            protection_profile,
            reserved: [0u8; 3],
            recipient_count: U32::new(recipient_count),
            ciphertext_length: U64::new(ciphertext_length),
        })
    }

    /// Parses and validates a 20-byte `Dme1HeaderWire` via safe `zerocopy`.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, DnamError> {
        let (wire, _) = Self::ref_from_prefix(bytes).map_err(|_| DnamError::BufferTooSmall {
            expected: size_of::<Self>(),
            actual: bytes.len(),
        })?;
        if wire.magic != DME1_WIRE_MAGIC {
            return Err(DnamError::InvalidMagic(wire.magic));
        }
        let n = wire.recipient_count.get();
        let c = wire.ciphertext_length.get();
        let min_ct = 4u64 + 32 * (n as u64) + 18 + 16;
        reject_if(
            !matches!(wire.protection_profile, 1 | 2)
                || wire.reserved != [0u8; 3]
                || !(1..=16).contains(&n)
                || !(min_ct..=4_096).contains(&c),
            "invalid encrypted mutable header",
        )?;
        Ok(wire)
    }

    /// Serializes this `DME1` header into 20 little-endian wire bytes.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 20] {
        let mut out = [0u8; 20];
        let _ = self.write_to_prefix(&mut out);
        out
    }
}

/// Packed 80-byte HPKE recipient slot wire record (`enc[32] || wrapped_cek_and_tag[48]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct HpkeSlotWire {
    pub enc: [u8; 32],
    pub wrapped_cek_and_tag: [u8; 48],
}

const _: () = assert!(size_of::<HpkeSlotWire>() == 80);
const _: () = assert!(align_of::<HpkeSlotWire>() == 1);

/// Packed 16-byte record descriptor wire entry (`offset: U64`, `ciphertext_length: U64`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct RecordDirEntryWire {
    pub offset: U64,
    pub ciphertext_length: U64,
}

const _: () = assert!(size_of::<RecordDirEntryWire>() == 16);
const _: () = assert!(align_of::<RecordDirEntryWire>() == 1);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_dnsl_dnpl_and_dme1_wire_validation_and_error_branches() {
        let dnsl = DnslPreambleWire::new(3).unwrap();
        let dnsl_bytes = dnsl.to_bytes();
        assert_eq!(DnslPreambleWire::from_bytes(&dnsl_bytes).unwrap(), dnsl);
        assert!(DnslPreambleWire::new(4).is_err());
        assert!(DnslPreambleWire::from_bytes(&[0u8; 32]).is_err());

        let mut bad_magic = dnsl_bytes;
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(DnslPreambleWire::from_bytes(&bad_magic).is_err());

        let mut bad_ver = dnsl_bytes;
        bad_ver[4..6].copy_from_slice(U16::new(2).as_bytes());
        assert!(DnslPreambleWire::from_bytes(&bad_ver).is_err());

        let dnpl = DnplPreambleWire::new(1, 1, 2).unwrap();
        let dnpl_bytes = dnpl.to_bytes();
        assert_eq!(DnplPreambleWire::from_bytes(&dnpl_bytes).unwrap(), dnpl);
        assert!(DnplPreambleWire::new(0, 0, 0).is_err());
        assert!(DnplPreambleWire::from_bytes(&[0u8; 16]).is_err());

        let dme1 = Dme1HeaderWire::new(2, 2, 128).unwrap();
        let dme1_bytes = dme1.to_bytes();
        assert_eq!(Dme1HeaderWire::from_bytes(&dme1_bytes).unwrap(), &dme1);
        assert!(Dme1HeaderWire::new(0, 2, 128).is_err());
        assert!(Dme1HeaderWire::new(2, 0, 128).is_err());
        assert!(Dme1HeaderWire::from_bytes(&[0u8; 8]).is_err());

        let dnsl_dir = DnslDirectoryWire {
            plan_offset: U64::new(192),
            plan_length: U64::new(256),
            slots_offset: U64::new(448),
            slots_length: U64::new(192),
            manifest_offset: U64::new(640),
            manifest_length: U64::new(128),
            shared_record_offset: U64::new(768),
            shared_ciphertext_length: U64::new(64),
            original_tier_ends: [
                U64::new(4_608),
                U64::new(7_936),
                U64::new(15_872),
                U64::new(30_720),
            ],
            reserved: [0u8; 32],
        };
        assert_eq!(
            DnslDirectoryWire::from_bytes(dnsl_dir.as_bytes()).unwrap(),
            &dnsl_dir
        );

        let inner_dir = InnerDirectoryWire {
            plan_offset: U64::new(320),
            plan_length: U64::new(256),
            slots_offset: U64::new(576),
            slots_length: U64::new(192),
            authentication_offset: U64::new(768),
            authentication_length: U64::new(320),
            shared_record_offset: U64::new(1_088),
            shared_ciphertext_length: U64::new(64),
            original_tier_ends: [
                U64::new(4_608),
                U64::new(7_936),
                U64::new(15_872),
                U64::new(30_720),
            ],
            logical_metadata_sha256: [0xABu8; 32],
        };
        assert_eq!(
            InnerDirectoryWire::from_bytes(inner_dir.as_bytes()).unwrap(),
            &inner_dir
        );
    }

    proptest! {
        #[test]
        fn prop_zerocopy_drm_wire_roundtrips(
            max_tier in 0u8..=3u8,
            profile in 1u8..=2u8,
            auth_kind in 0u8..=1u8,
            recips in 1u32..=16u32,
            ct_extra in 0u64..=1024u64,
            logical_sha in prop::array::uniform32(any::<u8>()),
        ) {
            let dnsl = DnslPreambleWire::new(max_tier).unwrap();
            prop_assert_eq!(DnslPreambleWire::from_bytes(&dnsl.to_bytes()).unwrap(), dnsl);

            let dnpl = DnplPreambleWire::new(profile, auth_kind, max_tier).unwrap();
            prop_assert_eq!(DnplPreambleWire::from_bytes(&dnpl.to_bytes()).unwrap(), dnpl);

            let min_ct = 4u64 + 32 * (recips as u64) + 18 + 16;
            let ct_len = (min_ct + ct_extra).min(4_096);
            let dme1 = Dme1HeaderWire::new(profile, recips, ct_len).unwrap();
            let dme1_bytes = dme1.to_bytes();
            prop_assert_eq!(Dme1HeaderWire::from_bytes(&dme1_bytes).unwrap(), &dme1);

            let inner = InnerDirectoryWire {
                plan_offset: U64::new(256),
                plan_length: U64::new(128),
                slots_offset: U64::new(384),
                slots_length: U64::new(64),
                authentication_offset: U64::new(448),
                authentication_length: U64::new(128),
                shared_record_offset: U64::new(576),
                shared_ciphertext_length: U64::new(64),
                original_tier_ends: [U64::new(4_608), U64::new(7_936), U64::new(15_872), U64::new(30_720)],
                logical_metadata_sha256: logical_sha,
            };
            prop_assert_eq!(InnerDirectoryWire::from_bytes(inner.as_bytes()).unwrap(), &inner);
        }
    }
}
