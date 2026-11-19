//! Allocation-free UTF-8 JSON scanner extracting [`ReloadHeaderFlags`] and enforcing
//! sticky `"architecture" == "sona"` plus `"container_kind" != "preview"` containment
//! (`r[sec.preview.containment]`).

use crate::container::ContainerWireError;
use crate::engine::ReloadHeaderFlags;

/// Allocation-free (`Copy`) error enum used across [`super::DnamEngine::reload_model_in_place`]
/// so real-time reload paths never emit `DnamError` `String` drop glue (`0` frees).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RtReloadError {
    Wire(ContainerWireError),
    InvalidJson,
    UnsupportedArchitecture,
    PreviewContainerRejected,
    ControlCapacityExceeded { required: usize, capacity: usize },
}

#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

pub(super) fn json_inner_eq_ascii(inner: &[u8], expected: &[u8]) -> bool {
    if !inner.contains(&b'\\') {
        return inner == expected;
    }
    let mut i = 0;
    let mut j = 0;
    while i < inner.len() {
        let ch: u8 = if inner[i] == b'\\' {
            i += 1;
            let Some(&esc) = inner.get(i) else {
                return false;
            };
            i += 1;
            match esc {
                b'"' | b'\\' | b'/' => esc,
                b'b' => 0x08,
                b'f' => 0x0C,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'u' => {
                    if i + 4 > inner.len() {
                        return false;
                    }
                    let mut cp = 0u16;
                    for &hx in &inner[i..i + 4] {
                        let Some(nib) = hex_nibble(hx) else {
                            return false;
                        };
                        cp = (cp << 4) | (nib as u16);
                    }
                    i += 4;
                    if cp > 0x7F {
                        return false;
                    }
                    cp as u8
                }
                _ => return false,
            }
        } else {
            let b = inner[i];
            i += 1;
            b
        };
        if expected.get(j).copied() != Some(ch) {
            return false;
        }
        j += 1;
    }
    j == expected.len()
}

struct JsonSliceCursor<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl<'b> JsonSliceCursor<'b> {
    #[inline]
    fn new(bytes: &'b [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline]
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn peek_skip_ws(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn expect_byte(&mut self, expected: u8) -> Result<(), RtReloadError> {
        if self.peek_skip_ws() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(RtReloadError::InvalidJson)
        }
    }

    fn parse_string_inner(&mut self) -> Result<&'b [u8], RtReloadError> {
        self.expect_byte(b'"')?;
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b'"' {
                let end = self.pos;
                self.pos += 1;
                return Ok(&self.bytes[start..end]);
            }
            if b < 0x20 {
                return Err(RtReloadError::InvalidJson);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = *self.bytes.get(self.pos).ok_or(RtReloadError::InvalidJson)?;
                if esc == b'u' {
                    self.pos += 1;
                    for _ in 0..4 {
                        let hx = *self.bytes.get(self.pos).ok_or(RtReloadError::InvalidJson)?;
                        if !hx.is_ascii_hexdigit() {
                            return Err(RtReloadError::InvalidJson);
                        }
                        self.pos += 1;
                    }
                } else if matches!(esc, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                    self.pos += 1;
                } else {
                    return Err(RtReloadError::InvalidJson);
                }
            } else {
                self.pos += 1;
            }
        }
        Err(RtReloadError::InvalidJson)
    }

    fn parse_bool(&mut self) -> Result<bool, RtReloadError> {
        self.skip_ws();
        let rest = &self.bytes[self.pos..];
        if rest.starts_with(b"true") {
            self.pos += 4;
            Ok(true)
        } else if rest.starts_with(b"false") {
            self.pos += 5;
            Ok(false)
        } else {
            Err(RtReloadError::InvalidJson)
        }
    }

    fn parse_number_str(&mut self) -> Result<&'b str, RtReloadError> {
        self.skip_ws();
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.bytes.get(self.pos).copied() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
            }
            _ => return Err(RtReloadError::InvalidJson),
        }
        if self.bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            if !self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                return Err(RtReloadError::InvalidJson);
            }
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                return Err(RtReloadError::InvalidJson);
            }
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        core::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| RtReloadError::InvalidJson)
    }

    #[inline]
    fn parse_f32(&mut self) -> Result<f32, RtReloadError> {
        let num = self.parse_number_str()?;
        let val = num.parse::<f32>().map_err(|_| RtReloadError::InvalidJson)?;
        if !val.is_finite() {
            return Err(RtReloadError::InvalidJson);
        }
        Ok(val)
    }

    #[inline]
    fn parse_u16(&mut self) -> Result<u16, RtReloadError> {
        let num = self.parse_number_str()?;
        num.parse::<u16>().map_err(|_| RtReloadError::InvalidJson)
    }

    fn skip_value(&mut self, max_depth: usize) -> Result<(), RtReloadError> {
        if max_depth == 0 {
            return Err(RtReloadError::InvalidJson);
        }
        match self.peek_skip_ws() {
            Some(b'"') => {
                let _ = self.parse_string_inner()?;
                Ok(())
            }
            Some(b'{') => self.scan_object(|cursor, _| cursor.skip_value(max_depth - 1)),
            Some(b'[') => {
                self.pos += 1;
                if self.peek_skip_ws() == Some(b']') {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value(max_depth - 1)?;
                    match self.peek_skip_ws() {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(());
                        }
                        _ => return Err(RtReloadError::InvalidJson),
                    }
                }
            }
            Some(b't' | b'f') => {
                let _ = self.parse_bool()?;
                Ok(())
            }
            Some(b'n') => {
                if self.bytes[self.pos..].starts_with(b"null") {
                    self.pos += 4;
                    Ok(())
                } else {
                    Err(RtReloadError::InvalidJson)
                }
            }
            Some(b'-' | b'0'..=b'9') => {
                let _ = self.parse_number_str()?;
                Ok(())
            }
            _ => Err(RtReloadError::InvalidJson),
        }
    }

    fn scan_object<F>(&mut self, mut on_field: F) -> Result<(), RtReloadError>
    where
        F: FnMut(&mut Self, &'b [u8]) -> Result<(), RtReloadError>,
    {
        self.expect_byte(b'{')?;
        if self.peek_skip_ws() == Some(b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            let key = self.parse_string_inner()?;
            self.expect_byte(b':')?;
            on_field(self, key)?;
            match self.peek_skip_ws() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return Err(RtReloadError::InvalidJson),
            }
        }
    }
}

fn scan_capabilities_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
    saw_supports_freeze: &mut Option<bool>,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if json_inner_eq_ascii(key, b"freeze_mutes_dry") {
            flags.freeze_mutes_dry = c.parse_bool()?;
        } else if json_inner_eq_ascii(key, b"supports_freeze") {
            *saw_supports_freeze = Some(c.parse_bool()?);
        } else if json_inner_eq_ascii(key, b"default_blend") {
            flags.default_blend = c.parse_f32()?.clamp(0.0, 1.0);
        } else if json_inner_eq_ascii(key, b"wet_polarity_inverted") {
            flags.wet_polarity_inverted = c.parse_bool()?;
        } else {
            c.skip_value(60)?;
        }
        Ok(())
    })
}

fn scan_leveling_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if json_inner_eq_ascii(key, b"auto_unity_enabled") {
            flags.auto_unity_enabled = c.parse_bool()?;
        } else if json_inner_eq_ascii(key, b"calibrated_trim_db") {
            flags.calibrated_trim_db = c.parse_f32()?;
        } else {
            c.skip_value(60)?;
        }
        Ok(())
    })
}

fn scan_io_config_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
    saw_supports_freeze: &mut Option<bool>,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if json_inner_eq_ascii(key, b"inputs") {
            let num = c.parse_number_str()?;
            let n = num.parse::<u64>().map_err(|_| RtReloadError::InvalidJson)?;
            flags.single_input = n == 1;
        } else if json_inner_eq_ascii(key, b"output_mode") {
            let val = c.parse_string_inner()?;
            flags.stereo_out = !json_inner_eq_ascii(val, b"mono");
        } else if json_inner_eq_ascii(key, b"capabilities") {
            scan_capabilities_obj(c, flags, saw_supports_freeze)?;
        } else if json_inner_eq_ascii(key, b"leveling") {
            scan_leveling_obj(c, flags)?;
        } else {
            c.skip_value(61)?;
        }
        Ok(())
    })
}

fn scan_stage0_router_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if json_inner_eq_ascii(key, b"sub_crossover_hz") {
            flags.sub_crossover_hz = c.parse_f32()?;
        } else {
            c.skip_value(61)?;
        }
        Ok(())
    })
}

fn scan_stage4_tank_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if json_inner_eq_ascii(key, b"dc_blocker_pole") {
            flags.dc_blocker_pole = c.parse_f32()?;
        } else if json_inner_eq_ascii(key, b"tzf_reference_tap_samples") {
            flags.tzf_reference_tap_samples = c.parse_f32()?;
        } else if json_inner_eq_ascii(key, b"switch_xfade_samples") {
            flags.switch_xfade_samples = c.parse_u16()?;
        } else {
            c.skip_value(61)?;
        }
        Ok(())
    })
}

/// Zero-allocation UTF-8 JSON scanner extracting [`ReloadHeaderFlags`] and enforcing sticky
/// `"architecture" == "sona"` plus `"container_kind" != "preview"` (`r[sec.preview.containment]`).
pub(super) fn scan_header_flags_rt(
    header_slice: &[u8],
) -> Result<ReloadHeaderFlags, RtReloadError> {
    let utf8_str = core::str::from_utf8(header_slice).map_err(|_| RtReloadError::InvalidJson)?;
    let mut cursor = JsonSliceCursor::new(utf8_str.as_bytes());
    let mut flags = ReloadHeaderFlags::default();
    let mut saw_supports_freeze: Option<bool> = None;
    let mut bad_arch = false;
    let mut is_preview = false;

    cursor.scan_object(|c, key| {
        if json_inner_eq_ascii(key, b"architecture") {
            let val = c.parse_string_inner()?;
            bad_arch |= !json_inner_eq_ascii(val, b"sona");
        } else if json_inner_eq_ascii(key, b"container_kind") {
            let val = c.parse_string_inner()?;
            is_preview |= json_inner_eq_ascii(val, b"preview");
        } else if json_inner_eq_ascii(key, b"io_config") {
            scan_io_config_obj(c, &mut flags, &mut saw_supports_freeze)?;
        } else if json_inner_eq_ascii(key, b"stage0_router") {
            scan_stage0_router_obj(c, &mut flags)?;
        } else if json_inner_eq_ascii(key, b"stage4_tank") {
            scan_stage4_tank_obj(c, &mut flags)?;
        } else {
            c.skip_value(62)?;
        }
        Ok(())
    })?;

    if cursor.peek_skip_ws().is_some() {
        return Err(RtReloadError::InvalidJson);
    }
    if bad_arch {
        return Err(RtReloadError::UnsupportedArchitecture);
    }
    if is_preview {
        return Err(RtReloadError::PreviewContainerRejected);
    }
    flags.supports_freeze = saw_supports_freeze.unwrap_or(true);
    Ok(flags)
}
