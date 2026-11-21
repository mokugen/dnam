//! Allocation-free UTF-8 JSON scanner extracting [`ReloadHeaderFlags`] and enforcing
//! sticky `"architecture" == "sona"`, `"container_kind" != "preview"` containment
//! (`r[sec.preview.containment]`), and legacy key rejection.

mod controls_scan;

use crate::container::ContainerWireError;
use crate::engine::ReloadHeaderFlags;
use crate::schema::BlendMode;

/// Allocation-free (`Copy`) error enum used across [`crate::DnamEngine::reload_model_in_place`]
/// so real-time reload paths never emit `DnamError` `String` drop glue (`0` frees).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RtReloadError {
    Wire(ContainerWireError),
    InvalidJson,
    UnsupportedArchitecture,
    PreviewContainerRejected,
    RejectedLegacyField,
    ControlCapacityExceeded { required: usize, capacity: usize },
}

#[inline]
pub(super) fn hex_nibble(b: u8) -> Option<u8> {
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

pub(super) struct JsonSliceCursor<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl<'b> JsonSliceCursor<'b> {
    #[inline]
    fn new(bytes: &'b [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline]
    pub(super) fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    pub(super) fn peek_skip_ws(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    pub(super) fn expect_byte(&mut self, expected: u8) -> Result<(), RtReloadError> {
        self.skip_ws();
        if self.bytes.get(self.pos).copied() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(RtReloadError::InvalidJson)
        }
    }

    pub(super) fn skip_null(&mut self) -> Result<(), RtReloadError> {
        self.skip_ws();
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(())
        } else {
            Err(RtReloadError::InvalidJson)
        }
    }

    pub(super) fn parse_string_inner(&mut self) -> Result<&'b [u8], RtReloadError> {
        self.expect_byte(b'"')?;
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b == b'"' {
                let end = self.pos;
                self.pos += 1;
                return Ok(&self.bytes[start..end]);
            }
            if b == b'\\' {
                self.pos += 1;
                let Some(&esc) = self.bytes.get(self.pos) else {
                    return Err(RtReloadError::InvalidJson);
                };
                self.pos += 1;
                match esc {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
                    b'u' => {
                        for _ in 0..4 {
                            let Some(&hx) = self.bytes.get(self.pos) else {
                                return Err(RtReloadError::InvalidJson);
                            };
                            if !hx.is_ascii_hexdigit() {
                                return Err(RtReloadError::InvalidJson);
                            }
                            self.pos += 1;
                        }
                    }
                    _ => return Err(RtReloadError::InvalidJson),
                }
                continue;
            }
            if b < 0x20 {
                return Err(RtReloadError::InvalidJson);
            }
            self.pos += 1;
        }
        Err(RtReloadError::InvalidJson)
    }

    fn parse_bool(&mut self) -> Result<bool, RtReloadError> {
        self.skip_ws();
        let rem = &self.bytes[self.pos..];
        if rem.starts_with(b"true") {
            self.pos += 4;
            Ok(true)
        } else if rem.starts_with(b"false") {
            self.pos += 5;
            Ok(false)
        } else {
            Err(RtReloadError::InvalidJson)
        }
    }

    fn parse_number_str(&mut self) -> Result<&'b str, RtReloadError> {
        self.skip_ws();
        let start = self.pos;
        if self.bytes.get(self.pos).copied() == Some(b'-') {
            self.pos += 1;
        }
        let d_start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == d_start {
            return Err(RtReloadError::InvalidJson);
        }
        if self.bytes.get(self.pos).copied() == Some(b'.') {
            self.pos += 1;
            let f_start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == f_start {
                return Err(RtReloadError::InvalidJson);
            }
        }
        if matches!(self.bytes.get(self.pos).copied(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.bytes.get(self.pos).copied(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let e_start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == e_start {
                return Err(RtReloadError::InvalidJson);
            }
        }
        core::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| RtReloadError::InvalidJson)
    }

    pub(super) fn parse_f32(&mut self) -> Result<f32, RtReloadError> {
        let s = self.parse_number_str()?;
        let v = s.parse::<f32>().map_err(|_| RtReloadError::InvalidJson)?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(RtReloadError::InvalidJson)
        }
    }

    fn parse_u16(&mut self) -> Result<u16, RtReloadError> {
        let s = self.parse_number_str()?;
        s.parse::<u16>().map_err(|_| RtReloadError::InvalidJson)
    }

    pub(super) fn skip_value(&mut self, max_depth: usize) -> Result<(), RtReloadError> {
        if max_depth == 0 {
            return Err(RtReloadError::InvalidJson);
        }
        match self.peek_skip_ws() {
            Some(b'"') => {
                let _ = self.parse_string_inner()?;
                Ok(())
            }
            Some(b'{') => self.scan_object(|c, _| c.skip_value(max_depth - 1)),
            Some(b'[') => self.scan_array(|c, _| c.skip_value(max_depth - 1)),
            Some(b't' | b'f') => {
                let _ = self.parse_bool()?;
                Ok(())
            }
            Some(b'n') => self.skip_null(),
            Some(b'-' | b'0'..=b'9') => {
                let _ = self.parse_number_str()?;
                Ok(())
            }
            _ => Err(RtReloadError::InvalidJson),
        }
    }

    pub(super) fn scan_array<F>(&mut self, mut on_elem: F) -> Result<(), RtReloadError>
    where
        F: FnMut(&mut Self, usize) -> Result<(), RtReloadError>,
    {
        self.expect_byte(b'[')?;
        if self.peek_skip_ws() == Some(b']') {
            self.pos += 1;
            return Ok(());
        }
        let mut idx = 0_usize;
        loop {
            on_elem(self, idx)?;
            idx += 1;
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

    pub(super) fn scan_object<F>(&mut self, mut on_field: F) -> Result<(), RtReloadError>
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

#[inline]
pub(super) fn is_rejected_legacy_key(key: &[u8]) -> bool {
    json_inner_eq_ascii(key, b"io_router")
        || json_inner_eq_ascii(key, b"sub_crossover_hz")
        || json_inner_eq_ascii(key, b"supports_freeze")
        || json_inner_eq_ascii(key, b"freeze_mutes_dry")
        || json_inner_eq_ascii(key, b"freeze_latch")
}

fn scan_capabilities_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"blend_mode") {
            let val = c.parse_string_inner()?;
            flags.blend_mode = if json_inner_eq_ascii(val, b"host_parallel") {
                BlendMode::HostParallel
            } else if json_inner_eq_ascii(val, b"modeled_control") {
                BlendMode::ModeledControl
            } else if json_inner_eq_ascii(val, b"none") {
                BlendMode::None
            } else {
                return Err(RtReloadError::InvalidJson);
            };
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
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"auto_unity_enabled") {
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
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"inputs") {
            let num = c.parse_number_str()?;
            let n = num.parse::<u64>().map_err(|_| RtReloadError::InvalidJson)?;
            flags.single_input = n == 1;
        } else if json_inner_eq_ascii(key, b"output_mode") {
            let val = c.parse_string_inner()?;
            flags.stereo_out = !json_inner_eq_ascii(val, b"mono");
        } else if json_inner_eq_ascii(key, b"capabilities") {
            scan_capabilities_obj(c, flags)?;
        } else if json_inner_eq_ascii(key, b"leveling") {
            scan_leveling_obj(c, flags)?;
        } else {
            c.skip_value(61)?;
        }
        Ok(())
    })
}

fn scan_transport_obj(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"dc_blocker_pole") {
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
/// `"architecture" == "sona"`, `"container_kind" != "preview"` (`r[sec.preview.containment]`),
/// and rejection of evicted legacy fields.
pub(super) fn scan_header_flags_rt(
    header_slice: &[u8],
) -> Result<ReloadHeaderFlags, RtReloadError> {
    let utf8_str = core::str::from_utf8(header_slice).map_err(|_| RtReloadError::InvalidJson)?;
    let mut cursor = JsonSliceCursor::new(utf8_str.as_bytes());
    let mut flags = ReloadHeaderFlags::default();
    let mut bad_arch = false;
    let mut is_preview = false;

    cursor.scan_object(|c, key| {
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"architecture") {
            let val = c.parse_string_inner()?;
            bad_arch |= !json_inner_eq_ascii(val, b"sona");
        } else if json_inner_eq_ascii(key, b"container_kind") {
            let val = c.parse_string_inner()?;
            is_preview |= json_inner_eq_ascii(val, b"preview");
        } else if json_inner_eq_ascii(key, b"io_config") {
            scan_io_config_obj(c, &mut flags)?;
        } else if json_inner_eq_ascii(key, b"transport") {
            scan_transport_obj(c, &mut flags)?;
        } else if json_inner_eq_ascii(key, b"controls_schema") {
            controls_scan::scan_controls_schema_array(c, &mut flags)?;
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
    Ok(flags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_json_inner_eq_ascii_and_scanner_edge_cases() {
        assert!(json_inner_eq_ascii(
            br#"\"\\\/\b\f\n\r\t"#,
            b"\"\\/\x08\x0c\n\r\t"
        ));
        assert!(json_inner_eq_ascii(br#"\u0073\u006f\u006E\u0061"#, b"sona"));
        assert!(!json_inner_eq_ascii(b"a\\", b"a"));
        assert!(!json_inner_eq_ascii(br#"\u12"#, b"x"));
        assert!(!json_inner_eq_ascii(br#"\u12zz"#, b"x"));
        assert!(!json_inner_eq_ascii(br#"\u0080"#, b"x"));
        assert!(!json_inner_eq_ascii(br#"\x"#, b"x"));
        assert!(!json_inner_eq_ascii(br#"\u0073x"#, b"sa"));

        let ok = concat!(
            r#"{"architecture":"sona","extra":["\"\\\/\b\f\n\r\t","#,
            r#"null,true,false,[],{},1.5e+2,-2E-1],"#,
            r#""io_config":{"inputs":1,"output_mode":"mono","#,
            r#""capabilities":{"default_blend":0.75,"wet_polarity_inverted":true},"#,
            r#""leveling":{"auto_unity_enabled":false,"calibrated_trim_db":1.5}},"#,
            r#""transport":{"dc_blocker_pole":9.9E-1,"switch_xfade_samples":64}}"#
        );
        let flags = scan_header_flags_rt(ok.as_bytes()).unwrap();
        assert!(flags.single_input);
        assert!(!flags.stereo_out);
        assert!((flags.default_blend - 0.75).abs() < 1e-6);
        assert!(flags.wet_polarity_inverted);
        assert!(!flags.auto_unity_enabled);
        assert!((flags.calibrated_trim_db - 1.5).abs() < 1e-6);
        assert!((flags.dc_blocker_pole - 0.99).abs() < 1e-4);
        assert_eq!(flags.switch_xfade_samples, 64);

        for legacy in [
            r#"{"architecture":"sona","io_router":{}}"#,
            r#"{"architecture":"sona","sub_crossover_hz":200.0}"#,
            r#"{"architecture":"sona","io_config":{"capabilities":{"supports_freeze":true}}}"#,
            r#"{"architecture":"sona","io_config":{"capabilities":{"freeze_mutes_dry":false}}}"#,
            r#"{"architecture":"sona","freeze_latch":true}"#,
        ] {
            assert_eq!(
                scan_header_flags_rt(legacy.as_bytes()),
                Err(RtReloadError::RejectedLegacyField)
            );
        }

        let deep = format!(
            "{{\"architecture\":\"sona\",\"extra\":{}{}}}",
            "[".repeat(64),
            "]".repeat(64)
        );
        assert_eq!(
            scan_header_flags_rt(deep.as_bytes()),
            Err(RtReloadError::InvalidJson)
        );

        for bad in [
            b"[]".as_slice(),
            b"{\"a\":\"\x01\"}".as_slice(),
            br#"{"a":"\u123Z"}"#.as_slice(),
            br#"{"a":"\q"}"#.as_slice(),
            br#"{"a":"unterminated"#.as_slice(),
            br#"{"io_config":{"capabilities":{"default_blend":true}}}"#.as_slice(),
            br#"{"io_config":{"inputs":true}}"#.as_slice(),
            br#"{"transport":{"dc_blocker_pole":1.}}"#.as_slice(),
            br#"{"transport":{"dc_blocker_pole":1e}}"#.as_slice(),
            br#"{"transport":{"dc_blocker_pole":1e40}}"#.as_slice(),
            br#"{"extra":[1 2]}"#.as_slice(),
            br#"{"extra":nulx}"#.as_slice(),
            br#"{"architecture":"sona"} 123"#.as_slice(),
        ] {
            assert_eq!(scan_header_flags_rt(bad), Err(RtReloadError::InvalidJson));
        }
    }
}
