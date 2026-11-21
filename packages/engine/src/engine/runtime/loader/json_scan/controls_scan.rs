//! Allocation-free `controls_schema` JSON scanner for [`ReloadHeaderFlags`].

use crate::engine::{ReloadCalibratedControl, ReloadHeaderFlags};
use crate::schema::ControlCalibrationRole;

use super::{
    JsonSliceCursor, RtReloadError, hex_nibble, is_rejected_legacy_key, json_inner_eq_ascii,
};

fn decode_json_ascii_trimmed(inner: &[u8], out: &mut [u8; 64]) -> Option<usize> {
    let mut len = 0_usize;
    let mut i = 0_usize;
    while i < inner.len() {
        let ch: u8 = if inner[i] == b'\\' {
            i += 1;
            let &esc = inner.get(i)?;
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
                        return None;
                    }
                    let mut cp = 0_u16;
                    for &hx in &inner[i..i + 4] {
                        cp = (cp << 4) | (hex_nibble(hx)? as u16);
                    }
                    i += 4;
                    if cp > 0x7F {
                        return None;
                    }
                    cp as u8
                }
                _ => return None,
            }
        } else {
            let b = inner[i];
            i += 1;
            b
        };
        if len == 0 && ch.is_ascii_whitespace() {
            continue;
        }
        if len >= out.len() {
            return None;
        }
        out[len] = ch;
        len += 1;
    }
    while len > 0 && out[len - 1].is_ascii_whitespace() {
        len -= 1;
    }
    Some(len)
}

#[inline]
fn stereo_width_for_json_option(inner: &[u8]) -> Option<f32> {
    let mut buf = [0_u8; 64];
    let len = decode_json_ascii_trimmed(inner, &mut buf)?;
    let trimmed = &buf[..len];
    if trimmed.eq_ignore_ascii_case(b"mono") || trimmed == b"0" {
        Some(0.0)
    } else if trimmed.eq_ignore_ascii_case(b"stereo") || trimmed == b"1" {
        Some(1.0)
    } else {
        None
    }
}

fn parse_f32_array<const N: usize>(
    cursor: &mut JsonSliceCursor<'_>,
) -> Result<[f32; N], RtReloadError> {
    let mut out = [0.0_f32; N];
    let mut count = 0_usize;
    cursor.scan_array(|c, idx| {
        if idx >= N {
            return Err(RtReloadError::InvalidJson);
        }
        out[idx] = c.parse_f32()?;
        count += 1;
        Ok(())
    })?;
    if count != N {
        return Err(RtReloadError::InvalidJson);
    }
    Ok(out)
}

fn scan_declared_options_array(
    cursor: &mut JsonSliceCursor<'_>,
    widths: &mut [f32; 16],
    mask: &mut u16,
) -> Result<(), RtReloadError> {
    cursor.scan_array(|c, idx| {
        let raw = c.parse_string_inner()?;
        if idx < 16
            && let Some(w) = stereo_width_for_json_option(raw)
        {
            widths[idx] = w;
            *mask |= 1_u16 << idx;
        }
        Ok(())
    })
}

fn scan_calibration_obj(
    cursor: &mut JsonSliceCursor<'_>,
    role: &mut Option<ControlCalibrationRole>,
    has_anchors: &mut bool,
    anchors: &mut [f32; 3],
) -> Result<(), RtReloadError> {
    cursor.scan_object(|c, key| {
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"role") {
            let val = c.parse_string_inner()?;
            *role = if json_inner_eq_ascii(val, b"stereo_width") {
                Some(ControlCalibrationRole::StereoWidth)
            } else if json_inner_eq_ascii(val, b"stereo_mode") {
                Some(ControlCalibrationRole::StereoMode)
            } else if json_inner_eq_ascii(val, b"mix") {
                Some(ControlCalibrationRole::Mix)
            } else {
                None
            };
        } else if json_inner_eq_ascii(key, b"lobatto_physical_anchors") {
            if c.peek_skip_ws() == Some(b'n') {
                c.skip_null()?;
            } else {
                *anchors = parse_f32_array::<3>(c)?;
                *has_anchors = true;
            }
        } else {
            c.skip_value(58)?;
        }
        Ok(())
    })
}

fn scan_single_control_obj(
    cursor: &mut JsonSliceCursor<'_>,
    ctrl_idx: usize,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    let mut role: Option<ControlCalibrationRole> = None;
    let mut decl_range: Option<[f32; 2]> = None;
    let mut has_anchors = false;
    let mut anchors = [0.0_f32; 3];
    let mut stereo_mode_opt_widths = [0.0_f32; 16];
    let mut stereo_mode_opt_mask = 0_u16;

    cursor.scan_object(|c, key| {
        if is_rejected_legacy_key(key) {
            return Err(RtReloadError::RejectedLegacyField);
        } else if json_inner_eq_ascii(key, b"type") {
            let val = c.parse_string_inner()?;
            if !json_inner_eq_ascii(val, b"range") && !json_inner_eq_ascii(val, b"switch") {
                return Err(RtReloadError::InvalidJson);
            }
        } else if json_inner_eq_ascii(key, b"declared_range") {
            if c.peek_skip_ws() == Some(b'n') {
                c.skip_null()?;
            } else {
                decl_range = Some(parse_f32_array::<2>(c)?);
            }
        } else if json_inner_eq_ascii(key, b"declared_options") {
            if c.peek_skip_ws() == Some(b'n') {
                c.skip_null()?;
            } else {
                scan_declared_options_array(
                    c,
                    &mut stereo_mode_opt_widths,
                    &mut stereo_mode_opt_mask,
                )?;
            }
        } else if json_inner_eq_ascii(key, b"calibration") {
            if c.peek_skip_ws() == Some(b'n') {
                c.skip_null()?;
            } else {
                scan_calibration_obj(c, &mut role, &mut has_anchors, &mut anchors)?;
            }
        } else {
            c.skip_value(59)?;
        }
        Ok(())
    })?;

    if let Some(matched_role) = role
        && ctrl_idx <= u8::MAX as usize
        && (flags.host_role_count as usize) < flags.host_role_slots.len()
    {
        let [decl_min, decl_max] = decl_range.unwrap_or(match matched_role {
            ControlCalibrationRole::Mix => [0.0, 1.0],
            _ => [0.0, 10.0],
        });
        let slot_i = flags.host_role_count as usize;
        flags.host_role_slots[slot_i] = ReloadCalibratedControl {
            ctrl_idx: ctrl_idx as u8,
            role: matched_role,
            decl_min,
            decl_max,
            has_anchors,
            anchors,
            stereo_mode_opt_widths,
            stereo_mode_opt_mask,
        };
        flags.host_role_count += 1;
    }
    Ok(())
}

/// Scans `controls_schema` without heap allocation, capturing up to 16 controls carrying
/// `StereoWidth`, `StereoMode`, or `Mix` calibration roles.
pub(super) fn scan_controls_schema_array(
    cursor: &mut JsonSliceCursor<'_>,
    flags: &mut ReloadHeaderFlags,
) -> Result<(), RtReloadError> {
    cursor.scan_array(|c, ctrl_idx| scan_single_control_obj(c, ctrl_idx, flags))
}

#[cfg(test)]
mod tests {
    use super::super::scan_header_flags_rt;
    use super::*;
    use crate::schema::BlendMode;

    #[test]
    fn test_scan_controls_schema_and_blend_mode_zero_alloc() {
        let json = concat!(
            r#"{"architecture":"sona","#,
            r#""io_config":{"capabilities":{"blend_mode":"host_parallel","default_blend":0.4}},"#,
            r#""controls_schema":["#,
            r#"{"id":"drive","type":"range","declared_range":[0.0,10.0],"declared_options":null,"calibration":null},"#,
            r#"{"id":"delay","type":"range","declared_range":[10.0,500.0],"calibration":{"role":"delay_time_ms","lobatto_physical_anchors":[10.0,100.0,500.0]}},"#,
            r#"{"id":"mode","type":"switch","declared_range":null,"declared_options":["  mono ","\u0053tereo","Wide"],"calibration":{"role":"stereo_mode"}},"#,
            r#"{"id":"width","type":"range","declared_range":[0.0,100.0],"calibration":{"role":"stereo_width","lobatto_physical_anchors":[0.0,0.8,1.8]}},"#,
            r#"{"id":"mix","type":"range","declared_range":null,"calibration":{"role":"mix","lobatto_physical_anchors":null}}"#,
            r#"]}"#
        );
        let flags = scan_header_flags_rt(json.as_bytes()).unwrap();
        assert_eq!(flags.blend_mode, BlendMode::HostParallel);
        assert_eq!(flags.host_role_count, 3);

        let s0 = flags.host_role_slots[0];
        assert_eq!(s0.ctrl_idx, 2);
        assert_eq!(s0.role, ControlCalibrationRole::StereoMode);
        assert_eq!(s0.stereo_mode_opt_mask, 0b0011);
        assert_eq!(s0.stereo_mode_opt_widths[0], 0.0);
        assert_eq!(s0.stereo_mode_opt_widths[1], 1.0);

        let s1 = flags.host_role_slots[1];
        assert_eq!(s1.ctrl_idx, 3);
        assert_eq!(s1.role, ControlCalibrationRole::StereoWidth);
        assert!(s1.has_anchors);
        assert_eq!(s1.anchors, [0.0, 0.8, 1.8]);
        assert_eq!([s1.decl_min, s1.decl_max], [0.0, 100.0]);

        let s2 = flags.host_role_slots[2];
        assert_eq!(s2.ctrl_idx, 4);
        assert_eq!(s2.role, ControlCalibrationRole::Mix);
        assert!(!s2.has_anchors);
        assert_eq!([s2.decl_min, s2.decl_max], [0.0, 1.0]);

        for bad in [
            r#"{"architecture":"sona","io_config":{"capabilities":{"blend_mode":"bad"}}}"#,
            r#"{"architecture":"sona","controls_schema":[{"type":"knob"}]}"#,
            r#"{"architecture":"sona","controls_schema":[{"type":"range","declared_range":[1.0]}]}"#,
            r#"{"architecture":"sona","controls_schema":[{"type":"range","calibration":{"role":"mix","lobatto_physical_anchors":[1.0,2.0]}}]}"#,
        ] {
            assert_eq!(
                scan_header_flags_rt(bad.as_bytes()),
                Err(RtReloadError::InvalidJson)
            );
        }
    }
}
