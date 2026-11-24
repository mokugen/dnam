#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Wasm32-safe in-memory 12/16/24/32-bit PCM & 32-bit IEEE-754 float RIFF/WAVE codec (`align::wav_codec`).

use zerocopy::byteorder::little_endian::{F32, I16, I32, U16, U32};
use zerocopy::{FromBytes, IntoBytes};

use super::wav_wire::{
    RiffChunkHeaderWire, RiffHeaderWire, WavExtensibleHeader68Wire, WavFmtBaseWire,
    WavFmtExtensibleWire, WavPcmHeader44Wire,
};
use super::{AlignError, WavData};
use crate::targets::CaptureBitDepth;

const KSDATAFORMAT_SUBTYPE_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

fn build_wav_riff_header(
    sample_count: usize,
    sample_rate_hz: u32,
    channels: u16,
    bytes_per_sample: u16,
    bits_per_sample: u16,
    audio_format: u16,
) -> Result<(Vec<u8>, u32), AlignError> {
    if channels == 0 || sample_rate_hz == 0 {
        return Err(AlignError::InvalidWav(
            "channels and sample_rate_hz must be >= 1".to_string(),
        ));
    }
    if !sample_count.is_multiple_of(usize::from(channels)) {
        return Err(AlignError::InvalidWav(
            "WAV sample count must be a multiple of channels".to_string(),
        ));
    }
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| AlignError::InvalidWav("WAV block_align overflow".to_string()))?;
    let byte_rate = sample_rate_hz
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| AlignError::InvalidWav("WAV byte_rate overflow".to_string()))?;
    let raw_data_len = sample_count
        .checked_mul(usize::from(bytes_per_sample))
        .ok_or_else(|| AlignError::InvalidWav("WAV payload size overflow".to_string()))?;
    let data_bytes_len = u32::try_from(raw_data_len)
        .map_err(|_| AlignError::InvalidWav("WAV payload exceeds 4 GiB limit".to_string()))?;
    let pad = data_bytes_len & 1;
    let riff_size = 36_u32
        .checked_add(data_bytes_len)
        .and_then(|v| v.checked_add(pad))
        .ok_or_else(|| AlignError::InvalidWav("WAV RIFF size exceeds 4 GiB limit".to_string()))?;
    let cap = raw_data_len
        .checked_add(44 + usize::from(pad == 1))
        .ok_or_else(|| AlignError::InvalidWav("WAV buffer capacity overflow".to_string()))?;

    let hdr = WavPcmHeader44Wire {
        riff: RiffHeaderWire {
            riff_id: *b"RIFF",
            riff_size: U32::new(riff_size),
            wave_id: *b"WAVE",
        },
        fmt_hdr: RiffChunkHeaderWire {
            chunk_id: *b"fmt ",
            chunk_len: U32::new(16),
        },
        fmt_body: WavFmtBaseWire {
            audio_format: U16::new(audio_format),
            channels: U16::new(channels),
            sample_rate_hz: U32::new(sample_rate_hz),
            avg_bytes_per_sec: U32::new(byte_rate),
            block_align: U16::new(block_align),
            bits_per_sample: U16::new(bits_per_sample),
        },
        data_hdr: RiffChunkHeaderWire {
            chunk_id: *b"data",
            chunk_len: U32::new(data_bytes_len),
        },
    };

    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(hdr.as_bytes());
    Ok((buf, pad))
}

#[inline]
fn finalize_wav_vec(mut buf: Vec<u8>, pad: u32) -> Vec<u8> {
    if pad == 1 {
        buf.push(0_u8);
    }
    buf
}

/// Encodes `samples` (`[-1.0, 1.0]`) to a 12-bit signed PCM `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`,
/// `wBitsPerSample = 16`, `wValidBitsPerSample = 12`, MSB-aligned) RIFF/WAVE byte buffer.
pub fn encode_wav_12bit(
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<Vec<u8>, AlignError> {
    if channels == 0 || sample_rate_hz == 0 {
        return Err(AlignError::InvalidWav(
            "channels and sample_rate_hz must be >= 1".to_string(),
        ));
    }
    if !samples.len().is_multiple_of(usize::from(channels)) {
        return Err(AlignError::InvalidWav(
            "WAV sample count must be a multiple of channels".to_string(),
        ));
    }
    let block_align = channels
        .checked_mul(2)
        .ok_or_else(|| AlignError::InvalidWav("WAV block_align overflow".to_string()))?;
    let byte_rate = sample_rate_hz
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| AlignError::InvalidWav("WAV byte_rate overflow".to_string()))?;
    let raw_data_len = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| AlignError::InvalidWav("WAV payload size overflow".to_string()))?;
    let data_bytes_len = u32::try_from(raw_data_len)
        .map_err(|_| AlignError::InvalidWav("WAV payload exceeds 4 GiB limit".to_string()))?;
    let pad = data_bytes_len & 1;
    let riff_size = 60_u32
        .checked_add(data_bytes_len)
        .and_then(|v| v.checked_add(pad))
        .ok_or_else(|| AlignError::InvalidWav("WAV RIFF size exceeds 4 GiB limit".to_string()))?;
    let cap = raw_data_len
        .checked_add(68 + usize::from(pad == 1))
        .ok_or_else(|| AlignError::InvalidWav("WAV buffer capacity overflow".to_string()))?;

    let channel_mask: u32 = match channels {
        1 => 0x0000_0004,
        2 => 0x0000_0003,
        _ => 0,
    };

    let hdr = WavExtensibleHeader68Wire {
        riff: RiffHeaderWire {
            riff_id: *b"RIFF",
            riff_size: U32::new(riff_size),
            wave_id: *b"WAVE",
        },
        fmt_hdr: RiffChunkHeaderWire {
            chunk_id: *b"fmt ",
            chunk_len: U32::new(40),
        },
        fmt_body: WavFmtExtensibleWire {
            base: WavFmtBaseWire {
                audio_format: U16::new(0xFFFE),
                channels: U16::new(channels),
                sample_rate_hz: U32::new(sample_rate_hz),
                avg_bytes_per_sec: U32::new(byte_rate),
                block_align: U16::new(block_align),
                bits_per_sample: U16::new(16),
            },
            cb_size: U16::new(22),
            valid_bits_per_sample: U16::new(12),
            channel_mask: U32::new(channel_mask),
            sub_format_tag: U16::new(1),
            sub_format_tail: KSDATAFORMAT_SUBTYPE_TAIL,
        },
        data_hdr: RiffChunkHeaderWire {
            chunk_id: *b"data",
            chunk_len: U32::new(data_bytes_len),
        },
    };

    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(hdr.as_bytes());

    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 2_047.0 / 2_048.0)
        } else {
            0.0_f64
        };
        let q12 = (clamped * 2_048.0)
            .round_ties_even()
            .clamp(-2_048.0, 2_047.0) as i16;
        let msb_aligned = q12 << 4;
        buf.extend_from_slice(I16::new(msb_aligned).as_bytes());
    }
    Ok(finalize_wav_vec(buf, pad))
}

/// Encodes `samples` (`[-1.0, 1.0]`) to a 16-bit signed PCM RIFF/WAVE byte buffer.
pub fn encode_wav_16bit(
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<Vec<u8>, AlignError> {
    let (mut buf, pad) = build_wav_riff_header(samples.len(), sample_rate_hz, channels, 2, 16, 1)?;
    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 32_767.0 / 32_768.0)
        } else {
            0.0_f64
        };
        let q = (clamped * 32_768.0).round().clamp(-32_768.0, 32_767.0) as i16;
        buf.extend_from_slice(I16::new(q).as_bytes());
    }
    Ok(finalize_wav_vec(buf, pad))
}

/// Encodes `samples` (`[-1.0, 1.0]`) to a 24-bit signed PCM RIFF/WAVE byte buffer.
pub fn encode_wav_24bit(
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<Vec<u8>, AlignError> {
    let (mut buf, pad) = build_wav_riff_header(samples.len(), sample_rate_hz, channels, 3, 24, 1)?;
    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 8_388_607.0 / 8_388_608.0)
        } else {
            0.0_f64
        };
        let q = (clamped * 8_388_608.0)
            .round()
            .clamp(-8_388_608.0, 8_388_607.0) as i32;
        buf.extend_from_slice(&I32::new(q).as_bytes()[..3]);
    }
    Ok(finalize_wav_vec(buf, pad))
}

/// Encodes `samples` (`[-1.0, 1.0]`) to a 32-bit signed PCM (`WAVE_FORMAT_PCM`, `format_tag = 1`)
/// RIFF/WAVE byte buffer.
pub fn encode_wav_32bit_int(
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<Vec<u8>, AlignError> {
    let (mut buf, pad) = build_wav_riff_header(samples.len(), sample_rate_hz, channels, 4, 32, 1)?;
    for &s in samples {
        let clamped = if s.is_finite() {
            f64::from(s).clamp(-1.0, 2_147_483_647.0 / 2_147_483_648.0)
        } else {
            0.0_f64
        };
        let q = (clamped * 2_147_483_648.0)
            .round()
            .clamp(-2_147_483_648.0, 2_147_483_647.0) as i32;
        buf.extend_from_slice(I32::new(q).as_bytes());
    }
    Ok(finalize_wav_vec(buf, pad))
}

/// Encodes `samples` to an unclipped 32-bit IEEE-754 float (`WAVE_FORMAT_IEEE_FLOAT`, `format_tag = 3`)
/// RIFF/WAVE byte buffer, preserving full 32-bit float headroom beyond `[-1.0, 1.0]`.
pub fn encode_wav_32bit_float(
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<Vec<u8>, AlignError> {
    let (mut buf, pad) = build_wav_riff_header(samples.len(), sample_rate_hz, channels, 4, 32, 3)?;
    for &s in samples {
        let safe_s = if s.is_finite() { s } else { 0.0_f32 };
        buf.extend_from_slice(F32::new(safe_s).as_bytes());
    }
    Ok(finalize_wav_vec(buf, pad))
}

/// Encodes `samples` into an in-memory RIFF/WAVE buffer at the requested [`CaptureBitDepth`].
pub fn encode_wav_with_bit_depth(
    samples: &[f32],
    sample_rate_hz: u32,
    channels: u16,
    bit_depth: CaptureBitDepth,
) -> Result<Vec<u8>, AlignError> {
    match bit_depth {
        CaptureBitDepth::Int12 => encode_wav_12bit(samples, sample_rate_hz, channels),
        CaptureBitDepth::Int16 => encode_wav_16bit(samples, sample_rate_hz, channels),
        CaptureBitDepth::Int24 => encode_wav_24bit(samples, sample_rate_hz, channels),
        CaptureBitDepth::Int32 => encode_wav_32bit_int(samples, sample_rate_hz, channels),
        CaptureBitDepth::Float32 => encode_wav_32bit_float(samples, sample_rate_hz, channels),
    }
}

fn parse_wav_fmt_chunk(body: &[u8]) -> Result<(u32, u16, u16, u16, bool), AlignError> {
    if body.len() == 17 {
        return Err(AlignError::InvalidWav(
            "Truncated WAV fmt chunk".to_string(),
        ));
    }
    let Ok((base, ext_tail)) = WavFmtBaseWire::ref_from_prefix(body) else {
        return Err(AlignError::InvalidWav(
            "Truncated WAV fmt chunk".to_string(),
        ));
    };
    if !ext_tail.is_empty() {
        let Ok((cb_size_wire, _)) = U16::ref_from_prefix(ext_tail) else {
            return Err(AlignError::InvalidWav(
                "Truncated WAV fmt chunk".to_string(),
            ));
        };
        let cb_size = usize::from(cb_size_wire.get());
        if 18_usize.checked_add(cb_size).is_none_or(|n| n > body.len()) {
            return Err(AlignError::InvalidWav(
                "Truncated WAV fmt extension bytes (18 + cbSize > chunk length)".to_string(),
            ));
        }
    }
    let audio_format = base.audio_format.get();
    let channels = base.channels.get();
    let sample_rate_hz = base.sample_rate_hz.get();
    let avg_bytes_per_sec = base.avg_bytes_per_sec.get();
    let block_align = base.block_align.get();
    let bits_per_sample = base.bits_per_sample.get();
    if channels == 0 || sample_rate_hz == 0 {
        return Err(AlignError::InvalidWav(
            "WAV fmt chunk requires channels >= 1 and sample_rate_hz >= 1".to_string(),
        ));
    }
    let (effective_bits, container_bytes, is_float) = match audio_format {
        1 => match bits_per_sample {
            12 | 16 => (bits_per_sample, 2_u16, false),
            24 => (24, 3_u16, false),
            32 => (32, 4_u16, false),
            _ => {
                return Err(AlignError::InvalidWav(format!(
                    "Unsupported WAV PCM bit depth ({bits_per_sample}-bit)"
                )));
            }
        },
        3 if bits_per_sample == 32 => (32, 4_u16, true),
        0xFFFE => {
            let Ok((ext, _)) = WavFmtExtensibleWire::ref_from_prefix(body) else {
                return Err(AlignError::InvalidWav(
                    "Unsupported or truncated WAVE_FORMAT_EXTENSIBLE SubFormat".to_string(),
                ));
            };
            if ext.cb_size.get() < 22 || ext.sub_format_tail != KSDATAFORMAT_SUBTYPE_TAIL {
                return Err(AlignError::InvalidWav(
                    "Unsupported or truncated WAVE_FORMAT_EXTENSIBLE SubFormat".to_string(),
                ));
            }
            let valid_bits = ext.valid_bits_per_sample.get();
            if valid_bits > bits_per_sample {
                return Err(AlignError::InvalidWav(format!(
                    "WAVE_FORMAT_EXTENSIBLE wValidBitsPerSample ({valid_bits}) exceeds wBitsPerSample ({bits_per_sample})"
                )));
            }
            let eff = if valid_bits == 0 {
                bits_per_sample
            } else {
                valid_bits
            };
            match ext.sub_format_tag.get() {
                1 if matches!(bits_per_sample, 16 | 24 | 32)
                    && matches!(eff, 12 | 16 | 24 | 32) =>
                {
                    (eff, bits_per_sample / 8, false)
                }
                3 if bits_per_sample == 32 && eff == 32 => (32, 4_u16, true),
                sub => {
                    return Err(AlignError::InvalidWav(format!(
                        "Unsupported WAVE_FORMAT_EXTENSIBLE SubFormat tag {sub} ({bits_per_sample}-bit, {eff} valid)"
                    )));
                }
            }
        }
        other => {
            return Err(AlignError::InvalidWav(format!(
                "Unsupported WAV format tag {other} ({bits_per_sample}-bit)"
            )));
        }
    };
    let expected_align = channels
        .checked_mul(container_bytes)
        .ok_or_else(|| AlignError::InvalidWav("WAV block_align overflow".to_string()))?;
    if block_align != expected_align {
        return Err(AlignError::InvalidWav(format!(
            "Invalid WAV block_align {block_align} (expected {expected_align})"
        )));
    }
    let expected_byte_rate = sample_rate_hz
        .checked_mul(u32::from(expected_align))
        .ok_or_else(|| AlignError::InvalidWav("WAV byte_rate overflow".to_string()))?;
    if avg_bytes_per_sec != expected_byte_rate {
        return Err(AlignError::InvalidWav(format!(
            "Invalid WAV byte_rate {avg_bytes_per_sec} (expected {expected_byte_rate})"
        )));
    }
    Ok((
        sample_rate_hz,
        channels,
        container_bytes,
        effective_bits,
        is_float,
    ))
}

fn decode_pcm_payload_with_format(
    payload: &[u8],
    container_bytes: u16,
    effective_bits: u16,
    is_float: bool,
) -> Result<Vec<f32>, AlignError> {
    if is_float {
        if container_bytes != 4 || effective_bits != 32 {
            return Err(AlignError::InvalidWav(format!(
                "Unsupported WAV float format ({container_bytes}B container, {effective_bits}-bit)"
            )));
        }
        let Ok(floats) = <[F32]>::ref_from_bytes(payload) else {
            return Err(AlignError::InvalidWav(
                "WAV data chunk length is not a multiple of 4 bytes (32-bit float)".to_string(),
            ));
        };
        let mut out = Vec::with_capacity(floats.len());
        for word in floats {
            let v = word.get();
            if !v.is_finite() {
                return Err(AlignError::InvalidWav(
                    "WAV 32-bit float payload contains non-finite (NaN or Inf) sample".to_string(),
                ));
            }
            out.push(v);
        }
        return Ok(out);
    }

    if !matches!(container_bytes, 2..=4)
        || !matches!(effective_bits, 12 | 16 | 24 | 32)
        || effective_bits > container_bytes * 8
    {
        return Err(AlignError::InvalidWav(format!(
            "Unsupported WAV PCM format ({container_bytes}B container, {effective_bits}-bit)"
        )));
    }
    let width = usize::from(container_bytes);
    if !payload.len().is_multiple_of(width) {
        return Err(AlignError::InvalidWav(format!(
            "WAV data chunk length is not a multiple of {width} bytes ({effective_bits}-bit PCM)"
        )));
    }
    let mask: i32 = if effective_bits < 32 {
        !((1_i32 << (32 - effective_bits)) - 1)
    } else {
        -1_i32
    };
    let mut out = Vec::with_capacity(payload.len() / width);
    match width {
        2 => {
            let Ok(words) = <[I16]>::ref_from_bytes(payload) else {
                return Err(AlignError::InvalidWav(
                    "WAV data chunk length is not a multiple of 2 bytes (PCM)".to_string(),
                ));
            };
            for w in words {
                let masked = (i32::from(w.get()) << 16) & mask;
                out.push((f64::from(masked) / 2_147_483_648.0) as f32);
            }
        }
        3 => {
            for chunk in payload.as_chunks::<3>().0 {
                let packed = [0_u8, chunk[0], chunk[1], chunk[2]];
                let v32 = I32::read_from_bytes(&packed).map_or(0, I32::get);
                let masked = v32 & mask;
                out.push((f64::from(masked) / 2_147_483_648.0) as f32);
            }
        }
        _ => {
            let Ok(words) = <[I32]>::ref_from_bytes(payload) else {
                return Err(AlignError::InvalidWav(
                    "WAV data chunk length is not a multiple of 4 bytes (PCM)".to_string(),
                ));
            };
            for w in words {
                let masked = w.get() & mask;
                out.push((f64::from(masked) / 2_147_483_648.0) as f32);
            }
        }
    }
    Ok(out)
}

/// Decodes an in-memory 12-bit, 16-bit, 24-bit, or 32-bit (signed PCM or IEEE-754 float)
/// RIFF/WAVE byte slice into [`WavData`].
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<WavData, AlignError> {
    let Ok((riff_hdr, _)) = RiffHeaderWire::ref_from_prefix(bytes) else {
        return Err(AlignError::InvalidWav(
            "Invalid RIFF/WAVE header".to_string(),
        ));
    };
    if &riff_hdr.riff_id != b"RIFF" || &riff_hdr.wave_id != b"WAVE" {
        return Err(AlignError::InvalidWav(
            "Invalid RIFF/WAVE header".to_string(),
        ));
    }
    let riff_size = riff_hdr.riff_size.get() as usize;
    let riff_end = 8_usize
        .checked_add(riff_size)
        .filter(|&end| riff_size >= 4 && end <= bytes.len())
        .ok_or_else(|| AlignError::InvalidWav("Truncated or invalid WAV RIFF size".to_string()))?;

    let mut fmt_info: Option<(u32, u16, u16, u16, bool)> = None;
    let mut data_samples: Option<Vec<f32>> = None;
    let mut offset = 12_usize;
    while offset < riff_end {
        let rem = &bytes[offset..riff_end];
        let Ok((ch_hdr, _)) = RiffChunkHeaderWire::ref_from_prefix(rem) else {
            return Err(AlignError::InvalidWav(
                "Truncated WAV RIFF chunk header".to_string(),
            ));
        };
        let chunk_id = ch_hdr.chunk_id;
        let chunk_len = ch_hdr.chunk_len.get() as usize;
        offset += 8;
        let end = offset
            .checked_add(chunk_len)
            .filter(|&e| e <= riff_end)
            .ok_or_else(|| AlignError::InvalidWav("Truncated WAV RIFF chunk".to_string()))?;
        let body = &bytes[offset..end];

        if &chunk_id == b"fmt " {
            if fmt_info.is_some() {
                return Err(AlignError::InvalidWav(
                    "Duplicate fmt chunk in WAV buffer".to_string(),
                ));
            }
            fmt_info = Some(parse_wav_fmt_chunk(body)?);
        } else if &chunk_id == b"data" {
            if data_samples.is_some() {
                return Err(AlignError::InvalidWav(
                    "Duplicate data chunk in WAV buffer".to_string(),
                ));
            }
            let (_, channels, container_bytes, effective_bits, is_float) =
                fmt_info.ok_or_else(|| {
                    AlignError::InvalidWav("WAV data chunk precedes fmt chunk".to_string())
                })?;
            let samples =
                decode_pcm_payload_with_format(body, container_bytes, effective_bits, is_float)?;
            if !samples.len().is_multiple_of(usize::from(channels)) {
                return Err(AlignError::InvalidWav(
                    "WAV data chunk contains an incomplete multi-channel frame".to_string(),
                ));
            }
            data_samples = Some(samples);
        }

        let padded_end = end
            .checked_add(chunk_len & 1)
            .filter(|&pe| pe <= riff_end)
            .ok_or_else(|| {
                AlignError::InvalidWav("Missing RIFF odd-chunk alignment pad byte".to_string())
            })?;
        offset = padded_end;
    }

    let (sample_rate_hz, channels, _, _, _) = fmt_info
        .ok_or_else(|| AlignError::InvalidWav("Missing fmt chunk in WAV buffer".to_string()))?;
    let samples = data_samples
        .ok_or_else(|| AlignError::InvalidWav("Missing data chunk in WAV buffer".to_string()))?;
    Ok(WavData {
        sample_rate_hz,
        channels,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_wav_all_5_bit_depths_and_24bit_stereo(
            samples in proptest::collection::vec(-0.999_f32..0.999_f32, 4..64),
        ) {
            let even_len = samples.len() & !1;
            let stereo = &samples[..even_len];
            let decoded = decode_wav_bytes(&encode_wav_24bit(stereo, 48_000, 2).unwrap()).unwrap();
            prop_assert_eq!((decoded.sample_rate_hz, decoded.channels, decoded.samples.len()), (48_000, 2, stereo.len()));
            for (&d, &t) in stereo.iter().zip(&decoded.samples) {
                prop_assert!((d - t).abs() < 2.0e-7_f32);
            }
            for (depth, tol) in [
                (CaptureBitDepth::Int12, 5.0e-4_f32),
                (CaptureBitDepth::Int16, 3.2e-5_f32),
                (CaptureBitDepth::Int24, 2.0e-7_f32),
                (CaptureBitDepth::Int32, 1.0e-7_f32),
                (CaptureBitDepth::Float32, 0.0_f32),
            ] {
                let bytes = encode_wav_with_bit_depth(&samples, 48_000, 1, depth).unwrap();
                let wav = decode_wav_bytes(&bytes).unwrap();
                for (&s, &d) in samples.iter().zip(&wav.samples) {
                    prop_assert!((s - d).abs() <= tol);
                }
            }
        }
    }

    #[test]
    fn test_wav_codec_edge_cases_and_malformed_chunks() {
        let odd_24 = encode_wav_24bit(&[-1.0, 0.0, 1.0, f32::NAN, 0.25], 44_100, 1).unwrap();
        assert_eq!(odd_24.len() % 2, 0);
        let dec_odd = decode_wav_bytes(&odd_24).unwrap();
        assert_eq!((dec_odd.samples.len(), dec_odd.samples[3]), (5, 0.0));

        let w12 = encode_wav_12bit(&[-1.0, 1.0, f32::INFINITY, -0.25], 96_000, 2).unwrap();
        assert_eq!(decode_wav_bytes(&w12).unwrap().channels, 2);
        let w12_4ch = encode_wav_12bit(&[0.1, -0.1, 0.2, -0.2], 48_000, 4).unwrap();
        assert_eq!(decode_wav_bytes(&w12_4ch).unwrap().channels, 4);

        let w32f = encode_wav_32bit_float(&[-2.5, 3.25, f32::NAN, 0.0], 192_000, 2).unwrap();
        assert_eq!(
            decode_wav_bytes(&w32f).unwrap().samples,
            vec![-2.5, 3.25, 0.0, 0.0]
        );
        let w16 = encode_wav_16bit(&[f32::NAN, 0.5], 48_000, 1).unwrap();
        assert_eq!(decode_wav_bytes(&w16).unwrap().samples[0], 0.0);
        let w32i = encode_wav_32bit_int(&[f32::NAN, -0.5], 48_000, 1).unwrap();
        assert_eq!(decode_wav_bytes(&w32i).unwrap().samples[0], 0.0);

        // Overflow and invalid header closures in build_wav_riff_header / encode_wav_12bit
        assert!(encode_wav_16bit(&[0.0], 0, 1).is_err());
        assert!(encode_wav_12bit(&[0.0], 48_000, 0).is_err());
        assert!(encode_wav_16bit(&[0.0, 0.1, 0.2], 48_000, 2).is_err());
        assert!(encode_wav_12bit(&[0.0, 0.1, 0.2], 48_000, 2).is_err());
        assert!(encode_wav_16bit(&[], 48_000, u16::MAX).is_err());
        assert!(encode_wav_12bit(&[], 48_000, u16::MAX).is_err());
        assert!(encode_wav_16bit(&[], u32::MAX, 2).is_err());
        assert!(encode_wav_12bit(&[], u32::MAX, 2).is_err());

        // Direct parse_wav_fmt_chunk edge cases (17B, cbSize, 0xFFFE <40B, 0xFFFE 32f, PCM 12b, overflows)
        assert!(parse_wav_fmt_chunk(&[0; 17]).is_err());
        assert!(parse_wav_fmt_chunk(&[0; 12]).is_err());
        let mut ext_bad_cb = w12[20..60].to_vec();
        ext_bad_cb[16..18].copy_from_slice(&99_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&ext_bad_cb[..20]).is_err());
        assert!(parse_wav_fmt_chunk(&ext_bad_cb[..30]).is_err());

        let mut ext_32f = w12[20..60].to_vec();
        ext_32f[8..12].copy_from_slice(&(96_000_u32 * 8).to_le_bytes());
        ext_32f[12..14].copy_from_slice(&8_u16.to_le_bytes());
        ext_32f[14..16].copy_from_slice(&32_u16.to_le_bytes());
        ext_32f[18..20].copy_from_slice(&32_u16.to_le_bytes());
        ext_32f[24..26].copy_from_slice(&3_u16.to_le_bytes());
        assert_eq!(
            parse_wav_fmt_chunk(&ext_32f).unwrap(),
            (96_000, 2, 4, 32, true)
        );

        let mut pcm12 = w16[20..36].to_vec();
        pcm12[14..16].copy_from_slice(&12_u16.to_le_bytes());
        assert_eq!(
            parse_wav_fmt_chunk(&pcm12).unwrap(),
            (48_000, 1, 2, 12, false)
        );

        let mut align_ovf = vec![0_u8; 16];
        align_ovf[0..2].copy_from_slice(&1_u16.to_le_bytes());
        align_ovf[2..4].copy_from_slice(&u16::MAX.to_le_bytes());
        align_ovf[4..8].copy_from_slice(&1_u32.to_le_bytes());
        align_ovf[14..16].copy_from_slice(&24_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&align_ovf).is_err());
        let mut br_ovf = align_ovf.clone();
        br_ovf[2..4].copy_from_slice(&2_u16.to_le_bytes());
        br_ovf[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        br_ovf[14..16].copy_from_slice(&16_u16.to_le_bytes());
        assert!(parse_wav_fmt_chunk(&br_ovf).is_err());

        // Direct decode_pcm_payload_with_format edge cases
        assert!(decode_pcm_payload_with_format(&[0; 4], 2, 32, true).is_err());
        assert!(decode_pcm_payload_with_format(&[0; 5], 4, 32, true).is_err());
        assert!(decode_pcm_payload_with_format(&[0; 4], 1, 16, false).is_err());
        assert!(decode_pcm_payload_with_format(&[0; 4], 2, 8, false).is_err());
        assert!(decode_pcm_payload_with_format(&[0; 4], 2, 24, false).is_err());
        assert!(decode_pcm_payload_with_format(&[0; 5], 2, 16, false).is_err());

        // Malformed RIFF container framings
        assert!(decode_wav_bytes(b"short").is_err());
        assert!(decode_wav_bytes(b"NOPE____WAVE").is_err());
        let mut trunc_hdr = w16[..16].to_vec();
        trunc_hdr[4..8].copy_from_slice(&8_u32.to_le_bytes());
        assert!(decode_wav_bytes(&trunc_hdr).is_err());

        let mut trunc_chunk = w16.clone();
        trunc_chunk[40..44].copy_from_slice(&999_u32.to_le_bytes());
        assert!(decode_wav_bytes(&trunc_chunk).is_err());

        let mut dup_fmt = w16.clone();
        dup_fmt.extend_from_slice(&w16[12..36]);
        let df_len = (dup_fmt.len() - 8) as u32;
        dup_fmt[4..8].copy_from_slice(&df_len.to_le_bytes());
        assert!(decode_wav_bytes(&dup_fmt).is_err());

        let mut dup_data = w16.clone();
        dup_data.extend_from_slice(&w16[36..]);
        let dd_len = (dup_data.len() - 8) as u32;
        dup_data[4..8].copy_from_slice(&dd_len.to_le_bytes());
        assert!(decode_wav_bytes(&dup_data).is_err());

        let mut odd_missing_pad = encode_wav_24bit(&[0.25], 48_000, 1).unwrap();
        odd_missing_pad.pop();
        let omp_len = (odd_missing_pad.len() - 8) as u32;
        odd_missing_pad[4..8].copy_from_slice(&omp_len.to_le_bytes());
        assert!(decode_wav_bytes(&odd_missing_pad).is_err());

        let mut no_fmt = b"RIFF____WAVEJUNK\x02\x00\x00\x00\x00\x00".to_vec();
        let nf_len = (no_fmt.len() - 8) as u32;
        no_fmt[4..8].copy_from_slice(&nf_len.to_le_bytes());
        assert!(decode_wav_bytes(&no_fmt).is_err());

        let mut nan_float = w32f.clone();
        let data_off = nan_float.len() - 16;
        nan_float[data_off..data_off + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_wav_bytes(&nan_float).is_err());
    }
}
