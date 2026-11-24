//! Pure-Rust RFC 8439 §2.3 ChaCha20 stream generator and unbiased bounded-index sampler
//! for deterministic CPU `PANAMA` ensemble bootstrap draws (`#![forbid(unsafe_code)]`,
//! `ref[train.audit.report.prescription]`).

use zerocopy::byteorder::little_endian::{U32, U64};
use zerocopy::{FromBytes, IntoBytes};

/// RFC 8439 `"expand 32-byte k"` 32-bit state constants.
const CHACHA20_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

const _: () = assert!(core::mem::size_of::<[u32; 16]>() == 64);

#[inline]
fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

#[inline]
fn read_u32_le(bytes: &[u8]) -> u32 {
    U32::read_from_prefix(bytes).map_or(0, |(w, _)| w.get())
}

/// Evaluates one 64-byte RFC 8439 §2.3 ChaCha20 block for `(key, counter, nonce)`.
#[must_use]
pub fn chacha20_block_rfc8439(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut initial = [0_u32; 16];
    initial[0..4].copy_from_slice(&CHACHA20_CONSTANTS);
    for (i, word) in initial[4..12].iter_mut().enumerate() {
        *word = read_u32_le(&key[i * 4..i * 4 + 4]);
    }
    initial[12] = counter;
    for (i, word) in initial[13..16].iter_mut().enumerate() {
        *word = read_u32_le(&nonce[i * 4..i * 4 + 4]);
    }

    let mut working = initial;
    for _ in 0..10 {
        quarter_round(&mut working, 0, 4, 8, 12);
        quarter_round(&mut working, 1, 5, 9, 13);
        quarter_round(&mut working, 2, 6, 10, 14);
        quarter_round(&mut working, 3, 7, 11, 15);

        quarter_round(&mut working, 0, 5, 10, 15);
        quarter_round(&mut working, 1, 6, 11, 12);
        quarter_round(&mut working, 2, 7, 8, 13);
        quarter_round(&mut working, 3, 4, 9, 14);
    }

    let mut words_le = [U32::new(0); 16];
    for (dst, (w, init)) in words_le.iter_mut().zip(working.iter().zip(initial.iter())) {
        *dst = U32::new(w.wrapping_add(*init));
    }
    let mut out = [0_u8; 64];
    out.copy_from_slice(words_le.as_bytes());
    out
}

/// Consecutive 64-bit little-endian keystream reader backed by RFC 8439 ChaCha20.
#[derive(Clone, Debug)]
pub struct ChaCha20Stream {
    key: [u8; 32],
    nonce: [u8; 12],
    counter: u32,
    block: [u8; 64],
    byte_offset: usize,
}

impl ChaCha20Stream {
    /// Initializes a stream starting at `(nonce, counter)`.
    #[must_use]
    pub const fn new(key: [u8; 32], nonce: [u8; 12], counter: u32) -> Self {
        Self {
            key,
            nonce,
            counter,
            block: [0_u8; 64],
            byte_offset: 64,
        }
    }

    /// Reads the next little-endian `u64` keystream word, advancing block `counter` as needed.
    pub fn next_u64_le(&mut self) -> u64 {
        if self.byte_offset >= 64 {
            self.block = chacha20_block_rfc8439(&self.key, self.counter, &self.nonce);
            self.counter = self.counter.wrapping_add(1);
            self.byte_offset = 0;
        }
        let s = self.byte_offset;
        self.byte_offset += 8;
        U64::read_from_prefix(&self.block[s..s + 8]).map_or(0, |(w, _)| w.get())
    }

    /// Draws a deterministic bootstrap index `floor((u64_le * bound) / 2^64)` inside `0..bound`.
    pub fn draw_uniform_index(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        let word = self.next_u64_le();
        ((u128::from(word) * (bound as u128)) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 8439 Section 2.3.2 canonical 64-byte ChaCha20 block test vector.
    #[test]
    fn test_chacha20_rfc8439_section_2_3_2_vector_and_draw_bounds() {
        let mut key = [0_u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let nonce = [
            0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00,
        ];
        let block = chacha20_block_rfc8439(&key, 1, &nonce);

        #[rustfmt::skip]
        const EXPECTED: [u8; 64] = [
            0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15,
            0x50, 0x0f, 0xdd, 0x1f, 0xa3, 0x20, 0x71, 0xc4,
            0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03,
            0x04, 0x22, 0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e,
            0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa, 0x09,
            0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2,
            0xb5, 0x12, 0x9c, 0xd1, 0xde, 0x16, 0x4e, 0xb9,
            0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e,
        ];
        assert_eq!(block, EXPECTED);

        let mut stream = ChaCha20Stream::new(key, nonce, 1);
        let first_word = stream.next_u64_le();
        assert_eq!(
            first_word,
            U64::read_from_prefix(&EXPECTED[0..8]).unwrap().0.get()
        );
        for _ in 0..16 {
            let idx = stream.draw_uniform_index(7);
            assert!(idx < 7);
        }
        assert_eq!(stream.draw_uniform_index(1), 0);
        assert_eq!(stream.draw_uniform_index(0), 0);
    }
}
