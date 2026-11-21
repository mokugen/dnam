//! Hardware ARMv8 `FEAT_SHA2` (`SHA256H`, `SHA256H2`, `SHA256SU0`, `SHA256SU1`) block
//! compressor for non-RT container digest verification (`provenance::sha256_digest`).

/// Hardware ARMv8 `FEAT_SHA2` (`SHA256H`, `SHA256H2`, `SHA256SU0`, `SHA256SU1`) block compressor
/// for non-RT container digest verification (`provenance::sha256_digest`).
#[cfg(target_feature = "sha2")]
pub(crate) fn aarch64_sha256_compress(state: &mut [u32; 8], blocks: &[[u8; 64]], k32: &[u32; 64]) {
    use core::arch::aarch64::{
        vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8, vsha256h2q_u32,
        vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
    };
    // SAFETY: Guarded by `#[cfg(target_feature = "sha2")]`; all slice bounds (`8`, `64`, `64`) are fixed at compile time.
    unsafe {
        let mut abcd = vld1q_u32(state.as_ptr());
        let mut efgh = vld1q_u32(state.as_ptr().add(4));

        for block in blocks {
            let abcd_orig = abcd;
            let efgh_orig = efgh;

            let mut s0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr())));
            let mut s1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(16))));
            let mut s2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(32))));
            let mut s3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(48))));

            let mut tmp = vaddq_u32(s0, vld1q_u32(k32.as_ptr()));
            let mut abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            tmp = vaddq_u32(s1, vld1q_u32(k32.as_ptr().add(4)));
            abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            tmp = vaddq_u32(s2, vld1q_u32(k32.as_ptr().add(8)));
            abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            tmp = vaddq_u32(s3, vld1q_u32(k32.as_ptr().add(12)));
            abcd_prev = abcd;
            abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
            efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

            let mut t = 16usize;
            while t < 64 {
                s0 = vsha256su1q_u32(vsha256su0q_u32(s0, s1), s2, s3);
                tmp = vaddq_u32(s0, vld1q_u32(k32.as_ptr().add(t)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

                s1 = vsha256su1q_u32(vsha256su0q_u32(s1, s2), s3, s0);
                tmp = vaddq_u32(s1, vld1q_u32(k32.as_ptr().add(t + 4)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

                s2 = vsha256su1q_u32(vsha256su0q_u32(s2, s3), s0, s1);
                tmp = vaddq_u32(s2, vld1q_u32(k32.as_ptr().add(t + 8)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);

                s3 = vsha256su1q_u32(vsha256su0q_u32(s3, s0), s1, s2);
                tmp = vaddq_u32(s3, vld1q_u32(k32.as_ptr().add(t + 12)));
                abcd_prev = abcd;
                abcd = vsha256hq_u32(abcd_prev, efgh, tmp);
                efgh = vsha256h2q_u32(efgh, abcd_prev, tmp);
                t += 16;
            }

            abcd = vaddq_u32(abcd, abcd_orig);
            efgh = vaddq_u32(efgh, efgh_orig);
        }

        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}
