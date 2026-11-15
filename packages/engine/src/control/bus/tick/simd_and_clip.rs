//! Control-bus 16-D modulation assembly, SIMD micro-ramping, trust-region clipping,
//! and little-endian 4-lane vector loads.

use crate::math::{MathBackend, SimdVec4};

use super::super::ControlBus;

impl<B: MathBackend> ControlBus<B> {
    /// Assembles the unified 16-D Modulation Bus `q_16[k] = [z_ctrl^(8), m_dyn^(8)]^T`.
    #[inline(always)]
    pub fn assemble_q16(z_ctrl: &[f32; 8], m_dyn: &[f32; 8]) -> [f32; 16] {
        let mut q = [0.0; 16];
        q[0..8].copy_from_slice(z_ctrl);
        q[8..16].copy_from_slice(m_dyn);
        q
    }

    /// Evaluates division-free linear micro-ramping deltas with precomputed step `alpha_ramp`
    /// (`ref[dsp.ctrl.dispatch.micro-ramp]`): `delta_theta = (theta_next - theta_curr) * alpha_ramp` (`0 FDIV`).
    #[inline(always)]
    pub fn compute_micro_ramp_deltas(
        theta_curr: &[f32; 16],
        theta_next: &[f32; 16],
        alpha_ramp: f32,
    ) -> [f32; 16] {
        let mut deltas = [0.0; 16];
        let c_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(theta_curr);
        let n_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(theta_next);
        let d_chunks: &mut [[f32; 4]; 4] = zerocopy::transmute_mut!(&mut deltas);
        let a_vec = B::Vec4::splat(alpha_ramp);
        for c in 0..4 {
            B::Vec4::load(&n_chunks[c])
                .sub(B::Vec4::load(&c_chunks[c]))
                .mul(a_vec)
                .store(&mut d_chunks[c]);
        }
        deltas
    }

    /// Convenience 16-sample micro-ramping delta evaluation (`alpha_ramp = 0.0625f32`).
    #[inline(always)]
    pub fn compute_micro_ramp_deltas_16(
        theta_curr: &[f32; 16],
        theta_next: &[f32; 16],
    ) -> [f32; 16] {
        Self::compute_micro_ramp_deltas(theta_curr, theta_next, 0.0625)
    }

    /// Audio-rate sample-by-sample parameter update via 4-lane SIMD addition.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_micro_ramp(theta_audio: &mut [f32; 16], deltas: &[f32; 16]) {
        let t_chunks: &mut [[f32; 4]; 4] = zerocopy::transmute_mut!(theta_audio);
        let d_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(deltas);
        for c in 0..4 {
            B::Vec4::load(&t_chunks[c])
                .add(B::Vec4::load(&d_chunks[c]))
                .store(&mut t_chunks[c]);
        }
    }

    /// Evaluates division-free linear micro-ramping deltas for the 12 active stage parameters (`0..12`).
    #[inline(always)]
    pub fn compute_micro_ramp_deltas_12(
        theta_curr: &[f32; 16],
        theta_next: &[f32; 16],
        alpha_ramp: f32,
    ) -> [f32; 16] {
        let mut deltas = [0.0; 16];
        let c_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(theta_curr);
        let n_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(theta_next);
        let d_chunks: &mut [[f32; 4]; 4] = zerocopy::transmute_mut!(&mut deltas);
        let a_vec = B::Vec4::splat(alpha_ramp);
        for c in 0..3 {
            B::Vec4::load(&n_chunks[c])
                .sub(B::Vec4::load(&c_chunks[c]))
                .mul(a_vec)
                .store(&mut d_chunks[c]);
        }
        deltas
    }

    /// Audio-rate sample-by-sample parameter update for the 12 active stage parameters (`0..12`).
    #[inline(always)]
    pub fn step_micro_ramp_12(theta_audio: &mut [f32; 16], deltas: &[f32; 16]) {
        let t_chunks: &mut [[f32; 4]; 4] = zerocopy::transmute_mut!(theta_audio);
        let d_chunks: &[[f32; 4]; 4] = zerocopy::transmute_ref!(deltas);
        for c in 0..3 {
            B::Vec4::load(&t_chunks[c])
                .add(B::Vec4::load(&d_chunks[c]))
                .store(&mut t_chunks[c]);
        }
    }

    /// Audio-rate sample-by-sample parameter update for all 16 unified stage parameters (`0..16`).
    #[inline(always)]
    pub fn step_micro_ramp_16(theta_audio: &mut [f32; 16], deltas: &[f32; 16]) {
        Self::step_micro_ramp(theta_audio, deltas);
    }
}

/// Applies the outward-checked 4-D trust-region clip (`ref[dsp.ctrl.op-n.trust-clip]`) with
/// fallback to `trust_region_softclip_4d`.
#[inline]
pub fn trust_clip_outward_4d(vec: &mut [f32; 4], radius_mh: f32) {
    let norm_sq = (vec[0] as f64) * (vec[0] as f64)
        + (vec[1] as f64) * (vec[1] as f64)
        + (vec[2] as f64) * (vec[2] as f64)
        + (vec[3] as f64) * (vec[3] as f64);
    let r = radius_mh as f64;
    if norm_sq.is_finite() && r > 0.0 && norm_sq + 1.0e-9 < r * r {
        return;
    }
    let in_bits = [
        vec[0].to_bits(),
        vec[1].to_bits(),
        vec[2].to_bits(),
        vec[3].to_bits(),
    ];
    if let Some((out_bits, _, _)) = crate::math::trust_clip_outward_checked_bits::<4>(
        &in_bits,
        radius_mh.to_bits(),
        (1.0e-12_f32).to_bits(),
    ) {
        *vec = [
            f32::from_bits(out_bits[0]),
            f32::from_bits(out_bits[1]),
            f32::from_bits(out_bits[2]),
            f32::from_bits(out_bits[3]),
        ];
    } else {
        crate::operators::op_n::trust_region_softclip_4d(vec, radius_mh);
    }
}

#[inline(always)]
pub(super) fn read_vec4_le<B: MathBackend>(bytes: &[u8], off: usize) -> B::Vec4 {
    #[cfg(target_endian = "little")]
    {
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&bytes[off..off + 16]);
        let arr: [f32; 4] = zerocopy::transmute!(raw);
        B::Vec4::load(&arr)
    }
    #[cfg(not(target_endian = "little"))]
    {
        let f0 = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let f1 = f32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        let f2 = f32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]);
        let f3 = f32::from_le_bytes([
            bytes[off + 12],
            bytes[off + 13],
            bytes[off + 14],
            bytes[off + 15],
        ]);
        B::Vec4::from_array([f0, f1, f2, f3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};

    #[test]
    fn test_assemble_q16_and_micro_ramps_12_and_16() {
        let _audio = crate::FtzDazGuard::new();
        let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = [-1.0, -2.0, -3.0, -4.0, 0.5, 0.25, 0.125, 0.0];
        let q16 = ControlBus::<ScalarBackend>::assemble_q16(&z, &m);
        assert_eq!(&q16[0..8], &z);
        assert_eq!(&q16[8..16], &m);

        let mut curr = [0.0f32; 16];
        let next = [16.0f32; 16];
        let deltas_16 =
            ControlBus::<DefaultMathBackend>::compute_micro_ramp_deltas_16(&curr, &next);
        assert_eq!(deltas_16, [1.0; 16]);
        ControlBus::<DefaultMathBackend>::step_micro_ramp_16(&mut curr, &deltas_16);
        assert_eq!(curr, [1.0; 16]);

        let mut curr12 = [0.0f32; 16];
        let deltas_12 =
            ControlBus::<ScalarBackend>::compute_micro_ramp_deltas_12(&curr12, &next, 0.25);
        for d in &deltas_12[..12] {
            assert!((d - 4.0).abs() < 1e-6);
        }
        for d in &deltas_12[12..] {
            assert_eq!(*d, 0.0);
        }
        ControlBus::<ScalarBackend>::step_micro_ramp_12(&mut curr12, &deltas_12);
        for v in &curr12[..12] {
            assert!((v - 4.0).abs() < 1e-6);
        }
        for v in &curr12[12..] {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_trust_clip_outward_4d_interior_outward_and_fallback() {
        let mut inside = [0.5, 0.5, 0.0, 0.0];
        trust_clip_outward_4d(&mut inside, 2.0);
        assert_eq!(inside, [0.5, 0.5, 0.0, 0.0]);

        let mut outside = [3.0, 4.0, 0.0, 0.0];
        trust_clip_outward_4d(&mut outside, 2.0);
        let norm = outside.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 2.0).abs() < 1e-4,
            "expected clipped norm 2.0, got {norm}"
        );

        // Non-positive or NaN radius falls back to trust_region_softclip_4d.
        let mut fallback = [2.0, 0.0, 0.0, 0.0];
        trust_clip_outward_4d(&mut fallback, -1.0);
        for v in fallback {
            assert!(v.is_finite());
        }
        let mut nan_vec = [f32::NAN, 1.0, 0.0, 0.0];
        trust_clip_outward_4d(&mut nan_vec, 2.0);
    }
}
