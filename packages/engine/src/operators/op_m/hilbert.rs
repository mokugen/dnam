//! 12-Allpass (`6` I-branch + `6` Q-branch) `H1` Hilbert Single-Sideband Lattice, 15-Rate Anchor
//! Table, Exact `u128` Ties-to-Even Interpolation, and History-Consistent Rate Remap
//! (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.hilbert.h1-anchors]`,
//! `ref[dsp.op-m.hilbert.ties-to-even]`, `ref[dsp.op-m.hilbert.rate-remap]`).

pub mod anchors;

pub use anchors::{
    HILBERT_H1_ANCHORS_15, encode_hilbert_h1_anchor_table_780b,
    interpolate_hilbert_h1_between_anchors_bits, interpolate_hilbert_h1_profile_bits,
};

use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4,
};

/// 12th-order (`6` I-branch + `6` Q-branch, `12` poles total) `H1` Hilbert Single-Sideband Lattice
/// (legacy compatibility struct retained for external bench/example callers; prefer
/// [`HilbertH1Lattice6Sec`] for rate-remappable stereo code, `ref[dsp.op-m.hilbert.lattice]`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`).
///
/// Dual 6-stage transposed direct-form II all-pass lattice generating `-90`-degree quadrature pair
/// `(I[n], Q[n])` (`ref[dsp.op-m.hilbert.lattice]`):
/// `H_I(z) = prod_{k=0..5} (alpha_{I,k} - z^-2) / (1 - alpha_{I,k} z^-2)`
/// `H_Q(z) = z^-1 prod_{k=0..5} (alpha_{Q,k} - z^-2) / (1 - alpha_{Q,k} z^-2)`
#[derive(Clone, Debug)]
pub struct HilbertSsbLattice10th {
    pub i_stages: [f32; 6],
    pub i_stages_z2: [f32; 6],
    pub q_stages: [f32; 6],
    pub q_stages_z2: [f32; 6],
    pub q_z1: f32,
    pub alpha_i: [f32; 6],
    pub alpha_q: [f32; 6],
}

impl Default for HilbertSsbLattice10th {
    fn default() -> Self {
        Self::new()
    }
}

impl HilbertSsbLattice10th {
    /// Canonical `48 kHz` I-branch all-pass coefficients (`H1_ANCHOR_COEFFS_BITS[1]`, derived from
    /// [`dnam_kernels::HILBERT_ALPHA_I`]).
    pub const ALPHA_I: [f32; 6] = dnam_kernels::HILBERT_ALPHA_I;
    /// Canonical `48 kHz` Q-branch all-pass coefficients (`H1_ANCHOR_COEFFS_BITS[1]`, derived from
    /// [`dnam_kernels::HILBERT_ALPHA_Q`]).
    pub const ALPHA_Q: [f32; 6] = dnam_kernels::HILBERT_ALPHA_Q;

    pub const fn new() -> Self {
        Self {
            i_stages: [0.0; 6],
            i_stages_z2: [0.0; 6],
            q_stages: [0.0; 6],
            q_stages_z2: [0.0; 6],
            q_z1: 0.0,
            alpha_i: Self::ALPHA_I,
            alpha_q: Self::ALPHA_Q,
        }
    }

    pub fn reset(&mut self) {
        self.i_stages.fill(0.0);
        self.i_stages_z2.fill(0.0);
        self.q_stages.fill(0.0);
        self.q_stages_z2.fill(0.0);
        self.q_z1 = 0.0;
    }

    /// Recomputes the 12-allpass `H1` coefficients for `sample_rate_hz` (`ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.ties-to-even]`).
    pub fn reconfigure_sample_rate(&mut self, sample_rate_hz: f64) {
        if let Some(bits) = interpolate_hilbert_h1_profile_bits(sample_rate_hz) {
            for k in 0..6 {
                self.alpha_i[k] = f32::from_bits(bits[k]);
                self.alpha_q[k] = f32::from_bits(bits[6 + k]);
            }
        }
    }

    /// Steps the Hilbert lattice, producing analytic signal `(I[n], Q[n])`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let mut curr_i = x;
        for k in 0..6 {
            let a = self.alpha_i[k];
            let s1 = self.i_stages_z2[k];
            let s0 = self.i_stages[k];
            let y = flush_recursive_state_f32(a * curr_i + s1);
            self.i_stages_z2[k] = s0;
            self.i_stages[k] = flush_recursive_state_f32(a * y - curr_i);
            curr_i = y;
        }

        let mut curr_q = self.q_z1;
        self.q_z1 = flush_recursive_state_f32(x);
        for k in 0..6 {
            let a = self.alpha_q[k];
            let s1 = self.q_stages_z2[k];
            let s0 = self.q_stages[k];
            let y = flush_recursive_state_f32(a * curr_q + s1);
            self.q_stages_z2[k] = s0;
            self.q_stages[k] = flush_recursive_state_f32(a * y - curr_q);
            curr_q = y;
        }

        (curr_i, curr_q)
    }

    /// Single-sideband frequency shifting by `(cos(w*n), sin(w*n))`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_ssb(&mut self, x: f32, cos_w: f32, sin_w: f32, upper_sideband: bool) -> f32 {
        let (i_val, q_val) = self.step(x);
        if upper_sideband {
            i_val * cos_w - q_val * sin_w
        } else {
            i_val * cos_w + q_val * sin_w
        }
    }
}

/// Normative 6-section-per-branch `H1` Dual All-Pass Hilbert Lattice (`96 B` raw state + retained
/// prior outputs for history-consistent rate transitions, `ref[dsp.op-m.hilbert.lattice]`,
/// `ref[dsp.op-m.hilbert.h1-anchors]`, `ref[dsp.op-m.hilbert.rate-remap]`).
#[derive(Clone, Debug)]
pub struct HilbertH1Lattice6Sec {
    pub alpha_i: [f32; 6],
    pub alpha_q: [f32; 6],
    /// Transposed all-pass state `[s[n-1], s[n-2]]` per section.
    pub i_state: [[f32; 2]; 6],
    pub q_state: [[f32; 2]; 6],
    /// Retained section outputs `[y[n-1], y[n-2]]` for history-consistent rate remap (`ref[dsp.op-m.hilbert.rate-remap]`).
    pub i_prev_y: [[f32; 2]; 6],
    pub q_prev_y: [[f32; 2]; 6],
    pub q_z1: f32,
    pub(crate) stereo_alpha: [[f32; 4]; 6],
    pub(crate) stereo_state: [[[f32; 4]; 6]; 2],
    pub(crate) stereo_prev_y: [[[f32; 4]; 6]; 2],
    pub(crate) stereo_q_z1: [f32; 2],
    pub(crate) stereo_parity: usize,
}

impl Default for HilbertH1Lattice6Sec {
    fn default() -> Self {
        Self::new_at_rate(48_000.0).unwrap()
    }
}

impl HilbertH1Lattice6Sec {
    pub const fn new_48khz() -> Self {
        const AI: [f32; 6] = HilbertSsbLattice10th::ALPHA_I;
        const AQ: [f32; 6] = HilbertSsbLattice10th::ALPHA_Q;
        Self {
            alpha_i: AI,
            alpha_q: AQ,
            i_state: [[0.0; 2]; 6],
            q_state: [[0.0; 2]; 6],
            i_prev_y: [[0.0; 2]; 6],
            q_prev_y: [[0.0; 2]; 6],
            q_z1: 0.0,
            stereo_alpha: [
                [AI[0], AQ[0], AI[0], AQ[0]],
                [AI[1], AQ[1], AI[1], AQ[1]],
                [AI[2], AQ[2], AI[2], AQ[2]],
                [AI[3], AQ[3], AI[3], AQ[3]],
                [AI[4], AQ[4], AI[4], AQ[4]],
                [AI[5], AQ[5], AI[5], AQ[5]],
            ],
            stereo_state: [[[0.0; 4]; 6]; 2],
            stereo_prev_y: [[[0.0; 4]; 6]; 2],
            stereo_q_z1: [0.0; 2],
            stereo_parity: 0,
        }
    }

    pub fn reset(&mut self) {
        self.i_state = [[0.0; 2]; 6];
        self.q_state = [[0.0; 2]; 6];
        self.i_prev_y = [[0.0; 2]; 6];
        self.q_prev_y = [[0.0; 2]; 6];
        self.q_z1 = 0.0;
        self.stereo_state = [[[0.0; 4]; 6]; 2];
        self.stereo_prev_y = [[[0.0; 4]; 6]; 2];
        self.stereo_q_z1 = [0.0; 2];
        self.stereo_parity = 0;
    }

    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let bits = interpolate_hilbert_h1_profile_bits(rate_hz)?;
        let alpha_i = core::array::from_fn(|k| f32::from_bits(bits[k]));
        let alpha_q = core::array::from_fn(|k| f32::from_bits(bits[6 + k]));
        let stereo_alpha =
            core::array::from_fn(|k| [alpha_i[k], alpha_q[k], alpha_i[k], alpha_q[k]]);
        Some(Self {
            alpha_i,
            alpha_q,
            i_state: [[0.0; 2]; 6],
            q_state: [[0.0; 2]; 6],
            i_prev_y: [[0.0; 2]; 6],
            q_prev_y: [[0.0; 2]; 6],
            q_z1: 0.0,
            stereo_alpha,
            stereo_state: [[[0.0; 4]; 6]; 2],
            stereo_prev_y: [[[0.0; 4]; 6]; 2],
            stereo_q_z1: [0.0; 2],
            stereo_parity: 0,
        })
    }

    /// Initializes all 12 section states and retained prior outputs to the resting DC equilibrium
    /// for constant input `x_dc` (`y_k = (-1)^(k+1) * x_dc`, `s_k = (-1)^(k+1) * (1 + alpha_k) * x_dc`,
    /// `ref[dsp.op-m.hilbert.rate-remap]`).
    pub fn init_dc_equilibrium(&mut self, x_dc: f32) {
        let x_clean = flush_recursive_state_f32(x_dc);
        let mut vi = x_clean;
        let mut vq = x_clean;
        self.q_z1 = x_clean;
        self.stereo_q_z1 = [x_clean, x_clean];
        for k in 0..6 {
            let yi = -vi;
            let si = flush_recursive_state_f32(self.alpha_i[k] * yi - vi);
            self.i_state[k] = [si, si];
            self.i_prev_y[k] = [yi, yi];
            vi = yi;

            let yq = -vq;
            let sq = flush_recursive_state_f32(self.alpha_q[k] * yq - vq);
            self.q_state[k] = [sq, sq];
            self.q_prev_y[k] = [yq, yq];
            vq = yq;

            self.stereo_state[0][k] = [si, sq, si, sq];
            self.stereo_state[1][k] = [si, sq, si, sq];
            self.stereo_prev_y[0][k] = [yi, yq, yi, yq];
            self.stereo_prev_y[1][k] = [yi, yq, yi, yq];
        }
    }

    /// Updates the sample rate to `new_rate_hz` and remaps all mono and interleaved-stereo section
    /// states from the retained prior outputs $s'[n-i] = s[n-i] + (\beta - \alpha) y[n-i]$
    /// (`ref[dsp.op-m.hilbert.rate-remap]`).
    pub fn retune_sample_rate(&mut self, new_rate_hz: f64) -> Option<()> {
        let bits = interpolate_hilbert_h1_profile_bits(new_rate_hz)?;
        for k in 0..6 {
            let new_ai = f32::from_bits(bits[k]);
            let da_i = new_ai - self.alpha_i[k];
            self.i_state[k][0] =
                flush_recursive_state_f32(self.i_state[k][0] + da_i * self.i_prev_y[k][0]);
            self.i_state[k][1] =
                flush_recursive_state_f32(self.i_state[k][1] + da_i * self.i_prev_y[k][1]);
            self.alpha_i[k] = new_ai;

            let new_aq = f32::from_bits(bits[6 + k]);
            let da_q = new_aq - self.alpha_q[k];
            self.q_state[k][0] =
                flush_recursive_state_f32(self.q_state[k][0] + da_q * self.q_prev_y[k][0]);
            self.q_state[k][1] =
                flush_recursive_state_f32(self.q_state[k][1] + da_q * self.q_prev_y[k][1]);
            self.alpha_q[k] = new_aq;

            let da_stereo = [da_i, da_q, da_i, da_q];
            for p in 0..2 {
                for (lane, &da) in da_stereo.iter().enumerate() {
                    self.stereo_state[p][k][lane] = flush_recursive_state_f32(
                        self.stereo_state[p][k][lane] + da * self.stereo_prev_y[p][k][lane],
                    );
                }
            }
            self.stereo_alpha[k] = [new_ai, new_aq, new_ai, new_aq];
        }
        Some(())
    }

    /// Steps one sample through the 6-section `H1` analytic pair (`y[n] = alpha * v[n] + s[n-2]`,
    /// `s[n] = alpha * y[n] - v[n]`, `ref[dsp.op-m.hilbert.lattice]`).
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> (f32, f32) {
        let mut vi = flush_recursive_state_f32(x);
        let mut vq = flush_recursive_state_f32(self.q_z1);
        self.q_z1 = vi;
        for k in 0..6 {
            let ai = self.alpha_i[k];
            let aq = self.alpha_q[k];
            let i_s1 = flush_recursive_state_f32(self.i_state[k][1]);
            let q_s1 = flush_recursive_state_f32(self.q_state[k][1]);
            let i_s0 = flush_recursive_state_f32(self.i_state[k][0]);
            let q_s0 = flush_recursive_state_f32(self.q_state[k][0]);
            let yi = flush_recursive_state_f32(ai * vi + i_s1);
            let yq = flush_recursive_state_f32(aq * vq + q_s1);
            self.i_state[k][1] = i_s0;
            self.q_state[k][1] = q_s0;
            self.i_state[k][0] = flush_recursive_state_f32(ai * yi - vi);
            self.q_state[k][0] = flush_recursive_state_f32(aq * yq - vq);
            self.i_prev_y[k][1] = flush_recursive_state_f32(self.i_prev_y[k][0]);
            self.q_prev_y[k][1] = flush_recursive_state_f32(self.q_prev_y[k][0]);
            self.i_prev_y[k][0] = yi;
            self.q_prev_y[k][0] = yq;
            vi = yi;
            vq = yq;
        }
        (vi, vq)
    }

    /// Returns the most recently updated 4-lane stereo section-5 state `[s5_il, s5_ql, s5_ir, s5_qr]`.
    #[inline(always)]
    pub fn last_stereo_s5(&self) -> &[f32; 4] {
        &self.stereo_state[(self.stereo_parity ^ 1) & 1][5]
    }

    /// Steps a stereo pair (`self` = Left, `_right` = symmetry handle) of 6-section `H1` analytic
    /// pairs simultaneously using `B::Vec4` SIMD across `(vi_l, vq_l, vi_r, vq_r)` while recording
    /// retained section outputs for clickless [`Self::retune_sample_rate`] transitions
    /// (`ref[dsp.op-m.hilbert.lattice]`, `ref[dsp.op-m.hilbert.rate-remap]`).
    #[inline(always)]
    pub fn step_stereo_with_backend<B: MathBackend>(
        &mut self,
        _right: &mut Self,
        x_l: f32,
        x_r: f32,
    ) -> (f32, f32, f32, f32) {
        let p = self.stereo_parity & 1;
        self.stereo_parity = p ^ 1;
        let v_in = [x_l, self.stereo_q_z1[0], x_r, self.stereo_q_z1[1]];
        let mut v = flush_recursive_state_vec4::<B::Vec4>(B::Vec4::load(&v_in));
        let mut v_clean = [0.0f32; 4];
        v.store(&mut v_clean);
        self.stereo_q_z1 = [v_clean[0], v_clean[2]];
        let state_p = &mut self.stereo_state[p];
        let prev_y_p = &mut self.stereo_prev_y[p];
        for k in 0..6 {
            let a = B::Vec4::load(&self.stereo_alpha[k]);
            let s_prev = flush_recursive_state_vec4::<B::Vec4>(B::Vec4::load(&state_p[k]));
            let y = flush_recursive_state_vec4::<B::Vec4>(s_prev.fma(a, v));
            y.store(&mut prev_y_p[k]);
            let s_next = flush_recursive_state_vec4::<B::Vec4>(a.mul(y).sub(v));
            s_next.store(&mut state_p[k]);
            v = y;
        }
        let mut v_out = [0.0f32; 4];
        v.store(&mut v_out);
        (v_out[0], v_out[1], v_out[2], v_out[3])
    }

    /// Steps a stereo pair (`self` = Left, `right` = symmetry handle) of 6-section `H1` analytic
    /// pairs simultaneously via [`DefaultMathBackend`] (`ref[dsp.op-m.hilbert.lattice]`).
    #[inline(always)]
    pub fn step_stereo(&mut self, right: &mut Self, x_l: f32, x_r: f32) -> (f32, f32, f32, f32) {
        self.step_stereo_with_backend::<DefaultMathBackend>(right, x_l, x_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use crate::math::ScalarBackend;
    use crate::operators::{
        normalize_tracker_phasor_exponent_split, validate_and_eval_stage3_radial_gate,
    };
    use proptest::prelude::*;

    /// verify[dsp.op-m.hilbert.lattice]
    #[test]
    fn test_hilbert_ssb_lattice_10th() {
        let _audio = FtzDazGuard::new();
        let mut hilbert = HilbertSsbLattice10th::default();
        hilbert.reset();

        let mut energy_i = 0.0_f64;
        let mut energy_q = 0.0_f64;
        for n in 0..4096 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (i_out, q_out) = hilbert.step(x);
            energy_i += (i_out as f64) * (i_out as f64);
            energy_q += (q_out as f64) * (q_out as f64);
        }
        assert!((energy_i - 1.0).abs() < 1e-3);
        assert!((energy_q - 1.0).abs() < 1e-3);

        let mut h_up = HilbertSsbLattice10th::default();
        let mut h_dn = HilbertSsbLattice10th::default();
        for n in 0..16 {
            let x = ((n as f32) * 0.3 + 0.1).sin();
            let y_up = h_up.step_ssb(x, 0.0, 1.0, true);
            let y_dn = h_dn.step_ssb(x, 0.0, 1.0, false);
            assert!((y_up + y_dn).abs() < 1e-6);
        }

        let mut h_quad = HilbertSsbLattice10th::default();
        let omega = core::f32::consts::TAU * 3_000.0 / 48_000.0;
        let mut dot_iq = 0.0_f64;
        let mut min_env = f64::INFINITY;
        let mut max_env = 0.0_f64;
        for n in 0..1024 {
            let x = ((n as f32) * omega).sin();
            let (i_v, q_v) = h_quad.step(x);
            if n >= 512 {
                let env2 = (i_v as f64) * (i_v as f64) + (q_v as f64) * (q_v as f64);
                min_env = min_env.min(env2);
                max_env = max_env.max(env2);
                dot_iq += (i_v as f64) * (q_v as f64);
            }
        }
        assert!(
            (max_env - min_env) < 0.10 && (min_env - 1.0).abs() < 0.06,
            "Hilbert quadrature envelope not constant: min_env={min_env}, max_env={max_env}"
        );
        assert!(
            (dot_iq / 512.0).abs() < 0.03,
            "Hilbert I/Q not orthogonal: avg(I*Q)={}",
            dot_iq / 512.0
        );

        hilbert.reset();
        assert_eq!(hilbert.q_z1, 0.0);
    }

    /// verify[dsp.op-m.hilbert.rate-remap]
    #[test]
    fn test_hilbert_h1_rate_remap_and_stereo_backend_parity() {
        let _guard = FtzDazGuard::new();
        let mut h1 = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut dummy = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        h1.init_dc_equilibrium(1.0);
        let (i_before, q_before) = h1.step(1.0);
        assert!((i_before - 1.0).abs() < 1e-6 && (q_before - 1.0).abs() < 1e-6);
        let st_before = h1.step_stereo(&mut dummy, 1.0, 1.0);
        assert!((st_before.0 - 1.0).abs() < 1e-6 && (st_before.3 - 1.0).abs() < 1e-6);

        h1.retune_sample_rate(96_000.0).unwrap();
        let (i_after, q_after) = h1.step(1.0);
        assert!((i_after - 1.0).abs() < 1e-6 && (q_after - 1.0).abs() < 1e-6);
        for _ in 0..4 {
            let st_after = h1.step_stereo(&mut dummy, 1.0, 1.0);
            assert!(
                (st_after.0 - 1.0).abs() < 1e-6
                    && (st_after.1 - 1.0).abs() < 1e-6
                    && (st_after.2 - 1.0).abs() < 1e-6
                    && (st_after.3 - 1.0).abs() < 1e-6
            );
        }

        // Dynamic stereo waveform across 48 kHz -> 96 kHz retune matches scalar mono reference and ScalarBackend
        let mut h_simd = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut h_scal = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut h_mono_l = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut h_mono_r = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut dummy_a = HilbertH1Lattice6Sec::default();
        let mut dummy_b = HilbertH1Lattice6Sec::default();
        for n in 0..40 {
            if n == 19 {
                h_simd.retune_sample_rate(96_000.0).unwrap();
                h_scal.retune_sample_rate(96_000.0).unwrap();
                h_mono_l.retune_sample_rate(96_000.0).unwrap();
                h_mono_r.retune_sample_rate(96_000.0).unwrap();
            }
            let x_l = ((n as f32) * 0.21).sin() * 0.5;
            let x_r = ((n as f32) * 0.37).cos() * 0.4;
            let (sil, sql, sir, sqr) = h_simd.step_stereo(&mut dummy_a, x_l, x_r);
            let (scl, qcl, scr, qcr) =
                h_scal.step_stereo_with_backend::<ScalarBackend>(&mut dummy_b, x_l, x_r);
            let (mil, mql) = h_mono_l.step(x_l);
            let (mir, mqr) = h_mono_r.step(x_r);
            assert!((sil - mil).abs() < 1e-6 && (sql - mql).abs() < 1e-6);
            assert!((sir - mir).abs() < 1e-6 && (sqr - mqr).abs() < 1e-6);
            assert!((sil - scl).abs() < 1e-6 && (sql - qcl).abs() < 1e-6);
            assert!((sir - scr).abs() < 1e-6 && (sqr - qcr).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    fn equal_transposed_cells_can_require_different_history_consistent_retunes() {
        use core::hint::black_box;
        fn step(coefficient: f32, state: &mut [f32; 2], input: f32) -> f32 {
            let output = black_box(black_box(coefficient * input) + state[1]);
            state[1] = state[0];
            state[0] = black_box(black_box(coefficient * output) - input);
            output
        }

        let _guard = FtzDazGuard::new();
        let old = black_box(f32::from_bits(0x3d91_bee6));
        let new = black_box(f32::from_bits(0x3db0_73c5));
        let mut driven_state = [0.0; 2];
        let mut outputs = [0.0; 3];
        for (sample, output) in [1.0, 0.0, -old].into_iter().zip(&mut outputs) {
            *output = step(old, &mut driven_state, sample);
        }
        assert_eq!(driven_state.map(f32::to_bits), [0; 2]);
        assert_eq!(&outputs[1..], &[0.0, -1.0]);

        let mut empty_state = [0.0; 2];
        let change = black_box(new - old);
        driven_state[0] = black_box(driven_state[0] + black_box(change * outputs[2]));
        driven_state[1] = black_box(driven_state[1] + black_box(change * outputs[1]));
        assert_ne!(driven_state, empty_state);

        assert_eq!(step(new, &mut driven_state, 0.0), 0.0);
        assert_eq!(step(new, &mut empty_state, 0.0), 0.0);
        let driven_second = step(new, &mut driven_state, 0.0);
        let empty_second = step(new, &mut empty_state, 0.0);
        assert!((-0.015..-0.014).contains(&driven_second));
        assert_eq!(empty_second, 0.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_op_m_t1_envelope_and_stage3_radial_gate(
            i_val in -50.0_f32..50.0_f32,
            q_val in -50.0_f32..50.0_f32,
            r_low in 0.85_f32..0.95_f32,
            dr in 0.0_f32..0.049_f32,
            theta_10 in -2.0_f32..2.0_f32,
        ) {
            if let Some((u_re, u_im, env)) = normalize_tracker_phasor_exponent_split(i_val, q_val) {
                prop_assert!(env >= 0.0 && env.is_finite());
                prop_assert!(u_re * u_re + u_im * u_im <= 1.001);
            }

            let (g_k, r_k, b_eff) =
                validate_and_eval_stage3_radial_gate(r_low, dr, 0.3, 1.0, theta_10).unwrap();
            prop_assert!((0.0..=1.0).contains(&g_k) && b_eff.is_finite());
            prop_assert!(r_k >= r_low && r_k <= r_low + dr && r_k < 1.0);
        }
    }

    #[test]
    fn test_hilbert_ssb_matches_shared_hilbert_h1_kernel_zero_ulp() {
        let _guard = FtzDazGuard::new();
        let mut lattice = HilbertSsbLattice10th::default();
        let mut kernel = dnam_kernels::HilbertH1Kernel::new();
        for n in 0..128 {
            let x = ((n as f32) * 0.31).sin() * 0.7;
            let (li, lq) = lattice.step(x);
            let (ki, kq) = kernel.step(x);
            assert_eq!(li.to_bits(), ki.to_bits());
            assert_eq!(lq.to_bits(), kq.to_bits());
        }
    }

    #[test]
    fn test_hilbert_h1_step_and_stereo_scrub_nonfinite_and_poisoned_state() {
        let _guard = FtzDazGuard::new();
        let poison_h1 = |h: &mut HilbertH1Lattice6Sec| {
            h.q_z1 = f32::NAN;
            h.i_state = [[f32::NAN, f32::INFINITY]; 6];
            h.q_state = [[f32::NEG_INFINITY, f32::NAN]; 6];
            h.i_prev_y = [[f32::INFINITY, f32::NAN]; 6];
            h.q_prev_y = [[f32::NAN, f32::NEG_INFINITY]; 6];
            h.stereo_q_z1 = [f32::NAN, f32::INFINITY];
            h.stereo_state = [[[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]; 6]; 2];
            h.stereo_prev_y = [[[f32::INFINITY, f32::NAN, f32::NAN, f32::NEG_INFINITY]; 6]; 2];
        };

        for bad_x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut mono = HilbertH1Lattice6Sec::new_48khz();
            poison_h1(&mut mono);
            let (yi, yq) = mono.step(bad_x);
            assert_eq!(yi.to_bits(), 0);
            assert_eq!(yq.to_bits(), 0);
            assert!(mono.q_z1.is_finite());
            assert!(mono.i_state.iter().flatten().all(|v| v.is_finite()));
            assert!(mono.q_state.iter().flatten().all(|v| v.is_finite()));
            assert!(mono.i_prev_y.iter().flatten().all(|v| v.is_finite()));
            assert!(mono.q_prev_y.iter().flatten().all(|v| v.is_finite()));

            let (ri, rq) = mono.step(0.5);
            assert!(ri.is_finite() && rq.is_finite());
            assert!(ri.abs() > 0.0);

            let mut st_scal = HilbertH1Lattice6Sec::new_48khz();
            let mut st_def = HilbertH1Lattice6Sec::new_48khz();
            let mut dummy_a = HilbertH1Lattice6Sec::new_48khz();
            let mut dummy_b = HilbertH1Lattice6Sec::new_48khz();
            poison_h1(&mut st_scal);
            poison_h1(&mut st_def);

            let os0 =
                st_scal.step_stereo_with_backend::<ScalarBackend>(&mut dummy_a, bad_x, -bad_x);
            let od0 =
                st_def.step_stereo_with_backend::<DefaultMathBackend>(&mut dummy_b, bad_x, -bad_x);
            assert_eq!([os0.0, os0.1, os0.2, os0.3].map(f32::to_bits), [0; 4]);
            assert_eq!([od0.0, od0.1, od0.2, od0.3].map(f32::to_bits), [0; 4]);

            // Step the second parity bank immediately with zero to verify both parity banks scrub.
            let os1 = st_scal.step_stereo_with_backend::<ScalarBackend>(&mut dummy_a, 0.0, 0.0);
            let od1 = st_def.step_stereo_with_backend::<DefaultMathBackend>(&mut dummy_b, 0.0, 0.0);
            assert_eq!([os1.0, os1.1, os1.2, os1.3].map(f32::to_bits), [0; 4]);
            assert_eq!([od1.0, od1.1, od1.2, od1.3].map(f32::to_bits), [0; 4]);
            for st in [&st_scal, &st_def] {
                assert!(st.stereo_q_z1.iter().all(|v| v.is_finite()));
                assert!(
                    st.stereo_state
                        .iter()
                        .flatten()
                        .flatten()
                        .all(|v| v.is_finite())
                );
                assert!(
                    st.stereo_prev_y
                        .iter()
                        .flatten()
                        .flatten()
                        .all(|v| v.is_finite())
                );
            }

            let rec_s = st_scal.step_stereo_with_backend::<ScalarBackend>(&mut dummy_a, 0.5, -0.25);
            let rec_d =
                st_def.step_stereo_with_backend::<DefaultMathBackend>(&mut dummy_b, 0.5, -0.25);
            assert!(rec_s.0.is_finite() && rec_s.2.is_finite());
            assert!(rec_d.0.is_finite() && rec_d.2.is_finite());
            assert!(rec_d.0.abs() > 0.0 && rec_d.2.abs() > 0.0);
        }
    }
}
