//! Cold Stage II Matryoshka GRU Block Parameter Struct, Bounds Clamping, and Continuous-Time ZOH Scaling.

use crate::container::ZohRescaledParams;
use crate::math::newton6_reciprocal_f64;

use super::block::{MAX_BLOCK_LANES, NUM_IN_FEATS_B0};
use super::math_pade::{TANH_PADE_D1_F64, TANH_PADE_D2_F64, TANH_PADE_N1_F64, TANH_PADE_N2_F64};

#[inline]
pub(crate) fn tanh_pade5_f64(x: f64) -> f64 {
    if x <= -4.0 {
        return -1.0;
    }
    if x >= 4.0 {
        return 1.0;
    }
    let s = x * x;
    let p = x * (1.0 + s * (TANH_PADE_N1_F64 + s * TANH_PADE_N2_F64));
    let q = 1.0 + s * (TANH_PADE_D1_F64 + s * TANH_PADE_D2_F64);
    (p * newton6_reciprocal_f64(q)).clamp(-1.0, 1.0)
}

#[inline]
pub(crate) fn sigmoid_pade5_f64(x: f64) -> f64 {
    (0.5 + 0.5 * tanh_pade5_f64(0.5 * x)).clamp(0.0, 1.0)
}

#[inline]
pub(crate) fn inv_tanh_pade5_pos_f64(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    if y >= 1.0 {
        return 4.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 4.0_f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if tanh_pade5_f64(mid) < y {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[inline]
pub(crate) fn inv_sigmoid_pade5_f64(p: f64) -> f64 {
    if p <= 0.5 {
        -2.0 * inv_tanh_pade5_pos_f64((1.0 - 2.0 * p).clamp(0.0, 1.0))
    } else {
        2.0 * inv_tanh_pade5_pos_f64((2.0 * p - 1.0).clamp(0.0, 1.0))
    }
}

/// Cold representation of one Matryoshka Stage II GRU block (`12` lanes/block max, lower-triangular).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3GruBlockCold {
    /// Update gate input weights `[feat_idx][dst_lane]`.
    pub w_in_z: [[f32; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    /// Reset gate input weights `[feat_idx][dst_lane]`.
    pub w_in_r: [[f32; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    /// Candidate input weights `[feat_idx][dst_lane]`.
    pub w_in_n: [[f32; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
    /// Inter-block update weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_prev_z: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Inter-block reset weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_prev_r: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Inter-block candidate weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_prev_n: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Recurrent update weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_rec_z: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Recurrent reset weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_rec_r: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Recurrent candidate weights `[dst_lane][src_lane]` (strictly block-lower-triangular).
    pub w_rec_n: [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    /// Update gate biases (`b_z in [-8.0, 8.2]`).
    pub b_z: [f32; MAX_BLOCK_LANES],
    /// Reset gate biases (`b_r in [-8.0, 8.2]`).
    pub b_r: [f32; MAX_BLOCK_LANES],
    /// Candidate equilibrium-cancelled biases (`b_n in [-3.8, 3.8]`).
    pub b_n: [f32; MAX_BLOCK_LANES],
    /// Multiplicative zero-DC power-supply headroom sag (`k_sag in [0.0, 8.0]`).
    pub k_sag: [f32; MAX_BLOCK_LANES],
    /// Zero-DC even-order triode square-law curvature (`k_quad in [-0.45, 0.45]`).
    pub k_quad: [f32; MAX_BLOCK_LANES],
}

impl Default for V3GruBlockCold {
    fn default() -> Self {
        Self {
            w_in_z: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_in_r: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_in_n: [[0.0; MAX_BLOCK_LANES]; NUM_IN_FEATS_B0],
            w_prev_z: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_prev_r: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_prev_n: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_z: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_r: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            w_rec_n: [[0.0; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
            b_z: [0.0; MAX_BLOCK_LANES],
            b_r: [1.5; MAX_BLOCK_LANES],
            b_n: [0.0; MAX_BLOCK_LANES],
            k_sag: [0.0; MAX_BLOCK_LANES],
            k_quad: [0.0; MAX_BLOCK_LANES],
        }
    }
}

/// Historical cold block type alias for module ergonomics.
pub type V3CascadeBlockCold = V3GruBlockCold;

impl V3GruBlockCold {
    /// Clamps biases, sag depth, and enforces strict block-lower-triangular sparsity (`src_g <= dst_g`).
    pub fn clamp_in_place(&mut self) {
        for dst in 0..MAX_BLOCK_LANES {
            self.b_z[dst] = self.b_z[dst].clamp(-8.0, 8.2);
            self.b_r[dst] = self.b_r[dst].clamp(-8.0, 8.2);
            self.b_n[dst] = self.b_n[dst].clamp(-3.8, 3.8);
            self.k_sag[dst] = self.k_sag[dst].clamp(0.0, 8.0);
            self.k_quad[dst] = self.k_quad[dst].clamp(-0.45, 0.45);
            let dst_g = dst >> 2;
            for src in 0..MAX_BLOCK_LANES {
                if (src >> 2) > dst_g {
                    self.w_prev_z[dst][src] = 0.0;
                    self.w_prev_r[dst][src] = 0.0;
                    self.w_prev_n[dst][src] = 0.0;
                    self.w_rec_z[dst][src] = 0.0;
                    self.w_rec_r[dst][src] = 0.0;
                    self.w_rec_n[dst][src] = 0.0;
                }
            }
        }
    }

    /// Rescales `2x` (`96 kHz`) normalized Padé `[5/4]` update-gate (`z`), standard reset-gate (`r`),
    /// and thresholded diode-comparator (`w_r_max > 6.0`) conduction rates to `1x` (`48 kHz`).
    pub fn rescale_2x_to_1x(&mut self) {
        self.clamp_in_place();
        for dst in 0..MAX_BLOCK_LANES {
            let bz_c = f64::from(self.b_z[dst]).clamp(-7.90, 7.90);
            let z2x = sigmoid_pade5_f64(bz_c);
            let mult = 2.0 - z2x;
            let z1x = (z2x * mult).clamp(1.0e-6, 0.9997);
            self.b_z[dst] = (inv_sigmoid_pade5_f64(z1x) as f32).clamp(-7.90, 7.90);
            let z_slope = (2.0 * newton6_reciprocal_f64(mult)) as f32;
            for src in 0..NUM_IN_FEATS_B0 {
                self.w_in_z[src][dst] *= z_slope;
            }
            for src in 0..MAX_BLOCK_LANES {
                self.w_prev_z[dst][src] *= z_slope;
                self.w_rec_z[dst][src] *= z_slope;
            }

            let mut w_r_max = 0.0_f32;
            for src in 0..NUM_IN_FEATS_B0 {
                w_r_max = w_r_max.max(self.w_in_r[src][dst].abs());
            }
            for src in 0..MAX_BLOCK_LANES {
                w_r_max = w_r_max
                    .max(self.w_prev_r[dst][src].abs())
                    .max(self.w_rec_r[dst][src].abs());
            }

            if w_r_max > 6.0 {
                if self.b_n[dst].abs() > 2.0 {
                    let ab = f64::from(self.b_n[dst].abs()).clamp(2.0, 3.8);
                    let gap_2x = (1.0 - tanh_pade5_f64(ab)).max(1.0e-9);
                    let gap_1x = (mult * gap_2x).clamp(1.0e-6, 0.5);
                    let b_n_mag = inv_tanh_pade5_pos_f64(1.0 - gap_1x);
                    self.b_n[dst] = (self.b_n[dst].signum() * (b_n_mag as f32)).clamp(-3.8, 3.8);
                } else {
                    let m_f32 = mult as f32;
                    for src in 0..NUM_IN_FEATS_B0 {
                        self.w_in_n[src][dst] *= m_f32;
                    }
                    for src in 0..MAX_BLOCK_LANES {
                        self.w_prev_n[dst][src] *= m_f32;
                        self.w_rec_n[dst][src] *= m_f32;
                    }
                }
            } else {
                let br_c = f64::from(self.b_r[dst]).clamp(-7.90, 7.90);
                let r2x = sigmoid_pade5_f64(br_c);
                let r1x = (r2x * mult).clamp(1.0e-4, 0.995);
                self.b_r[dst] = (inv_sigmoid_pade5_f64(r1x) as f32).clamp(-7.90, 7.90);
                let r_slope =
                    (((1.0 - r2x) * newton6_reciprocal_f64(1.0 - r1x)).clamp(0.5, 2.5)) as f32;
                for src in 0..NUM_IN_FEATS_B0 {
                    self.w_in_r[src][dst] *= r_slope;
                }
                for src in 0..MAX_BLOCK_LANES {
                    self.w_prev_r[dst][src] *= r_slope;
                    self.w_rec_r[dst][src] *= r_slope;
                }
            }
        }
        self.clamp_in_place();
    }

    /// Rescales update-gate retention rates $(1 - z_{\text{eff}}) = (1 - z_{\text{base}})^{\rho}$
    /// for host rate transitions (`ref[dsp.zoh.stage2-op-n]`).
    pub fn rescale_with_zoh(&mut self, zoh: &ZohRescaledParams) {
        self.clamp_in_place();
        if (zoh.rho_sr - 1.0).abs() <= 1.0e-7 {
            return;
        }
        for dst in 0..MAX_BLOCK_LANES {
            let bz_c = f64::from(self.b_z[dst]).clamp(-7.90, 7.90);
            let z_base = sigmoid_pade5_f64(bz_c);
            let one_minus_z_eff =
                crate::schema::cold_pow_f64((1.0 - z_base).clamp(1.0e-6, 0.99999), zoh.rho_sr);
            let z_eff = (1.0 - one_minus_z_eff).clamp(1.0e-6, 0.9997);
            self.b_z[dst] = (inv_sigmoid_pade5_f64(z_eff) as f32).clamp(-7.90, 7.90);
        }
        self.clamp_in_place();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rescale_2x_to_1x_and_zoh_preserves_lti_poles_and_diode_comparator() {
        let mut blk = V3GruBlockCold::default();
        blk.b_z[0] = -6.57;
        blk.b_r[0] = -6.4;
        blk.w_in_r[0][0] = 64.0;
        blk.w_in_n[0][0] = 0.00892;
        blk.b_n[0] = 0.0;

        blk.b_z[1] = -6.57;
        blk.b_r[1] = -6.4;
        blk.w_in_r[0][1] = 64.0;
        blk.w_in_n[0][1] = 32.0;
        blk.b_n[1] = 3.383;

        blk.b_z[2] = -2.0;
        blk.b_r[2] = -2.0;
        blk.w_in_r[0][2] = 0.5;

        let orig = blk;
        blk.rescale_2x_to_1x();
        assert!((blk.b_z[2] - blk.b_r[2]).abs() < 1.0e-5);
        assert_eq!(blk.b_r[0], orig.b_r[0]);
        assert_eq!(blk.b_r[1], orig.b_r[1]);
        assert_eq!(blk.w_in_r[0][0], orig.w_in_r[0][0]);
        assert_eq!(blk.w_in_r[0][1], orig.w_in_r[0][1]);

        let u = 0.50_f64;
        let step_lane = |b: &V3GruBlockCold, lane: usize, h_prev: f64| -> f64 {
            let z = sigmoid_pade5_f64(f64::from(b.b_z[lane]));
            let r = sigmoid_pade5_f64(f64::from(b.b_r[lane]) + f64::from(b.w_in_r[0][lane]) * u);
            let n = tanh_pade5_f64(f64::from(b.b_n[lane]) + f64::from(b.w_in_n[0][lane]) * u)
                - tanh_pade5_f64(f64::from(b.b_n[lane]));
            (1.0 - z) * h_prev + r * n
        };
        for lane in 0..2 {
            let h2_1 = step_lane(&orig, lane, 0.0);
            let h2_2 = step_lane(&orig, lane, h2_1);
            let h1_1 = step_lane(&blk, lane, 0.0);
            let rel = (h1_1 - h2_2).abs() / h2_2.abs();
            assert!(
                rel < 0.01,
                "lane {lane}: h2_2={h2_2} vs h1_1={h1_1} rel={rel}"
            );
        }

        let zoh_96 = ZohRescaledParams::compute(48_000, 96_000, 0.9993, 0.995, 64.0, 128).unwrap();
        let mut zoh_blk = orig;
        zoh_blk.rescale_with_zoh(&zoh_96);
        assert!(zoh_blk.b_z[2] < orig.b_z[2]);
    }
}
