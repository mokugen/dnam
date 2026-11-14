//! SIMD-accelerated `uplo='U'` Gram tile contraction & degree-7 Bézout crossover operator.

use dnam_engine::math::trainer::TrainerMathBackend;

use crate::solver::uplo_u_index;

/// Hardware-abstracted packed `uplo='U'` Gram matrix & Bézout 3-band contraction operator.
pub trait GramCholeskyBackend: TrainerMathBackend {
    /// Assembles packed upper-triangular `gram` (`uplo='U'`) and cross-correlation `rhs`
    /// from column vectors `phi_w` using `4x4` tiled SIMD blocks, returning the diagonal sum.
    fn assemble_uplo_gram_and_rhs(
        phi_w: &[Vec<f64>],
        y_w: &[f64],
        gram: &mut [f64],
        rhs: &mut [[f64; 2]],
    ) -> f64;

    /// Contracts an off-diagonal `[4 x 4]` `f64` micro-tile (`0-ULP` IEEE parity).
    fn contract_tile_4x4_ieee(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4];

    /// Contracts two paired off-diagonal `[4 x 4]` `f64` micro-tiles (`[8 x 4]`, `0-ULP` IEEE parity).
    fn contract_tile_8x4_ieee(
        f_i0: &[[f64; 4]],
        f_i1: &[[f64; 4]],
        wf_j: &[[f64; 4]],
    ) -> ([[f64; 4]; 4], [[f64; 4]; 4]);

    /// Advances a 7-stage cascaded 1-pole Bézout low-pass state across `dim` feature channels (`0-ULP`).
    fn advance_bezout7_lp_dim(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    );

    /// Evaluates degree-7 Binomial/Bézout (`LF`, `MF`, `HF`) bands across `dim` feature channels (`0-ULP`).
    fn readout_bezout7_bands_dim(
        fb: &[f32; 256],
        dim: usize,
        u1: &[[f64; 256]; 7],
        u2: &[[f64; 256]; 7],
        out_lf: &mut [f32; 256],
        out_mf: &mut [f32; 256],
        out_hf: &mut [f32; 256],
    );
}

impl<M: TrainerMathBackend> GramCholeskyBackend for M {
    fn assemble_uplo_gram_and_rhs(
        phi_w: &[Vec<f64>],
        y_w: &[f64],
        gram: &mut [f64],
        rhs: &mut [[f64; 2]],
    ) -> f64 {
        let p_dim = phi_w.len();
        let blocks = p_dim / 4;
        let rem_start = blocks * 4;
        let mut diag_sum = 0.0_f64;

        for bj in 0..blocks {
            let j0 = bj * 4;
            let cols_j = [
                phi_w[j0].as_slice(),
                phi_w[j0 + 1].as_slice(),
                phi_w[j0 + 2].as_slice(),
                phi_w[j0 + 3].as_slice(),
            ];
            for bi in 0..bj {
                let i0 = bi * 4;
                let cols_i = [
                    phi_w[i0].as_slice(),
                    phi_w[i0 + 1].as_slice(),
                    phi_w[i0 + 2].as_slice(),
                    phi_w[i0 + 3].as_slice(),
                ];
                let tile = Self::gram_tile_4x4_f64(cols_i, cols_j);
                for (lj, cj) in (j0..j0 + 4).enumerate() {
                    let base = uplo_u_index(i0, cj);
                    for (li, &row_vals) in tile.iter().enumerate() {
                        gram[base + li] = row_vals[lj];
                    }
                }
            }
            let sym = Self::gram_sym_block4_f64(cols_j);
            let mut k = 0_usize;
            for lj in 0..4 {
                let cj = j0 + lj;
                let base = uplo_u_index(j0, cj);
                for li in 0..=lj {
                    gram[base + li] = sym[k];
                    if li == lj {
                        diag_sum += sym[k];
                    }
                    k += 1;
                }
            }
        }

        for j in rem_start..p_dim {
            let base = uplo_u_index(0, j);
            for i in 0..=j {
                gram[base + i] = Self::dot_f64(&phi_w[i], &phi_w[j]);
            }
            diag_sum += gram[base + j];
        }

        for (j, col) in phi_w.iter().enumerate() {
            rhs[j][0] = Self::dot_f64(col, y_w);
        }
        diag_sum
    }

    #[inline(always)]
    fn contract_tile_4x4_ieee(f_i: &[[f64; 4]], wf_j: &[[f64; 4]]) -> [[f64; 4]; 4] {
        Self::contract_tile_4x4_ieee_f64(f_i, wf_j)
    }

    #[inline(always)]
    fn contract_tile_8x4_ieee(
        f_i0: &[[f64; 4]],
        f_i1: &[[f64; 4]],
        wf_j: &[[f64; 4]],
    ) -> ([[f64; 4]; 4], [[f64; 4]; 4]) {
        Self::contract_tile_8x4_ieee_f64(f_i0, f_i1, wf_j)
    }

    #[inline(always)]
    fn advance_bezout7_lp_dim(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    ) {
        Self::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
    }

    #[inline(always)]
    fn readout_bezout7_bands_dim(
        fb: &[f32; 256],
        dim: usize,
        u1: &[[f64; 256]; 7],
        u2: &[[f64; 256]; 7],
        out_lf: &mut [f32; 256],
        out_mf: &mut [f32; 256],
        out_hf: &mut [f32; 256],
    ) {
        Self::readout_bezout7_bands_dim_f64(fb, dim, u1, u2, out_lf, out_mf, out_hf);
    }
}
