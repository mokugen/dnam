//! Pack, unpack, coordinate scaling, and readout-mask helpers on [`ThetaLayout`].

use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::super::model::SonaModelF64;
use super::super::eval::Pass2ParamGrads;
use super::ThetaLayout;

impl ThetaLayout {
    /// Returns a boolean mask marking coordinates directly coupled to linear VarPro readouts
    /// (`ro_pole_48k`, `kautz.*`, and joint readout coordinates when present).
    #[must_use]
    pub fn readout_coupled_mask(&self) -> Vec<bool> {
        self.entries
            .iter()
            .map(|e| e.kind.is_readout_coupled())
            .collect()
    }

    /// Returns per-coordinate `ScaledCoordSpace` scales (`num_frames` for `zoh_hold_depth` to cancel
    /// secular `2 * pi * n` derivative growth, `1.0` elsewhere).
    #[must_use]
    pub fn coord_scales(&self, num_frames: usize) -> Vec<f64> {
        let mut scales = vec![1.0_f64; self.entries.len()];
        if let Some(idx) = self.zoh_hold_depth_idx
            && idx < scales.len()
        {
            scales[idx] = num_frames.max(1) as f64;
        }
        scales
    }

    /// Packs the active Pass-2 subspace from `m` alongside matching coordinate bijections.
    pub fn pack(&self, m: &SonaModelF64) -> (Vec<f64>, Vec<ParamBoundSpec>) {
        let mut x = Vec::with_capacity(self.entries.len());
        let mut b = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            x.push(entry.read(m));
            b.push(entry.bound);
        }
        (x, b)
    }

    /// Clamps and unpacks `cand` into `m`, snapping `lo == 0.0` `ComplementLog` values `< 1e-7` to
    /// `0.0`, preserving inactive Kautz `0.0` sentinels and discrete polarity/phase fields, and
    /// enforcing Table-4 stability bounds on exit.
    pub fn unpack(&self, cand: &[f64], m: &mut SonaModelF64) {
        for (i, entry) in self.entries.iter().enumerate() {
            entry.write(m, cand[i], &self.kautz_active_mask);
        }
        m.clamp_all_stability_bounds(true);
    }

    /// Writes `grads` into `out` in the exact coordinate order produced by [`Self::pack`],
    /// applying coordinate-adapter chain-rule factors (`dg/dr` for `CoreSvfG`/`SenseSvfG`) from `model`.
    pub fn write_grad(&self, model: &SonaModelF64, grads: &Pass2ParamGrads, out: &mut [f64]) {
        for (i, entry) in self.entries.iter().enumerate() {
            out[i] = entry.read_grad(model, grads, &self.kautz_active_mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ParamKind;
    use super::*;
    use crate::linalg::coords::ScaledCoordSpace;
    use crate::linalg::lbfgs::diag_precond_f32_from_jac;

    #[test]
    fn test_theta_codec_pack_unpack_scales_mask_and_grad() {
        let model = SonaModelF64::new(48_000.0);
        let layout = ThetaLayout::for_tier(0, true, 48_000.0);
        let (x, bounds) = layout.pack(&model);
        assert_eq!(x.len(), bounds.len());
        assert_eq!(layout.readout_coupled_mask().len(), x.len());
        let scales = layout.coord_scales(1_024);
        assert_eq!(scales.len(), x.len());
        assert_eq!(
            scales[layout.zoh_hold_depth_idx.expect("zoh idx in tier 0")],
            1_024.0
        );

        let mut rt = model.clone();
        layout.unpack(&x, &mut rt);
        let grads = Pass2ParamGrads::with_dims(layout.q_opt, layout.k_opt);
        let mut grad_out = vec![1.0_f64; x.len()];
        layout.write_grad(&model, &grads, &mut grad_out);
        assert!(grad_out.iter().all(|&g| g == 0.0));

        let (space, u) =
            ScaledCoordSpace::new("ThetaLayout::diag_precond_f32", &x, &bounds, Some(&scales))
                .expect("scaled coord space for tier 0 layout");
        let mut x_tmp = vec![0.0_f64; x.len()];
        let mut jac = vec![1.0_f64; x.len()];
        space.decode(&u, &mut x_tmp, &mut jac);
        let dp = diag_precond_f32_from_jac(&jac);
        assert_eq!(dp.len(), x.len());
        assert!(dp.iter().all(|&v| v.is_finite() && v > 0.0 && v <= 1.0));
    }

    #[test]
    fn test_u_space_roundtrip_and_conditioned_coord_fd_jacobians() {
        let mut model = SonaModelF64::new(48_000.0);
        model.core_quads[0].svf_g = [0.18, 0.35, 0.72, 1.45];
        model.core_quads[0].svf_k = [0.45, 0.75, 1.15, 2.2];
        model.core_quads[0].lru_alpha_atk = [0.05, 0.25, 0.65, 0.92];
        model.sense_svf_g = [0.12, 0.28, 0.64, 1.8];
        model.sense_svf_k = [0.35, 0.65, 1.25, 2.5];
        model.sense_lru_atk = [0.08, 0.32, 0.58, 0.88];

        let layout = ThetaLayout::for_tier(0, true, 128.0);
        let (x0, bounds) = layout.pack(&model);
        let scales = layout.coord_scales(512);
        let (space, u0) =
            ScaledCoordSpace::new("u_space_roundtrip", &x0, &bounds, Some(&scales)).unwrap();
        let mut x_dec = vec![0.0_f64; x0.len()];
        let mut jac = vec![0.0_f64; x0.len()];
        space.decode(&u0, &mut x_dec, &mut jac);
        let mut rt = model.clone();
        layout.unpack(&x_dec, &mut rt);
        for l in 0..4 {
            assert!((rt.core_quads[0].svf_g[l] - model.core_quads[0].svf_g[l]).abs() < 1.0e-8);
            assert!((rt.core_quads[0].svf_k[l] - model.core_quads[0].svf_k[l]).abs() < 1.0e-8);
            assert!(
                (rt.core_quads[0].lru_alpha_atk[l] - model.core_quads[0].lru_alpha_atk[l]).abs()
                    < 1.0e-8
            );
            assert!((rt.sense_svf_g[l] - model.sense_svf_g[l]).abs() < 1.0e-8);
            assert!((rt.sense_svf_k[l] - model.sense_svf_k[l]).abs() < 1.0e-8);
            assert!((rt.sense_lru_atk[l] - model.sense_lru_atk[l]).abs() < 1.0e-8);
        }

        // Verify analytical chain rule through ScaledCoordSpace + ThetaEntry::read_grad vs FD in u-space.
        let kinds = layout.entries_kinds();
        let target_indices: Vec<usize> = [
            kinds
                .iter()
                .position(|k| matches!(k, ParamKind::CoreSvfG(0, 1)))
                .unwrap(),
            kinds
                .iter()
                .position(|k| matches!(k, ParamKind::CoreSvfK(0, 1)))
                .unwrap(),
            kinds
                .iter()
                .position(|k| matches!(k, ParamKind::CoreLruAtk(0, 1)))
                .unwrap(),
            kinds
                .iter()
                .position(|k| matches!(k, ParamKind::SenseSvfG(2)))
                .unwrap(),
            kinds
                .iter()
                .position(|k| matches!(k, ParamKind::SenseSvfK(2)))
                .unwrap(),
            kinds
                .iter()
                .position(|k| matches!(k, ParamKind::SenseLruAtk(2)))
                .unwrap(),
        ]
        .to_vec();

        let mut grads = Pass2ParamGrads::with_dims(layout.q_opt, layout.k_opt);
        grads.core_quads[0].svf_g[1] = 1.7;
        grads.core_quads[0].svf_k[1] = -0.9;
        grads.core_quads[0].lru_alpha_atk[1] = 2.3;
        grads.sensory.d_svf_g[2] = -1.4;
        grads.sensory.d_svf_k[2] = 0.8;
        grads.sensory.d_lru_atk[2] = -1.9;
        let mut gx = vec![0.0_f64; layout.dim()];
        layout.write_grad(&rt, &grads, &mut gx);
        let mut gu = vec![0.0_f64; layout.dim()];
        space.pullback_grad(&gx, &jac, &mut gu);

        let eval_linear_model = |u_vec: &[f64]| -> f64 {
            let mut xv = vec![0.0_f64; layout.dim()];
            let mut jv = vec![0.0_f64; layout.dim()];
            space.decode(u_vec, &mut xv, &mut jv);
            let mut m = model.clone();
            layout.unpack(&xv, &mut m);
            1.7 * m.core_quads[0].svf_g[1] - 0.9 * m.core_quads[0].svf_k[1]
                + 2.3 * m.core_quads[0].lru_alpha_atk[1]
                - 1.4 * m.sense_svf_g[2]
                + 0.8 * m.sense_svf_k[2]
                - 1.9 * m.sense_lru_atk[2]
        };

        let eps = 1.0e-6_f64;
        for idx in target_indices {
            let (mut up, mut um) = (u0.clone(), u0.clone());
            up[idx] += eps;
            um[idx] -= eps;
            let fd = (eval_linear_model(&up) - eval_linear_model(&um)) / (2.0 * eps);
            let ana = gu[idx];
            assert!(
                (ana - fd).abs() < 1.0e-8 * ana.abs().max(fd.abs()).max(1.0),
                "u-space codec FD mismatch idx={idx}: ana={ana} fd={fd}"
            );
        }
    }
}
