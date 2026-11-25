//! Pack, unpack, coordinate scaling, and readout-mask helpers on [`ThetaLayout`].

use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::super::model::SonaModelF64;
use super::super::eval::Pass2ParamGrads;
use super::ThetaLayout;

impl ThetaLayout {
    /// Returns a boolean mask marking coordinates directly coupled to linear VarPro readouts
    /// (`ro_pole_48k`, `kautz.*`, and joint readout coordinates when present).
    #[must_use]
    pub(in super::super) fn readout_coupled_mask(&self) -> Vec<bool> {
        self.entries
            .iter()
            .map(|e| e.kind.is_readout_coupled())
            .collect()
    }

    /// Returns per-coordinate `ScaledCoordSpace` scales (`num_frames` for `zoh_hold_depth` to cancel
    /// secular `2 * pi * n` derivative growth, `1.0` elsewhere).
    #[must_use]
    pub(in super::super) fn coord_scales(&self, num_frames: usize) -> Vec<f64> {
        let mut scales = vec![1.0_f64; self.entries.len()];
        if let Some(idx) = self.zoh_hold_depth_idx
            && idx < scales.len()
        {
            scales[idx] = num_frames.max(1) as f64;
        }
        scales
    }

    /// Packs the active Pass-2 subspace from `m` alongside matching coordinate bijections.
    pub(in super::super) fn pack(&self, m: &SonaModelF64) -> (Vec<f64>, Vec<ParamBoundSpec>) {
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
    pub(in super::super) fn unpack(&self, cand: &[f64], m: &mut SonaModelF64) {
        for (i, entry) in self.entries.iter().enumerate() {
            entry.write(m, cand[i], &self.kautz_active_mask);
        }
        m.clamp_all_stability_bounds(true);
    }

    /// Writes `grads` into `out` in the exact coordinate order produced by [`Self::pack`].
    pub(in super::super) fn write_grad(&self, grads: &Pass2ParamGrads, out: &mut [f64]) {
        for (i, entry) in self.entries.iter().enumerate() {
            out[i] = entry.read_grad(grads, &self.kautz_active_mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        layout.write_grad(&grads, &mut grad_out);
        assert!(grad_out.iter().all(|&g| g == 0.0));
    }
}
