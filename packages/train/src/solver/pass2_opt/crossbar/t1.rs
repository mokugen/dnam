//! Pure `binary64` 2-stage Riemann `T1` sub-harmonic tracker mirror (`T1SubHarmonicF64`) and
//! exact `os2x`-aware radial/angular pullback into `1x` stereo `(I, Q)` gradients
//! (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`,
//! `ref[train.pipeline.backbone-trajectory-opt]`).

use super::CrossbarForwardTrace;

/// Silence squelch squared-magnitude threshold (`r^2 < 1e-12`, matching `T1SubHarmonicKernel`).
const T1_SILENCE_SQUELCH_R2: f64 = 1.0e-12;

/// Half-angle `(1 + |u_re|) * 0.5` singularity floor (matching `T1SubHarmonicKernel`).
const T1_HALF_ANGLE_EPS: f64 = 1.0e-12;

/// Near-origin antipodal branch-cut chord tolerance `2^-18` (matching `T1SubHarmonicKernel`).
const BRANCH_CUT_CHORD_TOL: f64 = 1.0 / ((1_u32 << 18) as f64);

const _: () = assert!(T1_SILENCE_SQUELCH_R2 > 0.0 && T1_SILENCE_SQUELCH_R2 < 1.0e-9);
const _: () = assert!(T1_HALF_ANGLE_EPS > 0.0 && T1_HALF_ANGLE_EPS < 1.0e-9);
const _: () = assert!(BRANCH_CUT_CHORD_TOL > 0.0 && BRANCH_CUT_CHORD_TOL < 1.0e-5);

#[inline]
fn tracker_crosses_branch_cut_f64(prev_re: f64, prev_im: f64, cur_re: f64, cur_im: f64) -> bool {
    let dot = prev_re * cur_re + prev_im * cur_im;
    let det = prev_re * cur_im - prev_im * cur_re;
    if dot < 0.0 && det.abs() <= BRANCH_CUT_CHORD_TOL {
        return false;
    }
    (prev_im >= 0.0 && cur_im < 0.0 && det > 0.0) || (prev_im < 0.0 && cur_im >= 0.0 && det < 0.0)
}

#[inline]
fn cancellation_free_half_angle_f64(re: f64, im: f64) -> (f64, f64) {
    let h = (0.5 * (1.0 + re.abs())).max(T1_HALF_ANGLE_EPS);
    let rho_half = h.sqrt().recip();
    let v_maj = (h * rho_half).clamp(0.0, 1.0);
    let v_min = (im.abs() * (0.5 * rho_half)).clamp(0.0, 1.0);
    let (c, abs_s) = if re >= 0.0 {
        (v_maj, v_min)
    } else {
        (v_min, v_maj)
    };
    let sigma_pos = if im >= 0.0 { 1.0_f64 } else { -1.0_f64 };
    (c, sigma_pos * abs_s)
}

/// Pure-`f64` mirror of `dnam_kernels::T1SubHarmonicKernel` enabling `< 1e-6` central-FD
/// gradient verification across `Op-P` without `f32` quantization noise.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct T1SubHarmonicF64 {
    prev_u_re: f64,
    prev_u_im: f64,
    sheet: f64,
    prev_w1_re: f64,
    prev_w1_im: f64,
    sheet2: f64,
    prev_valid: bool,
}

impl T1SubHarmonicF64 {
    #[inline]
    pub(super) const fn new() -> Self {
        Self {
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            sheet: 1.0,
            prev_w1_re: 1.0,
            prev_w1_im: 0.0,
            sheet2: 1.0,
            prev_valid: false,
        }
    }

    #[inline]
    pub(super) fn step(&mut self, i_in: f64, q_in: f64) -> [f64; 4] {
        if !i_in.is_finite() || !q_in.is_finite() {
            self.prev_valid = false;
            return [0.0; 4];
        }
        let r2 = i_in * i_in + q_in * q_in;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            self.prev_valid = false;
            return [0.0; 4];
        }

        let r = r2.sqrt();
        let inv_r = 1.0 / r;
        let u_re = (i_in * inv_r).clamp(-1.0, 1.0);
        let u_im = (q_in * inv_r).clamp(-1.0, 1.0);

        // Stage 1: z -> w1 = z^(1/2)
        if self.prev_valid
            && tracker_crosses_branch_cut_f64(self.prev_u_re, self.prev_u_im, u_re, u_im)
        {
            self.sheet = -self.sheet;
        }
        let (c1, s1) = cancellation_free_half_angle_f64(u_re, u_im);
        let sub1_cos = self.sheet * c1;
        let sub1_sin = self.sheet * s1;

        // Stage 2: w1 -> w2 = w1^(1/2) = z^(1/4)
        if self.prev_valid
            && tracker_crosses_branch_cut_f64(self.prev_w1_re, self.prev_w1_im, sub1_cos, sub1_sin)
        {
            self.sheet2 = -self.sheet2;
        }
        let (c2, s2) = cancellation_free_half_angle_f64(sub1_cos, sub1_sin);
        let sub2_cos = self.sheet2 * c2;
        let sub2_sin = self.sheet2 * s2;

        self.prev_u_re = u_re;
        self.prev_u_im = u_im;
        self.prev_w1_re = sub1_cos;
        self.prev_w1_im = sub1_sin;
        self.prev_valid = true;

        [sub1_cos * r, sub1_sin * r, sub2_cos * r, sub2_sin * r]
    }
}

/// Folds `d_feat` back to `1x` via the exact `os2x` midpoint transpose when `trace.os2x` is `true`,
/// applies the exact radial/angular `T1` Jacobian at each `1x` frame, and splits `0.5 * d_i_in`,
/// `0.5 * d_q_in` into both stereo `1x` `(I, Q)` gradient buffers.
pub(super) fn accumulate_t1_1x_iq_grads(
    trace: &CrossbarForwardTrace,
    d_feat: &[[f64; 4]],
    d_il_1x: &mut [f64],
    d_ql_1x: &mut [f64],
    d_ir_1x: &mut [f64],
    d_qr_1x: &mut [f64],
) {
    let n_1x = d_il_1x
        .len()
        .min(d_ql_1x.len())
        .min(d_ir_1x.len())
        .min(d_qr_1x.len());
    let expected_n = if trace.os2x { n_1x * 2 } else { n_1x };
    if n_1x == 0
        || d_feat.len() < expected_n
        || trace.iq_lr.len() < expected_n
        || trace.op_p_feat.len() < expected_n
    {
        return;
    }

    for t1x in 0..n_1x {
        let (idx_1x, df) = if trace.os2x {
            let mut acc = [0.0_f64; 4];
            let cur = d_feat[2 * t1x + 1];
            let mid_cur = d_feat[2 * t1x];
            let mid_next = if t1x + 1 < n_1x {
                d_feat[2 * (t1x + 1)]
            } else {
                [0.0; 4]
            };
            for l in 0..4 {
                acc[l] = cur[l] + 0.5 * mid_cur[l] + 0.5 * mid_next[l];
            }
            (2 * t1x + 1, acc)
        } else {
            (t1x, d_feat[t1x])
        };

        let [i_l, q_l, i_r, q_r] = trace.iq_lr[idx_1x];
        let i_in = 0.5 * (i_l + i_r);
        let q_in = 0.5 * (q_l + q_r);
        let r2 = i_in * i_in + q_in * q_in;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            continue;
        }

        let r = r2.sqrt();
        let inv_r = 1.0 / r;
        let u_re = (i_in * inv_r).clamp(-1.0, 1.0);
        let u_im = (q_in * inv_r).clamp(-1.0, 1.0);
        let f = trace.op_p_feat[idx_1x];
        let sub1_cos = f[0] * inv_r;
        let sub1_sin = f[1] * inv_r;
        let sub2_cos = f[2] * inv_r;
        let sub2_sin = f[3] * inv_r;

        let d_r = df[0] * sub1_cos + df[1] * sub1_sin + df[2] * sub2_cos + df[3] * sub2_sin;
        let d_theta_norm = 0.5 * (-df[0] * sub1_sin + df[1] * sub1_cos)
            + 0.25 * (-df[2] * sub2_sin + df[3] * sub2_cos);
        let d_i_in = u_re * d_r - u_im * d_theta_norm;
        let d_q_in = u_im * d_r + u_re * d_theta_norm;

        let half_di = 0.5 * d_i_in;
        let half_dq = 0.5 * d_q_in;
        d_il_1x[t1x] += half_di;
        d_ql_1x[t1x] += half_dq;
        d_ir_1x[t1x] += half_di;
        d_qr_1x[t1x] += half_dq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::{HILBERT_H1_STAGES, T1SubHarmonicKernel};
    use proptest::prelude::*;

    #[test]
    fn test_t1_subharmonic_f64_matches_kernel_f32_across_multi_revolution_sweeps() {
        let mut t1_f64 = T1SubHarmonicF64::new();
        let mut t1_f32 = T1SubHarmonicKernel::new();

        // Silence squelch & non-finite reset checks.
        assert_eq!(t1_f64.step(1.0e-7, -1.0e-7), [0.0; 4]);
        assert_eq!(t1_f32.step(1.0e-7, -1.0e-7), [0.0; 4]);
        assert_eq!(t1_f64.step(f64::NAN, 0.5), [0.0; 4]);
        assert_eq!(t1_f32.step(f32::NAN, 0.5), [0.0; 4]);

        // Positive & negative multi-revolution sweeps crossing Stage 1 and Stage 2 branch cuts.
        for step_idx in 0..240_usize {
            let dir = if step_idx < 120 { 1.0_f64 } else { -1.0_f64 };
            let theta = dir * (step_idx as f64) * 0.23;
            let amp = 0.25 + 0.65 * ((step_idx as f64) * 0.07).sin().abs();
            let i_in = amp * theta.cos();
            let q_in = amp * theta.sin();

            let out64 = t1_f64.step(i_in, q_in);
            let out32 = t1_f32.step(i_in as f32, q_in as f32);
            for k in 0..4 {
                assert!(
                    (out64[k] - f64::from(out32[k])).abs() < 1.0e-5,
                    "step {step_idx} lane {k}: f64={} f32={}",
                    out64[k],
                    out32[k]
                );
            }
        }
    }

    fn eval_t1_sequence_loss(
        iq_1x: &[[f64; 4]],
        d_feat: &[[f64; 4]],
        os2x: bool,
    ) -> (CrossbarForwardTrace, f64) {
        let n_1x = iq_1x.len();
        let out_n = if os2x { n_1x * 2 } else { n_1x };
        let mut t1 = T1SubHarmonicF64::new();
        let mut iq_lr = Vec::with_capacity(out_n);
        let mut op_p_feat = Vec::with_capacity(out_n);
        let mut prev_iq = [0.0_f64; 4];
        let mut prev_feat = [0.0_f64; 4];

        for &iq_cur in iq_1x {
            let feat_cur = t1.step(0.5 * (iq_cur[0] + iq_cur[2]), 0.5 * (iq_cur[1] + iq_cur[3]));
            if os2x {
                let iq_mid = core::array::from_fn(|k| 0.5 * (prev_iq[k] + iq_cur[k]));
                let feat_mid = core::array::from_fn(|k| 0.5 * (prev_feat[k] + feat_cur[k]));
                iq_lr.push(iq_mid);
                op_p_feat.push(feat_mid);
                iq_lr.push(iq_cur);
                op_p_feat.push(feat_cur);
                prev_iq = iq_cur;
                prev_feat = feat_cur;
            } else {
                iq_lr.push(iq_cur);
                op_p_feat.push(feat_cur);
            }
        }
        let loss: f64 = op_p_feat
            .iter()
            .zip(d_feat)
            .map(|(f, df)| (0..4).map(|l| f[l] * df[l]).sum::<f64>())
            .sum();
        let trace = CrossbarForwardTrace {
            n: out_n,
            q_count: 1,
            sample_rate_hz: 48_000.0,
            include_stereo_phase: false,
            os2x,
            active: true,
            dc_pole: 0.0,
            dc_gain: 1.0,
            alpha_i: [0.0; HILBERT_H1_STAGES],
            alpha_q: [0.0; HILBERT_H1_STAGES],
            iq_lr,
            op_p_feat,
            cb_add_l: Vec::new(),
            cb_add_r: Vec::new(),
        };
        (trace, loss)
    }

    #[test]
    fn test_t1_accumulate_iq_grads_silence_squelch_returns_zero() {
        let iq_1x = vec![[1.0e-7, -1.0e-7, 1.0e-7, -1.0e-7]; 4];
        let d_feat = vec![[1.0, -2.0, 0.5, 0.25]; 4];
        let (trace, _) = eval_t1_sequence_loss(&iq_1x, &d_feat, false);
        let (mut dil, mut dql, mut dir, mut dqr) =
            (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
        accumulate_t1_1x_iq_grads(&trace, &d_feat, &mut dil, &mut dql, &mut dir, &mut dqr);
        assert_eq!(dil, vec![0.0; 4]);
        assert_eq!(dql, vec![0.0; 4]);
        assert_eq!(dir, vec![0.0; 4]);
        assert_eq!(dqr, vec![0.0; 4]);
    }

    proptest! {
        #[test]
        fn prop_t1_accumulate_1x_iq_grads_matches_central_fd(
            os2x in any::<bool>(),
            theta0 in -core::f64::consts::PI..core::f64::consts::PI,
            omega in prop_oneof![0.25_f64..1.05_f64, -1.05_f64..-0.25_f64],
            r_base in 0.2_f64..1.5_f64,
            df_scale in prop::collection::vec(-1.5_f64..1.5_f64, 4),
            probe_t in 0_usize..10_usize,
            probe_k in 0_usize..4_usize,
        ) {
            let n_1x = 10_usize;
            let iq_1x: Vec<[f64; 4]> = (0..n_1x)
                .map(|t| {
                    let th = theta0 + omega * (t as f64);
                    let r = r_base * (1.0 + 0.15 * (t as f64 * 0.7).sin());
                    let i_mid = r * th.cos();
                    let q_mid = r * th.sin();
                    let asym = 0.05 * ((t + 1) as f64 * 0.3).cos();
                    [i_mid + asym, q_mid - asym, i_mid - asym, q_mid + asym]
                })
                .collect();
            let out_n = if os2x { n_1x * 2 } else { n_1x };
            let d_feat: Vec<[f64; 4]> = (0..out_n)
                .map(|t| {
                    core::array::from_fn(|l| {
                        df_scale[l] * (((t + 1) * (l + 2)) as f64 * 0.19).cos()
                    })
                })
                .collect();

            let (trace, _) = eval_t1_sequence_loss(&iq_1x, &d_feat, os2x);
            let (mut dil, mut dql, mut dir, mut dqr) = (
                vec![0.0; n_1x],
                vec![0.0; n_1x],
                vec![0.0; n_1x],
                vec![0.0; n_1x],
            );
            accumulate_t1_1x_iq_grads(&trace, &d_feat, &mut dil, &mut dql, &mut dir, &mut dqr);
            let ana = match probe_k {
                0 => dil[probe_t],
                1 => dql[probe_t],
                2 => dir[probe_t],
                _ => dqr[probe_t],
            };

            let eps = 1.0e-6_f64;
            let (mut p_iq, mut m_iq) = (iq_1x.clone(), iq_1x);
            p_iq[probe_t][probe_k] += eps;
            m_iq[probe_t][probe_k] -= eps;
            let (_, lp) = eval_t1_sequence_loss(&p_iq, &d_feat, os2x);
            let (_, lm) = eval_t1_sequence_loss(&m_iq, &d_feat, os2x);
            let fd = (lp - lm) / (2.0 * eps);
            let scale = ana.abs().max(fd.abs()).max(1.0e-3);
            prop_assert!(
                (ana - fd).abs() < 1.0e-6 * scale,
                "os2x={os2x} t={probe_t} k={probe_k}: ana={ana} fd={fd}"
            );
        }
    }
}
