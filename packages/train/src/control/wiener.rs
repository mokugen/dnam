//! Compile-time derived 4-band TPT-SVF bank (`250 Hz` & `1,800 Hz`, `Q = 0.8`), Two-Pass
//! 4-Band Wiener Recovery, data-driven sweep-window midpoint extraction, and `SonaVm`-backed
//! 32-slot control Jacobian identification (`ref[train.pipeline.control-projection-fit]`).

use core::f64::consts::PI;

use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::{DefaultMathBackend, const_tan_f64};
use dnam_vm::{SonaVm, Tier};

use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_adaptive_ridge;

/// Log-RMS matching weight whenever band phase correlation falls below `BAND_PHASE_DECORR_RHO_MAX`.
pub const FOUR_BAND_RMS_MATCH_WEIGHT: f64 = 2.5;
/// Minimum active band fraction of total Pass-1 energy required to contribute to Wiener weighting.
pub const ACTIVE_BAND_MIN_ENERGY_FRAC: f64 = 0.05;
/// Waveform correlation threshold below which pure time-domain MSE switches to log-RMS Wiener recovery.
pub const BAND_PHASE_DECORR_RHO_MAX: f64 = 0.65;

/// Canonical 4-band low crossover frequency in Hz (`const_eval`).
pub const FOUR_BAND_LO_HZ: f64 = 250.0;
/// Canonical 4-band high crossover frequency in Hz (`const_eval`).
pub const FOUR_BAND_HI_HZ: f64 = 1_800.0;
/// Canonical TPT-SVF crossover damping factor `k = 1 / Q` (`Q = 0.8`).
pub const FOUR_BAND_SVF_K: f64 = 1.0 / 0.8;

const _: () = assert!(FOUR_BAND_SVF_K == 1.25);

#[inline]
const fn svf_coeffs_const(fc_hz: f64, sr_hz: f64) -> (f64, f64, f64, f64) {
    let g = const_tan_f64(PI * fc_hz / sr_hz);
    let a1 = 1.0 / (1.0 + g * (g + FOUR_BAND_SVF_K));
    let a2 = g * a1;
    let a3 = g * a2;
    (g, a1, a2, a3)
}

/// Compile-time derived 4-band TPT-SVF coefficients for `48 kHz` (`ref[repo.code.constant-eval-and-derivations]`).
const SVF_48K_LO_G: f64 = const_tan_f64(PI * FOUR_BAND_LO_HZ / 48_000.0);
const SVF_48K_LO_ONEPOLE_BIG_G: f64 = SVF_48K_LO_G / (1.0 + SVF_48K_LO_G);
const SVF_48K_HI_G: f64 = const_tan_f64(PI * FOUR_BAND_HI_HZ / 48_000.0);
const _: () = assert!(SVF_48K_LO_G > 0.0 && SVF_48K_LO_G < SVF_48K_HI_G);
const _: () = assert!(SVF_48K_LO_ONEPOLE_BIG_G > 0.0 && SVF_48K_LO_ONEPOLE_BIG_G < 1.0);
const _: () = assert!(svf_coeffs_const(FOUR_BAND_HI_HZ, 48_000.0).1 < 1.0);

#[derive(Clone, Copy, Debug)]
struct TptSvfStage {
    a1: f64,
    a2: f64,
    a3: f64,
}

impl TptSvfStage {
    #[inline]
    fn from_fc_sr(fc_hz: f64, sr_hz: f64) -> Self {
        let g = (PI * fc_hz / sr_hz).clamp(1.0e-5, 0.49 * PI).tan();
        let a1 = 1.0 / (1.0 + g * (g + FOUR_BAND_SVF_K));
        let a2 = g * a1;
        let a3 = g * a2;
        Self { a1, a2, a3 }
    }

    #[inline]
    fn step(&self, x: f64, ic1eq: &mut f64, ic2eq: &mut f64) -> (f64, f64) {
        let v3 = x - *ic2eq;
        let v1 = self.a1 * *ic1eq + self.a2 * v3;
        let v2 = *ic2eq + self.a2 * *ic1eq + self.a3 * v3;
        *ic1eq = 2.0 * v1 - *ic1eq;
        *ic2eq = 2.0 * v2 - *ic2eq;
        (v1, v2)
    }
}

/// Canonical 4-band complementary TPT-SVF crossover bank (`Sub < 250 Hz` 1-pole TPT lowpass in
/// parallel with `1.8 kHz`, `Q = 0.8` 2-pole TPT-SVF, `ref[train.pipeline.control-projection-fit]`).
#[derive(Clone, Copy, Debug)]
pub struct FourBandTptBank {
    lo_big_g: f64,
    hi: TptSvfStage,
}

impl FourBandTptBank {
    /// Constructs a 4-band TPT-SVF bank prewarped for `sample_rate_hz`.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let sr = sample_rate_hz.clamp(8_000.0, 1_536_000.0);
        if (sr - 48_000.0).abs() < 1.0e-9 {
            let (_, a1_1, a2_1, a3_1) = svf_coeffs_const(FOUR_BAND_HI_HZ, 48_000.0);
            return Self {
                lo_big_g: SVF_48K_LO_ONEPOLE_BIG_G,
                hi: TptSvfStage {
                    a1: a1_1,
                    a2: a2_1,
                    a3: a3_1,
                },
            };
        }
        let g_sub = (PI * FOUR_BAND_LO_HZ / sr).clamp(1.0e-5, 0.49 * PI).tan();
        Self {
            lo_big_g: g_sub / (1.0 + g_sub),
            hi: TptSvfStage::from_fc_sr(FOUR_BAND_HI_HZ, sr),
        }
    }

    /// Splits `sig` into parallel `[sub, low_mid, presence, high]` with exact sample-by-sample partition of unity.
    #[must_use]
    pub fn split_4bands(&self, sig: &[f64]) -> [Vec<f64>; 4] {
        let n = sig.len();
        let mut out = [
            vec![0.0_f64; n],
            vec![0.0_f64; n],
            vec![0.0_f64; n],
            vec![0.0_f64; n],
        ];
        let (mut s_sub, mut s1, mut s2) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (i, &s) in sig.iter().enumerate() {
            let v_sub = self.lo_big_g * (s - s_sub);
            let b_sub = v_sub + s_sub;
            s_sub = b_sub + v_sub;
            let (v1, v2) = self.hi.step(s, &mut s1, &mut s2);
            out[0][i] = b_sub;
            out[1][i] = v2 - b_sub;
            out[2][i] = FOUR_BAND_SVF_K * v1;
            out[3][i] = s - FOUR_BAND_SVF_K * v1 - v2;
        }
        out
    }
}

/// Output of Two-Pass 4-Band Wiener Recovery combining StarCenter and Pass-1 renders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourBandWienerResult {
    /// Tier-averaged broadband loss (Pass-1 MSE or Pass-2 `0.25 * w_band` 4-band RMS envelope loss).
    pub loss: f64,
    /// Whether Pass-2 4-band Wiener recovery triggered under canonical energy/correlation predicates.
    pub pass2_triggered: bool,
    /// Per-band normalized octave energy errors `0.5 * log2(E_mdl / E_wet)` across `[Sub, LM, Pres, Hi]`.
    pub band_oct_err: [f64; 4],
    /// Per-band tier-averaged normalized waveform correlation `rho_b`.
    pub band_rho: [f64; 4],
    /// Active band mask gated on target wet energy (`E_wet,b >= 0.05 * E_wet,tot`).
    pub active_bands: [bool; 4],
    /// Recovered per-band gain correction ratios `[g0, g1, g2, g3]`.
    pub wiener_band_gains: [f64; 4],
}

fn corr_and_energy(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let mut ea = 0.0_f64;
    let mut eb = 0.0_f64;
    let mut dot = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        ea += x * x;
        eb += y * y;
        dot += x * y;
    }
    let den = (ea * eb).sqrt() + 1.0e-12;
    ((dot / den).clamp(-1.0, 1.0), ea, eb)
}

/// Evaluates Two-Pass 4-Band Wiener Recovery combining StarCenter (`y_star_u`, `y_star_l`) and
/// Pass-1 (`y_mdl_u`, `y_mdl_l`) renders against `target` (`ref[train.pipeline.control-projection-fit]`).
pub fn solve_two_pass_four_band_wiener(
    bank: &FourBandTptBank,
    target: &[f64],
    y_star_u: &[f64],
    y_star_l: &[f64],
    y_mdl_u: &[f64],
    y_mdl_l: &[f64],
    warmup: usize,
) -> Result<FourBandWienerResult, TrainError> {
    let n = target.len();
    if y_star_u.len() != n
        || y_star_l.len() != n
        || y_mdl_u.len() != n
        || y_mdl_l.len() != n
        || n <= warmup
    {
        return Err(TrainError::InvalidDimensions {
            context: "solve_two_pass_four_band_wiener",
            expected: n.max(warmup + 1),
            actual: y_mdl_u.len().min(y_mdl_l.len()),
        });
    }
    let b_t = bank.split_4bands(target);
    let b_su = bank.split_4bands(y_star_u);
    let b_sl = bank.split_4bands(y_star_l);
    let b_mu = bank.split_4bands(y_mdl_u);
    let b_ml = bank.split_4bands(y_mdl_l);

    let (rho_su, e_wet, es_u_tot) = corr_and_energy(&target[warmup..], &y_star_u[warmup..]);
    let (rho_sl, _, es_l_tot) = corr_and_energy(&target[warmup..], &y_star_l[warmup..]);
    let (_, _, em_u_tot) = corr_and_energy(&target[warmup..], &y_mdl_u[warmup..]);
    let (_, _, em_l_tot) = corr_and_energy(&target[warmup..], &y_mdl_l[warmup..]);
    let eps = 1.0e-12_f64;
    let e_star = 0.5 * (es_u_tot + es_l_tot);
    let e_mdl_1 = 0.5 * (em_u_tot + em_l_tot);
    let oct_tot = 0.5 * ((e_wet + eps) / (e_star + eps)).log2();
    let rho_wave = 0.5 * (rho_su + rho_sl);
    let oct_err_1 = 0.5 * ((e_mdl_1 + eps) / (e_wet + eps)).log2();

    let mut et_b = [0.0_f64; 4];
    let mut em_u_b = [0.0_f64; 4];
    let mut em_l_b = [0.0_f64; 4];
    let mut em_b = [0.0_f64; 4];
    let mut es_b = [0.0_f64; 4];
    for b in 0..4 {
        et_b[b] = b_t[b][warmup..].iter().map(|v| v * v).sum::<f64>();
        let (_, eu, el) = corr_and_energy(&b_mu[b][warmup..], &b_ml[b][warmup..]);
        em_u_b[b] = eu;
        em_l_b[b] = el;
        em_b[b] = 0.5 * (eu + el);
        let (_, _, es_u) = corr_and_energy(&b_t[b][warmup..], &b_su[b][warmup..]);
        let (_, _, es_l) = corr_and_energy(&b_t[b][warmup..], &b_sl[b][warmup..]);
        es_b[b] = 0.5 * (es_u + es_l);
    }
    let e_wet_tot = et_b.iter().sum::<f64>().max(eps);

    let mut band_oct_err = [0.0_f64; 4];
    let mut band_rho = [0.0_f64; 4];
    let mut active_bands = [false; 4];
    let mut wiener_band_gains = [1.0_f64; 4];
    let mut band_collapse = false;

    for b in 0..4 {
        let active = et_b[b] >= ACTIVE_BAND_MIN_ENERGY_FRAC * e_wet_tot;
        active_bands[b] = active;
        let (rho_mu, _, _) = corr_and_energy(&b_t[b][warmup..], &b_mu[b][warmup..]);
        let (rho_ml, _, _) = corr_and_energy(&b_t[b][warmup..], &b_ml[b][warmup..]);
        let rho_b = 0.5 * (rho_mu + rho_ml);
        band_rho[b] = rho_b;

        band_oct_err[b] = 0.5 * ((em_b[b] + eps) / (et_b[b] + eps)).log2();
        wiener_band_gains[b] = ((et_b[b] + eps) / (0.5 * em_b[b] + 0.5 * es_b[b] + eps))
            .sqrt()
            .clamp(0.125, 8.0);
        if active && rho_b < BAND_PHASE_DECORR_RHO_MAX && band_oct_err[b] < -0.55 {
            band_collapse = true;
        }
    }

    let pass2_triggered =
        oct_tot > -2.5 && rho_wave > -0.15 && (oct_err_1 < -0.55 || band_collapse);

    let mut loss = 0.0_f64;
    if pass2_triggered {
        let rt_t = et_b.map(f64::sqrt);
        let rt_u = em_u_b.map(f64::sqrt);
        let rt_l = em_l_b.map(f64::sqrt);
        for b in 0..4 {
            let du = rt_t[b] - rt_u[b];
            let dl = rt_t[b] - rt_l[b];
            loss += 0.25 * FOUR_BAND_RMS_MATCH_WEIGHT * (du * du + dl * dl);
        }
    } else {
        for b in 0..4 {
            if active_bands[b] {
                let mse_u: f64 = b_t[b][warmup..]
                    .iter()
                    .zip(&b_mu[b][warmup..])
                    .map(|(t, m)| (t - m).powi(2))
                    .sum::<f64>();
                let mse_l: f64 = b_t[b][warmup..]
                    .iter()
                    .zip(&b_ml[b][warmup..])
                    .map(|(t, m)| (t - m).powi(2))
                    .sum::<f64>();
                loss += 0.5 * (mse_u + mse_l) / (et_b[b] + eps);
            }
        }
    }

    Ok(FourBandWienerResult {
        loss,
        pass2_triggered,
        band_oct_err,
        band_rho,
        active_bands,
        wiener_band_gains,
    })
}

/// Active contiguous sweep segment window bounded purely by signal RMS activity (`label-free`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepWindowSegment {
    /// Inclusive start frame index.
    pub start_frame: usize,
    /// Exclusive end frame index.
    pub end_frame: usize,
    /// Exact temporal midpoint frame `(start_frame + end_frame) / 2`.
    pub midpoint_frame: usize,
}

/// Extracts active sweep-window segments and their temporal midpoints purely from waveform RMS
/// envelopes without referencing fixture/sweep metadata or fixed frequencies.
#[must_use]
pub fn extract_sweep_window_midpoints(
    sig: &[f64],
    hop_frames: usize,
    rms_thresh: f64,
) -> Vec<SweepWindowSegment> {
    let hop = hop_frames.max(16);
    let mut windows = Vec::new();
    let mut cur_start: Option<usize> = None;
    let mut off = 0_usize;
    while off < sig.len() {
        let end = (off + hop).min(sig.len());
        let rms = (sig[off..end].iter().map(|v| v * v).sum::<f64>() / ((end - off) as f64)).sqrt();
        if rms >= rms_thresh {
            cur_start.get_or_insert(off);
        } else if let Some(s) = cur_start.take() {
            windows.push(SweepWindowSegment {
                start_frame: s,
                end_frame: off,
                midpoint_frame: s + (off - s) / 2,
            });
        }
        off = end;
    }
    if let Some(s) = cur_start {
        windows.push(SweepWindowSegment {
            start_frame: s,
            end_frame: sig.len(),
            midpoint_frame: s + (sig.len() - s) / 2,
        });
    }
    windows
}

/// `SonaVm`-backed 32-slot regularized Jacobian Gauss–Newton solver computing bounded
/// `delta_stage_targets[0..32]` given any deterministic probe callback returning `SonaVm` audio.
pub fn solve_control_jacobian_32slot<F>(
    target: &[f64],
    base_targets: &[f64; STAGE_TARGET_SLOTS],
    active_slots: &[usize],
    eps_fd: f64,
    l2_reg: f64,
    mut eval_vm_render: F,
) -> Result<[f64; STAGE_TARGET_SLOTS], TrainError>
where
    F: FnMut(&[f64; STAGE_TARGET_SLOTS]) -> Result<Vec<f64>, TrainError>,
{
    let y0 = eval_vm_render(base_targets)?;
    let n = target.len().min(y0.len());
    if n == 0 || active_slots.is_empty() {
        return Ok([0.0; STAGE_TARGET_SLOTS]);
    }
    let m = active_slots.len();
    let mut jac = vec![vec![0.0_f64; n]; m];
    let eps = eps_fd.abs().max(1.0e-5);
    for (col, &slot_idx) in active_slots.iter().enumerate() {
        if slot_idx >= STAGE_TARGET_SLOTS {
            continue;
        }
        let mut p = *base_targets;
        let mut q = *base_targets;
        p[slot_idx] += eps;
        q[slot_idx] -= eps;
        let yp = eval_vm_render(&p)?;
        let yq = eval_vm_render(&q)?;
        let len = n.min(yp.len()).min(yq.len());
        for t in 0..len {
            jac[col][t] = (yp[t] - yq[t]) / (2.0 * eps);
        }
    }

    let packed_len = m * (m + 1) / 2;
    let mut g_u = vec![0.0_f64; packed_len];
    let mut rhs = vec![0.0_f64; m];
    for c in 0..m {
        let col_off = c * (c + 1) / 2;
        for r in 0..=c {
            let mut dot = 0.0_f64;
            for t in 0..n {
                dot += jac[r][t] * jac[c][t];
            }
            g_u[col_off + r] = dot;
        }
        let mut rd = 0.0_f64;
        for t in 0..n {
            rd += jac[c][t] * (target[t] - y0[t]);
        }
        rhs[c] = rd;
    }

    let (sol, _ridge) = solve_uplo_adaptive_ridge(m, &g_u, &rhs, &[], None, l2_reg.max(1.0e-6))?;
    let mut delta = [0.0_f64; STAGE_TARGET_SLOTS];
    for (col, &slot_idx) in active_slots.iter().enumerate() {
        if slot_idx < STAGE_TARGET_SLOTS {
            delta[slot_idx] = sol[col].clamp(-4.0, 4.0);
        }
    }
    Ok(delta)
}

/// Renders one deterministic mono probe via `SonaVm::<DefaultMathBackend>` for control Jacobian probes.
pub fn render_vm_mono_probe(
    dnam_bin: &[u8],
    tier: Tier,
    sample_rate_hz: f32,
    input: &[f32],
) -> Result<Vec<f64>, TrainError> {
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(dnam_bin, tier, sample_rate_hz)?;
    let n = input.len();
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    vm.process_block(input, input, &mut out_l, &mut out_r)?;
    Ok(out_l.into_iter().map(f64::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::DnamBinBuildInput;
    use dnam_vm::bin::encode_dnam_bin;
    use dnam_vm::math::TANK_DC_BLOCKER_48K_F32;

    #[test]
    fn test_four_band_bank_wiener_sweep_windows_and_32slot_jacobian() {
        assert!((SVF_48K_LO_G - (PI * 250.0 / 48_000.0).tan()).abs() < 1.0e-15);
        let bank = FourBandTptBank::new(48_000.0);
        let n = 1_024_usize;
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                0.5 * (2.0 * PI * 90.0 * t).sin()
                    + 0.4 * (2.0 * PI * 750.0 * t).cos()
                    + 0.3 * (2.0 * PI * 3_500.0 * t).sin()
            })
            .collect();
        let bands = bank.split_4bands(&sig);
        for i in 0..n {
            let sum = bands[0][i] + bands[1][i] + bands[2][i] + bands[3][i];
            assert!((sum - sig[i]).abs() < 1.0e-12);
        }

        // Test phase-decorrelated band path (`rho < 0.65`) via attenuated phase-inverted HF tone.
        let mut flipped = sig.clone();
        for i in 0..n {
            let t = (i as f64) / 48_000.0;
            flipped[i] = bands[0][i] + bands[1][i] - 0.1 * (2.0 * PI * 3_500.0 * t).sin();
        }
        let wr = solve_two_pass_four_band_wiener(
            &bank, &sig, &flipped, &flipped, &flipped, &flipped, 64,
        )
        .unwrap();
        assert!(wr.pass2_triggered && wr.band_rho[3] < BAND_PHASE_DECORR_RHO_MAX && wr.loss > 0.0);
        assert!(wr.band_oct_err[3] < -0.55);
        assert!(wr.wiener_band_gains[3] > 1.3);

        // Even when Pass-1 completely collapses a band (em_b -> 0), active_bands still gates on target energy.
        let mut collapsed = flipped.clone();
        for i in 0..n {
            collapsed[i] = bands[0][i] + bands[1][i];
        }
        let wr_col =
            solve_two_pass_four_band_wiener(&bank, &sig, &sig, &sig, &collapsed, &collapsed, 64)
                .unwrap();
        assert!(wr_col.active_bands[2] || wr_col.active_bands[3]);
        assert!(wr_col.pass2_triggered);

        // Sweep midpoints across two active bursts separated by silence.
        let mut sweep = vec![0.0_f64; 512];
        for v in &mut sweep[64..192] {
            *v = 0.5;
        }
        for v in &mut sweep[320..448] {
            *v = 0.4;
        }
        let segs = extract_sweep_window_midpoints(&sweep, 32, 0.1);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].midpoint_frame, 128);
        assert_eq!(segs[1].midpoint_frame, 384);

        // 32-slot Jacobian solver paired with live `SonaVm` probe helper.
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let bin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 40.0,
            ro_pole_48k: 0.9,
            switch_xfade_samples: 64,
            content_hash_prefix: [5; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let vm_out = render_vm_mono_probe(&bin, Tier::Lite, 48_000.0, &[0.25; 64]).unwrap();
        assert_eq!(vm_out.len(), 64);

        let true_s = [0.4_f64, -0.25_f64];
        let tgt: Vec<f64> = (0..64)
            .map(|i| {
                let x = ((i as f64) * 0.2).sin();
                true_s[0] * x + true_s[1] * x * x
            })
            .collect();
        let delta = solve_control_jacobian_32slot(
            &tgt,
            &[0.0; STAGE_TARGET_SLOTS],
            &[2, 22],
            1.0e-5,
            1.0e-9,
            |st| {
                Ok((0..64)
                    .map(|i| {
                        let x = ((i as f64) * 0.2).sin();
                        st[2] * x + st[22] * x * x
                    })
                    .collect())
            },
        )
        .unwrap();
        assert!((delta[2] - 0.4).abs() < 1.0e-5);
        assert!((delta[22] + 0.25).abs() < 1.0e-5);
    }
}
