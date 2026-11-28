//! `4`-frame-aligned micro-ramp trajectory (`K_ctrl(44.1k) = K_ctrl(48k) = 16`, `alpha_chunk = 0.25`),
//! un-ramped `kautz`/`psi` bypass, `0` static MAC idle skip, asymmetric LFO rate inertia slew,
//! and exact reverse-mode trajectory adjoint (`ref[dsp.ctrl.dispatch.micro-ramp]`,
//! `ref[dsp.ctrl.dispatch.sparse-and-active-ramps]`).

use dnam_vm::control::STAGE_TARGET_SLOTS;

use super::model::{
    GRA_NUM_EDGES, GRA_NUM_OP_T_MODS, GRA_Z_HEAD_DIM, GRA_Z_STAGE_DIM, K_CTRL_44K1_48K,
    MICRO_RAMP_FRAMES, const_gcd_u32,
};

/// Full `71`-float micro-ramped target vector:
/// `[stage_targets(32), slot31_r(1), z_stage(8), z_head(8), g_edge(6), op_t_q(16)]`.
pub const GRA_RAMP_VEC_LEN: usize =
    STAGE_TARGET_SLOTS + 1 + GRA_Z_STAGE_DIM + GRA_Z_HEAD_DIM + GRA_NUM_EDGES + GRA_NUM_OP_T_MODS;

const _: () = assert!(GRA_RAMP_VEC_LEN == 71);
const _: () = assert!(k_ctrl_aligned_4(44_100) == K_CTRL_44K1_48K);
const _: () = assert!(k_ctrl_aligned_4(48_000) == K_CTRL_44K1_48K);
const _: () = assert!(k_ctrl_aligned_4(88_200) == 28);
const _: () = assert!(k_ctrl_aligned_4(96_000) == 32);
const _: () = assert!(k_ctrl_aligned_4(192_000) == 64);
const _: () = assert!(const_gcd_u32(k_ctrl_aligned_4(44_100), MICRO_RAMP_FRAMES) == 4);

/// Computes the `4`-frame-aligned control decimation stride
/// $K_{\text{ctrl}}(f_s) = 4 \cdot \max(2, \mathrm{round}(f_s / 12{,}000))$ at compile time.
#[inline]
#[must_use]
pub const fn k_ctrl_aligned_4(fs_hz_u32: u32) -> u32 {
    if fs_hz_u32 == 0 {
        return K_CTRL_44K1_48K;
    }
    let rounded = (fs_hz_u32.saturating_add(6_000)) / 12_000;
    let units = if rounded < 2 { 2 } else { rounded };
    units * MICRO_RAMP_FRAMES
}

/// Computes the `4`-frame-aligned control stride `K_ctrl(fs)` for floating-point host rates.
#[inline]
#[must_use]
pub fn k_ctrl_for_sample_rate(sample_rate_hz: f64) -> u32 {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return K_CTRL_44K1_48K;
    }
    let rounded = (sample_rate_hz / 12_000.0).round().max(2.0) as u32;
    rounded * MICRO_RAMP_FRAMES
}

/// Computes `alpha_chunk(fs) = clamp(4.0 / K_ctrl(fs), 0.0, 1.0)` (`0.25` at `44.1 / 48 kHz`).
#[inline]
#[must_use]
pub fn alpha_chunk_for_sample_rate(sample_rate_hz: f64) -> f64 {
    let k_ctrl = k_ctrl_for_sample_rate(sample_rate_hz);
    (f64::from(MICRO_RAMP_FRAMES) / f64::from(k_ctrl)).clamp(0.0, 1.0)
}

/// Returns `true` for un-ramped zero-lag `kautz` / `tank_psi` slots (`10, 11, 14`).
#[inline]
#[must_use]
pub const fn is_unramped_kautz_or_psi_slot(idx: usize) -> bool {
    matches!(idx, 10 | 11 | 14)
}

/// One control-window input specifying target vector and dynamic/idle activity flags.
#[derive(Clone, Debug, PartialEq)]
pub struct GraRampTickInput {
    /// Target `71`-float state for this control window.
    pub target: [f64; GRA_RAMP_VEC_LEN],
    /// Target LFO rate prior to asymmetric inertia slew.
    pub lfo_rate_target: f64,
    /// Whether static/knob inputs changed since the previous control boundary.
    pub is_dirty: bool,
    /// Whether internal LFO/envelope modulators are active.
    pub has_internal_modulators: bool,
    /// Whether live OU analog drift is active.
    pub is_analog_drift_active: bool,
}

impl Default for GraRampTickInput {
    fn default() -> Self {
        Self {
            target: [0.0; GRA_RAMP_VEC_LEN],
            lfo_rate_target: 0.0,
            is_dirty: false,
            has_internal_modulators: false,
            is_analog_drift_active: false,
        }
    }
}

#[derive(Clone, Debug)]
struct TickTrace {
    active_tick: bool,
    chunks_in_tick: usize,
    lfo_rate_prev: f64,
    lfo_rate_target: f64,
    alpha_slew: f64,
    was_upward_slew: bool,
}

/// Output trajectory across all `4`-frame microchunks and cached trace for reverse-mode BPTT.
#[derive(Clone, Debug)]
pub struct GraRampTrajectory {
    /// Per-4-frame microchunk output `[total_chunks][71]`.
    pub chunk_states: Vec<[f64; GRA_RAMP_VEC_LEN]>,
    /// Per-control-tick slewed LFO rate `[num_ticks]`.
    pub tick_lfo_rates: Vec<f64>,
    /// Count of control windows skipped with `0` static MACs.
    pub idle_skipped_ticks: usize,
    alpha_chunk: f64,
    init_state: [f64; GRA_RAMP_VEC_LEN],
    traces: Vec<TickTrace>,
}

/// Evaluates a sequence of control-window ticks across `4`-frame microchunks (`K_ctrl / 4` chunks
/// per tick), enforcing `0` static MACs when idle and un-ramped bypass on `kautz`/`psi` slots.
#[must_use]
pub fn eval_micro_ramp_trajectory(
    ticks: &[GraRampTickInput],
    init_state: &[f64; GRA_RAMP_VEC_LEN],
    init_lfo_rate: f64,
    sample_rate_hz: f64,
    alpha_up: f64,
    alpha_dn: f64,
) -> GraRampTrajectory {
    let k_ctrl = k_ctrl_for_sample_rate(sample_rate_hz);
    let chunks_per_tick = (k_ctrl / MICRO_RAMP_FRAMES).max(1) as usize;
    let alpha_chunk = alpha_chunk_for_sample_rate(sample_rate_hz);
    let a_up = alpha_up.clamp(1.0e-5, 1.0);
    let a_dn = alpha_dn.clamp(1.0e-5, 1.0);

    let mut curr = *init_state;
    let mut lfo_rate = init_lfo_rate;
    let mut chunk_states = Vec::with_capacity(ticks.len() * chunks_per_tick);
    let mut tick_lfo_rates = Vec::with_capacity(ticks.len());
    let mut traces = Vec::with_capacity(ticks.len());
    let mut idle_skipped_ticks = 0_usize;
    let mut ramp_chunks_remaining = 0_usize;

    for tick in ticks {
        let active = tick.is_dirty
            || ramp_chunks_remaining > 0
            || tick.has_internal_modulators
            || tick.is_analog_drift_active;
        if !active {
            idle_skipped_ticks += 1;
            traces.push(TickTrace {
                active_tick: false,
                chunks_in_tick: chunks_per_tick,
                lfo_rate_prev: lfo_rate,
                lfo_rate_target: tick.lfo_rate_target,
                alpha_slew: 0.0,
                was_upward_slew: false,
            });
            for _ in 0..chunks_per_tick {
                chunk_states.push(curr);
            }
            tick_lfo_rates.push(lfo_rate);
            continue;
        }

        let start_state = curr;
        let lfo_rate_prev = lfo_rate;
        let upward = tick.lfo_rate_target >= lfo_rate_prev;
        let alpha_slew = if upward { a_up } else { a_dn };
        lfo_rate = lfo_rate_prev + alpha_slew * (tick.lfo_rate_target - lfo_rate_prev);
        tick_lfo_rates.push(lfo_rate);

        let mut delta = [0.0_f64; GRA_RAMP_VEC_LEN];
        for i in 0..GRA_RAMP_VEC_LEN {
            if is_unramped_kautz_or_psi_slot(i) {
                curr[i] = tick.target[i];
            } else {
                delta[i] = (tick.target[i] - start_state[i]) * alpha_chunk;
            }
        }
        ramp_chunks_remaining = chunks_per_tick;

        for c in 0..chunks_per_tick {
            if c + 1 >= chunks_per_tick {
                curr = tick.target;
                ramp_chunks_remaining = 0;
            } else {
                for i in 0..GRA_RAMP_VEC_LEN {
                    if !is_unramped_kautz_or_psi_slot(i) {
                        curr[i] += delta[i];
                    }
                }
                ramp_chunks_remaining = ramp_chunks_remaining.saturating_sub(1);
            }
            chunk_states.push(curr);
        }

        traces.push(TickTrace {
            active_tick: true,
            chunks_in_tick: chunks_per_tick,
            lfo_rate_prev,
            lfo_rate_target: tick.lfo_rate_target,
            alpha_slew,
            was_upward_slew: upward,
        });
    }

    GraRampTrajectory {
        chunk_states,
        tick_lfo_rates,
        idle_skipped_ticks,
        alpha_chunk,
        init_state: *init_state,
        traces,
    }
}

/// Adjoint outputs returned by [`pullback_micro_ramp_trajectory`].
#[derive(Clone, Debug, PartialEq)]
pub struct GraRampGrads {
    /// Gradient wrt each control window's `71`-float target vector.
    pub d_targets: Vec<[f64; GRA_RAMP_VEC_LEN]>,
    /// Gradient wrt each control window's `lfo_rate_target`.
    pub d_lfo_rate_targets: Vec<f64>,
    /// Gradient wrt `init_state`.
    pub d_init_state: [f64; GRA_RAMP_VEC_LEN],
    /// Gradient wrt `init_lfo_rate`.
    pub d_init_lfo_rate: f64,
    /// Gradient wrt `alpha_up`.
    pub d_alpha_up: f64,
    /// Gradient wrt `alpha_dn`.
    pub d_alpha_dn: f64,
}

/// Reverse-mode trajectory adjoint across `4`-frame microchunks (`0-peel` aligned).
#[must_use]
pub fn pullback_micro_ramp_trajectory(
    traj: &GraRampTrajectory,
    d_chunk_states: &[[f64; GRA_RAMP_VEC_LEN]],
    d_tick_lfo_rates: &[f64],
) -> GraRampGrads {
    let n_ticks = traj.traces.len();
    let mut d_targets = vec![[0.0_f64; GRA_RAMP_VEC_LEN]; n_ticks];
    let mut d_lfo_rate_targets = vec![0.0_f64; n_ticks];
    let mut d_curr = [0.0_f64; GRA_RAMP_VEC_LEN];
    let mut d_lfo = 0.0_f64;
    let mut d_alpha_up = 0.0_f64;
    let mut d_alpha_dn = 0.0_f64;
    let mut chunk_cursor = d_chunk_states.len();

    for (t_idx, tr) in traj.traces.iter().enumerate().rev() {
        let c_len = tr.chunks_in_tick;
        let c_start = chunk_cursor.saturating_sub(c_len);
        let tick_chunks = &d_chunk_states[c_start..chunk_cursor];
        chunk_cursor = c_start;

        let d_rate_out = d_lfo + d_tick_lfo_rates.get(t_idx).copied().unwrap_or(0.0);
        if !tr.active_tick {
            for ch in tick_chunks {
                for i in 0..GRA_RAMP_VEC_LEN {
                    d_curr[i] += ch[i];
                }
            }
            d_lfo = d_rate_out;
            continue;
        }

        let diff_rate = tr.lfo_rate_target - tr.lfo_rate_prev;
        if tr.alpha_slew > 1.0e-5 && tr.alpha_slew < 1.0 {
            if tr.was_upward_slew {
                d_alpha_up += d_rate_out * diff_rate;
            } else {
                d_alpha_dn += d_rate_out * diff_rate;
            }
        }
        d_lfo_rate_targets[t_idx] = d_rate_out * tr.alpha_slew;
        d_lfo = d_rate_out * (1.0 - tr.alpha_slew);

        let mut next_d_curr = [0.0_f64; GRA_RAMP_VEC_LEN];
        for (c, ch) in tick_chunks.iter().enumerate() {
            let is_last = c + 1 >= c_len;
            for i in 0..GRA_RAMP_VEC_LEN {
                if is_unramped_kautz_or_psi_slot(i) || is_last {
                    d_targets[t_idx][i] += ch[i];
                } else {
                    let frac = ((c + 1) as f64) * traj.alpha_chunk;
                    d_targets[t_idx][i] += ch[i] * frac;
                    next_d_curr[i] += ch[i] * (1.0 - frac);
                }
            }
        }
        // Carry upstream `d_curr` from subsequent ticks: the tick ends snapped to `target`.
        for i in 0..GRA_RAMP_VEC_LEN {
            d_targets[t_idx][i] += d_curr[i];
        }
        d_curr = next_d_curr;
    }

    let _ = traj.init_state;
    GraRampGrads {
        d_targets,
        d_lfo_rate_targets,
        d_init_state: d_curr,
        d_init_lfo_rate: d_lfo,
        d_alpha_up,
        d_alpha_dn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_ctrl_alignment_idle_zero_macs_and_trajectory_fd() {
        assert_eq!(k_ctrl_for_sample_rate(44_100.0), 16);
        assert_eq!(k_ctrl_for_sample_rate(48_000.0), 16);
        assert_eq!(k_ctrl_for_sample_rate(96_000.0), 32);
        assert!((alpha_chunk_for_sample_rate(44_100.0) - 0.25).abs() < 1.0e-12);
        assert!((alpha_chunk_for_sample_rate(48_000.0) - 0.25).abs() < 1.0e-12);

        let mut init = [0.0_f64; GRA_RAMP_VEC_LEN];
        init[0] = 0.2;
        init[10] = -0.3;

        let mut t1 = [0.0_f64; GRA_RAMP_VEC_LEN];
        t1[0] = 1.0;
        t1[10] = 0.5;
        t1[31] = -0.4;
        t1[32] = 0.4;
        t1[49] = 0.8;
        t1[55] = -0.6;

        let mut t2 = t1;
        t2[0] = 0.6;
        t2[14] = 0.7;

        let ticks = vec![
            GraRampTickInput {
                target: t1,
                lfo_rate_target: 1.4,
                is_dirty: true,
                has_internal_modulators: false,
                is_analog_drift_active: false,
            },
            GraRampTickInput {
                target: t1,
                lfo_rate_target: 1.4,
                is_dirty: false,
                has_internal_modulators: false,
                is_analog_drift_active: false,
            },
            GraRampTickInput {
                target: t2,
                lfo_rate_target: 0.3,
                is_dirty: true,
                has_internal_modulators: true,
                is_analog_drift_active: false,
            },
        ];

        let traj = eval_micro_ramp_trajectory(&ticks, &init, 0.5, 44_100.0, 0.6, 0.25);
        // 3 ticks * 4 chunks = 12 chunks; tick 1 is static idle -> skipped!
        assert_eq!(traj.chunk_states.len(), 12);
        assert_eq!(traj.idle_skipped_ticks, 1);
        // Un-ramped Kautz slot 10 snaps immediately on chunk 0.
        assert!((traj.chunk_states[0][10] - 0.5).abs() < 1.0e-12);
        // Ramped slot 0 ramps linearly: 0.2 -> 0.4 -> 0.6 -> 0.8 -> 1.0.
        assert!((traj.chunk_states[0][0] - 0.4).abs() < 1.0e-12);
        assert!((traj.chunk_states[1][0] - 0.6).abs() < 1.0e-12);
        assert!((traj.chunk_states[2][0] - 0.8).abs() < 1.0e-12);
        assert!((traj.chunk_states[3][0] - 1.0).abs() < 1.0e-12);

        // Central-FD trajectory adjoint verification.
        let d_chunks: Vec<[f64; GRA_RAMP_VEC_LEN]> = (0..12)
            .map(|c| core::array::from_fn(|i| (((c * 13 + i * 7 + 1) as f64) * 0.19).sin()))
            .collect();
        let d_rates = [0.7, -0.3, 1.1];
        let grads = pullback_micro_ramp_trajectory(&traj, &d_chunks, &d_rates);

        let eval_loss = |tks: &[GraRampTickInput],
                         ini: &[f64; GRA_RAMP_VEC_LEN],
                         ini_r: f64,
                         a_up: f64,
                         a_dn: f64|
         -> f64 {
            let tr = eval_micro_ramp_trajectory(tks, ini, ini_r, 44_100.0, a_up, a_dn);
            let s1: f64 = tr
                .chunk_states
                .iter()
                .zip(d_chunks.iter())
                .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>())
                .sum();
            let s2: f64 = tr
                .tick_lfo_rates
                .iter()
                .zip(d_rates.iter())
                .map(|(x, y)| x * y)
                .sum();
            s1 + s2
        };

        let eps = 1.0e-6_f64;
        for (t_idx, slot) in [(0, 0), (0, 10), (0, 31), (2, 0), (2, 14), (2, 55)] {
            let (mut tp, mut tm) = (ticks.clone(), ticks.clone());
            tp[t_idx].target[slot] += eps;
            tm[t_idx].target[slot] -= eps;
            let fd = (eval_loss(&tp, &init, 0.5, 0.6, 0.25)
                - eval_loss(&tm, &init, 0.5, 0.6, 0.25))
                / (2.0 * eps);
            let ana = grads.d_targets[t_idx][slot];
            assert!(
                (ana - fd).abs() < 1.0e-7 * ana.abs().max(1.0),
                "tick={t_idx} slot={slot}: ana={ana} fd={fd}"
            );
        }

        // Check init_state, init_lfo_rate, alpha_up, alpha_dn.
        let (mut ip, mut im) = (init, init);
        ip[0] += eps;
        im[0] -= eps;
        let fd_i0 = (eval_loss(&ticks, &ip, 0.5, 0.6, 0.25)
            - eval_loss(&ticks, &im, 0.5, 0.6, 0.25))
            / (2.0 * eps);
        assert!((grads.d_init_state[0] - fd_i0).abs() < 1.0e-7);

        let fd_up = (eval_loss(&ticks, &init, 0.5, 0.6 + eps, 0.25)
            - eval_loss(&ticks, &init, 0.5, 0.6 - eps, 0.25))
            / (2.0 * eps);
        assert!((grads.d_alpha_up - fd_up).abs() < 1.0e-7);

        let fd_dn = (eval_loss(&ticks, &init, 0.5, 0.6, 0.25 + eps)
            - eval_loss(&ticks, &init, 0.5, 0.6, 0.25 - eps))
            / (2.0 * eps);
        assert!((grads.d_alpha_dn - fd_dn).abs() < 1.0e-7);

        let fd_r0 = (eval_loss(&ticks, &init, 0.5 + eps, 0.6, 0.25)
            - eval_loss(&ticks, &init, 0.5 - eps, 0.6, 0.25))
            / (2.0 * eps);
        assert!((grads.d_init_lfo_rate - fd_r0).abs() < 1.0e-7);

        // Saturated `alpha_up` / `alpha_dn` on or outside `(1e-5, 1.0)` yield exact 0.0 gradient.
        for (a_up, a_dn) in [(1.0, 1.0), (1.5, -0.1), (1.0e-5, 1.0e-5)] {
            let tr_c = eval_micro_ramp_trajectory(&ticks, &init, 0.5, 44_100.0, a_up, a_dn);
            let gc = pullback_micro_ramp_trajectory(&tr_c, &d_chunks, &d_rates);
            assert_eq!(gc.d_alpha_up, 0.0);
            assert_eq!(gc.d_alpha_dn, 0.0);
        }
    }
}
