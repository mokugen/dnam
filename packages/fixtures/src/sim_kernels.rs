#![forbid(unsafe_code)]
//! Offline Physical ODE, Nodal Newton-Raphson & Partitioned FFT Simulation Kernels (`#![forbid(unsafe_code)]`).
//!
//! Uses [`nalgebra`] for continuous-to-discrete Tustin bilinear state-space discretization,
//! 4x4 orthogonal/Householder FDN reflections, and multi-node implicit nonlinear circuit solving,
//! paired with [`realfft`] for zero-latency uniform partitioned overlap-add cabinet convolution.

use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::{Matrix2, Matrix4, SMatrix, SVector, Vector2, Vector4};
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
    sanitize_sample_f64,
};

/// Unitary / orthogonal `4 x 4` FDN feedback matrix backed by [`nalgebra::Matrix4`].
#[derive(Clone, Copy, Debug)]
pub struct UnitaryHouseholder4 {
    matrix: Matrix4<f64>,
}

impl Default for UnitaryHouseholder4 {
    fn default() -> Self {
        Self::canonical_hadamard()
    }
}

impl UnitaryHouseholder4 {
    /// Canonical `4 x 4` Hadamard reflection `H = 0.5 * 1 1^T - I_4` (`H^T H = I_4`).
    #[must_use]
    pub fn canonical_hadamard() -> Self {
        Self {
            matrix: Matrix4::from_element(0.5) - Matrix4::identity(),
        }
    }

    /// Builds an orthogonal Householder reflection `I_4 - 2 (v v^T) / ||v||_2^2`.
    #[must_use]
    pub fn from_unit_normal(v: Vector4<f64>) -> Self {
        let norm_sq = v.norm_squared();
        if !norm_sq.is_finite() || norm_sq <= 1.0e-24 {
            return Self {
                matrix: Matrix4::identity(),
            };
        }
        Self {
            matrix: Matrix4::identity() - (2.0 / norm_sq) * (v * v.transpose()),
        }
    }

    /// Applies `y = H x`.
    #[inline]
    #[must_use]
    pub fn apply(&self, x: [f64; 4]) -> [f64; 4] {
        let y = self.matrix * Vector4::from(x);
        [y[0], y[1], y[2], y[3]]
    }

    /// Checks orthogonality `max(|H^T H - I_4|) <= tol`.
    #[must_use]
    pub fn is_orthogonal(&self, tol: f64) -> bool {
        let diff = self.matrix.transpose() * self.matrix - Matrix4::identity();
        diff.abs().max() <= tol
    }
}

/// Exact bilinear (Tustin) discretized `N`-state SISO state-space system via [`nalgebra`].
#[derive(Clone, Debug)]
pub struct DiscreteStateSpace<const N: usize> {
    a_d: SMatrix<f64, N, N>,
    b_d: SVector<f64, N>,
    c_d: SVector<f64, N>,
    d_d: f64,
    state: SVector<f64, N>,
}

impl<const N: usize> DiscreteStateSpace<N> {
    /// Discretizes `(A_c, B_c, C_c, D_c)` via Tustin transform at `sample_rate_hz` using `LU`.
    #[must_use]
    pub fn from_continuous_tustin(
        a_c: SMatrix<f64, N, N>,
        b_c: SVector<f64, N>,
        c_c: SVector<f64, N>,
        d_c: f64,
        sample_rate_hz: f64,
    ) -> Option<Self>
    where
        nalgebra::Const<N>: nalgebra::DimMin<nalgebra::Const<N>, Output = nalgebra::Const<N>>,
    {
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return None;
        }
        let dt = 1.0 / sample_rate_hz;
        let half_dt = 0.5 * dt;
        let eye = SMatrix::<f64, N, N>::identity();
        let m = eye - a_c * half_dt;
        let m_inv = m.lu().try_inverse()?;
        let a_d = (eye + a_c * half_dt) * m_inv;
        let b_d = (m_inv * b_c) * dt;
        let c_d = m_inv.transpose() * c_c;
        let d_d = d_c + half_dt * c_c.dot(&(m_inv * b_c));
        Some(Self {
            a_d,
            b_d,
            c_d,
            d_d,
            state: SVector::<f64, N>::zeros(),
        })
    }

    /// Zeros internal filter states to equilibrium.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Advances the discrete state-space model by one sample `u`.
    #[inline]
    pub fn step(&mut self, u: f64) -> f64 {
        let u_s = sanitize_sample_f64(u);
        let y = self.c_d.dot(&self.state) + self.d_d * u_s;
        self.state = self.a_d * self.state + self.b_d * u_s;
        y
    }
}

/// Coupled 2-node trapezoidal RC + antiparallel Shockley diode clipper solved via damped Newton (`nalgebra`).
#[derive(Clone, Debug)]
pub struct ImplicitNodalClipper2Node {
    g1: f64,
    g2: f64,
    gl: f64,
    g_c1: f64,
    g_c2: f64,
    is_sat: f64,
    v_t: f64,
    s_c1: f64,
    s_c2: f64,
    prev_v: [f64; 2],
}

impl ImplicitNodalClipper2Node {
    /// Creates a 2-node implicit clipper discretized via trapezoidal rule at `sample_rate_hz`.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let sr = if sample_rate_hz.is_finite() && sample_rate_hz >= 1_000.0 {
            sample_rate_hz
        } else {
            f64::from(DEFAULT_FIXTURE_SAMPLE_RATE_HZ)
        };
        let c1 = 4.7e-9_f64;
        let c2 = 10.0e-9_f64;
        Self {
            g1: 1.0 / 2_200.0,
            g2: 1.0 / 10_000.0,
            gl: 1.0 / 100_000.0,
            g_c1: 2.0 * c1 * sr,
            g_c2: 2.0 * c2 * sr,
            is_sat: 2.52e-9,
            v_t: 0.0517,
            s_c1: 0.0,
            s_c2: 0.0,
            prev_v: [0.0, 0.0],
        }
    }

    /// Resets trapezoidal capacitor companion memories to zero.
    pub fn reset(&mut self) {
        self.s_c1 = 0.0;
        self.s_c2 = 0.0;
        self.prev_v = [0.0, 0.0];
    }

    /// Solves the 2-node coupled KCL system for one sample with normalized `drive` (`0.0..=1.0`).
    #[inline]
    pub fn step(&mut self, v_in: f64, drive: f64) -> f64 {
        let drive_clamped = if drive.is_finite() {
            drive.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let u = sanitize_sample_f64(v_in).clamp(-10.0, 10.0) * (1.0 + 19.0 * drive_clamped);
        let a11 = self.g1 + self.g2 + self.g_c1;
        let a22_lin = self.g2 + self.gl + self.g_c2;
        let mut v = Vector2::new(self.prev_v[0], self.prev_v[1]);

        for _ in 0..12 {
            let arg = (v[1] / self.v_t).clamp(-24.0, 24.0);
            let i_d = 2.0 * self.is_sat * arg.sinh();
            let g_d = (2.0 * self.is_sat / self.v_t) * arg.cosh();
            let f1 = a11 * v[0] - self.g2 * v[1] - (self.g1 * u + self.s_c1);
            let f2 = -self.g2 * v[0] + a22_lin * v[1] + i_d - self.s_c2;
            let rhs = Vector2::new(-f1, -f2);
            if rhs.norm() <= 1.0e-11 {
                break;
            }
            let jac = Matrix2::new(a11, -self.g2, -self.g2, a22_lin + g_d);
            let delta = jac.lu().solve(&rhs).unwrap_or_else(Vector2::zeros);
            let max_step = delta[0].abs().max(delta[1].abs());
            let scale = if max_step > 0.15 {
                0.15 / max_step
            } else {
                1.0
            };
            v += delta * scale;
        }

        self.s_c1 = 2.0 * self.g_c1 * v[0] - self.s_c1;
        self.s_c2 = 2.0 * self.g_c2 * v[1] - self.s_c2;
        self.prev_v = [v[0], v[1]];
        v[1]
    }
}

/// Zero-latency uniform partitioned overlap-add FFT convolver backed by [`realfft`].
#[derive(Clone)]
pub struct FftConvolver {
    block_size: usize,
    ir_len: usize,
    head_ir: Vec<f64>,
    in_ring: Vec<f64>,
    ring_pos: usize,
    block_pos: usize,
    time_buf: Vec<f64>,
    tail_out: Vec<f64>,
    overlap_buf: Vec<f64>,
    tail_ir_specs: Vec<Vec<Complex<f64>>>,
    fdl: Vec<Vec<Complex<f64>>>,
    fdl_head: usize,
    spec_accum: Vec<Complex<f64>>,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    r2c_scratch: Vec<Complex<f64>>,
    c2r_scratch: Vec<Complex<f64>>,
}

impl core::fmt::Debug for FftConvolver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FftConvolver")
            .field("block_size", &self.block_size)
            .field("ir_len", &self.ir_len)
            .field("tail_partitions", &self.tail_ir_specs.len())
            .finish()
    }
}

impl FftConvolver {
    /// Creates a zero-latency uniform partitioned convolver for `ir` at power-of-two `block_size`.
    #[must_use]
    pub fn new(ir: &[f64], requested_block_size: usize) -> Self {
        let block_size = requested_block_size.max(16).next_power_of_two();
        let fft_len = block_size * 2;
        let bins = block_size + 1;
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_len);
        let c2r = planner.plan_fft_inverse(fft_len);
        let mut r2c_scratch = r2c.make_scratch_vec();
        let c2r_scratch = c2r.make_scratch_vec();

        let sanitized_ir: Vec<f64> = ir
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();
        let ir_len = sanitized_ir.len();
        let head_len = ir_len.min(block_size);
        let head_ir = sanitized_ir[..head_len].to_vec();

        let mut time_buf = vec![0.0_f64; fft_len];
        let mut tail_ir_specs = Vec::new();
        if ir_len > block_size {
            let num_tail = (ir_len - block_size).div_ceil(block_size);
            tail_ir_specs.reserve(num_tail);
            for p in 1..=num_tail {
                let start = p * block_size;
                let end = (start + block_size).min(ir_len);
                time_buf.fill(0.0);
                time_buf[..end - start].copy_from_slice(&sanitized_ir[start..end]);
                let mut spec = vec![Complex::new(0.0, 0.0); bins];
                let _ = r2c.process_with_scratch(&mut time_buf, &mut spec, &mut r2c_scratch);
                tail_ir_specs.push(spec);
            }
        }
        time_buf.fill(0.0);
        let num_tail = tail_ir_specs.len();

        Self {
            block_size,
            ir_len,
            head_ir,
            in_ring: vec![0.0_f64; block_size],
            ring_pos: 0,
            block_pos: 0,
            time_buf,
            tail_out: vec![0.0_f64; block_size],
            overlap_buf: vec![0.0_f64; block_size],
            tail_ir_specs,
            fdl: vec![vec![Complex::new(0.0, 0.0); bins]; num_tail],
            fdl_head: 0,
            spec_accum: vec![Complex::new(0.0, 0.0); bins],
            r2c,
            c2r,
            r2c_scratch,
            c2r_scratch,
        }
    }

    /// Active uniform partition block length (`>= 16`).
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Configured impulse response length in samples.
    #[must_use]
    pub const fn ir_len(&self) -> usize {
        self.ir_len
    }

    /// Resets convolution delay lines and overlap buffers to silence.
    pub fn reset(&mut self) {
        self.in_ring.fill(0.0);
        self.time_buf.fill(0.0);
        self.tail_out.fill(0.0);
        self.overlap_buf.fill(0.0);
        self.spec_accum.fill(Complex::new(0.0, 0.0));
        for slot in &mut self.fdl {
            slot.fill(Complex::new(0.0, 0.0));
        }
        self.ring_pos = 0;
        self.block_pos = 0;
        self.fdl_head = 0;
    }

    fn flush_block_fft(&mut self) {
        let b = self.block_size;
        let num_tail = self.tail_ir_specs.len();
        self.time_buf[b..2 * b].fill(0.0);
        let _ = self.r2c.process_with_scratch(
            &mut self.time_buf,
            &mut self.fdl[self.fdl_head],
            &mut self.r2c_scratch,
        );
        self.spec_accum.fill(Complex::new(0.0, 0.0));
        for (p, part_spec) in self.tail_ir_specs.iter().enumerate() {
            let slot = (self.fdl_head + num_tail - p) % num_tail;
            let x_spec = &self.fdl[slot];
            for (acc, (&xk, &hk)) in self
                .spec_accum
                .iter_mut()
                .zip(x_spec.iter().zip(part_spec.iter()))
            {
                *acc += xk * hk;
            }
        }
        self.fdl_head = (self.fdl_head + 1) % num_tail;
        self.spec_accum[0].im = 0.0;
        self.spec_accum[b].im = 0.0;
        let _ = self.c2r.process_with_scratch(
            &mut self.spec_accum,
            &mut self.time_buf,
            &mut self.c2r_scratch,
        );
        let norm = 1.0 / ((2 * b) as f64);
        for i in 0..b {
            self.tail_out[i] = self.time_buf[i] * norm + self.overlap_buf[i];
            self.overlap_buf[i] = self.time_buf[b + i] * norm;
        }
    }

    /// Processes a single input sample with zero sample latency.
    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let b = self.block_size;
        let mask = b - 1;
        let xs = sanitize_sample_f64(x);
        self.in_ring[self.ring_pos] = xs;
        self.time_buf[self.block_pos] = xs;

        let mut head = 0.0_f64;
        for (k, &hk) in self.head_ir.iter().enumerate() {
            let idx = (self.ring_pos + b - k) & mask;
            head += hk * self.in_ring[idx];
        }
        let y = head + self.tail_out[self.block_pos];

        self.ring_pos = (self.ring_pos + 1) & mask;
        self.block_pos += 1;
        if self.block_pos == b {
            self.block_pos = 0;
            if !self.tail_ir_specs.is_empty() {
                self.flush_block_fft();
            }
        }
        y
    }

    /// Processes a contiguous slice `input` into `output`.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        for (&src, dst) in input.iter().zip(output.iter_mut()) {
            *dst = self.process_sample(src);
        }
    }
}

const FFT_CAB_CONTROLS: [FixtureControlSpec; 3] = [
    FixtureControlSpec {
        id: "input_trim",
        name: "Input Trim",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
    FixtureControlSpec {
        id: "stereo_width",
        name: "Stereo Width",
        unit: "%",
        default_norm: 0.5,
        min_physical: 0.0,
        max_physical: 100.0,
    },
    FixtureControlSpec {
        id: "dry_wet",
        name: "Dry / Wet",
        unit: "%",
        default_norm: 1.0,
        min_physical: 0.0,
        max_physical: 100.0,
    },
];

const FFT_CAB_SWITCHES: [FixtureSwitchSpec; 1] = [FixtureSwitchSpec {
    id: "phase_invert",
    name: "Phase Invert",
    options: &["Normal", "Invert"],
    default_index: 0,
}];

fn synthetic_stereo_cabinet_ir(sample_rate_hz: u32) -> (Vec<f64>, Vec<f64>) {
    let sr = f64::from(clamp_sample_rate_hz(sample_rate_hz));
    let len = (sr * 0.032).round().clamp(128.0, 8192.0) as usize;
    let mut ir_l = vec![0.0_f64; len];
    let mut ir_r = vec![0.0_f64; len];
    let modes = [
        (110.0, 0.55, 22.0),
        (820.0, 0.30, 85.0),
        (2650.0, 0.15, 190.0),
    ];
    let tap_r_off = ((0.0008 * sr).round() as usize).clamp(1, len - 1);
    for n in 0..len {
        let t = (n as f64) / sr;
        let mut s_l = if n == 0 { 0.35 } else { 0.0 };
        let mut s_r = if n == tap_r_off { 0.32 } else { 0.0 };
        for &(f_hz, amp, damp) in &modes {
            let env = (-damp * t).exp();
            s_l += amp * env * (2.0 * PI * f_hz * t).cos();
            s_r += amp * 0.92 * env * (2.0 * PI * (f_hz * 1.03) * t + 0.25).cos();
        }
        ir_l[n] = s_l;
        ir_r[n] = s_r;
    }
    let scale_l = 0.85 / ir_l.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
    let scale_r = 0.85 / ir_r.iter().map(|v| v.abs()).sum::<f64>().max(1.0);
    for v in &mut ir_l {
        *v *= scale_l;
    }
    for v in &mut ir_r {
        *v *= scale_r;
    }
    (ir_l, ir_r)
}

/// Stereo `1 -> 2` FFT cabinet impulse-response black-box model backed by [`FftConvolver`].
#[derive(Clone)]
pub struct FftCabinetIrModel {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    block_size: usize,
    custom_ir: bool,
    convolver_l: FftConvolver,
    convolver_r: FftConvolver,
}

impl core::fmt::Debug for FftCabinetIrModel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FftCabinetIrModel")
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("bit_depth", &self.bit_depth)
            .field("block_size", &self.block_size)
            .field("convolver_l", &self.convolver_l)
            .field("convolver_r", &self.convolver_r)
            .finish()
    }
}

impl Default for FftCabinetIrModel {
    fn default() -> Self {
        Self::new()
    }
}

impl FftCabinetIrModel {
    /// Default stereo synthetic cabinet IR model at `48 kHz` / `Float32`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    /// Creates the synthetic stereo cabinet IR model at `(sample_rate_hz, bit_depth)`.
    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let (ir_l, ir_r) = synthetic_stereo_cabinet_ir(sr);
        let mut model = Self::with_stereo_ir(&ir_l, &ir_r, 64, sr, bit_depth);
        model.custom_ir = false;
        model
    }

    /// Wraps custom `(ir_l, ir_r)` impulse responses into a stereo [`BlackBoxModel`].
    #[must_use]
    pub fn with_stereo_ir(
        ir_l: &[f64],
        ir_r: &[f64],
        block_size: usize,
        sample_rate_hz: u32,
        bit_depth: FixtureBitDepth,
    ) -> Self {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let safe_l = if ir_l.is_empty() { &[1.0][..] } else { ir_l };
        let safe_r = if ir_r.is_empty() { safe_l } else { ir_r };
        let convolver_l = FftConvolver::new(safe_l, block_size);
        let b = convolver_l.block_size();
        Self {
            sample_rate_hz: sr,
            bit_depth,
            block_size: b,
            custom_ir: true,
            convolver_l,
            convolver_r: FftConvolver::new(safe_r, b),
        }
    }
}

impl BlackBoxModel for FftCabinetIrModel {
    fn reset(&mut self) {
        self.convolver_l.reset();
        self.convolver_r.reset();
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        let ir_ms = (self.convolver_l.ir_len() as f32) * 1000.0 / (self.sample_rate_hz as f32);
        FixtureDeviceMetadata {
            id: "Sim-FFT-Cabinet-IR-Stereo",
            name: "FFT Partitioned Cabinet & Room IR",
            category: "cabinet",
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            expected_tail_ms: ir_ms.max(25.0),
            supports_tempo_sync: false,
            controls: &FFT_CAB_CONTROLS,
            switches: &FFT_CAB_SWITCHES,
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        self.sample_rate_hz = sr;
        if !self.custom_ir {
            let (ir_l, ir_r) = synthetic_stereo_cabinet_ir(sr);
            self.convolver_l = FftConvolver::new(&ir_l, self.block_size);
            self.convolver_r = FftConvolver::new(&ir_r, self.block_size);
        } else {
            self.reset();
        }
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.bit_depth = bit_depth;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let trim_u = f64::from(sanitize_norm(controls.first().copied().unwrap_or(0.0), 0.0));
        let width_u = f64::from(sanitize_norm(controls.get(1).copied().unwrap_or(0.5), 0.5));
        let wet_u = f64::from(sanitize_norm(controls.get(2).copied().unwrap_or(1.0), 1.0));
        let sign = if switches.first().copied().unwrap_or(0) % 2 == 1 {
            -1.0
        } else {
            1.0
        };

        let in_raw = f64::from(inputs.first().copied().unwrap_or(0.0));
        let x_in = sanitize_sample_f64(in_raw) * 10.0_f64.powf(0.6 * trim_u) * sign;
        let wet_l = self.convolver_l.process_sample(x_in);
        let wet_r = self.convolver_r.process_sample(x_in);
        let mid = 0.5 * (wet_l + wet_r);
        let side = 0.5 * (wet_l - wet_r) * (0.5 * (width_u + 1.0));
        let mix = 0.5 * (wet_u + 1.0);
        let dry = x_in * (1.0 - mix);
        if let Some(o0) = outputs.first_mut() {
            *o0 = self.bit_depth.quantize_f64_to_f32(dry + mix * (mid + side));
        }
        if let Some(o1) = outputs.get_mut(1) {
            *o1 = self.bit_depth.quantize_f64_to_f32(dry + mix * (mid - side));
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_householder_and_tustin_state_space() {
        let h = UnitaryHouseholder4::default();
        assert!(h.is_orthogonal(1.0e-12));
        let n = Vector4::new(1.0, -2.0, 0.5, 1.5);
        let hr = UnitaryHouseholder4::from_unit_normal(n);
        assert!(hr.is_orthogonal(1.0e-12));
        let zero_hr = UnitaryHouseholder4::from_unit_normal(Vector4::zeros());
        assert_eq!(zero_hr.apply([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0]);

        // Second-order continuous lowpass with unit DC gain:
        // wc^2 / (s^2 + sqrt(2)*wc*s + wc^2)
        let wc = 2.0 * PI * 1_000.0;
        let a_c = Matrix2::new(0.0, 1.0, -wc * wc, -core::f64::consts::SQRT_2 * wc);
        let b_c = Vector2::new(0.0, wc * wc);
        let c_c = Vector2::new(1.0, 0.0);
        assert!(DiscreteStateSpace::from_continuous_tustin(a_c, b_c, c_c, 0.0, 0.0).is_none());
        let mut ss =
            DiscreteStateSpace::from_continuous_tustin(a_c, b_c, c_c, 0.0, 48_000.0).unwrap();
        let mut y = 0.0;
        for _ in 0..2_000 {
            y = ss.step(1.0);
        }
        assert!((y - 1.0).abs() < 1.0e-6);
        ss.reset();
        assert_eq!(ss.step(0.0), 0.0);
    }

    #[test]
    fn test_implicit_nodal_clipper_odd_symmetry_and_reset() {
        let mut a = ImplicitNodalClipper2Node::new(48_000.0);
        let mut b = ImplicitNodalClipper2Node::new(0.0);
        assert_eq!(b.step(0.0, 0.5), 0.0);
        let mut max_y = 0.0_f64;
        for i in 0..256 {
            let u = 0.8 * ((i as f64) * 0.13).sin();
            let ya = a.step(u, 0.85);
            let yb = b.step(-u, 0.85);
            assert!(ya.is_finite() && yb.is_finite());
            assert!((ya + yb).abs() < 1.0e-10);
            max_y = max_y.max(ya.abs());
        }
        assert!(max_y > 0.1 && max_y < 1.5);
        a.reset();
        assert_eq!(a.step(0.0, f64::NAN), 0.0);
    }

    #[test]
    fn test_fft_convolver_exact_parity_and_cabinet_model() {
        let ir: Vec<f64> = (0..180)
            .map(|k| (-0.03 * (k as f64)).exp() * ((k as f64) * 0.27).cos())
            .collect();
        let x: Vec<f64> = (0..260)
            .map(|n| 0.4 * ((n as f64) * 0.19).sin() - 0.15 * ((n as f64) * 0.41).cos())
            .collect();
        let mut direct = vec![0.0_f64; x.len()];
        for n in 0..x.len() {
            for k in 0..ir.len().min(n + 1) {
                direct[n] += ir[k] * x[n - k];
            }
        }
        let mut conv = FftConvolver::new(&ir, 32);
        assert_eq!(conv.block_size(), 32);
        assert_eq!(conv.ir_len(), 180);
        assert_ne!(format!("{conv:?}"), "");
        let mut fft_out = vec![0.0_f64; x.len()];
        conv.process_block(&x, &mut fft_out);
        for (d, f) in direct.iter().zip(&fft_out) {
            assert!((d - f).abs() < 1.0e-11);
        }
        conv.reset();
        assert_eq!(conv.process_sample(0.0), 0.0);

        let short_conv = FftConvolver::new(&[1.0, 0.5], 16);
        let mut sc = short_conv;
        assert!((sc.process_sample(2.0) - 2.0).abs() < 1.0e-12);
        assert!((sc.process_sample(0.0) - 1.0).abs() < 1.0e-12);

        let mut cab = FftCabinetIrModel::default();
        assert_eq!(cab.num_inputs(), 1);
        assert_eq!(cab.num_outputs(), 2);
        assert_eq!(
            cab.metadata().sample_rate_hz,
            DEFAULT_FIXTURE_SAMPLE_RATE_HZ
        );
        assert_ne!(format!("{cab:?}"), "");
        let mut out_a = [0.0_f32; 2];
        let mut out_b = [0.0_f32; 2];
        cab.process_frame(&[0.5], &mut out_a, &[0.0, 0.8, 1.0], &[0]);
        cab.set_sample_rate_hz(96_000);
        cab.set_bit_depth(FixtureBitDepth::Int24);
        cab.process_frame(&[0.5], &mut out_b, &[0.0, 0.8, 1.0], &[1]);
        assert!(out_a[0].is_finite() && out_b[0].is_finite());
        assert_eq!(out_a[0].signum(), -out_b[0].signum());

        let mut custom = FftCabinetIrModel::with_stereo_ir(
            &[],
            &[0.5, -0.25],
            16,
            44_100,
            FixtureBitDepth::Float32,
        );
        custom.set_sample_rate_hz(48_000);
        let mut cout = [0.0_f32; 2];
        custom.process_frame(&[0.25], &mut cout, &[0.0, 1.0, 1.0], &[0]);
        assert!(cout[0].is_finite() && cout[1].is_finite());
    }

    proptest! {
        #[test]
        fn prop_fft_convolver_matches_direct_convolution(
            ir in prop::collection::vec(-1.0_f64..1.0_f64, 1..150),
            x in prop::collection::vec(-1.0_f64..1.0_f64, 1..180),
        ) {
            let mut conv = FftConvolver::new(&ir, 32);
            for n in 0..x.len() {
                let mut ref_y = 0.0_f64;
                for k in 0..ir.len().min(n + 1) {
                    ref_y += ir[k] * x[n - k];
                }
                let y = conv.process_sample(x[n]);
                prop_assert!((y - ref_y).abs() <= 1.0e-10);
            }
        }
    }
}
