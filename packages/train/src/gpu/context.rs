//! Native `wgpu 30.0.2` adapter/device initialization, `< 1 ms` startup probes (`pow2_ceil_fp32`
//! plus hardware `ozaki_gram_rhs_i32` 0-ULP check and `>= 16,384 B` SRAM limit check), automatic
//! `Subgroups -> Portable` fallback, and compute pipeline cache (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use dnam_kernels::{accum_ozaki_chunk_i32, pow2_ceil_fp32, reconstruct_ozaki_dot_f64};
use zerocopy::IntoBytes;

use super::abi::{MAX_HARDWARE_WORKGROUP_SRAM_BYTES, OZAKI_WORDS_PER_PAIR, OzakiGramUniforms};
use super::dispatch::{
    SinglePassDispatch, StorageBinding, as_bytes, dispatch_single_pass, read_i32_vec,
};
use super::shaders::{KernelId, ShaderVariant, embedded_wgsl, verify_embedded_shader_digests};
use crate::error::TrainError;
use crate::planner::MAX_WEBGPU_BINDING_BYTES;

/// Cached native `wgpu 30.0.2` device, queue, verified shader variant, and compute pipelines.
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    variant: ShaderVariant,
    supports_subgroups: bool,
    max_workgroup_storage_bytes: u32,
    max_storage_binding_bytes: u64,
    probe_elapsed: Duration,
    pipelines: Mutex<[Option<wgpu::ComputePipeline>; 6]>,
}

impl fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuContext")
            .field("variant", &self.variant)
            .field("supports_subgroups", &self.supports_subgroups)
            .field(
                "max_workgroup_storage_bytes",
                &self.max_workgroup_storage_bytes,
            )
            .field("max_storage_binding_bytes", &self.max_storage_binding_bytes)
            .field("probe_elapsed", &self.probe_elapsed)
            .finish_non_exhaustive()
    }
}

fn kernel_index(id: KernelId) -> usize {
    match id {
        KernelId::PrescreenPopulation => 0,
        KernelId::FusedCoreScan256 => 1,
        KernelId::AdjointScan256 => 2,
        KernelId::GaborBezoutGrad => 3,
        KernelId::OzakiGramRhsI32 => 4,
        KernelId::InnerOptStep => 5,
    }
}

fn verify_pow2_ceil_startup_probe() -> Result<(), TrainError> {
    for &x in &[0.0_f32, -0.0, 1.0e-3, 1.0, 1.5, 1.0e4] {
        let (scale, inv) = pow2_ceil_fp32(x);
        if !scale.is_finite() || !inv.is_finite() || scale <= 0.0 || inv <= 0.0 {
            return Err(TrainError::GpuUnavailable(
                "pow2_ceil_fp32 startup probe non-finite",
            ));
        }
        if x.abs() > 0.0 && (scale * inv - 1.0).abs() > 1.0e-6 {
            return Err(TrainError::GpuUnavailable(
                "pow2_ceil_fp32 inverse mismatch",
            ));
        }
    }
    Ok(())
}

fn run_hardware_ozaki_startup_probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    max_binding_bytes: u64,
) -> Result<Duration, TrainError> {
    verify_pow2_ceil_startup_probe()?;

    let frames = 128_usize;
    let mut feat = vec![0.0_f32; 2 * frames];
    let mut tgt = vec![0.0_f32; frames];
    for t in 0..frames {
        feat[t] = match t {
            0 => 1.0e4,
            1 => -1.0e4,
            2 => 1.0e-3,
            3 => f32::NAN,
            4 => f32::INFINITY,
            _ => ((t as f32) * 0.13).sin() * 0.5,
        };
        feat[frames + t] = 1.0;
        tgt[t] = if t == 2 { 1.0 } else { 0.0 };
    }
    let (sx, isx) = pow2_ceil_fp32(1.0e4);
    let scales = [[sx, isx], [1.0_f32, 1.0_f32], [1.0_f32, 0.0_f32]];
    let u = OzakiGramUniforms {
        num_frames: frames as u32,
        feature_dim: 2,
        target_channels: 1,
        chunk_len: 1024,
        num_chunks: 1,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let out_init = vec![0_i32; 5 * OZAKI_WORDS_PER_PAIR];
    let t0 = Instant::now();
    let out = dispatch_single_pass(
        device,
        queue,
        pipeline,
        &SinglePassDispatch {
            label: "startup_probe_ozaki",
            workgroups: (5, 1, 1),
            uniform_bytes: u.as_bytes(),
            bindings: &[
                StorageBinding {
                    writable: false,
                    bytes: as_bytes(&feat),
                },
                StorageBinding {
                    writable: false,
                    bytes: as_bytes(&tgt),
                },
                StorageBinding {
                    writable: false,
                    bytes: as_bytes(&scales),
                },
                StorageBinding {
                    writable: true,
                    bytes: as_bytes(&out_init),
                },
            ],
            max_binding_bytes,
        },
    )?;
    let elapsed = t0.elapsed();
    let got = read_i32_vec(&out[0])?;
    let expected = accum_ozaki_chunk_i32(&feat[..frames], &feat[frames..], isx, 1.0);
    if got[OZAKI_WORDS_PER_PAIR..2 * OZAKI_WORDS_PER_PAIR] != expected {
        return Err(TrainError::GpuUnavailable(
            "ozaki startup probe 0-ULP check failed",
        ));
    }
    let recon = reconstruct_ozaki_dot_f64(&expected, sx, 1.0);
    if !recon.is_finite() {
        return Err(TrainError::GpuUnavailable(
            "ozaki startup probe non-finite dot",
        ));
    }
    Ok(elapsed)
}

fn compile_pipeline(
    device: &wgpu::Device,
    id: KernelId,
    variant: ShaderVariant,
) -> wgpu::ComputePipeline {
    let wgsl = embedded_wgsl(id, variant);
    let entry = id.entry_point();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn request_device_for_variant(
    adapter: &wgpu::Adapter,
    variant: ShaderVariant,
) -> Result<(wgpu::Device, wgpu::Queue), TrainError> {
    let required_features = match variant {
        ShaderVariant::Subgroups => wgpu::Features::SUBGROUP,
        ShaderVariant::Portable => wgpu::Features::empty(),
    };
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("dnam-train-wgpu"),
        required_features,
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|_| TrainError::GpuUnavailable("wgpu request_device failed"))
}

impl WgpuContext {
    /// Initializes `wgpu` using the fastest verified shader variant (`Subgroups` when hardware
    /// supported and startup-probe clean, otherwise `Portable`), checking `>= 16,384 B` SRAM and
    /// running the `< 1 ms` `ozaki_gram_rhs_i32` startup probe (`ref[train.pipeline.webgpu-scans-and-ozaki]`).
    pub fn try_new() -> Result<Self, TrainError> {
        verify_embedded_shader_digests()?;
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|_| TrainError::GpuUnavailable("no WebGPU adapter available"))?;
        let supports_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        if supports_subgroups
            && let Ok(ctx) = Self::from_adapter(&adapter, ShaderVariant::Subgroups, true)
        {
            return Ok(ctx);
        }
        Self::from_adapter(&adapter, ShaderVariant::Portable, supports_subgroups)
    }

    /// Initializes `wgpu` pinned to an explicit [`ShaderVariant`] (`Portable` or `Subgroups`).
    pub fn try_new_with_variant(variant: ShaderVariant) -> Result<Self, TrainError> {
        verify_embedded_shader_digests()?;
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|_| TrainError::GpuUnavailable("no WebGPU adapter available"))?;
        let supports_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        if matches!(variant, ShaderVariant::Subgroups) && !supports_subgroups {
            return Err(TrainError::GpuUnavailable(
                "adapter lacks wgpu::Features::SUBGROUP",
            ));
        }
        Self::from_adapter(&adapter, variant, supports_subgroups)
    }

    fn from_adapter(
        adapter: &wgpu::Adapter,
        variant: ShaderVariant,
        supports_subgroups: bool,
    ) -> Result<Self, TrainError> {
        let limits = adapter.limits();
        if limits.max_compute_workgroup_storage_size < MAX_HARDWARE_WORKGROUP_SRAM_BYTES as u32 {
            return Err(TrainError::GpuUnavailable(
                "max_compute_workgroup_storage_size < 16,384 B",
            ));
        }
        let max_storage_binding_bytes = limits
            .max_storage_buffer_binding_size
            .min(MAX_WEBGPU_BINDING_BYTES);
        let (device, queue) = request_device_for_variant(adapter, variant)?;
        let ozaki_pipe = compile_pipeline(&device, KernelId::OzakiGramRhsI32, variant);
        let probe_elapsed = run_hardware_ozaki_startup_probe(
            &device,
            &queue,
            &ozaki_pipe,
            max_storage_binding_bytes,
        )?;
        let mut slots: [Option<wgpu::ComputePipeline>; 6] = [None, None, None, None, None, None];
        slots[kernel_index(KernelId::OzakiGramRhsI32)] = Some(ozaki_pipe);

        Ok(Self {
            device,
            queue,
            variant,
            supports_subgroups,
            max_workgroup_storage_bytes: limits.max_compute_workgroup_storage_size,
            max_storage_binding_bytes,
            probe_elapsed,
            pipelines: Mutex::new(slots),
        })
    }

    /// Returns the active [`ShaderVariant`].
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> ShaderVariant {
        self.variant
    }

    /// Returns whether the underlying adapter supports `wgpu::Features::SUBGROUP`.
    #[inline]
    #[must_use]
    pub const fn supports_subgroups(&self) -> bool {
        self.supports_subgroups
    }

    /// Returns the reported `max_compute_workgroup_storage_size` in bytes.
    #[inline]
    #[must_use]
    pub const fn max_workgroup_storage_bytes(&self) -> u32 {
        self.max_workgroup_storage_bytes
    }

    /// Returns the effective storage buffer binding cap in bytes.
    #[inline]
    #[must_use]
    pub const fn max_storage_binding_bytes(&self) -> u64 {
        self.max_storage_binding_bytes
    }

    /// Returns the measured duration of the hardware Ozaki startup probe dispatch.
    #[inline]
    #[must_use]
    pub const fn startup_probe_elapsed(&self) -> Duration {
        self.probe_elapsed
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn pipeline(&self, id: KernelId) -> wgpu::ComputePipeline {
        let idx = kernel_index(id);
        let mut guard = self
            .pipelines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(ref existing) = guard[idx] {
            return existing.clone();
        }
        let built = compile_pipeline(&self.device, id, self.variant);
        guard[idx] = Some(built.clone());
        built
    }
}
