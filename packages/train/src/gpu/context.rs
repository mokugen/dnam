//! Native `wgpu 30.0.2` adapter/device initialization, `< 1 ms` startup probes (`pow2_ceil_fp32`
//! plus hardware `ozaki_gram_rhs_i32` 0-ULP check and `>= 16,384 B` SRAM limit check), automatic
//! `Subgroups -> Portable` fallback, and compute pipeline cache (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::fmt;
#[cfg(target_arch = "wasm32")]
use std::cell::OnceCell;
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use dnam_kernels::{accum_ozaki_chunk_i32, pow2_ceil_fp32, reconstruct_ozaki_dot_f64};
use zerocopy::IntoBytes;

use super::SharedGpuRef;
use super::abi::{
    MAX_HARDWARE_WORKGROUP_SRAM_BYTES, OZAKI_PAIR_TILE_ROWS_DEFAULT, OZAKI_SPLIT64_WORDS_PER_PAIR,
    OZAKI_WORDS_PER_PAIR, OzakiGramUniforms,
};
use super::dispatch::{
    as_bytes, create_storage_buffer, create_uniform_buffer, read_u32_vec, submit_and_readback,
};
use super::shaders::{KernelId, ShaderVariant, embedded_wgsl, verify_embedded_shader_digests};
use crate::error::TrainError;
use crate::planner::MAX_WEBGPU_BINDING_BYTES;

#[cfg(not(target_arch = "wasm32"))]
static SHARED_WGPU_CONTEXT: OnceLock<Result<SharedGpuRef<WgpuContext>, &'static str>> =
    OnceLock::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED_WGPU_CONTEXT: OnceCell<Result<SharedGpuRef<WgpuContext>, &'static str>> =
        const { OnceCell::new() };
}

/// Cached native `wgpu 30.0.2` device, queue, verified shader variant, and compute pipelines.
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    variant: ShaderVariant,
    supports_subgroups: bool,
    max_workgroup_storage_bytes: u32,
    max_storage_binding_bytes: u64,
    probe_elapsed: Duration,
    pipelines: Mutex<[Option<wgpu::ComputePipeline>; 7]>,
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
        KernelId::TileMixFwdAdj => 3,
        KernelId::GaborBezoutGrad => 4,
        KernelId::OzakiGramRhsI32 => 5,
        KernelId::InnerOptStep => 6,
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

fn ozaki_split64_expected_limbs(chunks: &[i32; OZAKI_WORDS_PER_PAIR]) -> [u32; 4] {
    let hi64 =
        ((i64::from(chunks[0]) << 16) + (i64::from(chunks[1]) << 8) + i64::from(chunks[2])) as u64;
    let lo64 = ((i64::from(chunks[3]) << 8) + i64::from(chunks[4])) as u64;
    [
        hi64 as u32,
        (hi64 >> 32) as u32,
        lo64 as u32,
        (lo64 >> 32) as u32,
    ]
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
    let num_pairs = 5_usize;
    let split64_words = num_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let total_words = split64_words + num_pairs * OZAKI_WORDS_PER_PAIR;
    let u_pass0 = OzakiGramUniforms {
        num_frames: frames as u32,
        feature_dim: 2,
        target_channels: 1,
        chunk_len: 1024,
        num_chunks: 1,
        pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
        reduce_pass: 0,
        _pad: 0,
    };
    let u_pass1 = OzakiGramUniforms {
        reduce_pass: 1,
        ..u_pass0
    };
    let out_init = vec![0_i32; total_words];
    let b_feat = create_storage_buffer(
        device,
        queue,
        "probe_feat",
        as_bytes(&feat),
        false,
        max_binding_bytes,
    )?;
    let b_tgt = create_storage_buffer(
        device,
        queue,
        "probe_tgt",
        as_bytes(&tgt),
        false,
        max_binding_bytes,
    )?;
    let b_scales = create_storage_buffer(
        device,
        queue,
        "probe_scales",
        as_bytes(&scales),
        false,
        max_binding_bytes,
    )?;
    let b_buckets = create_storage_buffer(
        device,
        queue,
        "probe_buckets",
        as_bytes(&out_init),
        true,
        max_binding_bytes,
    )?;
    let bgl = pipeline.get_bind_group_layout(0);
    let make_bg = |u_buf: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("startup_probe_ozaki_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_feat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_tgt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_buckets.as_entire_binding(),
                },
            ],
        })
    };
    let u_buf0 = create_uniform_buffer(device, "probe_u0", u_pass0.as_bytes());
    let u_buf1 = create_uniform_buffer(device, "probe_u1", u_pass1.as_bytes());
    let bg0 = make_bg(&u_buf0);
    let bg1 = make_bg(&u_buf1);

    let readback_bytes = (split64_words * size_of::<u32>()) as u64;
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_rb_80b"),
        size: readback_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("startup_probe_ozaki"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("startup_probe_pass0"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bg0, &[]);
        cpass.dispatch_workgroups(num_pairs as u32, 1, 1);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("startup_probe_pass1"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bg1, &[]);
        cpass.dispatch_workgroups(num_pairs as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&b_buckets, 0, &rb, 0, readback_bytes);

    #[cfg(not(target_arch = "wasm32"))]
    let t0 = Instant::now();
    let out = submit_and_readback(device, queue, encoder, vec![(rb, readback_bytes as usize)])?;
    #[cfg(not(target_arch = "wasm32"))]
    let elapsed = t0.elapsed();
    #[cfg(target_arch = "wasm32")]
    let elapsed = Duration::ZERO;

    let got = read_u32_vec(&out[0])?;
    let expected = accum_ozaki_chunk_i32(&feat[..frames], &feat[frames..], isx, 1.0);
    let expected_limbs = ozaki_split64_expected_limbs(&expected);
    if got[OZAKI_SPLIT64_WORDS_PER_PAIR..2 * OZAKI_SPLIT64_WORDS_PER_PAIR] != expected_limbs {
        return Err(TrainError::GpuUnavailable(
            "ozaki split64 startup probe 0-ULP check failed",
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
    /// Returns the process-wide shared `WgpuContext`, initializing on first use so repeated solves
    /// avoid re-creating GPU devices and recompiling shaders (`ref[train.pipeline.webgpu-scans-and-ozaki]`).
    pub fn shared_or_init() -> Result<SharedGpuRef<Self>, TrainError> {
        let init = || match Self::try_new() {
            Ok(ctx) => Ok(SharedGpuRef::new(ctx)),
            Err(TrainError::GpuUnavailable(msg)) => Err(msg),
            Err(_) => Err("wgpu shared initialization failed"),
        };
        let to_result = |cached: &Result<SharedGpuRef<Self>, &'static str>| match cached {
            Ok(ctx) => Ok(SharedGpuRef::clone(ctx)),
            Err(msg) => Err(TrainError::GpuUnavailable(msg)),
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            to_result(SHARED_WGPU_CONTEXT.get_or_init(init))
        }
        #[cfg(target_arch = "wasm32")]
        {
            SHARED_WGPU_CONTEXT.with(|cell| to_result(cell.get_or_init(init)))
        }
    }

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
        let mut slots: [Option<wgpu::ComputePipeline>; 7] =
            [None, None, None, None, None, None, None];
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
