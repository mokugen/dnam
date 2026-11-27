#![forbid(unsafe_code)]

//! Shared `wgpu` Metal/hardware dispatch harness, `zerocopy` byte-slice casting, and `CoreQuad`
//! CPU/GPU parameter packing for `wgpu_hardware_parity` tests.

use dnam_kernels::{
    CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, SonaParaLruDuhemCoeffs4, SonaToneFoldedBlock0,
    TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
};
use sona_shaders::abi::{QUAD_PARAMS_VEC4_STRIDE, quad_param_slots};
use sona_shaders::compiler::ShaderPaths;
use sona_shaders::manifest::KernelId;
use std::fs;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use zerocopy::{FromBytes, IntoBytes};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ShaderVariant {
    Portable,
    Subgroups,
}

impl ShaderVariant {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Subgroups => "subgroups",
        }
    }

    #[must_use]
    pub fn uses_subgroups(self) -> bool {
        matches!(self, Self::Subgroups)
    }
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub supports_subgroups: bool,
    paths: ShaderPaths,
    pipelines: Mutex<[Option<wgpu::ComputePipeline>; 14]>,
}

fn pipeline_slot(id: KernelId, variant: ShaderVariant) -> usize {
    let k = match id {
        KernelId::PrescreenPopulation => 0,
        KernelId::FusedCoreScan256 => 1,
        KernelId::AdjointScan256 => 2,
        KernelId::TileMixFwdAdj => 3,
        KernelId::GaborBezoutGrad => 4,
        KernelId::OzakiGramRhsI32 => 5,
        KernelId::InnerOptStep => 6,
    };
    k * 2 + usize::from(variant.uses_subgroups())
}

impl GpuContext {
    #[must_use]
    pub fn load_wgsl(&self, id: KernelId, variant: ShaderVariant) -> String {
        let p = self
            .paths
            .wgsl_file_path(id.entry_point(), variant.uses_subgroups());
        fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("failed to read WGSL '{}': {e}", p.display()))
    }

    pub fn pipeline(&self, id: KernelId, variant: ShaderVariant) -> wgpu::ComputePipeline {
        let slot = pipeline_slot(id, variant);
        let mut guard = self
            .pipelines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(ref existing) = guard[slot] {
            return existing.clone();
        }
        let wgsl = self.load_wgsl(id, variant);
        let entry = id.entry_point();
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let built = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        guard[slot] = Some(built.clone());
        built
    }

    #[must_use]
    pub fn variants(&self) -> Vec<ShaderVariant> {
        if self.supports_subgroups {
            vec![ShaderVariant::Portable, ShaderVariant::Subgroups]
        } else {
            vec![ShaderVariant::Portable]
        }
    }
}

#[must_use]
pub fn init_gpu_context() -> Option<GpuContext> {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
    let adapter = match adapter {
        Ok(a) => a,
        Err(err) => {
            if cfg!(target_os = "macos") {
                panic!("Apple Metal GPU adapter must be present on macOS: {err:?}");
            }
            return None;
        }
    };
    assert!(
        adapter.limits().max_compute_workgroup_storage_size >= 16_384,
        "adapter max_compute_workgroup_storage_size < 16384"
    );
    let supports_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
    if cfg!(target_os = "macos") {
        assert!(
            supports_subgroups,
            "Apple Silicon Metal adapter must expose wgpu::Features::SUBGROUP"
        );
    }
    let required_features = if supports_subgroups {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sona-shaders-parity"),
        required_features,
        ..wgpu::DeviceDescriptor::default()
    }))
    .expect("hardware wgpu device request must succeed");

    Some(GpuContext {
        device,
        queue,
        supports_subgroups,
        paths: ShaderPaths::workspace_default(),
        pipelines: Mutex::new(std::array::from_fn(|_| None)),
    })
}

pub fn dispatch_kernel(
    ctx: &GpuContext,
    id: KernelId,
    variant: ShaderVariant,
    workgroups: (u32, u32, u32),
    uniform_bytes: &[u8],
    buffers: &[(bool, &[u8])],
) -> (Vec<Vec<u8>>, Duration) {
    let entry = id.entry_point();
    let pipeline = ctx.pipeline(id, variant);

    let uniform_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: uniform_bytes.len().max(32) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&uniform_buf, 0, uniform_bytes);

    let mut gpu_bufs = Vec::with_capacity(buffers.len());
    let mut rw_indices = Vec::new();
    for (idx, &(is_rw, data)) in buffers.iter().enumerate() {
        let size = (data.len().max(16) as u64 + 3) & !3;
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if is_rw {
            usage |= wgpu::BufferUsages::COPY_SRC;
            rw_indices.push((idx, size));
        }
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("storage"),
            size,
            usage,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            ctx.queue.write_buffer(&buf, 0, data);
        }
        gpu_bufs.push(buf);
    }

    let mut bind_entries = Vec::with_capacity(1 + gpu_bufs.len());
    bind_entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buf.as_entire_binding(),
    });
    for (idx, buf) in gpu_bufs.iter().enumerate() {
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (idx + 1) as u32,
            resource: buf.as_entire_binding(),
        });
    }
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &bind_entries,
    });

    let mut readback_bufs = Vec::with_capacity(rw_indices.len());
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
    let t0 = Instant::now();
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    for &(idx, size) in &rw_indices {
        let rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&gpu_bufs[idx], 0, &rb, 0, size);
        readback_bufs.push(rb);
    }
    ctx.queue.submit([encoder.finish()]);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let elapsed = t0.elapsed();

    let mut out = Vec::with_capacity(readback_bufs.len());
    for rb in readback_bufs {
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice
            .get_mapped_range()
            .expect("mapped buffer range must be readable")
            .to_vec();
        rb.unmap();
        out.push(data);
    }
    (out, elapsed)
}

#[must_use]
pub fn read_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::read_from_bytes(c).unwrap())
        .collect()
}

#[must_use]
pub fn read_i32_vec(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::read_from_bytes(c).unwrap())
        .collect()
}

#[must_use]
pub fn read_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::read_from_bytes(c).unwrap())
        .collect()
}

#[must_use]
pub fn read_vec2_vec(bytes: &[u8]) -> Vec<[f32; 2]> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| <[f32; 2]>::read_from_bytes(c).unwrap())
        .collect()
}

#[must_use]
pub fn read_vec4_vec(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| <[f32; 4]>::read_from_bytes(c).unwrap())
        .collect()
}

pub fn build_core_quad_bundle(
    stem: usize,
    quad: usize,
) -> (
    CoreQuadV4ParamsStereo4<GpuVec4>,
    [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE],
) {
    build_core_quad_bundle_with_k_nl(stem, quad, [0.0; 4])
}

pub fn build_core_quad_bundle_with_k_nl(
    stem: usize,
    quad: usize,
    k_nl: [f32; 4],
) -> (
    CoreQuadV4ParamsStereo4<GpuVec4>,
    [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE],
) {
    let f = (stem * 2 + quad) as f32;
    let fc = [
        0.02 + 0.015 * f,
        0.07 + 0.012 * f,
        0.16 + 0.011 * f,
        0.28 + 0.010 * f,
    ];
    let g = GpuVec4::from_array(fc.map(prewarp_tan_pi_f_over_fs_f32));
    let k = GpuVec4::from_array([0.55 + 0.05 * f, 0.85, 1.15, 1.45]);
    let svf = TptSvfCoeffs4::from_gk(g, k);

    let tone = SonaToneFoldedBlock0 {
        c_lp: GpuVec4::from_array([0.60, 0.05, -0.50, -0.35]),
        c_bp: GpuVec4::from_array([0.25, 0.40, 0.30, -0.20]),
        c_kbp: GpuVec4::from_array([-0.10, -0.25, -0.45, -0.65]),
        c_u: GpuVec4::from_array([0.15, 0.20, 0.40, 0.80]),
    };

    let alpha_rel = GpuVec4::from_array([0.92, 0.86, 0.78, 0.68]);
    let alpha_atk = GpuVec4::from_array([0.25, 0.40, 0.55, 0.75]);
    let kappa = GpuVec4::from_array([0.15, 0.35, 0.60, 0.85]);
    let gamma_duhem = GpuVec4::from_array([0.12, 0.28, 0.45, 0.65]);
    let signed_mix = GpuVec4::from_array([0.20, 0.40, 0.55, 0.70]);
    let stereo_link = GpuVec4::from_array([0.08, 0.25, 0.45, 0.65]);
    let lru = SonaParaLruDuhemCoeffs4::from_clamped(
        alpha_rel,
        alpha_atk,
        kappa,
        gamma_duhem,
        signed_mix,
        stereo_link,
    );

    let two_svf_g = g.add(g);
    let svf_k_nl = GpuVec4::from_array(k_nl);
    let rect_sign = GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]);
    let g0 = GpuVec4::from_array([1.2, 0.4, 2.0, 0.8]);
    let g_inf = GpuVec4::from_array([0.6, 1.8, 0.5, 2.2]);
    let d_pre = GpuVec4::from_array([1.5, 3.0, 4.0, 0.5]);
    let se_w_sq = GpuVec4::from_array([1.0, 0.5, 0.25, 0.125]);
    let se_d_bus = GpuVec4::from_array([0.2, 0.8, 1.5, 2.5]);
    let adaa_drive = GpuVec4::from_array([1.15, 1.65, 2.10, 1.35]);
    let lru_bias_drive = GpuVec4::from_array([0.15, -0.22, 0.30, -0.18]);
    let lin_mix = GpuVec4::from_array([0.0, 0.20, 0.45, 0.80]);
    let wavefolder_mix = GpuVec4::from_array([0.0, 0.35, 0.70, 1.0]);
    let glu_mix = if quad.is_multiple_of(2) {
        GpuVec4::from_array([0.65, -0.45, 0.80, -0.55])
    } else {
        GpuVec4::from_array([-0.50, 0.70, -0.35, 0.45])
    };
    let out_gain = GpuVec4::from_array([1.0, -1.25, 1.50, -0.85]);

    let mut packed = [[0.0_f32; 4]; QUAD_PARAMS_VEC4_STRIDE];
    packed[quad_param_slots::SVF_A1] = svf.a1.to_array();
    packed[quad_param_slots::SVF_A2] = svf.a2.to_array();
    packed[quad_param_slots::SVF_A3] = svf.a3.to_array();
    packed[quad_param_slots::SVF_K] = svf.k.to_array();
    packed[quad_param_slots::TONE_C_LP] = tone.c_lp.to_array();
    packed[quad_param_slots::TONE_C_BP] = tone.c_bp.to_array();
    packed[quad_param_slots::TONE_C_KBP] = tone.c_kbp.to_array();
    packed[quad_param_slots::TONE_C_U] = tone.c_u.to_array();
    packed[quad_param_slots::LRU_ALPHA_REL] = lru.alpha_rel.to_array();
    packed[quad_param_slots::LRU_ALPHA_ATK] = lru.alpha_atk.to_array();
    packed[quad_param_slots::LRU_KAPPA] = lru.kappa.to_array();
    packed[quad_param_slots::LRU_GAMMA_DUHEM] = lru.gamma_duhem.to_array();
    packed[quad_param_slots::LRU_SIGNED_MIX] = lru.signed_mix.to_array();
    packed[quad_param_slots::LRU_STEREO_LINK] = lru.stereo_link.to_array();
    packed[quad_param_slots::SVF_G] = g.to_array();
    packed[quad_param_slots::SVF_K_NL] = svf_k_nl.to_array();
    packed[quad_param_slots::RECT_SIGN] = rect_sign.to_array();
    packed[quad_param_slots::G0] = g0.to_array();
    packed[quad_param_slots::G_INF] = g_inf.to_array();
    packed[quad_param_slots::D_PRE] = d_pre.to_array();
    packed[quad_param_slots::SE_W_SQ] = se_w_sq.to_array();
    packed[quad_param_slots::SE_D_BUS] = se_d_bus.to_array();
    packed[quad_param_slots::ADAA_DRIVE] = adaa_drive.to_array();
    packed[quad_param_slots::LRU_BIAS_DRIVE] = lru_bias_drive.to_array();
    packed[quad_param_slots::LIN_MIX] = lin_mix.to_array();
    packed[quad_param_slots::WAVEFOLDER_MIX] = wavefolder_mix.to_array();
    packed[quad_param_slots::GLU_MIX] = glu_mix.to_array();
    packed[quad_param_slots::OUT_GAIN] = out_gain.to_array();

    let cpu_params = CoreQuadV4ParamsStereo4 {
        svf,
        tone,
        lru,
        two_svf_g,
        svf_k_nl,
        rect_sign,
        g0,
        g_inf,
        d_pre,
        se_w_sq,
        se_d_bus,
        adaa_drive,
        lru_bias_drive,
        lin_mix,
        wavefolder_mix,
        glu_mix,
        out_gain,
    };
    (cpu_params, packed)
}

#[must_use]
pub fn cpu_params_from_packed(packed: &[[f32; 4]]) -> CoreQuadV4ParamsStereo4<GpuVec4> {
    let g = GpuVec4::from_array(packed[quad_param_slots::SVF_G]);
    let k = GpuVec4::from_array(packed[quad_param_slots::SVF_K]);
    let svf = TptSvfCoeffs4::from_gk(g, k);
    let tone = SonaToneFoldedBlock0 {
        c_lp: GpuVec4::from_array(packed[quad_param_slots::TONE_C_LP]),
        c_bp: GpuVec4::from_array(packed[quad_param_slots::TONE_C_BP]),
        c_kbp: GpuVec4::from_array(packed[quad_param_slots::TONE_C_KBP]),
        c_u: GpuVec4::from_array(packed[quad_param_slots::TONE_C_U]),
    };
    let lru = SonaParaLruDuhemCoeffs4::from_clamped(
        GpuVec4::from_array(packed[quad_param_slots::LRU_ALPHA_REL]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_ALPHA_ATK]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_KAPPA]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_GAMMA_DUHEM]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_SIGNED_MIX]),
        GpuVec4::from_array(packed[quad_param_slots::LRU_STEREO_LINK]),
    );
    CoreQuadV4ParamsStereo4 {
        svf,
        tone,
        lru,
        two_svf_g: g.add(g),
        svf_k_nl: GpuVec4::from_array(packed[quad_param_slots::SVF_K_NL]),
        rect_sign: GpuVec4::from_array(packed[quad_param_slots::RECT_SIGN]),
        g0: GpuVec4::from_array(packed[quad_param_slots::G0]),
        g_inf: GpuVec4::from_array(packed[quad_param_slots::G_INF]),
        d_pre: GpuVec4::from_array(packed[quad_param_slots::D_PRE]),
        se_w_sq: GpuVec4::from_array(packed[quad_param_slots::SE_W_SQ]),
        se_d_bus: GpuVec4::from_array(packed[quad_param_slots::SE_D_BUS]),
        adaa_drive: GpuVec4::from_array(packed[quad_param_slots::ADAA_DRIVE]),
        lru_bias_drive: GpuVec4::from_array(packed[quad_param_slots::LRU_BIAS_DRIVE]),
        lin_mix: GpuVec4::from_array(packed[quad_param_slots::LIN_MIX]),
        wavefolder_mix: GpuVec4::from_array(packed[quad_param_slots::WAVEFOLDER_MIX]),
        glu_mix: GpuVec4::from_array(packed[quad_param_slots::GLU_MIX]),
        out_gain: GpuVec4::from_array(packed[quad_param_slots::OUT_GAIN]),
    }
}

#[must_use]
pub fn as_byte_slice<T: IntoBytes + zerocopy::Immutable>(slice: &[T]) -> &[u8] {
    slice.as_bytes()
}

pub struct Adjoint3PassResult {
    pub du: Vec<[f32; 4]>,
    pub param_grads: Vec<[f32; 4]>,
    pub exit_boundary: Vec<[f32; 4]>,
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_adjoint_3pass(
    ctx: &GpuContext,
    variant: ShaderVariant,
    num_frames: usize,
    num_tiles: usize,
    num_quads: usize,
    num_stems: usize,
    wg_x: usize,
    block_idx: u32,
    soft_rect_eps: f32,
    tiny_crossing_eps: f32,
    core_inputs: &[u8],
    quad_params: &[u8],
    tile_checkpoints: &[u8],
    d_loss_d_features: &[u8],
    feature_outputs: &[u8],
    init_exit_adj_boundary: Option<&[u8]>,
) -> Adjoint3PassResult {
    use sona_shaders::abi::{
        AdjointScanUniforms, EXIT_ADJ_BOUNDARY_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE,
        TILE_ADJ_BOUNDARY_VEC4_STRIDE,
    };

    let mut du_buf = vec![0_u8; core_inputs.len()];
    let mut pgrad_buf =
        vec![0_u8; num_stems * num_quads * num_tiles * QUAD_PARAMS_VEC4_STRIDE * 16];
    let exit_bytes = num_stems * wg_x * EXIT_ADJ_BOUNDARY_VEC4_STRIDE * 16;
    let tile_bytes = num_stems * wg_x * num_tiles * TILE_ADJ_BOUNDARY_VEC4_STRIDE * 16;
    let mut adj_bound_buf = vec![0_u8; exit_bytes + tile_bytes];
    if let Some(seed) = init_exit_adj_boundary {
        assert!(seed.len() >= exit_bytes);
        adj_bound_buf[..exit_bytes].copy_from_slice(&seed[..exit_bytes]);
    }

    for (pass_mode, wg_y) in [(0_u32, num_tiles as u32), (1, 1), (2, num_tiles as u32)] {
        let u = AdjointScanUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_quads: num_quads as u32,
            num_stems: num_stems as u32,
            pass_mode,
            soft_rect_eps,
            tiny_crossing_eps,
            block_idx,
        };
        let (out, _) = dispatch_kernel(
            ctx,
            KernelId::AdjointScan256,
            variant,
            (wg_x as u32, wg_y, num_stems as u32),
            u.as_bytes(),
            &[
                (false, core_inputs),
                (false, quad_params),
                (false, tile_checkpoints),
                (false, d_loss_d_features),
                (true, &du_buf),
                (true, &pgrad_buf),
                (true, &adj_bound_buf),
                (false, feature_outputs),
            ],
        );
        du_buf = out[0].clone();
        pgrad_buf = out[1].clone();
        adj_bound_buf = out[2].clone();
    }

    Adjoint3PassResult {
        du: read_vec4_vec(&du_buf),
        param_grads: read_vec4_vec(&pgrad_buf),
        exit_boundary: read_vec4_vec(&adj_bound_buf[..exit_bytes]),
    }
}
