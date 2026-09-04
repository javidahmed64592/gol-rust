use crate::gpu_context::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Intermediate offscreen texture format: linear HDR, avoids double gamma encoding.
const OFFSCREEN_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// HSV visual parameters uploaded once at startup from `[visuals]` in `gol.toml`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VisualParams {
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,
    pub start_hue: f32,
    pub end_hue: f32,
    pub max_lifetime: f32,
    pub sat_min: f32,
    pub sat_max: f32,
    pub val_min: f32,
    pub val_max: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for VisualParams {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

/// Gaussian blur parameters; updated at runtime via `set_blur_*`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlurParams {
    pub enabled: u32,
    pub radius: f32, // Gaussian sigma in pixels
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Owns the two-pass render pipeline:
///   Pass 1 — cells → Rgba16Float offscreen texture (bilinear + HSV coloring)
///   Pass 2 — Gaussian blur (or blit) → swapchain surface
pub struct GpuRenderer {
    // Pass 1 — cell render
    render_pipeline: wgpu::RenderPipeline,
    cell_bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    visual_params_buf: wgpu::Buffer,
    visual_bg: wgpu::BindGroup,
    // Offscreen intermediate texture
    #[allow(dead_code)]
    offscreen_texture: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    // Pass 2 — blur / blit
    blur_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    blur_sampler: wgpu::Sampler,
    blur_tex_bg: wgpu::BindGroup,
    blur_params_buf: wgpu::Buffer,
    blur_params_bg: wgpu::BindGroup,
    blur_enabled: bool,
    blur_radius: f32,
}

impl GpuRenderer {
    pub fn new(
        ctx: &GpuContext,
        visual_params: VisualParams,
        blur_enabled: bool,
        blur_radius: f32,
    ) -> Self {
        // --- Offscreen texture (RENDER_ATTACHMENT + TEXTURE_BINDING) ---
        let offscreen_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width: ctx.surface_config.width,
                height: ctx.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FMT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let offscreen_view = offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Cell render pipeline: group(0) sim params+cells; group(1) VisualParams ---
        let render_shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/render.wgsl"));

        let cell_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("render_cell_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let visual_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("visual_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let visual_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visual_params"),
                contents: bytemuck::bytes_of(&visual_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let visual_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visual_bg"),
            layout: &visual_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: visual_params_buf.as_entire_binding(),
            }],
        });

        let render_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_layout"),
                bind_group_layouts: &[&cell_bind_group_layout, &visual_bg_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render_pipeline"),
                layout: Some(&render_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: OFFSCREEN_FMT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        // --- Blur pipeline: group(0) texture+sampler; group(1) BlurParams ---
        let blur_shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/blur.wgsl"));

        let blur_tex_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blur_tex_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blur_params_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blur_params_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let blur_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blur_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blur_tex_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_tex_bg"),
            layout: &blur_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&offscreen_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blur_sampler),
                },
            ],
        });

        let initial_blur = BlurParams {
            enabled: blur_enabled as u32,
            radius: blur_radius,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let blur_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blur_params"),
                contents: bytemuck::bytes_of(&initial_blur),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let blur_params_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_params_bg"),
            layout: &blur_params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: blur_params_buf.as_entire_binding(),
            }],
        });

        let blur_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blur_layout"),
                bind_group_layouts: &[&blur_tex_bgl, &blur_params_bgl],
                push_constant_ranges: &[],
            });

        let blur_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blur_pipeline"),
                layout: Some(&blur_layout),
                vertex: wgpu::VertexState {
                    module: &blur_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blur_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.surface_config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Self {
            render_pipeline,
            cell_bind_group_layout,
            visual_params_buf,
            visual_bg,
            offscreen_texture,
            offscreen_view,
            blur_pipeline,
            blur_sampler,
            blur_tex_bg,
            blur_params_buf,
            blur_params_bg,
            blur_enabled,
            blur_radius,
        }
    }

    pub fn cell_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.cell_bind_group_layout
    }

    pub fn set_blur_enabled(&mut self, queue: &wgpu::Queue, enabled: bool) {
        self.blur_enabled = enabled;
        self.upload_blur_params(queue);
    }

    pub fn set_blur_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.blur_radius = radius;
        self.upload_blur_params(queue);
    }

    fn upload_blur_params(&self, queue: &wgpu::Queue) {
        let p = BlurParams {
            enabled: self.blur_enabled as u32,
            radius: self.blur_radius,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.blur_params_buf, 0, bytemuck::bytes_of(&p));
    }

    /// Two-pass render: cells → offscreen, then blur/blit → swapchain target.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        cells_bg: &wgpu::BindGroup,
    ) {
        // Pass 1 — bilinear cell coloring into the Rgba16Float offscreen texture.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cell_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, cells_bg, &[]);
            rpass.set_bind_group(1, &self.visual_bg, &[]);
            rpass.draw(0..4, 0..1);
        }

        // Pass 2 — Gaussian blur (or passthrough blit) into the swapchain surface.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&self.blur_pipeline);
            rpass.set_bind_group(0, &self.blur_tex_bg, &[]);
            rpass.set_bind_group(1, &self.blur_params_bg, &[]);
            rpass.draw(0..4, 0..1);
        }
    }
}
