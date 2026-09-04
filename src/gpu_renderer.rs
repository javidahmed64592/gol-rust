use crate::gpu_context::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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

/// Owns the render pipeline and the bind-group layout that `GpuSim` uses
/// to wire the cell storage buffers into the fragment shader.
pub struct GpuRenderer {
    render_pipeline: wgpu::RenderPipeline,
    cell_bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)] // GPU buffer kept alive via bind group ref
    visual_params_buf: wgpu::Buffer,
    visual_bg: wgpu::BindGroup,
}

impl GpuRenderer {
    pub fn new(ctx: &GpuContext, visual_params: VisualParams) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/render.wgsl"));

        // group(0): Params uniform + read-only cell storage buffer
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

        // group(1): VisualParams uniform — HSV coloring configuration
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

        let pipeline_layout = ctx
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
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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
        }
    }

    pub fn cell_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.cell_bind_group_layout
    }

    /// Record a full-screen render pass that reads `cells_bg` into the output view.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        cells_bg: &wgpu::BindGroup,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
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
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, cells_bg, &[]);
        rpass.set_bind_group(1, &self.visual_bg, &[]);
        rpass.draw(0..4, 0..1); // 4 vertices → TriangleStrip quad
    }
}
