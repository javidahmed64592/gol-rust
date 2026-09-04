use crate::gpu_context::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform uploaded once; `toroidal` and grid dimensions are exposed for
/// Phase 4 runtime toggling via `queue.write_buffer`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub grid_w: u32,
    pub grid_h: u32,
    /// 1 = toroidal wrap, 0 = fixed boundary.
    pub toroidal: u32,
    /// Bitmask: bit N set → dead cell with N neighbours is born.
    pub birth_mask: u32,
    /// Bitmask: bit N set → live cell with N neighbours survives.
    pub survive_mask: u32,
    _pad: [u32; 3], // pad to 32 bytes for uniform buffer alignment
}

/// GPU simulation state: two ping-pong storage buffers, compute pipeline,
/// and pre-built bind groups for both compute and render passes.
///
/// After each `step()` call the internal `front` index flips; the renderer
/// always reads through `display_bind_group()`.
pub struct GpuSim {
    cell_bufs: [wgpu::Buffer; 2],
    #[allow(dead_code)] // bind groups hold a GPU ref; this handle prevents premature deallocation
    params_buf: wgpu::Buffer,
    front: usize,
    compute_pipeline: wgpu::ComputePipeline,
    // compute_bgs[i] reads cell_bufs[i], writes cell_bufs[1-i]
    compute_bgs: [wgpu::BindGroup; 2],
    // render_bgs[i] exposes cell_bufs[i] to the fragment shader
    render_bgs: [wgpu::BindGroup; 2],
    pub grid_w: u32,
    pub grid_h: u32,
}

impl GpuSim {
    pub fn new(
        ctx: &GpuContext,
        render_cell_bg_layout: &wgpu::BindGroupLayout,
        grid_w: u32,
        grid_h: u32,
        toroidal: bool,
        birth_mask: u32,
        survive_mask: u32,
    ) -> Self {
        // 2 f32 per cell: x = state, y = age (matches vec2<f32> in WGSL)
        let cell_size = (grid_w * grid_h) as u64 * 2 * std::mem::size_of::<f32>() as u64;

        let cell_bufs = [
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cells_a"),
                size: cell_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cells_b"),
                size: cell_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        // GoL defaults: B3/S23
        let params = Params {
            grid_w,
            grid_h,
            toroidal: toroidal as u32,
            birth_mask,
            survive_mask,
            _pad: [0; 3],
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sim_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compute bind group layout:
        //   binding 0 — Params uniform
        //   binding 1 — cells_in  (read-only storage)
        //   binding 2 — cells_out (read-write storage)
        let compute_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // compute_bgs[i] reads cell_bufs[i], writes cell_bufs[1-i]
        let compute_bg_0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg_0"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_bufs[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_bufs[1].as_entire_binding(),
                },
            ],
        });
        let compute_bg_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg_1"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_bufs[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_bufs[0].as_entire_binding(),
                },
            ],
        });

        // render_bgs[i] lets the fragment shader read cell_bufs[i]
        let render_bg_0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg_0"),
            layout: render_cell_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_bufs[0].as_entire_binding(),
                },
            ],
        });
        let render_bg_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg_1"),
            layout: render_cell_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_bufs[1].as_entire_binding(),
                },
            ],
        });

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shaders/sim.wgsl"));
        let compute_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });
        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gol_compute"),
                    layout: Some(&compute_layout),
                    module: &shader,
                    entry_point: "cs_main",
                    compilation_options: Default::default(),
                });

        Self {
            cell_bufs,
            params_buf,
            front: 0,
            compute_pipeline,
            compute_bgs: [compute_bg_0, compute_bg_1],
            render_bgs: [render_bg_0, render_bg_1],
            grid_w,
            grid_h,
        }
    }

    /// Copy CPU cell data into the current front buffer (call after seed/reset).
    pub fn upload_cells(&self, queue: &wgpu::Queue, cells: &[f32]) {
        queue.write_buffer(&self.cell_bufs[self.front], 0, bytemuck::cast_slice(cells));
    }

    /// Dispatch one GoL generation on the GPU, then flip front/back.
    ///
    /// The submitted compute work is guaranteed to complete before any render
    /// work submitted afterward on the same `queue`.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim_step"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gol_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bgs[self.front], &[]);
            cpass.dispatch_workgroups(self.grid_w.div_ceil(8), self.grid_h.div_ceil(8), 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        self.front = 1 - self.front;
    }

    /// The bind group the renderer should use to display the current state.
    pub fn display_bind_group(&self) -> &wgpu::BindGroup {
        &self.render_bgs[self.front]
    }
}
