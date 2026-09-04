use crate::gpu_context::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Full simulation uniform, shared by both the compute and render shaders.
/// Layout: 16 × 4-byte fields = 64 bytes, aligned for GPU uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub grid_w: u32,
    pub grid_h: u32,
    /// 1 = toroidal wrap, 0 = fixed dead boundary.
    pub toroidal: u32,
    /// 0 = discrete B/S GoL, 1 = SmoothLife continuous rule.
    pub mode: u32,
    /// Bitmask: bit N set → dead cell with N neighbours is born (GoL mode).
    pub birth_mask: u32,
    /// Bitmask: bit N set → live cell with N neighbours survives (GoL mode).
    pub survive_mask: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    /// Inner disk radius for self-density integral m (SmoothLife mode).
    pub inner_radius: f32,
    /// Outer ring radius for neighbour-density integral n (SmoothLife mode).
    pub outer_radius: f32,
    /// Lower threshold of the birth interval.
    pub birth_lo: f32,
    /// Upper threshold of the birth interval.
    pub birth_hi: f32,
    /// Lower threshold of the survival interval.
    pub survive_lo: f32,
    /// Upper threshold of the survival interval.
    pub survive_hi: f32,
    /// Logistic sigmoid steepness for smooth rule transitions.
    pub sigmoid_sharpness: f32,
    /// Energy level considered "alive" for age / birth tracking.
    pub age_threshold: f32,
}

/// GPU simulation state: two ping-pong storage buffers, compute pipeline,
/// and pre-built bind groups for both compute and render passes.
///
/// After each `step()` call the internal `front` index flips; the renderer
/// always reads through `display_bind_group()`.
pub struct GpuSim {
    cell_bufs: [wgpu::Buffer; 2],
    params_buf: wgpu::Buffer,
    params_cpu: Params, // CPU-side copy for runtime updates
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
        init_params: Params,
    ) -> Self {
        let grid_w = init_params.grid_w;
        let grid_h = init_params.grid_h;
        // 4 f32 per cell: energy, age, hue, _pad (matches vec4<f32> in WGSL)
        let cell_size = (grid_w * grid_h) as u64 * 4 * std::mem::size_of::<f32>() as u64;

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

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sim_params"),
                contents: bytemuck::bytes_of(&init_params),
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
            params_cpu: init_params,
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

    /// Switch between discrete GoL (0) and SmoothLife (1) at runtime.
    pub fn set_mode(&mut self, queue: &wgpu::Queue, mode: u32) {
        self.params_cpu.mode = mode;
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params_cpu));
    }

    #[allow(dead_code)]
    pub fn mode(&self) -> u32 {
        self.params_cpu.mode
    }

    /// Dispatch one simulation tick on the GPU, then flip front/back.
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
