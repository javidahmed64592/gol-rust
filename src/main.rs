mod config;
mod gpu_context;
mod gpu_renderer;
mod gpu_sim;
mod grid;
mod patterns;

use config::Config;
use gpu_context::GpuContext;
use gpu_renderer::{GpuRenderer, VisualParams};
use gpu_sim::GpuSim;
use grid::Grid;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

fn main() {
    env_logger::init(); // RUST_LOG=wgpu=warn cargo run  for GPU validation messages
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()).unwrap();
}

struct App {
    window: Option<Arc<Window>>,
    // Drop order: sim → renderer → context → window (declaration order).
    gpu: Option<(GpuContext, GpuRenderer, GpuSim)>,
    grid: Grid,
    paused: bool,
    last_tick: Instant,
    tick_interval: Duration,
    generation: u64,
    fps: f64,
    last_frame: Instant,
    cfg: Config,
}

impl App {
    fn new() -> Self {
        let cfg = Config::load();
        let mut grid = Grid::new(cfg.grid.width as usize, cfg.grid.height as usize);
        grid.seed_random(cfg.simulation.initial_density);
        Self {
            window: None,
            gpu: None,
            grid,
            paused: false,
            last_tick: Instant::now(),
            tick_interval: Duration::from_secs_f64(1.0 / cfg.simulation.ticks_per_second),
            generation: 0,
            fps: 0.0,
            last_frame: Instant::now(),
            cfg,
        }
    }

    fn reset_to(&mut self, seed: impl FnOnce(&mut Grid)) {
        self.grid.clear();
        seed(&mut self.grid);
        self.generation = 0;
        if let Some((ctx, _, sim)) = &self.gpu {
            sim.upload_cells(&ctx.queue, &self.grid.cells);
        }
    }

    fn step_once(&mut self) {
        if !self.paused {
            return;
        }
        if let Some((ctx, _, sim)) = &mut self.gpu {
            sim.step(&ctx.device, &ctx.queue);
            self.generation += 1;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let win_w = self.cfg.window.width;
        let win_h = self.cfg.window.height;
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Conway's Game of Life")
                        .with_inner_size(LogicalSize::new(win_w, win_h))
                        .with_resizable(false),
                )
                .expect("window creation failed"),
        );
        let ctx = pollster::block_on(GpuContext::new(Arc::clone(&window), win_w, win_h));
        let renderer = GpuRenderer::new(
            &ctx,
            VisualParams {
                bg_r: self.cfg.visuals.background_color[0],
                bg_g: self.cfg.visuals.background_color[1],
                bg_b: self.cfg.visuals.background_color[2],
                start_hue: self.cfg.visuals.start_hue,
                end_hue: self.cfg.visuals.end_hue,
                max_lifetime: self.cfg.visuals.max_lifetime,
                sat_min: self.cfg.visuals.sat_min,
                sat_max: self.cfg.visuals.sat_max,
                val_min: self.cfg.visuals.val_min,
                val_max: self.cfg.visuals.val_max,
                ..Default::default()
            },
        );
        let sim = GpuSim::new(
            &ctx,
            renderer.cell_bind_group_layout(),
            self.cfg.grid.width,
            self.cfg.grid.height,
            self.cfg.grid.toroidal,
            self.cfg.birth_mask(),
            self.cfg.survive_mask(),
        );
        sim.upload_cells(&ctx.queue, &self.grid.cells);
        self.window = Some(window);
        self.gpu = Some((ctx, renderer, sim));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => self.handle_key(code, event_loop),
            WindowEvent::RedrawRequested => self.render(event_loop),
            _ => {}
        }
    }
}

impl App {
    fn handle_key(&mut self, code: KeyCode, event_loop: &ActiveEventLoop) {
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::Enter => self.step_once(),
            KeyCode::KeyR => {
                let density = self.cfg.simulation.initial_density;
                self.reset_to(|g| g.seed_random(density));
            }
            KeyCode::KeyG => self.reset_to(|g| {
                g.seed_pattern(
                    patterns::GLIDER,
                    g.width as isize / 4,
                    g.height as isize / 4,
                );
            }),
            KeyCode::KeyB => self.reset_to(|g| {
                g.seed_pattern(
                    patterns::BLINKER,
                    g.width as isize / 2,
                    g.height as isize / 2,
                );
            }),
            KeyCode::KeyT => self.reset_to(|g| {
                g.seed_pattern(patterns::TOAD, g.width as isize / 2, g.height as isize / 2);
            }),
            KeyCode::KeyP => self.reset_to(|g| {
                g.seed_pattern(
                    patterns::R_PENTOMINO,
                    g.width as isize / 2,
                    g.height as isize / 2,
                );
            }),
            KeyCode::KeyC => self.reset_to(|g| {
                g.seed_pattern(
                    patterns::BEACON,
                    g.width as isize / 2,
                    g.height as isize / 2,
                );
            }),
            KeyCode::ArrowUp => {
                let tps = 1.0 / self.tick_interval.as_secs_f64();
                self.tick_interval = Duration::from_secs_f64(1.0 / (tps * 2.0).min(500.0));
            }
            KeyCode::ArrowDown => {
                let tps = 1.0 / self.tick_interval.as_secs_f64();
                self.tick_interval = Duration::from_secs_f64(1.0 / (tps / 2.0).max(0.5));
            }
            _ => {}
        }
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        // FPS: exponential moving average over recent frames.
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;
        if dt > 0.0 {
            let inst = 1.0 / dt;
            self.fps = if self.fps == 0.0 {
                inst
            } else {
                self.fps * 0.9 + inst * 0.1
            };
        }

        let Some((ctx, renderer, sim)) = &mut self.gpu else {
            return;
        };

        if !self.paused && self.last_tick.elapsed() >= self.tick_interval {
            sim.step(&ctx.device, &ctx.queue);
            self.generation += 1;
            self.last_tick = Instant::now();
        }

        let output = match ctx.surface.get_current_texture() {
            Ok(o) => o,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                ctx.reconfigure();
                return;
            }
            Err(e) => {
                eprintln!("surface error: {e}");
                event_loop.exit();
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        renderer.render(&mut encoder, &view, sim.display_bind_group());
        ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        let tps = (1.0 / self.tick_interval.as_secs_f64()).round() as u32;
        let rule = rule_notation(self.cfg.birth_mask(), self.cfg.survive_mask());
        if let Some(w) = &self.window {
            w.set_title(&format!(
                "GoL [{rule}]  Gen {gen}  {tps}tps  {state}\
                 \u{2003}  |  Space=Pause  Enter=Step  R=Random  ↑↓=TPS\
                 \u{2003}  |  G=Glider  B=Blinker  T=Toad  P=R-pentomino  C=Beacon",
                gen   = self.generation,
                state = if self.paused { "PAUSED" } else { "RUNNING" },
            ));
            w.request_redraw();
        }
    }
}

/// Format birth/survive bitmasks as standard B/S notation (e.g. "B3/S23").
fn rule_notation(birth: u32, survive: u32) -> String {
    let fmt = |mask: u32| -> String {
        (0u32..=8)
            .filter(|&n| (mask >> n) & 1 != 0)
            .map(|n| n.to_string())
            .collect()
    };
    format!("B{}/S{}", fmt(birth), fmt(survive))
}
