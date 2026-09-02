mod gpu_context;
mod gpu_renderer;
mod gpu_sim;
mod grid;
mod patterns;

use gpu_context::GpuContext;
use gpu_renderer::GpuRenderer;
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

// --- Configuration -------------------------------------------------------
// Change these constants to resize the grid or window; everything else
// (buffer allocation, dispatch count, UV mapping) scales automatically.
// Phase 4 will make these runtime-configurable.

const GRID_W: u32 = 256;
const GRID_H: u32 = 256;
/// Integer pixel scale factor (display pixels per cell edge).
const CELL_PX: u32 = 3;
const WIN_W: u32 = GRID_W * CELL_PX;
const WIN_H: u32 = GRID_H * CELL_PX;
const DEFAULT_TPS: f64 = 10.0;

fn main() {
    env_logger::init(); // RUST_LOG=wgpu=warn cargo run  for GPU validation messages
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()).unwrap();
}

struct App {
    window: Option<Arc<Window>>,
    // Tuple ordering matches creation order; also controls drop order
    // (renderer and sim dropped before context, context dropped before window).
    gpu: Option<(GpuContext, GpuRenderer, GpuSim)>,
    grid: Grid,
    paused: bool,
    last_tick: Instant,
    tick_interval: Duration,
    generation: u64,
}

impl App {
    fn new() -> Self {
        let mut grid = Grid::new(GRID_W as usize, GRID_H as usize, /*toroidal=*/ true);
        grid.seed_random(0.3);
        Self {
            window: None,
            gpu: None,
            grid,
            paused: false,
            last_tick: Instant::now(),
            tick_interval: Duration::from_secs_f64(1.0 / DEFAULT_TPS),
            generation: 0,
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Conway's Game of Life")
                        .with_inner_size(LogicalSize::new(WIN_W, WIN_H))
                        .with_resizable(false),
                )
                .expect("window creation failed"),
        );

        let ctx = pollster::block_on(GpuContext::new(Arc::clone(&window), WIN_W, WIN_H));
        let renderer = GpuRenderer::new(&ctx);
        let sim = GpuSim::new(&ctx, renderer.cell_bind_group_layout(), GRID_W, GRID_H);
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
            KeyCode::KeyR => self.reset_to(|g| g.seed_random(0.3)),
            KeyCode::KeyG => self.reset_to(|g| {
                g.seed_pattern(patterns::GLIDER, GRID_W as isize / 4, GRID_H as isize / 4);
            }),
            KeyCode::KeyB => self.reset_to(|g| {
                g.seed_pattern(patterns::BLINKER, GRID_W as isize / 2, GRID_H as isize / 2);
            }),
            KeyCode::KeyT => self.reset_to(|g| {
                g.seed_pattern(patterns::TOAD, GRID_W as isize / 2, GRID_H as isize / 2);
            }),
            KeyCode::KeyP => self.reset_to(|g| {
                g.seed_pattern(
                    patterns::R_PENTOMINO,
                    GRID_W as isize / 2,
                    GRID_H as isize / 2,
                );
            }),
            KeyCode::KeyC => self.reset_to(|g| {
                g.seed_pattern(patterns::BEACON, GRID_W as isize / 2, GRID_H as isize / 2);
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
        let Some((ctx, renderer, sim)) = &mut self.gpu else {
            return;
        };

        // Advance simulation (GPU compute) if unpaused and interval elapsed.
        if !self.paused && self.last_tick.elapsed() >= self.tick_interval {
            sim.step(&ctx.device, &ctx.queue);
            self.generation += 1;
            self.last_tick = Instant::now();
        }

        // Acquire swap-chain frame.
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
        if let Some(w) = &self.window {
            w.set_title(&format!(
                "GoL  gen={}  {} tps  {}  \
                 | Space=pause  R=random  G=glider  B=blinker  T=toad  P=R-pentomino  C=beacon  ↑↓=speed",
                self.generation,
                tps,
                if self.paused { "PAUSED" } else { "running" },
            ));
            w.request_redraw();
        }
    }
}
