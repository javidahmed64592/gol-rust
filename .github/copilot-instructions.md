# GitHub Copilot instructions — gol-rust

## Project summary

GPU-accelerated Conway's Game of Life in Rust.
Simulation and rendering both run on the GPU via `wgpu`; the CPU only handles window input, timing, and initial seeding.

## Architecture layers

| Layer           | Module(s)                                 | Responsibility                                                |
| --------------- | ----------------------------------------- | ------------------------------------------------------------- |
| Simulation data | `grid.rs`, `patterns.rs`                  | CPU-side `Vec<f32>` grid, B3/S23 tick, pattern seeding        |
| GPU init        | `gpu_context.rs`                          | `wgpu` adapter / device / surface / swap-chain config         |
| GPU simulation  | `gpu_sim.rs` + `shaders/sim.wgsl`         | Compute pipeline, ping-pong storage buffers, WGSL GoL shader  |
| GPU rendering   | `gpu_renderer.rs` + `shaders/render.wgsl` | Full-screen quad render pipeline; reads the front cell buffer |
| App / input     | `main.rs`                                 | `winit` `ApplicationHandler`, keyboard input, tick timing     |

## Key design decisions

- **Cell state is `f32`** (0.0 = dead, 1.0 = alive) even though Phase 1–3 only use binary values, so later phases can add continuous/aging state without restructuring GPU buffers.
- **Ping-pong storage buffers** (`cells_a`, `cells_b`): each compute tick reads from `front`, writes to `1-front`, then swaps. The renderer always reads `front`.
- **Toroidal wrapping** is done in the compute shader via modulo; a `Params` uniform holds `grid_w`, `grid_h`, and `toroidal` flag (currently always 1).
- **Tick rate is decoupled from frame rate**: `winit` requests redraws at vsync; `GpuSim::step` is dispatched only when the tick interval has elapsed.
- **No CPU↔GPU round-trips per frame**: after the initial `upload_cells` on seed/reset, all simulation state lives on the GPU.

## Module contracts

- `GpuContext` — owns `device`, `queue`, `surface`, `surface_config`; created once in `ApplicationHandler::resumed`.
- `GpuRenderer` — owns the render pipeline and the cell bind-group layout; exposes `cell_bind_group_layout()` so `GpuSim` can create matching render bind groups.
- `GpuSim` — owns both cell buffers, the params uniform, the compute pipeline, two compute bind groups, and two render bind groups (one per buffer). Exposes `step()`, `upload_cells()`, and `display_bind_group()`.

## Shader paths

Shaders live in `src/shaders/` and are embedded at compile time via `wgpu::include_wgsl!("shaders/<name>.wgsl")` (path relative to the calling `.rs` file).

## Build & test

```sh
cargo build            # check compilation
cargo test             # run GoL correctness unit tests (in grid.rs)
cargo run              # launch the simulation window
RUST_LOG=wgpu=warn cargo run   # show wgpu validation messages
```

## Explicitly deferred (do not implement)

- Continuous / multi-valued cell state (SmoothLife / Lenia)
- Age- or density-based coloring
- Post-process blur or interpolated rendering
- HashLife or other non-GPU scalability approaches
- Live-adjustable grid size / rule parameters (Phase 4)
- On-screen HUD / FPS counter (Phase 4)

## Extending this project

- **Phase 4 controls**: add a `push_constant` or update the `params_buf` uniform in `GpuSim` for runtime rule changes; the `Params` struct has a `toroidal` field ready for toggling.
- **Visual effects**: add a post-process pass after the cell render pass; the render pipeline is already separated from the simulation pipeline.
- **Larger grids**: change `GRID_W` / `GRID_H` constants in `main.rs`; the compute dispatch and buffer allocation scale automatically.
