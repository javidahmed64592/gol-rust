# GitHub Copilot instructions — gol-rust

## Project summary

GPU-accelerated Conway's Game of Life and SmoothLife in Rust. Both simulation and rendering run entirely on the GPU via `wgpu`; the CPU handles only window input, timing, and initial seeding. All runtime parameters are driven by `gol.toml`.

## Architecture layers

| Layer           | Module(s)                                 | Responsibility                                                          |
| --------------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| Simulation data | `grid.rs`, `patterns.rs`                  | CPU-side seed buffer, pattern seeding with initial hues                 |
| Config          | `config.rs` + `gol.toml`                  | Load and validate all runtime parameters                                |
| GPU init        | `gpu_context.rs`                          | `wgpu` adapter / device / surface / swap-chain config                   |
| GPU simulation  | `gpu_sim.rs` + `shaders/sim.wgsl`         | Compute pipeline, ping-pong storage buffers, GoL and SmoothLife shaders |
| GPU rendering   | `gpu_renderer.rs` + `shaders/render.wgsl` | Two-pass render: HSV cell colouring → Gaussian blur → swapchain         |
| App / input     | `main.rs`                                 | `winit` `ApplicationHandler`, keyboard input, tick timing               |

## Key design decisions

- **Cell buffer layout is `vec4<f32>` per cell** — `.x = energy` (0.0–1.0), `.y = age` (ticks alive above threshold), `.z = hue` (0–360°, inherited on birth), `.w = reserved`.
- **Two simulation modes** toggled at runtime with M: discrete GoL (B/S bitmasks) and SmoothLife (ring-kernel convolution + smooth sigmoid rule).
- **Hue inheritance on birth**: new cells take an energy-weighted circular mean of alive-neighbour hues (unit-vector method), in both modes.
- **Ping-pong storage buffers** (`cells_a`, `cells_b`): each tick reads from `front`, writes to `1-front`, then swaps. The renderer always reads `front`.
- **`Params` uniform** (64 bytes, shared by compute and render shaders) holds grid dims, mode flag, B/S bitmasks, all SmoothLife kernel params, and `age_threshold`.
- **Tick rate decoupled from frame rate**: `winit` requests redraws at vsync; `GpuSim::step` dispatches only when the tick interval has elapsed.
- **No CPU↔GPU round-trips per frame**: after `upload_cells` on seed/reset, all simulation state lives on the GPU.
- **Two-pass rendering**: cells → `Rgba16Float` offscreen texture (bilinear interpolation + HSV colouring); then Gaussian blur (or blit) → swapchain.

## Module contracts

- `GpuContext` — owns `device`, `queue`, `surface`, `surface_config`; created once in `ApplicationHandler::resumed`.
- `GpuRenderer` — owns the render pipeline, offscreen texture, blur pipeline, and `VisualParams` uniform. Exposes `cell_bind_group_layout()` so `GpuSim` can wire its cell buffers.
- `GpuSim` — owns both cell buffers, the `Params` uniform (shared by both shaders), compute pipeline, and bind groups. Exposes `step()`, `upload_cells()`, `display_bind_group()`, and `set_mode()`.

## `Params` uniform layout (64 bytes)

Both `sim.wgsl` and `render.wgsl` declare this struct; its WGSL layout must match the Rust `#[repr(C)]` layout exactly.

| Bytes | Field(s)                       | Purpose                       |
| ----- | ------------------------------ | ----------------------------- |
| 0–7   | `grid_w`, `grid_h`             | Grid dimensions               |
| 8–11  | `toroidal`                     | 1 = toroidal wrap             |
| 12–15 | `mode`                         | 0 = GoL, 1 = SmoothLife       |
| 16–23 | `birth_mask`, `survive_mask`   | GoL bitmasks                  |
| 24–31 | `_pad0`, `_pad1`               | —                             |
| 32–39 | `inner_radius`, `outer_radius` | SmoothLife ring radii         |
| 40–47 | `birth_lo`, `birth_hi`         | SmoothLife birth interval     |
| 48–55 | `survive_lo`, `survive_hi`     | SmoothLife survival interval  |
| 56–59 | `sigmoid_sharpness`            | Logistic sigmoid steepness    |
| 60–63 | `age_threshold`                | Energy level considered alive |

## Shader paths

Shaders live in `src/shaders/` and are embedded at compile time via `wgpu::include_wgsl!("shaders/<name>.wgsl")` (path relative to the calling `.rs` file).

## Build & run

```sh
cargo build                   # check compilation
cargo run                     # launch the simulation window
cargo run --release           # full-speed GPU path
RUST_LOG=wgpu=warn cargo run  # show wgpu validation messages
```

## Runtime configuration (`gol.toml`)

All parameters live in `gol.toml` at the working directory. Edit and restart to apply.
Sections: `[window]`, `[grid]`, `[rules]`, `[simulation]`, `[visuals]`, `[smoothlife]`.

## Extending this project

- **Interactive seeding**: capture mouse position and write cell energy via a staging buffer each frame.
- **Rule-space drift**: slowly interpolate `Params` SmoothLife fields over time between two presets for ever-changing visuals.
- **Multi-species**: run two continuous fields with different kernel parameters that influence each other's density sums.
- **GPU stats readback**: periodically map a small reduction buffer to read aggregate energy/cell-count without a full grid readback.

Simulation and rendering both run on the GPU via `wgpu`; the CPU only handles window input, timing, and initial seeding.
All runtime parameters (grid size, rules, tick rate, HSV coloring) are driven by `gol.toml`.

## Architecture layers

| Layer           | Module(s)                                 | Responsibility                                                               |
| --------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| Simulation data | `grid.rs`, `patterns.rs`                  | CPU-side seed buffer, pattern seeding (no CPU tick)                          |
| Config          | `config.rs` + `gol.toml`                  | Load and validate all runtime parameters                                     |
| GPU init        | `gpu_context.rs`                          | `wgpu` adapter / device / surface / swap-chain config                        |
| GPU simulation  | `gpu_sim.rs` + `shaders/sim.wgsl`         | Compute pipeline, ping-pong storage buffers, WGSL GoL shader                 |
| GPU rendering   | `gpu_renderer.rs` + `shaders/render.wgsl` | Full-screen quad render pipeline; HSV coloring from per-cell age and density |
| App / input     | `main.rs`                                 | `winit` `ApplicationHandler`, keyboard input, tick timing                    |

## Key design decisions

- **Cell buffer layout is `vec2<f32>` per cell** — `x = state` (0.0 dead / 1.0 alive), `y = age` (generations survived, reset on birth/death). The render shader reads both. A future `energy: f32` field will extend this layout without restructuring.
- **Ping-pong storage buffers** (`cells_a`, `cells_b`): each compute tick reads from `front`, writes to `1-front`, then swaps. The renderer always reads `front`.
- **Toroidal wrapping** is done in the compute shader via modulo; a `Params` uniform holds `grid_w`, `grid_h`, `toroidal` flag, and B/S rule bitmasks.
- **Tick rate is decoupled from frame rate**: `winit` requests redraws at vsync; `GpuSim::step` is dispatched only when the tick interval has elapsed.
- **No CPU↔GPU round-trips per frame**: after the initial `upload_cells` on seed/reset, all simulation state lives on the GPU.
- **HSV coloring**: hue = f(age), saturation = f(live-neighbour density), value = alive brightness. All ranges are configurable via `gol.toml` and a `VisualParams` uniform.

## Module contracts

- `GpuContext` — owns `device`, `queue`, `surface`, `surface_config`; created once in `ApplicationHandler::resumed`.
- `GpuRenderer` — owns the render pipeline, the cell bind-group layout (group 0), and the `VisualParams` uniform + bind group (group 1). Exposes `cell_bind_group_layout()` so `GpuSim` can wire its buffers.
- `GpuSim` — owns both cell buffers, the sim `Params` uniform, the compute pipeline, compute bind groups, and render bind groups (one per buffer). Exposes `step()`, `upload_cells()`, and `display_bind_group()`.

## Shader paths

Shaders live in `src/shaders/` and are embedded at compile time via `wgpu::include_wgsl!("shaders/<name>.wgsl")` (path relative to the calling `.rs` file).

## Build & run

```sh
cargo build                   # check compilation
cargo run                     # launch the simulation window
cargo run --release           # full-speed (target: 4096×4096 at 500 tps)
RUST_LOG=wgpu=warn cargo run  # show wgpu validation messages
```

## Runtime configuration (`gol.toml`)

All parameters live in `gol.toml` at the working directory. Edit and restart to apply.
Key sections: `[grid]`, `[rules]`, `[simulation]`, `[window]`, `[visuals]`.

## Explicitly deferred

- Continuous / multi-valued cell state (SmoothLife / Lenia rule replacement)
- Color inheritance / blending between neighbouring cells on birth
- Rendering-side smoothing / blur post-process pass (Step 3 of visuals phase)

## Extending this project

- **Step 3 (blur)**: add a second render target and a blur shader pass after the main draw; make it toggleable via `gol.toml`.
- **Energy field**: add `energy: f32` as the third component of the per-cell buffer struct, update the compute shader to maintain it, and pass it through to the fragment shader for value modulation.
- **Continuous-state rule**: swap the binary state in the compute shader for a continuous float; the age/coloring pipeline in the render shader is already independent of the rule.

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
