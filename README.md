[![Rust](https://img.shields.io/badge/Rust-1.95.0-blue?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/javidahmed64592/gol-rust/ci.yml?branch=main&style=flat-square&label=CI&logo=github)](https://github.com/javidahmed64592/gol-rust/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/javidahmed64592/gol-rust/docs.yml?branch=main&style=flat-square&label=Docs&logo=github)](https://github.com/javidahmed64592/gol-rust/actions/workflows/docs.yml)
[![Release](https://img.shields.io/github/actions/workflow/status/javidahmed64592/gol-rust/release.yml?style=flat-square&label=Release&logo=github)](https://github.com/javidahmed64592/gol-rust/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- omit from toc -->
# gol-rust

GPU-accelerated [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) and [SmoothLife](https://arxiv.org/abs/1111.1567) in Rust.

Both simulation and rendering run entirely on the GPU via `wgpu` compute and render pipelines — the CPU only handles window input, timing, and initial seeding. Everything is configurable at runtime via `gol.toml`.

## Features

- **Two simulation modes** (toggle with **M** at runtime):
  - **Conway's Game of Life** — fully configurable B/S rules (GoL, HighLife, Day & Night, and more)
  - **SmoothLife** — continuous energy field with ring-kernel convolution and smooth sigmoid transitions for fluid, organic dynamics
- **Hue inheritance** — new cells blend the hues of their contributing neighbours on birth, so colonies develop distinct colour identities as they grow and merge
- **HSV colouring** — hue driven by inherited birth colour, saturation by neighbour energy density, brightness by cell energy
- **Gaussian blur post-process** — optional screen-space softening (toggle with **F**)
- **All parameters configurable** via `gol.toml` — grid size, B/S rules, SmoothLife kernel shape, colour ranges, blur, and more

## Building & running

Requires a [Rust toolchain](https://rustup.rs/) and a Vulkan / Metal / DX12-capable GPU.

```sh
cargo run --release
```

Edit `gol.toml` and restart to apply changes.

## Controls

| Key | Action |
| --- | ------ |
| **Space** | Pause / resume |
| **Enter** | Step one tick (while paused) |
| **M** | Toggle GoL ↔ SmoothLife mode |
| **R** | Random seed |
| **G** | Seed a glider |
| **B** | Seed a blinker |
| **T** | Seed a toad |
| **P** | Seed an R-pentomino |
| **C** | Seed a beacon |
| **↑ / ↓** | Double / halve tick rate |
| **F** | Toggle blur |
| **[ / ]** | Decrease / increase blur radius |
| **Esc** | Quit |

## Configuration (`gol.toml`)

`gol.toml` is loaded from the working directory on startup. The key sections are:

### `[grid]`

```toml
width    = 256
height   = 256
toroidal = true   # wrap edges; false = dead fixed boundary
```

### `[rules]` — Conway's Game of Life

Neighbour counts that trigger birth or survival. Any B/S rule works.

```toml
birth   = [3]       # B3/S23 — standard Conway
survive = [2, 3]

# HighLife:   birth = [3, 6]          / survive = [2, 3]
# Day&Night:  birth = [3, 6, 7, 8]   / survive = [3, 4, 6, 7, 8]
# Seeds:      birth = [2]             / survive = []
```

### `[smoothlife]` — continuous SmoothLife

A ring kernel integrates cell energy over an inner disk (self-density `m`) and outer annulus (neighbour density `n`). A smooth sigmoid rule maps `(m, n)` to the next energy value.

```toml
enabled            = false   # start in SmoothLife mode (or toggle with M)
inner_radius       = 2.0     # radius of self-density disk
outer_radius       = 6.0     # outer radius of neighbourhood ring
birth_lo           = 0.278   # lower bound of birth interval
birth_hi           = 0.365   # upper bound of birth interval
survive_lo         = 0.267   # lower bound of survival interval
survive_hi         = 0.445   # upper bound of survival interval
sigmoid_sharpness  = 0.028   # lower = smoother / more fluid; try 0.005–0.1
age_threshold      = 0.1     # energy level considered "alive"
```

Increasing `outer_radius` produces larger, slower structures (higher GPU cost). Decreasing `sigmoid_sharpness` towards `0.005` creates very fluid, wave-like motion.

### `[visuals]`

```toml
background_color = [0.02, 0.02, 0.02]
start_hue        = 200.0   # hue for newly born cells (degrees, 0–360)
end_hue          =  10.0   # hue for long-lived cells
max_lifetime     = 100     # generations for the full hue sweep (fallback only)
sat_min          = 0.5
sat_max          = 1.0
val_min          = 0.8     # brightness at energy == age_threshold
val_max          = 1.0     # brightness at full energy
blur_enabled     = false
blur_radius      = 1.5     # Gaussian sigma in pixels
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
