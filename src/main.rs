mod grid;
mod patterns;
mod render;

use grid::Grid;
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::time::{Duration, Instant};

// --- Configuration -------------------------------------------------------
// Adjust these constants to change grid/window size without recompiling any
// other module.  Phase 4 will make these runtime-configurable.

const GRID_W: usize = 256;
const GRID_H: usize = 256;
/// Pixels per cell edge (integer scale factor).
const CELL_PX: usize = 3;
const WIN_W: usize = GRID_W * CELL_PX;
const WIN_H: usize = GRID_H * CELL_PX;
/// Starting simulation ticks per second (independent of display frame rate).
const DEFAULT_TPS: f64 = 10.0;

fn main() {
    let mut grid = Grid::new(GRID_W, GRID_H, /*toroidal=*/ true);
    grid.seed_random(0.3);

    let mut window = Window::new(
        "Conway's Game of Life",
        WIN_W,
        WIN_H,
        WindowOptions::default(),
    )
    .expect("failed to create window");

    // Cap display refresh to ~60 fps; simulation tick rate is managed separately.
    window.set_target_fps(60);

    let mut buffer = vec![0u32; WIN_W * WIN_H];
    let mut paused = false;
    let mut last_tick = Instant::now();
    let mut tick_interval = Duration::from_secs_f64(1.0 / DEFAULT_TPS);
    let mut generation: u64 = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // --- Input -------------------------------------------------------
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            grid.clear();
            grid.seed_random(0.3);
            generation = 0;
        }
        if window.is_key_pressed(Key::G, KeyRepeat::No) {
            grid.clear();
            grid.seed_pattern(patterns::GLIDER, GRID_W as isize / 4, GRID_H as isize / 4);
            generation = 0;
        }
        if window.is_key_pressed(Key::B, KeyRepeat::No) {
            grid.clear();
            grid.seed_pattern(patterns::BLINKER, GRID_W as isize / 2, GRID_H as isize / 2);
            generation = 0;
        }
        if window.is_key_pressed(Key::T, KeyRepeat::No) {
            grid.clear();
            grid.seed_pattern(patterns::TOAD, GRID_W as isize / 2, GRID_H as isize / 2);
            generation = 0;
        }
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            grid.clear();
            grid.seed_pattern(
                patterns::R_PENTOMINO,
                GRID_W as isize / 2,
                GRID_H as isize / 2,
            );
            generation = 0;
        }
        if window.is_key_pressed(Key::C, KeyRepeat::No) {
            grid.clear();
            grid.seed_pattern(patterns::BEACON, GRID_W as isize / 2, GRID_H as isize / 2);
            generation = 0;
        }
        if window.is_key_pressed(Key::Up, KeyRepeat::No) {
            let tps = 1.0 / tick_interval.as_secs_f64();
            tick_interval = Duration::from_secs_f64(1.0 / (tps * 2.0).min(500.0));
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::No) {
            let tps = 1.0 / tick_interval.as_secs_f64();
            tick_interval = Duration::from_secs_f64(1.0 / (tps / 2.0).max(0.5));
        }

        // --- Simulate ----------------------------------------------------
        if !paused && last_tick.elapsed() >= tick_interval {
            grid.tick();
            generation += 1;
            last_tick = Instant::now();
        }

        // --- Render ------------------------------------------------------
        render::draw_cells(&mut buffer, &grid, GRID_W, GRID_H, CELL_PX);

        let tps = (1.0 / tick_interval.as_secs_f64()).round() as u32;
        window.set_title(&format!(
            "GoL  gen={}  {} tps  {}  \
             | Space=pause  R=random  G=glider  B=blinker  T=toad  P=R-pentomino  C=beacon  ↑↓=speed",
            generation,
            tps,
            if paused { "PAUSED" } else { "running" },
        ));

        window
            .update_with_buffer(&buffer, WIN_W, WIN_H)
            .expect("window update failed");
    }
}
