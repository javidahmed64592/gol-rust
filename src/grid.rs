use rand::Rng;

pub struct Grid {
    pub width: usize,
    pub height: usize,
    /// Row-major cell values — 0.0 = dead, 1.0 = alive.
    pub cells: Vec<f32>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![0.0; width * height],
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(0.0);
    }

    pub fn seed_random(&mut self, density: f32) {
        let mut rng = rand::thread_rng();
        for cell in &mut self.cells {
            *cell = if rng.gen_bool(density as f64) {
                1.0
            } else {
                0.0
            };
        }
    }

    /// Place a pattern (slice of (dx, dy) offsets from center) onto the grid.
    pub fn seed_pattern(&mut self, offsets: &[(isize, isize)], cx: isize, cy: isize) {
        for &(dx, dy) in offsets {
            let x = (cx + dx).rem_euclid(self.width as isize) as usize;
            let y = (cy + dy).rem_euclid(self.height as isize) as usize;
            self.cells[y * self.width + x] = 1.0;
        }
    }
}
