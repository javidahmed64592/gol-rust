use rand::Rng;

pub struct Grid {
    pub width: usize,
    pub height: usize,
    /// Interleaved [state, age] per cell, row-major.
    /// Index of cell (x,y): base = (y*width + x)*2; state = base, age = base+1.
    pub cells: Vec<f32>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![0.0; width * height * 2],
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(0.0);
    }

    pub fn seed_random(&mut self, density: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..self.width * self.height {
            self.cells[i * 2] = if rng.gen_bool(density as f64) {
                1.0
            } else {
                0.0
            };
            // cells[i*2 + 1] (age) stays 0 — clear() is called before seeding
        }
    }

    /// Place a pattern (slice of (dx, dy) offsets from center) onto the grid.
    pub fn seed_pattern(&mut self, offsets: &[(isize, isize)], cx: isize, cy: isize) {
        for &(dx, dy) in offsets {
            let x = (cx + dx).rem_euclid(self.width as isize) as usize;
            let y = (cy + dy).rem_euclid(self.height as isize) as usize;
            self.cells[(y * self.width + x) * 2] = 1.0; // state slot; age slot stays 0
        }
    }
}
