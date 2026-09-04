use rand::Rng;

pub struct Grid {
    pub width: usize,
    pub height: usize,
    /// Interleaved [energy, age, hue, _pad] per cell, row-major.
    /// Index of cell (x,y): base = (y*width + x)*4;
    ///   energy = base, age = base+1, hue = base+2, _pad = base+3.
    pub cells: Vec<f32>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![0.0; width * height * 4],
        }
    }

    pub fn clear(&mut self) {
        self.cells.fill(0.0);
    }

    pub fn seed_random(&mut self, density: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..self.width * self.height {
            let alive = rng.gen_bool(density as f64);
            self.cells[i * 4] = if alive { 1.0 } else { 0.0 };
            // age (i*4+1) stays 0
            // All cells get a random hue so isolated births have something to inherit.
            self.cells[i * 4 + 2] = rng.gen_range(0.0_f32..360.0_f32);
            // pad (i*4+3) stays 0
        }
    }

    /// Place a pattern (slice of (dx, dy) offsets from centre) onto the grid.
    pub fn seed_pattern(&mut self, offsets: &[(isize, isize)], cx: isize, cy: isize, hue: f32) {
        for &(dx, dy) in offsets {
            let x = (cx + dx).rem_euclid(self.width as isize) as usize;
            let y = (cy + dy).rem_euclid(self.height as isize) as usize;
            let base = (y * self.width + x) * 4;
            self.cells[base] = 1.0; // energy
            // age stays 0
            self.cells[base + 2] = hue;
        }
    }
}
