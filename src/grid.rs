use rand::Rng;

/// Row-major grid of `f32` cell values (0.0 = dead, 1.0 = alive).
/// Using `f32` now so later phases can store continuous state (age, density)
/// without restructuring buffers.
pub struct Grid {
    pub width: usize,
    pub height: usize,
    /// Current generation cell states, row-major: index = y * width + x.
    pub cells: Vec<f32>,
    /// Toroidal (true) or fixed-boundary (false) wrapping.
    #[allow(dead_code)]
    pub toroidal: bool,
    #[allow(dead_code)]
    scratch: Vec<f32>,
}

impl Grid {
    pub fn new(width: usize, height: usize, toroidal: bool) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            cells: vec![0.0; n],
            toroidal,
            scratch: vec![0.0; n],
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

    #[inline]
    #[allow(dead_code)] // used in tests and Phase 4 save-state reads
    pub fn get(&self, x: isize, y: isize) -> f32 {
        if self.toroidal {
            let xi = x.rem_euclid(self.width as isize) as usize;
            let yi = y.rem_euclid(self.height as isize) as usize;
            self.cells[yi * self.width + xi]
        } else {
            if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
                return 0.0;
            }
            self.cells[y as usize * self.width + x as usize]
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn live_neighbors(&self, x: isize, y: isize) -> u8 {
        let mut n = 0u8;
        for dy in -1i8..=1 {
            for dx in -1i8..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                if self.get(x + dx as isize, y + dy as isize) > 0.5 {
                    n += 1;
                }
            }
        }
        n
    }

    /// Advance one generation using B3/S23 rules.
    #[allow(dead_code)] // used in tests; GPU sim runs in production
    pub fn tick(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let n = self.live_neighbors(x as isize, y as isize);
                let alive = self.cells[y * self.width + x] > 0.5;
                // B3/S23: born with 3 neighbours; survives with 2 or 3
                let next = n == 3 || (alive && n == 2);
                self.scratch[y * self.width + x] = if next { 1.0 } else { 0.0 };
            }
        }
        std::mem::swap(&mut self.cells, &mut self.scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alive(g: &Grid, x: usize, y: usize) -> bool {
        g.cells[y * g.width + x] > 0.5
    }

    #[test]
    fn still_block_is_stable() {
        // 2×2 block is a still life
        let mut g = Grid::new(6, 6, false);
        g.cells[2 * 6 + 2] = 1.0;
        g.cells[2 * 6 + 3] = 1.0;
        g.cells[3 * 6 + 2] = 1.0;
        g.cells[3 * 6 + 3] = 1.0;
        let before = g.cells.clone();
        g.tick();
        assert_eq!(g.cells, before);
    }

    #[test]
    fn blinker_oscillates() {
        // Horizontal blinker at y=2, x=1..3
        let mut g = Grid::new(5, 5, false);
        g.cells[2 * 5 + 1] = 1.0;
        g.cells[2 * 5 + 2] = 1.0;
        g.cells[2 * 5 + 3] = 1.0;

        g.tick();
        // Should become vertical: (2,1), (2,2), (2,3)
        assert!(alive(&g, 2, 1));
        assert!(alive(&g, 2, 2));
        assert!(alive(&g, 2, 3));
        assert!(!alive(&g, 1, 2));
        assert!(!alive(&g, 3, 2));

        g.tick();
        // Back to horizontal
        assert!(alive(&g, 1, 2));
        assert!(alive(&g, 2, 2));
        assert!(alive(&g, 3, 2));
        assert!(!alive(&g, 2, 1));
        assert!(!alive(&g, 2, 3));
    }

    #[test]
    fn birth_rule_b3() {
        // Dead cell with exactly 3 live neighbours is born
        let mut g = Grid::new(5, 5, false);
        g.cells[1 * 5 + 2] = 1.0; // (2,1)
        g.cells[2 * 5 + 1] = 1.0; // (1,2)
        g.cells[2 * 5 + 3] = 1.0; // (3,2)
        g.tick();
        assert!(alive(&g, 2, 2));
    }

    #[test]
    fn overpopulation_kills() {
        // Live cell with 4 neighbours dies
        let mut g = Grid::new(5, 5, false);
        g.cells[2 * 5 + 2] = 1.0; // centre
        g.cells[1 * 5 + 2] = 1.0;
        g.cells[3 * 5 + 2] = 1.0;
        g.cells[2 * 5 + 1] = 1.0;
        g.cells[2 * 5 + 3] = 1.0;
        g.tick();
        assert!(!alive(&g, 2, 2));
    }

    #[test]
    fn isolation_kills() {
        // Live cell with 1 neighbour dies
        let mut g = Grid::new(5, 5, false);
        g.cells[2 * 5 + 2] = 1.0;
        g.cells[2 * 5 + 3] = 1.0;
        g.tick();
        assert!(!alive(&g, 2, 2));
    }

    #[test]
    fn toroidal_wraps() {
        // Cell at (0,0) should have a neighbour at (W-1, H-1) on a toroidal grid
        let mut g = Grid::new(5, 5, true);
        g.cells[0 * 5 + 0] = 1.0; // (0,0)
        g.cells[0 * 5 + 4] = 1.0; // (4,0)
        g.cells[4 * 5 + 0] = 1.0; // (0,4)
        // (0,0) has neighbours at (4,4),(0,4),(1,4),(4,0),(1,0),(4,1),(0,1),(1,1)
        // alive: (4,0), (0,4) → 2 → survives
        g.tick();
        assert!(alive(&g, 0, 0));
    }
}
