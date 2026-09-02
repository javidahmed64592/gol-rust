// Phase 1 CPU renderer — draws each cell as a CELL_PX × CELL_PX block into
// a flat u32 buffer (0x00RRGGBB format expected by minifb).
// Replace this module entirely when moving to a wgpu render pipeline in Phase 2.

use crate::grid::Grid;

pub fn draw_cells(buffer: &mut [u32], grid: &Grid, grid_w: usize, grid_h: usize, cell_px: usize) {
    let win_w = grid_w * cell_px;
    for y in 0..grid_h {
        for x in 0..grid_w {
            let color: u32 = if grid.get(x as isize, y as isize) > 0.5 {
                0x00_FF_FF_FF // white
            } else {
                0x00_10_10_10 // near-black
            };
            for sy in 0..cell_px {
                let row_start = (y * cell_px + sy) * win_w + x * cell_px;
                buffer[row_start..row_start + cell_px].fill(color);
            }
        }
    }
}
