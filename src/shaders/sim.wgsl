// Conway's Game of Life compute shader — configurable B/S rules via bitmasks.
// Cell buffer layout: vec2<f32> per cell — x = state (0.0/1.0), y = age (generations alive).
// Each invocation handles one cell; workgroups are 8×8.

struct Params {
    grid_w:       u32,
    grid_h:       u32,
    toroidal:     u32, // 1 = wraparound, 0 = fixed dead boundary
    birth_mask:   u32, // bit N set → dead cell with N neighbours is born
    survive_mask: u32, // bit N set → live cell with N neighbours survives
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

@group(0) @binding(0) var<uniform>              params:    Params;
@group(0) @binding(1) var<storage, read>        cells_in:  array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write>  cells_out: array<vec2<f32>>;

fn get_cell(x: i32, y: i32) -> f32 {
    let w = i32(params.grid_w);
    let h = i32(params.grid_h);
    if params.toroidal == 1u {
        let xi = ((x % w) + w) % w;
        let yi = ((y % h) + h) % h;
        return cells_in[u32(yi) * params.grid_w + u32(xi)].x;
    } else {
        if x < 0 || y < 0 || x >= w || y >= h { return 0.0; }
        return cells_in[u32(y) * params.grid_w + u32(x)].x;
    }
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    if gid.x >= params.grid_w || gid.y >= params.grid_h { return; }

    // Count the 8 Moore neighbours (unrolled for clarity).
    let n: u32 =
        u32(get_cell(x - 1, y - 1) > 0.5) +
        u32(get_cell(x,     y - 1) > 0.5) +
        u32(get_cell(x + 1, y - 1) > 0.5) +
        u32(get_cell(x - 1, y    ) > 0.5) +
        u32(get_cell(x + 1, y    ) > 0.5) +
        u32(get_cell(x - 1, y + 1) > 0.5) +
        u32(get_cell(x,     y + 1) > 0.5) +
        u32(get_cell(x + 1, y + 1) > 0.5);

    let idx      = gid.y * params.grid_w + gid.x;
    let old_cell = cells_in[idx];
    let was_alive = old_cell.x > 0.5;

    // Birth/survive via bitmasks: bit N is set if N neighbours trigger the rule.
    let next = select(
        ((params.birth_mask   >> n) & 1u) != 0u,
        ((params.survive_mask >> n) & 1u) != 0u,
        was_alive,
    );

    // Age: increment each generation the cell survives; reset to 0 on birth or death.
    let new_age = select(0.0, old_cell.y + 1.0, next && was_alive);

    cells_out[idx] = vec2<f32>(select(0.0, 1.0, next), new_age);
}
