// Full-screen quad render: vertex shader generates a TriangleStrip quad;
// fragment shader maps each pixel to a grid cell and applies HSV coloring
// driven by per-cell age (hue) and live-neighbour density (saturation).
//
// Bind groups:
//   group(0): sim Params uniform + cells storage buffer (managed by GpuSim)
//   group(1): VisualParams uniform               (managed by GpuRenderer)

struct Params {
    grid_w:       u32,
    grid_h:       u32,
    toroidal:     u32,
    birth_mask:   u32, // unused in render shader
    survive_mask: u32, // unused in render shader
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}

struct VisualParams {
    start_hue:    f32, // hue at age 0 (degrees 0–360)
    end_hue:      f32, // hue at max_lifetime (degrees 0–360)
    max_lifetime: f32, // generations for the full hue sweep
    sat_min:      f32, // saturation at zero live-neighbour density
    sat_max:      f32, // saturation at full live-neighbour density
    val_min:      f32, // reserved for future energy-based value modulation
    val_max:      f32, // brightness of alive cells
    _pad:         f32,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

// Four corners of a clip-space quad, wound for TriangleStrip (CCW):
//   TL(-1, 1) → TR(1, 1) → BL(-1,-1) → BR(1,-1)
// Matching UVs keep (0,0) at top-left of the grid.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
    );
    var uv = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv  = uv[vi];
    return out;
}

@group(0) @binding(0) var<uniform>          params: Params;
@group(0) @binding(1) var<storage, read>    cells:  array<vec2<f32>>; // x=state, y=age

@group(1) @binding(0) var<uniform>          visual: VisualParams;

// Read the alive/dead state of a grid cell with boundary handling.
fn get_state_at(x: i32, y: i32) -> f32 {
    let w = i32(params.grid_w);
    let h = i32(params.grid_h);
    if params.toroidal == 1u {
        let xi = ((x % w) + w) % w;
        let yi = ((y % h) + h) % h;
        return cells[u32(yi) * params.grid_w + u32(xi)].x;
    } else {
        if x < 0 || y < 0 || x >= w || y >= h { return 0.0; }
        return cells[u32(y) * params.grid_w + u32(x)].x;
    }
}

// Count alive Moore neighbours (8 cells).
fn alive_neighbours(x: i32, y: i32) -> u32 {
    return
        u32(get_state_at(x - 1, y - 1) > 0.5) +
        u32(get_state_at(x,     y - 1) > 0.5) +
        u32(get_state_at(x + 1, y - 1) > 0.5) +
        u32(get_state_at(x - 1, y    ) > 0.5) +
        u32(get_state_at(x + 1, y    ) > 0.5) +
        u32(get_state_at(x - 1, y + 1) > 0.5) +
        u32(get_state_at(x,     y + 1) > 0.5) +
        u32(get_state_at(x + 1, y + 1) > 0.5);
}

// Standard HSV → RGB. h in [0, 360), s and v in [0, 1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let h6 = (h % 360.0) / 60.0;
    let i  = i32(h6);
    let f  = h6 - f32(i);
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));
    switch i {
        case 0:  { return vec3<f32>(v, t, p); }
        case 1:  { return vec3<f32>(q, v, p); }
        case 2:  { return vec3<f32>(p, v, t); }
        case 3:  { return vec3<f32>(p, q, v); }
        case 4:  { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Clamp to avoid OOB when uv == 1.0 at the far edge.
    let x = min(u32(in.uv.x * f32(params.grid_w)), params.grid_w - 1u);
    let y = min(u32(in.uv.y * f32(params.grid_h)), params.grid_h - 1u);
    let idx  = y * params.grid_w + x;
    let cell = cells[idx];

    if cell.x <= 0.5 {
        // Dead cell — dark background.
        return vec4<f32>(0.05, 0.05, 0.05, 1.0);
    }

    // Hue: sweep start_hue → end_hue over max_lifetime generations.
    let t   = clamp(cell.y / visual.max_lifetime, 0.0, 1.0);
    let hue = mix(visual.start_hue, visual.end_hue, t);

    // Saturation: driven by live-neighbour density.
    let n_alive = alive_neighbours(i32(x), i32(y));
    let density = f32(n_alive) / 8.0;
    let sat     = mix(visual.sat_min, visual.sat_max, density);

    // Value: alive cells use val_max (energy proxy = 1.0 for Phase 2).
    let val = visual.val_max;

    return vec4<f32>(hsv_to_rgb(hue, sat, val), 1.0);
}
