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
    bg_r:         f32,
    bg_g:         f32,
    bg_b:         f32,
    start_hue:    f32, // hue at age 0 (degrees 0–360)
    end_hue:      f32, // hue at max_lifetime (degrees 0–360)
    max_lifetime: f32, // generations for the full hue sweep
    sat_min:      f32, // saturation at zero live-neighbour density
    sat_max:      f32, // saturation at full live-neighbour density
    val_min:      f32, // reserved for future energy-based value modulation
    val_max:      f32, // brightness of alive cells
    _pad0:        f32,
    _pad1:        f32,
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

// Compute the RGBA colour for a single grid cell at integer coordinates.
// Handles boundary conditions (toroidal wrap or dead-border) and returns the
// background colour for dead cells, so bilinear interpolation blends edges.
fn cell_rgba(xi: i32, yi: i32) -> vec4<f32> {
    let w = i32(params.grid_w);
    let h = i32(params.grid_h);
    var cx: i32 = xi;
    var cy: i32 = yi;
    if params.toroidal == 1u {
        cx = ((xi % w) + w) % w;
        cy = ((yi % h) + h) % h;
    } else {
        if xi < 0 || yi < 0 || xi >= w || yi >= h {
            return vec4<f32>(visual.bg_r, visual.bg_g, visual.bg_b, 1.0);
        }
    }
    let cell = cells[u32(cy) * params.grid_w + u32(cx)];
    if cell.x <= 0.5 {
        return vec4<f32>(visual.bg_r, visual.bg_g, visual.bg_b, 1.0);
    }
    let t       = clamp(cell.y / visual.max_lifetime, 0.0, 1.0);
    let hue     = mix(visual.start_hue, visual.end_hue, t);
    let n_alive = alive_neighbours(cx, cy);
    let density = f32(n_alive) / 8.0;
    let sat     = mix(visual.sat_min, visual.sat_max, density);
    return vec4<f32>(hsv_to_rgb(hue, sat, visual.val_max), 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Bilinear interpolation: blend the 4 nearest grid cells.
    // Subtracting 0.5 centres the sample on each cell rather than its top-left corner.
    let gx = in.uv.x * f32(params.grid_w) - 0.5;
    let gy = in.uv.y * f32(params.grid_h) - 0.5;
    let x0 = i32(floor(gx));
    let y0 = i32(floor(gy));
    let fx = gx - floor(gx);
    let fy = gy - floor(gy);

    return mix(
        mix(cell_rgba(x0, y0),     cell_rgba(x0 + 1, y0),     fx),
        mix(cell_rgba(x0, y0 + 1), cell_rgba(x0 + 1, y0 + 1), fx),
        fy,
    );
}
