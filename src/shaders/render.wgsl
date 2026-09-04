// Full-screen quad render: vertex shader generates a TriangleStrip quad;
// fragment shader maps each pixel to a grid cell and applies HSV coloring.
//
// Cell buffer layout: vec4<f32> per cell
//   .x = energy  (0.0 – 1.0)
//   .y = age     (ticks alive above threshold)
//   .z = hue     (0 – 360°, inherited from neighbours on birth)
//   .w = (unused)
//
// Bind groups:
//   group(0): sim Params uniform + cells storage buffer  (managed by GpuSim)
//   group(1): VisualParams uniform                       (managed by GpuRenderer)

struct Params {
    grid_w:            u32,
    grid_h:            u32,
    toroidal:          u32,
    mode:              u32,
    birth_mask:        u32,
    survive_mask:      u32,
    _pad0:             u32,
    _pad1:             u32,
    inner_radius:      f32,
    outer_radius:      f32,
    birth_lo:          f32,
    birth_hi:          f32,
    survive_lo:        f32,
    survive_hi:        f32,
    sigmoid_sharpness: f32,
    age_threshold:     f32,
}

struct VisualParams {
    bg_r:         f32,
    bg_g:         f32,
    bg_b:         f32,
    start_hue:    f32, // fallback hue at age 0 when no inherited hue is set
    end_hue:      f32, // fallback hue at max_lifetime
    max_lifetime: f32,
    sat_min:      f32,
    sat_max:      f32,
    val_min:      f32, // brightness at energy == age_threshold
    val_max:      f32, // brightness at energy == 1.0
    _pad0:        f32,
    _pad1:        f32,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

// Four corners of a clip-space quad, wound for TriangleStrip (CCW).
// (0,0) UV is top-left of the grid.
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

@group(0) @binding(0) var<uniform>       params: Params;
@group(0) @binding(1) var<storage, read> cells:  array<vec4<f32>>; // .x=energy .y=age .z=hue

@group(1) @binding(0) var<uniform> visual: VisualParams;

fn get_cell_at(x: i32, y: i32) -> vec4<f32> {
    let w = i32(params.grid_w);
    let h = i32(params.grid_h);
    if params.toroidal == 1u {
        let xi = ((x % w) + w) % w;
        let yi = ((y % h) + h) % h;
        return cells[u32(yi) * params.grid_w + u32(xi)];
    } else {
        if x < 0 || y < 0 || x >= w || y >= h { return vec4<f32>(0.0); }
        return cells[u32(y) * params.grid_w + u32(x)];
    }
}

// Sum of Moore-neighbour energies normalised to [0, 1], used for saturation.
fn neighbour_density(x: i32, y: i32) -> f32 {
    return (
        get_cell_at(x - 1, y - 1).x +
        get_cell_at(x,     y - 1).x +
        get_cell_at(x + 1, y - 1).x +
        get_cell_at(x - 1, y    ).x +
        get_cell_at(x + 1, y    ).x +
        get_cell_at(x - 1, y + 1).x +
        get_cell_at(x,     y + 1).x +
        get_cell_at(x + 1, y + 1).x
    ) / 8.0;
}

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

fn cell_rgba(xi: i32, yi: i32) -> vec4<f32> {
    let bg = vec4<f32>(visual.bg_r, visual.bg_g, visual.bg_b, 1.0);
    let w  = i32(params.grid_w);
    let h  = i32(params.grid_h);
    var cx = xi;
    var cy = yi;
    if params.toroidal == 1u {
        cx = ((xi % w) + w) % w;
        cy = ((yi % h) + h) % h;
    } else {
        if xi < 0 || yi < 0 || xi >= w || yi >= h { return bg; }
    }

    let cell   = cells[u32(cy) * params.grid_w + u32(cx)];
    let energy = cell.x;
    let thresh = params.age_threshold;

    // Smooth blend from background to alive colour as energy crosses the threshold.
    let alive_alpha = smoothstep(0.0, thresh, energy);
    if alive_alpha <= 0.0 { return bg; }

    // Use inherited hue; fall back to age-based sweep if hue is still zero.
    var hue = cell.z;
    if hue == 0.0 {
        let t = clamp(cell.y / visual.max_lifetime, 0.0, 1.0);
        hue = mix(visual.start_hue, visual.end_hue, t);
    }

    // Saturation driven by sum of neighbour energies.
    let density = neighbour_density(cx, cy);
    let sat     = mix(visual.sat_min, visual.sat_max, density);

    // Value scaled by energy within the alive range [threshold, 1].
    let energy_t = clamp((energy - thresh) / max(1.0 - thresh, 1e-4), 0.0, 1.0);
    let val      = mix(visual.val_min, visual.val_max, energy_t);

    let rgb = hsv_to_rgb(hue, sat, val);
    return mix(bg, vec4<f32>(rgb, 1.0), alive_alpha);
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
        mix(cell_rgba(x0,     y0),     cell_rgba(x0 + 1, y0),     fx),
        mix(cell_rgba(x0,     y0 + 1), cell_rgba(x0 + 1, y0 + 1), fx),
        fy,
    );
}
