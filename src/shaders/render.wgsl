// Full-screen quad render: vertex shader generates a TriangleStrip quad;
// fragment shader maps each pixel to a grid cell and samples the storage buffer.

struct Params {
    grid_w: u32,
    grid_h: u32,
    toroidal: u32, // unused in render shader, kept in sync with sim Params
    _pad:    u32,
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
@group(0) @binding(1) var<storage, read>    cells:  array<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Clamp to avoid OOB when uv == 1.0 at the far edge.
    let x = min(u32(in.uv.x * f32(params.grid_w)), params.grid_w - 1u);
    let y = min(u32(in.uv.y * f32(params.grid_h)), params.grid_h - 1u);
    let alive = cells[y * params.grid_w + x] > 0.5;
    let c = select(0.063, 1.0, alive); // near-black vs white
    return vec4<f32>(c, c, c, 1.0);
}
