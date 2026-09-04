// Gaussian blur post-process pass.  Reads from an Rgba16Float intermediate texture
// produced by the cell render pass and outputs to the swapchain surface.
//
// When blur.enabled == 0 or blur.radius <= 0 the shader is a plain blit.
// Kernel radius is clamped to ±7 px so the inner loop is at most 15×15 = 225 taps.
//
// Bind groups:
//   group(0) binding(0): source texture_2d<f32>
//   group(0) binding(1): linear-clamp sampler
//   group(1) binding(0): BlurParams uniform

struct BlurParams {
    enabled: u32,
    radius:  f32,  // Gaussian sigma in pixels
    _pad0:   f32,
    _pad1:   f32,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

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

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(1) @binding(0) var<uniform> blur: BlurParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if blur.enabled == 0u || blur.radius <= 0.0 {
        return textureSample(src_tex, src_smp, in.uv);
    }

    let sigma = blur.radius;
    let r     = min(i32(ceil(2.5 * sigma)), 7);
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));

    var colour       = vec4<f32>(0.0);
    var total_weight = 0.0;

    for (var dy: i32 = -r; dy <= r; dy++) {
        for (var dx: i32 = -r; dx <= r; dx++) {
            let off    = vec2<f32>(f32(dx), f32(dy)) * texel;
            let w      = exp(-0.5 * f32(dx * dx + dy * dy) / (sigma * sigma));
            colour       += w * textureSample(src_tex, src_smp, in.uv + off);
            total_weight += w;
        }
    }

    return colour / total_weight;
}
