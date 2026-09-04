// SmoothLife / Conway's Game of Life compute shader.
// Cell buffer layout: vec4<f32> per cell
//   .x = energy  (0.0 – 1.0)
//   .y = age     (ticks spent at or above params.age_threshold)
//   .z = hue     (0 – 360°, inherited from neighbours on birth)
//   .w = (unused / reserved)
//
// params.mode == 0  →  discrete B/S GoL via birth/survive bitmasks
// params.mode == 1  →  SmoothLife continuous rule via ring-kernel + sigmoid

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

@group(0) @binding(0) var<uniform>             params:    Params;
@group(0) @binding(1) var<storage, read>       cells_in:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cells_out: array<vec4<f32>>;

const PI: f32 = 3.14159265358979;

fn get_cell(x: i32, y: i32) -> vec4<f32> {
    let w = i32(params.grid_w);
    let h = i32(params.grid_h);
    if params.toroidal == 1u {
        let xi = ((x % w) + w) % w;
        let yi = ((y % h) + h) % h;
        return cells_in[u32(yi) * params.grid_w + u32(xi)];
    } else {
        if x < 0 || y < 0 || x >= w || y >= h { return vec4<f32>(0.0); }
        return cells_in[u32(y) * params.grid_w + u32(x)];
    }
}

// Circular mean of hues via unit-vector method; returns degrees [0, 360).
fn blend_hues(sin_sum: f32, cos_sum: f32) -> f32 {
    let len = sqrt(sin_sum * sin_sum + cos_sum * cos_sum);
    if len < 1e-6 { return 0.0; }
    let deg = atan2(sin_sum, cos_sum) * (180.0 / PI);
    return select(deg + 360.0, deg, deg >= 0.0);
}

// ── Discrete GoL ─────────────────────────────────────────────────────────────

fn gol_step(x: i32, y: i32) -> vec4<f32> {
    let idx       = u32(y) * params.grid_w + u32(x);
    let old       = cells_in[idx];
    let was_alive = old.x > 0.5;

    let n: u32 =
        u32(get_cell(x - 1, y - 1).x > 0.5) +
        u32(get_cell(x,     y - 1).x > 0.5) +
        u32(get_cell(x + 1, y - 1).x > 0.5) +
        u32(get_cell(x - 1, y    ).x > 0.5) +
        u32(get_cell(x + 1, y    ).x > 0.5) +
        u32(get_cell(x - 1, y + 1).x > 0.5) +
        u32(get_cell(x,     y + 1).x > 0.5) +
        u32(get_cell(x + 1, y + 1).x > 0.5);

    let next = select(
        ((params.birth_mask   >> n) & 1u) != 0u,
        ((params.survive_mask >> n) & 1u) != 0u,
        was_alive,
    );

    let new_age    = select(0.0, old.y + 1.0, next && was_alive);
    let new_energy = select(0.0, 1.0, next);

    // On birth: blend neighbour hues (energy-weighted circular mean).
    var new_hue = old.z;
    if next && !was_alive {
        var sin_sum: f32 = 0.0;
        var cos_sum: f32 = 0.0;
        var w_sum:   f32 = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                if dx == 0 && dy == 0 { continue; }
                let nb = get_cell(x + dx, y + dy);
                if nb.x > 0.5 {
                    let angle = nb.z * (PI / 180.0);
                    sin_sum += sin(angle) * nb.x;
                    cos_sum += cos(angle) * nb.x;
                    w_sum   += nb.x;
                }
            }
        }
        if w_sum > 0.0 {
            new_hue = blend_hues(sin_sum, cos_sum);
        }
    }

    return vec4<f32>(new_energy, new_age, new_hue, 0.0);
}

// ── SmoothLife ────────────────────────────────────────────────────────────────

// Logistic sigmoid centred at `a`.
fn sigma1(x: f32, a: f32) -> f32 {
    let sharpness = max(params.sigmoid_sharpness, 1e-6);
    return 1.0 / (1.0 + exp(-(x - a) * 4.0 / sharpness));
}

// Smooth indicator for interval [a, b].
fn sigma2(x: f32, a: f32, b: f32) -> f32 {
    return sigma1(x, a) * (1.0 - sigma1(x, b));
}

// Mix birth and survival response based on inner density m.
fn sigma_m(birth_val: f32, survive_val: f32, m: f32) -> f32 {
    let t = sigma1(m, 0.5);
    return birth_val * (1.0 - t) + survive_val * t;
}

fn smoothlife_step(x: i32, y: i32) -> vec4<f32> {
    let idx = u32(y) * params.grid_w + u32(x);
    let old = cells_in[idx];

    let ri  = params.inner_radius;
    let ro  = params.outer_radius;
    let ri2 = ri * ri;
    let ro2 = ro * ro;
    let R   = i32(ceil(ro));

    var inner_sum:  f32 = 0.0;
    var inner_area: f32 = 0.0;
    var outer_sum:  f32 = 0.0;
    var outer_area: f32 = 0.0;
    var sin_sum:    f32 = 0.0;
    var cos_sum:    f32 = 0.0;
    var hue_weight: f32 = 0.0;

    for (var dy: i32 = -R; dy <= R; dy++) {
        for (var dx: i32 = -R; dx <= R; dx++) {
            let r2 = f32(dx * dx + dy * dy);
            if r2 > ro2 { continue; }
            let nb = get_cell(x + dx, y + dy);
            if r2 <= ri2 {
                inner_sum  += nb.x;
                inner_area += 1.0;
            } else {
                outer_sum  += nb.x;
                outer_area += 1.0;
                let angle   = nb.z * (PI / 180.0);
                sin_sum    += sin(angle) * nb.x;
                cos_sum    += cos(angle) * nb.x;
                hue_weight += nb.x;
            }
        }
    }

    let m = select(0.0, inner_sum / inner_area, inner_area > 0.0);
    let n = select(0.0, outer_sum / outer_area, outer_area > 0.0);

    let new_energy = clamp(
        sigma_m(
            sigma2(n, params.birth_lo,   params.birth_hi),
            sigma2(n, params.survive_lo, params.survive_hi),
            m,
        ),
        0.0, 1.0,
    );

    let threshold = params.age_threshold;
    let was_alive = old.x      >= threshold;
    let now_alive = new_energy >= threshold;
    let new_age   = select(0.0, old.y + 1.0, now_alive && was_alive);

    // Update inherited hue only when crossing the alive threshold upward.
    var new_hue = old.z;
    if !was_alive && now_alive && hue_weight > 0.0 {
        new_hue = blend_hues(sin_sum, cos_sum);
    }

    return vec4<f32>(new_energy, new_age, new_hue, 0.0);
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.grid_w || gid.y >= params.grid_h { return; }
    let x       = i32(gid.x);
    let y       = i32(gid.y);
    let out_idx = gid.y * params.grid_w + gid.x;
    if params.mode == 0u {
        cells_out[out_idx] = gol_step(x, y);
    } else {
        cells_out[out_idx] = smoothlife_step(x, y);
    }
}
