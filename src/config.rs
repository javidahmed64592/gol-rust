use serde::Deserialize;

#[derive(Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub window: WindowConfig,
    #[serde(default)]
    pub grid: GridConfig,
    #[serde(default)]
    pub rules: RulesConfig,
    #[serde(default)]
    pub simulation: SimulationConfig,
    #[serde(default)]
    pub visuals: VisualsConfig,
}

#[derive(Deserialize)]
pub struct WindowConfig {
    #[serde(default = "default_win_w")]
    pub width: u32,
    #[serde(default = "default_win_h")]
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: default_win_w(),
            height: default_win_h(),
        }
    }
}

fn default_win_w() -> u32 {
    768
}
fn default_win_h() -> u32 {
    768
}

#[derive(Deserialize)]
pub struct GridConfig {
    #[serde(default = "default_grid_w")]
    pub width: u32,
    #[serde(default = "default_grid_h")]
    pub height: u32,
    #[serde(default = "default_true")]
    pub toroidal: bool,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            width: default_grid_w(),
            height: default_grid_h(),
            toroidal: default_true(),
        }
    }
}

fn default_grid_w() -> u32 {
    256
}
fn default_grid_h() -> u32 {
    256
}
fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub struct RulesConfig {
    #[serde(default = "default_birth")]
    pub birth: Vec<u8>,
    #[serde(default = "default_survive")]
    pub survive: Vec<u8>,
}

impl Default for RulesConfig {
    fn default() -> Self {
        Self {
            birth: default_birth(),
            survive: default_survive(),
        }
    }
}

fn default_birth() -> Vec<u8> {
    vec![3]
}
fn default_survive() -> Vec<u8> {
    vec![2, 3]
}

#[derive(Deserialize)]
pub struct SimulationConfig {
    #[serde(default = "default_tps")]
    pub ticks_per_second: f64,
    #[serde(default = "default_density")]
    pub initial_density: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            ticks_per_second: default_tps(),
            initial_density: default_density(),
        }
    }
}

fn default_tps() -> f64 {
    10.0
}
fn default_density() -> f32 {
    0.3
}

#[derive(Deserialize)]
pub struct VisualsConfig {
    #[serde(default = "default_start_hue")]
    pub start_hue: f32,
    #[serde(default = "default_end_hue")]
    pub end_hue: f32,
    #[serde(default = "default_max_lifetime")]
    pub max_lifetime: f32,
    #[serde(default = "default_sat_min")]
    pub sat_min: f32,
    #[serde(default = "default_sat_max")]
    pub sat_max: f32,
    #[serde(default = "default_val_min")]
    pub val_min: f32,
    #[serde(default = "default_val_max")]
    pub val_max: f32,
}

impl Default for VisualsConfig {
    fn default() -> Self {
        Self {
            start_hue: default_start_hue(),
            end_hue: default_end_hue(),
            max_lifetime: default_max_lifetime(),
            sat_min: default_sat_min(),
            sat_max: default_sat_max(),
            val_min: default_val_min(),
            val_max: default_val_max(),
        }
    }
}

fn default_start_hue() -> f32 {
    200.0
}
fn default_end_hue() -> f32 {
    30.0
}
fn default_max_lifetime() -> f32 {
    50.0
}
fn default_sat_min() -> f32 {
    0.5
}
fn default_sat_max() -> f32 {
    1.0
}
fn default_val_min() -> f32 {
    0.8
}
fn default_val_max() -> f32 {
    1.0
}

impl Config {
    /// Load from `gol.toml` in the current directory; fall back to defaults if absent.
    pub fn load() -> Self {
        let path = std::path::Path::new("gol.toml");
        if !path.exists() {
            return Self::default();
        }
        let src = std::fs::read_to_string(path).expect("failed to read gol.toml");
        let cfg: Self = toml::from_str(&src).expect("invalid gol.toml");
        cfg.validate();
        cfg
    }

    fn validate(&self) {
        for &n in self.rules.birth.iter().chain(&self.rules.survive) {
            assert!(
                n <= 8,
                "gol.toml: neighbour count {n} is invalid — must be 0–8"
            );
        }
        assert!(
            self.grid.width > 0 && self.grid.height > 0,
            "gol.toml: grid width and height must be > 0"
        );
        assert!(
            (0.0_f32..=1.0).contains(&self.simulation.initial_density),
            "gol.toml: initial_density must be 0.0–1.0"
        );
    }

    pub fn birth_mask(&self) -> u32 {
        self.rules.birth.iter().fold(0u32, |m, &n| m | (1u32 << n))
    }

    pub fn survive_mask(&self) -> u32 {
        self.rules
            .survive
            .iter()
            .fold(0u32, |m, &n| m | (1u32 << n))
    }
}
