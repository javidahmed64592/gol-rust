/// (dx, dy) offsets from a centre point.  Seed via `Grid::seed_pattern`.
///
/// Classic glider — moves toward (+x, +y) every 4 generations.
///
/// ```text
/// . # .
/// . . #
/// # # #
/// ```
pub const GLIDER: &[(isize, isize)] = &[(0, -1), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Horizontal blinker — period-2 oscillator.
pub const BLINKER: &[(isize, isize)] = &[(-1, 0), (0, 0), (1, 0)];

/// Toad — period-2 oscillator.
///
/// ```text
/// . # # #
/// # # # .
/// ```
pub const TOAD: &[(isize, isize)] = &[(0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1)];

/// Beacon — period-2 oscillator (two touching blocks).
pub const BEACON: &[(isize, isize)] = &[
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// R-pentomino — 5-cell methuselah; evolves for 1103 generations before stabilising.
///
/// ```text
/// . # #
/// # # .
/// . # .
/// ```
pub const R_PENTOMINO: &[(isize, isize)] = &[(1, -1), (2, -1), (0, 0), (1, 0), (1, 1)];
