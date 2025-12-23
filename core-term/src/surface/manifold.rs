//! Terminal Manifold - A Conal Elliott style terminal renderer.
//!
//! The terminal is expressed as a composition of manifold combinators.
//! The type IS the AST - the grid is a tree of Select combinators that
//! the compiler monomorphizes to efficient SIMD code.
//!
//! # Architecture
//!
//! A terminal grid is built as a binary search tree of `Select` nodes:
//!
//! ```text
//! ColorManifold::new(
//!   Select { cond: Lt(X, mid), if_true: left_r, if_false: right_r },
//!   Select { cond: Lt(X, mid), if_true: left_g, if_false: right_g },
//!   ...
//! )
//! ```
//!
//! This gives O(log(cols) + log(rows)) depth, enabling fully vectorized
//! evaluation without extracting scalar values from SIMD lanes.
//!
//! # Key Insight
//!
//! A manifold is a functor. We never extract values - we compose manifolds.
//! The grid lookup is itself expressed as manifold composition.
//!
//! Uses `ColorManifold` from pixelflow-graphics for RGBA packing.

use pixelflow_core::{Discrete, Field, Lt, Manifold, Select, X, Y};
use pixelflow_graphics::render::rgba::Color as ColorManifold;

// ============================================================================
// Cell: Glyph Coverage + Colors → RGBA
// ============================================================================

/// A terminal cell that blends foreground/background based on glyph coverage.
///
/// This is the leaf node in the terminal manifold tree. It takes a glyph
/// (coverage manifold outputting Field) and produces a color blend.
#[derive(Clone)]
pub struct Cell<G> {
    /// The glyph coverage manifold (0.0 = background, 1.0 = foreground).
    pub glyph: G,
    /// Foreground color (R, G, B, A) normalized to [0, 1].
    pub fg: [f32; 4],
    /// Background color (R, G, B, A) normalized to [0, 1].
    pub bg: [f32; 4],
}

impl<G: Manifold<Output = Field>> Cell<G> {
    /// Create a new cell with glyph and colors.
    pub fn new(glyph: G, fg: [f32; 4], bg: [f32; 4]) -> Self {
        Self { glyph, fg, bg }
    }
}

/// Extracts a single channel from a Cell as a Field manifold.
///
/// This enables using the standard Select combinator (which works on Field)
/// for grid lookup.
#[derive(Clone)]
pub struct CellChannel<G, const CHANNEL: usize> {
    cell: Cell<G>,
}

impl<G, const CHANNEL: usize> CellChannel<G, CHANNEL> {
    /// Create a channel extractor for a cell.
    pub fn new(cell: Cell<G>) -> Self {
        Self { cell }
    }
}

impl<G: Manifold<Output = Field> + Clone, const CHANNEL: usize> Manifold for CellChannel<G, CHANNEL> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let coverage = self.cell.glyph.eval_raw(x, y, z, w);
        let c = coverage.field_max(Field::from(0.0)).field_min(Field::from(1.0));
        let omc = Field::from(1.0) - c;
        c * Field::from(self.cell.fg[CHANNEL]) + omc * Field::from(self.cell.bg[CHANNEL])
    }
}

// Type aliases for each channel
pub type CellR<G> = CellChannel<G, 0>;
pub type CellG<G> = CellChannel<G, 1>;
pub type CellB<G> = CellChannel<G, 2>;
pub type CellA<G> = CellChannel<G, 3>;

// ============================================================================
// Local Coordinate Transform
// ============================================================================

/// Transforms coordinates to be local within a cell.
#[derive(Clone, Copy, Debug)]
pub struct LocalCoords<M> {
    pub inner: M,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl<M> LocalCoords<M> {
    pub fn new(inner: M, offset_x: f32, offset_y: f32) -> Self {
        Self { inner, offset_x, offset_y }
    }
}

impl<M: Manifold> Manifold for LocalCoords<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> M::Output {
        let local_x = x - Field::from(self.offset_x);
        let local_y = y - Field::from(self.offset_y);
        self.inner.eval_raw(local_x, local_y, z, w)
    }
}

// ============================================================================
// Grid Builder Using ColorManifold + Select
// ============================================================================

/// Trait for types that can provide cell manifolds.
pub trait CellFactory: Send + Sync {
    /// The glyph manifold type.
    type Glyph: Manifold<Output = Field> + Clone + Send + Sync + 'static;

    /// Create a cell's glyph manifold for the given grid position.
    fn glyph(&self, col: usize, row: usize) -> Self::Glyph;

    /// Get foreground color for a cell.
    fn fg(&self, col: usize, row: usize) -> [f32; 4];

    /// Get background color for a cell.
    fn bg(&self, col: usize, row: usize) -> [f32; 4];

    /// Grid dimensions (cols, rows).
    fn dimensions(&self) -> (usize, usize);

    /// Cell dimensions in pixels.
    fn cell_size(&self) -> (f32, f32);
}

/// Builds a terminal grid as a ColorManifold with Select trees per channel.
///
/// The result is a `ColorManifold` where each channel (R, G, B, A) is a
/// binary search tree of Select combinators.
///
/// Type complexity is O(cols * rows) for the full grid structure.
pub fn build_grid<F: CellFactory>(
    factory: &F,
) -> ColorManifold<
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
> {
    let (cols, rows) = factory.dimensions();
    let (cell_w, cell_h) = factory.cell_size();

    // Build a Select tree for each channel
    let r = build_channel_tree::<F, 0>(factory, 0, cols, 0, rows, cell_w, cell_h);
    let g = build_channel_tree::<F, 1>(factory, 0, cols, 0, rows, cell_w, cell_h);
    let b = build_channel_tree::<F, 2>(factory, 0, cols, 0, rows, cell_w, cell_h);
    let a = build_channel_tree::<F, 3>(factory, 0, cols, 0, rows, cell_w, cell_h);

    ColorManifold::new(r, g, b, a)
}

/// Build a Select tree for a single color channel.
fn build_channel_tree<F: CellFactory, const CHANNEL: usize>(
    factory: &F,
    col_start: usize,
    col_end: usize,
    row_start: usize,
    row_end: usize,
    cell_w: f32,
    cell_h: f32,
) -> Box<dyn Manifold<Output = Field> + Send + Sync> {
    let col_count = col_end - col_start;
    let row_count = row_end - row_start;

    // Base case: single cell
    if col_count == 1 && row_count == 1 {
        let glyph = factory.glyph(col_start, row_start);
        let fg = factory.fg(col_start, row_start);
        let bg = factory.bg(col_start, row_start);

        let cell = Cell::new(glyph, fg, bg);
        let x_offset = col_start as f32 * cell_w;
        let y_offset = row_start as f32 * cell_h;

        // Extract the channel with local coordinate transform
        return Box::new(LocalCoords::new(
            CellChannel::<_, CHANNEL>::new(cell),
            x_offset,
            y_offset,
        ));
    }

    // Split on the larger dimension
    if col_count >= row_count && col_count > 1 {
        // Split columns
        let mid = col_start + col_count / 2;
        let threshold = mid as f32 * cell_w;

        let left = build_channel_tree::<F, CHANNEL>(
            factory, col_start, mid, row_start, row_end, cell_w, cell_h,
        );
        let right = build_channel_tree::<F, CHANNEL>(
            factory, mid, col_end, row_start, row_end, cell_w, cell_h,
        );

        // X < threshold ? left : right
        Box::new(Select {
            cond: Lt(X, threshold),
            if_true: left,
            if_false: right,
        })
    } else {
        // Split rows
        let mid = row_start + row_count / 2;
        let threshold = mid as f32 * cell_h;

        let top = build_channel_tree::<F, CHANNEL>(
            factory, col_start, col_end, row_start, mid, cell_w, cell_h,
        );
        let bottom = build_channel_tree::<F, CHANNEL>(
            factory, col_start, col_end, mid, row_end, cell_w, cell_h,
        );

        // Y < threshold ? top : bottom
        Box::new(Select {
            cond: Lt(Y, threshold),
            if_true: top,
            if_false: bottom,
        })
    }
}

// ============================================================================
// Solid Color (Background)
// ============================================================================

/// A constant color manifold - returns the same Discrete everywhere.
#[derive(Clone, Copy, Debug)]
pub struct SolidColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl SolidColor {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn from_rgba(rgba: [f32; 4]) -> Self {
        Self::new(rgba[0], rgba[1], rgba[2], rgba[3])
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }
}

impl Manifold for SolidColor {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Discrete {
        Discrete::pack(
            Field::from(self.r),
            Field::from(self.g),
            Field::from(self.b),
            Field::from(self.a),
        )
    }
}

// ============================================================================
// Constant Coverage (for empty cells or solid blocks)
// ============================================================================

/// A constant coverage manifold.
#[derive(Clone, Copy, Debug)]
pub struct ConstCoverage(pub f32);

impl Manifold for ConstCoverage {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Field {
        Field::from(self.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct MockFactory {
        cols: usize,
        rows: usize,
        cell_w: f32,
        cell_h: f32,
    }

    impl CellFactory for MockFactory {
        type Glyph = ConstCoverage;

        fn glyph(&self, _col: usize, _row: usize) -> Self::Glyph {
            ConstCoverage(1.0) // Full coverage
        }

        fn fg(&self, col: usize, row: usize) -> [f32; 4] {
            // Alternate white/black based on position
            if (col + row) % 2 == 0 {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            }
        }

        fn bg(&self, _col: usize, _row: usize) -> [f32; 4] {
            [0.5, 0.5, 0.5, 1.0]
        }

        fn dimensions(&self) -> (usize, usize) {
            (self.cols, self.rows)
        }

        fn cell_size(&self) -> (f32, f32) {
            (self.cell_w, self.cell_h)
        }
    }

    #[test]
    fn test_grid_construction() {
        let factory = MockFactory {
            cols: 4,
            rows: 4,
            cell_w: 8.0,
            cell_h: 16.0,
        };

        let grid = build_grid(&factory);

        // Evaluate at cell (0, 0) center - should be white (even position)
        let result = grid.eval_raw(
            Field::from(4.0),
            Field::from(8.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0u32; 16];
        result.store(&mut buf);

        let r = buf[0] & 0xFF;
        assert!(r > 200, "Expected white, got r={}", r);
    }

    #[test]
    fn test_cell_channel_blending() {
        let cell = Cell::new(
            ConstCoverage(0.5),
            [1.0, 0.0, 0.0, 1.0], // Red
            [0.0, 0.0, 1.0, 1.0], // Blue
        );

        // Use ColorManifold to pack channels
        let packed = ColorManifold::new(
            CellR::new(cell.clone()),
            CellG::new(cell.clone()),
            CellB::new(cell.clone()),
            CellA::new(cell),
        );

        let result = packed.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0u32; 16];
        result.store(&mut buf);

        // 50% coverage: R = 0.5*1.0 + 0.5*0.0 = 0.5 = 127
        // B = 0.5*0.0 + 0.5*1.0 = 0.5 = 127
        let r = buf[0] & 0xFF;
        let b = (buf[0] >> 16) & 0xFF;

        assert!(r > 100 && r < 160, "Expected ~127, got r={}", r);
        assert!(b > 100 && b < 160, "Expected ~127, got b={}", b);
    }

    #[test]
    fn test_solid_color() {
        let red = SolidColor::new(1.0, 0.0, 0.0, 1.0);

        let result = red.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let mut buf = [0u32; 16];
        result.store(&mut buf);

        let r = buf[0] & 0xFF;
        let g = (buf[0] >> 8) & 0xFF;
        let b = (buf[0] >> 16) & 0xFF;

        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }
}
