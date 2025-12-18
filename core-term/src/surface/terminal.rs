//! Terminal surface implementation.
//!
//! The TerminalSurface constructs an algebraic surface composition representing the
//! current terminal state using the Partition combinator for O(1) cell lookup.

use crate::surface::grid::GridBuffer;
use core::marker::PhantomData;
use pixelflow_core::batch::Batch;
use pixelflow_core::dsl::MaskExt; // for .over()
use pixelflow_core::surfaces::{Baked, Compute, Partition};
use pixelflow_core::traits::Manifold;
use pixelflow_graphics::fonts::{glyphs, Lazy};
use pixelflow_graphics::render::{font, Pixel};

use core::fmt::Debug;
use std::sync::Arc;

/// Converts a Manifold<u32, C> to Manifold<P, C> by applying P::batch_from_u32.
struct PixelConvert<S, P> {
    source: S,
    _pixel: PhantomData<P>,
}

impl<S, P: Pixel> PixelConvert<S, P> {
    fn new(source: S) -> Self {
        Self {
            source,
            _pixel: PhantomData,
        }
    }
}

impl<S, P, C> Manifold<P, C> for PixelConvert<S, P>
where
    S: Manifold<u32, C>,
    P: Pixel,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<P> {
        P::batch_from_u32(self.source.eval(x, y, z, w))
    }
}

/// Glyph factory type - closure that returns lazily-baked glyphs.
/// Glyphs are u32 pixels with coverage in alpha channel (R=G=B=255, A=coverage).
type GlyphFactory = Arc<dyn Fn(char) -> Lazy<'static, Baked<u32>> + Send + Sync>;

/// A terminal rendered as a functional surface.
///
/// This struct does not implement `eval` manually. Instead, it constructs
/// a `Arc<dyn Manifold>` pipeline that represents the grid.
pub struct TerminalSurface<P: Pixel> {
    /// The constructed pipeline for the current frame.
    pipeline: Arc<dyn Manifold<P, u32> + Send + Sync>,
}

impl<P: Pixel> TerminalSurface<P> {
    /// Creates a new terminal surface from a grid.
    ///
    /// Uses Partition combinator with compositional indexer for O(1) cell lookup.
    pub fn with_grid(
        grid: &GridBuffer,
        glyph_factory: GlyphFactory,
        cell_width: u32,
        cell_height: u32,
    ) -> Self {
        let rows = grid.rows;
        let cols = grid.cols;

        // Build flat array of all cells (row-major order)
        let mut cell_surfaces: Vec<Arc<dyn Manifold<u32, u32> + Send + Sync>> =
            Vec::with_capacity(rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                let cell = grid.get(col, row);
                let cx = (col as u32) * cell_width;
                let cy = (row as u32) * cell_height;

                // Glyph Mask (u32 pixels with coverage in alpha channel)
                // Use a constant Manifold for empty cells
                let glyph_surface: Arc<dyn Manifold<u32, u32> + Send + Sync> =
                    if cell.ch == ' ' || cell.ch == '\0' {
                        Arc::new(0u32)
                    } else {
                        let glyph_lazy = (glyph_factory)(cell.ch);
                        Arc::new(pixelflow_core::surfaces::Offset {
                            source: glyph_lazy,
                            dx: -(cx as i32),
                            dy: -(cy as i32),
                        })
                    };

                // Colors (as u32 packed RGBA)
                let fg = cell.fg.to_rgba().0;
                let bg = cell.bg.to_rgba().0;

                // Compose: Glyph.over(FG, BG)
                let cell_surface = glyph_surface.over(fg, bg);
                cell_surfaces.push(Arc::new(cell_surface));
            }
        }

        // Compositional indexer: (x, y) -> cell_index
        // cell_index = (y / cell_h) * cols + (x / cell_w)
        let indexer = Compute::new(
            move |x: Batch<u32>, y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| -> Batch<u32> {
                let col_idx = x / Batch::<u32>::splat(cell_width);
                let row_idx = y / Batch::<u32>::splat(cell_height);
                row_idx * Batch::<u32>::splat(cols as u32) + col_idx
            },
        );

        // Build Partition combinator
        // Type erase to remove Send + Sync bounds
        let cell_surfaces_erased: Vec<Arc<dyn Manifold<u32, u32>>> = cell_surfaces
            .into_iter()
            .map(|s| s as Arc<dyn Manifold<u32, u32>>)
            .collect();
        let u32_root = Partition::new(indexer, cell_surfaces_erased);

        // Convert to P
        Self {
            pipeline: Arc::new(PixelConvert::<_, P>::new(u32_root)),
        }
    }

    /// Creates a terminal surface from a pre-computed flat list of cells.
    /// This allows for incremental updates (caching cells).
    pub fn from_cells(
        cells: Vec<Arc<dyn Manifold<u32, u32>>>,
        cols: usize,
        cell_width: u32,
        cell_height: u32,
    ) -> Self {
        // Compositional indexer: (x, y) -> cell_index
        let indexer = Compute::new(
            move |x: Batch<u32>, y: Batch<u32>, _z: Batch<u32>, _w: Batch<u32>| -> Batch<u32> {
                let col_idx = x / Batch::<u32>::splat(cell_width);
                let row_idx = y / Batch::<u32>::splat(cell_height);
                row_idx * Batch::<u32>::splat(cols as u32) + col_idx
            },
        );

        let u32_root = Partition::new(indexer, cells);

        Self {
            pipeline: Arc::new(PixelConvert::<_, P>::new(u32_root)),
        }
    }
}

// Forward Manifold impl to the pipeline
impl<P: Pixel + Manifold<P>> Manifold<P> for TerminalSurface<P> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>, z: Batch<u32>, w: Batch<u32>) -> Batch<P> {
        Manifold::eval(&self.pipeline, x, y, z, w)
    }
}

// -----------------------------------------------------------------------------
// Backward compatibility constructors (for tests/app)
// -----------------------------------------------------------------------------
impl<P: Pixel + 'static> TerminalSurface<P> {
    pub fn new(cols: usize, rows: usize, cell_width: u32, cell_height: u32) -> Self {
        let f = font();
        let glyph_fn = glyphs(f.clone(), cell_width, cell_height);
        let grid = GridBuffer::new(cols, rows);

        Self::with_grid(&grid, Arc::new(glyph_fn), cell_width, cell_height)
    }
}
