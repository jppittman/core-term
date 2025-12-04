//! Terminal surface implementation.
//!
//! The TerminalSurface constructs an algebraic surface composition representing the
//! current terminal state. It uses `Select` and `Fix`-like recursion (laundered via
//! `Arc<dyn Surface>`) to build a scene graph of cells.

use crate::surface::grid::GridBuffer;
use core::marker::PhantomData;
use pixelflow_core::dsl::MaskExt; // for .over()
use pixelflow_core::geometry::Rect;
use pixelflow_core::surfaces::{Baked, Select};
use pixelflow_core::traits::Surface;
use pixelflow_core::batch::Batch;
use pixelflow_fonts::{glyphs, Lazy};
use pixelflow_render::font;
use pixelflow_render::Pixel;
use std::sync::Arc;

/// Glyph factory type - closure that returns lazily-baked glyphs.
type GlyphFactory = Arc<dyn Fn(char) -> Lazy<'static, Baked<u8>> + Send + Sync>;

/// A terminal rendered as a functional surface.
///
/// This struct does not implement `eval` manually. Instead, it constructs
/// a `Arc<dyn Surface>` pipeline that represents the grid.
pub struct TerminalSurface<P: Pixel + Surface<P>> {
    /// The constructed pipeline for the current frame.
    pipeline: Arc<dyn Surface<P> + Send + Sync>,
    /// Keep grid alive if needed, though we might just consume it for construction.
    /// For now, we keep it to match previous struct shape if useful, but really
    /// we just need the pipeline.
    _pixel: PhantomData<P>,
}

impl<P: Pixel + Surface<P> + 'static> TerminalSurface<P> {
    /// Creates a new terminal surface from a grid.
    ///
    /// This builds the entire scene graph.
    pub fn with_grid(
        grid: &GridBuffer,
        glyph_factory: GlyphFactory,
        cell_width: u32,
        cell_height: u32,
    ) -> Self {
        let rows = grid.rows;
        // Build all rows from scratch
        let row_surfaces: Vec<Arc<dyn Surface<P> + Send + Sync>> = (0..rows)
            .map(|row_idx| {
                build_cells(
                    row_idx,
                    0,
                    grid.cols,
                    grid,
                    &glyph_factory,
                    cell_width,
                    cell_height
                )
            })
            .collect();

        Self::from_rows(row_surfaces, cell_height)
    }

    /// Creates a terminal surface from a pre-computed list of row surfaces.
    /// This allows for incremental updates (caching rows).
    pub fn from_rows(
        rows: Vec<Arc<dyn Surface<P> + Send + Sync>>,
        cell_height: u32,
    ) -> Self {
        let root = build_row_tree(0, rows.len(), &rows, cell_height);
        
        Self {
            pipeline: root,
            _pixel: PhantomData,
        }
    }
}

// Recursive builder for the Tree of Rows (combines pre-built row surfaces)
fn build_row_tree<P: Pixel + Surface<P> + 'static>(
    start_row_idx: usize,
    count: usize,
    rows: &[Arc<dyn Surface<P> + Send + Sync>],
    cell_h: u32,
) -> Arc<dyn Surface<P> + Send + Sync> {
    if count == 0 {
        return Arc::new(P::from_u32(0)) as Arc<dyn Surface<P> + Send + Sync>;
    }
    if count == 1 {
        // Leaf of the row tree is one of our pre-built rows
        let row_surface = rows[start_row_idx].clone();
        
        // Mask to row height
        let row_y_start = (start_row_idx as u32) * cell_h;
        let mask = Rect::new(0, row_y_start as i32, u32::MAX, cell_h);
        
        return Arc::new(Select {
            mask,
            if_true: row_surface,
            if_false: Arc::new(P::from_u32(0)) as Arc<dyn Surface<P> + Send + Sync>
        });
    }
    
    // Split
    let half = count / 2;
    let top = build_row_tree(start_row_idx, half, rows, cell_h);
    let bottom = build_row_tree(start_row_idx + half, count - half, rows, cell_h);
    
    // Split Y at boundary
    let split_y = ((start_row_idx + half) as u32) * cell_h;
    let split_mask = Rect::new(0, 0, u32::MAX, split_y);
    
    Arc::new(Select {
        mask: split_mask,
        if_true: top,
        if_false: bottom
    })
}

// Recursive builder for Cells within a Row (Balanced Tree)
// Note: This is now public so the Worker can call it incrementally
pub fn build_cells<P: Pixel + Surface<P> + 'static>(
    row_idx: usize,
    start_col: usize,
    count: usize,
    grid: &GridBuffer,
    glyph_factory: &GlyphFactory,
    cell_w: u32,
    cell_h: u32,
) -> Arc<dyn Surface<P> + Send + Sync> {
    if count == 0 {
        return Arc::new(P::from_u32(0)) as Arc<dyn Surface<P> + Send + Sync>;
    }
    
    if count == 1 {
        let current_col_idx = start_col;
        let cell = grid.get(current_col_idx, row_idx);
        
        let cx = (current_col_idx as u32) * cell_w;
        let cy = (row_idx as u32) * cell_h;
        
        // 1. Glyph Mask
        let glyph_surface: Arc<dyn Surface<u8> + Send + Sync> = if cell.ch == ' ' || cell.ch == '\0' {
            Arc::new(Rect::new(0, 0, 0, 0))
        } else {
            // Use the Lazy surface directly - it caches itself on first eval
            let glyph_lazy = (glyph_factory)(cell.ch);
            
            Arc::new(pixelflow_core::surfaces::Offset {
                source: glyph_lazy, // No .get().clone() or Baked wrapper needed
                dx: -(cx as i32),
                dy: -(cy as i32),
            })
        };
        
        // 2. Colors
        let fg = P::from_u32(cell.fg.to_rgba().0);
        let bg = P::from_u32(cell.bg.to_rgba().0);
        
        // 3. Compose: Glyph.over(FG, BG)
        // Note: .over() creates an Over struct. We box (Arc) it to erase the type.
        let cell_surface = glyph_surface.over(fg, bg);
        
        // 4. Mask
        let cell_mask = Rect::new(cx as i32, 0, cell_w, u32::MAX);
        
        return Arc::new(Select {
            mask: cell_mask,
            if_true: Arc::new(cell_surface) as Arc<dyn Surface<P> + Send + Sync>,
            if_false: Arc::new(P::from_u32(0)) as Arc<dyn Surface<P> + Send + Sync>
        });
    }
    
    // Split
    let half = count / 2;
    let left = build_cells(row_idx, start_col, half, grid, glyph_factory, cell_w, cell_h);
    let right = build_cells(row_idx, start_col + half, count - half, grid, glyph_factory, cell_w, cell_h);
    
    // Split X at boundary
    let split_x = ((start_col + half) as u32) * cell_w;
    let split_mask = Rect::new(0, 0, split_x, u32::MAX);
    
    Arc::new(Select {
        mask: split_mask,
        if_true: left,
        if_false: right
    })
}

// Forward Surface impl to the pipeline
impl<P: Pixel + Surface<P>> Surface<P> for TerminalSurface<P> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        self.pipeline.eval(x, y)
    }
}

// -----------------------------------------------------------------------------
// Backward compatibility constructors (for tests/app)
// -----------------------------------------------------------------------------
impl<P: Pixel + Surface<P> + 'static> TerminalSurface<P> {
    pub fn new(cols: usize, rows: usize, cell_width: u32, cell_height: u32) -> Self {
        let f = font();
        let glyph_fn = glyphs(f.clone(), cell_width, cell_height);
        let grid = GridBuffer::new(cols, rows);
        
        Self::with_grid(&grid, Arc::new(glyph_fn), cell_width, cell_height)
    }
}