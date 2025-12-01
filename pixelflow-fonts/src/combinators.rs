use crate::curves::{Line, Quadratic, Segment};
use crate::font::Font;
use crate::glyph::{eval_curves, CellGlyph, CurveSurface, GlyphBounds};
use pixelflow_core::backend::SimdBatch;
use pixelflow_core::batch::Batch;
use pixelflow_core::ops::Baked;
use pixelflow_core::pipe::Surface;
use std::sync::{Arc, Mutex, OnceLock};

// ============================================================================
// Lazy
// ============================================================================

/// A surface wrapper that lazily evaluates and caches its result on the first use.
///
/// This is particularly useful for font rendering, where we want to defer the
/// costly "baking" (rasterizing to a texture) of a glyph until it is actually
/// requested by the render engine, and then share that baked texture across
/// all subsequent uses.
///
/// `Lazy` is thread-safe and cloneable (clones share the underlying cache).
pub struct Lazy<'a, S> {
    inner: Arc<LazyInner<'a, S>>,
}

struct LazyInner<'a, S> {
    cache: OnceLock<S>,
    factory: Mutex<Option<Box<dyn FnOnce() -> S + Send + Sync + 'a>>>,
}

impl<'a, S> Clone for Lazy<'a, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, S> Lazy<'a, S> {
    /// Creates a new `Lazy` surface from a factory function.
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce() -> S + Send + Sync + 'a,
    {
        Self {
            inner: Arc::new(LazyInner {
                cache: OnceLock::new(),
                factory: Mutex::new(Some(Box::new(f))),
            }),
        }
    }

    /// Access the inner surface, initializing it if necessary.
    pub fn get(&self) -> &S {
        self.inner.cache.get_or_init(|| {
            let mut lock = self.inner.factory.lock().expect("Lazy mutex poisoned");
            let f = lock.take().expect("Lazy factory already consumed");
            f()
        })
    }
}

impl<'a, S, P> Surface<P> for Lazy<'a, S>
where
    S: Surface<P>,
    P: pixelflow_core::Pixel,
{
    #[inline(always)]
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        self.get().eval(x, y)
    }
}

// ============================================================================
// Glyphs Factory
// ============================================================================

/// Returns a closure that yields shared, lazily-baked glyphs.
///
/// Glyphs are positioned within the cell using the font's ascender metric,
/// so that the baseline is correctly placed regardless of glyph shape.
pub fn glyphs<'a>(font: Font<'a>, w: u32, h: u32) -> impl Fn(char) -> Lazy<'a, Baked<u8>> {
    use std::collections::HashMap;
    use std::sync::RwLock;

    // Calculate the ascender position within the cell.
    // The cell height should fit the full line height (ascent - descent),
    // so we scale the ascender proportionally.
    let metrics = font.metrics();
    let line_height = metrics.ascent as f32 - metrics.descent as f32; // descent is negative
    let ascender = (metrics.ascent as f32 * h as f32 / line_height).round() as i32;

    let ascii: Vec<Lazy<'a, Baked<u8>>> = (0..128)
        .map(|i| {
            let c = i as u8 as char;
            let font = font.clone();
            Lazy::new(move || match font.glyph(c, h as f32) {
                Some(g) => {
                    let cell_glyph = CellGlyph::new(g, ascender);
                    Baked::new(&cell_glyph, w, h)
                }
                None => {
                    // Fallback for missing glyphs: return empty transparent surface
                    struct Empty;
                    impl Surface<u8> for Empty {
                        fn eval(&self, _: Batch<u32>, _: Batch<u32>) -> Batch<u8> {
                            Batch::<u8>::splat(0)
                        }
                    }
                    Baked::new(&Empty, w, h)
                }
            })
        })
        .collect();

    let other_cache: Arc<RwLock<HashMap<char, Lazy<'a, Baked<u8>>>>> =
        Arc::new(RwLock::new(HashMap::new()));

    move |c| {
        if (c as u32) < 128 {
            ascii[c as usize].clone()
        } else {
            // Check secondary cache for non-ASCII
            if let Ok(read) = other_cache.read() {
                if let Some(lazy) = read.get(&c) {
                    return lazy.clone();
                }
            }
            // Write lock to insert new glyph
            let mut write = other_cache.write().unwrap();
            if let Some(lazy) = write.get(&c) {
                return lazy.clone();
            }
            // Insert
            let font = font.clone();
            let lazy = Lazy::new(move || match font.glyph(c, h as f32) {
                Some(g) => {
                    let cell_glyph = CellGlyph::new(g, ascender);
                    Baked::new(&cell_glyph, w, h)
                }
                None => {
                    struct Empty;
                    impl Surface<u8> for Empty {
                        fn eval(&self, _: Batch<u32>, _: Batch<u32>) -> Batch<u8> {
                            Batch::<u8>::splat(0)
                        }
                    }
                    Baked::new(&Empty, w, h)
                }
            });
            write.insert(c, lazy.clone());
            lazy
        }
    }
}

// ============================================================================
// Combinators
// ============================================================================

// --- Bold ---

/// A surface combinator that simulates bold weight.
///
/// It works by dilating the Signed Distance Field of the underlying curve surface.
pub struct Bold<S> {
    pub source: S,
    pub amount: f32,
}

impl<S: CurveSurface> CurveSurface for Bold<S> {
    fn curves(&self) -> &[Segment] {
        self.source.curves()
    }
    fn bounds(&self) -> GlyphBounds {
        self.source.bounds()
    }
}

impl<S: CurveSurface> Surface<u8> for Bold<S> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        eval_curves(
            self.curves(),
            self.bounds(),
            x,
            y,
            Batch::<f32>::splat(self.amount),
        )
    }
}

impl<S: CurveSurface> Bold<S> {
    pub fn new(source: S, amount: f32) -> Self {
        Self { source, amount }
    }
}

// --- Hint (Stub) ---

/// A placeholder combinator for grid-fitting hints.
///
/// Currently a pass-through.
pub struct Hint<S> {
    source: S,
}

impl<S: CurveSurface> CurveSurface for Hint<S> {
    fn curves(&self) -> &[Segment] {
        self.source.curves()
    }
    fn bounds(&self) -> GlyphBounds {
        self.source.bounds()
    }
}

impl<S: CurveSurface> Surface<u8> for Hint<S> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        self.source.eval(x, y)
    }
}

// --- Slant ---

/// A surface combinator that applies a slant (shear) transformation.
///
/// Useful for synthesizing italic styles.
pub struct Slant<S> {
    source: S,
    _factor: f32,
    curves: Arc<[Segment]>,
}

impl<S: CurveSurface> CurveSurface for Slant<S> {
    fn curves(&self) -> &[Segment] {
        &self.curves
    }
    fn bounds(&self) -> GlyphBounds {
        self.source.bounds()
    }
}

impl<S: CurveSurface> Surface<u8> for Slant<S> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        eval_curves(self.curves(), self.bounds(), x, y, Batch::<f32>::splat(0.0))
    }
}

impl<S: CurveSurface> Slant<S> {
    pub fn new(source: S, factor: f32) -> Self {
        let curves = source
            .curves()
            .iter()
            .map(|seg| match seg {
                Segment::Line(l) => Segment::Line(Line {
                    p0: [l.p0[0] + l.p0[1] * factor, l.p0[1]],
                    p1: [l.p1[0] + l.p1[1] * factor, l.p1[1]],
                }),
                Segment::Quad(q) => {
                    let p0 = [q.p0[0] + q.p0[1] * factor, q.p0[1]];
                    let p1 = [q.p1[0] + q.p1[1] * factor, q.p1[1]];
                    let p2 = [q.p2[0] + q.p2[1] * factor, q.p2[1]];
                    if let Some(new_q) = Quadratic::try_new(p0, p1, p2) {
                        Segment::Quad(new_q)
                    } else {
                        // Fallback to line if quadratic becomes degenerate (unlikely with shear)
                        Segment::Line(Line { p0, p1: p2 })
                    }
                }
            })
            .collect::<Vec<_>>();

        Self {
            source,
            _factor: factor,
            curves: Arc::from(curves),
        }
    }
}

// --- Scale ---

/// A surface combinator that scales the geometry of the glyph.
pub struct Scale<S> {
    source: S,
    factor: f32,
    curves: Arc<[Segment]>,
}

impl<S: CurveSurface> CurveSurface for Scale<S> {
    fn curves(&self) -> &[Segment] {
        &self.curves
    }
    fn bounds(&self) -> GlyphBounds {
        let b = self.source.bounds();
        GlyphBounds {
            width: (b.width as f32 * self.factor).ceil() as u32,
            height: (b.height as f32 * self.factor).ceil() as u32,
            bearing_x: (b.bearing_x as f32 * self.factor) as i32,
            bearing_y: (b.bearing_y as f32 * self.factor) as i32,
        }
    }
}

impl<S: CurveSurface> Surface<u8> for Scale<S> {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<u8> {
        eval_curves(self.curves(), self.bounds(), x, y, Batch::<f32>::splat(0.0))
    }
}

impl<S: CurveSurface> Scale<S> {
    pub fn new(source: S, factor: f32) -> Self {
        let curves = source
            .curves()
            .iter()
            .map(|seg| match seg {
                Segment::Line(l) => Segment::Line(Line {
                    p0: [l.p0[0] * factor, l.p0[1] * factor],
                    p1: [l.p1[0] * factor, l.p1[1] * factor],
                }),
                Segment::Quad(q) => {
                    let p0 = [q.p0[0] * factor, q.p0[1] * factor];
                    let p1 = [q.p1[0] * factor, q.p1[1] * factor];
                    let p2 = [q.p2[0] * factor, q.p2[1] * factor];
                    if let Some(new_q) = Quadratic::try_new(p0, p1, p2) {
                        Segment::Quad(new_q)
                    } else {
                        Segment::Line(Line { p0, p1: p2 })
                    }
                }
            })
            .collect::<Vec<_>>();

        Self {
            source,
            factor,
            curves: Arc::from(curves),
        }
    }
}

// ============================================================================
// Extensions
// ============================================================================

/// Extension trait for `CurveSurface` types.
///
/// Provides a fluent API for applying combinators.
pub trait CurveSurfaceExt: CurveSurface + Sized {
    /// Applies a grid-fitting hint (currently a no-op).
    fn hint(self, _grid_size: f32) -> Hint<Self> {
        Hint { source: self }
    }

    /// Makes the glyph bolder by dilating the shape.
    ///
    /// `amount` is in pixels (e.g. 0.5 for half a pixel width increase).
    fn bold(self, amount: f32) -> Bold<Self> {
        Bold::new(self, amount)
    }

    /// Slants the glyph (simulated italic).
    ///
    /// `factor` is the shear amount (e.g. 0.2).
    fn slant(self, factor: f32) -> Slant<Self> {
        Slant::new(self, factor)
    }

    /// Scales the glyph curves.
    fn scale_curve(self, factor: f32) -> Scale<Self> {
        Scale::new(self, factor)
    }
}

impl<S: CurveSurface> CurveSurfaceExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyph::eval_curves_cell;

    #[test]
    fn debug_letter_f_rendering() {
        // Debug test: check what's happening with 'f' which has curves
        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let cell_width = 10u32;
        let cell_height = 16u32;

        let glyph = font.glyph('f', cell_height as f32).expect("No f glyph");
        let bounds = glyph.bounds();
        let metrics = font.metrics();
        let line_height = metrics.ascent as f32 - metrics.descent as f32;
        let ascender = (metrics.ascent as f32 * cell_height as f32 / line_height).round() as i32;

        eprintln!("=== Letter 'f' Debug ===");
        eprintln!("Ascender: {}, bounds: {:?}", ascender, bounds);
        eprintln!("Number of curves: {}", glyph.curves().len());

        // Count lines vs quads
        let mut lines = 0;
        let mut quads = 0;
        for seg in glyph.curves() {
            match seg {
                crate::curves::Segment::Line(_) => lines += 1,
                crate::curves::Segment::Quad(_) => quads += 1,
            }
        }
        eprintln!("Lines: {}, Quads: {}", lines, quads);

        let cell_glyph = CellGlyph::new(glyph.clone(), ascender);

        // Render point-by-point
        eprintln!("\nPoint-by-point 'f' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = cell_glyph.eval(x_batch, y_batch).first();
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", y, row);
        }

        // Test a specific point that might be problematic
        // Let's check (0, 0) - top-left corner
        let cx = 0.5 + bounds.bearing_x as f32;
        let cy = ascender as f32 - 0.5;
        eprintln!("\nPoint (0, 0) -> curve ({}, {})", cx, cy);
        eprintln!(
            "Glyph y range: {} to {} (bearing_y={})",
            bounds.bearing_y as f32 - bounds.height as f32,
            bounds.bearing_y,
            bounds.bearing_y
        );

        let mut total_winding: i32 = 0;
        let mut min_dist: f32 = 1000.0;
        for (i, seg) in glyph.curves().iter().enumerate() {
            let cx_batch = Batch::<f32>::splat(cx);
            let cy_batch = Batch::<f32>::splat(cy);
            let w = seg.winding_batch(cx_batch, cy_batch).first();
            let d = seg.min_distance_batch(cx_batch, cy_batch).first();
            let w_signed = w as i32; // wrapping handles negative
            total_winding = total_winding.wrapping_add(w_signed);
            min_dist = min_dist.abs().min(d.abs());

            // Print segments that might be contributing
            if d.abs() < 50.0 {
                // Also print the segment bbox
                let (seg_min_y, seg_max_y) = match seg {
                    crate::curves::Segment::Line(l) => (l.p0[1].min(l.p1[1]), l.p0[1].max(l.p1[1])),
                    crate::curves::Segment::Quad(q) => (
                        q.p0[1].min(q.p1[1]).min(q.p2[1]),
                        q.p0[1].max(q.p1[1]).max(q.p2[1]),
                    ),
                };
                let is_quad = matches!(seg, crate::curves::Segment::Quad(_));
                eprintln!(
                    "  Seg {} ({}): winding={}, dist={:.3}, y_range=[{:.1}, {:.1}]",
                    i,
                    if is_quad { "Q" } else { "L" },
                    w,
                    d,
                    seg_min_y,
                    seg_max_y
                );
            }
        }
        eprintln!(
            "Total winding (raw): {}, min_dist: {:.3}",
            total_winding, min_dist
        );

        // The problem: min_dist is 0.444 for a point that's 4.5 units ABOVE the glyph!
        // This small distance causes antialiasing to produce non-zero alpha.
        //
        // alpha = 0.5 - signed_dist = 0.5 - 0.444 = 0.056 -> 14 (the ',' character)
        //
        // The fix: we should only trust the Loop-Blinn distance when the point is
        // within the curve's parameter range (0 <= t <= 1).
    }

    #[test]
    fn debug_period_winding_at_top() {
        // Debug test: check what winding/distance we get for top of cell
        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let cell_height = 16u32;
        let glyph = font
            .glyph('.', cell_height as f32)
            .expect("No period glyph");
        let bounds = glyph.bounds();
        let metrics = font.metrics();
        let line_height = metrics.ascent as f32 - metrics.descent as f32;
        let ascender = (metrics.ascent as f32 * cell_height as f32 / line_height).round() as i32;

        eprintln!("Ascender: {}, Period bounds: {:?}", ascender, bounds);
        eprintln!("Number of curves: {}", glyph.curves().len());
        eprintln!("Period curves:");
        for (i, seg) in glyph.curves().iter().enumerate() {
            eprintln!("  {}: {:?}", i, seg);
        }

        // Test a point at top of cell (should be far from period)
        // Cell (0, 0) -> curve (bearing_x + 0.5, ascender - 0.5) = (4.5, 16.5)
        // Period is around y=0-2, so y=16.5 is ~14.5 pixels away
        let x = Batch::<u32>::splat(0);
        let y = Batch::<u32>::splat(0);
        let alpha = eval_curves_cell(
            glyph.curves(),
            bounds,
            ascender,
            x,
            y,
            Batch::<f32>::splat(0.0),
        );
        eprintln!("Alpha at (0, 0) via eval_curves_cell: {}", alpha.first());

        // Now test via CellGlyph
        let cell_glyph = CellGlyph::new(glyph.clone(), ascender);
        let alpha2 = cell_glyph
            .eval(Batch::<u32>::splat(0), Batch::<u32>::splat(0))
            .first();
        eprintln!("Alpha at (0, 0) via CellGlyph.eval: {}", alpha2);

        // Test (2, 0) which showed 107 in the point-by-point
        let alpha3 = cell_glyph
            .eval(Batch::<u32>::splat(2), Batch::<u32>::splat(0))
            .first();
        eprintln!("Alpha at (2, 0) via CellGlyph.eval: {}", alpha3);

        // Debug the coordinate transformation for (2, 0)
        // cx = 2 + 0.5 + bearing_x = 2.5 + 4 = 6.5
        // cy = ascender - 0.5 = 17 - 0.5 = 16.5
        eprintln!(
            "For (2, 0): cx = 2 + 0.5 + {} = {}, cy = {} - 0.5 = {}",
            bounds.bearing_x,
            2.5 + bounds.bearing_x as f32,
            ascender,
            ascender as f32 - 0.5
        );

        // Manually compute winding and distance for this point
        use crate::curves::Segment;
        let cx = 2.5 + bounds.bearing_x as f32;
        let cy = ascender as f32 - 0.5;
        let mut total_winding: i32 = 0;
        let mut min_dist: f32 = 1000.0;
        for (i, seg) in glyph.curves().iter().enumerate() {
            let cx_batch = Batch::<f32>::splat(cx);
            let cy_batch = Batch::<f32>::splat(cy);
            let w = seg.winding_batch(cx_batch, cy_batch).first();
            let d = seg.min_distance_batch(cx_batch, cy_batch).first();
            // winding is u32 with wrapping: 1 = +1, 0xFFFFFFFF = -1
            let w_signed = if w == 0 {
                0i32
            } else if w == 1 {
                1
            } else {
                -1
            };
            total_winding += w_signed;
            min_dist = min_dist.abs().min(d.abs());
            if w != 0 || d.abs() < 20.0 {
                eprintln!("  Seg {}: winding={} (u32={}), dist={}", i, w_signed, w, d);
            }
        }
        eprintln!("Total winding: {}, min_dist: {}", total_winding, min_dist);
        // alpha = 0.5 - signed_dist, where signed_dist = winding != 0 ? -min_dist : min_dist
        let signed_dist = if total_winding != 0 {
            -min_dist
        } else {
            min_dist
        };
        let alpha_calc = (0.5 - signed_dist).max(0.0).min(1.0) * 255.0;
        eprintln!("Calculated alpha: {}", alpha_calc);

        // Test center of cell
        let x = Batch::<u32>::splat(5);
        let y = Batch::<u32>::splat(8);
        let alpha = eval_curves_cell(
            glyph.curves(),
            bounds,
            ascender,
            x,
            y,
            Batch::<f32>::splat(0.0),
        );
        eprintln!("Alpha at (5, 8): {}", alpha.first());

        // Test bottom of cell (near the period)
        let x = Batch::<u32>::splat(5);
        let y = Batch::<u32>::splat(15);
        let alpha = eval_curves_cell(
            glyph.curves(),
            bounds,
            ascender,
            x,
            y,
            Batch::<f32>::splat(0.0),
        );
        eprintln!("Alpha at (5, 15): {}", alpha.first());
    }

    #[test]
    fn direct_bake_period() {
        // Test baking directly from CellGlyph without going through glyphs()
        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let cell_width = 10u32;
        let cell_height = 16u32;

        let glyph = font
            .glyph('.', cell_height as f32)
            .expect("No period glyph");
        let metrics = font.metrics();
        let line_height = metrics.ascent as f32 - metrics.descent as f32;
        let ascender = (metrics.ascent as f32 * cell_height as f32 / line_height).round() as i32;

        let cell_glyph = CellGlyph::new(glyph.clone(), ascender);

        // First, test point-by-point evaluation
        eprintln!("Point-by-point eval '.' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = cell_glyph.eval(x_batch, y_batch).first();
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", y, row);
        }

        // Now bake
        let baked: Baked<u8> = Baked::new(&cell_glyph, cell_width, cell_height);

        eprintln!("\nBaked '.' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                let alpha = baked.data()[(y * cell_width + x) as usize];
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", y, row);
        }

        // Check top half is transparent
        for y in 0..8u32 {
            for x in 0..cell_width {
                let alpha = baked.data()[(y * cell_width + x) as usize];
                assert_eq!(
                    alpha, 0,
                    "Direct bake: pixel ({}, {}) should be 0, got {}",
                    x, y, alpha
                );
            }
        }
    }

    #[test]
    fn baked_period_top_half_is_transparent() {
        // When we bake a '.' into a cell-sized buffer, the top half should be empty.
        // The period is a small dot near the baseline - it should only occupy
        // the bottom portion of the cell.

        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let cell_width = 10u32;
        let cell_height = 16u32;

        // First, let's understand the glyph metrics
        let glyph = font
            .glyph('.', cell_height as f32)
            .expect("No period glyph");
        let bounds = glyph.bounds();
        let metrics = font.metrics();
        let line_height = metrics.ascent as f32 - metrics.descent as f32;
        let ascender = (metrics.ascent as f32 * cell_height as f32 / line_height).round() as i32;

        eprintln!("Font metrics: {:?}", metrics);
        eprintln!("Line height: {}, Ascender: {}", line_height, ascender);
        eprintln!("Period bounds: {:?}", bounds);

        // The key insight: bearing_y is the TOP of the glyph bbox in curve space,
        // measured from the BASELINE (y=0 in font coords).
        //
        // For a period:
        // - bearing_y might be ~2-3 (top of the dot is slightly above baseline)
        // - The dot sits just above y=0 (baseline)
        //
        // When we sample at cell coordinate (px, py), the current code does:
        //   cx = px + 0.5 + bearing_x
        //   cy = bearing_y - (py + 0.5)
        //
        // For cell row 0 (top of cell):
        //   cy = bearing_y - 0.5 = ~2.5 (which is INSIDE the period!)
        //
        // This is WRONG. Cell row 0 should map to the TOP of the cell,
        // which is at y = ascender in curve space (not bearing_y).

        // Get the baked glyph using the glyphs factory (same as terminal uses)
        let glyph_fn = glyphs(font, cell_width, cell_height);
        let baked_lazy = glyph_fn('.');
        let baked: &Baked<u8> = baked_lazy.get();

        // Print the baked buffer for debugging
        eprintln!("Baked '.' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = baked.eval(x_batch, y_batch).first();
                let ch = if alpha > 200 {
                    '#'
                } else if alpha > 100 {
                    '+'
                } else if alpha > 50 {
                    '.'
                } else if alpha > 0 {
                    ','
                } else {
                    ' '
                };
                row.push(ch);
            }
            eprintln!("{:2}: |{}|", y, row);
        }

        // The top half (rows 0 through cell_height/2 - 1) should be completely transparent
        let top_half_end = cell_height / 2;
        for y in 0..top_half_end {
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = baked.eval(x_batch, y_batch).first();
                assert_eq!(
                    alpha, 0,
                    "Top half pixel ({}, {}) should be transparent, got {}",
                    x, y, alpha
                );
            }
        }

        // The bottom portion should have SOME opaque pixels (the actual dot)
        let mut found_opaque = false;
        for y in top_half_end..cell_height {
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = baked.eval(x_batch, y_batch).first();
                if alpha > 200 {
                    found_opaque = true;
                    break;
                }
            }
        }
        assert!(found_opaque, "Bottom half should contain the period dot");
    }
}
