use crate::curves::{Line, Quadratic, Segment};
use crate::font::Font;
use crate::glyph::{eval_curves, CellGlyph, CurveSurface, GlyphBounds};
use pixelflow_core::batch::Batch;
use pixelflow_core::surfaces::{Baked, Rasterize};
use pixelflow_core::traits::Manifold;
use pixelflow_core::SimdBatch;
use core::fmt::Debug;
use std::sync::{Arc, Mutex, OnceLock};

// ============================================================================
// Lazy
// ============================================================================

/// A surface wrapper that lazily evaluates and caches its result on the first use.
#[derive(Clone)]
pub struct Lazy<'a, S> {
    inner: Arc<LazyInner<'a, S>>,
}

struct LazyInner<'a, S> {
    cache: OnceLock<S>,
    factory: Mutex<Option<Box<dyn FnOnce() -> S + Send + Sync + 'a>>>,
}

impl<'a, S> Lazy<'a, S> {
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

    pub fn get(&self) -> &S {
        self.inner.cache.get_or_init(|| {
            let mut lock = self.inner.factory.lock().expect("Lazy mutex poisoned");
            let f = lock.take().expect("Lazy factory already consumed");
            f()
        })
    }
}

impl<'a, S, P, C> Manifold<P, C> for Lazy<'a, S>
where
    S: Manifold<P, C>,
    P: pixelflow_core::Pixel,
    C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: Batch<C>, y: Batch<C>, z: Batch<C>, w: Batch<C>) -> Batch<P> {
        self.get().eval(x, y, z, w)
    }
}

// ============================================================================
// Glyphs Factory
// ============================================================================

pub fn glyphs<'a>(font: Font<'a>, w: u32, h: u32) -> impl Fn(char) -> Lazy<'a, Baked<u32>> {
    use std::collections::HashMap;
    use std::sync::RwLock;

    let metrics = font.metrics();
    let line_height = metrics.ascent as f32 - metrics.descent as f32;
    let ascender = (metrics.ascent as f32 * h as f32 / line_height).floor() as i32;
    let glyph_size = h as f32 * metrics.units_per_em as f32 / line_height;

    let ascii: Vec<Lazy<'a, Baked<u32>>> = (0..128)
        .map(|i| {
            let c = i as u8 as char;
            let font = font.clone();
            Lazy::new(move || match font.glyph(c, glyph_size) {
                Some(g) => {
                    let cell_glyph = CellGlyph::new(g, ascender);
                    let rasterized = Rasterize(cell_glyph);
                    Baked::new(&rasterized, w, h)
                }
                None => {
                    struct Empty;
                    impl Manifold<u32> for Empty {
                        fn eval(&self, _: Batch<u32>, _: Batch<u32>, _: Batch<u32>, _: Batch<u32>) -> Batch<u32> {
                            Batch::<u32>::splat(0x00FFFFFF)
                        }
                    }
                    Baked::new(&Empty, w, h)
                }
            })
        })
        .collect();

    let other_cache: Arc<RwLock<HashMap<char, Lazy<'a, Baked<u32>>>>> =
        Arc::new(RwLock::new(HashMap::new()));

    move |c| {
        if (c as u32) < 128 {
            ascii[c as usize].clone()
        } else {
            if let Ok(read) = other_cache.read() {
                if let Some(lazy) = read.get(&c) {
                    return lazy.clone();
                }
            }
            let mut write = other_cache.write().unwrap();
            if let Some(lazy) = write.get(&c) {
                return lazy.clone();
            }
            let font = font.clone();
            let lazy = Lazy::new(move || match font.glyph(c, glyph_size) {
                Some(g) => {
                    let cell_glyph = CellGlyph::new(g, ascender);
                    let rasterized = Rasterize(cell_glyph);
                    Baked::new(&rasterized, w, h)
                }
                None => {
                    struct Empty;
                    impl Manifold<u32> for Empty {
                        fn eval(&self, _: Batch<u32>, _: Batch<u32>, _: Batch<u32>, _: Batch<u32>) -> Batch<u32> {
                            Batch::<u32>::splat(0x00FFFFFF)
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

impl<S: CurveSurface> Manifold<u32, f32> for Bold<S> {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, _z: Batch<f32>, _w: Batch<f32>) -> Batch<u32> {
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

impl<S: CurveSurface> Manifold<u32, f32> for Hint<S> {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, z: Batch<f32>, w: Batch<f32>) -> Batch<u32> {
        self.source.eval(x, y, z, w)
    }
}

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

impl<S: CurveSurface> Manifold<u32, f32> for Slant<S> {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, _z: Batch<f32>, _w: Batch<f32>) -> Batch<u32> {
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

impl<S: CurveSurface> Manifold<u32, f32> for Scale<S> {
    fn eval(&self, x: Batch<f32>, y: Batch<f32>, _z: Batch<f32>, _w: Batch<f32>) -> Batch<u32> {
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

pub trait CurveSurfaceExt: CurveSurface + Sized {
    fn hint(self, _grid_size: f32) -> Hint<Self> {
        Hint { source: self }
    }
    fn bold(self, amount: f32) -> Bold<Self> {
        Bold::new(self, amount)
    }
    fn slant(self, factor: f32) -> Slant<Self> {
        Slant::new(self, factor)
    }
    fn scale_curve(self, factor: f32) -> Scale<Self> {
        Scale::new(self, factor)
    }
}

impl<S: CurveSurface> CurveSurfaceExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyph::eval_curves_cell;
    use pixelflow_core::backend::SimdBatch;
    use pixelflow_core::batch::NativeBackend;
    use pixelflow_core::backend::Backend;
    use pixelflow_core::traits::Surface; // Needed for eval_one

    fn pixel_alpha(pixel: u32) -> u8 {
        (pixel >> 24) as u8
    }

    #[test]
    fn debug_letter_f_rendering() {
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

        let cell_glyph = CellGlyph::new(glyph.clone(), ascender);

        eprintln!("\nPoint-by-point 'f' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                // Update: convert u32 to f32 + 0.5
                let x_f = NativeBackend::u32_to_f32(Batch::<u32>::splat(x)) + Batch::<f32>::splat(0.5);
                let y_f = NativeBackend::u32_to_f32(Batch::<u32>::splat(y)) + Batch::<f32>::splat(0.5);
                let z_f = Batch::<f32>::splat(0.0);
                let w_f = Batch::<f32>::splat(0.0);

                let alpha = pixel_alpha(cell_glyph.eval(x_f, y_f, z_f, w_f).first());
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
    }

    #[test]
    fn debug_period_winding_at_top() {
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

        // Update to use f32
        let x = Batch::<f32>::splat(0.5); // x=0 -> 0.5
        let y = Batch::<f32>::splat(0.5); // y=0 -> 0.5
        let z = Batch::<f32>::splat(0.0);
        let w = Batch::<f32>::splat(0.0);

        let alpha = eval_curves_cell(
            glyph.curves(),
            bounds,
            ascender,
            x,
            y,
            Batch::<f32>::splat(0.0),
        );
        eprintln!("Alpha at (0, 0) via eval_curves_cell: {}", alpha.first());

        let cell_glyph = CellGlyph::new(glyph.clone(), ascender);
        let alpha2 = cell_glyph.eval(x, y, z, w).first();
        eprintln!("Alpha at (0, 0) via CellGlyph.eval: {}", alpha2);

        // (2, 0) -> 2.5, 0.5
        let x2 = Batch::<f32>::splat(2.5);
        let alpha3 = cell_glyph.eval(x2, y, z, w).first();
        eprintln!("Alpha at (2, 0) via CellGlyph.eval: {}", alpha3);
    }

    #[test]
    fn direct_bake_period() {
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

        eprintln!("Point-by-point eval '.' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                // Manually bridge u32 -> f32
                let x_f = NativeBackend::u32_to_f32(Batch::<u32>::splat(x)) + Batch::<f32>::splat(0.5);
                let y_f = NativeBackend::u32_to_f32(Batch::<u32>::splat(y)) + Batch::<f32>::splat(0.5);
                let z_f = Batch::<f32>::splat(0.0);
                let w_f = Batch::<f32>::splat(0.0);

                let alpha = pixel_alpha(cell_glyph.eval(x_f, y_f, z_f, w_f).first());
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

        // Must use Rasterize
        let rasterized = Rasterize(cell_glyph);
        let baked: Baked<u32> = Baked::new(&rasterized, cell_width, cell_height);

        eprintln!("\nBaked '.' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                let alpha = pixel_alpha(baked.data()[(y * cell_width + x) as usize]);
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

        for y in 0..8u32 {
            for x in 0..cell_width {
                let alpha = pixel_alpha(baked.data()[(y * cell_width + x) as usize]);
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
        let font_bytes = include_bytes!("../assets/NotoSansMono-Regular.ttf");
        let font = Font::from_bytes(font_bytes).expect("Failed to load font");

        let cell_width = 10u32;
        let cell_height = 16u32;

        // This test uses the factory which we fixed, so it should work as is
        let glyph_fn = glyphs(font, cell_width, cell_height);
        let baked_lazy = glyph_fn('.');
        let baked: &Baked<u32> = baked_lazy.get();

        eprintln!("Baked '.' ({}x{}):", cell_width, cell_height);
        for y in 0..cell_height {
            let mut row = String::new();
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                // Use Surface::eval to support baked's blanket impl
                let alpha = pixel_alpha(Surface::eval(baked, x_batch, y_batch).first());
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

        let top_half_end = cell_height / 2;
        for y in 0..top_half_end {
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = pixel_alpha(Surface::eval(baked, x_batch, y_batch).first());
                assert_eq!(
                    alpha, 0,
                    "Top half pixel ({}, {}) should be transparent, got {}",
                    x, y, alpha
                );
            }
        }

        let mut found_opaque = false;
        for y in top_half_end..cell_height {
            for x in 0..cell_width {
                let x_batch = Batch::<u32>::splat(x);
                let y_batch = Batch::<u32>::splat(y);
                let alpha = pixel_alpha(Surface::eval(baked, x_batch, y_batch).first());
                if alpha > 200 {
                    found_opaque = true;
                    break;
                }
            }
        }
        assert!(found_opaque, "Bottom half should contain the period dot");
    }
}
