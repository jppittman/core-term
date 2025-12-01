use crate::curves::{Segment, Line, Point, Quadratic};
use crate::glyph::{CurveSurface, Glyph, GlyphBounds, eval_curves};
use crate::font::Font;
use pixelflow_core::batch::Batch;
use pixelflow_core::backend::SimdBatch;
use pixelflow_core::pipe::Surface;
use pixelflow_core::ops::{Baked, Max};
use std::sync::{Arc, Mutex, OnceLock};

// ============================================================================
// Lazy
// ============================================================================

/// A surface that lazily evaluates and caches its result.
///
/// Clones share the same underlying cache.
pub struct Lazy<'a, S> {
    inner: Arc<LazyInner<'a, S>>,
}

struct LazyInner<'a, S> {
    cache: OnceLock<S>,
    factory: Mutex<Option<Box<dyn FnOnce() -> S + Send + Sync + 'a>>>,
}

impl<'a, S> Clone for Lazy<'a, S> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
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
pub fn glyphs<'a>(font: Font<'a>, w: u32, h: u32) -> impl Fn(char) -> Lazy<'a, Baked<u8>> {
    use std::collections::HashMap;
    use std::sync::RwLock;

    let ascii: Vec<Lazy<'a, Baked<u8>>> = (0..128).map(|i| {
        let c = i as u8 as char;
        let font = font.clone();
        Lazy::new(move || {
            match font.glyph(c, h as f32) {
                Some(g) => Baked::new(&g, w, h),
                None => {
                    struct Empty;
                    impl Surface<u8> for Empty {
                        fn eval(&self, _: Batch<u32>, _: Batch<u32>) -> Batch<u8> {
                            Batch::<u8>::splat(0)
                        }
                    }
                    Baked::new(&Empty, w, h)
                }
            }
        })
    }).collect();

    let other_cache: Arc<RwLock<HashMap<char, Lazy<'a, Baked<u8>>>>> = Arc::new(RwLock::new(HashMap::new()));

    move |c| {
        if (c as u32) < 128 {
            ascii[c as usize].clone()
        } else {
            // Check other cache
            if let Ok(read) = other_cache.read() {
                if let Some(lazy) = read.get(&c) {
                    return lazy.clone();
                }
            }
            // Write lock
            let mut write = other_cache.write().unwrap();
            if let Some(lazy) = write.get(&c) {
                return lazy.clone();
            }
            // Insert
            let font = font.clone();
            let lazy = Lazy::new(move || {
                match font.glyph(c, h as f32) {
                    Some(g) => Baked::new(&g, w, h),
                    None => {
                        struct Empty;
                        impl Surface<u8> for Empty {
                            fn eval(&self, _: Batch<u32>, _: Batch<u32>) -> Batch<u8> {
                                Batch::<u8>::splat(0)
                            }
                        }
                        Baked::new(&Empty, w, h)
                    }
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
        eval_curves(self.curves(), self.bounds(), x, y, Batch::<f32>::splat(self.amount))
    }
}

impl<S: CurveSurface> Bold<S> {
    pub fn new(source: S, amount: f32) -> Self {
        Self { source, amount }
    }
}

// --- Hint (Stub) ---

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

pub struct Slant<S> {
    source: S,
    factor: f32,
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
        let curves = source.curves().iter().map(|seg| {
            match seg {
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
            }
        }).collect::<Vec<_>>();

        Self {
            source,
            factor,
            curves: Arc::from(curves),
        }
    }
}

// --- Scale ---

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
        let curves = source.curves().iter().map(|seg| {
            match seg {
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
            }
        }).collect::<Vec<_>>();

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
