use super::curves::{Curve, Segment};
use super::glyph::{eval_curves, CurveSurface, GlyphBounds};
use pixelflow_core::{Field, Manifold};
use std::sync::Arc;

// Lazy and glyphs factory removed as they relied on legacy Rasterize/Baked infrastructure.
// TODO: Re-implement caching mechanism using new Materialize pipeline if needed.

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

impl<S: CurveSurface> Manifold for Bold<S> {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        eval_curves(
            self.curves(),
            self.bounds(),
            x,
            y,
            Field::from(self.amount),
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

impl<S: CurveSurface + Manifold<Output = Field>> Manifold for Hint<S> {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.source.eval_raw(x, y, z, w)
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

impl<S: CurveSurface> Manifold for Slant<S> {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        eval_curves(self.curves(), self.bounds(), x, y, Field::from(0.0))
    }
}

fn slant_point(p: [f32; 2], factor: f32) -> [f32; 2] {
    [p[0] + p[1] * factor, p[1]]
}

fn scale_point(p: [f32; 2], factor: f32) -> [f32; 2] {
    [p[0] * factor, p[1] * factor]
}

impl<S: CurveSurface> Slant<S> {
    pub fn new(source: S, factor: f32) -> Self {
        let curves = source
            .curves()
            .iter()
            .map(|seg| match *seg {
                Segment::Line(Curve([p0, p1])) => {
                    Segment::Line(Curve([slant_point(p0, factor), slant_point(p1, factor)]))
                }
                Segment::Quad(Curve([p0, p1, p2])) => Segment::Quad(Curve([
                    slant_point(p0, factor),
                    slant_point(p1, factor),
                    slant_point(p2, factor),
                ])),
                Segment::Cubic(Curve([p0, p1, p2, p3])) => Segment::Cubic(Curve([
                    slant_point(p0, factor),
                    slant_point(p1, factor),
                    slant_point(p2, factor),
                    slant_point(p3, factor),
                ])),
            })
            .collect::<Vec<_>>();

        Self {
            source,
            _factor: factor,
            curves: Arc::from(curves),
        }
    }
}

pub struct CurveScale<S> {
    source: S,
    factor: f32,
    curves: Arc<[Segment]>,
}

impl<S: CurveSurface> CurveSurface for CurveScale<S> {
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

impl<S: CurveSurface> Manifold for CurveScale<S> {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        eval_curves(self.curves(), self.bounds(), x, y, Field::from(0.0))
    }
}

impl<S: CurveSurface> CurveScale<S> {
    pub fn uniform(source: S, factor: f64) -> Self {
        Self::new(source, factor as f32)
    }

    pub fn new(source: S, factor: f32) -> Self {
        let curves = source
            .curves()
            .iter()
            .map(|seg| match *seg {
                Segment::Line(Curve([p0, p1])) => {
                    Segment::Line(Curve([scale_point(p0, factor), scale_point(p1, factor)]))
                }
                Segment::Quad(Curve([p0, p1, p2])) => Segment::Quad(Curve([
                    scale_point(p0, factor),
                    scale_point(p1, factor),
                    scale_point(p2, factor),
                ])),
                Segment::Cubic(Curve([p0, p1, p2, p3])) => Segment::Cubic(Curve([
                    scale_point(p0, factor),
                    scale_point(p1, factor),
                    scale_point(p2, factor),
                    scale_point(p3, factor),
                ])),
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
    fn scale_curve(self, factor: f32) -> CurveScale<Self> {
        CurveScale::new(self, factor)
    }
}

impl<S: CurveSurface> CurveSurfaceExt for S {}
