//! # Loop-Blinn Font Rendering
//!
//! GPU-style quadratic Bézier rendering using barycentric coordinates.
//! Each glyph is triangulated, with curved edges evaluated via u² - v ≤ 0.
//!
//! ## Architecture
//!
//! ```text
//! TTF Outline → Triangulation → Loop-Blinn Triangles → Manifold Composition
//!     │              │                  │                      │
//!   Parse      Ear-clipping      UV matrices         Select { bounds, curve }
//! ```
//!
//! ## The Loop-Blinn Insight
//!
//! A quadratic Bézier P(t) = (1-t)²P₀ + 2t(1-t)P₁ + t²P₂ can be rendered as:
//!
//! 1. Map the control triangle to texture space:
//!    - P₀ → (u=0, v=0)
//!    - P₁ → (u=0.5, v=0)
//!    - P₂ → (u=1, v=1)
//!
//! 2. The curve is the zero set: u² - v = 0
//!    - Inside (filled): u² - v < 0
//!    - Outside: u² - v > 0
//!
//! ## Manifold Composition
//!
//! Each triangle becomes a `LoopBlinnTriangle`:
//! ```text
//! Select {
//!     cond: barycentric_bounds,  // Is point inside triangle?
//!     if_true: curve_test,       // Solid 1.0 or u² - v ≤ 0
//!     if_false: 0.0,            // Outside triangle
//! }
//! ```
//!
//! Glyph = Sum of triangles. Types ARE the shader graph.

use pixelflow_core::jet::Jet2;
use pixelflow_core::{Add, At, Field, Manifold, ManifoldExt, Mul, MulAdd, Sub, X, Y, Z, W};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Sum Combinator - Glyph Composition via Max-Blend
// ═══════════════════════════════════════════════════════════════════════════

/// A monoidal sum of manifolds (max-blend for coverage).
///
/// Used to compose glyphs: `Sum<Translate<Glyph>>` for text,
/// `Sum<Affine<Glyph>>` for compound glyphs.
#[derive(Clone, Debug)]
pub struct Sum<T>(pub Arc<[T]>);

impl<T: Manifold<Field, Output = Field>> Manifold<Field> for Sum<T> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Max-blend: any glyph covering this pixel fills it
        // Use Field::max directly to avoid expression tree overhead
        let mut acc = Field::from(0.0);
        for term in self.0.iter() {
            let val = term.eval_raw(x, y, z, w);
            acc = Field::max(acc, val);
        }
        acc
    }
}

impl<T: Manifold<Jet2, Output = Jet2>> Manifold<Jet2> for Sum<T> {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        let zero = Jet2::constant(Field::from(0.0));
        self.0.iter().fold(zero, |acc, term| {
            acc.max(term.eval_raw(x, y, z, w))
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear Expression Tree - a*X + b*Y + c
// ═══════════════════════════════════════════════════════════════════════════

/// Linear manifold type: `a*X + b*Y + c`.
/// The type IS the computation graph - coefficients stored in AST nodes.
/// Expression `X * a + Y * b + c` produces MulAdd due to fusion optimization.
pub type Linear = Add<MulAdd<X, f32, Mul<Y, f32>>, f32>;

/// Construct a linear manifold expression tree.
#[inline]
pub fn linear(a: f32, b: f32, c: f32) -> Linear {
    X * a + Y * b + c
}


/// Curve manifold type: `u² - v` where u, v are Linear.
pub type Curve = Sub<Mul<Linear, Linear>, Linear>;

// ═══════════════════════════════════════════════════════════════════════════
// Loop-Blinn Triangle Primitive
// ═══════════════════════════════════════════════════════════════════════════

/// A single triangle in Loop-Blinn representation.
///
/// Stores expression trees for evaluation + scalar coefficients for bounds/Jet2.
/// Types ARE the shader graph - monomorphized into fused SIMD kernels.
#[derive(Clone, Copy, Debug)]
pub struct LoopBlinnTriangle {
    /// Edge functions as manifolds: e >= 0 means inside
    pub edges: [Linear; 3],
    /// Edge coefficients (a, b, c) for bounds and Jet2 evaluation
    pub edge_coeffs: [[f32; 3]; 3],

    /// The curve function f = u² - v
    pub curve: Curve,

    /// Precomputed gradient ∂f/∂x, ∂f/∂y as expression trees
    pub grad_x: Linear,
    pub grad_y: Linear,

    /// U coefficients for Taylor error bound
    pub ua: f32,
    pub ub: f32,
}

impl LoopBlinnTriangle {
    /// Create a solid (non-curved) triangle.
    /// UV is set so curve test always passes: u=0, v=1 → u²-v = -1 < 0
    #[inline]
    pub fn solid(vertices: [[f32; 2]; 3]) -> Self {
        let (edges, edge_coeffs) = compute_edges(vertices);
        let u = linear(0.0, 0.0, 0.0);
        let v = linear(0.0, 0.0, 1.0);
        let curve = u * u - v;
        // Gradient is 0 for solid triangles
        let grad_x = linear(0.0, 0.0, 0.0);
        let grad_y = linear(0.0, 0.0, 0.0);

        Self { edges, edge_coeffs, curve, grad_x, grad_y, ua: 0.0, ub: 0.0 }
    }

    /// Create a curved triangle with quadratic Bézier edge.
    /// Winding is baked into the UV matrix (v is negated for CW).
    #[inline]
    pub fn curved(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], winding: f32) -> Self {
        let (edges, edge_coeffs) = compute_edges([p0, p1, p2]);
        let [ua, ub, uc, va, vb, vc] = compute_uv_coeffs(p0, p1, p2, winding);

        // Build all manifolds from coefficients
        let u = linear(ua, ub, uc);
        let v = linear(va, vb, vc);
        let curve = u * u - v;

        // Gradient: ∂f/∂x = 2*u*ua - va, ∂f/∂y = 2*u*ub - vb
        let grad_x = linear(2.0 * ua * ua, 2.0 * ua * ub, 2.0 * ua * uc - va);
        let grad_y = linear(2.0 * ub * ua, 2.0 * ub * ub, 2.0 * ub * uc - vb);

        Self { edges, edge_coeffs, curve, grad_x, grad_y, ua, ub }
    }

    /// Compute curve value with precomputed gradient as Jet2.
    /// All expressions are stored - just contramap and evaluate.
    #[inline(always)]
    pub fn curve_jet(&self, x: Field, y: Field) -> Jet2 {
        let fzero = Field::from(0.0);
        let val = self.curve.at(x, y, fzero, fzero).constant();
        let dx = self.grad_x.at(x, y, fzero, fzero).constant();
        let dy = self.grad_y.at(x, y, fzero, fzero).constant();
        Jet2::new(val, dx, dy)
    }

    /// Check if triangle could contribute to a tile using Taylor bounds.
    #[inline(always)]
    pub fn might_affect_tile(&self, cx: Field, cy: Field, hx: f32, hy: f32) -> bool {
        let fzero = Field::from(0.0);

        // First check edge bounds - quick reject if tile is completely outside
        for i in 0..3 {
            let edge_at_center = self.edges[i].at(cx, cy, fzero, fzero).constant();
            let [a, b, _c] = self.edge_coeffs[i];
            let max_dev = Field::from(a.abs() * hx + b.abs() * hy);
            // Evaluate Field - Field to get concrete Field
            let min_edge: Field = (edge_at_center - max_dev).constant();
            if !Field::ge(min_edge, fzero).any() {
                return false;
            }
        }

        // Tile might be inside triangle - check curve bounds via analytical Jet2
        let jet = self.curve_jet(cx, cy);

        // Taylor bound: f ± |∂f/∂x|·hx ± |∂f/∂y|·hy ± R₂
        // R₂ = (ua·hx + ub·hy)² (exact for quadratic)
        let linear_dev = jet.dx.abs() * hx + jet.dy.abs() * hy;
        let u_dev = self.ua.abs() * hx + self.ub.abs() * hy;
        let total_dev = linear_dev + u_dev * u_dev;

        // Curve might contribute if min value < 0.5
        let min_f: Field = (jet.val - total_dev).constant();
        Field::lt(min_f, Field::from(0.5)).any()
    }
}

/// Compute edge functions as Linear manifolds + coefficients for bounds/Jet2.
#[inline]
fn compute_edges(v: [[f32; 2]; 3]) -> ([Linear; 3], [[f32; 3]; 3]) {
    let mut edges = [linear(0.0, 0.0, 0.0); 3];
    let mut coeffs = [[0.0f32; 3]; 3];

    for i in 0..3 {
        let [x0, y0] = v[i];
        let [x1, y1] = v[(i + 1) % 3];
        let a = y1 - y0;
        let b = x0 - x1;
        let c = -(a * x0 + b * y0);

        let [x2, y2] = v[(i + 2) % 3];
        let sign = if a * x2 + b * y2 + c >= 0.0 { 1.0 } else { -1.0 };
        let (a, b, c) = (a * sign, b * sign, c * sign);

        edges[i] = linear(a, b, c);
        coeffs[i] = [a, b, c];
    }
    (edges, coeffs)
}

/// Compute UV coefficients for Loop-Blinn curve test.
/// Maps: P0 → (0, 0), P1 → (0.5, 0), P2 → (1, 1)
/// Winding is baked in by scaling v.
#[inline]
fn compute_uv_coeffs(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], winding: f32) -> [f32; 6] {
    let [x0, y0] = p0;
    let [x1, y1] = p1;
    let [x2, y2] = p2;

    let det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1);
    if det.abs() < 1e-10 {
        // Degenerate: return "always inside" like solid
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    }
    let inv_det = 1.0 / det;

    // Inverse of vertex matrix columns
    let m01 = (y2 - y0) * inv_det;
    let m02 = (y0 - y1) * inv_det;
    let m11 = (x0 - x2) * inv_det;
    let m12 = (x1 - x0) * inv_det;
    let m21 = (x2 * y0 - x0 * y2) * inv_det;
    let m22 = (x0 * y1 - x1 * y0) * inv_det;

    // u = 0.5*col1 + 1.0*col2, v = 1.0*col2
    let ua = 0.5 * m01 + m02;
    let ub = 0.5 * m11 + m12;
    let uc = 0.5 * m21 + m22;
    // Bake winding into v: for CW (winding=-1), negate v
    // This makes curve_test = u² - v work correctly for both orientations
    let va = m02 * winding;
    let vb = m12 * winding;
    let vc = m22 * winding;

    [ua, ub, uc, va, vb, vc]
}

// ═══════════════════════════════════════════════════════════════════════════
// Manifold Implementation - Pure SIMD with Analytical Gradients
// ═══════════════════════════════════════════════════════════════════════════

impl Manifold<Field> for LoopBlinnTriangle {
    type Output = Field;

    /// Evaluate triangle coverage with analytical AA.
    ///
    /// Uses stored expression trees - no rebuilding.
    /// Contramap binds coordinates, Select handles early exit.
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        let fzero = Field::from(0.0);

        // Edge mask from stored Linear manifolds
        let [e0, e1, e2] = self.edges;
        let edge_mask = e0.ge(0.0) & e1.ge(0.0) & e2.ge(0.0);

        // Gradient magnitude from stored Linear manifolds
        let grad_mag = (self.grad_x * self.grad_x + self.grad_y * self.grad_y)
            .sqrt()
            .max(1e-6);

        // Coverage: 0.5 - f/|∇f|, clamped to [0,1]
        let signed_dist = self.curve / grad_mag;
        let coverage = (signed_dist * -1.0 + 0.5).max(0.0).min(1.0);

        // Select with edge mask (lazy, convergent optimization)
        let result = edge_mask.select(coverage, 0.0);

        // Contramap to concrete coordinates and evaluate
        result.at(x, y, fzero, fzero).constant()
    }
}

impl Manifold<Jet2> for LoopBlinnTriangle {
    type Output = Jet2;

    /// Jet2 implementation for composition with other AD-aware manifolds.
    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, _z: Jet2, _w: Jet2) -> Jet2 {
        let zero = Jet2::constant(Field::from(0.0));
        let half = Field::from(0.5);

        // Edge functions using stored coefficients
        let eval_edge = |coeffs: [f32; 3], x: Jet2, y: Jet2| -> Jet2 {
            let [a, b, c] = coeffs;
            x * Jet2::constant(Field::from(a))
                + y * Jet2::constant(Field::from(b))
                + Jet2::constant(Field::from(c))
        };

        let e0 = eval_edge(self.edge_coeffs[0], x, y);
        let e1 = eval_edge(self.edge_coeffs[1], x, y);
        let e2 = eval_edge(self.edge_coeffs[2], x, y);

        let inside_edges = Jet2::ge(e0, zero) & Jet2::ge(e1, zero) & Jet2::ge(e2, zero);
        if !inside_edges.val.any() {
            return zero;
        }

        // Evaluate curve at Jet2 coordinates (for composition)
        let fzero = Field::from(0.0);
        let curve_val = self.curve.at(x.val, y.val, fzero, fzero).constant();
        let grad_x_val = self.grad_x.at(x.val, y.val, fzero, fzero).constant();
        let grad_y_val = self.grad_y.at(x.val, y.val, fzero, fzero).constant();

        let grad_mag = (grad_x_val * grad_x_val + grad_y_val * grad_y_val)
            .sqrt()
            .max(Field::from(1e-6));
        let signed_dist = curve_val / grad_mag;
        let coverage: Field = (half - signed_dist).max(Field::from(0.0)).min(Field::from(1.0)).constant();

        (inside_edges & Jet2::constant(coverage)) | (!inside_edges & zero)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Glyph Composition
// ═══════════════════════════════════════════════════════════════════════════

/// A glyph as a sum of Loop-Blinn triangles.
#[derive(Clone, Debug)]
pub struct TriangulatedGlyph {
    pub triangles: Arc<[LoopBlinnTriangle]>,
}

impl Manifold<Field> for TriangulatedGlyph {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let mut acc = Field::from(0.0);

        // Max blend: any triangle covering this pixel fills it
        // Use Field::max directly to avoid expression tree overhead
        for tri in self.triangles.iter() {
            let val = tri.eval_raw(x, y, z, w);
            acc = Field::max(acc, val);
        }
        acc
    }
}

impl Manifold<Jet2> for TriangulatedGlyph {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        let zero = Jet2::constant(Field::from(0.0));

        self.triangles
            .iter()
            .map(|tri| tri.eval_raw(x, y, z, w))
            .fold(zero, |acc, val| acc.max(val))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Affine Transform (unchanged from original)
// ═══════════════════════════════════════════════════════════════════════════

/// Affine transform combinator.
#[derive(Clone, Debug)]
pub struct Affine<M> {
    pub inner: M,
    inv: [f32; 6],
}

pub fn affine<M>(inner: M, [a, b, c, d, tx, ty]: [f32; 6]) -> Affine<M> {
    let det = a * d - b * c;
    let inv_det = if det.abs() < 1e-6 { 0.0 } else { 1.0 / det };
    Affine {
        inner,
        inv: [d * inv_det, -b * inv_det, -c * inv_det, a * inv_det, tx, ty],
    }
}

impl<M: Manifold<Field>> Manifold<Field> for Affine<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
        let [a, b, c, d, tx, ty] = self.inv;
        let x2 = (X - tx) * a + (Y - ty) * b;
        let y2 = (X - tx) * c + (Y - ty) * d;
        At {
            inner: &self.inner,
            x: x2,
            y: y2,
            z: Z,
            w: W,
        }
        .eval_raw(x, y, z, w)
    }
}

impl<M: Manifold<Jet2>> Manifold<Jet2> for Affine<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Self::Output {
        let [a, b, c, d, tx, ty] = self.inv;
        let x2 = (X - tx) * a + (Y - ty) * b;
        let y2 = (X - tx) * c + (Y - ty) * d;
        At {
            inner: &self.inner,
            x: x2,
            y: y2,
            z: Z,
            w: W,
        }
        .eval_raw(x, y, z, w)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Glyph Enum
// ═══════════════════════════════════════════════════════════════════════════

pub type SimpleGlyph = Affine<TriangulatedGlyph>;

/// A glyph: empty, triangulated outline, or compound.
#[derive(Clone)]
pub enum Glyph {
    Empty,
    Simple(SimpleGlyph),
    Compound(Sum<Affine<Glyph>>),
}

impl core::fmt::Debug for Glyph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "Glyph::Empty"),
            Self::Simple(_) => write!(f, "Glyph::Simple(...)"),
            Self::Compound(_) => write!(f, "Glyph::Compound(...)"),
        }
    }
}

impl Manifold<Field> for Glyph {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        match self {
            Self::Empty => Field::from(0.0),
            Self::Simple(g) => g.eval_raw(x, y, z, w),
            Self::Compound(g) => g.eval_raw(x, y, z, w),
        }
    }
}

impl Manifold<Jet2> for Glyph {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        match self {
            Self::Empty => Jet2::constant(Field::from(0.0)),
            Self::Simple(g) => g.eval_raw(x, y, z, w),
            Self::Compound(g) => g.eval_raw(x, y, z, w),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TTF Reader (unchanged)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct R<'a>(&'a [u8], usize);

impl<'a> R<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.0.get(self.1)?;
        self.1 += 1;
        Some(v)
    }
    fn i8(&mut self) -> Option<i8> {
        self.u8().map(|v| v as i8)
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.0.get(self.1..self.1 + 2)?;
        self.1 += 2;
        Some(u16::from_be_bytes(s.try_into().ok()?))
    }
    fn i16(&mut self) -> Option<i16> {
        self.u16().map(|v| v as i16)
    }
    fn u32(&mut self) -> Option<u32> {
        let s = self.0.get(self.1..self.1 + 4)?;
        self.1 += 4;
        Some(u32::from_be_bytes(s.try_into().ok()?))
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.0.get(self.1..self.1 + n)?;
        self.1 += n;
        Some(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tables
// ═══════════════════════════════════════════════════════════════════════════

enum Loca<'a> {
    Short(&'a [u8]),
    Long(&'a [u8]),
}

impl Loca<'_> {
    fn get(&self, i: usize) -> Option<usize> {
        match self {
            Self::Short(d) => Some(R(*d, i * 2).u16()? as usize * 2),
            Self::Long(d) => Some(R(*d, i * 4).u32()? as usize),
        }
    }
}

enum Cmap<'a> {
    Fmt4(&'a [u8]),
    Fmt12(&'a [u8]),
}

impl Cmap<'_> {
    fn lookup(&self, c: u32) -> Option<u16> {
        match self {
            Self::Fmt4(d) if c <= 0xFFFF => {
                let n = R(*d, 6).u16()? as usize / 2;
                (0..n).find_map(|i| {
                    let end = R(*d, 14 + i * 2).u16()?;
                    if c as u16 > end {
                        return None;
                    }
                    let start = R(*d, 16 + n * 2 + i * 2).u16()?;
                    if (c as u16) < start {
                        return Some(0);
                    }
                    let delta = R(*d, 16 + n * 4 + i * 2).i16()?;
                    let range = R(*d, 16 + n * 6 + i * 2).u16()?;
                    Some(if range == 0 {
                        (c as i16).wrapping_add(delta) as u16
                    } else {
                        let off =
                            16 + n * 6 + i * 2 + range as usize + (c as u16 - start) as usize * 2;
                        let g = R(*d, off).u16()?;
                        if g == 0 {
                            0
                        } else {
                            (g as i16).wrapping_add(delta) as u16
                        }
                    })
                })
            }
            Self::Fmt12(d) => (0..R(*d, 12).u32()? as usize).find_map(|i| {
                let (s, e, g) = (
                    R(*d, 16 + i * 12).u32()?,
                    R(*d, 20 + i * 12).u32()?,
                    R(*d, 24 + i * 12).u32()?,
                );
                (c >= s && c <= e).then(|| (g + c - s) as u16)
            }),
            _ => None,
        }
    }
}

enum Kern<'a> {
    Fmt0 { data: &'a [u8], n_pairs: usize },
    None,
}

impl<'a> Kern<'a> {
    fn parse(data: &'a [u8]) -> Self {
        let Some(n_tables) = R(data, 2).u16() else {
            return Self::None;
        };
        let mut off = 4;

        for _ in 0..n_tables {
            let Some(length) = R(data, off + 2).u16() else {
                return Self::None;
            };
            let Some(coverage) = R(data, off + 4).u16() else {
                return Self::None;
            };

            let format = coverage >> 8;
            let horizontal = coverage & 1;

            if format == 0 && horizontal == 1 {
                let Some(n_pairs) = R(data, off + 6).u16() else {
                    return Self::None;
                };
                return Self::Fmt0 {
                    data: &data[off + 14..],
                    n_pairs: n_pairs as usize,
                };
            }
            off += length as usize;
        }
        Self::None
    }

    fn get(&self, left: u16, right: u16) -> i16 {
        match self {
            Self::Fmt0 { data, n_pairs } => {
                let key = ((left as u32) << 16) | (right as u32);
                let (mut lo, mut hi) = (0, *n_pairs);

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let pair = ((R(*data, mid * 6).u16().unwrap_or(0) as u32) << 16)
                        | (R(*data, mid * 6 + 2).u16().unwrap_or(0) as u32);

                    match pair.cmp(&key) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => {
                            return R(*data, mid * 6 + 4).i16().unwrap_or(0)
                        }
                    }
                }
                0
            }
            Self::None => 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Font
// ═══════════════════════════════════════════════════════════════════════════

pub struct Font<'a> {
    data: &'a [u8],
    glyf: usize,
    loca: Loca<'a>,
    cmap: Cmap<'a>,
    kern: Kern<'a>,
    hmtx: usize,
    num_hm: usize,
    pub units_per_em: u16,
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
}

impl<'a> Font<'a> {
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let num_tables = R(data, 4).u16()? as usize;
        let mut t = std::collections::HashMap::new();

        for i in 0..num_tables {
            let rec = 12 + i * 16;
            let tag = [data[rec], data[rec + 1], data[rec + 2], data[rec + 3]];
            let offset = R(data, rec + 8).u32()? as usize;
            t.insert(tag, offset);
        }

        let head = *t.get(b"head")?;
        let loca = *t.get(b"loca")?;
        let hhea = *t.get(b"hhea")?;

        Some(Self {
            data,
            glyf: *t.get(b"glyf")?,
            loca: if R(data, head + 50).i16()? != 0 {
                Loca::Long(&data[loca..])
            } else {
                Loca::Short(&data[loca..])
            },
            cmap: Self::find_cmap(&data[*t.get(b"cmap")?..])?,
            kern: t
                .get(b"kern")
                .map(|&off| Kern::parse(&data[off..]))
                .unwrap_or(Kern::None),
            hmtx: *t.get(b"hmtx")?,
            num_hm: R(data, hhea + 34).u16()? as usize,
            units_per_em: R(data, head + 18).u16()?,
            ascent: R(data, hhea + 4).i16()?,
            descent: R(data, hhea + 6).i16()?,
            line_gap: R(data, hhea + 8).i16()?,
        })
    }

    fn find_cmap(d: &'a [u8]) -> Option<Cmap<'a>> {
        (0..R(d, 2).u16()? as usize)
            .filter_map(|i| {
                let (p, e, o) = (
                    R(d, 4 + i * 8).u16()?,
                    R(d, 6 + i * 8).u16()?,
                    R(d, 8 + i * 8).u32()? as usize,
                );
                let f = R(d, o).u16()?;
                match (p, e, f) {
                    (3, 10, 12) | (0, 4, 12) => Some((2, o, f)),
                    (3, 1, 4) | (0, 3, 4) => Some((1, o, f)),
                    _ => None,
                }
            })
            .max_by_key(|x| x.0)
            .and_then(|(_, o, f)| match f {
                4 => Some(Cmap::Fmt4(&d[o..])),
                12 => Some(Cmap::Fmt12(&d[o..])),
                _ => None,
            })
    }

    #[inline]
    pub fn cmap_lookup(&self, ch: char) -> Option<u16> {
        self.cmap.lookup(ch as u32)
    }

    pub fn glyph(&self, ch: char) -> Option<Glyph> {
        self.compile(self.cmap.lookup(ch as u32)?)
    }

    #[inline]
    pub fn glyph_by_id(&self, id: u16) -> Option<Glyph> {
        self.compile(id)
    }

    pub fn glyph_scaled(&self, ch: char, size: f32) -> Option<Glyph> {
        let id = self.cmap.lookup(ch as u32)?;
        self.glyph_scaled_by_id(id, size)
    }

    pub fn glyph_scaled_by_id(&self, id: u16, size: f32) -> Option<Glyph> {
        let g = self.glyph_by_id(id)?;
        let scale = size / self.units_per_em as f32;
        let y_offset = self.ascent as f32 * scale;
        Some(Glyph::Compound(Sum(
            [affine(g, [scale, 0.0, 0.0, -scale, 0.0, y_offset])].into(),
        )))
    }

    pub fn advance(&self, ch: char) -> Option<f32> {
        let id = self.cmap.lookup(ch as u32)?;
        self.advance_by_id(id)
    }

    #[inline]
    pub fn advance_by_id(&self, id: u16) -> Option<f32> {
        let i = (id as usize).min(self.num_hm.saturating_sub(1));
        Some(R(self.data, self.hmtx + i * 4).u16()? as f32)
    }

    pub fn advance_scaled(&self, ch: char, size: f32) -> Option<f32> {
        Some(self.advance(ch)? * size / self.units_per_em as f32)
    }

    pub fn advance_scaled_by_id(&self, id: u16, size: f32) -> Option<f32> {
        Some(self.advance_by_id(id)? * size / self.units_per_em as f32)
    }

    pub fn kern(&self, left: char, right: char) -> f32 {
        let left_id = self.cmap.lookup(left as u32).unwrap_or(0);
        let right_id = self.cmap.lookup(right as u32).unwrap_or(0);
        self.kern_by_ids(left_id, right_id)
    }

    #[inline]
    pub fn kern_by_ids(&self, left_id: u16, right_id: u16) -> f32 {
        self.kern.get(left_id, right_id) as f32
    }

    pub fn kern_scaled(&self, left: char, right: char, size: f32) -> f32 {
        self.kern(left, right) * size / self.units_per_em as f32
    }

    fn compile(&self, id: u16) -> Option<Glyph> {
        let (a, b) = (self.loca.get(id as usize)?, self.loca.get(id as usize + 1)?);
        if a == b {
            return Some(Glyph::Empty);
        }
        let mut r = R(self.data, self.glyf + a);
        let n = r.i16()?;
        let x_min = r.i16()?;
        let y_min = r.i16()?;
        let x_max = r.i16()?;
        let y_max = r.i16()?;

        let width = (x_max - x_min) as f32;
        let height = (y_max - y_min) as f32;
        let max_dim = width.max(height).max(1.0);

        let norm_scale = 1.0 / max_dim;
        let norm_tx = -(x_min as f32) * norm_scale;
        let norm_ty = -(y_min as f32) * norm_scale;

        // Restore transform: scale back to font units, translate to original position
        // Do NOT flip Y here - glyph_scaled_by_id handles the Y flip
        let restore = [max_dim, 0.0, 0.0, max_dim, x_min as f32, y_min as f32];

        if n >= 0 {
            let triangles = self.triangulate(&mut r, n as usize, norm_scale, norm_tx, norm_ty)?;
            let glyph = TriangulatedGlyph { triangles: triangles.into() };
            Some(Glyph::Simple(affine(glyph, restore)))
        } else {
            self.compound(&mut r)
        }
    }

    fn triangulate(
        &self,
        r: &mut R,
        n: usize,
        scale: f32,
        tx: f32,
        ty: f32,
    ) -> Option<Vec<LoopBlinnTriangle>> {
        if n == 0 {
            return Some(vec![]);
        }

        let ends: Vec<_> = (0..n)
            .map(|_| r.u16().map(|v| v as usize))
            .collect::<Option<_>>()?;
        let np = *ends.last()? + 1;
        let instr_len = r.u16()? as usize;
        r.skip(instr_len)?;

        let mut fl = Vec::with_capacity(np);
        while fl.len() < np {
            let f = r.u8()?;
            fl.push(f);
            if f & 8 != 0 {
                for _ in 0..r.u8()?.min((np - fl.len()) as u8) {
                    fl.push(f);
                }
            }
        }

        let dec = |r: &mut R, s: u8, m: u8| {
            fl.iter()
                .try_fold((0i16, vec![]), |(mut v, mut out), &f| {
                    v += match (f & s != 0, f & m != 0) {
                        (true, true) => r.u8()? as i16,
                        (true, false) => -(r.u8()? as i16),
                        (false, true) => 0,
                        (false, false) => r.i16()?,
                    };
                    out.push(v);
                    Some((v, out))
                })
                .map(|(_, v)| v)
        };

        let (xs, ys) = (dec(r, 2, 16)?, dec(r, 4, 32)?);

        let pts: Vec<_> = (0..np)
            .map(|i| {
                (
                    (xs[i] as f32) * scale + tx,
                    (ys[i] as f32) * scale + ty,
                    fl[i] & 1 != 0, // on-curve
                )
            })
            .collect();

        let mut triangles = Vec::new();
        let mut start = 0;

        for &e in ends.iter() {
            let contour = &pts[start..=e];
            start = e + 1;
            triangulate_contour(contour, &mut triangles);
        }

        Some(triangles)
    }

    fn compound(&self, r: &mut R) -> Option<Glyph> {
        let mut kids = vec![];
        loop {
            let fl = r.u16()?;
            let id = r.u16()?;
            let (dx, dy) = if fl & 2 != 0 {
                if fl & 1 != 0 {
                    (r.i16()?, r.i16()?)
                } else {
                    (r.i8()? as i16, r.i8()? as i16)
                }
            } else {
                r.skip(if fl & 1 != 0 { 4 } else { 2 })?;
                (0, 0)
            };
            let mut m = [1.0, 0.0, 0.0, 1.0, dx as f32, dy as f32];
            if fl & 0x08 != 0 {
                let s = r.i16()? as f32 / 16384.0;
                m[0] = s;
                m[3] = s;
            } else if fl & 0x40 != 0 {
                m[0] = r.i16()? as f32 / 16384.0;
                m[3] = r.i16()? as f32 / 16384.0;
            } else if fl & 0x80 != 0 {
                m[0] = r.i16()? as f32 / 16384.0;
                m[1] = r.i16()? as f32 / 16384.0;
                m[2] = r.i16()? as f32 / 16384.0;
                m[3] = r.i16()? as f32 / 16384.0;
            }
            if let Some(g) = self.compile(id) {
                kids.push(affine(g, m));
            }
            if fl & 0x20 == 0 {
                break;
            }
        }
        Some(Glyph::Compound(Sum(kids.into())))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Triangulation - Ear Clipping with Quadratic Curves
// ═══════════════════════════════════════════════════════════════════════════

/// Triangulate a single contour into Loop-Blinn triangles.
///
/// For each quadratic curve (on-off-on), creates a curved triangle.
/// For straight edges, creates solid triangles via ear clipping.
fn triangulate_contour(pts: &[(f32, f32, bool)], out: &mut Vec<LoopBlinnTriangle>) {
    if pts.len() < 3 {
        return;
    }

    // Expand implicit on-curve points between consecutive off-curve points
    let expanded: Vec<_> = pts
        .iter()
        .enumerate()
        .flat_map(|(i, &(x, y, on))| {
            let (nx, ny, non) = pts[(i + 1) % pts.len()];
            if !on && !non {
                vec![(x, y, on), ((x + nx) / 2.0, (y + ny) / 2.0, true)]
            } else {
                vec![(x, y, on)]
            }
        })
        .collect();

    if expanded.len() < 3 {
        return;
    }

    // Find starting on-curve point
    let start = expanded.iter().position(|p| p.2).unwrap_or(0);

    // Extract curved triangles and build polygon for ear clipping
    let mut polygon: Vec<[f32; 2]> = Vec::new();
    let mut i = 0;

    while i < expanded.len() {
        let curr = (start + i) % expanded.len();
        let next = (start + i + 1) % expanded.len();

        let (x0, y0, on0) = expanded[curr];
        let (x1, y1, on1) = expanded[next];

        if on0 {
            polygon.push([x0, y0]);

            if !on1 {
                // Quadratic curve: on → off → on
                let next2 = (start + i + 2) % expanded.len();
                let (x2, y2, _) = expanded[next2];

                // Determine winding from contour direction
                let cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
                let winding = if cross >= 0.0 { 1.0 } else { -1.0 };

                out.push(LoopBlinnTriangle::curved(
                    [x0, y0],
                    [x1, y1],
                    [x2, y2],
                    winding,
                ));

                i += 2;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    // Ear clipping for the remaining polygon (solid triangles)
    if polygon.len() >= 3 {
        ear_clip_polygon(&polygon, out);
    }
}

/// Simple ear clipping triangulation for convex/simple polygons.
fn ear_clip_polygon(polygon: &[[f32; 2]], out: &mut Vec<LoopBlinnTriangle>) {
    if polygon.len() < 3 {
        return;
    }

    let mut indices: Vec<usize> = (0..polygon.len()).collect();

    while indices.len() > 3 {
        let mut ear_found = false;

        for i in 0..indices.len() {
            let prev = indices[(i + indices.len() - 1) % indices.len()];
            let curr = indices[i];
            let next = indices[(i + 1) % indices.len()];

            let p0 = polygon[prev];
            let p1 = polygon[curr];
            let p2 = polygon[next];

            // Check if this is a convex vertex (ear candidate)
            let cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            if cross <= 0.0 {
                continue; // Reflex vertex, not an ear
            }

            // Check no other vertices inside this triangle
            let mut is_ear = true;
            for &j in &indices {
                if j == prev || j == curr || j == next {
                    continue;
                }
                if point_in_triangle(polygon[j], p0, p1, p2) {
                    is_ear = false;
                    break;
                }
            }

            if is_ear {
                out.push(LoopBlinnTriangle::solid([p0, p1, p2]));
                indices.remove(i);
                ear_found = true;
                break;
            }
        }

        if !ear_found {
            // Degenerate case - just fan triangulate remaining
            break;
        }
    }

    // Final triangle
    if indices.len() == 3 {
        out.push(LoopBlinnTriangle::solid([
            polygon[indices[0]],
            polygon[indices[1]],
            polygon[indices[2]],
        ]));
    }
}

/// Check if point p is inside triangle (p0, p1, p2) using barycentric coordinates.
fn point_in_triangle(p: [f32; 2], p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> bool {
    let d00 = (p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0]);
    let d01 = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]);
    let d02 = (p0[0] - p2[0]) * (p[1] - p2[1]) - (p0[1] - p2[1]) * (p[0] - p2[0]);

    (d00 >= 0.0 && d01 >= 0.0 && d02 >= 0.0) || (d00 <= 0.0 && d01 <= 0.0 && d02 <= 0.0)
}
