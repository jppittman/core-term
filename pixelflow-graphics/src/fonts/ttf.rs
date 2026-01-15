//! pixelflow-graphics/src/fonts/ttf.rs
//!
//! Pure Manifold TTF Parser.
//!
//! Glyphs are parsed into unit space [0,1]², bounded with Select for
//! short-circuit evaluation, then wrapped in Affine transforms.

use crate::shapes::{square, Bounded};
use pixelflow_core::jet::Jet2;
use pixelflow_core::{
    Abs, At, Differentiable, Field, Ge, Manifold, ManifoldExt, Mul, MulAdd, Select, Sub, W, X,
    Y, Z,
};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Type Aliases for Concrete Kernel Types
// ═══════════════════════════════════════════════════════════════════════════

/// The concrete type returned by line_winding_field (captured via TAIT).
pub type LineKernel = impl Manifold<Field, Output = Field> + Clone;

/// The concrete type returned by quadratic_winding (captured via TAIT).
pub type QuadKernel = impl Manifold<Field, Output = Field> + Clone;

// Defining uses for TAIT - these functions establish what the opaque types actually are
#[doc(hidden)]
pub fn __line_kernel_definer([[x0, y0], [x1, y1]]: [[f32; 2]; 2]) -> LineKernel {
    line_winding_field([[x0, y0], [x1, y1]])
}

#[doc(hidden)]
pub fn __quad_kernel_definer([p0, p1, p2]: [[f32; 2]; 3]) -> QuadKernel {
    quadratic_winding(p0, p1, p2)
}

// ═══════════════════════════════════════════════════════════════════════════
// Combinators
// ═══════════════════════════════════════════════════════════════════════════

/// Affine coordinate transform AST node.
/// Computes: (X - tx) * a + (Y - ty) * b
///
/// Note: The compiler fuses this into `MulAdd`, so the type matches the fusion.
pub type AffineTransform = MulAdd<Sub<X, f32>, f32, Mul<Sub<Y, f32>, f32>>;

/// Affine transform combinator type alias.
///
/// Transforms coordinates via inverse matrix before sampling inner manifold.
/// x' = (x - tx) * a + (y - ty) * b
/// y' = (x - tx) * c + (y - ty) * d
pub type Affine<M> = At<AffineTransform, AffineTransform, Z, W, M>;

/// Create an affine-transformed manifold.
pub fn affine<M>(inner: M, [a, b, c, d, tx, ty]: [f32; 6]) -> Affine<M> {
    let det = a * d - b * c;
    let inv_det = if det.abs() < 1e-6 { 0.0 } else { 1.0 / det };

    let inv_a = d * inv_det;
    let inv_b = -b * inv_det;
    let inv_c = -c * inv_det;
    let inv_d = a * inv_det;
    let inv_tx = tx;
    let inv_ty = ty;

    // x' = (X - tx) * a + (Y - ty) * b
    let trans_x = (X - inv_tx) * inv_a + (Y - inv_ty) * inv_b;

    // y' = (X - tx) * c + (Y - ty) * d
    let trans_y = (X - inv_tx) * inv_c + (Y - inv_ty) * inv_d;

    // Construct At directly to avoid requiring M: Manifold bound for the type alias if it were a function
    At {
        inner,
        x: trans_x,
        y: trans_y,
        z: Z,
        w: W,
    }
}

/// Monoid sum - accumulates winding numbers from multiple segments/glyphs.
#[derive(Clone, Debug)]
pub struct Sum<M>(pub Arc<[M]>);

impl<M: Manifold<Field, Output = Field>> Manifold<Field> for Sum<M> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        if self.0.len() == 1 {
            return self.0[0].eval_raw(x, y, z, w);
        }
        // Build sum AST and evaluate - each iteration builds Add node, then evals
        let zero = Field::from(0.0);
        self.0.iter().fold(zero, |acc, m| {
            let val = m.eval_raw(x, y, z, w);
            (acc + val).eval_raw(zero, zero, zero, zero)
        })
    }
}

impl<M: Manifold<Jet2, Output = Jet2>> Manifold<Jet2> for Sum<M> {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        if self.0.len() == 1 {
            return self.0[0].eval_raw(x, y, z, w);
        }
        let zero = Jet2::constant(Field::from(0.0));
        self.0.iter().fold(zero, |acc, m| {
            let val = m.eval_raw(x, y, z, w);
            (acc + val).eval_raw(zero, zero, zero, zero)
        })
    }
}

/// Threshold combinator - converts winding number to inside/outside (0 or 1).
///
/// Applies the non-zero winding rule: |winding| >= 0.5 means inside.
/// Threshold: |winding| >= 0.5 → 1.0, else 0.0 (non-zero winding rule)
/// Defined as combinator composition, not imperative code.
pub type Threshold<M> = Select<Ge<Abs<M>, f32>, f32, f32>;

/// Create a threshold manifold from a winding number manifold.
#[inline(always)]
pub fn threshold<M>(m: M) -> Threshold<M> {
    Select {
        cond: Ge(Abs(m), 0.5f32),
        if_true: 1.0f32,
        if_false: 0.0f32,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometry
// ═══════════════════════════════════════════════════════════════════════════

// Legacy Curve type for benchmarking
#[derive(Clone, Copy, Debug)]
pub struct Curve<const N: usize>(pub [[f32; 2]; N]);

/// Compute winding number contribution from a quadratic Bézier curve.
///
/// Uses scanline intersection to find where the curve crosses a horizontal ray
/// from the point, then sums winding contributions based on crossing direction.
#[inline]
#[define_opaque(QuadKernel)]
fn quadratic_winding([x0, y0]: [f32; 2], [x1, y1]: [f32; 2], [x2, y2]: [f32; 2]) -> QuadKernel {
    let ay = y0 - 2.0 * y1 + y2;
    let by = 2.0 * (y1 - y0);
    let ax = x0 - 2.0 * x1 + x2;
    let bx = 2.0 * (x1 - x0);
    let cx = x0;
    let inv_2a = 0.5 / ay;
    let neg_b_2a = -by * inv_2a;
    let disc_const = by * by - 4.0 * ay * y0;
    let disc_slope = 4.0 * ay;

    // Layer 3 (innermost): X=screen_x, Y=t_plus, Z=t_minus
    let winding = {
        let x_plus = Y * Y * ax + Y * bx + cx;
        let x_minus = Z * Z * ax + Z * bx + cx;
        let dy_plus = Y * (2.0 * ay) + by;
        let dy_minus = Z * (2.0 * ay) + by;

        // Ray goes right from test point - count crossings to the right
        let valid_plus = Y.ge(0.0) & Y.le(1.0) & X.lt(x_plus);
        let valid_minus = Z.ge(0.0) & Z.le(1.0) & X.lt(x_minus);

        // In Y-down coordinates: dy > 0 means downward (above → below) → -1
        let sign_plus = dy_plus.gt(0.0).select(-1.0, 1.0);
        let sign_minus = dy_minus.gt(0.0).select(-1.0, 1.0);

        valid_plus.select(sign_plus, 0.0) + valid_minus.select(sign_minus, 0.0)
    };

    // Layer 2: X=screen_x, Y=screen_y, Z=sqrt_disc
    let with_roots = winding.at(
        X,
        Z * inv_2a + neg_b_2a,  // t_plus
        Z * -inv_2a + neg_b_2a, // t_minus
        W,
    );

    // Layer 1 (outermost): screen coords
    let disc = Y * disc_slope + disc_const;
    disc.ge(0.0)
        .select(with_roots.at(X, Y, disc.max(0.0).sqrt(), W), 0.0)
}

/// Quadratic Bézier curve with baked Loop-Blinn kernel.
/// All control point computations happen at load time.
/// Includes derivative (tangent) kernel for antialiasing.
#[derive(Clone)]
pub struct Quad<K, D> {
    kernel: K,
    derivative_kernel: D,
}

/// Create a quad with baked Loop-Blinn kernel from control points.
/// The derivative of the quadratic is a line: dB/dt = 2[(1-t)(P1-P0) + t(P2-P1)]
#[inline(always)]
pub fn make_quad(points: [[f32; 2]; 3]) -> Quad<QuadKernel, LineKernel> {
    let [p0, p1, p2] = points;

    // Derivative control points: 2(P1-P0) and 2(P2-P1)
    let deriv_p0 = [2.0 * (p1[0] - p0[0]), 2.0 * (p1[1] - p0[1])];
    let deriv_p1 = [2.0 * (p2[0] - p1[0]), 2.0 * (p2[1] - p1[1])];

    Quad {
        kernel: quadratic_winding(p0, p1, p2),
        derivative_kernel: line_winding_field([deriv_p0, deriv_p1]),
    }
}

/// Helper to create a quad with baked Loop-Blinn kernel (for benchmarks).
#[inline(always)]
pub fn loop_blinn_quad(points: [[f32; 2]; 3]) -> Quad<QuadKernel, LineKernel> {
    make_quad(points)
}

impl<K, D> Quad<K, D> {
    /// Create a quad with explicit kernels (for advanced use).
    #[inline(always)]
    pub fn with_kernels(kernel: K, derivative_kernel: D) -> Self {
        Self {
            kernel,
            derivative_kernel,
        }
    }
}

/// Line segment with baked winding kernel.
/// All control point computations happen at load time.
#[derive(Clone)]
pub struct Line<K> {
    kernel: K,
}

/// Build line winding kernel (hard edges, no AA).
#[define_opaque(LineKernel)]
fn line_winding_field([[x0, y0], [x1, y1]]: [[f32; 2]; 2]) -> LineKernel {
    let (dy, dx) = (y1 - y0, x1 - x0);

    let y_min = y0.min(y1);
    let y_max = y0.max(y1);
    let in_y = Y.ge(y_min) & Y.lt(y_max);

    // x_int = (Y - y0) * (dx / dy) + x0  (x position where line crosses current Y)
    // For degenerate horizontal lines (dy ≈ 0), use safe fallback
    let safe_dy = if dy.abs() < 1e-6 { 1.0 } else { dy };
    let x_int = (Y - y0) * (dx / safe_dy) + x0;
    // In Y-down coordinates: dy > 0 means downward (above → below) → -1
    let dir = if dy > 0.0 { -1.0 } else { 1.0 };

    // Winding contribution: ray goes right from test point, counts crossings to the right
    // Degenerate horizontal lines (dy ≈ 0) are filtered out by multiplying by a mask
    let non_degenerate = if dy.abs() < 1e-6 { 0.0 } else { 1.0 };
    in_y.select(X.lt(x_int).select(dir * non_degenerate, 0.0), 0.0f32)
}

/// Create a line with baked winding kernel from control points.
#[inline(always)]
pub fn make_line(points: [[f32; 2]; 2]) -> Line<LineKernel> {
    Line {
        kernel: line_winding_field(points),
    }
}

impl<K> Line<K> {
    /// Create a line with explicit kernel (for advanced use).
    #[inline(always)]
    pub fn with_kernel(kernel: K) -> Self {
        Self { kernel }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimized Geometry with Precomputed Reciprocals
// ═══════════════════════════════════════════════════════════════════════════

/// Line segment optimized with precomputed division reciprocal.
#[derive(Clone, Copy, Debug)]
pub struct OptLine {
    x0: f32,
    y0: f32,
    y_min: f32,
    y_max: f32,
    dx_over_dy: f32, // Precomputed dx/dy
    dir: f32,        // +1 or -1
}

impl OptLine {
    /// Create from two points. Returns None for horizontal lines.
    #[inline(always)]
    pub fn new([x0, y0]: [f32; 2], [x1, y1]: [f32; 2]) -> Option<Self> {
        let dy = y1 - y0;
        if dy.abs() < 1e-6 {
            return None;
        }
        let dx = x1 - x0;
        Some(Self {
            x0,
            y0,
            y_min: y0.min(y1),
            y_max: y0.max(y1),
            dx_over_dy: dx / dy, // Division at construction, not evaluation
            dir: if dy > 0.0 { 1.0 } else { -1.0 },
        })
    }
}

/// Quadratic curve optimized with precomputed reciprocals.
#[derive(Clone, Copy, Debug)]
pub struct OptQuad {
    // Bezier coefficients
    ax: f32,
    bx: f32,
    cx: f32,
    ay: f32,
    by: f32,
    cy: f32,
    two_ay: f32,
    // Precomputed reciprocals (0.0 if degenerate)
    inv_by: f32,  // 1/by for linear Y case
    inv_2ay: f32, // 1/(2*ay) for quadratic case
    // Precomputed quadratic formula values
    neg_by: f32,
    by_sq: f32,
    four_ay: f32,
    // Flag for which case we're in
    is_linear: bool,
    is_degenerate: bool,
}

impl OptQuad {
    /// Create from three control points.
    #[inline(always)]
    pub fn new([[x0, y0], [x1, y1], [x2, y2]]: [[f32; 2]; 3]) -> Self {
        let ay = y0 - 2.0 * y1 + y2;
        let by = 2.0 * (y1 - y0);
        let cy = y0;
        let ax = x0 - 2.0 * x1 + x2;
        let bx = 2.0 * (x1 - x0);
        let cx = x0;

        let is_linear = ay.abs() < 1e-6;
        let is_degenerate = is_linear && by.abs() < 1e-6;

        Self {
            ax,
            bx,
            cx,
            ay,
            by,
            cy,
            two_ay: 2.0 * ay,
            inv_by: if !is_linear || is_degenerate {
                0.0
            } else {
                1.0 / by
            },
            inv_2ay: if is_linear { 0.0 } else { 1.0 / (2.0 * ay) },
            neg_by: -by,
            by_sq: by * by,
            four_ay: 4.0 * ay,
            is_linear,
            is_degenerate,
        }
    }
}

// ─── Field Implementation (Winding Number Coverage) ────────────────────────

impl<K: Manifold<Field, Output = Field>> Manifold<Field> for Line<K> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.kernel.eval_raw(x, y, z, w)
    }
}

impl<K: Manifold<Jet2, Output = Jet2>> Manifold<Jet2> for Line<K> {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        self.kernel.eval_raw(x, y, z, w)
    }
}

impl<K: Manifold<Jet2, Output = Jet2>> Differentiable for Line<K> {
    type DiffWrt = (X, Y); // Spatial derivatives
}

impl<K: Manifold<Field, Output = Field>, D: Send + Sync> Manifold<Field> for Quad<K, D> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.kernel.eval_raw(x, y, z, w)
    }
}

impl<K: Manifold<Jet2, Output = Jet2>, D: Send + Sync> Manifold<Jet2> for Quad<K, D> {
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        self.kernel.eval_raw(x, y, z, w)
    }
}

impl<K: Manifold<Jet2, Output = Jet2>, D: Send + Sync> Differentiable for Quad<K, D> {
    type DiffWrt = (X, Y); // Spatial derivatives
}

// Old Quad implementation using quadratic formula (for benchmarking Curve<3>)
impl Manifold<Field> for Curve<3> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let [[x0, y0], [x1, y1], [x2, y2]] = self.0;
        let (ay, by, cy) = (y0 - 2.0 * y1 + y2, 2.0 * (y1 - y0), y0);
        let (ax, bx, cx) = (x0 - 2.0 * x1 + x2, 2.0 * (x1 - x0), x0);

        if ay.abs() < 1e-6 {
            if by.abs() < 1e-6 {
                return Field::from(0.0);
            }
            // Linear case: t = (Y - cy) / by
            let t = (Y - cy) / by;
            let in_t = t.ge(0.0f32) & t.lt(1.0f32);
            let x_int = (t * ax + bx) * t + cx;
            let dx_dt = t * (2.0 * ax) + bx;
            let dy_dt = t * (2.0 * ay) + by;
            let dir = dy_dt.gt(0.0f32).select(1.0f32, -1.0f32);
            let dist = x_int - X;
            let curve_grad_sq = dx_dt * dx_dt + dy_dt * dy_dt;
            let aa_scale = dy_dt.abs() * curve_grad_sq.max(1e-12f32).rsqrt();
            let coverage = (dist * aa_scale + 0.5).max(0.0f32).min(1.0f32);
            in_t.select(coverage * dir, 0.0f32).at(x, y, z, w).eval()
        } else {
            // Quadratic: t = (-by ± sqrt(by² - 4*ay*(cy - Y))) / (2*ay)
            // c_val = cy - Y, rewritten as -Y + cy for Manifold-first ordering
            let c_val = Y * (-1.0) + cy;
            let discriminant = (c_val * (-4.0 * ay)) + (by * by);
            let valid = discriminant.ge(0.0f32);
            let sd = discriminant.abs().sqrt();
            let inv_2ay = 1.0 / (2.0 * ay);

            // t1 = (-by - sd) / (2*ay)
            let t1 = (sd * (-1.0) - by) * inv_2ay;
            let in_t1 = t1.ge(0.0f32) & t1.lt(1.0f32);
            let x_int1 = (t1 * ax + bx) * t1 + cx;
            let dx_dt1 = t1 * (2.0 * ax) + bx;
            let dy_dt1 = t1 * (2.0 * ay) + by;
            let dir1 = dy_dt1.gt(0.0f32).select(1.0f32, -1.0f32);
            let dist1 = x_int1 - X;
            let grad_sq1 = dx_dt1 * dx_dt1 + dy_dt1 * dy_dt1;
            let aa1 = dy_dt1.abs() * grad_sq1.max(1e-12f32).rsqrt();
            let cov1 = (dist1 * aa1 + 0.5).max(0.0f32).min(1.0f32);
            let contrib1 = in_t1.select(cov1 * dir1, 0.0f32);

            // t2 = (-by + sd) / (2*ay)
            let t2 = (sd - by) * inv_2ay;
            let in_t2 = t2.ge(0.0f32) & t2.lt(1.0f32);
            let x_int2 = (t2 * ax + bx) * t2 + cx;
            let dx_dt2 = t2 * (2.0 * ax) + bx;
            let dy_dt2 = t2 * (2.0 * ay) + by;
            let dir2 = dy_dt2.gt(0.0f32).select(1.0f32, -1.0f32);
            let dist2 = x_int2 - X;
            let grad_sq2 = dx_dt2 * dx_dt2 + dy_dt2 * dy_dt2;
            let aa2 = dy_dt2.abs() * grad_sq2.max(1e-12f32).rsqrt();
            let cov2 = (dist2 * aa2 + 0.5).max(0.0f32).min(1.0f32);
            let contrib2 = in_t2.select(cov2 * dir2, 0.0f32);

            valid
                .select(contrib1 + contrib2, 0.0f32)
                .at(x, y, z, w)
                .eval()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Glyph (Compositional Scene Graph)
// ═══════════════════════════════════════════════════════════════════════════

/// Optimized geometry storage separating lines and quads to avoid enum dispatch.
#[derive(Clone)]
pub struct Geometry<L, Q> {
    pub lines: Arc<[L]>,
    pub quads: Arc<[Q]>,
}

impl<L: Manifold<Field, Output = Field>, Q: Manifold<Field, Output = Field>> Manifold<Field>
    for Geometry<L, Q>
{
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let fzero = Field::from(0.0);

        // Accumulate line and quad contributions into a single composite value.
        // Note: Field + Field produces Add<Field, Field> (a manifold), so we must
        // evaluate to get back a scalar. This pattern is more efficient than the
        // old code because we defer the final evaluation until after combining
        // all contributions, rather than collapsing after each addition.
        let total = self
            .lines
            .iter()
            .map(|l| l.eval_raw(x, y, z, w))
            .chain(self.quads.iter().map(|q| q.eval_raw(x, y, z, w)))
            .fold(fzero, |acc, contrib| {
                // Compose: acc + contrib produces Add<Field, Field>
                // Evaluate the sum to collapse back to Field
                (acc + contrib).eval_raw(fzero, fzero, fzero, fzero)
            });

        // Apply non-zero winding rule: |winding| becomes coverage (clamped to [0, 1])
        total
            .abs()
            .min(Field::from(1.0))
            .eval_raw(fzero, fzero, fzero, fzero)
    }
}

impl<L: Manifold<Jet2, Output = Jet2>, Q: Manifold<Jet2, Output = Jet2>> Manifold<Jet2>
    for Geometry<L, Q>
{
    type Output = Jet2;

    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        let zero = Jet2::constant(Field::from(0.0));
        let mut acc = zero;
        for l in self.lines.iter() {
            acc = acc + l.eval_raw(x, y, z, w);
        }
        for q in self.quads.iter() {
            acc = acc + q.eval_raw(x, y, z, w);
        }
        // Apply non-zero winding rule: |winding| becomes coverage
        acc.abs().min(Jet2::constant(Field::from(1.0)))
    }
}

/// A simple glyph: segments in unit space, bounded, then transformed.
///
/// The composition is: Affine<Select<UnitSquare, Geometry, 0.0>>
/// - Geometry: Optimized Sum of Lines and Quads (produces smooth 0.0-1.0 coverage)
/// - Select (via square): Bounds check with short-circuit
/// - Affine: Restores to font coordinate space
///
/// Note: Lines and Quads now produce analytically anti-aliased coverage directly,
/// so Threshold is no longer needed.
pub type SimpleGlyph<L, Q> = Affine<Bounded<Geometry<L, Q>>>;

/// A compound glyph: sum of transformed child glyphs.
pub type CompoundGlyph<L, Q> = Sum<Affine<Glyph<L, Q>>>;

/// A glyph is either empty, a simple outline, or a compound of sub-glyphs.
#[derive(Clone)]
pub enum Glyph<L, Q> {
    /// No geometry - evaluates to 0.
    Empty,
    /// Simple glyph: bounded, thresholded segments in unit space.
    Simple(SimpleGlyph<L, Q>),
    /// Compound glyph: sum of transformed child glyphs.
    Compound(Sum<Affine<Glyph<L, Q>>>),
}

impl<L, Q> core::fmt::Debug for Glyph<L, Q> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "Glyph::Empty"),
            Self::Simple(_) => write!(f, "Glyph::Simple(...)"),
            Self::Compound(_) => write!(f, "Glyph::Compound(...)"),
        }
    }
}

// Glyph evaluation - concrete impls because Line/Quad/Segment have
// different implementations for Field (hard) vs Jet2 (anti-aliased).

impl<L: Manifold<Field, Output = Field>, Q: Manifold<Field, Output = Field>> Manifold<Field>
    for Glyph<L, Q>
{
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

impl<L: Manifold<Jet2, Output = Jet2>, Q: Manifold<Jet2, Output = Jet2>> Manifold<Jet2>
    for Glyph<L, Q>
{
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
// Reader
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
// Tables (Dependent Types)
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
    /// Format 0: sorted pairs (left_glyph, right_glyph, value)
    Fmt0 { data: &'a [u8], n_pairs: usize },
    /// No kerning table
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
                    data: &data[off + 14..], // Skip header to pairs
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
                // Binary search: each pair is 6 bytes (left:2, right:2, value:2)
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

struct SimpleParams {
    n: usize,
    scale: f32,
    tx: f32,
    ty: f32,
}

impl<'a> Font<'a> {
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        // TTF header: sfntVersion(4) + numTables(2) + searchRange(2) + entrySelector(2) + rangeShift(2) = 12 bytes
        // Table record: tag(4) + checksum(4) + offset(4) + length(4) = 16 bytes
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

    /// Lookup a glyph ID from a codepoint (single CMAP lookup).
    ///
    /// Use this when you need the glyph ID to batch multiple operations,
    /// avoiding redundant CMAP lookups in tight loops.
    #[inline]
    pub fn cmap_lookup(&self, ch: char) -> Option<u16> {
        self.cmap.lookup(ch as u32)
    }

    pub fn glyph(&self, ch: char) -> Option<Glyph<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
        self.compile(self.cmap.lookup(ch as u32)?)
    }

    /// Get glyph by pre-looked-up glyph ID (avoids redundant CMAP lookup).
    #[inline]
    pub fn glyph_by_id(
        &self,
        id: u16,
    ) -> Option<Glyph<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
        self.compile(id)
    }

    pub fn glyph_scaled(
        &self,
        ch: char,
        size: f32,
    ) -> Option<Glyph<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
        let id = self.cmap.lookup(ch as u32)?;
        self.glyph_scaled_by_id(id, size)
    }

    /// Get scaled glyph by pre-looked-up glyph ID.
    ///
    /// Avoids redundant CMAP lookup when you already have the glyph ID.
    pub fn glyph_scaled_by_id(
        &self,
        id: u16,
        size: f32,
    ) -> Option<Glyph<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
        let g = self.glyph_by_id(id)?;
        // Scale based on total font height (ascent + |descent|) instead of units_per_em
        // This ensures descenders fit within the requested size
        let total_height = self.ascent as f32 + self.descent.abs() as f32;
        let scale = size / total_height;
        // Transform: scale X, flip Y (screen Y goes down), and translate by ascent
        // so the top of the text is at Y=0 in screen coordinates.
        let y_offset = self.ascent as f32 * scale;

        Some(Glyph::Compound(Sum([affine(
            g,
            [scale, 0.0, 0.0, -scale, 0.0, y_offset],
        )]
        .into())))
    }

    pub fn advance(&self, ch: char) -> Option<f32> {
        let id = self.cmap.lookup(ch as u32)?;
        self.advance_by_id(id)
    }

    /// Get advance width in font units by pre-looked-up glyph ID.
    ///
    /// Avoids redundant CMAP lookup when you already have the glyph ID.
    #[inline]
    pub fn advance_by_id(&self, id: u16) -> Option<f32> {
        let i = (id as usize).min(self.num_hm.saturating_sub(1));
        Some(R(self.data, self.hmtx + i * 4).u16()? as f32)
    }

    pub fn advance_scaled(&self, ch: char, size: f32) -> Option<f32> {
        Some(self.advance(ch)? * size / self.units_per_em as f32)
    }

    /// Get scaled advance width by pre-looked-up glyph ID.
    ///
    /// Avoids redundant CMAP lookup when you already have the glyph ID.
    pub fn advance_scaled_by_id(&self, id: u16, size: f32) -> Option<f32> {
        Some(self.advance_by_id(id)? * size / self.units_per_em as f32)
    }

    /// Get kerning adjustment between two characters in font units.
    pub fn kern(&self, left: char, right: char) -> f32 {
        let left_id = self.cmap.lookup(left as u32).unwrap_or(0);
        let right_id = self.cmap.lookup(right as u32).unwrap_or(0);
        self.kern_by_ids(left_id, right_id)
    }

    /// Get kerning adjustment between two pre-looked-up glyph IDs in font units.
    ///
    /// Avoids redundant CMAP lookups when you already have both glyph IDs.
    #[inline]
    pub fn kern_by_ids(&self, left_id: u16, right_id: u16) -> f32 {
        self.kern.get(left_id, right_id) as f32
    }

    /// Get kerning adjustment between two characters, scaled to size.
    pub fn kern_scaled(&self, left: char, right: char, size: f32) -> f32 {
        self.kern(left, right) * size / self.units_per_em as f32
    }

    fn compile(&self, id: u16) -> Option<Glyph<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
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

        if id == 36 {
            // ASCII 'A'
            eprintln!(
                "[DEBUG] compile id={}: bbox=[{}, {}, {}, {}]",
                id, x_min, y_min, x_max, y_max
            );
        }

        let width = (x_max - x_min) as f32;
        let height = (y_max - y_min) as f32;
        let max_dim = width.max(height).max(1.0); // Avoid div by 0

        // Normalize transform: map [x_min, x_min+max_dim] -> [0, 1]
        let norm_scale = 1.0 / max_dim;
        let norm_tx = -(x_min as f32) * norm_scale;
        let norm_ty = -(y_min as f32) * norm_scale;

        // The restore transform maps [0, 1] back to font units
        // x_world = x_local * max_dim + x_min
        // y_world = y_local * max_dim + y_min (no flip - keep Y-down from TrueType)
        let restore = [max_dim, 0.0, 0.0, max_dim, x_min as f32, y_min as f32];

        if n >= 0 {
            // Parse segments in normalized [0,1] space
            let params = SimpleParams {
                n: n as usize,
                scale: norm_scale,
                tx: norm_tx,
                ty: norm_ty,
            };
            let sum_segs = self.simple(&mut r, params)?;

            // Compose: Geometry (smooth AA coverage) -> Bounded (via square) -> Affine
            let bounded = square(sum_segs, 0.0f32);
            Some(Glyph::Simple(affine(bounded, restore)))
        } else {
            // Compound glyphs: children are already fully composed with their own bounds
            self.compound(&mut r)
        }
    }

    fn simple(
        &self,
        r: &mut R,
        params: SimpleParams,
    ) -> Option<Geometry<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
        let SimpleParams { n, scale, tx, ty } = params;

        if n == 0 {
            return Some(Geometry {
                lines: vec![].into(),
                quads: vec![].into(),
            });
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

        // Normalize points immediately
        let pts: Vec<_> = (0..np)
            .map(|i| {
                (
                    (xs[i] as f32) * scale + tx,
                    (ys[i] as f32) * scale + ty,
                    fl[i] & 1 != 0,
                )
            })
            .collect();

        // Partition segments into lines and quads
        let mut lines = Vec::new();
        let mut quads = Vec::new();

        let mut start = 0;
        for &e in ends.iter() {
            let c = &pts[start..=e];
            start = e + 1;
            push_segs(c, &mut lines, &mut quads);
        }

        Some(Geometry {
            lines: lines.into(),
            quads: quads.into(),
        })
    }

    fn compound(&self, r: &mut R) -> Option<Glyph<Line<LineKernel>, Quad<QuadKernel, LineKernel>>> {
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

fn push_segs(
    pts: &[(f32, f32, bool)],
    lines: &mut Vec<Line<LineKernel>>,
    quads: &mut Vec<Quad<QuadKernel, LineKernel>>,
) {
    if pts.is_empty() {
        return;
    }
    let exp: Vec<_> = pts
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

    if exp.is_empty() {
        return;
    }

    let start = exp.iter().position(|p| p.2).unwrap_or(0);
    let mut i = 0;
    while i < exp.len() {
        let p = |j: usize| {
            let (x, y, _) = exp[(start + j) % exp.len()];
            [x, y]
        };
        if exp[(start + i + 1) % exp.len()].2 {
            lines.push(make_line([p(i), p(i + 1)]));
            i += 1;
        } else {
            quads.push(make_quad([p(i), p(i + 1), p(i + 2)]));
            i += 2;
        }
    }
}
