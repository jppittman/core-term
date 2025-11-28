use crate::batch::Batch;
use crate::diff::{DiffCoord, DiffSurface};

/// A 3x2 Affine Matrix for 2D transformations.
///
/// ```text
/// [ x' ]   [ m00 m01 m02 ] [ x ]
/// [ y' ] = [ m10 m11 m12 ] [ y ]
///          [  0   0   1  ] [ 1 ]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Mat3x2 {
    /// Element at row 0, column 0.
    pub m00: f32,
    /// Element at row 0, column 1.
    pub m01: f32,
    /// Element at row 0, column 2.
    pub m02: f32,
    /// Element at row 1, column 0.
    pub m10: f32,
    /// Element at row 1, column 1.
    pub m11: f32,
    /// Element at row 1, column 2.
    pub m12: f32,
}

/// A quadratic Bézier curve as an implicit surface.
///
/// Uses Loop-Blinn rendering: transforms screen coordinates to canonical
/// space where the curve is simply f(u,v) = u² - v.
#[derive(Clone, Copy)]
pub struct QuadraticCurve {
    /// Matrix that maps screen(x,y) → canonical(u,v).
    /// Precomputed from control points P₀, P₁, P₂.
    pub matrix: Mat3x2,
}

impl QuadraticCurve {
    /// Construct from control points.
    ///
    /// Computes the inverse transform required to map the control triangle
    /// to the canonical triangle (0,0), (0.5,0), (1,1).
    pub fn new(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> Self {
        let ax = p0[0];
        let ay = p0[1];
        let bx = p1[0];
        let by = p1[1];
        let cx = p2[0];
        let cy = p2[1];

        // Determinant of the P matrix (ignoring z=1 for now, standard 2D affine inversion math)
        // Matrix A = [ P0 | P1 | P2 ] ? No, see derivation.
        // We solved [M] * [P] = [C].
        // Det of [P0x P0y 1; P1x P1y 1; P2x P2y 1]
        let det = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by);

        // If degenerate, return identity (or handling for line segment?)
        if det.abs() < 1e-6 {
            return Self {
                matrix: Mat3x2 {
                    m00: 0.0, m01: 0.0, m02: 0.0,
                    m10: 0.0, m11: 0.0, m12: 0.0,
                },
            };
        }

        let inv_det = 1.0 / det;

        // Cofactors of P matrix
        let _a00 = (by - cy) * inv_det;
        let a01 = (cy - ay) * inv_det;
        let a02 = (ay - by) * inv_det;

        let _a10 = (cx - bx) * inv_det;
        let a11 = (ax - cx) * inv_det;
        let a12 = (bx - ax) * inv_det;

        let _a20 = (bx * cy - cx * by) * inv_det;
        let a21 = (cx * ay - ax * cy) * inv_det;
        let a22 = (ax * by - bx * ay) * inv_det;

        // Matrix multiplication with C matrix
        // u coefficients
        let m00 = 0.5 * a01 + a02;
        let m01 = 0.5 * a11 + a12;
        let m02 = 0.5 * a21 + a22;

        // v coefficients
        let m10 = a02;
        let m11 = a12;
        let m12 = a22;

        Self {
            matrix: Mat3x2 {
                m00,
                m01,
                m02,
                m10,
                m11,
                m12,
            },
        }
    }
}

impl DiffSurface for QuadraticCurve {
    type Output = Batch<f32>;

    #[inline(always)]
    fn sample_diff(&self, x: DiffCoord, y: DiffCoord) -> Batch<f32> {
        let m = self.matrix;

        let m00 = DiffCoord::constant(Batch::splat(m.m00));
        let m01 = DiffCoord::constant(Batch::splat(m.m01));
        let m02 = DiffCoord::constant(Batch::splat(m.m02));
        let m10 = DiffCoord::constant(Batch::splat(m.m10));
        let m11 = DiffCoord::constant(Batch::splat(m.m11));
        let m12 = DiffCoord::constant(Batch::splat(m.m12));

        // u = x * m00 + y * m01 + m02
        let u = x * m00 + y * m01 + m02;

        // v = x * m10 + y * m11 + m12
        let v = x * m10 + y * m11 + m12;

        // Implicit function f(u,v) = u^2 - v
        // The DiffCoord arithmetic automatically propagates the chain rule
        // to compute df/dx and df/dy (stored in f.dx, f.dy).
        let f = u * u - v;

        // Compute gradient magnitude in screen space: |∇f| = sqrt( (df/dx)^2 + (df/dy)^2 )
        let grad_len = (f.dx * f.dx + f.dy * f.dy).sqrt();

        // Signed distance approx = f / |∇f|
        // If grad_len is very small (near flat region or singularity), clamp it?
        // For now, let it be (float division by zero gives inf, which is fine for distance).
        f.val / grad_len
    }
}
