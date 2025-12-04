use crate::backend::{Backend, SimdBatch};
use crate::batch::{Batch, NativeBackend};

/// A 3x3 matrix for 2D affine transformations.
///
/// Layout:
/// [ m00 m01 m02 ]
/// [ m10 m11 m12 ]
/// [ m20 m21 m22 ]
#[derive(Clone, Copy, Debug)]
pub struct Mat3 {
    /// Row-major matrix elements.
    pub m: [[f32; 3]; 3],
}

impl Mat3 {
    /// Identity matrix.
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Creates a transformation matrix that maps unit triangle to the given points.
    ///
    /// Maps:
    /// (0, 0) -> p0
    /// (0.5, 0) -> p1
    /// (1, 1) -> p2
    pub fn from_affine_points(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> Option<Self> {
        let c = p0[0];
        let f = p0[1];

        let a = (p1[0] - c) * 2.0;
        let d = (p1[1] - f) * 2.0;

        let b = p2[0] - a - c;
        let e = p2[1] - d - f;

        let forward = Self {
            m: [[a, b, c], [d, e, f], [0.0, 0.0, 1.0]],
        };

        forward.inverse()
    }

    /// Computes the inverse of the matrix.
    pub fn inverse(&self) -> Option<Self> {
        NativeBackend::inverse_mat3(self.m).map(|m| Self { m })
    }

    /// Transforms screen coordinates (x, y) into parameter space (u, v).
    /// Note: Assumes homogenous coordinate w=1.
    pub fn transform(&self, x: Batch<u32>, y: Batch<u32>) -> (Batch<f32>, Batch<f32>) {
        let x_f = NativeBackend::u32_to_f32(x);
        let y_f = NativeBackend::u32_to_f32(y);

        let m00 = Batch::<f32>::splat(self.m[0][0]);
        let m01 = Batch::<f32>::splat(self.m[0][1]);
        let m02 = Batch::<f32>::splat(self.m[0][2]);

        let m10 = Batch::<f32>::splat(self.m[1][0]);
        let m11 = Batch::<f32>::splat(self.m[1][1]);
        let m12 = Batch::<f32>::splat(self.m[1][2]);

        // u = m00*x + m01*y + m02
        let u = m00 * x_f + m01 * y_f + m02;

        // v = m10*x + m11*y + m12
        let v = m10 * x_f + m11 * y_f + m12;

        (u, v)
    }
}
