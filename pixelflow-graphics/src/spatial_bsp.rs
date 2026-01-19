use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt};

/// A Binary Space Partition (BSP) node for spatial partitioning.
///
/// Divides space into two halves based on a plane equation `nx*x + ny*y + nz*z + d > 0`.
///
/// This is a manifold: it evaluates its children recursively based on the coordinate.
#[derive(Clone)]
pub struct SpatialBsp<L, R> {
    /// Normal vector x component.
    pub nx: f32,
    /// Normal vector y component.
    pub ny: f32,
    /// Normal vector z component.
    pub nz: f32,
    /// Distance from origin.
    pub d: f32,
    /// Left child (positive half-space).
    pub left: L,
    /// Right child (negative half-space).
    pub right: R,
}

impl<L, R> SpatialBsp<L, R> {
    pub fn new(normal: (f32, f32, f32), d: f32, left: L, right: R) -> Self {
        let (nx, ny, nz) = normal;
        // Normalize the normal vector for correct distance calculation
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let inv_len = if len > 0.0 { 1.0 / len } else { 0.0 };
        Self {
            nx: nx * inv_len,
            ny: ny * inv_len,
            nz: nz * inv_len,
            d: d * inv_len,
            left,
            right,
        }
    }
}

// Specialization for Field (Rasterization)
// Only supporting Discrete output for now to avoid conflict
impl<L, R> Manifold<Field> for SpatialBsp<L, R>
where
    L: Manifold<Field, Output = Discrete>,
    R: Manifold<Field, Output = Discrete>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let nx = Field::from(self.nx);
        let ny = Field::from(self.ny);
        let nz = Field::from(self.nz);
        let d = Field::from(self.d);

        // Evaluate distance expression to a value (Field)
        // x, y, z are Field values. Arithmetic builds AST.
        // .constant() collapses it to Field value.
        let dist = (x * nx + y * ny + z * nz + d).constant();
        let zero = Field::from(0.0);
        let mask = dist.ge(zero); // Field::ge returns Field (mask)

        if mask.all() {
            return self.left.eval_raw(x, y, z, w);
        }
        if !mask.any() {
            return self.right.eval_raw(x, y, z, w);
        }

        let l_val = self.left.eval_raw(x, y, z, w);
        let r_val = self.right.eval_raw(x, y, z, w);

        Discrete::select(mask, l_val, r_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::{materialize_discrete, RgbaComponents};

    struct SolidColor(f32, f32, f32);
    impl Manifold<Field> for SolidColor {
        type Output = Discrete;
        fn eval_raw(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> Discrete {
            Discrete::pack(RgbaComponents {
                r: Field::from(self.0),
                g: Field::from(self.1),
                b: Field::from(self.2),
                a: Field::from(1.0),
            })
        }
    }

    #[test]
    fn test_bsp_split() {
        let left = SolidColor(1.0, 0.0, 0.0); // Red
        let right = SolidColor(0.0, 0.0, 1.0); // Blue
        // Plane x > 0
        let bsp = SpatialBsp::new((1.0, 0.0, 0.0), 0.0, left, right);

        // Test left side (x=10) -> Red
        let mut out = [0u32; 16];
        materialize_discrete(&bsp, 10.0, 0.0, &mut out);
        
        // RGBA packed: Red=0xFF0000FF (LE: A B G R -> FF 00 00 FF? No, R G B A in memory)
        // pack_rgba produces: r | (g<<8) | (b<<16) | (a<<24)
        // Red (1,0,0,1) -> 255 | 0 | 0 | 255<<24 = 0xFF0000FF
        assert_eq!(out[0], 0xFF0000FF);

        // Test right side (x=-10) -> Blue
        materialize_discrete(&bsp, -10.0, 0.0, &mut out);
        // Blue (0,0,1,1) -> 0 | 0 | 255<<16 | 255<<24 = 0xFFFF0000
        assert_eq!(out[0], 0xFFFF0000);
    }
}
