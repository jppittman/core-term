//! 3D Scene Rendering Engine built on PixelFlow JIT [`Kernel`].
//!
//! Provides coordinate mapping, 3D surface geometry, screen-space tangent frames,
//! and material shading compiled via `Lattice::bake`.

use pixelflow_core::Kernel;

/// JIT Kernel 3D scene construction utilities.
///
/// Builds 3D scene graphs as pure [`Kernel`] values. Tangent frames and surface
/// normals are derived symbolically via `Dwrt` (`.dx()` / `.dy()`), eliminating
/// legacy `Jet3` combinators and enabling direct JIT compilation through `Lattice::bake`.
pub mod kernel_3d {
    use super::*;

    /// Screen-to-ray direction transformation for JIT `Kernel` scenes.
    ///
    /// Maps `(sx, sy)` screen coordinate kernels into a unit ray direction vector `(dx, dy, dz)`.
    #[must_use]
    pub fn screen_to_ray(sx: &Kernel, sy: &Kernel, focal_len: f32) -> (Kernel, Kernel, Kernel) {
        let sz = Kernel::constant(focal_len);
        let len_sq = sx.mul(sx).add(&sy.mul(sy)).add(&sz.mul(&sz));
        let len = len_sq.sqrt();
        (sx.div(&len), sy.div(&len), sz.div(&len))
    }

    /// Unit sphere geometry centered at origin for JIT `Kernel` scenes.
    #[must_use]
    pub fn unit_sphere(dx: &Kernel, dy: &Kernel, dz: &Kernel) -> Kernel {
        let len_sq = dx.mul(dx).add(&dy.mul(dy)).add(&dz.mul(dz));
        Kernel::constant(1.0).div(&len_sq.sqrt())
    }

    /// Horizontal plane geometry at height `h` for JIT `Kernel` scenes.
    #[must_use]
    pub fn plane(height: f32, dy: &Kernel) -> Kernel {
        Kernel::constant(height).div(dy)
    }

    /// Sphere centered at `center` with radius `r` for JIT `Kernel` scenes.
    #[must_use]
    pub fn sphere_at(
        center: (f32, f32, f32),
        radius: f32,
        dx: &Kernel,
        dy: &Kernel,
        dz: &Kernel,
    ) -> Kernel {
        let cx = Kernel::constant(center.0);
        let cy = Kernel::constant(center.1);
        let cz = Kernel::constant(center.2);
        let r_sq = Kernel::constant(radius * radius);

        let d_dot_c = dx.mul(&cx).add(&dy.mul(&cy)).add(&dz.mul(&cz));
        let c_sq = cx.mul(&cx).add(&cy.mul(&cy)).add(&cz.mul(&cz));

        let discriminant = d_dot_c.mul(&d_dot_c).sub(&c_sq.sub(&r_sq));
        let eps = Kernel::constant(0.0001);
        d_dot_c.sub(&discriminant.add(&eps).sqrt())
    }

    /// Computes unit surface normal `(Nx, Ny, Nz)` from hit position `(Px, Py, Pz)`.
    ///
    /// Derives screen-space tangent vectors `Tx = ∂P/∂X` and `Ty = ∂P/∂Y` using `Dwrt`
    /// (`.dx()` and `.dy()`), and computes their normalized cross product `Tx × Ty`.
    #[must_use]
    pub fn surface_normal(px: &Kernel, py: &Kernel, pz: &Kernel) -> (Kernel, Kernel, Kernel) {
        let tx_x = px.dx();
        let tx_y = py.dx();
        let tx_z = pz.dx();

        let ty_x = px.dy();
        let ty_y = py.dy();
        let ty_z = pz.dy();

        let nx = tx_y.mul(&ty_z).sub(&tx_z.mul(&ty_y));
        let ny = tx_z.mul(&ty_x).sub(&tx_x.mul(&ty_z));
        let nz = tx_x.mul(&ty_y).sub(&tx_y.mul(&ty_x));

        let len_sq = nx.mul(&nx).add(&ny.mul(&ny)).add(&nz.mul(&nz));
        let inv_len = len_sq.sqrt().recip();

        (nx.mul(&inv_len), ny.mul(&inv_len), nz.mul(&inv_len))
    }

    /// Sky gradient material kernel.
    #[must_use]
    pub fn sky(dy: &Kernel) -> Kernel {
        let t = dy.mul(&Kernel::constant(0.5)).add(&Kernel::constant(0.5));
        let clamped = t.max(&Kernel::constant(0.0)).min(&Kernel::constant(1.0));
        Kernel::constant(0.1).add(&clamped.mul(&Kernel::constant(0.8)))
    }
}
