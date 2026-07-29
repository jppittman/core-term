//! 3D Scene Rendering Engine built on PixelFlow JIT [`Kernel`].
//!
//! Provides coordinate mapping, 3D surface geometry, screen-space tangent frames,
//! procedural materials, and Householder reflection compiled via `Lattice::bake`.

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

    /// 2D procedural checkerboard pattern over `(px, pz)`.
    #[must_use]
    pub fn checker(px: &Kernel, pz: &Kernel, scale: f32) -> Kernel {
        let inv_scale = Kernel::constant(1.0 / scale);
        let cx = px.mul(&inv_scale).floor();
        let cz = pz.mul(&inv_scale).floor();

        let two = Kernel::constant(2.0);
        let x_even = cx.sub(&cx.div(&two).floor().mul(&two));
        let z_even = cz.sub(&cz.div(&two).floor().mul(&two));

        let diff = x_even.sub(&z_even).abs();
        diff.select(&Kernel::constant(0.2), &Kernel::constant(0.8))
    }

    /// Householder reflection ray direction `(Rx, Ry, Rz)` from incident `(Dx, Dy, Dz)` and normal `(Nx, Ny, Nz)`.
    #[must_use]
    pub fn reflect(
        dx: &Kernel,
        dy: &Kernel,
        dz: &Kernel,
        nx: &Kernel,
        ny: &Kernel,
        nz: &Kernel,
    ) -> (Kernel, Kernel, Kernel) {
        let d_dot_n = dx.mul(nx).add(&dy.mul(ny)).add(&dz.mul(nz));
        let k = Kernel::constant(2.0).mul(&d_dot_n);
        (
            dx.sub(&k.mul(nx)),
            dy.sub(&k.mul(ny)),
            dz.sub(&k.mul(nz)),
        )
    }

    /// World scene background (checkerboard floor + sky).
    #[must_use]
    pub fn world(dx: &Kernel, dy: &Kernel, dz: &Kernel) -> Kernel {
        let floor_height = -1.0f32;
        let t_floor = Kernel::constant(floor_height).div(dy);

        let px_floor = dx.mul(&t_floor);
        let pz_floor = dz.mul(&t_floor);

        let floor_color = checker(&px_floor, &pz_floor, 1.0);
        let sky_color = sky(dy);

        let hit_floor = dy.lt(&Kernel::constant(0.0));
        hit_floor.select(&floor_color, &sky_color)
    }

    /// Complete 3D reflective chrome sphere scene over screen coordinates `(sx, sy)`.
    #[must_use]
    pub fn chrome_scene(sx: &Kernel, sy: &Kernel, sphere_center: (f32, f32, f32), radius: f32) -> Kernel {
        let (dx, dy, dz) = screen_to_ray(sx, sy, 1.0);
        let direct_world = world(&dx, &dy, &dz);

        let t_sphere = sphere_at(sphere_center, radius, &dx, &dy, &dz);
        let hit_sphere = t_sphere.gt(&Kernel::constant(0.0));

        let px = dx.mul(&t_sphere);
        let py = dy.mul(&t_sphere);
        let pz = dz.mul(&t_sphere);

        let (nx, ny, nz) = surface_normal(&px, &py, &pz);
        let (rx, ry, rz) = reflect(&dx, &dy, &dz, &nx, &ny, &nz);

        let reflected_world = world(&rx, &ry, &rz);

        hit_sphere.select(&reflected_world, &direct_world)
    }
}
