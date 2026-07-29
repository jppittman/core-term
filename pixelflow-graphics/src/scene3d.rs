//! 3D Scene Rendering Engine built on PixelFlow JIT [`Kernel`].
//!
//! Provides coordinate mapping, 3D surface geometry, screen-space tangent frames,
//! procedural RGB materials, and Householder reflection compiled via `Lattice::bake`.

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
    ///
    /// Returns `(t_sphere, hit_mask)` where `t_sphere` is the ray intersection distance
    /// and `hit_mask` is a boolean mask indicating valid non-NaN hits.
    #[must_use]
    pub fn sphere_at(
        center: (f32, f32, f32),
        radius: f32,
        dx: &Kernel,
        dy: &Kernel,
        dz: &Kernel,
    ) -> (Kernel, Kernel) {
        let cx = Kernel::constant(center.0);
        let cy = Kernel::constant(center.1);
        let cz = Kernel::constant(center.2);
        let r_sq = Kernel::constant(radius * radius);

        let d_dot_c = dx.mul(&cx).add(&dy.mul(&cy)).add(&dz.mul(&cz));
        let c_sq = cx.mul(&cx).add(&cy.mul(&cy)).add(&cz.mul(&cz));

        let discriminant = d_dot_c.mul(&d_dot_c).sub(&c_sq.sub(&r_sq));
        let zero = Kernel::constant(0.0);
        let disc_valid = discriminant.ge(&zero);

        // Clamp negative discriminant to 0.0 before sqrt to prevent NaN propagation
        let safe_disc = disc_valid.select(&discriminant, &zero);
        let t_sphere = d_dot_c.sub(&safe_disc.sqrt());
        let hit = disc_valid.and(&t_sphere.gt(&zero));

        (t_sphere, hit)
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

    /// Sky gradient scalar material kernel.
    #[must_use]
    pub fn sky(dy: &Kernel) -> Kernel {
        sky_rgb(dy).2
    }

    /// Sky gradient RGB material kernel (deep horizon sky blue).
    #[must_use]
    pub fn sky_rgb(dy: &Kernel) -> (Kernel, Kernel, Kernel) {
        let t = dy.mul(&Kernel::constant(0.5)).add(&Kernel::constant(0.5));
        let clamped = t.max(&Kernel::constant(0.0)).min(&Kernel::constant(1.0));

        let r = Kernel::constant(0.7).sub(&clamped.mul(&Kernel::constant(0.5)));
        let g = Kernel::constant(0.85).sub(&clamped.mul(&Kernel::constant(0.45)));
        let b = Kernel::constant(1.0).sub(&clamped.mul(&Kernel::constant(0.2)));

        (r, g, b)
    }

    /// 2D procedural warm/cool checkerboard pattern RGB over `(px, pz)`.
    #[must_use]
    pub fn checker_rgb(px: &Kernel, pz: &Kernel, scale: f32) -> (Kernel, Kernel, Kernel) {
        let inv_scale = Kernel::constant(1.0 / scale);
        let cx = px.mul(&inv_scale).floor();
        let cz = pz.mul(&inv_scale).floor();

        let two = Kernel::constant(2.0);
        let x_even = cx.sub(&cx.div(&two).floor().mul(&two));
        let z_even = cz.sub(&cz.div(&two).floor().mul(&two));

        let diff = x_even.sub(&z_even).abs();
        let is_even = diff.lt(&Kernel::constant(0.5));

        // Warm tile: (0.95, 0.90, 0.80), Cool tile: (0.20, 0.25, 0.30)
        let r = is_even.select(&Kernel::constant(0.95), &Kernel::constant(0.20));
        let g = is_even.select(&Kernel::constant(0.90), &Kernel::constant(0.25));
        let b = is_even.select(&Kernel::constant(0.80), &Kernel::constant(0.30));

        (r, g, b)
    }

    /// Householder reflection ray direction `(Rx, Ry, Rz)` from incident `(Dx, Dy, Dz)` and normal `(Nx, Ny, Nz)`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
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
        (dx.sub(&k.mul(nx)), dy.sub(&k.mul(ny)), dz.sub(&k.mul(nz)))
    }

    /// World scene background RGB (warm/cool checkerboard floor + sky gradient).
    #[must_use]
    pub fn world_rgb(dx: &Kernel, dy: &Kernel, dz: &Kernel) -> (Kernel, Kernel, Kernel) {
        let floor_height = -1.0f32;
        let t_floor = Kernel::constant(floor_height).div(dy);

        let px_floor = dx.mul(&t_floor);
        let pz_floor = dz.mul(&t_floor);

        let (fr, fg, fb) = checker_rgb(&px_floor, &pz_floor, 1.0);
        let (sr, sg, sb) = sky_rgb(dy);

        let hit_floor = dy.lt(&Kernel::constant(0.0));
        (
            hit_floor.select(&fr, &sr),
            hit_floor.select(&fg, &sg),
            hit_floor.select(&fb, &sb),
        )
    }

    /// Complete 3D reflective chrome sphere scene returning `(R, G, B)` Kernels over screen coordinates `(sx, sy)`.
    #[must_use]
    pub fn chrome_scene_rgb(
        sx: &Kernel,
        sy: &Kernel,
        sphere_center: (f32, f32, f32),
        radius: f32,
    ) -> (Kernel, Kernel, Kernel) {
        let (dx, dy, dz) = screen_to_ray(sx, sy, 1.0);
        let (dr, dg, db) = world_rgb(&dx, &dy, &dz);

        let (t_sphere, hit_sphere) = sphere_at(sphere_center, radius, &dx, &dy, &dz);

        let px = dx.mul(&t_sphere);
        let py = dy.mul(&t_sphere);
        let pz = dz.mul(&t_sphere);

        let (nx, ny, nz) = surface_normal(&px, &py, &pz);
        let (rx, ry, rz) = reflect(&dx, &dy, &dz, &nx, &ny, &nz);

        let (rr, rg, rb) = world_rgb(&rx, &ry, &rz);

        (
            hit_sphere.select(&rr, &dr),
            hit_sphere.select(&rg, &dg),
            hit_sphere.select(&rb, &db),
        )
    }
}
