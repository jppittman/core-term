//! Test: Analytic 3D rendering with JIT Kernel architecture (`kernel_3d`).

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::scene3d::kernel_3d;

#[test]
fn jit_kernel_3d_sky_and_normal() {
    const W: f32 = 32.0;
    const H: f32 = 32.0;
    let scale = 2.0 / H;

    let sx = Kernel::x()
        .sub(&Kernel::constant(W * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(H * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    let (dx, dy, dz) = kernel_3d::screen_to_ray(&sx, &sy, 1.0);
    let sky = kernel_3d::sky(&dy);

    let t_sphere = kernel_3d::unit_sphere(&dx, &dy, &dz);
    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);

    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);

    let lattice = Lattice::frame(32, 32, 0.0);
    let baked_sky = lattice.bake(&sky);
    assert_eq!(baked_sky.buffer().len(), 1024);

    let top_sample = baked_sky.buffer()[16];
    let bot_sample = baked_sky.buffer()[31 * 32 + 16];
    assert!(
        bot_sample > top_sample,
        "Horizon sky ({bot_sample}) should be brighter than zenith sky ({top_sample})"
    );

    let baked_ny = lattice.bake(&ny);
    assert_eq!(baked_ny.buffer().len(), 1024);
}

#[test]
fn jit_kernel_3d_sphere_on_plane() {
    const W: usize = 64;
    const H: usize = 48;
    let w_f32 = W as f32;
    let h_f32 = H as f32;
    let scale = 2.0 / h_f32;

    let sx = Kernel::x()
        .sub(&Kernel::constant(w_f32 * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(h_f32 * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    let (dx, dy, dz) = kernel_3d::screen_to_ray(&sx, &sy, 1.0);
    let sky = kernel_3d::sky(&dy);

    let (t_sphere, hit_sphere) = kernel_3d::sphere_at((0.0, 0.0, 4.0), 1.0, &dx, &dy, &dz);

    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);
    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);

    let scene = hit_sphere.select(&ny, &sky);

    let lattice = Lattice::frame(W, H, 0.0);
    let baked = lattice.bake(&scene);
    assert_eq!(baked.buffer().len(), W * H);
}
