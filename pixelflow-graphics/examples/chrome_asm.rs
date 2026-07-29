//! ASM inspection for JIT compiled 3D chrome sphere kernel.

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::scene3d::kernel_3d;
use std::hint::black_box;

fn build_scene() -> Kernel {
    let scale = 2.0 / 1080.0f32;
    let sx = Kernel::x()
        .sub(&Kernel::constant(1920.0 * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(1080.0 * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    let (dx, dy, dz) = kernel_3d::screen_to_ray(&sx, &sy, 1.0);
    let sky = kernel_3d::sky(&dy);

    let t_sphere = kernel_3d::sphere_at((0.0, 0.0, 4.0), 1.0, &dx, &dy, &dz);
    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);

    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);

    let hit_mask = t_sphere.gt(&Kernel::constant(0.0));
    hit_mask.select(&ny, &sky)
}

#[inline(never)]
pub fn bake_chrome_scene() {
    let scene = build_scene();
    let lattice = Lattice::frame(1920, 1080, 0.0);
    let baked = lattice.bake(&scene);
    black_box(baked.buffer());
}

fn main() {
    println!("Evaluating JIT compiled chrome scene...");
    bake_chrome_scene();
    println!("Done.");
}
