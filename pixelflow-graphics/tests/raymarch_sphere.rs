//! Test: 3D scene rendering with sphere + floor using JIT Kernel architecture (`kernel_3d`)

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::scene3d::kernel_3d;
use std::fs::File;
use std::io::Write;

#[test]
fn sphere_on_floor() {
    const W: usize = 400;
    const H: usize = 300;
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

    let (t_sphere, hit_sphere) = kernel_3d::sphere_at((0.0, 0.5, 4.0), 1.0, &dx, &dy, &dz);
    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);

    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);
    let scene = hit_sphere.select(&ny, &sky);

    let lattice = Lattice::frame(W, H, 0.0);
    let baked = lattice.bake(&scene);
    let buffer = baked.buffer();

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    for (i, &val) in buffer.iter().enumerate() {
        let b = (val.clamp(0.0, 1.0) * 255.0) as u8;
        frame.data[i] = Rgba8::new(b, b, b, 255);
    }

    let path = std::env::temp_dir().join("raymarch_sphere_jit.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    assert!(path.exists());
}
