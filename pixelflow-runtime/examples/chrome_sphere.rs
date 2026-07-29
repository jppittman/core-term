//! Chrome Sphere JIT Rendering Demo
//!
//! Renders the 3D chrome sphere scene (with checkerboard floor, sky gradient, and Householder reflections)
//! JIT-compiled via `Lattice::bake`.

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::scene3d::kernel_3d;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;

fn build_scene() -> Kernel {
    let scale = 2.0 / H as f32;
    let sx = Kernel::x()
        .sub(&Kernel::constant(W as f32 * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(H as f32 * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    kernel_3d::chrome_scene(&sx, &sy, (0.0, 0.0, 4.0), 1.0)
}

fn main() {
    println!("Chrome Sphere JIT Rendering Demo");
    println!("================================");
    println!(
        "Resolution: {}x{} ({:.1}M pixels)",
        W,
        H,
        (W * H) as f64 / 1_000_000.0
    );
    println!();

    let scene_kernel = build_scene();
    let lattice = Lattice::frame(W, H, 0.0);

    let start = Instant::now();
    let baked = lattice.bake(&scene_kernel);
    let elapsed = start.elapsed();

    let buffer = baked.buffer();
    let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let fps = 1.0 / elapsed.as_secs_f64();
    println!(
        "JIT Baked render: {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS)",
        elapsed.as_secs_f64() * 1000.0,
        mpps,
        fps
    );

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    for (i, &val) in buffer.iter().enumerate() {
        let b = (val.clamp(0.0, 1.0) * 255.0) as u8;
        frame.data[i] = Rgba8::new(b, b, b, 255);
    }

    let path = std::env::temp_dir().join("chrome_sphere_jit.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());
    println!("Done!");
}
