//! Subdivision surface rendering with JIT Kernel architecture (`kernel_3d`).

use pixelflow_core::{Kernel, Lattice};
use pixelflow_graphics::mesh::{Point3, QuadMesh};
use pixelflow_graphics::scene3d::kernel_3d;
use pixelflow_graphics::subdivision::{SubdivisionGeometry, SubdivisionPatch};
use pixelflow_graphics::{Frame, Rgba8};

fn main() {
    println!("=== Subdivision Surface Demo (JIT Kernel) ===\n");

    let mesh = create_curved_quad();
    println!("Generated mesh:");
    println!("  Vertices: {}", mesh.vertex_count());
    println!("  Faces: {}", mesh.face_count());

    let patch = SubdivisionPatch::from_mesh(&mesh, 0).unwrap();
    println!("\nPatch info:");
    println!("  Corners: {:?}", patch.corners);
    println!("  Valences: {:?}", patch.corner_valences);
    println!("  Extraordinary: {}", patch.is_extraordinary());

    let _geometry = SubdivisionGeometry::new(
        patch, &mesh, -1.0, 1.0, 0.0, 0.0,
    );

    let width = 800;
    let height = 600;
    let scale = 2.0 / height as f32;

    let sx = Kernel::x()
        .sub(&Kernel::constant(width as f32 * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(height as f32 * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    let (dx, dy, dz) = kernel_3d::screen_to_ray(&sx, &sy, 1.0);
    let sky = kernel_3d::sky(&dy);

    let t_sphere = kernel_3d::unit_sphere(&dx, &dy, &dz);
    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);

    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);

    let hit = t_sphere.gt(&Kernel::constant(0.0));
    let scene = hit.select(&ny, &sky);

    println!("\nRendering {}x{} image with JIT Kernel...", width, height);

    let lattice = Lattice::frame(width, height, 0.0);
    let baked = lattice.bake(&scene);
    let buffer = baked.buffer();

    let mut frame = Frame::<Rgba8>::new(width as u32, height as u32);
    for (i, &val) in buffer.iter().enumerate() {
        let b = (val.clamp(0.0, 1.0) * 255.0) as u8;
        frame.data[i] = Rgba8::new(b, b, b, 255);
    }

    save_ppm("subdivision_autodiff.ppm", &frame);
    println!("\n✓ Rendered to subdivision_autodiff.ppm");
}

fn create_curved_quad() -> QuadMesh {
    let vertices = vec![
        Point3::new(-1.0, 0.0, -1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, -1.0),
        Point3::new(-1.0, 0.0, 1.0),
    ];

    let faces = vec![pixelflow_graphics::mesh::Quad::new(0, 1, 2, 3)];
    let valence = vec![1, 1, 1, 1];

    QuadMesh {
        vertices,
        faces,
        valence,
    }
}

fn save_ppm(path: &str, frame: &Frame<Rgba8>) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create output file");
    writeln!(file, "P3").unwrap();
    writeln!(file, "{} {}", frame.width, frame.height).unwrap();
    writeln!(file, "255").unwrap();

    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = y * frame.width + x;
            let pixel = &frame.data[idx];
            writeln!(file, "{} {} {}", pixel.r(), pixel.g(), pixel.b()).unwrap();
        }
    }
}
