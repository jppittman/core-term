//! Subdivision surface rendering with automatic differentiation.
//!
//! Demonstrates forward-mode AD for surface normals:
//! - Generates a curved quad mesh procedurally
//! - Evaluates Catmull-Clark limit surface with Jet3
//! - Normals emerge from cross(dP/du, dP/dv) automatically
//! - No finite differences, no extra evaluations

use pixelflow_graphics::mesh::{Point3, QuadMesh};
use pixelflow_graphics::render::rasterizer::{rasterize, RenderOptions};
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface,
};
use pixelflow_graphics::subdivision::{SubdivisionGeometry, SubdivisionPatch};
use pixelflow_graphics::render::color::RgbaColorCube;
use pixelflow_graphics::{Frame, Rgba8};

fn main() {
    println!("=== Subdivision Surface Autodiff Demo ===\n");

    // Create a simple curved quad mesh
    let mesh = create_curved_quad();
    println!("Generated mesh:");
    println!("  Vertices: {}", mesh.vertex_count());
    println!("  Faces: {}", mesh.face_count());

    // Create subdivision geometry from first patch
    let patch = SubdivisionPatch::from_mesh(&mesh, 0).unwrap();
    println!("\nPatch info:");
    println!("  Corners: {:?}", patch.corners);
    println!("  Valences: {:?}", patch.corner_valences);
    println!("  Extraordinary: {}", patch.is_extraordinary());

    let geometry = SubdivisionGeometry::new(
        patch, &mesh, -1.0, // base_height (intersection plane)
        1.0,  // uv_scale
        0.0,  // center_x
        0.0,  // center_z
    );

    // Build scene: subdivision surface with checker floor and sky
    let floor = ColorSurface {
        geometry: pixelflow_graphics::scene3d::PlaneGeometry { height: -2.0 },
        material: ColorChecker::<RgbaColorCube>::default(),
        background: ColorSky::<RgbaColorCube>::default(),
    };

    let surface = ColorSurface {
        geometry,
        material: ColorReflect { inner: floor },
        background: ColorSky::<RgbaColorCube>::default(),
    };

    let scene = ColorScreenToDir { inner: surface };

    // Render
    let width = 800;
    let height = 600;
    println!("\nRendering {}x{} image...", width, height);

    let mut frame = Frame::<Rgba8>::new(width, height);
    rasterize(&scene, &mut frame, RenderOptions { num_threads: 1 });

    // Save to PPM (simple format, no dependencies)
    save_ppm("subdivision_autodiff.ppm", &frame);

    println!("\n✓ Rendered to subdivision_autodiff.ppm");
    println!("\nKey insight: The smooth normals came from Jet3 autodiff.");
    println!("No finite differences. No extra evaluations. Just algebra.");
}

/// Create a simple curved quad mesh for testing.
///
/// Returns a single quad with corners lifted to form a saddle surface.
fn create_curved_quad() -> QuadMesh {
    // Four corners of a quad, with Z displacement for curvature
    let vertices = vec![
        Point3::new(-1.0, 0.0, -1.0), // Bottom-left
        Point3::new(1.0, 0.0, 1.0),   // Bottom-right (lifted)
        Point3::new(1.0, 0.0, -1.0),  // Top-right
        Point3::new(-1.0, 0.0, 1.0),  // Top-left (lifted)
    ];

    let faces = vec![pixelflow_graphics::mesh::Quad::new(0, 1, 2, 3)];

    // Valence computation
    let valence = vec![1, 1, 1, 1];

    QuadMesh {
        vertices,
        faces,
        valence,
    }
}

/// Save frame to PPM format (simple, no external dependencies).
fn save_ppm(path: &str, frame: &Frame<Rgba8>) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create output file");

    // PPM header
    writeln!(file, "P3").unwrap();
    writeln!(file, "{} {}", frame.width, frame.height).unwrap();
    writeln!(file, "255").unwrap();

    // Pixel data (row-major)
    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = y * frame.width + x;
            let pixel = &frame.data[idx];
            writeln!(file, "{} {} {}", pixel.r(), pixel.g(), pixel.b()).unwrap();
        }
    }

    println!("Saved to {}", path);
}
