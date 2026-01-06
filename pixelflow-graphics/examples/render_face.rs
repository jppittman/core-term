//! Render a limit surface face using Stam's analytic method.
//!
//! Demonstrates:
//! - Jos Stam's eigenstructure method for Catmull-Clark subdivision
//! - Analytic Hilbert space evaluation (bicubic coefficients = eigenbasis coordinates)
//! - Multi-patch surface rendering via height field
//! - Pure manifold composition - no baking, no Newton iteration
//!
//! Run with: cargo run --release --example render_face -p pixelflow-graphics

use pixelflow_graphics::render::{execute, TensorShape};
use pixelflow_graphics::scene3d::{ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface};
use pixelflow_graphics::subdiv::{build_face_surface, LimitSurfaceGeometry};
use pixelflow_graphics::{Frame, Rgba8};

fn main() {
    println!("=== Limit Surface Face Rendering ===\n");
    println!("Using Jos Stam's eigenstructure method for Catmull-Clark subdivision.");
    println!("Analytic evaluation in Hilbert space - bicubic coefficients ARE eigenbasis coordinates.\n");

    // Build the limit surface from the face mesh
    let surface = build_face_surface();
    println!("Built limit surface:");
    println!("  Patches: {}", surface.patches.len());
    println!("  Bounds: ({:.2}, {:.2}, {:.2}) to ({:.2}, {:.2}, {:.2})",
        surface.bounds.min[0], surface.bounds.min[1], surface.bounds.min[2],
        surface.bounds.max[0], surface.bounds.max[1], surface.bounds.max[2]);

    // Create geometry with transform - no baking, pure analytic evaluation
    let geometry = LimitSurfaceGeometry::new(
        surface,
        2.0,              // scale
        [0.0, 0.0, -4.0], // offset: in front of camera, centered
    );

    println!("  Evaluation: analytic bicubic (16 coeffs per subpatch)");

    // Build scene: face with checker floor and sky background
    let floor = ColorSurface {
        geometry: pixelflow_graphics::scene3d::PlaneGeometry { height: -2.0 },
        material: ColorChecker,
        background: ColorSky,
    };

    let face_surface = ColorSurface {
        geometry,
        material: ColorReflect { inner: floor },
        background: ColorSky,
    };

    let scene = ColorScreenToDir { inner: face_surface };

    // Render
    let width = 800;
    let height = 600;
    println!("\nRendering {}x{} image...", width, height);

    let mut frame = Frame::<Rgba8>::new(width, height);
    execute(
        &scene,
        frame.as_slice_mut(),
        TensorShape::new(width as usize, height as usize),
    );

    // Save to PPM
    save_ppm("limit_face.ppm", &frame);

    println!("\nRendered to limit_face.ppm");
    println!("\nThe smooth surface comes from Stam's Hilbert space decomposition:");
    println!("  - Bicubic coefficients ARE coordinates in the orthogonal eigenbasis");
    println!("  - Each basis function is a bicubic polynomial (16 coeffs)");
    println!("  - Evaluation is just polynomial arithmetic - no iteration");
    println!("  - SIMD-parallel via Field operations on all lanes simultaneously");
}

/// Save frame to PPM format.
fn save_ppm(path: &str, frame: &Frame<Rgba8>) {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path).expect("Failed to create output file");

    // PPM header
    writeln!(file, "P3").unwrap();
    writeln!(file, "{} {}", frame.width, frame.height).unwrap();
    writeln!(file, "255").unwrap();

    // Pixel data
    for y in 0..frame.height {
        for x in 0..frame.width {
            let idx = y * frame.width + x;
            let pixel = &frame.data[idx];
            writeln!(file, "{} {} {}", pixel.r(), pixel.g(), pixel.b()).unwrap();
        }
    }

    println!("Saved to {}", path);
}
