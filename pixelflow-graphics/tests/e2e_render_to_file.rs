//! End-to-end test: render a scene and write it to a file.
//!
//! This test verifies the full pipeline from manifold composition
//! through rasterization to file output.

use pixelflow_graphics::render::color::{Lift, NamedColor, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, TensorShape};
use pixelflow_graphics::transform::{Scale, Translate};
use pixelflow_graphics::{Discrete, Field, Manifold, ManifoldExt, X, Y};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Write a frame to a PPM file (simple image format, no dependencies needed).
fn write_ppm<P: AsRef<Path>>(path: P, frame: &Frame<Rgba8>) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // PPM header: P6 means binary RGB
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", frame.width, frame.height)?;
    writeln!(file, "255")?;

    // Write RGB bytes (skip alpha)
    for pixel in &frame.data {
        file.write_all(&[pixel.r(), pixel.g(), pixel.b()])?;
    }

    Ok(())
}

/// A colorful gradient manifold that outputs Discrete pixels.
/// Creates a smooth color transition based on x and y coordinates.
#[derive(Clone, Copy)]
struct Gradient {
    width: f32,
    height: f32,
}

impl Manifold for Gradient {
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
        // Normalize coordinates to [0, 1]
        let r = x / Field::from(self.width);
        let g = y / Field::from(self.height);
        let b = (Field::from(1.0) - r + Field::from(1.0) - g) / Field::from(2.0);

        Discrete::pack(r, g, b, Field::from(1.0))
    }
}

#[test]
fn e2e_render_gradient() {
    const WIDTH: u32 = 400;
    const HEIGHT: u32 = 300;

    let scene = Gradient {
        width: WIDTH as f32,
        height: HEIGHT as f32,
    };

    let mut frame = Frame::<Rgba8>::new(WIDTH, HEIGHT);
    let shape = TensorShape::new(WIDTH as usize, HEIGHT as usize);

    execute(&scene, frame.as_slice_mut(), shape);

    // Verify some pixels
    // Top-left should be dark (low R, low G, high B due to gradient formula)
    let top_left = &frame.data[0];
    assert!(
        top_left.r() < 50,
        "Top-left red should be low, got {}",
        top_left.r()
    );
    assert!(
        top_left.g() < 50,
        "Top-left green should be low, got {}",
        top_left.g()
    );

    // Bottom-right should have high R and G
    let bottom_right = &frame.data[(HEIGHT - 1) as usize * WIDTH as usize + (WIDTH - 1) as usize];
    assert!(
        bottom_right.r() > 200,
        "Bottom-right red should be high, got {}",
        bottom_right.r()
    );
    assert!(
        bottom_right.g() > 200,
        "Bottom-right green should be high, got {}",
        bottom_right.g()
    );

    // Write to file for visual inspection
    let output_path = std::env::temp_dir().join("pixelflow_e2e_gradient.ppm");
    write_ppm(&output_path, &frame).expect("Failed to write PPM file");

    println!("Gradient image saved to: {}", output_path.display());

    // Verify file was written
    assert!(output_path.exists(), "Output file should exist");
}

/// A radial gradient from center (1.0) to edge (0.0).
/// Uses parabolic falloff (simpler than true radial).
#[derive(Clone, Copy)]
struct RadialGradient {
    cx: f32,
    cy: f32,
    radius_sq: f32,
}

impl Manifold for RadialGradient {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        let dx = x - Field::from(self.cx);
        let dy = y - Field::from(self.cy);
        let dist_sq = dx * dx + dy * dy;

        // 1.0 at center, 0.0 at edge (parabolic falloff)
        // Values outside radius will go negative, but Pixel::from_rgba clamps
        let normalized_sq = dist_sq / Field::from(self.radius_sq);
        Field::from(1.0) - normalized_sq
    }
}

#[test]
fn e2e_render_radial_gradient() {
    const SIZE: u32 = 200;

    // Use Lift to convert a scalar field to grayscale
    let radial = Lift(RadialGradient {
        cx: SIZE as f32 / 2.0,
        cy: SIZE as f32 / 2.0,
        radius_sq: (SIZE as f32 / 2.0) * (SIZE as f32 / 2.0),
    });

    let mut frame = Frame::<Rgba8>::new(SIZE, SIZE);
    let shape = TensorShape::new(SIZE as usize, SIZE as usize);

    execute(&radial, frame.as_slice_mut(), shape);

    // Center should be bright (close to white)
    let center_idx = (SIZE / 2) as usize * SIZE as usize + (SIZE / 2) as usize;
    let center = &frame.data[center_idx];
    assert!(
        center.r() > 200,
        "Center should be bright, got r={}",
        center.r()
    );
    assert_eq!(center.r(), center.g(), "Grayscale: R should equal G");
    assert_eq!(center.g(), center.b(), "Grayscale: G should equal B");

    // Corner should be dark (outside the radius, negative values clamped to 0)
    let corner = &frame.data[0];
    assert!(
        corner.r() == 0,
        "Corner should be black (clamped), got r={}",
        corner.r()
    );

    let output_path = std::env::temp_dir().join("pixelflow_e2e_radial.ppm");
    write_ppm(&output_path, &frame).expect("Failed to write PPM file");
    println!("Radial gradient saved to: {}", output_path.display());
}

/// A unit circle manifold (returns 1.0 inside, 0.0 outside).
/// Uses proper manifold composition with ManifoldExt.
#[derive(Clone, Copy)]
struct UnitCircle;

impl Manifold for UnitCircle {
    type Output = Field;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        // Build the manifold expression: x² + y² < 1 ? 1.0 : 0.0
        // Using ManifoldExt's lt() and select()
        let dist_sq = X * X + Y * Y;
        let mask = dist_sq.lt(1.0f32);
        let result = mask.select(1.0f32, 0.0f32);

        // Evaluate the composed manifold at the given coordinates
        result.eval_raw(x, y, z, w)
    }
}

#[test]
fn e2e_render_circle() {
    const SIZE: u32 = 100;

    // Unit circle at origin, scaled and translated to center of image
    let radius = SIZE as f32 / 2.0 - 5.0;
    let scaled = Scale {
        manifold: UnitCircle,
        factor: radius,
    };
    let centered = Translate {
        manifold: scaled,
        offset: [SIZE as f32 / 2.0, SIZE as f32 / 2.0],
    };

    // Lift to grayscale
    let scene = Lift(centered);

    let mut frame = Frame::<Rgba8>::new(SIZE, SIZE);
    let shape = TensorShape::new(SIZE as usize, SIZE as usize);

    execute(&scene, frame.as_slice_mut(), shape);

    // Center should be white (inside circle = 1.0)
    let center_idx = (SIZE / 2) as usize * SIZE as usize + (SIZE / 2) as usize;
    let center = &frame.data[center_idx];
    assert_eq!(
        center.r(),
        255,
        "Center should be white (inside circle), got {}",
        center.r()
    );

    // Corner should be black (outside circle = 0.0)
    let corner = &frame.data[0];
    assert_eq!(
        corner.r(),
        0,
        "Corner should be black (outside circle), got {}",
        corner.r()
    );

    let output_path = std::env::temp_dir().join("pixelflow_e2e_circle.ppm");
    write_ppm(&output_path, &frame).expect("Failed to write PPM file");
    println!("Circle image saved to: {}", output_path.display());
}

#[test]
fn e2e_solid_color_renders_correctly() {
    // Simplest possible test: render a solid color
    const SIZE: u32 = 50;

    let cyan = NamedColor::BrightCyan;

    let mut frame = Frame::<Rgba8>::new(SIZE, SIZE);
    let shape = TensorShape::new(SIZE as usize, SIZE as usize);

    execute(&cyan, frame.as_slice_mut(), shape);

    // Every pixel should be bright cyan (0, 255, 255)
    for (i, pixel) in frame.data.iter().enumerate() {
        assert_eq!(pixel.r(), 0, "Pixel {} red should be 0", i);
        assert_eq!(pixel.g(), 255, "Pixel {} green should be 255", i);
        assert_eq!(pixel.b(), 255, "Pixel {} blue should be 255", i);
        assert_eq!(pixel.a(), 255, "Pixel {} alpha should be 255", i);
    }

    let output_path = std::env::temp_dir().join("pixelflow_e2e_cyan.ppm");
    write_ppm(&output_path, &frame).expect("Failed to write PPM file");
    println!("Solid cyan image saved to: {}", output_path.display());
}

/// Test using the built-in shapes module
#[test]
fn e2e_render_using_builtin_shapes() {
    use pixelflow_graphics::shapes::{circle, EMPTY, SOLID};

    // Create a circle using the shapes module
    // The shapes::circle returns impl Manifold<Output=Field>
    let unit_circle = circle(SOLID, EMPTY);

    // Evaluate the circle at the origin - should return SOLID (1.0)
    let _at_origin = unit_circle.eval_raw(
        Field::from(0.0),
        Field::from(0.0),
        Field::from(0.0),
        Field::from(0.0),
    );

    // Evaluate outside the circle - should return EMPTY (0.0)
    let _outside = unit_circle.eval_raw(
        Field::from(2.0), // outside unit circle (x² = 4 > 1)
        Field::from(0.0),
        Field::from(0.0),
        Field::from(0.0),
    );

    // This is a smoke test that shapes compile and the API works
    // The actual pixel rendering is tested in e2e_render_circle
    println!("Built-in shapes module works! Circle evaluates at origin and outside.");
}

/// Test that Frame operations work correctly
#[test]
fn e2e_frame_operations() {
    const SIZE: u32 = 10;

    let mut frame = Frame::<Rgba8>::new(SIZE, SIZE);

    // Check initial state
    assert_eq!(frame.width, SIZE as usize);
    assert_eq!(frame.height, SIZE as usize);
    assert_eq!(frame.data.len(), (SIZE * SIZE) as usize);

    // All pixels should be default (black/transparent)
    for pixel in &frame.data {
        assert_eq!(pixel.r(), 0);
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 0);
        assert_eq!(pixel.a(), 0);
    }

    // Render something
    execute(&NamedColor::Red, frame.as_slice_mut(), TensorShape::new(SIZE as usize, SIZE as usize));

    // Now all should be red
    for pixel in &frame.data {
        assert_eq!(pixel.r(), 205); // ANSI Red
        assert_eq!(pixel.g(), 0);
        assert_eq!(pixel.b(), 0);
        assert_eq!(pixel.a(), 255);
    }

    // Test as_bytes
    let bytes = frame.as_bytes();
    assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize); // 4 bytes per pixel

    println!("Frame operations work correctly!");
}
