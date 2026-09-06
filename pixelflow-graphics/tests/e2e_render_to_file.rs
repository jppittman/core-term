//! End-to-end: compose a scene, render it into a frame, write it to a file.
//!
//! Every scene here is what a scene now is — four channel kernels in `[0, 1]`
//! compiled at the frame's lattice shape, packed to the frame's own byte order
//! inside the kernel — so the pipeline this exercises is the one production
//! takes. The goldens are the same pictures the per-batch rasterizer drew.

mod common;

use common::{assert_golden, write_ppm};
use pixelflow_core::{Field, Kernel, ManifoldCompat};
use pixelflow_graphics::render::color::{Color, NamedColor, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::scene::{compile_packed_for, constant_scene_for, Scene};
use pixelflow_graphics::scene3d::Rgba;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// R = G = B = `value`, opaque — the grayscale lift, as four channels.
fn grayscale(value: &Kernel) -> Rgba {
    Rgba::from([value.clone(), value.clone(), value.clone(), k(1.0)])
}

/// Compile `color` at `[w, h]` and render it single-threaded.
fn render(color: &Rgba, w: u32, h: u32) -> Frame<Rgba8> {
    let mut frame = Frame::<Rgba8>::new(w, h);
    Scene::Packed(compile_packed_for::<Rgba8>(color, [w, h]).bind(&[])).render(&mut frame, 1);
    frame
}

/// Write the frame somewhere a human can look at it after a failure.
fn save(name: &str, frame: &Frame<Rgba8>) {
    let path = std::env::temp_dir().join(format!("pixelflow_{name}.ppm"));
    write_ppm(&path, frame).expect("Failed to write PPM file");
    println!("{name} saved to: {}", path.display());
}

#[test]
fn e2e_render_gradient() {
    const WIDTH: u32 = 400;
    const HEIGHT: u32 = 300;

    // Red ramps across, green down, blue is what the other two leave.
    let r = Kernel::x().div(&k(WIDTH as f32));
    let g = Kernel::y().div(&k(HEIGHT as f32));
    let b = k(1.0).sub(&r).add(&k(1.0).sub(&g)).div(&k(2.0));
    let frame = render(&Rgba::from([r, g, b, k(1.0)]), WIDTH, HEIGHT);

    // Top-left is dark in red and green, bright in blue.
    let top_left = &frame.data[0];
    assert!(
        top_left.r() < 50,
        "Top-left red should be low, got {}",
        top_left.r()
    );
    assert!(
        top_left.b() > 200,
        "Top-left blue should be high, got {}",
        top_left.b()
    );

    // Bottom-right is the opposite.
    let bottom_right = &frame.data[frame.data.len() - 1];
    assert!(
        bottom_right.r() > 200,
        "Bottom-right red should be high, got {}",
        bottom_right.r()
    );
    assert!(
        bottom_right.b() < 50,
        "Bottom-right blue should be low, got {}",
        bottom_right.b()
    );

    save("e2e_gradient", &frame);
    assert_golden("e2e_render_gradient", &frame, 2, 0.01);
}

/// A radial gradient: 1.0 at the centre, 0.0 at `radius`, parabolic between.
fn radial_gradient(cx: f32, cy: f32, radius_sq: f32) -> Kernel {
    let dx = Kernel::x().sub(&k(cx));
    let dy = Kernel::y().sub(&k(cy));
    let dist_sq = dx.mul(&dx).add(&dy.mul(&dy));
    k(1.0).sub(&dist_sq.div(&k(radius_sq)))
}

#[test]
fn e2e_render_radial_gradient() {
    const SIZE: u32 = 200;
    let half = SIZE as f32 / 2.0;

    let frame = render(
        &grayscale(&radial_gradient(half, half, half * half)),
        SIZE,
        SIZE,
    );

    let center_idx = (SIZE / 2) as usize * SIZE as usize + (SIZE / 2) as usize;
    let center = &frame.data[center_idx];
    assert!(
        center.r() > 200,
        "Center should be bright, got r={}",
        center.r()
    );
    assert_eq!(center.r(), center.g(), "Grayscale: R should equal G");
    assert_eq!(center.g(), center.b(), "Grayscale: G should equal B");

    // Outside the radius the value is negative, and the pack clamps it.
    let corner = &frame.data[0];
    assert_eq!(
        corner.r(),
        0,
        "Corner should be black (clamped), got r={}",
        corner.r()
    );

    save("e2e_radial", &frame);
    assert_golden("e2e_render_radial_gradient", &frame, 2, 0.01);
}

#[test]
fn e2e_render_circle() {
    const SIZE: u32 = 100;

    // The unit circle, scaled to `radius` and moved to the frame's centre —
    // which is precomposition on the coordinates, in the language.
    let radius = SIZE as f32 / 2.0 - 5.0;
    let center = SIZE as f32 / 2.0;
    let u = Kernel::x().sub(&k(center)).div(&k(radius));
    let v = Kernel::y().sub(&k(center)).div(&k(radius));
    let inside = u.mul(&u).add(&v.mul(&v)).lt(&k(1.0));
    let frame = render(&grayscale(&inside.select(&k(1.0), &k(0.0))), SIZE, SIZE);

    let center_idx = (SIZE / 2) as usize * SIZE as usize + (SIZE / 2) as usize;
    assert_eq!(
        frame.data[center_idx].r(),
        255,
        "Center should be white (inside circle)"
    );
    assert_eq!(
        frame.data[0].r(),
        0,
        "Corner should be black (outside circle)"
    );

    save("e2e_circle", &frame);
    assert_golden("e2e_render_circle", &frame, 2, 0.01);
}

#[test]
fn e2e_solid_color_renders_correctly() {
    const SIZE: u32 = 50;

    let (r, g, b, a) = Color::Named(NamedColor::BrightCyan).to_f32_rgba();
    let mut frame = Frame::<Rgba8>::new(SIZE, SIZE);
    constant_scene_for::<Rgba8>([r, g, b, a], [SIZE, SIZE]).render(&mut frame, 1);

    for (i, pixel) in frame.data.iter().enumerate() {
        assert_eq!(pixel.r(), 0, "Pixel {i} red should be 0");
        assert_eq!(pixel.g(), 255, "Pixel {i} green should be 255");
        assert_eq!(pixel.b(), 255, "Pixel {i} blue should be 255");
        assert_eq!(pixel.a(), 255, "Pixel {i} alpha should be 255");
    }

    save("e2e_cyan", &frame);
    assert_golden("e2e_solid_color_renders_correctly", &frame, 2, 0.0);
}

/// The `shapes` module is the ZST combinator tier, which S4b retires. Until
/// then this is a smoke test that it still composes; nothing renders through
/// it.
#[test]
fn e2e_render_using_builtin_shapes() {
    use pixelflow_graphics::shapes::{circle, EMPTY, SOLID};

    let unit_circle = circle(SOLID, EMPTY);
    let at = |x: f32| {
        unit_circle.eval_raw(
            Field::from(x),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        )
    };
    let _at_origin = at(0.0);
    let _outside = at(2.0); // x² = 4 > 1
}

/// A `Frame` starts empty and ends as whatever was rendered into it.
#[test]
fn e2e_frame_operations() {
    const SIZE: u32 = 10;

    let mut frame = Frame::<Rgba8>::new(SIZE, SIZE);
    assert_eq!(frame.width, SIZE as usize);
    assert_eq!(frame.height, SIZE as usize);
    assert_eq!(frame.data.len(), (SIZE * SIZE) as usize);
    for pixel in &frame.data {
        assert_eq!((pixel.r(), pixel.g(), pixel.b(), pixel.a()), (0, 0, 0, 0));
    }

    let (r, g, b, a) = Color::Named(NamedColor::Red).to_f32_rgba();
    constant_scene_for::<Rgba8>([r, g, b, a], [SIZE, SIZE]).render(&mut frame, 1);
    for pixel in &frame.data {
        assert_eq!(
            (pixel.r(), pixel.g(), pixel.b(), pixel.a()),
            (205, 0, 0, 255)
        );
    }

    // 4 bytes per pixel.
    assert_eq!(frame.as_bytes().len(), (SIZE * SIZE * 4) as usize);
}
