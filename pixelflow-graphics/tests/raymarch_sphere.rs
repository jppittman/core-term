//! The ray/surface composition contracts, through the packed lane: a sphere
//! over a floor, reflective and matte, against goldens.
//!
//! Named in `.github/workflows/rust.yaml` ("Check ray/surface composition
//! contracts"), so a rename here is a red check there.
//!
//! There is no march here and never was: the sphere's `t` is the quadratic's
//! near root and the floor's is a division. The file keeps its name because
//! the contracts it pins are named with it.

mod common;

use pixelflow_core::Kernel;
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
use pixelflow_graphics::scene3d::{checker, sky, Hit, Plane, Ray, Rgba, Sphere};

const W: usize = 400;
const H: usize = 300;
/// The floor these scenes stand on.
const FLOOR: f32 = -0.5;
/// The sphere, lifted so it sits above the floor rather than through it.
const CENTER: [f32; 3] = [0.0, 0.5, 4.0];
const RADIUS: f32 = 1.0;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

fn camera() -> Ray {
    Ray::through_screen(W as f32, H as f32)
}

fn sphere(ray: &Ray) -> Hit {
    Sphere::new([k(CENTER[0]), k(CENTER[1]), k(CENTER[2])], k(RADIUS)).hit(ray)
}

/// The checker floor under the sky, seen along `ray`.
fn checker_world(ray: &Ray) -> Rgba {
    let floor = Plane::at_height(k(FLOOR)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
        &sky(ray),
    )
}

/// The same floor in flat grey — no pattern, so no filtering either.
fn matte_world(ray: &Ray) -> Rgba {
    Plane::at_height(k(FLOOR))
        .hit(ray)
        .select(&Rgba::opaque_gray(0.5), &sky(ray))
}

fn render(color: &Rgba) -> Frame<Rgba8> {
    let program = compile_packed_for::<Rgba8>(color, [W as u32, H as u32]);
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    Scene::Packed(program.bind(&[])).render(&mut frame, 1);
    frame
}

/// A chrome sphere over a checker floor: the reflected ray sees the same
/// world the primary one does.
fn chrome_scene() -> Rgba {
    let ray = camera();
    let sphere = sphere(&ray);
    let mirrored = ray.reflected(sphere.normal());
    sphere.select(&checker_world(&mirrored), &checker_world(&ray))
}

#[test]
fn sphere_on_floor() {
    let frame = render(&chrome_scene());

    let path = std::env::temp_dir().join("pixelflow_raymarch_sh.ppm");
    common::write_ppm(&path, &frame).unwrap();
    common::assert_golden("sphere_on_floor", &frame, 2, 0.01);
    println!("Saved: {}", path.display());

    let center = &frame.data[(H / 2) * W + (W / 2)];
    assert!(
        center.r() > 10 || center.g() > 10 || center.b() > 10,
        "the sphere reflects something rather than nothing"
    );
}

/// The same geometry with a flat grey material: the sphere's silhouette and
/// the floor's, with no reflection and no filtering in play.
#[test]
fn sphere_on_matte_floor() {
    let ray = camera();
    let frame = render(&sphere(&ray).select(&Rgba::opaque_gray(0.5), &matte_world(&ray)));

    let path = std::env::temp_dir().join("pixelflow_raymarch_matte.ppm");
    common::write_ppm(&path, &frame).unwrap();
    common::assert_golden("sphere_on_matte_floor", &frame, 2, 0.01);
    println!("Saved: {}", path.display());

    let center = &frame.data[(H / 2) * W + (W / 2)];
    assert!(
        center.r() > 100 && center.r() < 150,
        "the centre is the matte sphere: r={}",
        center.r()
    );
}

/// The chrome sphere over the checkerboard — the same scene
/// [`sphere_on_floor`] renders, as it was before this file moved to kernels;
/// both names are public contracts, so both stay.
#[test]
fn chrome_sphere_on_checkerboard() {
    let frame = render(&chrome_scene());

    let path = std::env::temp_dir().join("pixelflow_chrome_checker.ppm");
    common::write_ppm(&path, &frame).unwrap();
    common::assert_golden("chrome_sphere_on_checkerboard", &frame, 2, 0.01);
    println!("Saved: {}", path.display());

    let center = &frame.data[(H / 2) * W + (W / 2)];
    let bottom = &frame.data[(H * 3 / 4) * W + (W / 2)];
    let top = &frame.data[(H / 4) * W + (W / 2)];
    println!(
        "sphere r={} g={} b={}; floor r={}; sky r={} b={}",
        center.r(),
        center.g(),
        center.b(),
        bottom.r(),
        top.r(),
        top.b()
    );
    assert!(
        center.r() > 10 || center.g() > 10 || center.b() > 10,
        "the centre hits the chrome sphere"
    );
    assert!(
        bottom.r() > 10 || bottom.g() > 10 || bottom.b() > 10,
        "the bottom hits the floor"
    );
    assert!(top.b() > top.r(), "the top is sky");
}
