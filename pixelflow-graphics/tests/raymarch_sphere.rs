//! Test: Render a 3D sphere using Jet2-based raymarching

use pixelflow_graphics::raymarch::{MarchConfig, Raymarch};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, TensorShape};
use pixelflow_core::{Field, Jet2, Manifold};
use std::fs::File;
use std::io::Write;

/// Sphere as implicit surface: x² + y² + z² - r² = 0
#[derive(Clone, Copy)]
struct Sphere {
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
}

impl Manifold<Jet2> for Sphere {
    type Output = Jet2;

    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, _w: Jet2) -> Jet2 {
        let dx = x - Jet2::constant(Field::from(self.cx));
        let dy = y - Jet2::constant(Field::from(self.cy));
        let dz = z - Jet2::constant(Field::from(self.cz));
        let r2 = Jet2::constant(Field::from(self.radius * self.radius));
        dx * dx + dy * dy + dz * dz - r2
    }
}

#[test]
fn test_sphere_implicit() {
    // Test the implicit surface directly
    let sphere = Sphere {
        cx: 0.0,
        cy: 0.0,
        cz: 0.0,
        radius: 2.0,
    };

    // At (0, 0, -5): 0 + 0 + 25 - 4 = 21 (positive, outside)
    let v1 = sphere.eval_raw(
        Jet2::x(Field::from(0.0)),
        Jet2::y(Field::from(0.0)),
        Jet2::constant(Field::from(-5.0)),
        Jet2::constant(Field::from(0.0)),
    );
    println!("f(0,0,-5) = {:?} (expect ~21)", v1.val);

    // At (0, 0, -2): 0 + 0 + 4 - 4 = 0 (on surface)
    let v2 = sphere.eval_raw(
        Jet2::x(Field::from(0.0)),
        Jet2::y(Field::from(0.0)),
        Jet2::constant(Field::from(-2.0)),
        Jet2::constant(Field::from(0.0)),
    );
    println!("f(0,0,-2) = {:?} (expect ~0)", v2.val);

    // At (0, 0, 0): 0 + 0 + 0 - 4 = -4 (negative, inside)
    let v3 = sphere.eval_raw(
        Jet2::x(Field::from(0.0)),
        Jet2::y(Field::from(0.0)),
        Jet2::constant(Field::from(0.0)),
        Jet2::constant(Field::from(0.0)),
    );
    println!("f(0,0,0) = {:?} (expect ~-4)", v3.val);
}

#[test]
fn test_raymarch_sphere() {
    const WIDTH: usize = 200;
    const HEIGHT: usize = 200;

    let sphere = Sphere {
        cx: 0.0,
        cy: 0.0,
        cz: 2.0,  // Move sphere forward
        radius: 1.2,
    };

    let raymarch = Raymarch {
        scene: sphere,
        config: MarchConfig {
            max_steps: 200,
            step_size: 0.02,
            max_distance: 20.0,
            camera_z: -2.0,  // Move camera closer
        },
    };


    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);

    // Map from pixel coords to normalized [-1, 1]
    struct PixelToNorm<M> {
        inner: M,
        width: f32,
        height: f32,
    }

    impl<M> Manifold for PixelToNorm<M>
    where
        M: Manifold<Output = pixelflow_core::Discrete>,
    {
        type Output = pixelflow_core::Discrete;

        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Self::Output {
            let nx = (x / Field::from(self.width)) * Field::from(2.0) - Field::from(1.0);
            let ny = (y / Field::from(self.height)) * Field::from(2.0) - Field::from(1.0);
            self.inner.eval_raw(nx, ny, z, w)
        }
    }

    let pixel_raymarch = PixelToNorm {
        inner: raymarch,
        width: WIDTH as f32,
        height: HEIGHT as f32,
    };

    execute(&pixel_raymarch, frame.as_slice_mut(), TensorShape {
        width: WIDTH,
        height: HEIGHT,
    });

    // Write PPM
    let path = std::env::temp_dir().join("pixelflow_raymarch_sphere.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", WIDTH, HEIGHT).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Verify hit - center pixel should be sphere surface (not background)
    let center_px = &frame.data[(HEIGHT / 2) * WIDTH + (WIDTH / 2)];
    println!("Center: ({}, {}, {})", center_px.r(), center_px.g(), center_px.b());

    // Background is (20, 20, 40), sphere surface should be different (normal-mapped)
    assert_ne!(center_px.r(), 20, "Center should hit sphere, not background");
    assert_ne!(center_px.g(), 20, "Center should hit sphere, not background");
}
