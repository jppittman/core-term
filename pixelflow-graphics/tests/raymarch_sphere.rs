//! Test: Algebraic raymarching with sphere + floor using coherent optics model

use pixelflow_core::{Discrete, Field, Gt, Manifold, Min, Select, W};
use pixelflow_graphics::raymarch::{
    chrome, matte, CheckerMask, Environment, MarchConfig, Raymarch, Response, Scale, Translate,
    GroundPlane, UnitSphere,
};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, TensorShape};
use std::fs::File;
use std::io::Write;

/// Remap pixel coordinates to normalized screen coordinates.
/// Transforms [0, width] × [0, height] → [-aspect, aspect] × [-1, 1]
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: Manifold<Output = Discrete>> Manifold for ScreenRemap<M> {
    type Output = Discrete;

    fn eval_raw(&self, px: Field, py: Field, z: Field, w: Field) -> Discrete {
        let width = Field::from(self.width);
        let height = Field::from(self.height);
        let aspect = width / height;

        // Map pixel coords to normalized: [-aspect, aspect] × [1, -1] (flip y)
        let x = (px / width - Field::from(0.5)) * Field::from(2.0) * aspect;
        let y = (Field::from(0.5) - py / height) * Field::from(2.0);

        self.inner.eval_raw(x, y, z, w)
    }
}

#[test]
fn test_sphere_on_floor() {
    const W: usize = 400;
    const H: usize = 300;

    // Scene: sphere + floor via manifold composition
    let sphere = Translate {
        inner: Scale {
            inner: UnitSphere,
            factor: 1.0,
        },
        offset: (0.0, 0.5, 4.0),
    };

    let floor = Translate {
        inner: GroundPlane,
        offset: (0.0, -0.5, 0.0),
    };

    let scene = Min(sphere, floor);

    // Chrome material (will reflect environment)
    let material = chrome();

    // Environment lighting via spherical harmonics
    let env = Environment::sky_gradient();

    // Raymarch in normalized coords, wrapped with pixel→normalized transform
    let raymarch = Raymarch {
        scene,
        material,
        env,
        config: MarchConfig {
            max_step: 0.1,
            max_t: 20.0,
            epsilon: 0.001,
        },
    };

    let screen = ScreenRemap {
        inner: raymarch,
        width: W as f32,
        height: H as f32,
    };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    execute(
        &screen,
        frame.as_slice_mut(),
        TensorShape {
            width: W,
            height: H,
        },
    );

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_raymarch_sh.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Basic sanity: center should hit something (not pure sky blue)
    let center = &frame.data[(H / 2) * W + (W / 2)];
    assert!(
        center.r() != 128 || center.g() != 179 || center.b() != 255,
        "Center pixel should hit geometry, not sky"
    );
}

/// Test with matte gray material
#[test]
fn test_sphere_on_matte_floor() {
    const W: usize = 400;
    const H: usize = 300;

    let sphere = Translate {
        inner: Scale {
            inner: UnitSphere,
            factor: 1.0,
        },
        offset: (0.0, 0.5, 4.0),
    };

    let floor = Translate {
        inner: GroundPlane,
        offset: (0.0, -0.5, 0.0),
    };

    let scene = Min(sphere, floor);

    // Matte gray material (pure diffuse)
    let material = matte(0.5, 0.5, 0.5);

    let env = Environment::sky_gradient();

    let raymarch = Raymarch {
        scene,
        material,
        env,
        config: MarchConfig {
            max_step: 0.1,
            max_t: 20.0,
            epsilon: 0.001,
        },
    };

    let screen = ScreenRemap {
        inner: raymarch,
        width: W as f32,
        height: H as f32,
    };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    execute(
        &screen,
        frame.as_slice_mut(),
        TensorShape {
            width: W,
            height: H,
        },
    );

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_raymarch_matte.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Center should hit geometry
    let center = &frame.data[(H / 2) * W + (W / 2)];
    assert!(
        center.r() != 128 || center.g() != 179 || center.b() != 255,
        "Center pixel should hit geometry, not sky"
    );
}

/// Chrome sphere on checkerboard floor - material selection via manifold composition
#[test]
fn test_chrome_sphere_on_checkerboard() {
    const WIDTH: usize = 400;
    const HEIGHT: usize = 300;

    let sphere = Translate {
        inner: Scale {
            inner: UnitSphere,
            factor: 1.0,
        },
        offset: (0.0, 0.5, 4.0),
    };

    let floor = Translate {
        inner: GroundPlane,
        offset: (0.0, -0.5, 0.0),
    };

    let scene = Min(sphere, floor);

    // Condition: w > 0.9 means floor (normal.y ≈ 1.0), else sphere
    // w coordinate receives ny from the raymarch shading
    let is_floor = Gt(W, 0.9f32);

    // Checker pattern for the floor's diffuse
    let checker = CheckerMask { scale: 1.0 };

    // Material: Response with Select on each coefficient
    // When is_floor: checkerboard matte (absorption=0.1, reflection=0.0, scatter varies)
    // When sphere: chrome (absorption=0.0, reflection=0.95, scatter=0.05)
    //
    // Select(cond, if_true, if_false) - so is_floor=true → floor values
    let material = Response::new(
        // absorption: floor=0.1, sphere=0.0
        Select { cond: is_floor, if_true: 0.1f32, if_false: 0.0f32 },
        // reflection: floor=0.0, sphere=0.95
        Select { cond: is_floor, if_true: 0.0f32, if_false: 0.95f32 },
        // scatter_r: floor=checker pattern, sphere=0.05
        Select {
            cond: is_floor,
            if_true: Select { cond: Gt(checker, 0.5f32), if_true: 0.7f32, if_false: 0.3f32 },
            if_false: 0.05f32,
        },
        // scatter_g: same pattern
        Select {
            cond: is_floor,
            if_true: Select { cond: Gt(checker, 0.5f32), if_true: 0.7f32, if_false: 0.3f32 },
            if_false: 0.05f32,
        },
        // scatter_b: same pattern
        Select {
            cond: is_floor,
            if_true: Select { cond: Gt(checker, 0.5f32), if_true: 0.7f32, if_false: 0.3f32 },
            if_false: 0.05f32,
        },
    );

    let env = Environment::sky_gradient();

    let raymarch = Raymarch {
        scene,
        material,
        env,
        config: MarchConfig {
            max_step: 0.1,
            max_t: 20.0,
            epsilon: 0.001,
        },
    };

    let screen = ScreenRemap {
        inner: raymarch,
        width: WIDTH as f32,
        height: HEIGHT as f32,
    };

    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    execute(
        &screen,
        frame.as_slice_mut(),
        TensorShape {
            width: WIDTH,
            height: HEIGHT,
        },
    );

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_chrome_checker.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", WIDTH, HEIGHT).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Debug: print some pixel values
    let center = &frame.data[(HEIGHT / 2) * WIDTH + (WIDTH / 2)];
    println!("Center pixel (sphere): r={} g={} b={}", center.r(), center.g(), center.b());
    let bottom = &frame.data[(HEIGHT * 3 / 4) * WIDTH + (WIDTH / 2)];
    println!("Bottom pixel (floor): r={} g={} b={}", bottom.r(), bottom.g(), bottom.b());
    let top = &frame.data[(HEIGHT / 4) * WIDTH + (WIDTH / 2)];
    println!("Top pixel (sky): r={} g={} b={}", top.r(), top.g(), top.b());

    // Verify: center should be chrome sphere (reflective, grayish from sky)
    let center = &frame.data[(HEIGHT / 2) * WIDTH + (WIDTH / 2)];
    assert!(
        center.r() != 128 || center.g() != 179 || center.b() != 255,
        "Center pixel should hit chrome sphere, not sky"
    );

    // Verify: bottom should be checkerboard floor (alternating gray)
    let bottom = &frame.data[(HEIGHT * 3 / 4) * WIDTH + (WIDTH / 2)];
    assert!(
        bottom.r() != 128 || bottom.g() != 179 || bottom.b() != 255,
        "Bottom pixel should hit floor, not sky"
    );
}
