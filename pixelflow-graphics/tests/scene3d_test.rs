//! Test: Analytic 3D rendering with three-layer architecture
//!
//! Three Layers:
//! 1. Geometry: Returns `t` (Jet3) - UnitSphere, PlaneGeometry
//! 2. Surface: Warps `P = ray * t` - creates tangent frame via chain rule
//! 3. Material: Reconstructs normal from derivatives - Reflect, Checker, Sky

use pixelflow_core::{Discrete, Field, Jet3, Manifold};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, TensorShape};
use pixelflow_graphics::scene3d::{
    BlueSky, Checker, ColorChecker, PlaneGeometry, Reflect, ScreenToDir, Sky, SphereAt, Surface,
};
use std::fs::File;
use std::io::Write;

/// Convert grayscale Field to Discrete RGBA
struct GrayToRgba<M> {
    inner: M,
}

impl<M: Manifold<Output = Field>> Manifold for GrayToRgba<M> {
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let gray = self.inner.eval_raw(x, y, z, w);
        Discrete::pack(gray, gray, gray, Field::from(1.0))
    }
}

/// Remap pixel coordinates to normalized screen coordinates for ~60° FOV.
/// Transforms [0, width] × [0, height] → [-aspect, aspect] × [-1, 1]
/// where the values represent tan(angle) from optical axis.
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: Manifold<Output = Field>> Manifold for ScreenRemap<M> {
    type Output = Field;

    fn eval_raw(&self, px: Field, py: Field, z: Field, w: Field) -> Field {
        let width = Field::from(self.width);
        let height = Field::from(self.height);

        // Scale so that screen Y range is [-1, 1]
        // This gives ~53° vertical FOV (since atan(1) = 45°, the full range is 90°)
        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale;
        let y = (height * Field::from(0.5) - py) * scale; // Flip Y

        self.inner.eval_raw(x, y, z, w)
    }
}

// ============================================================================
// TESTS
// ============================================================================

/// Test: Chrome sphere at z=4 reflecting floor and sky.
/// Uses the new three-layer architecture:
/// - ScreenToDir: seeds jets with screen derivatives, normalizes direction
/// - Surface<SphereAt, Reflect<world>, world>: sphere reflecting world
/// - world = Surface<PlaneGeometry, Checker, Sky>: floor + sky
#[test]
fn test_chrome_unit_sphere() {
    const W: usize = 400;
    const H: usize = 300;

    // World = floor + sky
    // PlaneGeometry at y=-1: returns t = -1 / ry (negative plane)
    let world = Surface {
        geometry: PlaneGeometry { height: -1.0 },
        material: Checker,
        background: Sky,
    };

    // Scene = chrome sphere at (0, 0, 4) + world background
    // SphereAt solves quadratic for intersection distance
    let scene = Surface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: Reflect { inner: world },
        background: world,
    };

    // ScreenToDir: pixel coords → direction jets with derivatives
    let screen = ScreenRemap {
        inner: ScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    // Wrap in GrayToRgba for rendering
    let renderable = GrayToRgba { inner: screen };

    // Render
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    execute(
        &renderable,
        frame.as_slice_mut(),
        TensorShape {
            width: W,
            height: H,
        },
    );

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_chrome_unit_sphere.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Debug: print some pixel values
    let center = &frame.data[(H / 2) * W + (W / 2)];
    let bottom_sphere = &frame.data[(H * 5 / 8) * W + (W / 2)]; // Lower part
    let top_sphere = &frame.data[(H * 3 / 8) * W + (W / 2)];    // Upper part
    let corner = &frame.data[0]; // Top-left (sky)

    println!("Chrome center: r={}", center.r());
    println!("Chrome bottom: r={}", bottom_sphere.r());
    println!("Chrome top: r={}", top_sphere.r());
    println!("Corner (sky): r={}", corner.r());

    // Sanity checks
    assert!(center.r() > 10, "Center should not be black: r={}", center.r());
    // Sky gradient goes from 0.1 (dark) to 0.9 (bright) = 25 to 229
    assert!(corner.r() > 20, "Corner should be sky (not black): r={}", corner.r());
}

/// Test: Just the sky (no geometry)
#[test]
fn test_sky_only() {
    const W: usize = 200;
    const H: usize = 150;

    // Sky as the only "scene" - wraps it in a dummy Surface that always misses
    struct SkyOnly;

    impl Manifold<Jet3> for SkyOnly {
        type Output = Field;

        fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Field {
            // Same as Sky: gradient based on Y direction
            let t = (y.val * Field::from(0.5) + Field::from(0.5))
                .max(Field::from(0.0))
                .min(Field::from(1.0));
            Field::from(0.1) + t * Field::from(0.8)
        }
    }

    let screen = ScreenRemap {
        inner: ScreenToDir { inner: SkyOnly },
        width: W as f32,
        height: H as f32,
    };

    let renderable = GrayToRgba { inner: screen };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    execute(
        &renderable,
        frame.as_slice_mut(),
        TensorShape {
            width: W,
            height: H,
        },
    );

    // Save
    let path = std::env::temp_dir().join("pixelflow_sky_only.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Top should be brighter than bottom (gradient)
    let top = &frame.data[(H / 4) * W + (W / 2)];
    let bottom = &frame.data[(3 * H / 4) * W + (W / 2)];
    println!("Sky top: r={}", top.r());
    println!("Sky bottom: r={}", bottom.r());

    // Top looks "up" (positive y direction), should be brighter
    assert!(top.r() > bottom.r(), "Sky should be brighter at top");
}

/// Test: Floor only (plane with checker pattern)
#[test]
fn test_floor_only() {
    const W: usize = 400;
    const H: usize = 300;

    // Just floor + sky (no sphere)
    let scene = Surface {
        geometry: PlaneGeometry { height: -1.0 },
        material: Checker,
        background: Sky,
    };

    let screen = ScreenRemap {
        inner: ScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    };

    let renderable = GrayToRgba { inner: screen };

    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    execute(
        &renderable,
        frame.as_slice_mut(),
        TensorShape {
            width: W,
            height: H,
        },
    );

    // Save
    let path = std::env::temp_dir().join("pixelflow_floor_only.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Bottom half should hit floor, top half should hit sky
    let floor_pixel = &frame.data[(3 * H / 4) * W + (W / 2)];
    let sky_pixel = &frame.data[(H / 4) * W + (W / 2)];
    println!("Floor: r={}", floor_pixel.r());
    println!("Sky: r={}", sky_pixel.r());

    // Floor should have checkerboard values (either light or dark)
    assert!(
        floor_pixel.r() < 80 || floor_pixel.r() > 180,
        "Floor should be checker (dark or light): r={}",
        floor_pixel.r()
    );
}

/// Test: Color chrome sphere with blue sky
/// Runs the pipeline 3 times (R, G, B) and packs into Discrete.
#[test]
fn test_color_chrome_sphere() {
    const W: usize = 1920;
    const H: usize = 1080;

    // Build scene for each channel
    fn build_scene(
        channel: u8,
    ) -> impl Manifold<Output = Field> {
        let world = Surface {
            geometry: PlaneGeometry { height: -1.0 },
            material: ColorChecker { channel },
            background: BlueSky { channel },
        };

        let scene = Surface {
            geometry: SphereAt {
                center: (0.0, 0.0, 4.0),
                radius: 1.0,
            },
            material: Reflect { inner: world },
            background: world,
        };

        ScreenRemap {
            inner: ScreenToDir { inner: scene },
            width: W as f32,
            height: H as f32,
        }
    }

    // Three-channel renderer
    struct ColorRenderer<R, G, B> {
        r: R,
        g: G,
        b: B,
    }

    impl<R, G, B> Manifold for ColorRenderer<R, G, B>
    where
        R: Manifold<Output = Field>,
        G: Manifold<Output = Field>,
        B: Manifold<Output = Field>,
    {
        type Output = Discrete;

        fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
            let r = self.r.eval_raw(x, y, z, w);
            let g = self.g.eval_raw(x, y, z, w);
            let b = self.b.eval_raw(x, y, z, w);
            Discrete::pack(r, g, b, Field::from(1.0))
        }
    }

    let renderable = ColorRenderer {
        r: build_scene(0),
        g: build_scene(1),
        b: build_scene(2),
    };

    // Render
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
    let start = std::time::Instant::now();
    execute(
        &renderable,
        frame.as_slice_mut(),
        TensorShape {
            width: W,
            height: H,
        },
    );
    let elapsed = start.elapsed();
    let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("Color render: {:?} ({:.2} Mpix/s)", elapsed, mpps);

    // Save PPM
    let path = std::env::temp_dir().join("pixelflow_color_chrome.ppm");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("Saved: {}", path.display());

    // Debug pixels
    let center = &frame.data[(H / 2) * W + (W / 2)];
    let sky = &frame.data[0];
    println!("Center: r={} g={} b={}", center.r(), center.g(), center.b());
    println!("Sky: r={} g={} b={}", sky.r(), sky.g(), sky.b());

    // Sky should be blue-ish (B > R)
    assert!(sky.b() > sky.r(), "Sky should be blue: r={} b={}", sky.r(), sky.b());
}
