//! Fractal Rendering Demo
//!
//! Renders Mandelbrot, Julia, and Burning Ship fractals to PPM image files.
//! Demonstrates the fractal manifolds with grayscale color mapping.

use pixelflow_core::combinators::At;
use pixelflow_core::fractals::{BurningShip, Julia, Mandelbrot};
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_graphics::render::color::{ColorCube, Rgba8};
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, TensorShape};

use std::fs::File;
use std::io::Write;
use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;

/// Remap pixel coordinates to fractal space and apply grayscale coloring.
///
/// Takes the fractal manifold and maps screen pixels to a view window.
/// Uses a simple grayscale color scheme where escaped points are bright
/// and inside points are dark.
#[derive(Clone, Copy)]
struct FractalView<F> {
    fractal: F,
    // View window in fractal coordinates
    center_x: f32,
    center_y: f32,
    zoom: f32, // pixels per unit in fractal space
}

impl<F: Manifold<Output = Field>> Manifold for FractalView<F> {
    type Output = Discrete;

    fn eval_raw(&self, px: Field, py: Field, z: Field, w: Field) -> Discrete {
        let scale = Field::from(1.0 / self.zoom);

        // Precompute offsets as constants
        let offset_x = Field::from(W as f32 * 0.5 - self.center_x * self.zoom);
        let offset_y = Field::from(H as f32 * 0.5 + self.center_y * self.zoom);

        // Build transformed coordinates using At combinator
        let at = At {
            inner: &self.fractal,
            x: (px - offset_x) * scale,
            y: (offset_y - py) * scale,
            z,
            w,
        };

        // Evaluate the fractal to get iteration value (0.0 to 1.0)
        let iter_val = at.eval();

        // Map to grayscale using ColorCube with R=G=B=iter_val
        At {
            inner: ColorCube,
            x: iter_val, // Red
            y: iter_val, // Green
            z: iter_val, // Blue (giving grayscale)
            w: Field::from(1.0), // Alpha
        }
        .eval()
    }
}

fn save_ppm(frame: &Frame<Rgba8>, path: &str) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
    for p in &frame.data {
        file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
    }
    println!("  Saved: {}", path);
}

fn render_fractal<M: Manifold<Output = Discrete> + Sync>(
    name: &str,
    scene: M,
    path: &str,
) -> std::time::Duration {
    let shape = TensorShape {
        width: W,
        height: H,
    };
    let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);

    let start = Instant::now();
    execute(&scene, frame.as_slice_mut(), shape);
    let elapsed = start.elapsed();

    let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!(
        "{}: {:>7.2}ms ({:>5.1} Mpix/s)",
        name,
        elapsed.as_secs_f64() * 1000.0,
        mpps
    );

    save_ppm(&frame, path);
    elapsed
}

fn main() {
    println!("Fractal Rendering Demo");
    println!("======================");
    println!(
        "Resolution: {}x{} ({:.1}M pixels)",
        W,
        H,
        (W * H) as f64 / 1_000_000.0
    );
    println!();

    let temp = std::env::temp_dir();

    // Mandelbrot - classic view
    {
        let scene = FractalView {
            fractal: Mandelbrot::new(256),
            center_x: -0.5, // Slightly left to show the main cardioid
            center_y: 0.0,
            zoom: 300.0,
        };
        render_fractal(
            "Mandelbrot (classic)",
            scene,
            temp.join("mandelbrot.ppm").to_str().unwrap(),
        );
    }

    // Mandelbrot - zoomed into seahorse valley
    {
        let scene = FractalView {
            fractal: Mandelbrot::new(512),
            center_x: -0.745,
            center_y: 0.186,
            zoom: 5000.0,
        };
        render_fractal(
            "Mandelbrot (seahorse)",
            scene,
            temp.join("mandelbrot_seahorse.ppm").to_str().unwrap(),
        );
    }

    // Julia - rabbit
    {
        let scene = FractalView {
            fractal: Julia::rabbit(256),
            center_x: 0.0,
            center_y: 0.0,
            zoom: 350.0,
        };
        render_fractal(
            "Julia (rabbit)",
            scene,
            temp.join("julia_rabbit.ppm").to_str().unwrap(),
        );
    }

    // Julia - seahorse
    {
        let scene = FractalView {
            fractal: Julia::seahorse(256),
            center_x: 0.0,
            center_y: 0.0,
            zoom: 400.0,
        };
        render_fractal(
            "Julia (seahorse)",
            scene,
            temp.join("julia_seahorse.ppm").to_str().unwrap(),
        );
    }

    // Julia - spiral
    {
        let scene = FractalView {
            fractal: Julia::spiral(256),
            center_x: 0.0,
            center_y: 0.0,
            zoom: 300.0,
        };
        render_fractal(
            "Julia (spiral)",
            scene,
            temp.join("julia_spiral.ppm").to_str().unwrap(),
        );
    }

    // Burning Ship
    {
        let scene = FractalView {
            fractal: BurningShip::new(256),
            center_x: -1.75,
            center_y: -0.035,
            zoom: 800.0,
        };
        render_fractal(
            "Burning Ship",
            scene,
            temp.join("burning_ship.ppm").to_str().unwrap(),
        );
    }

    println!();
    println!("Done! Images saved to {}", temp.display());
}
