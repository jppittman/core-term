//! Chrome Sphere Parallel Rendering Demo
//!
//! Compares single-threaded vs parallel rasterization of the 3D chrome sphere scene.
//! Uses the mullet architecture: geometry once, colors as packed Discrete.

use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_graphics::render::color::Rgba8;
use pixelflow_graphics::render::frame::Frame;
use pixelflow_graphics::render::rasterizer::{execute, render_parallel, RenderOptions, TensorShape};
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, PlaneGeometry, SphereAt,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const W: usize = 1920;
const H: usize = 1080;

/// Remap pixel coordinates to normalized screen coordinates for ~60° FOV.
#[derive(Clone)]
struct ColorScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: Manifold<Output = Discrete>> Manifold for ColorScreenRemap<M> {
    type Output = Discrete;

    fn eval_raw(&self, px: Field, py: Field, z: Field, w: Field) -> Discrete {
        let width = Field::from(self.width);
        let height = Field::from(self.height);

        let scale = Field::from(2.0) / height;
        let x = (px - width * Field::from(0.5)) * scale;
        let y = (height * Field::from(0.5) - py) * scale;

        self.inner.eval_raw(x, y, z, w)
    }
}

/// Build the color scene using the mullet architecture.
/// Geometry runs once, colors flow as packed RGBA.
fn build_scene() -> impl Manifold<Output = Discrete> + Clone + Sync {
    let world = ColorSurface {
        geometry: PlaneGeometry { height: -1.0 },
        material: ColorChecker,
        background: ColorSky,
    };

    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: world },
        background: world,
    };

    ColorScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    }
}

fn main() {
    println!("Chrome Sphere Parallel Rendering Demo");
    println!("=====================================");
    println!("Resolution: {}x{} ({:.1}M pixels)", W, H, (W * H) as f64 / 1_000_000.0);
    println!();

    // Build the scene using mullet architecture (geometry once, colors as packed Discrete)
    let scene = build_scene();

    let shape = TensorShape { width: W, height: H };
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("Available CPU threads: {}", num_cpus);
    println!();

    // Warm-up run
    {
        let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
        execute(&scene, frame.as_slice_mut(), shape);
    }

    // Single-threaded benchmark
    let single_time = {
        let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
        let start = Instant::now();
        execute(&scene, frame.as_slice_mut(), shape);
        let elapsed = start.elapsed();

        let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        let fps = 1.0 / elapsed.as_secs_f64();
        println!("Single-threaded: {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS)",
            elapsed.as_secs_f64() * 1000.0, mpps, fps);

        // Save the image
        let path = std::env::temp_dir().join("chrome_sphere_single.ppm");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
        for p in &frame.data {
            file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
        }
        println!("  Saved: {}", path.display());

        elapsed
    };

    println!();

    // Parallel benchmarks with different thread counts
    for threads in [2, 4, 8, num_cpus].iter().filter(|&&t| t <= num_cpus) {
        let mut frame = Frame::<Rgba8>::new(W as u32, H as u32);
        let options = RenderOptions { num_threads: *threads };

        let start = Instant::now();
        render_parallel(&scene, frame.as_slice_mut(), shape, options);
        let elapsed = start.elapsed();

        let mpps = (W * H) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        let fps = 1.0 / elapsed.as_secs_f64();
        let speedup = single_time.as_secs_f64() / elapsed.as_secs_f64();

        println!("{:>2}-threaded:      {:>7.2}ms ({:>5.1} Mpix/s, {:>5.1} FPS) - {:.2}x speedup",
            threads, elapsed.as_secs_f64() * 1000.0, mpps, fps, speedup);

        if *threads == num_cpus {
            // Save the parallel result
            let path = std::env::temp_dir().join("chrome_sphere_parallel.ppm");
            let mut file = File::create(&path).unwrap();
            writeln!(file, "P6\n{} {}\n255", W, H).unwrap();
            for p in &frame.data {
                file.write_all(&[p.r(), p.g(), p.b()]).unwrap();
            }
            println!("  Saved: {}", path.display());
        }
    }

    println!();
    println!("Done! Images saved to /tmp/");
}
