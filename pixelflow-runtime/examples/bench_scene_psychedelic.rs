//! The S2 gate: the psychedelic shader rendered as a `Scene`, both ways.
//!
//! `cargo run --release -p pixelflow-runtime --example bench_scene_psychedelic`
//!
//! - **surface**: `Scene::Surface` — a monomorphized `kernel!` channel per
//!   colour, packed by `ColorCube`, evaluated once per SIMD batch through the
//!   `Manifold` trait by the work-stealing rasterizer.
//! - **packed**: `Scene::Packed` — the same three channels plus an opaque
//!   alpha as four `Kernel`s compiled at the frame's lattice shape with the
//!   pixel pack inside the kernel, collapsed one call per stripe.
//!
//! Same shader, same frame, same thread counts; the two are compared pixel by
//! pixel before either is timed, so a fast wrong answer cannot win.
//!
//! This is the per-stage measurement the plan of record
//! (`docs/plans/2026-09-06-kernel-with-a-lattice.md`) asks for, and S3 needs
//! it again to retire the surface lane — hence an example rather than a
//! scratch file. The surface baseline lives here because examples are the one
//! place a new `Scene::Surface` is still permitted.

use std::sync::Arc;
use std::time::Instant;

use pixelflow_compiler::kernel;
use pixelflow_core::{Field, Kernel};
use pixelflow_graphics::render::color::{Rgba8, RgbaColorCube};
use pixelflow_graphics::render::scene::{compile_packed_for, Scene};
use pixelflow_graphics::render::Frame;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

// ── the shader, as monomorphized expression templates (the surface lane) ──

kernel!(struct PsychRed = || Field -> Field {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = W + 1.3;
    let r_sq = x * x + y * y;
    let radial = (r_sq - 0.7).abs();
    let swirl_scale = (1.0 - radial) * 5.0;
    let vx = x * swirl_scale;
    let vy = y * swirl_scale;
    let phase = time * 0.5;
    let sin_w03 = (time * 0.3).sin();
    let sin_w20 = (time * 2.0).sin();
    let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
    let pulse = 1.0 + sin_w20 * 0.1;
    let radial_factor = (radial * -4.0 * pulse).exp();
    let raw = (y + sin_w03 * 0.2).exp() * radial_factor / swirl;
    (raw / (raw.abs() + 1.0) + 1.0) * 0.5
});

kernel!(struct PsychGreen = || Field -> Field {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = W + 1.3;
    let r_sq = x * x + y * y;
    let radial = (r_sq - 0.7).abs();
    let swirl_scale = (1.0 - radial) * 5.0;
    let vx = x * swirl_scale;
    let vy = y * swirl_scale;
    let phase = time * 0.5;
    let sin_w03 = (time * 0.3).sin();
    let sin_w20 = (time * 2.0).sin();
    let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
    let pulse = 1.0 + sin_w20 * 0.1;
    let radial_factor = (radial * -4.0 * pulse).exp();
    let raw = (y * -1.0 + sin_w03 * 0.2).exp() * radial_factor / swirl;
    (raw / (raw.abs() + 1.0) + 1.0) * 0.5
});

kernel!(struct PsychBlue = || Field -> Field {
    let scale = 2.0 / 1080.0;
    let x = (X - 960.0) * scale;
    let y = (540.0 - Y) * scale;
    let time = W + 1.3;
    let r_sq = x * x + y * y;
    let radial = (r_sq - 0.7).abs();
    let swirl_scale = (1.0 - radial) * 5.0;
    let vx = x * swirl_scale;
    let vy = y * swirl_scale;
    let phase = time * 0.5;
    let sin_w03 = (time * 0.3).sin();
    let sin_w20 = (time * 2.0).sin();
    let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
    let pulse = 1.0 + sin_w20 * 0.1;
    let radial_factor = (radial * -4.0 * pulse).exp();
    let raw = (y * -2.0 + sin_w03 * 0.2).exp() * radial_factor / swirl;
    (raw / (raw.abs() + 1.0) + 1.0) * 0.5
});

// ── the same shader, as four channel kernels (the packed lane) ──

/// The `y` weight is the only thing that separates the three channels.
fn psych_channel(y_weight: f32) -> Kernel {
    let k = Kernel::constant;
    let scale = 2.0 / 1080.0;
    let x = Kernel::x().sub(&k(960.0)).mul(&k(scale));
    let y = k(540.0).sub(&Kernel::y()).mul(&k(scale));
    let time = Kernel::w().add(&k(1.3));
    let r_sq = x.mul(&x).add(&y.mul(&y));
    let radial = r_sq.sub(&k(0.7)).abs();
    let swirl_scale = k(1.0).sub(&radial).mul(&k(5.0));
    let vx = x.mul(&swirl_scale);
    let vy = y.mul(&swirl_scale);
    let phase = time.mul(&k(0.5));
    let sin_w03 = time.mul(&k(0.3)).sin();
    let sin_w20 = time.mul(&k(2.0)).sin();
    let vxp = vx.add(&phase);
    let swirl = vxp
        .sin()
        .add(&k(1.0))
        .mul(&vxp.sub(&vy.add(&phase.mul(&k(0.7)))).abs())
        .mul(&k(0.2))
        .add(&k(0.001));
    let pulse = k(1.0).add(&sin_w20.mul(&k(0.1)));
    let radial_factor = radial.mul(&k(-4.0)).mul(&pulse).exp();
    let raw = y
        .mul(&k(y_weight))
        .add(&sin_w03.mul(&k(0.2)))
        .exp()
        .mul(&radial_factor)
        .div(&swirl);
    raw.div(&raw.abs().add(&k(1.0))).add(&k(1.0)).mul(&k(0.5))
}

fn surface_scene() -> Scene {
    Scene::Surface(Arc::new(RgbaColorCube::default().at(
        PsychRed {},
        PsychGreen {},
        PsychBlue {},
        1.0f32,
    )))
}

fn packed_scene() -> Scene {
    let channels = [
        psych_channel(1.0),
        psych_channel(-1.0),
        psych_channel(-2.0),
        Kernel::constant(1.0),
    ];
    Scene::Packed(compile_packed_for::<Rgba8>(&channels, [WIDTH as u32, HEIGHT as u32]).bind(&[]))
}

/// Median ns/pixel over `runs` frames, after `warm` warm-up frames.
fn time_ns_per_px(scene: &Scene, threads: usize, warm: usize, runs: usize) -> f64 {
    let mut frame = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    for _ in 0..warm {
        scene.render(&mut frame, threads);
        std::hint::black_box(&frame.data[0]);
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        scene.render(&mut frame, threads);
        samples.push(t.elapsed().as_nanos());
        std::hint::black_box(&frame.data[0]);
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / (WIDTH * HEIGHT) as f64
}

/// The largest per-channel disagreement between the two lanes, in bytes.
fn max_channel_delta(a: &Scene, b: &Scene) -> u8 {
    let mut fa = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    let mut fb = Frame::<Rgba8>::new(WIDTH as u32, HEIGHT as u32);
    a.render(&mut fa, 1);
    b.render(&mut fb, 1);
    let mut worst = 0u8;
    for (pa, pb) in fa.data.iter().zip(fb.data.iter()) {
        for (x, y) in [
            (pa.r(), pb.r()),
            (pa.g(), pb.g()),
            (pa.b(), pb.b()),
            (pa.a(), pb.a()),
        ] {
            worst = worst.max(x.abs_diff(y));
        }
    }
    worst
}

fn main() {
    const WARM: usize = 3;
    const RUNS: usize = 9;
    /// `PerformanceConfig::default().render_threads`.
    const RUNTIME_THREADS: usize = 12;

    let surface = surface_scene();
    let packed = packed_scene();

    println!(
        "psychedelic shader as a Scene, {WIDTH}x{HEIGHT}, median of {RUNS} frames (ns/pixel)\n"
    );
    println!(
        "  agreement: max channel delta = {}",
        max_channel_delta(&surface, &packed)
    );

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("\n  threads   surface (Scene::Surface)   packed (Scene::Packed)   speedup");
    for threads in [1, cores, RUNTIME_THREADS] {
        let s = time_ns_per_px(&surface, threads, WARM, RUNS);
        let p = time_ns_per_px(&packed, threads, WARM, RUNS);
        println!("  {threads:>7}   {s:>22.2}   {p:>20.2}   {:>6.2}x", s / p);
    }
}
