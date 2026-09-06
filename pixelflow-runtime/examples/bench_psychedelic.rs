//! Head-to-head on the psychedelic shader: the monomorphized `kernel_raw!`
//! and `kernel!` expression templates against the JIT, over ONE frame.
//!
//! Both sides are denominated over the same 1920x1080 lattice so the numbers
//! subtract: the templates through this benchmark's own loop (a benchmark
//! owns its loop — that is not an API), the JIT through `Lattice::bake`,
//! which is the only way it is reachable.
//!
//! Each pair is measured twice, the second time under FTZ/DAZ. This shader
//! produces no denormals, so the guard should change nothing; printing it is
//! what makes that a measurement rather than an assumption.
//!
//! cargo run --release -p pixelflow-runtime --example bench_psychedelic

use pixelflow_compiler::{kernel, kernel_jit, kernel_raw};
use pixelflow_core::combinator::Manifold;
use pixelflow_core::{FastMathGuard, Field, Kernel, Lattice, PARALLELISM};

#[cfg(target_os = "macos")]
#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

fn nanos_now() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // mach_absolute_time() ticks are NOT nanoseconds on native Apple
        // Silicon: the timebase is 125/3 (one tick = 41.67ns; 1:1 only holds
        // on Intel Macs and under Rosetta). Convert via mach_timebase_info,
        // queried once. See pixelflow-pipeline/src/jit_bench.rs::nanos_now
        // applied to the shared JIT bench harness.
        static TIMEBASE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
        let (numer, denom) = *TIMEBASE.get_or_init(|| {
            let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
            let rc = unsafe { mach_timebase_info(&mut info) };
            assert_eq!(rc, 0, "mach_timebase_info failed with {}", rc);
            assert_ne!(info.denom, 0, "mach_timebase_info returned denom=0");
            (info.numer, info.denom)
        });
        let ticks = unsafe { mach_absolute_time() };
        ((ticks as u128 * numer as u128) / denom as u128) as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let epoch = EPOCH.get_or_init(std::time::Instant::now);
        epoch.elapsed().as_nanos() as u64
    }
}

// LLVM only — no e-graph optimization
kernel_raw!(struct PsychRaw = || Field -> Field {
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
    let y_factor_r = (y + sin_w03 * 0.2).exp();
    let raw_r = y_factor_r * radial_factor / swirl;
    let red = (raw_r / (raw_r.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
    let raw_g = y_factor_g * radial_factor / swirl;
    let green = (raw_g / (raw_g.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
    let raw_b = y_factor_b * radial_factor / swirl;
    let blue = (raw_b / (raw_b.abs() + 1.0) + 1.0) * 0.5;
    red + green + blue
});

// NNUE + LLVM — e-graph saturation + neural extraction + DAG CSE + LLVM backend
kernel!(struct PsychOpt = || Field -> Field {
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
    let y_factor_r = (y + sin_w03 * 0.2).exp();
    let raw_r = y_factor_r * radial_factor / swirl;
    let red = (raw_r / (raw_r.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
    let raw_g = y_factor_g * radial_factor / swirl;
    let green = (raw_g / (raw_g.abs() + 1.0) + 1.0) * 0.5;
    let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
    let raw_b = y_factor_b * radial_factor / swirl;
    let blue = (raw_b / (raw_b.abs() + 1.0) + 1.0) * 0.5;
    red + green + blue
});

/// The frame every measurement in this file is denominated over.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Timed repetitions of a whole frame. Odd, so the median is a sample.
const SAMPLES: usize = 5;

/// Median sample, in nanoseconds per pixel of one frame.
fn ns_per_pixel(times: &mut [u64]) -> f64 {
    times.sort_unstable();
    times[times.len() / 2] as f64 / (WIDTH * HEIGHT) as f64
}

/// The JIT path, measured the only way it is reachable: one `Lattice::bake`
/// per frame.
///
/// There is no per-batch entry to time instead — a `Kernel` plus a `Lattice`
/// IS the evaluation API, and the loop nest, the invariant hoisting and the
/// register allocation across all of it belong to the compiler.
#[inline(never)]
fn bench_bake(shader: &Kernel) -> f64 {
    let lattice = Lattice::frame(WIDTH, HEIGHT, 0.0);

    // Warmup: pays the one-time JIT compile (cached thereafter).
    std::hint::black_box(lattice.bake(shader));

    let mut times = [0u64; SAMPLES];
    for t in &mut times {
        let start = nanos_now();
        std::hint::black_box(lattice.bake(shader));
        *t = nanos_now() - start;
    }
    ns_per_pixel(&mut times)
}

/// The expression templates over the same frame: this benchmark's own loop,
/// one `Manifold::eval` per SIMD batch, every row of the lattice the JIT
/// bakes. Same pixels, same coordinates, so the two numbers subtract.
#[inline(never)]
fn bench_templates<M: Manifold<Output = Field>>(shader: &M) -> f64 {
    let steps_x = WIDTH / PARALLELISM;
    let z = Field::from(0.0f32);
    let w = Field::from(0.0f32);

    let frame = || {
        for py in 0..HEIGHT {
            let y = Field::from(py as f32);
            for step in 0..steps_x {
                let x = Field::sequential((step * PARALLELISM) as f32);
                std::hint::black_box(shader.eval((x, y, z, w)));
            }
        }
    };

    frame(); // warmup

    let mut times = [0u64; SAMPLES];
    for t in &mut times {
        let start = nanos_now();
        frame();
        *t = nanos_now() - start;
    }
    ns_per_pixel(&mut times)
}

/// One pass over all three variants, so the FTZ/DAZ comparison is the same
/// measurement twice rather than two different ones.
struct Pass {
    raw: f64,
    opt: f64,
    bake: f64,
}

fn measure(raw: &PsychRaw, opt: &PsychOpt, jit: &Kernel) -> Pass {
    Pass {
        raw: bench_templates(raw),
        opt: bench_templates(opt),
        bake: bench_bake(jit),
    }
}

fn report(label: &str, p: &Pass) {
    println!("  {label}");
    println!(
        "    templates, frame loop (kernel_raw!): {:.3} ns/pixel",
        p.raw
    );
    println!(
        "    templates, frame loop (kernel!):     {:.3} ns/pixel",
        p.opt
    );
    println!(
        "    JIT, whole frame (Lattice::bake):    {:.3} ns/pixel",
        p.bake
    );
    println!("    bake vs kernel!: {:.2}x", p.opt / p.bake);
}

fn main() {
    let raw = PsychRaw {};
    let opt = PsychOpt {};
    let jit = kernel_jit!(|| {
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
        let swirl =
            ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;
        let pulse = 1.0 + sin_w20 * 0.1;
        let radial_factor = (radial * -4.0 * pulse).exp();
        let y_factor_r = (y + sin_w03 * 0.2).exp();
        let raw_r = y_factor_r * radial_factor / swirl;
        let red = (raw_r / (raw_r.abs() + 1.0) + 1.0) * 0.5;
        let y_factor_g = (y * -1.0 + sin_w03 * 0.2).exp();
        let raw_g = y_factor_g * radial_factor / swirl;
        let green = (raw_g / (raw_g.abs() + 1.0) + 1.0) * 0.5;
        let y_factor_b = (y * -2.0 + sin_w03 * 0.2).exp();
        let raw_b = y_factor_b * radial_factor / swirl;
        let blue = (raw_b / (raw_b.abs() + 1.0) + 1.0) * 0.5;
        red + green + blue
    });

    println!(
        "=== Psychedelic Shader (3ch, {WIDTH}x{HEIGHT}, {} SIMD lanes) ===\n",
        PARALLELISM
    );

    let plain = measure(&raw, &opt, &jit);
    let fast = {
        // SAFETY: single-threaded benchmark; the guard restores the FP
        // control state when it drops at the end of this block.
        let _guard = unsafe { FastMathGuard::new() };
        measure(&raw, &opt, &jit)
    };

    report("default FP mode", &plain);
    println!();
    report("FTZ/DAZ (FastMathGuard)", &fast);
}
