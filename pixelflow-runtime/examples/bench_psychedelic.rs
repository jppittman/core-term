//! Head-to-head: LLVM vs NNUE+LLVM vs JIT on the psychedelic shader.
//!
//! cargo run --release -p pixelflow-runtime --example bench_psychedelic

use pixelflow_compiler::{kernel, kernel_jit, kernel_raw};
use pixelflow_core::{Field, Manifold, PARALLELISM};

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

#[inline(never)]
fn bench_scanline<M: Manifold<Output = Field>>(shader: &M) -> f64 {
    let width = 1920usize;
    let height = 1080usize;
    let steps_x = width / PARALLELISM;
    let z = Field::from(0.0f32);
    let w = Field::from(0.0f32);

    // Warmup: full frame
    for py in (0..height).step_by(108) {
        let y = Field::from(py as f32);
        for step in 0..steps_x {
            let x = Field::sequential((step * PARALLELISM) as f32);
            std::hint::black_box(shader.eval((x, y, z, w)));
        }
    }

    // Benchmark: 10 scanlines at different Y positions
    // This prevents LLVM from hoisting Y-dependent computations
    let scanlines = 10usize;
    let total_pixels = width * scanlines;
    let samples = 50;
    let mut times = vec![0u64; samples];
    for t in &mut times {
        let start = nanos_now();
        for sy in 0..scanlines {
            let y = Field::from((sy * 108) as f32);
            for step in 0..steps_x {
                let x = Field::sequential((step * PARALLELISM) as f32);
                std::hint::black_box(shader.eval((x, y, z, w)));
            }
        }
        *t = nanos_now() - start;
    }
    times.sort();
    times[samples / 2] as f64 / total_pixels as f64
}

/// The JIT through its ROW entry point rather than one call per SIMD vector.
///
/// `bench_scanline` drives every tier through `Manifold::eval`, which for the
/// JIT means crossing the `extern "C"` boundary once per [`PARALLELISM`]
/// pixels — 120 calls per 1920px scanline — while the combinator tiers it is
/// compared against are inlined Rust with no call at all. That is not a
/// codegen difference, it is an entry-point difference, and `eval_row` is the
/// entry point that removes it: the loop lives inside the emitted kernel, so
/// the boundary is crossed once per row and the spill frame is set up once
/// instead of 120 times.
#[inline(never)]
fn bench_scanline_rows(jit: &pixelflow_core::__macro::codegen::JitManifold) -> f64 {
    use pixelflow_core::__macro::codegen::Point4;

    let width = 1920usize;
    let height = 1080usize;
    let mut row = vec![0.0f32; width];

    for py in (0..height).step_by(108) {
        jit.eval_row(&mut row, Point4::new(0.0, py as f32, 0.0, 0.0));
        std::hint::black_box(&row);
    }

    let scanlines = 10usize;
    let total_pixels = width * scanlines;
    let samples = 50;
    let mut times = vec![0u64; samples];
    for t in &mut times {
        let start = nanos_now();
        for sy in 0..scanlines {
            jit.eval_row(&mut row, Point4::new(0.0, (sy * 108) as f32, 0.0, 0.0));
            std::hint::black_box(&row);
        }
        *t = nanos_now() - start;
    }
    times.sort();
    times[samples / 2] as f64 / total_pixels as f64
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
        "=== Psychedelic Shader (3ch, 1920px scanline, {} SIMD lanes) ===\n",
        PARALLELISM
    );

    let raw_ns = bench_scanline(&raw);
    let opt_ns = bench_scanline(&opt);
    let jit_ns = bench_scanline(&jit);
    // A 2x speedup is the shape a bug makes too (a row that is never written
    // is very fast), so check eval_row computes the same pixels before
    // believing its timings. `eval_at` is the independent-enough witness
    // available here: same emitted kernel, but a one-group tile instead of a
    // 120-group one, so it exercises the loop bounds and the X induction
    // rather than sharing them.
    {
        use pixelflow_core::__macro::codegen::Point4;
        let width = 1920usize;
        let mut row = vec![0.0f32; width];
        let y_probe = 324.0f32;
        let jitm = jit.__compiled();
        jitm.eval_row(&mut row, Point4::new(0.0, y_probe, 0.0, 0.0));

        let mut mismatches = 0usize;
        for (i, &got) in row.iter().enumerate() {
            let want = jitm.eval_at(Point4::new(i as f32, y_probe, 0.0, 0.0));
            if got.to_bits() != want.to_bits() {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "eval_row disagrees with eval_at on {mismatches}/{width} pixels — \
             the row timing below would be measuring the wrong thing"
        );
        let lo = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi > lo,
            "eval_row produced a constant row ({lo}); it is not evaluating per-pixel"
        );
        println!("  (eval_row verified: {width} pixels match eval_at, range {lo:.3}..{hi:.3})\n");
    }

    let jit_row_ns = bench_scanline_rows(jit.__compiled());

    println!("  LLVM only (kernel_raw!):  {:.3}ns/pixel", raw_ns);
    println!("  NNUE + LLVM (kernel!):    {:.3}ns/pixel", opt_ns);
    println!("  JIT (kernel_jit!):        {:.3}ns/pixel", jit_ns);
    println!("  JIT, eval_row:            {:.3}ns/pixel", jit_row_ns);
    println!();
    println!(
        "  NNUE+LLVM vs LLVM: {:.1}%",
        (opt_ns / raw_ns - 1.0) * 100.0
    );
    println!(
        "  JIT vs LLVM:       {:.1}%",
        (jit_ns / raw_ns - 1.0) * 100.0
    );
    println!(
        "  JIT rows vs NNUE:  {:.1}%",
        (jit_row_ns / opt_ns - 1.0) * 100.0
    );
}
