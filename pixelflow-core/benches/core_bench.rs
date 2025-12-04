use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{
    SampleAtlas, SimdBatch, TensorView, TensorViewMut,
    batch::{Batch, NativeBackend},
    dsl::{MaskExt, SurfaceExt},
    execute,
    surfaces::Max,
};

// Benchmark constants to avoid magic numbers
const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const STRIDE: usize = 256;
const BUFFER_SIZE: usize = WIDTH * HEIGHT;

fn bench_gather_2d(c: &mut Criterion) {
    let data = vec![0u8; BUFFER_SIZE];
    let view = TensorView::new(&data, WIDTH, HEIGHT, STRIDE);

    let x = Batch::<u32>::sequential_from(0) * Batch::<u32>::splat(10);
    let y = Batch::<u32>::splat(10);

    c.bench_function("gather_2d", |b| {
        b.iter(|| unsafe {
            black_box(view.gather_2d::<NativeBackend>(black_box(x), black_box(y)));
        })
    });
}

fn bench_sample_4bit_bilinear(c: &mut Criterion) {
    let data = vec![0u8; BUFFER_SIZE];
    let view = TensorView::new(&data, WIDTH, HEIGHT, STRIDE);

    // 16.16 fixed point coordinates
    let u = (Batch::<u32>::sequential_from(1) * Batch::<u32>::splat(10)) << 16;
    let v = Batch::<u32>::splat(50 << 16);

    c.bench_function("sample_4bit_bilinear", |b| {
        b.iter(|| unsafe {
            black_box(view.sample_4bit_bilinear::<NativeBackend>(black_box(u), black_box(v)));
        })
    });
}

fn bench_pipeline_execution(c: &mut Criterion) {
    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);

    let mut target_data = vec![0u8; BUFFER_SIZE];

    let pipeline = SampleAtlas {
        atlas: atlas_view,
        step_x_fp: 65536,
        step_y_fp: 65536,
    };

    c.bench_function("pipeline_simple_sample", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            execute(
                &pipeline,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });
}

// --- New Benchmarks ---

// 1. Blend Math Micro-benchmark
#[inline(always)]
fn blend_math_impl(fg: Batch<u32>, bg: Batch<u32>, alpha: Batch<u32>) -> Batch<u32> {
    let inv_alpha = Batch::splat(256) - alpha;
    ((fg * alpha) + (bg * inv_alpha)) >> 8
}

fn bench_blend_math(c: &mut Criterion) {
    let fg = Batch::splat(0xFFFFFFFF);
    let bg = Batch::splat(0x00000000);
    let alpha = Batch::splat(128);

    c.bench_function("blend_math", |b| {
        b.iter(|| {
            black_box(blend_math_impl(
                black_box(fg),
                black_box(bg),
                black_box(alpha),
            ));
        })
    });
}

// 2. Operators Benchmarks
fn bench_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("operators");

    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);
    let source = SampleAtlas {
        atlas: atlas_view,
        step_x_fp: 65536,
        step_y_fp: 65536,
    };

    let mut target_data = vec![0u8; BUFFER_SIZE];

    // Offset
    group.bench_function("Offset", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = SurfaceExt::<u8>::offset(source, 10, 10);
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // Skew
    group.bench_function("Skew", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = SurfaceExt::<u8>::skew(source, 50);
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // Max
    group.bench_function("Max", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = Max(source, SurfaceExt::<u8>::offset(source, 1, 0));
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    group.finish();
}

// 3. Sampling Scaling Benchmarks
fn bench_sampling_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_scaling");

    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);
    let mut target_data = vec![0u8; BUFFER_SIZE];

    // 0.5x Scale (Minification) - Step = 2.0 (131072)
    group.bench_function("0.5x", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = SampleAtlas {
                atlas: atlas_view,
                step_x_fp: 131072,
                step_y_fp: 131072,
            };
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // 1.0x Scale - Step = 1.0 (65536)
    group.bench_function("1.0x", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = SampleAtlas {
                atlas: atlas_view,
                step_x_fp: 65536,
                step_y_fp: 65536,
            };
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // 2.0x Scale (Magnification) - Step = 0.5 (32768)
    group.bench_function("2.0x", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = SampleAtlas {
                atlas: atlas_view,
                step_x_fp: 32768,
                step_y_fp: 32768,
            };
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    group.finish();
}

// 4. Text Pipelines Benchmarks
fn bench_text_pipelines(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_pipelines");

    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);
    let source = SampleAtlas {
        atlas: atlas_view,
        step_x_fp: 65536,
        step_y_fp: 65536,
    };

    let mut target_data = vec![0u32; BUFFER_SIZE]; // u32 for colored output
    let fg = Batch::splat(0xFFFFFFFF);
    let bg = Batch::splat(0xFF000000);

    // Normal: Sampler -> Over
    group.bench_function("Normal", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let pipe = MaskExt::<u32>::over(source, fg, bg);
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // Bold: Max(Sampler, Sampler.offset) -> Over
    group.bench_function("Bold", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let bold = Max(source, SurfaceExt::<u32>::offset(source, 1, 0));
            let pipe = MaskExt::<u32>::over(bold, fg, bg);
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // Italic: Skew(Sampler) -> Over
    group.bench_function("Italic", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let italic = SurfaceExt::<u32>::skew(source, 50);
            let pipe = MaskExt::<u32>::over(italic, fg, bg);
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    // Bold Italic: Max(Italic, Italic.offset) -> Over
    group.bench_function("BoldItalic", |b| {
        b.iter(|| {
            let target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            let italic = SurfaceExt::<u32>::skew(source, 50);
            let bold_italic = Max(italic, SurfaceExt::<u32>::offset(italic, 1, 0));
            let pipe = MaskExt::<u32>::over(bold_italic, fg, bg);
            execute(
                &pipe,
                target_view.data,
                target_view.width,
                target_view.height,
            );
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gather_2d,
    bench_sample_4bit_bilinear,
    bench_pipeline_execution,
    bench_blend_math,
    bench_operators,
    bench_sampling_scaling,
    bench_text_pipelines
);
criterion_main!(benches);
