use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{TensorView, TensorViewMut, batch::Batch, execute, ops::SampleAtlas};

// Benchmark constants to avoid magic numbers
const WIDTH: usize = 256;
const HEIGHT: usize = 256;
const STRIDE: usize = 256;
const BUFFER_SIZE: usize = WIDTH * HEIGHT;

fn bench_gather_2d(c: &mut Criterion) {
    let data = vec![0u8; BUFFER_SIZE];
    let view = TensorView::new(&data, WIDTH, HEIGHT, STRIDE);

    let x = Batch::<u32>::new(0, 10, 20, 30);
    let y = Batch::<u32>::splat(10);

    c.bench_function("gather_2d", |b| {
        b.iter(|| unsafe {
            black_box(view.gather_2d(black_box(x), black_box(y)));
        })
    });
}

fn bench_sample_4bit_bilinear(c: &mut Criterion) {
    let data = vec![0u8; BUFFER_SIZE];
    let view = TensorView::new(&data, WIDTH, HEIGHT, STRIDE);

    // 16.16 fixed point coordinates
    let u = Batch::<u32>::new(10 << 16, 20 << 16, 30 << 16, 40 << 16);
    let v = Batch::<u32>::splat(50 << 16);

    c.bench_function("sample_4bit_bilinear", |b| {
        b.iter(|| unsafe {
            black_box(view.sample_4bit_bilinear(black_box(u), black_box(v)));
        })
    });
}

fn bench_pipeline_execution(c: &mut Criterion) {
    let atlas_data = vec![0u8; BUFFER_SIZE];
    let atlas_view = TensorView::new(&atlas_data, WIDTH, HEIGHT, STRIDE);

    let mut target_data = vec![0u8; BUFFER_SIZE];
    // We don't create the view here because execute takes a mutable reference and we need to create it inside the loop
    // or reset it, but TensorViewMut is just a struct, so creating it is cheap.

    let pipeline = SampleAtlas { atlas: atlas_view };

    c.bench_function("pipeline_simple_sample", |b| {
        b.iter(|| {
            let mut target_view = TensorViewMut::new(&mut target_data, WIDTH, HEIGHT, STRIDE);
            execute(pipeline, &mut target_view);
        })
    });
}

criterion_group!(
    benches,
    bench_gather_2d,
    bench_sample_4bit_bilinear,
    bench_pipeline_execution
);
criterion_main!(benches);
