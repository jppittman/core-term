use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pixelflow_graphics::{Color, NamedColor};
use pixelflow_core::{Field, Manifold, PARALLELISM};

fn bench_color_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_eval");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(0.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);
    let p = (x, y, z, w);

    let color = Color::Named(NamedColor::Red);

    group.bench_function("eval_named_color", |b| {
        b.iter(|| black_box(color.eval(black_box(p))))
    });

    group.finish();
}

criterion_group!(benches, bench_color_eval);
criterion_main!(benches);
