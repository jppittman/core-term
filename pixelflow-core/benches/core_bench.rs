use criterion::{Criterion, criterion_group, criterion_main};

fn bench_noop(_c: &mut Criterion) {
    // Benchmarks disabled by user request due to API drift
}

criterion_group!(benches, bench_noop);
criterion_main!(benches);
