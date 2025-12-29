//! Benchmark: Chebyshev trig functions
//!
//! Measures throughput and relative performance.
//! Accuracy is verified in integration tests.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pixelflow_core::ManifoldExt;
use pixelflow_core::{X, Y};
use std::f32::consts::PI;

fn sin_benchmark(c: &mut Criterion) {
    c.bench_function("cheby_sin_throughput", |b| {
        let test_angles: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                (t - 0.5) * 8.0 * PI
            })
            .collect();

        b.iter(|| {
            let manifold = X.sin();
            let mut acc = 0.0f32;
            for &angle in &test_angles {
                let mut out = [0.0f32; 16];
                pixelflow_core::materialize_scalar(&manifold, black_box(angle), 0.0, &mut out);
                acc += out[0];
            }
            black_box(acc)
        })
    });
}

fn cos_benchmark(c: &mut Criterion) {
    c.bench_function("cheby_cos_throughput", |b| {
        let test_angles: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                (t - 0.5) * 8.0 * PI
            })
            .collect();

        b.iter(|| {
            let manifold = X.cos();
            let mut acc = 0.0f32;
            for &angle in &test_angles {
                let mut out = [0.0f32; 16];
                pixelflow_core::materialize_scalar(&manifold, black_box(angle), 0.0, &mut out);
                acc += out[0];
            }
            black_box(acc)
        })
    });
}

fn atan2_benchmark(c: &mut Criterion) {
    c.bench_function("cheby_atan2_throughput", |b| {
        let test_data: Vec<(f32, f32)> = (0..1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                let angle = t * 2.0 * PI;
                let r = 1.0 + t;
                (r * angle.sin(), r * angle.cos())
            })
            .collect();

        b.iter(|| {
            let manifold = Y.atan2(X);
            let mut acc = 0.0f32;
            for &(y, x) in &test_data {
                let mut out = [0.0f32; 16];
                pixelflow_core::materialize_scalar(&manifold, black_box(x), black_box(y), &mut out);
                acc += out[0];
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, sin_benchmark, cos_benchmark, atan2_benchmark);
criterion_main!(benches);
