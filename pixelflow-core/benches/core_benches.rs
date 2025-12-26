//! Comprehensive benchmarks for pixelflow-core
//!
//! Tests SIMD operations, manifold evaluation, and composition performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pixelflow_core::{
    combinators::{
        ambient_light_sh, cosine_lobe_sh, directional_light_sh, irradiance, Basis,
        Coefficients, CompressedManifold, Fix, ShBasis,
    },
    Field, Jet2, Manifold, ManifoldExt, PARALLELISM, X, Y, Z,
};

// ============================================================================
// SIMD Field Benchmarks
// ============================================================================

fn bench_field_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_creation");

    group.bench_function("from_f32_splat", |b| {
        b.iter(|| black_box(Field::from(3.14159f32)))
    });

    group.bench_function("sequential", |b| {
        b.iter(|| black_box(Field::sequential(0.5)))
    });

    group.finish();
}

fn bench_field_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_arithmetic");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let a = Field::sequential(1.0);
    let b = Field::sequential(2.0);

    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box(black_box(a) + black_box(b)))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box(black_box(a) - black_box(b)))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(black_box(a) * black_box(b)))
    });

    group.bench_function("div", |bencher| {
        bencher.iter(|| black_box(black_box(a) / black_box(b)))
    });

    group.bench_function("chained_mad", |bencher| {
        // Multiply-add chain: a * b + c
        let c = Field::from(0.5);
        bencher.iter(|| black_box(black_box(a) * black_box(b) + black_box(c)))
    });

    group.finish();
}

fn bench_field_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_math");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let a = Field::sequential(1.0);
    let b = Field::sequential(2.0);

    group.bench_function("sqrt", |bencher| {
        bencher.iter(|| {
            let val = black_box(a);
            black_box(val.sqrt())
        })
    });

    group.bench_function("abs", |bencher| {
        let neg = Field::from(-3.5);
        bencher.iter(|| {
            let val = black_box(neg);
            black_box(val.abs())
        })
    });

    group.bench_function("min", |bencher| {
        bencher.iter(|| black_box(black_box(a).min(black_box(b))))
    });

    group.bench_function("max", |bencher| {
        bencher.iter(|| black_box(black_box(a).max(black_box(b))))
    });

    group.finish();
}

fn bench_field_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_comparisons");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    // Use manifold comparisons which return Field when evaluated
    group.bench_function("lt_manifold", |bencher| {
        let m = X.lt(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("le_manifold", |bencher| {
        let m = X.le(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("gt_manifold", |bencher| {
        let m = X.gt(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("ge_manifold", |bencher| {
        let m = X.ge(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

fn bench_field_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_select");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("select_manifold", |bencher| {
        // if x < 2 then 1.0 else 0.0
        let m = X.lt(2.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

fn bench_field_bitwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_bitwise");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("and_manifold", |bencher| {
        // x > 0 AND x < 3
        let m = X.gt(0.0f32) & X.lt(3.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("or_manifold", |bencher| {
        // x < 1 OR x > 2
        let m = X.lt(1.0f32) | X.gt(2.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

// ============================================================================
// Manifold Evaluation Benchmarks
// ============================================================================

fn bench_manifold_constants(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_constants");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("f32_constant", |bencher| {
        let m = 42.0f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("X_variable", |bencher| {
        bencher.iter(|| black_box(X.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("Y_variable", |bencher| {
        bencher.iter(|| black_box(Y.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_simple");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("X_plus_Y", |bencher| {
        let m = X + Y;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("X_mul_Y", |bencher| {
        let m = X * Y;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("X_squared", |bencher| {
        let m = X * X;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    // FMA benchmark: X * Y + Z goes through MulAdd combinator
    group.bench_function("fma_X_mul_Y_plus_Z", |bencher| {
        let m = X * Y + Z; // This is MulAdd<X, Y, Z> - uses vfmadd instruction
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("distance_squared", |bencher| {
        // x² + y² - this is MulAdd<X, X, MulAdd<Y, Y, ...>> due to chaining
        let m = X * X + Y * Y;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("distance_from_origin", |bencher| {
        // √(x² + y²)
        let m = (X * X + Y * Y).sqrt();
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_circle(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_circle");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("unit_circle_sdf", |bencher| {
        // Signed distance: √(x² + y²) - 1
        let m = (X * X + Y * Y).sqrt() - 1.0f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("circle_inside_test", |bencher| {
        // x² + y² < 1
        let m = (X * X + Y * Y).lt(1.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_select");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("simple_select", |bencher| {
        // if x < 2 then 1.0 else 0.0
        let m = X.lt(2.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("circle_select", |bencher| {
        // if inside circle then 1.0 else 0.0
        let m = (X * X + Y * Y).lt(100.0f32).select(1.0f32, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("nested_select", |bencher| {
        // Nested: if x < 2 then (if y < 3 then 1 else 0.5) else 0
        let inner = Y.lt(3.0f32).select(1.0f32, 0.5f32);
        let m = X.lt(2.0f32).select(inner, 0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

fn bench_manifold_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_complex");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(5.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("polynomial_degree3", |bencher| {
        // x³ + 2x² - 5x + 3
        let m = X * X * X + X * X * 2.0f32 - X * 5.0f32 + 3.0f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("bilinear_interp", |bencher| {
        // Bilinear-like computation using manifold operations
        // x*y for corner blending pattern
        let m = X * Y * 3.0f32 + X * 0.5f32 + Y * 0.25f32;
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.bench_function("min_max_chain", |bencher| {
        let m = X.max(Y).min(10.0f32).max(0.0f32);
        bencher.iter(|| black_box(m.eval_raw(black_box(x), black_box(y), z, w)))
    });

    group.finish();
}

// ============================================================================
// Jet2 Auto-Differentiation Benchmarks
// ============================================================================

fn bench_jet2_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_creation");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let val = Field::sequential(1.0);

    group.bench_function("x_seeded", |bencher| {
        bencher.iter(|| black_box(Jet2::x(black_box(val))))
    });

    group.bench_function("y_seeded", |bencher| {
        bencher.iter(|| black_box(Jet2::y(black_box(val))))
    });

    group.bench_function("constant", |bencher| {
        bencher.iter(|| black_box(Jet2::constant(black_box(val))))
    });

    group.finish();
}

fn bench_jet2_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_arithmetic");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Jet2::x(Field::sequential(1.0));
    let y = Jet2::y(Field::from(2.0));

    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box(black_box(x) + black_box(y)))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box(black_box(x) - black_box(y)))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(black_box(x) * black_box(y)))
    });

    group.bench_function("div", |bencher| {
        bencher.iter(|| black_box(black_box(x) / black_box(y)))
    });

    group.finish();
}

fn bench_jet2_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_math");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Jet2::x(Field::sequential(1.0));
    let y = Jet2::y(Field::from(2.0));

    group.bench_function("sqrt", |bencher| {
        bencher.iter(|| black_box(black_box(x).sqrt()))
    });

    group.bench_function("abs", |bencher| {
        let neg = Jet2::x(Field::from(-3.5));
        bencher.iter(|| black_box(black_box(neg).abs()))
    });

    group.bench_function("min", |bencher| {
        bencher.iter(|| black_box(black_box(x).min(black_box(y))))
    });

    group.bench_function("max", |bencher| {
        bencher.iter(|| black_box(black_box(x).max(black_box(y))))
    });

    group.finish();
}

fn bench_jet2_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("jet2_gradient");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    group.bench_function("circle_sdf_gradient", |bencher| {
        // Gradient of √(x² + y²) - r
        // ∂f/∂x = x / √(x² + y²)
        // ∂f/∂y = y / √(x² + y²)
        bencher.iter(|| {
            let x = Jet2::x(Field::sequential(3.0));
            let y = Jet2::y(Field::from(4.0));
            let dist = (x * x + y * y).sqrt();
            black_box(dist)
        })
    });

    group.bench_function("polynomial_gradient", |bencher| {
        // Gradient of x³ + xy²
        // ∂f/∂x = 3x² + y²
        // ∂f/∂y = 2xy
        bencher.iter(|| {
            let x = Jet2::x(Field::sequential(2.0));
            let y = Jet2::y(Field::from(3.0));
            let result = x * x * x + x * y * y;
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Fix Combinator Benchmarks
// ============================================================================

fn bench_fix_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("fix_iteration");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    let x = Field::sequential(0.0);
    let y = Field::from(0.0);
    let z = Field::from(0.0);
    let w = Field::from(0.0);

    group.bench_function("converge_fast_all_lanes", |bencher| {
        // All lanes converge immediately: w >= 10 (seed = 100)
        use pixelflow_core::W;
        let fix = Fix {
            seed: 100.0f32,           // All lanes start at 100
            step: W + 1.0f32,         // Increment (never used if done)
            done: W.ge(10.0f32),      // All lanes done immediately
        };
        bencher.iter(|| black_box(fix.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("converge_10_iterations", |bencher| {
        use pixelflow_core::W;
        // Each iteration: w += 1, done when w >= 10
        let fix = Fix {
            seed: 0.0f32,
            step: W + 1.0f32,
            done: W.ge(10.0f32),
        };
        bencher.iter(|| black_box(fix.eval_raw(black_box(x), y, z, w)))
    });

    group.bench_function("converge_variable_lanes", |bencher| {
        use pixelflow_core::W;
        // Different lanes converge at different times based on x
        // seed = x, done when w >= 5
        let fix = Fix {
            seed: X,                   // Lanes start at [0, 1, 2, 3, ...]
            step: W + 1.0f32,
            done: W.ge(5.0f32),        // Each lane needs 5-x iterations
        };
        bencher.iter(|| black_box(fix.eval_raw(black_box(x), y, z, w)))
    });

    group.finish();
}

// ============================================================================
// Evaluation Throughput Benchmark
// ============================================================================

fn bench_evaluation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation_throughput");

    // Different sizes to measure scaling
    for &size in &[64, 256, 1024] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("circle_sdf_{}px", size), |bencher| {
            let m = (X * X + Y * Y).sqrt() - 50.0f32;
            let z = Field::from(0.0);
            let w = Field::from(0.0);

            bencher.iter(|| {
                let mut total = Field::from(0.0);
                let rows = size / PARALLELISM;
                for row in 0..rows {
                    let y = Field::from(row as f32);
                    let x = Field::sequential(0.0);
                    total = total + m.eval_raw(x, y, z, w);
                }
                black_box(total)
            })
        });
    }

    group.finish();
}

// ============================================================================
// Kernel Algebra Benchmarks (Spherical Harmonics Lighting)
// ============================================================================

fn bench_sh_basis_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("sh_basis_eval");

    group.bench_function("eval_at_scalar_9coeff", |bencher| {
        // Normalized direction
        let (x, y, z) = (0.577f32, 0.577f32, 0.577f32);
        bencher.iter(|| {
            black_box(<ShBasis<9> as Basis>::eval_at_scalar(
                black_box(x),
                black_box(y),
                black_box(z),
            ))
        })
    });

    group.throughput(Throughput::Elements(PARALLELISM as u64));
    group.bench_function("eval_at_vectorized_9coeff", |bencher| {
        let x = Field::sequential(0.0) * Field::from(0.1);
        let y = Field::from(0.577);
        let z = Field::from(0.577);
        bencher.iter(|| {
            black_box(<ShBasis<9> as Basis>::eval_at(
                black_box(x),
                black_box(y),
                black_box(z),
            ))
        })
    });

    group.finish();
}

fn bench_sh_coefficients(c: &mut Criterion) {
    let mut group = c.benchmark_group("sh_coefficients");

    // Create two SH coefficient sets
    let cosine = cosine_lobe_sh((0.0, 1.0, 0.0));
    let light = directional_light_sh((0.577, 0.577, 0.577), 1.0);

    group.bench_function("dot_product_9coeff", |bencher| {
        bencher.iter(|| black_box(black_box(&cosine.coeffs).dot(black_box(&light.coeffs))))
    });

    group.bench_function("clebsch_gordan_multiply_9coeff", |bencher| {
        use pixelflow_core::combinators::CG_ORDER_2;
        bencher.iter(|| {
            black_box(
                black_box(&cosine.coeffs).multiply(black_box(&light.coeffs), CG_ORDER_2),
            )
        })
    });

    group.finish();
}

fn bench_compressed_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("compressed_creation");

    group.bench_function("cosine_lobe_sh", |bencher| {
        let normal = (0.0f32, 1.0f32, 0.0f32);
        bencher.iter(|| black_box(cosine_lobe_sh(black_box(normal))))
    });

    group.bench_function("directional_light_sh", |bencher| {
        let dir = (0.577f32, 0.577f32, 0.577f32);
        bencher.iter(|| black_box(directional_light_sh(black_box(dir), 1.0)))
    });

    group.bench_function("ambient_light_sh", |bencher| {
        bencher.iter(|| black_box(ambient_light_sh(black_box(1.0))))
    });

    group.finish();
}

fn bench_irradiance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("irradiance");

    // Pre-create environment light
    let env = directional_light_sh((0.577, 0.577, 0.577), 1.0);

    group.bench_function("single_normal", |bencher| {
        let normal = (0.0f32, 1.0f32, 0.0f32);
        bencher.iter(|| black_box(irradiance(black_box(&env), black_box(normal))))
    });

    group.bench_function("full_pipeline_diffuse", |bencher| {
        // Full diffuse lighting: create cosine lobe, multiply with env, extract DC
        let normal = (0.0f32, 1.0f32, 0.0f32);
        bencher.iter(|| {
            let cosine = cosine_lobe_sh(black_box(normal));
            let lit = cosine.coeffs.dot(&env.coeffs);
            black_box(lit)
        })
    });

    group.finish();
}

fn bench_compressed_manifold(c: &mut Criterion) {
    let mut group = c.benchmark_group("compressed_manifold");
    group.throughput(Throughput::Elements(PARALLELISM as u64));

    // Create a directional light as CompressedManifold
    let light = directional_light_sh((0.577, 0.577, 0.577), 1.0);
    let manifold = CompressedManifold::new(light);

    let w = Field::from(0.0);

    group.bench_function("eval_vectorized", |bencher| {
        // Sample at multiple directions (hemisphere-ish)
        let x = Field::sequential(0.0) * Field::from(0.1);
        let y = Field::from(0.5);
        let z = Field::from(0.5);
        bencher.iter(|| {
            black_box(manifold.eval_raw(black_box(x), black_box(y), black_box(z), w))
        })
    });

    group.bench_function("eval_batch_16_directions", |bencher| {
        // More realistic: sample 16 directions (simulating hemisphere sampling)
        bencher.iter(|| {
            let mut total = Field::from(0.0);
            for i in 0..16 {
                let angle = (i as f32) * 0.392699; // π/8
                let x = Field::from(angle.cos());
                let y = Field::from(0.5);
                let z = Field::from(angle.sin());
                total = total + manifold.eval_raw(x, y, z, w);
            }
            black_box(total)
        })
    });

    group.finish();
}

fn bench_lighting_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("lighting_scenarios");

    group.bench_function("sky_dome_3_lights", |bencher| {
        // Realistic: sun + sky + ground bounce
        bencher.iter(|| {
            let sun = directional_light_sh((0.2, 0.9, 0.4), 2.0);
            let sky = ambient_light_sh(0.3);
            let ground = directional_light_sh((0.0, -1.0, 0.0), 0.1);

            // Combine lights (add coefficients)
            let combined = sun.add(&sky).add(&ground);

            // Sample at normal pointing up
            let normal = (0.0f32, 1.0f32, 0.0f32);
            black_box(irradiance(&combined, normal))
        })
    });

    group.bench_function("irradiance_grid_8x8", |bencher| {
        // Pre-bake environment
        let env = {
            let sun = directional_light_sh((0.2, 0.9, 0.4), 2.0);
            let sky = ambient_light_sh(0.3);
            sun.add(&sky)
        };

        bencher.iter(|| {
            let mut total = 0.0f32;
            // 8x8 grid of varying normals
            for i in 0..8 {
                for j in 0..8 {
                    let nx = (i as f32 - 3.5) * 0.2;
                    let ny = 0.8f32;
                    let nz = (j as f32 - 3.5) * 0.2;
                    // Normalize
                    let len = libm::sqrtf(nx * nx + ny * ny + nz * nz);
                    let normal = (nx / len, ny / len, nz / len);
                    total += irradiance(&env, normal);
                }
            }
            black_box(total)
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    kernel_benches,
    bench_sh_basis_eval,
    bench_sh_coefficients,
    bench_compressed_creation,
    bench_irradiance_computation,
    bench_compressed_manifold,
    bench_lighting_scenarios,
);

criterion_group!(
    field_benches,
    bench_field_creation,
    bench_field_arithmetic,
    bench_field_math,
    bench_field_comparisons,
    bench_field_select,
    bench_field_bitwise,
);

criterion_group!(
    manifold_benches,
    bench_manifold_constants,
    bench_manifold_simple,
    bench_manifold_circle,
    bench_manifold_select,
    bench_manifold_complex,
);

criterion_group!(
    jet2_benches,
    bench_jet2_creation,
    bench_jet2_arithmetic,
    bench_jet2_math,
    bench_jet2_gradient,
);

criterion_group!(fix_benches, bench_fix_iteration,);

criterion_group!(throughput_benches, bench_evaluation_throughput,);

criterion_main!(
    field_benches,
    manifold_benches,
    jet2_benches,
    fix_benches,
    throughput_benches,
    kernel_benches
);
