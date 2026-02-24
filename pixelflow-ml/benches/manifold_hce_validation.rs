//! Validate HCE predictions against real compiled Manifold kernel performance
//!
//! This benchmark uses hand-written Manifold kernels (compiled to SIMD) and
//! compares their actual performance against HCE cost predictions.
//!
//! Run with: cargo bench -p pixelflow-ml --bench manifold_hce_validation

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y, Z};
use pixelflow_ml::nnue::{Expr, OpType};
use pixelflow_ml::evaluator::{extract_expr_features, default_expr_weights};

// ============================================================================
// Kernel Definitions: Manifold (compiled) + Expr (for HCE)
// ============================================================================

/// Kernel 1: Simple circle SDF
/// sqrt(x² + y²) - r
mod circle_sdf {
    use super::*;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        let r = 50.0f32;
        (X * X + Y * Y).sqrt() - r
    }

    pub fn expr() -> Expr {
        let r = 50.0f32;
        Expr::Binary(
            OpType::Sub,
            Box::new(Expr::Unary(
                OpType::Sqrt,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
                    Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(1)), Box::new(Expr::Var(1)))),
                )),
            )),
            Box::new(Expr::Const(r)),
        )
    }
}

/// Kernel 2: 3D distance (more complex)
/// sqrt(x² + y² + z²)
mod distance_3d {
    use super::*;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        (X * X + Y * Y + Z * Z).sqrt()
    }

    pub fn expr() -> Expr {
        Expr::Unary(
            OpType::Sqrt,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
                    Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(1)), Box::new(Expr::Var(1)))),
                )),
                Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(2)), Box::new(Expr::Var(2)))),
            )),
        )
    }
}

/// Kernel 3: Polynomial (no sqrt/div)
/// x³ + 2x²y + xy² + y³
mod polynomial {
    use super::*;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        X * X * X + X * X * Y * 2.0f32 + X * Y * Y + Y * Y * Y
    }

    pub fn expr() -> Expr {
        // x³ + 2x²y + xy² + y³
        let x = || Expr::Var(0);
        let y = || Expr::Var(1);
        let mul = |a, b| Expr::Binary(OpType::Mul, Box::new(a), Box::new(b));
        let add = |a, b| Expr::Binary(OpType::Add, Box::new(a), Box::new(b));

        // x³
        let x3 = mul(mul(x(), x()), x());
        // 2x²y
        let x2y_2 = mul(mul(mul(x(), x()), y()), Expr::Const(2.0));
        // xy²
        let xy2 = mul(mul(x(), y()), y());
        // y³
        let y3 = mul(mul(y(), y()), y());

        add(add(add(x3, x2y_2), xy2), y3)
    }
}

/// Kernel 4: Normalization (has division)
/// x / sqrt(x² + y²)
mod normalize {
    use super::*;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        X / (X * X + Y * Y).sqrt()
    }

    pub fn expr() -> Expr {
        Expr::Binary(
            OpType::Div,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Unary(
                OpType::Sqrt,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(0)))),
                    Box::new(Expr::Binary(OpType::Mul, Box::new(Expr::Var(1)), Box::new(Expr::Var(1)))),
                )),
            )),
        )
    }
}

/// Kernel 5: Multiple sqrts (expensive)
/// sqrt(x) + sqrt(y) + sqrt(z)
mod multi_sqrt {
    use super::*;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        X.sqrt() + Y.sqrt() + Z.sqrt()
    }

    pub fn expr() -> Expr {
        Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Unary(OpType::Sqrt, Box::new(Expr::Var(0)))),
                Box::new(Expr::Unary(OpType::Sqrt, Box::new(Expr::Var(1)))),
            )),
            Box::new(Expr::Unary(OpType::Sqrt, Box::new(Expr::Var(2)))),
        )
    }
}

/// Kernel 6: Deep chain (sequential dependencies)
/// ((((x + y) * z) + x) * y) + z
mod deep_chain {
    use super::*;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        ((((X + Y) * Z) + X) * Y) + Z
    }

    pub fn expr() -> Expr {
        let x = || Expr::Var(0);
        let y = || Expr::Var(1);
        let z = || Expr::Var(2);
        let mul = |a, b| Expr::Binary(OpType::Mul, Box::new(a), Box::new(b));
        let add = |a, b| Expr::Binary(OpType::Add, Box::new(a), Box::new(b));

        // ((((x + y) * z) + x) * y) + z
        let t1 = add(x(), y());
        let t2 = mul(t1, z());
        let t3 = add(t2, x());
        let t4 = mul(t3, y());
        add(t4, z())
    }
}

/// Kernel 7: Wide tree (parallel-friendly)
/// (x*x + y*y) + (z*z + w*w) where w=x for simplicity
mod wide_tree {
    use super::*;
    use pixelflow_core::W;

    pub fn manifold() -> impl Manifold<(Field, Field, Field, Field), Output = Field> {
        (X * X + Y * Y) + (Z * Z + W * W)
    }

    pub fn expr() -> Expr {
        let v = |i| Expr::Var(i);
        let mul = |a, b| Expr::Binary(OpType::Mul, Box::new(a), Box::new(b));
        let add = |a, b| Expr::Binary(OpType::Add, Box::new(a), Box::new(b));

        // (x*x + y*y) + (z*z + w*w)
        let left = add(mul(v(0), v(0)), mul(v(1), v(1)));
        let right = add(mul(v(2), v(2)), mul(v(3), v(3)));
        add(left, right)
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct KernelData {
    name: &'static str,
    expr: Expr,
    hce_cost: i32,
    critical_path: i32,
}

fn collect_kernel_data() -> Vec<KernelData> {
    let hce = default_expr_weights();

    let kernels = vec![
        ("circle_sdf", circle_sdf::expr()),
        ("distance_3d", distance_3d::expr()),
        ("polynomial", polynomial::expr()),
        ("normalize", normalize::expr()),
        ("multi_sqrt", multi_sqrt::expr()),
        ("deep_chain", deep_chain::expr()),
        ("wide_tree", wide_tree::expr()),
    ];

    kernels
        .into_iter()
        .map(|(name, expr)| {
            let features = extract_expr_features(&expr);
            KernelData {
                name,
                expr,
                hce_cost: hce.evaluate_linear(&features),
                critical_path: features.critical_path,
            }
        })
        .collect()
}

fn bench_manifold_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifold_kernels");

    let x = Field::sequential(1.0);
    let y = Field::from(2.0);
    let z = Field::from(3.0);
    let w = Field::from(0.5);

    // Print HCE predictions
    let kernel_data = collect_kernel_data();
    println!("\n=== HCE Predictions ===");
    for kd in &kernel_data {
        println!("  {}: HCE={}, critical_path={}, nodes={}",
            kd.name, kd.hce_cost, kd.critical_path, kd.expr.node_count());
    }
    println!();

    // Benchmark each kernel
    {
        let m = circle_sdf::manifold();
        group.bench_function("circle_sdf", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), z, w))))
        });
    }

    {
        let m = distance_3d::manifold();
        group.bench_function("distance_3d", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), black_box(z), w))))
        });
    }

    {
        let m = polynomial::manifold();
        group.bench_function("polynomial", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), z, w))))
        });
    }

    {
        let m = normalize::manifold();
        group.bench_function("normalize", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), z, w))))
        });
    }

    {
        let m = multi_sqrt::manifold();
        group.bench_function("multi_sqrt", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), black_box(z), w))))
        });
    }

    {
        let m = deep_chain::manifold();
        group.bench_function("deep_chain", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), black_box(z), w))))
        });
    }

    {
        let m = wide_tree::manifold();
        group.bench_function("wide_tree", |b| {
            b.iter(|| black_box(m.eval((black_box(x), black_box(y), black_box(z), black_box(w)))))
        });
    }

    group.finish();
}

fn bench_hce_ranking_validation(_c: &mut Criterion) {
    // After benchmarking, collect results and compare rankings
    println!("\n=== Ranking Validation ===");
    println!("Compare HCE ranking vs actual runtime ranking.");
    println!("If HCE is accurate, lower HCE cost should correlate with faster runtime.");
}

criterion_group!(
    name = manifold_validation;
    config = Criterion::default().sample_size(200);
    targets = bench_manifold_kernels, bench_hce_ranking_validation,
);

criterion_main!(manifold_validation);
