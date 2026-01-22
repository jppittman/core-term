//! Generate random kernels for NNUE training data collection.
//!
//! This generates:
//! 1. A Rust benchmark file with Manifold kernels
//! 2. A matching data file with Expr trees for feature extraction
//!
//! Usage:
//!   cargo run -p pixelflow-ml --example gen_kernels --features training -- --count 500
//!
//! Output files:
//!   - pixelflow-ml/benches/generated_kernels.rs (benchmark)
//!   - pixelflow-ml/data/generated_exprs.bin (serialized Expr trees)

use pixelflow_ml::nnue::{Expr, ExprGenConfig, ExprGenerator, OpType};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse --count N argument
    let mut count = 500;
    let mut seed = 12345u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--count" => {
                i += 1;
                if i < args.len() {
                    count = args[i].parse().expect("Invalid count");
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().expect("Invalid seed");
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("Generating {} kernels with seed {}", count, seed);

    // Find workspace root
    let workspace_root = find_workspace_root();
    let bench_path = workspace_root.join("pixelflow-ml/benches/generated_kernels.rs");
    let data_dir = workspace_root.join("pixelflow-ml/data");
    let data_path = data_dir.join("generated_exprs.txt");

    // Create data directory
    fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    // Configure generator for diverse expressions
    let config = ExprGenConfig {
        max_depth: 6,
        leaf_prob: 0.25,
        num_vars: 4,
        include_fused: false, // Skip MulAdd/MulRsqrt for simpler manifold code
    };

    let mut generator = ExprGenerator::new(seed, config);

    // Generate unique expressions (deduplicate by string representation)
    let mut kernels: Vec<(String, Expr)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    while kernels.len() < count {
        let expr = generator.generate();

        // Filter: skip trivial expressions
        if expr.node_count() < 3 {
            continue;
        }

        // Convert to Manifold code
        let code = expr_to_manifold(&expr);

        // Deduplicate
        if seen.contains(&code) {
            continue;
        }
        seen.insert(code.clone());

        let name = format!("k{:04}", kernels.len());
        kernels.push((name, expr));
    }

    println!("Generated {} unique kernels", kernels.len());

    // Generate benchmark file
    let bench_code = generate_benchmark_file(&kernels);
    let mut bench_file = File::create(&bench_path).expect("Failed to create benchmark file");
    bench_file.write_all(bench_code.as_bytes()).expect("Failed to write benchmark");
    println!("Wrote benchmark to {}", bench_path.display());

    // Generate expr data file (simple text format for now)
    let mut data_file = File::create(&data_path).expect("Failed to create data file");
    for (name, expr) in &kernels {
        // Format: name|node_count|depth|expr_string
        let expr_str = expr_to_string(expr);
        writeln!(data_file, "{}|{}|{}|{}", name, expr.node_count(), expr.depth(), expr_str)
            .expect("Failed to write data");
    }
    println!("Wrote expr data to {}", data_path.display());

    // Print summary stats
    let total_nodes: usize = kernels.iter().map(|(_, e)| e.node_count()).sum();
    let avg_nodes = total_nodes as f64 / kernels.len() as f64;
    let max_depth = kernels.iter().map(|(_, e)| e.depth()).max().unwrap_or(0);

    println!("\nStats:");
    println!("  Total kernels: {}", kernels.len());
    println!("  Average nodes: {:.1}", avg_nodes);
    println!("  Max depth: {}", max_depth);
    println!("\nTo run benchmarks:");
    println!("  cargo bench -p pixelflow-ml --bench generated_kernels");
}

/// Find workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return current;
                }
            }
        }
        if !current.pop() {
            panic!("Could not find workspace root");
        }
    }
}

/// Check if an expression is purely constant (no variables)
fn is_pure_const(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) => false,
        Expr::Const(_) => true,
        Expr::Unary(_, a) => is_pure_const(a),
        Expr::Binary(_, a, b) => is_pure_const(a) && is_pure_const(b),
        Expr::Ternary(_, a, b, c) => is_pure_const(a) && is_pure_const(b) && is_pure_const(c),
    }
}

/// Convert an Expr to Manifold Rust code string.
fn expr_to_manifold(expr: &Expr) -> String {
    match expr {
        Expr::Var(0) => "X".to_string(),
        Expr::Var(1) => "Y".to_string(),
        Expr::Var(2) => "Z".to_string(),
        Expr::Var(3) => "W".to_string(),
        Expr::Var(i) => format!("V{}", i),

        Expr::Const(v) => {
            if *v == 0.0 {
                "0.0f32".to_string()
            } else if *v == 1.0 {
                "1.0f32".to_string()
            } else if *v == -1.0 {
                "(-1.0f32)".to_string()
            } else {
                format!("({:.6}f32)", v)
            }
        }

        Expr::Unary(op, a) => {
            let a_code = expr_to_manifold(a);
            match op {
                OpType::Neg => format!("({} * -1.0f32)", a_code),
                OpType::Sqrt => format!("({}).sqrt()", a_code),
                OpType::Rsqrt => format!("(1.0f32 / ({}).sqrt())", a_code),
                OpType::Abs => format!("({}).abs()", a_code),
                _ => panic!("Unexpected unary op: {:?}", op),
            }
        }

        Expr::Binary(op, a, b) => {
            let a_code = expr_to_manifold(a);
            let b_code = expr_to_manifold(b);
            match op {
                OpType::Add => format!("({} + {})", a_code, b_code),
                OpType::Sub => format!("({} - {})", a_code, b_code),
                OpType::Mul => format!("({} * {})", a_code, b_code),
                OpType::Div => format!("({} / {})", a_code, b_code),
                // For min/max, ensure Manifold type is the receiver (not f32)
                OpType::Min => {
                    if is_pure_const(a) && !is_pure_const(b) {
                        // Swap so Manifold is receiver: b.min(a)
                        format!("({}).min({})", b_code, a_code)
                    } else {
                        format!("({}).min({})", a_code, b_code)
                    }
                }
                OpType::Max => {
                    if is_pure_const(a) && !is_pure_const(b) {
                        // Swap so Manifold is receiver: b.max(a)
                        format!("({}).max({})", b_code, a_code)
                    } else {
                        format!("({}).max({})", a_code, b_code)
                    }
                }
                OpType::MulRsqrt => format!("({} * (1.0f32 / ({}).sqrt()))", a_code, b_code),
                _ => panic!("Unexpected binary op: {:?}", op),
            }
        }

        Expr::Ternary(op, a, b, c) => {
            let a_code = expr_to_manifold(a);
            let b_code = expr_to_manifold(b);
            let c_code = expr_to_manifold(c);
            match op {
                OpType::MulAdd => format!("({} * {} + {})", a_code, b_code, c_code),
                _ => panic!("Unexpected ternary op: {:?}", op),
            }
        }
    }
}

/// Convert Expr to a parseable string format.
fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Var(i) => format!("V{}", i),
        Expr::Const(v) => format!("C{}", v),
        Expr::Unary(op, a) => format!("U{:?}({})", op, expr_to_string(a)),
        Expr::Binary(op, a, b) => format!("B{:?}({},{})", op, expr_to_string(a), expr_to_string(b)),
        Expr::Ternary(op, a, b, c) => format!("T{:?}({},{},{})", op, expr_to_string(a), expr_to_string(b), expr_to_string(c)),
    }
}

/// Generate the complete benchmark file.
fn generate_benchmark_file(kernels: &[(String, Expr)]) -> String {
    let mut code = String::new();

    // Header
    code.push_str(r#"//! Auto-generated kernels for NNUE training data collection.
//!
//! Generated by: cargo run -p pixelflow-ml --example gen_kernels --features training
//!
//! DO NOT EDIT MANUALLY - regenerate with the command above.

use criterion::{Criterion, BenchmarkId, black_box, criterion_group, criterion_main};
use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y, Z, W};

fn bench_generated_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("generated_kernels");
    group.sample_size(100);

    let xf = Field::sequential(1.0);
    let yf = Field::from(2.0);
    let zf = Field::from(3.0);
    let wf = Field::from(0.5);

"#);

    // Generate each kernel benchmark
    for (name, expr) in kernels {
        let manifold_code = expr_to_manifold(expr);

        code.push_str(&format!(r#"    // {} - {} nodes, depth {}
    {{
        let m = {};
        group.bench_function("{}", |b| {{
            b.iter(|| black_box(m.eval((black_box(xf), black_box(yf), black_box(zf), black_box(wf)))))
        }});
    }}

"#, name, expr.node_count(), expr.depth(), manifold_code, name));
    }

    // Footer
    code.push_str(r#"    group.finish();
}

criterion_group!(
    name = generated;
    config = Criterion::default().sample_size(100);
    targets = bench_generated_kernels,
);

criterion_main!(generated);
"#);

    code
}
