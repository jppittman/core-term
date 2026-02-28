//! Generate kernels for NNUE training data collection.
//!
//! Supports two generation modes:
//!
//! ## Forward Generation (FWD) - Original approach
//! Generates random expression trees directly.
//!
//! ## Backward Generation (BWD) - Lample & Charton 2019
//! Generates optimized expressions, then unfuses them to create
//! (unoptimized, optimized) training pairs. This approach:
//! - Guarantees every example has an optimization opportunity
//! - Creates natural distribution of optimization patterns
//! - Based on "Deep Learning for Symbolic Mathematics" (ICLR 2020)
//!
//! Usage:
//!   # Forward generation (original)
//!   cargo run -p pixelflow-ml --example gen_kernels --features training -- --count 500
//!
//!   # Backward generation (recommended)
//!   cargo run -p pixelflow-ml --example gen_kernels --features training -- --count 500 --mode bwd
//!
//! Output files:
//!   - pixelflow-ml/benches/generated_kernels.rs (benchmark)
//!   - pixelflow-ml/data/generated_exprs.txt (FWD mode)
//!   - pixelflow-ml/data/bwd_training_pairs.txt (BWD mode)

use pixelflow_ml::nnue::{
    BwdGenConfig, BwdGenerator, Expr, ExprGenConfig, ExprGenerator, OpType, count_fused_ops,
};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq)]
enum GenerationMode {
    Forward,
    Backward,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut count = 500;
    let mut seed = 12345u64;
    let mut mode = GenerationMode::Backward; // Default to BWD per Lample & Charton

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
            "--mode" => {
                i += 1;
                if i < args.len() {
                    mode = match args[i].as_str() {
                        "fwd" | "forward" => GenerationMode::Forward,
                        "bwd" | "backward" => GenerationMode::Backward,
                        _ => panic!("Invalid mode: use 'fwd' or 'bwd'"),
                    };
                }
            }
            "--help" | "-h" => {
                println!("Usage: gen_kernels [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --count N     Number of kernels to generate (default: 500)");
                println!("  --seed N      Random seed (default: 12345)");
                println!("  --mode MODE   Generation mode: 'fwd' or 'bwd' (default: bwd)");
                println!();
                println!("Modes:");
                println!("  fwd  Forward generation - random expression trees");
                println!("  bwd  Backward generation - optimized → unoptimized pairs");
                println!("       (Lample & Charton, ICLR 2020)");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let mode_str = match mode {
        GenerationMode::Forward => "forward (FWD)",
        GenerationMode::Backward => "backward (BWD)",
    };
    println!(
        "Generating {} kernels with seed {} using {} mode",
        count, seed, mode_str
    );

    // Find workspace root
    let workspace_root = find_workspace_root();
    let bench_path = workspace_root.join("pixelflow-ml/benches/generated_kernels.rs");
    let data_dir = workspace_root.join("pixelflow-ml/data");

    // Create data directory
    fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    match mode {
        GenerationMode::Forward => generate_forward(count, seed, &bench_path, &data_dir),
        GenerationMode::Backward => generate_backward(count, seed, &bench_path, &data_dir),
    }
}

/// Forward generation: random expression trees (original approach).
fn generate_forward(count: usize, seed: u64, bench_path: &PathBuf, data_dir: &PathBuf) {
    let data_path = data_dir.join("generated_exprs.txt");

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

    println!("Generated {} unique kernels (FWD)", kernels.len());

    // Generate benchmark file
    let bench_code = generate_benchmark_file(&kernels);
    let mut bench_file = File::create(bench_path).expect("Failed to create benchmark file");
    bench_file
        .write_all(bench_code.as_bytes())
        .expect("Failed to write benchmark");
    println!("Wrote benchmark to {}", bench_path.display());

    // Generate expr data file
    let mut data_file = File::create(&data_path).expect("Failed to create data file");
    for (name, expr) in &kernels {
        let expr_str = expr_to_string(expr);
        writeln!(
            data_file,
            "{}|{}|{}|{}",
            name,
            expr.node_count(),
            expr.depth(),
            expr_str
        )
        .expect("Failed to write data");
    }
    println!("Wrote expr data to {}", data_path.display());

    print_fwd_stats(&kernels);
}

/// Backward generation: optimized → unoptimized pairs (Lample & Charton 2019).
fn generate_backward(count: usize, seed: u64, bench_path: &PathBuf, data_dir: &PathBuf) {
    let pairs_path = data_dir.join("bwd_training_pairs.txt");

    // Configure backward generator with high fused op probability
    let config = BwdGenConfig {
        max_depth: 5,
        leaf_prob: 0.25,
        num_vars: 4,
        fused_op_prob: 0.5,   // 50% chance of fused ops
        max_unfuse_passes: 3, // Apply unfusing multiple times
        unfuse_prob: 0.8,     // High probability of applying unfusing
    };

    let mut generator = BwdGenerator::new(seed, config);

    // Generate training pairs
    let mut pairs: Vec<(String, Expr, Expr)> = Vec::new(); // (name, unoptimized, optimized)
    let mut seen = std::collections::HashSet::new();

    while pairs.len() < count {
        let pair = generator.generate();

        // Filter: require meaningful difference
        let opt_nodes = pair.optimized.node_count();
        let unopt_nodes = pair.unoptimized.node_count();

        // Skip if no unfusing happened (identical expressions)
        if opt_nodes >= unopt_nodes || opt_nodes < 2 {
            continue;
        }

        // Convert to Manifold code for deduplication
        let unopt_code = expr_to_manifold(&pair.unoptimized);
        if seen.contains(&unopt_code) {
            continue;
        }
        seen.insert(unopt_code);

        let name = format!("bwd{:04}", pairs.len());
        pairs.push((name, pair.unoptimized, pair.optimized));
    }

    println!("Generated {} unique training pairs (BWD)", pairs.len());

    // Generate benchmark file using UNOPTIMIZED expressions
    // (we benchmark what the model will see as input)
    let bench_kernels: Vec<(String, Expr)> = pairs
        .iter()
        .map(|(name, unopt, _)| (name.clone(), unopt.clone()))
        .collect();

    let bench_code = generate_benchmark_file(&bench_kernels);
    let mut bench_file = File::create(bench_path).expect("Failed to create benchmark file");
    bench_file
        .write_all(bench_code.as_bytes())
        .expect("Failed to write benchmark");
    println!("Wrote benchmark to {}", bench_path.display());

    // Generate training pairs data file
    // Format: name|unopt_nodes|opt_nodes|savings|unopt_expr|opt_expr
    let mut data_file = File::create(&pairs_path).expect("Failed to create data file");

    // Write header comment
    writeln!(data_file, "# BWD Training Pairs - Lample & Charton style").unwrap();
    writeln!(
        data_file,
        "# Format: name|unopt_nodes|opt_nodes|node_savings|unopt_expr|opt_expr"
    )
    .unwrap();
    writeln!(
        data_file,
        "# node_savings = unopt_nodes - opt_nodes (larger = more optimization)"
    )
    .unwrap();
    writeln!(data_file, "#").unwrap();

    for (name, unoptimized, optimized) in &pairs {
        let unopt_nodes = unoptimized.node_count();
        let opt_nodes = optimized.node_count();
        let savings = unopt_nodes.saturating_sub(opt_nodes);

        let unopt_str = expr_to_string(unoptimized);
        let opt_str = expr_to_string(optimized);

        writeln!(
            data_file,
            "{}|{}|{}|{}|{}|{}",
            name, unopt_nodes, opt_nodes, savings, unopt_str, opt_str
        )
        .expect("Failed to write data");
    }
    println!("Wrote training pairs to {}", pairs_path.display());

    print_bwd_stats(&pairs);
}

fn print_fwd_stats(kernels: &[(String, Expr)]) {
    let total_nodes: usize = kernels.iter().map(|(_, e)| e.node_count()).sum();
    let avg_nodes = total_nodes as f64 / kernels.len() as f64;
    let max_depth = kernels.iter().map(|(_, e)| e.depth()).max().unwrap_or(0);

    println!("\nStats (FWD):");
    println!("  Total kernels: {}", kernels.len());
    println!("  Average nodes: {:.1}", avg_nodes);
    println!("  Max depth: {}", max_depth);
    println!("\nTo run benchmarks:");
    println!("  cargo bench -p pixelflow-ml --bench generated_kernels");
}

fn print_bwd_stats(pairs: &[(String, Expr, Expr)]) {
    let total_unopt: usize = pairs.iter().map(|(_, u, _)| u.node_count()).sum();
    let total_opt: usize = pairs.iter().map(|(_, _, o)| o.node_count()).sum();
    let total_fused: usize = pairs.iter().map(|(_, _, o)| count_fused_ops(o)).sum();

    let avg_unopt = total_unopt as f64 / pairs.len() as f64;
    let avg_opt = total_opt as f64 / pairs.len() as f64;
    let avg_fused = total_fused as f64 / pairs.len() as f64;
    let avg_savings = avg_unopt - avg_opt;

    let max_savings = pairs
        .iter()
        .map(|(_, u, o)| u.node_count().saturating_sub(o.node_count()))
        .max()
        .unwrap_or(0);

    println!("\nStats (BWD - Lample & Charton style):");
    println!("  Total pairs: {}", pairs.len());
    println!("  Avg unoptimized nodes: {:.1}", avg_unopt);
    println!("  Avg optimized nodes: {:.1}", avg_opt);
    println!("  Avg node savings: {:.1}", avg_savings);
    println!("  Max node savings: {}", max_savings);
    println!("  Avg fused ops per optimized: {:.2}", avg_fused);
    println!("\nThe training signal:");
    println!("  Input: unoptimized expression features");
    println!("  Target: learn to recognize optimization opportunities");
    println!("\nTo run benchmarks:");
    println!("  cargo bench -p pixelflow-ml --bench generated_kernels");
}

/// Find workspace root by looking for Cargo.toml with [workspace]
fn find_workspace_root() -> PathBuf {
    let mut current = env::current_dir().expect("Failed to get current directory");

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists()
            && let Ok(contents) = fs::read_to_string(&cargo_toml)
                && contents.contains("[workspace]") {
                    return current;
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
        Expr::Ternary(op, a, b, c) => format!(
            "T{:?}({},{},{})",
            op,
            expr_to_string(a),
            expr_to_string(b),
            expr_to_string(c)
        ),
    }
}

/// Generate the complete benchmark file.
fn generate_benchmark_file(kernels: &[(String, Expr)]) -> String {
    let mut code = String::new();

    // Header
    code.push_str(
        r#"//! Auto-generated kernels for NNUE training data collection.
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

"#,
    );

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
    code.push_str(
        r#"    group.finish();
}

criterion_group!(
    name = generated;
    config = Criterion::default().sample_size(100);
    targets = bench_generated_kernels,
);

criterion_main!(generated);
"#,
    );

    code
}
