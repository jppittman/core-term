//! Cost Model Weight Optimizer
//!
//! This example demonstrates learning optimal CostModel weights from benchmark data.
//!
//! The algorithm:
//! 1. Start with initial weight estimates
//! 2. For each (unopt_expr, opt_expr, observed_speedup) pair:
//!    - Compute predicted cost ratio from current weights
//!    - Compare to actual benchmark ratio
//!    - Update weights to minimize prediction error
//! 3. Use gradient descent or Bayesian optimization to find optimal weights
//!
//! Run with: cargo run -p pixelflow-search --example cost_weight_optimizer
//!
//! In production, this would read benchmark results from:
//!   target/criterion/cost_training/*/estimates.json

use pixelflow_search::egraph::CostModel;

/// A benchmark observation: two equivalent expressions with measured costs
#[derive(Debug, Clone)]
struct BenchmarkObservation {
    name: String,
    /// Number of each operation type in unoptimized expr
    unopt_ops: OpCounts,
    /// Number of each operation type in optimized expr
    opt_ops: OpCounts,
    /// Measured time ratio (unopt_ns / opt_ns). > 1 means opt is faster.
    actual_speedup: f64,
}

/// Count of each operation type in an expression
#[derive(Debug, Clone, Default)]
struct OpCounts {
    add: usize,
    sub: usize,
    mul: usize,
    div: usize,
    neg: usize,
    sqrt: usize,
    recip: usize,
    rsqrt: usize,
    abs: usize,
    min: usize,
    max: usize,
    mul_add: usize,
}

impl OpCounts {
    /// Compute total cost with given weights
    fn total_cost(&self, w: &Weights) -> f64 {
        self.add as f64 * w.add
            + self.sub as f64 * w.sub
            + self.mul as f64 * w.mul
            + self.div as f64 * w.div
            + self.neg as f64 * w.neg
            + self.sqrt as f64 * w.sqrt
            + self.recip as f64 * w.recip
            + self.rsqrt as f64 * w.rsqrt
            + self.abs as f64 * w.abs
            + self.min as f64 * w.min
            + self.max as f64 * w.max
            + self.mul_add as f64 * w.mul_add
    }
}

/// Trainable weights (float version for gradient descent)
#[derive(Debug, Clone)]
struct Weights {
    add: f64,
    sub: f64,
    mul: f64,
    div: f64,
    neg: f64,
    sqrt: f64,
    recip: f64,
    rsqrt: f64,
    abs: f64,
    min: f64,
    max: f64,
    mul_add: f64,
}

impl Default for Weights {
    fn default() -> Self {
        // Start with CostModel::fully_optimized() values
        Self {
            add: 4.0,
            sub: 4.0,
            mul: 5.0,
            div: 15.0,
            neg: 1.0,
            sqrt: 15.0,
            recip: 4.0,
            rsqrt: 4.0,
            abs: 1.0,
            min: 4.0,
            max: 4.0,
            mul_add: 5.0,
        }
    }
}

impl Weights {
    /// Convert to CostModel (quantize to integers)
    fn to_cost_model(&self) -> CostModel {
        CostModel {
            add: self.add.round().max(1.0) as usize,
            sub: self.sub.round().max(1.0) as usize,
            mul: self.mul.round().max(1.0) as usize,
            div: self.div.round().max(1.0) as usize,
            neg: self.neg.round().max(1.0) as usize,
            sqrt: self.sqrt.round().max(1.0) as usize,
            recip: self.recip.round().max(1.0) as usize,
            rsqrt: self.rsqrt.round().max(1.0) as usize,
            abs: self.abs.round().max(1.0) as usize,
            min: self.min.round().max(1.0) as usize,
            max: self.max.round().max(1.0) as usize,
            mul_add: self.mul_add.round().max(1.0) as usize,
            depth_threshold: 32,
            depth_penalty: 100,
        }
    }

    /// Update weights using gradient descent
    fn update(&mut self, gradient: &Weights, lr: f64) {
        self.add -= lr * gradient.add;
        self.sub -= lr * gradient.sub;
        self.mul -= lr * gradient.mul;
        self.div -= lr * gradient.div;
        self.neg -= lr * gradient.neg;
        self.sqrt -= lr * gradient.sqrt;
        self.recip -= lr * gradient.recip;
        self.rsqrt -= lr * gradient.rsqrt;
        self.abs -= lr * gradient.abs;
        self.min -= lr * gradient.min;
        self.max -= lr * gradient.max;
        self.mul_add -= lr * gradient.mul_add;

        // Clamp weights to reasonable bounds
        self.clamp(1.0, 50.0);
    }

    fn clamp(&mut self, min: f64, max: f64) {
        self.add = self.add.clamp(min, max);
        self.sub = self.sub.clamp(min, max);
        self.mul = self.mul.clamp(min, max);
        self.div = self.div.clamp(min, max);
        self.neg = self.neg.clamp(min, max);
        self.sqrt = self.sqrt.clamp(min, max);
        self.recip = self.recip.clamp(min, max);
        self.rsqrt = self.rsqrt.clamp(min, max);
        self.abs = self.abs.clamp(min, max);
        self.min = self.min.clamp(min, max);
        self.max = self.max.clamp(min, max);
        self.mul_add = self.mul_add.clamp(min, max);
    }
}

/// Compute loss: squared error between predicted and actual speedup
fn compute_loss(obs: &BenchmarkObservation, weights: &Weights) -> f64 {
    let unopt_cost = obs.unopt_ops.total_cost(weights);
    let opt_cost = obs.opt_ops.total_cost(weights);

    // Predicted speedup = unopt_cost / opt_cost
    // Avoid division by zero
    let predicted_speedup = if opt_cost > 0.1 {
        unopt_cost / opt_cost
    } else {
        1.0
    };

    let error = predicted_speedup - obs.actual_speedup;
    error * error
}

/// Compute gradient of loss with respect to weights
fn compute_gradient(obs: &BenchmarkObservation, weights: &Weights) -> Weights {
    let unopt_cost = obs.unopt_ops.total_cost(weights);
    let opt_cost = obs.opt_ops.total_cost(weights);

    let predicted_speedup = if opt_cost > 0.1 {
        unopt_cost / opt_cost
    } else {
        1.0
    };

    let error = predicted_speedup - obs.actual_speedup;

    // d(loss)/d(w) = 2 * error * d(predicted_speedup)/d(w)
    // d(predicted_speedup)/d(w) = d(unopt_cost/opt_cost)/d(w)
    //   = (d_unopt/d(w)) / opt_cost - unopt_cost * (d_opt/d(w)) / opt_cost^2

    let scale = 2.0 * error;
    let inv_opt = if opt_cost > 0.1 { 1.0 / opt_cost } else { 0.0 };
    let inv_opt_sq = inv_opt * inv_opt;

    Weights {
        add: scale * (obs.unopt_ops.add as f64 * inv_opt - unopt_cost * obs.opt_ops.add as f64 * inv_opt_sq),
        sub: scale * (obs.unopt_ops.sub as f64 * inv_opt - unopt_cost * obs.opt_ops.sub as f64 * inv_opt_sq),
        mul: scale * (obs.unopt_ops.mul as f64 * inv_opt - unopt_cost * obs.opt_ops.mul as f64 * inv_opt_sq),
        div: scale * (obs.unopt_ops.div as f64 * inv_opt - unopt_cost * obs.opt_ops.div as f64 * inv_opt_sq),
        neg: scale * (obs.unopt_ops.neg as f64 * inv_opt - unopt_cost * obs.opt_ops.neg as f64 * inv_opt_sq),
        sqrt: scale * (obs.unopt_ops.sqrt as f64 * inv_opt - unopt_cost * obs.opt_ops.sqrt as f64 * inv_opt_sq),
        recip: scale * (obs.unopt_ops.recip as f64 * inv_opt - unopt_cost * obs.opt_ops.recip as f64 * inv_opt_sq),
        rsqrt: scale * (obs.unopt_ops.rsqrt as f64 * inv_opt - unopt_cost * obs.opt_ops.rsqrt as f64 * inv_opt_sq),
        abs: scale * (obs.unopt_ops.abs as f64 * inv_opt - unopt_cost * obs.opt_ops.abs as f64 * inv_opt_sq),
        min: scale * (obs.unopt_ops.min as f64 * inv_opt - unopt_cost * obs.opt_ops.min as f64 * inv_opt_sq),
        max: scale * (obs.unopt_ops.max as f64 * inv_opt - unopt_cost * obs.opt_ops.max as f64 * inv_opt_sq),
        mul_add: scale * (obs.unopt_ops.mul_add as f64 * inv_opt - unopt_cost * obs.opt_ops.mul_add as f64 * inv_opt_sq),
    }
}

fn main() {
    println!("=== Cost Model Weight Optimizer ===\n");

    // Simulated benchmark observations
    // In production, these would come from parsing criterion benchmark results
    let observations = vec![
        // FMA Fusion: (X * Y) + Z vs MulAdd(X, Y, Z)
        // Typically MulAdd is ~2x faster
        BenchmarkObservation {
            name: "fma_fusion".to_string(),
            unopt_ops: OpCounts { mul: 1, add: 1, ..Default::default() },
            opt_ops: OpCounts { mul_add: 1, ..Default::default() },
            actual_speedup: 1.8, // Unopt is 1.8x slower
        },
        // Rsqrt fusion: recip(sqrt(x)) vs rsqrt(x)
        BenchmarkObservation {
            name: "rsqrt_fusion".to_string(),
            unopt_ops: OpCounts { sqrt: 1, recip: 1, ..Default::default() },
            opt_ops: OpCounts { rsqrt: 1, ..Default::default() },
            actual_speedup: 3.5, // rsqrt is much faster
        },
        // Double negation: --x vs x
        BenchmarkObservation {
            name: "double_neg".to_string(),
            unopt_ops: OpCounts { neg: 2, ..Default::default() },
            opt_ops: OpCounts { ..Default::default() }, // Just a variable
            actual_speedup: 2.0,
        },
        // Division vs mul-recip: x / y vs x * recip(y)
        BenchmarkObservation {
            name: "div_to_mul_recip".to_string(),
            unopt_ops: OpCounts { div: 1, ..Default::default() },
            opt_ops: OpCounts { mul: 1, recip: 1, ..Default::default() },
            actual_speedup: 1.0, // Similar cost
        },
        // Complex expression with multiple optimizations
        BenchmarkObservation {
            name: "complex".to_string(),
            unopt_ops: OpCounts {
                mul: 3, add: 2, sqrt: 1, neg: 2, ..Default::default()
            },
            opt_ops: OpCounts {
                mul_add: 2, rsqrt: 1, ..Default::default()
            },
            actual_speedup: 2.5,
        },
    ];

    println!("Training on {} observations:\n", observations.len());
    for obs in &observations {
        println!("  {}: speedup = {:.2}x", obs.name, obs.actual_speedup);
    }

    // Initialize weights
    let mut weights = Weights::default();
    let learning_rate = 0.5;
    let epochs = 100;

    println!("\nInitial weights:");
    print_weights(&weights);

    // Training loop
    println!("\nTraining for {} epochs...\n", epochs);
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut total_grad = Weights {
            add: 0.0, sub: 0.0, mul: 0.0, div: 0.0, neg: 0.0,
            sqrt: 0.0, recip: 0.0, rsqrt: 0.0, abs: 0.0,
            min: 0.0, max: 0.0, mul_add: 0.0,
        };

        for obs in &observations {
            let loss = compute_loss(obs, &weights);
            let grad = compute_gradient(obs, &weights);

            total_loss += loss;
            total_grad.add += grad.add;
            total_grad.sub += grad.sub;
            total_grad.mul += grad.mul;
            total_grad.div += grad.div;
            total_grad.neg += grad.neg;
            total_grad.sqrt += grad.sqrt;
            total_grad.recip += grad.recip;
            total_grad.rsqrt += grad.rsqrt;
            total_grad.abs += grad.abs;
            total_grad.min += grad.min;
            total_grad.max += grad.max;
            total_grad.mul_add += grad.mul_add;
        }

        // Average gradient
        let n = observations.len() as f64;
        total_grad.add /= n;
        total_grad.sub /= n;
        total_grad.mul /= n;
        total_grad.div /= n;
        total_grad.neg /= n;
        total_grad.sqrt /= n;
        total_grad.recip /= n;
        total_grad.rsqrt /= n;
        total_grad.abs /= n;
        total_grad.min /= n;
        total_grad.max /= n;
        total_grad.mul_add /= n;

        weights.update(&total_grad, learning_rate);

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("Epoch {}: loss = {:.4}", epoch, total_loss / n);
        }
    }

    println!("\n=== Trained Weights ===\n");
    print_weights(&weights);

    // Validate predictions
    println!("\n=== Validation ===\n");
    for obs in &observations {
        let unopt_cost = obs.unopt_ops.total_cost(&weights);
        let opt_cost = obs.opt_ops.total_cost(&weights);
        let predicted = if opt_cost > 0.1 { unopt_cost / opt_cost } else { 1.0 };

        println!(
            "{:15}: predicted {:.2}x, actual {:.2}x (error: {:.2}%)",
            obs.name,
            predicted,
            obs.actual_speedup,
            (predicted - obs.actual_speedup).abs() / obs.actual_speedup * 100.0
        );
    }

    // Output as CostModel
    println!("\n=== Resulting CostModel ===\n");
    let cost_model = weights.to_cost_model();
    println!("CostModel {{");
    println!("    add: {},", cost_model.add);
    println!("    sub: {},", cost_model.sub);
    println!("    mul: {},", cost_model.mul);
    println!("    div: {},", cost_model.div);
    println!("    neg: {},", cost_model.neg);
    println!("    sqrt: {},", cost_model.sqrt);
    println!("    recip: {},", cost_model.recip);
    println!("    rsqrt: {},", cost_model.rsqrt);
    println!("    abs: {},", cost_model.abs);
    println!("    min: {},", cost_model.min);
    println!("    max: {},", cost_model.max);
    println!("    mul_add: {},", cost_model.mul_add);
    println!("}}");
}

fn print_weights(w: &Weights) {
    println!("  add:     {:.2}", w.add);
    println!("  sub:     {:.2}", w.sub);
    println!("  mul:     {:.2}", w.mul);
    println!("  div:     {:.2}", w.div);
    println!("  neg:     {:.2}", w.neg);
    println!("  sqrt:    {:.2}", w.sqrt);
    println!("  recip:   {:.2}", w.recip);
    println!("  rsqrt:   {:.2}", w.rsqrt);
    println!("  abs:     {:.2}", w.abs);
    println!("  min:     {:.2}", w.min);
    println!("  max:     {:.2}", w.max);
    println!("  mul_add: {:.2}", w.mul_add);
}
