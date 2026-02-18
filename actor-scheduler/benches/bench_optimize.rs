//! Bayesian optimization of scheduler parameters.
//!
//! Measures a weighted combination of latency and throughput across all three
//! lanes, then uses a Gaussian Process surrogate with Expected Improvement
//! to search for strictly better configurations.
//!
//! Run with: `cargo bench -p actor-scheduler --bench bench_optimize`

use actor_scheduler::{
    Actor, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message, SchedulerParams,
    SystemStatus,
};
use std::io::Write;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

fn flush() {
    std::io::stdout().flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// Cost function
// ═══════════════════════════════════════════════════════════════════════════

/// Weights for the multi-objective cost function.
///
/// The cost is: sum_i(weight_i * normalized_metric_i) where each metric
/// is normalized so baseline = 1.0.
///
/// For latency: measured/baseline  (higher = worse).
/// For throughput: baseline/measured (higher = worse).
struct CostWeights {
    control_latency: f64,
    management_latency: f64,
    data_throughput: f64,
    control_throughput: f64,
    mixed_throughput: f64,
    fairness: f64,
}

const WEIGHTS: CostWeights = CostWeights {
    control_latency: 0.25,
    management_latency: 0.10,
    data_throughput: 0.25,
    control_throughput: 0.10,
    mixed_throughput: 0.15,
    fairness: 0.15,
};

/// Raw measurements from a single evaluation.
#[derive(Debug, Clone)]
struct Measurements {
    control_latency_ns: f64,
    management_latency_ns: f64,
    data_throughput_msgs_per_sec: f64,
    control_throughput_msgs_per_sec: f64,
    mixed_throughput_msgs_per_sec: f64,
    /// Fraction of data messages delivered during a control flood (0..1)
    fairness_ratio: f64,
}

/// Evaluate a parameter configuration by running all benchmark scenarios.
fn evaluate(params: &SchedulerParams) -> Measurements {
    Measurements {
        control_latency_ns: measure_control_latency(params),
        management_latency_ns: measure_management_latency(params),
        data_throughput_msgs_per_sec: measure_data_throughput(params),
        control_throughput_msgs_per_sec: measure_control_throughput(params),
        mixed_throughput_msgs_per_sec: measure_mixed_throughput(params),
        fairness_ratio: measure_fairness_under_flood(params),
    }
}

/// Compute the scalar cost from raw measurements, normalized against a baseline.
fn cost(m: &Measurements, b: &Measurements) -> f64 {
    let ctrl_lat = m.control_latency_ns / b.control_latency_ns;
    let mgmt_lat = m.management_latency_ns / b.management_latency_ns;
    let data_tput = b.data_throughput_msgs_per_sec / m.data_throughput_msgs_per_sec.max(1.0);
    let ctrl_tput = b.control_throughput_msgs_per_sec / m.control_throughput_msgs_per_sec.max(1.0);
    let mixed_tput = b.mixed_throughput_msgs_per_sec / m.mixed_throughput_msgs_per_sec.max(1.0);
    let fairness = b.fairness_ratio / m.fairness_ratio.max(0.01);

    WEIGHTS.control_latency * ctrl_lat
        + WEIGHTS.management_latency * mgmt_lat
        + WEIGHTS.data_throughput * data_tput
        + WEIGHTS.control_throughput * ctrl_tput
        + WEIGHTS.mixed_throughput * mixed_tput
        + WEIGHTS.fairness * fairness
}

// ═══════════════════════════════════════════════════════════════════════════
// Measurement scenarios (tuned for fast execution in containers)
// ═══════════════════════════════════════════════════════════════════════════

struct LatencyActor {
    response_tx: mpsc::Sender<()>,
}

impl Actor<(), (), ()> for LatencyActor {
    fn handle_data(&mut self, _: ()) -> HandlerResult { Ok(()) }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        let _ = self.response_tx.send(());
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        let _ = self.response_tx.send(());
        Ok(())
    }
    fn park(&mut self, h: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(match h { SystemStatus::Idle => ActorStatus::Idle, SystemStatus::Busy => ActorStatus::Busy })
    }
}

fn measure_control_latency(params: &SchedulerParams) -> f64 {
    let (response_tx, response_rx) = mpsc::channel();
    let (tx, mut rx) = ActorScheduler::new_with_params(params.default_data_burst_limit, 64, *params);
    let h = thread::spawn(move || { let mut a = LatencyActor { response_tx }; rx.run(&mut a); });
    thread::sleep(Duration::from_millis(1));

    // Warmup
    for _ in 0..10 {
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
    }

    let rounds = 50;
    let mut lats = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        tx.send(Message::Control(())).unwrap();
        response_rx.recv().unwrap();
        lats.push(t.elapsed().as_nanos() as f64);
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    lats[lats.len() / 2]
}

fn measure_management_latency(params: &SchedulerParams) -> f64 {
    let (response_tx, response_rx) = mpsc::channel();
    let (tx, mut rx) = ActorScheduler::new_with_params(params.default_data_burst_limit, 64, *params);
    let h = thread::spawn(move || { let mut a = LatencyActor { response_tx }; rx.run(&mut a); });
    thread::sleep(Duration::from_millis(1));

    for _ in 0..10 {
        tx.send(Message::Management(())).unwrap();
        response_rx.recv().unwrap();
    }

    let rounds = 50;
    let mut lats = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        tx.send(Message::Management(())).unwrap();
        response_rx.recv().unwrap();
        lats.push(t.elapsed().as_nanos() as f64);
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    lats[lats.len() / 2]
}

struct CountingActor {
    data_count: Arc<AtomicUsize>,
    control_count: Arc<AtomicUsize>,
    mgmt_count: Arc<AtomicUsize>,
}

impl Actor<i32, (), ()> for CountingActor {
    fn handle_data(&mut self, _: i32) -> HandlerResult { self.data_count.fetch_add(1, Ordering::Relaxed); Ok(()) }
    fn handle_control(&mut self, _: ()) -> HandlerResult { self.control_count.fetch_add(1, Ordering::Relaxed); Ok(()) }
    fn handle_management(&mut self, _: ()) -> HandlerResult { self.mgmt_count.fetch_add(1, Ordering::Relaxed); Ok(()) }
    fn park(&mut self, h: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(match h { SystemStatus::Idle => ActorStatus::Idle, SystemStatus::Busy => ActorStatus::Busy })
    }
}

fn new_counting() -> (Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>, CountingActor) {
    let d = Arc::new(AtomicUsize::new(0));
    let c = Arc::new(AtomicUsize::new(0));
    let m = Arc::new(AtomicUsize::new(0));
    let actor = CountingActor { data_count: d.clone(), control_count: c.clone(), mgmt_count: m.clone() };
    (d, c, m, actor)
}

fn measure_data_throughput(params: &SchedulerParams) -> f64 {
    let n = 5_000;
    let (dc, _, _, mut actor) = new_counting();
    let (tx, mut rx) = ActorScheduler::new_with_params(params.default_data_burst_limit, 512, *params);
    let h = thread::spawn(move || rx.run(&mut actor));
    let t = Instant::now();
    for i in 0..n { tx.send(Message::Data(i)).unwrap(); }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    dc.load(Ordering::Relaxed) as f64 / t.elapsed().as_secs_f64()
}

fn measure_control_throughput(params: &SchedulerParams) -> f64 {
    let n = 2_000;
    let (_, cc, _, mut actor) = new_counting();
    let (tx, mut rx) = ActorScheduler::new_with_params(params.default_data_burst_limit, 64, *params);
    let h = thread::spawn(move || rx.run(&mut actor));
    let t = Instant::now();
    for _ in 0..n { tx.send(Message::Control(())).unwrap(); }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    cc.load(Ordering::Relaxed) as f64 / t.elapsed().as_secs_f64()
}

fn measure_mixed_throughput(params: &SchedulerParams) -> f64 {
    let per = 1_500;
    let (dc, cc, mc, mut actor) = new_counting();
    let (tx, mut rx) = ActorScheduler::new_with_params(params.default_data_burst_limit, 512, *params);
    let h = thread::spawn(move || rx.run(&mut actor));
    let t = Instant::now();
    for i in 0..per {
        tx.send(Message::Data(i)).unwrap();
        tx.send(Message::Control(())).unwrap();
        tx.send(Message::Management(())).unwrap();
    }
    tx.send(Message::Shutdown).unwrap();
    h.join().unwrap();
    let total = dc.load(Ordering::Relaxed) + cc.load(Ordering::Relaxed) + mc.load(Ordering::Relaxed);
    total as f64 / t.elapsed().as_secs_f64()
}

fn measure_fairness_under_flood(params: &SchedulerParams) -> f64 {
    let data_target = 100i32;
    let (dc, _, _, mut actor) = new_counting();
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, mut rx) = ActorScheduler::new_with_params(params.default_data_burst_limit, 128, *params);
    let h = thread::spawn(move || rx.run(&mut actor));

    let tx_f = tx.clone();
    let sf = stop.clone();
    let flooder = thread::spawn(move || { while !sf.load(Ordering::Relaxed) { let _ = tx_f.send(Message::Control(())); } });

    thread::sleep(Duration::from_millis(2));
    for i in 0..data_target { let _ = tx.send(Message::Data(i)); }
    thread::sleep(Duration::from_millis(15));

    stop.store(true, Ordering::Relaxed);
    flooder.join().unwrap();
    thread::sleep(Duration::from_millis(5));
    let processed = dc.load(Ordering::Relaxed);
    drop(tx);
    h.join().unwrap();
    processed as f64 / data_target as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Gaussian Process with RBF kernel
// ═══════════════════════════════════════════════════════════════════════════

const NDIM: usize = 10;

struct GaussianProcess {
    xs: Vec<[f64; NDIM]>,
    ys: Vec<f64>,
    length_scales: [f64; NDIM],
    signal_var: f64,
    noise_var: f64,
    chol: Vec<f64>,
    alpha: Vec<f64>,
}

impl GaussianProcess {
    fn new() -> Self {
        let bounds = SchedulerParams::bounds();
        let mut ls = [0.0; NDIM];
        for (i, (lo, hi)) in bounds.iter().enumerate() {
            ls[i] = (hi - lo) / 3.0;
        }
        Self { xs: Vec::new(), ys: Vec::new(), length_scales: ls, signal_var: 1.0, noise_var: 0.01, chol: Vec::new(), alpha: Vec::new() }
    }

    fn kernel(&self, a: &[f64; NDIM], b: &[f64; NDIM]) -> f64 {
        let mut d = 0.0;
        for i in 0..NDIM { let x = (a[i] - b[i]) / self.length_scales[i]; d += x * x; }
        self.signal_var * (-0.5 * d).exp()
    }

    fn observe(&mut self, x: [f64; NDIM], y: f64) {
        self.xs.push(x);
        self.ys.push(y);
        self.refit();
    }

    fn refit(&mut self) {
        let n = self.xs.len();
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = self.kernel(&self.xs[i], &self.xs[j]);
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
            k[i * n + i] += self.noise_var;
        }

        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for kk in 0..j { s += l[i * n + kk] * l[j * n + kk]; }
                l[i * n + j] = if i == j {
                    let d = k[i * n + i] - s;
                    if d > 0.0 { d.sqrt() } else { 1e-10 }
                } else {
                    (k[i * n + j] - s) / l[j * n + j]
                };
            }
        }

        let mut z = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..i { s += l[i * n + j] * z[j]; }
            z[i] = (self.ys[i] - s) / l[i * n + i];
        }

        let mut alpha = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = 0.0;
            for j in (i + 1)..n { s += l[j * n + i] * alpha[j]; }
            alpha[i] = (z[i] - s) / l[i * n + i];
        }

        self.chol = l;
        self.alpha = alpha;
    }

    fn predict(&self, x: &[f64; NDIM]) -> (f64, f64) {
        let n = self.xs.len();
        if n == 0 { return (0.0, self.signal_var); }

        let mut ks = vec![0.0; n];
        for i in 0..n { ks[i] = self.kernel(x, &self.xs[i]); }

        let mean: f64 = ks.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum();

        let mut v = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..i { s += self.chol[i * n + j] * v[j]; }
            v[i] = (ks[i] - s) / self.chol[i * n + i];
        }
        let vsq: f64 = v.iter().map(|vi| vi * vi).sum();
        (mean, (self.signal_var - vsq).max(1e-10))
    }
}

fn expected_improvement(mean: f64, var: f64, f_best: f64) -> f64 {
    let sigma = var.sqrt();
    if sigma < 1e-12 { return 0.0; }
    let z = (f_best - mean) / sigma;
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    (f_best - mean) * big_phi + sigma * phi
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════
// Deterministic xorshift64 PRNG
// ═══════════════════════════════════════════════════════════════════════════

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.uniform() }
}

fn latin_hypercube(n: usize, rng: &mut Rng) -> Vec<[f64; NDIM]> {
    let bounds = SchedulerParams::bounds();
    let mut result = vec![[0.0; NDIM]; n];
    for dim in 0..NDIM {
        let (lo, hi) = bounds[dim];
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() { let j = (rng.next_u64() as usize) % (i + 1); perm.swap(i, j); }
        let step = (hi - lo) / n as f64;
        for (i, &pi) in perm.iter().enumerate() {
            result[i][dim] = lo + step * (pi as f64 + rng.uniform());
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimizer
// ═══════════════════════════════════════════════════════════════════════════

fn run_optimization() {
    let initial_samples = 15;
    let bo_iterations = 25;
    let acq_candidates = 300;

    println!("=== Bayesian Scheduler Parameter Optimization ===\n");
    flush();

    // 1. Baseline
    println!("Measuring baseline (current defaults)..."); flush();
    let bp = SchedulerParams::default();
    let baseline = evaluate(&bp);
    let bc = cost(&baseline, &baseline);
    println!("Baseline measurements:");
    print_measurements(&baseline);
    println!("Baseline cost: {:.4} (normalized to 1.0)\n", bc);
    flush();

    let mut gp = GaussianProcess::new();
    let mut rng = Rng::new(42);
    let mut best_cost = bc;
    let mut best_params = bp;
    let mut best_meas = baseline.clone();

    gp.observe(bp.to_vec(), bc);

    // 2. LHS exploration
    println!("Phase 1: Latin Hypercube exploration ({initial_samples} samples)..."); flush();
    let lhs = latin_hypercube(initial_samples, &mut rng);

    for (i, pt) in lhs.iter().enumerate() {
        let p = SchedulerParams::from_vec(pt);
        if p.min_backoff > p.max_backoff || p.jitter_min_pct + p.jitter_range_pct > 100 {
            continue;
        }
        let m = evaluate(&p);
        let c = cost(&m, &baseline);
        let tag = if c < best_cost { " *BEST*" } else { "" };
        println!("  [{:2}/{initial_samples}] cost={c:.4}{tag}", i + 1);
        flush();
        if c < best_cost { best_cost = c; best_params = p; best_meas = m; }
        gp.observe(*pt, c);
    }

    println!("\nAfter exploration: best cost = {best_cost:.4}\n");
    flush();

    // 3. BO loop
    println!("Phase 2: Bayesian optimization ({bo_iterations} iterations)..."); flush();
    let bounds = SchedulerParams::bounds();

    for iter in 0..bo_iterations {
        let f_best = *gp.ys.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut best_ei = -1.0f64;
        let mut best_cand = [0.0; NDIM];

        for _ in 0..acq_candidates {
            let mut cand = [0.0; NDIM];
            for d in 0..NDIM { cand[d] = rng.uniform_range(bounds[d].0, bounds[d].1); }
            let (mu, var) = gp.predict(&cand);
            let ei = expected_improvement(mu, var, f_best);
            if ei > best_ei { best_ei = ei; best_cand = cand; }
        }

        let p = SchedulerParams::from_vec(&best_cand);
        if p.min_backoff > p.max_backoff || p.jitter_min_pct + p.jitter_range_pct > 100 {
            gp.observe(best_cand, best_cost * 2.0);
            continue;
        }

        let m = evaluate(&p);
        let c = cost(&m, &baseline);
        let tag = if c < best_cost { " *BEST*" } else { "" };
        println!("  BO [{:2}/{bo_iterations}] cost={c:.4} EI={best_ei:.6}{tag}", iter + 1);
        flush();
        if c < best_cost { best_cost = c; best_params = p; best_meas = m; }
        gp.observe(best_cand, c);
    }

    // 4. Report
    println!("\n{}", "=".repeat(55));
    println!("                    RESULTS");
    println!("{}\n", "=".repeat(55));

    println!("Baseline cost:  {bc:.4}");
    println!("Optimized cost: {best_cost:.4}");
    println!("Improvement:    {:.1}%\n", (1.0 - best_cost / bc) * 100.0);

    println!("--- Baseline measurements ---");
    print_measurements(&baseline);
    println!("\n--- Optimized measurements ---");
    print_measurements(&best_meas);

    println!("\n--- Optimized parameters ---");
    let v = best_params.to_vec();
    let dv = bp.to_vec();
    for (i, name) in SchedulerParams::NAMES.iter().enumerate() {
        let pct = if dv[i] > 0.0 { ((v[i] - dv[i]) / dv[i]) * 100.0 } else { 0.0 };
        println!("  {name:20} = {:>12.1}  (was {:>10.1}, {pct:+.1}%)", v[i], dv[i]);
    }

    println!("\n--- Copy-paste for SchedulerParams::DEFAULT ---");
    println!("pub const DEFAULT: Self = Self {{");
    println!("    control_mgmt_buffer_size: {},", best_params.control_mgmt_buffer_size);
    println!("    control_burst_multiplier: {},", best_params.control_burst_multiplier);
    println!("    management_burst_multiplier: {},", best_params.management_burst_multiplier);
    println!("    default_data_burst_limit: {},", best_params.default_data_burst_limit);
    println!("    spin_attempts: {},", best_params.spin_attempts);
    println!("    yield_attempts: {},", best_params.yield_attempts);
    println!("    min_backoff: Duration::from_micros({}),", best_params.min_backoff.as_micros());
    println!("    max_backoff: Duration::from_micros({}),", best_params.max_backoff.as_micros());
    println!("    jitter_min_pct: {},", best_params.jitter_min_pct);
    println!("    jitter_range_pct: {},", best_params.jitter_range_pct);
    println!("}};");
    flush();
}

fn print_measurements(m: &Measurements) {
    println!("  Control latency:     {:>10.0} ns", m.control_latency_ns);
    println!("  Management latency:  {:>10.0} ns", m.management_latency_ns);
    println!("  Data throughput:     {:>10.0} msg/s", m.data_throughput_msgs_per_sec);
    println!("  Control throughput:  {:>10.0} msg/s", m.control_throughput_msgs_per_sec);
    println!("  Mixed throughput:    {:>10.0} msg/s", m.mixed_throughput_msgs_per_sec);
    println!("  Fairness ratio:      {:>9.2}%", m.fairness_ratio * 100.0);
}

fn main() {
    run_optimization();
}
