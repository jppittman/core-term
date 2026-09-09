//! The bilinear rules × nodes filter, measured on the shipped kernels
//! (docs/plans/2026-09-08-rules-filter-bilinear-registration.md).
//!
//! ```text
//! rules_filter mint  --out samples.jsonl            # DEV under KeepAll, features at the seam, strict + tight labels
//! rules_filter train --samples samples.jsonl --out models/   # family-held-out folds + all-DEV, thresholds at ρ
//! rules_filter eval  --models models/ --out rows.jsonl [--held-out] [--shard i/n]
//! rules_filter report --rows rows.jsonl [--held-out-rows ...] --out-prefix docs/results/2026-09-08-rules-filter-bilinear
//! ```
//!
//! Every arm runs the production loop through `Optimizer::filter` at
//! `Budget::Applications(b)`; `Identity` at `b = B` (the applications the
//! production run fired) is asserted to reproduce production's `dag_cost`
//! and bytes, so the reference column is production and not a model of it.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use pixelflow_codegen::emit;
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};
use pixelflow_pipeline::real_corpus::{
    RealKernel, capture_stderr, dag_cost, legalize, parse_guard_telemetry, real_kernels,
};
use pixelflow_pipeline::training::guide_bilinear::{BilinearGuideCheckpoint, ops_from_histogram};
use pixelflow_pipeline::training::guide_linear::{auc_roc, average_precision};
use pixelflow_search::egraph::{
    ApplicationFilter, Budget, EClassId, EpisodeLabels, Fingerprint, InputSize, KeepAll,
    KeepJournal, Label, MatchRow, Optimizer, RuleId, RuleSet, Vocabulary, insert, reachable_count,
};
use pixelflow_search::nnue::guide::bilinear::{BilinearTrainer, ColdStart, SgdStep};
use pixelflow_search::nnue::guide::filter::{
    BilinearFilter, CellContext, Episode, FilterStats, PerRuleRateFilter, Reporting,
    UniformRandomFilter,
};

const SCHEMA: &str = "rules-filter-v1";
/// The registered keep-rates: primary first.
const KEEP_RATES: [f32; 2] = [0.25, 0.5];
/// The budget grid, as fractions of the kernel's own production `B`.
const BUDGET_DIVISORS: [(&str, u64); 4] = [("B", 1), ("B/2", 2), ("B/4", 4), ("B/8", 8)];
const UNIFORM_SEED: u64 = 1;
const FOLDS: [&str; 3] = ["glyph", "shader", "scene"];
const ALL: &str = "all";

#[derive(Parser)]
#[command(name = "rules_filter")]
#[command(
    about = "Mint, train, evaluate and report the bilinear rules × nodes filter on real shaders"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// DEV under KeepAll at the production budget: every cell's context at
    /// the seam, joined to its strict and tight hindsight label.
    Mint {
        #[arg(long)]
        out: PathBuf,
        /// Stride cap on samples written per kernel (0 = all).
        #[arg(long, default_value_t = 4000)]
        per_kernel_cap: usize,
        #[arg(long)]
        filter: Option<String>,
    },
    /// Family-held-out folds and the all-DEV model; thresholds at the
    /// registered keep-rates; intrinsic AUC / PR-AUC on the held-out fold.
    Train {
        #[arg(long)]
        samples: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Which models to train: any of `glyph`, `shader`, `scene` (the
        /// fold held out) and `all` (the all-DEV model). Default: all four.
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        folds: Option<Vec<String>>,
        #[command(flatten)]
        args: TrainArgs,
    },
    /// Every arm at every budget on every kernel; one row per run.
    Eval {
        #[arg(long)]
        models: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// NotoSansMono-Bold glyphs and chrome, with the all-DEV model only.
        /// Opened once, at the end.
        #[arg(long)]
        held_out: bool,
        /// `i/n`: run the i-th of n interleaved slices of the corpus.
        #[arg(long)]
        shard: Option<String>,
        #[arg(long)]
        filter: Option<String>,
    },
    /// Per-family tables, the dual curve, and the verdict.
    Report {
        #[arg(long, num_args = 1..)]
        rows: Vec<PathBuf>,
        #[arg(long, num_args = 0..)]
        held_out_rows: Vec<PathBuf>,
        #[arg(long)]
        models: PathBuf,
        #[arg(long)]
        out_prefix: PathBuf,
    },
}

// ── Corpus ──────────────────────────────────────────────────────────────────

fn fold_of(class: &str) -> &'static str {
    match class {
        "glyph16" | "glyph32" | "bench" | "bench_wide" => "glyph",
        "shader" => "shader",
        "psychedelic" | "cellgrid" => "scene",
        "chrome" | "chrome_channel" => "chrome",
        other => panic!("rules_filter: kernel class {other:?} belongs to no registered fold"),
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn dev_kernels(filter: Option<&str>) -> Vec<RealKernel> {
    let font = manifest_dir().join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf");
    let mut kernels = real_kernels(&font, filter);
    kernels.retain(|k| fold_of(&k.class) != "chrome");
    kernels
}

fn held_out_kernels(filter: Option<&str>) -> Vec<RealKernel> {
    let font = manifest_dir().join("../assets/font/Noto_Sans_Mono/static/NotoSansMono-Bold.ttf");
    let mut kernels = real_kernels(&font, filter);
    kernels.retain(|k| matches!(fold_of(&k.class), "glyph" | "chrome"));
    for k in &mut kernels {
        if fold_of(&k.class) == "glyph" {
            k.name = format!("bold_{}", k.name);
            k.class = format!("bold_{}", k.class);
        }
    }
    kernels
}

fn shard(kernels: Vec<RealKernel>, spec: Option<&str>) -> Vec<RealKernel> {
    let Some(spec) = spec else {
        return kernels;
    };
    let (i, n) = spec
        .split_once('/')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .unwrap_or_else(|| panic!("--shard {spec:?}: expected i/n"));
    assert!(n > 0 && i < n, "--shard {spec:?}: need 0 <= i < n");
    kernels
        .into_iter()
        .enumerate()
        .filter(|(k, _)| k % n == i)
        .map(|(_, k)| k)
        .collect()
}

/// One kernel, legalized and sized: what every run of it starts from.
struct Prepared<'a> {
    kernel: &'a RealKernel,
    arena: ExprArena,
    root: ExprId,
    shape: LatticeShape,
    node_count: usize,
    /// The production tier's application cap — `budget_fraction`'s
    /// denominator.
    cap: u64,
}

fn prepare(kernel: &RealKernel) -> Prepared<'_> {
    let (arena, root) = kernel.kernel.parts();
    let (legal, legal_root) = legalize(arena, root);
    let node_count = reachable_count(&legal, legal_root);
    // The tier is keyed on the node count and the hash-consed class count
    // (`InputSize`), and the latter exists only once the input is inserted:
    // one throwaway insertion here, so every run of this kernel reads the
    // same cap the production run resolves.
    let classes = {
        let mut egraph = Optimizer::production().egraph();
        insert(&legal, legal_root, &mut egraph, Vocabulary::Runtime)
            .unwrap_or_else(|_| panic!("{}: not representable", kernel.name));
        egraph.num_classes()
    };
    let cap = Optimizer::production()
        .limits_for(InputSize {
            nodes: node_count,
            classes,
        })
        .applications
        .expect("Budget::Production always caps applications");
    Prepared {
        kernel,
        arena: legal,
        root: legal_root,
        shape: LatticeShape::new(kernel.extent),
        node_count,
        cap,
    }
}

impl Prepared<'_> {
    fn episode(&self) -> Episode {
        Episode::new(self.node_count, self.cap)
    }
}

/// Emit an extracted arena the way production does and read the static
/// columns off it.
struct Emitted {
    dag_cost: usize,
    bytes: u32,
    guarded: Option<u64>,
    schedule: Option<u64>,
}

fn emit_extracted(p: &Prepared<'_>, extracted: &ExprArena, extracted_root: ExprId) -> Emitted {
    let (original, _) = p.kernel.kernel.parts();
    let (linked, root) = if original.buffers().is_empty() && original.uniforms().is_empty() {
        (extracted.clone(), extracted_root)
    } else {
        extracted.relink(extracted_root, original.buffers(), original.uniforms())
    };
    let (result, log) = capture_stderr(|| emit::compile(&linked, root));
    let result = result.expect("extracted kernel failed to compile");
    let guard = parse_guard_telemetry(&log);
    Emitted {
        dag_cost: dag_cost(&linked, root),
        bytes: result.code.len() as u32,
        guarded: guard.as_ref().map(|g| g.guarded),
        schedule: guard.as_ref().map(|g| g.schedule),
    }
}

// ── mint ────────────────────────────────────────────────────────────────────

struct RecordedCell {
    ordinal: u64,
    rule: RuleId,
    class: EClassId,
    context: CellContext,
}

/// `KeepAll` that writes down every cell it was shown, with the context the
/// deployed filter would have scored it on. Keeps everything, so the run it
/// records is production's.
struct Recording {
    episode: Episode,
    sink: Rc<RefCell<Vec<RecordedCell>>>,
}

impl ApplicationFilter for Recording {
    fn filter(&mut self, graph: &pixelflow_search::egraph::EGraph, row: &mut MatchRow) {
        let base = graph.application_count();
        let rule = row.rule();
        let mut sink = self.sink.borrow_mut();
        let mut last: Option<(EClassId, CellContext)> = None;
        for (i, (class, _)) in row.matches().iter().enumerate() {
            let context = match &last {
                Some((c, ctx)) if c == class => ctx.clone(),
                _ => CellContext::observe(graph, *class, &self.episode),
            };
            last = Some((*class, context.clone()));
            sink.push(RecordedCell {
                ordinal: base + i as u64,
                rule,
                class: *class,
                context,
            });
        }
    }

    fn fingerprint(&self) -> Fingerprint {
        KeepAll.fingerprint()
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Sample {
    /// Kernel name.
    k: String,
    /// Kernel class (family).
    f: String,
    /// Application ordinal.
    o: u64,
    /// Rule label.
    r: String,
    /// Rule id, hex.
    rid: String,
    /// Neighborhood op histogram.
    h: BTreeMap<String, usize>,
    bf: f32,
    mcn: usize,
    enc: usize,
    /// Strict label.
    s: u8,
    /// Tight label.
    t: u8,
}

fn histogram(ops: &[OpKind]) -> BTreeMap<String, usize> {
    let mut h = BTreeMap::new();
    for op in ops {
        *h.entry(format!("{op:?}")).or_insert(0) += 1;
    }
    h
}

fn mint(out: &Path, per_kernel_cap: usize, filter: Option<&str>) {
    assert!(
        std::env::var_os("PIXELFLOW_GUARD_TELEMETRY").is_some(),
        "set PIXELFLOW_GUARD_TELEMETRY=1: every subcommand reads the emitter's guard report"
    );
    let kernels = dev_kernels(filter);
    let mut sink = open_append(out);
    let rules = RuleSet::production();
    let label_of = |id: RuleId| -> String {
        rules
            .index_of(id)
            .and_then(|i| rules.label_of(i))
            .unwrap_or_else(|| panic!("rule {id} is not in the production rule set"))
    };
    let (mut total_cells, mut total_written, mut total_strict, mut total_tight) =
        (0u64, 0u64, 0u64, 0u64);
    for (i, kernel) in kernels.iter().enumerate() {
        let started = Instant::now();
        let p = prepare(kernel);
        let recorded: Rc<RefCell<Vec<RecordedCell>>> = Rc::default();
        let mut optimizer = Optimizer::production()
            .for_lattice(p.shape)
            .observe(Some(Box::new(KeepJournal)))
            .filter(Box::new(Recording {
                episode: p.episode(),
                sink: Rc::clone(&recorded),
            }))
            .no_ceiling();
        let mut egraph = optimizer.egraph();
        let root_class = insert(&p.arena, p.root, &mut egraph, Vocabulary::Runtime)
            .unwrap_or_else(|_| panic!("{}: not representable", kernel.name));
        let optimized = optimizer.run(&mut egraph, root_class, p.node_count);
        let tight = EpisodeLabels::compute_tight(&egraph, root_class, &optimized.choices);
        let strict = EpisodeLabels::compute_strict(&egraph, root_class, &optimized.choices);

        let cells = recorded.borrow();
        let by_ordinal: BTreeMap<u64, &RecordedCell> =
            cells.iter().map(|c| (c.ordinal, c)).collect();
        assert_eq!(
            by_ordinal.len(),
            cells.len(),
            "{}: two recorded cells share an ordinal",
            kernel.name
        );
        let mut joined: Vec<Sample> = Vec::new();
        let mut applications = 0u64;
        for (app_id, record) in egraph.provenance().applications() {
            applications += 1;
            let cell = by_ordinal.get(&app_id.as_u64()).unwrap_or_else(|| {
                panic!(
                    "{}: application {} has no recorded cell — the ordinal join is broken",
                    kernel.name,
                    app_id.as_u64()
                )
            });
            assert_eq!(
                record.rule,
                Some(cell.rule),
                "{}: application {} fired rule {:?} but the recorded cell names {}",
                kernel.name,
                app_id.as_u64(),
                record.rule,
                cell.rule
            );
            assert_eq!(
                record.match_root,
                cell.class,
                "{}: application {} matched {:?} but the recorded cell names {:?}",
                kernel.name,
                app_id.as_u64(),
                record.match_root,
                cell.class
            );
            let s = u8::from(strict.labels[&app_id] == Label::LoadBearing);
            let t = u8::from(tight.labels[&app_id] == Label::LoadBearing);
            joined.push(Sample {
                k: kernel.name.clone(),
                f: kernel.class.clone(),
                o: app_id.as_u64(),
                r: label_of(cell.rule),
                rid: format!("{}", cell.rule),
                h: histogram(&cell.context.neighborhood_ops),
                bf: cell.context.budget_fraction,
                mcn: cell.context.match_class_node_count,
                enc: cell.context.expr_node_count,
                s,
                t,
            });
        }
        assert_eq!(
            applications, optimized.stats.applications,
            "{}: the journal and the budget counter disagree",
            kernel.name
        );
        let unapplied = cells.len() as u64 - applications;
        let stride = if per_kernel_cap == 0 {
            1
        } else {
            joined.len().div_ceil(per_kernel_cap).max(1)
        };
        let n_strict = joined.iter().filter(|s| s.s == 1).count() as u64;
        let n_tight = joined.iter().filter(|s| s.t == 1).count() as u64;
        let mut written = 0u64;
        for sample in joined.iter().step_by(stride) {
            writeln!(
                sink,
                "{}",
                serde_json::to_string(sample).expect("serialize")
            )
            .expect("write");
            written += 1;
        }
        sink.flush().expect("flush");
        total_cells += applications;
        total_written += written;
        total_strict += n_strict;
        total_tight += n_tight;
        eprintln!(
            "[{}/{}] {} apps={} strict={} tight={} unapplied_tail={} stride={} written={} stop={:?} ({:.1}s)",
            i + 1,
            kernels.len(),
            kernel.name,
            applications,
            n_strict,
            n_tight,
            unapplied,
            stride,
            written,
            optimized.stats.stop,
            started.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "mint: {} applications, {} strict ({:.2}%), {} tight ({:.2}%), {} samples written",
        total_cells,
        total_strict,
        100.0 * total_strict as f64 / total_cells.max(1) as f64,
        total_tight,
        100.0 * total_tight as f64 / total_cells.max(1) as f64,
        total_written
    );
}

fn open_append(path: &Path) -> std::fs::File {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create out dir");
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    std::io::BufReader::new(file)
        .lines()
        .map(|l| l.expect("read line"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .collect()
}

// ── train ───────────────────────────────────────────────────────────────────

struct Loaded {
    rule: RuleId,
    ops: Vec<OpKind>,
    bf: f32,
    mcn: usize,
    enc: usize,
    strict: f32,
    tight: f32,
    fold: &'static str,
}

fn load_samples(path: &Path) -> Vec<Loaded> {
    let rows: Vec<Sample> = read_jsonl(path);
    rows.into_iter()
        .map(|s| {
            let rule = RuleId::from_label(&s.r);
            assert_eq!(
                format!("{rule}"),
                s.rid,
                "sample for rule {:?}: label and id disagree — minted against another rule set",
                s.r
            );
            Loaded {
                rule,
                ops: ops_from_histogram(&s.h),
                bf: s.bf,
                mcn: s.mcn,
                enc: s.enc,
                strict: f32::from(s.s),
                tight: f32::from(s.t),
                fold: fold_of(&s.f),
            }
        })
        .collect()
}

fn summary_of(
    trainer: &BilinearTrainer,
    s: &Loaded,
) -> pixelflow_search::nnue::guide::CandidateSummary {
    CellContext {
        neighborhood_ops: s.ops.clone(),
        match_class_node_count: s.mcn,
        budget_fraction: s.bf,
        expr_node_count: s.enc,
    }
    .summary(s.rule, trainer.rule_embed(s.rule))
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn weighted_bce_grad(p: f32, y: f32, pos_weight: f32) -> f32 {
    if y > 0.5 { pos_weight * (p - 1.0) } else { p }
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = (state >> 33) as usize % (i + 1);
        idx.swap(i, r);
    }
    idx
}

/// The `(1 − ρ)` quantile of `scores`: the value at which a fraction `ρ`
/// of them is at or above it.
fn keep_threshold(scores: &[f32], rho: f32) -> f32 {
    assert!(!scores.is_empty());
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("finite score"));
    let k = ((rho * sorted.len() as f32) as usize).min(sorted.len() - 1);
    sorted[k]
}

fn realized_keep_rate(scores: &[f32], threshold: f32) -> f64 {
    scores.iter().filter(|&&s| s >= threshold).count() as f64 / scores.len() as f64
}

#[derive(Serialize, Deserialize, Clone)]
struct Intrinsic {
    n: usize,
    positive_rate_tight: f64,
    positive_rate_strict: f64,
    bilinear_auc_tight: Option<f64>,
    bilinear_pr_auc_tight: Option<f64>,
    bilinear_auc_strict: Option<f64>,
    bilinear_pr_auc_strict: Option<f64>,
    per_rule_auc_tight: Option<f64>,
    per_rule_pr_auc_tight: Option<f64>,
    per_rule_auc_strict: Option<f64>,
    per_rule_pr_auc_strict: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
struct FoldModel {
    fold: String,
    checkpoint: String,
    train_samples: usize,
    train_folds: Vec<String>,
    train_positive_rate_tight: f64,
    train_positive_rate_strict: f64,
    /// Fraction of tight positives that strict does not see.
    tight_positives_invisible_to_strict: f64,
    pos_weight: f32,
    epoch_loss: Vec<f64>,
    /// Raw-score thresholds at each registered keep-rate, keyed by ρ.
    bilinear_threshold: BTreeMap<String, f32>,
    bilinear_realized_keep_rate: BTreeMap<String, f64>,
    /// Per-rule positive rate (tight) on the training samples.
    rule_rates: BTreeMap<String, f32>,
    per_rule_threshold: BTreeMap<String, f32>,
    per_rule_realized_keep_rate: BTreeMap<String, f64>,
    holdout: Option<Intrinsic>,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    schema: String,
    git_sha: String,
    #[serde(flatten)]
    args: TrainArgs,
    models: BTreeMap<String, FoldModel>,
}

/// The label a sample is trained against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TrainLabel {
    /// `EpisodeLabels::compute_tight` — the registered label (§5).
    Tight,
    /// `EpisodeLabels::compute_strict` — reported beside tight; blind to
    /// union-only rewrites.
    Strict,
}

/// Every hyperparameter of `train`, one definition: the clap defaults are
/// read off [`TrainArgs::default`], which is the configuration the
/// registered (untuned) run used, so trial 0 of a sweep is that run.
/// Written into the manifest whole; a manifest from before a knob existed
/// reads the default for it.
#[derive(Args, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct TrainArgs {
    #[arg(long, default_value_t = TrainArgs::default().epochs)]
    epochs: usize,
    #[arg(long, default_value_t = TrainArgs::default().lr)]
    lr: f32,
    /// Multiplier on the learning rate after each epoch.
    #[arg(long, default_value_t = TrainArgs::default().lr_decay)]
    lr_decay: f32,
    #[arg(long, default_value_t = TrainArgs::default().l2)]
    l2: f32,
    #[arg(long, default_value_t = TrainArgs::default().max_grad_norm)]
    max_grad_norm: f32,
    #[arg(long, default_value_t = TrainArgs::default().seed)]
    seed: u64,
    /// Label trained against.
    #[arg(long, value_enum, default_value_t = TrainArgs::default().label)]
    label: TrainLabel,
    /// `pos_weight = (effective negatives / positives) ^ power`: `1.0` is
    /// full rebalancing (the registered choice), `0.0` is unweighted BCE.
    #[arg(long, default_value_t = TrainArgs::default().pos_weight_power)]
    pos_weight_power: f32,
    /// Samples whose gradients are accumulated (mean) before one SGD step.
    #[arg(long, default_value_t = TrainArgs::default().batch_size)]
    batch_size: usize,
    /// Fraction of negatives each epoch trains on, decided per sample by
    /// a hash of `(seed, epoch, index)`; `pos_weight` is computed on the
    /// effective negative count. `1.0` is every negative.
    #[arg(long, default_value_t = TrainArgs::default().neg_keep)]
    neg_keep: f32,
    /// Multiplier on every random draw of the head's initialisation.
    #[arg(long, default_value_t = TrainArgs::default().init_scale)]
    init_scale: f32,
    /// Offset written over every ReLU bias at initialisation.
    #[arg(long, default_value_t = TrainArgs::default().relu_warm_bias)]
    relu_warm_bias: f32,
    /// Stride cap on the training samples per model (0 = all): a sweep
    /// trains on a deterministic stride of the fold's training set, and
    /// the held-out fold is never capped.
    #[arg(long, default_value_t = TrainArgs::default().train_cap)]
    train_cap: usize,
}

impl Default for TrainArgs {
    fn default() -> Self {
        let cold = ColdStart::seeded(17);
        Self {
            epochs: 3,
            lr: 0.01,
            lr_decay: 0.7,
            l2: 1e-4,
            max_grad_norm: 1.0,
            seed: cold.seed,
            label: TrainLabel::Tight,
            pos_weight_power: 1.0,
            batch_size: 1,
            neg_keep: 1.0,
            init_scale: cold.init_scale,
            relu_warm_bias: cold.relu_warm_bias,
            train_cap: 0,
        }
    }
}

impl TrainArgs {
    fn validate(&self) {
        assert!(self.epochs > 0, "--epochs must be positive");
        assert!(self.batch_size > 0, "--batch-size must be positive");
        assert!(
            (0.0..=1.0).contains(&self.neg_keep) && self.neg_keep > 0.0,
            "--neg-keep must be in (0, 1], got {}",
            self.neg_keep
        );
        assert!(
            self.pos_weight_power.is_finite() && self.pos_weight_power >= 0.0,
            "--pos-weight-power must be finite and non-negative, got {}",
            self.pos_weight_power
        );
        assert!(
            self.lr.is_finite() && self.lr > 0.0,
            "--lr must be finite and positive, got {}",
            self.lr
        );
        assert!(
            self.lr_decay.is_finite() && self.lr_decay > 0.0,
            "--lr-decay must be finite and positive, got {}",
            self.lr_decay
        );
    }

    fn cold_start(&self) -> ColdStart {
        ColdStart {
            seed: self.seed,
            init_scale: self.init_scale,
            relu_warm_bias: self.relu_warm_bias,
        }
    }

    fn label_of(&self, s: &Loaded) -> f32 {
        match self.label {
            TrainLabel::Tight => s.tight,
            TrainLabel::Strict => s.strict,
        }
    }
}

/// Whether the negative at `index` is trained on in `epoch`: a hash of
/// `(seed, epoch, index)` against `neg_keep`, so the subsample is a
/// deterministic function of the run and differs between epochs.
fn negative_is_kept(seed: u64, epoch: usize, index: usize, neg_keep: f32) -> bool {
    if neg_keep >= 1.0 {
        return true;
    }
    let mut z = seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (index as u64) << 1;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32 <= neg_keep
}

fn train(samples: &Path, out: &Path, folds: Option<&[String]>, args: &TrainArgs) {
    args.validate();
    let all = load_samples(samples);
    std::fs::create_dir_all(out).expect("create model dir");
    let rules = RuleSet::production();
    let mut models = BTreeMap::new();
    let registered: Vec<&str> = FOLDS.iter().copied().chain([ALL]).collect();
    let folds: Vec<&str> = match folds {
        None => registered.clone(),
        Some(requested) => requested
            .iter()
            .map(|f| {
                *registered
                    .iter()
                    .find(|r| **r == f.as_str())
                    .unwrap_or_else(|| panic!("--folds {f:?}: not one of {registered:?}"))
            })
            .collect(),
    };
    for held in folds {
        let started = Instant::now();
        let uncapped: Vec<usize> = (0..all.len())
            .filter(|&i| held == ALL || all[i].fold != held)
            .collect();
        let stride = if args.train_cap == 0 {
            1
        } else {
            uncapped.len().div_ceil(args.train_cap).max(1)
        };
        let train_idx: Vec<usize> = uncapped.into_iter().step_by(stride).collect();
        let hold_idx: Vec<usize> = (0..all.len())
            .filter(|&i| held != ALL && all[i].fold == held)
            .collect();
        assert!(!train_idx.is_empty(), "fold {held}: no training samples");
        let n_tight = train_idx.iter().filter(|&&i| all[i].tight > 0.5).count();
        let n_strict = train_idx.iter().filter(|&&i| all[i].strict > 0.5).count();
        let n_pos = train_idx
            .iter()
            .filter(|&&i| args.label_of(&all[i]) > 0.5)
            .count();
        let n_neg = train_idx.len() - n_pos;
        assert!(
            n_pos > 0 && n_neg > 0,
            "fold {held}: one class is absent from training"
        );
        let pos_weight = (n_neg as f32 * args.neg_keep / n_pos as f32).powf(args.pos_weight_power);
        assert!(
            pos_weight.is_finite() && pos_weight > 0.0,
            "fold {held}: pos_weight {pos_weight} is not a weight"
        );
        let invisible = train_idx
            .iter()
            .filter(|&&i| all[i].tight > 0.5 && all[i].strict < 0.5)
            .count() as f64
            / n_tight.max(1) as f64;

        let mut trainer = BilinearTrainer::cold_start(&rules, args.cold_start());
        let mut epoch_loss = Vec::new();
        let mut lr = args.lr;
        let batch_scale = 1.0 / args.batch_size as f32;
        for epoch in 0..args.epochs {
            let order = shuffled(train_idx.len(), args.seed.wrapping_add(epoch as u64));
            let step = SgdStep {
                lr,
                l2: args.l2,
                max_grad_norm: args.max_grad_norm,
            };
            let mut loss = 0.0f64;
            let mut seen = 0usize;
            let mut pending = 0usize;
            for &j in &order {
                let s = &all[train_idx[j]];
                let y = args.label_of(s);
                if y < 0.5 && !negative_is_kept(args.seed, epoch, j, args.neg_keep) {
                    continue;
                }
                let summary = summary_of(&trainer, s);
                let forward = trainer.forward(&summary);
                let z = forward.score();
                let p = sigmoid(z);
                let w = if y > 0.5 { pos_weight } else { 1.0 };
                let eps = 1e-7f32;
                let l = if y > 0.5 {
                    -(p.max(eps)).ln()
                } else {
                    -((1.0 - p).max(eps)).ln()
                };
                loss += f64::from(w * l);
                seen += 1;
                trainer.accumulate(&forward, weighted_bce_grad(p, y, pos_weight) * batch_scale);
                pending += 1;
                if pending == args.batch_size {
                    trainer.apply(step);
                    pending = 0;
                }
            }
            if pending > 0 {
                trainer.apply(step);
            }
            assert!(seen > 0, "fold {held}: epoch {epoch} trained on no samples");
            let mean = loss / seen as f64;
            epoch_loss.push(mean);
            eprintln!(
                "fold {held}: epoch {epoch} lr {lr:.4} samples {seen} mean weighted loss {mean:.4}"
            );
            lr *= args.lr_decay;
        }

        // Scores on the training samples calibrate the thresholds; on the
        // held-out fold they are the intrinsic metric.
        let score = |i: usize| trainer.score(&summary_of(&trainer, &all[i]));
        let train_scores: Vec<f32> = train_idx.iter().map(|&i| score(i)).collect();
        let mut rule_pos: BTreeMap<RuleId, (usize, usize)> = BTreeMap::new();
        for &i in &train_idx {
            let e = rule_pos.entry(all[i].rule).or_insert((0, 0));
            e.0 += 1;
            e.1 += usize::from(all[i].tight > 0.5);
        }
        let rule_rate =
            |r: RuleId| -> f32 { rule_pos.get(&r).map_or(0.0, |&(n, p)| p as f32 / n as f32) };
        let train_rates: Vec<f32> = train_idx.iter().map(|&i| rule_rate(all[i].rule)).collect();
        let mut bilinear_threshold = BTreeMap::new();
        let mut bilinear_realized = BTreeMap::new();
        let mut per_rule_threshold = BTreeMap::new();
        let mut per_rule_realized = BTreeMap::new();
        for rho in KEEP_RATES {
            let key = format!("{rho}");
            let t = keep_threshold(&train_scores, rho);
            bilinear_realized.insert(key.clone(), realized_keep_rate(&train_scores, t));
            bilinear_threshold.insert(key.clone(), t);
            let t = keep_threshold(&train_rates, rho);
            per_rule_realized.insert(key.clone(), realized_keep_rate(&train_rates, t));
            per_rule_threshold.insert(key, t);
        }
        let holdout = (!hold_idx.is_empty()).then(|| {
            let scores: Vec<f32> = hold_idx.iter().map(|&i| score(i)).collect();
            let rates: Vec<f32> = hold_idx.iter().map(|&i| rule_rate(all[i].rule)).collect();
            let tight: Vec<f32> = hold_idx.iter().map(|&i| all[i].tight).collect();
            let strict: Vec<f32> = hold_idx.iter().map(|&i| all[i].strict).collect();
            Intrinsic {
                n: hold_idx.len(),
                positive_rate_tight: tight.iter().sum::<f32>() as f64 / tight.len() as f64,
                positive_rate_strict: strict.iter().sum::<f32>() as f64 / strict.len() as f64,
                bilinear_auc_tight: auc_roc(&scores, &tight),
                bilinear_pr_auc_tight: average_precision(&scores, &tight),
                bilinear_auc_strict: auc_roc(&scores, &strict),
                bilinear_pr_auc_strict: average_precision(&scores, &strict),
                per_rule_auc_tight: auc_roc(&rates, &tight),
                per_rule_pr_auc_tight: average_precision(&rates, &tight),
                per_rule_auc_strict: auc_roc(&rates, &strict),
                per_rule_pr_auc_strict: average_precision(&rates, &strict),
            }
        });

        let weights = trainer.weights();
        let mut checkpoint = BilinearGuideCheckpoint {
            schema_identity: BilinearGuideCheckpoint::current_schema_identity(),
            label_source: "tight-at-seam".into(),
            trainer: "rules_filter train".into(),
            written_at_unix_s: pixelflow_pipeline::schema::unix_now_s(),
            seed: args.seed,
            epochs: args.epochs,
            lr_initial: args.lr,
            lr_decay: args.lr_decay,
            l2: args.l2,
            max_grad_norm: args.max_grad_norm,
            pos_weight,
            rule_fingerprint: format!("{}", weights.fingerprint),
            num_rules: rules.len(),
            op_names: OpKind::all().map(|op| format!("{op:?}")).collect(),
            parameters: weights.parameters.clone(),
            op_embeddings: weights.op_embeddings.clone(),
            train_samples: train_idx.len(),
            train_families: 0,
            train_positive_rate: n_pos as f64 / train_idx.len() as f64,
            dev_samples: hold_idx.len(),
            dev_families: usize::from(held != ALL),
            // The all-DEV model has no held-out fold; the checkpoint's two
            // informational metrics carry `-1.0` there (NaN serializes as
            // `null`, which the reader refuses). The manifest holds the
            // intrinsic numbers that count.
            dev_auc: holdout
                .as_ref()
                .and_then(|h| h.bilinear_auc_tight)
                .unwrap_or(-1.0),
            dev_pr_auc: holdout
                .as_ref()
                .and_then(|h| h.bilinear_pr_auc_tight)
                .unwrap_or(-1.0),
            weights_fnv64: String::new(),
        };
        checkpoint.weights_fnv64 = checkpoint.weights_fingerprint();
        let path = out.join(format!("bilinear_{held}.json"));
        std::fs::write(
            &path,
            serde_json::to_string(&checkpoint).expect("serialize"),
        )
        .expect("write checkpoint");

        let label_of = |id: RuleId| -> String {
            rules
                .index_of(id)
                .and_then(|i| rules.label_of(i))
                .unwrap_or_else(|| panic!("rule {id} is not in the production rule set"))
        };
        let model = FoldModel {
            fold: held.into(),
            checkpoint: path
                .file_name()
                .expect("file name")
                .to_string_lossy()
                .into(),
            train_samples: train_idx.len(),
            train_folds: FOLDS
                .iter()
                .filter(|&&f| held == ALL || f != held)
                .map(|f| (*f).into())
                .collect(),
            train_positive_rate_tight: n_tight as f64 / train_idx.len() as f64,
            train_positive_rate_strict: n_strict as f64 / train_idx.len() as f64,
            tight_positives_invisible_to_strict: invisible,
            pos_weight,
            epoch_loss,
            bilinear_threshold,
            bilinear_realized_keep_rate: bilinear_realized,
            rule_rates: rule_pos
                .keys()
                .map(|&r| (label_of(r), rule_rate(r)))
                .collect(),
            per_rule_threshold,
            per_rule_realized_keep_rate: per_rule_realized,
            holdout,
        };
        if let Some(h) = &model.holdout {
            eprintln!(
                "fold {held}: held-out n={} tight AUC {:?} PR-AUC {:?} (per-rule {:?} / {:?}); strict AUC {:?} PR-AUC {:?} (per-rule {:?} / {:?}) ({:.0}s)",
                h.n,
                h.bilinear_auc_tight,
                h.bilinear_pr_auc_tight,
                h.per_rule_auc_tight,
                h.per_rule_pr_auc_tight,
                h.bilinear_auc_strict,
                h.bilinear_pr_auc_strict,
                h.per_rule_auc_strict,
                h.per_rule_pr_auc_strict,
                started.elapsed().as_secs_f64()
            );
        }
        models.insert(held.to_string(), model);
    }
    let manifest = Manifest {
        schema: SCHEMA.into(),
        git_sha: head_sha(),
        args: args.clone(),
        models,
    };
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("serialize"),
    )
    .expect("write manifest");
}

fn head_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

// ── eval ────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Row {
    schema: String,
    git_sha: String,
    kernel: String,
    family: String,
    fold: String,
    /// `B`, `B/2`, `B/4`, `B/8`.
    b_label: String,
    budget: u64,
    /// The kernel's production application count.
    b: u64,
    /// `identity`, `per_rule`, `uniform`, `bilinear`.
    arm: String,
    /// Keep-rate ρ (0 for identity).
    rho: f32,
    /// `heldout` (the model that never saw this family), `all`, or `-`.
    model: String,
    dag_cost: usize,
    bytes: u32,
    guarded: Option<u64>,
    schedule: Option<u64>,
    applications: u64,
    iterations: usize,
    classes: usize,
    stop: String,
    cells: u64,
    kept: u64,
    scored: u64,
    macs: u64,
    filter_ms: f64,
    wall_ms: f64,
    /// `uniform_matched` only: the keep-rate it was run at — the realized
    /// keep-rate of the bilinear arm at the same budget and nominal ρ.
    matched_keep_rate: Option<f64>,
}

struct Models {
    manifest: Manifest,
    weights: BTreeMap<String, pixelflow_search::nnue::guide::bilinear::BilinearWeights>,
    rates: BTreeMap<String, BTreeMap<RuleId, f32>>,
}

fn load_models(dir: &Path) -> Models {
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("read manifest");
    let manifest: Manifest = serde_json::from_str(&text).expect("parse manifest");
    let rules = RuleSet::production();
    let mut weights = BTreeMap::new();
    let mut rates = BTreeMap::new();
    for (fold, model) in &manifest.models {
        let text = std::fs::read_to_string(dir.join(&model.checkpoint)).expect("read checkpoint");
        let checkpoint: BilinearGuideCheckpoint =
            serde_json::from_str(&text).expect("parse checkpoint");
        weights.insert(fold.clone(), checkpoint.to_weights(&rules));
        rates.insert(
            fold.clone(),
            model
                .rule_rates
                .iter()
                .map(|(label, &rate)| (RuleId::from_label(label), rate))
                .collect(),
        );
    }
    Models {
        manifest,
        weights,
        rates,
    }
}

enum Arm<'m> {
    Identity,
    PerRule {
        fold: &'m str,
        rho: f32,
    },
    Uniform {
        rho: f32,
    },
    /// `UniformRandom` at the keep-rate the bilinear arm actually realized
    /// at this budget and nominal ρ — the control the registration means by
    /// "the same keep-rate", since a threshold calibrated on the training
    /// distribution keeps a different fraction of a held-out family.
    UniformMatched {
        rho: f32,
        keep_rate: f32,
        model: &'static str,
    },
    Bilinear {
        fold: &'m str,
        rho: f32,
        model: &'static str,
    },
}

impl<'m> Arm<'m> {
    fn clone_shallow(&self) -> Arm<'m> {
        match self {
            Arm::Identity => Arm::Identity,
            Arm::PerRule { fold, rho } => Arm::PerRule { fold, rho: *rho },
            Arm::Uniform { rho } => Arm::Uniform { rho: *rho },
            Arm::UniformMatched {
                rho,
                keep_rate,
                model,
            } => Arm::UniformMatched {
                rho: *rho,
                keep_rate: *keep_rate,
                model,
            },
            Arm::Bilinear { fold, rho, model } => Arm::Bilinear {
                fold,
                rho: *rho,
                model,
            },
        }
    }

    fn columns(&self) -> (&'static str, f32, String) {
        match self {
            Arm::Identity => ("identity", 0.0, "-".into()),
            Arm::PerRule { rho, .. } => ("per_rule", *rho, "heldout".into()),
            Arm::Uniform { rho } => ("uniform", *rho, "-".into()),
            Arm::UniformMatched { rho, model, .. } => ("uniform_matched", *rho, (*model).into()),
            Arm::Bilinear { rho, model, .. } => ("bilinear", *rho, (*model).into()),
        }
    }

    fn filter(
        &self,
        models: &Models,
        episode: Episode,
        reporting: &Reporting,
    ) -> Box<dyn ApplicationFilter> {
        let rules = RuleSet::production();
        match self {
            Arm::Identity => Box::new(KeepAll),
            Arm::PerRule { fold, rho } => {
                let model = &models.manifest.models[*fold];
                let threshold = model.per_rule_threshold[&format!("{rho}")];
                Box::new(
                    PerRuleRateFilter::new(models.rates[*fold].clone(), threshold)
                        .reporting(Rc::clone(reporting)),
                )
            }
            Arm::Uniform { rho } => Box::new(
                UniformRandomFilter::new(UNIFORM_SEED, *rho).reporting(Rc::clone(reporting)),
            ),
            Arm::UniformMatched { keep_rate, .. } => Box::new(
                UniformRandomFilter::new(UNIFORM_SEED, *keep_rate).reporting(Rc::clone(reporting)),
            ),
            Arm::Bilinear { fold, rho, .. } => {
                let model = &models.manifest.models[*fold];
                let threshold = model.bilinear_threshold[&format!("{rho}")];
                Box::new(
                    BilinearFilter::new(&models.weights[*fold], &rules, threshold, episode)
                        .expect("checkpoint fits the production vocabulary")
                        .reporting(Rc::clone(reporting)),
                )
            }
        }
    }
}

struct RunOut {
    emitted: Emitted,
    stats: pixelflow_search::egraph::OptimizerStats,
    filter: FilterStats,
    wall_ms: f64,
}

fn run_arm(p: &Prepared<'_>, budget: Budget, arm: &Arm<'_>, models: &Models) -> RunOut {
    let reporting: Reporting = Rc::default();
    let mut optimizer = Optimizer::production()
        .for_lattice(p.shape)
        .budget(budget)
        .filter(arm.filter(models, p.episode(), &reporting))
        .no_ceiling();
    let mut egraph = optimizer.egraph();
    let root_class = insert(&p.arena, p.root, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|_| panic!("{}: not representable", p.kernel.name));
    let started = Instant::now();
    let optimized = optimizer.run(&mut egraph, root_class, p.node_count);
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let (arena, root) = optimized.to_arena(&egraph, root_class);
    RunOut {
        emitted: emit_extracted(p, &arena, root),
        stats: optimized.stats,
        filter: reporting.get(),
        wall_ms,
    }
}

fn eval(
    models_dir: &Path,
    out: &Path,
    held_out: bool,
    shard_spec: Option<&str>,
    filter: Option<&str>,
) {
    assert!(
        std::env::var_os("PIXELFLOW_GUARD_TELEMETRY").is_some(),
        "set PIXELFLOW_GUARD_TELEMETRY=1: the guard columns come from the emitter's own report"
    );
    let models = load_models(models_dir);
    let kernels = shard(
        if held_out {
            held_out_kernels(filter)
        } else {
            dev_kernels(filter)
        },
        shard_spec,
    );
    let mut sink = open_append(out);
    let git_sha = head_sha();
    eprintln!("eval: {} kernels, held_out={held_out}", kernels.len());
    for (i, kernel) in kernels.iter().enumerate() {
        let started = Instant::now();
        let p = prepare(kernel);
        let fold = fold_of(kernel.class.trim_start_matches("bold_"));
        let production = run_arm(&p, Budget::Production, &Arm::Identity, &models);
        let b = production.stats.applications;
        let mut arms: Vec<Arm<'_>> = vec![Arm::Identity];
        for rho in KEEP_RATES {
            if !held_out {
                arms.push(Arm::PerRule { fold, rho });
                arms.push(Arm::Bilinear {
                    fold,
                    rho,
                    model: "heldout",
                });
            }
            arms.push(Arm::Uniform { rho });
            arms.push(Arm::Bilinear {
                fold: ALL,
                rho,
                model: "all",
            });
        }
        let mut seen_budgets = BTreeSet::new();
        for (b_label, divisor) in BUDGET_DIVISORS {
            let budget = b / divisor;
            if budget == 0 || !seen_budgets.insert(budget) {
                continue;
            }
            let mut queue: Vec<Arm<'_>> = arms.iter().map(Arm::clone_shallow).collect();
            queue.reverse();
            while let Some(arm) = queue.pop() {
                let out = run_arm(&p, Budget::Applications(budget), &arm, &models);
                if b_label == "B" && matches!(arm, Arm::Identity) {
                    assert_eq!(
                        (out.emitted.dag_cost, out.emitted.bytes),
                        (production.emitted.dag_cost, production.emitted.bytes),
                        "{}: Identity at Applications(B = {b}) must reproduce production",
                        kernel.name
                    );
                }
                if let Arm::Bilinear { rho, model, .. } = &arm
                    && out.filter.cells > 0
                {
                    queue.push(Arm::UniformMatched {
                        rho: *rho,
                        keep_rate: (out.filter.kept as f64 / out.filter.cells as f64) as f32,
                        model,
                    });
                }
                let matched_keep_rate = match &arm {
                    Arm::UniformMatched { keep_rate, .. } => Some(f64::from(*keep_rate)),
                    _ => None,
                };
                let (arm_name, rho, model) = arm.columns();
                let row = Row {
                    schema: SCHEMA.into(),
                    git_sha: git_sha.clone(),
                    kernel: kernel.name.clone(),
                    family: kernel.class.clone(),
                    fold: fold.into(),
                    b_label: b_label.into(),
                    budget,
                    b,
                    arm: arm_name.into(),
                    rho,
                    model,
                    dag_cost: out.emitted.dag_cost,
                    bytes: out.emitted.bytes,
                    guarded: out.emitted.guarded,
                    schedule: out.emitted.schedule,
                    applications: out.stats.applications,
                    iterations: out.stats.iterations,
                    classes: out.stats.classes,
                    stop: format!("{:?}", out.stats.stop),
                    cells: out.filter.cells,
                    kept: out.filter.kept,
                    scored: out.filter.scored,
                    macs: out.filter.macs,
                    filter_ms: out.filter.filter_ns as f64 / 1e6,
                    wall_ms: out.wall_ms,
                    matched_keep_rate,
                };
                writeln!(sink, "{}", serde_json::to_string(&row).expect("serialize"))
                    .expect("write");
            }
            sink.flush().expect("flush");
        }
        eprintln!(
            "[{}/{}] {} B={} dag={} bytes={} ({:.1}s)",
            i + 1,
            kernels.len(),
            kernel.name,
            b,
            production.emitted.dag_cost,
            production.emitted.bytes,
            started.elapsed().as_secs_f64()
        );
    }
}

// ── report ──────────────────────────────────────────────────────────────────

fn median(v: &mut [f64]) -> f64 {
    assert!(!v.is_empty());
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let k = ((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
    v[k]
}

#[derive(Serialize, Clone)]
struct Cell {
    family: String,
    b_label: String,
    arm: String,
    rho: f32,
    model: String,
    n: usize,
    dag_median: f64,
    dag_p10: f64,
    dag_p90: f64,
    bytes_median: f64,
    bytes_p10: f64,
    bytes_p90: f64,
    guarded_sum_arm: u64,
    guarded_sum_identity: u64,
    applications_median: f64,
    classes_median: f64,
    iterations_median: f64,
    keep_rate: f64,
    macs_sum: u64,
    filter_ms_sum: f64,
    wall_ms_sum: f64,
    identity_wall_ms_sum: f64,
}

fn aggregate(rows: &[Row]) -> Vec<Cell> {
    let mut identity: BTreeMap<(String, String), &Row> = BTreeMap::new();
    for r in rows {
        if r.arm == "identity" {
            identity.insert((r.kernel.clone(), r.b_label.clone()), r);
        }
    }
    let mut groups: BTreeMap<(String, String, String, String, String), Vec<&Row>> = BTreeMap::new();
    for r in rows {
        groups
            .entry((
                r.family.clone(),
                r.b_label.clone(),
                r.arm.clone(),
                format!("{}", r.rho),
                r.model.clone(),
            ))
            .or_default()
            .push(r);
    }
    let mut out = Vec::new();
    for ((family, b_label, arm, rho, model), members) in groups {
        let mut dag = Vec::new();
        let mut bytes = Vec::new();
        let mut apps = Vec::new();
        let mut classes = Vec::new();
        let mut iters = Vec::new();
        let (mut g_arm, mut g_id, mut macs, mut cells, mut kept) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (mut f_ms, mut w_ms, mut iw_ms) = (0.0, 0.0, 0.0);
        for r in &members {
            let id = identity[&(r.kernel.clone(), r.b_label.clone())];
            dag.push(r.dag_cost as f64 / id.dag_cost.max(1) as f64);
            bytes.push(f64::from(r.bytes) / f64::from(id.bytes.max(1)));
            apps.push(r.applications as f64);
            classes.push(r.classes as f64);
            iters.push(r.iterations as f64);
            g_arm += r.guarded.unwrap_or(0);
            g_id += id.guarded.unwrap_or(0);
            macs += r.macs;
            cells += r.cells;
            kept += r.kept;
            f_ms += r.filter_ms;
            w_ms += r.wall_ms;
            iw_ms += id.wall_ms;
        }
        out.push(Cell {
            family,
            b_label,
            arm,
            rho: rho.parse().expect("rho"),
            model,
            n: members.len(),
            dag_median: median(&mut dag.clone()),
            dag_p10: quantile(&mut dag.clone(), 0.1),
            dag_p90: quantile(&mut dag, 0.9),
            bytes_median: median(&mut bytes.clone()),
            bytes_p10: quantile(&mut bytes.clone(), 0.1),
            bytes_p90: quantile(&mut bytes, 0.9),
            guarded_sum_arm: g_arm,
            guarded_sum_identity: g_id,
            applications_median: median(&mut apps),
            classes_median: median(&mut classes),
            iterations_median: median(&mut iters),
            keep_rate: if cells == 0 {
                1.0
            } else {
                kept as f64 / cells as f64
            },
            macs_sum: macs,
            filter_ms_sum: f_ms,
            wall_ms_sum: w_ms,
            identity_wall_ms_sum: iw_ms,
        });
    }
    out
}

/// `(family, arm, ρ, model)`.
type ArmKey = (String, String, String, String);
/// Per kernel, the `(budget label, dag_cost)` pairs one arm produced.
type KernelCosts<'a> = BTreeMap<String, Vec<(&'a str, usize)>>;

/// The dual: per (family, arm, ρ, model), how many kernels reach
/// `Identity@B`'s dag_cost at each budget of the grid.
#[derive(Serialize, Clone)]
struct Reach {
    family: String,
    arm: String,
    rho: f32,
    model: String,
    n: usize,
    reached_at: BTreeMap<String, usize>,
    never: usize,
}

fn reach(rows: &[Row]) -> Vec<Reach> {
    let mut identity_b: BTreeMap<String, usize> = BTreeMap::new();
    for r in rows {
        if r.arm == "identity" && r.b_label == "B" {
            identity_b.insert(r.kernel.clone(), r.dag_cost);
        }
    }
    let mut per: BTreeMap<ArmKey, KernelCosts<'_>> = BTreeMap::new();
    for r in rows {
        per.entry((
            r.family.clone(),
            r.arm.clone(),
            format!("{}", r.rho),
            r.model.clone(),
        ))
        .or_default()
        .entry(r.kernel.clone())
        .or_default()
        .push((r.b_label.as_str(), r.dag_cost));
    }
    let order: Vec<&str> = BUDGET_DIVISORS.iter().rev().map(|(l, _)| *l).collect();
    let mut out = Vec::new();
    for ((family, arm, rho, model), kernels) in per {
        let mut reached_at: BTreeMap<String, usize> =
            order.iter().map(|l| ((*l).to_string(), 0)).collect();
        let mut never = 0;
        for (kernel, costs) in &kernels {
            let target = identity_b[kernel];
            let first = order
                .iter()
                .find(|l| costs.iter().any(|(bl, c)| bl == *l && *c <= target));
            match first {
                Some(l) => *reached_at.get_mut(*l).expect("label") += 1,
                None => never += 1,
            }
        }
        out.push(Reach {
            family,
            arm,
            rho: rho.parse().expect("rho"),
            model,
            n: kernels.len(),
            reached_at,
            never,
        });
    }
    out
}

/// Inference-cost totals for one (family, arm).
#[derive(Default)]
struct CostTotals {
    cells: u64,
    scored: u64,
    macs: u64,
    filter_ms: f64,
    wall_ms: f64,
    identity_wall_ms: f64,
}

fn arm_label(c: &Cell) -> String {
    match c.arm.as_str() {
        "identity" => "Identity".into(),
        "per_rule" => format!("PerRuleRate ρ={}", c.rho),
        "uniform" => format!("UniformRandom ρ={}", c.rho),
        "uniform_matched" => format!(
            "UniformRandom@Bilinear[{}]'s realized rate, ρ={}",
            c.model, c.rho
        ),
        "bilinear" => format!("Bilinear[{}] ρ={}", c.model, c.rho),
        other => other.into(),
    }
}

fn family_table(md: &mut String, cells: &[Cell], b_label: &str) {
    use std::fmt::Write as _;
    let _ = writeln!(
        md,
        "| family | arm | n | dag_cost ratio median (p10 / p90) | bytes ratio median (p10 / p90) | Σ guarded arm / Identity | median applications | median classes | median rounds | keep-rate |"
    );
    let _ = writeln!(md, "|---|---|---:|---|---|---|---:|---:|---:|---:|");
    for c in cells.iter().filter(|c| c.b_label == b_label) {
        let _ = writeln!(
            md,
            "| {} | {} | {} | {:.3} ({:.3} / {:.3}) | {:.3} ({:.3} / {:.3}) | {} / {} | {:.0} | {:.0} | {:.0} | {:.2} |",
            c.family,
            arm_label(c),
            c.n,
            c.dag_median,
            c.dag_p10,
            c.dag_p90,
            c.bytes_median,
            c.bytes_p10,
            c.bytes_p90,
            c.guarded_sum_arm,
            c.guarded_sum_identity,
            c.applications_median,
            c.classes_median,
            c.iterations_median,
            c.keep_rate
        );
    }
}

fn report(
    rows_paths: &[PathBuf],
    held_out_paths: &[PathBuf],
    models_dir: &Path,
    out_prefix: &Path,
) {
    use std::fmt::Write as _;
    let rows: Vec<Row> = rows_paths
        .iter()
        .flat_map(|p| read_jsonl::<Row>(p))
        .collect();
    let held: Vec<Row> = held_out_paths
        .iter()
        .flat_map(|p| read_jsonl::<Row>(p))
        .collect();
    let models = load_models(models_dir);
    let cells = aggregate(&rows);
    let reaches = reach(&rows);
    let held_cells = if held.is_empty() {
        Vec::new()
    } else {
        aggregate(&held)
    };

    // Verdict per the registration §7, at B/2, ρ = 0.25, family-held-out.
    let mut verdict_lines = Vec::new();
    let families: BTreeSet<String> = rows.iter().map(|r| r.family.clone()).collect();
    for family in &families {
        let find = |arm: &str, model: &str| {
            cells.iter().find(|c| {
                c.family == *family
                    && c.b_label == "B/2"
                    && c.arm == arm
                    && (c.rho - 0.25).abs() < 1e-6
                    && c.model == model
            })
        };
        let Some(bi) = find("bilinear", "heldout") else {
            continue;
        };
        let uni = find("uniform_matched", "heldout").or_else(|| find("uniform", "-"));
        let prr = find("per_rule", "heldout");
        let clause1 = bi.dag_median <= 0.95;
        let clause2 = bi.bytes_median <= 1.0;
        let uniform_matches =
            uni.is_some_and(|u| u.dag_median >= bi.dag_p10 && u.dag_median <= bi.dag_p90);
        let per_rule_matches =
            prr.is_some_and(|p| p.dag_median >= bi.dag_p10 && p.dag_median <= bi.dag_p90);
        let verdict = if clause1 && clause2 && !uniform_matches && !per_rule_matches {
            "WIN on DEV (held-out sign pending)"
        } else if clause1 && clause2 {
            "NULL — a control matches within the band"
        } else {
            "NULL"
        };
        verdict_lines.push(format!(
            "| {family} | {:.3} | {:.3} | {} | {} | {} | {} |",
            bi.dag_median,
            bi.bytes_median,
            uni.map_or("-".into(), |u| format!("{:.3}", u.dag_median)),
            prr.map_or("-".into(), |p| format!("{:.3}", p.dag_median)),
            if uniform_matches || per_rule_matches {
                "yes"
            } else {
                "no"
            },
            verdict
        ));
    }

    let mut md = String::new();
    let _ = writeln!(md, "# The bilinear rules × nodes filter on real shaders\n");
    let _ = writeln!(
        md,
        "**Date:** 2026-09-08. **Registration:** docs/plans/2026-09-08-rules-filter-bilinear-registration.md — every number below is read against its §7 decision rule. Rows: `{}` at git `{}`; models: `{}` at git `{}`.\n",
        rows_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        rows.first().map_or("?", |r| r.git_sha.as_str()),
        models_dir.display(),
        models.manifest.git_sha
    );
    let _ = writeln!(
        md,
        "## Verdict (§7: B/2, ρ = 0.25, family-held-out model)\n"
    );
    let _ = writeln!(
        md,
        "| family | Bilinear dag_cost ratio (median) | Bilinear bytes ratio (median) | UniformRandom (matched keep-rate) dag median | PerRuleRate dag median | a control inside Bilinear's p10–p90 band | verdict |"
    );
    let _ = writeln!(md, "|---|---:|---:|---:|---:|---|---|");
    for l in &verdict_lines {
        let _ = writeln!(md, "{l}");
    }
    let _ = writeln!(
        md,
        "\n## Intrinsic metric (held-out fold samples; the training set never contains the fold)\n"
    );
    let _ = writeln!(
        md,
        "| fold | n held-out | tight positive rate | Bilinear AUC / PR-AUC (tight) | PerRuleRate AUC / PR-AUC (tight) | strict positive rate | Bilinear AUC / PR-AUC (strict) | PerRuleRate AUC / PR-AUC (strict) | train n | tight positives invisible to strict | keep-rate realized at ρ=0.25 (bilinear / per-rule) |"
    );
    let _ = writeln!(md, "|---|---:|---:|---|---|---:|---|---|---:|---:|---|");
    for (fold, m) in &models.manifest.models {
        let f = |v: Option<f64>| v.map_or("-".into(), |x| format!("{x:.3}"));
        let h = m.holdout.as_ref();
        let _ = writeln!(
            md,
            "| {fold} | {} | {} | {} / {} | {} / {} | {} | {} / {} | {} / {} | {} | {:.3} | {:.3} / {:.3} |",
            h.map_or(0, |h| h.n),
            h.map_or("-".into(), |h| format!("{:.3}", h.positive_rate_tight)),
            f(h.and_then(|h| h.bilinear_auc_tight)),
            f(h.and_then(|h| h.bilinear_pr_auc_tight)),
            f(h.and_then(|h| h.per_rule_auc_tight)),
            f(h.and_then(|h| h.per_rule_pr_auc_tight)),
            h.map_or("-".into(), |h| format!("{:.4}", h.positive_rate_strict)),
            f(h.and_then(|h| h.bilinear_auc_strict)),
            f(h.and_then(|h| h.bilinear_pr_auc_strict)),
            f(h.and_then(|h| h.per_rule_auc_strict)),
            f(h.and_then(|h| h.per_rule_pr_auc_strict)),
            m.train_samples,
            m.tight_positives_invisible_to_strict,
            m.bilinear_realized_keep_rate
                .get("0.25")
                .copied()
                .unwrap_or(f64::NAN),
            m.per_rule_realized_keep_rate
                .get("0.25")
                .copied()
                .unwrap_or(f64::NAN),
        );
    }
    for (b_label, _) in BUDGET_DIVISORS {
        let _ = writeln!(md, "\n## Per family at {b_label}\n");
        family_table(&mut md, &cells, b_label);
    }
    let _ = writeln!(
        md,
        "\n## The dual: kernels reaching Identity@B's dag_cost, by the first budget that does\n"
    );
    let _ = writeln!(
        md,
        "| family | arm | n | at B/8 | at B/4 | at B/2 | at B | never |"
    );
    let _ = writeln!(md, "|---|---|---:|---:|---:|---:|---:|---:|");
    for r in &reaches {
        let label = match r.arm.as_str() {
            "identity" => "Identity".to_string(),
            "per_rule" => format!("PerRuleRate ρ={}", r.rho),
            "uniform" => format!("UniformRandom ρ={}", r.rho),
            "uniform_matched" => format!(
                "UniformRandom@Bilinear[{}]'s realized rate, ρ={}",
                r.model, r.rho
            ),
            _ => format!("Bilinear[{}] ρ={}", r.model, r.rho),
        };
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            r.family,
            label,
            r.n,
            r.reached_at["B/8"],
            r.reached_at["B/4"],
            r.reached_at["B/2"],
            r.reached_at["B"],
            r.never
        );
    }
    let _ = writeln!(
        md,
        "\n## Inference cost (bilinear arms, summed over the family's kernels and budgets)\n"
    );
    let _ = writeln!(
        md,
        "| family | arm | Σ cells | Σ scored | Σ multiply-adds | Σ filter ms | Σ run ms | Σ Identity run ms | filter share of run |"
    );
    let _ = writeln!(md, "|---|---|---:|---:|---:|---:|---:|---:|---:|");
    let mut per_family: BTreeMap<(String, String), CostTotals> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.arm == "bilinear") {
        let id_wall = rows
            .iter()
            .find(|i| i.arm == "identity" && i.kernel == r.kernel && i.b_label == r.b_label)
            .map_or(0.0, |i| i.wall_ms);
        let e = per_family
            .entry((
                r.family.clone(),
                format!("Bilinear[{}] ρ={}", r.model, r.rho),
            ))
            .or_default();
        e.cells += r.cells;
        e.scored += r.scored;
        e.macs += r.macs;
        e.filter_ms += r.filter_ms;
        e.wall_ms += r.wall_ms;
        e.identity_wall_ms += id_wall;
    }
    for ((family, arm), t) in &per_family {
        let _ = writeln!(
            md,
            "| {family} | {arm} | {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.1}% |",
            t.cells,
            t.scored,
            t.macs,
            t.filter_ms,
            t.wall_ms,
            t.identity_wall_ms,
            100.0 * t.filter_ms / t.wall_ms.max(1e-9)
        );
    }
    if !held_cells.is_empty() {
        let _ = writeln!(md, "\n## HELD-OUT, opened once (all-DEV model)\n");
        for (b_label, _) in BUDGET_DIVISORS {
            let _ = writeln!(md, "\n### {b_label}\n");
            family_table(&mut md, &held_cells, b_label);
        }
    }

    std::fs::write(out_prefix.with_extension("md"), &md).expect("write md");
    let mut csv = String::from(
        "family,b_label,arm,rho,model,n,dag_median,dag_p10,dag_p90,bytes_median,bytes_p10,bytes_p90,guarded_sum_arm,guarded_sum_identity,applications_median,classes_median,iterations_median,keep_rate,macs_sum,filter_ms_sum,wall_ms_sum,identity_wall_ms_sum,set\n",
    );
    for (set, cs) in [("dev", &cells), ("held_out", &held_cells)] {
        for c in cs {
            let _ = writeln!(
                csv,
                "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.1},{:.1},{:.1},{:.4},{},{:.1},{:.1},{:.1},{set}",
                c.family,
                c.b_label,
                c.arm,
                c.rho,
                c.model,
                c.n,
                c.dag_median,
                c.dag_p10,
                c.dag_p90,
                c.bytes_median,
                c.bytes_p10,
                c.bytes_p90,
                c.guarded_sum_arm,
                c.guarded_sum_identity,
                c.applications_median,
                c.classes_median,
                c.iterations_median,
                c.keep_rate,
                c.macs_sum,
                c.filter_ms_sum,
                c.wall_ms_sum,
                c.identity_wall_ms_sum
            );
        }
    }
    std::fs::write(out_prefix.with_extension("csv"), csv).expect("write csv");
    let json = serde_json::json!({
        "schema": SCHEMA,
        "registration": "docs/plans/2026-09-08-rules-filter-bilinear-registration.md",
        "rows_git_sha": rows.first().map(|r| r.git_sha.clone()),
        "models": models.manifest,
        "dev": cells,
        "held_out": held_cells,
        "reach": reaches,
    });
    std::fs::write(
        out_prefix.with_extension("json"),
        serde_json::to_string_pretty(&json).expect("serialize"),
    )
    .expect("write json");
    eprintln!("report: wrote {}.{{md,csv,json}}", out_prefix.display());
}

fn main() {
    match Cli::parse().command {
        Command::Mint {
            out,
            per_kernel_cap,
            filter,
        } => mint(&out, per_kernel_cap, filter.as_deref()),
        Command::Train {
            samples,
            out,
            folds,
            args,
        } => train(&samples, &out, folds.as_deref(), &args),
        Command::Eval {
            models,
            out,
            held_out,
            shard,
            filter,
        } => eval(&models, &out, held_out, shard.as_deref(), filter.as_deref()),
        Command::Report {
            rows,
            held_out_rows,
            models,
            out_prefix,
        } => report(&rows, &held_out_rows, &models, &out_prefix),
    }
}
