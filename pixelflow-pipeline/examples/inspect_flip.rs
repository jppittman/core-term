//! Round 3 mechanism check (throwaway, not part of the training harness):
//! for a fixed list of DEV-tier kernel names, load TWO extraction-head
//! weight files (Round 2a's frozen-embedding checkpoint and Round 3's
//! trained-embedding checkpoint) under the SAME current e-graph/extractor
//! code, and print the NNUE-chosen form (plus the static-prior form and the
//! original) for each — isolating the weights' effect on the *choice* from
//! any code change between the two rounds.
//!
//! Run:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --example inspect_flip -- \
//!     --r2a-weights <path> --r3-weights <path> dev_bXX_fYY_ZZZZZ ...
//! ```

use std::path::PathBuf;

use clap::Parser;

use pixelflow_search::egraph::{CostModel, EGraph, IncrementalExtractor, all_rules, choices_to_arena, extract};
use pixelflow_search::nnue::ExprNnue;

use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::factored::arena_to_kernel_code;

const SATURATE_LIMIT: usize = 40;
const EXTRACT_TOP_K: usize = 8;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "pixelflow-pipeline/data/corpus_dev.bin")]
    corpus: PathBuf,
    #[arg(long)]
    r2a_weights: PathBuf,
    #[arg(long)]
    r3_weights: PathBuf,
    /// Print only "kernel r2a_nodes r3_nodes" lines, no expression text —
    /// for tallying node-count direction at scale.
    #[arg(long, default_value_t = false)]
    nodes_only: bool,
    kernels: Vec<String>,
}

fn main() {
    let args = Args::parse();
    let corpus = read_corpus(&args.corpus).expect("read corpus");
    let by_name: std::collections::HashMap<&str, (&pixelflow_ir::ExprArena, pixelflow_ir::ExprId)> =
        corpus.iter().map(|(n, a, r)| (n.as_str(), (a, *r))).collect();

    let r2a_bytes = std::fs::read(&args.r2a_weights).expect("read r2a weights");
    let r3_bytes = std::fs::read(&args.r3_weights).expect("read r3 weights");
    let r2a_nnue = ExprNnue::from_bytes(&r2a_bytes).expect("parse r2a weights");
    let r3_nnue = ExprNnue::from_bytes(&r3_bytes).expect("parse r3 weights");
    let static_costs = CostModel::latency_prior();

    for name in &args.kernels {
        let Some(&(arena, root)) = by_name.get(name.as_str()) else {
            println!("=== {name}: NOT FOUND IN CORPUS ===");
            continue;
        };
        let mut eg = EGraph::with_rules(all_rules());
        let root_class = eg.add_arena(arena, root);
        eg.saturate_with_limit(SATURATE_LIMIT);

        let mut node_counts = Vec::new();
        for nnue in [&r2a_nnue, &r3_nnue] {
            let extractor = IncrementalExtractor::new(nnue, EXTRACT_TOP_K);
            let (_cost, choices) = extractor.extract_choices_only(&eg, root_class);
            let (cand_arena, cand_root) = choices_to_arena(&choices);
            node_counts.push(cand_arena.node_count_subtree(cand_root));
        }

        if args.nodes_only {
            println!("{name} {} {}", node_counts[0], node_counts[1]);
            continue;
        }

        println!("\n=== {name} (orig nodes={}) ===", arena.node_count_subtree(root));
        println!("ORIGINAL : {}", arena_to_kernel_code(arena, root));

        let (static_arena, static_root, static_cost) = extract(&eg, root_class, &static_costs);
        println!(
            "STATIC   : nodes={} cost={:.3} :: {}",
            static_arena.node_count_subtree(static_root),
            static_cost,
            arena_to_kernel_code(&static_arena, static_root)
        );

        for (label, nnue) in [("R2A-NNUE", &r2a_nnue), ("R3-NNUE ", &r3_nnue)] {
            let extractor = IncrementalExtractor::new(nnue, EXTRACT_TOP_K);
            let (cost, choices) = extractor.extract_choices_only(&eg, root_class);
            let (cand_arena, cand_root) = choices_to_arena(&choices);
            println!(
                "{label} : nodes={} cost={:.3} :: {}",
                cand_arena.node_count_subtree(cand_root),
                cost,
                arena_to_kernel_code(&cand_arena, cand_root)
            );
        }
    }
}
