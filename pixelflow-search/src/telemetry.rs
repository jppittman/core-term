//! Feature-flagged saturation telemetry: one JSONL record per production
//! e-graph optimizer invocation, gated entirely behind the
//! `saturation-telemetry` cargo feature (std-only, default OFF).
//!
//! # Why a side channel, not a return value
//!
//! `EGraph::optimize_runtime_arena` (this crate's [`crate::runtime`]) and
//! `pixelflow_compiler::optimize::optimize` keep their existing signatures
//! and return types unchanged — telemetry is never threaded through what
//! either function hands back. Each production call site calls
//! [`record`] itself, immediately after its own `saturate_with_full_budget`
//! + extract, only when this feature is compiled in. With the feature off,
//! this module does not exist (see the `#[cfg]` on its declaration in
//! `lib.rs`) and there is nothing left in the binary to call.
//!
//! # Sink
//!
//! One JSON object per line, appended to the path named by the
//! `PIXELFLOW_SATURATION_TELEMETRY` environment variable if set, otherwise
//! written to stderr. Opening or writing the sink is a hard failure
//! (`panic!`) — a telemetry record that silently fails to land is
//! indistinguishable from "nothing happened", which is exactly the failure
//! mode this instrument exists to catch (see this workspace's no-silent-
//! failures rule).
//!
//! # Usage
//!
//! ```text
//! cargo run -p core-term --features saturation-telemetry
//! PIXELFLOW_SATURATION_TELEMETRY=/tmp/sat.jsonl cargo run -p core-term --features saturation-telemetry
//! ```

use std::io::Write as _;
use std::time::Duration;

use crate::egraph::{CostModel, SaturationResult, SaturationStop};
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

/// Everything one production optimizer invocation — one
/// `saturate_with_full_budget` call plus the extraction that followed it —
/// knows about itself, for [`record`] to serialize.
pub struct SaturationInvocation<'a> {
    /// Which tier invoked saturation: `"runtime"`
    /// ([`crate::runtime::optimize_runtime_arena`]) or `"macro"`
    /// (`pixelflow_compiler::optimize`, running inside rustc at macro
    /// expansion time).
    pub tier: &'static str,
    /// Size of the input, as passed to `config_for_node_count` to select the
    /// budget triple below.
    pub node_count: usize,
    /// The `SaturationConfig` budget this run was given.
    pub max_iterations: usize,
    pub max_classes: usize,
    pub hard_timeout: Duration,
    /// The result `saturate_with_full_budget` returned: iterations used,
    /// e-classes before/after, and — the field this feature exists to
    /// surface — why the run stopped.
    pub result: &'a SaturationResult,
    /// Rule-provenance counters, read off the e-graph's own journal at the
    /// moment saturation stopped (`Provenance::application_count` /
    /// `Provenance::union_count`) — never inferred from the stats above.
    pub application_count: usize,
    pub union_count: usize,
    /// The arena and root this invocation extracted, so [`record`] can cost
    /// it under the static latency-prior model regardless of which
    /// extraction policy actually chose it (static or NNUE).
    pub extracted_arena: &'a ExprArena,
    pub extracted_root: ExprId,
    /// Wall-clock of saturate+extract together. Indicative only — see
    /// `CLAUDE.md`'s floating-point-at-the-edges notes on why timing is a
    /// measurement, not a promised bound.
    pub wall_clock: Duration,
    /// A label for the kernel being optimized, when the call site has one
    /// (e.g. a named `kernel!`'s struct name, or a source span). Never
    /// invented: `None` when the call site genuinely has nothing to name
    /// (an anonymous kernel, or a runtime-composed `Kernel` with no source
    /// identity).
    pub kernel_label: Option<&'a str>,
}

/// Emit one JSONL telemetry record for `inv`. See the module docs for the
/// sink and its failure behavior.
pub fn record(inv: SaturationInvocation<'_>) {
    let cost = latency_prior_cost(inv.extracted_arena, inv.extracted_root);
    let line = format!(
        "{{\"tier\":\"{tier}\",\"node_count\":{node_count},\"max_iterations\":{max_iterations},\
         \"max_classes\":{max_classes},\"hard_timeout_us\":{hard_timeout_us},\
         \"stop_reason\":\"{stop_reason}\",\"iterations\":{iterations},\
         \"classes_at_stop\":{classes_at_stop},\"application_count\":{application_count},\
         \"union_count\":{union_count},\"extracted_latency_prior_cost\":{cost},\
         \"wall_clock_us\":{wall_clock_us},\"kernel_label\":{kernel_label}}}",
        tier = inv.tier,
        node_count = inv.node_count,
        max_iterations = inv.max_iterations,
        max_classes = inv.max_classes,
        hard_timeout_us = inv.hard_timeout.as_micros(),
        stop_reason = stop_str(inv.result.stop),
        iterations = inv.result.iterations,
        classes_at_stop = inv.result.classes_after,
        application_count = inv.application_count,
        union_count = inv.union_count,
        cost = cost,
        wall_clock_us = inv.wall_clock.as_micros(),
        kernel_label = json_opt_str(inv.kernel_label),
    );
    write_line(&line);
}

fn stop_str(stop: SaturationStop) -> &'static str {
    match stop {
        SaturationStop::Quiesced => "quiesced",
        SaturationStop::ClassCap => "class_cap",
        SaturationStop::IterationCeiling => "iteration_ceiling",
        SaturationStop::Timeout => "timeout",
    }
}

fn json_opt_str(s: Option<&str>) -> String {
    match s {
        Some(s) => format!("\"{}\"", escape_json(s)),
        None => "null".to_string(),
    }
}

/// Minimal JSON string escaping — telemetry labels are source identifiers
/// or spans, never arbitrary user text, but this keeps a stray quote or
/// backslash from producing an invalid JSONL line instead of guessing it
/// can't happen.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

/// Sum of `CostModel::latency_prior()` over every node reachable from
/// `root`. Computed independently here rather than threaded out of
/// extraction because neither `Extraction` nor `choices_to_arena`'s output
/// carries a total cost of its own (extraction tracks per-e-class best cost
/// internally, not on the materialized arena) — this mirrors
/// `crate::runtime`'s own `reachable_count` traversal shape.
fn latency_prior_cost(arena: &ExprArena, root: ExprId) -> usize {
    let costs = CostModel::latency_prior();
    let len = arena.nodes_raw().len();
    let mut seen = vec![false; len];
    let mut stack = vec![root];
    let mut total = 0usize;
    while let Some(id) = stack.pop() {
        if core::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        let op = match arena.node(id) {
            ExprNode::Unary(op, _)
            | ExprNode::Binary(op, _, _)
            | ExprNode::Ternary(op, _, _, _)
            | ExprNode::Nary(op, _, _) => Some(*op),
            ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) | ExprNode::Buffer(_) => {
                None
            }
        };
        if let Some(op) = op {
            total += costs.cost(op);
        }
        stack.extend(arena.children(id));
    }
    total
}

/// Append one line to the sink named by `PIXELFLOW_SATURATION_TELEMETRY`, or
/// stderr when unset. Open/write failures panic.
fn write_line(line: &str) {
    match std::env::var_os("PIXELFLOW_SATURATION_TELEMETRY") {
        Some(path) => {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| {
                    panic!("saturation-telemetry: failed to open {path:?} for append: {e}")
                });
            writeln!(file, "{line}").unwrap_or_else(|e| {
                panic!("saturation-telemetry: failed to write to {path:?}: {e}")
            });
        }
        None => {
            eprintln!("{line}");
        }
    }
}
