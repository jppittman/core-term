//! `KeepAll` ≡ production: the identity test the rules × nodes filter seam
//! owes (docs/plans/2026-09-08-rules-by-nodes-filter.md).
//!
//! Two probes over the same observables — emitted machine-code bytes,
//! `dag_cost`, extracted node count, stop reason, applications, rounds,
//! classes, unions:
//!
//! - [`pinned_dev_subset_is_byte_identical`] (fast, runs in CI): the twelve
//!   `shader_bench` ports plus four DejaVu glyph bakes, compared against the
//!   goldens in `tests/fixtures/rules_by_nodes_identity.json`, which were
//!   written by the loop **before** the seam existed (commit `e57760c0`, the
//!   denotation-only commit). A goldens diff means the seam is not a no-op;
//!   fix the wiring, never the goldens. Regenerate them only when the rules
//!   or the budget change, with [`regen_goldens`].
//! - [`full_dev_corpus_is_byte_identical`] (`#[ignore]`): every kernel
//!   `egraph_off_on` compiles — 210 rows — against a baseline JSONL the
//!   pre-change binary wrote, named by `PIXELFLOW_IDENTITY_BASELINE`.
//!
//! The fast probe compiles each kernel twice: once through the explicit
//! `Optimizer::production()` path (to read the saturation stats) and once
//! through `optimize_runtime_arena` (production's own entry), and asserts
//! the two emit the same bytes — so the numbers it pins are production's,
//! not a model of production.

use std::path::{Path, PathBuf};
use std::process::Command;

use pixelflow_codegen::emit;
use pixelflow_graphics::fonts::Font;
use pixelflow_ir::optimize::{Optimize, Rewritten};
use pixelflow_ir::passes::{ExpandReduce, LowerDwrt};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, pipeline};
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::egraph::{Optimizer, SaturationStop, Vocabulary, insert, reachable_count};
use serde::{Deserialize, Serialize};

const SHADER_EXTENT: [u32; 2] = [256, 256];
/// `GlyphAtlas::new(16pt, density, ..)`'s tile at density 1 and 2 — the
/// two tiles `egraph_off_on` warms.
const GLYPH_TILES: [u32; 2] = [16, 32];
const GLYPH_CHARS: [char; 2] = ['O', 'S'];
const GOLDENS: &str = "tests/fixtures/rules_by_nodes_identity.json";
const BASELINE_VAR: &str = "PIXELFLOW_IDENTITY_BASELINE";
/// Columns of an `egraph_off_on` row that are not a function of the input:
/// the clock, the commit, and the mode label.
const NONDETERMINISTIC_COLUMNS: [&str; 6] = [
    "git_sha",
    "optimize_ms",
    "emit_ms",
    "clock",
    "probe",
    "mode",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Golden {
    name: String,
    input_nodes: usize,
    extracted_nodes: usize,
    dag_cost: usize,
    tree_cost: usize,
    bytes: usize,
    code_fnv: u64,
    stop: String,
    iterations: usize,
    applications: u64,
    unions: usize,
    classes: usize,
}

fn fnv_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn legalize(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    match pipeline![LowerDwrt, ExpandReduce].optimize(arena, root) {
        Rewritten::Changed(a, r) => (a, r),
        Rewritten::Unchanged => (arena.clone(), root),
        Rewritten::Declined => panic!("legalizing prefix declined a DEV kernel"),
    }
}

fn reachable_nodes(arena: &ExprArena, root: ExprId) -> usize {
    reachable_count(arena, root)
}

fn stop_label(stop: SaturationStop) -> String {
    format!("{stop:?}")
}

/// The production pipeline spelled out — `optimize_runtime_arena_uncached`'s
/// three steps with the optimizer in hand, so the stats are readable — then
/// emitted.
fn observe(name: &str, arena: &ExprArena, root: ExprId, extent: [u32; 2]) -> Golden {
    let shape = LatticeShape::new(extent);
    let (legal, legal_root) = legalize(arena, root);
    let mut optimizer = Optimizer::production().for_lattice(shape);
    let mut egraph = optimizer.egraph();
    let root_class = insert(&legal, legal_root, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|d| panic!("{name}: runtime vocabulary declined the kernel: {d:?}"));
    let node_count = reachable_count(&legal, legal_root);
    let optimized = optimizer.run(&mut egraph, root_class, node_count);
    let (extracted, extracted_root) = optimized.to_arena(&egraph, root_class);
    let result = emit::compile(&extracted, extracted_root)
        .unwrap_or_else(|e| panic!("{name}: emit::compile failed: {e:?}"));
    let code = result.code.as_bytes();

    // Production's own entry point must agree byte for byte, or the numbers
    // above describe something other than production.
    let production = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape)
        .unwrap_or_else(|| panic!("{name}: optimize_runtime_arena declined"));
    let (p_arena, p_root) = (&production.0, production.1);
    let p_code = emit::compile(p_arena, p_root)
        .unwrap_or_else(|e| panic!("{name}: production emit failed: {e:?}"));
    assert_eq!(
        p_code.code.as_bytes(),
        code,
        "{name}: the explicit Optimizer path and optimize_runtime_arena emitted different bytes"
    );

    Golden {
        name: name.to_string(),
        input_nodes: reachable_nodes(&legal, legal_root),
        extracted_nodes: reachable_nodes(&extracted, extracted_root),
        dag_cost: optimized.cost.dag,
        tree_cost: optimized.cost.tree,
        bytes: code.len(),
        code_fnv: fnv_bytes(code),
        stop: stop_label(optimized.stats.stop),
        iterations: optimized.stats.iterations,
        applications: optimized.stats.applications,
        unions: optimized.stats.unions,
        classes: optimized.stats.classes,
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn dejavu() -> PathBuf {
    manifest_dir().join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf")
}

/// The pinned DEV subset, in a fixed order: twelve arena-native shader ports
/// and four `Kernel`-composed glyph bakes (two characters at both warmed
/// tiles), so both ways a real kernel reaches the e-graph are covered.
fn pinned_subset() -> Vec<Golden> {
    let mut out = Vec::new();
    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name).expect("registered shader");
        out.push(observe(
            &format!("shader_{name}"),
            &arena,
            root,
            SHADER_EXTENT,
        ));
    }
    let font = dejavu();
    let data = std::fs::read(&font).unwrap_or_else(|e| panic!("read {}: {e}", font.display()));
    let parsed = Font::parse(&data).expect("parse DejaVuSansMono-Fallback");
    for tile in GLYPH_TILES {
        for ch in GLYPH_CHARS {
            let kernel = parsed
                .glyph_kernel_scaled(ch, tile as f32)
                .unwrap_or_else(|| panic!("no glyph for {ch:?}"));
            let (arena, root) = kernel.parts();
            out.push(observe(
                &format!("glyph{tile}_U{:04X}", ch as u32),
                arena,
                root,
                [tile, tile],
            ));
        }
    }
    out
}

fn goldens_path() -> PathBuf {
    manifest_dir().join(GOLDENS)
}

#[test]
fn pinned_dev_subset_is_byte_identical() {
    let path = goldens_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e} — regenerate with `cargo test -p pixelflow-pipeline --release \
             --test rules_by_nodes_identity regen_goldens -- --ignored`",
            path.display()
        )
    });
    let expected: Vec<Golden> = serde_json::from_str(&text).expect("parse goldens");
    let actual = pinned_subset();
    assert_eq!(
        expected.len(),
        actual.len(),
        "the pinned subset changed size: goldens hold {} kernels, the corpus has {}",
        expected.len(),
        actual.len()
    );
    let mut differing = Vec::new();
    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(e.name, a.name, "pinned subset order changed");
        if e != a {
            differing.push(format!("{}:\n  expected {e:?}\n  actual   {a:?}", e.name));
        }
    }
    assert!(
        differing.is_empty(),
        "{} of {} pinned DEV kernels differ from the pre-seam loop — the seam is not a no-op; \
         fix the wiring, never the goldens:\n{}",
        differing.len(),
        actual.len(),
        differing.join("\n")
    );
}

/// Rewrite the goldens from the current loop. Only when the rules or the
/// budget change — never to make the identity test pass.
#[test]
#[ignore = "writes tests/fixtures/rules_by_nodes_identity.json from the current loop"]
fn regen_goldens() {
    let goldens = pinned_subset();
    let path = goldens_path();
    std::fs::create_dir_all(path.parent().expect("fixtures dir")).expect("create fixtures dir");
    let mut text = serde_json::to_string_pretty(&goldens).expect("serialize goldens");
    text.push('\n');
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    eprintln!("wrote {} goldens to {}", goldens.len(), path.display());
}

/// Every deterministic column of every `egraph_off_on` row, against a
/// baseline the pre-change binary wrote. Run as
///
/// ```text
/// PIXELFLOW_IDENTITY_BASELINE=/path/to/pre.jsonl \
///   cargo test -p pixelflow-pipeline --release --test rules_by_nodes_identity \
///   full_dev_corpus_is_byte_identical -- --ignored --nocapture
/// ```
#[test]
#[ignore = "full corpus (210 kernels, ~20 s release): needs PIXELFLOW_IDENTITY_BASELINE=<pre-change rows.jsonl>"]
fn full_dev_corpus_is_byte_identical() {
    let baseline = std::env::var_os(BASELINE_VAR).unwrap_or_else(|| {
        panic!(
            "{BASELINE_VAR} is unset: point it at a rows JSONL written by `egraph_off_on run \
             --no-clock --no-probe` from the pre-change binary"
        )
    });
    let out = std::env::temp_dir().join(format!(
        "rules_by_nodes_identity-{}.jsonl",
        std::process::id()
    ));
    if out.exists() {
        std::fs::remove_file(&out).expect("clear stale rows");
    }
    let status = Command::new(env!("CARGO_BIN_EXE_egraph_off_on"))
        .env("PIXELFLOW_GUARD_TELEMETRY", "1")
        .args(["run", "--no-clock", "--no-probe", "--out"])
        .arg(&out)
        .status()
        .expect("spawn egraph_off_on");
    assert!(status.success(), "egraph_off_on run failed: {status}");

    let expected = read_rows(Path::new(&baseline));
    let actual = read_rows(&out);
    assert_eq!(
        expected.len(),
        actual.len(),
        "baseline has {} rows, this run {}",
        expected.len(),
        actual.len()
    );
    let mut differing = Vec::new();
    for (e, a) in expected.iter().zip(&actual) {
        assert_eq!(e["name"], a["name"], "corpus order changed");
        let mut e = e.clone();
        let mut a = a.clone();
        for col in NONDETERMINISTIC_COLUMNS {
            e.as_object_mut().expect("row object").remove(col);
            a.as_object_mut().expect("row object").remove(col);
        }
        if e != a {
            differing.push(e["name"].to_string());
        }
    }
    eprintln!(
        "rules_by_nodes_identity: {} of {} rows byte-identical to {}",
        actual.len() - differing.len(),
        actual.len(),
        Path::new(&baseline).display()
    );
    assert!(
        differing.is_empty(),
        "{} of {} kernels differ from the pre-change loop: {differing:?}",
        differing.len(),
        actual.len()
    );
}

fn read_rows(path: &Path) -> Vec<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse row"))
        .collect()
}
