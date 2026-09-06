//! The kernels the harness measures, and the fixture format that stores them.
//!
//! A cost model is fitted to *what production compiles*, so the corpus is led
//! by the glyph bakes `core-term` runs (`GlyphAtlas::warm`, one fused kernel
//! per printable character at each display density) and filled out by three
//! synthetic families that isolate the structures the allocator trades
//! against. Each entry carries the **shape** it is baked at, because that is
//! what turns a static count into a dynamic one — the omission that made the
//! previous allocator measurements unable to see their own units
//! (`docs/plans/2026-09-01-register-allocation-escape-hatches.md`, 3″).
//!
//! Capture writes the corpus once, to files; the bench replays those files.
//! So the corpus is a fixture with a diff, not a side effect of whichever
//! test suite happened to run, and every allocation variant is measured on
//! byte-identical input.

use std::path::Path;

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId, ExprNode};

/// One kernel and the lattice it is baked at.
pub struct CollapseKernel {
    /// Unique within a corpus; the row key in the output.
    pub name: String,
    /// Which family it came from — the grouping the analysis reports by.
    pub family: String,
    pub arena: ExprArena,
    pub root: ExprId,
    /// The lattice extent, exactly as `Lattice::bake` would see it.
    pub extent: [u32; 4],
}

/// How many times each scope of the collapse nest runs, for one
/// `call_collapse` at this extent and vector width.
///
/// The collapse ABI's own arithmetic: the frame prologue runs once, the row
/// prologue once per row, the body once per full SIMD group per row. The
/// scalar tail `Lattice::bake` walks afterwards is not this kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trips {
    pub rows: u64,
    pub groups: u64,
}

impl Trips {
    /// # Panics
    /// If the extent is narrower than one SIMD group — the collapse kernel is
    /// never called for such a lattice, so there would be nothing to time —
    /// or if it names a Z/W plane, which is a separate call.
    #[must_use]
    pub fn of(extent: [u32; 4], lanes: u32) -> Self {
        assert!(
            extent[0] >= lanes,
            "extent {extent:?} is narrower than the {lanes}-lane batch: bake would run \
             the scalar tail only and never call the collapse kernel"
        );
        assert!(
            extent[2] == 1 && extent[3] == 1,
            "extent {extent:?}: the corpus is 2D — Z/W planes are separate calls"
        );
        Self {
            rows: u64::from(extent[1]),
            groups: u64::from(extent[0] / lanes),
        }
    }
}

// =============================================================================
// Fixture format
// =============================================================================

const HEADER: &str = "# pixelflow collapse corpus v1";

/// Write `kernels` into `dir`, one `.collapse` file each.
///
/// The node encoding is the arena dumpers' (`pixelflow-core`'s cell-grid
/// dumper, `pixelflow-graphics`'s glyph dumper): reachable nodes in ascending
/// id order with ids remapped dense, constants as bit patterns. The additions
/// are the `family` and `extent` lines — the shape, which is the point.
///
/// # Panics
/// If the directory cannot be created, a file cannot be written, or a kernel
/// contains a node kind the runtime optimizer bails on (`Param`, `Nary`,
/// `Buffer`), which would make the fixture unlike anything production bakes.
pub fn write_dir(dir: &Path, kernels: &[CollapseKernel]) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    for kernel in kernels {
        let path = dir.join(format!("{}.collapse", kernel.name));
        std::fs::write(&path, encode(kernel))
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

/// Read every `.collapse` file in `dir`, in name order.
///
/// # Panics
/// If the directory cannot be read or any file is not a v1 fixture.
#[must_use]
pub fn read_dir(dir: &Path) -> Vec<CollapseKernel> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "collapse"))
        .collect();
    paths.sort();
    paths.iter().map(|p| decode(p)).collect()
}

/// The fixture text for one kernel. Also the round-trip oracle: two kernels
/// with the same encoding are the same kernel.
#[must_use]
pub fn encode(kernel: &CollapseKernel) -> String {
    use std::fmt::Write as _;

    let (arena, root) = (&kernel.arena, kernel.root);
    let len = arena.nodes_raw().len();
    let mut reachable = vec![false; len];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut reachable[id.0 as usize], true) {
            continue;
        }
        stack.extend(arena.children(id));
    }

    let mut out = String::new();
    writeln!(out, "{HEADER}").expect("fmt");
    writeln!(out, "name {}", kernel.name).expect("fmt");
    writeln!(out, "family {}", kernel.family).expect("fmt");
    let [ex, ey, ez, ew] = kernel.extent;
    writeln!(out, "extent {ex} {ey} {ez} {ew}").expect("fmt");

    let mut dense: Vec<u32> = vec![u32::MAX; len];
    let mut next = 0u32;
    let d = |dense: &[u32], id: ExprId| -> u32 {
        let v = dense[id.0 as usize];
        assert_ne!(v, u32::MAX, "child dumped before parent");
        v
    };
    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let id = ExprId(idx as u32);
        match arena.node(id) {
            ExprNode::Var(i) => writeln!(out, "V {i}"),
            ExprNode::Const(v) => writeln!(out, "C {}", v.to_bits()),
            ExprNode::Unary(k, a) => writeln!(out, "U {k:?} {}", d(&dense, *a)),
            ExprNode::Binary(k, a, b) => {
                writeln!(out, "Bi {k:?} {} {}", d(&dense, *a), d(&dense, *b))
            }
            ExprNode::Ternary(k, a, b, c) => writeln!(
                out,
                "T {k:?} {} {} {}",
                d(&dense, *a),
                d(&dense, *b),
                d(&dense, *c)
            ),
            other => panic!(
                "{}: corpus kernels must be bakeable, but this one holds {other:?}",
                kernel.name
            ),
        }
        .expect("fmt");
        dense[idx] = next;
        next += 1;
    }
    writeln!(out, "root {}", d(&dense, root)).expect("fmt");
    out
}

fn decode(path: &Path) -> CollapseKernel {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some(HEADER),
        "{}: not a collapse corpus fixture",
        path.display()
    );

    let mut name = None;
    let mut family = None;
    let mut extent = None;
    let mut root = None;
    let mut arena = ExprArena::new();
    let mut next_id = 0u32;

    let op = |s: &str| -> OpKind {
        OpKind::all()
            .find(|k| format!("{k:?}") == s)
            .unwrap_or_else(|| panic!("{}: unknown OpKind {s:?}", path.display()))
    };
    let id = |s: &str| -> ExprId {
        ExprId(
            s.parse()
                .unwrap_or_else(|e| panic!("{}: bad id {s:?}: {e}", path.display())),
        )
    };
    let dim = |s: &str| -> u32 {
        s.parse()
            .unwrap_or_else(|e| panic!("{}: bad extent {s:?}: {e}", path.display()))
    };

    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        let pushed = match f.as_slice() {
            ["name", n] => {
                name = Some((*n).to_string());
                continue;
            }
            ["family", n] => {
                family = Some((*n).to_string());
                continue;
            }
            ["extent", x, y, z, w] => {
                extent = Some([dim(x), dim(y), dim(z), dim(w)]);
                continue;
            }
            ["root", r] => {
                root = Some(id(r));
                continue;
            }
            ["V", i] => arena.push_var(i.parse().expect("var index")),
            ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
            ["U", k, a] => arena.push_unary(op(k), id(a)),
            ["Bi", k, a, b] => arena.push_binary(op(k), id(a), id(b)),
            ["T", k, a, b, c] => arena.push_ternary(op(k), id(a), id(b), id(c)),
            other => panic!("{}: unparseable line {other:?}", path.display()),
        };
        assert_eq!(
            pushed,
            ExprId(next_id),
            "{}: replay drifted from dumped ids",
            path.display()
        );
        next_id += 1;
    }

    CollapseKernel {
        name: name.unwrap_or_else(|| panic!("{}: no name", path.display())),
        family: family.unwrap_or_else(|| panic!("{}: no family", path.display())),
        arena,
        root: root.unwrap_or_else(|| panic!("{}: no root", path.display())),
        extent: extent.unwrap_or_else(|| panic!("{}: no extent", path.display())),
    }
}

// =============================================================================
// The synthetic families
// =============================================================================

/// Shape the pressure families are baked at: wide enough that the body
/// dominates, small enough to keep a sample cheap.
const PRESSURE_EXTENT: [u32; 4] = [256, 64, 1, 1];
/// Where a loop-invariant term is amortized over many body iterations.
const INVARIANT_HOT_EXTENT: [u32; 4] = [256, 256, 1, 1];
/// Where it is not: two rows of a few batch groups pay the prologues in full.
const INVARIANT_COLD_EXTENT: [u32; 4] = [64, 2, 1, 1];

/// Every synthetic kernel, in a fixed order.
///
/// Three families, each isolating one thing the allocator trades:
/// - `wide{n}` — a balanced tree of `n` leaves: transient pressure only, no
///   loop-invariant structure, so the prologues stay empty and the whole cost
///   is the body's;
/// - `anchored{w}x{d}` — `w` values computed up front and folded in after a
///   chain of depth `d`, so `w` live ranges cross `d` instructions: the shape
///   that makes eviction choose;
/// - `invariant{n}` — `n` X-invariant terms each read once by the body, at a
///   hot and a cold shape: the same static code with the trip count changed,
///   which is exactly the axis a static memory-op count cannot see.
#[must_use]
pub fn synthetic() -> Vec<CollapseKernel> {
    let mut out = Vec::new();
    let mut push =
        |name: String, family: &str, extent: [u32; 4], build: &dyn Fn(&mut ExprArena) -> ExprId| {
            let mut arena = ExprArena::new();
            let root = build(&mut arena);
            out.push(CollapseKernel {
                name,
                family: family.to_string(),
                arena,
                root,
                extent,
            });
        };
    for n in [8usize, 16, 32, 64] {
        push(
            format!("wide{n:03}"),
            "wide",
            PRESSURE_EXTENT,
            &move |a: &mut ExprArena| wide(a, n),
        );
    }
    for (w, d) in [(8usize, 24usize), (12, 40), (16, 64)] {
        push(
            format!("anchored{w:02}x{d:02}"),
            "anchored",
            PRESSURE_EXTENT,
            &move |a: &mut ExprArena| anchored(a, w, d),
        );
    }
    for n in [4usize, 8, 16, 48] {
        for (tag, extent) in [
            ("hot", INVARIANT_HOT_EXTENT),
            ("cold", INVARIANT_COLD_EXTENT),
        ] {
            push(
                format!("invariant{n:02}_{tag}"),
                &format!("invariant_{tag}"),
                extent,
                &move |a: &mut ExprArena| invariants(a, n),
            );
        }
    }
    out
}

/// A leaf that varies in X, salted so the tree is not one common
/// subexpression the optimizer folds away.
fn x_leaf(a: &mut ExprArena, salt: usize) -> ExprId {
    let x = a.push_var(0);
    let c = a.push_const(0.125 + (salt % 13) as f32 * 0.0625);
    a.push_binary(OpKind::Mul, x, c)
}

/// A balanced Add/Sub tree over `n` X-varying leaves.
fn wide(a: &mut ExprArena, n: usize) -> ExprId {
    assert!(n.is_power_of_two(), "wide takes a power of two, got {n}");
    let mut level: Vec<ExprId> = (0..n).map(|i| x_leaf(a, i)).collect();
    let mut salt = 0usize;
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                salt += 1;
                let op = if salt.is_multiple_of(3) {
                    OpKind::Sub
                } else {
                    OpKind::Add
                };
                a.push_binary(op, pair[0], pair[1])
            })
            .collect();
    }
    level[0]
}

/// `w` anchors computed first, a dependent chain of depth `d`, then the
/// anchors folded in — so every anchor is live across the whole chain.
fn anchored(a: &mut ExprArena, w: usize, d: usize) -> ExprId {
    let anchors: Vec<ExprId> = (0..w)
        .map(|i| {
            let leaf = x_leaf(a, i * 7 + 1);
            a.push_unary(OpKind::Sqrt, leaf)
        })
        .collect();
    let mut chain = x_leaf(a, 991);
    for i in 0..d {
        let c = a.push_const(1.0 + (i % 5) as f32 * 0.25);
        chain = a.push_ternary(OpKind::MulAdd, chain, c, chain);
    }
    anchors.iter().fold(chain, |acc, &anchor| {
        a.push_binary(OpKind::Add, acc, anchor)
    })
}

/// `n` terms invariant in X — half of them invariant in Y as well, so both
/// prologues get work — each read exactly once by an X-varying body term.
fn invariants(a: &mut ExprArena, n: usize) -> ExprId {
    let y = a.push_var(1);
    let z = a.push_var(2);
    let terms: Vec<ExprId> = (0..n)
        .map(|i| {
            let c = a.push_const(0.5 + i as f32 * 0.125);
            let base = if i.is_multiple_of(2) {
                // Frame scope: reads neither X nor Y.
                a.push_binary(OpKind::Mul, z, c)
            } else {
                // Row scope: reads Y.
                let scaled = a.push_binary(OpKind::Mul, y, c);
                a.push_binary(OpKind::Add, scaled, z)
            };
            let one = a.push_const(1.0);
            let positive = a.push_binary(OpKind::Add, base, one);
            a.push_unary(OpKind::Sqrt, positive)
        })
        .collect();
    let x = a.push_var(0);
    terms.iter().enumerate().fold(x, |acc, (i, &term)| {
        let leaf = x_leaf(a, i * 3 + 5);
        let scaled = a.push_binary(OpKind::Mul, leaf, term);
        a.push_binary(OpKind::Add, acc, scaled)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_synthetic_kernel_has_a_unique_name_and_a_bakeable_shape() {
        let kernels = synthetic();
        let mut names: Vec<&str> = kernels.iter().map(|k| k.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate kernel name in the corpus");
        for kernel in &kernels {
            // 16 lanes is the widest tier this repo emits; a corpus entry that
            // cannot fill one group there is one the bench would skip.
            let trips = Trips::of(kernel.extent, 16);
            assert!(trips.rows > 0 && trips.groups > 0, "{}", kernel.name);
        }
    }

    #[test]
    fn a_written_corpus_reads_back_identical() {
        let dir =
            std::env::temp_dir().join(format!("pixelflow-collapse-corpus-{}", std::process::id()));
        let kernels = synthetic();
        write_dir(&dir, &kernels);
        let back = read_dir(&dir);
        assert_eq!(back.len(), kernels.len());
        let mut sorted: Vec<&CollapseKernel> = kernels.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        for (want, got) in sorted.iter().zip(&back) {
            assert_eq!(
                encode(want),
                encode(got),
                "{}: the fixture did not round trip",
                want.name
            );
        }
        std::fs::remove_dir_all(&dir).expect("clean up");
    }
}
