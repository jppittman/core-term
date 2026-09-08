//! The real-kernel arena corpus, and the summary statistics the offline
//! probes report over it.
//!
//! Two `#[ignore]`d measurement probes now read the same corpus of `.arena`
//! dumps — `runtime::congruence_gap_probe` (issue #1106) and
//! `extraction_gap::probe` (this workflow's extraction-quality measurement).
//! The loader is the exact inverse of the three dumpers and has to stay that
//! way; a second copy of it is a future divergence, so it lives here once.

use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, BufferId, BufferIdentity, ExprArena, ExprId};
use std::path::Path;

/// Inverse of the dumpers' `dump_arena` (`pixelflow-core/src/lattice/cell_grid.rs`,
/// `pixelflow-graphics/tests/production_glyph_arena_dump.rs`,
/// `pixelflow-pipeline/tests/shader_and_psychedelic_arena_dump.rs`):
/// replays reachable nodes in original id order through the public
/// `push_*` API, which never hash-conses, so the rebuilt arena has
/// exactly the dumped node multiset.
pub fn load_arena_dump(path: &Path) -> (String, ExprArena, ExprId) {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("# pixelflow arena dump v1"),
        "{}: bad header",
        path.display()
    );
    let mut name = None;
    let mut arena = ExprArena::new();
    let mut idents: Vec<BufferIdentity> = Vec::new();
    let mut root = None;
    let mut next_id: u32 = 0;
    let mut buf_count: u16 = 0;
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
    for line in lines {
        let f: Vec<&str> = line.split_whitespace().collect();
        let pushed = match f.as_slice() {
            ["name", n] => {
                name = Some((*n).to_string());
                continue;
            }
            ["buf", ord, w, h] => {
                let ord: usize = ord.parse().expect("buf ordinal");
                while idents.len() <= ord {
                    idents.push(BufferIdentity::mint());
                }
                let slot = arena.declare_buffer(BufferDecl {
                    id: idents[ord],
                    width: w.parse().expect("buf width"),
                    height: h.parse().expect("buf height"),
                });
                assert_eq!(
                    slot.0,
                    buf_count,
                    "{}: buffer slot order drifted",
                    path.display()
                );
                buf_count += 1;
                continue;
            }
            ["root", r] => {
                root = Some(id(r));
                continue;
            }
            ["V", i] => arena.push_var(i.parse().expect("var index")),
            ["C", bits] => arena.push_const(f32::from_bits(bits.parse().expect("const bits"))),
            ["B", slot] => arena.push_buffer(BufferId(slot.parse().expect("buffer slot"))),
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
    let name = name.unwrap_or_else(|| panic!("{}: no name line", path.display()));
    let root = root.unwrap_or_else(|| panic!("{}: no root line", path.display()));
    (name, arena, root)
}

/// Which dumper produced a file, read off its name.
pub fn category_of(filename: &str) -> &'static str {
    if filename.starts_with("cellgrid_") {
        "cellgrid"
    } else if filename.starts_with("shader_") {
        "shader"
    } else if filename.starts_with("psychedelic") {
        "psychedelic"
    } else if filename.starts_with("glyph") {
        "glyph"
    } else {
        "unknown"
    }
}

/// Median of `xs`, sorting in place. NaN for an empty slice.
pub fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// Nearest-rank percentile of `xs`, sorting in place. NaN for an empty slice.
pub fn percentile(xs: &mut [f64], p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    let idx = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    xs[idx.min(n - 1)]
}
