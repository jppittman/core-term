//! The gate for the `Extractor` seam's one load-bearing property:
//! **`Beam` is never dearer than `Greedy`.**
//!
//! `pixelflow-search`'s own unit tests hold this on four hand-built saturated
//! graphs, and it passed there while `Beam::width(64)` returned a **dearer**
//! term than `Greedy` on six of the 95 DejaVu glyphs
//! (`docs/results/2026-09-08-beam-extraction.md`). The gap was the *repair*:
//! `Beam` compared its states before `repair_choices_well_founded` ran, and a
//! state cheaper pre-repair can repair to a dearer term. Small graphs never
//! exercise it, because the repair only fires on the cyclic choice maps
//! saturation produces at scale.
//!
//! So the check is here, on the exact six glyphs that regressed, at the exact
//! cap they regressed at — a class of bug invisible to every job we ran is a
//! job to write, not a caveat in a PR description. `Beam` now folds the DP's
//! own repaired term back in as a third arm; this is what says so next time.

use pixelflow_graphics::fonts::Font;
use pixelflow_ir::LatticeShape;
use pixelflow_pipeline::shader_bench::{SHADERTOY_KERNEL_NAMES, named_shadertoy_kernel};
use pixelflow_search::egraph::{
    Beam, Budget, CostModel, EClassId, EGraph, Extractor, Greedy, Optimizer, Vocabulary, insert,
};

/// The `glyph16` tile the class-cap sweep measures, and its class cap.
const TILE_PX: f32 = 16.0;
const GLYPH_CLASS_CAP: usize = 5_000;

/// The six glyphs `Beam::width(64)` returned a dearer term for than `Greedy`
/// before the DP arm was folded back in — `S`, `f`, `j`, `l`, `t`, `~`.
const REGRESSED: [char; 6] = ['S', 'f', 'j', 'l', 't', '~'];

/// Above every shader's own production tier, so these graphs are big enough
/// for the repair to have something to repair.
const SHADER_CLASS_CAP: usize = 10_000;

fn saturated(
    arena: &pixelflow_ir::ExprArena,
    root: pixelflow_ir::ExprId,
    cap: usize,
    what: &str,
) -> (EGraph, EClassId) {
    let mut optimizer = Optimizer::production().budget(Budget::Explicit {
        iterations: 30,
        classes: cap,
        applications: Some(cap as u64 * 40),
    });
    let mut egraph = optimizer.egraph();
    let root_class = insert(arena, root, &mut egraph, Vocabulary::Runtime)
        .unwrap_or_else(|d| panic!("{what}: not e-graph representable ({d:?})"));
    // Saturate once; both extractors then read the SAME graph, so any
    // difference between them is the extractor's and nothing else's.
    let _ = optimizer.run(&mut egraph, root_class, 0);
    (egraph, root_class)
}

fn assert_beam_never_dearer(egraph: &EGraph, root: EClassId, what: &str, widths: &[usize]) {
    let costs = CostModel::latency_prior();
    let shape = LatticeShape::POINT;
    let greedy = Greedy.extract(egraph, root, &costs, shape);
    for &k in widths {
        let beam = Beam::width(k).extract(egraph, root, &costs, shape);
        assert!(
            beam.dag_cost <= greedy.dag_cost,
            "{what}: Beam({k}) returned {} against Greedy's {} — Beam folds the DP's own \
             repaired term back in precisely so this cannot happen",
            beam.dag_cost,
            greedy.dag_cost
        );
    }
}

fn production_font() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../pixelflow-graphics/assets/DejaVuSansMono-Fallback.ttf");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn beam_is_never_dearer_than_greedy_on_the_six_glyphs_that_regressed() {
    let data = production_font();
    let font = Font::parse(&data).expect("parse the production font");
    for ch in REGRESSED {
        let kernel = font
            .glyph_kernel_scaled(ch, TILE_PX)
            .unwrap_or_else(|| panic!("the production font has no glyph for {ch:?}"));
        let (arena, root) = kernel.parts();
        let what = alloc_name(ch);
        let (egraph, root_class) = saturated(arena, root, GLYPH_CLASS_CAP, &what);
        assert_beam_never_dearer(&egraph, root_class, &what, &[16, 64]);
    }
}

#[test]
fn beam_is_never_dearer_than_greedy_on_the_shader_corpus() {
    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name).expect("registered shader");
        let (egraph, root_class) = saturated(&arena, root, SHADER_CLASS_CAP, name);
        assert_beam_never_dearer(&egraph, root_class, name, &[4, 16]);
    }
}

/// Width 1 is `Greedy`, on real saturated graphs and not only on hand-built
/// ones — the identity the whole seam rests on.
#[test]
fn width_one_is_greedy_on_the_shader_corpus() {
    let costs = CostModel::latency_prior();
    for name in SHADERTOY_KERNEL_NAMES {
        let (arena, root) = named_shadertoy_kernel(name).expect("registered shader");
        let (egraph, root_class) = saturated(&arena, root, SHADER_CLASS_CAP, name);
        for shape in [LatticeShape::POINT, LatticeShape::new([256, 256])] {
            let a = Beam::width(1).extract(&egraph, root_class, &costs, shape);
            let b = Greedy.extract(&egraph, root_class, &costs, shape);
            assert_eq!(
                a.choices, b.choices,
                "{name} @ {shape:?}: choice maps differ"
            );
            assert_eq!(
                a.dag_cost, b.dag_cost,
                "{name} @ {shape:?}: dag_cost differs"
            );
            assert_eq!(a.report, b.report, "{name} @ {shape:?}: report differs");
        }
    }
}

fn alloc_name(ch: char) -> String {
    format!("glyph16_U{:04X}", ch as u32)
}
