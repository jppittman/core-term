//! **What the runtime tier emits for a glyph, bounded.**
//!
//! Every other glyph test asks whether the pixels are right. None of them
//! asks how much code it took, so a change that leaves coverage identical and
//! doubles the instruction count ships green — which is exactly what happened
//! when the pipeline's legalizing steps were moved after saturation
//! (docs/plans/2026-09-09-a-fold-is-a-node.md §9): goldens, the JIT-vs-
//! interpreter oracle and the winding oracle all passed, and the emitted DAG
//! grew 23–42%. The measurement existed as an example nobody runs. This is
//! the same measurement as a gate.
//!
//! **Ceilings, not pins.** Saturation is deterministic — budgets are
//! deterministic functions of the input, so two machines cannot disagree
//! (CLAUDE.md, "A kernel built differently on two machines?") — and an exact
//! pin would therefore be legitimate. It would also have to be edited by
//! every change that *improves* the number, which is how a pin stops being
//! read and starts being rubber-stamped. A ceiling with ~10% headroom passes
//! silently when the compiler gets better and fails when it gets a fifth
//! worse. If one of these fails low by a wide margin, lower it — that is the
//! ratchet, done deliberately.

use pixelflow_graphics::fonts::Font;
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::LatticeShape;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// `(character, tile px, ceiling)`. The ceilings are the counts measured on
/// 2026-09-09 plus ~10%: `A` 1041, `O` 2938, `8` 6747.
///
/// One line-segment glyph, one all-quadratic, and the one whose waist
/// tangency is the knife edge the class-cap sweep is blocked on
/// (`egraph::saturate::CLASSICAL_CLASS_CEILING`) — so if that unblocks and
/// the cap rises, this notices.
const CEILINGS: [(char, usize, usize); 3] = [('A', 16, 1150), ('O', 16, 3250), ('8', 32, 7450)];

fn reachable(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = vec![false; arena.len()];
    let mut stack = vec![root];
    let mut n = 0;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += 1;
        stack.extend(arena.children(id));
    }
    n
}

#[test]
fn a_glyph_costs_no_more_than_it_did() {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let mut report = String::new();
    let mut over = Vec::new();

    for (ch, px, ceiling) in CEILINGS {
        let glyph = font
            .glyph_kernel_scaled(ch, px as f32)
            .unwrap_or_else(|| panic!("the font has no glyph for {ch:?}"));
        let coverage = glyph.kernel();
        let (arena, root) = coverage.parts();
        let shape = LatticeShape::new([px as u32, px as u32]);

        // The whole tier, as production runs it: link, legalize, saturate.
        // `None` is not a pass — it means the pipeline declined, and the
        // caller would then compile an arena that still holds a fold, which
        // the emitter has no instruction for.
        let optimized = pixelflow_search::runtime::optimize_runtime_arena(arena, root, shape)
            .unwrap_or_else(|| panic!("{ch}@{px}: the runtime pipeline declined"));
        let (out, out_root) = &*optimized;
        let nodes = reachable(out, *out_root);

        report.push_str(&format!("  {ch}@{px}: {nodes} nodes (ceiling {ceiling})\n"));
        if nodes > ceiling {
            over.push(format!("{ch}@{px}: {nodes} > {ceiling}"));
        }
    }

    assert!(
        over.is_empty(),
        "the runtime tier emits more than it used to for {}:\n{report}\n\
         Coverage being unchanged does not make this fine — the pixels are \
         pinned elsewhere, and this is the only test that reads the size of \
         the code that draws them.",
        over.join(", ")
    );
    println!("{report}");
}
