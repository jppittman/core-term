//! What a laid-out string costs to *build* and to *legalize*, as the string
//! grows: arena size, reachable nodes, piece count, and the size of the
//! legalized (reference-linked, derivative-lowered, reduce-unrolled) arena
//! that reaches the JIT. One line per length. Both columns must stay linear
//! in the piece count — the measurement behind
//! docs/plans/2026-09-09-composition-is-linking.md §7.
//!
//! Run: `cargo run --release -p pixelflow-graphics --example text_kernel_cost`

use std::time::Instant;

use pixelflow_core::Kernel;
use pixelflow_graphics::fonts::{text, Font};
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::passes::legalize;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono-Fallback.ttf");

/// The winding sum's piece table has this many columns per piece
/// (`loop_blinn::PIECE_ROW_COLS`, private to that module).
const PIECE_ROW_COLS: usize = 13;

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

fn main() {
    let font = Font::parse(FONT_BYTES).expect("parse font");
    let size = 32.0f32;
    let alphabet = "abcdefghijklmnopqrstuvwxyz";
    println!("chars  construct_us  arena_len  reachable  pieces  legalized_reachable  legalize_us");
    for n in [1usize, 2, 4, 8, 13, 26] {
        let s = &alphabet[..n];
        let t0 = Instant::now();
        let glyph = text(&font, s, size);
        let kernel = glyph.kernel.at(
            &Kernel::x().add(&Kernel::constant(0.5)),
            &Kernel::y().add(&Kernel::constant(0.5)),
        );
        let construct = t0.elapsed();
        let (arena, root) = kernel.parts();
        let pieces: usize = kernel
            .buffer_data()
            .map(|(_, data)| data.len())
            .sum::<usize>()
            / PIECE_ROW_COLS;
        let t1 = Instant::now();
        let (legal, legal_root) = legalize(arena, root).expect("legalize");
        let legalize_t = t1.elapsed();
        println!(
            "{n:>5}  {:>12}  {:>9}  {:>9}  {pieces:>6}  {:>19}  {:>11}",
            construct.as_micros(),
            arena.len(),
            reachable(arena, root),
            reachable(&legal, legal_root),
            legalize_t.as_micros(),
        );
    }
}
