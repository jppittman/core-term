
// ── gate probes ─────────────────────────────────────────────────────────────
// Appended to this file by .wip/gates/lib.sh (`build_probe`) for the length of
// one build and restored with `git checkout --`; never committed. Everything
// it names is already in scope here: `program`, `optimize_runtime_arena`,
// `LatticeShape`, `program_digest`, and (through `use super::*`) `Manifold`.

/// The piece counts the gates read: `n = 0` is the blank program (space), the
/// rest are the counts printable ASCII reaches at 16 px. 29 of them.
const ZZ_GATE_COUNTS: [usize; 29] = [
    0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 32,
    34, 36, 38, 40,
];

/// The tile side the gates draw at, in px.
const ZZ_GATE_SIDE: u32 = 16;

/// Gates 1b and 2. One line per program, in `ZZ_GATE_COUNTS` order:
///
/// `ZZGATE n=<pieces> nodes=<arena len> key=<KernelKey::bits> digest=<program_digest>`
///
/// of what `optimize_runtime_arena` returns for `program(n)`, built exactly as
/// `the_optimizer_selects_what_it_selected` builds it (`linked_parts`, then the
/// optimizer at the shape). Its n = 2 and n = 40 digests at 16 px are that
/// test's pinned `SELECTED_PROGRAM_DIGESTS`: the harness's self-check.
#[test]
#[ignore = "gate probe: run by .wip/gates/gate1b.sh and gate2.sh"]
fn zz_gate_font_kernels() {
    for count in ZZ_GATE_COUNTS {
        let (arena, root) = program(count).linked_parts();
        let optimized = optimize_runtime_arena(
            &arena,
            root,
            LatticeShape::new([ZZ_GATE_SIDE, ZZ_GATE_SIDE]),
        )
        .unwrap_or_else(|| panic!("{count} pieces at {ZZ_GATE_SIDE} px: the program declined"));
        let (out, out_root) = &*optimized;
        println!(
            "ZZGATE n={count} nodes={} key={:016x} digest={:016x}",
            out.len(),
            pixelflow_ir::key::KernelKey::of(out, *out_root).bits(),
            program_digest(out, *out_root),
        );
    }
}

/// Gate 5's workload: the cold compile of all 29 programs, one process, as a
/// font does it the first time each piece count is drawn
/// (`FontPrograms::draw`'s `Manifold::compile(&program(n), extent)`: build the
/// program, optimize it through the runtime tier, emit it). Nothing is drawn.
#[test]
#[ignore = "gate probe: run by .wip/gates/measure.sh"]
fn zz_prof_all_cold() {
    for count in ZZ_GATE_COUNTS {
        let compiled = Manifold::compile(&program(count), [ZZ_GATE_SIDE, ZZ_GATE_SIDE]);
        std::hint::black_box(&compiled);
    }
}
