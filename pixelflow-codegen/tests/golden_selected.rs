//! **The selection pipeline's code, byte for byte.**
//!
//! `GOLDEN_SELECTED` is `emit::tests::sibling_folds`'s `GOLDEN` for the new
//! pipeline: the length and FNV-1a 64 digest of what each of `rows::TABLE`'s
//! kernels compiles to. It is born with AVX2, and gains AVX-512 and NEON as they
//! get a backend. It obeys the same rule from birth:
//!
//! **A refactor does not edit this table; an intentional byte change does, in
//! a commit of its own that says why.** A commit that edits it beside other
//! work cannot be told apart from one that moved bytes by accident, which is
//! the thing it exists to catch. When it fails, the failure prints the whole
//! recomputed table.
//!
//! Unlike `GOLDEN`, which emits all three targets from any host through the
//! crate's own entry points, this reaches the pipeline through `compile`, so it
//! is the host's tier and the process's knob: it does nothing unless run on the
//! selection pipeline, at a tier with a table below. The `isa-matrix` job does
//! for the x86 tiers, and the macOS job (and `qemu-aarch64`, with the knob) for
//! NEON. `GOLDEN`'s columns pin the same code for the tiers whose default is
//! selection.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::{fnv1a64, jit_vector_bytes};
use pixelflow_ir::LatticeShape;

mod rows {
    include!("support/sibling_rows.rs");
}
include!("support/knob.rs");

const BYTES_PER_LANE: usize = 4;

/// The vector widths of the tiers the tables are for.
const AVX2_VECTOR_BYTES: usize = 32;
const AVX512_VECTOR_BYTES: usize = 64;
const NEON_VECTOR_BYTES: usize = 16;

/// A row's emitted code: its length in bytes and the FNV-1a 64 digest of those
/// bytes.
type Bytes = (usize, u64);

/// `rows::TABLE`'s bytes through the selection pipeline, on AVX2. A row's
/// provenance is `git log -L` on it.
const GOLDEN_SELECTED_AVX2: [Bytes; 11] = [
    (580, 0xcd22a56b8b260b1a),
    (568, 0x86d60c1545a26d56),
    (880, 0x3ebb4798982a8c6d),
    (896, 0x609dcba2dd47b291),
    (204992, 0x8cca9719cdafb323),
    (2292, 0x6f731e42104d8439),
    (992, 0x99dfdad6d446b32d),
    (852, 0x2c228ca476b28b53),
    (556, 0x8403cb1c517f09a5),
    (452, 0x737225ee0861c2c9),
    (356460, 0x1acf2d0ad97001c3),
];

/// The same, on AVX-512.
const GOLDEN_SELECTED_AVX512: [Bytes; 11] = [
    (512, 0xabb746a4fdea20ab),
    (516, 0xf62e0615af4611a3),
    (780, 0xb1440de7d447321b),
    (864, 0xdeca0265eef3cc0e),
    (232304, 0x7b3738d606745b02),
    (2224, 0x284d1cf632c0dd2b),
    (984, 0x3f1775672612760b),
    (852, 0x485e5615c1be05cb),
    (556, 0xef8b747d224acea3),
    (484, 0xe5056286f1efe99d),
    (401852, 0xb7be255b2b5b4d12),
];

/// The same, on NEON, which the macOS job (and `qemu-aarch64`, under the knob)
/// runs. `deep_frame` is 61,104 bytes of frame here, a few KiB under what
/// `ldr q` reaches; `tests/deep_frame.rs` has the kernel past it.
const GOLDEN_SELECTED_NEON: [Bytes; 11] = [
    (320, 0x1e2cc2ccd2c39688),
    (320, 0xaf15a3b51b626a3b),
    (512, 0xf2c73927a15e6f3a),
    (560, 0x441eee7bc834ff63),
    (142560, 0x9eb8bd206736568b),
    (1872, 0x85e1e22c75b380c7),
    (624, 0xace3ddd873b0de8f),
    (672, 0x3404e179b1577f67),
    (384, 0x836d82c90fed0e27),
    (432, 0x7e2dc82a4db53468),
    (188320, 0xcfe28253787ecf9d),
];

#[test]
fn the_sibling_fold_rows_emit_the_recorded_bytes_through_selection() {
    let pins = match jit_vector_bytes() {
        AVX2_VECTOR_BYTES => GOLDEN_SELECTED_AVX2,
        AVX512_VECTOR_BYTES => GOLDEN_SELECTED_AVX512,
        NEON_VECTOR_BYTES => GOLDEN_SELECTED_NEON,
        _ => return,
    };
    if !selection() {
        return;
    }
    let lanes = (jit_vector_bytes() / BYTES_PER_LANE) as u32;
    let mut recomputed = Vec::new();
    let mut moved = Vec::new();
    for (row, pin) in rows::TABLE.iter().zip(pins) {
        let (arena, root) = (row.build)();
        let shape = LatticeShape::new([row.width.columns(lanes), rows::ROWS]);
        let result = compile(&arena, root, shape).expect("a sibling-fold row compiles");
        let code = result.code.as_bytes();
        let actual = (code.len(), fnv1a64(code));
        if actual != pin {
            moved.push(format!(
                "{}: pinned {pin:x?}, emitted {actual:x?}",
                row.name
            ));
        }
        recomputed.push(format!("    ({}, {:#018x}),", actual.0, actual.1));
    }
    assert!(
        moved.is_empty(),
        "emitted bytes moved from GOLDEN_SELECTED:\n{}\n\n\
         if the change is intentional, re-baseline it in its own commit with:\n{}",
        moved.join("\n"),
        recomputed.join("\n")
    );
}
