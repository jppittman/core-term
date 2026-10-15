//! `Eq` and `Ne` on the JIT: exact, NaN-correct, and a mask.
//!
//! The contract (CLAUDE.md, "Floating point at the edges"): `Eq` and `Ne`
//! are an *exact* comparison, so NaN is never equal to anything (itself
//! included) and always unequal, and `-0.0 == +0.0`. Every comparison's
//! result is a mask — an all-ones lane for true, all-zero for false — never
//! `1.0`. The mask is observed here only through its one public consumer,
//! `select`, which must return exactly one of its two arms, bit for bit, for
//! every lane. A blend of the arms (what a `1.0` mask would produce through
//! `andps`) returns a value neither arm held, so bit-exact selection is the
//! mask-ness check.
//!
//! The oracle is Rust's own `f32 ==` and `!=`, which share no code with the
//! emitter. Both operands arrive as gathers over buffers, so every lane's
//! pair is a runtime value the emitter cannot fold away, and the lattice is
//! wider than a batch with a remainder, so the masks vary by lane within a
//! batch as well as from one batch to the next.
//!
//! Every case runs twice: straight from the arena to the emitter, and through
//! the optimizing entry (`jit_cache::compile`), which the e-graph reaches
//! first. Eq/Ne are exact, so the optimizer may not change one bit of the
//! selected arm.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::compile;
use pixelflow_codegen::jit_cache;
use pixelflow_ir::arena::{BufferDecl, BufferIdentity, ExprArena, ExprId};
use pixelflow_ir::{Kernel, LatticeShape, OpKind};

/// Lanes per row: deliberately not a multiple of any tier's batch, so every
/// row ends in a remainder.
const WIDTH: usize = 37;

const NAN: f32 = f32::NAN;
const INF: f32 = f32::INFINITY;

#[derive(Clone, Copy, Debug)]
enum Cmp {
    Eq,
    Ne,
}

impl Cmp {
    fn op(self) -> OpKind {
        match self {
            Cmp::Eq => OpKind::Eq,
            Cmp::Ne => OpKind::Ne,
        }
    }

    /// The oracle: Rust's own comparison.
    fn holds(self, a: f32, b: f32) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
        }
    }
}

/// Which arm sits where: `select(a CMP b, a, b)` or `select(a CMP b, b, a)`.
/// The swap makes an arm that was chosen wrongly visible even when the two
/// operands would otherwise be interchangeable (`-0.0` against `+0.0`).
#[derive(Clone, Copy, Debug)]
enum Arms {
    Direct,
    Swapped,
}

/// Which compiled path a case goes through.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// Arena straight to the emitter, no optimizer.
    Emitted,
    /// The optimizing entry: the e-graph runs before emission.
    Optimized,
}

/// The arm the program must return for one lane, in the oracle's terms.
fn expected(cmp: Cmp, arms: Arms, a: f32, b: f32) -> f32 {
    let (if_true, if_false) = match arms {
        Arms::Direct => (a, b),
        Arms::Swapped => (b, a),
    };
    if cmp.holds(a, b) { if_true } else { if_false }
}

/// `select(gather(A) CMP gather(B), ...)` over a `WIDTH × rows` lattice,
/// where `A` and `B` are the two buffers the caller fills.
fn build(cmp: Cmp, arms: Arms, rows: usize, ids: [BufferIdentity; 2]) -> (ExprArena, ExprId) {
    let mut arena = ExprArena::new();
    let decl = |id| BufferDecl {
        id,
        width: WIDTH as u32,
        height: rows as u32,
    };
    let buf_a = arena.declare_buffer(decl(ids[0]));
    let buf_b = arena.declare_buffer(decl(ids[1]));
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let a = arena.push_gather(buf_a, x, y);
    let b = arena.push_gather(buf_b, x, y);
    let mask = arena.push_binary(cmp.op(), a, b);
    let (if_true, if_false) = match arms {
        Arms::Direct => (a, b),
        Arms::Swapped => (b, a),
    };
    let root = arena.push_ternary(OpKind::If, mask, if_true, if_false);
    (arena, root)
}

/// Run one `(cmp, arms)` kernel over every pair and return the row-major
/// output, `rows * WIDTH` long. Pad lanes hold `(0.0, 0.0)` and are ignored.
fn run(path: Path, cmp: Cmp, arms: Arms, pairs: &[(f32, f32)]) -> Vec<f32> {
    let rows = pairs.len().div_ceil(WIDTH);
    let mut padded = pairs.to_vec();
    padded.resize(rows * WIDTH, (0.0, 0.0));
    let a_data: Vec<f32> = padded.iter().map(|p| p.0).collect();
    let b_data: Vec<f32> = padded.iter().map(|p| p.1).collect();
    let ids = [BufferIdentity::mint(), BufferIdentity::mint()];
    let (arena, root) = build(cmp, arms, rows, ids);
    let shape = LatticeShape::new([WIDTH as u32, rows as u32]);
    let mut out = vec![f32::NAN; rows * WIDTH];
    let origin = [0.0f32, 0.0];
    let no_uniforms: [f32; 0] = [];

    match path {
        Path::Emitted => {
            let code = compile(&arena, root, shape).expect("the kernel compiles");
            // Buffers in declaration order, then the uniform block, then the origin.
            let ctx: [*const f32; 4] = [
                a_data.as_ptr(),
                b_data.as_ptr(),
                no_uniforms.as_ptr(),
                origin.as_ptr(),
            ];
            // SAFETY: each buffer holds `rows * WIDTH` samples, the uniform
            // block is never read (the arena declares none), the origin is
            // two floats, and `out` holds `rows * WIDTH` with pitch `WIDTH`.
            unsafe { code.code.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
        }
        Path::Optimized => {
            let linked = jit_cache::compile(&Kernel::from_parts(arena, root), shape)
                .expect("the optimized kernel compiles");
            // The link says which buffer each slot binds, in slot order.
            let slot_data: Vec<*const f32> = linked
                .buffers
                .iter()
                .map(|decl| {
                    if decl.id == ids[0] {
                        a_data.as_ptr()
                    } else {
                        b_data.as_ptr()
                    }
                })
                .collect();
            let mut ctx = slot_data;
            ctx.push(no_uniforms.as_ptr());
            ctx.push(origin.as_ptr());
            // SAFETY: as above; the context carries one pointer per buffer the
            // link names, then the uniform block, then the origin.
            unsafe { linked.kernel.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
        }
    }
    out
}

/// Every lane of `out` must be bit-identical to the oracle's chosen arm.
fn check(path: Path, cmp: Cmp, arms: Arms, pairs: &[(f32, f32)], labels: &[&str]) {
    let out = run(path, cmp, arms, pairs);
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let got = out[(i / WIDTH) * WIDTH + i % WIDTH];
        let want = expected(cmp, arms, a, b);
        assert!(
            got.to_bits() == want.to_bits(),
            "{path:?} select({cmp:?}, {arms:?}) on {}: a={a:e} ({:#010x}), b={b:e} ({:#010x}): \
             got {got:e} ({:#010x}), want {want:e} ({:#010x})",
            labels.get(i).copied().unwrap_or("generated"),
            a.to_bits(),
            b.to_bits(),
            got.to_bits(),
            want.to_bits(),
        );
    }
}

fn all_kernels(path: Path, pairs: &[(f32, f32)], labels: &[&str]) {
    for cmp in [Cmp::Eq, Cmp::Ne] {
        for arms in [Arms::Direct, Arms::Swapped] {
            check(path, cmp, arms, pairs, labels);
        }
    }
}

// ── Named cases, with the documented answer written out ──────────────────────

/// `(label, a, b, a == b)`. The last column is the contract, not the oracle:
/// it is what CLAUDE.md states, and the test also checks it against Rust.
const NAMED: &[(&str, f32, f32, bool)] = &[
    ("equal finite", 1.5, 1.5, true),
    ("unequal finite", 1.5, -1.5, false),
    ("+0 vs -0", 0.0, -0.0, true),
    ("-0 vs +0", -0.0, 0.0, true),
    ("NaN vs NaN, same bits", NAN, NAN, false),
    (
        "NaN vs NaN, other payload",
        f32::from_bits(0x7fc0_0001),
        NAN,
        false,
    ),
    (
        "NaN vs negative NaN",
        NAN,
        f32::from_bits(0xffc0_0000),
        false,
    ),
    ("NaN vs number", NAN, 1.0, false),
    ("number vs NaN", 1.0, NAN, false),
    ("1 vs next_up(1)", 1.0, f32::from_bits(0x3f80_0001), false),
    ("next_up(1) vs 1", f32::from_bits(0x3f80_0001), 1.0, false),
    (
        "-1 vs next_down(-1)",
        -1.0,
        f32::from_bits(0xbf80_0001),
        false,
    ),
    ("inf vs inf", INF, INF, true),
    ("inf vs -inf", INF, -INF, false),
    ("max finite vs inf", f32::MAX, INF, false),
    ("smallest subnormal vs zero", f32::from_bits(1), 0.0, false),
    (
        "adjacent subnormals",
        f32::from_bits(1),
        f32::from_bits(2),
        false,
    ),
    (
        "equal subnormal",
        f32::from_bits(1),
        f32::from_bits(1),
        true,
    ),
    (
        "2^24 vs next_up(2^24)",
        16_777_216.0,
        f32::from_bits(0x4b80_0001),
        false,
    ),
];

#[test]
fn named_cases_follow_ieee_equality_on_the_jit() {
    // The documented answers agree with Rust's own comparison, so the table
    // cannot quietly disagree with the oracle it is checked against.
    for &(label, a, b, eq) in NAMED {
        assert_eq!(
            a == b,
            eq,
            "the table's contract for {label:?} is not Rust's =="
        );
    }
    let labels: Vec<&str> = NAMED.iter().map(|c| c.0).collect();
    let pairs: Vec<(f32, f32)> = NAMED.iter().map(|c| (c.1, c.2)).collect();
    all_kernels(Path::Emitted, &pairs, &labels);
    all_kernels(Path::Optimized, &pairs, &labels);
}

// ── Generated pairs: the special values crossed with each other, adjacent
// floats, and deterministic random bit patterns ─────────────────────────────

fn next_up(x: f32) -> f32 {
    if x.is_nan() || x == INF {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        3.0,
        0.1,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        INF,
        -INF,
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        NAN,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0000),
        16_777_216.0,
    ]
}

/// xorshift64: a fixed sequence, so every run checks the same pairs.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn float(&mut self) -> f32 {
        f32::from_bits(self.next_u32())
    }
}

fn generated_pairs() -> Vec<(f32, f32)> {
    let specials = specials();
    let mut pairs = Vec::new();
    // Every special against every special: -0/+0, NaN against numbers, NaN
    // against NaN in both payloads, infinities, subnormals, the 2^24 step.
    for &a in &specials {
        for &b in &specials {
            pairs.push((a, b));
        }
    }
    // Each special against the float next to it, both ways.
    for &s in &specials {
        pairs.push((s, next_up(s)));
        pairs.push((next_up(s), s));
    }
    // Random bit patterns, each paired with a copy, a neighbour, a sign flip,
    // or an unrelated pattern — equal, adjacent, opposite, or unrelated.
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..4000 {
        let a = rng.float();
        let b = match rng.next_u32() % 4 {
            0 => a,
            1 => next_up(a),
            2 => f32::from_bits(a.to_bits() ^ 0x8000_0000),
            _ => rng.float(),
        };
        pairs.push((a, b));
    }
    pairs
}

#[test]
fn generated_pairs_match_rust_equality_bit_for_bit_on_the_jit() {
    let pairs = generated_pairs();
    all_kernels(Path::Emitted, &pairs, &[]);
    all_kernels(Path::Optimized, &pairs, &[]);
}
