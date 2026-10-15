//! Generated kernels against an f64 oracle, at the points where one can be asked.
//!
//! The contract (docs/process/2026-10-13-tests-check-the-as-if-contract.md, item
//! 1): a compiled kernel's output equals its meaning, within a tolerance, where
//! the meaning is well-conditioned; elsewhere only its range is promised. The
//! hand-written tests state that for one op at a time. This states it for
//! compositions nobody wrote down: a small deterministic generator builds
//! random expression DAGs through the arena builder, each is compiled through
//! the optimizing entry (`jit_cache::compile`) over a lattice wider than a
//! batch, and every sample is compared with an f64 evaluation of the same DAG.
//!
//! The oracle shares no code with the compiler: it walks the generator's own
//! record of what it built, in f64, with Rust's arithmetic.
//!
//! **Well-conditioned.** The optimizer may reassociate, fuse and reorder, and
//! the hardware rounds every op, so equal bits are not promised (item 4), and
//! near a cancellation or a discontinuity even the last bits are amplified
//! without bound. A point is compared only when the oracle's answer is
//! insensitive to relative perturbations of every value in the DAG: with each
//! node scaled by `1 ± 1e-6` (an independent sign per node, eight sign
//! patterns), the result moves by at most 64 times the relative perturbation.
//! Perturbing every node, not only `X` and `Y`, is what sees `a - b` with
//! `a` and `b` both functions of `x`; a comparison must also clear its own
//! margin, because a flipped mask swaps whole arms. The filter is on the
//! oracle's conditioning, never on the output's magnitude.
//!
//! **Tolerance.** Every op but `Div` is a correctly rounded f32 instruction or a
//! fused multiply-add, worth 8 ulps of the result's magnitude plus `1e-6`
//! absolute. The optimizer turns `a / b` into `a * recip(b)`, and the x86
//! `Recip` row of CLAUDE.md "Floating point at the edges" is an estimate
//! (`rcpps`, bounded at `1.5 * 2^-12`), so the allowance also adds the
//! result's response to a `1.5 * 2^-12` reciprocal error at each `Div`,
//! computed from the oracle. That bound is the x86 row, which is why the file
//! is x86-64 only.
//!
//! **Range.** At the points that fail the conditioning test the answer is only
//! required to be finite whenever the oracle is finite and stays inside a range
//! f32 squares comfortably.

#![cfg(target_arch = "x86_64")]

use pixelflow_codegen::jit_cache;
use pixelflow_ir::{ExprArena, ExprId, Kernel, LatticeShape, OpKind};

const SEED: u64 = 0x005e_ed0f_6e4e_7a7e;
const KERNELS: usize = 200;
const MIN_DEPTH: usize = 3;
const MAX_DEPTH: usize = 6;

const WIDTH: usize = 32;
const HEIGHT: usize = 4;
/// Where the lattice's sample `(0, 0)` lies: off the integers, and never on
/// zero, so no sample sits on an axis.
const ORIGIN: [f32; 2] = [-12.5, -1.5];

/// The constants' range.
const CONSTANT_RANGE: f64 = 4.0;
/// The `ε` of `sqrt(|e| + ε)`.
const SQRT_FLOOR: f32 = 0.0625;
/// The denominator of `Div` is `|e| + DIV_FLOOR`, so at least this.
const DIV_FLOOR: f32 = 1.0;

/// The relative perturbation the conditioning test applies to every node.
const PERTURBATION: f64 = 1e-6;
/// A point is well-conditioned when the result moves by no more than this many
/// times the relative perturbation.
const CONDITION_LIMIT: f64 = 64.0;
/// Sign patterns tried per point.
const PATTERNS: usize = 8;
/// The relative error of the reciprocal estimate the optimizer divides by:
/// `rcpps`, the widest tier's estimate, is "~12 bits" (CLAUDE.md, "Floating
/// point at the edges"), which the Intel manual bounds at 1.5 * 2^-12. The
/// optimizer is free to turn `a / b` into `a * recip(b)`, so a division
/// carries this.
const RECIP_ERROR: f64 = 1.5 / 4096.0;
/// Ulps of the result's magnitude the JIT may differ from the oracle by.
const ULPS: f64 = 8.0;
const ABSOLUTE: f64 = 1e-6;
/// Values beyond this are not asserted finite: their squares are the edge of
/// what f32 holds, and an intermediate the optimizer reassociates could cross.
const COMFORTABLE: f64 = 1e18;

/// Deterministic generator: an LCG, no dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }

    /// Uniform in `[-CONSTANT_RANGE, CONSTANT_RANGE]`.
    fn constant(&mut self) -> f32 {
        let unit = f64::from(self.next()) / f64::from(1u32 << 31);
        ((unit * 2.0 - 1.0) * CONSTANT_RANGE) as f32
    }
}

/// What the generator built at one arena node, in the oracle's terms.
enum Node {
    Axis(u8),
    Const(f32),
    Apply(OpKind, Vec<usize>),
}

impl Node {
    fn is_mask(&self) -> bool {
        matches!(
            self,
            Node::Apply(OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge, _)
        )
    }
}

/// One DAG, built in the arena and recorded for the oracle side by side.
/// `nodes[i]` is what `ids[i]` is; children precede parents.
struct Generated {
    rng: Lcg,
    arena: ExprArena,
    ids: Vec<ExprId>,
    nodes: Vec<Node>,
}

impl Generated {
    fn build(rng: Lcg, depth: usize) -> (Self, usize) {
        let mut g = Generated {
            rng,
            arena: ExprArena::new(),
            ids: Vec::new(),
            nodes: Vec::new(),
        };
        let root = g.expression(depth);
        (g, root)
    }

    fn leaf(&mut self) -> usize {
        let (id, node) = match self.rng.below(3) {
            0 => (self.arena.push_var(0), Node::Axis(0)),
            1 => (self.arena.push_var(1), Node::Axis(1)),
            _ => {
                let c = self.rng.constant();
                (self.arena.push_const(c), Node::Const(c))
            }
        };
        self.record(id, node)
    }

    fn constant(&mut self, c: f32) -> usize {
        let id = self.arena.push_const(c);
        self.record(id, Node::Const(c))
    }

    fn apply(&mut self, op: OpKind, args: &[usize]) -> usize {
        let id = match *args {
            [a] => self.arena.push_unary(op, self.ids[a]),
            [a, b] => self.arena.push_binary(op, self.ids[a], self.ids[b]),
            [a, b, c] => self
                .arena
                .push_ternary(op, self.ids[a], self.ids[b], self.ids[c]),
            _ => unreachable!("the generator builds one- to three-operand ops"),
        };
        self.record(id, Node::Apply(op, args.to_vec()))
    }

    fn record(&mut self, id: ExprId, node: Node) -> usize {
        self.ids.push(id);
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn expression(&mut self, depth: usize) -> usize {
        if depth == 0 || self.rng.below(8) == 0 {
            return self.leaf();
        }
        // A shared sub-DAG: the arena is a DAG, not a tree.
        let values: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].is_mask())
            .collect();
        if !values.is_empty() && self.rng.below(5) == 0 {
            return values[self.rng.below(values.len())];
        }
        let d = depth - 1;
        match self.rng.below(11) {
            0 => self.binary(OpKind::Add, d),
            1 => self.binary(OpKind::Sub, d),
            2 => self.binary(OpKind::Mul, d),
            3 => {
                let [a, b, c] = [(); 3].map(|()| self.expression(d));
                self.apply(OpKind::MulAdd, &[a, b, c])
            }
            4 => self.binary(OpKind::Min, d),
            5 => self.binary(OpKind::Max, d),
            6 => self.unary(OpKind::Abs, d),
            7 => self.unary(OpKind::Neg, d),
            8 => {
                // sqrt(|e| + ε)
                let e = self.expression(d);
                let magnitude = self.apply(OpKind::Abs, &[e]);
                let floor = self.constant(SQRT_FLOOR);
                let radicand = self.apply(OpKind::Add, &[magnitude, floor]);
                self.apply(OpKind::Sqrt, &[radicand])
            }
            9 => {
                // n / (|e| + 1)
                let n = self.expression(d);
                let e = self.expression(d);
                let magnitude = self.apply(OpKind::Abs, &[e]);
                let floor = self.constant(DIV_FLOOR);
                let denominator = self.apply(OpKind::Add, &[magnitude, floor]);
                self.apply(OpKind::Div, &[n, denominator])
            }
            _ => {
                let cmp = [OpKind::Lt, OpKind::Le, OpKind::Gt, OpKind::Ge][self.rng.below(4)];
                let [a, b, t, f] = [(); 4].map(|()| self.expression(d));
                let mask = self.apply(cmp, &[a, b]);
                self.apply(OpKind::If, &[mask, t, f])
            }
        }
    }

    fn binary(&mut self, op: OpKind, depth: usize) -> usize {
        let a = self.expression(depth);
        let b = self.expression(depth);
        self.apply(op, &[a, b])
    }

    fn unary(&mut self, op: OpKind, depth: usize) -> usize {
        let a = self.expression(depth);
        self.apply(op, &[a])
    }

    /// The oracle: every node's value at `(x, y)`, in f64, each non-mask node
    /// then scaled by `1 + scale[i]`. A mask is `1.0` or `0.0` here, a
    /// boolean the oracle's `If` reads; it is not the JIT's lane pattern.
    fn evaluate(&self, (x, y): (f64, f64), scale: &[f64]) -> Vec<f64> {
        let mut v: Vec<f64> = Vec::with_capacity(self.nodes.len());
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        for (i, node) in self.nodes.iter().enumerate() {
            let value = match node {
                Node::Axis(0) => x,
                Node::Axis(_) => y,
                Node::Const(c) => f64::from(*c),
                Node::Apply(op, a) => match (op, a.as_slice()) {
                    (OpKind::Add, &[a, b]) => v[a] + v[b],
                    (OpKind::Sub, &[a, b]) => v[a] - v[b],
                    (OpKind::Mul, &[a, b]) => v[a] * v[b],
                    (OpKind::Div, &[a, b]) => v[a] / v[b],
                    (OpKind::Min, &[a, b]) => v[a].min(v[b]),
                    (OpKind::Max, &[a, b]) => v[a].max(v[b]),
                    (OpKind::MulAdd, &[a, b, c]) => v[a].mul_add(v[b], v[c]),
                    (OpKind::Abs, &[a]) => v[a].abs(),
                    (OpKind::Neg, &[a]) => -v[a],
                    (OpKind::Sqrt, &[a]) => v[a].sqrt(),
                    (OpKind::Lt, &[a, b]) => flag(v[a] < v[b]),
                    (OpKind::Le, &[a, b]) => flag(v[a] <= v[b]),
                    (OpKind::Gt, &[a, b]) => flag(v[a] > v[b]),
                    (OpKind::Ge, &[a, b]) => flag(v[a] >= v[b]),
                    (OpKind::If, &[m, t, f]) => {
                        if v[m] != 0.0 {
                            v[t]
                        } else {
                            v[f]
                        }
                    }
                    (op, args) => unreachable!("oracle has no {op:?} of {} operands", args.len()),
                },
            };
            v.push(if node.is_mask() {
                value
            } else {
                value * (1.0 + scale[i])
            });
        }
        v
    }

    /// Whether the oracle's answer at this point is insensitive to the
    /// perturbations, so that f32 rounding and a reassociation cannot move it
    /// by more than the tolerance. `base` is the unperturbed evaluation.
    fn well_conditioned(&self, point: (f64, f64), base: &[f64], patterns: &mut Lcg) -> bool {
        let root = base.len() - 1;
        // A comparison's operands must be farther apart than the largest error
        // a value can carry (a division's), amplified as far as the limit
        // allows: a flipped mask swaps whole arms.
        let margin = CONDITION_LIMIT * RECIP_ERROR;
        let comparisons_clear = self.nodes.iter().all(|node| match node {
            Node::Apply(OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge, a) => {
                let (l, r) = (base[a[0]], base[a[1]]);
                (l - r).abs() >= margin * l.abs().max(r.abs())
            }
            _ => true,
        });
        comparisons_clear
            && (0..PATTERNS).all(|_| {
                let scale: Vec<f64> = (0..base.len())
                    .map(|_| {
                        if patterns.below(2) == 0 {
                            PERTURBATION
                        } else {
                            -PERTURBATION
                        }
                    })
                    .collect();
                let moved = self.evaluate(point, &scale)[root];
                (moved - base[root]).abs() <= CONDITION_LIMIT * PERTURBATION * base[root].abs()
            })
    }

    /// How far the result can move when every division carries the error the
    /// optimizer's reciprocal may add (`RECIP_ERROR`): the sum, over the
    /// divisions, of the result's response to that error at that division alone.
    fn division_allowance(&self, point: (f64, f64), base: &[f64]) -> f64 {
        let root = base.len() - 1;
        let mut scale = vec![0.0; base.len()];
        let mut total = 0.0;
        for i in 0..base.len() {
            if matches!(self.nodes[i], Node::Apply(OpKind::Div, _)) {
                scale[i] = RECIP_ERROR;
                total += (self.evaluate(point, &scale)[root] - base[root]).abs();
                scale[i] = 0.0;
            }
        }
        total
    }
}

/// The JIT's answer for every sample, row-major.
fn run(g: &Generated, root: usize) -> Vec<f32> {
    let kernel = Kernel::from_parts(g.arena.clone(), g.ids[root]);
    let shape = LatticeShape::new([WIDTH as u32, HEIGHT as u32]);
    let linked = jit_cache::compile(&kernel, shape).expect("a generated kernel compiles");
    assert!(
        linked.buffers.is_empty() && linked.uniforms.is_empty(),
        "a generated kernel reads no buffer and no uniform"
    );
    let mut out = vec![f32::NAN; WIDTH * HEIGHT];
    // No buffers, so the context is the uniform block's slot (unread) and the
    // origin block.
    let ctx: [*const f32; 2] = [core::ptr::null(), ORIGIN.as_ptr()];
    // SAFETY: the kernel declares no buffer and no uniform, so `ctx[0]` is
    // never read; `ctx[1]` is the origin block of two floats; `out` holds the
    // `WIDTH * HEIGHT` samples of the extent it was compiled at, with pitch
    // `WIDTH`.
    unsafe { linked.kernel.call(ctx.as_ptr(), out.as_mut_ptr(), WIDTH) };
    out
}

#[test]
fn generated_kernels_agree_with_an_f64_oracle_where_well_conditioned() {
    let mut rng = Lcg(SEED);
    let (mut compared, mut total) = (0usize, 0usize);

    for k in 0..KERNELS {
        let depth = MIN_DEPTH + k % (MAX_DEPTH - MIN_DEPTH + 1);
        let (g, root) = Generated::build(
            Lcg(u64::from(rng.next()) << 32 | u64::from(rng.next())),
            depth,
        );
        let mut patterns = Lcg(u64::from(rng.next()));
        let out = run(&g, root);
        let no_scale = vec![0.0; g.nodes.len()];

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                total += 1;
                let point = (
                    f64::from(ORIGIN[0]) + col as f64,
                    f64::from(ORIGIN[1]) + row as f64,
                );
                let got = out[row * WIDTH + col];
                let base = g.evaluate(point, &no_scale);
                let want = base[root];
                let describe = || {
                    format!(
                        "kernel {k} (depth {depth}, {} nodes) at {point:?}: got {got:e}, oracle {want:e}",
                        g.nodes.len()
                    )
                };
                let in_range = base.iter().all(|v| v.is_finite() && v.abs() < COMFORTABLE);
                if !in_range {
                    continue;
                }
                assert!(got.is_finite(), "range: {}", describe());
                if !g.well_conditioned(point, &base, &mut patterns) {
                    continue;
                }
                compared += 1;
                let tolerance = ULPS * f64::from(f32::EPSILON) * want.abs()
                    + ABSOLUTE
                    + g.division_allowance(point, &base);
                assert!(
                    (f64::from(got) - want).abs() <= tolerance,
                    "value: {} (tolerance {tolerance:e})",
                    describe()
                );
            }
        }
    }
    // The filter must leave most of the points to be compared, or the test
    // has quietly become a range check.
    assert!(
        compared * 2 >= total,
        "only {compared} of {total} points were well-conditioned"
    );
}
