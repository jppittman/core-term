//! `dag::Builder`/`Rooted` against `ExprArena` on the operations they share:
//! building a term graph, and walking it back.
//!
//! Three questions, three groups:
//! - `build/no_sharing`: a wide expression with no duplicate subexpressions
//!   — the worst case for consing (pure hashmap overhead, no dedup to show
//!   for it). Compares `ExprArena::push_*` against `Builder::intern` (paying
//!   for the memo) and `Builder::push_unique` (apples-to-apples append, no
//!   hashing — isolates consing's cost from `Dag`'s base append cost).
//! - `build/naive_sharing`: the *same* subexpression re-pushed N times as
//!   children of a wide sum, the way a caller who doesn't bother tracking
//!   `ExprId`/`Id` reuse would write it. `ExprArena` and `push_unique` both
//!   grow linearly in N; `intern` collapses to the subexpression's own size
//!   regardless of N. Reports final node counts, not just time — the point
//!   here is what consing buys structurally, not only speed.
//! - `traverse`: build one moderate graph each way, then repeatedly count
//!   reachable nodes via `ExprArena::node_count_subtree` (no dedup —
//!   multiplicity per reference, matching a tree-walking evaluator) against
//!   `Node::descendants()` (deduped, fresh allocation per call) and
//!   `Node::descendants_in(&mut Scratch)` (deduped, scratch reused across
//!   calls) — isolating the per-call allocation `Scratch` exists to avoid.

use criterion::{Criterion, criterion_group, criterion_main};
use pixelflow_ir::arena::{ExprArena, ExprId};
use pixelflow_ir::kind::OpKind;
use pixelflow_ir::{Builder, Rooted};
use std::hint::black_box;

// `Ord` too, not just `Hash`: `Builder<T: Key>`'s bound depends on whether
// `hash-memo` is on (wants `Hash`) or off (the `BTreeMap` fallback wants
// `Ord`; see dag.rs), and this needs to build either way — matching
// `dag.rs`'s own `Op` test fixture.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Payload {
    Var(u8),
    Const(u32),
    Op(OpKind),
}

// ───────────────────────── no_sharing: N independent terms ───────────────

/// `((x*s0 + y*s0) + (x*s1 + y*s1) + ... )` — N terms, no two subtrees equal
/// (each carries a distinct constant), so nothing here is a candidate for
/// consing to collapse. 6 nodes per term plus one Add to fold it in.
fn arena_no_sharing(n: u32) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let mut acc = a.push_const(0.0);
    for i in 0..n {
        let s = a.push_const(i as f32 + 1.0);
        let xs = a.push_binary(OpKind::Mul, x, s);
        let ys = a.push_binary(OpKind::Mul, y, s);
        let term = a.push_binary(OpKind::Add, xs, ys);
        acc = a.push_binary(OpKind::Add, acc, term);
    }
    (a, acc)
}

fn dag_no_sharing(n: u32, intern: bool) -> Rooted<Payload> {
    let mut b: Builder<Payload> = Builder::with_capacity(6 * n as usize + 4, 0);
    let push = |b: &mut Builder<Payload>, v: Payload, kids: &[_]| {
        if intern {
            b.intern(v, kids)
        } else {
            b.push_unique(v, kids)
        }
    };
    let x = push(&mut b, Payload::Var(0), &[]);
    let y = push(&mut b, Payload::Var(1), &[]);
    let mut acc = push(&mut b, Payload::Const(0.0f32.to_bits()), &[]);
    for i in 0..n {
        let s = push(&mut b, Payload::Const((i as f32 + 1.0).to_bits()), &[]);
        let xs = push(&mut b, Payload::Op(OpKind::Mul), &[x, s]);
        let ys = push(&mut b, Payload::Op(OpKind::Mul), &[y, s]);
        let term = push(&mut b, Payload::Op(OpKind::Add), &[xs, ys]);
        acc = push(&mut b, Payload::Op(OpKind::Add), &[acc, term]);
    }
    b.finish(&[acc])
}

fn bench_build_no_sharing(c: &mut Criterion) {
    const N: u32 = 30_000;
    let (a, _) = arena_no_sharing(N);
    let interned = dag_no_sharing(N, true);
    let uninterned = dag_no_sharing(N, false);
    eprintln!(
        "[build/no_sharing] N={N} nodes: arena={}, dag(intern)={}, dag(push_unique)={} \
         (no duplicates exist, so intern buys nothing here — it's the overhead case)",
        a.len(),
        interned.len(),
        uninterned.len(),
    );

    let mut g = c.benchmark_group("build/no_sharing");
    g.bench_function("arena_push", |b| {
        b.iter(|| black_box(arena_no_sharing(black_box(N))))
    });
    g.bench_function("dag_intern", |b| {
        b.iter(|| black_box(dag_no_sharing(black_box(N), true)))
    });
    g.bench_function("dag_push_unique", |b| {
        b.iter(|| black_box(dag_no_sharing(black_box(N), false)))
    });
    g.finish();
}

// ───────────────────────── naive_sharing: one subtree, pushed N times ────

/// `d0 + d1 + ... + d(N-1)` where every `di` is a *freshly re-pushed* copy
/// of the same 4-node subtree `(x*c + y*c)` — never the same `ExprId`/`Id`
/// reused, simulating a caller who builds without tracking sharing.
fn arena_naive_sharing(n: u32) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let mut acc = a.push_const(0.0);
    for _ in 0..n {
        let x = a.push_var(0);
        let y = a.push_var(1);
        let c = a.push_const(1.5);
        let xs = a.push_binary(OpKind::Mul, x, c);
        let ys = a.push_binary(OpKind::Mul, y, c);
        let d = a.push_binary(OpKind::Add, xs, ys);
        acc = a.push_binary(OpKind::Add, acc, d);
    }
    (a, acc)
}

fn dag_naive_sharing(n: u32, intern: bool) -> Rooted<Payload> {
    let mut b: Builder<Payload> = Builder::new();
    let push = |b: &mut Builder<Payload>, v: Payload, kids: &[_]| {
        if intern {
            b.intern(v, kids)
        } else {
            b.push_unique(v, kids)
        }
    };
    let mut acc = push(&mut b, Payload::Const(0.0f32.to_bits()), &[]);
    for _ in 0..n {
        let x = push(&mut b, Payload::Var(0), &[]);
        let y = push(&mut b, Payload::Var(1), &[]);
        let c = push(&mut b, Payload::Const(1.5f32.to_bits()), &[]);
        let xs = push(&mut b, Payload::Op(OpKind::Mul), &[x, c]);
        let ys = push(&mut b, Payload::Op(OpKind::Mul), &[y, c]);
        let d = push(&mut b, Payload::Op(OpKind::Add), &[xs, ys]);
        acc = push(&mut b, Payload::Op(OpKind::Add), &[acc, d]);
    }
    b.finish(&[acc])
}

fn bench_build_naive_sharing(c: &mut Criterion) {
    const N: u32 = 30_000;
    let (a, _) = arena_naive_sharing(N);
    let interned = dag_naive_sharing(N, true);
    let uninterned = dag_naive_sharing(N, false);
    eprintln!(
        "[build/naive_sharing] N={N} re-pushes of one 4-node subtree: \
         arena={} nodes, dag(intern)={} nodes, dag(push_unique)={} nodes \
         (intern collapses the shared subtree; the other two grow with N)",
        a.len(),
        interned.len(),
        uninterned.len(),
    );

    let mut g = c.benchmark_group("build/naive_sharing");
    g.bench_function("arena_push", |b| {
        b.iter(|| black_box(arena_naive_sharing(black_box(N))))
    });
    g.bench_function("dag_intern", |b| {
        b.iter(|| black_box(dag_naive_sharing(black_box(N), true)))
    });
    g.bench_function("dag_push_unique", |b| {
        b.iter(|| black_box(dag_naive_sharing(black_box(N), false)))
    });
    g.finish();
}

// ───────────────────────── traverse: count reachable nodes, repeatedly ───

fn bench_traverse(c: &mut Criterion) {
    const N: u32 = 30_000;
    let (arena, root) = arena_naive_sharing(N);
    let dag = dag_naive_sharing(N, true);
    eprintln!(
        "[traverse] arena has {} nodes (multiplicity-counted subtree size); \
         dag has {} nodes (deduped) after consing the same input",
        arena.node_count_subtree(root),
        dag.len(),
    );

    let mut g = c.benchmark_group("traverse");
    g.bench_function("arena_node_count_subtree", |b| {
        b.iter(|| black_box(arena.node_count_subtree(black_box(root))))
    });
    g.bench_function("dag_descendants_alloc", |b| {
        b.iter(|| black_box(dag.entry().descendants().count()))
    });
    g.bench_function("dag_descendants_in_scratch_reused", |b| {
        let mut scratch = dag.scratch();
        b.iter(|| black_box(dag.entry().descendants_in(black_box(&mut scratch)).count()))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_build_no_sharing,
    bench_build_naive_sharing,
    bench_traverse
);
criterion_main!(benches);
