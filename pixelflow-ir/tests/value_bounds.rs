//! A fold's bound may be a term, and every public walker takes it.
//!
//! A bound that is a value is a child of its `Reduce`, after the body
//! (docs/language/2026-10-20-a-bound-is-a-value.md §3.1), so each reader of
//! the arena that follows children must follow it, and each copier must carry
//! it. This file builds one fold with a value bound that is deeper than its
//! body, so a walker that read the body alone would answer differently, and
//! asks the public API about it. The private pieces (`over_values`, the rooted
//! variance walk) are pinned in the crate's own unit tests.

use pixelflow_ir::arena::{Axis, ExprArena, ExprId, ExprNode};
use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::variance::{Variance, compute_arena_variance};
use pixelflow_ir::{Kernel, KernelKey, Lit, OpKind, Uniform};

/// The `slot`-th binder.
fn binder(slot: u8) -> Binder {
    Binder::from_slot(slot).expect("a binder slot")
}

/// The sum binding the `slot`-th binder.
fn sum(slot: u8) -> Fold {
    Fold::new(Monoid::SUM, binder(slot))
}

/// The fixture: an outer `Σ_{j<3}` (slot 1) around an inner `Σ_{i<t}` (slot 0)
/// whose upper bound is the term `t = (j + u) + 2`, deeper than the body
/// `i·Y`. The inner bound reads the enclosing index and a uniform, which L3
/// admits.
struct Fixture {
    arena: ExprArena,
    outer: ExprId,
    inner: ExprId,
    body: ExprId,
    bound: ExprId,
}

fn fixture() -> Fixture {
    let mut arena = ExprArena::new();
    let decl = arena.declare_uniform(Uniform::new(2.0).decl());
    let u = arena.push_uniform(decl);
    let y = arena.push_var(Axis::Y.var());
    let i = arena.push_var(binder(0).var());
    let j = arena.push_var(binder(1).var());
    let body = arena.push_binary(OpKind::Mul, i, y);
    let reads = arena.push_binary(OpKind::Add, j, u);
    let two = arena.push_const(2.0);
    let bound = arena.push_binary(OpKind::Add, reads, two);
    let inner = arena.push_reduce(sum(0), Bound::Lit(0), Bound::Value(bound), body);
    let outer = arena.push_reduce(sum(1), Bound::Lit(0), Bound::Lit(3), inner);
    Fixture {
        arena,
        outer,
        inner,
        body,
        bound,
    }
}

/// Whether `copy`'s `root` is the fixture's term and its inner fold's upper
/// bound is still a value.
fn is_the_fixture(copy: &ExprArena, root: ExprId, fixture: &Fixture) -> bool {
    let ExprNode::Reduce { body, .. } = copy.node(root) else {
        return false;
    };
    let ExprNode::Reduce { hi, .. } = copy.node(body) else {
        return false;
    };
    matches!(hi, Bound::Value(_)) && copy.subtree_eq(root, &fixture.arena, fixture.outer)
}

/// Every reader that follows children follows a value bound: it counts the
/// bound's depth and nodes, finds a `Var` or a degenerate term only the bound
/// holds, and every copier carries it as a value.
#[test]
fn every_walker_takes_a_value_bound() {
    let f = fixture();
    let a = &f.arena;
    assert_eq!(a.depth(f.bound), 3);
    assert_eq!(a.depth(f.body), 2);
    assert_eq!(a.depth(f.inner), a.depth(f.bound) + 1);
    assert_eq!(
        a.node_count_subtree(f.inner),
        1 + a.node_count_subtree(f.body) + a.node_count_subtree(f.bound),
        "the bound's nodes are the fold's"
    );

    let mut only_in_the_bound = ExprArena::new();
    let one = only_in_the_bound.push_const(1.0);
    let index = only_in_the_bound.push_var(binder(1).var());
    let var_fold = only_in_the_bound.push_reduce(sum(0), Bound::Lit(0), Bound::Value(index), one);
    assert!(only_in_the_bound.has_var(var_fold));

    let mut degenerate = ExprArena::new();
    let u = degenerate.declare_uniform(Uniform::new(2.0).decl());
    let count = degenerate.push_uniform(u);
    let zero = degenerate.push_const(0.0);
    let div = degenerate.push_binary(OpKind::Div, count, zero);
    let one = degenerate.push_const(1.0);
    let bad = degenerate.push_reduce(sum(0), Bound::Lit(0), Bound::Value(div), one);
    assert!(degenerate.has_degenerate(bad));

    let mut spliced = ExprArena::new();
    let root = spliced.splice(a, f.outer);
    assert!(is_the_fixture(&spliced, root, &f), "splice");

    let (compact, root) = a.compact(f.outer);
    assert!(is_the_fixture(&compact, root, &f), "compact");

    let mut rebuilt = ExprArena::new();
    let root = a.rebuild_into(f.outer, &mut rebuilt);
    assert!(is_the_fixture(&rebuilt, root, &f), "rebuild_into");

    let kernel = Kernel::from_parts(a.clone(), f.outer);
    let (parts, root) = kernel.parts();
    assert!(is_the_fixture(parts, root, &f), "from_parts and parts");

    // `fmt_expr`'s own spelling: the header with `_` for the child it stands
    // for, then the body, then the bound.
    assert_eq!(
        format!("{}", a.display(f.inner)),
        "reduce[add_4[0.._)](mul(Var(4), Var(1)), add(add(Var(5), Uniform(0)), Const(2)))",
        "the bound is printed after the body"
    );
}

/// Substituting a coordinate rewrites the body that reads it and leaves a bound
/// that reads none alone: the same node. (The caller's shape: coordinates, not
/// a slot a nested fold binds.)
#[test]
fn substituting_a_coordinate_leaves_a_bound_alone() {
    let mut f = fixture();
    let c = f.arena.push_const(5.0);
    let root = f.arena.substitute_vars_with(f.outer, &[(Axis::Y.var(), c)]);
    let ExprNode::Reduce { body: inner, .. } = f.arena.node(root) else {
        panic!("a fold, got {}", f.arena.display(root));
    };
    let ExprNode::Reduce { hi, body, .. } = f.arena.node(inner) else {
        panic!("a fold, got {}", f.arena.display(inner));
    };
    assert_eq!(hi, Bound::Value(f.bound), "the bound is the same node");
    assert_ne!(body, f.body, "the body read Y");
    assert!(!f.arena.has_var(c));
}

/// A key is a fold's header and its children's keys, and the header has no
/// field for a child yet: a value-bounded kernel is refused by name rather than
/// keyed as another fold.
#[test]
#[should_panic(expected = "a fold bound that is a value")]
fn a_value_bounded_kernel_has_no_key_until_a4() {
    let f = fixture();
    let _refused = KernelKey::of(&f.arena, f.outer);
}

/// A bound that reads a binder no fold around it binds makes the term open: the
/// inner fold alone is open in the enclosing index, and the fixture, which
/// binds it, is closed.
#[test]
fn a_bound_reading_a_free_binder_makes_the_term_open() {
    let f = fixture();
    assert_eq!(f.arena.free_index(f.inner), Some(binder(1).var()));
    assert_eq!(f.arena.free_index(f.outer), None);
}

/// L4 at the public function: the inner fold varies with its body less its
/// index and with its bound, unmasked, so it reads Y and the enclosing index;
/// the outer binds that, and reads Y.
#[test]
fn variance_follows_l4_at_the_public_api() {
    let f = fixture();
    let v = compute_arena_variance(&f.arena);
    assert_eq!(
        v[f.inner.0 as usize],
        Variance::Y.union(Variance::from_var(binder(1).var()))
    );
    assert_eq!(v[f.outer.0 as usize], Variance::Y);
}

/// A constant that names a count is the fold's data however the caller spelled
/// it, and one that names none is a value the fold keeps.
#[test]
fn a_constant_bound_is_read_where_it_is_pushed() {
    let mut a = ExprArena::new();
    let body = a.push_var(binder(0).var());
    let four = a.push_lit(Lit::Real(4.0));
    let by_term = a.push_reduce(sum(0), Bound::Lit(0), Bound::Value(four), body);
    let by_literal = a.push_reduce(sum(0), Bound::Lit(0), Bound::Lit(4), body);
    assert_eq!(by_term, by_literal);
    let half = a.push_lit(Lit::Real(0.5));
    let kept = a.push_reduce(sum(0), Bound::Lit(0), Bound::Value(half), body);
    assert_ne!(kept, by_literal);
    assert_eq!(a.known_range(kept), None);
}
