//! **A bounded fold is a node the e-graph can hold.**
//!
//! It could not be, and the reason was the encoding rather than anything
//! about folds. `Reduce` was `Nary(Reduce, [Const(combiner), Const(var),
//! Const(extent), body])`, so its metadata arrived as *e-classes*: the
//! `Const(4.0)` naming the `Add` combiner was the same class as any literal
//! `4.0` elsewhere in the kernel, and a trip count sat where every arithmetic
//! rule in the set could reach it. `op_from_kind` therefore declined
//! `OpKind::Reduce` outright and every pipeline lowered folds away before
//! saturation — which is also why the decompositions the user asked for
//! (`⊕_{[a,b)} f = f(a) ⊕ ⊕_{[a+1,b)} f`) had nowhere to be stated.
//!
//! With the metadata in the node's identity ([`pixelflow_ir::Fold`]) the
//! obstruction is gone. These are the properties that say so: a fold inserts,
//! hash-conses on exactly its meaning, keeps its binder out of the variance
//! it reports, and survives a round trip through extraction unchanged.
//!
//! Not yet asserted here, because it is not yet built: that the graph can
//! *rewrite* a fold. The decomposition rules are the next change, and the
//! pipeline reorder (legalize after saturation, as the fallback for what the
//! graph declined) is the one after that.

use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval::eval_scalar;
use pixelflow_ir::{Binder, ExprArena, ExprId, ExprNode, Fold, Monoid, OpKind};
use pixelflow_search::egraph::{CostModel, EGraph, Vocabulary, extract, insert};

/// The `slot`-th binder.
fn binder(slot: u8) -> Binder {
    Binder::from_slot(slot).expect("a live binder")
}

/// `⊕_{k ∈ range} (k · k)` — a body that genuinely reads the index, so a
/// mis-substituted binder shows up as a wrong number rather than as nothing.
fn squares(
    arena: &mut ExprArena,
    monoid: Monoid,
    b: Binder,
    range: core::ops::Range<u32>,
) -> ExprId {
    let k = arena.push_var(b.var());
    let body = arena.push_binary(OpKind::Mul, k, k);
    arena.push_reduce(Fold::new(monoid, b, range), body)
}

/// The headline: inserting a fold no longer declines.
#[test]
fn a_fold_enters_the_graph() {
    let mut arena = ExprArena::new();
    let root = squares(&mut arena, Monoid::SUM, binder(0), 0..4);

    let mut eg = EGraph::new();
    let class = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");

    // Two classes and no more: the fold, and its body's multiply — plus the
    // binder leaf. The metadata contributes *nothing*, which is the whole
    // point: three `Const` children would have been three more classes, each
    // one shared with any literal of the same value in the kernel.
    let nodes = eg.nodes(class);
    assert_eq!(nodes.len(), 1);
    assert!(
        matches!(&nodes[0], pixelflow_search::egraph::ENode::Reduce { .. }),
        "the root class holds the fold itself"
    );
    assert_eq!(
        eg.num_classes(),
        3,
        "binder leaf, body multiply, fold — and nothing for the metadata"
    );
}

/// Hash-consing is on exactly the fold's meaning: same algebra, same binder,
/// same range, same body is one node; any difference is another.
#[test]
fn folds_hash_cons_on_their_meaning() {
    let mut arena = ExprArena::new();
    let a = squares(&mut arena, Monoid::SUM, binder(0), 0..4);
    let same = squares(&mut arena, Monoid::SUM, binder(0), 0..4);
    let longer = squares(&mut arena, Monoid::SUM, binder(0), 0..5);
    let shifted = squares(&mut arena, Monoid::SUM, binder(0), 1..4);
    let other_algebra = squares(&mut arena, Monoid::PRODUCT, binder(0), 0..4);
    let other_binder = squares(&mut arena, Monoid::SUM, binder(1), 0..4);

    let mut eg = EGraph::new();
    let insert_one =
        |eg: &mut EGraph, id| insert(&arena, id, eg, Vocabulary::Runtime).expect("a fold inserts");
    let ca = insert_one(&mut eg, a);
    assert_eq!(ca, insert_one(&mut eg, same), "the same fold is one node");
    for (label, id) in [
        ("a longer range", longer),
        ("a shifted range", shifted),
        ("another algebra", other_algebra),
        ("another binder", other_binder),
    ] {
        assert_ne!(
            ca,
            insert_one(&mut eg, id),
            "{label} is a different fold and must not share a class"
        );
    }
}

/// The binder is *bound*. `Σ_{k} k·k` reads no coordinate, so it is a
/// constant of the lattice — which is what makes such a fold hoistable out of
/// the pixel loop, and what the old encoding could only report by giving up
/// (a malformed binder came back as `Variance::ALL`).
#[test]
fn a_folds_binder_is_not_free_in_its_result() {
    let mut arena = ExprArena::new();
    let root = squares(&mut arena, Monoid::SUM, binder(0), 0..4);
    let v = pixelflow_ir::variance::compute_arena_variance(&arena);

    let ExprNode::Reduce { body, .. } = *arena.node(root) else {
        panic!("expected a fold");
    };
    assert!(
        v[body.0 as usize].depends_on_binder(),
        "inside the fold the index is free"
    );
    assert!(
        v[root.0 as usize].is_const(),
        "outside it, the fold reads nothing at all: {:?}",
        v[root.0 as usize]
    );
}

/// A fold survives extraction as a fold, and the arena that comes back
/// evaluates to what went in. Extraction prices it at the prohibitive
/// sentinel — a surviving fold must be unrolled by a legalizer that runs
/// after everything that could have folded across the copies — but a class
/// holding only a fold has nothing else to choose, so it comes back.
#[test]
fn a_fold_round_trips_through_extraction() {
    let mut arena = ExprArena::new();
    let root = squares(&mut arena, Monoid::SUM, binder(0), 0..4);

    let mut eg = EGraph::new();
    let class = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
    let (out, out_root, _cost) = extract(&eg, class, &CostModel::latency_prior());

    let ExprNode::Reduce { fold, .. } = *out.node(out_root) else {
        panic!("extraction must return the fold, not something else");
    };
    assert_eq!(fold.monoid(), Monoid::SUM);
    assert_eq!(fold.range(), 0..4);
    assert_eq!(fold.binder(), binder(0));

    // Σ_{k<4} k² = 0 + 1 + 4 + 9. The interpreter reads the fold directly,
    // so this checks the extracted arena means what the original did.
    let bindings = BindingTable::empty();
    assert_eq!(eval_scalar(&out, out_root, &[0.0; 2], &bindings), 14.0);
    assert_eq!(
        eval_scalar(&arena, root, &[0.0; 2], &bindings),
        eval_scalar(&out, out_root, &[0.0; 2], &bindings)
    );
}

/// An empty fold is its monoid's identity whatever its body says — and the
/// range that makes it empty is a field, so no rewrite can turn a non-empty
/// fold into this one by folding a `Const` child to zero.
#[test]
fn an_empty_fold_is_the_identity() {
    for (monoid, want) in [
        (Monoid::SUM, 0.0),
        (Monoid::PRODUCT, 1.0),
        (Monoid::MIN, f32::INFINITY),
        (Monoid::MAX, f32::NEG_INFINITY),
    ] {
        let mut arena = ExprArena::new();
        let root = squares(&mut arena, monoid, binder(0), 3..3);
        let mut eg = EGraph::new();
        let class = insert(&arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let (out, out_root, _cost) = extract(&eg, class, &CostModel::latency_prior());
        assert_eq!(
            eval_scalar(&out, out_root, &[0.0; 2], &BindingTable::empty()),
            want,
            "{monoid:?} over an empty range"
        );
    }
}
