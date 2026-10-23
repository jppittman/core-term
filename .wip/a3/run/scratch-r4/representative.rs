//! Round-4 soundness probe (scratch, not part of the repo's build).
//!
//! Claim under test: an L7 substitution (`HalveFold`, `PeelFold`) rebuilds a
//! nested fold's value bound from its class's *first member*
//! (`fold_rules.rs`'s `representative`), and that member's syntactic variance
//! can exceed the class's fact (the meet over members). When the first member
//! reads X, the rebuilt bound reads X, `EGraph::add`'s L3 backstop refuses the
//! rule's own fold, and saturation panics with `BoundVaries` on a term every
//! check admitted.
//!
//! The union below asserts an equality the algebra derives
//! (`((i + X) + 1) - X = i + 1`, an R-truth of the kind `(x + y) - y = x`
//! the rules make): the older class, whose first member reads X, becomes the
//! root, and `HalveFold`'s own `i + 1` lands in it.

use pixelflow_ir::fold::{Binder, Bound, Fold, Monoid};
use pixelflow_ir::{ExprArena, OpKind};
use pixelflow_search::egraph::{
    CostModel, ENode, HalveFold, PeelFold, RuleSet, Rewrite, Saturation, SaturationConfig,
    Vocabulary, extract, insert,
};

fn b(slot: u8) -> Binder {
    Binder::from_slot(slot).expect("slot")
}

/// `Σ_{i<n} Σ_{j < O} X·j`, where the class `O` holds `((i + X) + 1) - X`
/// (first, older) and `i + 1` (joined by a union). Returns the graph and the
/// outer fold's class.
fn hand_union(
    rules: Vec<Box<dyn Rewrite>>,
    n: u64,
) -> (Saturation, pixelflow_search::egraph::EClassId) {
    let mut eg = Saturation::over(RuleSet::new(rules));
    let x = eg.add(ENode::Var(0));
    let i = eg.add(ENode::Var(b(1).var()));
    let one = eg.add(ENode::constant(1.0));
    let ix = eg.add(ENode::apply(OpKind::Add, &[i, x]));
    let ix1 = eg.add(ENode::apply(OpKind::Add, &[ix, one]));
    let older = eg.add(ENode::apply(OpKind::Sub, &[ix1, x]));
    let plain = eg.add(ENode::apply(OpKind::Add, &[i, one]));
    let o = eg.union(older, plain);
    eg.rebuild();
    let first = *eg.nodes(o).next().expect("a member");
    println!("O's first member: {first:?}");
    let j = eg.add(ENode::Var(b(0).var()));
    let body = eg.add(ENode::apply(OpKind::Mul, &[x, j]));
    // L3 holds for the class: its fact is the meet, `{i}`.
    let inner = eg.add(ENode::reduce(
        Fold::new(Monoid::SUM, b(0)),
        Bound::Lit(0),
        Bound::Value(o),
        body,
    ));
    let outer = eg.add(ENode::reduce(
        Fold::new(Monoid::SUM, b(1)),
        Bound::Lit(0),
        Bound::Lit(n),
        inner,
    ));
    println!("inserted: the inner fold's bound class passed L3");
    (eg, outer)
}

/// The inner fold's upper bound: the enclosing index itself (a leaf, what
/// `Kernel::over_values` builds for a triangular fold), or `i + 1` (bound
/// arithmetic `push_reduce` admits).
#[derive(Clone, Copy, Debug)]
enum InnerBound {
    Leaf,
    PlusOne,
}

/// Where the outer body holds `((i + X) + 1) - X`, if at all.
#[derive(Clone, Copy, Debug)]
enum Junk {
    First,
    Last,
    Absent,
}

/// `Σ_{i<n} (junk + Σ_{j < bound} X·j)` from an arena, under the production
/// runtime rule set, saturated and extracted.
fn runtime_kernel(bound_kind: InnerBound, junk_at: Junk, n: u64) {
    runtime_kernel_under(bound_kind, junk_at, n, None);
}

/// `runtime_kernel` under the runtime rule set less the rule named `without`.
fn runtime_kernel_under(bound_kind: InnerBound, junk_at: Junk, n: u64, without: Option<&str>) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let i = a.push_var(b(1).var());
    let j = a.push_var(b(0).var());
    let one = a.push_const(1.0);
    let ix = a.push_binary(OpKind::Add, i, x);
    let ix1 = a.push_binary(OpKind::Add, ix, one);
    let junk = a.push_binary(OpKind::Sub, ix1, x);
    let bound = match bound_kind {
        InnerBound::Leaf => i,
        InnerBound::PlusOne => a.push_binary(OpKind::Add, i, one),
    };
    let body = a.push_binary(OpKind::Mul, x, j);
    let inner = a.push_reduce(
        Fold::new(Monoid::SUM, b(0)),
        Bound::Lit(0),
        Bound::Value(bound),
        body,
    );
    let sum = match junk_at {
        Junk::First => a.push_binary(OpKind::Add, junk, inner),
        Junk::Last => a.push_binary(OpKind::Add, inner, junk),
        Junk::Absent => inner,
    };
    let outer = a.push_reduce(Fold::new(Monoid::SUM, b(1)), Bound::Lit(0), Bound::Lit(n), sum);
    println!("{bound_kind:?} {junk_at:?} n={n}: {}", a.display(outer));
    let rules = match without {
        None => RuleSet::runtime(),
        Some(name) => {
            let mut rules = pixelflow_search::egraph::all_rules();
            rules.extend(
                pixelflow_search::egraph::fold_rules()
                    .into_iter()
                    .filter(|rule| rule.name() != name),
            );
            RuleSet::new(rules)
        }
    };
    println!("  rules: runtime less {without:?}");
    let mut eg = Saturation::over(rules);
    let class = insert(&a, outer, &mut eg, Vocabulary::Runtime).expect("inserts");
    let stats = SaturationConfig::compatibility(12).run(&mut eg);
    println!("  saturated: {:?} after {} iterations", stats.stop, stats.iterations);
    let (out, root, _) = extract(&eg, class, &CostModel::latency_prior());
    println!("  extracted: {}", out.display(root));
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    match which.as_str() {
        "halve" => {
            let (mut eg, _) = hand_union(vec![Box::new(HalveFold)], 4);
            SaturationConfig::compatibility(1).run(&mut eg);
            println!("halve: no panic");
        }
        "peel" => {
            let (mut eg, _) = hand_union(vec![Box::new(PeelFold)], 1);
            SaturationConfig::compatibility(1).run(&mut eg);
            println!("peel: no panic");
        }
        "runtime" => runtime_kernel(InnerBound::PlusOne, Junk::First, 8),
        "leaf-first" => runtime_kernel(InnerBound::Leaf, Junk::First, 8),
        "leaf-last" => runtime_kernel(InnerBound::Leaf, Junk::Last, 8),
        "leaf-absent" => runtime_kernel(InnerBound::Leaf, Junk::Absent, 8),
        "plus-absent" => runtime_kernel(InnerBound::PlusOne, Junk::Absent, 8),
        "leaf-last-no-halve" => {
            runtime_kernel_under(InnerBound::Leaf, Junk::Last, 8, Some("halve-fold"))
        }
        "leaf-last-no-peel" => {
            runtime_kernel_under(InnerBound::Leaf, Junk::Last, 8, Some("peel-fold"))
        }
        other => panic!("usage: representative halve|peel|runtime|leaf-first|leaf-last|leaf-absent|plus-absent (got {other:?})"),
    }
}
