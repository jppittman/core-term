//! The decompositions of a bounded fold, as e-graph rewrites.
//!
//! ```text
//! ⊕_{[lo,hi)} f  =  f(lo) ⊕ ⊕_{[lo+1,hi)} f      (peel)
//! ⊕_{[lo,lo)} f  =  identity(⊕)                   (empty)
//! ```
//!
//! These are the rules the encoding used to make unstatable. While a fold's
//! algebra, binder and range were `Const` children, a rewrite that changed the
//! range would have had to rewrite an *e-class* holding a number — the same
//! class as any literal of that value elsewhere in the kernel. With the
//! metadata in the node's identity ([`pixelflow_ir::Fold`]) a peel changes
//! only the node, and the tail shares the original body's class.
//!
//! **Peel to exhaustion is the unroll.** `passes::expand_reduce` — the
//! legalizer that runs when a fold survives extraction, because codegen has no
//! iteration binder — is the same operation at a different budget, and both go
//! through [`Fold::peel`]. That is why the pipeline can put the legalizer
//! *after* saturation: what the graph unrolled, it unrolled and then folded and
//! CSE'd across; what it declined, the legalizer finishes.
//!
//! ## Substituting under a binder, in an e-graph
//!
//! Peeling needs `body[binder := lo]`, and substitution is where e-graphs and
//! binders meet. Two things make it affordable and sound here:
//!
//! - **A representative suffices.** Every node in a class denotes the same
//!   value, and `f ≡ g ⟹ f[x:=c] ≡ g[x:=c]`, so substituting through one
//!   representative gives a term equal to the substitution of the whole class.
//!   It is less *complete* — nodes added to the class later get no substituted
//!   twin — never wrong. `ChainRule` differentiates a representative for the
//!   same reason.
//! - **Hash-consing is the invariance test.** A subtree that does not mention
//!   the binder rebuilds to the identical node, which `EGraph::add` returns the
//!   existing class for. The rule short-circuits those into metavariables
//!   anyway, so the template it emits is only the binder-dependent spine.
//!
//! A class can reach itself (`neg(neg(x)) = x`, merged), and a substitution
//! through a cycle does not terminate. The walk detects re-entry and declines
//! the whole rewrite, which is always sound.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use pixelflow_ir::{Binder, Fold, Monoid};

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops;
use super::rewrite::{Rewrite, RewriteAction};

/// What a [`HeadNode`]'s child is: an earlier entry in the same plan, or an
/// e-class that already exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadRef {
    /// Entry `i` of the plan, which precedes this one.
    Plan(u32),
    /// A class the substitution did not move — the sharing that makes a peel
    /// cost the binder-dependent spine and nothing else.
    Class(EClassId),
}

/// One node of a peeled term, in build order.
///
/// A *plan* rather than an [`ExprArena`](pixelflow_ir::ExprArena) template,
/// because the head is a copy of a term the graph already holds and may
/// therefore contain any op the graph holds — a `Gather`, a mask — while
/// [`ops::op_from_kind`], which is what turns an arena template into nodes,
/// deliberately resolves only the ops a *rewrite rule* may name. Carrying the
/// `&'static dyn Op` from the node it was copied from needs no resolver at
/// all, and cannot disagree with the one `insert` used.
#[derive(Clone, Debug)]
pub enum HeadNode {
    /// The literal the peeled index substituted in, by bit pattern.
    Const(u32),
    Op {
        op: &'static dyn ops::Op,
        children: Vec<HeadRef>,
    },
    Reduce {
        fold: Fold,
        body: HeadRef,
    },
}

/// `⊕_{[lo,hi)} f = ⊕_{[lo,hi-1)} f ⊕ f(hi-1)`.
///
/// From the *back*, so running it to exhaustion builds the same left-leaning
/// chain `passes::expand_reduce` does. Peeling from the front is the same
/// value in the opposite association, and the difference is not cosmetic: it
/// measured 23–42% more emitted nodes on production glyphs, because the graph
/// then has to reassociate an n-deep chain to reach the shape the cost model
/// and the fusion rules were tuned on, and spends its class budget doing it.
/// See docs/plans/2026-09-09-a-fold-is-a-node.md §9.
pub struct PeelFold;

/// `⊕_{[lo,lo)} f = identity(⊕)` — whatever the body says.
pub struct EmptyFold;

impl Rewrite for PeelFold {
    fn name(&self) -> &str {
        "peel-fold"
    }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, body } = node else {
            return None;
        };
        let (rest, last) = fold.peel_back()?;
        // The combiner must be nameable as an `Op` before any work is done:
        // declining early costs one lookup, declining late costs a walk of
        // the whole body.
        combiner_op(fold.monoid())?;
        let (head, head_root) = substituted_body(egraph, *body, fold.binder(), last as f32)?;
        Some(RewriteAction::PeelFold {
            head,
            head_root,
            rest,
            body: *body,
        })
    }
}

impl Rewrite for EmptyFold {
    fn name(&self) -> &str {
        "empty-fold"
    }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let ENode::Reduce { fold, .. } = node else {
            return None;
        };
        fold.is_empty()
            .then(|| RewriteAction::Create(ENode::constant(fold.monoid().identity())))
    }
}

/// The two fold decompositions. Inert for a kernel with no folds in it.
#[must_use]
pub fn fold_rules() -> Vec<Box<dyn Rewrite>> {
    alloc::vec![Box::new(PeelFold) as Box<dyn Rewrite>, Box::new(EmptyFold)]
}

/// The `Op` that combines a fold's terms.
///
/// Local rather than [`ops::op_from_kind`] because it must cover the two mask
/// algebras, whose ops are deliberately *not* globally registered: doing that
/// would hand them to the AOT macro tier, where a `Dwrt` travelling with a
/// mask resolves before composition and miscompiles under a warp. Here the
/// caller has already named an algebra, so there is no opcode to leak.
pub(crate) fn combiner_op(monoid: Monoid) -> Option<&'static dyn ops::Op> {
    match monoid {
        Monoid::SUM => Some(&ops::Add),
        Monoid::PRODUCT => Some(&ops::Mul),
        Monoid::MIN => Some(&ops::Min),
        Monoid::MAX => Some(&ops::Max),
        Monoid::ANY => Some(ops::mask_or()),
        Monoid::ALL => Some(ops::mask_and()),
        _ => None,
    }
}

/// What a finished class contributed: where it landed, and whether anything
/// under it mentioned the binder.
#[derive(Clone, Copy)]
struct Done {
    at: HeadRef,
    varies: bool,
}

/// Build `body[binder := value]` as a plan, walking one representative per
/// e-class.
///
/// Returns `None` when the walk re-enters a class it is already inside: a
/// merged class can reach itself, and a substitution through a cycle does not
/// terminate. Declining costs completeness and never soundness.
fn substituted_body(
    egraph: &EGraph,
    body: EClassId,
    binder: Binder,
    value: f32,
) -> Option<(Vec<HeadNode>, HeadRef)> {
    enum Task {
        Visit(EClassId),
        Build(EClassId),
    }

    let mut plan: Vec<HeadNode> = Vec::new();
    let mut memo: BTreeMap<EClassId, Done> = BTreeMap::new();
    let mut on_stack: BTreeSet<EClassId> = BTreeSet::new();
    let mut built: Vec<Done> = Vec::new();
    let mut tasks = alloc::vec![Task::Visit(body)];

    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(class) => {
                let class = egraph.find(class);
                if let Some(&done) = memo.get(&class) {
                    built.push(done);
                    continue;
                }
                if !on_stack.insert(class) {
                    return None;
                }
                let node = egraph.nodes(class).first()?;
                // The binder itself: the one place the substitution bites.
                if matches!(node, ENode::Var(v) if *v == binder.var()) {
                    plan.push(HeadNode::Const(value.to_bits()));
                    let done = Done {
                        at: HeadRef::Plan(plan.len() as u32 - 1),
                        varies: true,
                    };
                    on_stack.remove(&class);
                    memo.insert(class, done);
                    built.push(done);
                    continue;
                }
                if node.children_slice().is_empty() {
                    let done = named(class);
                    on_stack.remove(&class);
                    memo.insert(class, done);
                    built.push(done);
                    continue;
                }
                tasks.push(Task::Build(class));
                for &child in node.children_slice().iter().rev() {
                    tasks.push(Task::Visit(child));
                }
            }
            Task::Build(class) => {
                let node = egraph.nodes(class).first()?.clone();
                let arity = node.children_slice().len();
                let start = built.len().checked_sub(arity)?;
                let kids: Vec<Done> = built.drain(start..).collect();
                let done = if kids.iter().any(|k| k.varies) {
                    plan.push(rebuild(&node, &kids)?);
                    Done {
                        at: HeadRef::Plan(plan.len() as u32 - 1),
                        varies: true,
                    }
                } else {
                    // Nothing below moved, so neither does this: the
                    // substitution is the identity on a subtree that never
                    // mentions the binder, and naming the class is how the
                    // peel shares it rather than copying it.
                    named(class)
                };
                on_stack.remove(&class);
                memo.insert(class, done);
                built.push(done);
            }
        }
    }

    // The root is named rather than assumed to be the last entry: a body that
    // never mentions the binder plans *nothing* and its head is the body's own
    // class — `⊕_{[lo,hi)} c` peeling to `c ⊕ ⊕_{[lo+1,hi)} c`, with no copy
    // made anywhere.
    Some((plan, built.pop()?.at))
}

/// A class reused as-is.
fn named(class: EClassId) -> Done {
    Done {
        at: HeadRef::Class(class),
        varies: false,
    }
}

/// One plan node, over already-planned children.
fn rebuild(node: &ENode, kids: &[Done]) -> Option<HeadNode> {
    match node {
        // A nested fold binds a slot of its own — `lowest_free_binder` never
        // reissues a live one — so the outer substitution passes through its
        // body without capture.
        ENode::Reduce { fold, .. } => Some(HeadNode::Reduce {
            fold: *fold,
            body: kids.first()?.at,
        }),
        ENode::Op { op, .. } => Some(HeadNode::Op {
            op: *op,
            children: kids.iter().map(|k| k.at).collect(),
        }),
        // Leaves never reach here: `Task::Build` is scheduled only for a node
        // with children.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::CostModel;
    use crate::egraph::extract::extract;
    use crate::egraph::saturate::SaturationConfig;
    use crate::egraph::{Vocabulary, insert};
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind, eval_scalar};

    fn binder(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live binder")
    }

    fn eval(arena: &ExprArena, root: ExprId, x: f32) -> f32 {
        eval_scalar(arena, root, &[x, 0.0], &BindingTable::empty())
    }

    /// Saturate with the fold rules alone and extract. Isolating them keeps
    /// the assertion about *these* rules rather than about whatever the whole
    /// algebra does to the residue afterwards.
    fn unroll_by_rule(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(arena, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        SaturationConfig::compatibility(60).run(&mut eg);
        let (out, out_root, _cost) = extract(&eg, class, &CostModel::latency_prior());
        (out, out_root)
    }

    fn has_fold(arena: &ExprArena) -> bool {
        arena
            .nodes_raw()
            .iter()
            .any(|n| matches!(n, ExprNode::Reduce { .. }))
    }

    /// The headline: the graph can now unroll a fold *itself*, and the result
    /// is the value the fold denotes. Extraction prefers it because a
    /// surviving fold is priced at the sentinel — codegen would have to
    /// unroll it anyway, after everything that could fold across the copies.
    #[test]
    fn the_graph_unrolls_a_fold_and_agrees_with_the_legalizer() {
        // Σ_{k ∈ [0,4)} (X + k) = 4X + 6.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let k = a.push_var(binder(0).var());
        let body = a.push_binary(OpKind::Add, x, k);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..4), body);

        let (out, out_root) = unroll_by_rule(&a, root);
        assert!(
            !has_fold(&out),
            "the extracted form must be binder-free: {}",
            out.display(out_root)
        );

        // Against the legalizer, which is the same operation at a different
        // budget — `expand_reduce` is `Fold::peel` run to exhaustion.
        let (legalized, legal_root) = pixelflow_ir::passes::expand_reduce_owned(&a, root);
        for x in [0.0f32, 1.5, -3.25, 10.0] {
            let by_rule = eval(&out, out_root, x);
            assert_eq!(by_rule, eval(&legalized, legal_root, x), "at X={x}");
            assert_eq!(by_rule, 4.0 * x + 6.0, "at X={x}");
        }
    }

    /// Every monoid, not just `Σ`. The mask algebras are the ones whose
    /// combiner `op_from_kind` deliberately will not resolve, so they are the
    /// reason `combiner_op` exists.
    #[test]
    fn every_algebra_peels() {
        for monoid in [
            Monoid::SUM,
            Monoid::PRODUCT,
            Monoid::MIN,
            Monoid::MAX,
            Monoid::ANY,
            Monoid::ALL,
        ] {
            let mut a = ExprArena::new();
            let k = a.push_var(binder(0).var());
            let body = a.push_binary(OpKind::Add, k, k);
            let root = a.push_reduce(Fold::new(monoid, binder(0), 0..3), body);

            let (out, out_root) = unroll_by_rule(&a, root);
            // Assert the *form* first: `eval_scalar` interprets a surviving
            // fold perfectly well, so a value-only assertion here would pass
            // whether or not a single peel ever fired.
            assert!(
                !has_fold(&out),
                "{monoid:?}: the rule must have unrolled it, not the interpreter"
            );
            let (legalized, legal_root) = pixelflow_ir::passes::expand_reduce_owned(&a, root);
            assert_eq!(
                eval(&out, out_root, 0.0).to_bits(),
                eval(&legalized, legal_root, 0.0).to_bits(),
                "{monoid:?}: the rule and the legalizer must agree bit for bit"
            );
        }
    }

    /// **A peel moves the range, not the body.** `Σ_{[0,3)} X` has a body
    /// that ignores the binder, so the peeled head must be *X's own e-class*
    /// — no copy — and the tail must name the same body class the original
    /// did. This is the property the range exists for: with an extent the
    /// tail would have been `Σ_{[0,2)} X(·+1)`, a rebuilt body every time.
    #[test]
    fn a_peel_shares_the_body_it_folds() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..3), x);

        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let body_class = match eg.nodes(class).first() {
            Some(ENode::Reduce { body, .. }) => eg.find(*body),
            other => panic!("expected a fold, got {other:?}"),
        };

        // One round, so exactly one peel has happened.
        SaturationConfig::compatibility(1).run(&mut eg);

        let sum = eg
            .nodes(class)
            .iter()
            .find_map(|n| match n {
                ENode::Op { op, children } if op.kind() == OpKind::Add => Some(children.clone()),
                _ => None,
            })
            .expect("the class must now also hold the peeled sum");
        // `rest` on the left, the peeled term on the right: the peel takes
        // the *last* index, so the chain leans the way `expand_reduce` builds
        // it (§9 of the plan — the other association measured 23–42% worse).
        assert_eq!(
            eg.find(sum[1]),
            body_class,
            "the peeled term is the body itself: substituting a binder the \
             body never reads is the identity, and hash-consing says so"
        );
        let rest_class = eg.find(sum[0]);
        match eg.nodes(rest_class).iter().find_map(ENode::fold) {
            Some(rest) => {
                assert_eq!(rest.range(), 0..2, "the rest is the shorter range");
                assert_eq!(
                    eg.find(match eg.nodes(rest_class).first() {
                        Some(ENode::Reduce { body, .. }) => *body,
                        other => panic!("expected a fold, got {other:?}"),
                    }),
                    body_class,
                    "and it folds the same body class, unchanged"
                );
            }
            None => panic!("the rest must be a fold"),
        }
    }

    /// The empty rule, in the graph rather than in the constructor.
    #[test]
    fn an_empty_fold_rewrites_to_its_identity() {
        for (monoid, want) in [
            (Monoid::SUM, 0.0f32),
            (Monoid::PRODUCT, 1.0),
            (Monoid::MIN, f32::INFINITY),
            (Monoid::MAX, f32::NEG_INFINITY),
        ] {
            let mut a = ExprArena::new();
            let k = a.push_var(binder(0).var());
            let root = a.push_reduce(Fold::new(monoid, binder(0), 2..2), k);
            let (out, out_root) = unroll_by_rule(&a, root);
            // The form, again before the value: an empty fold *evaluates* to
            // its identity with no rule at all, so only the shape shows that
            // the rewrite is what produced it.
            assert!(
                matches!(out.node(out_root), ExprNode::Const(_)),
                "{monoid:?}: the rule must leave a bare constant, got {}",
                out.display(out_root)
            );
            assert_eq!(eval(&out, out_root, 0.0), want, "{monoid:?}");
        }
    }

    /// A nested fold survives being peeled through: the outer substitution
    /// passes into the inner body, which binds a slot of its own.
    #[test]
    fn peeling_reaches_through_a_nested_fold() {
        // Σ_{i ∈ [0,2)} Σ_{j ∈ [0,2)} (i + j) = 0+1+1+2 = 4.
        let mut a = ExprArena::new();
        let i = a.push_var(binder(0).var());
        let j = a.push_var(binder(1).var());
        let body = a.push_binary(OpKind::Add, i, j);
        let inner = a.push_reduce(Fold::new(Monoid::SUM, binder(1), 0..2), body);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..2), inner);

        let (out, out_root) = unroll_by_rule(&a, root);
        assert!(!has_fold(&out), "both binders must be gone");
        assert_eq!(eval(&out, out_root, 0.0), 4.0);
    }
}

#[cfg(test)]
mod production_shape_tests {
    use super::*;
    use crate::egraph::extract::extract;
    use crate::egraph::saturate::SaturationConfig;
    use crate::egraph::{CostModel, Vocabulary, insert};
    use pixelflow_ir::arena::{BufferDecl, BufferIdentity};
    use pixelflow_ir::{ExprArena, ExprNode, OpKind};

    fn binder(slot: u8) -> Binder {
        Binder::from_slot(slot).expect("a live binder")
    }

    fn has_fold(arena: &ExprArena) -> bool {
        arena
            .nodes_raw()
            .iter()
            .any(|n| matches!(n, ExprNode::Reduce { .. }))
    }

    /// A glyph's winding is `Σ_i table[i]`-shaped: the body reads a *bound
    /// buffer* at the binder. That is the production shape, and it is the one
    /// the toy tests above do not cover.
    ///
    /// Extent 40 rather than 4, too: a fold long enough that peeling it is
    /// real work is where "the graph unrolls it" stops being obvious.
    #[test]
    fn a_table_reading_fold_unrolls() {
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            id: BufferIdentity::mint(),
            width: 64,
            height: 1,
        });
        let i = a.push_var(binder(0).var());
        let zero = a.push_const(0.0);
        let read = a.push_gather(buf, i, zero);
        let x = a.push_var(0);
        let body = a.push_binary(OpKind::Mul, read, x);
        let root = a.push_reduce(Fold::new(Monoid::SUM, binder(0), 0..40), body);

        let mut eg = EGraph::with_rules(fold_rules());
        let class = insert(&a, root, &mut eg, Vocabulary::Runtime).expect("a fold inserts");
        let stats = SaturationConfig::compatibility(200).run(&mut eg);
        let (out, out_root, _cost) = extract(&eg, class, &CostModel::latency_prior());

        assert!(
            !has_fold(&out),
            "40 terms over a bound table must unroll, not survive \
             (saturation {stats:?}); extracted: {} nodes",
            out.len()
        );
        assert!(out.len() > 40, "and the unrolled form is 40 terms wide");
    }
}
