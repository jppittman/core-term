//! Scoping: a flat schedule to the nest the allocator and the emitter walk.
//!
//! Lowering ([`super::lower`]) produces one flat schedule. Scoping carves each
//! surviving `Reduce` out into a scope of its own and moves every value to the
//! outermost scope that binds what it depends on. [`Nest`] is the one way in;
//! nothing here names a register or a byte.

use alloc::vec::Vec;

use super::ownership::Positions;
use super::{
    Def, ScheduledOp, Scope, ScopeFold, ScopeRegion, ScopedSchedule, ValueId, all_operands,
    structural_children,
};
use crate::error::CompileError;
use crate::pass::Pass;

/// Split a flat schedule into its nest, placing every value in the outermost
/// scope that binds all the binders it depends on.
///
/// Two steps, and the second is the placement. [`extract_folds`] carves each
/// surviving `Reduce`'s closure out into a scope of its own, one level per
/// fold; what that leaves in a fold's schedule is everything its body
/// reaches whose variance names no binder deeper than the fold's own — which
/// still includes values invariant *in* the fold, computed in it once per
/// trip. [`place_roots`] then moves each of those up to the outermost scope
/// binding its binders, where it is computed once, and takes it out of the
/// fold's schedule, which holds only what the fold defines. That is loop-invariant code motion
/// out of every fold at once — the lattice's rows and columns included, so
/// a per-call value is computed per call and a per-row one per row
/// (docs/plans/2026-09-16-collapse-is-a-fold.md §2.2) — asked as one
/// question of the variance rather than as a tier per loop.
pub(crate) struct Nest;

impl Pass for Nest {
    type Input = Vec<Def>;
    type Output = ScopedSchedule<()>;

    fn run(self, schedule: Vec<Def>) -> Result<ScopedSchedule<()>, CompileError> {
        use pixelflow_ir::variance::Variance;
        // The carving asks for an operand's variance by `ValueId` at every
        // level of the nest, so one table, from the full schedule before it
        // is split. Lowering numbers its defs `0..len`, so every slot is
        // written; an id past `len` panics here rather than read another's.
        let mut variance = alloc::vec![Variance::ALL; schedule.len()];
        for def in &schedule {
            variance[def.value.0 as usize] = def.variance;
        }
        let (body, pending) = extract_folds(schedule, &variance);
        let mut scoped = ScopedSchedule {
            body: ScopeRegion {
                schedule: body,
                analysis: (),
            },
            folds: Vec::new(),
        };
        attach_folds(&mut scoped, pending);
        place_roots(&mut scoped);
        Ok(scoped)
    }
}

impl<A> ScopedSchedule<A> {
    /// The closure of one scope, which [`Own`](super::Own) asserts of each
    /// scope before it reads off the scope's ownership. Ownership sees a
    /// loop's reads at its `Reduce`, one level down
    /// ([`super::guards::FoldReads`]), so a read it did not see would let an
    /// arm own a value a loop outside the arm reads. Closure is asserted both
    /// ways:
    ///
    /// - **Inward.** Every value a def of the scope reads is a def of it or in
    ///   its `outer`.
    /// - **Outward.** Every id in the `outer` of a fold the scope opens is
    ///   defined ahead of the fold's `Reduce`, or is in the scope's own `outer`;
    ///   and the fold's body root is a def of the fold or in its `outer`.
    ///
    /// Linear: this scope's defs and its `outer` are indexed densely by
    /// `ValueId` once, and each fold it opens costs the size of its own `outer`
    /// and schedule.
    pub(super) fn assert_closed(&self, scope: Scope) {
        let (schedule, outer, _) = self.at(scope);
        let defined = Positions::of(schedule);
        // Ids are dense (handed out sequentially), so membership is one index.
        let capacity = outer.iter().map(|id| id.0 as usize + 1).max().unwrap_or(0);
        let mut from_outside = alloc::vec![false; capacity];
        for id in outer {
            from_outside[id.0 as usize] = true;
        }
        let read_from_outside =
            |id: ValueId| from_outside.get(id.0 as usize).copied().unwrap_or(false);

        for def in schedule {
            for read in all_operands(&def.op) {
                assert!(
                    defined.get(read).is_some() || read_from_outside(read),
                    "{:?} reads {read:?}, which {scope:?} neither defines nor reads from outside",
                    def.value
                );
            }
        }
        for fold in self.folds.iter().filter(|fold| fold.parent == scope) {
            let (opens, body) = match defined.get(fold.reduce).map(|at| (at, &schedule[at].op)) {
                Some((at, &ScheduledOp::Reduce(_, body))) => (at, body),
                _ => panic!(
                    "{:?} opens a fold but is not a Reduce def of {scope:?}",
                    fold.reduce
                ),
            };
            for read in &fold.outer {
                let ahead = defined
                    .get(*read)
                    .map_or_else(|| read_from_outside(*read), |at| at < opens);
                assert!(
                    ahead,
                    "{read:?}, read inside the fold opening at {:?}, is neither defined ahead of \
                     it in {scope:?} nor read by {scope:?} from outside",
                    fold.reduce
                );
            }
            assert!(
                fold.schedule.iter().any(|def| def.value == body) || fold.outer.contains(&body),
                "{body:?}, the body of the fold opening at {:?}, is neither defined by it nor \
                 read from outside",
                fold.reduce
            );
        }
    }
}

/// The second half of [`Nest`]: in every fold, each
/// def whose variance does not name the fold's binder is computed by an
/// enclosing scope — the outermost binding every binder it *does* name — and
/// read here from there.
///
/// The value is already in that scope's schedule: a fold's closure was carved
/// out of its parent's, and a value invariant in the fold has no bit deeper
/// than the parent's, so the parent kept it — and took it out in turn if it
/// does not bind it either. What changes is that the fold's own copy is
/// removed and the fold records it as `outer`: a fold's schedule holds only
/// what the fold defines. Two kinds of def stay whatever their variance: a
/// binder's `Var`, and a `Reduce` that opens a fold here. Any other `Reduce`
/// is a sibling fold's result, computed by the scope that opened that fold.
fn place_roots(scoped: &mut ScopedSchedule<()>) {
    use pixelflow_ir::variance::Variance;

    // What each fold binds, read off before any def is edited: its own
    // binder.
    //
    // Not the lane binder of a store the fold holds. The lane fold is inlined
    // into the storing scope, so nothing *deeper* may compute a lane-varying
    // value (`extract_folds_bound_by` keeps them out of the folds within) —
    // but the lanes themselves are the same vector in every batch, so a value
    // that varies by lane and by nothing else is invariant over the whole
    // call and belongs to the body, like any other invariant.
    let binds: Vec<Variance> = (0..scoped.folds.len())
        .map(|j| Variance::from_var(binder_of_fold(scoped, j)))
        .collect();
    for (j, binder) in binds.into_iter().enumerate() {
        // The `Reduce` defs in this fold's schedule at which a fold of its own
        // opens.
        let opened: Vec<ValueId> = scoped
            .folds
            .iter()
            .filter(|fold| fold.parent == Scope::Fold(j))
            .map(|fold| fold.reduce)
            .collect();
        let mut outer: Vec<ValueId> = Vec::new();
        let mut kept: Vec<Def> = Vec::new();
        for def in core::mem::take(&mut scoped.folds[j].schedule) {
            let defined_here = def.variance.bits() & binder.bits() != 0
                || matches!(def.op, ScheduledOp::Var(_))
                || opened.contains(&def.value);
            if defined_here {
                kept.push(def);
                continue;
            }
            outer.push(def.value);
        }
        scoped.folds[j].schedule = kept;
        scoped.folds[j].outer = outer;
    }
}

/// The lane binders a scope binds by holding a store: the lane fold is
/// executed by lanes, inlined into the scope its `Write` def sits in, so
/// that scope is where a lane-varying value lives — nothing deeper binds
/// the lane, and nothing shallower may compute a value that varies by it.
fn lane_binders(schedule: &[Def]) -> pixelflow_ir::variance::Variance {
    use pixelflow_ir::variance::Variance;
    schedule
        .iter()
        .fold(Variance::CONST, |acc, def| match def.op {
            ScheduledOp::Write { lane, .. } => acc.union(Variance::from_var(lane.var())),
            _ => acc,
        })
}

/// A fold's binder, read off the `Reduce` def it opens at in its parent.
fn binder_of_fold(scoped: &ScopedSchedule<()>, j: usize) -> u8 {
    let fold = &scoped.folds[j];
    let schedule = match fold.parent {
        Scope::Body => &scoped.body.schedule,
        Scope::Fold(p) => &scoped.folds[p].schedule,
    };
    let Some(ScheduledOp::Reduce(meta, _)) = schedule
        .iter()
        .find(|def| def.value == fold.reduce)
        .map(|def| &def.op)
    else {
        panic!("Fold({j}) opens at a def that is not a Reduce")
    };
    meta.binder().var()
}

/// A fold `extract_folds` carved out of a flat schedule, still looking for
/// its position: everything the body's closure computed, in topological
/// order, ending at the value the `Reduce` combines each iteration.
struct PendingFold {
    /// The `Reduce` def's own `ValueId` — a schedule permutation (LICM) may
    /// move it, but never renames it, so this is what
    /// `attach_folds` searches the post-partition schedule for.
    reduce_vid: ValueId,
    /// The fold's own per-iteration computation, in topological order,
    /// ending at the body's root.
    schedule: Vec<Def>,
    /// The folds whose `Reduce` def sits in `schedule`: a fold inside this
    /// one's body. The nest is a tree, and this is the recursion.
    children: Vec<PendingFold>,
}

/// Carve every surviving `Reduce`'s body out of `schedule`.
///
/// A fold inside a fold's body is carved the same way, one level down: the
/// outer fold's closure is a schedule like any other, and the inner fold is
/// a `Reduce` def in it. `variance` is every def's [`Def::variance`] by
/// `ValueId`, built once by [`Nest`]; a fold's
/// schedule keeps its parent's ids, which is what lets one array answer an
/// operand's variance at every level.
fn extract_folds(
    schedule: Vec<Def>,
    variance: &[pixelflow_ir::variance::Variance],
) -> (Vec<Def>, Vec<PendingFold>) {
    let top = alloc::vec![false; variance.len()];
    extract_folds_bound_by(
        schedule,
        variance,
        pixelflow_ir::variance::Variance::CONST,
        &top,
    )
}

/// [`extract_folds`] for one scope, `bound` being the binders that scope is
/// *inside*: none at the top, a fold's own binder and its ancestors' for
/// that fold's body.
///
/// `bound` is what decides which values leave. Nothing outside a fold can
/// read its own binder — that is what makes it a binder — so a value whose
/// variance names a binder this scope is *not* inside can only belong to a
/// fold nested deeper, and dropping it here is safe by construction. A value
/// a fold's closure also reached but whose variance names no deeper binder
/// (a shared invariant leaf, or one that varies only with an enclosing
/// binder) is not dropped: it stays here too, and stays in the fold's own
/// schedule as well — where [`place_roots`] takes it out, to be read from
/// the enclosing scope. Getting this backwards — removing the
/// whole closure — would orphan exactly that shared leaf's other consumer.
/// The closure ends at such a value: what it is built from is the computing
/// scope's business, not the fold's, and is not carried in behind it.
///
/// The one thing that is *not* recomputed inside a fold is another fold
/// that does not depend on its binder: a whole loop per iteration is the
/// glyph's winding sum run once per piece of its distance fold, and once
/// more at the top for the coverage that reads it. Such a `Reduce` stays
/// the enclosing scope's fold, and the fold that reads its result keeps its
/// def as a **placeholder** for now — in the schedule, so the closure walk
/// resolves, but opening no scope here; [`place_roots`] takes it out, and the
/// fold reads the result the enclosing scope's loop bound.
/// `placeholder` is that verdict from the level above, by `ValueId`, so a
/// level never mistakes one for a fold of its own.
fn extract_folds_bound_by(
    schedule: Vec<Def>,
    variance: &[pixelflow_ir::variance::Variance],
    bound: pixelflow_ir::variance::Variance,
    placeholder: &[bool],
) -> (Vec<Def>, Vec<PendingFold>) {
    use pixelflow_ir::variance::Variance;

    // `ValueId` space, not this schedule's length: a fold's schedule is a
    // subset of its parent's, keeping the parent's ids.
    let n = variance.len();
    let mut position: Vec<Option<usize>> = alloc::vec![None; n];
    for (i, def) in schedule.iter().enumerate() {
        position[def.value.0 as usize] = Some(i);
    }
    // The lane fold is executed by lanes, inlined into the scope holding its
    // store — so that scope binds its binder, and a lane-varying value is
    // computed here, not somewhere deeper.
    let bound = bound.union(lane_binders(&schedule));
    let deeper = Variance::BINDERS.bits() & !bound.bits();
    let hoisted =
        |v: ValueId| placeholder[v.0 as usize] || variance[v.0 as usize].bits() & deeper == 0;

    let mut pending: Vec<PendingFold> = Vec::new();
    // A `Reduce` inside another's closure is that one's child, not this
    // scope's own fold. The schedule is topological, so an outer fold's def
    // comes after everything its body reaches; walking it backwards meets
    // the outer fold first, and what its closure claims is skipped here and
    // carved out by the recursion instead.
    let mut claimed = alloc::vec![false; n];

    for def in schedule.iter().rev() {
        let ScheduledOp::Reduce(fold, body_vid) = def.op else {
            continue;
        };
        if claimed[def.value.0 as usize] || placeholder[def.value.0 as usize] {
            continue;
        }
        // The fold's own closure: everything its body needs, reachable by
        // operand from its root. This *includes* any invariant leaf it
        // shares with code outside it (a `Uniform`, a shared sub-expression)
        // — reachability says nothing about whether such a value depends on
        // *this* binder, which is why it is not what decides removal below.
        //
        // It stops *at* such a leaf, though. A value the enclosing scope
        // computes is a placeholder in this fold and in every fold inside
        // it — read from its park, never emitted — so nothing behind it is
        // this fold's to read. Following through would carry its operands
        // in as placeholders no def here reads: roots the computing scope
        // must then park, for nobody. The `Uniform`s' base pointer was that
        // root, stored to the frame once per call for a fold that reads only
        // the uniforms themselves.
        //
        // A nested `Reduce` that depends on a binder bound here is followed
        // *into*: its body is not an operand (`operands` says so —
        // the def's own emission never reads it), but it is this closure's
        // to carry — `structural_children`, which the walk below
        // follows, yields it — so the recursion below can carve it out again
        // one level down. One that does not is a placeholder like any other
        // hoisted value, remembered so the level below does not mistake it
        // for a fold of its own.
        let mut mark = alloc::vec![false; n];
        let mut placeholder_here = alloc::vec![false; n];
        let op_of = |v: ValueId| position[v.0 as usize].map(|p| &schedule[p].op);
        let is_fold = |v: ValueId| matches!(op_of(v), Some(ScheduledOp::Reduce(..)));
        let mut stack = Vec::new();
        mark[body_vid.0 as usize] = true;
        // The root too: a body that *is* another fold's result reads that
        // result from the enclosing scope, and the whole schedule is the placeholder.
        if hoisted(body_vid) {
            placeholder_here[body_vid.0 as usize] = is_fold(body_vid);
        } else {
            stack.push(body_vid);
        }
        while let Some(v) = stack.pop() {
            let at = position[v.0 as usize].unwrap_or_else(|| {
                panic!(
                    "{v:?} is read by {:?}'s body but is not in its scope",
                    def.value
                )
            });
            let op = &schedule[at].op;
            if matches!(op, ScheduledOp::Reduce(..)) {
                claimed[v.0 as usize] = true;
            }
            for operand in structural_children(op) {
                if mark[operand.0 as usize] {
                    continue;
                }
                mark[operand.0 as usize] = true;
                if hoisted(operand) {
                    placeholder_here[operand.0 as usize] = is_fold(operand);
                    continue;
                }
                stack.push(operand);
            }
        }
        let fold_schedule: Vec<Def> = schedule
            .iter()
            .filter(|d| mark[d.value.0 as usize])
            .cloned()
            .collect();
        let inside = bound.union(Variance::from_var(fold.binder().var()));
        let (fold_schedule, children) =
            extract_folds_bound_by(fold_schedule, variance, inside, &placeholder_here);
        pending.push(PendingFold {
            reduce_vid: def.value,
            schedule: fold_schedule,
            children,
        });
    }
    // Schedule order, so fold indices are stable whichever way this walked.
    pending.reverse();

    if pending.is_empty() {
        return (schedule, pending);
    }

    let remaining: Vec<Def> = schedule
        .into_iter()
        .filter(|def| def.variance.bits() & deeper == 0)
        .collect();
    (remaining, pending)
}

/// Record each [`PendingFold`] of the body as a [`ScopeFold`], and its
/// children inside it.
///
/// A fold names the `Reduce` def it opens at by value, not by position:
/// `place_roots`, which runs after this, edits the schedules, and it moves a
/// `Def` without renaming the `ValueId` it defines. That the def is in the
/// parent's schedule is asserted where ownership asks for the fold's reads
/// ([`super::guards::FoldReads::new`]).
fn attach_folds(scoped: &mut ScopedSchedule<()>, pending: Vec<PendingFold>) {
    for fold in pending {
        attach_fold(scoped, fold, Scope::Body);
    }
}

/// Record `fold` as a [`ScopeFold`] opening in `parent`, then each of its
/// children inside it — depth first, so a parent's index is always below its
/// children's, which is the order the `Allocate` pass (`regalloc::local`) and
/// the frame layout both walk the tree in.
fn attach_fold(scoped: &mut ScopedSchedule<()>, fold: PendingFold, parent: Scope) {
    let PendingFold {
        reduce_vid,
        schedule,
        children,
    } = fold;
    let index = scoped.folds.len();
    scoped.folds.push(ScopeFold {
        parent,
        reduce: reduce_vid,
        outer: Vec::new(),
        schedule,
        analysis: (),
    });
    for child in children {
        attach_fold(scoped, child, Scope::Fold(index));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tests::schedule_for;
    use pixelflow_ir::LatticeShape;
    use pixelflow_ir::arena::{ExprArena, ExprId, UniformDecl, UniformIdentity};
    use pixelflow_ir::kind::OpKind;

    /// Lanes in one SIMD batch at the tier this host selected.
    fn lanes() -> usize {
        crate::jit_vector_bytes() / core::mem::size_of::<f32>()
    }

    /// One full batch of one row: `x` runs `x0 .. x0 + lanes()`, which is
    /// what a test about per-lane behaviour needs.
    fn batch() -> LatticeShape {
        LatticeShape::new([lanes() as u32, 1])
    }

    /// [`schedule_for`] at this host's own lane count.
    fn native_schedule(a: &ExprArena, root: ExprId, shape: LatticeShape) -> Vec<Def> {
        schedule_for(a, root, shape, lanes() as u32)
    }

    /// `y·k + x·k`: one constant, read by a row-invariant term and a
    /// column-varying one, so the inner scope needs a value the outer one
    /// computes.
    fn shared_leaf_kernel() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let k = a.push_const(3.5);
        let invariant = a.push_binary(OpKind::Mul, y, k);
        let varying = a.push_binary(OpKind::Mul, x, k);
        let root = a.push_binary(OpKind::Add, invariant, varying);
        (a, root)
    }

    fn decl(default: f32) -> UniformDecl {
        UniformDecl {
            id: UniformIdentity::mint(),
            default,
        }
    }

    /// A constant shared between a lattice-invariant expression and a varying
    /// one is computed by the outer scope and parked for the inner one, like
    /// any other value the inner scope reads but does not vary.
    ///
    /// It used to be computed in both: `place_roots` left a leaf where it was,
    /// on the theory that nothing is saved by parking a value one instruction
    /// rebuilds. Two instructions on x86, per read, per trip — and a parked
    /// root is carried in a register when one is free, which is none.
    #[test]
    fn a_leaf_feeding_both_scopes_is_parked_by_the_outer_one() {
        let (a, root) = shared_leaf_kernel();
        let schedule = native_schedule(&a, root, batch());
        let scoped = Nest.run(schedule).expect("nest");

        let k = scoped
            .body
            .schedule
            .iter()
            .find(|d| matches!(d.op, ScheduledOp::Const(v) if v == 3.5))
            .map(|d| d.value)
            .expect("the body computes the constant");
        let in_a_fold = scoped
            .folds
            .iter()
            .flat_map(|f| f.schedule.iter())
            .any(|d| d.value == k);
        assert!(
            !in_a_fold && scoped.folds.iter().any(|f| f.outer.contains(&k)),
            "the fold reads it from the body, never through a copy of its own"
        );
    }

    /// `x + u·u`: the uniform's load and the product that depends on it
    /// alone are per-call work. Asserted on the nest — which scope holds
    /// them — not on timing.
    #[test]
    fn a_uniform_and_what_depends_on_it_alone_land_in_the_body() {
        let mut a = ExprArena::new();
        let u = a.declare_uniform(decl(3.0));
        let x = a.push_var(0);
        let uu = a.push_uniform(u);
        let sq = a.push_binary(OpKind::Mul, uu, uu);
        let root = a.push_binary(OpKind::Add, x, sq);

        let schedule = native_schedule(&a, root, batch());
        let scoped = Nest.run(schedule).expect("nest");

        // The kernel's own uniform, told from the origin's two by the
        // block it is read from: the link's, at the context slot after
        // the (empty) buffer table, rather than the origin block after
        // that. The block's pointer is a `Context` def of its own,
        // loaded once per call in the body.
        let link_block = scoped
            .body
            .schedule
            .iter()
            .find(|d| matches!(d.op, ScheduledOp::Context(0)))
            .map(|d| d.value)
            .expect("the link's block pointer is loaded once per call");
        let kernel_uniform =
            |op: &ScheduledOp| matches!(op, ScheduledOp::Uniform(base, _) if *base == link_block);
        assert!(
            scoped.body.schedule.iter().any(|d| kernel_uniform(&d.op)),
            "the broadcast load is once per call"
        );
        let sq_vid = scoped
            .body
            .schedule
            .iter()
            .find(|d| match d.op {
                ScheduledOp::Binary(OpKind::Mul, l, r) => l == r,
                _ => false,
            })
            .map(|d| d.value)
            .expect("u·u is computed once per call");
        assert!(
            scoped.folds.iter().any(|f| f.outer.contains(&sq_vid)),
            "the product is read by the scopes inside, not recomputed"
        );
        assert!(
            !scoped
                .folds
                .iter()
                .flat_map(|f| f.schedule.iter())
                .any(|d| kernel_uniform(&d.op)),
            "the folds read the parked product, never the block"
        );
    }
}
