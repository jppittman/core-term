//! What a loop a scope opens reads from it: the fact about a schedule that
//! the flow and ownership need to keep an arm from skipping a value a loop reads.
//!
//! [`FoldReads`] is the fold table — the edges a `Reduce` def's body adds to
//! the scope that opens it. Which arms are branched over, and where each
//! goes, is `layout`'s: it is chosen from who owns what (`ownership`).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::program::{Def, ScheduledOp, ValueId};

/// What each fold a scope opens reads from that scope, keyed by the fold's
/// `Reduce` def.
///
/// To [`operands`](crate::program::operands) a `Reduce` def is a
/// leaf, and for the registers the def's own instruction reads that is right:
/// once `scopes::extract_folds` has carved a fold's body into a scope of its own, the
/// def is only where the loop opens. But the loop *runs* there, and its body
/// reads this scope — a sibling fold's result from its accumulator slot, a
/// value invariant in the fold from the park this scope leaves it in. Nothing
/// in this scope's schedule recorded those reads, so to everything here that
/// asks what a value is needed by, or needs, the fold read nothing: a guard
/// could skip a fold whose result a sibling fold's body still reads, or a
/// value a fold reads, which the fold then finds unwritten
/// (`pixelflow-core/tests/guard_sibling_fold.rs` pins both). These are the
/// missing edges, and with them the def consumes what its fold reads the way
/// any def consumes its operands.
///
/// They take no walk of a body. A fold's schedule is carved out of its
/// parent's and keeps the parent's `ValueId`s, so what a fold reads from its
/// parent is among the ids it names: its schedule's, and the ones
/// `scopes::place_roots` took out of it for an enclosing scope to compute.
/// A fold nested in the fold is carved out of *that* schedule in turn, so
/// whatever it reads from here is named by the outer one too — the edges are
/// transitive by construction, and `scopes::lay_out` asserts it, both ways: a
/// fold's `outer` is defined in its parent ahead of the fold or read by the
/// parent from outside, and a fold's defs read only its own defs and its
/// `outer`. Nothing else would see a read these edges missed. Every named id
/// is an edge, a leaf the body rebuilds for itself included: that
/// over-approximates (such a leaf is kept out of an arm the fold does not run
/// in).
///
/// Keyed by a loop's `Reduce` def; the value is the ids of this scope's that
/// the loop reads.
#[derive(Default)]
pub(super) struct FoldReads(BTreeMap<ValueId, Vec<ValueId>>);

impl FoldReads {
    /// The folds opened in `scope`, each given as its `Reduce` def's
    /// `ValueId` and the ids the fold names (`ScopeFold::reads`).
    ///
    /// # Panics
    ///
    /// If a fold's `ValueId` is not a `Reduce` def of `scope`: a loop opens
    /// where its def is, and nowhere else.
    pub(super) fn new(
        scope: &[Def],
        folds: impl IntoIterator<Item = (ValueId, impl IntoIterator<Item = ValueId>)>,
    ) -> Self {
        let capacity = scope
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        // Ids are dense (handed out sequentially), so membership is one index.
        let mut here = alloc::vec![false; capacity];
        let mut opens = BTreeSet::new();
        for def in scope {
            here[def.value.0 as usize] = true;
            if matches!(def.op, ScheduledOp::Reduce(..)) {
                opens.insert(def.value);
            }
        }
        let opened = folds
            .into_iter()
            .map(|(reduce, named)| {
                assert!(
                    opens.contains(&reduce),
                    "{reduce:?} opens a fold but is not a Reduce def of its scope"
                );
                let reads = named
                    .into_iter()
                    .filter(|value| here.get(value.0 as usize).copied().unwrap_or(false))
                    .collect();
                (reduce, reads)
            })
            .collect();
        Self(opened)
    }

    /// Everything the def of `value` by `op` reads in this scope: its
    /// operands of both register classes — a gather's base pointer is a read
    /// as much as its index is — and, for a `Reduce` that opens a fold here,
    /// what the fold reads.
    pub(super) fn reads<'a>(
        &'a self,
        value: ValueId,
        op: &'a ScheduledOp,
    ) -> impl Iterator<Item = ValueId> + 'a {
        let fold: &[ValueId] = match op {
            ScheduledOp::Reduce(..) => self.0.get(&value).map_or(&[], Vec::as_slice),
            _ => &[],
        };
        crate::program::all_operands(op).chain(fold.iter().copied())
    }
}
