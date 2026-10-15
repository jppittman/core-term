//! The store's per-id columns: what an id indexes, and who may index it.
//!
//! Every id [`EGraph::add`](super::EGraph::add) mints owns one entry in each
//! column, pushed together by [`Columns::push`] and by nothing else, so the
//! columns are equally long because the type has no other way to grow them.
//!
//! An id is read three ways, and the columns keep the readings apart by the
//! type that indexes them:
//!
//! - **per node**: the id names a node and the slot it was minted into
//!   (`parent`, and `classes`, whose slots only the class a node founded fills
//!   or drains). Indexed by [`EClassId`].
//! - **per representative**: the id is the answer of [`Columns::root`], the
//!   class whose block of the equivalence the facts describe. Indexed by
//!   [`Root`], which only `root` and `root_mut` return, so a fact read at a
//!   merged-away id, where it is silently stale, does not type-check.
//! - **per worklist slot**: the id is where a [`Slot`] was pushed, which
//!   `queued` is keyed by, whatever that id has since been merged into.
//!   Indexed by [`Slot`].

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use pixelflow_ir::Variance;

use super::EClass;
use crate::egraph::node::EClassId;

/// A class that was canonical when it was asked for: the answer of
/// [`Columns::root`] or [`Columns::root_mut`], and nothing else makes one.
///
/// It is a name for the class at that moment. A union that merges the class
/// away leaves its `Root` naming a slot that is no longer a representative,
/// so a `Root` is not held across a union of its class: ask again.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Root(EClassId);

impl Root {
    /// The class's id.
    pub(super) fn id(self) -> EClassId {
        self.0
    }
}

/// A place on the rebuild worklist: the class a union pushed, as it was
/// named then. The class has usually been merged away by the time the slot
/// pops, so [`Columns::root_of`] asks what it is now, and `queued` is cleared
/// by the slot, which is the key a union's guard set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot(EClassId);

impl Slot {
    /// The slot a union pushes for `root`.
    pub(super) fn at(root: Root) -> Self {
        Self(root.0)
    }
}

/// One entry per id, indexed by the type `I` that names what the entry
/// describes. Its only growth is [`Columns::push`].
pub(crate) struct Column<I, T> {
    entries: Vec<T>,
    index: PhantomData<fn(I)>,
}

impl<I, T> Column<I, T> {
    fn push(&mut self, entry: T) {
        self.entries.push(entry);
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn iter(&self) -> core::slice::Iter<'_, T> {
        self.entries.iter()
    }
}

impl<I, T> Default for Column<I, T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            index: PhantomData,
        }
    }
}

impl<I, T: Clone> Clone for Column<I, T> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            index: PhantomData,
        }
    }
}

/// `column[key]` for a column indexed by `$column` and a key of type `$key`
/// whose entry is at `$at(key)`.
macro_rules! index_by {
    ($column:ty, $key:ty, $at:expr) => {
        impl<T> Index<$key> for Column<$column, T> {
            type Output = T;

            fn index(&self, key: $key) -> &T {
                &self.entries[$at(key)]
            }
        }
    };
}

/// `column[key] = ..`, which is only offered for a key that names a class
/// that is canonical or a slot, never for a raw id.
macro_rules! index_mut_by {
    ($column:ty, $key:ty, $at:expr) => {
        impl<T> IndexMut<$key> for Column<$column, T> {
            fn index_mut(&mut self, key: $key) -> &mut T {
                &mut self.entries[$at(key)]
            }
        }
    };
}

// A per-node column reads at any id and, being per-node, at a `Root` too.
index_by!(EClassId, EClassId, |id: EClassId| id.index());
index_by!(EClassId, Root, |root: Root| root.0.index());
index_mut_by!(EClassId, Root, |root: Root| root.0.index());
// A per-representative column reads only at a representative.
index_by!(Root, Root, |root: Root| root.0.index());
index_mut_by!(Root, Root, |root: Root| root.0.index());
// A per-slot column reads only at a slot.
index_by!(Slot, Slot, |slot: Slot| slot.0.index());
index_mut_by!(Slot, Slot, |slot: Slot| slot.0.index());

/// What `add` seeds one id's entries with.
pub(super) struct Birth {
    pub(super) class: EClass,
    pub(super) const_fact: Option<u32>,
    pub(super) variance: Variance,
    /// The mutation counter's value as of this `add`.
    pub(super) stamp: u64,
}

/// The per-id columns and the union-find they share. See the module docs.
#[derive(Clone, Default)]
pub(crate) struct Columns {
    /// `parent[t]` is the next id toward `t`'s representative; a
    /// representative is its own parent.
    parent: Column<EClassId, EClassId>,
    /// The nodes of the class founded at each id, and their tags. A merged-away
    /// class's slot is empty: `union` drains it.
    pub(crate) classes: Column<EClassId, EClass>,
    /// The constant each class is known to equal, as f32 bits — maintained
    /// independently of `EClass::nodes` on purpose.
    /// `rebuild` drains a class's nodes with `mem::take` and only THEN
    /// performs congruence unions, so a guard that scanned the node vector
    /// saw an empty class and waved contradictory merges through at exactly
    /// the moment congruence closure does its work. The fact must outlive the
    /// nodes.
    pub(super) const_fact: Column<Root, Option<u32>>,
    /// Which binders each class may depend on — the per-class variance fact
    /// (docs/archive/plans/2026-09-23-an-integral-is-a-fold.md §1, §3), kept beside
    /// `const_fact` and out of `EClass::nodes` for the same reason: a guard
    /// read mid-`rebuild` must not see a drained class.
    ///
    /// **Law:** `var_fact[C] ⊇ var(⟦C⟧)` — the fact over-approximates the
    /// variance of the one function every member of `C` denotes, so a bit
    /// absent from the fact is a binder the class is provably constant along.
    ///
    /// - **Seeded** in `EGraph::add` by `ENode::variance`, the node's
    ///   transfer function, over its children's facts *as they stand then*. Each child's fact over-approximates the child, and the
    ///   transfer is monotone, so the seed over-approximates the node.
    /// - **Merged** in `EGraph::union` by *intersection*. Both classes denote the same function and both facts
    ///   over-approximate it, so a bit either one lacks is a bit the function
    ///   lacks. A fact only ever shrinks from its seed and never grows up
    ///   from `∅` — the greatest fixpoint, not the least — which is why a
    ///   cycle cannot certify an invariance: a class holding `x` and `x + 0`
    ///   got each member's fact from facts that already existed, never from
    ///   an assumption about itself.
    /// - **Never repaired upward.** A union that shrinks a child's fact does
    ///   not revisit the parents seeded from the larger one. A stale parent
    ///   fact is still a superset, so a rule reading it may miss an
    ///   opportunity but can never fire wrongly; soundness needs no repair.
    pub(super) var_fact: Column<Root, Variance>,
    /// The mutation counter's value as of the most recent time canonical
    /// class `c`'s own node list gained content — set at creation in `add`
    /// and bumped in `union` when `c` survives as the merge's parent.
    pub(super) last_changed: Column<Root, u64>,
    /// Whether the slot has an entry on the worklist awaiting
    /// `rebuild_budgeted`. Guards `union`'s push: several unions landing on
    /// the same parent in one round used to enqueue it once per union, and
    /// `rebuild_budgeted` paid a full canonicalize-and-memo-probe pass over
    /// that class's node vector per redundant entry even though only the
    /// state after the *last* union in the round matters.
    pub(super) queued: Column<Slot, bool>,
}

impl Columns {
    /// The number of ids minted: every column's length.
    pub(super) fn len(&self) -> usize {
        self.parent.len()
    }

    /// Mint the next id, with its entry in every column. The one place a
    /// column grows.
    pub(super) fn push(&mut self, birth: Birth) -> EClassId {
        let id = EClassId::from_index(self.len());
        self.parent.push(id);
        self.classes.push(birth.class);
        self.const_fact.push(birth.const_fact);
        self.var_fact.push(birth.variance);
        self.last_changed.push(birth.stamp);
        self.queued.push(false);
        id
    }

    /// The representative of `id`'s block.
    pub(super) fn root(&self, id: EClassId) -> Root {
        let mut current = id;
        while self.parent[current] != current {
            current = self.parent[current];
        }
        Root(current)
    }

    /// [`Self::root`], compressing the path it walks.
    pub(super) fn root_mut(&mut self, id: EClassId) -> Root {
        let mut current = id;
        let mut path = Vec::new();
        while self.parent[current] != current {
            path.push(current);
            current = self.parent[current];
        }
        for node in path {
            self.parent.entries[node.index()] = current;
        }
        Root(current)
    }

    /// The representative `slot`'s class has now.
    pub(super) fn root_of(&self, slot: Slot) -> Root {
        self.root(slot.0)
    }

    /// Make `child`'s block part of `parent`'s.
    pub(super) fn link(&mut self, child: Root, parent: Root) {
        self.parent.entries[child.0.index()] = parent.0;
    }
}
