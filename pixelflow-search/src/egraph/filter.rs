//! The rules × nodes application filter: `F : M → M'`, run after matching
//! and before application.
//!
//! Each saturation round the e-graph has a sparse match matrix `M` over
//! rules × e-classes; a cell holds the [`RewriteAction`] a rule's match at
//! that class produced — the right-hand side with its bindings substituted.
//! An [`ApplicationFilter`] maps `M` to an `M'` of the same shape and the
//! loop commits `M'`. [`KeepAll`] is `M' = M`, and under it production is
//! byte-identical to the loop before this seam existed — pinned by
//! `pixelflow-pipeline/tests/rules_by_nodes_identity.rs`, not claimed.
//!
//! The matrix arrives **one row at a time**, because that is the order the
//! loop produces it: within a round, rule `r+1` matches against rule `r`'s
//! committed additions (plus an interleaved partial rebuild), so a row cannot
//! be enumerated before the previous row is committed without changing what
//! the row *is*. See docs/plans/2026-09-08-rules-by-nodes-filter.md.
//!
//! A filter may only **drop** cells: [`MatchRow`] exposes the row for reading
//! and a `retain`, and no constructor for an action the rules did not
//! produce. That is what makes law L4 (`egraph/optimizer.rs`) hold for every
//! filter without a per-filter correctness argument — the graph after `F`
//! holds a subset of the equalities the unfiltered graph would.

use alloc::vec::Vec;

use super::graph::EGraph;
use super::node::EClassId;
use super::rewrite::RewriteAction;
use super::rules::RuleId;

/// One row of the match matrix: every match one rule found in one scan,
/// each with its binding, in scan order — matched, not yet committed.
///
/// This *is* the scan's staged action list, moved here for the filter's
/// inspection and moved back for the commit; constructing it allocates
/// nothing.
pub struct MatchRow {
    rule: RuleId,
    rule_idx: usize,
    matches: Vec<(EClassId, RewriteAction)>,
}

impl MatchRow {
    pub(crate) fn new(
        rule_idx: usize,
        rule: RuleId,
        matches: Vec<(EClassId, RewriteAction)>,
    ) -> Self {
        Self {
            rule,
            rule_idx,
            matches,
        }
    }

    /// The rule every cell of this row belongs to.
    #[must_use]
    pub fn rule(&self) -> RuleId {
        self.rule
    }

    /// The rule's index in the graph's rule table.
    #[must_use]
    pub fn rule_index(&self) -> usize {
        self.rule_idx
    }

    /// The row's cells: the class each match was found at, and the action
    /// the rule produced there.
    #[must_use]
    pub fn matches(&self) -> &[(EClassId, RewriteAction)] {
        &self.matches
    }

    /// Keep only the cells `keep` accepts. The one mutation a filter has:
    /// it can shrink a row, never reorder it or add to it.
    pub fn retain(&mut self, keep: impl FnMut(&(EClassId, RewriteAction)) -> bool) {
        self.matches.retain(keep);
    }

    pub(crate) fn into_matches(self) -> Vec<(EClassId, RewriteAction)> {
        self.matches
    }
}

/// `F : M → M'` — decide which of a round's matches the loop commits.
///
/// Called once per rule per round, on that rule's freshly matched row, with
/// the graph exactly as the row's bindings refer to it. Never called on an
/// empty row: `F(∅) = ∅` for every filter, since a filter cannot invent a
/// cell.
pub trait ApplicationFilter {
    /// Shrink `row` to the cells that should be applied.
    fn filter(&mut self, graph: &EGraph, row: &mut MatchRow);
}

/// `M' = M` — the identity on the match matrix: commit everything that
/// matched. Production.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeepAll;

impl ApplicationFilter for KeepAll {
    fn filter(&mut self, _graph: &EGraph, _row: &mut MatchRow) {}
}
