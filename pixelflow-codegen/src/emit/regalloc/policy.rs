//! What the allocator prefers, apart from how it finds the registers.
//!
//! Two policies, both pure: what giving up a register costs
//! ([`EvictionRank`]), and which roots to carry across a loop ([`carried`]).
//! Neither looks at a schedule or a register, so whatever finds the registers
//! asks them the same questions.

use alloc::vec;
use alloc::vec::Vec;

use crate::emit::FileId;

/// Whether giving up a value's register costs a store, cheapest first.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Store {
    /// The value is defined again where it is read: its definition is
    /// rematerializable, so it has no slot and never will.
    Never,
    /// The slot already holds the value.
    NotNeeded,
    /// The register holds the only copy.
    Needed,
}

/// What giving up a register costs, cheapest first — the order eviction picks
/// its loser in.
///
/// A value defined again where it is read needs no store, nor does one whose
/// slot already holds it; anything else has to be written out. Belady's
/// distance breaks ties *within* a tier and only within one: the traffic an
/// eviction causes outweighs how long it waits to cause it.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct EvictionRank {
    /// Whether losing the register costs a store: not for a value defined
    /// again where it is read, nor once the slot holds it.
    store: Store,
    /// Nearest next read *last*, so the cheapest loser is the one used
    /// farthest out.
    nearest: core::cmp::Reverse<usize>,
}

impl EvictionRank {
    /// The rank of a value whose next read is `distance` instructions away if
    /// it has one.
    pub(super) fn new(store: Store, distance: Option<usize>) -> Self {
        Self {
            store,
            nearest: core::cmp::Reverse(distance.unwrap_or(usize::MAX)),
        }
    }
}

/// The per-file budgets [`carried`] fills, indexed by `FileId as usize`. A
/// carry holds one register of one file from before a loop to its latch;
/// `Flags` is never carried, and the Opmask budget is zero: a predicate live
/// into a loop head is stored there.
pub(super) type Budget = [usize; 4];

/// The vector registers a loop's body keeps for itself when it carries as many
/// as it can: a file's carry budget is its members minus this.
///
/// Seven is the constant the carry pricing was fitted with (escape-hatches,
/// 2026-09-04 and 05, where three principled replacements lost to it): 16
/// vector registers carry 9. What the body needs from the seven is the widest
/// instruction's operands (a `MulAdd` reads three and writes one) and the
/// reloads and copies the allocator inserts beside it.
pub(super) const CARRY_RESERVE: usize = 7;

/// The same for the general file. Five leaves x86's nine members four carries:
/// the context pointer, `out` and `pitch`, which are leased like any other
/// value, and one more; constants are RIP-relative, so there is no pool base.
/// aarch64 keeps the same reserve.
pub(super) const GENERAL_CARRY_RESERVE: usize = 5;

/// A root that could be carried, and what it would cost the scopes it is
/// live across.
pub(super) struct Candidate<R> {
    pub(super) weight: usize,
    pub(super) class: FileId,
    /// The scopes the carry is live across, by index.
    pub(super) live_across: Vec<usize>,
    pub(super) root: R,
}

/// The roots to carry, hottest first: the candidates by weight, then taken
/// greedily while no scope they are live across has more of their class
/// carried than `above_floor` allows.
///
/// `scopes` is how many scopes `live_across` indexes into. The sort is
/// stable, so the order the candidates arrive in breaks ties.
pub(super) fn carried<R>(
    mut candidates: Vec<Candidate<R>>,
    scopes: usize,
    above_floor: Budget,
) -> Vec<R> {
    candidates.sort_by_key(|c| core::cmp::Reverse(c.weight));

    let mut count: Vec<Budget> = vec![[0; 4]; scopes];
    let mut taken = Vec::new();
    for candidate in candidates {
        let class = candidate.class as usize;
        if candidate
            .live_across
            .iter()
            .any(|&s| count[s][class] >= above_floor[class])
        {
            continue;
        }
        for &s in &candidate.live_across {
            count[s][class] += 1;
        }
        taken.push(candidate.root);
    }
    taken
}
