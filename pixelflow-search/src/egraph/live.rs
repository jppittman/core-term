//! Sets of classes, kept as bitmaps.
//!
//! The *live* set is the first: a class is live when it is canonical (a root
//! of the union-find) and holds at least one node, the classes a rule scan
//! visits. Deciding that from the graph costs a `find` per class ever minted,
//! while the answer changes only where a class is minted, merged or drained,
//! and a scan runs once per rule per sweep. So the graph restates liveness at
//! each of those events and a scan reads the set.
//!
//! The second use is the per-kind index: for each [`OpKind`], the classes that
//! have held a node of that kind, which narrows a scan to the classes its rule
//! can root in (`EGraph::candidate_classes`).
//!
//! A set is a bitmap indexed by class, so iterating it yields ids in ascending
//! order, the order [`EGraph::class_ids`](super::EGraph::class_ids) defines
//! and a scan visits them in.

use super::node::EClassId;
use pixelflow_ir::kind::{OpKind, OpMap, OpSet};

const WORD_BITS: usize = u64::BITS as usize;

/// A set of classes. See the module docs.
#[derive(Clone, Debug, Default)]
pub(super) struct ClassSet {
    words: Vec<u64>,
}

impl ClassSet {
    /// Add `id`.
    pub(super) fn insert(&mut self, id: EClassId) {
        let (word, bit) = (id.index() / WORD_BITS, id.index() % WORD_BITS);
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << bit;
    }

    /// Remove `id`. A class never added has nothing to clear.
    pub(super) fn remove(&mut self, id: EClassId) {
        let (word, bit) = (id.index() / WORD_BITS, id.index() % WORD_BITS);
        if let Some(bits) = self.words.get_mut(word) {
            *bits &= !(1 << bit);
        }
    }

    /// Whether `id` is in the set.
    pub(super) fn contains(&self, id: EClassId) -> bool {
        self.word(id.index() / WORD_BITS) >> (id.index() % WORD_BITS) & 1 == 1
    }

    /// The bits of classes `64 * word ..`, none past the end of the set.
    pub(super) fn word(&self, word: usize) -> u64 {
        self.words.get(word).copied().unwrap_or(0)
    }

    /// The classes, in ascending id order.
    pub(super) fn iter(&self) -> impl Iterator<Item = EClassId> + '_ {
        self.iter_within(|_| u64::MAX)
    }

    /// The classes that are also in the `within` of their word, in ascending
    /// id order: `within(w)` is the bits of classes `64 * w ..` to keep. One
    /// AND per word, so a set of 100 000 classes narrowed to a few hundred
    /// costs about 1 600 words rather than 100 000 membership tests.
    pub(super) fn iter_within<'a>(
        &'a self,
        within: impl Fn(usize) -> u64 + 'a,
    ) -> impl Iterator<Item = EClassId> + 'a {
        self.words
            .iter()
            .enumerate()
            .flat_map(move |(word, &bits)| {
                let mut remaining = bits & within(word);
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        return None;
                    }
                    let bit = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    Some(EClassId::from_index(word * WORD_BITS + bit))
                })
            })
    }
}

/// For each [`OpKind`], the set of classes that have held a node of that kind.
///
/// It only grows, so it is a superset of the classes holding the kind *now*:
/// a node that leaves a class (a refused rebuild item retires it) leaves its
/// bit behind, and a scan that reads a stale bit pays one per-node check that
/// rejects it. See `Columns::kind_mask`, whose union it mirrors.
#[derive(Clone, Debug)]
pub(super) struct ClassesByKind {
    sets: OpMap<ClassSet>,
}

impl Default for ClassesByKind {
    fn default() -> Self {
        Self {
            sets: OpMap::from_fn(|_| ClassSet::default()),
        }
    }
}

impl ClassesByKind {
    /// Record that `id` holds a node of `kind`.
    pub(super) fn insert(&mut self, kind: OpKind, id: EClassId) {
        self.sets[kind].insert(id);
    }

    /// Whether `id` has held a node of `kind`.
    pub(super) fn contains(&self, kind: OpKind, id: EClassId) -> bool {
        self.sets[kind].contains(id)
    }

    /// The bits of classes `64 * word ..` that have held a node of any of
    /// `kinds`.
    pub(super) fn word_of_any(&self, kinds: OpSet, word: usize) -> u64 {
        kinds
            .iter()
            .fold(0, |bits, kind| bits | self.sets[kind].word(word))
    }
}
