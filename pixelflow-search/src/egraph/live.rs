//! The live classes of an e-graph, kept as a set.
//!
//! A class is *live* when it is canonical (a root of the union-find) and holds
//! at least one node: the classes a rule scan visits. Deciding that from the
//! graph costs a `find` per class ever minted, while the answer changes only
//! where a class is minted, merged or drained, and a scan runs once per rule
//! per sweep. So the graph restates liveness at each of those events and a
//! scan reads the set.
//!
//! The set is a bitmap indexed by class, so iterating it yields the live ids
//! in ascending order, the order
//! [`EGraph::class_ids`](super::EGraph::class_ids) defines and a scan visits
//! them in.

use super::node::EClassId;

const WORD_BITS: usize = u64::BITS as usize;

/// The set of live classes. See the module docs.
#[derive(Clone, Debug, Default)]
pub(super) struct LiveClasses {
    words: Vec<u64>,
}

impl LiveClasses {
    /// Mark `id` live.
    pub(super) fn insert(&mut self, id: EClassId) {
        let (word, bit) = (id.index() / WORD_BITS, id.index() % WORD_BITS);
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << bit;
    }

    /// Mark `id` not live. A class never marked live has nothing to clear.
    pub(super) fn remove(&mut self, id: EClassId) {
        let (word, bit) = (id.index() / WORD_BITS, id.index() % WORD_BITS);
        if let Some(bits) = self.words.get_mut(word) {
            *bits &= !(1 << bit);
        }
    }

    /// The live classes, in ascending id order.
    pub(super) fn iter(&self) -> impl Iterator<Item = EClassId> + '_ {
        self.words.iter().enumerate().flat_map(|(word, &bits)| {
            let mut remaining = bits;
            std::iter::from_fn(move || {
                if remaining == 0 {
                    return None;
                }
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                Some(EClassId((word * WORD_BITS + bit) as u32))
            })
        })
    }
}
