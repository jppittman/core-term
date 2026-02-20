//! MCTS tree node with visit counts and Q-values.
//!
//! Each node represents a state in the search tree. Children are created
//! lazily when the node is expanded.

extern crate alloc;

use alloc::vec::Vec;
use super::state::MctsAction;

/// Reference to an MCTS node (index into arena).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeRef(pub usize);

impl NodeRef {
    /// Get the underlying index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// MCTS tree node.
///
/// Stores statistics for UCB selection and policy improvement.
#[derive(Clone)]
pub struct MctsNode {
    /// Action that led to this node (None for root).
    pub action: Option<MctsAction>,

    /// Parent node reference (None for root).
    pub parent: Option<NodeRef>,

    /// Child node references.
    pub children: Vec<NodeRef>,

    /// Visit count N(s).
    pub visit_count: u32,

    /// Total value accumulated W(s).
    /// Q(s) = W(s) / N(s)
    pub total_value: f32,

    /// Prior probability P(s, a) from policy network.
    pub prior: f32,

    /// Whether this node has been expanded (children created).
    pub is_expanded: bool,

    /// Whether this is a terminal state.
    pub is_terminal: bool,

    /// Depth in the tree (root = 0).
    pub depth: usize,
}

impl MctsNode {
    /// Create a new root node.
    #[must_use]
    pub fn root() -> Self {
        Self {
            action: None,
            parent: None,
            children: Vec::new(),
            visit_count: 0,
            total_value: 0.0,
            prior: 1.0,
            is_expanded: false,
            is_terminal: false,
            depth: 0,
        }
    }

    /// Create a child node.
    #[must_use]
    pub fn child(action: MctsAction, parent: NodeRef, prior: f32, depth: usize) -> Self {
        Self {
            action: Some(action),
            parent: Some(parent),
            children: Vec::new(),
            visit_count: 0,
            total_value: 0.0,
            prior,
            is_expanded: false,
            is_terminal: false,
            depth,
        }
    }

    /// Get Q-value (mean value).
    #[must_use]
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    /// Compute UCB score for action selection.
    ///
    /// UCB(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
    ///
    /// # Arguments
    /// * `parent_visits` - N(parent), the visit count of the parent node
    /// * `c_puct` - Exploration constant (typically 1.0-2.0)
    #[must_use]
    pub fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let exploration = c_puct * self.prior * libm::sqrtf(parent_visits as f32)
            / (1.0 + self.visit_count as f32);

        self.q_value() + exploration
    }

    /// Update node statistics after a simulation.
    ///
    /// # Arguments
    /// * `value` - The value observed from this simulation
    pub fn update(&mut self, value: f32) {
        self.visit_count += 1;
        self.total_value += value;
    }

    /// Check if this node should be expanded.
    #[must_use]
    pub fn should_expand(&self) -> bool {
        !self.is_expanded && !self.is_terminal && self.visit_count > 0
    }

    /// Get the best child by visit count (for move selection).
    #[must_use]
    pub fn best_child_by_visits<'a>(&self, arena: &'a MctsArena) -> Option<&'a MctsNode> {
        self.children
            .iter()
            .map(|&ref_| arena.get(ref_))
            .max_by_key(|node| node.visit_count)
    }

    /// Get the best child by Q-value (for greedy selection).
    #[must_use]
    pub fn best_child_by_value<'a>(&self, arena: &'a MctsArena) -> Option<&'a MctsNode> {
        self.children
            .iter()
            .map(|&ref_| arena.get(ref_))
            .max_by(|a, b| a.q_value().partial_cmp(&b.q_value()).unwrap_or(core::cmp::Ordering::Equal))
    }

    /// Get visit distribution over children (for policy target).
    #[must_use]
    pub fn visit_distribution(&self, arena: &MctsArena) -> Vec<(MctsAction, f32)> {
        let total: u32 = self.children
            .iter()
            .map(|&ref_| arena.get(ref_).visit_count)
            .sum();

        if total == 0 {
            return Vec::new();
        }

        self.children
            .iter()
            .filter_map(|&ref_| {
                let node = arena.get(ref_);
                node.action.map(|action| {
                    (action, node.visit_count as f32 / total as f32)
                })
            })
            .collect()
    }
}

/// Arena-based storage for MCTS nodes.
///
/// Nodes are stored in a vector and referenced by index.
/// This avoids heap allocation overhead for each node.
pub struct MctsArena {
    nodes: Vec<MctsNode>,
}

impl MctsArena {
    /// Create a new arena with a root node.
    #[must_use]
    pub fn new() -> Self {
        let mut arena = Self { nodes: Vec::new() };
        arena.nodes.push(MctsNode::root());
        arena
    }

    /// Create an arena with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut arena = Self {
            nodes: Vec::with_capacity(capacity),
        };
        arena.nodes.push(MctsNode::root());
        arena
    }

    /// Get root node reference.
    #[must_use]
    pub fn root(&self) -> NodeRef {
        NodeRef(0)
    }

    /// Get a node by reference.
    #[must_use]
    pub fn get(&self, node_ref: NodeRef) -> &MctsNode {
        &self.nodes[node_ref.0]
    }

    /// Get a mutable node by reference.
    pub fn get_mut(&mut self, node_ref: NodeRef) -> &mut MctsNode {
        &mut self.nodes[node_ref.0]
    }

    /// Add a child node and return its reference.
    pub fn add_child(
        &mut self,
        parent: NodeRef,
        action: MctsAction,
        prior: f32,
    ) -> NodeRef {
        let depth = self.get(parent).depth + 1;
        let child = MctsNode::child(action, parent, prior, depth);
        let child_ref = NodeRef(self.nodes.len());
        self.nodes.push(child);
        self.get_mut(parent).children.push(child_ref);
        child_ref
    }

    /// Number of nodes in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if arena is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the arena and reset with a fresh root.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(MctsNode::root());
    }

    /// Select best child using UCB.
    ///
    /// Returns the child with highest UCB score.
    #[must_use]
    pub fn select_child_ucb(&self, parent: NodeRef, c_puct: f32) -> Option<NodeRef> {
        let parent_node = self.get(parent);
        if parent_node.children.is_empty() {
            return None;
        }

        let parent_visits = parent_node.visit_count;
        parent_node
            .children
            .iter()
            .max_by(|&&a, &&b| {
                let a_score = self.get(a).ucb_score(parent_visits, c_puct);
                let b_score = self.get(b).ucb_score(parent_visits, c_puct);
                a_score.partial_cmp(&b_score).unwrap_or(core::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Backpropagate value through ancestors.
    pub fn backpropagate(&mut self, mut node_ref: NodeRef, value: f32) {
        loop {
            let node = self.get_mut(node_ref);
            node.update(value);

            match node.parent {
                Some(parent) => node_ref = parent,
                None => break,
            }
        }
    }
}

impl Default for MctsArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_value_zero_visits() {
        let node = MctsNode::root();
        assert!((node.q_value() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_q_value_after_updates() {
        let mut node = MctsNode::root();
        node.update(10.0);
        node.update(20.0);
        // Q = (10 + 20) / 2 = 15
        assert!((node.q_value() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_ucb_prefers_unvisited() {
        let action = MctsAction { class_idx: 0, rule_idx: 0 };
        let parent = NodeRef(0);

        // Visited child with zero Q-value (neutral outcome)
        let mut visited = MctsNode::child(action, parent, 0.5, 1);
        visited.update(0.0); // Q = 0, but now visit_count = 1

        // Unvisited child with same prior
        let unvisited = MctsNode::child(action, parent, 0.5, 1);

        let parent_visits = 10;
        let c_puct = 1.0;

        // With same Q-values (both effectively 0), unvisited should have higher UCB
        // because exploration term is P * sqrt(N) / (1 + n):
        // - Visited: 0.5 * sqrt(10) / 2 = 0.79
        // - Unvisited: 0.5 * sqrt(10) / 1 = 1.58
        assert!(
            unvisited.ucb_score(parent_visits, c_puct)
                > visited.ucb_score(parent_visits, c_puct),
            "Unvisited UCB {} should be > visited UCB {}",
            unvisited.ucb_score(parent_visits, c_puct),
            visited.ucb_score(parent_visits, c_puct)
        );
    }

    #[test]
    fn test_arena_add_child() {
        let mut arena = MctsArena::new();
        let root = arena.root();

        let action = MctsAction { class_idx: 0, rule_idx: 1 };
        let child = arena.add_child(root, action, 0.8);

        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(child).parent, Some(root));
        assert_eq!(arena.get(child).action, Some(action));
        assert!((arena.get(child).prior - 0.8).abs() < 1e-6);
        assert_eq!(arena.get(root).children.len(), 1);
    }

    #[test]
    fn test_backpropagate() {
        let mut arena = MctsArena::new();
        let root = arena.root();

        let action1 = MctsAction { class_idx: 0, rule_idx: 0 };
        let child1 = arena.add_child(root, action1, 0.5);

        let action2 = MctsAction { class_idx: 1, rule_idx: 1 };
        let child2 = arena.add_child(child1, action2, 0.3);

        arena.backpropagate(child2, 10.0);

        assert_eq!(arena.get(child2).visit_count, 1);
        assert_eq!(arena.get(child1).visit_count, 1);
        assert_eq!(arena.get(root).visit_count, 1);
        assert!((arena.get(child2).total_value - 10.0).abs() < 1e-6);
        assert!((arena.get(child1).total_value - 10.0).abs() < 1e-6);
        assert!((arena.get(root).total_value - 10.0).abs() < 1e-6);
    }
}
