//! Monte Carlo Tree Search (MCTS) for guided rewrite exploration.
//!
//! MCTS balances exploration of new rewrites with exploitation of known-good paths.
//! Combined with NNUE evaluation, this enables neural-guided optimization search.

/// Configuration for the MCTS algorithm.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Exploration constant `C` in UCB1. Higher = more exploration.
    /// Default: sqrt(2) ≈ 1.414
    pub exploration_constant: f64,
    /// Maximum iterations before returning best action.
    pub max_iterations: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            exploration_constant: core::f64::consts::SQRT_2,
            max_iterations: 1000,
            max_depth: 50,
        }
    }
}

/// A node in the MCTS tree.
#[derive(Clone, Debug)]
pub struct MctsNode<S, M> {
    /// State at this node.
    pub state: S,
    /// Action that led here (None for root).
    pub action: Option<M>,
    /// Visit count N(node).
    pub visits: u32,
    /// Total accumulated value Q(node).
    pub total_value: f64,
    /// Child nodes.
    pub children: Vec<MctsNode<S, M>>,
    /// Actions not yet expanded.
    untried_actions: Vec<M>,
}

impl<S: Clone, M: Clone> MctsNode<S, M> {
    /// Create a new node with given state and available actions.
    pub fn new(state: S, actions: Vec<M>) -> Self {
        Self {
            state,
            action: None,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            untried_actions: actions,
        }
    }

    /// Whether this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Whether all actions have been tried.
    pub fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }

    /// Expand by trying one untried action. Returns the new child.
    pub fn expand<F>(&mut self, apply_action: F, available_actions: &[M]) -> &mut MctsNode<S, M>
    where
        F: Fn(&S, &M) -> S,
    {
        let action = self.untried_actions.pop().expect("no untried actions");
        let new_state = apply_action(&self.state, &action);
        let mut child = MctsNode::new(new_state, available_actions.to_vec());
        child.action = Some(action);
        self.children.push(child);
        self.children.last_mut().unwrap()
    }

    /// Select child with highest UCB1 score.
    pub fn select_child(&self, exploration_constant: f64) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }

        let parent_visits = self.visits;
        self.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = ucb1_score(a.visits, a.total_value, parent_visits, exploration_constant);
                let score_b = ucb1_score(b.visits, b.total_value, parent_visits, exploration_constant);
                score_a.partial_cmp(&score_b).unwrap_or(core::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

/// The MCTS tree.
#[derive(Clone, Debug)]
pub struct MctsTree<S, M> {
    /// Root node.
    pub root: MctsNode<S, M>,
    /// Configuration.
    pub config: MctsConfig,
    /// Available actions (shared across nodes).
    available_actions: Vec<M>,
}

impl<S: Clone, M: Clone> MctsTree<S, M> {
    /// Create a new tree from initial state and available actions.
    pub fn new(state: S, actions: Vec<M>, config: MctsConfig) -> Self {
        Self {
            root: MctsNode::new(state, actions.clone()),
            config,
            available_actions: actions,
        }
    }

    /// Backpropagate value through path from leaf to root.
    pub fn backpropagate(&mut self, path: &[usize], value: f64) {
        self.root.visits += 1;
        self.root.total_value += value;

        let mut node = &mut self.root;
        for &idx in path {
            node = &mut node.children[idx];
            node.visits += 1;
            node.total_value += value;
        }
    }

    /// Run one MCTS iteration: select → expand → simulate → backpropagate.
    pub fn iterate<F, V>(&mut self, apply_action: F, evaluate: V)
    where
        F: Fn(&S, &M) -> S,
        V: Fn(&S) -> f64,
    {
        let exploration_c = self.config.exploration_constant;
        let actions = self.available_actions.clone();
        let mut path = Vec::new();

        // Selection: descend tree using UCB1
        {
            let mut node = &mut self.root;
            while node.is_fully_expanded() && !node.is_leaf() {
                if let Some(idx) = node.select_child(exploration_c) {
                    path.push(idx);
                    node = &mut node.children[idx];
                } else {
                    break;
                }
            }
        }

        // Navigate to selected node and expand/evaluate
        let value = {
            let mut node = &mut self.root;
            for &idx in &path {
                node = &mut node.children[idx];
            }

            if !node.is_fully_expanded() {
                let child_idx = node.children.len();
                let child = node.expand(&apply_action, &actions);
                path.push(child_idx);
                evaluate(&child.state)
            } else {
                evaluate(&node.state)
            }
        };

        // Backpropagation
        self.backpropagate(&path, value);
    }

    /// Run multiple iterations.
    pub fn search<F, V>(&mut self, apply_action: F, evaluate: V)
    where
        F: Fn(&S, &M) -> S + Copy,
        V: Fn(&S) -> f64 + Copy,
    {
        for _ in 0..self.config.max_iterations {
            self.iterate(apply_action, evaluate);
        }
    }

    /// Return the best action (most visited child of root).
    pub fn best_action(&self) -> Option<M> {
        self.root
            .children
            .iter()
            .max_by_key(|c| c.visits)
            .and_then(|c| c.action.clone())
    }
}

/// Calculate UCB1 score for node selection.
///
/// UCB1 = Q(node)/N(node) + C * sqrt(ln(N(parent)) / N(node))
///
/// Returns infinity for unvisited nodes to ensure they're selected first.
#[inline]
pub fn ucb1_score(
    node_visits: u32,
    node_value: f64,
    parent_visits: u32,
    exploration_constant: f64,
) -> f64 {
    if node_visits == 0 {
        return f64::INFINITY;
    }

    let exploitation = node_value / node_visits as f64;
    let exploration =
        exploration_constant * ((parent_visits as f64).ln() / node_visits as f64).sqrt();

    exploitation + exploration
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestState(i32);

    #[derive(Clone, Debug, PartialEq)]
    struct TestAction(i32);

    fn apply(state: &TestState, action: &TestAction) -> TestState {
        TestState(state.0 + action.0)
    }

    fn evaluate(state: &TestState) -> f64 {
        state.0 as f64
    }

    #[test]
    fn test_ucb1_unvisited_is_infinity() {
        let score = ucb1_score(0, 0.0, 10, 1.414);
        assert!(score.is_infinite());
    }

    #[test]
    fn test_ucb1_higher_visits_lower_exploration() {
        let score_low = ucb1_score(1, 1.0, 100, 1.414);
        let score_high = ucb1_score(10, 10.0, 100, 1.414);
        // Same average value (1.0), but higher visits = lower exploration bonus
        assert!(score_low > score_high);
    }

    #[test]
    fn test_expansion_adds_one_child() {
        let actions = vec![TestAction(1), TestAction(2)];
        let mut node = MctsNode::new(TestState(0), actions.clone());
        assert_eq!(node.children.len(), 0);
        node.expand(apply, &actions);
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn test_backpropagation_increments_visits() {
        let actions = vec![TestAction(1)];
        let mut tree = MctsTree::new(TestState(0), actions.clone(), MctsConfig::default());
        tree.root.expand(apply, &actions);
        tree.backpropagate(&[0], 1.0);
        assert_eq!(tree.root.visits, 1);
        assert_eq!(tree.root.children[0].visits, 1);
    }

    #[test]
    fn test_best_action_returns_most_visited() {
        let actions = vec![TestAction(1), TestAction(2), TestAction(3)];
        let mut tree = MctsTree::new(TestState(0), actions.clone(), MctsConfig::default());

        // Expand all three
        tree.root.expand(apply, &actions);
        tree.root.expand(apply, &actions);
        tree.root.expand(apply, &actions);

        // Make middle child most visited
        tree.backpropagate(&[0], 1.0);
        tree.backpropagate(&[1], 1.0);
        tree.backpropagate(&[1], 1.0);
        tree.backpropagate(&[2], 1.0);

        assert_eq!(tree.best_action(), Some(TestAction(2)));
    }

    #[test]
    fn test_iterate_expands_and_backprops() {
        let actions = vec![TestAction(1), TestAction(2)];
        let mut tree = MctsTree::new(TestState(0), actions, MctsConfig::default());
        tree.iterate(apply, evaluate);
        assert_eq!(tree.root.visits, 1);
        assert_eq!(tree.root.children.len(), 1);
        assert_eq!(tree.root.children[0].visits, 1);
    }

    #[test]
    fn test_search_runs_iterations() {
        let actions = vec![TestAction(1), TestAction(-1)];
        let mut config = MctsConfig::default();
        config.max_iterations = 100;
        let mut tree = MctsTree::new(TestState(0), actions, config);
        tree.search(apply, evaluate);
        assert!(tree.root.visits >= 100);
    }
}
