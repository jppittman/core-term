/// Represents the configuration for the Monte Carlo Tree Search algorithm.
///
/// This struct holds parameters that control the behavior of the MCTS search,
/// allowing for tuning of the exploration-exploitation balance and search limits.
pub struct MctsConfig {
    /// The exploration constant, `C` in the UCB1 formula. This value balances
    /// the trade-off between exploring less-visited nodes and exploiting nodes
    /// known to have high values. A higher value encourages more exploration.
    pub exploration_constant: f64,

    /// The maximum number of iterations the MCTS algorithm will run before
    /// returning the best-found action. This limits the total computational
    /// budget for a single search.
    pub max_iterations: usize,

    /// The maximum depth of the search tree. This prevents the tree from growing
    /// excessively deep, which can be important for games or problems with long
    /// sequences of actions.
    pub max_depth: usize,
}

/// Represents a single node in the Monte Carlo Search Tree.
///
/// Each node corresponds to a specific state in the search space and stores
/// statistics about the outcomes of simulations that have passed through it.
pub struct MctsNode<S, M> {
    /// The state of the environment that this node represents.
    pub state: S,

    /// The action that led to this state from its parent node. This is `None`
    /// for the root of the tree, as it is the starting point.
    pub action: Option<M>,

    /// The number of times this node has been visited during the search. This
    /// is `N(node)` in the UCB1 formula.
    pub visits: u32,

    /// The sum of all simulation values that have been backpropagated through
    /// this node. This is `Q(node)` in the UCB1 formula.
    pub total_value: f64,

    /// A list of child nodes, representing states reachable from the current node.
    pub children: Vec<MctsNode<S, M>>,

    /// A list of actions that have not yet been tried from this node. When a
    /// node is expanded, one of these actions is chosen to create a new child node.
    pub untried_actions: Vec<M>,
}

/// Represents the entire Monte Carlo Search Tree.
///
/// This struct is the main entry point for running the MCTS algorithm. It holds
/// the root of the search tree and the configuration that governs the search process.
pub struct MctsTree<S, M> {
    /// The root node of the search tree, representing the initial state from
    /// which the search begins.
    pub root: MctsNode<S, M>,

    /// The configuration settings for the MCTS algorithm.
    pub config: MctsConfig,
}

/// Calculate UCB1 score for node selection.
///
/// The UCB1 formula is used to balance exploration and exploitation in the MCTS algorithm.
/// It is defined as: `Q(node)/N(node) + C * sqrt(ln(N(parent)) / N(node))`.
///
/// # Arguments
///
/// * `node_visits` - The number of times the node has been visited (`N(node)`).
/// * `node_value` - The total value accumulated at the node (`Q(node)`).
/// * `parent_visits` - The number of times the parent node has been visited (`N(parent)`).
/// * `exploration_constant` - The exploration constant `C`.
///
/// # Returns
///
/// The UCB1 score for the node. If the node has not been visited (`node_visits` is 0),
/// it returns `f64::INFINITY` to ensure that unvisited nodes are selected first.
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
    let exploration = exploration_constant
        * ( (parent_visits as f64).ln() / node_visits as f64 ).sqrt();

    exploitation + exploration
}

impl<S, M> MctsNode<S, M> {
    /// Select a child node with the highest UCB1 score.
    ///
    /// This method iterates through all children of the current node, calculates their
    /// UCB1 score, and returns the child with the highest score. This is the selection
    /// step of the MCTS algorithm, where the tree is traversed to find the most
    /// promising node to expand.
    ///
    /// # Arguments
    ///
    /// * `exploration_constant` - The exploration constant `C` used in the UCB1 calculation.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the selected child node. If the current
    /// node has no children, it returns `None`.
    pub fn select_child(&self, exploration_constant: f64) -> Option<&MctsNode<S, M>> {
        self.children
            .iter()
            .max_by(|a, b| {
                let a_score = ucb1_score(a.visits, a.total_value, self.visits, exploration_constant);
                let b_score = ucb1_score(b.visits, b.total_value, self.visits, exploration_constant);
                a_score.partial_cmp(&b_score).unwrap()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucb1_returns_infinity_for_unvisited_nodes() {
        let score = ucb1_score(0, 0.0, 10, 1.414);
        assert_eq!(score, f64::INFINITY, "UCB1 score should be infinite for unvisited nodes");
    }

    #[test]
    fn higher_visit_counts_reduce_exploration_bonus() {
        let score1 = ucb1_score(10, 5.0, 100, 1.414);
        let score2 = ucb1_score(20, 5.0, 100, 1.414);
        assert!(score1 > score2, "Exploration bonus should decrease with more visits");
    }

    #[test]
    fn selection_prefers_high_value_nodes_when_visits_are_equal() {
        let parent = MctsNode {
            state: (),
            action: None,
            visits: 20,
            total_value: 10.0,
            children: vec![
                MctsNode { // Child 1: Lower value
                    state: (),
                    action: Some(1),
                    visits: 10,
                    total_value: 5.0, // exploitation = 0.5
                    children: vec![],
                    untried_actions: vec![],
                },
                MctsNode { // Child 2: Higher value
                    state: (),
                    action: Some(2),
                    visits: 10,
                    total_value: 8.0, // exploitation = 0.8
                    children: vec![],
                    untried_actions: vec![],
                },
            ],
            untried_actions: vec![],
        };

        let selected = parent.select_child(1.414).unwrap();
        assert_eq!(selected.action, Some(2), "Should select child with higher value when visits are equal");
    }

    #[test]
    fn selection_balances_exploration_and_exploitation() {
        let parent = MctsNode {
            state: (),
            action: None,
            visits: 100,
            total_value: 50.0,
            children: vec![
                MctsNode { // Child 1: High value, high visits (less exploration)
                    state: (),
                    action: Some(1),
                    visits: 80,
                    total_value: 70.0, // exploitation = 0.875
                    children: vec![],
                    untried_actions: vec![],
                },
                MctsNode { // Child 2: Lower value, low visits (more exploration)
                    state: (),
                    action: Some(2),
                    visits: 20,
                    total_value: 10.0, // exploitation = 0.5
                    children: vec![],
                    untried_actions: vec![],
                },
            ],
            untried_actions: vec![],
        };

        let selected = parent.select_child(1.414).unwrap();
        assert_eq!(selected.action, Some(2), "Should select child with higher exploration bonus");
    }
}