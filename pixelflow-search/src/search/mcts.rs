
#[derive(Clone, Debug)]
pub struct MctsNode<S, M> {
    pub state: S,
    pub action: Option<M>,
    pub children: Vec<MctsNode<S, M>>,
    untried_actions: Vec<M>,
    all_actions: Vec<M>,
    pub visits: u32,
    pub value: f64,
}

impl<S: Clone, M: Clone> MctsNode<S, M> {
    pub fn new(state: S, all_actions: Vec<M>) -> Self {
        let untried_actions = all_actions.clone();
        Self {
            state,
            action: None,
            children: Vec::new(),
            untried_actions,
            all_actions,
            visits: 0,
            value: 0.0,
        }
    }

    /// Expand node by trying one untried action
    /// Returns reference to the new child node
    pub fn expand<F>(&mut self, apply_action: F) -> &mut MctsNode<S, M>
    where
        F: Fn(&S, &M) -> S,
    {
        let action = self.untried_actions.pop().unwrap();
        let new_state = apply_action(&self.state, &action);
        let mut new_node = MctsNode::new(new_state, self.all_actions.clone());
        new_node.action = Some(action);
        self.children.push(new_node);
        self.children.last_mut().unwrap()
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn is_fully_expanded(&self) -> bool {
        self.untried_actions.is_empty()
    }
}
#[derive(Clone, Debug)]
pub struct MctsTree<S, M> {
    pub root: MctsNode<S, M>,
}

impl<S: Clone, M: Clone> MctsTree<S, M> {
    pub fn new(root: MctsNode<S, M>) -> Self {
        Self { root }
    }

    /// Backpropagate value through the path from leaf to root
    /// path: indices of children taken at each level
    pub fn backpropagate(&mut self, path: &[usize], value: f64) {
        let mut current_node = &mut self.root;
        current_node.visits += 1;
        current_node.value += value;
        for &index in path {
            current_node = &mut current_node.children[index];
            current_node.visits += 1;
            current_node.value += value;
        }
    }

    /// Run one MCTS iteration: select -> expand -> simulate -> backpropagate
    pub fn iterate<F, V>(&mut self, apply_action: F, evaluate: V)
    where
        F: Fn(&S, &M) -> S + Copy,
        V: Fn(&S) -> f64,
    {
        let mut path = Vec::new();
        let mut node = &mut self.root;

        // --- Selection ---
        while node.is_fully_expanded() && !node.is_leaf() {
            let best_child_index = (0..node.children.len())
                .max_by(|&a, &b| {
                    let child_a = &node.children[a];
                    let child_b = &node.children[b];
                    let score = |child: &MctsNode<S, M>| {
                        if child.visits == 0 {
                            std::f64::INFINITY
                        } else {
                            child.value / child.visits as f64
                                + (2.0f64.sqrt())
                                    * ((node.visits as f64).ln() / child.visits as f64).sqrt()
                        }
                    };
                    score(child_a).partial_cmp(&score(child_b)).unwrap()
                })
                .unwrap();
            path.push(best_child_index);
            node = &mut node.children[best_child_index];
        }

        let value = if !node.is_fully_expanded() {
            let new_node_index = node.children.len();
            let new_node = node.expand(apply_action);
            path.push(new_node_index);
            evaluate(&new_node.state)
        } else {
            evaluate(&node.state)
        };

        self.backpropagate(&path, value);
    }

    /// Return the best action (most visited child of root)
    pub fn best_action(&self) -> Option<M> {
        self.root
            .children
            .iter()
            .max_by_key(|c| c.visits)
            .and_then(|c| c.action.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestState(i32);

    #[derive(Clone, Debug, PartialEq)]
    struct TestAction(i32);

    fn apply_action(state: &TestState, action: &TestAction) -> TestState {
        TestState(state.0 + action.0)
    }

    fn evaluate(state: &TestState) -> f64 {
        state.0 as f64
    }

    #[test]
    fn test_expansion_adds_one_child() {
        let mut node = MctsNode::new(TestState(0), vec![TestAction(1), TestAction(2)]);
        assert_eq!(node.children.len(), 0);
        node.expand(apply_action);
        assert_eq!(node.children.len(), 1);
        assert_eq!(node.children[0].state, TestState(2));
    }

    #[test]
    fn test_backpropagation_increments_visits() {
        let root = MctsNode::new(TestState(0), vec![TestAction(1)]);
        let mut tree = MctsTree::new(root);
        tree.root.expand(apply_action);
        tree.backpropagate(&[0], 1.0);
        assert_eq!(tree.root.visits, 1);
        assert_eq!(tree.root.children[0].visits, 1);
    }

    #[test]
    fn test_best_action_returns_most_visited() {
        let root = MctsNode::new(
            TestState(0),
            vec![TestAction(1), TestAction(2), TestAction(3)],
        );
        let mut tree = MctsTree::new(root);

        // Expand three times
        tree.root.expand(apply_action);
        tree.root.expand(apply_action);
        tree.root.expand(apply_action);

        // Backpropagate to make one action more visited
        tree.backpropagate(&[0], 1.0);
        tree.backpropagate(&[1], 1.0);
        tree.backpropagate(&[1], 1.0);
        tree.backpropagate(&[2], 1.0);

        assert_eq!(tree.best_action(), Some(TestAction(2)));
    }

    #[test]
    fn test_mcts_iteration() {
        let root = MctsNode::new(TestState(0), vec![TestAction(1), TestAction(2)]);
        let mut tree = MctsTree::new(root);
        tree.iterate(apply_action, evaluate);
        assert_eq!(tree.root.visits, 1);
        assert_eq!(tree.root.children.len(), 1);
        assert_eq!(tree.root.children[0].visits, 1);
    }
}
