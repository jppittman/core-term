//! Best-First Search for E-Graph Optimization
//!
//! This is just Dijkstra/A* with NNUE as the heuristic.
//!
//! ## The Graph Search Mental Model
//!
//! | ML/MCTS Name | Graph/Dijkstra Name |
//! |--------------|---------------------|
//! | State | Node in search graph |
//! | Rewrite | Edge |
//! | NNUE | Heuristic (estimated distance to optimal) |
//! | Objective | Minimize cost (runtime of kernel) |
//!
//! ## Why This is Simple
//!
//! 1. **No Policy Head**: Just score the result of each move
//! 2. **No Rollouts**: NNUE already knows that `a*(b+c)` beats `a*b+a*c`
//! 3. **Easy Debugging**: Print the path, see why NNUE said cost was X
//!
//! ## Chess Time Management
//!
//! Inspired by Stockfish's time management, the search dynamically allocates
//! time based on:
//! - Expression complexity (larger kernels get more time)
//! - E-graph growth rate (exploding graphs trigger early stop)
//! - Time pressure (past 80% of budget → pure exploitation)
//!
//! Key metrics passed to the evaluator:
//! - `time_pressure()`: 0.0 (fresh) to 1.0 (out of time)
//! - `size_pressure()`: 0.0 (small) to 1.0 (at limit)
//! - `in_time_trouble()`: true when past 80% of time budget
//! - `in_size_trouble()`: true when past 80% of size limit

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use super::{EGraph, EClassId, ExprTree, CostModel, Rewrite};
use super::cost::CostFunction;

// ============================================================================
// Search State Context (passed to evaluator)
// ============================================================================

/// Search state context passed to the NNUE evaluator.
///
/// This provides all the information needed for feature extraction:
/// - The expression tree itself (for structural features)
/// - Search progress metrics (for temporal features)
/// - Time pressure features (for chess-style time management)
///
/// The evaluator can use these to learn patterns like:
/// - "Early in search, explore more aggressively"
/// - "When frontier is large, prefer simpler trees"
/// - "When improvement has stalled, try different rewrites"
/// - "When in time trouble, prefer low-cost trees"
#[derive(Clone, Debug)]
pub struct BestFirstContext<'a> {
    /// The expression tree being evaluated.
    pub tree: &'a ExprTree,
    /// Cost of this tree (ground truth from CostModel).
    pub tree_cost: usize,
    /// Depth in the search tree (0 = root).
    pub depth: usize,
    /// Current best cost found anywhere in search.
    pub best_cost: usize,
    /// Number of expansions performed so far.
    pub expansions: usize,
    /// Number of states in the frontier (open set).
    pub frontier_size: usize,
    /// Number of visited states (closed set).
    pub visited_count: usize,
    /// Initial cost before any optimization.
    pub initial_cost: usize,

    // Time features (chess-style time management)
    /// Time elapsed since search started.
    pub time_elapsed: Duration,
    /// Total time budget for this kernel.
    pub time_budget: Duration,
    /// Size budget (max classes allowed).
    pub max_classes: usize,
    /// Current e-graph size.
    pub current_classes: usize,
}

impl<'a> BestFirstContext<'a> {
    /// Improvement ratio: how much better is current best vs initial?
    /// Returns 0.0 (no improvement) to 1.0 (perfect optimization).
    pub fn improvement_ratio(&self) -> f64 {
        if self.initial_cost == 0 {
            0.0
        } else {
            1.0 - (self.best_cost as f64 / self.initial_cost as f64)
        }
    }

    /// Search progress: fraction of max_expansions used.
    pub fn search_progress(&self, max_expansions: usize) -> f64 {
        self.expansions as f64 / max_expansions.max(1) as f64
    }

    /// Frontier pressure: how crowded is the open set?
    pub fn frontier_pressure(&self, max_queue_size: usize) -> f64 {
        self.frontier_size as f64 / max_queue_size.max(1) as f64
    }

    // Time pressure features (for NNUE to learn from)

    /// Time fraction used: 0.0 (fresh) to 1.0 (out of time).
    ///
    /// This is the primary time pressure signal. NNUE can learn to
    /// behave differently based on how much budget remains.
    pub fn time_pressure(&self) -> f64 {
        self.time_elapsed.as_secs_f64() / self.time_budget.as_secs_f64().max(0.001)
    }

    /// Size pressure: 0.0 (small) to 1.0 (at limit).
    ///
    /// E-graphs can explode exponentially. This signal indicates
    /// how close we are to the memory limit.
    pub fn size_pressure(&self) -> f64 {
        self.current_classes as f64 / self.max_classes.max(1) as f64
    }

    /// In time trouble? (past 80% of budget)
    ///
    /// When true, the search should switch to pure exploitation:
    /// - Skip exploration (ε=0)
    /// - Prefer known-good rewrites
    /// - Don't try risky optimizations
    pub fn in_time_trouble(&self) -> bool {
        self.time_pressure() > 0.8
    }

    /// In size trouble? (past 80% of limit)
    ///
    /// When true, the e-graph is getting large and we should
    /// stop expanding to avoid memory issues.
    pub fn in_size_trouble(&self) -> bool {
        self.size_pressure() > 0.8
    }

    /// Combined pressure: max of time and size pressure.
    ///
    /// Useful for heuristics that want a single "urgency" signal.
    pub fn combined_pressure(&self) -> f64 {
        self.time_pressure().max(self.size_pressure())
    }
}

/// A search state: an e-graph at some point in the search.
#[derive(Clone)]
struct SearchState {
    /// The e-graph state (cloneable via Arc-shared rules)
    egraph: EGraph,
    /// Root e-class being optimized
    root: EClassId,
    /// Current best extracted tree
    best_tree: ExprTree,
    /// Current best cost (ground truth from CostModel)
    best_cost: usize,
    /// NNUE-predicted potential (lower = more promising)
    priority: i64,
    /// Depth in search tree (for tie-breaking)
    depth: usize,
    /// Hash for cycle detection
    state_hash: u64,
}

impl SearchState {
    /// Create initial state from an expression tree with rules.
    fn new(tree: &ExprTree, costs: &CostModel, rules: Vec<Box<dyn Rewrite>>) -> Self {
        let mut egraph = EGraph::with_rules(rules);
        let root = egraph.add_expr(tree);
        let (best_tree, best_cost) = egraph.extract_best(root, costs);
        let state_hash = Self::compute_hash(&egraph, root);

        Self {
            egraph,
            root,
            best_tree,
            best_cost,
            priority: best_cost as i64,
            depth: 0,
            state_hash,
        }
    }

    /// Expand this state by applying all rules once.
    fn expand<C: CostFunction>(&self, costs: &C) -> Option<SearchState> {
        let mut child_egraph = self.egraph.clone();
        let changes = child_egraph.apply_rules_once();

        if changes == 0 {
            return None; // No new states reachable
        }

        let (tree, cost) = child_egraph.extract_best(self.root, costs);
        let state_hash = Self::compute_hash(&child_egraph, self.root);

        Some(SearchState {
            egraph: child_egraph,
            root: self.root,
            best_tree: tree,
            best_cost: cost,
            priority: cost as i64, // Will be overwritten by evaluator
            depth: self.depth + 1,
            state_hash,
        })
    }

    /// Compute a hash for cycle detection.
    ///
    /// Uses the number of classes and nodes as a rough fingerprint.
    /// More sophisticated hashing could use the actual graph structure.
    fn compute_hash(egraph: &EGraph, root: EClassId) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        egraph.num_classes().hash(&mut hasher);
        egraph.node_count().hash(&mut hasher);
        root.hash(&mut hasher);
        hasher.finish()
    }
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.depth == other.depth
    }
}

impl Eq for SearchState {}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority = better, so reverse the comparison
        // Tie-break by depth (prefer shallower for faster solutions)
        other.priority.cmp(&self.priority)
            .then_with(|| other.depth.cmp(&self.depth))
    }
}

/// Configuration for best-first search.
///
/// ## Chess-Style Time Control
///
/// Like chess engines, we use multiple types of limits:
/// - **Soft timeout**: Reduce exploration after this (switch to ε=0)
/// - **Hard timeout**: Stop immediately and return best-so-far
/// - **Size limits**: Prevent memory explosion from e-graph growth
///
/// ## Presets
///
/// - `blitz()`: Fast compilation, minimal optimization (5-15ms)
/// - `rapid()`: Balanced speed/quality (20-50ms, default)
/// - `classical()`: Thorough optimization for critical code (100-200ms)
#[derive(Clone, Debug)]
pub struct BestFirstConfig {
    /// Maximum states to expand before stopping.
    pub max_expansions: usize,
    /// Maximum queue size (prune worst if exceeded).
    pub max_queue_size: usize,
    /// Epsilon for ε-greedy exploration (0.0 = pure greedy, 1.0 = random).
    pub epsilon: f64,
    /// Node count threshold below which we just saturate.
    pub saturation_threshold: usize,
    /// Random seed for reproducibility.
    pub seed: u64,

    // Time control (chess-style)
    /// Soft time limit - reduce exploration after this.
    pub soft_timeout: Duration,
    /// Hard time limit - stop immediately.
    pub hard_timeout: Duration,
    /// Maximum e-graph classes (memory limit).
    pub max_classes: usize,
    /// Maximum e-graph nodes (memory limit).
    pub max_nodes: usize,

    // Superguide: periodic deep exploration
    /// Every N expansions, do a "superguide" burst with higher budget.
    /// None = disabled, Some(20) = every 20 expansions.
    pub superguide_interval: Option<usize>,
    /// Multiplier for superguide burst (e.g., 10 = 10x normal budget).
    pub superguide_multiplier: usize,
}

impl Default for BestFirstConfig {
    fn default() -> Self {
        Self::rapid()
    }
}

impl BestFirstConfig {
    /// Blitz: fast compilation for quick iteration.
    ///
    /// Use for small expressions or when compilation speed matters.
    /// - 50ms soft, 100ms hard timeout
    /// - Small e-graph limits (500 classes)
    /// - 500 max expansions
    ///
    /// Even "blitz" gives enough time for essential optimizations like
    /// identity elimination and FMA fusion.
    pub fn blitz() -> Self {
        Self {
            max_expansions: 500,
            max_queue_size: 1000,
            epsilon: 0.1,
            saturation_threshold: 100,
            seed: 42,
            soft_timeout: Duration::from_millis(50),
            hard_timeout: Duration::from_millis(100),
            max_classes: 500,
            max_nodes: 2_500,
            superguide_interval: None,
            superguide_multiplier: 10,
        }
    }

    /// Rapid: balanced speed and quality (default).
    ///
    /// Use for normal compilation workflow.
    /// - 100ms soft, 200ms hard timeout
    /// - Medium e-graph limits (5000 classes)
    /// - 1000 max expansions
    pub fn rapid() -> Self {
        Self {
            max_expansions: 1000,
            max_queue_size: 10000,
            epsilon: 0.1,
            saturation_threshold: 100,
            seed: 42,
            soft_timeout: Duration::from_millis(100),
            hard_timeout: Duration::from_millis(200),
            max_classes: 5_000,
            max_nodes: 25_000,
            superguide_interval: None,
            superguide_multiplier: 10,
        }
    }

    /// Classical: thorough optimization for critical code.
    ///
    /// Use for release builds or performance-critical kernels.
    /// - 500ms soft, 1000ms hard timeout
    /// - Large e-graph limits (20000 classes)
    /// - 5000 max expansions
    pub fn classical() -> Self {
        Self {
            max_expansions: 5000,
            max_queue_size: 50000,
            epsilon: 0.1,
            saturation_threshold: 100,
            seed: 42,
            soft_timeout: Duration::from_millis(500),
            hard_timeout: Duration::from_millis(1000),
            max_classes: 20_000,
            max_nodes: 100_000,
            superguide_interval: None,
            superguide_multiplier: 10,
        }
    }

    /// Set epsilon for exploration.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set maximum expansions.
    pub fn with_max_expansions(mut self, max: usize) -> Self {
        self.max_expansions = max;
        self
    }

    /// Set saturation threshold.
    pub fn with_saturation_threshold(mut self, threshold: usize) -> Self {
        self.saturation_threshold = threshold;
        self
    }

    /// Set soft timeout.
    pub fn with_soft_timeout(mut self, timeout: Duration) -> Self {
        self.soft_timeout = timeout;
        self
    }

    /// Set hard timeout.
    pub fn with_hard_timeout(mut self, timeout: Duration) -> Self {
        self.hard_timeout = timeout;
        self
    }

    /// Set maximum e-graph classes.
    pub fn with_max_classes(mut self, max: usize) -> Self {
        self.max_classes = max;
        self
    }

    /// Set maximum e-graph nodes.
    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = max;
        self
    }

    /// Training mode: higher exploration.
    pub fn training_mode(mut self) -> Self {
        self.epsilon = 0.2;
        self
    }

    /// Inference mode: pure greedy.
    pub fn inference_mode(mut self) -> Self {
        self.epsilon = 0.0;
        self
    }

    /// Enable superguide: periodic deep exploration bursts.
    ///
    /// Every `interval` expansions, the search will do a "superguide" burst
    /// where it expands `multiplier` times more states than normal.
    ///
    /// This helps escape local minima by occasionally doing deeper exploration.
    ///
    /// ## Example
    ///
    /// ```ignore
    /// // Every 20 expansions, do 10x more exploration
    /// let config = BestFirstConfig::rapid()
    ///     .with_superguide(20, 10);
    /// ```
    pub fn with_superguide(mut self, interval: usize, multiplier: usize) -> Self {
        self.superguide_interval = Some(interval);
        self.superguide_multiplier = multiplier;
        self
    }
}

/// Result of best-first search.
#[derive(Clone, Debug)]
pub struct BestFirstResult {
    /// The best expression tree found.
    pub best_tree: ExprTree,
    /// Cost of the best tree.
    pub best_cost: usize,
    /// Number of states expanded.
    pub expansions: usize,
    /// Whether saturation was used (for small kernels).
    pub used_saturation: bool,
    /// Initial cost before optimization.
    pub initial_cost: usize,
    /// Time spent searching.
    pub elapsed: Duration,
    /// Why the search stopped.
    pub stop_reason: StopReason,
}

/// Reason the search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// E-graph saturated (optimal result).
    Saturated,
    /// Maximum expansions reached.
    MaxExpansions,
    /// Hard timeout reached.
    HardTimeout,
    /// E-graph size limit reached.
    SizeLimit,
    /// Open set exhausted (no more rewrites).
    Exhausted,
}

// ============================================================================
// Trajectory Recording for Search Head Training
// ============================================================================

/// A single step in a search trajectory.
///
/// Records the state that was expanded and the best cost at that moment.
#[derive(Clone, Debug)]
pub struct TrajectoryStep {
    /// The expression tree that was in the heap.
    pub tree: ExprTree,
    /// Cost of this tree.
    pub tree_cost: usize,
    /// Best cost found so far at this step.
    pub best_cost_so_far: usize,
    /// Expansion number when this was visited.
    pub expansion: usize,
    /// Number of e-graph classes at this point.
    pub num_classes: usize,
}

/// Full search trajectory for training.
///
/// Records all states visited during search, enabling training signal
/// computation after we know the final (or target) cost.
#[derive(Clone, Debug)]
pub struct SearchTrajectory {
    /// Initial cost before any optimization.
    pub initial_cost: usize,
    /// Steps in order of expansion.
    pub steps: Vec<TrajectoryStep>,
    /// Final best cost achieved.
    pub final_cost: usize,
    /// Why the search stopped.
    pub stop_reason: StopReason,
}

impl SearchTrajectory {
    /// Label each step with the training target.
    ///
    /// The target is the final achievable cost (from more compute or saturation).
    /// Returns (tree, target_cost) pairs for training.
    pub fn labeled_samples(&self, target_cost: usize) -> Vec<(ExprTree, usize)> {
        self.steps
            .iter()
            .map(|step| (step.tree.clone(), target_cost))
            .collect()
    }

    /// Label with distance-to-target at each step.
    ///
    /// Returns (tree, gap) where gap = best_cost_so_far - target_cost.
    /// Useful for training "how much improvement is left from here?"
    pub fn labeled_gaps(&self, target_cost: usize) -> Vec<(ExprTree, i64)> {
        self.steps
            .iter()
            .map(|step| {
                let gap = step.best_cost_so_far as i64 - target_cost as i64;
                (step.tree.clone(), gap)
            })
            .collect()
    }
}

/// Simple LCG random number generator.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn gen_f64(&mut self) -> f64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn gen_range(&mut self, max: usize) -> usize {
        (self.gen_f64() * max as f64) as usize
    }
}

/// Best-first search for e-graph optimization.
///
/// This is Dijkstra/A* where NNUE provides the heuristic.
/// Much simpler than MCTS - just a priority queue.
pub struct BestFirstPlanner {
    /// Open set: states to explore, ordered by priority (lower = better)
    open_set: BinaryHeap<SearchState>,
    /// Closed set: visited state hashes (cycle detection)
    visited: HashSet<u64>,
    /// Best state found so far
    best_found: SearchState,
    /// Configuration
    config: BestFirstConfig,
    /// Cost model for extraction
    costs: CostModel,
    /// RNG for epsilon-greedy
    rng: Rng,
    /// Number of expansions performed
    expansions: usize,
    /// Initial cost (for computing improvement)
    initial_cost: usize,
}

impl BestFirstPlanner {
    /// Create a new planner from an expression tree with rules.
    ///
    /// Rules are application-defined. Use `standard_rules()` from pixelflow-compiler
    /// for the canonical PixelFlow rule set.
    pub fn from_tree_with_rules(
        tree: &ExprTree,
        config: BestFirstConfig,
        rules: Vec<Box<dyn Rewrite>>,
    ) -> Self {
        let costs = CostModel::default();
        let initial = SearchState::new(tree, &costs, rules);
        let initial_cost = initial.best_cost;
        let seed = config.seed;

        let mut open_set = BinaryHeap::new();
        let mut visited = HashSet::new();

        visited.insert(initial.state_hash);
        open_set.push(initial.clone());

        Self {
            open_set,
            visited,
            best_found: initial,
            config,
            costs,
            rng: Rng::new(seed),
            expansions: 0,
            initial_cost,
        }
    }

    /// Create a new planner from an expression tree (no rules).
    ///
    /// This creates an e-graph with no rewrite rules. Use `from_tree_with_rules`
    /// to provide rules for algebraic simplification.
    pub fn from_tree(tree: &ExprTree, config: BestFirstConfig) -> Self {
        Self::from_tree_with_rules(tree, config, Vec::new())
    }

    /// Check if the kernel is small enough to just saturate.
    pub fn should_saturate(&self) -> bool {
        self.best_found.egraph.node_count() < self.config.saturation_threshold
    }

    /// Run search with a heuristic evaluator function.
    ///
    /// The evaluator receives a `BestFirstContext` containing:
    /// - The expression tree to evaluate
    /// - Search progress metrics (depth, expansions, frontier size, etc.)
    /// - Time pressure features (for chess-style time management)
    ///
    /// Returns a priority score (lower = more promising).
    /// This is where NNUE integration happens.
    ///
    /// ## Time Control (Chess-Style)
    ///
    /// The search respects multiple limits:
    /// - **Soft timeout**: Switch to pure exploitation (ε=0)
    /// - **Hard timeout**: Stop immediately and return best-so-far
    /// - **Size limit**: Stop if e-graph exceeds max_classes
    pub fn run<F>(&mut self, mut evaluator: F) -> BestFirstResult
    where
        F: FnMut(BestFirstContext<'_>) -> i64,
    {
        let start = Instant::now();

        // Small kernel? Just saturate.
        if self.should_saturate() {
            return self.run_saturation(start);
        }

        let mut stop_reason = StopReason::Exhausted;

        // Best-first search with ε-greedy exploration and time control
        while self.expansions < self.config.max_expansions && !self.open_set.is_empty() {
            let elapsed = start.elapsed();

            // HARD LIMIT: Stop immediately
            if elapsed >= self.config.hard_timeout {
                stop_reason = StopReason::HardTimeout;
                break;
            }

            // SIZE LIMIT: Stop if e-graph exploded
            let current_classes = self.best_found.egraph.num_classes();
            if current_classes > self.config.max_classes {
                stop_reason = StopReason::SizeLimit;
                break;
            }

            // SOFT LIMIT: Reduce exploration when in time trouble
            let in_time_trouble = elapsed >= self.config.soft_timeout;
            let effective_epsilon = if in_time_trouble { 0.0 } else { self.config.epsilon };

            // SUPERGUIDE: periodic deep exploration burst
            let is_superguide = self.config.superguide_interval
                .map(|interval| interval > 0 && self.expansions % interval == 0)
                .unwrap_or(false);
            let burst_count = if is_superguide && !in_time_trouble {
                self.config.superguide_multiplier
            } else {
                1
            };

            // Expand `burst_count` states (1 normally, more during superguide)
            for _ in 0..burst_count {
                if self.open_set.is_empty() {
                    break;
                }

                // ε-greedy: occasionally pick random instead of best (unless in time trouble)
                let current = if self.rng.gen_f64() < effective_epsilon && self.open_set.len() > 1 {
                    self.pick_random()
                } else {
                    self.open_set.pop().unwrap()
                };

                // Expand: apply one round of rewrites
                if let Some(mut neighbor) = current.expand(&self.costs) {
                    // Skip if already visited (cycle detection)
                    if self.visited.contains(&neighbor.state_hash) {
                        continue;
                    }

                    // Build context for evaluator (with time features)
                    let context = BestFirstContext {
                        tree: &neighbor.best_tree,
                        tree_cost: neighbor.best_cost,
                        depth: neighbor.depth,
                        best_cost: self.best_found.best_cost,
                        expansions: self.expansions,
                        frontier_size: self.open_set.len(),
                        visited_count: self.visited.len(),
                        initial_cost: self.initial_cost,
                        // Time features (chess-style)
                        time_elapsed: elapsed,
                        time_budget: self.config.hard_timeout,
                        max_classes: self.config.max_classes,
                        current_classes: neighbor.egraph.num_classes(),
                    };

                    // Score with NNUE heuristic
                    neighbor.priority = evaluator(context);

                    // Update global best if improved
                    if neighbor.best_cost < self.best_found.best_cost {
                        self.best_found = neighbor.clone();
                    }

                    // Add to frontier
                    self.visited.insert(neighbor.state_hash);
                    self.open_set.push(neighbor);
                }
            }

            self.expansions += 1;

            // Prune queue if too large
            if self.open_set.len() > self.config.max_queue_size {
                self.prune_queue();
            }
        }

        // Determine final stop reason if we didn't already set one
        if stop_reason == StopReason::Exhausted {
            if self.expansions >= self.config.max_expansions {
                stop_reason = StopReason::MaxExpansions;
            }
            // Otherwise it truly exhausted (open_set empty)
        }

        BestFirstResult {
            best_tree: self.best_found.best_tree.clone(),
            best_cost: self.best_found.best_cost,
            expansions: self.expansions,
            used_saturation: false,
            initial_cost: self.initial_cost,
            elapsed: start.elapsed(),
            stop_reason,
        }
    }

    /// Run with default evaluator (just uses CostModel).
    ///
    /// Uses a time-aware heuristic:
    /// - When time is plentiful, prefer exploring deeper trees
    /// - When in time trouble, just minimize cost
    pub fn run_default(&mut self) -> BestFirstResult {
        self.run(|ctx| {
            // Time-aware heuristic
            if ctx.in_time_trouble() {
                // In time trouble: just minimize cost, no exploration bonus
                ctx.tree_cost as i64
            } else {
                // Normal mode: balance cost with exploration potential
                // Prefer shallower trees (they have more room to optimize)
                (ctx.tree_cost as i64) + (ctx.tree.depth() as i64 * 5)
            }
        })
    }

    /// Run with dual-head NNUE for learned priorities.
    ///
    /// Uses the **search head** of the dual-head NNUE to determine
    /// which states to expand (heap ordering).
    ///
    /// ## How It Works
    ///
    /// ```text
    /// During search:
    ///   1. Pop state from heap (ordered by SEARCH HEAD priority)
    ///   2. Expand → get new expressions
    ///   3. For each new expression:
    ///      - CostModel: "What's the cost?" → update best_found if lower
    ///      - SEARCH HEAD: "What's the priority?" → push to heap
    ///   4. At end: return best_found (already tracked, just remembered)
    /// ```
    ///
    /// The search head learns which expressions are worth exploring further,
    /// while the CostModel (or value head) determines which is cheapest.
    ///
    /// ## Example
    ///
    /// ```ignore
    /// let nnue = ExprNnue::new_with_latency_prior(42);
    /// let mut planner = BestFirstPlanner::from_tree(&tree, config);
    /// let result = planner.run_with_nnue(&nnue);
    /// ```
    /// Run search using the dual-head NNUE for both cost evaluation and priority.
    ///
    /// This is the proper way to use the trained NNUE:
    /// - **Value head**: Evaluates expression cost (replaces hardcoded CostModel)
    /// - **Search head**: Determines heap priority (which states to explore)
    ///
    /// The CostModel is still used for tree extraction (picking which nodes
    /// from the e-graph), but cost *comparison* uses the learned value head.
    #[cfg(feature = "nnue")]
    pub fn run_with_nnue(&mut self, nnue: &crate::nnue::ExprNnue) -> BestFirstResult {
        use super::nnue_adapter::{predict_tree_cost, predict_tree_priority, NnueCostAdapter};

        // Use NNUE for extraction costs instead of hardcoded CostModel
        let nnue_costs = NnueCostAdapter::new(nnue);

        // Re-evaluate initial state with NNUE value head
        let initial_nnue_cost = predict_tree_cost(&self.best_found.best_tree, nnue);
        self.best_found.best_cost = (initial_nnue_cost * 1000.0) as usize; // Scale for integer comparison
        self.initial_cost = self.best_found.best_cost;

        let start = Instant::now();
        let mut stop_reason = StopReason::Exhausted;

        while self.expansions < self.config.max_expansions && !self.open_set.is_empty() {
            // Time limits
            let elapsed = start.elapsed();
            if elapsed > self.config.hard_timeout {
                stop_reason = StopReason::HardTimeout;
                break;
            }
            let in_time_trouble = elapsed > self.config.soft_timeout;

            // Expansion limit
            if self.expansions >= self.config.max_expansions {
                stop_reason = StopReason::MaxExpansions;
                break;
            }

            // Size limit
            let current_classes = self.best_found.egraph.num_classes();
            if current_classes > self.config.max_classes {
                stop_reason = StopReason::SizeLimit;
                break;
            }

            // SUPERGUIDE: periodic deep exploration burst
            let is_superguide = self.config.superguide_interval
                .map(|interval| interval > 0 && self.expansions % interval == 0)
                .unwrap_or(false);
            let burst_count = if is_superguide && !in_time_trouble {
                self.config.superguide_multiplier
            } else {
                1
            };

            // Expand `burst_count` states (1 normally, more during superguide)
            for _ in 0..burst_count {
                let Some(mut current) = self.open_set.pop() else {
                    break;
                };

                // Re-evaluate current state's cost with NNUE
                let current_nnue_cost = predict_tree_cost(&current.best_tree, nnue);
                current.best_cost = (current_nnue_cost * 1000.0) as usize;

                // Expand using NNUE costs for extraction
                if let Some(neighbor) = current.expand(&nnue_costs) {
                    if !self.visited.contains(&neighbor.state_hash) {
                        self.visited.insert(neighbor.state_hash);

                        // Evaluate neighbor with NNUE value head
                        let neighbor_nnue_cost = predict_tree_cost(&neighbor.best_tree, nnue);
                        let neighbor_cost = (neighbor_nnue_cost * 1000.0) as usize;

                        // Compute priority using search head
                        let ctx = BestFirstContext {
                            tree: &neighbor.best_tree,
                            tree_cost: neighbor_cost,
                            depth: neighbor.depth,
                            best_cost: self.best_found.best_cost,
                            expansions: self.expansions,
                            frontier_size: self.open_set.len(),
                            visited_count: self.visited.len(),
                            initial_cost: self.initial_cost,
                            time_elapsed: elapsed,
                            time_budget: self.config.hard_timeout,
                            max_classes: self.config.max_classes,
                            current_classes,
                        };

                        let priority = if ctx.in_time_trouble() {
                            neighbor_cost as i64
                        } else {
                            predict_tree_priority(&neighbor.best_tree, nnue)
                        };

                        let mut neighbor = neighbor;
                        neighbor.best_cost = neighbor_cost;
                        neighbor.priority = priority;

                        // Update best if improved (using NNUE-evaluated cost)
                        if neighbor.best_cost < self.best_found.best_cost {
                            self.best_found = neighbor.clone();
                        }

                        self.open_set.push(neighbor);
                    }
                }
            }

            self.expansions += 1;
        }

        BestFirstResult {
            best_tree: self.best_found.best_tree.clone(),
            best_cost: self.best_found.best_cost,
            expansions: self.expansions,
            used_saturation: false,
            initial_cost: self.initial_cost,
            elapsed: start.elapsed(),
            stop_reason,
        }
    }

    /// Run NNUE-guided search while recording the trajectory for training.
    ///
    /// Combines `run_with_nnue()` (NNUE-guided search) with trajectory recording
    /// for collecting training data using the trained model.
    #[cfg(feature = "nnue")]
    pub fn run_recording_with_nnue(
        &mut self,
        nnue: &crate::nnue::ExprNnue,
    ) -> (BestFirstResult, SearchTrajectory) {
        use super::nnue_adapter::{predict_tree_cost, predict_tree_priority};

        let start = Instant::now();
        let mut steps = Vec::new();

        // Re-evaluate initial state with NNUE value head
        let initial_nnue_cost = predict_tree_cost(&self.best_found.best_tree, nnue);
        self.best_found.best_cost = (initial_nnue_cost * 1000.0) as usize;
        self.initial_cost = self.best_found.best_cost;

        // Small kernel? Just saturate
        if self.should_saturate() {
            let result = self.run_saturation(start);
            let trajectory = SearchTrajectory {
                initial_cost: self.initial_cost,
                steps,
                final_cost: result.best_cost,
                stop_reason: result.stop_reason,
            };
            return (result, trajectory);
        }

        let mut stop_reason = StopReason::Exhausted;

        while let Some(mut current) = self.open_set.pop() {
            // Re-evaluate current state's cost with NNUE
            let current_nnue_cost = predict_tree_cost(&current.best_tree, nnue);
            current.best_cost = (current_nnue_cost * 1000.0) as usize;

            let elapsed = start.elapsed();
            if elapsed > self.config.hard_timeout {
                stop_reason = StopReason::HardTimeout;
                self.open_set.push(current);
                break;
            }
            if elapsed > self.config.soft_timeout && self.expansions > 0 {
                stop_reason = StopReason::HardTimeout;
                self.open_set.push(current);
                break;
            }
            if self.expansions >= self.config.max_expansions {
                stop_reason = StopReason::MaxExpansions;
                self.open_set.push(current);
                break;
            }

            let current_classes = self.best_found.egraph.num_classes();
            if current_classes > self.config.max_classes {
                stop_reason = StopReason::SizeLimit;
                self.open_set.push(current);
                break;
            }

            self.expansions += 1;

            // Record step BEFORE expansion
            steps.push(TrajectoryStep {
                tree: current.best_tree.clone(),
                tree_cost: current.best_cost,
                best_cost_so_far: self.best_found.best_cost,
                expansion: self.expansions,
                num_classes: current.egraph.num_classes(),
            });

            if let Some(neighbor) = current.expand(&self.costs) {
                if !self.visited.contains(&neighbor.state_hash) {
                    self.visited.insert(neighbor.state_hash);

                    // Evaluate with NNUE
                    let neighbor_nnue_cost = predict_tree_cost(&neighbor.best_tree, nnue);
                    let neighbor_cost = (neighbor_nnue_cost * 1000.0) as usize;

                    let ctx = BestFirstContext {
                        tree: &neighbor.best_tree,
                        tree_cost: neighbor_cost,
                        depth: neighbor.depth,
                        best_cost: self.best_found.best_cost,
                        expansions: self.expansions,
                        frontier_size: self.open_set.len(),
                        visited_count: self.visited.len(),
                        initial_cost: self.initial_cost,
                        time_elapsed: elapsed,
                        time_budget: self.config.hard_timeout,
                        max_classes: self.config.max_classes,
                        current_classes,
                    };

                    let priority = if ctx.in_time_trouble() {
                        neighbor_cost as i64
                    } else {
                        predict_tree_priority(&neighbor.best_tree, nnue)
                    };

                    let mut neighbor = neighbor;
                    neighbor.best_cost = neighbor_cost;
                    neighbor.priority = priority;

                    if neighbor.best_cost < self.best_found.best_cost {
                        self.best_found = neighbor.clone();
                    }

                    self.open_set.push(neighbor);
                }
            }
        }

        let result = BestFirstResult {
            best_tree: self.best_found.best_tree.clone(),
            best_cost: self.best_found.best_cost,
            expansions: self.expansions,
            used_saturation: false,
            initial_cost: self.initial_cost,
            elapsed: start.elapsed(),
            stop_reason,
        };

        let trajectory = SearchTrajectory {
            initial_cost: self.initial_cost,
            steps,
            final_cost: result.best_cost,
            stop_reason,
        };

        (result, trajectory)
    }

    /// Run search while recording the trajectory for training.
    ///
    /// Returns both the result and a trajectory that can be labeled with
    /// hindsight information (e.g., the target cost from saturation or
    /// longer search).
    ///
    /// ## Training Data Collection
    ///
    /// ```ignore
    /// // Run with budget B, record trajectory
    /// let (result_b, trajectory) = planner.run_recording(|ctx| ctx.tree_cost as i64);
    ///
    /// // Get target cost (from saturation or more compute)
    /// let target_cost = saturate_and_extract(&tree).cost;
    ///
    /// // Label trajectory for training
    /// for (tree, target) in trajectory.labeled_samples(target_cost) {
    ///     trainer.add_search_sample(tree_to_expr(&tree), target);
    /// }
    /// ```
    pub fn run_recording<F>(&mut self, mut evaluator: F) -> (BestFirstResult, SearchTrajectory)
    where
        F: FnMut(BestFirstContext<'_>) -> i64,
    {
        let start = Instant::now();
        let mut steps = Vec::new();

        // Small kernel? Just saturate (no trajectory - it's optimal).
        if self.should_saturate() {
            let result = self.run_saturation(start);
            let trajectory = SearchTrajectory {
                initial_cost: self.initial_cost,
                steps: Vec::new(), // No steps - saturated directly
                final_cost: result.best_cost,
                stop_reason: result.stop_reason,
            };
            return (result, trajectory);
        }

        let mut stop_reason = StopReason::Exhausted;

        while self.expansions < self.config.max_expansions && !self.open_set.is_empty() {
            let elapsed = start.elapsed();

            if elapsed >= self.config.hard_timeout {
                stop_reason = StopReason::HardTimeout;
                break;
            }

            let current_classes = self.best_found.egraph.num_classes();
            if current_classes > self.config.max_classes {
                stop_reason = StopReason::SizeLimit;
                break;
            }

            let in_time_trouble = elapsed >= self.config.soft_timeout;
            let effective_epsilon = if in_time_trouble { 0.0 } else { self.config.epsilon };

            let current = if self.rng.gen_f64() < effective_epsilon && self.open_set.len() > 1 {
                self.pick_random()
            } else {
                self.open_set.pop().unwrap()
            };

            // RECORD: the state we're about to expand
            steps.push(TrajectoryStep {
                tree: current.best_tree.clone(),
                tree_cost: current.best_cost,
                best_cost_so_far: self.best_found.best_cost,
                expansion: self.expansions,
                num_classes: current.egraph.num_classes(),
            });

            if let Some(mut neighbor) = current.expand(&self.costs) {
                if self.visited.contains(&neighbor.state_hash) {
                    continue;
                }

                let context = BestFirstContext {
                    tree: &neighbor.best_tree,
                    tree_cost: neighbor.best_cost,
                    depth: neighbor.depth,
                    best_cost: self.best_found.best_cost,
                    expansions: self.expansions,
                    frontier_size: self.open_set.len(),
                    visited_count: self.visited.len(),
                    initial_cost: self.initial_cost,
                    time_elapsed: elapsed,
                    time_budget: self.config.hard_timeout,
                    max_classes: self.config.max_classes,
                    current_classes: neighbor.egraph.num_classes(),
                };

                neighbor.priority = evaluator(context);

                if neighbor.best_cost < self.best_found.best_cost {
                    self.best_found = neighbor.clone();
                }

                self.visited.insert(neighbor.state_hash);
                self.open_set.push(neighbor);
            }

            self.expansions += 1;

            if self.open_set.len() > self.config.max_queue_size {
                self.prune_queue();
            }
        }

        if stop_reason == StopReason::Exhausted {
            if self.expansions >= self.config.max_expansions {
                stop_reason = StopReason::MaxExpansions;
            }
        }

        let result = BestFirstResult {
            best_tree: self.best_found.best_tree.clone(),
            best_cost: self.best_found.best_cost,
            expansions: self.expansions,
            used_saturation: false,
            initial_cost: self.initial_cost,
            elapsed: start.elapsed(),
            stop_reason,
        };

        let trajectory = SearchTrajectory {
            initial_cost: self.initial_cost,
            steps,
            final_cost: result.best_cost,
            stop_reason,
        };

        (result, trajectory)
    }

    /// Saturate the e-graph completely (for small kernels).
    fn run_saturation(&mut self, start: Instant) -> BestFirstResult {
        self.best_found.egraph.saturate();
        let (tree, cost) = self.best_found.egraph.extract_best(self.best_found.root, &self.costs);

        BestFirstResult {
            best_tree: tree,
            best_cost: cost,
            expansions: 0,
            used_saturation: true,
            initial_cost: self.initial_cost,
            elapsed: start.elapsed(),
            stop_reason: StopReason::Saturated,
        }
    }

    /// Pick a random state from the queue.
    fn pick_random(&mut self) -> SearchState {
        let mut all: Vec<_> = std::iter::from_fn(|| self.open_set.pop()).collect();
        let idx = self.rng.gen_range(all.len());
        let picked = all.remove(idx);
        for s in all {
            self.open_set.push(s);
        }
        picked
    }

    /// Prune the queue to max size by removing worst states.
    fn prune_queue(&mut self) {
        let mut all: Vec<_> = std::iter::from_fn(|| self.open_set.pop()).collect();
        all.truncate(self.config.max_queue_size / 2);
        for s in all {
            self.open_set.push(s);
        }
    }

    /// Get current best cost.
    pub fn best_cost(&self) -> usize {
        self.best_found.best_cost
    }

    /// Get current best tree.
    pub fn best_tree(&self) -> &ExprTree {
        &self.best_found.best_tree
    }

    /// Get number of expansions performed.
    pub fn expansions(&self) -> usize {
        self.expansions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{Leaf, ops};
    use crate::math::algebra::{AddNeg, MulRecip, Canonicalize, Cancellation, InverseAnnihilation, Involution, Annihilator, Commutative, Identity};

    /// Create standard algebraic rules for testing.
    fn test_rules() -> Vec<Box<dyn Rewrite>> {
        vec![
            // InversePair rules
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            // Commutativity
            Commutative::new(&ops::Add),
            Commutative::new(&ops::Mul),
            // Identity
            Identity::new(&ops::Add),
            Identity::new(&ops::Mul),
            // Annihilator
            Annihilator::new(&ops::Mul),
        ]
    }

    #[test]
    fn test_small_kernel_saturates() {
        // x + 0 should simplify to x
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Const(0.0)),
            ],
        };

        let config = BestFirstConfig::default()
            .with_saturation_threshold(100);

        let mut planner = BestFirstPlanner::from_tree_with_rules(&tree, config, test_rules());
        let result = planner.run_default();

        assert!(result.used_saturation);
        // Should simplify to just Var(0)
        assert!(matches!(result.best_tree, ExprTree::Leaf(Leaf::Var(0))));
    }

    #[test]
    fn test_epsilon_greedy() {
        let tree = ExprTree::Op {
            op: &ops::Mul,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Const(1.0)),
            ],
        };

        // High epsilon = lots of exploration
        let config = BestFirstConfig::default()
            .with_epsilon(0.5)
            .with_max_expansions(10);

        let mut planner = BestFirstPlanner::from_tree_with_rules(&tree, config, test_rules());
        let result = planner.run_default();

        // Should still find the optimum
        assert!(result.best_cost <= 2); // Var(0) has cost ~1
    }

    #[test]
    fn test_egraph_is_clone() {
        // Verify EGraph can be cloned (needed for search branching)
        let mut eg = EGraph::new();
        let x = eg.add_expr(&ExprTree::Leaf(Leaf::Var(0)));
        let _cloned = eg.clone();
        assert!(eg.num_classes() > 0);
        let _ = x; // silence warning
    }

    // ========================================================================
    // Time Control Tests (Chess-Style Time Management)
    // ========================================================================

    #[test]
    fn test_hard_timeout_respected() {
        // Create a complex expression that would take many iterations
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Op {
                    op: &ops::Mul,
                    children: vec![
                        ExprTree::Leaf(Leaf::Var(0)),
                        ExprTree::Leaf(Leaf::Var(1)),
                    ],
                },
                ExprTree::Op {
                    op: &ops::Mul,
                    children: vec![
                        ExprTree::Leaf(Leaf::Var(2)),
                        ExprTree::Leaf(Leaf::Var(3)),
                    ],
                },
            ],
        };

        // Short timeout (10ms) with huge expansion limit
        let config = BestFirstConfig::default()
            .with_hard_timeout(Duration::from_millis(10))
            .with_soft_timeout(Duration::from_millis(5))
            .with_max_expansions(1_000_000)
            .with_saturation_threshold(0); // Force best-first search

        let mut planner = BestFirstPlanner::from_tree(&tree, config);
        let start = std::time::Instant::now();
        let result = planner.run_default();
        let elapsed = start.elapsed();

        // Should complete within reasonable time (allow 500ms for overhead)
        assert!(elapsed < Duration::from_millis(500),
                "Search took too long: {:?}", elapsed);

        // Should report timeout as stop reason (or exhausted if it finished fast)
        assert!(matches!(result.stop_reason, StopReason::HardTimeout | StopReason::Exhausted),
                "Expected HardTimeout or Exhausted, got {:?}", result.stop_reason);
    }

    #[test]
    fn test_blitz_config_is_fast() {
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Const(0.0)),
            ],
        };

        let config = BestFirstConfig::blitz();
        let mut planner = BestFirstPlanner::from_tree(&tree, config);
        let result = planner.run_default();

        // Blitz config should be reasonably fast (< 200ms for simple expressions)
        assert!(result.elapsed < Duration::from_millis(200),
                "Blitz took too long: {:?}", result.elapsed);
    }

    #[test]
    fn test_time_pressure_calculation() {
        // Create a context and verify time pressure calculations
        let tree = ExprTree::Leaf(Leaf::Var(0));
        let context = BestFirstContext {
            tree: &tree,
            tree_cost: 10,
            depth: 0,
            best_cost: 10,
            expansions: 0,
            frontier_size: 0,
            visited_count: 0,
            initial_cost: 10,
            time_elapsed: Duration::from_millis(45), // 90% of budget
            time_budget: Duration::from_millis(50),
            max_classes: 1000,
            current_classes: 850, // 85% of limit
        };

        // Time pressure: 45ms / 50ms = 0.9
        assert!((context.time_pressure() - 0.9).abs() < 0.01);

        // Size pressure: 850 / 1000 = 0.85
        assert!((context.size_pressure() - 0.85).abs() < 0.01);

        // Past 80% threshold → in trouble
        assert!(context.in_time_trouble());
        assert!(context.in_size_trouble());
    }

    #[test]
    fn test_stop_reason_variants() {
        // Test that we get different stop reasons for different limits

        // 1. Exhausted (very small tree, no rewrites needed)
        let tree = ExprTree::Leaf(Leaf::Var(0));
        let config = BestFirstConfig::default()
            .with_saturation_threshold(0); // Force best-first search
        let mut planner = BestFirstPlanner::from_tree(&tree, config);
        let result = planner.run_default();
        // Single var can't be rewritten, so exhausted
        assert_eq!(result.stop_reason, StopReason::Exhausted,
                   "Expected Exhausted for single var, got {:?}", result.stop_reason);

        // 2. MaxExpansions
        let tree = ExprTree::Op {
            op: &ops::Add,
            children: vec![
                ExprTree::Leaf(Leaf::Var(0)),
                ExprTree::Leaf(Leaf::Var(1)),
            ],
        };
        let config = BestFirstConfig::default()
            .with_max_expansions(1)
            .with_hard_timeout(Duration::from_secs(60))
            .with_saturation_threshold(0);
        let mut planner = BestFirstPlanner::from_tree(&tree, config);
        let result = planner.run_default();
        assert_eq!(result.stop_reason, StopReason::MaxExpansions,
                   "Expected MaxExpansions, got {:?}", result.stop_reason);
    }

    #[test]
    fn test_config_presets() {
        // Verify the presets have the expected relative properties
        let blitz = BestFirstConfig::blitz();
        let rapid = BestFirstConfig::rapid();
        let classical = BestFirstConfig::classical();

        // Blitz has shorter timeouts than rapid
        assert!(blitz.hard_timeout <= rapid.hard_timeout,
                "Blitz should have shorter timeout than rapid");
        assert!(blitz.max_expansions <= rapid.max_expansions,
                "Blitz should have fewer expansions than rapid");

        // Classical is most thorough
        assert!(classical.hard_timeout > rapid.hard_timeout,
                "Classical should have longer timeout than rapid");
        assert!(classical.max_classes > rapid.max_classes,
                "Classical should have more classes than rapid");

        // Default is rapid
        let default = BestFirstConfig::default();
        assert_eq!(default.hard_timeout, rapid.hard_timeout);
    }
}
