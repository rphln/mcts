use std::{
    iter::successors,
    time::{Duration, Instant},
};

use indexmap::IndexMap;
use rand::prelude::*;
use rustc_hash::FxBuildHasher;

use crate::game::{Game, Move};

/// An implementation of the Monte Carlo Tree Search algorithm with the
/// following modifications:
///
/// **Soft rewards.** Nodes use continuous reward values instead of binary win
/// or loss outcomes.
///
/// **Epsilon-greedy exploration.** Chooses a random action with probability ε,
/// otherwise selects the most visited node based on average reward.
///
/// **Tree flattening.** The entire tree is stored in a single array, with
/// parent–child relationships managed by indices.
///
/// **One-reply extensions.** When a node has only one legal move, the search
/// skips it, directly advancing to the next decision node.
///
/// **Batched evaluation.** Multiple leaf nodes can be expanded for batch
/// evaluation before their rewards are back-propagated.
///
/// **Delayed expansion.** Nodes are only expanded after reaching a minimum
/// number of visits. Disabled by default.
///
/// **Garbage collection.** When invoked, prunes nodes with visits below a
/// threshold.
///
/// The search is performed iteratively, and the best move found so far can be
/// queried after each iteration with [`Mcts::best_child`]. The entire principal
/// variation is available through [`Mcts::principal_variation`].
///
/// # References
///
/// 1. <https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md>
/// 2. <https://modelassist.epixanalytics.com/space/EA/26575264>
/// 3. <https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf#page=45>
/// 4. <https://gist.github.com/rphln/fbbab3e0a432b95ec93d1e29e16acb03>
/// 5. <https://repository.falmouth.ac.uk/2782/1/MemoryLimiting.pdf>
#[derive(Clone, Debug)]
pub struct Mcts {
    /// Flattened tree representation.
    ///
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub nodes: Vec<Node>,
    /// Per-move statistics, shared across the tree, with move interning.
    pub moves: IndexMap<Move, History, FxBuildHasher>,
    /// Visit threshold for node expansion.
    ///
    /// See <https://stackoverflow.com/a/35666246>.
    pub visits_to_expand: u32,
    /// Exploration rate for ε-greedy selection.
    pub exploration_rate: f64,
}

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct Node {
    /// Move used to reach this node.
    pub mov: usize,
    /// Number of visits.
    pub visits: u32,
    /// Estimated reward.
    pub value: f64,
    /// Parent index.
    pub parent: usize,
    /// First child index (inclusive).
    pub head: usize,
    /// Last child index (exclusive).
    pub last: usize,
}

/// Aggregated statistics for a move.
#[derive(Clone, Default, Debug)]
pub struct History {
    /// Number of visits.
    pub visits: u32,
    /// Estimated reward.
    pub value: f64,
}

/// Placeholder for an unset node index.
const SENTINEL: usize = !0;

impl Node {
    /// Creates a new node with default statistics.
    #[must_use]
    const fn new(parent: usize, mov: usize) -> Node {
        Node { mov, visits: 0, value: 0., parent, head: SENTINEL, last: SENTINEL }
    }

    /// Returns whether this node is the root node.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.parent == SENTINEL
    }

    /// Returns whether this node is a leaf.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.head == self.last
    }
}

impl History {
    /// Incorporates a batch of samples with the given mean into the history.
    pub fn update_batch(&mut self, mean: f64, num_samples: u32) {
        if num_samples == 0 {
            return;
        }

        let n = f64::from(self.visits);
        let k = f64::from(num_samples);

        self.value += (k / (n + k)) * (mean - self.value);
        self.visits += num_samples;
    }

    /// Removes a batch of samples with the given mean from the history.
    ///
    /// # Panics
    ///
    /// Panics if `num_samples` exceeds `visits`.
    pub fn remove_batch(&mut self, mean: f64, num_samples: u32) {
        if num_samples == 0 {
            return;
        }

        assert!(num_samples <= self.visits, "cannot remove more samples than present");

        if num_samples == self.visits {
            self.visits = 0;
            self.value = 0.;

            return;
        }

        let n = f64::from(self.visits);
        let k = f64::from(num_samples);

        self.value -= (k / (n - k)) * (mean - self.value);
        self.visits -= num_samples;
    }
}

impl Mcts {
    /// Creates a new Monte Carlo Tree Search instance.
    ///
    /// The `root_move` is an arbitrary move that represents the root of the
    /// tree.
    #[must_use]
    pub fn new(root_move: Move) -> Mcts {
        let mut moves = IndexMap::with_hasher(FxBuildHasher);

        let entry = moves.entry(root_move);
        let handle = entry.index();

        let _ = entry.or_default();

        Mcts {
            nodes: vec![Node::new(SENTINEL, handle)],
            moves,
            visits_to_expand: 1,
            exploration_rate: 0.2,
        }
    }

    /// Returns the number of nodes in the tree.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the tree only has the root.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 1
    }

    /// Returns an iterator over the sequence of best moves found so far.
    pub fn principal_variation(&self) -> impl Iterator<Item = &Node> {
        successors(self.nodes.first(), |parent| {
            self.nodes
                .get(parent.head..parent.last)?
                .iter()
                .max_by_key(|node| node.visits)
        })
    }

    /// Returns an iterator over the ancestors of `node`, starting at `node`.
    pub fn ancestors(&self, node: usize) -> impl Iterator<Item = &Node> {
        successors(self.nodes.get(node), |node| self.nodes.get(node.parent))
    }

    /// Recursively traverses the tree from a given `node` and selects the next
    /// leaf node to explore. Expands nodes as needed and updates the game
    /// state along the selected path.
    ///
    /// This method allows for batched evaluation and back-propagation of the
    /// batched evaluations.
    ///
    /// # Panics
    ///
    /// Panics if there are no legal moves available during the search.
    pub fn select_and_expand(
        &mut self,
        node: usize,
        game: &mut Game,
        rng: &mut impl Rng,
    ) -> usize {
        let skip_expansion = self.nodes[node].visits < self.visits_to_expand;
        if skip_expansion || game.is_over() {
            return node;
        }

        if self.nodes[node].is_leaf() {
            self.expand_node(node, game, rng);
        }

        let next = self.select_node(node, rng);

        let (&mov, _) = self.moves.get_index(self.nodes[next].mov).unwrap();
        game.play(mov);

        self.select_and_expand(next, game, rng)
    }

    /// Expands a leaf node by generating all legal moves.
    ///
    /// Extends the search upon reaching single-child nodes (i.e., states with
    /// forced moves), as there is no decision to be made and evaluating the
    /// next state is equivalent. See [1].
    ///
    /// Children are shuffled to reduce selection bias.
    ///
    /// [1]: <https://www.chessprogramming.org/One_Reply_Extensions>
    ///
    /// # Panics
    ///
    /// Panics if `node` is not a leaf or if there are no legal moves available.
    fn expand_node(&mut self, node: usize, game: &Game, rng: &mut impl Rng) {
        assert!(self.nodes[node].is_leaf(), "`node` should be a leaf");

        let head = self.nodes.len();

        for mov in game.moves() {
            let entry = self.moves.entry(mov);
            let handle = entry.index();

            let _ = entry.or_default();

            self.nodes.push(Node::new(node, handle));
        }

        let last = self.nodes.len();
        assert_ne!(head, last, "No legal moves.");

        self.nodes[node].head = head;
        self.nodes[node].last = last;

        self.nodes[head..last].shuffle(rng);

        if last - head == 1 {
            let [parent, child] = self.nodes.get_disjoint_mut([node, head]).unwrap();

            child.visits = parent.visits;
            child.value = -parent.value;

            self.moves[child.mov].update_batch(child.value, child.visits);
        }
    }

    /// Selects a child node using the epsilon-greedy policy.
    ///
    /// # Panics
    ///
    /// Panics if `parent` is a leaf.
    fn select_node(&self, parent: usize, rng: &mut impl Rng) -> usize {
        let head = self.nodes[parent].head;
        let last = self.nodes[parent].last;

        assert_ne!(head, last, "`parent` should not be a leaf");

        if last - head == 1 {
            return head;
        }

        if rng.random_bool(self.exploration_rate) {
            return rng.random_range(head..last);
        }

        let mut best_index = SENTINEL;
        let mut best_value = f64::NEG_INFINITY;

        for index in head..last {
            let node = &self.nodes[index];
            let history = &self.moves[node.mov];

            if history.visits < 1 {
                return index;
            }

            let n = f64::from(node.visits);

            // Found empirically. See Section 8.4.2 in [1] for other schedules.
            //
            // [1]: <https://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/1029/paper_thesis.pdf>
            let beta = f64::sqrt(1. / (1. + n));
            std::assert_matches!(beta, 0.0..=1.0, "`beta` should be in [0, 1]");

            let value = (1. - beta) * node.value + beta * history.value;
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }

        best_index
    }

    /// See <https://www.sciencedirect.com/science/article/pii/S0304397516302717>.
    ///
    /// # Panics
    ///
    /// Panics if no legal moves are available during the roll-out.
    pub fn default_policy(&self, game: &mut Game, _rng: &mut impl Rng) -> f64 {
        game.evaluate(game.color)
    }

    /// Back-propagates the reward from a leaf node up to the root.
    ///
    /// Rewards must be provided from the perspective of the side moving at the
    /// leaf.
    pub fn backward(&mut self, node: usize, value: f64) {
        if node == SENTINEL {
            return;
        }

        // See <https://www.chessprogramming.org/Negamax>.
        let value = -value;

        let entry = &mut self.nodes[node];
        let history = &mut self.moves[entry.mov];

        entry.value += (value - entry.value) / f64::from(entry.visits + 1);
        entry.visits += 1;

        history.value += (value - history.value) / f64::from(history.visits + 1);
        history.visits += 1;

        let parent = entry.parent;
        self.backward(parent, value);
    }

    /// Searches from `node` until `predicate` is false.
    ///
    /// # Panics
    ///
    /// Panics if there are no legal moves available during the search, which
    /// represents a bug in either the MCTS or the game logic.
    pub fn search_while(
        &mut self,
        node: usize,
        game: &Game,
        rng: &mut impl Rng,
        mut predicate: impl FnMut(&Mcts, &Node, &Game) -> bool,
    ) -> Option<&Node> {
        loop {
            let mut game = game.clone();
            let next = self.select_and_expand(node, &mut game, rng);

            let reward = self.default_policy(&mut game, rng);
            self.backward(next, reward);

            if !predicate(self, &self.nodes[next], &game) {
                break;
            }
        }

        self.principal_variation().nth(1)
    }

    /// Searches from `node` until one of `max_time`, `max_iters` or `max_nodes`
    /// is reached.
    ///
    /// # Panics
    ///
    /// Panics if there are no legal moves available during the search, which
    /// represents a bug in either the MCTS or the game logic.
    pub fn search(
        &mut self,
        node: usize,
        game: &Game,
        rng: &mut impl Rng,
        max_time: Option<Duration>,
        max_iters: Option<u32>,
        max_nodes: Option<usize>,
    ) -> Option<&Node> {
        let mut iters = 0;
        let mut nodes = 0;

        let start_time = Instant::now();
        let root_depth = game.depth;

        self.search_while(node, game, rng, |_mcts, _node, game| {
            iters += 1;
            nodes += game.depth - root_depth;

            max_time.is_none_or(|t| start_time.elapsed() < t)
                && max_iters.is_none_or(|n| iters < n)
                && max_nodes.is_none_or(|n| nodes < n)
        })
    }

    /// Severs the descendants of each node for which `should_prune` returns
    /// `true`, leaving the node itself as a leaf.
    ///
    /// Descendants of an already-severed node are discarded without being
    /// visited.
    ///
    /// This method invalidates all existing indices into the tree.
    pub fn prune(&mut self, should_prune: impl FnMut(&Node) -> bool) {
        self.compact(0, should_prune);
    }

    /// Discards everything outside the subtree rooted at `node`, making `node`
    /// the new root of the tree, accessible at index `0` after the call.
    ///
    /// This method invalidates all existing indices into the tree.
    ///
    /// # Panics
    ///
    /// Panics if `node` is out of bounds.
    pub fn reroot(&mut self, node: usize) {
        self.compact(node, |_node| false);
    }

    /// Combined mark-and-sweep and re-rooting primitive that retains the
    /// subtree rooted at `root` and pruning it according to `should_prune`.
    ///
    /// Severs the descendants of each node for which `should_prune` returns
    /// `true`, leaving the node itself as a leaf. Descendants of an already-
    /// severed node are discarded without being visited.
    ///
    /// The node at `root` is never removed; severing it collapses the subtree
    /// to a single node.
    ///
    /// # Panics
    ///
    /// Panics if `root` is out of bounds.
    fn compact(&mut self, root: usize, mut should_prune: impl FnMut(&Node) -> bool) {
        let len = self.nodes.len();
        assert!(root < len, "root out of bounds");

        // region: Mark phase.

        // Relocation map: `map[src] = dst`, or `SENTINEL` if reclaimed.
        let mut map = vec![SENTINEL; len];
        let mut dst = 0;

        // The root is always retained; pruning it merely severs its children.
        map[root] = 0;
        dst += 1;

        if should_prune(&self.nodes[root]) {
            self.nodes[root].head = SENTINEL;
            self.nodes[root].last = SENTINEL;
        }

        // Combined mark and relocation table pass. Walking parents before
        // children lets each node observe its parent's already-decided fate.
        for src in (root + 1)..len {
            let node = &self.nodes[src];

            let parent = node.parent;
            assert!(parent < src, "tree must be topologically ordered");

            // Recursively remove the descendants of anything already reclaimed or
            // severed.
            if map[parent] == SENTINEL || self.nodes[parent].head == SENTINEL {
                self.moves[node.mov].remove_batch(node.value, node.visits);

                continue;
            }

            map[src] = dst;
            dst += 1;

            if should_prune(node) {
                self.nodes[src].head = SENTINEL;
                self.nodes[src].last = SENTINEL;
            }
        }

        // endregion
        // region: Sweep phase.

        // Compact survivors and rewrite their pointers.
        for src in root..len {
            let dst = map[src];
            if dst == SENTINEL {
                continue;
            }

            assert!(dst <= src);
            self.nodes.swap(src, dst);

            let node = &mut self.nodes[dst];

            if !node.is_root() {
                node.parent = map[node.parent];
            }

            if !node.is_leaf() {
                let head = map[node.head];
                let last = map[node.head] + (node.last - node.head);

                node.head = head;
                node.last = last;
            }
        }

        // endregion

        self.nodes.truncate(dst);
    }
}
