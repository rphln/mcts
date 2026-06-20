use std::{
    iter::successors,
    time::{Duration, Instant},
};

use indexmap::IndexMap;
use rand::prelude::*;
use rustc_hash::FxBuildHasher;

use crate::game::{Game, Move};

/// Placeholder for an unset node index.
const SENTINEL: usize = !0;

/// An implementation of Monte Carlo Tree Search (MCTS), following the general
/// framework surveyed by Browne et al. (2012)[^1].
///
/// The search is performed iteratively; the best move found so far is available
/// via [`Mcts::best_child`], and the full principal variation is available via
/// [`Mcts::principal_variation`].
///
/// # Features
///
/// **Tree flattening.** Nodes are stored in a single contiguous array with
/// parent–child relationships managed by integer indices (Sampson, 2023)[^2].
///
/// **Negamax.** `backward` inverts the reward sign at each level as it
/// propagates toward the root (Chess Programming Wiki, 2018)[^3], so that
/// every node's value reflects the moving player's perspective at that ply.
///
/// **Lazy expansion.** Nodes are expanded only upon their first visit, saving
/// memory.
///
/// **One-reply extensions.** Upon reaching a single-child node, the search
/// extends until the next decision node (Chess Programming Wiki, 2018)[^4],
/// because the value of a forced position is wholly determined by its
/// successor.
///
/// **Soft rewards.** Nodes store continuous reward values rather than binary
/// win or loss outcomes.
///
/// **Epsilon-greedy exploration.** `select_node` chooses a random child with
/// probability ε, otherwise defers to the MC-RAVE policy.
///
/// **MC-RAVE selection.** In the greedy branch, moves are scored by a weighted
/// blend of local and global statistics (Gelly and Silver, 2011)[^5].
///
/// **Early playout termination.** Random playouts are truncated after a fixed
/// number of moves (Lorentz, 2016)[^6], reducing memory pressure and either
/// improving playing strength or remaining a non-regression.
///
/// **Batched evaluation.** Multiple leaf nodes can be expanded and evaluated
/// before their rewards are back-propagated.
///
/// **Garbage collection.** When invoked, `prune` severs subtrees with visits
/// below a threshold (Powley et al., 2017)[^7].
///
/// [^1]: C.B. Browne et al. A Survey of Monte Carlo Tree Search Methods.
///        *IEEE Trans. Comput. Intell. AI Games*, 2012.
///        <https://doi.org/10.1109/TCIAIG.2012.2186810>
///
/// [^2]: A. Sampson. Flattening ASTs (and Other Compiler Data Structures). 2023.
///        <https://www.cs.cornell.edu/~asampson/blog/flattening.html>
///
/// [^3]: Chess Programming Wiki. Negamax. 2018.
///        <https://www.chessprogramming.org/Negamax>
///
/// [^4]: Chess Programming Wiki. One Reply Extensions. 2018.
///        <https://www.chessprogramming.org/One_Reply_Extensions>
///
/// [^5]: S. Gelly and D. Silver. Monte-Carlo Tree Search and Rapid Action
///        Value Estimation in Computer Go. *Artificial Intelligence*, 2011.
///        <https://doi.org/10.1016/j.artint.2011.03.007>
///
/// [^6]: R.J. Lorentz. Early Playout Termination in MCTS. *Theoretical
///        Computer Science*, 2016.
///        <https://doi.org/10.1016/j.tcs.2016.06.026>
///
/// [^7]: E.J. Powley, P.I. Cowling, and D. Whitehouse. Memory Bounded Monte
///        Carlo Tree Search. *AIIDE*, 2017.
///        <https://doi.org/10.1609/aiide.v13i1.12932>
#[derive(Clone, Debug)]
pub struct Mcts {
    /// Flattened tree representation (Sampson, 2023) [2].
    pub nodes: Vec<Node>,
    /// Per-move statistics, shared across the tree, with move interning.
    pub moves: IndexMap<Move, History, FxBuildHasher>,
    /// Exploration rate for ε-greedy selection.
    pub exploration_rate: f64,
    /// Number of moves to make in the playout phase (Lorentz, 2016) [6].
    pub termination_moves: u32,
}

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct Node {
    /// Move used to reach this node.
    pub mov: usize,
    /// Parent index.
    pub parent: usize,
    /// First child index (inclusive).
    pub head: usize,
    /// Last child index (exclusive).
    pub last: usize,
    /// Number of visits.
    pub visits: u32,
    /// Estimated reward.
    pub value: f64,
}

/// Aggregated statistics for a move.
#[derive(Clone, Default, Debug)]
pub struct History {
    /// Number of visits.
    pub visits: u32,
    /// Estimated reward.
    pub value: f64,
}

impl Node {
    /// Creates a new node with default statistics.
    #[must_use]
    const fn new(parent: usize, mov: usize) -> Node {
        Node { mov, parent, head: SENTINEL, last: SENTINEL, visits: 0, value: 0.0 }
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
    fn insert_batch(&mut self, mean: f64, num_samples: u32) {
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
    fn remove_batch(&mut self, mean: f64, num_samples: u32) {
        if num_samples == 0 {
            return;
        }

        assert!(num_samples <= self.visits, "cannot remove more samples than present");

        if num_samples == self.visits {
            self.visits = 0;
            self.value = 0.0;

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
            exploration_rate: 0.2,
            // Going from 0 to 2 is a gainer; going from 2 to 16 is a non-regression,
            // but with substantial memory savings.
            //
            // Surprisingly, this coincides with the results from Lorentz (2016) [6]: in
            // practice, this is ≈ 8 real moves in our game, which was also the optimal
            // number of moves in their experiments.
            termination_moves: 16,
        }
    }

    /// Returns the number of nodes in the tree.
    #[must_use]
    #[expect(clippy::len_without_is_empty, reason = "The tree can never be empty")]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the most-visited child of `node`, or `None` if `node` is a leaf.
    #[must_use]
    pub fn best_child(&self, node: usize) -> Option<usize> {
        let parent = &self.nodes[node];

        let head = parent.head;
        let last = parent.last;

        (head..last).into_iter().max_by_key(|&child| self.nodes[child].visits)
    }

    /// Returns an iterator over the principal variation rooted at `node`.
    pub fn principal_variation(&self, node: usize) -> impl Iterator<Item = usize> {
        successors(self.best_child(node), |&next| self.best_child(next))
    }

    /// Recursively traverses the tree from a given `node` and selects the next
    /// leaf node to explore. Expands nodes as needed and updates the game
    /// state along the selected path.
    ///
    /// Returning the leaf index without immediately back-propagating supports
    /// batched evaluation across multiple calls.
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
        if self.nodes[node].visits == 0 || game.is_over() {
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

    /// Expands a leaf node by generating all legal moves. Upon reaching a
    /// single-child node, the search extends until the next decision node [4].
    ///
    /// Children are shuffled to reduce selection bias.
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

            self.moves[child.mov].insert_batch(child.value, child.visits);
        }
    }

    /// Selects a child node using an epsilon-greedy policy.
    ///
    /// In the greedy branch, scores are computed as a weighted blend of local
    /// and global move statistics following Gelly and Silver (2011) [5].
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

            let n = f64::from(node.visits);
            let m = f64::from(history.visits);

            // See section 4.6 in Gelly and Silver (2011) [5].
            let beta = m / (m + n + 0.05 * m * n + f64::EPSILON);
            let value = node.value + beta * (history.value - node.value);

            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }

        best_index
    }

    /// Performs a random playout.
    ///
    /// # Panics
    ///
    /// Panics if no legal moves are available during the playout.
    pub fn default_policy(&self, game: &mut Game, rng: &mut impl Rng) -> f64 {
        let color = game.color;

        for _ in 0..self.termination_moves {
            if game.is_over() {
                break;
            }

            let mov = game.moves().choose(rng).unwrap();
            game.play(mov);
        }

        game.evaluate(color)
    }

    /// Back-propagates the reward from a leaf node up to the root.
    ///
    /// Rewards must be provided from the perspective of the side moving at the
    /// leaf.
    pub fn backward(&mut self, node: usize, value: f64) {
        if node == SENTINEL {
            return;
        }

        let value = -value;

        let entry = &mut self.nodes[node];

        entry.value += (value - entry.value) / f64::from(entry.visits + 1);
        entry.visits += 1;

        let history = &mut self.moves[entry.mov];

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
    ) -> Option<usize> {
        if !predicate(self, &self.nodes[node], game) {
            return self.best_child(node);
        }

        loop {
            let mut game = game.clone();
            let next = self.select_and_expand(node, &mut game, rng);

            let reward = self.default_policy(&mut game, rng);
            self.backward(next, reward);

            if !predicate(self, &self.nodes[next], &game) {
                break;
            }
        }

        self.best_child(node)
    }

    /// Searches from `node` until one of `max_time`, `max_iters`, or
    /// `max_nodes` is reached.
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
    ) -> Option<usize> {
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

    /// Searches for the best move, adaptively managing the time budget.
    ///
    /// Allocates a total budget of `T = clock/20 + increment/2`. At each step,
    /// given remaining budget `t`, runs two searches each with allocation
    /// `t/4`. Stops early if both searches agree on the best move, otherwise
    /// continues recursively on the remaining budget until `t` is exhausted.
    ///
    /// # Returns
    ///
    /// The total time spent searching.
    ///
    /// # Panics
    ///
    /// Panics if there are no legal moves available during the search, which
    /// represents a bug in either the MCTS or the game logic.
    pub fn search_timed(
        &mut self,
        node: usize,
        game: &Game,
        rng: &mut impl Rng,
        clock: Duration,
        increment: Duration,
    ) -> Duration {
        let max_time = clock / 20 + increment / 2;
        let min_time = Duration::from_millis(1);

        let mut elapsed = Duration::ZERO;

        loop {
            let s1 = max_time.saturating_sub(elapsed) / 4;
            let s2 = max_time.saturating_sub(elapsed) / 4;

            elapsed += s1 + s2;

            let first = self
                .search(node, game, rng, Some(s1), None, None)
                .expect("`node` should have children");
            let second = self
                .search(node, game, rng, Some(s2), None, None)
                .expect("`node` should have children");

            if first == second || elapsed + min_time >= max_time {
                break;
            }
        }

        elapsed
    }

    /// Severs the descendants of each node for which `should_prune` returns
    /// `true`, leaving the node itself as a leaf (Powley et al., 2017) [7].
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

    /// Discards everything outside the subtree reached by `mov`, making the
    /// corresponding child the new root of the tree, accessible at index `0`
    /// after the call.
    ///
    /// Returns `None` if the current root has no child corresponding to `mov`.
    ///
    /// This method invalidates all existing indices into the tree.
    #[must_use]
    pub fn reroot_to_move(mut self, mov: Move) -> Option<Mcts> {
        let entry = self.moves.entry(mov);
        let key = entry.index();

        let _ = entry.or_default();

        let head = self.nodes[0].head;
        let last = self.nodes[0].last;

        let child = (head..last).find(|&child| self.nodes[child].mov == key)?;
        self.reroot(child);

        Some(self)
    }

    /// Combined mark-and-sweep and re-rooting primitive that retains the
    /// subtree rooted at `root` and prunes it according to `should_prune`.
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

            // Skip descendants of anything already reclaimed or severed.
            if map[parent] == SENTINEL || self.nodes[parent].head == SENTINEL {
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

        // Clean up the history entries.
        for (node, &dst) in self.nodes.iter().zip(&map) {
            if dst == SENTINEL {
                self.moves[node.mov].remove_batch(node.value, node.visits);
            }
        }

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
