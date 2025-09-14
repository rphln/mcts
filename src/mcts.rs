use std::iter::successors;

use rand::{Rng, seq::SliceRandom};
use rustc_hash::FxHashMap;

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
/// queried after each iteration with [`Self::best_child`]. The entire principal
/// variation is available through [`Self::principal_variation`].
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
    /// Flattened tree arena.
    ///
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub tree: Vec<MctsNode>,
    /// Global history statistics for moves.
    pub history: FxHashMap<Move, HistoryEntry>,
    /// See <https://stackoverflow.com/a/35666246>.
    pub visits_to_expand: u32,
    /// Exploration rate (ε) for the ε-greedy policy.
    pub exploration_rate: f64,
}

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct MctsNode {
    /// Move used to reach this node.
    pub mov: Move,
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

/// Global statistics for a move across the entire tree.
#[derive(Clone, Debug, Default)]
pub struct HistoryEntry {
    /// Visit count.
    pub visits: u32,
    /// Estimated reward delta.
    pub value: f64,
}

impl MctsNode {
    /// Creates a new node with default statistics.
    #[must_use]
    fn new(parent: usize, mov: Move) -> Self {
        Self { mov, visits: 0, value: 0., parent, head: 0, last: 0 }
    }

    /// Returns whether this node is the root node.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent == Mcts::SENTINEL
    }

    /// Returns whether this node is a leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.head == self.last
    }
}

impl Mcts {
    /// Sentinel for the root's parent.
    const SENTINEL: usize = usize::MAX;

    /// Creates a new Monte Carlo Tree Search instance.
    ///
    /// The `root_move` is an arbitrary move that represents the root of the
    /// tree.
    #[must_use]
    pub fn new(root_move: Move) -> Self {
        Self {
            tree: vec![MctsNode::new(Self::SENTINEL, root_move)],
            history: FxHashMap::default(),
            visits_to_expand: 1,
            exploration_rate: 0.2,
        }
    }

    /// Returns the best child node found so far.
    #[must_use]
    pub fn best_child(&self) -> Option<&MctsNode> {
        self.principal_variation().nth(1)
    }

    /// Returns an iterator over the sequence of best moves found so far.
    pub fn principal_variation(&self) -> impl Iterator<Item = &MctsNode> {
        successors(self.tree.first(), |parent| {
            self.tree[parent.head..parent.last].iter().max_by_key(|node| node.visits)
        })
    }

    /// Returns an iterator over the ancestors of `node`, starting at `node`.
    pub fn ancestors(&self, node: usize) -> impl Iterator<Item = &MctsNode> {
        successors(self.tree.get(node), |node| self.tree.get(node.parent))
    }

    /// Recursively traverses the tree from the root and selects the next leaf
    /// node to explore. Expands nodes as needed and updates the game state
    /// along the selected path.
    ///
    /// This method allows for batched evaluation and back-propagation of the
    /// batched evaluations.
    ///
    /// # Panics
    ///
    /// Panics if there are no legal moves available during the search.
    pub fn expand_and_select(&mut self, game: &mut Game, rng: &mut impl Rng) -> usize {
        self.expand_and_select_node(0, game, rng)
    }

    /// Traverses the tree from a given node and selects the next leaf node to
    /// explore.
    ///
    /// Internal implementation of [`Mcts::select_and_expand`].
    pub fn expand_and_select_node(
        &mut self,
        node: usize,
        game: &mut Game,
        rng: &mut impl Rng,
    ) -> usize {
        let skip_expansion = self.tree[node].visits < self.visits_to_expand;
        if skip_expansion || game.is_over() {
            return node;
        }

        if self.tree[node].is_leaf() {
            self.expand_node(node, game, rng);
        }

        let next = self.select_node(node, rng);
        game.play(self.tree[next].mov);

        self.expand_and_select_node(next, game, rng)
    }

    /// Expands a leaf node by generating all legal moves.
    ///
    /// Extends the search upon reaching single-child nodes (i.e., states with
    /// forced moves), as there is no decision to be made and evaluating the
    /// next state is equivalent. See [1].
    ///
    /// Children are shuffled to reduce selection bias.
    ///
    /// # References
    ///
    /// [1]: https://www.chessprogramming.org/One_Reply_Extensions
    fn expand_node(&mut self, node: usize, game: &Game, rng: &mut impl Rng) {
        assert!(self.tree[node].is_leaf(), "`node` should be a leaf");

        let head = self.tree.len();

        for mov in game.moves() {
            self.tree.push(MctsNode::new(node, mov));
        }

        let last = self.tree.len();
        assert_ne!(head, last, "No legal moves.");

        self.tree[node].head = head;
        self.tree[node].last = last;

        self.tree[head..last].shuffle(rng);

        if last - head == 1 {
            self.tree[head].visits = self.tree[node].visits;
            self.tree[head].value = -self.tree[node].value;
        }
    }

    /// Selects a child node using an epsilon-greedy policy with history-based
    /// weighting.
    ///
    /// # Panics
    ///
    /// Panics if `parent` is a leaf.
    fn select_node(&self, parent: usize, rng: &mut impl Rng) -> usize {
        let head = self.tree[parent].head;
        let last = self.tree[parent].last;

        assert_ne!(head, last, "`parent` should not be a leaf");

        if last - head == 1 {
            return head;
        }

        if rng.random_bool(self.exploration_rate) {
            return rng.random_range(head..last);
        }

        let parent_value = -self.tree[parent].value;

        let mut best_index = 0;
        let mut best_value = f64::NEG_INFINITY;

        for index in head..last {
            let child = &self.tree[index];

            let Some(entry) = self.history.get(&child.mov) else {
                return index;
            };

            let m = f64::from(entry.visits);
            let n = f64::from(child.visits);

            let alpha = f64::sqrt(n / (n + m)); // Found empirically.
            let value =
                alpha * child.value + (1. - alpha) * (parent_value + entry.value);

            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }

        best_index
    }

    /// Back-propagates the reward from a leaf node up to the root.
    ///
    /// Rewards must be provided from the perspective of the side moving at the
    /// leaf.
    pub fn backward(&mut self, node: usize, value: f64) {
        if node == Self::SENTINEL {
            return;
        }

        // See <https://www.chessprogramming.org/Negamax>.
        let value = -value;

        self.tree[node].visits += 1;
        self.tree[node].value +=
            (value - self.tree[node].value) / f64::from(self.tree[node].visits);

        self.backward(self.tree[node].parent, value);

        if node == 0 {
            return;
        }

        let target = {
            let parent = self.tree[node].parent;
            let parent_value = -self.tree[parent].value;

            value - parent_value
        };

        let entry = self.history.entry(self.tree[node].mov).or_default();

        entry.visits += 1;
        entry.value += (target - entry.value) / f64::from(entry.visits);
    }
}
