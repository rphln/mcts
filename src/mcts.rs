use std::iter::successors;

use rand::{Rng, seq::SliceRandom};
use rustc_hash::FxHashMap;

use crate::game::{Command, Game};

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
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub tree: Vec<MctsNode>,
    /// Global value estimator.
    pub heuristic: FxHashMap<Command, f64>,
    /// See <https://stackoverflow.com/a/35666246>.
    pub visits_to_expand: usize,
    /// Exploration rate (ε) for the ε-greedy policy.
    pub exploration_rate: f64,
}

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct MctsNode {
    /// Move used to reach this node.
    pub mov: Command,
    /// Number of visits.
    pub visits: usize,
    /// Estimated reward.
    pub value: f64,
    /// Parent index.
    pub parent: usize,
    /// First child index (inclusive).
    pub head: usize,
    /// Last child index (exclusive).
    pub last: usize,
}

impl MctsNode {
    /// Creates a new node with default statistics.
    #[must_use]
    fn new(parent: usize, mov: Command) -> Self {
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
    pub fn new(root_move: Command) -> Self {
        Self {
            tree: vec![MctsNode::new(Self::SENTINEL, root_move)],
            heuristic: FxHashMap::default(),
            visits_to_expand: 1,
            exploration_rate: 0.2,
        }
    }

    /// Returns the total number of nodes in the tree.
    #[expect(
        clippy::len_without_is_empty,
        reason = "The tree always contains at least the root."
    )]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Returns the root node of the tree.
    #[must_use]
    pub fn root(&self) -> &MctsNode {
        &self.tree[0]
    }

    /// Returns the child nodes of the root.
    #[must_use]
    pub fn children(&self) -> &[MctsNode] {
        let head = self.tree[0].head;
        let last = self.tree[0].last;

        &self.tree[head..last]
    }

    /// Returns the depth of a `node`.
    #[must_use]
    pub fn depth(&self, node: usize) -> usize {
        if node == 0 {
            return 0;
        }

        1 + self.depth(self.tree[node].parent)
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

    /// Traverses the tree from the root and selects the next leaf node to
    /// explore.
    ///
    /// This method allows for batched evaluation and back-propagation of the
    /// batched evaluations.
    ///
    /// # Panics
    ///
    /// Panics if there are no legal moves available during the search.
    pub fn select_and_expand(&mut self, game: &mut Game, rng: &mut impl Rng) -> usize {
        self.select_and_expand_node(0, game, rng)
    }

    /// Traverses the tree from a given node and selects the next leaf node to
    /// explore.
    ///
    /// Internal implementation of [`Mcts::select_and_expand`].
    pub fn select_and_expand_node(
        &mut self,
        node: usize,
        game: &mut Game,
        rng: &mut impl Rng,
    ) -> usize {
        let skip_expansion = self.tree[node].visits < self.visits_to_expand;
        if skip_expansion || game.is_over() {
            return node;
        }

        let mut head = self.tree[node].head;
        let mut last = self.tree[node].last;

        if head == last {
            head = self.tree.len();

            for mov in game.moves() {
                self.tree.push(MctsNode::new(node, mov));
            }

            last = self.tree.len();
            assert_ne!(head, last, "No legal moves.");

            self.tree[node].head = head;
            self.tree[node].last = last;

            // Shuffle to reduce ordering bias during selection.
            self.tree[head..last].shuffle(rng);

            // Advance through the non-decision nodes.
            //
            // See <https://www.chessprogramming.org/One_Reply_Extensions>.
            if last - head == 1 {
                self.tree[head].visits = self.tree[node].visits;
                self.tree[head].value = -self.tree[node].value;
            }
        }

        let parent_value = -self.tree[node].value;
        let t = f64::sqrt(self.tree[node].visits as f64);

        let next = head
            + epsilon_greedy_policy(
                &self.tree[head..last],
                self.exploration_rate,
                rng,
                |child| {
                    let relative_heuristic =
                        self.heuristic.get(&child.mov).copied().unwrap_or_default();
                    let heuristic = parent_value + relative_heuristic;

                    let n = f64::sqrt(child.visits as f64);
                    let alpha = n / (t + n);

                    alpha * child.value + (1. - alpha) * heuristic
                },
            );
        game.play(self.tree[next].mov);

        self.select_and_expand_node(next, game, rng)
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
            (value - self.tree[node].value) / (self.tree[node].visits as f64);

        self.backward(self.tree[node].parent, value);

        if node == 0 {
            return;
        }

        let node_value = self.tree[node].value;
        let parent_value = -self.tree[self.tree[node].parent].value;

        let target = node_value - parent_value;

        let pred = self.heuristic.entry(self.tree[node].mov).or_default();
        *pred += 0.5 * (target - *pred);
    }
}

fn epsilon_greedy_policy(
    nodes: &[MctsNode],
    exploration_rate: f64,
    rng: &mut impl Rng,
    evaluate: impl Fn(&MctsNode) -> f64,
) -> usize {
    assert!(!nodes.is_empty(), "`nodes` must be non-empty.");

    if nodes.len() == 1 {
        return 0;
    }

    if rng.random_bool(exploration_rate) {
        return rng.random_range(0..nodes.len());
    }

    let mut best_index = 0;
    let mut best_value = f64::NEG_INFINITY;

    for (index, node) in nodes.iter().enumerate() {
        let value = evaluate(node);
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }

    best_index
}
