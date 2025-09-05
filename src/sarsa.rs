use std::iter::successors;

use rand::{Rng, seq::SliceRandom};

use crate::game::{Command, Game};

/// Stores data in a tree for tabular SARSA.
#[derive(Clone, Debug)]
pub struct Sarsa {
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub tree: Vec<SarsaNode>,
    /// See <https://stackoverflow.com/a/35666246>.
    pub visits_to_expand: usize,
    /// Exploration rate (ε) for the ε-greedy policy.
    pub exploration_rate: f64,
    /// Discount factor (γ) for the reward backprogation.
    pub discount_factor: f64,
    /// Learning rate (α) for the temporal difference update.
    pub learning_rate: f64,
}

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct SarsaNode {
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

impl SarsaNode {
    /// Creates a new node with default statistics.
    #[must_use]
    fn new(parent: usize, mov: Command) -> Self {
        Self { mov, visits: 0, value: 0., parent, head: 0, last: 0 }
    }

    /// Returns whether this node is the root node.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent == Sarsa::SENTINEL
    }

    /// Returns whether this node is a leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.head == self.last
    }
}

impl Sarsa {
    /// Sentinel for the root's parent.
    const SENTINEL: usize = usize::MAX;

    /// Creates a new Monte Carlo Tree Search instance.
    ///
    /// The `root_move` is an arbitrary move that represents the root of the
    /// tree.
    #[must_use]
    pub fn new(root_move: Command) -> Self {
        Self {
            tree: vec![SarsaNode::new(Self::SENTINEL, root_move)],
            visits_to_expand: 1,
            exploration_rate: 0.1,
            learning_rate: 0.05,
            discount_factor: 0.999,
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

    /// Returns the child nodes of the root.
    #[must_use]
    pub fn children(&self) -> &[SarsaNode] {
        let head = self.tree[0].head;
        let last = self.tree[0].last;

        &self.tree[head..last]
    }

    /// Returns the best child node found so far.
    #[must_use]
    pub fn best_child(&self) -> Option<&SarsaNode> {
        self.principal_variation().nth(1)
    }

    /// Returns an iterator over the sequence of best moves found so far.
    pub fn principal_variation(&self) -> impl Iterator<Item = &SarsaNode> {
        successors(self.tree.first(), |parent| {
            self.tree[parent.head..parent.last]
                .iter()
                .filter(|node| node.visits > 0)
                .max_by(|left, right| {
                    f64::partial_cmp(&left.value, &right.value).unwrap()
                })
        })
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
    /// Internal implementation of [`Sarsa::select_and_expand`].
    fn select_and_expand_node(
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
                self.tree.push(SarsaNode::new(node, mov));
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
                self.tree[head].value = -self.tree[node].value / self.discount_factor;
            }
        }

        let next = head
            + epsilon_greedy_policy(&self.tree[head..last], self.exploration_rate, rng);
        game.play(self.tree[next].mov);

        self.select_and_expand_node(next, game, rng)
    }

    /// Back-propagates the reward from a leaf node up to the root.
    ///
    /// Rewards must be provided from the perspective of the side moving at the
    /// leaf.
    pub fn backward(&mut self, node: usize, value: f64) {
        // See <https://www.chessprogramming.org/Negamax>.
        let value = -value;

        // See <https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html#q-value-initialisation>.
        self.tree[node].value = value;
        self.tree[node].visits += 1;

        self.backward_sarsa(self.tree[node].parent, self.tree[node].value);
    }

    pub fn backward_sarsa(&mut self, node: usize, value: f64) {
        if node == Self::SENTINEL {
            return;
        }

        // See <https://www.chessprogramming.org/Negamax>.
        let value = -value;

        let reward = 0.; // No intermediate rewards for now.
        let target = reward + self.discount_factor * value;

        self.tree[node].visits += 1;
        self.tree[node].value += self.learning_rate * (target - self.tree[node].value);

        self.backward_sarsa(self.tree[node].parent, self.tree[node].value);
    }
}

fn epsilon_greedy_policy(
    nodes: &[SarsaNode],
    exploration_rate: f64,
    rng: &mut impl Rng,
) -> usize {
    assert!(!nodes.is_empty(), "`nodes` must be non-empty.");

    if nodes.len() == 1 {
        return 0;
    }

    if exploration_rate >= 1. || rng.random_bool(exploration_rate) {
        return rng.random_range(0..nodes.len());
    }

    let mut best_index = 0;
    let mut best_value = f64::NEG_INFINITY;

    for (index, node) in nodes.iter().enumerate() {
        if node.visits < 1 {
            return index;
        }

        if node.value > best_value {
            best_index = index;
            best_value = node.value;
        }
    }

    best_index
}
