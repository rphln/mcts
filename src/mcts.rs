use std::iter::successors;

use rand::{Rng, seq::SliceRandom};

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
#[derive(Clone, Debug)]
pub struct Mcts {
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub tree: Vec<MctsNode>,
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
            visits_to_expand: 1,
            exploration_rate: 0.1,
            learning_rate: 0.05,
            discount_factor: 0.999,
        }
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

    /// Returns the best child node found so far.
    #[must_use]
    pub fn best_child(&self) -> Option<&MctsNode> {
        self.principal_variation().nth(1)
    }

    /// Returns an iterator over the sequence of best moves found so far.
    pub fn principal_variation(&self) -> impl Iterator<Item = &MctsNode> {
        successors(self.tree.first(), |parent| {
            self.tree[parent.head..parent.last]
                .iter()
                .filter(|node| node.visits > 0)
                .max_by(|left, right| {
                    f64::partial_cmp(&left.value, &right.value).unwrap()
                })
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

        self.backward_q_learning(self.tree[node].parent);
    }

    pub fn backward_q_learning(&mut self, node: usize) {
        if node == Self::SENTINEL {
            return;
        }

        let value = {
            let head = self.tree[node].head;
            let last = self.tree[node].last;

            let value = self.tree[head..last]
                .iter()
                .filter(|node| node.visits > 0)
                .map(|node| node.value)
                .reduce(f64::max)
                .expect("`node` should not be a leaf");
            -value
        };

        let reward = 0.; // No intermediate rewards for now.
        let target = reward + self.discount_factor * value;

        self.tree[node].visits += 1;
        self.tree[node].value += self.learning_rate * (target - self.tree[node].value);

        self.backward_q_learning(self.tree[node].parent);
    }

    /// Returns a copy of the subtree rooted at the given node.
    ///
    /// Behavior is undefined if the given node is not in the tree.
    #[must_use]
    pub fn clone_subtree(&self, root: &MctsNode) -> Self {
        let mut subtree = vec![MctsNode { parent: Self::SENTINEL, ..root.clone() }];

        for parent in 0.. {
            let Some(node) = subtree.get(parent) else {
                break;
            };

            let head = subtree.len();

            // At first, we copy the children as-is, and so `head` and `last`
            // will point to the old tree until we visit them in this loop. This
            // trick allows us to have an implicit queue of nodes to visit
            // entirely for free.
            for child in &self.tree[node.head..node.last] {
                subtree.push(MctsNode { parent, ..child.clone() });
            }

            let last = subtree.len();

            // If `node` is a leaf, then `head` and `last` are equal. This is
            // exactly the same criterion that the expansion step uses to check
            // for leaves: it doesn't matter *where* they point to, as long as
            // they're equal.
            subtree[parent].head = head;
            subtree[parent].last = last;
        }

        Self {
            tree: subtree,
            visits_to_expand: self.visits_to_expand,
            exploration_rate: self.exploration_rate,
            learning_rate: self.learning_rate,
            discount_factor: self.discount_factor,
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

    /// Runs garbage collection on the tree.
    ///
    /// See <https://repository.falmouth.ac.uk/2782/1/MemoryLimiting.pdf>.
    pub fn gc(&mut self, threshold: usize) {
        if self.tree.len() == 1 {
            return;
        }

        let mut subtree = vec![self.root().clone()];

        for parent in 0.. {
            let Some(node) = subtree.get(parent) else {
                break;
            };

            let head = subtree.len();

            // At first, we copy the children as-is, and so `head` and `last`
            // will point to the old tree until we visit them in this loop. This
            // trick allows us to have an implicit queue of nodes to visit
            // entirely for free.
            for next in node.head..node.last {
                let mut child = MctsNode { parent, ..self.tree[next].clone() };
                if child.visits < threshold {
                    child.head = 0;
                    child.last = 0;
                }

                subtree.push(child);
            }

            let last = subtree.len();

            // If `node` is a leaf, then `head` and `last` are equal. This is
            // exactly the same criterion that the expansion step uses to check
            // for leaves: it doesn't matter *where* they point to, as long as
            // they're equal.
            subtree[parent].head = head;
            subtree[parent].last = last;
        }

        self.tree = subtree;
    }
}

fn epsilon_greedy_policy(
    nodes: &[MctsNode],
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
