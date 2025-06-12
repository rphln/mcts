use std::iter::successors;

use rand::Rng;

use crate::game::{Command, Game};

// TODO: <https://gibberblot.github.io/rl-notes/single-agent/reward-shaping.html>

/// An implementation of the Monte Carlo Tree Search algorithm with the
/// following modifications:
///
/// **Soft rewards.** Nodes use continuous reward values instead of binary win
/// or loss outcomes.
///
/// **Epsilon-greedy selection.** Chooses a random action with probability ε,
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
#[derive(Clone, Debug)]
pub struct Mcts {
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub tree: Vec<MctsNode>,
    /// See <https://stackoverflow.com/a/35666246>.
    pub visits_to_expand: usize,
    /// Discount factor applied to rewards during back-propagation.
    pub discount_factor: f64,
}

/// A node that stores statistics for one ply in the game.
#[derive(Clone, Debug)]
pub struct MctsNode {
    /// The move used to reach this node.
    pub mov: Command,
    /// Number of visits to this node.
    pub visits: usize,
    /// Expected reward from this node.
    pub utility: f64,
    /// Index of the parent node; used for back-propagation of rewards.
    pub parent: usize,
    /// Index of the first child node.
    pub head: usize,
    /// Index of the last child node.
    pub last: usize,
}

impl MctsNode {
    /// Creates a new node with default statistics.
    #[must_use]
    fn new(parent: usize, mov: Command) -> Self {
        Self { mov, visits: 0, utility: 0., parent, head: 0, last: 0 }
    }
}

impl Mcts {
    /// Creates a new Monte Carlo Tree Search instance.
    ///
    /// The `root_move` is an arbitrary move that represents the root of the
    /// tree.
    #[must_use]
    pub fn new(root_move: Command) -> Self {
        Self {
            tree: vec![MctsNode::new(0, root_move)],
            visits_to_expand: 1,
            discount_factor: 0.95,
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
            self.tree[parent.head..parent.last].iter().max_by_key(|node| node.visits)
        })
    }

    /// Returns an iterator over the sequence of ancestors of a node.
    pub fn ancestors(&self, first: usize) -> impl Iterator<Item = &MctsNode> {
        let indices = successors(Some(first), |&node| {
            let parent = self.tree[node].parent;
            if parent == node { None } else { Some(parent) }
        });

        indices.map(|node| &self.tree[node])
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
        // See [1] for a reference on our implementation and the possibility of
        // using graphs instead of trees.
        //
        // These are the selection policies we can use:
        //
        // 1. Thompson sampling from the beta distribution.
        // 2. Thompson sampling, approximating the beta distribution with a normal
        //    distribution. [2]
        // 3. UCT.
        // 4. Bayesian UCB.
        //
        // We don't need to use virtual losses with Thompson sampling [3], which
        // is good if we decide to use multithreading for the selection, as it
        // removes the need for most write locks.
        //
        // In the single-threaded version, it's better to use Bayesian UCB with
        // virtual losses instead, because sampling from the beta distribution
        // is slow; sampling from the normal distribution is faster, but it's
        // still slower than UCB.
        //
        // On the other hand, Thompson sampling with the beta prior is more
        // accurate than the other options; if our estimator (i.e., the neural
        // network) is slow but accurate, we should prefer it, as the cost of
        // the sampling becomes negligible.
        //
        // There's also the option of combining them *with* a binary-search
        // policy [4] to speed up the search at the cost of accuracy.
        //
        // [1]: https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md
        // [2]: https://modelassist.epixanalytics.com/space/EA/26575264
        // [3]: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf#page=45
        // [4]: https://gist.github.com/rphln/fbbab3e0a432b95ec93d1e29e16acb03

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

            // Remove the non-decision nodes from the tree.
            //
            // See <https://www.chessprogramming.org/One_Reply_Extensions>.
            if last - head == 1 {
                self.tree[head].visits = self.tree[node].visits;
            }
        }

        let next = head + epsilon_greedy_policy(&self.tree[head..last], rng);
        game.play(self.tree[next].mov);

        self.select_and_expand_node(next, game, rng)
    }

    /// Back-propagates the reward from a leaf node up to the root.
    ///
    /// Rewards must be provided from the perspective of the side to move.
    pub fn backward(&mut self, node: usize, reward: f64) {
        // See <https://www.chessprogramming.org/Negamax>.
        let reward = -reward;

        self.tree[node].visits += 1;
        self.tree[node].utility +=
            (reward - self.tree[node].utility) / (self.tree[node].visits as f64);

        if self.tree[node].parent == node {
            return;
        }

        self.backward(self.tree[node].parent, self.discount_factor * reward);
    }

    /// Returns a copy of the subtree rooted at the given node.
    ///
    /// Behavior is undefined if the given node is not in the tree.
    #[must_use]
    pub fn clone_subtree(&self, root: &MctsNode) -> Self {
        let mut subtree = vec![MctsNode { parent: 0, ..root.clone() }];

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

fn epsilon_greedy_policy(tree: &[MctsNode], rng: &mut impl Rng) -> usize {
    if tree.len() == 1 {
        return 0;
    }

    if rng.random_bool(0.1) {
        return rng.random_range(0..tree.len());
    }

    let mut best_index = 0;
    let mut best_value = f64::NEG_INFINITY;

    for (index, node) in tree.iter().enumerate() {
        if node.visits < 1 {
            return index;
        }

        let value = node.utility;

        if value > best_value || (value == best_value && rng.random()) {
            best_index = index;
            best_value = value;
        }
    }

    best_index
}
