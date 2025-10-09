use std::{assert_matches::assert_matches, iter::successors};

use rand::prelude::*;
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
    /// See <https://www.cs.cornell.edu/~asampson/blog/flattening.html>.
    pub tree: Vec<Node>,
    /// Global history statistics for moves.
    pub history: FxHashMap<Move, History>,
    /// See <https://stackoverflow.com/a/35666246>.
    pub visits_to_expand: u32,
    /// Exploration rate (ε) for the ε-greedy policy.
    pub exploration_rate: f64,
}

/// Statistics for a single ply in the game tree.
#[derive(Clone, Debug)]
pub struct Node {
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
    fn new(parent: usize, mov: Move) -> Self {
        Self { mov, visits: 0, value: 0., parent, head: SENTINEL, last: SENTINEL }
    }

    /// Returns whether this node is the root node.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent == SENTINEL
    }

    /// Returns whether this node is a leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.head == SENTINEL
    }
}

impl Mcts {
    /// Creates a new Monte Carlo Tree Search instance.
    ///
    /// The `root_move` is an arbitrary move that represents the root of the
    /// tree.
    #[must_use]
    pub fn new(root_move: Move) -> Self {
        Self {
            tree: vec![Node::new(SENTINEL, root_move)],
            history: FxHashMap::default(),
            visits_to_expand: 1,
            exploration_rate: 0.1,
        }
    }

    /// Returns an iterator over the sequence of best moves found so far.
    pub fn principal_variation(&self) -> impl Iterator<Item = &Node> {
        successors(self.tree.first(), |parent| {
            self.tree[parent.head..parent.last].iter().max_by_key(|node| node.visits)
        })
    }

    /// Returns an iterator over the ancestors of `node`, starting at `node`.
    pub fn ancestors(&self, node: usize) -> impl Iterator<Item = &Node> {
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

        become self.expand_and_select_node(next, game, rng);
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
            self.tree.push(Node::new(node, mov));
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

        let mut best_index = SENTINEL;
        let mut best_value = f64::NEG_INFINITY;

        for index in head..last {
            let entry = &self.tree[index];
            let Some(history) = self.history.get(&entry.mov) else {
                return index;
            };

            let m = f64::from(history.visits);
            let n = f64::from(entry.visits);

            // Found empirically. See Section 8.4.2 in [1] for other schedules.
            //
            // [1]: <https://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/1029/paper_thesis.pdf>
            let alpha = f64::sqrt(n / (n + m));
            assert_matches!(alpha, 0.0..=1.0, "`alpha` should be in [0, 1]");

            let value = alpha * entry.value + (1. - alpha) * history.value;
            if value > best_value {
                best_index = index;
                best_value = value;
            }
        }

        assert_ne!(best_index, SENTINEL, "`best_index` should be set");

        best_index
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

        let entry = &mut self.tree[node];
        let history = self.history.entry(entry.mov).or_default();

        entry.visits += 1;
        entry.value += (value - entry.value) / f64::from(entry.visits);

        history.visits += 1;
        history.value += (value - history.value) / f64::from(history.visits);

        let parent = entry.parent;
        become self.backward(parent, value);
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
        max_time: Option<u64>,
        max_iters: Option<u32>,
        max_nodes: Option<usize>,
    ) -> usize {
        let mut iters = 0;
        let mut nodes = 0;

        let now = thread_time_ms();
        let root_depth = game.depth;

        while max_time.is_none_or(|t| thread_time_ms() - now < t)
            && max_iters.is_none_or(|n| iters < n)
            && max_nodes.is_none_or(|n| nodes < n)
        {
            let mut game = game.clone();
            let next = self.expand_and_select_node(node, &mut game, rng);

            // See <https://www.sciencedirect.com/science/article/pii/S0304397516302717>.
            for _ in 0..2 {
                if game.is_over() {
                    break;
                }

                let &mov =
                    game.moves().choose(rng).expect("`moves` should be non-empty");
                game.play(mov);
            }

            let reward = game.evaluate();
            self.backward(next, reward);

            iters += 1;
            nodes += game.depth - root_depth;
        }

        let head = self.tree[node].head;
        let last = self.tree[node].last;

        (head..last)
            .max_by_key(|&next| self.tree[next].visits)
            .expect("`head..last` should be non-empty")
    }

    /// Removes parent nodes that satisfy `predicate`.
    pub fn gc(&mut self, pred: impl Fn(&Node) -> bool) {
        let len = self.tree.len();

        let mut map = vec![SENTINEL; len];
        let mut dst = 0;

        // Invariant: always keep the root. Removing it would break the tree.
        map[0] = 0;
        dst += 1;

        for src in 1..len {
            let node = &self.tree[src];
            let parent = &self.tree[node.parent];

            if map[node.parent] == SENTINEL || pred(parent) {
                continue;
            }

            map[src] = dst;
            dst += 1;
        }

        for src in 0..len {
            let dst = map[src];
            if dst == SENTINEL {
                continue;
            }

            assert!(dst <= src);
            self.tree.swap(src, dst);

            let node = &mut self.tree[dst];

            if !node.is_root() {
                node.parent = map[node.parent];
            }

            if !node.is_leaf() {
                node.last = map[node.head].saturating_add(node.last - node.head);
                node.head = map[node.head];
            }
        }

        self.tree.truncate(dst);
    }
}

/// Returns the current CPU time in nanoseconds.
///
/// # Panics
///
/// Panics if the system call to get the CPU time fails.
fn thread_time_ms() -> u64 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };

    unsafe {
        assert_ne!(
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut time),
            -1,
            "Failed to get CPU time"
        );
    };

    let tv_sec = u64::try_from(time.tv_sec).unwrap();
    let tv_nsec = u64::try_from(time.tv_nsec).unwrap();

    tv_sec * 1_000 + tv_nsec / 1_000_000
}
